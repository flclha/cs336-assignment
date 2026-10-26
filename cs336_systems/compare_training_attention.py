"""CS336: inference vs AdamW training; attention time vs operation counts.

python compare_training_attention.py --tool torch --mode inference
python compare_training_attention.py --tool torch --mode train
nsys profile -t cuda,nvtx --sample=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o report python compare_training_attention.py \
  --tool nsys --mode train
"""
import argparse
from contextlib import contextmanager, nullcontext
import gc
import json
import math
from pathlib import Path
import statistics
from timeit import default_timer

import torch
import torch.nn.functional as F
from einops import einsum
import cs336_basics.model as model_module
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizer import AdamW

CONFIGS = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (1600, 6400, 48, 25),
    "2.7B": (2560, 10240, 32, 32),
}
TOOL = "python"


@contextmanager
def scope(name):
    # Same semantic regions in PyTorch Profiler and Nsight; no synchronization
    # within attention, so kernel launch scheduling is not changed per operation.
    record = torch.profiler.record_function(name) if TOOL == "torch" else nullcontext()
    with torch.cuda.nvtx.range(name), record:
        yield


def annotated_attention(Q, K, V, mask=None):
    with scope("attention_total"):
        with scope("attention_qk"):
            scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
        with scope("attention_scale"):
            scores = scores / math.sqrt(K.shape[-1])
        if mask is not None:
            with scope("attention_mask"):
                scores = torch.where(mask, scores, float("-inf"))
        with scope("attention_softmax"):
            weights = softmax(scores, dim=-1)
        with scope("attention_pv"):
            output = einsum(weights, V, "... query key, ... key d_v -> ... query d_v")
    return output


def device_us(event, self_time=False):
    modern = "self_device_time_total" if self_time else "device_time_total"
    legacy = "self_cuda_time_total" if self_time else "cuda_time_total"
    if hasattr(event, modern):
        return getattr(event, modern)
    return getattr(event, legacy, 0.0)


def main():
    global TOOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=CONFIGS, default="small")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--mode", choices=["inference", "forward", "train"], default="inference")
    parser.add_argument("--tool", choices=["python", "torch", "nsys"], default="torch")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--trace", type=Path, help="Optional Chrome trace JSON for --tool torch")
    args = parser.parse_args()
    if args.context < 1 or args.warmups < 1 or args.steps < 2:
        parser.error("context >= 1, warmups >= 1, steps >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    TOOL = args.tool
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Verify instrumentation preserves both values and gradients, including mask.
    original = model_module.scaled_dot_product_attention
    q = torch.randn(1, 2, 16, 16, device="cuda", requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    mask = torch.ones(16, 16, device="cuda", dtype=torch.bool).tril()
    reference = original(q, k, v, mask)
    actual = annotated_attention(q, k, v, mask)
    torch.testing.assert_close(actual, reference)
    for a, b in zip(torch.autograd.grad(actual.sum(), (q, k, v)),
                    torch.autograd.grad(reference.sum(), (q, k, v))):
        torch.testing.assert_close(a, b)
    del q, k, v, mask, reference, actual, a, b
    model_module.scaled_dot_product_attention = annotated_attention

    model = optimizer = x = targets = None
    try:
        d_model, d_ff, layers, heads = CONFIGS[args.size]
        model = model_module.BasicsTransformerLM(
            vocab_size=10000, context_length=args.context, d_model=d_model,
            d_ff=d_ff, num_layers=layers, num_heads=heads, rope_theta=10000.0,
        )
        model = model.to(device="cuda", dtype=torch.float32)
        model.train(args.mode != "inference")
        x = torch.randint(10000, (4, args.context), device="cuda")
        targets = torch.randint(10000, (4, args.context), device="cuda")
        if args.mode == "train":
            optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

        def step():
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times = {}
            with scope("step"):
                with scope("forward"):
                    start = default_timer()
                    with torch.set_grad_enabled(args.mode != "inference"):
                        logits = model(x)
                    torch.cuda.synchronize()
                    times["forward_ms"] = (default_timer() - start) * 1000
                if args.mode == "train":
                    with scope("loss"):
                        loss = F.cross_entropy(logits.reshape(-1, 10000), targets.reshape(-1))
                        torch.cuda.synchronize()
                    with scope("backward"):
                        loss.backward()
                        torch.cuda.synchronize()
                    with scope("optimizer"):
                        optimizer.step()
                        torch.cuda.synchronize()
            return times

        for _ in range(args.warmups):
            step()
        torch.cuda.synchronize()
        profiler = (torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) if args.tool == "torch" else nullcontext())
        if args.tool == "nsys":
            torch.cuda.cudart().cudaProfilerStart()
        try:
            with profiler as prof:
                samples = [step() for _ in range(args.steps)]
                torch.cuda.synchronize()
        finally:
            if args.tool == "nsys":
                torch.cuda.cudart().cudaProfilerStop()

        print("\nSETTINGS", json.dumps(vars(args), default=str))
        print("GPU", torch.cuda.get_device_name(0), "PyTorch", torch.__version__)
        print("Validated attention output AND gradients against original implementation.")
        print("FP32, TF32 off, batch=4, vocab=10000; stage-end CUDA synchronization.")
        forward_times = [s["forward_ms"] for s in samples]
        print(f"Python forward wall time: {statistics.mean(forward_times):.4f} +/- "
              f"{statistics.stdev(forward_times):.4f} ms/step (under selected tool)")

        if args.tool == "torch":
            averages = prof.key_averages()
            cpu = [e for e in averages if e.device_type == torch.autograd.DeviceType.CPU]
            # Non-overlapping self-device attribution: never sum inclusive parent
            # totals such as aten::matmul and its nested aten::bmm together.
            total = sum(device_us(e, True) for e in cpu)
            matmul_names = {"aten::mm", "aten::bmm", "aten::addmm", "aten::baddbmm"}
            mm = sum(device_us(e, True) for e in cpu if e.key in matmul_names)
            by_name = {e.key: e for e in cpu}
            regions = {name: device_us(by_name[name]) / args.steps / 1000
                       for name in ["attention_total", "attention_qk", "attention_scale",
                                    "attention_mask", "attention_softmax", "attention_pv"]
                       if name in by_name}
            print("\n(a) Operator-attributed GPU time, per step (not CPU wall time):")
            print(f"All attributed GPU time: {total / args.steps / 1000:.4f} ms")
            print(f"GEMM leaf operators: {mm / args.steps / 1000:.4f} ms; "
                  f"share={100 * mm / total if total else 0:.2f}%")
            print("GEMM classification:", sorted(matmul_names))
            print("These are PyTorch operator-attributed times; use Nsight kernel summary for kernel-level confirmation.")
            print("\n(b) Forward attention regions, summed over ALL layers, ms/step:")
            for name, value in regions.items():
                print(f"{name}: {value:.4f}")
            qkpv = regions.get("attention_qk", 0) + regions.get("attention_pv", 0)
            soft = regions.get("attention_softmax", 0)
            print(f"QK+PV GPU-time / softmax GPU-time: {qkpv / soft if soft else float('nan'):.3f}")
            print("\nTop CPU operators by self-attributed GPU time (ms/step):")
            for event in sorted(cpu, key=lambda e: device_us(e, True), reverse=True)[:15]:
                print(f"{event.key}: {device_us(event, True) / args.steps / 1000:.4f}")
            if args.trace:
                args.trace.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(args.trace))

        B, T, H, D = 4, args.context, heads, d_model // heads
        elements = layers * B * H * T * T
        mm_flops = 4 * elements * D
        # Convention: max comparisons and exp each count as one operation.
        # Per row: (T-1) max comparisons, T subtracts, T exp,
        # (T-1) adds, T divides = 5*T-2 operations.
        soft_ops = layers * B * H * T * (5 * T - 2)
        print("\nForward attention operation counts, all layers per step:")
        print(f"QK+PV FLOPs (multiply+add=2): {mm_flops:,}")
        print(f"Stable softmax approximate operations: {soft_ops:,}")
        print(f"Matmul FLOPs / softmax approximate ops: {mm_flops / soft_ops:.3f}")
        print("Softmax convention counts exp/comparison as 1 op; not equivalent-cost arithmetic FLOPs.")
        print("Exclude mask/scaling from QK+PV counts; full matrices are computed despite causal mask.")
        print("Do not add attention_total to its nested region times (double counting).")
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
        model = optimizer = x = targets = None
        model_module.scaled_dot_product_attention = original
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
