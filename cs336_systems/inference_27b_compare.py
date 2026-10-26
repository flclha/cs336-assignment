"""2.7B inference, batch=1: memory peak and FP32/BF16 latency; no optimizer."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from timeit import default_timer


def worker(context_length, precision):
    import torch
    from cs336_basics.model import BasicsTransformerLM

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported")
    metadata = dict(size="2.7B", batch=1, context=context_length,
                    mode="inference", precision=precision, parameter_dtype="float32",
                    warmups=1, memory_steps=1, latency_steps=10, seed=42,
                    vocab_size=10000, tf32=False,
                    gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                    cuda=torch.version.cuda,
                    allocator=os.environ.get("PYTORCH_ALLOC_CONF"))
    print("SETTINGS " + json.dumps(metadata), flush=True)
    stage = "model_loading"
    try:
        model = BasicsTransformerLM(
            vocab_size=10000, context_length=context_length,
            d_model=2560, d_ff=10240, num_layers=32, num_heads=32,
            rope_theta=10000.0,
        ).to(device="cuda", dtype=torch.float32)
        model.eval()
        x = torch.randint(10000, (1, context_length), device="cuda")
        parameter_count = sum(p.numel() for p in model.parameters())

        def step():
            context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                       if precision == "bf16" else nullcontext())
            torch.cuda.synchronize()
            start = default_timer()
            with torch.no_grad(), context:
                logits = model(x)
                torch.cuda.synchronize()
                elapsed = (default_timer() - start) * 1000
                dtype = str(logits.dtype)
            return elapsed, dtype

        stage = "warmup"
        step()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        stage = "measured_forward"
        _, dtype = step()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        stage = "latency_measurements"
        samples = [step()[0] for _ in range(10)]
        result = dict(metadata, status="OK", parameter_count=parameter_count,
                      logits_dtype=dtype, baseline_MiB=baseline / 1024**2,
                      peak_allocated_MiB=peak / 1024**2,
                      peak_reserved_MiB=reserved / 1024**2,
                      forward_mean_ms=statistics.mean(samples),
                      forward_std_ms=statistics.stdev(samples), latency_samples_ms=samples)
        assert dtype == ("torch.bfloat16" if precision == "bf16" else "torch.float32")
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        result = dict(metadata, status="OOM", failed_stage=stage, error=str(error))
    out = Path("inference_27b_results_20261026")
    out.mkdir(exist_ok=True)
    (out / f"ctx{context_length}_{precision}.json").write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--context", type=int, choices=[128, 256, 512], default=128)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    args = p.parse_args()
    if args.worker:
        worker(args.context, args.precision)
        return
    env = dict(os.environ, PYTORCH_ALLOC_CONF="expandable_segments:True")
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    for context in [128, 256, 512]:
        for precision in ["fp32", "bf16"]:
            free = float(subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
            ], text=True).splitlines()[0])
            if free < 20000:
                print("BLOCKED external GPU memory use; free MiB", free, flush=True)
                return
            print("RUN", context, precision, flush=True)
            completed = subprocess.run([
                sys.executable, __file__, "--worker", "--context", str(context),
                "--precision", precision,
            ], env=env, check=False)
            if completed.returncode:
                print("FAILED", context, precision, completed.returncode, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
