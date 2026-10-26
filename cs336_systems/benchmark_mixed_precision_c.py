"""Assignment mixed precision (c): FP32 vs BF16, independent processes, no CSV."""
import argparse
from contextlib import nullcontext
import json
import statistics
import subprocess
import sys
from timeit import default_timer

CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def worker(size, precision):
    import platform
    import torch
    import torch.nn.functional as F
    from cs336_basics.model import BasicsTransformerLM

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = dict(size=size, precision=precision, warmups=5, trials=10,
                    context_length=128, batch_size=4, vocab_size=10000,
                    seed=42, parameter_dtype="float32", tf32=False,
                    gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                    python=platform.python_version(), cuda=torch.version.cuda)
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this GPU")
    stage = "model_initialization"
    try:
        print("SETTINGS " + json.dumps(settings), flush=True)
        model = BasicsTransformerLM(
            vocab_size=10000, context_length=128, rope_theta=10000.0,
            **CONFIGS[size],
        ).to(device="cuda", dtype=torch.float32)
        model.train()
        x = torch.randint(10000, (4, 128), device="cuda")
        targets = torch.randint(10000, (4, 128), device="cuda")

        def step():
            nonlocal stage
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                       if precision == "bf16" else nullcontext())
            with context:
                stage = "forward"
                start = default_timer()
                logits = model(x)
                torch.cuda.synchronize()
                forward = (default_timer() - start) * 1000
                stage = "loss"
                loss = F.cross_entropy(logits.reshape(-1, 10000), targets.reshape(-1))
                torch.cuda.synchronize()
            # Backward is outside autocast; no GradScaler is needed for BF16.
            stage = "backward"
            start = default_timer()
            loss.backward()
            torch.cuda.synchronize()
            backward = (default_timer() - start) * 1000
            return dict(forward_ms=forward, backward_ms=backward,
                        loss=float(loss.detach()), logits_dtype=str(logits.dtype))

        for _ in range(5):
            step()
        samples = [step() for _ in range(10)]
        if not all(__import__("math").isfinite(s["loss"]) for s in samples):
            raise RuntimeError("Nonfinite loss encountered")
        result = dict(settings, status="OK", samples=samples)
        for metric in ["forward_ms", "backward_ms"]:
            values = [s[metric] for s in samples]
            result[metric + "_mean"] = statistics.mean(values)
            result[metric + "_std"] = statistics.stdev(values)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        result = dict(settings, status="OOM", failed_stage=stage, error=str(error))
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=CONFIGS)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()
    if args.size:
        worker(args.size, args.precision)
    else:
        for size in CONFIGS:
            for precision in ["fp32", "bf16"]:
                print(f"RUN {size} {precision}", flush=True)
                process = subprocess.run(
                    [sys.executable, __file__, "--size", size, "--precision", precision],
                    check=False)
                if process.returncode:
                    print("FAILED " + json.dumps(dict(size=size, precision=precision,
                                                      returncode=process.returncode)), flush=True)
        print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
