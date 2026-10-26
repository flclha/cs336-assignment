"""Run each CS336 1.1.3 configuration in a fresh process; print JSON, no CSV."""
import argparse
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


def worker(size, warmups):
    import platform
    import torch
    import torch.nn.functional as F
    from cs336_basics.model import BasicsTransformerLM

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = dict(size=size, warmups=warmups, trials=10, context_length=128,
                    batch_size=4, vocab_size=10000, seed=42, dtype="float32",
                    tf32=False, gpu=torch.cuda.get_device_name(0),
                    python=platform.python_version(), torch=str(torch.__version__),
                    cuda=torch.version.cuda, config=CONFIGS[size])
    print("SETTINGS " + json.dumps(settings), flush=True)
    try:
        model = BasicsTransformerLM(vocab_size=10000, context_length=128,
                                   rope_theta=10000.0, **CONFIGS[size]).to(
                                       device="cuda", dtype=torch.float32)
        model.train()
        x = torch.randint(10000, (4, 128), device="cuda")
        targets = torch.randint(10000, (4, 128), device="cuda")

        def step():
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = default_timer()
            logits = model(x)
            torch.cuda.synchronize()
            forward = (default_timer() - start) * 1000
            loss = F.cross_entropy(logits.reshape(-1, 10000), targets.reshape(-1))
            torch.cuda.synchronize()
            start = default_timer()
            loss.backward()
            torch.cuda.synchronize()
            backward = (default_timer() - start) * 1000
            return forward, backward

        for _ in range(warmups):
            step()
        samples = [step() for _ in range(10)]
        f, b = zip(*samples)
        result = dict(settings, status="OK", forward_mean_ms=statistics.mean(f),
                      forward_std_ms=statistics.stdev(f),
                      backward_mean_ms=statistics.mean(b),
                      backward_std_ms=statistics.stdev(b), samples=samples)
    except torch.cuda.OutOfMemoryError as error:
        result = dict(settings, status="OOM", error=str(error))
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=CONFIGS)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    if args.size:
        worker(args.size, args.warmups)
    else:
        for size in CONFIGS:
            # The warmup=5 results also serve as question (b).
            for warmups in [5, 0, 1, 2]:
                print(f"RUN size={size} warmups={warmups}", flush=True)
                completed = subprocess.run(
                    [sys.executable, __file__, "--size", size,
                     "--warmups", str(warmups)], check=False)
                if completed.returncode:
                    print("RUN_FAILED " + json.dumps(dict(
                        size=size, warmups=warmups,
                        returncode=completed.returncode)), flush=True)
        print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
