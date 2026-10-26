"""Compare the notebook's three attention implementations: forward only.

Primary table: B=1, d=64, FP32/IEEE, no mask, no_grad, no optimizer.
N <= 4096: 3 warmups + 10 trials; larger N: 1 warmup + 3 trials.
Each implementation/shape runs in a fresh process. Original notebook untouched.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from timeit import default_timer

OUT = Path("attention_forward_compare_20261026")
LENGTHS = [256, 1024, 2048, 4096, 8192, 16384]


def extract_implementations():
    notebook_path = Path("作业1.ipynb")
    notebook = json.loads(notebook_path.read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    pytorch_source = next(s for s in sources if "class FlashAttentionAutograd(" in s)
    triton_source = next(s for s in sources if "class FlashAttentionTriton(" in s)
    # Freeze exact notebook implementations; only align FP32 matmul precision.
    assert "tl.dot(Q_i, K_j.T)" in triton_source
    assert "tl.dot(P_cast, V_j)" in triton_source
    triton_source = triton_source.replace("tl.dot(Q_i, K_j.T)",
        'tl.dot(Q_i, K_j.T, input_precision="ieee")')
    triton_source = triton_source.replace("tl.dot(P_cast, V_j)",
        'tl.dot(P_cast, V_j, input_precision="ieee")')
    OUT.mkdir(exist_ok=True)
    (OUT / "impl_pytorch.py").write_text(pytorch_source)
    (OUT / "impl_triton.py").write_text(triton_source)
    (OUT / "source_manifest.json").write_text(json.dumps(dict(
        notebook=str(notebook_path.resolve()),
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        adjustment="Triton forward dot precision explicitly ieee; original tile sizes preserved",
    ), indent=2))


def worker(args):
    import torch
    sys.path.insert(0, str(OUT.resolve()))
    from impl_pytorch import FlashAttentionAutograd
    from impl_triton import FlashAttentionTriton
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    N = args.length
    warmups, trials = (3, 10) if N <= 4096 else (1, 3)
    metadata = dict(implementation=args.impl, N=N, B=1, d=64, dtype="float32",
                    causal=False, requires_grad=False, tf32=False,
                    warmups=warmups, trials=trials, seed=42,
                    gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__))
    print("SETTINGS " + json.dumps(metadata), flush=True)
    stage = "inputs"
    try:
        Q, K, V = [torch.randn(1, N, 64, device="cuda") for _ in range(3)]
        def naive():
            scores = (Q @ K.transpose(-2, -1)) / 8.0
            probabilities = torch.softmax(scores, dim=-1)
            return probabilities @ V
        fn = dict(naive=naive,
                  pytorch_blocked=lambda: FlashAttentionAutograd.apply(Q, K, V),
                  triton=lambda: FlashAttentionTriton.apply(Q, K, V, False))[args.impl]

        with torch.no_grad():
            stage = "correctness"
            output = fn()
            # Full reference for small N; sample 64 query rows for large N.
            # All keys/values are used, so the selected rows remain exact attention.
            count = N if N <= 1024 else 64
            query_ids = torch.linspace(0, N - 1, count, device="cuda").long()
            reference = torch.softmax((Q[:, query_ids] @ K.transpose(-2, -1)) / 8.0, dim=-1) @ V
            selected = output[:, query_ids]
            max_abs_error = (selected - reference).abs().max().item()
            torch.testing.assert_close(selected, reference, rtol=1e-4, atol=1e-5)
            del output, selected, reference, query_ids
            gc.collect()
            stage = "warmup"
            for _ in range(warmups):
                output = fn()
                del output
            torch.cuda.synchronize()

            # Peak of one forward, including Q/K/V baseline; no live old output.
            stage = "memory"
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = fn()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            del output

            stage = "timing"
            times = []
            for _ in range(trials):
                torch.cuda.synchronize()
                start = default_timer()
                output = fn()
                torch.cuda.synchronize()
                times.append((default_timer() - start) * 1000)
                del output

        result = dict(metadata, status="OK", mean_ms=statistics.mean(times),
                      std_ms=statistics.stdev(times), samples_ms=times,
                      baseline_MiB=baseline / 1024**2,
                      peak_allocated_MiB=peak / 1024**2,
                      extra_peak_MiB=(peak - baseline) / 1024**2,
                      checked_query_rows=count, max_abs_error=max_abs_error)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        result = dict(metadata, status="OOM", failed_stage=stage, error=str(error))
    except AssertionError as error:
        result = dict(metadata, status="INCORRECT", failed_stage=stage, error=str(error))
    except Exception as error:
        result = dict(metadata, status="ERROR", failed_stage=stage,
                      error=type(error).__name__ + ": " + str(error))
    OUT.mkdir(exist_ok=True)
    (OUT / f"{args.impl}_N{N}.json").write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--impl", choices=["naive", "pytorch_blocked", "triton"], default="naive")
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    p.add_argument("--extended", action="store_true", help="Naive/Triton only, N=32768 and 65536")
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    extract_implementations()
    cases = ([(n, impl) for n in [32768, 65536] for impl in ["naive", "triton"]]
             if args.extended else
             [(n, impl) for n in args.lengths for impl in ["naive", "pytorch_blocked", "triton"]])
    for n, impl in cases:
        free = float(subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        ], text=True).splitlines()[0])
        if free < 20000:
            print("BLOCKED external GPU memory use; free MiB", free, flush=True)
            return
        print("RUN", n, impl, flush=True)
        try:
            completed = subprocess.run([
                sys.executable, __file__, "--worker", "--impl", impl, "--length", str(n),
            ], check=False, timeout=600)
            if completed.returncode:
                print("FAILED", n, impl, completed.returncode, flush=True)
        except subprocess.TimeoutExpired:
            print("TIMEOUT", n, impl, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
