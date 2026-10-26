"""Memory questions b/c using saved notebook functions, one process per case."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def source(c):
    s = c.get("source", [])
    return s if isinstance(s, str) else "".join(s)


def worker(args):
    import torch
    cells = json.loads(Path("作业四.ipynb").read_text())["cells"]
    sources = [source(c) for c in cells if c["cell_type"] == "code"]
    ns = {"__name__": "memory_bc_worker"}
    exec(next(s for s in sources if "import inspect" in s), ns)
    exec(next(s for s in sources if "CONFIGS = {" in s and "SIZE =" in s), ns)
    ns.update(SIZE="large", BATCH_SIZE=1, CONTEXT_LENGTH=args.context,
              MODE=args.mode, USE_BF16=args.precision == "bf16", NUM_WARMUPS=1)
    ns["CONFIGS"]["large"] = dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20)
    ns["snapshot_dir"] = Path("memory_bc_results_20261026")
    ns["snapshot_dir"].mkdir(exist_ok=True)
    ns["description"] = f"large_batch1_ctx{args.context}_{args.mode}_{args.precision}"
    metadata = dict(size="large", batch=1, context=args.context, mode=args.mode,
                    precision=args.precision, warmups=1, measured_steps=1,
                    parameter_dtype="float32", gpu=torch.cuda.get_device_name(0),
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    allocator=os.environ.get("PYTORCH_ALLOC_CONF"))
    print("SETTINGS " + json.dumps(metadata), flush=True)
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported")
    stage = "model_loading"
    try:
        exec(next(s for s in sources if "model = BasicsTransformerLM(" in s), ns)
        observed = {}
        handle = ns["model"].register_forward_hook(lambda module, inputs, output: observed.update(logits_dtype=str(output.dtype)))
        exec(next(s for s in sources if "def run_model(" in s), ns)
        exec(next(s for s in sources if "def memory_profile(" in s), ns)
        stage = "warmup_or_measured_step"
        result = ns["memory_profile"](ns["description"], ns["run_model"], num_warmups=1)
        if not isinstance(result, dict) or "peak_allocated_MiB" not in result:
            raise RuntimeError("Notebook memory_profile did not return peak results")
        handle.remove()
        result = dict(result, **metadata, **observed, status="OK")
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        result = dict(metadata, status="OOM", failed_stage=stage, error=str(error))
    output = ns["snapshot_dir"] / (ns["description"] + ".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("RESULT " + json.dumps(result), flush=True)
    # This process exits after each case, releasing tensors, gradients and AdamW.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--context", type=int, choices=[128, 256, 512], default=128)
    p.add_argument("--mode", choices=["inference", "train"], default="inference")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    env = dict(os.environ, PYTORCH_ALLOC_CONF="expandable_segments:True")
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    for precision in ["fp32", "bf16"]:
        for context in [128, 256, 512]:
            for mode in ["inference", "train"]:
                free = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
                if free < 20000:
                    print("BLOCKED external GPU memory use; free MiB", free, flush=True)
                    return
                print(f"RUN {precision} {context} {mode}", flush=True)
                result = subprocess.run([
                    sys.executable, __file__, "--worker", "--context", str(context),
                    "--mode", mode, "--precision", precision,
                ], env=env, check=False)
                if result.returncode:
                    print("FAILED", precision, context, mode, result.returncode, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
