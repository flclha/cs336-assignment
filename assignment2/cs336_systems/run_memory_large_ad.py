"""Execute selected cells of the user's memory notebook in a fresh process."""
import argparse
import json
from pathlib import Path


MEMORY_PROFILE_SOURCE = '''def memory_profile(description, run, num_warmups=1):
    # Warmup initializes the existing model and AdamW state, not a new model.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize()
    path = snapshot_dir / f"{description}.pickle"
    recording = False
    try:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        recording = True
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        stage_memory = run()
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_reserved_bytes = torch.cuda.max_memory_reserved()
        torch.cuda.memory._dump_snapshot(str(path))
        result = {
            "configuration": description,
            "baseline_MiB": baseline_bytes / 1024**2,
            "peak_allocated_MiB": peak_bytes / 1024**2,
            "peak_reserved_MiB": peak_reserved_bytes / 1024**2,
            "extra_above_baseline_MiB": (peak_bytes-baseline_bytes) / 1024**2,
            "snapshot": str(path.resolve()),
            "stage_end_memory": stage_memory,
        }
        print(result, flush=True)
        return result
    except torch.cuda.OutOfMemoryError:
        if recording:
            torch.cuda.memory._dump_snapshot(str(path.with_suffix(".oom.pickle")))
        raise
    finally:
        if recording:
            torch.cuda.memory._record_memory_history(enabled=None)
'''


def source(cell):
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inference", "train"], default="inference")
    parser.add_argument("--patch", action="store_true")
    args = parser.parse_args()
    path = Path("作业四.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if args.patch:
        from datetime import datetime
        backup = path.with_name(path.stem + ".before_large_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".ipynb")
        backup.write_bytes(path.read_bytes())
        for cell in cells:
            s = source(cell)
            if cell["cell_type"] == "code" and "CONFIGS = {" in s and "SIZE =" in s:
                import re
                s = re.sub(r'^SIZE\s*=.*$', 'SIZE = "large"  # 按用户要求替代2.7B；非原题规模', s, flags=re.M)
                s = re.sub(r'^CONTEXT_LENGTH\s*=.*$', 'CONTEXT_LENGTH = 512  # (a)要求的上下文长度', s, flags=re.M)
                s = re.sub(r'^MODE\s*=.*$', 'MODE = "inference"  # 第二次实验改为train并重建模型', s, flags=re.M)
                s = re.sub(r'^USE_BF16\s*=.*$', 'USE_BF16 = False  # (a) FP32基线', s, flags=re.M)
                if '"large"' not in s[s.index('CONFIGS = {'):]:
                    s = s.replace('CONFIGS = {', 'CONFIGS = {\n    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),', 1)
                cell["source"] = s.splitlines(True)
                cell["execution_count"] = None
                cell["outputs"] = []
            elif cell["cell_type"] == "code" and "def memory_profile(" in s:
                cell["source"] = MEMORY_PROFILE_SOURCE.splitlines(True)
                cell["execution_count"] = None
                cell["outputs"] = []
            elif cell["cell_type"] == "code" and "size_mib =" in s and "2560" in s:
                s = s.replace("2560", "CONFIGS[SIZE]['d_model']")
                cell["source"] = s.splitlines(True)
                cell["execution_count"] = None
                cell["outputs"] = []
        cells.insert(0, dict(cell_type="markdown", metadata={}, source=[
            "# 本次实验修改说明\n",
            "按用户要求用 **large** 替代PDF的2.7B：d_model=1280、d_ff=5120、36层、20头。\n",
            "本次(a)使用context=512、batch=4、FP32，分别测inference/train；原讲义中2.7B的要求与数值仅供对照，不是本次large结果。\n",
            "(d)本次单个残差流激活为[4,512,1280]，FP32为10 MiB。\n",
            "已补齐memory_profile的记录、峰值统计与finally停止逻辑；run_model沿用当前文件。\n",
        ], id="large-substitution-note"))
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print("PATCHED", path, "BACKUP", backup, flush=True)
        return

    namespace = {"__name__": "memory_notebook_runner"}
    imports = next(source(c) for c in cells if c["cell_type"] == "code" and "import inspect" in source(c))
    params = next(source(c) for c in cells if c["cell_type"] == "code" and "CONFIGS = {" in source(c) and "SIZE =" in source(c))
    build = next(source(c) for c in cells if c["cell_type"] == "code" and "model = BasicsTransformerLM(" in source(c))
    run = next(source(c) for c in cells if c["cell_type"] == "code" and "def run_model(" in source(c))
    profile = next(source(c) for c in cells if c["cell_type"] == "code" and "def memory_profile(" in source(c))
    exec(imports, namespace)
    exec(params, namespace)
    namespace.update(MODE=args.mode, SIZE="large", CONTEXT_LENGTH=512, USE_BF16=False, BATCH_SIZE=1)
    namespace["description"] = f"large_ctx512_{args.mode}_fp32_batch1"
    stage = "model_loading"
    try:
        exec(build, namespace)
        exec(run, namespace)
        exec(profile, namespace)
        stage = "warmup_or_measurement"
        result = namespace["memory_profile"](namespace["description"], namespace["run_model"], namespace["NUM_WARMUPS"])
        result.update(mode=args.mode, size="large", context=512, batch=1, parameter_dtype="float32",
                      replacement_for="2.7B", residual_activation_MiB=2.5)
        output = Path("memory_snapshots_assignment4") / (namespace["description"] + ".result.json")
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        # Use PyTorch's own Active Memory Timeline renderer, not a substitute plot.
        import pickle
        from torch.cuda._memory_viz import trace_plot
        with open(result["snapshot"], "rb") as f:
            snapshot = pickle.load(f)
        html = trace_plot(snapshot, device=0)
        output.with_suffix(".html").write_text(html)
        print("RESULT_JSON", json.dumps(result), flush=True)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        print("OOM", stage, str(error), flush=True)
        raise
    finally:
        model = namespace.get("model")
        if model is not None:
            model.zero_grad(set_to_none=True)
        model = None
        for key in ["model", "optimizer", "x", "y"]:
            namespace[key] = None
        import gc
        import torch
        torch.cuda.memory._record_memory_history(enabled=None)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
