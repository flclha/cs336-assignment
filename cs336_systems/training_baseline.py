import argparse
import gc
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from timeit import default_timer
import statistics
from torch.profiler import profile, ProfilerActivity

CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def build_state(size="small", context=128, mode="forward"):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = BasicsTransformerLM(
        vocab_size=10000, context_length=context,
        rope_theta=10000.0, **CONFIGS[size],
    ).to(device="cuda", dtype=torch.float32)
    model.train(mode != "inference")
    x = torch.randint(10000, (4, context), device="cuda")
    y = torch.randint(10000, (4, context), device="cuda")
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1) if mode == "train" else None
    return dict(model=model, x=x, y=y, optimizer=optimizer)

def run_step(state, mode="forward-backward"):
    model = state["model"]
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    forward_ms = 0.0
    backward_ms = 0.0
    loss_ms = 0.0
    optimizer_ms = 0.0

    # TODO A：在前向语句前后加入 default_timer 和 CUDA 同步。
    # TODO C：给前向代码块加 with torch.cuda.nvtx.range("forward")
    # 增加前向传播计时
    with torch.cuda.nvtx.range("forward"): # cuda计时
        torch.cuda.synchronize() # python计时
        start = default_timer()

        # 前向传播 (修复：必须放进 with 缩进内，否则 NVTX 抓不到)
        with torch.set_grad_enabled(mode != "inference"):
            logits = model(state["x"])
        torch.cuda.synchronize()
        forward_ms = ( default_timer()-start)*1000

    if mode in ("forward-backward", "train"):
        # TODO C：给 loss 代码块加 NVTX 标记 "loss"。
        with torch.cuda.nvtx.range("loss"):
        # 增加loss计时
            torch.cuda.synchronize()
            start = default_timer()

            loss = F.cross_entropy(logits.reshape(-1, 10000), state["y"].reshape(-1))
            torch.cuda.synchronize()
            loss_ms = (default_timer()-start)*1000

        # TODO C：给 backward 代码块加 NVTX 标记 "backward"。
        with torch.cuda.nvtx.range("backward"):
            torch.cuda.synchronize()
            start = default_timer()
            loss.backward()

            torch.cuda.synchronize()
            backward_ms = ( default_timer()-start)*1000
        if mode == "train":
            # TODO C：给 optimizer 代码块加 NVTX 标记 "optimizer"。
            with torch.cuda.nvtx.range('optimizer'): # 配合nsight使用的，在GPU时间线图上画带颜色标签 (修复: 改为 nvtx.range)
                torch.cuda.synchronize()
                start = default_timer()
                state["optimizer"].step()
                torch.cuda.synchronize()
                optimizer_ms = (default_timer()-start)*1000

    # TODO A：返回只含数字的计时字典；不要返回 logits 或 loss。
    return {
        "forward_ms": forward_ms,
        "loss_ms": loss_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
    }

def release_state(state):
    model = state.get("model")
    if model is not None:
        model.zero_grad(set_to_none=True)
    model = None
    state.clear()
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=CONFIGS, default="small")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--mode", choices=["forward", "inference", "forward-backward", "train"], default="inference")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    # TODO D：增加 --capture 这个 store_true 参数，再加入采集控制。
    # 增加开关参数
    parser.add_argument(
    "--torch-profile",
    action="store_true",
    help="开启 PyTorch Profiler",
)

    parser.add_argument("--capture", action = "store_true")
    args = parser.parse_args()
    if args.context < 1 or args.warmups < 0 or args.steps < 1:
        parser.error("context >= 1, warmups >= 0, steps >= 1")
    print(vars(args), flush=True)
    state = {}
    try:
        state = build_state(args.size, args.context, args.mode)
        for _ in range(args.warmups):
            run_step(state, args.mode)
        torch.cuda.synchronize()

        # TODO D：采集开始应在这里，在 warmup 之后。
        if args.capture:
            torch.cuda.cudart().cudaProfilerStart()
            
        samples = []
        try: # try, finally 异常处理代码，如果显存不够等导致try后的代码未能完整执行，finally中也会正常关闭profile分析
            for _ in range(args.steps):
                # TODO C：给整个单步增加 "step" NVTX 范围。
                with torch.cuda.nvtx.range("step"): # 对整个过程的宏观测试，把forward等多个单步用step包裹，形成完成的系统组成
                    samples.append(run_step(state, args.mode))
            torch.cuda.synchronize()
        finally:
            # TODO D：采集结束应在这里，用 try/finally 保证停止。
            if args.capture:
                torch.cuda.cudart().cudaProfilerStop()

        # TODO A：在采集结束后对 samples 内的计时数字求均值和样本标准差。
        # =========================
        # 统计各阶段耗时
        # =========================
        print("\n========== Benchmark Results ==========")
        for key in [
            "forward_ms",
            "loss_ms",
            "backward_ms",
            "optimizer_ms",
        ]:
            times = [sample[key] for sample in samples]

            mean_ms = statistics.mean(times)

            std_ms = (
                statistics.stdev(times)
                if len(times) > 1
                else 0.0
            )

            print(
                f"{key:<15} "
                f"mean = {mean_ms:9.4f} ms | "
                f"std = {std_ms:9.4f} ms"
            )
            
    finally:
        release_state(state)

if __name__ == "__main__":
    main()