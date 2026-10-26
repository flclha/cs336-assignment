# Mixed precision (c) measured results

RTX 3090; PyTorch 2.9.1+cu128; Python 3.12.15; FP32 parameters; TF32 off; batch=4; vocab=10000; context=128; warmups=5; trials=10; no optimizer; each run in a new process.

| Model | FP32 forward ms | BF16 forward ms | Speedup | FP32 backward ms | BF16 backward ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| small | 38.531 | 45.728 | 0.843x | 43.554 | 50.830 | 0.857x |
| medium | 84.359 | 94.685 | 0.891x | 93.814 | 107.474 | 0.873x |
| large | 124.976 | 138.501 | 0.902x | 144.965 | 156.392 | 0.927x |
| xl | 169.370 | 186.780 | 0.907x | 307.067 | 232.431 | 1.321x |
| 2.7B | OOM | OOM | - | OOM | OOM | - |

Speedup = FP32/BF16. These are means of 10 individual steps, not grouped means. Backward excludes forward and loss computation.

2.7B failed during warmup: FP32 backward OOM; BF16 forward OOM. No complete measurements for this model.

BF16 logits verified for all successful BF16 samples; losses finite. GPU returned to idle (~711 MiB desktop usage) after all workers exited.

## Sample standard deviations (ms)

| Model | FP32 forward | BF16 forward | FP32 backward | BF16 backward |
|---|---:|---:|---:|---:|
| small | 0.609 | 0.947 | 0.386 | 0.795 |
| medium | 0.467 | 0.688 | 0.523 | 0.896 |
| large | 0.771 | 1.158 | 0.424 | 3.723 |
| xl | 0.848 | 3.594 | 0.723 | 1.998 |
