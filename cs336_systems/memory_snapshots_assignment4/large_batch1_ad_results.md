## (a)、(d) 本次large替代实验实测结果

配置：large(d_model=1280,d_ff=5120,36层,20头)，FP32，context=512，batch=1，vocab=10000；预热1步后采集1步。原题2.7B/batch4未满足，此为用户授权的替代实验。

| 模式 | 基线/MiB | 峰值allocated/MiB | 峰值reserved/MiB |
|---|---:|---:|---:|
| inference | 3814.271 | 3907.067 | 3990.000 |
| train | 11426.421 | 15829.898 | 17728.000 |

(a) 推理仅保留常驻权重和短暂前向工作空间，时间线上各层重复出现较小的分配/释放峰值。完整训练在AdamW预热后已有优化器状态，前向保存反向激活使显存明显增长；反向释放激活的同时产生参数梯度，随后AdamW更新产生临时分配。不能仅凭一个尖峰断言阶段，应结合分配栈与下方阶段结束时显存确认。

![large batch1 inference](memory_snapshots_assignment4/large_inference_batch1.png)

![large batch1 train](memory_snapshots_assignment4/large_train_batch1.png)

(d) 单个残差流激活形状[1,512,1280]，FP32每元素4字节：1*512*1280*4/1024²=2.5 MiB；不含其他层、梯度、参数或attention矩阵。

补充：large/batch4/context512的完整训练仍OOM；batch1训练使用PYTORCH_ALLOC_CONF=expandable_segments:True，推理使用默认分配器，因此reserved峰值不宜用作纯模式对比；主表使用allocated。快照在预热后开启，参数与优化器常驻状态可能没有原始分配调用栈。截图来自PyTorch官方Active Memory Timeline渲染器，横向为分配事件序列，不能直接当作毫秒时间轴。

阶段结束时显存（MiB，不是各阶段峰值）：

```json
{
  "inference": {
    "after_forward_MiB": 3834.2705078125
  },
  "train": {
    "after_forward_MiB": 15684.8134765625,
    "after_backward_MiB": 15247.18408203125,
    "after_optimizer_MiB": 15245.18408203125
  }
}
```
