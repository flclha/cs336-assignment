"""Build a measured report and notebook from allreduce_assignment1_results/*.json."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'allreduce_assignment1_results'
rows=[json.loads(p.read_text()) for p in OUT.glob('*_p*MiB.json')]
lookup={(r['backend'],r['size_MiB'],r['world_size']):r for r in rows}

def table(backend,field):
    t='| 每个进程的张量大小 | 2进程 | 4进程 | 6进程 |\n|---|---:|---:|---:|\n'
    for size,label in [(1,'1 MiB'),(10,'10 MiB'),(100,'100 MiB'),(1024,'1 GiB')]:
        v=[]
        for w in [2,4,6]:
            r=lookup.get((backend,size,w))
            v.append('未测' if r is None else (f"{r[field]:.3f}" if r['status']=='OK' else ('硬件不足' if r['status']=='UNAVAILABLE' else r['status'])))
        t+=f'| {label} | '+' | '.join(v)+' |\n'
    return t

intro='''# 作业2.ipynb：作业一 all-reduce 通信实验

本实验对应 distributed_communication_single_node，不是 attention 或 Transformer 计时。原 notebook 保留，新增独立脚本 `allreduce_assignment1_benchmark.py` 和这份实验记录。

实验设备为单节点 Intel Xeon E5-2697 v4（18核36线程、约62 GiB内存），1张 RTX 3090；PyTorch 2.9.1+cu128。Gloo 使用 CPU 张量和本机 loopback 接口，每个 rank 是独立进程，PyTorch 算子线程数固定为1。

容量定义：MiB=2²⁰字节，GiB=2³⁰字节。每个 rank 的 FP32 张量分别为1、10、100 MiB和1024 MiB（1 GiB）。原示例把1024²字节写成MB、且最大配置写成1000，这里明确采用二进制单位，并将最后一项设为完整1 GiB。

每组先验证整个张量的 SUM 结果，再预热5次，barrier对齐后连续执行20次 all_reduce。对每个 rank 的总耗时除以20，再取各rank中的最大均值作为集体通信耗时；初始化、数据构造、正确性验证、barrier和结果汇总均不计时。每组在独立进程中运行，超时限制280秒，结束销毁进程组。

计时阶段使用零张量，避免反复SUM导致数值溢出；张量大小、dtype和通信操作保持不变。本结果是20次调用的平均耗时，不是一次调用，也没有测量单次耗时的标准差。
'''
analysis='''大张量配置的单次 all-reduce 耗时总体更长，因为需要传输和归约的数据更多，小张量则更容易受到调用、同步和进程调度开销影响。增加进程数不会自动加快 all-reduce：每个rank仍持有同样大的张量，同时更多进程竞争单机CPU、内存带宽和loopback通信资源，具体趋势应以实测表格为准。当前只有一张GPU，无法完成要求的2/4/6卡NCCL对比，因此不能从CPU数据判断两种后端的性能差距。'''
report=intro+'\n## Gloo + CPU：平均耗时（ms/次）\n\n'+table('gloo','mean_ms')
report+='\n## Gloo + CPU：算法带宽（GB/s）\n\n'+table('gloo','algorithm_GB_s')
report+='\n这里的算法带宽仅为“每rank张量字节数 / 单次耗时”，GB按10⁹字节计算，不是物理链路带宽或理论峰值。\n'
report+='\n## NCCL + GPU：平均耗时（ms/次）\n\n'+table('nccl','mean_ms')
report+='\n机器只有1张GPU，多进程NCCL需要各进程独占使用的GPU，不能把多个rank挤在同一张卡上作为上述多卡实验。\n'
report+='\n## 可用于报告的三句话分析\n\n'+analysis+'\n'
report+='\n## 如何重新运行\n\n```bash\ncd ~/projects/diy-llm/coursework/assignment2-systems/cs336_systems\n../.venv/bin/python -u allreduce_assignment1_benchmark.py\n```\n\n以后换到至少6卡的机器，脚本会自动运行完整NCCL组合；目前自动将其标为UNAVAILABLE。单组CPU测试示例：\n\n```bash\n../.venv/bin/python allreduce_assignment1_benchmark.py --case --backend gloo --world-size 4 --size-mib 100\n```\n'
report+='\n方法参考：[PyTorch distributed官方文档](https://docs.pytorch.org/docs/stable/distributed.html)。\n'
(OUT/'实验结果.md').write_text(report,encoding='utf-8')
(OUT/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for w in [2,4,6]:
    rs=[lookup[('gloo',s,w)] for s in [1,10,100,1024] if ('gloo',s,w) in lookup and lookup[('gloo',s,w)]['status']=='OK']
    for ax,field in zip(axes,['mean_ms','algorithm_GB_s']):
        ax.plot([r['size_MiB'] for r in rs],[r[field] for r in rs],marker='o',label=f'{w} processes')
for ax,title,ylabel in zip(axes,['All-reduce mean latency','Algorithm bandwidth: payload / time'],['Latency (ms / collective)','Algorithm bandwidth (GB/s)']):
    ax.set_xscale('log'); ax.set_xlabel('FP32 tensor size per rank (MiB)')
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend()
axes[0].set_yscale('log')
fig.suptitle('Single-node Gloo / CPU | 5 warmups, 20 measured all-reduces')
fig.savefig(OUT/'allreduce_comparison.png',dpi=180)
plt.close(fig)

cells=[dict(cell_type='markdown',metadata={},source=report.splitlines(keepends=True)),
       dict(cell_type='markdown',metadata={},source=['![实测图](allreduce_assignment1_results/allreduce_comparison.png)']),
       dict(cell_type='code',metadata={},execution_count=None,outputs=[],source='''from pathlib import Path
import json, sys, subprocess
RERUN = False  # 改为True才会重新运行全部组合并覆盖结果
if RERUN:
    subprocess.run([sys.executable, "-u", "allreduce_assignment1_benchmark.py"], check=True)

data = [json.loads(p.read_text()) for p in Path("allreduce_assignment1_results").glob("*_p*MiB.json")]
for r in sorted(data, key=lambda r: (r["backend"], r["size_MiB"], r["world_size"])):
    print(r["backend"], r["world_size"], r["size_MiB"], r["status"],
          f"{r['mean_ms']:.3f} ms" if r["status"] == "OK" else r.get("reason", ""))
'''.splitlines(keepends=True))]
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (project .venv)',language='python',name='python3'),language_info=dict(name='python')),nbformat=4,nbformat_minor=5)
(ROOT/'作业2_作业1实验结果.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('Built report and notebook; records:',len(rows))
