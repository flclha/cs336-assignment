"""Append runnable supplements and measured results; never rewrite original cells."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import ast
import uuid

ROOT=Path(__file__).resolve().parent
PATH=ROOT/'作业2.ipynb'
OUT=ROOT/'ddp_assignment_results'
MARKER='DDP_ASSIGNMENT_SUPPLEMENT_20261026'
original_bytes=PATH.read_bytes()
nb=json.loads(original_bytes)
original_cells=[]
for c in nb['cells']:
    if MARKER in ''.join(c.get('source',[])):
        break
    original_cells.append(c)
backup=PATH.with_name('作业2.before_ddp_supplement_20261026.ipynb')
if not backup.exists():shutil.copy2(PATH,backup)
nb['cells']=list(original_cells)
def md(text):
    nb['cells'].append(dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=text.splitlines(keepends=True)))
def code(text,output=None):
    cell=dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True))
    if output is not None:
        cell['outputs']=[dict(output_type='stream',name='stdout',text=output.splitlines(keepends=True))]
    nb['cells'].append(cell)
def source_cell(name):
    code(f'%%writefile {name}\n'+(ROOT/name).read_text())

correctness=json.loads((OUT/'correctness.json').read_text())
tests=json.loads((OUT/'pytest_summary.json').read_text())
simulations=[json.loads(p.read_text()) for p in OUT.glob('simulation_*.json')]
by_mode={r['mode']:r for r in simulations}
assert len(tests)==5 and all(r['returncode']==0 for r in tests),'Complete five official test runs first'
assert len(simulations)==7 and all(r['status']=='OK' for r in simulations),'Complete simulation cases first'

md(f'''# 作业二～五：新增实现、实测结果与结论
<!-- {MARKER} -->

原有 {len(original_cells)} 个单元格逐格保留；下面是补充区。建议直接从本区开始运行，不必重跑上方MNIST下载和单卡演示。新增区所有 `%%writefile` 仅写入新的补充文件，不覆盖原演示脚本；测试适配器的6个相关函数会接入新实现，并保留备份。

**本次性能数据是单GPU模拟：1张RTX 3090、2个普通Python进程均使用GPU 0，通过CPU＋Gloo同步梯度，路径为GPU→CPU→Gloo→GPU；不是2张GPU、不是NCCL性能，也不是题目指定XL模型的结果。** 两份small模型各128,625,408参数，d_model=768、d_ff=3072、12层、12头，词表10000、context=128、每进程batch=1、全局batch=2，FP32且关闭TF32。完整XL的两份FP32训练副本无法放进当前24GiB显存，因此此处使用small配置，仅用于学习流程和对比本次模拟中的开销。

## 0. 原代码审查与保留理由

| 原有内容 | 是否直接满足题意 | 本次处理 |
|---|---|---|
| naive逐参数SUM后除world_size | 核心同步算法可用 | 保留算法与原示例，新增随机数据/全局batch权重对照 |
| MNIST训练与计时 | 不足：没有单进程权重对照；计时漏前向、缺少CUDA同步；不是XL | 原示例不动，新增完整step测量 |
| flat梯度展平、单次all_reduce、写回 | 算法可用；原脚本只跑单rank MNIST | 保留算法，统一与naive相同模型、数据形状和计时范围 |
| 作业三DDPOverlapBucketed | 分桶不是逐参数；queue_callback在反向末尾触发，不能证明计算通信重叠 | 新增post-accumulate hook逐参数异步实现 |
| 分桶基线DDPNoBucket | 常规grad hook中p.grad未必写入；可能漏发通信 | 新增经双进程权重对照验证的基线和分桶版本 |
| 模拟XLModel | 原代码自己标明约0.5B，并非§1.1.2指定的XL | 新脚本明确small/XL两种配置，当前只实测small模拟 |
| ShardedOptimizer | 动态add_param_group重建优化器会丢失旧动量，参数索引缓存也可能失效 | 新增保存owner及旧state的实现，验证动态参数组 |

新实现适用于稠密梯度、各rank相同静态计算图、每step一次backward，不支持未使用的可训练参数、稀疏梯度或多次backward梯度累积。梯度共享参数与冻结参数已由项目测试覆盖。

参考：[PyTorch hook触发时机](https://docs.pytorch.org/docs/main/notes/autograd.html#backward-hooks-execution)、[分布式通信说明](https://docs.pytorch.org/docs/stable/distributed.html)。
''')
md('''## 1. 公共实现：逐参数、展平、异步逐参数、分桶和优化器分片

学习时重点比较：NaiveDDP.finish_gradient_synchronization中的循环，与FlatDDP中一次all_reduce的区别；IndividualDDP在每个参数梯度累积完成后就发送；BucketedDDP等待桶中所有梯度就绪才发送。`finish_gradient_synchronization()`必须在optimizer.step()之前调用。

分桶使用反向参数顺序，并固定跨rank的通信顺序。单个参数大于目标桶大小时独立成桶、不切分参数，因此1/10MiB是目标软上限；实际桶数见结果表。GPU＋Gloo仅在本次模拟时显式暂存到CPU；真实双GPU模式用NCCL，不走暂存路径。
''')
source_cell('ddp_assignment_solutions.py')
md('''## 2. 作业二：复用原SimpleNet，在随机数据上验证权重

下面脚本直接从上方ddp_model_demo.py代码格提取原SimpleNet类，不下载MNIST。单进程每步处理32条样本；两rank各处理其中不重叠的16条，使用mean loss、梯度SUM/world_size，再执行相同AdamW更新。每步都断言权重与全局batch参考一致，而不是仅检查loss下降。
''')
source_cell('validate_ddp_assignment.py')
code('''import sys, subprocess, json
from pathlib import Path

# 当前应在cs336_systems目录，内核使用项目.venv。
print("Python:", sys.executable)
subprocess.run([sys.executable, "validate_ddp_assignment.py"], check=True)
''')
table='| 同步方式 | 对照步数 | 每步all_reduce次数 | 最大权重绝对误差 | 结果 |\n|---|---:|---:|---:|---|\n'
for r in correctness:
    if 'allreduce_calls_per_step' in r:
        table+=f"| {r['test']} | {r['steps']} | {r['allreduce_calls_per_step'][0]} | {r['max_abs_weight_error']:.3e} | {r['status']} |\n"
md('### 作业二正确性结果（CPU/Gloo，真实双进程）\n\n'+table+'\n结论：在相同初始化、相同全局batch和相同优化器设置下，局部mean loss对应的梯度经SUM/world_size后，与单进程处理全局batch的权重更新在上述容差内一致。这个测试验证实现正确性，不是GPU加速效果。\n')

md('''## 3. 作业三、四、五：接入项目适配器并运行指定测试

只接入DDP和分片优化器的6个适配器；保留attention适配器。原项目__init__.py查询的发行包名与当前pyproject.toml不一致，适配器因此按补充模块路径导入，避免修改原包初始化文件。

项目测试文件的helper会在检测到GPU时优先使用CUDA，即使backend写gloo。因此测试子进程明确设置CUDA_VISIBLE_DEVICES为空，确保此次确实是CPU/Gloo双进程正确性验证。不会把单GPU共享冒充双GPU测试。
''')
source_cell('install_ddp_assignment_adapters.py')
source_cell('run_ddp_assignment_tests.py')
code('''subprocess.run([sys.executable, "install_ddp_assignment_adapters.py"], check=True)
# 将连续运行5轮项目测试，通常需要数分钟。
RERUN_TESTS = False
if RERUN_TESTS:
    subprocess.run([sys.executable, "run_ddp_assignment_tests.py"], check=True)
''')
testtable='| 轮次 | 项目测试结果 | 退出码 |\n|---:|---|---:|\n'
for r in tests:testtable+=f"| {r['repeat']} | {r['summary']} | {r['returncode']} |\n"
md('### 指定测试的实测结果\n\n'+testtable+'''\n每轮包含：逐参数DDP 2项、分桶DDP 6项、分片优化器2项，总计10项，5轮共50项通过。测试覆盖冻结参数、共享权重，以及分布式训练权重与单进程参考的一致性。

作业三结论：梯度在post-accumulate hook中累积完成后，立即发出对应异步all_reduce；不再等到反向结束才发起。CPU正确性和“反向返回前已发出通信”的检查不等于证明真实双GPU上已获得重叠加速，性能要看相应设备上的profile和计时。
''')

md('''## 4. 作业二性能、flat对照、作业四(a)：单GPU双进程模拟

下面脚本固定模型、输入形状、精度、优化器和计时范围，预热3步、正式10步，计时包括前向、交叉熵、反向、梯度同步和AdamW更新，排除随机输入创建及每步开头的barrier。每步取两个rank中较大的耗时，再计算均值及单步样本标准差。每个配置独立启动两进程，退出后释放显存，并检查训练后两副本参数一致。

**同步时间的口径要区分：** naive/flat在反向结束后同步，因此该列包含完整的梯度打包、设备拷贝、Gloo归约和写回；异步逐参数/分桶已在反向内部发起部分工作，该列仅是反向结束后暴露出来的等待与写回，不能当作全部通信时间，也不能把该列占比降低直接解释为网络更快。
''')
source_cell('benchmark_single_gpu_two_processes.py')
code('''RERUN_SIMULATION = False
if RERUN_SIMULATION:
    # 两个普通进程共享GPU 0；不是NCCL双卡实验。
    subprocess.run([sys.executable, "-u", "benchmark_single_gpu_two_processes.py"], check=True)

for p in sorted(Path("ddp_assignment_results").glob("simulation_*.json")):
    r = json.loads(p.read_text())
    print(r["mode"], f"{r['mean_step_ms']:.3f} ms/step",
          f"post-backward synchronization={r['mean_sync_after_backward_ms']:.3f} ms",
          f"replica error={r['max_replica_error']:.2e}")
''')
simtable='| 方式 | 每步梯度归约次数/桶数 | 完整step均值(ms) | 单步标准差(ms) | 反向后同步(ms) | 同步占比 | 相对flat速度比 |\n|---|---:|---:|---:|---:|---:|---:|\n'
for mode in ['naive','flat','individual','bucket_1','bucket_10','bucket_100','bucket_1000']:
    r=by_mode[mode]
    simtable+=f"| {mode} | {r['bucket_count']} | {r['mean_step_ms']:.3f} | {r['std_step_ms']:.3f} | {r['mean_sync_after_backward_ms']:.3f} | {100*r['sync_after_backward_fraction']:.2f}% | {by_mode['flat']['mean_step_ms']/r['mean_step_ms']:.3f} |\n"
md('### 模拟性能实测表：1张GPU、2进程、small，非双GPU/XL\n\n'+simtable)
naive,flat=by_mode['naive'],by_mode['flat']
best=min(simulations,key=lambda r:r['mean_step_ms'])
ratio=naive['mean_step_ms']/flat['mean_step_ms']
md(f'''### 作业二与flat对照结论

本次模拟naive每步 {naive['mean_step_ms']:.3f} ms，反向后完整梯度同步 {naive['mean_sync_after_backward_ms']:.3f} ms，占 {100*naive['sync_after_backward_fraction']:.2f}%；flat每步 {flat['mean_step_ms']:.3f} ms，完整梯度同步 {flat['mean_sync_after_backward_ms']:.3f} ms，占 {100*flat['sync_after_backward_fraction']:.2f}%。naive/flat耗时比为 {ratio:.3f}，实测并不预设展平一定更快，因为展平还涉及大缓冲区、拷贝和同步；这些数值仅描述单GPU共享与CPU暂存通信。

### 作业四(a)的模拟结论

七种配置中，本轮完整step最低的是 `{best['mode']}`，均值 {best['mean_step_ms']:.3f} ms；表中以反向后单次归约的flat作为“无分桶”参照，速度比大于1才表示更快。通常预期小桶较早就绪有利于重叠，而大桶能减少调用次数，但会推迟发起通信。本次两个进程争用同一GPU，且GPU→CPU暂存会造成同步，不能据此判断双GPU/NCCL上的最优桶大小；模拟结果与理想重叠不一致并不反驳分桶机制。若要验证真实重叠，应在两张独立GPU上使用NCCL、相同XL模型、相同全局batch和足够预热，并从所有rank采集profile核实通信与反向时间线的重叠。

### 正式作业要求的硬件数据状态

| 正式配置 | 完整step | 梯度通信/反向后尾部 | 状态 |
|---|---|---|---|
| 2 GPU、XL、naive | 未测 | 未测 | 当前仅1张GPU |
| 2 GPU、XL、flat | 未测 | 未测 | 当前仅1张GPU |
| 2 GPU、XL、桶1/10/100/1000 MiB | 未测 | 未测 | 当前仅1张GPU |

换到显存足够的双GPU机器后，从cs336_systems目录执行下面命令（FP32 XL可能要求更大显存）：

```bash
../.venv/bin/python benchmark_single_gpu_two_processes.py --real-two-gpu --model-size xl
```

真实模式将rank i绑定GPU i，使用NCCL，结果写为real_xl_*.json，不覆盖simulation_*.json。本机不要执行该命令来代替模拟，它会先检查至少2张GPU。
''')

md(r'''## 5. 作业四(b)：通信开销模型与最优桶大小

设 $n_b$ 个等大桶，每桶字节数 $b=s/n_b$，每桶反向计算时间为 $c=b/w$，归约耗时为 $c+o$。理想化假设计算与通信可以并行、通信串行处理、忽略拷贝和争用，则第一桶在 $c$ 时刻开始通信，最后结束于 $c+n_b(c+o)$；反向在 $n_bc$ 时刻结束，因此反向之后暴露的通信开销为

$$T_{mathrm{overhead}}(n_b)=n_bo+rac{s}{n_bw}.$$

对连续变量求导得 $o-s/(wn_b^2)=0$，所以

$$n_b^*=sqrt{rac{s}{wo}},qquad b^*=rac{s}{n_b^*}=sqrt{swo}.$$

实际 $n_b$ 至少为1且为整数，应比较连续最优值附近的可行桶数，并受参数边界及实际计算顺序限制。这里 $s$ 用字节、$w$ 用字节/秒、$o$ 用秒；本次CPU暂存模拟存在额外同步与资源争用，不能直接用这个理想模型拟合单GPU共享耗时。
''')

opttable='| 附加验证 | 更新步数 | 最大参数绝对误差 | 结果 |\n|---|---:|---:|---|\n'
for r in correctness:
    if r['test'].startswith('sharded_'):
        opttable+=f"| {r['test']} | {r['steps']} | {r['max_abs_weight_error']:.3e} | {r['status']} |\n"
md('''## 6. 作业五：分片优化器的结果与结论

实现已包含在公共模块，并由get_sharded_optimizer适配器接入。每个rank只为归自己负责的参数创建优化器状态，step后将owner更新的参数广播给其他rank；使用前仍需先平均梯度，优化器分片本身不代替DDP同步。

'''+opttable+'''
除了项目测试5轮通过，附加测试还检查了动态添加参数组不会清空旧Adam动量、各组不同学习率、外部修改学习率、closure恰好调用一次，以及step后副本一致。参数是按完整张量贪心分配，因此状态大小不保证精确均分；例如某个大权重矩阵可能使某rank占用更多状态。该方法减少优化器状态冗余，仍保留完整参数与梯度，并增加参数同步开销。

## 7. 文件与执行顺序

已有实测结果保存在ddp_assignment_results目录。需要从头复现时：先运行公共实现的writefile，再写出并运行验证脚本、安装适配器、执行项目测试，最后在没有其他CPU/GPU重任务时运行单GPU模拟。不要同时运行正确性测试与性能计时。

原notebook备份：作业2.before_ddp_supplement_20261026.ipynb；适配器备份：../tests/adapters.before_ddp_assignment.py.bak。原有单元格和旧输出只是演示历史，不作为本次正确性或性能结论的证据。
''')
for c in nb['cells'][len(original_cells):]:
    if c['cell_type']=='code':
        source=''.join(c['source'])
        if source.startswith('%%writefile'):source='\n'.join(source.splitlines()[1:])
        ast.parse(source)
assert PATH.read_bytes()==original_bytes,'Notebook changed while building; refusing to overwrite'
PATH.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
saved=json.loads(PATH.read_text())
assert saved['cells'][:len(original_cells)]==original_cells
(OUT/'notebook_update_manifest.json').write_text(json.dumps(dict(original_sha256=hashlib.sha256(original_bytes).hexdigest(),
    original_cells_preserved=len(original_cells),added_cells=len(nb['cells'])-len(original_cells),
    backup=str(backup),simulation_only=True),indent=2))
print('Updated notebook:',len(original_cells),'original cells preserved;',len(nb['cells'])-len(original_cells),'added')
