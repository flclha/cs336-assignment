"""Patch only the six DDP/sharded adapter bodies; preserve attention adapters."""
import ast
from pathlib import Path
import shutil

path = Path(__file__).resolve().parent.parent/'tests'/'adapters.py'
source = path.read_text()
bodies = {
    'get_ddp_individual_parameters': 'from ddp_assignment_solutions import IndividualDDP\nreturn IndividualDDP(module)',
    'ddp_individual_parameters_on_after_backward': 'ddp_model.finish_gradient_synchronization()',
    'get_ddp_bucketed': 'from ddp_assignment_solutions import BucketedDDP\nreturn BucketedDDP(module, bucket_size_mb)',
    'ddp_bucketed_on_after_backward': 'ddp_model.finish_gradient_synchronization()',
    'ddp_bucketed_on_train_batch_start': 'pass  # State reset occurs in forward().',
    'get_sharded_optimizer': 'from ddp_assignment_solutions import ShardedOptimizer\nreturn ShardedOptimizer(params, optimizer_cls, **kwargs)',
}
lines = source.splitlines(keepends=True)
changes = []
for node in ast.parse(source).body:
    if isinstance(node,ast.FunctionDef) and node.name in bodies:
        # Keep signature and docstring, replace implementation only.
        first = node.body[1] if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str) else node.body[0]
        changes.append((first.lineno-1,node.end_lineno,''.join('    '+line+'\n' for line in bodies[node.name].splitlines())))
assert len(changes)==len(bodies)
backup = path.with_name('adapters.before_ddp_assignment.py.bak')
if not backup.exists():
    shutil.copy2(path,backup)
for start,end,replacement in sorted(changes,reverse=True):
    lines[start:end]=[replacement]
updated = ''.join(lines)
# The downloaded package __init__ queries distribution metadata for a different
# project name. Load this teaching module directly without changing that file.
if '# DDP assignment module path' not in updated:
    prefix="\n# DDP assignment module path\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cs336_systems'))\n"
    tree=ast.parse(updated)
    after_future=max((n.end_lineno for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='__future__'),default=0)
    updated_lines=updated.splitlines(keepends=True)
    updated_lines.insert(after_future,prefix)
    updated=''.join(updated_lines)
ast.parse(updated)
path.write_text(updated)
print('Installed 6 DDP/optimizer adapters; attention adapters preserved; backup:',backup)
