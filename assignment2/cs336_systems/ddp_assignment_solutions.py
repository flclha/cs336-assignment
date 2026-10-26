"""Supplement to 作业2.ipynb. One backward per step; dense gradients.

Use post-accumulate hooks so p.grad exists before launching communication.
All ranks must use the same static graph and parameter order. Unused trainable
parameters / gradient accumulation across multiple backwards are not supported.
"""
import torch
from torch import nn
import torch.distributed as dist
from torch.optim import Optimizer


def broadcast_tensor(tensor, src):
    if tensor.is_cuda and dist.get_backend() == 'gloo':
        # Explicit single-GPU simulation transport, not NCCL/multi-GPU.
        host = tensor.detach().cpu()
        dist.broadcast(host,src=src)
        tensor.detach().copy_(host)
    else:
        dist.broadcast(tensor.detach(),src=src)


class _StagedWork:
    def __init__(self,work,host,destination):
        self.work,self.host,self.destination=work,host,destination
    def wait(self):
        self.work.wait()
        self.destination.copy_(self.host)


def reduce_buffer(buffer,async_op=False):
    if buffer.is_cuda and dist.get_backend() == 'gloo':
        host=buffer.detach().cpu()
        work=dist.all_reduce(host,async_op=async_op)
        if async_op:
            return _StagedWork(work,host,buffer)
        buffer.copy_(host)
        return None
    return dist.all_reduce(buffer,async_op=async_op)


def broadcast_module(module):
    for tensor in list(module.parameters()) + list(module.buffers()):
        broadcast_tensor(tensor,src=0)


class NaiveDDP(nn.Module):
    """Existing notebook algorithm: one blocking all-reduce per parameter."""
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        broadcast_module(module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for p in self.module.parameters():
            if p.grad is not None:
                reduce_buffer(p.grad)
                p.grad.div_(self.world_size)


class FlatDDP(NaiveDDP):
    """Existing flatten -> one all-reduce -> copy back algorithm."""
    def finish_gradient_synchronization(self):
        params = [p for p in self.module.parameters() if p.grad is not None]
        if not params:
            return
        buffer = torch.cat([p.grad.reshape(-1) for p in params])
        reduce_buffer(buffer)
        buffer.div_(self.world_size)
        offset = 0
        for p in params:
            p.grad.copy_(buffer[offset:offset+p.numel()].view_as(p))
            offset += p.numel()


class BucketedDDP(nn.Module):
    """Launch ready buckets during backward, in deterministic bucket order."""
    def __init__(self, module, bucket_size_mb=None, individual=False):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        broadcast_module(module)
        limit = float('inf') if bucket_size_mb is None else bucket_size_mb*1024**2
        if limit <= 0:
            raise ValueError('bucket_size_mb must be positive')
        groups, current, size = [], [], 0
        for p in reversed(list(module.parameters())):
            if not p.requires_grad:
                continue
            nbytes = p.numel()*p.element_size()
            incompatible = current and (p.dtype != current[0].dtype or p.device != current[0].device)
            if current and (individual or size+nbytes > limit or incompatible):
                groups.append(current)
                current, size = [], 0
            current.append(p)
            size += nbytes
        if current:
            groups.append(current)
        self.buckets = []
        self.hooks = []
        self.next_bucket = 0
        self.active = False
        for idx, params in enumerate(groups):
            buf = torch.empty(sum(p.numel() for p in params),device=params[0].device,dtype=params[0].dtype)
            bucket = dict(params=params,buffer=buf,ready=set(),work=None,offsets={})
            offset = 0
            for p in params:
                bucket['offsets'][p] = offset
                offset += p.numel()
                self.hooks.append(p.register_post_accumulate_grad_hook(
                    lambda param, i=idx: self._ready(i,param)))
            self.buckets.append(bucket)

    def forward(self, *args, **kwargs):
        if self.active:
            raise RuntimeError('Call finish_gradient_synchronization before next forward')
        self.active = torch.is_grad_enabled() and any(p.requires_grad for p in self.module.parameters())
        self.next_bucket = 0
        for b in self.buckets:
            b['ready'].clear()
            b['work'] = None
        return self.module(*args, **kwargs)

    def _ready(self, index, param):
        b = self.buckets[index]
        if param in b['ready']:
            raise RuntimeError('Only one backward per step is supported')
        offset = b['offsets'][param]
        b['buffer'][offset:offset+param.numel()].copy_(param.grad.reshape(-1))
        b['ready'].add(param)
        # Fixed order is important: all ranks issue matching collectives.
        while self.next_bucket < len(self.buckets):
            next_b = self.buckets[self.next_bucket]
            if len(next_b['ready']) != len(next_b['params']):
                break
            next_b['work'] = reduce_buffer(next_b['buffer'],async_op=True)
            self.next_bucket += 1

    def finish_gradient_synchronization(self):
        if self.next_bucket != len(self.buckets):
            raise RuntimeError('Unused trainable parameter or backward was not run')
        for b in self.buckets:
            b['work'].wait()
            b['buffer'].div_(self.world_size)
            for p in b['params']:
                offset = b['offsets'][p]
                p.grad.copy_(b['buffer'][offset:offset+p.numel()].view_as(p))
        self.active = False


class IndividualDDP(BucketedDDP):
    """One parameter per bucket: actual per-parameter asynchronous reduction."""
    def __init__(self,module):
        super().__init__(module,individual=True)


class ShardedOptimizer(Optimizer):
    """ZeRO-1: assign whole parameters by greedy numel balancing.

    Existing owners and optimizer state are retained on add_param_group.
    Gradients must already be averaged across ranks before step().
    """
    def __init__(self,params,optimizer_cls,**kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.owners = {}
        self.loads = [0]*self.world_size
        self.local_optimizer = None
        super().__init__(params,defaults=dict(kwargs))
        local_groups = [self._local_group(g) for g in self.param_groups]
        self.local_optimizer = optimizer_cls(local_groups,**kwargs)
        self.state = self.local_optimizer.state

    def _local_group(self,group):
        return dict(group,params=[p for p in group['params'] if self.owners[p]==self.rank])

    def add_param_group(self,param_group):
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for p in group['params']:
            owner = min(range(self.world_size),key=lambda rank:self.loads[rank])
            self.owners[p] = owner
            self.loads[owner] += p.numel()
        if self.local_optimizer is not None:
            self.local_optimizer.add_param_group(self._local_group(group))

    def step(self,closure=None,**kwargs):
        for global_group,local_group in zip(self.param_groups,self.local_optimizer.param_groups):
            for key,value in global_group.items():
                if key != 'params':
                    local_group[key] = value
        loss = self.local_optimizer.step(closure=closure,**kwargs)
        for group in self.param_groups:
            for p in group['params']:
                broadcast_tensor(p,src=self.owners[p])
        return loss

    def load_state_dict(self,state_dict):
        super().load_state_dict(state_dict)
        self.local_optimizer.state = self.state
