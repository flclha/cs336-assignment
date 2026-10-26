"""CPU/Gloo two-rank equivalence checks; no multi-GPU performance claims."""
import ast
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from ddp_assignment_solutions import NaiveDDP,FlatDDP,IndividualDDP,BucketedDDP,ShardedOptimizer

OUT=Path('ddp_assignment_results')


def existing_toy():
    # Reuse the notebook's exact SimpleNet class without importing torchvision
    # or launching its long MNIST training demo.
    notebook=json.loads(Path('作业2.ipynb').read_text())
    source=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and '%%writefile ddp_model_demo.py' in ''.join(c['source']))
    source='\n'.join(source.splitlines()[1:])
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='SimpleNet')
    scope=dict(nn=nn,F=F)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<existing notebook SimpleNet>','exec'),scope)
    return scope['SimpleNet'](784,50,10)


def worker(rank,port):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),GLOO_SOCKET_IFNAME='lo')
    dist.init_process_group('gloo',rank=rank,world_size=2,timeout=timedelta(seconds=120))
    results=[]
    try:
        for mode in ['naive','flat','individual','bucket_0.001','bucket_1']:
            torch.manual_seed(rank+42)
            model=existing_toy()
            if mode=='naive': ddp=NaiveDDP(model)
            elif mode=='flat': ddp=FlatDDP(model)
            elif mode=='individual': ddp=IndividualDDP(model)
            else: ddp=BucketedDDP(model,float(mode.split('_')[1]))
            baseline=deepcopy(model)
            opt=torch.optim.AdamW(ddp.parameters(),lr=1e-3)
            refopt=torch.optim.AdamW(baseline.parameters(),lr=1e-3)
            errors=[]
            call_counts=[]
            for step in range(5):
                gen=torch.Generator().manual_seed(100+step)
                x=torch.randn(32,784,generator=gen)
                y=torch.randint(10,(32,),generator=gen)
                opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
                F.nll_loss(baseline(x),y).backward();refopt.step()
                counter=[0]
                original=dist.all_reduce
                def counted(*args,**kwargs):
                    counter[0]+=1
                    return original(*args,**kwargs)
                dist.all_reduce=counted
                try:
                    F.nll_loss(ddp(x[rank*16:(rank+1)*16]),y[rank*16:(rank+1)*16]).backward()
                    during_backward=counter[0]
                    ddp.finish_gradient_synchronization()
                finally:dist.all_reduce=original
                opt.step()
                error=max((p-r).abs().max().item() for p,r in zip(model.parameters(),baseline.parameters()))
                for p,r in zip(model.parameters(),baseline.parameters()):
                    torch.testing.assert_close(p,r,atol=1e-6,rtol=1e-5)
                errors.append(error);call_counts.append(counter[0])
                if mode.startswith(('individual','bucket')):
                    assert during_backward==counter[0]>0,'Communication not launched during backward'
            results.append(dict(test=mode,status='PASS',steps=5,max_abs_weight_error=max(errors),
                step_errors=errors,allreduce_calls_per_step=call_counts,
                backend='gloo',world_size=2,global_batch=32,local_batch=16,
                model='existing notebook SimpleNet(784,50,10)',optimizer='AdamW'))

        # Dynamic parameter groups must preserve old Adam state; also check SGD,
        # closure execution, different group hyperparameters and outer lr edits.
        for optimizer_cls in [torch.optim.AdamW,torch.optim.SGD]:
            torch.manual_seed(8)
            params=[nn.Parameter(torch.randn(5)),nn.Parameter(torch.randn(3))]
            refs=[nn.Parameter(p.detach().clone()) for p in params]
            kwargs=dict(lr=.01,weight_decay=.1)
            sh=ShardedOptimizer([{'params':[params[0]],'lr':.02},{'params':[params[1]]}],optimizer_cls,**kwargs)
            normal=optimizer_cls([{'params':[refs[0]],'lr':.02},{'params':[refs[1]]}],**kwargs)
            maxerror=0
            for step in range(6):
                if step==3:
                    params.append(nn.Parameter(torch.randn(7)))
                    refs.append(nn.Parameter(params[-1].detach().clone()))
                    sh.add_param_group({'params':[params[-1]],'lr':.03})
                    normal.add_param_group({'params':[refs[-1]],'lr':.03})
                if step==4:
                    sh.param_groups[0]['lr']=.005
                    normal.param_groups[0]['lr']=.005
                calls=[0]
                def closure():
                    calls[0]+=1
                    sh.zero_grad()
                    loss=sum(p.square().sum() for p in params)
                    loss.backward()
                    return loss
                def reference_closure():
                    normal.zero_grad()
                    loss=sum(p.square().sum() for p in refs)
                    loss.backward()
                    return loss
                loss=sh.step(closure);ref_loss=normal.step(reference_closure)
                assert calls[0]==1
                torch.testing.assert_close(loss,ref_loss)
                for p,r in zip(params,refs):
                    maxerror=max(maxerror,(p-r).abs().max().item())
                    torch.testing.assert_close(p,r,atol=1e-7,rtol=1e-6)
            state_bytes=sum(v.numel()*v.element_size() for state in sh.local_optimizer.state.values() for v in state.values() if torch.is_tensor(v))
            state_ranks=[None]*2
            dist.all_gather_object(state_ranks,state_bytes)
            results.append(dict(test='sharded_dynamic_'+optimizer_cls.__name__,status='PASS',steps=6,
                max_abs_weight_error=maxerror,state_bytes_per_rank=state_ranks,
                tested='add_param_group preserves state; lr changes; closure once; parameter sync'))
        if rank==0:
            OUT.mkdir(exist_ok=True)
            (OUT/'correctness.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(results,indent=2),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,),nprocs=2,join=True)
