"""SIMULATION ONLY: 2 Python ranks sharing GPU 0, CPU/Gloo gradient transport.

Not a two-GPU/NCCL benchmark. Does not reproduce cross-GPU topology/bandwidth.
Small language model instead of XL to fit both training replicas on 24GiB.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

OUT=Path('ddp_assignment_results')
MODES=['naive','flat','individual','bucket_1','bucket_10','bucket_100','bucket_1000']


def worker(rank,mode,port,steps,real_two_gpu=False,model_size='small'):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from cs336_basics.model import BasicsTransformerLM
    from ddp_assignment_solutions import NaiveDDP,FlatDDP,IndividualDDP,BucketedDDP
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device=f'cuda:{rank if real_two_gpu else 0}'
    backend='nccl' if real_two_gpu else 'gloo'
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42)
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),GLOO_SOCKET_IFNAME='lo')
    dist.init_process_group(backend,rank=rank,world_size=2,timeout=timedelta(seconds=180))
    try:
        config=dict(vocab_size=10000,context_length=128,d_model=768,d_ff=3072,
                    num_layers=12,num_heads=12,rope_theta=10000.)
        if model_size=='xl':
            config.update(d_model=1600,d_ff=6400,num_layers=48,num_heads=25)
        model=BasicsTransformerLM(**config).to(device=device,dtype=torch.float32)
        if mode=='naive': ddp=NaiveDDP(model)
        elif mode=='flat': ddp=FlatDDP(model)
        elif mode=='individual': ddp=IndividualDDP(model)
        else: ddp=BucketedDDP(model,float(mode.split('_')[1]))
        opt=torch.optim.AdamW(ddp.parameters(),lr=3e-4,weight_decay=.1,foreach=False)
        gen=torch.Generator(device=device).manual_seed(100+rank)
        x=torch.randint(10000,(1,128),device=device,generator=gen)
        y=torch.randint(10000,(1,128),device=device,generator=gen)
        samples=[]
        def step(measure):
            opt.zero_grad(set_to_none=True)
            dist.barrier()  # Excluded from timing; align ranks each step.
            torch.cuda.synchronize()
            start=time.perf_counter()
            logits=ddp(x)
            loss=F.cross_entropy(logits.reshape(-1,10000),y.reshape(-1))
            loss.backward()
            torch.cuda.synchronize()
            backward_end=time.perf_counter()
            ddp.finish_gradient_synchronization()
            torch.cuda.synchronize()
            synchronization_end=time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            end=time.perf_counter()
            if measure:
                samples.append(dict(step_ms=(end-start)*1000,
                    forward_backward_ms=(backward_end-start)*1000,
                    sync_after_backward_ms=(synchronization_end-backward_end)*1000,
                    optimizer_ms=(end-synchronization_end)*1000))
        for _ in range(3):step(False)
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(steps):step(True)
        peak=torch.cuda.max_memory_allocated()/1024**2
        # Check replicas still agree after training. One parameter at a time
        # to avoid another complete model copy on GPU/CPU.
        max_error=0.
        for p in model.parameters():
            local=p.detach() if real_two_gpu else p.detach().cpu()
            root=local.clone()
            dist.broadcast(root,src=0)
            max_error=max(max_error,(local-root).abs().max().item())
        assert max_error<1e-6, f'Replicas diverged: {max_error}'
        gathered=[None]*2
        dist.all_gather_object(gathered,dict(rank=rank,samples=samples,
            peak_allocated_MiB=peak,max_replica_error=max_error))
        if rank==0:
            # Per iteration, take the maximum elapsed time across the ranks.
            times=[max(r['samples'][i]['step_ms'] for r in gathered) for i in range(steps)]
            sync_times=[max(r['samples'][i]['sync_after_backward_ms'] for r in gathered) for i in range(steps)]
            mean=statistics.mean(times)
            sync_mean=statistics.mean(sync_times)
            result=dict(status='OK',mode=mode,simulation=not real_two_gpu,physical_gpus=2 if real_two_gpu else 1,
                world_size=2,gpu=torch.cuda.get_device_name(0),backend=backend,
                gradient_transport='NCCL GPU all_reduce' if real_two_gpu else 'GPU -> CPU -> Gloo all_reduce -> GPU',
                model_size=model_size,config=config,local_batch=1,global_batch=2,
                dtype='float32',tf32=False,optimizer='torch.optim.AdamW foreach=False',
                warmups=3,steps=steps,mean_step_ms=mean,std_step_ms=statistics.stdev(times),
                mean_sync_after_backward_ms=sync_mean,sync_after_backward_fraction=sync_mean/mean,
                synchronization_meaning=('full blocking gradient synchronization including staging and packing'
                    if mode in ['naive','flat'] else 'exposed post-backward wait and copy-back; excludes communication launched during backward'),
                mean_peak_per_rank_MiB=statistics.mean(r['peak_allocated_MiB'] for r in gathered),
                parameter_count=sum(p.numel() for p in model.parameters()),
                parameter_tensors=sum(p.requires_grad for p in model.parameters()),
                bucket_count=len(ddp.buckets) if hasattr(ddp,'buckets') else (1 if mode=='flat' else sum(p.requires_grad for p in model.parameters())),
                max_replica_error=max(r['max_replica_error'] for r in gathered),ranks=gathered)
            prefix=f'real_{model_size}' if real_two_gpu else 'simulation'
            (OUT/f'{prefix}_{mode}.json').write_text(json.dumps(result,indent=2))
            print('RESULT',json.dumps({k:v for k,v in result.items() if k!='ranks'}),flush=True)
    finally:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',action='store_true')
    p.add_argument('--mode',choices=MODES,default='naive')
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    p.add_argument('--real-two-gpu',action='store_true',help='Real NCCL, rank i -> GPU i; needs >=2 GPUs')
    p.add_argument('--model-size',choices=['small','xl'],default='small')
    args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.case:
        import torch.multiprocessing as mp
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        mp.spawn(worker,args=(args.mode,port,args.steps,args.real_two_gpu,args.model_size),nprocs=2,join=True)
        return
    for mode in args.modes:
        print('RUN REAL' if args.real_two_gpu else 'RUN SIMULATION',mode,flush=True)
        extra=['--real-two-gpu'] if args.real_two_gpu else []
        if args.real_two_gpu:
            import torch
            if torch.cuda.device_count()<2:
                raise RuntimeError('Real two-GPU benchmark requires two distinct GPUs')
        proc=subprocess.Popen([sys.executable,'-u',__file__,'--case','--mode',mode,'--steps',str(args.steps),
            '--model-size',args.model_size,*extra],
            start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        try:
            proc.wait(timeout=280)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise RuntimeError('Simulation case timed out; stopped')
        if proc.returncode:
            print('FAILED',mode,proc.returncode,flush=True)
            raise RuntimeError('Stop on failure; investigate before collecting more cases')
    print('SIMULATION_DONE',flush=True)


if __name__=='__main__':main()
