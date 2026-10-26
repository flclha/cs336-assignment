"""Assignment 2 notebook / assignment 1: single-node all-reduce benchmark.

Run with the project Python. Each configuration is a fresh subprocess, <5min.
Sizes use binary units explicitly: 1, 10, 100 MiB and 1 GiB per rank.
"""
import argparse
from datetime import timedelta
import json
import os
import signal
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

OUT = Path('allreduce_assignment1_results')


def rank_worker(rank, world, size_mib, backend, port):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    if backend == 'nccl':
        torch.cuda.set_device(rank)
    device = torch.device('cpu' if backend == 'gloo' else f'cuda:{rank}')
    dist.init_process_group(backend, rank=rank, world_size=world,
                            timeout=timedelta(seconds=120))
    try:
        count = size_mib * 1024**2 // 4
        x = torch.full((count,), float(rank+1), dtype=torch.float32, device=device)
        dist.all_reduce(x)
        expected = world * (world+1) / 2
        # Full tensor correctness verification, outside timing.
        assert bool(torch.all(x == expected).item()), 'Incorrect all-reduce sum'
        # Repeated SUM on zeros avoids overflow; same payload/collective.
        x.zero_()
        sync = (lambda: torch.cuda.synchronize(device)) if backend == 'nccl' else (lambda: None)
        for _ in range(5):
            dist.all_reduce(x)
        sync()
        dist.barrier()
        sync()
        # Match notebook: measure a block of 20 all-reduces, divide by 20.
        start = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(x)
        sync()
        elapsed = time.perf_counter() - start
        local = dict(rank=rank, elapsed_s=elapsed, mean_ms=elapsed*1000/20)
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        if rank == 0:
            rank_means = [v['mean_ms'] for v in gathered]
            mean_ms = max(rank_means)
            result = dict(status='OK', backend=backend, device=str(device.type),
                          world_size=world, size_MiB=size_mib, bytes_per_rank=count*4,
                          dtype='float32', warmups=5, iterations=20,
                          mean_ms=mean_ms, rank_mean_ms=rank_means,
                          rank_mean_spread_ms=max(rank_means)-min(rank_means),
                          algorithm_GB_s=(count*4)/(mean_ms/1000)/1e9,
                          correctness='full tensor SUM verified', ranks=gathered,
                          torch=str(torch.__version__), cpu_threads_per_rank=1)
            (OUT / f'{backend}_p{world}_{size_mib}MiB.json').write_text(json.dumps(result,indent=2))
            print('RESULT', json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', action='store_true')
    p.add_argument('--backend', choices=['gloo','nccl'], default='gloo')
    p.add_argument('--world-size', type=int, default=2)
    p.add_argument('--size-mib', type=int, default=1)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.case:
        import torch.multiprocessing as mp
        with socket.socket() as s:
            s.bind(('127.0.0.1',0))
            port = s.getsockname()[1]
        mp.spawn(rank_worker,args=(args.world_size,args.size_mib,args.backend,port),
                 nprocs=args.world_size,join=True)
        return
    import torch
    gpu_count = torch.cuda.device_count()
    metadata = dict(gpu_count=gpu_count, gpus=[torch.cuda.get_device_name(i) for i in range(gpu_count)],
                    cpu_model=subprocess.check_output(['lscpu'],text=True),
                    torch=str(torch.__version__), size_convention='MiB=2**20 bytes; 1GiB=1024MiB',
                    metric='max across ranks of block elapsed / 20; barriers and setup excluded')
    (OUT/'environment.json').write_text(json.dumps(metadata,indent=2))
    for backend in ['gloo','nccl']:
        for size in [1,10,100,1024]:
            for world in [2,4,6]:
                path=OUT/f'{backend}_p{world}_{size}MiB.json'
                if backend=='nccl' and gpu_count<world:
                    path.write_text(json.dumps(dict(status='UNAVAILABLE',backend=backend,
                        world_size=world,size_MiB=size,
                        reason=f'Needs {world} distinct GPUs; only {gpu_count} available'),indent=2))
                    print('SKIP',backend,world,size,'insufficient GPUs',flush=True)
                    continue
                print('RUN',backend,world,size,flush=True)
                start=time.perf_counter()
                try:
                    proc=subprocess.Popen([sys.executable,'-u',__file__,'--case','--backend',backend,
                          '--world-size',str(world),'--size-mib',str(size)],start_new_session=True,
                          env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                   OPENBLAS_NUM_THREADS='1',GLOO_SOCKET_IFNAME='lo'))
                    proc.wait(timeout=280)
                    if proc.returncode:
                        path.write_text(json.dumps(dict(status='FAILED',backend=backend,
                            world_size=world,size_MiB=size,returncode=proc.returncode)))
                    elif path.exists():
                        r=json.loads(path.read_text())
                        r['case_wall_s']=time.perf_counter()-start
                        path.write_text(json.dumps(r,indent=2))
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
                    path.write_text(json.dumps(dict(status='TIMEOUT',backend=backend,
                        world_size=world,size_MiB=size)))
                    raise RuntimeError('Case exceeded 280s; stop rather than run overlapping cases')
    print('ALL_DONE',flush=True)


if __name__=='__main__':
    main()
