"""Run the supplied DDP/optimizer tests five times on CPU (no GPU contention)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'ddp_assignment_results'
OUT.mkdir(exist_ok=True)
results=[]
for repeat in range(1,6):
    start=time.perf_counter()
    cmd=[sys.executable,'-m','pytest','-q','tests/test_ddp_individual_parameters.py',
         'tests/test_ddp.py','tests/test_sharded_optimizer.py',
         f'--junitxml={OUT / f"pytest_run_{repeat}.xml"}']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',GLOO_SOCKET_IFNAME='lo',PYTHONPATH=str(ROOT))
    proc=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (OUT/f'pytest_run_{repeat}.log').write_text(proc.stdout)
    results.append(dict(repeat=repeat,returncode=proc.returncode,elapsed_s=time.perf_counter()-start,
                        summary=proc.stdout.splitlines()[-1] if proc.stdout else 'no output'))
    (OUT/'pytest_summary.json').write_text(json.dumps(results,indent=2))
    print(results[-1],flush=True)
    if proc.returncode:
        print(proc.stdout,flush=True)
        break
