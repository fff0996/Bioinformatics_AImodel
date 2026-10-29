import subprocess,os,datetime,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/cellpose');active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert not active,active
run='cellpose_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LATEST').write_text(run+'\n');args=['sg','docker','-c',f'docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v {R}:/work -e RUN_ID={run} -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba -e CELLPOSE_LOCAL_MODELS_PATH=/work/models nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/benchmark.py'];(D/'command.json').write_text(json.dumps(args,indent=2));print(run,flush=True)
with (D/'run.log').open('w') as f:code=subprocess.call(args,stdout=f,stderr=subprocess.STDOUT)
(D/'exit_code.json').write_text(json.dumps({'exit_code':code,'finished':datetime.datetime.now().isoformat()}));print('EXIT',code,flush=True)
