import subprocess,datetime,json,sys,shlex
from pathlib import Path
name=sys.argv[1];script=sys.argv[2] if len(sys.argv)>2 else 'benchmark.py';R=Path('/home/hyein/ai-benchmark')/name
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU job is running'
if script=='benchmark.py':
 run=name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LATEST').write_text(run+'\n')
else:run=(R/'results/LATEST').read_text().strip();D=R/'results'/run
cmd=['sg','docker','-c',f'docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v {R}:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID={run} -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/{script}'];stem=script.removesuffix('.py');(D/(stem+'_command.json')).write_text(json.dumps(cmd,indent=2));print(run,script,flush=True)
with (D/(stem+'.log')).open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/(stem+'_exit.json')).write_text(json.dumps({'exit_code':code,'finished':datetime.datetime.now().isoformat()}));print('EXIT',code,flush=True)
