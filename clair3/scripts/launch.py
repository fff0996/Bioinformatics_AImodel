import subprocess,datetime,json,sys
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/clair3');script=sys.argv[1] if len(sys.argv)>1 else 'benchmark.py';assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
if script=='benchmark.py':run='clair3_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LATEST').write_text(run+'\n')
else:run=(R/'results/LATEST').read_text().strip();D=R/'results'/run
cmd=['sg','docker','-c',f'docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v {R}:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID={run} -e PYTHONPATH=/work/scripts:/work/source -e MPLCONFIGDIR=/work/.cache/matplotlib local/clair3-spark:9d1dd3f /work/.venv/bin/python -u /work/scripts/{script}'];stem=script.removesuffix('.py');(D/(stem+'_command.json')).write_text(json.dumps(cmd,indent=2));print(run,script,flush=True)
with (D/(stem+'.log')).open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/(stem+'_exit.json')).write_text(json.dumps({'exit_code':code}));print('EXIT',code,flush=True)
