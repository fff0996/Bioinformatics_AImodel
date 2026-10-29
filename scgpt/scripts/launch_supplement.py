import subprocess,datetime,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/scgpt');assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();run='supplement_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/SUPPLEMENT_LATEST').write_text(run+'\n');cmd=['sg','docker','-c',f'docker exec --user 1000:1000 -e RUN_ID={run} -e MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib -e NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba scgpt-spark python -u /workspace/scgpt/scripts/supplement.py'];(D/'command.json').write_text(json.dumps(cmd,indent=2));print(run,flush=True)
with (D/'run.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'exit_code.json').write_text(json.dumps({'exit_code':code}));print('EXIT',code,flush=True)
