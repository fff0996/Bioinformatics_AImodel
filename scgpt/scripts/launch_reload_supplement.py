import subprocess,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/scgpt');run=(R/'results/SUPPLEMENT_LATEST').read_text().strip();D=R/'results'/run;assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();cmd=['sg','docker','-c',f'docker exec --user 1000:1000 -e RUN_ID={run} scgpt-spark python -u /workspace/scgpt/scripts/reload_supplement.py'];(D/'reload_command.json').write_text(json.dumps(cmd,indent=2))
with (D/'reload.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'reload_exit.json').write_text(json.dumps({'exit_code':code}));print('RELOAD_EXIT',code,flush=True)
