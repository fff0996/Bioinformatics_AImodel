import subprocess,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/scgpt');run=(R/'results/SUPPLEMENT_LATEST').read_text().strip();D=R/'results'/run;assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();cmd=['sg','docker','-c',f'docker exec --user 1000:1000 -e MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib -e NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba scgpt-spark python -u /workspace/scgpt/scripts/train_util_supplement.py --out /workspace/scgpt/results/{run}/training_util --model-out /workspace/scgpt/models/finetuned/{run} --n-cells 1000 --epochs 1 --batch-size 8 --max-length 1200'];(D/'training_command.json').write_text(json.dumps(cmd,indent=2))
with (D/'training.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'training_exit.json').write_text(json.dumps({'exit_code':code}));print('TRAIN_EXIT',code,flush=True)
