import subprocess,datetime,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/dnabert2');assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();run='supplement_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LATEST').write_text(run+'\n');cmd=['sg','docker','-c',f'docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/dnabert2-benchmark:/workspace:ro -v {R}:/work -e RUN_ID={run} -e HF_MODULES_CACHE=/work/.cache/modules -e HF_HOME=/work/.cache/huggingface nvcr.io/nvidia/pytorch:26.06-py3 /workspace/miniforge/envs/dnabert2/bin/python -u /work/scripts/supplement.py'];(D/'command.json').write_text(json.dumps(cmd,indent=2));print(run,flush=True)
with (D/'run.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'exit_code.json').write_text(json.dumps({'exit_code':code}));print('EXIT',code,flush=True)
