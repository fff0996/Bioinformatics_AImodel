"""Resume an explicitly paused run into a new folder without overwriting its results."""
from pathlib import Path
import json,sys,datetime,shutil,subprocess,shlex,time
R=Path('/home/hyein/ai-benchmark/scgpt');old=sys.argv[1] if len(sys.argv)>1 else (R/'results/LARGE_LATEST').read_text().strip();source=R/'results'/old;state=json.loads((source/'pause_checkpoint.json').read_text());assert json.loads((source/'summary.json').read_text())['status']=='paused';assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU compute job active'
run='actual_300k_resume_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir()
for b,n in state['completed_prefixes'].items():
 if n:shutil.copy2(source/f'embeddings_batch{b}.npy',D/f'embeddings_batch{b}.npy')
(D/'resume_state.json').write_text(json.dumps(state,indent=2));(R/'results/LARGE_LATEST').write_text(run+'\n');cmd=['docker','exec','--user','1000:1000','-e','MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba','-e','RUN_ID='+run,'scgpt-spark','python','-u','/workspace/scgpt/scripts/large_scale/resumable_inference.py'];(D/'inference_command.json').write_text(json.dumps(cmd,indent=2));print('RESUME',old,'->',run,state['completed_prefixes'],flush=True)
with (D/'inference.log').open('w') as f:code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
(D/'inference_exit.json').write_text(json.dumps({'exit_code':code,'ended_at':time.time()}));subprocess.run([sys.executable,str(R/'scripts/large_scale/report.py'),run],check=True);sys.exit(code)
