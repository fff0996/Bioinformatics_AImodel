from pathlib import Path
import subprocess,json,time,shlex
R=Path('/home/hyein/ai-benchmark/scgpt');run=(R/'results/REALDATA_LATEST').read_text().strip();D=R/'results'/run
for batch in [256,224]:
 success=True
 for mode in ['train','test']:
  C=D/f'batch_{batch}';C.mkdir(exist_ok=True);done=C/f'{mode}_exit.json'
  if done.exists() and json.loads(done.read_text())['exit_code']==0:continue
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  out=f'/workspace/scgpt/results/{run}/batch_{batch}/'+mode;bundle=f'/workspace/scgpt/models/finetuned/{run}/batch_{batch}';args=['--mode','train','--n-cells','0','--epochs','1','--model-out',bundle] if mode=='train' else ['--mode','predict','--bundle',bundle+'/epoch_001','--input','/workspace/scgpt/data/pancreas_demo/demo_test.h5ad'];cmd=['docker','exec','--user','1000:1000','-e','SCGPT_GUARD_DIR='+f'/workspace/scgpt/results/{run}/batch_{batch}','-e','MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba','scgpt-spark','python','-u','/workspace/scgpt/scripts/full_realdata.py',*args,'--out',out,'--batch-size',str(batch),'--max-length','1200','--seed','42'];(C/f'{mode}_command.json').write_text(json.dumps(cmd,indent=2));start=time.time();print('START',batch,mode,flush=True)
  with (C/f'{mode}.log').open('w') as f:code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  done.write_text(json.dumps({'exit_code':code,'total_subprocess_wall_seconds':time.time()-start}));print('EXIT',batch,mode,code,flush=True)
  if code:success=False;break
 if success:(D/'comparison_cases.json').write_text(json.dumps([8,batch]));print('COMPLETED_CASES',8,batch,flush=True);break
 time.sleep(5)
