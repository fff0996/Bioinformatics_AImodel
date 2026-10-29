from pathlib import Path
import subprocess,time,json,datetime,shlex,os
R=Path('/home/hyein/ai-benchmark/scgpt');run='actual_300k_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LARGE_LATEST').write_text(run+'\n');print(run,flush=True)
def active(script):
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if any(a.decode(errors='ignore').endswith(script) for a in args):return True
 return False
while True:
 p=R/'data_external/tabula_sapiens/download_source.json'
 try:meta=json.loads(p.read_text());ready=len(meta.get('sha256',''))==64
 except Exception:ready=False
 if ready:break
 if not active('large_scale/download.py'):raise RuntimeError('download stopped before completion')
 time.sleep(10)
while active('scripts/resume_full_realdata.py'):time.sleep(10)
def runstage(name,args):
 cmd=['docker','exec','--user','1000:1000','-e','MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba',*args];(D/(name+'_command.json')).write_text(json.dumps(cmd,indent=2));start=time.time()
 with (D/(name+'.log')).open('w') as f:code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 (D/(name+'_exit.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start}));print(name,'EXIT',code,flush=True)
 if code:raise RuntimeError((name,code))
runstage('prepare',['-e','CUDA_VISIBLE_DEVICES=','scgpt-spark','python','-u','/workspace/scgpt/scripts/large_scale/prepare.py'])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU job is running'
runstage('inference',['-e','RUN_ID='+run,'scgpt-spark','python','-u','/workspace/scgpt/scripts/large_scale/inference.py'])
(D/'exit_code.json').write_text(json.dumps({'exit_code':0}));print('REAL300K_COMPLETE',flush=True)
