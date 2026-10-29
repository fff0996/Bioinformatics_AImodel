"""Explicitly pause ONLY this scGPT experiment, retaining completed measured prefixes."""
from pathlib import Path
import os,signal,json,subprocess,time,sys
R=Path('/home/hyein/ai-benchmark/scgpt');run=sys.argv[1] if len(sys.argv)>1 else (R/'results/LARGE_LATEST').read_text().strip();D=R/'results'/run
pids=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
 try:
  pid=int(line.strip());cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes();env=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
  if (b'large_scale/inference.py' in cmd or b'large_scale/resumable_inference.py' in cmd) and ('RUN_ID='+run).encode() in env:pids.append(pid)
 except (OSError,ValueError):pass
assert len(pids)==1,('expected one matching experiment',pids);pid=pids[0];os.kill(pid,signal.SIGSTOP)
try:
 s=json.loads((D/'summary.json').read_text());assert s['status']=='running';prefix={str(b):max([c['unique_cells'] for c in s['conditions'] if c['batch_size']==b],default=0) for b in [8,256]}
 for b,n in prefix.items():
  if n:
   with (D/f'embeddings_batch{b}.npy').open('r+b') as f:os.fsync(f.fileno())
 checkpoint={'source_run':run,'conditions':s['conditions'],'completed_prefixes':prefix,'saved_at':time.time(),'note':'Only completed measured milestones are durable resume points. Work after the latest milestone will be repeated. Download and token cache reused. Resume timing is a separate segment.'};(D/'pause_checkpoint.json').write_text(json.dumps(checkpoint,indent=2));s['status']='paused';s['pause_checkpoint']=checkpoint;(D/'summary.json').write_text(json.dumps(s,indent=2));os.kill(pid,signal.SIGTERM)
finally:
 try:os.kill(pid,signal.SIGCONT)
 except ProcessLookupError:pass
for _ in range(100):
 if not Path('/proc',str(pid)).exists():break
 time.sleep(.1)
assert not Path('/proc',str(pid)).exists(),'process has not exited';print(json.dumps(checkpoint,indent=2));print('PAUSED GPU process exited; use resume.py',run)
