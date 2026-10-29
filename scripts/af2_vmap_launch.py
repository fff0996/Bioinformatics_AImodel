from pathlib import Path
import os,json,subprocess,time,shlex,sys
import psutil
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'colabfold';run=(R/'results/BATCH_LATEST').read_text().strip();D=R/'results'/run;assert json.loads((D/'capture_exit.json').read_text())['exit_code']==0
clair=ROOT/'clair3/results'/(ROOT/'clair3/results/BATCH_LATEST').read_text().strip()
while json.loads((clair/'summary.json').read_text())['status']=='running':time.sleep(10)
S={'status':'running','run_id':run,'workers':[],'failed':[],'scope':'experimental JAX vmap independent monomer batch axis on existing ColabFold model_1, no simultaneous GPU jobs','settings':{'guard_available_gib':16,'guard_swap_increase_gib':2}};swap0=psutil.swap_memory().used
def save():(D/'summary.json').write_text(json.dumps(S,indent=2))
save();print('AF2_BATCH_START',run,flush=True)
for batch in [1,2,4]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'other GPU work'
 if psutil.virtual_memory().available<16*2**30:S['stop_reason']='available_memory_guard';break
 name='af2-vmap-'+run+'-'+str(batch);cmd=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--ipc=host','--user','1000:1000','-e','RUN_ID='+run,'-e','BATCH='+str(batch),'-e','JAX_PLATFORMS=cuda,cpu','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','-e','MPLCONFIGDIR=/work/results/'+run+'/mplcache','-v',str(R)+':/work','-v',str(R/'cache/colabfold')+':/cache:ro','ghcr.io/sokrypton/colabfold:1.6.2-cuda13','python','-u','/work/scripts/batch_followup/worker.py'];(D/f'batch{batch}_command.json').write_text(json.dumps(cmd,indent=2));start=time.time();reason=None
 with (D/f'batch{batch}.log').open('w') as f:
  p=subprocess.Popen(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',name])],stdout=f,stderr=subprocess.STDOUT);break
   time.sleep(.5)
  code=p.wait()
 (D/f'batch{batch}_exit.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start,'reason':reason}));print('AF2_BATCH_EXIT',batch,code,flush=True)
 if (D/f'batch_{batch}/summary.json').exists():S['workers'].append(json.loads((D/f'batch_{batch}/summary.json').read_text()))
 if code:S['failed'].append({'batch':batch,'exit_code':code});S['stop_reason']='worker_error_or_guard_see_log';break
 save()
else:S['stop_reason']='actual_four_query_pool_complete'
S['status']='complete' if not S['failed'] and len(S['workers'])==3 else 'stopped';save();subprocess.run([sys.executable,str(ROOT/'scripts/two_batch_report.py'),'af2',str(D)],check=True);print('AF2_FINISHED',S['status'],flush=True)
