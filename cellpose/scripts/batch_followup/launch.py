from pathlib import Path
import os,json,time,datetime,subprocess,shlex,sys,hashlib
import psutil
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'cellpose';run='tile_batch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');print(run,flush=True);swap0=psutil.swap_memory().used
S={'status':'running','run_id':run,'conditions':[],'failed':[],'checkpoint':'cpsam_v2','checkpoint_sha256':hashlib.file_digest((R/'models/cpsam_v2').open('rb'),'sha256').hexdigest(),'settings':{'precision':'BF16 inference, CPU normalized data FP32','tile_size':256,'tile_overlap':.1,'seed':42,'tf32':False,'inference_only':True,'time_scope':'official core.run_net: CPU tiling/padding, transfer, forward, CPU output, tile blending; excludes normalization/load/warmup/mask dynamics/save','guard_available_gib':16,'guard_swap_increase_gib':2,'minimum_repeats':2,'minimum_measured_seconds':15,'fresh_container_each_batch':True},'largest_successful_batch':None}
def save():(D/'summary.json').write_text(json.dumps(S,indent=2))
def stage(name,script,batch=None):
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'other GPU compute job'
 container='cp-'+run.lower()+'-'+name;args=['docker','run','--rm','--name',container,'--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/shared:ro','-e','RUN_ID='+run,'-e','MPLCONFIGDIR=/work/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/work/.cache/numba','-e','CELLPOSE_LOCAL_MODELS_PATH=/work/models']
 if batch is not None:args+=['--gpus','all','-e','TILE_BATCH='+str(batch)]
 args+=['nvcr.io/nvidia/pytorch:26.06-py3','/work/.venv/bin/python','-u','/work/scripts/batch_followup/'+script];(D/(name+'_command.json')).write_text(json.dumps(args,indent=2));t=time.time()
 with (D/(name+'.log')).open('w') as f:
  p=subprocess.Popen(['sg','docker','-c',shlex.join(args)],stdout=f,stderr=subprocess.STDOUT)
  reason=None
  while p.poll() is None:
   if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',container])],stdout=f,stderr=subprocess.STDOUT);break
   time.sleep(.5)
  code=p.wait()
 (D/(name+'_exit.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-t,'stop_reason':reason}));print(name,'EXIT',code,flush=True);return code
save()
if stage('prepare','prepare.py')!=0:S['status']='failed_prepare';save();sys.exit(1)
audit=json.loads((D/'input_audit.json').read_text());save()
for batch in [8,12,24,48,96,192,384,576,768,960,1152,1536,1920,2400]:
 n=max(1,batch//12)
 if n>audit['unique_source_images']:S['stop_reason']='available_real_image_pool';break
 available=psutil.virtual_memory().available/2**30
 if S['conditions']:
  last=S['conditions'][-1];base=last['model_only_allocated_gib'];estimate=base+(last['memory']['peak_reserved_gib']-base)*batch/last['actual_max_tile_batch']*1.25+3
  if estimate>available-16:S['stop_reason']='predicted_reservation_exceeds_available_budget';S['skipped_next_batch']={'requested':batch,'estimated_reserved_gib':estimate,'available_gib':available};break
 code=stage('batch'+str(batch),'worker.py',batch)
 if code:S['failed'].append({'batch':batch,'exit_code':code});S['stop_reason']='worker_stopped_see_log';break
 worker=json.loads((D/f'batch_{batch}/summary.json').read_text());assert worker['status']=='complete';row=worker['condition'];row['model_only_allocated_gib']=worker['model_only_postload']['allocated_gib'];S['conditions'].append(row);S['largest_successful_batch']=row['actual_max_tile_batch'];save()
else:S['stop_reason']='planned_list_complete'
S['status']='complete' if not S['failed'] else 'stopped';S['reached80']=any(c['allocated_fraction_shared_ram']>=.8 for c in S['conditions']);S['reached90']=any(c['allocated_fraction_shared_ram']>=.9 for c in S['conditions']);save();subprocess.run([sys.executable,str(R/'scripts/batch_followup/report.py'),str(D)],check=True);(D/'exit.json').write_text(json.dumps({'exit_code':0,'scope':'controller finished; worker failures recorded separately'}));print('FINISHED',S.get('stop_reason'),flush=True)
