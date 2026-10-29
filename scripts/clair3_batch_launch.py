from pathlib import Path
import json,subprocess,shlex,os,sys,time
import psutil
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'clair3';run=(R/'results/BATCH_LATEST').read_text().strip();D=R/'results'/run;assert json.loads((D/'capture_exit.json').read_text())['exit_code']==0
S={'status':'running','run_id':run,'conditions':[],'failed':[],'settings':{'task':'native candidate model inference only','threads':4,'precision':'FP32','tf32':False,'guard_available_gib':16,'guard_swap_increase_gib':2,'baseline_calling_batch':1000,'note':'BAM candidate extraction completed once; actual cached candidate rows reused; no artificial repeated rows within batch'}};swap0=psutil.swap_memory().used
def save():(D/'summary.json').write_text(json.dumps(S,indent=2))
def stage(label,script,envextra=()):
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU job'
 name='clair3-'+label+'-'+run;cmd=['docker','run','--rm','--name',name,'--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','PYTHONPATH=/work/source:/work/scripts']
 for e in envextra:cmd+=['-e',e]
 cmd+=['local/clair3-spark:9d1dd3f','/work/.venv/bin/python','-u','/work/scripts/batch_followup/'+script];(D/(label+'_command.json')).write_text(json.dumps(cmd));t=time.time();reason=None
 with (D/(label+'.log')).open('w') as f:
  p=subprocess.Popen(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',name])],stdout=f,stderr=subprocess.STDOUT);break
   time.sleep(.5)
  code=p.wait()
 (D/(label+'_exit.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-t,'reason':reason}));print(label,code,flush=True);return code
save();assert stage('prepare','prepare_cache.py')==0;audit=json.loads((D/'input_audit.json').read_text());S['input_audit']=audit;save()
for kind in ['Clair3_P','Clair3_F']:
 pool=audit[kind]['unique_feature_rows'];last=None
 for B in sorted(set([b for b in [1,8,32,128,512,1000,2048,4096,8192,16384] if b<=pool]+[pool])):
  if last:
   base=last['model_only_allocated_gib'];estimate=base+(last['memory']['peak_reserved_gib']-base)*B/last['batch_size']*1.25+3
   if estimate>psutil.virtual_memory().available/2**30-16:S.setdefault('stop_reasons',{})[kind]='predicted_reservation_guard';break
  code=stage(kind.lower()+'-'+str(B),'worker.py',['CANDIDATE_MODEL='+kind,'BATCH='+str(B)])
  if code:S['failed'].append({'model':kind,'batch':B,'exit_code':code});break
  w=json.loads((D/f'{kind}_batch_{B}/summary.json').read_text());last=w['condition'];last['model']=kind;last['model_only_allocated_gib']=w['model_only_postload']['allocated_gib'];S['conditions'].append(last);save()
 else:S.setdefault('stop_reasons',{})[kind]='available_real_candidate_cache_limit'
S['status']='complete' if not S['failed'] else 'stopped';save();subprocess.run([sys.executable,str(ROOT/'scripts/two_batch_report.py'),'clair3',str(D)],check=True);print('CLAIR3_FINISHED',S['status'],flush=True)
