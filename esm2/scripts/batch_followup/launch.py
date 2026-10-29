from pathlib import Path
import json,datetime,subprocess,sys,hashlib,random,time,os
import psutil
R=Path('/home/hyein/ai-benchmark/esm2');run='batch_inference_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');print(run,flush=True)
source=R/'data/uniprot_reviewed_900_950_20261028.fasta';rows=[]
for line in source.read_text().splitlines():
 if line.startswith('>'):rows.append({'id':line[1:].split()[0],'description':line[1:],'sequence':''})
 else:rows[-1]['sequence']+=line.strip()
seen=set();selected=[]
for r in rows:
 s=r['sequence'];assert 900<=len(s)<=950 and set(s)<=set('ACDEFGHIKLMNPQRSTVWYBXZOU');assert r['id'] not in {x['id'] for x in selected}
 if s not in seen:selected.append(r);seen.add(s)
random.Random(42).shuffle(selected);longest=max(range(len(selected)),key=lambda i:len(selected[i]['sequence']));selected.insert(0,selected.pop(longest));assert len(selected[0]['sequence'])==950
(D/'inputs.json').write_text(json.dumps(selected,indent=2));sources=json.loads((R/'data/sources.json').read_text());path=R/'models/esm2_t33_650M_UR50D.pt';sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==next(s['sha256'] for s in sources if s['path']=='models/esm2_t33_650M_UR50D.pt')
summary={'status':'running','run_id':run,'conditions':[],'failed':[],'checkpoint':'esm2_t33_650M_UR50D','checkpoint_sha256':sha,'source':json.loads(source.with_suffix('.source.json').read_text()),'input_audit':{'original_fasta_rows':len(rows),'unique_sequences':len(selected),'sequence_length_range':[min(len(r['sequence']) for r in selected),max(len(r['sequence']) for r in selected)],'sequence_1_length':950,'actual_padded_token_length_all_batches':952,'selection':'seed42 shuffle, longest sequence first to keep actual max length950 constant; nested unique prefixes, no within-batch copies','labels':None},'settings':{'task':'embedding inference only; no training','precision':'FP32','tf32':False,'seed':42,'guard_available_gib':16,'guard_swap_increase_gib':2,'minimum_repeats':2,'minimum_measured_seconds':15,'time_includes':'cached CPU tokens to GPU, full official forward including unused logits, mean pooling toCPU; excludes tokenization/load/warmup/filesave','framework_note':'native fair-esm2.0.0, existing host venv unchanged, not comparable to scGPT framework memory as model-only effect'},'largest_successful_batch':None}
def save():(D/'summary.json').write_text(json.dumps(summary,indent=2))
save();swap0=psutil.swap_memory().used
for batch in [1,2,4,8,16,32,64,96,128,160,192,224,256,288,320,384,448,500]:
 if batch>len(selected):summary['stop_reason']='available_unique_sequence_limit';break
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU compute job exists'
 available=psutil.virtual_memory().available/2**30
 if available<16 or psutil.swap_memory().used>swap0+2*2**30:summary['stop_reason']='system_memory_guard_before_launch';break
 if summary['conditions']:
  prior=summary['conditions'][-1];m=prior['memory'];b0=prior['batch_size'];base=prior['model_only_allocated_gib'];estimated=base+(m['peak_reserved_gib']-base)*(batch/b0)*1.25+3
  if estimated>available-16:summary['stop_reason']='predicted_reservation_exceeds_available_budget';summary['skipped_next_batch']={'batch':batch,'estimated_reserved_gib':estimated,'available_gib':available};break
 cmd=[str(R/'.venv/bin/python'),'-u',str(R/'scripts/batch_followup/worker.py'),str(D),str(batch)];(D/f'command_batch{batch}.json').write_text(json.dumps(cmd));start=time.time()
 with (D/f'batch{batch}.log').open('w') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 stopreason=None
 while p.poll() is None:
  if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:stopreason='parent_memory_guard';p.terminate();break
  time.sleep(.5)
 code=p.wait();(D/f'exit_batch{batch}.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start,'stop_reason':stopreason}));print('BATCH',batch,'EXIT',code,flush=True)
 if code:
  summary['failed'].append({'batch':batch,'exit_code':code,'reason':stopreason or 'worker_failure_see_log'});summary['stop_reason']='stopped_after_worker_failure';break
 s=json.loads((D/f'batch_{batch}/summary.json').read_text());assert s['status']=='complete';r=s['conditions'][0];r['model_only_allocated_gib']=s['model_only_postload']['allocated_gib'];summary['conditions'].append(r);summary['largest_successful_batch']=batch;save()
else:summary['stop_reason']='planned_batch_list_complete'
summary['status']='complete' if not summary['failed'] else 'stopped';summary['reached80']=any(r['allocated_fraction_shared_ram']>=.8 for r in summary['conditions']);summary['reached90']=any(r['allocated_fraction_shared_ram']>=.9 for r in summary['conditions']);save();subprocess.run([sys.executable,str(R/'scripts/batch_followup/report.py'),str(D)],check=True);(D/'exit.json').write_text(json.dumps({'exit_code':0,'scope':'controller finished; individual failures separately recorded'}));print('FINISHED',summary['stop_reason'],flush=True)
