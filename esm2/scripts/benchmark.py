"""Sequential ESM-2 benchmark; no changes to shared base environment."""
import os,sys,time,json,pathlib,datetime,random,hashlib,threading,subprocess,csv,pickle,traceback,platform,shutil
import numpy as np,torch,esm,psutil,lmdb
from scipy.stats import spearmanr
R=pathlib.Path(__file__).resolve().parents[1]
RUN=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'); O=R/'results'/RUN; O.mkdir(parents=True)
(R/'results'/'LATEST').write_text(RUN+'\n')
shutil.copy2(__file__,O/'benchmark.py'); shutil.copy2(R/'data/sources.json',O/'sources.json')
def write(name,obj): (O/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
def log(*s): print(datetime.datetime.now().isoformat(),*s,flush=True)
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
phase='setup'; stop=threading.Event(); samples=[]; proc=psutil.Process()
def monitor():
 with (O/'resources.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['time_unix','phase','gpu_util_percent','rss_bytes','system_available_bytes','swap_used_bytes']); w.writeheader()
  while not stop.is_set():
   start=time.time(); ph=phase
   try:
    q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
    util=float(q.stdout.strip().splitlines()[0]) if q.returncode==0 and q.stdout.strip().splitlines()[0].strip().isdigit() else None
   except Exception: util=None
   vm=psutil.virtual_memory(); row=dict(time_unix=start,phase=ph,gpu_util_percent=util,rss_bytes=proc.memory_info().rss,system_available_bytes=vm.available,swap_used_bytes=psutil.swap_memory().used)
   samples.append(row); w.writerow(row);f.flush(); stop.wait(max(0,.5-(time.time()-start)))
th=threading.Thread(target=monitor,daemon=True);th.start()
def stats(ph):
 rows=[s for s in samples if s['phase']==ph]; u=[s['gpu_util_percent'] for s in rows if s['gpu_util_percent'] is not None]
 return dict(sample_count=len(rows),gpu_util_mean_percent=float(np.mean(u)) if u else None,gpu_util_max_percent=max(u) if u else None,rss_sampled_peak_gib=max([s['rss_bytes'] for s in rows],default=0)/2**30,system_available_min_gib=min([s['system_available_bytes'] for s in rows],default=0)/2**30,swap_max_bytes=max([s['swap_used_bytes'] for s in rows],default=0))
def guard():
 if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>initial_swap+2*2**30: raise RuntimeError('Memory pressure: halt input expansion')
initial_swap=psutil.swap_memory().used
summary={'run_id':RUN,'status':'running','settings':{'checkpoint':'esm2_t33_650M_UR50D','precision':'FP32, TF32 disabled','seed':42,'token_budget':4096,'batch_policy':'official FastaBatchedDataset.get_batch_indices(extra_toks_per_seq=1)','length_limit':1022,'sampling_seconds':.5,'inference_repeats':3,'threads':8},'conditions':[]}
write('summary.json',summary)
try:
 # Inspect actual files before applying the model.
 sequences=[]; ids=[]
 for line in (R/'data/uniprot_reviewed.fasta').read_text().splitlines():
  if line.startswith('>'): ids.append(line[1:].split()[0]);sequences.append('')
  else: sequences[-1]+=line.strip()
 seen=set(); records=[]
 for i,s in zip(ids,sequences):
  if s not in seen: records.append((i,s));seen.add(s)
 bins=[(50,128),(129,256),(257,512),(513,1022)]; groups=[]
 for a,b in bins:
  g=[r for r in records if a<=len(r[1])<=b];random.Random(SEED).shuffle(g);groups.append(g)
 nbin=min(40,min(map(len,groups))); countset=[g[j] for j in range(nbin) for g in groups]
 ds={}; inputs={'uniprot':{'rows':len(sequences),'unique_sequences':len(records),'duplicate_sequences_removed':len(sequences)-len(records),'length_min':min(map(len,sequences)),'length_max':max(map(len,sequences)),'characters':sorted(set(''.join(sequences))),'labels':None,'dtype':'FASTA text; tokenized torch.int64','preprocessing':'deduplicate exact sequences, no truncation necessary','bins':[{ 'range':[a,b],'available_unique':len(g)} for (a,b),g in zip(bins,groups)]},'fluorescence':{}}
 for split in ['train','valid','test']:
  env=lmdb.open(str(R/'data/fluorescence'/('fluorescence_'+split+'.lmdb')),readonly=True,lock=False)
  with env.begin() as tx:
   n=pickle.loads(tx.get(b'num_examples')); allrows=[pickle.loads(tx.get(str(i).encode())) for i in range(n)]
  env.close(); assert all(np.isfinite(r['log_fluorescence']).all() and len(r['primary'])==r['protein_length'] for r in allrows)
  assert all(set(r['primary'])<=set('ACDEFGHIKLMNPQRSTVWY') for r in allrows)
  ds[split]=allrows; inputs['fluorescence'][split]=dict(rows=n,unique=len({r['primary'] for r in allrows}),length_min=min(len(r['primary']) for r in allrows),length_max=max(len(r['primary']) for r in allrows),label_dtype=str(allrows[0]['log_fluorescence'].dtype),label_shape=list(allrows[0]['log_fluorescence'].shape),label_mean=float(np.mean([r['log_fluorescence'][0] for r in allrows])),label_min=float(min(r['log_fluorescence'][0] for r in allrows)),label_max=float(max(r['log_fluorescence'][0] for r in allrows)),missing_or_nonfinite_labels=0)
 for a,b in [('train','valid'),('train','test'),('valid','test')]:
  overlap=len({r['primary'] for r in ds[a]}&{r['primary'] for r in ds[b]}); assert overlap==0;inputs['fluorescence'][a+'_'+b+'_exact_overlap']=overlap
 write('input_report.json',inputs);log('INPUT VERIFIED',inputs['uniprot'])
 write('environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,esm='fair-esm 2.0.0',host=platform.node(),platform=platform.platform(),gpu=torch.cuda.get_device_name(0),cuda_available=torch.cuda.is_available(),system_memory_bytes=psutil.virtual_memory().total,initial_available_bytes=psutil.virtual_memory().available,initial_swap_bytes=initial_swap,gpu_processes=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,docker='Permission denied for current user; host isolated venv used'))
 (O/'pip_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout)
 phase='model_loading'; t=time.perf_counter()
 # Official FAIR checkpoint is a trusted pickle from the recorded HTTPS source;
 # explicit weights_only=False supports its argparse.Namespace cfg on torch>=2.6.
 model_data=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False)
 reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True)
 model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',model_data,reg);del model_data,reg
 model=model.cuda().eval();torch.cuda.synchronize();summary['model_load_seconds']=time.perf_counter()-t
 summary['model_parameters']=sum(p.numel() for p in model.parameters());summary['embedding_dimension']=model.embed_dim;summary['layers']=model.num_layers
 converter=alphabet.get_batch_converter(1022)
 def batches(recs):
  data=esm.FastaBatchedDataset([r[0] for r in recs],[r[1] for r in recs]);return [[recs[i] for i in ix] for ix in data.get_batch_indices(4096,extra_toks_per_seq=1)]
 def embed(recs):
  _,strings,tok=converter(recs); tok=tok.cuda(); out=model(tok,repr_layers=[33],return_contacts=False)['representations'][33]
  return torch.stack([out[i,1:len(s)+1].mean(0) for i,s in enumerate(strings)]),tok.shape
 phase='warmup';t=time.perf_counter()
 with torch.no_grad():
  smoke,shape=embed(countset[:2]);torch.cuda.synchronize();assert smoke.shape==(2,1280) and torch.isfinite(smoke).all(); torch.save({'ids':[r[0] for r in countset[:2]],'embeddings':smoke.cpu()},O/'smoke.pt')
 summary['warmup_seconds']=time.perf_counter()-t;write('smoke.json',{'passed':True,'output_shape':list(smoke.shape),'token_shape':list(shape),'device':str(smoke.device),'dtype':str(smoke.dtype)});del smoke
 log('SMOKE PASSED; parameters',summary['model_parameters'])
 def inference(name,recs):
  global phase
  guard();phase=name;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();twall=time.perf_counter();t=time.perf_counter(); bs=batches(recs);prep=time.perf_counter()-t
  write(name+'_inputs.json',[dict(id=i,sequence=s,original_residues=len(s),actual_residues=min(1022,len(s)),truncated=len(s)>1022) for i,s in recs])
  durations=[];bmeta=[];save=0
  with torch.no_grad():
   for repeat in range(3):
    torch.cuda.synchronize();t=time.perf_counter();em=[];order=[]
    for b in bs:
     e,shape=embed(b);em.append(e.cpu());order.extend(r[0] for r in b)
     if repeat==0:bmeta.append(dict(samples=len(b),token_shape=list(shape),nonpadding_tokens=sum(len(r[1])+2 for r in b),padded_tokens=shape[0]*shape[1],residue_lengths=[len(r[1]) for r in b]))
    torch.cuda.synchronize();durations.append(time.perf_counter()-t);t=time.perf_counter();values=torch.cat(em);assert torch.isfinite(values).all();torch.save(dict(ids=order,embeddings=values),O/(name+'_r'+str(repeat)+'.pt'));save+=time.perf_counter()-t
    guard()
  phase='between_conditions';time.sleep(.6)
  row=dict(condition=name,mode='inference',unique_inputs=len(recs),processed_inputs=3*len(recs),repeats=3,length_min=min(len(r[1]) for r in recs),length_max=max(len(r[1]) for r in recs),length_mean=float(np.mean([len(r[1]) for r in recs])),preprocess_seconds=prep,execution_seconds=sum(durations),repeat_seconds=durations,save_seconds=save,wall_seconds=time.perf_counter()-twall,throughput_per_second=3*len(recs)/sum(durations),cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**stats(name))
  write(name+'_batches.json',bmeta);write(name+'_metrics.json',row);summary['conditions'].append(row);write('summary.json',summary);log('DONE',name,row)
 for n in [32,96,160]: inference('count_'+str(n),countset[:n])
 for (a,b),g in zip(bins,groups): inference('length_'+str(a)+'_'+str(b),g[:16])
 # Genuine, limited backbone fine-tuning: final transformer block + final norm + regression head.
 selections={}
 for split,n in [('train',128),('valid',64),('test',128)]:
  ix=random.Random(SEED).sample(range(len(ds[split])),n);selections[split]=[dict(id=split+'_'+str(i),sequence=ds[split][i]['primary'],label=float(ds[split][i]['log_fluorescence'][0]),source_index=i,num_mutations=int(ds[split][i]['num_mutations'])) for i in ix]
 write('training_inputs.json',selections)
 for p in model.parameters():p.requires_grad_(False)
 for p in model.layers[-1].parameters():p.requires_grad_(True)
 for p in model.emb_layer_norm_after.parameters():p.requires_grad_(True)
 head=torch.nn.Linear(1280,1).cuda(); trainparams=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW([dict(params=trainparams,lr=1e-4),dict(params=head.parameters(),lr=1e-3)],weight_decay=.01)
 mu=float(np.mean([r['label'] for r in selections['train']]));std=float(np.std([r['label'] for r in selections['train']]));assert std>0
 def evaluate(split):
  model.eval();head.eval();preds=[];truth=[];idx=[]
  records=[(r['id'],r['sequence']) for r in selections[split]];lookup={r['id']:r['label'] for r in selections[split]}
  with torch.no_grad():
   for b in batches(records):
    e,_=embed(b);p=head(e).squeeze(-1).cpu().numpy()*std+mu;preds.extend(map(float,p));truth.extend(lookup[i] for i,s in b);idx.extend(i for i,s in b)
  pred=np.asarray(preds);truth=np.asarray(truth);metrics=dict(n=len(pred),mse=float(np.mean((pred-truth)**2)),mae=float(np.mean(np.abs(pred-truth))),spearman=float(spearmanr(pred,truth).statistic));return dict(ids=idx,predictions=preds,targets=truth.tolist(),metrics=metrics)
 phase='baseline_validation';baseline=evaluate('valid');write('baseline_validation.json',baseline)
 before=model.layers[-1].fc1.weight.detach().clone();head_before=head.weight.detach().clone()
 phase='finetuning';torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();losses=[]
 model.train();head.train();records=[(r['id'],r['sequence']) for r in selections['train']];lookup={r['id']:r['label'] for r in selections['train']};bs=batches(records);random.Random(SEED).shuffle(bs)
 for step,b in enumerate(bs):
  guard();opt.zero_grad(set_to_none=True);e,_=embed(b);target=torch.tensor([(lookup[i]-mu)/std for i,s in b],device='cuda',dtype=torch.float32);loss=torch.nn.functional.mse_loss(head(e).squeeze(-1),target);assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()));log('TRAIN',step+1,'/',len(bs),'loss',losses[-1])
 torch.cuda.synchronize();trainsec=time.perf_counter()-t
 changed=not torch.equal(before,model.layers[-1].fc1.weight);headchanged=not torch.equal(head_before,head.weight);assert changed and headchanged;del before,head_before
 trainmetric=dict(mode='partial_backbone_finetuning',epochs=1,train_inputs=128,validation_inputs=64,test_inputs=128,trainable_backbone_parameters=sum(p.numel() for p in trainparams),head_parameters=sum(p.numel() for p in head.parameters()),trainable_total=sum(p.numel() for p in trainparams)+sum(p.numel() for p in head.parameters()),complete_model_parameters=summary['model_parameters']+sum(p.numel() for p in head.parameters()),execution_seconds=trainsec,throughput_per_second=128/trainsec,cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,backbone_updated=changed,head_updated=headchanged,losses=losses)
 phase='validation';time.sleep(.6);trainmetric.update(stats('finetuning'));validation=evaluate('valid');write('validation.json',validation)
 phase='saving_model';t=time.perf_counter();M=R/'models'/'finetuned'/RUN;M.mkdir(parents=True)
 adapter={k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith('layers.32.') or k.startswith('emb_layer_norm_after.')}
 checkpoint=dict(backbone_updates=adapter,head=head.state_dict(),target_mean=mu,target_std=std,base_checkpoint='esm2_t33_650M_UR50D.pt',base_sha256=json.loads((R/'data/sources.json').read_text())[2]['sha256'],epoch=1,seed=42,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),training_scope='final transformer block + final layer norm + linear regression head',resume_training_verified=False)
 torch.save(checkpoint,M/'checkpoint.pt');trainmetric['save_seconds']=time.perf_counter()-t;trainmetric['model_path']=str(M)
 (M/'metadata.json').write_text(json.dumps({k:v for k,v in checkpoint.items() if k not in ['backbone_updates','head','optimizer','torch_rng','cuda_rng','python_rng']},indent=2))
 phase='test_original';original=evaluate('test');write('test_original.json',original)
 # Perturb updated weights before reloading, proving the stored state is used.
 with torch.no_grad():model.layers[-1].fc1.weight.zero_();head.weight.zero_()
 reloaded=torch.load(M/'checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(reloaded['backbone_updates'],strict=False);head.load_state_dict(reloaded['head']);mu=reloaded['target_mean'];std=reloaded['target_std']
 phase='test_reloaded';retest=evaluate('test');write('test_reloaded.json',retest)
 delta=float(np.max(np.abs(np.asarray(original['predictions'])-np.asarray(retest['predictions']))));assert delta<=1e-5
 trainmetric.update(reload_verified=True,reload_max_abs_difference=delta,independent_test_evaluated=True,test_metrics=retest['metrics'],validation_metrics=validation['metrics'],initial_validation_metrics=baseline['metrics'])
 write('training_metrics.json',trainmetric);summary['training']=trainmetric;summary['status']='success';summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write('summary.json',summary);log('SUCCESS',O)
except Exception as exc:
 summary['status']='failed';summary['error']=str(exc);summary['traceback']=traceback.format_exc();write('summary.json',summary);log(traceback.format_exc());raise
finally:
 stop.set();th.join(timeout=5)
