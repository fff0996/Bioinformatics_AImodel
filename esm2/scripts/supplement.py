"""Longer measurement of existing length inputs; fresh-process reload verification."""
import sys,os,json,pathlib,time,datetime,threading,subprocess,traceback,csv,hashlib,shutil
import torch,esm,numpy as np,psutil
R=pathlib.Path(__file__).resolve().parents[1]; OLD=R/'results'/'20261027_164300_125163'; O=pathlib.Path(sys.argv[1]); O.mkdir(parents=True,exist_ok=True)
shutil.copy2(__file__,O/'supplement.py')
def write(name,obj): (O/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
def log(*args): print(datetime.datetime.now().isoformat(),*args,flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[];stop=threading.Event();phase='setup';process=psutil.Process();initial_swap=psutil.swap_memory().used
summary={'status':'running','parent_run':str(OLD),'run_path':str(O),'sampling_seconds':.5,'repeats':40,'warmup_passes_per_condition':1,'conditions':[]}
write('summary.json',summary)
def monitor():
 with (O/'resources.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['query_start','query_end','phase','gpu_util_percent','rss_bytes','available_bytes','swap_bytes']);w.writeheader()
  while not stop.is_set():
   start=time.time();ph=phase
   try:
    q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3);value=float(q.stdout.strip()) if q.returncode==0 else None
   except Exception:value=None
   row=dict(query_start=start,query_end=time.time(),phase=ph,gpu_util_percent=value,rss_bytes=process.memory_info().rss,available_bytes=psutil.virtual_memory().available,swap_bytes=psutil.swap_memory().used); rows.append(row);w.writerow(row);f.flush();stop.wait(max(0,.5-(time.time()-start)))
def guard():
 if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>initial_swap+2*2**30:raise RuntimeError('Memory pressure; stop expansion')
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 # CPU inspection of every saved repeat: ids, shape, finiteness, repeat agreement.
 verified=[]
 for name in ['count_32','count_96','count_160','length_50_128','length_129_256','length_257_512','length_513_1022']:
  expected=json.loads((OLD/(name+'_inputs.json')).read_text());base=None
  for rep in range(3):
   data=torch.load(OLD/(name+'_r'+str(rep)+'.pt'),map_location='cpu',weights_only=True);assert len(data['ids'])==len(set(data['ids']))==len(expected);assert set(data['ids'])=={r['id'] for r in expected};assert data['embeddings'].shape==(len(expected),1280);assert torch.isfinite(data['embeddings']).all()
   if base is not None:assert data['ids']==base['ids'];assert torch.allclose(data['embeddings'],base['embeddings'],atol=1e-6,rtol=1e-5)
   else:base=data
  verified.append(dict(condition=name,all_three_outputs_valid=True))
 write('original_artifact_validation.json',verified)
 log('ORIGINAL ARTIFACTS VALID')
 source=json.loads((R/'data/sources.json').read_text());expectedhash=next(r['sha256'] for r in source if r['path']=='models/esm2_t33_650M_UR50D.pt');assert sha(R/'models/esm2_t33_650M_UR50D.pt')==expectedhash
 phase='model_loading';t=time.perf_counter();modeldata=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False);reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',modeldata,reg);del modeldata,reg;model=model.cuda().eval();torch.cuda.synchronize();summary['model_load_seconds']=time.perf_counter()-t
 summary['model_parameters']=sum(p.numel() for p in model.parameters());summary['encoder_parameters_excluding_lm_contact_heads']=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('lm_head.','contact_head.')))
 convert=alphabet.get_batch_converter(1022)
 def batch(recs):
  ds=esm.FastaBatchedDataset([r[0] for r in recs],[r[1] for r in recs]);return [[recs[i] for i in ix] for ix in ds.get_batch_indices(4096,extra_toks_per_seq=1)]
 def embed(b):
  _,ss,tokens=convert(b);tokens=tokens.cuda();out=model(tokens,repr_layers=[33],return_contacts=False)['representations'][33];return torch.stack([out[i,1:len(s)+1].mean(0) for i,s in enumerate(ss)]).cpu()
 for name in ['length_50_128','length_129_256','length_257_512','length_513_1022']:
  guard();src=json.loads((OLD/(name+'_inputs.json')).read_text());recs=[(r['id'],r['sequence']) for r in src];write(name+'_inputs.json',src)
  wall=time.perf_counter();phase=name+'_preprocessing';t=time.perf_counter();bs=batch(recs);preprocess=time.perf_counter()-t
  phase=name+'_warmup';t=time.perf_counter()
  with torch.no_grad():
   for b in bs:embed(b)
  torch.cuda.synchronize();warmup=time.perf_counter()-t
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();phase=name+'_execution';torch.cuda.synchronize();start=time.time();t=time.perf_counter();dur=[]
  with torch.no_grad():
   for rep in range(40):
    rt=time.perf_counter();values=[]
    for b in bs:values.append(embed(b))
    torch.cuda.synchronize();dur.append(time.perf_counter()-rt);guard()
  torch.cuda.synchronize();elapsed=time.perf_counter()-t;end=time.time();phase=name+'_save';t=time.perf_counter();out=torch.cat(values);assert torch.isfinite(out).all();torch.save(dict(ids=[i for b in bs for i,s in b],embeddings=out),O/(name+'_last_repeat.pt'));save=time.perf_counter()-t
  subset=[r for r in rows if r['query_start']>=start and r['query_end']<=end];steady=[r['gpu_util_percent'] for r in subset if r['query_start']>=start+2 and r['gpu_util_percent'] is not None];util=[r['gpu_util_percent'] for r in subset if r['gpu_util_percent'] is not None]
  result=dict(condition=name,unique_inputs=16,processed_inputs=640,repeats=40,warmup_processed_inputs_excluded=16,length_min=min(len(s) for i,s in recs),length_max=max(len(s) for i,s in recs),length_mean=float(np.mean([len(s) for i,s in recs])),preprocess_seconds=preprocess,warmup_seconds=warmup,execution_seconds=elapsed,repeat_seconds=dur,save_seconds=save,wall_seconds=time.perf_counter()-wall,start_unix=start,end_unix=end,throughput_per_second=640/elapsed,gpu_util_mean_percent=float(np.mean(util)) if util else None,gpu_util_max_percent=max(util) if util else None,gpu_util_steady_mean_percent=float(np.mean(steady)) if steady else None,sample_count=len(subset),cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,rss_sampled_peak_gib=max([r['rss_bytes'] for r in subset],default=0)/2**30,system_available_min_gib=min([r['available_bytes'] for r in subset],default=0)/2**30,swap_max_bytes=max([r['swap_bytes'] for r in subset],default=0))
  write(name+'_metrics.json',result);summary['conditions'].append(result);write('summary.json',summary);log('DONE',name,'seconds',round(elapsed,3),'GPU mean',result['gpu_util_mean_percent'])
 # Reconstruct fine-tuned model from original base in a new OS process.
 phase='fresh_reload';t=time.perf_counter();M=R/'models/finetuned/20261027_164300_125163';ck=torch.load(M/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['base_sha256']==expectedhash
 missing=model.load_state_dict(ck['backbone_updates'],strict=False);assert not missing.unexpected_keys;head=torch.nn.Linear(1280,1).cuda();head.load_state_dict(ck['head']);head.eval();model.eval();mu=ck['target_mean'];std=ck['target_std'];selected=json.loads((OLD/'training_inputs.json').read_text())['test'];rec=[(r['id'],r['sequence']) for r in selected];ids=[];pred=[]
 with torch.no_grad():
  for b in batch(rec):
   # embed() returns CPU for inference; transfer pooled 1280-vector for head.
   e=embed(b).cuda();p=(head(e).squeeze(-1)*std+mu).cpu().tolist();ids.extend(i for i,s in b);pred.extend(p)
 torch.cuda.synchronize();original=json.loads((OLD/'test_original.json').read_text());assert ids==original['ids'];delta=float(np.max(np.abs(np.asarray(pred)-original['predictions'])));assert delta<=1e-5
 write('fresh_reload_predictions.json',dict(ids=ids,predictions=pred));summary['fresh_reload']=dict(passed=True,independent_process=True,max_abs_difference=delta,tolerance=1e-5,test_inputs=len(pred),seconds=time.perf_counter()-t,base_hash_verified=True,training_resume_verified=False)
 summary['status']='success';summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write('summary.json',summary);log('SUCCESS',summary['fresh_reload'])
except Exception as e:
 summary['status']='failed';summary['error']=str(e);summary['traceback']=traceback.format_exc();write('summary.json',summary);log(traceback.format_exc());raise
finally:stop.set();thread.join(timeout=5)
