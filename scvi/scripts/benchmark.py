"""scVI raw-count training and input-size GPU profiling, one GPU job at a time."""
import pathlib,sys,os,time,json,csv,datetime,threading,subprocess,random,hashlib,copy,traceback,gc,shutil,platform
import numpy as np,pandas as pd,torch,scanpy as sc,scvi,psutil
from lightning.pytorch.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
R=pathlib.Path('/work');O=R/'results'/sys.argv[1];O.mkdir(exist_ok=True);I=R/'results'/(R/'results/INPUT_LATEST').read_text().strip();shutil.copy2(__file__,O/'benchmark.py');shutil.copy2(I/'input_report.json',O/'input_report.json');shutil.copy2(I/'splits.json',O/'splits.json');shutil.copy2(I/'pip_freeze.txt',O/'pip_freeze.txt');shutil.copy2(R/'data/sources.json',O/'sources.json')
phase='setup';condition='setup';stop=threading.Event();samples=[];proc=psutil.Process();initialswap=psutil.swap_memory().used;allstart=time.perf_counter()
def write(name,obj):(O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def event(kind,**kw):
 row={'time_unix':time.time(),'kind':kind,'condition':condition,'phase':phase,**kw}
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
 print(json.dumps(row,ensure_ascii=False,default=str),flush=True)
def monitor():
 with (O/'resources.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['query_start','query_end','condition','phase','gpu_util_percent','rss_bytes','available_bytes','swap_bytes']);w.writeheader()
  while not stop.is_set():
   start=time.time();ph=phase;c=condition
   try:
    p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3);util=float(p.stdout.strip()) if p.returncode==0 else None
   except Exception:util=None
   row=dict(query_start=start,query_end=time.time(),condition=c,phase=ph,gpu_util_percent=util,rss_bytes=proc.memory_info().rss,available_bytes=psutil.virtual_memory().available,swap_bytes=psutil.swap_memory().used);samples.append(row);w.writerow(row);f.flush()
   if row['available_bytes']<8*2**30 or row['swap_bytes']>initialswap+2*2**30:event('memory_pressure_stop');os._exit(70)
   stop.wait(max(0,.5-(time.time()-start)))
def profile(c,start,end,ph=None):
 rows=[s for s in samples if s['condition']==c and start<=s['query_start'] and s['query_end']<=end and (ph is None or s['phase']==ph)];v=[s['gpu_util_percent'] for s in rows if s['gpu_util_percent'] is not None]
 return dict(gpu_util_mean_percent=float(np.mean(v)) if v else None,gpu_util_max_percent=max(v) if v else None,sample_count=len(rows),rss_sampled_peak_gib=max([s['rss_bytes'] for s in rows],default=0)/2**30,available_min_gib=min([s['available_bytes'] for s in rows],default=0)/2**30,swap_max_bytes=max([s['swap_bytes'] for s in rows],default=0))
def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def finitefloat(x):return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
thread=threading.Thread(target=monitor,daemon=True);thread.start();summary=dict(status='running',settings=dict(scvi_version=scvi.__version__,training='from scratch, not pretrained fine-tuning',batch_size=128,precision='32-true FP32; TF32 disabled',seed=42,epochs=400,validation_every_n_epochs=20,early_stopping=False,n_genes=2000,n_hidden=128,n_latent=10,n_layers=1,gene_likelihood='zinb',dispersion='gene',dropout_rate=.1,optimizer='Adam',lr=.001,weight_decay=1e-6,eps=.01,kl_warmup_epochs=400,num_workers=0,sampling_seconds=.5),training=[],inference=[]);write('summary.json',summary)
class Timing(Callback):
 def __init__(self,model):self.model=model;self.t0=None;self.train_seconds=0;self.val_seconds=0;self.epochs=[];self.gradient_names=[]
 def on_train_epoch_start(self,tr,pl):
  global phase
  torch.cuda.synchronize();phase='training';self.t0=time.perf_counter();self.begin=time.time()
 def close_train(self,tr):
  if self.t0 is not None:
   torch.cuda.synchronize();sec=time.perf_counter()-self.t0;self.train_seconds+=sec;self.epochs.append(dict(epoch=tr.current_epoch,start_unix=self.begin,end_unix=time.time(),seconds=sec));self.t0=None
 def on_validation_start(self,tr,pl):
  global phase
  self.close_train(tr);phase='validation';torch.cuda.synchronize();self.v0=time.perf_counter()
 def on_validation_end(self,tr,pl):
  global phase
  torch.cuda.synchronize();self.val_seconds+=time.perf_counter()-self.v0;phase='epoch_boundary'
 def on_train_epoch_end(self,tr,pl):
  global phase
  self.close_train(tr);phase='epoch_boundary'
  if tr.current_epoch in [0,19,99,199,299,399]:event('epoch',epoch=tr.current_epoch+1,global_step=tr.global_step,metrics={k:finitefloat(v) for k,v in tr.callback_metrics.items() if torch.is_tensor(v) and v.numel()==1 and torch.isfinite(v)})
 def on_after_backward(self,tr,pl):
  if not self.gradient_names:self.gradient_names=[n for n,p in self.model.module.named_parameters() if p.grad is not None]
try:
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;scvi.settings.seed=42;scvi.settings.batch_size=128;scvi.settings.dl_num_workers=0
 phase='data_loading';t=time.perf_counter();full=sc.read_h5ad(R/'data/processed_counts.h5ad');assert np.isfinite(full.layers['counts'].data).all() and np.all(np.asarray(full.layers['counts'].sum(1)).ravel()>0);splits=json.loads((I/'splits.json').read_text());summary['data_load_seconds']=time.perf_counter()-t
 write('environment.json',dict(python=sys.version,scvi=scvi.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),host=platform.node(),memory_bytes=psutil.virtual_memory().total,initial_available_bytes=psutil.virtual_memory().available,initial_swap_bytes=initialswap,nvidia_smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout))
 canonical=None;canonical_hash=None;paths=[]
 if len(sys.argv)>2:
  previous=R/'results'/sys.argv[2];oldsummary=json.loads((previous/'summary.json').read_text());canonical=torch.load(previous/'shared_initialization.pt',map_location='cpu',weights_only=True);canonical_hash=digest(canonical)
  for reused in oldsummary['training']:
   assert reused['training_success'] and reused['model_saved'] and pathlib.Path(reused['model_path'],'model.pt').exists() and reused['initialization_sha256']==canonical_hash
   reused.setdefault('source_run',sys.argv[2]);reused['reuse_reason']='Completed training and save reused; retry only remaining loading/inference stages';summary['training'].append(reused);paths.append(reused['model_path'])
  for p in previous.glob('train_*'):
   if p.is_file():shutil.copy2(p,O/p.name)
  torch.save(canonical,O/'shared_initialization.pt');write('recovery.json',dict(parent_run=sys.argv[2],reused_conditions=[r['condition'] for r in summary['training']]));event('reused_completed_training',source_run=sys.argv[2],models=paths)
 for key in ['train_1024','train_4096','train']:
  if any(r['condition']=='train_'+str(len(splits[key])) for r in summary['training']):continue
  train=np.array(splits[key],dtype=int);val=np.array(splits['validation'],dtype=int);test=np.array(splits['test'],dtype=int);N=len(train);condition='train_'+str(N);phase='preprocessing';twall=time.perf_counter();t=time.perf_counter();idx=np.concatenate([train,val,test]);a=full[idx].copy();scvi.model.SCVI.setup_anndata(a,layer='counts',batch_key='sample_id');preprocess=time.perf_counter()-t
  localtrain=np.arange(N);localval=np.arange(N,N+len(val));localtest=np.arange(N+len(val),len(idx));phase='initialization';scvi.settings.seed=42;t=time.perf_counter();model=scvi.model.SCVI(a,n_hidden=128,n_latent=10,n_layers=1,dropout_rate=.1,gene_likelihood='zinb',dispersion='gene')
  if canonical is None:canonical={k:v.detach().cpu().clone() for k,v in model.module.state_dict().items()};canonical_hash=digest(canonical);torch.save(canonical,O/'shared_initialization.pt')
  else:model.module.load_state_dict(canonical)
  assert digest(model.module.state_dict())==canonical_hash;model.to_device('cuda:0');torch.cuda.synchronize();initsec=time.perf_counter()-t;total=sum(p.numel() for p in model.module.parameters());trainable=sum(p.numel() for p in model.module.parameters() if p.requires_grad);encoder=sum(p.numel() for p in model.module.z_encoder.parameters())
  if key=='train_1024':
   phase='smoke';model.module.eval();bt=next(iter(model._make_data_loader(adata=a,indices=np.arange(32),batch_size=128)));bt={k:v.cuda() if torch.is_tensor(v) else v for k,v in bt.items()}
   with torch.no_grad():io,go,loss=model.module(bt);torch.cuda.synchronize();assert io['z'].shape==(32,10) and torch.isfinite(io['z']).all();write('smoke.json',dict(passed=True,input_shape=list(bt['X'].shape),latent_shape=list(io['z'].shape),device=str(io['z'].device),dtype=str(bt['X'].dtype)))
   del bt,io,go,loss
  scvi.settings.seed=42;cb=Timing(model);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();phase='fit';start=time.time();t=time.perf_counter()
  model.train(max_epochs=400,accelerator='gpu',devices=1,batch_size=128,early_stopping=False,datasplitter_kwargs={'external_indexing':[localtrain,localval,localtest],'num_workers':0},check_val_every_n_epoch=20,num_sanity_val_steps=0,precision='32-true',callbacks=[cb],enable_progress_bar=False,enable_model_summary=False,enable_checkpointing=False,log_every_n_steps=10,default_root_dir=str(O/condition))
  torch.cuda.synchronize();fitsec=time.perf_counter()-t;end=time.time();peakalloc=torch.cuda.max_memory_allocated()/2**30;peakreserved=torch.cuda.max_memory_reserved()/2**30;phase='evaluation';model.module.eval();assert set(model.train_indices)==set(localtrain) and set(model.validation_indices)==set(localval) and set(model.test_indices)==set(localtest)
  updated=[n for n,p in model.module.named_parameters() if not torch.equal(p.detach().cpu(),canonical[n])];assert updated;scvi.settings.seed=42;torch.cuda.synchronize();t=time.perf_counter();test_elbo=finitefloat(model.get_elbo(adata=a,indices=localtest,batch_size=128));val_elbo=finitefloat(model.get_elbo(adata=a,indices=localval,batch_size=128));testz=model.get_latent_representation(adata=a,indices=localtest,batch_size=128,give_mean=True);torch.cuda.synchronize();evaltime=time.perf_counter()-t;assert np.isfinite(testz).all() and testz.shape==(len(test),10)
  np.save(O/(condition+'_test_latent.npy'),testz);pd.DataFrame(testz,index=a.obs_names[localtest]).to_csv(O/(condition+'_test_latent.csv'));M=R/'models'/sys.argv[1]/condition;phase='saving';t=time.perf_counter();model.save(M,overwrite=False,save_anndata=True);model.trainer.save_checkpoint(M/'trainer.ckpt');torch.save(dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()),M/'rng.pt');save=time.perf_counter()-t
  for h,v in model.history.items():v.to_csv(O/(condition+'_'+h+'.csv'))
  write(condition+'_epoch_timings.json',cb.epochs);(M/'genes.csv').write_text('\n'.join(a.var_names)+'\n');a.obs.to_csv(M/'cell_metadata.csv')
  result=dict(condition=condition,unique_training_cells=N,processed_training_cells=N*400,validation_cells=len(val),test_cells=len(test),epochs=400,optimizer_steps=int(model.trainer.global_step),genes=2000,latent_dim=10,total_parameters=total,configured_trainable_parameters=trainable,encoder_parameters=encoder,observed_gradient_parameters=sum(p.numel() for n,p in model.module.named_parameters() if n in cb.gradient_names),updated_parameter_tensors=len(updated),initialization_sha256=canonical_hash,preprocess_seconds=preprocess,initialization_seconds=initsec,training_loop_seconds=cb.train_seconds,validation_loop_seconds=cb.val_seconds,fit_wall_seconds=fitsec,evaluation_seconds=evaltime,save_seconds=save,condition_wall_seconds=time.perf_counter()-twall,training_cells_per_second=N*400/cb.train_seconds,test_elbo=test_elbo,validation_elbo=val_elbo,elbo_definition='scVI 1.5.1 get_elbo returns ELBO; higher is better',cuda_peak_allocated_gib=peakalloc,cuda_peak_reserved_gib=peakreserved,fit_profile=profile(condition,start,end),training_phase_profile=profile(condition,start,end,'training'),validation_phase_profile=profile(condition,start,end,'validation'),model_path=str(M),training_success=True,model_saved=True,reload_verified=False,training_resume_verified=False)
  write(condition+'_metrics.json',result);(M/'benchmark_metadata.json').write_text(json.dumps(result,indent=2));summary['training'].append(result);paths.append(str(M));write('summary.json',summary);event('training_complete',**result);del model,a,testz,cb;gc.collect();torch.cuda.empty_cache()
 # All count inference conditions use one fixed saved model, never retrain between them.
 condition='primary_reload';phase='loading';t=time.perf_counter();primary=scvi.model.SCVI.load(paths[-1],accelerator='gpu',device=1);torch.cuda.synchronize();summary['primary_reload_seconds']=time.perf_counter()-t;check=primary.get_latent_representation(indices=np.flatnonzero(primary.adata.obs['split'].to_numpy()=='test'),batch_size=128,give_mean=True)
 # Reloaded local test order is train+validation+test. Stored trainer indices may not persist.
 expected=np.load(O/'train_10176_test_latent.npy');assert check.shape==expected.shape;delta=float(np.max(np.abs(check-expected)));assert delta<=1e-5;summary['primary_reload_max_abs_difference']=delta
 pool=np.arange(full.n_obs);strata=full.obs['sample_id'].astype(str)+'_q'+pd.qcut(full.obs['total_umi'].rank(method='first'),4,labels=False).astype(str);med,_=train_test_split(pool,train_size=4096,random_state=42,stratify=strata);small,_=train_test_split(med,train_size=1024,random_state=42,stratify=strata.iloc[med]);inference_indices=[np.sort(small),np.sort(med),pool]
 phase='warmup';t=time.perf_counter();warm=primary.get_latent_representation(adata=full,indices=small[:128],batch_size=128,give_mean=True);torch.cuda.synchronize();summary['inference_warmup_seconds']=time.perf_counter()-t
 for ix in inference_indices:
  N=len(ix);condition='inference_'+str(N);phase='inference';torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time();t=time.perf_counter();dur=[];repeats=0;first=None
  while time.perf_counter()-t<15 or repeats<3:
   rt=time.perf_counter();z=primary.get_latent_representation(adata=full,indices=ix,batch_size=128,give_mean=True);torch.cuda.synchronize();dur.append(time.perf_counter()-rt);assert z.shape==(N,10) and np.isfinite(z).all()
   if first is None:first=z.copy()
   else:assert np.allclose(z,first,atol=1e-6,rtol=1e-5)
   repeats+=1
  sec=time.perf_counter()-t;end=time.time();alloc=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30;phase='saving';st=time.perf_counter();np.save(O/(condition+'_latent.npy'),first);full.obs.iloc[ix].to_csv(O/(condition+'_cells.csv'));sav=time.perf_counter()-st
  row=dict(condition=condition,unique_inputs=N,processed_inputs=N*repeats,repeats=repeats,genes=2000,batch_size=128,input_tensor_shape='[<=128,2000] float32',latent_shape=list(z.shape),execution_seconds=sec,repeat_seconds=dur,save_seconds=sav,cells_per_second=N*repeats/sec,cuda_peak_allocated_gib=alloc,cuda_peak_reserved_gib=reserved,**profile(condition,start,end));write(condition+'_metrics.json',row);summary['inference'].append(row);write('summary.json',summary);event('inference_complete',**{k:v for k,v in row.items() if k!='repeat_seconds'})
 # Batch integration diagnostics on held-out cells; no cell-type labels are fabricated.
 condition='integration';phase='integration_evaluation';latent=primary.get_latent_representation(adata=full,batch_size=128,give_mean=True);np.save(O/'all_cells_latent.npy',latent);baseline=full.copy();sc.pp.normalize_total(baseline,target_sum=1e4);sc.pp.log1p(baseline);train=np.array(splits['train']);test=np.array(splits['test']);pca=PCA(n_components=10,random_state=42);pca.fit(baseline.X[train].toarray());pc=pca.transform(baseline.X.toarray());labels=full.obs['sample_id'].cat.codes.to_numpy();integration={}
 for name,x in [('log_normalized_PCA',pc),('SCVI_latent',latent)]:
  tdata=x[test];nn=NearestNeighbors(n_neighbors=31).fit(tdata);neighbors=nn.kneighbors(tdata,return_distance=False)[:,1:];p=labels[test][neighbors].mean(1);entropy=-(p*np.log(np.maximum(p,1e-12))+(1-p)*np.log(np.maximum(1-p,1e-12)))/np.log(2);clf=LogisticRegression(max_iter=1000,random_state=42).fit(x[train],labels[train]);integration[name]=dict(test_cells=len(test),batch_silhouette=float(silhouette_score(tdata,labels[test])),mean_normalized_batch_knn_entropy=float(entropy.mean()),batch_prediction_balanced_accuracy=float(balanced_accuracy_score(labels[test],clf.predict(tdata))),knn=30)
 normalized=primary.get_normalized_expression(adata=full,indices=test[:128],transform_batch=['pbmc4k','pbmc8k'],n_samples=1,library_size=1e4,return_numpy=True);assert normalized.shape==(128,2000) and np.isfinite(normalized).all();np.save(O/'batch_averaged_normalized_expression_example.npy',normalized);write('integration_metrics.json',dict(metrics=integration,cell_type_labels=None,limitations='Same donor technical batches; mixing alone cannot establish biological conservation; not scANVI cell classification',normalized_expression_example_shape=list(normalized.shape)));np.save(O/'baseline_pca.npy',pc)
 summary['integration']=integration;summary['status']='success';summary['model_paths']=paths;summary['wall_seconds']=time.perf_counter()-allstart;write('summary.json',summary);event('success',run=str(O))
except Exception as e:
 summary['status']='failed';summary['error']=str(e);summary['traceback']=traceback.format_exc();write('summary.json',summary);event('failed',error=str(e));traceback.print_exc();raise
finally:stop.set();thread.join(timeout=5)
