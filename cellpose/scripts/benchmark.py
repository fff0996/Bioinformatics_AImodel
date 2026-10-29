import os,time,json,csv,threading,subprocess,logging,hashlib,random,sys,gc,platform,importlib.metadata
from pathlib import Path
import numpy as np, tifffile, psutil, torch
from cellpose import models,core,train,metrics,dynamics
R=Path('/work');runid=os.environ['RUN_ID'];D=R/'results'/runid;D.mkdir(exist_ok=True);M=R/'models'/runid;M.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(message)s',handlers=[logging.StreamHandler()])
torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
phase='initialization';condition='';stop=threading.Event();events=[];samples=[];G=1024**3
S={'run_id':runid,'seed':42,'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cellpose':importlib.metadata.version('cellpose'),'gpu':torch.cuda.get_device_name(0),'platform':platform.platform(),'tf32':torch.backends.cuda.matmul.allow_tf32},'conditions':[]}
def dump():
 (D/'summary.json').write_text(json.dumps(S,indent=2,default=str));(D/'events.json').write_text(json.dumps(events,indent=2,default=str))
def mem():
 v=psutil.virtual_memory();return {'allocated_gib':torch.cuda.memory_allocated()/G,'reserved_gib':torch.cuda.memory_reserved()/G,'peak_allocated_gib':torch.cuda.max_memory_allocated()/G,'peak_reserved_gib':torch.cuda.max_memory_reserved()/G,'rss_gib':psutil.Process().memory_info().rss/G,'system_available_gib':v.available/G,'system_total_gib':v.total/G,'swap_used_gib':psutil.swap_memory().used/G}
def event(event_name,**kw):
 e={'time_unix':time.time(),'phase':phase,'condition':condition,'event':event_name,**kw};events.append(e);print('EVENT',json.dumps(e,default=str),flush=True)
def monitor():
 with (D/'resources.csv').open('w',newline='') as f:
  cols=['query_start','query_end','phase','condition','gpu_util_percent','rss_gib','available_gib','swap_gib'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  while not stop.is_set():
   t=time.time();p=phase;c=condition
   try:
    q=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).strip().splitlines()[0];u=float(q)
   except Exception:u=None
   v=psutil.virtual_memory();row=dict(query_start=t,query_end=time.time(),phase=p,condition=c,gpu_util_percent=u,rss_gib=psutil.Process().memory_info().rss/G,available_gib=v.available/G,swap_gib=psutil.swap_memory().used/G);samples.append(row);w.writerow(row);f.flush()
   if v.available<8*G or psutil.swap_memory().used>initialswap+2*G:
    (D/'memory_guard.json').write_text(json.dumps(row));os._exit(86)
   stop.wait(.5)
def stats(t0,t1):
 rows=[r for r in samples if r['query_start']>=t0 and r['query_end']<=t1];us=[r['gpu_util_percent'] for r in rows if r['gpu_util_percent'] is not None]
 return {'sample_count':len(rows),'gpu_util_avg':float(np.mean(us)) if us else None,'gpu_util_max':max(us) if us else None,'rss_sampled_peak_gib':max([r['rss_gib'] for r in rows],default=None),'available_sampled_min_gib':min([r['available_gib'] for r in rows],default=None)}
initialswap=psutil.swap_memory().used;th=threading.Thread(target=monitor,daemon=True);th.start()
audit=json.loads((R/'data/input_audit.json').read_text());data={s:[tifffile.imread(R/r['image']) for r in rows] for s,rows in audit['selected'].items()};gt={s:[np.load(R/r['mask']) for r in rows] for s,rows in audit['selected'].items()};S['input_audit']='data/input_audit.json';S['settings']={'inference_dtype':'bfloat16','tile_batch_size':8,'tile_size':256,'tile_overlap':.1,'augment':False,'normalize':'official default 1st/99th percentile; inference1channel, training grayscale+two zero channels','diameter':None,'rescale':None,'flow_threshold':.4,'cellprob_threshold':0.,'min_size':15,'max_size_fraction':.4,'training_batch':1,'training_epochs':100,'learning_rate':1e-5,'weight_decay':.1,'training_dtype':'float32 (official automatic conversion)','training_augmentation':'official random rotate/resize, scale_range=.5, rescale=False, 256 crop','seed_note':'global42; official trainer resets numpy seed to epoch index and validation42'}
forwardrows=[];origforward=core._forward
def forward(net,x):
 torch.cuda.synchronize();t=time.time();y=origforward(net,x);torch.cuda.synchronize();forwardrows.append({'start':t,'end':time.time(),'phase':phase,'condition':condition,'shape':list(x.shape),'dtype':str(net.dtype)});return y
core._forward=forward
opts=dict(batch_size=8,normalize=True,diameter=None,rescale=None,flow_threshold=.4,cellprob_threshold=0.,min_size=15,max_size_fraction=.4,augment=False,tile_overlap=.1,bsize=256,do_3D=False)
def evaluate(model,imgs):return model.eval(imgs,**opts)
def infer_condition(model,name,imgs,minseconds=15):
 global phase,condition
 phase='inference';condition=name;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=mem();t=time.time();idx=len(forwardrows);repeats=0;first=None
 while repeats==0 or time.time()-t<minseconds:
  out=evaluate(model,imgs)
  if first is None:first=out
  repeats+=1
 torch.cuda.synchronize();end=time.time();fs=forwardrows[idx:];row={'name':name,'unique_images':len(imgs),'raw_shapes':[list(x.shape) for x in imgs],'repeats':repeats,'processed_images':len(imgs)*repeats,'seconds':end-t,'images_per_second':len(imgs)*repeats/(end-t),'forward_including_transfers_seconds':sum(f['end']-f['start'] for f in fs),'tile_forward_calls':len(fs),'actual_forward_shapes':sorted({str(f['shape']) for f in fs}),'processed_tiles':sum(f['shape'][0] for f in fs),'baseline':base,'memory':mem(),**stats(t,end)};row['peak_delta_allocated_gib']=row['memory']['peak_allocated_gib']-base['allocated_gib'];S['conditions'].append(row);event('condition_complete',**row);dump();return first
try:
 torch.cuda.init();torch.cuda.empty_cache();S['before_load']=mem();phase='loading';t=time.time();model=models.CellposeModel(gpu=True,pretrained_model=str(R/'models/cpsam_v2'),use_bfloat16=True);torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=mem();S['parameters']={'total':sum(p.numel() for p in model.net.parameters()),'requires_grad':sum(p.numel() for p in model.net.parameters() if p.requires_grad),'parameter_dtypes':{str(d):sum(p.numel() for p in model.net.parameters() if p.dtype==d) for d in {p.dtype for p in model.net.parameters()}},'theoretical_current_storage_gib':sum(p.numel()*p.element_size() for p in model.net.parameters())/G};event('model_loaded',**S['parameters'],memory=S['model_only_postload']);dump()
 phase='warmup';condition='native_one';t=time.time();warm=evaluate(model,data['test'][:1]);torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;np.save(D/'warmup_mask.npy',warm[0][0]);assert warm[0][0].shape==data['test'][0].shape;dump()
 for n in [1,4,16]:infer_condition(model,f'count_{n}',data['test'][:n])
 for size in [128,256,512]:
  imgs=[x[(x.shape[0]-size)//2:(x.shape[0]+size)//2,(x.shape[1]-size)//2:(x.shape[1]+size)//2] for x in data['test'][:4]]
  infer_condition(model,f'crop_{size}',imgs)
 phase='evaluation';condition='pretrained_test';t=time.time();pre=evaluate(model,data['test']);torch.cuda.synchronize();S['pretrained_test_seconds']=time.time()-t
 def score(pred,name):
  ap,tp,fp,fn=metrics.average_precision(gt['test'],pred,threshold=[.5,.75,.9]);row={'mean_AP50_AP75_AP90':ap.mean(0).tolist(),'pooled_TP_FP_FN':{'tp':tp.sum(0).tolist(),'fp':fp.sum(0).tolist(),'fn':fn.sum(0).tolist()},'per_image_ap':ap.tolist(),'predicted_instances':[int(p.max()) for p in pred]};S[name]=row;return row
 score(pre[0],'pretrained_accuracy');np.savez_compressed(D/'pretrained_test_masks.npz',**{str(i):m for i,m in enumerate(pre[0])});dump()
 # Preserve exact forward values to verify a fresh-process reload after training.
 before={n:p.detach().float().cpu().clone() for n,p in model.net.named_parameters() if n in ['out.weight','encoder.blocks.0.attn.qkv.weight','encoder.patch_embed.proj.weight']}
 # Observe official conversion before preprocessing and optimizer without changing training algorithm.
 oldprocess=train._process_train_test
 def process(*a,**kw):
  global phase
  phase='training_preprocessing';torch.cuda.synchronize();S['training_fp32_baseline']=mem();t=time.time();res=oldprocess(*a,**kw);torch.cuda.synchronize();S['training_preprocessing_seconds']=time.time()-t;phase='training';event('training_preprocessing_complete');dump();return res
 train._process_train_test=process
 oldadam=torch.optim.AdamW;optimizerbox=[]
 def adam(*a,**kw):
  o=oldadam(*a,**kw);optimizerbox.append(o);return o
 torch.optim.AdamW=adam
 shapes={};gradcounts={};stepn=[0];oldnet=model.net.forward
 def netforward(x):
  global phase
  phase='training' if model.net.training else ('training_validation' if condition=='finetune' else phase)
  key=str((tuple(x.shape),str(x.dtype),model.net.training));shapes[key]=shapes.get(key,0)+1
  return oldnet(x)
 model.net.forward=netforward
 oldstep=oldadam.step
 def step(self,*a,**kw):
  if stepn[0]==1:
   gradcounts['parameters_with_gradient']=sum(p.numel() for p in model.net.parameters() if p.grad is not None);gradcounts['names_without_gradient']=[n for n,p in model.net.named_parameters() if p.requires_grad and p.grad is None]
  result=oldstep(self,*a,**kw);stepn[0]+=1
  if stepn[0]%16==0:print('TRAIN_STEP',stepn[0],flush=True)
  return result
 oldadam.step=step
 phase='training';condition='finetune';torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.time();trained_path,losses,valloss=train.train_seg(model.net,train_data=data['training'],train_labels=gt['training'],test_data=data['validation'],test_labels=gt['validation'],batch_size=1,learning_rate=1e-5,n_epochs=100,weight_decay=.1,save_path=str(M),save_every=100,rescale=False,scale_range=.5,bsize=256,min_train_masks=5,model_name='cpsam_v2_bbbc039_finetuned');torch.cuda.synchronize();end=time.time();S['training']={'seconds_including_flow_preprocessing_validation_saves':end-t,'checkpoint':str(Path(trained_path).relative_to(R)),'steps':stepn[0],'epochs':100,'train_images':16,'validation_images':8,'input_shapes':shapes,'gradient_observation':gradcounts,'losses':losses.tolist(),'validation_losses':valloss.tolist(),'validation_epochs':[i for i in range(100) if i==5 or i%10==0],'memory':mem(),**stats(t,end)};assert np.isfinite(losses).all();np.savez(D/'training_losses.npz',training=losses,validation=valloss)
 S['parameter_change_examples']={n:float((dict(model.net.named_parameters())[n].detach().float().cpu()-v).abs().max()) for n,v in before.items()};phase='saving';torch.save({'optimizer':optimizerbox[0].state_dict(),'epoch':100,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'note':'state saved; resume not tested; official train_seg schedule needs explicit continuation'},M/'optimizer_rng.pt');dump();del optimizerbox[:];torch.optim.AdamW=oldadam;oldadam.step=oldstep;model.net.forward=oldnet;train._process_train_test=oldprocess;gc.collect();torch.cuda.empty_cache()
 phase='evaluation';condition='finetuned_test';t=time.time();post=evaluate(model,data['test']);torch.cuda.synchronize();S['finetuned_test_seconds']=time.time()-t;score(post[0],'finetuned_accuracy');np.savez_compressed(D/'finetuned_test_masks.npz',**{str(i):m for i,m in enumerate(post[0])});np.save(D/'reload_reference_cellprob.npy',post[1][0][2]);np.save(D/'reload_reference_flows.npy',post[1][0][1]);S['checkpoint_sha256']=hashlib.file_digest(Path(trained_path).open('rb'),'sha256').hexdigest();S['status']='training_and_evaluation_complete_pending_independent_reload';dump()
except BaseException as e:
 S['error']=repr(e);S['status']='failed';dump();raise
finally:
 stop.set();th.join(timeout=5);(D/'forward_calls.json').write_text(json.dumps(forwardrows,indent=2));dump()
