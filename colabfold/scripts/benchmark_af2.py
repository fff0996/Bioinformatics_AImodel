"""Instrument existing ColabFold; serialize all GPU work; explicit AOT compile timing."""
import os,sys,pathlib,time,json,threading,subprocess,csv,traceback,inspect,functools,datetime,hashlib,shutil,resource
import numpy as np,jax
from colabfold import batch
from colabfold.alphafold import models
from alphafold.model import model as afmodel
from colabfold.input import get_queries
from colabfold.utils import setup_logging
COMPLEX=len(sys.argv)>3 and sys.argv[3]=='multimer'
MANIFEST='complex_manifest.json' if COMPLEX else 'input_manifest.json'
MODELTYPE='alphafold2_multimer_v3' if COMPLEX else 'alphafold2_ptm'
RECYCLES=20 if COMPLEX else 3
MAXSEQ=508 if COMPLEX else 512
MAXEXTRA=2048 if COMPLEX else 5120
R=pathlib.Path('/work');O=R/'results'/sys.argv[1];O.mkdir(exist_ok=True);MSA=R/'results'/sys.argv[2]/'msa';phase='setup';current='setup';stop=threading.Event();samples=[];predictions=[];calls=[];events=[];query_stats=[];allstart=time.perf_counter()
shutil.copy2(__file__,O/'benchmark_af2.py')
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def event(kind,**kw):
 row=dict(time_unix=time.time(),kind=kind,query=current,phase=phase,**kw);events.append(row)
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
 print('BENCH',json.dumps(row,default=str),flush=True)
def memory():
 d={}
 for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
  k,v=line.split(':');d[k]=int(v.strip().split()[0])*1024
 rss=next(int(l.split()[1])*1024 for l in pathlib.Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'))
 return rss,d['MemAvailable'],d['SwapTotal']-d['SwapFree'],d['MemTotal']
initial_swap=memory()[2];dev=jax.devices('gpu')[0]
def dmem():
 try:return {k:int(v) for k,v in (dev.memory_stats() or {}).items()}
 except Exception:return {}
def monitor():
 with (O/'resources.csv').open('w') as f:
  fields=['query_start','query_end','query','phase','gpu_util_percent','rss_bytes','system_available_bytes','swap_used_bytes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  while not stop.is_set():
   start=time.time();ph=phase;qid=current
   try:
    p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3);v=float(p.stdout.strip()) if p.returncode==0 else None
   except Exception:v=None
   rss,available,swap,total=memory();row=dict(query_start=start,query_end=time.time(),query=qid,phase=ph,gpu_util_percent=v,rss_bytes=rss,system_available_bytes=available,swap_used_bytes=swap);samples.append(row);w.writerow(row);f.flush()
   if available<8*2**30 or swap>initial_swap+2*2**30:
    event('memory_pressure_stop',available_bytes=available,swap_bytes=swap);os._exit(70)
   stop.wait(max(0,.5-(time.time()-start)))
thread=threading.Thread(target=monitor,daemon=True);thread.start()
summary=dict(status='running',run=str(O),settings=dict(model_type=MODELTYPE,num_models=5,model_order=[1,2,3,4,5],num_recycles=RECYCLES,recycle_early_stop_tolerance=0.0,num_ensemble=1,num_seeds=1,random_seed=42,use_templates=False,num_relax=0,dropout=False,msa_mode='mmseqs2_uniref_env',max_seq=MAXSEQ,max_extra_seq=MAXEXTRA,use_bfloat16=True,use_pallas=False,compile_mode='tuned',recompile_padding=10,batch='one protein at a time, five weight sets sequentially',preallocation=False),predictions=predictions)
write('summary.json',summary)
class TimedJit:
 def __init__(self,fn):self.fn=fn;self.cache={}
 def __getattr__(self,key):return getattr(self.fn,key)
 def __call__(self,*args,**kw):
  global phase
  leaves,tree=jax.tree_util.tree_flatten((args,kw));signature=(str(tree),tuple((tuple(getattr(x,'shape',())),str(getattr(x,'dtype',type(x)))) for x in leaves));oldphase=phase
  if signature not in self.cache:
   phase='compilation';t=time.perf_counter();start=time.time();compiled=self.fn.lower(*args,**kw).compile();seconds=time.perf_counter()-t;self.cache[signature]=compiled
   try:analysis=str(compiled.memory_analysis())
   except Exception:analysis=None
   event('compilation',start_unix=start,end_unix=time.time(),seconds=seconds,memory_analysis=analysis,shape_variants=len(self.cache))
  phase='execution';start=time.time();t=time.perf_counter();out=self.cache[signature](*args,**kw);jax.block_until_ready(out);seconds=time.perf_counter()-t;end=time.time();row=dict(query=current,start_unix=start,end_unix=end,seconds=seconds,device_memory=dmem());calls.append(row);event('gpu_apply',**{k:v for k,v in row.items() if k!='query'});phase=oldphase;return out
original_init=afmodel.RunModel.__init__
def timed_init(self,*args,**kw):original_init(self,*args,**kw);self.apply=TimedJit(self.apply)
afmodel.RunModel.__init__=timed_init
origload=models.load_models_and_params
@functools.wraps(origload)
def load(*args,**kw):
 global phase
 old=phase;phase='model_loading';t=time.perf_counter();out=origload(*args,**kw);jax.block_until_ready([p for n,r,p in out]);seconds=time.perf_counter()-t;counts=[dict(model=n,parameters=sum(int(x.size) for x in jax.tree_util.tree_leaves(p)),parameter_dtypes=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(p)})) for n,r,p in out];event('model_loading',seconds=seconds,models=counts);summary['loaded_models']=counts;summary['model_loading_seconds']=seconds;phase=old;return out
models.load_models_and_params=load
# Time host preprocessing separately; never alter returned features.
def timed_function(obj,name,label):
 old=getattr(obj,name)
 @functools.wraps(old)
 def fn(*a,**kw):
  global phase,current
  if name=='generate_input_feature':
   bound=inspect.signature(old).bind_partial(*a,**kw);seqs=bound.arguments.get('query_seqs_unique',[]);seq=''.join(seqs) if isinstance(seqs,list) else str(seqs)
   current=next((q['id'] for q in json.loads((R/'data'/MANIFEST).read_text())['queries'] if q['sequence']==seq),current)
  ph=phase;phase=label;t=time.perf_counter();result=old(*a,**kw);event(label,function=name,seconds=time.perf_counter()-t);phase=ph;return result
 setattr(obj,name,fn)
for name in ['unserialize_msa','generate_input_feature']:timed_function(batch,name,'preprocessing')
timed_function(afmodel.RunModel,'process_features','preprocessing')
origpredict=afmodel.RunModel.predict
@functools.wraps(origpredict)
def predict(self,feat,*args,**kw):
 global phase
 phase='prediction';start=time.perf_counter();event('actual_model_input',shapes={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in feat.items() if hasattr(v,'shape')},bfloat16=bool(self.config.model.global_config.bfloat16),training=False)
 result=origpredict(self,feat,*args,**kw);event('predict_total',seconds=time.perf_counter()-start,recycles=int(result[1]));phase='output';return result
afmodel.RunModel.predict=predict
origstructure=batch.predict_structure
@functools.wraps(origstructure)
def structure(*a,**kw):
 global current,phase
 bound=inspect.signature(origstructure).bind_partial(*a,**kw);current=str(bound.arguments['prefix']);phase='query';t=time.perf_counter();start=time.time();res=origstructure(*a,**kw);end=time.time();stat=dict(query=current,wall_seconds=time.perf_counter()-t,start_unix=start,end_unix=end);query_stats.append(stat);event('query_complete',**{k:v for k,v in stat.items() if k!='query'});write('query_stats.json',query_stats);phase='between_queries';return res
batch.predict_structure=structure
# Track explicit text output writes as a lower bound on serialization time.
origwrite=pathlib.Path.write_text
@functools.wraps(origwrite)
def timedwrite(self,*args,**kw):
 t=time.perf_counter();result=origwrite(self,*args,**kw)
 if str(self).startswith(str(O)) and self.suffix in ['.pdb','.json','.a3m'] and self.name not in ['summary.json','query_stats.json','input_report.json']:
  event('file_write',path=str(self),seconds=time.perf_counter()-t)
 return result
pathlib.Path.write_text=timedwrite

def callback(protein,lengths,result,features,tag):
 row=dict(query=current,tag=tag[0],lengths=list(map(int,lengths)),mean_plddt=float(np.mean(result['plddt'][:sum(lengths)])),ptm=float(result['ptm']) if 'ptm' in result else None,iptm=float(result['iptm']) if 'iptm' in result else None,plddt_shape=list(result['plddt'].shape),pae_shape=list(result['predicted_aligned_error'].shape) if 'predicted_aligned_error' in result else None,coordinates_shape=list(protein.atom_positions.shape),finite_coordinates=bool(np.isfinite(protein.atom_positions).all()),finite_plddt=bool(np.isfinite(result['plddt']).all()))
 assert row['finite_coordinates'] and row['finite_plddt'];predictions.append(row);event('validated_prediction',**{k:v for k,v in row.items() if k!='query'});write('predictions.json',predictions)
try:
 setup_logging(O/'log.txt');manifest=json.loads((R/'data'/MANIFEST).read_text());queries=[];report=[]
 for q in manifest['queries']:
  p=MSA/(q['id']+'.a3m');text=p.read_text();seqs=[]
  for line in text.splitlines():
   if line.startswith(('>','#')):continue
   if line.strip():seqs.append(''.join(c for c in line if not c.islower()))
  assert seqs and seqs[0]==q['sequence'];assert all(len(s)==q['length'] for s in seqs)
  report.append(dict(id=q['id'],residues=q['length'],msa_depth=len(seqs),msa_aligned_columns=len(seqs[0]),msa_characters=sorted(set(''.join(seqs))),msa_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),msa_bytes=p.stat().st_size,token_dtype='int32 features',truncated=False,templates=False))
  queries.append((q['id'],q['chains'] if COMPLEX else q['sequence'],[text],None))
 write('input_report.json',report);event('input_verified',queries=report);event('initial_memory',rss_bytes=memory()[0],available_bytes=memory()[1],system_total_bytes=memory()[3],gpu=dmem())
 phase='pipeline';start=time.perf_counter()
 batch.run(queries=queries,result_dir=O/'predictions',num_models=5,is_complex=COMPLEX,num_recycles=RECYCLES,recycle_early_stop_tolerance=0.0,model_order=[1,2,3,4,5],num_ensemble=1,model_type=MODELTYPE,msa_mode='mmseqs2_uniref_env',use_templates=False,num_relax=0,keep_existing_results=False,data_dir=pathlib.Path('/cache'),random_seed=42,num_seeds=1,recompile_padding=10,prediction_callback=callback,feature_dict_callback=lambda feat:event('raw_feature_shapes',shapes={k:list(v.shape) for k,v in feat.items() if hasattr(v,'shape')}),use_bfloat16=True,use_pallas=False,compile_mode='tuned',max_seq=MAXSEQ,max_extra_seq=MAXEXTRA)
 summary['pipeline_wall_seconds']=time.perf_counter()-start;assert len(predictions)==5*len(queries),(len(predictions),len(queries));summary['status']='success';summary['gpu_apply_seconds']=sum(c['seconds'] for c in calls);summary['compilation_seconds']=sum(e['seconds'] for e in events if e['kind']=='compilation');summary['final_device_memory']=dmem();summary['wall_seconds']=time.perf_counter()-allstart;write('summary.json',summary);write('apply_calls.json',calls);event('success',summary_path=str(O/'summary.json'))
except Exception as e:
 summary['status']='failed';summary['error']=str(e);summary['traceback']=traceback.format_exc();write('summary.json',summary);event('failed',error=str(e));traceback.print_exc();raise
finally:stop.set();thread.join(timeout=5)
