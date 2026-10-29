import os,json,time,hashlib
from pathlib import Path
import numpy as np,torch,psutil
G=2**30

def install(namespace):
 original_load=namespace['_load_torch_checkpoint'];original_predict=namespace['_torch_predict'];state={'first':True}
 def emit(r):
  root=Path(os.environ['CLAIR3_METRICS_DIR']);root.mkdir(parents=True,exist_ok=True)
  with (root/f'worker_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps({'time_unix':time.time(),'pid':os.getpid(),**r})+'\n')
 def mem():return dict(allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,peak_allocated_gib=torch.cuda.max_memory_allocated()/G,peak_reserved_gib=torch.cuda.max_memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,available_gib=psutil.virtual_memory().available/G)
 def load(model,path,device):
  t=time.time();original_load(model,path,device)
  if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  state['model']=type(model).__name__;state['checkpoint']=path;emit(dict(event='model_only_postload',model=state['model'],checkpoint=path,total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),load_seconds=time.time()-t,dtype=str(next(model.parameters()).dtype),tf32=torch.backends.cuda.matmul.allow_tf32,memory=mem()))
 def predict(model,device,X):
  if device.type=='cuda':torch.cuda.synchronize()
  t=time.time();Y=original_predict(model,device,X)
  if device.type=='cuda':torch.cuda.synchronize()
  emit(dict(event='forward',model=state.get('model'),shape=list(X.shape),dtype=str(X.dtype),seconds=time.time()-t,memory=mem()))
  cache_dir=os.environ.get('CLAIR3_FULL_CANDIDATE_CACHE')
  if cache_dir:
   dest=Path(cache_dir);dest.mkdir(parents=True,exist_ok=True);idx=state.get('cache_index',0);state['cache_index']=idx+1;name=f"{state.get('model')}_{os.getpid()}_{idx}.npz";np.savez_compressed(dest/name,X=X,Y=Y);emit(dict(event='full_actual_candidate_cache',file=str(dest/name),model=state.get('model'),shape=list(X.shape),synthetic_rows=0))

  if state['first']:
   root=Path(os.environ['CLAIR3_METRICS_DIR']);np.savez_compressed(root/f'reload_input_{os.getpid()}.npz',X=X[:min(32,len(X))],Y=Y[:min(32,len(Y))]);emit(dict(event='reload_sample',file=f'reload_input_{os.getpid()}.npz',model=state.get('model'),checkpoint=state.get('checkpoint')));state['first']=False
  return Y
 namespace['_load_torch_checkpoint']=load;namespace['_torch_predict']=predict
