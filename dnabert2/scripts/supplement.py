import csv,json,time,threading,subprocess,os,gc,hashlib
from pathlib import Path
import numpy as np,psutil,torch
from transformers import AutoModel,AutoTokenizer,BertConfig
R=Path('/work');SRC=Path('/workspace/dnabert2_eval');D=R/'results'/os.environ['RUN_ID'];D.mkdir(exist_ok=True);G=2**30;torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
modelpath=SRC/'model';tokenizer=AutoTokenizer.from_pretrained(str(modelpath),trust_remote_code=True,local_files_only=True);config=BertConfig.from_pretrained(str(modelpath),local_files_only=True)
S={'environment':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'model_revision':(SRC/'model_revision.txt').read_text(),'settings':{'batch':8,'dtype':'FP32','tf32':False,'seed':42,'pooling':'masked mean including special tokens','training':'resource probe only; original completed saved training retained','learning_rate':2e-5,'weight_decay':.01},'conditions':[]};samples=[];phase='initial';stop=threading.Event();swap0=psutil.swap_memory().used

def mem():return dict(allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,peak_allocated_gib=torch.cuda.max_memory_allocated()/G,peak_reserved_gib=torch.cuda.max_memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,available_gib=psutil.virtual_memory().available/G,total_gib=psutil.virtual_memory().total/G)
def monitor():
 with (D/'resources.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['time_unix','phase','gpu_util','rss_gib','available_gib','swap_gib'])
  while not stop.is_set():
   t=time.time();p=phase
   try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
   except Exception:u=None
   v=psutil.virtual_memory();row=[t,p,u,psutil.Process().memory_info().rss/G,v.available/G,psutil.swap_memory().used/G];samples.append(row);w.writerow(row);f.flush()
   if v.available<8*G or psutil.swap_memory().used>swap0+2*G:os._exit(86)
   stop.wait(.5)
def save(): (D/'summary.json').write_text(json.dumps(S,indent=2))
th=threading.Thread(target=monitor,daemon=True);th.start()
try:
 phase='loading';t=time.time();encoder=AutoModel.from_pretrained(str(modelpath),config=config,trust_remote_code=True,local_files_only=True).float().cuda();torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=mem();S['parameters']=sum(p.numel() for p in encoder.parameters());save()
 prepared=SRC/'external_data/BEND/prepared/20261012_001503_n1000_seed42';data={}
 for bp in [512,1000,2000,4000]:
  rows=list(csv.DictReader((prepared/f'train_n1000_bp{bp}.csv').open()));assert len(rows)==1000 and len({r['id'] for r in rows})==1000 and all(len(r['sequence'])==bp and set(r['sequence'])<=set('ACGT') for r in rows);data[bp]=rows
 assert all([r['id'] for r in data[b]]==[r['id'] for r in data[512]] for b in data)
 (D/'input_examples.json').write_text(json.dumps({str(bp):rows[0] for bp,rows in data.items()},indent=2));S['input_source']=str(prepared);S['input_ids_sha256']=hashlib.sha256('\n'.join(r['id'] for r in data[512]).encode()).hexdigest()
 def batch(rows):
  x=tokenizer([r['sequence'] for r in rows],padding=True,truncation=False,return_tensors='pt');lens=x['attention_mask'].sum(1).tolist();return {k:x[k].cuda() for k in ['input_ids','attention_mask']},lens
 def pool(x):
  y=encoder(**x)[0];m=x['attention_mask'].unsqueeze(-1).to(y.dtype);return (y*m).sum(1)/m.sum(1).clamp_min(1)
 encoder.eval();phase='warmup';t=time.time()
 with torch.inference_mode():
  for _ in range(3):x,_=batch(data[2000][:8]);z=pool(x);del x,z
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t
 for bp,n in [(2000,32),(2000,128),(2000,1000),(512,128),(1000,128),(4000,128)]:
  phase=f'inference_bp{bp}_n{n}';torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=mem();t=time.time();repeats=0;lengths=[]
  with torch.inference_mode():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,8):
     x,lens=batch(data[bp][i:i+8]);z=pool(x);lengths.extend(lens);del x,z
    torch.cuda.synchronize();repeats+=1
  end=time.time();us=[r[2] for r in samples if r[1]==phase and r[0]>=t and r[2] is not None];row={'condition':phase,'mode':'inference','bp':bp,'unique_inputs':n,'repeats':repeats,'processed':n*repeats,'seconds':end-t,'throughput':n*repeats/(end-t),'tokens_min':min(lengths),'tokens_mean':float(np.mean(lengths)),'tokens_max':max(lengths),'baseline':base,'memory':mem(),'gpu_util_avg':float(np.mean(us)) if us else None,'gpu_util_max':max(us) if us else None,'samples':len(us)};S['conditions'].append(row);print(json.dumps(row),flush=True);save()
 # Fresh initialization for each training resource probe; saved full training results are reused.
 del encoder;gc.collect();torch.cuda.empty_cache()
 for bp in [512,1000,2000,4000]:
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);phase=f'training_load_{bp}';encoder=AutoModel.from_pretrained(str(modelpath),config=config,trust_remote_code=True,local_files_only=True).float().cuda();head=torch.nn.Linear(config.hidden_size,18).float().cuda();encoder.train();opt=torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()),lr=2e-5,weight_decay=.01);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=mem();phase=f'training_bp{bp}_n128';t=time.time();losses=[];lengths=[]
  for i in range(0,128,8):
   rows=data[bp][i:i+8];x,lens=batch(rows);lengths.extend(lens);targets=torch.zeros(len(rows),18,device='cuda')
   for j,r in enumerate(rows):targets[j,json.loads(r['labels'])]=1
   opt.zero_grad(set_to_none=True);y=head(pool(x));loss=torch.nn.functional.binary_cross_entropy_with_logits(y,targets);loss.backward();opt.step();losses.append(float(loss.detach()));del x,y,loss,targets
  torch.cuda.synchronize();end=time.time();us=[r[2] for r in samples if r[1]==phase and r[0]>=t and r[2] is not None];row={'condition':phase,'mode':'training_resource_probe','bp':bp,'unique_inputs':128,'repeats':1,'processed':128,'steps':16,'seconds':end-t,'throughput':128/(end-t),'tokens_min':min(lengths),'tokens_mean':float(np.mean(lengths)),'tokens_max':max(lengths),'baseline':base,'memory':mem(),'gpu_util_avg':float(np.mean(us)) if us else None,'gpu_util_max':max(us) if us else None,'samples':len(us),'losses':losses};S['conditions'].append(row);print(json.dumps(row),flush=True);save();del encoder,head,opt;gc.collect();torch.cuda.empty_cache()
 S['status']='complete';save()
finally:stop.set();th.join(timeout=5);save()
