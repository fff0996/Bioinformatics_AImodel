from pathlib import Path
import sys,os,time,json,random,gc,hashlib
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import torch,numpy as np
from sklearn.metrics import average_precision_score
from openspliceai.train_base.openspliceai import SpliceAI
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'source/models/openspliceai-mane/10000nt';W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'checkpoint':'official OSAI_MANE10000 seeds10..14 five-model ensemble','batch':12,'basis':'official predict/transfer CL10000 code BATCH_SIZE=6*N_GPUS with hardcoded N_GPUS2; run uses single GPU with batch12, not two GPUs','dtype':'FP32','tf32':False,'seed':42,'context_nt':10000,'normal_output_nt':5000,'fine_tune_checkpoint':'seed10 only, compare single-seed10 baseline','fine_tune_epochs':10,'lr':1e-4,'optimizer':'AdamW default weight_decay.01; cosine restart T0=5,Tmult1,eta_min1e-5','train_n':96,'validation_n':24,'test_n':36,'fine_tune':'all parameters, curated packaged GRCh38 annotation'} }
def save():write_json(D/'summary.json',S)
def crop(r,sl):return r['sequence'][10000-(sl+10000)//2:10000+(sl+10000)//2],np.array(r['labels_10000'][5000-sl//2:5000+sl//2],dtype=np.int64)
def encode(rows,sl):
 seqs=[crop(r,sl)[0] for r in rows];tab=np.zeros((256,4),dtype=np.float32)
 for j,c in enumerate('ACGT'):tab[ord(c),j]=1
 x=np.stack([tab[np.frombuffer(seq.encode(),dtype=np.uint8)].T for seq in seqs]);y=np.stack([crop(r,sl)[1] for r in rows]);return torch.from_numpy(x).cuda(),y
try:
 data=json.loads((R/'data/inputs.json').read_text());meter.phase='loading';t=time.time();models=[];sources=[]
 for seed in range(10,15):
  p=P/f'model_10000nt_rs{seed}.pt';m=SpliceAI(32,W,AR).float().cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True));m.eval();models.append(m);sources.append({'checkpoint':str(p.relative_to(R)),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
 torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'per_model':sum(p.numel() for p in models[0].parameters()),'ensemble_total':sum(p.numel() for m in models for p in m.parameters()),'trainable_single':sum(p.numel() for p in models[0].parameters() if p.requires_grad)};S['checkpoint_sources']=sources;save()
 def predict(ms,rows,sl):
  x,y=encode(rows,sl);out=None
  for m in ms:
   z=m(x);out=z if out is None else out+z
  return out/len(ms),y
 meter.phase='warmup';t=time.time()
 with torch.no_grad():
  for _ in range(3):z,y=predict(models,data['test'][:12],5000);del z,y
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t
 for sl,n in [(5000,12),(5000,24),(5000,36),(1000,12),(10000,12)]:
  name=f'ensemble_output{sl}_n{n}';t,base=meter.begin(name);repeats=0
  with torch.no_grad():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,12):z,y=predict(models,data['test'][i:i+12],sl);assert z.shape==(min(12,n-i),3,sl);del z,y
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='ensemble_inference',bp=sl+10000,output_nt=sl,unique_inputs=n,repeats=repeats,processed=n*repeats);row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',name,row['seconds'],row['memory']['peak_allocated_gib'],flush=True);save()
 def score(ms,rows):
  arrays=[];labels=[]
  for m in ms:m.eval()
  with torch.no_grad():
   for i in range(0,len(rows),12):z,y=predict(ms,rows[i:i+12],5000);arrays.append(z.cpu().numpy());labels.append(y)
  z=np.concatenate(arrays);y=np.concatenate(labels);res={}
  for k,name in [(1,'acceptor'),(2,'donor')]:
   valid=y>=0;true=(y==k)[valid];probs=z[:,k,:][valid];num=int(true.sum());idx=np.argpartition(probs,-num)[-num:] if num else [];res[name]={'positive_sites':num,'auprc':float(average_precision_score(true,probs)),'top_k_recall':float(true[idx].mean()) if num else None}
  return res,z
 meter.phase='initial_test';S['ensemble_test_before'],_=score(models,data['test']);S['single_seed10_test_before'],_=score(models[:1],data['test']);del models,m;gc.collect();torch.cuda.empty_cache();meter.phase='classifier_loading';model=SpliceAI(32,W,AR).float().cuda();model.load_state_dict(torch.load(P/'model_10000nt_rs10.pt',map_location='cuda',weights_only=True));S['single_model_only']=meter.memory();before=model.initial_conv.weight.detach().cpu().clone();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=5,T_mult=1,eta_min=1e-5);history=[];grad=None
 for epoch in range(10):
  model.train();order=np.random.default_rng(42+epoch).permutation(96);t,base=meter.begin(f'training_epoch{epoch+1}');losses=[]
  for step in range(0,96,12):
   rows=[data['train'][int(i)] for i in order[step:step+12]];x,y=encode(rows,5000);yy=torch.from_numpy(y).cuda();valid=yy>=0;labels=torch.nn.functional.one_hot(yy.clamp_min(0),3).permute(0,2,1).float()*valid.unsqueeze(1);opt.zero_grad(set_to_none=True);z=model(x);loss=-(labels*torch.log(z+1e-10)).sum(1).mean();assert torch.isfinite(loss);loss.backward()
   if grad is None:grad=sum(p.numel() for p in model.parameters() if p.grad is not None)
   opt.step();losses.append(float(loss.detach()));del x,yy,labels,z,loss
  sched.step(epoch+1);row=meter.end(t,base,name=f'training_epoch{epoch+1}',mode='single_model_training',bp=15000,output_nt=5000,unique_inputs=96,repeats=1,processed=96,steps=8,loss_mean=float(np.mean(losses)));row['throughput']=96/row['seconds'];S['conditions'].append(row);meter.phase='validation';val,_=score([model],data['validation']);history.append({'epoch':epoch+1,'loss':row['loss_mean'],'validation':val});print('EPOCH',epoch+1,row['loss_mean'],val,flush=True);S['epochs']=history;save()
 S['parameters']['observed_gradient']=grad;S['weight_change_max_abs']=float((model.initial_conv.weight.detach().cpu()-before).abs().max());assert S['weight_change_max_abs']>0;meter.phase='saving';t=time.time();model.eval();torch.save(model.state_dict(),M/'model_seed10_finetuned.pt');torch.save({'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':10,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},M/'training_state.pt');write_json(M/'task.json',{'L':32,'W':W.tolist(),'AR':AR.tolist(),'context':10000,'output':5000,'input_channels':['A','C','G','T'],'labels':['nonsplice','acceptor','donor'],'annotation':'source/openspliceai/variant/annotations/grch38.txt','resume_tested':False});S['save_seconds']=time.time()-t;S['saved_model']=str((M/'model_seed10_finetuned.pt').relative_to(R));meter.phase='final_test';S['finetuned_test'],z=score([model],data['test']);np.save(D/'test_probabilities.npy',z);write_json(D/'test_window_ids.json',[r['id'] for r in data['test']]);S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
