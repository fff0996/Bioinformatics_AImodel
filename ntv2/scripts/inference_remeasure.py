import sys,os,time,json,gc,random,hashlib
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,torch
from transformers import AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'models/checkpoint';torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'batch':8,'dtype':'FP32','tf32':False,'seed':42,'lr':1e-5,'weight_decay':0.,'epochs':2,'train':512,'validation':128,'test':400,'inference':'MLM model.esm backbone, masked mean including special tokens; MLM logits not computed','finetuning':'official custom EsmForSequenceClassification CLS MLP head; all trainable parameters'}}
def save():write_json(D/'summary.json',S)
def encode(seqs):
 a=tokenizer(seqs,padding=True,truncation=False,return_tensors='pt');lens=a['attention_mask'].sum(1).tolist();assert max(lens)<=2048,(max(lens),len(seqs[0]));return {k:a[k].cuda() for k in ['input_ids','attention_mask']},lens
try:
 previous=json.loads((D/'summary.json').read_text());write_json(D/'summary_before_remeasurement.json',previous);S=previous;oldparams=dict(S['parameters']);S['conditions']=[r for r in S['conditions'] if r['mode']=='training'];S['status']='remeasuring_inference';
 inputs=json.loads((R/'data/inputs.json').read_text());tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);tok={}
 for bp in [600,3000,6000,12276]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference']];v=tokenizer(seqs,padding=False,truncation=False)['input_ids'];lens=list(map(len,v));assert max(lens)<=2048;tok[str(bp)]={'min':min(lens),'max':max(lens),'mean':float(np.mean(lens))}
 S['token_audit']=tok;S['input_audit']='data/input_audit.json';save();meter.phase='loading_mlm';t=time.time();model=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True).float().cuda().eval();torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'mlm_total':sum(p.numel() for p in model.parameters()),'embedding_backbone':sum(p.numel() for p in model.esm.parameters()),'mlm_head':sum(p.numel() for p in model.lm_head.parameters())};save()
 def embedding(seqs):
  x,lens=encode(seqs);h=model.esm(**x,return_dict=True).last_hidden_state;m=x['attention_mask'].unsqueeze(-1);z=(h*m).sum(1)/m.sum(1);return z,lens
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():
  for _ in range(3):z,_=embedding([r['sequence'][:600] for r in inputs['long_inference'][:8]])
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;np.save(D/'example_embeddings.npy',z.cpu().numpy());del z
 for bp,n in [(3000,8),(3000,32),(3000,128),(600,32),(6000,32),(12276,32)]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference'][:n]];name=f'inference_bp{bp}_n{n}';t,base=meter.begin(name);repeats=0;lengths=[]
  with torch.inference_mode():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,8):z,lens=embedding(seqs[i:i+8]);lengths.extend(lens);del z
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='inference',bp=bp,unique_inputs=n,repeats=repeats,processed=n*repeats,tokens_min=min(lengths),tokens_mean=float(np.mean(lengths)),tokens_max=max(lengths));row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',json.dumps(row),flush=True);save()
 S['parameters']={**oldparams,**S['parameters']};S['resource_files']=[p.name for p in D.glob('resources*.csv')];S['recovery_note']='Initial inference CSV was overwritten by first recovery. Inference six conditions remeasured in fresh process, training retained, each CSV now unique.';S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['error']=repr(e);S['status']='failed';save();raise
finally:meter.close();save()
