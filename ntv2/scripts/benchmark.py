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
 del model;gc.collect();torch.cuda.empty_cache();torch.manual_seed(42);meter.phase='loading_classifier';t=time.time();model=AutoModelForSequenceClassification.from_pretrained(P,trust_remote_code=True,local_files_only=True,num_labels=2).float().cuda();torch.cuda.synchronize();S['classifier_loading_seconds']=time.time()-t;S['classifier_model_only']=meter.memory();S['parameters']['classifier_total']=sum(p.numel() for p in model.parameters());S['parameters']['classifier_trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad);save()
 def evaluate(rows):
  model.eval();logits=[]
  with torch.no_grad():
   for i in range(0,len(rows),8):
    x,_=encode([r['sequence'] for r in rows[i:i+8]]);z=model(**x).logits;logits.append(z.cpu().numpy())
  p=np.concatenate(logits);y=np.array([r['label'] for r in rows]);pred=p.argmax(1);prob=torch.from_numpy(p).softmax(-1)[:,1].numpy();return {'accuracy':float(accuracy_score(y,pred)),'f1':float(f1_score(y,pred)),'mcc':float(matthews_corrcoef(y,pred)),'auroc':float(roc_auc_score(y,prob))},p
 meter.phase='pretrain_validation';S['initial_validation'],_=evaluate(inputs['validation']);before={'backbone':model.esm.encoder.layer[-1].output.dense.weight.detach().cpu().clone(),'head':next(model.classifier.parameters()).detach().cpu().clone()};opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.);S['epochs']=[];grad=None
 for epoch in range(2):
  model.train();order=np.random.default_rng(42+epoch).permutation(512);losses=[];t,base=meter.begin(f'train_epoch{epoch+1}')
  for step in range(0,512,8):
   rows=[inputs['train'][int(i)] for i in order[step:step+8]];x,lens=encode([r['sequence'] for r in rows]);labels=torch.tensor([r['label'] for r in rows],device='cuda');opt.zero_grad(set_to_none=True);result=model(**x,labels=labels);loss=result.loss;assert torch.isfinite(loss);loss.backward()
   if grad is None:grad={'numel_with_gradient':sum(p.numel() for p in model.parameters() if p.grad is not None),'no_gradient_names':[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]}
   opt.step();losses.append(float(loss.detach()));del result,loss,x,labels
   if step%128==0:print('TRAIN',epoch+1,step,'loss',losses[-1],flush=True)
  row=meter.end(t,base,name=f'train_epoch{epoch+1}',mode='training',unique_inputs=512,repeats=1,processed=512,steps=64,loss_mean=float(np.mean(losses)),losses=losses);row['throughput']=512/row['seconds'];S['conditions'].append(row);meter.phase='validation';val,log=evaluate(inputs['validation']);S['epochs'].append({'epoch':epoch+1,'training':row,'validation':val});print('EPOCH',epoch+1,val,flush=True);save()
 S['gradient_observation']=grad;S['parameter_changes']={'backbone_max_abs':float((model.esm.encoder.layer[-1].output.dense.weight.detach().cpu()-before['backbone']).abs().max()),'head_max_abs':float((next(model.classifier.parameters()).detach().cpu()-before['head']).abs().max())};meter.phase='saving';t=time.time();dest=M/'classifier';model.save_pretrained(dest,safe_serialization=True);tokenizer.save_pretrained(dest);torch.save({'optimizer':opt.state_dict(),'epoch':2,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state()},M/'optimizer_rng.pt');S['save_seconds']=time.time()-t;S['saved_model']=str(dest.relative_to(R));write_json(M/'task.json',{'labels':{'0':'non-enhancer','1':'enhancer'},'settings':S['settings'],'resume_tested':False,'source_checkpoint':'models/checkpoint','input_split':'data/inputs.json'});meter.phase='test';t=time.time();S['test'],logits=evaluate(inputs['test']);torch.cuda.synchronize();S['test_seconds']=time.time()-t;np.save(D/'test_logits.npy',logits);write_json(D/'test_predictions.json',[{'id':r['id'],'label':r['label'],'prediction':int(p.argmax())} for r,p in zip(inputs['test'],logits)]);S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['error']=repr(e);S['status']='failed';save();raise
finally:meter.close();save()
