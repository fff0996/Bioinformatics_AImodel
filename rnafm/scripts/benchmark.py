from pathlib import Path
import sys,os,time,json,random,argparse
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import torch,numpy as np,fm
from sklearn.metrics import accuracy_score,f1_score
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.serialization.add_safe_globals([argparse.Namespace]);meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'rna_fm':fm.__version__,'gpu':torch.cuda.get_device_name(0)},'settings':{'inference_batch_cap':50,'head_training_batch':8,'dtype':'FP32','tf32':False,'seed':42,'epochs':100,'lr':.001,'optimizer':'Adam','training':'frozen pretrained RNA-FM + linear640->28 classifier; backbone NOT finetuned','pooling':'mean only actual nucleotide tokens, excludes BOS/EOS/PAD','reference':'official RNA type tutorial chunk50/head batch8/lr1e-3/100epochs; corrected padded-mean pooling'}}
def save():write_json(D/'summary.json',S)
try:
 data=json.loads((R/'data/inputs.json').read_text());meter.phase='loading';t=time.time();backbone,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');backbone=backbone.float().cuda().eval();backbone.requires_grad_(False);convert=alphabet.get_batch_converter();torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'backbone_total':sum(p.numel() for p in backbone.parameters()),'backbone_trainable':sum(p.numel() for p in backbone.parameters() if p.requires_grad),'head':640*28+28};save()
 def embed(rows):
  pairs=[(r['id'],r['sequence']) for r in rows];_,_,tokens=convert(pairs);assert tokens.shape[1]<=1024;tokens=tokens.cuda();result=backbone(tokens,repr_layers=[12],return_contacts=False);h=result['representations'][12];z=torch.stack([h[i,1:len(r['sequence'])+1].mean(0) for i,r in enumerate(rows)]);return z,h,tokens
 meter.phase='warmup';t=time.time()
 with torch.no_grad():
  for _ in range(3):z,h,tok=embed(data['train'][:8]);del z,h,tok
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t
 for length,n,group in [(512,8,'count512'),(512,32,'count512'),(512,64,'count512'),(128,7,'long'),(256,7,'long'),(512,7,'long'),(960,7,'long')]:
  rows=[{**r,'sequence':r['sequence'][:length]} for r in data[group][:n]];name=f'inference_nt{length}_n{n}';t,base=meter.begin(name);repeats=0;shapes=set()
  with torch.no_grad():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,50):z,h,tok=embed(rows[i:i+50]);shapes.add(str(list(tok.shape)));del z,h,tok
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='inference',length=length,unique_inputs=n,repeats=repeats,processed=n*repeats,actual_token_shapes=sorted(shapes));row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',name,row['seconds'],row['memory']['peak_allocated_gib'],flush=True);save()
 features={}
 for split in ['train','validation','test']:
  rows=data[split];t,base=meter.begin('extract_'+split);parts=[];shapes=[]
  with torch.no_grad():
   for i in range(0,len(rows),50):
    z,h,tok=embed(rows[i:i+50]);parts.append(z.cpu().numpy());shapes.append(list(tok.shape))
    if split=='train' and i==0:np.save(D/'example_token_embedding.npy',h[0,:len(rows[0]['sequence'])+2].cpu().numpy())
    del z,h,tok
  row=meter.end(t,base,name='extract_'+split,mode='native_embedding',unique_inputs=len(rows),repeats=1,processed=len(rows),actual_token_shapes=shapes);row['throughput']=len(rows)/row['seconds'];S['conditions'].append(row);features[split]=np.concatenate(parts);np.save(D/(split+'_embeddings.npy'),features[split]);save()
 head=torch.nn.Linear(640,28).float().cuda();opt=torch.optim.Adam(head.parameters(),lr=.001);criterion=torch.nn.CrossEntropyLoss();Ys={k:np.array([r['label'] for r in data[k]]) for k in features};before=head.weight.detach().cpu().clone();originalbackbone=next(backbone.parameters()).detach().cpu().clone();history=[];best=-1;beststate=None;bestepoch=None
 def evaluate(split):
  head.eval();parts=[]
  with torch.no_grad():
   for i in range(0,len(features[split]),8):parts.append(head(torch.from_numpy(features[split][i:i+8]).cuda()).cpu().numpy())
  logits=np.concatenate(parts);pred=logits.argmax(1);return {'accuracy':float(accuracy_score(Ys[split],pred)),'macro_f1':float(f1_score(Ys[split],pred,average='macro',labels=list(range(28)),zero_division=0))},logits
 t,base=meter.begin('linear_head_training_100epochs')
 for epoch in range(100):
  head.train();order=np.random.default_rng(42+epoch).permutation(len(features['train']));losses=[]
  for i in range(0,len(order),8):
   ids=order[i:i+8];x=torch.from_numpy(features['train'][ids]).cuda();y=torch.from_numpy(Ys['train'][ids]).cuda();opt.zero_grad(set_to_none=True);loss=criterion(head(x),y);loss.backward();opt.step();losses.append(float(loss.detach()))
  val,_=evaluate('validation');history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':val})
  if val['accuracy']>best:best=val['accuracy'];bestepoch=epoch+1;beststate={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
  if (epoch+1)%20==0:print('HEAD_EPOCH',epoch+1,val,flush=True)
 row=meter.end(t,base,name='linear_head_training_100epochs',mode='linear_head_training_with_validation',unique_inputs=len(features['train']),repeats=100,processed=len(features['train'])*100,steps=int(np.ceil(len(features['train'])/8))*100);row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);S['training_history']=history;S['best_epoch']=bestepoch;S['head_weight_change_max_abs']=float((head.weight.detach().cpu()-before).abs().max());S['backbone_weight_change_max_abs']=float((next(backbone.parameters()).detach().cpu()-originalbackbone).abs().max());assert S['backbone_weight_change_max_abs']==0;meter.phase='saving';t=time.time();torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'epoch':100,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},M/'final_training_state.pt');torch.save(beststate,M/'best_head.pt');write_json(M/'task.json',{'input_dim':640,'labels':json.loads((R/'data/input_audit.json').read_text())['labels'],'backbone_checkpoint':'models/RNA-FM_pretrained.pth','pooling':S['settings']['pooling'],'selected_epoch':bestepoch,'resume_tested':False});S['save_seconds']=time.time()-t;head.load_state_dict(beststate);meter.phase='test';S['test'],logits=evaluate('test');np.save(D/'test_logits.npy',logits);write_json(D/'test_predictions.json',[{'id':r['id'],'label':r['label'],'prediction':int(v.argmax())} for r,v in zip(data['test'],logits)]);S['saved_head']=str((M/'best_head.pt').relative_to(R));S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
