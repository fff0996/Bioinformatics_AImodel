"""Actual batch capacity probe; fixed FP32/1200 cached gene tokens, fresh start per batch."""
from pathlib import Path
import os,time,json,gc,random,re
import numpy as np,torch,psutil
from scgpt.model import TransformerModel
from gpu_support import Meter,write_json
R=Path('/workspace/scgpt');D=R/'results'/os.environ['RUN_ID'];D.mkdir(exist_ok=True);source=R/'results'/(R/'results/SUPPLEMENT_LATEST').read_text().strip();T=source/'training_util';bundle=R/'models/finetuned'/source.name/'epoch_001';torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.manual_seed(42);np.random.seed(42);random.seed(42);meter=Meter(D);S={'status':'running','experiment':'followup_batch_memory_capacity','conditions':[],'source_basic_run':source.name,'settings':{'mode':'training_resource_probe','dtype':'FP32','tf32':False,'max_length_including_CLS':1200,'optimizer':'AdamW lr1e-4 wd.01 betas.9/.999 eps1e-8 clip1.0','initialization':'same pretrained encoder and seed42 fresh13-class classifier restored before every batch','min_seconds_per_condition':15,'min_steps':3,'target_allocated_fraction_of_shared_RAM':[.8,.9],'minimum_system_available_gib':8,'swap_increase_stop_gib':2,'batch_only_variable':True,'quality_training':'not an accuracy experiment; updated probe weights saved separately','batch_repetition':'same nested firstB real training cells across steps, no replicated cells inside a batch'}}
def save():write_json(D/'summary.json',S)
try:
 config=json.loads((bundle/'model_config.json').read_text());vocab=json.loads((bundle/'vocab.json').read_text());model=TransformerModel(**config,vocab=vocab);raw=torch.load(R/'models/whole-human/best_model.pt',map_location='cpu',weights_only=True);mapped={}
 for key,value in raw.items():
  for a,b in [(r'self_attn\._impl\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\._impl\.out_proj\.','self_attn.out_proj.')]:key=re.sub(a,b,key)
  mapped[key]=value
 current=model.state_dict();core=('encoder.','value_encoder.','transformer_encoder.');required=[k for k in current if k.startswith(core)];loaded={k:mapped[k] for k in required if k in mapped and mapped[k].shape==current[k].shape};assert len(loaded)==len(required);model.load_state_dict(loaded,strict=False);del raw,mapped,current,loaded
 for name,p in model.named_parameters():p.requires_grad_(name.startswith(core+('cls_decoder.',)))
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.float().cuda();torch.cuda.synchronize();S['model_only_postload']=meter.memory();S['parameters']={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)};cache=np.load(T/'epoch_001/training_tokenized_inputs.npz');genes=cache['gene_ids'];values=cache['expression_bins'];labels=cache['labels'];assert genes.shape==(1000,1200) and values.shape==genes.shape and len(labels)==1000;pad=vocab[config['pad_token']];S['input_audit']={'unique_available_cells':1000,'gene_values_shape':list(genes.shape),'labels':{str(int(k)):int(n) for k,n in zip(*np.unique(labels,return_counts=True))},'source':str(T/'epoch_001/training_tokenized_inputs.npz'),'token_min':int((genes!=pad).sum(1).min()),'token_max':int((genes!=pad).sum(1).max()),'cache_is_real_training_cells':True};write_json(D/'model_config.json',config);(D/'probe_models').mkdir(exist_ok=True);torch.save(initial,D/'initial_state.pt');save()
 batches=[8,16,32,48,64,96,128,160,192,224,256,288,320,384,448,512,640,768,896,1000];stop_reason='available_unique_input_limit';previous=None
 for proposed in batches:
  model.zero_grad(set_to_none=True);model.cur_gene_token_embs=None;gc.collect();torch.cuda.empty_cache();budget=meter.memory()['reserved_gib']+psutil.virtual_memory().available/2**30-8
  b=proposed
  if previous:
   projected=previous['memory']['peak_reserved_gib']*b/previous['batch_size']*1.08
   if projected>budget:
    cap=int(previous['batch_size']*budget/(previous['memory']['peak_reserved_gib']*1.08));cap=min(b,cap)
    if cap<=previous['batch_size']+max(1,previous['batch_size']//30):stop_reason='predicted_reservation_would_violate_8GiB_available_guard';S['stopped_before_batch']={'proposed':b,'projected_reserved_gib':projected,'reservation_budget_gib':budget,'safe_cap_estimate':cap};save();break
    b=cap;S['next_batch_capped_by_memory_prediction']={'proposed':proposed,'tested':b,'reservation_budget_gib':budget}
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);model.load_state_dict(initial);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8);torch.cuda.synchronize();t,base=meter.begin(f'train_batch{b}');steps=0;losses=[];print('START_BATCH',b,'availableGiB',base['system_available_gib'],flush=True);save()
  try:
   while steps<3 or time.time()-t<15:
    opt.zero_grad(set_to_none=True);g=torch.from_numpy(genes[:b].copy()).cuda();v=torch.from_numpy(values[:b].copy()).cuda();y=torch.from_numpy(labels[:b].copy()).cuda();h=model._encode(g,v,src_key_padding_mask=g.eq(pad));logits=model.cls_decoder(h[:,0,:]);loss=torch.nn.functional.cross_entropy(logits,y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();losses.append(float(loss.detach()));steps+=1;del g,v,y,h,logits,loss
   row=meter.end(t,base,name=f'train_batch{b}',batch_size=b,unique_inputs=b,steps=steps,processed=b*steps,loss_first=losses[0],loss_last=losses[-1],actual_input_shape=[b,1200],actual_mean_nonpad_tokens=float((genes[:b]!=pad).sum(1).mean()));row['throughput_cells_per_second']=row['processed']/row['seconds'];row['allocated_fraction_of_shared_RAM']=row['memory']['peak_allocated_gib']/row['memory']['system_total_gib'];row['reserved_fraction_of_shared_RAM']=row['memory']['peak_reserved_gib']/row['memory']['system_total_gib'];row['reached_80pct_allocated']=row['allocated_fraction_of_shared_RAM']>=.8;row['reached_90pct_allocated']=row['allocated_fraction_of_shared_RAM']>=.9;S['conditions'].append(row);print('RESULT',json.dumps(row),flush=True);save();meter.phase='save_probe';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},D/'probe_models'/f'batch_{b}.pt');previous=row
  except torch.cuda.OutOfMemoryError as e:
   S['oom']={'batch':b,'error':str(e)};stop_reason='CUDA_OOM';save();break
  finally:
   opt.zero_grad(set_to_none=True);del opt;model.zero_grad(set_to_none=True);model.cur_gene_token_embs=None;gc.collect();torch.cuda.empty_cache()
  if row['reached_90pct_allocated']:stop_reason='90pct_allocated_target_observed';break
  if b<proposed:stop_reason='tested_predicted_safe_cap';break
 S['status']='complete';S['stop_reason']=stop_reason;S['largest_successful_batch']=max(r['batch_size'] for r in S['conditions']);S['maximum_peak_allocated_gib']=max(r['memory']['peak_allocated_gib'] for r in S['conditions']);S['reached_80pct_allocated']=any(r['reached_80pct_allocated'] for r in S['conditions']);S['reached_90pct_allocated']=any(r['reached_90pct_allocated'] for r in S['conditions']);save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
