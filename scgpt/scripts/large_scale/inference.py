from pathlib import Path
import os,time,json,gc,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scgpt.model import TransformerModel
from gpu_support import Meter,write_json
R=Path('/workspace/scgpt');P=R/'data_external/tabula_sapiens/prepared_300k_seed42';D=R/'results'/os.environ['RUN_ID'];D.mkdir(exist_ok=True);torch.set_num_threads(8);torch.manual_seed(42);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;meter=Meter(D);S={'status':'running','dataset':'Tabula Sapiens actual human cells, Census2024-07-01','conditions':[],'settings':{'batches':[8,256],'dtype':'FP32','tf32':False,'seed':42,'max_length':1200,'padding':'actual batch maximum cached nonpad length','task':'pretrained whole-human normalized CLS embeddings, no cell-type classifier and no test accuracy claim','count_points':[10000,100000,300000],'measurement':'cumulative nested prefix endpoints within one300k pass per batch, includes cached batch construction/transfers/forward/CPU outputs/memmap writes','excludes':'download, h5ad audit, tokenization cache, model load, warmup','input_policy':'same cached tokens/values and same original cells/order for both batch settings'}}
def save():write_json(D/'summary.json',S)
try:
 audit=json.loads((P/'input_audit.json').read_text());assert audit['selected_unique_cells']==300000;S['input_audit']=str(P/'input_audit.json');gcache=np.load(P/'gene_ids.npy',mmap_mode='r');vcache=np.load(P/'expression_bins.npy',mmap_mode='r');lengths=np.load(P/'lengths.npy',mmap_mode='r');source=R/'results'/(R/'results/SUPPLEMENT_LATEST').read_text().strip();bundle=R/'models/finetuned'/source.name/'epoch_001';vocab=json.loads((bundle/'vocab.json').read_text());config=json.loads((bundle/'model_config.json').read_text());config['n_cls']=1;config['do_mvc']=True;meter.phase='model_loading';t=time.time();model=TransformerModel(**config,vocab=vocab);state=torch.load(R/'models/whole-human/best_model.pt',map_location='cpu',weights_only=True);mapped={}
 for key,value in state.items():
  for a,b in [(r'self_attn\._impl\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\._impl\.out_proj\.','self_attn.out_proj.')]:key=re.sub(a,b,key)
  mapped[key]=value
 current=model.state_dict();compatible={k:v for k,v in mapped.items() if k in current and current[k].shape==v.shape};required=[k for k in current if k.startswith(('encoder.','value_encoder.','transformer_encoder.'))];assert set(required)<=set(compatible);model.load_state_dict(compatible,strict=False);del state,mapped,current,compatible;gc.collect();model.float().cuda().eval();torch.cuda.synchronize();S['model_loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']=sum(p.numel() for p in model.parameters());S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'scgpt':__import__('scgpt').__version__,'gpu':torch.cuda.get_device_name(0)};pad=vocab[config['pad_token']];save()
 def forward(start,end):
  width=int(lengths[start:end].max());g=torch.from_numpy(np.array(gcache[start:end,:width],dtype=np.int64)).cuda();v=torch.from_numpy(np.array(vcache[start:end,:width],dtype=np.float32)).cuda();h=model._encode(g,v,src_key_padding_mask=g.eq(pad));z=torch.nn.functional.normalize(h[:,0,:],p=2,dim=1).cpu().numpy();assert np.isfinite(z).all();return z,width
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():z,_=forward(0,8)
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del z;model.cur_gene_token_embs=None
 for batch in [8,256]:
  target=D/f'embeddings_batch{batch}.npy';emb=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(300000,512));model.cur_gene_token_embs=None;gc.collect();t,base=meter.begin(f'actual_cells_batch{batch}');start=0;shapes={};log_next=4096;paused=0.
  with torch.inference_mode():
   for endpoint in [10000,100000,300000]:
    while start<endpoint:
     end=min(start+batch,endpoint);z,width=forward(start,end);emb[start:end]=z;shapes[f'{end-start}x{width}']=shapes.get(f'{end-start}x{width}',0)+1;start=end
     if start>=log_next:print('REAL_CELLS',batch,start,'/',300000,'seconds',round(time.time()-t-paused,1),flush=True);log_next=start+4096
    r=meter.end(t,base,name=f'actual_cells_n{endpoint}_batch{batch}',batch_size=batch,unique_cells=endpoint,processed_cells=endpoint,measurement_repeats=1,actual_token_shapes=dict(shapes));r['seconds']-=paused;r['cells_per_second']=endpoint/r['seconds'];S['conditions'].append(r);print('PREFIX_COMPLETE',json.dumps({k:v for k,v in r.items() if k not in ['actual_token_shapes','memory','baseline']}),flush=True);p=time.time();save();emb.flush();paused+=time.time()-p
  emb.flush();del emb;model.cur_gene_token_embs=None;gc.collect();torch.cuda.empty_cache()
 a=np.load(D/'embeddings_batch8.npy',mmap_mode='r');b=np.load(D/'embeddings_batch256.npy',mmap_mode='r');ix=np.random.default_rng(42).choice(300000,256,replace=False);x=np.array(a[ix]);y=np.array(b[ix]);S['output_verification']={'sampled_actual_cells':256,'max_abs_embedding_difference_between_batches':float(np.abs(x-y).max()),'minimum_cosine_similarity':float((x*y).sum(1).min()),'output_shape':[300000,512],'all_outputs_finite_checked_during_forward':True};assert S['output_verification']['minimum_cosine_similarity']>.999;S['status']='complete';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
