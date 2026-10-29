from pathlib import Path
import sys,json,os,time,gc
from gpu_support import Meter,write_json
import numpy as np,torch
R=Path('/workspace/scgpt');D=R/'results'/os.environ['RUN_ID'];meter=Meter(D);S={'status':'running','conditions':[],'settings':{'batch':8,'max_length':1200,'dtype':'FP32','tf32':False,'seed':42,'inference':'same native pretrained whole-human encoder; CPU cache tokenization once, normalized CLS embeddings','input_count':'nested first128/1024/all4218 test cells, same cached genes/values','length':'same128cells, first256/512/1024/1200 tokens of once-sampled1200 token cache; nested controlled supplement, differs from original independent random sampling by length'}}
def save():write_json(D/'summary.json',S)
try:
 source=R/'scripts/01_inference.py';prefix=source.read_text().split('# Warm the CUDA execution path, then reset random seeds before the measured run.')[0];sys.argv=[str(source),'--split','test','--n-cells','0','--batch-size','8','--max-length','1200','--out',str(D/'setup')];meter.phase='setup';ns={'__file__':str(source),'__name__':'__scgpt_setup__'};exec(compile(prefix,str(source),'exec'),ns);model=ns['model'];S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'scgpt':__import__('scgpt').__version__};S['parameters']={'total':ns['parameter_count_total'],'encoder':ns['parameter_count_encoder']};S['model_only_postload']=meter.memory();S['model_loading_seconds']=ns['model_loading_seconds'];S['input_preparation_seconds']=ns['input_preparation_seconds'];save()
 meter.phase='tokenize_cache';t=time.time();n=ns['n'];pad=ns['pad_id'];genes=np.full((n,1200),pad,np.int64);values=np.full((n,1200),ns['pad_value'],np.float32);ns['seed_all'](42)
 for i in range(0,n,8):
  b=ns['make_batch'](i,min(i+8,n));l=b['gene'].shape[1];genes[i:i+len(b['gene']),:l]=b['gene'].numpy();values[i:i+len(b['gene']),:l]=b['expr'].numpy()
 S['tokenization_cache_seconds']=time.time()-t;np.savez_compressed(D/'tokenized_all.npz',gene_ids=genes,expression_bins=values);ns['cell_metadata'].to_csv(D/'input_cells.tsv',sep='\t');np.save(D/'raw_first_cell_expression.npy',ns['matrix'][0]);S['input_audit']={'cells':n,'source_genes':3000,'matched_genes':2999,'matrix_dtype':'float32','tokens_min':int((genes!=pad).sum(1).min()),'tokens_mean':float((genes!=pad).sum(1).mean()),'tokens_max':int((genes!=pad).sum(1).max()),'raw_count_provenance':'not verified, values used as supplied, official51-bin preprocessing','labels':'Celltype metadata not provided to embedding encoder'};save()
 def batch(i,end,length):return {'gene':torch.from_numpy(genes[i:end,:length].copy()),'expr':torch.from_numpy(values[i:end,:length].copy())}
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():z=ns['encode_batch'](batch(0,8,1200))
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;np.save(D/'example_embeddings.npy',z);model.cur_gene_token_embs=None
 for nsel,length in [(128,1200),(1024,1200),(4218,1200),(128,256),(128,512),(128,1024)]:
  model.cur_gene_token_embs=None;gc.collect();t,base=meter.begin(f'inference_n{nsel}_cap{length}');repeats=0
  with torch.inference_mode():
   while repeats==0 or time.time()-t<15:
    for i in range(0,nsel,8):z=ns['encode_batch'](batch(i,min(i+8,nsel),length))
    torch.cuda.synchronize();repeats+=1
  counts=(genes[:nsel,:length]!=pad).sum(1);r=meter.end(t,base,name=f'inference_n{nsel}_cap{length}',mode='inference',length=length,unique_inputs=nsel,repeats=repeats,processed=nsel*repeats,tokens_min=int(counts.min()),tokens_mean=float(counts.mean()),tokens_max=int(counts.max()));r['throughput']=r['processed']/r['seconds'];S['conditions'].append(r);print('CONDITION',r['name'],r['seconds'],r['gpu_util_avg'],r['memory']['peak_allocated_gib'],flush=True);save()
 S['status']='inference_complete';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
