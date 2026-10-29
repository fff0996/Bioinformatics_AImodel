from pathlib import Path
import os,sys,time,json,gc
import numpy as np,torch
sys.path[:0]=['/support','/work/source'];from gpu_support import Meter,write_json
from clair3.model import Clair3_P,Clair3_F
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];kind=os.environ['CANDIDATE_MODEL'];B=int(os.environ['BATCH']);C=D/f'{kind}_batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'batch':B,'task':'actual candidate tensor native model inference, excludes BAM preprocessing/VCF calling'}
try:
 torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='loading';t=time.time();model=(Clair3_P if kind=='Clair3_P' else Clair3_F)(predict=True,add_indel_length=kind=='Clair3_F').float().cuda().eval();p=R/'models/ont'/('pileup.pt' if kind=='Clair3_P' else 'full_alignment.pt');ck=torch.load(p,map_location='cpu',weights_only=False);model.load_state_dict(ck.get('state_dict',ck));del ck;torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']=sum(p.numel() for p in model.parameters());S['parameters_trained_in_run']=0
 x=np.load(D/(kind+'_inputs.npy'),mmap_mode='r');assert len(x)>=B;values=np.array(x[:B]);S['input_shape']=list(values.shape);S['input_dtype']=str(values.dtype);S['unique_candidates']=B
 def forward():
  with torch.inference_mode():return model(torch.from_numpy(values).cuda()).cpu().numpy()
 meter.phase='warmup';t=time.time();out=forward();torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del out;gc.collect();t,base=meter.begin('candidate_inference');repeats=0
 while repeats<2 or time.time()-t<15:out=forward();assert np.isfinite(out).all();torch.cuda.synchronize();repeats+=1
 r=meter.end(t,base,batch_size=B,unique_candidates=B,repeats=repeats,processed_candidates=B*repeats);r['candidates_per_second']=B*repeats/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];S['condition']=r;np.save(C/'predictions.npy',out);assert np.array_equal(np.load(C/'predictions.npy'),out);S['output_shape']=list(out.shape);ref=D/(kind+'_reference_first.npy')
 if ref.exists():S['batch_invariance_max_abs']=float(np.abs(np.load(ref)-out[0]).max());assert np.allclose(np.load(ref),out[0],atol=1e-4,rtol=1e-4)
 else:np.save(ref,out[0])
 S['finite_and_exact_save_reload']=True;S['status']='complete';write_json(C/'summary.json',S);print('BATCH_COMPLETE',kind,B,r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write_json(C/'summary.json',S);raise
finally:meter.close()
