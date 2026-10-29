from pathlib import Path
import os,sys,json,time,gc,importlib.metadata
import numpy as np,torch
from cellpose import models,core,transforms
sys.path.insert(0,'/shared');from gpu_support import Meter,write_json
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];B=int(os.environ['TILE_BATCH']);C=D/f'batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','task':'official run_net tiled-image network inference; excludes mask dynamics','requested_tile_batch':B}
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='model_loading';t=time.time();model=models.CellposeModel(gpu=True,pretrained_model=str(R/'models/cpsam_v2'),use_bfloat16=True);torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for p in model.net.parameters());S['parameters_trainable_in_run']=0;S['weights_dtype']=str(model.net.dtype);S['environment']={'cellpose':importlib.metadata.version('cellpose'),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)};write_json(C/'summary.json',S)
 source=np.load(D/'normalized_images.npy',mmap_mode='r');N=max(1,B//12);imgs=np.asarray(source[:N]);assert len(imgs)==N;actual=[];original=core._forward
 def forward(net,x):
  actual.append(list(x.shape));return original(net,x)
 core._forward=forward
 def run():
  flows,styles=core.run_net(model.net,imgs,batch_size=B,augment=False,tile_overlap=.1,bsize=256,rsz=None);assert flows.shape==(N,520,696,3) and np.isfinite(flows).all();return flows
 meter.phase='warmup';t=time.time();flow=run();torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;S['warmup_actual_forward_shapes']=actual[:];actual.clear();del flow;gc.collect();t,base=meter.begin('official_run_net_inference');steps=0
 while steps<2 or time.time()-t<15:flow=run();torch.cuda.synchronize();steps+=1
 r=meter.end(t,base,requested_tile_batch=B,unique_images=N,unique_tiles=N*12,repeats=steps,processed_images=N*steps,processed_tiles=N*12*steps);r['actual_forward_shapes']=actual;r['actual_max_tile_batch']=max(x[0] for x in actual);r['images_per_second']=N*steps/r['seconds'];r['tiles_per_second']=N*12*steps/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];S['condition']=r
 np.save(C/'first_image_flow_cellprob.npy',flow[0]);saved=np.load(C/'first_image_flow_cellprob.npy');assert np.array_equal(saved,flow[0]);S['output_saved_exact_reload']=True;S['all_output_shape']=[N,520,696,3];S['all_outputs_finite']=True
 if B==8:
  audit=json.loads((D/'input_audit.json').read_text());import tifffile
  raw=tifffile.imread(R/audit['images'][0]['source_image']);mask,fl,style=model.eval(raw,batch_size=8,bsize=256,tile_overlap=.1,augment=False,normalize=True,diameter=None,flow_threshold=.4,cellprob_threshold=0.,min_size=15,max_size_fraction=.4);reference=np.concatenate([np.moveaxis(fl[1],0,-1),fl[2][...,None]],axis=-1);diff=float(np.abs(reference-flow[0]).max());S['whole_image_eval_reference_max_abs_difference']=diff;assert np.allclose(reference,flow[0],atol=1e-5,rtol=1e-5);np.save(C/'reference_segmentation_mask.npy',mask);S['reference_mask_shape']=list(mask.shape)
 else:
  ref=np.load(D/'batch_8/first_image_flow_cellprob.npy');delta=np.abs(ref-flow[0]);S['batch_invariance']={'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'cosine':float(np.dot(ref.ravel(),flow[0].ravel())/(np.linalg.norm(ref)*np.linalg.norm(flow[0])))};assert S['batch_invariance']['cosine']>.99
 S['status']='complete';write_json(C/'summary.json',S);print('BATCH_COMPLETE',B,'actual',r['actual_max_tile_batch'],'peak',r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write_json(C/'summary.json',S);raise
finally:meter.close()
