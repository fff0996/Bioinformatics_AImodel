from pathlib import Path
import os,sys,time,json,gc,hashlib
import numpy as np,torch
from cellpose import models,core,metrics
sys.path.insert(0,'/support');from gpu_support import Meter,write_json
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];audit=json.loads((R/'data/brain_nucmm/input_audit.json').read_text());meter=Meter(D);S={'status':'running','run_id':D.name,'input_audit':audit,'conditions':[],'settings':{'checkpoint':'cpsam_v2','precision':'BF16 model, native preprocessing/flows float32','seed':42,'tile_batch':8,'tile_size':256,'tile_overlap':.1,'augment':False,'rescale':None,'diameter':None,'normalize':True,'invert':audit['polarity_policy']['invert'],'inversion_selection':'training label intensity contrast only; validation not used for choice','flow_threshold':.4,'cellprob_threshold':0.,'min_size':15,'max_size_fraction':.4,'anisotropy':1.0,'anisotropy_note':'compute assumes equal voxel spacing; physical calibration not verified','training':'none, zero-shot inference with existing weights','validation':'original4val patches; same parent brain, no held-out donor or independent test'} }
frames=[];stage_times=[]
def save():write_json(D/'summary.json',S)
def data(row):
 P=R/'data/brain_nucmm/prepared'/row['split'];return np.load(P/(row['id']+'_image.npy')),np.load(P/(row['id']+'_label.npy'))
def relabel(x):
 ids,inv=np.unique(x,return_inverse=True);assert ids[0]==0;return inv.reshape(x.shape).astype(np.int32),ids.tolist()
def score(gt,pred):
 g,ids=relabel(gt);p,pids=relabel(pred);ap,tp,fp,fn=metrics.average_precision([g],[p],threshold=[.5,.75,.9]);return {'AP50_AP75_AP90':ap[0].tolist(),'TP':tp[0].tolist(),'FP':fp[0].tolist(),'FN':fn[0].tolist(),'gt_instances':len(ids)-1,'predicted_instances':len(pids)-1,'gt_original_label_ids':ids}
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='model_loading';t=time.time();model=models.CellposeModel(gpu=True,pretrained_model=str(R/'models/cpsam_v2'),use_bfloat16=True);torch.cuda.synchronize();S['model_loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['checkpoint_sha256']=hashlib.file_digest((R/'models/cpsam_v2').open('rb'),'sha256').hexdigest();S['parameters_total']=sum(p.numel() for p in model.net.parameters());S['parameters_trained_in_run']=0;S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'cellpose':__import__('importlib.metadata',fromlist=['version']).version('cellpose'),'gpu':torch.cuda.get_device_name(0)};save()
 oldforward=core._forward
 def forward(net,x):
  torch.cuda.synchronize();t=time.time();y=oldforward(net,x);torch.cuda.synchronize();frames.append({'phase':meter.phase,'shape':list(x.shape),'seconds':time.time()-t,'weights_dtype':str(net.dtype)});return y
 core._forward=forward
 for method in ['_run_net','_compute_masks']:
  original=getattr(model,method)
  def wrapper(*args,_method=method,_fn=original,**kw):
   torch.cuda.synchronize();t=time.time();res=_fn(*args,**kw);torch.cuda.synchronize();stage_times.append({'phase':meter.phase,'stage':_method,'seconds':time.time()-t});return res
  setattr(model,method,wrapper)
 opts={'batch_size':8,'bsize':256,'tile_overlap':.1,'augment':False,'normalize':{'normalize':True,'invert':audit['polarity_policy']['invert']},'diameter':None,'flow_threshold':.4,'cellprob_threshold':0.,'min_size':15,'max_size_fraction':.4}
 trainrow=next(r for r in audit['rows'] if r['split']=='train');x,y=data(trainrow);meter.phase='warmup_2d_train_slice';t=time.time();model.eval(x[x.shape[0]//2],do_3D=False,**opts);torch.cuda.synchronize();S['warmup_2d_seconds']=time.time()-t;S['warmup_note']='one train central slice only; first3D includes normal first-use setup';del x,y;save();vals=[r for r in audit['rows'] if r['split']=='val'];images=[];labels=[]
 for row in vals:
  x,y=data(row);images.append(x[x.shape[0]//2]);labels.append(y[y.shape[0]//2])
 startframes=len(frames);startstages=len(stage_times);t,base=meter.begin('2d_central_validation_slices');repeats=0
 while repeats<2 or time.time()-t<15:
  out=model.eval(images,do_3D=False,**opts);torch.cuda.synchronize();repeats+=1
 end=meter.end(t,base,name='2d_central_validation_slices',mode='2d',unique_images=4,repeats=repeats,processed_images=4*repeats);end['images_per_second']=4*repeats/end['seconds'];end['actual_forward_shapes']=sorted({str(f['shape']) for f in frames[startframes:]});end['processed_tiles']=sum(f['shape'][0] for f in frames[startframes:]);end['network_stage_seconds']=sum(z['seconds'] for z in stage_times[startstages:] if z['stage']=='_run_net');end['mask_stage_seconds']=sum(z['seconds'] for z in stage_times[startstages:] if z['stage']=='_compute_masks');end['scores']=[]
 for i,row in enumerate(vals):
  pred=out[0][i];assert pred.shape==images[i].shape and np.isfinite(pred).all();np.save(D/(row['id']+'_2d_mask.npy'),pred);np.save(D/(row['id']+'_2d_gt.npy'),labels[i]);np.save(D/(row['id']+'_2d_image.npy'),images[i]);np.savez_compressed(D/(row['id']+'_2d_flows.npz'),dP=out[1][i][1],cellprob=out[1][i][2]);assert np.array_equal(np.load(D/(row['id']+'_2d_mask.npy')),pred);end['scores'].append({'id':row['id'],**score(labels[i],pred)})
 S['conditions'].append(end);save();print('BRAIN_2D_COMPLETE',end['seconds'],flush=True)
 for row in vals:
  x,y=data(row);name='3d_'+row['id'];startframes=len(frames);startstages=len(stage_times);t,base=meter.begin(name);out=model.eval(x[...,None],channel_axis=3,z_axis=0,do_3D=True,anisotropy=1.,**opts);torch.cuda.synchronize();res=meter.end(t,base,name=name,mode='3d',unique_volumes=1,original_volume_shape=list(x.shape),original_slices=int(x.shape[0]),voxels=int(x.size),repeats=1);pred=out[0];assert pred.shape==x.shape and np.isfinite(pred).all();res['voxels_per_second']=x.size/res['seconds'];res['actual_forward_shapes']=sorted({str(f['shape']) for f in frames[startframes:]});res['processed_tiles']=sum(f['shape'][0] for f in frames[startframes:]);res['forward_calls']=len(frames[startframes:]);res['network_stage_seconds']=sum(z['seconds'] for z in stage_times[startstages:] if z['stage']=='_run_net');res['mask_stage_seconds']=sum(z['seconds'] for z in stage_times[startstages:] if z['stage']=='_compute_masks');res['scores']=[{'id':row['id'],**score(y,pred)}];np.save(D/(row['id']+'_3d_mask.npy'),pred);np.savez_compressed(D/(row['id']+'_3d_flows.npz'),dP=out[1][1],cellprob=out[1][2]);assert np.array_equal(np.load(D/(row['id']+'_3d_mask.npy')),pred);S['conditions'].append(res);save();print('BRAIN_3D_COMPLETE',row['id'],res['seconds'],res['scores'][0]['AP50_AP75_AP90'],flush=True);del x,y,out,pred;gc.collect();torch.cuda.empty_cache()
 S['status']='complete';S['saved_predictions_reloaded_exactly']=True;save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:write_json(D/'forward_calls.json',frames);write_json(D/'stage_times.json',stage_times);meter.close()
