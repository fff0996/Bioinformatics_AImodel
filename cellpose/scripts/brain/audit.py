from pathlib import Path
import json,hashlib
import h5py,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/work/data/brain_nucmm');manifest=json.loads((R/'download_manifest.json').read_text());rows=[];inside=[];outside=[];previews=[]
for split in ['train','val']:
 for path in sorted((R/'Image'/split).glob('*.h5')):
  lab=R/'Label'/split/path.name.replace('img_','seg_',1)
  with h5py.File(path,'r') as f:keys=list(f);assert keys==['main'];x=f['main'][:]
  with h5py.File(lab,'r') as f:y=f['main'][:]
  assert x.shape==y.shape and x.ndim==3 and np.isfinite(x).all() and np.isfinite(y).all() and y.min()==0 and np.issubdtype(y.dtype,np.integer);ids=np.unique(y);assert len(ids)>1
  prepared=R/'prepared'/split;prepared.mkdir(parents=True,exist_ok=True);np.save(prepared/(path.stem+'_image.npy'),x);np.save(prepared/(path.stem+'_label.npy'),y)
  row={'id':path.stem,'split':split,'image':str(path.relative_to(Path('/work'))),'label':str(lab.relative_to(Path('/work'))),'shape':list(x.shape),'image_dtype':str(x.dtype),'label_dtype':str(y.dtype),'image_min':float(x.min()),'image_max':float(x.max()),'nuclei_instances_in_3d_patch':int(len(ids)-1),'max_original_instance_id':int(y.max()),'image_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'mask_sha256':hashlib.file_digest(lab.open('rb'),'sha256').hexdigest(),'foreground_mean_intensity':float(x[y>0].mean()),'background_mean_intensity':float(x[y==0].mean())};rows.append(row)
  if split=='train':inside.append(row['foreground_mean_intensity']);outside.append(row['background_mean_intensity'])
  if len(previews)<2 or (split=='val' and len(previews)<3):previews.append((row['id'],x[x.shape[0]//2].copy(),y[y.shape[0]//2].copy()))
assert len(rows)==8 and len({r['id'] for r in rows})==8
invert=bool(np.mean(inside)<np.mean(outside));audit={'dataset':'NucMM-M public train/val patches, mouse visual cortex micro-CT','source_manifest':'data/brain_nucmm/download_manifest.json','actual_unique_3d_patches':len(rows),'actual_2d_slices':sum(r['shape'][0] for r in rows),'selected_train_volumes':4,'selected_validation_volumes':4,'independence':'all patches from same parent mouse brain volume; no independent donor/test cohort; pretrained training overlap not known','axes':'HDF5 arrays used as Z/Y/X for compute; anatomical orientation not verified','voxel_size_um':None,'modality_resolution':'primary project and original release text specify micro-CT, contradicting mirror README EM claim','polarity_policy':{'invert':invert,'selection_data':'train labels only; no val-based choice','train_foreground_mean':float(np.mean(inside)),'train_background_mean':float(np.mean(outside))},'rows':rows};(R/'input_audit.json').write_text(json.dumps(audit,indent=2))
fig,axes=plt.subplots(len(previews),2,figsize=(8,4*len(previews)))
for i,(name,x,y) in enumerate(previews):
 axes[i,0].imshow(x,cmap='gray');axes[i,0].set_title(name+' raw central slice');axes[i,1].imshow(x,cmap='gray');axes[i,1].contour(y>0,levels=[.5],colors='red',linewidths=.6);axes[i,1].set_title('original nuclear annotation');axes[i,0].axis('off');axes[i,1].axis('off')
fig.tight_layout();fig.savefig(R/'input_preview.png',dpi=120);print(json.dumps({'shapes':[r['shape'] for r in rows],'invert':invert,'train_fg':float(np.mean(inside)),'train_bg':float(np.mean(outside))},indent=2))
