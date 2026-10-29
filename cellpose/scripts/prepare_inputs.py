import json, hashlib
from pathlib import Path
import numpy as np
import tifffile
from skimage import io, measure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/work'); out=R/'data/prepared'; out.mkdir(exist_ok=True)
rng=np.random.default_rng(42); records=[]; selected={}
for split,n in [('training',16),('validation',8),('test',16)]:
 names=(R/f'data/metadata/metadata/{split}.txt').read_text().splitlines()
 chosen=set(rng.choice(sorted(names),n,replace=False)); selected[split]=[]
 for name in names:
  stem=Path(name).stem; ip=R/f'data/images/images/{stem}.tif'; mp=R/f'data/masks/masks/{name}'
  x=tifffile.imread(ip); rgb=io.imread(mp); mask=measure.label(rgb[:,:,0],background=0).astype(np.uint16)
  assert x.shape==mask.shape and x.ndim==2 and np.isfinite(x).all(), (name,x.shape,mask.shape)
  rec={'id':stem,'split':split,'image':str(ip.relative_to(R)),'source_mask':str(mp.relative_to(R)),'shape':list(x.shape),'dtype':str(x.dtype),'min':int(x.min()),'max':int(x.max()),'instances':int(mask.max()),'image_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),'selected':name in chosen}
  if name in chosen:
   dest=out/f'{stem}_mask.npy'; np.save(dest,mask); rec['mask']=str(dest.relative_to(R)); selected[split].append(rec)
  records.append(rec)
assert len({r['id'] for r in records})==len(records)
(R/'data/input_audit.json').write_text(json.dumps({'all':records,'selected':selected,'seed':42,'pixel_size_um':None,'mask_decode':'skimage.measure.label(PNG first channel, background=0), official BBBC039 decoding','channels':'raw grayscale; official Cellpose conversion adds two zero channels','split_policy':'official original-image split, deterministic subset before crops'},indent=2))
a=selected['training'][0]; x=tifffile.imread(R/a['image']); m=np.load(R/a['mask']); fig,ax=plt.subplots(1,3,figsize=(13,4)); ax[0].imshow(x,cmap='gray',vmin=np.percentile(x,1),vmax=np.percentile(x,99)); ax[0].set_title(f'Raw {x.shape} {x.dtype}'); ax[1].imshow(m,cmap='nipy_spectral'); ax[1].set_title(f'Ground truth: {m.max()} nuclei'); ax[2].imshow(x,cmap='gray',vmin=np.percentile(x,1),vmax=np.percentile(x,99)); ax[2].contour(m>0,levels=[.5],colors='red',linewidths=.4); ax[2].set_title('Raw + ground-truth boundary'); [a.axis('off') for a in ax]; fig.tight_layout(); fig.savefig(R/'reports/input_example.png',dpi=150); plt.close(fig)
print(json.dumps({'total':len(records),'source_split_counts':{s:sum(r['split']==s for r in records) for s in selected},'selected':{s:len(v) for s,v in selected.items()},'example':a},indent=2))
