from pathlib import Path
import json,os
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
R=Path('/work');D=Path(os.environ['BRAIN_RUN_DIR']);S=json.loads((D/'summary.json').read_text());rows=[r for r in S['input_audit']['rows'] if r['split']=='val'];fig,axes=plt.subplots(4,4,figsize=(13,13))
for i,r in enumerate(rows):
 x=np.load(D/(r['id']+'_2d_image.npy'));gt=np.load(D/(r['id']+'_2d_gt.npy'));p2=np.load(D/(r['id']+'_2d_mask.npy'));p3path=D/(r['id']+'_3d_mask.npy');p3=np.load(p3path,mmap_mode='r')[r['shape'][0]//2] if p3path.exists() else None
 for j,(title,mask,color) in enumerate([('raw micro-CT',None,None),('GT nuclei',gt,'red'),('Cellpose 2D',p2,'lime'),('Cellpose 3D central slice',p3,'cyan')]):
  ax=axes[i,j];ax.imshow(x,cmap='gray')
  if mask is not None:
   boundary=find_boundaries(mask,mode='outer');overlay=np.zeros((*mask.shape,4));rgb=matplotlib.colors.to_rgb(color);overlay[boundary,:3]=rgb;overlay[boundary,3]=.9;ax.imshow(overlay)
  ax.set_title(title+('\n'+r['id'] if j==0 else ''),fontsize=9);ax.axis('off')
fig.tight_layout();fig.savefig(D/'brain_prediction_examples.png',dpi=140)
