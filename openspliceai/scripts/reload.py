from pathlib import Path
import os,sys,json
sys.path.insert(0,'/support')
from gpu_support import write_json
import numpy as np,torch
from openspliceai.train_base.openspliceai import SpliceAI
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);m=SpliceAI(32,W,AR).float().cuda().eval();m.load_state_dict(torch.load(R/s['saved_model'],map_location='cuda',weights_only=True));data=json.loads((R/'data/inputs.json').read_text())['test'];tab=np.zeros((256,4),np.float32)
for j,c in enumerate('ACGT'):tab[ord(c),j]=1
pred=[]
with torch.no_grad():
 for i in range(0,len(data),12):
  x=np.stack([tab[np.frombuffer(r['sequence'][2500:17500].encode(),np.uint8)].T for r in data[i:i+12]]);pred.append(m(torch.from_numpy(x).cuda()).cpu().numpy())
y=np.concatenate(pred);ref=np.load(D/'test_probabilities.npy');r={'independent_process':True,'test_windows':len(data),'input_shape':[12,4,15000],'max_abs_difference':float(np.abs(y-ref).max()),'all_base_classes_match':bool(np.array_equal(y.argmax(1),ref.argmax(1)))};r['passed']=r['max_abs_difference']<=1e-5 and r['all_base_classes_match'];write_json(D/'reload.json',r);assert r['passed'];s['reload']=r;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(r),flush=True)
