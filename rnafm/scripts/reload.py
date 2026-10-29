from pathlib import Path
import sys,json,os,argparse
sys.path.insert(0,'/support')
from gpu_support import write_json
import torch,numpy as np,fm
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());torch.serialization.add_safe_globals([argparse.Namespace]);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');model=model.float().cuda().eval();convert=alphabet.get_batch_converter();head=torch.nn.Linear(640,28).cuda();head.load_state_dict(torch.load(R/s['saved_head'],weights_only=True,map_location='cuda'));head.eval();rows=json.loads((R/'data/inputs.json').read_text())['test'];parts=[]
with torch.no_grad():
 for i in range(0,len(rows),50):
  rs=rows[i:i+50];_,_,tokens=convert([(r['id'],r['sequence']) for r in rs]);h=model(tokens.cuda(),repr_layers=[12],return_contacts=False)['representations'][12];z=torch.stack([h[j,1:len(r['sequence'])+1].mean(0) for j,r in enumerate(rs)]);parts.append(z.cpu().numpy())
 X=np.concatenate(parts);refX=np.load(D/'test_embeddings.npy');pred=[]
with torch.no_grad():
 for i in range(0,len(X),8):pred.append(head(torch.from_numpy(X[i:i+8]).cuda()).cpu().numpy())
y=np.concatenate(pred);ref=np.load(D/'test_logits.npy');r={'independent_process':True,'inputs':len(rows),'embedding_max_abs_difference':float(np.abs(X-refX).max()),'logits_max_abs_difference':float(np.abs(y-ref).max()),'predictions_match':bool(np.array_equal(y.argmax(1),ref.argmax(1)))};r['passed']=r['embedding_max_abs_difference']<=1e-5 and r['logits_max_abs_difference']<=1e-5 and r['predictions_match'];write_json(D/'reload.json',r);assert r['passed'];s['reload']=r;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(r),flush=True)
