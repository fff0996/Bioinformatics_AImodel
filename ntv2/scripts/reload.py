from pathlib import Path
import os,json,sys,time
sys.path.insert(0,'/support')
from gpu_support import write_json
import torch,numpy as np
from transformers import AutoModelForSequenceClassification,AutoTokenizer
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());p=R/s['saved_model'];torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;m=AutoModelForSequenceClassification.from_pretrained(p,trust_remote_code=True,local_files_only=True).float().cuda().eval();tok=AutoTokenizer.from_pretrained(p,trust_remote_code=True,local_files_only=True);rows=json.loads((R/'data/inputs.json').read_text())['test'];pred=[]
with torch.inference_mode():
 for i in range(0,len(rows),8):
  x=tok([r['sequence'] for r in rows[i:i+8]],padding=True,truncation=False,return_tensors='pt');pred.append(m(**{k:x[k].cuda() for k in ['input_ids','attention_mask']}).logits.cpu().numpy())
y=np.concatenate(pred);ref=np.load(D/'test_logits.npy');diff=float(np.abs(y-ref).max());r={'independent_process':True,'test_inputs':len(rows),'max_abs_difference':diff,'predictions_exact_match':bool(np.array_equal(y.argmax(-1),ref.argmax(-1))),'passed':bool(diff<=1e-5)};write_json(D/'reload.json',r);assert r['passed'];s['reload']=r;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(r),flush=True)
