from pathlib import Path
import sys,os,json,gc
sys.path[:0]=['/support','/work/source']
from gpu_support import write_json
import numpy as np,torch
from clair3.model import Clair3_P,Clair3_F
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());torch.set_num_threads(1);results=[];tested=set()
for row in s['conditions']:
 dest=D/row['name']/'instrumentation'
 for file in dest.glob('worker_*.jsonl'):
  events=[json.loads(l) for l in file.read_text().splitlines()];load=next((e for e in events if e['event']=='model_only_postload'),None);sample=next((e for e in events if e['event']=='reload_sample'),None)
  if not load or not sample or load['model'] in tested:continue
  cl=load['model'];p=load['checkpoint'];p=p if p.endswith('.pt') else p+'.pt';v=np.load(dest/sample['file']);m=(Clair3_P if cl=='Clair3_P' else Clair3_F)(predict=True,add_indel_length=cl=='Clair3_F').cuda().eval();checkpoint=torch.load(p,map_location='cuda');m.load_state_dict(checkpoint.get('state_dict',checkpoint));torch.backends.cuda.matmul.allow_tf32=load['tf32']
  full_shape=next(e['shape'] for e in events if e['event']=='forward');x=np.zeros(full_shape,dtype=v['X'].dtype);x[:len(v['X'])]=v['X']
  with torch.inference_mode():y=m(torch.from_numpy(x).cuda()).cpu().numpy()[:len(v['X'])]
  diff=float(np.abs(y-v['Y']).max());r={'model':cl,'actual_input_shape':list(v['X'].shape),'independent_process':True,'max_abs_difference':diff,'passed':diff<=1e-4,'original_batch_shape':full_shape,'note':'first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4'};results.append(r);write_json(D/'reload.json',results);print(json.dumps(r),flush=True);assert r['passed'];tested.add(cl);del m,checkpoint;gc.collect();torch.cuda.empty_cache()
assert tested=={'Clair3_P','Clair3_F'},tested;write_json(D/'reload.json',results);s['reload']=results;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(results),flush=True)
