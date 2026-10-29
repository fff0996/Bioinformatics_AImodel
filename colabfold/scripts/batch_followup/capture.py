from pathlib import Path
import os,json,numpy as np,hashlib
from colabfold import batch
from colabfold.utils import setup_logging
from alphafold.model import model as afmodel
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];MSA=R/'results'/(R/'results/MSA_LATEST').read_text().strip()/'msa';manifest=json.loads((R/'data/input_manifest.json').read_text());queries=[r for r in manifest['queries'] if r['length']==115];assert len(queries)==4;setup_logging(D/'feature_capture.log');current={};audit=[]
class Captured(Exception):pass
def capture(self,feat,*args,**kwargs):
 p=D/(current['id']+'_processed_features.npz');np.savez_compressed(p,**{k:np.asarray(v) for k,v in feat.items()});(D/'runner_config.json').write_text(json.dumps(self.config.to_dict(),indent=2));print('CAPTURED',current['id'],{k:list(v.shape) for k,v in feat.items() if k in ['aatype','msa_feat','extra_msa']},flush=True);raise Captured('intentional feature cache; GPU prediction deferred to controlled vmap worker')
afmodel.RunModel.predict=capture
for q in queries:
 current=q;text=(MSA/(q['id']+'.a3m')).read_text();depth=sum(bool(l.strip()) and not l.startswith(('>','#')) for l in text.splitlines());audit.append({'id':q['id'],'sequence':q['sequence'],'length':q['length'],'msa_depth_before_feature_sampling':depth,'msa_sha256':hashlib.sha256(text.encode()).hexdigest()})
 try:batch.run(queries=[(q['id'],q['sequence'],[text],None)],result_dir=D/('capture_'+q['id']),num_models=1,model_order=[1],is_complex=False,num_recycles=3,recycle_early_stop_tolerance=0.,num_ensemble=1,model_type='alphafold2_ptm',msa_mode='mmseqs2_uniref_env',use_templates=False,num_relax=0,data_dir=Path('/cache'),random_seed=42,num_seeds=1,use_bfloat16=True,use_pallas=False,compile_mode='tuned',max_seq=512,max_extra_seq=5120,recompile_padding=10)
 except Captured:pass
 assert (D/(q['id']+'_processed_features.npz')).exists(),q['id']
(D/'input_audit.json').write_text(json.dumps({'queries':audit,'unique_queries':len(audit),'features':'official ColabFold processed features captured before predict, JAX_PLATFORMS=cpu during capture','preset':'alphafold2_ptm model_1, actual runner shared model3 architecture without templates','num_recycles':3,'num_ensemble':1,'max_seq':512,'max_extra_seq':5120,'templates':False,'relaxation':False,'seed':42},indent=2));print('FEATURE_CAPTURE_COMPLETE',flush=True)
