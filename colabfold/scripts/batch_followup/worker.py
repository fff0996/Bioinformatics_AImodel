from pathlib import Path
import os,time,json,threading,csv,subprocess,gc
import numpy as np,jax,jax.numpy as jnp
from colabfold.alphafold.models import load_models_and_params
from alphafold.common import protein
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];B=int(os.environ['BATCH']);C=D/f'batch_{B}';C.mkdir();dev=jax.devices('gpu')[0];stop=threading.Event();rows=[];phase='loading';G=2**30
S={'status':'running','batch':B,'implementation':'experimental jax.vmap over unmodified ColabFold RunModel.apply; shared fixed model_1 weights, independent actual query features and RNG; not a native CLI batch option','conditions':[]}
def write():(C/'summary.json').write_text(json.dumps(S,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def mem():
 info={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};rss=next(int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'));return {'rss_gib':rss/G,'available_gib':info['MemAvailable']/G,'total_gib':info['MemTotal']/G,'swap_gib':(info['SwapTotal']-info['SwapFree'])/G}
def dmem():return {k:int(v) for k,v in (dev.memory_stats() or {}).items()}
swap0=mem()['swap_gib']
def monitor():
 with (C/'resources.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['query_start','query_end','phase','gpu_util','rss_gib','available_gib','swap_gib','jax_bytes_in_use']);w.writeheader()
  while not stop.is_set():
   t=time.time();ph=phase
   try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip())
   except Exception:u=None
   m=mem();r={'query_start':t,'query_end':time.time(),'phase':ph,'gpu_util':u,'rss_gib':m['rss_gib'],'available_gib':m['available_gib'],'swap_gib':m['swap_gib'],'jax_bytes_in_use':dmem().get('bytes_in_use')};rows.append(r);w.writerow(r);f.flush()
   if m['available_gib']<16 or m['swap_gib']>swap0+2:(C/'memory_guard.json').write_text(json.dumps(r));os._exit(86)
   stop.wait(.5)
th=threading.Thread(target=monitor,daemon=True);th.start()
def aligned_rmsd(x,y):
 x=x-x.mean(0);y=y-y.mean(0);u,s,v=np.linalg.svd(x.T@y);q=np.eye(3);q[-1,-1]=np.linalg.det(u@v);return float(np.sqrt(np.mean(np.sum((x@(u@q@v)-y)**2,axis=1))))
try:
 audit=json.loads((D/'input_audit.json').read_text());queries=audit['queries'];allfeat=[dict(np.load(D/(q['id']+'_processed_features.npz'))) for q in queries];keys=set(allfeat[0]);assert all(set(f)==keys for f in allfeat);maxshape={k:tuple(max(f[k].shape[d] for f in allfeat) for d in range(allfeat[0][k].ndim)) for k in keys};features=[]
 for f in allfeat:
  features.append({k:np.pad(v,[(0,maxshape[k][d]-v.shape[d]) for d in range(v.ndim)],mode='constant') for k,v in f.items()})
 selected=features[:B];S['input']={'unique_queries':B,'query_ids':[q['id'] for q in queries[:B]],'actual_residues':[q['length'] for q in queries[:B]],'feature_shapes_per_query':{k:list(v.shape) for k,v in features[0].items()},'padding':'pool-wide max feature shapes; zero-padded extra-MSA/cluster entries masked by zero masks; no sequence duplication','actual_msa_valid_rows':[int((f['msa_mask'][0].sum(-1)>0).sum()) for f in selected]};t=time.time();loaded=load_models_and_params(num_models=1,use_templates=False,num_recycles=3,recycle_early_stop_tolerance=0.,num_ensemble=1,model_order=[1],model_type='alphafold2_ptm',data_dir=Path('/cache'),max_seq=512,max_extra_seq=5120,use_bfloat16=False,use_pallas=False,compile_mode='tuned');name,runner,params=loaded[0];runner.params=params;jax.block_until_ready(params);S['loading_seconds']=time.time()-t;S['parameters']=sum(int(x.size) for x in jax.tree_util.tree_leaves(params));S['weights_dtype']=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(params)});S['model_only_postload']={'jax':dmem(),'system':mem()};S['environment']={'jax':jax.__version__,'device':str(dev),'preallocate':os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),'compute':'FP32 activations/weights, explicit separate validation condition after BF16 vmap reference drift','recycles':3,'ensembles':1,'templates':False,'relaxation':False,'seed_per_query':42};write()
 full={k:jnp.asarray(np.stack([f[k] for f in selected])) for k in keys};L=int(selected[0]['aatype'].shape[1]);zero_prev={'prev_msa_first_row':jnp.zeros((B,L,256),jnp.float16),'prev_pair':jnp.zeros((B,L,L,128),jnp.float16),'prev_pos':jnp.zeros((B,L,37,3),jnp.float16)};rng=jnp.stack([jax.random.PRNGKey(42)]*B);first={k:v[:,0:1] for k,v in full.items()};first['prev']=zero_prev;vmapped=jax.jit(jax.vmap(runner.apply.__wrapped__,in_axes=(None,0,0)),compiler_options={'xla_gpu_override_gemm_autotuner':'block_m: 64 block_n: 64 block_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1'});phase='compilation';t=time.time();compiled=vmapped.lower(params,rng,first).compile();S['compilation_seconds']=time.time()-t;S['compiled_memory_analysis']=str(compiled.memory_analysis());write()
 def predict():
  prev=zero_prev;key=jnp.stack([jax.random.PRNGKey(42)]*B)
  for recycle in range(4):
   pair=jax.vmap(jax.random.split)(key);key=pair[:,0];subkey=pair[:,1];subfeat={k:v[:,recycle:recycle+1] for k,v in full.items()};subfeat['prev']=prev;result=compiled(params,subkey,subfeat);jax.block_until_ready(result);result=jax.tree_util.tree_map(lambda x:np.asarray(x,np.float16),result);prev={k:jnp.asarray(v) for k,v in result.pop('prev').items()}
  return result
 phase='warmup';t=time.time();result=predict();S['warmup_seconds']=time.time()-t;phase='execution';t=time.time();repeats=0
 while repeats<2 or time.time()-t<15:result=predict();repeats+=1
 end=time.time();stats=dmem();observed=[r for r in rows if r['query_start']>=t and r['query_end']<=end];utils=[r['gpu_util'] for r in observed if r['gpu_util'] is not None];S['conditions']=[{'batch':B,'unique_queries':B,'repeats':repeats,'processed_queries':B*repeats,'seconds':end-t,'queries_per_second':B*repeats/(end-t),'gpu_util_avg':float(np.mean(utils)) if utils else None,'gpu_util_max':max(utils) if utils else None,'sample_count':len(utils),'jax_memory_stats':stats,'memory_definition':'peak_bytes_in_use is cumulative process allocator peak including compilation and warmup; execution bytes_in_use samples separate, no unsupported reset','execution_sampled_peak_bytes_in_use':max([r['jax_bytes_in_use'] for r in observed if r['jax_bytes_in_use'] is not None],default=None),'rss_sampled_peak_gib':max([r['rss_gib'] for r in observed],default=None),'available_sampled_min_gib':min([r['available_gib'] for r in observed],default=None)}];phase='validation';preds=[]
 for i,q in enumerate(queries[:B]):
  single=jax.tree_util.tree_map(lambda x:x[i],result);coords=single['structure_module']['final_atom_positions'];plddt=single['plddt'];assert coords.shape==(L,37,3) and np.isfinite(coords).all() and np.isfinite(plddt).all();np.savez(C/(q['id']+'_prediction.npz'),atom_positions=coords,plddt=plddt);prot=protein.from_prediction(features=selected[i],result=single,b_factors=np.repeat(plddt[:,None],37,axis=1));pdb=protein.to_pdb(prot);(C/(q['id']+'.pdb')).write_text(pdb);assert sum(l.startswith('ATOM') and l[12:16].strip()=='CA' for l in pdb.splitlines())==q['length'];preds.append({'id':q['id'],'residues':q['length'],'mean_plddt':float(plddt.mean()),'finite_and_pdb_valid':True})
 S['predictions']=preds;reference=D/'batch_1'/ (queries[0]['id']+'_prediction.npz')
 if B==1:
  phase='single_native_reference';t0=time.time();ref,recycles=runner.predict(selected[0],random_seed=42);S['native_reference_seconds_excluded']=time.time()-t0;refcoords=ref['structure_module']['final_atom_positions'];S['native_reference_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(refcoords[:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max()),'native_recycles':int(recycles)};assert S['native_reference_comparison']['ca_aligned_rmsd_angstrom']<.5;np.savez(C/'native_single_reference.npz',atom_positions=refcoords,plddt=ref['plddt'])
 else:
  ref=np.load(reference);S['first_query_batch_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(ref['atom_positions'][:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max())}
 S['status']='complete';write();print('AF2_VMAP_COMPLETE',B,flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write();raise
finally:stop.set();th.join(timeout=5)
