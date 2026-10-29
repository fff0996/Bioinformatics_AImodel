from pathlib import Path
import sys,json,time,gc,os,hashlib,platform,shutil
import numpy as np,torch,esm,psutil
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'));from gpu_support import Meter,write_json
R=ROOT/'esm2';D=Path(sys.argv[1]);B=int(sys.argv[2]);C=D/f'batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','batch':B,'task':'pretrained protein mean embeddings inference only','conditions':[]};write_json(C/'summary.json',S)
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='model_loading';t=time.time();data=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False);reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',data,reg);del data,reg;gc.collect();model.float().cuda().eval();torch.cuda.synchronize();S['model_loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for p in model.parameters());S['parameters_backbone']=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('lm_head.','contact_head.')));S['parameters_trainable_in_this_run']=0;S['environment']={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'esm':esm.__version__,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform()};write_json(C/'summary.json',S)
 records=json.loads((D/'inputs.json').read_text())[:B];assert len(records)==B;recs=[(r['id'],r['sequence']) for r in records];convert=alphabet.get_batch_converter(1022);_,seqs,tokens=convert(recs);assert tokens.shape==(B,952);torch.save({'ids':[i for i,s in recs],'tokens':tokens},C/'tokenized_inputs.pt');S['input']={'unique_sequences':B,'length_min':min(map(len,seqs)),'length_max':max(map(len,seqs)),'length_mean':float(np.mean(list(map(len,seqs)))),'actual_token_shape':list(tokens.shape),'dtype':str(tokens.dtype),'truncation':False,'padding':'official converter, batch longest950AA + BOS/EOS =952 positions'}
 def forward():
  x=tokens.cuda();out=model(x,repr_layers=[33],return_contacts=False);h=out['representations'][33];e=torch.stack([h[i,1:len(seq)+1].mean(0) for i,seq in enumerate(seqs)]).cpu();assert torch.isfinite(e).all();return e
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():embedding=forward()
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del embedding;gc.collect();t,base=meter.begin('embedding_inference');steps=0
 with torch.inference_mode():
  while steps<2 or time.time()-t<15:
   embedding=forward();torch.cuda.synchronize();steps+=1
 r=meter.end(t,base,batch_size=B,unique_inputs=B,processed_inputs=B*steps,repeats=steps,input_shape=list(tokens.shape));r['sequences_per_second']=B*steps/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];S['conditions']=[r];torch.save({'ids':[i for i,s in recs],'embeddings':embedding},C/'embeddings.pt');saved=torch.load(C/'embeddings.pt',map_location='cpu',weights_only=True);assert saved['embeddings'].shape==(B,1280) and torch.equal(saved['embeddings'],embedding);S['saved_output_exact_reload']=True
 ref=D/'batch_1/embeddings.pt'
 if B>1:
  reference=torch.load(ref,map_location='cpu',weights_only=True);diff=float((reference['embeddings'][0]-embedding[0]).abs().max());assert torch.allclose(reference['embeddings'][0],embedding[0],atol=1e-4,rtol=1e-4);S['first_sequence_batch_invariance_max_abs']=diff
 S['status']='complete';write_json(C/'summary.json',S);print('BATCH_COMPLETE',B,round(r['memory']['peak_allocated_gib'],3),round(r['sequences_per_second'],3),flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write_json(C/'summary.json',S);raise
finally:meter.close()
