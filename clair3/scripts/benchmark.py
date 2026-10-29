import sys,os,json,time,subprocess,hashlib,collections
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,pysam,torch
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];meter=Meter(D);S={'status':'running','conditions':[],'settings':{'version':'2.0.4','commit':'9d1dd3f2475072b6255900e406c34c6b0545fc88','threads':4,'gpu_predictor_workers':1,'batch_policy':'official predictBatchSize200*5 for GPU, actual batches logged','platform':'ont','checkpoint':'models/ont/{pileup,full_alignment}.pt','precision':'FP32 weights, original integer input features; model converts/scales input','phasing':'default whatshap intermediate phasing','preprocessing':'native ARM64 CFFI; CPython3.12 replaces optional PyPy accelerator','warmup':'N/A, end-to-end includes each worker load and startup','training':'supported by official Train; this run performs variant calling only'},'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}}
def save():write_json(D/'summary.json',S)
def must(cmd,**kw):
 p=subprocess.run(cmd,**kw);assert p.returncode==0,(cmd,p.returncode)
try:
 # Actual input and truth files were audited before GPU calling.
 S['input_audit']=json.loads((R/'data/input_audit.json').read_text());save();half=R/'data/HG003_half.bam'
 if not half.exists():
  must(['samtools','view','-s','42.5','-b','-o',str(half),str(R/'data/HG003_chr20_demo.bam')]);must(['samtools','index',str(half)])
 normtruth=D/'truth_normalized.vcf.gz';must(['bcftools','norm','-f',str(R/'data/GRCh38_no_alt_chr20.fa'),'-m','-any','-Oz','-o',str(normtruth),str(R/'data/HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz')]);must(['bcftools','index','-t',str(normtruth)])
 for length,bamname in [(50000,'HG003_chr20_demo.bam'),(100000,'HG003_chr20_demo.bam'),(200000,'HG003_chr20_demo.bam'),(200000,'HG003_half.bam')]:
  name=f'region_{length}_'+('half' if 'half' in bamname else 'full');dest=D/name;dest.mkdir();env=os.environ.copy();env['CLAIR3_METRICS_DIR']=str(dest/'instrumentation');env['PYTHONPATH']='/work/scripts:/work/source';env['PATH']='/work/.venv/bin:'+env['PATH'];cmd=['/work/.venv/bin/python','/work/source/run_clair3.py','--bam_fn='+str(R/'data'/bamname),'--ref_fn=/work/data/GRCh38_no_alt_chr20.fa','--threads=4','--platform=ont','--model_path=/work/models/ont','--output='+str(dest/'calls'),'--bed_fn='+str(R/f'data/region_{length}.bed'),'--python=/work/.venv/bin/python','--pypy=/work/.venv/bin/python','--whatshap=/work/.venv/bin/whatshap','--device=cuda:0','--use_gpu'];write_json(dest/'command.json',cmd);t,base=meter.begin(name)
  with (dest/'run.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd='/work/source')
  end=time.time();write_json(dest/'exit_code.json',{'exit_code':code});assert code==0,(name,code);workers=[]
  for p in (dest/'instrumentation').glob('worker_*.jsonl'):workers.extend(json.loads(line) for line in p.read_text().splitlines())
  loads=[r for r in workers if r['event']=='model_only_postload'];forwards=[r for r in workers if r['event']=='forward'];assert loads and forwards and all(r['memory']['allocated_gib']>0 for r in loads)
  if 'model_only_postload' not in S:S['model_only_postload']={**loads[0]['memory'],'system_available_gib':loads[0]['memory']['available_gib'],'system_total_gib':meter.memory()['system_total_gib']};S['parameters']={r['model']:r['total_parameters'] for r in loads}
  row=meter.end(t,base,name=name,mode='variant_calling',bp=length,unique_inputs=1,repeats=1,processed=length,bam=bamname,worker_loads=loads,actual_input_shapes=sorted({str(r['shape']) for r in forwards}),forward_calls=len(forwards),forward_seconds=sum(r['seconds'] for r in forwards),candidate_tensors=sum(r['shape'][0] for r in forwards));row['seconds']=end-t;row['throughput']=length/row['seconds'];row['memory']=dict(peak_allocated_gib=max(r['memory']['peak_allocated_gib'] for r in forwards),peak_reserved_gib=max(r['memory']['peak_reserved_gib'] for r in forwards));row['baseline']={'allocated_gib':max(r['memory']['allocated_gib'] for r in loads)};row['peak_delta_gib']=row['memory']['peak_allocated_gib']-row['baseline']['allocated_gib'];row['memory_scope']='maximum per-worker allocator peak, not sum of processes; multiple models run serially';S['conditions'].append(row);save();print('CONDITION',name,'seconds',row['seconds'],'peak',row['memory']['peak_allocated_gib'],flush=True)
  calls=dest/'calls/merge_output.vcf.gz';assert calls.exists();norm=dest/'calls_normalized.vcf.gz';must(['bcftools','norm','-f',str(R/'data/GRCh38_no_alt_chr20.fa'),'-m','-any','-Oz','-o',str(norm),str(calls)]);must(['bcftools','index','-t',str(norm)])
  # Genomic region vs coverage experiments remain separate; audit actual depth.
  b=pysam.AlignmentFile(str(R/'data'/bamname));depth=np.zeros(length,dtype=np.int32)
  for col in b.pileup('chr20',100000,100000+length,truncate=True,stepper='samtools',min_base_quality=0,max_depth=10000):depth[col.reference_pos-100000]=col.nsegments
  row['coverage_mean']=float(depth.mean());row['reads_overlapping']=b.count('chr20',100000,100000+length);b.close()
  beds=[]
  for line in (R/'data/HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed').read_text().splitlines():
   if not line or line.startswith('#'):continue
   c,a,z,*_=line.split()
   if c=='chr20':beds.append((int(a),int(z)))
  def keys(path):
   out={'SNP':set(),'INDEL':set()};v=pysam.VariantFile(str(path))
   for rec in v.fetch('chr20',100000,100000+length):
    if rec.pos-1<100000 or rec.stop>100000+length or not any(a<=rec.pos-1 and rec.stop<=z for a,z in beds):continue
    if set(rec.filter.keys())-{'PASS','.'}:continue
    gt=next(iter(rec.samples.values())).get('GT',None)
    for i,alt in enumerate(rec.alts or [],1):
     if gt and i not in gt:continue
     if not alt or alt.startswith('<') or alt=='*':continue
     kind='SNP' if len(rec.ref)==len(alt)==1 else 'INDEL';out[kind].add((rec.contig,rec.pos,rec.ref,alt))
   return out
  truth=keys(normtruth);query=keys(norm);accuracy={}
  for k in truth:
   tp=len(truth[k]&query[k]);fp=len(query[k]-truth[k]);fn=len(truth[k]-query[k]);accuracy[k]={'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
  row['normalized_exact_allele_metrics']=accuracy;write_json(dest/'metrics.json',accuracy);row['output_vcf']=str(calls.relative_to(R));row['output_vcf_sha256']=hashlib.file_digest(calls.open('rb'),'sha256').hexdigest();save()
 S['status']='calling_evaluated_pending_reload';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
