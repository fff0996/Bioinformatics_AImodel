import urllib.request,json,re,hashlib
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/clair3');base='https://www.bio8.cs.hku.hk/clair3/'
for path in ['clair3_models_pytorch/','clair3_models_pytorch/ont/']:
 try:
  b=urllib.request.urlopen(base+path,timeout=45).read();(R/'results'/('index_'+path.strip('/').replace('/','_')+'.html')).write_bytes(b);print(path,b.decode()[:10000],flush=True)
 except Exception as e:print(path,repr(e),flush=True)
base+='demo/quick_demo/ont/';rows=[]
files=['GRCh38_no_alt_chr20.fa','GRCh38_no_alt_chr20.fa.fai','HG003_chr20_demo.bam','HG003_chr20_demo.bam.bai','HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz','HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz.tbi','HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed']
for name in files:
 p=R/'data'/name;print('DOWNLOAD',name,flush=True)
 if not p.exists():
  with urllib.request.urlopen(base+name,timeout=120) as f:p.write_bytes(f.read())
 rows.append({'file':name,'url':base+name,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()});(R/'data/sources.json').write_text(json.dumps(rows,indent=2));print('DONE',name,p.stat().st_size,flush=True)
