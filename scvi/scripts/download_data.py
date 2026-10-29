from pathlib import Path
import urllib.request,ssl,hashlib,json,tarfile,shutil,datetime
R=Path('/home/hyein/ai-benchmark/scvi');D=R/'data';ctx=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt');sources=[]
for name in ['pbmc4k','pbmc8k']:
 url='https://cf.10xgenomics.com/samples/cell-exp/2.1.0/'+name+'/'+name+'_filtered_gene_bc_matrices.tar.gz';p=D/(name+'.tar.gz');print('DOWNLOAD',url,flush=True)
 if not p.exists():
  with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"}),context=ctx,timeout=180) as f,p.with_suffix('.part').open('wb') as out:headers=dict(f.headers);shutil.copyfileobj(f,out)
  p.with_suffix('.part').rename(p)
 else:headers={}
 h=hashlib.sha256(p.read_bytes()).hexdigest();sources.append(dict(batch=name,url=url,file=str(p),bytes=p.stat().st_size,sha256=h,headers=headers,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),description='Cell Ranger 2.1.0 filtered UMI count matrix; real sample/run batch, same healthy donor for 4k and 8k',license='CC BY 4.0'))
 dest=D/name;dest.mkdir(exist_ok=True)
 with tarfile.open(p) as tf:
  for m in tf.getmembers():
   assert (dest/m.name).resolve().is_relative_to(dest.resolve()) and not m.issym() and not m.islnk()
  tf.extractall(dest)
 print(name,p.stat().st_size,flush=True)
 (D/'sources.json').write_text(json.dumps(sources,indent=2))
