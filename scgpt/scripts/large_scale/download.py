from pathlib import Path
import urllib.request,json,concurrent.futures,hashlib,shutil,time
R=Path('/home/hyein/ai-benchmark/scgpt/data_external/tabula_sapiens');url='https://cellxgene-data-public.s3.amazonaws.com/cell-census/2024-07-01/h5ads/53d208b0-2cfd-4366-9866-c3c6114081bc.h5ad';size=10026988999;p=R/'tabula_sapiens_20240701.h5ad';chunks=R/'download_chunks';chunks.mkdir(exist_ok=True);step=32*1024**2
with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as f:etag=f.headers['ETag'];assert int(f.headers['Content-Length'])==size
(R/'download_source.json').write_text(json.dumps({'url':url,'census_version':'2024-07-01','expected_bytes':size,'etag':etag,'dataset_id':'53d208b0-2cfd-4366-9866-c3c6114081bc','expected_cells_from_official_census_documentation':483152},indent=2))
def get(a):
 b=min(a+step,size)-1;dest=chunks/str(a)
 if dest.exists() and dest.stat().st_size==b-a+1:return dest
 for retry in range(5):
  try:
   req=urllib.request.Request(url+f'?benchmark_chunk={a}',headers={'Range':f'bytes={a}-{b}','If-Match':etag})
   with urllib.request.urlopen(req,timeout=180) as f:
    assert f.status==206 and f.headers['Content-Range'].startswith(f'bytes {a}-{b}/')
    with dest.open('wb') as out:shutil.copyfileobj(f,out,length=1024**2)
   assert dest.stat().st_size==b-a+1;return dest
  except Exception:
   if retry==4:raise
   time.sleep(1)
if not p.exists():
 ranges=range(0,size,step)
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  for i,d in enumerate(pool.map(get,ranges)):
   if i%12==0:print('CHUNKS',i+1,'/',len(ranges),flush=True)
 tmp=p.with_suffix('.part')
 with tmp.open('wb') as out:
  for a in ranges:
   with (chunks/str(a)).open('rb') as f:shutil.copyfileobj(f,out,length=1024**2)
 assert tmp.stat().st_size==size;tmp.rename(p)
s=json.loads((R/'download_source.json').read_text());s['actual_bytes']=p.stat().st_size
with p.open('rb') as f:s['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
(R/'download_source.json').write_text(json.dumps(s,indent=2));print('COMPLETE',s['actual_bytes'],s['sha256'],flush=True)
