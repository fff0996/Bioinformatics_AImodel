import urllib.request,json,hashlib,concurrent.futures,time
from pathlib import Path

def fetch(url,p,size=None,sha=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  if size and size>32*1024**2:
   chunks=p.with_name(p.name+'.chunks');chunks.mkdir(exist_ok=True);step=4*1024**2
   def get(a):
    b=min(a+step,size)-1;d=chunks/str(a)
    if d.exists() and d.stat().st_size==b-a+1:return d
    for retry in range(4):
     try:
      req=urllib.request.Request(url+f'?download=true&part={a}',headers={'Range':f'bytes={a}-{b}'})
      with urllib.request.urlopen(req,timeout=120) as f:
       assert f.status==206 and f.headers['Content-Range'].startswith(f'bytes {a}-{b}/');data=f.read();assert len(data)==b-a+1;d.write_bytes(data)
      return d
     except Exception:
      if retry==3:raise
      time.sleep(1)
   ranges=range(0,size,step)
   with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for i,d in enumerate(pool.map(get,ranges)):
     if i%64==0:print(p.name,'chunks',i+1,'/',len(ranges),flush=True)
   tmp=p.with_name(p.name+'.part')
   with tmp.open('wb') as f:
    for a in ranges:f.write((chunks/str(a)).read_bytes())
   tmp.rename(p)
  else:
   with urllib.request.urlopen(url,timeout=120) as f:data=f.read()
   p.write_bytes(data)
 h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert size is None or p.stat().st_size==size;assert sha is None or h==sha,(p,h,sha)
 return {'path':str(p),'url':url,'sha256':h,'bytes':p.stat().st_size}
if __name__=='__main__':
 R=Path('/home/hyein/ai-benchmark/ntv2');rows=[]
 for folder,kind,repo in [('models','models','InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'),('data','datasets','InstaDeepAI/nucleotide_transformer_downstream_tasks')]:
  a=json.loads((R/folder/'repository.json').read_text());rev=a['meta']['sha']
  for f in a['tree']:
   name=f['path']
   choose=(folder=='models' and '/' not in name and name not in ['pytorch_model.bin','.gitattributes'] and f['type']=='file') or (folder=='data' and name in ['enhancers/train.parquet','enhancers/test.parquet'])
   if choose:
    url='https://huggingface.co/'+('datasets/' if kind=='datasets' else '')+repo+'/resolve/'+rev+'/'+name
    rows.append(fetch(url,R/folder/('checkpoint/'+name if folder=='models' else name),f['size'],f.get('lfs',{}).get('oid')));(R/'results/sources.json').write_text(json.dumps(rows,indent=2));print('FETCHED',name,flush=True)
