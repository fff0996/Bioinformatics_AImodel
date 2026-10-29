import os,signal,time,json,urllib.request,hashlib,concurrent.futures
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/cellpose')
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try: args=(proc/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if len(args)>1 and args[1]==b'/home/hyein/ai-benchmark/cellpose/scripts/download.py':os.kill(int(proc.name),signal.SIGTERM)
time.sleep(1)
p=R/'models/cpsam_v2'; part=p.with_name(p.name+'.part'); start=part.stat().st_size; total=1233586851
rev=json.loads((R/'models/repository.json').read_text())['sha']; url=f'https://huggingface.co/mouseland/cellpose-sam/resolve/{rev}/cpsam_v2?download=true'
chunkdir=R/'models/download_chunks';chunkdir.mkdir(exist_ok=True); step=4*1024*1024
ranges=[(s,min(s+step,total)-1) for s in range(start,total,step)]
def get(r):
 a,b=r;dest=chunkdir/f'{a}_{b}'
 if dest.exists() and dest.stat().st_size==b-a+1:return dest
 for retry in range(5):
  try:
   req=urllib.request.Request(url+f'&range_start={a}',headers={'Range':f'bytes={a}-{b}'})
   with urllib.request.urlopen(req,timeout=90) as f:
    assert f.status==206 and f.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/')
    data=f.read();assert len(data)==b-a+1;dest.write_bytes(data)
   return dest
  except Exception:
   if retry==4:raise
 return dest
print('RESUME',start,'chunks',len(ranges),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
 for i,d in enumerate(pool.map(get,ranges)):
  if i%16==0:print('chunks_done',i+1,'/',len(ranges),flush=True)
with part.open('ab') as f:
 for a,b in ranges:f.write((chunkdir/f'{a}_{b}').read_bytes())
assert part.stat().st_size==total;part.rename(p)
h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();rows=json.loads((R/'data/sources.json').read_text()); rows.append({'path':'models/cpsam_v2','url':url,'revision':rev,'bytes':total,'sha256':h,'method':'resumed verified Content-Range chunks'});(R/'data/sources.json').write_text(json.dumps(rows,indent=2));print('COMPLETE',total,h,flush=True)
