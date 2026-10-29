from pathlib import Path
import urllib.request,ssl,json,hashlib,shutil,zipfile,datetime
R=Path('/home/hyein/ai-benchmark/cellpose');ctx=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt');rows=[]
with urllib.request.urlopen('https://huggingface.co/api/models/mouseland/cellpose-sam',context=ctx,timeout=60) as f:meta=json.load(f)
(R/'models/repository.json').write_text(json.dumps(meta,indent=2));revision=meta['sha']
files=[('data/'+x+'.zip','https://data.broadinstitute.org/bbbc/BBBC039/'+x+'.zip') for x in ['images','masks','metadata']]+[('models/cpsam_v2','https://huggingface.co/mouseland/cellpose-sam/resolve/'+revision+'/cpsam_v2')]
for name,url in files:
 p=R/name;print('DOWNLOAD',name,flush=True)
 if not p.exists():
  with urllib.request.urlopen(url,context=ctx,timeout=180) as src,p.with_name(p.name+'.part').open('wb') as dst:headers=dict(src.headers);shutil.copyfileobj(src,dst)
  p.with_name(p.name+'.part').rename(p)
 else:headers={}
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 rows.append(dict(path=name,url=url,revision=revision if 'models/' in name else 'BBBC039v1',bytes=p.stat().st_size,sha256=h.hexdigest(),headers=headers,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),license_note='BBBC039v1 CC0; model license notes preserved from official source'))
 (R/'data/sources.json').write_text(json.dumps(rows,indent=2))
 if p.suffix=='.zip':
  dest=R/'data'/p.stem;dest.mkdir(exist_ok=True)
  with zipfile.ZipFile(p) as z:
   for n in z.namelist():assert (dest/n).resolve().is_relative_to(dest.resolve())
   z.extractall(dest)
 print(name,p.stat().st_size,flush=True)
