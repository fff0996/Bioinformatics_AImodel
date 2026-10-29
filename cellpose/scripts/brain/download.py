from pathlib import Path
import urllib.request,json,hashlib,time,concurrent.futures
R=Path('/home/hyein/ai-benchmark/cellpose/data/brain_nucmm');repo='MedOtter/NucMM-M'
with urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo,timeout=30) as f:meta=json.load(f)
revision=meta['sha'];(R/'repository_metadata.json').write_text(json.dumps(meta,indent=2));files=[s['rfilename'] for s in meta['siblings'] if s['rfilename'].endswith(('.h5','.md','.txt'))]
def fetch(name):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);url='https://huggingface.co/datasets/'+repo+'/resolve/'+revision+'/'+name
 if not p.exists():
  for attempt in range(4):
   try:
    with urllib.request.urlopen(url,timeout=90) as r,p.with_suffix(p.suffix+'.part').open('wb') as f:
     for b in iter(lambda:r.read(1024*1024),b''):f.write(b)
    p.with_suffix(p.suffix+'.part').rename(p);break
   except Exception:
    if attempt==3:raise
    time.sleep(2)
 h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();print(name,p.stat().st_size,flush=True);return {'file':name,'url':url,'bytes':p.stat().st_size,'sha256':h}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(fetch,files))
(R/'download_manifest.json').write_text(json.dumps({'source':'public third-party mirror, pinned revision; original README_original.txt and primary NucMM project used for modality and release specification','repo':repo,'revision':revision,'files':out,'primary_project':'https://connectomics-bazaar.github.io/proj/nucMM/index.html','original_release_folder':'https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8','mirrored_readme_discrepancies':['README.md incorrectly calls mouse volume EM; primary paper and README_original.txt identify micro-CT','README.md says200cubed; original release says192cubed; actual h5 shape must be inspected']},indent=2));print('DOWNLOAD_COMPLETE',flush=True)
