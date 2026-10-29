import datetime,hashlib,json,pathlib,ssl,urllib.request,tarfile,shutil
R=pathlib.Path('/home/hyein/ai-benchmark/esm2')
ctx=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt')
sources={
 'data/uniprot_reviewed.fasta':'https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20length%3A%5B50%20TO%201022%5D&format=fasta&size=500',
 'data/fluorescence.tar.gz':'https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz',
 'models/esm2_t33_650M_UR50D.pt':'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt',
 'models/esm2_t33_650M_UR50D-contact-regression.pt':'https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt',
 'data/official_extract.py':'https://raw.githubusercontent.com/facebookresearch/esm/main/scripts/extract.py',
}
manifest=[]
for name,url in sources.items():
 p=R/name
 if not p.exists():
  print('DOWNLOAD',url,flush=True)
  with urllib.request.urlopen(url,context=ctx,timeout=180) as src,p.with_suffix(p.suffix+'.part').open('wb') as dst:
   headers=dict(src.headers); shutil.copyfileobj(src,dst)
  p.with_suffix(p.suffix+'.part').rename(p)
 else: headers={}
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 manifest.append(dict(path=name,url=url,bytes=p.stat().st_size,sha256=h.hexdigest(),headers=headers,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 (R/'data/sources.json').write_text(json.dumps(manifest,indent=2))
 print(name,p.stat().st_size,flush=True)
with tarfile.open(R/'data/fluorescence.tar.gz') as tf:
 for m in tf.getmembers():
  dest=(R/'data'/m.name).resolve()
  if not dest.is_relative_to((R/'data').resolve()) or m.issym() or m.islnk(): raise ValueError(m.name)
 tf.extractall(R/'data')
print('PREPARE DONE',flush=True)
