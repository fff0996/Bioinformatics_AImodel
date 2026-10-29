from pathlib import Path
import sys,json,urllib.request
sys.path.insert(0,'/home/hyein/ai-benchmark/scripts')
from fetch_hf_files import fetch
R=Path('/home/hyein/ai-benchmark/rnafm');repo='cuhkaih/rnafm';m=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo));rev=m['sha'];tree=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'/tree/'+rev));(R/'models/repository.json').write_text(json.dumps({'meta':m,'tree':tree},indent=2));f=next(r for r in tree if r['path']=='RNA-FM_pretrained.pth');row=fetch('https://huggingface.co/'+repo+'/resolve/'+rev+'/'+f['path'],R/'models'/f['path'],f['size'],f.get('lfs',{}).get('oid'));(R/'models/sources.json').write_text(json.dumps([row],indent=2));print(row,flush=True)
