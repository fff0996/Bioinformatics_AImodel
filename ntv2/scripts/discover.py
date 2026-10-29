import json,urllib.request
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/ntv2')
for kind,repo in [('models','InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'),('datasets','InstaDeepAI/nucleotide_transformer_downstream_tasks')]:
 meta=json.load(urllib.request.urlopen('https://huggingface.co/api/'+kind+'/'+repo,timeout=60));tree=json.load(urllib.request.urlopen('https://huggingface.co/api/'+kind+'/'+repo+'/tree/'+meta['sha']+'?recursive=true',timeout=60));(R/('models' if kind=='models' else 'data')/'repository.json').write_text(json.dumps({'meta':meta,'tree':tree},indent=2));print(kind,meta['sha'],[(r['path'],r.get('size')) for r in tree if kind=='models' or 'enhancers/' in r['path']],flush=True)
