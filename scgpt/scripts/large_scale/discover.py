from pathlib import Path
import urllib.request,json
R=Path('/home/hyein/ai-benchmark/scgpt/data_external/tabula_sapiens');u='https://api.cellxgene.cziscience.com/curation/v1/collections/e5f58829-1a66-40b5-a624-9046778e74f5'
with urllib.request.urlopen(u,timeout=60) as f:a=json.load(f)
(R/'collection_metadata.json').write_text(json.dumps(a,indent=2));print(a.keys(),flush=True)
for d in a.get('datasets',[]):
 if d.get('dataset_id')=='53d208b0-2cfd-4366-9866-c3c6114081bc' or 'All Cells' in d.get('title',''):
  (R/'dataset_metadata.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2),flush=True)
