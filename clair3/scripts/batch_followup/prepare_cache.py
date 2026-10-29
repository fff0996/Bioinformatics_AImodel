from pathlib import Path
import os,json,hashlib
import numpy as np
D=Path('/work/results')/os.environ['RUN_ID'];audit={}
for kind in ['Clair3_P','Clair3_F']:
 files=sorted((D/'candidate_cache').glob(kind+'_*.npz'));assert files;rows=[];sources=[];seen=set()
 for file in files:
  v=np.load(file);x=v['X'];sources.append({'file':file.name,'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.file_digest(file.open('rb'),'sha256').hexdigest()})
  for r in x:
   digest=hashlib.sha256(r.tobytes()).hexdigest()
   if digest not in seen:rows.append(r);seen.add(digest)
 a=np.stack(rows);np.save(D/(kind+'_inputs.npy'),a);audit[kind]={'actual_feature_tensor_rows_before_deduplication':sum(s['shape'][0] for s in sources),'unique_feature_rows':len(a),'feature_shape':list(a.shape[1:]),'dtype':str(a.dtype),'source_caches':sources,'identity_limitation':'deduplicated exact feature tensors; genomic candidate coordinates not exported by this capture hook, no invented position IDs','source':'real HG003 ONT BAM, GRCh38 chr20 100000..300000, existing official Clair3 extraction; no zero-filled or synthetic performance inputs'}
(D/'input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v['unique_feature_rows'] for k,v in audit.items()}),flush=True)
