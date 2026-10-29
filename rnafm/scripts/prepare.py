from pathlib import Path
import json,hashlib,collections,random
R=Path('/home/hyein/ai-benchmark/rnafm');p=R/'source/tutorials/tutorial_data/format_rnacentral_active.100.sample-Max50.fasta';rows=[]
for part in p.read_text().split('>')[1:]:
 ls=part.splitlines();head=ls[0];seq=''.join(ls[1:]).upper();assert seq and set(seq)<=set('ACGUN');rows.append({'id':head.split()[0],'description':head,'type':head.split()[1],'sequence':seq,'sha256':hashlib.sha256(seq.encode()).hexdigest()})
byseq=collections.defaultdict(list)
for r in rows:byseq[r['sha256']].append(r)
unique=[];conflicts=0
for rs in byseq.values():
 if len({r['type'] for r in rs})>1:conflicts+=len(rs);continue
 unique.append(rs[0])
labels=sorted({r['type'] for r in unique});mapping={v:i for i,v in enumerate(labels)};groups=collections.defaultdict(list)
for r in unique:r['label']=mapping[r['type']];groups[r['type']].append(r)
rng=random.Random(42);out={'train':[],'validation':[],'test':[]}
for name,rs in sorted(groups.items()):
 rng.shuffle(rs);n=len(rs);nt=max(1,round(n*.2));nv=max(1,round(n*.16));out['test']+=rs[:nt];out['validation']+=rs[nt:nt+nv];out['train']+=rs[nt+nv:]
for rs in out.values():rng.shuffle(rs)
out['long']=[r for r in unique if len(r['sequence'])>=960];out['count512']=[r for r in unique if len(r['sequence'])>=512][:64];assert len(out['long'])==7 and len(out['count512'])==64
sets=[{r['sha256'] for r in out[k]} for k in ['train','validation','test']];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
audit={'source_file':str(p.relative_to(R)),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_rows':len(rows),'unique_unambiguous':len(unique),'conflicting_label_rows_removed':conflicts,'labels':mapping,'splits':{k:{'n':len(rs),'length_min':min(len(r['sequence']) for r in rs),'length_max':max(len(r['sequence']) for r in rs),'labels':dict(collections.Counter(r['type'] for r in rs)),'first_example':rs[0]} for k,rs in out.items()},'policy':'exact-sequence dedup and conflicting-label removal before stratified per-type seed42 split; homologs not clustered; RNACentral may overlap pretraining'};(R/'data/inputs.json').write_text(json.dumps(out,indent=2));(R/'data/input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v['n'] for k,v in audit['splits'].items()}));print('labels',len(labels),'dedup',len(unique),'conflicts',conflicts)
