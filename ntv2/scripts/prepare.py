from pathlib import Path
import json,hashlib,collections
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
R=Path('/work');src=Path('/source/external_data/BEND');out={};audit={};official={}
for split in ['train','test']:
 df=pd.read_parquet(R/f'data/enhancers/{split}.parquet');print(split,df.shape,df.columns.tolist(),df.iloc[0].to_dict(),flush=True);official[split]=df
tr=official['train'];te=official['test'];assert {'sequence','label'}<=set(tr.columns)
for df in [tr,te]:assert not df[['sequence','label']].isnull().any().any() and set(df.label.unique())=={0,1}
testseq=set(te.sequence);tr=tr.loc[~tr.sequence.isin(testseq)].drop_duplicates('sequence').copy();te=te.drop_duplicates('sequence').copy();trainids,validids=train_test_split(np.arange(len(tr)),test_size=.15,stratify=tr.label,random_state=42)
def subset(df,indices,n):
 ids,_=train_test_split(indices,train_size=n,stratify=df.iloc[indices].label,random_state=42) if len(indices)>n else (indices,[])
 return [{'id':str(df.iloc[i].get('id',df.index[i])),'sequence':df.iloc[i].sequence,'label':int(df.iloc[i].label),'sequence_sha256':hashlib.sha256(df.iloc[i].sequence.encode()).hexdigest()} for i in ids]
out['train']=subset(tr,trainids,512);out['validation']=subset(tr,validids,128);out['test']=subset(te,np.arange(len(te)),min(400,len(te)))
sets=[set(r['sequence_sha256'] for r in out[k]) for k in out];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
manifest=json.loads((src/'prepared/20261012_001503_n1000_seed42/manifest.json').read_text());fa=src/'genomes/GRCh38.primary_assembly.genome.fa';idx={}
for line in Path(str(fa)+'.fai').read_text().splitlines():
 name,length,offset,bases,width,*_=line.split('\t');idx[name]=tuple(map(int,[length,offset,bases,width]))
def getseq(f,chrom,start,end):
 if chrom not in idx:chrom=chrom.removeprefix('chr')
 length,off,bases,width=idx[chrom];assert start>=0 and end<=length
 lo=off+start//bases*width+start%bases;hi=off+(end-1)//bases*width+(end-1)%bases+1;f.seek(lo);return f.read(hi-lo).replace(b'\n',b'').replace(b'\r',b'').decode().upper()
long=[]
with fa.open('rb') as f:
 for pos in manifest['positions']:
  center=pos['center'];seq=getseq(f,pos['chromosome'],center-6141,center+6141)
  if len(seq)!=12282 or not set(seq)<=set('ACGT'):continue
  if pos['strand']=='-':seq=seq.translate(str.maketrans('ACGT','TGCA'))[::-1]
  long.append(dict(id=pos['id'],sequence=seq,chromosome=pos['chromosome'],center=center,strand=pos['strand']))
  if len(long)==128:break
assert len(long)==128;out['long_inference']=long
for k,rows in out.items():audit[k]={'n':len(rows),'length_min':min(len(r['sequence']) for r in rows),'length_max':max(len(r['sequence']) for r in rows),'labels':dict(collections.Counter(r.get('label','unlabeled') for r in rows)),'first_example':rows[0]}
(R/'data/inputs.json').write_text(json.dumps(out,indent=2));(R/'data/input_audit.json').write_text(json.dumps({'splits':audit,'official_train_rows':len(official['train']),'official_test_rows':len(official['test']),'train_after_test_overlap_and_duplicates_removed':len(tr),'policy':'official test untouched for selection; train/val stratified seed42; exact sequence hash disjoint; long unlabeled reference regions inference only'},indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a!='first_example'} for k,v in audit.items()},indent=2))
