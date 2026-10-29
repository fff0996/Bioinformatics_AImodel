from pathlib import Path
import json,hashlib,ssl,urllib.request,datetime
R=Path('/home/hyein/ai-benchmark/colabfold');D=R/'data';ctx=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt');ss=[]
for line in (R.parent/'esm2/data/uniprot_reviewed.fasta').read_text().splitlines():
 if line.startswith('>'):ss.append([line[1:].split()[0],''])
 else:ss[-1][1]+=line
pool=[(i,s) for i,s in ss if len(s)==115];unique=[];seen=set()
for i,s in pool:
 if s not in seen:unique.append((i,s));seen.add(s)
assert len(unique)>=4
queries=[]
for n,(i,s) in enumerate(unique[:4],1):queries.append(dict(id='count'+str(n)+'_'+i.split('|')[1],sequence=s,length=len(s),kind='count',source='UniProtKB reviewed 2026_03; reused ESM-2 source snapshot',source_id=i))
for pdb in ['1UBQ','1LYZ','1TIM']:
 p=D/(pdb+'.fasta');seq=''.join(l.strip() for l in p.read_text().splitlines() if not l.startswith('>'));queries.append(dict(id=pdb,sequence=seq,length=len(seq),kind='length',source='https://www.rcsb.org/fasta/entry/'+pdb,reference_chain='A'))
 ref=D/(pdb+'.pdb')
 if not ref.exists():
  with urllib.request.urlopen('https://files.rcsb.org/download/'+pdb+'.pdb',context=ctx,timeout=90) as f:ref.write_bytes(f.read())
for q in queries:
 assert set(q['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY') and q['length']>0
queries.sort(key=lambda q:(q['length'],q['id']))
(D/'queries.fasta').write_text(''.join('>'+q['id']+'\n'+q['sequence']+'\n' for q in queries));(D/'input_manifest.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unique_sequences=len(set(q['sequence'] for q in queries)),queries=queries,format='FASTA amino-acid sequences; PDB reference chain A; no labels for count proteins',missing_sequences=0,sequence_dtype='text',preprocessed=False,msa='not yet generated'),indent=2))
print([(q['id'],q['length']) for q in queries],flush=True)
