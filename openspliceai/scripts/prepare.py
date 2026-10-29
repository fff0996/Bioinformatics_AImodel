from pathlib import Path
import csv,json,hashlib,bisect,collections
import pysam,numpy as np
R=Path('/work');fa=pysam.FastaFile('/reference/GRCh38.primary_assembly.genome.fa');annotation=R/'source/openspliceai/variant/annotations/grch38.txt';rows=list(csv.DictReader(annotation.open(),delimiter='\t'));sites=collections.defaultdict(lambda:{1:set(),2:set()})
for row in rows:
 starts=[int(v) for v in row['EXON_START'].split(',') if v];ends=[int(v) for v in row['EXON_END'].split(',') if v];assert len(starts)==len(ends)
 if len(starts)<2:continue
 key=(row['CHROM'],row['STRAND']);donors=[e-1 for e in ends[:-1]];acceptors=starts[1:]
 if row['STRAND']=='-':donors,acceptors=acceptors,donors
 sites[key][2].update(donors);sites[key][1].update(acceptors)
sites={k:{t:sorted(v) for t,v in x.items()} for k,x in sites.items()};out={};audit={'reference':'GRCh38.primary_assembly.genome.fa (existing BEND reference)','annotation':str(annotation.relative_to(R)),'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),'coordinate_policy':'genomic0based; plus donor exon_end-1, acceptor next_exon_start; minus swapped then reverse-complement; labeled exon bases per official create-data motif convention','source_checkpoint':'official OSAI_MANE10000 ensemble seeds10..14; annotation here is bundled grch38 transcript table, not asserted identical MANE training annotation','splits':{}}
for split,chroms,n in [('train',['2','4'],96),('validation',['6'],24),('test',['1','3'],36)]:
 selected=[];lastend={};seen=set()
 for row in rows:
  chrom=row['CHROM'];strand=row['STRAND'];key=(chrom,strand)
  if chrom not in chroms or key not in sites or row['#NAME'] in seen:continue
  starts=[int(v) for v in row['EXON_START'].split(',') if v];ends=[int(v) for v in row['EXON_END'].split(',') if v]
  if len(starts)<2:continue
  center=(ends[0]-1+starts[1])//2;start=center-10000;end=center+10000;ctg='chr'+chrom
  if start<0 or end>fa.get_reference_length(ctg) or start<lastend.get(ctg,-1):continue
  seq=fa.fetch(ctg,start,end).upper()
  if not set(seq)<=set('ACGTN') or seq.count('N')>100:continue
  labels=np.zeros(10000,dtype=np.int64);out_start=center-5000;out_end=center+5000
  for kind,coords in sites[key].items():
   for pos in coords[bisect.bisect_left(coords,out_start):bisect.bisect_left(coords,out_end)]:
    idx=pos-out_start if strand=='+' else out_end-1-pos
    if labels[idx] not in [0,kind]:labels[idx]=-1
    else:labels[idx]=kind
  if not np.any(labels==1) or not np.any(labels==2):continue
  if strand=='-':seq=seq.translate(str.maketrans('ACGTN','TGCAN'))[::-1]
  rec={'id':row['#NAME']+'_'+chrom+'_'+str(center)+'_'+strand,'gene':row['#NAME'],'chromosome':ctg,'strand':strand,'genomic_input_start':start,'genomic_input_end':end,'sequence':seq,'labels_10000':labels.tolist(),'label_counts':dict(collections.Counter(map(int,labels)))};selected.append(rec);seen.add(row['#NAME']);lastend[ctg]=end
  if len(selected)==n:break
 assert len(selected)==n,(split,len(selected));out[split]=selected;motifs=collections.defaultdict(collections.Counter)
 for r in selected:
  seq=r['sequence'][5000:-5000]
  for i,v in enumerate(r['labels_10000']):
   if v==2:motifs['donor'][seq[i+1:i+3]]+=1
   if v==1:motifs['acceptor'][seq[max(0,i-2):i]]+=1
 audit['splits'][split]={'n':len(selected),'chromosomes':chroms,'input_nt':20000,'output_label_nt':10000,'motifs':{k:dict(v) for k,v in motifs.items()},'first_example':{k:v for k,v in selected[0].items() if k not in ['sequence','labels_10000']}}
 for kind,canonical in [('donor','GT'),('acceptor','AG')]:assert motifs[kind][canonical]/sum(motifs[kind].values())>.75,(split,kind,motifs[kind])
allrows=[r for rs in out.values() for r in rs];assert len({hashlib.sha256(r['sequence'].encode()).hexdigest() for r in allrows})==len(allrows);(R/'data/inputs.json').write_text(json.dumps(out));(R/'data/input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit['splits'],indent=2))
