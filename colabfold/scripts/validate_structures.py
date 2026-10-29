"""CPU-only validation of persisted predicted structures and experimental references."""
import pathlib,json,csv,hashlib,sys,numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.Align import PairwiseAligner
R=pathlib.Path('/work');complex_mode=len(sys.argv)>1 and sys.argv[1]=='multimer';run=(R/'results'/('MULTIMER_LATEST' if complex_mode else 'AF2_LATEST')).read_text().strip();O=R/'results'/run;manifest=json.loads((R/'data'/('complex_manifest.json' if complex_mode else 'input_manifest.json')).read_text());qs={q['id']:q for q in manifest['queries']};parser=PDBParser(QUIET=True);results=[]
def ca_chain(path,chain='A'):
 structure=parser.get_structure('p',path)[0];c=structure[chain];res=[r for r in c if r.id[0]==' ' and 'CA' in r];return ''.join(seq1(r.resname) for r in res),np.array([r['CA'].coord for r in res],dtype=float)
def align_metrics(pseq,px,rseq,rx):
 aligner=PairwiseAligner();aligner.mode='global';aligner.match_score=2;aligner.mismatch_score=-1;aligner.open_gap_score=-5;aligner.extend_gap_score=-.5;a=aligner.align(pseq,rseq)[0];ip=[];ir=[]
 for (s,e),(rs,re) in zip(a.aligned[0],a.aligned[1]):ip.extend(range(s,e));ir.extend(range(rs,re))
 x=px[ip];y=rx[ir];xc=x-x.mean(0);yc=y-y.mean(0);u,_,vt=np.linalg.svd(xc.T@yc);rot=u@vt
 if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vt
 rmsd=float(np.sqrt(np.mean(np.sum((xc@rot-yc)**2,axis=1))));dp=np.linalg.norm(x[:,None]-x[None,:],axis=-1);dr=np.linalg.norm(y[:,None]-y[None,:],axis=-1);mask=(dr<15)&(dr>0);dist=np.abs(dp-dr);lddt=float(np.mean(sum((dist[mask]<t).astype(float) for t in [.5,1,2,4])/4)) if mask.any() else None
 return dict(aligned_ca=len(ip),predicted_ca=len(px),reference_ca=len(rx),sequence_identity=float(np.mean([pseq[i]==rseq[j] for i,j in zip(ip,ir)])),ca_rmsd_angstrom=rmsd,ca_lddt=lddt,definition='Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT.')
for qid,q in qs.items():
 files=sorted((O/'predictions').glob(qid+'_unrelaxed_rank_*.pdb'));assert len(files)==5,(qid,len(files));scorefiles=sorted((O/'predictions').glob(qid+'_scores_rank_*.json'));assert len(scorefiles)==5
 for p in files:
  seq,xyz=ca_chain(p)
  if complex_mode:
   seqb,xyzb=ca_chain(p,'B');assert [seq,seqb]==q['chains'];seq+=seqb;xyz=np.concatenate([xyz,xyzb])
  assert seq==q['sequence'],(qid,len(seq),q['length']);assert np.isfinite(xyz).all();rank=p.name.split('_rank_')[1][:3];scorepath=next(s for s in scorefiles if '_rank_'+rank+'_' in s.name);score=json.loads(scorepath.read_text());pl=np.asarray(score['plddt']);assert len(pl)==q['length'] and np.isfinite(pl).all() and pl.min()>=0 and pl.max()<=100
  pae=np.asarray(score['pae']);assert pae.shape==(q['length'],q['length']) and np.isfinite(pae).all()
  row=dict(query=qid,rank=int(rank),pdb_path=str(p),scores_path=str(scorepath),residues=len(seq),mean_plddt=float(pl.mean()),ptm=float(score['ptm']),pdb_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),finite=True)
  if qid in ['1UBQ','1LYZ','1TIM']:
   rs,rx=ca_chain(R/'data'/(qid+'.pdb'));row['reference']=str(R/'data'/(qid+'.pdb'));row.update(align_metrics(seq,xyz,rs,rx))
  if complex_mode:
   ra,xa=ca_chain(R/'data/3G5O.pdb','A');rb,xb=ca_chain(R/'data/3G5O.pdb','B');pa,pxa=ca_chain(p,'A');pb,pxb=ca_chain(p,'B');row['chain_A']=align_metrics(pa,pxa,ra,xa);row['chain_B']=align_metrics(pb,pxb,rb,xb);row['reference_chain_mapping']={'predicted A':'3G5O A','predicted B':'3G5O B'};row['complex_alignment']=align_metrics(seq,xyz,ra+rb,np.concatenate([xa,xb]));row['iptm']=float(score['iptm'])
  results.append(row)
(O/'structure_validation.json').write_text(json.dumps(dict(passed=True,files_validated=len(results),results=results),indent=2));print('VALIDATED PDB+scores:',len(results))
