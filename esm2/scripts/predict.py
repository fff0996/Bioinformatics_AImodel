"""Reconstruct saved partial ESM-2 fine-tune and predict fluorescence for a FASTA."""
import argparse,csv,hashlib,json,pathlib,torch,esm
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=pathlib.Path);p.add_argument('--fasta',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--device',default='cuda');a=p.parse_args()
r=next(p for p in pathlib.Path(__file__).resolve().parents if (p/'models/esm2_t33_650M_UR50D.pt').is_file());ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
h=hashlib.sha256()
with (r/'models'/ck['base_checkpoint']).open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert h.hexdigest()==ck['base_sha256'],'Base checkpoint hash mismatch'
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
base=torch.load(r/'models'/ck['base_checkpoint'],map_location='cpu',weights_only=False);reg=torch.load(r/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',base,reg);del base,reg
result=model.load_state_dict(ck['backbone_updates'],strict=False);assert not result.unexpected_keys
model.to(a.device).eval();head=torch.nn.Linear(1280,1).to(a.device);head.load_state_dict(ck['head']);head.eval();conv=alphabet.get_batch_converter(1022);ds=esm.FastaBatchedDataset.from_file(a.fasta)
assert all(set(s)<=set('ACDEFGHIKLMNPQRSTVWY') and len(s)>0 for s in ds.sequence_strs)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as f,torch.no_grad():
 w=csv.writer(f);w.writerow(['id','original_length','actual_residues','truncated','predicted_log_fluorescence'])
 for ix in ds.get_batch_indices(4096,extra_toks_per_seq=1):
  ids,ss,tok=conv([ds[i] for i in ix]);o=model(tok.to(a.device),repr_layers=[33],return_contacts=False)['representations'][33];e=torch.stack([o[i,1:min(len(s),1022)+1].mean(0) for i,s in enumerate(ss)]);pred=(head(e).squeeze(-1)*ck['target_std']+ck['target_mean']).cpu().tolist()
  for i,s,v in zip(ids,ss,pred):w.writerow([i,len(s),min(len(s),1022),len(s)>1022,v])
print(a.output)
