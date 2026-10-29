"""Export latent posterior means from a saved SCVI model (known batches and gene set)."""
import argparse,pathlib,numpy as np,scanpy as sc,scvi,torch
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--counts');p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args();assert not a.output.exists(),'Output already exists';torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;scvi.settings.seed=42
data=sc.read_h5ad(a.counts) if a.counts else None
if data is not None:
 assert 'counts' in data.layers and 'sample_id' in data.obs
 values=data.layers['counts'].data;assert np.isfinite(values).all() and (values>=0).all() and (values==np.floor(values)).all()
m=scvi.model.SCVI.load(a.model,adata=data,accelerator='gpu',device=1);z=m.get_latent_representation(batch_size=128,give_mean=True);assert np.isfinite(z).all();a.output.parent.mkdir(parents=True,exist_ok=True);np.save(a.output,z);m.adata.obs.to_csv(a.output.with_suffix('.cells.csv'));print(a.output,z.shape)
