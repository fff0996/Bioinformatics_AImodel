import sys,pathlib,json,inspect,importlib.metadata,os
import jax,numpy as np
from alphafold.model import model,config
from colabfold.alphafold import models
from colabfold import batch
r=pathlib.Path('/work');o=r/'results/environment_verified';o.mkdir(exist_ok=True)
info=dict(python=sys.version,versions={n:importlib.metadata.version(n) for n in ['colabfold','jax','jaxlib','dm-haiku','numpy']},devices=[str(d) for d in jax.devices()],environment={k:v for k,v in os.environ.items() if k.startswith(('XLA','JAX','TF_','CUDA'))},run_signature=str(inspect.signature(batch.run)),load_signature=str(inspect.signature(models.load_models_and_params)))
print(json.dumps(info,indent=2));(o/'environment.json').write_text(json.dumps(info,indent=2))
for obj,name in [(model,'alphafold_model.py'),(models,'colabfold_models.py'),(batch,'batch.py')]:
 (o/name).write_text(inspect.getsource(obj))
a=jax.numpy.ones((32,32));(a@a).block_until_ready();print('JAX GPU execution verified')
