from pathlib import Path
import sys,subprocess,importlib.metadata,json
R=Path('/work');(R/'constraints.txt').write_text('torch=='+importlib.metadata.version('torch')+'\n');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True)
subprocess.run(['/work/.venv/bin/pip','install','-c','/work/constraints.txt','--cache-dir','/work/.cache/pip','hdf5plugin','numexpr','tqdm','cffi','torchmetrics','pysam','whatshap','mpmath==1.2.1','intervaltree'],check=True)
subprocess.run(['/work/.venv/bin/python','build.py'],cwd='/work/source',check=True)
(R/'results/setup_done.json').write_text(json.dumps({'torch':importlib.metadata.version('torch'),'source_commit':'9d1dd3f2475072b6255900e406c34c6b0545fc88','native':'CFFI ARM64 compiled with system htslib; whatshap phasing; CPython preprocessing instead of PyPy'}))
