import sys,subprocess,importlib.metadata,json
from pathlib import Path
R=Path('/work');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True);subprocess.run(['/work/.venv/bin/pip','install','--no-deps','/work/source'],check=True);(R/'constraints.txt').write_text('torch=='+importlib.metadata.version('torch')+'\n');subprocess.run(['/work/.venv/bin/pip','install','-c','/work/constraints.txt','--cache-dir','/work/.cache/pip','scikit-learn','pysam','gffutils','pyfaidx','biopython'],check=True)
