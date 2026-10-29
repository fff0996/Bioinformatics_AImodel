import subprocess,sys,json,importlib.metadata
from pathlib import Path
R=Path('/work');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True);subprocess.run(['/work/.venv/bin/pip','install','--cache-dir','/work/.cache/pip','--no-deps','/work/source'],check=True);subprocess.run(['/work/.venv/bin/pip','install','scikit-learn','biopython','ptflops'],check=True);(R/'results/setup_done.json').write_text(json.dumps({'torch_preserved':importlib.metadata.version('torch')}))
