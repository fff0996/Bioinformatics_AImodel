import sys,subprocess,json,importlib.metadata
from pathlib import Path
R=Path('/work');(R/'constraints.txt').write_text('torch=='+importlib.metadata.version('torch')+'\n');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True)
cmd=['/work/.venv/bin/pip','install','--cache-dir','/work/.cache/pip','-c','/work/constraints.txt','transformers==4.44.2','pyarrow','scikit-learn','psutil'];code=subprocess.call(cmd);(R/'results/setup_exit.json').write_text(json.dumps({'exit_code':code}));sys.exit(code)
