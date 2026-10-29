from pathlib import Path
import sys,subprocess,json,importlib.metadata,datetime
R=Path('/work');O=R/'results'/('setup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir();(R/'results/SETUP_LATEST').write_text(O.name+'\n');v={k:importlib.metadata.version(k) for k in ['torch','torchvision']};(R/'constraints.txt').write_text('\n'.join(k+'=='+x for k,x in v.items())+'\n');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True);cmd=['/work/.venv/bin/python','-m','pip','install','--cache-dir','/work/.cache/pip','-c','/work/constraints.txt','cellpose==4.2.1','psutil'];(O/'command.json').write_text(json.dumps(cmd,indent=2))
with (O/'install.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(O/'exit_code.json').write_text(json.dumps(dict(exit_code=p.returncode),indent=2));print('SETUP',O,p.returncode,flush=True);sys.exit(p.returncode)
