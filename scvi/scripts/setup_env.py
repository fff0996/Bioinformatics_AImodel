import pathlib,subprocess,sys,json,importlib.metadata,datetime
R=pathlib.Path('/work');O=R/'results'/('setup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir();(R/'results/SETUP_LATEST').write_text(O.name+'\n')
versions={}
for n in ['torch','torchvision','torchaudio','numpy','scipy','pandas']:
 try:versions[n]=importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError:pass
(O/'base_versions.json').write_text(json.dumps(versions,indent=2));constraint=R/'constraints.txt';constraint.write_text('\n'.join(n+'=='+v for n,v in versions.items() if n in ['torch','torchvision','torchaudio'])+'\n');print(versions,flush=True)
subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(R/'.venv')],check=True)
cmd=[str(R/'.venv/bin/python'),'-m','pip','install','--cache-dir',str(R/'.cache/pip'),'-c',str(constraint),'scvi-tools==1.5.1','psutil'];(O/'install_command.json').write_text(json.dumps(cmd,indent=2))
with (O/'install.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(O/'exit_code.json').write_text(json.dumps(dict(exit_code=p.returncode),indent=2));print('SETUP_EXIT',p.returncode,flush=True);raise SystemExit(p.returncode)
