from pathlib import Path
import subprocess,datetime,shlex,json
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'colabfold';run='vmap_batch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');print(run,flush=True);cmd=['docker','run','--rm','--network','none','--ipc=host','--user','1000:1000','-e','JAX_PLATFORMS=cpu','-e','RUN_ID='+run,'-e','MPLCONFIGDIR=/work/results/'+run+'/mplcache','-v',str(R)+':/work','-v',str(R/'cache/colabfold')+':/cache:ro','ghcr.io/sokrypton/colabfold:1.6.2-cuda13','python','-u','/work/scripts/batch_followup/capture.py'];(D/'capture_command.json').write_text(json.dumps(cmd,indent=2))
with (D/'capture.log').open('w') as f:c=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
(D/'capture_exit.json').write_text(json.dumps({'exit_code':c}));print('CAPTURE_EXIT',c,flush=True)
