from pathlib import Path
import subprocess,time,json,sys
R=Path('/home/hyein/ai-benchmark');run=sys.argv[1];D=R/'scgpt/results'/run
while True:
 try:s=json.loads((D/'summary.json').read_text())
 except Exception:time.sleep(30);continue
 subprocess.run([sys.executable,str(R/'scgpt/scripts/large_scale/report.py'),run],check=True)
 if s['status'] in ['complete','failed','paused']:
  (D/'final_status.json').write_text(json.dumps({'status':s['status'],'observed_at':time.time(),'completed_conditions':len(s['conditions'])}))
  with (R/'STATUS.md').open('a') as f:f.write('\n실제30만세포 실행 '+run+' 최종 상태: '+s['status']+'; 조건수 '+str(len(s['conditions']))+'; 보고서 scgpt/reports/scGPT_actual_300k.md.\n')
  break
 time.sleep(30)
