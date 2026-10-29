"""Finish existing scGPT run: verify CPU outputs, update reports, then exit. No new GPU experiments."""
from pathlib import Path
import subprocess,json,sys,time,shlex
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'scgpt';run=sys.argv[1];D=R/'results'/run
while not (D/'inference_exit.json').exists():time.sleep(20)
exit_info=json.loads((D/'inference_exit.json').read_text());s=json.loads((D/'summary.json').read_text())
if exit_info['exit_code']!=0 or s['status']!='complete':
 (D/'finalization_status.json').write_text(json.dumps({'status':'not_complete','exit':exit_info,'summary_status':s['status']}));sys.exit(1)
code='''from pathlib import Path
import numpy as np,json,hashlib
D=Path('/workspace/scgpt/results')/RUN
s=json.loads((D/'summary.json').read_text());old=Path('/workspace/scgpt/results')/s['resume_from_run'];a=np.load(D/'embeddings_batch8.npy',mmap_mode='r');b=np.load(D/'embeddings_batch256.npy',mmap_mode='r');original=np.load(old/'embeddings_batch8.npy',mmap_mode='r');assert a.shape==b.shape==(300000,512);same=bool(np.array_equal(a[:100000],original[:100000]));assert same;minimum=1.;maxdiff=0.;normerror=0.
for i in range(0,300000,4096):
 x=np.asarray(a[i:i+4096]);y=np.asarray(b[i:i+4096]);assert np.isfinite(x).all() and np.isfinite(y).all();minimum=min(minimum,float((x*y).sum(1).min()));maxdiff=max(maxdiff,float(np.abs(x-y).max()));normerror=max(normerror,float(np.abs(np.linalg.norm(x,axis=1)-1).max()),float(np.abs(np.linalg.norm(y,axis=1)-1).max()))
assert minimum>.999 and normerror<1e-4
r={'verified_unique_cells_per_batch':300000,'output_shape_per_batch':[300000,512],'all_outputs_finite':True,'prepause_first100000_outputs_exactly_preserved':same,'all_cells_minimum_batch_cosine_similarity':minimum,'all_cells_max_abs_batch_embedding_difference':maxdiff,'all_cells_max_unit_norm_error':normerror,'verification_device':'CPU'};(D/'final_output_verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''.replace('RUN',repr(run))
cmd=['docker','exec','--user','1000:1000','-e','CUDA_VISIBLE_DEVICES=','scgpt-spark','python','-c',code]
with (D/'final_output_verification.log').open('w') as f:rc=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
assert rc==0,'final output verification failed; see log'
subprocess.run([sys.executable,str(R/'scripts/large_scale/report.py'),run],check=True)
s=json.loads((D/'summary.json').read_text());whole=[]
for batch in [8,256]:
 c=next(c for c in s['conditions'] if c['batch_size']==batch and c['unique_cells']==300000);parts=[c]
 if c.get('resume_start_cell',0):parts.insert(0,next(p for p in s['conditions'] if p['batch_size']==batch and p['unique_cells']==c['resume_start_cell']))
 seconds=sum(p['seconds'] for p in parts);samples=sum(p['samples'] for p in parts);whole.append({'batch':batch,'unique_cells':300000,'active_seconds_sum':seconds,'cells_per_second':300000/seconds,'allocated_peak_gib_max_segments':max(p['memory']['peak_allocated_gib'] for p in parts),'reserved_peak_gib_max_segments':max(p['memory']['peak_reserved_gib'] for p in parts),'gpu_util_mean_sample_weighted':sum(p['gpu_util_avg']*p['samples'] for p in parts)/samples,'gpu_util_max':max(p['gpu_util_max'] for p in parts),'segments':len(parts)})
(D/'whole_dataset_comparison.json').write_text(json.dumps(whole,indent=2))
report=R/'reports/scGPT_actual_300k.md'
with report.open('a') as f:
 f.write('\n## 30만 고유세포 전체 처리 최종 비교\n\nbatch8은 전날10만+재개20만의 유효 처리시간 합이며 야간 중단·모델 재적재·warmup은 제외한다. batch256은 한 세션의30만 누적시간이다. 두 조건의 시간 범위는 전처리cache 이후 GPU처리/반환/기록이며, 세션 분할에 따른 차이는 남는다. 메모리는 각 구간의 reset peak 중 최대이고 baseline 차이는 원본 행에서 확인한다. GPU 평균은 표본 수로 가중했다.\n\n|batch|고유세포|유효초 합|세포/초|allocated peak GiB|reserved peak GiB|GPU 평균/최대%|세션 구간 수|\n|---|---|---|---|---|---|---|---|\n')
 for w in whole:f.write(f"|{w['batch']}|300000|{w['active_seconds_sum']:.2f}|{w['cells_per_second']:.2f}|{w['allocated_peak_gib_max_segments']:.3f}|{w['reserved_peak_gib_max_segments']:.3f}|{w['gpu_util_mean_sample_weighted']:.2f}/{w['gpu_util_max']:.0f}|{w['segments']}|\n")
 f.write('\n[전체30만세포 출력 검증](../results/'+run+'/final_output_verification.json): 각 조건300000×512, 전체finite·단위norm·세포별batch간cosine 검사 및 전날10만출력의 정확한 보존 확인. 새로운 GPU실험은 이어서 시작하지 않는다.\n')
for name in ['MEMORY_RESULTS.md','RESULTS.md','STATUS.md']:
 p=ROOT/name;text=p.read_text();text=text.replace('— 사용자 승인으로 일시중단. batch8의10만세포 완료·출력검증, 나머지 및batch256 미완료.','— 2026-10-28 재개 후 batch8/256 실제30만세포 처리·최종출력검증 완료. 최신 보고서 참조.')
 text+='\n2026-10-28 최종 완료: scGPT 실제30만 고유세포 batch8/256 추론 및 전체출력검증 완료. 실행 `'+run+'`, [최종 보고서](scgpt/reports/scGPT_actual_300k.md). 후속 실험 범위는 scGPT만이며 다른 모델은 진행하지 않는다. 최신 상태는 이 항목을 우선한다.\n';p.write_text(text)
(D/'finalization_status.json').write_text(json.dumps({'status':'complete','output_verification_passed':True,'reports_updated':True,'new_gpu_experiments_started':False}));print('FINALIZED',run,flush=True)
