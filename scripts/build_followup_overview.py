from pathlib import Path
import json
ROOT=Path('/home/hyein/ai-benchmark');RAM=121.68751525878906
read=lambda p:json.loads(p.read_text())
rows=[]
def add(name,task,batch,peak,util,limits,report):rows.append([name,task,str(batch),f'{peak:.3f}',f'{100*peak/RAM:.2f}',f'{util:.2f}' if util is not None else 'N/A',limits,f'[보고서]({report})'])
for name,title,report in [('esm2','ESM-2','ESM2_batch_memory.md'),('cellpose','Cellpose','Cellpose_batch_memory.md')]:
 r=ROOT/name/'results';s=read(r/(r/'BATCH_LATEST').read_text().strip()/'summary.json');c=s['conditions'][-1];add(title,'추론',c.get('batch_size',c.get('actual_max_tile_batch')),c['memory']['peak_allocated_gib'],c['gpu_util_avg'],'다음 예약량 예측 guard; 실제 OOM 아님',name+'/reports/'+report)
s=read(ROOT/'scgpt/results/actual_300k_resume_20261028_093701_696230/whole_dataset_comparison.json')[-1];add('scGPT','실제30만세포 임베딩 추론',s['batch'],s['allocated_peak_gib_max_segments'],s['gpu_util_mean_sample_weighted'],'같은30만세포 batch8/256 비교','scgpt/reports/scGPT_actual_300k.md')
for name in ['scvi','dnabert2','ntv2','rnafm','openspliceai']:
 r=ROOT/name/'results';s=read(r/(r/'BATCH_LATEST').read_text().strip()/'summary.json');assert s['status']=='complete';c=s['conditions'][-1];add(name,'추론',c['batch_size'],c['memory']['peak_allocated_gib'],c['gpu_util_avg'],s['stop_reason'],name+'/reports/Batch_memory_followup.md')
r=ROOT/'clair3/results';s=read(r/(r/'BATCH_LATEST').read_text().strip()/'summary.json');assert s['status']=='complete'
for kind in ['Clair3_P','Clair3_F']:
 c=[c for c in s['conditions'] if c['model']==kind][-1];add('Clair3 '+('pileup' if kind.endswith('_P') else 'full-alignment'),'실제 후보 network 추론',c['batch_size'],c['memory']['peak_allocated_gib'],c['gpu_util_avg'],'확보한 고유feature pool 끝; 최대한계 아님','clair3/reports/Batch_memory_followup.md')
r=ROOT/'colabfold/results';s=read(r/(r/'BATCH_LATEST').read_text().strip()/'summary.json');assert s['status']=='complete';w=s['workers'][-1];c=w['conditions'][0];add('AlphaFold2 — ColabFold','실험적 FP32 JAX vmap',c['batch'],c['jax_memory_stats']['peak_bytes_in_use']/2**30,c['gpu_util_avg'],'실제4query 끝; JAX peak는컴파일/warmup포함 누적값','colabfold/reports/Batch_memory_followup.md')
order={'ESM-2':0,'AlphaFold2 — ColabFold':1,'scvi':2,'Cellpose':3,'dnabert2':4,'ntv2':5,'Clair3 pileup':6,'Clair3 full-alignment':7,'rnafm':8,'openspliceai':9,'scGPT':10};rows.sort(key=lambda r:order[r[0]])
lines=['# 10종 후속 실험 최종 결과','','10종 모두 후속 실행과 각 범위의 출력 검증·저장·보고서를 완료했다. 모델별 후속 범위와 정밀도·길이·메모리 정의가 다르며 같은 과제/조건의 모델 간 성능 순위로 해석하지 않는다. 아래 배치는 **시험한 최대 성공 조건**이며 이론적 최대 지원 배치가 아니다.','','|모델|후속 작업|최대 시험 성공batch|메모리 peak GiB|공유RAM대비%|GPU평균%|종료/해석 제한|상세|','|---|---|---|---|---|---|---|---|']
lines+=['|'+ '|'.join(r)+'|' for r in rows]
lines+=['','메모리 비율의 분모는CPU/GPU공유RAM121.688GiB다. GPU 연산사용률과 메모리 비율은 다른 지표이고, 전용VRAM점유율이 아니며RSS와CUDA를더하지않는다. PyTorch행은CUDAallocated peak다. AF2는JAX프로세스누적allocator peak로컴파일/warmup을포함하므로실행구간peak와직접비교하지않는다. 예약reserved와sampled RSS/가용량도각보고서에별도로있다.','','## scGPT 별도 학습 용량 결과','','FP32/1200열고정padding 학습probe의최대성공batch292: allocated100.750GiB,전체공유RAM대비82.79%. 이는위표의추론batch256·5.477GiB와다른실험이다. 90%는관측하지못했다. [학습용량보고서](scgpt/reports/scGPT_memory_batch.md)','','## 마지막 두 모델의 구현과 검증','','Clair3는실제HG003 BAM에서전체candidate tensor를추출해pileup/full-alignment를따로배치확대했다. 원래native pipeline의동일입력/확률대조와저장·재로드를검증했다. 후보좌표를내보내지못해생물학적고유위치수가아닌중복제거feature수로보고한다. VCF호출전체시간과candidate network시간은구분한다.','','AlphaFold2는CLI기본배치가아닌JAX vmap 별도실험이다. 단량체model_1/115AA실제4query/기존MSA/3recycle을사용했다. BF16벡터화참조검증은구조차이로실패해보존했고,FP32고정별도조건의batch1/2/4로실행했다. batch1과native참조CA정렬RMSD0.1795Å/pLDDT최대차0.1875로사전0.5Å기준통과. 출력PDB·좌표·pLDDT·query수검증을수행했고batch별수치차이도보존했다. 기존기본BF16·5모델순차예측과동일조건으로표현하지않는다.','','기존미실시/중단/실패기록은인수인계이력으로남기며현재최신상태는이최종표를우선한다. 원본run/명령/로그/출력/환경/입력선택/자원CSV는각모델results와보고서에있다. 추가GPU모델작업은자동시작하지않는다.']
(ROOT/'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines)+'\n')
for name in ['RESULTS.md','MEMORY_RESULTS.md','STATUS.md']:
 with (ROOT/name).open('a') as f:f.write('\n10종 후속최종정리완료. 이전AlphaFold2/Clair3미실시상태는새실행으로갱신됐다. 구현·정밀도·검증범위차이를포함한최신전체표: [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md). 추가GPU작업없음.\n')
print('FOLLOWUP_OVERVIEW_COMPLETE')
