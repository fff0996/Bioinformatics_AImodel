"""Readable input -> GPU utilization + memory overview from actual recorded results."""
from pathlib import Path
import json,datetime
ROOT=Path(__file__).resolve().parents[1];RAM=130660974592/2**30;out=[]
def read(p):return json.loads(p.read_text())
def add(s=''):out.append(s)
def table(headers,rows):
 add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
 for r in rows:add('| '+' | '.join(map(str,r))+' |')
 add()
def mem(x):return f'{x:.3f} GiB ({x/RAM*100:.2f}%)'
def util(a,b):return '미측정' if a is None else f'{a:.1f}% / {b:.0f}%'
add('# 모델별 실제 입력·GPU 사용률·메모리');add();add('갱신: '+datetime.datetime.now().isoformat(timespec='seconds'));add();add('**GPU 사용률(%)은 계산 활동 시간의 비율이고, 메모리(GiB)는 저장 공간이다.** 메모리 괄호 %는 프레임워크의 프로세스 할당량을 전체 공유 RAM 121.688 GiB로 나눈 참고값이며 시스템 전체 메모리 사용률이 아니다. Spark에서 CPU/GPU 메모리는 공유하므로 RSS와 GPU 할당량을 더하지 않는다.');add();add('DNABERT-2의 기존 4,000bp 파인튜닝에서는 10.571 GiB(공유 RAM 대비 약8.69%)를 관측했다. Cellpose 등의 추가 결과는 아래 표를 함께 확인한다. **메모리90%(약109.5 GiB) 도달 조건은 아직 확인되지 않았다.**');add()
add('## 모델을 입력 없이 적재했을 때');add();bl=[]
e=ROOT/'esm2/results/LOADING_LATEST'
if e.exists():
 b=read(e.parent/e.read_text().strip()/'loading.json');bl.append(['ESM-2 650M',format(b['total_parameters'],','),'FP32',f"{b['cuda_after_load_allocated_gib']:.6f}",f"{b['rss_after_load_gib']:.3f}",'새 프로세스, forward 없음'])
for preset in ['alphafold2_ptm','alphafold2_multimer_v3']:
 q=ROOT/'colabfold/results'/('LOADING_'+preset+'_LATEST')
 if q.exists():
  b=read(q.parent/q.read_text().strip()/'loading.json');bl.append([preset,format(b['parameters_per_model'][0]['parameters'],',')+' × 5 sets','FP32 weights / BF16 compute',f"{b['before_first_forward_ready_allocated_gib']:.6f}",f"{b['rss_after_load_gib']:.3f}",'5개 weight set 보관; 첫 모델 선택 뒤 ready 상태'])
q=ROOT/'scvi/results/LATEST'
if q.exists():
 bpath=q.parent/q.read_text().strip()/'loading_baselines.json'
 if bpath.exists():
  b=read(bpath)[0];bl.append(['scVI SCVI 기본 구조',format(b['total_parameters'],','),'FP32',f"{b['cuda_after_load_allocated_gib']:.6f}",f"{b['rss_after_model_load_gib']:.3f}",'학습 모델 로드; AnnData는 CPU에 먼저 읽음; optimizer 없음'])
q=ROOT/'cellpose/results/LATEST'
if q.exists():
 cp=q.parent/q.read_text().strip()/'summary.json'
 if cp.exists():
  c=read(cp);b=c['model_only_postload'];bl.append(['Cellpose-SAM cpsam_v2',format(c['parameters']['total'],','),'BF16 inference / FP32 training',f"{b['allocated_gib']:.6f}",f"{b['rss_gib']:.3f}",'fresh process, first forward before input'])
q=Path('/home/hyein/dnabert2-benchmark/dnabert2_eval/results/20261007_015418_n100000_b8_bp300_fp32_1791338058787754457/summary.json')
if q.exists():
 b=read(q);bl.append(['DNABERT-2',format(b['parameters'],','),'FP32',f"{b['post_load_gpu_allocated_gib']:.6f}",f"{b['post_load_rss_gib']:.3f}",'기존 run post-load 기록 재사용'])
table(['모델','실행 구성 파라미터','정밀도','GPU allocated GiB','프로세스 RSS GiB','측정 범위'],bl);add('가중치 용량 이론값과 위 실측은 다르다. GPU allocator 외 드라이버·runtime과 CPU 데이터도 RAM을 쓰며 RSS에 포함될 수 있다. 로딩 기준과 아래 실행 peak가 다른 run이면 차이는 참고값이다.');add()
headers=['모델·작업','입력당 크기','실제 배치','고유 입력','GPU 평균 / 최대','GPU 할당 peak (공유 RAM 대비 참고 %)']
esm=ROOT/'esm2';base=esm/'results/20261027_164300_125163';sup=esm/'results'/(esm/'results/SUPPLEMENT_LATEST').read_text().strip();rows=[]
for r in read(sup/'summary.json')['conditions']:
 bs=read(base/(r['condition']+'_batches.json'));a=min(b['samples'] for b in bs);b=max(b['samples'] for b in bs);rows.append(['ESM-2 650M 추론',f"{r['length_min']}–{r['length_max']} 아미노산",str(a) if a==b else f'{a}–{b}',r['unique_inputs'],util(r['gpu_util_mean_percent'],r['gpu_util_max_percent']),mem(r['cuda_peak_allocated_gib'])])
t=read(base/'summary.json')['training'];rows.append(['ESM-2 부분 파인튜닝','235–237 아미노산','토큰 예산 4096',128,util(t['gpu_util_mean_percent'],t['gpu_util_max_percent']),mem(t['cuda_peak_allocated_gib'])]);add('## ESM-2');add();table(headers,rows);add('추론은 각 고유 서열 16개를 40회 반복 측정했다. FP32, 토큰 예산 4096. 최대 96%는 별도 기본 실행에서 관측됐고 위 표는 관측 구간을 늘린 보완 실행 수치다. 파인튜닝은 마지막 Transformer block·LayerNorm·회귀 head만 학습했다.');add();add('[상세 보고서](esm2/reports/ESM2_20261027_164300_125163.md)');add()
af=ROOT/'colabfold/reports/metrics.json'
if af.exists():
 rows=[]
 for r in read(af):rows.append(['AF2 Multimer v3' if 'multimer' in r['run'] else 'AF2 PTM',f"{r['original_residues']} 아미노산 (pad {r['padded_residue_axis']})",'1; 모델 5개 순차',1,util(r['gpu_util_mean_percent'],r['gpu_util_max_percent']),mem(r['jax_allocator_peak_gib_since_process_start'])])
 add('## AlphaFold2 — ColabFold');add();table(headers,rows);add('JAX 메모리는 각 프로세스 시작 이후의 **누적 allocator peak**이며 입력별로 초기화하지 않았다. 단일체 3 recycles, 복합체 20 recycles의 별도 preset이며 BF16 계산·FP32 가중치 경로다.');add();add('[상세 보고서](colabfold/reports/AlphaFold2_ColabFold.md) · [비교 CSV](colabfold/reports/comparison.csv)');add()
dna=Path('/home/hyein/dnabert2-benchmark/dnabert2_eval/bend_length_results');rows=[]
for p in sorted(dna.glob('*training*bp*/summary.json')):
 r=read(p);rows.append(['DNABERT-2 파인튜닝',f"{r['settings']['bp']} bp → 평균 {r['tokens_mean']:.1f} tokens",r['settings']['batch'],r['settings']['n'],'미측정',mem(r['gpu_peak_allocated_gib'])])
add('## DNABERT-2 기존 기록');add();table(headers,rows);add('위 기존 길이 실험에는 GPU-Util이 없다. 메모리 비율을 GPU 사용률로 바꿔 해석하면 안 된다. 최신 GPU 사용률 보완 실험은 지정된 후속 순서에 남아 있다.');add()
scvi=ROOT/'scvi/results/LATEST'
if scvi.exists():
 o=scvi.parent/scvi.read_text().strip();s=read(o/'summary.json');rows=[]
 for r in s['training']:
  p=r['training_phase_profile'];rows.append(['scVI 처음부터 학습',f"세포당 {r['genes']} 유전자",128,r['unique_training_cells'],util(p['gpu_util_mean_percent'],p['gpu_util_max_percent']),mem(r['cuda_peak_allocated_gib'])])
 for r in s['inference']:rows.append(['scVI latent 추론',f"세포당 {r['genes']} 유전자",128,r['unique_inputs'],util(r['gpu_util_mean_percent'],r['gpu_util_max_percent']),mem(r['cuda_peak_allocated_gib'])])
 add('## scVI');add();table(headers,rows);add('실행 상태: '+s['status']+'. 학습은 400 epochs, FP32, batch 128을 고정한다. 학습 메모리는 해당 fit 전체(검증 포함)의 peak이고 GPU 평균은 training phase 표본이다. 완료 전 행은 후속 모델 재로드 검증이 남아 있을 수 있다.');add()
q=ROOT/'cellpose/results/LATEST'
if q.exists():
 cp=q.parent/q.read_text().strip()/'summary.json'
 if cp.exists():
  c=read(cp);rows=[]
  for r in c['conditions']:rows.append(['Cellpose-SAM inference',str(r['raw_shapes'][0]),'up to8 tiles, 256x256',r['unique_images'],util(r['gpu_util_avg'],r['gpu_util_max']),mem(r['memory']['peak_allocated_gib'])])
  if 'training' in c:
   t=c['training'];rows.append(['Cellpose-SAM fine-tuning','native520x696 ->256crop','1 crop',16,util(t['gpu_util_avg'],t['gpu_util_max']),mem(t['memory']['peak_allocated_gib'])])
  add('## Cellpose-SAM');add();table(headers,rows);add('Status: '+c.get('status','running')+'. BF16 inference / official FP32 fine-tuning, fixed tile policy. Unique image counts exclude measurement repeats. Training utilization includes label preprocessing and validation.');add();add('[Cellpose 상세 보고서](cellpose/reports/Cellpose.md) · [입력·정답 예시](cellpose/reports/input_example.png) · [비교 CSV](cellpose/reports/comparison.csv)');add()
q=ROOT/'dnabert2/results/LATEST'
if q.exists():
 cp=q.parent/q.read_text().strip()/'summary.json'
 if cp.exists():
  c=read(cp);rows=[]
  for r in c['conditions']:rows.append(['DNABERT-2 '+r['mode'],str(r['bp'])+'bp / mean '+str(round(r['tokens_mean'],1))+' tokens',8,r['unique_inputs'],util(r['gpu_util_avg'],r['gpu_util_max']),mem(r['memory']['peak_allocated_gib'])])
  add('## DNABERT-2 GPU 사용률 보완');add();table(headers,rows);add('기존 학습 모델은 보존했다. 새 학습 행은128개16step 자원 probe이며, 원본1000개 학습 결과에 사용률을 소급해 붙인 값이 아니다.');add();add('[DNABERT-2 보완 보고서](dnabert2/reports/DNABERT2.md) · [비교 CSV](dnabert2/reports/comparison.csv)');add()
q=ROOT/'scgpt/results/finetuned_test_20261023_034345_99523/summary.json'
if q.exists():
 c=read(q);add('## scGPT 기존 완료 결과');add();add('기본 추론·파인튜닝·저장·재로드·test·길이 비교 완료. test4,218세포 중 알려진 라벨4,213세포 기준 Accuracy '+str(round(c['accuracy'],4))+', 학습13클래스 macroF1 '+str(round(c['macro_f1_all_training_classes'],4))+'. GPU 개수/사용률 보완은 별도 후속 순서다.');add();add('[저장 모델 test summary](scgpt/results/finetuned_test_20261023_034345_99523/summary.json) · [길이 비교 CSV](scgpt/results/length_compare_20261023_040313_501423/comparison.csv)');add()
for name,title,report in [('ntv2','Nucleotide Transformer V2','NTv2'),('clair3','Clair3','Clair3'),('rnafm','RNA-FM','RNAFM'),('openspliceai','OpenSpliceAI','OpenSpliceAI')]:
 q=ROOT/name/'results/LATEST'
 if q.exists():
  cp=q.parent/q.read_text().strip()/'summary.json'
  if cp.exists():
   c=read(cp);rows=[]
   for r in c.get('conditions',[]):
    if 'memory' not in r:continue
    rows.append([title+' '+r.get('mode',''),str(r.get('bp',r.get('length','see report'))),'see report',r.get('unique_inputs','N/A'),util(r.get('gpu_util_avg'),r.get('gpu_util_max')),mem(r['memory']['peak_allocated_gib'])])
   add('## '+title);add();add('상태: '+c.get('status','running'));add();b=c.get('model_only_postload')
   if b:add('모델 적재 후/첫 forward 전 CUDA allocated '+str(round(b['allocated_gib'],6))+'GiB; reserved '+str(round(b['reserved_gib'],3))+'GiB; RSS '+str(round(b['rss_gib'],3))+'GiB.');add()
   if rows:table(headers,rows)
   if (ROOT/name/'reports'/(report+'.md')).exists():add('[상세 보고서]('+name+'/reports/'+report+'.md)');add()
q=ROOT/'scgpt/results/SUPPLEMENT_LATEST'
if q.exists():
 c=read(q.parent/q.read_text().strip()/'summary.json');rows=[]
 for r in c['conditions']:rows.append(['scGPT embedding','cap'+str(r['length'])+' / mean '+str(round(r['tokens_mean'],1))+' tokens',8,r['unique_inputs'],util(r['gpu_util_avg'],r['gpu_util_max']),mem(r['memory']['peak_allocated_gib'])])
 if 'training_resource_supplement' in c:
  t=c['training_resource_supplement'];e=t['epochs_detail'][0];rows.append(['scGPT training utilization supplement','max1200 incl CLS',8,t['training_cells'],util(e['gpu_util_mean_percent'],e['gpu_util_max_percent']),mem(e['gpu_peak_allocated_gib'])])
 add('## scGPT GPU 사용률 보완');add();table(headers,rows);b=c['model_only_postload'];add('새 프로세스 추론모델 적재 allocated '+str(round(b['allocated_gib'],6))+'GiB; reserved '+str(round(b['reserved_gib'],3))+'GiB; RSS '+str(round(b['rss_gib'],3))+'GiB. 기존기본/길이/test결과와분리해보완했다.');add();add('[scGPT 상세 보고서](scgpt/reports/scGPT.md) · [보완 비교 CSV](scgpt/reports/comparison_supplement.csv)');add()
add('## 후속 메모리 확대 실험');add();add('사용자가 batch를단계적으로늘려메모리80~90% 도달가능성과처리량을확인하는후속실험을추가했다. **후속실험은별도run으로실행중이다.** 위표는기본고정batch 실험/보완결과이며최대지원batch/최대점유한계가아니다. [후속메모리실험](MEMORY_RESULTS.md)');add()
add('[scVI 상세 보고서](scvi/reports/scVI.md) · [실제 학습 입력 예](scvi/reports/training_input_example.csv)');add();add('[전체 진행 상태](STATUS.md)');(ROOT/'RESULTS.md').write_text('\n'.join(out)+'\n');print(ROOT/'RESULTS.md')
