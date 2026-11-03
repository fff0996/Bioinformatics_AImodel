# Clair3 최종 통합 보고서

## 1. 모델과 실제 입력

|항목|이번 실험에서 사용한 모델|
|---|---|
|체크포인트·버전|Clair3 v2.0.4,ONT pileup.pt/full_alignment.pt|
|구조|pileup 양방향LSTM 및 full-alignment CNN 후보분류 네트워크|
|전체·구성별 파라미터 수|Clair3_P2,074,520개(약207.5만 개)(2.075M);Clair3_F2,986,522개(약298.7만 개)(2.987M);단계별 실행|
|정밀도|FP32|
|모델 입력 → 출력|long-read BAM+참조유전체 →후보특징 →SNV/Indel VCF|
|가중치와 실행 메모리의 차이|파라미터 수×dtype bytes는 이론 가중치 용량이다. 실행 peak에는 입력·중간 활성값이 더해지고,파인튜닝에는 gradient/optimizer 상태도 추가된다.|
|메모리% 정의|CUDA allocated peak / CPU·GPU 공유 RAM121.6875GiB ×100; GPU-Util과 별개. AlphaFold2는 JAX 계측을 별도로 표시|

### 실제 데이터와 학습 방식 구분

|구분|데이터·실제 처리 내용|
|---|---|
|추론 데이터|HG003 ONT 실제BAM+GRCh38 chr20소구간. cache 고유feature P5708/F894|
|학습·검증·test 데이터|이번 작업에서 학습·파인튜닝 미실시. 본 실험용 훈련 정답tensor 없음|
|학습 입력 X / 정답 y|추론 X:P[B,33,18],F[B,89,33,8]. GIAB truthVCF/평가구간은 변이검출 정확도 평가용이며 학습에 사용하지 않음|
|가중치 갱신 방식|공식학습기능 존재와 별개로 이번에는 사전학습ONT 가중치의 추론만 수행|
|원본·변환 입력 저장 위치|`data/HG003_chr20_demo.bam;data/GRCh38_no_alt_chr20.fa;candidate run/*_inputs.npy`|
|해석 범위|후보batch는read개수가 아님. 작은구간 후보pool을 전부 사용한 결과는 전장유전체 최대메모리 한계가 아님|

**용어:** 처음부터 학습=무작위 초기화에서 시작; 전체/일부 층 파인튜닝=공개 가중치에서 해당 부분 갱신; 분류기만 학습=본체를 동결하고 head만 갱신. **용량probe는 학습 방식이 아니라 짧은 자원 측정 목적**이며,실제로 수행한 가중치 갱신 방식과 함께 표기한다. 원본 전체 데이터 수와 해당 run에서 실제 사용한 부분집합 수를 구분한다.

### 실제 파일 → 전처리 → 모델 표현 → 최종 출력: 단계별 실제 예시

**Clair3는단백질임베딩모델이아니라 정렬된read에서후보변이특징을만들어변이확률을계산한다.** 이예제는실제cache의첫후보이며,캐시에genomicposition ID가저장되지않았으므로BAM의특정read·표시된VCF행과동일좌표라고연결하지않는다.

#### 0–1. 원본 입력과 평가 정답

입력은 `data/HG003_chr20_demo.bam`, `data/GRCh38_no_alt_chr20.fa`및BAM index다. BAM의정렬read·염기품질·mapping정보와참조유전체에서후보특징을계산한다. GIAB truthVCF와평가구간은정확도평가용이며이번에학습한라벨tensor가아니다.

#### 2. 후보 특징 tensor

|모델분기|입력shape|dtype|
|---|---|---|
|Clair3_P|[1, 33, 18]|int8|
|Clair3_F|[1, 89, 33, 8]|int8|

pileup의33위치×18특징과full-alignment의89read행×33위치×8채널이다. batch는DNA문자수나원본read총수가아니라동시에처리하는후보tensor수다.

#### 3–4. 분기별 특징 계산

pileup은양방향LSTM,full-alignment는CNN경로로후보의표현을계산한다. **이CPU확인에서는그내부hidden tensor를캡처하지않았으므로실제중간embedding값은N/A**다. 존재하지않는평균임베딩단계를추가하지않는다.

#### 5. 최종 후보별 확률

- **Clair3_P**:출력[1, 24],앞24개실제값 `[0.000028, 0.000000, 0.003323, 0.000000, 0.000001, 0.000003, 0.000000, 0.996599, 0.000007, 0.000000, 0.000003, 0.000000, 0.000000, 0.000002, 0.000000, 0.000024, 0.000000, 0.000000, 0.000011, 0.000000, 0.000000, 0.000856, 0.982968, 0.016176]`. 원GPU같은행과최대차이 2.980e-08.
- **Clair3_F**:출력[1, 90],앞24개실제값 `[0.997822, 0.000000, 0.000005, 0.000002, 0.000002, 0.000000, 0.000000, 0.000002, 0.000000, 0.000001, 0.000002, 0.001639, 0.000000, 0.000000, 0.000000, 0.000000, 0.000525, 0.000000, 0.000000, 0.000000, 0.000000, 0.991283, 0.000006, 0.008711]`. 원GPU같은행과최대차이 1.192e-07.

P의24출력은GT21+genotype3, F의90출력은GT21+genotype3+indel길이33+33개head를연결한것이다. 전체90개를단일softmax90클래스로해석하지않는다.

#### 6. 후보확률 → 실제 VCF·평가

공식pipeline은후보별확률과위치·ref/alt정보를조합해VCF의CHROM/POS/REF/ALT/QUAL/GT등을작성한다. cache예제의좌표mapping이없어이확률과특정VCF행의직접대조는미검증이다. 원래BAM→VCF실행·truth평가결과는아래상세에보존했다. 훈련·파인튜닝은수행하지않았다.

#### 7. 저장 파일

|단계별 저장 배열|실제 shape·dtype|의미|
|---|---|---|
|Clair3_P_input|[1, 33, 18] int8|pileup후보입력|
|Clair3_P_probabilities|[1, 24] float32|pileuphead확률|
|Clair3_F_input|[1, 89, 33, 8] int8|full-alignment후보입력|
|Clair3_F_probabilities|[1, 90] float32|full-alignmenthead확률|

로컬 전체 배열: `results/real_input_capacity_20261102_173336_910773/stage_trace_cpu/arrays.npz`. `arrays.npz::key`는 같은NPZ 파일 안의 배열이름이다. 아래 재현코드로 저장모델과 입력에서 다시 확인할 수 있다.

**값의 출처:** 저장된 모델을 CPU에서 새로 재확인한 중간값이다. 원GPU 자원계측 중에 캡처했던 것처럼 해석하지 않는다. 이 CPU 확인시간은 기존 GPU 벤치마크 시간에 포함하지 않는다.

### 모델 적재 직후 — 첫 입력 forward 이전

|적재 조건|allocated GiB(JAX는별도표기)|reserved GiB|프로세스 RSS GiB|원본 근거|
|---|---|---|---|---|
|추론/기본 모델 적재 기준 (Clair3_F)|0.011145|0.0234375|0.620361328125|clair3/results/real_input_capacity_20261102_173336_910773/Clair3_F_batch_894/summary.json|
|추론/기본 모델 적재 기준 (Clair3_P)|0.007729|0.0234375|0.7008628845214844|clair3/results/real_input_capacity_20261102_173336_910773/Clair3_P_batch_5708/summary.json|
|추론/기본 모델 적재 기준 (분기 미기록)|0.011145|0.025390625|0.6190414428710938|clair3/results/clair3_20261027_203339_768667/summary.json|

위 값은 **가중치를 GPU에 적재하고 첫 입력을 계산하기 전의 프레임워크 할당량**이다. CUDA/JAX context·driver·시스템 전체 GPU사용량과 같지 않으며 CPU입력 준비 포함 여부는 해당 실행 코드를 따른다. **실행 peak와 모델 적재 기준값의 차이는 같은 run에서만 계산**한다. 파인튜닝 중 추가량에는 입력뿐 아니라 중간값·gradient·optimizer 상태가 포함된다. 서로 다른 조건/정밀도의 기준값을 하나의 모델 고유 메모리로 단정하지 않는다.

## 2. 실행 조건과 측정 기준

<!-- ENV_MEASUREMENT_OVERVIEW_START -->
### 실행 환경·자원 측정 방법 요약

|항목|이번 보고서의 측정 기준|
|---|---|
|장비·공유메모리|DGX Spark,NVIDIA GB10,ARM64;시스템공유RAM121.6875GiB. CPU/GPU RAM을별도로더하지않음|
|환경기록출처|clair3/results/clair3_20261027_203339_768667/summary.json|
|Python·프레임워크|{'torch': '2.13.0a0+8145d630e8.nv26.06', 'cuda': '13.3'}|
|실험병렬정책|GPU벤치마크한번에1개. 이단계별CPU검증은별도작업이며GPU측정시간과합치지않음|
|GPU-Util|nvidia-smi utilization.gpu(%);관측기간중커널활동시간비율. 최대FLOPS/메모리비율아님|
|GPU메모리|PyTorch allocated/reserved및peak;JAX는bytes_in_use/peak_bytes_in_use별도. 모델적재직후·실행구간을분리|
|프로세스·시스템메모리|psutil RSS,MemAvailable,swap;시계열의sampled peak/최소값이며정확한연속최고치아님|
|표본간격|기본0.5초대기+GPU조회시간;측정구간의평균·최대·표본수저장. 짧은실행은동일조건반복하고고유입력과반복처리량구분|
|시간동기화|PyTorch구간앞뒤CUDA synchronize;JAX block_until_ready. 비동기호출반환시간만측정하지않음|
|시간포함범위|조건별로입력준비/전송/forward/backward/optimizer/validation/save포함여부를기록. 서로다른범위의시간을직접배수비교하지않음|
|loading·compile·warmup|구간을분리한다. JAX누적allocator peak가compile/warmup을포함하는경우본문에명시하고CUDA구간peak와직접동일시하지않음|
|메모리기준값과증가량|첫forward전모델적재기준값과같은run의실행peak를비교. 증가량에는학습gradient/optimizer/중간활성값이포함|
|누락·관측한계|미지원/미관측값은N/A. 장치전체GPU-Util에는다른프로세스·디스플레이활동이섞일수있음. 단일실행은통계적반복검증으로쓰지않음|

<!-- ENV_MEASUREMENT_OVERVIEW_END -->

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

<!-- FACTOR_EFFECTS_START -->
### 시간·메모리에 영향을 주는 요인 — 통제 비교 상태

아래 원리는 모든 모델에서 정량 검증을 마쳤다는 뜻이 아니다. **실측 비교와 미실험을 구분**한다. 배치 sweep 중 정밀도·입력길이·체크포인트를 고정하며,다른 과제/다른head 결과를 같은 조건으로 섞지 않는다.

|요인|총시간·처리량 해석|최대 allocated 해석|이 모델의 실험 근거/상태|
|---|---|---|---|
|배치|한step은 길어질 수 있고step수는 줄어든다. 처리량 상승/포화는 실측으로 판단|같은입력길이·정밀도에서 동시중간값 증가가 주된영향|실제후보P5708/F894까지batch비교|
|샘플당 실제입력 크기|GPU에들어간실제token/타일 크기가 커졌을때 계산량 변화 비교|토큰길이·패치/타일크기·attention구조에 따라 증가;원본크기와 다를수있음|구간·coverage 실험과후보네트워크시간 분리;reads수와candidatebatch 구분|
|고유 데이터수|같은batch·epoch에서 총처리step가늘어 총시간이늘어나는지 비교|고정batch의동시입력이같으면 데이터총수만으로peak가크게늘지않을수있음|입력개수비교는본문참조;새지도학습데이터수비교는아래결과가생긴조건만완료|
|epoch|같은train입력을 반복하므로 누적시간 증가;검증성능개선은 보장안됨|동일batch·입력에서epoch자체가peak를선형증가시키는요인은아님|훈련 미실시;epoch 비교 해당 없음|
|학습방식·갱신범위|추론/본체동결head학습/부분·전체미세조정/처음부터학습은 계산범위가다름|gradient·optimizer·역전파용중간값의유무가다름|1절의가중치갱신방식과X/y를참조;다른데이터·목적의시간을단순배수비교하지않음|
|모델·정밀도·전처리|구조·dtype·토큰화·타일화·컴파일·후처리도영향|모델적재기준값과실행중추가량을분리|원인후보/고정조건으로기록;모델크기나dtype의단독영향sweep은미실험|

추가 통제 비교에서는 **batch8·FP32·공개초기weight·seed42·optimizer·원래전체pool의padding폭·검증/test를고정**하고,(1)같은train의epoch1/2/4,(2)epoch2의중첩train1/4·1/2·전체를각각실행한다. 전체데이터를바꾸면서epoch도함께바꾼행끼리는단독효과로해석하지않는다. 단일실행이며데이터부분집합의label분포변화는정확도해석의제약이다. 이비교는해당지도학습지원모델에만적용한다.


추가epoch/데이터수통제비교 수치: **아직미실험/대기**. 기존측정을추정해서채우지않는다.

<!-- FACTOR_EFFECTS_END -->

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|Clair3_P 후보추론|실제HG003 ONT 후보feature고유1개|batch1×[33, 18],FP32모델|18096회/총18096후보|15.00초|1206.37후보/초|0.057 GiB / 0.05%|native pipeline확률대조통과;새variant정확도평가아님;80%미도달|
|Clair3_P 후보추론|실제HG003 ONT 후보feature고유5708개|batch5708×[33, 18],FP32모델|25회/총142700후보|15.10초|9447.45후보/초|4.111 GiB / 3.38%|native pipeline확률대조통과;새variant정확도평가아님;80%미도달|
|Clair3_F 후보추론|실제HG003 ONT 후보feature고유1개|batch1×[89, 33, 8],FP32모델|7026회/총7026후보|15.00초|468.34후보/초|0.178 GiB / 0.15%|native pipeline확률대조통과;새variant정확도평가아님;80%미도달|
|Clair3_F 후보추론|실제HG003 ONT 후보feature고유894개|batch894×[89, 33, 8],FP32모델|116회/총103704후보|15.04초|6896.99후보/초|1.499 GiB / 1.23%|native pipeline확률대조통과;새variant정확도평가아님;80%미도달|
|학습|이번작업미실시|—|—|미실시|—|미측정|학습지원기능과이번실행을구분|

<!-- REAL_TRAINING_V3_START -->
### 실제 데이터 추가 실험 — real_input_capacity_20261102_173336_910773

진행 중인 실행의 완료 조건만 아래에 갱신한다. probe는 짧은 용량 측정이며 전체 학습이나 평가 완료가 아니다. 메모리%는 CUDA allocated peak / 시스템 공유 RAM ×100이며 GPU-Util과 다르다. 독립 실행 1회 기준.

|조건|가중치 갱신 방식·실험 목적|상태|고유 입력|배치|epoch|optimizer steps|실측 시간(s)|입력/s|allocated peak GiB|reserved peak GiB|공유 RAM 대비 allocated %|GPU-Util 평균%|최대%|독립 test 지표|저장 모델 재로드|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Clair3_F_batch_894|추론(가중치 갱신 없음)|complete|894|894|N/A|N/A|15.1108|7040.4114|1.4991|2.4219|1.2319|91.7143|96.0000|N/A|N/A|
|Clair3_P_batch_5708|추론(가중치 갱신 없음)|complete|5708|5708|N/A|N/A|15.3003|9699.6523|4.1114|6.7832|3.3787|95.2143|96.0000|N/A|N/A|

이 run의 추론 capacity 실험과 지도학습 실험은 입력 과제·길이·학습 헤드가 다를 수 있어 각각 비교한다. 3step probe는 전체 학습이 아니다. full_training은 동일한 공개 초기 backbone와 seed42,FP32,AdamW(lr1e-5,weight_decay0.01),고정 전체 split pool padding으로2epoch 수행한다. 각 학습 조건의 원본 summary에 split ID,실제 길이·padding 모양,사용 헤드·학습 범위,라벨,gradient가 관측된 파라미터 수,검증/test지표,저장 모델 재로드 차이를 기록한다. scVI는 아래 설명이 적용되지 않는 별도 처음부터 학습 모델이며 AF2·Clair3에는 이 공통 학습 절차를 적용하지 않는다.

입력 및 메모리 기준값(같은 실행에서 측정):

|조건|실제 모델 입력 모양/정책|모델 적재 직후 allocated GiB|실행 시작 allocated GiB|peak 추가 allocated GiB|프로세스 RSS sampled peak GiB|시스템 가용 RAM sampled 최소 GiB|
|---|---|---|---|---|---|---|
|Clair3_F_batch_894|[894, 89, 33, 8]|0.0111|0.0424|1.4567|1.2525|111.2624|
|Clair3_P_batch_5708|[5708, 33, 18]|0.0077|0.0390|4.0725|0.9269|107.0182|

실행 상태: `measured`; 측정범위·제약: `actual_unique_BAM_candidate_cache_limit; no synthetic candidates or pseudo truth labels`.

실행 코드 블록의 파일은 원문SHA256으로 검증하며, 명령JSON에 기록된 실행 경로로 해당 snapshot을 복사하면 복원할 수 있다. 대용량 원본 데이터·가중치는 아래 전처리/다운로드 코드와 출처를 사용하고 로컬 원본도 보존한다.

로컬 원시 결과: `/home/hyein/ai-benchmark/clair3/results/real_input_capacity_20261102_173336_910773`. 메모리 부족/guard 실패는 성공 조건으로 기재하지 않는다.
<!-- REAL_TRAINING_V3_END -->

## 4. 결과 해석

<!-- MODEL_SUMMARY_START -->
### 요약 결론

- **학습 방식:** 공식학습기능 존재와 별개로 이번에는 사전학습ONT 가중치의 추론만 수행.
- **입력 구분:** 후보batch는read개수가 아님. 작은구간 후보pool을 전부 사용한 결과는 전장유전체 최대메모리 한계가 아님.
- **최신 추가 run 최대 성공 메모리:** `Clair3_P_batch_5708` **4.111GiB /3.38%**. 이 값은 해당 최신run의 성공 조건에서의 최대이며,다른 run의 최대와 구분한다. probe이면 전체 epoch·독립test 평가 완료를 뜻하지 않는다.
- **학습 상태:** 이번 실행에서 훈련·미세조정은 수행하지 않았다. 참고 구조/진실변이는 평가 자료이며 학습 정답으로 사용한 것이 아니다.
- **시간·정확도 해석:** 결과는 같은 행의 입력·배치·학습범위에만 연결한다. 메모리 상승 자체가 처리량·정확도 상승을 보장하지 않는다.
<!-- MODEL_SUMMARY_END -->

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### Clair3 — DGX Spark long-read 변이 호출


run `clair3_20261027_203339_768667` · 상태 **complete** · summary (원본 기록 로컬 보관: `clair3/results/clair3_20261027_203339_768667/summary.json`) · 로그 (로컬 보관: `../results/clair3_20261027_203339_768667/benchmark.log`) · 자원 시계열 (원본 기록 로컬 보관: `clair3/results/clair3_20261027_203339_768667/resources.csv`)

#### 모델과 실제 실행 환경

Clair3 공식 소스 v2.0.4, commit `9d1dd3f2475072b6255900e406c34c6b0545fc88`. v2.0.4 이름의 git tag는 없어 실제 main commit과 shared/version.py를 고정했다. 공식 PyTorch `ont/pileup.pt`와 `ont/full_alignment.pt`를 사용하며 Guppy3.6 ONT 데모에 맞춘 구형 ONT 모델이다. 최신 R10.4.1/Dorado 모델과 혼용하지 않는다.

환경 {"torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "gpu": "NVIDIA GB10"}

원래 NGC PyTorch/CUDA 이미지를 보존하고 별도 ARM64 이미지를 만들었다. CFFI native 전처리는 시스템htslib으로 컴파일, phasing은공식 기본whatshap, 선택적 가속용PyPy 대신 CPython3.12를 썼다. Spark의 nvidia-smi memory.free=N/A 때문에 공식 VRAM 잔량 기반 worker 산정을 그대로 쓸 수 없어, GPU predictor worker1개로 명시하는 호환 변경을 적용했다. 모형 forward·배치·변이 호출 알고리즘은 유지했다. 원본·patch는 `results/benchmark_compatibility.patch`에 저장했다.

공식 GPU 배치 정책 predictBatchSize200×5를 고정하며 마지막 batch는 후보 수에 따라 작다. CPU threads4, GPU0, FP32가중치, 실제 입력 특징은 pileup int32/full-alignment int8이며 모델 내부에서 float로 변환/스케일한다. GPU 다른 실험은동시에 실행하지 않았다.

| 모델 | 실제파라미터 | 적재 allocatedGiB | reservedGiB | RSSGiB | 가용GiB |
| --- | --- | --- | --- | --- | --- |
| Clair3_F | 2986522 | 0.011 | 0.025 | 0.619 | 113.028 |
| Clair3_P | 2074520 | 0.008 | 0.023 | 0.698 | 112.746 |

이론 FP32가중치 용량은각 파라미터수×4byte다. 위는 각 worker의 checkpoint 적재를 완료하고 첫forward전의 실측이며 입력tensor/optimizer는없다. 모델 두개는순차적으로 쓰며 파라미터합을한번에 올린 크기로 가정하지 않는다.

#### 실제 입력과 정답

공식 ONT quick demo: GIAB HG003, Guppy3.6, GRCh38_no_alt chr20, chr20:100000–300000(BED0-based half-open). 원본 BAM은coordinate정렬·index·참조 contig길이를 확인했다. GIAB HG003 v4.2.1 VCF와 confident BED도 확보했다. 데이터·모델 URL/바이트/SHA-256은 `data/sources.json`과 `models/ont/sources.json`.

첫 실제 read: {"name": "dc9c6024-1ac4-499b-87b8-2d7e64475b46", "reference_start_0based": 60000, "cigar": "22735S14M1D32M1I6M3D4M1D13M1D21M1D103M1I79M1I80M1D54M1D12M1I19M1I11M1I25M1D79M1D6M1D18M2D88M2D6M1D45M1D9M1D46M1D24M1D9M1I59M1I2M1D32M1D38M1D31M1D46M1D45M1D29M1D5M2D67M1D57M2D11M2D52M1I7M1D24M1D129M1D9M1D13M1I9M3D11M1529I31M1D27M3D4M1D19M1D3M1I9M2D23M1D21M3I11M2D12M1D2M1D50M2D27M1D99M1D10M1D84M1D14M1D29M3D33M2D68M2D49M1D19M1D27M2D28M1D26M1D11M1D45M1D22M1D42M2D47M1I6M2D34M1D40M1D39M1I5M1D40M2D9M1I9M3D37M1D4M1D24M3D42M2D5M1D44M1D2M1I5M1I7M1D4M1D15M1I15M1D23M1I1M1I12M1D9M1I47M1I69M1D19M5I35M1D70M5D63M45D590M1I28M1I46M1D2M2D96M1D1M2D4M2D19M1I11M1D50M2D25M2D4M1I6M1D54M2D58M1D1M2D39M1D1M3D1M1D28M7D72M1D71M1D23M3I4M1I5M1I77M1I2M2D15M1I37M1I82M2I7M1D32M1D134M3D48M1D1M1D19M1I8M1D4M1I115M1D52M2D10M1D15M1D24M1D2M1D77M2D4M1D90M1D30M1I39M1D2M2I5M4I34M1I99M2D24M1D49M2I20M1D35M2D73M2D39M1D4M1D4M1D63M2D47M1D11M1D4M1D4M1D38M1D153M1D30M9I146M2D12M1D7M1D3M1I27M1I41M1I2M1I28M2D25M1D40M1D9M1I4M1I59M1D59M1D4M1D33M2D4M1D2M1I42M1D32M1D19M1D4M1I30M1D10M2D4M1D3M2D19M1I18M1D34M1D4M1I5M1D6M1D4M1D3M2D3M2D8M2D14M3D2M1D18M2D2M1D6M1D4M1D37M1D13M1D3M1I38M2D4M1D22M1D81M1D2M1I18M3D5M1D47M1D74M1D18M1D76M1I31M3I17M2D9M1I32M1I54M1D8M2D92M1I8M1D109M1I50M1I32M1D10M3D6M1D13M2D37M1D112M1I14M1I8M1I16M2D120M1D1M2D51M1D5M2I67M2D6M1D53M1D95M3D33M1D41M1D14M3D5M1D47M2D91M1I31M2D13M1D76M1D1M1D63M1D19M1D92M1D7M5D11M1I10M1D11M1I87M1D68M2D6M1D32M1D3M1D8M3I5M1I37M1D58M3I76M2I30M3I9M2I39M1I15M2I189M1D17M2D6M1D4M1I2M2D9M1I55M2I43M2D150M1D28M1D68M2D1M2D47M1D30M1I126M1I13M1D28M1I12M1I50M1D62M2I53M1D43M1D29M1I37M1D9M2D17M2D13M1D6M1D141M1I13M1D11M2D33M3D210M1D7M2D11M2D10M3I11M4D88M1D40M1D52M1I22M1D66M1I52M2I12M3I33M1D110M1D46M1D1M1D2M1D23M1I5M1I43M1I57M1I49M1D172M1D49M1I105M1I89M1D27M1I2M1I83M1D68M1D54M1I19M1D52M1D6M1D18M2D6M2D42M1I49M1D20M1I81M1D146M2D65M1I138M1I2M1I78M2I19M1I45M1D19M2I76M2D17M1I19M1I48M1I14M1D292M1I4M1D5M1D27M1I64M1I29M1D72M1I77M1D78M2D20M1D30M1D10M1I29M4D9M1I47M1D68M1I70M2I2M2D94M1D54M3D125M1D169M1D30M1D151M2D5M2I2M2D30M1I52M1D140M1D41M1D134M1D60M3D4M2D39M1D27M1I66M2D19M1D3M1D18M1I48M1D2M2I11M1D66M1I110M1I69M2I83M2D24M1I103M1D117M1D118M2I28M1D10M2D35M2I21M1I7M1D33M2D92M2D19M2D74M1I108M1D24M1D76M1I98M1I8M1D84M1D97M2D93M2I68M1I3M1D38M1D15M2D21M2D35M2D35M1I142M1D17M2I7M1D35M1D20M1D49M1D34M1D117M1D15M1D52M1D159M1D34M1D173M1I70M1D2M1D93M1D14M1D4M1I53M1I18M1D78M1D22M1D157M1D4M1D33M4D3M2D5M3I5M4D20M2I51M1D10M2I102M1I5M1D14M1I6M2I7M1I6M1D3M1D31M2D38M3D11M1D14M1D13M1D35M1I12M1D72M2I82M1D31M2I10M2D30M1D37M1D29M4D32M1D28M1D2M1I4M4D4M1D16M2D40M2D2M1D2M1D2M1D5M1D13M2D20M1D32M1D146M8I9M1I9M1I7M2I27M1D3M1D56M2D15M1D38M1I15M5D123M1I9M2I12M1I223M1D268M1I22M2D70M1D3M1D65M1D29M1I29M1D10M1D42M1D2M1D6M3D2M1D29M1D33M2I30M1D88M2D47M1I7M1I31M1D3M4D3M3D14M1I20M2D68M1D13M1D3M1D20M3I82M1I15M1D48M3I18M1I39M2D125M1I3M1I2M1I36M1I2M1I1M3I15M2D63M1D10M1D11M4D62M1D36M1D3M1D10M2D47M2D29M2D47M2D2M4D2M3I3M1D18M1D2M1D18M1D8M2I5M1D9M2D4M2D25M1D12M2D48M1I53M2D97M1D33M1D78M1D15M1D24M1D59M1D24M1D19M4D181M1I15M4I25M1I8M1D28M1D102M2D22M1D2M1D10M1D12M6D85M1I81M1D119M1I23M1D11M2I17M5D3M1D4M2D9M1D6M1D7M2D90M1D8M5D33M2I38M1I6M1D17M2D1M1I87M2D29M1I31M1I44M2I14M2D8M2D48M1D75M1I50M1D6M1I47M1D32M1D38M2D33M1D50M2D17M3I6M1D33M1I29M1I64M1I1M1I6M5I15M2D14M2D12M4D4M2D16M1D7M1I96M1D108M1D22M3I14M2D22M1D19M1D3M1I47M2D29M2D20M3D82M1D15M1D4M4D33M2D80M2I70M1D24M1I21M3I50M1I4M1I5M1I35M1D73M1D58M1D5M3I114M1D4M1D18M1I37M2D18M1I25M2D34M2I51M1I35M2D4M1I9M1D7M1I8M2D24M2I9M1I33M2D4M1I1M2I33M1D56M1I3M1I5M1D38M2I16M1D34M1I23M1D56M1I76M1I32M3D29M4D4M1D7M1D10M10I68M1I32M1D90M1I48M2D21M1D5M1D7M1D2M3D23M1I14M1D40M1I58M3D21M1I85M3D138M1D71M7D4M1I5M1D6M2I11M2I29M2I3M2I26M3D5M1D11M1I17M2I52M3D4M1I60M3D50M1D6M1I85M1D5M1I46M1D61M1I61M1D5M1D59M3D20M2D138M1D40M2I120M1D52M1D133M1D24M1I43M1D21M2I6M1D24M6I128M3D13M4D87M1D10M1D10M2D4M1D14M1D1M1D27M1D12M1D144M1I1M1I29M1D28M1I6M1D8M3D4M1I69M1I33M1D36M1I2M1I11M1I53M1I83M1D13M1I9M2D10M2D16M1D7M2D28M1I2M2D4M1D51M1I69M1D88M2D71M2D59M1I56M2I21M1D131M1D31M3D6M1I96M4D46M1I29M1D5M1D73M1I150M1I40M1I77M1D115M3D9M1D43M1I46M1D37M1D10M1I89M3I27M1D4M1D51M1D52M1D54M1D113M1D29M1I131M1D5M1D43M3D41M3I34M1D30M1I89M4D13M1I4M1D8M1I85M1D130M1I15M1D40M1I15M1D116M4D36M1D21M1D101M1D37M1D77M1I8M1I46M3I4M1D6M1D63M1D62M1I46M2D14M2D13M3D173M1D47M1D72M3I43M1D100M1D4M1D32M1I33M1D86M2D52M4I64M1D1M1D108M1D36M3D18M1D53M4D38M1I55M2D16M1D5M1D49M1D34M1D54M1D46M1I27M2I8M2D126M1D24M1D129M1D61M2D57M1D25M2D21M2D15M1I44M1D84M3D4M2I25M3D1M1I2M1I3M1I26M1D4M1I145M1I28M1D43M2I14M1D12M1D85M2I10M1I43M2I96M1I2M2D20M3I28M1D33M1D49M1I12M2D13M1D36M1I95M1I24M1D28M1I51M1D27M2D4M2D49M1D45M1D35M1D50M2D150M1D24M2D11M3D5M1D31M1I26M1I45M1D12M1D37M1I98M1D30M1D101M2D31M1D9M1D229M1I62M1I135M2D44M2D45M1I16M1I55M1D130M1I198M2D5M2I55M2D28M1D15M2D8M1I23M3D45M1I34M1D3M1D23M1D4M1D30M1D6M1I3M1D3M7D3M1I4M2I10M1I7M3D3M1I215M1D8M1D42M2I31M2D9M1D17M1I73M1D33M1D1M1D64M1I32M1I35M4D11M1D117M1D112M1D17M1I1M2D7M1D13M1I63M3I1M1I7M1I261M2D186M2D9M1D74M1I20M1D52M1I159M1D6M1D28M1D19M1D53M1I8M1I63M1D45M1D61M1D158M1D23M1I8M1D4M2D10M3D160M2D98M1D203M5I2M3I225M1D121M1D149M1I180M1I25M1I7M2I23M1D21M2D108M1I88M3D122M1D8M1I28M1D12M3D6M1D8M3D12M1I13M2D20M1I10M1D9M1D5M1D2M1D2M2I18M1I5M2D4M2I10M2D9M1D6M4D6M2D5M2D4M1D30M1I17M2D8M1I7M1I2M1D8M1D17M1I15M2D7M1D6M1D3M1D4M1D3M1I4M1I120M1D7M1D18M1D42M3D74M1I5M1I26M2I11M2D35M1D13M1D6M1D3M3I1M1D220M8D61M1I24M1D134M4I40M3D60M1I98M1D25M1D56M1D166M1D56M1D50M1I9M1D42M1D2M1D157M2D6M1I26M1D31M2I19M2I16M1I91M1D57M3I30M1D102M1I21M1D2M1D5M1I33M1D4M2I8M1D7M1D20M1D31M1D31M1D51M1D30M1I41M2D39M1D56M1D12M3D60M1D38M1I18M2I62M1D11M2I49M1D29M1D40M2I136M1D14M2D10M2D79M1I48M3D47M1D116M3D25M2D3M1D24M2D50M2D87M2I149M2D3M1D4M1D10M1I3M1D53M2D15M1D24M1D86M3D127M1D28M1D24M2D19M2D120M2I16M1D23M1D5M1D4M1I3M1D14M2D7M1D18M1D31M2I2M1D170M1I76M1D8M2D19M4I1M1I14M1I29M3D1M1D70M1D10M1D14M5D7M1D4M2I5M1D19M3I12M1D37M3I11M1D4M1D19M1D36M3D23M2I41M1D57M1D41M1D20M1I58M3I15M1D14M1D17M1D6M2I3M1I25M3D8M2I12M1D11M1I18M4I6M4D16M1D10M2D7M1D10M1D18M1D32M4D9M2I16M1I10M1I23M1I23M1I25M1D4M2D4M2D24M3D6M2D16M1D69M1D97M1D45M1D9M1D8M1D25M1D39M1D16M1I6M2D85M1D40M2I81M1D48M4D6M1I18M2I122M1D11M1I38M1D14M2D49M2D1M1D20M1I65M1D7M2D8M1D23M2D49M2D9M1I64M1I29M2I19M1D9M2I13M1D175M2D4M2I42M3D7M7D6M1D38M1D69M1I27M1D3M1D5M1I9M1D163M1I21M1I6M1I1M2D59M2I20M4D3M3D1M2I89M1I43M2D35M1D7M3D21M1D24M1D2M2D11M2I5M3D3M2I37M1D5M1D31M1I24M2I3M1I55M1D37M1D30M1I16M2D3M1D37M1I17M1D3M1D15M1I2M1D5M2I12M1I60M1D11M3I2M1I2M4D27M2D43M1D19M1D13M3D10M1I23M1I16M2I33M2D2M2D17M2I19M1I8M2D35M1D14M1D41M1I48M1D84M1I31M1I49M1I21M1D37M1D17M1I1M1I62M3I4M37I22M1D74M1D136M1D82M2I35M1D7M1I3M1I63M1I56M1D26M1I4M1D4M1D7M1D20M1I10M1D5M1D1M1D47M1D44M2I15M1I63M2D16M1D28M1I25M1D27M2D50M2I93M1D55M1D5M3D17M2D66M1D6M1D17M2I13M1D13M1I4M1I38M1D10M1D9M2D47M1D32M1D6M1D5M2I32M1D77M2I5M2D16M1D4M1D22M2D7M1I32M1D1M1D4M2D18M1I42M1D7M1D3M1D25M1D7M1D16M1I29M1I2M1D28M1D17M1I10M1I7M1D11M1D3M3D28M1I15M1D24M1D41M1D7M1I28M3I5M1D9M1D36M1D85M4D62M2D7M1I5M1I42M1D28M2I6M1D3M1D36M1I61M1I16M9I82M1D112M1I3M1D8M1I84M1I2M1I2M6D152M2I14M2D4M1D21M1I102M1D1M2I25M1I36M1I98M1I16M1I26M1I5M1I44M1D14M2I28M1D56M1D127M3I1M2D13M1D86M1I12M1D41M1I63M2I98M1I86M1D15M2D19M1I3M1D13M2D35M2D87M1I10M1I28M1I24M1D2M1D1M3I36M1D27M1D21M2D35M1D63M1D14M1D3M2I11M2D8M2D79M1D8M1D17M1D7M1D13M1D3M1I3M1D18M3D37M1I4M1D4M1D7M1D1M1D15M2D13M1D76M1D1M1D31M3D6M1I13M1I4M1I30M1D93M1D2M1D5M1I39M3D7M1D20M1I92M2I1M1D7M2I28M1D47M1I53M1D6M2D12M1D3M1D52M1I6M1I29M1I47M1D10M2I8M1D36M1I14M1D3M1D152M4I17M2D21M1D6M2I4M2I7M1I21M3D21M1D7M1I46M1D87M2I40M1I14M3I12M2D4M1D6M1I12M1I6M1D7M2D16M1D22M1D5M1I7M1D16M2D18M2D34M6D26M1I58M1D12M5D11M1I51M2I10M1I13M2I10M1I31M1I63M1I51M1D42M1I50M2I2M1D61M1D23M1I4M1I2M1D7M4I4M3D14M2I67M1I2M2D9M1D35M1D3M1I3M1D96M1D126M1I89M1I18M1D195M1I8M1D19M1I47M1I22M2D12M2D24M1D35M1D17M1I31M1I96M1D6M1D34M1D26M1D39M2D27M6D45M1I2M1D22M1I36M3D77M2D22M1D30M1D61M1D38M1D15M1D27M1D22M1I17M2D23M1D80M1I84M3I7M1D39M1D200M1D37M1I198M1D58M1D68M1I23M2D26M5D138M1I58M1D9M1D3M1I52M2I45M2D131M1I39M2I22M1D8M1I37M1I175M1D108M2D4M1I56M1D11M2D214M1I11M2I47M1D197M1I54M2I21M1D57M1I14M2D51M3I12M1D6M3D44M1I62M1D44M1D50M2D3M1I17M1D36M2D32M1I75M1I100M1I34M3I2M1D40M1I54M1D64M1D96M1D155M2I55M1D163M1I5M1I17M1D89M1I16M1I24M2I7M1D126M1D29M1D19M1I9M2D34M2D45M2D71M2D53M1I42M1I25M1D18M1I10M3D76M2D3M2D36M1I19M2D78M2I67M4D5M1D100M2D48M2D32M1I43M1D131M3I12M1D67M1I78M1D8M1I6M1D132M1I73M2I14M1D19M1D20M3D5M1I19M3D16M2I7M2D61M1D6M2I15M2I22M1D6M1D65M5D23M3I42M4D64M2D13M1D17M1D51M3I99M2I77M1D90M1I8M1D71M4D3M2I32M1D73M1D9M1D98M1D14M2D20M2I44M1D1M2I110M1D21M1I43M1I5M1D67M1D11M1I21M1D21M2D9M1I61M1D14M1D39M1D45M2I40M1D43M1D204M3I2M1D71M1I63M1D135M1D144M1D49M1D10M1D149M1I74M1I42M3D49M1D26M1D41M1D93M3D57M5I41M1I135M1I16M1D252M2I84M1D34M1I19M1D16M1D76M1D13M2I14M1I2M1I18M1I27M3I1M1I6M1D3M1D15M1D201M1I33M1D8M1I106M1D48M3D87M1D101M1I5M1I58M1D12M1I3M1I6M1D7M4D41M1D5M1I30M4D54M1I27M2D107M1D72M1D67M1D13M2D48M2D33M1D8M1I68M1D28M1D13M1D47M1I4M2I121M1D128M3D80M1D13M1D209M3I1M1I14M2D52M1I7M1D9M1D16M1I9M2D10M2D48M1D82M1D74M2D16M1I46M1I30M1D8M2I19M1D25M1I41M1D21M1I11M1I23M1I18M3D9M1D4M1D22M1I4M1D5M1D8M1I2M2D57M2D41M1I22M1I7M1D22M3I48M2D36M1I78M1D2M1D3M1D65M2D6M2I23M1I3M1I15M1I17M2I34M2I68M1I15M1I21M2D8M1D1M1D118M1D55M3D9M1D14M1I53M1I18M1D12M1D24M1D3M2D16M1D109M1D66M1I20M1D1M2D15M1D9M1D81M3I12M1I242M1D17M1D2M2D37M2I26M2I42M2D30M1I96M1I72M1D11M2I48M1D40M2D26M2D50M1D16M2D135M1D24M1D72M2I63M1I38M1I39M1D47M2I51M1D84M1D94M1D10M1I15M1D32M1D17M1D41M1D20M1D28M1I5M5D8M2D2M1I72M1D37M3D59M1D27M1I27M1D26M1D15M1I2M1D46M5D81M1D91M1I16M1I116M1D136M1D151M1D13M1I98M4D50M2D29M1D88M1I4M1I51M2I39M1I79M1D41M1D2M1D30M1I13M1I35M1D68M4I38M1D62M2D53M1I23M1I140M1D64M2D46M1I39M2D81M1I11M2D69M2D12M1I89M2I108M2D13M2I3M1D1M1D4M1D16M2I27M3D47M1I3M1D62M3D77M1I31M1I89M3D1M1D38M2D39M6D7M2I41M2D1M1D1M1D86M2I188M1D243M11I75M2I3M1D15M1D169M1I40M1D18M1D157M1D14M2D86M1I27M1I160M2D10M2D12M1I78M1I47M1D1M1D41M2I27M1I20M1I14M1I12M4D35M2I15M1I31M1I69M1I3M1I37M1D37M2D8M6D9M1D17M1D55M1I75M2D27M2D19M1I57M2I13M1D17M1D33M1D119M2D14M2I2M2D66M1D3M1D3M1I37M2D54M1I5M1D3M1D15M1D19M1I22M1D24M1I41M1I31M1D62M1D65M2D50M1D16M1D44M1I148M2I24M5D300M1I63M1D4M1D111M1I106M1I3M1D14M1D54M1D40M1D58M2D37M1D30M1I83M1D4M1D13M3I22M1I82M2I16M1I27M1I52M1I26M1I184M2I37M1D59M1D2M1I13M2I9M1D19M2D43M1I32M2I18M1D46M1D5M1D9M2I11M2D17M1D7M6D37M3D97M1D3M1D75M1I53M2D4M2D2M1D15M1I237M1I1M1I22M2I31M1I6M2I176M1D18M1D29M3D3M2D11M1D6M2D24M1I2M1I26M2D4M2D11M1I27M1D15M4D14M2D5M2I16M1D4M1I30M1D21M2D2M2I2M1I131M1D6M2I11M1D45M2D14M1D36M2I25M1D11M1D15M1I2M1I7M1D19M1D50M1D3M2D59M1D19M1I7M1I16M1D102M1I4M1I17M1D55M2D18M1I17M2D29M1D8M2D13M2D52M1D69M2D33M1D37M2I12M1D8M1D20M3I19M1I20M1D41M4D14M1I55M2I16M1D36M1I27M1I25M1D15M1D8M1D32M1I16M1D6M1D8M2D2M2I12M4I5M2D9M1D47M1D9M2D71M1I26M1D46M4D19M1D102M2D7M2D5M1I150M1D2M1D11M2I27M5D64M1D30M2D2M1D95M1D8M1D16M1D2M2I40M2I42M1I10M2I24M1I27M1D75M1I26M1D86M1D4M1I29M1I35M1I34M1I5M1I36M2I9M1D44M1D4M2D8M2D1M1D17M1I3M1D1M2D7M1D24M2D14M2D21M1I4M1I31M1D14M2I44M1I15M1D33M1D79M1I106M1D41M1D31M4D12M1D7M1I15M2D12M1D83M2D8M1D4M1I2M3D6M1I62M2D73M2D30M3D3M2D16M1D1M1D31M4D31M1I42M1D38M3D24M1D3M1I5M2D1M3D25M1D19M1D5M3D16M1I3M1D7M1I9M3I1M1I14M1D6M3D5M1D21M1I25M1I8M1D35M1D5M1I10M1D55M1D2M1D10M1I79M1D5M1I80M2D13M1I11M2D32M1D29M2D24M1D7M1I49M1D1M1D16M2D13M5D32M2I18M1I12M1D52M1I11M1D13M1I85M1D37M1I12M1I7M1D38M8D1M1D3M1D7M1D46M1D4M2I5M1I4M2D15M1D90M1I64M1D4M1D3M1D5M2D7M4I66M1I15M1I35M1D10M1D22M1D12M1I7M1D13M1D44M1D25M1I4M1D3M1D41M1D24M1I9M2I60M1D6M2D130M2I3M1I7M1D3M1D39M2D6M1I4M1D11M1I15M1D44M5D4M2D5M1I25M1I25M1I115M1I30M1D2M2I28M3I17M2I21M1I130M1D5M2D18M2D14M1D5M1I92M1I25M2D9M1I5M1I7M1D2M1D40M4D85M1I5M2I2M4D4M1D60M1I44M1D11M1I108M2I59M1D39M3D8M1D4M3D78M1D6M1I24M1I78M1D2M2I48M1I13M2I12M1D13M3D54M1I14M72I10M3I4M3I17M2D26M1D21M1I6M1D35M1I6M2D21M1D50M1D3M2I10M1D6M1I17M1D51M1D128M1D17M1D23M4D14M1D5M2D64M1D10M1D11M1I90M3D32M2D4M1I21M3D2M2D5M3I14M1D25M1D44M1D6M1I4M1D14M2D28M2D6M1D51M1I3M2D2M1I3M1D12M2D19M1I61M1I13M1I56M2I19M2D12M1D21M1D4M1I39M6D1M1D12M3D1M1D8M1I9M3D18M1D3M2D67M1D14M2D7M1D12M1D23M2D21M1D4M1D15M1D2M2I12M1I14M4D5M2I21M2I31M3D66M1I34M2I6M1D43M1D23M1D14M1I13M1I6M1D32M4D4M1D4M2I1M1D3M1D40M1D10M2I2M2D3M1D60M1I28M2I10M1D9M3I14M3D32M3D35M2I32M1D86M1I7M1D20M1D25M1D96M1D24M1I80M1I10M4D30M2D12M1D9M1I98M3D81M2I18M1D22M2D51M1I69M2D37M1I26M1D34M4D44M1I7M2I22M2I12M1D10M1D4M1D11M1D103M1I20M1I20M1D16M1D8M1D98M2D50M16D12M2D9M1D63M1D11M1D76M2I33M5D25M3D63M1D18M1I7M2D31M1D21M1D7M2D15M1I82M1D59M1I66M1I13M1D73M1D8M1D13M2D11M1I7M1I65M1I31M1D8M1D18M1I72M1I31M1I191M1D80M1D9M1D65M13D81M1I90M1D5M1I4M1D2M1D6M1D49M1D15M1D31M1D5M1I33M2I74M1D10M3I46M3I22M2D59M1D5M1D84M2I2M1D13M1I18M2D5M2I3M2I2M20I117M1I13M5D41M2D1M1D16M1S", "length": 117851, "mapping_quality": 60, "sequence_first100": "TTACTTCGTTCAGTTACGTATTGCTTTGAAGGCTGGGCTACCTCAGTGTGGAAGGTGGGCAGTTCCACAGGAATGGTGCCAGGGCCAGAGGGCAATACCA", "tags": [["NM", 7874], ["ms", 151392], ["AS", 152519], ["nn", 725], ["tp", "P"], ["cm", 8416], ["s1", 58736], ["s2", 625], ["dv", 0.041600000113248825], ["SA", "chr12,16970,+,21S18762M358D99068S,52,2368;chr3,76029363,+,18278S3125M142I96306S,1,555;chr19,75654,+,21822S1087M10D94942S,0,83;"]]}

| 구간bp | 겹치는read | 평균coverage | 중앙coverage | 최소/최대 | 평균read길이 |
| --- | --- | --- | --- | --- | --- |
| 50000 | 381 | 102.974 | 103.000 | 93/122 | 27879.129 |
| 100000 | 682 | 107.131 | 106.000 | 93/126 | 24254.419 |
| 200000 | 1305 | 107.528 | 107.000 | 84/129 | 20632.951 |

분석 구간 확대와 coverage 확대는 별개다. coverage조건은동일200kb에서 samtools view -s42.5(seed42,fraction0.5)로 read를 downsample했다. 실제 coverage는 각 실행후 다시 기록했다. 긴read는여러구간에겹칠수있어 read개수가단순구간길이에비례하지않는다.

모델은 BAM자체를tensor로보지않는다. pileup후보의33위치×채널특징과 full-alignment후보의read-depth×33위치×채널 특징으로바뀐다. 실제shape와 첫후보tensor/예측을 저장했다. 최종출력은좌표·REF/ALT·genotype·quality를 가진VCF다.

#### 분석 입력별 GPU 사용률·메모리

| 조건 | bp | coverage | read | 후보tensor처리수 | 전체초 | bp/초 | GPU평균/최대% | 시작GiB | worker peakGiB | 증가GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| region_50000_full | 50000 | 102.974 | 381 | 1486 | 16.475 | 3034.986 | 7.290/18.000 | 0.011 | 0.148 | 0.137 |
| region_100000_full | 100000 | 107.131 | 682 | 3134 | 16.695 | 5989.694 | 7.188/20.000 | 0.011 | 0.347 | 0.336 |
| region_200000_full | 200000 | 107.528 | 1305 | 6611 | 17.932 | 11153.156 | 11.265/88.000 | 0.011 | 0.684 | 0.673 |
| region_200000_half | 200000 | 50.804 | 629 | 9308 | 17.233 | 11605.571 | 12.091/58.000 | 0.011 | 0.763 | 0.752 |

- region_50000_full 실제batch입력: `['[102, 33, 18]', '[103, 33, 18]', '[105, 33, 18]', '[108, 33, 18]', '[126, 33, 18]', '[181, 89, 33, 8]', '[89, 33, 18]', '[90, 33, 18]', '[92, 33, 18]', '[93, 33, 18]', '[97, 33, 18]']`; GPU forward+전송/CPU출력 합 0.702초.
- region_100000_full 실제batch입력: `['[178, 33, 18]', '[183, 33, 18]', '[187, 33, 18]', '[192, 33, 18]', '[208, 33, 18]', '[210, 33, 18]', '[215, 33, 18]', '[216, 33, 18]', '[218, 33, 18]', '[221, 33, 18]', '[232, 33, 18]', '[238, 33, 18]', '[444, 89, 33, 8]']`; GPU forward+전송/CPU출력 합 0.788초.
- region_200000_full 실제batch입력: `['[376, 33, 18]', '[384, 33, 18]', '[388, 33, 18]', '[392, 33, 18]', '[398, 33, 18]', '[399, 33, 18]', '[403, 33, 18]', '[414, 33, 18]', '[426, 33, 18]', '[468, 33, 18]', '[528, 33, 18]', '[554, 33, 18]', '[587, 33, 18]', '[894, 89, 33, 8]']`; GPU forward+전송/CPU출력 합 1.196초.
- region_200000_half 실제batch입력: `['[1000, 89, 33, 8]', '[141, 89, 33, 8]', '[453, 33, 18]', '[471, 33, 18]', '[493, 33, 18]', '[545, 33, 18]', '[579, 33, 18]', '[585, 33, 18]', '[595, 33, 18]', '[603, 33, 18]', '[667, 33, 18]', '[670, 33, 18]', '[775, 33, 18]', '[813, 33, 18]', '[918, 33, 18]']`; GPU forward+전송/CPU출력 합 1.552초.

전체시간은worker시작·모델적재·CPU후보생성·I/O·GPU추론·whatshap·후처리·VCF저장을포함하는end-to-end다. GIAB평가·다운샘플·입력감사·정규화는제외했다. 독립warmup/compile시간은N/A이며worker시작비용을전체에포함한다. 순수forward시간은별도기록하고CUDA synchronize했다. 표의 처리량은bp/초이며 후보수분포/coverage가다르면엄밀한비교에한계가있다.

GPU평균/최대는전체pipeline구간의활동시간비율이다. CPU전처리·phasing·I/O기간이포함돼평균이낮을수있다. GPU커널구간만의FLOPS나memory비율이아니다. 전체RAM121.688GiB를90%채우는것은목표가아니다. RSS와CUDA를더하지않는다.

CUDA peak는여러subprocess 중 **가장큰단일worker allocator peak**이며프로세스합계/장치전체점유율이아니다. 각worker모델적재후peak를reset했다. baseline과delta는동일peakworker로맞춰기록했다. 시계열CSV allocated/reserved는감시parent값이므로모델메모리로쓰지않는다. 실제worker메모리는각조건instrumentation/worker_*.jsonl에있다. RSS는parent시계열과worker스냅샷으로구분한다.

#### 결과 검증

호출VCF와truth를bcftools norm(-f참조,-m-any)으로left-normalize/split한뒤,해당분석구간및GIABconfident BED 안의PASS실제ALTallele exact matching을계산했다. genotype포함haplotype-aware hap.py/vcfeval평가가아니며복잡한동등표현변이는FP/FN으로나뉠수있다. 공식quick demo의hap.py수치와직접비교하지않는다.

| 조건 | 종류 | TP | FP | FN | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| region_50000_full | SNP | 57 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| region_50000_full | INDEL | 12 | 0 | 3 | 1.000 | 0.800 | 0.889 |
| region_100000_full | SNP | 208 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| region_100000_full | INDEL | 28 | 0 | 5 | 1.000 | 0.848 | 0.918 |
| region_200000_full | SNP | 404 | 1 | 0 | 0.998 | 1.000 | 0.999 |
| region_200000_full | INDEL | 46 | 4 | 15 | 0.920 | 0.754 | 0.829 |
| region_200000_half | SNP | 404 | 1 | 0 | 0.998 | 1.000 | 0.999 |
| region_200000_half | INDEL | 41 | 1 | 20 | 0.976 | 0.672 | 0.796 |

원본VCF·normalizedVCF·metrics.json·명령·exit code·예측tensor·실제입력예시를각조건폴더에저장했다. 독립프로세스checkpoint reload: [{"model": "Clair3_F", "actual_input_shape": [32, 89, 33, 8], "independent_process": true, "max_abs_difference": 0.0, "passed": true, "original_batch_shape": [181, 89, 33, 8], "note": "first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4"}, {"model": "Clair3_P", "actual_input_shape": [32, 33, 18], "independent_process": true, "max_abs_difference": 0.0, "passed": true, "original_batch_shape": [103, 33, 18], "note": "first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4"}]

공식Train은PyTorch학습을지원하지만이번Clair3과제범위는long-read germline호출·자원측정이다. 파인튜닝은수행하지않았으며학습완료라고쓰지않는다. GIAB단일샘플의작은구간실험이므로새샘플일반화나WGS전체성능은미확인이다.

설치시curl개발라이브러리충돌을전용이미지내에서정리했고,첫호출전처리의pigz누락을보완했다. 실패run과로그를보존했다.

#### 공식 출처

- [Clair3 공식소스·설치·모델](https://github.com/HKU-BAL/Clair3)
- [공식 ONT quick demo](https://github.com/HKU-BAL/Clair3/blob/main/docs/quick_demo/ont_quick_demo.md)
- [공식 PyTorch ONT 가중치](https://www.bio8.cs.hku.hk/clair3/clair3_models_pytorch/ont/)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### Clair3 실제 후보 tensor의 배치 확대 후속 실험


상태 **complete** · candidate_batch_20261028_185641_089149

기존HG003 ONT BAM/GRCh38 chr20:100000~300000/threads4/한GPUworker의공식variant calling을다시실행해실제후보feature tensor 전체를cache했다. 모델입력이나호출결과를바꾸지않는opt-in capture hook을사용했다. CLI옵션이없더라도모델의실제forward 입력을분리해batch를조절할수있음을확인한실험이다. 후보생성과VCF호출은별도단계이며후속시간은CPUcache→GPU nativeforward→CPU확률반환만포함한다.

pileup과full-alignment는다른입력형태/모델로따로측정한다. 같은feature값은중복제거했고가짜zero row/반복복제row를batch에채우지않았다. 원본candidate좌표는capturehook으로내보내지못했으므로고유생물학적위치수대신고유feature row수를보고한다. cache파일SHA256·원본shape/dtype·선택pool은input_audit.json에있다.

|모델|고유feature pool|각후보shape|dtype|
|---|---|---|---|
|Clair3_P|5708|[33, 18]|int8|
|Clair3_F|894|[89, 33, 8]|int8|

|모델|batch|반복|고유후보|총처리|초|후보/초|모델적재GiB|allocated peak GiB|추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Clair3_P|1|18096|1|18096|15.0004|1206.3659|0.0077|0.0573|0.0496|0.0898|0.0471|84.4828|89.0|
|Clair3_P|8|16249|8|129992|15.0008|8665.6579|0.0077|0.063|0.0553|0.0977|0.0518|86.1724|90.0|
|Clair3_P|32|13518|32|432576|15.0008|28836.7784|0.0077|0.0803|0.0725|0.127|0.066|87.3103|91.0|
|Clair3_P|128|1345|128|172160|15.0071|11471.893|0.0077|0.1475|0.1398|0.2773|0.1212|89.1429|95.0|
|Clair3_P|512|293|512|150016|15.0105|9994.0699|0.0077|0.4207|0.413|0.8945|0.3457|92.3571|96.0|
|Clair3_P|1000|147|1000|147000|15.1032|9733.0169|0.0077|0.7687|0.761|1.2422|0.6317|91.2963|96.0|
|Clair3_P|2048|72|2048|147456|15.0313|9809.9265|0.0077|1.5113|1.5036|2.4746|1.242|89.2963|96.0|
|Clair3_P|4096|36|4096|147456|15.2014|9700.1604|0.0077|2.9666|2.9589|4.8828|2.4379|92.1429|96.0|
|Clair3_P|5708|25|5708|142700|15.1046|9447.4542|0.0077|4.1114|4.1037|6.7832|3.3787|88.4815|96.0|
|Clair3_F|1|7026|1|7026|15.002|468.3375|0.0111|0.1784|0.1673|0.2969|0.1466|89.8571|93.0|
|Clair3_F|8|9756|8|78048|15.0008|5202.9243|0.0111|0.0476|0.0365|0.0566|0.0391|87.0|93.0|
|Clair3_F|32|3205|32|102560|15.0042|6835.4007|0.0111|0.1232|0.1121|0.1484|0.1013|91.75|95.0|
|Clair3_F|128|823|128|105344|15.0178|7014.5928|0.0111|0.3766|0.3655|0.5156|0.3095|91.8929|96.0|
|Clair3_F|512|196|512|100352|15.0386|6672.9585|0.0111|0.972|0.9609|1.4902|0.7988|88.1071|96.0|
|Clair3_F|894|116|894|103704|15.0361|6896.9851|0.0111|1.4991|1.488|2.4219|1.2319|91.6786|96.0|

FP32/TF32=False/seed42, 기존공식ont가중치를고정했다. 각batch는새프로세스로실행한다. 전체파라미터/적재기준값/실제입력dtype·shape/출력shape/표본메모리는각worker summary에있다.
원래공식pipeline에서기록한동일후보의출력확률과batch1재추론대조도통과했다(P최대오차4.17e-6,F1.06e-5). native_pipeline_candidate_equivalence.json에저장했다.
전체출력finite·저장재로드정확일치·같은첫후보의batch간확률일치(atol/rtol1e-4)를검증했다. 실제BAM호출VCF도별도보존했다. 이단계는새로운variant accuracy평가나학습이아니며기존truth정확도결과를참조한다.

종료이유: {"Clair3_P": "available_real_candidate_cache_limit", "Clair3_F": "available_real_candidate_cache_limit"}
확보한실제후보pool에서측정한최대batch이며모델절대한계가아니다. GPU사용률은메모리비율과다르다. CUDAallocated/reserved는프로세스allocator,분모는CPU/GPU공유RAM121.688GiB다. RSS와CUDA를더하지않는다. 가용16GiB·swap+2GiB guard,0.5초대기+조회표본간격. RSS/가용량sampled값과CUDApeak를구분한다.

[공식 Clair3](https://github.com/HKU-BAL/Clair3) · [기본 결과](Clair3.md)

원본실행: results/candidate_batch_20261028_185641_089149; 명령·소스·가중치출처·inputcache·출력·자원시계열·exit·검증값을보존한다.

### 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.


## 6. 실제 실행 코드·명령어 전문

<!-- EXECUTED_SCRIPTS_START -->

### 원문과 출처 검증

아래 코드 블록은 원본 내용을 생략하거나 재작성하지 않았다. 각 파일의 원래 경로·원문 SHA256·출처 상태를 함께 기록했다. `실행 run 저장 snapshot`을 우선하고, 현재 소스만 남은 경우에는 당시 파일과 같다고 단정하지 않는다. 동일 바이트 버전은 한 번만 수록하고 해당 경로들을 함께 표시한다.

GitHub에는 `.py`/`.sh`를 별도 파일로 두지 않는다. 코드 블록을 원래 경로에 저장하거나 [복원 안내](Clair3.md)의 코드로 로컬에서 복원한다. 라이브러리 전체 소스는 이 사용자 실행 스크립트와 구분하며 패키지 버전/출처를 따른다.

### 코드 목록

|원래 파일|원문 SHA256|출처 상태|
|---|---|---|
|`clair3/Dockerfile`|`042a78fd10943f77204509c11fc0da187828bed569ef522383d9f045d363152e`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/audit.py`|`60de3f0256af3c437887d44ec911f949547b5d311f6cc525095d23361b9ca4a0`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/batch_followup/capture.py`|`2d5c7a4f07920135dd918e1602af7e1c00f448aa2504961479429874d7eee39e`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/candidate_batch_20261028_185641_089149/source_capture.py`|`2d5c7a4f07920135dd918e1602af7e1c00f448aa2504961479429874d7eee39e`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/scripts/batch_followup/prepare_cache.py`|`6bf8fe7d725872c0e6220d4b3a66beb60ecd2a745e4290bdda49547c33452a2f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/candidate_batch_20261028_185641_089149/source_prepare_cache.py`|`6bf8fe7d725872c0e6220d4b3a66beb60ecd2a745e4290bdda49547c33452a2f`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/scripts/batch_followup/worker.py`|`e860b7101a462406f629583836ca9ae616b7fd98101aa92c9fd27fd8d06b832c`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/candidate_batch_20261028_185641_089149/source_worker.py`|`e860b7101a462406f629583836ca9ae616b7fd98101aa92c9fd27fd8d06b832c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/scripts/benchmark.py`|`4e1312610a86ee145ac0707ba0a3cb26d0e355a492de7851ea4525cef140439f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/discover_download.py`|`c80f3c16ef6b26b3629d404a42669612e8fa76b72e9e2a563709914f276ab47d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/instrument.py`|`9130c01daabcb4faca2e3d4437182ab38fa3449ab71d1baf07946acd875aa84c`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/candidate_batch_20261028_185641_089149/source_instrument.py`|`9130c01daabcb4faca2e3d4437182ab38fa3449ab71d1baf07946acd875aa84c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/scripts/launch.py`|`032dc38ce8ceda6481c9938e762b6cfa50d33fac9c04b53e3e389d4773296e74`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/reload.py`|`1311d548e5698ddb3f660e2c7ed232adc657d8908750554b063598da1d4c60b9`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/report.py`|`1d088fa9f4579dafd0ff4edae1d86c28658db578ae4248497a691388092e6763`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/setup.py`|`deb5e12b435a19d7e8c8083fb70136fcece1a56cdba24734202356bd70c4b10d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/CallVariantsFromCffiGPU.original.py`|`5c55410ab45d1b1ea012ac204b9fd374c46fb2bbbab1dcd0c49353c4a40a3e72`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/results/CallVariantsFromCffi.original.py`|`024fe816b279a416d84ab487b6948ad2899088fcca607bb431264e0e89a96ace`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|

<details>
<summary>clair3/Dockerfile — 원문 전문 (280 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/Dockerfile", "sha256": "042a78fd10943f77204509c11fc0da187828bed569ef522383d9f045d363152e", "encoding": "utf-8", "added_final_newline": false} -->
```dockerfile
FROM nvcr.io/nvidia/pytorch:26.06-py3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends samtools bcftools tabix parallel pigz libhts-dev libcurl4-gnutls-dev libssl-dev liblzma-dev libbz2-dev zlib1g-dev libdeflate-dev && apt-get clean
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/audit.py — 원문 전문 (1,716 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/audit.py", "sha256": "60de3f0256af3c437887d44ec911f949547b5d311f6cc525095d23361b9ca4a0", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,pysam,numpy as np
R=Path('/work');bam=R/'data/HG003_chr20_demo.bam';fa=R/'data/GRCh38_no_alt_chr20.fa';b=pysam.AlignmentFile(bam,'rb');f=pysam.FastaFile(fa);assert b.has_index() and f.get_reference_length('chr20')==b.get_reference_length('chr20');rows=[]
for size in [50000,100000,200000]:
 start=100000;end=start+size;(R/f'data/region_{size}.bed').write_text(f'chr20\t{start}\t{end}\n');depth=np.zeros(size,dtype=np.int32)
 for col in b.pileup('chr20',start,end,truncate=True,stepper='samtools',min_base_quality=0,max_depth=10000):depth[col.reference_pos-start]=col.nsegments
 reads=list(b.fetch('chr20',start,end));rows.append({'region_0based_halfopen':['chr20',start,end],'length_bp':size,'reads_overlapping':len(reads),'coverage_mean':float(depth.mean()),'coverage_median':float(np.median(depth)),'coverage_min':int(depth.min()),'coverage_max':int(depth.max()),'read_length_mean':float(np.mean([r.query_length for r in reads]))})
r=next(b.fetch('chr20',100000,300000));out={'bam_header':b.header.to_dict(),'sort_order':b.header.to_dict().get('HD',{}).get('SO'),'reference_length':f.get_reference_length('chr20'),'regions':rows,'first_read':{'name':r.query_name,'reference_start_0based':r.reference_start,'cigar':r.cigarstring,'length':r.query_length,'mapping_quality':r.mapping_quality,'sequence_first100':r.query_sequence[:100],'tags':r.get_tags()},'source':'official Clair3 ONT quick demo HG003 Guppy3.6 ~85x; ONT non-dwell model ont','truth':'GIAB HG003 GRCh38 chr20 v4.2.1 VCF and confident BED'};(R/'data/input_audit.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({k:v for k,v in out.items() if k not in ['bam_header','first_read']},indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/batch_followup/capture.py — 원문 전문 (996 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/batch_followup/capture.py", "sha256": "2d5c7a4f07920135dd918e1602af7e1c00f448aa2504961479429874d7eee39e", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,subprocess,json,time
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];env=os.environ.copy();env['CLAIR3_METRICS_DIR']=str(D/'instrumentation');env['CLAIR3_FULL_CANDIDATE_CACHE']=str(D/'candidate_cache');env['PYTHONPATH']='/work/scripts:/work/source';env['PATH']='/work/.venv/bin:'+env['PATH'];cmd=['/work/.venv/bin/python','/work/source/run_clair3.py','--bam_fn=/work/data/HG003_chr20_demo.bam','--ref_fn=/work/data/GRCh38_no_alt_chr20.fa','--threads=4','--platform=ont','--model_path=/work/models/ont','--output='+str(D/'calls'),'--bed_fn=/work/data/region_200000.bed','--python=/work/.venv/bin/python','--pypy=/work/.venv/bin/python','--whatshap=/work/.venv/bin/whatshap','--device=cuda:0','--use_gpu'];(D/'capture_command.json').write_text(json.dumps(cmd,indent=2));t=time.time();p=subprocess.run(cmd,env=env,cwd='/work/source');(D/'capture_exit.json').write_text(json.dumps({'exit_code':p.returncode,'wall_seconds':time.time()-t}));assert p.returncode==0
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "clair3/results/candidate_batch_20261028_185641_089149/source_capture.py", "source": "clair3/scripts/batch_followup/capture.py", "sha256": "2d5c7a4f07920135dd918e1602af7e1c00f448aa2504961479429874d7eee39e"} -->

</details>

<details>
<summary>clair3/scripts/batch_followup/prepare_cache.py — 원문 전문 (1,271 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/batch_followup/prepare_cache.py", "sha256": "6bf8fe7d725872c0e6220d4b3a66beb60ecd2a745e4290bdda49547c33452a2f", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,hashlib
import numpy as np
D=Path('/work/results')/os.environ['RUN_ID'];audit={}
for kind in ['Clair3_P','Clair3_F']:
 files=sorted((D/'candidate_cache').glob(kind+'_*.npz'));assert files;rows=[];sources=[];seen=set()
 for file in files:
  v=np.load(file);x=v['X'];sources.append({'file':file.name,'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.file_digest(file.open('rb'),'sha256').hexdigest()})
  for r in x:
   digest=hashlib.sha256(r.tobytes()).hexdigest()
   if digest not in seen:rows.append(r);seen.add(digest)
 a=np.stack(rows);np.save(D/(kind+'_inputs.npy'),a);audit[kind]={'actual_feature_tensor_rows_before_deduplication':sum(s['shape'][0] for s in sources),'unique_feature_rows':len(a),'feature_shape':list(a.shape[1:]),'dtype':str(a.dtype),'source_caches':sources,'identity_limitation':'deduplicated exact feature tensors; genomic candidate coordinates not exported by this capture hook, no invented position IDs','source':'real HG003 ONT BAM, GRCh38 chr20 100000..300000, existing official Clair3 extraction; no zero-filled or synthetic performance inputs'}
(D/'input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v['unique_feature_rows'] for k,v in audit.items()}),flush=True)
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "clair3/results/candidate_batch_20261028_185641_089149/source_prepare_cache.py", "source": "clair3/scripts/batch_followup/prepare_cache.py", "sha256": "6bf8fe7d725872c0e6220d4b3a66beb60ecd2a745e4290bdda49547c33452a2f"} -->

</details>

<details>
<summary>clair3/scripts/batch_followup/worker.py — 원문 전문 (2,680 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/batch_followup/worker.py", "sha256": "e860b7101a462406f629583836ca9ae616b7fd98101aa92c9fd27fd8d06b832c", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,time,json,gc
import numpy as np,torch
sys.path[:0]=['/support','/work/source'];from gpu_support import Meter,write_json
from clair3.model import Clair3_P,Clair3_F
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];kind=os.environ['CANDIDATE_MODEL'];B=int(os.environ['BATCH']);C=D/f'{kind}_batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'batch':B,'task':'actual candidate tensor native model inference, excludes BAM preprocessing/VCF calling'}
try:
 torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='loading';t=time.time();model=(Clair3_P if kind=='Clair3_P' else Clair3_F)(predict=True,add_indel_length=kind=='Clair3_F').float().cuda().eval();p=R/'models/ont'/('pileup.pt' if kind=='Clair3_P' else 'full_alignment.pt');ck=torch.load(p,map_location='cpu',weights_only=False);model.load_state_dict(ck.get('state_dict',ck));del ck;torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']=sum(p.numel() for p in model.parameters());S['parameters_trained_in_run']=0
 x=np.load(D/(kind+'_inputs.npy'),mmap_mode='r');assert len(x)>=B;values=np.array(x[:B]);S['input_shape']=list(values.shape);S['input_dtype']=str(values.dtype);S['unique_candidates']=B
 def forward():
  with torch.inference_mode():return model(torch.from_numpy(values).cuda()).cpu().numpy()
 meter.phase='warmup';t=time.time();out=forward();torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del out;gc.collect();t,base=meter.begin('candidate_inference');repeats=0
 while repeats<2 or time.time()-t<15:out=forward();assert np.isfinite(out).all();torch.cuda.synchronize();repeats+=1
 r=meter.end(t,base,batch_size=B,unique_candidates=B,repeats=repeats,processed_candidates=B*repeats);r['candidates_per_second']=B*repeats/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];S['condition']=r;np.save(C/'predictions.npy',out);assert np.array_equal(np.load(C/'predictions.npy'),out);S['output_shape']=list(out.shape);ref=D/(kind+'_reference_first.npy')
 if ref.exists():S['batch_invariance_max_abs']=float(np.abs(np.load(ref)-out[0]).max());assert np.allclose(np.load(ref),out[0],atol=1e-4,rtol=1e-4)
 else:np.save(ref,out[0])
 S['finite_and_exact_save_reload']=True;S['status']='complete';write_json(C/'summary.json',S);print('BATCH_COMPLETE',kind,B,r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write_json(C/'summary.json',S);raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "clair3/results/candidate_batch_20261028_185641_089149/source_worker.py", "source": "clair3/scripts/batch_followup/worker.py", "sha256": "e860b7101a462406f629583836ca9ae616b7fd98101aa92c9fd27fd8d06b832c"} -->

</details>

<details>
<summary>clair3/scripts/benchmark.py — 원문 전문 (6,709 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/benchmark.py", "sha256": "4e1312610a86ee145ac0707ba0a3cb26d0e355a492de7851ea4525cef140439f", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,os,json,time,subprocess,hashlib,collections
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,pysam,torch
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];meter=Meter(D);S={'status':'running','conditions':[],'settings':{'version':'2.0.4','commit':'9d1dd3f2475072b6255900e406c34c6b0545fc88','threads':4,'gpu_predictor_workers':1,'batch_policy':'official predictBatchSize200*5 for GPU, actual batches logged','platform':'ont','checkpoint':'models/ont/{pileup,full_alignment}.pt','precision':'FP32 weights, original integer input features; model converts/scales input','phasing':'default whatshap intermediate phasing','preprocessing':'native ARM64 CFFI; CPython3.12 replaces optional PyPy accelerator','warmup':'N/A, end-to-end includes each worker load and startup','training':'supported by official Train; this run performs variant calling only'},'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}}
def save():write_json(D/'summary.json',S)
def must(cmd,**kw):
 p=subprocess.run(cmd,**kw);assert p.returncode==0,(cmd,p.returncode)
try:
 # Actual input and truth files were audited before GPU calling.
 S['input_audit']=json.loads((R/'data/input_audit.json').read_text());save();half=R/'data/HG003_half.bam'
 if not half.exists():
  must(['samtools','view','-s','42.5','-b','-o',str(half),str(R/'data/HG003_chr20_demo.bam')]);must(['samtools','index',str(half)])
 normtruth=D/'truth_normalized.vcf.gz';must(['bcftools','norm','-f',str(R/'data/GRCh38_no_alt_chr20.fa'),'-m','-any','-Oz','-o',str(normtruth),str(R/'data/HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz')]);must(['bcftools','index','-t',str(normtruth)])
 for length,bamname in [(50000,'HG003_chr20_demo.bam'),(100000,'HG003_chr20_demo.bam'),(200000,'HG003_chr20_demo.bam'),(200000,'HG003_half.bam')]:
  name=f'region_{length}_'+('half' if 'half' in bamname else 'full');dest=D/name;dest.mkdir();env=os.environ.copy();env['CLAIR3_METRICS_DIR']=str(dest/'instrumentation');env['PYTHONPATH']='/work/scripts:/work/source';env['PATH']='/work/.venv/bin:'+env['PATH'];cmd=['/work/.venv/bin/python','/work/source/run_clair3.py','--bam_fn='+str(R/'data'/bamname),'--ref_fn=/work/data/GRCh38_no_alt_chr20.fa','--threads=4','--platform=ont','--model_path=/work/models/ont','--output='+str(dest/'calls'),'--bed_fn='+str(R/f'data/region_{length}.bed'),'--python=/work/.venv/bin/python','--pypy=/work/.venv/bin/python','--whatshap=/work/.venv/bin/whatshap','--device=cuda:0','--use_gpu'];write_json(dest/'command.json',cmd);t,base=meter.begin(name)
  with (dest/'run.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd='/work/source')
  end=time.time();write_json(dest/'exit_code.json',{'exit_code':code});assert code==0,(name,code);workers=[]
  for p in (dest/'instrumentation').glob('worker_*.jsonl'):workers.extend(json.loads(line) for line in p.read_text().splitlines())
  loads=[r for r in workers if r['event']=='model_only_postload'];forwards=[r for r in workers if r['event']=='forward'];assert loads and forwards and all(r['memory']['allocated_gib']>0 for r in loads)
  if 'model_only_postload' not in S:S['model_only_postload']={**loads[0]['memory'],'system_available_gib':loads[0]['memory']['available_gib'],'system_total_gib':meter.memory()['system_total_gib']};S['parameters']={r['model']:r['total_parameters'] for r in loads}
  row=meter.end(t,base,name=name,mode='variant_calling',bp=length,unique_inputs=1,repeats=1,processed=length,bam=bamname,worker_loads=loads,actual_input_shapes=sorted({str(r['shape']) for r in forwards}),forward_calls=len(forwards),forward_seconds=sum(r['seconds'] for r in forwards),candidate_tensors=sum(r['shape'][0] for r in forwards));row['seconds']=end-t;row['throughput']=length/row['seconds'];row['memory']=dict(peak_allocated_gib=max(r['memory']['peak_allocated_gib'] for r in forwards),peak_reserved_gib=max(r['memory']['peak_reserved_gib'] for r in forwards));row['baseline']={'allocated_gib':max(r['memory']['allocated_gib'] for r in loads)};row['peak_delta_gib']=row['memory']['peak_allocated_gib']-row['baseline']['allocated_gib'];row['memory_scope']='maximum per-worker allocator peak, not sum of processes; multiple models run serially';S['conditions'].append(row);save();print('CONDITION',name,'seconds',row['seconds'],'peak',row['memory']['peak_allocated_gib'],flush=True)
  calls=dest/'calls/merge_output.vcf.gz';assert calls.exists();norm=dest/'calls_normalized.vcf.gz';must(['bcftools','norm','-f',str(R/'data/GRCh38_no_alt_chr20.fa'),'-m','-any','-Oz','-o',str(norm),str(calls)]);must(['bcftools','index','-t',str(norm)])
  # Genomic region vs coverage experiments remain separate; audit actual depth.
  b=pysam.AlignmentFile(str(R/'data'/bamname));depth=np.zeros(length,dtype=np.int32)
  for col in b.pileup('chr20',100000,100000+length,truncate=True,stepper='samtools',min_base_quality=0,max_depth=10000):depth[col.reference_pos-100000]=col.nsegments
  row['coverage_mean']=float(depth.mean());row['reads_overlapping']=b.count('chr20',100000,100000+length);b.close()
  beds=[]
  for line in (R/'data/HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed').read_text().splitlines():
   if not line or line.startswith('#'):continue
   c,a,z,*_=line.split()
   if c=='chr20':beds.append((int(a),int(z)))
  def keys(path):
   out={'SNP':set(),'INDEL':set()};v=pysam.VariantFile(str(path))
   for rec in v.fetch('chr20',100000,100000+length):
    if rec.pos-1<100000 or rec.stop>100000+length or not any(a<=rec.pos-1 and rec.stop<=z for a,z in beds):continue
    if set(rec.filter.keys())-{'PASS','.'}:continue
    gt=next(iter(rec.samples.values())).get('GT',None)
    for i,alt in enumerate(rec.alts or [],1):
     if gt and i not in gt:continue
     if not alt or alt.startswith('<') or alt=='*':continue
     kind='SNP' if len(rec.ref)==len(alt)==1 else 'INDEL';out[kind].add((rec.contig,rec.pos,rec.ref,alt))
   return out
  truth=keys(normtruth);query=keys(norm);accuracy={}
  for k in truth:
   tp=len(truth[k]&query[k]);fp=len(query[k]-truth[k]);fn=len(truth[k]-query[k]);accuracy[k]={'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
  row['normalized_exact_allele_metrics']=accuracy;write_json(dest/'metrics.json',accuracy);row['output_vcf']=str(calls.relative_to(R));row['output_vcf_sha256']=hashlib.file_digest(calls.open('rb'),'sha256').hexdigest();save()
 S['status']='calling_evaluated_pending_reload';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/discover_download.py — 원문 전문 (1,169 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/discover_download.py", "sha256": "c80f3c16ef6b26b3629d404a42669612e8fa76b72e9e2a563709914f276ab47d", "encoding": "utf-8", "added_final_newline": false} -->
```python
import urllib.request,json,re,hashlib
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/clair3');base='https://www.bio8.cs.hku.hk/clair3/'
for path in ['clair3_models_pytorch/','clair3_models_pytorch/ont/']:
 try:
  b=urllib.request.urlopen(base+path,timeout=45).read();(R/'results'/('index_'+path.strip('/').replace('/','_')+'.html')).write_bytes(b);print(path,b.decode()[:10000],flush=True)
 except Exception as e:print(path,repr(e),flush=True)
base+='demo/quick_demo/ont/';rows=[]
files=['GRCh38_no_alt_chr20.fa','GRCh38_no_alt_chr20.fa.fai','HG003_chr20_demo.bam','HG003_chr20_demo.bam.bai','HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz','HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz.tbi','HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed']
for name in files:
 p=R/'data'/name;print('DOWNLOAD',name,flush=True)
 if not p.exists():
  with urllib.request.urlopen(base+name,timeout=120) as f:p.write_bytes(f.read())
 rows.append({'file':name,'url':base+name,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()});(R/'data/sources.json').write_text(json.dumps(rows,indent=2));print('DONE',name,p.stat().st_size,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/instrument.py — 원문 전문 (2,499 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/instrument.py", "sha256": "9130c01daabcb4faca2e3d4437182ab38fa3449ab71d1baf07946acd875aa84c", "encoding": "utf-8", "added_final_newline": false} -->
```python
import os,json,time,hashlib
from pathlib import Path
import numpy as np,torch,psutil
G=2**30

def install(namespace):
 original_load=namespace['_load_torch_checkpoint'];original_predict=namespace['_torch_predict'];state={'first':True}
 def emit(r):
  root=Path(os.environ['CLAIR3_METRICS_DIR']);root.mkdir(parents=True,exist_ok=True)
  with (root/f'worker_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps({'time_unix':time.time(),'pid':os.getpid(),**r})+'\n')
 def mem():return dict(allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,peak_allocated_gib=torch.cuda.max_memory_allocated()/G,peak_reserved_gib=torch.cuda.max_memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,available_gib=psutil.virtual_memory().available/G)
 def load(model,path,device):
  t=time.time();original_load(model,path,device)
  if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  state['model']=type(model).__name__;state['checkpoint']=path;emit(dict(event='model_only_postload',model=state['model'],checkpoint=path,total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),load_seconds=time.time()-t,dtype=str(next(model.parameters()).dtype),tf32=torch.backends.cuda.matmul.allow_tf32,memory=mem()))
 def predict(model,device,X):
  if device.type=='cuda':torch.cuda.synchronize()
  t=time.time();Y=original_predict(model,device,X)
  if device.type=='cuda':torch.cuda.synchronize()
  emit(dict(event='forward',model=state.get('model'),shape=list(X.shape),dtype=str(X.dtype),seconds=time.time()-t,memory=mem()))
  cache_dir=os.environ.get('CLAIR3_FULL_CANDIDATE_CACHE')
  if cache_dir:
   dest=Path(cache_dir);dest.mkdir(parents=True,exist_ok=True);idx=state.get('cache_index',0);state['cache_index']=idx+1;name=f"{state.get('model')}_{os.getpid()}_{idx}.npz";np.savez_compressed(dest/name,X=X,Y=Y);emit(dict(event='full_actual_candidate_cache',file=str(dest/name),model=state.get('model'),shape=list(X.shape),synthetic_rows=0))

  if state['first']:
   root=Path(os.environ['CLAIR3_METRICS_DIR']);np.savez_compressed(root/f'reload_input_{os.getpid()}.npz',X=X[:min(32,len(X))],Y=Y[:min(32,len(Y))]);emit(dict(event='reload_sample',file=f'reload_input_{os.getpid()}.npz',model=state.get('model'),checkpoint=state.get('checkpoint')));state['first']=False
  return Y
 namespace['_load_torch_checkpoint']=load;namespace['_torch_predict']=predict
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "clair3/results/candidate_batch_20261028_185641_089149/source_instrument.py", "source": "clair3/scripts/instrument.py", "sha256": "9130c01daabcb4faca2e3d4437182ab38fa3449ab71d1baf07946acd875aa84c"} -->

</details>

<details>
<summary>clair3/scripts/launch.py — 원문 전문 (1,172 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/launch.py", "sha256": "032dc38ce8ceda6481c9938e762b6cfa50d33fac9c04b53e3e389d4773296e74", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,datetime,json,sys
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/clair3');script=sys.argv[1] if len(sys.argv)>1 else 'benchmark.py';assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
if script=='benchmark.py':run='clair3_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LATEST').write_text(run+'\n')
else:run=(R/'results/LATEST').read_text().strip();D=R/'results'/run
cmd=['sg','docker','-c',f'docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v {R}:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID={run} -e PYTHONPATH=/work/scripts:/work/source -e MPLCONFIGDIR=/work/.cache/matplotlib local/clair3-spark:9d1dd3f /work/.venv/bin/python -u /work/scripts/{script}'];stem=script.removesuffix('.py');(D/(stem+'_command.json')).write_text(json.dumps(cmd,indent=2));print(run,script,flush=True)
with (D/(stem+'.log')).open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/(stem+'_exit.json')).write_text(json.dumps({'exit_code':code}));print('EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/reload.py — 원문 전문 (2,072 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/reload.py", "sha256": "1311d548e5698ddb3f660e2c7ed232adc657d8908750554b063598da1d4c60b9", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,os,json,gc
sys.path[:0]=['/support','/work/source']
from gpu_support import write_json
import numpy as np,torch
from clair3.model import Clair3_P,Clair3_F
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());torch.set_num_threads(1);results=[];tested=set()
for row in s['conditions']:
 dest=D/row['name']/'instrumentation'
 for file in dest.glob('worker_*.jsonl'):
  events=[json.loads(l) for l in file.read_text().splitlines()];load=next((e for e in events if e['event']=='model_only_postload'),None);sample=next((e for e in events if e['event']=='reload_sample'),None)
  if not load or not sample or load['model'] in tested:continue
  cl=load['model'];p=load['checkpoint'];p=p if p.endswith('.pt') else p+'.pt';v=np.load(dest/sample['file']);m=(Clair3_P if cl=='Clair3_P' else Clair3_F)(predict=True,add_indel_length=cl=='Clair3_F').cuda().eval();checkpoint=torch.load(p,map_location='cuda');m.load_state_dict(checkpoint.get('state_dict',checkpoint));torch.backends.cuda.matmul.allow_tf32=load['tf32']
  full_shape=next(e['shape'] for e in events if e['event']=='forward');x=np.zeros(full_shape,dtype=v['X'].dtype);x[:len(v['X'])]=v['X']
  with torch.inference_mode():y=m(torch.from_numpy(x).cuda()).cpu().numpy()[:len(v['X'])]
  diff=float(np.abs(y-v['Y']).max());r={'model':cl,'actual_input_shape':list(v['X'].shape),'independent_process':True,'max_abs_difference':diff,'passed':diff<=1e-4,'original_batch_shape':full_shape,'note':'first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4'};results.append(r);write_json(D/'reload.json',results);print(json.dumps(r),flush=True);assert r['passed'];tested.add(cl);del m,checkpoint;gc.collect();torch.cuda.empty_cache()
assert tested=={'Clair3_P','Clair3_F'},tested;write_json(D/'reload.json',results);s['reload']=results;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(results),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/report.py — 원문 전문 (8,455 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/report.py", "sha256": "1d088fa9f4579dafd0ff4edae1d86c28658db578ae4248497a691388092e6763", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,csv
R=Path('/home/hyein/ai-benchmark/clair3');run=(R/'results/LATEST').read_text().strip();D=R/'results'/run;s=json.loads((D/'summary.json').read_text());out=[]
def a(t=''):out.append(t)
def table(h,rows):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rows:a('| '+' | '.join(map(str,r))+' |')
 a()
def f(x):return 'N/A' if x is None else f'{x:.3f}'
a('# Clair3 — DGX Spark long-read 변이 호출');a();a(f'run `{run}` · 상태 **{s["status"]}** · [summary](../results/{run}/summary.json) · [로그](../results/{run}/benchmark.log) · [자원 시계열](../results/{run}/resources.csv)');a();a('## 모델과 실제 실행 환경');a();a('Clair3 공식 소스 v2.0.4, commit `9d1dd3f2475072b6255900e406c34c6b0545fc88`. v2.0.4 이름의 git tag는 없어 실제 main commit과 shared/version.py를 고정했다. 공식 PyTorch `ont/pileup.pt`와 `ont/full_alignment.pt`를 사용하며 Guppy3.6 ONT 데모에 맞춘 구형 ONT 모델이다. 최신 R10.4.1/Dorado 모델과 혼용하지 않는다.');a();a('환경 '+json.dumps(s['environment']));a();a('원래 NGC PyTorch/CUDA 이미지를 보존하고 별도 ARM64 이미지를 만들었다. CFFI native 전처리는 시스템htslib으로 컴파일, phasing은공식 기본whatshap, 선택적 가속용PyPy 대신 CPython3.12를 썼다. Spark의 nvidia-smi memory.free=N/A 때문에 공식 VRAM 잔량 기반 worker 산정을 그대로 쓸 수 없어, GPU predictor worker1개로 명시하는 호환 변경을 적용했다. 모형 forward·배치·변이 호출 알고리즘은 유지했다. 원본·patch는 `results/benchmark_compatibility.patch`에 저장했다.');a();a('공식 GPU 배치 정책 predictBatchSize200×5를 고정하며 마지막 batch는 후보 수에 따라 작다. CPU threads4, GPU0, FP32가중치, 실제 입력 특징은 pileup int32/full-alignment int8이며 모델 내부에서 float로 변환/스케일한다. GPU 다른 실험은동시에 실행하지 않았다.');a();loads={}
for r in s['conditions']:
 for v in r['worker_loads']:loads.setdefault(v['model'],v)
if loads:table(['모델','실제파라미터','적재 allocatedGiB','reservedGiB','RSSGiB','가용GiB'],[[k,v['total_parameters'],*[f(v['memory'][x]) for x in ['allocated_gib','reserved_gib','rss_gib','available_gib']]] for k,v in loads.items()])
a('이론 FP32가중치 용량은각 파라미터수×4byte다. 위는 각 worker의 checkpoint 적재를 완료하고 첫forward전의 실측이며 입력tensor/optimizer는없다. 모델 두개는순차적으로 쓰며 파라미터합을한번에 올린 크기로 가정하지 않는다.');a();a('## 실제 입력과 정답');a();audit=s['input_audit'];a('공식 ONT quick demo: GIAB HG003, Guppy3.6, GRCh38_no_alt chr20, chr20:100000–300000(BED0-based half-open). 원본 BAM은coordinate정렬·index·참조 contig길이를 확인했다. GIAB HG003 v4.2.1 VCF와 confident BED도 확보했다. 데이터·모델 URL/바이트/SHA-256은 `data/sources.json`과 `models/ont/sources.json`.');a();a('첫 실제 read: '+json.dumps(audit['first_read'],ensure_ascii=False));a();table(['구간bp','겹치는read','평균coverage','중앙coverage','최소/최대','평균read길이'],[[r['length_bp'],r['reads_overlapping'],f(r['coverage_mean']),f(r['coverage_median']),str(r['coverage_min'])+'/'+str(r['coverage_max']),f(r['read_length_mean'])] for r in audit['regions']]);a('분석 구간 확대와 coverage 확대는 별개다. coverage조건은동일200kb에서 samtools view -s42.5(seed42,fraction0.5)로 read를 downsample했다. 실제 coverage는 각 실행후 다시 기록했다. 긴read는여러구간에겹칠수있어 read개수가단순구간길이에비례하지않는다.');a();a('모델은 BAM자체를tensor로보지않는다. pileup후보의33위치×채널특징과 full-alignment후보의read-depth×33위치×채널 특징으로바뀐다. 실제shape와 첫후보tensor/예측을 저장했다. 최종출력은좌표·REF/ALT·genotype·quality를 가진VCF다.');a();a('## 분석 입력별 GPU 사용률·메모리');a();rows=[];flat=[]
for r in s['conditions']:
 rows.append([r['name'],r['bp'],f(r.get('coverage_mean')),r.get('reads_overlapping','N/A'),r['candidate_tensors'],f(r['seconds']),f(r['throughput']),f(r['gpu_util_avg'])+'/'+f(r['gpu_util_max']),f(r['baseline']['allocated_gib']),f(r['memory']['peak_allocated_gib']),f(r['peak_delta_gib'])]);flat.append({'name':r['name'],'region_bp':r['bp'],'coverage_mean':r.get('coverage_mean'),'reads':r.get('reads_overlapping'),'candidates':r['candidate_tensors'],'seconds':r['seconds'],'bp_per_second':r['throughput'],'gpu_util_avg':r['gpu_util_avg'],'gpu_util_max':r['gpu_util_max'],'baseline_allocated_gib':r['baseline']['allocated_gib'],'peak_worker_allocated_gib':r['memory']['peak_allocated_gib'],'peak_worker_reserved_gib':r['memory']['peak_reserved_gib'],'peak_delta_gib':r['peak_delta_gib'],'forward_seconds':r['forward_seconds'],'samples':r['samples']})
table(['조건','bp','coverage','read','후보tensor처리수','전체초','bp/초','GPU평균/최대%','시작GiB','worker peakGiB','증가GiB'],rows)
if flat:
 with (R/'reports/comparison.csv').open('w',newline='') as f0:w=csv.DictWriter(f0,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for r in s['conditions']:a('- '+r['name']+' 실제batch입력: `'+str(r['actual_input_shapes'])+'`; GPU forward+전송/CPU출력 합 '+f(r['forward_seconds'])+'초.')
a();a('전체시간은worker시작·모델적재·CPU후보생성·I/O·GPU추론·whatshap·후처리·VCF저장을포함하는end-to-end다. GIAB평가·다운샘플·입력감사·정규화는제외했다. 독립warmup/compile시간은N/A이며worker시작비용을전체에포함한다. 순수forward시간은별도기록하고CUDA synchronize했다. 표의 처리량은bp/초이며 후보수분포/coverage가다르면엄밀한비교에한계가있다.');a();a('GPU평균/최대는전체pipeline구간의활동시간비율이다. CPU전처리·phasing·I/O기간이포함돼평균이낮을수있다. GPU커널구간만의FLOPS나memory비율이아니다. 전체RAM121.688GiB를90%채우는것은목표가아니다. RSS와CUDA를더하지않는다.');a();a('CUDA peak는여러subprocess 중 **가장큰단일worker allocator peak**이며프로세스합계/장치전체점유율이아니다. 각worker모델적재후peak를reset했다. baseline과delta는동일peakworker로맞춰기록했다. 시계열CSV allocated/reserved는감시parent값이므로모델메모리로쓰지않는다. 실제worker메모리는각조건instrumentation/worker_*.jsonl에있다. RSS는parent시계열과worker스냅샷으로구분한다.');a();a('## 결과 검증');a();a('호출VCF와truth를bcftools norm(-f참조,-m-any)으로left-normalize/split한뒤,해당분석구간및GIABconfident BED 안의PASS실제ALTallele exact matching을계산했다. genotype포함haplotype-aware hap.py/vcfeval평가가아니며복잡한동등표현변이는FP/FN으로나뉠수있다. 공식quick demo의hap.py수치와직접비교하지않는다.');a();metrics=[]
for r in s['conditions']:
 for k,v in r.get('normalized_exact_allele_metrics',{}).items():metrics.append([r['name'],k,v['TP'],v['FP'],v['FN'],f(v['precision']),f(v['recall']),f(v['f1'])])
table(['조건','종류','TP','FP','FN','precision','recall','F1'],metrics);a('원본VCF·normalizedVCF·metrics.json·명령·exit code·예측tensor·실제입력예시를각조건폴더에저장했다. 독립프로세스checkpoint reload: '+json.dumps(s.get('reload',{'status':'pending'})));a();a('공식Train은PyTorch학습을지원하지만이번Clair3과제범위는long-read germline호출·자원측정이다. 파인튜닝은수행하지않았으며학습완료라고쓰지않는다. GIAB단일샘플의작은구간실험이므로새샘플일반화나WGS전체성능은미확인이다.');a();a('설치시curl개발라이브러리충돌을전용이미지내에서정리했고,첫호출전처리의pigz누락을보완했다. 실패run과로그를보존했다.');a();a('## 공식 출처');a();a('- [Clair3 공식소스·설치·모델](https://github.com/HKU-BAL/Clair3)\n- [공식 ONT quick demo](https://github.com/HKU-BAL/Clair3/blob/main/docs/quick_demo/ont_quick_demo.md)\n- [공식 PyTorch ONT 가중치](https://www.bio8.cs.hku.hk/clair3/clair3_models_pytorch/ont/)');(R/'reports/Clair3.md').write_text('\n'.join(out)+'\n');print(R/'reports/Clair3.md')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/setup.py — 원문 전문 (842 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/setup.py", "sha256": "deb5e12b435a19d7e8c8083fb70136fcece1a56cdba24734202356bd70c4b10d", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,subprocess,importlib.metadata,json
R=Path('/work');(R/'constraints.txt').write_text('torch=='+importlib.metadata.version('torch')+'\n');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True)
subprocess.run(['/work/.venv/bin/pip','install','-c','/work/constraints.txt','--cache-dir','/work/.cache/pip','hdf5plugin','numexpr','tqdm','cffi','torchmetrics','pysam','whatshap','mpmath==1.2.1','intervaltree'],check=True)
subprocess.run(['/work/.venv/bin/python','build.py'],cwd='/work/source',check=True)
(R/'results/setup_done.json').write_text(json.dumps({'torch':importlib.metadata.version('torch'),'source_commit':'9d1dd3f2475072b6255900e406c34c6b0545fc88','native':'CFFI ARM64 compiled with system htslib; whatshap phasing; CPython preprocessing instead of PyPy'}))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/results/CallVariantsFromCffiGPU.original.py — 원문 전문 (32,174 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/results/CallVariantsFromCffiGPU.original.py", "sha256": "5c55410ab45d1b1ea012ac204b9fd374c46fb2bbbab1dcd0c49353c4a40a3e72", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys
import logging
from argparse import ArgumentParser, SUPPRESS
import subprocess
import os

from shared.utils import str2bool, log_error, log_warning, str_none
logging.basicConfig(format='%(message)s', level=logging.INFO)

file_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
main_entry = os.path.join(file_directory, "clair3.py")

def rocm_smi_free_memory():
    """Free VRAM in MB per physical GPU id from rocm-smi, or {} when it is unavailable."""
    try:
        output = subprocess.check_output(
            "rocm-smi --showmeminfo vram --csv".split(), stderr=subprocess.DEVNULL).decode('ascii')
    except (OSError, subprocess.CalledProcessError):
        return {}

    rows = [line.split(',') for line in output.splitlines() if line.strip()]
    if not rows:
        return {}
    header = [cell.strip().lower() for cell in rows[0]]
    total_idx = next((i for i, cell in enumerate(header) if 'total memory' in cell), None)
    used_idx = next((i for i, cell in enumerate(header) if 'used memory' in cell), None)
    if total_idx is None or used_idx is None:
        return {}

    free_memory = {}
    for row in rows[1:]:
        if len(row) <= max(total_idx, used_idx):
            continue
        # rocm-smi labels rows 'card0', 'card1', ...; the suffix is the physical id.
        name = row[0].strip()
        gpu_id = name[len('card'):] if name.startswith('card') else name
        if not gpu_id.isdigit():
            continue
        try:
            free_memory[int(gpu_id)] = (int(row[total_idx]) - int(row[used_idx])) // 1024 // 1024
        except ValueError:
            continue
    return free_memory

def get_gpu_memory(gpu_id):
    """Free VRAM in MB for a *physical* GPU id, via nvidia-smi (CUDA) or rocm-smi (ROCm).

    Queried out of process on purpose: creating a GPU context here would both consume
    VRAM and skew the very reading used to size the workers.
    """
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv "
        if gpu_id is not None:
            command += f" --id={gpu_id}"
        memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        return [int(x.split()[0]) for x in memory_free_info]
    except (OSError, subprocess.CalledProcessError):
        pass

    free_memory = rocm_smi_free_memory()
    if not free_memory:
        raise RuntimeError("Neither nvidia-smi nor rocm-smi is available to query GPU memory")
    if gpu_id is None:
        return [free_memory[key] for key in sorted(free_memory)]
    return [free_memory[gpu_id]]

def get_all_gpu_ids():
    """Physical GPU ids on this host, as reported by nvidia-smi (CUDA) or rocm-smi (ROCm).

    Both tools ignore CUDA_VISIBLE_DEVICES, so these are physical indices -- the same
    numbering that `--device=cuda:N` and CUDA_VISIBLE_DEVICES themselves refer to.
    """
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=index --format=csv,noheader".split()).decode('ascii')
        nvidia_ids = sorted(int(token) for token in output.split() if token.strip().isdigit())
    except (OSError, subprocess.CalledProcessError):
        nvidia_ids = []
    return nvidia_ids or sorted(rocm_smi_free_memory())

def parse_device_ids(device):
    """Turn a `--device` value such as 'cuda:3' or 'cuda:0,1' into physical GPU ids."""
    parts = [part.strip() for part in device.split("cuda:")[-1].split(",")]
    if not parts or any(not part.isdigit() for part in parts):
        print(log_error("Please specify the device as '--device=cuda:0' or '--device=cuda:0,1'"), file=sys.stderr)
        sys.exit(1)
    return [int(part) for part in parts]

def check_gpu_memory(memory, device_ids=None, print_log=True):
    """Work out how many worker slots each GPU can take.

    device_ids are *physical* GPU ids, as requested through --device or
    CUDA_VISIBLE_DEVICES; None means every GPU on the host. The ids handed back are
    physical too, because that is what the workers put into CUDA_VISIBLE_DEVICES and
    what `nvidia-smi --id=` and `rocm-smi` report. Mixing those physical ids with the
    0-based indices of the visible devices is what used to send every worker to GPU 0
    (issue #472).
    """
    all_gpu_ids = get_all_gpu_ids()
    if not all_gpu_ids:
        print(log_error("No GPU available, Please disabling --use_gpu for variant calling, exiting."), file=sys.stderr)
        sys.exit(1)

    if device_ids is None:
        device_ids = all_gpu_ids
    unknown_ids = [gpu_id for gpu_id in device_ids if gpu_id not in all_gpu_ids]
    if unknown_ids:
        print(log_error("Requested GPU id(s) {} not found on this host (available: {}), exiting.".format(
            unknown_ids, all_gpu_ids)), file=sys.stderr)
        sys.exit(1)

    gpu_id_list = []
    for gpu_id in device_ids:
        free_mem = get_gpu_memory(gpu_id=gpu_id)[0]  # Convert to MB
        gpu_threads = free_mem // memory
        gpu_id_list += [gpu_id] * gpu_threads
        if print_log:
            print(f"GPU {gpu_id} free memory: {free_mem} MB, assigning {memory} MB per thread, {gpu_threads} threads available")

    if len(gpu_id_list) == 0:
        print(log_error("No memory in GPU, Please assign GPU memory first, exiting."), file=sys.stderr)
        sys.exit(1)
    return gpu_id_list

def Run(args):
    try:
        rc = subprocess.check_call('time', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time = 'time '
    except subprocess.CalledProcessError as e:
        time = ''
    full_alignment_cpu_threads = max(args.cpu_threads//4, 1)
    prefix = "full_alignment" if not args.pileup else "pileup"
    dwell_flag = ' --enable_dwell_time True' if args.enable_dwell_time else ''

    pileup_per_thread_gpu_memory = 5000  # GB
    full_alignment_per_thread_gpu_memory = 8000  # GB

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")

    if args.device and args.device != "EMPTY":
        if not args.device.startswith('cuda:'):
            print(log_error("Please specify the device as '--device=cuda:0' or '--device=cuda:0,1'"), file=sys.stderr)
            sys.exit(1)
        device_ids = parse_device_ids(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(item) for item in device_ids])
    elif cuda_visible_devices not in ("-1", ""):
        # No --device given: honour a preset CUDA_VISIBLE_DEVICES instead of quietly
        # using every GPU on the host.
        device_ids = [int(x) for x in cuda_visible_devices.split(",") if x.strip().isdigit()] or None
        if device_ids:
            print(f"[INFO] ENV 'CUDA_VISIBLE_DEVICES' is set to {device_ids}, using these GPUs only.")
    else:
        device_ids = None

    gpu_id_list = check_gpu_memory(pileup_per_thread_gpu_memory if args.pileup else full_alignment_per_thread_gpu_memory, device_ids, print_log=False)

    if args.pileup:
        cp_command = time + args.parallel + ' -C " " '
        cp_command += ' --joblog ' + args.output_dir + '/log/parallel_1_pileup_create_tensor.log'
        cp_command += ' -j ' + str(args.cpu_threads*2)
        cp_command += ' --retries 4'
        cp_command += ' ' + args.python + ' ' + main_entry + " CallVariantsFromCffi"
        cp_command += ' --chkpnt_fn ' + args.chkpnt_fn
        cp_command += ' --bam_fn ' + args.bam_fn
        cp_command += " --call_fn " + os.path.join(args.call_fn, "pileup_{1}_{2}_call.vcf.gz")
        cp_command += ' --sampleName ' + args.sampleName
        cp_command += ' --ref_fn ' + args.ref_fn if args.ref_fn else ''
        cp_command += ' --vcf_fn ' + args.vcf_fn if args.vcf_fn != "EMPTY" else ''
        cp_command += ' --extend_bed ' + args.output_dir + "/tmp/split_beds/{1}"
        if args.bed_fn != "" and args.bed_fn is not None:
            cp_command += ' --bed_fn ' + str(args.bed_fn)
        cp_command += ' --threads ' + str(args.internal_threads)
        cp_command += ' --ctgName {1} '
        cp_command += ' --chunk_id {2} '
        cp_command += ' --chunk_num {3} '
        cp_command += ' --platform ' + args.platform
        cp_command += ' --fast_mode ' + str(args.fast_mode)
        cp_command += ' --snp_min_af ' + str(args.snp_min_af)
        cp_command += ' --indel_min_af ' + str(args.indel_min_af)
        cp_command += ' --minMQ ' + str(args.minMQ)
        cp_command += ' --minCoverage ' + str(args.minCoverage)
        cp_command += ' --call_snp_only ' + str(args.call_snp_only)
        cp_command += ' --enable_variant_calling_at_sequence_head_and_tail ' + str(
            args.enable_variant_calling_at_sequence_head_and_tail)
        cp_command += ' --gvcf ' + str(args.gvcf)
        cp_command += ' --base_err ' + str(args.base_err)
        cp_command += ' --gq_bin_size ' + str(args.gq_bin_size)
        cp_command += ' --enable_long_indel ' + str(args.enable_long_indel)
        cp_command += ' --samtools ' + str(args.samtools)
        cp_command += ' --temp_file_dir ' + args.output_dir + "/tmp/gvcf_tmp_output"
        cp_command += ' --pileup '
        cp_command += ' --keep_iupac_bases ' + str(args.keep_iupac_bases)
        cp_command += ' --cmd_fn ' + str(args.cmd_fn)
        cp_command += ' --use_gpu ' + str(args.use_gpu)
        cp_command += " --tensor_can_fn " + args.output_dir + f"/tmp/pileup_output/{prefix}" + "_{1}_{2}"
        cp_command += ' :::: ' + args.output_dir + "/tmp/CHUNK_LIST"

        try:
            return_code = subprocess.check_call(cp_command, shell=True, stdout=sys.stdout)
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                "ERROR in Pileup tensor generation, THE FOLLOWING COMMAND FAILED: {}\n".format(cp_command))
            exit(1)

        file_list = []
        ctg_set = set()
        with open(args.output_dir + "/tmp/CHUNK_LIST") as f:
            for row in f:
                ctg, chunk_id, chunk_num = row.rstrip().split(' ')
                if os.path.exists(
                        os.path.join(args.output_dir, 'tmp', 'pileup_output', f'{prefix}_{ctg}_{chunk_id}.npy')):
                    file_list.append(
                        os.path.join(args.output_dir, 'tmp', 'pileup_output', f'{prefix}_{ctg}_{chunk_id}'))
                    ctg_set.add(ctg)
        if len(file_list) == 0:
            return
        logging.info("[INFO] Found {} pileup tensor files to process".format(len(file_list)))


        # check gpu resource again, split the file based on the GPU resource
        gpu_id_list = check_gpu_memory(pileup_per_thread_gpu_memory, device_ids)

        total_gpu_treads = len(gpu_id_list)
        each_gpu_file_count = len(file_list) // total_gpu_treads if len(file_list) % total_gpu_treads == 0 else len(
            file_list) // total_gpu_treads + 1

        all_gpu_chunk_file_list = os.path.join(args.output_dir, 'tmp', f'{prefix}_all_gpu_chunk')
        with open(all_gpu_chunk_file_list, 'w') as all_f:
            for i in range(total_gpu_treads):
                gpu_file_list = file_list[i * each_gpu_file_count: (i + 1) * each_gpu_file_count]
                if len(gpu_file_list) == 0:
                    continue

                gpu_chunk_num_fn = os.path.join(args.output_dir, 'tmp', f'{prefix}_gpu_chunk_{i}')
                with open(gpu_chunk_num_fn, 'w') as f:
                    for fn in gpu_file_list:
                        f.write(fn + '\n')
                all_f.write(' '.join([str(i), str(gpu_id_list[i]), gpu_chunk_num_fn]) + '\n')


        for f in os.listdir(args.call_fn):
            if f.startswith(prefix) and f.endswith('.vcf'):
                os.remove(os.path.join(args.call_fn, f))

        cp_command = time + args.parallel + ' -C " " '
        cp_command += ' --joblog ' + args.output_dir + '/log/parallel_1_pileup_call_variant.log'
        cp_command += ' -j ' + str(total_gpu_treads)
        cp_command += ' --retries 4'
        cp_command += ' ' + args.python + ' ' + main_entry + " CallVariantsFromCffi"
        cp_command += ' --chkpnt_fn ' + args.chkpnt_fn
        cp_command += ' --bam_fn ' + args.bam_fn
        cp_command += " --call_fn " + os.path.join(args.call_fn, f"{prefix}_" + "{1}.vcf")
        cp_command += ' --sampleName ' + args.sampleName
        cp_command += ' --ref_fn ' + args.ref_fn if args.ref_fn else ''
        cp_command += ' --vcf_fn ' + args.vcf_fn if args.vcf_fn != "EMPTY" else ''
        if args.bed_fn != "" and args.bed_fn is not None:
            cp_command += ' --bed_fn ' + str(args.bed_fn)
        cp_command += ' --threads ' + str(args.internal_threads)
        cp_command += ' --platform ' + args.platform
        cp_command += ' --fast_mode ' + str(args.fast_mode)
        cp_command += ' --snp_min_af ' + str(args.snp_min_af)
        cp_command += ' --indel_min_af ' + str(args.indel_min_af)
        cp_command += ' --minMQ ' + str(args.minMQ)
        cp_command += ' --minCoverage ' + str(args.minCoverage)
        cp_command += ' --call_snp_only ' + str(args.call_snp_only)
        cp_command += ' --enable_variant_calling_at_sequence_head_and_tail ' + str(
            args.enable_variant_calling_at_sequence_head_and_tail)
        cp_command += ' --gvcf ' + str(args.gvcf)
        cp_command += ' --base_err ' + str(args.base_err)
        cp_command += ' --gq_bin_size ' + str(args.gq_bin_size)
        cp_command += ' --enable_long_indel ' + str(args.enable_long_indel)
        cp_command += ' --samtools ' + str(args.samtools)
        cp_command += ' --temp_file_dir ' + args.output_dir + "/tmp/gvcf_tmp_output"
        cp_command += ' --pileup '
        cp_command += ' --keep_iupac_bases ' + str(args.keep_iupac_bases)
        cp_command += ' --cmd_fn ' + str(args.cmd_fn)
        cp_command += ' --gpu_id {2} '
        cp_command += ' --cpu_threads ' + str(args.cpu_threads)
        cp_command += ' --use_gpu ' + str(args.use_gpu)
        cp_command += " --output_tensor_can_fn_list {3} "
        cp_command += ' :::: ' + all_gpu_chunk_file_list

        try:
            return_code = subprocess.check_call(cp_command, shell=True, stdout=sys.stdout)
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                "ERROR in Pileup model calling, THE FOLLOWING COMMAND FAILED: {}\n".format(cp_command))
            exit(1)

        print("[INFO] Removing temporary tensor files...")
        for f in file_list:
            if os.path.exists(f + ".npy"):
                os.remove(f + ".npy")
            if os.path.exists(f + ".info"):
                os.remove(f + ".info")

    else:
        # create full-alignment tensor
        ct_command = '(' + time + args.parallel + ' -C " "'
        ct_command += ' --joblog ' + args.output_dir + '/log/parallel_6_full_alignment_create_tensor.log'
        ct_command += ' -j ' + str(args.threads)
        ct_command += ' --retries 4'
        ct_command += ' ' + args.python + ' ' + main_entry + " CallVariantsFromCffi"
        ct_command += ' --chkpnt_fn ' + args.chkpnt_fn
        ct_command += ' --bam_fn ' + args.bam_fn
        ct_command += ' --sampleName ' + args.sampleName
        ct_command += ' --vcf_fn ' + args.vcf_fn if args.vcf_fn != "EMPTY" else ''
        ct_command += ' --ref_fn ' + args.ref_fn if args.ref_fn else ''
        ct_command += ' --full_aln_regions {1}'
        ct_command += ' --ctgName {1/.} '
        ct_command += " --add_indel_length" if args.add_indel_length else ''
        ct_command += ' --no_phasing_for_fa ' + str(args.no_phasing_for_fa) if args.no_phasing_for_fa else ''
        ct_command += ' --minMQ ' + str(args.minMQ)
        ct_command += ' --minCoverage ' + str(args.minCoverage)
        ct_command += ' --phased_vcf_fn ' + os.path.join(args.phased_vcf_fn,
                                                         "phased_{1/.}.vcf.gz") if args.phased_vcf_fn else ''
        ct_command += ' --gvcf ' + str(args.gvcf)
        ct_command += ' --enable_long_indel ' + str(args.enable_long_indel)
        ct_command += ' --samtools ' + args.samtools
        ct_command += ' --use_gpu ' + str(args.use_gpu)
        ct_command += ' --keep_iupac_bases ' + str(args.keep_iupac_bases)
        ct_command += ' --cmd_fn ' + args.cmd_fn if args.cmd_fn else ''
        ct_command += ' --platform ' + args.platform
        ct_command += dwell_flag
        ct_command += " --tensor_can_fn " + args.output_dir + "/tmp/full_alignment_output/{1/}"
        ct_command += ' :::: ' + args.full_aln_files
        ct_command += ' ) 2>&1 | tee ' + args.output_dir + '/log/6_full_alignment_create_tensor.log'

        try:
            return_code = subprocess.check_call(ct_command, shell=True, stdout=sys.stdout)
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                "ERROR in Full-alignment create tensor, THE FOLLOWING COMMAND FAILED: {}\n".format(ct_command))
            exit(1)

        # check gpu resource again, split the file based on the GPU resource
        gpu_id_list = check_gpu_memory(full_alignment_per_thread_gpu_memory, device_ids)


        file_list = []
        for f in open(args.full_aln_files, 'r').readlines():
            f = f.strip()
            if not f:
                continue
            suffix = f.split('/')[-1]
            if os.path.exists(args.output_dir + f"/tmp/full_alignment_output/{suffix}.npy"):
                file_list.append(args.output_dir + f"/tmp/full_alignment_output/{suffix}")

        total_gpu_treads = len(gpu_id_list)
        each_gpu_file_count = len(file_list) // total_gpu_treads if len(file_list) % total_gpu_treads == 0 else len(
            file_list) // total_gpu_treads + 1

        all_gpu_chunk_file_list = os.path.join(args.output_dir, 'tmp', f'{prefix}_all_gpu_chunk')
        with open(all_gpu_chunk_file_list, 'w') as all_f:
            for i in range(total_gpu_treads):
                gpu_file_list = file_list[i * each_gpu_file_count: (i + 1) * each_gpu_file_count]
                if len(gpu_file_list) == 0:
                    continue

                gpu_chunk_num_fn = os.path.join(args.output_dir, 'tmp', f'{prefix}_gpu_chunk_{i}')
                with open(gpu_chunk_num_fn, 'w') as f:
                    for fn in gpu_file_list:
                        f.write(fn + '\n')
                all_f.write(' '.join([str(i), str(gpu_id_list[i]), gpu_chunk_num_fn]) + '\n')

        for f in os.listdir(args.call_fn):
            if f.startswith(prefix) and f.endswith('.vcf'):
                os.remove(os.path.join(args.call_fn, f))

        #full-alignment variant calling
        ct_command = '(' + time + args.parallel + ' -C " "'
        ct_command += ' --joblog ' + args.output_dir + '/log/parallel_6_full_alignment_call_variant.log'
        ct_command += ' -j ' + str(total_gpu_treads)
        ct_command += ' --retries 4'
        ct_command += ' ' + args.python + ' ' + main_entry + " CallVariantsFromCffi"
        ct_command += ' --chkpnt_fn ' + args.chkpnt_fn
        ct_command += ' --bam_fn ' + args.bam_fn
        ct_command += ' --call_fn ' + os.path.join(args.call_fn, f"{prefix}_" + "{1}.vcf")
        ct_command += ' --sampleName ' + args.sampleName
        ct_command += ' --vcf_fn ' + args.vcf_fn if args.vcf_fn != "EMPTY" else ''
        ct_command += ' --ref_fn ' + args.ref_fn if args.ref_fn else ''
        ct_command += " --add_indel_length" if args.add_indel_length else ''
        ct_command += ' --no_phasing_for_fa ' + str(args.no_phasing_for_fa) if args.no_phasing_for_fa else ''
        ct_command += ' --minMQ ' + str(args.minMQ)
        ct_command += ' --minCoverage ' + str(args.minCoverage)
        ct_command += ' --phased_vcf_fn ' + os.path.join(args.phased_vcf_fn,
                                                         "phased_{2}.vcf.gz") if args.phased_vcf_fn else ''
        ct_command += ' --gvcf ' + str(args.gvcf)
        ct_command += ' --enable_long_indel ' + str(args.enable_long_indel)
        ct_command += ' --samtools ' + args.samtools
        ct_command += ' --use_gpu ' + str(args.use_gpu)
        ct_command += ' --keep_iupac_bases ' + str(args.keep_iupac_bases)
        ct_command += ' --cmd_fn ' + args.cmd_fn if args.cmd_fn else ''
        ct_command += ' --platform ' + args.platform
        ct_command += ' --gpu_id {2}'
        ct_command += ' --cpu_threads ' + str(full_alignment_cpu_threads)
        ct_command += ' --use_gpu ' + str(args.use_gpu)
        ct_command += dwell_flag
        ct_command += ' --output_tensor_can_fn_list {3} '
        ct_command += ' :::: ' + all_gpu_chunk_file_list
        ct_command += ' ) 2>&1 | tee ' + args.output_dir + '/log/6_full_alignment_call_variant.log'

        try:
            return_code = subprocess.check_call(ct_command, shell=True, stdout=sys.stdout)
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                "ERROR in Full-alignment model calling, THE FOLLOWING COMMAND FAILED: {}\n".format(ct_command))
            exit(1)

        print("[INFO] Removing temporary tensor files...")
        for f in file_list:
            if os.path.exists(f + ".npy"):
                os.remove(f + ".npy")
            if os.path.exists(f + ".info"):
                os.remove(f + ".info")

def main():
    parser = ArgumentParser(description="The wrapper of GPU Call variants")

    parser.add_argument('--platform', type=str, default="ont",
                        help="Sequencing platform of the input. Options: 'ont,hifi,ilmn', default: %(default)s")

    parser.add_argument('--tensor_fn', type=str, default="PIPE",
                        help="Tensor input filename, or stdin if not set")

    parser.add_argument('--full_aln_files', type=str, default=None,
                        help="Tensor input filename, or stdin if not set")

    parser.add_argument('--output_dir', type=str, default=None,
                        help="Tensor input filename, or stdin if not set")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a trained model for variant calling, required")

    parser.add_argument('--call_fn', type=str, default="clair3",
                        help="VCF output filename, or stdout if not set")

    parser.add_argument('--gvcf', type=str2bool, default=False,
                        help="Enable GVCF output, default: disabled")

    parser.add_argument('--ref_fn', type=str, default=None,
                        help="Reference fasta file input, required if --gvcf is enabled")

    parser.add_argument('--ctgName', type=str, default=None,
                        help="The name of the sequence to be processed")

    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-based starting position of the sequence to be processed, optional, will process the whole --ctgName if not set")

    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The 1-based inclusive ending position of the sequence to be processed, optional, will process the whole --ctgName if not set")

    parser.add_argument('--sampleName', type=str, default="SAMPLE",
                        help="Define the sample name to be shown in the VCF file, optional")

    parser.add_argument('--qual', type=int, default=2,
                        help="If set, variants with >=$qual will be marked 'PASS', or 'LowQual' otherwise, optional")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', samtools version >= 1.10 is required, default: %(default)s")

    # options for advanced users
    parser.add_argument('--temp_file_dir', type=str, default='./',
                        help="EXPERIMENTAL: The cache directory for storing temporary non-variant information if --gvcf is enabled, default: %(default)s")

    parser.add_argument('--haploid_precise', action='store_true',
                        help="EXPERIMENTAL: Enable haploid calling mode. Only 1/1 is considered as a variant")

    parser.add_argument('--haploid_sensitive', action='store_true',
                        help="EXPERIMENTAL: Enable haploid calling mode. 0/1 and 1/1 are considered as a variant")

    parser.add_argument('--enable_long_indel', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable long Indel variants(>50 bp) calling")

    parser.add_argument('--keep_iupac_bases', type=str2bool, default=False,
                        help="EXPERIMENTAL: Keep IUPAC (non ACGTN) reference and alternate bases, default: convert all IUPAC bases to N")

    parser.add_argument('--enable_dwell_time', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable dwell time channel for full-alignment tensors")

    # options for debug purpose
    parser.add_argument('--use_gpu', type=str2bool, default=True,
                        help="Use GPU for calling")

    parser.add_argument('--threads', type=int, default=1,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument('--predict_fn', type=str, default=None,
                        help="DEBUG: Output network output probabilities for further analysis")

    parser.add_argument('--input_probabilities', action='store_true',
                        help="DEBUG: Use network probability outputs as input and generate variants from them")

    parser.add_argument('--output_probabilities', action='store_true',
                        help="DEBUG: Output the network probabilities of gt21, genotype, indel_length_1 and indel_length_2")

    parser.add_argument('--enable_variant_calling_at_sequence_head_and_tail', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable variant calling in sequence head and tail start or end regions that flanking 16bp windows having no read support. Default: disable.")

    # options for internal process control
    ## In pileup mode or not (full alignment mode), default: False
    parser.add_argument('--pileup', action='store_true',
                        help=SUPPRESS)

    ## Include indel length in training and calling, false for pileup and true for raw alignment
    parser.add_argument('--add_indel_length', action='store_true',
                        help=SUPPRESS)

    ## The number of chucks to be divided into for parallel processing
    parser.add_argument('--chunk_num', type=int, default=None,
                        help=SUPPRESS)

    ## The chuck ID to work on
    parser.add_argument('--chunk_id', type=int, default=None,
                        help=SUPPRESS)

    ## Enable debug mode, default: False, optional
    parser.add_argument('--debug', action='store_true',
                        help=SUPPRESS)

    ## Generating outputs for ensemble model calling
    parser.add_argument('--output_for_ensemble', action='store_true',
                        help=SUPPRESS)

    ## Use bin file from HDF5 to speed up calling.
    parser.add_argument('--is_from_tables', action='store_true',
                        help=SUPPRESS)

    ## Output reference calls
    parser.add_argument('--showRef', type=str2bool, default=True,
                        help=SUPPRESS)

    # Pileup create tensor options for pileup calling
    parser.add_argument('--bam_fn', type=str, default="input.bam", required=True,
                        help="Sorted BAM file input, required")

    parser.add_argument('--bed_fn', type=str, nargs='?', action="store", default=None,
                        help="Call variant only in the provided regions. Will take an intersection if --ctgName and/or (--ctgStart, --ctgEnd) are set")

    parser.add_argument('--snp_min_af', type=float, default=0.08,
                        help="Minimum snp allele frequency for a site to be considered as a candidate site, default: %(default)f")

    parser.add_argument('--indel_min_af', type=float, default=0.15,
                        help="Minimum indel allele frequency for a site to be considered as a candidate site, default: %(default)f")

    parser.add_argument('--extend_bed', nargs='?', action="store", type=str, default=None,
                        help="DEBUG: Extend the regions in the --bed_fn by a few bp for tensor creation, default extend 16bp")

    parser.add_argument('--vcf_fn', type=str, default=None,
                        help="Candidate sites VCF file input, if provided, variants will only be called at the sites in the VCF file,  default: %(default)s")

    parser.add_argument('--minCoverage', type=int, default=2,
                        help="EXPERIMENTAL: Minimum coverage required to call a variant, default: %(default)f")

    parser.add_argument('--minMQ', type=int, default=5,
                        help="EXPERIMENTAL: If set, reads with mapping quality with <$minMQ are filtered, default: %(default)d")

    parser.add_argument('--minBQ', type=int, default=0,
                        help="EXPERIMENTAL: If set, bases with base quality with <$minBQ are filtered, default: %(default)d")

    parser.add_argument('--max_depth', type=int, default=144,
                        help="EXPERIMENTAL: Maximum full alignment depth to be processed. default: %(default)s")

    parser.add_argument('--fast_mode', type=str2bool, default=False,
                        help="EXPERIMENTAL: Skip variant candidates with AF <= 0.15, default: %(default)s")

    parser.add_argument('--call_snp_only', type=str2bool, default=False,
                        help="EXPERIMENTAL: Call candidates pass snp minimum AF only, ignore Indel candidates")

    parser.add_argument('--base_err', default=0.001, type=float,
                        help='DEBUG: Estimated base error rate in gvcf option, default: %(default)f')

    parser.add_argument('--gq_bin_size', default=5, type=int,
                        help='DEBUG: Default gq bin size for merge non-variant block in gvcf option, default: %(default)d')

    parser.add_argument('--bp_resolution', action='store_true',
                        help="DEBUG: Enable bp resolution for GVCF, default: disabled")

    # Full-alignment create tensor options for full-alignment calling
    parser.add_argument('--phased_vcf_fn', type=str, default=None,
                        help="Use heterozygous SNP variants in phased vcf file for haplotaging")

    parser.add_argument('--no_phasing_for_fa', type=str2bool, default=False,
                        help="EXPERIMENTAL: Call variants without whatshap or longphase phasing in full alignment calling")

    ## Provide the regions to be included in full-alignment based calling
    parser.add_argument('--full_aln_regions', type=str, default=None,
                        help=SUPPRESS)

    ## If defined, added command line into VCF header
    parser.add_argument('--cmd_fn', type=str_none, default=None,
                        help=SUPPRESS)

    parser.add_argument(
        "--parallel", type=str, default="parallel",
                        help="Absolute path of parallel, parallel >= 20191122 is required."
    )

    parser.add_argument(
        "--pypy", type=str, default="pypy3",
                        help="Absolute path of pypy3, pypy3 >= 3.6 is required."
    )

    parser.add_argument(
        "--python", type=str, default="python3",
                        help="Absolute path of pypy3, pypy3 >= 3.6 is required."
    )

    parser.add_argument(
        '--cpu_threads', type=int, default=4,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument(
        '--gpu_threads', type=int, default=4,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument(
        '--internal_threads', type=int, default=1,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument(
        '--device', type=str, default=None,
                        help="Specify the GPUs for calling default: 'cuda:auto'")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)


if __name__ == "__main__":
    main()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/results/CallVariantsFromCffi.original.py — 원문 전문 (26,510 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/results/CallVariantsFromCffi.original.py", "sha256": "024fe816b279a416d84ab487b6948ad2899088fcca607bb431264e0e89a96ace", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys
import os
import copy
import csv
import torch
import logging
from time import time
import numpy as np

from argparse import ArgumentParser, SUPPRESS
from concurrent.futures import ProcessPoolExecutor, as_completed

from shared.utils import str2bool, log_error, log_warning, str_none, get_header
from shared.interval_tree import bed_tree_from
from clair3.CallVariants import OutputConfig, output_utilties_from, batch_output

logging.basicConfig(format='%(message)s', level=logging.INFO)

def _load_torch_checkpoint(model, checkpoint_path, device):
    #add .pt extension if not present
    if not checkpoint_path.endswith('.pt'):
        checkpoint_path = checkpoint_path + '.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def _select_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _limit_gpu_memory(memory_mb, device):
    if device.type != "cuda":
        return
    total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
    fraction = min(1.0, memory_mb / total_mb)
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    except Exception:
        pass


def _torch_predict(model, device, X):
    with torch.inference_mode():
        X_tensor = torch.from_numpy(X).to(device)
        Y = model(X_tensor)
    return Y.detach().cpu().numpy()


def Run(args):
    threads = args.threads
    os.environ["OMP_NUM_THREADS"] = f"{threads}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{threads}"
    os.environ["MKL_NUM_THREADS"] = f"{threads}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{threads}"

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(2 if threads > 1 else 1)

    global test_pos
    test_pos = None
    global param
    if args.pileup:
        import shared.param_p as param
    else:
        import shared.param_f as param

    if args.enable_long_indel:
        maximum_variant_length_that_need_infer = param.maximum_variant_length_that_need_infer_include_long_indel
    else:
        maximum_variant_length_that_need_infer = param.maximum_variant_length_that_need_infer

    output_config = OutputConfig(
        is_show_reference=args.showRef,
        is_debug=args.debug,
        is_haploid_precise_mode_enabled=args.haploid_precise,
        is_haploid_sensitive_mode_enabled=args.haploid_sensitive,
        is_output_for_ensemble=args.output_for_ensemble,
        quality_score_for_pass=args.qual,
        tensor_fn=args.tensor_fn,
        input_probabilities=args.input_probabilities,
        add_indel_length=args.add_indel_length,
        gvcf=args.gvcf,
        pileup=args.pileup,
        enable_long_indel=args.enable_long_indel,
        maximum_variant_length_that_need_infer=maximum_variant_length_that_need_infer,
        keep_iupac_bases=args.keep_iupac_bases
    )
    output_utilities = output_utilties_from(
        sample_name=args.sampleName,
        is_debug=args.debug,
        is_output_for_ensemble=args.output_for_ensemble,
        reference_file_path=args.ref_fn,
        output_file_path=args.call_fn,
        output_probabilities=args.output_probabilities,
        cmd_fn=args.cmd_fn
    )

    call_variants_from_cffi(args=args, output_config=output_config, output_utilities=output_utilities)

def tensor_generator_for_chunk(gen_cls, args, batch_size=50):
    if args.output_tensor_can_fn_list is not None:
        f_list = open(args.output_tensor_can_fn_list, 'r').read()
        f_list = f_list.strip().split('\n')
        import numpy as np
        for f in f_list:
            if f == '': continue
            parent_dir = os.path.dirname(args.output_tensor_can_fn_list)
            tensor = np.load(os.path.join(parent_dir, f +'.npy'))
            pos_alt = os.path.join(parent_dir, f + '.info')
            pos_alt_info = open(pos_alt, 'r').read().strip().split('\n')
            all_alt_info = []
            all_position = []
            for pos_alt in pos_alt_info:
                pos_alt = pos_alt.split('\t')
                all_position.append(pos_alt[0])
                all_alt_info.append(pos_alt[1])
            total_data = len(tensor)
            assert total_data == len(all_alt_info)
            assert total_data == len(all_position)
            batch_size = batch_size
            total_chunk = total_data // batch_size if total_data % batch_size == 0 else total_data // batch_size + 1
            for chunk_id in range(total_chunk):
                chunk_start = chunk_id * batch_size
                chunk_end = (chunk_id + 1) * batch_size if chunk_id < total_chunk - 1 else total_data
                yield (tensor[chunk_start:chunk_end],
                       all_position[chunk_start:chunk_end],
                       all_alt_info[chunk_start:chunk_end])
    else:
        tensor, all_position, all_alt_info = gen_cls(args)
        if tensor is None:
            return
        total_data = len(tensor)
        assert total_data == len(all_alt_info)
        assert total_data == len(all_position)
        batch_size = batch_size
        total_chunk = total_data // batch_size if total_data % batch_size == 0 else total_data // batch_size + 1
        for chunk_id in range(total_chunk):
            chunk_start = chunk_id * batch_size
            chunk_end = (chunk_id + 1) * batch_size if chunk_id < total_chunk - 1 else total_data
            yield (tensor[chunk_start:chunk_end],
                   all_position[chunk_start:chunk_end],
                   all_alt_info[chunk_start:chunk_end])

def tensor_generator_for_unchunked(gen_cls, args, batch_size=200):
    if args.bed_fn:
        tree = bed_tree_from(args.bed_fn)
        for contig in tree.values():
            contig.merge_overlaps()
        chunks = [(ctg_name, I.begin, I.end) for ctg_name, ctg_tree in tree.items() for I in ctg_tree.all_intervals]
    else:
        fai = f"{args.ref_fn}.fai"
        chunks = [(ctg, 0, ctg_len) for ctg, ctg_len, *_ in csv.reader(open(fai), delimiter='\t')]
    for ctg, start, end in chunks:
        temp_args = copy.copy(args)
        temp_args.ctgName = ctg
        temp_args.ctgStart = int(start)
        temp_args.ctgEnd = int(end)
        for batch in tensor_generator_for_chunk(gen_cls, temp_args, batch_size=batch_size):
            yield batch




def batch_output_worker(position, alt_info_list, shm_name, shape, dtype,
                        output_config, output_utilities):
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        Y = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()

        result = batch_output(position, alt_info_list, Y, output_config, output_utilities)

        return result
    finally:
        if 'shm' in locals():
            shm.close()
            shm.unlink()

def call_variants_from_cffi(args, output_config, output_utilities):
    if args.enable_dwell_time and args.pileup:
        sys.exit(log_error("[ERROR] --enable_dwell_time is only supported in full-alignment mode"))

    if args.pileup:
        from preprocess.CreateTensorPileupFromCffi import CreateTensorPileup as CT
    else:
        if args.enable_dwell_time:
            from preprocess.CreateTensorFullAlignmentFromCffiWithDwell import CreateTensorFullAlignment as CT
        else:
            from preprocess.CreateTensorFullAlignmentFromCffi import CreateTensorFullAlignment as CT

    if args.output_tensor_can_fn_list is None and args.use_gpu:
        CT(args)
        return

    use_triton_gpu = args.use_triton_gpu
    triton_client = None
    device = torch.device("cpu")
    if use_triton_gpu:
        import tritonclient.grpc as tritongrpcclient
        server_url = 'localhost:8001'
        try:
            triton_client = tritongrpcclient.InferenceServerClient(
                url=server_url,
                verbose=False
            )
        except Exception as e:
            print("channel creation failed: " + str(e), file=sys.stderr)
            sys.exit()
    elif args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.gpu_id else ""
        device = _select_device(True)
        _limit_gpu_memory(args.max_gpu_memory, device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = _select_device(False)

    global param
    if args.pileup:
        import shared.param_p as param
        if use_triton_gpu:
            model_name = 'pileup'
            input_dtype = 'INT32'
        else:
            from clair3.model import Clair3_P
            input_channels = (param.ont_input_shape if args.platform == 'ont' else param.input_shape)[-1]
            m = Clair3_P(add_indel_length=args.add_indel_length, predict=True, input_channels=input_channels)
    else:
        import shared.param_f as param
        if use_triton_gpu:
            model_name = 'alignment'
            input_dtype = 'INT8'
        else:
            from clair3.model import Clair3_F
            input_channels = (param.ont_input_shape if args.platform == 'ont' else param.input_shape)[-1]
            if args.enable_dwell_time:
                input_channels += 1
            m = Clair3_F(add_indel_length=args.add_indel_length, predict=True, input_channels=input_channels)

    if not use_triton_gpu:
        m.to(device)
        m.eval()
        _load_torch_checkpoint(m, args.chkpnt_fn, device)
    args.output_file = open(args.call_fn, 'w') if args.call_fn != 'PIPE' else sys.stdout

    header_str = get_header(reference_file_path=args.ref_fn, cmd_fn=args.cmd_fn, sample_name=args.sampleName)
    header_str += '\n'
    args.output_file.write(header_str)

    chunk_id = args.chunk_id - 1 if args.chunk_id else None  # 1-base to 0-base
    chunk_num = args.chunk_num
    full_alignment_mode = not args.pileup

    logging.info("Calling variants ...")
    variant_call_start_time = time()

    batch_output_method = batch_output
    total = 0

    if not args.use_gpu:
        batch_size = param.predictBatchSize * args.threads
    else:
        # increase bs in gpu for speed up
        batch_size = param.predictBatchSize * 5
    if args.ctgName == 'None' and args.pileup:
        batch_gen = tensor_generator_for_unchunked(CT, args, batch_size=batch_size)
    else:
        batch_gen = tensor_generator_for_chunk(CT, args, batch_size=batch_size)

    if not args.use_gpu:
        for (X, position, alt_info_list) in batch_gen:
            total += len(X)
            if args.pileup:
                for alt_idx, alt_info in enumerate(alt_info_list):
                    depth = int(alt_info.split('-', maxsplit=1)[0])
                    max_depth = param.max_depth_dict[args.platform]
                    # for extreme high coverage data, make sure we could have a truncated coverage
                    if depth > 0 and depth > max_depth * 1.5:
                        scale_factor = depth / max_depth
                        X[alt_idx] = X[alt_idx] / scale_factor

            if use_triton_gpu:
                inputs = [];
                outputs = []
                inputs.append(tritongrpcclient.InferInput('input_1', X.shape, input_dtype))
                outputs.append(tritongrpcclient.InferRequestedOutput('output_1'))
                inputs[0].set_data_from_numpy(X)
                results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
                Y = results.as_numpy('output_1')
            else:
                Y = _torch_predict(m, device, X)

            output = batch_output_method(position, alt_info_list, Y, output_config, output_utilities, args)
    else:
        from multiprocessing import shared_memory
        max_pending = args.cpu_threads * 2  # Limit concurrent tasks to bound memory
        with ProcessPoolExecutor(max_workers=args.cpu_threads) as executor:
            pending_futures = {}  # future -> shm_name for cleanup tracking
            batch_iter = iter(batch_gen)
            finished = False

            while pending_futures or not finished:
                # Submit new tasks up to max_pending limit
                while len(pending_futures) < max_pending and not finished:
                    try:
                        X, position, alt_info_list = next(batch_iter)
                    except StopIteration:
                        finished = True
                        break

                    total += len(X)
                    Y = _torch_predict(m, device, X)
                    shm = shared_memory.SharedMemory(create=True, size=Y.nbytes)
                    shared_Y = np.ndarray(Y.shape, dtype=Y.dtype, buffer=shm.buf)
                    np.copyto(shared_Y, Y)

                    future = executor.submit(
                        batch_output_worker,
                        position, alt_info_list,
                        shm.name, Y.shape, Y.dtype,
                        output_config, output_utilities
                    )
                    pending_futures[future] = shm.name
                    shm.close()

                if not pending_futures:
                    break

                # Wait for at least one task to complete
                done_futures = []
                for future in as_completed(pending_futures.keys()):
                    done_futures.append(future)
                    break  # Process one at a time to submit new tasks promptly

                for future in done_futures:
                    shm_name = pending_futures.pop(future)
                    try:
                        result = future.result()
                        args.output_file.write(result)
                    except Exception as e:
                        print(f"[Error] Task failed: {e}", flush=True, file=sys.stderr)
                        # Cleanup shared memory if worker failed before unlink
                        try:
                            orphan_shm = shared_memory.SharedMemory(name=shm_name)
                            orphan_shm.close()
                            orphan_shm.unlink()
                        except FileNotFoundError:
                            pass  # Already cleaned up by worker

    if chunk_id is not None:
        logging.info("Total processed positions in {} (chunk {}/{}) : {}".format(args.ctgName, chunk_id+1, chunk_num, total))
    elif full_alignment_mode:
        try:
            chunk_infos = args.call_fn.split('.')[-2]
            c_id, c_num = chunk_infos.split('_')
            c_id = int(c_id) + 1 # 0-index to 1-index
            logging.info("Total processed positions in {} (chunk {}/{}) : {}".format(args.ctgName, c_id, c_num, total))
        except:
            logging.info("Total processed positions {}: {}".format(args.ctgName if args.ctgName else "", total))
    else:
        logging.info("Total processed positions {}: {}".format(args.ctgName if args.ctgName else "", total))

    if full_alignment_mode and total == 0:
        logging.info(log_warning("[WARNING] No full-alignment output for file {}/{}".format(args.ctgName, args.call_fn)))

    logging.info("Total time elapsed: %.2f s" % (time() - variant_call_start_time))

    args.output_file.close()
    # remove file if on variant in output
    if os.path.exists(args.call_fn):
        for row in open(args.call_fn, 'r'):
            if row[0] != '#':
                return
        logging.info("[INFO] No vcf output for file {}, remove empty file".format(args.call_fn))
        os.remove(args.call_fn)


def main():
    parser = ArgumentParser(description="Call variants using a trained model and tensors of candidate variants")

    parser.add_argument('--platform', type=str, default="ont",
                        help="Sequencing platform of the input. Options: 'ont,hifi,ilmn', default: %(default)s")

    parser.add_argument('--tensor_fn', type=str, default="PIPE",
                        help="Tensor input filename, or stdin if not set")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a trained model for variant calling, required")

    parser.add_argument('--call_fn', type=str, default="clair3",
                        help="VCF output filename, or stdout if not set")

    parser.add_argument('--gvcf', type=str2bool, default=False,
                        help="Enable GVCF output, default: disabled")

    parser.add_argument('--ref_fn', type=str, default=None,
                        help="Reference fasta file input, required if --gvcf is enabled")

    parser.add_argument('--ctgName', type=str, default=None,
                        help="The name of the sequence to be processed")

    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-based starting position of the sequence to be processed, optional, will process the whole --ctgName if not set")

    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The 1-based inclusive ending position of the sequence to be processed, optional, will process the whole --ctgName if not set")

    parser.add_argument('--sampleName', type=str, default="SAMPLE",
                        help="Define the sample name to be shown in the VCF file, optional")

    parser.add_argument('--qual', type=int, default=2,
                        help="If set, variants with >=$qual will be marked 'PASS', or 'LowQual' otherwise, optional")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', samtools version >= 1.10 is required, default: %(default)s")

    # options for advanced users
    parser.add_argument('--temp_file_dir', type=str, default='./',
                        help="EXPERIMENTAL: The cache directory for storing temporary non-variant information if --gvcf is enabled, default: %(default)s")

    parser.add_argument('--haploid_precise', action='store_true',
                        help="EXPERIMENTAL: Enable haploid calling mode. Only 1/1 is considered as a variant")

    parser.add_argument('--haploid_sensitive', action='store_true',
                        help="EXPERIMENTAL: Enable haploid calling mode. 0/1 and 1/1 are considered as a variant")

    parser.add_argument('--enable_long_indel', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable long Indel variants(>50 bp) calling")

    parser.add_argument('--keep_iupac_bases', type=str2bool, default=False,
                        help="EXPERIMENTAL: Keep IUPAC (non ACGTN) reference and alternate bases, default: convert all IUPAC bases to N")

    # options for debug purpose
    parser.add_argument('--use_gpu', type=str2bool, default=False,
                        help="DEBUG: Use GPU for calling. Speed up is mostly insignficiant. Only use this for building your own pipeline")

    parser.add_argument('--threads', type=int, default=1,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument('--predict_fn', type=str, default=None,
                        help="DEBUG: Output network output probabilities for further analysis")

    parser.add_argument('--input_probabilities', action='store_true',
                        help="DEBUG: Use network probability outputs as input and generate variants from them")

    parser.add_argument('--output_probabilities', action='store_true',
                        help="DEBUG: Output the network probabilities of gt21, genotype, indel_length_1 and indel_length_2")

    parser.add_argument('--enable_variant_calling_at_sequence_head_and_tail', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable variant calling in sequence head and tail start or end regions that flanking 16bp windows having no read support. Default: disable.")

    # options for internal process control
    ## In pileup mode or not (full alignment mode), default: False
    parser.add_argument('--pileup', action='store_true',
                        help=SUPPRESS)

    ## Include indel length in training and calling, false for pileup and true for raw alignment
    parser.add_argument('--add_indel_length', action='store_true',
                        help=SUPPRESS)

    ## The number of chucks to be divided into for parallel processing
    parser.add_argument('--chunk_num', type=int, default=None,
                        help=SUPPRESS)

    ## The chuck ID to work on
    parser.add_argument('--chunk_id', type=int, default=None,
                        help=SUPPRESS)

    ## Enable debug mode, default: False, optional
    parser.add_argument('--debug', action='store_true',
                        help=SUPPRESS)

    ## Generating outputs for ensemble model calling
    parser.add_argument('--output_for_ensemble', action='store_true',
                        help=SUPPRESS)

    ## Use bin file from HDF5 to speed up calling.
    parser.add_argument('--is_from_tables', action='store_true',
                        help=SUPPRESS)

    ## Output reference calls
    parser.add_argument('--showRef', type=str2bool, default=True,
                        help=SUPPRESS)

    # Pileup create tensor options for pileup calling
    parser.add_argument('--bam_fn', type=str, default="input.bam", required=True,
                        help="Sorted BAM file input, required")

    parser.add_argument('--bed_fn', type=str, nargs='?', action="store", default=None,
                        help="Call variant only in the provided regions. Will take an intersection if --ctgName and/or (--ctgStart, --ctgEnd) are set")

    parser.add_argument('--snp_min_af', type=float, default=0.08,
                        help="Minimum snp allele frequency for a site to be considered as a candidate site, default: %(default)f")

    parser.add_argument('--indel_min_af', type=float, default=0.15,
                        help="Minimum indel allele frequency for a site to be considered as a candidate site, default: %(default)f")

    parser.add_argument('--extend_bed', nargs='?', action="store", type=str, default=None,
                        help="DEBUG: Extend the regions in the --bed_fn by a few bp for tensor creation, default extend 16bp")

    parser.add_argument('--vcf_fn', type=str, default=None,
                        help="Candidate sites VCF file input, if provided, variants will only be called at the sites in the VCF file,  default: %(default)s")

    parser.add_argument('--minCoverage', type=int, default=2,
                        help="EXPERIMENTAL: Minimum coverage required to call a variant, default: %(default)f")

    parser.add_argument('--minMQ', type=int, default=5,
                        help="EXPERIMENTAL: If set, reads with mapping quality with <$minMQ are filtered, default: %(default)d")

    parser.add_argument('--minBQ', type=int, default=0,
                        help="EXPERIMENTAL: If set, bases with base quality with <$minBQ are filtered, default: %(default)d")

    parser.add_argument('--max_depth', type=int, default=144,
                        help="EXPERIMENTAL: Maximum full alignment depth to be processed. default: %(default)s")

    parser.add_argument('--fast_mode', type=str2bool, default=False,
                        help="EXPERIMENTAL: Skip variant candidates with AF <= 0.15, default: %(default)s")

    parser.add_argument('--call_snp_only', type=str2bool, default=False,
                        help="EXPERIMENTAL: Call candidates pass snp minimum AF only, ignore Indel candidates")

    parser.add_argument('--base_err', default=0.001, type=float,
                        help='DEBUG: Estimated base error rate in gvcf option, default: %(default)f')

    parser.add_argument('--gq_bin_size', default=5, type=int,
                        help='DEBUG: Default gq bin size for merge non-variant block in gvcf option, default: %(default)d')

    parser.add_argument('--bp_resolution', action='store_true',
                        help="DEBUG: Enable bp resolution for GVCF, default: disabled")

    # Full-alignment create tensor options for full-alignment calling
    parser.add_argument('--phased_vcf_fn', type=str, default=None,
                        help="Use heterozygous SNP variants in phased vcf file for haplotaging")

    parser.add_argument('--no_phasing_for_fa', type=str2bool, default=False,
                        help="EXPERIMENTAL: Call variants without whatshap or longphase phasing in full alignment calling")

    parser.add_argument('--enable_dwell_time', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable dwell time channel when generating full-alignment tensors")

    ## Provide the regions to be included in full-alignment based calling
    parser.add_argument('--full_aln_regions', type=str, default=None,
                        help=SUPPRESS)

    ## If defined, added command line into VCF header
    parser.add_argument('--cmd_fn', type=str_none, default=None,
                        help=SUPPRESS)

    parser.add_argument('--tensor_can_fn', type=str, default="PIPE",
                        help="Tensor output, stdout by default, default: %(default)s")

    parser.add_argument('--gpu_id', type=str, default=None,
                        help="Tensor output, stdout by default, default: %(default)s")

    parser.add_argument('--max_gpu_memory', type=int, default=5000,
                        help="Tensor output, stdout by default, default: %(default)s")

    parser.add_argument('--use_triton_gpu', type=str2bool, default=False,
                        help="DEBUG: Use GPU for calling. Speed up is mostly insignficiant. Only use this for building your own pipeline")

    parser.add_argument('--output_tensor_can_fn_list', type=str, default=None,
                        help="Tensor output, stdout by default, default: %(default)s")

    parser.add_argument('--cpu_threads', type=int, default=4,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument('--output_file', type=str, default=None,
                        help="How many threads for python to use for parallelization default: %(default)f")


    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)


if __name__ == "__main__":
    main()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/gpu_support.py — 원문 전문 (2,858 bytes)</summary>


<!-- RESTORE_FILE {"path": "scripts/gpu_support.py", "sha256": "349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2", "encoding": "utf-8", "added_final_newline": false} -->
```python
import time,json,csv,threading,subprocess,os
from pathlib import Path
import numpy as np,psutil,torch
G=2**30
class Meter:
 def __init__(self,root):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.csv_path=self.root/'resources.csv';
  if self.csv_path.exists():self.csv_path=self.root/('resources_'+str(time.time_ns())+'.csv')
  self.phase='initial';self.rows=[];self.stop=threading.Event();self.swap0=psutil.swap_memory().used;self.th=threading.Thread(target=self.monitor,daemon=True);self.th.start()
 def memory(self):
  v=psutil.virtual_memory();return dict(allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,peak_allocated_gib=torch.cuda.max_memory_allocated()/G,peak_reserved_gib=torch.cuda.max_memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,system_available_gib=v.available/G,system_total_gib=v.total/G)
 def monitor(self):
  with self.csv_path.open('w',newline='') as f:
   keys=['query_start','query_end','phase','gpu_util','allocated_gib','reserved_gib','rss_gib','available_gib','swap_gib'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
   while not self.stop.is_set():
    t=time.time();phase=self.phase
    try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).strip().splitlines()[0])
    except Exception:u=None
    v=psutil.virtual_memory();r=dict(query_start=t,query_end=time.time(),phase=phase,gpu_util=u,allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,available_gib=v.available/G,swap_gib=psutil.swap_memory().used/G);self.rows.append(r);w.writerow(r);f.flush()
    if v.available<8*G or psutil.swap_memory().used>self.swap0+2*G:(self.root/'memory_guard.json').write_text(json.dumps(r));os._exit(86)
    self.stop.wait(.5)
 def stats(self,t,end):
  rows=[r for r in self.rows if r['query_start']>=t and r['query_end']<=end];u=[r['gpu_util'] for r in rows if r['gpu_util'] is not None];return dict(samples=len(u),gpu_util_avg=float(np.mean(u)) if u else None,gpu_util_max=max(u) if u else None,rss_sampled_peak_gib=max((r['rss_gib'] for r in rows),default=None),available_sampled_min_gib=min((r['available_gib'] for r in rows),default=None))
 def begin(self,name):
  self.phase=name;torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();return time.time(),self.memory()
 def end(self,t,base,**kw):
  torch.cuda.synchronize();end=time.time();m=self.memory();return dict(**kw,seconds=end-t,baseline=base,memory=m,peak_delta_gib=m['peak_allocated_gib']-base['allocated_gib'],**self.stats(t,end))
 def close(self):self.stop.set();self.th.join(timeout=5)
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))
```
<!-- END_RESTORE_FILE -->


</details>

### record-31b34816e116

원래 파일: `clair3/data/input_audit.json` · 원문 SHA256: `c19b4ddf8961699e445671cc09bef83fbd18d3294d28ed4b26c1c62ee56d68be`

<details>
<summary>clair3/data/input_audit.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/data/input_audit.json", "sha256": "c19b4ddf8961699e445671cc09bef83fbd18d3294d28ed4b26c1c62ee56d68be", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "bam_header": {
    "HD": {
      "VN": "1.6",
      "SO": "coordinate"
    },
    "SQ": [
      {
        "SN": "chr1",
        "LN": 248956422
      },
      {
        "SN": "chr2",
        "LN": 242193529
      },
      {
        "SN": "chr3",
        "LN": 198295559
      },
      {
        "SN": "chr4",
        "LN": 190214555
      },
      {
        "SN": "chr5",
        "LN": 181538259
      },
      {
        "SN": "chr6",
        "LN": 170805979
      },
      {
        "SN": "chr7",
        "LN": 159345973
      },
      {
        "SN": "chr8",
        "LN": 145138636
      },
      {
        "SN": "chr9",
        "LN": 138394717
      },
      {
        "SN": "chr10",
        "LN": 133797422
      },
      {
        "SN": "chr11",
        "LN": 135086622
      },
      {
        "SN": "chr12",
        "LN": 133275309
      },
      {
        "SN": "chr13",
        "LN": 114364328
      },
      {
        "SN": "chr14",
        "LN": 107043718
      },
      {
        "SN": "chr15",
        "LN": 101991189
      },
      {
        "SN": "chr16",
        "LN": 90338345
      },
      {
        "SN": "chr17",
        "LN": 83257441
      },
      {
        "SN": "chr18",
        "LN": 80373285
      },
      {
        "SN": "chr19",
        "LN": 58617616
      },
      {
        "SN": "chr20",
        "LN": 64444167
      },
      {
        "SN": "chr21",
        "LN": 46709983
      },
      {
        "SN": "chr22",
        "LN": 50818468
      },
      {
        "SN": "chrX",
        "LN": 156040895
      },
      {
        "SN": "chrY",
        "LN": 57227415
      },
      {
        "SN": "chrM",
        "LN": 16569
      },
      {
        "SN": "chr1_KI270706v1_random",
        "LN": 175055
      },
      {
        "SN": "chr1_KI270707v1_random",
        "LN": 32032
      },
      {
        "SN": "chr1_KI270708v1_random",
        "LN": 127682
      },
      {
        "SN": "chr1_KI270709v1_random",
        "LN": 66860
      },
      {
        "SN": "chr1_KI270710v1_random",
        "LN": 40176
      },
      {
        "SN": "chr1_KI270711v1_random",
        "LN": 42210
      },
      {
        "SN": "chr1_KI270712v1_random",
        "LN": 176043
      },
      {
        "SN": "chr1_KI270713v1_random",
        "LN": 40745
      },
      {
        "SN": "chr1_KI270714v1_random",
        "LN": 41717
      },
      {
        "SN": "chr2_KI270715v1_random",
        "LN": 161471
      },
      {
        "SN": "chr2_KI270716v1_random",
        "LN": 153799
      },
      {
        "SN": "chr3_GL000221v1_random",
        "LN": 155397
      },
      {
        "SN": "chr4_GL000008v2_random",
        "LN": 209709
      },
      {
        "SN": "chr5_GL000208v1_random",
        "LN": 92689
      },
      {
        "SN": "chr9_KI270717v1_random",
        "LN": 40062
      },
      {
        "SN": "chr9_KI270718v1_random",
        "LN": 38054
      },
      {
        "SN": "chr9_KI270719v1_random",
        "LN": 176845
      },
      {
        "SN": "chr9_KI270720v1_random",
        "LN": 39050
      },
      {
        "SN": "chr11_KI270721v1_random",
        "LN": 100316
      },
      {
        "SN": "chr14_GL000009v2_random",
        "LN": 201709
      },
      {
        "SN": "chr14_GL000225v1_random",
        "LN": 211173
      },
      {
        "SN": "chr14_KI270722v1_random",
        "LN": 194050
      },
      {
        "SN": "chr14_GL000194v1_random",
        "LN": 191469
      },
      {
        "SN": "chr14_KI270723v1_random",
        "LN": 38115
      },
      {
        "SN": "chr14_KI270724v1_random",
        "LN": 39555
      },
      {
        "SN": "chr14_KI270725v1_random",
        "LN": 172810
      },
      {
        "SN": "chr14_KI270726v1_random",
        "LN": 43739
      },
      {
        "SN": "chr15_KI270727v1_random",
        "LN": 448248
      },
      {
        "SN": "chr16_KI270728v1_random",
        "LN": 1872759
      },
      {
        "SN": "chr17_GL000205v2_random",
        "LN": 185591
      },
      {
        "SN": "chr17_KI270729v1_random",
        "LN": 280839
      },
      {
        "SN": "chr17_KI270730v1_random",
        "LN": 112551
      },
      {
        "SN": "chr22_KI270731v1_random",
        "LN": 150754
      },
      {
        "SN": "chr22_KI270732v1_random",
        "LN": 41543
      },
      {
        "SN": "chr22_KI270733v1_random",
        "LN": 179772
      },
      {
        "SN": "chr22_KI270734v1_random",
        "LN": 165050
      },
      {
        "SN": "chr22_KI270735v1_random",
        "LN": 42811
      },
      {
        "SN": "chr22_KI270736v1_random",
        "LN": 181920
      },
      {
        "SN": "chr22_KI270737v1_random",
        "LN": 103838
      },
      {
        "SN": "chr22_KI270738v1_random",
        "LN": 99375
      },
      {
        "SN": "chr22_KI270739v1_random",
        "LN": 73985
      },
      {
        "SN": "chrY_KI270740v1_random",
        "LN": 37240
      },
      {
        "SN": "chrUn_KI270302v1",
        "LN": 2274
      },
      {
        "SN": "chrUn_KI270304v1",
        "LN": 2165
      },
      {
        "SN": "chrUn_KI270303v1",
        "LN": 1942
      },
      {
        "SN": "chrUn_KI270305v1",
        "LN": 1472
      },
      {
        "SN": "chrUn_KI270322v1",
        "LN": 21476
      },
      {
        "SN": "chrUn_KI270320v1",
        "LN": 4416
      },
      {
        "SN": "chrUn_KI270310v1",
        "LN": 1201
      },
      {
        "SN": "chrUn_KI270316v1",
        "LN": 1444
      },
      {
        "SN": "chrUn_KI270315v1",
        "LN": 2276
      },
      {
        "SN": "chrUn_KI270312v1",
        "LN": 998
      },
      {
        "SN": "chrUn_KI270311v1",
        "LN": 12399
      },
      {
        "SN": "chrUn_KI270317v1",
        "LN": 37690
      },
      {
        "SN": "chrUn_KI270412v1",
        "LN": 1179
      },
      {
        "SN": "chrUn_KI270411v1",
        "LN": 2646
      },
      {
        "SN": "chrUn_KI270414v1",
        "LN": 2489
      },
      {
        "SN": "chrUn_KI270419v1",
        "LN": 1029
      },
      {
        "SN": "chrUn_KI270418v1",
        "LN": 2145
      },
      {
        "SN": "chrUn_KI270420v1",
        "LN": 2321
      },
      {
        "SN": "chrUn_KI270424v1",
        "LN": 2140
      },
      {
        "SN": "chrUn_KI270417v1",
        "LN": 2043
      },
      {
        "SN": "chrUn_KI270422v1",
        "LN": 1445
      },
      {
        "SN": "chrUn_KI270423v1",
        "LN": 981
      },
      {
        "SN": "chrUn_KI270425v1",
        "LN": 1884
      },
      {
        "SN": "chrUn_KI270429v1",
        "LN": 1361
      },
      {
        "SN": "chrUn_KI270442v1",
        "LN": 392061
      },
      {
        "SN": "chrUn_KI270466v1",
        "LN": 1233
      },
      {
        "SN": "chrUn_KI270465v1",
        "LN": 1774
      },
      {
        "SN": "chrUn_KI270467v1",
        "LN": 3920
      },
      {
        "SN": "chrUn_KI270435v1",
        "LN": 92983
      },
      {
        "SN": "chrUn_KI270438v1",
        "LN": 112505
      },
      {
        "SN": "chrUn_KI270468v1",
        "LN": 4055
      },
      {
        "SN": "chrUn_KI270510v1",
        "LN": 2415
      },
      {
        "SN": "chrUn_KI270509v1",
        "LN": 2318
      },
      {
        "SN": "chrUn_KI270518v1",
        "LN": 2186
      },
      {
        "SN": "chrUn_KI270508v1",
        "LN": 1951
      },
      {
        "SN": "chrUn_KI270516v1",
        "LN": 1300
      },
      {
        "SN": "chrUn_KI270512v1",
        "LN": 22689
      },
      {
        "SN": "chrUn_KI270519v1",
        "LN": 138126
      },
      {
        "SN": "chrUn_KI270522v1",
        "LN": 5674
      },
      {
        "SN": "chrUn_KI270511v1",
        "LN": 8127
      },
      {
        "SN": "chrUn_KI270515v1",
        "LN": 6361
      },
      {
        "SN": "chrUn_KI270507v1",
        "LN": 5353
      },
      {
        "SN": "chrUn_KI270517v1",
        "LN": 3253
      },
      {
        "SN": "chrUn_KI270529v1",
        "LN": 1899
      },
      {
        "SN": "chrUn_KI270528v1",
        "LN": 2983
      },
      {
        "SN": "chrUn_KI270530v1",
        "LN": 2168
      },
      {
        "SN": "chrUn_KI270539v1",
        "LN": 993
      },
      {
        "SN": "chrUn_KI270538v1",
        "LN": 91309
      },
      {
        "SN": "chrUn_KI270544v1",
        "LN": 1202
      },
      {
        "SN": "chrUn_KI270548v1",
        "LN": 1599
      },
      {
        "SN": "chrUn_KI270583v1",
        "LN": 1400
      },
      {
        "SN": "chrUn_KI270587v1",
        "LN": 2969
      },
      {
        "SN": "chrUn_KI270580v1",
        "LN": 1553
      },
      {
        "SN": "chrUn_KI270581v1",
        "LN": 7046
      },
      {
        "SN": "chrUn_KI270579v1",
        "LN": 31033
      },
      {
        "SN": "chrUn_KI270589v1",
        "LN": 44474
      },
      {
        "SN": "chrUn_KI270590v1",
        "LN": 4685
      },
      {
        "SN": "chrUn_KI270584v1",
        "LN": 4513
      },
      {
        "SN": "chrUn_KI270582v1",
        "LN": 6504
      },
      {
        "SN": "chrUn_KI270588v1",
        "LN": 6158
      },
      {
        "SN": "chrUn_KI270593v1",
        "LN": 3041
      },
      {
        "SN": "chrUn_KI270591v1",
        "LN": 5796
      },
      {
        "SN": "chrUn_KI270330v1",
        "LN": 1652
      },
      {
        "SN": "chrUn_KI270329v1",
        "LN": 1040
      },
      {
        "SN": "chrUn_KI270334v1",
        "LN": 1368
      },
      {
        "SN": "chrUn_KI270333v1",
        "LN": 2699
      },
      {
        "SN": "chrUn_KI270335v1",
        "LN": 1048
      },
      {
        "SN": "chrUn_KI270338v1",
        "LN": 1428
      },
      {
        "SN": "chrUn_KI270340v1",
        "LN": 1428
      },
      {
        "SN": "chrUn_KI270336v1",
        "LN": 1026
      },
      {
        "SN": "chrUn_KI270337v1",
        "LN": 1121
      },
      {
        "SN": "chrUn_KI270363v1",
        "LN": 1803
      },
      {
        "SN": "chrUn_KI270364v1",
        "LN": 2855
      },
      {
        "SN": "chrUn_KI270362v1",
        "LN": 3530
      },
      {
        "SN": "chrUn_KI270366v1",
        "LN": 8320
      },
      {
        "SN": "chrUn_KI270378v1",
        "LN": 1048
      },
      {
        "SN": "chrUn_KI270379v1",
        "LN": 1045
      },
      {
        "SN": "chrUn_KI270389v1",
        "LN": 1298
      },
      {
        "SN": "chrUn_KI270390v1",
        "LN": 2387
      },
      {
        "SN": "chrUn_KI270387v1",
        "LN": 1537
      },
      {
        "SN": "chrUn_KI270395v1",
        "LN": 1143
      },
      {
        "SN": "chrUn_KI270396v1",
        "LN": 1880
      },
      {
        "SN": "chrUn_KI270388v1",
        "LN": 1216
      },
      {
        "SN": "chrUn_KI270394v1",
        "LN": 970
      },
      {
        "SN": "chrUn_KI270386v1",
        "LN": 1788
      },
      {
        "SN": "chrUn_KI270391v1",
        "LN": 1484
      },
      {
        "SN": "chrUn_KI270383v1",
        "LN": 1750
      },
      {
        "SN": "chrUn_KI270393v1",
        "LN": 1308
      },
      {
        "SN": "chrUn_KI270384v1",
        "LN": 1658
      },
      {
        "SN": "chrUn_KI270392v1",
        "LN": 971
      },
      {
        "SN": "chrUn_KI270381v1",
        "LN": 1930
      },
      {
        "SN": "chrUn_KI270385v1",
        "LN": 990
      },
      {
        "SN": "chrUn_KI270382v1",
        "LN": 4215
      },
      {
        "SN": "chrUn_KI270376v1",
        "LN": 1136
      },
      {
        "SN": "chrUn_KI270374v1",
        "LN": 2656
      },
      {
        "SN": "chrUn_KI270372v1",
        "LN": 1650
      },
      {
        "SN": "chrUn_KI270373v1",
        "LN": 1451
      },
      {
        "SN": "chrUn_KI270375v1",
        "LN": 2378
      },
      {
        "SN": "chrUn_KI270371v1",
        "LN": 2805
      },
      {
        "SN": "chrUn_KI270448v1",
        "LN": 7992
      },
      {
        "SN": "chrUn_KI270521v1",
        "LN": 7642
      },
      {
        "SN": "chrUn_GL000195v1",
        "LN": 182896
      },
      {
        "SN": "chrUn_GL000219v1",
        "LN": 179198
      },
      {
        "SN": "chrUn_GL000220v1",
        "LN": 161802
      },
      {
        "SN": "chrUn_GL000224v1",
        "LN": 179693
      },
      {
        "SN": "chrUn_KI270741v1",
        "LN": 157432
      },
      {
        "SN": "chrUn_GL000226v1",
        "LN": 15008
      },
      {
        "SN": "chrUn_GL000213v1",
        "LN": 164239
      },
      {
        "SN": "chrUn_KI270743v1",
        "LN": 210658
      },
      {
        "SN": "chrUn_KI270744v1",
        "LN": 168472
      },
      {
        "SN": "chrUn_KI270745v1",
        "LN": 41891
      },
      {
        "SN": "chrUn_KI270746v1",
        "LN": 66486
      },
      {
        "SN": "chrUn_KI270747v1",
        "LN": 198735
      },
      {
        "SN": "chrUn_KI270748v1",
        "LN": 93321
      },
      {
        "SN": "chrUn_KI270749v1",
        "LN": 158759
      },
      {
        "SN": "chrUn_KI270750v1",
        "LN": 148850
      },
      {
        "SN": "chrUn_KI270751v1",
        "LN": 150742
      },
      {
        "SN": "chrUn_KI270752v1",
        "LN": 27745
      },
      {
        "SN": "chrUn_KI270753v1",
        "LN": 62944
      },
      {
        "SN": "chrUn_KI270754v1",
        "LN": 40191
      },
      {
        "SN": "chrUn_KI270755v1",
        "LN": 36723
      },
      {
        "SN": "chrUn_KI270756v1",
        "LN": 79590
      },
      {
        "SN": "chrUn_KI270757v1",
        "LN": 71251
      },
      {
        "SN": "chrUn_GL000214v1",
        "LN": 137718
      },
      {
        "SN": "chrUn_KI270742v1",
        "LN": 186739
      },
      {
        "SN": "chrUn_GL000216v2",
        "LN": 176608
      },
      {
        "SN": "chrUn_GL000218v1",
        "LN": 161147
      },
      {
        "SN": "chrEBV",
        "LN": 171823
      }
    ],
    "PG": [
      {
        "ID": "minimap2",
        "PN": "minimap2",
        "VN": "2.14-r883",
        "CL": "minimap2 -t 40 -aL -z 600,200 -x map-ont /nas4/yswong/Clair_multi/fasta/GRCh38_no_alt_analysis_set.fasta /bal1-1/rbluo/Clairvoyante/pfda2/hg003/HG003_GM24149_1_2_3_Guppy_3.6.0_prom.fastq.gz"
      },
      {
        "ID": "samtools",
        "PN": "samtools",
        "PP": "minimap2",
        "VN": "1.10",
        "CL": "samtools view -@ 12 -bh -o HG003_chr20_demo.bam /mnt/bal36/zxzheng/testData/ont/data/HG003_GM24149_1_2_3_Guppy_3.6.0_prom.bam chr20:100000-300000"
      }
    ]
  },
  "sort_order": "coordinate",
  "reference_length": 64444167,
  "regions": [
    {
      "region_0based_halfopen": [
        "chr20",
        100000,
        150000
      ],
      "length_bp": 50000,
      "reads_overlapping": 381,
      "coverage_mean": 102.97378,
      "coverage_median": 103.0,
      "coverage_min": 93,
      "coverage_max": 122,
      "read_length_mean": 27879.128608923886
    },
    {
      "region_0based_halfopen": [
        "chr20",
        100000,
        200000
      ],
      "length_bp": 100000,
      "reads_overlapping": 682,
      "coverage_mean": 107.1313,
      "coverage_median": 106.0,
      "coverage_min": 93,
      "coverage_max": 126,
      "read_length_mean": 24254.41935483871
    },
    {
      "region_0based_halfopen": [
        "chr20",
        100000,
        300000
      ],
      "length_bp": 200000,
      "reads_overlapping": 1305,
      "coverage_mean": 107.527935,
      "coverage_median": 107.0,
      "coverage_min": 84,
      "coverage_max": 129,
      "read_length_mean": 20632.950957854406
    }
  ],
  "first_read": {
    "name": "dc9c6024-1ac4-499b-87b8-2d7e64475b46",
    "reference_start_0based": 60000,
    "cigar": "22735S14M1D32M1I6M3D4M1D13M1D21M1D103M1I79M1I80M1D54M1D12M1I19M1I11M1I25M1D79M1D6M1D18M2D88M2D6M1D45M1D9M1D46M1D24M1D9M1I59M1I2M1D32M1D38M1D31M1D46M1D45M1D29M1D5M2D67M1D57M2D11M2D52M1I7M1D24M1D129M1D9M1D13M1I9M3D11M1529I31M1D27M3D4M1D19M1D3M1I9M2D23M1D21M3I11M2D12M1D2M1D50M2D27M1D99M1D10M1D84M1D14M1D29M3D33M2D68M2D49M1D19M1D27M2D28M1D26M1D11M1D45M1D22M1D42M2D47M1I6M2D34M1D40M1D39M1I5M1D40M2D9M1I9M3D37M1D4M1D24M3D42M2D5M1D44M1D2M1I5M1I7M1D4M1D15M1I15M1D23M1I1M1I12M1D9M1I47M1I69M1D19M5I35M1D70M5D63M45D590M1I28M1I46M1D2M2D96M1D1M2D4M2D19M1I11M1D50M2D25M2D4M1I6M1D54M2D58M1D1M2D39M1D1M3D1M1D28M7D72M1D71M1D23M3I4M1I5M1I77M1I2M2D15M1I37M1I82M2I7M1D32M1D134M3D48M1D1M1D19M1I8M1D4M1I115M1D52M2D10M1D15M1D24M1D2M1D77M2D4M1D90M1D30M1I39M1D2M2I5M4I34M1I99M2D24M1D49M2I20M1D35M2D73M2D39M1D4M1D4M1D63M2D47M1D11M1D4M1D4M1D38M1D153M1D30M9I146M2D12M1D7M1D3M1I27M1I41M1I2M1I28M2D25M1D40M1D9M1I4M1I59M1D59M1D4M1D33M2D4M1D2M1I42M1D32M1D19M1D4M1I30M1D10M2D4M1D3M2D19M1I18M1D34M1D4M1I5M1D6M1D4M1D3M2D3M2D8M2D14M3D2M1D18M2D2M1D6M1D4M1D37M1D13M1D3M1I38M2D4M1D22M1D81M1D2M1I18M3D5M1D47M1D74M1D18M1D76M1I31M3I17M2D9M1I32M1I54M1D8M2D92M1I8M1D109M1I50M1I32M1D10M3D6M1D13M2D37M1D112M1I14M1I8M1I16M2D120M1D1M2D51M1D5M2I67M2D6M1D53M1D95M3D33M1D41M1D14M3D5M1D47M2D91M1I31M2D13M1D76M1D1M1D63M1D19M1D92M1D7M5D11M1I10M1D11M1I87M1D68M2D6M1D32M1D3M1D8M3I5M1I37M1D58M3I76M2I30M3I9M2I39M1I15M2I189M1D17M2D6M1D4M1I2M2D9M1I55M2I43M2D150M1D28M1D68M2D1M2D47M1D30M1I126M1I13M1D28M1I12M1I50M1D62M2I53M1D43M1D29M1I37M1D9M2D17M2D13M1D6M1D141M1I13M1D11M2D33M3D210M1D7M2D11M2D10M3I11M4D88M1D40M1D52M1I22M1D66M1I52M2I12M3I33M1D110M1D46M1D1M1D2M1D23M1I5M1I43M1I57M1I49M1D172M1D49M1I105M1I89M1D27M1I2M1I83M1D68M1D54M1I19M1D52M1D6M1D18M2D6M2D42M1I49M1D20M1I81M1D146M2D65M1I138M1I2M1I78M2I19M1I45M1D19M2I76M2D17M1I19M1I48M1I14M1D292M1I4M1D5M1D27M1I64M1I29M1D72M1I77M1D78M2D20M1D30M1D10M1I29M4D9M1I47M1D68M1I70M2I2M2D94M1D54M3D125M1D169M1D30M1D151M2D5M2I2M2D30M1I52M1D140M1D41M1D134M1D60M3D4M2D39M1D27M1I66M2D19M1D3M1D18M1I48M1D2M2I11M1D66M1I110M1I69M2I83M2D24M1I103M1D117M1D118M2I28M1D10M2D35M2I21M1I7M1D33M2D92M2D19M2D74M1I108M1D24M1D76M1I98M1I8M1D84M1D97M2D93M2I68M1I3M1D38M1D15M2D21M2D35M2D35M1I142M1D17M2I7M1D35M1D20M1D49M1D34M1D117M1D15M1D52M1D159M1D34M1D173M1I70M1D2M1D93M1D14M1D4M1I53M1I18M1D78M1D22M1D157M1D4M1D33M4D3M2D5M3I5M4D20M2I51M1D10M2I102M1I5M1D14M1I6M2I7M1I6M1D3M1D31M2D38M3D11M1D14M1D13M1D35M1I12M1D72M2I82M1D31M2I10M2D30M1D37M1D29M4D32M1D28M1D2M1I4M4D4M1D16M2D40M2D2M1D2M1D2M1D5M1D13M2D20M1D32M1D146M8I9M1I9M1I7M2I27M1D3M1D56M2D15M1D38M1I15M5D123M1I9M2I12M1I223M1D268M1I22M2D70M1D3M1D65M1D29M1I29M1D10M1D42M1D2M1D6M3D2M1D29M1D33M2I30M1D88M2D47M1I7M1I31M1D3M4D3M3D14M1I20M2D68M1D13M1D3M1D20M3I82M1I15M1D48M3I18M1I39M2D125M1I3M1I2M1I36M1I2M1I1M3I15M2D63M1D10M1D11M4D62M1D36M1D3M1D10M2D47M2D29M2D47M2D2M4D2M3I3M1D18M1D2M1D18M1D8M2I5M1D9M2D4M2D25M1D12M2D48M1I53M2D97M1D33M1D78M1D15M1D24M1D59M1D24M1D19M4D181M1I15M4I25M1I8M1D28M1D102M2D22M1D2M1D10M1D12M6D85M1I81M1D119M1I23M1D11M2I17M5D3M1D4M2D9M1D6M1D7M2D90M1D8M5D33M2I38M1I6M1D17M2D1M1I87M2D29M1I31M1I44M2I14M2D8M2D48M1D75M1I50M1D6M1I47M1D32M1D38M2D33M1D50M2D17M3I6M1D33M1I29M1I64M1I1M1I6M5I15M2D14M2D12M4D4M2D16M1D7M1I96M1D108M1D22M3I14M2D22M1D19M1D3M1I47M2D29M2D20M3D82M1D15M1D4M4D33M2D80M2I70M1D24M1I21M3I50M1I4M1I5M1I35M1D73M1D58M1D5M3I114M1D4M1D18M1I37M2D18M1I25M2D34M2I51M1I35M2D4M1I9M1D7M1I8M2D24M2I9M1I33M2D4M1I1M2I33M1D56M1I3M1I5M1D38M2I16M1D34M1I23M1D56M1I76M1I32M3D29M4D4M1D7M1D10M10I68M1I32M1D90M1I48M2D21M1D5M1D7M1D2M3D23M1I14M1D40M1I58M3D21M1I85M3D138M1D71M7D4M1I5M1D6M2I11M2I29M2I3M2I26M3D5M1D11M1I17M2I52M3D4M1I60M3D50M1D6M1I85M1D5M1I46M1D61M1I61M1D5M1D59M3D20M2D138M1D40M2I120M1D52M1D133M1D24M1I43M1D21M2I6M1D24M6I128M3D13M4D87M1D10M1D10M2D4M1D14M1D1M1D27M1D12M1D144M1I1M1I29M1D28M1I6M1D8M3D4M1I69M1I33M1D36M1I2M1I11M1I53M1I83M1D13M1I9M2D10M2D16M1D7M2D28M1I2M2D4M1D51M1I69M1D88M2D71M2D59M1I56M2I21M1D131M1D31M3D6M1I96M4D46M1I29M1D5M1D73M1I150M1I40M1I77M1D115M3D9M1D43M1I46M1D37M1D10M1I89M3I27M1D4M1D51M1D52M1D54M1D113M1D29M1I131M1D5M1D43M3D41M3I34M1D30M1I89M4D13M1I4M1D8M1I85M1D130M1I15M1D40M1I15M1D116M4D36M1D21M1D101M1D37M1D77M1I8M1I46M3I4M1D6M1D63M1D62M1I46M2D14M2D13M3D173M1D47M1D72M3I43M1D100M1D4M1D32M1I33M1D86M2D52M4I64M1D1M1D108M1D36M3D18M1D53M4D38M1I55M2D16M1D5M1D49M1D34M1D54M1D46M1I27M2I8M2D126M1D24M1D129M1D61M2D57M1D25M2D21M2D15M1I44M1D84M3D4M2I25M3D1M1I2M1I3M1I26M1D4M1I145M1I28M1D43M2I14M1D12M1D85M2I10M1I43M2I96M1I2M2D20M3I28M1D33M1D49M1I12M2D13M1D36M1I95M1I24M1D28M1I51M1D27M2D4M2D49M1D45M1D35M1D50M2D150M1D24M2D11M3D5M1D31M1I26M1I45M1D12M1D37M1I98M1D30M1D101M2D31M1D9M1D229M1I62M1I135M2D44M2D45M1I16M1I55M1D130M1I198M2D5M2I55M2D28M1D15M2D8M1I23M3D45M1I34M1D3M1D23M1D4M1D30M1D6M1I3M1D3M7D3M1I4M2I10M1I7M3D3M1I215M1D8M1D42M2I31M2D9M1D17M1I73M1D33M1D1M1D64M1I32M1I35M4D11M1D117M1D112M1D17M1I1M2D7M1D13M1I63M3I1M1I7M1I261M2D186M2D9M1D74M1I20M1D52M1I159M1D6M1D28M1D19M1D53M1I8M1I63M1D45M1D61M1D158M1D23M1I8M1D4M2D10M3D160M2D98M1D203M5I2M3I225M1D121M1D149M1I180M1I25M1I7M2I23M1D21M2D108M1I88M3D122M1D8M1I28M1D12M3D6M1D8M3D12M1I13M2D20M1I10M1D9M1D5M1D2M1D2M2I18M1I5M2D4M2I10M2D9M1D6M4D6M2D5M2D4M1D30M1I17M2D8M1I7M1I2M1D8M1D17M1I15M2D7M1D6M1D3M1D4M1D3M1I4M1I120M1D7M1D18M1D42M3D74M1I5M1I26M2I11M2D35M1D13M1D6M1D3M3I1M1D220M8D61M1I24M1D134M4I40M3D60M1I98M1D25M1D56M1D166M1D56M1D50M1I9M1D42M1D2M1D157M2D6M1I26M1D31M2I19M2I16M1I91M1D57M3I30M1D102M1I21M1D2M1D5M1I33M1D4M2I8M1D7M1D20M1D31M1D31M1D51M1D30M1I41M2D39M1D56M1D12M3D60M1D38M1I18M2I62M1D11M2I49M1D29M1D40M2I136M1D14M2D10M2D79M1I48M3D47M1D116M3D25M2D3M1D24M2D50M2D87M2I149M2D3M1D4M1D10M1I3M1D53M2D15M1D24M1D86M3D127M1D28M1D24M2D19M2D120M2I16M1D23M1D5M1D4M1I3M1D14M2D7M1D18M1D31M2I2M1D170M1I76M1D8M2D19M4I1M1I14M1I29M3D1M1D70M1D10M1D14M5D7M1D4M2I5M1D19M3I12M1D37M3I11M1D4M1D19M1D36M3D23M2I41M1D57M1D41M1D20M1I58M3I15M1D14M1D17M1D6M2I3M1I25M3D8M2I12M1D11M1I18M4I6M4D16M1D10M2D7M1D10M1D18M1D32M4D9M2I16M1I10M1I23M1I23M1I25M1D4M2D4M2D24M3D6M2D16M1D69M1D97M1D45M1D9M1D8M1D25M1D39M1D16M1I6M2D85M1D40M2I81M1D48M4D6M1I18M2I122M1D11M1I38M1D14M2D49M2D1M1D20M1I65M1D7M2D8M1D23M2D49M2D9M1I64M1I29M2I19M1D9M2I13M1D175M2D4M2I42M3D7M7D6M1D38M1D69M1I27M1D3M1D5M1I9M1D163M1I21M1I6M1I1M2D59M2I20M4D3M3D1M2I89M1I43M2D35M1D7M3D21M1D24M1D2M2D11M2I5M3D3M2I37M1D5M1D31M1I24M2I3M1I55M1D37M1D30M1I16M2D3M1D37M1I17M1D3M1D15M1I2M1D5M2I12M1I60M1D11M3I2M1I2M4D27M2D43M1D19M1D13M3D10M1I23M1I16M2I33M2D2M2D17M2I19M1I8M2D35M1D14M1D41M1I48M1D84M1I31M1I49M1I21M1D37M1D17M1I1M1I62M3I4M37I22M1D74M1D136M1D82M2I35M1D7M1I3M1I63M1I56M1D26M1I4M1D4M1D7M1D20M1I10M1D5M1D1M1D47M1D44M2I15M1I63M2D16M1D28M1I25M1D27M2D50M2I93M1D55M1D5M3D17M2D66M1D6M1D17M2I13M1D13M1I4M1I38M1D10M1D9M2D47M1D32M1D6M1D5M2I32M1D77M2I5M2D16M1D4M1D22M2D7M1I32M1D1M1D4M2D18M1I42M1D7M1D3M1D25M1D7M1D16M1I29M1I2M1D28M1D17M1I10M1I7M1D11M1D3M3D28M1I15M1D24M1D41M1D7M1I28M3I5M1D9M1D36M1D85M4D62M2D7M1I5M1I42M1D28M2I6M1D3M1D36M1I61M1I16M9I82M1D112M1I3M1D8M1I84M1I2M1I2M6D152M2I14M2D4M1D21M1I102M1D1M2I25M1I36M1I98M1I16M1I26M1I5M1I44M1D14M2I28M1D56M1D127M3I1M2D13M1D86M1I12M1D41M1I63M2I98M1I86M1D15M2D19M1I3M1D13M2D35M2D87M1I10M1I28M1I24M1D2M1D1M3I36M1D27M1D21M2D35M1D63M1D14M1D3M2I11M2D8M2D79M1D8M1D17M1D7M1D13M1D3M1I3M1D18M3D37M1I4M1D4M1D7M1D1M1D15M2D13M1D76M1D1M1D31M3D6M1I13M1I4M1I30M1D93M1D2M1D5M1I39M3D7M1D20M1I92M2I1M1D7M2I28M1D47M1I53M1D6M2D12M1D3M1D52M1I6M1I29M1I47M1D10M2I8M1D36M1I14M1D3M1D152M4I17M2D21M1D6M2I4M2I7M1I21M3D21M1D7M1I46M1D87M2I40M1I14M3I12M2D4M1D6M1I12M1I6M1D7M2D16M1D22M1D5M1I7M1D16M2D18M2D34M6D26M1I58M1D12M5D11M1I51M2I10M1I13M2I10M1I31M1I63M1I51M1D42M1I50M2I2M1D61M1D23M1I4M1I2M1D7M4I4M3D14M2I67M1I2M2D9M1D35M1D3M1I3M1D96M1D126M1I89M1I18M1D195M1I8M1D19M1I47M1I22M2D12M2D24M1D35M1D17M1I31M1I96M1D6M1D34M1D26M1D39M2D27M6D45M1I2M1D22M1I36M3D77M2D22M1D30M1D61M1D38M1D15M1D27M1D22M1I17M2D23M1D80M1I84M3I7M1D39M1D200M1D37M1I198M1D58M1D68M1I23M2D26M5D138M1I58M1D9M1D3M1I52M2I45M2D131M1I39M2I22M1D8M1I37M1I175M1D108M2D4M1I56M1D11M2D214M1I11M2I47M1D197M1I54M2I21M1D57M1I14M2D51M3I12M1D6M3D44M1I62M1D44M1D50M2D3M1I17M1D36M2D32M1I75M1I100M1I34M3I2M1D40M1I54M1D64M1D96M1D155M2I55M1D163M1I5M1I17M1D89M1I16M1I24M2I7M1D126M1D29M1D19M1I9M2D34M2D45M2D71M2D53M1I42M1I25M1D18M1I10M3D76M2D3M2D36M1I19M2D78M2I67M4D5M1D100M2D48M2D32M1I43M1D131M3I12M1D67M1I78M1D8M1I6M1D132M1I73M2I14M1D19M1D20M3D5M1I19M3D16M2I7M2D61M1D6M2I15M2I22M1D6M1D65M5D23M3I42M4D64M2D13M1D17M1D51M3I99M2I77M1D90M1I8M1D71M4D3M2I32M1D73M1D9M1D98M1D14M2D20M2I44M1D1M2I110M1D21M1I43M1I5M1D67M1D11M1I21M1D21M2D9M1I61M1D14M1D39M1D45M2I40M1D43M1D204M3I2M1D71M1I63M1D135M1D144M1D49M1D10M1D149M1I74M1I42M3D49M1D26M1D41M1D93M3D57M5I41M1I135M1I16M1D252M2I84M1D34M1I19M1D16M1D76M1D13M2I14M1I2M1I18M1I27M3I1M1I6M1D3M1D15M1D201M1I33M1D8M1I106M1D48M3D87M1D101M1I5M1I58M1D12M1I3M1I6M1D7M4D41M1D5M1I30M4D54M1I27M2D107M1D72M1D67M1D13M2D48M2D33M1D8M1I68M1D28M1D13M1D47M1I4M2I121M1D128M3D80M1D13M1D209M3I1M1I14M2D52M1I7M1D9M1D16M1I9M2D10M2D48M1D82M1D74M2D16M1I46M1I30M1D8M2I19M1D25M1I41M1D21M1I11M1I23M1I18M3D9M1D4M1D22M1I4M1D5M1D8M1I2M2D57M2D41M1I22M1I7M1D22M3I48M2D36M1I78M1D2M1D3M1D65M2D6M2I23M1I3M1I15M1I17M2I34M2I68M1I15M1I21M2D8M1D1M1D118M1D55M3D9M1D14M1I53M1I18M1D12M1D24M1D3M2D16M1D109M1D66M1I20M1D1M2D15M1D9M1D81M3I12M1I242M1D17M1D2M2D37M2I26M2I42M2D30M1I96M1I72M1D11M2I48M1D40M2D26M2D50M1D16M2D135M1D24M1D72M2I63M1I38M1I39M1D47M2I51M1D84M1D94M1D10M1I15M1D32M1D17M1D41M1D20M1D28M1I5M5D8M2D2M1I72M1D37M3D59M1D27M1I27M1D26M1D15M1I2M1D46M5D81M1D91M1I16M1I116M1D136M1D151M1D13M1I98M4D50M2D29M1D88M1I4M1I51M2I39M1I79M1D41M1D2M1D30M1I13M1I35M1D68M4I38M1D62M2D53M1I23M1I140M1D64M2D46M1I39M2D81M1I11M2D69M2D12M1I89M2I108M2D13M2I3M1D1M1D4M1D16M2I27M3D47M1I3M1D62M3D77M1I31M1I89M3D1M1D38M2D39M6D7M2I41M2D1M1D1M1D86M2I188M1D243M11I75M2I3M1D15M1D169M1I40M1D18M1D157M1D14M2D86M1I27M1I160M2D10M2D12M1I78M1I47M1D1M1D41M2I27M1I20M1I14M1I12M4D35M2I15M1I31M1I69M1I3M1I37M1D37M2D8M6D9M1D17M1D55M1I75M2D27M2D19M1I57M2I13M1D17M1D33M1D119M2D14M2I2M2D66M1D3M1D3M1I37M2D54M1I5M1D3M1D15M1D19M1I22M1D24M1I41M1I31M1D62M1D65M2D50M1D16M1D44M1I148M2I24M5D300M1I63M1D4M1D111M1I106M1I3M1D14M1D54M1D40M1D58M2D37M1D30M1I83M1D4M1D13M3I22M1I82M2I16M1I27M1I52M1I26M1I184M2I37M1D59M1D2M1I13M2I9M1D19M2D43M1I32M2I18M1D46M1D5M1D9M2I11M2D17M1D7M6D37M3D97M1D3M1D75M1I53M2D4M2D2M1D15M1I237M1I1M1I22M2I31M1I6M2I176M1D18M1D29M3D3M2D11M1D6M2D24M1I2M1I26M2D4M2D11M1I27M1D15M4D14M2D5M2I16M1D4M1I30M1D21M2D2M2I2M1I131M1D6M2I11M1D45M2D14M1D36M2I25M1D11M1D15M1I2M1I7M1D19M1D50M1D3M2D59M1D19M1I7M1I16M1D102M1I4M1I17M1D55M2D18M1I17M2D29M1D8M2D13M2D52M1D69M2D33M1D37M2I12M1D8M1D20M3I19M1I20M1D41M4D14M1I55M2I16M1D36M1I27M1I25M1D15M1D8M1D32M1I16M1D6M1D8M2D2M2I12M4I5M2D9M1D47M1D9M2D71M1I26M1D46M4D19M1D102M2D7M2D5M1I150M1D2M1D11M2I27M5D64M1D30M2D2M1D95M1D8M1D16M1D2M2I40M2I42M1I10M2I24M1I27M1D75M1I26M1D86M1D4M1I29M1I35M1I34M1I5M1I36M2I9M1D44M1D4M2D8M2D1M1D17M1I3M1D1M2D7M1D24M2D14M2D21M1I4M1I31M1D14M2I44M1I15M1D33M1D79M1I106M1D41M1D31M4D12M1D7M1I15M2D12M1D83M2D8M1D4M1I2M3D6M1I62M2D73M2D30M3D3M2D16M1D1M1D31M4D31M1I42M1D38M3D24M1D3M1I5M2D1M3D25M1D19M1D5M3D16M1I3M1D7M1I9M3I1M1I14M1D6M3D5M1D21M1I25M1I8M1D35M1D5M1I10M1D55M1D2M1D10M1I79M1D5M1I80M2D13M1I11M2D32M1D29M2D24M1D7M1I49M1D1M1D16M2D13M5D32M2I18M1I12M1D52M1I11M1D13M1I85M1D37M1I12M1I7M1D38M8D1M1D3M1D7M1D46M1D4M2I5M1I4M2D15M1D90M1I64M1D4M1D3M1D5M2D7M4I66M1I15M1I35M1D10M1D22M1D12M1I7M1D13M1D44M1D25M1I4M1D3M1D41M1D24M1I9M2I60M1D6M2D130M2I3M1I7M1D3M1D39M2D6M1I4M1D11M1I15M1D44M5D4M2D5M1I25M1I25M1I115M1I30M1D2M2I28M3I17M2I21M1I130M1D5M2D18M2D14M1D5M1I92M1I25M2D9M1I5M1I7M1D2M1D40M4D85M1I5M2I2M4D4M1D60M1I44M1D11M1I108M2I59M1D39M3D8M1D4M3D78M1D6M1I24M1I78M1D2M2I48M1I13M2I12M1D13M3D54M1I14M72I10M3I4M3I17M2D26M1D21M1I6M1D35M1I6M2D21M1D50M1D3M2I10M1D6M1I17M1D51M1D128M1D17M1D23M4D14M1D5M2D64M1D10M1D11M1I90M3D32M2D4M1I21M3D2M2D5M3I14M1D25M1D44M1D6M1I4M1D14M2D28M2D6M1D51M1I3M2D2M1I3M1D12M2D19M1I61M1I13M1I56M2I19M2D12M1D21M1D4M1I39M6D1M1D12M3D1M1D8M1I9M3D18M1D3M2D67M1D14M2D7M1D12M1D23M2D21M1D4M1D15M1D2M2I12M1I14M4D5M2I21M2I31M3D66M1I34M2I6M1D43M1D23M1D14M1I13M1I6M1D32M4D4M1D4M2I1M1D3M1D40M1D10M2I2M2D3M1D60M1I28M2I10M1D9M3I14M3D32M3D35M2I32M1D86M1I7M1D20M1D25M1D96M1D24M1I80M1I10M4D30M2D12M1D9M1I98M3D81M2I18M1D22M2D51M1I69M2D37M1I26M1D34M4D44M1I7M2I22M2I12M1D10M1D4M1D11M1D103M1I20M1I20M1D16M1D8M1D98M2D50M16D12M2D9M1D63M1D11M1D76M2I33M5D25M3D63M1D18M1I7M2D31M1D21M1D7M2D15M1I82M1D59M1I66M1I13M1D73M1D8M1D13M2D11M1I7M1I65M1I31M1D8M1D18M1I72M1I31M1I191M1D80M1D9M1D65M13D81M1I90M1D5M1I4M1D2M1D6M1D49M1D15M1D31M1D5M1I33M2I74M1D10M3I46M3I22M2D59M1D5M1D84M2I2M1D13M1I18M2D5M2I3M2I2M20I117M1I13M5D41M2D1M1D16M1S",
    "length": 117851,
    "mapping_quality": 60,
    "sequence_first100": "TTACTTCGTTCAGTTACGTATTGCTTTGAAGGCTGGGCTACCTCAGTGTGGAAGGTGGGCAGTTCCACAGGAATGGTGCCAGGGCCAGAGGGCAATACCA",
    "tags": [
      [
        "NM",
        7874
      ],
      [
        "ms",
        151392
      ],
      [
        "AS",
        152519
      ],
      [
        "nn",
        725
      ],
      [
        "tp",
        "P"
      ],
      [
        "cm",
        8416
      ],
      [
        "s1",
        58736
      ],
      [
        "s2",
        625
      ],
      [
        "dv",
        0.041600000113248825
      ],
      [
        "SA",
        "chr12,16970,+,21S18762M358D99068S,52,2368;chr3,76029363,+,18278S3125M142I96306S,1,555;chr19,75654,+,21822S1087M10D94942S,0,83;"
      ]
    ]
  },
  "source": "official Clair3 ONT quick demo HG003 Guppy3.6 ~85x; ONT non-dwell model ont",
  "truth": "GIAB HG003 GRCh38 chr20 v4.2.1 VCF and confident BED"
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-695c86abb06e

원래 파일: `clair3/data/sources.json` · 원문 SHA256: `8b43ce5b945d14288e46ca0e2217b32cf271698abfa0ba666472a343ec67e478`

<details>
<summary>clair3/data/sources.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/data/sources.json", "sha256": "8b43ce5b945d14288e46ca0e2217b32cf271698abfa0ba666472a343ec67e478", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "file": "GRCh38_no_alt_chr20.fa",
    "url": "https://www.bio8.cs.hku.hk/clair3/demo/quick_demo/ont/GRCh38_no_alt_chr20.fa",
    "bytes": 65518244,
    "sha256": "0ff3eab72be69fad01973c2261b965e1e4bd6bc49b77bb858d603ffbe11456d7"
  },
  {
    "file": "GRCh38_no_alt_chr20.fa.fai",
    "url": "https://www.bio8.cs.hku.hk/clair3/demo/quick_demo/ont/GRCh38_no_alt_chr20.fa.fai",
    "bytes": 23,
    "sha256": "7455fb06dab8ff448f729300bc8fb43ab486a06cfb97b8d91851696c1995a887"
  },
  {
    "file": "HG003_chr20_demo.bam",
    "url": "https://www.bio8.cs.hku.hk/clair3/demo/quick_demo/ont/HG003_chr20_demo.bam",
    "bytes": 27541300,
    "sha256": "ed4022b5b77533375fe9a8fe96a25636f95412386f65f4c618897b74eeee061f"
  },
  {
    "file": "HG003_chr20_demo.bam.bai",
    "url": "https://www.bio8.cs.hku.hk/clair3/demo/quick_demo/ont/HG003_chr20_demo.bam.bai",
    "bytes": 9920,
    "sha256": "90504152e97b89a4374c02a552861da1e048a4ee145a30ca9b99547180ef933c"
  },
  {
    "file": "HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz",
    "url": "https://www.bio8.cs.hku.hk/clair3/demo/quick_demo/ont/HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz",
    "bytes": 3344467,
    "sha256": "da3c275487ad01616809463f1364176bdaaee2ea90c9c745ae9752feb9bb35dc"
  },
  {
    "file": "HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz.tbi",
    "url": "https://www.bio8.cs.hku.hk/clair3/demo/quick_demo/ont/HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz.tbi",
    "bytes": 36079,
    "sha256": "deb63977231e56c18bb6d5d5c51d0174accaf86203579b3bb9a3b9e142223d06"
  },
  {
    "file": "HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed",
    "url": "https://www.bio8.cs.hku.hk/clair3/demo/quick_demo/ont/HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed",
    "bytes": 274339,
    "sha256": "29d03d1395f3f55b1d25407fe53f782f528e30b386b7f8b4c0c8821c654b71f3"
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-8f2e0476309d

원래 파일: `clair3/models/ont/sources.json` · 원문 SHA256: `4014c39ae3e46003e5bb3c6086578c55dbe63e411e4b495981da67462feb6dc1`

<details>
<summary>clair3/models/ont/sources.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/models/ont/sources.json", "sha256": "4014c39ae3e46003e5bb3c6086578c55dbe63e411e4b495981da67462feb6dc1", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "file": "pileup.pt",
    "url": "https://www.bio8.cs.hku.hk/clair3/clair3_models_pytorch/ont/pileup.pt",
    "sha256": "8fe535c5ebe287c43af365430492db9a4f30ffde2aae0d8e3bcb10635b9a3a35",
    "bytes": 8304746
  },
  {
    "file": "full_alignment.pt",
    "url": "https://www.bio8.cs.hku.hk/clair3/clair3_models_pytorch/ont/full_alignment.pt",
    "sha256": "cffb6e68c54095056b104e98023e91df7df52401c156ae2ea2ea6e0c4b81a162",
    "bytes": 11983414
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-94a05e390c73

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/capture_command.json` · 원문 SHA256: `949cf824181683569eccb94e6c7158457e6a18721ece95cf8a91eed98082859b`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/capture_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/capture_command.json", "sha256": "949cf824181683569eccb94e6c7158457e6a18721ece95cf8a91eed98082859b", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_chr20_demo.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/candidate_batch_20261028_185641_089149/calls",
  "--bed_fn=/work/data/region_200000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-6ff66c650b61

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-128_command.json` · 원문 SHA256: `1d8572f5403c8f3617ad7c92e57d49c31fc91066234d7e3a2a5aba68c99aa6c6`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_f-128_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-128_command.json", "sha256": "1d8572f5403c8f3617ad7c92e57d49c31fc91066234d7e3a2a5aba68c99aa6c6", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-128-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=128", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-814f36b7357d

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-1_command.json` · 원문 SHA256: `1d96bcc893c60356ec312399349c3043191e3f1829dc149739eee6406b4a1dc3`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_f-1_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-1_command.json", "sha256": "1d96bcc893c60356ec312399349c3043191e3f1829dc149739eee6406b4a1dc3", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-1-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=1", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-74aac22b890a

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-32_command.json` · 원문 SHA256: `c8ecf0b51fdf8170f588a59a4975837b33bdd2804ec6f3fcc7e38b6ec73a999b`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_f-32_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-32_command.json", "sha256": "c8ecf0b51fdf8170f588a59a4975837b33bdd2804ec6f3fcc7e38b6ec73a999b", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-32-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=32", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-881edfd6a0b0

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-512_command.json` · 원문 SHA256: `62d319b5ce384ab7a8ac2d7814224cc9335ff26c53d2ecf956def8f245ba219d`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_f-512_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-512_command.json", "sha256": "62d319b5ce384ab7a8ac2d7814224cc9335ff26c53d2ecf956def8f245ba219d", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-512-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=512", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-af7acbf1b9a8

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-894_command.json` · 원문 SHA256: `5c86cea7c2039efe3b1ea1dff47af9993ca9c0c6bb604faf0406d84cc104f76f`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_f-894_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-894_command.json", "sha256": "5c86cea7c2039efe3b1ea1dff47af9993ca9c0c6bb604faf0406d84cc104f76f", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-894-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=894", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-bf343cc19e42

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-8_command.json` · 원문 SHA256: `6ba5f550df8017c782834c7ce3588199e51d4e1ed031f22acb48e5c8b96c190f`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_f-8_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-8_command.json", "sha256": "6ba5f550df8017c782834c7ce3588199e51d4e1ed031f22acb48e5c8b96c190f", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-8-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=8", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-01beba8af330

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1000_command.json` · 원문 SHA256: `368e91ae82641349cb30f2671c85010d6677fb592c0f18cedca44d73c1590f28`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1000_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1000_command.json", "sha256": "368e91ae82641349cb30f2671c85010d6677fb592c0f18cedca44d73c1590f28", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-1000-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=1000", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-e2a28053273d

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-128_command.json` · 원문 SHA256: `baba4ff65e2e6872eb49be7b1139dd73af8a130f24fef43b0f6af8b90abe9afe`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_p-128_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-128_command.json", "sha256": "baba4ff65e2e6872eb49be7b1139dd73af8a130f24fef43b0f6af8b90abe9afe", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-128-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=128", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-106e7c500508

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1_command.json` · 원문 SHA256: `cabadd14ce2deb154356b248cdc0ce49a687f0ee963243e5c5840aba87f080af`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1_command.json", "sha256": "cabadd14ce2deb154356b248cdc0ce49a687f0ee963243e5c5840aba87f080af", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-1-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=1", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-4a220d77dfe6

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-2048_command.json` · 원문 SHA256: `18fa2b832c2cff24e9054d1c11ccce292f1aeec0ce26596513ae6631ccf4c181`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_p-2048_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-2048_command.json", "sha256": "18fa2b832c2cff24e9054d1c11ccce292f1aeec0ce26596513ae6631ccf4c181", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-2048-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=2048", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-5f125c013de2

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-32_command.json` · 원문 SHA256: `9f2403182aef7010b37f05c7feb27e23f2aa00a3d5e3e79b963b10f5bb4ea783`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_p-32_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-32_command.json", "sha256": "9f2403182aef7010b37f05c7feb27e23f2aa00a3d5e3e79b963b10f5bb4ea783", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-32-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=32", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-1ed92db4fc50

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-4096_command.json` · 원문 SHA256: `eebeeb98d0d87efd0970092ede2b8b3a3aa461488b77f821a49e970e6c3c9d56`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_p-4096_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-4096_command.json", "sha256": "eebeeb98d0d87efd0970092ede2b8b3a3aa461488b77f821a49e970e6c3c9d56", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-4096-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=4096", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-73df5b9a7fd4

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-512_command.json` · 원문 SHA256: `d98e82d23d095c2b49756ca5b9cd9cd7d4b73ed6048800f0fd9ffa74a3c34ad5`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_p-512_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-512_command.json", "sha256": "d98e82d23d095c2b49756ca5b9cd9cd7d4b73ed6048800f0fd9ffa74a3c34ad5", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-512-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=512", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-b5b62109f993

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-5708_command.json` · 원문 SHA256: `ef8c0c305cee19f98e5a6af54f8a639195d85f41d390daadfd0067c9b4cec748`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_p-5708_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-5708_command.json", "sha256": "ef8c0c305cee19f98e5a6af54f8a639195d85f41d390daadfd0067c9b4cec748", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-5708-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=5708", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-3f77f1f2cb64

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-8_command.json` · 원문 SHA256: `3abecbbfcbbb88cbc8a2093c5fd60fff08d7b95a33c1b2abdc35d2cf61d72eec`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/clair3_p-8_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-8_command.json", "sha256": "3abecbbfcbbb88cbc8a2093c5fd60fff08d7b95a33c1b2abdc35d2cf61d72eec", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-8-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=8", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-6e10b0702df4

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/host_capture_command.json` · 원문 SHA256: `c7a879beaacc81e1faaddc3eb97eda0227fecd8020629d6b7c23fd17ca8d1f99`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/host_capture_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/host_capture_command.json", "sha256": "c7a879beaacc81e1faaddc3eb97eda0227fecd8020629d6b7c23fd17ca8d1f99", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--gpus",
  "all",
  "--ipc=host",
  "--network",
  "none",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/clair3:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=candidate_batch_20261028_185641_089149",
  "-e",
  "PYTHONPATH=/work/scripts:/work/source",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "local/clair3-spark:9d1dd3f",
  "/work/.venv/bin/python",
  "-u",
  "/work/scripts/batch_followup/capture.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-1208cced6bbf

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/input_audit.json` · 원문 SHA256: `1f0d7bc8fa275ab93e2d356ca758d15eb7adcf9f5c695c19d5e947397271c22d`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/input_audit.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/input_audit.json", "sha256": "1f0d7bc8fa275ab93e2d356ca758d15eb7adcf9f5c695c19d5e947397271c22d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "Clair3_P": {
    "actual_feature_tensor_rows_before_deduplication": 5717,
    "unique_feature_rows": 5708,
    "feature_shape": [
      33,
      18
    ],
    "dtype": "int8",
    "source_caches": [
      {
        "file": "Clair3_P_364_0.npz",
        "shape": [
          384,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "d51110780bb2baa652943b335daaa63feaebd3c0f6df5603d07a712e6dbb9d5d"
      },
      {
        "file": "Clair3_P_364_1.npz",
        "shape": [
          426,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "101702cc5b9774b13cf07055dabfdd16fdd0074d0bb68af6dcc7b6bbb3cdaef5"
      },
      {
        "file": "Clair3_P_364_10.npz",
        "shape": [
          528,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "e6a1ab8d7e88158279a029783af4518300fcb1c59238e0c04a64e0d6a7b228ec"
      },
      {
        "file": "Clair3_P_364_11.npz",
        "shape": [
          554,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "1276ae724fe1be162945a11d0d5d9d591969b068de77fbba00c012ffb8c27ee7"
      },
      {
        "file": "Clair3_P_364_12.npz",
        "shape": [
          587,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "ce9bdc29c5ebff159f37d64dba63a56f11e3162530b3398ab71c19afdfbcbfde"
      },
      {
        "file": "Clair3_P_364_2.npz",
        "shape": [
          398,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "90e4dc800c0c50fc5e6303e8c6ef24f18ccc8d791bc5ce99fac713558627e399"
      },
      {
        "file": "Clair3_P_364_3.npz",
        "shape": [
          403,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "bb70e0f17a2fd8891d8d1f2571242584f68f776500dc98c00b8f5b7f126a56b8"
      },
      {
        "file": "Clair3_P_364_4.npz",
        "shape": [
          468,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "c606cb062ba8dbc537fc52318aaf2ba2975fd9f82c779cad57be5ecf6f9b2f56"
      },
      {
        "file": "Clair3_P_364_5.npz",
        "shape": [
          399,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "206c94596948941d11aec2b82c2090a8886d4a9953b4eac4d85b38a9de858156"
      },
      {
        "file": "Clair3_P_364_6.npz",
        "shape": [
          392,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "22a00c0619d232b524542dc43a1a3e5450c72a84411675f8b61e12272f3c8fb6"
      },
      {
        "file": "Clair3_P_364_7.npz",
        "shape": [
          376,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "a2f70d834b2e66403fb9d3984d96883cc47699a0f7e83995ea9576bd97334063"
      },
      {
        "file": "Clair3_P_364_8.npz",
        "shape": [
          388,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "34823de50bbff28a79e96763f70ad626881d2b8fb930d83d78911b34d9423178"
      },
      {
        "file": "Clair3_P_364_9.npz",
        "shape": [
          414,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "f84e002b1fa4d1ee0a94d66bf1446548d7e0f3755ad7da984c25ad871f4d5c22"
      }
    ],
    "identity_limitation": "deduplicated exact feature tensors; genomic candidate coordinates not exported by this capture hook, no invented position IDs",
    "source": "real HG003 ONT BAM, GRCh38 chr20 100000..300000, existing official Clair3 extraction; no zero-filled or synthetic performance inputs"
  },
  "Clair3_F": {
    "actual_feature_tensor_rows_before_deduplication": 894,
    "unique_feature_rows": 894,
    "feature_shape": [
      89,
      33,
      8
    ],
    "dtype": "int8",
    "source_caches": [
      {
        "file": "Clair3_F_562_0.npz",
        "shape": [
          894,
          89,
          33,
          8
        ],
        "dtype": "int8",
        "sha256": "fcb088913ed72ce7479b546432381f80948ba4da186e109c580fee135d6fa872"
      }
    ],
    "identity_limitation": "deduplicated exact feature tensors; genomic candidate coordinates not exported by this capture hook, no invented position IDs",
    "source": "real HG003 ONT BAM, GRCh38 chr20 100000..300000, existing official Clair3 extraction; no zero-filled or synthetic performance inputs"
  }
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-e2002103343c

원래 파일: `clair3/results/candidate_batch_20261028_185641_089149/prepare_command.json` · 원문 SHA256: `c3efc3e1993a9f254815cf3fa16ea5ca15c855060a326f7d58fdf6be2560837d`

<details>
<summary>clair3/results/candidate_batch_20261028_185641_089149/prepare_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/prepare_command.json", "sha256": "c3efc3e1993a9f254815cf3fa16ea5ca15c855060a326f7d58fdf6be2560837d", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-prepare-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/prepare_cache.py"]
```
<!-- END_RESTORE_FILE -->


</details>

### record-c8bd73f6b0a0

원래 파일: `clair3/results/clair3_20261027_203222_759031/benchmark_command.json` · 원문 SHA256: `7a415096730e28a352362d3ef7a6f5df4292d68de28f14f35e99906b60db749b`

<details>
<summary>clair3/results/clair3_20261027_203222_759031/benchmark_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203222_759031/benchmark_command.json", "sha256": "7a415096730e28a352362d3ef7a6f5df4292d68de28f14f35e99906b60db749b", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/clair3:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=clair3_20261027_203222_759031 -e PYTHONPATH=/work/scripts:/work/source -e MPLCONFIGDIR=/work/.cache/matplotlib local/clair3-spark:9d1dd3f /work/.venv/bin/python -u /work/scripts/benchmark.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-8560a2f733b0

원래 파일: `clair3/results/clair3_20261027_203222_759031/region_50000_full/command.json` · 원문 SHA256: `c1aa7986a959a7086934382c993b12f49e7b5c812ac247a3943c093c2f43345f`

<details>
<summary>clair3/results/clair3_20261027_203222_759031/region_50000_full/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203222_759031/region_50000_full/command.json", "sha256": "c1aa7986a959a7086934382c993b12f49e7b5c812ac247a3943c093c2f43345f", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_chr20_demo.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/clair3_20261027_203222_759031/region_50000_full/calls",
  "--bed_fn=/work/data/region_50000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-5d08764c1b00

원래 파일: `clair3/results/clair3_20261027_203339_768667/benchmark_command.json` · 원문 SHA256: `63b2d5d641ce3e45020d7e9f6bf523732b4b447c9da0a4f1c8a1811ce3b61887`

<details>
<summary>clair3/results/clair3_20261027_203339_768667/benchmark_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/benchmark_command.json", "sha256": "63b2d5d641ce3e45020d7e9f6bf523732b4b447c9da0a4f1c8a1811ce3b61887", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/clair3:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=clair3_20261027_203339_768667 -e PYTHONPATH=/work/scripts:/work/source -e MPLCONFIGDIR=/work/.cache/matplotlib local/clair3-spark:9d1dd3f /work/.venv/bin/python -u /work/scripts/benchmark.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-80c9df8addf5

원래 파일: `clair3/results/clair3_20261027_203339_768667/different_batch_reload_command.json` · 원문 SHA256: `c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3`

<details>
<summary>clair3/results/clair3_20261027_203339_768667/different_batch_reload_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/different_batch_reload_command.json", "sha256": "c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/clair3:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=clair3_20261027_203339_768667 -e PYTHONPATH=/work/scripts:/work/source -e MPLCONFIGDIR=/work/.cache/matplotlib local/clair3-spark:9d1dd3f /work/.venv/bin/python -u /work/scripts/reload.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-c04f0ce190c3

원래 파일: `clair3/results/clair3_20261027_203339_768667/region_100000_full/command.json` · 원문 SHA256: `597deb7e845b2d4e2aa0dab872effcddc07daafa692a549a7a679bd148eb315d`

<details>
<summary>clair3/results/clair3_20261027_203339_768667/region_100000_full/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_100000_full/command.json", "sha256": "597deb7e845b2d4e2aa0dab872effcddc07daafa692a549a7a679bd148eb315d", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_chr20_demo.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/clair3_20261027_203339_768667/region_100000_full/calls",
  "--bed_fn=/work/data/region_100000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-65b481af4022

원래 파일: `clair3/results/clair3_20261027_203339_768667/region_200000_full/command.json` · 원문 SHA256: `d7397e0f3db35c331c171335765b85ea58aa7c63a3d69be54c06971d64478dcb`

<details>
<summary>clair3/results/clair3_20261027_203339_768667/region_200000_full/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_200000_full/command.json", "sha256": "d7397e0f3db35c331c171335765b85ea58aa7c63a3d69be54c06971d64478dcb", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_chr20_demo.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/clair3_20261027_203339_768667/region_200000_full/calls",
  "--bed_fn=/work/data/region_200000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-cf5df12c8701

원래 파일: `clair3/results/clair3_20261027_203339_768667/region_200000_half/command.json` · 원문 SHA256: `c509b115d1d876f16b5e112f9046ec4bf69dbc977b7c1c02db2ad269e108b227`

<details>
<summary>clair3/results/clair3_20261027_203339_768667/region_200000_half/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_200000_half/command.json", "sha256": "c509b115d1d876f16b5e112f9046ec4bf69dbc977b7c1c02db2ad269e108b227", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_half.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/clair3_20261027_203339_768667/region_200000_half/calls",
  "--bed_fn=/work/data/region_200000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-e29b4da7971b

원래 파일: `clair3/results/clair3_20261027_203339_768667/region_50000_full/command.json` · 원문 SHA256: `152c0bb041222e6731f4632148bbb358aeaeea1375e9693b0803079350a407cf`

<details>
<summary>clair3/results/clair3_20261027_203339_768667/region_50000_full/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_50000_full/command.json", "sha256": "152c0bb041222e6731f4632148bbb358aeaeea1375e9693b0803079350a407cf", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_chr20_demo.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/clair3_20261027_203339_768667/region_50000_full/calls",
  "--bed_fn=/work/data/region_50000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-01b19b43542f

원래 파일: `clair3/results/clair3_20261027_203339_768667/reload_command.json` · 원문 SHA256: `c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3`

<details>
<summary>clair3/results/clair3_20261027_203339_768667/reload_command.json — 명령/설정/환경 원문</summary>


동일 원문은 `clair3/results/clair3_20261027_203339_768667/different_batch_reload_command.json` 블록에 전문 수록했다. 아래 별칭은 동일한 바이트를 원래 경로로 복원한다.
<!-- RESTORE_ALIAS {"path": "clair3/results/clair3_20261027_203339_768667/reload_command.json", "source": "clair3/results/clair3_20261027_203339_768667/different_batch_reload_command.json", "sha256": "c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3"} -->


</details>

### record-73a9f4c12b24

원래 파일: `scripts/clair3_batch_launch.py` · 원문 SHA256: `b8724c07ea44c668b75692f6f11f1c62090676709f754db4e207018948f8e0f9`

<details>
<summary>scripts/clair3_batch_launch.py — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "scripts/clair3_batch_launch.py", "sha256": "b8724c07ea44c668b75692f6f11f1c62090676709f754db4e207018948f8e0f9", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,subprocess,shlex,os,sys,time
import psutil
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'clair3';run=(R/'results/BATCH_LATEST').read_text().strip();D=R/'results'/run;assert json.loads((D/'capture_exit.json').read_text())['exit_code']==0
S={'status':'running','run_id':run,'conditions':[],'failed':[],'settings':{'task':'native candidate model inference only','threads':4,'precision':'FP32','tf32':False,'guard_available_gib':16,'guard_swap_increase_gib':2,'baseline_calling_batch':1000,'note':'BAM candidate extraction completed once; actual cached candidate rows reused; no artificial repeated rows within batch'}};swap0=psutil.swap_memory().used
def save():(D/'summary.json').write_text(json.dumps(S,indent=2))
def stage(label,script,envextra=()):
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU job'
 name='clair3-'+label+'-'+run;cmd=['docker','run','--rm','--name',name,'--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','PYTHONPATH=/work/source:/work/scripts']
 for e in envextra:cmd+=['-e',e]
 cmd+=['local/clair3-spark:9d1dd3f','/work/.venv/bin/python','-u','/work/scripts/batch_followup/'+script];(D/(label+'_command.json')).write_text(json.dumps(cmd));t=time.time();reason=None
 with (D/(label+'.log')).open('w') as f:
  p=subprocess.Popen(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',name])],stdout=f,stderr=subprocess.STDOUT);break
   time.sleep(.5)
  code=p.wait()
 (D/(label+'_exit.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-t,'reason':reason}));print(label,code,flush=True);return code
save();assert stage('prepare','prepare_cache.py')==0;audit=json.loads((D/'input_audit.json').read_text());S['input_audit']=audit;save()
for kind in ['Clair3_P','Clair3_F']:
 pool=audit[kind]['unique_feature_rows'];last=None
 for B in sorted(set([b for b in [1,8,32,128,512,1000,2048,4096,8192,16384] if b<=pool]+[pool])):
  if last:
   base=last['model_only_allocated_gib'];estimate=base+(last['memory']['peak_reserved_gib']-base)*B/last['batch_size']*1.25+3
   if estimate>psutil.virtual_memory().available/2**30-16:S.setdefault('stop_reasons',{})[kind]='predicted_reservation_guard';break
  code=stage(kind.lower()+'-'+str(B),'worker.py',['CANDIDATE_MODEL='+kind,'BATCH='+str(B)])
  if code:S['failed'].append({'model':kind,'batch':B,'exit_code':code});break
  w=json.loads((D/f'{kind}_batch_{B}/summary.json').read_text());last=w['condition'];last['model']=kind;last['model_only_allocated_gib']=w['model_only_postload']['allocated_gib'];S['conditions'].append(last);save()
 else:S.setdefault('stop_reasons',{})[kind]='available_real_candidate_cache_limit'
S['status']='complete' if not S['failed'] else 'stopped';save();subprocess.run([sys.executable,str(ROOT/'scripts/two_batch_report.py'),'clair3',str(D)],check=True);print('CLAIR3_FINISHED',S['status'],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

### record-9e446b3aa245

원래 파일: `scripts/two_batch_report.py` · 원문 SHA256: `a4fa6195788965f1fd6a0599718a24fcc35feebdc681fa10f26bcab070c22277`

<details>
<summary>scripts/two_batch_report.py — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "scripts/two_batch_report.py", "sha256": "a4fa6195788965f1fd6a0599718a24fcc35feebdc681fa10f26bcab070c22277", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,sys,csv
ROOT=Path('/home/hyein/ai-benchmark');kind=sys.argv[1];D=Path(sys.argv[2]);S=json.loads((D/'summary.json').read_text());R=ROOT/('colabfold' if kind=='af2' else 'clair3');out=[]
def a(x=''):out.append(x)
if kind=='clair3':
 a('# Clair3 실제 후보 tensor의 배치 확대 후속 실험');a();a('상태 **'+S['status']+'** · '+D.name);a();a('기존HG003 ONT BAM/GRCh38 chr20:100000~300000/threads4/한GPUworker의공식variant calling을다시실행해실제후보feature tensor 전체를cache했다. 모델입력이나호출결과를바꾸지않는opt-in capture hook을사용했다. CLI옵션이없더라도모델의실제forward 입력을분리해batch를조절할수있음을확인한실험이다. 후보생성과VCF호출은별도단계이며후속시간은CPUcache→GPU nativeforward→CPU확률반환만포함한다.');a();a('pileup과full-alignment는다른입력형태/모델로따로측정한다. 같은feature값은중복제거했고가짜zero row/반복복제row를batch에채우지않았다. 원본candidate좌표는capturehook으로내보내지못했으므로고유생물학적위치수대신고유feature row수를보고한다. cache파일SHA256·원본shape/dtype·선택pool은input_audit.json에있다.');a();a('|모델|고유feature pool|각후보shape|dtype|');a('|---|---|---|---|')
 for k,v in S['input_audit'].items():a(f"|{k}|{v['unique_feature_rows']}|{v['feature_shape']}|{v['dtype']}|")
 a();a('|모델|batch|반복|고유후보|총처리|초|후보/초|모델적재GiB|allocated peak GiB|추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|');a('|---|---|---|---|---|---|---|---|---|---|---|---|---|');flat=[]
 for c in S['conditions']:
  m=c['memory'];base=c['model_only_allocated_gib'];r={'model':c['model'],'batch':c['batch_size'],'repeats':c['repeats'],'unique_candidates':c['unique_candidates'],'processed_candidates':c['processed_candidates'],'seconds':c['seconds'],'candidates_per_second':c['candidates_per_second'],'model_loaded_gib':base,'allocated_peak_gib':m['peak_allocated_gib'],'additional_from_load_gib':m['peak_allocated_gib']-base,'reserved_peak_gib':m['peak_reserved_gib'],'percent_shared_ram':100*c['allocated_fraction_shared_ram'],'gpu_avg':c['gpu_util_avg'],'gpu_max':c['gpu_util_max']};flat.append(r);a('|'+ '|'.join(str(round(v,4)) if isinstance(v,float) else str(v) for v in r.values())+'|')
 a();a('FP32/TF32=False/seed42, 기존공식ont가중치를고정했다. 각batch는새프로세스로실행한다. 전체파라미터/적재기준값/실제입력dtype·shape/출력shape/표본메모리는각worker summary에있다.');a('원래공식pipeline에서기록한동일후보의출력확률과batch1재추론대조도통과했다(P최대오차4.17e-6,F1.06e-5). native_pipeline_candidate_equivalence.json에저장했다.');a('전체출력finite·저장재로드정확일치·같은첫후보의batch간확률일치(atol/rtol1e-4)를검증했다. 실제BAM호출VCF도별도보존했다. 이단계는새로운variant accuracy평가나학습이아니며기존truth정확도결과를참조한다.');a();a('종료이유: '+json.dumps(S.get('stop_reasons',{}),ensure_ascii=False));a('확보한실제후보pool에서측정한최대batch이며모델절대한계가아니다. GPU사용률은메모리비율과다르다. CUDAallocated/reserved는프로세스allocator,분모는CPU/GPU공유RAM121.688GiB다. RSS와CUDA를더하지않는다. 가용16GiB·swap+2GiB guard,0.5초대기+조회표본간격. RSS/가용량sampled값과CUDApeak를구분한다.');a();a('[공식 Clair3](https://github.com/HKU-BAL/Clair3) · [기본 결과](Clair3.md)')
else:
 a('# AlphaFold2 — ColabFold JAX vmap 배치 후속 실험');a();a('상태 **'+S['status']+'** · '+D.name);a();a('공식CLI가제공하는배치기능이아닌**별도실험구현**이다. 기존ColabFold1.6.2/AlphaFold2 PTM model_1의가중치·forward를사용하고 JAX vmap으로서로독립인실제단백질query축을추가했다. 복합체사슬을batch로세거나다른GPU작업을동시에띄우지않았다. 이번비교는FP32활성값/가중치를고정했다. 앞선BF16벡터화는native대조CA RMSD1.075Å로사전0.5Å검증기준을넘어실패기록을보존했고,수치민감도를확인하기위해별도FP32조건을분리했다. 기본BF16성능수치와직접비교하지않는다. 모델수는고정1개,3recycle(초기pass포함4회),ensemble1,seed42/query,dropout비활성,template/relaxation없음이다. 기본보고서의5모델순차실행과다른조건으로구분한다.');a();a('기존실제단량체4개는각115잔기이며기존MSA를재사용한다. 원래MSA깊이·SHA256은input_audit.json,실제처리msa mask깊이와각feature shape은worker summary에있다. extra-MSA/cluster축은pool내최대shape로zero pad하고마스크도0으로유지했다. 잔기서열을복제하거나추가하지않았다. 큰모델에서vmap이동작하는지부터단일native predict와검증하는pilot이며길이900AA같은다른조건의결과로확대해해석하지않는다.');a();a('|batch/고유query|반복|초|query/초|GPU평균/최대%|프로세스누적JAX peak GiB|실행중sampled bytes-in-use peak GiB|컴파일초|');a('|---|---|---|---|---|---|---|---|');flat=[]
 for w in S['workers']:
  if w['status']!='complete':continue
  c=w['conditions'][0];m=c['jax_memory_stats'];sample=c['execution_sampled_peak_bytes_in_use'];r={'batch':c['batch'],'repeats':c['repeats'],'seconds':c['seconds'],'queries_per_second':c['queries_per_second'],'gpu_avg':c['gpu_util_avg'],'gpu_max':c['gpu_util_max'],'cumulative_process_peak_gib':m['peak_bytes_in_use']/2**30 if 'peak_bytes_in_use' in m else None,'execution_sampled_peak_gib':sample/2**30 if sample is not None else None,'compile_seconds':w['compilation_seconds']};flat.append(r);a('|'+ '|'.join(str(round(v,4)) if isinstance(v,float) else str(v) for v in r.values())+'|')
 a();a('시간'+'은동기화된4passforward·recycle의FP16반환·prev재입력을포함한다. MSA/featurecache구성·모델로딩·컴파일·warmup·PDB저장·native참조검증은별도시간이다. JAXpeak는프로세스생성후컴파일/warmup포함누적allocator값이며실행구간reset peak라고쓰지않는다. compiled.memory_analysis도별도기록한다. 실행중bytes_in_use/RSS/시스템가용량은0.5초대기+조회sampled값이다. CPU/GPU공유RAM에서RSS와JAX값을더하지않는다.');a();a('단일batch1과원래runner.predict의동일query/seed출력은CA정렬RMSD·pLDDT차이로비교했다. 다른batch도첫query를batch1과비교해벡터화의수치차이를보존했다. 모든좌표/pLDDTfinite·잔기수·PDB의CA수검증을수행했다. 신뢰도와정답구조정확도를혼동하지않으며새독립구조정확도평가는하지않았다. 실제4querypool끝에서종료하므로80/90%메모리최대가능성을검증한실험은아니다.');a();a('[공식 ColabFold](https://github.com/sokrypton/ColabFold) · [JAX vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) · [기본 결과](AlphaFold2_ColabFold.md)')
a();a('원본실행: results/'+D.name+'; 명령·소스·가중치출처·inputcache·출력·자원시계열·exit·검증값을보존한다.');(R/'reports/Batch_memory_followup.md').write_text('\n'.join(out)+'\n')
if flat:
 with (R/'reports/batch_memory_followup.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for name in ['STATUS.md','RESULTS.md','MEMORY_RESULTS.md','FOLLOWUP_RESULTS.md']:
 with (ROOT/name).open('a') as f:f.write('\n'+kind+' 전용추가후속 '+S['status']+'; [결과]('+str(R.relative_to(ROOT))+'/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.\n')

qpath=ROOT/'FOLLOWUP_QUEUE.json'
if qpath.exists():
 q=json.loads(qpath.read_text());label='AlphaFold2' if kind=='af2' else 'Clair3';q.setdefault('additional_models',{})[label]={'status':S['status'],'run_id':D.name,'report':str(R.relative_to(ROOT))+'/reports/Batch_memory_followup.md'};q['applicability_notes'][label]='후속 실제 실행 결과가 기존 미실시 기록을 갱신함: '+S['status']+'. CLI 기본 배치와 별도 구현/계측 범위는 새 보고서 참조.'
 if all(q['additional_models'].get(k,{}).get('status')=='complete' for k in ['AlphaFold2','Clair3']):q['status']='all_ten_followups_completed_with_scope_differences'
 qpath.write_text(json.dumps(q,indent=2,ensure_ascii=False))
```
<!-- END_RESTORE_FILE -->


</details>

### record-30dedcecbf5f

원래 파일: `scripts/two_remaining_launch.py` · 원문 SHA256: `a93cb7bf32db3e72a391b9a3f420c4cd040776644a5c1c25850e0ee5b2d970d2`

<details>
<summary>scripts/two_remaining_launch.py — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "scripts/two_remaining_launch.py", "sha256": "a93cb7bf32db3e72a391b9a3f420c4cd040776644a5c1c25850e0ee5b2d970d2", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,subprocess,datetime,shlex,json,time
ROOT=Path('/home/hyein/ai-benchmark');run='candidate_batch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=ROOT/'clair3/results'/run;D.mkdir();(D.parent/'BATCH_LATEST').write_text(run+'\n');print(run,flush=True);assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();cmd=['docker','run','--rm','--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(ROOT/'clair3')+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','PYTHONPATH=/work/scripts:/work/source','-e','MPLCONFIGDIR=/work/.cache/matplotlib','local/clair3-spark:9d1dd3f','/work/.venv/bin/python','-u','/work/scripts/batch_followup/capture.py'];(D/'host_capture_command.json').write_text(json.dumps(cmd,indent=2))
with (D/'capture.log').open('w') as f:code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
print('CAPTURE_EXIT',code,flush=True);assert code==0
```
<!-- END_RESTORE_FILE -->


</details>

### 원문 확인·로컬 복원

아래 코드는실제로해시검증·독립임시폴더복원검증에사용한도구전문이다. 필요할때로컬의restore.py에저장하여저장소루트에서`python3 restore.py`(검증),`python3 restore.py --write`(복원)로사용한다. 실제모델실험은자동실행하지않는다. 코드/설정/명령은본문코드블록에수록했고원본데이터·모델가중치·원시자원시계열/대형예측은장비로컬에보존했다. 원본로그를모두보고서본문으로옮긴것은아니다.

```python
from pathlib import Path
import argparse,base64,hashlib,json,re,sys
p=argparse.ArgumentParser(description='Verify or restore original files embedded in Markdown. No benchmark is executed.');p.add_argument('--write',action='store_true',help='restore verified original files under repository root');args=p.parse_args();root=Path.cwd().resolve();files={};aliases=[];blocks=0
for md in root.rglob('*.md'):
 if '.git' in md.parts:continue
 lines=md.read_text(encoding='utf-8').splitlines(keepends=True);i=0
 while i<len(lines):
  line=lines[i].strip()
  if line.startswith('<!-- RESTORE_ALIAS ') and line.endswith(' -->'):aliases.append(json.loads(line[len('<!-- RESTORE_ALIAS '):-len(' -->')]))
  if not line.startswith('<!-- RESTORE_FILE ') or not line.endswith(' -->'):i+=1;continue
  meta=json.loads(line[len('<!-- RESTORE_FILE '):-len(' -->')]);i+=1;opening=re.match(r'^(`{3,})[^\n]*\n?$',lines[i]);assert opening,(md,i);fence=opening.group(1);i+=1;content=[]
  while i<len(lines) and lines[i].strip()!=fence:content.append(lines[i]);i+=1
  assert i<len(lines),(md,'missing closing fence');payload=''.join(content)
  if meta['encoding']=='base64':data=base64.b64decode(payload.strip(),validate=True)
  else:
   if meta.get('added_final_newline'):assert payload.endswith('\n');payload=payload[:-1]
   data=payload.encode('utf-8')
  actual=hashlib.sha256(data).hexdigest();assert actual==meta['sha256'],(md,meta['path'],'SHA256 mismatch');path=(root/meta['path']).resolve();assert root in path.parents,(meta['path'],'path outside repository');previous=files.get(meta['path']);assert previous is None or previous['data']==data,(meta['path'],'conflicting code versions');files[meta['path']]={'data':data,'sha256':actual};blocks+=1;i+=1
pending=aliases[:]
while pending:
 progress=False
 for meta in pending[:]:
  if meta['source'] not in files:continue
  src=files[meta['source']];assert src['sha256']==meta['sha256'];previous=files.get(meta['path']);assert previous is None or previous['data']==src['data'];files[meta['path']]=src;pending.remove(meta);progress=True
 assert progress,('unresolved aliases',pending)
if args.write:
 for name,item in files.items():
  dest=(root/name).resolve();assert root in dest.parents;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and dest.read_bytes()!=item['data']:raise RuntimeError('Refusing to overwrite different existing file: '+name)
  dest.write_bytes(item['data'])
print(json.dumps({'verified_embedded_blocks':blocks,'restorable_original_paths':len(files),'restored':args.write},ensure_ascii=False))

```

<details><summary>추가 실험 실행 원문: Clair3_F_full_real_pool_executed_source.py</summary>

<!-- RESTORE_FILE {"path": "clair3/results/real_input_capacity_20261102_173336_910773/Clair3_F_full_real_pool_executed_source.py", "sha256": "84cf5afc14e8632adf01694a0c6f3facf7465d6733e9e71f91eece9044c9d40c", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,time,json,gc
import numpy as np,torch
sys.path[:0]=['/support','/work/source'];from gpu_support import Meter,write_json
from clair3.model import Clair3_P,Clair3_F
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];
def repair_cached_links():
 for link in D.iterdir():
  if link.is_symlink():
   target=os.readlink(link);prefix='/home/hyein/ai-benchmark/clair3/'
   if target.startswith(prefix):
    relative=os.path.relpath(R/target[len(prefix):],link.parent);link.unlink();link.symlink_to(relative)
repair_cached_links()
kind=os.environ['CANDIDATE_MODEL'];B=int(os.environ['BATCH']);C=D/f'{kind}_batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'batch':B,'task':'actual candidate tensor native model inference, excludes BAM preprocessing/VCF calling'}
try:
 torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='loading';t=time.time();model=(Clair3_P if kind=='Clair3_P' else Clair3_F)(predict=True,add_indel_length=kind=='Clair3_F').float().cuda().eval();p=R/'models/ont'/('pileup.pt' if kind=='Clair3_P' else 'full_alignment.pt');ck=torch.load(p,map_location='cpu',weights_only=False);model.load_state_dict(ck.get('state_dict',ck));del ck;torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']=sum(p.numel() for p in model.parameters());S['parameters_trained_in_run']=0
 x=np.load(D/(kind+'_inputs.npy'),mmap_mode='r');assert len(x)>=B;values=np.array(x[:B]);S['input_shape']=list(values.shape);S['input_dtype']=str(values.dtype);S['unique_candidates']=B
 def forward():
  with torch.inference_mode():return model(torch.from_numpy(values).cuda()).cpu().numpy()
 meter.phase='warmup';t=time.time();out=forward();torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del out;gc.collect();t,base=meter.begin('candidate_inference');repeats=0
 while repeats<2 or time.time()-t<15:out=forward();assert np.isfinite(out).all();torch.cuda.synchronize();repeats+=1
 r=meter.end(t,base,batch_size=B,unique_candidates=B,repeats=repeats,processed_candidates=B*repeats);r['candidates_per_second']=B*repeats/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];S['condition']=r;np.save(C/'predictions.npy',out);assert np.array_equal(np.load(C/'predictions.npy'),out);S['output_shape']=list(out.shape);ref=D/(kind+'_reference_first.npy')
 if ref.exists():S['batch_invariance_max_abs']=float(np.abs(np.load(ref)-out[0]).max());assert np.allclose(np.load(ref),out[0],atol=1e-4,rtol=1e-4)
 else:np.save(ref,out[0])
 S['finite_and_exact_save_reload']=True;S['status']='complete';write_json(C/'summary.json',S);print('BATCH_COMPLETE',kind,B,r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write_json(C/'summary.json',S);raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: Clair3_F_full_real_pool_command.json</summary>

<!-- RESTORE_FILE {"path": "clair3/results/real_input_capacity_20261102_173336_910773/Clair3_F_full_real_pool_command.json", "sha256": "813a9e7cc3c422a0cea7ecdb730a1b8a45b0eabd490a59596f3f2cca3174efdd", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--gpus",
  "all",
  "--ipc=host",
  "--network",
  "none",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/clair3:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_173336_910773",
  "-e",
  "CANDIDATE_MODEL=Clair3_F",
  "-e",
  "BATCH=894",
  "local/clair3-spark:9d1dd3f",
  "/work/.venv/bin/python",
  "-u",
  "/work/scripts/batch_followup/worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: Clair3_P_full_real_pool_command.json</summary>

<!-- RESTORE_FILE {"path": "clair3/results/real_input_capacity_20261102_173336_910773/Clair3_P_full_real_pool_command.json", "sha256": "1948b1b3ef069d4d15be8265fe445f3fbd132937424a7efd70d86be8a2fbcf1c", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--gpus",
  "all",
  "--ipc=host",
  "--network",
  "none",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/clair3:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_173336_910773",
  "-e",
  "CANDIDATE_MODEL=Clair3_P",
  "-e",
  "BATCH=5708",
  "local/clair3-spark:9d1dd3f",
  "/work/.venv/bin/python",
  "-u",
  "/work/scripts/batch_followup/worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: remaining_queue_152f0b87bce32235.py</summary>

<!-- RESTORE_FILE {"path": "clair3/results/real_input_capacity_20261102_173336_910773/reproduction_sources/remaining_queue_152f0b87bce32235.py", "sha256": "152f0b87bce32235a41f8616d74f03deabd611eef85dde6419bd68da175973a6", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,json,datetime,subprocess,shlex,time,shutil,csv
ROOT=Path('/home/hyein/ai-benchmark');P=ROOT/'SUPPLEMENT_EXPERIMENT_STATUS.json';FETCH=ROOT/'scripts/real_training_v3/fetched_inputs'
def update(**kw):
 state=json.loads(P.read_text());state.update(**kw);P.write_text(json.dumps(state,indent=2,ensure_ascii=False))
def idle():assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another compute job exists'
def launch(D,label,cmd,source):
 idle();shutil.copy2(source,D/(label+'_executed_source.py'));(D/(label+'_command.json')).write_text(json.dumps(cmd,indent=2));update(active_stage=label);print('START',label,flush=True);t=time.time()
 with (D/(label+'.log')).open('w') as f:code=subprocess.call(cmd if cmd[0]!='docker' else ['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 (D/(label+'_exit.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-t}));print('EXIT',label,code,flush=True);return code
if (ROOT/'STOP_AFTER_CELLPOSE').exists():
 update(status='stopped_after_cellpose_by_user',active_model=None,active_stage=None)
 print('STOP_AFTER_CELLPOSE: user requested stop; no remaining model launched',flush=True)
 sys.exit(0)
for model in ['esm2','dnabert2','ntv2','rnafm','openspliceai','scvi','clair3','colabfold']:
 if os.environ.get('RESUME_REMAINING')=='1' and model in [v['model'] for v in json.loads(P.read_text()).get('additional_models_measured',[])]:continue
 run='real_input_capacity_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');R=ROOT/model;D=R/'results'/run;D.mkdir();(R/'results/REAL_TRAINING_LATEST').write_text(run+'\n');update(active_model=model,active_run=run,status='running');S={'status':'running','run_id':run,'model':model,'scope':'actual unique inputs, inference capacity extension; existing verified training results retained separately','conditions':[],'failed':[],'allocator_policy':'PyTorch process memory cap100GiB, guard available8GiB/swap+2GiB; JAX no preallocation'}
 if model in ['esm2','rnafm']:
  src=FETCH/(model+'.json')
  if not src.exists():S.update(status='blocked',reason='real_input_download_failed');(D/'summary.json').write_text(json.dumps(S));continue
  records=json.loads(src.read_text());sizes=([1,384,448,512,544,560,576,584,592,600] if model=='esm2' else [1,128,256,512,768,1024,1280,1536,1792,2048,2304,2400])
 elif model=='dnabert2':records=list(csv.DictReader(Path('/home/hyein/dnabert2-benchmark/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv').open()));sizes=[8,512,640,768,800,816,832,848,864,880,896,912,928,944,960,976,992,1000]
 elif model in ['ntv2','openspliceai']:
  records=json.loads((R/'data/real_training_v3/inference_inputs.json').read_text());sizes=([8,128,256,384,448,512,544,560,576,592,608,624,640] if model=='ntv2' else [8,512,1024,2048,4096,6144,8192,9216,10240,11264,12288,13312,14336,15360,16000])
 elif model=='scvi':records=[];sizes=[128,4096,8192,10176]
 else:records=[];sizes=[]
 if records:
  assert len({r['sequence'] for r in records})==len(records);(D/'inputs.json').write_text(json.dumps(records));S['unique_available_sequences']=len(records)
 if model in ['esm2','dnabert2','ntv2','rnafm','openspliceai','scvi']:
  for b in sizes:
   if records and b>len(records):break
   if model=='esm2':source=R/'scripts/real_training_v3/inference_worker.py';cmd=[str(R/'.venv/bin/python'),'-u',str(source),str(D),str(b)]
   else:
    source=ROOT/'scripts/real_training_v3/inference_worker.py';cmd=['docker','run','--rm','--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','MODEL_KIND='+model,'-e','BATCH='+str(b),'-e','MPLCONFIGDIR=/work/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/work/.cache/numba','-e','HF_HOME=/work/.cache/huggingface','-e','HF_MODULES_CACHE=/work/.cache/modules','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1'];python='/work/.venv/bin/python'
    if model=='dnabert2':cmd+=['-v','/home/hyein/dnabert2-benchmark:/workspace:ro'];python='/workspace/miniforge/envs/dnabert2/bin/python'
    cmd+=['nvcr.io/nvidia/pytorch:26.06-py3',python,'-u','/support/real_training_v3/inference_worker.py']
   code=launch(D,'inference_b'+str(b),cmd,source)
   if code:S['failed'].append({'batch':b,'exit_code':code});S['stop_reason']='actual_worker_error_or_guard';break
   s=json.loads((D/f'batch_{b}/summary.json').read_text());r=s['conditions'][0];S['conditions'].append(r);S['largest_successful_batch']=b;(D/'summary.json').write_text(json.dumps(S,indent=2))
   if r['allocated_fraction_shared_ram']>=.8:S['stop_reason']='allocated80pct_shared_ram_reached';break
  else:S['stop_reason']='tested_real_input_pool_or_planned_range_limit'
 elif model=='clair3':
  old=R/'results/candidate_batch_20261028_185641_089149'
  for kind in ['Clair3_P','Clair3_F']:
   for suffix in ['_inputs.npy','_reference_first.npy']:
    os.symlink(old/(kind+suffix),D/(kind+suffix))
   pool=json.loads((old/'input_audit.json').read_text())[kind]['unique_feature_rows'];b=pool;source=R/'scripts/batch_followup/worker.py';cmd=['docker','run','--rm','--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','CANDIDATE_MODEL='+kind,'-e','BATCH='+str(b),'local/clair3-spark:9d1dd3f','/work/.venv/bin/python','-u','/work/scripts/batch_followup/worker.py'];code=launch(D,kind+'_full_real_pool',cmd,source)
   if code:S['failed'].append({'kind':kind,'exit_code':code});continue
   s=json.loads((D/f'{kind}_batch_{b}/summary.json').read_text());S['conditions'].append(s['condition'])
  S['stop_reason']='actual_unique_BAM_candidate_cache_limit; no synthetic candidates or pseudo truth labels'
 elif model=='colabfold':
  old=R/'results/vmap_fp32_20261028_190843_576148'
  for p in old.glob('*_processed_features.npz'):os.symlink(p,D/p.name)
  os.symlink(old/'input_audit.json',D/'input_audit.json');os.symlink(old/'batch_1',D/'batch_1');source=R/'scripts/batch_followup/worker.py';cmd=['docker','run','--rm','--gpus','all','--network','none','--ipc=host','--user','1000:1000','-v',str(R)+':/work','-v',str(R/'cache/colabfold')+':/cache:ro','-e','RUN_ID='+run,'-e','BATCH=4','-e','JAX_PLATFORMS=cuda,cpu','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','ghcr.io/sokrypton/colabfold:1.6.2-cuda13','python','-u','/work/scripts/batch_followup/worker.py'];code=launch(D,'four_real_queries',cmd,source)
  if code:S['failed'].append({'batch':4,'exit_code':code})
  else:S['conditions']=json.loads((D/'batch_4/summary.json').read_text())['conditions']
  S['stop_reason']='four_actual_cached_MSA_query_pool_limit; no native training in installed ColabFold execution path'
 if model in ['esm2','dnabert2','ntv2','rnafm','openspliceai']:
  source=ROOT/'scripts/real_training_v3/sequence_training_worker.py';sizes={'esm2':[8,32,64,96,128,160,192,224,256,288,320,384,448,512,768,1024],'dnabert2':[8,32,48,64,72,80,88,96,104,112,120,128],'ntv2':[8,64,128,256,384,448,512],'rnafm':[8,32,64,96,128,160,192,224,256,288,320,384,448,512,640,768,896],'openspliceai':[8,32,64,96]}[model];safe=[];target=None;S['training_probes']=[]
  def traincommand(b,mode):
   env={'RUN_ID':run,'MODEL_KIND':model,'BATCH':str(b),'TRAIN_MODE':mode,'MPLCONFIGDIR':'/work/.cache/matplotlib','NUMBA_CACHE_DIR':'/work/.cache/numba','HF_HOME':'/work/.cache/huggingface','HF_MODULES_CACHE':'/work/.cache/modules','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
   if model=='esm2':
    env.update(MODEL_ROOT=str(R),SUPPORT_PATH=str(ROOT/'scripts'));return ['env']+[k+'='+v for k,v in env.items()]+[str(R/'.venv/bin/python'),'-u',str(source)]
   cmd=['docker','run','--rm','--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro'];python='/work/.venv/bin/python'
   if model=='dnabert2':cmd+=['-v','/home/hyein/dnabert2-benchmark:/workspace:ro'];python='/workspace/miniforge/envs/dnabert2/bin/python'
   for k,v in env.items():cmd+=['-e',k+'='+v]
   return cmd+['nvcr.io/nvidia/pytorch:26.06-py3',python,'-u','/support/real_training_v3/sequence_training_worker.py']
  for b in sizes:
   code=launch(D,'training_probe_b'+str(b),traincommand(b,'probe'),source)
   if code:S['training_probe_stop']={'batch':b,'exit_code':code};break
   w=json.loads((D/f'probe_training_b{b}/summary.json').read_text());r=w['training'];S['training_probes'].append(r);safe.append(b)
   if r['allocated_fraction_shared_ram']>=.8:target=b;break
  S['training80_batch']=target;S['full_training_cases']=[]
  for b in sorted(set(([8] if safe else [])+([target] if target else ([max(safe)] if safe and max(safe)>8 else [])))):
   code=launch(D,'full_training_b'+str(b),traincommand(b,'full'),source)
   S['full_training_cases'].append({'batch':b,'exit_code':code})
   if not code:
    w=json.loads((D/f'full_training_b{b}/summary.json').read_text());S['full_training_cases'][-1]['summary_path']=str(D/f'full_training_b{b}/summary.json')
  (D/'summary.json').write_text(json.dumps(S,indent=2))
 S['reached80']=any(r.get('allocated_fraction_shared_ram',0)>=.8 for r in S['conditions']);S['status']='stopped' if S['failed'] else 'measured';(D/'summary.json').write_text(json.dumps(S,indent=2));subprocess.run([sys.executable,str(ROOT/'scripts/real_training_v3/update_reports.py')]);state=json.loads(P.read_text());done=state.get('additional_models_measured',[]);done.append({'model':model,'run':run,'status':S['status'],'reached80':S['reached80']});update(additional_models_measured=done);print('MODEL_DONE',model,S['status'],flush=True)
update(status='all_ten_additional_runs_finished_review_required',active_model=None,active_stage=None)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: update_reports_305433d7fed68d6f.py</summary>

<!-- RESTORE_FILE {"path": "clair3/results/real_input_capacity_20261102_173336_910773/reproduction_sources/update_reports_305433d7fed68d6f.py", "sha256": "305433d7fed68d6ff20850b0d00d18708a0de3d92b36924d41dbbfe168034b8e", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,re,hashlib,shutil
ROOT=Path('/home/hyein/ai-benchmark');PUB=Path('/home/hyein/ai-benchmark-publication');mapping={'scgpt':'scGPT.md','cellpose':'Cellpose.md','esm2':'ESM-2.md','colabfold':'AlphaFold2.md','scvi':'scVI.md','dnabert2':'DNABERT-2.md','ntv2':'Nucleotide_Transformer_V2.md','clair3':'Clair3.md','rnafm':'RNA-FM.md','openspliceai':'OpenSpliceAI.md'}
def f(v):return 'N/A' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
def block(path):
 b=path.read_bytes();s=b.decode();meta={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest(),'encoding':'utf-8','added_final_newline':not s.endswith('\n')};n=max([3]+[len(m.group())+1 for m in re.finditer(r'`{3,}',s)]);fence='`'*n
 return '\n<!-- RESTORE_FILE '+json.dumps(meta)+' -->\n'+fence+('python' if path.suffix=='.py' else 'json')+'\n'+s+('' if s.endswith('\n') else '\n')+fence+'\n<!-- END_RESTORE_FILE -->\n'
for model,filename in mapping.items():
 pointer=ROOT/model/'results/REAL_TRAINING_LATEST'
 if not pointer.exists():continue
 D=pointer.parent/pointer.read_text().strip();rows=[]
 for p in sorted(D.glob('*/summary.json')):
  s=json.loads(p.read_text());r=s.get('training') or s.get('condition') or next(iter(s.get('conditions',[])),None)
  if not r:continue
  mem=r.get('memory',{})
  def pick(d,*keys):return next((d[k] for k in keys if d.get(k) is not None),None)
  score={k:v for k,v in s.get('test',{}).items() if k in ['accuracy','micro_f1','macro_f1','macro_f1_present_known_classes','mse','spearman','acceptor_donor_AUPRC','AP50_AP75_AP90']}
  method='추론(가중치 갱신 없음)'
  if s.get('training'):
   method='처음부터 학습' if model=='scvi' else '본체·분류기 파인튜닝' if model=='scgpt' else '본체·분할출력층 파인튜닝' if model=='cellpose' else '전체backbone·과제head 파인튜닝'
   if 'probe' in p.parent.name:method+=' /용량측정'
  values=[p.parent.name,method,s.get('status'),pick(r,'unique_training_cells','unique_train_images','unique_inputs','unique_candidates','unique_queries'),pick(r,'batch_size','batch') or s.get('batch'),r.get('epochs',s.get('epochs')),pick(r,'optimizer_steps','steps'),r.get('seconds'),pick(r,'cells_per_second','crops_per_second_trainer_scope','inputs_per_second','sequences_per_second','candidates_per_second','queries_per_second'),mem.get('peak_allocated_gib'),mem.get('peak_reserved_gib'),r['allocated_fraction_shared_ram']*100 if 'allocated_fraction_shared_ram' in r else None,r.get('gpu_util_avg'),r.get('gpu_util_max'),score or None,s.get('saved_model_reload_verified',s.get('reload_mask_equal'))]
  rows.append('|'+ '|'.join(map(f,values))+'|')
 text='<!-- REAL_TRAINING_V3_START -->\n### 실제 데이터 추가 실험 — '+D.name+'\n\n진행 중인 실행의 완료 조건만 아래에 갱신한다. probe는 짧은 용량 측정이며 전체 학습이나 평가 완료가 아니다. 메모리%는 CUDA allocated peak / 시스템 공유 RAM ×100이며 GPU-Util과 다르다. 독립 실행 1회 기준.\n\n|조건|가중치 갱신 방식·실험 목적|상태|고유 입력|배치|epoch|optimizer steps|실측 시간(s)|입력/s|allocated peak GiB|reserved peak GiB|공유 RAM 대비 allocated %|GPU-Util 평균%|최대%|독립 test 지표|저장 모델 재로드|\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'+'\n'.join(rows)+'\n\n'
 if model=='scgpt':text+='입력은 Tabula Sapiens 실제 고유 30만 세포에서 donor 단위로 분리한 학습238,868 / 검증35,738 / test25,394세포이다. 학습 donor의 실제 cell_type161종을 정답으로 사용한다. 새 분류 헤드161개는 기존 pancreas13클래스 실험과 별도 조건이다. 이 실행의 전체 학습 비교는 동일한 고유1만 세포,1epoch,FP32,최대1200토큰(CLS 포함),실제 배치 최장 길이 padding,동일 초기 backbone/seed42/AdamW에서 배치만 바꾼다. 검증2,000 / test5,000세포 고정 부분집합을 평가하며 미등장 라벨은 제외 수를 별도 저장한다. 같은 epoch에서도 배치별 optimizer step수가 달라 정확도 차이를 순수 배치 성능으로 단정하지 않는다. 학습 시간은 주기적 체크포인트 저장 포함이며 순수 step 시간은 원본 summary에 따로 기록한다.\n\n'
 if model not in ['scgpt','cellpose']:text+='이 run의 추론 capacity 실험과 지도학습 실험은 입력 과제·길이·학습 헤드가 다를 수 있어 각각 비교한다. 3step probe는 전체 학습이 아니다. full_training은 동일한 공개 초기 backbone와 seed42,FP32,AdamW(lr1e-5,weight_decay0.01),고정 전체 split pool padding으로2epoch 수행한다. 각 학습 조건의 원본 summary에 split ID,실제 길이·padding 모양,사용 헤드·학습 범위,라벨,gradient가 관측된 파라미터 수,검증/test지표,저장 모델 재로드 차이를 기록한다. scVI는 아래 설명이 적용되지 않는 별도 처음부터 학습 모델이며 AF2·Clair3에는 이 공통 학습 절차를 적용하지 않는다.\n\n'
 if model=='dnabert2':text+='새 보완 학습은 기존 BEND train1000 안에서 seed42로800/100/100을 분리한다. 이는 기존 독립 test 평가를 대체하지 않으며 염색체 독립 평가가 아니다. 18개 다중 라벨,masked-mean+linear head,전체 backbone 학습이다.\n\n'
 if model=='esm2':text+='새 학습은 TAPE fluorescence의 실제 GFP train1024/valid128/test512를 고유 부분집합으로 사용한다. masked-mean+linear regression,학습 라벨 평균/표준편차로 정규화,전체 backbone 학습으로 이전 일부 계층 학습과 구분한다. 새 추론은 별도 UniProt reviewed900–950AA 실제 서열을 사용한다.\n\n'
 if model=='rnafm':text+='새 지도학습은 기존 RNAcentral 타입 라벨 train896/validation224/test280을 유지하고 전체 backbone+선형 분류기를 갱신한다. 기존 frozen backbone+head-only와 구분한다. 새 추론은 GENCODEv47 실제 lncRNA transcript 첫1000nt(T→U)2400개를 사용하며 타입 라벨 학습 데이터와 다르다.\n\n'
 if model=='ntv2':text+='새 학습은 기존 enhancer train512/validation128/test400,199–200nt 정답을 사용한다. 새 masked-mean+linear head는 이전 native CLS MLP head와 다른 별도 조건이며 이번 배치 비교 내에서는 고정한다. 추론은 실제 GRCh38 비중첩6000nt window1024개를 별도로 사용한다.\n\n'
 if model=='openspliceai':text+='추론은 GRCh38 실제 비중첩15000nt window16000개,기존5개 모델 앙상블이다. 학습은 기존 라벨이 검증된96/24/36개15000nt 입력→5000nt 정답,seed10단일 모델만 갱신한다. 비지도 reference window에 임의 정답을 만들지 않는다.\n\n'
 if model=='cellpose':text+='BBBC039 원본 이미지 단위 공식 분리100train/50validation/50test를 확인했다. 학습100장 중 정답 객체가 없는2장은 공식 min_train_masks5 정책으로 제외하여 실제 학습98장이다. 검증50장에는 빈 마스크1장이 있으며 test는50장이다. 원본520×696 uint16 grayscale와 정답 instance mask를 공식256×256 무작위 crop/회전/크기 증강으로 학습한다. FP32,AdamW,lr1e-5,weight_decay0.1,seed42 고정. 네트워크 입력은[B,3,256,256]이며 grayscale 외 두 채널은0이다. 공식 warmup의 epoch0 학습률0이므로 probe도2epoch 실행하고 실제 가중치 변화를 확인한다. 전체 학습 비교는10epoch이고 독립 test50장의 AP50/AP75/AP90이다. CUDA allocator budget100GiB로 캐시 예약이 공유 RAM을 잠식하는 것을 제한한 별도 정책이며 모든 배치에서 동일하다. 학습 시간에는 전처리/검증/체크포인트 저장이 포함된다. flow 준비는 별도 실행으로 재사용한다.\n\n'
 text+='입력 및 메모리 기준값(같은 실행에서 측정):\n\n|조건|실제 모델 입력 모양/정책|모델 적재 직후 allocated GiB|실행 시작 allocated GiB|peak 추가 allocated GiB|프로세스 RSS sampled peak GiB|시스템 가용 RAM sampled 최소 GiB|\n|---|---|---|---|---|---|---|\n'
 for p in sorted(D.glob('*/summary.json')):
  v=json.loads(p.read_text());q=v.get('training') or v.get('condition') or next(iter(v.get('conditions',[])),None)
  if not q:continue
  inp=v.get('input',{});shape=inp.get('actual_token_shape',inp.get('input_shapes',v.get('input_shape',v.get('training_input',v.get('fixed_pool_padded_shapes','상세 전처리 정책 참조')))))
  text+='|'+'|'.join(map(f,[p.parent.name,shape,v.get('model_only_postload',v.get('model_postload',{})).get('allocated_gib'),q.get('baseline',{}).get('allocated_gib'),q.get('peak_delta_gib'),q.get('rss_sampled_peak_gib'),q.get('available_sampled_min_gib')]))+'|\n'
 text+='\n'
 run_summary=D/'summary.json'
 if run_summary.exists():
  v=json.loads(run_summary.read_text());text+='실행 상태: `'+str(v.get('status','기록 확인'))+'`; 중단/관측 범위: `'+str(v.get('stop_reason','해당 조건별 상태 참조'))+'`.\n\n'
 if model=='colabfold':
  for pp in sorted(D.glob('batch_*/summary.json')):
   v=json.loads(pp.read_text());qq=next(iter(v.get('conditions',[])),{});jm=qq.get('jax_memory_stats',{});text+='JAX 계측 '+pp.parent.name+': 누적 process peak_bytes_in_use='+str(jm.get('peak_bytes_in_use'))+'bytes,실제 실행 구간 sampled peak_bytes_in_use='+str(qq.get('execution_sampled_peak_bytes_in_use'))+'bytes. 누적 peak에는 compile/warmup이 포함되며 CUDA allocated/reserved와 직접 동일시하지 않는다.\n\n'
 if model=='cellpose':text+='이 Cellpose run은 학습·test 평가·저장 모델 재로드가 완료됐다. 용량probe는 배치26까지90.5200GiB(공유 RAM74.39%) 성공,배치28은100GiB allocator budget에서OOM이 발생해 종료했다. 전체10epoch 학습은 배치1만 수행했으며 80% 학습 조건은 확인되지 않았다.\n\n'
 text+='실행 코드 블록의 파일은 원문SHA256으로 검증하며, 명령JSON에 기록된 실행 경로로 해당 snapshot을 복사하면 복원할 수 있다. 대용량 원본 데이터·가중치는 아래 전처리/다운로드 코드와 출처를 사용하고 로컬 원본도 보존한다.\n\n'
 text+='로컬 원시 결과: `'+str(D)+'`. 메모리 부족/guard 실패는 성공 조건으로 기재하지 않는다.\n<!-- REAL_TRAINING_V3_END -->\n'
 path=PUB/filename;old=path.read_text()
 if '<!-- REAL_TRAINING_V3_START -->' in old:old=re.sub(r'<!-- REAL_TRAINING_V3_START -->[\s\S]*?<!-- REAL_TRAINING_V3_END -->\n',lambda _:text,old)
 else:
  m=re.search(r'^## 4[.\s]',old,re.M);pos=m.start() if m else old.index('## 4');old=old[:pos]+text+'\n'+old[pos:]
 extra_sources=[ROOT/'scripts/gpu_support.py',ROOT/'scripts/real_training_v3/remaining_queue.py',ROOT/'scripts/real_training_v3/update_reports.py'] if model not in ['scgpt','cellpose'] else []
 if model in ['esm2','rnafm']:extra_sources += [ROOT/'scripts/real_training_v3/fetch_real_sequences.py',ROOT/'scripts/real_training_v3/fetched_inputs'/('uniprot_source.json' if model=='esm2' else 'gencode_source.json')]
 if model in ['ntv2','openspliceai']:extra_sources += [ROOT/'scripts/real_training_v3/prepare_genome_pools.py',ROOT/model/'data/real_training_v3/inference_audit.json']
 for source in sorted(D.glob('*_executed_source.py'))+sorted(D.glob('*_controller_source.py'))+sorted(D.glob('*_command.json'))+extra_sources:
  sha=hashlib.sha256(source.read_bytes()).hexdigest()
  if source in extra_sources:
   snapdir=D/'reproduction_sources';snapdir.mkdir(exist_ok=True);snapshot=snapdir/(source.stem+'_'+sha[:16]+source.suffix)
   if not snapshot.exists():shutil.copy2(source,snapshot)
   source=snapshot
  if ('"sha256": "'+sha+'"') not in old:old+='\n<details><summary>추가 실험 실행 원문: '+source.name+'</summary>\n'+block(source)+'\n</details>\n'
 path.write_text(old);shutil.copy2(path,ROOT/filename);print(model,'report updated',len(rows),'rows')

# Keep the model/input/learning-scope overview consistent as every new run finishes.
import subprocess,sys
subprocess.run([sys.executable,str(ROOT/'scripts/real_training_v3/standardize_overview.py')],check=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>단계별 입력·출력 확인/보고 원문: trace_special_cpu.py</summary>

<!-- RESTORE_FILE {"path": "scripts/report_stages/source_snapshots/trace_special_cpu_04aa435d868170c0.py", "sha256": "04aa435d868170c035402ba565423c90588e061e0e45f8d170b8f1bf510b1b49", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,gc,csv
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
kind=os.environ['MODEL_KIND'];R=Path('/work');torch.set_num_threads(8);S={'model':kind,'provenance':'New CPU-only saved-model replay; no new GPU benchmark or parameter update'};arrays={}
if kind=='scvi':
 import scvi
 P=R/'models/scvi_20261027_185327_767988/train_10176';m=scvi.model.SCVI.load(str(P),accelerator='cpu');m.module.eval();ad=m.adata;O=R/'results'/(R/'results/REAL_TRAINING_LATEST').read_text().strip()/'stage_trace_cpu';O.mkdir(exist_ok=True);x=ad.layers['counts'][:1];x=x.toarray() if hasattr(x,'toarray') else np.asarray(x);capture={}
 def hook(mod,args,out):capture['encoder_hidden']=out.detach().cpu().numpy()
 h=m.module.z_encoder.encoder.register_forward_hook(hook);z=m.get_latent_representation(adata=ad[:1].copy(),batch_size=1,give_mean=True);h.remove();ref=np.load(O.parent/'batch_128/outputs.npy')[0];arrays.update(raw_counts=x,encoder_hidden=capture['encoder_hidden'],latent=z);ix=np.flatnonzero(x[0]);S.update(sample_id=str(ad.obs_names[0]),raw_matrix_shape=list(ad.shape),gene_names_first_nonzero=ad.var_names[ix[:8]].tolist(),counts_first_nonzero=x[0,ix[:8]].tolist(),batch_covariate=str(ad.obs['sample_id'].iloc[0]),latent=z[0].tolist(),reference_max_abs=float(np.max(np.abs(z[0]-ref))),saved_model=str(P.relative_to(R)),target='count reconstruction; no external cell-type label')
elif kind=='cellpose':
 import tifffile
 from skimage.io import imread
 from skimage.measure import label
 from cellpose import models
 D=R/'results'/(R/'results/REAL_TRAINING_LATEST').read_text().strip();C=D/'train_b1_e10';old=json.loads((C/'summary.json').read_text());rows=json.loads((R/'data/input_audit.json').read_text())['all'];row=next(r for r in rows if r['id']==old['split_ids']['test'][0]);raw=tifffile.imread(R/row['image']);gt=label(imread(R/row['source_mask'])[:,:,0],background=0).astype('int32');O=C/'stage_trace_cpu';O.mkdir(exist_ok=True);m=models.CellposeModel(gpu=False,device=torch.device('cpu'),pretrained_model=old['checkpoint'],use_bfloat16=False);capture={}
 def pre(mod,args):
  if 'first_model_input' not in capture:capture['first_model_input']=args[0].detach().cpu().numpy()
 def patch(mod,args,out):
  if 'first_patch_features' not in capture:capture['first_patch_features']=out.detach().cpu().numpy()
 a=m.net.register_forward_pre_hook(pre);b=m.net.encoder.patch_embed.proj.register_forward_hook(patch);mask,flow,style=m.eval(raw,batch_size=1,bsize=256,tile_overlap=.1,augment=False,normalize=True,flow_threshold=.4,cellprob_threshold=0.,min_size=15,max_size_fraction=.4);a.remove();b.remove();gpu=np.load(C/'test_masks.npz')['0'];arrays.update(raw_image=raw,ground_truth_mask=gt,**capture,flows=flow[1],cell_probability=flow[2],predicted_mask=mask,saved_gpu_mask=gpu);S.update(sample_id=row['id'],raw_image_file=row['image'],raw_mask_file=row['source_mask'],first_feature_values=capture['first_patch_features'].ravel()[:8].tolist(),truth_instances=int(gt.max()),predicted_instances=int(mask.max()),first16_mask_ids=mask.ravel()[:16].tolist(),cpu_gpu_mask_equal=bool(np.array_equal(mask,gpu)),cpu_gpu_foreground_difference_pixels=int(np.count_nonzero((mask>0)!=(gpu>0))),saved_model=old['checkpoint'],note='CPU tile batch1 for this replay; original evaluated GPU tilebatch8. Not a new timing/accuracy sweep.')
elif kind=='scgpt':
 from scgpt.model import TransformerModel
 D=R/'results'/(R/'results/REAL_TRAINING_LATEST').read_text().strip();C=D/'train_b284_n10000';O=C/'stage_trace_cpu';O.mkdir(exist_ok=True);ck=torch.load(C/'final_model.pt',map_location='cpu',weights_only=True);model=TransformerModel(**ck['config'],vocab=ck['vocab']);model.load_state_dict(ck['model'],strict=True);model.float().eval();vocab=ck['vocab'];names={v:k for k,v in ck['labels'].items()};del ck;gc.collect();row=int(np.load(C/'test_rows.npy')[0]);P=R/'data_external/tabula_sapiens/prepared_300k_seed42';g=np.load(P/'gene_ids.npy',mmap_mode='r');values=np.load(P/'expression_bins.npy',mmap_mode='r');length=np.load(P/'lengths.npy',mmap_mode='r');width=int(length[row]);gg=torch.from_numpy(np.array(g[row:row+1,:width],np.int64));vv=torch.from_numpy(np.array(values[row:row+1,:width],np.float32));target=int(np.load(D/'labels.npy')[row]);pad=vocab['<pad>']
 with torch.no_grad():h=model._encode(gg,vv,src_key_padding_mask=gg.eq(pad));embed=h[:,0,:];logits=model.cls_decoder(embed);prob=logits.softmax(-1)
 ref=np.load(C/'test_logits.npy')[0];arrays.update(gene_ids=gg.numpy(),expression_bins=vv.numpy(),token_representations=h.numpy(),cls_embedding=embed.numpy(),logits=logits.numpy(),probabilities=prob.numpy(),target=np.asarray([target],np.int64));cell={}
 with (P/'cells.tsv').open() as f:
  for i,item in enumerate(csv.DictReader(f,delimiter='\t')):
   if i==row:cell=item;break
 inv={v:k for k,v in vocab.items()};S.update(sample_id=cell.get('cell_id',str(row)),cell_metadata=cell,prepared_row=row,raw_matrix_note='sourcecells x58482genes;preparedcache300000x1200;tokenization includes CLS/nonzero-gene selection/binning',first16_gene_ids=gg[0,:16].tolist(),first16_gene_names=[inv.get(int(i),'?') for i in gg[0,:16]],first16_expression_bins=vv[0,:16].tolist(),embedding_first8=embed[0,:8].tolist(),predicted_id=int(prob.argmax(1)),predicted_label=names.get(int(prob.argmax(1))),true_id=target,true_label=names.get(target),top_probability=float(prob.max()),reference_max_abs=float(np.max(np.abs(logits.numpy()[0]-ref))),saved_model=str((C/'final_model.pt').relative_to(R)))
elif kind=='clair3':
 import sys
 sys.path.insert(0,str(R/'source'));from clair3.model import Clair3_P,Clair3_F
 D=R/'results'/(R/'results/REAL_TRAINING_LATEST').read_text().strip();O=D/'stage_trace_cpu';O.mkdir(exist_ok=True);results={}
 for name,ctor,file in [('Clair3_P',Clair3_P,'pileup.pt'),('Clair3_F',Clair3_F,'full_alignment.pt')]:
  net=ctor(predict=True,add_indel_length=name=='Clair3_F').float().eval();ck=torch.load(R/'models/ont'/file,map_location='cpu',weights_only=False);net.load_state_dict(ck.get('state_dict',ck));x=np.array(np.load(D/(name+'_inputs.npy'),mmap_mode='r')[:1]);
  with torch.no_grad():pred=net(torch.from_numpy(x)).numpy()
  arrays[name+'_input']=x;arrays[name+'_probabilities']=pred;sub=next(D.glob(name+'_batch_*/predictions.npy'));ref=np.load(sub,mmap_mode='r')[0];results[name]={'shape':list(x.shape),'dtype':str(x.dtype),'input_first16':x.ravel()[:16].tolist(),'output_shape':list(pred.shape),'probability_first24':pred[0,:24].tolist(),'head21_argmax':int(pred[0,:21].argmax()),'genotype3_argmax':int(pred[0,21:24].argmax()),'reference_max_abs':float(np.max(np.abs(pred[0]-ref)))}
 S.update(branches=results,sample_id='actual cached candidate row0 in each branch; no proven cross-branch position mapping',note='Candidate cache did not retain genomicposition IDs; do not claim these two rows or a shown VCF record represent the same genomic locus.',raw_input_files=['data/HG003_chr20_demo.bam','data/GRCh38_no_alt_chr20.fa'])
S['arrays']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()};np.savez_compressed(O/'arrays.npz',**arrays);S['saved_arrays']=str((O/'arrays.npz').relative_to(R));(O/'trace.json').write_text(json.dumps(S,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)));print(kind,'trace_saved',str(O),flush=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>단계별 입력·출력 확인/보고 원문: build_reports.py</summary>

<!-- RESTORE_FILE {"path": "scripts/report_stages/source_snapshots/build_reports_35f8ba6f14095856.py", "sha256": "35f8ba6f140958564b6f078d83efd13d998a0c6855ffdcaebb07c3c824f48501", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import json,re,hashlib,shutil,platform
ROOT=Path('/home/hyein/ai-benchmark');PUB=Path('/home/hyein/ai-benchmark-publication');MAP={'dnabert2':'DNABERT-2.md','ntv2':'Nucleotide_Transformer_V2.md','rnafm':'RNA-FM.md','openspliceai':'OpenSpliceAI.md','scvi':'scVI.md','cellpose':'Cellpose.md','scgpt':'scGPT.md','clair3':'Clair3.md','colabfold':'AlphaFold2.md','esm2':'ESM-2.md'}
def j(x):return '```json\n'+json.dumps(x,ensure_ascii=False,indent=2)+'\n```\n'
def vector(x):return '['+', '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in x)+']'
def shape(s,k):
 x=s['arrays'][k];return str(x['shape'])+' '+x['dtype']
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join('---' for _ in headers)+'|\n'+''.join('|'+ '|'.join(str(v).replace('|','／') for v in row)+'|\n' for row in rows)+'\n'
def artifact(s,items):return table(['단계별 저장 배열','실제 shape·dtype','의미'],[(k,shape(s,k),desc) for k,desc in items])+'로컬 전체 배열: `'+s['saved_arrays']+'`. `arrays.npz::key`는 같은NPZ 파일 안의 배열이름이다. 아래 재현코드로 저장모델과 입력에서 다시 확인할 수 있다.\n\n'
def cpu_note(s):
 d=s.get('saved_gpu_output_max_abs_difference',s.get('reference_max_abs'))
 return '**값의 출처:** 저장된 모델을 CPU에서 새로 재확인한 중간값이다. 원GPU 자원계측 중에 캡처했던 것처럼 해석하지 않는다. '+(f'같은 저장GPU 출력과 최대 절대차이는 **{d:.3e}**이다. ' if d is not None else '')+'이 CPU 확인시간은 기존 GPU 벤치마크 시간에 포함하지 않는다.\n\n'
TRACES={}
for model,filename in MAP.items():
 if model=='esm2':continue
 r=ROOT/model;allp=list((r/'results').glob('**/stage_trace_cpu/trace.json'))+list((r/'results').glob('**/stage_trace_saved/trace.json'));assert allp,model;p=max(allp,key=lambda p:p.stat().st_mtime);s=json.loads(p.read_text());TRACES[model]=(p,s)
 t='### 실제 파일 → 전처리 → 모델 표현 → 최종 출력: 단계별 실제 예시\n\n'
 if model in ['dnabert2','ntv2','rnafm']:
  raw=s['raw_record'];classes={'dnabert2':18,'ntv2':2,'rnafm':28}[model];hidden={'dnabert2':768,'ntv2':1024,'rnafm':640}[model];task={'dnabert2':'18개 histone 표지 다중라벨 분류','ntv2':'enhancer 이진분류','rnafm':'RNA유형28종 분류'}[model]
  t+=f"이 예시는 **{task}**, 저장된 추가 전체backbone 파인튜닝 모델의 **test `{s['sample_id']}`**다. 아래 모든값은 같은 레코드에서 이어진다.\n\n"
  source={'dnabert2':'BEND CSV의 id,sequence,labels(원train1000 내부800/100/100분할)','ntv2':'data/inputs.json의 test: id,sequence,label','rnafm':'data/inputs.json의 test: id,sequence,type,label'}[model]
  t+='#### 0–1. 파일 레코드 — X와 정답 y\n\n'+source+'. 아래는 실제 레코드에서 앞80염기만 발췌한 표시이며, 모델에는 전체서열을 넣었다.\n\n'
  example={'id':s['sample_id'],'sequence_length':len(raw['sequence']),'sequence_first80':raw['sequence'][:80]}
  if model=='dnabert2':example['positive_label_indices']=json.loads(raw['labels']);example['target_18_multihot']=s['target']
  else:example['label']=raw['label'];example.update({'type':raw['type']} if 'type' in raw else {})
  t+=j(example)+'**X는 염기서열, y는 위 label**이다. 임베딩을 정답으로 쓰는 것이 아니다.\n\n'
  t+='#### 2. 토큰화 — 실제 모델 입력\n\n'+table(['항목','실제값'],[['tokens',shape(s,'input_ids')],['attention_mask',shape(s,'attention_mask')],['원서열 길이',str(s['raw_nt'])+'nt'],['유효token 수',s['valid_token_count']],['token ID 앞16개',vector(s['token_ids_first16'])]])
  t+=('DNA BPE는 몇nt가 몇token으로 바뀌는지 서열마다 달라진다.' if model=='dnabert2' else '6-mer 중심 토큰화로 염기수와token수가 다르다.' if model=='ntv2' else '염기당token+BOS/EOS이며, 원래전체pool의최대폭1022로padding해batch비교조건을고정했다.')+' 숫자ID는vocabulary index이고, 정답라벨번호와 다른 배열이다.\n\n'
  t+='#### 3. 본체 출력 — 위치별 표현\n\n`'+shape(s,'token_representations')+'`이며 각token에'+str(hidden)+'개 특징값이 생긴다. 첫위치의 앞8개 실제값: `'+vector(s['feature_first8'])+'`. 이 단계는 최종 class확률이 아니다.\n\n'
  t+='#### 4. Pooling — 서열당 벡터1개\n\n방식: **'+s['pooling']+'**. 출력 `'+shape(s,'embedding')+'`. 임베딩 앞8개 실제값: `'+vector(s['embedding_first8'])+'`.\n\n'
  t+='#### 5. 과제 head — 최종 예측으로 변환\n\n'+f'추가 파인튜닝의 linear head는 `{hidden}→{classes}`다. 기본보고서의 다른head/학습범위 조건과 구분한다.\n\n'
  t+=table(['단계','shape','의미'],[['head logits',shape(s,'logits'),'확률변환 전 실수'],['확률',shape(s,'probabilities'),'sigmoid:18개 독립확률' if model=='dnabert2' else f'softmax:{classes}개중한class의확률'],['판정',s['prediction'],'각확률>0.5를양성판정' if model=='dnabert2' else '최대확률class의ID']])
  t+='logits 앞8개: `'+vector(s['logits'][:8])+'`; 확률 앞8개: `'+vector(s['probabilities'][:8])+'`.\n\n'
  t+='#### 6. 정답과 비교\n\n'+table(['항목','실제값'],[['정답',s['target']],['예측',s['prediction']]])
  if model=='dnabert2':t+='정답은18개중복수양성인multi-hot이고, 예측도18개를독립판정한다. 이번추가통제실험의test micro-F1=0이며, 실행·재현검증을유효한분류성능달성으로해석하면안된다. 원래독립test와이번train내부분할test도구분한다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('input_ids','입력token'),('attention_mask','유효위치'),('token_representations','token별표현'),('embedding','서열당임베딩'),('logits','head출력'),('probabilities','class확률'),('target','정답')])+f"원GPU run의 `test_outputs.npy`는 **logits**이며 확률파일이 아니다. 값의 순서는 저장된 split_ids['test']와 일치한다.\n\n"+cpu_note(s)
 elif model=='openspliceai':
  raw=s['raw_record'];t+='**이 모델은 서열당임베딩벡터를분류하는방식이아니라 각염기위치의splice확률을출력한다.** 같은test구간 `'+s['sample_id']+'`를끝까지따라간다.\n\n'
  t+='#### 0–1. 실제 레코드와 정답\n\n`data/inputs.json`의 test 레코드다.\n\n'+j({k:raw[k] for k in ['id','chromosome','strand','genomic_input_start','genomic_input_end']})+'원레코드 X는20,000nt, 정답은가운데10,000위치의0/1/2주석이다. X를 `[2500:17500]`의15,000nt로, y를 `[2500:7500]`의5,000위치로대응되게선택했다. 0=nonsplice,1=acceptor,2=donor,-1=충돌주석무시다.\n\n'
  t+='#### 2. One-hot 입력\n\n'+shape(s,'onehot')+';채널순서는A/C/G/T, 앞12염기는 `'+s['input_first12_bases']+'`. 한위치가A이면 `[1,0,0,0]`이고 token ID로변환하는모델이아니다.\n\n'
  t+='#### 3–4. 합성곱 특징과 문맥 처리\n\n첫합성곱의실측특징은 `'+shape(s,'initial_conv_features')+'`;첫채널앞8개는 `'+vector(s['feature_first8'])+'`. 이후dilated residual CNN이넓은문맥을처리하며,단백질언어모델처럼전체서열을평균임베딩하진않는다.\n\n'
  t+='#### 5. 최종 위치별 출력\n\n출력은 **'+shape(s,'output_probabilities')+'**이다. 위치5,000개각각에3종확률이있다. 앞5위치실제확률(행=위치,열=nonsplice/acceptor/donor):\n\n'+j(s['first5_position_probabilities'])
  t+='#### 6. 같은 위치의 정답과 비교\n\n정답5,000개중라벨별개수: `'+str(s['target_counts'])+'`. 주석상splice위치의첫index들(출력window내0-based): `'+str(s['positive_target_positions_first8'])+'`. 전체구간한개에라벨하나를붙인과제가아니다. 평가는위치별acceptor/donor AUPRC등이며,seed10단일모델미세조정과5모델ensemble추론은별도조건이다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('onehot','DNA입력'),('initial_conv_features','첫CNN특징'),('output_probabilities','위치별3확률'),('target','각출력위치정답')])+'원GPU `test_outputs.npy`는 `[36,3,5000]`의확률이다.\n\n'+cpu_note(s)
 elif model=='scvi':
  t+='**scVI는처음부터학습한count 생성VAE이며,세포유형분류head를붙이지않았다.** 저장모델의 실제세포 `'+s['sample_id']+'`를사용한다.\n\n'
  t+='#### 0–1. 원본 파일과 행렬의 한 행\n\n`data/processed_counts.h5ad`와저장모델의`adata.h5ad`에raw-count가보관된다. 원본처리행렬은세포12,721×유전자2,000이다. 행1개=세포1개,열1개=유전자1개다.\n\n'+j({'cell_id':s['sample_id'],'nonzero_gene_names_first8':s['gene_names_first_nonzero'],'raw_counts_same_genes':s['counts_first_nonzero'],'biological_batch_sample_id':s['batch_covariate']})+'생물학적sample_id는GPU batch크기가아니다. 세포유형라벨을정답으로학습한모델도아니다.\n\n'
  t+='#### 2. 모델 입력\n\ncounts는 `'+shape(s,'raw_counts')+'`이며batch공변량을함께사용한다. Raw count를입력하고같은count의생성분포를학습한다.\n\n'
  t+='#### 3. Encoder 특징\n\n실제encoder hidden출력은 `'+shape(s,'encoder_hidden')+'`이다.\n\n'
  t+='#### 4. 세포별 latent\n\n`give_mean=True`로사후분포평균을반환했다. 실측 '+shape(s,'latent')+'의전체10개값은:\n\n```text\n'+vector(s['latent'])+'\n```\n\n'
  t+='#### 5–6. 출력과 학습목표\n\n이번latent추론의최종출력은세포당10차원벡터다. 별도분류기확률/세포유형이아니다. **훈련시에는**decoder의count분포와raw count를이용한재구성항및KL항으로VAE를학습한다. 따라서정답구조는원래count이고외부cell-type정답을만들지않았다. 이예제CPU추적은latent경로만확인했으며decoder중간값을저장했다고주장하지않는다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('raw_counts','원래count행'),('encoder_hidden','인코더중간표현'),('latent','10차원최종latent')])+'전체입력의GPU추론배열은각batch의 `outputs.npy`;이예제는그첫행과대조했다.\n\n'+cpu_note(s)
 elif model=='cellpose':
  t+='**Cellpose는이미지를평균임베딩하나로줄여분류하는모델이아니라 픽셀별flow·cell probability를만들어instance mask로변환한다.** 실제test이미지 `'+s['sample_id']+'`를사용한다.\n\n'
  t+='#### 0–1. 원본 이미지와 정답 mask\n\n'+table(['역할','실제파일','shape·dtype'],[['입력X',s['raw_image_file'],shape(s,'raw_image')],['정답y',s['raw_mask_file'],shape(s,'ground_truth_mask')]])+f"원본은520×696 흑백uint16 TIFF,정답PNG를공식방법으로연결성분instance-ID로변환하면핵 **{s['truth_instances']}개**다. mask값0은배경,양의정수각각은서로다른객체다.\n\n"
  t+='#### 2. 정규화·타일화 → 네트워크 입력\n\n공식백분위정규화와256×256타일화를적용한다. 이번CPU확인의첫실제입력은 **'+shape(s,'first_model_input')+'**. 추론경로의흑백1채널입력이고,파인튜닝의3채널crop(흑백+0채널2개)과구분한다. 원본520×696전체가한번에같은모양으로GPU에올라가는것은아니다.\n\n'
  t+='#### 3–4. SAM encoder의 공간 특징\n\n첫encoder block의실제출력은 **'+shape(s,'first_encoder_block_features')+'**. 공간격자에채널별특징이있고, `[1,1280]`같은이미지전체임베딩이최종결과인모델이아니다. 펼친첫8값: `'+vector(s['first_feature_values'])+'`.\n\n'
  t+='#### 5. 타일 합치기 → flow·cell probability\n\n'+table(['출력','shape·dtype','의미'],[['flow',shape(s,'flows'),'각픽셀의Y/X방향흐름'],['cell probability',shape(s,'cell_probability'),'cellprob_threshold를적용할네트워크점수;0~1로보정된확률로단정하지않음']])+ '타일결과를원래이미지크기에맞춰합친다. 이후flow dynamics·threshold·크기필터를거쳐픽셀을서로다른객체로묶는다.\n\n'
  t+='#### 6. 최종 instance mask와 정답 비교\n\n최종예측은 **'+shape(s,'predicted_mask')+'**,예측객체'+str(s['predicted_instances'])+'개다. 원래같은크기의정답mask와객체를매칭해AP50/AP75/AP90을계산한다. mask ID는객체식별번호이지세포유형라벨이아니다. 이예제CPU/GPU mask일치: **'+str(s['cpu_gpu_mask_equal'])+'**. 이번CPU타일batch는1이고원GPU평가는8이므로새속도비교로쓰지않는다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('raw_image','입력이미지'),('ground_truth_mask','정답객체mask'),('first_model_input','첫네트워크타일'),('first_encoder_block_features','encoder공간특징'),('flows','전체이미지flow'),('cell_probability','전체이미지점수'),('predicted_mask','최종객체IDmask'),('saved_gpu_mask','원GPU결과대조')])+cpu_note(s)
 elif model=='scgpt':
  t+='같은실제test세포 `'+str(s['sample_id'])+'`를사용한다. **세포1개=행렬1행이며,그행의유전자별발현값을token으로바꾼다.** 이번예제는Tabula Sapiens161클래스파인튜닝모델이고pancreas13클래스와별도다.\n\n'
  t+='#### 0–1. 원본 h5ad의 count 행\n\n원본Tabula Sapiens는483,152세포×58,482유전자이고원본`raw/X`에서실제선택세포한행을확인했다. 아래원본유전자index와count는같은순서다.\n\n'+j(s.get('raw_count_row',{'note':'원본count의새확인값없음'}))+'정답은세포metadata의cell_type이며,유전자ID나발현량이아니다.\n\n'
  t+='#### 2. 유전자 선택·발현 binning → 두 입력 배열\n\n공개vocabulary와매칭한유전자에서세포별nonzero gene을선택하고발현을bin으로변환한다. 최대1200자리에는CLS가포함된다. 이번세포의입력앞16개는:\n\n'+j({'gene_ids':s['first16_gene_ids'],'gene_names_same_order':s['first16_gene_names'],'expression_bins_same_order':s['first16_expression_bins']})+table(['배열','실측shape·dtype'],[['gene_ids',shape(s,'gene_ids')],['expression_bins',shape(s,'expression_bins')]])+'처음의CLS와특수발현값은일반유전자의raw count가아니다. bin값도원래UMI count와다른전처리결과다. 원본58482열전체를그대로1200열로복사한것이아니다.\n\n'
  t+='#### 3. token별 Transformer 표현\n\n실제출력: **'+shape(s,'token_representations')+'**. 각CLS/유전자token마다512개특징값이생긴다.\n\n'
  t+='#### 4. 세포 임베딩\n\nCLS위치의벡터를선택해 **'+shape(s,'cls_embedding')+'**로만든다. 앞8개실제값:`'+vector(s['embedding_first8'])+'`. 이분류경로는raw CLS를사용하며,별도임베딩추론의L2정규화와구분한다.\n\n'
  t+='#### 5. 분류 head\n\nhead logits는 **'+shape(s,'logits')+'**,softmax확률도 `[1,161]`이다. 세포당161개유형중최대확률유형을선택한다.\n\n'
  t+='#### 6. 실제 cell_type 정답과 비교\n\n'+table(['항목','실제값'],[['예측class',str(s['predicted_id'])+' /'+str(s['predicted_label'])],['예측최대확률',f"{s['top_probability']:.6f}"],['정답class',str(s['true_id'])+' /'+str(s['true_label'])]])+'전체test성능은별도5000세포지표이며,한세포예측결과가전체정확도를뜻하지않는다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('gene_ids','유전자ID입력'),('expression_bins','발현bin입력'),('token_representations','위치별특징'),('cls_embedding','세포임베딩'),('logits','class점수'),('probabilities','class확률'),('target','실제유형ID')])+'원GPU배열은 `test_logits.npy` `[5000,161]`와 `test_rows.npy`;같은행을재확인했다.\n\n'+cpu_note(s)
 elif model=='clair3':
  t+='**Clair3는단백질임베딩모델이아니라 정렬된read에서후보변이특징을만들어변이확률을계산한다.** 이예제는실제cache의첫후보이며,캐시에genomicposition ID가저장되지않았으므로BAM의특정read·표시된VCF행과동일좌표라고연결하지않는다.\n\n'
  t+='#### 0–1. 원본 입력과 평가 정답\n\n입력은 `'+ '`, `'.join(s['raw_input_files'])+'`및BAM index다. BAM의정렬read·염기품질·mapping정보와참조유전체에서후보특징을계산한다. GIAB truthVCF와평가구간은정확도평가용이며이번에학습한라벨tensor가아니다.\n\n'
  t+='#### 2. 후보 특징 tensor\n\n'+table(['모델분기','입력shape','dtype'],[(k,v['shape'],v['dtype']) for k,v in s['branches'].items()])+'pileup의33위치×18특징과full-alignment의89read행×33위치×8채널이다. batch는DNA문자수나원본read총수가아니라동시에처리하는후보tensor수다.\n\n'
  t+='#### 3–4. 분기별 특징 계산\n\npileup은양방향LSTM,full-alignment는CNN경로로후보의표현을계산한다. **이CPU확인에서는그내부hidden tensor를캡처하지않았으므로실제중간embedding값은N/A**다. 존재하지않는평균임베딩단계를추가하지않는다.\n\n'
  t+='#### 5. 최종 후보별 확률\n\n'
  for k,v in s['branches'].items():t+='- **'+k+'**:출력'+str(v['output_shape'])+',앞24개실제값 `'+vector(v['probability_first24'])+'`. 원GPU같은행과최대차이 '+f"{v['reference_max_abs']:.3e}"+'.\n'
  t+='\nP의24출력은GT21+genotype3, F의90출력은GT21+genotype3+indel길이33+33개head를연결한것이다. 전체90개를단일softmax90클래스로해석하지않는다.\n\n'
  t+='#### 6. 후보확률 → 실제 VCF·평가\n\n공식pipeline은후보별확률과위치·ref/alt정보를조합해VCF의CHROM/POS/REF/ALT/QUAL/GT등을작성한다. cache예제의좌표mapping이없어이확률과특정VCF행의직접대조는미검증이다. 원래BAM→VCF실행·truth평가결과는아래상세에보존했다. 훈련·파인튜닝은수행하지않았다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('Clair3_P_input','pileup후보입력'),('Clair3_P_probabilities','pileuphead확률'),('Clair3_F_input','full-alignment후보입력'),('Clair3_F_probabilities','full-alignmenthead확률')])+cpu_note(s)
 elif model=='colabfold':
  raw=s['raw_record'];t+='**AlphaFold2는단백질서열과MSA를받아구조를만든다. 평균임베딩뒤에회귀head를붙인ESM-2 과제와다르다.** 실제독립query `'+s['sample_id']+'`의저장된입력·출력을확인했다.\n\n'
  t+='#### 0–1. 원본 서열·MSA\n\n```fasta\n>'+s['sample_id']+'\n'+raw['sequence']+'\n```\n\n서열115AA,feature sampling전원MSA깊이'+str(raw['msa_depth_before_feature_sampling'])+'이다. 이query에는정답구조를학습시킨것이아니며,이번작업은추론이다.\n\n'
  t+='#### 2. 모델 입력 features\n\n'+table(['실제NPZ key','실측shape·dtype'],[(k,str(v['shape'])+' '+v['dtype']) for k,v in s['feature_shapes'].items() if k in ['aatype','target_feat','msa_feat','msa_mask','extra_msa','extra_msa_mask','residue_index']])+'recycle/ensemble feature축,MSA행,잔기축을원본서열개수나독립query batch와혼동하지않는다. zero-padding된MSA행은mask로제외한다.\n\n'
  t+='#### 3–4. Evoformer 표현·구조 모듈\n\nEvoformer가MSA와잔기쌍표현을처리하고structure module이원자좌표를계산한다. **원GPU실행은내부Evoformer embedding을저장하지않았으므로중간숫자는N/A**다. 이번확인은저장된features와최종예측배열을읽은것이며새JAX/GPU추론을실행한것이아니다.\n\n'
  t+='#### 5. 최종 구조·신뢰도 출력\n\n'+table(['출력','shape·dtype','실제예시'],[['atom_positions',str(s['output_shapes']['atom_positions']['shape'])+' '+s['output_shapes']['atom_positions']['dtype'],'첫잔기CA좌표 '+vector(s['first_residue_CA_xyz'])],['plddt',str(s['output_shapes']['plddt']['shape'])+' '+s['output_shapes']['plddt']['dtype'],'앞8값 '+vector(s['plddt_first8'])]])+'좌표의마지막3개값은X/Y/Z좌표이며,37원자slot중유효원자는residue별mask를따른다.\n\n'
  t+='#### 6. 정답·점수 해석\n\n평균pLDDT는 '+f"{s['mean_plddt']:.3f}"+'이며 **모델의신뢰도**다. 이query의정답실험구조가없어이를구조정확도로쓰지않는다. PDB1UBQ등정답구조대조는별도입력조건이다.\n\n'
  t+='#### 7. 저장 파일\n\n`'+s['pdb_file']+'`의실제처음5개ATOM레코드:\n\n```text\n'+'\n'.join(s['first5_PDB_ATOM_lines'])+'\n```\n\n저장최종NPZ는atom_positions/plddt이며준비features와함께 `'+s['saved_arrays']+'`에사본을보관했다. 별도3D좌표/PDB출력이지class라벨이아니다.\n\n'
 templates=ROOT/'scripts/report_stages/templates';templates.mkdir(exist_ok=True)
 (templates/(model+'.txt')).write_text(t)
# Record actual replay source snapshots and insert environment/measurement overview into all ten existing reports.
for model,filename in MAP.items():
 p=PUB/filename;s=p.read_text();rr=ROOT/model/'results';env={};source_path='상세원본환경기록 참조'
 candidates=[]
 for pointer in ['REAL_TRAINING_LATEST','LATEST']:
  pp=rr/pointer
  if pp.exists():
   dd=rr/pp.read_text().strip();candidates+=sorted(dd.glob('*/summary.json'))+[dd/'summary.json']
 candidates+=list(rr.glob('*/summary.json'))
 for q in candidates:
  if q.exists():
   v=json.loads(q.read_text())
   if isinstance(v.get('environment'),dict) and v['environment']:env=v['environment'];source_path=str(q.relative_to(ROOT));break
 envrows=[('장비·공유메모리','DGX Spark,NVIDIA GB10,ARM64;시스템공유RAM121.6875GiB. CPU/GPU RAM을별도로더하지않음'),('환경기록출처',source_path),('Python·프레임워크',str({k:v for k,v in env.items() if k in ['python','torch','jax','transformers','cuda','scgpt','cellpose','platform','precision']}) or 'N/A;5절의환경및실행코드참조'),('실험병렬정책','GPU벤치마크한번에1개. 이단계별CPU검증은별도작업이며GPU측정시간과합치지않음'),('GPU-Util','nvidia-smi utilization.gpu(%);관측기간중커널활동시간비율. 최대FLOPS/메모리비율아님'),('GPU메모리','PyTorch allocated/reserved및peak;JAX는bytes_in_use/peak_bytes_in_use별도. 모델적재직후·실행구간을분리'),('프로세스·시스템메모리','psutil RSS,MemAvailable,swap;시계열의sampled peak/최소값이며정확한연속최고치아님'),('표본간격','기본0.5초대기+GPU조회시간;측정구간의평균·최대·표본수저장. 짧은실행은동일조건반복하고고유입력과반복처리량구분'),('시간동기화','PyTorch구간앞뒤CUDA synchronize;JAX block_until_ready. 비동기호출반환시간만측정하지않음'),('시간포함범위','조건별로입력준비/전송/forward/backward/optimizer/validation/save포함여부를기록. 서로다른범위의시간을직접배수비교하지않음'),('loading·compile·warmup','구간을분리한다. JAX누적allocator peak가compile/warmup을포함하는경우본문에명시하고CUDA구간peak와직접동일시하지않음'),('메모리기준값과증가량','첫forward전모델적재기준값과같은run의실행peak를비교. 증가량에는학습gradient/optimizer/중간활성값이포함'),('누락·관측한계','미지원/미관측값은N/A. 장치전체GPU-Util에는다른프로세스·디스플레이활동이섞일수있음. 단일실행은통계적반복검증으로쓰지않음')]
 if model=='colabfold':envrows.append(('JAX메모리해석','XLA사전할당비활성. bytes_in_use와누적peak를분리하며미지원CUDA allocated/reserved는N/A'))
 content='<!-- ENV_MEASUREMENT_OVERVIEW_START -->\n### 실행 환경·자원 측정 방법 요약\n\n'+table(['항목','이번 보고서의 측정 기준'],envrows)+'<!-- ENV_MEASUREMENT_OVERVIEW_END -->\n\n'
 if '<!-- ENV_MEASUREMENT_OVERVIEW_START -->' in s:s=re.sub(r'<!-- ENV_MEASUREMENT_OVERVIEW_START -->[\s\S]*?<!-- ENV_MEASUREMENT_OVERVIEW_END -->\n*',lambda _:content,s)
 else:
  a=s.index('## 2.');i=s.index('\n',a)+1;s=s[:i]+'\n'+content+s[i:]
 p.write_text(s);shutil.copy2(p,ROOT/filename)
print('Generated nine model-specific stage sections and environment/measurement tables for all ten')
````
<!-- END_RESTORE_FILE -->

</details>

<details><summary>단계별 입력·출력 확인/보고 원문: standardize_overview.py</summary>

<!-- RESTORE_FILE {"path": "scripts/report_stages/source_snapshots/standardize_overview_766e505c1dc769df.py", "sha256": "766e505c1dc769df028985eb29add8c42d3e1a70c2e6ade11d912ddc2cec09ab", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,re,shutil
ROOT=Path('/home/hyein/ai-benchmark');PUB=Path('/home/hyein/ai-benchmark-publication')
META={'esm2': ('ESM-2.md', 'esm2_t33_650M_UR50D', 'Transformer encoder,33층,1280차원', '651,043,254개(651.04M);encoder649,400,320개', 'FP32', '단백질 서열 →1280차원 임베딩 / GFP 형광값 예측', '추론:UniProt reviewed 단백질. 추가 run은900–950AA이며 GFP 학습 서열과 별개', 'TAPE fluorescence의 실제 GFP 변이체 약237AA. 기본 train128/val64/test128;추가 비교1024/128/512', 'X:아미노산→int64 tokens[B,L+2];y:GFP log_fluorescence[B]. 라벨 표준화에는 train 평균·표준편차만 사용', '기본:일부 backbone 층+회귀head 파인튜닝;추가:전체 backbone+linear 회귀head 파인튜닝', 'data/fluorescence/fluorescence_{train,valid,test}.lmdb', '긴 UniProt 추론 서열에는 본 실험의 GFP 정답이 없다. 부분 미세조정과 전체 미세조정 수치를 섞지 않는다'), 'colabfold': ('AlphaFold2.md', 'ColabFold1.6.2,alphafold2_ptm/alphafold2_multimer_v3', 'Evoformer+Structure Module', '실제PTM tree92,907,930개(92.91M)/Multimer93,223,705개(93.22M);원본NPZ 구성은 상세표 참조', '기본 BF16;별도 실험vmap FP32', '단백질 서열+MSA →3D좌표,pLDDT/PAE', '서로 다른 UniProt115AA 단백질4개,PDB1UBQ/1LYZ/1TIM;복합체3G5O A/B 예제', '훈련·파인튜닝 미실시. 이번 작업에 사용한 학습 데이터셋 없음', '추론 X:서열·MSA features. 참고 PDB는 평가용 정답 구조이며 학습에 사용하지 않음', '추론만 수행. 설치된 ColabFold 실행 경로에서 훈련하지 않음', 'data/queries.fasta;각run processed_features.npz/MSA', 'pLDDT는 신뢰도이며 실제 정답 구조 대비 정확도와 다르다. vmap은 별도 실험적 추론 경로'), 'scvi': ('scVI.md', 'SCVI,seed42무작위초기화;사전학습미사용 checkpoint', 'ZINB count 생성 VAE,hidden128,1층,latent10', '1,293,526개(1.294M);latent z_encoder258,964개;첫backward gradient778,884개', 'FP32', 'raw-count 세포×유전자+sample_id →10차원 latent /count 분포', 'PBMC4k+8k raw count에 저장 모델을 적용해 latent 추출. 세포유형 분류 과제 아님', '실제10x PBMC4k+8k12721세포×2000HVG. train10176/validation1272/test1273;작은 중첩 train부분집합도 별도 비교', 'X:counts[B,2000];재구성 목표는 같은 raw count. sample_id는 실제 생물학적batch 공변량. cell-type 정답 없음', '사전학습 가중치 없이 무작위 초기화에서 처음부터 학습,400epoch. 파인튜닝 아님', 'data/processed_counts.h5ad;results/<input_run>/splits.json', '재구성 기반 비지도 학습이며 생물학적batch와 GPU batch는 다르다. 통합 진단을 세포유형 정확도로 부르지 않음'), 'cellpose': ('Cellpose.md', 'Cellpose4.2.1,cpsam_v2（Cellpose-SAM）', 'SAM ViT-L 기반 이미지 분할', '304,628,930개(304.63M)', '추론BF16;파인튜닝FP32', '흑백 이미지 →핵/세포별 서로 다른ID의 instance mask', 'BBBC039200장520×696 uint16 핵 형광 이미지. 뇌 추가 입력은 NucMM-M192³ micro-CT patch4개', 'BBBC039 공식100/50/50;기본부분집합16/8/16. 추가 학습은 빈mask2장 제외 후 train98/validation50/test50', 'X:원본520×696에서 공식 무작위crop[B,3,256,256](흑백+0채널2개). y:같은 crop에 맞춰 변환한 instance mask·cell probability·flow. 세포유형 정답 아님', 'cpsam_v2 사전학습 가중치에서 backbone·분할출력층 파인튜닝. 실제 일부 파라미터는 gradient가 관측되지 않아 수를 별도 기록', 'data/input_audit.json;data/images/images/*.tif;data/masks/masks/*.png;추가run/prepared/*_mask.npy', 'batch26은2epoch 파인튜닝 용량 측정;전체10epoch 독립test·재로드 완료는batch1. 뇌patch는zero-shot 추론만 수행'), 'dnabert2': ('DNABERT-2.md', 'DNABERT-2-117M,revision7bce263b15377fc15361f52cfab88f8b586abda0', 'DNA BPE tokenizer+Transformer encoder,hidden768', 'encoder117,068,544개(117.07M);18출력linear head13,842개 추가', 'FP32', 'DNA →BPE tokens →임베딩 /18종histone 다중라벨', '실제BEND GRCh38구간.512/1000/2000/4000bp 비교;bp수와token수는 별개', 'BEND histone_modification 실제구간 및18개표지. 기본train1000. 추가비교는 원train1000 안에서 seed42로800/100/100 분리', 'X:input_ids/attention_mask[B,Ltokens]. y:[B,18] float32 multi-hot이며18중 하나를 고르는 과제 아님. 신규추론4000bp pool padding851tokens', '공개encoder+linear 다중라벨head 전체 파인튜닝. 사용하지 않는pooler 등 분기와 실제gradient 수는 별도 기록', '/home/hyein/dnabert2-benchmark/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv', '추가800/100/100은 기존train 내부 분할이며 원래 독립test 결과를 대체하지 않음. 염색체독립 검증 아님'), 'ntv2': ('Nucleotide_Transformer_V2.md', 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species', 'DNA6-mer 중심 tokenizer+ESM 구현 Transformer', 'MLM498,345,436개(498.35M);embedding backbone493,084,113개;기본CLS분류구성494,135,763개', 'FP32', 'DNA →token 임베딩 /enhancer 이진분류', 'GRCh38 실제긴구간. 추가추론:서로 겹치지 않는6000nt window1024개,약1001tokens', '실제enhancer 라벨 서열199–200nt. train512/validation128/test400', 'X:token_ids/attention_mask[B,실제token길이]. y:[B]정수0/1. 긴6000nt 추론 window에는 이 enhancer 정답이 없음', '기본:전체backbone+원생CLS MLP 파인튜닝. 추가:전체backbone+masked-mean linear head 파인튜닝,별도 조건', 'data/inputs.json의 train/validation/test;data/real_training_v3/inference_inputs.json추론전용', '긴 유전체 추론window와 짧은enhancer 학습 서열을 구분. 정확도는 해당 라벨·split에서만 평가'), 'clair3': ('Clair3.md', 'Clair3 v2.0.4,ONT pileup.pt/full_alignment.pt', 'pileup 양방향LSTM 및 full-alignment CNN 후보분류 네트워크', 'Clair3_P2,074,520개(2.075M);Clair3_F2,986,522개(2.987M);단계별 실행', 'FP32', 'long-read BAM+참조유전체 →후보특징 →SNV/Indel VCF', 'HG003 ONT 실제BAM+GRCh38 chr20소구간. cache 고유feature P5708/F894', '이번 작업에서 학습·파인튜닝 미실시. 본 실험용 훈련 정답tensor 없음', '추론 X:P[B,33,18],F[B,89,33,8]. GIAB truthVCF/평가구간은 변이검출 정확도 평가용이며 학습에 사용하지 않음', '공식학습기능 존재와 별개로 이번에는 사전학습ONT 가중치의 추론만 수행', 'data/HG003_chr20_demo.bam;data/GRCh38_no_alt_chr20.fa;candidate run/*_inputs.npy', '후보batch는read개수가 아님. 작은구간 후보pool을 전부 사용한 결과는 전장유전체 최대메모리 한계가 아님'), 'rnafm': ('RNA-FM.md', 'RNA-FM ncRNA,RNA-FM_pretrained.pth（fm0.2.2）', 'RNA Transformer12층,hidden640', 'backbone99,521,546개(99.52M);28종linear head17,948개', 'FP32', 'RNA →640차원임베딩 /RNA유형 분류', '기본:RNAcentral 실제RNA. 추가추론:GENCODEv47 lncRNA transcript 첫1000nt,T→U,고유2400개', 'RNAcentral1400서열/28종류. train896/validation224/test280. 실제길이가 다양하며 최대1020nt', '기본head-only X:동결RNA-FM에서 추출한embedding[B,640]. y:[B]28종 유형ID. 추가backbone파인튜닝 X:RNA tokens[B,L+2]와 같은유형정답', '기본:본체동결·분류기만학습. 추가전체backbone+linear 분류기 파인튜닝은 해당run의 실제 완료상태를 확인', 'data/inputs.json의 train/validation/test;data/input_audit.json의 label매핑;GENCODE추론전용', '기본head-only 메모리를 전체RNA-FM 미세조정 메모리로 쓰지 않음. 신규lncRNA 추론2400개는28종 지도학습셋과 별개'), 'openspliceai': ('OpenSpliceAI.md', 'MANE10000nt,seed10–145개모델;파인튜닝은seed10', '팽창합성곱 residual CNN,10000nt 문맥', '단일696,739개(0.697M);5모델합계3,483,695개(3.484M)', 'FP32', 'DNA4채널one-hot →위치별nonsplice/acceptor/donor 확률', '기본:GRCh38실제156구간. 추가추론:고유비중첩15000nt window16000개,5모델ensemble', 'exon주석을확인한 GRCh38영역 train96/validation24/test36,염색체별분리. 원레코드20000nt/정답10000nt→실사용15000nt/정답5000nt', 'X:[B,4,15000] A/C/G/T one-hot. y:[B,5000]정수0/1/2;충돌위치-1무시. 구간전체에 정답1개가 아니라 위치별정답', 'seed10 단일모델 전체파인튜닝. seed10–14의5모델ensemble은 추론용이며5개동시학습 아님', 'data/inputs.json의 sequence및labels_10000;source/openspliceai/variant/annotations/grch38.txt', '추가무라벨reference window를 임의정답으로 학습시키지 않음. 지도학습정답은exon주석에서 변환'), 'scgpt': ('scGPT.md', 'scGPT whole-human,best_model.pt', 'Transformer12층,hidden512;geneID+expression bin embedding', 'encoder50,278,400개(50.28M);기본추론구성51,856,898개;13클래스구성51,338,254개;161클래스구성은run참조', 'FP32', '세포×유전자발현 →유전자token/bin →512차원임베딩 /세포유형분류', 'pancreas테스트세포;Tabula Sapiens 실제고유30만세포embedding 추론. 전부미세조정한 것은 아님', 'pancreas기본 train1000/val2120/test4218;전체8480/2120/4218. Tabula donor분리238868/35738/25394 중 실제train10000/val2000/test5000 사용', 'X:gene_ids[B,L]int64와expression_bins[B,L]float32,L≤1200(CLS포함). y:[B]Celltype/cell_type유형ID. 유전자가 분류정답인 것은 아님', '공개encoder/value encoder/Transformer+분류기 파인튜닝. pancreas13종과Tabula161종은 별도분류설정. 미사용분기와실제trainable/gradient수를 구분', 'data/pancreas_demo/*.h5ad;data_external/tabula_sapiens/prepared_300k_seed42/{gene_ids,expression_bins}.npy;추가run/labels.npy및*_rows.npy', 'batch는동시에처리하는세포수,각세포는표현행렬1행.1200은CLS+유전자token자리이며 세포수가아님')}
def ko(s):return s
for model,meta in META.items():
 filename,checkpoint,arch,params,precision,io,infer,training,xy,method,paths,limit=meta
 def human_count(match):
  n=int(match.group(1).replace(',',''));human=f'약{n/100000000:.2f}억 개' if n>=100000000 else f'약{n/10000:,.1f}만 개';return match.group(0)+'('+human+')'
 params=re.sub(r'(\d{1,3}(?:,\d{3})+)개',human_count,params)
 path=PUB/filename;s=path.read_text();table='|항목|이번 실험에서 사용한 모델|\n|---|---|\n'
 for key,value in [('체크포인트·버전',checkpoint),('구조',arch),('전체·구성별 파라미터 수',params),('정밀도',precision),('모델 입력 → 출력',io),('가중치와 실행 메모리의 차이','파라미터 수×dtype bytes는 이론 가중치 용량이다. 실행 peak에는 입력·중간 활성값이 더해지고,파인튜닝에는 gradient/optimizer 상태도 추가된다.'),('메모리% 정의','CUDA allocated peak / CPU·GPU 공유 RAM121.6875GiB ×100; GPU-Util과 별개. AlphaFold2는 JAX 계측을 별도로 표시')]:table+='|'+key+'|'+ko(value)+'|\n'
 table+='\n### 실제 데이터와 학습 방식 구분\n\n|구분|데이터·실제 처리 내용|\n|---|---|\n'
 for key,value in [('추론 데이터',infer),('학습·검증·test 데이터',training),('학습 입력 X / 정답 y',xy),('가중치 갱신 방식',method),('원본·변환 입력 저장 위치','`'+paths+'`'),('해석 범위',limit)]:table+='|'+key+'|'+ko(value)+'|\n'
 table+='\n**용어:** 처음부터 학습=무작위 초기화에서 시작; 전체/일부 층 파인튜닝=공개 가중치에서 해당 부분 갱신; 분류기만 학습=본체를 동결하고 head만 갱신. **용량probe는 학습 방식이 아니라 짧은 자원 측정 목적**이며,실제로 수행한 가중치 갱신 방식과 함께 표기한다. 원본 전체 데이터 수와 해당 run에서 실제 사용한 부분집합 수를 구분한다.\n\n'
 template=ROOT/'scripts/report_stages/templates'/(model+'.txt')
 if template.exists():table+=template.read_text()
 elif (ROOT/model/'data/report_input_section.txt').exists():table+=(ROOT/model/'data/report_input_section.txt').read_text()
 # Bring measured post-load baselines forward; do not infer them from parameter counts.
 candidates=[];rr=ROOT/model/'results'
 for pointer_name in ['REAL_TRAINING_LATEST','BATCH_LATEST','LATEST']:
  pp=rr/pointer_name
  if pp.exists():
   dd=rr/pp.read_text().strip();candidates+=sorted(dd.glob('*/summary.json'))+[dd/'summary.json']
 found={}
 for qp in candidates:
  if not qp.exists():continue
  vv=json.loads(qp.read_text());base=vv.get('model_only_postload',vv.get('model_postload'))
  if not isinstance(base,dict):continue
  training_case=any(x in qp.parent.name for x in ['training_b','train_b','probe_b'])
  role='파인튜닝 적재 기준' if training_case else '추론/기본 모델 적재 기준'
  if model=='clair3':role+=' ('+str(vv.get('model','분기 미기록'))+')'
  if 'allocated_gib' in base:
   if role in found:continue
   found[role]=(f"{base['allocated_gib']:.6f}",str(base.get('reserved_gib','N/A')),str(base.get('rss_gib','N/A')),str(qp.relative_to(ROOT)))
  elif 'jax' in base and role not in found:
   value=base['jax'].get('bytes_in_use');found[role]=('JAX '+(f'{value/2**30:.6f}' if value is not None else 'N/A'),'N/A',str(base.get('system',{}).get('rss_gib','N/A')),str(qp.relative_to(ROOT)))
 table+='### 모델 적재 직후 — 첫 입력 forward 이전\n\n|적재 조건|allocated GiB(JAX는별도표기)|reserved GiB|프로세스 RSS GiB|원본 근거|\n|---|---|---|---|---|\n'
 for role,values in found.items():table+='|'+role+'|'+'|'.join(values)+'|\n'
 if not found:table+='|기준값 원본을 추가 확인 중|N/A|N/A|N/A|기존 상세 기록 참조;추정값으로 채우지 않음|\n'
 table+='\n위 값은 **가중치를 GPU에 적재하고 첫 입력을 계산하기 전의 프레임워크 할당량**이다. CUDA/JAX context·driver·시스템 전체 GPU사용량과 같지 않으며 CPU입력 준비 포함 여부는 해당 실행 코드를 따른다. **실행 peak와 모델 적재 기준값의 차이는 같은 run에서만 계산**한다. 파인튜닝 중 추가량에는 입력뿐 아니라 중간값·gradient·optimizer 상태가 포함된다. 서로 다른 조건/정밀도의 기준값을 하나의 모델 고유 메모리로 단정하지 않는다.\n\n'
 a=s.index('## 1. 모델과 실제 입력')+len('## 1. 모델과 실제 입력');b=s.index('## 2.',a);s=s[:a]+'\n\n'+table+s[b:]
 # Keep historical detailed results/code intact; add an up-to-date conclusion at section4.
 marker='<!-- MODEL_SUMMARY_START -->';end='<!-- MODEL_SUMMARY_END -->';summary='### 요약 결론\n\n- **학습 방식:** '+ko(method)+'.\n- **입력 구분:** '+ko(limit)+'.\n'
 pointer=ROOT/model/'results/REAL_TRAINING_LATEST'
 if pointer.exists():
  d=pointer.parent/pointer.read_text().strip();records=[]
  for p in d.glob('*/summary.json'):
   v=json.loads(p.read_text());q=v.get('training') or v.get('condition') or next(iter(v.get('conditions',[])),None)
   if v.get('status')=='complete' and q and 'peak_allocated_gib' in q.get('memory',{}):records.append((q['memory']['peak_allocated_gib'],p.parent.name,q))
  if records:
   peak,case,q=max(records,key=lambda x:x[0]);ratio=peak/q['memory']['system_total_gib']*100;summary+=f'- **최신 추가 run 최대 성공 메모리:** `{case}` **{peak:.3f}GiB /{ratio:.2f}%**. 이 값은 해당 최신run의 성공 조건에서의 최대이며,다른 run의 최대와 구분한다. probe이면 전체 epoch·독립test 평가 완료를 뜻하지 않는다.\n'
 if model=='cellpose':summary+='- **전체 실험 구분:** 추론 타일batch384는43.397GiB/35.66%;FP32 파인튜닝 용량probe batch26은90.520GiB/74.39%. 전체10epoch·test평가·재로드는batch1에서 완료(AP50=0.9278). batch28은100GiB allocator 예산에서OOM;80% 성공은 미확인.\n'
 if model=='scgpt':summary+='- **실제1만세포 파인튜닝 완료:** batch284는98.003GiB/80.54%,1epoch442.37초,test5000세포Accuracy0.4674,재로드검증 완료. 이전고정padding batch292의82.79%는 별도 짧은probe이며 실제1만세포 조건과 다르다.\n'
 if model in ['colabfold','clair3']:summary+='- **학습 상태:** 이번 실행에서 훈련·미세조정은 수행하지 않았다. 참고 구조/진실변이는 평가 자료이며 학습 정답으로 사용한 것이 아니다.\n'
 summary+='- **시간·정확도 해석:** 결과는 같은 행의 입력·배치·학습범위에만 연결한다. 메모리 상승 자체가 처리량·정확도 상승을 보장하지 않는다.\n'
 content=marker+'\n'+summary+end+'\n\n'
 if marker in s:s=re.sub(re.escape(marker)+r'[\s\S]*?'+re.escape(end)+r'\n*',lambda _:content,s)
 else:
  m=re.search(r'^## 4\.[^\n]*\n',s,re.M);s=s[:m.end()]+'\n'+content+s[m.end():]
 path.write_text(s);shutil.copy2(path,ROOT/filename)
print('Updated model/input tables and conclusions in all10Markdown reports')

# Refresh control-factor evidence in the same ten reports.
import subprocess,sys
subprocess.run([sys.executable,str(ROOT/'scripts/real_training_v3/factor_reports.py')],check=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>단계별 입력·출력 확인/보고 원문: build_reports.py</summary>

<!-- RESTORE_FILE {"path": "scripts/report_stages/source_snapshots/build_reports_28387c329c36e548.py", "sha256": "28387c329c36e5489ac208e5fffa3e1a78259f0ee92733f27179217f613cd675", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import json,re,hashlib,shutil,platform
ROOT=Path('/home/hyein/ai-benchmark');PUB=Path('/home/hyein/ai-benchmark-publication');MAP={'dnabert2':'DNABERT-2.md','ntv2':'Nucleotide_Transformer_V2.md','rnafm':'RNA-FM.md','openspliceai':'OpenSpliceAI.md','scvi':'scVI.md','cellpose':'Cellpose.md','scgpt':'scGPT.md','clair3':'Clair3.md','colabfold':'AlphaFold2.md','esm2':'ESM-2.md'}
def j(x):return '```json\n'+json.dumps(x,ensure_ascii=False,indent=2)+'\n```\n'
def vector(x):return '['+', '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in x)+']'
def shape(s,k):
 x=s['arrays'][k];return str(x['shape'])+' '+x['dtype']
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join('---' for _ in headers)+'|\n'+''.join('|'+ '|'.join(str(v).replace('|','／') for v in row)+'|\n' for row in rows)+'\n'
def artifact(s,items):return table(['단계별 저장 배열','실제 shape·dtype','의미'],[(k,shape(s,k),desc) for k,desc in items])+'로컬 전체 배열: `'+s['saved_arrays']+'`. `arrays.npz::key`는 같은NPZ 파일 안의 배열이름이다. 아래 재현코드로 저장모델과 입력에서 다시 확인할 수 있다.\n\n'
def cpu_note(s):
 d=s.get('saved_gpu_output_max_abs_difference',s.get('reference_max_abs'))
 return '**값의 출처:** 저장된 모델을 CPU에서 새로 재확인한 중간값이다. 원GPU 자원계측 중에 캡처했던 것처럼 해석하지 않는다. '+(f'같은 저장GPU 출력과 최대 절대차이는 **{d:.3e}**이다. ' if d is not None else '')+'이 CPU 확인시간은 기존 GPU 벤치마크 시간에 포함하지 않는다.\n\n'
TRACES={}
for model,filename in MAP.items():
 if model=='esm2':continue
 r=ROOT/model;allp=list((r/'results').glob('**/stage_trace_cpu/trace.json'))+list((r/'results').glob('**/stage_trace_saved/trace.json'));assert allp,model;p=max(allp,key=lambda p:p.stat().st_mtime);s=json.loads(p.read_text());TRACES[model]=(p,s)
 t='### 실제 파일 → 전처리 → 모델 표현 → 최종 출력: 단계별 실제 예시\n\n'
 if model in ['dnabert2','ntv2','rnafm']:
  raw=s['raw_record'];classes={'dnabert2':18,'ntv2':2,'rnafm':28}[model];hidden={'dnabert2':768,'ntv2':1024,'rnafm':640}[model];task={'dnabert2':'18개 histone 표지 다중라벨 분류','ntv2':'enhancer 이진분류','rnafm':'RNA유형28종 분류'}[model]
  t+=f"이 예시는 **{task}**, 저장된 추가 전체backbone 파인튜닝 모델의 **test `{s['sample_id']}`**다. 아래 모든값은 같은 레코드에서 이어진다.\n\n"
  source={'dnabert2':'BEND CSV의 id,sequence,labels(원train1000 내부800/100/100분할)','ntv2':'data/inputs.json의 test: id,sequence,label','rnafm':'data/inputs.json의 test: id,sequence,type,label'}[model]
  t+='#### 0–1. 파일 레코드 — X와 정답 y\n\n'+source+'. 아래는 실제 레코드에서 앞80염기만 발췌한 표시이며, 모델에는 전체서열을 넣었다.\n\n'
  example={'id':s['sample_id'],'sequence_length':len(raw['sequence']),'sequence_first80':raw['sequence'][:80]}
  if model=='dnabert2':example['positive_label_indices']=json.loads(raw['labels']);example['target_18_multihot']=s['target']
  else:example['label']=raw['label'];example.update({'type':raw['type']} if 'type' in raw else {})
  t+=j(example)+'**X는 염기서열, y는 위 label**이다. 임베딩을 정답으로 쓰는 것이 아니다.\n\n'
  t+='#### 2. 토큰화 — 실제 모델 입력\n\n'+table(['항목','실제값'],[['tokens',shape(s,'input_ids')],['attention_mask',shape(s,'attention_mask')],['원서열 길이',str(s['raw_nt'])+'nt'],['유효token 수',s['valid_token_count']],['token ID 앞16개',vector(s['token_ids_first16'])]])
  t+=('DNA BPE는 몇nt가 몇token으로 바뀌는지 서열마다 달라진다.' if model=='dnabert2' else '6-mer 중심 토큰화로 염기수와token수가 다르다.' if model=='ntv2' else '염기당token+BOS/EOS이며, 원래전체pool의최대폭1022로padding해batch비교조건을고정했다.')+' 숫자ID는vocabulary index이고, 정답라벨번호와 다른 배열이다.\n\n'
  t+='#### 3. 본체 출력 — 위치별 표현\n\n`'+shape(s,'token_representations')+'`이며 각token에'+str(hidden)+'개 특징값이 생긴다. 첫위치의 앞8개 실제값: `'+vector(s['feature_first8'])+'`. 이 단계는 최종 class확률이 아니다.\n\n'
  t+='#### 4. Pooling — 서열당 벡터1개\n\n방식: **'+s['pooling']+'**. 출력 `'+shape(s,'embedding')+'`. 임베딩 앞8개 실제값: `'+vector(s['embedding_first8'])+'`.\n\n'
  t+='#### 5. 과제 head — 최종 예측으로 변환\n\n'+f'추가 파인튜닝의 linear head는 `{hidden}→{classes}`다. 기본보고서의 다른head/학습범위 조건과 구분한다.\n\n'
  t+=table(['단계','shape','의미'],[['head logits',shape(s,'logits'),'확률변환 전 실수'],['확률',shape(s,'probabilities'),'sigmoid:18개 독립확률' if model=='dnabert2' else f'softmax:{classes}개중한class의확률'],['판정',s['prediction'],'각확률>0.5를양성판정' if model=='dnabert2' else '최대확률class의ID']])
  t+='logits 앞8개: `'+vector(s['logits'][:8])+'`; 확률 앞8개: `'+vector(s['probabilities'][:8])+'`.\n\n'
  t+='#### 6. 정답과 비교\n\n'+table(['항목','실제값'],[['정답',s['target']],['예측',s['prediction']]])
  if model=='dnabert2':t+='정답은18개중복수양성인multi-hot이고, 예측도18개를독립판정한다. 이번추가통제실험의test micro-F1=0이며, 실행·재현검증을유효한분류성능달성으로해석하면안된다. 원래독립test와이번train내부분할test도구분한다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('input_ids','입력token'),('attention_mask','유효위치'),('token_representations','token별표현'),('embedding','서열당임베딩'),('logits','head출력'),('probabilities','class확률'),('target','정답')])+f"원GPU run의 `test_outputs.npy`는 **logits**이며 확률파일이 아니다. 값의 순서는 저장된 split_ids['test']와 일치한다.\n\n"+cpu_note(s)
 elif model=='openspliceai':
  raw=s['raw_record'];t+='**이 모델은 서열당임베딩벡터를분류하는방식이아니라 각염기위치의splice확률을출력한다.** 같은test구간 `'+s['sample_id']+'`를끝까지따라간다.\n\n'
  t+='#### 0–1. 실제 레코드와 정답\n\n`data/inputs.json`의 test 레코드다.\n\n'+j({k:raw[k] for k in ['id','chromosome','strand','genomic_input_start','genomic_input_end']})+'원레코드 X는20,000nt, 정답은가운데10,000위치의0/1/2주석이다. X를 `[2500:17500]`의15,000nt로, y를 `[2500:7500]`의5,000위치로대응되게선택했다. 0=nonsplice,1=acceptor,2=donor,-1=충돌주석무시다.\n\n'
  t+='#### 2. One-hot 입력\n\n'+shape(s,'onehot')+';채널순서는A/C/G/T, 앞12염기는 `'+s['input_first12_bases']+'`. 한위치가A이면 `[1,0,0,0]`이고 token ID로변환하는모델이아니다.\n\n'
  t+='#### 3–4. 합성곱 특징과 문맥 처리\n\n첫합성곱의실측특징은 `'+shape(s,'initial_conv_features')+'`;첫채널앞8개는 `'+vector(s['feature_first8'])+'`. 이후dilated residual CNN이넓은문맥을처리하며,단백질언어모델처럼전체서열을평균임베딩하진않는다.\n\n'
  t+='#### 5. 최종 위치별 출력\n\n출력은 **'+shape(s,'output_probabilities')+'**이다. 위치5,000개각각에3종확률이있다. 앞5위치실제확률(행=위치,열=nonsplice/acceptor/donor):\n\n'+j(s['first5_position_probabilities'])
  t+='#### 6. 같은 위치의 정답과 비교\n\n정답5,000개중라벨별개수: `'+str(s['target_counts'])+'`. 주석상splice위치의첫index들(출력window내0-based): `'+str(s['positive_target_positions_first8'])+'`. 전체구간한개에라벨하나를붙인과제가아니다. 평가는위치별acceptor/donor AUPRC등이며,seed10단일모델미세조정과5모델ensemble추론은별도조건이다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('onehot','DNA입력'),('initial_conv_features','첫CNN특징'),('output_probabilities','위치별3확률'),('target','각출력위치정답')])+'원GPU `test_outputs.npy`는 `[36,3,5000]`의확률이다.\n\n'+cpu_note(s)
 elif model=='scvi':
  t+='**scVI는처음부터학습한count 생성VAE이며,세포유형분류head를붙이지않았다.** 저장모델의 실제세포 `'+s['sample_id']+'`를사용한다.\n\n'
  t+='#### 0–1. 원본 파일과 행렬의 한 행\n\n`data/processed_counts.h5ad`와저장모델의`adata.h5ad`에raw-count가보관된다. 원본처리행렬은세포12,721×유전자2,000이다. 행1개=세포1개,열1개=유전자1개다.\n\n'+j({'cell_id':s['sample_id'],'nonzero_gene_names_first8':s['gene_names_first_nonzero'],'raw_counts_same_genes':s['counts_first_nonzero'],'biological_batch_sample_id':s['batch_covariate']})+'생물학적sample_id는GPU batch크기가아니다. 세포유형라벨을정답으로학습한모델도아니다.\n\n'
  t+='#### 2. 모델 입력\n\ncounts는 `'+shape(s,'raw_counts')+'`이며batch공변량을함께사용한다. Raw count를입력하고같은count의생성분포를학습한다.\n\n'
  t+='#### 3. Encoder 특징\n\n실제encoder hidden출력은 `'+shape(s,'encoder_hidden')+'`이다.\n\n'
  t+='#### 4. 세포별 latent\n\n`give_mean=True`로사후분포평균을반환했다. 실측 '+shape(s,'latent')+'의전체10개값은:\n\n```text\n'+vector(s['latent'])+'\n```\n\n'
  t+='#### 5–6. 출력과 학습목표\n\n이번latent추론의최종출력은세포당10차원벡터다. 별도분류기확률/세포유형이아니다. **훈련시에는**decoder의count분포와raw count를이용한재구성항및KL항으로VAE를학습한다. 따라서정답구조는원래count이고외부cell-type정답을만들지않았다. 이예제CPU추적은latent경로만확인했으며decoder중간값을저장했다고주장하지않는다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('raw_counts','원래count행'),('encoder_hidden','인코더중간표현'),('latent','10차원최종latent')])+'전체입력의GPU추론배열은각batch의 `outputs.npy`;이예제는그첫행과대조했다.\n\n'+cpu_note(s)
 elif model=='cellpose':
  t+='**Cellpose는이미지를평균임베딩하나로줄여분류하는모델이아니라 픽셀별flow·cell probability를만들어instance mask로변환한다.** 실제test이미지 `'+s['sample_id']+'`를사용한다.\n\n'
  t+='#### 0–1. 원본 이미지와 정답 mask\n\n'+table(['역할','실제파일','shape·dtype'],[['입력X',s['raw_image_file'],shape(s,'raw_image')],['정답y',s['raw_mask_file'],shape(s,'ground_truth_mask')]])+f"원본은520×696 흑백uint16 TIFF,정답PNG를공식방법으로연결성분instance-ID로변환하면핵 **{s['truth_instances']}개**다. mask값0은배경,양의정수각각은서로다른객체다.\n\n"
  t+='#### 2. 정규화·타일화 → 네트워크 입력\n\n공식백분위정규화와256×256타일화를적용한다. 이번CPU확인의첫실제입력은 **'+shape(s,'first_model_input')+'**. 추론경로의흑백1채널입력이고,파인튜닝의3채널crop(흑백+0채널2개)과구분한다. 원본520×696전체가한번에같은모양으로GPU에올라가는것은아니다.\n\n'
  t+='#### 3–4. SAM encoder의 공간 특징\n\n첫encoder block의실제출력은 **'+shape(s,'first_encoder_block_features')+'**. 공간격자에채널별특징이있고, `[1,1280]`같은이미지전체임베딩이최종결과인모델이아니다. 펼친첫8값: `'+vector(s['first_feature_values'])+'`.\n\n'
  t+='#### 5. 타일 합치기 → flow·cell probability\n\n'+table(['출력','shape·dtype','의미'],[['flow',shape(s,'flows'),'각픽셀의Y/X방향흐름'],['cell probability',shape(s,'cell_probability'),'cellprob_threshold를적용할네트워크점수;0~1로보정된확률로단정하지않음']])+ '타일결과를원래이미지크기에맞춰합친다. 이후flow dynamics·threshold·크기필터를거쳐픽셀을서로다른객체로묶는다.\n\n'
  t+='#### 6. 최종 instance mask와 정답 비교\n\n최종예측은 **'+shape(s,'predicted_mask')+'**,예측객체'+str(s['predicted_instances'])+'개다. 원래같은크기의정답mask와객체를매칭해AP50/AP75/AP90을계산한다. mask ID는객체식별번호이지세포유형라벨이아니다. 이예제CPU/GPU mask일치: **'+str(s['cpu_gpu_mask_equal'])+'**. 이번CPU타일batch는1이고원GPU평가는8이므로새속도비교로쓰지않는다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('raw_image','입력이미지'),('ground_truth_mask','정답객체mask'),('first_model_input','첫네트워크타일'),('first_encoder_block_features','encoder공간특징'),('flows','전체이미지flow'),('cell_probability','전체이미지점수'),('predicted_mask','최종객체IDmask'),('saved_gpu_mask','원GPU결과대조')])+cpu_note(s)
 elif model=='scgpt':
  t+='같은실제test세포 `'+str(s['sample_id'])+'`를사용한다. **세포1개=행렬1행이며,그행의유전자별발현값을token으로바꾼다.** 이번예제는Tabula Sapiens161클래스파인튜닝모델이고pancreas13클래스와별도다.\n\n'
  t+='#### 0–1. 원본 h5ad의 count 행\n\n원본Tabula Sapiens는483,152세포×58,482유전자이고원본`raw/X`에서실제선택세포한행을확인했다. 아래원본유전자index와count는같은순서다.\n\n'+j(s.get('raw_count_row',{'note':'원본count의새확인값없음'}))+'정답은세포metadata의cell_type이며,유전자ID나발현량이아니다.\n\n'
  t+='#### 2. 유전자 선택·발현 binning → 두 입력 배열\n\n공개vocabulary와매칭한유전자에서세포별nonzero gene을선택하고발현을bin으로변환한다. 최대1200자리에는CLS가포함된다. 이번세포의입력앞16개는:\n\n'+j({'gene_ids':s['first16_gene_ids'],'gene_names_same_order':s['first16_gene_names'],'expression_bins_same_order':s['first16_expression_bins']})+table(['배열','실측shape·dtype'],[['gene_ids',shape(s,'gene_ids')],['expression_bins',shape(s,'expression_bins')]])+'처음의CLS와특수발현값은일반유전자의raw count가아니다. bin값도원래UMI count와다른전처리결과다. 원본58482열전체를그대로1200열로복사한것이아니다.\n\n'
  t+='#### 3. token별 Transformer 표현\n\n실제출력: **'+shape(s,'token_representations')+'**. 각CLS/유전자token마다512개특징값이생긴다.\n\n'
  t+='#### 4. 세포 임베딩\n\nCLS위치의벡터를선택해 **'+shape(s,'cls_embedding')+'**로만든다. 앞8개실제값:`'+vector(s['embedding_first8'])+'`. 이분류경로는raw CLS를사용하며,별도임베딩추론의L2정규화와구분한다.\n\n'
  t+='#### 5. 분류 head\n\nhead logits는 **'+shape(s,'logits')+'**,softmax확률도 `[1,161]`이다. 세포당161개유형중최대확률유형을선택한다.\n\n'
  t+='#### 6. 실제 cell_type 정답과 비교\n\n'+table(['항목','실제값'],[['예측class',str(s['predicted_id'])+' /'+str(s['predicted_label'])],['예측최대확률',f"{s['top_probability']:.6f}"],['정답class',str(s['true_id'])+' /'+str(s['true_label'])]])+'전체test성능은별도5000세포지표이며,한세포예측결과가전체정확도를뜻하지않는다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('gene_ids','유전자ID입력'),('expression_bins','발현bin입력'),('token_representations','위치별특징'),('cls_embedding','세포임베딩'),('logits','class점수'),('probabilities','class확률'),('target','실제유형ID')])+'원GPU배열은 `test_logits.npy` `[5000,161]`와 `test_rows.npy`;같은행을재확인했다.\n\n'+cpu_note(s)
 elif model=='clair3':
  t+='**Clair3는단백질임베딩모델이아니라 정렬된read에서후보변이특징을만들어변이확률을계산한다.** 이예제는실제cache의첫후보이며,캐시에genomicposition ID가저장되지않았으므로BAM의특정read·표시된VCF행과동일좌표라고연결하지않는다.\n\n'
  t+='#### 0–1. 원본 입력과 평가 정답\n\n입력은 `'+ '`, `'.join(s['raw_input_files'])+'`및BAM index다. BAM의정렬read·염기품질·mapping정보와참조유전체에서후보특징을계산한다. GIAB truthVCF와평가구간은정확도평가용이며이번에학습한라벨tensor가아니다.\n\n'
  t+='#### 2. 후보 특징 tensor\n\n'+table(['모델분기','입력shape','dtype'],[(k,v['shape'],v['dtype']) for k,v in s['branches'].items()])+'pileup의33위치×18특징과full-alignment의89read행×33위치×8채널이다. batch는DNA문자수나원본read총수가아니라동시에처리하는후보tensor수다.\n\n'
  t+='#### 3–4. 분기별 특징 계산\n\npileup은양방향LSTM,full-alignment는CNN경로로후보의표현을계산한다. **이CPU확인에서는그내부hidden tensor를캡처하지않았으므로실제중간embedding값은N/A**다. 존재하지않는평균임베딩단계를추가하지않는다.\n\n'
  t+='#### 5. 최종 후보별 확률\n\n'
  for k,v in s['branches'].items():t+='- **'+k+'**:출력'+str(v['output_shape'])+',앞24개실제값 `'+vector(v['probability_first24'])+'`. 원GPU같은행과최대차이 '+f"{v['reference_max_abs']:.3e}"+'.\n'
  t+='\nP의24출력은GT21+genotype3, F의90출력은GT21+genotype3+indel길이33+33개head를연결한것이다. 전체90개를단일softmax90클래스로해석하지않는다.\n\n'
  t+='#### 6. 후보확률 → 실제 VCF·평가\n\n공식pipeline은후보별확률과위치·ref/alt정보를조합해VCF의CHROM/POS/REF/ALT/QUAL/GT등을작성한다. cache예제의좌표mapping이없어이확률과특정VCF행의직접대조는미검증이다. 원래BAM→VCF실행·truth평가결과는아래상세에보존했다. 훈련·파인튜닝은수행하지않았다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('Clair3_P_input','pileup후보입력'),('Clair3_P_probabilities','pileuphead확률'),('Clair3_F_input','full-alignment후보입력'),('Clair3_F_probabilities','full-alignmenthead확률')])+cpu_note(s)
 elif model=='colabfold':
  raw=s['raw_record'];t+='**AlphaFold2는단백질서열과MSA를받아구조를만든다. 평균임베딩뒤에회귀head를붙인ESM-2 과제와다르다.** 실제독립query `'+s['sample_id']+'`의저장된입력·출력을확인했다.\n\n'
  t+='#### 0–1. 원본 서열·MSA\n\n```fasta\n>'+s['sample_id']+'\n'+raw['sequence']+'\n```\n\n서열115AA,feature sampling전원MSA깊이'+str(raw['msa_depth_before_feature_sampling'])+'이다. 이query에는정답구조를학습시킨것이아니며,이번작업은추론이다.\n\n'
  t+='#### 2. 모델 입력 features\n\n'+table(['실제NPZ key','실측shape·dtype'],[(k,str(v['shape'])+' '+v['dtype']) for k,v in s['feature_shapes'].items() if k in ['aatype','target_feat','msa_feat','msa_mask','extra_msa','extra_msa_mask','residue_index']])+'recycle/ensemble feature축,MSA행,잔기축을원본서열개수나독립query batch와혼동하지않는다. zero-padding된MSA행은mask로제외한다.\n\n'
  t+='#### 3–4. Evoformer 표현·구조 모듈\n\nEvoformer가MSA와잔기쌍표현을처리하고structure module이원자좌표를계산한다. **원GPU실행은내부Evoformer embedding을저장하지않았으므로중간숫자는N/A**다. 이번확인은저장된features와최종예측배열을읽은것이며새JAX/GPU추론을실행한것이아니다.\n\n'
  t+='#### 5. 최종 구조·신뢰도 출력\n\n'+table(['출력','shape·dtype','실제예시'],[['atom_positions',str(s['output_shapes']['atom_positions']['shape'])+' '+s['output_shapes']['atom_positions']['dtype'],'첫잔기CA좌표 '+vector(s['first_residue_CA_xyz'])],['plddt',str(s['output_shapes']['plddt']['shape'])+' '+s['output_shapes']['plddt']['dtype'],'앞8값 '+vector(s['plddt_first8'])]])+'좌표의마지막3개값은X/Y/Z좌표이며,37원자slot중유효원자는residue별mask를따른다.\n\n'
  t+='#### 6. 정답·점수 해석\n\n평균pLDDT는 '+f"{s['mean_plddt']:.3f}"+'이며 **모델의신뢰도**다. 이query의정답실험구조가없어이를구조정확도로쓰지않는다. PDB1UBQ등정답구조대조는별도입력조건이다.\n\n'
  t+='#### 7. 저장 파일\n\n`'+s['pdb_file']+'`의실제처음5개ATOM레코드:\n\n```text\n'+'\n'.join(s['first5_PDB_ATOM_lines'])+'\n```\n\n저장최종NPZ는atom_positions/plddt이며준비features와함께 `'+s['saved_arrays']+'`에사본을보관했다. 별도3D좌표/PDB출력이지class라벨이아니다.\n\n'
 templates=ROOT/'scripts/report_stages/templates';templates.mkdir(exist_ok=True)
 (templates/(model+'.txt')).write_text(t)
# Record actual replay source snapshots and insert environment/measurement overview into all ten existing reports.
for model,filename in MAP.items():
 p=PUB/filename;s=p.read_text();rr=ROOT/model/'results';env={};source_path='상세원본환경기록 참조'
 candidates=[]
 for pointer in ['REAL_TRAINING_LATEST','LATEST']:
  pp=rr/pointer
  if pp.exists():
   dd=rr/pp.read_text().strip();candidates+=sorted(dd.glob('*/summary.json'))+[dd/'summary.json']
 candidates+=list(rr.glob('*/summary.json'))
 for q in candidates:
  if q.exists():
   v=json.loads(q.read_text())
   if isinstance(v.get('environment'),dict) and v['environment']:env=v['environment'];source_path=str(q.relative_to(ROOT));break
 envrows=[('장비·공유메모리','DGX Spark,NVIDIA GB10,ARM64;시스템공유RAM121.6875GiB. CPU/GPU RAM을별도로더하지않음'),('환경기록출처',source_path),('Python·프레임워크',str({k:v for k,v in env.items() if k in ['python','torch','jax','transformers','cuda','scgpt','cellpose','platform','precision']}) or 'N/A;5절의환경및실행코드참조'),('실험병렬정책','GPU벤치마크한번에1개. 이단계별CPU검증은별도작업이며GPU측정시간과합치지않음'),('GPU-Util','nvidia-smi utilization.gpu(%);관측기간중커널활동시간비율. 최대FLOPS/메모리비율아님'),('GPU메모리','PyTorch allocated/reserved및peak;JAX는bytes_in_use/peak_bytes_in_use별도. 모델적재직후·실행구간을분리'),('프로세스·시스템메모리','psutil RSS,MemAvailable,swap;시계열의sampled peak/최소값이며정확한연속최고치아님'),('표본간격','기본0.5초대기+GPU조회시간;측정구간의평균·최대·표본수저장. 짧은실행은동일조건반복하고고유입력과반복처리량구분'),('시간동기화','PyTorch구간앞뒤CUDA synchronize;JAX block_until_ready. 비동기호출반환시간만측정하지않음'),('시간포함범위','조건별로입력준비/전송/forward/backward/optimizer/validation/save포함여부를기록. 서로다른범위의시간을직접배수비교하지않음'),('loading·compile·warmup','구간을분리한다. JAX누적allocator peak가compile/warmup을포함하는경우본문에명시하고CUDA구간peak와직접동일시하지않음'),('메모리기준값과증가량','첫forward전모델적재기준값과같은run의실행peak를비교. 증가량에는학습gradient/optimizer/중간활성값이포함'),('누락·관측한계','미지원/미관측값은N/A. 장치전체GPU-Util에는다른프로세스·디스플레이활동이섞일수있음. 단일실행은통계적반복검증으로쓰지않음')]
 if model=='colabfold':envrows.append(('JAX메모리해석','XLA사전할당비활성. bytes_in_use와누적peak를분리하며미지원CUDA allocated/reserved는N/A'))
 content='<!-- ENV_MEASUREMENT_OVERVIEW_START -->\n### 실행 환경·자원 측정 방법 요약\n\n'+table(['항목','이번 보고서의 측정 기준'],envrows)+'<!-- ENV_MEASUREMENT_OVERVIEW_END -->\n\n'
 if '<!-- ENV_MEASUREMENT_OVERVIEW_START -->' in s:s=re.sub(r'<!-- ENV_MEASUREMENT_OVERVIEW_START -->[\s\S]*?<!-- ENV_MEASUREMENT_OVERVIEW_END -->\n*',lambda _:content,s)
 else:
  a=re.search(r'^## 2\.',s,re.M).start();i=s.index('\n',a)+1;s=s[:i]+'\n'+content+s[i:]
 p.write_text(s);shutil.copy2(p,ROOT/filename)
print('Generated nine model-specific stage sections and environment/measurement tables for all ten')
````
<!-- END_RESTORE_FILE -->

</details>

<details><summary>단계별 입력·출력 확인/보고 원문: standardize_overview.py</summary>

<!-- RESTORE_FILE {"path": "scripts/report_stages/source_snapshots/standardize_overview_d4186f52199664e9.py", "sha256": "d4186f52199664e9a2e6b440321cd3fcb0c4fe01e44287fa72285514a07b4606", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,re,shutil
ROOT=Path('/home/hyein/ai-benchmark');PUB=Path('/home/hyein/ai-benchmark-publication')
META={'esm2': ('ESM-2.md', 'esm2_t33_650M_UR50D', 'Transformer encoder,33층,1280차원', '651,043,254개(651.04M);encoder649,400,320개', 'FP32', '단백질 서열 →1280차원 임베딩 / GFP 형광값 예측', '추론:UniProt reviewed 단백질. 추가 run은900–950AA이며 GFP 학습 서열과 별개', 'TAPE fluorescence의 실제 GFP 변이체 약237AA. 기본 train128/val64/test128;추가 비교1024/128/512', 'X:아미노산→int64 tokens[B,L+2];y:GFP log_fluorescence[B]. 라벨 표준화에는 train 평균·표준편차만 사용', '기본:일부 backbone 층+회귀head 파인튜닝;추가:전체 backbone+linear 회귀head 파인튜닝', 'data/fluorescence/fluorescence_{train,valid,test}.lmdb', '긴 UniProt 추론 서열에는 본 실험의 GFP 정답이 없다. 부분 미세조정과 전체 미세조정 수치를 섞지 않는다'), 'colabfold': ('AlphaFold2.md', 'ColabFold1.6.2,alphafold2_ptm/alphafold2_multimer_v3', 'Evoformer+Structure Module', '실제PTM tree92,907,930개(92.91M)/Multimer93,223,705개(93.22M);원본NPZ 구성은 상세표 참조', '기본 BF16;별도 실험vmap FP32', '단백질 서열+MSA →3D좌표,pLDDT/PAE', '서로 다른 UniProt115AA 단백질4개,PDB1UBQ/1LYZ/1TIM;복합체3G5O A/B 예제', '훈련·파인튜닝 미실시. 이번 작업에 사용한 학습 데이터셋 없음', '추론 X:서열·MSA features. 참고 PDB는 평가용 정답 구조이며 학습에 사용하지 않음', '추론만 수행. 설치된 ColabFold 실행 경로에서 훈련하지 않음', 'data/queries.fasta;각run processed_features.npz/MSA', 'pLDDT는 신뢰도이며 실제 정답 구조 대비 정확도와 다르다. vmap은 별도 실험적 추론 경로'), 'scvi': ('scVI.md', 'SCVI,seed42무작위초기화;사전학습미사용 checkpoint', 'ZINB count 생성 VAE,hidden128,1층,latent10', '1,293,526개(1.294M);latent z_encoder258,964개;첫backward gradient778,884개', 'FP32', 'raw-count 세포×유전자+sample_id →10차원 latent /count 분포', 'PBMC4k+8k raw count에 저장 모델을 적용해 latent 추출. 세포유형 분류 과제 아님', '실제10x PBMC4k+8k12721세포×2000HVG. train10176/validation1272/test1273;작은 중첩 train부분집합도 별도 비교', 'X:counts[B,2000];재구성 목표는 같은 raw count. sample_id는 실제 생물학적batch 공변량. cell-type 정답 없음', '사전학습 가중치 없이 무작위 초기화에서 처음부터 학습,400epoch. 파인튜닝 아님', 'data/processed_counts.h5ad;results/<input_run>/splits.json', '재구성 기반 비지도 학습이며 생물학적batch와 GPU batch는 다르다. 통합 진단을 세포유형 정확도로 부르지 않음'), 'cellpose': ('Cellpose.md', 'Cellpose4.2.1,cpsam_v2（Cellpose-SAM）', 'SAM ViT-L 기반 이미지 분할', '304,628,930개(304.63M)', '추론BF16;파인튜닝FP32', '흑백 이미지 →핵/세포별 서로 다른ID의 instance mask', 'BBBC039200장520×696 uint16 핵 형광 이미지. 뇌 추가 입력은 NucMM-M192³ micro-CT patch4개', 'BBBC039 공식100/50/50;기본부분집합16/8/16. 추가 학습은 빈mask2장 제외 후 train98/validation50/test50', 'X:원본520×696에서 공식 무작위crop[B,3,256,256](흑백+0채널2개). y:같은 crop에 맞춰 변환한 instance mask·cell probability·flow. 세포유형 정답 아님', 'cpsam_v2 사전학습 가중치에서 backbone·분할출력층 파인튜닝. 실제 일부 파라미터는 gradient가 관측되지 않아 수를 별도 기록', 'data/input_audit.json;data/images/images/*.tif;data/masks/masks/*.png;추가run/prepared/*_mask.npy', 'batch26은2epoch 파인튜닝 용량 측정;전체10epoch 독립test·재로드 완료는batch1. 뇌patch는zero-shot 추론만 수행'), 'dnabert2': ('DNABERT-2.md', 'DNABERT-2-117M,revision7bce263b15377fc15361f52cfab88f8b586abda0', 'DNA BPE tokenizer+Transformer encoder,hidden768', 'encoder117,068,544개(117.07M);18출력linear head13,842개 추가', 'FP32', 'DNA →BPE tokens →임베딩 /18종histone 다중라벨', '실제BEND GRCh38구간.512/1000/2000/4000bp 비교;bp수와token수는 별개', 'BEND histone_modification 실제구간 및18개표지. 기본train1000. 추가비교는 원train1000 안에서 seed42로800/100/100 분리', 'X:input_ids/attention_mask[B,Ltokens]. y:[B,18] float32 multi-hot이며18중 하나를 고르는 과제 아님. 신규추론4000bp pool padding851tokens', '공개encoder+linear 다중라벨head 전체 파인튜닝. 사용하지 않는pooler 등 분기와 실제gradient 수는 별도 기록', '/home/hyein/dnabert2-benchmark/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv', '추가800/100/100은 기존train 내부 분할이며 원래 독립test 결과를 대체하지 않음. 염색체독립 검증 아님'), 'ntv2': ('Nucleotide_Transformer_V2.md', 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species', 'DNA6-mer 중심 tokenizer+ESM 구현 Transformer', 'MLM498,345,436개(498.35M);embedding backbone493,084,113개;기본CLS분류구성494,135,763개', 'FP32', 'DNA →token 임베딩 /enhancer 이진분류', 'GRCh38 실제긴구간. 추가추론:서로 겹치지 않는6000nt window1024개,약1001tokens', '실제enhancer 라벨 서열199–200nt. train512/validation128/test400', 'X:token_ids/attention_mask[B,실제token길이]. y:[B]정수0/1. 긴6000nt 추론 window에는 이 enhancer 정답이 없음', '기본:전체backbone+원생CLS MLP 파인튜닝. 추가:전체backbone+masked-mean linear head 파인튜닝,별도 조건', 'data/inputs.json의 train/validation/test;data/real_training_v3/inference_inputs.json추론전용', '긴 유전체 추론window와 짧은enhancer 학습 서열을 구분. 정확도는 해당 라벨·split에서만 평가'), 'clair3': ('Clair3.md', 'Clair3 v2.0.4,ONT pileup.pt/full_alignment.pt', 'pileup 양방향LSTM 및 full-alignment CNN 후보분류 네트워크', 'Clair3_P2,074,520개(2.075M);Clair3_F2,986,522개(2.987M);단계별 실행', 'FP32', 'long-read BAM+참조유전체 →후보특징 →SNV/Indel VCF', 'HG003 ONT 실제BAM+GRCh38 chr20소구간. cache 고유feature P5708/F894', '이번 작업에서 학습·파인튜닝 미실시. 본 실험용 훈련 정답tensor 없음', '추론 X:P[B,33,18],F[B,89,33,8]. GIAB truthVCF/평가구간은 변이검출 정확도 평가용이며 학습에 사용하지 않음', '공식학습기능 존재와 별개로 이번에는 사전학습ONT 가중치의 추론만 수행', 'data/HG003_chr20_demo.bam;data/GRCh38_no_alt_chr20.fa;candidate run/*_inputs.npy', '후보batch는read개수가 아님. 작은구간 후보pool을 전부 사용한 결과는 전장유전체 최대메모리 한계가 아님'), 'rnafm': ('RNA-FM.md', 'RNA-FM ncRNA,RNA-FM_pretrained.pth（fm0.2.2）', 'RNA Transformer12층,hidden640', 'backbone99,521,546개(99.52M);28종linear head17,948개', 'FP32', 'RNA →640차원임베딩 /RNA유형 분류', '기본:RNAcentral 실제RNA. 추가추론:GENCODEv47 lncRNA transcript 첫1000nt,T→U,고유2400개', 'RNAcentral1400서열/28종류. train896/validation224/test280. 실제길이가 다양하며 최대1020nt', '기본head-only X:동결RNA-FM에서 추출한embedding[B,640]. y:[B]28종 유형ID. 추가backbone파인튜닝 X:RNA tokens[B,L+2]와 같은유형정답', '기본:본체동결·분류기만학습. 추가전체backbone+linear 분류기 파인튜닝은 해당run의 실제 완료상태를 확인', 'data/inputs.json의 train/validation/test;data/input_audit.json의 label매핑;GENCODE추론전용', '기본head-only 메모리를 전체RNA-FM 미세조정 메모리로 쓰지 않음. 신규lncRNA 추론2400개는28종 지도학습셋과 별개'), 'openspliceai': ('OpenSpliceAI.md', 'MANE10000nt,seed10–145개모델;파인튜닝은seed10', '팽창합성곱 residual CNN,10000nt 문맥', '단일696,739개(0.697M);5모델합계3,483,695개(3.484M)', 'FP32', 'DNA4채널one-hot →위치별nonsplice/acceptor/donor 확률', '기본:GRCh38실제156구간. 추가추론:고유비중첩15000nt window16000개,5모델ensemble', 'exon주석을확인한 GRCh38영역 train96/validation24/test36,염색체별분리. 원레코드20000nt/정답10000nt→실사용15000nt/정답5000nt', 'X:[B,4,15000] A/C/G/T one-hot. y:[B,5000]정수0/1/2;충돌위치-1무시. 구간전체에 정답1개가 아니라 위치별정답', 'seed10 단일모델 전체파인튜닝. seed10–14의5모델ensemble은 추론용이며5개동시학습 아님', 'data/inputs.json의 sequence및labels_10000;source/openspliceai/variant/annotations/grch38.txt', '추가무라벨reference window를 임의정답으로 학습시키지 않음. 지도학습정답은exon주석에서 변환'), 'scgpt': ('scGPT.md', 'scGPT whole-human,best_model.pt', 'Transformer12층,hidden512;geneID+expression bin embedding', 'encoder50,278,400개(50.28M);기본추론구성51,856,898개;13클래스구성51,338,254개;161클래스구성은run참조', 'FP32', '세포×유전자발현 →유전자token/bin →512차원임베딩 /세포유형분류', 'pancreas테스트세포;Tabula Sapiens 실제고유30만세포embedding 추론. 전부미세조정한 것은 아님', 'pancreas기본 train1000/val2120/test4218;전체8480/2120/4218. Tabula donor분리238868/35738/25394 중 실제train10000/val2000/test5000 사용', 'X:gene_ids[B,L]int64와expression_bins[B,L]float32,L≤1200(CLS포함). y:[B]Celltype/cell_type유형ID. 유전자가 분류정답인 것은 아님', '공개encoder/value encoder/Transformer+분류기 파인튜닝. pancreas13종과Tabula161종은 별도분류설정. 미사용분기와실제trainable/gradient수를 구분', 'data/pancreas_demo/*.h5ad;data_external/tabula_sapiens/prepared_300k_seed42/{gene_ids,expression_bins}.npy;추가run/labels.npy및*_rows.npy', 'batch는동시에처리하는세포수,각세포는표현행렬1행.1200은CLS+유전자token자리이며 세포수가아님')}
def ko(s):return s
for model,meta in META.items():
 filename,checkpoint,arch,params,precision,io,infer,training,xy,method,paths,limit=meta
 def human_count(match):
  n=int(match.group(1).replace(',',''));human=f'약{n/100000000:.2f}억 개' if n>=100000000 else f'약{n/10000:,.1f}만 개';return match.group(0)+'('+human+')'
 params=re.sub(r'(\d{1,3}(?:,\d{3})+)개',human_count,params)
 path=PUB/filename;s=path.read_text();table='|항목|이번 실험에서 사용한 모델|\n|---|---|\n'
 for key,value in [('체크포인트·버전',checkpoint),('구조',arch),('전체·구성별 파라미터 수',params),('정밀도',precision),('모델 입력 → 출력',io),('가중치와 실행 메모리의 차이','파라미터 수×dtype bytes는 이론 가중치 용량이다. 실행 peak에는 입력·중간 활성값이 더해지고,파인튜닝에는 gradient/optimizer 상태도 추가된다.'),('메모리% 정의','CUDA allocated peak / CPU·GPU 공유 RAM121.6875GiB ×100; GPU-Util과 별개. AlphaFold2는 JAX 계측을 별도로 표시')]:table+='|'+key+'|'+ko(value)+'|\n'
 table+='\n### 실제 데이터와 학습 방식 구분\n\n|구분|데이터·실제 처리 내용|\n|---|---|\n'
 for key,value in [('추론 데이터',infer),('학습·검증·test 데이터',training),('학습 입력 X / 정답 y',xy),('가중치 갱신 방식',method),('원본·변환 입력 저장 위치','`'+paths+'`'),('해석 범위',limit)]:table+='|'+key+'|'+ko(value)+'|\n'
 table+='\n**용어:** 처음부터 학습=무작위 초기화에서 시작; 전체/일부 층 파인튜닝=공개 가중치에서 해당 부분 갱신; 분류기만 학습=본체를 동결하고 head만 갱신. **용량probe는 학습 방식이 아니라 짧은 자원 측정 목적**이며,실제로 수행한 가중치 갱신 방식과 함께 표기한다. 원본 전체 데이터 수와 해당 run에서 실제 사용한 부분집합 수를 구분한다.\n\n'
 template=ROOT/'scripts/report_stages/templates'/(model+'.txt')
 if template.exists():table+=template.read_text()
 elif (ROOT/model/'data/report_input_section.txt').exists():table+=(ROOT/model/'data/report_input_section.txt').read_text()
 # Bring measured post-load baselines forward; do not infer them from parameter counts.
 candidates=[];rr=ROOT/model/'results'
 for pointer_name in ['REAL_TRAINING_LATEST','BATCH_LATEST','LATEST']:
  pp=rr/pointer_name
  if pp.exists():
   dd=rr/pp.read_text().strip();candidates+=sorted(dd.glob('*/summary.json'))+[dd/'summary.json']
 found={}
 for qp in candidates:
  if not qp.exists():continue
  vv=json.loads(qp.read_text());base=vv.get('model_only_postload',vv.get('model_postload'))
  if not isinstance(base,dict):continue
  training_case=any(x in qp.parent.name for x in ['training_b','train_b','probe_b'])
  role='파인튜닝 적재 기준' if training_case else '추론/기본 모델 적재 기준'
  if model=='clair3':role+=' ('+str(vv.get('model','분기 미기록'))+')'
  if 'allocated_gib' in base:
   if role in found:continue
   found[role]=(f"{base['allocated_gib']:.6f}",str(base.get('reserved_gib','N/A')),str(base.get('rss_gib','N/A')),str(qp.relative_to(ROOT)))
  elif 'jax' in base and role not in found:
   value=base['jax'].get('bytes_in_use');found[role]=('JAX '+(f'{value/2**30:.6f}' if value is not None else 'N/A'),'N/A',str(base.get('system',{}).get('rss_gib','N/A')),str(qp.relative_to(ROOT)))
 table+='### 모델 적재 직후 — 첫 입력 forward 이전\n\n|적재 조건|allocated GiB(JAX는별도표기)|reserved GiB|프로세스 RSS GiB|원본 근거|\n|---|---|---|---|---|\n'
 for role,values in found.items():table+='|'+role+'|'+'|'.join(values)+'|\n'
 if not found:table+='|기준값 원본을 추가 확인 중|N/A|N/A|N/A|기존 상세 기록 참조;추정값으로 채우지 않음|\n'
 table+='\n위 값은 **가중치를 GPU에 적재하고 첫 입력을 계산하기 전의 프레임워크 할당량**이다. CUDA/JAX context·driver·시스템 전체 GPU사용량과 같지 않으며 CPU입력 준비 포함 여부는 해당 실행 코드를 따른다. **실행 peak와 모델 적재 기준값의 차이는 같은 run에서만 계산**한다. 파인튜닝 중 추가량에는 입력뿐 아니라 중간값·gradient·optimizer 상태가 포함된다. 서로 다른 조건/정밀도의 기준값을 하나의 모델 고유 메모리로 단정하지 않는다.\n\n'
 a=s.index('## 1. 모델과 실제 입력')+len('## 1. 모델과 실제 입력');b=a+re.search(r'^## 2\.',s[a:],re.M).start();s=s[:a]+'\n\n'+table+s[b:]
 # Keep historical detailed results/code intact; add an up-to-date conclusion at section4.
 marker='<!-- MODEL_SUMMARY_START -->';end='<!-- MODEL_SUMMARY_END -->';summary='### 요약 결론\n\n- **학습 방식:** '+ko(method)+'.\n- **입력 구분:** '+ko(limit)+'.\n'
 pointer=ROOT/model/'results/REAL_TRAINING_LATEST'
 if pointer.exists():
  d=pointer.parent/pointer.read_text().strip();records=[]
  for p in d.glob('*/summary.json'):
   v=json.loads(p.read_text());q=v.get('training') or v.get('condition') or next(iter(v.get('conditions',[])),None)
   if v.get('status')=='complete' and q and 'peak_allocated_gib' in q.get('memory',{}):records.append((q['memory']['peak_allocated_gib'],p.parent.name,q))
  if records:
   peak,case,q=max(records,key=lambda x:x[0]);ratio=peak/q['memory']['system_total_gib']*100;summary+=f'- **최신 추가 run 최대 성공 메모리:** `{case}` **{peak:.3f}GiB /{ratio:.2f}%**. 이 값은 해당 최신run의 성공 조건에서의 최대이며,다른 run의 최대와 구분한다. probe이면 전체 epoch·독립test 평가 완료를 뜻하지 않는다.\n'
 if model=='cellpose':summary+='- **전체 실험 구분:** 추론 타일batch384는43.397GiB/35.66%;FP32 파인튜닝 용량probe batch26은90.520GiB/74.39%. 전체10epoch·test평가·재로드는batch1에서 완료(AP50=0.9278). batch28은100GiB allocator 예산에서OOM;80% 성공은 미확인.\n'
 if model=='scgpt':summary+='- **실제1만세포 파인튜닝 완료:** batch284는98.003GiB/80.54%,1epoch442.37초,test5000세포Accuracy0.4674,재로드검증 완료. 이전고정padding batch292의82.79%는 별도 짧은probe이며 실제1만세포 조건과 다르다.\n'
 if model in ['colabfold','clair3']:summary+='- **학습 상태:** 이번 실행에서 훈련·미세조정은 수행하지 않았다. 참고 구조/진실변이는 평가 자료이며 학습 정답으로 사용한 것이 아니다.\n'
 summary+='- **시간·정확도 해석:** 결과는 같은 행의 입력·배치·학습범위에만 연결한다. 메모리 상승 자체가 처리량·정확도 상승을 보장하지 않는다.\n'
 content=marker+'\n'+summary+end+'\n\n'
 if marker in s:s=re.sub(re.escape(marker)+r'[\s\S]*?'+re.escape(end)+r'\n*',lambda _:content,s)
 else:
  m=re.search(r'^## 4\.[^\n]*\n',s,re.M);s=s[:m.end()]+'\n'+content+s[m.end():]
 path.write_text(s);shutil.copy2(path,ROOT/filename)
print('Updated model/input tables and conclusions in all10Markdown reports')

# Refresh control-factor evidence in the same ten reports.
import subprocess,sys
subprocess.run([sys.executable,str(ROOT/'scripts/real_training_v3/factor_reports.py')],check=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>단계별 입력·출력 확인/보고 원문: build_reports.py</summary>

<!-- RESTORE_FILE {"path": "scripts/report_stages/source_snapshots/build_reports_9e7b25407dfd214d.py", "sha256": "9e7b25407dfd214d86df35adcb20fdda2f9eb7ae6d9d88eabb21e9fe46c4d65c", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import json,re,hashlib,shutil,platform
ROOT=Path('/home/hyein/ai-benchmark');PUB=Path('/home/hyein/ai-benchmark-publication');MAP={'dnabert2':'DNABERT-2.md','ntv2':'Nucleotide_Transformer_V2.md','rnafm':'RNA-FM.md','openspliceai':'OpenSpliceAI.md','scvi':'scVI.md','cellpose':'Cellpose.md','scgpt':'scGPT.md','clair3':'Clair3.md','colabfold':'AlphaFold2.md','esm2':'ESM-2.md'}
def j(x):return '```json\n'+json.dumps(x,ensure_ascii=False,indent=2)+'\n```\n'
def vector(x):return '['+', '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in x)+']'
def shape(s,k):
 x=s['arrays'][k];return str(x['shape'])+' '+x['dtype']
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join('---' for _ in headers)+'|\n'+''.join('|'+ '|'.join(str(v).replace('|','／') for v in row)+'|\n' for row in rows)+'\n'
def artifact(s,items):return table(['단계별 저장 배열','실제 shape·dtype','의미'],[(k,shape(s,k),desc) for k,desc in items])+'로컬 전체 배열: `'+s['saved_arrays']+'`. `arrays.npz::key`는 같은NPZ 파일 안의 배열이름이다. 아래 재현코드로 저장모델과 입력에서 다시 확인할 수 있다.\n\n'
def cpu_note(s):
 d=s.get('saved_gpu_output_max_abs_difference',s.get('reference_max_abs'))
 return '**값의 출처:** 저장된 모델을 CPU에서 새로 재확인한 중간값이다. 원GPU 자원계측 중에 캡처했던 것처럼 해석하지 않는다. '+(f'같은 저장GPU 출력과 최대 절대차이는 **{d:.3e}**이다. ' if d is not None else '')+'이 CPU 확인시간은 기존 GPU 벤치마크 시간에 포함하지 않는다.\n\n'
TRACES={}
for model,filename in MAP.items():
 if model=='esm2':continue
 r=ROOT/model;allp=list((r/'results').glob('**/stage_trace_cpu/trace.json'))+list((r/'results').glob('**/stage_trace_saved/trace.json'));assert allp,model;p=max(allp,key=lambda p:p.stat().st_mtime);s=json.loads(p.read_text());TRACES[model]=(p,s)
 t='### 실제 파일 → 전처리 → 모델 표현 → 최종 출력: 단계별 실제 예시\n\n'
 if model in ['dnabert2','ntv2','rnafm']:
  raw=s['raw_record'];classes={'dnabert2':18,'ntv2':2,'rnafm':28}[model];hidden={'dnabert2':768,'ntv2':1024,'rnafm':640}[model];task={'dnabert2':'18개 histone 표지 다중라벨 분류','ntv2':'enhancer 이진분류','rnafm':'RNA유형28종 분류'}[model]
  t+=f"이 예시는 **{task}**, 저장된 추가 전체backbone 파인튜닝 모델의 **test `{s['sample_id']}`**다. 아래 모든값은 같은 레코드에서 이어진다.\n\n"
  source={'dnabert2':'BEND CSV의 id,sequence,labels(원train1000 내부800/100/100분할)','ntv2':'data/inputs.json의 test: id,sequence,label','rnafm':'data/inputs.json의 test: id,sequence,type,label'}[model]
  t+='#### 0–1. 파일 레코드 — X와 정답 y\n\n'+source+'. 아래는 실제 레코드에서 앞80염기만 발췌한 표시이며, 모델에는 전체서열을 넣었다.\n\n'
  example={'id':s['sample_id'],'sequence_length':len(raw['sequence']),'sequence_first80':raw['sequence'][:80]}
  if model=='dnabert2':example['positive_label_indices']=json.loads(raw['labels']);example['target_18_multihot']=s['target']
  else:example['label']=raw['label'];example.update({'type':raw['type']} if 'type' in raw else {})
  t+=j(example)+'**X는 염기서열, y는 위 label**이다. 임베딩을 정답으로 쓰는 것이 아니다.\n\n'
  t+='#### 2. 토큰화 — 실제 모델 입력\n\n'+table(['항목','실제값'],[['tokens',shape(s,'input_ids')],['attention_mask',shape(s,'attention_mask')],['원서열 길이',str(s['raw_nt'])+'nt'],['유효token 수',s['valid_token_count']],['token ID 앞16개',vector(s['token_ids_first16'])]])
  t+=('DNA BPE는 몇nt가 몇token으로 바뀌는지 서열마다 달라진다.' if model=='dnabert2' else '6-mer 중심 토큰화로 염기수와token수가 다르다.' if model=='ntv2' else '염기당token+BOS/EOS이며, 원래전체pool의최대폭1022로padding해batch비교조건을고정했다.')+' 숫자ID는vocabulary index이고, 정답라벨번호와 다른 배열이다.\n\n'
  t+='#### 3. 본체 출력 — 위치별 표현\n\n`'+shape(s,'token_representations')+'`이며 각token에'+str(hidden)+'개 특징값이 생긴다. 첫위치의 앞8개 실제값: `'+vector(s['feature_first8'])+'`. 이 단계는 최종 class확률이 아니다.\n\n'
  t+='#### 4. Pooling — 서열당 벡터1개\n\n방식: **'+s['pooling']+'**. 출력 `'+shape(s,'embedding')+'`. 임베딩 앞8개 실제값: `'+vector(s['embedding_first8'])+'`.\n\n'
  t+='#### 5. 과제 head — 최종 예측으로 변환\n\n'+f'추가 파인튜닝의 linear head는 `{hidden}→{classes}`다. 기본보고서의 다른head/학습범위 조건과 구분한다.\n\n'
  t+=table(['단계','shape','의미'],[['head logits',shape(s,'logits'),'확률변환 전 실수'],['확률',shape(s,'probabilities'),'sigmoid:18개 독립확률' if model=='dnabert2' else f'softmax:{classes}개중한class의확률'],['판정',s['prediction'],'각확률>0.5를양성판정' if model=='dnabert2' else '최대확률class의ID']])
  t+='logits 앞8개: `'+vector(s['logits'][:8])+'`; 확률 앞8개: `'+vector(s['probabilities'][:8])+'`.\n\n'
  t+='#### 6. 정답과 비교\n\n'+table(['항목','실제값'],[['정답',s['target']],['예측',s['prediction']]])
  if model=='dnabert2':t+='정답은18개중복수양성인multi-hot이고, 예측도18개를독립판정한다. 이번추가통제실험의test micro-F1=0이며, 실행·재현검증을유효한분류성능달성으로해석하면안된다. 원래독립test와이번train내부분할test도구분한다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('input_ids','입력token'),('attention_mask','유효위치'),('token_representations','token별표현'),('embedding','서열당임베딩'),('logits','head출력'),('probabilities','class확률'),('target','정답')])+f"원GPU run의 `test_outputs.npy`는 **logits**이며 확률파일이 아니다. 값의 순서는 저장된 split_ids['test']와 일치한다.\n\n"+cpu_note(s)
 elif model=='openspliceai':
  raw=s['raw_record'];t+='**이 모델은 서열당임베딩벡터를분류하는방식이아니라 각염기위치의splice확률을출력한다.** 같은test구간 `'+s['sample_id']+'`를끝까지따라간다.\n\n'
  t+='#### 0–1. 실제 레코드와 정답\n\n`data/inputs.json`의 test 레코드다.\n\n'+j({k:raw[k] for k in ['id','chromosome','strand','genomic_input_start','genomic_input_end']})+'원레코드 X는20,000nt, 정답은가운데10,000위치의0/1/2주석이다. X를 `[2500:17500]`의15,000nt로, y를 `[2500:7500]`의5,000위치로대응되게선택했다. 0=nonsplice,1=acceptor,2=donor,-1=충돌주석무시다.\n\n'
  t+='#### 2. One-hot 입력\n\n'+shape(s,'onehot')+';채널순서는A/C/G/T, 앞12염기는 `'+s['input_first12_bases']+'`. 한위치가A이면 `[1,0,0,0]`이고 token ID로변환하는모델이아니다.\n\n'
  t+='#### 3–4. 합성곱 특징과 문맥 처리\n\n첫합성곱의실측특징은 `'+shape(s,'initial_conv_features')+'`;첫채널앞8개는 `'+vector(s['feature_first8'])+'`. 이후dilated residual CNN이넓은문맥을처리하며,단백질언어모델처럼전체서열을평균임베딩하진않는다.\n\n'
  t+='#### 5. 최종 위치별 출력\n\n출력은 **'+shape(s,'output_probabilities')+'**이다. 위치5,000개각각에3종확률이있다. 앞5위치실제확률(행=위치,열=nonsplice/acceptor/donor):\n\n'+j(s['first5_position_probabilities'])
  t+='#### 6. 같은 위치의 정답과 비교\n\n정답5,000개중라벨별개수: `'+str(s['target_counts'])+'`. 주석상splice위치의첫index들(출력window내0-based): `'+str(s['positive_target_positions_first8'])+'`. 전체구간한개에라벨하나를붙인과제가아니다. 평가는위치별acceptor/donor AUPRC등이며,seed10단일모델미세조정과5모델ensemble추론은별도조건이다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('onehot','DNA입력'),('initial_conv_features','첫CNN특징'),('output_probabilities','위치별3확률'),('target','각출력위치정답')])+'원GPU `test_outputs.npy`는 `[36,3,5000]`의확률이다.\n\n'+cpu_note(s)
 elif model=='scvi':
  t+='**scVI는처음부터학습한count 생성VAE이며,세포유형분류head를붙이지않았다.** 저장모델의 실제세포 `'+s['sample_id']+'`를사용한다.\n\n'
  t+='#### 0–1. 원본 파일과 행렬의 한 행\n\n`data/processed_counts.h5ad`와저장모델의`adata.h5ad`에raw-count가보관된다. 원본처리행렬은세포12,721×유전자2,000이다. 행1개=세포1개,열1개=유전자1개다.\n\n'+j({'cell_id':s['sample_id'],'nonzero_gene_names_first8':s['gene_names_first_nonzero'],'raw_counts_same_genes':s['counts_first_nonzero'],'biological_batch_sample_id':s['batch_covariate']})+'생물학적sample_id는GPU batch크기가아니다. 세포유형라벨을정답으로학습한모델도아니다.\n\n'
  t+='#### 2. 모델 입력\n\ncounts는 `'+shape(s,'raw_counts')+'`이며batch공변량을함께사용한다. Raw count를입력하고같은count의생성분포를학습한다.\n\n'
  t+='#### 3. Encoder 특징\n\n실제encoder hidden출력은 `'+shape(s,'encoder_hidden')+'`이다.\n\n'
  t+='#### 4. 세포별 latent\n\n`give_mean=True`로사후분포평균을반환했다. 실측 '+shape(s,'latent')+'의전체10개값은:\n\n```text\n'+vector(s['latent'])+'\n```\n\n'
  t+='#### 5–6. 출력과 학습목표\n\n이번latent추론의최종출력은세포당10차원벡터다. 별도분류기확률/세포유형이아니다. **훈련시에는**decoder의count분포와raw count를이용한재구성항및KL항으로VAE를학습한다. 따라서정답구조는원래count이고외부cell-type정답을만들지않았다. 이예제CPU추적은latent경로만확인했으며decoder중간값을저장했다고주장하지않는다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('raw_counts','원래count행'),('encoder_hidden','인코더중간표현'),('latent','10차원최종latent')])+'전체입력의GPU추론배열은각batch의 `outputs.npy`;이예제는그첫행과대조했다.\n\n'+cpu_note(s)
 elif model=='cellpose':
  t+='**Cellpose는이미지를평균임베딩하나로줄여분류하는모델이아니라 픽셀별flow·cell probability를만들어instance mask로변환한다.** 실제test이미지 `'+s['sample_id']+'`를사용한다.\n\n'
  t+='#### 0–1. 원본 이미지와 정답 mask\n\n'+table(['역할','실제파일','shape·dtype'],[['입력X',s['raw_image_file'],shape(s,'raw_image')],['정답y',s['raw_mask_file'],shape(s,'ground_truth_mask')]])+f"원본은520×696 흑백uint16 TIFF,정답PNG를공식방법으로연결성분instance-ID로변환하면핵 **{s['truth_instances']}개**다. mask값0은배경,양의정수각각은서로다른객체다.\n\n"
  t+='#### 2. 정규화·타일화 → 네트워크 입력\n\n공식백분위정규화와256×256타일화를적용한다. 이번CPU확인의첫실제입력은 **'+shape(s,'first_model_input')+'**. 추론경로의흑백1채널입력이고,파인튜닝의3채널crop(흑백+0채널2개)과구분한다. 원본520×696전체가한번에같은모양으로GPU에올라가는것은아니다.\n\n'
  t+='#### 3–4. SAM encoder의 공간 특징\n\n첫encoder block의실제출력은 **'+shape(s,'first_encoder_block_features')+'**. 공간격자에채널별특징이있고, `[1,1280]`같은이미지전체임베딩이최종결과인모델이아니다. 펼친첫8값: `'+vector(s['first_feature_values'])+'`.\n\n'
  t+='#### 5. 타일 합치기 → flow·cell probability\n\n'+table(['출력','shape·dtype','의미'],[['flow',shape(s,'flows'),'각픽셀의Y/X방향흐름'],['cell probability',shape(s,'cell_probability'),'cellprob_threshold를적용할네트워크점수;0~1로보정된확률로단정하지않음']])+ '타일결과를원래이미지크기에맞춰합친다. 이후flow dynamics·threshold·크기필터를거쳐픽셀을서로다른객체로묶는다.\n\n'
  t+='#### 6. 최종 instance mask와 정답 비교\n\n최종예측은 **'+shape(s,'predicted_mask')+'**,예측객체'+str(s['predicted_instances'])+'개다. 원래같은크기의정답mask와객체를매칭해AP50/AP75/AP90을계산한다. mask ID는객체식별번호이지세포유형라벨이아니다. 이예제CPU/GPU mask일치: **'+str(s['cpu_gpu_mask_equal'])+'**. 이번CPU타일batch는1이고원GPU평가는8이므로새속도비교로쓰지않는다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('raw_image','입력이미지'),('ground_truth_mask','정답객체mask'),('first_model_input','첫네트워크타일'),('first_encoder_block_features','encoder공간특징'),('flows','전체이미지flow'),('cell_probability','전체이미지점수'),('predicted_mask','최종객체IDmask'),('saved_gpu_mask','원GPU결과대조')])+cpu_note(s)
 elif model=='scgpt':
  t+='같은실제test세포 `'+str(s['sample_id'])+'`를사용한다. **세포1개=행렬1행이며,그행의유전자별발현값을token으로바꾼다.** 이번예제는Tabula Sapiens161클래스파인튜닝모델이고pancreas13클래스와별도다.\n\n'
  t+='#### 0–1. 원본 h5ad의 count 행\n\n원본Tabula Sapiens는483,152세포×58,482유전자이고원본`raw/X`에서실제선택세포한행을확인했다. 아래원본유전자index와count는같은순서다.\n\n'+j(s.get('raw_count_row',{'note':'원본count의새확인값없음'}))+'정답은세포metadata의cell_type이며,유전자ID나발현량이아니다.\n\n'
  t+='#### 2. 유전자 선택·발현 binning → 두 입력 배열\n\n공개vocabulary와매칭한유전자에서세포별nonzero gene을선택하고발현을bin으로변환한다. 최대1200자리에는CLS가포함된다. 이번세포의입력앞16개는:\n\n'+j({'gene_ids':s['first16_gene_ids'],'gene_names_same_order':s['first16_gene_names'],'expression_bins_same_order':s['first16_expression_bins']})+table(['배열','실측shape·dtype'],[['gene_ids',shape(s,'gene_ids')],['expression_bins',shape(s,'expression_bins')]])+'처음의CLS와특수발현값은일반유전자의raw count가아니다. bin값도원래UMI count와다른전처리결과다. 원본58482열전체를그대로1200열로복사한것이아니다.\n\n'
  t+='#### 3. token별 Transformer 표현\n\n실제출력: **'+shape(s,'token_representations')+'**. 각CLS/유전자token마다512개특징값이생긴다.\n\n'
  t+='#### 4. 세포 임베딩\n\nCLS위치의벡터를선택해 **'+shape(s,'cls_embedding')+'**로만든다. 앞8개실제값:`'+vector(s['embedding_first8'])+'`. 이분류경로는raw CLS를사용하며,별도임베딩추론의L2정규화와구분한다.\n\n'
  t+='#### 5. 분류 head\n\nhead logits는 **'+shape(s,'logits')+'**,softmax확률도 `[1,161]`이다. 세포당161개유형중최대확률유형을선택한다.\n\n'
  t+='#### 6. 실제 cell_type 정답과 비교\n\n'+table(['항목','실제값'],[['예측class',str(s['predicted_id'])+' /'+str(s['predicted_label'])],['예측최대확률',f"{s['top_probability']:.6f}"],['정답class',str(s['true_id'])+' /'+str(s['true_label'])]])+'전체test성능은별도5000세포지표이며,한세포예측결과가전체정확도를뜻하지않는다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('gene_ids','유전자ID입력'),('expression_bins','발현bin입력'),('token_representations','위치별특징'),('cls_embedding','세포임베딩'),('logits','class점수'),('probabilities','class확률'),('target','실제유형ID')])+'원GPU배열은 `test_logits.npy` `[5000,161]`와 `test_rows.npy`;같은행을재확인했다.\n\n'+cpu_note(s)
 elif model=='clair3':
  t+='**Clair3는단백질임베딩모델이아니라 정렬된read에서후보변이특징을만들어변이확률을계산한다.** 이예제는실제cache의첫후보이며,캐시에genomicposition ID가저장되지않았으므로BAM의특정read·표시된VCF행과동일좌표라고연결하지않는다.\n\n'
  t+='#### 0–1. 원본 입력과 평가 정답\n\n입력은 `'+ '`, `'.join(s['raw_input_files'])+'`및BAM index다. BAM의정렬read·염기품질·mapping정보와참조유전체에서후보특징을계산한다. GIAB truthVCF와평가구간은정확도평가용이며이번에학습한라벨tensor가아니다.\n\n'
  t+='#### 2. 후보 특징 tensor\n\n'+table(['모델분기','입력shape','dtype'],[(k,v['shape'],v['dtype']) for k,v in s['branches'].items()])+'pileup의33위치×18특징과full-alignment의89read행×33위치×8채널이다. batch는DNA문자수나원본read총수가아니라동시에처리하는후보tensor수다.\n\n'
  t+='#### 3–4. 분기별 특징 계산\n\npileup은양방향LSTM,full-alignment는CNN경로로후보의표현을계산한다. **이CPU확인에서는그내부hidden tensor를캡처하지않았으므로실제중간embedding값은N/A**다. 존재하지않는평균임베딩단계를추가하지않는다.\n\n'
  t+='#### 5. 최종 후보별 확률\n\n'
  for k,v in s['branches'].items():t+='- **'+k+'**:출력'+str(v['output_shape'])+',앞24개실제값 `'+vector(v['probability_first24'])+'`. 원GPU같은행과최대차이 '+f"{v['reference_max_abs']:.3e}"+'.\n'
  t+='\nP의24출력은GT21+genotype3, F의90출력은GT21+genotype3+indel길이33+33개head를연결한것이다. 전체90개를단일softmax90클래스로해석하지않는다.\n\n'
  t+='#### 6. 후보확률 → 실제 VCF·평가\n\n공식pipeline은후보별확률과위치·ref/alt정보를조합해VCF의CHROM/POS/REF/ALT/QUAL/GT등을작성한다. cache예제의좌표mapping이없어이확률과특정VCF행의직접대조는미검증이다. 원래BAM→VCF실행·truth평가결과는아래상세에보존했다. 훈련·파인튜닝은수행하지않았다.\n\n'
  t+='#### 7. 저장 파일\n\n'+artifact(s,[('Clair3_P_input','pileup후보입력'),('Clair3_P_probabilities','pileuphead확률'),('Clair3_F_input','full-alignment후보입력'),('Clair3_F_probabilities','full-alignmenthead확률')])+cpu_note(s)
 elif model=='colabfold':
  raw=s['raw_record'];t+='**AlphaFold2는단백질서열과MSA를받아구조를만든다. 평균임베딩뒤에회귀head를붙인ESM-2 과제와다르다.** 실제독립query `'+s['sample_id']+'`의저장된입력·출력을확인했다.\n\n'
  t+='#### 0–1. 원본 서열·MSA\n\n```fasta\n>'+s['sample_id']+'\n'+raw['sequence']+'\n```\n\n서열115AA,feature sampling전원MSA깊이'+str(raw['msa_depth_before_feature_sampling'])+'이다. 이query에는정답구조를학습시킨것이아니며,이번작업은추론이다.\n\n'
  t+='#### 2. 모델 입력 features\n\n'+table(['실제NPZ key','실측shape·dtype'],[(k,str(v['shape'])+' '+v['dtype']) for k,v in s['feature_shapes'].items() if k in ['aatype','target_feat','msa_feat','msa_mask','extra_msa','extra_msa_mask','residue_index']])+'recycle/ensemble feature축,MSA행,잔기축을원본서열개수나독립query batch와혼동하지않는다. zero-padding된MSA행은mask로제외한다.\n\n'
  t+='#### 3–4. Evoformer 표현·구조 모듈\n\nEvoformer가MSA와잔기쌍표현을처리하고structure module이원자좌표를계산한다. **원GPU실행은내부Evoformer embedding을저장하지않았으므로중간숫자는N/A**다. 이번확인은저장된features와최종예측배열을읽은것이며새JAX/GPU추론을실행한것이아니다.\n\n'
  t+='#### 5. 최종 구조·신뢰도 출력\n\n'+table(['출력','shape·dtype','실제예시'],[['atom_positions',str(s['output_shapes']['atom_positions']['shape'])+' '+s['output_shapes']['atom_positions']['dtype'],'첫잔기CA좌표 '+vector(s['first_residue_CA_xyz'])],['plddt',str(s['output_shapes']['plddt']['shape'])+' '+s['output_shapes']['plddt']['dtype'],'앞8값 '+vector(s['plddt_first8'])]])+'좌표의마지막3개값은X/Y/Z좌표이며,37원자slot중유효원자는residue별mask를따른다.\n\n'
  t+='#### 6. 정답·점수 해석\n\n평균pLDDT는 '+f"{s['mean_plddt']:.3f}"+'이며 **모델의신뢰도**다. 이query의정답실험구조가없어이를구조정확도로쓰지않는다. PDB1UBQ등정답구조대조는별도입력조건이다.\n\n'
  t+='#### 7. 저장 파일\n\n`'+s['pdb_file']+'`의실제처음5개ATOM레코드:\n\n```text\n'+'\n'.join(s['first5_PDB_ATOM_lines'])+'\n```\n\n저장최종NPZ는atom_positions/plddt이며준비features와함께 `'+s['saved_arrays']+'`에사본을보관했다. 별도3D좌표/PDB출력이지class라벨이아니다.\n\n'
 templates=ROOT/'scripts/report_stages/templates';templates.mkdir(exist_ok=True)
 (templates/(model+'.txt')).write_text(t)
# Record actual replay source snapshots and insert environment/measurement overview into all ten existing reports.
for model,filename in MAP.items():
 p=PUB/filename;s=p.read_text();rr=ROOT/model/'results';env={};source_path='상세원본환경기록 참조'
 candidates=[]
 for pointer in ['REAL_TRAINING_LATEST','LATEST']:
  pp=rr/pointer
  if pp.exists():
   dd=rr/pp.read_text().strip();candidates+=sorted(dd.glob('*/summary.json'))+[dd/'summary.json']
 candidates+=list(rr.glob('*/summary.json'))
 for q in candidates:
  if q.exists():
   v=json.loads(q.read_text())
   if isinstance(v.get('environment'),dict) and v['environment']:env=v['environment'];source_path=str(q.relative_to(ROOT));break
 envrows=[('장비·공유메모리','DGX Spark,NVIDIA GB10,ARM64;시스템공유RAM121.6875GiB. CPU/GPU RAM을별도로더하지않음'),('환경기록출처',source_path),('Python·프레임워크',str({k:v for k,v in env.items() if k in ['python','torch','jax','transformers','cuda','scgpt','cellpose','platform','precision']}) or 'N/A;5절의환경및실행코드참조'),('실험병렬정책','GPU벤치마크한번에1개. 이단계별CPU검증은별도작업이며GPU측정시간과합치지않음'),('GPU-Util','nvidia-smi utilization.gpu(%);관측기간중커널활동시간비율. 최대FLOPS/메모리비율아님'),('GPU메모리','PyTorch allocated/reserved및peak;JAX는bytes_in_use/peak_bytes_in_use별도. 모델적재직후·실행구간을분리'),('프로세스·시스템메모리','psutil RSS,MemAvailable,swap;시계열의sampled peak/최소값이며정확한연속최고치아님'),('표본간격','기본0.5초대기+GPU조회시간;측정구간의평균·최대·표본수저장. 짧은실행은동일조건반복하고고유입력과반복처리량구분'),('시간동기화','PyTorch구간앞뒤CUDA synchronize;JAX block_until_ready. 비동기호출반환시간만측정하지않음'),('시간포함범위','조건별로입력준비/전송/forward/backward/optimizer/validation/save포함여부를기록. 서로다른범위의시간을직접배수비교하지않음'),('loading·compile·warmup','구간을분리한다. JAX누적allocator peak가compile/warmup을포함하는경우본문에명시하고CUDA구간peak와직접동일시하지않음'),('메모리기준값과증가량','첫forward전모델적재기준값과같은run의실행peak를비교. 증가량에는학습gradient/optimizer/중간활성값이포함'),('누락·관측한계','미지원/미관측값은N/A. 장치전체GPU-Util에는다른프로세스·디스플레이활동이섞일수있음. 단일실행은통계적반복검증으로쓰지않음')]
 if model=='colabfold':envrows.append(('JAX메모리해석','XLA사전할당비활성. bytes_in_use와누적peak를분리하며미지원CUDA allocated/reserved는N/A'))
 content='<!-- ENV_MEASUREMENT_OVERVIEW_START -->\n### 실행 환경·자원 측정 방법 요약\n\n'+table(['항목','이번 보고서의 측정 기준'],envrows)+'<!-- ENV_MEASUREMENT_OVERVIEW_END -->\n\n'
 if '<!-- ENV_MEASUREMENT_OVERVIEW_START -->' in s:s=re.sub(r'^<!-- ENV_MEASUREMENT_OVERVIEW_START -->\n[\s\S]*?^<!-- ENV_MEASUREMENT_OVERVIEW_END -->\n*',lambda _:content,s,flags=re.M)
 else:
  a=re.search(r'^## 2\.',s,re.M).start();i=s.index('\n',a)+1;s=s[:i]+'\n'+content+s[i:]
 p.write_text(s);shutil.copy2(p,ROOT/filename)
print('Generated nine model-specific stage sections and environment/measurement tables for all ten')
````
<!-- END_RESTORE_FILE -->

</details>

<details><summary>단계별 입력·출력 확인/보고 원문: standardize_overview.py</summary>

<!-- RESTORE_FILE {"path": "scripts/report_stages/source_snapshots/standardize_overview_da8843e93a5751fd.py", "sha256": "da8843e93a5751fd2140d5b1ad070fb01819e3afe0138761bc162e9aa0b35f85", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,re,shutil
ROOT=Path('/home/hyein/ai-benchmark');PUB=Path('/home/hyein/ai-benchmark-publication')
META={'esm2': ('ESM-2.md', 'esm2_t33_650M_UR50D', 'Transformer encoder,33층,1280차원', '651,043,254개(651.04M);encoder649,400,320개', 'FP32', '단백질 서열 →1280차원 임베딩 / GFP 형광값 예측', '추론:UniProt reviewed 단백질. 추가 run은900–950AA이며 GFP 학습 서열과 별개', 'TAPE fluorescence의 실제 GFP 변이체 약237AA. 기본 train128/val64/test128;추가 비교1024/128/512', 'X:아미노산→int64 tokens[B,L+2];y:GFP log_fluorescence[B]. 라벨 표준화에는 train 평균·표준편차만 사용', '기본:일부 backbone 층+회귀head 파인튜닝;추가:전체 backbone+linear 회귀head 파인튜닝', 'data/fluorescence/fluorescence_{train,valid,test}.lmdb', '긴 UniProt 추론 서열에는 본 실험의 GFP 정답이 없다. 부분 미세조정과 전체 미세조정 수치를 섞지 않는다'), 'colabfold': ('AlphaFold2.md', 'ColabFold1.6.2,alphafold2_ptm/alphafold2_multimer_v3', 'Evoformer+Structure Module', '실제PTM tree92,907,930개(92.91M)/Multimer93,223,705개(93.22M);원본NPZ 구성은 상세표 참조', '기본 BF16;별도 실험vmap FP32', '단백질 서열+MSA →3D좌표,pLDDT/PAE', '서로 다른 UniProt115AA 단백질4개,PDB1UBQ/1LYZ/1TIM;복합체3G5O A/B 예제', '훈련·파인튜닝 미실시. 이번 작업에 사용한 학습 데이터셋 없음', '추론 X:서열·MSA features. 참고 PDB는 평가용 정답 구조이며 학습에 사용하지 않음', '추론만 수행. 설치된 ColabFold 실행 경로에서 훈련하지 않음', 'data/queries.fasta;각run processed_features.npz/MSA', 'pLDDT는 신뢰도이며 실제 정답 구조 대비 정확도와 다르다. vmap은 별도 실험적 추론 경로'), 'scvi': ('scVI.md', 'SCVI,seed42무작위초기화;사전학습미사용 checkpoint', 'ZINB count 생성 VAE,hidden128,1층,latent10', '1,293,526개(1.294M);latent z_encoder258,964개;첫backward gradient778,884개', 'FP32', 'raw-count 세포×유전자+sample_id →10차원 latent /count 분포', 'PBMC4k+8k raw count에 저장 모델을 적용해 latent 추출. 세포유형 분류 과제 아님', '실제10x PBMC4k+8k12721세포×2000HVG. train10176/validation1272/test1273;작은 중첩 train부분집합도 별도 비교', 'X:counts[B,2000];재구성 목표는 같은 raw count. sample_id는 실제 생물학적batch 공변량. cell-type 정답 없음', '사전학습 가중치 없이 무작위 초기화에서 처음부터 학습,400epoch. 파인튜닝 아님', 'data/processed_counts.h5ad;results/<input_run>/splits.json', '재구성 기반 비지도 학습이며 생물학적batch와 GPU batch는 다르다. 통합 진단을 세포유형 정확도로 부르지 않음'), 'cellpose': ('Cellpose.md', 'Cellpose4.2.1,cpsam_v2（Cellpose-SAM）', 'SAM ViT-L 기반 이미지 분할', '304,628,930개(304.63M)', '추론BF16;파인튜닝FP32', '흑백 이미지 →핵/세포별 서로 다른ID의 instance mask', 'BBBC039200장520×696 uint16 핵 형광 이미지. 뇌 추가 입력은 NucMM-M192³ micro-CT patch4개', 'BBBC039 공식100/50/50;기본부분집합16/8/16. 추가 학습은 빈mask2장 제외 후 train98/validation50/test50', 'X:원본520×696에서 공식 무작위crop[B,3,256,256](흑백+0채널2개). y:같은 crop에 맞춰 변환한 instance mask·cell probability·flow. 세포유형 정답 아님', 'cpsam_v2 사전학습 가중치에서 backbone·분할출력층 파인튜닝. 실제 일부 파라미터는 gradient가 관측되지 않아 수를 별도 기록', 'data/input_audit.json;data/images/images/*.tif;data/masks/masks/*.png;추가run/prepared/*_mask.npy', 'batch26은2epoch 파인튜닝 용량 측정;전체10epoch 독립test·재로드 완료는batch1. 뇌patch는zero-shot 추론만 수행'), 'dnabert2': ('DNABERT-2.md', 'DNABERT-2-117M,revision7bce263b15377fc15361f52cfab88f8b586abda0', 'DNA BPE tokenizer+Transformer encoder,hidden768', 'encoder117,068,544개(117.07M);18출력linear head13,842개 추가', 'FP32', 'DNA →BPE tokens →임베딩 /18종histone 다중라벨', '실제BEND GRCh38구간.512/1000/2000/4000bp 비교;bp수와token수는 별개', 'BEND histone_modification 실제구간 및18개표지. 기본train1000. 추가비교는 원train1000 안에서 seed42로800/100/100 분리', 'X:input_ids/attention_mask[B,Ltokens]. y:[B,18] float32 multi-hot이며18중 하나를 고르는 과제 아님. 신규추론4000bp pool padding851tokens', '공개encoder+linear 다중라벨head 전체 파인튜닝. 사용하지 않는pooler 등 분기와 실제gradient 수는 별도 기록', '/home/hyein/dnabert2-benchmark/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv', '추가800/100/100은 기존train 내부 분할이며 원래 독립test 결과를 대체하지 않음. 염색체독립 검증 아님'), 'ntv2': ('Nucleotide_Transformer_V2.md', 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species', 'DNA6-mer 중심 tokenizer+ESM 구현 Transformer', 'MLM498,345,436개(498.35M);embedding backbone493,084,113개;기본CLS분류구성494,135,763개', 'FP32', 'DNA →token 임베딩 /enhancer 이진분류', 'GRCh38 실제긴구간. 추가추론:서로 겹치지 않는6000nt window1024개,약1001tokens', '실제enhancer 라벨 서열199–200nt. train512/validation128/test400', 'X:token_ids/attention_mask[B,실제token길이]. y:[B]정수0/1. 긴6000nt 추론 window에는 이 enhancer 정답이 없음', '기본:전체backbone+원생CLS MLP 파인튜닝. 추가:전체backbone+masked-mean linear head 파인튜닝,별도 조건', 'data/inputs.json의 train/validation/test;data/real_training_v3/inference_inputs.json추론전용', '긴 유전체 추론window와 짧은enhancer 학습 서열을 구분. 정확도는 해당 라벨·split에서만 평가'), 'clair3': ('Clair3.md', 'Clair3 v2.0.4,ONT pileup.pt/full_alignment.pt', 'pileup 양방향LSTM 및 full-alignment CNN 후보분류 네트워크', 'Clair3_P2,074,520개(2.075M);Clair3_F2,986,522개(2.987M);단계별 실행', 'FP32', 'long-read BAM+참조유전체 →후보특징 →SNV/Indel VCF', 'HG003 ONT 실제BAM+GRCh38 chr20소구간. cache 고유feature P5708/F894', '이번 작업에서 학습·파인튜닝 미실시. 본 실험용 훈련 정답tensor 없음', '추론 X:P[B,33,18],F[B,89,33,8]. GIAB truthVCF/평가구간은 변이검출 정확도 평가용이며 학습에 사용하지 않음', '공식학습기능 존재와 별개로 이번에는 사전학습ONT 가중치의 추론만 수행', 'data/HG003_chr20_demo.bam;data/GRCh38_no_alt_chr20.fa;candidate run/*_inputs.npy', '후보batch는read개수가 아님. 작은구간 후보pool을 전부 사용한 결과는 전장유전체 최대메모리 한계가 아님'), 'rnafm': ('RNA-FM.md', 'RNA-FM ncRNA,RNA-FM_pretrained.pth（fm0.2.2）', 'RNA Transformer12층,hidden640', 'backbone99,521,546개(99.52M);28종linear head17,948개', 'FP32', 'RNA →640차원임베딩 /RNA유형 분류', '기본:RNAcentral 실제RNA. 추가추론:GENCODEv47 lncRNA transcript 첫1000nt,T→U,고유2400개', 'RNAcentral1400서열/28종류. train896/validation224/test280. 실제길이가 다양하며 최대1020nt', '기본head-only X:동결RNA-FM에서 추출한embedding[B,640]. y:[B]28종 유형ID. 추가backbone파인튜닝 X:RNA tokens[B,L+2]와 같은유형정답', '기본:본체동결·분류기만학습. 추가전체backbone+linear 분류기 파인튜닝은 해당run의 실제 완료상태를 확인', 'data/inputs.json의 train/validation/test;data/input_audit.json의 label매핑;GENCODE추론전용', '기본head-only 메모리를 전체RNA-FM 미세조정 메모리로 쓰지 않음. 신규lncRNA 추론2400개는28종 지도학습셋과 별개'), 'openspliceai': ('OpenSpliceAI.md', 'MANE10000nt,seed10–145개모델;파인튜닝은seed10', '팽창합성곱 residual CNN,10000nt 문맥', '단일696,739개(0.697M);5모델합계3,483,695개(3.484M)', 'FP32', 'DNA4채널one-hot →위치별nonsplice/acceptor/donor 확률', '기본:GRCh38실제156구간. 추가추론:고유비중첩15000nt window16000개,5모델ensemble', 'exon주석을확인한 GRCh38영역 train96/validation24/test36,염색체별분리. 원레코드20000nt/정답10000nt→실사용15000nt/정답5000nt', 'X:[B,4,15000] A/C/G/T one-hot. y:[B,5000]정수0/1/2;충돌위치-1무시. 구간전체에 정답1개가 아니라 위치별정답', 'seed10 단일모델 전체파인튜닝. seed10–14의5모델ensemble은 추론용이며5개동시학습 아님', 'data/inputs.json의 sequence및labels_10000;source/openspliceai/variant/annotations/grch38.txt', '추가무라벨reference window를 임의정답으로 학습시키지 않음. 지도학습정답은exon주석에서 변환'), 'scgpt': ('scGPT.md', 'scGPT whole-human,best_model.pt', 'Transformer12층,hidden512;geneID+expression bin embedding', 'encoder50,278,400개(50.28M);기본추론구성51,856,898개;13클래스구성51,338,254개;161클래스구성은run참조', 'FP32', '세포×유전자발현 →유전자token/bin →512차원임베딩 /세포유형분류', 'pancreas테스트세포;Tabula Sapiens 실제고유30만세포embedding 추론. 전부미세조정한 것은 아님', 'pancreas기본 train1000/val2120/test4218;전체8480/2120/4218. Tabula donor분리238868/35738/25394 중 실제train10000/val2000/test5000 사용', 'X:gene_ids[B,L]int64와expression_bins[B,L]float32,L≤1200(CLS포함). y:[B]Celltype/cell_type유형ID. 유전자가 분류정답인 것은 아님', '공개encoder/value encoder/Transformer+분류기 파인튜닝. pancreas13종과Tabula161종은 별도분류설정. 미사용분기와실제trainable/gradient수를 구분', 'data/pancreas_demo/*.h5ad;data_external/tabula_sapiens/prepared_300k_seed42/{gene_ids,expression_bins}.npy;추가run/labels.npy및*_rows.npy', 'batch는동시에처리하는세포수,각세포는표현행렬1행.1200은CLS+유전자token자리이며 세포수가아님')}
def ko(s):return s
for model,meta in META.items():
 filename,checkpoint,arch,params,precision,io,infer,training,xy,method,paths,limit=meta
 def human_count(match):
  n=int(match.group(1).replace(',',''));human=f'약{n/100000000:.2f}억 개' if n>=100000000 else f'약{n/10000:,.1f}만 개';return match.group(0)+'('+human+')'
 params=re.sub(r'(\d{1,3}(?:,\d{3})+)개',human_count,params)
 path=PUB/filename;s=path.read_text();table='|항목|이번 실험에서 사용한 모델|\n|---|---|\n'
 for key,value in [('체크포인트·버전',checkpoint),('구조',arch),('전체·구성별 파라미터 수',params),('정밀도',precision),('모델 입력 → 출력',io),('가중치와 실행 메모리의 차이','파라미터 수×dtype bytes는 이론 가중치 용량이다. 실행 peak에는 입력·중간 활성값이 더해지고,파인튜닝에는 gradient/optimizer 상태도 추가된다.'),('메모리% 정의','CUDA allocated peak / CPU·GPU 공유 RAM121.6875GiB ×100; GPU-Util과 별개. AlphaFold2는 JAX 계측을 별도로 표시')]:table+='|'+key+'|'+ko(value)+'|\n'
 table+='\n### 실제 데이터와 학습 방식 구분\n\n|구분|데이터·실제 처리 내용|\n|---|---|\n'
 for key,value in [('추론 데이터',infer),('학습·검증·test 데이터',training),('학습 입력 X / 정답 y',xy),('가중치 갱신 방식',method),('원본·변환 입력 저장 위치','`'+paths+'`'),('해석 범위',limit)]:table+='|'+key+'|'+ko(value)+'|\n'
 table+='\n**용어:** 처음부터 학습=무작위 초기화에서 시작; 전체/일부 층 파인튜닝=공개 가중치에서 해당 부분 갱신; 분류기만 학습=본체를 동결하고 head만 갱신. **용량probe는 학습 방식이 아니라 짧은 자원 측정 목적**이며,실제로 수행한 가중치 갱신 방식과 함께 표기한다. 원본 전체 데이터 수와 해당 run에서 실제 사용한 부분집합 수를 구분한다.\n\n'
 template=ROOT/'scripts/report_stages/templates'/(model+'.txt')
 if template.exists():table+=template.read_text()
 elif (ROOT/model/'data/report_input_section.txt').exists():table+=(ROOT/model/'data/report_input_section.txt').read_text()
 # Bring measured post-load baselines forward; do not infer them from parameter counts.
 candidates=[];rr=ROOT/model/'results'
 for pointer_name in ['REAL_TRAINING_LATEST','BATCH_LATEST','LATEST']:
  pp=rr/pointer_name
  if pp.exists():
   dd=rr/pp.read_text().strip();candidates+=sorted(dd.glob('*/summary.json'))+[dd/'summary.json']
 found={}
 for qp in candidates:
  if not qp.exists():continue
  vv=json.loads(qp.read_text());base=vv.get('model_only_postload',vv.get('model_postload'))
  if not isinstance(base,dict):continue
  training_case=any(x in qp.parent.name for x in ['training_b','train_b','probe_b'])
  role='파인튜닝 적재 기준' if training_case else '추론/기본 모델 적재 기준'
  if model=='clair3':role+=' ('+str(vv.get('model','분기 미기록'))+')'
  if 'allocated_gib' in base:
   if role in found:continue
   found[role]=(f"{base['allocated_gib']:.6f}",str(base.get('reserved_gib','N/A')),str(base.get('rss_gib','N/A')),str(qp.relative_to(ROOT)))
  elif 'jax' in base and role not in found:
   value=base['jax'].get('bytes_in_use');found[role]=('JAX '+(f'{value/2**30:.6f}' if value is not None else 'N/A'),'N/A',str(base.get('system',{}).get('rss_gib','N/A')),str(qp.relative_to(ROOT)))
 table+='### 모델 적재 직후 — 첫 입력 forward 이전\n\n|적재 조건|allocated GiB(JAX는별도표기)|reserved GiB|프로세스 RSS GiB|원본 근거|\n|---|---|---|---|---|\n'
 for role,values in found.items():table+='|'+role+'|'+'|'.join(values)+'|\n'
 if not found:table+='|기준값 원본을 추가 확인 중|N/A|N/A|N/A|기존 상세 기록 참조;추정값으로 채우지 않음|\n'
 table+='\n위 값은 **가중치를 GPU에 적재하고 첫 입력을 계산하기 전의 프레임워크 할당량**이다. CUDA/JAX context·driver·시스템 전체 GPU사용량과 같지 않으며 CPU입력 준비 포함 여부는 해당 실행 코드를 따른다. **실행 peak와 모델 적재 기준값의 차이는 같은 run에서만 계산**한다. 파인튜닝 중 추가량에는 입력뿐 아니라 중간값·gradient·optimizer 상태가 포함된다. 서로 다른 조건/정밀도의 기준값을 하나의 모델 고유 메모리로 단정하지 않는다.\n\n'
 a=s.index('## 1. 모델과 실제 입력')+len('## 1. 모델과 실제 입력');b=a+re.search(r'^## 2\.',s[a:],re.M).start();s=s[:a]+'\n\n'+table+s[b:]
 # Keep historical detailed results/code intact; add an up-to-date conclusion at section4.
 marker='<!-- MODEL_SUMMARY_START -->';end='<!-- MODEL_SUMMARY_END -->';summary='### 요약 결론\n\n- **학습 방식:** '+ko(method)+'.\n- **입력 구분:** '+ko(limit)+'.\n'
 pointer=ROOT/model/'results/REAL_TRAINING_LATEST'
 if pointer.exists():
  d=pointer.parent/pointer.read_text().strip();records=[]
  for p in d.glob('*/summary.json'):
   v=json.loads(p.read_text());q=v.get('training') or v.get('condition') or next(iter(v.get('conditions',[])),None)
   if v.get('status')=='complete' and q and 'peak_allocated_gib' in q.get('memory',{}):records.append((q['memory']['peak_allocated_gib'],p.parent.name,q))
  if records:
   peak,case,q=max(records,key=lambda x:x[0]);ratio=peak/q['memory']['system_total_gib']*100;summary+=f'- **최신 추가 run 최대 성공 메모리:** `{case}` **{peak:.3f}GiB /{ratio:.2f}%**. 이 값은 해당 최신run의 성공 조건에서의 최대이며,다른 run의 최대와 구분한다. probe이면 전체 epoch·독립test 평가 완료를 뜻하지 않는다.\n'
 if model=='cellpose':summary+='- **전체 실험 구분:** 추론 타일batch384는43.397GiB/35.66%;FP32 파인튜닝 용량probe batch26은90.520GiB/74.39%. 전체10epoch·test평가·재로드는batch1에서 완료(AP50=0.9278). batch28은100GiB allocator 예산에서OOM;80% 성공은 미확인.\n'
 if model=='scgpt':summary+='- **실제1만세포 파인튜닝 완료:** batch284는98.003GiB/80.54%,1epoch442.37초,test5000세포Accuracy0.4674,재로드검증 완료. 이전고정padding batch292의82.79%는 별도 짧은probe이며 실제1만세포 조건과 다르다.\n'
 if model in ['colabfold','clair3']:summary+='- **학습 상태:** 이번 실행에서 훈련·미세조정은 수행하지 않았다. 참고 구조/진실변이는 평가 자료이며 학습 정답으로 사용한 것이 아니다.\n'
 summary+='- **시간·정확도 해석:** 결과는 같은 행의 입력·배치·학습범위에만 연결한다. 메모리 상승 자체가 처리량·정확도 상승을 보장하지 않는다.\n'
 content=marker+'\n'+summary+end+'\n\n'
 if marker in s:s=re.sub('^'+re.escape(marker)+r'\n[\s\S]*?^'+re.escape(end)+r'\n*',lambda _:content,s,flags=re.M)
 else:
  m=re.search(r'^## 4\.[^\n]*\n',s,re.M);s=s[:m.end()]+'\n'+content+s[m.end():]
 path.write_text(s);shutil.copy2(path,ROOT/filename)
print('Updated model/input tables and conclusions in all10Markdown reports')

# Refresh control-factor evidence in the same ten reports.
import subprocess,sys
subprocess.run([sys.executable,str(ROOT/'scripts/real_training_v3/factor_reports.py')],check=True)
```
<!-- END_RESTORE_FILE -->

</details>
