# AlphaFold2 — ColabFold 최종 통합 보고서

## 1. 모델과 실제 입력

|항목|이번 실험에서 사용한 모델|
|---|---|
|체크포인트·버전|ColabFold1.6.2,alphafold2_ptm/alphafold2_multimer_v3|
|구조|Evoformer+Structure Module|
|전체·구성별 파라미터 수|실제PTM tree92,907,930개(약9,290.8만 개)(92.91M)/Multimer93,223,705개(약9,322.4만 개)(93.22M);원본NPZ 구성은 상세표 참조|
|정밀도|기본 BF16;별도 실험vmap FP32|
|모델 입력 → 출력|단백질 서열+MSA →3D좌표,pLDDT/PAE|
|가중치와 실행 메모리의 차이|파라미터 수×dtype bytes는 이론 가중치 용량이다. 실행 peak에는 입력·중간 활성값이 더해지고,파인튜닝에는 gradient/optimizer 상태도 추가된다.|
|메모리% 정의|CUDA allocated peak / CPU·GPU 공유 RAM121.6875GiB ×100; GPU-Util과 별개. AlphaFold2는 JAX 계측을 별도로 표시|

### 실제 데이터와 학습 방식 구분

|구분|데이터·실제 처리 내용|
|---|---|
|추론 데이터|서로 다른 UniProt115AA 단백질4개,PDB1UBQ/1LYZ/1TIM;복합체3G5O A/B 예제|
|학습·검증·test 데이터|훈련·파인튜닝 미실시. 이번 작업에 사용한 학습 데이터셋 없음|
|학습 입력 X / 정답 y|추론 X:서열·MSA features. 참고 PDB는 평가용 정답 구조이며 학습에 사용하지 않음|
|가중치 갱신 방식|추론만 수행. 설치된 ColabFold 실행 경로에서 훈련하지 않음|
|원본·변환 입력 저장 위치|`data/queries.fasta;각run processed_features.npz/MSA`|
|해석 범위|pLDDT는 신뢰도이며 실제 정답 구조 대비 정확도와 다르다. vmap은 별도 실험적 추론 경로|

**용어:** 처음부터 학습=무작위 초기화에서 시작; 전체/일부 층 파인튜닝=공개 가중치에서 해당 부분 갱신; 분류기만 학습=본체를 동결하고 head만 갱신. **용량probe는 학습 방식이 아니라 짧은 자원 측정 목적**이며,실제로 수행한 가중치 갱신 방식과 함께 표기한다. 원본 전체 데이터 수와 해당 run에서 실제 사용한 부분집합 수를 구분한다.

### 실제 파일 → 전처리 → 모델 표현 → 최종 출력: 단계별 실제 예시

**AlphaFold2는단백질서열과MSA를받아구조를만든다. 평균임베딩뒤에회귀head를붙인ESM-2 과제와다르다.** 실제독립query `count1_A2RU14`의저장된입력·출력을확인했다.

#### 0–1. 원본 서열·MSA

```fasta
>count1_A2RU14
MAGTVLGVGAGVFILALLWVAVLLLCVLLSRASGAARFSVIFLFFGAVIITSVLLLFPRAGEFPAPEVEVKIVDDFFIGRYVLLAFLSAIFLGGLFLVLIHYVLEPIYAKPLHSY
```

서열115AA,feature sampling전원MSA깊이568이다. 이query에는정답구조를학습시킨것이아니며,이번작업은추론이다.

#### 2. 모델 입력 features

|실제NPZ key|실측shape·dtype|
|---|---|
|aatype|[4, 115] int32|
|residue_index|[4, 115] int32|
|msa_mask|[4, 512, 115] float32|
|extra_msa|[4, 55, 115] int32|
|extra_msa_mask|[4, 55, 115] float32|
|msa_feat|[4, 512, 115, 49] float32|
|target_feat|[4, 115, 22] float32|

recycle/ensemble feature축,MSA행,잔기축을원본서열개수나독립query batch와혼동하지않는다. zero-padding된MSA행은mask로제외한다.

#### 3–4. Evoformer 표현·구조 모듈

Evoformer가MSA와잔기쌍표현을처리하고structure module이원자좌표를계산한다. **원GPU실행은내부Evoformer embedding을저장하지않았으므로중간숫자는N/A**다. 이번확인은저장된features와최종예측배열을읽은것이며새JAX/GPU추론을실행한것이아니다.

#### 5. 최종 구조·신뢰도 출력

|출력|shape·dtype|실제예시|
|---|---|---|
|atom_positions|[115, 37, 3] float16|첫잔기CA좌표 [27.703125, 4.125000, -0.321289]|
|plddt|[115] float16|앞8값 [53.156250, 57.000000, 65.125000, 76.812500, 77.875000, 81.812500, 81.375000, 83.687500]|

좌표의마지막3개값은X/Y/Z좌표이며,37원자slot중유효원자는residue별mask를따른다.

#### 6. 정답·점수 해석

평균pLDDT는 83.875이며 **모델의신뢰도**다. 이query의정답실험구조가없어이를구조정확도로쓰지않는다. PDB1UBQ등정답구조대조는별도입력조건이다.

#### 7. 저장 파일

`results/real_input_capacity_20261102_173415_016493/batch_4/count1_A2RU14.pdb`의실제처음5개ATOM레코드:

```text
ATOM      1  N   MET A   1      28.281   4.988  -1.348  1.00 53.16           N  
ATOM      2  CA  MET A   1      27.703   4.125  -0.321  1.00 53.16           C  
ATOM      3  C   MET A   1      26.797   4.926   0.617  1.00 53.16           C  
ATOM      4  CB  MET A   1      28.797   3.426   0.479  1.00 53.16           C  
ATOM      5  O   MET A   1      26.734   4.633   1.812  1.00 53.16           O  
```

저장최종NPZ는atom_positions/plddt이며준비features와함께 `results/real_input_capacity_20261102_173415_016493/stage_trace_saved/arrays.npz`에사본을보관했다. 별도3D좌표/PDB출력이지class라벨이아니다.

### 모델 적재 직후 — 첫 입력 forward 이전

|적재 조건|allocated GiB(JAX는별도표기)|reserved GiB|프로세스 RSS GiB|원본 근거|
|---|---|---|---|---|
|추론/기본 모델 적재 기준|JAX 0.348193|N/A|1.5773162841796875|colabfold/results/real_input_capacity_20261102_173415_016493/batch_1/summary.json|

위 값은 **가중치를 GPU에 적재하고 첫 입력을 계산하기 전의 프레임워크 할당량**이다. CUDA/JAX context·driver·시스템 전체 GPU사용량과 같지 않으며 CPU입력 준비 포함 여부는 해당 실행 코드를 따른다. **실행 peak와 모델 적재 기준값의 차이는 같은 run에서만 계산**한다. 파인튜닝 중 추가량에는 입력뿐 아니라 중간값·gradient·optimizer 상태가 포함된다. 서로 다른 조건/정밀도의 기준값을 하나의 모델 고유 메모리로 단정하지 않는다.

## 2. 실행 조건과 측정 기준

<!-- ENV_MEASUREMENT_OVERVIEW_START -->
### 실행 환경·자원 측정 방법 요약

|항목|이번 보고서의 측정 기준|
|---|---|
|장비·공유메모리|DGX Spark,NVIDIA GB10,ARM64;시스템공유RAM121.6875GiB. CPU/GPU RAM을별도로더하지않음|
|환경기록출처|colabfold/results/real_input_capacity_20261102_173415_016493/batch_1/summary.json|
|Python·프레임워크|{'jax': '0.10.2'}|
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
|JAX메모리해석|XLA사전할당비활성. bytes_in_use와누적peak를분리하며미지원CUDA allocated/reserved는N/A|

<!-- ENV_MEASUREMENT_OVERVIEW_END -->

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

<!-- FACTOR_EFFECTS_START -->
### 시간·메모리에 영향을 주는 요인 — 통제 비교 상태

아래 원리는 모든 모델에서 정량 검증을 마쳤다는 뜻이 아니다. **실측 비교와 미실험을 구분**한다. 배치 sweep 중 정밀도·입력길이·체크포인트를 고정하며,다른 과제/다른head 결과를 같은 조건으로 섞지 않는다.

|요인|총시간·처리량 해석|최대 allocated 해석|이 모델의 실험 근거/상태|
|---|---|---|---|
|배치|한step은 길어질 수 있고step수는 줄어든다. 처리량 상승/포화는 실측으로 판단|같은입력길이·정밀도에서 동시중간값 증가가 주된영향|실제 독립query1/2/4 실험vmap,공식CLI와 구분|
|샘플당 실제입력 크기|GPU에들어간실제token/타일 크기가 커졌을때 계산량 변화 비교|토큰길이·패치/타일크기·attention구조에 따라 증가;원본크기와 다를수있음|단백질수·잔기길이·단량체/복합체 실험 있음;MSA깊이 차이는 별도|
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
|실험적JAX vmap 추론|서로다른실제단백질1개,각115AA+기존MSA|batch1,model_1,FP32|3recycle/초기포함4pass;2회|22.85초|0.0875단백질/초|2.162 GiB / 1.78%|JAX프로세스누적peak(컴파일/warmup포함);PDB검증;80%미도달|
|실험적JAX vmap 추론|서로다른실제단백질2개,각115AA+기존MSA|batch2,model_1,FP32|3recycle/초기포함4pass;2회|44.39초|0.0901단백질/초|2.158 GiB / 1.77%|JAX프로세스누적peak(컴파일/warmup포함);PDB검증;80%미도달|
|실험적JAX vmap 추론|서로다른실제단백질4개,각115AA+기존MSA|batch4,model_1,FP32|3recycle/초기포함4pass;2회|90.66초|0.0882단백질/초|2.308 GiB / 1.90%|JAX프로세스누적peak(컴파일/warmup포함);PDB검증;80%미도달|
|학습|이번환경미실시|—|—|미실시|—|미측정|전체학습파이프라인실행완료로표현하지않음|

<!-- REAL_TRAINING_V3_START -->
### 실제 데이터 추가 실험 — real_input_capacity_20261102_173415_016493

진행 중인 실행의 완료 조건만 아래에 갱신한다. probe는 짧은 용량 측정이며 전체 학습이나 평가 완료가 아니다. 메모리%는 CUDA allocated peak / 시스템 공유 RAM ×100이며 GPU-Util과 다르다. 독립 실행 1회 기준.

|조건|가중치 갱신 방식·실험 목적|상태|고유 입력|배치|epoch|optimizer steps|실측 시간(s)|입력/s|allocated peak GiB|reserved peak GiB|공유 RAM 대비 allocated %|GPU-Util 평균%|최대%|독립 test 지표|저장 모델 재로드|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|batch_1|추론(가중치 갱신 없음)|complete|1|1|N/A|N/A|22.8477|0.0875|N/A|N/A|N/A|93.3721|95.0000|N/A|N/A|
|batch_4|추론(가중치 갱신 없음)|complete|4|4|N/A|N/A|88.9949|0.0899|N/A|N/A|N/A|95.4847|96.0000|N/A|N/A|

이 run의 추론 capacity 실험과 지도학습 실험은 입력 과제·길이·학습 헤드가 다를 수 있어 각각 비교한다. 3step probe는 전체 학습이 아니다. full_training은 동일한 공개 초기 backbone와 seed42,FP32,AdamW(lr1e-5,weight_decay0.01),고정 전체 split pool padding으로2epoch 수행한다. 각 학습 조건의 원본 summary에 split ID,실제 길이·padding 모양,사용 헤드·학습 범위,라벨,gradient가 관측된 파라미터 수,검증/test지표,저장 모델 재로드 차이를 기록한다. scVI는 아래 설명이 적용되지 않는 별도 처음부터 학습 모델이며 AF2·Clair3에는 이 공통 학습 절차를 적용하지 않는다.

입력 및 메모리 기준값(같은 실행에서 측정):

|조건|실제 모델 입력 모양/정책|모델 적재 직후 allocated GiB|실행 시작 allocated GiB|peak 추가 allocated GiB|프로세스 RSS sampled peak GiB|시스템 가용 RAM sampled 최소 GiB|
|---|---|---|---|---|---|---|
|batch_1|상세 전처리 정책 참조|N/A|N/A|N/A|2.3146|107.4613|
|batch_4|상세 전처리 정책 참조|N/A|N/A|N/A|2.7746|108.0341|

실행 상태: `measured`; 측정범위·제약: `four_actual_cached_MSA_query_pool_limit; no native training in installed ColabFold execution path`.

JAX 계측 batch_1: 누적 process peak_bytes_in_use=2321053696bytes,실제 실행 구간 sampled peak_bytes_in_use=881670656bytes. 누적 peak에는 compile/warmup이 포함되며 CUDA allocated/reserved와 직접 동일시하지 않는다.

JAX 계측 batch_4: 누적 process peak_bytes_in_use=2484081152bytes,실제 실행 구간 sampled peak_bytes_in_use=2484081152bytes. 누적 peak에는 compile/warmup이 포함되며 CUDA allocated/reserved와 직접 동일시하지 않는다.

실행 코드 블록의 파일은 원문SHA256으로 검증하며, 명령JSON에 기록된 실행 경로로 해당 snapshot을 복사하면 복원할 수 있다. 대용량 원본 데이터·가중치는 아래 전처리/다운로드 코드와 출처를 사용하고 로컬 원본도 보존한다.

로컬 원시 결과: `/home/hyein/ai-benchmark/colabfold/results/real_input_capacity_20261102_173415_016493`. 메모리 부족/guard 실패는 성공 조건으로 기재하지 않는다.
<!-- REAL_TRAINING_V3_END -->

## 4. 결과 해석

<!-- MODEL_SUMMARY_START -->
### 요약 결론

- **학습 방식:** 추론만 수행. 설치된 ColabFold 실행 경로에서 훈련하지 않음.
- **입력 구분:** pLDDT는 신뢰도이며 실제 정답 구조 대비 정확도와 다르다. vmap은 별도 실험적 추론 경로.
- **학습 상태:** 이번 실행에서 훈련·미세조정은 수행하지 않았다. 참고 구조/진실변이는 평가 자료이며 학습 정답으로 사용한 것이 아니다.
- **시간·정확도 해석:** 결과는 같은 행의 입력·배치·학습범위에만 연결한다. 메모리 상승 자체가 처리량·정확도 상승을 보장하지 않는다.
<!-- MODEL_SUMMARY_END -->

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### AlphaFold2 — ColabFold를 통한 실행: DGX Spark 검증


작성 시각: 2026-10-27T18:23:34+09:00

#### 완료 범위와 주요 결과

기존 ColabFold 1.6.2 ARM64 이미지·가중치를 재사용해 단일체 7개 × 5개 모델 = 35개 구조와 복합체 1개 × Multimer v3 모델 5개 = 5개 구조를 실제 GB10에서 순차 예측했다. 모든 실행은 별도 호스트 run 폴더에 저장했으며 단일체와 복합체 GPU 작업을 겹치지 않았다. PDB·신뢰도 JSON·MSA·좌표의 유한값과 실제 서열을 검증했다. 구조 예측용 ColabFold 경로를 사용했으므로 공식 전체 AlphaFold2 데이터베이스 검색 파이프라인과 구분한다.

개수 비교는 동일 115 잔기 단백질 4개를 한 스트림으로 처리하고, 첫 1/2/4개 중첩 prefix를 집계했다. 조건별 독립 재실행이나 동일 입력 반복 측정이 아니다. 길이 비교는 76/129/247 잔기 실제 단백질을 사용한다. MSA는 각각 따로 준비하고 실제 깊이 차이를 공개한다. GPU 80%를 맞추기 위한 배치·정밀도 sweep은 하지 않았다.

#### 입력 전 모델 적재와 실행 메모리 비교

같은 이미지와 TF_FORCE_UNIFIED_MEMORY=1, XLA_PYTHON_CLIENT_MEM_FRACTION=4.0, preallocation=False 조건에서 모델만 로딩했다. 5개 weight set을 한꺼번에 보관하되 추론은 하나씩 한다. Loader 종료 시에는 초기 runner parameter 사본이 추가로 남을 수 있어, 첫 weight set을 선택하고 사용하지 않는 참조를 해제한 **입력 forward 직전 ready 상태**도 따로 측정했다. JAX bytes_in_use 기준이다.

| preset | loader 직후 allocated GiB | 첫 모델 선택 후 ready GiB | loading peak GiB | RSS 적재 후 GiB |
| --- | --- | --- | --- | --- |
| alphafold2_ptm | 2.178 | 1.817 | 2.186 | 3.556 |
| alphafold2_multimer_v3 | 2.178 | 1.801 | 2.218 | 3.379 |

| 실제 입력 | GPU 평균 / 최대 % | 모델 ready 기준 GiB | 실행 시작 후 누적 allocator peak GiB | 참고 차이 GiB |
| --- | --- | --- | --- | --- |
| 1UBQ (76 residues) | 91.85 / 94 | 1.817 | 2.166 | 0.349 |
| count1_A2RU14 (115 residues) | 92.88 / 95 | 1.817 | 2.273 | 0.456 |
| count2_A6NFY7 (115 residues) | 92.82 / 94 | 1.817 | 2.273 | 0.456 |
| count3_O14519 (115 residues) | 91.55 / 95 | 1.817 | 2.273 | 0.456 |
| count4_P01270 (115 residues) | 91.36 / 95 | 1.817 | 2.273 | 0.456 |
| 1LYZ (129 residues) | 93.33 / 95 | 1.817 | 2.307 | 0.490 |
| 1TIM (247 residues) | 93.67 / 95 | 1.817 | 2.706 | 0.889 |
| 3G5O_A_3G5O_B (180 residues) | 92.92 / 95 | 1.801 | 2.170 | 0.369 |

차이는 다른 run의 ready 상태를 기준으로 계산한 참고값이다. 실행 peak는 프로세스 시작 후 누적치이며 현재 query 자체의 activation peak만 뜻하지 않는다. 모델은 추론만 했으므로 자체 학습 데이터 예시가 없고, FASTA/MSA가 입력이며 정답 PDB는 평가용이다.

#### 환경과 실행 조건

| 항목 | 단일체 | 복합체 |
| --- | --- | --- |
| 이미지 | ghcr.io/sokrypton/colabfold:1.6.2-cuda13 | 동일 이미지 |
| digest | sha256:c9eab0253fe324b7ec2ec644607141539d7580a4e88d1d3788abba8c92d61ef1 | 동일 |
| GPU / 아키텍처 | NVIDIA GB10 / ARM64 | 동일 |
| Python / JAX / jaxlib | 3.12.12 / 0.10.2 / 0.10.2 | 동일 |
| NumPy / Haiku | 2.5.1 / 0.0.16 | 동일 |
| 모델 preset | alphafold2_ptm | alphafold2_multimer_v3 |
| 순차 weight set 수 | 5 (model_1~5) | 5 (model_1~5) |
| 실행 파라미터/tree (각 model) | 92,907,930 | 93,223,705 |
| 학습 가능 / 학습 수행 | 0 / 추론만 수행 | 0 / 추론만 수행 |
| recycle | 3 (최초 pass 포함 4회) | 20 (최초 pass 포함 21회) |
| 조기 중단 tolerance | 0.0 (비교를 위해 고정) | 0.0 (고정) |
| ensemble / seed | 1 / 42, seed 수 1 | 동일 |
| MSA 정책 | MMseqs2 UniRef + environmental | unpaired_paired, greedy |
| MSA 모델 capacity | 512 clustered + 5120 extra | 508 clustered + 2048 extra |
| 정밀도 | BF16 계산, FP32 weights, FP16 recycle buffers/출력 변환 | 동일 혼합 정밀도 정책 |
| template / dropout / relaxation | 모두 비활성화 | 모두 비활성화 |
| compile mode / Pallas | tuned / False | 동일 |
| 배치 | 단백질 1개씩, weight set 5개를 순차 실행 | 복합체 1개씩, weight set 5개를 순차 실행 |
| JAX preallocation | False | False |
| JAX backends | cuda,cpu: CPU feature 준비 + GPU forward | 동일 |

일반적인 ColabFold 기본 5개 모델·기본 BF16 경로를 사용했다. 단일체 recycle 3과 Multimer v3 recycle 20을 명시하고 tolerance를 0으로 고정했다. 모델별 기본 recycle이 다르므로 단일체와 복합체 시간을 같은 길이 효과로 직접 비교하지 않는다. BF16은 전체 배열이 BF16이라는 뜻이 아니다. GPU 메모리를 채우기 위한 정밀도 변경은 하지 않았다.

기존 모델·이미지·가중치·scGPT 컨테이너를 변경하지 않았다. Docker 그룹 등록은 사용자가 수행했고 현재 명령은 sg docker로 등록된 그룹을 적용했다. 가중치는 read-only `/cache`, 결과는 host bind `/work`에 저장했다. 예측 컨테이너는 network none이므로 MSA API 시간과 GPU 예측을 분리했다.

#### 입력·MSA·출처 확인

| 입력 | 원본/실제 residues | 원본 A3M 행 | 실제 feature residue 축 | MSA tensor shape | truncation |
| --- | --- | --- | --- | --- | --- |
| 1UBQ | 76 | 21577 | 86 | [4, 512, 86] | 없음 |
| count1_A2RU14 | 115 | 568 | 125 | [4, 512, 125] | 없음 |
| count2_A6NFY7 | 115 | 2382 | 125 | [4, 512, 125] | 없음 |
| count3_O14519 | 115 | 1295 | 125 | [4, 512, 125] | 없음 |
| count4_P01270 | 115 | 971 | 125 | [4, 512, 125] | 없음 |
| 1LYZ | 129 | 5508 | 139 | [4, 512, 139] | 없음 |
| 1TIM | 247 | 16469 | 247 | [4, 512, 247] | 없음 |
| 3G5O_A_3G5O_B | 180 | 21350 | 180 | [3072, 180] | 없음 |

개수용 UniProt 단백질은 ESM-2 단계에서 확보한 UniProtKB reviewed 2026_03 원본을 재사용했으며 서로 다른 고유 서열 4개를 검사했다. 모두 115 residues지만 MSA의 깊이·내용은 다르다. 길이용 FASTA와 실험 구조는 RCSB PDB 1UBQ/1LYZ/1TIM의 공개 데이터를 확보했다. 복합체는 ColabFold v1.6.2 공식 test-data/complex/input.csv의 3G5O A/B 예제(93+87 residues)이며, RCSB 전체 construct에 있는 tag 일부가 생략된 입력이다. 3G5O의 전체 biological assembly를 모두 예측한 것이 아니다.

A3M에서 소문자 insertion을 제외한 alignment column 수와 첫 query가 원본 입력과 일치하는지 확인했다. 표의 A3M 행은 검색 결과의 저장 행이고 feature 구성 과정의 중복 제거 후 깊이는 events.jsonl의 raw_feature_shapes에 따로 기록된다. 단일체 feature는 4개 recycle용 입력 축과 512 clustered/5120 extra capacity로 padding된다. 복합체는 원본 alignment와 모델 내부 sampled MSA capacity를 구분해야 한다. dummy template tensor가 존재해도 실제 template 정보는 사용하지 않는다. 실제 모델 입력의 shape·dtype는 events.jsonl에 보존했다.

#### 실제 입력·출력 예

```text
>1UBQ
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG

ATOM      2  CA  MET A   1      -5.727 -12.086   3.102  1.00 95.56           C  
```

저장된 1UBQ 1위 구조: `/home/hyein/ai-benchmark/colabfold/results/af2_20261027_172813_247681/predictions/1UBQ_unrelaxed_rank_001_alphafold2_ptm_model_4_seed_042.pdb`. 점수 JSON의 pLDDT 첫 5개 값은 [95.56, 97.81, 98.56, 98.56, 98.5], pLDDT shape=[76], PAE shape=[76,76]이다. PDB의 B-factor 필드는 예측 pLDDT를 기록하며 실험 B-factor가 아니다.

#### 시간·자원 계측 방법

기존 RunModel의 JIT apply를 의미가 같은 lower(...).compile()과 compiled callable 호출로 나누어 컴파일 시간을 측정했다. 새로운 shape에서만 AOT 컴파일하며 같은 shape·dtype·PyTree 구조의 다음 호출에는 실행 파일을 재사용했다. 매 GPU apply 결과에 jax.block_until_ready를 적용했다. GPU apply는 forward 1회이며 recycle과 model 개수를 합쳐 집계한다. CPU feature 구성, pLDDT/pTM·JSON/PDB 후처리, MSA API, 컴파일과는 범위가 다르다. 최초 shape의 첫 apply는 cold kernel 실행이며 포함한 상태로 보고한다. 별도 인위적 warmup을 수행하지 않았고 단일 프로세스 측정이다.

0.5초 간격으로 nvidia-smi GPU-Util, 프로세스 RSS, 시스템 MemAvailable·swap을 수집했다. 평균·최대는 조회 시작과 종료가 모두 GPU apply 구간 안에 있는 표본만 집계한다. 장치 전체 GPU-Util에는 디스플레이·브라우저 부하가 포함될 수 있으며 FLOPS나 SM occupancy 비율이 아니다. 각 apply 경계의 짧은 표본은 제외돼 표본 수가 단순 시간/0.5와 다를 수 있다.

JAX memory_stats의 peak_bytes_in_use는 프로세스 시작 후 누적 최고치이며 query별로 초기화하지 않았다. per-query 표는 이를 명확히 구분한다. 실제 apply 반환 시점의 live allocation도 별도 JSON에 기록했다. Compile memory_analysis는 정적 buffer 분석으로 실제 allocator peak와 다르다. TF_FORCE_UNIFIED_MEMORY=1, XLA_PYTHON_CLIENT_MEM_FRACTION=4.0은 기존 ColabFold 코드가 적용하는 설정이며 bytes_limit가 물리 RAM의 4배로 표시돼도 이를 전용 VRAM 용량이나 점유율 분모로 사용하지 않았다. Spark의 공유 DRAM에서 RSS와 CUDA/JAX 수치를 더하지 않았다. nvidia-smi 장치 memory 사용량은 Not Supported이므로 N/A다.

#### 개수 증가: 동일 길이 115 residues, 스트림 누적 prefix

| 고유/처리 입력 | 구조 수 | GPU apply 수 | GPU 실행 합계 s | protein/s | GPU 평균/최대 % | 표본 수 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / 1 | 5 | 20 | 49.700 | 0.02012 | 92.88 / 95 | 95 |
| 2 / 2 | 10 | 40 | 99.317 | 0.02014 | 92.85 / 95 | 195 |
| 4 / 4 | 20 | 80 | 194.667 | 0.02055 | 92.16 / 95 | 383 |

| 입력 수 | query wall 합계 s | 컴파일 s (wall에 포함) | JAX 누적 peak GiB | RSS sampled peak GiB | MemAvailable 최소 GiB |
| --- | --- | --- | --- | --- | --- |
| 1 | 58.289 | 7.710 | 2.273 | 5.420 | 109.224 |
| 2 | 108.738 | 7.710 | 2.273 | 5.576 | 109.224 |
| 4 | 205.893 | 7.710 | 2.273 | 5.604 | 109.132 |

모든 prefix는 같은 125 padded residue 축과 같은 MSA capacity로 처리됐다. 첫 115-residue query가 새 shape 컴파일 비용을 부담하고 뒤 query는 재사용한다. 따라서 query wall 차이와 GPU apply 시간을 구분한다. 고유 입력 1/2/4는 한 번 실행한 스트림의 중첩 집계이므로 독립 반복 실행의 평균이나 신뢰구간을 의미하지 않는다. 개수 증가에 따라 순간 GPU 메모리와 사용률이 비례해서 증가한다고 해석하지 않는다.

#### 단일체 길이 비교와 복합체 개별 결과

| 입력 | 길이 | GPU apply s | 컴파일 s | query wall s | GPU 평균/최대 % | 표본 수 |
| --- | --- | --- | --- | --- | --- | --- |
| 1UBQ | 76 | 33.637 | 8.366 | 43.253 | 91.85 / 94 | 65 |
| 1LYZ | 129 | 59.395 | 9.292 | 69.709 | 93.33 / 95 | 120 |
| 1TIM | 247 | 114.715 | 11.483 | 127.518 | 93.67 / 95 | 226 |
| 3G5O_A_3G5O_B | 180 | 334.304 | 19.504 | 357.369 | 92.92 / 95 | 660 |

| 입력 | JAX 누적 peak GiB | apply 반환 live 최대 GiB | RSS sampled peak GiB | MemAvailable 최소 GiB | swap bytes |
| --- | --- | --- | --- | --- | --- |
| 1UBQ | 2.166 | 1.794 | 4.968 | 109.909 | 0 |
| 1LYZ | 2.307 | 1.804 | 5.884 | 108.882 | 0 |
| 1TIM | 2.706 | 1.837 | 6.862 | 107.214 | 0 |
| 3G5O_A_3G5O_B | 2.170 | 1.818 | 5.119 | 109.463 | 0 |

단일체 길이가 증가하면서 실행 시간이 증가했다. 서로 다른 단백질과 MSA 깊이를 사용했으므로 완전한 길이 단독 인과 실험은 아니다. MSA와 template·모델·정밀도 정책은 고정했으며 실제 padding과 MSA shape를 공개했다. 복합체는 다른 Multimer v3 체크포인트와 20 recycles를 사용하므로 180 길이를 단일체 길이 곡선에 섞지 않았다. 단일체·복합체 모두 지원되는 작은 입력에서 이미 GPU 평균 80%를 넘었고 더 큰 입력으로 사용률을 억지로 맞추지 않았다.

![입력 규모별 시간과 GPU 사용률](https://raw.githubusercontent.com/fff0996/Bioinformatics_AImodel/49661373a7b790bbbbcce0d7ef8bf96b8ce83dbd/colabfold/reports/performance.png)

#### MSA·모델 로딩·컴파일·실행·저장 구분

| 실행 | MSA 준비 wall s | 모델 로딩 s | 컴파일 s | GPU apply s | pipeline wall s | 컨테이너 포함 wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 단일체 7개 | 57.020 | 2.396 | 36.852 | 402.414 | 454.267 | 456.572 |
| 복합체 1개 | 20.187 | 3.789 | 19.504 | 334.304 | 362.510 | 364.547 |

MSA wall에는 컨테이너 시작·API 대기/다운로드·MSA 저장·일부 전처리/coverage 그림이 포함돼 순수 검색 서버 계산 시간이 아니다. GPU 예측에서는 저장된 A3M을 재사용했고 네트워크 접근은 차단했다. pipeline wall은 로딩·전처리·컴파일·GPU·후처리·모든 파일 저장을 포함한다. query wall은 predict_structure 단위로 그 앞의 raw MSA 파싱·일부 feature 생성과 global 모델 로딩을 제외한다. 파일 쓰기 이벤트는 PDB/JSON/A3M text write 시간의 하한이며 PNG·ranking rename·모든 serialization을 완전 분리한 저장 시간은 N/A다. Relaxation은 0회로 실행하지 않았다. 서로 다른 계측 범위를 직접 속도 개선으로 비교하지 않는다.

#### 구조 검증과 정확도 해석

| 1위 구조 | pLDDT 평균 | pTM | CA RMSD Å | CA-only lDDT | 정렬 CA 수 |
| --- | --- | --- | --- | --- | --- |
| 1UBQ | 95.96 | 0.840 | 0.801 | 0.982 | 76 |
| count1_A2RU14 | 86.08 | 0.520 | N/A | N/A | N/A |
| count2_A6NFY7 | 74.01 | 0.530 | N/A | N/A | N/A |
| count3_O14519 | 71.37 | 0.450 | N/A | N/A | N/A |
| count4_P01270 | 70.26 | 0.310 | N/A | N/A | N/A |
| 1LYZ | 98.20 | 0.910 | 0.546 | 0.949 | 129 |
| 1TIM | 96.49 | 0.940 | 1.044 | 0.885 | 247 |

| 복합체 1위 항목 | 결과 |
| --- | --- |
| pLDDT 평균 | 94.59 |
| pTM / ipTM | 0.890 / 0.880 |
| 사슬 A CA RMSD Å / aligned CA | 1.286 / 91 |
| 사슬 B CA RMSD Å / aligned CA | 1.460 / 81 |
| 전체 정렬 CA RMSD Å | 1.598 |
| 전체 정렬 CA-only lDDT | 0.969 |

모든 구조에서 CA 서열이 입력과 일치하고 좌표·pLDDT·PAE가 유한하며, pLDDT [L], PAE [L,L]을 검사했다. 정답 구조 대비 수치는 global sequence alignment로 대응 CA를 선정한 후 Kabsch RMSD와 CA-only lDDT(15Å 이내 쌍, 0.5/1/2/4Å 기준)를 계산했다. CA-only lDDT를 공식 all-atom lDDT나 자체 예측 pLDDT와 동일시하지 않는다. pLDDT/pTM/ipTM은 신뢰도이며 정답 구조 대비 정확도와 별개다.

3G5O reference는 PDB A/B 사슬이며 query에 생략된 tag와 실험 구조의 missing residue가 있어 alignment 범위를 명시했다. 전체 complex 정렬은 고정된 A/B 순서의 concatenated sequence 대응을 사용한 sanity check이며 DockQ나 CAPRI 평가를 수행하지 않았다. 이 단백질들은 오래된 공개 PDB 구조로 학습 데이터와의 독립성을 입증하지 못한다. 따라서 정상 작동 확인용 구조 일치 결과이며 신규 단백질 일반화 성능을 입증하는 독립 benchmark가 아니다.

원본 NPZ 파라미터 수: PTM model_1/2는 각각 93,237,338개, model_3/4/5는 각각 92,907,930개, Multimer v3는 각각 93,223,705개다. 실제 추론에 전달한 tree 수와 원본 전체를 구분한다. 표의 학습 가능 0은 이 실행에서 optimizer가 갱신한 대상이 없다는 뜻이며 모든 weight가 수학적으로 미분 불가능하다는 뜻이 아니다. MSA 서버의 정확한 데이터베이스 revision은 확인하지 못해 N/A로 남긴다.

#### 학습·저장·재현 상태

현재 ColabFold CLI와 호출한 RunModel 경로는 구조 추론용이다. optimizer·훈련 loop·지도학습 split을 구성하지 않았고 파인튜닝을 수행하지 않았다. is_training 인자가 존재하는 것만으로 이 실행에서 지원되는 완성형 학습 workflow가 있다고 표현하지 않는다. 새 학습 checkpoint·optimizer·재개 학습·학습 모델 재로드는 N/A다. 기존의 공식 가중치를 재사용했고 출력 PDB·JSON을 새 CPU 프로세스에서 다시 읽어 저장 결과를 검증했다.

최초 단일체 시도는 JAX_PLATFORMS=cuda만 지정해 CPU 전처리 backend가 비활성화돼 실패했다. ColabFold의 일반적인 "Not Enough GPU memory?" 로그와 달리 실제 예외는 "Unknown backend cpu"였고 OOM이 아니었다. 실패 run과 로그를 보존한 뒤 cuda,cpu로 수정해 새 run에서 성공했다. GPU 실행은 dev=cuda:0와 실제 JAX GPU apply로 확인했다.

#### 저장 경로와 실행 방법

| 항목 | 호스트 경로 |
| --- | --- |
| 단일체 run | /home/hyein/ai-benchmark/colabfold/results/af2_20261027_172813_247681 |
| 복합체 run | /home/hyein/ai-benchmark/colabfold/results/af2_multimer_20261027_181640_431467 |
| 단일체 MSA | /home/hyein/ai-benchmark/colabfold/results/msa_20261027_172359_775627 |
| 복합체 MSA | /home/hyein/ai-benchmark/colabfold/results/msa_complex_20261027_173649_931405 |
| 환경·원본 source | /home/hyein/ai-benchmark/colabfold/results/environment_verified |
| 입력·정답 PDB | /home/hyein/ai-benchmark/colabfold/data |
| 기존 가중치 | /home/hyein/ai-benchmark/colabfold/cache/colabfold/params |
| 계측 코드 | /home/hyein/ai-benchmark/colabfold/scripts/benchmark_af2.py |
| CPU 구조 검증 | /home/hyein/ai-benchmark/colabfold/scripts/validate_structures.py |
| 비교 CSV / metrics JSON | /home/hyein/ai-benchmark/colabfold/reports/comparison.csv / metrics.json |
| weight SHA-256 / NPZ 파라미터 | /home/hyein/ai-benchmark/colabfold/reports/weight_manifest.json |
| 보고서 | /home/hyein/ai-benchmark/colabfold/reports/AlphaFold2_ColabFold.md |

각 run에는 command.json, console.log, log.txt, events.jsonl, resources.csv, exit_code.json, summary.json, input_report.json, apply_calls.json, query_stats.json, predictions/와 structure_validation.json이 있다. 모든 컨테이너 출력은 호스트 bind mount 아래에 보존된다. run별 코드 snapshot도 보관했다. 10개 실제 사용 가중치 NPZ의 해시·dtype·원본 파라미터 수는 weight_manifest.json에 저장했다. 템플릿 비활성화 시 model3 구조를 공유하는 loaded tree는 원본 NPZ 전체와 파라미터 수가 다를 수 있다.

```bash
cd /home/hyein/ai-benchmark
# 기존 출력은 보존하고 새 run ID를 생성한다. GPU compute 작업이 있으면 launch를 중단한다.
python3 colabfold/scripts/launch.py
python3 colabfold/scripts/launch_multimer.py
```

#### 한계·다음 작업

메모리 압박·swap 급증·OOM은 관찰되지 않았다. GPU 사용률과 메모리 용량을 혼동하지 않았으며 순간 메모리 또는 물리 RAM을 80% 채우는 것을 목표로 하지 않았다. 단일 seed·단일 프로세스의 작은 입력 범위이며 confidence calibration, 대규모 생산 처리량, 여러 MSA seed, docking accuracy, relaxation과 학습 성능은 미검증이다. 다음 모델 순서는 scVI이고, 기존 scGPT 정규화 이력이 불명확한 입력을 raw count로 그대로 재사용하면 안 된다.

#### 공식 출처

- [ColabFold 저장소·사용법](https://github.com/sokrypton/ColabFold)
- [설치 버전의 모델 로딩·BF16 설정](https://github.com/sokrypton/ColabFold/blob/v1.6.2/colabfold/alphafold/models.py)
- [설치 버전의 batch CLI·MSA·padding](https://github.com/sokrypton/ColabFold/blob/v1.6.2/colabfold/batch.py)
- [공식 복합체 예제](https://github.com/sokrypton/ColabFold/blob/v1.6.2/test-data/complex/input.csv)
- [AlphaFold2 공식 저장소](https://github.com/google-deepmind/alphafold)
- [RCSB PDB 1UBQ](https://www.rcsb.org/structure/1UBQ)
- [RCSB PDB 1LYZ](https://www.rcsb.org/structure/1LYZ)
- [RCSB PDB 1TIM](https://www.rcsb.org/structure/1TIM)
- [RCSB PDB 3G5O](https://www.rcsb.org/structure/3G5O)
- [JAX AOT 컴파일](https://docs.jax.dev/en/latest/aot.html)
- [Spark 공유 메모리 계측 한계](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### AlphaFold2 — ColabFold JAX vmap 배치 후속 실험


상태 **complete** · vmap_fp32_20261028_190843_576148

공식CLI가제공하는배치기능이아닌**별도실험구현**이다. 기존ColabFold1.6.2/AlphaFold2 PTM model_1의가중치·forward를사용하고 JAX vmap으로서로독립인실제단백질query축을추가했다. 복합체사슬을batch로세거나다른GPU작업을동시에띄우지않았다. 이번비교는FP32활성값/가중치를고정했다. 앞선BF16벡터화는native대조CA RMSD1.075Å로사전0.5Å검증기준을넘어실패기록을보존했고,수치민감도를확인하기위해별도FP32조건을분리했다. 기본BF16성능수치와직접비교하지않는다. 모델수는고정1개,3recycle(초기pass포함4회),ensemble1,seed42/query,dropout비활성,template/relaxation없음이다. 기본보고서의5모델순차실행과다른조건으로구분한다.

기존실제단량체4개는각115잔기이며기존MSA를재사용한다. 원래MSA깊이·SHA256은input_audit.json,실제처리msa mask깊이와각feature shape은worker summary에있다. extra-MSA/cluster축은pool내최대shape로zero pad하고마스크도0으로유지했다. 잔기서열을복제하거나추가하지않았다. 큰모델에서vmap이동작하는지부터단일native predict와검증하는pilot이며길이900AA같은다른조건의결과로확대해해석하지않는다.

|batch/고유query|반복|초|query/초|GPU평균/최대%|프로세스누적JAX peak GiB|실행중sampled bytes-in-use peak GiB|컴파일초|
|---|---|---|---|---|---|---|---|
|1|2|22.8477|0.0875|93.3721|95.0|2.1616|0.8211|7.9508|
|2|2|44.3891|0.0901|93.6951|96.0|2.1575|1.3424|10.2016|
|4|2|90.6622|0.0882|94.4759|96.0|2.3083|2.3083|12.5127|

시간은동기화된4passforward·recycle의FP16반환·prev재입력을포함한다. MSA/featurecache구성·모델로딩·컴파일·warmup·PDB저장·native참조검증은별도시간이다. JAXpeak는프로세스생성후컴파일/warmup포함누적allocator값이며실행구간reset peak라고쓰지않는다. compiled.memory_analysis도별도기록한다. 실행중bytes_in_use/RSS/시스템가용량은0.5초대기+조회sampled값이다. CPU/GPU공유RAM에서RSS와JAX값을더하지않는다.

단일batch1과원래runner.predict의동일query/seed출력은CA정렬RMSD·pLDDT차이로비교했다. 다른batch도첫query를batch1과비교해벡터화의수치차이를보존했다. 모든좌표/pLDDTfinite·잔기수·PDB의CA수검증을수행했다. 신뢰도와정답구조정확도를혼동하지않으며새독립구조정확도평가는하지않았다. 실제4querypool끝에서종료하므로80/90%메모리최대가능성을검증한실험은아니다.

[공식 ColabFold](https://github.com/sokrypton/ColabFold) · [JAX vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) · [기본 결과](AlphaFold2.md)

원본실행: results/vmap_fp32_20261028_190843_576148; 명령·소스·가중치출처·inputcache·출력·자원시계열·exit·검증값을보존한다.

### 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.


## 6. 실제 실행 코드·명령어 전문

<!-- EXECUTED_SCRIPTS_START -->

### 원문과 출처 검증

아래 코드 블록은 원본 내용을 생략하거나 재작성하지 않았다. 각 파일의 원래 경로·원문 SHA256·출처 상태를 함께 기록했다. `실행 run 저장 snapshot`을 우선하고, 현재 소스만 남은 경우에는 당시 파일과 같다고 단정하지 않는다. 동일 바이트 버전은 한 번만 수록하고 해당 경로들을 함께 표시한다.

GitHub에는 `.py`/`.sh`를 별도 파일로 두지 않는다. 코드 블록을 원래 경로에 저장하거나 [복원 안내](AlphaFold2.md)의 코드로 로컬에서 복원한다. 라이브러리 전체 소스는 이 사용자 실행 스크립트와 구분하며 패키지 버전/출처를 따른다.

### 코드 목록

|원래 파일|원문 SHA256|출처 상태|
|---|---|---|
|`colabfold/scripts/batch_followup/capture.py`|`d9f662d352182b1ceaafeb7bc3aed41b58c83d54d0eb982b4a27d1a7e971c3ff`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/results/vmap_batch_20261028_185758_495006/source_capture.py`|`d9f662d352182b1ceaafeb7bc3aed41b58c83d54d0eb982b4a27d1a7e971c3ff`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`colabfold/scripts/batch_followup/worker.py`|`3d83bfc23d5a62f73522ebc6ddd26ad91c3f0a0a666f7a76b97eb4da74da008a`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/results/vmap_fp32_20261028_190843_576148/source_worker.py`|`3d83bfc23d5a62f73522ebc6ddd26ad91c3f0a0a666f7a76b97eb4da74da008a`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`colabfold/scripts/benchmark_af2.py`|`f11c9e66c3b8f3514780db7513ff135ca9f8f9fdc706dc0884f8b06b648cdee3`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/results/af2_multimer_20261027_181640_431467/benchmark_af2.py`|`f11c9e66c3b8f3514780db7513ff135ca9f8f9fdc706dc0884f8b06b648cdee3`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`colabfold/scripts/inspect_environment.py`|`2fcfbabaa25bad65104e67a2c41bebaaa1ba77dd4b4696ff956fe36acf894c3d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/scripts/launch.py`|`c5a623b4a8a1e0efcf840b18ac32c52d82447095e594d328fcd34c2227a05f53`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/scripts/launch_multimer.py`|`ef4104935d86cc6e0f25ee6e08f4af0e1ce26f8494b77ef71650e950e8d6d73f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/scripts/measure_loading.py`|`39667da7763f11bbaacf0a1b1e1264bb28b641c51600b6ba4af63cf5355c1b50`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/scripts/prepare_inputs.py`|`2a327312228528a70d4ba6e8003cee0096ce9059a1b0d67c510b05d7561b1e7f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/scripts/report_af2.py`|`34c4855a7ffc5dd9b66aa88a94f1f69230d9a5d5a05917028eb48d667d6e3cb3`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/scripts/validate_structures.py`|`8137f03a6ec35c630f5843b9fb7c61dfe97715857676cc0b562b3b6a07a45500`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`colabfold/results/vmap_batch_20261028_185758_495006/source_worker.py`|`c999c2ae00fa1fb9f9c3bdedbbc575a47b5f34b7c541e91e293e1d70bc7d5c64`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`colabfold/results/environment_verified/colabfold_models.py`|`bd637765535c7a199d76631874264e20e2d496e332251fb1d16e144542c03894`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`colabfold/results/environment_verified/alphafold_model.py`|`61e0d1ebc73e644defb9f649cfbdedab21b839829b41cf9f8102d2f53f3092df`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`colabfold/results/environment_verified/batch.py`|`1001e5685ce9b104cefbe0391cf803815d0788f90d47387883b6991338bebcd7`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`colabfold/results/af2_20261027_172813_247681/benchmark_af2.py`|`626767be4e7429cd4bb57737e77bba58877133c44dcbe9760e8c8845ff0bf323`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`colabfold/results/af2_20261027_172609_696571/benchmark_af2.py`|`a89e96d7103c3b3784dff3dfc1b012b42a25c73b6616771eaba2af510382aa04`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`colabfold/results/vmap_batch_retry_20261028_190537_289704/source_worker.py`|`1a123408b7acb82b46c8643e3ab9070e5c539b3064ec06f6450eaa86919eae3d`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|

<details>
<summary>colabfold/scripts/batch_followup/capture.py — 원문 전문 (2,352 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/batch_followup/capture.py", "sha256": "d9f662d352182b1ceaafeb7bc3aed41b58c83d54d0eb982b4a27d1a7e971c3ff", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,numpy as np,hashlib
from colabfold import batch
from colabfold.utils import setup_logging
from alphafold.model import model as afmodel
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];MSA=R/'results'/(R/'results/MSA_LATEST').read_text().strip()/'msa';manifest=json.loads((R/'data/input_manifest.json').read_text());queries=[r for r in manifest['queries'] if r['length']==115];assert len(queries)==4;setup_logging(D/'feature_capture.log');current={};audit=[]
class Captured(Exception):pass
def capture(self,feat,*args,**kwargs):
 p=D/(current['id']+'_processed_features.npz');np.savez_compressed(p,**{k:np.asarray(v) for k,v in feat.items()});(D/'runner_config.json').write_text(json.dumps(self.config.to_dict(),indent=2));print('CAPTURED',current['id'],{k:list(v.shape) for k,v in feat.items() if k in ['aatype','msa_feat','extra_msa']},flush=True);raise Captured('intentional feature cache; GPU prediction deferred to controlled vmap worker')
afmodel.RunModel.predict=capture
for q in queries:
 current=q;text=(MSA/(q['id']+'.a3m')).read_text();depth=sum(bool(l.strip()) and not l.startswith(('>','#')) for l in text.splitlines());audit.append({'id':q['id'],'sequence':q['sequence'],'length':q['length'],'msa_depth_before_feature_sampling':depth,'msa_sha256':hashlib.sha256(text.encode()).hexdigest()})
 try:batch.run(queries=[(q['id'],q['sequence'],[text],None)],result_dir=D/('capture_'+q['id']),num_models=1,model_order=[1],is_complex=False,num_recycles=3,recycle_early_stop_tolerance=0.,num_ensemble=1,model_type='alphafold2_ptm',msa_mode='mmseqs2_uniref_env',use_templates=False,num_relax=0,data_dir=Path('/cache'),random_seed=42,num_seeds=1,use_bfloat16=True,use_pallas=False,compile_mode='tuned',max_seq=512,max_extra_seq=5120,recompile_padding=10)
 except Captured:pass
 assert (D/(q['id']+'_processed_features.npz')).exists(),q['id']
(D/'input_audit.json').write_text(json.dumps({'queries':audit,'unique_queries':len(audit),'features':'official ColabFold processed features captured before predict, JAX_PLATFORMS=cpu during capture','preset':'alphafold2_ptm model_1, actual runner shared model3 architecture without templates','num_recycles':3,'num_ensemble':1,'max_seq':512,'max_extra_seq':5120,'templates':False,'relaxation':False,'seed':42},indent=2));print('FEATURE_CAPTURE_COMPLETE',flush=True)
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "colabfold/results/vmap_batch_20261028_185758_495006/source_capture.py", "source": "colabfold/scripts/batch_followup/capture.py", "sha256": "d9f662d352182b1ceaafeb7bc3aed41b58c83d54d0eb982b4a27d1a7e971c3ff"} -->

</details>

<details>
<summary>colabfold/scripts/batch_followup/worker.py — 원문 전문 (8,561 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/batch_followup/worker.py", "sha256": "3d83bfc23d5a62f73522ebc6ddd26ad91c3f0a0a666f7a76b97eb4da74da008a", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,time,json,threading,csv,subprocess,gc
import numpy as np,jax,jax.numpy as jnp
from colabfold.alphafold.models import load_models_and_params
from alphafold.common import protein
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];B=int(os.environ['BATCH']);C=D/f'batch_{B}';C.mkdir();dev=jax.devices('gpu')[0];stop=threading.Event();rows=[];phase='loading';G=2**30
S={'status':'running','batch':B,'implementation':'experimental jax.vmap over unmodified ColabFold RunModel.apply; shared fixed model_1 weights, independent actual query features and RNG; not a native CLI batch option','conditions':[]}
def write():(C/'summary.json').write_text(json.dumps(S,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def mem():
 info={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};rss=next(int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'));return {'rss_gib':rss/G,'available_gib':info['MemAvailable']/G,'total_gib':info['MemTotal']/G,'swap_gib':(info['SwapTotal']-info['SwapFree'])/G}
def dmem():return {k:int(v) for k,v in (dev.memory_stats() or {}).items()}
swap0=mem()['swap_gib']
def monitor():
 with (C/'resources.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['query_start','query_end','phase','gpu_util','rss_gib','available_gib','swap_gib','jax_bytes_in_use']);w.writeheader()
  while not stop.is_set():
   t=time.time();ph=phase
   try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip())
   except Exception:u=None
   m=mem();r={'query_start':t,'query_end':time.time(),'phase':ph,'gpu_util':u,'rss_gib':m['rss_gib'],'available_gib':m['available_gib'],'swap_gib':m['swap_gib'],'jax_bytes_in_use':dmem().get('bytes_in_use')};rows.append(r);w.writerow(r);f.flush()
   if m['available_gib']<16 or m['swap_gib']>swap0+2:(C/'memory_guard.json').write_text(json.dumps(r));os._exit(86)
   stop.wait(.5)
th=threading.Thread(target=monitor,daemon=True);th.start()
def aligned_rmsd(x,y):
 x=x-x.mean(0);y=y-y.mean(0);u,s,v=np.linalg.svd(x.T@y);q=np.eye(3);q[-1,-1]=np.linalg.det(u@v);return float(np.sqrt(np.mean(np.sum((x@(u@q@v)-y)**2,axis=1))))
try:
 audit=json.loads((D/'input_audit.json').read_text());queries=audit['queries'];allfeat=[dict(np.load(D/(q['id']+'_processed_features.npz'))) for q in queries];keys=set(allfeat[0]);assert all(set(f)==keys for f in allfeat);maxshape={k:tuple(max(f[k].shape[d] for f in allfeat) for d in range(allfeat[0][k].ndim)) for k in keys};features=[]
 for f in allfeat:
  features.append({k:np.pad(v,[(0,maxshape[k][d]-v.shape[d]) for d in range(v.ndim)],mode='constant') for k,v in f.items()})
 selected=features[:B];S['input']={'unique_queries':B,'query_ids':[q['id'] for q in queries[:B]],'actual_residues':[q['length'] for q in queries[:B]],'feature_shapes_per_query':{k:list(v.shape) for k,v in features[0].items()},'padding':'pool-wide max feature shapes; zero-padded extra-MSA/cluster entries masked by zero masks; no sequence duplication','actual_msa_valid_rows':[int((f['msa_mask'][0].sum(-1)>0).sum()) for f in selected]};t=time.time();loaded=load_models_and_params(num_models=1,use_templates=False,num_recycles=3,recycle_early_stop_tolerance=0.,num_ensemble=1,model_order=[1],model_type='alphafold2_ptm',data_dir=Path('/cache'),max_seq=512,max_extra_seq=5120,use_bfloat16=False,use_pallas=False,compile_mode='tuned');name,runner,params=loaded[0];runner.params=params;jax.block_until_ready(params);S['loading_seconds']=time.time()-t;S['parameters']=sum(int(x.size) for x in jax.tree_util.tree_leaves(params));S['weights_dtype']=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(params)});S['model_only_postload']={'jax':dmem(),'system':mem()};S['environment']={'jax':jax.__version__,'device':str(dev),'preallocate':os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),'compute':'FP32 activations/weights, explicit separate validation condition after BF16 vmap reference drift','recycles':3,'ensembles':1,'templates':False,'relaxation':False,'seed_per_query':42};write()
 full={k:jnp.asarray(np.stack([f[k] for f in selected])) for k in keys};L=int(selected[0]['aatype'].shape[1]);zero_prev={'prev_msa_first_row':jnp.zeros((B,L,256),jnp.float16),'prev_pair':jnp.zeros((B,L,L,128),jnp.float16),'prev_pos':jnp.zeros((B,L,37,3),jnp.float16)};rng=jnp.stack([jax.random.PRNGKey(42)]*B);first={k:v[:,0:1] for k,v in full.items()};first['prev']=zero_prev;vmapped=jax.jit(jax.vmap(runner.apply.__wrapped__,in_axes=(None,0,0)),compiler_options={'xla_gpu_override_gemm_autotuner':'block_m: 64 block_n: 64 block_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1'});phase='compilation';t=time.time();compiled=vmapped.lower(params,rng,first).compile();S['compilation_seconds']=time.time()-t;S['compiled_memory_analysis']=str(compiled.memory_analysis());write()
 def predict():
  prev=zero_prev;key=jnp.stack([jax.random.PRNGKey(42)]*B)
  for recycle in range(4):
   pair=jax.vmap(jax.random.split)(key);key=pair[:,0];subkey=pair[:,1];subfeat={k:v[:,recycle:recycle+1] for k,v in full.items()};subfeat['prev']=prev;result=compiled(params,subkey,subfeat);jax.block_until_ready(result);result=jax.tree_util.tree_map(lambda x:np.asarray(x,np.float16),result);prev={k:jnp.asarray(v) for k,v in result.pop('prev').items()}
  return result
 phase='warmup';t=time.time();result=predict();S['warmup_seconds']=time.time()-t;phase='execution';t=time.time();repeats=0
 while repeats<2 or time.time()-t<15:result=predict();repeats+=1
 end=time.time();stats=dmem();observed=[r for r in rows if r['query_start']>=t and r['query_end']<=end];utils=[r['gpu_util'] for r in observed if r['gpu_util'] is not None];S['conditions']=[{'batch':B,'unique_queries':B,'repeats':repeats,'processed_queries':B*repeats,'seconds':end-t,'queries_per_second':B*repeats/(end-t),'gpu_util_avg':float(np.mean(utils)) if utils else None,'gpu_util_max':max(utils) if utils else None,'sample_count':len(utils),'jax_memory_stats':stats,'memory_definition':'peak_bytes_in_use is cumulative process allocator peak including compilation and warmup; execution bytes_in_use samples separate, no unsupported reset','execution_sampled_peak_bytes_in_use':max([r['jax_bytes_in_use'] for r in observed if r['jax_bytes_in_use'] is not None],default=None),'rss_sampled_peak_gib':max([r['rss_gib'] for r in observed],default=None),'available_sampled_min_gib':min([r['available_gib'] for r in observed],default=None)}];phase='validation';preds=[]
 for i,q in enumerate(queries[:B]):
  single=jax.tree_util.tree_map(lambda x:x[i],result);coords=single['structure_module']['final_atom_positions'];plddt=single['plddt'];assert coords.shape==(L,37,3) and np.isfinite(coords).all() and np.isfinite(plddt).all();np.savez(C/(q['id']+'_prediction.npz'),atom_positions=coords,plddt=plddt);prot=protein.from_prediction(features=selected[i],result=single,b_factors=np.repeat(plddt[:,None],37,axis=1));pdb=protein.to_pdb(prot);(C/(q['id']+'.pdb')).write_text(pdb);assert sum(l.startswith('ATOM') and l[12:16].strip()=='CA' for l in pdb.splitlines())==q['length'];preds.append({'id':q['id'],'residues':q['length'],'mean_plddt':float(plddt.mean()),'finite_and_pdb_valid':True})
 S['predictions']=preds;reference=D/'batch_1'/ (queries[0]['id']+'_prediction.npz')
 if B==1:
  phase='single_native_reference';t0=time.time();ref,recycles=runner.predict(selected[0],random_seed=42);S['native_reference_seconds_excluded']=time.time()-t0;refcoords=ref['structure_module']['final_atom_positions'];S['native_reference_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(refcoords[:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max()),'native_recycles':int(recycles)};assert S['native_reference_comparison']['ca_aligned_rmsd_angstrom']<.5;np.savez(C/'native_single_reference.npz',atom_positions=refcoords,plddt=ref['plddt'])
 else:
  ref=np.load(reference);S['first_query_batch_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(ref['atom_positions'][:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max())}
 S['status']='complete';write();print('AF2_VMAP_COMPLETE',B,flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write();raise
finally:stop.set();th.join(timeout=5)
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/source_worker.py", "source": "colabfold/scripts/batch_followup/worker.py", "sha256": "3d83bfc23d5a62f73522ebc6ddd26ad91c3f0a0a666f7a76b97eb4da74da008a"} -->

</details>

<details>
<summary>colabfold/scripts/benchmark_af2.py — 원문 전문 (11,120 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/benchmark_af2.py", "sha256": "f11c9e66c3b8f3514780db7513ff135ca9f8f9fdc706dc0884f8b06b648cdee3", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Instrument existing ColabFold; serialize all GPU work; explicit AOT compile timing."""
import os,sys,pathlib,time,json,threading,subprocess,csv,traceback,inspect,functools,datetime,hashlib,shutil,resource
import numpy as np,jax
from colabfold import batch
from colabfold.alphafold import models
from alphafold.model import model as afmodel
from colabfold.input import get_queries
from colabfold.utils import setup_logging
COMPLEX=len(sys.argv)>3 and sys.argv[3]=='multimer'
MANIFEST='complex_manifest.json' if COMPLEX else 'input_manifest.json'
MODELTYPE='alphafold2_multimer_v3' if COMPLEX else 'alphafold2_ptm'
RECYCLES=20 if COMPLEX else 3
MAXSEQ=508 if COMPLEX else 512
MAXEXTRA=2048 if COMPLEX else 5120
R=pathlib.Path('/work');O=R/'results'/sys.argv[1];O.mkdir(exist_ok=True);MSA=R/'results'/sys.argv[2]/'msa';phase='setup';current='setup';stop=threading.Event();samples=[];predictions=[];calls=[];events=[];query_stats=[];allstart=time.perf_counter()
shutil.copy2(__file__,O/'benchmark_af2.py')
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def event(kind,**kw):
 row=dict(time_unix=time.time(),kind=kind,query=current,phase=phase,**kw);events.append(row)
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
 print('BENCH',json.dumps(row,default=str),flush=True)
def memory():
 d={}
 for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
  k,v=line.split(':');d[k]=int(v.strip().split()[0])*1024
 rss=next(int(l.split()[1])*1024 for l in pathlib.Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'))
 return rss,d['MemAvailable'],d['SwapTotal']-d['SwapFree'],d['MemTotal']
initial_swap=memory()[2];dev=jax.devices('gpu')[0]
def dmem():
 try:return {k:int(v) for k,v in (dev.memory_stats() or {}).items()}
 except Exception:return {}
def monitor():
 with (O/'resources.csv').open('w') as f:
  fields=['query_start','query_end','query','phase','gpu_util_percent','rss_bytes','system_available_bytes','swap_used_bytes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  while not stop.is_set():
   start=time.time();ph=phase;qid=current
   try:
    p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3);v=float(p.stdout.strip()) if p.returncode==0 else None
   except Exception:v=None
   rss,available,swap,total=memory();row=dict(query_start=start,query_end=time.time(),query=qid,phase=ph,gpu_util_percent=v,rss_bytes=rss,system_available_bytes=available,swap_used_bytes=swap);samples.append(row);w.writerow(row);f.flush()
   if available<8*2**30 or swap>initial_swap+2*2**30:
    event('memory_pressure_stop',available_bytes=available,swap_bytes=swap);os._exit(70)
   stop.wait(max(0,.5-(time.time()-start)))
thread=threading.Thread(target=monitor,daemon=True);thread.start()
summary=dict(status='running',run=str(O),settings=dict(model_type=MODELTYPE,num_models=5,model_order=[1,2,3,4,5],num_recycles=RECYCLES,recycle_early_stop_tolerance=0.0,num_ensemble=1,num_seeds=1,random_seed=42,use_templates=False,num_relax=0,dropout=False,msa_mode='mmseqs2_uniref_env',max_seq=MAXSEQ,max_extra_seq=MAXEXTRA,use_bfloat16=True,use_pallas=False,compile_mode='tuned',recompile_padding=10,batch='one protein at a time, five weight sets sequentially',preallocation=False),predictions=predictions)
write('summary.json',summary)
class TimedJit:
 def __init__(self,fn):self.fn=fn;self.cache={}
 def __getattr__(self,key):return getattr(self.fn,key)
 def __call__(self,*args,**kw):
  global phase
  leaves,tree=jax.tree_util.tree_flatten((args,kw));signature=(str(tree),tuple((tuple(getattr(x,'shape',())),str(getattr(x,'dtype',type(x)))) for x in leaves));oldphase=phase
  if signature not in self.cache:
   phase='compilation';t=time.perf_counter();start=time.time();compiled=self.fn.lower(*args,**kw).compile();seconds=time.perf_counter()-t;self.cache[signature]=compiled
   try:analysis=str(compiled.memory_analysis())
   except Exception:analysis=None
   event('compilation',start_unix=start,end_unix=time.time(),seconds=seconds,memory_analysis=analysis,shape_variants=len(self.cache))
  phase='execution';start=time.time();t=time.perf_counter();out=self.cache[signature](*args,**kw);jax.block_until_ready(out);seconds=time.perf_counter()-t;end=time.time();row=dict(query=current,start_unix=start,end_unix=end,seconds=seconds,device_memory=dmem());calls.append(row);event('gpu_apply',**{k:v for k,v in row.items() if k!='query'});phase=oldphase;return out
original_init=afmodel.RunModel.__init__
def timed_init(self,*args,**kw):original_init(self,*args,**kw);self.apply=TimedJit(self.apply)
afmodel.RunModel.__init__=timed_init
origload=models.load_models_and_params
@functools.wraps(origload)
def load(*args,**kw):
 global phase
 old=phase;phase='model_loading';t=time.perf_counter();out=origload(*args,**kw);jax.block_until_ready([p for n,r,p in out]);seconds=time.perf_counter()-t;counts=[dict(model=n,parameters=sum(int(x.size) for x in jax.tree_util.tree_leaves(p)),parameter_dtypes=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(p)})) for n,r,p in out];event('model_loading',seconds=seconds,models=counts);summary['loaded_models']=counts;summary['model_loading_seconds']=seconds;phase=old;return out
models.load_models_and_params=load
# Time host preprocessing separately; never alter returned features.
def timed_function(obj,name,label):
 old=getattr(obj,name)
 @functools.wraps(old)
 def fn(*a,**kw):
  global phase,current
  if name=='generate_input_feature':
   bound=inspect.signature(old).bind_partial(*a,**kw);seqs=bound.arguments.get('query_seqs_unique',[]);seq=''.join(seqs) if isinstance(seqs,list) else str(seqs)
   current=next((q['id'] for q in json.loads((R/'data'/MANIFEST).read_text())['queries'] if q['sequence']==seq),current)
  ph=phase;phase=label;t=time.perf_counter();result=old(*a,**kw);event(label,function=name,seconds=time.perf_counter()-t);phase=ph;return result
 setattr(obj,name,fn)
for name in ['unserialize_msa','generate_input_feature']:timed_function(batch,name,'preprocessing')
timed_function(afmodel.RunModel,'process_features','preprocessing')
origpredict=afmodel.RunModel.predict
@functools.wraps(origpredict)
def predict(self,feat,*args,**kw):
 global phase
 phase='prediction';start=time.perf_counter();event('actual_model_input',shapes={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in feat.items() if hasattr(v,'shape')},bfloat16=bool(self.config.model.global_config.bfloat16),training=False)
 result=origpredict(self,feat,*args,**kw);event('predict_total',seconds=time.perf_counter()-start,recycles=int(result[1]));phase='output';return result
afmodel.RunModel.predict=predict
origstructure=batch.predict_structure
@functools.wraps(origstructure)
def structure(*a,**kw):
 global current,phase
 bound=inspect.signature(origstructure).bind_partial(*a,**kw);current=str(bound.arguments['prefix']);phase='query';t=time.perf_counter();start=time.time();res=origstructure(*a,**kw);end=time.time();stat=dict(query=current,wall_seconds=time.perf_counter()-t,start_unix=start,end_unix=end);query_stats.append(stat);event('query_complete',**{k:v for k,v in stat.items() if k!='query'});write('query_stats.json',query_stats);phase='between_queries';return res
batch.predict_structure=structure
# Track explicit text output writes as a lower bound on serialization time.
origwrite=pathlib.Path.write_text
@functools.wraps(origwrite)
def timedwrite(self,*args,**kw):
 t=time.perf_counter();result=origwrite(self,*args,**kw)
 if str(self).startswith(str(O)) and self.suffix in ['.pdb','.json','.a3m'] and self.name not in ['summary.json','query_stats.json','input_report.json']:
  event('file_write',path=str(self),seconds=time.perf_counter()-t)
 return result
pathlib.Path.write_text=timedwrite

def callback(protein,lengths,result,features,tag):
 row=dict(query=current,tag=tag[0],lengths=list(map(int,lengths)),mean_plddt=float(np.mean(result['plddt'][:sum(lengths)])),ptm=float(result['ptm']) if 'ptm' in result else None,iptm=float(result['iptm']) if 'iptm' in result else None,plddt_shape=list(result['plddt'].shape),pae_shape=list(result['predicted_aligned_error'].shape) if 'predicted_aligned_error' in result else None,coordinates_shape=list(protein.atom_positions.shape),finite_coordinates=bool(np.isfinite(protein.atom_positions).all()),finite_plddt=bool(np.isfinite(result['plddt']).all()))
 assert row['finite_coordinates'] and row['finite_plddt'];predictions.append(row);event('validated_prediction',**{k:v for k,v in row.items() if k!='query'});write('predictions.json',predictions)
try:
 setup_logging(O/'log.txt');manifest=json.loads((R/'data'/MANIFEST).read_text());queries=[];report=[]
 for q in manifest['queries']:
  p=MSA/(q['id']+'.a3m');text=p.read_text();seqs=[]
  for line in text.splitlines():
   if line.startswith(('>','#')):continue
   if line.strip():seqs.append(''.join(c for c in line if not c.islower()))
  assert seqs and seqs[0]==q['sequence'];assert all(len(s)==q['length'] for s in seqs)
  report.append(dict(id=q['id'],residues=q['length'],msa_depth=len(seqs),msa_aligned_columns=len(seqs[0]),msa_characters=sorted(set(''.join(seqs))),msa_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),msa_bytes=p.stat().st_size,token_dtype='int32 features',truncated=False,templates=False))
  queries.append((q['id'],q['chains'] if COMPLEX else q['sequence'],[text],None))
 write('input_report.json',report);event('input_verified',queries=report);event('initial_memory',rss_bytes=memory()[0],available_bytes=memory()[1],system_total_bytes=memory()[3],gpu=dmem())
 phase='pipeline';start=time.perf_counter()
 batch.run(queries=queries,result_dir=O/'predictions',num_models=5,is_complex=COMPLEX,num_recycles=RECYCLES,recycle_early_stop_tolerance=0.0,model_order=[1,2,3,4,5],num_ensemble=1,model_type=MODELTYPE,msa_mode='mmseqs2_uniref_env',use_templates=False,num_relax=0,keep_existing_results=False,data_dir=pathlib.Path('/cache'),random_seed=42,num_seeds=1,recompile_padding=10,prediction_callback=callback,feature_dict_callback=lambda feat:event('raw_feature_shapes',shapes={k:list(v.shape) for k,v in feat.items() if hasattr(v,'shape')}),use_bfloat16=True,use_pallas=False,compile_mode='tuned',max_seq=MAXSEQ,max_extra_seq=MAXEXTRA)
 summary['pipeline_wall_seconds']=time.perf_counter()-start;assert len(predictions)==5*len(queries),(len(predictions),len(queries));summary['status']='success';summary['gpu_apply_seconds']=sum(c['seconds'] for c in calls);summary['compilation_seconds']=sum(e['seconds'] for e in events if e['kind']=='compilation');summary['final_device_memory']=dmem();summary['wall_seconds']=time.perf_counter()-allstart;write('summary.json',summary);write('apply_calls.json',calls);event('success',summary_path=str(O/'summary.json'))
except Exception as e:
 summary['status']='failed';summary['error']=str(e);summary['traceback']=traceback.format_exc();write('summary.json',summary);event('failed',error=str(e));traceback.print_exc();raise
finally:stop.set();thread.join(timeout=5)
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "colabfold/results/af2_multimer_20261027_181640_431467/benchmark_af2.py", "source": "colabfold/scripts/benchmark_af2.py", "sha256": "f11c9e66c3b8f3514780db7513ff135ca9f8f9fdc706dc0884f8b06b648cdee3"} -->

</details>

<details>
<summary>colabfold/scripts/inspect_environment.py — 원문 전문 (971 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/inspect_environment.py", "sha256": "2fcfbabaa25bad65104e67a2c41bebaaa1ba77dd4b4696ff956fe36acf894c3d", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,pathlib,json,inspect,importlib.metadata,os
import jax,numpy as np
from alphafold.model import model,config
from colabfold.alphafold import models
from colabfold import batch
r=pathlib.Path('/work');o=r/'results/environment_verified';o.mkdir(exist_ok=True)
info=dict(python=sys.version,versions={n:importlib.metadata.version(n) for n in ['colabfold','jax','jaxlib','dm-haiku','numpy']},devices=[str(d) for d in jax.devices()],environment={k:v for k,v in os.environ.items() if k.startswith(('XLA','JAX','TF_','CUDA'))},run_signature=str(inspect.signature(batch.run)),load_signature=str(inspect.signature(models.load_models_and_params)))
print(json.dumps(info,indent=2));(o/'environment.json').write_text(json.dumps(info,indent=2))
for obj,name in [(model,'alphafold_model.py'),(models,'colabfold_models.py'),(batch,'batch.py')]:
 (o/name).write_text(inspect.getsource(obj))
a=jax.numpy.ones((32,32));(a@a).block_until_ready();print('JAX GPU execution verified')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/scripts/launch.py — 원문 전문 (1,438 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/launch.py", "sha256": "c5a623b4a8a1e0efcf840b18ac32c52d82447095e594d328fcd34c2227a05f53", "encoding": "utf-8", "added_final_newline": false} -->
```python
import pathlib,subprocess,json,datetime,sys
r=pathlib.Path(__file__).resolve().parents[1];msa=(r/'results/MSA_LATEST').read_text().strip();meta=json.loads((r/'results'/msa/'exit_code.json').read_text());assert meta['exit_code']==0
q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True);assert q.returncode==0 and not q.stdout.strip(),'Another GPU compute job exists; do not launch'
name='af2_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');o=r/'results'/name;o.mkdir();(r/'results/AF2_LATEST').write_text(name+'\n');cmd=['sg','docker','-c','docker run --rm --name '+name+' --gpus all --network none --user 1000:1000 -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e MPLCONFIGDIR=/work/results/'+name+'/mplcache -v '+str(r)+':/work -v '+str(r/'cache/colabfold')+':/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 python -u /work/scripts/benchmark_af2.py '+name+' '+msa];(o/'command.json').write_text(json.dumps(cmd,indent=2));print('AF2_RUN',name,flush=True);start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (o/'console.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(o/'exit_code.json').write_text(json.dumps(dict(exit_code=p.returncode,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2));print('AF2_EXIT',p.returncode,flush=True);sys.exit(p.returncode)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/scripts/launch_multimer.py — 원문 전문 (1,590 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/launch_multimer.py", "sha256": "ef4104935d86cc6e0f25ee6e08f4af0e1ce26f8494b77ef71650e950e8d6d73f", "encoding": "utf-8", "added_final_newline": false} -->
```python
import pathlib,subprocess,json,datetime,sys
r=pathlib.Path(__file__).resolve().parents[1];mono=(r/'results/AF2_LATEST').read_text().strip();assert json.loads((r/'results'/mono/'exit_code.json').read_text())['exit_code']==0
q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True);assert q.returncode==0 and not q.stdout.strip(),'Another GPU compute job exists'
msa=(r/'results/MSA_COMPLEX_LATEST').read_text().strip();assert json.loads((r/'results'/msa/'exit_code.json').read_text())['exit_code']==0
name='af2_multimer_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');o=r/'results'/name;o.mkdir();(r/'results/MULTIMER_LATEST').write_text(name+'\n');cmd=['sg','docker','-c','docker run --rm --name '+name+' --gpus all --network none --user 1000:1000 -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e MPLCONFIGDIR=/work/results/'+name+'/mplcache -v '+str(r)+':/work -v '+str(r/'cache/colabfold')+':/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 python -u /work/scripts/benchmark_af2.py '+name+' '+msa+' multimer'];(o/'command.json').write_text(json.dumps(cmd,indent=2));print('MULTIMER_RUN',name,flush=True);start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (o/'console.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(o/'exit_code.json').write_text(json.dumps(dict(exit_code=p.returncode,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2));print('MULTIMER_EXIT',p.returncode,flush=True);sys.exit(p.returncode)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/scripts/measure_loading.py — 원문 전문 (2,257 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/measure_loading.py", "sha256": "39667da7763f11bbaacf0a1b1e1264bb28b641c51600b6ba4af63cf5355c1b50", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,time,datetime,json,gc
from colabfold import batch
import jax
from colabfold.alphafold.models import load_models_and_params
R=Path('/work');preset=sys.argv[1];O=R/'results'/('loading_'+preset+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));O.mkdir();(R/'results'/('LOADING_'+preset+'_LATEST')).write_text(O.name+'\n');d=jax.devices('gpu')[0]
def rss():return next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
before=rss();mbefore=d.memory_stats();t=time.perf_counter();out=load_models_and_params(num_models=5,use_templates=False,num_recycles=20 if 'multimer' in preset else 3,recycle_early_stop_tolerance=0.,num_ensemble=1,model_order=[1,2,3,4,5],model_type=preset,data_dir=Path('/cache'),max_seq=508 if 'multimer' in preset else 512,max_extra_seq=2048 if 'multimer' in preset else 5120,use_bfloat16=True,use_pallas=False,compile_mode='tuned');jax.block_until_ready([p for n,r,p in out]);stats=d.memory_stats();out[0][1].params=out[0][2];gc.collect();jax.block_until_ready(out[0][1].params);ready=d.memory_stats();row=dict(model_type=preset,weight_sets=5,parameters_per_model=[dict(name=n,parameters=sum(int(x.size) for x in jax.tree_util.tree_leaves(p))) for n,r,p in out],parameter_dtype='float32',compute_configuration='bfloat16',load_seconds=time.perf_counter()-t,device_memory_before=mbefore,device_memory_after=stats,after_load_allocated_gib=stats['bytes_in_use']/2**30,before_first_forward_ready_allocated_gib=ready['bytes_in_use']/2**30,device_memory_ready=ready,environment=dict(TF_FORCE_UNIFIED_MEMORY=__import__('os').environ.get('TF_FORCE_UNIFIED_MEMORY'),XLA_PYTHON_CLIENT_MEM_FRACTION=__import__('os').environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')),loading_peak_gib=stats['peak_bytes_in_use']/2**30,rss_before_load_gib=before/2**30,rss_after_load_gib=rss()/2**30,scope='Fresh process and same image; five official weight sets loaded, no input features, forward, compile or optimizer; loader snapshot may keep an extra initial runner parameter reference; ready snapshot selects model_1 as normal batch.run does, then releases unused references');(O/'loading.json').write_text(json.dumps(row,indent=2));print(json.dumps(row,indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/scripts/prepare_inputs.py — 원문 전문 (1,884 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/prepare_inputs.py", "sha256": "2a327312228528a70d4ba6e8003cee0096ce9059a1b0d67c510b05d7561b1e7f", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,hashlib,ssl,urllib.request,datetime
R=Path('/home/hyein/ai-benchmark/colabfold');D=R/'data';ctx=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt');ss=[]
for line in (R.parent/'esm2/data/uniprot_reviewed.fasta').read_text().splitlines():
 if line.startswith('>'):ss.append([line[1:].split()[0],''])
 else:ss[-1][1]+=line
pool=[(i,s) for i,s in ss if len(s)==115];unique=[];seen=set()
for i,s in pool:
 if s not in seen:unique.append((i,s));seen.add(s)
assert len(unique)>=4
queries=[]
for n,(i,s) in enumerate(unique[:4],1):queries.append(dict(id='count'+str(n)+'_'+i.split('|')[1],sequence=s,length=len(s),kind='count',source='UniProtKB reviewed 2026_03; reused ESM-2 source snapshot',source_id=i))
for pdb in ['1UBQ','1LYZ','1TIM']:
 p=D/(pdb+'.fasta');seq=''.join(l.strip() for l in p.read_text().splitlines() if not l.startswith('>'));queries.append(dict(id=pdb,sequence=seq,length=len(seq),kind='length',source='https://www.rcsb.org/fasta/entry/'+pdb,reference_chain='A'))
 ref=D/(pdb+'.pdb')
 if not ref.exists():
  with urllib.request.urlopen('https://files.rcsb.org/download/'+pdb+'.pdb',context=ctx,timeout=90) as f:ref.write_bytes(f.read())
for q in queries:
 assert set(q['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY') and q['length']>0
queries.sort(key=lambda q:(q['length'],q['id']))
(D/'queries.fasta').write_text(''.join('>'+q['id']+'\n'+q['sequence']+'\n' for q in queries));(D/'input_manifest.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unique_sequences=len(set(q['sequence'] for q in queries)),queries=queries,format='FASTA amino-acid sequences; PDB reference chain A; no labels for count proteins',missing_sequences=0,sequence_dtype='text',preprocessed=False,msa='not yet generated'),indent=2))
print([(q['id'],q['length']) for q in queries],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/scripts/report_af2.py — 원문 전문 (24,538 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/report_af2.py", "sha256": "34c4855a7ffc5dd9b66aa88a94f1f69230d9a5d5a05917028eb48d667d6e3cb3", "encoding": "utf-8", "added_final_newline": false} -->
````python
"""CPU-only report of measured AlphaFold2/ColabFold runs."""
from pathlib import Path
import json,csv,statistics,datetime,hashlib,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/work');HOST='/home/hyein/ai-benchmark/colabfold';REPORTS=R/'reports';REPORTS.mkdir(exist_ok=True)
def host(p):return str(p).replace('/work',HOST,1)
def read(p):return json.loads(p.read_text())
def fmt(x,n=3):return 'N/A' if x is None else f'{x:.{n}f}'
mono=R/'results'/(R/'results/AF2_LATEST').read_text().strip();multi=R/'results'/(R/'results/MULTIMER_LATEST').read_text().strip();runs=[mono,multi];summary=[read(o/'summary.json') for o in runs];assert all(s['status']=='success' for s in summary);assert all(read(o/'exit_code.json')['exit_code']==0 for o in runs);assert all(read(o/'structure_validation.json')['passed'] for o in runs)
allrows=[];aggregate=[]
for o in runs:
 events=[json.loads(l) for l in (o/'events.jsonl').read_text().splitlines()];samples=list(csv.DictReader((o/'resources.csv').open()));calls=read(o/'apply_calls.json');qstats=read(o/'query_stats.json');inputs=read(o/'input_report.json');validation=read(o/'structure_validation.json')['results'];stats={q['query']:q for q in qstats}
 for q in inputs:
  name=q['id'];cc=[c for c in calls if c['query']==name];ss=[s for s in samples if any(float(s['query_start'])>=c['start_unix'] and float(s['query_end'])<=c['end_unix'] for c in cc)];us=[float(s['gpu_util_percent']) for s in ss if s['gpu_util_percent']];actual=[e for e in events if e['kind']=='actual_model_input' and e['query']==name];first=actual[0]['shapes'];row=dict(run=o.name,query=name,unique_inputs=1,total_processed_proteins=1,predicted_models=5,gpu_apply_calls=len(cc),original_residues=q['residues'],actual_residues=q['residues'],padded_residue_axis=first['aatype']['shape'][-1],original_a3m_rows=q['msa_depth'],actual_feature_msa_shape=first.get('msa',first.get('msa_mask'))['shape'],extra_msa_shape=first.get('extra_msa',{}).get('shape'),truncated=False,compute_seconds=sum(c['seconds'] for c in cc),compilation_seconds=sum(e['seconds'] for e in events if e['kind']=='compilation' and e['query']==name),query_wall_seconds=stats[name]['wall_seconds'],protein_throughput_per_second=1/sum(c['seconds'] for c in cc),model_throughput_per_second=5/sum(c['seconds'] for c in cc),gpu_util_mean_percent=statistics.mean(us) if us else None,gpu_util_max_percent=max(us) if us else None,sample_count=len(ss),rss_sampled_peak_gib=max([int(s['rss_bytes']) for s in ss],default=0)/2**30,available_min_gib=min([int(s['system_available_bytes']) for s in ss],default=0)/2**30,swap_max_bytes=max([int(s['swap_used_bytes']) for s in ss],default=0),jax_allocator_peak_gib_since_process_start=max([c['device_memory'].get('peak_bytes_in_use',0) for c in cc],default=0)/2**30,jax_live_bytes_max_at_apply_return_gib=max([c['device_memory'].get('bytes_in_use',0) for c in cc],default=0)/2**30,query_start=stats[name]['start_unix'],query_end=stats[name]['end_unix'])
  aggregate.append(row)
 for v in validation:allrows.append(dict(run=o.name,**v))
 (o/'aggregated_metrics.json').write_text(json.dumps([r for r in aggregate if r['run']==o.name],indent=2))
weights=[]
for p in sorted(Path('/cache/params').glob('params_model_*')):
 if not (p.name.endswith('_ptm.npz') or p.name.endswith('_multimer_v3.npz')):continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 with np.load(p,allow_pickle=False) as d:count=sum(int(np.prod(d[k].shape)) for k in d.files);dtypes=sorted({str(d[k].dtype) for k in d.files})
 weights.append(dict(file=p.name,bytes=p.stat().st_size,parameters_in_npz=count,dtypes=dtypes,sha256=h.hexdigest()))
(REPORTS/'weight_manifest.json').write_text(json.dumps(weights,indent=2));(REPORTS/'metrics.json').write_text(json.dumps(aggregate,indent=2))
fields=[k for k in aggregate[0] if k not in ['actual_feature_msa_shape','extra_msa_shape']]
with (REPORTS/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(aggregate)
# Nested streaming prefixes: same 115-residue proteins, shape and fixed-capacity policy.
counts=[r for r in aggregate if r['query'].startswith('count')];counts.sort(key=lambda x:x['query']);prefix=[]
for n in [1,2,4]:
 rows=counts[:n];sec=sum(r['compute_seconds'] for r in rows);total_samples=sum(r['sample_count'] for r in rows);prefix.append(dict(unique_inputs=n,processed_inputs=n,predicted_models=5*n,gpu_forward_calls=20*n,compute_seconds=sec,query_wall_sum_seconds=sum(r['query_wall_seconds'] for r in rows),compile_seconds=sum(r['compilation_seconds'] for r in rows),proteins_per_second=n/sec,models_per_second=5*n/sec,gpu_util_mean_percent=sum(r['gpu_util_mean_percent']*r['sample_count'] for r in rows)/total_samples,gpu_util_max_percent=max(r['gpu_util_max_percent'] for r in rows),sample_count=total_samples,jax_peak_gib_cumulative=max(r['jax_allocator_peak_gib_since_process_start'] for r in rows),rss_sampled_peak_gib=max(r['rss_sampled_peak_gib'] for r in rows),available_min_gib=min(r['available_min_gib'] for r in rows)))
(REPORTS/'count_prefix_metrics.json').write_text(json.dumps(prefix,indent=2))
# Standalone scientific charts, with explicit timing scope and units.
lengths=[next(r for r in aggregate if r['query']==q) for q in ['1UBQ','1LYZ','1TIM']]
fig,ax=plt.subplots(1,3,figsize=(14,4));ax[0].plot([x['unique_inputs'] for x in prefix],[x['compute_seconds'] for x in prefix],'o-');ax[0].set(xlabel='Unique proteins (115 residues each)',ylabel='Synchronized GPU apply total (s)',title='Nested stream prefixes (5 models/protein)');ax[1].plot([x['original_residues'] for x in lengths],[x['compute_seconds'] for x in lengths],'o-');ax[1].set(xlabel='Residues per protein',ylabel='Synchronized GPU apply total (s)',title='Monomers; original MSA depths differ');ax[2].bar([x['query'] for x in lengths]+['complex'],[x['gpu_util_mean_percent'] for x in lengths]+[aggregate[-1]['gpu_util_mean_percent']]);ax[2].axhline(80,color='grey',linestyle='--');ax[2].set(ylabel='Device GPU-Util sample mean (%)',ylim=(0,100),title='Only samples inside GPU apply intervals');fig.tight_layout();fig.savefig(REPORTS/'performance.png',dpi=180);plt.close(fig)
out=[]
def add(s=''):out.append(s)
def table(h,rows):
 add('| '+' | '.join(h)+' |');add('| '+' | '.join(['---']*len(h))+' |')
 for row in rows:add('| '+' | '.join(map(str,row))+' |')
 add()
msanames=[(R/'results/MSA_LATEST').read_text().strip(),(R/'results/MSA_COMPLEX_LATEST').read_text().strip()];msameta=[read(R/'results'/n/'exit_code.json') for n in msanames]
add('# AlphaFold2 — ColabFold를 통한 실행: DGX Spark 검증');add();add('작성 시각: '+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds'));add();add('## 완료 범위와 주요 결과');add();add('기존 ColabFold 1.6.2 ARM64 이미지·가중치를 재사용해 단일체 7개 × 5개 모델 = 35개 구조와 복합체 1개 × Multimer v3 모델 5개 = 5개 구조를 실제 GB10에서 순차 예측했다. 모든 실행은 별도 호스트 run 폴더에 저장했으며 단일체와 복합체 GPU 작업을 겹치지 않았다. PDB·신뢰도 JSON·MSA·좌표의 유한값과 실제 서열을 검증했다. 구조 예측용 ColabFold 경로를 사용했으므로 공식 전체 AlphaFold2 데이터베이스 검색 파이프라인과 구분한다.');add();add('개수 비교는 동일 115 잔기 단백질 4개를 한 스트림으로 처리하고, 첫 1/2/4개 중첩 prefix를 집계했다. 조건별 독립 재실행이나 동일 입력 반복 측정이 아니다. 길이 비교는 76/129/247 잔기 실제 단백질을 사용한다. MSA는 각각 따로 준비하고 실제 깊이 차이를 공개한다. GPU 80%를 맞추기 위한 배치·정밀도 sweep은 하지 않았다.');add()
add('## 환경과 실행 조건');add();table(['항목','단일체','복합체'],[['이미지','ghcr.io/sokrypton/colabfold:1.6.2-cuda13','동일 이미지'],['digest','sha256:c9eab0253fe324b7ec2ec644607141539d7580a4e88d1d3788abba8c92d61ef1','동일'],['GPU / 아키텍처','NVIDIA GB10 / ARM64','동일'],['Python / JAX / jaxlib','3.12.12 / 0.10.2 / 0.10.2','동일'],['NumPy / Haiku','2.5.1 / 0.0.16','동일'],['모델 preset','alphafold2_ptm','alphafold2_multimer_v3'],['순차 weight set 수','5 (model_1~5)','5 (model_1~5)'],['실행 파라미터/tree (각 model)',f"{summary[0]['loaded_models'][0]['parameters']:,}",f"{summary[1]['loaded_models'][0]['parameters']:,}"],['학습 가능 / 학습 수행','0 / 추론만 수행','0 / 추론만 수행'],['recycle','3 (최초 pass 포함 4회)','20 (최초 pass 포함 21회)'],['조기 중단 tolerance','0.0 (비교를 위해 고정)','0.0 (고정)'],['ensemble / seed','1 / 42, seed 수 1','동일'],['MSA 정책','MMseqs2 UniRef + environmental','unpaired_paired, greedy'],['MSA 모델 capacity','512 clustered + 5120 extra','508 clustered + 2048 extra'],['정밀도','BF16 계산, FP32 weights, FP16 recycle buffers/출력 변환','동일 혼합 정밀도 정책'],['template / dropout / relaxation','모두 비활성화','모두 비활성화'],['compile mode / Pallas','tuned / False','동일'],['배치','단백질 1개씩, weight set 5개를 순차 실행','복합체 1개씩, weight set 5개를 순차 실행'],['JAX preallocation','False','False'],['JAX backends','cuda,cpu: CPU feature 준비 + GPU forward','동일']]);add('일반적인 ColabFold 기본 5개 모델·기본 BF16 경로를 사용했다. 단일체 recycle 3과 Multimer v3 recycle 20을 명시하고 tolerance를 0으로 고정했다. 모델별 기본 recycle이 다르므로 단일체와 복합체 시간을 같은 길이 효과로 직접 비교하지 않는다. BF16은 전체 배열이 BF16이라는 뜻이 아니다. GPU 메모리를 채우기 위한 정밀도 변경은 하지 않았다.');add();add('기존 모델·이미지·가중치·scGPT 컨테이너를 변경하지 않았다. Docker 그룹 등록은 사용자가 수행했고 현재 명령은 sg docker로 등록된 그룹을 적용했다. 가중치는 read-only `/cache`, 결과는 host bind `/work`에 저장했다. 예측 컨테이너는 network none이므로 MSA API 시간과 GPU 예측을 분리했다.');add()
add('## 입력·MSA·출처 확인');add();table(['입력','원본/실제 residues','원본 A3M 행','실제 feature residue 축','MSA tensor shape','truncation'],[[r['query'],r['original_residues'],r['original_a3m_rows'],r['padded_residue_axis'],str(r['actual_feature_msa_shape']),'없음'] for r in aggregate]);add('개수용 UniProt 단백질은 ESM-2 단계에서 확보한 UniProtKB reviewed 2026_03 원본을 재사용했으며 서로 다른 고유 서열 4개를 검사했다. 모두 115 residues지만 MSA의 깊이·내용은 다르다. 길이용 FASTA와 실험 구조는 RCSB PDB 1UBQ/1LYZ/1TIM의 공개 데이터를 확보했다. 복합체는 ColabFold v1.6.2 공식 test-data/complex/input.csv의 3G5O A/B 예제(93+87 residues)이며, RCSB 전체 construct에 있는 tag 일부가 생략된 입력이다. 3G5O의 전체 biological assembly를 모두 예측한 것이 아니다.');add();add('A3M에서 소문자 insertion을 제외한 alignment column 수와 첫 query가 원본 입력과 일치하는지 확인했다. 표의 A3M 행은 검색 결과의 저장 행이고 feature 구성 과정의 중복 제거 후 깊이는 events.jsonl의 raw_feature_shapes에 따로 기록된다. 단일체 feature는 4개 recycle용 입력 축과 512 clustered/5120 extra capacity로 padding된다. 복합체는 원본 alignment와 모델 내부 sampled MSA capacity를 구분해야 한다. dummy template tensor가 존재해도 실제 template 정보는 사용하지 않는다. 실제 모델 입력의 shape·dtype는 events.jsonl에 보존했다.');add()
add('## 시간·자원 계측 방법');add();add('기존 RunModel의 JIT apply를 의미가 같은 lower(...).compile()과 compiled callable 호출로 나누어 컴파일 시간을 측정했다. 새로운 shape에서만 AOT 컴파일하며 같은 shape·dtype·PyTree 구조의 다음 호출에는 실행 파일을 재사용했다. 매 GPU apply 결과에 jax.block_until_ready를 적용했다. GPU apply는 forward 1회이며 recycle과 model 개수를 합쳐 집계한다. CPU feature 구성, pLDDT/pTM·JSON/PDB 후처리, MSA API, 컴파일과는 범위가 다르다. 최초 shape의 첫 apply는 cold kernel 실행이며 포함한 상태로 보고한다. 별도 인위적 warmup을 수행하지 않았고 단일 프로세스 측정이다.');add();add('0.5초 간격으로 nvidia-smi GPU-Util, 프로세스 RSS, 시스템 MemAvailable·swap을 수집했다. 평균·최대는 조회 시작과 종료가 모두 GPU apply 구간 안에 있는 표본만 집계한다. 장치 전체 GPU-Util에는 디스플레이·브라우저 부하가 포함될 수 있으며 FLOPS나 SM occupancy 비율이 아니다. 각 apply 경계의 짧은 표본은 제외돼 표본 수가 단순 시간/0.5와 다를 수 있다.');add();add('JAX memory_stats의 peak_bytes_in_use는 프로세스 시작 후 누적 최고치이며 query별로 초기화하지 않았다. per-query 표는 이를 명확히 구분한다. 실제 apply 반환 시점의 live allocation도 별도 JSON에 기록했다. Compile memory_analysis는 정적 buffer 분석으로 실제 allocator peak와 다르다. TF_FORCE_UNIFIED_MEMORY=1, XLA_PYTHON_CLIENT_MEM_FRACTION=4.0은 기존 ColabFold 코드가 적용하는 설정이며 bytes_limit가 물리 RAM의 4배로 표시돼도 이를 전용 VRAM 용량이나 점유율 분모로 사용하지 않았다. Spark의 공유 DRAM에서 RSS와 CUDA/JAX 수치를 더하지 않았다. nvidia-smi 장치 memory 사용량은 Not Supported이므로 N/A다.');add()
add('## 개수 증가: 동일 길이 115 residues, 스트림 누적 prefix');add();table(['고유/처리 입력','구조 수','GPU apply 수','GPU 실행 합계 s','protein/s','GPU 평균/최대 %','표본 수'],[[f"{x['unique_inputs']} / {x['processed_inputs']}",x['predicted_models'],x['gpu_forward_calls'],fmt(x['compute_seconds']),fmt(x['proteins_per_second'],5),f"{x['gpu_util_mean_percent']:.2f} / {x['gpu_util_max_percent']:.0f}",x['sample_count']] for x in prefix]);table(['입력 수','query wall 합계 s','컴파일 s (wall에 포함)','JAX 누적 peak GiB','RSS sampled peak GiB','MemAvailable 최소 GiB'],[[x['unique_inputs'],fmt(x['query_wall_sum_seconds']),fmt(x['compile_seconds']),fmt(x['jax_peak_gib_cumulative']),fmt(x['rss_sampled_peak_gib']),fmt(x['available_min_gib'])] for x in prefix]);add('모든 prefix는 같은 125 padded residue 축과 같은 MSA capacity로 처리됐다. 첫 115-residue query가 새 shape 컴파일 비용을 부담하고 뒤 query는 재사용한다. 따라서 query wall 차이와 GPU apply 시간을 구분한다. 고유 입력 1/2/4는 한 번 실행한 스트림의 중첩 집계이므로 독립 반복 실행의 평균이나 신뢰구간을 의미하지 않는다. 개수 증가에 따라 순간 GPU 메모리와 사용률이 비례해서 증가한다고 해석하지 않는다.');add()
add('## 단일체 길이 비교와 복합체 개별 결과');add();table(['입력','길이','GPU apply s','컴파일 s','query wall s','GPU 평균/최대 %','표본 수'],[[x['query'],x['original_residues'],fmt(x['compute_seconds']),fmt(x['compilation_seconds']),fmt(x['query_wall_seconds']),f"{x['gpu_util_mean_percent']:.2f} / {x['gpu_util_max_percent']:.0f}",x['sample_count']] for x in lengths+[aggregate[-1]]]);table(['입력','JAX 누적 peak GiB','apply 반환 live 최대 GiB','RSS sampled peak GiB','MemAvailable 최소 GiB','swap bytes'],[[x['query'],fmt(x['jax_allocator_peak_gib_since_process_start']),fmt(x['jax_live_bytes_max_at_apply_return_gib']),fmt(x['rss_sampled_peak_gib']),fmt(x['available_min_gib']),x['swap_max_bytes']] for x in lengths+[aggregate[-1]]]);add('단일체 길이가 증가하면서 실행 시간이 증가했다. 서로 다른 단백질과 MSA 깊이를 사용했으므로 완전한 길이 단독 인과 실험은 아니다. MSA와 template·모델·정밀도 정책은 고정했으며 실제 padding과 MSA shape를 공개했다. 복합체는 다른 Multimer v3 체크포인트와 20 recycles를 사용하므로 180 길이를 단일체 길이 곡선에 섞지 않았다. 단일체·복합체 모두 지원되는 작은 입력에서 이미 GPU 평균 80%를 넘었고 더 큰 입력으로 사용률을 억지로 맞추지 않았다.');add();add('![입력 규모별 시간과 GPU 사용률](performance.png)');add()
add('## MSA·모델 로딩·컴파일·실행·저장 구분');add();table(['실행','MSA 준비 wall s','모델 로딩 s','컴파일 s','GPU apply s','pipeline wall s','컨테이너 포함 wall s'],[[label,fmt((datetime.datetime.fromisoformat(mm['end_utc'])-datetime.datetime.fromisoformat(mm['start_utc'])).total_seconds()),fmt(ss['model_loading_seconds']),fmt(ss['compilation_seconds']),fmt(ss['gpu_apply_seconds']),fmt(ss['pipeline_wall_seconds']),fmt((datetime.datetime.fromisoformat(read(o/'exit_code.json')['end_utc'])-datetime.datetime.fromisoformat(read(o/'exit_code.json')['start_utc'])).total_seconds())] for label,mm,ss,o in zip(['단일체 7개','복합체 1개'],msameta,summary,runs)]);add('MSA wall에는 컨테이너 시작·API 대기/다운로드·MSA 저장·일부 전처리/coverage 그림이 포함돼 순수 검색 서버 계산 시간이 아니다. GPU 예측에서는 저장된 A3M을 재사용했고 네트워크 접근은 차단했다. pipeline wall은 로딩·전처리·컴파일·GPU·후처리·모든 파일 저장을 포함한다. query wall은 predict_structure 단위로 그 앞의 raw MSA 파싱·일부 feature 생성과 global 모델 로딩을 제외한다. 파일 쓰기 이벤트는 PDB/JSON/A3M text write 시간의 하한이며 PNG·ranking rename·모든 serialization을 완전 분리한 저장 시간은 N/A다. Relaxation은 0회로 실행하지 않았다. 서로 다른 계측 범위를 직접 속도 개선으로 비교하지 않는다.');add()
add('## 구조 검증과 정확도 해석');add();table(['1위 구조','pLDDT 평균','pTM','CA RMSD Å','CA-only lDDT','정렬 CA 수'],[[x['query'],fmt(x['mean_plddt'],2),fmt(x['ptm']),fmt(x.get('ca_rmsd_angstrom')),fmt(x.get('ca_lddt')),x.get('aligned_ca','N/A')] for x in allrows if x['run']==mono.name and x['rank']==1]);complexbest=next(x for x in allrows if x['run']==multi.name and x['rank']==1);table(['복합체 1위 항목','결과'],[['pLDDT 평균',fmt(complexbest['mean_plddt'],2)],['pTM / ipTM',fmt(complexbest['ptm'])+' / '+fmt(complexbest['iptm'])],['사슬 A CA RMSD Å / aligned CA',fmt(complexbest['chain_A']['ca_rmsd_angstrom'])+' / '+str(complexbest['chain_A']['aligned_ca'])],['사슬 B CA RMSD Å / aligned CA',fmt(complexbest['chain_B']['ca_rmsd_angstrom'])+' / '+str(complexbest['chain_B']['aligned_ca'])],['전체 정렬 CA RMSD Å',fmt(complexbest['complex_alignment']['ca_rmsd_angstrom'])],['전체 정렬 CA-only lDDT',fmt(complexbest['complex_alignment']['ca_lddt'])]]);add('모든 구조에서 CA 서열이 입력과 일치하고 좌표·pLDDT·PAE가 유한하며, pLDDT [L], PAE [L,L]을 검사했다. 정답 구조 대비 수치는 global sequence alignment로 대응 CA를 선정한 후 Kabsch RMSD와 CA-only lDDT(15Å 이내 쌍, 0.5/1/2/4Å 기준)를 계산했다. CA-only lDDT를 공식 all-atom lDDT나 자체 예측 pLDDT와 동일시하지 않는다. pLDDT/pTM/ipTM은 신뢰도이며 정답 구조 대비 정확도와 별개다.');add();add('3G5O reference는 PDB A/B 사슬이며 query에 생략된 tag와 실험 구조의 missing residue가 있어 alignment 범위를 명시했다. 전체 complex 정렬은 고정된 A/B 순서의 concatenated sequence 대응을 사용한 sanity check이며 DockQ나 CAPRI 평가를 수행하지 않았다. 이 단백질들은 오래된 공개 PDB 구조로 학습 데이터와의 독립성을 입증하지 못한다. 따라서 정상 작동 확인용 구조 일치 결과이며 신규 단백질 일반화 성능을 입증하는 독립 benchmark가 아니다.');add()
add('## 학습·저장·재현 상태');add();add('현재 ColabFold CLI와 호출한 RunModel 경로는 구조 추론용이다. optimizer·훈련 loop·지도학습 split을 구성하지 않았고 파인튜닝을 수행하지 않았다. is_training 인자가 존재하는 것만으로 이 실행에서 지원되는 완성형 학습 workflow가 있다고 표현하지 않는다. 새 학습 checkpoint·optimizer·재개 학습·학습 모델 재로드는 N/A다. 기존의 공식 가중치를 재사용했고 출력 PDB·JSON을 새 CPU 프로세스에서 다시 읽어 저장 결과를 검증했다.');add();add('최초 단일체 시도는 JAX_PLATFORMS=cuda만 지정해 CPU 전처리 backend가 비활성화돼 실패했다. ColabFold의 일반적인 "Not Enough GPU memory?" 로그와 달리 실제 예외는 "Unknown backend cpu"였고 OOM이 아니었다. 실패 run과 로그를 보존한 뒤 cuda,cpu로 수정해 새 run에서 성공했다. GPU 실행은 dev=cuda:0와 실제 JAX GPU apply로 확인했다.');add()
add('## 저장 경로와 실행 방법');add();table(['항목','호스트 경로'],[['단일체 run',host(mono)],['복합체 run',host(multi)],['단일체 MSA',host(R/'results'/msanames[0])],['복합체 MSA',host(R/'results'/msanames[1])],['환경·원본 source',host(R/'results/environment_verified')],['입력·정답 PDB',host(R/'data')],['기존 가중치',HOST+'/cache/colabfold/params'],['계측 코드',HOST+'/scripts/benchmark_af2.py'],['CPU 구조 검증',HOST+'/scripts/validate_structures.py'],['비교 CSV / metrics JSON',HOST+'/reports/comparison.csv / metrics.json'],['weight SHA-256 / NPZ 파라미터',HOST+'/reports/weight_manifest.json'],['보고서',HOST+'/reports/AlphaFold2_ColabFold.md']]);add('각 run에는 command.json, console.log, log.txt, events.jsonl, resources.csv, exit_code.json, summary.json, input_report.json, apply_calls.json, query_stats.json, predictions/와 structure_validation.json이 있다. 모든 컨테이너 출력은 호스트 bind mount 아래에 보존된다. run별 코드 snapshot도 보관했다. 10개 실제 사용 가중치 NPZ의 해시·dtype·원본 파라미터 수는 weight_manifest.json에 저장했다. 템플릿 비활성화 시 model3 구조를 공유하는 loaded tree는 원본 NPZ 전체와 파라미터 수가 다를 수 있다.');add();add('```bash');add('cd /home/hyein/ai-benchmark');add('# 기존 출력은 보존하고 새 run ID를 생성한다. GPU compute 작업이 있으면 launch를 중단한다.');add('python3 colabfold/scripts/launch.py');add('python3 colabfold/scripts/launch_multimer.py');add('```');add()
add('## 한계·다음 작업');add();add('메모리 압박·swap 급증·OOM은 관찰되지 않았다. GPU 사용률과 메모리 용량을 혼동하지 않았으며 순간 메모리 또는 물리 RAM을 80% 채우는 것을 목표로 하지 않았다. 단일 seed·단일 프로세스의 작은 입력 범위이며 confidence calibration, 대규모 생산 처리량, 여러 MSA seed, docking accuracy, relaxation과 학습 성능은 미검증이다. 다음 모델 순서는 scVI이고, 기존 scGPT 정규화 이력이 불명확한 입력을 raw count로 그대로 재사용하면 안 된다.');add()
add('## 공식 출처');add();
for title,url in [('ColabFold 저장소·사용법','https://github.com/sokrypton/ColabFold'),('설치 버전의 모델 로딩·BF16 설정','https://github.com/sokrypton/ColabFold/blob/v1.6.2/colabfold/alphafold/models.py'),('설치 버전의 batch CLI·MSA·padding','https://github.com/sokrypton/ColabFold/blob/v1.6.2/colabfold/batch.py'),('공식 복합체 예제','https://github.com/sokrypton/ColabFold/blob/v1.6.2/test-data/complex/input.csv'),('AlphaFold2 공식 저장소','https://github.com/google-deepmind/alphafold'),('RCSB PDB 1UBQ','https://www.rcsb.org/structure/1UBQ'),('RCSB PDB 1LYZ','https://www.rcsb.org/structure/1LYZ'),('RCSB PDB 1TIM','https://www.rcsb.org/structure/1TIM'),('RCSB PDB 3G5O','https://www.rcsb.org/structure/3G5O'),('JAX AOT 컴파일','https://docs.jax.dev/en/latest/aot.html'),('Spark 공유 메모리 계측 한계','https://docs.nvidia.com/dgx/dgx-spark/known-issues.html')]:add('- ['+title+']('+url+')')
(REPORTS/'AlphaFold2_ColabFold.md').write_text('\n'.join(out)+'\n')
print('REPORT',host(REPORTS/'AlphaFold2_ColabFold.md'));print('Validated',len(allrows),'structures; plot/CSV/JSON saved')
````
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/scripts/validate_structures.py — 원문 전문 (3,770 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/scripts/validate_structures.py", "sha256": "8137f03a6ec35c630f5843b9fb7c61dfe97715857676cc0b562b3b6a07a45500", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""CPU-only validation of persisted predicted structures and experimental references."""
import pathlib,json,csv,hashlib,sys,numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.Align import PairwiseAligner
R=pathlib.Path('/work');complex_mode=len(sys.argv)>1 and sys.argv[1]=='multimer';run=(R/'results'/('MULTIMER_LATEST' if complex_mode else 'AF2_LATEST')).read_text().strip();O=R/'results'/run;manifest=json.loads((R/'data'/('complex_manifest.json' if complex_mode else 'input_manifest.json')).read_text());qs={q['id']:q for q in manifest['queries']};parser=PDBParser(QUIET=True);results=[]
def ca_chain(path,chain='A'):
 structure=parser.get_structure('p',path)[0];c=structure[chain];res=[r for r in c if r.id[0]==' ' and 'CA' in r];return ''.join(seq1(r.resname) for r in res),np.array([r['CA'].coord for r in res],dtype=float)
def align_metrics(pseq,px,rseq,rx):
 aligner=PairwiseAligner();aligner.mode='global';aligner.match_score=2;aligner.mismatch_score=-1;aligner.open_gap_score=-5;aligner.extend_gap_score=-.5;a=aligner.align(pseq,rseq)[0];ip=[];ir=[]
 for (s,e),(rs,re) in zip(a.aligned[0],a.aligned[1]):ip.extend(range(s,e));ir.extend(range(rs,re))
 x=px[ip];y=rx[ir];xc=x-x.mean(0);yc=y-y.mean(0);u,_,vt=np.linalg.svd(xc.T@yc);rot=u@vt
 if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vt
 rmsd=float(np.sqrt(np.mean(np.sum((xc@rot-yc)**2,axis=1))));dp=np.linalg.norm(x[:,None]-x[None,:],axis=-1);dr=np.linalg.norm(y[:,None]-y[None,:],axis=-1);mask=(dr<15)&(dr>0);dist=np.abs(dp-dr);lddt=float(np.mean(sum((dist[mask]<t).astype(float) for t in [.5,1,2,4])/4)) if mask.any() else None
 return dict(aligned_ca=len(ip),predicted_ca=len(px),reference_ca=len(rx),sequence_identity=float(np.mean([pseq[i]==rseq[j] for i,j in zip(ip,ir)])),ca_rmsd_angstrom=rmsd,ca_lddt=lddt,definition='Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT.')
for qid,q in qs.items():
 files=sorted((O/'predictions').glob(qid+'_unrelaxed_rank_*.pdb'));assert len(files)==5,(qid,len(files));scorefiles=sorted((O/'predictions').glob(qid+'_scores_rank_*.json'));assert len(scorefiles)==5
 for p in files:
  seq,xyz=ca_chain(p)
  if complex_mode:
   seqb,xyzb=ca_chain(p,'B');assert [seq,seqb]==q['chains'];seq+=seqb;xyz=np.concatenate([xyz,xyzb])
  assert seq==q['sequence'],(qid,len(seq),q['length']);assert np.isfinite(xyz).all();rank=p.name.split('_rank_')[1][:3];scorepath=next(s for s in scorefiles if '_rank_'+rank+'_' in s.name);score=json.loads(scorepath.read_text());pl=np.asarray(score['plddt']);assert len(pl)==q['length'] and np.isfinite(pl).all() and pl.min()>=0 and pl.max()<=100
  pae=np.asarray(score['pae']);assert pae.shape==(q['length'],q['length']) and np.isfinite(pae).all()
  row=dict(query=qid,rank=int(rank),pdb_path=str(p),scores_path=str(scorepath),residues=len(seq),mean_plddt=float(pl.mean()),ptm=float(score['ptm']),pdb_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),finite=True)
  if qid in ['1UBQ','1LYZ','1TIM']:
   rs,rx=ca_chain(R/'data'/(qid+'.pdb'));row['reference']=str(R/'data'/(qid+'.pdb'));row.update(align_metrics(seq,xyz,rs,rx))
  if complex_mode:
   ra,xa=ca_chain(R/'data/3G5O.pdb','A');rb,xb=ca_chain(R/'data/3G5O.pdb','B');pa,pxa=ca_chain(p,'A');pb,pxb=ca_chain(p,'B');row['chain_A']=align_metrics(pa,pxa,ra,xa);row['chain_B']=align_metrics(pb,pxb,rb,xb);row['reference_chain_mapping']={'predicted A':'3G5O A','predicted B':'3G5O B'};row['complex_alignment']=align_metrics(seq,xyz,ra+rb,np.concatenate([xa,xb]));row['iptm']=float(score['iptm'])
  results.append(row)
(O/'structure_validation.json').write_text(json.dumps(dict(passed=True,files_validated=len(results),results=results),indent=2));print('VALIDATED PDB+scores:',len(results))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/results/vmap_batch_20261028_185758_495006/source_worker.py — 원문 전문 (8,356 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/source_worker.py", "sha256": "c999c2ae00fa1fb9f9c3bdedbbc575a47b5f34b7c541e91e293e1d70bc7d5c64", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,time,json,threading,csv,subprocess,gc
import numpy as np,jax,jax.numpy as jnp
from colabfold.alphafold.models import load_models_and_params
from alphafold.common import protein
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];B=int(os.environ['BATCH']);C=D/f'batch_{B}';C.mkdir();dev=jax.devices('gpu')[0];stop=threading.Event();rows=[];phase='loading';G=2**30
S={'status':'running','batch':B,'implementation':'experimental jax.vmap over unmodified ColabFold RunModel.apply; shared fixed model_1 weights, independent actual query features and RNG; not a native CLI batch option','conditions':[]}
def write():(C/'summary.json').write_text(json.dumps(S,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def mem():
 info={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};rss=next(int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'));return {'rss_gib':rss/G,'available_gib':info['MemAvailable']/G,'total_gib':info['MemTotal']/G,'swap_gib':(info['SwapTotal']-info['SwapFree'])/G}
def dmem():return {k:int(v) for k,v in (dev.memory_stats() or {}).items()}
swap0=mem()['swap_gib']
def monitor():
 with (C/'resources.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['query_start','query_end','phase','gpu_util','rss_gib','available_gib','swap_gib','jax_bytes_in_use']);w.writeheader()
  while not stop.is_set():
   t=time.time();ph=phase
   try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip())
   except Exception:u=None
   m=mem();r={'query_start':t,'query_end':time.time(),'phase':ph,'gpu_util':u,'rss_gib':m['rss_gib'],'available_gib':m['available_gib'],'swap_gib':m['swap_gib'],'jax_bytes_in_use':dmem().get('bytes_in_use')};rows.append(r);w.writerow(r);f.flush()
   if m['available_gib']<16 or m['swap_gib']>swap0+2:(C/'memory_guard.json').write_text(json.dumps(r));os._exit(86)
   stop.wait(.5)
th=threading.Thread(target=monitor,daemon=True);th.start()
def aligned_rmsd(x,y):
 x=x-x.mean(0);y=y-y.mean(0);u,s,v=np.linalg.svd(x.T@y);q=np.eye(3);q[-1,-1]=np.linalg.det(u@v);return float(np.sqrt(np.mean(np.sum((x@(u@q@v)-y)**2,axis=1))))
try:
 audit=json.loads((D/'input_audit.json').read_text());queries=audit['queries'];allfeat=[dict(np.load(D/(q['id']+'_processed_features.npz'))) for q in queries];keys=set(allfeat[0]);assert all(set(f)==keys for f in allfeat);maxshape={k:tuple(max(f[k].shape[d] for f in allfeat) for d in range(allfeat[0][k].ndim)) for k in keys};features=[]
 for f in allfeat:
  features.append({k:np.pad(v,[(0,maxshape[k][d]-v.shape[d]) for d in range(v.ndim)],mode='constant') for k,v in f.items()})
 selected=features[:B];S['input']={'unique_queries':B,'query_ids':[q['id'] for q in queries[:B]],'actual_residues':[q['length'] for q in queries[:B]],'feature_shapes_per_query':{k:list(v.shape) for k,v in features[0].items()},'padding':'pool-wide max feature shapes; zero-padded extra-MSA/cluster entries masked by zero masks; no sequence duplication','actual_msa_valid_rows':[int((f['msa_mask'][0].sum(-1)>0).sum()) for f in selected]};t=time.time();loaded=load_models_and_params(num_models=1,use_templates=False,num_recycles=3,recycle_early_stop_tolerance=0.,num_ensemble=1,model_order=[1],model_type='alphafold2_ptm',data_dir=Path('/cache'),max_seq=512,max_extra_seq=5120,use_bfloat16=True,use_pallas=False,compile_mode='tuned');name,runner,params=loaded[0];runner.params=params;jax.block_until_ready(params);S['loading_seconds']=time.time()-t;S['parameters']=sum(int(x.size) for x in jax.tree_util.tree_leaves(params));S['weights_dtype']=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(params)});S['model_only_postload']={'jax':dmem(),'system':mem()};S['environment']={'jax':jax.__version__,'device':str(dev),'preallocate':os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),'compute':'original BF16 config, FP32 weights','recycles':3,'ensembles':1,'templates':False,'relaxation':False,'seed_per_query':42};write()
 full={k:jnp.asarray(np.stack([f[k] for f in selected])) for k in keys};L=int(selected[0]['aatype'].shape[1]);zero_prev={'prev_msa_first_row':jnp.zeros((B,L,256),jnp.float16),'prev_pair':jnp.zeros((B,L,L,128),jnp.float16),'prev_pos':jnp.zeros((B,L,37,3),jnp.float16)};rng=jnp.stack([jax.random.PRNGKey(42)]*B);first={k:v[:,0:1] for k,v in full.items()};first['prev']=zero_prev;vmapped=jax.jit(jax.vmap(runner.apply,in_axes=(None,0,0)));phase='compilation';t=time.time();compiled=vmapped.lower(params,rng,first).compile();S['compilation_seconds']=time.time()-t;S['compiled_memory_analysis']=str(compiled.memory_analysis());write()
 def predict():
  prev=zero_prev;key=jnp.stack([jax.random.PRNGKey(42)]*B)
  for recycle in range(4):
   pair=jax.vmap(jax.random.split)(key);key=pair[:,0];subkey=pair[:,1];subfeat={k:v[:,recycle:recycle+1] for k,v in full.items()};subfeat['prev']=prev;result=compiled(params,subkey,subfeat);jax.block_until_ready(result);result=jax.tree_util.tree_map(lambda x:np.asarray(x,np.float16),result);prev={k:jnp.asarray(v) for k,v in result.pop('prev').items()}
  return result
 phase='warmup';t=time.time();result=predict();S['warmup_seconds']=time.time()-t;phase='execution';t=time.time();repeats=0
 while repeats<2 or time.time()-t<15:result=predict();repeats+=1
 end=time.time();stats=dmem();observed=[r for r in rows if r['query_start']>=t and r['query_end']<=end];utils=[r['gpu_util'] for r in observed if r['gpu_util'] is not None];S['conditions']=[{'batch':B,'unique_queries':B,'repeats':repeats,'processed_queries':B*repeats,'seconds':end-t,'queries_per_second':B*repeats/(end-t),'gpu_util_avg':float(np.mean(utils)) if utils else None,'gpu_util_max':max(utils) if utils else None,'sample_count':len(utils),'jax_memory_stats':stats,'memory_definition':'peak_bytes_in_use is cumulative process allocator peak including compilation and warmup; execution bytes_in_use samples separate, no unsupported reset','execution_sampled_peak_bytes_in_use':max([r['jax_bytes_in_use'] for r in observed if r['jax_bytes_in_use'] is not None],default=None),'rss_sampled_peak_gib':max([r['rss_gib'] for r in observed],default=None),'available_sampled_min_gib':min([r['available_gib'] for r in observed],default=None)}];phase='validation';preds=[]
 for i,q in enumerate(queries[:B]):
  single=jax.tree_util.tree_map(lambda x:x[i],result);coords=single['structure_module']['final_atom_positions'];plddt=single['plddt'];assert coords.shape==(L,37,3) and np.isfinite(coords).all() and np.isfinite(plddt).all();np.savez(C/(q['id']+'_prediction.npz'),atom_positions=coords,plddt=plddt);prot=protein.from_prediction(features=selected[i],result=single,b_factors=np.repeat(plddt[:,None],37,axis=1));pdb=protein.to_pdb(prot);(C/(q['id']+'.pdb')).write_text(pdb);assert sum(l.startswith('ATOM') and l[12:16].strip()=='CA' for l in pdb.splitlines())==q['length'];preds.append({'id':q['id'],'residues':q['length'],'mean_plddt':float(plddt.mean()),'finite_and_pdb_valid':True})
 S['predictions']=preds;reference=D/'batch_1'/ (queries[0]['id']+'_prediction.npz')
 if B==1:
  phase='single_native_reference';t0=time.time();ref,recycles=runner.predict(selected[0],random_seed=42);S['native_reference_seconds_excluded']=time.time()-t0;refcoords=ref['structure_module']['final_atom_positions'];S['native_reference_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(refcoords[:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max()),'native_recycles':int(recycles)};assert S['native_reference_comparison']['ca_aligned_rmsd_angstrom']<.5;np.savez(C/'native_single_reference.npz',atom_positions=refcoords,plddt=ref['plddt'])
 else:
  ref=np.load(reference);S['first_query_batch_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(ref['atom_positions'][:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max())}
 S['status']='complete';write();print('AF2_VMAP_COMPLETE',B,flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write();raise
finally:stop.set();th.join(timeout=5)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/results/environment_verified/colabfold_models.py — 원문 전문 (9,160 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/environment_verified/colabfold_models.py", "sha256": "bd637765535c7a199d76631874264e20e2d496e332251fb1d16e144542c03894", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
from functools import wraps, partialmethod
from typing import Tuple, List, Optional
import haiku
from alphafold.model import model, config, data
from alphafold.model.modules import AlphaFold
from alphafold.model.modules_multimer import AlphaFold as AlphaFoldMultimer

def _compile_jit_kwargs(compile_mode: str) -> dict:
    import jaxlib
    from packaging.version import Version
    # These params won across GB10/L40S/A100, however exact tile is not very important anyway
    if Version(jaxlib.__version__) < Version("0.10.1"):
        # jaxlib < 0.10 checks for split_k, now deprecated
        tuned = "block_m: 64 block_n: 64 block_k: 16 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1"
    else:
        tuned = "block_m: 64 block_n: 64 block_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1"
    copts = {
        "fast":  {"xla_gpu_enable_triton_gemm": False},
        "tuned": {"xla_gpu_override_gemm_autotuner": tuned},
        "full":  None,
    }.get(compile_mode)
    return {"compiler_options": copts} if copts else {}


def get_model_haiku_params(
    data_dir: str,
    model_type: str,
    model_number: str,
    use_fuse: bool = True,
    to_jnp: bool = True,
) -> haiku.Params:
    """Get the Haiku parameters from a model type and number."""
    import os
    import numpy as np
    from alphafold.model import utils
    import jax.numpy as jnp

    is_deepfold = False
    if model_type == "alphafold2_multimer_v1":
        file = f"params_model_{model_number}_multimer.npz"
    elif model_type == "alphafold2_multimer_v2":
        file = f"params_model_{model_number}_multimer_v2.npz"
    elif model_type == "alphafold2_multimer_v3":
        file = f"params_model_{model_number}_multimer_v3.npz"
    elif model_type == "alphafold2_ptm":
        file = f"params_model_{model_number}_ptm.npz"
    elif model_type == "alphafold2":
        file = f"params_model_{model_number}.npz"
    elif model_type == "deepfold_v1":
        file = f"deepfold_model_{model_number}.npz"
        is_deepfold = True
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    path = os.path.join(data_dir, "params", file)
    params = np.load(path, allow_pickle=False)
    return utils.flat_params_to_haiku(params, fuse=use_fuse, to_jnp=to_jnp)


def model_to_config_name(model_type: str, model_number: str) -> str:
    if model_type == "alphafold2_multimer_v1":
        return f"model_{model_number}_multimer"
    elif model_type == "alphafold2_multimer_v2":
        return f"model_{model_number}_multimer_v2"
    elif model_type == "alphafold2_multimer_v3":
        return f"model_{model_number}_multimer_v3"
    elif model_type == "alphafold2_ptm":
        return f"model_{model_number}_ptm"
    elif model_type == "alphafold2":
        return f"model_{model_number}"
    elif model_type == "deepfold_v1":
        return f"model_{model_number}"
    else:
        raise ValueError(f"Unknown model_type {model_type}")


def load_models_and_params(
    num_models: int,
    use_templates: bool,
    num_recycles: Optional[int] = None,
    recycle_early_stop_tolerance: Optional[float] = None,
    num_ensemble: int = 1,
    model_order: Optional[List[int]] = None,
    model_type: str = "",
    data_dir: Path = Path("."),
    stop_at_score: float = 100,
    rank_by: str = "auto",
    max_seq: Optional[int] = None,
    max_extra_seq: Optional[int] = None,
    use_cluster_profile: bool = True,
    use_fuse: bool = True,
    use_bfloat16: bool = True,
    use_dropout: bool = False,
    save_all: bool = False,
    calc_extra_ptm: bool = False,
    use_probs_extra: bool = True,
    use_pallas: bool = False,
    compile_mode: str = "tuned"
) -> List[Tuple[str, model.RunModel, haiku.Params]]:
    """We use only two actual models and swap the parameters to avoid recompiling.

    Note that models 1 and 2 have a different number of parameters compared to models 3, 4 and 5,
    so we load model 1 and model 3.
    """

    # Use only two model and later swap params to avoid recompiling
    model_runner_and_params: [Tuple[str, model.RunModel, haiku.Params]] = []

    if model_order is None:
        model_order = [1, 2, 3, 4, 5]
    else:
        model_order.sort()

    model_build_order = [3, 4, 5, 1, 2]
    if "multimer" in model_type:
        models_need_compilation = [3]
    else:
        # only models 1,2 use templates
        models_need_compilation = [1, 3] if use_templates else [3]
    
    model_runner_and_params_build_order: [Tuple[str, model.RunModel, haiku.Params]] = []
    model_runner = None
    for model_number in model_build_order:
        if model_number in models_need_compilation:
            # get configurations
            config_name = model_to_config_name(model_type, model_number)
            model_config = config.model_config(config_name)
            model_config.model.stop_at_score = float(stop_at_score)
            model_config.model.rank_by = rank_by

            # set dropouts
            model_config.model.global_config.eval_dropout = use_dropout

            # set bfloat options
            model_config.model.global_config.bfloat16 = use_bfloat16

            # cuEquivariance fused kernels
            model_config.model.global_config.use_pallas = use_pallas

            # set fuse options
            model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.fuse_projection_weights = use_fuse
            model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.fuse_projection_weights = use_fuse
            if "multimer" in config_name or model_number in [1,2]:
                model_config.model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_incoming.fuse_projection_weights = use_fuse
                model_config.model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_outgoing.fuse_projection_weights = use_fuse
                        
            # set number of sequences options
            if max_seq is not None:
                if "multimer" in config_name:
                    model_config.model.embeddings_and_evoformer.num_msa = max_seq
                else:
                    model_config.data.eval.max_msa_clusters = max_seq
            
            if max_extra_seq is not None:
                if "multimer" in config_name:
                    model_config.model.embeddings_and_evoformer.num_extra_msa = max_extra_seq
                else:
                    model_config.data.common.max_extra_msa = max_extra_seq

            # disable some outputs if not being saved
            if not save_all:
                if not calc_extra_ptm:
                    model_config.model.heads.distogram.weight = 0.0
                model_config.model.heads.masked_msa.weight = 0.0
                model_config.model.heads.experimentally_resolved.weight = 0.0

            # set number of recycles and ensembles            
            if "multimer" in config_name:
                if num_recycles is not None:
                    model_config.model.num_recycle = num_recycles
                model_config.model.embeddings_and_evoformer.use_cluster_profile = use_cluster_profile
                model_config.model.num_ensemble_eval = num_ensemble
            else:
                if num_recycles is not None:
                    model_config.data.common.num_recycle = num_recycles
                    model_config.model.num_recycle = num_recycles
                model_config.data.eval.num_ensemble = num_ensemble

            if recycle_early_stop_tolerance is not None:
                model_config.model.recycle_early_stop_tolerance = recycle_early_stop_tolerance
            
            # get model runner
            params = get_model_haiku_params(
                model_type=model_type,
                model_number=model_number,
                data_dir=str(data_dir),
                use_fuse=use_fuse
            )

            model_runner = model.RunModel(
                model_config,
                params,
                extended_ptm_config={'calc_extended_ptm': calc_extra_ptm,
                                     'use_probs_extended': use_probs_extra},
                jit_kwargs=_compile_jit_kwargs(compile_mode)
            )
        
        params = get_model_haiku_params(
            model_type=model_type,
            model_number=model_number,
            data_dir=str(data_dir),
            use_fuse=use_fuse,
        )
        # keep only parameters of compiled model
        params_subset = {}
        for k in model_runner.params.keys():
            params_subset[k] = params[k]

        model_name = f"model_{model_number}"
        model_runner_and_params_build_order.append(
            (model_name, model_runner, params_subset)
        )
    # reorder model
    for n, model_number in enumerate(model_order):
        if n == num_models:
            break
        model_name = f"model_{model_number}"
        for m in model_runner_and_params_build_order:
            if model_name == m[0]:
                model_runner_and_params.append(m)
                break
    return model_runner_and_params
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/results/environment_verified/alphafold_model.py — 원문 전문 (7,437 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/environment_verified/alphafold_model.py", "sha256": "61e0d1ebc73e644defb9f649cfbdedab21b839829b41cf9f8102d2f53f3092df", "encoding": "utf-8", "added_final_newline": false} -->
```python
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for constructing the model."""
from typing import Any, Mapping, Optional, Union

from absl import logging
from alphafold.common import confidence
from alphafold.model import features
from alphafold.model import modules
from alphafold.model import modules_multimer
from alphafold.common import residue_constants
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from alphafold.model.tf import np_backend as tf
import tree

class RunModel:
  """Container for JAX model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               params: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
               is_training = False, extended_ptm_config=None, jit_kwargs=None):
    
    self.config = config
    self.params = params
    self.multimer_mode = config.model.global_config.multimer_mode
    self.config.model.calc_extended_ptm = extended_ptm_config['calc_extended_ptm'] if extended_ptm_config else False
    self.config.model.use_probs_extended = extended_ptm_config['use_probs_extended'] if extended_ptm_config else False

    if self.multimer_mode:
      def _forward_fn(batch):
        model = modules_multimer.AlphaFold(self.config.model)
        return model(batch, is_training=is_training)
    else:
      def _forward_fn(batch):
        if self.config.data.eval.num_ensemble == 1:
          model = modules.AlphaFold_noE(self.config.model)
          return model(batch, is_training=is_training)
        else:
          model = modules.AlphaFold(self.config.model)
          return model(
              batch,
              is_training=is_training,
              compute_loss=False,
              ensemble_representations=True)

    jit_kwargs = jit_kwargs or {}
    self.apply = jax.jit(hk.transform(_forward_fn).apply, **jit_kwargs)
    self.init = jax.jit(hk.transform(_forward_fn).init, **jit_kwargs)

  def init_params(self, feat: features.FeatureDict, random_seed: int = 0):
    """Initializes the model parameters.

    If none were provided when this class was instantiated then the parameters
    are randomly initialized.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.
      random_seed: A random seed to use to initialize the parameters if none
        were set when this class was initialized.
    """
    if not self.params:
      # Init params randomly.
      rng = jax.random.PRNGKey(random_seed)
      self.params = hk.data_structures.to_mutable_dict(
          self.init(rng, feat))
      logging.warning('Initialized parameters randomly')

  def process_features(
      self,
      raw_features: Union[tf.train.Example, features.FeatureDict],
      random_seed: int) -> features.FeatureDict:
    """Processes features to prepare for feeding them into the model.

    Args:
      raw_features: The output of the data pipeline either as a dict of NumPy
        arrays or as a tf.train.Example.
      random_seed: The random seed to use when processing the features.

    Returns:
      A dict of NumPy feature arrays suitable for feeding into the model.
    """

    if self.multimer_mode:
      return raw_features

    # Single-chain mode.
    if isinstance(raw_features, dict):
      return features.np_example_to_features(
          np_example=raw_features,
          config=self.config,
          random_seed=random_seed)
    else:
      return features.tf_example_to_features(
          tf_example=raw_features,
          config=self.config,
          random_seed=random_seed)

  def eval_shape(self, feat: features.FeatureDict) -> jax.ShapeDtypeStruct:
    self.init_params(feat)
    logging.debug('Running eval_shape with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))
    shape = jax.eval_shape(self.apply, self.params, jax.random.PRNGKey(0), feat)
    logging.info('Output shape was %s', shape)
    return shape

  def predict(self,
              feat: features.FeatureDict,
              random_seed: int = 0,
              return_representations: bool = False,
              callback: Any = None) -> Mapping[str, Any]:
    """Makes a prediction by inferencing the model on the provided features.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.
      random_seed: The random seed to use when running the model. In the
        multimer model this controls the MSA sampling.

    Returns:
      A dictionary of model outputs.
    """
    self.init_params(feat)
    logging.info('Running predict with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))
    
    # get shapes
    aatype = feat["aatype"]
    num_iters = self.config.model.num_recycle + 1
    if self.multimer_mode:
      L = aatype.shape[0]
    else:
      num_ensemble = self.config.data.eval.num_ensemble
      L = aatype.shape[1]
    
    # initialize
    zeros = lambda shape: np.zeros(shape, dtype=np.float16)
    prev = {'prev_msa_first_row': zeros([L,256]),
            'prev_pair':          zeros([L,L,128])}
    
    # initial guess
    if "all_atom_positions" in feat:
      logging.info("INFO: using provided all_atom_positions as initial guess")
      prev["prev_pos"] = feat["all_atom_positions"]
    else:
      prev["prev_pos"] = zeros([L,37,3])
    
    def run(key, feat, prev):
      def _jnp_to_np(x):
        for k, v in x.items():
          if isinstance(v, dict):
            x[k] = _jnp_to_np(v)
          else:
            x[k] = np.asarray(v,np.float16)
        return x
      result = _jnp_to_np(self.apply(self.params, key, {**feat, "prev":prev}))
      prev = result.pop("prev")
      return result, prev


    # initialize random key
    key = jax.random.PRNGKey(random_seed)
    
    # iterate through recycles
    for r in range(num_iters):      
        # grab subset of features
        if self.multimer_mode:
          sub_feat = feat
        else:
          s = r * num_ensemble
          e = (r+1) * num_ensemble
          sub_feat = jax.tree_util.tree_map(lambda x:x[s:e], feat)
            
        # run
        key, sub_key = jax.random.split(key)
        result, prev = run(sub_key, sub_feat, prev)
        
        if return_representations:

          result["representations"] = {"pair":   prev["prev_pair"],
                                       "single": prev["prev_msa_first_row"]}
                                       
        # callback
        if callback is not None: callback(result, r)

        # decide when to stop
        if result["ranking_confidence"] > self.config.model.stop_at_score:
          break
        if r > 0 and result["tol"] < self.config.model.recycle_early_stop_tolerance:
          break
    logging.info('Output shape was %s', tree.map_structure(lambda x: x.shape, result))
    return result, r
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/results/environment_verified/batch.py — 원문 전문 (91,330 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/environment_verified/batch.py", "sha256": "1001e5685ce9b104cefbe0391cf803815d0788f90d47387883b6991338bebcd7", "encoding": "utf-8", "added_final_newline": false} -->
```python
from __future__ import annotations

import os
ENV = {"TF_FORCE_UNIFIED_MEMORY":"1", "XLA_PYTHON_CLIENT_MEM_FRACTION":"4.0"}
for k,v in ENV.items():
    if k not in os.environ: os.environ[k] = v

import warnings
from Bio import BiopythonDeprecationWarning # what can possibly go wrong...
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)

import json
hasOrjson = False
try:
    import orjson
    hasOrjson = True
except ImportError:
    pass
import logging
import math
import sys
import time
import zipfile
import shutil
import pickle
import gzip

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from io import StringIO

import importlib_metadata
import numpy as np

try:
    import alphafold
except ModuleNotFoundError:
    raise RuntimeError(
        "\n\nalphafold is not installed. Please run `pip install colabfold[alphafold]`\n"
    )

from alphafold.common import protein, residue_constants

# delay imports of jax and numpy
# loading these for type checking only can take around 10 seconds just to show a CLI usage message
if TYPE_CHECKING:
    import haiku
    from alphafold.model import model
    from numpy import ndarray

from alphafold.common.protein import Protein
from alphafold.data import (
    feature_processing,
    msa_pairing,
    pipeline,
    pipeline_multimer,
    templates,
)
from alphafold.data.tools import hhsearch
from colabfold.citations import write_bibtex
from colabfold.download import default_data_dir, download_alphafold_params
from colabfold.utils import (
    ACCEPT_DEFAULT_TERMS,
    DEFAULT_API_SERVER,
    NO_GPU_FOUND,
    CIF_REVISION_DATE,
    get_commit,
    setup_logging,
    CFMMCIFIO,
    AF3Utils,
)
from colabfold.input import (
    pair_msa,
    msa_to_str,
    get_queries,
    safe_filename,
    modified_mapping,
    pdb_to_string,
)
from colabfold.relax import relax_me
from colabfold.alphafold import extra_ptm

from Bio.PDB import MMCIFParser, PDBParser, MMCIF2Dict
from Bio.PDB.PDBIO import Select

# logging settings
logger = logging.getLogger(__name__)

def _str2bool(v):
    """argparse type for an optional-value bool flag (--flag / --flag true/false)."""
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("1", "true", "yes", "on", "t"):
        return True
    if s in ("0", "false", "no", "off", "f"):
        return False
    from argparse import ArgumentTypeError
    raise ArgumentTypeError(f"expected true/false, got {v!r}")

from jax import local_devices

# from jax 0.4.6, jax._src.lib.xla_bridge moved to jax._src.xla_bridge
# suppress warnings: Unable to initialize backend 'rocm' or 'tpu'
logging.getLogger('jax._src.xla_bridge').addFilter(lambda _: False) # jax >=0.4.6
logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False) # jax < 0.4.5

def mk_mock_template(
    query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(
            output_confidence_scores[None], [num_temp, 1]
        ),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_release_date": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features

def mk_template(
    a3m_lines: str,
    template_path: str,
    query_sequence: str,
    max_template_date="2100-01-01",
    max_hits=20,
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_path,
        max_template_date=max_template_date,
        max_hits=max_hits,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/pdb70"]
    )

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)

def validate_and_fix_mmcif(cif_file: Path):
    """validate presence of _entity_poly_seq in cif file and add revision_date if missing"""
    # check that required poly_seq and revision_date fields are present
    cif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)
    required = [
        "_chem_comp.id",
        "_chem_comp.type",
        "_struct_asym.id",
        "_struct_asym.entity_id",
        "_entity_poly_seq.mon_id",
    ]
    for r in required:
        if r not in cif_dict:
            raise ValueError(f"mmCIF file {cif_file} is missing required field {r}.")
    if "_pdbx_audit_revision_history.revision_date" not in cif_dict:
        logger.info(
            f"Adding missing field revision_date to {cif_file}. Backing up original file to {cif_file}.bak."
        )
        shutil.copy2(cif_file, str(cif_file) + ".bak")
        with open(cif_file, "a") as f:
            f.write(CIF_REVISION_DATE)

class ReplaceOrRemoveHetatmSelect(Select):
  def accept_residue(self, residue):
    hetfield, _, _ = residue.get_id()
    if hetfield != " ":
      if residue.resname in modified_mapping:
        # set unmodified resname
        residue.resname = modified_mapping[residue.resname]
        # clear hetatm flag
        residue._id = (" ", residue._id[1], " ")
        t = residue.full_id
        residue.full_id = (t[0], t[1], t[2], residue._id)
        return 1
      return 0
    else:
      return 1

def convert_pdb_to_mmcif(pdb_file: Path):
    """convert existing pdb files into mmcif with the required poly_seq and revision_date"""
    i = pdb_file.stem
    cif_file = pdb_file.parent.joinpath(f"{i}.cif")
    if cif_file.is_file():
        return
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(i, pdb_file)
    cif_io = CFMMCIFIO()
    cif_io.set_structure(structure)
    cif_io.save(str(cif_file), ReplaceOrRemoveHetatmSelect())

def mk_hhsearch_single_entry_db(cif_file: Path, dbdir_cache_path: str):
    dbdir = Path(dbdir_cache_path)
    dbdir.mkdir(parents=True, exist_ok=True)
    tmp_cif_path = str(dbdir_cache_path) + "/1dmy.cif"
    shutil.copy2(cif_file, tmp_cif_path)
    # clear internal AF2 cache of mmCIF files
    templates._read_file.cache_clear()
    cif_file = Path(tmp_cif_path)
    pdb70_db_files = dbdir.glob("pdb70*")
    for f in pdb70_db_files:
        os.remove(f)

    with open(dbdir.joinpath("pdb70_a3m.ffdata"), "w") as a3m, open(
        dbdir.joinpath("pdb70_cs219.ffindex"), "w"
    ) as cs219_index, open(
        dbdir.joinpath("pdb70_a3m.ffindex"), "w"
    ) as a3m_index, open(
        dbdir.joinpath("pdb70_cs219.ffdata"), "w"
    ) as cs219:
        n = 1000000
        index_offset = 0
        with open(cif_file) as f:
            cif_string = f.read()
        cif_fh = StringIO(cif_string)
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("none", cif_fh)
        models = list(structure.get_models())
        if len(models) != 1:
            logger.warning(f"WARNING: Found {len(models)} models in {cif_file}. The first model will be used as a template.", )
            # raise ValueError(
            #     f"Only single model PDBs are supported. Found {len(models)} models in {cif_file}."
            # )
        model = models[0]
        for chain in model:
            amino_acid_res = []
            for res in chain:
                if res.id[2] != " ":
                    logger.warning(f"WARNING: Found insertion code at chain {chain.id} and residue index {res.id[1]} of {cif_file}. "
                                    "This file cannot be used as a template.")
                    continue
                    # raise ValueError(
                    #     f"PDB {cif_file} contains an insertion code at chain {chain.id} and residue "
                    #     f"index {res.id[1]}. These are not supported."
                    # )
                amino_acid_res.append(
                    residue_constants.restype_3to1.get(res.resname, "X")
                )

            protein_str = "".join(amino_acid_res)
            a3m_str = f">{cif_file.stem}_{chain.id}\n{protein_str}\n\0"
            a3m_str_len = len(a3m_str)
            a3m_index.write(f"{n}\t{index_offset}\t{a3m_str_len}\n")
            cs219_index.write(f"{n}\t{index_offset}\t{len(protein_str)}\n")
            index_offset += a3m_str_len
            a3m.write(a3m_str)
            cs219.write("\n\0")
            n += 1

def mk_hhsearch_db(template_dir: str):
    template_path = Path(template_dir)

    cif_files = template_path.glob("*.cif")
    for cif_file in cif_files:
        validate_and_fix_mmcif(cif_file)

    pdb_files = template_path.glob("*.pdb")
    for pdb_file in pdb_files:
        convert_pdb_to_mmcif(pdb_file)

    pdb70_db_files = template_path.glob("pdb70*")
    for f in pdb70_db_files:
        os.remove(f)

    with open(template_path.joinpath("pdb70_a3m.ffdata"), "w") as a3m, open(
        template_path.joinpath("pdb70_cs219.ffindex"), "w"
    ) as cs219_index, open(
        template_path.joinpath("pdb70_a3m.ffindex"), "w"
    ) as a3m_index, open(
        template_path.joinpath("pdb70_cs219.ffdata"), "w"
    ) as cs219:
        n = 1000000
        index_offset = 0
        cif_files = template_path.glob("*.cif")
        for cif_file in cif_files:
            with open(cif_file) as f:
                cif_string = f.read()
            cif_fh = StringIO(cif_string)
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("none", cif_fh)
            models = list(structure.get_models())
            if len(models) != 1:
                logger.warning(f"WARNING: Found {len(models)} models in {cif_file}. The first model will be used as a template.", )
                # raise ValueError(
                #     f"Only single model PDBs are supported. Found {len(models)} models in {cif_file}."
                # )
            model = models[0]
            for chain in model:
                amino_acid_res = []
                for res in chain:
                    if res.id[2] != " ":
                        logger.warning(f"WARNING: Found insertion code at chain {chain.id} and residue index {res.id[1]} of {cif_file}. "
                                       "This file cannot be used as a template.")
                        continue
                        # raise ValueError(
                        #     f"PDB {cif_file} contains an insertion code at chain {chain.id} and residue "
                        #     f"index {res.id[1]}. These are not supported."
                        # )
                    amino_acid_res.append(
                        residue_constants.restype_3to1.get(res.resname, "X")
                    )

                protein_str = "".join(amino_acid_res)
                a3m_str = f">{cif_file.stem}_{chain.id}\n{protein_str}\n\0"
                a3m_str_len = len(a3m_str)
                a3m_index.write(f"{n}\t{index_offset}\t{a3m_str_len}\n")
                cs219_index.write(f"{n}\t{index_offset}\t{len(protein_str)}\n")
                index_offset += a3m_str_len
                a3m.write(a3m_str)
                cs219.write("\n\0")
                n += 1

def pad_input(
    input_features: model.features.FeatureDict,
    model_runner: model.RunModel,
    model_name: str,
    pad_len: int,
    use_templates: bool,
) -> model.features.FeatureDict:
    from colabfold.alphafold.msa import make_fixed_size

    model_config = model_runner.config
    eval_cfg = model_config.data.eval
    crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

    max_msa_clusters = eval_cfg.max_msa_clusters
    max_extra_msa = model_config.data.common.max_extra_msa
    # templates models
    if (model_name == "model_1" or model_name == "model_2") and use_templates:
        pad_msa_clusters = max_msa_clusters - eval_cfg.max_templates
    else:
        pad_msa_clusters = max_msa_clusters

    max_msa_clusters = pad_msa_clusters

    # let's try pad (num_res + X)
    input_fix = make_fixed_size(
        input_features,
        crop_feats,
        msa_cluster_size=max_msa_clusters,  # true_msa (4, 512, 68)
        extra_msa_size=max_extra_msa,  # extra_msa (4, 5120, 68)
        num_res=pad_len,  # aatype (4, 68)
        num_templates=4,
    )  # template_mask (4, 4) second value
    return input_fix

class file_manager:
    def __init__(self, prefix: str, result_dir: Path):
        self.prefix = prefix
        self.result_dir = result_dir
        self.tag = None
        self.files = {}

    def get(self, x: str, ext:str) -> Path:
        if self.tag not in self.files:
            self.files[self.tag] = []
        file = self.result_dir.joinpath(f"{self.prefix}_{x}_{self.tag}.{ext}")
        self.files[self.tag].append([x,ext,file])
        return file

    def set_tag(self, tag):
        self.tag = tag

def predict_structure(
    prefix: str,
    result_dir: Path,
    feature_dict: Dict[str, Any],
    is_complex: bool,
    use_templates: bool,
    sequences_lengths: List[int],
    pad_len: int,
    model_type: str,
    model_runner_and_params: List[Tuple[str, model.RunModel, haiku.Params]],
    initial_guess: str = None,
    msa_pad_depth: int = 0,
    num_relax: int = 0,
    relax_max_iterations: int = 0,
    relax_tolerance: float = 2.39,
    relax_stiffness: float = 10.0,
    relax_max_outer_iterations: int = 3,
    rank_by: str = "auto",
    random_seed: int = 0,
    num_seeds: int = 1,
    stop_at_score: float = 100,
    prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
    use_gpu_relax: bool = False,
    save_all: bool = False,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    save_recycles: bool = False,
    calc_extra_ptm: bool = False,
    use_probs_extra: bool = True,
):
    """Predicts structure using AlphaFold for the given sequence."""
    mean_scores = []
    conf = []
    unrelaxed_pdb_lines = []
    prediction_times = []
    model_names = []
    files = file_manager(prefix, result_dir)
    seq_len = sum(sequences_lengths)

    # iterate through random seeds
    for seed_num, seed in enumerate(range(random_seed, random_seed+num_seeds)):

        # iterate through models
        for model_num, (model_name, model_runner, params) in enumerate(model_runner_and_params):

            # swap params to avoid recompiling
            model_runner.params = params

            #########################
            # process input features
            #########################
            if "multimer" in model_type:
                if model_num == 0 and seed_num == 0:
                    # TODO: add pad_input_mulitmer()
                    input_features = feature_dict
                    input_features["asym_id"] = input_features["asym_id"] - input_features["asym_id"][...,0]
                    # Pad MSA depth up to msa_pad_depth to share one compiled shape
                    # Padded rows are masked (msa_mask=0), so the prediction is unchanged
                    if msa_pad_depth > input_features["msa"].shape[0]:
                        input_features = pipeline_multimer.pad_msa(input_features, min_num_seq=msa_pad_depth)
            else:
                if model_num == 0:
                    input_features = model_runner.process_features(feature_dict, random_seed=seed)
                    r = input_features["aatype"].shape[0]
                    input_features["asym_id"] = np.tile(feature_dict["asym_id"],r).reshape(r,-1)
                    if seq_len < pad_len:
                        input_features = pad_input(input_features, model_runner,
                            model_name, pad_len, use_templates)
                        logger.info(f"Padding length to {pad_len}")


            tag = f"{model_type}_{model_name}_seed_{seed:03d}"
            model_names.append(tag)
            files.set_tag(tag)

            # initial guess
            if initial_guess:
                input_guess = Path(initial_guess)
                if input_guess.suffix == ".pdb":
                    pdb_string = pdb_to_string(initial_guess)
                    input_features["all_atom_positions"] = protein.from_pdb_string(pdb_string).atom_positions
                elif input_guess.suffix == ".cif":
                    input_features["all_atom_positions"] = protein.from_mmcif_string(input_guess.read_text()).atom_positions
                else:
                    raise ValueError(f"Unsupported initial guess file format: {initial_guess}")
                

            ########################
            # predict
            ########################
            start = time.time()

            # monitor intermediate results
            def callback(result, recycles):
                if recycles == 0: result.pop("tol",None)
                if not is_complex: result.pop("iptm",None)
                print_line = ""
                for x,y in [["mean_plddt","pLDDT"],["ptm","pTM"],["iptm","ipTM"],["tol","tol"]]:
                  if x in result:
                    print_line += f" {y}={result[x]:.3g}"
                logger.info(f"{tag} recycle={recycles}{print_line}")

                if save_recycles:
                    final_atom_mask = result["structure_module"]["final_atom_mask"]
                    b_factors = result["plddt"][:, None] * final_atom_mask
                    unrelaxed_protein = protein.from_prediction(
                        features=input_features,
                        result=result, b_factors=b_factors,
                        remove_leading_feature_dimension=("multimer" not in model_type))
                    files.get("unrelaxed",f"r{recycles}.pdb").write_text(protein.to_pdb(unrelaxed_protein))

                    if save_all:
                        with files.get("all",f"r{recycles}.pickle").open("wb") as handle:
                            pickle.dump(result, handle)
                    del unrelaxed_protein

            return_representations = save_all or save_single_representations or save_pair_representations

            # predict
            result, recycles = \
            model_runner.predict(input_features,
                random_seed=seed,
                return_representations=return_representations,
                callback=callback)

            if calc_extra_ptm and 'predicted_aligned_error' in result.keys():
                extra_ptm_output = extra_ptm.get_chain_and_interface_metrics(result, input_features['asym_id'],
                    use_probs_extra=use_probs_extra,
                    use_jnp=False)
                result.pop('pae_matrix_with_logits', None)
                result['actifptm'] = extra_ptm_output['actifptm']
            else:
                calc_extra_ptm = False
            prediction_times.append(time.time() - start)

            ########################
            # parse results
            ########################

            # summary metrics
            mean_scores.append(result["ranking_confidence"])
            if recycles == 0: result.pop("tol",None)
            if not is_complex: result.pop("iptm",None)
            print_line = ""
            conf.append({})
            for x,y in [["mean_plddt","pLDDT"],["ptm","pTM"],["iptm","ipTM"], ['actifptm', 'actifpTM']]:
              if x in result:
                print_line += f" {y}={result[x]:.3g}"
                conf[-1][x] = float(result[x])
            conf[-1]["print_line"] = print_line
            logger.info(f"{tag} took {prediction_times[-1]:.1f}s ({recycles} recycles)")

            # create protein object
            final_atom_mask = result["structure_module"]["final_atom_mask"]
            b_factors = result["plddt"][:, None] * final_atom_mask
            unrelaxed_protein = protein.from_prediction(
                features=input_features,
                result=result,
                b_factors=b_factors,
                remove_leading_feature_dimension=("multimer" not in model_type))

            # callback for visualization
            if prediction_callback is not None:
                prediction_callback(unrelaxed_protein, sequences_lengths,
                                    result, input_features, (tag, False))

            #########################
            # save results
            #########################

            # save pdb
            protein_lines = protein.to_pdb(unrelaxed_protein)
            files.get("unrelaxed","pdb").write_text(protein_lines)
            unrelaxed_pdb_lines.append(protein_lines)

            # save raw outputs
            if save_all:
                with files.get("all","pickle").open("wb") as handle:
                    pickle.dump(result, handle)
            if save_single_representations:
                np.save(files.get("single_repr","npy"),result["representations"]["single"])
            if save_pair_representations:
                np.save(files.get("pair_repr","npy"),result["representations"]["pair"])

            # write an easy-to-use format (pAE and pLDDT)
            plddt = result["plddt"][:seq_len]
            scores = {"plddt": np.around(plddt.astype(float), 2).tolist()}
            if "predicted_aligned_error" in result:
                pae = result["predicted_aligned_error"][:seq_len,:seq_len]
                scores.update({"max_pae": pae.max().astype(float).item(),
                                "pae": np.around(pae.astype(float), 2).tolist()})
                if calc_extra_ptm:
                    scores.update(extra_ptm_output)
                for k in ["ptm", "iptm"]:
                    if k in conf[-1]:
                        scores[k] = np.around(conf[-1][k], 2).item()
                del pae
            del plddt
            file = files.get("scores", "json")
            if hasOrjson:
                file.write_bytes(orjson.dumps(scores))
            else:
                file.write_text(json.dumps(scores))

            del result, unrelaxed_protein

            # early stop criteria fulfilled
            if mean_scores[-1] > stop_at_score: break

        # early stop criteria fulfilled
        if mean_scores[-1] > stop_at_score: break

        # cleanup
        if "multimer" not in model_type: del input_features
    if "multimer" in model_type: del input_features

    ###################################################
    # rerank models based on predicted confidence
    ###################################################

    rank, metric = [],[]
    result_files = []
    logger.info(f"reranking models by '{rank_by}' metric")
    model_rank = np.array(mean_scores).argsort()[::-1]
    for n, key in enumerate(model_rank):
        metric.append(conf[key])
        tag = model_names[key]
        files.set_tag(tag)
        # save relaxed pdb
        if n < num_relax:
            start = time.time()
            pdb_lines = relax_me(
                pdb_lines=unrelaxed_pdb_lines[key],
                max_iterations=relax_max_iterations,
                tolerance=relax_tolerance,
                stiffness=relax_stiffness,
                max_outer_iterations=relax_max_outer_iterations,
                use_gpu=use_gpu_relax)
            files.get("relaxed","pdb").write_text(pdb_lines)
            logger.info(f"Relaxation took {(time.time() - start):.1f}s")

        # rename files to include rank
        new_tag = f"rank_{(n+1):03d}_{tag}"
        rank.append(new_tag)
        logger.info(f"{new_tag}{metric[-1]['print_line']}")
        for x, ext, file in files.files[tag]:
            new_file = result_dir.joinpath(f"{prefix}_{x}_{new_tag}.{ext}")
            file.rename(new_file)
            result_files.append(new_file)

    return {"rank":rank,
            "metric":metric,
            "result_files":result_files}

def get_msa_and_templates(
    jobname: str,
    query_sequences: Union[str, List[str]],
    a3m_lines: Optional[List[str]],
    result_dir: Path,
    msa_mode: str,
    use_templates: bool,
    custom_template_path: str,
    pair_mode: str,
    pairing_strategy: str = "greedy",
    host_url: str = DEFAULT_API_SERVER,
    user_agent: str = "",
    max_template_date="2100-01-01",
    max_template_hits=20,
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    from colabfold.colabfold import run_mmseqs2

    use_env = msa_mode == "mmseqs2_uniref_env" or msa_mode == "mmseqs2_uniref_env_envpair"
    use_envpair = msa_mode == "mmseqs2_uniref_env_envpair"
    if isinstance(query_sequences, str): query_sequences = [query_sequences]

    # remove duplicates before searching
    query_seqs_unique = []
    for x in query_sequences:
        if x not in query_seqs_unique:
            query_seqs_unique.append(x)

    # determine how many times is each sequence is used
    query_seqs_cardinality = [0] * len(query_seqs_unique)
    for seq in query_sequences:
        seq_idx = query_seqs_unique.index(seq)
        query_seqs_cardinality[seq_idx] += 1

    # get template features
    template_features = []
    if use_templates:
        # Skip template search when custom_template_path is provided
        if custom_template_path is not None:
            if msa_mode == "single_sequence":
                a3m_lines = []
                num = 101
                for i, seq in enumerate(query_seqs_unique):
                    a3m_lines.append(f">{num + i}\n{seq}")

            if a3m_lines is None:
                a3m_lines_mmseqs2 = run_mmseqs2(
                    query_seqs_unique,
                    str(result_dir.joinpath(jobname)),
                    use_env,
                    use_templates=False,
                    host_url=host_url,
                    user_agent=user_agent,
                )
            else:
                a3m_lines_mmseqs2 = a3m_lines
            template_paths = {}
            for index in range(0, len(query_seqs_unique)):
                template_paths[index] = custom_template_path
        else:
            a3m_lines_mmseqs2, template_paths = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_templates=True,
                host_url=host_url,
                user_agent=user_agent,
            )
        if template_paths is None:
            logger.info("No template detected")
            for index in range(0, len(query_seqs_unique)):
                template_feature = mk_mock_template(query_seqs_unique[index])
                template_features.append(template_feature)
        else:
            for index in range(0, len(query_seqs_unique)):
                if template_paths[index] is not None:
                    template_feature = mk_template(
                        a3m_lines_mmseqs2[index],
                        template_paths[index],
                        query_seqs_unique[index],
                        max_template_date=max_template_date,
                        max_hits=max_template_hits,
                    )
                    if len(template_feature["template_domain_names"]) == 0:
                        template_feature = mk_mock_template(query_seqs_unique[index])
                        logger.info(f"Sequence {index} found no templates")
                    else:
                        logger.info(
                            f"Sequence {index} found templates: {template_feature['template_domain_names'].astype(str).tolist()}"
                        )
                else:
                    template_feature = mk_mock_template(query_seqs_unique[index])
                    logger.info(f"Sequence {index} found no templates")

                template_features.append(template_feature)
    else:
        for index in range(0, len(query_seqs_unique)):
            template_feature = mk_mock_template(query_seqs_unique[index])
            template_features.append(template_feature)

    if len(query_sequences) == 1:
        pair_mode = "none"

    if pair_mode == "none" or pair_mode == "unpaired" or pair_mode == "unpaired_paired":
        if msa_mode == "single_sequence":
            a3m_lines = []
            num = 101
            for i, seq in enumerate(query_seqs_unique):
                a3m_lines.append(f">{num + i}\n{seq}")
        else:
            # find normal a3ms
            a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=False,
                host_url=host_url,
                user_agent=user_agent,
            )
    else:
        a3m_lines = None

    if msa_mode != "single_sequence" and (
        pair_mode == "paired" or pair_mode == "unpaired_paired"
    ):
        # find paired a3m if not a homooligomers
        if len(query_seqs_unique) > 1:
            paired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_envpair,
                use_pairing=True,
                pairing_strategy=pairing_strategy,
                host_url=host_url,
                user_agent=user_agent,
            )
        else:
            # homooligomers
            num = 101
            paired_a3m_lines = []
            for i in range(0, query_seqs_cardinality[0]):
                paired_a3m_lines.append(f">{num+i}\n{query_seqs_unique[0]}\n")
    else:
        paired_a3m_lines = None

    return (
        a3m_lines,
        paired_a3m_lines,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )

def build_monomer_feature(
    sequence: str, unpaired_msa: str, template_features: Dict[str, Any]
):
    msa = pipeline.parsers.parse_a3m(unpaired_msa)
    # gather features
    return {
        **pipeline.make_sequence_features(
            sequence=sequence, description="none", num_res=len(sequence)
        ),
        **pipeline.make_msa_features([msa]),
        **template_features,
    }

def build_multimer_feature(paired_msa: str) -> Dict[str, ndarray]:
    parsed_paired_msa = pipeline.parsers.parse_a3m(paired_msa)
    return {
        f"{k}_all_seq": v
        for k, v in pipeline.make_msa_features([parsed_paired_msa]).items()
    }

def process_multimer_features(
    features_for_chain: Dict[str, Dict[str, ndarray]],
    min_num_seq: int = 512,
) -> Dict[str, ndarray]:
    all_chain_features = {}
    for chain_id, chain_features in features_for_chain.items():
        all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
            chain_features, chain_id
        )

    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    # np_example = feature_processing.pair_and_merge(
    #    all_chain_features=all_chain_features, is_prokaryote=is_prokaryote)
    feature_processing.process_unmerged_features(all_chain_features)
    np_chains_list = list(all_chain_features.values())
    # noinspection PyProtectedMember
    pair_msa_sequences = not feature_processing._is_homomer_or_monomer(np_chains_list)
    chains = list(np_chains_list)
    chain_keys = chains[0].keys()
    updated_chains = []
    for chain_num, chain in enumerate(chains):
        new_chain = {k: v for k, v in chain.items() if "_all_seq" not in k}
        for feature_name in chain_keys:
            if feature_name.endswith("_all_seq"):
                feats_padded = msa_pairing.pad_features(
                    chain[feature_name], feature_name
                )
                new_chain[feature_name] = feats_padded
        new_chain["num_alignments_all_seq"] = np.asarray(
            len(np_chains_list[chain_num]["msa_all_seq"])
        )
        updated_chains.append(new_chain)
    np_chains_list = updated_chains
    np_chains_list = feature_processing.crop_chains(
        np_chains_list,
        msa_crop_size=feature_processing.MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    # merge_chain_features crashes if there are additional features only present in one chain
    # remove all features that are not present in all chains
    common_features = set([*np_chains_list[0]]).intersection(*np_chains_list)
    np_chains_list = [
        {key: value for (key, value) in chain.items() if key in common_features}
        for chain in np_chains_list
    ]
    np_example = feature_processing.msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    np_example = feature_processing.process_final(np_example)

    # Pad MSA to avoid zero-sized extra_msa.
    np_example = pipeline_multimer.pad_msa(np_example, min_num_seq=min_num_seq)
    return np_example

def generate_input_feature(
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
    unpaired_msa: List[str],
    paired_msa: List[str],
    template_features: List[Dict[str, Any]],
    is_complex: bool,
    model_type: str,
    max_seq: int,
) -> Tuple[Dict[str, Any], Dict[str, str]]:

    input_feature = {}
    domain_names = {}
    if is_complex and "multimer" not in model_type:

        full_sequence = ""
        Ls = []
        for sequence_index, sequence in enumerate(query_seqs_unique):
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                full_sequence += sequence
                Ls.append(len(sequence))

        # bugfix
        a3m_lines = f">0\n{full_sequence}\n"
        a3m_lines += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)

        input_feature = build_monomer_feature(full_sequence, a3m_lines, mk_mock_template(full_sequence))
        input_feature["residue_index"] = np.concatenate([np.arange(L) for L in Ls])
        input_feature["asym_id"] = np.concatenate([np.full(L,n) for n,L in enumerate(Ls)])
        if any(
            [
                template != b"none"
                for i in template_features
                for template in i["template_domain_names"]
            ]
        ):
            logger.warning(
                f"{model_type} complex does not consider templates. Chose multimer model-type for template support."
            )

    else:
        features_for_chain = {}
        chain_cnt = 0
        # for each unique sequence
        for sequence_index, sequence in enumerate(query_seqs_unique):

            # get unpaired msa
            if unpaired_msa is None:
                input_msa = f">{101 + sequence_index}\n{sequence}"
            else:
                input_msa = unpaired_msa[sequence_index]

            feature_dict = build_monomer_feature(
                sequence, input_msa, template_features[sequence_index])

            if "multimer" in model_type:
                # get paired msa
                if paired_msa is None:
                    input_msa = f">{101 + sequence_index}\n{sequence}"
                else:
                    input_msa = paired_msa[sequence_index]
                feature_dict.update(build_multimer_feature(input_msa))

            # for each copy
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                features_for_chain[protein.PDB_CHAIN_IDS[chain_cnt]] = feature_dict
                chain_cnt += 1

        if "multimer" in model_type:
            # combine features across all chains
            input_feature = process_multimer_features(features_for_chain, min_num_seq=max_seq + 4)
            domain_names = {
                chain: [
                    name.decode("UTF-8")
                    for name in feature["template_domain_names"]
                    if name != b"none"
                ]
                for (chain, feature) in features_for_chain.items()
            }
        else:
            input_feature = features_for_chain[protein.PDB_CHAIN_IDS[0]]
            input_feature["asym_id"] = np.zeros(input_feature["aatype"].shape[0],dtype=int)
            domain_names = {
                protein.PDB_CHAIN_IDS[0]: [
                    name.decode("UTF-8")
                    for name in input_feature["template_domain_names"]
                    if name != b"none"
                ]
            }
    return (input_feature, domain_names)

def normalize_a3m(lines: list[str]) -> list[str]:
    out = []
    i = 0

    # keep meta header
    if lines and lines[0].startswith("#"):
        out.append(lines[0].rstrip("\n"))
        i = 1

    header = None
    seq_chunks = []
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                out.append(header)
                out.append("".join(seq_chunks))
            header = line
            seq_chunks = []
        else:
            # remove all whitespace inside sequence lines
            seq_chunks.append("".join(line.split()))
    if header is not None:
        out.append(header)
        out.append("".join(seq_chunks))

    return out

def unserialize_msa(
    a3m_lines: List[str], query_sequence: Union[List[str], str]
) -> Tuple[
    Optional[List[str]],
    Optional[List[str]],
    List[str],
    List[int],
    List[Dict[str, Any]],
]:
    a3m_lines = a3m_lines[0].replace("\x00", "").splitlines()
    a3m_lines = normalize_a3m(a3m_lines)
    if not a3m_lines[0].startswith("#") or len(a3m_lines[0][1:].split("\t")) != 2:
        assert isinstance(query_sequence, str)
        return (
            ["\n".join(a3m_lines)],
            None,
            [query_sequence],
            [1],
            [mk_mock_template(query_sequence)],
        )

    if len(a3m_lines) < 3:
        raise ValueError(f"Unknown file format a3m")
    tab_sep_entries = a3m_lines[0][1:].split("\t")
    query_seq_len = tab_sep_entries[0].split(",")
    query_seq_len = list(map(int, query_seq_len))
    query_seqs_cardinality = tab_sep_entries[1].split(",")
    query_seqs_cardinality = list(map(int, query_seqs_cardinality))
    num_chains = len(query_seq_len)
    is_homooligomer = (
        True if num_chains == 1 and query_seqs_cardinality[0] > 1 else False
    )
    is_single_protein = (
        True if num_chains == 1 and query_seqs_cardinality[0] == 1 else False
    )

    query_seqs_unique = []
    qcat = a3m_lines[2]
    prev = 0
    for qlen in query_seq_len:
        nxt = prev + max(int(qlen), 0)
        query_seqs_unique.append(qcat[prev:nxt])
        prev = nxt

    paired_chunks = [[] for _ in range(num_chains)]
    unpaired_chunks = [[] for _ in range(num_chains)]
    already_in = set()

    qlens_local = query_seq_len
    def _split_by_aln(s):
        segments = [""] * num_chains
        has_aa = [False] * num_chains
        seg_idx = 0
        aln_count = 0
        start = 0
        curr_has_aa = False

        dash = '-'
        A, Z = 'A', 'Z'
        a, z = 'a', 'z'
        n = len(s)
        i = 0
        while i < n and seg_idx < num_chains:
            c = s[i]
            # non-lowercase are aligned columns
            if not (a <= c <= z):
                if c != dash and (A <= c <= Z):
                    curr_has_aa = True
                aln_count += 1
                if aln_count == qlens_local[seg_idx]:
                    # close current segment
                    segments[seg_idx] = s[start:i+1]
                    has_aa[seg_idx] = curr_has_aa
                    seg_idx += 1
                    aln_count = 0
                    start = i + 1
                    curr_has_aa = False
            i += 1
        return segments, has_aa

    i = 1
    end = len(a3m_lines)
    while i + 1 < end:
        header = a3m_lines[i]
        seq = a3m_lines[i + 1]
        i += 2

        key = (header, seq)
        if key in already_in:
            continue
        already_in.add(key)

        segments, has_aa = _split_by_aln(seq)

        # Paired if multi-chain (not single protein), not homo-oligomer, >=2 segments have AA
        if (not is_single_protein) and (not is_homooligomer) and (sum(has_aa) > 1):
            header_no_gt = header.replace(">", "")
            header_fields = header_no_gt.split("\t")
            for j, seg in enumerate(segments):
                label = header_fields[j] if j < len(header_fields) else (header_fields[-1] if header_fields else "")
                pc = paired_chunks[j]
                pc.append(">")
                pc.append(label)
                pc.append("\n")
                pc.append(seg)
                pc.append("\n")
        else:
            for j, seg in enumerate(segments):
                if has_aa[j]:
                    uc = unpaired_chunks[j]
                    uc.append(header)
                    uc.append("\n")
                    uc.append(seg)
                    uc.append("\n")

    if is_homooligomer:
        num = 101
        count = max(query_seqs_cardinality[0], 0)
        q = query_seqs_unique[0] if query_seqs_unique else ""
        paired_msa = [f">{num + k}\n{q}\n" for k in range(count)]
    else:
        if is_single_protein:
            paired_msa = None
        else:
            paired_msa = ["".join(ch) for ch in paired_chunks]


    unpaired_msa = ["".join(ch) for ch in unpaired_chunks]
    template_features = [mk_mock_template(q) for q in query_seqs_unique]

    return (
        unpaired_msa,
        paired_msa,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )

def put_mmciffiles_into_resultdir(
    pdb_hit_file: Path,
    local_pdb_path: Path,
    result_dir: Path,
    max_num_templates: int = 20,
):
    """Put mmcif files from local_pdb_path into result_dir and unzip them.
    max_num_templates is the maximum number of templates to use (default: 20).
    Args:
        pdb_hit_file (Path): Path to pdb_hit_file
        local_pdb_path (Path): Path to local_pdb_path
        result_dir (Path): Path to result_dir
        max_num_templates (int): Maximum number of templates to use
    """
    pdb_hit_file = Path(pdb_hit_file)
    local_pdb_path = Path(local_pdb_path)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    query_ids = []
    with open(pdb_hit_file, "r") as f:
        for line in f:
            query_id = line.split("\t")[0]
            query_ids.append(query_id)
            if query_ids.count(query_id) > max_num_templates:
                continue
            else:
                pdb_id = line.split("\t")[1][0:4]
                divided_pdb_id = pdb_id[1:3]
                gzipped_divided_mmcif_file = local_pdb_path / divided_pdb_id / (pdb_id + ".cif.gz")
                gzipped_mmcif_file = local_pdb_path / (pdb_id + ".cif.gz")
                unzipped_mmcif_file = local_pdb_path / (pdb_id + ".cif")
                result_file = result_dir / (pdb_id + ".cif")
                possible_files = [gzipped_divided_mmcif_file, gzipped_mmcif_file, unzipped_mmcif_file]
                for file in possible_files:
                    if file == gzipped_divided_mmcif_file or file == gzipped_mmcif_file:
                        if file.exists():
                            with gzip.open(file, "rb") as f_in:
                                with open(result_file, "wb") as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                                    break
                    else:
                        # unzipped_mmcif_file
                        if file.exists():
                            shutil.copyfile(file, result_file)
                            break
                if not result_file.exists():
                    print(f"WARNING: {pdb_id} does not exist in {local_pdb_path}.")


def run(
    queries: List[Tuple[str, Union[str, List[str]], Optional[List[str]], Optional[List[Tuple[str, str,int]]]]],
    result_dir: Union[str, Path],
    num_models: int,
    is_complex: bool,
    num_recycles: Optional[int] = None,
    recycle_early_stop_tolerance: Optional[float] = None,
    model_order: List[int] = [1,2,3,4,5],
    initial_guess: str = None,
    num_ensemble: int = 1,
    model_type: str = "auto",
    msa_mode: str = "mmseqs2_uniref_env",
    use_templates: bool = False,
    custom_template_path: str = None,
    custom_template_cache_path: str = None,
    num_relax: int = 0,
    relax_max_iterations: int = 0,
    relax_tolerance: float = 2.39,
    relax_stiffness: float = 10.0,
    relax_max_outer_iterations: int = 3,
    keep_existing_results: bool = True,
    rank_by: str = "auto",
    pair_mode: str = "unpaired_paired",
    pairing_strategy: str = "greedy",
    data_dir: Union[str, Path] = default_data_dir,
    host_url: str = DEFAULT_API_SERVER,
    user_agent: str = "",
    random_seed: int = 0,
    num_seeds: int = 1,
    recompile_padding: Union[int, float] = 10,
    zip_results: bool = False,
    prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    skip_output: List[str] = [],
    jobname_prefix: Optional[str] = None,
    save_all: bool = False,
    save_recycles: bool = False,
    use_dropout: bool = False,
    use_gpu_relax: bool = False,
    stop_at_score: float = 100,
    dpi: int = 200,
    max_seq: Optional[int] = None,
    max_extra_seq: Optional[int] = None,
    pdb_hit_file: Optional[Path] = None,
    local_pdb_path: Optional[Path] = None,
    use_cluster_profile: bool = True,
    feature_dict_callback: Callable[[Any], Any] = None,
    calc_extra_ptm: bool = False,
    use_probs_extra: bool = True,
    max_template_date: str = "2100-01-01",
    max_template_hits: int = 20,
    **kwargs
):
    # check what device is available
    try:
        # check if TPU is available
        from tpu_info import device
        if len(device.get_local_chips()) > 0:
            import jax.tools.colab_tpu
            jax.tools.colab_tpu.setup_tpu()
            logger.info('Running on TPU')
            use_gpu_relax = False
    except:
        if local_devices()[0].platform == 'cpu':
            logger.info("WARNING: no GPU detected, will be using CPU")
            use_gpu_relax = False
        else:
            logger.info('Running on GPU')

    from colabfold.alphafold.models import load_models_and_params

    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    model_type = set_model_type(is_complex, model_type)

    # backward-compatibility with old options
    old_names = {"MMseqs2 (UniRef+Environmental)":"mmseqs2_uniref_env",
                 "MMseqs2 (UniRef+Environmental+Env. Pairing)":"mmseqs2_uniref_env_envpair",
                 "MMseqs2 (UniRef only)":"mmseqs2_uniref",
                 "unpaired+paired":"unpaired_paired"}
    msa_mode   = old_names.get(msa_mode,msa_mode)
    pair_mode  = old_names.get(pair_mode,pair_mode)
    feature_dict_callback = kwargs.pop("input_features_callback", feature_dict_callback)
    use_dropout           = kwargs.pop("training", use_dropout)
    use_fuse              = kwargs.pop("use_fuse", True)
    use_bfloat16          = kwargs.pop("use_bfloat16", True)
    use_pallas            = kwargs.pop("use_pallas", False)
    compile_mode          = kwargs.pop("compile_mode", "tuned")
    if use_pallas and not use_bfloat16:
        raise ValueError("use_pallas requires bfloat16")
    max_msa               = kwargs.pop("max_msa",None)
    if max_msa is not None:
        max_seq, max_extra_seq = [int(x) for x in max_msa.split(":")]

    if kwargs.pop("use_amber", False) and num_relax == 0:
        num_relax = num_models * num_seeds

    if len(kwargs) > 0:
        print(f"WARNING: the following options are not being used: {kwargs}")

    # decide how to rank outputs
    if rank_by == "auto":
        rank_by = "multimer" if is_complex else "plddt"
    if "ptm" not in model_type and "multimer" not in model_type:
        rank_by = "plddt"

    # added for actifptm calculation
    if not is_complex and calc_extra_ptm:
        logger.info("Calculating extra pTM is not supported for single chain prediction, skipping it.")
        calc_extra_ptm = False

    # get max length
    max_len = 0
    max_num = 0
    for _, query_sequence, _, _ in queries:
        N = 1 if isinstance(query_sequence,str) else len(query_sequence)
        L = len("".join(query_sequence))
        if L > max_len: max_len = L
        if N > max_num: max_num = N

    # get max sequences
    # 512 5120 = alphafold_ptm (models 1,3,4)
    # 512 1024 = alphafold_ptm (models 2,5)
    # 508 2048 = alphafold-multimer_v3 (models 1,2,3)
    # 508 1152 = alphafold-multimer_v3 (models 4,5)
    # 252 1152 = alphafold-multimer_v[1,2]

    set_if = lambda x,y: y if x is None else x
    if model_type in ["alphafold2_multimer_v1","alphafold2_multimer_v2"]:
        (max_seq, max_extra_seq) = (set_if(max_seq,252), set_if(max_extra_seq,1152))
    elif model_type == "alphafold2_multimer_v3":
        (max_seq, max_extra_seq) = (set_if(max_seq,508), set_if(max_extra_seq,2048))
    else:
        (max_seq, max_extra_seq) = (set_if(max_seq,512), set_if(max_extra_seq,5120))

    if msa_mode == "single_sequence":
        num_seqs = 1
        if is_complex and "multimer" not in model_type: num_seqs += max_num
        if use_templates: num_seqs += 4
        max_seq = min(num_seqs, max_seq)
        max_extra_seq = max(min(num_seqs - max_seq, max_extra_seq), 1)

    # sort model order
    model_order.sort()

    # initial guess
    if initial_guess is not None:
        logger.info(f'Using initial guess: {initial_guess}')

    # Record the parameters of this run
    config = {
        "num_queries": len(queries),
        "use_templates": use_templates,
        "num_relax": num_relax,
        "relax_max_iterations": relax_max_iterations,
        "relax_tolerance": relax_tolerance,
        "relax_stiffness": relax_stiffness,
        "relax_max_outer_iterations": relax_max_outer_iterations,
        "msa_mode": msa_mode,
        "model_type": model_type,
        "num_models": num_models,
        "num_recycles": num_recycles,
        "recycle_early_stop_tolerance": recycle_early_stop_tolerance,
        "num_ensemble": num_ensemble,
        "model_order": model_order,
        "initial_guess": initial_guess,
        "keep_existing_results": keep_existing_results,
        "rank_by": rank_by,
        "max_seq": max_seq,
        "max_extra_seq": max_extra_seq,
        "pair_mode": pair_mode,
        "pairing_strategy": pairing_strategy,
        "host_url": host_url,
        "user_agent": user_agent,
        "stop_at_score": stop_at_score,
        "random_seed": random_seed,
        "num_seeds": num_seeds,
        "recompile_padding": recompile_padding,
        "commit": get_commit(),
        "use_dropout": use_dropout,
        "use_cluster_profile": use_cluster_profile,
        "use_fuse": use_fuse,
        "use_bfloat16": use_bfloat16,
        "use_pallas": use_pallas,
        "compile_mode": compile_mode,
        "version": importlib_metadata.version("colabfold"),
        "calc_extra_ptm": calc_extra_ptm,
        "use_probs_extra": use_probs_extra,
        "max_template_date": max_template_date,
        "max_template_hits": max_template_hits,
    }
    config_out_file = result_dir.joinpath("config.json")
    config_out_file.write_text(json.dumps(config, indent=4))
    use_env = "env" in msa_mode
    use_msa = "mmseqs2" in msa_mode
    use_amber = num_models > 0 and num_relax > 0

    bibtex_file = write_bibtex(
        model_type if num_models > 0 else "", use_msa, use_env, use_templates, use_amber, result_dir
    )

    if pdb_hit_file is not None:
        if local_pdb_path is None:
            raise ValueError("local_pdb_path is not specified.")
        else:
            custom_template_path = result_dir / "templates"
            put_mmciffiles_into_resultdir(pdb_hit_file, local_pdb_path, custom_template_path)


    if custom_template_path is not None:
        mk_hhsearch_db(custom_template_path)

    pad_len = 0
    msa_pad_depth = 0
    ranks, metrics = [],[]
    first_job = True
    job_number = 0
    for job_number, (raw_jobname, query_sequence, a3m_lines, custom_template_path_per_entry) in enumerate(queries):

        if use_templates and custom_template_path_per_entry is not None and isinstance(custom_template_path_per_entry, Path):
            mk_hhsearch_single_entry_db(custom_template_path_per_entry, custom_template_cache_path)
            custom_template_path = custom_template_cache_path

        if jobname_prefix is not None:
            # pad job number based on number of queries
            fill = len(str(len(queries)))
            jobname = safe_filename(jobname_prefix) + "_" + str(job_number).zfill(fill)
            job_number += 1
        else:
            jobname = safe_filename(raw_jobname)

        #######################################
        # check if job has already finished
        #######################################
        # In the colab version and with --zip we know we're done when a zip file has been written
        result_zip = result_dir.joinpath(jobname).with_suffix(".result.zip")
        if keep_existing_results and result_zip.is_file():
            logger.info(f"Skipping {jobname} (result.zip)")
            continue
        # In the local version we use a marker file
        is_done_marker = result_dir.joinpath(jobname + ".done.txt")
        if keep_existing_results and is_done_marker.is_file():
            logger.info(f"Skipping {jobname} (already done)")
            continue

        seq_len = len("".join(query_sequence))
        logger.info(f"Query {job_number + 1}/{len(queries)}: {jobname} (length {seq_len})")

        ###########################################
        # generate MSA (a3m_lines) and templates
        ###########################################
        try:
            pickled_msa_and_templates = result_dir.joinpath(f"{jobname}.pickle")
            if pickled_msa_and_templates.is_file():
                with open(pickled_msa_and_templates, 'rb') as f:
                    (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) = pickle.load(f)
                logger.info(f"Loaded {pickled_msa_and_templates}")

            else:
                if a3m_lines is None:
                    (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) \
                    = get_msa_and_templates(
                        jobname, query_sequence, a3m_lines, result_dir, msa_mode, use_templates,
                        custom_template_path, pair_mode, pairing_strategy, host_url, user_agent,
                        max_template_date=max_template_date, max_template_hits=max_template_hits,
                    )

                elif a3m_lines is not None:
                    if(isinstance(a3m_lines, Path)):
                        a3m_lines = [a3m_lines.read_text()]
                    (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) \
                    = unserialize_msa(a3m_lines, query_sequence)
                    if use_templates:
                        (_, _, _, _, template_features) \
                            = get_msa_and_templates(
                                jobname, query_seqs_unique, unpaired_msa, result_dir, 'single_sequence', use_templates,
                                custom_template_path, pair_mode, pairing_strategy, host_url, user_agent,
                                max_template_date=max_template_date, max_template_hits=max_template_hits,
                            )

                if num_models == 0:
                    with open(pickled_msa_and_templates, 'wb') as f:
                        pickle.dump((unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features), f)
                    logger.info(f"Saved {pickled_msa_and_templates}")

            # save a3m
            if not 'msa' in skip_output:
                msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
                result_dir.joinpath(f"{jobname}.a3m").write_text(msa)

        except Exception as e:
            logger.exception(f"Could not get MSA/templates for {jobname}: {e}")
            continue

        #######################
        # generate features
        #######################
        try:
            (feature_dict, domain_names) \
            = generate_input_feature(query_seqs_unique, query_seqs_cardinality, unpaired_msa, paired_msa,
                                     template_features, is_complex, model_type, max_seq=max_seq)

            # to allow display of MSA info during colab/chimera run (thanks tomgoddard)
            if feature_dict_callback is not None:
                feature_dict_callback(feature_dict)

        except Exception as e:
            logger.exception(f"Could not generate input features {jobname}: {e}")
            continue

        ###############
        # save plots not requiring prediction
        ###############

        result_files = []

        # make msa plot
        if not 'plots' in skip_output:
            from colabfold.plot import plot_msa_v2
            msa_plot = plot_msa_v2(feature_dict, dpi=dpi)
            coverage_png = result_dir.joinpath(f"{jobname}_coverage.png")
            msa_plot.savefig(str(coverage_png), bbox_inches='tight')
            msa_plot.close()
            result_files.append(coverage_png)

        if use_templates:
            templates_file = result_dir.joinpath(f"{jobname}_template_domain_names.json")
            templates_file.write_text(json.dumps(domain_names))
            result_files.append(templates_file)

        result_files.append(result_dir.joinpath(jobname + ".a3m"))
        result_files += [bibtex_file, config_out_file]

        ######################
        # predict structures
        ######################
        if num_models > 0:
            try:
                # get list of lengths
                query_sequence_len_array = sum([[len(x)] * y
                    for x,y in zip(query_seqs_unique, query_seqs_cardinality)],[])

                # decide how much to pad (to avoid recompiling)
                if seq_len > pad_len:
                    if isinstance(recompile_padding, float):
                        pad_len = math.ceil(seq_len * recompile_padding)
                    else:
                        pad_len = seq_len + recompile_padding
                    pad_len = min(pad_len, max_len)

                # prep model and params
                if first_job:
                    # if one job input adjust max settings
                    if len(queries) == 1 and msa_mode != "single_sequence":
                        # get number of sequences
                        if "msa_mask" in feature_dict:
                            num_seqs = int(sum(feature_dict["msa_mask"].max(-1) == 1))
                        else:
                            num_seqs = int(len(feature_dict["msa"]))

                        if use_templates: num_seqs += 4

                        # adjust max settings
                        max_seq = min(num_seqs, max_seq)
                        max_extra_seq = max(min(num_seqs - max_seq, max_extra_seq), 1)
                        logger.info(f"Setting max_seq={max_seq}, max_extra_seq={max_extra_seq}")

                    model_runner_and_params = load_models_and_params(
                        num_models=num_models,
                        use_templates=use_templates,
                        num_recycles=num_recycles,
                        num_ensemble=num_ensemble,
                        model_order=model_order,
                        model_type=model_type,
                        data_dir=data_dir,
                        stop_at_score=stop_at_score,
                        rank_by=rank_by,
                        use_dropout=use_dropout,
                        max_seq=max_seq,
                        max_extra_seq=max_extra_seq,
                        use_cluster_profile=use_cluster_profile,
                        recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                        use_fuse=use_fuse,
                        use_bfloat16=use_bfloat16,
                        save_all=save_all,
                        calc_extra_ptm=calc_extra_ptm,
                        use_pallas=use_pallas,
                        compile_mode=compile_mode
                    )
                    first_job = False

                # Track the deepest multimer MSA seen so far
                if "multimer" in model_type and "msa" in feature_dict:
                    msa_pad_depth = max(msa_pad_depth, len(feature_dict["msa"]))

                results = predict_structure(
                    prefix=jobname,
                    result_dir=result_dir,
                    feature_dict=feature_dict,
                    msa_pad_depth=msa_pad_depth,
                    is_complex=is_complex,
                    use_templates=use_templates,
                    sequences_lengths=query_sequence_len_array,
                    pad_len=pad_len,
                    initial_guess=initial_guess,
                    model_type=model_type,
                    model_runner_and_params=model_runner_and_params,
                    num_relax=num_relax,
                    relax_max_iterations=relax_max_iterations,
                    relax_tolerance=relax_tolerance,
                    relax_stiffness=relax_stiffness,
                    relax_max_outer_iterations=relax_max_outer_iterations,
                    rank_by=rank_by,
                    stop_at_score=stop_at_score,
                    prediction_callback=prediction_callback,
                    use_gpu_relax=use_gpu_relax,
                    random_seed=random_seed,
                    num_seeds=num_seeds,
                    save_all=save_all,
                    save_single_representations=save_single_representations,
                    save_pair_representations=save_pair_representations,
                    save_recycles=save_recycles,
                    calc_extra_ptm=calc_extra_ptm,
                    use_probs_extra=use_probs_extra,
                )
                
                result_files += results["result_files"]
                ranks.append(results["rank"])
                metrics.append(results["metric"])

            except RuntimeError as e:
                # This normally happens on OOM. TODO: Filter for the specific OOM error message
                logger.error(f"Could not predict {jobname}. Not Enough GPU memory? {e}")
                continue

            ###############
            # save prediction plots
            ###############

            # load the scores
            if not 'pae_json' in skip_output:
                scores = []
                for r in results["rank"][:5]:
                    scores_file = result_dir.joinpath(f"{jobname}_scores_{r}.json")
                    with scores_file.open("r") as handle:
                        scores.append(json.load(handle))

                # write alphafold-db format (pAE)
                if "pae" in scores[0]:
                    af_pae_file = result_dir.joinpath(f"{jobname}_predicted_aligned_error_v1.json")
                    af_pae_file.write_text(json.dumps({
                        "predicted_aligned_error":scores[0]["pae"],
                        "max_predicted_aligned_error":scores[0]["max_pae"]}))
                    result_files.append(af_pae_file)

                    # make pAE plots
                    if not 'plots' in skip_output:
                        from colabfold.colabfold import plot_paes
                        paes_plot = plot_paes([np.asarray(x["pae"]) for x in scores],
                            Ls=query_sequence_len_array, dpi=dpi)
                        pae_png = result_dir.joinpath(f"{jobname}_pae.png")
                        paes_plot.savefig(str(pae_png), bbox_inches='tight')
                        paes_plot.close()
                        result_files.append(pae_png)

                    # make pairwise interface metric plots and chainwise ptm plot
                    if calc_extra_ptm:
                        ext_metric_png = result_dir.joinpath(f"{jobname}_ext_metrics.png")
                        extra_ptm.plot_chain_pairwise_analysis(scores, fig_path=ext_metric_png)

                # make pLDDT plot
                if not 'plots' in skip_output:
                    from colabfold.colabfold import plot_plddts
                    plddt_plot = plot_plddts([np.asarray(x["plddt"]) for x in scores],
                        Ls=query_sequence_len_array, dpi=dpi)
                    plddt_png = result_dir.joinpath(f"{jobname}_plddt.png")
                    plddt_plot.savefig(str(plddt_png), bbox_inches='tight')
                    plddt_plot.close()
                    result_files.append(plddt_png)

        if zip_results:
            with zipfile.ZipFile(result_zip, "w") as result_zip:
                for file in result_files:
                    result_zip.write(file, arcname=file.name)

            # Delete only after the zip was successful, and also not the bibtex and config because we need those again
            for file in result_files:
                if file != bibtex_file and file != config_out_file:
                    file.unlink()
        else:
            if num_models > 0:
                is_done_marker.touch()

    logger.info("Done")
    return {"rank":ranks,"metric":metrics}

def set_model_type(is_complex: bool, model_type: str) -> str:
    # backward-compatibility with old options
    old_names = {
        "AlphaFold2-multimer-v1":"alphafold2_multimer_v1",
        "AlphaFold2-multimer-v2":"alphafold2_multimer_v2",
        "AlphaFold2-multimer-v3":"alphafold2_multimer_v3",
        "AlphaFold2-ptm":        "alphafold2_ptm",
        "AlphaFold2":            "alphafold2",
        "DeepFold":              "deepfold_v1",
    }
    model_type = old_names.get(model_type, model_type)
    if model_type == "auto":
        if is_complex:
            model_type = "alphafold2_multimer_v3"
        else:
            model_type = "alphafold2_ptm"
    return model_type

def generate_af3_input(
    queries: List[Tuple[str, Union[str, List[str]], Optional[List[str]], Optional[List[Tuple[str, str, int]]]]],
    result_dir: Union[str, Path],
    msa_mode: str = "mmseqs2_uniref_env",
    pair_mode: str = "unpaired_paired",
    pairing_strategy: str = "greedy",
    use_templates: bool = False,
    custom_template_path: str = None,
    jobname_prefix: Optional[str] = None,
    host_url: str = DEFAULT_API_SERVER,
    user_agent: str = "",
    max_template_date: str = "2100-01-01",
    max_template_hits: int = 20,
    # is_complex: bool,
    # model_type: str = "auto",
):
    result_dir = Path(result_dir) 
    result_dir.mkdir(exist_ok=True)

    job_number = 0

    for job_number, (raw_jobname, query_sequences, a3m_lines, other_molecules) in enumerate(queries):
        if jobname_prefix is not None:
            # pad job number based on number of queries
            fill = len(str(len(queries)))
            jobname = safe_filename(jobname_prefix) + "_" + str(job_number).zfill(fill)
            # job_number += 1 # Why add?
        else:
            jobname = safe_filename(raw_jobname)

        ###########################################
        # generate MSA (a3m_lines) and templates
        ###########################################
        try:
            if a3m_lines is None:
                (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) \
                = get_msa_and_templates(
                    jobname, query_sequences, a3m_lines, result_dir, msa_mode, use_templates,
                    custom_template_path, pair_mode, pairing_strategy, host_url, user_agent,
                    max_template_date=max_template_date, max_template_hits=max_template_hits,
                )

            elif a3m_lines is not None:
                (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) \
                = unserialize_msa(a3m_lines, query_sequences)
                # if use_templates:
                #     (_, _, _, _, template_features) \
                #         = get_msa_and_templates(
                #               jobname, query_seqs_unique, unpaired_msa, result_dir, 'single_sequence', use_templates,
                #               custom_template_path, pair_mode, pairing_strategy, host_url, user_agent,
                #               max_template_date=max_template_date, max_template_hits=max_template_hits,
                #           )

            # save json
            af3 = AF3Utils(jobname, query_seqs_unique, query_seqs_cardinality, unpaired_msa, paired_msa, other_molecules)
            with open(result_dir.joinpath(f"{jobname}.json"), "w") as f:
                f.write(json.dumps(af3.content, indent = 4))
                
            # save a3m
            msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
            result_dir.joinpath(f"{jobname}.a3m").write_text(msa)

        except Exception as e:
            logger.exception(f"Failed to generate AF3 input json for {jobname}: Could not get MSA/templates. {e}")
            continue

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "input",
        default="input",
        help="One of: 1) directory with FASTA/A3M files, 2) CSV/TSV file, 3) FASTA file or 4) A3M file.",
    )
    parser.add_argument("results", help="Results output directory.")

    msa_group = parser.add_argument_group("MSA arguments", "")
    msa_group.add_argument(
        "--msa-only",
        action="store_true",
        help="Query and store MSAs from the MSA server without structure prediction",
    )
    msa_group.add_argument(
        "--msa-mode",
        default="mmseqs2_uniref_env",
        choices=[
            "mmseqs2_uniref_env",
            "mmseqs2_uniref_env_envpair",
            "mmseqs2_uniref",
            "single_sequence",
        ],
        help="Databases to use to create the MSA: UniRef30+Environmental (default), UniRef30 only or None. "
        "Using an A3M file as input overwrites this option.",
    )
    msa_group.add_argument(
        "--pair-mode",
        help="Multimer MSA pairing mode for complex prediction: unpaired MSA only, paired MSA only, both (default).",
        type=str,
        default="unpaired_paired",
        choices=["unpaired", "paired", "unpaired_paired"],
    )
    msa_group.add_argument(
        "--pair-strategy",
        help="How sequences are paired during MSA pairing for complex prediction. "
        "complete: MSA sequences should only be paired if the same species exists in all MSAs. "
        "greedy: MSA sequences should only be paired if the same species exists in at least two MSAs. "
        "Typically, greedy produces better predictions as it results in more paired sequences. "
        "However, in some cases complete pairing might help, especially if MSAs are already large and can be well paired. ",
        type=str,
        default="greedy",
        choices=["complete", "greedy"],
    )
    msa_group.add_argument(
        "--templates",
        default=False,
        action="store_true",
        help="Query PDB templates from the MSA server. "
        'If this parameter is not set, "--custom-template-path" and "--pdb-hit-file" will not be used. '
        "Warning: This can result in the MSA server being queried with A3M input. "
    )
    msa_group.add_argument(
        "--custom-template-path",
        type=str,
        default=None,
        help="Directory with PDB files to provide as custom templates to the predictor. "
        "No templates will be queried from the MSA server. "
        "'--templates' argument is also required to enable this.",
    )
    msa_group.add_argument(
        "--custom-template-cache-path",
        type=str,
        default=None,
        help="Directory to generate temporary HHsearch databases for custom templates.",
    )
    msa_group.add_argument(
        "--max-template-date",
        type=str,
        default="2100-01-01",
        help="Maximum release date (YYYY-MM-DD) for templates to be considered."
    )
    msa_group.add_argument(
        "--max-template-hits",
        type=int,
        default=20,
        help="Maximum number of template hits to consider."
    )
    msa_group.add_argument(
        "--pdb-hit-file",
        default=None,
        help="Path to a BLAST-m8 formatted PDB hit file corresponding to the input A3M file (e.g. pdb70.m8). "
        "Typically, this parameter should be used for a MSA generated by 'colabfold_search'. "
        "'--templates' argument is also required to enable this.",
    )
    msa_group.add_argument(
        "--local-pdb-path",
        default=None,
        help="Directory of a local mirror of the PDB mmCIF database (e.g. /path/to/pdb/divided). "
        "If provided, PDB files from the directory are used for templates specified by '--pdb-hit-file'. ",
    )

    pred_group = parser.add_argument_group("Prediction arguments", "")
    pred_group.add_argument(
        "--num-recycle",
        help="Number of prediction recycles. "
        "Increasing recycles can improve the prediction quality but slows down the prediction.",
        type=int,
        default=None,
    )
    pred_group.add_argument(
        "--recycle-early-stop-tolerance",
        help="Specify convergence criteria. "
        "Run recycles until the distance between recycles is within the given tolerance value.",
        type=float,
        default=None,
    )
    pred_group.add_argument(
        "--num-ensemble",
        help="Number of ensembles. "
        "The trunk of the network is run multiple times with different random choices for the MSA cluster centers. "
        "This can result in a better prediction at the cost of longer runtime. ",
        type=int,
        default=1,
    )
    pred_group.add_argument(
        "--num-seeds",
        help="Number of seeds to try. Will iterate from range(random_seed, random_seed+num_seeds). "
        "This can result in a better/different prediction at the cost of longer runtime. ",
        type=int,
        default=1,
    )
    pred_group.add_argument(
        "--random-seed",
        help="Changing the seed for the random number generator can result in better/different structure predictions.",
        type=int,
        default=0,
    )
    pred_group.add_argument(
        "--num-models",
        help="Number of models to use for structure prediction. "
        "Reducing the number of models speeds up the prediction but results in lower quality.",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
    )
    pred_group.add_argument(
        "--model-type",
        help="Predict structure/complex using the given model. "
        'Auto will pick "alphafold2_ptm" for structure predictions and "alphafold2_multimer_v3" for complexes. '
        "Older versions of the AF2 models are generally worse, however they can sometimes result in better predictions. "
        "If the model is not already downloaded, it will be automatically downloaded. ",
        type=str,
        default="auto",
        choices=[
            "auto",
            "alphafold2",
            "alphafold2_ptm",
            "alphafold2_multimer_v1",
            "alphafold2_multimer_v2",
            "alphafold2_multimer_v3",
            "deepfold_v1",
        ],
    )
    pred_group.add_argument("--model-order", default="1,2,3,4,5", type=str)
    pred_group.add_argument(
        "--initial-guess",
        nargs="?",
        const=True,
        help="Specify a starting model for the prediction. If the main input file is a PDB format, "
        "it will be used as the initial guess. Otherwise, you can provide an input file with this flag, "
        "which will override the main input."
    )
    pred_group.add_argument(
        "--use-dropout",
        default=False,
        action="store_true",
        help="Activate dropouts during inference to sample from uncertainty of the models. "
        "This can result in different predictions and can be (carefully!) used for conformations sampling.",
    )
    pred_group.add_argument(
        "--max-seq",
        help="Number of sequence clusters to use. "
        "This can result in different predictions and can be (carefully!) used for conformations sampling.",
        type=int,
        default=None,
    )
    pred_group.add_argument(
        "--max-extra-seq",
        help="Number of extra sequences to use. "
        "This can result in different predictions and can be (carefully!) used for conformations sampling.",
        type=int,
        default=None,
    )
    pred_group.add_argument(
        "--max-msa",
        help="Defines: `max-seq:max-extra-seq` number of sequences to use in one go. "
        '"--max-seq" and "--max-extra-seq" are ignored if this parameter is set.',
        type=str,
        default=None,
    )
    pred_group.add_argument(
        "--disable-cluster-profile",
        default=False,
        action="store_true",
        help="Experimental: For multimer models, disable cluster profiles.",
    )
    pred_group.add_argument(
        "--calc-extra-ptm",
        default=False,
        action="store_true",
        help="Experimental: calculate pairwise metrics (ipTM and actifpTM), and also chain-wise pTM",
    )
    pred_group.add_argument(
        "--no-use-probs-extra",
        default=False,
        action="store_true",
        help="Experimental: instead of contact probabilities form use binary contacts for extra metrics calculation",
    )
    pred_group.add_argument("--data", help="Path to AlphaFold2 weights directory.")

    relax_group = parser.add_argument_group("Relaxation arguments", "")
    relax_group.add_argument(
        "--amber",
        default=False,
        action="store_true",
        help="Enable OpenMM/Amber for structure relaxation. "
        "Can improve the quality of side-chains at a cost of longer runtime. "
    )
    relax_group.add_argument(
        "--num-relax",
        help="Specify how many of the top ranked structures to relax using OpenMM/Amber. "
        "Typically, relaxing the top-ranked prediction is enough and speeds up the runtime. ",
        type=int,
        default=0,
    )
    relax_group.add_argument(
        "--relax-max-iterations",
        type=int,
        default=2000,
        help="Maximum number of iterations for the relaxation process. "
        "AlphaFold2 sets this to unlimited (0), however, we found that this can lead to very long relaxation times for some inputs.",
    )
    relax_group.add_argument(
        "--relax-tolerance",
        type=float,
        default=2.39,
        help="Tolerance threshold for relaxation convergence.",
    )
    relax_group.add_argument(
        "--relax-stiffness",
        type=float,
        default=10.0,
        help="Stiffness parameter for relaxation.",
    )
    relax_group.add_argument(
        "--relax-max-outer-iterations",
        type=int,
        default=3,
        help="Maximum number of outer iterations for the relaxation process.",
    )
    relax_group.add_argument(
        "--use-gpu-relax",
        default=False,
        action="store_true",
        help="Run OpenMM/Amber on GPU instead of CPU. "
        "This can significantly speed up the relaxation runtime, however, might lead to compatibility issues with CUDA. "
        "Unsupported on AMD/ROCM and Apple Silicon.",
    )

    output_group = parser.add_argument_group("Output arguments", "")
    output_group.add_argument(
        "--rank",
        help='Choose metric to rank the "--num-models" predicted models.',
        type=str,
        default="auto",
        choices=["auto", "plddt", "ptm", "iptm", "multimer"],
    )
    output_group.add_argument(
        "--stop-at-score",
        help="Compute models until pLDDT (single chain) or pTM-score (multimer) > threshold is reached. "
        "This speeds up prediction by running less models for easier queries.",
        type=float,
        default=100,
    )
    output_group.add_argument(
        "--jobname-prefix",
        help="If set, the jobname will be prefixed with the given string and a running number, instead of the input headers/accession.",
        type=str,
        default=None,
    )
    output_group.add_argument(
        "--save-all",
        default=False,
        action="store_true",
        help="Save all raw outputs from model to a pickle file. "
        "Useful for downstream use in other models."
    )
    output_group.add_argument(
        "--save-recycles",
        default=False,
        action="store_true",
        help="Save all intermediate predictions at each recycle iteration.",
    )
    output_group.add_argument(
        "--save-single-representations",
        default=False,
        action="store_true",
        help="Save the single representation embeddings of all models.",
    )
    output_group.add_argument(
        "--save-pair-representations",
        default=False,
        action="store_true",
        help="Save the pair representation embeddings of all models.",
    )
    def comma_separated_list(arg_string):
        return [item.strip() for item in arg_string.split(',') if item.strip() in ['msa', 'plots', 'pae_json']]
    output_group.add_argument(
        "--skip-output",
        help="Comma-separated list of output types to skip: msa, plots, pae_json.",
        type=comma_separated_list,
        default="",
    )
    output_group.add_argument(
        "--overwrite-existing-results",
        default=False,
        action="store_true",
        help="Do not recompute results, if a query has already been predicted.",
    )
    output_group.add_argument(
        "--zip",
        default=False,
        action="store_true",
        help="Zip all results into one <jobname>.result.zip and delete the original files.",
    )
    output_group.add_argument(
        "--sort-queries-by",
        help="Sort input queries by: none, length, msa_depth, random. "
        "Sorting by length or depth speeds up monomer or multimer prediction as models are recompiled less often.",
        type=str,
        default="length",
        choices=["none", "length", "msa_depth", "random"],
    )

    adv_group = parser.add_argument_group(
        "Advanced arguments", ""
    )
    adv_group.add_argument(
        "--host-url",
        default=DEFAULT_API_SERVER,
        help="Which MSA server should be queried. By default, the free public MSA server hosted by the ColabFold team is queried. "
    )
    adv_group.add_argument(
        "--disable-unified-memory",
        default=False,
        action="store_true",
        help="If you are getting TensorFlow/Jax errors, it might help to disable this.",
    )
    adv_group.add_argument(
        "--recompile-padding",
        type=int,
        default=10,
        help="Whenever the input length changes, the model needs to be recompiled. "
        "We pad sequences by the specified length, so we can e.g., compute sequences from length 100 to 110 without recompiling. "
        "Individual predictions will become marginally slower due to longer input, "
        "but overall performance increases due to not recompiling. "
        "Set to 0 to disable.",
    )
    adv_group.add_argument(
        "--use-pallas",
        nargs="?",
        const=True,
        default=False,
        type=_str2bool,
        metavar="BOOL",
        help="Use Pallas/Triton kernels for faster prediction",
    )
    adv_group.add_argument(
        "--compile-mode",
        choices=["fast", "tuned", "full"],
        default="tuned",
        help="Kernel autotuning effort, trading compile time for inference speed: "
        "'fast': cuBLAS only, fastest compile, ~4%% slower inference, single/one-off predictions. "
        "'tuned': fixed, tuned kernel shape compiled with cuBLAS and Pallas, fast compile and ~1%% off optimal. "
        "'full': unbounded Kernel autotune, can take tens of minutes to compile for longsequences  but optimal inference, can be worth it for large batches."
    )
    adv_group.add_argument(
        "--debug-logging",
        default=False,
        action="store_true",
        help="Enable debug message logging.",
    )

    af3_group = parser.add_argument_group(
        "AlphaFold3 arguments", ""
    )
    af3_group.add_argument(
        "--af3-json",
        help="Generate input JSON for AlphaFold3 from the provided FASTA/A3M file.",
        action="store_true",
    )
    
    args = parser.parse_args()

    if (args.custom_template_path is not None) and (args.pdb_hit_file is not None):
        raise RuntimeError("Arguments --pdb-hit-file and --custom-template-path cannot be used simultaneously.")
    # disable unified memory
    if args.disable_unified_memory:
        for k in ENV.keys():
            if k in os.environ: del os.environ[k]

    setup_logging(Path(args.results).joinpath("log.txt"), verbose=args.debug_logging)

    version = importlib_metadata.version("colabfold")
    commit = get_commit()
    if commit:
        version += f" ({commit})"

    logger.info(f"Running colabfold {version}")

    data_dir = Path(args.data or default_data_dir)

    queries, is_complex = get_queries(args.input, args.sort_queries_by)

    has_per_entry_templates = any(isinstance(q[3], Path) for q in queries)
    if has_per_entry_templates and args.custom_template_cache_path is None:
        raise ValueError("--custom-template-cache-path must be set when using per-entry template paths in CSV input")
    if has_per_entry_templates and args.custom_template_path is not None:
        raise ValueError("--custom-template-path and per-entry template paths in CSV input cannot be used simultaneously")

    model_type = set_model_type(is_complex, args.model_type)

    # use pdb or cif input as initial guess
    if args.initial_guess is not None:
        if isinstance(args.initial_guess, str) and Path(args.initial_guess).suffix in (".pdb", ".cif"):
            initial_guess = args.initial_guess
        elif Path(args.input).suffix in (".pdb", ".cif"):
            initial_guess = args.input
        else:
            raise ValueError("Provide PDB or CIF file for initial guess.")
    else:
        initial_guess = None

    if args.msa_only:
        args.num_models = 0

    if args.num_models > 0:
        download_alphafold_params(model_type, data_dir)

    if args.msa_mode != "single_sequence" and not args.templates:
        uses_api = any((query[2] is None for query in queries))
        if uses_api and args.host_url == DEFAULT_API_SERVER:
            print(ACCEPT_DEFAULT_TERMS, file=sys.stderr)

    model_order = [int(i) for i in args.model_order.split(",")]

    assert args.recompile_padding >= 0, "Can't apply negative padding"

    # backward compatibility
    if args.amber and args.num_relax == 0:
        args.num_relax = args.num_models * args.num_seeds

    # added for actifptm calculation
    use_probs_extra = False if args.no_use_probs_extra else True

    user_agent = f"colabfold/{version}"

    if args.af3_json:
        generate_af3_input(
            queries=queries,
            result_dir=args.results,
            msa_mode=args.msa_mode,
            pair_mode=args.pair_mode,
            pairing_strategy=args.pair_strategy,
            use_templates=args.templates,
            custom_template_path=args.custom_template_path,
            jobname_prefix=args.jobname_prefix,
            host_url=args.host_url,
            user_agent=user_agent,
            max_template_date=args.max_template_date,
            max_template_hits=args.max_template_hits,
            # extra_molecules=extra_molecules,
        )
        return
    
    run(
        queries=queries,
        result_dir=args.results,
        use_templates=args.templates,
        custom_template_path=args.custom_template_path,
        custom_template_cache_path=args.custom_template_cache_path,
        num_relax=args.num_relax,
        relax_max_iterations=args.relax_max_iterations,
        relax_tolerance=args.relax_tolerance,
        relax_stiffness=args.relax_stiffness,
        relax_max_outer_iterations=args.relax_max_outer_iterations,
        msa_mode=args.msa_mode,
        model_type=model_type,
        num_models=args.num_models,
        num_recycles=args.num_recycle,
        recycle_early_stop_tolerance=args.recycle_early_stop_tolerance,
        num_ensemble=args.num_ensemble,
        model_order=model_order,
        initial_guess=initial_guess,
        is_complex=is_complex,
        keep_existing_results=not args.overwrite_existing_results,
        rank_by=args.rank,
        pair_mode=args.pair_mode,
        pairing_strategy=args.pair_strategy,
        data_dir=data_dir,
        host_url=args.host_url,
        user_agent=user_agent,
        random_seed=args.random_seed,
        num_seeds=args.num_seeds,
        stop_at_score=args.stop_at_score,
        recompile_padding=args.recompile_padding,
        zip_results=args.zip,
        save_single_representations=args.save_single_representations,
        save_pair_representations=args.save_pair_representations,
        skip_output=args.skip_output,
        use_dropout=args.use_dropout,
        max_seq=args.max_seq,
        max_extra_seq=args.max_extra_seq,
        max_msa=args.max_msa,
        pdb_hit_file=args.pdb_hit_file,
        local_pdb_path=args.local_pdb_path,
        use_cluster_profile=not args.disable_cluster_profile,
        use_gpu_relax = args.use_gpu_relax,
        jobname_prefix=args.jobname_prefix,
        save_all=args.save_all,
        save_recycles=args.save_recycles,
        calc_extra_ptm=args.calc_extra_ptm,
        use_probs_extra=use_probs_extra,
        max_template_date=args.max_template_date,
        max_template_hits=args.max_template_hits,
        use_pallas=args.use_pallas,
        compile_mode=args.compile_mode,
    )

if __name__ == "__main__":
    main()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/results/af2_20261027_172813_247681/benchmark_af2.py — 원문 전문 (10,754 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/benchmark_af2.py", "sha256": "626767be4e7429cd4bb57737e77bba58877133c44dcbe9760e8c8845ff0bf323", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Instrument existing ColabFold; serialize all GPU work; explicit AOT compile timing."""
import os,sys,pathlib,time,json,threading,subprocess,csv,traceback,inspect,functools,datetime,hashlib,shutil,resource
import numpy as np,jax
from colabfold import batch
from colabfold.alphafold import models
from alphafold.model import model as afmodel
from colabfold.input import get_queries
from colabfold.utils import setup_logging
R=pathlib.Path('/work');O=R/'results'/sys.argv[1];O.mkdir(exist_ok=True);MSA=R/'results'/sys.argv[2]/'msa';phase='setup';current='setup';stop=threading.Event();samples=[];predictions=[];calls=[];events=[];query_stats=[];allstart=time.perf_counter()
shutil.copy2(__file__,O/'benchmark_af2.py')
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def event(kind,**kw):
 row=dict(time_unix=time.time(),kind=kind,query=current,phase=phase,**kw);events.append(row)
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
 print('BENCH',json.dumps(row,default=str),flush=True)
def memory():
 d={}
 for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
  k,v=line.split(':');d[k]=int(v.strip().split()[0])*1024
 rss=next(int(l.split()[1])*1024 for l in pathlib.Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'))
 return rss,d['MemAvailable'],d['SwapTotal']-d['SwapFree'],d['MemTotal']
initial_swap=memory()[2];dev=jax.devices('gpu')[0]
def dmem():
 try:return {k:int(v) for k,v in (dev.memory_stats() or {}).items()}
 except Exception:return {}
def monitor():
 with (O/'resources.csv').open('w') as f:
  fields=['query_start','query_end','query','phase','gpu_util_percent','rss_bytes','system_available_bytes','swap_used_bytes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  while not stop.is_set():
   start=time.time();ph=phase;qid=current
   try:
    p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3);v=float(p.stdout.strip()) if p.returncode==0 else None
   except Exception:v=None
   rss,available,swap,total=memory();row=dict(query_start=start,query_end=time.time(),query=qid,phase=ph,gpu_util_percent=v,rss_bytes=rss,system_available_bytes=available,swap_used_bytes=swap);samples.append(row);w.writerow(row);f.flush()
   if available<8*2**30 or swap>initial_swap+2*2**30:
    event('memory_pressure_stop',available_bytes=available,swap_bytes=swap);os._exit(70)
   stop.wait(max(0,.5-(time.time()-start)))
thread=threading.Thread(target=monitor,daemon=True);thread.start()
summary=dict(status='running',run=str(O),settings=dict(model_type='alphafold2_ptm',num_models=5,model_order=[1,2,3,4,5],num_recycles=3,recycle_early_stop_tolerance=0.0,num_ensemble=1,num_seeds=1,random_seed=42,use_templates=False,num_relax=0,dropout=False,msa_mode='mmseqs2_uniref_env',max_seq=512,max_extra_seq=5120,use_bfloat16=True,use_pallas=False,compile_mode='tuned',recompile_padding=10,batch='one protein at a time, five weight sets sequentially',preallocation=False),predictions=predictions)
write('summary.json',summary)
class TimedJit:
 def __init__(self,fn):self.fn=fn;self.cache={}
 def __getattr__(self,key):return getattr(self.fn,key)
 def __call__(self,*args,**kw):
  global phase
  leaves,tree=jax.tree_util.tree_flatten((args,kw));signature=(str(tree),tuple((tuple(getattr(x,'shape',())),str(getattr(x,'dtype',type(x)))) for x in leaves));oldphase=phase
  if signature not in self.cache:
   phase='compilation';t=time.perf_counter();start=time.time();compiled=self.fn.lower(*args,**kw).compile();seconds=time.perf_counter()-t;self.cache[signature]=compiled
   try:analysis=str(compiled.memory_analysis())
   except Exception:analysis=None
   event('compilation',start_unix=start,end_unix=time.time(),seconds=seconds,memory_analysis=analysis,shape_variants=len(self.cache))
  phase='execution';start=time.time();t=time.perf_counter();out=self.cache[signature](*args,**kw);jax.block_until_ready(out);seconds=time.perf_counter()-t;end=time.time();row=dict(query=current,start_unix=start,end_unix=end,seconds=seconds,device_memory=dmem());calls.append(row);event('gpu_apply',**{k:v for k,v in row.items() if k!='query'});phase=oldphase;return out
original_init=afmodel.RunModel.__init__
def timed_init(self,*args,**kw):original_init(self,*args,**kw);self.apply=TimedJit(self.apply)
afmodel.RunModel.__init__=timed_init
origload=models.load_models_and_params
@functools.wraps(origload)
def load(*args,**kw):
 global phase
 old=phase;phase='model_loading';t=time.perf_counter();out=origload(*args,**kw);jax.block_until_ready([p for n,r,p in out]);seconds=time.perf_counter()-t;counts=[dict(model=n,parameters=sum(int(x.size) for x in jax.tree_util.tree_leaves(p)),parameter_dtypes=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(p)})) for n,r,p in out];event('model_loading',seconds=seconds,models=counts);summary['loaded_models']=counts;summary['model_loading_seconds']=seconds;phase=old;return out
models.load_models_and_params=load
# Time host preprocessing separately; never alter returned features.
def timed_function(obj,name,label):
 old=getattr(obj,name)
 @functools.wraps(old)
 def fn(*a,**kw):
  global phase,current
  if name=='generate_input_feature':
   bound=inspect.signature(old).bind_partial(*a,**kw);seqs=bound.arguments.get('query_seqs_unique',[]);seq=''.join(seqs) if isinstance(seqs,list) else str(seqs)
   current=next((q['id'] for q in json.loads((R/'data/input_manifest.json').read_text())['queries'] if q['sequence']==seq),current)
  ph=phase;phase=label;t=time.perf_counter();result=old(*a,**kw);event(label,function=name,seconds=time.perf_counter()-t);phase=ph;return result
 setattr(obj,name,fn)
for name in ['unserialize_msa','generate_input_feature']:timed_function(batch,name,'preprocessing')
timed_function(afmodel.RunModel,'process_features','preprocessing')
origpredict=afmodel.RunModel.predict
@functools.wraps(origpredict)
def predict(self,feat,*args,**kw):
 global phase
 phase='prediction';start=time.perf_counter();event('actual_model_input',shapes={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in feat.items() if hasattr(v,'shape')},bfloat16=bool(self.config.model.global_config.bfloat16),training=False)
 result=origpredict(self,feat,*args,**kw);event('predict_total',seconds=time.perf_counter()-start,recycles=int(result[1]));phase='output';return result
afmodel.RunModel.predict=predict
origstructure=batch.predict_structure
@functools.wraps(origstructure)
def structure(*a,**kw):
 global current,phase
 bound=inspect.signature(origstructure).bind_partial(*a,**kw);current=str(bound.arguments['prefix']);phase='query';t=time.perf_counter();start=time.time();res=origstructure(*a,**kw);end=time.time();stat=dict(query=current,wall_seconds=time.perf_counter()-t,start_unix=start,end_unix=end);query_stats.append(stat);event('query_complete',**{k:v for k,v in stat.items() if k!='query'});write('query_stats.json',query_stats);phase='between_queries';return res
batch.predict_structure=structure
# Track explicit text output writes as a lower bound on serialization time.
origwrite=pathlib.Path.write_text
@functools.wraps(origwrite)
def timedwrite(self,*args,**kw):
 t=time.perf_counter();result=origwrite(self,*args,**kw)
 if str(self).startswith(str(O)) and self.suffix in ['.pdb','.json','.a3m'] and self.name not in ['summary.json','query_stats.json','input_report.json']:
  event('file_write',path=str(self),seconds=time.perf_counter()-t)
 return result
pathlib.Path.write_text=timedwrite

def callback(protein,lengths,result,features,tag):
 row=dict(query=current,tag=tag[0],lengths=list(map(int,lengths)),mean_plddt=float(np.mean(result['plddt'][:sum(lengths)])),ptm=float(result['ptm']) if 'ptm' in result else None,plddt_shape=list(result['plddt'].shape),pae_shape=list(result['predicted_aligned_error'].shape) if 'predicted_aligned_error' in result else None,coordinates_shape=list(protein.atom_positions.shape),finite_coordinates=bool(np.isfinite(protein.atom_positions).all()),finite_plddt=bool(np.isfinite(result['plddt']).all()))
 assert row['finite_coordinates'] and row['finite_plddt'];predictions.append(row);event('validated_prediction',**{k:v for k,v in row.items() if k!='query'});write('predictions.json',predictions)
try:
 setup_logging(O/'log.txt');manifest=json.loads((R/'data/input_manifest.json').read_text());queries=[];report=[]
 for q in manifest['queries']:
  p=MSA/(q['id']+'.a3m');text=p.read_text();seqs=[]
  for line in text.splitlines():
   if line.startswith(('>','#')):continue
   if line.strip():seqs.append(''.join(c for c in line if not c.islower()))
  assert seqs and seqs[0]==q['sequence'];assert all(len(s)==q['length'] for s in seqs)
  report.append(dict(id=q['id'],residues=q['length'],msa_depth=len(seqs),msa_aligned_columns=len(seqs[0]),msa_characters=sorted(set(''.join(seqs))),msa_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),msa_bytes=p.stat().st_size,token_dtype='int32 features',truncated=False,templates=False))
  queries.append((q['id'],q['sequence'],[text],None))
 write('input_report.json',report);event('input_verified',queries=report);event('initial_memory',rss_bytes=memory()[0],available_bytes=memory()[1],system_total_bytes=memory()[3],gpu=dmem())
 phase='pipeline';start=time.perf_counter()
 batch.run(queries=queries,result_dir=O/'predictions',num_models=5,is_complex=False,num_recycles=3,recycle_early_stop_tolerance=0.0,model_order=[1,2,3,4,5],num_ensemble=1,model_type='alphafold2_ptm',msa_mode='mmseqs2_uniref_env',use_templates=False,num_relax=0,keep_existing_results=False,data_dir=pathlib.Path('/cache'),random_seed=42,num_seeds=1,recompile_padding=10,prediction_callback=callback,feature_dict_callback=lambda feat:event('raw_feature_shapes',shapes={k:list(v.shape) for k,v in feat.items() if hasattr(v,'shape')}),use_bfloat16=True,use_pallas=False,compile_mode='tuned',max_seq=512,max_extra_seq=5120)
 summary['pipeline_wall_seconds']=time.perf_counter()-start;assert len(predictions)==5*len(queries),(len(predictions),len(queries));summary['status']='success';summary['gpu_apply_seconds']=sum(c['seconds'] for c in calls);summary['compilation_seconds']=sum(e['seconds'] for e in events if e['kind']=='compilation');summary['final_device_memory']=dmem();summary['wall_seconds']=time.perf_counter()-allstart;write('summary.json',summary);write('apply_calls.json',calls);event('success',summary_path=str(O/'summary.json'))
except Exception as e:
 summary['status']='failed';summary['error']=str(e);summary['traceback']=traceback.format_exc();write('summary.json',summary);event('failed',error=str(e));traceback.print_exc();raise
finally:stop.set();thread.join(timeout=5)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/results/af2_20261027_172609_696571/benchmark_af2.py — 원문 전문 (10,416 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172609_696571/benchmark_af2.py", "sha256": "a89e96d7103c3b3784dff3dfc1b012b42a25c73b6616771eaba2af510382aa04", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Instrument existing ColabFold; serialize all GPU work; explicit AOT compile timing."""
import os,sys,pathlib,time,json,threading,subprocess,csv,traceback,inspect,functools,datetime,hashlib,shutil,resource
import numpy as np,jax
from colabfold import batch
from colabfold.alphafold import models
from alphafold.model import model as afmodel
from colabfold.input import get_queries
from colabfold.utils import setup_logging
R=pathlib.Path('/work');O=R/'results'/sys.argv[1];O.mkdir(exist_ok=True);MSA=R/'results'/sys.argv[2]/'msa';phase='setup';current='setup';stop=threading.Event();samples=[];predictions=[];calls=[];events=[];query_stats=[];allstart=time.perf_counter()
shutil.copy2(__file__,O/'benchmark_af2.py')
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def event(kind,**kw):
 row=dict(time_unix=time.time(),kind=kind,query=current,phase=phase,**kw);events.append(row)
 with (O/'events.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
 print('BENCH',json.dumps(row,default=str),flush=True)
def memory():
 d={}
 for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
  k,v=line.split(':');d[k]=int(v.strip().split()[0])*1024
 rss=next(int(l.split()[1])*1024 for l in pathlib.Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'))
 return rss,d['MemAvailable'],d['SwapTotal']-d['SwapFree'],d['MemTotal']
initial_swap=memory()[2];dev=jax.devices('gpu')[0]
def dmem():
 try:return {k:int(v) for k,v in (dev.memory_stats() or {}).items()}
 except Exception:return {}
def monitor():
 with (O/'resources.csv').open('w') as f:
  fields=['query_start','query_end','query','phase','gpu_util_percent','rss_bytes','system_available_bytes','swap_used_bytes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  while not stop.is_set():
   start=time.time();ph=phase;qid=current
   try:
    p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3);v=float(p.stdout.strip()) if p.returncode==0 else None
   except Exception:v=None
   rss,available,swap,total=memory();row=dict(query_start=start,query_end=time.time(),query=qid,phase=ph,gpu_util_percent=v,rss_bytes=rss,system_available_bytes=available,swap_used_bytes=swap);samples.append(row);w.writerow(row);f.flush()
   if available<8*2**30 or swap>initial_swap+2*2**30:
    event('memory_pressure_stop',available_bytes=available,swap_bytes=swap);os._exit(70)
   stop.wait(max(0,.5-(time.time()-start)))
thread=threading.Thread(target=monitor,daemon=True);thread.start()
summary=dict(status='running',run=str(O),settings=dict(model_type='alphafold2_ptm',num_models=5,model_order=[1,2,3,4,5],num_recycles=3,recycle_early_stop_tolerance=0.0,num_ensemble=1,num_seeds=1,random_seed=42,use_templates=False,num_relax=0,dropout=False,msa_mode='mmseqs2_uniref_env',max_seq=512,max_extra_seq=5120,use_bfloat16=True,use_pallas=False,compile_mode='tuned',recompile_padding=10,batch='one protein at a time, five weight sets sequentially',preallocation=False),predictions=predictions)
write('summary.json',summary)
class TimedJit:
 def __init__(self,fn):self.fn=fn;self.cache={}
 def __getattr__(self,key):return getattr(self.fn,key)
 def __call__(self,*args,**kw):
  global phase
  leaves,tree=jax.tree_util.tree_flatten((args,kw));signature=(str(tree),tuple((tuple(getattr(x,'shape',())),str(getattr(x,'dtype',type(x)))) for x in leaves));oldphase=phase
  if signature not in self.cache:
   phase='compilation';t=time.perf_counter();start=time.time();compiled=self.fn.lower(*args,**kw).compile();seconds=time.perf_counter()-t;self.cache[signature]=compiled
   try:analysis=str(compiled.memory_analysis())
   except Exception:analysis=None
   event('compilation',start_unix=start,end_unix=time.time(),seconds=seconds,memory_analysis=analysis,shape_variants=len(self.cache))
  phase='execution';start=time.time();t=time.perf_counter();out=self.cache[signature](*args,**kw);jax.block_until_ready(out);seconds=time.perf_counter()-t;end=time.time();row=dict(query=current,start_unix=start,end_unix=end,seconds=seconds,device_memory=dmem());calls.append(row);event('gpu_apply',**{k:v for k,v in row.items() if k!='query'});phase=oldphase;return out
original_init=afmodel.RunModel.__init__
def timed_init(self,*args,**kw):original_init(self,*args,**kw);self.apply=TimedJit(self.apply)
afmodel.RunModel.__init__=timed_init
origload=models.load_models_and_params
@functools.wraps(origload)
def load(*args,**kw):
 global phase
 old=phase;phase='model_loading';t=time.perf_counter();out=origload(*args,**kw);jax.block_until_ready([p for n,r,p in out]);seconds=time.perf_counter()-t;counts=[dict(model=n,parameters=sum(int(x.size) for x in jax.tree_util.tree_leaves(p)),parameter_dtypes=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(p)})) for n,r,p in out];event('model_loading',seconds=seconds,models=counts);summary['loaded_models']=counts;summary['model_loading_seconds']=seconds;phase=old;return out
models.load_models_and_params=load
# Time host preprocessing separately; never alter returned features.
def timed_function(obj,name,label):
 old=getattr(obj,name)
 @functools.wraps(old)
 def fn(*a,**kw):
  global phase
  ph=phase;phase=label;t=time.perf_counter();result=old(*a,**kw);event(label,function=name,seconds=time.perf_counter()-t);phase=ph;return result
 setattr(obj,name,fn)
for name in ['unserialize_msa','generate_input_feature']:timed_function(batch,name,'preprocessing')
timed_function(afmodel.RunModel,'process_features','preprocessing')
origpredict=afmodel.RunModel.predict
@functools.wraps(origpredict)
def predict(self,feat,*args,**kw):
 global phase
 phase='prediction';start=time.perf_counter();event('actual_model_input',shapes={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in feat.items() if hasattr(v,'shape')},bfloat16=bool(self.config.model.global_config.bfloat16),training=False)
 result=origpredict(self,feat,*args,**kw);event('predict_total',seconds=time.perf_counter()-start,recycles=int(result[1]));phase='output';return result
afmodel.RunModel.predict=predict
origstructure=batch.predict_structure
@functools.wraps(origstructure)
def structure(*a,**kw):
 global current,phase
 bound=inspect.signature(origstructure).bind_partial(*a,**kw);current=str(bound.arguments['prefix']);phase='query';t=time.perf_counter();start=time.time();res=origstructure(*a,**kw);end=time.time();stat=dict(query=current,wall_seconds=time.perf_counter()-t,start_unix=start,end_unix=end);query_stats.append(stat);event('query_complete',**{k:v for k,v in stat.items() if k!='query'});write('query_stats.json',query_stats);phase='between_queries';return res
batch.predict_structure=structure
# Track explicit text output writes as a lower bound on serialization time.
origwrite=pathlib.Path.write_text
@functools.wraps(origwrite)
def timedwrite(self,*args,**kw):
 t=time.perf_counter();result=origwrite(self,*args,**kw)
 if str(self).startswith(str(O)) and self.suffix in ['.pdb','.json','.a3m'] and self.name not in ['summary.json','query_stats.json','input_report.json']:
  event('file_write',path=str(self),seconds=time.perf_counter()-t)
 return result
pathlib.Path.write_text=timedwrite

def callback(protein,lengths,result,features,tag):
 row=dict(query=current,tag=tag[0],lengths=list(map(int,lengths)),mean_plddt=float(np.mean(result['plddt'][:sum(lengths)])),ptm=float(result['ptm']) if 'ptm' in result else None,plddt_shape=list(result['plddt'].shape),pae_shape=list(result['predicted_aligned_error'].shape) if 'predicted_aligned_error' in result else None,coordinates_shape=list(protein.atom_positions.shape),finite_coordinates=bool(np.isfinite(protein.atom_positions).all()),finite_plddt=bool(np.isfinite(result['plddt']).all()))
 assert row['finite_coordinates'] and row['finite_plddt'];predictions.append(row);event('validated_prediction',**{k:v for k,v in row.items() if k!='query'});write('predictions.json',predictions)
try:
 setup_logging(O/'log.txt');manifest=json.loads((R/'data/input_manifest.json').read_text());queries=[];report=[]
 for q in manifest['queries']:
  p=MSA/(q['id']+'.a3m');text=p.read_text();seqs=[]
  for line in text.splitlines():
   if line.startswith(('>','#')):continue
   if line.strip():seqs.append(''.join(c for c in line if not c.islower()))
  assert seqs and seqs[0]==q['sequence'];assert all(len(s)==q['length'] for s in seqs)
  report.append(dict(id=q['id'],residues=q['length'],msa_depth=len(seqs),msa_aligned_columns=len(seqs[0]),msa_characters=sorted(set(''.join(seqs))),msa_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),msa_bytes=p.stat().st_size,token_dtype='int32 features',truncated=False,templates=False))
  queries.append((q['id'],q['sequence'],[text],None))
 write('input_report.json',report);event('input_verified',queries=report);event('initial_memory',rss_bytes=memory()[0],available_bytes=memory()[1],system_total_bytes=memory()[3],gpu=dmem())
 phase='pipeline';start=time.perf_counter()
 batch.run(queries=queries,result_dir=O/'predictions',num_models=5,is_complex=False,num_recycles=3,recycle_early_stop_tolerance=0.0,model_order=[1,2,3,4,5],num_ensemble=1,model_type='alphafold2_ptm',msa_mode='mmseqs2_uniref_env',use_templates=False,num_relax=0,keep_existing_results=False,data_dir=pathlib.Path('/cache'),random_seed=42,num_seeds=1,recompile_padding=10,prediction_callback=callback,feature_dict_callback=lambda feat:event('raw_feature_shapes',shapes={k:list(v.shape) for k,v in feat.items() if hasattr(v,'shape')}),use_bfloat16=True,use_pallas=False,compile_mode='tuned',max_seq=512,max_extra_seq=5120)
 summary['pipeline_wall_seconds']=time.perf_counter()-start;assert len(predictions)==5*len(queries),(len(predictions),len(queries));summary['status']='success';summary['gpu_apply_seconds']=sum(c['seconds'] for c in calls);summary['compilation_seconds']=sum(e['seconds'] for e in events if e['kind']=='compilation');summary['final_device_memory']=dmem();summary['wall_seconds']=time.perf_counter()-allstart;write('summary.json',summary);write('apply_calls.json',calls);event('success',summary_path=str(O/'summary.json'))
except Exception as e:
 summary['status']='failed';summary['error']=str(e);summary['traceback']=traceback.format_exc();write('summary.json',summary);event('failed',error=str(e));traceback.print_exc();raise
finally:stop.set();thread.join(timeout=5)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>colabfold/results/vmap_batch_retry_20261028_190537_289704/source_worker.py — 원문 전문 (8,498 bytes)</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/source_worker.py", "sha256": "1a123408b7acb82b46c8643e3ab9070e5c539b3064ec06f6450eaa86919eae3d", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,time,json,threading,csv,subprocess,gc
import numpy as np,jax,jax.numpy as jnp
from colabfold.alphafold.models import load_models_and_params
from alphafold.common import protein
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];B=int(os.environ['BATCH']);C=D/f'batch_{B}';C.mkdir();dev=jax.devices('gpu')[0];stop=threading.Event();rows=[];phase='loading';G=2**30
S={'status':'running','batch':B,'implementation':'experimental jax.vmap over unmodified ColabFold RunModel.apply; shared fixed model_1 weights, independent actual query features and RNG; not a native CLI batch option','conditions':[]}
def write():(C/'summary.json').write_text(json.dumps(S,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def mem():
 info={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};rss=next(int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'));return {'rss_gib':rss/G,'available_gib':info['MemAvailable']/G,'total_gib':info['MemTotal']/G,'swap_gib':(info['SwapTotal']-info['SwapFree'])/G}
def dmem():return {k:int(v) for k,v in (dev.memory_stats() or {}).items()}
swap0=mem()['swap_gib']
def monitor():
 with (C/'resources.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['query_start','query_end','phase','gpu_util','rss_gib','available_gib','swap_gib','jax_bytes_in_use']);w.writeheader()
  while not stop.is_set():
   t=time.time();ph=phase
   try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip())
   except Exception:u=None
   m=mem();r={'query_start':t,'query_end':time.time(),'phase':ph,'gpu_util':u,'rss_gib':m['rss_gib'],'available_gib':m['available_gib'],'swap_gib':m['swap_gib'],'jax_bytes_in_use':dmem().get('bytes_in_use')};rows.append(r);w.writerow(r);f.flush()
   if m['available_gib']<16 or m['swap_gib']>swap0+2:(C/'memory_guard.json').write_text(json.dumps(r));os._exit(86)
   stop.wait(.5)
th=threading.Thread(target=monitor,daemon=True);th.start()
def aligned_rmsd(x,y):
 x=x-x.mean(0);y=y-y.mean(0);u,s,v=np.linalg.svd(x.T@y);q=np.eye(3);q[-1,-1]=np.linalg.det(u@v);return float(np.sqrt(np.mean(np.sum((x@(u@q@v)-y)**2,axis=1))))
try:
 audit=json.loads((D/'input_audit.json').read_text());queries=audit['queries'];allfeat=[dict(np.load(D/(q['id']+'_processed_features.npz'))) for q in queries];keys=set(allfeat[0]);assert all(set(f)==keys for f in allfeat);maxshape={k:tuple(max(f[k].shape[d] for f in allfeat) for d in range(allfeat[0][k].ndim)) for k in keys};features=[]
 for f in allfeat:
  features.append({k:np.pad(v,[(0,maxshape[k][d]-v.shape[d]) for d in range(v.ndim)],mode='constant') for k,v in f.items()})
 selected=features[:B];S['input']={'unique_queries':B,'query_ids':[q['id'] for q in queries[:B]],'actual_residues':[q['length'] for q in queries[:B]],'feature_shapes_per_query':{k:list(v.shape) for k,v in features[0].items()},'padding':'pool-wide max feature shapes; zero-padded extra-MSA/cluster entries masked by zero masks; no sequence duplication','actual_msa_valid_rows':[int((f['msa_mask'][0].sum(-1)>0).sum()) for f in selected]};t=time.time();loaded=load_models_and_params(num_models=1,use_templates=False,num_recycles=3,recycle_early_stop_tolerance=0.,num_ensemble=1,model_order=[1],model_type='alphafold2_ptm',data_dir=Path('/cache'),max_seq=512,max_extra_seq=5120,use_bfloat16=True,use_pallas=False,compile_mode='tuned');name,runner,params=loaded[0];runner.params=params;jax.block_until_ready(params);S['loading_seconds']=time.time()-t;S['parameters']=sum(int(x.size) for x in jax.tree_util.tree_leaves(params));S['weights_dtype']=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(params)});S['model_only_postload']={'jax':dmem(),'system':mem()};S['environment']={'jax':jax.__version__,'device':str(dev),'preallocate':os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),'compute':'original BF16 config, FP32 weights','recycles':3,'ensembles':1,'templates':False,'relaxation':False,'seed_per_query':42};write()
 full={k:jnp.asarray(np.stack([f[k] for f in selected])) for k in keys};L=int(selected[0]['aatype'].shape[1]);zero_prev={'prev_msa_first_row':jnp.zeros((B,L,256),jnp.float16),'prev_pair':jnp.zeros((B,L,L,128),jnp.float16),'prev_pos':jnp.zeros((B,L,37,3),jnp.float16)};rng=jnp.stack([jax.random.PRNGKey(42)]*B);first={k:v[:,0:1] for k,v in full.items()};first['prev']=zero_prev;vmapped=jax.jit(jax.vmap(runner.apply.__wrapped__,in_axes=(None,0,0)),compiler_options={'xla_gpu_override_gemm_autotuner':'block_m: 64 block_n: 64 block_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1'});phase='compilation';t=time.time();compiled=vmapped.lower(params,rng,first).compile();S['compilation_seconds']=time.time()-t;S['compiled_memory_analysis']=str(compiled.memory_analysis());write()
 def predict():
  prev=zero_prev;key=jnp.stack([jax.random.PRNGKey(42)]*B)
  for recycle in range(4):
   pair=jax.vmap(jax.random.split)(key);key=pair[:,0];subkey=pair[:,1];subfeat={k:v[:,recycle:recycle+1] for k,v in full.items()};subfeat['prev']=prev;result=compiled(params,subkey,subfeat);jax.block_until_ready(result);result=jax.tree_util.tree_map(lambda x:np.asarray(x,np.float16),result);prev={k:jnp.asarray(v) for k,v in result.pop('prev').items()}
  return result
 phase='warmup';t=time.time();result=predict();S['warmup_seconds']=time.time()-t;phase='execution';t=time.time();repeats=0
 while repeats<2 or time.time()-t<15:result=predict();repeats+=1
 end=time.time();stats=dmem();observed=[r for r in rows if r['query_start']>=t and r['query_end']<=end];utils=[r['gpu_util'] for r in observed if r['gpu_util'] is not None];S['conditions']=[{'batch':B,'unique_queries':B,'repeats':repeats,'processed_queries':B*repeats,'seconds':end-t,'queries_per_second':B*repeats/(end-t),'gpu_util_avg':float(np.mean(utils)) if utils else None,'gpu_util_max':max(utils) if utils else None,'sample_count':len(utils),'jax_memory_stats':stats,'memory_definition':'peak_bytes_in_use is cumulative process allocator peak including compilation and warmup; execution bytes_in_use samples separate, no unsupported reset','execution_sampled_peak_bytes_in_use':max([r['jax_bytes_in_use'] for r in observed if r['jax_bytes_in_use'] is not None],default=None),'rss_sampled_peak_gib':max([r['rss_gib'] for r in observed],default=None),'available_sampled_min_gib':min([r['available_gib'] for r in observed],default=None)}];phase='validation';preds=[]
 for i,q in enumerate(queries[:B]):
  single=jax.tree_util.tree_map(lambda x:x[i],result);coords=single['structure_module']['final_atom_positions'];plddt=single['plddt'];assert coords.shape==(L,37,3) and np.isfinite(coords).all() and np.isfinite(plddt).all();np.savez(C/(q['id']+'_prediction.npz'),atom_positions=coords,plddt=plddt);prot=protein.from_prediction(features=selected[i],result=single,b_factors=np.repeat(plddt[:,None],37,axis=1));pdb=protein.to_pdb(prot);(C/(q['id']+'.pdb')).write_text(pdb);assert sum(l.startswith('ATOM') and l[12:16].strip()=='CA' for l in pdb.splitlines())==q['length'];preds.append({'id':q['id'],'residues':q['length'],'mean_plddt':float(plddt.mean()),'finite_and_pdb_valid':True})
 S['predictions']=preds;reference=D/'batch_1'/ (queries[0]['id']+'_prediction.npz')
 if B==1:
  phase='single_native_reference';t0=time.time();ref,recycles=runner.predict(selected[0],random_seed=42);S['native_reference_seconds_excluded']=time.time()-t0;refcoords=ref['structure_module']['final_atom_positions'];S['native_reference_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(refcoords[:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max()),'native_recycles':int(recycles)};assert S['native_reference_comparison']['ca_aligned_rmsd_angstrom']<.5;np.savez(C/'native_single_reference.npz',atom_positions=refcoords,plddt=ref['plddt'])
 else:
  ref=np.load(reference);S['first_query_batch_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(ref['atom_positions'][:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max())}
 S['status']='complete';write();print('AF2_VMAP_COMPLETE',B,flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write();raise
finally:stop.set();th.join(timeout=5)
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

### record-f519caea368d

원래 파일: `colabfold/data/sources.json` · 원문 SHA256: `1004dd2b9badd383344dcfaefdc80a44cab1f70cd29ddbe1fd064fa764dc72e6`

<details>
<summary>colabfold/data/sources.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/data/sources.json", "sha256": "1004dd2b9badd383344dcfaefdc80a44cab1f70cd29ddbe1fd064fa764dc72e6", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "files": [
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1LYZ.fasta",
      "bytes": 190,
      "sha256": "bd71bd464c5247807690c913b1183b196c3ac97c76175d6c00a56a98cf6a8c50",
      "url": "https://www.rcsb.org/fasta/entry/1LYZ",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1LYZ.pdb",
      "bytes": 145962,
      "sha256": "28e95e819d4e0fc48acdf44209e37ba06f5c9d5d93d9950d67a760b32964326e",
      "url": "https://files.rcsb.org/download/1LYZ.pdb",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1TIM.fasta",
      "bytes": 315,
      "sha256": "4bd6b869b65f358356d971ffd95e2ce8bccf332c19446f67b124f35c71e4045d",
      "url": "https://www.rcsb.org/fasta/entry/1TIM",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1TIM.pdb",
      "bytes": 351783,
      "sha256": "79b35b419696a1e368aad2121ae2712e46444b5978de1e86bdc286f482222cdc",
      "url": "https://files.rcsb.org/download/1TIM.pdb",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1UBQ.fasta",
      "bytes": 123,
      "sha256": "f00b1a97f51c157133e1e2b3f3ed868aa2edf79591e8b5206a1bdd0f3b7c94d2",
      "url": "https://www.rcsb.org/fasta/entry/1UBQ",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1UBQ.pdb",
      "bytes": 78570,
      "sha256": "d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161",
      "url": "https://files.rcsb.org/download/1UBQ.pdb",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/3G5O.fasta",
      "bytes": 384,
      "sha256": "683c5fb83e2d8a7d2ce9d1ab790d362cbbd2d769b2a5c3ce68136cddd1db0821",
      "url": "https://www.rcsb.org/fasta/entry/3G5O",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/3G5O.pdb",
      "bytes": 294273,
      "sha256": "b18e40f567838e810fea65b1c83f93b8bd156e774705d89ba265ce0ef1164798",
      "url": "https://files.rcsb.org/download/3G5O.pdb",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/a3m_listing.json",
      "bytes": 2784,
      "sha256": "43525ad6c3c61f572677d162448ebb9272e57fb1f4b7078345764a66109ef5e6",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/batch_listing.json",
      "bytes": 2498,
      "sha256": "ac38f4f512acdc6ad60014197f231ec39e469c2e00a17ee609a7eab818645c8c",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/complex.fasta",
      "bytes": 197,
      "sha256": "b5e3fcbd9d0327489b05906830279b7f0356eaaec34ac97192827a25c1021702",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/complex_input.csv",
      "bytes": 225,
      "sha256": "847500a212bcd8a20c4737c5fd908e46af81a74eb7037032930a54176ffb295c",
      "url": "https://raw.githubusercontent.com/sokrypton/ColabFold/v1.6.2/test-data/complex/input.csv",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/complex_listing.json",
      "bytes": 1847,
      "sha256": "1310d8c8f8858e5b46f03e54f03cc8800e9b3344d7a22f07c57f62437708a333",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/complex_manifest.json",
      "bytes": 763,
      "sha256": "fa58017a4f6964f391a0782e93ea1a60a580b73145250fc103380fa319c23593",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/input_manifest.json",
      "bytes": 2679,
      "sha256": "4984b41e06e061e5bbfac95bd693c73e080c2be5b825c71d21d79d9c5a237ba9",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/queries.fasta",
      "bytes": 997,
      "sha256": "ad75f1a3dd58b303cf99ada9c2a42a8823acc2c86093ec746f4b000ae4ce9294",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_batch_5AWL_1_listing.json",
      "bytes": 2087,
      "sha256": "fe9283e08f5d623171aa55180ecf1dcfbd62946a2dacec91fc0e3ed0c7bbb118",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_batch_input_listing.json",
      "bytes": 2977,
      "sha256": "6c71103bb1c40b7efdcb04a27a146d293ae9af574bf83ca00fa26d3cb4ccd20b",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_complex_3G5O_A_3G5O_B_listing.json",
      "bytes": 2201,
      "sha256": "69c2bc9e0cc70893f5622ad490adec41cee737c56d57a84681e10af8dc4aadfa",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_mmseqs-api-reponses_listing.json",
      "bytes": 7616,
      "sha256": "1beee2fedfcdb9832f9975065d94d458a6598b84fff6a899500c5c0d1238b8c9",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_single_5AWL_1_listing.json",
      "bytes": 2099,
      "sha256": "ef33c852576bdff2f72d1dbb9840a091ac7d6f854a8566f6d152e62d309eee77",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    }
  ],
  "uniprot_parent_source": "/home/hyein/ai-benchmark/esm2/data/sources.json",
  "msa_service": "https://api.colabfold.com",
  "msa_mode": "mmseqs2_uniref_env",
  "server_database_revision": null
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-d891102ff9d3

원래 파일: `colabfold/results/af2_20261027_172609_696571/command.json` · 원문 SHA256: `0f3d43ff40cc0197ce7c2f1973d7fd25e24d62e4d3aa93318391e7072ebe21d3`

<details>
<summary>colabfold/results/af2_20261027_172609_696571/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172609_696571/command.json", "sha256": "0f3d43ff40cc0197ce7c2f1973d7fd25e24d62e4d3aa93318391e7072ebe21d3", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --name af2_20261027_172609_696571 --gpus all --network none --user 1000:1000 -e JAX_PLATFORMS=cuda -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e MPLCONFIGDIR=/work/results/af2_20261027_172609_696571/mplcache -v /home/hyein/ai-benchmark/colabfold:/work -v /home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 python -u /work/scripts/benchmark_af2.py af2_20261027_172609_696571 msa_20261027_172359_775627"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-96e58e815995

원래 파일: `colabfold/results/af2_20261027_172813_247681/command.json` · 원문 SHA256: `f6a82754f3156ad61690f725fd155583ddc8135c5218b470ca22b5c50359dab8`

<details>
<summary>colabfold/results/af2_20261027_172813_247681/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/command.json", "sha256": "f6a82754f3156ad61690f725fd155583ddc8135c5218b470ca22b5c50359dab8", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --name af2_20261027_172813_247681 --gpus all --network none --user 1000:1000 -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e MPLCONFIGDIR=/work/results/af2_20261027_172813_247681/mplcache -v /home/hyein/ai-benchmark/colabfold:/work -v /home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 python -u /work/scripts/benchmark_af2.py af2_20261027_172813_247681 msa_20261027_172359_775627"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-4a254dc280ae

원래 파일: `colabfold/results/af2_multimer_20261027_181640_431467/command.json` · 원문 SHA256: `05ecf1ac194852ce6d9973180e81edf806414c530c1411c34089922f87780eb1`

<details>
<summary>colabfold/results/af2_multimer_20261027_181640_431467/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_multimer_20261027_181640_431467/command.json", "sha256": "05ecf1ac194852ce6d9973180e81edf806414c530c1411c34089922f87780eb1", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --name af2_multimer_20261027_181640_431467 --gpus all --network none --user 1000:1000 -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e MPLCONFIGDIR=/work/results/af2_multimer_20261027_181640_431467/mplcache -v /home/hyein/ai-benchmark/colabfold:/work -v /home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 python -u /work/scripts/benchmark_af2.py af2_multimer_20261027_181640_431467 msa_complex_20261027_173649_931405 multimer"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-a917cbc6fc52

원래 파일: `colabfold/results/environment_verified/environment.json` · 원문 SHA256: `a083282cdcbcb0e329f3b5ef6cc9d1c058e8f97aa44bfa51b5f08c829c7ef47f`

<details>
<summary>colabfold/results/environment_verified/environment.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/environment_verified/environment.json", "sha256": "a083282cdcbcb0e329f3b5ef6cc9d1c058e8f97aa44bfa51b5f08c829c7ef47f", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "python": "3.12.12 | packaged by conda-forge | (main, Oct 22 2025, 23:16:53) [GCC 14.3.0]",
  "versions": {
    "colabfold": "1.6.2",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "dm-haiku": "0.0.16",
    "numpy": "2.5.1"
  },
  "devices": [
    "cuda:0"
  ],
  "environment": {
    "JAX_PLATFORMS": "cuda",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "TF_CPP_MIN_LOG_LEVEL": "1",
    "TF_FORCE_UNIFIED_MEMORY": "1",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0"
  },
  "run_signature": "(queries: 'List[Tuple[str, Union[str, List[str]], Optional[List[str]], Optional[List[Tuple[str, str, int]]]]]', result_dir: 'Union[str, Path]', num_models: 'int', is_complex: 'bool', num_recycles: 'Optional[int]' = None, recycle_early_stop_tolerance: 'Optional[float]' = None, model_order: 'List[int]' = [1, 2, 3, 4, 5], initial_guess: 'str' = None, num_ensemble: 'int' = 1, model_type: 'str' = 'auto', msa_mode: 'str' = 'mmseqs2_uniref_env', use_templates: 'bool' = False, custom_template_path: 'str' = None, custom_template_cache_path: 'str' = None, num_relax: 'int' = 0, relax_max_iterations: 'int' = 0, relax_tolerance: 'float' = 2.39, relax_stiffness: 'float' = 10.0, relax_max_outer_iterations: 'int' = 3, keep_existing_results: 'bool' = True, rank_by: 'str' = 'auto', pair_mode: 'str' = 'unpaired_paired', pairing_strategy: 'str' = 'greedy', data_dir: 'Union[str, Path]' = PosixPath('/cache/colabfold'), host_url: 'str' = 'https://api.colabfold.com', user_agent: 'str' = '', random_seed: 'int' = 0, num_seeds: 'int' = 1, recompile_padding: 'Union[int, float]' = 10, zip_results: 'bool' = False, prediction_callback: 'Callable[[Any, Any, Any, Any, Any], Any]' = None, save_single_representations: 'bool' = False, save_pair_representations: 'bool' = False, skip_output: 'List[str]' = [], jobname_prefix: 'Optional[str]' = None, save_all: 'bool' = False, save_recycles: 'bool' = False, use_dropout: 'bool' = False, use_gpu_relax: 'bool' = False, stop_at_score: 'float' = 100, dpi: 'int' = 200, max_seq: 'Optional[int]' = None, max_extra_seq: 'Optional[int]' = None, pdb_hit_file: 'Optional[Path]' = None, local_pdb_path: 'Optional[Path]' = None, use_cluster_profile: 'bool' = True, feature_dict_callback: 'Callable[[Any], Any]' = None, calc_extra_ptm: 'bool' = False, use_probs_extra: 'bool' = True, max_template_date: 'str' = '2100-01-01', max_template_hits: 'int' = 20, **kwargs)",
  "load_signature": "(num_models: int, use_templates: bool, num_recycles: Optional[int] = None, recycle_early_stop_tolerance: Optional[float] = None, num_ensemble: int = 1, model_order: Optional[List[int]] = None, model_type: str = '', data_dir: pathlib.Path = PosixPath('.'), stop_at_score: float = 100, rank_by: str = 'auto', max_seq: Optional[int] = None, max_extra_seq: Optional[int] = None, use_cluster_profile: bool = True, use_fuse: bool = True, use_bfloat16: bool = True, use_dropout: bool = False, save_all: bool = False, calc_extra_ptm: bool = False, use_probs_extra: bool = True, use_pallas: bool = False, compile_mode: str = 'tuned') -> List[Tuple[str, alphafold.model.model.RunModel, collections.abc.Mapping[str, collections.abc.Mapping[str, jax.Array]]]]"
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-7db6d4397295

원래 파일: `colabfold/results/msa_20261027_172359_775627/command.json` · 원문 SHA256: `380b7ba942ec309bc1be2b1d7aef81ee6563075845936312da6ba2df64d297e5`

<details>
<summary>colabfold/results/msa_20261027_172359_775627/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_20261027_172359_775627/command.json", "sha256": "380b7ba942ec309bc1be2b1d7aef81ee6563075845936312da6ba2df64d297e5", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --user 1000:1000 -e JAX_PLATFORMS=cuda -e SSL_CERT_FILE=/etc/ssl/certs/host-ca.crt -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/host-ca.crt -v /etc/ssl/certs/ca-certificates.crt:/etc/ssl/certs/host-ca.crt:ro -v /home/hyein/ai-benchmark/colabfold:/work -v /home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 colabfold_batch /work/data/queries.fasta /work/results/msa_20261027_172359_775627/msa --msa-only --data /cache --sort-queries-by length"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-4321b4b89e25

원래 파일: `colabfold/results/msa_complex_20261027_173649_931405/command.json` · 원문 SHA256: `ca377a53d691f51d0df0c294eae298b2993d8093e619de949b8c38f1cd8a70bf`

<details>
<summary>colabfold/results/msa_complex_20261027_173649_931405/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_complex_20261027_173649_931405/command.json", "sha256": "ca377a53d691f51d0df0c294eae298b2993d8093e619de949b8c38f1cd8a70bf", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --network bridge --user 1000:1000 -e JAX_PLATFORMS=cpu -e SSL_CERT_FILE=/etc/ssl/certs/host-ca.crt -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/host-ca.crt -v /etc/ssl/certs/ca-certificates.crt:/etc/ssl/certs/host-ca.crt:ro -v /home/hyein/ai-benchmark/colabfold:/work -v /home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 colabfold_batch /work/data/complex.fasta /work/results/msa_complex_20261027_173649_931405/msa --msa-only --model-type alphafold2_multimer_v3 --data /cache"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-4bb8424a9b6d

원래 파일: `colabfold/results/vmap_batch_20261028_185758_495006/batch1_command.json` · 원문 SHA256: `57590c787aadfcd194e74c93df19506c8b2c8e4d5f35af183c47d468faeb9d2f`

<details>
<summary>colabfold/results/vmap_batch_20261028_185758_495006/batch1_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/batch1_command.json", "sha256": "57590c787aadfcd194e74c93df19506c8b2c8e4d5f35af183c47d468faeb9d2f", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "af2-vmap-vmap_batch_20261028_185758_495006-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "RUN_ID=vmap_batch_20261028_185758_495006",
  "-e",
  "BATCH=1",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_batch_20261028_185758_495006/mplcache",
  "-v",
  "/home/hyein/ai-benchmark/colabfold:/work",
  "-v",
  "/home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro",
  "ghcr.io/sokrypton/colabfold:1.6.2-cuda13",
  "python",
  "-u",
  "/work/scripts/batch_followup/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-cb0e0c6f0f0e

원래 파일: `colabfold/results/vmap_batch_20261028_185758_495006/capture_command.json` · 원문 SHA256: `16fefa7b2e2667dc7cc32e01c60a60d2e01f742eff1c54df62a859b7622cca37`

<details>
<summary>colabfold/results/vmap_batch_20261028_185758_495006/capture_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/capture_command.json", "sha256": "16fefa7b2e2667dc7cc32e01c60a60d2e01f742eff1c54df62a859b7622cca37", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "JAX_PLATFORMS=cpu",
  "-e",
  "RUN_ID=vmap_batch_20261028_185758_495006",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_batch_20261028_185758_495006/mplcache",
  "-v",
  "/home/hyein/ai-benchmark/colabfold:/work",
  "-v",
  "/home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro",
  "ghcr.io/sokrypton/colabfold:1.6.2-cuda13",
  "python",
  "-u",
  "/work/scripts/batch_followup/capture.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-038429aff83f

원래 파일: `colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json` · 원문 SHA256: `34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d`

<details>
<summary>colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json", "sha256": "34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "queries": [
    {
      "id": "count1_A2RU14",
      "sequence": "MAGTVLGVGAGVFILALLWVAVLLLCVLLSRASGAARFSVIFLFFGAVIITSVLLLFPRAGEFPAPEVEVKIVDDFFIGRYVLLAFLSAIFLGGLFLVLIHYVLEPIYAKPLHSY",
      "length": 115,
      "msa_depth_before_feature_sampling": 568,
      "msa_sha256": "65b27286d30511570201072127c6c4bf4deb6d09d4b4f1fbc948e7d7faaaf5ea"
    },
    {
      "id": "count2_A6NFY7",
      "sequence": "MSRHSRLQRQVLSLYRDLLRAGRGKPGAEARVRAEFRQHAGLPRSDVLRIEYLYRRGRRQLQLLRSGHATAMGAFVRPRAPTGEPGGVGCQPDDGDSPRNPHDSTGAPETRPDGR",
      "length": 115,
      "msa_depth_before_feature_sampling": 2382,
      "msa_sha256": "50133c0bebf212ec0461ad8d17aa6b90256b188aec35e7b01738c23761643560"
    },
    {
      "id": "count3_O14519",
      "sequence": "MSYKPNLAAHMPAAALNAAGSVHSPSTSMATSSQYRQLLSDYGPPSLGYTQGTGNSQVPQSKYAELLAIIEELGKEIRPTYAGSKSAMERLKRGIIHARGLVRECLAETERNARS",
      "length": 115,
      "msa_depth_before_feature_sampling": 1295,
      "msa_sha256": "eda7682338958cb5814920cb653b6c7b32d974298f72fada8d1100f371bfc76d"
    },
    {
      "id": "count4_P01270",
      "sequence": "MIPAKDMAKVMIVMLAICFLTKSDGKSVKKRSVSEIQLMHNLGKHLNSMERVEWLRKKLQDVHNFVALGAPLAPRDAGSQRPRKKEDNVLVESHEKSLGEADKADVNVLTKAKSQ",
      "length": 115,
      "msa_depth_before_feature_sampling": 971,
      "msa_sha256": "b08538aaaf14fecf190c2233a0e9adc1e887b79cef16364fc0c0134d14af6c53"
    }
  ],
  "unique_queries": 4,
  "features": "official ColabFold processed features captured before predict, JAX_PLATFORMS=cpu during capture",
  "preset": "alphafold2_ptm model_1, actual runner shared model3 architecture without templates",
  "num_recycles": 3,
  "num_ensemble": 1,
  "max_seq": 512,
  "max_extra_seq": 5120,
  "templates": false,
  "relaxation": false,
  "seed": 42
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-fac40a0d65de

원래 파일: `colabfold/results/vmap_batch_retry_20261028_190537_289704/batch1_command.json` · 원문 SHA256: `51ee18f08e6f1d05d975f908caaf4ed2f3ea9a816aa4ed03f480655df1556dbc`

<details>
<summary>colabfold/results/vmap_batch_retry_20261028_190537_289704/batch1_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/batch1_command.json", "sha256": "51ee18f08e6f1d05d975f908caaf4ed2f3ea9a816aa4ed03f480655df1556dbc", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "af2-vmap-vmap_batch_retry_20261028_190537_289704-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "RUN_ID=vmap_batch_retry_20261028_190537_289704",
  "-e",
  "BATCH=1",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_batch_retry_20261028_190537_289704/mplcache",
  "-v",
  "/home/hyein/ai-benchmark/colabfold:/work",
  "-v",
  "/home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro",
  "ghcr.io/sokrypton/colabfold:1.6.2-cuda13",
  "python",
  "-u",
  "/work/scripts/batch_followup/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-00419e3d78fd

원래 파일: `colabfold/results/vmap_batch_retry_20261028_190537_289704/input_audit.json` · 원문 SHA256: `34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d`

<details>
<summary>colabfold/results/vmap_batch_retry_20261028_190537_289704/input_audit.json — 명령/설정/환경 원문</summary>


동일 원문은 `colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json` 블록에 전문 수록했다. 아래 별칭은 동일한 바이트를 원래 경로로 복원한다.
<!-- RESTORE_ALIAS {"path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/input_audit.json", "source": "colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json", "sha256": "34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d"} -->


</details>

### record-830df92fc0fd

원래 파일: `colabfold/results/vmap_fp32_20261028_190843_576148/batch1_command.json` · 원문 SHA256: `ad73ef573ce45307ddd3c0386e5374c451da1e20908f849c99560468707f9f14`

<details>
<summary>colabfold/results/vmap_fp32_20261028_190843_576148/batch1_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch1_command.json", "sha256": "ad73ef573ce45307ddd3c0386e5374c451da1e20908f849c99560468707f9f14", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "af2-vmap-vmap_fp32_20261028_190843_576148-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "RUN_ID=vmap_fp32_20261028_190843_576148",
  "-e",
  "BATCH=1",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_fp32_20261028_190843_576148/mplcache",
  "-v",
  "/home/hyein/ai-benchmark/colabfold:/work",
  "-v",
  "/home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro",
  "ghcr.io/sokrypton/colabfold:1.6.2-cuda13",
  "python",
  "-u",
  "/work/scripts/batch_followup/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-36fa58f4b3fe

원래 파일: `colabfold/results/vmap_fp32_20261028_190843_576148/batch2_command.json` · 원문 SHA256: `08e32def6516646237a84e9558052a62015dfb3d7031a3e8ca41e08dc4ab0ce8`

<details>
<summary>colabfold/results/vmap_fp32_20261028_190843_576148/batch2_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch2_command.json", "sha256": "08e32def6516646237a84e9558052a62015dfb3d7031a3e8ca41e08dc4ab0ce8", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "af2-vmap-vmap_fp32_20261028_190843_576148-2",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "RUN_ID=vmap_fp32_20261028_190843_576148",
  "-e",
  "BATCH=2",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_fp32_20261028_190843_576148/mplcache",
  "-v",
  "/home/hyein/ai-benchmark/colabfold:/work",
  "-v",
  "/home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro",
  "ghcr.io/sokrypton/colabfold:1.6.2-cuda13",
  "python",
  "-u",
  "/work/scripts/batch_followup/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-4e2b87343e37

원래 파일: `colabfold/results/vmap_fp32_20261028_190843_576148/batch4_command.json` · 원문 SHA256: `74e4c1cf908f63d2330b27060a571896d7c79e36aa3337e345ff1154d2c8ff78`

<details>
<summary>colabfold/results/vmap_fp32_20261028_190843_576148/batch4_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch4_command.json", "sha256": "74e4c1cf908f63d2330b27060a571896d7c79e36aa3337e345ff1154d2c8ff78", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "af2-vmap-vmap_fp32_20261028_190843_576148-4",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "RUN_ID=vmap_fp32_20261028_190843_576148",
  "-e",
  "BATCH=4",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_fp32_20261028_190843_576148/mplcache",
  "-v",
  "/home/hyein/ai-benchmark/colabfold:/work",
  "-v",
  "/home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro",
  "ghcr.io/sokrypton/colabfold:1.6.2-cuda13",
  "python",
  "-u",
  "/work/scripts/batch_followup/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-16252fd1c1bd

원래 파일: `colabfold/results/vmap_fp32_20261028_190843_576148/input_audit.json` · 원문 SHA256: `34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d`

<details>
<summary>colabfold/results/vmap_fp32_20261028_190843_576148/input_audit.json — 명령/설정/환경 원문</summary>


동일 원문은 `colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json` 블록에 전문 수록했다. 아래 별칭은 동일한 바이트를 원래 경로로 복원한다.
<!-- RESTORE_ALIAS {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/input_audit.json", "source": "colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json", "sha256": "34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d"} -->


</details>

### record-13e93a79041b

원래 파일: `scripts/af2_feature_capture_launch.py` · 원문 SHA256: `2d62f809b99dd4b97b04835333d35d25b3eb86c98d7ee514a7df20f422b6e6da`

<details>
<summary>scripts/af2_feature_capture_launch.py — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "scripts/af2_feature_capture_launch.py", "sha256": "2d62f809b99dd4b97b04835333d35d25b3eb86c98d7ee514a7df20f422b6e6da", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import subprocess,datetime,shlex,json
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'colabfold';run='vmap_batch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');print(run,flush=True);cmd=['docker','run','--rm','--network','none','--ipc=host','--user','1000:1000','-e','JAX_PLATFORMS=cpu','-e','RUN_ID='+run,'-e','MPLCONFIGDIR=/work/results/'+run+'/mplcache','-v',str(R)+':/work','-v',str(R/'cache/colabfold')+':/cache:ro','ghcr.io/sokrypton/colabfold:1.6.2-cuda13','python','-u','/work/scripts/batch_followup/capture.py'];(D/'capture_command.json').write_text(json.dumps(cmd,indent=2))
with (D/'capture.log').open('w') as f:c=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
(D/'capture_exit.json').write_text(json.dumps({'exit_code':c}));print('CAPTURE_EXIT',c,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

### record-5c150770ebdc

원래 파일: `scripts/af2_vmap_launch.py` · 원문 SHA256: `68c706f5a2fcb6da7c57452936689f1e79ca218d52e2020548915db603480c01`

<details>
<summary>scripts/af2_vmap_launch.py — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "scripts/af2_vmap_launch.py", "sha256": "68c706f5a2fcb6da7c57452936689f1e79ca218d52e2020548915db603480c01", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,subprocess,time,shlex,sys
import psutil
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'colabfold';run=(R/'results/BATCH_LATEST').read_text().strip();D=R/'results'/run;assert json.loads((D/'capture_exit.json').read_text())['exit_code']==0
clair=ROOT/'clair3/results'/(ROOT/'clair3/results/BATCH_LATEST').read_text().strip()
while json.loads((clair/'summary.json').read_text())['status']=='running':time.sleep(10)
S={'status':'running','run_id':run,'workers':[],'failed':[],'scope':'experimental JAX vmap independent monomer batch axis on existing ColabFold model_1, no simultaneous GPU jobs','settings':{'guard_available_gib':16,'guard_swap_increase_gib':2}};swap0=psutil.swap_memory().used
def save():(D/'summary.json').write_text(json.dumps(S,indent=2))
save();print('AF2_BATCH_START',run,flush=True)
for batch in [1,2,4]:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'other GPU work'
 if psutil.virtual_memory().available<16*2**30:S['stop_reason']='available_memory_guard';break
 name='af2-vmap-'+run+'-'+str(batch);cmd=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--ipc=host','--user','1000:1000','-e','RUN_ID='+run,'-e','BATCH='+str(batch),'-e','JAX_PLATFORMS=cuda,cpu','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','-e','MPLCONFIGDIR=/work/results/'+run+'/mplcache','-v',str(R)+':/work','-v',str(R/'cache/colabfold')+':/cache:ro','ghcr.io/sokrypton/colabfold:1.6.2-cuda13','python','-u','/work/scripts/batch_followup/worker.py'];(D/f'batch{batch}_command.json').write_text(json.dumps(cmd,indent=2));start=time.time();reason=None
 with (D/f'batch{batch}.log').open('w') as f:
  p=subprocess.Popen(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',name])],stdout=f,stderr=subprocess.STDOUT);break
   time.sleep(.5)
  code=p.wait()
 (D/f'batch{batch}_exit.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start,'reason':reason}));print('AF2_BATCH_EXIT',batch,code,flush=True)
 if (D/f'batch_{batch}/summary.json').exists():S['workers'].append(json.loads((D/f'batch_{batch}/summary.json').read_text()))
 if code:S['failed'].append({'batch':batch,'exit_code':code});S['stop_reason']='worker_error_or_guard_see_log';break
 save()
else:S['stop_reason']='actual_four_query_pool_complete'
S['status']='complete' if not S['failed'] and len(S['workers'])==3 else 'stopped';save();subprocess.run([sys.executable,str(ROOT/'scripts/two_batch_report.py'),'af2',str(D)],check=True);print('AF2_FINISHED',S['status'],flush=True)
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

<details><summary>추가 실험 실행 원문: four_real_queries_executed_source.py</summary>

<!-- RESTORE_FILE {"path": "colabfold/results/real_input_capacity_20261102_173415_016493/four_real_queries_executed_source.py", "sha256": "dd766a320f0004e8ff1ebf662e693a93c30c9590aebc8bf65367efbc410e8a94", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,time,json,threading,csv,subprocess,gc
import numpy as np,jax,jax.numpy as jnp
from colabfold.alphafold.models import load_models_and_params
from alphafold.common import protein
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];
def repair_cached_links():
 for link in D.iterdir():
  if link.is_symlink():
   target=os.readlink(link);prefix='/home/hyein/ai-benchmark/colabfold/'
   if target.startswith(prefix):
    relative=os.path.relpath(R/target[len(prefix):],link.parent);link.unlink();link.symlink_to(relative)
repair_cached_links()
B=int(os.environ['BATCH']);C=D/f'batch_{B}';C.mkdir();dev=jax.devices('gpu')[0];stop=threading.Event();rows=[];phase='loading';G=2**30
S={'status':'running','batch':B,'implementation':'experimental jax.vmap over unmodified ColabFold RunModel.apply; shared fixed model_1 weights, independent actual query features and RNG; not a native CLI batch option','conditions':[]}
def write():(C/'summary.json').write_text(json.dumps(S,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def mem():
 info={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};rss=next(int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'));return {'rss_gib':rss/G,'available_gib':info['MemAvailable']/G,'total_gib':info['MemTotal']/G,'swap_gib':(info['SwapTotal']-info['SwapFree'])/G}
def dmem():return {k:int(v) for k,v in (dev.memory_stats() or {}).items()}
swap0=mem()['swap_gib']
def monitor():
 with (C/'resources.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['query_start','query_end','phase','gpu_util','rss_gib','available_gib','swap_gib','jax_bytes_in_use']);w.writeheader()
  while not stop.is_set():
   t=time.time();ph=phase
   try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip())
   except Exception:u=None
   m=mem();r={'query_start':t,'query_end':time.time(),'phase':ph,'gpu_util':u,'rss_gib':m['rss_gib'],'available_gib':m['available_gib'],'swap_gib':m['swap_gib'],'jax_bytes_in_use':dmem().get('bytes_in_use')};rows.append(r);w.writerow(r);f.flush()
   if m['available_gib']<16 or m['swap_gib']>swap0+2:(C/'memory_guard.json').write_text(json.dumps(r));os._exit(86)
   stop.wait(.5)
th=threading.Thread(target=monitor,daemon=True);th.start()
def aligned_rmsd(x,y):
 x=x-x.mean(0);y=y-y.mean(0);u,s,v=np.linalg.svd(x.T@y);q=np.eye(3);q[-1,-1]=np.linalg.det(u@v);return float(np.sqrt(np.mean(np.sum((x@(u@q@v)-y)**2,axis=1))))
try:
 audit=json.loads((D/'input_audit.json').read_text());queries=audit['queries'];allfeat=[dict(np.load(D/(q['id']+'_processed_features.npz'))) for q in queries];keys=set(allfeat[0]);assert all(set(f)==keys for f in allfeat);maxshape={k:tuple(max(f[k].shape[d] for f in allfeat) for d in range(allfeat[0][k].ndim)) for k in keys};features=[]
 for f in allfeat:
  features.append({k:np.pad(v,[(0,maxshape[k][d]-v.shape[d]) for d in range(v.ndim)],mode='constant') for k,v in f.items()})
 selected=features[:B];S['input']={'unique_queries':B,'query_ids':[q['id'] for q in queries[:B]],'actual_residues':[q['length'] for q in queries[:B]],'feature_shapes_per_query':{k:list(v.shape) for k,v in features[0].items()},'padding':'pool-wide max feature shapes; zero-padded extra-MSA/cluster entries masked by zero masks; no sequence duplication','actual_msa_valid_rows':[int((f['msa_mask'][0].sum(-1)>0).sum()) for f in selected]};t=time.time();loaded=load_models_and_params(num_models=1,use_templates=False,num_recycles=3,recycle_early_stop_tolerance=0.,num_ensemble=1,model_order=[1],model_type='alphafold2_ptm',data_dir=Path('/cache'),max_seq=512,max_extra_seq=5120,use_bfloat16=False,use_pallas=False,compile_mode='tuned');name,runner,params=loaded[0];runner.params=params;jax.block_until_ready(params);S['loading_seconds']=time.time()-t;S['parameters']=sum(int(x.size) for x in jax.tree_util.tree_leaves(params));S['weights_dtype']=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(params)});S['model_only_postload']={'jax':dmem(),'system':mem()};S['environment']={'jax':jax.__version__,'device':str(dev),'preallocate':os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),'compute':'FP32 activations/weights, explicit separate validation condition after BF16 vmap reference drift','recycles':3,'ensembles':1,'templates':False,'relaxation':False,'seed_per_query':42};write()
 full={k:jnp.asarray(np.stack([f[k] for f in selected])) for k in keys};L=int(selected[0]['aatype'].shape[1]);zero_prev={'prev_msa_first_row':jnp.zeros((B,L,256),jnp.float16),'prev_pair':jnp.zeros((B,L,L,128),jnp.float16),'prev_pos':jnp.zeros((B,L,37,3),jnp.float16)};rng=jnp.stack([jax.random.PRNGKey(42)]*B);first={k:v[:,0:1] for k,v in full.items()};first['prev']=zero_prev;vmapped=jax.jit(jax.vmap(runner.apply.__wrapped__,in_axes=(None,0,0)),compiler_options={'xla_gpu_override_gemm_autotuner':'block_m: 64 block_n: 64 block_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1'});phase='compilation';t=time.time();compiled=vmapped.lower(params,rng,first).compile();S['compilation_seconds']=time.time()-t;S['compiled_memory_analysis']=str(compiled.memory_analysis());write()
 def predict():
  prev=zero_prev;key=jnp.stack([jax.random.PRNGKey(42)]*B)
  for recycle in range(4):
   pair=jax.vmap(jax.random.split)(key);key=pair[:,0];subkey=pair[:,1];subfeat={k:v[:,recycle:recycle+1] for k,v in full.items()};subfeat['prev']=prev;result=compiled(params,subkey,subfeat);jax.block_until_ready(result);result=jax.tree_util.tree_map(lambda x:np.asarray(x,np.float16),result);prev={k:jnp.asarray(v) for k,v in result.pop('prev').items()}
  return result
 phase='warmup';t=time.time();result=predict();S['warmup_seconds']=time.time()-t;phase='execution';t=time.time();repeats=0
 while repeats<2 or time.time()-t<15:result=predict();repeats+=1
 end=time.time();stats=dmem();observed=[r for r in rows if r['query_start']>=t and r['query_end']<=end];utils=[r['gpu_util'] for r in observed if r['gpu_util'] is not None];S['conditions']=[{'batch':B,'unique_queries':B,'repeats':repeats,'processed_queries':B*repeats,'seconds':end-t,'queries_per_second':B*repeats/(end-t),'gpu_util_avg':float(np.mean(utils)) if utils else None,'gpu_util_max':max(utils) if utils else None,'sample_count':len(utils),'jax_memory_stats':stats,'memory_definition':'peak_bytes_in_use is cumulative process allocator peak including compilation and warmup; execution bytes_in_use samples separate, no unsupported reset','execution_sampled_peak_bytes_in_use':max([r['jax_bytes_in_use'] for r in observed if r['jax_bytes_in_use'] is not None],default=None),'rss_sampled_peak_gib':max([r['rss_gib'] for r in observed],default=None),'available_sampled_min_gib':min([r['available_gib'] for r in observed],default=None)}];phase='validation';preds=[]
 for i,q in enumerate(queries[:B]):
  single=jax.tree_util.tree_map(lambda x:x[i],result);coords=single['structure_module']['final_atom_positions'];plddt=single['plddt'];assert coords.shape==(L,37,3) and np.isfinite(coords).all() and np.isfinite(plddt).all();np.savez(C/(q['id']+'_prediction.npz'),atom_positions=coords,plddt=plddt);prot=protein.from_prediction(features=selected[i],result=single,b_factors=np.repeat(plddt[:,None],37,axis=1));pdb=protein.to_pdb(prot);(C/(q['id']+'.pdb')).write_text(pdb);assert sum(l.startswith('ATOM') and l[12:16].strip()=='CA' for l in pdb.splitlines())==q['length'];preds.append({'id':q['id'],'residues':q['length'],'mean_plddt':float(plddt.mean()),'finite_and_pdb_valid':True})
 S['predictions']=preds;reference=D/'batch_1'/ (queries[0]['id']+'_prediction.npz')
 if B==1:
  phase='single_native_reference';t0=time.time();ref,recycles=runner.predict(selected[0],random_seed=42);S['native_reference_seconds_excluded']=time.time()-t0;refcoords=ref['structure_module']['final_atom_positions'];S['native_reference_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(refcoords[:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max()),'native_recycles':int(recycles)};assert S['native_reference_comparison']['ca_aligned_rmsd_angstrom']<.5;np.savez(C/'native_single_reference.npz',atom_positions=refcoords,plddt=ref['plddt'])
 else:
  ref=np.load(reference);S['first_query_batch_comparison']={'ca_aligned_rmsd_angstrom':aligned_rmsd(np.asarray(ref['atom_positions'][:,1],float),np.asarray(result['structure_module']['final_atom_positions'][0,:,1],float)),'plddt_max_abs':float(np.abs(ref['plddt']-result['plddt'][0]).max())}
 S['status']='complete';write();print('AF2_VMAP_COMPLETE',B,flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write();raise
finally:stop.set();th.join(timeout=5)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: four_real_queries_command.json</summary>

<!-- RESTORE_FILE {"path": "colabfold/results/real_input_capacity_20261102_173415_016493/four_real_queries_command.json", "sha256": "3caa9497b37136b0297c311229a881ec7dc4646953ab61ea8ce6fd8b8eca5b2d", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/colabfold:/work",
  "-v",
  "/home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_173415_016493",
  "-e",
  "BATCH=4",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "ghcr.io/sokrypton/colabfold:1.6.2-cuda13",
  "python",
  "-u",
  "/work/scripts/batch_followup/worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: remaining_queue_152f0b87bce32235.py</summary>

<!-- RESTORE_FILE {"path": "colabfold/results/real_input_capacity_20261102_173415_016493/reproduction_sources/remaining_queue_152f0b87bce32235.py", "sha256": "152f0b87bce32235a41f8616d74f03deabd611eef85dde6419bd68da175973a6", "encoding": "utf-8", "added_final_newline": false} -->
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

<!-- RESTORE_FILE {"path": "colabfold/results/real_input_capacity_20261102_173415_016493/reproduction_sources/update_reports_305433d7fed68d6f.py", "sha256": "305433d7fed68d6ff20850b0d00d18708a0de3d92b36924d41dbbfe168034b8e", "encoding": "utf-8", "added_final_newline": false} -->
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

<details><summary>단계별 입력·출력 확인/보고 원문: collect_af2.py</summary>

<!-- RESTORE_FILE {"path": "scripts/report_stages/source_snapshots/collect_af2_8f9b1714e03b3f29.py", "sha256": "8f9b1714e03b3f29d8b632b1748f6be8a564352c110f24486a59b2a43677774e", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,numpy as np
R=Path('/home/hyein/ai-benchmark/colabfold');D=R/'results'/(R/'results/REAL_TRAINING_LATEST').read_text().strip();O=D/'stage_trace_saved';O.mkdir(exist_ok=True);audit=json.loads((D/'input_audit.json').read_text());row=audit['queries'][0];features=dict(np.load(D/(row['id']+'_processed_features.npz')));pred=dict(np.load(D/'batch_4'/(row['id']+'_prediction.npz')));pdb=(D/'batch_4'/(row['id']+'.pdb')).read_text();arrays={**{k:v for k,v in features.items() if k in ['aatype','residue_index','target_feat','msa_feat','msa_mask','extra_msa','extra_msa_mask']},**pred};np.savez_compressed(O/'arrays.npz',**arrays);S={'model':'colabfold','sample_id':row['id'],'raw_record':row,'provenance':'Read only already saved processed features and final GPU outputs; no rerun, no new intermediate neural capture','feature_shapes':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in features.items()},'output_shapes':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in pred.items()},'plddt_first8':pred['plddt'][:8].tolist(),'mean_plddt':float(pred['plddt'].mean()),'first_residue_CA_xyz':pred['atom_positions'][0,1].tolist(),'first5_PDB_ATOM_lines':[l for l in pdb.splitlines() if l.startswith('ATOM')][:5],'pdb_file':str((D/'batch_4'/(row['id']+'.pdb')).relative_to(R)),'saved_arrays':str((O/'arrays.npz').relative_to(R)),'neural_intermediate_values':'not saved; do not invent Evoformer embedding values','ground_truth':'none for this UniProt query; pLDDT is confidence, not groundtruth structure accuracy'};(O/'trace.json').write_text(json.dumps(S,ensure_ascii=False,indent=2));print('AF2 saved input/output collected',row['id'])
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
