# Nucleotide Transformer V2 최종 통합 보고서

## 1. 모델과 실제 입력

|항목|이번 실험에서 사용한 모델|
|---|---|
|체크포인트·버전|InstaDeepAI/nucleotide-transformer-v2-500m-multi-species|
|구조|DNA6-mer 중심 tokenizer+ESM 구현 Transformer|
|전체·구성별 파라미터 수|MLM498,345,436개(약4.98억 개)(498.35M);embedding backbone493,084,113개(약4.93억 개);기본CLS분류구성494,135,763개(약4.94억 개)|
|정밀도|FP32|
|모델 입력 → 출력|DNA →token 임베딩 /enhancer 이진분류|
|가중치와 실행 메모리의 차이|파라미터 수×dtype bytes는 이론 가중치 용량이다. 실행 peak에는 입력·중간 활성값이 더해지고,파인튜닝에는 gradient/optimizer 상태도 추가된다.|
|메모리% 정의|CUDA allocated peak / CPU·GPU 공유 RAM121.6875GiB ×100; GPU-Util과 별개. AlphaFold2는 JAX 계측을 별도로 표시|

### 실제 데이터와 학습 방식 구분

|구분|데이터·실제 처리 내용|
|---|---|
|추론 데이터|GRCh38 실제긴구간. 추가추론:서로 겹치지 않는6000nt window1024개,약1001tokens|
|학습·검증·test 데이터|실제enhancer 라벨 서열199–200nt. train512/validation128/test400|
|학습 입력 X / 정답 y|X:token_ids/attention_mask[B,실제token길이]. y:[B]정수0/1. 긴6000nt 추론 window에는 이 enhancer 정답이 없음|
|가중치 갱신 방식|기본:전체backbone+원생CLS MLP 파인튜닝. 추가:전체backbone+masked-mean linear head 파인튜닝,별도 조건|
|원본·변환 입력 저장 위치|`data/inputs.json의 train/validation/test;data/real_training_v3/inference_inputs.json추론전용`|
|해석 범위|긴 유전체 추론window와 짧은enhancer 학습 서열을 구분. 정확도는 해당 라벨·split에서만 평가|

**용어:** 처음부터 학습=무작위 초기화에서 시작; 전체/일부 층 파인튜닝=공개 가중치에서 해당 부분 갱신; 분류기만 학습=본체를 동결하고 head만 갱신. **용량probe는 학습 방식이 아니라 짧은 자원 측정 목적**이며,실제로 수행한 가중치 갱신 방식과 함께 표기한다. 원본 전체 데이터 수와 해당 run에서 실제 사용한 부분집합 수를 구분한다.

### 실제 파일 → 전처리 → 모델 표현 → 최종 출력: 단계별 실제 예시

이 예시는 **enhancer 이진분류**, 저장된 추가 전체backbone 파인튜닝 모델의 **test `0`**다. 아래 모든값은 같은 레코드에서 이어진다.

#### 0–1. 파일 레코드 — X와 정답 y

data/inputs.json의 test: id,sequence,label. 아래는 실제 레코드에서 앞80염기만 발췌한 표시이며, 모델에는 전체서열을 넣었다.

```json
{
  "id": "0",
  "sequence_length": 200,
  "sequence_first80": "AATTTTCTCATTTTCTCATAAAGTTTAACAGTTGTTTATTTGAGTCAGAATTCAAATAAGCTTCTGTACATTACAATTGG",
  "label": 0
}
```
**X는 염기서열, y는 위 label**이다. 임베딩을 정답으로 쓰는 것이 아니다.

#### 2. 토큰화 — 실제 모델 입력

|항목|실제값|
|---|---|
|tokens|[1, 36] int64|
|attention_mask|[1, 36] int64|
|원서열 길이|200nt|
|유효token 수|36|
|token ID 앞16개|[3, 91, 2443, 1438, 1043, 1294, 3451, 1117, 873, 94, 73, 2413, 1163, 523, 3931, 220]|

6-mer 중심 토큰화로 염기수와token수가 다르다. 숫자ID는vocabulary index이고, 정답라벨번호와 다른 배열이다.

#### 3. 본체 출력 — 위치별 표현

`[1, 36, 1024] float32`이며 각token에1024개 특징값이 생긴다. 첫위치의 앞8개 실제값: `[0.140821, -0.019736, -0.052619, 0.359133, -0.370083, 0.265481, -0.800648, 1.219445]`. 이 단계는 최종 class확률이 아니다.

#### 4. Pooling — 서열당 벡터1개

방식: **masked mean includes valid special tokens**. 출력 `[1, 1024] float32`. 임베딩 앞8개 실제값: `[0.376638, 0.007368, -0.342578, 0.387146, -0.168588, 0.254605, -0.465110, 0.280753]`.

#### 5. 과제 head — 최종 예측으로 변환

추가 파인튜닝의 linear head는 `1024→2`다. 기본보고서의 다른head/학습범위 조건과 구분한다.

|단계|shape|의미|
|---|---|---|
|head logits|[1, 2] float32|확률변환 전 실수|
|확률|[1, 2] float32|softmax:2개중한class의확률|
|판정|0|최대확률class의ID|

logits 앞8개: `[1.237789, -1.546300]`; 확률 앞8개: `[0.941810, 0.058190]`.

#### 6. 정답과 비교

|항목|실제값|
|---|---|
|정답|[0]|
|예측|0|

#### 7. 저장 파일

|단계별 저장 배열|실제 shape·dtype|의미|
|---|---|---|
|input_ids|[1, 36] int64|입력token|
|attention_mask|[1, 36] int64|유효위치|
|token_representations|[1, 36, 1024] float32|token별표현|
|embedding|[1, 1024] float32|서열당임베딩|
|logits|[1, 2] float32|head출력|
|probabilities|[1, 2] float32|class확률|
|target|[1] int64|정답|

로컬 전체 배열: `results/factor_control_20261103_093231_014557/full_training_b8_e2_n512/stage_trace_cpu/arrays.npz`. `arrays.npz::key`는 같은NPZ 파일 안의 배열이름이다. 아래 재현코드로 저장모델과 입력에서 다시 확인할 수 있다.

원GPU run의 `test_outputs.npy`는 **logits**이며 확률파일이 아니다. 값의 순서는 저장된 split_ids['test']와 일치한다.

**값의 출처:** 저장된 모델을 CPU에서 새로 재확인한 중간값이다. 원GPU 자원계측 중에 캡처했던 것처럼 해석하지 않는다. 같은 저장GPU 출력과 최대 절대차이는 **5.960e-07**이다. 이 CPU 확인시간은 기존 GPU 벤치마크 시간에 포함하지 않는다.

### 모델 적재 직후 — 첫 입력 forward 이전

|적재 조건|allocated GiB(JAX는별도표기)|reserved GiB|프로세스 RSS GiB|원본 근거|
|---|---|---|---|---|
|추론/기본 모델 적재 기준|1.856512|1.873046875|0.9985618591308594|ntv2/results/real_input_capacity_20261102_152621_008750/batch_128/summary.json|
|파인튜닝 적재 기준|1.836920|1.85546875|0.9943008422851562|ntv2/results/real_input_capacity_20261102_152621_008750/full_training_b512/summary.json|

위 값은 **가중치를 GPU에 적재하고 첫 입력을 계산하기 전의 프레임워크 할당량**이다. CUDA/JAX context·driver·시스템 전체 GPU사용량과 같지 않으며 CPU입력 준비 포함 여부는 해당 실행 코드를 따른다. **실행 peak와 모델 적재 기준값의 차이는 같은 run에서만 계산**한다. 파인튜닝 중 추가량에는 입력뿐 아니라 중간값·gradient·optimizer 상태가 포함된다. 서로 다른 조건/정밀도의 기준값을 하나의 모델 고유 메모리로 단정하지 않는다.

## 2. 토큰화 — 실제 모델 입력

<!-- ENV_MEASUREMENT_OVERVIEW_START -->
### 실행 환경·자원 측정 방법 요약

|항목|이번 보고서의 측정 기준|
|---|---|
|장비·공유메모리|DGX Spark,NVIDIA GB10,ARM64;시스템공유RAM121.6875GiB. CPU/GPU RAM을별도로더하지않음|
|환경기록출처|ntv2/results/real_input_capacity_20261102_152621_008750/batch_128/summary.json|
|Python·프레임워크|{'torch': '2.13.0a0+8145d630e8.nv26.06', 'cuda': '13.3', 'precision': 'FP32'}|
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

|항목|실제값|
|---|---|
|tokens|[1, 36] int64|
|attention_mask|[1, 36] int64|
|원서열 길이|200nt|
|유효token 수|36|
|token ID 앞16개|[3, 91, 2443, 1438, 1043, 1294, 3451, 1117, 873, 94, 73, 2413, 1163, 523, 3931, 220]|

6-mer 중심 토큰화로 염기수와token수가 다르다. 숫자ID는vocabulary index이고, 정답라벨번호와 다른 배열이다.

#### 3. 본체 출력 — 위치별 표현

`[1, 36, 1024] float32`이며 각token에1024개 특징값이 생긴다. 첫위치의 앞8개 실제값: `[0.140821, -0.019736, -0.052619, 0.359133, -0.370083, 0.265481, -0.800648, 1.219445]`. 이 단계는 최종 class확률이 아니다.

#### 4. Pooling — 서열당 벡터1개

방식: **masked mean includes valid special tokens**. 출력 `[1, 1024] float32`. 임베딩 앞8개 실제값: `[0.376638, 0.007368, -0.342578, 0.387146, -0.168588, 0.254605, -0.465110, 0.280753]`.

#### 5. 과제 head — 최종 예측으로 변환

추가 파인튜닝의 linear head는 `1024→2`다. 기본보고서의 다른head/학습범위 조건과 구분한다.

|단계|shape|의미|
|---|---|---|
|head logits|[1, 2] float32|확률변환 전 실수|
|확률|[1, 2] float32|softmax:2개중한class의확률|
|판정|0|최대확률class의ID|

logits 앞8개: `[1.237789, -1.546300]`; 확률 앞8개: `[0.941810, 0.058190]`.

#### 6. 정답과 비교

|항목|실제값|
|---|---|
|정답|[0]|
|예측|0|

#### 7. 저장 파일

|단계별 저장 배열|실제 shape·dtype|의미|
|---|---|---|
|input_ids|[1, 36] int64|입력token|
|attention_mask|[1, 36] int64|유효위치|
|token_representations|[1, 36, 1024] float32|token별표현|
|embedding|[1, 1024] float32|서열당임베딩|
|logits|[1, 2] float32|head출력|
|probabilities|[1, 2] float32|class확률|
|target|[1] int64|정답|

로컬 전체 배열: `results/factor_control_20261103_093231_014557/full_training_b8_e2_n512/stage_trace_cpu/arrays.npz`. `arrays.npz::key`는 같은NPZ 파일 안의 배열이름이다. 아래 재현코드로 저장모델과 입력에서 다시 확인할 수 있다.

원GPU run의 `test_outputs.npy`는 **logits**이며 확률파일이 아니다. 값의 순서는 저장된 split_ids['test']와 일치한다.

**값의 출처:** 저장된 모델을 CPU에서 새로 재확인한 중간값이다. 원GPU 자원계측 중에 캡처했던 것처럼 해석하지 않는다. 같은 저장GPU 출력과 최대 절대차이는 **5.960e-07**이다. 이 CPU 확인시간은 기존 GPU 벤치마크 시간에 포함하지 않는다.

### 모델 적재 직후 — 첫 입력 forward 이전

|적재 조건|allocated GiB(JAX는별도표기)|reserved GiB|프로세스 RSS GiB|원본 근거|
|---|---|---|---|---|
|추론/기본 모델 적재 기준|1.856512|1.873046875|0.9985618591308594|ntv2/results/real_input_capacity_20261102_152621_008750/batch_128/summary.json|
|파인튜닝 적재 기준|1.836920|1.85546875|0.9943008422851562|ntv2/results/real_input_capacity_20261102_152621_008750/full_training_b512/summary.json|

위 값은 **가중치를 GPU에 적재하고 첫 입력을 계산하기 전의 프레임워크 할당량**이다. CUDA/JAX context·driver·시스템 전체 GPU사용량과 같지 않으며 CPU입력 준비 포함 여부는 해당 실행 코드를 따른다. **실행 peak와 모델 적재 기준값의 차이는 같은 run에서만 계산**한다. 파인튜닝 중 추가량에는 입력뿐 아니라 중간값·gradient·optimizer 상태가 포함된다. 서로 다른 조건/정밀도의 기준값을 하나의 모델 고유 메모리로 단정하지 않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

<!-- FACTOR_EFFECTS_START -->
### 시간·메모리에 영향을 주는 요인 — 통제 비교 상태

아래 원리는 모든 모델에서 정량 검증을 마쳤다는 뜻이 아니다. **실측 비교와 미실험을 구분**한다. 배치 sweep 중 정밀도·입력길이·체크포인트를 고정하며,다른 과제/다른head 결과를 같은 조건으로 섞지 않는다.

|요인|총시간·처리량 해석|최대 allocated 해석|이 모델의 실험 근거/상태|
|---|---|---|---|
|배치|한step은 길어질 수 있고step수는 줄어든다. 처리량 상승/포화는 실측으로 판단|같은입력길이·정밀도에서 동시중간값 증가가 주된영향|긴6000nt 추론batch확대 진행;짧은enhancer 훈련은 별도|
|샘플당 실제입력 크기|GPU에들어간실제token/타일 크기가 커졌을때 계산량 변화 비교|토큰길이·패치/타일크기·attention구조에 따라 증가;원본크기와 다를수있음|기본길이600/3000/6000/12276nt 및개수 비교 있음|
|고유 데이터수|같은batch·epoch에서 총처리step가늘어 총시간이늘어나는지 비교|고정batch의동시입력이같으면 데이터총수만으로peak가크게늘지않을수있음|입력개수비교는본문참조;새지도학습데이터수비교는아래결과가생긴조건만완료|
|epoch|같은train입력을 반복하므로 누적시간 증가;검증성능개선은 보장안됨|동일batch·입력에서epoch자체가peak를선형증가시키는요인은아님|같은enhancer train512의epoch1/2/4 및 train128/256/512 비교 완료 → 실제완료 5/5조건(test·재로드확인)|
|학습방식·갱신범위|추론/본체동결head학습/부분·전체미세조정/처음부터학습은 계산범위가다름|gradient·optimizer·역전파용중간값의유무가다름|1절의가중치갱신방식과X/y를참조;다른데이터·목적의시간을단순배수비교하지않음|
|모델·정밀도·전처리|구조·dtype·토큰화·타일화·컴파일·후처리도영향|모델적재기준값과실행중추가량을분리|원인후보/고정조건으로기록;모델크기나dtype의단독영향sweep은미실험|

추가 통제 비교에서는 **batch8·FP32·공개초기weight·seed42·optimizer·원래전체pool의padding폭·검증/test를고정**하고,(1)같은train의epoch1/2/4,(2)epoch2의중첩train1/4·1/2·전체를각각실행한다. 전체데이터를바꾸면서epoch도함께바꾼행끼리는단독효과로해석하지않는다. 단일실행이며데이터부분집합의label분포변화는정확도해석의제약이다. 이비교는해당지도학습지원모델에만적용한다.

통제실험run:`factor_control_20261103_093231_014557`

|조건|고유train수|batch|epoch|optimizersteps|훈련시간s|처리입력/s|allocatedpeakGiB|reservedpeakGiB|GPU-Util평균%|독립test지표|재로드|
|---|---|---|---|---|---|---|---|---|---|---|---|
|full_training_b8_e1_n512|512|8|1|64|18.5851|27.5490|9.2280|9.3340|89.6061|{'mean_loss': 0.6200778299570083, 'accuracy': 0.715, 'macro_f1': 0.7148859543817527}|True|
|full_training_b8_e2_n128|128|8|2|32|9.7094|26.3663|9.2280|9.3340|83.1176|{'mean_loss': 0.5596214205026626, 'accuracy': 0.72, 'macro_f1': 0.7198248905565978}|True|
|full_training_b8_e2_n256|256|8|2|64|18.7907|27.2475|9.2280|9.3340|91.4545|{'mean_loss': 0.7668325234949589, 'accuracy': 0.6775, 'macro_f1': 0.6629895957834513}|True|
|full_training_b8_e2_n512|512|8|2|128|36.5120|28.0456|9.2280|9.3340|92.3538|{'mean_loss': 0.6576204341650009, 'accuracy': 0.7125, 'macro_f1': 0.7117053879756078}|True|
|full_training_b8_e4_n512|512|8|4|256|73.9990|27.6761|9.2280|9.3340|93.2946|{'mean_loss': 0.8709892931208014, 'accuracy': 0.7175, 'macro_f1': 0.7169888110399409}|True|

**현재계획한5조건의학습·test평가·저장모델재로드완료.**

- 고정train512개,epoch1/2/4의훈련시간: 18.59초 / 36.51초 / 74.00초. 같은순서의allocated peak: 9.228GiB / 9.228GiB / 9.228GiB.
- 고정epoch2,train128/256/512개의훈련시간: 9.71초 / 18.79초 / 36.51초.
- 이번관측에서는epoch·처리데이터수증가에따라총훈련시간이늘었고,batch·padding폭이같아최대allocated는거의유지됐다. 정확도는단조증가하지않아개별test지표를확인해야한다.

<!-- FACTOR_EFFECTS_END -->

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|배치 추론|고유1개; GRCh38 실제6000nt DNA|batch1, input_ids:[1, 1001] / attention_mask:[1, 1001],FP32|78회/총78건|15.12초|5.16입력/초|2.058 GiB / 1.69%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|배치 추론|고유128개; GRCh38 실제6000nt DNA|batch128, input_ids:[128, 1001] / attention_mask:[128, 1001],FP32|2회/총256건|56.80초|4.51입력/초|21.592 GiB / 17.74%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|전체backbone 파인튜닝|실제enhancer train512/val128/test400;199~200nt|batch8,FP32|2epoch/128step|38.13초(학습구간합)|26.86서열/초|9.252 GiB / 7.60%|동일저장모델test 지표본문;80%미도달|

<!-- REAL_TRAINING_V3_START -->
### 실제 데이터 추가 실험 — real_input_capacity_20261102_152621_008750

진행 중인 실행의 완료 조건만 아래에 갱신한다. probe는 짧은 용량 측정이며 전체 학습이나 평가 완료가 아니다. 메모리%는 CUDA allocated peak / 시스템 공유 RAM ×100이며 GPU-Util과 다르다. 독립 실행 1회 기준.

|조건|가중치 갱신 방식·실험 목적|상태|고유 입력|배치|epoch|optimizer steps|실측 시간(s)|입력/s|allocated peak GiB|reserved peak GiB|공유 RAM 대비 allocated %|GPU-Util 평균%|최대%|독립 test 지표|저장 모델 재로드|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|batch_128|추론(가중치 갱신 없음)|complete|128|128|N/A|N/A|55.4481|4.6169|21.5923|22.3652|17.7440|95.3168|96.0000|N/A|N/A|
|batch_256|추론(가중치 갱신 없음)|complete|256|256|N/A|N/A|110.3050|4.6417|41.2917|42.7793|33.9326|95.6684|96.0000|N/A|N/A|
|batch_384|추론(가중치 갱신 없음)|complete|384|384|N/A|N/A|126.8018|6.0567|60.9779|63.2109|50.1102|95.1459|96.0000|N/A|N/A|
|batch_448|추론(가중치 갱신 없음)|complete|448|448|N/A|N/A|149.7532|5.9832|70.8216|73.4199|58.1996|95.4176|96.0000|N/A|N/A|
|batch_512|추론(가중치 갱신 없음)|complete|512|512|N/A|N/A|177.5420|5.7676|80.6642|83.6289|66.2880|95.4554|96.0000|N/A|N/A|
|batch_544|추론(가중치 갱신 없음)|complete|544|544|N/A|N/A|235.5105|4.6198|85.5954|88.7402|70.3403|95.6434|96.0000|N/A|N/A|
|batch_560|추론(가중치 갱신 없음)|complete|560|560|N/A|N/A|240.5824|4.6554|88.0509|91.2910|72.3582|95.8027|96.0000|N/A|N/A|
|batch_576|추론(가중치 갱신 없음)|complete|576|576|N/A|N/A|245.6093|4.6904|90.5095|93.8574|74.3786|95.7111|96.0000|N/A|N/A|
|batch_592|추론(가중치 갱신 없음)|complete|592|592|N/A|N/A|255.4055|4.6358|92.9725|96.4023|76.4027|95.5978|96.0000|N/A|N/A|
|batch_608|추론(가중치 갱신 없음)|complete|608|608|N/A|N/A|259.3899|4.6879|95.4383|98.9492|78.4290|95.7957|96.0000|N/A|N/A|
|batch_8|추론(가중치 갱신 없음)|complete|8|8|N/A|N/A|15.6182|4.6100|3.1400|3.2051|2.5804|96.0000|96.0000|N/A|N/A|
|full_training_b512|전체backbone·과제head 파인튜닝|complete|512|512|2|2|13.5239|75.7178|56.3002|57.5508|46.2662|83.8000|96.0000|{'accuracy': 0.635, 'macro_f1': 0.5918937805730258}|True|
|full_training_b8|전체backbone·과제head 파인튜닝|complete|512|8|2|128|36.4871|28.0647|9.2280|9.3340|7.5834|92.4615|96.0000|{'accuracy': 0.7125, 'macro_f1': 0.7117053879756078}|True|
|probe_training_b128|전체backbone·과제head 파인튜닝 /용량측정|complete|384|128|0|3|5.7473|66.8139|18.2551|18.7324|15.0016|74.1818|96.0000|N/A|N/A|
|probe_training_b256|전체backbone·과제head 파인튜닝 /용량측정|complete|512|256|0|2|7.1731|71.3781|30.9369|31.5840|25.4233|64.7692|96.0000|N/A|N/A|
|probe_training_b384|전체backbone·과제head 파인튜닝 /용량측정|complete|512|384|0|2|7.3745|69.4287|39.9597|40.7871|32.8380|76.2143|96.0000|N/A|N/A|
|probe_training_b448|전체backbone·과제head 파인튜닝 /용량측정|complete|512|448|0|2|7.4502|68.7230|46.5279|47.5273|38.2356|70.8571|96.0000|N/A|N/A|
|probe_training_b512|전체backbone·과제head 파인튜닝 /용량측정|complete|512|512|0|1|7.3393|69.7617|52.6411|53.7520|43.2592|79.8462|96.0000|N/A|N/A|
|probe_training_b64|전체backbone·과제head 파인튜닝 /용량측정|complete|192|64|0|3|3.2823|58.4959|11.9243|12.2891|9.7991|64.1667|96.0000|N/A|N/A|
|probe_training_b8|전체backbone·과제head 파인튜닝 /용량측정|complete|24|8|0|3|1.5871|15.1220|9.2280|9.3340|7.5834|0.0000|0.0000|N/A|N/A|

이 run의 추론 capacity 실험과 지도학습 실험은 입력 과제·길이·학습 헤드가 다를 수 있어 각각 비교한다. 3step probe는 전체 학습이 아니다. full_training은 동일한 공개 초기 backbone와 seed42,FP32,AdamW(lr1e-5,weight_decay0.01),고정 전체 split pool padding으로2epoch 수행한다. 각 학습 조건의 원본 summary에 split ID,실제 길이·padding 모양,사용 헤드·학습 범위,라벨,gradient가 관측된 파라미터 수,검증/test지표,저장 모델 재로드 차이를 기록한다. scVI는 아래 설명이 적용되지 않는 별도 처음부터 학습 모델이며 AF2·Clair3에는 이 공통 학습 절차를 적용하지 않는다.

새 학습은 기존 enhancer train512/validation128/test400,199–200nt 정답을 사용한다. 새 masked-mean+linear head는 이전 native CLS MLP head와 다른 별도 조건이며 이번 배치 비교 내에서는 고정한다. 추론은 실제 GRCh38 비중첩6000nt window1024개를 별도로 사용한다.

입력 및 메모리 기준값(같은 실행에서 측정):

|조건|실제 모델 입력 모양/정책|모델 적재 직후 allocated GiB|실행 시작 allocated GiB|peak 추가 allocated GiB|프로세스 RSS sampled peak GiB|시스템 가용 RAM sampled 최소 GiB|
|---|---|---|---|---|---|---|
|batch_128|{'input_ids': [128, 1001], 'attention_mask': [128, 1001]}|1.8565|1.9016|19.6906|1.3871|90.8226|
|batch_256|{'input_ids': [256, 1001], 'attention_mask': [256, 1001]}|1.8565|1.9016|39.3901|1.3893|70.1442|
|batch_384|{'input_ids': [384, 1001], 'attention_mask': [384, 1001]}|1.8565|1.9016|59.0763|1.3907|49.4144|
|batch_448|{'input_ids': [448, 1001], 'attention_mask': [448, 1001]}|1.8565|1.9016|68.9200|1.3913|39.0663|
|batch_512|{'input_ids': [512, 1001], 'attention_mask': [512, 1001]}|1.8565|1.9016|78.7626|1.3839|28.7434|
|batch_544|{'input_ids': [544, 1001], 'attention_mask': [544, 1001]}|1.8565|1.9016|83.6937|1.3842|23.4627|
|batch_560|{'input_ids': [560, 1001], 'attention_mask': [560, 1001]}|1.8565|1.9016|86.1493|1.3842|20.8860|
|batch_576|{'input_ids': [576, 1001], 'attention_mask': [576, 1001]}|1.8565|1.9016|88.6079|1.3842|18.1728|
|batch_592|{'input_ids': [592, 1001], 'attention_mask': [592, 1001]}|1.8565|1.9016|91.0709|1.3855|15.4026|
|batch_608|{'input_ids': [608, 1001], 'attention_mask': [608, 1001]}|1.8565|1.9016|93.5367|1.3815|13.0359|
|batch_8|{'input_ids': [8, 1001], 'attention_mask': [8, 1001]}|1.8565|1.9016|1.2384|1.4043|109.7493|
|full_training_b512|{'input_ids': [1040, 36], 'attention_mask': [1040, 36]}|1.8369|1.8369|54.4632|1.7312|53.6212|
|full_training_b8|{'input_ids': [1040, 36], 'attention_mask': [1040, 36]}|1.8369|1.8369|7.3911|1.6948|102.5719|
|probe_training_b128|{'input_ids': [1040, 36], 'attention_mask': [1040, 36]}|1.8369|1.8369|16.4182|1.7311|92.8043|
|probe_training_b256|{'input_ids': [1040, 36], 'attention_mask': [1040, 36]}|1.8369|1.8369|29.1000|1.7309|79.8491|
|probe_training_b384|{'input_ids': [1040, 36], 'attention_mask': [1040, 36]}|1.8369|1.8369|38.1228|1.7311|70.6711|
|probe_training_b448|{'input_ids': [1040, 36], 'attention_mask': [1040, 36]}|1.8369|1.8369|44.6910|1.7336|63.7909|
|probe_training_b512|{'input_ids': [1040, 36], 'attention_mask': [1040, 36]}|1.8369|1.8369|50.8042|1.6596|57.5020|
|probe_training_b64|{'input_ids': [1040, 36], 'attention_mask': [1040, 36]}|1.8369|1.8369|10.0874|1.7146|99.4670|
|probe_training_b8|{'input_ids': [1040, 36], 'attention_mask': [1040, 36]}|1.8369|1.8369|7.3911|1.6942|102.8477|

측정 상태: `메모리 제약/오류 조건 포함`; 측정범위·제약: `actual_worker_error_or_guard`.

실행 코드 블록의 파일은 원문SHA256으로 검증하며, 명령JSON에 기록된 실행 경로로 해당 snapshot을 복사하면 복원할 수 있다. 대용량 원본 데이터·가중치는 아래 전처리/다운로드 코드와 출처를 사용하고 로컬 원본도 보존한다.

로컬 원시 결과: `/home/hyein/ai-benchmark/ntv2/results/real_input_capacity_20261102_152621_008750`. 메모리 부족/guard 실패는 성공 조건으로 기재하지 않는다.
<!-- REAL_TRAINING_V3_END -->

## 4. 결과 해석

<!-- MODEL_SUMMARY_START -->
### 요약 결론

- **학습 방식:** 기본:전체backbone+원생CLS MLP 파인튜닝. 추가:전체backbone+masked-mean linear head 파인튜닝,별도 조건.
- **입력 구분:** 긴 유전체 추론window와 짧은enhancer 학습 서열을 구분. 정확도는 해당 라벨·split에서만 평가.
- **최신 추가 run 최대 성공 메모리:** `batch_608` **95.438GiB /78.43%**. 이 값은 해당 최신run의 성공 조건에서의 최대이며,다른 run의 최대와 구분한다. probe이면 전체 epoch·독립test 평가 완료를 뜻하지 않는다.
- **시간·정확도 해석:** 결과는 같은 행의 입력·배치·학습범위에만 연결한다. 메모리 상승 자체가 처리량·정확도 상승을 보장하지 않는다.
<!-- MODEL_SUMMARY_END -->

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### Nucleotide Transformer V2 — DGX Spark


실행 `ntv2_20261027_201332_251243` · 상태 **complete** · 원본 summary (원본 기록 로컬 보관: `ntv2/results/ntv2_20261027_201332_251243/summary.json`) · 실행 로그 (로컬 보관: `../results/ntv2_20261027_201332_251243/benchmark.log`) · GPU·메모리 시계열 (원본 기록 로컬 보관: `ntv2/results/ntv2_20261027_201332_251243/resources.csv`)

#### 모델과 기준 조건

공식 `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`, 리비전 `06615c1660c892fc199840c18123f8385b3542a8`. V2 500M을 대표로 선택했으며 V1·V3로 바꾸지 않았다. 다운로드된 model.safetensors의 SHA-256은 공식 LFS oid와 일치함을 검증했고 전체 출처는 `results/sources.json`에 보관했다.

실제 파라미터: {"mlm_total": 498345436, "embedding_backbone": 493084113, "mlm_head": 5261323, "classifier_total": 494135763, "classifier_trainable": 494135763}

MLM 전체 모델을 로드하되 임베딩 측정은 `model.esm` backbone의 최종 hidden state만 계산해 masked mean(특수토큰 포함)으로B×1024를 만든다. MLM head와 모든 층 hidden state를 계산/저장하는 경로와 메모리·시간이 다르다. 분류는 공식 custom EsmForSequenceClassification의 CLS MLP head를 사용하고 전 학습 가능 backbone·head를 함께 갱신했다.

환경: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "4.44.2", "cuda": "13.3", "gpu": "NVIDIA GB10"}

batch8,FP32,TF32=False,seed42,8CPU threads. 공식 fine-tuning notebook의 batch8/lr1e-5/2epoch를 참고하되 NT-v2와 enhancer binary 소규모 subset에 적용했다. notebook의1000maxsteps나 전체 데이터 재현 실험이 아니다. 추론에서도 batch8을 고정하며 배치 증가 sweep을 수행하지 않았다.

| 상태 | allocated GiB | reserved GiB | RSS GiB | system available GiB |
| --- | --- | --- | --- | --- |
| MLM 적재 직후/첫 입력 전 | 1.857 | 1.873 | 1.125 | 111.229 |

이론 FP32 가중치 용량은 MLM 파라미터수×4byte/2³⁰이며 실측과 구분한다. 공유 RAM121.688GiB에서 RSS와CUDA를 더하지 않는다. 적재 11.959초; warmup 0.596초. CPU 입력 JSON 로딩은 적재 전에 수행했다.

#### 실제 입력·라벨·분할

| split | 고유 서열 | 실제 길이 | 라벨 분포 |
| --- | --- | --- | --- |
| train | 512 | 199–200nt | {'1': 240, '0': 272} |
| validation | 128 | 199–200nt | {'1': 60, '0': 68} |
| test | 400 | 200–200nt | {'0': 200, '1': 200} |
| long_inference | 128 | 12282–12282nt | {'unlabeled': 128} |

공식 NT downstream enhancer train에서 seed42 stratified 학습/검증을 분리하고 공식 test400개는 평가 전용으로 유지했다. train/test exact sequence 중복과 train 내 중복을 제거한 뒤 부분집합을 선택했다. genomic homology 및 사전학습 genome 중복까지 제거한 것은 아니다.0=non-enhancer,1=enhancer. 원본 길이는199–200nt로 임의 정규화나 길이 증폭을 하지 않았다.

실제 학습 예: ID `10230`, label `1`, 199nt

```text
GGGCGGTGTAATGCATGGAGAGAAGCCTCTCCATAAGGAGGCAGTGGCTTACTGGCAGAGAAAGTGAGCGGTGCCTCAAGTTATGTTAGTGCGAGTCTGCTGGAAGAGAGCCTGGCGGTAAGGCTCGCTGGGAGCTCGTAATCAATTCACAGTTTAGAGGAAAGATGCTTGACTCAAACTGCCTATCTTGTTGGGGCAA
```

입력은6-mer 중심 토큰화 후 int64 input_ids/attention_mask의B×실제token수, 라벨은B 정수다. [검사 결과](Nucleotide_Transformer_V2.md#record-5c86518a2a1e)와 선택 서열·split·해시 `data/inputs.json`을 보관했다.

길이/개수 실험은 별도의 GRCh38 참조 유전체 동일 구간128개를 사용한다. BEND에서 이미 확인한 좌표 중심으로12,282nt를 추출하고 ACGT 외 문자가 없는 실제 구간을 택했다. 같은 구간의 중심 crop600/3000/6000/12276nt를 사용하며 무라벨 추론 전용이다. 음의 가닥은 reverse-complement했다. 짧은 enhancer 학습 데이터의 정확도와 이 긴 무라벨 서열을 혼동하지 않는다.

#### 입력별 GPU 사용률·메모리

| 조건 | nt | 고유 개수 | 평균token | 반복 | 초 | 건/초 | GPU 평균/최대% | 시작GiB | peakGiB | 증가GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_epoch1 | 199–200 | 512 | 학습 입력 | 1 | 19.540 | 26.203 | 92.314/95.000 | 1.873 | 9.250 | 7.377 |
| train_epoch2 | 199–200 | 512 | 학습 입력 | 1 | 18.586 | 27.548 | 92.500/95.000 | 7.409 | 9.252 | 1.843 |
| inference_bp3000_n8 | 3000 | 8 | 501.0 | 21 | 15.750 | 10.666 | 90.138/95.000 | 1.889 | 2.275 | 0.386 |
| inference_bp3000_n32 | 3000 | 32 | 501.0 | 5 | 15.062 | 10.623 | 95.000/95.000 | 1.895 | 2.275 | 0.380 |
| inference_bp3000_n128 | 3000 | 128 | 501.0 | 2 | 24.124 | 10.612 | 95.000/95.000 | 1.895 | 2.275 | 0.380 |
| inference_bp600_n32 | 600 | 32 | 101.0 | 35 | 15.398 | 72.738 | 94.655/95.000 | 1.895 | 1.958 | 0.063 |
| inference_bp6000_n32 | 6000 | 32 | 1001.0 | 3 | 22.126 | 4.339 | 94.659/95.000 | 1.889 | 3.140 | 1.251 |
| inference_bp12276_n32 | 12276 | 32 | 2047.0 | 1 | 20.474 | 1.563 | 93.237/95.000 | 1.902 | 6.475 | 4.573 |

실측 tokenizer 결과: {"600": {"min": 101, "max": 101, "mean": 101.0}, "3000": {"min": 501, "max": 501, "mean": 501.0}, "6000": {"min": 1001, "max": 1001, "mean": 1001.0}, "12276": {"min": 2047, "max": 2047, "mean": 2047.0}}

추론은 최소15초 조건 반복, 고유8/32/128개와 총처리수는 구분했다. 길이 비교는같은32개, 개수 비교는같은3000nt의 중첩 부분집합이다. truncation=False이며 실제 token이2048이하임을 assert했다. 각 조건 CUDA peak를 reset하고 cache를 비운 뒤 같은 run의 시작 allocated를 기준으로 추가량을 계산했다.

시간은 메모리에 준비한 원본 서열의 CPU 토큰화·GPU 전송·forward·pooling까지 포함하며 결과는 GPU tensor다. 반복 구간의 CPU 결과 반환/디스크 저장은 제외한다. 학습은 토큰화·전송·forward·loss·backward·AdamW까지 포함하고 epoch별 validation은 제외했다. CUDA 경계를 synchronize했다. 디스크 입력 준비와 파일 저장은 제외하며 저장은 별도 시간으로 기록했다. compile은 warmup에 섞여 별도값N/A다.

GPU-Util은 장치 활동시간 비율로 FLOPS/메모리 비율이 아니다.0.5초 대기+조회 구간이 측정 안에 완전히 포함된 표본만 집계하며 디스플레이 영향이 포함될 수 있다. RSS/available/allocated/reserved 시계열은 sampled값이다. 짧은 epoch의 평균은 표본수와 함께 해석한다.

#### 파인튜닝·저장·독립 test

| epoch | train loss | validation Accuracy | F1 | MCC | AUROC |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.608 | 0.836 | 0.837 | 0.680 | 0.909 |
| 2 | 0.231 | 0.891 | 0.881 | 0.780 | 0.967 |

최종 epoch2를 고정 평가했고 test로 epoch/조건을 선택하지 않았다. test400개: {"accuracy": 0.6975, "f1": 0.6629526462395543, "mcc": 0.40357105287424294, "auroc": 0.7861750000000001}

실제 gradient 관찰: {"numel_with_gradient": 492036098, "no_gradient_names": ["esm.embeddings.position_embeddings.weight", "esm.contact_head.regression.weight", "esm.contact_head.regression.bias"]}

backbone과head 변경 검증: {"backbone_max_abs": 0.0010109208524227142, "head_max_abs": 0.001017133705317974}

모델 저장: `models/ntv2_20261027_201332_251243/classifier`; 저장 36.483초. optimizer/RNG/epoch도 저장했지만 학습 재개는 검증하지 않았다. test logits와 실제 예측 라벨을 보존했다. 독립 프로세스 재로드: {"independent_process": true, "test_inputs": 400, "max_abs_difference": 0.0, "predictions_exact_match": true, "passed": true}

검증 Accuracy0.8906에 비해 공식 test Accuracy0.6975로 낮아, 검증 성능을 외부 일반화 성능으로 대체해 보고하지 않는다. 첫 독립 재로드에서 TF32 기본 설정 불일치로logits차이0.00575가 나타났으나 분류 라벨은 모두 일치했고, 원 FP32/TF32=False로 맞춰 재검증했다.

소규모2epoch 실행 검증으로, 전체 benchmark 재현이나 최고의 enhancer 정확도를 주장하지 않는다.

복구 이력: 첫 학습 진입에서 rotary cache/inference_mode 충돌이 발생해 validation을 no_grad로 바꾸고 학습을 새 체크포인트에서 복구했다. 첫 복구 시 초기 추론 시계열 파일을 덮어쓴 문제가 있어, 추론6조건을 별도 프로세스로 재계측하고 학습은 재실행하지 않았다. 최종 표는 재계측 추론과 완료된2epoch 학습이다. 자원 CSV는 단계마다 별도 파일을 보존한다. 실패 로그와 복구 전 summary도 남겼다.

#### 해석과 공식 출처

길이가 커질 때 attention 중간값 등 추가 메모리가 커질 수 있지만, 데이터 개수가 늘어도 batch정책이 같으면 순간 peak가비례 증가하지 않는다. 공유RAM90% 채우기는실험목표가아니다.

- [공식 NT-v2 모델 카드](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)
- [공식 소스·학습 예제 링크](https://github.com/instadeepai/nucleotide-transformer)
- [공식 enhancer 데이터](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)
- [공식 분류 fine-tuning notebook](https://github.com/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### ntv2 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_143713_869625` · 측정범위·제약 `available_unique_input_limit`

#### 입력과 고정 조건

{
  "source": "existing real GRCh38 regions, central6000nt crops -> native ~1001tokens",
  "unique_available_inputs": 128,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

#### 배치별 측정

측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.

|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|78|78|15.1193|5.159|1.8565|2.0578|0.2013|2.0781|1.691|90.7931|96.0|1.3639|109.9901|
|2|36|72|15.3145|4.7014|1.8565|2.2095|0.3529|2.2578|1.8157|95.4286|96.0|1.3637|109.5496|
|4|18|72|15.7872|4.5607|1.8565|2.5201|0.6636|2.5547|2.071|95.5333|96.0|1.3639|109.1396|
|8|9|72|16.122|4.466|1.8565|3.14|1.2835|3.2051|2.5804|95.4667|96.0|1.3638|108.3269|
|16|5|80|17.8771|4.475|1.8565|4.3629|2.5064|4.4902|3.5854|95.3636|96.0|1.3638|107.0482|
|32|3|96|21.3241|4.5019|1.8565|6.8339|4.9773|7.0371|5.6159|95.4872|96.0|1.3637|104.5106|
|64|2|128|28.8515|4.4365|1.8565|11.7469|9.8904|12.1543|9.6534|95.3585|96.0|1.3638|99.4116|
|96|2|192|42.5332|4.5141|1.8565|16.6768|14.8203|17.2461|13.7046|93.1772|96.0|1.364|94.6979|
|128|2|256|56.8039|4.5067|1.8565|21.5923|19.7357|22.3652|17.744|94.8269|96.0|1.364|89.7241|

#### 결과 해석과 검증

최대시험성공batch 128. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) · [기존 모델 보고서 디렉터리](Nucleotide_Transformer_V2.md)

### 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.


## 6. 실제 실행 코드·명령어 전문

<!-- EXECUTED_SCRIPTS_START -->

### 원문과 출처 검증

아래 코드 블록은 원본 내용을 생략하거나 재작성하지 않았다. 각 파일의 원래 경로·원문 SHA256·출처 상태를 함께 기록했다. `실행 run 저장 snapshot`을 우선하고, 현재 소스만 남은 경우에는 당시 파일과 같다고 단정하지 않는다. 동일 바이트 버전은 한 번만 수록하고 해당 경로들을 함께 표시한다.

GitHub에는 `.py`/`.sh`를 별도 파일로 두지 않는다. 코드 블록을 원래 경로에 저장하거나 [복원 안내](Nucleotide_Transformer_V2.md)의 코드로 로컬에서 복원한다. 라이브러리 전체 소스는 이 사용자 실행 스크립트와 구분하며 패키지 버전/출처를 따른다.

### 코드 목록

|원래 파일|원문 SHA256|출처 상태|
|---|---|---|
|`ntv2/scripts/benchmark.py`|`be49bb893e908f3438d91e2b6431a2a25782031c258e8a4c72824f4f4060102b`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/discover.py`|`a9fbd2115bef84a39db1d46e118746f0c59ae67016760723252cb8fadc381572`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/finetune.py`|`8b2557f1f9c116146dbb262496379dac60f5adcd1395e9b890a34eba2d9f0e87`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/inference_remeasure.py`|`16cbf5cc2d7f96d71cd1d2a2dc37a347d9d5fc4c9077ff7a3614a651aa95ebc4`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/prepare.py`|`5fb03badb6d1462dbf126094b4ba43d5d6d43465a4c42a5fb3dec95eeaff3fb2`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/reload.py`|`4cd7bfc5003bbc767a0c501defc4fbf5e416f1cd1d109a7bf2a6583ad2821dc4`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/report.py`|`6dedc5616c522f8c1e0ce15ade7421f98ca9ea361787d373e8525a9b0f905657`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/setup.py`|`a366ab833d086bdeffe8b27c8dae210a62f53c2721339fbd02dce08ff0218cdb`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|
|`scripts/remaining_followups/launch.py`|`84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/report.py`|`8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/worker.py`|`e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|

<details>
<summary>ntv2/scripts/benchmark.py — 원문 전문 (7,687 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/benchmark.py", "sha256": "be49bb893e908f3438d91e2b6431a2a25782031c258e8a4c72824f4f4060102b", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,os,time,json,gc,random,hashlib
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,torch
from transformers import AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'models/checkpoint';torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'batch':8,'dtype':'FP32','tf32':False,'seed':42,'lr':1e-5,'weight_decay':0.,'epochs':2,'train':512,'validation':128,'test':400,'inference':'MLM model.esm backbone, masked mean including special tokens; MLM logits not computed','finetuning':'official custom EsmForSequenceClassification CLS MLP head; all trainable parameters'}}
def save():write_json(D/'summary.json',S)
def encode(seqs):
 a=tokenizer(seqs,padding=True,truncation=False,return_tensors='pt');lens=a['attention_mask'].sum(1).tolist();assert max(lens)<=2048,(max(lens),len(seqs[0]));return {k:a[k].cuda() for k in ['input_ids','attention_mask']},lens
try:
 inputs=json.loads((R/'data/inputs.json').read_text());tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);tok={}
 for bp in [600,3000,6000,12276]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference']];v=tokenizer(seqs,padding=False,truncation=False)['input_ids'];lens=list(map(len,v));assert max(lens)<=2048;tok[str(bp)]={'min':min(lens),'max':max(lens),'mean':float(np.mean(lens))}
 S['token_audit']=tok;S['input_audit']='data/input_audit.json';save();meter.phase='loading_mlm';t=time.time();model=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True).float().cuda().eval();torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'mlm_total':sum(p.numel() for p in model.parameters()),'embedding_backbone':sum(p.numel() for p in model.esm.parameters()),'mlm_head':sum(p.numel() for p in model.lm_head.parameters())};save()
 def embedding(seqs):
  x,lens=encode(seqs);h=model.esm(**x,return_dict=True).last_hidden_state;m=x['attention_mask'].unsqueeze(-1);z=(h*m).sum(1)/m.sum(1);return z,lens
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():
  for _ in range(3):z,_=embedding([r['sequence'][:600] for r in inputs['long_inference'][:8]])
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;np.save(D/'example_embeddings.npy',z.cpu().numpy());del z
 for bp,n in [(3000,8),(3000,32),(3000,128),(600,32),(6000,32),(12276,32)]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference'][:n]];name=f'inference_bp{bp}_n{n}';t,base=meter.begin(name);repeats=0;lengths=[]
  with torch.inference_mode():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,8):z,lens=embedding(seqs[i:i+8]);lengths.extend(lens);del z
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='inference',bp=bp,unique_inputs=n,repeats=repeats,processed=n*repeats,tokens_min=min(lengths),tokens_mean=float(np.mean(lengths)),tokens_max=max(lengths));row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',json.dumps(row),flush=True);save()
 del model;gc.collect();torch.cuda.empty_cache();torch.manual_seed(42);meter.phase='loading_classifier';t=time.time();model=AutoModelForSequenceClassification.from_pretrained(P,trust_remote_code=True,local_files_only=True,num_labels=2).float().cuda();torch.cuda.synchronize();S['classifier_loading_seconds']=time.time()-t;S['classifier_model_only']=meter.memory();S['parameters']['classifier_total']=sum(p.numel() for p in model.parameters());S['parameters']['classifier_trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad);save()
 def evaluate(rows):
  model.eval();logits=[]
  with torch.no_grad():
   for i in range(0,len(rows),8):
    x,_=encode([r['sequence'] for r in rows[i:i+8]]);z=model(**x).logits;logits.append(z.cpu().numpy())
  p=np.concatenate(logits);y=np.array([r['label'] for r in rows]);pred=p.argmax(1);prob=torch.from_numpy(p).softmax(-1)[:,1].numpy();return {'accuracy':float(accuracy_score(y,pred)),'f1':float(f1_score(y,pred)),'mcc':float(matthews_corrcoef(y,pred)),'auroc':float(roc_auc_score(y,prob))},p
 meter.phase='pretrain_validation';S['initial_validation'],_=evaluate(inputs['validation']);before={'backbone':model.esm.encoder.layer[-1].output.dense.weight.detach().cpu().clone(),'head':next(model.classifier.parameters()).detach().cpu().clone()};opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.);S['epochs']=[];grad=None
 for epoch in range(2):
  model.train();order=np.random.default_rng(42+epoch).permutation(512);losses=[];t,base=meter.begin(f'train_epoch{epoch+1}')
  for step in range(0,512,8):
   rows=[inputs['train'][int(i)] for i in order[step:step+8]];x,lens=encode([r['sequence'] for r in rows]);labels=torch.tensor([r['label'] for r in rows],device='cuda');opt.zero_grad(set_to_none=True);result=model(**x,labels=labels);loss=result.loss;assert torch.isfinite(loss);loss.backward()
   if grad is None:grad={'numel_with_gradient':sum(p.numel() for p in model.parameters() if p.grad is not None),'no_gradient_names':[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]}
   opt.step();losses.append(float(loss.detach()));del result,loss,x,labels
   if step%128==0:print('TRAIN',epoch+1,step,'loss',losses[-1],flush=True)
  row=meter.end(t,base,name=f'train_epoch{epoch+1}',mode='training',unique_inputs=512,repeats=1,processed=512,steps=64,loss_mean=float(np.mean(losses)),losses=losses);row['throughput']=512/row['seconds'];S['conditions'].append(row);meter.phase='validation';val,log=evaluate(inputs['validation']);S['epochs'].append({'epoch':epoch+1,'training':row,'validation':val});print('EPOCH',epoch+1,val,flush=True);save()
 S['gradient_observation']=grad;S['parameter_changes']={'backbone_max_abs':float((model.esm.encoder.layer[-1].output.dense.weight.detach().cpu()-before['backbone']).abs().max()),'head_max_abs':float((next(model.classifier.parameters()).detach().cpu()-before['head']).abs().max())};meter.phase='saving';t=time.time();dest=M/'classifier';model.save_pretrained(dest,safe_serialization=True);tokenizer.save_pretrained(dest);torch.save({'optimizer':opt.state_dict(),'epoch':2,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state()},M/'optimizer_rng.pt');S['save_seconds']=time.time()-t;S['saved_model']=str(dest.relative_to(R));write_json(M/'task.json',{'labels':{'0':'non-enhancer','1':'enhancer'},'settings':S['settings'],'resume_tested':False,'source_checkpoint':'models/checkpoint','input_split':'data/inputs.json'});meter.phase='test';t=time.time();S['test'],logits=evaluate(inputs['test']);torch.cuda.synchronize();S['test_seconds']=time.time()-t;np.save(D/'test_logits.npy',logits);write_json(D/'test_predictions.json',[{'id':r['id'],'label':r['label'],'prediction':int(p.argmax())} for r,p in zip(inputs['test'],logits)]);S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['error']=repr(e);S['status']='failed';save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/discover.py — 원문 전문 (725 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/discover.py", "sha256": "a9fbd2115bef84a39db1d46e118746f0c59ae67016760723252cb8fadc381572", "encoding": "utf-8", "added_final_newline": false} -->
```python
import json,urllib.request
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/ntv2')
for kind,repo in [('models','InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'),('datasets','InstaDeepAI/nucleotide_transformer_downstream_tasks')]:
 meta=json.load(urllib.request.urlopen('https://huggingface.co/api/'+kind+'/'+repo,timeout=60));tree=json.load(urllib.request.urlopen('https://huggingface.co/api/'+kind+'/'+repo+'/tree/'+meta['sha']+'?recursive=true',timeout=60));(R/('models' if kind=='models' else 'data')/'repository.json').write_text(json.dumps({'meta':meta,'tree':tree},indent=2));print(kind,meta['sha'],[(r['path'],r.get('size')) for r in tree if kind=='models' or 'enhancers/' in r['path']],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/finetune.py — 원문 전문 (5,939 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/finetune.py", "sha256": "8b2557f1f9c116146dbb262496379dac60f5adcd1395e9b890a34eba2d9f0e87", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,os,time,json,gc,random,hashlib
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,torch
from transformers import AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'models/checkpoint';torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'batch':8,'dtype':'FP32','tf32':False,'seed':42,'lr':1e-5,'weight_decay':0.,'epochs':2,'train':512,'validation':128,'test':400,'inference':'MLM model.esm backbone, masked mean including special tokens; MLM logits not computed','finetuning':'official custom EsmForSequenceClassification CLS MLP head; all trainable parameters'}}
def save():write_json(D/'summary.json',S)
def encode(seqs):
 a=tokenizer(seqs,padding=True,truncation=False,return_tensors='pt');lens=a['attention_mask'].sum(1).tolist();assert max(lens)<=2048,(max(lens),len(seqs[0]));return {k:a[k].cuda() for k in ['input_ids','attention_mask']},lens
try:
 inputs=json.loads((R/'data/inputs.json').read_text());tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);old=json.loads((D/'summary.json').read_text());write_json(D/'summary_before_recovery.json',old);S=old;S['status']='recovering_finetuning';S['recovery']='rotary cache created under inference_mode cannot be reused for autograd; validation now uses no_grad; six completed inference conditions retained';S.pop('error',None)
 gc.collect();torch.cuda.empty_cache();torch.manual_seed(42);meter.phase='loading_classifier';t=time.time();model=AutoModelForSequenceClassification.from_pretrained(P,trust_remote_code=True,local_files_only=True,num_labels=2).float().cuda();torch.cuda.synchronize();S['classifier_loading_seconds']=time.time()-t;S['classifier_model_only']=meter.memory();S['parameters']['classifier_total']=sum(p.numel() for p in model.parameters());S['parameters']['classifier_trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad);save()
 def evaluate(rows):
  model.eval();logits=[]
  with torch.no_grad():
   for i in range(0,len(rows),8):
    x,_=encode([r['sequence'] for r in rows[i:i+8]]);z=model(**x).logits;logits.append(z.cpu().numpy())
  p=np.concatenate(logits);y=np.array([r['label'] for r in rows]);pred=p.argmax(1);prob=torch.from_numpy(p).softmax(-1)[:,1].numpy();return {'accuracy':float(accuracy_score(y,pred)),'f1':float(f1_score(y,pred)),'mcc':float(matthews_corrcoef(y,pred)),'auroc':float(roc_auc_score(y,prob))},p
 meter.phase='pretrain_validation';S['initial_validation'],_=evaluate(inputs['validation']);before={'backbone':model.esm.encoder.layer[-1].output.dense.weight.detach().cpu().clone(),'head':next(model.classifier.parameters()).detach().cpu().clone()};opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.);S['epochs']=[];grad=None
 for epoch in range(2):
  model.train();order=np.random.default_rng(42+epoch).permutation(512);losses=[];t,base=meter.begin(f'train_epoch{epoch+1}')
  for step in range(0,512,8):
   rows=[inputs['train'][int(i)] for i in order[step:step+8]];x,lens=encode([r['sequence'] for r in rows]);labels=torch.tensor([r['label'] for r in rows],device='cuda');opt.zero_grad(set_to_none=True);result=model(**x,labels=labels);loss=result.loss;assert torch.isfinite(loss);loss.backward()
   if grad is None:grad={'numel_with_gradient':sum(p.numel() for p in model.parameters() if p.grad is not None),'no_gradient_names':[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]}
   opt.step();losses.append(float(loss.detach()));del result,loss,x,labels
   if step%128==0:print('TRAIN',epoch+1,step,'loss',losses[-1],flush=True)
  row=meter.end(t,base,name=f'train_epoch{epoch+1}',mode='training',unique_inputs=512,repeats=1,processed=512,steps=64,loss_mean=float(np.mean(losses)),losses=losses);row['throughput']=512/row['seconds'];S['conditions'].append(row);meter.phase='validation';val,log=evaluate(inputs['validation']);S['epochs'].append({'epoch':epoch+1,'training':row,'validation':val});print('EPOCH',epoch+1,val,flush=True);save()
 S['gradient_observation']=grad;S['parameter_changes']={'backbone_max_abs':float((model.esm.encoder.layer[-1].output.dense.weight.detach().cpu()-before['backbone']).abs().max()),'head_max_abs':float((next(model.classifier.parameters()).detach().cpu()-before['head']).abs().max())};meter.phase='saving';t=time.time();dest=M/'classifier';model.save_pretrained(dest,safe_serialization=True);tokenizer.save_pretrained(dest);torch.save({'optimizer':opt.state_dict(),'epoch':2,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state()},M/'optimizer_rng.pt');S['save_seconds']=time.time()-t;S['saved_model']=str(dest.relative_to(R));write_json(M/'task.json',{'labels':{'0':'non-enhancer','1':'enhancer'},'settings':S['settings'],'resume_tested':False,'source_checkpoint':'models/checkpoint','input_split':'data/inputs.json'});meter.phase='test';t=time.time();S['test'],logits=evaluate(inputs['test']);torch.cuda.synchronize();S['test_seconds']=time.time()-t;np.save(D/'test_logits.npy',logits);write_json(D/'test_predictions.json',[{'id':r['id'],'label':r['label'],'prediction':int(p.argmax())} for r,p in zip(inputs['test'],logits)]);S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['error']=repr(e);S['status']='failed';save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/inference_remeasure.py — 원문 전문 (4,418 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/inference_remeasure.py", "sha256": "16cbf5cc2d7f96d71cd1d2a2dc37a347d9d5fc4c9077ff7a3614a651aa95ebc4", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,os,time,json,gc,random,hashlib
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,torch
from transformers import AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'models/checkpoint';torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'batch':8,'dtype':'FP32','tf32':False,'seed':42,'lr':1e-5,'weight_decay':0.,'epochs':2,'train':512,'validation':128,'test':400,'inference':'MLM model.esm backbone, masked mean including special tokens; MLM logits not computed','finetuning':'official custom EsmForSequenceClassification CLS MLP head; all trainable parameters'}}
def save():write_json(D/'summary.json',S)
def encode(seqs):
 a=tokenizer(seqs,padding=True,truncation=False,return_tensors='pt');lens=a['attention_mask'].sum(1).tolist();assert max(lens)<=2048,(max(lens),len(seqs[0]));return {k:a[k].cuda() for k in ['input_ids','attention_mask']},lens
try:
 previous=json.loads((D/'summary.json').read_text());write_json(D/'summary_before_remeasurement.json',previous);S=previous;oldparams=dict(S['parameters']);S['conditions']=[r for r in S['conditions'] if r['mode']=='training'];S['status']='remeasuring_inference';
 inputs=json.loads((R/'data/inputs.json').read_text());tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);tok={}
 for bp in [600,3000,6000,12276]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference']];v=tokenizer(seqs,padding=False,truncation=False)['input_ids'];lens=list(map(len,v));assert max(lens)<=2048;tok[str(bp)]={'min':min(lens),'max':max(lens),'mean':float(np.mean(lens))}
 S['token_audit']=tok;S['input_audit']='data/input_audit.json';save();meter.phase='loading_mlm';t=time.time();model=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True).float().cuda().eval();torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'mlm_total':sum(p.numel() for p in model.parameters()),'embedding_backbone':sum(p.numel() for p in model.esm.parameters()),'mlm_head':sum(p.numel() for p in model.lm_head.parameters())};save()
 def embedding(seqs):
  x,lens=encode(seqs);h=model.esm(**x,return_dict=True).last_hidden_state;m=x['attention_mask'].unsqueeze(-1);z=(h*m).sum(1)/m.sum(1);return z,lens
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():
  for _ in range(3):z,_=embedding([r['sequence'][:600] for r in inputs['long_inference'][:8]])
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;np.save(D/'example_embeddings.npy',z.cpu().numpy());del z
 for bp,n in [(3000,8),(3000,32),(3000,128),(600,32),(6000,32),(12276,32)]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference'][:n]];name=f'inference_bp{bp}_n{n}';t,base=meter.begin(name);repeats=0;lengths=[]
  with torch.inference_mode():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,8):z,lens=embedding(seqs[i:i+8]);lengths.extend(lens);del z
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='inference',bp=bp,unique_inputs=n,repeats=repeats,processed=n*repeats,tokens_min=min(lengths),tokens_mean=float(np.mean(lengths)),tokens_max=max(lengths));row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',json.dumps(row),flush=True);save()
 S['parameters']={**oldparams,**S['parameters']};S['resource_files']=[p.name for p in D.glob('resources*.csv')];S['recovery_note']='Initial inference CSV was overwritten by first recovery. Inference six conditions remeasured in fresh process, training retained, each CSV now unique.';S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['error']=repr(e);S['status']='failed';save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/prepare.py — 원문 전문 (3,379 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/prepare.py", "sha256": "5fb03badb6d1462dbf126094b4ba43d5d6d43465a4c42a5fb3dec95eeaff3fb2", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,hashlib,collections
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
R=Path('/work');src=Path('/source/external_data/BEND');out={};audit={};official={}
for split in ['train','test']:
 df=pd.read_parquet(R/f'data/enhancers/{split}.parquet');print(split,df.shape,df.columns.tolist(),df.iloc[0].to_dict(),flush=True);official[split]=df
tr=official['train'];te=official['test'];assert {'sequence','label'}<=set(tr.columns)
for df in [tr,te]:assert not df[['sequence','label']].isnull().any().any() and set(df.label.unique())=={0,1}
testseq=set(te.sequence);tr=tr.loc[~tr.sequence.isin(testseq)].drop_duplicates('sequence').copy();te=te.drop_duplicates('sequence').copy();trainids,validids=train_test_split(np.arange(len(tr)),test_size=.15,stratify=tr.label,random_state=42)
def subset(df,indices,n):
 ids,_=train_test_split(indices,train_size=n,stratify=df.iloc[indices].label,random_state=42) if len(indices)>n else (indices,[])
 return [{'id':str(df.iloc[i].get('id',df.index[i])),'sequence':df.iloc[i].sequence,'label':int(df.iloc[i].label),'sequence_sha256':hashlib.sha256(df.iloc[i].sequence.encode()).hexdigest()} for i in ids]
out['train']=subset(tr,trainids,512);out['validation']=subset(tr,validids,128);out['test']=subset(te,np.arange(len(te)),min(400,len(te)))
sets=[set(r['sequence_sha256'] for r in out[k]) for k in out];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
manifest=json.loads((src/'prepared/20261012_001503_n1000_seed42/manifest.json').read_text());fa=src/'genomes/GRCh38.primary_assembly.genome.fa';idx={}
for line in Path(str(fa)+'.fai').read_text().splitlines():
 name,length,offset,bases,width,*_=line.split('\t');idx[name]=tuple(map(int,[length,offset,bases,width]))
def getseq(f,chrom,start,end):
 if chrom not in idx:chrom=chrom.removeprefix('chr')
 length,off,bases,width=idx[chrom];assert start>=0 and end<=length
 lo=off+start//bases*width+start%bases;hi=off+(end-1)//bases*width+(end-1)%bases+1;f.seek(lo);return f.read(hi-lo).replace(b'\n',b'').replace(b'\r',b'').decode().upper()
long=[]
with fa.open('rb') as f:
 for pos in manifest['positions']:
  center=pos['center'];seq=getseq(f,pos['chromosome'],center-6141,center+6141)
  if len(seq)!=12282 or not set(seq)<=set('ACGT'):continue
  if pos['strand']=='-':seq=seq.translate(str.maketrans('ACGT','TGCA'))[::-1]
  long.append(dict(id=pos['id'],sequence=seq,chromosome=pos['chromosome'],center=center,strand=pos['strand']))
  if len(long)==128:break
assert len(long)==128;out['long_inference']=long
for k,rows in out.items():audit[k]={'n':len(rows),'length_min':min(len(r['sequence']) for r in rows),'length_max':max(len(r['sequence']) for r in rows),'labels':dict(collections.Counter(r.get('label','unlabeled') for r in rows)),'first_example':rows[0]}
(R/'data/inputs.json').write_text(json.dumps(out,indent=2));(R/'data/input_audit.json').write_text(json.dumps({'splits':audit,'official_train_rows':len(official['train']),'official_test_rows':len(official['test']),'train_after_test_overlap_and_duplicates_removed':len(tr),'policy':'official test untouched for selection; train/val stratified seed42; exact sequence hash disjoint; long unlabeled reference regions inference only'},indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a!='first_example'} for k,v in audit.items()},indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/reload.py — 원문 전문 (1,382 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/reload.py", "sha256": "4cd7bfc5003bbc767a0c501defc4fbf5e416f1cd1d109a7bf2a6583ad2821dc4", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,sys,time
sys.path.insert(0,'/support')
from gpu_support import write_json
import torch,numpy as np
from transformers import AutoModelForSequenceClassification,AutoTokenizer
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());p=R/s['saved_model'];torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;m=AutoModelForSequenceClassification.from_pretrained(p,trust_remote_code=True,local_files_only=True).float().cuda().eval();tok=AutoTokenizer.from_pretrained(p,trust_remote_code=True,local_files_only=True);rows=json.loads((R/'data/inputs.json').read_text())['test'];pred=[]
with torch.inference_mode():
 for i in range(0,len(rows),8):
  x=tok([r['sequence'] for r in rows[i:i+8]],padding=True,truncation=False,return_tensors='pt');pred.append(m(**{k:x[k].cuda() for k in ['input_ids','attention_mask']}).logits.cpu().numpy())
y=np.concatenate(pred);ref=np.load(D/'test_logits.npy');diff=float(np.abs(y-ref).max());r={'independent_process':True,'test_inputs':len(rows),'max_abs_difference':diff,'predictions_exact_match':bool(np.array_equal(y.argmax(-1),ref.argmax(-1))),'passed':bool(diff<=1e-5)};write_json(D/'reload.json',r);assert r['passed'];s['reload']=r;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(r),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/report.py — 원문 전문 (9,157 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/report.py", "sha256": "6dedc5616c522f8c1e0ce15ade7421f98ca9ea361787d373e8525a9b0f905657", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import json,csv,hashlib
R=Path('/home/hyein/ai-benchmark/ntv2');run=(R/'results/LATEST').read_text().strip();D=R/'results'/run;s=json.loads((D/'summary.json').read_text());audit=json.loads((R/'data/input_audit.json').read_text());meta=json.loads((R/'models/repository.json').read_text());out=[]
def a(t=''):out.append(t)
def table(h,rows):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rows:a('| '+' | '.join(map(str,r))+' |')
 a()
def f(x):return 'N/A' if x is None else f'{x:.3f}'
a('# Nucleotide Transformer V2 — DGX Spark');a();a(f'실행 `{run}` · 상태 **{s["status"]}** · [원본 summary](../results/{run}/summary.json) · [실행 로그](../results/{run}/benchmark.log) · [GPU·메모리 시계열](../results/{run}/resources.csv)');a();a('## 모델과 기준 조건');a();a('공식 `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`, 리비전 `'+meta['meta']['sha']+'`. V2 500M을 대표로 선택했으며 V1·V3로 바꾸지 않았다. 다운로드된 model.safetensors의 SHA-256은 공식 LFS oid와 일치함을 검증했고 전체 출처는 `results/sources.json`에 보관했다.');a();a('실제 파라미터: '+json.dumps(s.get('parameters',{}),ensure_ascii=False));a();a('MLM 전체 모델을 로드하되 임베딩 측정은 `model.esm` backbone의 최종 hidden state만 계산해 masked mean(특수토큰 포함)으로B×1024를 만든다. MLM head와 모든 층 hidden state를 계산/저장하는 경로와 메모리·시간이 다르다. 분류는 공식 custom EsmForSequenceClassification의 CLS MLP head를 사용하고 전 학습 가능 backbone·head를 함께 갱신했다.');a();a('환경: '+json.dumps(s['environment']));a();a('batch8,FP32,TF32=False,seed42,8CPU threads. 공식 fine-tuning notebook의 batch8/lr1e-5/2epoch를 참고하되 NT-v2와 enhancer binary 소규모 subset에 적용했다. notebook의1000maxsteps나 전체 데이터 재현 실험이 아니다. 추론에서도 batch8을 고정하며 배치 증가 sweep을 수행하지 않았다.');a();b=s['model_only_postload'];table(['상태','allocated GiB','reserved GiB','RSS GiB','system available GiB'],[['MLM 적재 직후/첫 입력 전',*[f(b[k]) for k in ['allocated_gib','reserved_gib','rss_gib','system_available_gib']]]]);a('이론 FP32 가중치 용량은 MLM 파라미터수×4byte/2³⁰이며 실측과 구분한다. 공유 RAM'+f(b['system_total_gib'])+'GiB에서 RSS와CUDA를 더하지 않는다. 적재 '+f(s['loading_seconds'])+'초; warmup '+f(s.get('warmup_seconds'))+'초. CPU 입력 JSON 로딩은 적재 전에 수행했다.');a();a('## 실제 입력·라벨·분할');a();rows=[]
for k,v in audit['splits'].items():rows.append([k,v['n'],f"{v['length_min']}–{v['length_max']}nt",v['labels']])
table(['split','고유 서열','실제 길이','라벨 분포'],rows);a('공식 NT downstream enhancer train에서 seed42 stratified 학습/검증을 분리하고 공식 test400개는 평가 전용으로 유지했다. train/test exact sequence 중복과 train 내 중복을 제거한 뒤 부분집합을 선택했다. genomic homology 및 사전학습 genome 중복까지 제거한 것은 아니다.0=non-enhancer,1=enhancer. 원본 길이는199–200nt로 임의 정규화나 길이 증폭을 하지 않았다.');a();r=audit['splits']['train']['first_example'];a('실제 학습 예: ID `'+r['id']+'`, label `'+str(r['label'])+'`, '+str(len(r['sequence']))+'nt');a();a('```text\n'+r['sequence']+'\n```');a();a('입력은6-mer 중심 토큰화 후 int64 input_ids/attention_mask의B×실제token수, 라벨은B 정수다. [검사 결과](../data/input_audit.json)와 선택 서열·split·해시 `data/inputs.json`을 보관했다.');a();a('길이/개수 실험은 별도의 GRCh38 참조 유전체 동일 구간128개를 사용한다. BEND에서 이미 확인한 좌표 중심으로12,282nt를 추출하고 ACGT 외 문자가 없는 실제 구간을 택했다. 같은 구간의 중심 crop600/3000/6000/12276nt를 사용하며 무라벨 추론 전용이다. 음의 가닥은 reverse-complement했다. 짧은 enhancer 학습 데이터의 정확도와 이 긴 무라벨 서열을 혼동하지 않는다.');a();a('## 입력별 GPU 사용률·메모리');a();rs=[];flat=[]
for r in s['conditions']:
 bp=r.get('bp','199–200');n=r['unique_inputs'];m=r['memory'];rs.append([r['name'],bp,n,r.get('tokens_mean','학습 입력'),r['repeats'],f(r['seconds']),f(r['throughput']),f(r['gpu_util_avg'])+'/'+f(r['gpu_util_max']),f(r['baseline']['allocated_gib']),f(m['peak_allocated_gib']),f(r['peak_delta_gib'])]);flat.append({k:v for k,v in r.items() if k not in ['memory','baseline','losses']});flat[-1].update(allocated_baseline_gib=r['baseline']['allocated_gib'],peak_allocated_gib=m['peak_allocated_gib'],peak_reserved_gib=m['peak_reserved_gib'])
table(['조건','nt','고유 개수','평균token','반복','초','건/초','GPU 평균/최대%','시작GiB','peakGiB','증가GiB'],rs)
if flat:
 keys=list(dict.fromkeys(k for r in flat for k in r))
 with (R/'reports/comparison.csv').open('w',newline='') as f0:w=csv.DictWriter(f0,fieldnames=keys);w.writeheader();w.writerows(flat)
a('실측 tokenizer 결과: '+json.dumps(s.get('token_audit',{})));a();a('추론은 최소15초 조건 반복, 고유8/32/128개와 총처리수는 구분했다. 길이 비교는같은32개, 개수 비교는같은3000nt의 중첩 부분집합이다. truncation=False이며 실제 token이2048이하임을 assert했다. 각 조건 CUDA peak를 reset하고 cache를 비운 뒤 같은 run의 시작 allocated를 기준으로 추가량을 계산했다.');a();a('시간은 메모리에 준비한 원본 서열의 CPU 토큰화·GPU 전송·forward·pooling까지 포함하며 결과는 GPU tensor다. 반복 구간의 CPU 결과 반환/디스크 저장은 제외한다. 학습은 토큰화·전송·forward·loss·backward·AdamW까지 포함하고 epoch별 validation은 제외했다. CUDA 경계를 synchronize했다. 디스크 입력 준비와 파일 저장은 제외하며 저장은 별도 시간으로 기록했다. compile은 warmup에 섞여 별도값N/A다.');a();a('GPU-Util은 장치 활동시간 비율로 FLOPS/메모리 비율이 아니다.0.5초 대기+조회 구간이 측정 안에 완전히 포함된 표본만 집계하며 디스플레이 영향이 포함될 수 있다. RSS/available/allocated/reserved 시계열은 sampled값이다. 짧은 epoch의 평균은 표본수와 함께 해석한다.');a()
if 'test' in s:
 a('## 파인튜닝·저장·독립 test');a();table(['epoch','train loss','validation Accuracy','F1','MCC','AUROC'],[[r['epoch'],f(r['training']['loss_mean']),*[f(r['validation'][k]) for k in ['accuracy','f1','mcc','auroc']]] for r in s['epochs']]);a('최종 epoch2를 고정 평가했고 test로 epoch/조건을 선택하지 않았다. test400개: '+json.dumps(s['test']));a();a('실제 gradient 관찰: '+json.dumps(s['gradient_observation']));a();a('backbone과head 변경 검증: '+json.dumps(s['parameter_changes']));a();a('모델 저장: `'+s['saved_model']+'`; 저장 '+f(s['save_seconds'])+'초. optimizer/RNG/epoch도 저장했지만 학습 재개는 검증하지 않았다. test logits와 실제 예측 라벨을 보존했다. 독립 프로세스 재로드: '+json.dumps(s.get('reload',{'status':'pending'})));a();a('검증 Accuracy0.8906에 비해 공식 test Accuracy0.6975로 낮아, 검증 성능을 외부 일반화 성능으로 대체해 보고하지 않는다. 첫 독립 재로드에서 TF32 기본 설정 불일치로logits차이0.00575가 나타났으나 분류 라벨은 모두 일치했고, 원 FP32/TF32=False로 맞춰 재검증했다.');a();a('소규모2epoch 실행 검증으로, 전체 benchmark 재현이나 최고의 enhancer 정확도를 주장하지 않는다.');a()
a('복구 이력: 첫 학습 진입에서 rotary cache/inference_mode 충돌이 발생해 validation을 no_grad로 바꾸고 학습을 새 체크포인트에서 복구했다. 첫 복구 시 초기 추론 시계열 파일을 덮어쓴 문제가 있어, 추론6조건을 별도 프로세스로 재계측하고 학습은 재실행하지 않았다. 최종 표는 재계측 추론과 완료된2epoch 학습이다. 자원 CSV는 단계마다 별도 파일을 보존한다. 실패 로그와 복구 전 summary도 남겼다.');a();a('## 해석과 공식 출처');a();a('길이가 커질 때 attention 중간값 등 추가 메모리가 커질 수 있지만, 데이터 개수가 늘어도 batch정책이 같으면 순간 peak가비례 증가하지 않는다. 공유RAM90% 채우기는실험목표가아니다.');a();a('- [공식 NT-v2 모델 카드](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)\n- [공식 소스·학습 예제 링크](https://github.com/instadeepai/nucleotide-transformer)\n- [공식 enhancer 데이터](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)\n- [공식 분류 fine-tuning notebook](https://github.com/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb)');(R/'reports/NTv2.md').write_text('\n'.join(out)+'\n');print(R/'reports/NTv2.md')
````
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/setup.py — 원문 전문 (538 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/setup.py", "sha256": "a366ab833d086bdeffe8b27c8dae210a62f53c2721339fbd02dce08ff0218cdb", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,subprocess,json,importlib.metadata
from pathlib import Path
R=Path('/work');(R/'constraints.txt').write_text('torch=='+importlib.metadata.version('torch')+'\n');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True)
cmd=['/work/.venv/bin/pip','install','--cache-dir','/work/.cache/pip','-c','/work/constraints.txt','transformers==4.44.2','pyarrow','scikit-learn','psutil'];code=subprocess.call(cmd);(R/'results/setup_exit.json').write_text(json.dumps({'exit_code':code}));sys.exit(code)
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

<details>
<summary>scripts/remaining_followups/launch.py — 원문 전문 (7,883 bytes)</summary>


<!-- RESTORE_FILE {"path": "scripts/remaining_followups/launch.py", "sha256": "84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,subprocess,time,datetime,sys,os,shlex,csv,hashlib
import psutil
ROOT=Path('/home/hyein/ai-benchmark');plan=['scvi','dnabert2','ntv2','rnafm','openspliceai'];stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');Q={'status':'running','queue_id':stamp,'order':plan,'models':{},'scope':'inference batch expansion, same existing weights; sequential GPU only','applicability_notes':{'AlphaFold2':'ColabFold query list is processed sequentially; no independent-query GPU batch axis exposed in existing official path. Complex chain count changes biological input, not batch. No artificial concurrent GPU jobs or replacement model. Existing baseline retained; dedicated batch sweep not performed.','Clair3':'Existing official calling pipeline controls batch internally (predictBatchSize and GPU multiplier), no explicit CLI minibatch argument in installed path. Do not confuse candidate count, threads/workers and batch. Dedicated source-level candidate-cache batch sweep not performed; existing measured GPU batch1000 and coverage/region results retained.'}}
qp=ROOT/'FOLLOWUP_QUEUE.json'
def queue_save():
 qp.write_text(json.dumps(Q,indent=2));lines=['# 후속 배치 실험 진행','',Q['scope'],'','|모델|상태|최대 성공batch|보고서|','|---|---|---|---|']
 for model in plan:
  s=Q['models'].get(model,{});lines.append('|'+model+'|'+s.get('status','queued')+'|'+str(s.get('largest_successful_batch',''))+'|['+model+']('+model+'/reports/Batch_memory_followup.md)|')
 lines+=['','## 배치 확대 적용 범위','']+[k+': '+v for k,v in Q['applicability_notes'].items()];(ROOT/'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines)+'\n')
queue_save()
for model in plan:
 R=ROOT/model;run='batch_followup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');records=[]
 if model=='dnabert2':records=list(csv.DictReader(Path('/home/hyein/dnabert2-benchmark/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv').open()));source='existing BEND prepared n1000, fixed4000bp, actual native tokenizer length verified worker'
 elif model=='ntv2':
  original=json.loads((R/'data/inputs.json').read_text())['long_inference'];records=[{**r,'sequence':r['sequence'][(len(r['sequence'])-6000)//2:(len(r['sequence'])-6000)//2+6000]} for r in original];source='existing real GRCh38 regions, central6000nt crops -> native ~1001tokens'
 elif model=='rnafm':
  original=json.loads((R/'data/inputs.json').read_text());records=[{**r,'sequence':r['sequence'][:512]} for split in ['train','validation','test'] for r in original[split] if len(r['sequence'])>=512];source='official tutorial RNAcentral actual unique sequences >=512nt, first512nt explicit crop; pool limit retained'
 elif model=='openspliceai':
  original=json.loads((R/'data/inputs.json').read_text());records=[{**r,'sequence':r['sequence'][2500:17500]} for split in ['train','validation','test'] for r in original[split]];source='actual GRCh38 curated regions; input15000nt -> output5000nt; original five-model MANE10000 ensemble'
 else:source='saved full10176cell SCVI model and its raw-count AnnData, fixed2000genes; latent posterior mean'
 if records:
  seen=set();unique=[]
  for r in records:
   assert r['sequence'] and set(r['sequence'])<=set('ACGTUNRYKMSWBDHVX')
   if r['sequence'] not in seen:unique.append(r);seen.add(r['sequence'])
  records=unique;assert len({r['id'] for r in records})==len(records);(D/'inputs.json').write_text(json.dumps(records,indent=2));pool=len(records)
 else:pool=10176
 S={'status':'running','model':model,'run_id':run,'conditions':[],'failed':[],'input_audit':{'source':source,'unique_available_inputs':pool,'selection':'nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained','token_policy':'fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers'},'largest_successful_batch':None};swap0=psutil.swap_memory().used
 def save():(D/'summary.json').write_text(json.dumps(S,indent=2));Q['models'][model]={'status':S['status'],'run_id':run,'largest_successful_batch':S.get('largest_successful_batch')};queue_save()
 save();print('MODEL_START',model,run,pool,flush=True)
 sizes=[1,2,4,8,16,32,64,96,128,192,256,384,512,768,1024,2048,4096,8192] if model!='scvi' else [128,256,512,1024,2048,4096,8192]
 sizes=sorted(set([b for b in sizes if b<=pool]+[pool]))
 for batch in sizes:
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if active:S['stop_reason']='other_GPU_job_present';S['status']='waiting_external_gpu';save();raise RuntimeError('Do not overlap existing GPU job '+active)
  available=psutil.virtual_memory().available/2**30
  if available<16 or psutil.swap_memory().used>swap0+2*2**30:S['stop_reason']='system_memory_guard';break
  if S['conditions']:
   last=S['conditions'][-1];base=last['model_only_allocated_gib'];estimate=base+(last['memory']['peak_reserved_gib']-base)*batch/last['batch_size']*1.25+3
   if estimate>available-16:S['stop_reason']='predicted_reservation_exceeds_available_budget';S['skipped_next_batch']={'batch':batch,'estimated_reserved_gib':estimate,'available_gib':available};break
  name='followup-'+model+'-'+run.lower()+'-'+str(batch);cmd=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--ipc=host','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','MODEL_KIND='+model,'-e','BATCH='+str(batch),'-e','MPLCONFIGDIR=/work/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/work/.cache/numba','-e','HF_HOME=/work/.cache/huggingface','-e','HF_MODULES_CACHE=/work/.cache/modules','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1']
  python='/work/.venv/bin/python'
  if model=='dnabert2':cmd+=['-v','/home/hyein/dnabert2-benchmark:/workspace:ro'];python='/workspace/miniforge/envs/dnabert2/bin/python'
  cmd+=['nvcr.io/nvidia/pytorch:26.06-py3',python,'-u','/support/remaining_followups/worker.py'];(D/f'batch{batch}_command.json').write_text(json.dumps(cmd,indent=2));start=time.time();reason=None
  with (D/f'batch{batch}.log').open('w') as f:
   p=subprocess.Popen(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
   while p.poll() is None:
    if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',name])],stdout=f,stderr=subprocess.STDOUT);break
    time.sleep(.5)
   code=p.wait()
  (D/f'batch{batch}_exit.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start,'reason':reason}));print('BATCH_EXIT',model,batch,code,flush=True)
  if code:S['failed'].append({'batch':batch,'exit_code':code,'reason':reason or 'worker_error_see_log'});S['stop_reason']='worker_stopped_see_log';break
  worker=json.loads((D/f'batch_{batch}/summary.json').read_text());assert worker['status']=='complete';S['conditions'].append(worker['conditions'][0]);S['largest_successful_batch']=batch;save()
 else:S['stop_reason']='available_unique_input_limit'
 S['status']='complete' if not S['failed'] else 'stopped';S['reached80']=any(c['allocated_fraction_shared_ram']>=.8 for c in S['conditions']);S['reached90']=any(c['allocated_fraction_shared_ram']>=.9 for c in S['conditions']);save();subprocess.run([sys.executable,str(ROOT/'scripts/remaining_followups/report.py'),model,str(D)],check=True);print('MODEL_FINISH',model,S['status'],S.get('stop_reason'),flush=True)
Q['status']='finished_with_unresolved_items' if any(s['status']!='complete' for s in Q['models'].values()) else 'complete_for_supported_sweeps';queue_save();print('QUEUE_FINISHED',Q['status'],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/remaining_followups/report.py — 원문 전문 (5,292 bytes)</summary>


<!-- RESTORE_FILE {"path": "scripts/remaining_followups/report.py", "sha256": "8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,sys,csv
ROOT=Path('/home/hyein/ai-benchmark');model=sys.argv[1];D=Path(sys.argv[2]);S=json.loads((D/'summary.json').read_text());R=ROOT/model;lines=[f'# {model} 실제 입력의 배치별 추론 자원','',f"상태 **{S['status']}** · run `{D.name}` · 종료사유 `{S.get('stop_reason','running')}`",'','## 입력과 고정 조건','',json.dumps(S['input_audit'],ensure_ascii=False,indent=2),'','배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.','','FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.','','## 배치별 측정','','측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.','', '|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|','|---|---|---|---|---|---|---|---|---|---|---|---|---|'];flat=[]
for c in S['conditions']:
 m=c['memory'];b=c['model_only_allocated_gib'];r={'batch':c['batch_size'],'repeats':c['repeats'],'processed_inputs':c['processed_inputs'],'seconds':c['seconds'],'inputs_per_second':c['inputs_per_second'],'loaded_allocated_gib':b,'allocated_peak_gib':m['peak_allocated_gib'],'additional_from_same_run_load_gib':m['peak_allocated_gib']-b,'reserved_peak_gib':m['peak_reserved_gib'],'percent_shared_ram':100*c['allocated_fraction_shared_ram'],'gpu_mean':c['gpu_util_avg'],'gpu_max':c['gpu_util_max'],'rss_sampled_peak_gib':c['rss_sampled_peak_gib'],'available_sampled_min_gib':c['available_sampled_min_gib']};flat.append(r);lines.append('|'+ '|'.join(str(round(v,4)) if isinstance(v,float) else str(v) for v in r.values())+'|')
lines+=['','## 결과 해석과 검증','',f"최대시험성공batch {S.get('largest_successful_batch')}. allocated80%도달 {S.get('reached80')},90%도달 {S.get('reached90')}. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.",'','출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.','','GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.','','원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.']
sources={'scvi':'https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html','dnabert2':'https://github.com/MAGICS-LAB/DNABERT_2','ntv2':'https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species','rnafm':'https://github.com/ml4bio/RNA-FM','openspliceai':'https://github.com/Kuanhao-Chao/OpenSpliceAI'}
lines+=['','[공식 출처]('+sources[model]+') · [기존 모델 보고서 디렉터리](.)']
(R/'reports/Batch_memory_followup.md').write_text('\n'.join(lines)+'\n')
if flat:
 with (R/'reports/batch_memory_followup.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for name in ['STATUS.md','RESULTS.md','MEMORY_RESULTS.md']:
 with (ROOT/name).open('a') as f:f.write('\n'+model+' 배치후속 '+S['status']+', 최대성공batch '+str(S.get('largest_successful_batch'))+'. [보고서]('+model+'/reports/Batch_memory_followup.md). 종료사유 '+S.get('stop_reason','running')+'.\n')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/remaining_followups/worker.py — 원문 전문 (6,920 bytes)</summary>


<!-- RESTORE_FILE {"path": "scripts/remaining_followups/worker.py", "sha256": "e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,time,json,gc,argparse
import numpy as np,torch
sys.path.insert(0,'/support');from gpu_support import Meter,write_json
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];kind=os.environ['MODEL_KIND'];B=int(os.environ['BATCH']);C=D/f'batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'batch':B,'task':'inference resource batch sweep only, no training','conditions':[]}
def save():write_json(C/'summary.json',S)
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='load';t=time.time();tokenizer=None
 if kind=='scvi':
  import scvi
  P=R/'models/scvi_20261027_185327_767988/train_10176';m=scvi.model.SCVI.load(str(P),accelerator='cpu');m.to_device('cuda:0');m.module.eval();nets=[m.module];ad=m.adata;assert ad.n_vars==2000 and ad.n_obs>=B;S['checkpoint']=str(P);S['input']={'unique_cells':B,'genes':ad.n_vars,'raw_counts_source':'saved adata.h5ad layers counts, actual 10x PBMC4k+8k','original_pool_cells':ad.n_obs,'batch_covariate':'sample_id, biological batch distinct from GPU batch'}
  x=ad[:B].copy()
  def forward():return m.get_latent_representation(adata=x,batch_size=B,give_mean=True)
 else:
  records=json.loads((D/'inputs.json').read_text());assert len(records)>=B
  if kind=='dnabert2':
   from transformers import AutoModel,AutoTokenizer,BertConfig
   P=Path('/workspace/dnabert2_eval/model');tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);m=AutoModel.from_pretrained(P,config=BertConfig.from_pretrained(P),trust_remote_code=True,local_files_only=True).float().cuda().eval();nets=[m]
  elif kind=='ntv2':
   from transformers import AutoModelForMaskedLM,AutoTokenizer
   P=R/'models/checkpoint';tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);m=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True).float().cuda().eval();nets=[m]
  elif kind=='rnafm':
   import fm
   torch.serialization.add_safe_globals([argparse.Namespace]);m,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');m=m.float().cuda().eval();nets=[m];converter=alphabet.get_batch_converter()
  elif kind=='openspliceai':
   from openspliceai.train_base.openspliceai import SpliceAI
   W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);nets=[]
   for seed in range(10,15):
    net=SpliceAI(32,W,AR).float().cuda();net.load_state_dict(torch.load(R/f'source/models/openspliceai-mane/10000nt/model_10000nt_rs{seed}.pt',map_location='cpu',weights_only=True));net.eval();nets.append(net)
  else:raise ValueError(kind)
 torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for net in nets for p in net.parameters());S['parameters_trained_in_this_run']=0;S['theoretical_weight_storage_gib']=sum(p.numel()*p.element_size() for net in nets for p in net.parameters())/2**30;S['parameters_used_backbone']=sum(p.numel() for p in m.esm.parameters()) if kind=='ntv2' else S['parameters_total'];S['checkpoint_reference']={'scvi':'models/scvi_20261027_185327_767988/train_10176','dnabert2':'/workspace/dnabert2_eval/model + model_revision.txt','ntv2':'models/checkpoint; existing setup/download provenance','rnafm':'models/RNA-FM_pretrained.pth; existing data/sources.json','openspliceai':'source/models/openspliceai-mane/10000nt seeds10..14'}[kind];S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'precision':'FP32','tf32':False};save()
 if kind!='scvi':
  cache=D/'token_cache.pt'
  if not cache.exists():
   if tokenizer is not None:
    encoded=tokenizer([r['sequence'] for r in records],padding=True,truncation=False,return_tensors='pt');encoded={k:encoded[k] for k in ['input_ids','attention_mask']}
   elif kind=='rnafm':
    _,_,tok=converter([(r['id'],r['sequence']) for r in records]);encoded={'input_ids':tok}
   else:
    tab=np.zeros((256,4),np.uint8)
    for j,c in enumerate('ACGT'):tab[ord(c),j]=1
    encoded={'onehot':torch.from_numpy(np.stack([tab[np.frombuffer(r['sequence'].encode(),np.uint8)].T for r in records]))}
   torch.save(encoded,cache)
  encoded=torch.load(cache,map_location='cpu',weights_only=True);cpu={k:v[:B] for k,v in encoded.items()};S['input']={'unique_sequences':B,'pool_sequences':len(records),'raw_length_min':min(len(r['sequence']) for r in records[:B]),'raw_length_max':max(len(r['sequence']) for r in records[:B]),'input_shapes':{k:list(v.shape) for k,v in cpu.items()},'tokenizer_truncation':False,'padding':'same pool-wide longest token width fixed across batches; explicit crop lengths in inputs provenance','ids':[r['id'] for r in records[:B]]}
  if kind=='ntv2':assert cpu['input_ids'].shape[1]<=2048
  if kind=='rnafm':assert cpu['input_ids'].shape[1]<=1024
  def forward():
   x={k:(v.cuda().float() if k=='onehot' else v.cuda()) for k,v in cpu.items()}
   if kind in ['dnabert2','ntv2']:
    h=m(**x)[0] if kind=='dnabert2' else m.esm(**x,return_dict=True).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1).to(h.dtype);out=(h*mask).sum(1)/mask.sum(1)
   elif kind=='rnafm':
    h=m(x['input_ids'],repr_layers=[12],return_contacts=False)['representations'][12];out=torch.stack([h[i,1:len(r['sequence'])+1].mean(0) for i,r in enumerate(records[:B])])
   else:
    out=None
    for net in nets:
     z=net(x['onehot']);out=z if out is None else out+z
    out=out/len(nets)
   return out.cpu().numpy()
 meter.phase='warmup';t=time.time()
 with torch.no_grad():out=forward()
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;assert out.shape[0]==B and np.isfinite(out).all();del out;gc.collect();t,base=meter.begin('inference');steps=0
 with torch.no_grad():
  while steps<2 or time.time()-t<15:out=forward();assert np.isfinite(out).all();torch.cuda.synchronize();steps+=1
 r=meter.end(t,base,batch_size=B,unique_inputs=B,processed_inputs=B*steps,repeats=steps);r['inputs_per_second']=B*steps/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];r['model_only_allocated_gib']=S['model_only_postload']['allocated_gib'];S['conditions']=[r];np.save(C/'outputs.npy',out);assert np.array_equal(np.load(C/'outputs.npy'),out);S['output_shape']=list(out.shape);S['finite_and_exact_save_reload']=True
 first=D/'reference_first.npy'
 if not first.exists():np.save(first,out[0]);S['reference_first_created']=True
 else:
  ref=np.load(first);diff=float(np.abs(ref-out[0]).max());S['batch_invariance_max_abs']=diff;assert np.allclose(ref,out[0],atol=1e-4,rtol=1e-4),(kind,diff)
 S['status']='complete';save();print('BATCH_COMPLETE',kind,B,r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->


</details>

### record-5c86518a2a1e

원래 파일: `ntv2/data/input_audit.json` · 원문 SHA256: `f604a2c6ed67c8c07f92702f0f98736b4165c7b83819dfb4ce7e8606ada2f37d`

<details>
<summary>ntv2/data/input_audit.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/data/input_audit.json", "sha256": "f604a2c6ed67c8c07f92702f0f98736b4165c7b83819dfb4ce7e8606ada2f37d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "splits": {
    "train": {
      "n": 512,
      "length_min": 199,
      "length_max": 200,
      "labels": {
        "1": 240,
        "0": 272
      },
      "first_example": {
        "id": "10230",
        "sequence": "GGGCGGTGTAATGCATGGAGAGAAGCCTCTCCATAAGGAGGCAGTGGCTTACTGGCAGAGAAAGTGAGCGGTGCCTCAAGTTATGTTAGTGCGAGTCTGCTGGAAGAGAGCCTGGCGGTAAGGCTCGCTGGGAGCTCGTAATCAATTCACAGTTTAGAGGAAAGATGCTTGACTCAAACTGCCTATCTTGTTGGGGCAA",
        "label": 1,
        "sequence_sha256": "08febc7b70c5adb450c025dd67539d67ccf5363854881941a0b256c736ca368e"
      }
    },
    "validation": {
      "n": 128,
      "length_min": 199,
      "length_max": 200,
      "labels": {
        "1": 60,
        "0": 68
      },
      "first_example": {
        "id": "9052",
        "sequence": "TCCCAGCTCGGCCTCCGTGAAGACAGAGTGCTTATTAAAAGAGTTTGTGACCACACGTCACCTTATGTCAGCCTTTGCAGAGAGTAAAGCTGCCAGTCTGGCAGTGAGGCACATAAAAGCCACTGTAATCAGCAACCAAAATGTTCATTTTCAGGAGAGGGAAGAGGCATATTCTACAAAATGTGGCAATTCTGAAGAT",
        "label": 1,
        "sequence_sha256": "e29627744fb1217c0acbbdf81ba53c3cec7864c5719131d8e79d9e7d30ee5c23"
      }
    },
    "test": {
      "n": 400,
      "length_min": 200,
      "length_max": 200,
      "labels": {
        "0": 200,
        "1": 200
      },
      "first_example": {
        "id": "0",
        "sequence": "AATTTTCTCATTTTCTCATAAAGTTTAACAGTTGTTTATTTGAGTCAGAATTCAAATAAGCTTCTGTACATTACAATTGGTTTTAAGTTCTTATAAGACTCTATAGGTTTTCCCTTCATAATTTTTCTTGCAATTTATTTGTTAAAGAAATTGGGTCATTTGTCCTATTGAGTGCTCCACTGTCTGTTTTTATTATTGTA",
        "label": 0,
        "sequence_sha256": "4e85654d7875bb3dc735d3e7fb45d1eb1e2a54db0a2314a356730c2c95436154"
      }
    },
    "long_inference": {
      "n": 128,
      "length_min": 12282,
      "length_max": 12282,
      "labels": {
        "unlabeled": 128
      },
      "first_example": {
        "id": "chr3:152302071-152302583:+",
        "sequence": "GCTAGCCAGGGCAGGGTCCTCTTCCTCCTGCCATGTCACCGTCTGCATGTTGGAGTTTCTTCAGCACATTTGGAAGCTAAGAGAAGGAACTGCTGAAAGACAGGGCGTGCATGTATAGAGGAGGCAGAAAAAGAACTAGATCTTGAGAATGACTGAGTCAAGGTGGAGGCTTGTGGAAGCTTAACTGTGGGCAGGAGGAGGGAGAACTGGGAGACCAAGAAAGATACAGAAAGATAGGTTAAAGACTTGTGAGTAGGAAGGTAAGAAAAAGTTCATGCTTAAGGGCTACTGTTATGGCTATGAAATATTCGTGACGCATAATATTATTATCCACTATTGCTAATGAACATGGAGAAAGAGTAATGAAGTTTTAGAAAGGCTGCCGTGTGCAATGAGATAGGGAGCTGTCCAAGGACAGGGCTAGATTGTGAGGCAGTGAAAGGGCCTATGTGACCCTGGAACATAATAATTTATAGAGGCAATGATTGTGTGTGTGTGTGTGTGTGTGTGTGTGTTTTCAAATGTATGTTTCAGTGTGTTCATGTGTTGTATTCATATATGTGTTCATGTGTGTGTGTTTGTATATTTTGTACCCTAGAAGTAATCCAAATCCTGATGCAGCAGTTGATAATTTAGTTTACCAATATTTTTGGAACTCCATCTATTTTGCATCATTATGGCTAAGCCATGGGGTTACTGAGATGAATAATATATTTTTTCTGTCCTTAAAAACATATGGTCTAGTGGGTCAGATATATAAGGAAATAAAACACTGTATAGGGCATAGGTGCAGAGATAATTCAGAGAAGGCAGTATTACGATATGGATGGATAGAGATGGGGATAATGACATCTTGGTAGCTTTTTGGAGGAAGTAACATCTTAGCTGGCTTTAGAGGAAGAAAATAGTTACATGTGAGGCACTTGAAGATTCTGAAGAGAAGGAGTAAAGACATTGTTGACCATGCAGGTGGTCCATTCTCAAGGCATGAAATCATTGAGTAATTCTTAATGCAGAACAAAGTTTCAGGAAGATAGACACAGCTTACAGTTTATGACTGGACCTTTGTTTTTTTTTTTTTTTTTTTTATAGAAACTGGCATTAAGACCTAGGAACTTTTTTGAAAACCAAAATATTTCTTGAACACTGGGGATGAGGTTTTTTTTTTTTTTTTTTTGATGGAGTCTCACACTGTCTTCCGGCCTGGAGTGCAGTGGCGCAATCTCGGCTCACTGCAACATTCACCTCCCAGGTTCTAAGTGATTCCAGGAGAATCACCCAGCCTCCCAAGGAGCTGGGATTACAGGTGCCTGCCACCATGCCCGGCTAATTTTTTGTATTTCAGTATTTTTAGAGACAGGGTTTTGCCATGTTGACCAGGTTGGTCTCGAACACCTGACCTCGTGATCTGCCTGCCTCGGCCTCCCAAAGTGCTGGGATTACAGGTATGAGCCACTGTGCCCAGCCATGGGTTTTTTAAATTTTTTTTAGACAGAGTCTTGCTCTGTCACCCAAGCTGGAGTGCAGTGGTGCAATCTCAGCTCACTGCAGCTTCCGCCTCCCGGGTTCAAGTGATTCTTCTGCCTCAGCCTCCTGAGTAGCTGGGATTATAGGTGCCACCACCACGCCCAGCTAATTTTTGTATTTTTAATAGATATGGGGTTTTACCATGTTGGCCAGGCTGGTCTTGAACTCCTGACCTTGTGATCCACCCACCTCAGCCTCCCAAAGTGCGGGGATTACAGGCGTGATCCACCAAGCTTGATTGATTTTTTAAATCTCTATCAAAAGTTGTAGTCGTACTGTAGTATTTTCCAAATAGTTTTGGAAAGTAGTTTTTTAAACGAAATTTAGTGTTAAAATGGCAATATTAATCTTAAACAATTTAAGTTCTAGATGTTCGTGTTTTTTAATGTACGTATGTATTTATATCCTATACTCTAGTTTCCAAAAATTATAAATATACCATATGCCAATAGAGCAATTTTGGCTCATGAAGAAGTATCATCAAATACTAGTTATCATACTGTGTCATTGCCTCTTTAAAATGTGAAAATTTTACTTTTTAAAGCCTTGTTTTCCTCATTGGCAACATTCAAACAAAAAATAAAGAAGAAAAATCCTGATAGAAAAAGTAAGTTGTTAAAAGTAAAAATGCCCTGCTCTAAGAAGTAGTCTTGTGGTGCTGAGTTAACTGATCATCAGGCAACAGGGGATCCCTCATCCCCCACCCTCCAGCTTTGTTTGGAATTGTACTTCTGAAGTGCATCACGCTGCTCAAATCAAATTGCTCACTTAATATTTAATGTTTCTGCGATGGAATCACTTTCTCCCTTGAGGTTGACTTCAGATTTACCAGACACACTGACAACACTGTACTCCTTATTTTCCCCACATGACCTCAGCTTCTGCTTTAAAAATGTACGAATAAAGGATGCTTTTGTTCATATCTTGCCTGCCTGTGATATATTGACAAAGTGTCTGCGGGCTCAATGGAATAGTTTTACAGTATGAGTGCTGTTTTCAGTGGAGGGGCAGGACTTGGGCTTTAGAAGGAAGTGAATTGGATAGGTCAGGAGAATGTACCATTTGTAAACACCCCTTTCCTTTTTTATTCACGTGTCAGGACAGCAGGAGCAGGCTGTTCAGTGCCCGTCCCTGCGCAGAGCCGGAGGGAAAGCAGCTTGGAAGTCAGCTGCAATTTTTAAACCTTGCTTTTTATTCCCTGGTAATGATCTTCAAGTGCTTAGACTTGTCTGAGAAGCTGTTTTGAAACTAACATGGTTTAGTCCACTAACTGCATGTTGGGTAAATTCAAAACCCACATGCTCGTCCTCTTGCAGTGGAATAAGTCACATCTGATGGACATTTTCTGTGCTTATAGCATAGTAATGAACGTCTGACAGGCGCGACCTTTCATAAGACAACCCACACTATTGGCTTTCTGCCCAGAATATTGCTGCAACACTATCTGTGATTGGTTATCTCTCTATCATGCATTGCTTCACAAGGGGAAACGGCCCCTTTTTTTAATAAATCTACGATGGCTGGCTGCAATATGCCTCACTGTAAGTATTTTGTGTATGTATGTGTGGGAGAGAGAGAGAGAGGCTGACTTGAATATTACTTAGTCTGGTATCACATGGCAAGGTTGTGATACTGTAGGACATCAGTGAAGTGCTACTTAGTAAATCTTAACCTATCTCTTTTTTCTACAGGTTGGTACTAAGAAGTGCCTTTCCTGACGTCTCTGCTGCTTGGAACCGCTTCTAGAGCAGTCTCTGCTTTTGCCTTGCTTGCTGCCAGCTAGACTGTGACGACAGCACATCCACCCTCCACCTCTAGCCCAGACACCCCCATTTCTACTTATAATCAAGAGAAAAGCTCTAAGTATCTGGCATTGCCCTAGGCTGCTTTAGTGTTAAAAGAAAAGTTTGCTGAAAAAGTAAGATATCTTCTGCCAGGAAATCAAGGAGGAAAAAAAAAATCATTTTCTCGATTTTGCTCTAAACTGCTGCATCTGTCTATGCCAAACTAATCAATACCGATTGCACCACCAAACTCCATTGCAAATTCAGCTGTGAGGAGATTCCCTTTCAGACAACTTTGCTGAAAGCAGCTTGGAAATTCGGTGTCGAAGGGTCTGCCACGTTTTCATGCTTGCATTTTGGGCTCCAAATTGGCACTGGGAAGGGGTTACTGAGAGCACAAGGCTGATACCAGGCCCTACTTTTAAACGTTCATCTACTTACAATCCTAGTATTTCTCTAAAAACCAAAACCTCTTTGAATTAACAGTTTCATGCTGTGAATTTCTAGTGGGAGATCTTTTCCTTGATATTGACGACACAATTTTCCATGTACTTTTAAAGCAGGGAGTGGGGAAAAGTATTTTGAGGGGACATTTTCATCATCAGTTCAGCTTTTTTTTTTTGGTTGTTGCTCTTTTTTGGGGGGGTTGGGTTTGTTGGTTTCACTGAAACATTTAACTACCTGTAAAATCTAAACATGGCTGTTAGTGTCACACCAATTCGGGACACAAAATGGCTAACACTGGAAGTATGTAGAGAGTTCCAGAGGGGGACTTGCTCACGGCCAGACACGGAATGTAAATTTGCACATCCTTCGAAAAGCTGCCAAGTTGAAAATGGACGAGTAATCGCCTGCTTTGATTCATTGAAAGTGAGTAACTATTATATTCTTTTAAGGATATTCAATGATTGAATGAGGGGTATATGGACATACAGTTCTCTGCATATGGGATTATGGATTGCTTTGTTCATAGTTTAGTTATACAGTGTGTTGATGATGTTGGGCTGGGGAACTTTACGAGAAGCAGTAGAATCAAGTCTGGCCAAACTTTGTTATGAATAACTTTATCAGTCCTTTCACTGCATGTTTTTTTTTGCTAGGGCAGAAAGTGTTGACATATTTTTCACTTAGTAATGGCAAAGTGTCTGTCTGTCCCTGGAAATGGTAAATGTAGCTACCTGTGAGTGTTTAGCATATGCATATATTATATATAAGGCCATGGCAATAGCTGAATGAGGATGATTTTTATTTGTTGGTAAAGGCATTGTACAGTTGCAAAAGAGAAGCAGATTTAAAATCAGCTAAACTTGGATAGACTTTTTACTCTTTTAATATTTTGACATTTTAGATTTTCTTTTTATTTTGTGGGCATTTTATTTTCAGGCCAGAGCCACGCATTTTATGGAAAATGTAACTAGAAGTAAGGTGCTGAACTGCTGCTACAGCACCTTACCTGAGGCACAAACTTTGTTGCATGTGAGTTATATAAATAAGTAGAAACTCATAAGTACTATTGTTTATTGGATTTTTCCAGCTTCAAAAATTCTAATGTGAATTAGCAGTTTATTTTCTGTTAGATCATGATTTTTTTTTATTTGTAATAGTCTTAGCTAGAAATGTTAAAAGTTAGCTTTTGTGGGCACCATAATACTAAAATATTTACTTTAGTGTTTCTTGACATTCAAAAATGTCATTGGATTTGTAATTCAGGTATCATATTTGAAAATGAGTCTTTTAAAAGATAACATAAATATCTTTATTTTGACACACAAGGTCAAGACTAGAAATGTGTTCCTGGGTACTTTCAGCCTACTTGGTTTAATCAAATTGCTTTTGAATATGAATGTCCTAATTTAATTCTTTGGACCTTTGAGGGGAGGACACTATCACTTCTACATATGTAGAGAAGTAAAAGTCTCATAGATCCATCTTGTCTTTAAAAATATATGGTGCTATAATAATCTTTTCTTTAATAACATCAGGGTGAGTTATGAGTTAATTGTTTCAGTTCATTGTTTCCTGGTCCATTTTAGGCTGTTCATTTTGCTAAAGTTTTTGCCAAAGATTTCGAGAATATGATCTTTAGAGTTGAACAGAAAGGAGGTATTAACAACAAAAATATTTTAAAATGTTAGATTCTTCTTAATGTCATAGTCCTAAAGATCAATAAGCAAAAGTTATAAACTCCCATTGAGACTTTCTTCACCCTTTCCTTTGAGAGGCATCAGTGAAGGGCACAGTGGTTGTACCTCCGTGAATGTGTATTCCTAAATTCATGCTGGCTGAATTCAGGTGTAATTAGAGACTGAGACCCATAAAATGAGGGGCAAGGAGCTAATTACTCTTGTCGGTAACCAGCTGTGGCCATAAGTAACCCTGTTTTGACCCAATTAATAGTAAATCCAATTTTTAATCCTACAGACAGTTGATGGTGCACTCGTAGCTGCAAAATGAAGGACATTAAATTTGGCTGTGACATTGAGTGGTTACACAGAAGACCTGCTTCTACAACATTTGGAAACATTGTGAAACATTACTATTCAGTTGCTATTTTTCCATCTCAAAATATTTTTTTTTAGTTGCTGCTCACTTATGGCCACAGAACAGTCCCTTTTTGAGTAAAGCAGCAAGGTTGCTTAATGACATGCTATGTGGTTAACAAGATTCAACTGCCACATCAAGGACTTTAAATAATTTCAAGATTTTAAATGTATTTATGAAATGAACTCATTACAGCAGCTAATCAATGCAAACTTTGTCCTTTTTCTTTGTCTTTCTAAGAGAAACTGAATTTAATTTAAATTTAAATTACAGTGTCTAAAACTTGGTAAAATATTTTAAATTTGAATAAGAGGAAAAAAAAAACATGAATTTATTTATGATGGAGCTTCATACCCAGGACAGTAGCTTCTTAAATGTTAGTAAAATAATTATGTTAAGCATTAAAACAATTTTGAATATGTAATTAAGCATTTATGTTAAAAAACATGTTTCTTACGTGTGTGTGTATATATTTATAACTGACACATGTGATTCTTTGTACAGTATTTAGAATCCTTCCCTTCAGATAGCAGTCCTCTTGTATTCAGTCTTGTGGCTAAAATAGATCAAAAACATATTTTAAAACTGTTAAAGTAACTGTATGACTCATGCTTGAAGTTCTGAAATTTGGGAGCGTTATGAGGTTTCCATGCCATTAACCGCTAGATCATTTGCATTCTCTCACAGAGTAAACGGAAAAAAATTTAGAAAACTTGTAATAATCCTGACTTCTTTAATAAAATGTGTCCTGTGTATTGCTTAAATCCATAGTCCAGCCTTTTCTGAGTTCTTGAGCCATGAGATGGAGGTGGCCTGTGTAATCTACTATTTTTAGAAAGTTAGGAAAGCAAACCAAATACTCTCTGAGAGGACCCTATGAAATGGCTGTACCCTTCAGCTGTGCTGCTTCAGACTTTGGCCAGGCTCCCAGGTGACTTACGTTAATAGCAGACTTCTAGTTAGAAGAGGAGAATTTATTTCATTAAAAAAAAAAAAAATAAGTCAAACAGTTCAAGAAAGTATTCATTTATGTAAAAGAAGTCAATTGGTATGTTACTAAAGATAATAGGAAGAAAATTAGCTCACTATTTTAGTTGCATACTGATGATTTTAAGGTTTAAATAATTAGATTTAAATATACCGTATGTAAATGTAGAAGTCTGTATCGTAATAAATCTGTTGCTAAGTAAAATGGGAATACATGACAACGAGGAATGTAGGTATTAAGAATGTCAGAATTAGAACGAGAACAGTAAAAGAGCAAAAGCTGTAATAATGTGTGCTAGTATATAAACAATACTCTAAATACTTAGTTTCTCTAATTTTCCTTTTACAATTGCAGTTTTAATTTATAATAGTCTAAGAGCGTATTCAAGCTGAGACTTAAATTGCTTACGTGCTATAGATTTAGGTGTATTAGCTATTTCAGAAATCCAGAACATTTGCAAAGGTGCATTATTTTATGCCAAAGAGGATTATGTTCCCCTATGGCTTGTGAAAGAGATAGGAAAAAAGAAAAGAAAAATATTAACACTGCAAGCCACCATTAGAAAATTATATTTCAAAAATGATGTTTCAAATCAATGACAAATTGCCTATTCAGTCATGAGCTCTCAACTAAAATTTGGGGAGTCAATGTAAAGAAAGCCTACAGAATGTATTTTAATATTTGCCAAGTTATTTATACCCTCAAAAAAAAAGCTTTTACGATATGATCAACAGAAATAGACTTATTGTTTACTGAATTCTATTTCAAATACCCATTCATTTTTCAGGTACTAGGGATGGGAGAAAGAGAATATAACCTTTATATTTACTATCTATTTTCCAAAAATCTAGCAAAATTCTTTTACCTTTGAAAACTTTAGTCACTTGTTCCTTTATCAAATTGAGCCATTTTATTTAAGATAATTAAAATTTCTTTTTGCTATATACAGTGATATCTATCTATAAATAGCACCCCGATCAGATTTATCATAGACACATTTGAGATCAGATTCCACTGCAAAGTGGCATACAATCAAGGGTGCTGAGAACAGTTTTAAAGAGCCTCTGCAGGCTTAGCAGAATATATGAAGTATGGAAATTAGGTATTAGAAGGCACACAAATATATGTACATACAGGTAAATGAACAACTAAAGCACTCAAAATCAGTAAGGCAAACAAGTATCCAGATGGTCACATATGGATTGGATGGCATAACTAGACATGATAAAGTAGTCCCAAAGGTGGTTATAGTTAAATTTTAGAAGATCCTCTGTTGCTAGCATGGTTTATCCTCAATCCAGAAATTGCCTGTGTTGCTGCTGTTCTAAGCCCCAAGCATAGACTTTTCATCATATTTTAAAAACCAGTTTCACTGATAATCCTGATAAAATATCTGCAAGTTTTAGACTAAATATGCAATTTTCAATTAAAATTTTGATCTATTAGGGTCCTAACTGGAATCACTGTTACCAGGAAGTTTTTTTGTGGTTTGATTTTTAACCACTGGTATCTCCCATGTATTAAAACTTTGACAGATTATTATATATACTGTGCTGGCACATATATTCAATGTGTCTTTCAGTGACAGATTGACATATCACTTCCTTGTTTTCTCCTAATAACTTAGGAGATGAACTAACTATAAGGCATAACCTGCGCAGCAGTATGTTTCATATAAAAGGTCCCTAAGGAGAAACTTCACAACTTTGTAAAGAATATCCTGTGAATATGCTGCAGTGTCATCAACTCTGACATAAACCTTTCTTTAATGAAATTCTCAATAGCTGCAGTTGCTTTGGTGAAACTAGTAATTGTATTCATTTGCCTGTTTATATTAGAACTTTTTTAGAAAACTTTTTTTTATGAGTAAAGGTTTTTTTAGACTATTAGATAAATTGTAAACCATGTTTCGAATCAAGCTACCTGCTGCAGTAATTCTAGAACTATTTATTTTGCTTCATATTATAAGATTTTGGTACACTGCGTCTACTTTTTTTCTCATGAAATCTTGCTTCACTGTGTGAGTGTGACAAACTAAAAACAATTTAGACCTCAAGCAATGTGGCACTTACAGCTGAATTCAGTTTCATTGTAATTCAGATGTCTGAAGCTAATCAAAATTCCACGGAGTACCCTTGAAACAATTGCATATATAAGGCTGTTTCTCACTTAAAAGAGACAATGCAGTGTGGGGTGAATTGTCTTTTGAAGAACTGTATGACACATGCCCAGTTCAGGTGAAAGGTGAGAAATGGAATGTTTCTCAGACCTAGCACAGACACCTAGGGACCTAGTTTGGGAAGCTTATCTTGTCATTTTCCTTGCCAAGGATAAGTAGAGGATTTCAGCTATCAACATTGCTAATCCGTTACCTTTCATGCTGAGCAATGTCTCTGCCTTCCCACAGACTTAGACAAAGCTTCTTATTTTCTTACAAAGAGGTCCTATTGAGGAAATTTGAATTTTTTTTTTTTTTTTAAACTTAGCAGTACTACATCCTGTGCTTTTAGATAGGTGAGCAATTTTCCATGTTAACAGCTAGGCCAAAAATCATGACTGCCAGTAGAAGTGGGTAACTAGATAGATAGATAGATGGCCTTTGATTTAGCACTGATGATAGTTATTAGCATGTAGCAACTTAAGTAAAAGCAACTGGAAAGGCTTGATAGAAATTCTGAGTTGTTTGAAGTATAATAGATTATTAGGATAGTAATTTCTTTGTGAATTTTACTGTAAACAAGCTTGATTCTAGTGAGGACTTATGCTAATTGCTCACTTATACTAACAGCAACTACAGAGTTTTTATTTAATACTGTTTCTTATACAAATATCATCAAATGCACATCTACCTACAGCATCATCCCTACATTTTTGGCACATGTTTGAAATATTTTCACTTTGAGATAAAGTTTGATTTTTTTGCTAGCGATAAAACTTTCTATGCTATGAGCTTGTGTTTTGAAAATTACTGATAATTACTAAACAGTATAACCCCTGATATCTTCTAATTTGGTTGACAGGCTGTTTGTCCTCTGGATTCCTGAGGGCCTTGGCCATCTCTGACCATTACATTTCTTCTTTTGTTATCTCTGAGTTCTTTCCAGCTTTCTAATTGTCCATCTTTAGAACTATTGGTTCCGATCTTAAATATGTCTTTCCATAGTCACACAGAAGTGATTTAGACATTTTGTCTGTGAAGAATTAGATTCAGCTTTGCATAACAGAGACACCTCCCCTACCCTCCCCAGCATTTGGTATAAACAAATTAGGGGTTTATATTTCAGTCGTCTAGGTCTATTTGGCGACTCTATAGTCATTGGAACTCAGGTTCCTTCTCTCTTAGTTCCTAAGGAGCACATGGCTTCCAACTTTAGGATTATTTCAAAATTGCAATGTGGCCACTGCAGTGCCAGCCATCATATCCTTAGTCTAGCAGGACGCAGAAAAGGCAAGACACCATAACTCAGCCCAGCCAGGCTCCTTTTAAGCAAAGGTTCTCTGAGCTTCAGCCAGTAATTTTTCAGTCTCTTATTGACTGATTTTGGTAGCAGGGATTGCTAGGAAGTATTTTTTTGTATGTTTGTTTGTTTTGTTTTGTTTTTATTTATTATTATTTTAAAAAACTGTTGCCTTGATTGAAATCAGGAGTCTGGTAGGTAAGACAGAAGGGGAGAATGGAAATGGATGTTCAGGAGGGCCATTTCTGGAATAAAAGAGTGTTTATTTTTCCTCTCATAAAACTTTCAGAATTAAGTAAATCTCTCTGGAAAGCATCTGGGTAAGGGAGGTGGTTGAAGAGATACCACTTTTGAAGAGAAACTGAAGTCTTCTCATCTTGACAAACTTGTCATCACTTCTGTCTTGCTTATTTAGTAAATAAAACCTGGCATATCACAAATGGATTTATTTTTCCTTATAGAGAAAACTAAAACAGCTTCATTTTTTATCTTCTTATAACTGTACTTATTTATTTATTTATTTGAGACAGAGTCTTACTCTGTTGTCCAGGCTTGAGTGCAGTGGTGTGATCTCGGCTCATTGTAACCTTCGCCTCCCACGTTCAAGCAATTCTCCTGCCTCAACCTCCCGAGTAGCTGGTCCTACAGGCGTGTATCACCATGCCCAGCTAATTTTTGCATTTTTAGTAGAGACAGGGTTTCACCGTGTTGGCCAGGCTGGCTTGAACTCTTGACCTCAGGTGATCAGCCCGCCTTGGCCTTCCAAAGTGCTGGGATTACAGGCGTGAGTCACCGCGCCCAGCCTACTATTCATATTTATATAAACAGTCACATTTTTATAACTGGAATATAGCAGAAATGCCATGTCATTAAAAACAGGTCCCAAGTGATTTTATATGACAAAAGAATCTGCACTTTGATTTGTATTTCTGCTGCTTCCACAAATACACTTTCTTATTTTTTCCGTATGGAGTAAAATAACAAGGTGAGTTAATATTTATTGAAACTTAATGTATGTCAATCAGTGTTTTCAGTGCTTTTCATTTTTTTCATAGTATTTTGTCCCTATAAAATCTCATCGAGATAGTTACTGGTGTTACTTCCATTTCACAGATGGACACACTGAAATACAGAGAAGTTAGGTAATACATGCAAGGTCAGTCAGCCAGTAAAATGTGGACTTGGAGCCTCGTAATCTGATTTCACAGCCTGTACTCTTAATTGTTTTGTTATGTTTGCTGGAAAACATTGCTTCTTAATTTCCATATGCATATTCTAAGGATTCGACAAAGAAAATATAACTAACAGCAGTACTTAAGAATAAAAATGAATATCCAAGACAAATTTATAGTACAGGGCCTAAAACAGTAAGTACTCAGAAGTAATCACTGTATTTTAAGTCAACAAATGAGTCAAGGAGTGAACCCAAAATAGCAGAAAGAATGCTGCCACCAACTTTCAAAGATGTTGTATCTGTTGTAAAATATATGGCCATATAGGAAAAATTTTAAATGAATCATCTTAGATAACTAGTCACTGTATAAACTATATAGCTGCCTCCCCAGTGCATAAAAAGCCATATTCAAAAGGTTTAATTGTTTAGAAAATGTTTTTCCACAGAAGCAAGGTGATTAAGTCATCGGGTCAACCCTGAGAAGCCAAAGTAGCCCAGAGGATGCTTGAGTTATAGTTCAGTACAAACAGTACCTTGGAAACCCTGGTTATTAATTAAACCCAAGAGGCTCCATAGAAGTGTTATCTTATCATAAATCTTGAGAACTCCTAGAGTCACAGACCCCAGCTGCTTTTGTCAAACTATTTCTGTGGCAAAGATGTATTTTAAATTTCAGTTTAGGGTTCCAGAAATTCAGCTTGCCTAGTGCAGGGGAGGTGGCCTTTCCCAAAGTCCTGGAACTCATGCTGTGAGCCAGTGTAGTTTTCCCCAGCTTTAACTTTGGGCTGCTTCTGTTTCCTGAAGCTAGTC",
        "chromosome": "chr3",
        "center": 152302327,
        "strand": "+"
      }
    }
  },
  "official_train_rows": 14968,
  "official_test_rows": 400,
  "train_after_test_overlap_and_duplicates_removed": 13902,
  "policy": "official test untouched for selection; train/val stratified seed42; exact sequence hash disjoint; long unlabeled reference regions inference only"
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-4bdfabf0fbea

원래 파일: `ntv2/results/batch_followup_20261028_143713_869625/batch128_command.json` · 원문 SHA256: `64195cfd88aef528aef25c321a553dffd729015c3007f3a29f727f9c6678cb8a`

<details>
<summary>ntv2/results/batch_followup_20261028_143713_869625/batch128_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch128_command.json", "sha256": "64195cfd88aef528aef25c321a553dffd729015c3007f3a29f727f9c6678cb8a", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-128",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=128",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-70fae65b73ec

원래 파일: `ntv2/results/batch_followup_20261028_143713_869625/batch16_command.json` · 원문 SHA256: `926d30380833c37c983526f916308871f1f1cd9e77193a658bbcd634fc50ab71`

<details>
<summary>ntv2/results/batch_followup_20261028_143713_869625/batch16_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch16_command.json", "sha256": "926d30380833c37c983526f916308871f1f1cd9e77193a658bbcd634fc50ab71", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-16",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=16",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-3ebcaf44da60

원래 파일: `ntv2/results/batch_followup_20261028_143713_869625/batch1_command.json` · 원문 SHA256: `60ec8ce27554da1d6c1d9184e79cbb2869dfb62fb798f43914ebb9d7c0887deb`

<details>
<summary>ntv2/results/batch_followup_20261028_143713_869625/batch1_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch1_command.json", "sha256": "60ec8ce27554da1d6c1d9184e79cbb2869dfb62fb798f43914ebb9d7c0887deb", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=1",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-933e30c7320a

원래 파일: `ntv2/results/batch_followup_20261028_143713_869625/batch2_command.json` · 원문 SHA256: `62aed622624751d90e4ded758d73a5e647abf40dd7089d4a7de4cb59a675b908`

<details>
<summary>ntv2/results/batch_followup_20261028_143713_869625/batch2_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch2_command.json", "sha256": "62aed622624751d90e4ded758d73a5e647abf40dd7089d4a7de4cb59a675b908", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-2",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=2",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-9bfea73d4c4f

원래 파일: `ntv2/results/batch_followup_20261028_143713_869625/batch32_command.json` · 원문 SHA256: `a8342d0a3b194126fce49311bba2211d8b4742acd99db35cd868a7fec5369694`

<details>
<summary>ntv2/results/batch_followup_20261028_143713_869625/batch32_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch32_command.json", "sha256": "a8342d0a3b194126fce49311bba2211d8b4742acd99db35cd868a7fec5369694", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-32",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=32",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-a5fab56f8d30

원래 파일: `ntv2/results/batch_followup_20261028_143713_869625/batch4_command.json` · 원문 SHA256: `3d9411f0ed2dafe31b3d768dfbd91d7416b193c74afb16ada01fff531ec6590f`

<details>
<summary>ntv2/results/batch_followup_20261028_143713_869625/batch4_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch4_command.json", "sha256": "3d9411f0ed2dafe31b3d768dfbd91d7416b193c74afb16ada01fff531ec6590f", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-4",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=4",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-dbff3d56a698

원래 파일: `ntv2/results/batch_followup_20261028_143713_869625/batch64_command.json` · 원문 SHA256: `01e1ecaa85dd479825577aea3322153312f9ec466a103c86eeb532667876fdca`

<details>
<summary>ntv2/results/batch_followup_20261028_143713_869625/batch64_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch64_command.json", "sha256": "01e1ecaa85dd479825577aea3322153312f9ec466a103c86eeb532667876fdca", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-64",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=64",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-179e55ddf098

원래 파일: `ntv2/results/batch_followup_20261028_143713_869625/batch8_command.json` · 원문 SHA256: `341519bf05c3674abaf27a0ff8feee2ff9b66b8768e8c55f346f2795155c8f1a`

<details>
<summary>ntv2/results/batch_followup_20261028_143713_869625/batch8_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch8_command.json", "sha256": "341519bf05c3674abaf27a0ff8feee2ff9b66b8768e8c55f346f2795155c8f1a", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-8",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-7ebeb2b48e67

원래 파일: `ntv2/results/batch_followup_20261028_143713_869625/batch96_command.json` · 원문 SHA256: `6b7a9c747e67f2f89fc8b98869e784fcb8013a1988e66b9bb4effffda17ed0ea`

<details>
<summary>ntv2/results/batch_followup_20261028_143713_869625/batch96_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch96_command.json", "sha256": "6b7a9c747e67f2f89fc8b98869e784fcb8013a1988e66b9bb4effffda17ed0ea", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-96",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=96",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-bf2e6eb83f30

원래 파일: `ntv2/results/ntv2_20261027_201332_251243/benchmark_command.json` · 원문 SHA256: `4deee39ebeab2df553341f013f453111138a5b5b1865badc972d495ad7f47225`

<details>
<summary>ntv2/results/ntv2_20261027_201332_251243/benchmark_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/ntv2_20261027_201332_251243/benchmark_command.json", "sha256": "4deee39ebeab2df553341f013f453111138a5b5b1865badc972d495ad7f47225", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/ntv2:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=ntv2_20261027_201332_251243 -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/benchmark.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-e4c88afefccb

원래 파일: `ntv2/results/ntv2_20261027_201332_251243/finetune_command.json` · 원문 SHA256: `c041f16fd03aaa324cb121679ba5726d1155b620644dca792c4f7d871ef3b646`

<details>
<summary>ntv2/results/ntv2_20261027_201332_251243/finetune_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/ntv2_20261027_201332_251243/finetune_command.json", "sha256": "c041f16fd03aaa324cb121679ba5726d1155b620644dca792c4f7d871ef3b646", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/ntv2:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=ntv2_20261027_201332_251243 -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/finetune.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-5c72c77338bf

원래 파일: `ntv2/results/ntv2_20261027_201332_251243/inference_remeasure_command.json` · 원문 SHA256: `2d75e42b9c9caed3793c6d253446c02f1530da14ff7b5e3537ff2f9a81bddeab`

<details>
<summary>ntv2/results/ntv2_20261027_201332_251243/inference_remeasure_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/ntv2_20261027_201332_251243/inference_remeasure_command.json", "sha256": "2d75e42b9c9caed3793c6d253446c02f1530da14ff7b5e3537ff2f9a81bddeab", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/ntv2:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=ntv2_20261027_201332_251243 -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/inference_remeasure.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-429eea5864bb

원래 파일: `ntv2/results/ntv2_20261027_201332_251243/reload_command.json` · 원문 SHA256: `d8acd57a0875e2f5ea5eabbf285fea45472ef7c54e717369a8c4d5413064f7f1`

<details>
<summary>ntv2/results/ntv2_20261027_201332_251243/reload_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/ntv2_20261027_201332_251243/reload_command.json", "sha256": "d8acd57a0875e2f5ea5eabbf285fea45472ef7c54e717369a8c4d5413064f7f1", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/ntv2:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=ntv2_20261027_201332_251243 -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/reload.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-4ce07af029c8

원래 파일: `ntv2/results/ntv2_20261027_201332_251243/tf32_mismatch_reload_command.json` · 원문 SHA256: `d8acd57a0875e2f5ea5eabbf285fea45472ef7c54e717369a8c4d5413064f7f1`

<details>
<summary>ntv2/results/ntv2_20261027_201332_251243/tf32_mismatch_reload_command.json — 명령/설정/환경 원문</summary>


동일 원문은 `ntv2/results/ntv2_20261027_201332_251243/reload_command.json` 블록에 전문 수록했다. 아래 별칭은 동일한 바이트를 원래 경로로 복원한다.
<!-- RESTORE_ALIAS {"path": "ntv2/results/ntv2_20261027_201332_251243/tf32_mismatch_reload_command.json", "source": "ntv2/results/ntv2_20261027_201332_251243/reload_command.json", "sha256": "d8acd57a0875e2f5ea5eabbf285fea45472ef7c54e717369a8c4d5413064f7f1"} -->


</details>

### record-777e836fcf8a

원래 파일: `ntv2/results/sources.json` · 원문 SHA256: `ec9d718268e93f262d932a04fc41c4a1946062bf9ce8a1ed886734cb65d4a740`

<details>
<summary>ntv2/results/sources.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/sources.json", "sha256": "ec9d718268e93f262d932a04fc41c4a1946062bf9ce8a1ed886734cb65d4a740", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/README.md",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/README.md",
    "sha256": "16352a70466c2eac7db56fde1c18bdf24f53515c06c7d55237f3428a032286c7",
    "bytes": 6346
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/config.json",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/config.json",
    "sha256": "27e19a9e3fbba65b933bde793b503734744b5c6e376c74dcd1cb7317876efab8",
    "bytes": 1065
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/esm_config.py",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/esm_config.py",
    "sha256": "a44e859baa08465ecdcd76b0a73f6e4fe011245a212931c403de1149ae9613ec",
    "bytes": 14876
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/model.safetensors",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/model.safetensors",
    "sha256": "7a00ffaaf8f3ca627741a61e568d7e550b5c79aada70c76c8cb861192fbe7c59",
    "bytes": 1993438552
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/modeling_esm.py",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/modeling_esm.py",
    "sha256": "f2b003f45d2fa4f94e92d8bbef927b719dfdd7ab7c7a95743daa2b10ab140cb3",
    "bytes": 58205
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/special_tokens_map.json",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/special_tokens_map.json",
    "sha256": "d6dc30bf018166daab248b0abf7efda6fd1b1e0a2d1bee5b31b23db2ebdaee77",
    "bytes": 101
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/tokenizer_config.json",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/tokenizer_config.json",
    "sha256": "253d338919eba938e50b776f3243cc739462c207fe64e3d2e81cc5e681bee45b",
    "bytes": 129
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/vocab.txt",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/vocab.txt",
    "sha256": "c00e0ad166d6ab3f7540ebc92270392e581bb3106763412f29034b49323e1052",
    "bytes": 28718
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/data/enhancers/test.parquet",
    "url": "https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks/resolve/96d86d567d4cd33536e49b429dc7983121619a08/enhancers/test.parquet",
    "sha256": "0e1862a1809124051014b60ea9021e9d414561c194af44fde0ce0eac5ac220e1",
    "bytes": 41136
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/data/enhancers/train.parquet",
    "url": "https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks/resolve/96d86d567d4cd33536e49b429dc7983121619a08/enhancers/train.parquet",
    "sha256": "698167c5c09b1f98e240560c51e81e540b27aa91e162e3ff1cfcc204ae1ee6e8",
    "bytes": 1465411
  }
]
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

<details><summary>추가 실험 실행 원문: full_training_b512_executed_source.py</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/full_training_b512_executed_source.py", "sha256": "dc1c9a02d54e9a816169e1fa0e2bbd51d8c40db6895fc12a6ec36141f4246f3f", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,os,json,time,csv,gc,argparse,random,pickle
import numpy as np,torch
sys.path.insert(0,os.environ.get('SUPPORT_PATH','/support'));from gpu_support import Meter,write_json
kind=os.environ['MODEL_KIND'];R=Path(os.environ.get('MODEL_ROOT','/work'));D=R/'results'/os.environ['RUN_ID'];B=int(os.environ['BATCH']);mode=os.environ['TRAIN_MODE'];E=2;C=D/f'{mode}_training_b{B}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'mode':mode,'batch':B,'epochs':E if mode=='full' else 0,'seed':42,'precision':'FP32','training_scope':'all backbone parameters plus task head; new separate comparison from earlier partial/frozen-head runs','conditions':[],'allocator_budget_gib':100}
def save():write_json(C/'summary.json',S)
try:
 torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(min(.9,100*2**30/torch.cuda.get_device_properties(0).total_memory));tokenizer=None;alphabet=None
 if kind=='dnabert2':
  from transformers import AutoModel,AutoTokenizer,BertConfig
  P=Path('/workspace/dnabert2_eval/model');tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);net=AutoModel.from_pretrained(P,config=BertConfig.from_pretrained(P),trust_remote_code=True,local_files_only=True);rows=list(csv.DictReader(Path('/workspace/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv').open()));order=np.random.default_rng(42).permutation(len(rows));data={s:[rows[i] for i in idx] for s,idx in [('train',order[:800]),('validation',order[800:900]),('test',order[900:])]};head=torch.nn.Linear(net.config.hidden_size,18);task='multilabel';S['split_note']='new seed42 800/100/100 within BEND originaltrain1000; prior independent originaltest metrics retained; no donor/chromosome independent validation claimed'
 elif kind=='ntv2':
  from transformers import AutoModelForMaskedLM,AutoTokenizer
  P=R/'models/checkpoint';tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);base=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True);net=base.esm;del base;data=json.loads((R/'data/inputs.json').read_text());head=torch.nn.Linear(net.config.hidden_size,2);task='classification';S['head_note']='masked-mean pooled new linear head, distinct from earlier native CLS MLP head; fixed within this batch comparison'
 elif kind=='rnafm':
  import fm
  torch.serialization.add_safe_globals([argparse.Namespace]);net,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');data=json.loads((R/'data/inputs.json').read_text());head=torch.nn.Linear(640,max(r['label'] for r in data['train'])+1);task='classification';S['head_note']='all backbone plus linear RNA-type classifier trained; unlike previous frozen embeddings+linear-only training'
 elif kind=='esm2':
  import esm,lmdb
  raw=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False);reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);net,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',raw,reg);del raw,reg;data={}
  for split,n in [('train',1024),('valid',128),('test',512)]:
   env=lmdb.open(str(R/'data/fluorescence'/f'fluorescence_{split}.lmdb'),readonly=True,lock=False)
   with env.begin() as txn:
    size=pickle.loads(txn.get(b'num_examples'));indices=random.Random(42).sample(range(size),n);rs=[pickle.loads(txn.get(str(i).encode())) for i in indices]
   env.close();data['validation' if split=='valid' else split]=[{'id':split+'_'+str(i),'sequence':r['primary'],'label':float(r['log_fluorescence'][0])} for i,r in zip(indices,rs)]
  head=torch.nn.Linear(1280,1);task='regression';S['training_scope']='full backbone+linear GFP regression, distinct from earlier partial adapter/head training';mu=float(np.mean([r['label'] for r in data['train']]));sd=float(np.std([r['label'] for r in data['train']]));S['target_standardization']={'mean':mu,'std':sd}
 elif kind=='openspliceai':
  from openspliceai.train_base.openspliceai import SpliceAI
  W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);net=SpliceAI(32,W,AR);net.load_state_dict(torch.load(R/'source/models/openspliceai-mane/10000nt/model_10000nt_rs10.pt',map_location='cpu',weights_only=True));head=torch.nn.Identity();data=json.loads((R/'data/inputs.json').read_text());task='splice';S['training_scope']='one seed10model full training, not five-model inference ensemble'
 else:raise ValueError(kind)
 data={s:data[s] for s in ['train','validation','test']};S['split_ids']={s:[r['id'] for r in rs] for s,rs in data.items()};S['split_counts']={s:len(rs) for s,rs in data.items()};assert B<=len(data['train']);allrows=[r for rs in data.values() for r in rs];offsets={};encoded={};ys={};pos=0
 if tokenizer is not None:
  z=tokenizer([r['sequence'] for r in allrows],padding=True,truncation=False,return_tensors='pt');encoded={k:z[k] for k in ['input_ids','attention_mask']}
 elif alphabet is not None:
  _,_,z=alphabet.get_batch_converter()([(r['id'],r['sequence']) for r in allrows]);encoded={'input_ids':z,'attention_mask':z.ne(alphabet.padding_idx).long()}
 else:
  tab=np.zeros((256,4),np.float32)
  for j,c in enumerate('ACGT'):tab[ord(c),j]=1
  encoded={'onehot':torch.from_numpy(np.stack([tab[np.frombuffer(r['sequence'][2500:17500].encode(),np.uint8)].T for r in allrows]))}
 S['fixed_pool_padded_shapes']={k:list(v.shape) for k,v in encoded.items()};S['raw_training_lengths']={'min':min(len(r['sequence']) for r in data['train']),'max':max(len(r['sequence']) for r in data['train']),'mean':float(np.mean([len(r['sequence']) for r in data['train']]))};S['padding_policy']='one fixed full split pool width across all batch sizes, no truncation; valid lengths separately reported'
 for split,rs in data.items():
  offsets[split]=np.arange(pos,pos+len(rs));pos+=len(rs)
  if task=='multilabel':
   y=torch.zeros(len(rs),18)
   for i,r in enumerate(rs):y[i,json.loads(r['labels'])]=1
  elif task=='splice':y=torch.tensor([r['labels_10000'][2500:7500] for r in rs])
  else:y=torch.tensor([r['label'] for r in rs],dtype=torch.float32 if task=='regression' else torch.long)
  ys[split]=y
 net=net.float().cuda();head=head.float().cuda();torch.cuda.synchronize();S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for p in net.parameters())+sum(p.numel() for p in head.parameters());params=list(net.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01);before=next(head.parameters()).detach().cpu().clone() if task!='splice' else net.initial_conv.weight.detach().cpu().clone()
 def output(ix):
  x={k:v[ix].cuda() for k,v in encoded.items()}
  if task=='splice':return net(x['onehot'])
  if kind=='dnabert2':h=net(input_ids=x['input_ids'],attention_mask=x['attention_mask'])[0];mask=x['attention_mask'].unsqueeze(-1)
  elif kind=='ntv2':h=net(input_ids=x['input_ids'],attention_mask=x['attention_mask'],return_dict=True).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1)
  else:
   layer=33 if kind=='esm2' else 12;h=net(x['input_ids'],repr_layers=[layer],return_contacts=False)['representations'][layer];mask=x['attention_mask'].clone();mask[:,0]=0;last=mask.sum(1);mask[torch.arange(len(mask),device='cuda'),last]=0;mask=mask.unsqueeze(-1)
  mask=mask.to(h.dtype);return head((h*mask).sum(1)/mask.sum(1).clamp_min(1))
 def lossfn(z,y):
  if task=='regression':return torch.nn.functional.mse_loss(z.squeeze(-1),(y-mu)/sd)
  if task=='multilabel':return torch.nn.functional.binary_cross_entropy_with_logits(z,y)
  if task=='splice':return torch.nn.functional.nll_loss(torch.log(z.clamp_min(1e-10)),y,ignore_index=-1)
  return torch.nn.functional.cross_entropy(z,y)
 def evaluate(split):
  net.eval();head.eval();outs=[];losses=[]
  with torch.no_grad():
   for start in range(0,len(data[split]),8):
    idx=offsets[split][start:start+8];z=output(idx);y=ys[split][start:start+8].cuda();losses.append(float(lossfn(z,y))*len(idx));outs.append(z.cpu().numpy())
  z=np.concatenate(outs);y=ys[split].numpy();metrics={'mean_loss':sum(losses)/len(y),'unique_samples':len(y)}
  if task in ['classification','multilabel']:
   from sklearn.metrics import accuracy_score,f1_score
  if task=='classification':metrics.update(accuracy=float(accuracy_score(y,z.argmax(1))),macro_f1=float(f1_score(y,z.argmax(1),average='macro',zero_division=0)))
  if task=='multilabel':metrics.update(micro_f1=float(f1_score(y,z>0,average='micro',zero_division=0)))
  if task=='regression':
   from scipy.stats import spearmanr
   pred=z[:,0]*sd+mu;metrics.update(mse=float(np.mean((pred-y)**2)),spearman=float(spearmanr(pred,y).statistic))
  if task=='splice':
   from sklearn.metrics import average_precision_score
   metrics['acceptor_donor_AUPRC']=[float(average_precision_score((y==k).ravel(),z[:,k,:].ravel())) for k in [1,2]]
  return metrics,z
 net.train();head.train();t,base=meter.begin('actual_supervised_training');steps=[];n=len(data['train']);limit=min(n,3*B) if mode=='probe' else n
 for epoch in range(1 if mode=='probe' else E):
  order=np.random.default_rng(42+epoch).permutation(n)[:limit]
  for start in range(0,limit,B):
   local=order[start:start+B];ix=offsets['train'][local];torch.cuda.synchronize();t0=time.time();opt.zero_grad(set_to_none=True);z=output(ix);loss=lossfn(z,ys['train'][local].cuda());assert torch.isfinite(loss);loss.backward();S.setdefault('observed_parameters_with_gradient',sum(p.numel() for p in params if p.grad is not None));opt.step();torch.cuda.synchronize();steps.append({'epoch':epoch,'n':len(ix),'seconds':time.time()-t0,'loss':float(loss.detach())});print('STEP',kind,B,epoch,start+len(ix),'/',limit,'loss',steps[-1]['loss'],flush=True);del z,loss
 S['training']=meter.end(t,base);r=S['training'];r.update(batch_size=B,unique_training_cells=n if mode=='full' else limit,epochs=E if mode=='full' else 0,optimizer_steps=len(steps),processed_inputs=sum(s['n'] for s in steps),allocated_fraction_shared_ram=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib']);r['inputs_per_second']=r['processed_inputs']/r['seconds'];S['training_steps']=steps;after=next(head.parameters()).detach().cpu() if task!='splice' else net.initial_conv.weight.detach().cpu();S['head_or_conv_weights_updated']=not torch.equal(before,after);assert S['head_or_conv_weights_updated'];save();opt.zero_grad(set_to_none=True);del opt;gc.collect();torch.cuda.empty_cache()
 if mode=='full':
  S['validation'],_=evaluate('validation');S['test'],test=evaluate('test');np.save(C/'test_outputs.npy',test);ck=C/'model.pt';torch.save({'backbone':net.state_dict(),'head':head.state_dict(),'model':kind,'training_settings':{'seed':42,'lr':1e-5,'weight_decay':.01,'epochs':2,'batch':B},'split_ids':S['split_ids'],'source_weights':'original public checkpoint in local models folder','labels':json.loads((R/'data/input_audit.json').read_text()).get('labels') if (R/'data/input_audit.json').exists() else None,'target_standardization':S.get('target_standardization')},ck);saved=torch.load(ck,map_location='cpu',weights_only=True);net.load_state_dict(saved['backbone']);head.load_state_dict(saved['head']);del saved;_,again=evaluate('test');S['reload_max_abs_outputs']=float(np.abs(test-again).max());assert np.allclose(test,again,atol=1e-4,rtol=1e-4);S['saved_model_reload_verified']=True
 S['status']='complete';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b128_executed_source.py</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b128_executed_source.py", "sha256": "11095b56bf1768b5e12be9814eb1ec572938ec22412ab702bc3ee3c7e8babce6", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,time,json,gc,argparse
import numpy as np,torch
sys.path.insert(0,'/support');from gpu_support import Meter,write_json
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];kind=os.environ['MODEL_KIND'];B=int(os.environ['BATCH']);C=D/f'batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'batch':B,'task':'inference resource batch sweep only, no training','conditions':[]}
def save():write_json(C/'summary.json',S)
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(min(.9,100*2**30/torch.cuda.get_device_properties(0).total_memory));S['allocator_budget_gib']=100;meter.phase='load';t=time.time();tokenizer=None
 if kind=='scvi':
  import scvi
  P=R/'models/scvi_20261027_185327_767988/train_10176';m=scvi.model.SCVI.load(str(P),accelerator='cpu');m.to_device('cuda:0');m.module.eval();nets=[m.module];ad=m.adata;assert ad.n_vars==2000 and ad.n_obs>=B;S['checkpoint']=str(P);S['input']={'unique_cells':B,'genes':ad.n_vars,'raw_counts_source':'saved adata.h5ad layers counts, actual 10x PBMC4k+8k','original_pool_cells':ad.n_obs,'batch_covariate':'sample_id, biological batch distinct from GPU batch'}
  x=ad[:B].copy()
  def forward():return m.get_latent_representation(adata=x,batch_size=B,give_mean=True)
 else:
  records=json.loads((D/'inputs.json').read_text());assert len(records)>=B
  if kind=='dnabert2':
   from transformers import AutoModel,AutoTokenizer,BertConfig
   P=Path('/workspace/dnabert2_eval/model');tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);m=AutoModel.from_pretrained(P,config=BertConfig.from_pretrained(P),trust_remote_code=True,local_files_only=True).float().cuda().eval();nets=[m]
  elif kind=='ntv2':
   from transformers import AutoModelForMaskedLM,AutoTokenizer
   P=R/'models/checkpoint';tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);m=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True).float().cuda().eval();nets=[m]
  elif kind=='rnafm':
   import fm
   torch.serialization.add_safe_globals([argparse.Namespace]);m,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');m=m.float().cuda().eval();nets=[m];converter=alphabet.get_batch_converter()
  elif kind=='openspliceai':
   from openspliceai.train_base.openspliceai import SpliceAI
   W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);nets=[]
   for seed in range(10,15):
    net=SpliceAI(32,W,AR).float().cuda();net.load_state_dict(torch.load(R/f'source/models/openspliceai-mane/10000nt/model_10000nt_rs{seed}.pt',map_location='cpu',weights_only=True));net.eval();nets.append(net)
  else:raise ValueError(kind)
 torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for net in nets for p in net.parameters());S['parameters_trained_in_this_run']=0;S['theoretical_weight_storage_gib']=sum(p.numel()*p.element_size() for net in nets for p in net.parameters())/2**30;S['parameters_used_backbone']=sum(p.numel() for p in m.esm.parameters()) if kind=='ntv2' else S['parameters_total'];S['checkpoint_reference']={'scvi':'models/scvi_20261027_185327_767988/train_10176','dnabert2':'/workspace/dnabert2_eval/model + model_revision.txt','ntv2':'models/checkpoint; existing setup/download provenance','rnafm':'models/RNA-FM_pretrained.pth; existing data/sources.json','openspliceai':'source/models/openspliceai-mane/10000nt seeds10..14'}[kind];S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'precision':'FP32','tf32':False};save()
 if kind!='scvi':
  cache=D/'token_cache.pt'
  if not cache.exists():
   if tokenizer is not None:
    encoded=tokenizer([r['sequence'] for r in records],padding=True,truncation=False,return_tensors='pt');encoded={k:encoded[k] for k in ['input_ids','attention_mask']}
   elif kind=='rnafm':
    _,_,tok=converter([(r['id'],r['sequence']) for r in records]);encoded={'input_ids':tok}
   else:
    tab=np.zeros((256,4),np.uint8)
    for j,c in enumerate('ACGT'):tab[ord(c),j]=1
    encoded={'onehot':torch.from_numpy(np.stack([tab[np.frombuffer(r['sequence'].encode(),np.uint8)].T for r in records]))}
   torch.save(encoded,cache)
  encoded=torch.load(cache,map_location='cpu',weights_only=True);cpu={k:v[:B] for k,v in encoded.items()};S['input']={'unique_sequences':B,'pool_sequences':len(records),'raw_length_min':min(len(r['sequence']) for r in records[:B]),'raw_length_max':max(len(r['sequence']) for r in records[:B]),'input_shapes':{k:list(v.shape) for k,v in cpu.items()},'tokenizer_truncation':False,'padding':'same pool-wide longest token width fixed across batches; explicit crop lengths in inputs provenance','ids':[r['id'] for r in records[:B]]}
  if kind=='ntv2':assert cpu['input_ids'].shape[1]<=2048
  if kind=='rnafm':assert cpu['input_ids'].shape[1]<=1024
  def forward():
   x={k:(v.cuda().float() if k=='onehot' else v.cuda()) for k,v in cpu.items()}
   if kind in ['dnabert2','ntv2']:
    h=m(**x)[0] if kind=='dnabert2' else m.esm(**x,return_dict=True).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1).to(h.dtype);out=(h*mask).sum(1)/mask.sum(1)
   elif kind=='rnafm':
    h=m(x['input_ids'],repr_layers=[12],return_contacts=False)['representations'][12];out=torch.stack([h[i,1:len(r['sequence'])+1].mean(0) for i,r in enumerate(records[:B])])
   else:
    out=None
    for net in nets:
     z=net(x['onehot']);out=z if out is None else out+z
    out=out/len(nets)
   return out.cpu().numpy()
 meter.phase='warmup';t=time.time()
 with torch.no_grad():out=forward()
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;assert out.shape[0]==B and np.isfinite(out).all();del out;gc.collect();t,base=meter.begin('inference');steps=0
 with torch.no_grad():
  while steps<2 or time.time()-t<15:out=forward();assert np.isfinite(out).all();torch.cuda.synchronize();steps+=1
 r=meter.end(t,base,batch_size=B,unique_inputs=B,processed_inputs=B*steps,repeats=steps);r['inputs_per_second']=B*steps/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];r['model_only_allocated_gib']=S['model_only_postload']['allocated_gib'];S['conditions']=[r];np.save(C/'outputs.npy',out);assert np.array_equal(np.load(C/'outputs.npy'),out);S['output_shape']=list(out.shape);S['finite_and_exact_save_reload']=True
 first=D/'reference_first.npy'
 if not first.exists():np.save(first,out[0]);S['reference_first_created']=True
 else:
  ref=np.load(first);diff=float(np.abs(ref-out[0]).max());S['batch_invariance_max_abs']=diff;assert np.allclose(ref,out[0],atol=1e-4,rtol=1e-4),(kind,diff)
 S['status']='complete';save();print('BATCH_COMPLETE',kind,B,r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: full_training_b512_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/full_training_b512_command.json", "sha256": "5d5ef67dcad02b1a73c66696ab4bf0dc3b2e7a22adaf2d94b457a607adc12547", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=512",
  "-e",
  "TRAIN_MODE=full",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: full_training_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/full_training_b8_command.json", "sha256": "d5cec48c8e6d28e95f1cf87134fe7c6d1e88bef984d64a0821e1bc03b7e905ee", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "TRAIN_MODE=full",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b128_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b128_command.json", "sha256": "e8dfab5cba315951916ea2f74843490f4767491ccf4eba8c6883408af635aac0", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=128",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b256_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b256_command.json", "sha256": "a3930b8afd2501d3fa0da8561ec6eb27ace7993b3c99e63e856889caa451b88a", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=256",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b384_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b384_command.json", "sha256": "297cfe7faf3ed52c584382a1d4f5555ec93997d410e91564e8dea16c3b097514", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=384",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b448_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b448_command.json", "sha256": "8f615827ec7ee61a2ff5c53b9270a8b9af4efca38cdbcd077ecf3f684fc2a390", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=448",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b512_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b512_command.json", "sha256": "3312cf01bf90799a61fd7de9a56d3156ce31739c2fa773203de3679451307aeb", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=512",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b544_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b544_command.json", "sha256": "00733f16285f78ad884ba74fe371dcede4d35c0aaf4f873966d8e0ce51b8ff00", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=544",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b560_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b560_command.json", "sha256": "247e45cb75c823907bd3f0e2fa0a5cb64001da100b910340f42ff6f3a43f708c", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=560",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b576_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b576_command.json", "sha256": "ad90644a82e0ae7264c0ad3e8020bce04e8cc57541efb141c354e831e31d00c5", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=576",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b592_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b592_command.json", "sha256": "0a00902408b70e9a7aa75dec98c32f542fe274ea864a8dd150817a4ce9351403", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=592",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b608_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b608_command.json", "sha256": "0b519ab53f3900f9184631fc075ae709e2f630add7f0fcc5465be6976e045343", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=608",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b624_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b624_command.json", "sha256": "0aa65d453d5a295c3de18aa8376db28fdb19ba7fa817a16d76ac31f4c88cf843", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=624",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/inference_b8_command.json", "sha256": "0ce0c56b05e0ffba6d82b2c079c375991e691e1fb1543614009da31d817a0720", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/inference_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b128_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/training_probe_b128_command.json", "sha256": "3eb2def60bfa4c5a86b2beeac31418496360993c57a99c725d578f4859269bbf", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=128",
  "-e",
  "TRAIN_MODE=probe",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b256_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/training_probe_b256_command.json", "sha256": "23b925f827662e84437a771cd1b0f7128f5486f333541b5f8dd949a44eaa13fc", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=256",
  "-e",
  "TRAIN_MODE=probe",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b384_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/training_probe_b384_command.json", "sha256": "2987cf5c29ee78b8791ae111d873685728ca562badfcc98055c4ed837d19170e", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=384",
  "-e",
  "TRAIN_MODE=probe",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b448_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/training_probe_b448_command.json", "sha256": "af660150ec4a6ae40bd6363908520fb9240b8166ae65c286692eb7c60ae19d40", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=448",
  "-e",
  "TRAIN_MODE=probe",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b512_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/training_probe_b512_command.json", "sha256": "b33cae018588aa97f7e41b905af132798664b03f11b27a5056651f294e8f09fa", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=512",
  "-e",
  "TRAIN_MODE=probe",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b64_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/training_probe_b64_command.json", "sha256": "6052ddda7f6e65c2b39112e2b22d0b4b2ffa36811ffdf56ac091ae7235c45604", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=64",
  "-e",
  "TRAIN_MODE=probe",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/training_probe_b8_command.json", "sha256": "e3a06fe7834e0de456627374604558900328421244cd3be7ddf851c45ec4c3be", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_152621_008750",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "TRAIN_MODE=probe",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: remaining_queue_152f0b87bce32235.py</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/reproduction_sources/remaining_queue_152f0b87bce32235.py", "sha256": "152f0b87bce32235a41f8616d74f03deabd611eef85dde6419bd68da175973a6", "encoding": "utf-8", "added_final_newline": false} -->
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

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/reproduction_sources/update_reports_305433d7fed68d6f.py", "sha256": "305433d7fed68d6ff20850b0d00d18708a0de3d92b36924d41dbbfe168034b8e", "encoding": "utf-8", "added_final_newline": false} -->
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

<details><summary>추가 실험 실행 원문: prepare_genome_pools_46b78d96d41c9f09.py</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/reproduction_sources/prepare_genome_pools_46b78d96d41c9f09.py", "sha256": "46b78d96d41c9f095a49916c206deb4ce29f0d16065ee5dd436a0fdd02bb8070", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,os,hashlib
import pysam
R=Path('/work');kind=os.environ['MODEL_KIND'];fa=pysam.FastaFile('/reference/GRCh38.primary_assembly.genome.fa');length=6000 if kind=='ntv2' else 15000;target=1024 if kind=='ntv2' else 16000;records=[];seen=set();chrom='chr2'
# Unique nonoverlapping real reference windows; input-only benchmark, never claimed labelled training data.
for chrom in ['chr2','chr4']:
 for start in range(2000000,fa.get_reference_length(chrom)-length,length):
  seq=fa.fetch(chrom,start,start+length).upper()
  if set(seq)<=set('ACGT') and seq not in seen:
   seen.add(seq);records.append({'id':f'{chrom}:{start}-{start+length}','sequence':seq,'reference':'GRCh38.primary_assembly.genome.fa','chromosome':chrom,'start0':start,'end0':start+length,'task':'unlabelled embedding/splice-score inference only'})
  if len(records)>=target:break
 if len(records)>=target:break
assert len(records)==target
out=R/'data/real_training_v3';out.mkdir(exist_ok=True);(out/'inference_inputs.json').write_text(json.dumps(records));(out/'inference_audit.json').write_text(json.dumps({'unique_sequences':len(records),'raw_nt':length,'source_reference':'existing BEND GRCh38.primary_assembly.genome.fa','selection':'chr2 then chr4 from2Mb, fixed nonoverlapping reference windows, ACGT only, exactsequence dedup; no repeated or synthetic sequences','labels':None,'task':'inference only; not substituted for labelled training splits'},indent=2));print(kind,len(records),length,flush=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_audit_f09c12349d92f115.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/real_input_capacity_20261102_152621_008750/reproduction_sources/inference_audit_f09c12349d92f115.json", "sha256": "f09c12349d92f11508f7369c4ebe046413020efa4eb183d16b183c8248164a80", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "unique_sequences": 1024,
  "raw_nt": 6000,
  "source_reference": "existing BEND GRCh38.primary_assembly.genome.fa",
  "selection": "chr2 from2Mb, fixed nonoverlapping reference windows, ACGT only, exactsequence dedup; no repeated or synthetic sequences",
  "labels": null,
  "task": "inference only; not substituted for labelled training splits"
}
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 요인 비교 실행·보고 원문: executed_worker.py</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/factor_control_20261102_182012_734207/report_snapshots/executed_worker_edd5f6f99647c80e.py", "sha256": "edd5f6f99647c80e961dd6d68cc22ce156cb57bd447583b1d837358246c5ca57", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,os,json,time,csv,gc,argparse,random,pickle
import numpy as np,torch
sys.path.insert(0,os.environ.get('SUPPORT_PATH','/support'));from gpu_support import Meter,write_json
kind=os.environ['MODEL_KIND'];R=Path(os.environ.get('MODEL_ROOT','/work'));D=R/'results'/os.environ['RUN_ID'];B=int(os.environ['BATCH']);mode=os.environ['TRAIN_MODE'];E=int(os.environ.get('EPOCHS','2'));N=int(os.environ.get('TRAIN_N','0'));C=D/f'{mode}_training_b{B}_e{E}_n{N}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'mode':mode,'batch':B,'epochs':E if mode=='full' else 0,'seed':42,'precision':'FP32','training_scope':'all backbone parameters plus task head; new separate comparison from earlier partial/frozen-head runs','conditions':[],'allocator_budget_gib':100}
def save():write_json(C/'summary.json',S)
try:
 torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(min(.9,100*2**30/torch.cuda.get_device_properties(0).total_memory));tokenizer=None;alphabet=None
 if kind=='dnabert2':
  from transformers import AutoModel,AutoTokenizer,BertConfig
  P=Path('/workspace/dnabert2_eval/model');tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);net=AutoModel.from_pretrained(P,config=BertConfig.from_pretrained(P),trust_remote_code=True,local_files_only=True);rows=list(csv.DictReader(Path('/workspace/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv').open()));order=np.random.default_rng(42).permutation(len(rows));data={s:[rows[i] for i in idx] for s,idx in [('train',order[:800]),('validation',order[800:900]),('test',order[900:])]};head=torch.nn.Linear(net.config.hidden_size,18);task='multilabel';S['split_note']='new seed42 800/100/100 within BEND originaltrain1000; prior independent originaltest metrics retained; no donor/chromosome independent validation claimed'
 elif kind=='ntv2':
  from transformers import AutoModelForMaskedLM,AutoTokenizer
  P=R/'models/checkpoint';tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);base=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True);net=base.esm;del base;data=json.loads((R/'data/inputs.json').read_text());head=torch.nn.Linear(net.config.hidden_size,2);task='classification';S['head_note']='masked-mean pooled new linear head, distinct from earlier native CLS MLP head; fixed within this batch comparison'
 elif kind=='rnafm':
  import fm
  torch.serialization.add_safe_globals([argparse.Namespace]);net,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');data=json.loads((R/'data/inputs.json').read_text());head=torch.nn.Linear(640,max(r['label'] for r in data['train'])+1);task='classification';S['head_note']='all backbone plus linear RNA-type classifier trained; unlike previous frozen embeddings+linear-only training'
 elif kind=='esm2':
  import esm,lmdb
  raw=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False);reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);net,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',raw,reg);del raw,reg;data={}
  for split,n in [('train',1024),('valid',128),('test',512)]:
   env=lmdb.open(str(R/'data/fluorescence'/f'fluorescence_{split}.lmdb'),readonly=True,lock=False)
   with env.begin() as txn:
    size=pickle.loads(txn.get(b'num_examples'));indices=random.Random(42).sample(range(size),n);rs=[pickle.loads(txn.get(str(i).encode())) for i in indices]
   env.close();data['validation' if split=='valid' else split]=[{'id':split+'_'+str(i),'sequence':r['primary'],'label':float(r['log_fluorescence'][0])} for i,r in zip(indices,rs)]
  head=torch.nn.Linear(1280,1);task='regression';S['training_scope']='full backbone+linear GFP regression, distinct from earlier partial adapter/head training';mu=float(np.mean([r['label'] for r in data['train']]));sd=float(np.std([r['label'] for r in data['train']]));S['target_standardization']={'mean':mu,'std':sd}
 elif kind=='openspliceai':
  from openspliceai.train_base.openspliceai import SpliceAI
  W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);net=SpliceAI(32,W,AR);net.load_state_dict(torch.load(R/'source/models/openspliceai-mane/10000nt/model_10000nt_rs10.pt',map_location='cpu',weights_only=True));head=torch.nn.Identity();data=json.loads((R/'data/inputs.json').read_text());task='splice';S['training_scope']='one seed10model full training, not five-model inference ensemble'
 else:raise ValueError(kind)
 data={s:data[s] for s in ['train','validation','test']};S['split_ids']={s:[r['id'] for r in rs] for s,rs in data.items()};S['split_counts']={s:len(rs) for s,rs in data.items()};assert B<=len(data['train']);allrows=[r for rs in data.values() for r in rs];offsets={};encoded={};ys={};pos=0
 if tokenizer is not None:
  z=tokenizer([r['sequence'] for r in allrows],padding=True,truncation=False,return_tensors='pt');encoded={k:z[k] for k in ['input_ids','attention_mask']}
 elif alphabet is not None:
  _,_,z=alphabet.get_batch_converter()([(r['id'],r['sequence']) for r in allrows]);encoded={'input_ids':z,'attention_mask':z.ne(alphabet.padding_idx).long()}
 else:
  tab=np.zeros((256,4),np.float32)
  for j,c in enumerate('ACGT'):tab[ord(c),j]=1
  encoded={'onehot':torch.from_numpy(np.stack([tab[np.frombuffer(r['sequence'][2500:17500].encode(),np.uint8)].T for r in allrows]))}
 S['fixed_pool_padded_shapes']={k:list(v.shape) for k,v in encoded.items()};S['raw_training_lengths']={'min':min(len(r['sequence']) for r in data['train']),'max':max(len(r['sequence']) for r in data['train']),'mean':float(np.mean([len(r['sequence']) for r in data['train']]))};S['padding_policy']='one fixed full split pool width across all batch sizes, no truncation; valid lengths separately reported'
 for split,rs in data.items():
  offsets[split]=np.arange(pos,pos+len(rs));pos+=len(rs)
  if task=='multilabel':
   y=torch.zeros(len(rs),18)
   for i,r in enumerate(rs):y[i,json.loads(r['labels'])]=1
  elif task=='splice':y=torch.tensor([r['labels_10000'][2500:7500] for r in rs])
  else:y=torch.tensor([r['label'] for r in rs],dtype=torch.float32 if task=='regression' else torch.long)
  ys[split]=y
 if N:
  assert 0<N<=len(data['train']);data['train']=data['train'][:N];S['split_ids']['train']=S['split_ids']['train'][:N];S['split_counts']['train']=N
 selected=data['train'];S['raw_training_lengths']={'min':min(len(r['sequence']) for r in selected),'max':max(len(r['sequence']) for r in selected),'mean':float(np.mean([len(r['sequence']) for r in selected]))}
 y_selected=ys['train'][:len(selected)].numpy()
 if task=='multilabel':S['selected_train_positive_counts_per_class']=y_selected.sum(0).tolist()
 elif task in ['classification','splice']:
  keys,counts=np.unique(y_selected,return_counts=True);S['selected_train_label_counts']={str(int(k)):int(v) for k,v in zip(keys,counts)}
 else:S['selected_train_label_summary']={'mean':float(y_selected.mean()),'std':float(y_selected.std()),'min':float(y_selected.min()),'max':float(y_selected.max())}
 S['factor_control']={'fixed_model':True,'fixed_precision':'FP32','fixed_seed':42,'fixed_batch':B,'epochs':E,'actual_unique_train_n':len(data['train']),'padding':'original full split token cache width held fixed even when train count is reduced','initialization':'reload same public pretrained weights and fresh seed42 head each independent condition','optimizer':'AdamW lr1e-5 weight_decay.01','label_standardization':'ESM target mean/std kept from the complete1024train pool across count subsets'}
 net=net.float().cuda();head=head.float().cuda();torch.cuda.synchronize();S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for p in net.parameters())+sum(p.numel() for p in head.parameters());params=list(net.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01);before=next(head.parameters()).detach().cpu().clone() if task!='splice' else net.initial_conv.weight.detach().cpu().clone()
 def output(ix):
  x={k:v[ix].cuda() for k,v in encoded.items()}
  if task=='splice':return net(x['onehot'])
  if kind=='dnabert2':h=net(input_ids=x['input_ids'],attention_mask=x['attention_mask'])[0];mask=x['attention_mask'].unsqueeze(-1)
  elif kind=='ntv2':h=net(input_ids=x['input_ids'],attention_mask=x['attention_mask'],return_dict=True).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1)
  else:
   layer=33 if kind=='esm2' else 12;h=net(x['input_ids'],repr_layers=[layer],return_contacts=False)['representations'][layer];mask=x['attention_mask'].clone();mask[:,0]=0;last=mask.sum(1);mask[torch.arange(len(mask),device='cuda'),last]=0;mask=mask.unsqueeze(-1)
  mask=mask.to(h.dtype);return head((h*mask).sum(1)/mask.sum(1).clamp_min(1))
 def lossfn(z,y):
  if task=='regression':return torch.nn.functional.mse_loss(z.squeeze(-1),(y-mu)/sd)
  if task=='multilabel':return torch.nn.functional.binary_cross_entropy_with_logits(z,y)
  if task=='splice':return torch.nn.functional.nll_loss(torch.log(z.clamp_min(1e-10)),y,ignore_index=-1)
  return torch.nn.functional.cross_entropy(z,y)
 def evaluate(split):
  net.eval();head.eval();outs=[];losses=[]
  with torch.no_grad():
   for start in range(0,len(data[split]),8):
    idx=offsets[split][start:start+8];z=output(idx);y=ys[split][start:start+8].cuda();losses.append(float(lossfn(z,y))*len(idx));outs.append(z.cpu().numpy())
  z=np.concatenate(outs);y=ys[split].numpy();metrics={'mean_loss':sum(losses)/len(y),'unique_samples':len(y)}
  if task in ['classification','multilabel']:
   from sklearn.metrics import accuracy_score,f1_score
  if task=='classification':metrics.update(accuracy=float(accuracy_score(y,z.argmax(1))),macro_f1=float(f1_score(y,z.argmax(1),average='macro',zero_division=0)))
  if task=='multilabel':metrics.update(micro_f1=float(f1_score(y,z>0,average='micro',zero_division=0)))
  if task=='regression':
   from scipy.stats import spearmanr
   pred=z[:,0]*sd+mu;metrics.update(mse=float(np.mean((pred-y)**2)),spearman=float(spearmanr(pred,y).statistic))
  if task=='splice':
   from sklearn.metrics import average_precision_score
   metrics['acceptor_donor_AUPRC']=[float(average_precision_score((y==k).ravel(),z[:,k,:].ravel())) for k in [1,2]]
  return metrics,z
 net.train();head.train();t,base=meter.begin('actual_supervised_training');steps=[];n=len(data['train']);limit=min(n,3*B) if mode=='probe' else n
 for epoch in range(1 if mode=='probe' else E):
  order=np.random.default_rng(42+epoch).permutation(n)[:limit]
  for start in range(0,limit,B):
   local=order[start:start+B];ix=offsets['train'][local];torch.cuda.synchronize();t0=time.time();opt.zero_grad(set_to_none=True);z=output(ix);loss=lossfn(z,ys['train'][local].cuda());assert torch.isfinite(loss);loss.backward();S.setdefault('observed_parameters_with_gradient',sum(p.numel() for p in params if p.grad is not None));opt.step();torch.cuda.synchronize();steps.append({'epoch':epoch,'n':len(ix),'seconds':time.time()-t0,'loss':float(loss.detach())});print('STEP',kind,B,epoch,start+len(ix),'/',limit,'loss',steps[-1]['loss'],flush=True);del z,loss
 S['training']=meter.end(t,base);r=S['training'];r.update(batch_size=B,unique_training_cells=n if mode=='full' else limit,epochs=E if mode=='full' else 0,optimizer_steps=len(steps),processed_inputs=sum(s['n'] for s in steps),allocated_fraction_shared_ram=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib']);r['inputs_per_second']=r['processed_inputs']/r['seconds'];S['training_steps']=steps;after=next(head.parameters()).detach().cpu() if task!='splice' else net.initial_conv.weight.detach().cpu();S['head_or_conv_weights_updated']=not torch.equal(before,after);assert S['head_or_conv_weights_updated'];save();opt.zero_grad(set_to_none=True);del opt;gc.collect();torch.cuda.empty_cache()
 if mode=='full':
  S['validation'],_=evaluate('validation');S['test'],test=evaluate('test');np.save(C/'test_outputs.npy',test);ck=C/'model.pt';torch.save({'backbone':net.state_dict(),'head':head.state_dict(),'model':kind,'training_settings':{'seed':42,'lr':1e-5,'weight_decay':.01,'epochs':2,'batch':B},'split_ids':S['split_ids'],'source_weights':'original public checkpoint in local models folder','labels':json.loads((R/'data/input_audit.json').read_text()).get('labels') if (R/'data/input_audit.json').exists() else None,'target_standardization':S.get('target_standardization')},ck);saved=torch.load(ck,map_location='cpu',weights_only=True);net.load_state_dict(saved['backbone']);head.load_state_dict(saved['head']);del saved;_,again=evaluate('test');S['reload_max_abs_outputs']=float(np.abs(test-again).max());assert np.allclose(test,again,atol=1e-4,rtol=1e-4);S['saved_model_reload_verified']=True
 S['status']='complete';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 요인 비교 실행·보고 원문: executed_controller.py</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/factor_control_20261102_182012_734207/report_snapshots/executed_controller_73cb685d1ccd06e4.py", "sha256": "73cb685d1ccd06e466865497eda421c3e07ccf539d2f8b8d1ecebfc1431ffb61", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,json,time,datetime,subprocess,shlex,shutil
ROOT=Path('/home/hyein/ai-benchmark');P=ROOT/'FACTOR_EXPERIMENT_STATUS.json';STATE={'status':'waiting_existing_queue','scope':'one-factor-at-a-time epochs/count training comparison; original batch/length inference comparisons retained','conditions':[]};P.write_text(json.dumps(STATE,indent=2))
# Wait for the existing sequential queue plus its recovery supervisor; never overlap GPU work.
for pid in map(int,sys.argv[1:]):
 while True:
  try:os.kill(pid,0)
  except ProcessLookupError:break
  if (ROOT/'STOP_AFTER_CELLPOSE').exists():sys.exit(0)
  time.sleep(5)
for model,n in [('esm2',1024),('dnabert2',800),('ntv2',512),('rnafm',896),('openspliceai',96)]:
 if (ROOT/'STOP_AFTER_CELLPOSE').exists():break
 R=ROOT/model;run='factor_control_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/FACTOR_LATEST').write_text(run+'\n');source=ROOT/'scripts/real_training_v3/factor_training_worker.py';shutil.copy2(source,D/'executed_worker.py');shutil.copy2(__file__,D/'executed_controller.py');STATE.update(status='running',active_model=model,active_run=run)
 for e,nn in [(1,n),(2,n),(4,n),(2,n//4),(2,n//2)]:
  if (ROOT/'STOP_AFTER_CELLPOSE').exists():break
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU compute job exists'
  env={'RUN_ID':run,'MODEL_KIND':model,'BATCH':'8','TRAIN_MODE':'full','EPOCHS':str(e),'TRAIN_N':str(nn),'MPLCONFIGDIR':'/work/.cache/matplotlib','NUMBA_CACHE_DIR':'/work/.cache/numba','HF_HOME':'/work/.cache/huggingface','HF_MODULES_CACHE':'/work/.cache/modules','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
  if model=='esm2':
   env.update(MODEL_ROOT=str(R),SUPPORT_PATH=str(ROOT/'scripts'));cmd=['env']+[k+'='+v for k,v in env.items()]+[str(R/'.venv/bin/python'),'-u',str(source)]
  else:
   cmd=['docker','run','--rm','--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro'];python='/work/.venv/bin/python'
   if model=='dnabert2':cmd+=['-v','/home/hyein/dnabert2-benchmark:/workspace:ro'];python='/workspace/miniforge/envs/dnabert2/bin/python'
   for k,v in env.items():cmd+=['-e',k+'='+v]
   cmd+=['nvcr.io/nvidia/pytorch:26.06-py3',python,'-u','/support/real_training_v3/factor_training_worker.py']
  label=f'epochs{e}_n{nn}_b8';(D/(label+'_command.json')).write_text(json.dumps(cmd,indent=2));STATE['active_condition']=label;P.write_text(json.dumps(STATE,indent=2));print('START',model,label,flush=True);t=time.time()
  with (D/(label+'.log')).open('w') as f:code=subprocess.call(cmd if cmd[0]!='docker' else ['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  rec={'model':model,'run':run,'condition':label,'exit_code':code,'wall_seconds':time.time()-t};(D/(label+'_exit.json')).write_text(json.dumps(rec));STATE['conditions'].append(rec);P.write_text(json.dumps(STATE,indent=2));print('EXIT',model,label,code,flush=True)
  if code:STATE.setdefault('failed',[]).append(rec)
  subprocess.run([sys.executable,str(ROOT/'scripts/real_training_v3/factor_reports.py')],check=True)
STATE.update(status='finished_review_required',active_model=None,active_condition=None);P.write_text(json.dumps(STATE,indent=2));subprocess.run([sys.executable,str(ROOT/'scripts/real_training_v3/factor_reports.py')],check=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 요인 비교 실행·보고 원문: epochs1_n512_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/factor_control_20261102_182012_734207/report_snapshots/epochs1_n512_b8_command_50cb3c2f3d7dad30.json", "sha256": "50cb3c2f3d7dad30f3b956dad475e2d09bb801ee4373664bfd8f9c33210cf02b", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=factor_control_20261102_182012_734207",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "TRAIN_MODE=full",
  "-e",
  "EPOCHS=1",
  "-e",
  "TRAIN_N=512",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/factor_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 요인 비교 실행·보고 원문: epochs2_n512_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/factor_control_20261102_182012_734207/report_snapshots/epochs2_n512_b8_command_e886f1320f59f5fe.json", "sha256": "e886f1320f59f5fec8d74ee0070690400cda0b3cb74b5cf0e3fc76323cb100e7", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=factor_control_20261102_182012_734207",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "TRAIN_MODE=full",
  "-e",
  "EPOCHS=2",
  "-e",
  "TRAIN_N=512",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/factor_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 요인 비교 실행·보고 원문: epochs4_n512_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/factor_control_20261102_182012_734207/report_snapshots/epochs4_n512_b8_command_a53a35d86ac0e007.json", "sha256": "a53a35d86ac0e0073a9c937439c7fadf9ee9c788bca971222b7b09d9176964ca", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=factor_control_20261102_182012_734207",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "TRAIN_MODE=full",
  "-e",
  "EPOCHS=4",
  "-e",
  "TRAIN_N=512",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/factor_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 요인 비교 실행·보고 원문: epochs2_n128_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/factor_control_20261102_182012_734207/report_snapshots/epochs2_n128_b8_command_6eb6c43fe3d06529.json", "sha256": "6eb6c43fe3d0652994c444d0de1bfee25203564f5c900c338e329cc2640d98c6", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=factor_control_20261102_182012_734207",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "TRAIN_MODE=full",
  "-e",
  "EPOCHS=2",
  "-e",
  "TRAIN_N=128",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/factor_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 요인 비교 실행·보고 원문: epochs2_n256_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/factor_control_20261102_182012_734207/report_snapshots/epochs2_n256_b8_command_f0ffdc2c6b733d89.json", "sha256": "f0ffdc2c6b733d89852c5e86373496d490a81083d86cf4f266f68b3f61a6449e", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=factor_control_20261102_182012_734207",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "TRAIN_MODE=full",
  "-e",
  "EPOCHS=2",
  "-e",
  "TRAIN_N=256",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/factor_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 요인 비교 실행·보고 원문: executed_controller.py</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/factor_control_20261103_093231_014557/report_snapshots/executed_controller_55f68e6b36f806b3.py", "sha256": "55f68e6b36f806b344b1b0a58cbf7845e61c08acb17cd8a77dc4a045fff30a0f", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,json,time,datetime,subprocess,shlex,shutil
ROOT=Path('/home/hyein/ai-benchmark');P=ROOT/'FACTOR_EXPERIMENT_STATUS.json';STATE=json.loads(P.read_text()) if os.environ.get('RESUME_FACTORS')=='1' else {'scope':'one-factor-at-a-time epochs/count training comparison; original batch/length inference comparisons retained','conditions':[]};STATE['status']='waiting_existing_queue';P.write_text(json.dumps(STATE,indent=2))
# Wait for the existing sequential queue plus its recovery supervisor; never overlap GPU work.
for pid in map(int,sys.argv[1:]):
 while True:
  try:os.kill(pid,0)
  except ProcessLookupError:break
  if (ROOT/'STOP_AFTER_CELLPOSE').exists():sys.exit(0)
  time.sleep(5)
for model,n in [('esm2',1024),('dnabert2',800),('ntv2',512),('rnafm',896),('openspliceai',96)]:
 if (ROOT/'STOP_AFTER_CELLPOSE').exists():break
 expected=[f'epochs{e}_n{nn}_b8' for e,nn in [(1,n),(2,n),(4,n),(2,n//4),(2,n//2)]]
 completed={v['condition']:v for v in STATE.get('conditions',[]) if v['model']==model and v['exit_code']==0}
 if all(label in completed for label in expected):
  print('REUSE_MODEL',model,len(completed),flush=True);continue
 R=ROOT/model;run='factor_control_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/FACTOR_LATEST').write_text(run+'\n');source=ROOT/'scripts/real_training_v3/factor_training_worker.py';shutil.copy2(source,D/'executed_worker.py');shutil.copy2(__file__,D/'executed_controller.py');STATE.update(status='running',active_model=model,active_run=run)
 reused=[]
 for label,record in completed.items():
  match=__import__('re').fullmatch(r'epochs(\d+)_n(\d+)_b8',label)
  if not match:continue
  ee,nnn=map(int,match.groups());oldD=R/'results'/record['run'];folder=f'full_training_b8_e{ee}_n{nnn}';oldcase=oldD/folder;summary=json.loads((oldcase/'summary.json').read_text());assert summary['status']=='complete' and summary.get('saved_model_reload_verified')
  (D/folder).symlink_to(os.path.relpath(oldcase,D),target_is_directory=True);reused.append(record)
  oldcommand=oldD/(label+'_command.json')
  if oldcommand.exists():shutil.copy2(oldcommand,D/(label+'_reused_command.json'))
 (D/'reused_conditions.json').write_text(json.dumps(reused,indent=2))
 for e,nn in [(1,n),(2,n),(4,n),(2,n//4),(2,n//2)]:
  if (ROOT/'STOP_AFTER_CELLPOSE').exists():break
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU compute job exists'
  env={'RUN_ID':run,'MODEL_KIND':model,'BATCH':'8','TRAIN_MODE':'full','EPOCHS':str(e),'TRAIN_N':str(nn),'MPLCONFIGDIR':'/work/.cache/matplotlib','NUMBA_CACHE_DIR':'/work/.cache/numba','HF_HOME':'/work/.cache/huggingface','HF_MODULES_CACHE':'/work/.cache/modules','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
  if model=='esm2':
   env.update(MODEL_ROOT=str(R),SUPPORT_PATH=str(ROOT/'scripts'));cmd=['env']+[k+'='+v for k,v in env.items()]+[str(R/'.venv/bin/python'),'-u',str(source)]
  else:
   cmd=['docker','run','--rm','--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro'];python='/work/.venv/bin/python'
   if model=='dnabert2':cmd+=['-v','/home/hyein/dnabert2-benchmark:/workspace:ro'];python='/workspace/miniforge/envs/dnabert2/bin/python'
   for k,v in env.items():cmd+=['-e',k+'='+v]
   cmd+=['nvcr.io/nvidia/pytorch:26.06-py3',python,'-u','/support/real_training_v3/factor_training_worker.py']
  label=f'epochs{e}_n{nn}_b8'
  if label in completed:print('REUSE_CONDITION',model,label,flush=True);continue
  (D/(label+'_command.json')).write_text(json.dumps(cmd,indent=2));STATE['active_condition']=label;P.write_text(json.dumps(STATE,indent=2));print('START',model,label,flush=True);t=time.time()
  with (D/(label+'.log')).open('w') as f:code=subprocess.call(cmd if cmd[0]!='docker' else ['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  rec={'model':model,'run':run,'condition':label,'exit_code':code,'wall_seconds':time.time()-t};(D/(label+'_exit.json')).write_text(json.dumps(rec));STATE['conditions'].append(rec);P.write_text(json.dumps(STATE,indent=2));print('EXIT',model,label,code,flush=True)
  if code:STATE.setdefault('failed',[]).append(rec)
  subprocess.run([sys.executable,str(ROOT/'scripts/real_training_v3/factor_reports.py')],check=True)
STATE.update(status='finished_review_required',active_model=None,active_condition=None);P.write_text(json.dumps(STATE,indent=2));subprocess.run([sys.executable,str(ROOT/'scripts/real_training_v3/factor_reports.py')],check=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 요인 비교 실행·보고 원문: epochs2_n256_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "ntv2/results/factor_control_20261103_093231_014557/report_snapshots/epochs2_n256_b8_command_1194600779a53d13.json", "sha256": "1194600779a53d1375c7ef258a18bd6858f0177cd2cec6315bc8921bd9dfd852", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=factor_control_20261103_093231_014557",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=8",
  "-e",
  "TRAIN_MODE=full",
  "-e",
  "EPOCHS=2",
  "-e",
  "TRAIN_N=256",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/real_training_v3/factor_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>단계별 입력·출력 확인/보고 원문: trace_sequences_cpu.py</summary>

<!-- RESTORE_FILE {"path": "scripts/report_stages/source_snapshots/trace_sequences_cpu_c727a4a85ad9ed25.py", "sha256": "c727a4a85ad9ed252d7a4e94720c6605c17a918fe918bc1bee24cac387b2446a", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,json,csv,argparse,gc
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
R=Path('/work');kind=os.environ['MODEL_KIND'];d=R/'results'/(R/'results/FACTOR_LATEST').read_text().strip();cases=[p for p in d.glob('full_training_b8_e2_n*/summary.json') if json.loads(p.read_text()).get('status')=='complete'];q=max(cases,key=lambda p:json.loads(p.read_text())['training']['unique_training_cells']);C=q.parent;S0=json.loads(q.read_text());O=C/'stage_trace_cpu';O.mkdir(exist_ok=True);torch.set_num_threads(8);torch.manual_seed(42);ck=torch.load(C/'model.pt',map_location='cpu',weights_only=True);sample=ck['split_ids']['test'][0]
if kind=='dnabert2':
 from transformers import AutoModel,AutoTokenizer,BertConfig
 P=Path('/workspace/dnabert2_eval/model');tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);net=AutoModel.from_pretrained(P,config=BertConfig.from_pretrained(P),trust_remote_code=True,local_files_only=True);records=list(csv.DictReader(Path('/workspace/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv').open()));raw=next(r for r in records if r['id']==sample);head=torch.nn.Linear(net.config.hidden_size,18);y=np.zeros(18,np.float32);y[json.loads(raw['labels'])]=1
elif kind=='ntv2':
 from transformers import AutoModelForMaskedLM,AutoTokenizer
 P=R/'models/checkpoint';tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);base=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True);net=base.esm;del base;raw=next(r for r in json.loads((R/'data/inputs.json').read_text())['test'] if r['id']==sample);head=torch.nn.Linear(net.config.hidden_size,2);y=np.asarray([raw['label']],np.int64)
elif kind=='rnafm':
 import fm
 torch.serialization.add_safe_globals([argparse.Namespace]);net,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');raw=next(r for r in json.loads((R/'data/inputs.json').read_text())['test'] if r['id']==sample);head=torch.nn.Linear(640,28);y=np.asarray([raw['label']],np.int64)
elif kind=='openspliceai':
 from openspliceai.train_base.openspliceai import SpliceAI
 net=SpliceAI(32,np.array([11]*8+[21]*4+[41]*4),np.array([1]*4+[4]*4+[10]*4+[25]*4));head=torch.nn.Identity();raw=next(r for r in json.loads((R/'data/inputs.json').read_text())['test'] if r['id']==sample);y=np.asarray(raw['labels_10000'][2500:7500],np.int64)
net.load_state_dict(ck['backbone'],strict=True);head.load_state_dict(ck['head'],strict=True);del ck;gc.collect();net.float().eval();head.float().eval();seq=raw['sequence'];arrays={'target':y};S={'model':kind,'sample_id':sample,'source_case':str(C.relative_to(R)),'raw_record':raw,'training_scope':S0['training_scope'],'precision':'CPU FP32','provenance':'New CPU-only saved-model replay; intermediate tensors are not historical GPU benchmark captures'}
with torch.no_grad():
 if kind=='openspliceai':
  used=seq[2500:17500];tab=np.zeros((256,4),np.float32)
  for j,ch in enumerate('ACGT'):tab[ord(ch),j]=1
  x=torch.from_numpy(tab[np.frombuffer(used.encode(),np.uint8)].T[None]);box={}
  def capture(mod,args,out):box['feature']=out.detach().cpu()
  handle=net.initial_conv.register_forward_hook(capture);z=net(x);handle.remove();arrays.update(onehot=x.numpy(),initial_conv_features=box['feature'].numpy(),output_probabilities=z.numpy());S['used_sequence_length']=len(used);S['input_first12_bases']=used[:12];S['target_counts']={str(int(a)):int(b) for a,b in zip(*np.unique(y,return_counts=True))};S['positive_target_positions_first8']=np.flatnonzero(y>0)[:8].tolist();S['first5_position_probabilities']=z[0,:,:5].T.tolist();S['prediction_indices_first8']=z.argmax(1)[0,:8].tolist();S['feature_first8']=box['feature'][0,0,:8].tolist()
 else:
  width=S0['fixed_pool_padded_shapes']['input_ids'][1]
  if kind in ['dnabert2','ntv2']:x=tokenizer([seq],padding='max_length',max_length=width,truncation=False,return_tensors='pt');x={k:x[k] for k in ['input_ids','attention_mask']}
  else:
   _,_,t=alphabet.get_batch_converter()([(sample,seq)]);t=torch.nn.functional.pad(t,(0,width-t.shape[1]),value=alphabet.padding_idx);x={'input_ids':t,'attention_mask':t.ne(alphabet.padding_idx).long()}
  if kind=='dnabert2':h=net(**x)[0];mask=x['attention_mask'].unsqueeze(-1)
  elif kind=='ntv2':h=net(**x,return_dict=True).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1)
  else:
   h=net(x['input_ids'],repr_layers=[12],return_contacts=False)['representations'][12];mask=x['attention_mask'].clone();mask[:,0]=0;last=mask.sum(1);mask[torch.arange(len(mask)),last]=0;mask=mask.unsqueeze(-1)
  embed=(h*mask).sum(1)/mask.sum(1).clamp_min(1);z=head(embed);probs=z.sigmoid() if kind=='dnabert2' else z.softmax(-1);arrays.update({k:v.numpy() for k,v in x.items()});arrays.update(token_representations=h.numpy(),pool_mask=mask.numpy(),embedding=embed.numpy(),logits=z.numpy(),probabilities=probs.numpy());S.update(raw_nt=len(seq),token_ids_first16=x['input_ids'][0,:16].tolist(),valid_token_count=int(x['attention_mask'].sum()),feature_first8=h[0,0,:8].tolist(),embedding_first8=embed[0,:8].tolist(),pooling='masked mean includes valid special tokens' if kind in ['dnabert2','ntv2'] else 'mean of actual bases, BOS/EOS/PAD excluded',logits=z[0].tolist(),probabilities=probs[0].tolist(),target=y.tolist(),prediction=(probs[0]>.5).long().tolist() if kind=='dnabert2' else int(probs[0].argmax()))
ref=np.load(C/'test_outputs.npy')[0];delta=float(np.max(np.abs(z.numpy()[0]-ref)));S['saved_gpu_output_max_abs_difference']=delta;S['saved_gpu_output_matches']=bool(np.allclose(z.numpy()[0],ref,atol=5e-4,rtol=1e-4));S['arrays']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()};np.savez_compressed(O/'arrays.npz',**arrays);S['saved_arrays']=str((O/'arrays.npz').relative_to(R));(O/'trace.json').write_text(json.dumps(S,ensure_ascii=False,indent=2));print(kind,sample,'saved GPU match',S['saved_gpu_output_matches'],'max_abs',delta,flush=True)
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
