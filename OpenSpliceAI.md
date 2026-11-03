# OpenSpliceAI 최종 통합 보고서

## 1. 모델과 실제 입력

|항목|이번 실험에서 사용한 모델|
|---|---|
|체크포인트·버전|MANE10000nt,seed10–145개모델;파인튜닝은seed10|
|구조|팽창합성곱 residual CNN,10000nt 문맥|
|전체·구성별 파라미터 수|단일696,739개(0.697M);5모델합계3,483,695개(3.484M)|
|정밀도|FP32|
|모델 입력 → 출력|DNA4채널one-hot →위치별nonsplice/acceptor/donor 확률|
|가중치와 실행 메모리의 차이|파라미터 수×dtype bytes는 이론 가중치 용량이다. 실행 peak에는 입력·중간 활성값이 더해지고,파인튜닝에는 gradient/optimizer 상태도 추가된다.|
|메모리% 정의|CUDA allocated peak / CPU·GPU 공유 RAM121.6875GiB ×100; GPU-Util과 별개. AlphaFold2는 JAX 계측을 별도로 표시|

### 실제 데이터와 학습 방식 구분

|구분|데이터·실제 처리 내용|
|---|---|
|추론 데이터|기본:GRCh38실제156구간. 추가추론:고유비중첩15000nt window16000개,5모델ensemble|
|학습·검증·test 데이터|exon주석을확인한 GRCh38영역 train96/validation24/test36,염색체별분리. 원레코드20000nt/정답10000nt→실사용15000nt/정답5000nt|
|학습 입력 X / 정답 y|X:[B,4,15000] A/C/G/T one-hot. y:[B,5000]정수0/1/2;충돌위치-1무시. 구간전체에 정답1개가 아니라 위치별정답|
|가중치 갱신 방식|seed10 단일모델 전체파인튜닝. seed10–14의5모델ensemble은 추론용이며5개동시학습 아님|
|원본·변환 입력 저장 위치|`data/inputs.json의 sequence및labels_10000;source/openspliceai/variant/annotations/grch38.txt`|
|해석 범위|추가무라벨reference window를 임의정답으로 학습시키지 않음. 지도학습정답은exon주석에서 변환|

**용어:** 처음부터 학습=무작위 초기화에서 시작; 전체/일부 층 파인튜닝=공개 가중치에서 해당 부분 갱신; 분류기만 학습=본체를 동결하고 head만 갱신. **용량probe는 학습 방식이 아니라 짧은 자원 측정 목적**이며,실제로 수행한 가중치 갱신 방식과 함께 표기한다. 원본 전체 데이터 수와 해당 run에서 실제 사용한 부분집합 수를 구분한다.

### 모델 적재 직후 — 첫 입력 forward 이전

|적재 조건|allocated GiB(JAX는별도표기)|reserved GiB|프로세스 RSS GiB|원본 근거|
|---|---|---|---|---|
|추론/기본 모델 적재 기준|0.013392|0.013671875|0.8437080383300781|openspliceai/results/real_input_capacity_20261102_163800_779795/batch_1024/summary.json|
|파인튜닝 적재 기준|0.002678|0.00390625|0.6713600158691406|openspliceai/results/real_input_capacity_20261102_163800_779795/full_training_b8/summary.json|

위 값은 **가중치를 GPU에 적재하고 첫 입력을 계산하기 전의 프레임워크 할당량**이다. CUDA/JAX context·driver·시스템 전체 GPU사용량과 같지 않으며 CPU입력 준비 포함 여부는 해당 실행 코드를 따른다. **실행 peak와 모델 적재 기준값의 차이는 같은 run에서만 계산**한다. 파인튜닝 중 추가량에는 입력뿐 아니라 중간값·gradient·optimizer 상태가 포함된다. 서로 다른 조건/정밀도의 기준값을 하나의 모델 고유 메모리로 단정하지 않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

<!-- FACTOR_EFFECTS_START -->
### 시간·메모리에 영향을 주는 요인 — 통제 비교 상태

아래 원리는 모든 모델에서 정량 검증을 마쳤다는 뜻이 아니다. **실측 비교와 미실험을 구분**한다. 배치 sweep 중 정밀도·입력길이·체크포인트를 고정하며,다른 과제/다른head 결과를 같은 조건으로 섞지 않는다.

|요인|총시간·처리량 해석|최대 allocated 해석|이 모델의 실험 근거/상태|
|---|---|---|---|
|배치|한step은 길어질 수 있고step수는 줄어든다. 처리량 상승/포화는 실측으로 판단|같은입력길이·정밀도에서 동시중간값 증가가 주된영향|기존5모델추론batch확대;무라벨reference window확대 대기|
|샘플당 실제입력 크기|GPU에들어간실제token/타일 크기가 커졌을때 계산량 변화 비교|토큰길이·패치/타일크기·attention구조에 따라 증가;원본크기와 다를수있음|전체구간수와고정15000nt window분리,지도훈련96개별도|
|고유 데이터수|같은batch·epoch에서 총처리step가늘어 총시간이늘어나는지 비교|고정batch의동시입력이같으면 데이터총수만으로peak가크게늘지않을수있음|입력개수비교는본문참조;새지도학습데이터수비교는아래결과가생긴조건만완료|
|epoch|같은train입력을 반복하므로 누적시간 증가;검증성능개선은 보장안됨|동일batch·입력에서epoch자체가peak를선형증가시키는요인은아님|같은라벨train96의epoch1/2/4 및 train24/48/96 비교 대기|
|학습방식·갱신범위|추론/본체동결head학습/부분·전체미세조정/처음부터학습은 계산범위가다름|gradient·optimizer·역전파용중간값의유무가다름|1절의가중치갱신방식과X/y를참조;다른데이터·목적의시간을단순배수비교하지않음|
|모델·정밀도·전처리|구조·dtype·토큰화·타일화·컴파일·후처리도영향|모델적재기준값과실행중추가량을분리|원인후보/고정조건으로기록;모델크기나dtype의단독영향sweep은미실험|

추가 통제 비교에서는 **batch8·FP32·공개초기weight·seed42·optimizer·원래전체pool의padding폭·검증/test를고정**하고,(1)같은train의epoch1/2/4,(2)epoch2의중첩train1/4·1/2·전체를각각실행한다. 전체데이터를바꾸면서epoch도함께바꾼행끼리는단독효과로해석하지않는다. 단일실행이며데이터부분집합의label분포변화는정확도해석의제약이다. 이비교는해당지도학습지원모델에만적용한다.


추가epoch/데이터수통제비교 수치: **아직미실험/대기**. 기존측정을추정해서채우지않는다.

<!-- FACTOR_EFFECTS_END -->

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|배치 추론|고유1개; GRCh38 실제15000nt 구간,5모델ensemble|batch1, onehot:[1, 4, 15000],FP32|791회/총791건|15.02초|52.67입력/초|0.023 GiB / 0.02%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|배치 추론|고유156개; GRCh38 실제15000nt 구간,5모델ensemble|batch156, onehot:[156, 4, 15000],FP32|4회/총624건|16.36초|38.14입력/초|1.463 GiB / 1.20%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|단일seed10전체파인튜닝|train96/val24/test36;실제splice 위치라벨|12×4×15000,FP32|10epoch/80step|37.81초(학습구간합)|25.39구간/초|2.160 GiB / 1.78%|동일저장모델test지표본문;80%미도달|

<!-- REAL_TRAINING_V3_START -->
### 실제 데이터 추가 실험 — real_input_capacity_20261102_163800_779795

진행 중인 실행의 완료 조건만 아래에 갱신한다. probe는 짧은 용량 측정이며 전체 학습이나 평가 완료가 아니다. 메모리%는 CUDA allocated peak / 시스템 공유 RAM ×100이며 GPU-Util과 다르다. 독립 실행 1회 기준.

|조건|가중치 갱신 방식·실험 목적|상태|고유 입력|배치|epoch|optimizer steps|실측 시간(s)|입력/s|allocated peak GiB|reserved peak GiB|공유 RAM 대비 allocated %|GPU-Util 평균%|최대%|독립 test 지표|저장 모델 재로드|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|batch_1024|추론(가중치 갱신 없음)|complete|1024|1024|N/A|N/A|51.7259|39.5933|9.5169|9.6934|7.8207|95.0211|96.0000|N/A|N/A|
|batch_10240|추론(가중치 갱신 없음)|complete|10240|10240|N/A|N/A|513.7861|39.8609|94.9998|96.7168|78.0687|94.6189|96.0000|N/A|N/A|
|batch_2048|추론(가중치 갱신 없음)|complete|2048|2048|N/A|N/A|103.4740|39.5848|19.0114|19.3574|15.6231|94.5851|96.0000|N/A|N/A|
|batch_4096|추론(가중치 갱신 없음)|complete|4096|4096|N/A|N/A|207.5839|39.4636|38.0088|38.6973|31.2347|94.9840|96.0000|N/A|N/A|
|batch_512|추론(가중치 갱신 없음)|complete|512|512|N/A|N/A|25.8792|39.5685|4.7667|4.8535|3.9172|94.0833|96.0000|N/A|N/A|
|batch_6144|추론(가중치 갱신 없음)|complete|6144|6144|N/A|N/A|306.6353|40.0737|57.0054|58.0391|46.8457|95.3066|96.0000|N/A|N/A|
|batch_8|추론(가중치 갱신 없음)|complete|8|8|N/A|N/A|15.0901|46.6532|0.0876|0.1133|0.0720|93.7857|96.0000|N/A|N/A|
|batch_8192|추론(가중치 갱신 없음)|complete|8192|8192|N/A|N/A|408.8274|40.0756|76.0041|77.3789|62.4584|95.0319|96.0000|N/A|N/A|
|batch_9216|추론(가중치 갱신 없음)|complete|9216|9216|N/A|N/A|463.1459|39.7974|85.5069|87.0527|70.2676|94.8590|96.0000|N/A|N/A|
|full_training_b8|전체backbone·과제head 파인튜닝|complete|96|8|2|24|7.0011|27.4243|1.4403|1.5957|1.1836|81.3846|96.0000|{'acceptor_donor_AUPRC': [0.9073879441303617, 0.9367551297535762]}|True|
|full_training_b96|전체backbone·과제head 파인튜닝|complete|96|96|2|2|6.1369|31.2863|17.2170|17.2441|14.1485|80.3636|96.0000|{'acceptor_donor_AUPRC': [0.9158408798575189, 0.9405058999801762]}|True|
|probe_training_b32|전체backbone·과제head 파인튜닝 /용량측정|complete|96|32|0|3|3.3196|28.9188|5.7372|5.8926|4.7147|55.0000|96.0000|N/A|N/A|
|probe_training_b64|전체backbone·과제head 파인튜닝 /용량측정|complete|96|64|0|2|3.3248|28.8736|11.5405|11.5684|9.4837|57.5000|96.0000|N/A|N/A|
|probe_training_b8|전체backbone·과제head 파인튜닝 /용량측정|complete|24|8|0|3|1.3630|17.6084|1.4403|1.5957|1.1836|31.0000|93.0000|N/A|N/A|
|probe_training_b96|전체backbone·과제head 파인튜닝 /용량측정|complete|96|96|0|1|3.4699|27.6667|17.2117|17.2441|14.1442|63.6667|96.0000|N/A|N/A|

이 run의 추론 capacity 실험과 지도학습 실험은 입력 과제·길이·학습 헤드가 다를 수 있어 각각 비교한다. 3step probe는 전체 학습이 아니다. full_training은 동일한 공개 초기 backbone와 seed42,FP32,AdamW(lr1e-5,weight_decay0.01),고정 전체 split pool padding으로2epoch 수행한다. 각 학습 조건의 원본 summary에 split ID,실제 길이·padding 모양,사용 헤드·학습 범위,라벨,gradient가 관측된 파라미터 수,검증/test지표,저장 모델 재로드 차이를 기록한다. scVI는 아래 설명이 적용되지 않는 별도 처음부터 학습 모델이며 AF2·Clair3에는 이 공통 학습 절차를 적용하지 않는다.

추론은 GRCh38 실제 비중첩15000nt window16000개,기존5개 모델 앙상블이다. 학습은 기존 라벨이 검증된96/24/36개15000nt 입력→5000nt 정답,seed10단일 모델만 갱신한다. 비지도 reference window에 임의 정답을 만들지 않는다.

입력 및 메모리 기준값(같은 실행에서 측정):

|조건|실제 모델 입력 모양/정책|모델 적재 직후 allocated GiB|실행 시작 allocated GiB|peak 추가 allocated GiB|프로세스 RSS sampled peak GiB|시스템 가용 RAM sampled 최소 GiB|
|---|---|---|---|---|---|---|
|batch_1024|{'onehot': [1024, 4, 15000]}|0.0134|0.0134|9.5035|2.3017|102.7473|
|batch_10240|{'onehot': [10240, 4, 15000]}|0.0134|0.0134|94.9865|3.3022|13.3281|
|batch_2048|{'onehot': [2048, 4, 15000]}|0.0134|0.0134|18.9980|2.3827|92.8002|
|batch_4096|{'onehot': [4096, 4, 15000]}|0.0134|0.0134|37.9954|2.6008|72.9810|
|batch_512|{'onehot': [512, 4, 15000]}|0.0134|0.0134|4.7533|2.2420|107.5267|
|batch_6144|{'onehot': [6144, 4, 15000]}|0.0134|0.0134|56.9920|2.8699|53.0944|
|batch_8|{'onehot': [8, 4, 15000]}|0.0134|0.0134|0.0742|2.8079|110.9461|
|batch_8192|{'onehot': [8192, 4, 15000]}|0.0134|0.0134|75.9907|3.0888|33.2196|
|batch_9216|{'onehot': [9216, 4, 15000]}|0.0134|0.0134|85.4935|3.1929|23.2696|
|full_training_b8|{'onehot': [156, 4, 15000]}|0.0027|0.0027|1.4376|1.6271|111.1853|
|full_training_b96|{'onehot': [156, 4, 15000]}|0.0027|0.0027|17.2143|1.6716|95.3779|
|probe_training_b32|{'onehot': [156, 4, 15000]}|0.0027|0.0027|5.7345|1.6451|107.0048|
|probe_training_b64|{'onehot': [156, 4, 15000]}|0.0027|0.0027|11.5378|1.6543|101.1871|
|probe_training_b8|{'onehot': [156, 4, 15000]}|0.0027|0.0027|1.4376|1.6240|111.3810|
|probe_training_b96|{'onehot': [156, 4, 15000]}|0.0027|0.0027|17.2090|1.4922|95.4440|

실행 상태: `stopped`; 중단/관측 범위: `actual_worker_error_or_guard`.

실행 코드 블록의 파일은 원문SHA256으로 검증하며, 명령JSON에 기록된 실행 경로로 해당 snapshot을 복사하면 복원할 수 있다. 대용량 원본 데이터·가중치는 아래 전처리/다운로드 코드와 출처를 사용하고 로컬 원본도 보존한다.

로컬 원시 결과: `/home/hyein/ai-benchmark/openspliceai/results/real_input_capacity_20261102_163800_779795`. 메모리 부족/guard 실패는 성공 조건으로 기재하지 않는다.
<!-- REAL_TRAINING_V3_END -->

## 4. 결과 해석

<!-- MODEL_SUMMARY_START -->
### 요약 결론

- **학습 방식:** seed10 단일모델 전체파인튜닝. seed10–14의5모델ensemble은 추론용이며5개동시학습 아님.
- **입력 구분:** 추가무라벨reference window를 임의정답으로 학습시키지 않음. 지도학습정답은exon주석에서 변환.
- **최신 추가 run 최대 성공 메모리:** `batch_10240` **95.000GiB /78.07%**. 이 값은 해당 최신run의 성공 조건에서의 최대이며,다른 run의 최대와 구분한다. probe이면 전체 epoch·독립test 평가 완료를 뜻하지 않는다.
- **시간·정확도 해석:** 결과는 같은 행의 입력·배치·학습범위에만 연결한다. 메모리 상승 자체가 처리량·정확도 상승을 보장하지 않는다.
<!-- MODEL_SUMMARY_END -->

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### OpenSpliceAI — DGX Spark splice-site 추론·파인튜닝


run `openspliceai_20261027_205554_995618` · 상태 **complete** · summary (원본 기록 로컬 보관: `openspliceai/results/openspliceai_20261027_205554_995618/summary.json`) · 로그 (로컬 보관: `../results/openspliceai_20261027_205554_995618/benchmark.log`) · GPU·메모리 시계열 (원본 기록 로컬 보관: `openspliceai/results/openspliceai_20261027_205554_995618/resources.csv`)

#### 공식 모델과 환경

공식OpenSpliceAI0.0.7, source commit `369ec06283af59447195283a57d0384f4448ad03`. OSAI_MANE-10000nt의공식5개seed10–14 checkpoint를모두불러와순차ensemble했다. checkpoint경로·SHA-256은summary.checkpoint_sources에있다. TensorFlow/Keras원본SpliceAI와구분한다.

실제파라미터 {"per_model": 696739, "ensemble_total": 3483695, "trainable_single": 696739, "observed_gradient": 696739}

각모델32채널,16개residual unit,dilatedConv1d이며입력B×4×(출력길이+10000),출력B×3×출력길이다. 총context10000nt는양쪽5000nt씩제거되며출력3채널은non-splice/acceptor/donor확률이다. 입력크기비교에서context·가중치를고정하고중앙출력구간길이만변경했다.

환경 {"torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "gpu": "NVIDIA GB10"}

FP32,TF32=False,seed42,8CPUthreads. 공식predict/transfer소스의context10000설정은BATCH_SIZE=6*N_GPUS와N_GPUS=2상수여서실제batch12를그대로채택했다. 실험GPU는1개이며2개GPU를썼다는뜻이아니다. 파인튜닝은seed10단일모델의전파라미터,AdamW1e-4/weight_decay.01,CosineAnnealingWarmRestarts(T0=5,Tmult1,eta_min1e-5),10epochs다. 공식categorical_crossentropy_2d와같은확률log-loss를계산했다.

| 입력forward전 | allocatedGiB | reservedGiB | RSSGiB | 시스템가용GiB |
| --- | --- | --- | --- | --- |
| 5모델적재직후 | 0.0134 | 0.0176 | 0.7642 | 113.5768 |

이론FP32파라미터수×4byte와위실측은다르다. 적재 0.2750초,3배치warmup 1.2730초. CPU입력JSON은적재전에준비했다.

#### 실제 서열·splice-site 정답

기존검증된GRCh38.primary_assembly참조와공식패키지의grch38 transcript/exon표를사용했다. 표의SHA-256과전처리정책은 [input_audit.json](OpenSpliceAI.md#record-246beda1e980),실제서열·염기별라벨은 inputs.json (로컬 보관: `../data/inputs.json`)에있다. pretrained모델은MANE모델이지만이평가주석표가동일MANE원본이라고주장하지않는다. 주석기반정답이며새로운실험검증라벨은아니다.

| split | 구간수 | 허용염색체 | 원본입력nt | 보관라벨nt |
| --- | --- | --- | --- | --- |
| train | 96 | ['2', '4'] | 20000 | 10000 |
| validation | 24 | ['6'] | 20000 | 10000 |
| test | 36 | ['1', '3'] | 20000 | 10000 |

염색체단위로학습/검증/test를분리하고각구간20kb는같은염색체의다른선택구간과겹치지않게했다. 실제사용염색체는각inputs레코드에있으며위표는허용목록이다. 원래사전학습모델이인간유전체를봤으므로이번test는사전학습독립외부일반화시험이아니다.

plus strand donor는exon_end−1,acceptor는next_exon_start의exon염기좌표다. minus strand는역할을바꾸고서열·라벨을reverse-complement방향으로정렬했다. 같은염색체/strand의제공된모든주석site를라벨로합쳐인접유전자site누락을줄였다. 모호한좌표는−1로무시하며A/C/G/Tone-hot,N은모두0이다.

주석과좌표검증용실제motif:

| split | donor motif counts | acceptor motif counts |
| --- | --- | --- |
| train | {'GT': 286, 'CT': 1, 'AT': 2, 'GC': 1} | {'AG': 304, 'AC': 2} |
| validation | {'GT': 56, 'GC': 1} | {'AG': 63} |
| test | {'GT': 154, 'GC': 2, 'AT': 1} | {'AG': 149, 'AA': 1} |

각split의GTdonor/AGacceptor비율이75%를넘는지assert했고대부분canonicalmotif다. 입력한구간의정답행렬B×출력길이와모델출력B×3×출력길이일치를검사했다. 학습은원본20kb중중앙15kb입력→5kb출력,실제B×4×15000FP32와B×3×5000one-hot라벨을사용했다.

실제학습예 FAM110C_2_43532_- / - / 20kb중앞120nt:

```text
AGCGATCCTCCTGCCTCAGTCTCCCAAAGTGCTCAGATTAAGATATGAGCCATTGCACCCAGCCTAGGGTTTTGTCTCTAATTGCATTGAGTTCCTCTTTAAAAAAAATTTAAGACGATT
```

라벨수 {'0': 9998, '2': 1, '1': 1}

#### 입력별 GPU 사용률·메모리

| 조건 | 입력nt | 출력nt | 고유구간 | 반복 | 초 | 구간/초 | GPU평균/최대% | 시작GiB | peakGiB | 증가GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_output5000_n12 | 15000 | 5000 | 12 | 48 | 15.2936 | 37.6627 | 94.1034/95.0000 | 0.0134 | 0.1248 | 0.1114 |
| ensemble_output5000_n24 | 15000 | 5000 | 24 | 24 | 15.3384 | 37.5527 | 93.9655/95.0000 | 0.0134 | 0.1248 | 0.1114 |
| ensemble_output5000_n36 | 15000 | 5000 | 36 | 16 | 15.3107 | 37.6207 | 94.6552/95.0000 | 0.0134 | 0.1248 | 0.1114 |
| ensemble_output1000_n12 | 11000 | 1000 | 12 | 70 | 15.0066 | 55.9754 | 94.6429/95.0000 | 0.0134 | 0.0943 | 0.0809 |
| ensemble_output10000_n12 | 20000 | 10000 | 12 | 35 | 15.4010 | 27.2710 | 94.4138/95.0000 | 0.0134 | 0.1661 | 0.1527 |
| training_epoch1 | 15000 | 5000 | 96 | 1 | 3.9867 | 24.0803 | 75.6250/95.0000 | 0.0027 | 2.1603 | 2.1576 |
| training_epoch2 | 15000 | 5000 | 96 | 1 | 3.7874 | 25.3469 | 86.0000/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch3 | 15000 | 5000 | 96 | 1 | 3.7764 | 25.4212 | 77.2857/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch4 | 15000 | 5000 | 96 | 1 | 3.8076 | 25.2130 | 82.3750/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch5 | 15000 | 5000 | 96 | 1 | 3.7894 | 25.3340 | 81.2857/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch6 | 15000 | 5000 | 96 | 1 | 3.8006 | 25.2589 | 83.5714/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch7 | 15000 | 5000 | 96 | 1 | 3.7831 | 25.3763 | 88.7143/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch8 | 15000 | 5000 | 96 | 1 | 3.7392 | 25.6740 | 86.4286/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch9 | 15000 | 5000 | 96 | 1 | 3.6669 | 26.1801 | 92.4286/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch10 | 15000 | 5000 | 96 | 1 | 3.6711 | 26.1505 | 94.0000/95.0000 | 0.0106 | 2.1603 | 2.1496 |

추론5조건최소15초반복이며개수12/24/36은같은15000nt입력중첩subset이다. 크기비교는같은12구간에서입력11000/15000/20000nt(출력1000/5000/10000)를비교한다.5모델은같은입력으로순차forward후확률평균한다. epoch행은단일모델학습으로ensemble추론과조건이다르다.

시간에는메모리에준비된서열의one-hot/라벨구성·GPU전송·모델연산과추론확률평균을포함한다. 추론출력은GPUtensor이며CPU반환·디스크준비/저장·모델적재·warmup은제외했다. 학습은forward/loss/backward/optimizer까지이며epoch별validation은제외했다. CUDA synchronize와조건별peak reset을적용하고증가량은같은run의실제시작allocated대비다. gradient·optimizer·중간값도포함한다.

GPU-Util은0.5초대기+조회전체가조건안인장치활동표본으로FLOPS/메모리%가아니다. epoch가짧으면표본수한계가있어단독평균대표성이낮다. 공유RAM121.688GiB에서allocated와RSS를더하지않는다. allocated/reserved/RSS/가용량시계열은sampled값이며장치전체물리점유량과구분한다.

#### 주석 test·학습·저장·재로드

| 모델 | site | 양성수 | AUPRC(AP) | top-k recall |
| --- | --- | --- | --- | --- |
| 사전학습5ensemble | acceptor | 107 | 0.9121 | 0.8879 |
| 사전학습5ensemble | donor | 107 | 0.9537 | 0.9065 |
| 사전학습seed10 | acceptor | 107 | 0.9179 | 0.8879 |
| 사전학습seed10 | donor | 107 | 0.9410 | 0.8692 |
| 파인튜닝seed10 | acceptor | 107 | 0.9148 | 0.8785 |
| 파인튜닝seed10 | donor | 107 | 0.9472 | 0.8785 |

top-k는주석양성수k만큼가장높은score를뽑았을때정답비율이다. nonsplice가압도적으로많아일반accuracy보다donor/acceptorAUPRC를보고한다. 5모델ensemble과단일파인튜닝결과를동일조건향상으로주장하지않으며seed10전후를비교한다.

고정10epoch후test를평가했고test로epoch를선택하지않았다. 초기conv가중치최대변화 0.0032로실제갱신을확인했다. 저장 `models/openspliceai_20261027_205554_995618/model_seed10_finetuned.pt`,optimizer/scheduler/RNG/epoch와task.json도보관했다. 저장 0.0223초. 학습재개는검증하지않았다.

독립프로세스재로드 {"independent_process": true, "test_windows": 36, "input_shape": [12, 4, 15000], "max_abs_difference": 0.0, "all_base_classes_match": true, "passed": true}

#### 한계·공식 출처

작은염색체별주석구간에대한동작·학습·계측검증이다. curatedannotation의완전성및사전학습중복/동족성이한계이며전체인간유전체평가를대체하지않는다. 이번과제는splice-site예측이며변이delta scoreCLI평가는수행하지않았다.

- [공식코드·모델](https://github.com/Kuanhao-Chao/OpenSpliceAI)
- [공식transfer예제](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/transfer/transfer_example.sh)
- [공식모델구조](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/openspliceai/train_base/openspliceai.py)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### openspliceai 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_144737_167765` · 종료사유 `available_unique_input_limit`

#### 입력과 고정 조건

{
  "source": "actual GRCh38 curated regions; input15000nt -> output5000nt; original five-model MANE10000 ensemble",
  "unique_available_inputs": 156,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

#### 배치별 측정

측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.

|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|791|791|15.0172|52.6729|0.0134|0.0227|0.0093|0.0352|0.0186|92.3103|96.0|1.0836|112.8657|
|2|378|756|15.0195|50.3345|0.0134|0.0319|0.0186|0.0352|0.0263|92.1071|94.0|1.0863|112.8673|
|4|179|716|15.0013|47.7292|0.0134|0.0505|0.0371|0.0762|0.0415|90.1429|96.0|1.0791|112.8994|
|8|83|664|15.0214|44.2037|0.0134|0.0876|0.0742|0.1133|0.072|93.3571|96.0|1.0862|112.8627|
|16|36|576|15.0106|38.3728|0.0134|0.1652|0.1518|0.1836|0.1358|91.9286|96.0|1.0865|112.8381|
|32|19|608|15.8191|38.4346|0.0134|0.3102|0.2968|0.3262|0.2549|94.9667|96.0|1.091|112.5204|
|64|10|640|16.6657|38.4022|0.0134|0.611|0.5976|0.625|0.5021|93.2258|96.0|1.0958|112.3472|
|96|7|672|17.3643|38.7|0.0134|0.905|0.8916|0.9355|0.7437|88.5758|96.0|1.0964|111.9866|
|128|5|640|16.6323|38.4793|0.0134|1.2021|1.1887|1.2441|0.9878|93.3226|96.0|1.1011|111.6641|
|156|4|624|16.3604|38.1408|0.0134|1.4628|1.4494|1.5|1.2021|95.6452|96.0|1.1071|111.4199|

#### 결과 해석과 검증

최대시험성공batch 156. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://github.com/Kuanhao-Chao/OpenSpliceAI) · [기존 모델 보고서 디렉터리](OpenSpliceAI.md)

### 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.


## 6. 실제 실행 코드·명령어 전문

<!-- EXECUTED_SCRIPTS_START -->

### 원문과 출처 검증

아래 코드 블록은 원본 내용을 생략하거나 재작성하지 않았다. 각 파일의 원래 경로·원문 SHA256·출처 상태를 함께 기록했다. `실행 run 저장 snapshot`을 우선하고, 현재 소스만 남은 경우에는 당시 파일과 같다고 단정하지 않는다. 동일 바이트 버전은 한 번만 수록하고 해당 경로들을 함께 표시한다.

GitHub에는 `.py`/`.sh`를 별도 파일로 두지 않는다. 코드 블록을 원래 경로에 저장하거나 [복원 안내](OpenSpliceAI.md)의 코드로 로컬에서 복원한다. 라이브러리 전체 소스는 이 사용자 실행 스크립트와 구분하며 패키지 버전/출처를 따른다.

### 코드 목록

|원래 파일|원문 SHA256|출처 상태|
|---|---|---|
|`openspliceai/scripts/benchmark.py`|`35c95257b0d5eeb42fcd31019f1dc360e3b0c745535035074fa7e050f0107861`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`openspliceai/scripts/prepare.py`|`e38ffac7abb52de06b6439d9a614c9c02d1bbbead1255611528fad8b6f38327f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`openspliceai/scripts/reload.py`|`215a839e220cff1598db44d3078c65f2d5998947d281d809d851550184940401`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`openspliceai/scripts/report.py`|`468acc5b8012f133bbc1421c28ca7d6f003db0153bc44ff3838a33be75a3d8be`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`openspliceai/scripts/setup.py`|`34c5931018bf7c317205782f0379ef7592b9458706a69397443e186ede95ae48`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|
|`scripts/remaining_followups/launch.py`|`84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/report.py`|`8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/worker.py`|`e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|

<details>
<summary>openspliceai/scripts/benchmark.py — 원문 전문 (7,294 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/benchmark.py", "sha256": "35c95257b0d5eeb42fcd31019f1dc360e3b0c745535035074fa7e050f0107861", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,os,time,json,random,gc,hashlib
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import torch,numpy as np
from sklearn.metrics import average_precision_score
from openspliceai.train_base.openspliceai import SpliceAI
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'source/models/openspliceai-mane/10000nt';W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'checkpoint':'official OSAI_MANE10000 seeds10..14 five-model ensemble','batch':12,'basis':'official predict/transfer CL10000 code BATCH_SIZE=6*N_GPUS with hardcoded N_GPUS2; run uses single GPU with batch12, not two GPUs','dtype':'FP32','tf32':False,'seed':42,'context_nt':10000,'normal_output_nt':5000,'fine_tune_checkpoint':'seed10 only, compare single-seed10 baseline','fine_tune_epochs':10,'lr':1e-4,'optimizer':'AdamW default weight_decay.01; cosine restart T0=5,Tmult1,eta_min1e-5','train_n':96,'validation_n':24,'test_n':36,'fine_tune':'all parameters, curated packaged GRCh38 annotation'} }
def save():write_json(D/'summary.json',S)
def crop(r,sl):return r['sequence'][10000-(sl+10000)//2:10000+(sl+10000)//2],np.array(r['labels_10000'][5000-sl//2:5000+sl//2],dtype=np.int64)
def encode(rows,sl):
 seqs=[crop(r,sl)[0] for r in rows];tab=np.zeros((256,4),dtype=np.float32)
 for j,c in enumerate('ACGT'):tab[ord(c),j]=1
 x=np.stack([tab[np.frombuffer(seq.encode(),dtype=np.uint8)].T for seq in seqs]);y=np.stack([crop(r,sl)[1] for r in rows]);return torch.from_numpy(x).cuda(),y
try:
 data=json.loads((R/'data/inputs.json').read_text());meter.phase='loading';t=time.time();models=[];sources=[]
 for seed in range(10,15):
  p=P/f'model_10000nt_rs{seed}.pt';m=SpliceAI(32,W,AR).float().cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True));m.eval();models.append(m);sources.append({'checkpoint':str(p.relative_to(R)),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
 torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'per_model':sum(p.numel() for p in models[0].parameters()),'ensemble_total':sum(p.numel() for m in models for p in m.parameters()),'trainable_single':sum(p.numel() for p in models[0].parameters() if p.requires_grad)};S['checkpoint_sources']=sources;save()
 def predict(ms,rows,sl):
  x,y=encode(rows,sl);out=None
  for m in ms:
   z=m(x);out=z if out is None else out+z
  return out/len(ms),y
 meter.phase='warmup';t=time.time()
 with torch.no_grad():
  for _ in range(3):z,y=predict(models,data['test'][:12],5000);del z,y
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t
 for sl,n in [(5000,12),(5000,24),(5000,36),(1000,12),(10000,12)]:
  name=f'ensemble_output{sl}_n{n}';t,base=meter.begin(name);repeats=0
  with torch.no_grad():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,12):z,y=predict(models,data['test'][i:i+12],sl);assert z.shape==(min(12,n-i),3,sl);del z,y
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='ensemble_inference',bp=sl+10000,output_nt=sl,unique_inputs=n,repeats=repeats,processed=n*repeats);row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',name,row['seconds'],row['memory']['peak_allocated_gib'],flush=True);save()
 def score(ms,rows):
  arrays=[];labels=[]
  for m in ms:m.eval()
  with torch.no_grad():
   for i in range(0,len(rows),12):z,y=predict(ms,rows[i:i+12],5000);arrays.append(z.cpu().numpy());labels.append(y)
  z=np.concatenate(arrays);y=np.concatenate(labels);res={}
  for k,name in [(1,'acceptor'),(2,'donor')]:
   valid=y>=0;true=(y==k)[valid];probs=z[:,k,:][valid];num=int(true.sum());idx=np.argpartition(probs,-num)[-num:] if num else [];res[name]={'positive_sites':num,'auprc':float(average_precision_score(true,probs)),'top_k_recall':float(true[idx].mean()) if num else None}
  return res,z
 meter.phase='initial_test';S['ensemble_test_before'],_=score(models,data['test']);S['single_seed10_test_before'],_=score(models[:1],data['test']);del models,m;gc.collect();torch.cuda.empty_cache();meter.phase='classifier_loading';model=SpliceAI(32,W,AR).float().cuda();model.load_state_dict(torch.load(P/'model_10000nt_rs10.pt',map_location='cuda',weights_only=True));S['single_model_only']=meter.memory();before=model.initial_conv.weight.detach().cpu().clone();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=5,T_mult=1,eta_min=1e-5);history=[];grad=None
 for epoch in range(10):
  model.train();order=np.random.default_rng(42+epoch).permutation(96);t,base=meter.begin(f'training_epoch{epoch+1}');losses=[]
  for step in range(0,96,12):
   rows=[data['train'][int(i)] for i in order[step:step+12]];x,y=encode(rows,5000);yy=torch.from_numpy(y).cuda();valid=yy>=0;labels=torch.nn.functional.one_hot(yy.clamp_min(0),3).permute(0,2,1).float()*valid.unsqueeze(1);opt.zero_grad(set_to_none=True);z=model(x);loss=-(labels*torch.log(z+1e-10)).sum(1).mean();assert torch.isfinite(loss);loss.backward()
   if grad is None:grad=sum(p.numel() for p in model.parameters() if p.grad is not None)
   opt.step();losses.append(float(loss.detach()));del x,yy,labels,z,loss
  sched.step(epoch+1);row=meter.end(t,base,name=f'training_epoch{epoch+1}',mode='single_model_training',bp=15000,output_nt=5000,unique_inputs=96,repeats=1,processed=96,steps=8,loss_mean=float(np.mean(losses)));row['throughput']=96/row['seconds'];S['conditions'].append(row);meter.phase='validation';val,_=score([model],data['validation']);history.append({'epoch':epoch+1,'loss':row['loss_mean'],'validation':val});print('EPOCH',epoch+1,row['loss_mean'],val,flush=True);S['epochs']=history;save()
 S['parameters']['observed_gradient']=grad;S['weight_change_max_abs']=float((model.initial_conv.weight.detach().cpu()-before).abs().max());assert S['weight_change_max_abs']>0;meter.phase='saving';t=time.time();model.eval();torch.save(model.state_dict(),M/'model_seed10_finetuned.pt');torch.save({'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':10,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},M/'training_state.pt');write_json(M/'task.json',{'L':32,'W':W.tolist(),'AR':AR.tolist(),'context':10000,'output':5000,'input_channels':['A','C','G','T'],'labels':['nonsplice','acceptor','donor'],'annotation':'source/openspliceai/variant/annotations/grch38.txt','resume_tested':False});S['save_seconds']=time.time()-t;S['saved_model']=str((M/'model_seed10_finetuned.pt').relative_to(R));meter.phase='final_test';S['finetuned_test'],z=score([model],data['test']);np.save(D/'test_probabilities.npy',z);write_json(D/'test_window_ids.json',[r['id'] for r in data['test']]);S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>openspliceai/scripts/prepare.py — 원문 전문 (3,950 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/prepare.py", "sha256": "e38ffac7abb52de06b6439d9a614c9c02d1bbbead1255611528fad8b6f38327f", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import csv,json,hashlib,bisect,collections
import pysam,numpy as np
R=Path('/work');fa=pysam.FastaFile('/reference/GRCh38.primary_assembly.genome.fa');annotation=R/'source/openspliceai/variant/annotations/grch38.txt';rows=list(csv.DictReader(annotation.open(),delimiter='\t'));sites=collections.defaultdict(lambda:{1:set(),2:set()})
for row in rows:
 starts=[int(v) for v in row['EXON_START'].split(',') if v];ends=[int(v) for v in row['EXON_END'].split(',') if v];assert len(starts)==len(ends)
 if len(starts)<2:continue
 key=(row['CHROM'],row['STRAND']);donors=[e-1 for e in ends[:-1]];acceptors=starts[1:]
 if row['STRAND']=='-':donors,acceptors=acceptors,donors
 sites[key][2].update(donors);sites[key][1].update(acceptors)
sites={k:{t:sorted(v) for t,v in x.items()} for k,x in sites.items()};out={};audit={'reference':'GRCh38.primary_assembly.genome.fa (existing BEND reference)','annotation':str(annotation.relative_to(R)),'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),'coordinate_policy':'genomic0based; plus donor exon_end-1, acceptor next_exon_start; minus swapped then reverse-complement; labeled exon bases per official create-data motif convention','source_checkpoint':'official OSAI_MANE10000 ensemble seeds10..14; annotation here is bundled grch38 transcript table, not asserted identical MANE training annotation','splits':{}}
for split,chroms,n in [('train',['2','4'],96),('validation',['6'],24),('test',['1','3'],36)]:
 selected=[];lastend={};seen=set()
 for row in rows:
  chrom=row['CHROM'];strand=row['STRAND'];key=(chrom,strand)
  if chrom not in chroms or key not in sites or row['#NAME'] in seen:continue
  starts=[int(v) for v in row['EXON_START'].split(',') if v];ends=[int(v) for v in row['EXON_END'].split(',') if v]
  if len(starts)<2:continue
  center=(ends[0]-1+starts[1])//2;start=center-10000;end=center+10000;ctg='chr'+chrom
  if start<0 or end>fa.get_reference_length(ctg) or start<lastend.get(ctg,-1):continue
  seq=fa.fetch(ctg,start,end).upper()
  if not set(seq)<=set('ACGTN') or seq.count('N')>100:continue
  labels=np.zeros(10000,dtype=np.int64);out_start=center-5000;out_end=center+5000
  for kind,coords in sites[key].items():
   for pos in coords[bisect.bisect_left(coords,out_start):bisect.bisect_left(coords,out_end)]:
    idx=pos-out_start if strand=='+' else out_end-1-pos
    if labels[idx] not in [0,kind]:labels[idx]=-1
    else:labels[idx]=kind
  if not np.any(labels==1) or not np.any(labels==2):continue
  if strand=='-':seq=seq.translate(str.maketrans('ACGTN','TGCAN'))[::-1]
  rec={'id':row['#NAME']+'_'+chrom+'_'+str(center)+'_'+strand,'gene':row['#NAME'],'chromosome':ctg,'strand':strand,'genomic_input_start':start,'genomic_input_end':end,'sequence':seq,'labels_10000':labels.tolist(),'label_counts':dict(collections.Counter(map(int,labels)))};selected.append(rec);seen.add(row['#NAME']);lastend[ctg]=end
  if len(selected)==n:break
 assert len(selected)==n,(split,len(selected));out[split]=selected;motifs=collections.defaultdict(collections.Counter)
 for r in selected:
  seq=r['sequence'][5000:-5000]
  for i,v in enumerate(r['labels_10000']):
   if v==2:motifs['donor'][seq[i+1:i+3]]+=1
   if v==1:motifs['acceptor'][seq[max(0,i-2):i]]+=1
 audit['splits'][split]={'n':len(selected),'chromosomes':chroms,'input_nt':20000,'output_label_nt':10000,'motifs':{k:dict(v) for k,v in motifs.items()},'first_example':{k:v for k,v in selected[0].items() if k not in ['sequence','labels_10000']}}
 for kind,canonical in [('donor','GT'),('acceptor','AG')]:assert motifs[kind][canonical]/sum(motifs[kind].values())>.75,(split,kind,motifs[kind])
allrows=[r for rs in out.values() for r in rs];assert len({hashlib.sha256(r['sequence'].encode()).hexdigest() for r in allrows})==len(allrows);(R/'data/inputs.json').write_text(json.dumps(out));(R/'data/input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit['splits'],indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>openspliceai/scripts/reload.py — 원문 전문 (1,449 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/reload.py", "sha256": "215a839e220cff1598db44d3078c65f2d5998947d281d809d851550184940401", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,json
sys.path.insert(0,'/support')
from gpu_support import write_json
import numpy as np,torch
from openspliceai.train_base.openspliceai import SpliceAI
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);m=SpliceAI(32,W,AR).float().cuda().eval();m.load_state_dict(torch.load(R/s['saved_model'],map_location='cuda',weights_only=True));data=json.loads((R/'data/inputs.json').read_text())['test'];tab=np.zeros((256,4),np.float32)
for j,c in enumerate('ACGT'):tab[ord(c),j]=1
pred=[]
with torch.no_grad():
 for i in range(0,len(data),12):
  x=np.stack([tab[np.frombuffer(r['sequence'][2500:17500].encode(),np.uint8)].T for r in data[i:i+12]]);pred.append(m(torch.from_numpy(x).cuda()).cpu().numpy())
y=np.concatenate(pred);ref=np.load(D/'test_probabilities.npy');r={'independent_process':True,'test_windows':len(data),'input_shape':[12,4,15000],'max_abs_difference':float(np.abs(y-ref).max()),'all_base_classes_match':bool(np.array_equal(y.argmax(1),ref.argmax(1)))};r['passed']=r['max_abs_difference']<=1e-5 and r['all_base_classes_match'];write_json(D/'reload.json',r);assert r['passed'];s['reload']=r;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(r),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>openspliceai/scripts/report.py — 원문 전문 (8,518 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/report.py", "sha256": "468acc5b8012f133bbc1421c28ca7d6f003db0153bc44ff3838a33be75a3d8be", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import json,csv
R=Path('/home/hyein/ai-benchmark/openspliceai');run=(R/'results/LATEST').read_text().strip();D=R/'results'/run;s=json.loads((D/'summary.json').read_text());audit=json.loads((R/'data/input_audit.json').read_text());out=[]
def a(t=''):out.append(t)
def table(h,rows):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rows:a('| '+' | '.join(map(str,r))+' |')
 a()
def f(v):return 'N/A' if v is None else f'{v:.4f}'
a('# OpenSpliceAI — DGX Spark splice-site 추론·파인튜닝');a();a(f'run `{run}` · 상태 **{s["status"]}** · [summary](../results/{run}/summary.json) · [로그](../results/{run}/benchmark.log) · [GPU·메모리 시계열](../results/{run}/resources.csv)');a();a('## 공식 모델과 환경');a();a('공식OpenSpliceAI0.0.7, source commit `369ec06283af59447195283a57d0384f4448ad03`. OSAI_MANE-10000nt의공식5개seed10–14 checkpoint를모두불러와순차ensemble했다. checkpoint경로·SHA-256은summary.checkpoint_sources에있다. TensorFlow/Keras원본SpliceAI와구분한다.');a();a('실제파라미터 '+json.dumps(s.get('parameters',{})));a();a('각모델32채널,16개residual unit,dilatedConv1d이며입력B×4×(출력길이+10000),출력B×3×출력길이다. 총context10000nt는양쪽5000nt씩제거되며출력3채널은non-splice/acceptor/donor확률이다. 입력크기비교에서context·가중치를고정하고중앙출력구간길이만변경했다.');a();a('환경 '+json.dumps(s['environment']));a();a('FP32,TF32=False,seed42,8CPUthreads. 공식predict/transfer소스의context10000설정은BATCH_SIZE=6*N_GPUS와N_GPUS=2상수여서실제batch12를그대로채택했다. 실험GPU는1개이며2개GPU를썼다는뜻이아니다. 파인튜닝은seed10단일모델의전파라미터,AdamW1e-4/weight_decay.01,CosineAnnealingWarmRestarts(T0=5,Tmult1,eta_min1e-5),10epochs다. 공식categorical_crossentropy_2d와같은확률log-loss를계산했다.');a();b=s['model_only_postload'];table(['입력forward전','allocatedGiB','reservedGiB','RSSGiB','시스템가용GiB'],[['5모델적재직후',*[f(b[k]) for k in ['allocated_gib','reserved_gib','rss_gib','system_available_gib']]]]);a('이론FP32파라미터수×4byte와위실측은다르다. 적재 '+f(s['load_seconds'])+'초,3배치warmup '+f(s.get('warmup_seconds'))+'초. CPU입력JSON은적재전에준비했다.');a();a('## 실제 서열·splice-site 정답');a();a('기존검증된GRCh38.primary_assembly참조와공식패키지의grch38 transcript/exon표를사용했다. 표의SHA-256과전처리정책은 [input_audit.json](../data/input_audit.json),실제서열·염기별라벨은 [inputs.json](../data/inputs.json)에있다. pretrained모델은MANE모델이지만이평가주석표가동일MANE원본이라고주장하지않는다. 주석기반정답이며새로운실험검증라벨은아니다.');a();table(['split','구간수','허용염색체','원본입력nt','보관라벨nt'],[[k,v['n'],v['chromosomes'],v['input_nt'],v['output_label_nt']] for k,v in audit['splits'].items()]);a('염색체단위로학습/검증/test를분리하고각구간20kb는같은염색체의다른선택구간과겹치지않게했다. 실제사용염색체는각inputs레코드에있으며위표는허용목록이다. 원래사전학습모델이인간유전체를봤으므로이번test는사전학습독립외부일반화시험이아니다.');a();a('plus strand donor는exon_end−1,acceptor는next_exon_start의exon염기좌표다. minus strand는역할을바꾸고서열·라벨을reverse-complement방향으로정렬했다. 같은염색체/strand의제공된모든주석site를라벨로합쳐인접유전자site누락을줄였다. 모호한좌표는−1로무시하며A/C/G/Tone-hot,N은모두0이다.');a();a('주석과좌표검증용실제motif:');a();table(['split','donor motif counts','acceptor motif counts'],[[k,v['motifs']['donor'],v['motifs']['acceptor']] for k,v in audit['splits'].items()]);a('각split의GTdonor/AGacceptor비율이75%를넘는지assert했고대부분canonicalmotif다. 입력한구간의정답행렬B×출력길이와모델출력B×3×출력길이일치를검사했다. 학습은원본20kb중중앙15kb입력→5kb출력,실제B×4×15000FP32와B×3×5000one-hot라벨을사용했다.');a();ex=json.loads((R/'data/inputs.json').read_text())['train'][0];a('실제학습예 '+ex['id']+' / '+ex['strand']+' / 20kb중앞120nt:');a();a('```text\n'+ex['sequence'][:120]+'\n```');a();a('라벨수 '+str(ex['label_counts']));a();a('## 입력별 GPU 사용률·메모리');a();rows=[];flat=[]
for r in s['conditions']:
 rows.append([r['name'],r['bp'],r['output_nt'],r['unique_inputs'],r['repeats'],f(r['seconds']),f(r['throughput']),f(r['gpu_util_avg'])+'/'+f(r['gpu_util_max']),f(r['baseline']['allocated_gib']),f(r['memory']['peak_allocated_gib']),f(r['peak_delta_gib'])]);d={k:v for k,v in r.items() if k not in ['baseline','memory']};d.update(baseline_allocated_gib=r['baseline']['allocated_gib'],peak_allocated_gib=r['memory']['peak_allocated_gib'],peak_reserved_gib=r['memory']['peak_reserved_gib']);flat.append(d)
table(['조건','입력nt','출력nt','고유구간','반복','초','구간/초','GPU평균/최대%','시작GiB','peakGiB','증가GiB'],rows)
if flat:
 keys=list(dict.fromkeys(k for r in flat for k in r))
 with (R/'reports/comparison.csv').open('w',newline='') as f0:w=csv.DictWriter(f0,fieldnames=keys);w.writeheader();w.writerows(flat)
a('추론5조건최소15초반복이며개수12/24/36은같은15000nt입력중첩subset이다. 크기비교는같은12구간에서입력11000/15000/20000nt(출력1000/5000/10000)를비교한다.5모델은같은입력으로순차forward후확률평균한다. epoch행은단일모델학습으로ensemble추론과조건이다르다.');a();a('시간에는메모리에준비된서열의one-hot/라벨구성·GPU전송·모델연산과추론확률평균을포함한다. 추론출력은GPUtensor이며CPU반환·디스크준비/저장·모델적재·warmup은제외했다. 학습은forward/loss/backward/optimizer까지이며epoch별validation은제외했다. CUDA synchronize와조건별peak reset을적용하고증가량은같은run의실제시작allocated대비다. gradient·optimizer·중간값도포함한다.');a();a('GPU-Util은0.5초대기+조회전체가조건안인장치활동표본으로FLOPS/메모리%가아니다. epoch가짧으면표본수한계가있어단독평균대표성이낮다. 공유RAM121.688GiB에서allocated와RSS를더하지않는다. allocated/reserved/RSS/가용량시계열은sampled값이며장치전체물리점유량과구분한다.');a()
if 'finetuned_test' in s:
 a('## 주석 test·학습·저장·재로드');a();rows=[]
 for name,key in [('사전학습5ensemble','ensemble_test_before'),('사전학습seed10','single_seed10_test_before'),('파인튜닝seed10','finetuned_test')]:
  for kind,v in s[key].items():rows.append([name,kind,v['positive_sites'],f(v['auprc']),f(v['top_k_recall'])])
 table(['모델','site','양성수','AUPRC(AP)','top-k recall'],rows);a('top-k는주석양성수k만큼가장높은score를뽑았을때정답비율이다. nonsplice가압도적으로많아일반accuracy보다donor/acceptorAUPRC를보고한다. 5모델ensemble과단일파인튜닝결과를동일조건향상으로주장하지않으며seed10전후를비교한다.');a();a('고정10epoch후test를평가했고test로epoch를선택하지않았다. 초기conv가중치최대변화 '+f(s['weight_change_max_abs'])+'로실제갱신을확인했다. 저장 `'+s['saved_model']+'`,optimizer/scheduler/RNG/epoch와task.json도보관했다. 저장 '+f(s['save_seconds'])+'초. 학습재개는검증하지않았다.');a();a('독립프로세스재로드 '+json.dumps(s.get('reload',{'status':'pending'})));a()
a('## 한계·공식 출처');a();a('작은염색체별주석구간에대한동작·학습·계측검증이다. curatedannotation의완전성및사전학습중복/동족성이한계이며전체인간유전체평가를대체하지않는다. 이번과제는splice-site예측이며변이delta scoreCLI평가는수행하지않았다.');a();a('- [공식코드·모델](https://github.com/Kuanhao-Chao/OpenSpliceAI)\n- [공식transfer예제](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/transfer/transfer_example.sh)\n- [공식모델구조](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/openspliceai/train_base/openspliceai.py)');(R/'reports/OpenSpliceAI.md').write_text('\n'.join(out)+'\n');print(R/'reports/OpenSpliceAI.md')
````
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>openspliceai/scripts/setup.py — 원문 전문 (534 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/setup.py", "sha256": "34c5931018bf7c317205782f0379ef7592b9458706a69397443e186ede95ae48", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,subprocess,importlib.metadata,json
from pathlib import Path
R=Path('/work');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True);subprocess.run(['/work/.venv/bin/pip','install','--no-deps','/work/source'],check=True);(R/'constraints.txt').write_text('torch=='+importlib.metadata.version('torch')+'\n');subprocess.run(['/work/.venv/bin/pip','install','-c','/work/constraints.txt','--cache-dir','/work/.cache/pip','scikit-learn','pysam','gffutils','pyfaidx','biopython'],check=True)
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

### record-246beda1e980

원래 파일: `openspliceai/data/input_audit.json` · 원문 SHA256: `967ab007ff41bf8c6f7796afd752b6e246ce9c12dc6b84634ca961f2f205fb23`

<details>
<summary>openspliceai/data/input_audit.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/data/input_audit.json", "sha256": "967ab007ff41bf8c6f7796afd752b6e246ce9c12dc6b84634ca961f2f205fb23", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "reference": "GRCh38.primary_assembly.genome.fa (existing BEND reference)",
  "annotation": "source/openspliceai/variant/annotations/grch38.txt",
  "annotation_sha256": "565c0ea83f66b2210182364c3d626fef526229e3d02f444bab247329b6695f2c",
  "coordinate_policy": "genomic0based; plus donor exon_end-1, acceptor next_exon_start; minus swapped then reverse-complement; labeled exon bases per official create-data motif convention",
  "source_checkpoint": "official OSAI_MANE10000 ensemble seeds10..14; annotation here is bundled grch38 transcript table, not asserted identical MANE training annotation",
  "splits": {
    "train": {
      "n": 96,
      "chromosomes": [
        "2",
        "4"
      ],
      "input_nt": 20000,
      "output_label_nt": 10000,
      "motifs": {
        "donor": {
          "GT": 286,
          "CT": 1,
          "AT": 2,
          "GC": 1
        },
        "acceptor": {
          "AG": 304,
          "AC": 2
        }
      },
      "first_example": {
        "id": "FAM110C_2_43532_-",
        "gene": "FAM110C",
        "chromosome": "chr2",
        "strand": "-",
        "genomic_input_start": 33532,
        "genomic_input_end": 53532,
        "label_counts": {
          "0": 9998,
          "2": 1,
          "1": 1
        }
      }
    },
    "validation": {
      "n": 24,
      "chromosomes": [
        "6"
      ],
      "input_nt": 20000,
      "output_label_nt": 10000,
      "motifs": {
        "donor": {
          "GT": 56,
          "GC": 1
        },
        "acceptor": {
          "AG": 63
        }
      },
      "first_example": {
        "id": "IRF4_6_392452_+",
        "gene": "IRF4",
        "chromosome": "chr6",
        "strand": "+",
        "genomic_input_start": 382452,
        "genomic_input_end": 402452,
        "label_counts": {
          "0": 9991,
          "2": 5,
          "1": 4
        }
      }
    },
    "test": {
      "n": 36,
      "chromosomes": [
        "1",
        "3"
      ],
      "input_nt": 20000,
      "output_label_nt": 10000,
      "motifs": {
        "donor": {
          "GT": 154,
          "GC": 2,
          "AT": 1
        },
        "acceptor": {
          "AG": 149,
          "AA": 1
        }
      },
      "first_example": {
        "id": "SAMD11_1_925860_+",
        "gene": "SAMD11",
        "chromosome": "chr1",
        "strand": "+",
        "genomic_input_start": 915860,
        "genomic_input_end": 935860,
        "label_counts": {
          "0": 9995,
          "2": 3,
          "1": 2
        }
      }
    }
  }
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-73963a404d36

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch128_command.json` · 원문 SHA256: `509e0dd2626d8714f7162a5c1c8faf9c00ffa9b73f07c77a43fd4c4ff7790912`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch128_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch128_command.json", "sha256": "509e0dd2626d8714f7162a5c1c8faf9c00ffa9b73f07c77a43fd4c4ff7790912", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-128",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

### record-793f93f8c8ac

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch156_command.json` · 원문 SHA256: `4f64ec299f1d44fb7fcfcbee1c57527306252f06e1db1583593651fee65c11f2`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch156_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch156_command.json", "sha256": "4f64ec299f1d44fb7fcfcbee1c57527306252f06e1db1583593651fee65c11f2", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-156",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=156",
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

### record-aecf0be5edb0

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch16_command.json` · 원문 SHA256: `f44b35a2884c74723f87fc7a37360aad440828686844fad64a5ed9cde56c3b5c`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch16_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch16_command.json", "sha256": "f44b35a2884c74723f87fc7a37360aad440828686844fad64a5ed9cde56c3b5c", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-16",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

### record-abdde78862ed

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch1_command.json` · 원문 SHA256: `4ab0d538c90fdeb24e73ba46a747dc85fe9bc5695f109b2fc2e0903f6c6ca367`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch1_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch1_command.json", "sha256": "4ab0d538c90fdeb24e73ba46a747dc85fe9bc5695f109b2fc2e0903f6c6ca367", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

### record-9db12ec6d98a

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch2_command.json` · 원문 SHA256: `d5cf5779121fd9553ae002fb3ac278dde509621c105bdfced5ddb2465c58e5c0`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch2_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch2_command.json", "sha256": "d5cf5779121fd9553ae002fb3ac278dde509621c105bdfced5ddb2465c58e5c0", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-2",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

### record-8f32c23253c9

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch32_command.json` · 원문 SHA256: `7bac3e531e025c718400e7800a4b49d3ad90a527ac3624f7b195993e089839c1`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch32_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch32_command.json", "sha256": "7bac3e531e025c718400e7800a4b49d3ad90a527ac3624f7b195993e089839c1", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-32",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

### record-e95bcbe9d3b2

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch4_command.json` · 원문 SHA256: `8cbf74294d02b62e5a522b0e58cd4b59219776cffe822801a204a26804afc005`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch4_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch4_command.json", "sha256": "8cbf74294d02b62e5a522b0e58cd4b59219776cffe822801a204a26804afc005", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-4",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

### record-12e765c2e921

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch64_command.json` · 원문 SHA256: `79b9c434ecc55a6a98a964479ff5cc96eec790e8ec44646dca00d06715f2664a`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch64_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch64_command.json", "sha256": "79b9c434ecc55a6a98a964479ff5cc96eec790e8ec44646dca00d06715f2664a", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-64",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

### record-a561c102fa27

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch8_command.json` · 원문 SHA256: `be04a54132547eff58f545302cfc9f876bb90d4ddfd4ec151a777f324b09dd14`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch8_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch8_command.json", "sha256": "be04a54132547eff58f545302cfc9f876bb90d4ddfd4ec151a777f324b09dd14", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-8",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

### record-0db4a85fc2bb

원래 파일: `openspliceai/results/batch_followup_20261028_144737_167765/batch96_command.json` · 원문 SHA256: `aab03c1e4834fc673a8bb460c6a53a6ecccdaf06b43ce9b944e97a3ad79d7288`

<details>
<summary>openspliceai/results/batch_followup_20261028_144737_167765/batch96_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch96_command.json", "sha256": "aab03c1e4834fc673a8bb460c6a53a6ecccdaf06b43ce9b944e97a3ad79d7288", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-96",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

### record-44219c85124c

원래 파일: `openspliceai/results/openspliceai_20261027_205554_995618/benchmark_command.json` · 원문 SHA256: `9bb11141e70c4ac9bd19a55d1e8623e655e56edfe6e1d905cd731425bce05796`

<details>
<summary>openspliceai/results/openspliceai_20261027_205554_995618/benchmark_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/openspliceai_20261027_205554_995618/benchmark_command.json", "sha256": "9bb11141e70c4ac9bd19a55d1e8623e655e56edfe6e1d905cd731425bce05796", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/openspliceai:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=openspliceai_20261027_205554_995618 -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/benchmark.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-dd3d4c63435e

원래 파일: `openspliceai/results/openspliceai_20261027_205554_995618/reload_command.json` · 원문 SHA256: `b5bc083c3a7d068b4fdab4827af8302f533f90cc4e57c7ab921da2ab5f3b18f6`

<details>
<summary>openspliceai/results/openspliceai_20261027_205554_995618/reload_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/openspliceai_20261027_205554_995618/reload_command.json", "sha256": "b5bc083c3a7d068b4fdab4827af8302f533f90cc4e57c7ab921da2ab5f3b18f6", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/openspliceai:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=openspliceai_20261027_205554_995618 -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/reload.py"
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

<details><summary>추가 실험 실행 원문: full_training_b8_executed_source.py</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/full_training_b8_executed_source.py", "sha256": "dc1c9a02d54e9a816169e1fa0e2bbd51d8c40db6895fc12a6ec36141f4246f3f", "encoding": "utf-8", "added_final_newline": false} -->
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

<details><summary>추가 실험 실행 원문: inference_b10240_executed_source.py</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b10240_executed_source.py", "sha256": "11095b56bf1768b5e12be9814eb1ec572938ec22412ab702bc3ee3c7e8babce6", "encoding": "utf-8", "added_final_newline": false} -->
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

<details><summary>추가 실험 실행 원문: full_training_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/full_training_b8_command.json", "sha256": "5c4e06c47809f79664dfa382f19b988dddfc73c481cb7af88bb3df9fe29f0dfe", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
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

<details><summary>추가 실험 실행 원문: full_training_b96_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/full_training_b96_command.json", "sha256": "4f697671270c0fb017337413649e88223053d1fe0c52b9b718d36ce09418739c", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=96",
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

<details><summary>추가 실험 실행 원문: inference_b10240_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b10240_command.json", "sha256": "70b70fd24164d05f925a758039bf0d87b663c9af9f4e8d11cf28c3e5867e4706", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=10240",
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

<details><summary>추가 실험 실행 원문: inference_b1024_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b1024_command.json", "sha256": "b9303aeda5c8e4a965831249a2a48937ddbfd2c1d5f517ea43d47371a62d31fe", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=1024",
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

<details><summary>추가 실험 실행 원문: inference_b11264_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b11264_command.json", "sha256": "45888368902f60e2cd830074ba70b180b805ad0546463beff1a842ef035cdd71", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=11264",
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

<details><summary>추가 실험 실행 원문: inference_b2048_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b2048_command.json", "sha256": "bb9a7a360c4931c6442774ee195aafa1ba149713063e320d09d75196800940f9", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=2048",
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

<details><summary>추가 실험 실행 원문: inference_b4096_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b4096_command.json", "sha256": "57fcf03bab8cf2eafe7d7f247b15ac422e93ae28664c2470f152e2644a46efba", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=4096",
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

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b512_command.json", "sha256": "3c09e99ce09a073f20295892f0db39567b58858cfc258f4d4c1aa3dfc80d985c", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
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

<details><summary>추가 실험 실행 원문: inference_b6144_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b6144_command.json", "sha256": "289bde755577a7ff2c01769740929f6ad623242d900e982c38ca514445267ffb", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=6144",
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

<details><summary>추가 실험 실행 원문: inference_b8192_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b8192_command.json", "sha256": "f559ef1061986fdd27497f0a46bfcf5092c61694ea721fb721ab687ccdd96331", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=8192",
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

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b8_command.json", "sha256": "448e0837a9eee1c768c9c8f2fb64acbde2781be8144006f2489af61968aeda9c", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
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

<details><summary>추가 실험 실행 원문: inference_b9216_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/inference_b9216_command.json", "sha256": "e6dc8cf5fafaa8d665699715b1fac0ecfb5bddac4e10677863bac071b3738cac", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=9216",
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

<details><summary>추가 실험 실행 원문: training_probe_b32_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/training_probe_b32_command.json", "sha256": "26d4aacf964ca329f3599aae6e7ec97a2b5021c2036fe41ca1080386cdd04d24", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=32",
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

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/training_probe_b64_command.json", "sha256": "14e83d98608587132d7ecf3c7fb923b47d2ee4f0bba47e6a1b1a7ad9d5b1ee6e", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
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

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/training_probe_b8_command.json", "sha256": "cb8145104ca3f0b010ce6724bdea37e0960fe6c2322d639fa49af7cc4ca11c43", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
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

<details><summary>추가 실험 실행 원문: training_probe_b96_command.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/training_probe_b96_command.json", "sha256": "134d16831d4cba8ad1a8361b9999504af631734855215d2c7f7075394e1fe3f6", "encoding": "utf-8", "added_final_newline": true} -->
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
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=real_input_capacity_20261102_163800_779795",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=96",
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

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/reproduction_sources/remaining_queue_152f0b87bce32235.py", "sha256": "152f0b87bce32235a41f8616d74f03deabd611eef85dde6419bd68da175973a6", "encoding": "utf-8", "added_final_newline": false} -->
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

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/reproduction_sources/update_reports_305433d7fed68d6f.py", "sha256": "305433d7fed68d6ff20850b0d00d18708a0de3d92b36924d41dbbfe168034b8e", "encoding": "utf-8", "added_final_newline": false} -->
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

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/reproduction_sources/prepare_genome_pools_46b78d96d41c9f09.py", "sha256": "46b78d96d41c9f095a49916c206deb4ce29f0d16065ee5dd436a0fdd02bb8070", "encoding": "utf-8", "added_final_newline": false} -->
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

<details><summary>추가 실험 실행 원문: inference_audit_0d19e0cd49cea3a9.json</summary>

<!-- RESTORE_FILE {"path": "openspliceai/results/real_input_capacity_20261102_163800_779795/reproduction_sources/inference_audit_0d19e0cd49cea3a9.json", "sha256": "0d19e0cd49cea3a982541e99831a3af3b65d006cd241f768037c5cdb5587b4ba", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "unique_sequences": 16000,
  "raw_nt": 15000,
  "source_reference": "existing BEND GRCh38.primary_assembly.genome.fa",
  "selection": "chr2 then chr4 from2Mb, fixed nonoverlapping reference windows, ACGT only, exactsequence dedup; no repeated or synthetic sequences",
  "labels": null,
  "task": "inference only; not substituted for labelled training splits"
}
```
<!-- END_RESTORE_FILE -->

</details>
