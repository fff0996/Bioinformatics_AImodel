# ESM-2 최종 통합 보고서

## 1. 모델과 실제 입력

|항목|이번 실험에서 사용한 모델|
|---|---|
|체크포인트·버전|esm2_t33_650M_UR50D|
|구조|Transformer encoder,33층,1280차원|
|전체·구성별 파라미터 수|651,043,254개(651.04M);encoder649,400,320개|
|정밀도|FP32|
|모델 입력 → 출력|단백질 서열 →1280차원 임베딩 / GFP 형광값 예측|
|가중치와 실행 메모리의 차이|파라미터 수×dtype bytes는 이론 가중치 용량이다. 실행 peak에는 입력·중간 활성값이 더해지고,파인튜닝에는 gradient/optimizer 상태도 추가된다.|
|메모리% 정의|CUDA allocated peak / CPU·GPU 공유 RAM121.6875GiB ×100; GPU-Util과 별개. AlphaFold2는 JAX 계측을 별도로 표시|

### 실제 데이터와 학습 방식 구분

|구분|데이터·실제 처리 내용|
|---|---|
|추론 데이터|추론:UniProt reviewed 단백질. 추가 run은900–950AA이며 GFP 학습 서열과 별개|
|학습·검증·test 데이터|TAPE fluorescence의 실제 GFP 변이체 약237AA. 기본 train128/val64/test128;추가 비교1024/128/512|
|학습 입력 X / 정답 y|X:아미노산→int64 tokens[B,L+2];y:GFP log_fluorescence[B]. 라벨 표준화에는 train 평균·표준편차만 사용|
|가중치 갱신 방식|기본:일부 backbone 층+회귀head 파인튜닝;추가:전체 backbone+linear 회귀head 파인튜닝|
|원본·변환 입력 저장 위치|`data/fluorescence/fluorescence_{train,valid,test}.lmdb`|
|해석 범위|긴 UniProt 추론 서열에는 본 실험의 GFP 정답이 없다. 부분 미세조정과 전체 미세조정 수치를 섞지 않는다|

**용어:** 처음부터 학습=무작위 초기화에서 시작; 전체/일부 층 파인튜닝=공개 가중치에서 해당 부분 갱신; 분류기만 학습=본체를 동결하고 head만 갱신. **용량probe는 학습 방식이 아니라 짧은 자원 측정 목적**이며,실제로 수행한 가중치 갱신 방식과 함께 표기한다. 원본 전체 데이터 수와 해당 run에서 실제 사용한 부분집합 수를 구분한다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|단백질 임베딩 추론|고유1개,UniProt reviewed900~950AA|1×952토큰,FP32|56회/총56건|15.03초|3.72서열/초|2.700 GiB / 2.22%|정확도과제아닌임베딩; 80%미도달|
|단백질 임베딩 추론|고유384개,UniProt reviewed900~950AA|384×952토큰,FP32|2회/총768건|216.96초|3.54서열/초|64.416 GiB / 52.94%|정확도과제아닌임베딩; 80%미도달|
|부분backbone 파인튜닝|train128/val64/test128; 237AA fluorescence 라벨|토큰예산4096 가변batch,FP32|1epoch/8step|7.22초|17.74서열/초|3.623 GiB / 2.98%|동일저장모델test Spearman 0.0241;80%미도달|

<!-- REAL_TRAINING_V3_START -->
### 실제 데이터 추가 실험 — real_input_capacity_20261102_143107_778283

진행 중인 실행의 완료 조건만 아래에 갱신한다. probe는 짧은 용량 측정이며 전체 학습이나 평가 완료가 아니다. 메모리%는 CUDA allocated peak / 시스템 공유 RAM ×100이며 GPU-Util과 다르다. 독립 실행 1회 기준.

|조건|가중치 갱신 방식·실험 목적|상태|고유 입력|배치|epoch|optimizer steps|실측 시간(s)|입력/s|allocated peak GiB|reserved peak GiB|공유 RAM 대비 allocated %|GPU-Util 평균%|최대%|독립 test 지표|저장 모델 재로드|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|batch_1|추론(가중치 갱신 없음)|complete|1|1|N/A|N/A|15.1089|4.1697|2.7001|2.8496|2.2189|92.6552|96.0000|N/A|N/A|
|batch_384|추론(가중치 갱신 없음)|complete|384|384|N/A|N/A|215.8087|3.5587|64.4157|71.0352|52.9353|95.1533|96.0000|N/A|N/A|
|batch_448|추론(가중치 갱신 없음)|complete|448|448|N/A|N/A|244.0931|3.6707|74.7245|82.4453|61.4069|95.1304|96.0000|N/A|N/A|
|batch_512|추론(가중치 갱신 없음)|complete|512|512|N/A|N/A|277.9745|3.6838|85.0354|93.8359|69.8801|95.4217|96.0000|N/A|N/A|
|batch_544|추론(가중치 갱신 없음)|complete|544|544|N/A|N/A|297.5399|3.6567|90.1932|99.5488|74.1187|95.5602|96.0000|N/A|N/A|
|full_training_b64|전체backbone·과제head 파인튜닝|complete|1024|64|2|32|273.3533|7.4921|75.4426|79.5410|61.9970|95.3992|96.0000|{'mse': 2.104167938232422, 'spearman': 0.12595808476331907}|True|
|full_training_b8|전체backbone·과제head 파인튜닝|complete|1024|8|2|256|318.9110|6.4219|16.0687|16.8711|13.2049|95.3800|96.0000|{'mse': 2.1216509342193604, 'spearman': 0.2993711444496251}|True|
|probe_training_b32|전체backbone·과제head 파인튜닝 /용량측정|complete|96|32|0|3|13.9860|6.8640|41.5641|43.3789|34.1564|89.0385|96.0000|N/A|N/A|
|probe_training_b64|전체backbone·과제head 파인튜닝 /용량측정|complete|192|64|0|3|26.2989|7.3007|75.4426|79.2480|61.9970|92.2917|96.0000|N/A|N/A|
|probe_training_b8|전체backbone·과제head 파인튜닝 /용량측정|complete|24|8|0|3|4.4249|5.4239|16.0687|16.8711|13.2049|76.5000|96.0000|N/A|N/A|

이 run의 추론 capacity 실험과 지도학습 실험은 입력 과제·길이·학습 헤드가 다를 수 있어 각각 비교한다. 3step probe는 전체 학습이 아니다. full_training은 동일한 공개 초기 backbone와 seed42,FP32,AdamW(lr1e-5,weight_decay0.01),고정 전체 split pool padding으로2epoch 수행한다. 각 학습 조건의 원본 summary에 split ID,실제 길이·padding 모양,사용 헤드·학습 범위,라벨,gradient가 관측된 파라미터 수,검증/test지표,저장 모델 재로드 차이를 기록한다. scVI는 아래 설명이 적용되지 않는 별도 처음부터 학습 모델이며 AF2·Clair3에는 이 공통 학습 절차를 적용하지 않는다.

새 학습은 TAPE fluorescence의 실제 GFP train1024/valid128/test512를 고유 부분집합으로 사용한다. masked-mean+linear regression,학습 라벨 평균/표준편차로 정규화,전체 backbone 학습으로 이전 일부 계층 학습과 구분한다. 새 추론은 별도 UniProt reviewed900–950AA 실제 서열을 사용한다.

입력 및 메모리 기준값(같은 실행에서 측정):

|조건|실제 모델 입력 모양/정책|모델 적재 직후 allocated GiB|실행 시작 allocated GiB|peak 추가 allocated GiB|프로세스 RSS sampled peak GiB|시스템 가용 RAM sampled 최소 GiB|
|---|---|---|---|---|---|---|
|batch_1|[1, 952]|2.4898|2.5360|0.1640|2.6279|108.7578|
|batch_384|[384, 952]|2.4898|2.5360|61.8796|2.8065|39.3551|
|batch_448|[448, 952]|2.4898|2.5360|72.1885|2.6188|28.0765|
|batch_512|[512, 952]|2.4898|2.5360|82.4993|2.8672|16.5720|
|batch_544|[544, 952]|2.4898|2.5360|87.6572|2.8607|10.4114|
|full_training_b64|{'input_ids': [1664, 239], 'attention_mask': [1664, 239]}|2.4898|2.4898|72.9528|3.2248|30.4910|
|full_training_b8|{'input_ids': [1664, 239], 'attention_mask': [1664, 239]}|2.4898|2.4898|13.5789|3.0951|94.1325|
|probe_training_b32|{'input_ids': [1664, 239], 'attention_mask': [1664, 239]}|2.4898|2.4898|39.0743|3.0843|66.9297|
|probe_training_b64|{'input_ids': [1664, 239], 'attention_mask': [1664, 239]}|2.4898|2.4898|72.9528|3.1995|30.4346|
|probe_training_b8|{'input_ids': [1664, 239], 'attention_mask': [1664, 239]}|2.4898|2.4898|13.5789|3.2036|93.5097|

실행 상태: `stopped`; 중단/관측 범위: `actual_worker_error_or_guard`.

실행 코드 블록의 파일은 원문SHA256으로 검증하며, 명령JSON에 기록된 실행 경로로 해당 snapshot을 복사하면 복원할 수 있다. 대용량 원본 데이터·가중치는 아래 전처리/다운로드 코드와 출처를 사용하고 로컬 원본도 보존한다.

로컬 원시 결과: `/home/hyein/ai-benchmark/esm2/results/real_input_capacity_20261102_143107_778283`. 메모리 부족/guard 실패는 성공 조건으로 기재하지 않는다.
<!-- REAL_TRAINING_V3_END -->

## 4. 결과 해석

<!-- MODEL_SUMMARY_START -->
### 요약 결론

- **학습 방식:** 기본:일부 backbone 층+회귀head 파인튜닝;추가:전체 backbone+linear 회귀head 파인튜닝.
- **입력 구분:** 긴 UniProt 추론 서열에는 본 실험의 GFP 정답이 없다. 부분 미세조정과 전체 미세조정 수치를 섞지 않는다.
- **최신 추가 run 최대 성공 메모리:** `batch_544` **90.193GiB /74.12%**. 이 값은 해당 최신run의 성공 조건에서의 최대이며,다른 run의 최대와 구분한다. probe이면 전체 epoch·독립test 평가 완료를 뜻하지 않는다.
- **시간·정확도 해석:** 결과는 같은 행의 입력·배치·학습범위에만 연결한다. 메모리 상승 자체가 처리량·정확도 상승을 보장하지 않는다.
<!-- MODEL_SUMMARY_END -->

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### DGX Spark ESM-2 성능 검증 보고서


작성: 2026-10-27T17:01:39 (KST). 기본 run: `20261027_164300_125163`, 보완 run: `supplement_20261027_165443_595867`.

#### GPU 사용률 결론

전체 ESM-2 실험의 관측 최대 GPU-Util은 **96%**다. 기본 run의 긴 서열 조건(length_513_1022, 실제 516–976 residues)에서 기록됐다. 관측 구간을 늘린 40회 보완 측정의 같은 긴 서열 조건에서는 **평균 94.10%, 최대 95%**였다. 다른 실행에서 기록한 순간 최대와 지속 평균을 구분해야 한다.

배치(토큰 예산 4096)·FP32·650M 체크포인트를 사용률에 맞춰 바꾸지 않은 기준 조건에서 이미 평균 80%를 초과했다. 파인튜닝은 평균 88.14%, 최대 95%였다. GPU-Util은 장치 전체 커널 활동 시간 비율이며 메모리 점유율이나 최대 연산 성능 비율이 아니다. 메모리는 별도 계측으로, 학습 중 CUDA peak allocated는 3.623 GiB였다.

#### 결과와 완료 범위

실제 GB10 GPU에서 ESM-2 650M 임베딩 추론, 입력 개수·서열 길이 비교, 공개 fluorescence 라벨 기반 부분 파인튜닝, 저장, 독립 프로세스 재로드 및 test 예측 검증을 완료했다. 기본 작업은 중단 중에도 계속 실행돼 2026-10-27 16:45:26 KST에 종료 코드 0으로 끝났다. 기존 실행을 중복 시작하지 않았고, 보완 측정은 기본 작업 종료를 확인한 후 별도 폴더에서 순차 실행했다.

기본 개수 실험에서 평균 GPU-Util 89.4–92.5%가 관찰됐다. 80%를 맞추기 위한 배치·정밀도·모델 변경은 하지 않았다. 모델 실행·저장 성공과 예측 정확도는 구분해야 한다. 128개 test의 Spearman은 0.0241로 매우 낮아 실용적인 fluorescence 예측 성능을 입증하지 못했다.

#### 모델 적재 기준값과 입력별 추가 메모리

입력 forward 전 별도 새 프로세스에서 원본 공개 체크포인트를 로딩만 했다. CUDA는 먼저 초기화했고 optimizer·tokenization·forward는 실행하지 않았다. 이론 FP32 가중치는 2.425 GiB이고 실측 post-load allocated는 **2.490 GiB**, reserved는 2.559 GiB였다. GPU 할당은 가중치 이론값 외 프레임워크 할당도 포함한다. RSS는 2.087 GiB였다.

| 실제 서열 길이 | 실제 batch | GPU 평균 / 최대 % | 모델 로드 allocated GiB | 실행 allocated peak GiB | 기준 대비 참고 증가 GiB |
| --- | --- | --- | --- | --- | --- |
| 56–127 | 16 | 93.56 / 95 | 2.490 | 2.713 | 0.223 |
| 140–254 | 16 | 92.62 / 95 | 2.490 | 2.904 | 0.414 |
| 280–461 | 6–10 | 93.51 / 95 | 2.490 | 2.911 | 0.421 |
| 516–976 | 4–7 | 94.10 / 95 | 2.490 | 3.214 | 0.725 |

적재 기준은 다른 run에서 측정했으므로 차이는 참고값이며 같은 run의 전후 차이가 아니다. 추가량은 입력·중간값·작업 buffer 등을 포함한다. 학습 추가량에는 gradient·optimizer가 더 포함된다. 기준 원본: `/home/hyein/ai-benchmark/esm2/results/loading_20261027_190701_789190/loading.json`.

##### 실제 파인튜닝 데이터 예

TAPE fluorescence는 단백질 서열을 입력받고 측정된 log_fluorescence 연속값을 예측하는 회귀 과제다.

| split / ID | 실제 길이 | 실제 서열 앞부분 | 정답 log_fluorescence |
| --- | --- | --- | --- |
| train / train_20952 | 237 | SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGD… | 3.789218 |
| valid / valid_5238 | 237 | SKGEELFTGEVPILVELDGDVNGHKFSVSGEGEGD… | 3.443987 |
| test / test_20952 | 236 | SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGD… | 3.200874 |

가중치를 갱신한 대상은 전체 650M backbone이 아닌 마지막 block·LayerNorm·회귀 head다.

#### 모델·환경·기준 조건

| 항목 | 확인값 |
| --- | --- |
| 체크포인트 | esm2_t33_650M_UR50D, FAIR 공식 가중치 |
| 공식 구현 | fair-esm 2.0.0 |
| 구조 | Transformer 33층, hidden 1280, attention heads 20, residue 평균 pooling |
| 전체 추론 구성 파라미터 | 651,043,254 (LM/contact head 포함) |
| encoder 파라미터 | 649,400,320 (tied token embedding 포함; LM/contact의 고유 head 파라미터 제외) |
| 회귀 구성 전체 파라미터 | 651,044,535 (기존 모델 + 회귀 head) |
| 학습 가능 backbone / 회귀 head | 19,680,000 / 1,281 |
| 학습 가능 합계 | 19,681,281 |
| 장비 | spark-e4b6, NVIDIA GB10, ARM64 |
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0-1019-nvidia |
| Python / PyTorch / CUDA | 3.10.21 / 2.14.1+cu130 / 13.0 |
| NVIDIA 드라이버 | 580.178.04 |
| 시스템 물리 RAM | 130,660,974,592 bytes (121.688 GiB) |
| 배치 정책 | 공식 FastaBatchedDataset: token budget 4096, extra_toks_per_seq=1 |
| 정밀도·seed·CPU threads | FP32, TF32 disabled, seed 42, threads 8 |
| 모델 입력 | torch.int64 [B, max_residues+2], BOS/EOS와 padding |
| 길이 제한 | 1022 residues, 실제 비교 입력은 모두 제한 이내 |
| 실행 방식 | 호스트 venv, 기존 DNABERT-2 환경의 system-site-packages를 읽어 PyTorch 재사용 |
| 기존 환경 보존 | 기반 환경 수정 없음; 새 venv에 fair-esm, lmdb 설치 |
| 컨테이너 | Docker 접근 거부로 사용하지 않음; 모든 결과는 호스트 경로에 저장 |

토큰 예산은 공식 배치 구성 함수의 추정치다. 함수가 extra token을 1개로 계산하지만 ESM-2 변환은 BOS/EOS 2개를 붙이므로 실제 padded token은 일부 배치에서 4,102개였다. 예산과 실제 shape를 구분해 저장했다.

#### 입력 확인과 데이터 출처

UniProtKB reviewed release 2026_03 (2026-09-02), API의 첫 500개 FASTA 레코드를 확보했다. 전체 UniProt 또는 무작위 종 대표 표본이 아니다. 원본 500개, 고유 서열 499개, 길이 56–1020 residues, 표준 아미노산 20종, 빈 서열 없음이다. 동일 서열 1개는 개수 실험에서 제거했다. 라벨이 없는 임베딩 실험 입력이다.

개수 비교는 50–128 / 129–256 / 257–512 / 513–1022의 4개 길이 구간에서 각각 seed 42로 섞어 교차 배치한 중첩 부분집합 32/96/160개다. 구간별 개수는 조건마다 각각 8/24/40개로 같지만 구간 내부 길이 분포와 최대 길이는 완전히 일치하지 않는다. 개수 조건의 평균 길이는 328.28/341.73/340.82여서 처리량 차이를 순수한 개수 효과로 단정할 수 없다. 길이별 입력 목록, 실제 batch shape, padding을 저장했다.

TAPE 공식 fluorescence LMDB를 사용했다. 단백질 GFP 변이 서열에서 float32 [1] log_fluorescence를 예측하는 회귀 과제다. 분류 정확도 지표를 사용하지 않았다.

| 원본 split | 레코드 | 길이 범위 | 라벨 평균 | 누락/비유한 라벨 |
| --- | --- | --- | --- | --- |
| train | 21446 | 235–237 | 3.181 | 0 |
| valid | 5362 | 235–237 | 3.173 | 0 |
| test | 27217 | 235–237 | 2.083 | 0 |

원본 split 간 동일 서열 중복은 모두 0이다. 기존 split을 유지한 채 train 128 / validation 64 / test 128개를 각 split에서 독립적으로 seed 42로 선정했다. test로 하이퍼파라미터나 epoch를 선택하지 않았다. GFP 변이들이기 때문에 높은 상동성은 남으며, 독립 단백질 family 일반화를 검증하지 않는다. 사전학습 데이터와의 중복도 검증하지 않았다.

##### 실제 입력·출력 예

```text
>sp|P01764|HV323_HUMAN
MEFGLSWLFLVAILKGVQCEVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYA...
original residues = 117; truncated = False
smoke tokens = [2, 196]; pooled embeddings = [2, 1280] float32 on CUDA
fluorescence: amino-acid sequence → mean pooled 1280-vector → Linear(1280,1) → log_fluorescence
```

#### 측정 범위와 방법

전·후 torch.cuda.synchronize()를 호출했다. 추론 실행 시간에는 CPU 토큰화, CUDA 전송, forward, pooling, pooled 결과의 CPU 전송이 포함되며 모델 로딩, warmup, torch.save는 제외한다. preprocess_seconds는 batch 목록 구성 시간만 의미한다. 초기 다운로드와 전체 데이터 검사 시간은 별도 계측하지 않아 N/A다. 모델 로딩에는 체크포인트 읽기·모델 생성·CUDA 전송을 포함한다. 컴파일은 사용하지 않았다.

기본 run 모델 로딩 2.344초, smoke/warmup 0.507초. 보완 run 모델 로딩 2.344초, 각 길이 조건 전체 1 pass를 별도 warmup했다. 기본 개수 조건은 3회 반복, 보완 길이 조건은 동일 16개 입력을 40회 반복했다. 반복은 고유 데이터 추가가 아니다.

GPU-Util과 RAM은 0.5초 간격으로 표본 수집했다. GPU-Util은 장치 전체의 커널 실행 시간 비율로 최대 FLOPS/SM occupancy가 아니며 디스플레이·Firefox 등도 포함된다. nvidia-smi 자체 내부 집계 창과 조회 지연이 있어 독립 표본으로 해석하지 않는다. 보완 실험은 query 시작·종료가 모두 동기화된 실행 구간 안에 있는 표본만 포함하고 첫 2초를 제외한 평균도 별도 보관했다. 기본 run은 phase 단위이므로 매우 짧은 입력 목록·파일 저장 시간도 GPU/RAM 표본에 섞일 수 있다.

CUDA peak allocated/reserved는 PyTorch allocator 값이다. Spark는 CPU/GPU 공유 DRAM이므로 RSS와 CUDA 메모리를 더하지 않았다. RSS는 0.5초 sampled peak이며 정확한 최대값이 아니다. nvidia-smi 장치 전체 memory 사용량은 Not Supported이므로 N/A다. 시스템 MemAvailable과 swap은 별도 기록했다.

#### A. 서열 개수 증가: 원본 완료 결과

| 고유 / 총 처리 | 평균 길이 | 실행 합계 s | 평균 1회 s | 서열/s | GPU 평균 / 최대 % |
| --- | --- | --- | --- | --- | --- |
| 32 / 96 | 328.28 | 11.205 | 3.735 | 8.567 | 89.77 / 95 |
| 96 / 288 | 341.73 | 29.960 | 9.987 | 9.613 | 89.37 / 95 |
| 160 / 480 | 340.82 | 46.721 | 15.574 | 10.274 | 92.49 / 95 |

| 고유 입력 | CUDA alloc / reserved GiB | RSS 표본 최대 GiB | MemAvailable 최소 GiB | 표본 수 | swap bytes |
| --- | --- | --- | --- | --- | --- |
| 32 | 2.971 / 3.676 | 1.455 | 108.860 | 22 | 0 |
| 96 | 3.128 / 5.205 | 1.293 | 108.225 | 60 | 0 |
| 160 | 3.162 / 6.664 | 1.296 | 106.819 | 94 | 0 |

| 조건 | batch 수/pass | batch 샘플 범위 | 실제 padded token 최대 | batch 준비 s | 저장 s | 기록 wall s |
| --- | --- | --- | --- | --- | --- | --- |
| count_32 | 4 | 3–16 | 4096 | 0.000039 | 0.021 | 11.828 |
| count_96 | 10 | 2–27 | 4050 | 0.000088 | 0.012 | 30.574 |
| count_160 | 16 | 2–32 | 4102 | 0.000134 | 0.016 | 47.340 |

기본 condition wall에는 인위적인 표본 종료 대기 0.6초가 포함된다. 위 실행 시간과 처리량에는 이 대기를 포함하지 않는다. 전체 원본 프로세스 wall의 독립 perf-counter 계측은 N/A이며, run ID 시작 시각부터 종료 시각의 차이는 약 146초다. 다운로드·설치 전체 소요 시간으로 해석하지 않는다.

개수 증가에 따라 총 처리 시간이 증가했지만 GPU 평균은 약 89–92%로 이미 높다. 순간 allocated peak는 2.97→3.16 GiB로 완만히 변했다. 입력 최대 길이·batch packing·allocator 상태의 변화가 함께 작용하므로 개수만 늘려 메모리가 비례 증가했다고 설명하지 않는다.

#### B. 서열 길이 비교: 관측 구간을 늘린 보완 결과

각 조건 고유 16개, 측정 40회 × 16 = 640건, 별도 warmup 16건 제외. 토큰 예산·정밀도·checkpoint는 동일하고 토큰 예산 정책에 따라 실제 batch 샘플 수가 달라진다. 고정 batch size에서의 순수 길이 효과와는 구분한다.

| 실제 길이 / 평균 | 실행 s | 서열/s | GPU 평균 / 최대 % | 첫 2초 제외 평균 % | 표본 수 |
| --- | --- | --- | --- | --- | --- |
| 56–127 / 101.69 | 18.273 | 35.025 | 93.56 / 95 | 93.38 | 36 |
| 140–254 / 190.31 | 39.685 | 16.127 | 92.62 / 95 | 92.52 | 79 |
| 280–461 / 374.31 | 71.068 | 9.005 | 93.51 / 95 | 93.48 | 142 |
| 516–976 / 690.31 | 151.429 | 4.226 | 94.10 / 95 | 94.09 | 303 |

| 길이 구간 | CUDA alloc / reserved GiB | RSS sampled peak GiB | MemAvailable 최소 GiB | warmup s | 저장 s | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| length_50_128 | 2.713 / 2.900 | 2.961 | 108.597 | 0.702 | 0.011 | 18.990 |
| length_129_256 | 2.904 / 3.082 | 2.960 | 108.494 | 0.988 | 0.001 | 40.688 |
| length_257_512 | 2.911 / 3.182 | 2.959 | 108.262 | 1.863 | 0.001 | 72.942 |
| length_513_1022 | 3.214 / 4.209 | 2.959 | 107.306 | 3.651 | 0.001 | 155.105 |

길어질수록 처리량이 낮아졌다. 입력 내용이 다른 실제 단백질 그룹이므로 길이만 바꾼 동일 서열 실험은 아니다. 모든 입력은 truncation 없이 처리됐으며, pooling에서 BOS/EOS와 padding을 제외했다. 기본 3회 길이 측정도 보존했지만 56–127 구간은 실행 1.325초·표본 3개여서 GPU 평균 59.7%가 불안정했다. 보완 결과로 사용률 해석을 갱신했으며, 실행 시간을 인위적으로 배치 sweep으로 늘리지 않았다.

#### 부분 파인튜닝·검증·저장

fair-esm의 PyTorch ESM2 backbone에 사용자 정의 선형 회귀 head를 붙였다. 공식 패키지의 원클릭 fluorescence 학습 명령을 실행한 것이 아니다. 마지막 Transformer block(layers.32)과 최종 LayerNorm, head만 갱신하고 앞선 32개 block과 embedding은 동결했다. 전체 backbone fine-tuning으로 표현하지 않는다. train split의 라벨 평균·표준편차로만 표준화하고 예측 시 원 척도로 역변환했다.

| 항목 | 결과 |
| --- | --- |
| train / validation / test | 128 / 64 / 128 |
| 학습 | 1 epoch, 8 optimizer steps, single run |
| optimizer | AdamW, backbone lr=1e-4, head lr=1e-3, weight_decay=0.01 |
| loss | 표준화된 label의 MSE |
| 학습 실행 시간 s | 7.216 |
| 학습 처리량 서열/s | 17.738 |
| GPU 평균 / 최대 % | 88.14 / 95 |
| GPU 관측 표본 | 14 |
| CUDA alloc / reserved GiB | 3.623 / 3.814 |
| RSS sampled peak GiB | 1.640 |
| MemAvailable 최소 GiB | 109.441 |
| 모델 저장 s | 1.106 |
| backbone / head 가중치 갱신 | True / True |
| 동일 프로세스 재로드 차이 | 0.0 |
| 새 프로세스 재구성 차이 | 0.0 |
| 학습 재개 검증 | 미수행 (optimizer, epoch, RNG 저장만 수행) |

학습 구간은 7.2초로 비교적 짧고 표본 14개이므로 GPU 평균을 장기 학습의 확정값으로 사용하지 않는다. 입력 규모별 학습 비교는 수행하지 않았다. 학습 조건은 1개이며 이후 정확도 개선용 재학습이나 test 기반 튜닝은 하지 않았다.

| 평가 | n | MSE | MAE | Spearman |
| --- | --- | --- | --- | --- |
| 학습 전 validation (무작위 head) | 64 | 1.196664 | 0.749147 | -0.100116 |
| 학습 후 validation | 64 | 1.007656 | 0.847753 | -0.137204 |
| 독립 test | 128 | 1.961232 | 1.302845 | 0.024090 |

train 평균 상수 예측의 test MSE는 2.265732이다. test MSE가 이보다 낮더라도 Spearman 0.0241은 기능적 순위 예측을 입증하지 못한다. train/validation은 1–3 mutation, test는 4–12 mutation의 분포 차이가 있고, 소규모 1 epoch만 학습했다. 이는 제한된 동작 검증이다. 전체 TAPE 학습·최적 수렴·독립 family 평가를 완료한 결과가 아니다.

#### 검증·재현과 산출물

기본 7조건 × 3회 = 21개 embedding 파일의 입력 ID·유일성·[N,1280] shape·유한값과 반복 결과 일치를 확인했다. 최초 저장 모델 재로드는 가중치를 임시로 0으로 만든 후 저장본을 복원해 예측 일치를 검사했다. 보완 실행은 새 OS 프로세스에서 공식 base hash를 검사하고 저장된 final block·LayerNorm·head를 적용해 128 test 예측이 tolerance 1e-5 안에서 일치함을 확인했다.

checkpoint.pt는 전체 650M base 사본을 중복 저장하지 않고 갱신된 backbone 파라미터와 head, target 표준화 값, base 이름·해시, optimizer·epoch·RNG를 보관한다. 따라서 base checkpoint와 contact regression 파일을 함께 유지해야 한다. metadata, alphabet, 독립 predict.py와 사용법도 저장했다.

```bash
cd /home/hyein/ai-benchmark
esm2/.venv/bin/python esm2/scripts/predict.py \
  --checkpoint esm2/models/finetuned/20261027_164300_125163/checkpoint.pt \
  --fasta input.fasta --output predictions.csv
```

| 산출물 | 호스트 경로 |
| --- | --- |
| 최신 지침 | /home/hyein/ai-benchmark/AGENTS.md |
| 기본 코드 | /home/hyein/ai-benchmark/esm2/scripts/benchmark.py |
| 준비·입력 출처 | /home/hyein/ai-benchmark/esm2/scripts/prepare.py / /home/hyein/ai-benchmark/esm2/data/sources.json |
| 기본 run | /home/hyein/ai-benchmark/esm2/results/20261027_164300_125163 |
| 보완 run | /home/hyein/ai-benchmark/esm2/results/supplement_20261027_165443_595867 |
| 학습 모델 | /home/hyein/ai-benchmark/esm2/models/finetuned/20261027_164300_125163 |
| 독립 예측 코드 | /home/hyein/ai-benchmark/esm2/scripts/predict.py |
| 비교 CSV | /home/hyein/ai-benchmark/esm2/reports/comparison_20261027_164300_125163.csv |
| 전체 상태 | /home/hyein/ai-benchmark/STATUS.md |
| 현재 보고서 | /home/hyein/ai-benchmark/esm2/reports/ESM2_20261027_164300_125163.md |

기본 작업의 최초 stdout은 파일로 redirect하지 않아 전체 원본 console.log가 없다. 세션 도구가 반환한 exit_code=0·SUCCESS와 복구 확인을 process_recovery.json에 기록했고, 원본 resources.csv·phase 지표는 보존돼 있다. 보완 실행은 console.log·command.json·exit_code.json을 저장한다. 기본 원본의 JSON은 수정하지 않았다. 다운로드 URL·서버 버전·크기·SHA-256은 sources.json에 기록했다. 기존 DNABERT-2 경로와 문서 tar를 조사했으나 기존 한국어 최종 Markdown 보고서는 발견하지 못해 기존 summary.json 계측 항목과 최신 사용자 요구를 기준으로 구성했다.

#### 한계와 다음 단계

가용 메모리는 모든 측정에서 충분했고 swap 증가는 없었으며 OOM은 발생하지 않았다. 총 샘플 수를 늘리는 것과 단일 GPU 입력을 키우는 것은 다르다. 사용률 80%는 일반 조건에서 이미 초과했으며 그 자체가 모델 정확도나 최대 하드웨어 성능을 뜻하지 않는다. 반복 3/40회는 한 프로세스 내 warm 상태 반복으로 독립 실행·독립 seed 실험이 아니다. 입력·모델의 사전 다운로드 캐시는 warm 상태이며 전체 생물학적 활용 범위를 검증한 것은 아니다.

후속 순서는 AlphaFold2 — ColabFold를 통한 실행이다. 현재 host hyein의 실제 그룹에 docker가 없고 /var/run/docker.sock은 root:docker 0660이다. Docker 조회는 샌드박스 밖에서도 permission denied여서 기존 이미지·컨테이너·마운트·가중치를 검증할 수 없다. 자동 권한 변경이나 sudoers 변경은 하지 않았다. 사용자가 현재 세션의 Docker 접근을 복구해야 기존 ColabFold 검증을 이어갈 수 있다.

#### 공식 출처

- [ESM-2 공식 구현·체크포인트](https://github.com/facebookresearch/esm)
- [공식 토큰 예산·입력 truncation 구현](https://github.com/facebookresearch/esm/blob/main/scripts/extract.py)
- [TAPE 과제·데이터](https://github.com/songlab-cal/tape)
- [TAPE 공식 데이터 다운로드 목록](https://github.com/songlab-cal/tape/blob/master/download_data.sh)
- [UniProt 공식 REST API](https://www.uniprot.org/help/api_queries)
- [GPU-Util 정의·계측 지원](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [DGX Spark 공유 메모리·nvidia-smi memory 제한](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### ESM-2 실제 긴 단백질 서열의 배치별 추론 메모리


실행 `batch_inference_20261028_112900_601721` · 상태 **complete** · 종료 사유 `predicted_reservation_exceeds_available_budget`

#### 입력과 배치의 의미

배치는 한 번의 GPU forward에 넣는 **단백질 서열 개수**다. 아미노산 하나가 보통 토큰 하나이며, 이번 입력은 실제 UniProt reviewed 서열900~950아미노산이다. 모든배치에950아미노산 서열을 포함해 BOS/EOS 포함GPU입력 길이를952로 유지했다. 원본 전체 개수와 동시에 계산하는 batch를 구분한다. 전체pool에서 중첩고유부분집합을 사용하고, 같은batch를 측정 반복하므로 고유서열수와총처리건수는다르다. 자르거나 인공서열을 만들지 않았다. 라벨은없고임베딩추론만수행했다.

#### 고정 조건과 측정 정의

공식 esm2_t33_650M_UR50D, FP32·TF32=False·seed42·기존fair-esm 환경. 각batch를새프로세스로시작해allocator예약누적을분리했다. 파라미터수와동일run의모델적재직후allocated/reserved·RSS·MemAvailable은batch_B/summary.json에보존한다. 가중치SHA256은root summary.json,데이터릴리스·다운로드URL·SHA256은source필드에있다. 입력토큰및서열·ID를보존했다.

표의시간은cached CPU tokens의GPU전송·공식forward(사용하지않는logits계산도포함)·평균임베딩pooling·CPU반환을포함하고전처리/로딩/warmup/파일저장은제외한다. 동기화후측정하며최소2회·15초를확보한다. warmup별시간은각worker summary에있다. GPU-Util은장치전체활동시간비율,allocated/reserved는PyTorch프로세스allocator값이다. 메모리비율은전체CPU/GPU공유RAM121.688GiB대비참고값이며전용VRAM비율이아니다. RSS와CUDA를더하지않는다.

|배치/고유서열|측정반복|총처리건수|초|서열/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|56|56|15.035|3.725|2.49|2.7|0.21|2.85|2.219|92.25|95.0|2.765|107.815|
|2|24|48|15.537|3.089|2.49|2.858|0.368|2.977|2.349|94.724|95.0|1.354|109.103|
|4|12|48|15.544|3.088|2.49|3.181|0.691|3.326|2.614|94.724|95.0|2.606|107.751|
|8|6|48|15.487|3.099|2.49|3.831|1.341|4.043|3.148|94.893|95.0|1.332|108.114|
|16|3|48|15.253|3.147|2.49|5.114|2.624|5.475|4.203|94.69|95.0|2.688|105.665|
|32|2|64|20.113|3.182|2.49|7.692|5.202|8.322|6.321|91.676|95.0|2.607|102.858|
|64|2|128|40.876|3.131|2.49|12.852|10.362|14.018|10.562|94.338|95.0|2.58|97.175|
|96|2|192|61.545|3.12|2.49|18.005|15.515|19.729|14.796|93.248|95.0|2.582|91.43|
|128|2|256|80.744|3.171|2.49|23.166|20.676|25.426|19.037|94.156|95.0|2.691|85.454|
|160|2|320|100.887|3.172|2.49|28.32|25.83|31.119|23.273|93.573|95.0|2.585|79.802|
|192|2|384|120.854|3.177|2.49|33.473|30.984|36.832|27.508|94.018|95.0|2.585|73.926|
|224|2|448|127.947|3.501|2.49|38.633|36.143|42.529|31.748|94.446|96.0|2.749|68.063|
|256|2|512|145.698|3.514|2.49|43.787|41.297|48.223|35.983|94.733|96.0|2.613|62.33|
|288|2|576|163.838|3.516|2.49|48.942|46.452|53.932|40.219|94.876|96.0|2.64|56.609|
|320|2|640|181.644|3.523|2.49|54.101|51.611|59.629|44.459|95.295|96.0|2.752|50.745|
|384|2|768|216.958|3.54|2.49|64.416|61.926|71.035|52.935|94.789|96.0|2.615|39.285|

#### 결과

시험한 최대 성공배치 **384**. allocated80%도달: False,90%도달: False. 이는관측조건의결과이며모델의절대최대한계가아니다.

시스템가용RAM16GiB·swap증가2GiB guard와다음배치예약량예측으로확대를중단한다. worker실패가있으면그조건의exit/log를보존하며성공행에포함하지않는다. 예상guard에의해다음배치를건너뛴것은그배치의실제실패나90%불가능증명이아니다. RSS/가용량은0.5초대기+조회시계열의sampled값이다. 메모리할당peak는CUDAcounter이다.

각조건embedding shape B×1280·finite·저장후정확재로드를확인했고,같은첫서열의batch1대비임베딩일치도도검사했다. 학습·정확도평가·독립생물학적일반화평가를수행한결과가아니다. 일반조건인토큰예산4096경로와다른명시적batch용량실험이므로기본성능결과를대체하지않는다.

[공식ESM](https://github.com/facebookresearch/esm) · [UniProt 데이터출처](https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20length%3A%5B900%20TO%20950%5D&format=fasta&size=500) · [기본보고서](ESM-2.md)

원본: ../results/batch_inference_20261028_112900_601721/summary.json, batch_B/summary.json, resources.csv, embeddings.pt, tokenized_inputs.pt, 명령·exit·로그. 재현코드는scripts/batch_followup에있다.

### 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.


## 6. 실제 실행 코드·명령어 전문

<!-- EXECUTED_SCRIPTS_START -->

### 원문과 출처 검증

아래 코드 블록은 원본 내용을 생략하거나 재작성하지 않았다. 각 파일의 원래 경로·원문 SHA256·출처 상태를 함께 기록했다. `실행 run 저장 snapshot`을 우선하고, 현재 소스만 남은 경우에는 당시 파일과 같다고 단정하지 않는다. 동일 바이트 버전은 한 번만 수록하고 해당 경로들을 함께 표시한다.

GitHub에는 `.py`/`.sh`를 별도 파일로 두지 않는다. 코드 블록을 원래 경로에 저장하거나 [복원 안내](ESM-2.md)의 코드로 로컬에서 복원한다. 라이브러리 전체 소스는 이 사용자 실행 스크립트와 구분하며 패키지 버전/출처를 따른다.

### 코드 목록

|원래 파일|원문 SHA256|출처 상태|
|---|---|---|
|`esm2/scripts/batch_followup/launch.py`|`894b721991490d3b66a0c80b641a2ddad27fb2109efabbe80d7b2dd4d9f9885b`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`esm2/scripts/batch_followup/report.py`|`66216a98513911eb4413bdc08aad2dc418908c2fe3e9a03df2754e7ac5bb45a6`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`esm2/scripts/batch_followup/worker.py`|`02bf8c3072ca143a9b0765278e62cf1380434c78a240008cf214e0a59ab02f89`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`esm2/scripts/benchmark.py`|`20863f34231101de503f46b8b5ef61c1ea1c775eb0ff7ebb49320c5548f9e94d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`esm2/results/20261027_164300_125163/benchmark.py`|`20863f34231101de503f46b8b5ef61c1ea1c775eb0ff7ebb49320c5548f9e94d`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`esm2/scripts/measure_loading.py`|`1c021da223b492c31233ca4c23267eb7ffc4a61b5616fcd0257bf8477e53d5a1`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`esm2/scripts/predict.py`|`1ff266dbfc0eebeb26c4e8f9b6995529c8c75b59b25c43c70e1ba10bd19bda9b`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`esm2/scripts/prepare.py`|`29f09e4a7d6cc687bbb14ddd3a2fd8cb8f7cfec87e2c61d682b3b57238c6a154`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`esm2/scripts/report.py`|`13e34868ad22d1e2c0e83c3fb84e93c10ed055452992983f57b0bb73a2d44b0b`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`esm2/scripts/supplement.py`|`ceac1a1593c7aaa06e2ded3fff60a8b930858d9c5ef8a4508baec89a324ae266`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`esm2/results/supplement_20261027_165443_595867/supplement.py`|`ceac1a1593c7aaa06e2ded3fff60a8b930858d9c5ef8a4508baec89a324ae266`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|

<details>
<summary>esm2/scripts/batch_followup/launch.py — 원문 전문 (5,161 bytes)</summary>


<!-- RESTORE_FILE {"path": "esm2/scripts/batch_followup/launch.py", "sha256": "894b721991490d3b66a0c80b641a2ddad27fb2109efabbe80d7b2dd4d9f9885b", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,datetime,subprocess,sys,hashlib,random,time,os
import psutil
R=Path('/home/hyein/ai-benchmark/esm2');run='batch_inference_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');print(run,flush=True)
source=R/'data/uniprot_reviewed_900_950_20261028.fasta';rows=[]
for line in source.read_text().splitlines():
 if line.startswith('>'):rows.append({'id':line[1:].split()[0],'description':line[1:],'sequence':''})
 else:rows[-1]['sequence']+=line.strip()
seen=set();selected=[]
for r in rows:
 s=r['sequence'];assert 900<=len(s)<=950 and set(s)<=set('ACDEFGHIKLMNPQRSTVWYBXZOU');assert r['id'] not in {x['id'] for x in selected}
 if s not in seen:selected.append(r);seen.add(s)
random.Random(42).shuffle(selected);longest=max(range(len(selected)),key=lambda i:len(selected[i]['sequence']));selected.insert(0,selected.pop(longest));assert len(selected[0]['sequence'])==950
(D/'inputs.json').write_text(json.dumps(selected,indent=2));sources=json.loads((R/'data/sources.json').read_text());path=R/'models/esm2_t33_650M_UR50D.pt';sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==next(s['sha256'] for s in sources if s['path']=='models/esm2_t33_650M_UR50D.pt')
summary={'status':'running','run_id':run,'conditions':[],'failed':[],'checkpoint':'esm2_t33_650M_UR50D','checkpoint_sha256':sha,'source':json.loads(source.with_suffix('.source.json').read_text()),'input_audit':{'original_fasta_rows':len(rows),'unique_sequences':len(selected),'sequence_length_range':[min(len(r['sequence']) for r in selected),max(len(r['sequence']) for r in selected)],'sequence_1_length':950,'actual_padded_token_length_all_batches':952,'selection':'seed42 shuffle, longest sequence first to keep actual max length950 constant; nested unique prefixes, no within-batch copies','labels':None},'settings':{'task':'embedding inference only; no training','precision':'FP32','tf32':False,'seed':42,'guard_available_gib':16,'guard_swap_increase_gib':2,'minimum_repeats':2,'minimum_measured_seconds':15,'time_includes':'cached CPU tokens to GPU, full official forward including unused logits, mean pooling toCPU; excludes tokenization/load/warmup/filesave','framework_note':'native fair-esm2.0.0, existing host venv unchanged, not comparable to scGPT framework memory as model-only effect'},'largest_successful_batch':None}
def save():(D/'summary.json').write_text(json.dumps(summary,indent=2))
save();swap0=psutil.swap_memory().used
for batch in [1,2,4,8,16,32,64,96,128,160,192,224,256,288,320,384,448,500]:
 if batch>len(selected):summary['stop_reason']='available_unique_sequence_limit';break
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU compute job exists'
 available=psutil.virtual_memory().available/2**30
 if available<16 or psutil.swap_memory().used>swap0+2*2**30:summary['stop_reason']='system_memory_guard_before_launch';break
 if summary['conditions']:
  prior=summary['conditions'][-1];m=prior['memory'];b0=prior['batch_size'];base=prior['model_only_allocated_gib'];estimated=base+(m['peak_reserved_gib']-base)*(batch/b0)*1.25+3
  if estimated>available-16:summary['stop_reason']='predicted_reservation_exceeds_available_budget';summary['skipped_next_batch']={'batch':batch,'estimated_reserved_gib':estimated,'available_gib':available};break
 cmd=[str(R/'.venv/bin/python'),'-u',str(R/'scripts/batch_followup/worker.py'),str(D),str(batch)];(D/f'command_batch{batch}.json').write_text(json.dumps(cmd));start=time.time()
 with (D/f'batch{batch}.log').open('w') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 stopreason=None
 while p.poll() is None:
  if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:stopreason='parent_memory_guard';p.terminate();break
  time.sleep(.5)
 code=p.wait();(D/f'exit_batch{batch}.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start,'stop_reason':stopreason}));print('BATCH',batch,'EXIT',code,flush=True)
 if code:
  summary['failed'].append({'batch':batch,'exit_code':code,'reason':stopreason or 'worker_failure_see_log'});summary['stop_reason']='stopped_after_worker_failure';break
 s=json.loads((D/f'batch_{batch}/summary.json').read_text());assert s['status']=='complete';r=s['conditions'][0];r['model_only_allocated_gib']=s['model_only_postload']['allocated_gib'];summary['conditions'].append(r);summary['largest_successful_batch']=batch;save()
else:summary['stop_reason']='planned_batch_list_complete'
summary['status']='complete' if not summary['failed'] else 'stopped';summary['reached80']=any(r['allocated_fraction_shared_ram']>=.8 for r in summary['conditions']);summary['reached90']=any(r['allocated_fraction_shared_ram']>=.9 for r in summary['conditions']);save();subprocess.run([sys.executable,str(R/'scripts/batch_followup/report.py'),str(D)],check=True);(D/'exit.json').write_text(json.dumps({'exit_code':0,'scope':'controller finished; individual failures separately recorded'}));print('FINISHED',summary['stop_reason'],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>esm2/scripts/batch_followup/report.py — 원문 전문 (5,693 bytes)</summary>


<!-- RESTORE_FILE {"path": "esm2/scripts/batch_followup/report.py", "sha256": "66216a98513911eb4413bdc08aad2dc418908c2fe3e9a03df2754e7ac5bb45a6", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,csv,sys
D=Path(sys.argv[1]);R=D.parents[1];s=json.loads((D/'summary.json').read_text());lines=['# ESM-2 실제 긴 단백질 서열의 배치별 추론 메모리','',f"실행 `{D.name}` · 상태 **{s['status']}** · 종료 사유 `{s.get('stop_reason','running')}`",'','## 입력과 배치의 의미','','배치는 한 번의 GPU forward에 넣는 **단백질 서열 개수**다. 아미노산 하나가 보통 토큰 하나이며, 이번 입력은 실제 UniProt reviewed 서열900~950아미노산이다. 모든배치에950아미노산 서열을 포함해 BOS/EOS 포함GPU입력 길이를952로 유지했다. 원본 전체 개수와 동시에 계산하는 batch를 구분한다. 전체pool에서 중첩고유부분집합을 사용하고, 같은batch를 측정 반복하므로 고유서열수와총처리건수는다르다. 자르거나 인공서열을 만들지 않았다. 라벨은없고임베딩추론만수행했다.','','## 고정 조건과 측정 정의','','공식 esm2_t33_650M_UR50D, FP32·TF32=False·seed42·기존fair-esm 환경. 각batch를새프로세스로시작해allocator예약누적을분리했다. 파라미터수와동일run의모델적재직후allocated/reserved·RSS·MemAvailable은batch_B/summary.json에보존한다. 가중치SHA256은root summary.json,데이터릴리스·다운로드URL·SHA256은source필드에있다. 입력토큰및서열·ID를보존했다.','','표의시간은cached CPU tokens의GPU전송·공식forward(사용하지않는logits계산도포함)·평균임베딩pooling·CPU반환을포함하고전처리/로딩/warmup/파일저장은제외한다. 동기화후측정하며최소2회·15초를확보한다. warmup별시간은각worker summary에있다. GPU-Util은장치전체활동시간비율,allocated/reserved는PyTorch프로세스allocator값이다. 메모리비율은전체CPU/GPU공유RAM121.688GiB대비참고값이며전용VRAM비율이아니다. RSS와CUDA를더하지않는다.','','|배치/고유서열|측정반복|총처리건수|초|서열/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|','|---|---|---|---|---|---|---|---|---|---|---|---|---|'];flat=[]
for r in s['conditions']:
 m=r['memory'];b=r['batch_size'];base=r['model_only_allocated_gib'];row={'batch':b,'repeats':r['repeats'],'processed_inputs':r['processed_inputs'],'seconds':r['seconds'],'sequences_per_second':r['sequences_per_second'],'model_loaded_allocated_gib':base,'peak_allocated_gib':m['peak_allocated_gib'],'additional_from_same_run_load_gib':m['peak_allocated_gib']-base,'peak_reserved_gib':m['peak_reserved_gib'],'fraction_shared_ram_percent':100*r['allocated_fraction_shared_ram'],'gpu_util_mean':r['gpu_util_avg'],'gpu_util_max':r['gpu_util_max'],'rss_sampled_peak_gib':r['rss_sampled_peak_gib'],'available_sampled_min_gib':r['available_sampled_min_gib']};flat.append(row);lines.append('|'+ '|'.join(str(round(v,3)) if isinstance(v,float) else str(v) for v in row.values())+'|')
lines+=['','## 결과','',f"시험한 최대 성공배치 **{s.get('largest_successful_batch')}**. allocated80%도달: {s.get('reached80')},90%도달: {s.get('reached90')}. 이는관측조건의결과이며모델의절대최대한계가아니다.",'','시스템가용RAM16GiB·swap증가2GiB guard와다음배치예약량예측으로확대를중단한다. worker실패가있으면그조건의exit/log를보존하며성공행에포함하지않는다. 예상guard에의해다음배치를건너뛴것은그배치의실제실패나90%불가능증명이아니다. RSS/가용량은0.5초대기+조회시계열의sampled값이다. 메모리할당peak는CUDAcounter이다.','','각조건embedding shape B×1280·finite·저장후정확재로드를확인했고,같은첫서열의batch1대비임베딩일치도도검사했다. 학습·정확도평가·독립생물학적일반화평가를수행한결과가아니다. 일반조건인토큰예산4096경로와다른명시적batch용량실험이므로기본성능결과를대체하지않는다.','','[공식ESM](https://github.com/facebookresearch/esm) · [UniProt 데이터출처]('+s['source']['url']+') · [기본보고서](ESM2_20261027_164300_125163.md)', '',f"원본: ../results/{D.name}/summary.json, batch_B/summary.json, resources.csv, embeddings.pt, tokenized_inputs.pt, 명령·exit·로그. 재현코드는scripts/batch_followup에있다."]
(R/'reports/ESM2_batch_memory.md').write_text('\n'.join(lines)+'\n')
if flat:
 with (R/'reports/batch_memory_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
root=R.parent
with (root/'STATUS.md').open('a') as f:f.write('\nESM-2 후속 긴서열배치추론 '+D.name+' 상태 '+s['status']+', 최대성공batch '+str(s.get('largest_successful_batch'))+', 종료사유 '+s.get('stop_reason','running')+'. 보고서 esm2/reports/ESM2_batch_memory.md.\n')

if s['status'] in ['complete','stopped']:
 for name in ['RESULTS.md','MEMORY_RESULTS.md']:
  p=root/name;text=p.read_text();text=text.replace('ESM-2 긴 실제서열900~950AA 배치확대 **추론** 실험 진행중.','ESM-2 긴 실제서열900~950AA 배치확대 **추론** 실험 종료.').replace('완료후 [배치 메모리 보고서]','결과는 [배치 메모리 보고서]')
  text+='\nESM-2 후속 최종 상태 '+s['status']+', 최대성공batch '+str(s.get('largest_successful_batch'))+',80%도달 '+str(s.get('reached80'))+',90%도달 '+str(s.get('reached90'))+'. [상세](esm2/reports/ESM2_batch_memory.md). 새로운 다른 모델 실험은 자동 시작하지 않는다.\n';p.write_text(text)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>esm2/scripts/batch_followup/worker.py — 원문 전문 (3,958 bytes)</summary>


<!-- RESTORE_FILE {"path": "esm2/scripts/batch_followup/worker.py", "sha256": "02bf8c3072ca143a9b0765278e62cf1380434c78a240008cf214e0a59ab02f89", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,json,time,gc,os,hashlib,platform,shutil
import numpy as np,torch,esm,psutil
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'));from gpu_support import Meter,write_json
R=ROOT/'esm2';D=Path(sys.argv[1]);B=int(sys.argv[2]);C=D/f'batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','batch':B,'task':'pretrained protein mean embeddings inference only','conditions':[]};write_json(C/'summary.json',S)
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='model_loading';t=time.time();data=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False);reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',data,reg);del data,reg;gc.collect();model.float().cuda().eval();torch.cuda.synchronize();S['model_loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for p in model.parameters());S['parameters_backbone']=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('lm_head.','contact_head.')));S['parameters_trainable_in_this_run']=0;S['environment']={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'esm':esm.__version__,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform()};write_json(C/'summary.json',S)
 records=json.loads((D/'inputs.json').read_text())[:B];assert len(records)==B;recs=[(r['id'],r['sequence']) for r in records];convert=alphabet.get_batch_converter(1022);_,seqs,tokens=convert(recs);assert tokens.shape==(B,952);torch.save({'ids':[i for i,s in recs],'tokens':tokens},C/'tokenized_inputs.pt');S['input']={'unique_sequences':B,'length_min':min(map(len,seqs)),'length_max':max(map(len,seqs)),'length_mean':float(np.mean(list(map(len,seqs)))),'actual_token_shape':list(tokens.shape),'dtype':str(tokens.dtype),'truncation':False,'padding':'official converter, batch longest950AA + BOS/EOS =952 positions'}
 def forward():
  x=tokens.cuda();out=model(x,repr_layers=[33],return_contacts=False);h=out['representations'][33];e=torch.stack([h[i,1:len(seq)+1].mean(0) for i,seq in enumerate(seqs)]).cpu();assert torch.isfinite(e).all();return e
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():embedding=forward()
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del embedding;gc.collect();t,base=meter.begin('embedding_inference');steps=0
 with torch.inference_mode():
  while steps<2 or time.time()-t<15:
   embedding=forward();torch.cuda.synchronize();steps+=1
 r=meter.end(t,base,batch_size=B,unique_inputs=B,processed_inputs=B*steps,repeats=steps,input_shape=list(tokens.shape));r['sequences_per_second']=B*steps/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];S['conditions']=[r];torch.save({'ids':[i for i,s in recs],'embeddings':embedding},C/'embeddings.pt');saved=torch.load(C/'embeddings.pt',map_location='cpu',weights_only=True);assert saved['embeddings'].shape==(B,1280) and torch.equal(saved['embeddings'],embedding);S['saved_output_exact_reload']=True
 ref=D/'batch_1/embeddings.pt'
 if B>1:
  reference=torch.load(ref,map_location='cpu',weights_only=True);diff=float((reference['embeddings'][0]-embedding[0]).abs().max());assert torch.allclose(reference['embeddings'][0],embedding[0],atol=1e-4,rtol=1e-4);S['first_sequence_batch_invariance_max_abs']=diff
 S['status']='complete';write_json(C/'summary.json',S);print('BATCH_COMPLETE',B,round(r['memory']['peak_allocated_gib'],3),round(r['sequences_per_second'],3),flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write_json(C/'summary.json',S);raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>esm2/scripts/benchmark.py — 원문 전문 (15,891 bytes)</summary>


<!-- RESTORE_FILE {"path": "esm2/scripts/benchmark.py", "sha256": "20863f34231101de503f46b8b5ef61c1ea1c775eb0ff7ebb49320c5548f9e94d", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Sequential ESM-2 benchmark; no changes to shared base environment."""
import os,sys,time,json,pathlib,datetime,random,hashlib,threading,subprocess,csv,pickle,traceback,platform,shutil
import numpy as np,torch,esm,psutil,lmdb
from scipy.stats import spearmanr
R=pathlib.Path(__file__).resolve().parents[1]
RUN=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'); O=R/'results'/RUN; O.mkdir(parents=True)
(R/'results'/'LATEST').write_text(RUN+'\n')
shutil.copy2(__file__,O/'benchmark.py'); shutil.copy2(R/'data/sources.json',O/'sources.json')
def write(name,obj): (O/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
def log(*s): print(datetime.datetime.now().isoformat(),*s,flush=True)
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
phase='setup'; stop=threading.Event(); samples=[]; proc=psutil.Process()
def monitor():
 with (O/'resources.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['time_unix','phase','gpu_util_percent','rss_bytes','system_available_bytes','swap_used_bytes']); w.writeheader()
  while not stop.is_set():
   start=time.time(); ph=phase
   try:
    q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
    util=float(q.stdout.strip().splitlines()[0]) if q.returncode==0 and q.stdout.strip().splitlines()[0].strip().isdigit() else None
   except Exception: util=None
   vm=psutil.virtual_memory(); row=dict(time_unix=start,phase=ph,gpu_util_percent=util,rss_bytes=proc.memory_info().rss,system_available_bytes=vm.available,swap_used_bytes=psutil.swap_memory().used)
   samples.append(row); w.writerow(row);f.flush(); stop.wait(max(0,.5-(time.time()-start)))
th=threading.Thread(target=monitor,daemon=True);th.start()
def stats(ph):
 rows=[s for s in samples if s['phase']==ph]; u=[s['gpu_util_percent'] for s in rows if s['gpu_util_percent'] is not None]
 return dict(sample_count=len(rows),gpu_util_mean_percent=float(np.mean(u)) if u else None,gpu_util_max_percent=max(u) if u else None,rss_sampled_peak_gib=max([s['rss_bytes'] for s in rows],default=0)/2**30,system_available_min_gib=min([s['system_available_bytes'] for s in rows],default=0)/2**30,swap_max_bytes=max([s['swap_used_bytes'] for s in rows],default=0))
def guard():
 if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>initial_swap+2*2**30: raise RuntimeError('Memory pressure: halt input expansion')
initial_swap=psutil.swap_memory().used
summary={'run_id':RUN,'status':'running','settings':{'checkpoint':'esm2_t33_650M_UR50D','precision':'FP32, TF32 disabled','seed':42,'token_budget':4096,'batch_policy':'official FastaBatchedDataset.get_batch_indices(extra_toks_per_seq=1)','length_limit':1022,'sampling_seconds':.5,'inference_repeats':3,'threads':8},'conditions':[]}
write('summary.json',summary)
try:
 # Inspect actual files before applying the model.
 sequences=[]; ids=[]
 for line in (R/'data/uniprot_reviewed.fasta').read_text().splitlines():
  if line.startswith('>'): ids.append(line[1:].split()[0]);sequences.append('')
  else: sequences[-1]+=line.strip()
 seen=set(); records=[]
 for i,s in zip(ids,sequences):
  if s not in seen: records.append((i,s));seen.add(s)
 bins=[(50,128),(129,256),(257,512),(513,1022)]; groups=[]
 for a,b in bins:
  g=[r for r in records if a<=len(r[1])<=b];random.Random(SEED).shuffle(g);groups.append(g)
 nbin=min(40,min(map(len,groups))); countset=[g[j] for j in range(nbin) for g in groups]
 ds={}; inputs={'uniprot':{'rows':len(sequences),'unique_sequences':len(records),'duplicate_sequences_removed':len(sequences)-len(records),'length_min':min(map(len,sequences)),'length_max':max(map(len,sequences)),'characters':sorted(set(''.join(sequences))),'labels':None,'dtype':'FASTA text; tokenized torch.int64','preprocessing':'deduplicate exact sequences, no truncation necessary','bins':[{ 'range':[a,b],'available_unique':len(g)} for (a,b),g in zip(bins,groups)]},'fluorescence':{}}
 for split in ['train','valid','test']:
  env=lmdb.open(str(R/'data/fluorescence'/('fluorescence_'+split+'.lmdb')),readonly=True,lock=False)
  with env.begin() as tx:
   n=pickle.loads(tx.get(b'num_examples')); allrows=[pickle.loads(tx.get(str(i).encode())) for i in range(n)]
  env.close(); assert all(np.isfinite(r['log_fluorescence']).all() and len(r['primary'])==r['protein_length'] for r in allrows)
  assert all(set(r['primary'])<=set('ACDEFGHIKLMNPQRSTVWY') for r in allrows)
  ds[split]=allrows; inputs['fluorescence'][split]=dict(rows=n,unique=len({r['primary'] for r in allrows}),length_min=min(len(r['primary']) for r in allrows),length_max=max(len(r['primary']) for r in allrows),label_dtype=str(allrows[0]['log_fluorescence'].dtype),label_shape=list(allrows[0]['log_fluorescence'].shape),label_mean=float(np.mean([r['log_fluorescence'][0] for r in allrows])),label_min=float(min(r['log_fluorescence'][0] for r in allrows)),label_max=float(max(r['log_fluorescence'][0] for r in allrows)),missing_or_nonfinite_labels=0)
 for a,b in [('train','valid'),('train','test'),('valid','test')]:
  overlap=len({r['primary'] for r in ds[a]}&{r['primary'] for r in ds[b]}); assert overlap==0;inputs['fluorescence'][a+'_'+b+'_exact_overlap']=overlap
 write('input_report.json',inputs);log('INPUT VERIFIED',inputs['uniprot'])
 write('environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,esm='fair-esm 2.0.0',host=platform.node(),platform=platform.platform(),gpu=torch.cuda.get_device_name(0),cuda_available=torch.cuda.is_available(),system_memory_bytes=psutil.virtual_memory().total,initial_available_bytes=psutil.virtual_memory().available,initial_swap_bytes=initial_swap,gpu_processes=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,docker='Permission denied for current user; host isolated venv used'))
 (O/'pip_freeze.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout)
 phase='model_loading'; t=time.perf_counter()
 # Official FAIR checkpoint is a trusted pickle from the recorded HTTPS source;
 # explicit weights_only=False supports its argparse.Namespace cfg on torch>=2.6.
 model_data=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False)
 reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True)
 model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',model_data,reg);del model_data,reg
 model=model.cuda().eval();torch.cuda.synchronize();summary['model_load_seconds']=time.perf_counter()-t
 summary['model_parameters']=sum(p.numel() for p in model.parameters());summary['embedding_dimension']=model.embed_dim;summary['layers']=model.num_layers
 converter=alphabet.get_batch_converter(1022)
 def batches(recs):
  data=esm.FastaBatchedDataset([r[0] for r in recs],[r[1] for r in recs]);return [[recs[i] for i in ix] for ix in data.get_batch_indices(4096,extra_toks_per_seq=1)]
 def embed(recs):
  _,strings,tok=converter(recs); tok=tok.cuda(); out=model(tok,repr_layers=[33],return_contacts=False)['representations'][33]
  return torch.stack([out[i,1:len(s)+1].mean(0) for i,s in enumerate(strings)]),tok.shape
 phase='warmup';t=time.perf_counter()
 with torch.no_grad():
  smoke,shape=embed(countset[:2]);torch.cuda.synchronize();assert smoke.shape==(2,1280) and torch.isfinite(smoke).all(); torch.save({'ids':[r[0] for r in countset[:2]],'embeddings':smoke.cpu()},O/'smoke.pt')
 summary['warmup_seconds']=time.perf_counter()-t;write('smoke.json',{'passed':True,'output_shape':list(smoke.shape),'token_shape':list(shape),'device':str(smoke.device),'dtype':str(smoke.dtype)});del smoke
 log('SMOKE PASSED; parameters',summary['model_parameters'])
 def inference(name,recs):
  global phase
  guard();phase=name;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();twall=time.perf_counter();t=time.perf_counter(); bs=batches(recs);prep=time.perf_counter()-t
  write(name+'_inputs.json',[dict(id=i,sequence=s,original_residues=len(s),actual_residues=min(1022,len(s)),truncated=len(s)>1022) for i,s in recs])
  durations=[];bmeta=[];save=0
  with torch.no_grad():
   for repeat in range(3):
    torch.cuda.synchronize();t=time.perf_counter();em=[];order=[]
    for b in bs:
     e,shape=embed(b);em.append(e.cpu());order.extend(r[0] for r in b)
     if repeat==0:bmeta.append(dict(samples=len(b),token_shape=list(shape),nonpadding_tokens=sum(len(r[1])+2 for r in b),padded_tokens=shape[0]*shape[1],residue_lengths=[len(r[1]) for r in b]))
    torch.cuda.synchronize();durations.append(time.perf_counter()-t);t=time.perf_counter();values=torch.cat(em);assert torch.isfinite(values).all();torch.save(dict(ids=order,embeddings=values),O/(name+'_r'+str(repeat)+'.pt'));save+=time.perf_counter()-t
    guard()
  phase='between_conditions';time.sleep(.6)
  row=dict(condition=name,mode='inference',unique_inputs=len(recs),processed_inputs=3*len(recs),repeats=3,length_min=min(len(r[1]) for r in recs),length_max=max(len(r[1]) for r in recs),length_mean=float(np.mean([len(r[1]) for r in recs])),preprocess_seconds=prep,execution_seconds=sum(durations),repeat_seconds=durations,save_seconds=save,wall_seconds=time.perf_counter()-twall,throughput_per_second=3*len(recs)/sum(durations),cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**stats(name))
  write(name+'_batches.json',bmeta);write(name+'_metrics.json',row);summary['conditions'].append(row);write('summary.json',summary);log('DONE',name,row)
 for n in [32,96,160]: inference('count_'+str(n),countset[:n])
 for (a,b),g in zip(bins,groups): inference('length_'+str(a)+'_'+str(b),g[:16])
 # Genuine, limited backbone fine-tuning: final transformer block + final norm + regression head.
 selections={}
 for split,n in [('train',128),('valid',64),('test',128)]:
  ix=random.Random(SEED).sample(range(len(ds[split])),n);selections[split]=[dict(id=split+'_'+str(i),sequence=ds[split][i]['primary'],label=float(ds[split][i]['log_fluorescence'][0]),source_index=i,num_mutations=int(ds[split][i]['num_mutations'])) for i in ix]
 write('training_inputs.json',selections)
 for p in model.parameters():p.requires_grad_(False)
 for p in model.layers[-1].parameters():p.requires_grad_(True)
 for p in model.emb_layer_norm_after.parameters():p.requires_grad_(True)
 head=torch.nn.Linear(1280,1).cuda(); trainparams=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW([dict(params=trainparams,lr=1e-4),dict(params=head.parameters(),lr=1e-3)],weight_decay=.01)
 mu=float(np.mean([r['label'] for r in selections['train']]));std=float(np.std([r['label'] for r in selections['train']]));assert std>0
 def evaluate(split):
  model.eval();head.eval();preds=[];truth=[];idx=[]
  records=[(r['id'],r['sequence']) for r in selections[split]];lookup={r['id']:r['label'] for r in selections[split]}
  with torch.no_grad():
   for b in batches(records):
    e,_=embed(b);p=head(e).squeeze(-1).cpu().numpy()*std+mu;preds.extend(map(float,p));truth.extend(lookup[i] for i,s in b);idx.extend(i for i,s in b)
  pred=np.asarray(preds);truth=np.asarray(truth);metrics=dict(n=len(pred),mse=float(np.mean((pred-truth)**2)),mae=float(np.mean(np.abs(pred-truth))),spearman=float(spearmanr(pred,truth).statistic));return dict(ids=idx,predictions=preds,targets=truth.tolist(),metrics=metrics)
 phase='baseline_validation';baseline=evaluate('valid');write('baseline_validation.json',baseline)
 before=model.layers[-1].fc1.weight.detach().clone();head_before=head.weight.detach().clone()
 phase='finetuning';torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();losses=[]
 model.train();head.train();records=[(r['id'],r['sequence']) for r in selections['train']];lookup={r['id']:r['label'] for r in selections['train']};bs=batches(records);random.Random(SEED).shuffle(bs)
 for step,b in enumerate(bs):
  guard();opt.zero_grad(set_to_none=True);e,_=embed(b);target=torch.tensor([(lookup[i]-mu)/std for i,s in b],device='cuda',dtype=torch.float32);loss=torch.nn.functional.mse_loss(head(e).squeeze(-1),target);assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()));log('TRAIN',step+1,'/',len(bs),'loss',losses[-1])
 torch.cuda.synchronize();trainsec=time.perf_counter()-t
 changed=not torch.equal(before,model.layers[-1].fc1.weight);headchanged=not torch.equal(head_before,head.weight);assert changed and headchanged;del before,head_before
 trainmetric=dict(mode='partial_backbone_finetuning',epochs=1,train_inputs=128,validation_inputs=64,test_inputs=128,trainable_backbone_parameters=sum(p.numel() for p in trainparams),head_parameters=sum(p.numel() for p in head.parameters()),trainable_total=sum(p.numel() for p in trainparams)+sum(p.numel() for p in head.parameters()),complete_model_parameters=summary['model_parameters']+sum(p.numel() for p in head.parameters()),execution_seconds=trainsec,throughput_per_second=128/trainsec,cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,backbone_updated=changed,head_updated=headchanged,losses=losses)
 phase='validation';time.sleep(.6);trainmetric.update(stats('finetuning'));validation=evaluate('valid');write('validation.json',validation)
 phase='saving_model';t=time.perf_counter();M=R/'models'/'finetuned'/RUN;M.mkdir(parents=True)
 adapter={k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith('layers.32.') or k.startswith('emb_layer_norm_after.')}
 checkpoint=dict(backbone_updates=adapter,head=head.state_dict(),target_mean=mu,target_std=std,base_checkpoint='esm2_t33_650M_UR50D.pt',base_sha256=json.loads((R/'data/sources.json').read_text())[2]['sha256'],epoch=1,seed=42,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),training_scope='final transformer block + final layer norm + linear regression head',resume_training_verified=False)
 torch.save(checkpoint,M/'checkpoint.pt');trainmetric['save_seconds']=time.perf_counter()-t;trainmetric['model_path']=str(M)
 (M/'metadata.json').write_text(json.dumps({k:v for k,v in checkpoint.items() if k not in ['backbone_updates','head','optimizer','torch_rng','cuda_rng','python_rng']},indent=2))
 phase='test_original';original=evaluate('test');write('test_original.json',original)
 # Perturb updated weights before reloading, proving the stored state is used.
 with torch.no_grad():model.layers[-1].fc1.weight.zero_();head.weight.zero_()
 reloaded=torch.load(M/'checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(reloaded['backbone_updates'],strict=False);head.load_state_dict(reloaded['head']);mu=reloaded['target_mean'];std=reloaded['target_std']
 phase='test_reloaded';retest=evaluate('test');write('test_reloaded.json',retest)
 delta=float(np.max(np.abs(np.asarray(original['predictions'])-np.asarray(retest['predictions']))));assert delta<=1e-5
 trainmetric.update(reload_verified=True,reload_max_abs_difference=delta,independent_test_evaluated=True,test_metrics=retest['metrics'],validation_metrics=validation['metrics'],initial_validation_metrics=baseline['metrics'])
 write('training_metrics.json',trainmetric);summary['training']=trainmetric;summary['status']='success';summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write('summary.json',summary);log('SUCCESS',O)
except Exception as exc:
 summary['status']='failed';summary['error']=str(exc);summary['traceback']=traceback.format_exc();write('summary.json',summary);log(traceback.format_exc());raise
finally:
 stop.set();th.join(timeout=5)
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "esm2/results/20261027_164300_125163/benchmark.py", "source": "esm2/scripts/benchmark.py", "sha256": "20863f34231101de503f46b8b5ef61c1ea1c775eb0ff7ebb49320c5548f9e94d"} -->

</details>

<details>
<summary>esm2/scripts/measure_loading.py — 원문 전문 (1,578 bytes)</summary>


<!-- RESTORE_FILE {"path": "esm2/scripts/measure_loading.py", "sha256": "1c021da223b492c31233ca4c23267eb7ffc4a61b5616fcd0257bf8477e53d5a1", "encoding": "utf-8", "added_final_newline": false} -->
```python
import pathlib,json,time,datetime,torch,esm,psutil,platform
R=pathlib.Path(__file__).resolve().parents[1];O=R/'results'/('loading_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));O.mkdir();(R/'results/LOADING_LATEST').write_text(O.name+'\n');torch.cuda.init();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();proc=psutil.Process();before=proc.memory_info().rss;t=time.perf_counter();data=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False);reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',data,reg);del data,reg;model=model.cuda().eval();torch.cuda.synchronize();row=dict(checkpoint='esm2_t33_650M_UR50D',total_parameters=sum(p.numel() for p in model.parameters()),parameter_dtype=str(next(model.parameters()).dtype),torch=torch.__version__,cuda=torch.version.cuda,host=platform.node(),load_seconds=time.perf_counter()-t,cuda_after_load_allocated_gib=torch.cuda.memory_allocated()/2**30,cuda_after_load_reserved_gib=torch.cuda.memory_reserved()/2**30,cuda_loading_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,rss_before_load_gib=before/2**30,rss_after_load_gib=proc.memory_info().rss/2**30,system_available_gib=psutil.virtual_memory().available/2**30,scope='Fresh process, CUDA initialized before load; original pretrained weights only; no tokenization, forward or optimizer');(O/'loading.json').write_text(json.dumps(row,indent=2));print(O);print(json.dumps(row,indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>esm2/scripts/predict.py — 원문 전문 (2,224 bytes)</summary>


<!-- RESTORE_FILE {"path": "esm2/scripts/predict.py", "sha256": "1ff266dbfc0eebeb26c4e8f9b6995529c8c75b59b25c43c70e1ba10bd19bda9b", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Reconstruct saved partial ESM-2 fine-tune and predict fluorescence for a FASTA."""
import argparse,csv,hashlib,json,pathlib,torch,esm
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=pathlib.Path);p.add_argument('--fasta',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--device',default='cuda');a=p.parse_args()
r=next(p for p in pathlib.Path(__file__).resolve().parents if (p/'models/esm2_t33_650M_UR50D.pt').is_file());ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
h=hashlib.sha256()
with (r/'models'/ck['base_checkpoint']).open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
assert h.hexdigest()==ck['base_sha256'],'Base checkpoint hash mismatch'
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
base=torch.load(r/'models'/ck['base_checkpoint'],map_location='cpu',weights_only=False);reg=torch.load(r/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',base,reg);del base,reg
result=model.load_state_dict(ck['backbone_updates'],strict=False);assert not result.unexpected_keys
model.to(a.device).eval();head=torch.nn.Linear(1280,1).to(a.device);head.load_state_dict(ck['head']);head.eval();conv=alphabet.get_batch_converter(1022);ds=esm.FastaBatchedDataset.from_file(a.fasta)
assert all(set(s)<=set('ACDEFGHIKLMNPQRSTVWY') and len(s)>0 for s in ds.sequence_strs)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as f,torch.no_grad():
 w=csv.writer(f);w.writerow(['id','original_length','actual_residues','truncated','predicted_log_fluorescence'])
 for ix in ds.get_batch_indices(4096,extra_toks_per_seq=1):
  ids,ss,tok=conv([ds[i] for i in ix]);o=model(tok.to(a.device),repr_layers=[33],return_contacts=False)['representations'][33];e=torch.stack([o[i,1:min(len(s),1022)+1].mean(0) for i,s in enumerate(ss)]);pred=(head(e).squeeze(-1)*ck['target_std']+ck['target_mean']).cpu().tolist()
  for i,s,v in zip(ids,ss,pred):w.writerow([i,len(s),min(len(s),1022),len(s)>1022,v])
print(a.output)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>esm2/scripts/prepare.py — 원문 전문 (1,851 bytes)</summary>


<!-- RESTORE_FILE {"path": "esm2/scripts/prepare.py", "sha256": "29f09e4a7d6cc687bbb14ddd3a2fd8cb8f7cfec87e2c61d682b3b57238c6a154", "encoding": "utf-8", "added_final_newline": false} -->
```python
import datetime,hashlib,json,pathlib,ssl,urllib.request,tarfile,shutil
R=pathlib.Path('/home/hyein/ai-benchmark/esm2')
ctx=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt')
sources={
 'data/uniprot_reviewed.fasta':'https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20length%3A%5B50%20TO%201022%5D&format=fasta&size=500',
 'data/fluorescence.tar.gz':'https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz',
 'models/esm2_t33_650M_UR50D.pt':'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt',
 'models/esm2_t33_650M_UR50D-contact-regression.pt':'https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt',
 'data/official_extract.py':'https://raw.githubusercontent.com/facebookresearch/esm/main/scripts/extract.py',
}
manifest=[]
for name,url in sources.items():
 p=R/name
 if not p.exists():
  print('DOWNLOAD',url,flush=True)
  with urllib.request.urlopen(url,context=ctx,timeout=180) as src,p.with_suffix(p.suffix+'.part').open('wb') as dst:
   headers=dict(src.headers); shutil.copyfileobj(src,dst)
  p.with_suffix(p.suffix+'.part').rename(p)
 else: headers={}
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 manifest.append(dict(path=name,url=url,bytes=p.stat().st_size,sha256=h.hexdigest(),headers=headers,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 (R/'data/sources.json').write_text(json.dumps(manifest,indent=2))
 print(name,p.stat().st_size,flush=True)
with tarfile.open(R/'data/fluorescence.tar.gz') as tf:
 for m in tf.getmembers():
  dest=(R/'data'/m.name).resolve()
  if not dest.is_relative_to((R/'data').resolve()) or m.issym() or m.islnk(): raise ValueError(m.name)
 tf.extractall(R/'data')
print('PREPARE DONE',flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>esm2/scripts/report.py — 원문 전문 (23,131 bytes)</summary>


<!-- RESTORE_FILE {"path": "esm2/scripts/report.py", "sha256": "13e34868ad22d1e2c0e83c3fb84e93c10ed055452992983f57b0bb73a2d44b0b", "encoding": "utf-8", "added_final_newline": false} -->
````python
"""Create the Korean ESM-2 report from measured files; no GPU work."""
from pathlib import Path
import json,csv,datetime,hashlib,shutil,statistics
R=Path(__file__).resolve().parents[1];B=R/'results/20261027_164300_125163';U=R/'results'/(R/'results/SUPPLEMENT_LATEST').read_text().strip();s=json.loads((B/'summary.json').read_text());u=json.loads((U/'summary.json').read_text());assert s['status']==u['status']=='success'
t=s['training'];env=json.loads((B/'environment.json').read_text());inputs=json.loads((B/'input_report.json').read_text());src=json.loads((B/'sources.json').read_text());M=Path(t['model_path']);report=R/'reports'/('ESM2_'+s['run_id']+'.md');out=[]
def add(x=''):out.append(x)
def f(x,n=3):return 'N/A' if x is None else format(x,f'.{n}f')
def table(headers,rows):
 add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
 for row in rows:add('| '+' | '.join(map(str,row))+' |')
 add()
add('# DGX Spark ESM-2 성능 검증 보고서');add();add(f'작성: {datetime.datetime.now().isoformat(timespec="seconds")} (KST). 기본 run: `{B.name}`, 보완 run: `{U.name}`.');add()
add('## 결과와 완료 범위');add();add('실제 GB10 GPU에서 ESM-2 650M 임베딩 추론, 입력 개수·서열 길이 비교, 공개 fluorescence 라벨 기반 부분 파인튜닝, 저장, 독립 프로세스 재로드 및 test 예측 검증을 완료했다. 기본 작업은 중단 중에도 계속 실행돼 2026-10-27 16:45:26 KST에 종료 코드 0으로 끝났다. 기존 실행을 중복 시작하지 않았고, 보완 측정은 기본 작업 종료를 확인한 후 별도 폴더에서 순차 실행했다.');add();add('기본 개수 실험에서 평균 GPU-Util 89.4–92.5%가 관찰됐다. 80%를 맞추기 위한 배치·정밀도·모델 변경은 하지 않았다. 모델 실행·저장 성공과 예측 정확도는 구분해야 한다. 128개 test의 Spearman은 0.0241로 매우 낮아 실용적인 fluorescence 예측 성능을 입증하지 못했다.');add()
add('## 모델·환경·기준 조건');add();table(['항목','확인값'],[
 ['체크포인트','esm2_t33_650M_UR50D, FAIR 공식 가중치'],['공식 구현','fair-esm 2.0.0'],['구조','Transformer 33층, hidden 1280, attention heads 20, residue 평균 pooling'],['전체 추론 구성 파라미터',f"{s['model_parameters']:,} (LM/contact head 포함)"],['encoder 파라미터',f"{u['encoder_parameters_excluding_lm_contact_heads']:,} (tied token embedding 포함; LM/contact의 고유 head 파라미터 제외)"],['회귀 구성 전체 파라미터',f"{t['complete_model_parameters']:,} (기존 모델 + 회귀 head)"],['학습 가능 backbone / 회귀 head',f"{t['trainable_backbone_parameters']:,} / {t['head_parameters']:,}"],['학습 가능 합계',f"{t['trainable_total']:,}"],['장비','spark-e4b6, NVIDIA GB10, ARM64'],['OS','Ubuntu 24.04.5 LTS, Linux 7.0.0-1019-nvidia'],['Python / PyTorch / CUDA','3.10.21 / 2.14.1+cu130 / 13.0'],['NVIDIA 드라이버','580.178.04'],['시스템 물리 RAM',f"{env['system_memory_bytes']:,} bytes ({env['system_memory_bytes']/2**30:.3f} GiB)"],['배치 정책','공식 FastaBatchedDataset: token budget 4096, extra_toks_per_seq=1'],['정밀도·seed·CPU threads','FP32, TF32 disabled, seed 42, threads 8'],['모델 입력','torch.int64 [B, max_residues+2], BOS/EOS와 padding'],['길이 제한','1022 residues, 실제 비교 입력은 모두 제한 이내'],['실행 방식','호스트 venv, 기존 DNABERT-2 환경의 system-site-packages를 읽어 PyTorch 재사용'],['기존 환경 보존','기반 환경 수정 없음; 새 venv에 fair-esm, lmdb 설치'],['컨테이너','Docker 접근 거부로 사용하지 않음; 모든 결과는 호스트 경로에 저장']]);add('토큰 예산은 공식 배치 구성 함수의 추정치다. 함수가 extra token을 1개로 계산하지만 ESM-2 변환은 BOS/EOS 2개를 붙이므로 실제 padded token은 일부 배치에서 4,102개였다. 예산과 실제 shape를 구분해 저장했다.');add()
add('## 입력 확인과 데이터 출처');add();add('UniProtKB reviewed release 2026_03 (2026-09-02), API의 첫 500개 FASTA 레코드를 확보했다. 전체 UniProt 또는 무작위 종 대표 표본이 아니다. 원본 500개, 고유 서열 499개, 길이 56–1020 residues, 표준 아미노산 20종, 빈 서열 없음이다. 동일 서열 1개는 개수 실험에서 제거했다. 라벨이 없는 임베딩 실험 입력이다.');add();add('개수 비교는 50–128 / 129–256 / 257–512 / 513–1022의 4개 길이 구간에서 각각 seed 42로 섞어 교차 배치한 중첩 부분집합 32/96/160개다. 구간별 개수는 조건마다 각각 8/24/40개로 같지만 구간 내부 길이 분포와 최대 길이는 완전히 일치하지 않는다. 개수 조건의 평균 길이는 328.28/341.73/340.82여서 처리량 차이를 순수한 개수 효과로 단정할 수 없다. 길이별 입력 목록, 실제 batch shape, padding을 저장했다.');add()
add('TAPE 공식 fluorescence LMDB를 사용했다. 단백질 GFP 변이 서열에서 float32 [1] log_fluorescence를 예측하는 회귀 과제다. 분류 정확도 지표를 사용하지 않았다.');add();table(['원본 split','레코드','길이 범위','라벨 평균','누락/비유한 라벨'],[[k,inputs['fluorescence'][k]['rows'],f"{inputs['fluorescence'][k]['length_min']}–{inputs['fluorescence'][k]['length_max']}",f(inputs['fluorescence'][k]['label_mean']),0] for k in ['train','valid','test']]);add('원본 split 간 동일 서열 중복은 모두 0이다. 기존 split을 유지한 채 train 128 / validation 64 / test 128개를 각 split에서 독립적으로 seed 42로 선정했다. test로 하이퍼파라미터나 epoch를 선택하지 않았다. GFP 변이들이기 때문에 높은 상동성은 남으며, 독립 단백질 family 일반화를 검증하지 않는다. 사전학습 데이터와의 중복도 검증하지 않았다.');add()
ex=json.loads((B/'count_32_inputs.json').read_text())[0];smoke=json.loads((B/'smoke.json').read_text());add('### 실제 입력·출력 예');add();add('```text');add('>'+ex['id']);add(ex['sequence'][:80]+'...');add(f"original residues = {ex['original_residues']}; truncated = {ex['truncated']}");add(f"smoke tokens = {smoke['token_shape']}; pooled embeddings = {smoke['output_shape']} float32 on CUDA");add('fluorescence: amino-acid sequence → mean pooled 1280-vector → Linear(1280,1) → log_fluorescence');add('```');add()
add('## 測정 범위와 방법'.replace('測','측'));add();add('전·후 torch.cuda.synchronize()를 호출했다. 추론 실행 시간에는 CPU 토큰화, CUDA 전송, forward, pooling, pooled 결과의 CPU 전송이 포함되며 모델 로딩, warmup, torch.save는 제외한다. preprocess_seconds는 batch 목록 구성 시간만 의미한다. 초기 다운로드와 전체 데이터 검사 시간은 별도 계측하지 않아 N/A다. 모델 로딩에는 체크포인트 읽기·모델 생성·CUDA 전송을 포함한다. 컴파일은 사용하지 않았다.');add();add(f"기본 run 모델 로딩 {s['model_load_seconds']:.3f}초, smoke/warmup {s['warmup_seconds']:.3f}초. 보완 run 모델 로딩 {u['model_load_seconds']:.3f}초, 각 길이 조건 전체 1 pass를 별도 warmup했다. 기본 개수 조건은 3회 반복, 보완 길이 조건은 동일 16개 입력을 40회 반복했다. 반복은 고유 데이터 추가가 아니다.");add();add('GPU-Util과 RAM은 0.5초 간격으로 표본 수집했다. GPU-Util은 장치 전체의 커널 실행 시간 비율로 최대 FLOPS/SM occupancy가 아니며 디스플레이·Firefox 등도 포함된다. nvidia-smi 자체 내부 집계 창과 조회 지연이 있어 독립 표본으로 해석하지 않는다. 보완 실험은 query 시작·종료가 모두 동기화된 실행 구간 안에 있는 표본만 포함하고 첫 2초를 제외한 평균도 별도 보관했다. 기본 run은 phase 단위이므로 매우 짧은 입력 목록·파일 저장 시간도 GPU/RAM 표본에 섞일 수 있다.');add();add('CUDA peak allocated/reserved는 PyTorch allocator 값이다. Spark는 CPU/GPU 공유 DRAM이므로 RSS와 CUDA 메모리를 더하지 않았다. RSS는 0.5초 sampled peak이며 정확한 최대값이 아니다. nvidia-smi 장치 전체 memory 사용량은 Not Supported이므로 N/A다. 시스템 MemAvailable과 swap은 별도 기록했다.');add()
add('## A. 서열 개수 증가: 원본 완료 결과');add();counts=[c for c in s['conditions'] if c['condition'].startswith('count_')];table(['고유 / 총 처리','평균 길이','실행 합계 s','평균 1회 s','서열/s','GPU 평균 / 최대 %'],[[f"{c['unique_inputs']} / {c['processed_inputs']}",f(c['length_mean'],2),f(c['execution_seconds']),f(statistics.mean(c['repeat_seconds'])),f(c['throughput_per_second']),f"{c['gpu_util_mean_percent']:.2f} / {c['gpu_util_max_percent']:.0f}"] for c in counts]);table(['고유 입력','CUDA alloc / reserved GiB','RSS 표본 최대 GiB','MemAvailable 최소 GiB','표본 수','swap bytes'],[[c['unique_inputs'],f"{c['cuda_peak_allocated_gib']:.3f} / {c['cuda_peak_reserved_gib']:.3f}",f(c['rss_sampled_peak_gib']),f(c['system_available_min_gib']),c['sample_count'],c['swap_max_bytes']] for c in counts]);table(['조건','batch 수/pass','batch 샘플 범위','실제 padded token 최대','batch 준비 s','저장 s','기록 wall s'],[[c['condition'],len(bs),f"{min(b['samples'] for b in bs)}–{max(b['samples'] for b in bs)}",max(b['padded_tokens'] for b in bs),f(c['preprocess_seconds'],6),f(c['save_seconds']),f(c['wall_seconds'])] for c in counts for bs in [json.loads((B/(c['condition']+'_batches.json')).read_text())]]);add('기본 condition wall에는 인위적인 표본 종료 대기 0.6초가 포함된다. 위 실행 시간과 처리량에는 이 대기를 포함하지 않는다. 전체 원본 프로세스 wall의 독립 perf-counter 계측은 N/A이며, run ID 시작 시각부터 종료 시각의 차이는 약 146초다. 다운로드·설치 전체 소요 시간으로 해석하지 않는다.');add();add('개수 증가에 따라 총 처리 시간이 증가했지만 GPU 평균은 약 89–92%로 이미 높다. 순간 allocated peak는 2.97→3.16 GiB로 완만히 변했다. 입력 최대 길이·batch packing·allocator 상태의 변화가 함께 작용하므로 개수만 늘려 메모리가 비례 증가했다고 설명하지 않는다.');add()
add('## B. 서열 길이 비교: 관측 구간을 늘린 보완 결과');add();add('각 조건 고유 16개, 측정 40회 × 16 = 640건, 별도 warmup 16건 제외. 토큰 예산·정밀도·checkpoint는 동일하고 토큰 예산 정책에 따라 실제 batch 샘플 수가 달라진다. 고정 batch size에서의 순수 길이 효과와는 구분한다.');add();table(['실제 길이 / 평균','실행 s','서열/s','GPU 평균 / 최대 %','첫 2초 제외 평균 %','표본 수'],[[f"{c['length_min']}–{c['length_max']} / {c['length_mean']:.2f}",f(c['execution_seconds']),f(c['throughput_per_second']),f"{c['gpu_util_mean_percent']:.2f} / {c['gpu_util_max_percent']:.0f}",f(c['gpu_util_steady_mean_percent'],2),c['sample_count']] for c in u['conditions']]);table(['길이 구간','CUDA alloc / reserved GiB','RSS sampled peak GiB','MemAvailable 최소 GiB','warmup s','저장 s','wall s'],[[c['condition'],f"{c['cuda_peak_allocated_gib']:.3f} / {c['cuda_peak_reserved_gib']:.3f}",f(c['rss_sampled_peak_gib']),f(c['system_available_min_gib']),f(c['warmup_seconds']),f(c['save_seconds']),f(c['wall_seconds'])] for c in u['conditions']]);add('길어질수록 처리량이 낮아졌다. 입력 내용이 다른 실제 단백질 그룹이므로 길이만 바꾼 동일 서열 실험은 아니다. 모든 입력은 truncation 없이 처리됐으며, pooling에서 BOS/EOS와 padding을 제외했다. 기본 3회 길이 측정도 보존했지만 56–127 구간은 실행 1.325초·표본 3개여서 GPU 평균 59.7%가 불안정했다. 보완 결과로 사용률 해석을 갱신했으며, 실행 시간을 인위적으로 배치 sweep으로 늘리지 않았다.');add()
add('## 부분 파인튜닝·검증·저장');add();add('fair-esm의 PyTorch ESM2 backbone에 사용자 정의 선형 회귀 head를 붙였다. 공식 패키지의 원클릭 fluorescence 학습 명령을 실행한 것이 아니다. 마지막 Transformer block(layers.32)과 최종 LayerNorm, head만 갱신하고 앞선 32개 block과 embedding은 동결했다. 전체 backbone fine-tuning으로 표현하지 않는다. train split의 라벨 평균·표준편차로만 표준화하고 예측 시 원 척도로 역변환했다.');add();table(['항목','결과'],[['train / validation / test','128 / 64 / 128'],['학습','1 epoch, 8 optimizer steps, single run'],['optimizer','AdamW, backbone lr=1e-4, head lr=1e-3, weight_decay=0.01'],['loss','표준화된 label의 MSE'],['학습 실행 시간 s',f(t['execution_seconds'])],['학습 처리량 서열/s',f(t['throughput_per_second'])],['GPU 평균 / 최대 %',f"{t['gpu_util_mean_percent']:.2f} / {t['gpu_util_max_percent']:.0f}"],['GPU 관측 표본',t['sample_count']],['CUDA alloc / reserved GiB',f"{t['cuda_peak_allocated_gib']:.3f} / {t['cuda_peak_reserved_gib']:.3f}"],['RSS sampled peak GiB',f(t['rss_sampled_peak_gib'])],['MemAvailable 최소 GiB',f(t['system_available_min_gib'])],['모델 저장 s',f(t['save_seconds'])],['backbone / head 가중치 갱신',f"{t['backbone_updated']} / {t['head_updated']}"],['동일 프로세스 재로드 차이',t['reload_max_abs_difference']],['새 프로세스 재구성 차이',u['fresh_reload']['max_abs_difference']],['학습 재개 검증','미수행 (optimizer, epoch, RNG 저장만 수행)']]);add('학습 구간은 7.2초로 비교적 짧고 표본 14개이므로 GPU 평균을 장기 학습의 확정값으로 사용하지 않는다. 입력 규모별 학습 비교는 수행하지 않았다. 학습 조건은 1개이며 이후 정확도 개선용 재학습이나 test 기반 튜닝은 하지 않았다.');add();table(['평가','n','MSE','MAE','Spearman'],[[label,met['n'],f(met['mse'],6),f(met['mae'],6),f(met['spearman'],6)] for label,met in [('학습 전 validation (무작위 head)',t['initial_validation_metrics']),('학습 후 validation',t['validation_metrics']),('독립 test',t['test_metrics'])]]);
sel=json.loads((B/'training_inputs.json').read_text());mu=statistics.mean(r['label'] for r in sel['train']);baseline=statistics.mean((r['label']-mu)**2 for r in sel['test']);add(f'train 평균 상수 예측의 test MSE는 {baseline:.6f}이다. test MSE가 이보다 낮더라도 Spearman 0.0241은 기능적 순위 예측을 입증하지 못한다. train/validation은 1–3 mutation, test는 4–12 mutation의 분포 차이가 있고, 소규모 1 epoch만 학습했다. 이는 제한된 동작 검증이다. 전체 TAPE 학습·최적 수렴·독립 family 평가를 완료한 결과가 아니다.');add()
add('## 검증·재현과 산출물');add();add('기본 7조건 × 3회 = 21개 embedding 파일의 입력 ID·유일성·[N,1280] shape·유한값과 반복 결과 일치를 확인했다. 최초 저장 모델 재로드는 가중치를 임시로 0으로 만든 후 저장본을 복원해 예측 일치를 검사했다. 보완 실행은 새 OS 프로세스에서 공식 base hash를 검사하고 저장된 final block·LayerNorm·head를 적용해 128 test 예측이 tolerance 1e-5 안에서 일치함을 확인했다.');add();add('checkpoint.pt는 전체 650M base 사본을 중복 저장하지 않고 갱신된 backbone 파라미터와 head, target 표준화 값, base 이름·해시, optimizer·epoch·RNG를 보관한다. 따라서 base checkpoint와 contact regression 파일을 함께 유지해야 한다. metadata, alphabet, 독립 predict.py와 사용법도 저장했다.');add();add('```bash');add('cd /home/hyein/ai-benchmark');add('esm2/.venv/bin/python esm2/scripts/predict.py \\');add('  --checkpoint esm2/models/finetuned/20261027_164300_125163/checkpoint.pt \\');add('  --fasta input.fasta --output predictions.csv');add('```');add();table(['산출물','호스트 경로'],[['최신 지침',str(R.parent/'AGENTS.md')],['기본 코드',str(R/'scripts/benchmark.py')],['준비·입력 출처',str(R/'scripts/prepare.py')+' / '+str(R/'data/sources.json')],['기본 run',str(B)],['보완 run',str(U)],['학습 모델',str(M)],['독립 예측 코드',str(R/'scripts/predict.py')],['비교 CSV',str(R/'reports'/('comparison_'+s['run_id']+'.csv'))],['전체 상태',str(R.parent/'STATUS.md')],['현재 보고서',str(report)]]);add('기본 작업의 최초 stdout은 파일로 redirect하지 않아 전체 원본 console.log가 없다. 세션 도구가 반환한 exit_code=0·SUCCESS와 복구 확인을 process_recovery.json에 기록했고, 원본 resources.csv·phase 지표는 보존돼 있다. 보완 실행은 console.log·command.json·exit_code.json을 저장한다. 기본 원본의 JSON은 수정하지 않았다. 다운로드 URL·서버 버전·크기·SHA-256은 sources.json에 기록했다. 기존 DNABERT-2 경로와 문서 tar를 조사했으나 기존 한국어 최종 Markdown 보고서는 발견하지 못해 기존 summary.json 계측 항목과 최신 사용자 요구를 기준으로 구성했다.');add()
add('## 한계와 다음 단계');add();add('가용 메모리는 모든 측정에서 충분했고 swap 증가는 없었으며 OOM은 발생하지 않았다. 총 샘플 수를 늘리는 것과 단일 GPU 입력을 키우는 것은 다르다. 사용률 80%는 일반 조건에서 이미 초과했으며 그 자체가 모델 정확도나 최대 하드웨어 성능을 뜻하지 않는다. 반복 3/40회는 한 프로세스 내 warm 상태 반복으로 독립 실행·독립 seed 실험이 아니다. 입력·모델의 사전 다운로드 캐시는 warm 상태이며 전체 생물학적 활용 범위를 검증한 것은 아니다.');add();add('후속 순서는 AlphaFold2 — ColabFold를 통한 실행이다. 현재 host hyein의 실제 그룹에 docker가 없고 /var/run/docker.sock은 root:docker 0660이다. Docker 조회는 샌드박스 밖에서도 permission denied여서 기존 이미지·컨테이너·마운트·가중치를 검증할 수 없다. 자동 권한 변경이나 sudoers 변경은 하지 않았다. 사용자가 현재 세션의 Docker 접근을 복구해야 기존 ColabFold 검증을 이어갈 수 있다.');add()
add('## 공식 출처');add();
for title,url in [('ESM-2 공식 구현·체크포인트','https://github.com/facebookresearch/esm'),('공식 토큰 예산·입력 truncation 구현','https://github.com/facebookresearch/esm/blob/main/scripts/extract.py'),('TAPE 과제·데이터','https://github.com/songlab-cal/tape'),('TAPE 공식 데이터 다운로드 목록','https://github.com/songlab-cal/tape/blob/master/download_data.sh'),('UniProt 공식 REST API','https://www.uniprot.org/help/api_queries'),('GPU-Util 정의·계측 지원','https://docs.nvidia.com/deploy/nvidia-smi/index.html'),('DGX Spark 공유 메모리·nvidia-smi memory 제한','https://docs.nvidia.com/dgx/dgx-spark/known-issues.html')]:add('- ['+title+']('+url+')')
report.write_text('\n'.join(out)+'\n')
comparison=R/'reports'/('comparison_'+s['run_id']+'.csv');cs=[]
for c in s['conditions']:cs.append(dict(source_run=B.name,**{k:v for k,v in c.items() if not isinstance(v,(list,dict))}))
for c in u['conditions']:cs.append(dict(source_run=U.name,**{k:v for k,v in c.items() if not isinstance(v,(list,dict))}))
fields=list(dict.fromkeys(k for c in cs for k in c))
with comparison.open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(cs)
# All metadata is separate from original recorded measurements.
alphabet={'architecture':'ESM-1b','tokens':['<cls>','<pad>','<eos>','<unk>','L','A','G','V','S','E','R','T','I','D','P','K','Q','N','F','Y','M','H','W','C','X','B','U','Z','O','.','-','<null_1>','<mask>'],'residue_limit':1022,'prepend_bos':True,'append_eos':True,'pooling':'residue mean; exclude BOS/EOS/padding','label':'continuous TAPE log_fluorescence','label_mapping':None}
(M/'preprocessing.json').write_text(json.dumps(alphabet,indent=2))
training={'optimizer':'AdamW','learning_rate_backbone':1e-4,'learning_rate_head':1e-3,'weight_decay':.01,'epochs':1,'steps':8,'token_budget':4096,'precision':'FP32','tf32':False,'seed':42,'trainable_blocks':['layers.32','emb_layer_norm_after','regression_head'],'training_inputs':str(B/'training_inputs.json'),'base_environment':str(R/'.venv/pyvenv.cfg')}
(M/'training_config.json').write_text(json.dumps(training,indent=2));shutil.copy2(B/'training_inputs.json',M/'training_inputs.json');shutil.copy2(B/'pip_freeze.txt',M/'pip_freeze.txt')
# Evidence-based overview. Original scGPT is reused and explicitly not rerun.
status='''# DGX Spark 바이오 AI 진행 상태\n\n갱신: '''+datetime.datetime.now().isoformat(timespec='seconds')+''' KST. GPU 실험은 한 번에 하나씩 실행한다.\n\n| 순서 | 모델 | 상태 | 최신 경로 / 제약 |\n| --- | --- | --- | --- |\n| 1 | ESM-2 | 입력·추론·개수/길이 비교·부분 backbone 학습·저장·독립 재로드·test·보고서 완료 | esm2/reports/'''+report.name+''' |\n| 2 | AlphaFold2 (ColabFold) | 환경 확인 차단 | Docker socket permission denied; hyein에 docker 그룹 없음. 이미지·마운트·가중치 미검증 |\n| 3 | scVI | 미착수 | AlphaFold2 후 순차 진행; raw count 별도 확인 필요 |\n| 4 | Cellpose | 미착수 | 버전·가중치·정답 mask 미확인 |\n| 5 | DNABERT-2 | 기존 기본 결과 보존; 새 규모/사용률 보완 미착수 | /home/hyein/dnabert2-benchmark/dnabert2_eval; 원본 summary·저장 모델 존재 확인 |\n| 6 | Nucleotide Transformer V2 | 미착수 | - |\n| 7 | Clair3 | 미착수 | - |\n| 8 | RNA-FM | 미착수 | - |\n| 9 | OpenSpliceAI | 미착수 | - |\n| 10 | scGPT | 기존 결과 보존; 새 개수/사용률 보완 미착수 | scgpt/results; 기존 기본·길이 실험 재실행 없음 |\n\nESM-2 기본 run: esm2/results/'''+B.name+''' (종료 코드 0).\n보완 run: esm2/results/'''+U.name+''' (완료 및 exit_code.json 확인).\n기존 ESM-2 작업은 세션 중단 후에도 정상 완료됐고 중복 실행하지 않았다. 보완은 기존 작업 종료 확인 후 실시했다.\n부분 파인튜닝 정확도: test n=128, Spearman 0.0241; 동작 검증이며 높은 예측 성능 완료로 표현하지 않는다.\n\n후속 사용자 조치: hyein으로 새 로그인 세션에서 docker ps가 sudo 없이 실행되도록 Docker 접근 권한을 복구해야 한다. 현재 사용자의 승인 없이 그룹·socket·sudoers 권한을 변경하지 않았다.\n'''
(R.parent/'STATUS.md').write_text(status)
manifest=[]
for p in sorted(list(B.glob('*'))+list(U.glob('*'))+list(M.glob('*'))+[report,comparison,R.parent/'STATUS.md']):
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8388608),b''):h.update(b)
  manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(R/'reports'/('artifacts_'+s['run_id']+'.json')).write_text(json.dumps(manifest,indent=2));print(report)
````
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>esm2/scripts/supplement.py — 원문 전문 (8,873 bytes)</summary>


<!-- RESTORE_FILE {"path": "esm2/scripts/supplement.py", "sha256": "ceac1a1593c7aaa06e2ded3fff60a8b930858d9c5ef8a4508baec89a324ae266", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Longer measurement of existing length inputs; fresh-process reload verification."""
import sys,os,json,pathlib,time,datetime,threading,subprocess,traceback,csv,hashlib,shutil
import torch,esm,numpy as np,psutil
R=pathlib.Path(__file__).resolve().parents[1]; OLD=R/'results'/'20261027_164300_125163'; O=pathlib.Path(sys.argv[1]); O.mkdir(parents=True,exist_ok=True)
shutil.copy2(__file__,O/'supplement.py')
def write(name,obj): (O/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
def log(*args): print(datetime.datetime.now().isoformat(),*args,flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[];stop=threading.Event();phase='setup';process=psutil.Process();initial_swap=psutil.swap_memory().used
summary={'status':'running','parent_run':str(OLD),'run_path':str(O),'sampling_seconds':.5,'repeats':40,'warmup_passes_per_condition':1,'conditions':[]}
write('summary.json',summary)
def monitor():
 with (O/'resources.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['query_start','query_end','phase','gpu_util_percent','rss_bytes','available_bytes','swap_bytes']);w.writeheader()
  while not stop.is_set():
   start=time.time();ph=phase
   try:
    q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3);value=float(q.stdout.strip()) if q.returncode==0 else None
   except Exception:value=None
   row=dict(query_start=start,query_end=time.time(),phase=ph,gpu_util_percent=value,rss_bytes=process.memory_info().rss,available_bytes=psutil.virtual_memory().available,swap_bytes=psutil.swap_memory().used); rows.append(row);w.writerow(row);f.flush();stop.wait(max(0,.5-(time.time()-start)))
def guard():
 if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>initial_swap+2*2**30:raise RuntimeError('Memory pressure; stop expansion')
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 # CPU inspection of every saved repeat: ids, shape, finiteness, repeat agreement.
 verified=[]
 for name in ['count_32','count_96','count_160','length_50_128','length_129_256','length_257_512','length_513_1022']:
  expected=json.loads((OLD/(name+'_inputs.json')).read_text());base=None
  for rep in range(3):
   data=torch.load(OLD/(name+'_r'+str(rep)+'.pt'),map_location='cpu',weights_only=True);assert len(data['ids'])==len(set(data['ids']))==len(expected);assert set(data['ids'])=={r['id'] for r in expected};assert data['embeddings'].shape==(len(expected),1280);assert torch.isfinite(data['embeddings']).all()
   if base is not None:assert data['ids']==base['ids'];assert torch.allclose(data['embeddings'],base['embeddings'],atol=1e-6,rtol=1e-5)
   else:base=data
  verified.append(dict(condition=name,all_three_outputs_valid=True))
 write('original_artifact_validation.json',verified)
 log('ORIGINAL ARTIFACTS VALID')
 source=json.loads((R/'data/sources.json').read_text());expectedhash=next(r['sha256'] for r in source if r['path']=='models/esm2_t33_650M_UR50D.pt');assert sha(R/'models/esm2_t33_650M_UR50D.pt')==expectedhash
 phase='model_loading';t=time.perf_counter();modeldata=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False);reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',modeldata,reg);del modeldata,reg;model=model.cuda().eval();torch.cuda.synchronize();summary['model_load_seconds']=time.perf_counter()-t
 summary['model_parameters']=sum(p.numel() for p in model.parameters());summary['encoder_parameters_excluding_lm_contact_heads']=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('lm_head.','contact_head.')))
 convert=alphabet.get_batch_converter(1022)
 def batch(recs):
  ds=esm.FastaBatchedDataset([r[0] for r in recs],[r[1] for r in recs]);return [[recs[i] for i in ix] for ix in ds.get_batch_indices(4096,extra_toks_per_seq=1)]
 def embed(b):
  _,ss,tokens=convert(b);tokens=tokens.cuda();out=model(tokens,repr_layers=[33],return_contacts=False)['representations'][33];return torch.stack([out[i,1:len(s)+1].mean(0) for i,s in enumerate(ss)]).cpu()
 for name in ['length_50_128','length_129_256','length_257_512','length_513_1022']:
  guard();src=json.loads((OLD/(name+'_inputs.json')).read_text());recs=[(r['id'],r['sequence']) for r in src];write(name+'_inputs.json',src)
  wall=time.perf_counter();phase=name+'_preprocessing';t=time.perf_counter();bs=batch(recs);preprocess=time.perf_counter()-t
  phase=name+'_warmup';t=time.perf_counter()
  with torch.no_grad():
   for b in bs:embed(b)
  torch.cuda.synchronize();warmup=time.perf_counter()-t
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();phase=name+'_execution';torch.cuda.synchronize();start=time.time();t=time.perf_counter();dur=[]
  with torch.no_grad():
   for rep in range(40):
    rt=time.perf_counter();values=[]
    for b in bs:values.append(embed(b))
    torch.cuda.synchronize();dur.append(time.perf_counter()-rt);guard()
  torch.cuda.synchronize();elapsed=time.perf_counter()-t;end=time.time();phase=name+'_save';t=time.perf_counter();out=torch.cat(values);assert torch.isfinite(out).all();torch.save(dict(ids=[i for b in bs for i,s in b],embeddings=out),O/(name+'_last_repeat.pt'));save=time.perf_counter()-t
  subset=[r for r in rows if r['query_start']>=start and r['query_end']<=end];steady=[r['gpu_util_percent'] for r in subset if r['query_start']>=start+2 and r['gpu_util_percent'] is not None];util=[r['gpu_util_percent'] for r in subset if r['gpu_util_percent'] is not None]
  result=dict(condition=name,unique_inputs=16,processed_inputs=640,repeats=40,warmup_processed_inputs_excluded=16,length_min=min(len(s) for i,s in recs),length_max=max(len(s) for i,s in recs),length_mean=float(np.mean([len(s) for i,s in recs])),preprocess_seconds=preprocess,warmup_seconds=warmup,execution_seconds=elapsed,repeat_seconds=dur,save_seconds=save,wall_seconds=time.perf_counter()-wall,start_unix=start,end_unix=end,throughput_per_second=640/elapsed,gpu_util_mean_percent=float(np.mean(util)) if util else None,gpu_util_max_percent=max(util) if util else None,gpu_util_steady_mean_percent=float(np.mean(steady)) if steady else None,sample_count=len(subset),cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,rss_sampled_peak_gib=max([r['rss_bytes'] for r in subset],default=0)/2**30,system_available_min_gib=min([r['available_bytes'] for r in subset],default=0)/2**30,swap_max_bytes=max([r['swap_bytes'] for r in subset],default=0))
  write(name+'_metrics.json',result);summary['conditions'].append(result);write('summary.json',summary);log('DONE',name,'seconds',round(elapsed,3),'GPU mean',result['gpu_util_mean_percent'])
 # Reconstruct fine-tuned model from original base in a new OS process.
 phase='fresh_reload';t=time.perf_counter();M=R/'models/finetuned/20261027_164300_125163';ck=torch.load(M/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['base_sha256']==expectedhash
 missing=model.load_state_dict(ck['backbone_updates'],strict=False);assert not missing.unexpected_keys;head=torch.nn.Linear(1280,1).cuda();head.load_state_dict(ck['head']);head.eval();model.eval();mu=ck['target_mean'];std=ck['target_std'];selected=json.loads((OLD/'training_inputs.json').read_text())['test'];rec=[(r['id'],r['sequence']) for r in selected];ids=[];pred=[]
 with torch.no_grad():
  for b in batch(rec):
   # embed() returns CPU for inference; transfer pooled 1280-vector for head.
   e=embed(b).cuda();p=(head(e).squeeze(-1)*std+mu).cpu().tolist();ids.extend(i for i,s in b);pred.extend(p)
 torch.cuda.synchronize();original=json.loads((OLD/'test_original.json').read_text());assert ids==original['ids'];delta=float(np.max(np.abs(np.asarray(pred)-original['predictions'])));assert delta<=1e-5
 write('fresh_reload_predictions.json',dict(ids=ids,predictions=pred));summary['fresh_reload']=dict(passed=True,independent_process=True,max_abs_difference=delta,tolerance=1e-5,test_inputs=len(pred),seconds=time.perf_counter()-t,base_hash_verified=True,training_resume_verified=False)
 summary['status']='success';summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write('summary.json',summary);log('SUCCESS',summary['fresh_reload'])
except Exception as e:
 summary['status']='failed';summary['error']=str(e);summary['traceback']=traceback.format_exc();write('summary.json',summary);log(traceback.format_exc());raise
finally:stop.set();thread.join(timeout=5)
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "esm2/results/supplement_20261027_165443_595867/supplement.py", "source": "esm2/scripts/supplement.py", "sha256": "ceac1a1593c7aaa06e2ded3fff60a8b930858d9c5ef8a4508baec89a324ae266"} -->

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

### record-d065fc584239

원래 파일: `esm2/data/sources.json` · 원문 SHA256: `e7ebd5ffacbf41ccbd2decc584223c4ce9cd442ef1feef7559022c9c52a80f71`

<details>
<summary>esm2/data/sources.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "esm2/data/sources.json", "sha256": "e7ebd5ffacbf41ccbd2decc584223c4ce9cd442ef1feef7559022c9c52a80f71", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "path": "data/uniprot_reviewed.fasta",
    "url": "https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20length%3A%5B50%20TO%201022%5D&format=fasta&size=500",
    "bytes": 259221,
    "sha256": "37222bb198046868760dd240413b9dd46fcd44aefc0ea68d67929f9f43a8c7d3",
    "headers": {
      "Vary": "accept,accept-encoding,x-uniprot-release,x-api-deployment-date, User-Agent",
      "Cache-Control": "public, max-age=43200",
      "x-cache": "miss cached",
      "Content-Type": "text/plain;format=fasta",
      "Access-Control-Allow-Credentials": "true",
      "Access-Control-Expose-Headers": "Link, X-Total-Results, X-UniProt-Release, X-UniProt-Release-Date, X-API-Deployment-Date",
      "X-API-Deployment-Date": "07-October-2026",
      "strict-transport-security": "max-age=31536000; includeSubDomains; preload",
      "Date": "Tue, 27 Oct 2026 07:40:19 GMT",
      "Access-Control-Max-Age": "1728000",
      "X-UniProt-Release": "2026_03",
      "Link": "<https://rest.uniprot.org/uniprotkb/search?format=fasta&query=reviewed%3Atrue%20AND%20length%3A%5B50%20TO%201022%5D&cursor=c9bacmxsqhkqgdwiuhth8jj7to99sscq47ho0&size=500>; rel=\"next\"",
      "X-Total-Results": "544243",
      "Transfer-Encoding": "chunked",
      "Access-Control-Allow-Origin": "*",
      "Connection": "close",
      "Access-Control-Allow-Methods": "GET, PUT, POST, DELETE, PATCH, OPTIONS",
      "Access-Control-Allow-Headers": "DNT,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization",
      "X-UniProt-Release-Date": "02-September-2026"
    },
    "checked_utc": "2026-10-27T07:40:20.661126+00:00"
  },
  {
    "path": "data/fluorescence.tar.gz",
    "url": "https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz",
    "bytes": 1635678,
    "sha256": "20c9d114234a5136c4087920480e7fa2d586a653061f0cd141f626a984ede8bd",
    "headers": {
      "x-amz-id-2": "oO4IwE+XhyJAc2m+yxgfS8OMDfIPWKIfh9F/LmyoAgsMMm859bgl92/nQBHcb2k2UOSVt/RDOZ4=",
      "x-amz-request-id": "Y3QREGECEFEMDCWG",
      "Server": "AmazonS3",
      "Date": "Tue, 27 Oct 2026 07:40:22 GMT",
      "Last-Modified": "Mon, 02 Aug 2021 21:35:32 GMT",
      "ETag": "\"d63d1d51ec8c20ff0d981e4cbd67457a\"",
      "x-amz-server-side-encryption": "AES256",
      "x-amz-version-id": "XaIPt6wWSWvLXtAM7KoacyXabIlxP6bz",
      "Accept-Ranges": "bytes",
      "Content-Type": "application/x-tar",
      "Content-Length": "1635678",
      "Connection": "close"
    },
    "checked_utc": "2026-10-27T07:40:23.125362+00:00"
  },
  {
    "path": "models/esm2_t33_650M_UR50D.pt",
    "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
    "bytes": 2604537549,
    "sha256": "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
    "headers": {
      "Content-Type": "binary/octet-stream",
      "Content-Length": "2604537549",
      "Connection": "close",
      "Accept-Ranges": "bytes",
      "Server": "AmazonS3",
      "Last-Modified": "Thu, 18 Aug 2022 13:26:44 GMT",
      "x-amz-version-id": "2G4typsUSwOKQLH35sqPMJ27ZLX1AEJG",
      "x-amz-server-side-encryption": "AES256",
      "x-amz-storage-class": "INTELLIGENT_TIERING",
      "Date": "Tue, 27 Oct 2026 07:40:24 GMT",
      "ETag": "\"12a18098227c0ff911354647d25d494d-311\"",
      "X-Cache": "RefreshHit from cloudfront",
      "Via": "1.1 540723249b706f7cf8fcf2324ac3eb4c.cloudfront.net (CloudFront)",
      "X-Amz-Cf-Pop": "ICN57-P4",
      "X-Amz-Cf-Id": "RWk98Ya6B5Jbk-isYd-U9BFY_lei3kuOkQVcbRh-WEY9MWDsKaEP5g=="
    },
    "checked_utc": "2026-10-27T07:40:47.438005+00:00"
  },
  {
    "path": "models/esm2_t33_650M_UR50D-contact-regression.pt",
    "url": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt",
    "bytes": 3687,
    "sha256": "8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575",
    "headers": {
      "Content-Type": "binary/octet-stream",
      "Content-Length": "3687",
      "Connection": "close",
      "Accept-Ranges": "bytes",
      "Server": "AmazonS3",
      "Last-Modified": "Thu, 18 Aug 2022 13:25:01 GMT",
      "x-amz-version-id": "cEcpCgy3_xlt3NCWsoXOzeIrh7.EDOBu",
      "x-amz-server-side-encryption": "AES256",
      "Date": "Tue, 27 Oct 2026 07:40:48 GMT",
      "ETag": "\"743354dd846c9460fd2fe50892033543\"",
      "X-Cache": "RefreshHit from cloudfront",
      "Via": "1.1 7685cf04fa239254cb7d83b3bae5a1ec.cloudfront.net (CloudFront)",
      "X-Amz-Cf-Pop": "ICN57-P4",
      "X-Amz-Cf-Id": "TJN0wv0A9IFE3WwZ2GmEqHCg6O7Gx64gVnqWZUmbVj2M6y3KxT3d0Q=="
    },
    "checked_utc": "2026-10-27T07:40:47.889818+00:00"
  },
  {
    "path": "data/official_extract.py",
    "url": "https://raw.githubusercontent.com/facebookresearch/esm/main/scripts/extract.py",
    "bytes": 5161,
    "sha256": "9bcd76f69d434af390e9ac27f9cc1f09e701bfdfc7836a1409679cdc24b9c348",
    "headers": {
      "Connection": "close",
      "Content-Length": "5161",
      "Cache-Control": "max-age=300",
      "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
      "Content-Type": "text/plain; charset=utf-8",
      "ETag": "\"2fdc61ed643211c77d3f2044d04429de35eec769ff8ce95fc922c9da099192a6\"",
      "Strict-Transport-Security": "max-age=31536000",
      "X-Content-Type-Options": "nosniff",
      "X-Frame-Options": "deny",
      "X-XSS-Protection": "1; mode=block",
      "X-GitHub-Request-Id": "B96C:95E3B:156E9:2A52B:6AE0557F",
      "x-github-edge-region": "koreacentral",
      "Accept-Ranges": "bytes",
      "Date": "Tue, 27 Oct 2026 07:40:48 GMT",
      "Via": "1.1 varnish",
      "X-Served-By": "cache-icn1450066-ICN",
      "X-Cache": "MISS",
      "X-Cache-Hits": "0",
      "X-Timer": "S1793086848.916004,VS0,VE203",
      "Vary": "Authorization,Accept-Encoding",
      "Access-Control-Allow-Origin": "*",
      "Cross-Origin-Resource-Policy": "cross-origin",
      "X-Fastly-Request-ID": "4b23bbad0924f8e2d25757d66817c83911d6bb2f",
      "Expires": "Tue, 27 Oct 2026 07:45:48 GMT",
      "Source-Age": "0"
    },
    "checked_utc": "2026-10-27T07:40:48.126357+00:00"
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-03d7122ffc65

원래 파일: `esm2/models/finetuned/20261027_164300_125163/preprocessing.json` · 원문 SHA256: `56a6c2288bd2b753f8b7ba30650efbca8666c114a9aa38a77686a01dff861d6d`

<details>
<summary>esm2/models/finetuned/20261027_164300_125163/preprocessing.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "esm2/models/finetuned/20261027_164300_125163/preprocessing.json", "sha256": "56a6c2288bd2b753f8b7ba30650efbca8666c114a9aa38a77686a01dff861d6d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "architecture": "ESM-1b",
  "tokens": [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "<null_1>",
    "<mask>"
  ],
  "residue_limit": 1022,
  "prepend_bos": true,
  "append_eos": true,
  "pooling": "residue mean; exclude BOS/EOS/padding",
  "label": "continuous TAPE log_fluorescence",
  "label_mapping": null
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-8f9b60f68a91

원래 파일: `esm2/results/20261027_164300_125163/environment.json` · 원문 SHA256: `674c9c318509d61bf417ad6423fbc7c4873b24fbab2f9e4befd70aed032ecad9`

<details>
<summary>esm2/results/20261027_164300_125163/environment.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "esm2/results/20261027_164300_125163/environment.json", "sha256": "674c9c318509d61bf417ad6423fbc7c4873b24fbab2f9e4befd70aed032ecad9", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "python": "3.10.21 | packaged by conda-forge | (main, Aug 21 2026, 22:36:13) [GCC 14.4.0]",
  "torch": "2.14.1+cu130",
  "cuda": "13.0",
  "esm": "fair-esm 2.0.0",
  "host": "spark-e4b6",
  "platform": "Linux-7.0.0-1019-nvidia-aarch64-with-glibc2.39",
  "gpu": "NVIDIA GB10",
  "cuda_available": true,
  "system_memory_bytes": 130660974592,
  "initial_available_bytes": 122790346752,
  "initial_swap_bytes": 0,
  "gpu_processes": "Tue Oct 27 16:43:00 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 580.178.04             Driver Version: 580.178.04     CUDA Version: 13.0     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GB10                    On  |   0000000F:01:00.0  On |                  N/A |\n| N/A   39C    P0              7W /  N/A  | Not Supported          |      4%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            3503      G   /usr/lib/xorg/Xorg                      341MiB |\n|    0   N/A  N/A            3701      G   /usr/bin/gnome-shell                    206MiB |\n|    0   N/A  N/A            5827      G   .../9112/usr/lib/firefox/firefox        413MiB |\n+-----------------------------------------------------------------------------------------+\n",
  "docker": "Permission denied for current user; host isolated venv used"
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-fef3de053a15

원래 파일: `esm2/results/20261027_164300_125163/sources.json` · 원문 SHA256: `e7ebd5ffacbf41ccbd2decc584223c4ce9cd442ef1feef7559022c9c52a80f71`

<details>
<summary>esm2/results/20261027_164300_125163/sources.json — 명령/설정/환경 원문</summary>


동일 원문은 `esm2/data/sources.json` 블록에 전문 수록했다. 아래 별칭은 동일한 바이트를 원래 경로로 복원한다.
<!-- RESTORE_ALIAS {"path": "esm2/results/20261027_164300_125163/sources.json", "source": "esm2/data/sources.json", "sha256": "e7ebd5ffacbf41ccbd2decc584223c4ce9cd442ef1feef7559022c9c52a80f71"} -->


</details>

### record-fbd30c6d4c7c

원래 파일: `esm2/results/supplement_20261027_165443_595867/command.json` · 원문 SHA256: `e50a382b05f7e96e148491848a8612d5470041ec85be1bca28aa542c438ca0b8`

<details>
<summary>esm2/results/supplement_20261027_165443_595867/command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "esm2/results/supplement_20261027_165443_595867/command.json", "sha256": "e50a382b05f7e96e148491848a8612d5470041ec85be1bca28aa542c438ca0b8", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/esm2/scripts/supplement.py",
  "/home/hyein/ai-benchmark/esm2/results/supplement_20261027_165443_595867"
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

<details><summary>추가 실험 실행 원문: inference_b1_executed_source.py</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/inference_b1_executed_source.py", "sha256": "3c4d7fa1feba9ea690199979b63d4bdec34b4e802dae514e1bb5efdaf017e17b", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,json,time,gc,os,hashlib,platform,shutil
import numpy as np,torch,esm,psutil
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'));from gpu_support import Meter,write_json
R=ROOT/'esm2';D=Path(sys.argv[1]);B=int(sys.argv[2]);C=D/f'batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','batch':B,'task':'pretrained protein mean embeddings inference only','conditions':[]};write_json(C/'summary.json',S)
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();torch.cuda.set_per_process_memory_fraction(min(.9,100*2**30/torch.cuda.get_device_properties(0).total_memory));S['allocator_budget_gib']=100;meter.phase='model_loading';t=time.time();data=torch.load(R/'models/esm2_t33_650M_UR50D.pt',map_location='cpu',weights_only=False);reg=torch.load(R/'models/esm2_t33_650M_UR50D-contact-regression.pt',map_location='cpu',weights_only=True);model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50D',data,reg);del data,reg;gc.collect();model.float().cuda().eval();torch.cuda.synchronize();S['model_loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for p in model.parameters());S['parameters_backbone']=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('lm_head.','contact_head.')));S['parameters_trainable_in_this_run']=0;S['environment']={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'esm':esm.__version__,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform()};write_json(C/'summary.json',S)
 records=json.loads((D/'inputs.json').read_text())[:B];assert len(records)==B;recs=[(r['id'],r['sequence']) for r in records];convert=alphabet.get_batch_converter(1022);_,seqs,tokens=convert(recs);assert tokens.shape==(B,952);torch.save({'ids':[i for i,s in recs],'tokens':tokens},C/'tokenized_inputs.pt');S['input']={'unique_sequences':B,'length_min':min(map(len,seqs)),'length_max':max(map(len,seqs)),'length_mean':float(np.mean(list(map(len,seqs)))),'actual_token_shape':list(tokens.shape),'dtype':str(tokens.dtype),'truncation':False,'padding':'official converter, batch longest950AA + BOS/EOS =952 positions'}
 def forward():
  x=tokens.cuda();out=model(x,repr_layers=[33],return_contacts=False);h=out['representations'][33];e=torch.stack([h[i,1:len(seq)+1].mean(0) for i,seq in enumerate(seqs)]).cpu();assert torch.isfinite(e).all();return e
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():embedding=forward()
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del embedding;gc.collect();t,base=meter.begin('embedding_inference');steps=0
 with torch.inference_mode():
  while steps<2 or time.time()-t<15:
   embedding=forward();torch.cuda.synchronize();steps+=1
 r=meter.end(t,base,batch_size=B,unique_inputs=B,processed_inputs=B*steps,repeats=steps,input_shape=list(tokens.shape));r['sequences_per_second']=B*steps/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];S['conditions']=[r];torch.save({'ids':[i for i,s in recs],'embeddings':embedding},C/'embeddings.pt');saved=torch.load(C/'embeddings.pt',map_location='cpu',weights_only=True);assert saved['embeddings'].shape==(B,1280) and torch.equal(saved['embeddings'],embedding);S['saved_output_exact_reload']=True
 ref=D/'batch_1/embeddings.pt'
 if B>1:
  reference=torch.load(ref,map_location='cpu',weights_only=True);diff=float((reference['embeddings'][0]-embedding[0]).abs().max());assert torch.allclose(reference['embeddings'][0],embedding[0],atol=1e-4,rtol=1e-4);S['first_sequence_batch_invariance_max_abs']=diff
 S['status']='complete';write_json(C/'summary.json',S);print('BATCH_COMPLETE',B,round(r['memory']['peak_allocated_gib'],3),round(r['sequences_per_second'],3),flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write_json(C/'summary.json',S);raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: remaining_queue_controller_source.py</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/remaining_queue_controller_source.py", "sha256": "98b8fc16e66dc36fb430c95166b23c5fdcbc408a3dd6075060b4f2987ea43b2c", "encoding": "utf-8", "added_final_newline": false} -->
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
   import numpy as np
   pool=len(np.load(D/(kind+'_inputs.npy'),mmap_mode='r'));b=pool;source=R/'scripts/batch_followup/worker.py';cmd=['docker','run','--rm','--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','CANDIDATE_MODEL='+kind,'-e','BATCH='+str(b),'local/clair3-spark:9d1dd3f','/work/.venv/bin/python','-u','/work/scripts/batch_followup/worker.py'];code=launch(D,kind+'_full_real_pool',cmd,source)
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

<details><summary>추가 실험 실행 원문: inference_b1_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/inference_b1_command.json", "sha256": "620dbf2c0dbfb72981147d612f590f31c636359ccfcbe8392a23e716cdb9deed", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/esm2/scripts/real_training_v3/inference_worker.py",
  "/home/hyein/ai-benchmark/esm2/results/real_input_capacity_20261102_143107_778283",
  "1"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b384_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/inference_b384_command.json", "sha256": "830791c7f0c724d946341abe28b7f2c37c422f3fb0b858ea19ebdec63c65bb31", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/esm2/scripts/real_training_v3/inference_worker.py",
  "/home/hyein/ai-benchmark/esm2/results/real_input_capacity_20261102_143107_778283",
  "384"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: update_reports_1d886f7a51ad44f6.py</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/reproduction_sources/update_reports_1d886f7a51ad44f6.py", "sha256": "1d886f7a51ad44f6b5b9f98fd00fc10a8128c5a5817c692e3528debb3574f15d", "encoding": "utf-8", "added_final_newline": false} -->
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
  values=[p.parent.name,s.get('status'),pick(r,'unique_training_cells','unique_train_images','unique_inputs','unique_candidates','unique_queries'),pick(r,'batch_size','batch') or s.get('batch'),r.get('epochs',s.get('epochs')),pick(r,'optimizer_steps','steps'),r.get('seconds'),pick(r,'cells_per_second','crops_per_second_trainer_scope','inputs_per_second','sequences_per_second','candidates_per_second','queries_per_second'),mem.get('peak_allocated_gib'),mem.get('peak_reserved_gib'),r.get('allocated_fraction_shared_ram',0)*100,r.get('gpu_util_avg'),r.get('gpu_util_max'),score or None,s.get('saved_model_reload_verified',s.get('reload_mask_equal'))]
  rows.append('|'+ '|'.join(map(f,values))+'|')
 text='<!-- REAL_TRAINING_V3_START -->\n### 실제 데이터 추가 실험 — '+D.name+'\n\n진행 중인 실행의 완료 조건만 아래에 갱신한다. probe는 짧은 용량 측정이며 전체 학습이나 평가 완료가 아니다. 메모리%는 CUDA allocated peak / 시스템 공유 RAM ×100이며 GPU-Util과 다르다. 독립 실행 1회 기준.\n\n|조건|상태|고유 입력|배치|epoch|optimizer steps|실측 시간(s)|입력/s|allocated peak GiB|reserved peak GiB|공유 RAM 대비 allocated %|GPU-Util 평균%|최대%|독립 test 지표|저장 모델 재로드|\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'+'\n'.join(rows)+'\n\n'
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
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: fetch_real_sequences_2921d351875d2a70.py</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/reproduction_sources/fetch_real_sequences_2921d351875d2a70.py", "sha256": "2921d351875d2a705e8f9de0484f3cb659da350e1061ae987f6bbed3ede74d31", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import urllib.request,json,hashlib,datetime,gzip,random
ROOT=Path('/home/hyein/ai-benchmark');out=ROOT/'scripts/real_training_v3/fetched_inputs';out.mkdir(exist_ok=True)
# Follow official UniProt pagination. Keep the original500rows and add the next real page.
p=ROOT/'esm2/data/uniprot_reviewed_900_950_20261028.fasta';meta=json.loads(p.with_suffix('.source.json').read_text());url=meta['headers']['Link'].split('<')[1].split('>')[0]
with urllib.request.urlopen(url,timeout=90) as response:body=response.read();headers=dict(response.headers)
(out/'uniprot_page2.fasta').write_bytes(body);rows=[]
for line in (p.read_text()+'\n'+body.decode()).splitlines():
 if line.startswith('>'):rows.append({'id':line[1:].split()[0],'sequence':''})
 elif line.strip():rows[-1]['sequence']+=line.strip()
seen=set();unique=[]
for r in rows:
 if r['sequence'] in seen:continue
 assert 900<=len(r['sequence'])<=950;seen.add(r['sequence']);unique.append(r)
random.Random(42).shuffle(unique);i=max(range(len(unique)),key=lambda i:len(unique[i]['sequence']));unique.insert(0,unique.pop(i));(out/'esm2.json').write_text(json.dumps(unique));(out/'uniprot_source.json').write_text(json.dumps({'url':url,'headers':headers,'sha256':hashlib.sha256(body).hexdigest(),'total_unique':len(unique),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('ESM_REAL_POOL',len(unique),flush=True)
url='https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.lncRNA_transcripts.fa.gz'
try:
 with urllib.request.urlopen(url,timeout=120) as response:body=response.read();headers=dict(response.headers)
 (out/'gencode.v47.lncRNA_transcripts.fa.gz').write_bytes(body);text=gzip.decompress(body).decode();records=[];seen=set()
 for part in text.split('>')[1:]:
  lines=part.splitlines();seq=''.join(lines[1:]).upper().replace('T','U')
  if len(seq)<1000 or not set(seq)<=set('ACGUN'):continue
  seq=seq[:1000]
  if seq in seen:continue
  seen.add(seq);records.append({'id':lines[0].split('|')[0],'sequence':seq,'original_length':sum(map(len,lines[1:])),'crop':'first1000nt, transcriptTtoU'})
  if len(records)>=2400:break
 (out/'rnafm.json').write_text(json.dumps(records));(out/'gencode_source.json').write_text(json.dumps({'url':url,'sha256':hashlib.sha256(body).hexdigest(),'headers':headers,'unique':len(records),'note':'actual lncRNA transcripts, first1000nt, unlabelled embedding task'},indent=2));print('RNA_REAL_POOL',len(records),flush=True)
except Exception as e:(out/'gencode_fetch_error.json').write_text(json.dumps({'error':repr(e),'url':url}));print('RNA_FETCH_FAILED',repr(e),flush=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: uniprot_source_111f2f05517f577c.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/reproduction_sources/uniprot_source_111f2f05517f577c.json", "sha256": "111f2f05517f577c29e921fa2f79b20ee76190c1d30e47c107152a01c3fdd335", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "url": "https://rest.uniprot.org/uniprotkb/search?format=fasta&query=reviewed%3Atrue%20AND%20length%3A%5B900%20TO%20950%5D&cursor=c9bacmxsqhkqgdwa659orqyetr0ii434c477c&size=500",
  "headers": {
    "Vary": "accept,accept-encoding,x-uniprot-release,x-api-deployment-date, User-Agent",
    "Cache-Control": "public, max-age=43200",
    "x-cache": "miss cached",
    "Content-Type": "text/plain;format=fasta",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Expose-Headers": "Link, X-Total-Results, X-UniProt-Release, X-UniProt-Release-Date, X-API-Deployment-Date",
    "X-API-Deployment-Date": "07-October-2026",
    "strict-transport-security": "max-age=31536000; includeSubDomains; preload",
    "Date": "Thu, 29 Oct 2026 08:29:15 GMT",
    "Access-Control-Max-Age": "1728000",
    "X-UniProt-Release": "2026_03",
    "Link": "<https://rest.uniprot.org/uniprotkb/search?format=fasta&query=reviewed%3Atrue%20AND%20length%3A%5B900%20TO%20950%5D&cursor=28m7xk8oeejl5mhhhq5qpvqaft8565vcoje2omo&size=500>; rel=\"next\"",
    "X-Total-Results": "4294",
    "Transfer-Encoding": "chunked",
    "Access-Control-Allow-Origin": "*",
    "Connection": "close",
    "Access-Control-Allow-Methods": "GET, PUT, POST, DELETE, PATCH, OPTIONS",
    "Access-Control-Allow-Headers": "DNT,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization",
    "X-UniProt-Release-Date": "02-September-2026"
  },
  "sha256": "4eee9488cf8db14047659e6a747095a3848191fa491454b7e4921688f3566752",
  "total_unique": 997,
  "utc": "2026-10-29T08:29:22.688403+00:00"
}
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: update_reports_b4b01e29a437c822.py</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/reproduction_sources/update_reports_b4b01e29a437c822.py", "sha256": "b4b01e29a437c822666331ea78a331d10dfc72fa096c867333b819b7fbd7f433", "encoding": "utf-8", "added_final_newline": false} -->
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
  values=[p.parent.name,s.get('status'),pick(r,'unique_training_cells','unique_train_images','unique_inputs','unique_candidates','unique_queries'),pick(r,'batch_size','batch') or s.get('batch'),r.get('epochs',s.get('epochs')),pick(r,'optimizer_steps','steps'),r.get('seconds'),pick(r,'cells_per_second','crops_per_second_trainer_scope','inputs_per_second','sequences_per_second','candidates_per_second','queries_per_second'),mem.get('peak_allocated_gib'),mem.get('peak_reserved_gib'),r['allocated_fraction_shared_ram']*100 if 'allocated_fraction_shared_ram' in r else None,r.get('gpu_util_avg'),r.get('gpu_util_max'),score or None,s.get('saved_model_reload_verified',s.get('reload_mask_equal'))]
  rows.append('|'+ '|'.join(map(f,values))+'|')
 text='<!-- REAL_TRAINING_V3_START -->\n### 실제 데이터 추가 실험 — '+D.name+'\n\n진행 중인 실행의 완료 조건만 아래에 갱신한다. probe는 짧은 용량 측정이며 전체 학습이나 평가 완료가 아니다. 메모리%는 CUDA allocated peak / 시스템 공유 RAM ×100이며 GPU-Util과 다르다. 독립 실행 1회 기준.\n\n|조건|상태|고유 입력|배치|epoch|optimizer steps|실측 시간(s)|입력/s|allocated peak GiB|reserved peak GiB|공유 RAM 대비 allocated %|GPU-Util 평균%|최대%|독립 test 지표|저장 모델 재로드|\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'+'\n'.join(rows)+'\n\n'
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
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: full_training_b64_executed_source.py</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/full_training_b64_executed_source.py", "sha256": "dc1c9a02d54e9a816169e1fa0e2bbd51d8c40db6895fc12a6ec36141f4246f3f", "encoding": "utf-8", "added_final_newline": false} -->
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

<details><summary>추가 실험 실행 원문: full_training_b64_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/full_training_b64_command.json", "sha256": "c937a4ed7b7a090d51726ce92cf46a39b5ce424e103561bcf4e50530ee60b1ca", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "env",
  "RUN_ID=real_input_capacity_20261102_143107_778283",
  "MODEL_KIND=esm2",
  "BATCH=64",
  "TRAIN_MODE=full",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "HF_HOME=/work/.cache/huggingface",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "HF_HUB_OFFLINE=1",
  "TRANSFORMERS_OFFLINE=1",
  "MODEL_ROOT=/home/hyein/ai-benchmark/esm2",
  "SUPPORT_PATH=/home/hyein/ai-benchmark/scripts",
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/scripts/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: full_training_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/full_training_b8_command.json", "sha256": "8f51b1301336723ad2e8133b83e654c1447138fd176c8a3b3ad54727f6878384", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "env",
  "RUN_ID=real_input_capacity_20261102_143107_778283",
  "MODEL_KIND=esm2",
  "BATCH=8",
  "TRAIN_MODE=full",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "HF_HOME=/work/.cache/huggingface",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "HF_HUB_OFFLINE=1",
  "TRANSFORMERS_OFFLINE=1",
  "MODEL_ROOT=/home/hyein/ai-benchmark/esm2",
  "SUPPORT_PATH=/home/hyein/ai-benchmark/scripts",
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/scripts/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b448_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/inference_b448_command.json", "sha256": "beb2c8a482a4ef02e1f88d1b4d1cb68fc2f3f0c0084184497740d005ec237b62", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/esm2/scripts/real_training_v3/inference_worker.py",
  "/home/hyein/ai-benchmark/esm2/results/real_input_capacity_20261102_143107_778283",
  "448"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b512_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/inference_b512_command.json", "sha256": "72709dbd4bfac477ee94069351cfeca3a51fca31f6326063dab2caa6ec623365", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/esm2/scripts/real_training_v3/inference_worker.py",
  "/home/hyein/ai-benchmark/esm2/results/real_input_capacity_20261102_143107_778283",
  "512"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b544_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/inference_b544_command.json", "sha256": "e2fb224269401d50d7b199ff09aec1c1a5372c7de13a536222fa6ffef166d0ac", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/esm2/scripts/real_training_v3/inference_worker.py",
  "/home/hyein/ai-benchmark/esm2/results/real_input_capacity_20261102_143107_778283",
  "544"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: inference_b560_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/inference_b560_command.json", "sha256": "4aab592f3faf920db24c53baf766c2bf99c5afff99ce9e1e5052313404eb80c3", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/esm2/scripts/real_training_v3/inference_worker.py",
  "/home/hyein/ai-benchmark/esm2/results/real_input_capacity_20261102_143107_778283",
  "560"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b32_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/training_probe_b32_command.json", "sha256": "15b073dcb3006ba9c64a410a9f8bf5897685b4318c733d8beff61b5ea7c219d8", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "env",
  "RUN_ID=real_input_capacity_20261102_143107_778283",
  "MODEL_KIND=esm2",
  "BATCH=32",
  "TRAIN_MODE=probe",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "HF_HOME=/work/.cache/huggingface",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "HF_HUB_OFFLINE=1",
  "TRANSFORMERS_OFFLINE=1",
  "MODEL_ROOT=/home/hyein/ai-benchmark/esm2",
  "SUPPORT_PATH=/home/hyein/ai-benchmark/scripts",
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/scripts/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b64_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/training_probe_b64_command.json", "sha256": "b433cc87833c74a0cd90a766e7fac442401d04d8c74453bc7732a389944e2581", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "env",
  "RUN_ID=real_input_capacity_20261102_143107_778283",
  "MODEL_KIND=esm2",
  "BATCH=64",
  "TRAIN_MODE=probe",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "HF_HOME=/work/.cache/huggingface",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "HF_HUB_OFFLINE=1",
  "TRANSFORMERS_OFFLINE=1",
  "MODEL_ROOT=/home/hyein/ai-benchmark/esm2",
  "SUPPORT_PATH=/home/hyein/ai-benchmark/scripts",
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/scripts/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b8_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/training_probe_b8_command.json", "sha256": "8af0d47758236e9d9a16552e03356b973f5526e4d7525bd5b86554b71bdb6c9b", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "env",
  "RUN_ID=real_input_capacity_20261102_143107_778283",
  "MODEL_KIND=esm2",
  "BATCH=8",
  "TRAIN_MODE=probe",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "HF_HOME=/work/.cache/huggingface",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "HF_HUB_OFFLINE=1",
  "TRANSFORMERS_OFFLINE=1",
  "MODEL_ROOT=/home/hyein/ai-benchmark/esm2",
  "SUPPORT_PATH=/home/hyein/ai-benchmark/scripts",
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/scripts/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: training_probe_b96_command.json</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/training_probe_b96_command.json", "sha256": "1cf2e6fd1769cb7100ebd4d4adfc78bc71e60f6419f6f0ad6a8dbb220d296ba1", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "env",
  "RUN_ID=real_input_capacity_20261102_143107_778283",
  "MODEL_KIND=esm2",
  "BATCH=96",
  "TRAIN_MODE=probe",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "HF_HOME=/work/.cache/huggingface",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "HF_HUB_OFFLINE=1",
  "TRANSFORMERS_OFFLINE=1",
  "MODEL_ROOT=/home/hyein/ai-benchmark/esm2",
  "SUPPORT_PATH=/home/hyein/ai-benchmark/scripts",
  "/home/hyein/ai-benchmark/esm2/.venv/bin/python",
  "-u",
  "/home/hyein/ai-benchmark/scripts/real_training_v3/sequence_training_worker.py"
]
```
<!-- END_RESTORE_FILE -->

</details>

<details><summary>추가 실험 실행 원문: remaining_queue_152f0b87bce32235.py</summary>

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/reproduction_sources/remaining_queue_152f0b87bce32235.py", "sha256": "152f0b87bce32235a41f8616d74f03deabd611eef85dde6419bd68da175973a6", "encoding": "utf-8", "added_final_newline": false} -->
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

<!-- RESTORE_FILE {"path": "esm2/results/real_input_capacity_20261102_143107_778283/reproduction_sources/update_reports_305433d7fed68d6f.py", "sha256": "305433d7fed68d6ff20850b0d00d18708a0de3d92b36924d41dbbfe168034b8e", "encoding": "utf-8", "added_final_newline": false} -->
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
