# ESM-2 최종 통합 보고서

## 1. 모델과 실제 입력

단백질 아미노산 서열 → 단백질당1280차원 임베딩 또는 fluorescence 예측. 체크포인트 esm2_t33_650M_UR50D(명목650M).

실제파일·입력/정답예시·전처리와모델구조설명은이문서5절에전문을포함했다. 추론데이터와지도학습label을구분하며,세포/유전자·서열/토큰·이미지/타일의단위를섞지않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|단백질 임베딩 추론|고유1개,UniProt reviewed900~950AA|1×952토큰,FP32|56회/총56건|15.03초|3.72서열/초|2.700 GiB / 2.22%|정확도과제아닌임베딩; 80%미도달|
|단백질 임베딩 추론|고유384개,UniProt reviewed900~950AA|384×952토큰,FP32|2회/총768건|216.96초|3.54서열/초|64.416 GiB / 52.94%|정확도과제아닌임베딩; 80%미도달|
|부분backbone 파인튜닝|train128/val64/test128; 237AA fluorescence 라벨|토큰예산4096 가변batch,FP32|1epoch/8step|7.22초|17.74서열/초|3.623 GiB / 2.98%|동일저장모델test Spearman 0.0241;80%미도달|

## 4. 결과 해석

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

[공식ESM](https://github.com/facebookresearch/esm) · [UniProt 데이터출처](https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20length%3A%5B900%20TO%20950%5D&format=fasta&size=500) · [기본보고서](ESM2_20261027_164300_125163.md)

원본: ../results/batch_inference_20261028_112900_601721/summary.json, batch_B/summary.json, resources.csv, embeddings.pt, tokenized_inputs.pt, 명령·exit·로그. 재현코드는scripts/batch_followup에있다.

### 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.


## 6. 실제 실행 코드·명령어 전문

<!-- EXECUTED_SCRIPTS_START -->

### 원문과 출처 검증

아래 코드 블록은 원본 내용을 생략하거나 재작성하지 않았다. 각 파일의 원래 경로·원문 SHA256·출처 상태를 함께 기록했다. `실행 run 저장 snapshot`을 우선하고, 현재 소스만 남은 경우에는 당시 파일과 같다고 단정하지 않는다. 동일 바이트 버전은 한 번만 수록하고 해당 경로들을 함께 표시한다.

GitHub에는 `.py`/`.sh`를 별도 파일로 두지 않는다. 코드 블록을 원래 경로에 저장하거나 [복원 안내](../../docs/RESTORE_FILES.md)의 코드로 로컬에서 복원한다. 라이브러리 전체 소스는 이 사용자 실행 스크립트와 구분하며 패키지 버전/출처를 따른다.

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

