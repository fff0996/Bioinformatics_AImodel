# ESM-2 최종 통합 보고서

기본 실험과 모든 후속 실험을 이 문서에 함께 수록했다. 다른 기본/후속 문서를 따로 읽을 필요 없이 아래 요약과 본문을 보면 된다.

## 1. 입력과 메모리 결론

단백질 아미노산 서열 → 단백질당1280차원 임베딩 또는 fluorescence 예측. 체크포인트 esm2_t33_650M_UR50D(명목650M).

**목표는 메모리 할당80% 조건 확인**이다. 전체공유RAM121.688GiB의80%는약97.35GiB다. GPU-Util은별도연산활동지표이며이목표의성공판정에쓰지않는다.

|작업|관측메모리 peak GiB|공유RAM대비|해당 입력·배치·정밀도|80% 확인/제약|
|---|---|---|---|---|---|
|추론|**64.416**|52.94%|실제900~950아미노산 단백질384개. GPU토큰 배열384×952(BOS/EOS포함),FP32.|80% 미확인; 다음배치를예상예약량guard로건너뜀|
|부분 파인튜닝|**3.623**|2.98%|실제237아미노산 서열128개 학습. 토큰예산4096 가변배치,FP32; 마지막Transformer block·최종LayerNorm·회귀head만학습.|학습배치확대실험 미실시; 전체모델학습값 아님|

메모리는PyTorch의경우CUDA allocated peak다. reserved는예약량으로별도다. **AlphaFold2는컴파일·warmup등을포함한JAX프로세스누적peak**이므로CUDA구간peak와같은정의로비교하지않는다. RSS와CUDA를더하지않는다. 표는시험한조건의관측값이며모델의절대최대나모든모델의80%달성을의미하지않는다.

## 2. 기본 정보·입력 검사·추론·학습·평가

## DGX Spark ESM-2 성능 검증 보고서


작성: 2026-10-27T17:01:39 (KST). 기본 run: `20261027_164300_125163`, 보완 run: `supplement_20261027_165443_595867`.

### GPU 사용률 결론

전체 ESM-2 실험의 관측 최대 GPU-Util은 **96%**다. 기본 run의 긴 서열 조건(length_513_1022, 실제 516–976 residues)에서 기록됐다. 관측 구간을 늘린 40회 보완 측정의 같은 긴 서열 조건에서는 **평균 94.10%, 최대 95%**였다. 다른 실행에서 기록한 순간 최대와 지속 평균을 구분해야 한다.

배치(토큰 예산 4096)·FP32·650M 체크포인트를 사용률에 맞춰 바꾸지 않은 기준 조건에서 이미 평균 80%를 초과했다. 파인튜닝은 평균 88.14%, 최대 95%였다. GPU-Util은 장치 전체 커널 활동 시간 비율이며 메모리 점유율이나 최대 연산 성능 비율이 아니다. 메모리는 별도 계측으로, 학습 중 CUDA peak allocated는 3.623 GiB였다.

### 결과와 완료 범위

실제 GB10 GPU에서 ESM-2 650M 임베딩 추론, 입력 개수·서열 길이 비교, 공개 fluorescence 라벨 기반 부분 파인튜닝, 저장, 독립 프로세스 재로드 및 test 예측 검증을 완료했다. 기본 작업은 중단 중에도 계속 실행돼 2026-10-27 16:45:26 KST에 종료 코드 0으로 끝났다. 기존 실행을 중복 시작하지 않았고, 보완 측정은 기본 작업 종료를 확인한 후 별도 폴더에서 순차 실행했다.

기본 개수 실험에서 평균 GPU-Util 89.4–92.5%가 관찰됐다. 80%를 맞추기 위한 배치·정밀도·모델 변경은 하지 않았다. 모델 실행·저장 성공과 예측 정확도는 구분해야 한다. 128개 test의 Spearman은 0.0241로 매우 낮아 실용적인 fluorescence 예측 성능을 입증하지 못했다.

### 모델 적재 기준값과 입력별 추가 메모리

입력 forward 전 별도 새 프로세스에서 원본 공개 체크포인트를 로딩만 했다. CUDA는 먼저 초기화했고 optimizer·tokenization·forward는 실행하지 않았다. 이론 FP32 가중치는 2.425 GiB이고 실측 post-load allocated는 **2.490 GiB**, reserved는 2.559 GiB였다. GPU 할당은 가중치 이론값 외 프레임워크 할당도 포함한다. RSS는 2.087 GiB였다.

| 실제 서열 길이 | 실제 batch | GPU 평균 / 최대 % | 모델 로드 allocated GiB | 실행 allocated peak GiB | 기준 대비 참고 증가 GiB |
| --- | --- | --- | --- | --- | --- |
| 56–127 | 16 | 93.56 / 95 | 2.490 | 2.713 | 0.223 |
| 140–254 | 16 | 92.62 / 95 | 2.490 | 2.904 | 0.414 |
| 280–461 | 6–10 | 93.51 / 95 | 2.490 | 2.911 | 0.421 |
| 516–976 | 4–7 | 94.10 / 95 | 2.490 | 3.214 | 0.725 |

적재 기준은 다른 run에서 측정했으므로 차이는 참고값이며 같은 run의 전후 차이가 아니다. 추가량은 입력·중간값·작업 buffer 등을 포함한다. 학습 추가량에는 gradient·optimizer가 더 포함된다. 기준 원본: `/home/hyein/ai-benchmark/esm2/results/loading_20261027_190701_789190/loading.json`.

#### 실제 파인튜닝 데이터 예

TAPE fluorescence는 단백질 서열을 입력받고 측정된 log_fluorescence 연속값을 예측하는 회귀 과제다.

| split / ID | 실제 길이 | 실제 서열 앞부분 | 정답 log_fluorescence |
| --- | --- | --- | --- |
| train / train_20952 | 237 | SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGD… | 3.789218 |
| valid / valid_5238 | 237 | SKGEELFTGEVPILVELDGDVNGHKFSVSGEGEGD… | 3.443987 |
| test / test_20952 | 236 | SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGD… | 3.200874 |

가중치를 갱신한 대상은 전체 650M backbone이 아닌 마지막 block·LayerNorm·회귀 head다.

### 모델·환경·기준 조건

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

### 입력 확인과 데이터 출처

UniProtKB reviewed release 2026_03 (2026-09-02), API의 첫 500개 FASTA 레코드를 확보했다. 전체 UniProt 또는 무작위 종 대표 표본이 아니다. 원본 500개, 고유 서열 499개, 길이 56–1020 residues, 표준 아미노산 20종, 빈 서열 없음이다. 동일 서열 1개는 개수 실험에서 제거했다. 라벨이 없는 임베딩 실험 입력이다.

개수 비교는 50–128 / 129–256 / 257–512 / 513–1022의 4개 길이 구간에서 각각 seed 42로 섞어 교차 배치한 중첩 부분집합 32/96/160개다. 구간별 개수는 조건마다 각각 8/24/40개로 같지만 구간 내부 길이 분포와 최대 길이는 완전히 일치하지 않는다. 개수 조건의 평균 길이는 328.28/341.73/340.82여서 처리량 차이를 순수한 개수 효과로 단정할 수 없다. 길이별 입력 목록, 실제 batch shape, padding을 저장했다.

TAPE 공식 fluorescence LMDB를 사용했다. 단백질 GFP 변이 서열에서 float32 [1] log_fluorescence를 예측하는 회귀 과제다. 분류 정확도 지표를 사용하지 않았다.

| 원본 split | 레코드 | 길이 범위 | 라벨 평균 | 누락/비유한 라벨 |
| --- | --- | --- | --- | --- |
| train | 21446 | 235–237 | 3.181 | 0 |
| valid | 5362 | 235–237 | 3.173 | 0 |
| test | 27217 | 235–237 | 2.083 | 0 |

원본 split 간 동일 서열 중복은 모두 0이다. 기존 split을 유지한 채 train 128 / validation 64 / test 128개를 각 split에서 독립적으로 seed 42로 선정했다. test로 하이퍼파라미터나 epoch를 선택하지 않았다. GFP 변이들이기 때문에 높은 상동성은 남으며, 독립 단백질 family 일반화를 검증하지 않는다. 사전학습 데이터와의 중복도 검증하지 않았다.

#### 실제 입력·출력 예

```text
>sp|P01764|HV323_HUMAN
MEFGLSWLFLVAILKGVQCEVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYA...
original residues = 117; truncated = False
smoke tokens = [2, 196]; pooled embeddings = [2, 1280] float32 on CUDA
fluorescence: amino-acid sequence → mean pooled 1280-vector → Linear(1280,1) → log_fluorescence
```

### 측정 범위와 방법

전·후 torch.cuda.synchronize()를 호출했다. 추론 실행 시간에는 CPU 토큰화, CUDA 전송, forward, pooling, pooled 결과의 CPU 전송이 포함되며 모델 로딩, warmup, torch.save는 제외한다. preprocess_seconds는 batch 목록 구성 시간만 의미한다. 초기 다운로드와 전체 데이터 검사 시간은 별도 계측하지 않아 N/A다. 모델 로딩에는 체크포인트 읽기·모델 생성·CUDA 전송을 포함한다. 컴파일은 사용하지 않았다.

기본 run 모델 로딩 2.344초, smoke/warmup 0.507초. 보완 run 모델 로딩 2.344초, 각 길이 조건 전체 1 pass를 별도 warmup했다. 기본 개수 조건은 3회 반복, 보완 길이 조건은 동일 16개 입력을 40회 반복했다. 반복은 고유 데이터 추가가 아니다.

GPU-Util과 RAM은 0.5초 간격으로 표본 수집했다. GPU-Util은 장치 전체의 커널 실행 시간 비율로 최대 FLOPS/SM occupancy가 아니며 디스플레이·Firefox 등도 포함된다. nvidia-smi 자체 내부 집계 창과 조회 지연이 있어 독립 표본으로 해석하지 않는다. 보완 실험은 query 시작·종료가 모두 동기화된 실행 구간 안에 있는 표본만 포함하고 첫 2초를 제외한 평균도 별도 보관했다. 기본 run은 phase 단위이므로 매우 짧은 입력 목록·파일 저장 시간도 GPU/RAM 표본에 섞일 수 있다.

CUDA peak allocated/reserved는 PyTorch allocator 값이다. Spark는 CPU/GPU 공유 DRAM이므로 RSS와 CUDA 메모리를 더하지 않았다. RSS는 0.5초 sampled peak이며 정확한 최대값이 아니다. nvidia-smi 장치 전체 memory 사용량은 Not Supported이므로 N/A다. 시스템 MemAvailable과 swap은 별도 기록했다.

### A. 서열 개수 증가: 원본 완료 결과

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

### B. 서열 길이 비교: 관측 구간을 늘린 보완 결과

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

### 부분 파인튜닝·검증·저장

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

### 검증·재현과 산출물

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

### 한계와 다음 단계

가용 메모리는 모든 측정에서 충분했고 swap 증가는 없었으며 OOM은 발생하지 않았다. 총 샘플 수를 늘리는 것과 단일 GPU 입력을 키우는 것은 다르다. 사용률 80%는 일반 조건에서 이미 초과했으며 그 자체가 모델 정확도나 최대 하드웨어 성능을 뜻하지 않는다. 반복 3/40회는 한 프로세스 내 warm 상태 반복으로 독립 실행·독립 seed 실험이 아니다. 입력·모델의 사전 다운로드 캐시는 warm 상태이며 전체 생물학적 활용 범위를 검증한 것은 아니다.

후속 순서는 AlphaFold2 — ColabFold를 통한 실행이다. 현재 host hyein의 실제 그룹에 docker가 없고 /var/run/docker.sock은 root:docker 0660이다. Docker 조회는 샌드박스 밖에서도 permission denied여서 기존 이미지·컨테이너·마운트·가중치를 검증할 수 없다. 자동 권한 변경이나 sudoers 변경은 하지 않았다. 사용자가 현재 세션의 Docker 접근을 복구해야 기존 ColabFold 검증을 이어갈 수 있다.

### 공식 출처

- [ESM-2 공식 구현·체크포인트](https://github.com/facebookresearch/esm)
- [공식 토큰 예산·입력 truncation 구현](https://github.com/facebookresearch/esm/blob/main/scripts/extract.py)
- [TAPE 과제·데이터](https://github.com/songlab-cal/tape)
- [TAPE 공식 데이터 다운로드 목록](https://github.com/songlab-cal/tape/blob/master/download_data.sh)
- [UniProt 공식 REST API](https://www.uniprot.org/help/api_queries)
- [GPU-Util 정의·계측 지원](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [DGX Spark 공유 메모리·nvidia-smi memory 제한](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)

## 3. 후속 실험 통합 — 배치 확대 및 자원 검증

## ESM-2 실제 긴 단백질 서열의 배치별 추론 메모리


실행 `batch_inference_20261028_112900_601721` · 상태 **complete** · 종료 사유 `predicted_reservation_exceeds_available_budget`

### 입력과 배치의 의미

배치는 한 번의 GPU forward에 넣는 **단백질 서열 개수**다. 아미노산 하나가 보통 토큰 하나이며, 이번 입력은 실제 UniProt reviewed 서열900~950아미노산이다. 모든배치에950아미노산 서열을 포함해 BOS/EOS 포함GPU입력 길이를952로 유지했다. 원본 전체 개수와 동시에 계산하는 batch를 구분한다. 전체pool에서 중첩고유부분집합을 사용하고, 같은batch를 측정 반복하므로 고유서열수와총처리건수는다르다. 자르거나 인공서열을 만들지 않았다. 라벨은없고임베딩추론만수행했다.

### 고정 조건과 측정 정의

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

### 결과

시험한 최대 성공배치 **384**. allocated80%도달: False,90%도달: False. 이는관측조건의결과이며모델의절대최대한계가아니다.

시스템가용RAM16GiB·swap증가2GiB guard와다음배치예약량예측으로확대를중단한다. worker실패가있으면그조건의exit/log를보존하며성공행에포함하지않는다. 예상guard에의해다음배치를건너뛴것은그배치의실제실패나90%불가능증명이아니다. RSS/가용량은0.5초대기+조회시계열의sampled값이다. 메모리할당peak는CUDAcounter이다.

각조건embedding shape B×1280·finite·저장후정확재로드를확인했고,같은첫서열의batch1대비임베딩일치도도검사했다. 학습·정확도평가·독립생물학적일반화평가를수행한결과가아니다. 일반조건인토큰예산4096경로와다른명시적batch용량실험이므로기본성능결과를대체하지않는다.

[공식ESM](https://github.com/facebookresearch/esm) · [UniProt 데이터출처](https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20length%3A%5B900%20TO%20950%5D&format=fasta&size=500) · [기본보고서](ESM2_20261027_164300_125163.md)

원본: ../results/batch_inference_20261028_112900_601721/summary.json, batch_B/summary.json, resources.csv, embeddings.pt, tokenized_inputs.pt, 명령·exit·로그. 재현코드는scripts/batch_followup에있다.

## 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.
