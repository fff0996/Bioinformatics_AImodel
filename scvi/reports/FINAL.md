# scVI 최종 통합 보고서

기본 실험과 모든 후속 실험을 이 문서에 함께 수록했다. 다른 기본/후속 문서를 따로 읽을 필요 없이 아래 요약과 본문을 보면 된다.

## 1. 입력과 메모리 결론

raw count의 세포×유전자 행렬과 실제생물학적batch → 세포당10차원 latent. 처음부터학습한SCVI 모델.

**목표는 메모리 할당80% 조건 확인**이다. 전체공유RAM121.688GiB의80%는약97.35GiB다. GPU-Util은별도연산활동지표이며이목표의성공판정에쓰지않는다.

|작업|관측메모리 peak GiB|공유RAM대비|해당 입력·배치·정밀도|80% 확인/제약|
|---|---|---|---|---|---|
|추론|**0.270**|0.22%|한번에10176세포×2000유전자,FP32. 기존저장모델의latent posterior mean.|80% 미확인; 확보한학습세포10176개전체batch까지시험|
|처음부터 학습|**0.097**|0.08%|학습세포10176개를128세포×2000유전자씩분할,400epoch,FP32.|파인튜닝아님; 학습batch확대 미실시|

메모리는PyTorch의경우CUDA allocated peak다. reserved는예약량으로별도다. **AlphaFold2는컴파일·warmup등을포함한JAX프로세스누적peak**이므로CUDA구간peak와같은정의로비교하지않는다. RSS와CUDA를더하지않는다. 표는시험한조건의관측값이며모델의절대최대나모든모델의80%달성을의미하지않는다.

## 2. 기본 정보·입력 검사·추론·학습·평가

## scVI: 모델 크기·실제 학습 입력·GPU 사용률·메모리 검증


작성: 2026-10-27T19:13:56+09:00

### 결과 요약

10x PBMC의 원시 UMI count로 scVI를 데이터별로 처음부터 학습했다. 1024/4096/10176개 훈련세포를 같은 2000개 유전자·batch 128·FP32·400 epochs로 비교하고, 마지막 저장 모델 한 개로 1024/4096/12721세포의 latent 추론을 측정했다. GPU 사용률과 GPU 메모리를 같은 표에 제시한다. 모든 GPU 작업은 순차 실행했다.

세포 수를 늘리면 전체 처리 시간은 늘었지만 한 배치의 크기는 같으므로 peak 메모리는 거의 같았다. 이 조건에서는 80% GPU-Util에 도달하지 않았으며 배치·정밀도·모델을 사용률 목표에 맞춰 바꾸지 않았다. batch 통합 관련 지표는 원래 baseline에서도 두 batch가 잘 섞여 있어 뚜렷한 개선을 입증하지 못했다.

### 1. 어떤 공개 모델이며 얼마나 큰가

| 항목 | 확인값 |
| --- | --- |
| 공개 구현 | scvi-tools 1.5.1, scvi.model.SCVI |
| 공개 사전학습 체크포인트 | 사용하지 않음. SCVI 기본 구조를 seed42로 초기화해 데이터별 학습 |
| 전체 파라미터 | 1,293,526 |
| configured requires_grad 파라미터 | 1,293,526 |
| 실제 첫 backward에서 gradient가 있던 파라미터 | 778,884 |
| latent 추론 z_encoder 파라미터 | 258,964 |
| 구조 | 2000 genes → hidden128, 1 layer → Normal latent10; decoder는 batch-conditioned ZINB |
| dropout / dispersion | 0.1 / gene |
| 가중치·계산 정밀도 | FP32, TF32 disabled, Lightning 32-true |
| 이론 FP32 가중치 용량 | 0.004819 GiB (parameter count × 4 bytes; 실행 메모리 아님) |
| 모델 초기화 동일성 | 세 학습 조건 모두 SHA-256 a1a88a8f3b1b835b70a4716a83fe2013d646d8c0ef2bfff34afc5165b4bf4926 |
| 환경 | DGX Spark GB10 ARM64; NVIDIA PyTorch 26.06 이미지 내 별도 venv |
| PyTorch / CUDA | 2.13.0a0+8145d630e8.nv26.06 / CUDA13.3 forward compatibility |
| 기존 환경 보존 | scGPT 컨테이너·이미지·기존 host PyTorch 수정 없음 |

requires_grad 개수와 실제 사용되는 branch의 gradient 개수는 다르다. observed library size를 사용하는 기본 경로에서 쓰이지 않는 branch도 module의 전체 파라미터에 포함된다. 학습 후 17개 parameter tensor의 갱신을 확인했다. scVI를 사전학습 모델 파인튜닝이나 scANVI 세포 유형 분류로 표현하지 않는다.

### 2. 모델만 로드하면 얼마를 할당하는가

| 저장 모델 | 로드 전 allocated GiB | 로드 직후 allocated GiB | 모델 로드 증가 GiB | 로드 peak GiB | CPU RSS 직후 GiB |
| --- | --- | --- | --- | --- | --- |
| train_1024 | 0.000000 | 0.004826 | 0.004826 | 0.009651 | 1.054 |
| train_4096 | 0.031250 | 0.036076 | 0.004826 | 0.040901 | 1.557 |
| train_10176 | 0.031250 | 0.036076 | 0.004826 | 0.040901 | 1.604 |

첫 모델의 깨끗한 CUDA allocator 기준으로 **모델 적재 직후 0.004826 GiB (약 4.94 MiB)**였다. CUDA 초기화와 saved AnnData의 CPU 읽기는 기준 시점 전에 끝냈고, GPU forward와 optimizer 로드는 하지 않았다. 따라서 GPU에는 모델만 올라간 상태지만 CPU RSS에는 미리 읽은 데이터·라이브러리·registry가 포함된다.

두 번째·세 번째 모델은 같은 독립 검증 프로세스에서 앞선 inference 뒤의 0.03125 GiB 상주 할당이 남아 있어 absolute post-load가 더 크다. 이 32 MiB의 원인은 본 실험에서 세분화하지 않았다. 각 모델 적재의 전후 차이는 같았다. 물리 메모리 사용량과 PyTorch allocator 수치를 동일시하지 않는다.

### 3. 학습 데이터는 어떻게 생겼는가

10x Genomics Cell Ranger 2.1.0의 pbmc4k·pbmc8k **filtered cell barcode의 raw UMI count**를 사용했다. filtered는 세포 선택을 뜻하며 log-normalization이 적용됐다는 뜻이 아니다. count 값은 정수값을 가진 float32 CSR sparse이고 비음수·유한값·고유 cell ID를 검사했다. 원본 12721세포 × 33694유전자에서 훈련세포만으로 batch-aware Seurat v3 HVG 2000개를 선택했다. count layer에는 normalize/log1p를 적용하지 않았다.

| 자료 | 원본 shape | 실제 batch | donor | 출처 |
| --- | --- | --- | --- | --- |
| PBMC4k | 4340 × 33694 | pbmc4k | 동일 healthy donor | 10x Cell Ranger2.1.0 |
| PBMC8k | 8381 × 33694 | pbmc8k | 동일 healthy donor | 10x Cell Ranger2.1.0 |
| 학습 입력 | 12721 × 2000 | 2개 실제 실험 batch | 1명 | HVG는 training cells에서만 선택 |

세포 유형 정답 라벨은 없다. 학습의 복원 대상은 입력 raw count 자체이며 sample_id는 batch covariate다. 이를 정답 cell-type 라벨로 쓰지 않았다. 다음은 각 batch의 앞 3개 훈련세포와, 이 6개 세포에서 count 합이 큰 8개 유전자를 고른 실제 행렬 예시다. 독자가 비영 count를 확인하기 위한 전시용 선택이며 실제 학습에는 선정된 2000개 유전자를 전부 사용했다.

| cell | TMSB4X | FTL | MT-CO1 | LYZ | S100A8 | MT-CO2 | S100A9 | MT-CO3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pbmc4k_AAACCTGAGAAGGCCT-1 | 51 | 24 | 26 | 31 | 12 | 20 | 12 | 17 |
| pbmc4k_AAACCTGAGACAGACC-1 | 51 | 62 | 36 | 51 | 21 | 29 | 33 | 21 |
| pbmc4k_AAACCTGAGATAGTCA-1 | 43 | 20 | 31 | 31 | 62 | 20 | 33 | 18 |
| pbmc8k_AAACCTGAGCATCATC-1 | 22 | 7 | 22 | 0 | 1 | 12 | 2 | 12 |
| pbmc8k_AAACCTGAGCTAACTC-1 | 42 | 43 | 27 | 16 | 14 | 12 | 14 | 15 |
| pbmc8k_AAACCTGCACATTAGC-1 | 32 | 6 | 3 | 0 | 0 | 6 | 1 | 6 |

같은 행에는 sample_id, original_barcode, total_umi, detected_genes, split 메타데이터가 연결된다. 전체 셀·유전자 목록은 input run의 cells.csv/genes.csv, 실제 행렬 예시는 reports/training_input_example.csv에 보관했다. latent 출력은 각 세포당 float32 숫자 10개이며 latent_example.csv에 실제 값을 저장했다.

```text
원본 [12721 cells, 33694 genes] raw UMI
 → training-only HVG 선택 [12721, 2000]
 → GPU minibatch [최대 128, 2000] float32
 → latent posterior mean [최대 128, 10]
 → batch-conditioned ZINB decoder / count reconstruction
```

#### 분리와 규모 비교

| 집합 | 세포 수 | pbmc4k / pbmc8k |
| --- | --- | --- |
| train_1024 | 1024 | 350 / 674 |
| train_4096 | 4096 | 1398 / 2698 |
| train | 10176 | 3472 / 6704 |
| validation | 1272 | 434 / 838 |
| test | 1273 | 434 / 839 |

training 부분집합은 중첩 구조이고 batch·UMI 깊이 구간 분포를 층화했다. 깊이 구간은 전체 자료 QC 정보를 사용하므로 완전히 blind한 표본 선정은 아니다. Validation 1272개와 test 1273개는 세 모델에서 동일하며 parameter update에 사용하지 않았다. optimizer와 epoch 수·중단 정책을 고정했고 test로 조건을 선택하지 않았다. 같은 donor 안에서의 random split이며 donor-independent 일반화 검증은 아니다.

### 4. 입력 조건에 따른 GPU 사용률과 메모리

기본 batch 128, FP32를 고정했다. 아래 모델 로드 기준과 실행 peak는 서로 다른 run에서 얻었으므로 증가량은 **참고 차이**다. 정확히 같은 run의 baseline 전후 차이라고 쓰지 않는다. 학습 추가량에는 gradient·optimizer·중간값·작업 buffer가 포함된다.

| 학습 고유세포 / genes | 실제 GPU 입력 | GPU 평균 / 최대 % | 모델 로드 기준 GiB | fit allocated peak GiB | 기준 대비 참고 증가 GiB |
| --- | --- | --- | --- | --- | --- |
| 1024 / 2000 | [≤128, 2000] | 21.80 / 30 | 0.004826 | 0.097363 | 0.092537 |
| 4096 / 2000 | [≤128, 2000] | 26.55 / 32 | 0.004826 | 0.097439 | 0.092614 |
| 10176 / 2000 | [≤128, 2000] | 25.76 / 33 | 0.004826 | 0.097439 | 0.092614 |

| 추론 고유세포 / genes | 실제 GPU 입력 | GPU 평균 / 최대 % | 모델 로드 기준 GiB | allocated peak GiB | 기준 대비 참고 증가 GiB |
| --- | --- | --- | --- | --- | --- |
| 1024 / 2000 | [≤128, 2000] | 24.07 / 29 | 0.004826 | 0.039019 | 0.034193 |
| 4096 / 2000 | [≤128, 2000] | 27.73 / 31 | 0.004826 | 0.039019 | 0.034193 |
| 12721 / 2000 | [≤128, 2000] | 29.10 / 33 | 0.004826 | 0.039019 | 0.034193 |

세포당 2000개 입력 유전자와 hidden/latent 구조가 같아 모델 크기는 동일하다. 세포 수는 전체 처리 건수를 늘릴 뿐 GPU에 한번에 올라가는 크기를 바꾸지 않으므로 peak가 거의 일정했다. 유전자 수를 바꾸면 encoder/decoder parameter 수와 복원 과제가 함께 바뀌므로 같은 모델의 단순 길이 비교로 취급하지 않았다.

### 5. 학습·추론 시간과 처리량

| 학습 세포 | epochs / optimizer steps | 총 학습 처리 건수 | train loop s | validation loop s | fit wall s | cell/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 400 / 3200 | 409600 | 19.424 | 0.420 | 20.263 | 21087.3 |
| 4096 | 400 / 12800 | 1638400 | 60.201 | 0.424 | 61.060 | 27215.3 |
| 10176 | 400 / 32000 | 4070400 | 159.065 | 0.415 | 159.903 | 25589.5 |

| 추론 고유세포 | 반복 / 총 처리 | 실행 s | 저장 s | cell/s | RSS sampled peak GiB | MemAvailable 최소 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 3546 / 3631104 | 15.002 | 0.003 | 242037.7 | 1.616 | 113.014 |
| 4096 | 1020 / 4177920 | 15.010 | 0.006 | 278335.3 | 1.613 | 112.974 |
| 12721 | 352 / 4477792 | 15.013 | 0.016 | 298263.8 | 1.615 | 113.044 |

추론은 동일한 10176세포 학습 모델 하나를 사용했고 posterior mean을 얻었다. 0.5초 GPU 표본을 충분히 확보하려고 조건마다 최소 15초·최소 3회 반복했다. 많은 총 처리 건수는 동일 데이터의 반복이며 고유 데이터가 그만큼 많다는 뜻이 아니다. 매 반복 출력이 첫 출력과 일치하는지도 검사했다.

| 학습 세포 | 입력 준비 s | 초기화 s | 평가 s | 모델/AnnData 저장 s | RSS peak GiB | MemAvailable 최소 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 0.016 | 0.214 | 0.042 | 0.095 | 1.955 | 112.789 |
| 4096 | 0.017 | 0.168 | 0.039 | 0.065 | 1.989 | 112.478 |
| 10176 | 0.017 | 0.010 | 0.032 | 0.061 | 2.060 | 112.388 |

초기 count 입력 준비는 3.832초였다. training은 400 epochs, Adam lr=1e-3, weight_decay=1e-6, eps=0.01, KL warmup400 epochs, validation 주기20, early stopping False다. 각 조건은 단일 seed·단일 학습 실행이다. 큰 training set은 고유세포뿐 아니라 optimizer step 수도 많아 정확도 차이를 단순한 입력 개수 효과로 분리할 수 없다.

### 6. 평가·batch integration·재로드

| 학습 고유세포 | validation ELBO | 독립 test ELBO | 독립 재로드 test cells | 재로드 최대 절대 차이 |
| --- | --- | --- | --- | --- |
| 1024 | -620.406 | -597.572 | 1273 | 0.0 |
| 4096 | -567.465 | -548.418 | 1273 | 0.0 |
| 10176 | -554.151 | -536.274 | 1273 | 0.0 |

scvi-tools 1.5.1의 get_elbo는 ELBO를 반환해 높을수록 좋다. 학습 history/그래프의 elbo_train은 loss 관례의 양수 negative ELBO라 부호를 구분해야 한다. ELBO는 cell-type accuracy가 아니다.

| held-out test1273 지표 | log-normalized PCA | SCVI latent |
| --- | --- | --- |
| batch_silhouette | 0.000836 | 0.001080 |
| mean_normalized_batch_knn_entropy | 0.908868 | 0.909826 |
| batch_prediction_balanced_accuracy | 0.508183 | 0.500000 |

batch silhouette는 0에 가깝고 30-neighbor batch entropy는 두 방법 모두 높다. 실제 batch를 예측하는 진단 분류기의 balanced accuracy도 둘 다 chance 수준이다. baseline부터 잘 섞여 있던 같은 donor 자료이므로 큰 batch correction 개선을 주장할 수 없다. cell-type 라벨 없이 과도한 biological mixing 여부를 검증하지 못한다. scVI 조건부 생성 기능으로 두 batch를 평균한 normalized expression 예시 [128,2000]도 저장했지만 이 값은 raw 학습 입력이 아니다.

![실제 count 입력·학습곡선·batch별 임베딩](scvi_data_and_training.png)

저장 모델 3개를 새 OS 프로세스에서 다시 불러와 각 1273개 test의 10차원 latent가 원본과 정확히 일치함을 확인했다. 공식 model.pt·registry·init 설정·var 순서·AnnData를 보관했고 별도로 trainer.ckpt(optimizer/epoch), RNG도 저장했다. 상태 저장은 했지만 학습 재개 자체를 검증하지 않아 재개 성공으로 쓰지 않는다.

### 계측 범위·제약

시간 구간 경계에서 torch.cuda.synchronize()를 호출했다. train loop는 데이터 로딩·전송·forward·backward·optimizer·일부 프레임워크 overhead를 포함하며 validation loop와 구분한다. fit wall은 Trainer 전체다. inference 시간은 표준 scVI 데이터 로더·전송·encoder·CPU latent 반환·유한값/반복 일치 검사까지 포함하고 파일 저장은 제외한다. 별도 JIT/torch.compile은 사용하지 않았다. 작은 GPU smoke [32,2000]→[32,10] CUDA float32를 확인했고 추론 첫128셀 warmup 0.004초는 본 측정에서 제외했다.

GPU-Util·RSS·MemAvailable·swap은 0.5초 간격 sampled 값이다. GPU 평균은 해당 phase에 시작하고 동기화 측정 구간 안에서 끝난 query 표본을 사용한다. NVML 내부 집계 window는 phase 경계를 걸칠 수 있다. validation은 합계 약0.4초처럼 짧아 단독 GPU 평균을 신뢰할 수 없어 주 비교표에서 사용하지 않았다. RSS sampled peak는 정확한 최고치가 아니다. GPU-Util은 장치 전체 커널 활동 시간 비율이며 FLOPS/SM occupancy/메모리 비율이 아니다. 디스플레이·브라우저 부하도 포함될 수 있다.

메모리 비교는 PyTorch allocator allocated/reserved, 프로세스 RSS, 시스템 MemAvailable을 구분한다. 공유 DRAM121.688GiB에서 이를 합산하지 않았고 GPU dedicated VRAM 비율로 표현하지 않았다. 시스템 swap 증가는 없고 가용 RAM은 충분했다. 낮은 GPU 사용률의 특정 원인을 CPU profiler로 확정하지 않았다. 작은 encoder/MLP·기본 batch·일반 데이터 로딩 경로가 가능한 설명이지만 병목 분해는 미수행이다.

복구 과정: 첫1024세포 학습과 저장 후 로그 event에 condition 키가 중복돼 Python 오류가 났고, 다음 실행에서 중복 학습 없이 저장 결과를 재사용했다. 4096/10176 학습 완료 후에는 load API에 device=0을 장치 개수로 전달해 오류가 나서 device=1로 수정했다. 세 학습을 모두 재사용하고 로딩·추론 단계만 새 run에서 이어갔다. 실패 run의 종료 코드1과 로그를 보존했으며 최신 완료 run은 종료 코드0이다.

### 저장 경로·실행

| 항목 | 호스트 경로 |
| --- | --- |
| 입력 확인 | /home/hyein/ai-benchmark/scvi/results/inputs_20261027_093652_887319 |
| 최종 run | /home/hyein/ai-benchmark/scvi/results/scvi_20261027_185901_460537 |
| raw source·processed_counts.h5ad | /home/hyein/ai-benchmark/scvi/data |
| 세 학습 모델 | /home/hyein/ai-benchmark/scvi/models/scvi_20261027_184921_028161/train_1024, /home/hyein/ai-benchmark/scvi/models/scvi_20261027_185327_767988/train_4096, /home/hyein/ai-benchmark/scvi/models/scvi_20261027_185327_767988/train_10176 |
| 독립 재로드·모델 로딩 기준값 | /home/hyein/ai-benchmark/scvi/results/scvi_20261027_185901_460537/independent_reload.json / /home/hyein/ai-benchmark/scvi/results/scvi_20261027_185901_460537/loading_baselines_annotated.json |
| 입력 예시 | /home/hyein/ai-benchmark/scvi/reports/training_input_example.csv |
| 비교 CSV | /home/hyein/ai-benchmark/scvi/reports/comparison.csv |
| 보고서 | /home/hyein/ai-benchmark/scvi/reports/scVI.md |

각 학습 source_run의 resources.csv와 epoch_timings, 학습곡선, test latent, 명령·환경·패키지 버전·입력split·초기화 해시를 보관했다. 최신 run에서 재사용한 학습 숫자의 원본 시계열은 source_run으로 연결된다. 모든 컨테이너 결과는 host bind /work 경로에 저장된다.

```bash
cd /home/hyein/ai-benchmark
# 동일한 준비 입력으로 새로운 run을 만들며 기존 결과를 보존한다.
python3 scvi/scripts/launch.py
# 재로드 검증은 기존 GPU 작업이 끝난 후 하나씩 실행한다.
```

### 공식 출처

- [scVI API·기본 구조](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html)
- [공식 count 입력·학습 예제](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html)
- [scvi-tools1.5.1 source](https://github.com/scverse/scvi-tools/tree/1.5.1)
- [10x PBMC4k (CC BY4.0)](https://www.10xgenomics.com/datasets/4-k-pbm-cs-from-a-healthy-donor-2-standard-2-1-0)
- [10x PBMC8k (CC BY4.0)](https://www.10xgenomics.com/datasets/8-k-pbm-cs-from-a-healthy-donor-2-standard-2-1-0)
- [Spark 공유 메모리](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)

## 3. 후속 실험 통합 — 배치 확대 및 자원 검증

## scvi 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_142752_517769` · 종료사유 `available_unique_input_limit`

### 입력과 고정 조건

{
  "source": "saved full10176cell SCVI model and its raw-count AnnData, fixed2000genes; latent posterior mean",
  "unique_available_inputs": 10176,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

### 배치별 측정

측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.

|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|128|13607|1741696|15.001|116105.6941|0.0048|0.039|0.0342|0.041|0.032|15.6897|19.0|1.5399|112.4895|
|256|11926|3053056|15.0007|203528.1284|0.0048|0.0419|0.0371|0.0566|0.0345|17.4138|20.0|1.5383|112.371|
|512|10058|5149696|15.0006|343299.507|0.0048|0.0478|0.0429|0.0566|0.0393|20.1724|24.0|1.5434|112.4101|
|1024|7756|7942144|15.0008|529446.5185|0.0048|0.0595|0.0546|0.0762|0.0489|26.1724|29.0|1.5465|112.4435|
|2048|4718|9662464|15.0018|644088.376|0.0048|0.084|0.0791|0.0859|0.069|24.2414|26.0|1.534|112.3655|
|4096|2211|9056256|15.0009|603715.4809|0.0048|0.1318|0.127|0.1504|0.1083|24.8621|28.0|1.5626|112.3741|
|8192|154|1261568|15.0271|83952.6489|0.0048|0.2232|0.2184|0.2461|0.1834|5.7931|7.0|1.6075|112.1286|
|10176|122|1241472|15.0815|82317.5007|0.0048|0.2696|0.2647|0.2871|0.2215|4.7241|8.0|1.6315|112.0164|

### 결과 해석과 검증

최대시험성공batch 10176. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html) · [기존 모델 보고서 디렉터리](.)

## 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.
