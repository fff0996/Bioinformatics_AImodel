# 모델별 실제 입력·GPU 사용률·메모리

> 최신 읽기 경로: [기본·후속을 합친 최종 보고](FINAL_REPORT.md). 이 문서는 이전 실행/계측 기록이며, 80% 메모리 조건 확인은 scGPT 학습용량probe에 한정된다. 다른 모델은 미확인이다.

갱신: 2026-10-27T21:38:29

**GPU 사용률(%)은 계산 활동 시간의 비율이고, 메모리(GiB)는 저장 공간이다.** 메모리 괄호 %는 프레임워크의 프로세스 할당량을 전체 공유 RAM 121.688 GiB로 나눈 참고값이며 시스템 전체 메모리 사용률이 아니다. Spark에서 CPU/GPU 메모리는 공유하므로 RSS와 GPU 할당량을 더하지 않는다.

DNABERT-2의 기존 4,000bp 파인튜닝에서는 10.571 GiB(공유 RAM 대비 약8.69%)를 관측했다. Cellpose 등의 추가 결과는 아래 표를 함께 확인한다. **메모리90%(약109.5 GiB) 도달 조건은 아직 확인되지 않았다.**

## 모델을 입력 없이 적재했을 때

| 모델 | 실행 구성 파라미터 | 정밀도 | GPU allocated GiB | 프로세스 RSS GiB | 측정 범위 |
| --- | --- | --- | --- | --- | --- |
| ESM-2 650M | 651,043,254 | FP32 | 2.489795 | 2.087 | 새 프로세스, forward 없음 |
| alphafold2_ptm | 92,907,930 × 5 sets | FP32 weights / BF16 compute | 1.817159 | 3.556 | 5개 weight set 보관; 첫 모델 선택 뒤 ready 상태 |
| alphafold2_multimer_v3 | 93,223,705 × 5 sets | FP32 weights / BF16 compute | 1.800525 | 3.379 | 5개 weight set 보관; 첫 모델 선택 뒤 ready 상태 |
| scVI SCVI 기본 구조 | 1,293,526 | FP32 | 0.004826 | 1.054 | 학습 모델 로드; AnnData는 CPU에 먼저 읽음; optimizer 없음 |
| Cellpose-SAM cpsam_v2 | 304,628,930 | BF16 inference / FP32 training | 0.568277 | 0.954 | fresh process, first forward before input |
| DNABERT-2 | 117,068,544 | FP32 | 0.436118 | 1.829 | 기존 run post-load 기록 재사용 |

가중치 용량 이론값과 위 실측은 다르다. GPU allocator 외 드라이버·runtime과 CPU 데이터도 RAM을 쓰며 RSS에 포함될 수 있다. 로딩 기준과 아래 실행 peak가 다른 run이면 차이는 참고값이다.

## ESM-2

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| ESM-2 650M 추론 | 56–127 아미노산 | 16 | 16 | 93.6% / 95% | 2.713 GiB (2.23%) |
| ESM-2 650M 추론 | 140–254 아미노산 | 16 | 16 | 92.6% / 95% | 2.904 GiB (2.39%) |
| ESM-2 650M 추론 | 280–461 아미노산 | 6–10 | 16 | 93.5% / 95% | 2.911 GiB (2.39%) |
| ESM-2 650M 추론 | 516–976 아미노산 | 4–7 | 16 | 94.1% / 95% | 3.214 GiB (2.64%) |
| ESM-2 부분 파인튜닝 | 235–237 아미노산 | 토큰 예산 4096 | 128 | 88.1% / 95% | 3.623 GiB (2.98%) |

추론은 각 고유 서열 16개를 40회 반복 측정했다. FP32, 토큰 예산 4096. 최대 96%는 별도 기본 실행에서 관측됐고 위 표는 관측 구간을 늘린 보완 실행 수치다. 파인튜닝은 마지막 Transformer block·LayerNorm·회귀 head만 학습했다.

[상세 보고서](esm2/reports/ESM2_20261027_164300_125163.md)

## AlphaFold2 — ColabFold

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| AF2 PTM | 76 아미노산 (pad 86) | 1; 모델 5개 순차 | 1 | 91.8% / 94% | 2.166 GiB (1.78%) |
| AF2 PTM | 115 아미노산 (pad 125) | 1; 모델 5개 순차 | 1 | 92.9% / 95% | 2.273 GiB (1.87%) |
| AF2 PTM | 115 아미노산 (pad 125) | 1; 모델 5개 순차 | 1 | 92.8% / 94% | 2.273 GiB (1.87%) |
| AF2 PTM | 115 아미노산 (pad 125) | 1; 모델 5개 순차 | 1 | 91.5% / 95% | 2.273 GiB (1.87%) |
| AF2 PTM | 115 아미노산 (pad 125) | 1; 모델 5개 순차 | 1 | 91.4% / 95% | 2.273 GiB (1.87%) |
| AF2 PTM | 129 아미노산 (pad 139) | 1; 모델 5개 순차 | 1 | 93.3% / 95% | 2.307 GiB (1.90%) |
| AF2 PTM | 247 아미노산 (pad 247) | 1; 모델 5개 순차 | 1 | 93.7% / 95% | 2.706 GiB (2.22%) |
| AF2 Multimer v3 | 180 아미노산 (pad 180) | 1; 모델 5개 순차 | 1 | 92.9% / 95% | 2.170 GiB (1.78%) |

JAX 메모리는 각 프로세스 시작 이후의 **누적 allocator peak**이며 입력별로 초기화하지 않았다. 단일체 3 recycles, 복합체 20 recycles의 별도 preset이며 BF16 계산·FP32 가중치 경로다.

[상세 보고서](colabfold/reports/AlphaFold2_ColabFold.md) · [비교 CSV](colabfold/reports/RECORDS.md#file-2e1e35330a6c)

## DNABERT-2 기존 기록

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| DNABERT-2 파인튜닝 | 512 bp → 평균 103.9 tokens | 8 | 1000 | 미측정 | 2.261 GiB (1.86%) |
| DNABERT-2 파인튜닝 | 1000 bp → 평균 198.6 tokens | 8 | 1000 | 미측정 | 3.103 GiB (2.55%) |
| DNABERT-2 파인튜닝 | 2000 bp → 평균 391.8 tokens | 8 | 1000 | 미측정 | 5.173 GiB (4.25%) |
| DNABERT-2 파인튜닝 | 4000 bp → 평균 778.9 tokens | 8 | 1000 | 미측정 | 10.571 GiB (8.69%) |

위 기존 길이 실험에는 GPU-Util이 없다. 메모리 비율을 GPU 사용률로 바꿔 해석하면 안 된다. 최신 GPU 사용률 보완 실험은 지정된 후속 순서에 남아 있다.

## scVI

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| scVI 처음부터 학습 | 세포당 2000 유전자 | 128 | 1024 | 21.8% / 30% | 0.097 GiB (0.08%) |
| scVI 처음부터 학습 | 세포당 2000 유전자 | 128 | 4096 | 26.5% / 32% | 0.097 GiB (0.08%) |
| scVI 처음부터 학습 | 세포당 2000 유전자 | 128 | 10176 | 25.8% / 33% | 0.097 GiB (0.08%) |
| scVI latent 추론 | 세포당 2000 유전자 | 128 | 1024 | 24.1% / 29% | 0.039 GiB (0.03%) |
| scVI latent 추론 | 세포당 2000 유전자 | 128 | 4096 | 27.7% / 31% | 0.039 GiB (0.03%) |
| scVI latent 추론 | 세포당 2000 유전자 | 128 | 12721 | 29.1% / 33% | 0.039 GiB (0.03%) |

실행 상태: success. 학습은 400 epochs, FP32, batch 128을 고정한다. 학습 메모리는 해당 fit 전체(검증 포함)의 peak이고 GPU 평균은 training phase 표본이다. 완료 전 행은 후속 모델 재로드 검증이 남아 있을 수 있다.

## Cellpose-SAM

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| Cellpose-SAM inference | [520, 696] | up to8 tiles, 256x256 | 1 | 91.3% / 95% | 1.491 GiB (1.23%) |
| Cellpose-SAM inference | [520, 696] | up to8 tiles, 256x256 | 4 | 88.9% / 95% | 1.491 GiB (1.23%) |
| Cellpose-SAM inference | [520, 696] | up to8 tiles, 256x256 | 16 | 89.6% / 95% | 1.491 GiB (1.23%) |
| Cellpose-SAM inference | [128, 128] | up to8 tiles, 256x256 | 4 | 78.3% / 86% | 0.705 GiB (0.58%) |
| Cellpose-SAM inference | [256, 256] | up to8 tiles, 256x256 | 4 | 89.5% / 95% | 1.046 GiB (0.86%) |
| Cellpose-SAM inference | [512, 512] | up to8 tiles, 256x256 | 4 | 79.8% / 95% | 1.491 GiB (1.23%) |
| Cellpose-SAM fine-tuning | native520x696 ->256crop | 1 crop | 16 | 93.1% / 95% | 7.985 GiB (6.56%) |

Status: complete. BF16 inference / official FP32 fine-tuning, fixed tile policy. Unique image counts exclude measurement repeats. Training utilization includes label preprocessing and validation.

[Cellpose 상세 보고서](cellpose/reports/Cellpose.md) · [입력·정답 예시](https://raw.githubusercontent.com/fff0996/Bioinformatics_AImodel/49661373a7b790bbbbcce0d7ef8bf96b8ce83dbd/cellpose/reports/input_example.png) · [비교 CSV](cellpose/reports/RECORDS.md#file-d8da379c30cc)

## DNABERT-2 GPU 사용률 보완

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| DNABERT-2 inference | 2000bp / mean 387.0 tokens | 8 | 32 | 94.1% / 95% | 0.745 GiB (0.61%) |
| DNABERT-2 inference | 2000bp / mean 391.0 tokens | 8 | 128 | 94.7% / 95% | 0.756 GiB (0.62%) |
| DNABERT-2 inference | 2000bp / mean 391.8 tokens | 8 | 1000 | 94.4% / 95% | 0.765 GiB (0.63%) |
| DNABERT-2 inference | 512bp / mean 103.4 tokens | 8 | 128 | 89.2% / 94% | 0.532 GiB (0.44%) |
| DNABERT-2 inference | 1000bp / mean 198.2 tokens | 8 | 128 | 92.9% / 95% | 0.589 GiB (0.48%) |
| DNABERT-2 inference | 4000bp / mean 777.9 tokens | 8 | 128 | 94.6% / 95% | 1.420 GiB (1.17%) |
| DNABERT-2 training_resource_probe | 512bp / mean 103.4 tokens | 8 | 128 | 69.0% / 91% | 2.261 GiB (1.86%) |
| DNABERT-2 training_resource_probe | 1000bp / mean 198.2 tokens | 8 | 128 | 86.0% / 94% | 3.095 GiB (2.54%) |
| DNABERT-2 training_resource_probe | 2000bp / mean 391.0 tokens | 8 | 128 | 89.5% / 95% | 5.118 GiB (4.21%) |
| DNABERT-2 training_resource_probe | 4000bp / mean 777.9 tokens | 8 | 128 | 92.9% / 95% | 10.376 GiB (8.53%) |

기존 학습 모델은 보존했다. 새 학습 행은128개16step 자원 probe이며, 원본1000개 학습 결과에 사용률을 소급해 붙인 값이 아니다.

[DNABERT-2 보완 보고서](dnabert2/reports/DNABERT2.md) · [비교 CSV](dnabert2/reports/RECORDS.md#file-acbefee1f605)

## scGPT 기존 완료 결과

기본 추론·파인튜닝·저장·재로드·test·길이 비교 완료. test4,218세포 중 알려진 라벨4,213세포 기준 Accuracy 0.938, 학습13클래스 macroF1 0.535. GPU 개수/사용률 보완은 별도 후속 순서다.

[저장 모델 test summary](scgpt/results/finetuned_test_20261023_034345_99523/RECORDS.md#file-9cd02c9a4f70) · [길이 비교 CSV](scgpt/results/length_compare_20261023_040313_501423/RECORDS.md#file-d50c252c8230)

## Nucleotide Transformer V2

상태: complete

모델 적재 후/첫 forward 전 CUDA allocated 1.856512GiB; reserved 1.873GiB; RSS 1.125GiB.

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| Nucleotide Transformer V2 training | see report | see report | 512 | 92.3% / 95% | 9.250 GiB (7.60%) |
| Nucleotide Transformer V2 training | see report | see report | 512 | 92.5% / 95% | 9.252 GiB (7.60%) |
| Nucleotide Transformer V2 inference | 3000 | see report | 8 | 90.1% / 95% | 2.275 GiB (1.87%) |
| Nucleotide Transformer V2 inference | 3000 | see report | 32 | 95.0% / 95% | 2.275 GiB (1.87%) |
| Nucleotide Transformer V2 inference | 3000 | see report | 128 | 95.0% / 95% | 2.275 GiB (1.87%) |
| Nucleotide Transformer V2 inference | 600 | see report | 32 | 94.7% / 95% | 1.958 GiB (1.61%) |
| Nucleotide Transformer V2 inference | 6000 | see report | 32 | 94.7% / 95% | 3.140 GiB (2.58%) |
| Nucleotide Transformer V2 inference | 12276 | see report | 32 | 93.2% / 95% | 6.475 GiB (5.32%) |

[상세 보고서](ntv2/reports/NTv2.md)

## Clair3

상태: complete

모델 적재 후/첫 forward 전 CUDA allocated 0.011145GiB; reserved 0.025GiB; RSS 0.619GiB.

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| Clair3 variant_calling | 50000 | see report | 1 | 7.3% / 18% | 0.148 GiB (0.12%) |
| Clair3 variant_calling | 100000 | see report | 1 | 7.2% / 20% | 0.347 GiB (0.29%) |
| Clair3 variant_calling | 200000 | see report | 1 | 11.3% / 88% | 0.684 GiB (0.56%) |
| Clair3 variant_calling | 200000 | see report | 1 | 12.1% / 58% | 0.763 GiB (0.63%) |

[상세 보고서](clair3/reports/Clair3.md)

## RNA-FM

상태: complete

모델 적재 후/첫 forward 전 CUDA allocated 0.370748GiB; reserved 0.42GiB; RSS 1.221GiB.

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| RNA-FM inference | 512 | see report | 8 | 92.0% / 95% | 0.766 GiB (0.63%) |
| RNA-FM inference | 512 | see report | 32 | 94.7% / 95% | 1.852 GiB (1.52%) |
| RNA-FM inference | 512 | see report | 64 | 94.8% / 95% | 2.665 GiB (2.19%) |
| RNA-FM inference | 128 | see report | 7 | 94.4% / 95% | 0.479 GiB (0.39%) |
| RNA-FM inference | 256 | see report | 7 | 94.8% / 95% | 0.561 GiB (0.46%) |
| RNA-FM inference | 512 | see report | 7 | 94.5% / 95% | 0.721 GiB (0.59%) |
| RNA-FM inference | 960 | see report | 7 | 94.8% / 95% | 1.456 GiB (1.20%) |
| RNA-FM native_embedding | see report | see report | 896 | 94.6% / 95% | 8.870 GiB (7.29%) |
| RNA-FM native_embedding | see report | see report | 224 | 87.6% / 95% | 8.147 GiB (6.69%) |
| RNA-FM native_embedding | see report | see report | 280 | 91.6% / 95% | 7.828 GiB (6.43%) |
| RNA-FM linear_head_training_with_validation | see report | see report | 896 | 19.2% / 95% | 0.434 GiB (0.36%) |

[상세 보고서](rnafm/reports/RNAFM.md)

## OpenSpliceAI

상태: complete

모델 적재 후/첫 forward 전 CUDA allocated 0.013392GiB; reserved 0.018GiB; RSS 0.764GiB.

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| OpenSpliceAI ensemble_inference | 15000 | see report | 12 | 94.1% / 95% | 0.125 GiB (0.10%) |
| OpenSpliceAI ensemble_inference | 15000 | see report | 24 | 94.0% / 95% | 0.125 GiB (0.10%) |
| OpenSpliceAI ensemble_inference | 15000 | see report | 36 | 94.7% / 95% | 0.125 GiB (0.10%) |
| OpenSpliceAI ensemble_inference | 11000 | see report | 12 | 94.6% / 95% | 0.094 GiB (0.08%) |
| OpenSpliceAI ensemble_inference | 20000 | see report | 12 | 94.4% / 95% | 0.166 GiB (0.14%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 75.6% / 95% | 2.160 GiB (1.78%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 86.0% / 95% | 2.160 GiB (1.78%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 77.3% / 95% | 2.160 GiB (1.78%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 82.4% / 95% | 2.160 GiB (1.78%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 81.3% / 95% | 2.160 GiB (1.78%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 83.6% / 95% | 2.160 GiB (1.78%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 88.7% / 95% | 2.160 GiB (1.78%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 86.4% / 95% | 2.160 GiB (1.78%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 92.4% / 95% | 2.160 GiB (1.78%) |
| OpenSpliceAI single_model_training | 15000 | see report | 96 | 94.0% / 95% | 2.160 GiB (1.78%) |

[상세 보고서](openspliceai/reports/OpenSpliceAI.md)

## scGPT GPU 사용률 보완

| 모델·작업 | 입력당 크기 | 실제 배치 | 고유 입력 | GPU 평균 / 최대 | GPU 할당 peak (공유 RAM 대비 참고 %) |
| --- | --- | --- | --- | --- | --- |
| scGPT embedding | cap1200 / mean 871.4 tokens | 8 | 128 | 91.5% / 95% | 0.367 GiB (0.30%) |
| scGPT embedding | cap1200 / mean 708.1 tokens | 8 | 1024 | 94.8% / 95% | 0.374 GiB (0.31%) |
| scGPT embedding | cap1200 / mean 598.9 tokens | 8 | 4218 | 94.2% / 95% | 0.380 GiB (0.31%) |
| scGPT embedding | cap256 / mean 256.0 tokens | 8 | 128 | 92.2% / 94% | 0.260 GiB (0.21%) |
| scGPT embedding | cap512 / mean 512.0 tokens | 8 | 128 | 93.8% / 94% | 0.295 GiB (0.24%) |
| scGPT embedding | cap1024 / mean 858.9 tokens | 8 | 128 | 94.3% / 95% | 0.357 GiB (0.29%) |
| scGPT training utilization supplement | max1200 incl CLS | 8 | 1000 | 88.7% / 95% | 3.374 GiB (2.77%) |

새 프로세스 추론모델 적재 allocated 0.193183GiB; reserved 0.199GiB; RSS 1.153GiB. 기존기본/길이/test결과와분리해보완했다.

[scGPT 상세 보고서](scgpt/reports/scGPT.md) · [보완 비교 CSV](scgpt/reports/RECORDS.md#file-2388700259bf)

## 후속 메모리 확대 실험

사용자가 batch를단계적으로늘려메모리80~90% 도달가능성과처리량을확인하는후속실험을추가했다. **후속 범위는 scGPT만이며, 실제 세포 실험은10만세포 완료 지점에서 사용자 승인으로 일시중단했다.** 위표는기본고정batch 실험/보완결과이며최대지원batch/최대점유한계가아니다. [후속메모리실험](MEMORY_RESULTS.md)

[scVI 상세 보고서](scvi/reports/scVI.md) · [실제 학습 입력 예](scvi/reports/RECORDS.md#file-e9267afe1b49)

[전체 진행 상태](STATUS.md)

실제 대규모 세포 처리: [scGPT Tabula Sapiens 30만 고유 세포](scgpt/reports/scGPT_actual_300k.md) — 2026-10-28 재개 후 batch8/256 실제30만세포 처리·최종출력검증 완료. 최신 보고서 참조. [pancreas 전체14,818세포 학습·검증·test](scgpt/reports/scGPT_realdata_full.md): batch8 완료, 큰 학습 batch284/256/224는 가용RAM guard 중단.

2026-10-28 최종 완료: scGPT 실제30만 고유세포 batch8/256 추론 및 전체출력검증 완료. 실행 `actual_300k_resume_20261028_093701_696230`, [최종 보고서](scgpt/reports/scGPT_actual_300k.md). 후속 실험 범위는 scGPT만이며 다른 모델은 진행하지 않는다. 최신 상태는 이 항목을 우선한다.

2026-10-28 사용자 추가승인: ESM-2 긴 실제서열900~950AA 배치확대 **추론** 실험 종료. run esm2/results/batch_inference_20261028_112900_601721. 결과는 [배치 메모리 보고서](esm2/reports/ESM2_batch_memory.md)에정리한다. scGPT결과보존.

ESM-2 후속 최종 상태 complete, 최대성공batch 384,80%도달 False,90%도달 False. [상세](esm2/reports/ESM2_batch_memory.md). 새로운 다른 모델 실험은 자동 시작하지 않는다.

Cellpose 타일배치후속 tile_batch_20261028_134130_265230 상태 running, 최대성공타일batch 8,80%도달 None,90%도달 None. [보고서](cellpose/reports/Cellpose_batch_memory.md). 추가모델실험자동시작없음.

Cellpose 타일배치후속 tile_batch_20261028_134130_265230 상태 complete, 최대성공타일batch 384,80%도달 False,90%도달 False. [보고서](cellpose/reports/Cellpose_batch_memory.md). 추가모델실험자동시작없음.

남은 후속 모델은사용자승인으로자동순차진행중이다. [통합후속진행](FOLLOWUP_RESULTS.md)에서모델별실제상태·결과와배치확대미실시항목을확인한다.

scvi 배치후속 complete, 최대성공batch 10176. [보고서](scvi/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

dnabert2 배치후속 complete, 최대성공batch 512. [보고서](dnabert2/reports/Batch_memory_followup.md). 종료사유 predicted_reservation_exceeds_available_budget.

ntv2 배치후속 complete, 최대성공batch 128. [보고서](ntv2/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

rnafm 배치후속 complete, 최대성공batch 76. [보고서](rnafm/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

openspliceai 배치후속 complete, 최대성공batch 156. [보고서](openspliceai/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

clair3 전용추가후속 complete; [결과](clair3/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

af2 전용추가후속 stopped; [결과](colabfold/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

clair3 전용추가후속 complete; [결과](clair3/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

af2 전용추가후속 stopped; [결과](colabfold/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

clair3 전용추가후속 complete; [결과](clair3/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

af2 전용추가후속 complete; [결과](colabfold/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

10종 후속최종정리완료. 이전AlphaFold2/Clair3미실시상태는새실행으로갱신됐다. 구현·정밀도·검증범위차이를포함한최신전체표: [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md). 추가GPU작업없음.

뇌이미지 추가 Cellpose NucMM-M brain_nucmm_20261029_110603_965442 상태 complete. [보고서](cellpose/reports/Cellpose_brain_NucMM.md). 일반뇌MRI가아닌세포핵micro-CT·정답instance검증,기본/배치실험과별도run.
