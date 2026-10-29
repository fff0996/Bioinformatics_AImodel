# DGX Spark 바이오 AI 10종 벤치마크

DGX Spark에서 실제 공개 입력으로 모델 추론·지원되는 학습·결과 검증과 자원 계측을 수행한 기록이다. GPU 실험은 한 번에 하나씩 실행했다.

## 결과 바로 보기

- **[10종 후속 배치 실험 최종 결과](FOLLOWUP_RESULTS.md)**
- [기본 실험 및 입력별 GPU·메모리 결과](RESULTS.md)
- [메모리 실험 해석과 추가 결과](MEMORY_RESULTS.md)
- [Cellpose 뇌 조직 2D·3D 추가 실험](cellpose/reports/Cellpose_brain_NucMM.md)
- [실행 이력과 완료 범위](STATUS.md)
- [재현 방법 및 Git에 포함한 범위](docs/REPRODUCIBILITY.md)

## 핵심 해석

GPU 연산 사용률(%)과 메모리 할당량(GiB)은 다른 지표다. Spark의 약121.7GiB는 CPU/GPU 공유 RAM이며 별도 전용 VRAM이 아니다. RSS와 CUDA 메모리를 독립적인 물리 메모리처럼 더하지 않는다.

배치를 고정하면 전체 입력 개수 증가는 주로 총 처리시간을 늘린다. 순간 메모리는 동시에 처리하는 배치, 입력당 길이·크기, 정밀도, 모델 및 추론/학습 상태에 영향을 받는다. 이번에 시험한 최대 성공 배치는 모델의 절대 한계가 아니다.

scGPT 학습 용량 probe는 batch292에서 CUDA allocated peak100.75GiB(공유RAM대비82.79%)를 관측했다. 실제30만세포 추론 batch256의5.477GiB와 다른 실험이며, 90% 도달은 확인하지 못했다. AlphaFold2 후속은 별도 FP32 JAX vmap 구현이고 JAX peak는 컴파일/warmup 포함 누적값이다. 모델별 조건·계측 범위가 달라 수치를 단순 모델 순위로 해석하지 않는다.

## 모델별 보고서

| 모델 | 기본 결과 | 후속 결과 |
|---|---|---|
| ESM-2 | [보고서](esm2/reports/ESM2_20261027_164300_125163.md) | [배치](esm2/reports/ESM2_batch_memory.md) |
| AlphaFold2 — ColabFold | [보고서](colabfold/reports/AlphaFold2_ColabFold.md) | [실험적 배치](colabfold/reports/Batch_memory_followup.md) |
| scVI | [보고서](scvi/reports/scVI.md) | [배치](scvi/reports/Batch_memory_followup.md) |
| Cellpose | [보고서](cellpose/reports/Cellpose.md) | [타일 배치](cellpose/reports/Cellpose_batch_memory.md), [뇌 조직](cellpose/reports/Cellpose_brain_NucMM.md) |
| DNABERT-2 | [보고서](dnabert2/reports/DNABERT2.md) | [배치](dnabert2/reports/Batch_memory_followup.md) |
| Nucleotide Transformer V2 | [보고서](ntv2/reports/NTv2.md) | [배치](ntv2/reports/Batch_memory_followup.md) |
| Clair3 | [보고서](clair3/reports/Clair3.md) | [실제 후보 배치](clair3/reports/Batch_memory_followup.md) |
| RNA-FM | [보고서](rnafm/reports/RNAFM.md) | [배치](rnafm/reports/Batch_memory_followup.md) |
| OpenSpliceAI | [보고서](openspliceai/reports/OpenSpliceAI.md) | [배치](openspliceai/reports/Batch_memory_followup.md) |
| scGPT | [보고서](scgpt/reports/scGPT.md) | [학습 용량](scgpt/reports/scGPT_memory_batch.md), [실제30만세포](scgpt/reports/scGPT_actual_300k.md) |

원본 데이터·모델 가중치·대용량 배열·가상환경은 업로드하지 않았다. 출처·리비전·해시 및 공개된 예시/지표를 통해 실행 조건을 확인할 수 있다. 최상위의 기존 ESM2 보고서는 이전 업로드 파일로 보존했으며 현재 모델별 자료는 위 경로를 이용한다.
