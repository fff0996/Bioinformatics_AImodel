# DNABERT-2 기존 결과와 GPU 사용률 보완

보완 run: `supplement_20261027_195301_706540`; 상태: complete. 기존 원본은 `/home/hyein/dnabert2-benchmark/dnabert2_eval`에 보존했다.

## 모델·실행 조건

공식 DNABERT-2-117M, revision `7bce263b15377fc15361f52cfab88f8b586abda0`, 실제 encoder 파라미터 117,068,544. FP32, batch8, TF32=False, seed42,8 CPU threads. 토큰화는 padding=True/truncation=False, masked mean pooling(특수 토큰 포함)이다. 18개 출력의 선형 head는 13,842개 파라미터를 추가한다. Bert pooler가 초기화되었다는 공식 loader 경고가 있으나 이 과제는 hidden-state masked mean을 사용해 pooler 출력은 사용하지 않는다. MLM head도 임베딩 모델에는 적재하지 않는다.

환경: {"torch": "2.14.1+cu130", "transformers": "4.29.2", "cuda": "13.0", "gpu": "NVIDIA GB10"}

기존 conda 환경을 원래 /workspace 경로로 읽기 전용 mount하여 사용했다. Triton/FlashAttention이 없어 기존과 같은 PyTorch attention fallback으로 실행됐다. 다른 모델/환경과의 속도 차이를 입력 길이의 효과로만 해석하면 안 된다.

입력 forward 전 모델만 GPU에 올린 allocated **0.436118GiB**, reserved 0.479GiB, RSS 1.100GiB, 시스템 available 112.949GiB. 적재 0.888초. 이론 FP32 파라미터 용량 0.436114GiB와 실측은 구분한다.

## 실제 입력과 정답

기존 확보한 BEND histone_modification train의 동일1,000개 genomic region을 재사용했다. 참조 GRCh38, 음의 가닥 reverse-complement, A/C/G/T 및 길이·고유 ID·길이별 동일 ID 순서를 검사했다. 원본 train/validation/test를 바꾸지 않았다. 원본 region의 18개 히스톤 표지 multi-label을 0/1 타깃으로 사용한다. bp 수는 토큰 수가 아니며 BPE 결과는 아래에 별도 표기한다. 길이별 crop의 라벨은 원본 region에서 상속하므로 잘라낸 범위의 염기별 정답이라고 주장하지 않는다.

실제2,000bp 입력의 앞120bp:

```text
CTTTTGAATATGAATGTCCTAATTTAATTCTTTGGACCTTTGAGGGGAGGACACTATCACTTCTACATATGTAGAGAAGTAAAAGTCTCATAGATCCATCTTGTCTTTAAAAATATATGG
```

ID: `chr3:152302071-152302583:+`; 양성 라벨 인덱스: `[4, 16]`. 전체 서열·라벨 예시는 input_examples.json (로컬 보관: `../results/supplement_20261027_195301_706540/input_examples.json`). 학습 입력은 input_ids/attention_mask의 B×실제토큰길이, 정답은 B×18 float32 multi-hot이다.

## 입력별 GPU 사용률·메모리 동시 측정

| 모드 | bp | 고유 개수 | token 최소/평균/최대 | 반복 | 초 | 건/초 | GPU 평균/최대% | 시작 GiB | peak GiB | 증가 GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inference | 2000 | 32 | 352/387.0/419 | 29 | 15.25 | 60.83 | 94.1/95 | 0.479 | 0.745 | 0.266 |
| inference | 2000 | 128 | 352/391.0/429 | 8 | 17.08 | 59.94 | 94.7/95 | 0.479 | 0.756 | 0.277 |
| inference | 2000 | 1000 | 349/391.8/438 | 1 | 16.71 | 59.85 | 94.4/95 | 0.479 | 0.765 | 0.286 |
| inference | 512 | 128 | 87/103.4/118 | 36 | 15.34 | 300.45 | 89.2/94 | 0.479 | 0.532 | 0.053 |
| inference | 1000 | 128 | 173/198.2/222 | 17 | 15.19 | 143.28 | 92.9/95 | 0.479 | 0.589 | 0.110 |
| inference | 4000 | 128 | 709/777.9/846 | 3 | 16.94 | 22.66 | 94.6/95 | 0.479 | 1.420 | 0.941 |
| training_resource_probe | 512 | 128 | 87/103.4/118 | 1 | 2.35 | 54.48 | 69.0/91 | 0.467 | 2.261 | 1.793 |
| training_resource_probe | 1000 | 128 | 173/198.2/222 | 1 | 3.62 | 35.32 | 86.0/94 | 0.499 | 3.095 | 2.596 |
| training_resource_probe | 2000 | 128 | 352/391.0/429 | 1 | 7.31 | 17.50 | 89.5/95 | 0.499 | 5.118 | 4.620 |
| training_resource_probe | 4000 | 128 | 709/777.9/846 | 1 | 16.30 | 7.85 | 92.9/95 | 0.499 | 10.376 | 9.878 |

추론은 최소15초 동안 전체 subset을 반복했다. 개수는2,000bp에서32/128/1000 중첩 부분집합, 길이는 동일128개에서512/1000/2000/4000bp를 비교했다. 토큰 길이 분포는 subset마다 조금 다르므로 개수만의 완전한 통제 실험은 아니다. batch8을 고정하고 데이터를 통째로 GPU에 올리지 않았다.

학습 행은 기존 완료된1,000개·1epoch 학습에 빠져 있던 사용률을 보완하는 **별도128개·16step 자원 probe**다. 각 길이는 동일 초기 체크포인트·seed·AdamW(lr2e-5,weight_decay.01)로 시작한다. 1,000개 학습이나 새로운 test 평가를 완료했다는 뜻은 아니다. probe 가중치는 최종 모델로 채택하지 않고 기존 저장·재로드 검증된 학습 모델을 보존했다. 짧은 학습 구간은 반복하지 않았으므로 표본 수와 초기 optimizer 할당 영향을 고려한다.

512bp 학습2.35초/4표본,1000bp 학습3.62초/7표본은 짧은 관찰로 평균 GPU 사용률의 대표성이 낮다. nvidia-smi 자체 관측창 때문에 앞 구간 활동이 섞일 수 있다. 2000bp7.31초/13표본,4000bp16.30초/30표본으로 구간 길이도 다르다. 이 값들을 일반적인 학습 사용률의 확정 비교로 쓰지 않는다.

추론 warmup 뒤 baseline이 모델 적재량보다 큰 것은 실제 실행 후 유지되는 버퍼/상태를 포함한 관찰값이다. 정확한 내부 원인은 계측하지 않았다. 입력당 순수 버퍼 증가량이라고 표현하지 않는다. 체크포인트와 코드 SHA-256은 verification.json에 저장했다.

시계열은 [resources.csv](../results/supplement_20261027_195301_706540/resources.csv)에 저장했다. nvidia-smi GPU 활동시간 비율을0.5초 대기+조회 간격으로 표본화하며 최대 FLOPS/메모리%가 아니다. 장치 전체에 디스플레이 영향이 포함될 수 있다. 같은 run의 현재 allocated에서 peak 증가량을 계산했고 peak는 조건마다 reset했다. 학습 증가는 gradient·optimizer·중간값을 포함한다. RSS/available/swap은 시계열 표본이다. Spark 공유 RAM121.688GiB에서 CUDA와RSS를 더하면 안 된다.

시간은 CPU메모리에 준비한 데이터의 토큰화·전송·forward·pooling, 학습은 backward·optimizer까지 포함하며 CUDA 경계는 동기화했다. 디스크 읽기/저장 시간은 제외하고 별도 미측정(N/A), compile은 분리되지 않았다. 추론3배치 warmup은 별도 수행했다. 학습은 초기 optimizer 생성을 포함한 실제 첫 step부터 측정한다.

## 기존 완료 결과의 위치

기존2,000bp 학습1,000개·batch8 FP32는 평균391.834 tokens(349–438), peak5.172566GiB였다. 4,000bp 학습 peak10.5712GiB(공유 RAM 약8.69%)였다. 이 과거 run에는 GPU-Util이 없어서 이번 별도 probe 사용률을 과거 값처럼 붙이지 않는다.

원본 `bend_length_results/*training*/saved_model` 및 `finetune_results/*/saved_model`에 기존 학습 모델과 save/reload 결과가 있다. 이번 보완은 새로운 정확도 향상이나 독립 test 성능을 주장하지 않는다. 전체 기존 결과는 삭제하거나 덮어쓰지 않았다.

## scGPT도 이미 완료

scGPT 기본 추론·파인튜닝·저장·재로드·test·길이 비교는 완료돼 있다. 새로 필요한 것은 입력 개수와 GPU 사용률의 보완이다.

[scGPT 저장 모델 test summary](../../scgpt/results/finetuned_test_20261023_034345_99523/summary.json) · [기존 길이 비교](../../scgpt/results/length_compare_20261023_040313_501423/comparison.csv)

test4,218세포 중 알려진 라벨4,213세포 기준 Accuracy0.9380489, 학습13클래스 macroF1 0.5349519, weightedF1 0.9311712. test-only 라벨5세포는 해당 지표에서 제외했다.

## 공식 출처

- [DNABERT-2 공식 모델 카드](https://huggingface.co/zhihan1996/DNABERT-2-117M)
- [공식 구현](https://github.com/MAGICS-LAB/DNABERT_2)
- [BEND 공식 저장소](https://github.com/nyuolab/BEND)
