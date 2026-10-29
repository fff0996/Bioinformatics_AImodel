# Nucleotide Transformer V2 — DGX Spark

실행 `ntv2_20261027_201332_251243` · 상태 **complete** · [원본 summary](../results/ntv2_20261027_201332_251243/summary.json) · 실행 로그 (로컬 보관: `../results/ntv2_20261027_201332_251243/benchmark.log`) · [GPU·메모리 시계열](../results/ntv2_20261027_201332_251243/resources.csv)

## 모델과 기준 조건

공식 `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`, 리비전 `06615c1660c892fc199840c18123f8385b3542a8`. V2 500M을 대표로 선택했으며 V1·V3로 바꾸지 않았다. 다운로드된 model.safetensors의 SHA-256은 공식 LFS oid와 일치함을 검증했고 전체 출처는 `results/sources.json`에 보관했다.

실제 파라미터: {"mlm_total": 498345436, "embedding_backbone": 493084113, "mlm_head": 5261323, "classifier_total": 494135763, "classifier_trainable": 494135763}

MLM 전체 모델을 로드하되 임베딩 측정은 `model.esm` backbone의 최종 hidden state만 계산해 masked mean(특수토큰 포함)으로B×1024를 만든다. MLM head와 모든 층 hidden state를 계산/저장하는 경로와 메모리·시간이 다르다. 분류는 공식 custom EsmForSequenceClassification의 CLS MLP head를 사용하고 전 학습 가능 backbone·head를 함께 갱신했다.

환경: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "4.44.2", "cuda": "13.3", "gpu": "NVIDIA GB10"}

batch8,FP32,TF32=False,seed42,8CPU threads. 공식 fine-tuning notebook의 batch8/lr1e-5/2epoch를 참고하되 NT-v2와 enhancer binary 소규모 subset에 적용했다. notebook의1000maxsteps나 전체 데이터 재현 실험이 아니다. 추론에서도 batch8을 고정하며 배치 증가 sweep을 수행하지 않았다.

| 상태 | allocated GiB | reserved GiB | RSS GiB | system available GiB |
| --- | --- | --- | --- | --- |
| MLM 적재 직후/첫 입력 전 | 1.857 | 1.873 | 1.125 | 111.229 |

이론 FP32 가중치 용량은 MLM 파라미터수×4byte/2³⁰이며 실측과 구분한다. 공유 RAM121.688GiB에서 RSS와CUDA를 더하지 않는다. 적재 11.959초; warmup 0.596초. CPU 입력 JSON 로딩은 적재 전에 수행했다.

## 실제 입력·라벨·분할

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

입력은6-mer 중심 토큰화 후 int64 input_ids/attention_mask의B×실제token수, 라벨은B 정수다. [검사 결과](../data/input_audit.json)와 선택 서열·split·해시 `data/inputs.json`을 보관했다.

길이/개수 실험은 별도의 GRCh38 참조 유전체 동일 구간128개를 사용한다. BEND에서 이미 확인한 좌표 중심으로12,282nt를 추출하고 ACGT 외 문자가 없는 실제 구간을 택했다. 같은 구간의 중심 crop600/3000/6000/12276nt를 사용하며 무라벨 추론 전용이다. 음의 가닥은 reverse-complement했다. 짧은 enhancer 학습 데이터의 정확도와 이 긴 무라벨 서열을 혼동하지 않는다.

## 입력별 GPU 사용률·메모리

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

## 파인튜닝·저장·독립 test

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

## 해석과 공식 출처

길이가 커질 때 attention 중간값 등 추가 메모리가 커질 수 있지만, 데이터 개수가 늘어도 batch정책이 같으면 순간 peak가비례 증가하지 않는다. 공유RAM90% 채우기는실험목표가아니다.

- [공식 NT-v2 모델 카드](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)
- [공식 소스·학습 예제 링크](https://github.com/instadeepai/nucleotide-transformer)
- [공식 enhancer 데이터](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)
- [공식 분류 fine-tuning notebook](https://github.com/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb)
