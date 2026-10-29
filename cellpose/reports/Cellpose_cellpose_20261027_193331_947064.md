# Cellpose-SAM: DGX Spark 실행 보고서

> 최신 통합판: [기본·모든 후속 실험을 합친 최종 보고서](FINAL.md). 입력·추론·학습별 메모리와 80% 목표 상태는 최종 보고서 첫 표를 확인한다.

실행: `cellpose_20261027_193331_947064`. 상태: **complete**. 원본 측정: [summary.json](../results/cellpose_20261027_193331_947064/RECORDS.md#file-2e96f1dc76e2), 로그 (로컬 보관: `../results/cellpose_20261027_193331_947064/run.log`), [자원 시계열](../results/cellpose_20261027_193331_947064/RECORDS.md#file-7e3493105ec7), [실제 forward 입력](../results/cellpose_20261027_193331_947064/RECORDS.md#file-51657b33486d).

## 모델과 환경

Cellpose 4.2.1, 공식 기본 체크포인트 **cpsam_v2**, SAM ViT-L 기반 Cellpose-SAM이다. Cellpose3 cyto·CPDINO와 별개다. 전체 파라미터 304,628,930, requires_grad 304,592,064. 이론적 BF16 파라미터 저장량 0.567416 GiB.

체크포인트 revision: `7c61431b5fbb078f3296754bd15d9f51b320f837`; SHA-256: `0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667`. 공식 파일 메타데이터 `../models/repository_files.json`의 LFS oid와 대조해 일치했다. 데이터/모델 출처 목록은 `../data/sources.json`에 보존했다.

환경: {"python": "3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]", "torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "cellpose": "4.2.1", "gpu": "NVIDIA GB10", "platform": "Linux-7.0.0-1019-nvidia-aarch64-with-glibc2.39", "tf32": true}

8 CPU threads, seed42. 공식 trainer는 epoch별 NumPy seed를 epoch 번호로, 검증 때42로 설정한다. GPU 작업은 한 번에 하나씩 실행했다. PyTorch NGC 26.06 ARM64 전용 venv이며 기존 환경은 보존했다. BF16 추론, 공식 학습 API가 자동 변환하는 FP32 파인튜닝이다. TF32 허용은 Cellpose 공식 코드 기본값이다.

| 적재 상태 | CUDA allocated GiB | reserved GiB | RSS GiB | system available GiB |
| --- | --- | --- | --- | --- |
| 모델 적재 직후, 첫 입력 이전 | 0.568277 | 1.740 | 0.954 | 111.969 |

적재 시간 1.129초; 적재 중 allocator peak 1.718GiB. 이는 현재 적재량과 다르다. 첫 warmup 1.377초는 초기 kernel/Numba 준비와 후처리를 포함하며 정상 반복 성능 표에서 제외했다.

## 실제 데이터와 학습 입력

Broad BBBC039v1 공개 CC0 데이터: U2OS 세포의 Hoechst DNA 채널, 200장, uint16 흑백 TIFF 520×696. 공식 원본 이미지 분할 train100/validation50/test50에서 seed42로 16/8/16장을 선택했다. 원본 이미지 단계에서 분리한 뒤 학습 crop을 생성한다. 픽셀 크기(µm)는 확보한 메타데이터에서 확인되지 않아 N/A로 남긴다.

정답은 사람이 표시한 핵 instance PNG의 첫 채널을 공식 예제처럼 connected-component label로 변환했다. 0은 배경, 양의 정수는 개별 핵 ID이며 연속값 회귀 라벨이 아니다. 원본 영상·마스크 shape 일치, 유한값, 이미지 식별자 중복 및 split 분리를 확인했다. 전체200장에는 핵이 적은 영상도 있으나 선택한 학습16장은 모두 최소5개 기준을 충족한다.

![실제 학습 영상과 정답](https://raw.githubusercontent.com/fff0996/Bioinformatics_AImodel/49661373a7b790bbbbcce0d7ef8bf96b8ce83dbd/cellpose/reports/input_example.png)

| split | 선택 장수 | 첫 원본 ID | shape | dtype | 핵 수 | 픽셀 범위 |
| --- | --- | --- | --- | --- | --- | --- |
| training | 16 | IXMtest_B12_s9_w1ABE4C6B8-6110-4D7D-B782-94C43FDC2E1F | [520, 696] | uint16 | 109 | 119…3398 |
| validation | 8 | IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78 | [520, 696] | uint16 | 120 | 125…3271 |
| test | 16 | IXMtest_A12_s7_w1EAEEA614-51ED-43B3-A4FF-088730911E4C | [520, 696] | uint16 | 69 | 112…3885 |

원본 2D grayscale은 실제 추론 forward에서 1채널로 유지된다(아래 실측 NCHW). 학습 API에서는 첫 채널 + 두 zero 채널로 변환된다. 1/99 percentile 정규화 후 float32 전처리, 추론 forward에서 BF16으로 변환한다. 학습은 영상과 flow label을 함께 random rotate/resize하여 실제 1×3×256×256 FP32 입력을 사용한다. 각 epoch마다 각 학습 이미지에서 crop을 하나 뽑는 정책이다.

실제 전체 선택 목록·파일 해시·마스크 경로: [input_audit.json](../data/RECORDS.md#file-4c66cd9b8b0d). BBBC039는 DSB2018과 일부 겹치며 사전학습 Cellpose 데이터와의 중복 여부를 확정하지 못했다. 본 test는 이번 파인튜닝에서만 분리한 평가이며, 사전학습까지 완전히 독립적인 외부 성능이라고 주장하지 않는다.

## 입력 규모에 따른 GPU 사용률과 메모리

공식 eval 기본 타일 배치 상한8, tile256, overlap0.1, augment=False, diameter/rescale 없음, flow_threshold0.4, cellprob_threshold0, min_size15, max_size_fraction0.4, 동일 사전학습 체크포인트를 고정했다. 영상별로 순차 처리하며, 실제 batch는 타일 수에 따라8보다 작을 수 있다. 개수 비교는 같은 크기의 중첩1/4/16장, 크기 비교는 같은4장의 중심128/256/512 crop이며 native520×696 비교는 count4 행이다. crop은 확대보간하지 않은 부분 시야다.

| 조건 | 원본 H×W | 고유 장수 | 반복 | 초 | 장/초 | GPU 평균/최대 % | 시작 allocated GiB | peak GiB | 같은 run 증가 GiB | 실제 forward NCHW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count_1 | [520, 696] | 1 | 23 | 15.296 | 1.504 | 91.3 / 95 | 0.600 | 1.491 | 0.892 | ['[4, 1, 256, 256]', '[8, 1, 256, 256]'] |
| count_4 | [520, 696] | 4 | 6 | 16.512 | 1.453 | 88.9 / 95 | 0.600 | 1.491 | 0.892 | ['[4, 1, 256, 256]', '[8, 1, 256, 256]'] |
| count_16 | [520, 696] | 16 | 2 | 22.150 | 1.445 | 89.6 / 95 | 0.600 | 1.491 | 0.892 | ['[4, 1, 256, 256]', '[8, 1, 256, 256]'] |
| crop_128 | [128, 128] | 4 | 55 | 15.266 | 14.411 | 78.3 / 86 | 0.600 | 0.705 | 0.106 | ['[1, 1, 256, 256]'] |
| crop_256 | [256, 256] | 4 | 17 | 15.284 | 4.449 | 89.5 / 95 | 0.600 | 1.046 | 0.446 | ['[4, 1, 256, 256]'] |
| crop_512 | [512, 512] | 4 | 8 | 16.350 | 1.957 | 79.8 / 95 | 0.600 | 1.491 | 0.892 | ['[1, 1, 256, 256]', '[8, 1, 256, 256]'] |

실제 타일 수는 이미지당128crop=1,256crop=4,512crop=9,520×696원본=12다. 입력 padding과 overlap 때문에256crop도4타일이 된다. 공식 소스의 padding 후 타일 정책을 그대로 사용했다.

각 조건 최소15초 동안 완전한 eval 반복을 실행했다. 고유 장수와 반복 처리량을 구분한다. 측정 시간은 메모리에 읽어둔 입력의 정규화·타일링·전송·모델 forward·flow dynamics·마스크 후처리를 포함하며 원본 디스크 읽기와 결과 파일 저장은 제외한다. forward 별도 시간은 CPU↔GPU 전송과 반환 변환을 포함한다. CUDA 경계는 synchronize했다. 반복 구간의 CUDA peak는 조건마다 reset하고 empty_cache 뒤의 실제 baseline을 함께 기록한다.

원본 디스크 읽기·개별 결과 파일 저장 시간은 별도 계측하지 않아 N/A다. warmup은 초기 준비 시간을 포함하며 독립적인 compile 시간은 분리되지 않았다. 학습 총시간에는 공식 checkpoint 저장이 포함되지만 저장만의 시간은 N/A다.

GPU-Util은 nvidia-smi의 장치 활동 시간 비율이며 최대 FLOPS나 메모리 사용률이 아니다. 0.5초 대기+조회 소요의 시계열에서 조회 전체가 조건 안에 들어간 표본만 집계했다. 장치 전체에는 디스플레이 영향이 포함될 수 있다. RSS/available은 sampled 값이며 정확한 순간 peak가 아니다. 별도 CSV에 reserved peak와 sampled RSS/available, forward 시간도 저장했다.

추론 최대 GPU-Util 95%; 80% 관측 여부: 관측됨. 추론 최대 allocated 1.491GiB는 전체 공유 RAM 121.688GiB의 1.23% 참고값이다. GPU 활동 시간 90%와 공유 RAM90% 사용은 다른 뜻이다. RSS와 CUDA 수치를 더하지 않는다. 이미지가 커져도 타일 입력은256×256으로 유지되며 타일 수·마지막 batch·후처리 비용이 변한다. 입력 개수 증가가 peak를 비례 증가시킨다고 해석할 수 없다.

## 정답 마스크 파인튜닝·검증·저장

공식 train_seg 경로, 전 학습 가능 구성의 AdamW(lr1e-5, weight_decay0.1), batch1, 100epochs, epoch당16장, rescale=False, scale_range0.5,256crop을 고정했다. 공식10epoch warmup과 마지막50epoch 감쇠를 사용했다. validation8장은 API의 test_data 인자에 넣었으며 최종 test16장은 학습에 넣지 않았다. 기본 min_train_masks5로 선택 학습 영상 모두 유지됐다.

완료 1600 optimizer steps, 총 672.75초(정답 flow 준비·검증·저장 포함). flow/정규화 준비 0.80초. 학습 진입 FP32 baseline 1.166GiB, fit peak 7.985GiB, 증가 6.819GiB. 전체 fit GPU 평균/최대 93.1/95%. 이 평균에는 flow 준비와 검증·저장도 포함한다.

실제 gradient를 관찰한 파라미터: {"parameters_with_gradient": 304394432, "names_without_gradient": ["encoder.patch_embed.proj.weight", "encoder.patch_embed.proj.bias"]}

일부 patch embedding은 공식 forward가 .data를 사용하므로 requires_grad=True여도 gradient가 없다. 실제 변경 예시(최대 절댓값): {"encoder.patch_embed.proj.weight": 0.0, "encoder.blocks.0.attn.qkv.weight": 0.001953125, "out.weight": 0.00390625}

저장 경로: `models/cellpose_20261027_193331_947064/models/cpsam_v2_bbbc039_finetuned`. 공식 trainer의 저장 가중치는 FP32이고 평가 시 BF16으로 복원한다. 별도 optimizer/RNG/epoch 상태도 저장했으나 중단 후 학습 재개는 검증하지 않았다.

| 동일 test16장 | 평균 AP@0.5 | AP@0.75 | AP@0.9 |
| --- | --- | --- | --- |
| 사전학습 | 0.9054 | 0.8545 | 0.5991 |
| 파인튜닝 | 0.9138 | 0.8515 | 0.5937 |

AP는 Cellpose instance matching의 TP/(TP+FP+FN)을 이미지별 계산한 평균으로 COCO mAP와 다르다. 작은 파인튜닝 subset과100epochs의 관찰 결과이며 향상을 보장하지 않는다. epoch 선택에 test 지표를 사용하지 않았다.

독립 새 프로세스 재로드 결과: {"checkpoint_sha256": "3acb3f69377a915bcd523e0eb5889b73674b8575a77d7c1e3c4a76c6ebfec360", "load_seconds": 1.168588638305664, "cellprob_max_abs_difference": 0.0, "flow_max_abs_difference": 0.0, "mask_exact_match": true, "independent_process": true, "shape": [520, 696], "passed": true}

![평가 원본과 예측](https://raw.githubusercontent.com/fff0996/Bioinformatics_AImodel/49661373a7b790bbbbcce0d7ef8bf96b8ce83dbd/cellpose/reports/prediction_example.png)

![학습과 검증 loss](https://raw.githubusercontent.com/fff0996/Bioinformatics_AImodel/49661373a7b790bbbbcce0d7ef8bf96b8ce83dbd/cellpose/reports/training_loss.png)

## 재현과 한계

코드: `scripts/setup.py`, `prepare_inputs.py`, `benchmark.py`, `launch.py`, `reload.py`, `report.py`. 실행 명령과 로그는 각 run 폴더에 있다. 첫 실행의 결과 기록 함수 이름 충돌은 GPU 추론 이후 발생해 해당 실패 run을 보존하고 수정 후 새 run으로 재실행했다. 가중치 구간 다운로드 조립 크기 검증이 처음 실패해 기존 다운로드를 실제 종료하고 보존 prefix 길이에 맞춰 재조립했으며 최종 공식 SHA-256 일치로 검증했다.

데이터 선택·tile 정책을 고정한 단일 장비 단일 실행 관찰이다. GPU 연산 커널만의 순수 FLOPS, 다른 장비 대비 속도 향상, 광범위한 생물학적 일반화를 주장하지 않는다. 큰 이미지의 병목은 실제 타일 batch·메모리·시간을 바탕으로 해석하며, 확인하지 않은 원인은 추정으로 남긴다.

## 공식 출처

- [Cellpose 모델·tile 정책](https://cellpose.readthedocs.io/en/latest/models.html)
- [공식 학습 안내](https://cellpose.readthedocs.io/en/latest/train.html)
- [공식 소스](https://github.com/MouseLand/cellpose)
- [공식 체크포인트](https://huggingface.co/mouseland/cellpose-sam)
- [BBBC039 데이터 설명·분할](https://bbbc.broadinstitute.org/BBBC039)
- [공식 BBBC039 mask decoding 예](https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075)
