# Cellpose 최종 통합 보고서

## 1. 모델과 실제 입력

이미지 → 핵/세포별 instance mask. cpsam_v2(약304.6M),영상개수와동시GPU타일개수를구분한다.

실제파일·입력/정답예시·전처리와모델구조설명은이문서5절에전문을포함했다. 추론데이터와지도학습label을구분하며,세포/유전자·서열/토큰·이미지/타일의단위를섞지않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

BBBC039는실제U2OS핵형광이미지이고,NucMM-M은생쥐뇌조직micro-CT다. **원본이미지개수와동시에GPU에넣는타일개수는별개**다. 타일network구간과최종instance mask까지의전체분할시간도구분한다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step참고|최대 allocated / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|타일 network 추론|고유이미지1장(520×696),고유타일12개|최대8×1×256×256,BF16|24회반복; 총288타일|15.54초|18.53타일/초|1.491 GiB / 1.23%|mask후처리제외; 해당자원조건의정확도별도미평가; 80%미도달|
|타일 network 추론|고유이미지32장(520×696),고유타일384개|최대384×1×256×256,BF16|2회반복; 총768타일|41.40초|18.55타일/초|43.397 GiB / 35.66%|mask후처리제외; 해당자원조건의정확도별도미평가; 80%미도달|
|정답마스크 기반 파인튜닝|train16/val8/test16개 실제U2OS형광이미지|1×3×256×256 crop,FP32|100epoch/1600optimizer step|672.75초|전체wall÷1600=0.420초/step 참고값|7.985 GiB / 6.56%|저장모델test AP50 0.9138; 80%미도달|
|뇌 micro-CT 2D 전체분할|4개val volume의중앙단면4장,각192×192|요청tilebatch8; 실제tilebatch1,1×1×256×256,BF16|59회반복/총236장|15.18초|15.54장/초|0.705 GiB / 0.58%|동일4단면 AP50 0.4407; 80%미도달|
|뇌 micro-CT 3D 전체분할|4개실제val patch,각192×192×192복셀; 원본총768슬라이스|tilebatch8,8×1×256×256,BF16|각volume1회|114.89초(4개합)|0.0348volume/초|1.491 GiB / 1.23%|동일4volume 3D AP50 0.4222; 80%미도달|

## 4. 결과 해석

- 동일256타일·BF16 network경로에서batch8→384는1.491→43.397GiB로메모리가늘었지만처리량은18.53→18.55타일/초로거의같았다.
- 원본520×696이미지1장은12타일이다. batch384는원본384장이아니라32장분량의384타일을동시에계산한조건이다.
- 학습은FP32·gradient/optimizer·증강을포함해추론과다르다. 기존batch1 학습7.985GiB만으로큰batch의학습한계를판정할수없고,학습batch확대는아직완료하지않았다.
- 뇌영상은192³의작은조직patch4개이지5000건의뇌MRI나전체뇌가아니다. 3D AP50평균0.422로추가학습·검증없이자동라벨의정확성을보장하지않는다.
- 80%목표조건은현재미확인이다. 이전batch576은예상guard로건너뛴것이고실제실패나불가능증명이아니다.

### 시간·성능 표를 읽는 주의점

타일network시간은mask후처리를제외한다. 파인튜닝672.75초는원래공식trainer의flow전처리·validation·저장도포함한전체시간이며순수step시간이아니다. 0.420초/step는그전체를1600으로나눈참고값이므로scGPT의순수학습step시간과직접비교하지않는다. 뇌2D/3D행은정규화·network·mask후처리를포함하고다운로드/모델적재/평가지표/파일저장을제외한다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### Cellpose-SAM: DGX Spark 실행 보고서


실행: `cellpose_20261027_193331_947064`. 상태: **complete**. 원본 측정: [summary.json](../results/cellpose_20261027_193331_947064/RECORDS.md#file-2e96f1dc76e2), 로그 (로컬 보관: `../results/cellpose_20261027_193331_947064/run.log`), [자원 시계열](../results/cellpose_20261027_193331_947064/RECORDS.md#file-7e3493105ec7), [실제 forward 입력](../results/cellpose_20261027_193331_947064/RECORDS.md#file-51657b33486d).

#### 모델과 환경

Cellpose 4.2.1, 공식 기본 체크포인트 **cpsam_v2**, SAM ViT-L 기반 Cellpose-SAM이다. Cellpose3 cyto·CPDINO와 별개다. 전체 파라미터 304,628,930, requires_grad 304,592,064. 이론적 BF16 파라미터 저장량 0.567416 GiB.

체크포인트 revision: `7c61431b5fbb078f3296754bd15d9f51b320f837`; SHA-256: `0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667`. 공식 파일 메타데이터 `../models/repository_files.json`의 LFS oid와 대조해 일치했다. 데이터/모델 출처 목록은 `../data/sources.json`에 보존했다.

환경: {"python": "3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]", "torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "cellpose": "4.2.1", "gpu": "NVIDIA GB10", "platform": "Linux-7.0.0-1019-nvidia-aarch64-with-glibc2.39", "tf32": true}

8 CPU threads, seed42. 공식 trainer는 epoch별 NumPy seed를 epoch 번호로, 검증 때42로 설정한다. GPU 작업은 한 번에 하나씩 실행했다. PyTorch NGC 26.06 ARM64 전용 venv이며 기존 환경은 보존했다. BF16 추론, 공식 학습 API가 자동 변환하는 FP32 파인튜닝이다. TF32 허용은 Cellpose 공식 코드 기본값이다.

| 적재 상태 | CUDA allocated GiB | reserved GiB | RSS GiB | system available GiB |
| --- | --- | --- | --- | --- |
| 모델 적재 직후, 첫 입력 이전 | 0.568277 | 1.740 | 0.954 | 111.969 |

적재 시간 1.129초; 적재 중 allocator peak 1.718GiB. 이는 현재 적재량과 다르다. 첫 warmup 1.377초는 초기 kernel/Numba 준비와 후처리를 포함하며 정상 반복 성능 표에서 제외했다.

#### 실제 데이터와 학습 입력

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

#### 입력 규모에 따른 GPU 사용률과 메모리

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

#### 정답 마스크 파인튜닝·검증·저장

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

#### 재현과 한계

코드: `scripts/setup.py`, `prepare_inputs.py`, `benchmark.py`, `launch.py`, `reload.py`, `report.py`. 실행 명령과 로그는 각 run 폴더에 있다. 첫 실행의 결과 기록 함수 이름 충돌은 GPU 추론 이후 발생해 해당 실패 run을 보존하고 수정 후 새 run으로 재실행했다. 가중치 구간 다운로드 조립 크기 검증이 처음 실패해 기존 다운로드를 실제 종료하고 보존 prefix 길이에 맞춰 재조립했으며 최종 공식 SHA-256 일치로 검증했다.

데이터 선택·tile 정책을 고정한 단일 장비 단일 실행 관찰이다. GPU 연산 커널만의 순수 FLOPS, 다른 장비 대비 속도 향상, 광범위한 생물학적 일반화를 주장하지 않는다. 큰 이미지의 병목은 실제 타일 batch·메모리·시간을 바탕으로 해석하며, 확인하지 않은 원인은 추정으로 남긴다.

#### 공식 출처

- [Cellpose 모델·tile 정책](https://cellpose.readthedocs.io/en/latest/models.html)
- [공식 학습 안내](https://cellpose.readthedocs.io/en/latest/train.html)
- [공식 소스](https://github.com/MouseLand/cellpose)
- [공식 체크포인트](https://huggingface.co/mouseland/cellpose-sam)
- [BBBC039 데이터 설명·분할](https://bbbc.broadinstitute.org/BBBC039)
- [공식 BBBC039 mask decoding 예](https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### Cellpose-SAM 실제 이미지의 타일 배치별 추론 자원


실행 `tile_batch_20261028_134130_265230` · 상태 **complete** · 종료 사유 `predicted_reservation_exceeds_available_budget`

#### 이미지 개수와 GPU 배치의 차이

원본BBBC039v1 이미지는200개, 각520×696·1채널·uint16이다. 각이미지를공식기본백분위정규화후float32 CPU배열로보관하고, padding·tile overlap0.1·256×256 고정타일·augment=False·resize없음을유지한다. 픽셀물리크기는확인되지않아입력검사에미확인으로표기했다. Cellpose-SAM cpsam_v2 BF16가중치이며기존체크포인트SHA256은원본summary에기록했다. Cellpose버전·Torch/CUDA·전체파라미터수·모델적재직후메모리는batch별summary에있다(기본모델304628930파라미터). BF16 가중치 이론용량은304628930×2bytes≈0.567GiB이며실측적재량과구분한다. 학습가능파라미터는이번추론에서0이다.

이미지당타일12개가생긴다. **batch8은원본이미지8개가아닌한번에타일최대8개**, 실제forward는8+4로분할된다. batch24는원본2이미지의24타일을동시에처리한다. 이미지내타일은서로겹치므로독립생물학적샘플24개라는뜻은아니다. 입력shape는실제 `[타일수,1,256,256]`,모델내부변환·중간표현메모리는별도다. 큰batch는실제여러이미지의타일을함께묶으며인공확대나같은타일복제는하지않는다. 최소2회·15초이상측정반복을하므로고유이미지수와총처리량을구분한다. 원본split메타데이터는유지하며자원pool에여러split을포함하지만학습이나조건별정확도선택은하지않았다.

#### 측정 범위

공식 core.run_net의CPU타일링·padding·전송·networkforward·CPU반환·tileblending을포함한다. 입력정규화·모델로딩·warmup·mask dynamics·파일저장은제외한다. 따라서전체segmentation 처리시간이아닌타일네트워크추론구간이다. 기존작은배치전체mask추론과직접속도비교하지않는다. 모델·타일크기·dtype·전처리를고정하고각조건fresh container로allocator누적을분리했다. 각구간CUDA동기화,0.5초대기+조회시계열을사용한다. RSS와시스템가용량은sampled값이다.

|요청batch|실제최대타일batch|고유이미지|고유타일|반복|총처리타일|초|타일/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|8|8|1|12|24|288|15.542|18.531|0.568|1.491|0.923|1.783|1.226|91.069|95.0|1.781|110.113|
|12|12|1|12|23|276|15.117|18.258|0.568|1.938|1.369|2.354|1.592|93.714|95.0|1.648|109.92|
|24|24|2|24|12|288|15.393|18.709|0.568|3.276|2.707|4.104|2.692|94.321|95.0|1.787|108.071|
|48|48|4|48|6|288|16.113|17.874|0.568|5.949|5.381|7.549|4.889|87.733|95.0|1.84|104.572|
|96|96|8|96|3|288|16.575|17.376|0.568|11.299|10.731|14.486|9.285|85.29|96.0|1.911|97.392|
|192|192|16|192|2|384|20.291|18.924|0.568|21.998|21.43|28.361|18.078|88.0|96.0|2.102|82.89|
|384|384|32|384|2|768|41.396|18.552|0.568|43.397|42.828|56.113|35.662|89.84|96.0|2.661|54.578|

#### 결과와 검증

시험한최대성공타일batch **384**. allocated80%도달: False,90%도달: False. 다음예상예약량과가용RAM16GiB·swap증가2GiB guard에의해확대를중단한다. 건너뛴batch는실제실패나모델절대한계가아니다.

출력flow/cellprob은전체이미지에재조립돼[N,520,696,3]이며각반복전체finite를확인했다. 첫이미지출력은저장후정확재로드를검증한다. batch8의전체공식model.eval과flow값일치를검증하고실제instance mask도저장했다. 다른batch의첫이미지flow차이·cosine은각summary에보존하며BF16의batch별수치차이를비트단위동일성으로표현하지않는다. 새학습·독립정확도평가를한실험은아니며기존학습/정답mask평가결과는기본보고서에있다.

GPU-Util은GPU활동시간비율,allocated/reserved는프로세스PyTorchallocator다. 분모121.688GiB는CPU/GPU공유RAM으로전용VRAM점유율과다르며RSS와CUDA를더하지않는다. 모델적재기준값과추가메모리는같은run에서구분했다. 정확한입력·출력·실제forwardshape·로딩/warmup시간·환경·메모리시계열·명령·로그·종료코드는원본결과폴더에저장했다.

원본 `results/tile_batch_20261028_134130_265230` · 실행코드 `scripts/batch_followup`

[공식 모델](https://cellpose.readthedocs.io/en/latest/models.html) · [공식 배치/타일 API](https://cellpose.readthedocs.io/en/latest/api.html) · [기본 Cellpose 결과](Cellpose.md)

### 4. 후속 실험 통합 — 뇌 조직 2D·3D 검증

### Cellpose-SAM 추가 검증: 생쥐 뇌 조직의 세포핵


상태 **complete** · 실행 `brain_nucmm_20261029_110603_965442`

#### 어떤 뇌 이미지인가?

NucMM-M의생쥐시각피질 **마이크로CT** 공개train/val패치다. 일반적인임상뇌MRI/CT의뇌영역·종양분할과제와다르며,과제는뇌조직안의**개별세포핵instance분할**이다. 기존U2OS형광이미지와촬영방식·입력구조가다르다.

원본연구프로젝트·README_original.txt와실제HDF5를대조했다. 배포미러README.md에는EM/200³라는잘못된설명이있지만원본설명은micro-CT/192³이고실제8파일shape도192³로확인됐다. 미러HF리비전5c489be7532664b14d90f4c028c6b2f9b6fcd782와파일별URL/SHA256·두README를data/brain_nucmm/download_manifest.json에보존했다. 원본GoogleDrive와byte동일성은독립적으로검증하지않았으며실제다운로드출처는공개미러임을명시한다.

|원본분할|고유3D패치|패치당shape|원본dtype|정답|
|---|---|---|---|---|
|train|4|192×192×192|uint8|uint16,핵별instance ID|
|val|4|192×192×192|uint8|uint16,핵별instance ID|

전체8volume은1536개원본슬라이스에해당하지만5000개독립뇌영상이아니다. 2D평가는val각volume중앙1장씩총4장,3D평가는val4volume전체총768슬라이스를사용한다. 모든패치가같은parent뇌volume에서나왔으며독립donor/test는아니다. 실제물리voxel크기·해부학적축방향은확인되지않았다. 계산시배열축Z/Y/X,anisotropy1을가정하므로물리보정된정확도주장으로해석하지않는다.

#### 모델·전처리·검증 정책

기존Cellpose4.2.1 cpsam_v2 체크포인트를그대로사용했다. 전체304628930파라미터,학습갱신0,BF16가중치·공식FP32흐름/CPU전처리·TF32=False·seed42. 같은run의모델적재직후allocated/reserved·RSS·MemAvailable과SHA256은summary에있다. 추가파인튜닝은수행하지않았다.

tile256×256·tilebatch8·overlap0.1·augment=False·resize/diameter변경없음·공식백분위정규화·flow_threshold0.4·cellprob_threshold0·min_size15·max_size_fraction0.4를고정했다. 원본핵은배경보다어두워**train4패치의정답영역/배경평균만으로invert=True를선택**했다. 검증정답으로조건을선택하지않았다. train분할은가중치학습에쓰지않고이정책확인·2Dwarmup에만사용했다.

2D는중앙단면별정규화와최종mask생성을포함하고,3D는공식do_3D=True 경로로직교방향network추론후3D흐름/마스크를생성했다. 3D의flow_threshold동작은공식경로를따르며2D와같은기능이라고가정하지않는다. 원본192×192평면은network입력에맞게padding되며256×256타일입력과구분한다. 실제forwardshape·타일처리수는forward_calls.json과각행에보존했다.

#### 전체 segmentation 시간과 자원

시간은입력배열이CPU에준비된후정규화·타일링·전송·networkforward·재조립·**최종마스크후처리까지포함**한다. 다운로드/HDF5변환·모델로딩·입력검사·평가지표·파일저장은제외한다. 2D는train중앙단면warmup후15초이상반복했고,3D는volume당1회측정으로첫3D설정비용도포함한다. network·mask단계는CUDA동기화로따로계측했다. 기존후속타일forward시간과직접비교하지않는다.

|조건|고유입력|반복|전체초|network초|mask후처리초|allocated peak GiB|reserved peak GiB|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|
|2d_central_validation_slices|4|59|15.1835|12.8156|1.9246|0.7054|0.7363|75.931|87.0|2.2465|111.5072|
|3d_img_000_200_576|1|1|28.7865|28.2204|0.2124|1.4914|1.7832|92.9245|96.0|2.5235|110.3805|
|3d_img_000_604_200|1|1|28.6605|28.1252|0.1728|1.4914|1.7852|90.7358|96.0|2.5215|110.363|
|3d_img_508_200_200|1|1|28.692|28.164|0.1704|1.4914|1.7832|89.1509|96.0|2.5214|110.3917|
|3d_img_508_604_576|1|1|28.7559|28.2129|0.1655|1.4914|1.7832|92.1698|96.0|2.5215|110.3909|

GPU-Util은커널활동시간비율이며메모리점유율이아니다. CUDAallocated/reserved는프로세스allocator,전체121.688GiB는CPU/GPU공유RAM이다. RSS와CUDA를더하지않는다. 시스템가용량·RSS는0.5초대기+조회간격의sampled값이다. 가용8GiB미만/swap+2GiBguard를사용한다.

#### 정답 마스크 대비 평가

|조건/패치|GT핵수|예측핵수|AP50|AP75|AP90|
|---|---|---|---|---|---|
|2d/img_000_200_576|23|9|0.3333|0.1852|0.0|
|2d/img_000_604_200|14|15|0.6111|0.2609|0.0357|
|2d/img_508_200_200|11|9|0.8182|0.3333|0.1111|
|2d/img_508_604_576|6|0|0.0|0.0|0.0|
|3d/img_000_200_576|282|314|0.6329|0.0643|0.0|
|3d/img_000_604_200|192|176|0.5593|0.1185|0.0|
|3d/img_508_200_200|101|78|0.2878|0.0783|0.0|
|3d/img_508_604_576|88|51|0.2087|0.0451|0.0|

3D평가는3D instance IoU,2D평가는해당단면의instance IoU를사용한다. GT의비연속ID는object를합치지않고연속재번호화했으며원래ID매핑을저장했다. AP는TP/(TP+FP+FN)인Cellpose instance평가지표다. pLDDT같은신뢰도나영상단위분류정확도가아니다. 평가4volume/같은parent뇌·외부사전학습데이터중복불확실성때문에임상/전체뇌일반화성능을입증하지않는다. 부정확한예측은정확히보고하고이미지예시와함께확인한다.

#### 실제 입력·정답·예측 예시

![뇌조직 중앙단면의 원본·정답·2D예측·3D예측](https://raw.githubusercontent.com/fff0996/Bioinformatics_AImodel/49661373a7b790bbbbcce0d7ef8bf96b8ce83dbd/cellpose/results/brain_nucmm_20261029_110603_965442/brain_prediction_examples.png)

#### 보관 파일

원본HDF5·다운로드SHA/리비전·입력검사·정규화선택근거는data/brain_nucmm. 결과run에는원본코드snapshot·명령·exit·로그·GPU/RAM CSV·stage/forward계측·2D/3D마스크·flow/cellprob·원본GT ID·평가JSON·예시PNG를보존한다. 저장마스크는재로드해정확히같음을확인했다. 새모델을학습하지않아새학습체크포인트나학습재개를검증했다는주장은하지않는다.

[NucMM 원본 연구/데이터 설명](https://connectomics-bazaar.github.io/proj/nucMM/index.html) · [실제 배포 미러](https://huggingface.co/datasets/MedOtter/NucMM-M) · [Cellpose 공식 API](https://cellpose.readthedocs.io/en/latest/api.html)

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
|`cellpose/scripts/batch_followup/launch.py`|`e33d79d4d385e1b831c4991c024816c49c041892427e7eabb8db1b2bcb5a99f2`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/batch_followup/prepare.py`|`18524351c8e37842e56f80af19fad5f73724eec977c39b41d9d03e7f796b81c5`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/batch_followup/report.py`|`1696a39a4bf09cd93f4be02df32e86efb9d5940501b91f2d77fc7dc6e8f8f3c3`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/batch_followup/worker.py`|`dbaf2034b557ac3f6e7b44c1af00054f597788f14e9d2ff53d6f42a80c7ce154`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/benchmark.py`|`92bb0bc8aa6660bb49907f9a7489e0fb0186a1e9534a4c2ac61773ffe2122064`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/brain/audit.py`|`0800b4eba9381146a96f8f3a83e7ba6fb13e4d16f8af2c46e58bf7621a27b9f4`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/brain/benchmark.py`|`bf1a7ffbbd62f210a25901f2a3fc63c2ba6ea95a41e886c33cc980856e13cf79`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/results/brain_nucmm_20261029_110603_965442/benchmark_source.py`|`bf1a7ffbbd62f210a25901f2a3fc63c2ba6ea95a41e886c33cc980856e13cf79`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`cellpose/scripts/brain/download.py`|`db6a3d031b4a69c6d478c98d47005a8c4ebb6d86dba455332c89d203975e0f4d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/brain/launch.py`|`6a129f4bbd1b4f9c7b53a8b296bfa8dbf587258701fd8380bec6cdea03366bbf`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/brain/plot.py`|`e312391b7ab33c4d293fc352b41898d1cd88561c5d3daad6dbb496f696cec6f4`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/brain/report.py`|`f450e53ccb5d0f977deb9c331c02639f62e1328d085b4a3f4af18a5be87a7566`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/download.py`|`d41375c0f1b41314cca8065c9d538d6529ac44a03675b83e054957ea8753e89d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/download_ranges.py`|`e8e12fd27fdc81583f69f09f8e73cc0140214f8d206e78588938ac41dd1bbf51`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/launch.py`|`c51e34d1f8266b54455565b6225b61c3ad566af1e0c14d008bc9ae6737613fe1`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/prepare_inputs.py`|`fc95d36e4d480267e1581dfb7a22ad837005d9913b0cd74b3cf690e312e8dcd2`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/reload.py`|`bd6883884c6fdf28df35379367163b07c5b50f00df9cff6348ebaa5a61b3ff38`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/report.py`|`1418a73e6c2424df48d44c6488993adb772aeccd29bfe8a6a14f6cfe93eb9d44`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`cellpose/scripts/setup.py`|`60b6d7fa24925ede16a3e8d4f3d52b8d96a807e369475641e53ac88d2ccee005`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|

<details>
<summary>cellpose/scripts/batch_followup/launch.py — 원문 전문 (4,273 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/batch_followup/launch.py", "sha256": "e33d79d4d385e1b831c4991c024816c49c041892427e7eabb8db1b2bcb5a99f2", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,time,datetime,subprocess,shlex,sys,hashlib
import psutil
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'cellpose';run='tile_batch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');print(run,flush=True);swap0=psutil.swap_memory().used
S={'status':'running','run_id':run,'conditions':[],'failed':[],'checkpoint':'cpsam_v2','checkpoint_sha256':hashlib.file_digest((R/'models/cpsam_v2').open('rb'),'sha256').hexdigest(),'settings':{'precision':'BF16 inference, CPU normalized data FP32','tile_size':256,'tile_overlap':.1,'seed':42,'tf32':False,'inference_only':True,'time_scope':'official core.run_net: CPU tiling/padding, transfer, forward, CPU output, tile blending; excludes normalization/load/warmup/mask dynamics/save','guard_available_gib':16,'guard_swap_increase_gib':2,'minimum_repeats':2,'minimum_measured_seconds':15,'fresh_container_each_batch':True},'largest_successful_batch':None}
def save():(D/'summary.json').write_text(json.dumps(S,indent=2))
def stage(name,script,batch=None):
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'other GPU compute job'
 container='cp-'+run.lower()+'-'+name;args=['docker','run','--rm','--name',container,'--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/shared:ro','-e','RUN_ID='+run,'-e','MPLCONFIGDIR=/work/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/work/.cache/numba','-e','CELLPOSE_LOCAL_MODELS_PATH=/work/models']
 if batch is not None:args+=['--gpus','all','-e','TILE_BATCH='+str(batch)]
 args+=['nvcr.io/nvidia/pytorch:26.06-py3','/work/.venv/bin/python','-u','/work/scripts/batch_followup/'+script];(D/(name+'_command.json')).write_text(json.dumps(args,indent=2));t=time.time()
 with (D/(name+'.log')).open('w') as f:
  p=subprocess.Popen(['sg','docker','-c',shlex.join(args)],stdout=f,stderr=subprocess.STDOUT)
  reason=None
  while p.poll() is None:
   if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',container])],stdout=f,stderr=subprocess.STDOUT);break
   time.sleep(.5)
  code=p.wait()
 (D/(name+'_exit.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-t,'stop_reason':reason}));print(name,'EXIT',code,flush=True);return code
save()
if stage('prepare','prepare.py')!=0:S['status']='failed_prepare';save();sys.exit(1)
audit=json.loads((D/'input_audit.json').read_text());save()
for batch in [8,12,24,48,96,192,384,576,768,960,1152,1536,1920,2400]:
 n=max(1,batch//12)
 if n>audit['unique_source_images']:S['stop_reason']='available_real_image_pool';break
 available=psutil.virtual_memory().available/2**30
 if S['conditions']:
  last=S['conditions'][-1];base=last['model_only_allocated_gib'];estimate=base+(last['memory']['peak_reserved_gib']-base)*batch/last['actual_max_tile_batch']*1.25+3
  if estimate>available-16:S['stop_reason']='predicted_reservation_exceeds_available_budget';S['skipped_next_batch']={'requested':batch,'estimated_reserved_gib':estimate,'available_gib':available};break
 code=stage('batch'+str(batch),'worker.py',batch)
 if code:S['failed'].append({'batch':batch,'exit_code':code});S['stop_reason']='worker_stopped_see_log';break
 worker=json.loads((D/f'batch_{batch}/summary.json').read_text());assert worker['status']=='complete';row=worker['condition'];row['model_only_allocated_gib']=worker['model_only_postload']['allocated_gib'];S['conditions'].append(row);S['largest_successful_batch']=row['actual_max_tile_batch'];save()
else:S['stop_reason']='planned_list_complete'
S['status']='complete' if not S['failed'] else 'stopped';S['reached80']=any(c['allocated_fraction_shared_ram']>=.8 for c in S['conditions']);S['reached90']=any(c['allocated_fraction_shared_ram']>=.9 for c in S['conditions']);save();subprocess.run([sys.executable,str(R/'scripts/batch_followup/report.py'),str(D)],check=True);(D/'exit.json').write_text(json.dumps({'exit_code':0,'scope':'controller finished; worker failures recorded separately'}));print('FINISHED',S.get('stop_reason'),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/batch_followup/prepare.py — 원문 전문 (2,155 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/batch_followup/prepare.py", "sha256": "18524351c8e37842e56f80af19fad5f73724eec977c39b41d9d03e7f796b81c5", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,os,hashlib
import numpy as np,tifffile
from cellpose import transforms,models
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];audit=json.loads((R/'data/input_audit.json').read_text());rows=audit['all'];images=[];meta=[]
for r in rows:
 x=tifffile.imread(R/r['image']);assert list(x.shape)==[520,696] and str(x.dtype)=='uint16';assert hashlib.file_digest((R/r['image']).open('rb'),'sha256').hexdigest()==r['image_sha256'];y=transforms.convert_image(x,channel_axis=None,z_axis=None,do_3D=False);y=y[None] if y.ndim<4 else y;opts=dict(models.normalize_default);opts['norm3D']=False;y=transforms.normalize_img(y,**opts);assert y.shape==(1,520,696,1) and np.isfinite(y).all();images.append(y[0]);meta.append({'id':r['id'],'source_image':r['image'],'split':r['split'],'image_sha256':r['image_sha256'],'original_dtype':str(x.dtype),'original_shape':list(x.shape),'pixel_size_um':'not verified; no physical resizing applied'})
stack=np.stack(images).astype(np.float32);np.save(D/'normalized_images.npy',stack);pad=transforms.get_pad_yx(520,696,min_size=(256,256));padded=np.pad(stack[0].transpose(2,0,1),[[0,0],[pad[0],pad[1]],[pad[2],pad[3]]]);tiles,ys,xs,ly,lx=transforms.make_tiles(padded,bsize=256,augment=False,tile_overlap=.1);ntiles=tiles.shape[0]*tiles.shape[1];assert ntiles==12 and len({m['id'] for m in meta})==len(meta);s={'source_dataset':'BBBC039v1','unique_source_images':len(meta),'normalized_image_shape':list(stack.shape),'normalization':'official models.normalize_default, per image, norm3D=False, no resizing','tiles_per_image':int(ntiles),'tile_shape':[1,256,256],'tile_overlap':.1,'augment':False,'pad_yx':list(map(int,pad)),'tile_y_positions':ys,'tile_x_positions':xs,'images':meta,'labels':'instance masks retained in source dataset; resource sweep only, no training and no new accuracy claim','splits':'resource-only pool includes existing train/validation/test images; no weight updates or test-based hyperparameter selection'};(D/'input_audit.json').write_text(json.dumps(s,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x)));print('PREPARED',stack.shape,ntiles,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/batch_followup/report.py — 원문 전문 (5,963 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/batch_followup/report.py", "sha256": "1696a39a4bf09cd93f4be02df32e86efb9d5940501b91f2d77fc7dc6e8f8f3c3", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,csv,sys
D=Path(sys.argv[1]);R=D.parents[1];S=json.loads((D/'summary.json').read_text());audit=json.loads((D/'input_audit.json').read_text());lines=['# Cellpose-SAM 실제 이미지의 타일 배치별 추론 자원','',f"실행 `{D.name}` · 상태 **{S['status']}** · 종료 사유 `{S.get('stop_reason','running')}`",'','## 이미지 개수와 GPU 배치의 차이','','원본BBBC039v1 이미지는200개, 각520×696·1채널·uint16이다. 각이미지를공식기본백분위정규화후float32 CPU배열로보관하고, padding·tile overlap0.1·256×256 고정타일·augment=False·resize없음을유지한다. 픽셀물리크기는확인되지않아입력검사에미확인으로표기했다. Cellpose-SAM cpsam_v2 BF16가중치이며기존체크포인트SHA256은원본summary에기록했다. Cellpose버전·Torch/CUDA·전체파라미터수·모델적재직후메모리는batch별summary에있다(기본모델304628930파라미터). BF16 가중치 이론용량은304628930×2bytes≈0.567GiB이며실측적재량과구분한다. 학습가능파라미터는이번추론에서0이다.','','이미지당타일12개가생긴다. **batch8은원본이미지8개가아닌한번에타일최대8개**, 실제forward는8+4로분할된다. batch24는원본2이미지의24타일을동시에처리한다. 이미지내타일은서로겹치므로독립생물학적샘플24개라는뜻은아니다. 입력shape는실제 `[타일수,1,256,256]`,모델내부변환·중간표현메모리는별도다. 큰batch는실제여러이미지의타일을함께묶으며인공확대나같은타일복제는하지않는다. 최소2회·15초이상측정반복을하므로고유이미지수와총처리량을구분한다. 원본split메타데이터는유지하며자원pool에여러split을포함하지만학습이나조건별정확도선택은하지않았다.','','## 측정 범위','','공식 core.run_net의CPU타일링·padding·전송·networkforward·CPU반환·tileblending을포함한다. 입력정규화·모델로딩·warmup·mask dynamics·파일저장은제외한다. 따라서전체segmentation 처리시간이아닌타일네트워크추론구간이다. 기존작은배치전체mask추론과직접속도비교하지않는다. 모델·타일크기·dtype·전처리를고정하고각조건fresh container로allocator누적을분리했다. 각구간CUDA동기화,0.5초대기+조회시계열을사용한다. RSS와시스템가용량은sampled값이다.','','|요청batch|실제최대타일batch|고유이미지|고유타일|반복|총처리타일|초|타일/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'];flat=[]
for c in S['conditions']:
 m=c['memory'];base=c['model_only_allocated_gib'];r={'requested_batch':c['requested_tile_batch'],'actual_max_tile_batch':c['actual_max_tile_batch'],'unique_images':c['unique_images'],'unique_tiles':c['unique_tiles'],'repeats':c['repeats'],'processed_tiles':c['processed_tiles'],'seconds':c['seconds'],'tiles_per_second':c['tiles_per_second'],'loaded_allocated_gib':base,'peak_allocated_gib':m['peak_allocated_gib'],'additional_from_same_run_load_gib':m['peak_allocated_gib']-base,'peak_reserved_gib':m['peak_reserved_gib'],'percent_shared_ram':100*c['allocated_fraction_shared_ram'],'gpu_avg':c['gpu_util_avg'],'gpu_max':c['gpu_util_max'],'rss_sampled_peak_gib':c['rss_sampled_peak_gib'],'available_sampled_min_gib':c['available_sampled_min_gib']};flat.append(r);lines.append('|'+ '|'.join(str(round(v,3)) if isinstance(v,float) else str(v) for v in r.values())+'|')
lines+=['','## 결과와 검증','',f"시험한최대성공타일batch **{S.get('largest_successful_batch')}**. allocated80%도달: {S.get('reached80')},90%도달: {S.get('reached90')}. 다음예상예약량과가용RAM16GiB·swap증가2GiB guard에의해확대를중단한다. 건너뛴batch는실제실패나모델절대한계가아니다.",'','출력flow/cellprob은전체이미지에재조립돼[N,520,696,3]이며각반복전체finite를확인했다. 첫이미지출력은저장후정확재로드를검증한다. batch8의전체공식model.eval과flow값일치를검증하고실제instance mask도저장했다. 다른batch의첫이미지flow차이·cosine은각summary에보존하며BF16의batch별수치차이를비트단위동일성으로표현하지않는다. 새학습·독립정확도평가를한실험은아니며기존학습/정답mask평가결과는기본보고서에있다.','','GPU-Util은GPU활동시간비율,allocated/reserved는프로세스PyTorchallocator다. 분모121.688GiB는CPU/GPU공유RAM으로전용VRAM점유율과다르며RSS와CUDA를더하지않는다. 모델적재기준값과추가메모리는같은run에서구분했다. 정확한입력·출력·실제forwardshape·로딩/warmup시간·환경·메모리시계열·명령·로그·종료코드는원본결과폴더에저장했다.','',f"원본 `results/{D.name}` · 실행코드 `scripts/batch_followup`",'','[공식 모델](https://cellpose.readthedocs.io/en/latest/models.html) · [공식 배치/타일 API](https://cellpose.readthedocs.io/en/latest/api.html) · [기본 Cellpose 결과](Cellpose.md)']
(R/'reports/Cellpose_batch_memory.md').write_text('\n'.join(lines)+'\n')
if flat:
 with (R/'reports/batch_memory_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for filename in ['STATUS.md','RESULTS.md','MEMORY_RESULTS.md']:
 with (R.parent/filename).open('a') as f:f.write('\nCellpose 타일배치후속 '+D.name+' 상태 '+S['status']+', 최대성공타일batch '+str(S.get('largest_successful_batch'))+',80%도달 '+str(S.get('reached80'))+',90%도달 '+str(S.get('reached90'))+'. [보고서](cellpose/reports/Cellpose_batch_memory.md). 추가모델실험자동시작없음.\n')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/batch_followup/worker.py — 원문 전문 (3,893 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/batch_followup/worker.py", "sha256": "dbaf2034b557ac3f6e7b44c1af00054f597788f14e9d2ff53d6f42a80c7ce154", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,json,time,gc,importlib.metadata
import numpy as np,torch
from cellpose import models,core,transforms
sys.path.insert(0,'/shared');from gpu_support import Meter,write_json
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];B=int(os.environ['TILE_BATCH']);C=D/f'batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','task':'official run_net tiled-image network inference; excludes mask dynamics','requested_tile_batch':B}
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='model_loading';t=time.time();model=models.CellposeModel(gpu=True,pretrained_model=str(R/'models/cpsam_v2'),use_bfloat16=True);torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for p in model.net.parameters());S['parameters_trainable_in_run']=0;S['weights_dtype']=str(model.net.dtype);S['environment']={'cellpose':importlib.metadata.version('cellpose'),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)};write_json(C/'summary.json',S)
 source=np.load(D/'normalized_images.npy',mmap_mode='r');N=max(1,B//12);imgs=np.asarray(source[:N]);assert len(imgs)==N;actual=[];original=core._forward
 def forward(net,x):
  actual.append(list(x.shape));return original(net,x)
 core._forward=forward
 def run():
  flows,styles=core.run_net(model.net,imgs,batch_size=B,augment=False,tile_overlap=.1,bsize=256,rsz=None);assert flows.shape==(N,520,696,3) and np.isfinite(flows).all();return flows
 meter.phase='warmup';t=time.time();flow=run();torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;S['warmup_actual_forward_shapes']=actual[:];actual.clear();del flow;gc.collect();t,base=meter.begin('official_run_net_inference');steps=0
 while steps<2 or time.time()-t<15:flow=run();torch.cuda.synchronize();steps+=1
 r=meter.end(t,base,requested_tile_batch=B,unique_images=N,unique_tiles=N*12,repeats=steps,processed_images=N*steps,processed_tiles=N*12*steps);r['actual_forward_shapes']=actual;r['actual_max_tile_batch']=max(x[0] for x in actual);r['images_per_second']=N*steps/r['seconds'];r['tiles_per_second']=N*12*steps/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];S['condition']=r
 np.save(C/'first_image_flow_cellprob.npy',flow[0]);saved=np.load(C/'first_image_flow_cellprob.npy');assert np.array_equal(saved,flow[0]);S['output_saved_exact_reload']=True;S['all_output_shape']=[N,520,696,3];S['all_outputs_finite']=True
 if B==8:
  audit=json.loads((D/'input_audit.json').read_text());import tifffile
  raw=tifffile.imread(R/audit['images'][0]['source_image']);mask,fl,style=model.eval(raw,batch_size=8,bsize=256,tile_overlap=.1,augment=False,normalize=True,diameter=None,flow_threshold=.4,cellprob_threshold=0.,min_size=15,max_size_fraction=.4);reference=np.concatenate([np.moveaxis(fl[1],0,-1),fl[2][...,None]],axis=-1);diff=float(np.abs(reference-flow[0]).max());S['whole_image_eval_reference_max_abs_difference']=diff;assert np.allclose(reference,flow[0],atol=1e-5,rtol=1e-5);np.save(C/'reference_segmentation_mask.npy',mask);S['reference_mask_shape']=list(mask.shape)
 else:
  ref=np.load(D/'batch_8/first_image_flow_cellprob.npy');delta=np.abs(ref-flow[0]);S['batch_invariance']={'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'cosine':float(np.dot(ref.ravel(),flow[0].ravel())/(np.linalg.norm(ref)*np.linalg.norm(flow[0])))};assert S['batch_invariance']['cosine']>.99
 S['status']='complete';write_json(C/'summary.json',S);print('BATCH_COMPLETE',B,'actual',r['actual_max_tile_batch'],'peak',r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write_json(C/'summary.json',S);raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/benchmark.py — 원문 전문 (11,779 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/benchmark.py", "sha256": "92bb0bc8aa6660bb49907f9a7489e0fb0186a1e9534a4c2ac61773ffe2122064", "encoding": "utf-8", "added_final_newline": false} -->
```python
import os,time,json,csv,threading,subprocess,logging,hashlib,random,sys,gc,platform,importlib.metadata
from pathlib import Path
import numpy as np, tifffile, psutil, torch
from cellpose import models,core,train,metrics,dynamics
R=Path('/work');runid=os.environ['RUN_ID'];D=R/'results'/runid;D.mkdir(exist_ok=True);M=R/'models'/runid;M.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(message)s',handlers=[logging.StreamHandler()])
torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
phase='initialization';condition='';stop=threading.Event();events=[];samples=[];G=1024**3
S={'run_id':runid,'seed':42,'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cellpose':importlib.metadata.version('cellpose'),'gpu':torch.cuda.get_device_name(0),'platform':platform.platform(),'tf32':torch.backends.cuda.matmul.allow_tf32},'conditions':[]}
def dump():
 (D/'summary.json').write_text(json.dumps(S,indent=2,default=str));(D/'events.json').write_text(json.dumps(events,indent=2,default=str))
def mem():
 v=psutil.virtual_memory();return {'allocated_gib':torch.cuda.memory_allocated()/G,'reserved_gib':torch.cuda.memory_reserved()/G,'peak_allocated_gib':torch.cuda.max_memory_allocated()/G,'peak_reserved_gib':torch.cuda.max_memory_reserved()/G,'rss_gib':psutil.Process().memory_info().rss/G,'system_available_gib':v.available/G,'system_total_gib':v.total/G,'swap_used_gib':psutil.swap_memory().used/G}
def event(event_name,**kw):
 e={'time_unix':time.time(),'phase':phase,'condition':condition,'event':event_name,**kw};events.append(e);print('EVENT',json.dumps(e,default=str),flush=True)
def monitor():
 with (D/'resources.csv').open('w',newline='') as f:
  cols=['query_start','query_end','phase','condition','gpu_util_percent','rss_gib','available_gib','swap_gib'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  while not stop.is_set():
   t=time.time();p=phase;c=condition
   try:
    q=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).strip().splitlines()[0];u=float(q)
   except Exception:u=None
   v=psutil.virtual_memory();row=dict(query_start=t,query_end=time.time(),phase=p,condition=c,gpu_util_percent=u,rss_gib=psutil.Process().memory_info().rss/G,available_gib=v.available/G,swap_gib=psutil.swap_memory().used/G);samples.append(row);w.writerow(row);f.flush()
   if v.available<8*G or psutil.swap_memory().used>initialswap+2*G:
    (D/'memory_guard.json').write_text(json.dumps(row));os._exit(86)
   stop.wait(.5)
def stats(t0,t1):
 rows=[r for r in samples if r['query_start']>=t0 and r['query_end']<=t1];us=[r['gpu_util_percent'] for r in rows if r['gpu_util_percent'] is not None]
 return {'sample_count':len(rows),'gpu_util_avg':float(np.mean(us)) if us else None,'gpu_util_max':max(us) if us else None,'rss_sampled_peak_gib':max([r['rss_gib'] for r in rows],default=None),'available_sampled_min_gib':min([r['available_gib'] for r in rows],default=None)}
initialswap=psutil.swap_memory().used;th=threading.Thread(target=monitor,daemon=True);th.start()
audit=json.loads((R/'data/input_audit.json').read_text());data={s:[tifffile.imread(R/r['image']) for r in rows] for s,rows in audit['selected'].items()};gt={s:[np.load(R/r['mask']) for r in rows] for s,rows in audit['selected'].items()};S['input_audit']='data/input_audit.json';S['settings']={'inference_dtype':'bfloat16','tile_batch_size':8,'tile_size':256,'tile_overlap':.1,'augment':False,'normalize':'official default 1st/99th percentile; inference1channel, training grayscale+two zero channels','diameter':None,'rescale':None,'flow_threshold':.4,'cellprob_threshold':0.,'min_size':15,'max_size_fraction':.4,'training_batch':1,'training_epochs':100,'learning_rate':1e-5,'weight_decay':.1,'training_dtype':'float32 (official automatic conversion)','training_augmentation':'official random rotate/resize, scale_range=.5, rescale=False, 256 crop','seed_note':'global42; official trainer resets numpy seed to epoch index and validation42'}
forwardrows=[];origforward=core._forward
def forward(net,x):
 torch.cuda.synchronize();t=time.time();y=origforward(net,x);torch.cuda.synchronize();forwardrows.append({'start':t,'end':time.time(),'phase':phase,'condition':condition,'shape':list(x.shape),'dtype':str(net.dtype)});return y
core._forward=forward
opts=dict(batch_size=8,normalize=True,diameter=None,rescale=None,flow_threshold=.4,cellprob_threshold=0.,min_size=15,max_size_fraction=.4,augment=False,tile_overlap=.1,bsize=256,do_3D=False)
def evaluate(model,imgs):return model.eval(imgs,**opts)
def infer_condition(model,name,imgs,minseconds=15):
 global phase,condition
 phase='inference';condition=name;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=mem();t=time.time();idx=len(forwardrows);repeats=0;first=None
 while repeats==0 or time.time()-t<minseconds:
  out=evaluate(model,imgs)
  if first is None:first=out
  repeats+=1
 torch.cuda.synchronize();end=time.time();fs=forwardrows[idx:];row={'name':name,'unique_images':len(imgs),'raw_shapes':[list(x.shape) for x in imgs],'repeats':repeats,'processed_images':len(imgs)*repeats,'seconds':end-t,'images_per_second':len(imgs)*repeats/(end-t),'forward_including_transfers_seconds':sum(f['end']-f['start'] for f in fs),'tile_forward_calls':len(fs),'actual_forward_shapes':sorted({str(f['shape']) for f in fs}),'processed_tiles':sum(f['shape'][0] for f in fs),'baseline':base,'memory':mem(),**stats(t,end)};row['peak_delta_allocated_gib']=row['memory']['peak_allocated_gib']-base['allocated_gib'];S['conditions'].append(row);event('condition_complete',**row);dump();return first
try:
 torch.cuda.init();torch.cuda.empty_cache();S['before_load']=mem();phase='loading';t=time.time();model=models.CellposeModel(gpu=True,pretrained_model=str(R/'models/cpsam_v2'),use_bfloat16=True);torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=mem();S['parameters']={'total':sum(p.numel() for p in model.net.parameters()),'requires_grad':sum(p.numel() for p in model.net.parameters() if p.requires_grad),'parameter_dtypes':{str(d):sum(p.numel() for p in model.net.parameters() if p.dtype==d) for d in {p.dtype for p in model.net.parameters()}},'theoretical_current_storage_gib':sum(p.numel()*p.element_size() for p in model.net.parameters())/G};event('model_loaded',**S['parameters'],memory=S['model_only_postload']);dump()
 phase='warmup';condition='native_one';t=time.time();warm=evaluate(model,data['test'][:1]);torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;np.save(D/'warmup_mask.npy',warm[0][0]);assert warm[0][0].shape==data['test'][0].shape;dump()
 for n in [1,4,16]:infer_condition(model,f'count_{n}',data['test'][:n])
 for size in [128,256,512]:
  imgs=[x[(x.shape[0]-size)//2:(x.shape[0]+size)//2,(x.shape[1]-size)//2:(x.shape[1]+size)//2] for x in data['test'][:4]]
  infer_condition(model,f'crop_{size}',imgs)
 phase='evaluation';condition='pretrained_test';t=time.time();pre=evaluate(model,data['test']);torch.cuda.synchronize();S['pretrained_test_seconds']=time.time()-t
 def score(pred,name):
  ap,tp,fp,fn=metrics.average_precision(gt['test'],pred,threshold=[.5,.75,.9]);row={'mean_AP50_AP75_AP90':ap.mean(0).tolist(),'pooled_TP_FP_FN':{'tp':tp.sum(0).tolist(),'fp':fp.sum(0).tolist(),'fn':fn.sum(0).tolist()},'per_image_ap':ap.tolist(),'predicted_instances':[int(p.max()) for p in pred]};S[name]=row;return row
 score(pre[0],'pretrained_accuracy');np.savez_compressed(D/'pretrained_test_masks.npz',**{str(i):m for i,m in enumerate(pre[0])});dump()
 # Preserve exact forward values to verify a fresh-process reload after training.
 before={n:p.detach().float().cpu().clone() for n,p in model.net.named_parameters() if n in ['out.weight','encoder.blocks.0.attn.qkv.weight','encoder.patch_embed.proj.weight']}
 # Observe official conversion before preprocessing and optimizer without changing training algorithm.
 oldprocess=train._process_train_test
 def process(*a,**kw):
  global phase
  phase='training_preprocessing';torch.cuda.synchronize();S['training_fp32_baseline']=mem();t=time.time();res=oldprocess(*a,**kw);torch.cuda.synchronize();S['training_preprocessing_seconds']=time.time()-t;phase='training';event('training_preprocessing_complete');dump();return res
 train._process_train_test=process
 oldadam=torch.optim.AdamW;optimizerbox=[]
 def adam(*a,**kw):
  o=oldadam(*a,**kw);optimizerbox.append(o);return o
 torch.optim.AdamW=adam
 shapes={};gradcounts={};stepn=[0];oldnet=model.net.forward
 def netforward(x):
  global phase
  phase='training' if model.net.training else ('training_validation' if condition=='finetune' else phase)
  key=str((tuple(x.shape),str(x.dtype),model.net.training));shapes[key]=shapes.get(key,0)+1
  return oldnet(x)
 model.net.forward=netforward
 oldstep=oldadam.step
 def step(self,*a,**kw):
  if stepn[0]==1:
   gradcounts['parameters_with_gradient']=sum(p.numel() for p in model.net.parameters() if p.grad is not None);gradcounts['names_without_gradient']=[n for n,p in model.net.named_parameters() if p.requires_grad and p.grad is None]
  result=oldstep(self,*a,**kw);stepn[0]+=1
  if stepn[0]%16==0:print('TRAIN_STEP',stepn[0],flush=True)
  return result
 oldadam.step=step
 phase='training';condition='finetune';torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.time();trained_path,losses,valloss=train.train_seg(model.net,train_data=data['training'],train_labels=gt['training'],test_data=data['validation'],test_labels=gt['validation'],batch_size=1,learning_rate=1e-5,n_epochs=100,weight_decay=.1,save_path=str(M),save_every=100,rescale=False,scale_range=.5,bsize=256,min_train_masks=5,model_name='cpsam_v2_bbbc039_finetuned');torch.cuda.synchronize();end=time.time();S['training']={'seconds_including_flow_preprocessing_validation_saves':end-t,'checkpoint':str(Path(trained_path).relative_to(R)),'steps':stepn[0],'epochs':100,'train_images':16,'validation_images':8,'input_shapes':shapes,'gradient_observation':gradcounts,'losses':losses.tolist(),'validation_losses':valloss.tolist(),'validation_epochs':[i for i in range(100) if i==5 or i%10==0],'memory':mem(),**stats(t,end)};assert np.isfinite(losses).all();np.savez(D/'training_losses.npz',training=losses,validation=valloss)
 S['parameter_change_examples']={n:float((dict(model.net.named_parameters())[n].detach().float().cpu()-v).abs().max()) for n,v in before.items()};phase='saving';torch.save({'optimizer':optimizerbox[0].state_dict(),'epoch':100,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'note':'state saved; resume not tested; official train_seg schedule needs explicit continuation'},M/'optimizer_rng.pt');dump();del optimizerbox[:];torch.optim.AdamW=oldadam;oldadam.step=oldstep;model.net.forward=oldnet;train._process_train_test=oldprocess;gc.collect();torch.cuda.empty_cache()
 phase='evaluation';condition='finetuned_test';t=time.time();post=evaluate(model,data['test']);torch.cuda.synchronize();S['finetuned_test_seconds']=time.time()-t;score(post[0],'finetuned_accuracy');np.savez_compressed(D/'finetuned_test_masks.npz',**{str(i):m for i,m in enumerate(post[0])});np.save(D/'reload_reference_cellprob.npy',post[1][0][2]);np.save(D/'reload_reference_flows.npy',post[1][0][1]);S['checkpoint_sha256']=hashlib.file_digest(Path(trained_path).open('rb'),'sha256').hexdigest();S['status']='training_and_evaluation_complete_pending_independent_reload';dump()
except BaseException as e:
 S['error']=repr(e);S['status']='failed';dump();raise
finally:
 stop.set();th.join(timeout=5);(D/'forward_calls.json').write_text(json.dumps(forwardrows,indent=2));dump()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/brain/audit.py — 원문 전문 (3,345 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/brain/audit.py", "sha256": "0800b4eba9381146a96f8f3a83e7ba6fb13e4d16f8af2c46e58bf7621a27b9f4", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,hashlib
import h5py,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/work/data/brain_nucmm');manifest=json.loads((R/'download_manifest.json').read_text());rows=[];inside=[];outside=[];previews=[]
for split in ['train','val']:
 for path in sorted((R/'Image'/split).glob('*.h5')):
  lab=R/'Label'/split/path.name.replace('img_','seg_',1)
  with h5py.File(path,'r') as f:keys=list(f);assert keys==['main'];x=f['main'][:]
  with h5py.File(lab,'r') as f:y=f['main'][:]
  assert x.shape==y.shape and x.ndim==3 and np.isfinite(x).all() and np.isfinite(y).all() and y.min()==0 and np.issubdtype(y.dtype,np.integer);ids=np.unique(y);assert len(ids)>1
  prepared=R/'prepared'/split;prepared.mkdir(parents=True,exist_ok=True);np.save(prepared/(path.stem+'_image.npy'),x);np.save(prepared/(path.stem+'_label.npy'),y)
  row={'id':path.stem,'split':split,'image':str(path.relative_to(Path('/work'))),'label':str(lab.relative_to(Path('/work'))),'shape':list(x.shape),'image_dtype':str(x.dtype),'label_dtype':str(y.dtype),'image_min':float(x.min()),'image_max':float(x.max()),'nuclei_instances_in_3d_patch':int(len(ids)-1),'max_original_instance_id':int(y.max()),'image_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'mask_sha256':hashlib.file_digest(lab.open('rb'),'sha256').hexdigest(),'foreground_mean_intensity':float(x[y>0].mean()),'background_mean_intensity':float(x[y==0].mean())};rows.append(row)
  if split=='train':inside.append(row['foreground_mean_intensity']);outside.append(row['background_mean_intensity'])
  if len(previews)<2 or (split=='val' and len(previews)<3):previews.append((row['id'],x[x.shape[0]//2].copy(),y[y.shape[0]//2].copy()))
assert len(rows)==8 and len({r['id'] for r in rows})==8
invert=bool(np.mean(inside)<np.mean(outside));audit={'dataset':'NucMM-M public train/val patches, mouse visual cortex micro-CT','source_manifest':'data/brain_nucmm/download_manifest.json','actual_unique_3d_patches':len(rows),'actual_2d_slices':sum(r['shape'][0] for r in rows),'selected_train_volumes':4,'selected_validation_volumes':4,'independence':'all patches from same parent mouse brain volume; no independent donor/test cohort; pretrained training overlap not known','axes':'HDF5 arrays used as Z/Y/X for compute; anatomical orientation not verified','voxel_size_um':None,'modality_resolution':'primary project and original release text specify micro-CT, contradicting mirror README EM claim','polarity_policy':{'invert':invert,'selection_data':'train labels only; no val-based choice','train_foreground_mean':float(np.mean(inside)),'train_background_mean':float(np.mean(outside))},'rows':rows};(R/'input_audit.json').write_text(json.dumps(audit,indent=2))
fig,axes=plt.subplots(len(previews),2,figsize=(8,4*len(previews)))
for i,(name,x,y) in enumerate(previews):
 axes[i,0].imshow(x,cmap='gray');axes[i,0].set_title(name+' raw central slice');axes[i,1].imshow(x,cmap='gray');axes[i,1].contour(y>0,levels=[.5],colors='red',linewidths=.6);axes[i,1].set_title('original nuclear annotation');axes[i,0].axis('off');axes[i,1].axis('off')
fig.tight_layout();fig.savefig(R/'input_preview.png',dpi=120);print(json.dumps({'shapes':[r['shape'] for r in rows],'invert':invert,'train_fg':float(np.mean(inside)),'train_bg':float(np.mean(outside))},indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/brain/benchmark.py — 원문 전문 (6,940 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/brain/benchmark.py", "sha256": "bf1a7ffbbd62f210a25901f2a3fc63c2ba6ea95a41e886c33cc980856e13cf79", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,time,json,gc,hashlib
import numpy as np,torch
from cellpose import models,core,metrics
sys.path.insert(0,'/support');from gpu_support import Meter,write_json
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];audit=json.loads((R/'data/brain_nucmm/input_audit.json').read_text());meter=Meter(D);S={'status':'running','run_id':D.name,'input_audit':audit,'conditions':[],'settings':{'checkpoint':'cpsam_v2','precision':'BF16 model, native preprocessing/flows float32','seed':42,'tile_batch':8,'tile_size':256,'tile_overlap':.1,'augment':False,'rescale':None,'diameter':None,'normalize':True,'invert':audit['polarity_policy']['invert'],'inversion_selection':'training label intensity contrast only; validation not used for choice','flow_threshold':.4,'cellprob_threshold':0.,'min_size':15,'max_size_fraction':.4,'anisotropy':1.0,'anisotropy_note':'compute assumes equal voxel spacing; physical calibration not verified','training':'none, zero-shot inference with existing weights','validation':'original4val patches; same parent brain, no held-out donor or independent test'} }
frames=[];stage_times=[]
def save():write_json(D/'summary.json',S)
def data(row):
 P=R/'data/brain_nucmm/prepared'/row['split'];return np.load(P/(row['id']+'_image.npy')),np.load(P/(row['id']+'_label.npy'))
def relabel(x):
 ids,inv=np.unique(x,return_inverse=True);assert ids[0]==0;return inv.reshape(x.shape).astype(np.int32),ids.tolist()
def score(gt,pred):
 g,ids=relabel(gt);p,pids=relabel(pred);ap,tp,fp,fn=metrics.average_precision([g],[p],threshold=[.5,.75,.9]);return {'AP50_AP75_AP90':ap[0].tolist(),'TP':tp[0].tolist(),'FP':fp[0].tolist(),'FN':fn[0].tolist(),'gt_instances':len(ids)-1,'predicted_instances':len(pids)-1,'gt_original_label_ids':ids}
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='model_loading';t=time.time();model=models.CellposeModel(gpu=True,pretrained_model=str(R/'models/cpsam_v2'),use_bfloat16=True);torch.cuda.synchronize();S['model_loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['checkpoint_sha256']=hashlib.file_digest((R/'models/cpsam_v2').open('rb'),'sha256').hexdigest();S['parameters_total']=sum(p.numel() for p in model.net.parameters());S['parameters_trained_in_run']=0;S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'cellpose':__import__('importlib.metadata',fromlist=['version']).version('cellpose'),'gpu':torch.cuda.get_device_name(0)};save()
 oldforward=core._forward
 def forward(net,x):
  torch.cuda.synchronize();t=time.time();y=oldforward(net,x);torch.cuda.synchronize();frames.append({'phase':meter.phase,'shape':list(x.shape),'seconds':time.time()-t,'weights_dtype':str(net.dtype)});return y
 core._forward=forward
 for method in ['_run_net','_compute_masks']:
  original=getattr(model,method)
  def wrapper(*args,_method=method,_fn=original,**kw):
   torch.cuda.synchronize();t=time.time();res=_fn(*args,**kw);torch.cuda.synchronize();stage_times.append({'phase':meter.phase,'stage':_method,'seconds':time.time()-t});return res
  setattr(model,method,wrapper)
 opts={'batch_size':8,'bsize':256,'tile_overlap':.1,'augment':False,'normalize':{'normalize':True,'invert':audit['polarity_policy']['invert']},'diameter':None,'flow_threshold':.4,'cellprob_threshold':0.,'min_size':15,'max_size_fraction':.4}
 trainrow=next(r for r in audit['rows'] if r['split']=='train');x,y=data(trainrow);meter.phase='warmup_2d_train_slice';t=time.time();model.eval(x[x.shape[0]//2],do_3D=False,**opts);torch.cuda.synchronize();S['warmup_2d_seconds']=time.time()-t;S['warmup_note']='one train central slice only; first3D includes normal first-use setup';del x,y;save();vals=[r for r in audit['rows'] if r['split']=='val'];images=[];labels=[]
 for row in vals:
  x,y=data(row);images.append(x[x.shape[0]//2]);labels.append(y[y.shape[0]//2])
 startframes=len(frames);startstages=len(stage_times);t,base=meter.begin('2d_central_validation_slices');repeats=0
 while repeats<2 or time.time()-t<15:
  out=model.eval(images,do_3D=False,**opts);torch.cuda.synchronize();repeats+=1
 end=meter.end(t,base,name='2d_central_validation_slices',mode='2d',unique_images=4,repeats=repeats,processed_images=4*repeats);end['images_per_second']=4*repeats/end['seconds'];end['actual_forward_shapes']=sorted({str(f['shape']) for f in frames[startframes:]});end['processed_tiles']=sum(f['shape'][0] for f in frames[startframes:]);end['network_stage_seconds']=sum(z['seconds'] for z in stage_times[startstages:] if z['stage']=='_run_net');end['mask_stage_seconds']=sum(z['seconds'] for z in stage_times[startstages:] if z['stage']=='_compute_masks');end['scores']=[]
 for i,row in enumerate(vals):
  pred=out[0][i];assert pred.shape==images[i].shape and np.isfinite(pred).all();np.save(D/(row['id']+'_2d_mask.npy'),pred);np.save(D/(row['id']+'_2d_gt.npy'),labels[i]);np.save(D/(row['id']+'_2d_image.npy'),images[i]);np.savez_compressed(D/(row['id']+'_2d_flows.npz'),dP=out[1][i][1],cellprob=out[1][i][2]);assert np.array_equal(np.load(D/(row['id']+'_2d_mask.npy')),pred);end['scores'].append({'id':row['id'],**score(labels[i],pred)})
 S['conditions'].append(end);save();print('BRAIN_2D_COMPLETE',end['seconds'],flush=True)
 for row in vals:
  x,y=data(row);name='3d_'+row['id'];startframes=len(frames);startstages=len(stage_times);t,base=meter.begin(name);out=model.eval(x[...,None],channel_axis=3,z_axis=0,do_3D=True,anisotropy=1.,**opts);torch.cuda.synchronize();res=meter.end(t,base,name=name,mode='3d',unique_volumes=1,original_volume_shape=list(x.shape),original_slices=int(x.shape[0]),voxels=int(x.size),repeats=1);pred=out[0];assert pred.shape==x.shape and np.isfinite(pred).all();res['voxels_per_second']=x.size/res['seconds'];res['actual_forward_shapes']=sorted({str(f['shape']) for f in frames[startframes:]});res['processed_tiles']=sum(f['shape'][0] for f in frames[startframes:]);res['forward_calls']=len(frames[startframes:]);res['network_stage_seconds']=sum(z['seconds'] for z in stage_times[startstages:] if z['stage']=='_run_net');res['mask_stage_seconds']=sum(z['seconds'] for z in stage_times[startstages:] if z['stage']=='_compute_masks');res['scores']=[{'id':row['id'],**score(y,pred)}];np.save(D/(row['id']+'_3d_mask.npy'),pred);np.savez_compressed(D/(row['id']+'_3d_flows.npz'),dP=out[1][1],cellprob=out[1][2]);assert np.array_equal(np.load(D/(row['id']+'_3d_mask.npy')),pred);S['conditions'].append(res);save();print('BRAIN_3D_COMPLETE',row['id'],res['seconds'],res['scores'][0]['AP50_AP75_AP90'],flush=True);del x,y,out,pred;gc.collect();torch.cuda.empty_cache()
 S['status']='complete';S['saved_predictions_reloaded_exactly']=True;save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:write_json(D/'forward_calls.json',frames);write_json(D/'stage_times.json',stage_times);meter.close()
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "cellpose/results/brain_nucmm_20261029_110603_965442/benchmark_source.py", "source": "cellpose/scripts/brain/benchmark.py", "sha256": "bf1a7ffbbd62f210a25901f2a3fc63c2ba6ea95a41e886c33cc980856e13cf79"} -->

</details>

<details>
<summary>cellpose/scripts/brain/download.py — 원문 전문 (1,912 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/brain/download.py", "sha256": "db6a3d031b4a69c6d478c98d47005a8c4ebb6d86dba455332c89d203975e0f4d", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import urllib.request,json,hashlib,time,concurrent.futures
R=Path('/home/hyein/ai-benchmark/cellpose/data/brain_nucmm');repo='MedOtter/NucMM-M'
with urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo,timeout=30) as f:meta=json.load(f)
revision=meta['sha'];(R/'repository_metadata.json').write_text(json.dumps(meta,indent=2));files=[s['rfilename'] for s in meta['siblings'] if s['rfilename'].endswith(('.h5','.md','.txt'))]
def fetch(name):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);url='https://huggingface.co/datasets/'+repo+'/resolve/'+revision+'/'+name
 if not p.exists():
  for attempt in range(4):
   try:
    with urllib.request.urlopen(url,timeout=90) as r,p.with_suffix(p.suffix+'.part').open('wb') as f:
     for b in iter(lambda:r.read(1024*1024),b''):f.write(b)
    p.with_suffix(p.suffix+'.part').rename(p);break
   except Exception:
    if attempt==3:raise
    time.sleep(2)
 h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();print(name,p.stat().st_size,flush=True);return {'file':name,'url':url,'bytes':p.stat().st_size,'sha256':h}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(fetch,files))
(R/'download_manifest.json').write_text(json.dumps({'source':'public third-party mirror, pinned revision; original README_original.txt and primary NucMM project used for modality and release specification','repo':repo,'revision':revision,'files':out,'primary_project':'https://connectomics-bazaar.github.io/proj/nucMM/index.html','original_release_folder':'https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8','mirrored_readme_discrepancies':['README.md incorrectly calls mouse volume EM; primary paper and README_original.txt identify micro-CT','README.md says200cubed; original release says192cubed; actual h5 shape must be inspected']},indent=2));print('DOWNLOAD_COMPLETE',flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/brain/launch.py — 원문 전문 (1,424 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/brain/launch.py", "sha256": "6a129f4bbd1b4f9c7b53a8b296bfa8dbf587258701fd8380bec6cdea03366bbf", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import datetime,subprocess,json,shlex,sys,time,shutil
R=Path('/home/hyein/ai-benchmark/cellpose');ROOT=R.parent;run='brain_nucmm_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BRAIN_LATEST').write_text(run+'\n');assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'other GPU job';shutil.copy2(R/'scripts/brain/benchmark.py',D/'benchmark_source.py');shutil.copy2(R/'data/brain_nucmm/input_audit.json',D/'input_audit.json');cmd=['docker','run','--rm','--name',run,'--gpus','all','--ipc=host','--network','none','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','MPLCONFIGDIR=/work/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/work/.cache/numba','-e','CELLPOSE_LOCAL_MODELS_PATH=/work/models','nvcr.io/nvidia/pytorch:26.06-py3','/work/.venv/bin/python','-u','/work/scripts/brain/benchmark.py'];(D/'command.json').write_text(json.dumps(cmd,indent=2));print(run,flush=True);t=time.time()
with (D/'run.log').open('w') as f:code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
(D/'exit.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-t}));subprocess.run([sys.executable,str(R/'scripts/brain/report.py'),str(D)],check=True);print('BRAIN_FINISHED',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/brain/plot.py — 원문 전문 (1,194 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/brain/plot.py", "sha256": "e312391b7ab33c4d293fc352b41898d1cd88561c5d3daad6dbb496f696cec6f4", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,os
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
R=Path('/work');D=Path(os.environ['BRAIN_RUN_DIR']);S=json.loads((D/'summary.json').read_text());rows=[r for r in S['input_audit']['rows'] if r['split']=='val'];fig,axes=plt.subplots(4,4,figsize=(13,13))
for i,r in enumerate(rows):
 x=np.load(D/(r['id']+'_2d_image.npy'));gt=np.load(D/(r['id']+'_2d_gt.npy'));p2=np.load(D/(r['id']+'_2d_mask.npy'));p3path=D/(r['id']+'_3d_mask.npy');p3=np.load(p3path,mmap_mode='r')[r['shape'][0]//2] if p3path.exists() else None
 for j,(title,mask,color) in enumerate([('raw micro-CT',None,None),('GT nuclei',gt,'red'),('Cellpose 2D',p2,'lime'),('Cellpose 3D central slice',p3,'cyan')]):
  ax=axes[i,j];ax.imshow(x,cmap='gray')
  if mask is not None:
   boundary=find_boundaries(mask,mode='outer');overlay=np.zeros((*mask.shape,4));rgb=matplotlib.colors.to_rgb(color);overlay[boundary,:3]=rgb;overlay[boundary,3]=.9;ax.imshow(overlay)
  ax.set_title(title+('\n'+r['id'] if j==0 else ''),fontsize=9);ax.axis('off')
fig.tight_layout();fig.savefig(D/'brain_prediction_examples.png',dpi=140)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/brain/report.py — 원문 전문 (8,033 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/brain/report.py", "sha256": "f450e53ccb5d0f977deb9c331c02639f62e1328d085b4a3f4af18a5be87a7566", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,json,csv,subprocess,shlex
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'cellpose';D=Path(sys.argv[1]);S=json.loads((D/'summary.json').read_text());rows=S['input_audit']['rows'];out=[]
def a(x=''):out.append(x)
a('# Cellpose-SAM 추가 검증: 생쥐 뇌 조직의 세포핵');a();a('상태 **'+S['status']+'** · 실행 `'+D.name+'`');a();a('## 어떤 뇌 이미지인가?');a();a('NucMM-M의생쥐시각피질 **마이크로CT** 공개train/val패치다. 일반적인임상뇌MRI/CT의뇌영역·종양분할과제와다르며,과제는뇌조직안의**개별세포핵instance분할**이다. 기존U2OS형광이미지와촬영방식·입력구조가다르다.');a();a('원본연구프로젝트·README_original.txt와실제HDF5를대조했다. 배포미러README.md에는EM/200³라는잘못된설명이있지만원본설명은micro-CT/192³이고실제8파일shape도192³로확인됐다. 미러HF리비전5c489be7532664b14d90f4c028c6b2f9b6fcd782와파일별URL/SHA256·두README를data/brain_nucmm/download_manifest.json에보존했다. 원본GoogleDrive와byte동일성은독립적으로검증하지않았으며실제다운로드출처는공개미러임을명시한다.');a();a('|원본분할|고유3D패치|패치당shape|원본dtype|정답|');a('|---|---|---|---|---|');a('|train|4|192×192×192|uint8|uint16,핵별instance ID|');a('|val|4|192×192×192|uint8|uint16,핵별instance ID|');a();a('전체8volume은1536개원본슬라이스에해당하지만5000개독립뇌영상이아니다. 2D평가는val각volume중앙1장씩총4장,3D평가는val4volume전체총768슬라이스를사용한다. 모든패치가같은parent뇌volume에서나왔으며독립donor/test는아니다. 실제물리voxel크기·해부학적축방향은확인되지않았다. 계산시배열축Z/Y/X,anisotropy1을가정하므로물리보정된정확도주장으로해석하지않는다.');a();a('## 모델·전처리·검증 정책');a();a('기존Cellpose4.2.1 cpsam_v2 체크포인트를그대로사용했다. 전체304628930파라미터,학습갱신0,BF16가중치·공식FP32흐름/CPU전처리·TF32=False·seed42. 같은run의모델적재직후allocated/reserved·RSS·MemAvailable과SHA256은summary에있다. 추가파인튜닝은수행하지않았다.');a();a('tile256×256·tilebatch8·overlap0.1·augment=False·resize/diameter변경없음·공식백분위정규화·flow_threshold0.4·cellprob_threshold0·min_size15·max_size_fraction0.4를고정했다. 원본핵은배경보다어두워**train4패치의정답영역/배경평균만으로invert=True를선택**했다. 검증정답으로조건을선택하지않았다. train분할은가중치학습에쓰지않고이정책확인·2Dwarmup에만사용했다.');a();a('2D는중앙단면별정규화와최종mask생성을포함하고,3D는공식do_3D=True 경로로직교방향network추론후3D흐름/마스크를생성했다. 3D의flow_threshold동작은공식경로를따르며2D와같은기능이라고가정하지않는다. 원본192×192평면은network입력에맞게padding되며256×256타일입력과구분한다. 실제forwardshape·타일처리수는forward_calls.json과각행에보존했다.');a();a('모델 적재 직후 기준값: '+json.dumps(S.get('model_only_postload',{}),ensure_ascii=False));a();a('## 전체 segmentation 시간과 자원');a();a('시간은입력배열이CPU에준비된후정규화·타일링·전송·networkforward·재조립·**최종마스크후처리까지포함**한다. 다운로드/HDF5변환·모델로딩·입력검사·평가지표·파일저장은제외한다. 2D는train중앙단면warmup후15초이상반복했고,3D는volume당1회측정으로첫3D설정비용도포함한다. network·mask단계는CUDA동기화로따로계측했다. 기존후속타일forward시간과직접비교하지않는다.');a();a('|조건|고유입력|반복|총처리입력|입력/초|전체초|network초|mask후처리초|allocated peak GiB|reserved peak GiB|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|');a('|---|---|---|---|---|---|---|---|---|---|---|---|---|');flat=[]
for c in S['conditions']:
 m=c['memory'];r={'condition':c['name'],'unique_inputs':c.get('unique_images',c.get('unique_volumes')),'repeats':c['repeats'],'processed_inputs':c.get('processed_images',c.get('unique_volumes',1)*c['repeats']),'inputs_per_second':c.get('processed_images',c.get('unique_volumes',1)*c['repeats'])/c['seconds'],'total_seconds':c['seconds'],'network_seconds':c['network_stage_seconds'],'mask_seconds':c['mask_stage_seconds'],'allocated_peak_gib':m['peak_allocated_gib'],'reserved_peak_gib':m['peak_reserved_gib'],'gpu_avg':c['gpu_util_avg'],'gpu_max':c['gpu_util_max'],'rss_sampled_peak_gib':c['rss_sampled_peak_gib'],'available_sampled_min_gib':c['available_sampled_min_gib']};flat.append(r);a('|'+ '|'.join(str(round(v,4)) if isinstance(v,float) else str(v) for v in r.values())+'|')
a();a('GPU-Util은커널활동시간비율이며메모리점유율이아니다. CUDAallocated/reserved는프로세스allocator,전체121.688GiB는CPU/GPU공유RAM이다. RSS와CUDA를더하지않는다. 시스템가용량·RSS는0.5초대기+조회간격의sampled값이다. 가용8GiB미만/swap+2GiBguard를사용한다.')
a();a('## 정답 마스크 대비 평가');a();a('|조건/패치|GT핵수|예측핵수|AP50|AP75|AP90|');a('|---|---|---|---|---|---|')
for c in S['conditions']:
 for v in c['scores']:a('|'+ '|'.join(map(str,[c['mode']+'/'+v['id'],v['gt_instances'],v['predicted_instances']]+[round(x,4) for x in v['AP50_AP75_AP90']]))+'|')
a();a('3D평가는3D instance IoU,2D평가는해당단면의instance IoU를사용한다. GT의비연속ID는object를합치지않고연속재번호화했으며원래ID매핑을저장했다. AP는TP/(TP+FP+FN)인Cellpose instance평가지표다. pLDDT같은신뢰도나영상단위분류정확도가아니다. 평가4volume/같은parent뇌·외부사전학습데이터중복불확실성때문에임상/전체뇌일반화성능을입증하지않는다. 부정확한예측은정확히보고하고이미지예시와함께확인한다.');a();a('## 실제 입력·정답·예측 예시');a();a(f'![뇌조직 중앙단면의 원본·정답·2D예측·3D예측](../results/{D.name}/brain_prediction_examples.png)');a();a('## 보관 파일');a();a('원본HDF5·다운로드SHA/리비전·입력검사·정규화선택근거는data/brain_nucmm. 결과run에는원본코드snapshot·명령·exit·로그·GPU/RAM CSV·stage/forward계측·2D/3D마스크·flow/cellprob·원본GT ID·평가JSON·예시PNG를보존한다. 저장마스크는재로드해정확히같음을확인했다. 새모델을학습하지않아새학습체크포인트나학습재개를검증했다는주장은하지않는다.');a();a('[NucMM 원본 연구/데이터 설명](https://connectomics-bazaar.github.io/proj/nucMM/index.html) · [실제 배포 미러](https://huggingface.co/datasets/MedOtter/NucMM-M) · [Cellpose 공식 API](https://cellpose.readthedocs.io/en/latest/api.html)')
(R/'reports/Cellpose_brain_NucMM.md').write_text('\n'.join(out)+'\n')
if flat:
 with (R/'reports/brain_nucmm_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
cmd=['docker','run','--rm','--network','none','--user','1000:1000','-v',str(R)+':/work','-e','MPLCONFIGDIR=/work/.cache/matplotlib','-e','BRAIN_RUN_DIR=/work/results/'+D.name,'nvcr.io/nvidia/pytorch:26.06-py3','/work/.venv/bin/python','/work/scripts/brain/plot.py'];code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=subprocess.DEVNULL);(D/'plot_exit.json').write_text(json.dumps({'exit_code':code}))
for name in ['STATUS.md','RESULTS.md','FOLLOWUP_RESULTS.md']:
 with (ROOT/name).open('a') as f:f.write('\n뇌이미지 추가'+' Cellpose NucMM-M '+D.name+' 상태 '+S['status']+'. [보고서](cellpose/reports/Cellpose_brain_NucMM.md). 일반뇌MRI가아닌세포핵micro-CT·정답instance검증,기본/배치실험과별도run.\n')
print('BRAIN_REPORT',S['status'],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/download.py — 원문 전문 (1,687 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/download.py", "sha256": "d41375c0f1b41314cca8065c9d538d6529ac44a03675b83e054957ea8753e89d", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import urllib.request,ssl,json,hashlib,shutil,zipfile,datetime
R=Path('/home/hyein/ai-benchmark/cellpose');ctx=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt');rows=[]
with urllib.request.urlopen('https://huggingface.co/api/models/mouseland/cellpose-sam',context=ctx,timeout=60) as f:meta=json.load(f)
(R/'models/repository.json').write_text(json.dumps(meta,indent=2));revision=meta['sha']
files=[('data/'+x+'.zip','https://data.broadinstitute.org/bbbc/BBBC039/'+x+'.zip') for x in ['images','masks','metadata']]+[('models/cpsam_v2','https://huggingface.co/mouseland/cellpose-sam/resolve/'+revision+'/cpsam_v2')]
for name,url in files:
 p=R/name;print('DOWNLOAD',name,flush=True)
 if not p.exists():
  with urllib.request.urlopen(url,context=ctx,timeout=180) as src,p.with_name(p.name+'.part').open('wb') as dst:headers=dict(src.headers);shutil.copyfileobj(src,dst)
  p.with_name(p.name+'.part').rename(p)
 else:headers={}
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 rows.append(dict(path=name,url=url,revision=revision if 'models/' in name else 'BBBC039v1',bytes=p.stat().st_size,sha256=h.hexdigest(),headers=headers,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),license_note='BBBC039v1 CC0; model license notes preserved from official source'))
 (R/'data/sources.json').write_text(json.dumps(rows,indent=2))
 if p.suffix=='.zip':
  dest=R/'data'/p.stem;dest.mkdir(exist_ok=True)
  with zipfile.ZipFile(p) as z:
   for n in z.namelist():assert (dest/n).resolve().is_relative_to(dest.resolve())
   z.extractall(dest)
 print(name,p.stat().st_size,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/download_ranges.py — 원문 전문 (2,053 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/download_ranges.py", "sha256": "e8e12fd27fdc81583f69f09f8e73cc0140214f8d206e78588938ac41dd1bbf51", "encoding": "utf-8", "added_final_newline": false} -->
```python
import os,signal,time,json,urllib.request,hashlib,concurrent.futures
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/cellpose')
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try: args=(proc/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if len(args)>1 and args[1]==b'/home/hyein/ai-benchmark/cellpose/scripts/download.py':os.kill(int(proc.name),signal.SIGTERM)
time.sleep(1)
p=R/'models/cpsam_v2'; part=p.with_name(p.name+'.part'); start=part.stat().st_size; total=1233586851
rev=json.loads((R/'models/repository.json').read_text())['sha']; url=f'https://huggingface.co/mouseland/cellpose-sam/resolve/{rev}/cpsam_v2?download=true'
chunkdir=R/'models/download_chunks';chunkdir.mkdir(exist_ok=True); step=4*1024*1024
ranges=[(s,min(s+step,total)-1) for s in range(start,total,step)]
def get(r):
 a,b=r;dest=chunkdir/f'{a}_{b}'
 if dest.exists() and dest.stat().st_size==b-a+1:return dest
 for retry in range(5):
  try:
   req=urllib.request.Request(url+f'&range_start={a}',headers={'Range':f'bytes={a}-{b}'})
   with urllib.request.urlopen(req,timeout=90) as f:
    assert f.status==206 and f.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/')
    data=f.read();assert len(data)==b-a+1;dest.write_bytes(data)
   return dest
  except Exception:
   if retry==4:raise
 return dest
print('RESUME',start,'chunks',len(ranges),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
 for i,d in enumerate(pool.map(get,ranges)):
  if i%16==0:print('chunks_done',i+1,'/',len(ranges),flush=True)
with part.open('ab') as f:
 for a,b in ranges:f.write((chunkdir/f'{a}_{b}').read_bytes())
assert part.stat().st_size==total;part.rename(p)
h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();rows=json.loads((R/'data/sources.json').read_text()); rows.append({'path':'models/cpsam_v2','url':url,'revision':rev,'bytes':total,'sha256':h,'method':'resumed verified Content-Range chunks'});(R/'data/sources.json').write_text(json.dumps(rows,indent=2));print('COMPLETE',total,h,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/launch.py — 원문 전문 (1,042 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/launch.py", "sha256": "c51e34d1f8266b54455565b6225b61c3ad566af1e0c14d008bc9ae6737613fe1", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,os,datetime,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/cellpose');active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert not active,active
run='cellpose_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LATEST').write_text(run+'\n');args=['sg','docker','-c',f'docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v {R}:/work -e RUN_ID={run} -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba -e CELLPOSE_LOCAL_MODELS_PATH=/work/models nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/benchmark.py'];(D/'command.json').write_text(json.dumps(args,indent=2));print(run,flush=True)
with (D/'run.log').open('w') as f:code=subprocess.call(args,stdout=f,stderr=subprocess.STDOUT)
(D/'exit_code.json').write_text(json.dumps({'exit_code':code,'finished':datetime.datetime.now().isoformat()}));print('EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/prepare_inputs.py — 원문 전문 (2,561 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/prepare_inputs.py", "sha256": "fc95d36e4d480267e1581dfb7a22ad837005d9913b0cd74b3cf690e312e8dcd2", "encoding": "utf-8", "added_final_newline": false} -->
```python
import json, hashlib
from pathlib import Path
import numpy as np
import tifffile
from skimage import io, measure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/work'); out=R/'data/prepared'; out.mkdir(exist_ok=True)
rng=np.random.default_rng(42); records=[]; selected={}
for split,n in [('training',16),('validation',8),('test',16)]:
 names=(R/f'data/metadata/metadata/{split}.txt').read_text().splitlines()
 chosen=set(rng.choice(sorted(names),n,replace=False)); selected[split]=[]
 for name in names:
  stem=Path(name).stem; ip=R/f'data/images/images/{stem}.tif'; mp=R/f'data/masks/masks/{name}'
  x=tifffile.imread(ip); rgb=io.imread(mp); mask=measure.label(rgb[:,:,0],background=0).astype(np.uint16)
  assert x.shape==mask.shape and x.ndim==2 and np.isfinite(x).all(), (name,x.shape,mask.shape)
  rec={'id':stem,'split':split,'image':str(ip.relative_to(R)),'source_mask':str(mp.relative_to(R)),'shape':list(x.shape),'dtype':str(x.dtype),'min':int(x.min()),'max':int(x.max()),'instances':int(mask.max()),'image_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),'selected':name in chosen}
  if name in chosen:
   dest=out/f'{stem}_mask.npy'; np.save(dest,mask); rec['mask']=str(dest.relative_to(R)); selected[split].append(rec)
  records.append(rec)
assert len({r['id'] for r in records})==len(records)
(R/'data/input_audit.json').write_text(json.dumps({'all':records,'selected':selected,'seed':42,'pixel_size_um':None,'mask_decode':'skimage.measure.label(PNG first channel, background=0), official BBBC039 decoding','channels':'raw grayscale; official Cellpose conversion adds two zero channels','split_policy':'official original-image split, deterministic subset before crops'},indent=2))
a=selected['training'][0]; x=tifffile.imread(R/a['image']); m=np.load(R/a['mask']); fig,ax=plt.subplots(1,3,figsize=(13,4)); ax[0].imshow(x,cmap='gray',vmin=np.percentile(x,1),vmax=np.percentile(x,99)); ax[0].set_title(f'Raw {x.shape} {x.dtype}'); ax[1].imshow(m,cmap='nipy_spectral'); ax[1].set_title(f'Ground truth: {m.max()} nuclei'); ax[2].imshow(x,cmap='gray',vmin=np.percentile(x,1),vmax=np.percentile(x,99)); ax[2].contour(m>0,levels=[.5],colors='red',linewidths=.4); ax[2].set_title('Raw + ground-truth boundary'); [a.axis('off') for a in ax]; fig.tight_layout(); fig.savefig(R/'reports/input_example.png',dpi=150); plt.close(fig)
print(json.dumps({'total':len(records),'source_split_counts':{s:sum(r['split']==s for r in records) for s in selected},'selected':{s:len(v) for s,v in selected.items()},'example':a},indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/reload.py — 원문 전문 (1,590 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/reload.py", "sha256": "bd6883884c6fdf28df35379367163b07c5b50f00df9cff6348ebaa5a61b3ff38", "encoding": "utf-8", "added_final_newline": false} -->
```python
import os,json,time,hashlib
from pathlib import Path
import numpy as np,tifffile,torch
from cellpose import models
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());audit=json.loads((R/'data/input_audit.json').read_text());x=tifffile.imread(R/audit['selected']['test'][0]['image']);torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.manual_seed_all(42);t=time.time();m=models.CellposeModel(gpu=True,pretrained_model=str(R/s['training']['checkpoint']),use_bfloat16=True);torch.cuda.synchronize();load=time.time()-t
masks,flows,styles=m.eval([x],batch_size=8,normalize=True,diameter=None,rescale=None,flow_threshold=.4,cellprob_threshold=0.,min_size=15,max_size_fraction=.4,augment=False,tile_overlap=.1,bsize=256,do_3D=False)
a=np.load(D/'reload_reference_cellprob.npy');b=np.load(D/'reload_reference_flows.npy');ref=np.load(D/'finetuned_test_masks.npz')['0'];r={'checkpoint_sha256':hashlib.file_digest((R/s['training']['checkpoint']).open('rb'),'sha256').hexdigest(),'load_seconds':load,'cellprob_max_abs_difference':float(np.max(np.abs(flows[0][2]-a))),'flow_max_abs_difference':float(np.max(np.abs(flows[0][1]-b))),'mask_exact_match':bool(np.array_equal(ref,masks[0])),'independent_process':True,'shape':list(masks[0].shape)};r['passed']=r['cellprob_max_abs_difference']<=1e-5 and r['flow_max_abs_difference']<=1e-5 and r['mask_exact_match'];(D/'reload.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));assert r['passed'];s['status']='complete';s['reload']=r;(D/'summary.json').write_text(json.dumps(s,indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/report.py — 원문 전문 (14,017 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/report.py", "sha256": "1418a73e6c2424df48d44c6488993adb772aeccd29bfe8a6a14f6cfe93eb9d44", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,csv,datetime
import numpy as np,tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/work');run=(R/'results/LATEST').read_text().strip();D=R/'results'/run;s=json.loads((D/'summary.json').read_text());a=json.loads((R/'data/input_audit.json').read_text());src=json.loads((R/'data/sources.json').read_text());out=[]
def add(x=''):out.append(x)
def table(h,rs):
 add('| '+' | '.join(h)+' |');add('| '+' | '.join(['---']*len(h))+' |')
 for row in rs:add('| '+' | '.join(map(str,row))+' |')
 add()
def f(x,n=3):return 'N/A' if x is None else f'{x:.{n}f}'
add('# Cellpose-SAM: DGX Spark 실행 보고서');add();add(f'실행: `{run}`. 상태: **{s.get("status","running")}**. 원본 측정: [summary.json](../results/'+run+'/summary.json), [로그](../results/'+run+'/run.log), [자원 시계열](../results/'+run+'/resources.csv), [실제 forward 입력](../results/'+run+'/forward_calls.json).');add()
add('## 모델과 환경');add();p=s['parameters'];b=s['model_only_postload'];add(f'Cellpose {s["environment"]["cellpose"]}, 공식 기본 체크포인트 **cpsam_v2**, SAM ViT-L 기반 Cellpose-SAM이다. Cellpose3 cyto·CPDINO와 별개다. 전체 파라미터 {p["total"]:,}, requires_grad {p["requires_grad"]:,}. 이론적 BF16 파라미터 저장량 {f(p["theoretical_current_storage_gib"],6)} GiB.');add();checkpoint=next(r for r in src if r['path']=='models/cpsam_v2');add('체크포인트 revision: `'+checkpoint['revision']+'`; SHA-256: `'+checkpoint['sha256']+'`. 공식 파일 메타데이터 `../models/repository_files.json`의 LFS oid와 대조해 일치했다. 데이터/모델 출처 목록은 `../data/sources.json`에 보존했다.');add();add('환경: '+json.dumps(s['environment'],ensure_ascii=False));add();add('8 CPU threads, seed42. 공식 trainer는 epoch별 NumPy seed를 epoch 번호로, 검증 때42로 설정한다. GPU 작업은 한 번에 하나씩 실행했다. PyTorch NGC 26.06 ARM64 전용 venv이며 기존 환경은 보존했다. BF16 추론, 공식 학습 API가 자동 변환하는 FP32 파인튜닝이다. TF32 허용은 Cellpose 공식 코드 기본값이다.');add()
table(['적재 상태','CUDA allocated GiB','reserved GiB','RSS GiB','system available GiB'],[['모델 적재 직후, 첫 입력 이전',f(b['allocated_gib'],6),f(b['reserved_gib']),f(b['rss_gib']),f(b['system_available_gib'])]])
add(f'적재 시간 {f(s["loading_seconds"])}초; 적재 중 allocator peak {f(b["peak_allocated_gib"])}GiB. 이는 현재 적재량과 다르다. 첫 warmup {f(s["warmup_seconds"])}초는 초기 kernel/Numba 준비와 후처리를 포함하며 정상 반복 성능 표에서 제외했다.');add()
add('## 실제 데이터와 학습 입력');add();add('Broad BBBC039v1 공개 CC0 데이터: U2OS 세포의 Hoechst DNA 채널, 200장, uint16 흑백 TIFF 520×696. 공식 원본 이미지 분할 train100/validation50/test50에서 seed42로 16/8/16장을 선택했다. 원본 이미지 단계에서 분리한 뒤 학습 crop을 생성한다. 픽셀 크기(µm)는 확보한 메타데이터에서 확인되지 않아 N/A로 남긴다.');add();add('정답은 사람이 표시한 핵 instance PNG의 첫 채널을 공식 예제처럼 connected-component label로 변환했다. 0은 배경, 양의 정수는 개별 핵 ID이며 연속값 회귀 라벨이 아니다. 원본 영상·마스크 shape 일치, 유한값, 이미지 식별자 중복 및 split 분리를 확인했다. 전체200장에는 핵이 적은 영상도 있으나 선택한 학습16장은 모두 최소5개 기준을 충족한다.');add();add('![실제 학습 영상과 정답](input_example.png)');add();rs=[]
for split,rows in a['selected'].items():
 r=rows[0];rs.append([split,len(rows),r['id'],str(r['shape']),r['dtype'],r['instances'],f"{r['min']}…{r['max']}"])
table(['split','선택 장수','첫 원본 ID','shape','dtype','핵 수','픽셀 범위'],rs)
add('원본 2D grayscale은 실제 추론 forward에서 1채널로 유지된다(아래 실측 NCHW). 학습 API에서는 첫 채널 + 두 zero 채널로 변환된다. 1/99 percentile 정규화 후 float32 전처리, 추론 forward에서 BF16으로 변환한다. 학습은 영상과 flow label을 함께 random rotate/resize하여 실제 1×3×256×256 FP32 입력을 사용한다. 각 epoch마다 각 학습 이미지에서 crop을 하나 뽑는 정책이다.');add();add('실제 전체 선택 목록·파일 해시·마스크 경로: [input_audit.json](../data/input_audit.json). BBBC039는 DSB2018과 일부 겹치며 사전학습 Cellpose 데이터와의 중복 여부를 확정하지 못했다. 본 test는 이번 파인튜닝에서만 분리한 평가이며, 사전학습까지 완전히 독립적인 외부 성능이라고 주장하지 않는다.');add()
add('## 입력 규모에 따른 GPU 사용률과 메모리');add();add('공식 eval 기본 타일 배치 상한8, tile256, overlap0.1, augment=False, diameter/rescale 없음, flow_threshold0.4, cellprob_threshold0, min_size15, max_size_fraction0.4, 동일 사전학습 체크포인트를 고정했다. 영상별로 순차 처리하며, 실제 batch는 타일 수에 따라8보다 작을 수 있다. 개수 비교는 같은 크기의 중첩1/4/16장, 크기 비교는 같은4장의 중심128/256/512 crop이며 native520×696 비교는 count4 행이다. crop은 확대보간하지 않은 부분 시야다.');add();rows=[];export=[]
for r in s['conditions']:
 m=r['memory'];rows.append([r['name'],str(r['raw_shapes'][0]),r['unique_images'],r['repeats'],f(r['seconds']),f(r['images_per_second']),f(r['gpu_util_avg'],1)+' / '+f(r['gpu_util_max'],0),f(r['baseline']['allocated_gib']),f(m['peak_allocated_gib']),f(r['peak_delta_allocated_gib']),str(r['actual_forward_shapes'])]);export.append({'condition':r['name'],'unique_images':r['unique_images'],'repeats':r['repeats'],'seconds':r['seconds'],'images_per_second':r['images_per_second'],'gpu_util_avg':r['gpu_util_avg'],'gpu_util_max':r['gpu_util_max'],'baseline_allocated_gib':r['baseline']['allocated_gib'],'peak_allocated_gib':m['peak_allocated_gib'],'peak_delta_gib':r['peak_delta_allocated_gib'],'peak_reserved_gib':m['peak_reserved_gib'],'rss_sampled_peak_gib':r['rss_sampled_peak_gib'],'available_sampled_min_gib':r['available_sampled_min_gib'],'forward_seconds_including_transfers':r['forward_including_transfers_seconds'],'processed_tiles':r['processed_tiles'],'tiles_per_image':r['processed_tiles']/r['processed_images']})
table(['조건','원본 H×W','고유 장수','반복','초','장/초','GPU 평균/최대 %','시작 allocated GiB','peak GiB','같은 run 증가 GiB','실제 forward NCHW'],rows)
with (R/'reports/comparison.csv').open('w',newline='') as f0:
 w=csv.DictWriter(f0,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
add('실제 타일 수는 이미지당128crop=1,256crop=4,512crop=9,520×696원본=12다. 입력 padding과 overlap 때문에256crop도4타일이 된다. 공식 소스의 padding 후 타일 정책을 그대로 사용했다.');add();add('각 조건 최소15초 동안 완전한 eval 반복을 실행했다. 고유 장수와 반복 처리량을 구분한다. 측정 시간은 메모리에 읽어둔 입력의 정규화·타일링·전송·모델 forward·flow dynamics·마스크 후처리를 포함하며 원본 디스크 읽기와 결과 파일 저장은 제외한다. forward 별도 시간은 CPU↔GPU 전송과 반환 변환을 포함한다. CUDA 경계는 synchronize했다. 반복 구간의 CUDA peak는 조건마다 reset하고 empty_cache 뒤의 실제 baseline을 함께 기록한다.');add();add('원본 디스크 읽기·개별 결과 파일 저장 시간은 별도 계측하지 않아 N/A다. warmup은 초기 준비 시간을 포함하며 독립적인 compile 시간은 분리되지 않았다. 학습 총시간에는 공식 checkpoint 저장이 포함되지만 저장만의 시간은 N/A다.');add();add('GPU-Util은 nvidia-smi의 장치 활동 시간 비율이며 최대 FLOPS나 메모리 사용률이 아니다. 0.5초 대기+조회 소요의 시계열에서 조회 전체가 조건 안에 들어간 표본만 집계했다. 장치 전체에는 디스플레이 영향이 포함될 수 있다. RSS/available은 sampled 값이며 정확한 순간 peak가 아니다. 별도 CSV에 reserved peak와 sampled RSS/available, forward 시간도 저장했다.');add();mx=max(r['memory']['peak_allocated_gib'] for r in s['conditions']);u=max(r['gpu_util_max'] or 0 for r in s['conditions']);add(f'추론 최대 GPU-Util {u:.0f}%; 80% 관측 여부: {"관측됨" if u>=80 else "미관측"}. 추론 최대 allocated {mx:.3f}GiB는 전체 공유 RAM {b["system_total_gib"]:.3f}GiB의 {mx/b["system_total_gib"]*100:.2f}% 참고값이다. GPU 활동 시간 90%와 공유 RAM90% 사용은 다른 뜻이다. RSS와 CUDA 수치를 더하지 않는다. 이미지가 커져도 타일 입력은256×256으로 유지되며 타일 수·마지막 batch·후처리 비용이 변한다. 입력 개수 증가가 peak를 비례 증가시킨다고 해석할 수 없다.');add()
if 'training' in s:
 t=s['training'];add('## 정답 마스크 파인튜닝·검증·저장');add();add('공식 train_seg 경로, 전 학습 가능 구성의 AdamW(lr1e-5, weight_decay0.1), batch1, 100epochs, epoch당16장, rescale=False, scale_range0.5,256crop을 고정했다. 공식10epoch warmup과 마지막50epoch 감쇠를 사용했다. validation8장은 API의 test_data 인자에 넣었으며 최종 test16장은 학습에 넣지 않았다. 기본 min_train_masks5로 선택 학습 영상 모두 유지됐다.');add();add(f'완료 {t["steps"]} optimizer steps, 총 {t["seconds_including_flow_preprocessing_validation_saves"]:.2f}초(정답 flow 준비·검증·저장 포함). flow/정규화 준비 {s["training_preprocessing_seconds"]:.2f}초. 학습 진입 FP32 baseline {s["training_fp32_baseline"]["allocated_gib"]:.3f}GiB, fit peak {t["memory"]["peak_allocated_gib"]:.3f}GiB, 증가 {t["memory"]["peak_allocated_gib"]-s["training_fp32_baseline"]["allocated_gib"]:.3f}GiB. 전체 fit GPU 평균/최대 {f(t["gpu_util_avg"],1)}/{f(t["gpu_util_max"],0)}%. 이 평균에는 flow 준비와 검증·저장도 포함한다.');add();add('실제 gradient를 관찰한 파라미터: '+json.dumps(t['gradient_observation'],ensure_ascii=False));add();add('일부 patch embedding은 공식 forward가 .data를 사용하므로 requires_grad=True여도 gradient가 없다. 실제 변경 예시(최대 절댓값): '+json.dumps(s['parameter_change_examples']));add();add('저장 경로: `'+t['checkpoint']+'`. 공식 trainer의 저장 가중치는 FP32이고 평가 시 BF16으로 복원한다. 별도 optimizer/RNG/epoch 상태도 저장했으나 중단 후 학습 재개는 검증하지 않았다.');add();table(['동일 test16장','평균 AP@0.5','AP@0.75','AP@0.9'],[[name,*[f(x,4) for x in s[k]['mean_AP50_AP75_AP90']]] for name,k in [('사전학습','pretrained_accuracy'),('파인튜닝','finetuned_accuracy')]])
 add('AP는 Cellpose instance matching의 TP/(TP+FP+FN)을 이미지별 계산한 평균으로 COCO mAP와 다르다. 작은 파인튜닝 subset과100epochs의 관찰 결과이며 향상을 보장하지 않는다. epoch 선택에 test 지표를 사용하지 않았다.');add();add('독립 새 프로세스 재로드 결과: '+json.dumps(s.get('reload',{'status':'pending'}),ensure_ascii=False));add()
 x=tifffile.imread(R/a['selected']['test'][0]['image']);g=np.load(R/a['selected']['test'][0]['mask']);pre=np.load(D/'pretrained_test_masks.npz')['0'];post=np.load(D/'finetuned_test_masks.npz')['0'];fig,axs=plt.subplots(1,4,figsize=(16,4))
 for ax,v,title in zip(axs,[x,g,pre,post],['Raw held-out image','Ground truth','Pretrained','Fine-tuned']):ax.imshow(v,cmap='gray' if title.startswith('Raw') else 'nipy_spectral');ax.set_title(title);ax.axis('off')
 fig.tight_layout();fig.savefig(R/'reports/prediction_example.png',dpi=150);plt.close(fig);add('![평가 원본과 예측](prediction_example.png)');add();fig,ax=plt.subplots(figsize=(8,4));ax.plot(t['losses'],label='train');ve=t['validation_epochs'];ax.plot(ve,[t['validation_losses'][i] for i in ve],label='validation');ax.set_xlabel('Epoch (0-based)');ax.set_ylabel('Cellpose flow + cellprob loss');ax.legend();fig.tight_layout();fig.savefig(R/'reports/training_loss.png',dpi=150);plt.close(fig);add('![학습과 검증 loss](training_loss.png)');add()
add('## 재현과 한계');add();add('코드: `scripts/setup.py`, `prepare_inputs.py`, `benchmark.py`, `launch.py`, `reload.py`, `report.py`. 실행 명령과 로그는 각 run 폴더에 있다. 첫 실행의 결과 기록 함수 이름 충돌은 GPU 추론 이후 발생해 해당 실패 run을 보존하고 수정 후 새 run으로 재실행했다. 가중치 구간 다운로드 조립 크기 검증이 처음 실패해 기존 다운로드를 실제 종료하고 보존 prefix 길이에 맞춰 재조립했으며 최종 공식 SHA-256 일치로 검증했다.');add();add('데이터 선택·tile 정책을 고정한 단일 장비 단일 실행 관찰이다. GPU 연산 커널만의 순수 FLOPS, 다른 장비 대비 속도 향상, 광범위한 생물학적 일반화를 주장하지 않는다. 큰 이미지의 병목은 실제 타일 batch·메모리·시간을 바탕으로 해석하며, 확인하지 않은 원인은 추정으로 남긴다.');add();add('## 공식 출처');add();add('- [Cellpose 모델·tile 정책](https://cellpose.readthedocs.io/en/latest/models.html)\n- [공식 학습 안내](https://cellpose.readthedocs.io/en/latest/train.html)\n- [공식 소스](https://github.com/MouseLand/cellpose)\n- [공식 체크포인트](https://huggingface.co/mouseland/cellpose-sam)\n- [BBBC039 데이터 설명·분할](https://bbbc.broadinstitute.org/BBBC039)\n- [공식 BBBC039 mask decoding 예](https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075)');text='\n'.join(out)+'\n';(R/'reports/Cellpose.md').write_text(text);(R/'reports'/f'Cellpose_{run}.md').write_text(text);print(R/'reports/Cellpose.md')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>cellpose/scripts/setup.py — 원문 전문 (914 bytes)</summary>


<!-- RESTORE_FILE {"path": "cellpose/scripts/setup.py", "sha256": "60b6d7fa24925ede16a3e8d4f3d52b8d96a807e369475641e53ac88d2ccee005", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,subprocess,json,importlib.metadata,datetime
R=Path('/work');O=R/'results'/('setup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));O.mkdir();(R/'results/SETUP_LATEST').write_text(O.name+'\n');v={k:importlib.metadata.version(k) for k in ['torch','torchvision']};(R/'constraints.txt').write_text('\n'.join(k+'=='+x for k,x in v.items())+'\n');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True);cmd=['/work/.venv/bin/python','-m','pip','install','--cache-dir','/work/.cache/pip','-c','/work/constraints.txt','cellpose==4.2.1','psutil'];(O/'command.json').write_text(json.dumps(cmd,indent=2))
with (O/'install.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(O/'exit_code.json').write_text(json.dumps(dict(exit_code=p.returncode),indent=2));print('SETUP',O,p.returncode,flush=True);sys.exit(p.returncode)
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

