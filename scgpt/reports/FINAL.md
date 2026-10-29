# scGPT 최종 통합 보고서

## 1. 모델과 실제 입력

세포×유전자발현 행렬 → 세포별유전자ID·발현bin값 → 임베딩/세포유형. batch는세포행수. 최대1200토큰은CLS+유전자+padding자리.

실제파일·입력/정답예시·전처리와모델구조설명은이문서5절에전문을포함했다. 추론데이터와지도학습label을구분하며,세포/유전자·서열/토큰·이미지/타일의단위를섞지않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

**실제 공개 데이터로 실행했다.** 실제30만세포추론,실제전체pancreas학습,작은실제세포cache를반복한고정padding용량probe는서로다른과제/입력조건이다. **82.8%는용량probe에서만확인했고전체실데이터학습조건의성공으로표현하지않는다.**

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량|최대 allocated / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|실제30만세포 임베딩 추론|고유300000세포; 세포별실제유전자최대1199+CLS|8세포 × 최대1200토큰,FP32|1회전체처리|2663.19초|112.65세포/초|0.389 GiB / 0.32%|분류정확도평가대상아님; 전체출력검증; 80%미도달|
|실제30만세포 임베딩 추론|고유300000세포; 세포별실제유전자최대1199+CLS|256세포 × 최대1200토큰,FP32|1회전체처리|2628.60초|114.13세포/초|5.477 GiB / 4.50%|분류정확도평가대상아님; 전체출력검증; 80%미도달|
|실제 전체 pancreas 파인튜닝|train8480/val2120/test4218세포; 원본3000유전자→일치2999|8세포 × 배치내실제길이(최대1200),FP32|1epoch/1060step|202.88초|41.80세포/초|3.374 GiB / 2.77%|저장모델test Accuracy 0.9221(알려진라벨4213세포); 80%미도달|
|고정padding 학습 **용량 probe**|고유8세포; 같은batch반복총288건|8×1200자리,FP32; 평균nonpad 403.6|36step; 전체epoch학습아님|15.28초|18.85세포/초|3.390 GiB / 2.79%|독립정확도미평가; 80%미도달|
|고정padding 학습 **용량 probe**|고유284세포; 같은batch반복총852건|284×1200자리,FP32; 평균nonpad 307.0|3step; 전체epoch학습아님|42.03초|20.27세포/초|98.002 GiB / 80.54%|독립정확도미평가; 80%도달|
|고정padding 학습 **용량 probe**|고유292세포; 같은batch반복총876건|292×1200자리,FP32; 평균nonpad 306.5|3step; 전체epoch학습아님|42.92초|20.41세포/초|100.750 GiB / 82.79%|독립정확도미평가; 80%도달|

## 4. 결과 해석

- 같은30만세포입력에서batch8→256은메모리를0.389→5.477GiB로늘렸지만처리량은112.65→114.13세포/초로거의같았다. 메모리를더할당한다고학습/추론이그만큼빨라지는것은아니다.
- 고정1200자리학습probe의batch8→284→292에서3.390→98.002→100.750GiB로늘어80%를넘었다. 다만실제평균유효토큰은약306~404로padding을많이포함한다.
- 실제8480세포의1epoch학습은202.88초/1060step이다. 실제입력길이와학습량이달라고정padding3step과학습시간·정확도를직접비교하지않는다.
- 다음실제학습분석은동일데이터/split·모델·학습범위·길이·epoch를고정하고batch별시간·처리량·메모리·검증성능을함께확인해야한다. 현재큰batch의전체pancreas학습은guard중단기록이있다.

### 시간·성능 표를 읽는 주의점

30만세포batch8 시간은전날10만+재개20만구간의합이며야간중단제외. 임베딩추론은cache후배치/전송/forward/CPU반환/기록시간이다. 실제pancreas학습시간은토큰처리·전송·forward/backward/optimizer를포함하고validation/저장/test를제외한다. 용량probe시간은cache후batch구성·전송·forward/backward/optimizer와동기화를포함한다. 시간범위가서로다르므로원본행의범위를읽고비교한다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### scGPT 기존 완료 결과와 입력 규모·GPU 사용률 보완


보완run `supplement_20261027_211638_936991` · 상태 **complete** · [원본summary](../results/supplement_20261027_211638_936991/RECORDS.md#file-d2249d667745) · [자원시계열](../results/supplement_20261027_211638_936991/RECORDS.md#file-9e2c53c779d6)

#### 기존 기본 실험은 이미 완료

2026-10-23에입력검사·사전학습임베딩추론·1,000세포파인튜닝·저장/재로드·독립test·최대길이256/512/1024/1200비교를완료했다. 원본파일과학습모델을보존했고이번보완값을과거GPU-Util실측처럼소급해붙이지않는다.

[기존test](../results/finetuned_test_20261023_034345_99523/RECORDS.md#file-9cd02c9a4f70) · [기존길이비교](../results/length_compare_20261023_040313_501423/RECORDS.md#file-d50c252c8230)

기존저장모델test4218세포중학습라벨이있는4213세포의Accuracy 0.9380,학습13클래스macroF1 0.5350,weightedF1 0.9312.test-onlyMHCclassII5세포는해당지표에서제외했다. 기존1epoch파인튜닝CUDApeak3.3741GiB,저장모델test추론peak0.3811GiB다.

#### 동일 환경·모델·실제 입력

기존scgpt-spark컨테이너와whole-human pretrained checkpoint/args/vocab를그대로재사용했다. 이전컨테이너는sleep infinity상태로중지돼있어다시시작했으며기존실험을자동재실행하는명령은아니었다.

환경 {"torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "gpu": "NVIDIA GB10", "scgpt": "0.2.5"}

파라미터 {"total": 51856898, "encoder": 50278400}.추론구성전체와실제로쓰는encoder를구분한다. 원본checkpointSHA-256/소스snapshot/누락weight검사는보완run의setup/provenance.json·weight_loading.json에있다. nativeTransformer(use_fast_transformer=False),FP32,TF32=False,batch8,seed42,max_length1200(CLS포함)를유지했다.

| 첫forward전 | allocatedGiB | reservedGiB | RSSGiB | 시스템가용GiB |
| --- | --- | --- | --- | --- |
| 추론모델적재직후 | 0.1932 | 0.1992 | 1.1530 | 113.3687 |

FP32이론값은파라미터수×4byte이며실측과구분한다. 적재 0.3596초,입력준비 0.2015초,한번수행한token화cache 0.4650초,warmup 0.4634초다.

실제입력은TOSICA저자배포pancreas demo_train10600×3000/demo_test4218×3000 h5ad.사용가능vocab유전자2999개,Celltype은임베딩입력에넣지않는metadata다. raw/layers및정규화provenance가없어원시count라고단정하지않고제공값그대로51bin으로변환했다.

입력감사 {"cells": 4218, "source_genes": 3000, "matched_genes": 2999, "matrix_dtype": "float32", "tokens_min": 185, "tokens_mean": 598.9435751541015, "tokens_max": 1200, "raw_count_provenance": "not verified, values used as supplied, official51-bin preprocessing", "labels": "Celltype metadata not provided to embedding encoder"}

실제test세포ID·nonzero유전자수·원래행·Celltype은input_cells.tsv,첫세포원표현은raw_first_cell_expression.npy,실제geneIDs/binned값은tokenized_all.npz다. source/setup/selected_cells.tsv·matched_genes.tsv도저장했다. 모델입력은B×최대token수geneID(int64)와binvalue(float32),출력은L2정규화된CLS B×512embedding이다.

#### 개수·길이별 GPU 사용률과 메모리

| 조건 | 고유세포 | cap | 평균실제token | 반복 | 초 | 세포/초 | GPU평균/최대% | 시작GiB | peakGiB | 증가GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inference_n128_cap1200 | 128 | 1200 | 871.3516 | 21 | 15.6261 | 172.0194 | 91.4667/95.0000 | 0.2244 | 0.3673 | 0.1429 |
| inference_n1024_cap1200 | 1024 | 1200 | 708.1250 | 4 | 18.7142 | 218.8718 | 94.7778/95.0000 | 0.2244 | 0.3745 | 0.1501 |
| inference_n4218_cap1200 | 4218 | 1200 | 598.9436 | 1 | 16.3140 | 258.5510 | 94.2258/95.0000 | 0.2244 | 0.3802 | 0.1558 |
| inference_n128_cap256 | 128 | 256 | 256.0000 | 90 | 15.0944 | 763.1972 | 92.1724/94.0000 | 0.2244 | 0.2596 | 0.0352 |
| inference_n128_cap512 | 128 | 512 | 512.0000 | 41 | 15.2236 | 344.7286 | 93.7931/94.0000 | 0.2244 | 0.2948 | 0.0704 |
| inference_n128_cap1024 | 128 | 1024 | 858.9062 | 20 | 15.2164 | 168.2392 | 94.2759/95.0000 | 0.2244 | 0.3571 | 0.1326 |

개수는같은cached1200cap의중첩128/1024/4218세포다. 세포별실제nonzero유전자수분포가달라subset별평균token도달라지며이를표에기록했다. 길이는동일128세포의한번샘플링한1200tokencache앞256/512/1024개를사용해중첩유전자집합을만들었다. 과거길이실험의cap별독립randomsampling과정책이다르므로속도향상/정확도변화로직접비교하지않는다.

각추론조건최소15초반복,고유입력과반복처리를구분했다. 시간은cached배치구성·CPU→GPU·encoder·CLS정규화·GPU→CPU반환을포함하며원본h5ad읽기·token화·모델적재·warmup·저장은제외했다. 각조건CUDApeak reset/empty_cache/synchronize,같은run실제시작allocated대비delta다. GPU-Util은활동시간비율로FLOPS나메모리%가아니다.0.5초대기+조회가조건안에포함된표본만집계했다.

#### 학습 GPU 사용률 보완

| 조건 | train세포 | epochs | batch | 초 | GPU평균/최대% | 모델적재GiB | 학습peakGiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 누락사용률계측보완 | 1000 | 1 | 8 | 24.1723 | 88.6887/95.0000 | 0.1913 | 3.3741 |

GPU사용률이없었던기존학습조건을계측하기위해동일1000세포/1epoch를별도run에서실행했다. 원본모델/결과는유지했고새모델은별도경로에저장했다. 원본02_finetune.py복사본에GPU-Util시계열과적재baseline만추가해학습알고리즘·split·optimizer를유지했다.

전체train10600에서동일seed42cell단위validation2120,별도train1000을골랐으며13유형정답으로encoder+classifier를갱신했다. 저장모델bundle /workspace/scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001.새보완모델은validation과재로드를검증했고독립test를새로평가한것은아니다. 위기존testAccuracy와새모델을혼동하지않는다.

학습sampler는0.2초대기+조회시간과시작/끝표본을포함하므로첫/끝nvidia-smi관측창이주변활동을포함할수있다. resource CSV는training_util/epoch_001/training_resources.csv다. GPU메모리는fit구간allocatorpeak,validation/저장은시간에서제외했다. optimizer/RNG/epoch저장,encoder/head변경,원본strict reload검증을수행했다. 독립프로세스validation재로드 {"independent_process": true, "validation_cells": 2120, "max_abs_difference": 0.0, "predictions_match": true, "passed": true}

#### 해석과 한계

Spark는공유RAM121.688GiB이므로RSS와CUDA를더하지않는다. allocated는사용tensor, reserved는cache포함allocator예약이며장치전체점유율과다르다. rawcount유래·donor독립검증은확인되지않았다. 개수증가가peak를비례증가시킨다고가정하지않으며batch8이고정돼메모리증가는작았다. 메모리90%도달은별도승인된후속batch확대실험에서검토한다.

공식소스: [scGPT](https://github.com/bowang-lab/scGPT). 정확한설치리비전·원본모델hash와코드는각기존/보완run의provenance와environment에보존했다.

### 3. 후속 실험 통합 — 학습 배치 확대

### scGPT 후속: batch 확대와 실제 메모리 용량


#### 입력을 읽는 법: 세포 행렬, 배치, 유전자 토큰

원본 발현 행렬의 구조는 **세포 × 유전자**다. 행 하나가 세포 하나, 열 하나가 유전자 하나이며, 교차하는 값이 그 세포에서 해당 유전자의 발현값이다. 따라서 **batch292는 세포292행을 한 번에 처리한다는 뜻**이다. 유전자292열을 선택한다는 뜻이 아니다.

다음은 설명용 예시이며 실제 측정 데이터의 값은 아니다.

| 세포(행) | 유전자 A(열) | 유전자 B(열) | 유전자 C(열) |
|---|---|---|---|
| 세포1 | 0 | 5 | 12 |
| 세포2 | 7 | 0 | 3 |

scGPT 전처리는 세포별 발현이 있는 유전자를 선택하고, 각 유전자의 ID에 변환한 발현값을 짝지어 입력한다. 이번 실험은 공식51-bin 전처리를 사용했다. **일반적인 유전자 토큰 하나는 유전자 하나를 나타내며, 발현값은 같은 위치에 대응하는 별도 값**이다. ID와 발현값을 각각 두 토큰으로 세지 않는다.

설명용 세포1의 입력은 `[CLS] [유전자 B + 변환한 발현값] [유전자 C + 변환한 발현값] …`처럼 생각하면 된다. CLS는 세포 전체의 표현을 모으는 특별 토큰이고, padding은 빈자리를 채우는 토큰이다. CLS 자체에 붙는 입력 값은 실제 유전자 발현값이 아니다.

**최대1200토큰 = CLS 1자리 + 유전자 최대1199자리**다. 실제 유전자가 적으면 남은 자리를 padding으로 채우고, 많으면 설정된 sampling 정책에 따라 일부 유전자를 선택한다. 따라서 모든 세포에 실제 유전자1200개를 넣었다는 뜻이 아니다. 토큰 위치는 원본 행렬의 고정된 유전자 열 번호와 다르며 세포별 유전자 ID로 구분한다.

GPU에는 유전자 ID 배열과 대응 발현값 배열이 들어간다. batch292의 고정 길이 학습 용량 실험에서는 두 배열의 입력 shape가 각각 **292세포 × 1200토큰 자리**였다. 그 뒤 모델 내부에서 위치별 벡터, attention 중간값 등이 생성되며 학습에서는 gradient와 optimizer 상태도 메모리를 사용한다. 입력 배열292×1200의 크기만으로 전체 메모리를 계산할 수 없다.

##### 왜 전체 세포 수보다 배치가 순간 메모리에 영향을 주는가?

전체30만세포를 한꺼번에 GPU에 올리지 않고, `세포8개 전송 → 계산 → 결과를 CPU/파일에 저장 → 다음8개`처럼 나눠 처리한다. 배치와 실제 토큰 길이 등 조건이 같으면 전체 세포 수 증가는 주로 처리 횟수·총시간·출력 파일 크기를 늘린다. 순간 GPU 메모리가 세포 수에 비례해 증가하지는 않는다. 세포별 길이 분포가 달라지면 dynamic padding을 통해 메모리가 달라질 수 있다.

##### 추론 결과와 학습 용량 결과를 구분하기

| 실험 | 배치(세포 수) | 입력 길이 정책 | CUDA allocated peak | 전체 공유RAM121.688GiB 대비 |
|---|---|---|---|---|
| 실제30만 고유세포 임베딩 추론 | 8 | 최대1200, 배치 내 실제 최대길이로 padding | 0.389GiB | 약0.32% |
| 같은 실제30만 고유세포 임베딩 추론 | 256 | 위와 동일한 cache·길이 정책 | 5.477GiB | 약4.50% |
| 별도 학습 용량 probe | 292 | 모든 입력을1200자리로 고정 padding | 100.750GiB | 약82.79% |

**82.79%는 batch292의 고정 padding 학습 용량 실험에서 관측한 값**이다. batch256 실제데이터 추론에서80%를 사용한 것이 아니다. 서로 다른 학습 상태·padding 정책·데이터 조건이므로 위 행 사이의 차이를 배치만의 효과로 해석하지 않는다. 학습 용량 probe는 실제1000학습세포 cache의 부분집합을 측정 step 동안 반복한 실험이며30만세포 학습 실험이 아니다.

GPU 연산 사용률(예:95%)은 GPU 커널이 활동한 시간 비율이다. 메모리 비율은 여기서 CUDA allocated peak를 CPU/GPU 공유RAM121.688GiB로 나눈 참고값이다. 둘은 다른 지표이며 별도 전용VRAM 점유율로 해석하지 않는다. reserved는 allocator가 확보한 예약 공간으로 allocated와 구분한다.

기본10종 완료 이후 별도 승인으로 실행한 **학습 자원용량 실험**이다. FP32·최대1200token·동일모델·전처리·optimizer를 고정하고 batch를 확대했다. 모델정확도 개선 실험이 아니다.

관측 최대 **batch292: CUDA allocated peak 100.750GiB / 공유RAM 121.688GiB = 82.79%**. 80% 초과: True; 90% 도달: False.

80% 기준 97.350GiB,90% 기준 109.519GiB다. 공유메모리에서이비율은전용VRAM점유율이나시스템전체사용률과같지않다. 최대성공batch는시험범위중최대이며이론적한계가아니다.

| batch | 고유세포 | 실제입력shape | step/반복 | 초 | 세포/초 | allocatedpeakGiB | reservedpeakGiB | 할당/공유RAM% | GPU평균/최대% | 가용RAM sampled최소GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 8 | [8, 1200] | 36 | 15.28 | 18.85 | 3.390 | 3.922 | 2.79 | 89.5/95 | 108.295 |
| 16 | 16 | [16, 1200] | 19 | 15.20 | 20.00 | 6.153 | 6.562 | 5.06 | 89.1/95 | 105.676 |
| 32 | 32 | [32, 1200] | 10 | 15.51 | 20.63 | 11.704 | 11.812 | 9.62 | 92.1/95 | 100.381 |
| 48 | 48 | [48, 1200] | 7 | 15.93 | 21.09 | 17.097 | 17.379 | 14.05 | 91.7/95 | 94.677 |
| 64 | 64 | [64, 1200] | 5 | 15.15 | 21.13 | 22.574 | 22.695 | 18.55 | 90.8/95 | 89.274 |
| 96 | 96 | [96, 1200] | 4 | 18.15 | 21.16 | 33.629 | 33.889 | 27.64 | 93.7/95 | 77.938 |
| 128 | 128 | [128, 1200] | 3 | 18.25 | 21.05 | 44.513 | 44.598 | 36.58 | 93.7/95 | 66.771 |
| 160 | 160 | [160, 1200] | 3 | 22.81 | 21.04 | 55.566 | 55.736 | 45.66 | 91.9/95 | 55.690 |
| 192 | 192 | [192, 1200] | 3 | 27.43 | 21.00 | 66.436 | 66.689 | 54.60 | 94.2/96 | 44.629 |
| 224 | 224 | [224, 1200] | 3 | 32.31 | 20.80 | 77.510 | 77.828 | 63.70 | 93.6/96 | 32.619 |
| 256 | 256 | [256, 1200] | 3 | 36.64 | 20.96 | 88.375 | 88.783 | 72.62 | 93.3/96 | 21.983 |
| 278 | 278 | [278, 1200] | 3 | 40.28 | 20.70 | 95.947 | 96.430 | 78.85 | 93.7/96 | 14.050 |
| 284 | 284 | [284, 1200] | 3 | 42.03 | 20.27 | 98.002 | 98.465 | 80.54 | 92.3/95 | 11.926 |
| 292 | 292 | [292, 1200] | 3 | 42.92 | 20.41 | 100.750 | 101.738 | 82.79 | 94.3/96 | 8.863 |

batch8 대비 최대메모리 조건의처리량은약1.08배다. 메모리는 3.39→100.75GiB로늘었다. 중간batch에서이미처리량이비슷하면더큰batch의추가메모리효용은작다.

![batch별메모리와처리량](https://raw.githubusercontent.com/fff0996/Bioinformatics_AImodel/49661373a7b790bbbbcce0d7ef8bf96b8ce83dbd/scgpt/reports/memory_batch_curve.png)

#### 정확한 입력·학습·시간 범위

기존실제train1000세포의학습token화cache(gene IDs,expression bins,13개세포유형정답)를사용했다. 한batch내중복세포는없으며firstB중첩부분집합을같은조건에서여러step반복한다. 각condition은최소3step/15초이고실제step수를표에썼다. 작은batch는더많은측정step을반복하므로다른학습량의정확도비교로쓰지않는다.

각batch는동일pretrainedencoder·seed42새classifier·새AdamW에서시작한다. FP32/TF32=False,AdamWlr1e-4/wd.01/betas.9,.999/eps1e-8/gradientclip1.0을고정했다. 원본51bin·CLS전처리이며가중치이론량·실측적재량과실행중activation/gradient/optimizer증가를구분한다. 각run summary의parameters와model_only_postload에있다.

**이용량실험은모든batch를1200열까지padding한cache를그대로사용한다.** 원본일반학습은batch내최대실제길이에맞춘dynamicpadding을사용하므로평균시간을원본실험과직접비교하지않는다. 실제nonpad token평균은약300대이며,나머지padding도dense학습의메모리/계산비용을발생시킨다. 따라서90%에가까운할당이모두유효한유전자계산량이라는뜻은아니다. 일관된1200길이조건의capacity상한조사다.

시간에는cached batch구성·CPU→GPU·forward/classifier/loss·backward·gradientclip·optimizer와step경계CUDA동기화를포함한다. imports·모델적재·H5읽기·token화·probe가중치저장은제외했다. 각조건allocatorpeak는reset하고현재baseline도별도로기록했다. GPU-Util은활동시간비율로FLOPS/메모리%가아니다.

원본기본결과는보존했다. capacity probe후가중치는각run/probe_models/batch_B.pt에별도저장했으며모델초기화는initial_state.pt,구조는model_config.json이다. 이probe가중치를검증된생물학적최종모델로채택하지않았고,학습재개/정확도평가를주장하지않는다.

#### 중단·측정 한계

시스템MemAvailable8GiB미만또는swap+2GiB에서자체프로세스를중단한다. 다음batch예상reserved가현재예약량+가용량−8GiB예산을넘으면확대전에중단하거나예상안전batch로축소한다. 첫coarse sweep은8%예측여유,80%경계refine은이전실측에대해2%여유를적용했다. 중단은90%불가능증명이나모델절대최대가아니다.

- `memory_batch_20261027_213310_982032`: complete / tested_predicted_safe_cap
- `memory_batch_refine_20261027_214026_294469`: complete / available_unique_input_limit

reserved는cache/단편화포함이며실제살아있는tensor할당과다르다. nvidia-smi는SparkVRAM용량을제공하지않으므로PyTorchcounter와전체공유RAM·시스템가용량을구분했다. RSS와CUDA는더하지않는다. RSS/가용량은0.5초대기+조회시계열sampled값이며순간최저를보장하지않는다.

#### 원본 경로

- [run memory_batch_20261027_213310_982032](../results/memory_batch_20261027_213310_982032/RECORDS.md#file-07bb42ecb3c2) · 로그 (로컬 보관: `../results/memory_batch_20261027_213310_982032/run.log`) · [시계열](../results/memory_batch_20261027_213310_982032/RECORDS.md#file-0c5cfa466baf)
- [run memory_batch_refine_20261027_214026_294469](../results/memory_batch_refine_20261027_214026_294469/RECORDS.md#file-e3eae8ef5503) · 로그 (로컬 보관: `../results/memory_batch_refine_20261027_214026_294469/run.log`) · [시계열](../results/memory_batch_refine_20261027_214026_294469/RECORDS.md#file-fa362b498035)

[기본 scGPT 보고서](scGPT.md) · [기존모델·공식소스](https://github.com/bowang-lab/scGPT)

### 4. 후속 실험 통합 — 실제 전체세포 학습

### scGPT 실제 pancreas 전체 데이터 처리


실행: `realdata_full_20261027_215022_050816`. 총 14,818개 실제 세포: 학습 8,480 / validation 2,120 / 독립 test 4,218. FP32, batch8, seed42, 최대1,200토큰, 원래 dynamic padding, 1epoch. 입력 ID 중복·파일 간 겹침 없음은 source_data_audit.json에서 확인했다. 원시 count provenance는 확인되지 않았다.

|조건|학습시간|allocated peak|GPU 평균/최대|독립 test Accuracy|
|---|---|---|---|---|
|batch8|202.88초|3.374 GiB|93.02/95%|0.92215|

모델 저장 및 validation strict reload 최대 logit 오차0. 별도 프로세스에서 test4,218세포를 예측했고, 학습 라벨에 속한4,213세포로 정확도를 계산했다. test-only5세포는 알려진 클래스 지표에서 제외했다. macroF1(학습13클래스) 0.52963, weightedF1 0.91658. 학습1epoch/1060 optimizer updates이며 donor-independent 검증은 아니다.

batch284·256·224 학습은 시스템 가용 RAM8GiB guard가 발동해 중단됐다. 완료된 학습·배치 비교 결과로 쓰지 않는다. 각각 실패 로그 및 memory_guard.json을 보존했다. 샘플링 감지 지연으로224 조건은 가용4.424GiB에서 중단됐으며, 안전한 최대 배치가 검증됐다는 의미가 아니다. 추가 대형 학습 배치 재시도 대신 사용자 우선 요청인 실제30만세포 임베딩 추론을 진행한다.

[원본 결과](../results/realdata_full_20261027_215022_050816/RECORDS.md#file-def4ed1b1494) · [실행·입력·저장 모델 코드](FINAL.md)

CUDA allocated와 reserved는 프로세스 allocator 값이다. Spark 공유 RAM에서 RSS와 CUDA를 독립 물리 메모리처럼 더하지 않는다. GPU-Util은 메모리 점유율이 아니다. 학습시간은 전처리/전송/forward/backward/optimizer를 포함하며 validation·모델저장은 제외한다.

### 5. 후속 실험 통합 — 실제30만세포 추론

### scGPT 실제30만 세포 임베딩 및 자원 측정


#### 입력을 읽는 법: 세포 행렬, 배치, 유전자 토큰

원본 발현 행렬의 구조는 **세포 × 유전자**다. 행 하나가 세포 하나, 열 하나가 유전자 하나이며, 교차하는 값이 그 세포에서 해당 유전자의 발현값이다. 따라서 **batch292는 세포292행을 한 번에 처리한다는 뜻**이다. 유전자292열을 선택한다는 뜻이 아니다.

다음은 설명용 예시이며 실제 측정 데이터의 값은 아니다.

| 세포(행) | 유전자 A(열) | 유전자 B(열) | 유전자 C(열) |
|---|---|---|---|
| 세포1 | 0 | 5 | 12 |
| 세포2 | 7 | 0 | 3 |

scGPT 전처리는 세포별 발현이 있는 유전자를 선택하고, 각 유전자의 ID에 변환한 발현값을 짝지어 입력한다. 이번 실험은 공식51-bin 전처리를 사용했다. **일반적인 유전자 토큰 하나는 유전자 하나를 나타내며, 발현값은 같은 위치에 대응하는 별도 값**이다. ID와 발현값을 각각 두 토큰으로 세지 않는다.

설명용 세포1의 입력은 `[CLS] [유전자 B + 변환한 발현값] [유전자 C + 변환한 발현값] …`처럼 생각하면 된다. CLS는 세포 전체의 표현을 모으는 특별 토큰이고, padding은 빈자리를 채우는 토큰이다. CLS 자체에 붙는 입력 값은 실제 유전자 발현값이 아니다.

**최대1200토큰 = CLS 1자리 + 유전자 최대1199자리**다. 실제 유전자가 적으면 남은 자리를 padding으로 채우고, 많으면 설정된 sampling 정책에 따라 일부 유전자를 선택한다. 따라서 모든 세포에 실제 유전자1200개를 넣었다는 뜻이 아니다. 토큰 위치는 원본 행렬의 고정된 유전자 열 번호와 다르며 세포별 유전자 ID로 구분한다.

GPU에는 유전자 ID 배열과 대응 발현값 배열이 들어간다. batch292의 고정 길이 학습 용량 실험에서는 두 배열의 입력 shape가 각각 **292세포 × 1200토큰 자리**였다. 그 뒤 모델 내부에서 위치별 벡터, attention 중간값 등이 생성되며 학습에서는 gradient와 optimizer 상태도 메모리를 사용한다. 입력 배열292×1200의 크기만으로 전체 메모리를 계산할 수 없다.

##### 왜 전체 세포 수보다 배치가 순간 메모리에 영향을 주는가?

전체30만세포를 한꺼번에 GPU에 올리지 않고, `세포8개 전송 → 계산 → 결과를 CPU/파일에 저장 → 다음8개`처럼 나눠 처리한다. 배치와 실제 토큰 길이 등 조건이 같으면 전체 세포 수 증가는 주로 처리 횟수·총시간·출력 파일 크기를 늘린다. 순간 GPU 메모리가 세포 수에 비례해 증가하지는 않는다. 세포별 길이 분포가 달라지면 dynamic padding을 통해 메모리가 달라질 수 있다.

##### 추론 결과와 학습 용량 결과를 구분하기

| 실험 | 배치(세포 수) | 입력 길이 정책 | CUDA allocated peak | 전체 공유RAM121.688GiB 대비 |
|---|---|---|---|---|
| 실제30만 고유세포 임베딩 추론 | 8 | 최대1200, 배치 내 실제 최대길이로 padding | 0.389GiB | 약0.32% |
| 같은 실제30만 고유세포 임베딩 추론 | 256 | 위와 동일한 cache·길이 정책 | 5.477GiB | 약4.50% |
| 별도 학습 용량 probe | 292 | 모든 입력을1200자리로 고정 padding | 100.750GiB | 약82.79% |

**82.79%는 batch292의 고정 padding 학습 용량 실험에서 관측한 값**이다. batch256 실제데이터 추론에서80%를 사용한 것이 아니다. 서로 다른 학습 상태·padding 정책·데이터 조건이므로 위 행 사이의 차이를 배치만의 효과로 해석하지 않는다. 학습 용량 probe는 실제1000학습세포 cache의 부분집합을 측정 step 동안 반복한 실험이며30만세포 학습 실험이 아니다.

GPU 연산 사용률(예:95%)은 GPU 커널이 활동한 시간 비율이다. 메모리 비율은 여기서 CUDA allocated peak를 CPU/GPU 공유RAM121.688GiB로 나눈 참고값이다. 둘은 다른 지표이며 별도 전용VRAM 점유율로 해석하지 않는다. reserved는 allocator가 확보한 예약 공간으로 allocated와 구분한다.

상태: **complete**. 실행 `actual_300k_resume_20261028_093701_696230`. [원본 summary](../results/actual_300k_resume_20261028_093701_696230/RECORDS.md#file-7ffcf339190d) · 입력 검사 (로컬 보관: `../data_external/tabula_sapiens/prepared_300k_seed42/input_audit.json`)

#### 실제 입력과 전처리

Tabula Sapiens 고정 Census2024-07-01 원본 483,152세포 × 58,482 features에서 seed42로 **300,000개 고유 실제 세포**를 선택했다. 반복 복제한 입력이 아니다. source raw.X를 사용하며 vocabulary와 일치한 유전자 40,474개를 처리한다. 원본 파일 SHA256은 download_source.json에 저장했다. raw.X의 의미와 count provenance는 원본 metadata와 구분해 확인하며, 정수형 값 표본만으로 원시 count를 확정하지 않는다.

공식51-bin 전처리, 비영 유전자, 최대1200토큰(CLS포함). 선택·유전자 sampling·binning 결과는 한번 cache해 두 배치 조건에 동일하게 제공한다. 실제 GPU 입력은 배치 내 최대 유효길이로 padding한다. 1만/10만/30만은 동일 순서의 중첩 부분집합이며, 각 batch의30만세포 단일 pass에서 누적 지점을 측정한다. 독립적인3회 실행이 아니다.

|고유 세포|평균 실제 토큰|최소/최대 토큰|길이 제한으로 sampling된 세포|
|---|---|---|---|
|10000|1159.37|189/1200|8960|
|100000|1160.36|180/1200|89708|
|300000|1160.89|176/1200|269172|

tissue·assay·cell_type 분포와 실제 cell ID/원본 행은 input_audit.json 및 cells.tsv에 보존했다. cell_type은 metadata이며 이번 임베딩 추론의 정답 분류 평가가 아니다.

재개 실행인 경우 기존 완료 지점의 출력을 재사용하고 나머지만 새 구간으로 측정한다. 해당 행의 시간/처리량은 이번 구간 처리 세포 수 기준이며 전체 누적 세포 처리 시간으로 해석하지 않는다.

#### 모델과 측정

whole-human pretrained scGPT, 구성 파라미터 51,856,898, FP32, TF32=False, seed42. 동일 모델을 유지하며 batch8→256 순차 실행한다. classifier fine-tuning은 수행하지 않는다. pretrained encoder/value encoder/transformer의 필수 가중치 이름·shape를 검증해 적재했다. 모델 출처와 가중치 파일은 기존 기본 보고서 및 models/whole-human에 있다.

계측 범위: cache의 배치 구성·전송·forward·정규화CLS 반환·CPU memmap 저장. 다운로드/입력검사/tokenization/model loading/warmup 제외. CUDA 동기화 경계를 사용한다. 누적 지점의 JSON/flush 일시 정지는 시간에서 제외하지만 GPU 표본 구간에는 포함될 수 있다. RSS와 가용 RAM은0.5초 대기+조회 시간 간격의 sampled 값이다. GPU-Util은 nvidia-smi 장치 전체값으로 디스플레이 등도 포함될 수 있다. 다른 GPU compute job과 겹치지 않게 시작했다.

모델 적재 직후: allocated 0.1932 / reserved 0.1992 GiB, RSS 1.069 GiB, 시스템 MemAvailable 113.080 GiB. CUDA 초기화 및 CPU token cache mmap 열기를 포함하며 전체 cache를 GPU에 올리지 않았다.

|batch|누적 출력 세포|이번 측정 구간 처리 세포|초|세포/초|allocated peak GiB|추가 peak GiB|reserved peak GiB|GPU 평균/최대%|RSS sampled peak GiB|가용 최소 GiB|
|---|---|---|---|---|---|---|---|---|---|---|
|8|10000|10000|90.4098|110.6074|0.3894|0.1649|0.4043|94.1228|95.0|1.7432|112.0446|
|8|100000|100000|903.027|110.7387|0.3894|0.1649|0.4043|94.7725|95.0|2.4253|111.365|
|8|300000|200000|1760.1656|113.6257|0.3894|0.1649|0.4043|94.9524|96.0|3.15|111.9998|
|256|10000|10000|85.8946|116.4217|5.4254|5.2009|12.7871|94.8827|96.0|2.844|99.5076|
|256|100000|100000|879.885|113.6512|5.4621|5.2377|17.9863|94.9513|96.0|3.5202|94.0468|
|256|300000|300000|2628.6047|114.129|5.4773|5.2528|21.4746|95.1017|96.0|3.9045|90.2655|

추가 peak은 해당 조건 시작 baseline 대비 차이이며 모델 적재 직후 값과 동일하지 않을 수 있다.
CUDA allocated/reserved는 프로세스 allocator 수치다. 전체121.688GiB는 CPU/GPU 공유 RAM이며 별도 전용 VRAM이 아니다. RSS와 CUDA를 더하지 않는다. 80% 도달은 성공 조건이 아니며 실제 관측치를 보고한다. 세포 수 증가와 batch 증가는 다른 축이다.

#### 출력 검증과 한계

{"sampled_actual_cells": 256, "max_abs_embedding_difference_between_batches": 2.682209014892578e-07, "minimum_cosine_similarity": 0.9999998211860657, "output_shape": [300000, 512], "all_outputs_finite_checked_during_forward": true}

출력은 각 batch의 embeddings_batchB.npy, 300000×512 float32이다. 각 forward에서 유한값을 검사하며, 완료 후256개 실제 세포의 배치 간 embedding을 비교한다. 실제30만세포 임베딩 처리 결과이며 지도학습 정확도·donor 독립 평가·사전학습 데이터와의 독립성을 입증하지 않는다. 별도 pancreas 전체 학습 결과와 구분한다.

[공식 scGPT](https://github.com/bowang-lab/scGPT) · [공식 Census 원본 다운로드](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.download_source_h5ad.html) · [Tabula Sapiens collection](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5)

실행 코드: scripts/large_scale/prepare.py, resume.py, resumable_inference.py, finalize.py. 로그·GPU 시계열·설정·예측은 results/actual_300k_resume_20261028_093701_696230에 보존한다. 전날 최초 시도는 import 경로 오류로 실패해 retry1로 실행했고, 10만세포 지점에서 사용자 요청으로 중단한 뒤 오늘 새 run에서 나머지부터 재개했다. 최초 실패·중단·재개 로그를 모두 보존했다.

#### 30만 고유세포 전체 처리 최종 비교

batch8은 전날10만+재개20만의 유효 처리시간 합이며 야간 중단·모델 재적재·warmup은 제외한다. batch256은 한 세션의30만 누적시간이다. 두 조건의 시간 범위는 전처리cache 이후 GPU처리/반환/기록이며, 세션 분할에 따른 차이는 남는다. 메모리는 각 구간의 reset peak 중 최대이고 baseline 차이는 원본 행에서 확인한다. GPU 평균은 표본 수로 가중했다.

|batch|고유세포|유효초 합|세포/초|allocated peak GiB|reserved peak GiB|GPU 평균/최대%|세션 구간 수|
|---|---|---|---|---|---|---|---|
|8|300000|2663.19|112.65|0.389|0.404|94.89/96|2|
|256|300000|2628.60|114.13|5.477|21.475|95.10/96|1|

[전체30만세포 출력 검증](../results/actual_300k_resume_20261028_093701_696230/RECORDS.md#file-ae261a5d4f78): 각 조건300000×512, 전체finite·단위norm·세포별batch간cosine 검사 및 전날10만출력의 정확한 보존 확인. 새로운 GPU실험은 이어서 시작하지 않는다.

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
|`scgpt/scripts/00_prepare_inputs.py`|`d02cbe419cb536c00582aa100374e99467da69a5091f224f7ae860b7fb7e8dc9`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/results/input_check_20261023_030617_498960/00_prepare_inputs.py`|`d02cbe419cb536c00582aa100374e99467da69a5091f224f7ae860b7fb7e8dc9`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`scgpt/scripts/01_inference.py`|`2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/inference_20261023_031632_499710/01_inference.py`|`2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/inference_L1200/01_inference.py`|`2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/inference_L1024/01_inference.py`|`2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/source/01_inference.py`|`2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/inference_L0256/01_inference.py`|`2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/inference_L0512/01_inference.py`|`2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/scripts/01_run_inference.sh`|`07f6bef99e3ce0c503f4a937ecd1ea04af324b7b3c9baf11a20cfd1058b86005`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/02_finetune.py`|`e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/finetune_20261023_033701_500560/02_finetune.py`|`e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/02_finetune.py`|`e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/02_finetune.py`|`e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/source/02_finetune.py`|`e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/02_finetune.py`|`e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/02_finetune.py`|`e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존; 해당 SHA256이 실행 기록에 존재|
|`scgpt/scripts/02_run_finetune.sh`|`22f965af22beafe155a54c31a5ecc0cd24a7190b8c1a67043f98cfa8be4e0f5a`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/full_realdata.py`|`9368e97d5d003484531e9f52b6ef93e01836ed89eb9f986dd25b330a95450ab9`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증; 해당 SHA256이 실행 기록에 존재|
|`scgpt/scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|
|`scgpt/scripts/large_scale/discover.py`|`78bfd3cc3bfcc861d3e8f9955a259bdc86185b17775ba2654ff1f894f784ffad`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/download.py`|`4b991acb386eb2afd20cf69b398709754363756212d536249a9053d57199dea0`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/finalize.py`|`af0870b22df477c08140f85d6d3809bc97b0bf2d83db2746015771fe218f548a`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/inference.py`|`67dd1ecf01629fa5d7f0668cbf84a11abe049e294694a58ff4a6339c7cb5fe3a`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/launch.py`|`3488ebd0224604a3f2e0454a220e2ab1831b211d2741a84772083fbd613f382d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/pause.py`|`3e2ff120f03a8a44ee41e32f6d0e0029321039e0ac4835e0873d619d92326dbb`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/prepare.py`|`f5a10495f6d64b55f763bd3048475ac2cad61ecad2af64c391ab92484da22b0a`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/report.py`|`7e4412df1ca29c40cc86a43bdf84af035efae38af8a79f7c4ba7b9ca14633286`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/resumable_inference.py`|`21a2fe15e04c4b1c6fffc7586bedc67b97c2919bbd5c02b23cc6822b828741bc`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/resume.py`|`880321b874d8e549d78d4693d210d37341923c53a9767fe8e82f109eee566d21`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/large_scale/watch.py`|`408105c2ba3904123322025616bed935a9c291943ee728de3670ece5167fd5ac`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/launch_full_realdata.py`|`7fcd5a15c2e5b884fa5e84086dbec1e43df91dcfbe2e733aef689be6891fc299`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/launch_memory_refine.py`|`9593eb00e81a5c6a9efa072f1466e9155331b8e0069c6231060e44f81f59cba3`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/launch_memory_sweep.py`|`a599f7b196a46781b22f4d84def09d382b03541b0dd35d0d305cb935751a505f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/launch_reload_supplement.py`|`3acde93e464dbbeb01f510748deaa21899d21fb40ff2e4444b4cec09caba5465`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/launch_supplement.py`|`d324d418971d6c647398cdd366b761f84d3a87fc52895ca441caddd5df338d5b`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/launch_train_util.py`|`cd056e3375c7f3cd147c8a0225ea702be574e0c2f97aa2f4fdd6d43e4709e24d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/memory_batch_refine.py`|`71bd415ddfdbcd1f646e42a990904af5f13da38867a4b138612c0a2932d37606`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/memory_batch_sweep.py`|`8c2ad5b07edfd7a4290562891973db3ad1afdb4e65cf1f50f731a9896c4c9f80`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/reload_supplement.py`|`b732330bec4e8271c9c08f9ada25d3a531798873cdcf827439b26e59e76e5911`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/report_full_realdata.py`|`c53d84d82f5bd0e60d9e65b3f36670591d13250bc328b65311392871d9cb5a46`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/report_memory_sweep.py`|`fbedd2d0f6fcb119f5b76624661f60b966f0d299e4a483f8f13f665b49d97836`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/report_supplement.py`|`65b76ad7e9d5f118d2411e8a4f04d488a8835588462c8112540677a4e14f6ba9`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/resume_full_realdata.py`|`50356d8db966815221816ce92d63434e30c341d8b8c3dc89298a278eba5a544a`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/supplement.py`|`b903ef892ac0ccb8d759b02e72225adc41f41229974937b36c1728faee6a1b8c`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scgpt/scripts/train_util_supplement.py`|`d2019b4af252c4e04b4da9afc9606a036f76ac9158bf48357880070f91b7e3cc`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증; 해당 SHA256이 실행 기록에 존재|
|`scgpt/results/length_compare_20261023_040313_501423/compare_lengths.py`|`647d484d773956dc5f6591aa8db8de3ce6721b57fe08a6b6d97be476ff3ebb71`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|

<details>
<summary>scgpt/scripts/00_prepare_inputs.py — 원문 전문 (8,171 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/00_prepare_inputs.py", "sha256": "d02cbe419cb536c00582aa100374e99467da69a5091f224f7ae860b7fb7e8dc9", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Download author-provided pancreas data and inspect it without preprocessing.

No expression values, labels, gene names or train/test assignments are changed.
"""
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import requests
from scipy import sparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

root, output = map(Path, sys.argv[1:3])
data = root / "data/pancreas_demo"
data.mkdir(parents=True, exist_ok=True)
model = root / "models/whole-human"
vocab = json.loads((model / "vocab.json").read_text())
args = json.loads((model / "args.json").read_text())
if not (model / "best_model.pt").is_file():
    raise FileNotFoundError(model / "best_model.pt")

# TOSICA authors' Figshare releases; do not claim byte identity with scGPT's Drive copy.
sources = [
    ("demo_train.h5ad", "39009503", "642c52f7556acd6e6f60e00fc52c4509", "21980552"),
    ("demo_test.h5ad", "39009500", "5083cd8fafe9d6f4c389d9cde43ce7f2", "21980549"),
]
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"], respect_retry_after_header=True)))

def digests(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()

report = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "input inspection only; no preprocessing, inference or training",
    "source": "Author-provided TOSICA pancreas demo files on Figshare",
    "files": [],
    "model_config": {key: args.get(key) for key in [
        "embsize", "nheads", "d_hid", "nlayers", "n_bins", "max_seq_len", "input_style"]},
}
labels_by_file, genes_by_file = {}, {}
for name, file_id, expected_md5, article_id in sources:
    path = data / name
    url = f"https://ndownloader.figshare.com/files/{file_id}"
    if not path.exists():
        print("Downloading:", name, flush=True)
        temp = path.with_suffix(".h5ad.part")
        with session.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temp.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        actual_md5, _ = digests(temp)
        if actual_md5 != expected_md5:
            raise ValueError(f"Download checksum mismatch: {temp}; retry the script")
        temp.replace(path)
    actual_md5, sha256 = digests(path)
    if actual_md5 != expected_md5:
        raise ValueError(f"Existing file differs from the specified release: {path}. It was not overwritten.")

    a = ad.read_h5ad(path)
    if a.X is None or a.n_obs == 0 or a.n_vars == 0:
        raise ValueError(f"No expression matrix in {path}")
    # Small dense slices are used only for display/statistics; the original is unchanged.
    block = a.X[:5, :5]
    block = block.toarray() if sparse.issparse(block) else np.asarray(block)
    sample = a.X[:min(256, a.n_obs), :]
    sample = sample.toarray() if sparse.issparse(sample) else np.asarray(sample)
    finite = np.isfinite(sample)
    nonzero = sample[finite & (sample != 0)]
    finite_values = sample[finite]
    celltypes = a.obs["Celltype"] if "Celltype" in a.obs else None
    label_counts = {} if celltypes is None else {
        str(key): int(value) for key, value in celltypes.dropna().value_counts().items() if value > 0}
    gene_match = int(sum(str(gene) in vocab for gene in a.var_names))
    var_matches = {}
    for column in a.var.columns:
        if a.var[column].dtype.kind in "OUS" or str(a.var[column].dtype) == "category":
            var_matches[str(column)] = int(a.var[column].astype(str).isin(vocab).sum())
    labels_by_file[name] = set(label_counts)
    genes_by_file[name] = a.var_names.tolist()
    item = {
        "filename": name, "source_url": url,
        "article_url": f"https://figshare.com/articles/dataset/{path.stem}_h5ad/{article_id}",
        "size_bytes": path.stat().st_size, "md5": actual_md5, "sha256": sha256,
        "shape_cells_by_genes": list(a.shape), "x_type": type(a.X).__name__,
        "x_dtype": str(a.X.dtype), "obs_columns": a.obs.columns.tolist(),
        "var_columns": a.var.columns.tolist(), "layers": list(a.layers.keys()),
        "raw_shape": None if a.raw is None else list(a.raw.shape),
        "uns_keys": list(a.uns.keys()), "has_log1p_metadata": "log1p" in a.uns,
        "gene_names_first_10": a.var_names[:10].tolist(),
        "gene_names_unique": bool(a.var_names.is_unique),
        "cell_ids_unique": bool(a.obs_names.is_unique),
        "vocab_matches_var_names": gene_match, "vocab_matches_var_columns": var_matches,
        "label_column": None if celltypes is None else "Celltype",
        "label_counts": label_counts,
        "missing_labels": None if celltypes is None else int(celltypes.isna().sum()),
        "value_sample": {
            "shape": list(sample.shape), "nonfinite_count": int((~finite).sum()),
            "min": None if finite_values.size == 0 else float(finite_values.min()),
            "max": None if finite_values.size == 0 else float(finite_values.max()),
            "zero_fraction": float(np.mean(sample == 0)),
            "nonzero_integer_like_fraction": None if nonzero.size == 0 else float(
                np.mean(np.isclose(nonzero, np.rint(nonzero), atol=1e-6, rtol=0))),
            "note": "Sample statistics alone do not establish raw-count or normalization provenance.",
        },
    }
    report["files"].append(item)
    a.obs.head(10).to_csv(output / f"{path.stem}_obs_head.tsv", sep="\t")
    a.var.head(10).to_csv(output / f"{path.stem}_var_head.tsv", sep="\t")
    import pandas as pd
    block_frame = pd.DataFrame(block, index=a.obs_names[:5], columns=a.var_names[:5])
    block_frame.to_csv(output / f"{path.stem}_expression_5x5.tsv", sep="\t")

    print("\n[scGPT input]", name)
    print("shape (cells x genes):", a.shape, "X:", item["x_type"], item["x_dtype"])
    print("obs columns:", item["obs_columns"], "var columns:", item["var_columns"])
    print("genes first 10:", item["gene_names_first_10"])
    print("vocab matches:", f"{gene_match}/{a.n_vars}", "var columns:", var_matches)
    print("layers:", item["layers"], "raw shape:", item["raw_shape"], "uns:", item["uns_keys"])
    print("expression 5 x 5:\n" + block_frame.to_string())
    print("Celltype counts:", json.dumps(label_counts, ensure_ascii=False))
    print("missing labels:", item["missing_labels"])
    print("expression sample:", json.dumps(item["value_sample"], ensure_ascii=False))
    del a

train, test = "demo_train.h5ad", "demo_test.h5ad"
report["cross_file_checks"] = {
    "identical_gene_order": genes_by_file[train] == genes_by_file[test],
    "shared_genes": len(set(genes_by_file[train]) & set(genes_by_file[test])),
    "test_only_labels": sorted(labels_by_file[test] - labels_by_file[train]),
    "train_only_labels": sorted(labels_by_file[train] - labels_by_file[test]),
    "note": "Test-only classes require an explicit evaluation policy before classification training.",
}
print("\n[Train/test checks]", json.dumps(report["cross_file_checks"], ensure_ascii=False))
print("[Model config]", json.dumps(report["model_config"]))
(output / "input_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
versions = {}
for package in ["torch", "scgpt", "anndata", "scanpy", "numpy", "scipy", "pandas", "requests"]:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
(output / "environment.json").write_text(json.dumps({
    "python": platform.python_version(), "machine": platform.machine(), "packages": versions,
}, indent=2) + "\n")
print("report:", output / "input_report.json")
print("INPUT_CHECK_COMPLETE")
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "scgpt/results/input_check_20261023_030617_498960/00_prepare_inputs.py", "source": "scgpt/scripts/00_prepare_inputs.py", "sha256": "d02cbe419cb536c00582aa100374e99467da69a5091f224f7ae860b7fb7e8dc9"} -->

</details>

<details>
<summary>scgpt/scripts/01_inference.py — 원문 전문 (19,350 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/01_inference.py", "sha256": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Measured scGPT embeddings, adapted from official scgpt/tasks/cell_emb.py.

Input: author-distributed pancreas h5ad, already inspected on Spark.
Expression values are used as supplied, with per-cell quantile binning only.
Celltype is metadata, never an input to the encoder; no classifier is trained.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer import GeneVocab

ROOT = Path('/workspace/scgpt')
GIB = 1024 ** 3

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ResourceSampler:
    """Sample this Python process and CUDA allocator every 0.2 seconds."""
    def __init__(self):
        self.process = psutil.Process()
        self.rows = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def sample(self):
        with self.lock:
            vm = psutil.virtual_memory()
            self.rows.append({
                'elapsed_seconds': time.perf_counter() - self.start_time,
                'process_rss_bytes': self.process.memory_info().rss,
                'process_cpu_percent': self.process.cpu_percent(),
                'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'cuda_reserved_bytes': torch.cuda.memory_reserved(),
                'system_available_bytes': vm.available,
            })

    def loop(self):
        while not self.stop_event.wait(0.2):
            self.sample()

    def start(self):
        self.start_time = time.perf_counter()
        self.process.cpu_percent()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--split', choices=['train', 'test'], default='test')
parser.add_argument('--n-cells', type=int, default=1000, help='0 means all cells')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--max-length', type=int, default=1200, help='includes the CLS token')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out', type=Path, required=True)
opt = parser.parse_args()
if opt.batch_size < 1 or opt.max_length < 2 or opt.n_cells < 0:
    parser.error('batch-size >= 1, max-length >= 2, n-cells >= 0 required')
out = opt.out
out.mkdir(parents=True, exist_ok=True)
if (out / 'summary.json').exists():
    raise FileExistsError(f'Results already exist: {out}')
started_at = datetime.now(timezone.utc).isoformat()
torch.set_num_threads(8)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; stopping instead of measuring CPU inference')
torch.cuda.set_device(0)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
seed_all(opt.seed)

model_dir = ROOT / 'models/whole-human'
model_args = json.loads((model_dir / 'args.json').read_text())
if model_args.get('n_bins') != 51 or model_args.get('input_style') != 'binned':
    raise ValueError('This script requires the inspected 51-bin, binned-input checkpoint')
if model_args.get('input_emb_style', 'continuous') != 'continuous':
    raise ValueError('This script requires a continuous expression-value encoder')
if model_args.get('pre_norm', False):
    raise ValueError('This script implements the inspected post-norm checkpoint')
vocab = GeneVocab.from_file(model_dir / 'vocab.json')
for special in [model_args['pad_token'], '<cls>', '<eoc>']:
    if special not in vocab:
        vocab.append_token(special)
pad_id, cls_id = vocab[model_args['pad_token']], vocab['<cls>']
pad_value = model_args['pad_value']
vocab.set_default_index(pad_id)

input_path = ROOT / f'data/pancreas_demo/demo_{opt.split}.h5ad'
t0 = time.perf_counter()
adata = ad.read_h5ad(input_path)
expected = (10600, 3000) if opt.split == 'train' else (4218, 3000)
if adata.shape != expected or 'Celltype' not in adata.obs:
    raise ValueError(f'Input differs from the inspected data: {adata.shape}, {adata.obs.columns.tolist()}')
if not adata.var_names.is_unique or not adata.obs_names.is_unique:
    raise ValueError('Gene names and cell IDs must be unique')
n = adata.n_obs if opt.n_cells == 0 else opt.n_cells
if n > adata.n_obs:
    raise ValueError(f'Requested {n} cells, but input contains {adata.n_obs}')
indices = (np.arange(adata.n_obs) if n == adata.n_obs else
           np.sort(np.random.default_rng(opt.seed).choice(adata.n_obs, n, replace=False)))
gene_names = adata.var_names.astype(str).to_numpy()
gene_mask = np.array([g in vocab for g in gene_names], dtype=bool)
if int(gene_mask.sum()) != 2999:
    raise ValueError(f'Expected 2999 matched genes; got {int(gene_mask.sum())}')
selected = adata[indices, gene_mask].copy()
matrix = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
matrix = matrix.astype(np.float32, copy=False)
if not np.isfinite(matrix).all() or (matrix < 0).any():
    raise ValueError('Selected expression matrix must be finite and nonnegative')
nnz = np.count_nonzero(matrix, axis=1)
if (nnz == 0).any():
    raise ValueError('Some selected cells have no nonzero expression after vocabulary filtering')
gene_ids = np.array(vocab(selected.var_names.tolist()), dtype=np.int64)
cell_metadata = selected.obs.copy()
cell_metadata.index.name = 'cell_id'
cell_metadata.insert(0, 'original_row_index', indices)
cell_metadata.insert(1, 'embedding_row', np.arange(n))
cell_metadata['nonzero_genes_before_sampling'] = nnz
cell_metadata['tokens_before_sampling_including_cls'] = nnz + 1
cell_metadata['tokens_used_including_cls'] = np.minimum(nnz + 1, opt.max_length)
cell_metadata['sampled_due_to_length'] = nnz + 1 > opt.max_length
cell_metadata.to_csv(out / 'selected_cells.tsv', sep='\t')
pd.DataFrame({'gene_name': selected.var_names, 'vocab_id': gene_ids}).to_csv(
    out / 'matched_genes.tsv', sep='\t', index=False)
(out / 'excluded_genes.txt').write_text('\n'.join(gene_names[~gene_mask]) + '\n')
del adata, selected
gc.collect()
input_preparation_seconds = time.perf_counter() - t0
print(f'[Input] source={opt.split}; selected matrix={matrix.shape}; matched genes=2999/3000', flush=True)
print(f'[Length] max_length={opt.max_length} includes CLS; sampled cells={int((nnz + 1 > opt.max_length).sum())}/{n}', flush=True)

# Snapshot configuration, source code and file identities for reproducibility.
for name in ['args.json', 'vocab.json']:
    shutil.copy2(model_dir / name, out / f'pretrained_{name}')
source_dir = out / 'source_snapshot'
source_dir.mkdir(exist_ok=True)
for name, obj in [('data_collator.py', DataCollator), ('model.py', TransformerModel),
                  ('gene_tokenizer.py', GeneVocab), ('preprocess.py', binning)]:
    shutil.copy2(inspect.getfile(obj), source_dir / name)
versions = {}
for pkg in ['torch', 'scgpt', 'anndata', 'scanpy', 'numpy', 'pandas', 'scipy', 'psutil']:
    versions[pkg] = metadata.version(pkg)
direct_url = metadata.distribution('scgpt').read_text('direct_url.json')
device = torch.cuda.get_device_properties(0)
write_json(out / 'environment.json', {
    'python': sys.version, 'architecture': platform.machine(), 'packages': versions,
    'scgpt_install_source': json.loads(direct_url) if direct_url else None,
    'cuda_version': torch.version.cuda, 'gpu_name': device.name,
    'gpu_compute_capability': [device.major, device.minor],
    'cuda_reported_total_memory_gib': device.total_memory / GIB,
    'system_total_memory_gib': psutil.virtual_memory().total / GIB,
    'torch_cpu_threads': torch.get_num_threads(),
})
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
(out / 'requirements_freeze.txt').write_text(freeze.stdout)
write_json(out / 'provenance.json', {
    'input': {'path': str(input_path), 'sha256': sha256(input_path)},
    'checkpoint': {'path': str(model_dir / 'best_model.pt'), 'sha256': sha256(model_dir / 'best_model.pt')},
    'script_sha256': sha256(__file__),
    'reference_implementation': 'https://github.com/bowang-lab/scGPT/blob/cebd6fae655b9c585a4807daa3ac31bb764f06b4/scgpt/tasks/cell_emb.py',
    'tutorial': 'https://scgpt.readthedocs.io/en/latest/tutorial_reference_mapping.html',
})
configuration = {
    'started_at_utc': started_at, 'task': 'pretrained cell embeddings',
    'split': opt.split, 'n_cells': n, 'batch_size': opt.batch_size,
    'max_length_including_cls': opt.max_length, 'seed': opt.seed,
    'precision': 'FP32', 'autocast': False, 'tf32': False,
    'backend': 'native PyTorch Transformer; use_fast_transformer=False',
    'additional_normalize_total': False, 'additional_log1p': False,
    'expression_provenance': 'Values supplied in h5ad; no raw layer or preprocessing metadata',
    'binning': 'official DataCollator: per-cell quantiles, n_bins=51; nonzero expression only',
    'length_policy': 'keep CLS; if over length, randomly sample gene/value pairs; pad to batch maximum capped at max_length',
    'pooling': 'CLS position; L2-normalized 512-dimensional embedding',
    'labels_used_as_model_input': False, 'classifier_training': False,
    'timed_scope': 'gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording',
    'excluded_from_inference_timer': 'imports, h5ad loading, cell/gene selection, model loading, one-batch warmup, file saving, accuracy evaluation',
    'measurement_repeats': 1,
}
write_json(out / 'run_config.json', configuration)

t0 = time.perf_counter()
model = TransformerModel(
    ntoken=len(vocab), d_model=model_args['embsize'], nhead=model_args['nheads'],
    d_hid=model_args['d_hid'], nlayers=model_args['nlayers'],
    nlayers_cls=model_args['n_layers_cls'], n_cls=1, vocab=vocab,
    dropout=model_args['dropout'], pad_token=model_args['pad_token'], pad_value=pad_value,
    do_mvc=True, do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
    input_emb_style='continuous', cell_emb_style='cls', explicit_zero_prob=False,
    use_fast_transformer=False, pre_norm=False,
)
state = torch.load(model_dir / 'best_model.pt', map_location='cpu', weights_only=True)
if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
    raise TypeError('Expected a plain tensor state_dict; checkpoint format requires review')
# These are the official load_pretrained native-transformer rename rules.
rules = {
    r'self_attn\._impl\.Wqkv\.': 'self_attn.in_proj_',
    r'self_attn\.Wqkv\.': 'self_attn.in_proj_',
    r'self_attn\._impl\.out_proj\.': 'self_attn.out_proj.',
}
converted = {}
for key, value in state.items():
    for pattern, replacement in rules.items():
        key = re.sub(pattern, replacement, key)
    if key in converted:
        raise ValueError(f'Duplicate checkpoint key after conversion: {key}')
    converted[key] = value
current = model.state_dict()
compatible = {k: v for k, v in converted.items() if k in current and v.shape == current[k].shape}
core_prefixes = ('encoder.', 'value_encoder.', 'transformer_encoder.')
required = [k for k in current if k.startswith(core_prefixes)]
missing_core = [k for k in required if k not in compatible]
load_report = {
    'required_embedding_keys': len(required), 'missing_embedding_keys': missing_core,
    'loaded_keys': sorted(compatible),
    'unused_or_incompatible_checkpoint_keys': sorted(set(converted) - set(compatible)),
    'model_keys_not_loaded': sorted(set(current) - set(compatible)),
    'note': 'All parameters used by _encode must load; decoder heads are unused in this embedding run.',
}
write_json(out / 'weight_loading.json', load_report)
if missing_core:
    raise RuntimeError(f'Embedding weights did not fully load: {missing_core[:10]}')
model.load_state_dict(compatible, strict=False)
parameter_count_total = sum(p.numel() for p in model.parameters())
parameter_count_encoder = sum(p.numel() for name, p in model.named_parameters() if name.startswith(core_prefixes))
del state, converted, current, compatible
gc.collect()
model.float().to('cuda').eval()
torch.cuda.synchronize()
model_loading_seconds = time.perf_counter() - t0
print(f'[Model] embedding weights loaded: {len(required)}/{len(required)}; output_dim={model_args["embsize"]}', flush=True)

collator = DataCollator(do_padding=True, pad_token_id=pad_id, pad_value=pad_value,
    do_mlm=False, do_binning=True, max_length=opt.max_length, sampling=True, keep_first_n_tokens=1)

def make_batch(start, end):
    examples = []
    for i in range(start, end):
        nz = np.flatnonzero(matrix[i])
        genes = np.concatenate([np.array([cls_id], dtype=np.int64), gene_ids[nz]])
        values = np.concatenate([np.array([pad_value], dtype=np.float32), matrix[i, nz]])
        examples.append({'genes': torch.from_numpy(genes), 'expressions': torch.from_numpy(values)})
    return collator(examples)

def encode_batch(batch):
    genes = batch['gene'].to('cuda')
    values = batch['expr'].to('cuda')
    encoded = model._encode(genes, values, src_key_padding_mask=genes.eq(pad_id))
    result = torch.nn.functional.normalize(encoded[:, 0, :], p=2, dim=1)
    return result.cpu().numpy()

# Warm the CUDA execution path, then reset random seeds before the measured run.
seed_all(opt.seed)
t0 = time.perf_counter()
with torch.inference_mode():
    warm_batch = make_batch(0, min(opt.batch_size, n))
    warm_result = encode_batch(warm_batch)
torch.cuda.synchronize()
warmup_seconds = time.perf_counter() - t0
del warm_batch, warm_result
model.cur_gene_token_embs = None
gc.collect()
seed_all(opt.seed)
torch.cuda.reset_peak_memory_stats()
embeddings = np.empty((n, model_args['embsize']), dtype=np.float32)
tokens_saved = np.full((n, opt.max_length), pad_id, dtype=np.int64)
values_saved = np.full((n, opt.max_length), pad_value, dtype=np.float32)
batch_records = []
sampler = ResourceSampler()
sampler.start()
torch.cuda.synchronize()
t0 = time.perf_counter()
try:
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, n, opt.batch_size)):
            end = min(start + opt.batch_size, n)
            batch = make_batch(start, end)
            length = batch['gene'].shape[1]
            tokens_saved[start:end, :length] = batch['gene'].numpy()
            values_saved[start:end, :length] = batch['expr'].numpy()
            embeddings[start:end] = encode_batch(batch)
            batch_records.append({'batch': batch_index, 'start': start, 'end': end,
                                  'batch_size': end - start, 'padded_token_length': length})
            if batch_index == 0:
                print(f'[Model input] gene IDs={tuple(batch["gene"].shape)}, expression bins={tuple(batch["expr"].shape)}', flush=True)
            if (batch_index + 1) % 25 == 0 or end == n:
                print(f'[Inference] {end}/{n} cells', flush=True)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t0
finally:
    sampler.stop()
    pd.DataFrame(sampler.rows).to_csv(out / 'resources.csv', index=False)

if not np.isfinite(embeddings).all():
    raise FloatingPointError('Embeddings contain NaN/Inf')
norms = np.linalg.norm(embeddings, axis=1)
if not np.allclose(norms, 1, atol=1e-4):
    raise ValueError('Embedding L2 norms are not approximately 1')
peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
t0 = time.perf_counter()
np.save(out / 'embeddings.npy', embeddings)
np.savez_compressed(out / 'tokenized_inputs.npz', gene_ids=tokens_saved, expression_bins=values_saved)
pd.DataFrame(batch_records).to_csv(out / 'batch_shapes.tsv', sep='\t', index=False)
itos = vocab.get_itos()
used_first = int(cell_metadata['tokens_used_including_cls'].iloc[0])
pd.DataFrame({
    'position': np.arange(min(20, used_first)),
    'gene_id': tokens_saved[0, :min(20, used_first)],
    'gene_name': [itos[int(i)] for i in tokens_saved[0, :min(20, used_first)]],
    'expression_bin': values_saved[0, :min(20, used_first)],
}).to_csv(out / 'first_cell_token_example.tsv', sep='\t', index=False)
output_save_seconds = time.perf_counter() - t0
summary = {
    'status': 'complete', 'task': 'pretrained cell embeddings', 'split': opt.split,
    'cells': n, 'matched_genes': int(gene_mask.sum()), 'embedding_shape': list(embeddings.shape),
    'batch_size': opt.batch_size, 'max_length_including_cls': opt.max_length,
    'precision': 'FP32', 'seed': opt.seed,
    'parameter_count_constructed_model': parameter_count_total,
    'parameter_count_embedding_encoder': parameter_count_encoder,
    'input_preparation_seconds': input_preparation_seconds,
    'model_loading_seconds': model_loading_seconds, 'warmup_seconds': warmup_seconds,
    'inference_seconds': inference_seconds, 'cells_per_second': n / inference_seconds,
    'output_save_seconds': output_save_seconds,
    'gpu_peak_allocated_gib': peak_allocated / GIB,
    'gpu_peak_reserved_gib': peak_reserved / GIB,
    'python_process_rss_peak_observed_gib': max(r['process_rss_bytes'] for r in sampler.rows) / GIB,
    'resource_sampling_interval_seconds': 0.2,
    'length_sampling_cells': int((nnz + 1 > opt.max_length).sum()),
    'tokens_before_sampling_min_mean_max': [int((nnz + 1).min()), float((nnz + 1).mean()), int((nnz + 1).max())],
    'tokens_after_sampling_min_mean_max': [int(np.minimum(nnz + 1, opt.max_length).min()), float(np.minimum(nnz + 1, opt.max_length).mean()), int(np.minimum(nnz + 1, opt.max_length).max())],
    'embedding_l2_norm_min_max': [float(norms.min()), float(norms.max())],
    'accuracy_evaluated': False, 'weights_updated': False,
    'timing_scope': configuration['timed_scope'],
    'memory_notes': 'CUDA allocator peaks are process-local PyTorch counters, not total device usage. Reserved memory can include warmup cache. RSS is sampled, not an exact peak. Spark uses shared memory; do not sum RSS and CUDA counters as independent physical pools.',
}
write_json(out / 'summary.json', summary)
print('\n' + json.dumps({key: summary[key] for key in [
    'cells', 'embedding_shape', 'batch_size', 'max_length_including_cls', 'precision',
    'inference_seconds', 'cells_per_second', 'gpu_peak_allocated_gib',
    'gpu_peak_reserved_gib', 'python_process_rss_peak_observed_gib', 'length_sampling_cells',
]}, indent=2), flush=True)
print('SCGPT_INFERENCE_COMPLETE', flush=True)
print('Results:', out, flush=True)
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "scgpt/results/inference_20261023_031632_499710/01_inference.py", "source": "scgpt/scripts/01_inference.py", "sha256": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/01_inference.py", "source": "scgpt/scripts/01_inference.py", "sha256": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/01_inference.py", "source": "scgpt/scripts/01_inference.py", "sha256": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/source/01_inference.py", "source": "scgpt/scripts/01_inference.py", "sha256": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/01_inference.py", "source": "scgpt/scripts/01_inference.py", "sha256": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/01_inference.py", "source": "scgpt/scripts/01_inference.py", "sha256": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c"} -->

</details>

<details>
<summary>scgpt/scripts/01_run_inference.sh — 원문 전문 (20,910 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/01_run_inference.sh", "sha256": "07f6bef99e3ce0c503f4a937ecd1ea04af324b7b3c9baf11a20cfd1058b86005", "encoding": "utf-8", "added_final_newline": false} -->
```bash
#!/usr/bin/env bash
# scGPT step 01: measured pretrained cell-embedding inference, with saved outputs.
# Default: 1,000 test cells, batch 8, max length 1,200 including CLS, FP32.
# Run on Spark as the normal user: bash scgpt_infer.sh
set -euo pipefail
root="$HOME/ai-benchmark/scgpt"
container="scgpt-spark"
run_id="inference_$(date +%Y%m%d_%H%M%S)_$$"
sudo -v
mount_source=$(sudo docker inspect "$container" --format '{{range .Mounts}}{{if eq .Destination "/workspace/scgpt"}}{{.Source}}{{end}}{{end}}')
if [[ -z "$mount_source" || "$(readlink -f "$mount_source")" != "$(readlink -f "$root")" ]]; then
    echo "ERROR: expected $root mounted at /workspace/scgpt; found $mount_source" >&2
    exit 1
fi
mkdir -p "$root/scripts" "$root/results/$run_id"
runner="$root/scripts/01_run_inference.sh"
if [[ "$(readlink -f "$0")" != "$(readlink -f "$runner")" ]]; then cp -- "$0" "$runner"; fi

cat > "$root/scripts/01_inference.py" <<'PY'
"""Measured scGPT embeddings, adapted from official scgpt/tasks/cell_emb.py.

Input: author-distributed pancreas h5ad, already inspected on Spark.
Expression values are used as supplied, with per-cell quantile binning only.
Celltype is metadata, never an input to the encoder; no classifier is trained.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer import GeneVocab

ROOT = Path('/workspace/scgpt')
GIB = 1024 ** 3

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ResourceSampler:
    """Sample this Python process and CUDA allocator every 0.2 seconds."""
    def __init__(self):
        self.process = psutil.Process()
        self.rows = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def sample(self):
        with self.lock:
            vm = psutil.virtual_memory()
            self.rows.append({
                'elapsed_seconds': time.perf_counter() - self.start_time,
                'process_rss_bytes': self.process.memory_info().rss,
                'process_cpu_percent': self.process.cpu_percent(),
                'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'cuda_reserved_bytes': torch.cuda.memory_reserved(),
                'system_available_bytes': vm.available,
            })

    def loop(self):
        while not self.stop_event.wait(0.2):
            self.sample()

    def start(self):
        self.start_time = time.perf_counter()
        self.process.cpu_percent()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--split', choices=['train', 'test'], default='test')
parser.add_argument('--n-cells', type=int, default=1000, help='0 means all cells')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--max-length', type=int, default=1200, help='includes the CLS token')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out', type=Path, required=True)
opt = parser.parse_args()
if opt.batch_size < 1 or opt.max_length < 2 or opt.n_cells < 0:
    parser.error('batch-size >= 1, max-length >= 2, n-cells >= 0 required')
out = opt.out
out.mkdir(parents=True, exist_ok=True)
if (out / 'summary.json').exists():
    raise FileExistsError(f'Results already exist: {out}')
started_at = datetime.now(timezone.utc).isoformat()
torch.set_num_threads(8)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; stopping instead of measuring CPU inference')
torch.cuda.set_device(0)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
seed_all(opt.seed)

model_dir = ROOT / 'models/whole-human'
model_args = json.loads((model_dir / 'args.json').read_text())
if model_args.get('n_bins') != 51 or model_args.get('input_style') != 'binned':
    raise ValueError('This script requires the inspected 51-bin, binned-input checkpoint')
if model_args.get('input_emb_style', 'continuous') != 'continuous':
    raise ValueError('This script requires a continuous expression-value encoder')
if model_args.get('pre_norm', False):
    raise ValueError('This script implements the inspected post-norm checkpoint')
vocab = GeneVocab.from_file(model_dir / 'vocab.json')
for special in [model_args['pad_token'], '<cls>', '<eoc>']:
    if special not in vocab:
        vocab.append_token(special)
pad_id, cls_id = vocab[model_args['pad_token']], vocab['<cls>']
pad_value = model_args['pad_value']
vocab.set_default_index(pad_id)

input_path = ROOT / f'data/pancreas_demo/demo_{opt.split}.h5ad'
t0 = time.perf_counter()
adata = ad.read_h5ad(input_path)
expected = (10600, 3000) if opt.split == 'train' else (4218, 3000)
if adata.shape != expected or 'Celltype' not in adata.obs:
    raise ValueError(f'Input differs from the inspected data: {adata.shape}, {adata.obs.columns.tolist()}')
if not adata.var_names.is_unique or not adata.obs_names.is_unique:
    raise ValueError('Gene names and cell IDs must be unique')
n = adata.n_obs if opt.n_cells == 0 else opt.n_cells
if n > adata.n_obs:
    raise ValueError(f'Requested {n} cells, but input contains {adata.n_obs}')
indices = (np.arange(adata.n_obs) if n == adata.n_obs else
           np.sort(np.random.default_rng(opt.seed).choice(adata.n_obs, n, replace=False)))
gene_names = adata.var_names.astype(str).to_numpy()
gene_mask = np.array([g in vocab for g in gene_names], dtype=bool)
if int(gene_mask.sum()) != 2999:
    raise ValueError(f'Expected 2999 matched genes; got {int(gene_mask.sum())}')
selected = adata[indices, gene_mask].copy()
matrix = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
matrix = matrix.astype(np.float32, copy=False)
if not np.isfinite(matrix).all() or (matrix < 0).any():
    raise ValueError('Selected expression matrix must be finite and nonnegative')
nnz = np.count_nonzero(matrix, axis=1)
if (nnz == 0).any():
    raise ValueError('Some selected cells have no nonzero expression after vocabulary filtering')
gene_ids = np.array(vocab(selected.var_names.tolist()), dtype=np.int64)
cell_metadata = selected.obs.copy()
cell_metadata.index.name = 'cell_id'
cell_metadata.insert(0, 'original_row_index', indices)
cell_metadata.insert(1, 'embedding_row', np.arange(n))
cell_metadata['nonzero_genes_before_sampling'] = nnz
cell_metadata['tokens_before_sampling_including_cls'] = nnz + 1
cell_metadata['tokens_used_including_cls'] = np.minimum(nnz + 1, opt.max_length)
cell_metadata['sampled_due_to_length'] = nnz + 1 > opt.max_length
cell_metadata.to_csv(out / 'selected_cells.tsv', sep='\t')
pd.DataFrame({'gene_name': selected.var_names, 'vocab_id': gene_ids}).to_csv(
    out / 'matched_genes.tsv', sep='\t', index=False)
(out / 'excluded_genes.txt').write_text('\n'.join(gene_names[~gene_mask]) + '\n')
del adata, selected
gc.collect()
input_preparation_seconds = time.perf_counter() - t0
print(f'[Input] source={opt.split}; selected matrix={matrix.shape}; matched genes=2999/3000', flush=True)
print(f'[Length] max_length={opt.max_length} includes CLS; sampled cells={int((nnz + 1 > opt.max_length).sum())}/{n}', flush=True)

# Snapshot configuration, source code and file identities for reproducibility.
for name in ['args.json', 'vocab.json']:
    shutil.copy2(model_dir / name, out / f'pretrained_{name}')
source_dir = out / 'source_snapshot'
source_dir.mkdir(exist_ok=True)
for name, obj in [('data_collator.py', DataCollator), ('model.py', TransformerModel),
                  ('gene_tokenizer.py', GeneVocab), ('preprocess.py', binning)]:
    shutil.copy2(inspect.getfile(obj), source_dir / name)
versions = {}
for pkg in ['torch', 'scgpt', 'anndata', 'scanpy', 'numpy', 'pandas', 'scipy', 'psutil']:
    versions[pkg] = metadata.version(pkg)
direct_url = metadata.distribution('scgpt').read_text('direct_url.json')
device = torch.cuda.get_device_properties(0)
write_json(out / 'environment.json', {
    'python': sys.version, 'architecture': platform.machine(), 'packages': versions,
    'scgpt_install_source': json.loads(direct_url) if direct_url else None,
    'cuda_version': torch.version.cuda, 'gpu_name': device.name,
    'gpu_compute_capability': [device.major, device.minor],
    'cuda_reported_total_memory_gib': device.total_memory / GIB,
    'system_total_memory_gib': psutil.virtual_memory().total / GIB,
    'torch_cpu_threads': torch.get_num_threads(),
})
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
(out / 'requirements_freeze.txt').write_text(freeze.stdout)
write_json(out / 'provenance.json', {
    'input': {'path': str(input_path), 'sha256': sha256(input_path)},
    'checkpoint': {'path': str(model_dir / 'best_model.pt'), 'sha256': sha256(model_dir / 'best_model.pt')},
    'script_sha256': sha256(__file__),
    'reference_implementation': 'https://github.com/bowang-lab/scGPT/blob/cebd6fae655b9c585a4807daa3ac31bb764f06b4/scgpt/tasks/cell_emb.py',
    'tutorial': 'https://scgpt.readthedocs.io/en/latest/tutorial_reference_mapping.html',
})
configuration = {
    'started_at_utc': started_at, 'task': 'pretrained cell embeddings',
    'split': opt.split, 'n_cells': n, 'batch_size': opt.batch_size,
    'max_length_including_cls': opt.max_length, 'seed': opt.seed,
    'precision': 'FP32', 'autocast': False, 'tf32': False,
    'backend': 'native PyTorch Transformer; use_fast_transformer=False',
    'additional_normalize_total': False, 'additional_log1p': False,
    'expression_provenance': 'Values supplied in h5ad; no raw layer or preprocessing metadata',
    'binning': 'official DataCollator: per-cell quantiles, n_bins=51; nonzero expression only',
    'length_policy': 'keep CLS; if over length, randomly sample gene/value pairs; pad to batch maximum capped at max_length',
    'pooling': 'CLS position; L2-normalized 512-dimensional embedding',
    'labels_used_as_model_input': False, 'classifier_training': False,
    'timed_scope': 'gene/value tokenization, binning, length sampling/padding, CPU-to-GPU transfer, encoder, CLS selection, L2 normalization, GPU-to-CPU transfer, in-memory input/output recording',
    'excluded_from_inference_timer': 'imports, h5ad loading, cell/gene selection, model loading, one-batch warmup, file saving, accuracy evaluation',
    'measurement_repeats': 1,
}
write_json(out / 'run_config.json', configuration)

t0 = time.perf_counter()
model = TransformerModel(
    ntoken=len(vocab), d_model=model_args['embsize'], nhead=model_args['nheads'],
    d_hid=model_args['d_hid'], nlayers=model_args['nlayers'],
    nlayers_cls=model_args['n_layers_cls'], n_cls=1, vocab=vocab,
    dropout=model_args['dropout'], pad_token=model_args['pad_token'], pad_value=pad_value,
    do_mvc=True, do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
    input_emb_style='continuous', cell_emb_style='cls', explicit_zero_prob=False,
    use_fast_transformer=False, pre_norm=False,
)
state = torch.load(model_dir / 'best_model.pt', map_location='cpu', weights_only=True)
if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
    raise TypeError('Expected a plain tensor state_dict; checkpoint format requires review')
# These are the official load_pretrained native-transformer rename rules.
rules = {
    r'self_attn\._impl\.Wqkv\.': 'self_attn.in_proj_',
    r'self_attn\.Wqkv\.': 'self_attn.in_proj_',
    r'self_attn\._impl\.out_proj\.': 'self_attn.out_proj.',
}
converted = {}
for key, value in state.items():
    for pattern, replacement in rules.items():
        key = re.sub(pattern, replacement, key)
    if key in converted:
        raise ValueError(f'Duplicate checkpoint key after conversion: {key}')
    converted[key] = value
current = model.state_dict()
compatible = {k: v for k, v in converted.items() if k in current and v.shape == current[k].shape}
core_prefixes = ('encoder.', 'value_encoder.', 'transformer_encoder.')
required = [k for k in current if k.startswith(core_prefixes)]
missing_core = [k for k in required if k not in compatible]
load_report = {
    'required_embedding_keys': len(required), 'missing_embedding_keys': missing_core,
    'loaded_keys': sorted(compatible),
    'unused_or_incompatible_checkpoint_keys': sorted(set(converted) - set(compatible)),
    'model_keys_not_loaded': sorted(set(current) - set(compatible)),
    'note': 'All parameters used by _encode must load; decoder heads are unused in this embedding run.',
}
write_json(out / 'weight_loading.json', load_report)
if missing_core:
    raise RuntimeError(f'Embedding weights did not fully load: {missing_core[:10]}')
model.load_state_dict(compatible, strict=False)
parameter_count_total = sum(p.numel() for p in model.parameters())
parameter_count_encoder = sum(p.numel() for name, p in model.named_parameters() if name.startswith(core_prefixes))
del state, converted, current, compatible
gc.collect()
model.float().to('cuda').eval()
torch.cuda.synchronize()
model_loading_seconds = time.perf_counter() - t0
print(f'[Model] embedding weights loaded: {len(required)}/{len(required)}; output_dim={model_args["embsize"]}', flush=True)

collator = DataCollator(do_padding=True, pad_token_id=pad_id, pad_value=pad_value,
    do_mlm=False, do_binning=True, max_length=opt.max_length, sampling=True, keep_first_n_tokens=1)

def make_batch(start, end):
    examples = []
    for i in range(start, end):
        nz = np.flatnonzero(matrix[i])
        genes = np.concatenate([np.array([cls_id], dtype=np.int64), gene_ids[nz]])
        values = np.concatenate([np.array([pad_value], dtype=np.float32), matrix[i, nz]])
        examples.append({'genes': torch.from_numpy(genes), 'expressions': torch.from_numpy(values)})
    return collator(examples)

def encode_batch(batch):
    genes = batch['gene'].to('cuda')
    values = batch['expr'].to('cuda')
    encoded = model._encode(genes, values, src_key_padding_mask=genes.eq(pad_id))
    result = torch.nn.functional.normalize(encoded[:, 0, :], p=2, dim=1)
    return result.cpu().numpy()

# Warm the CUDA execution path, then reset random seeds before the measured run.
seed_all(opt.seed)
t0 = time.perf_counter()
with torch.inference_mode():
    warm_batch = make_batch(0, min(opt.batch_size, n))
    warm_result = encode_batch(warm_batch)
torch.cuda.synchronize()
warmup_seconds = time.perf_counter() - t0
del warm_batch, warm_result
model.cur_gene_token_embs = None
gc.collect()
seed_all(opt.seed)
torch.cuda.reset_peak_memory_stats()
embeddings = np.empty((n, model_args['embsize']), dtype=np.float32)
tokens_saved = np.full((n, opt.max_length), pad_id, dtype=np.int64)
values_saved = np.full((n, opt.max_length), pad_value, dtype=np.float32)
batch_records = []
sampler = ResourceSampler()
sampler.start()
torch.cuda.synchronize()
t0 = time.perf_counter()
try:
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, n, opt.batch_size)):
            end = min(start + opt.batch_size, n)
            batch = make_batch(start, end)
            length = batch['gene'].shape[1]
            tokens_saved[start:end, :length] = batch['gene'].numpy()
            values_saved[start:end, :length] = batch['expr'].numpy()
            embeddings[start:end] = encode_batch(batch)
            batch_records.append({'batch': batch_index, 'start': start, 'end': end,
                                  'batch_size': end - start, 'padded_token_length': length})
            if batch_index == 0:
                print(f'[Model input] gene IDs={tuple(batch["gene"].shape)}, expression bins={tuple(batch["expr"].shape)}', flush=True)
            if (batch_index + 1) % 25 == 0 or end == n:
                print(f'[Inference] {end}/{n} cells', flush=True)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t0
finally:
    sampler.stop()
    pd.DataFrame(sampler.rows).to_csv(out / 'resources.csv', index=False)

if not np.isfinite(embeddings).all():
    raise FloatingPointError('Embeddings contain NaN/Inf')
norms = np.linalg.norm(embeddings, axis=1)
if not np.allclose(norms, 1, atol=1e-4):
    raise ValueError('Embedding L2 norms are not approximately 1')
peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
t0 = time.perf_counter()
np.save(out / 'embeddings.npy', embeddings)
np.savez_compressed(out / 'tokenized_inputs.npz', gene_ids=tokens_saved, expression_bins=values_saved)
pd.DataFrame(batch_records).to_csv(out / 'batch_shapes.tsv', sep='\t', index=False)
itos = vocab.get_itos()
used_first = int(cell_metadata['tokens_used_including_cls'].iloc[0])
pd.DataFrame({
    'position': np.arange(min(20, used_first)),
    'gene_id': tokens_saved[0, :min(20, used_first)],
    'gene_name': [itos[int(i)] for i in tokens_saved[0, :min(20, used_first)]],
    'expression_bin': values_saved[0, :min(20, used_first)],
}).to_csv(out / 'first_cell_token_example.tsv', sep='\t', index=False)
output_save_seconds = time.perf_counter() - t0
summary = {
    'status': 'complete', 'task': 'pretrained cell embeddings', 'split': opt.split,
    'cells': n, 'matched_genes': int(gene_mask.sum()), 'embedding_shape': list(embeddings.shape),
    'batch_size': opt.batch_size, 'max_length_including_cls': opt.max_length,
    'precision': 'FP32', 'seed': opt.seed,
    'parameter_count_constructed_model': parameter_count_total,
    'parameter_count_embedding_encoder': parameter_count_encoder,
    'input_preparation_seconds': input_preparation_seconds,
    'model_loading_seconds': model_loading_seconds, 'warmup_seconds': warmup_seconds,
    'inference_seconds': inference_seconds, 'cells_per_second': n / inference_seconds,
    'output_save_seconds': output_save_seconds,
    'gpu_peak_allocated_gib': peak_allocated / GIB,
    'gpu_peak_reserved_gib': peak_reserved / GIB,
    'python_process_rss_peak_observed_gib': max(r['process_rss_bytes'] for r in sampler.rows) / GIB,
    'resource_sampling_interval_seconds': 0.2,
    'length_sampling_cells': int((nnz + 1 > opt.max_length).sum()),
    'tokens_before_sampling_min_mean_max': [int((nnz + 1).min()), float((nnz + 1).mean()), int((nnz + 1).max())],
    'tokens_after_sampling_min_mean_max': [int(np.minimum(nnz + 1, opt.max_length).min()), float(np.minimum(nnz + 1, opt.max_length).mean()), int(np.minimum(nnz + 1, opt.max_length).max())],
    'embedding_l2_norm_min_max': [float(norms.min()), float(norms.max())],
    'accuracy_evaluated': False, 'weights_updated': False,
    'timing_scope': configuration['timed_scope'],
    'memory_notes': 'CUDA allocator peaks are process-local PyTorch counters, not total device usage. Reserved memory can include warmup cache. RSS is sampled, not an exact peak. Spark uses shared memory; do not sum RSS and CUDA counters as independent physical pools.',
}
write_json(out / 'summary.json', summary)
print('\n' + json.dumps({key: summary[key] for key in [
    'cells', 'embedding_shape', 'batch_size', 'max_length_including_cls', 'precision',
    'inference_seconds', 'cells_per_second', 'gpu_peak_allocated_gib',
    'gpu_peak_reserved_gib', 'python_process_rss_peak_observed_gib', 'length_sampling_cells',
]}, indent=2), flush=True)
print('SCGPT_INFERENCE_COMPLETE', flush=True)
print('Results:', out, flush=True)
PY

cp -- "$root/scripts/01_inference.py" "$root/results/$run_id/"
cp -- "$runner" "$root/results/$run_id/"
sudo docker inspect "$container" --format 'Image={{.Config.Image}} ImageID={{.Image}}' \
    > "$root/results/$run_id/container_image.txt"
command=(sudo docker exec "$container" python -u /workspace/scgpt/scripts/01_inference.py \
    --out "/workspace/scgpt/results/$run_id" "$@")
printf '%q ' "${command[@]}" > "$root/results/$run_id/run_command.sh"
printf '\n' >> "$root/results/$run_id/run_command.sh"
"${command[@]}" 2>&1 | tee "$root/results/$run_id/run.log"
printf '\nHost results: %s\n' "$root/results/$run_id"
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/02_finetune.py — 원문 전문 (37,835 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/02_finetune.py", "sha256": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""scGPT pancreas classification: supervised fine-tuning and saved-bundle prediction.

Run train through scgpt_finetune.sh. Reload example (inside the same container):
python predict.py --mode predict --bundle /path/to/epoch_001 \
    --input /path/to/demo_test.h5ad --out /path/to/new_prediction_results
No package installation or test-set training is performed.
"""
import argparse
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer import GeneVocab

ROOT = Path('/workspace/scgpt')
GIB = 1024 ** 3
CORE = ('encoder.', 'value_encoder.', 'transformer_encoder.')
TRAINABLE = CORE + ('cls_decoder.',)
MEMORY_NOTES = ('CUDA peaks are process-local PyTorch allocator counters, not total device usage. '
                'RSS is sampled every 0.2 s, not an exact peak. Spark uses shared memory: '
                'do not add RSS and CUDA as independent physical memory pools.')


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResourceSampler:
    def __init__(self):
        self.process = psutil.Process()
        self.rows = []
        self.event = threading.Event()
        self.lock = threading.Lock()

    def sample(self):
        with self.lock:
            self.rows.append({
                'elapsed_seconds': time.perf_counter() - self.start_time,
                'process_rss_bytes': self.process.memory_info().rss,
                'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'cuda_reserved_bytes': torch.cuda.memory_reserved(),
                'system_available_bytes': psutil.virtual_memory().available,
            })

    def loop(self):
        while not self.event.wait(0.2):
            self.sample()

    def start(self):
        self.start_time = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()
        self.sample()

    def metrics(self):
        return {
            'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated() / GIB,
            'gpu_peak_reserved_gib': torch.cuda.max_memory_reserved() / GIB,
            'python_process_rss_peak_observed_gib': max(r['process_rss_bytes'] for r in self.rows) / GIB,
        }


def preserve_environment(out):
    source = out / 'source_snapshot'
    source.mkdir(exist_ok=True)
    for name, obj in [('model.py', TransformerModel), ('data_collator.py', DataCollator),
                      ('preprocess.py', binning), ('gene_tokenizer.py', GeneVocab)]:
        shutil.copy2(inspect.getfile(obj), source / name)
    packages = {p: metadata.version(p) for p in
                ['torch', 'scgpt', 'numpy', 'anndata', 'scanpy', 'scipy', 'pandas', 'scikit-learn', 'psutil']}
    direct_url = metadata.distribution('scgpt').read_text('direct_url.json')
    device = torch.cuda.get_device_properties(0)
    write_json(out / 'environment.json', {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'architecture': platform.machine(), 'packages': packages,
        'scgpt_install_source': json.loads(direct_url) if direct_url else None,
        'cuda_version': torch.version.cuda, 'gpu_name': device.name,
        'gpu_compute_capability': [device.major, device.minor],
        'cuda_reported_total_memory_gib': device.total_memory / GIB,
        'system_total_memory_gib': psutil.virtual_memory().total / GIB,
        'torch_cpu_threads': torch.get_num_threads(),
    })
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (out / 'requirements_freeze.txt').write_text(freeze.stdout)


def choose_training_subset(pool, y, requested, seed):
    """Proportional subset with at least one cell from every training class."""
    if requested == 0 or requested == len(pool):
        return np.sort(pool)
    classes, counts = np.unique(y[pool], return_counts=True)
    if requested < len(classes) or requested > len(pool):
        raise ValueError(f'n-cells must be 0 (all) or {len(classes)}..{len(pool)}')
    ideal = counts * (requested / len(pool))
    quota = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while quota.sum() < requested:
        available = np.flatnonzero(quota < counts)
        k = available[np.argmax((ideal - quota)[available])]
        quota[k] += 1
    while quota.sum() > requested:
        available = np.flatnonzero(quota > 1)
        k = available[np.argmax((quota - ideal)[available])]
        quota[k] -= 1
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(pool[y[pool] == c], int(q), replace=False)
                               for c, q in zip(classes, quota)])
    return np.sort(selected)


def extract_matrix(adata, genes):
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Cell IDs and gene names must be unique')
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise ValueError(f'Missing {len(missing)} training genes, examples: {sorted(missing)[:5]}')
    selected = adata[:, genes]
    matrix = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
    matrix = matrix.astype(np.float32, copy=True)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('Expression values must be finite and nonnegative')
    if np.any(np.count_nonzero(matrix, axis=1) == 0):
        raise ValueError('A cell has no nonzero expression after vocabulary filtering')
    return matrix


class Tokenizer:
    def __init__(self, matrix, gene_names, vocab, config):
        self.matrix = matrix
        self.gene_ids = np.asarray(vocab(gene_names), dtype=np.int64)
        self.pad_id = vocab[config['pad_token']]
        self.cls_id = vocab['<cls>']
        self.pad_value = config['pad_value']
        self.max_length = config['max_length_including_cls']
        self.collator = DataCollator(do_padding=True, pad_token_id=self.pad_id,
            pad_value=self.pad_value, do_mlm=False, do_binning=True,
            max_length=self.max_length, sampling=True, keep_first_n_tokens=1)

    def batch(self, rows):
        examples = []
        for i in rows:
            nz = np.flatnonzero(self.matrix[i])
            genes = np.concatenate([np.array([self.cls_id], dtype=np.int64), self.gene_ids[nz]])
            values = np.concatenate([np.array([self.pad_value], dtype=np.float32), self.matrix[i, nz]])
            examples.append({'genes': torch.from_numpy(genes), 'expressions': torch.from_numpy(values)})
        return self.collator(examples)

    def empty_recording(self, n):
        return (np.full((n, self.max_length), self.pad_id, dtype=np.int64),
                np.full((n, self.max_length), self.pad_value, dtype=np.float32))

    def cache(self, rows, batch_size):
        genes, values = self.empty_recording(len(rows))
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch = self.batch(rows[start:end])
            length = batch['gene'].shape[1]
            genes[start:end, :length] = batch['gene'].numpy()
            values[start:end, :length] = batch['expr'].numpy()
        return genes, values


def cached_batch(cache, start, end, pad_id):
    genes, values = cache
    length = int(np.count_nonzero(genes[start:end] != pad_id, axis=1).max())
    return {'gene': torch.from_numpy(genes[start:end, :length]),
            'expr': torch.from_numpy(values[start:end, :length])}


def classify(model, batch, pad_id):
    genes = batch['gene'].to('cuda')
    values = batch['expr'].to('cuda')
    encoded = model._encode(genes, values, src_key_padding_mask=genes.eq(pad_id))
    # Official classification path uses raw CLS, not L2-normalized embeddings.
    return model.cls_decoder(encoded[:, 0, :])


def load_pretrained_backbone(model, checkpoint, out):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError('Expected a plain tensor state_dict')
    converted = {}
    for key, value in state.items():
        for pattern, replacement in [
            (r'self_attn\._impl\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\._impl\.out_proj\.', 'self_attn.out_proj.'),
        ]:
            key = re.sub(pattern, replacement, key)
        if key in converted:
            raise ValueError(f'Duplicate renamed checkpoint key: {key}')
        converted[key] = value
    current = model.state_dict()
    required = [k for k in current if k.startswith(CORE)]
    loaded = {k: converted[k] for k in required
              if k in converted and converted[k].shape == current[k].shape}
    missing = sorted(set(required) - set(loaded))
    write_json(out / 'weight_loading.json', {
        'required_backbone_keys': len(required), 'loaded_backbone_keys': len(loaded),
        'missing_backbone_keys': missing, 'loaded_keys': sorted(loaded),
        'classifier': 'new random initialization; pretrained classifier not used',
        'unused_checkpoint_keys': sorted(set(converted) - set(loaded)),
    })
    if missing:
        raise RuntimeError(f'Incomplete pretrained backbone: {missing[:10]}')
    model.load_state_dict(loaded, strict=False)


def evaluate(model, cache, y, cell_ids, labels, pad_id, batch_size, out):
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    logits_all = np.empty((len(y), len(labels)), dtype=np.float32)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sampler = ResourceSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(y), batch_size):
                end = min(start + batch_size, len(y))
                batch = cached_batch(cache, start, end, pad_id)
                logits_all[start:end] = classify(model, batch, pad_id).cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
    finally:
        sampler.stop()
        pd.DataFrame(sampler.rows).to_csv(out / 'resources.csv', index=False)
    if not np.isfinite(logits_all).all():
        raise FloatingPointError('Non-finite prediction logits')
    probabilities = torch.from_numpy(logits_all).softmax(dim=1).numpy()
    predicted = logits_all.argmax(axis=1)
    known = y >= 0
    metrics = {
        'cells_total': len(y), 'cells_evaluated_known_labels': int(known.sum()),
        'cells_without_known_training_label': int((~known).sum()),
        'forward_seconds': seconds, 'cells_per_second': len(y) / seconds,
        'timing_scope': 'cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving',
        **sampler.metrics(),
    }
    if known.any():
        true, pred = y[known], predicted[known]
        metrics.update({
            'loss': float(torch.nn.functional.cross_entropy(torch.from_numpy(logits_all[known]),
                                                           torch.from_numpy(true)).item()),
            'accuracy': float(accuracy_score(true, pred)),
            'macro_f1_all_training_classes': float(f1_score(true, pred, labels=np.arange(len(labels)), average='macro', zero_division=0)),
            'macro_f1_present_true_classes': float(f1_score(true, pred, labels=np.unique(true), average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(true, pred, average='weighted', zero_division=0)),
        })
        write_json(out / 'classification_report.json', classification_report(
            true, pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0))
        pd.DataFrame(confusion_matrix(true, pred, labels=np.arange(len(labels))),
                     index=labels, columns=labels).rename_axis('true_label').to_csv(out / 'confusion_matrix.csv')
    pd.DataFrame({
        'cell_id': cell_ids, 'true_label_id': y,
        'true_label': [labels[i] if i >= 0 else '<unknown_or_unlabeled>' for i in y],
        'predicted_label_id': predicted, 'predicted_label': [labels[i] for i in predicted],
        'max_probability': probabilities.max(axis=1), 'included_in_metrics': known,
    }).to_csv(out / 'predictions.tsv', sep='\t', index=False)
    np.savez_compressed(out / 'logits_probabilities.npz', logits=logits_all, probabilities=probabilities)
    write_json(out / 'metrics.json', metrics)
    model.cur_gene_token_embs = None
    return metrics, logits_all


def rng_state():
    np_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {'algorithm': np_state[0], 'keys': np_state[1].tolist(),
                  'position': int(np_state[2]), 'has_gauss': int(np_state[3]), 'cached_gaussian': float(np_state[4])},
        'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
    }


def train(opt):
    out = opt.out
    pretrained = ROOT / 'models/whole-human'
    args = json.loads((pretrained / 'args.json').read_text())
    if (args.get('n_bins') != 51 or args.get('input_style') != 'binned'
            or args.get('input_emb_style', 'continuous') != 'continuous' or args.get('pre_norm', False)):
        raise ValueError('Requires the inspected 51-bin continuous-value post-norm checkpoint')
    vocab = GeneVocab.from_file(pretrained / 'vocab.json')
    for token in [args['pad_token'], '<cls>', '<eoc>']:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[args['pad_token']])
    input_path = ROOT / 'data/pancreas_demo/demo_train.h5ad'
    t0 = time.perf_counter()
    data = ad.read_h5ad(input_path)
    if data.shape != (10600, 3000) or 'Celltype' not in data.obs or data.obs['Celltype'].isna().any():
        raise ValueError('Input differs from the inspected 10600 x 3000 labeled training file')
    label_values = data.obs['Celltype'].astype(str).to_numpy()
    labels = sorted(set(label_values))
    if len(labels) != 13:
        raise ValueError(f'Expected 13 training classes; found {len(labels)}')
    label_to_id = {name: i for i, name in enumerate(labels)}
    y = np.asarray([label_to_id[x] for x in label_values], dtype=np.int64)
    all_rows = np.arange(data.n_obs)
    pool, validation = train_test_split(all_rows, test_size=opt.validation_fraction,
                                        random_state=opt.seed, stratify=y)
    validation = np.sort(validation)
    training = choose_training_subset(pool, y, opt.n_cells, opt.seed)
    if len(np.intersect1d(training, validation)):
        raise RuntimeError('Training and validation overlap')
    if set(y[training]) != set(range(len(labels))) or set(y[validation]) != set(range(len(labels))):
        raise ValueError('Every class must be represented in training and validation')
    genes = [str(g) for g in data.var_names if g in vocab]
    if len(genes) != 2999:
        raise ValueError(f'Expected 2999 vocabulary-matched genes, found {len(genes)}')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    roles = np.full(len(y), 'train_pool_unused', dtype=object)
    roles[validation] = 'validation'
    roles[training] = 'train_used'
    pd.DataFrame({'original_row_index': all_rows, 'cell_id': cell_ids,
                  'Celltype': label_values, 'label_id': y, 'role': roles}).to_csv(
                      out / 'split_assignments.tsv', sep='\t', index=False)
    pd.crosstab(pd.Series(label_values, name='Celltype'), pd.Series(roles, name='role')).to_csv(
        out / 'split_class_counts.csv')
    input_seconds = time.perf_counter() - t0
    del data
    preprocess = {
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'], 'n_bins': 51,
        'max_length_including_cls': opt.max_length,
        'additional_normalize_total': False, 'additional_log1p': False,
        'expression_provenance': 'Supplied h5ad values; no raw layer or preprocessing metadata',
        'tokenization': 'nonzero genes only; prepend CLS; official per-cell quantile binning; random overlength sampling; dynamic padding',
        'pooling': 'raw CLS features into official cls_decoder; no L2 normalization',
        'prediction_genes': 'require every saved training gene; reorder to training gene order; ignore additional genes',
    }
    constructor = {
        'ntoken': len(vocab), 'd_model': args['embsize'], 'nhead': args['nheads'],
        'd_hid': args['d_hid'], 'nlayers': args['nlayers'],
        'nlayers_cls': args['n_layers_cls'], 'n_cls': len(labels), 'dropout': args['dropout'],
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'],
        'do_mvc': False, 'do_dab': False, 'use_batch_labels': False,
        'domain_spec_batchnorm': False, 'input_emb_style': 'continuous', 'cell_emb_style': 'cls',
        'explicit_zero_prob': False, 'use_fast_transformer': False, 'pre_norm': False,
    }
    config = {
        'task': 'supervised cell-type classification', 'seed': opt.seed,
        'train_source_cells': len(y), 'training_pool_cells': len(pool), 'training_cells': len(training),
        'validation_cells': len(validation), 'classes': len(labels), 'epochs': opt.epochs,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'autocast': False, 'tf32': False,
        'optimizer': {'name': 'AdamW', 'lr': opt.lr, 'weight_decay': opt.weight_decay,
                      'betas': [0.9, 0.999], 'eps': 1e-8},
        'scheduler': None, 'gradient_clip_max_norm': 1.0,
        'loss': 'unweighted cross entropy', 'trainable_prefixes': list(TRAINABLE),
        'unused_expression_decoder': 'frozen and bypassed',
        'classifier': 'new official ClsDecoder; all classifier layers trainable',
        'split': 'stratified by Celltype before subsampling; no donor/batch metadata available',
        'validation_limit': 'within-file cell split; not donor-independent validation',
        'validation_tokens': 'generated once with seed+10000, saved and reused',
        'epoch_seed': 'seed+epoch; reinitializes RNG before each epoch',
        'checkpoint_selection': 'highest validation macro-F1 over all 13 training classes',
        'test_used_for_training_or_selection': False,
        'training_warmup_steps': 0,
        'training_timing_scope': 'tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization',
        'training_timer_excludes': 'imports, h5ad/model loading, validation token cache, validation, disk saving, reload verification',
        'memory_notes': MEMORY_NOTES, 'preprocessing': preprocess,
        'reproducibility': 'Seeds, exact tokens and environment saved; bitwise reproducibility across platforms is not guaranteed',
    }
    for filename, obj in [('run_config.json', config), ('preprocessing.json', preprocess),
                          ('model_config.json', constructor), ('labels.json', {'labels': labels, 'label_to_id': label_to_id}),
                          ('genes.json', genes), ('vocab.json', vocab.get_stoi())]:
        write_json(out / filename, obj)
    shutil.copy2(pretrained / 'args.json', out / 'pretrained_args.json')
    write_json(out / 'provenance.json', {
        'input': {'path': str(input_path), 'sha256': sha256(input_path)},
        'pretrained_checkpoint': {'path': str(pretrained / 'best_model.pt'), 'sha256': sha256(pretrained / 'best_model.pt')},
        'script_sha256': sha256(__file__),
        'tutorial': 'https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html',
        'dataset_source': 'TOSICA author-distributed Figshare demo_train.h5ad (file 39009503)',
    })
    print(f'[Split] train={len(training)}, validation={len(validation)}, classes={len(labels)}, matched_genes={len(genes)}', flush=True)
    print('[Input] expression values as supplied -> 51 bins; no additional normalization/log1p', flush=True)
    t0 = time.perf_counter()
    seed_all(opt.seed)
    model = TransformerModel(**constructor, vocab=vocab)
    load_pretrained_backbone(model, pretrained / 'best_model.pt', out)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE))
    model.float().to('cuda')
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=opt.lr, weight_decay=opt.weight_decay,
                                 betas=(0.9, 0.999), eps=1e-8)
    torch.cuda.synchronize()
    model_loading_seconds = time.perf_counter() - t0
    probes = ['transformer_encoder.layers.0.self_attn.in_proj_weight', 'cls_decoder.out_layer.weight']
    initial = {k: model.state_dict()[k].detach().cpu().clone() for k in probes}
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed + 10000)
    t0 = time.perf_counter()
    val_cache = tokenizer.cache(validation, opt.batch_size)
    cache_seconds = time.perf_counter() - t0
    np.savez_compressed(out / 'validation_tokenized_inputs.npz', gene_ids=val_cache[0],
                        expression_bins=val_cache[1], original_row_indices=validation, labels=y[validation])
    print('[Validation] fixed input tokens saved', flush=True)
    opt.model_out.mkdir(parents=True, exist_ok=False)
    epoch_summaries = []
    best = None
    for epoch in range(1, opt.epochs + 1):
        epoch_out = out / f'epoch_{epoch:03d}'
        epoch_out.mkdir()
        seed_all(opt.seed + epoch)
        order = np.random.default_rng(opt.seed + epoch).permutation(training)
        saved_genes, saved_values = tokenizer.empty_recording(len(order))
        steps = []
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler = ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        loss_sum, correct = 0.0, 0
        try:
            for step, start in enumerate(range(0, len(order), opt.batch_size), 1):
                end = min(start + opt.batch_size, len(order))
                rows = order[start:end]
                batch = tokenizer.batch(rows)
                length = batch['gene'].shape[1]
                saved_genes[start:end, :length] = batch['gene'].numpy()
                saved_values[start:end, :length] = batch['expr'].numpy()
                target = torch.from_numpy(y[rows]).to('cuda')
                optimizer.zero_grad(set_to_none=True)
                logits = classify(model, batch, tokenizer.pad_id)
                loss = torch.nn.functional.cross_entropy(logits, target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Non-finite training loss at epoch {epoch}, step {step}')
                loss.backward()
                if epoch == 1 and step == 1:
                    missing_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
                    if missing_grad:
                        raise RuntimeError(f'Trainable parameters without gradients: {missing_grad}')
                    print(f'[Model input] gene/value={tuple(batch["gene"].shape)}, labels={tuple(target.shape)}, logits={tuple(logits.shape)}', flush=True)
                    print('[Backward] all trainable parameters received gradients', flush=True)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_value = float(loss.detach().item())
                batch_correct = int((logits.detach().argmax(dim=1) == target).sum().item())
                loss_sum += loss_value * len(rows)
                correct += batch_correct
                steps.append({'epoch': epoch, 'step': step, 'cells': len(rows), 'padded_length': length,
                              'loss': loss_value, 'correct': batch_correct, 'grad_norm_before_clip': float(grad_norm.item())})
                if step % 25 == 0 or end == len(order):
                    print(f'[Train {epoch}/{opt.epochs}] {end}/{len(order)} cells, online_loss={loss_sum/end:.4f}', flush=True)
                del logits, loss, target
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - t0
        finally:
            sampler.stop()
            pd.DataFrame(sampler.rows).to_csv(epoch_out / 'training_resources.csv', index=False)
        training_metrics = sampler.metrics()
        optimizer.zero_grad(set_to_none=True)
        model.cur_gene_token_embs = None
        pd.DataFrame(steps).to_csv(epoch_out / 'training_steps.csv', index=False)
        np.savez_compressed(epoch_out / 'training_tokenized_inputs.npz', gene_ids=saved_genes,
                            expression_bins=saved_values, original_row_indices=order, labels=y[order])
        del saved_genes, saved_values
        deltas = {k: float((model.state_dict()[k].detach().cpu() - initial[k]).abs().max().item()) for k in probes}
        if not all(np.isfinite(v) and v > 0 for v in deltas.values()):
            raise RuntimeError(f'Expected encoder and classifier weights to change: {deltas}')
        validation_metrics, val_logits = evaluate(model, val_cache, y[validation], cell_ids[validation],
            labels, tokenizer.pad_id, opt.batch_size, epoch_out / 'validation')
        bundle = opt.model_out / f'epoch_{epoch:03d}'
        bundle.mkdir()
        save_start = time.perf_counter()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, bundle / 'model.pt')
        torch.save({'epoch_completed': epoch, 'optimizer': optimizer.state_dict(),
                    'rng_state': rng_state(), 'run_config': config}, bundle / 'training_state.pt')
        for filename in ['model_config.json', 'preprocessing.json', 'labels.json', 'genes.json', 'vocab.json',
                         'run_config.json', 'pretrained_args.json', 'environment.json', 'requirements_freeze.txt']:
            shutil.copy2(out / filename, bundle / filename)
        shutil.copy2(__file__, bundle / 'predict.py')
        (bundle / 'README.txt').write_text(
            'Saved scGPT encoder + trained cell-type classifier. model.pt is a plain state_dict.\n'
            'Run prediction inside the scgpt-spark container using its existing environment:\n'
            'python /path/to/epoch_001/predict.py --mode predict --bundle /path/to/epoch_001 '
            '--input /workspace/scgpt/data/pancreas_demo/demo_test.h5ad --out /workspace/scgpt/results/NEW_NAME\n'
            'All input cells are retained. Unknown labels are excluded from closed-set metrics and reported separately.\n'
            'No unknown-class detector is trained; softmax confidence is not an unknown detector.\n'
            'training_state.pt stores optimizer and RNG states for recovery/development; this script has no resume CLI.\n'
            'Validation is a cell split within the supplied train file, not a donor-independent benchmark.\n')
        save_seconds = time.perf_counter() - save_start
        # Verify strict reload and identical predictions from the saved weights.
        reloaded = TransformerModel(**constructor, vocab=vocab)
        reloaded.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        reloaded.float().to('cuda').eval()
        end = min(opt.batch_size, len(validation))
        with torch.inference_mode():
            reload_logits = classify(reloaded, cached_batch(val_cache, 0, end, tokenizer.pad_id), tokenizer.pad_id).cpu().numpy()
        reload_error = float(np.max(np.abs(reload_logits - val_logits[:end])))
        if not np.allclose(reload_logits, val_logits[:end], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f'Saved-model reload mismatch: max error {reload_error}')
        del reloaded
        gc.collect()
        write_json(bundle / 'manifest.json', {
            'model_sha256': sha256(bundle / 'model.pt'),
            'training_state_sha256': sha256(bundle / 'training_state.pt'),
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
            'results_directory': str(out),
        })
        record = {
            'epoch': epoch, 'training_cells': len(training), 'training_seconds': training_seconds,
            'training_cells_per_second': len(training) / training_seconds,
            'online_training_loss': loss_sum / len(training), 'online_training_accuracy': correct / len(training),
            **training_metrics, 'validation': validation_metrics, 'weight_probe_max_abs_deltas': deltas,
            'model_bundle': str(bundle), 'checkpoint_save_seconds': save_seconds,
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
        }
        write_json(epoch_out / 'summary.json', record)
        epoch_summaries.append(record)
        score = validation_metrics['macro_f1_all_training_classes']
        if best is None or score > best['validation_macro_f1']:
            best = {'epoch': epoch, 'validation_macro_f1': score, 'model_bundle': str(bundle)}
            write_json(opt.model_out / 'best.json', best)
        print(f'[Validation] accuracy={validation_metrics["accuracy"]:.4f}, macro-F1={score:.4f}', flush=True)
        print(f'[Saved] {bundle}; reload check passed', flush=True)
    summary = {
        'status': 'complete', 'task': 'scGPT supervised cell-type fine-tuning',
        'training_cells': len(training), 'validation_cells': len(validation), 'classes': len(labels),
        'matched_genes': len(genes), 'epochs': opt.epochs, 'batch_size': opt.batch_size,
        'max_length_including_cls': opt.max_length, 'precision': 'FP32', 'seed': opt.seed,
        'parameter_count_constructed_model': sum(p.numel() for p in model.parameters()),
        'parameter_count_trainable': sum(p.numel() for p in trainable),
        'parameter_count_encoder': sum(p.numel() for name, p in model.named_parameters() if name.startswith(CORE)),
        'input_preparation_seconds': input_seconds, 'model_loading_seconds': model_loading_seconds,
        'validation_tokenization_seconds': cache_seconds,
        'training_seconds_total': sum(e['training_seconds'] for e in epoch_summaries),
        'training_gpu_peak_allocated_gib': max(e['gpu_peak_allocated_gib'] for e in epoch_summaries),
        'training_gpu_peak_reserved_gib': max(e['gpu_peak_reserved_gib'] for e in epoch_summaries),
        'training_process_rss_peak_observed_gib': max(e['python_process_rss_peak_observed_gib'] for e in epoch_summaries),
        'best_checkpoint': best, 'validation': epoch_summaries[best['epoch'] - 1]['validation'],
        'weights_updated': True, 'encoder_and_classifier_updated': True, 'saved_model_reload_verified': True,
        'test_evaluated': False, 'epochs_detail': epoch_summaries,
        'timing_scope': config['training_timing_scope'], 'memory_notes': MEMORY_NOTES,
        'interpretation': 'Small-run fine-tuning and validation check; not a full-data or donor-independent accuracy benchmark',
    }
    write_json(out / 'summary.json', summary)
    print('\n' + json.dumps({k: summary[k] for k in [
        'status', 'training_cells', 'validation_cells', 'epochs', 'training_seconds_total',
        'training_gpu_peak_allocated_gib', 'training_process_rss_peak_observed_gib',
        'best_checkpoint', 'saved_model_reload_verified', 'test_evaluated']}, indent=2), flush=True)
    print('SCGPT_FINETUNE_COMPLETE', flush=True)


def predict(opt):
    if opt.bundle is None or opt.input is None:
        raise ValueError('Prediction requires --bundle and --input')
    bundle = opt.bundle
    constructor = json.loads((bundle / 'model_config.json').read_text())
    preprocess = json.loads((bundle / 'preprocessing.json').read_text())
    labels = json.loads((bundle / 'labels.json').read_text())['labels']
    label_to_id = {name: i for i, name in enumerate(labels)}
    genes = json.loads((bundle / 'genes.json').read_text())
    vocab = GeneVocab.from_file(bundle / 'vocab.json')
    vocab.set_default_index(vocab[preprocess['pad_token']])
    data = ad.read_h5ad(opt.input)
    if data.n_obs == 0:
        raise ValueError('Input has no cells')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    originals = (data.obs['Celltype'].astype('string').fillna('<unlabeled>').astype(str).to_numpy()
                 if 'Celltype' in data.obs else np.full(data.n_obs, '<unlabeled>'))
    y = np.asarray([label_to_id.get(label, -1) for label in originals], dtype=np.int64)
    pd.DataFrame({'cell_id': cell_ids, 'original_label': originals, 'known_training_label': y >= 0}).to_csv(
        opt.out / 'input_cells.tsv', sep='\t', index=False)
    write_json(opt.out / 'prediction_config.json', {
        'bundle': str(bundle), 'model_sha256': sha256(bundle / 'model.pt'),
        'input': str(opt.input), 'input_sha256': sha256(opt.input), 'seed': opt.seed,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'preprocessing': preprocess,
        'unknown_label_counts': pd.Series(originals[y < 0]).value_counts().astype(int).to_dict(),
        'unknown_policy': 'retain all predictions; evaluate known ground-truth labels only; no unknown-class detector',
    })
    model = TransformerModel(**constructor, vocab=vocab)
    model.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    model.float().to('cuda').eval()
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed)
    t0 = time.perf_counter()
    cache = tokenizer.cache(np.arange(len(y)), opt.batch_size)
    tokenization_seconds = time.perf_counter() - t0
    np.savez_compressed(opt.out / 'tokenized_inputs.npz', gene_ids=cache[0], expression_bins=cache[1], labels=y)
    metrics, _ = evaluate(model, cache, y, cell_ids, labels, tokenizer.pad_id, opt.batch_size, opt.out)
    summary = {'status': 'complete', 'task': 'fine-tuned cell-type predictions',
               'tokenization_seconds': tokenization_seconds, 'warmup_steps': 0,
               'weights_updated': False, 'model_bundle': str(bundle), 'memory_notes': MEMORY_NOTES, **metrics}
    write_json(opt.out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print('SCGPT_PREDICTION_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-out', type=Path)
    parser.add_argument('--n-cells', type=int, default=1000, help='training subset size; 0=all training-pool cells')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=1200)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--input', type=Path)
    opt = parser.parse_args()
    if (opt.batch_size < 1 or opt.epochs < 1 or opt.n_cells < 0 or opt.max_length < 2
            or not 0 < opt.validation_fraction < 1 or not np.isfinite(opt.lr) or opt.lr <= 0
            or not np.isfinite(opt.weight_decay) or opt.weight_decay < 0 or not 0 <= opt.seed < 2**32 - 10000):
        parser.error('Invalid numeric option')
    if opt.mode == 'train' and opt.model_out is None:
        parser.error('Training requires --model-out')
    opt.out.mkdir(parents=True, exist_ok=True)
    if (opt.out / 'run_config.json').exists() or (opt.out / 'summary.json').exists() or (opt.out / 'prediction_config.json').exists():
        raise FileExistsError(f'Use a new output directory: {opt.out}')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; stopping instead of running on CPU')
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_all(opt.seed)
        preserve_environment(opt.out)
        train(opt) if opt.mode == 'train' else predict(opt)
    except Exception as exc:
        write_json(opt.out / 'failure.json', {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "scgpt/results/finetune_20261023_033701_500560/02_finetune.py", "source": "scgpt/scripts/02_finetune.py", "sha256": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/02_finetune.py", "source": "scgpt/scripts/02_finetune.py", "sha256": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/02_finetune.py", "source": "scgpt/scripts/02_finetune.py", "sha256": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/source/02_finetune.py", "source": "scgpt/scripts/02_finetune.py", "sha256": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/02_finetune.py", "source": "scgpt/scripts/02_finetune.py", "sha256": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc"} -->
<!-- RESTORE_ALIAS {"path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/02_finetune.py", "source": "scgpt/scripts/02_finetune.py", "sha256": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc"} -->

</details>

<details>
<summary>scgpt/scripts/02_run_finetune.sh — 원문 전문 (39,641 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/02_run_finetune.sh", "sha256": "22f965af22beafe155a54c31a5ecc0cd24a7190b8c1a67043f98cfa8be4e0f5a", "encoding": "utf-8", "added_final_newline": false} -->
```bash
#!/usr/bin/env bash
# scGPT step 02: supervised encoder + classifier fine-tuning.
# Defaults: 1,000 training cells, 2,120 validation cells, 1 epoch, batch 8, FP32.
# Run on Spark as the normal user: bash scgpt_finetune.sh
# Saves source, inputs, metrics, models, optimizer states and reload/prediction code.
set -euo pipefail
root="$HOME/ai-benchmark/scgpt"
container="scgpt-spark"
run_id="finetune_$(date +%Y%m%d_%H%M%S)_$$"
sudo -v
mount_source=$(sudo docker inspect "$container" --format '{{range .Mounts}}{{if eq .Destination "/workspace/scgpt"}}{{.Source}}{{end}}{{end}}')
if [[ -z "$mount_source" || "$(readlink -f "$mount_source")" != "$(readlink -f "$root")" ]]; then
    echo "ERROR: expected $root mounted at /workspace/scgpt; found $mount_source" >&2
    exit 1
fi
mkdir -p "$root/scripts" "$root/results/$run_id" "$root/models/finetuned"
runner="$root/scripts/02_run_finetune.sh"
if [[ "$(readlink -f "$0")" != "$(readlink -f "$runner")" ]]; then cp -- "$0" "$runner"; fi
cat > "$root/scripts/02_finetune.py" <<'SCGPT_PYTHON'
"""scGPT pancreas classification: supervised fine-tuning and saved-bundle prediction.

Run train through scgpt_finetune.sh. Reload example (inside the same container):
python predict.py --mode predict --bundle /path/to/epoch_001 \
    --input /path/to/demo_test.h5ad --out /path/to/new_prediction_results
No package installation or test-set training is performed.
"""
import argparse
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer import GeneVocab

ROOT = Path('/workspace/scgpt')
GIB = 1024 ** 3
CORE = ('encoder.', 'value_encoder.', 'transformer_encoder.')
TRAINABLE = CORE + ('cls_decoder.',)
MEMORY_NOTES = ('CUDA peaks are process-local PyTorch allocator counters, not total device usage. '
                'RSS is sampled every 0.2 s, not an exact peak. Spark uses shared memory: '
                'do not add RSS and CUDA as independent physical memory pools.')


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResourceSampler:
    def __init__(self):
        self.process = psutil.Process()
        self.rows = []
        self.event = threading.Event()
        self.lock = threading.Lock()

    def sample(self):
        with self.lock:
            self.rows.append({
                'elapsed_seconds': time.perf_counter() - self.start_time,
                'process_rss_bytes': self.process.memory_info().rss,
                'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'cuda_reserved_bytes': torch.cuda.memory_reserved(),
                'system_available_bytes': psutil.virtual_memory().available,
            })

    def loop(self):
        while not self.event.wait(0.2):
            self.sample()

    def start(self):
        self.start_time = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()
        self.sample()

    def metrics(self):
        return {
            'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated() / GIB,
            'gpu_peak_reserved_gib': torch.cuda.max_memory_reserved() / GIB,
            'python_process_rss_peak_observed_gib': max(r['process_rss_bytes'] for r in self.rows) / GIB,
        }


def preserve_environment(out):
    source = out / 'source_snapshot'
    source.mkdir(exist_ok=True)
    for name, obj in [('model.py', TransformerModel), ('data_collator.py', DataCollator),
                      ('preprocess.py', binning), ('gene_tokenizer.py', GeneVocab)]:
        shutil.copy2(inspect.getfile(obj), source / name)
    packages = {p: metadata.version(p) for p in
                ['torch', 'scgpt', 'numpy', 'anndata', 'scanpy', 'scipy', 'pandas', 'scikit-learn', 'psutil']}
    direct_url = metadata.distribution('scgpt').read_text('direct_url.json')
    device = torch.cuda.get_device_properties(0)
    write_json(out / 'environment.json', {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'architecture': platform.machine(), 'packages': packages,
        'scgpt_install_source': json.loads(direct_url) if direct_url else None,
        'cuda_version': torch.version.cuda, 'gpu_name': device.name,
        'gpu_compute_capability': [device.major, device.minor],
        'cuda_reported_total_memory_gib': device.total_memory / GIB,
        'system_total_memory_gib': psutil.virtual_memory().total / GIB,
        'torch_cpu_threads': torch.get_num_threads(),
    })
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (out / 'requirements_freeze.txt').write_text(freeze.stdout)


def choose_training_subset(pool, y, requested, seed):
    """Proportional subset with at least one cell from every training class."""
    if requested == 0 or requested == len(pool):
        return np.sort(pool)
    classes, counts = np.unique(y[pool], return_counts=True)
    if requested < len(classes) or requested > len(pool):
        raise ValueError(f'n-cells must be 0 (all) or {len(classes)}..{len(pool)}')
    ideal = counts * (requested / len(pool))
    quota = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while quota.sum() < requested:
        available = np.flatnonzero(quota < counts)
        k = available[np.argmax((ideal - quota)[available])]
        quota[k] += 1
    while quota.sum() > requested:
        available = np.flatnonzero(quota > 1)
        k = available[np.argmax((quota - ideal)[available])]
        quota[k] -= 1
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(pool[y[pool] == c], int(q), replace=False)
                               for c, q in zip(classes, quota)])
    return np.sort(selected)


def extract_matrix(adata, genes):
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Cell IDs and gene names must be unique')
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise ValueError(f'Missing {len(missing)} training genes, examples: {sorted(missing)[:5]}')
    selected = adata[:, genes]
    matrix = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
    matrix = matrix.astype(np.float32, copy=True)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('Expression values must be finite and nonnegative')
    if np.any(np.count_nonzero(matrix, axis=1) == 0):
        raise ValueError('A cell has no nonzero expression after vocabulary filtering')
    return matrix


class Tokenizer:
    def __init__(self, matrix, gene_names, vocab, config):
        self.matrix = matrix
        self.gene_ids = np.asarray(vocab(gene_names), dtype=np.int64)
        self.pad_id = vocab[config['pad_token']]
        self.cls_id = vocab['<cls>']
        self.pad_value = config['pad_value']
        self.max_length = config['max_length_including_cls']
        self.collator = DataCollator(do_padding=True, pad_token_id=self.pad_id,
            pad_value=self.pad_value, do_mlm=False, do_binning=True,
            max_length=self.max_length, sampling=True, keep_first_n_tokens=1)

    def batch(self, rows):
        examples = []
        for i in rows:
            nz = np.flatnonzero(self.matrix[i])
            genes = np.concatenate([np.array([self.cls_id], dtype=np.int64), self.gene_ids[nz]])
            values = np.concatenate([np.array([self.pad_value], dtype=np.float32), self.matrix[i, nz]])
            examples.append({'genes': torch.from_numpy(genes), 'expressions': torch.from_numpy(values)})
        return self.collator(examples)

    def empty_recording(self, n):
        return (np.full((n, self.max_length), self.pad_id, dtype=np.int64),
                np.full((n, self.max_length), self.pad_value, dtype=np.float32))

    def cache(self, rows, batch_size):
        genes, values = self.empty_recording(len(rows))
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch = self.batch(rows[start:end])
            length = batch['gene'].shape[1]
            genes[start:end, :length] = batch['gene'].numpy()
            values[start:end, :length] = batch['expr'].numpy()
        return genes, values


def cached_batch(cache, start, end, pad_id):
    genes, values = cache
    length = int(np.count_nonzero(genes[start:end] != pad_id, axis=1).max())
    return {'gene': torch.from_numpy(genes[start:end, :length]),
            'expr': torch.from_numpy(values[start:end, :length])}


def classify(model, batch, pad_id):
    genes = batch['gene'].to('cuda')
    values = batch['expr'].to('cuda')
    encoded = model._encode(genes, values, src_key_padding_mask=genes.eq(pad_id))
    # Official classification path uses raw CLS, not L2-normalized embeddings.
    return model.cls_decoder(encoded[:, 0, :])


def load_pretrained_backbone(model, checkpoint, out):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError('Expected a plain tensor state_dict')
    converted = {}
    for key, value in state.items():
        for pattern, replacement in [
            (r'self_attn\._impl\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\._impl\.out_proj\.', 'self_attn.out_proj.'),
        ]:
            key = re.sub(pattern, replacement, key)
        if key in converted:
            raise ValueError(f'Duplicate renamed checkpoint key: {key}')
        converted[key] = value
    current = model.state_dict()
    required = [k for k in current if k.startswith(CORE)]
    loaded = {k: converted[k] for k in required
              if k in converted and converted[k].shape == current[k].shape}
    missing = sorted(set(required) - set(loaded))
    write_json(out / 'weight_loading.json', {
        'required_backbone_keys': len(required), 'loaded_backbone_keys': len(loaded),
        'missing_backbone_keys': missing, 'loaded_keys': sorted(loaded),
        'classifier': 'new random initialization; pretrained classifier not used',
        'unused_checkpoint_keys': sorted(set(converted) - set(loaded)),
    })
    if missing:
        raise RuntimeError(f'Incomplete pretrained backbone: {missing[:10]}')
    model.load_state_dict(loaded, strict=False)


def evaluate(model, cache, y, cell_ids, labels, pad_id, batch_size, out):
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    logits_all = np.empty((len(y), len(labels)), dtype=np.float32)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sampler = ResourceSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(y), batch_size):
                end = min(start + batch_size, len(y))
                batch = cached_batch(cache, start, end, pad_id)
                logits_all[start:end] = classify(model, batch, pad_id).cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
    finally:
        sampler.stop()
        pd.DataFrame(sampler.rows).to_csv(out / 'resources.csv', index=False)
    if not np.isfinite(logits_all).all():
        raise FloatingPointError('Non-finite prediction logits')
    probabilities = torch.from_numpy(logits_all).softmax(dim=1).numpy()
    predicted = logits_all.argmax(axis=1)
    known = y >= 0
    metrics = {
        'cells_total': len(y), 'cells_evaluated_known_labels': int(known.sum()),
        'cells_without_known_training_label': int((~known).sum()),
        'forward_seconds': seconds, 'cells_per_second': len(y) / seconds,
        'timing_scope': 'cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving',
        **sampler.metrics(),
    }
    if known.any():
        true, pred = y[known], predicted[known]
        metrics.update({
            'loss': float(torch.nn.functional.cross_entropy(torch.from_numpy(logits_all[known]),
                                                           torch.from_numpy(true)).item()),
            'accuracy': float(accuracy_score(true, pred)),
            'macro_f1_all_training_classes': float(f1_score(true, pred, labels=np.arange(len(labels)), average='macro', zero_division=0)),
            'macro_f1_present_true_classes': float(f1_score(true, pred, labels=np.unique(true), average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(true, pred, average='weighted', zero_division=0)),
        })
        write_json(out / 'classification_report.json', classification_report(
            true, pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0))
        pd.DataFrame(confusion_matrix(true, pred, labels=np.arange(len(labels))),
                     index=labels, columns=labels).rename_axis('true_label').to_csv(out / 'confusion_matrix.csv')
    pd.DataFrame({
        'cell_id': cell_ids, 'true_label_id': y,
        'true_label': [labels[i] if i >= 0 else '<unknown_or_unlabeled>' for i in y],
        'predicted_label_id': predicted, 'predicted_label': [labels[i] for i in predicted],
        'max_probability': probabilities.max(axis=1), 'included_in_metrics': known,
    }).to_csv(out / 'predictions.tsv', sep='\t', index=False)
    np.savez_compressed(out / 'logits_probabilities.npz', logits=logits_all, probabilities=probabilities)
    write_json(out / 'metrics.json', metrics)
    model.cur_gene_token_embs = None
    return metrics, logits_all


def rng_state():
    np_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {'algorithm': np_state[0], 'keys': np_state[1].tolist(),
                  'position': int(np_state[2]), 'has_gauss': int(np_state[3]), 'cached_gaussian': float(np_state[4])},
        'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
    }


def train(opt):
    out = opt.out
    pretrained = ROOT / 'models/whole-human'
    args = json.loads((pretrained / 'args.json').read_text())
    if (args.get('n_bins') != 51 or args.get('input_style') != 'binned'
            or args.get('input_emb_style', 'continuous') != 'continuous' or args.get('pre_norm', False)):
        raise ValueError('Requires the inspected 51-bin continuous-value post-norm checkpoint')
    vocab = GeneVocab.from_file(pretrained / 'vocab.json')
    for token in [args['pad_token'], '<cls>', '<eoc>']:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[args['pad_token']])
    input_path = ROOT / 'data/pancreas_demo/demo_train.h5ad'
    t0 = time.perf_counter()
    data = ad.read_h5ad(input_path)
    if data.shape != (10600, 3000) or 'Celltype' not in data.obs or data.obs['Celltype'].isna().any():
        raise ValueError('Input differs from the inspected 10600 x 3000 labeled training file')
    label_values = data.obs['Celltype'].astype(str).to_numpy()
    labels = sorted(set(label_values))
    if len(labels) != 13:
        raise ValueError(f'Expected 13 training classes; found {len(labels)}')
    label_to_id = {name: i for i, name in enumerate(labels)}
    y = np.asarray([label_to_id[x] for x in label_values], dtype=np.int64)
    all_rows = np.arange(data.n_obs)
    pool, validation = train_test_split(all_rows, test_size=opt.validation_fraction,
                                        random_state=opt.seed, stratify=y)
    validation = np.sort(validation)
    training = choose_training_subset(pool, y, opt.n_cells, opt.seed)
    if len(np.intersect1d(training, validation)):
        raise RuntimeError('Training and validation overlap')
    if set(y[training]) != set(range(len(labels))) or set(y[validation]) != set(range(len(labels))):
        raise ValueError('Every class must be represented in training and validation')
    genes = [str(g) for g in data.var_names if g in vocab]
    if len(genes) != 2999:
        raise ValueError(f'Expected 2999 vocabulary-matched genes, found {len(genes)}')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    roles = np.full(len(y), 'train_pool_unused', dtype=object)
    roles[validation] = 'validation'
    roles[training] = 'train_used'
    pd.DataFrame({'original_row_index': all_rows, 'cell_id': cell_ids,
                  'Celltype': label_values, 'label_id': y, 'role': roles}).to_csv(
                      out / 'split_assignments.tsv', sep='\t', index=False)
    pd.crosstab(pd.Series(label_values, name='Celltype'), pd.Series(roles, name='role')).to_csv(
        out / 'split_class_counts.csv')
    input_seconds = time.perf_counter() - t0
    del data
    preprocess = {
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'], 'n_bins': 51,
        'max_length_including_cls': opt.max_length,
        'additional_normalize_total': False, 'additional_log1p': False,
        'expression_provenance': 'Supplied h5ad values; no raw layer or preprocessing metadata',
        'tokenization': 'nonzero genes only; prepend CLS; official per-cell quantile binning; random overlength sampling; dynamic padding',
        'pooling': 'raw CLS features into official cls_decoder; no L2 normalization',
        'prediction_genes': 'require every saved training gene; reorder to training gene order; ignore additional genes',
    }
    constructor = {
        'ntoken': len(vocab), 'd_model': args['embsize'], 'nhead': args['nheads'],
        'd_hid': args['d_hid'], 'nlayers': args['nlayers'],
        'nlayers_cls': args['n_layers_cls'], 'n_cls': len(labels), 'dropout': args['dropout'],
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'],
        'do_mvc': False, 'do_dab': False, 'use_batch_labels': False,
        'domain_spec_batchnorm': False, 'input_emb_style': 'continuous', 'cell_emb_style': 'cls',
        'explicit_zero_prob': False, 'use_fast_transformer': False, 'pre_norm': False,
    }
    config = {
        'task': 'supervised cell-type classification', 'seed': opt.seed,
        'train_source_cells': len(y), 'training_pool_cells': len(pool), 'training_cells': len(training),
        'validation_cells': len(validation), 'classes': len(labels), 'epochs': opt.epochs,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'autocast': False, 'tf32': False,
        'optimizer': {'name': 'AdamW', 'lr': opt.lr, 'weight_decay': opt.weight_decay,
                      'betas': [0.9, 0.999], 'eps': 1e-8},
        'scheduler': None, 'gradient_clip_max_norm': 1.0,
        'loss': 'unweighted cross entropy', 'trainable_prefixes': list(TRAINABLE),
        'unused_expression_decoder': 'frozen and bypassed',
        'classifier': 'new official ClsDecoder; all classifier layers trainable',
        'split': 'stratified by Celltype before subsampling; no donor/batch metadata available',
        'validation_limit': 'within-file cell split; not donor-independent validation',
        'validation_tokens': 'generated once with seed+10000, saved and reused',
        'epoch_seed': 'seed+epoch; reinitializes RNG before each epoch',
        'checkpoint_selection': 'highest validation macro-F1 over all 13 training classes',
        'test_used_for_training_or_selection': False,
        'training_warmup_steps': 0,
        'training_timing_scope': 'tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization',
        'training_timer_excludes': 'imports, h5ad/model loading, validation token cache, validation, disk saving, reload verification',
        'memory_notes': MEMORY_NOTES, 'preprocessing': preprocess,
        'reproducibility': 'Seeds, exact tokens and environment saved; bitwise reproducibility across platforms is not guaranteed',
    }
    for filename, obj in [('run_config.json', config), ('preprocessing.json', preprocess),
                          ('model_config.json', constructor), ('labels.json', {'labels': labels, 'label_to_id': label_to_id}),
                          ('genes.json', genes), ('vocab.json', vocab.get_stoi())]:
        write_json(out / filename, obj)
    shutil.copy2(pretrained / 'args.json', out / 'pretrained_args.json')
    write_json(out / 'provenance.json', {
        'input': {'path': str(input_path), 'sha256': sha256(input_path)},
        'pretrained_checkpoint': {'path': str(pretrained / 'best_model.pt'), 'sha256': sha256(pretrained / 'best_model.pt')},
        'script_sha256': sha256(__file__),
        'tutorial': 'https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html',
        'dataset_source': 'TOSICA author-distributed Figshare demo_train.h5ad (file 39009503)',
    })
    print(f'[Split] train={len(training)}, validation={len(validation)}, classes={len(labels)}, matched_genes={len(genes)}', flush=True)
    print('[Input] expression values as supplied -> 51 bins; no additional normalization/log1p', flush=True)
    t0 = time.perf_counter()
    seed_all(opt.seed)
    model = TransformerModel(**constructor, vocab=vocab)
    load_pretrained_backbone(model, pretrained / 'best_model.pt', out)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE))
    model.float().to('cuda')
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=opt.lr, weight_decay=opt.weight_decay,
                                 betas=(0.9, 0.999), eps=1e-8)
    torch.cuda.synchronize()
    model_loading_seconds = time.perf_counter() - t0
    probes = ['transformer_encoder.layers.0.self_attn.in_proj_weight', 'cls_decoder.out_layer.weight']
    initial = {k: model.state_dict()[k].detach().cpu().clone() for k in probes}
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed + 10000)
    t0 = time.perf_counter()
    val_cache = tokenizer.cache(validation, opt.batch_size)
    cache_seconds = time.perf_counter() - t0
    np.savez_compressed(out / 'validation_tokenized_inputs.npz', gene_ids=val_cache[0],
                        expression_bins=val_cache[1], original_row_indices=validation, labels=y[validation])
    print('[Validation] fixed input tokens saved', flush=True)
    opt.model_out.mkdir(parents=True, exist_ok=False)
    epoch_summaries = []
    best = None
    for epoch in range(1, opt.epochs + 1):
        epoch_out = out / f'epoch_{epoch:03d}'
        epoch_out.mkdir()
        seed_all(opt.seed + epoch)
        order = np.random.default_rng(opt.seed + epoch).permutation(training)
        saved_genes, saved_values = tokenizer.empty_recording(len(order))
        steps = []
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler = ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        loss_sum, correct = 0.0, 0
        try:
            for step, start in enumerate(range(0, len(order), opt.batch_size), 1):
                end = min(start + opt.batch_size, len(order))
                rows = order[start:end]
                batch = tokenizer.batch(rows)
                length = batch['gene'].shape[1]
                saved_genes[start:end, :length] = batch['gene'].numpy()
                saved_values[start:end, :length] = batch['expr'].numpy()
                target = torch.from_numpy(y[rows]).to('cuda')
                optimizer.zero_grad(set_to_none=True)
                logits = classify(model, batch, tokenizer.pad_id)
                loss = torch.nn.functional.cross_entropy(logits, target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Non-finite training loss at epoch {epoch}, step {step}')
                loss.backward()
                if epoch == 1 and step == 1:
                    missing_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
                    if missing_grad:
                        raise RuntimeError(f'Trainable parameters without gradients: {missing_grad}')
                    print(f'[Model input] gene/value={tuple(batch["gene"].shape)}, labels={tuple(target.shape)}, logits={tuple(logits.shape)}', flush=True)
                    print('[Backward] all trainable parameters received gradients', flush=True)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_value = float(loss.detach().item())
                batch_correct = int((logits.detach().argmax(dim=1) == target).sum().item())
                loss_sum += loss_value * len(rows)
                correct += batch_correct
                steps.append({'epoch': epoch, 'step': step, 'cells': len(rows), 'padded_length': length,
                              'loss': loss_value, 'correct': batch_correct, 'grad_norm_before_clip': float(grad_norm.item())})
                if step % 25 == 0 or end == len(order):
                    print(f'[Train {epoch}/{opt.epochs}] {end}/{len(order)} cells, online_loss={loss_sum/end:.4f}', flush=True)
                del logits, loss, target
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - t0
        finally:
            sampler.stop()
            pd.DataFrame(sampler.rows).to_csv(epoch_out / 'training_resources.csv', index=False)
        training_metrics = sampler.metrics()
        optimizer.zero_grad(set_to_none=True)
        model.cur_gene_token_embs = None
        pd.DataFrame(steps).to_csv(epoch_out / 'training_steps.csv', index=False)
        np.savez_compressed(epoch_out / 'training_tokenized_inputs.npz', gene_ids=saved_genes,
                            expression_bins=saved_values, original_row_indices=order, labels=y[order])
        del saved_genes, saved_values
        deltas = {k: float((model.state_dict()[k].detach().cpu() - initial[k]).abs().max().item()) for k in probes}
        if not all(np.isfinite(v) and v > 0 for v in deltas.values()):
            raise RuntimeError(f'Expected encoder and classifier weights to change: {deltas}')
        validation_metrics, val_logits = evaluate(model, val_cache, y[validation], cell_ids[validation],
            labels, tokenizer.pad_id, opt.batch_size, epoch_out / 'validation')
        bundle = opt.model_out / f'epoch_{epoch:03d}'
        bundle.mkdir()
        save_start = time.perf_counter()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, bundle / 'model.pt')
        torch.save({'epoch_completed': epoch, 'optimizer': optimizer.state_dict(),
                    'rng_state': rng_state(), 'run_config': config}, bundle / 'training_state.pt')
        for filename in ['model_config.json', 'preprocessing.json', 'labels.json', 'genes.json', 'vocab.json',
                         'run_config.json', 'pretrained_args.json', 'environment.json', 'requirements_freeze.txt']:
            shutil.copy2(out / filename, bundle / filename)
        shutil.copy2(__file__, bundle / 'predict.py')
        (bundle / 'README.txt').write_text(
            'Saved scGPT encoder + trained cell-type classifier. model.pt is a plain state_dict.\n'
            'Run prediction inside the scgpt-spark container using its existing environment:\n'
            'python /path/to/epoch_001/predict.py --mode predict --bundle /path/to/epoch_001 '
            '--input /workspace/scgpt/data/pancreas_demo/demo_test.h5ad --out /workspace/scgpt/results/NEW_NAME\n'
            'All input cells are retained. Unknown labels are excluded from closed-set metrics and reported separately.\n'
            'No unknown-class detector is trained; softmax confidence is not an unknown detector.\n'
            'training_state.pt stores optimizer and RNG states for recovery/development; this script has no resume CLI.\n'
            'Validation is a cell split within the supplied train file, not a donor-independent benchmark.\n')
        save_seconds = time.perf_counter() - save_start
        # Verify strict reload and identical predictions from the saved weights.
        reloaded = TransformerModel(**constructor, vocab=vocab)
        reloaded.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        reloaded.float().to('cuda').eval()
        end = min(opt.batch_size, len(validation))
        with torch.inference_mode():
            reload_logits = classify(reloaded, cached_batch(val_cache, 0, end, tokenizer.pad_id), tokenizer.pad_id).cpu().numpy()
        reload_error = float(np.max(np.abs(reload_logits - val_logits[:end])))
        if not np.allclose(reload_logits, val_logits[:end], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f'Saved-model reload mismatch: max error {reload_error}')
        del reloaded
        gc.collect()
        write_json(bundle / 'manifest.json', {
            'model_sha256': sha256(bundle / 'model.pt'),
            'training_state_sha256': sha256(bundle / 'training_state.pt'),
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
            'results_directory': str(out),
        })
        record = {
            'epoch': epoch, 'training_cells': len(training), 'training_seconds': training_seconds,
            'training_cells_per_second': len(training) / training_seconds,
            'online_training_loss': loss_sum / len(training), 'online_training_accuracy': correct / len(training),
            **training_metrics, 'validation': validation_metrics, 'weight_probe_max_abs_deltas': deltas,
            'model_bundle': str(bundle), 'checkpoint_save_seconds': save_seconds,
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
        }
        write_json(epoch_out / 'summary.json', record)
        epoch_summaries.append(record)
        score = validation_metrics['macro_f1_all_training_classes']
        if best is None or score > best['validation_macro_f1']:
            best = {'epoch': epoch, 'validation_macro_f1': score, 'model_bundle': str(bundle)}
            write_json(opt.model_out / 'best.json', best)
        print(f'[Validation] accuracy={validation_metrics["accuracy"]:.4f}, macro-F1={score:.4f}', flush=True)
        print(f'[Saved] {bundle}; reload check passed', flush=True)
    summary = {
        'status': 'complete', 'task': 'scGPT supervised cell-type fine-tuning',
        'training_cells': len(training), 'validation_cells': len(validation), 'classes': len(labels),
        'matched_genes': len(genes), 'epochs': opt.epochs, 'batch_size': opt.batch_size,
        'max_length_including_cls': opt.max_length, 'precision': 'FP32', 'seed': opt.seed,
        'parameter_count_constructed_model': sum(p.numel() for p in model.parameters()),
        'parameter_count_trainable': sum(p.numel() for p in trainable),
        'parameter_count_encoder': sum(p.numel() for name, p in model.named_parameters() if name.startswith(CORE)),
        'input_preparation_seconds': input_seconds, 'model_loading_seconds': model_loading_seconds,
        'validation_tokenization_seconds': cache_seconds,
        'training_seconds_total': sum(e['training_seconds'] for e in epoch_summaries),
        'training_gpu_peak_allocated_gib': max(e['gpu_peak_allocated_gib'] for e in epoch_summaries),
        'training_gpu_peak_reserved_gib': max(e['gpu_peak_reserved_gib'] for e in epoch_summaries),
        'training_process_rss_peak_observed_gib': max(e['python_process_rss_peak_observed_gib'] for e in epoch_summaries),
        'best_checkpoint': best, 'validation': epoch_summaries[best['epoch'] - 1]['validation'],
        'weights_updated': True, 'encoder_and_classifier_updated': True, 'saved_model_reload_verified': True,
        'test_evaluated': False, 'epochs_detail': epoch_summaries,
        'timing_scope': config['training_timing_scope'], 'memory_notes': MEMORY_NOTES,
        'interpretation': 'Small-run fine-tuning and validation check; not a full-data or donor-independent accuracy benchmark',
    }
    write_json(out / 'summary.json', summary)
    print('\n' + json.dumps({k: summary[k] for k in [
        'status', 'training_cells', 'validation_cells', 'epochs', 'training_seconds_total',
        'training_gpu_peak_allocated_gib', 'training_process_rss_peak_observed_gib',
        'best_checkpoint', 'saved_model_reload_verified', 'test_evaluated']}, indent=2), flush=True)
    print('SCGPT_FINETUNE_COMPLETE', flush=True)


def predict(opt):
    if opt.bundle is None or opt.input is None:
        raise ValueError('Prediction requires --bundle and --input')
    bundle = opt.bundle
    constructor = json.loads((bundle / 'model_config.json').read_text())
    preprocess = json.loads((bundle / 'preprocessing.json').read_text())
    labels = json.loads((bundle / 'labels.json').read_text())['labels']
    label_to_id = {name: i for i, name in enumerate(labels)}
    genes = json.loads((bundle / 'genes.json').read_text())
    vocab = GeneVocab.from_file(bundle / 'vocab.json')
    vocab.set_default_index(vocab[preprocess['pad_token']])
    data = ad.read_h5ad(opt.input)
    if data.n_obs == 0:
        raise ValueError('Input has no cells')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    originals = (data.obs['Celltype'].astype('string').fillna('<unlabeled>').astype(str).to_numpy()
                 if 'Celltype' in data.obs else np.full(data.n_obs, '<unlabeled>'))
    y = np.asarray([label_to_id.get(label, -1) for label in originals], dtype=np.int64)
    pd.DataFrame({'cell_id': cell_ids, 'original_label': originals, 'known_training_label': y >= 0}).to_csv(
        opt.out / 'input_cells.tsv', sep='\t', index=False)
    write_json(opt.out / 'prediction_config.json', {
        'bundle': str(bundle), 'model_sha256': sha256(bundle / 'model.pt'),
        'input': str(opt.input), 'input_sha256': sha256(opt.input), 'seed': opt.seed,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'preprocessing': preprocess,
        'unknown_label_counts': pd.Series(originals[y < 0]).value_counts().astype(int).to_dict(),
        'unknown_policy': 'retain all predictions; evaluate known ground-truth labels only; no unknown-class detector',
    })
    model = TransformerModel(**constructor, vocab=vocab)
    model.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    model.float().to('cuda').eval()
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed)
    t0 = time.perf_counter()
    cache = tokenizer.cache(np.arange(len(y)), opt.batch_size)
    tokenization_seconds = time.perf_counter() - t0
    np.savez_compressed(opt.out / 'tokenized_inputs.npz', gene_ids=cache[0], expression_bins=cache[1], labels=y)
    metrics, _ = evaluate(model, cache, y, cell_ids, labels, tokenizer.pad_id, opt.batch_size, opt.out)
    summary = {'status': 'complete', 'task': 'fine-tuned cell-type predictions',
               'tokenization_seconds': tokenization_seconds, 'warmup_steps': 0,
               'weights_updated': False, 'model_bundle': str(bundle), 'memory_notes': MEMORY_NOTES, **metrics}
    write_json(opt.out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print('SCGPT_PREDICTION_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-out', type=Path)
    parser.add_argument('--n-cells', type=int, default=1000, help='training subset size; 0=all training-pool cells')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=1200)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--input', type=Path)
    opt = parser.parse_args()
    if (opt.batch_size < 1 or opt.epochs < 1 or opt.n_cells < 0 or opt.max_length < 2
            or not 0 < opt.validation_fraction < 1 or not np.isfinite(opt.lr) or opt.lr <= 0
            or not np.isfinite(opt.weight_decay) or opt.weight_decay < 0 or not 0 <= opt.seed < 2**32 - 10000):
        parser.error('Invalid numeric option')
    if opt.mode == 'train' and opt.model_out is None:
        parser.error('Training requires --model-out')
    opt.out.mkdir(parents=True, exist_ok=True)
    if (opt.out / 'run_config.json').exists() or (opt.out / 'summary.json').exists() or (opt.out / 'prediction_config.json').exists():
        raise FileExistsError(f'Use a new output directory: {opt.out}')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; stopping instead of running on CPU')
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_all(opt.seed)
        preserve_environment(opt.out)
        train(opt) if opt.mode == 'train' else predict(opt)
    except Exception as exc:
        write_json(opt.out / 'failure.json', {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

SCGPT_PYTHON

cp -- "$root/scripts/02_finetune.py" "$root/results/$run_id/"
cp -- "$runner" "$root/results/$run_id/"
sudo docker inspect "$container" --format 'Image={{.Config.Image}} ImageID={{.Image}}' \
    > "$root/results/$run_id/container_image.txt"
command=(sudo docker exec "$container" python -u "/workspace/scgpt/results/$run_id/02_finetune.py" \
    --out "/workspace/scgpt/results/$run_id" \
    --model-out "/workspace/scgpt/models/finetuned/$run_id" "$@")
printf '%q ' "${command[@]}" > "$root/results/$run_id/run_command.sh"
printf '\n' >> "$root/results/$run_id/run_command.sh"
"${command[@]}" 2>&1 | tee "$root/results/$run_id/run.log"
printf '\nHost results: %s\n' "$root/results/$run_id"
printf 'Host models:  %s\n' "$root/models/finetuned/$run_id"
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/full_realdata.py — 원문 전문 (39,178 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/full_realdata.py", "sha256": "9368e97d5d003484531e9f52b6ef93e01836ed89eb9f986dd25b330a95450ab9", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""scGPT pancreas classification: supervised fine-tuning and saved-bundle prediction.

Run train through scgpt_finetune.sh. Reload example (inside the same container):
python predict.py --mode predict --bundle /path/to/epoch_001 \
    --input /path/to/demo_test.h5ad --out /path/to/new_prediction_results
No package installation or test-set training is performed.
"""
import argparse
import os
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer import GeneVocab

ROOT = Path('/workspace/scgpt')
GIB = 1024 ** 3
CORE = ('encoder.', 'value_encoder.', 'transformer_encoder.')
TRAINABLE = CORE + ('cls_decoder.',)
MEMORY_NOTES = ('CUDA peaks are process-local PyTorch allocator counters, not total device usage. '
                'RSS is sampled every 0.2 s, not an exact peak. Spark uses shared memory: '
                'do not add RSS and CUDA as independent physical memory pools.')


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


INITIAL_SWAP_BYTES = psutil.swap_memory().used

class ResourceSampler:
    def __init__(self):
        self.process = psutil.Process()
        self.rows = []
        self.event = threading.Event()
        self.lock = threading.Lock()

    def sample(self):
        available=psutil.virtual_memory().available
        swap=psutil.swap_memory().used
        if available<8*GIB or swap>INITIAL_SWAP_BYTES+2*GIB:
            guard={'time_unix':time.time(),'available_gib':available/GIB,'swap_gib':swap/GIB,'cuda_allocated_gib':torch.cuda.memory_allocated()/GIB,'cuda_reserved_gib':torch.cuda.memory_reserved()/GIB}
            write_json(Path(os.environ['SCGPT_GUARD_DIR'])/'memory_guard.json',guard)
            print('[MEMORY_GUARD]',guard,flush=True)
            os._exit(86)
        with self.lock:
            self.rows.append({
                'elapsed_seconds': time.perf_counter() - self.start_time,
                'gpu_util_percent': float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]),
                'process_rss_bytes': self.process.memory_info().rss,
                'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'cuda_reserved_bytes': torch.cuda.memory_reserved(),
                'system_available_bytes': psutil.virtual_memory().available,
                'system_swap_used_bytes': psutil.swap_memory().used,
            })

    def loop(self):
        while not self.event.wait(0.2):
            self.sample()

    def start(self):
        self.start_time = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()
        self.sample()

    def metrics(self):
        return {
            'gpu_util_mean_percent': float(np.mean([r['gpu_util_percent'] for r in self.rows])),
            'gpu_util_max_percent': max(r['gpu_util_percent'] for r in self.rows),
            'gpu_util_samples': len(self.rows),
            'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated() / GIB,
            'gpu_peak_reserved_gib': torch.cuda.max_memory_reserved() / GIB,
            'python_process_rss_peak_observed_gib': max(r['process_rss_bytes'] for r in self.rows) / GIB,
        }


def preserve_environment(out):
    source = out / 'source_snapshot'
    source.mkdir(exist_ok=True)
    for name, obj in [('model.py', TransformerModel), ('data_collator.py', DataCollator),
                      ('preprocess.py', binning), ('gene_tokenizer.py', GeneVocab)]:
        shutil.copy2(inspect.getfile(obj), source / name)
    packages = {p: metadata.version(p) for p in
                ['torch', 'scgpt', 'numpy', 'anndata', 'scanpy', 'scipy', 'pandas', 'scikit-learn', 'psutil']}
    direct_url = metadata.distribution('scgpt').read_text('direct_url.json')
    device = torch.cuda.get_device_properties(0)
    write_json(out / 'environment.json', {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'architecture': platform.machine(), 'packages': packages,
        'scgpt_install_source': json.loads(direct_url) if direct_url else None,
        'cuda_version': torch.version.cuda, 'gpu_name': device.name,
        'gpu_compute_capability': [device.major, device.minor],
        'cuda_reported_total_memory_gib': device.total_memory / GIB,
        'system_total_memory_gib': psutil.virtual_memory().total / GIB,
        'torch_cpu_threads': torch.get_num_threads(),
    })
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (out / 'requirements_freeze.txt').write_text(freeze.stdout)


def choose_training_subset(pool, y, requested, seed):
    """Proportional subset with at least one cell from every training class."""
    if requested == 0 or requested == len(pool):
        return np.sort(pool)
    classes, counts = np.unique(y[pool], return_counts=True)
    if requested < len(classes) or requested > len(pool):
        raise ValueError(f'n-cells must be 0 (all) or {len(classes)}..{len(pool)}')
    ideal = counts * (requested / len(pool))
    quota = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while quota.sum() < requested:
        available = np.flatnonzero(quota < counts)
        k = available[np.argmax((ideal - quota)[available])]
        quota[k] += 1
    while quota.sum() > requested:
        available = np.flatnonzero(quota > 1)
        k = available[np.argmax((quota - ideal)[available])]
        quota[k] -= 1
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(pool[y[pool] == c], int(q), replace=False)
                               for c, q in zip(classes, quota)])
    return np.sort(selected)


def extract_matrix(adata, genes):
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Cell IDs and gene names must be unique')
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise ValueError(f'Missing {len(missing)} training genes, examples: {sorted(missing)[:5]}')
    selected = adata[:, genes]
    matrix = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
    matrix = matrix.astype(np.float32, copy=True)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('Expression values must be finite and nonnegative')
    if np.any(np.count_nonzero(matrix, axis=1) == 0):
        raise ValueError('A cell has no nonzero expression after vocabulary filtering')
    return matrix


class Tokenizer:
    def __init__(self, matrix, gene_names, vocab, config):
        self.matrix = matrix
        self.gene_ids = np.asarray(vocab(gene_names), dtype=np.int64)
        self.pad_id = vocab[config['pad_token']]
        self.cls_id = vocab['<cls>']
        self.pad_value = config['pad_value']
        self.max_length = config['max_length_including_cls']
        self.collator = DataCollator(do_padding=True, pad_token_id=self.pad_id,
            pad_value=self.pad_value, do_mlm=False, do_binning=True,
            max_length=self.max_length, sampling=True, keep_first_n_tokens=1)

    def batch(self, rows):
        examples = []
        for i in rows:
            nz = np.flatnonzero(self.matrix[i])
            genes = np.concatenate([np.array([self.cls_id], dtype=np.int64), self.gene_ids[nz]])
            values = np.concatenate([np.array([self.pad_value], dtype=np.float32), self.matrix[i, nz]])
            examples.append({'genes': torch.from_numpy(genes), 'expressions': torch.from_numpy(values)})
        return self.collator(examples)

    def empty_recording(self, n):
        return (np.full((n, self.max_length), self.pad_id, dtype=np.int64),
                np.full((n, self.max_length), self.pad_value, dtype=np.float32))

    def cache(self, rows, batch_size):
        genes, values = self.empty_recording(len(rows))
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch = self.batch(rows[start:end])
            length = batch['gene'].shape[1]
            genes[start:end, :length] = batch['gene'].numpy()
            values[start:end, :length] = batch['expr'].numpy()
        return genes, values


def cached_batch(cache, start, end, pad_id):
    genes, values = cache
    length = int(np.count_nonzero(genes[start:end] != pad_id, axis=1).max())
    return {'gene': torch.from_numpy(genes[start:end, :length]),
            'expr': torch.from_numpy(values[start:end, :length])}


def classify(model, batch, pad_id):
    genes = batch['gene'].to('cuda')
    values = batch['expr'].to('cuda')
    encoded = model._encode(genes, values, src_key_padding_mask=genes.eq(pad_id))
    # Official classification path uses raw CLS, not L2-normalized embeddings.
    return model.cls_decoder(encoded[:, 0, :])


def load_pretrained_backbone(model, checkpoint, out):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError('Expected a plain tensor state_dict')
    converted = {}
    for key, value in state.items():
        for pattern, replacement in [
            (r'self_attn\._impl\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\._impl\.out_proj\.', 'self_attn.out_proj.'),
        ]:
            key = re.sub(pattern, replacement, key)
        if key in converted:
            raise ValueError(f'Duplicate renamed checkpoint key: {key}')
        converted[key] = value
    current = model.state_dict()
    required = [k for k in current if k.startswith(CORE)]
    loaded = {k: converted[k] for k in required
              if k in converted and converted[k].shape == current[k].shape}
    missing = sorted(set(required) - set(loaded))
    write_json(out / 'weight_loading.json', {
        'required_backbone_keys': len(required), 'loaded_backbone_keys': len(loaded),
        'missing_backbone_keys': missing, 'loaded_keys': sorted(loaded),
        'classifier': 'new random initialization; pretrained classifier not used',
        'unused_checkpoint_keys': sorted(set(converted) - set(loaded)),
    })
    if missing:
        raise RuntimeError(f'Incomplete pretrained backbone: {missing[:10]}')
    model.load_state_dict(loaded, strict=False)


def evaluate(model, cache, y, cell_ids, labels, pad_id, batch_size, out):
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    logits_all = np.empty((len(y), len(labels)), dtype=np.float32)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sampler = ResourceSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(y), batch_size):
                end = min(start + batch_size, len(y))
                batch = cached_batch(cache, start, end, pad_id)
                logits_all[start:end] = classify(model, batch, pad_id).cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
    finally:
        sampler.stop()
        pd.DataFrame(sampler.rows).to_csv(out / 'resources.csv', index=False)
    if not np.isfinite(logits_all).all():
        raise FloatingPointError('Non-finite prediction logits')
    probabilities = torch.from_numpy(logits_all).softmax(dim=1).numpy()
    predicted = logits_all.argmax(axis=1)
    known = y >= 0
    metrics = {
        'cells_total': len(y), 'cells_evaluated_known_labels': int(known.sum()),
        'cells_without_known_training_label': int((~known).sum()),
        'forward_seconds': seconds, 'cells_per_second': len(y) / seconds,
        'timing_scope': 'cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving',
        **sampler.metrics(),
    }
    if known.any():
        true, pred = y[known], predicted[known]
        metrics.update({
            'loss': float(torch.nn.functional.cross_entropy(torch.from_numpy(logits_all[known]),
                                                           torch.from_numpy(true)).item()),
            'accuracy': float(accuracy_score(true, pred)),
            'macro_f1_all_training_classes': float(f1_score(true, pred, labels=np.arange(len(labels)), average='macro', zero_division=0)),
            'macro_f1_present_true_classes': float(f1_score(true, pred, labels=np.unique(true), average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(true, pred, average='weighted', zero_division=0)),
        })
        write_json(out / 'classification_report.json', classification_report(
            true, pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0))
        pd.DataFrame(confusion_matrix(true, pred, labels=np.arange(len(labels))),
                     index=labels, columns=labels).rename_axis('true_label').to_csv(out / 'confusion_matrix.csv')
    pd.DataFrame({
        'cell_id': cell_ids, 'true_label_id': y,
        'true_label': [labels[i] if i >= 0 else '<unknown_or_unlabeled>' for i in y],
        'predicted_label_id': predicted, 'predicted_label': [labels[i] for i in predicted],
        'max_probability': probabilities.max(axis=1), 'included_in_metrics': known,
    }).to_csv(out / 'predictions.tsv', sep='\t', index=False)
    np.savez_compressed(out / 'logits_probabilities.npz', logits=logits_all, probabilities=probabilities)
    write_json(out / 'metrics.json', metrics)
    model.cur_gene_token_embs = None
    return metrics, logits_all


def rng_state():
    np_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {'algorithm': np_state[0], 'keys': np_state[1].tolist(),
                  'position': int(np_state[2]), 'has_gauss': int(np_state[3]), 'cached_gaussian': float(np_state[4])},
        'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
    }


def train(opt):
    out = opt.out
    pretrained = ROOT / 'models/whole-human'
    args = json.loads((pretrained / 'args.json').read_text())
    if (args.get('n_bins') != 51 or args.get('input_style') != 'binned'
            or args.get('input_emb_style', 'continuous') != 'continuous' or args.get('pre_norm', False)):
        raise ValueError('Requires the inspected 51-bin continuous-value post-norm checkpoint')
    vocab = GeneVocab.from_file(pretrained / 'vocab.json')
    for token in [args['pad_token'], '<cls>', '<eoc>']:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[args['pad_token']])
    input_path = ROOT / 'data/pancreas_demo/demo_train.h5ad'
    t0 = time.perf_counter()
    data = ad.read_h5ad(input_path)
    if data.shape != (10600, 3000) or 'Celltype' not in data.obs or data.obs['Celltype'].isna().any():
        raise ValueError('Input differs from the inspected 10600 x 3000 labeled training file')
    label_values = data.obs['Celltype'].astype(str).to_numpy()
    labels = sorted(set(label_values))
    if len(labels) != 13:
        raise ValueError(f'Expected 13 training classes; found {len(labels)}')
    label_to_id = {name: i for i, name in enumerate(labels)}
    y = np.asarray([label_to_id[x] for x in label_values], dtype=np.int64)
    all_rows = np.arange(data.n_obs)
    pool, validation = train_test_split(all_rows, test_size=opt.validation_fraction,
                                        random_state=opt.seed, stratify=y)
    validation = np.sort(validation)
    training = choose_training_subset(pool, y, opt.n_cells, opt.seed)
    if len(np.intersect1d(training, validation)):
        raise RuntimeError('Training and validation overlap')
    if set(y[training]) != set(range(len(labels))) or set(y[validation]) != set(range(len(labels))):
        raise ValueError('Every class must be represented in training and validation')
    genes = [str(g) for g in data.var_names if g in vocab]
    if len(genes) != 2999:
        raise ValueError(f'Expected 2999 vocabulary-matched genes, found {len(genes)}')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    roles = np.full(len(y), 'train_pool_unused', dtype=object)
    roles[validation] = 'validation'
    roles[training] = 'train_used'
    pd.DataFrame({'original_row_index': all_rows, 'cell_id': cell_ids,
                  'Celltype': label_values, 'label_id': y, 'role': roles}).to_csv(
                      out / 'split_assignments.tsv', sep='\t', index=False)
    pd.crosstab(pd.Series(label_values, name='Celltype'), pd.Series(roles, name='role')).to_csv(
        out / 'split_class_counts.csv')
    input_seconds = time.perf_counter() - t0
    del data
    preprocess = {
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'], 'n_bins': 51,
        'max_length_including_cls': opt.max_length,
        'additional_normalize_total': False, 'additional_log1p': False,
        'expression_provenance': 'Supplied h5ad values; no raw layer or preprocessing metadata',
        'tokenization': 'nonzero genes only; prepend CLS; official per-cell quantile binning; random overlength sampling; dynamic padding',
        'pooling': 'raw CLS features into official cls_decoder; no L2 normalization',
        'prediction_genes': 'require every saved training gene; reorder to training gene order; ignore additional genes',
    }
    constructor = {
        'ntoken': len(vocab), 'd_model': args['embsize'], 'nhead': args['nheads'],
        'd_hid': args['d_hid'], 'nlayers': args['nlayers'],
        'nlayers_cls': args['n_layers_cls'], 'n_cls': len(labels), 'dropout': args['dropout'],
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'],
        'do_mvc': False, 'do_dab': False, 'use_batch_labels': False,
        'domain_spec_batchnorm': False, 'input_emb_style': 'continuous', 'cell_emb_style': 'cls',
        'explicit_zero_prob': False, 'use_fast_transformer': False, 'pre_norm': False,
    }
    config = {
        'task': 'supervised cell-type classification', 'seed': opt.seed,
        'train_source_cells': len(y), 'training_pool_cells': len(pool), 'training_cells': len(training),
        'validation_cells': len(validation), 'classes': len(labels), 'epochs': opt.epochs,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'autocast': False, 'tf32': False,
        'optimizer': {'name': 'AdamW', 'lr': opt.lr, 'weight_decay': opt.weight_decay,
                      'betas': [0.9, 0.999], 'eps': 1e-8},
        'scheduler': None, 'gradient_clip_max_norm': 1.0,
        'loss': 'unweighted cross entropy', 'trainable_prefixes': list(TRAINABLE),
        'unused_expression_decoder': 'frozen and bypassed',
        'classifier': 'new official ClsDecoder; all classifier layers trainable',
        'split': 'stratified by Celltype before subsampling; no donor/batch metadata available',
        'validation_limit': 'within-file cell split; not donor-independent validation',
        'validation_tokens': 'generated once with seed+10000, saved and reused',
        'epoch_seed': 'seed+epoch; reinitializes RNG before each epoch',
        'checkpoint_selection': 'highest validation macro-F1 over all 13 training classes',
        'test_used_for_training_or_selection': False,
        'training_warmup_steps': 0,
        'training_timing_scope': 'tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization',
        'training_timer_excludes': 'imports, h5ad/model loading, validation token cache, validation, disk saving, reload verification',
        'memory_notes': MEMORY_NOTES, 'preprocessing': preprocess,
        'reproducibility': 'Seeds, exact tokens and environment saved; bitwise reproducibility across platforms is not guaranteed',
    }
    for filename, obj in [('run_config.json', config), ('preprocessing.json', preprocess),
                          ('model_config.json', constructor), ('labels.json', {'labels': labels, 'label_to_id': label_to_id}),
                          ('genes.json', genes), ('vocab.json', vocab.get_stoi())]:
        write_json(out / filename, obj)
    shutil.copy2(pretrained / 'args.json', out / 'pretrained_args.json')
    write_json(out / 'provenance.json', {
        'input': {'path': str(input_path), 'sha256': sha256(input_path)},
        'pretrained_checkpoint': {'path': str(pretrained / 'best_model.pt'), 'sha256': sha256(pretrained / 'best_model.pt')},
        'script_sha256': sha256(__file__),
        'tutorial': 'https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html',
        'dataset_source': 'TOSICA author-distributed Figshare demo_train.h5ad (file 39009503)',
    })
    print(f'[Split] train={len(training)}, validation={len(validation)}, classes={len(labels)}, matched_genes={len(genes)}', flush=True)
    print('[Input] expression values as supplied -> 51 bins; no additional normalization/log1p', flush=True)
    t0 = time.perf_counter()
    seed_all(opt.seed)
    model = TransformerModel(**constructor, vocab=vocab)
    load_pretrained_backbone(model, pretrained / 'best_model.pt', out)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE))
    model.float().to('cuda')
    torch.cuda.synchronize()
    write_json(out/'model_loading_baseline.json', {'allocated_gib':torch.cuda.memory_allocated()/GIB,'reserved_gib':torch.cuda.memory_reserved()/GIB,'rss_gib':psutil.Process().memory_info().rss/GIB,'system_available_gib':psutil.virtual_memory().available/GIB})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=opt.lr, weight_decay=opt.weight_decay,
                                 betas=(0.9, 0.999), eps=1e-8)
    torch.cuda.synchronize()
    model_loading_seconds = time.perf_counter() - t0
    probes = ['transformer_encoder.layers.0.self_attn.in_proj_weight', 'cls_decoder.out_layer.weight']
    initial = {k: model.state_dict()[k].detach().cpu().clone() for k in probes}
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed + 10000)
    t0 = time.perf_counter()
    val_cache = tokenizer.cache(validation, opt.batch_size)
    cache_seconds = time.perf_counter() - t0
    np.savez_compressed(out / 'validation_tokenized_inputs.npz', gene_ids=val_cache[0],
                        expression_bins=val_cache[1], original_row_indices=validation, labels=y[validation])
    print('[Validation] fixed input tokens saved', flush=True)
    opt.model_out.mkdir(parents=True, exist_ok=False)
    epoch_summaries = []
    best = None
    for epoch in range(1, opt.epochs + 1):
        epoch_out = out / f'epoch_{epoch:03d}'
        epoch_out.mkdir()
        seed_all(opt.seed + epoch)
        order = np.random.default_rng(opt.seed + epoch).permutation(training)
        saved_genes, saved_values = tokenizer.empty_recording(len(order))
        steps = []
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler = ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        loss_sum, correct = 0.0, 0
        try:
            for step, start in enumerate(range(0, len(order), opt.batch_size), 1):
                end = min(start + opt.batch_size, len(order))
                rows = order[start:end]
                batch = tokenizer.batch(rows)
                length = batch['gene'].shape[1]
                saved_genes[start:end, :length] = batch['gene'].numpy()
                saved_values[start:end, :length] = batch['expr'].numpy()
                target = torch.from_numpy(y[rows]).to('cuda')
                optimizer.zero_grad(set_to_none=True)
                logits = classify(model, batch, tokenizer.pad_id)
                loss = torch.nn.functional.cross_entropy(logits, target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Non-finite training loss at epoch {epoch}, step {step}')
                loss.backward()
                if epoch == 1 and step == 1:
                    missing_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
                    if missing_grad:
                        raise RuntimeError(f'Trainable parameters without gradients: {missing_grad}')
                    print(f'[Model input] gene/value={tuple(batch["gene"].shape)}, labels={tuple(target.shape)}, logits={tuple(logits.shape)}', flush=True)
                    print('[Backward] all trainable parameters received gradients', flush=True)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_value = float(loss.detach().item())
                batch_correct = int((logits.detach().argmax(dim=1) == target).sum().item())
                loss_sum += loss_value * len(rows)
                correct += batch_correct
                steps.append({'epoch': epoch, 'step': step, 'cells': len(rows), 'padded_length': length,
                              'loss': loss_value, 'correct': batch_correct, 'grad_norm_before_clip': float(grad_norm.item())})
                if step % 25 == 0 or end == len(order):
                    print(f'[Train {epoch}/{opt.epochs}] {end}/{len(order)} cells, online_loss={loss_sum/end:.4f}', flush=True)
                del logits, loss, target
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - t0
        finally:
            sampler.stop()
            pd.DataFrame(sampler.rows).to_csv(epoch_out / 'training_resources.csv', index=False)
        training_metrics = sampler.metrics()
        optimizer.zero_grad(set_to_none=True)
        model.cur_gene_token_embs = None
        pd.DataFrame(steps).to_csv(epoch_out / 'training_steps.csv', index=False)
        np.savez_compressed(epoch_out / 'training_tokenized_inputs.npz', gene_ids=saved_genes,
                            expression_bins=saved_values, original_row_indices=order, labels=y[order])
        del saved_genes, saved_values
        deltas = {k: float((model.state_dict()[k].detach().cpu() - initial[k]).abs().max().item()) for k in probes}
        if not all(np.isfinite(v) and v > 0 for v in deltas.values()):
            raise RuntimeError(f'Expected encoder and classifier weights to change: {deltas}')
        validation_metrics, val_logits = evaluate(model, val_cache, y[validation], cell_ids[validation],
            labels, tokenizer.pad_id, opt.batch_size, epoch_out / 'validation')
        bundle = opt.model_out / f'epoch_{epoch:03d}'
        bundle.mkdir()
        save_start = time.perf_counter()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, bundle / 'model.pt')
        torch.save({'epoch_completed': epoch, 'optimizer': optimizer.state_dict(),
                    'rng_state': rng_state(), 'run_config': config}, bundle / 'training_state.pt')
        for filename in ['model_config.json', 'preprocessing.json', 'labels.json', 'genes.json', 'vocab.json',
                         'run_config.json', 'pretrained_args.json', 'environment.json', 'requirements_freeze.txt']:
            shutil.copy2(out / filename, bundle / filename)
        shutil.copy2(__file__, bundle / 'predict.py')
        (bundle / 'README.txt').write_text(
            'Saved scGPT encoder + trained cell-type classifier. model.pt is a plain state_dict.\n'
            'Run prediction inside the scgpt-spark container using its existing environment:\n'
            'python /path/to/epoch_001/predict.py --mode predict --bundle /path/to/epoch_001 '
            '--input /workspace/scgpt/data/pancreas_demo/demo_test.h5ad --out /workspace/scgpt/results/NEW_NAME\n'
            'All input cells are retained. Unknown labels are excluded from closed-set metrics and reported separately.\n'
            'No unknown-class detector is trained; softmax confidence is not an unknown detector.\n'
            'training_state.pt stores optimizer and RNG states for recovery/development; this script has no resume CLI.\n'
            'Validation is a cell split within the supplied train file, not a donor-independent benchmark.\n')
        save_seconds = time.perf_counter() - save_start
        # Verify strict reload and identical predictions from the saved weights.
        reloaded = TransformerModel(**constructor, vocab=vocab)
        reloaded.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        reloaded.float().to('cuda').eval()
        end = min(opt.batch_size, len(validation))
        with torch.inference_mode():
            reload_logits = classify(reloaded, cached_batch(val_cache, 0, end, tokenizer.pad_id), tokenizer.pad_id).cpu().numpy()
        reload_error = float(np.max(np.abs(reload_logits - val_logits[:end])))
        if not np.allclose(reload_logits, val_logits[:end], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f'Saved-model reload mismatch: max error {reload_error}')
        del reloaded
        gc.collect()
        write_json(bundle / 'manifest.json', {
            'model_sha256': sha256(bundle / 'model.pt'),
            'training_state_sha256': sha256(bundle / 'training_state.pt'),
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
            'results_directory': str(out),
        })
        record = {
            'epoch': epoch, 'training_cells': len(training), 'training_seconds': training_seconds,
            'training_cells_per_second': len(training) / training_seconds,
            'online_training_loss': loss_sum / len(training), 'online_training_accuracy': correct / len(training),
            **training_metrics, 'validation': validation_metrics, 'weight_probe_max_abs_deltas': deltas,
            'model_bundle': str(bundle), 'checkpoint_save_seconds': save_seconds,
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
        }
        write_json(epoch_out / 'summary.json', record)
        epoch_summaries.append(record)
        score = validation_metrics['macro_f1_all_training_classes']
        if best is None or score > best['validation_macro_f1']:
            best = {'epoch': epoch, 'validation_macro_f1': score, 'model_bundle': str(bundle)}
            write_json(opt.model_out / 'best.json', best)
        print(f'[Validation] accuracy={validation_metrics["accuracy"]:.4f}, macro-F1={score:.4f}', flush=True)
        print(f'[Saved] {bundle}; reload check passed', flush=True)
    summary = {
        'status': 'complete', 'task': 'scGPT supervised cell-type fine-tuning',
        'training_cells': len(training), 'validation_cells': len(validation), 'classes': len(labels),
        'matched_genes': len(genes), 'epochs': opt.epochs, 'batch_size': opt.batch_size,
        'max_length_including_cls': opt.max_length, 'precision': 'FP32', 'seed': opt.seed,
        'parameter_count_constructed_model': sum(p.numel() for p in model.parameters()),
        'parameter_count_trainable': sum(p.numel() for p in trainable),
        'parameter_count_encoder': sum(p.numel() for name, p in model.named_parameters() if name.startswith(CORE)),
        'input_preparation_seconds': input_seconds, 'model_loading_seconds': model_loading_seconds,
        'validation_tokenization_seconds': cache_seconds,
        'training_seconds_total': sum(e['training_seconds'] for e in epoch_summaries),
        'training_gpu_peak_allocated_gib': max(e['gpu_peak_allocated_gib'] for e in epoch_summaries),
        'training_gpu_peak_reserved_gib': max(e['gpu_peak_reserved_gib'] for e in epoch_summaries),
        'training_process_rss_peak_observed_gib': max(e['python_process_rss_peak_observed_gib'] for e in epoch_summaries),
        'best_checkpoint': best, 'validation': epoch_summaries[best['epoch'] - 1]['validation'],
        'weights_updated': True, 'encoder_and_classifier_updated': True, 'saved_model_reload_verified': True,
        'test_evaluated': False, 'epochs_detail': epoch_summaries,
        'timing_scope': config['training_timing_scope'], 'memory_notes': MEMORY_NOTES,
        'interpretation': 'Small-run fine-tuning and validation check; not a full-data or donor-independent accuracy benchmark',
    }
    write_json(out / 'summary.json', summary)
    print('\n' + json.dumps({k: summary[k] for k in [
        'status', 'training_cells', 'validation_cells', 'epochs', 'training_seconds_total',
        'training_gpu_peak_allocated_gib', 'training_process_rss_peak_observed_gib',
        'best_checkpoint', 'saved_model_reload_verified', 'test_evaluated']}, indent=2), flush=True)
    print('SCGPT_FINETUNE_COMPLETE', flush=True)


def predict(opt):
    if opt.bundle is None or opt.input is None:
        raise ValueError('Prediction requires --bundle and --input')
    bundle = opt.bundle
    constructor = json.loads((bundle / 'model_config.json').read_text())
    preprocess = json.loads((bundle / 'preprocessing.json').read_text())
    labels = json.loads((bundle / 'labels.json').read_text())['labels']
    label_to_id = {name: i for i, name in enumerate(labels)}
    genes = json.loads((bundle / 'genes.json').read_text())
    vocab = GeneVocab.from_file(bundle / 'vocab.json')
    vocab.set_default_index(vocab[preprocess['pad_token']])
    data = ad.read_h5ad(opt.input)
    if data.n_obs == 0:
        raise ValueError('Input has no cells')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    originals = (data.obs['Celltype'].astype('string').fillna('<unlabeled>').astype(str).to_numpy()
                 if 'Celltype' in data.obs else np.full(data.n_obs, '<unlabeled>'))
    y = np.asarray([label_to_id.get(label, -1) for label in originals], dtype=np.int64)
    pd.DataFrame({'cell_id': cell_ids, 'original_label': originals, 'known_training_label': y >= 0}).to_csv(
        opt.out / 'input_cells.tsv', sep='\t', index=False)
    write_json(opt.out / 'prediction_config.json', {
        'bundle': str(bundle), 'model_sha256': sha256(bundle / 'model.pt'),
        'input': str(opt.input), 'input_sha256': sha256(opt.input), 'seed': opt.seed,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'preprocessing': preprocess,
        'unknown_label_counts': pd.Series(originals[y < 0]).value_counts().astype(int).to_dict(),
        'unknown_policy': 'retain all predictions; evaluate known ground-truth labels only; no unknown-class detector',
    })
    model = TransformerModel(**constructor, vocab=vocab)
    model.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    model.float().to('cuda').eval()
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed)
    t0 = time.perf_counter()
    cache = tokenizer.cache(np.arange(len(y)), opt.batch_size)
    tokenization_seconds = time.perf_counter() - t0
    np.savez_compressed(opt.out / 'tokenized_inputs.npz', gene_ids=cache[0], expression_bins=cache[1], labels=y)
    metrics, _ = evaluate(model, cache, y, cell_ids, labels, tokenizer.pad_id, opt.batch_size, opt.out)
    summary = {'status': 'complete', 'task': 'fine-tuned cell-type predictions',
               'tokenization_seconds': tokenization_seconds, 'warmup_steps': 0,
               'weights_updated': False, 'model_bundle': str(bundle), 'memory_notes': MEMORY_NOTES, **metrics}
    write_json(opt.out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print('SCGPT_PREDICTION_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-out', type=Path)
    parser.add_argument('--n-cells', type=int, default=1000, help='training subset size; 0=all training-pool cells')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=1200)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--input', type=Path)
    opt = parser.parse_args()
    if (opt.batch_size < 1 or opt.epochs < 1 or opt.n_cells < 0 or opt.max_length < 2
            or not 0 < opt.validation_fraction < 1 or not np.isfinite(opt.lr) or opt.lr <= 0
            or not np.isfinite(opt.weight_decay) or opt.weight_decay < 0 or not 0 <= opt.seed < 2**32 - 10000):
        parser.error('Invalid numeric option')
    if opt.mode == 'train' and opt.model_out is None:
        parser.error('Training requires --model-out')
    opt.out.mkdir(parents=True, exist_ok=True)
    if (opt.out / 'run_config.json').exists() or (opt.out / 'summary.json').exists() or (opt.out / 'prediction_config.json').exists():
        raise FileExistsError(f'Use a new output directory: {opt.out}')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; stopping instead of running on CPU')
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_all(opt.seed)
        preserve_environment(opt.out)
        train(opt) if opt.mode == 'train' else predict(opt)
    except Exception as exc:
        write_json(opt.out / 'failure.json', {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/gpu_support.py — 원문 전문 (2,858 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/gpu_support.py", "sha256": "349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2", "encoding": "utf-8", "added_final_newline": false} -->
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

<!-- RESTORE_ALIAS {"path": "scripts/gpu_support.py", "source": "scgpt/scripts/gpu_support.py", "sha256": "349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2"} -->

</details>

<details>
<summary>scgpt/scripts/large_scale/discover.py — 원문 전문 (617 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/discover.py", "sha256": "78bfd3cc3bfcc861d3e8f9955a259bdc86185b17775ba2654ff1f894f784ffad", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import urllib.request,json
R=Path('/home/hyein/ai-benchmark/scgpt/data_external/tabula_sapiens');u='https://api.cellxgene.cziscience.com/curation/v1/collections/e5f58829-1a66-40b5-a624-9046778e74f5'
with urllib.request.urlopen(u,timeout=60) as f:a=json.load(f)
(R/'collection_metadata.json').write_text(json.dumps(a,indent=2));print(a.keys(),flush=True)
for d in a.get('datasets',[]):
 if d.get('dataset_id')=='53d208b0-2cfd-4366-9866-c3c6114081bc' or 'All Cells' in d.get('title',''):
  (R/'dataset_metadata.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/download.py — 원문 전문 (2,100 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/download.py", "sha256": "4b991acb386eb2afd20cf69b398709754363756212d536249a9053d57199dea0", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import urllib.request,json,concurrent.futures,hashlib,shutil,time
R=Path('/home/hyein/ai-benchmark/scgpt/data_external/tabula_sapiens');url='https://cellxgene-data-public.s3.amazonaws.com/cell-census/2024-07-01/h5ads/53d208b0-2cfd-4366-9866-c3c6114081bc.h5ad';size=10026988999;p=R/'tabula_sapiens_20240701.h5ad';chunks=R/'download_chunks';chunks.mkdir(exist_ok=True);step=32*1024**2
with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as f:etag=f.headers['ETag'];assert int(f.headers['Content-Length'])==size
(R/'download_source.json').write_text(json.dumps({'url':url,'census_version':'2024-07-01','expected_bytes':size,'etag':etag,'dataset_id':'53d208b0-2cfd-4366-9866-c3c6114081bc','expected_cells_from_official_census_documentation':483152},indent=2))
def get(a):
 b=min(a+step,size)-1;dest=chunks/str(a)
 if dest.exists() and dest.stat().st_size==b-a+1:return dest
 for retry in range(5):
  try:
   req=urllib.request.Request(url+f'?benchmark_chunk={a}',headers={'Range':f'bytes={a}-{b}','If-Match':etag})
   with urllib.request.urlopen(req,timeout=180) as f:
    assert f.status==206 and f.headers['Content-Range'].startswith(f'bytes {a}-{b}/')
    with dest.open('wb') as out:shutil.copyfileobj(f,out,length=1024**2)
   assert dest.stat().st_size==b-a+1;return dest
  except Exception:
   if retry==4:raise
   time.sleep(1)
if not p.exists():
 ranges=range(0,size,step)
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  for i,d in enumerate(pool.map(get,ranges)):
   if i%12==0:print('CHUNKS',i+1,'/',len(ranges),flush=True)
 tmp=p.with_suffix('.part')
 with tmp.open('wb') as out:
  for a in ranges:
   with (chunks/str(a)).open('rb') as f:shutil.copyfileobj(f,out,length=1024**2)
 assert tmp.stat().st_size==size;tmp.rename(p)
s=json.loads((R/'download_source.json').read_text());s['actual_bytes']=p.stat().st_size
with p.open('rb') as f:s['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
(R/'download_source.json').write_text(json.dumps(s,indent=2));print('COMPLETE',s['actual_bytes'],s['sha256'],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/finalize.py — 원문 전문 (5,520 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/finalize.py", "sha256": "af0870b22df477c08140f85d6d3809bc97b0bf2d83db2746015771fe218f548a", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Finish existing scGPT run: verify CPU outputs, update reports, then exit. No new GPU experiments."""
from pathlib import Path
import subprocess,json,sys,time,shlex
ROOT=Path('/home/hyein/ai-benchmark');R=ROOT/'scgpt';run=sys.argv[1];D=R/'results'/run
while not (D/'inference_exit.json').exists():time.sleep(20)
exit_info=json.loads((D/'inference_exit.json').read_text());s=json.loads((D/'summary.json').read_text())
if exit_info['exit_code']!=0 or s['status']!='complete':
 (D/'finalization_status.json').write_text(json.dumps({'status':'not_complete','exit':exit_info,'summary_status':s['status']}));sys.exit(1)
code='''from pathlib import Path
import numpy as np,json,hashlib
D=Path('/workspace/scgpt/results')/RUN
s=json.loads((D/'summary.json').read_text());old=Path('/workspace/scgpt/results')/s['resume_from_run'];a=np.load(D/'embeddings_batch8.npy',mmap_mode='r');b=np.load(D/'embeddings_batch256.npy',mmap_mode='r');original=np.load(old/'embeddings_batch8.npy',mmap_mode='r');assert a.shape==b.shape==(300000,512);same=bool(np.array_equal(a[:100000],original[:100000]));assert same;minimum=1.;maxdiff=0.;normerror=0.
for i in range(0,300000,4096):
 x=np.asarray(a[i:i+4096]);y=np.asarray(b[i:i+4096]);assert np.isfinite(x).all() and np.isfinite(y).all();minimum=min(minimum,float((x*y).sum(1).min()));maxdiff=max(maxdiff,float(np.abs(x-y).max()));normerror=max(normerror,float(np.abs(np.linalg.norm(x,axis=1)-1).max()),float(np.abs(np.linalg.norm(y,axis=1)-1).max()))
assert minimum>.999 and normerror<1e-4
r={'verified_unique_cells_per_batch':300000,'output_shape_per_batch':[300000,512],'all_outputs_finite':True,'prepause_first100000_outputs_exactly_preserved':same,'all_cells_minimum_batch_cosine_similarity':minimum,'all_cells_max_abs_batch_embedding_difference':maxdiff,'all_cells_max_unit_norm_error':normerror,'verification_device':'CPU'};(D/'final_output_verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''.replace('RUN',repr(run))
cmd=['docker','exec','--user','1000:1000','-e','CUDA_VISIBLE_DEVICES=','scgpt-spark','python','-c',code]
with (D/'final_output_verification.log').open('w') as f:rc=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
assert rc==0,'final output verification failed; see log'
subprocess.run([sys.executable,str(R/'scripts/large_scale/report.py'),run],check=True)
s=json.loads((D/'summary.json').read_text());whole=[]
for batch in [8,256]:
 c=next(c for c in s['conditions'] if c['batch_size']==batch and c['unique_cells']==300000);parts=[c]
 if c.get('resume_start_cell',0):parts.insert(0,next(p for p in s['conditions'] if p['batch_size']==batch and p['unique_cells']==c['resume_start_cell']))
 seconds=sum(p['seconds'] for p in parts);samples=sum(p['samples'] for p in parts);whole.append({'batch':batch,'unique_cells':300000,'active_seconds_sum':seconds,'cells_per_second':300000/seconds,'allocated_peak_gib_max_segments':max(p['memory']['peak_allocated_gib'] for p in parts),'reserved_peak_gib_max_segments':max(p['memory']['peak_reserved_gib'] for p in parts),'gpu_util_mean_sample_weighted':sum(p['gpu_util_avg']*p['samples'] for p in parts)/samples,'gpu_util_max':max(p['gpu_util_max'] for p in parts),'segments':len(parts)})
(D/'whole_dataset_comparison.json').write_text(json.dumps(whole,indent=2))
report=R/'reports/scGPT_actual_300k.md'
with report.open('a') as f:
 f.write('\n## 30만 고유세포 전체 처리 최종 비교\n\nbatch8은 전날10만+재개20만의 유효 처리시간 합이며 야간 중단·모델 재적재·warmup은 제외한다. batch256은 한 세션의30만 누적시간이다. 두 조건의 시간 범위는 전처리cache 이후 GPU처리/반환/기록이며, 세션 분할에 따른 차이는 남는다. 메모리는 각 구간의 reset peak 중 최대이고 baseline 차이는 원본 행에서 확인한다. GPU 평균은 표본 수로 가중했다.\n\n|batch|고유세포|유효초 합|세포/초|allocated peak GiB|reserved peak GiB|GPU 평균/최대%|세션 구간 수|\n|---|---|---|---|---|---|---|---|\n')
 for w in whole:f.write(f"|{w['batch']}|300000|{w['active_seconds_sum']:.2f}|{w['cells_per_second']:.2f}|{w['allocated_peak_gib_max_segments']:.3f}|{w['reserved_peak_gib_max_segments']:.3f}|{w['gpu_util_mean_sample_weighted']:.2f}/{w['gpu_util_max']:.0f}|{w['segments']}|\n")
 f.write('\n[전체30만세포 출력 검증](../results/'+run+'/final_output_verification.json): 각 조건300000×512, 전체finite·단위norm·세포별batch간cosine 검사 및 전날10만출력의 정확한 보존 확인. 새로운 GPU실험은 이어서 시작하지 않는다.\n')
for name in ['MEMORY_RESULTS.md','RESULTS.md','STATUS.md']:
 p=ROOT/name;text=p.read_text();text=text.replace('— 사용자 승인으로 일시중단. batch8의10만세포 완료·출력검증, 나머지 및batch256 미완료.','— 2026-10-28 재개 후 batch8/256 실제30만세포 처리·최종출력검증 완료. 최신 보고서 참조.')
 text+='\n2026-10-28 최종 완료: scGPT 실제30만 고유세포 batch8/256 추론 및 전체출력검증 완료. 실행 `'+run+'`, [최종 보고서](scgpt/reports/scGPT_actual_300k.md). 후속 실험 범위는 scGPT만이며 다른 모델은 진행하지 않는다. 최신 상태는 이 항목을 우선한다.\n';p.write_text(text)
(D/'finalization_status.json').write_text(json.dumps({'status':'complete','output_verification_passed':True,'reports_updated':True,'new_gpu_experiments_started':False}));print('FINALIZED',run,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/inference.py — 원문 전문 (5,543 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/inference.py", "sha256": "67dd1ecf01629fa5d7f0668cbf84a11abe049e294694a58ff4a6339c7cb5fe3a", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,time,json,gc,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scgpt.model import TransformerModel
from gpu_support import Meter,write_json
R=Path('/workspace/scgpt');P=R/'data_external/tabula_sapiens/prepared_300k_seed42';D=R/'results'/os.environ['RUN_ID'];D.mkdir(exist_ok=True);torch.set_num_threads(8);torch.manual_seed(42);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;meter=Meter(D);S={'status':'running','dataset':'Tabula Sapiens actual human cells, Census2024-07-01','conditions':[],'settings':{'batches':[8,256],'dtype':'FP32','tf32':False,'seed':42,'max_length':1200,'padding':'actual batch maximum cached nonpad length','task':'pretrained whole-human normalized CLS embeddings, no cell-type classifier and no test accuracy claim','count_points':[10000,100000,300000],'measurement':'cumulative nested prefix endpoints within one300k pass per batch, includes cached batch construction/transfers/forward/CPU outputs/memmap writes','excludes':'download, h5ad audit, tokenization cache, model load, warmup','input_policy':'same cached tokens/values and same original cells/order for both batch settings'}}
def save():write_json(D/'summary.json',S)
try:
 audit=json.loads((P/'input_audit.json').read_text());assert audit['selected_unique_cells']==300000;S['input_audit']=str(P/'input_audit.json');gcache=np.load(P/'gene_ids.npy',mmap_mode='r');vcache=np.load(P/'expression_bins.npy',mmap_mode='r');lengths=np.load(P/'lengths.npy',mmap_mode='r');source=R/'results'/(R/'results/SUPPLEMENT_LATEST').read_text().strip();bundle=R/'models/finetuned'/source.name/'epoch_001';vocab=json.loads((bundle/'vocab.json').read_text());config=json.loads((bundle/'model_config.json').read_text());config['n_cls']=1;config['do_mvc']=True;meter.phase='model_loading';t=time.time();model=TransformerModel(**config,vocab=vocab);state=torch.load(R/'models/whole-human/best_model.pt',map_location='cpu',weights_only=True);mapped={}
 for key,value in state.items():
  for a,b in [(r'self_attn\._impl\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\._impl\.out_proj\.','self_attn.out_proj.')]:key=re.sub(a,b,key)
  mapped[key]=value
 current=model.state_dict();compatible={k:v for k,v in mapped.items() if k in current and current[k].shape==v.shape};required=[k for k in current if k.startswith(('encoder.','value_encoder.','transformer_encoder.'))];assert set(required)<=set(compatible);model.load_state_dict(compatible,strict=False);del state,mapped,current,compatible;gc.collect();model.float().cuda().eval();torch.cuda.synchronize();S['model_loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']=sum(p.numel() for p in model.parameters());S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'scgpt':__import__('scgpt').__version__,'gpu':torch.cuda.get_device_name(0)};pad=vocab[config['pad_token']];save()
 def forward(start,end):
  width=int(lengths[start:end].max());g=torch.from_numpy(np.array(gcache[start:end,:width],dtype=np.int64)).cuda();v=torch.from_numpy(np.array(vcache[start:end,:width],dtype=np.float32)).cuda();h=model._encode(g,v,src_key_padding_mask=g.eq(pad));z=torch.nn.functional.normalize(h[:,0,:],p=2,dim=1).cpu().numpy();assert np.isfinite(z).all();return z,width
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():z,_=forward(0,8)
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del z;model.cur_gene_token_embs=None
 for batch in [8,256]:
  target=D/f'embeddings_batch{batch}.npy';emb=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(300000,512));model.cur_gene_token_embs=None;gc.collect();t,base=meter.begin(f'actual_cells_batch{batch}');start=0;shapes={};log_next=4096;paused=0.
  with torch.inference_mode():
   for endpoint in [10000,100000,300000]:
    while start<endpoint:
     end=min(start+batch,endpoint);z,width=forward(start,end);emb[start:end]=z;shapes[f'{end-start}x{width}']=shapes.get(f'{end-start}x{width}',0)+1;start=end
     if start>=log_next:print('REAL_CELLS',batch,start,'/',300000,'seconds',round(time.time()-t-paused,1),flush=True);log_next=start+4096
    r=meter.end(t,base,name=f'actual_cells_n{endpoint}_batch{batch}',batch_size=batch,unique_cells=endpoint,processed_cells=endpoint,measurement_repeats=1,actual_token_shapes=dict(shapes));r['seconds']-=paused;r['cells_per_second']=endpoint/r['seconds'];S['conditions'].append(r);print('PREFIX_COMPLETE',json.dumps({k:v for k,v in r.items() if k not in ['actual_token_shapes','memory','baseline']}),flush=True);p=time.time();save();emb.flush();paused+=time.time()-p
  emb.flush();del emb;model.cur_gene_token_embs=None;gc.collect();torch.cuda.empty_cache()
 a=np.load(D/'embeddings_batch8.npy',mmap_mode='r');b=np.load(D/'embeddings_batch256.npy',mmap_mode='r');ix=np.random.default_rng(42).choice(300000,256,replace=False);x=np.array(a[ix]);y=np.array(b[ix]);S['output_verification']={'sampled_actual_cells':256,'max_abs_embedding_difference_between_batches':float(np.abs(x-y).max()),'minimum_cosine_similarity':float((x*y).sum(1).min()),'output_shape':[300000,512],'all_outputs_finite_checked_during_forward':True};assert S['output_verification']['minimum_cosine_similarity']>.999;S['status']='complete';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/launch.py — 원문 전문 (1,990 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/launch.py", "sha256": "3488ebd0224604a3f2e0454a220e2ab1831b211d2741a84772083fbd613f382d", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import subprocess,time,json,datetime,shlex,os
R=Path('/home/hyein/ai-benchmark/scgpt');run='actual_300k_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LARGE_LATEST').write_text(run+'\n');print(run,flush=True)
def active(script):
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if any(a.decode(errors='ignore').endswith(script) for a in args):return True
 return False
while True:
 p=R/'data_external/tabula_sapiens/download_source.json'
 try:meta=json.loads(p.read_text());ready=len(meta.get('sha256',''))==64
 except Exception:ready=False
 if ready:break
 if not active('large_scale/download.py'):raise RuntimeError('download stopped before completion')
 time.sleep(10)
while active('scripts/resume_full_realdata.py'):time.sleep(10)
def runstage(name,args):
 cmd=['docker','exec','--user','1000:1000','-e','MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba',*args];(D/(name+'_command.json')).write_text(json.dumps(cmd,indent=2));start=time.time()
 with (D/(name+'.log')).open('w') as f:code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 (D/(name+'_exit.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start}));print(name,'EXIT',code,flush=True)
 if code:raise RuntimeError((name,code))
runstage('prepare',['-e','CUDA_VISIBLE_DEVICES=','scgpt-spark','python','-u','/workspace/scgpt/scripts/large_scale/prepare.py'])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU job is running'
runstage('inference',['-e','RUN_ID='+run,'scgpt-spark','python','-u','/workspace/scgpt/scripts/large_scale/inference.py'])
(D/'exit_code.json').write_text(json.dumps({'exit_code':0}));print('REAL300K_COMPLETE',flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/pause.py — 원문 전문 (2,009 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/pause.py", "sha256": "3e2ff120f03a8a44ee41e32f6d0e0029321039e0ac4835e0873d619d92326dbb", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Explicitly pause ONLY this scGPT experiment, retaining completed measured prefixes."""
from pathlib import Path
import os,signal,json,subprocess,time,sys
R=Path('/home/hyein/ai-benchmark/scgpt');run=sys.argv[1] if len(sys.argv)>1 else (R/'results/LARGE_LATEST').read_text().strip();D=R/'results'/run
pids=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
 try:
  pid=int(line.strip());cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes();env=(Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
  if (b'large_scale/inference.py' in cmd or b'large_scale/resumable_inference.py' in cmd) and ('RUN_ID='+run).encode() in env:pids.append(pid)
 except (OSError,ValueError):pass
assert len(pids)==1,('expected one matching experiment',pids);pid=pids[0];os.kill(pid,signal.SIGSTOP)
try:
 s=json.loads((D/'summary.json').read_text());assert s['status']=='running';prefix={str(b):max([c['unique_cells'] for c in s['conditions'] if c['batch_size']==b],default=0) for b in [8,256]}
 for b,n in prefix.items():
  if n:
   with (D/f'embeddings_batch{b}.npy').open('r+b') as f:os.fsync(f.fileno())
 checkpoint={'source_run':run,'conditions':s['conditions'],'completed_prefixes':prefix,'saved_at':time.time(),'note':'Only completed measured milestones are durable resume points. Work after the latest milestone will be repeated. Download and token cache reused. Resume timing is a separate segment.'};(D/'pause_checkpoint.json').write_text(json.dumps(checkpoint,indent=2));s['status']='paused';s['pause_checkpoint']=checkpoint;(D/'summary.json').write_text(json.dumps(s,indent=2));os.kill(pid,signal.SIGTERM)
finally:
 try:os.kill(pid,signal.SIGCONT)
 except ProcessLookupError:pass
for _ in range(100):
 if not Path('/proc',str(pid)).exists():break
 time.sleep(.1)
assert not Path('/proc',str(pid)).exists(),'process has not exited';print(json.dumps(checkpoint,indent=2));print('PAUSED GPU process exited; use resume.py',run)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/prepare.py — 원문 전문 (5,474 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/prepare.py", "sha256": "f5a10495f6d64b55f763bd3048475ac2cad61ecad2af64c391ab92484da22b0a", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,time,hashlib,collections
import anndata as ad,numpy as np,torch,psutil
from scgpt.data_collator import DataCollator
R=Path('/workspace/scgpt');D=R/'data_external/tabula_sapiens';P=D/'prepared_300k_seed42';P.mkdir(exist_ok=True);source=D/'tabula_sapiens_20240701.h5ad';a=ad.read_h5ad(source,backed='r');print('SOURCE',a.shape,list(a.var.columns),list(a.obs.columns),flush=True)
if (P/'complete.json').exists():print('ALREADY_PREPARED',flush=True);raise SystemExit(0)
basic=R/'results'/(R/'results/SUPPLEMENT_LATEST').read_text().strip();bundle=R/'models/finetuned'/basic.name/'epoch_001';vocab=json.loads((bundle/'vocab.json').read_text());args=json.loads((R/'models/whole-human/args.json').read_text());pad=vocab[args['pad_token']];cls=vocab['<cls>'];np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(8)
X=a.raw.X if a.raw is not None else a.X;var=a.raw.var if a.raw is not None else a.var;column=next((k for k in ['feature_name','gene_symbols','gene_symbol','gene_name'] if k in var),None);names=np.asarray(var[column].astype(str) if column else var.index.astype(str));counts=collections.Counter(names);mapping=np.array([vocab.get(name,-1) if counts[name]==1 else -1 for name in names],np.int32);assert (mapping>=0).sum()>1000,'Gene symbol mapping failed'
valid=np.ones(a.n_obs,bool)
if 'is_primary_data' in a.obs:valid &= a.obs.is_primary_data.astype(str).str.lower().isin(['true','1']).to_numpy()
if 'organism' in a.obs:valid &= a.obs.organism.astype(str).eq('Homo sapiens').to_numpy()
eligible=np.flatnonzero(valid);assert len(eligible)>=300000,len(eligible);selected=np.random.default_rng(42).permutation(eligible)[:300000];np.save(P/'selected_source_rows.npy',selected);meta=a.obs.iloc[selected].copy();meta.insert(0,'source_row',selected);meta.index.name='cell_id';meta.to_csv(P/'cells.tsv',sep='\t');assert meta.index.is_unique
n=len(selected);genes=np.lib.format.open_memmap(P/'gene_ids.npy',mode='w+',dtype=np.int32,shape=(n,1200));values=np.lib.format.open_memmap(P/'expression_bins.npy',mode='w+',dtype=np.int8,shape=(n,1200));lengths=np.lib.format.open_memmap(P/'lengths.npy',mode='w+',dtype=np.uint16,shape=(n,));nonzero=np.lib.format.open_memmap(P/'nonzero_genes.npy',mode='w+',dtype=np.uint32,shape=(n,));genes[:]=pad;values[:]=int(args['pad_value']);collator=DataCollator(do_padding=True,pad_token_id=pad,pad_value=args['pad_value'],do_mlm=False,do_binning=True,max_length=1200,sampling=True,keep_first_n_tokens=1);order=np.argsort(selected);start=time.time();sample_stats=None
for offset in range(0,n,256):
 positions=order[offset:offset+256];src=selected[positions];chunk=X[src,:].tocsr();examples=[]
 assert np.isfinite(chunk.data).all() and (chunk.data>=0).all()
 if sample_stats is None:sample_stats={'dtype':str(chunk.dtype),'values_min':float(chunk.data.min()),'values_max':float(chunk.data.max()),'integer_like_sample':bool(np.allclose(chunk.data,np.rint(chunk.data),atol=1e-5)),'matrix_type':str(type(X))}
 for j in range(len(positions)):
  lo,hi=chunk.indptr[j:j+2];idx=chunk.indices[lo:hi];v=chunk.data[lo:hi];mask=(mapping[idx]>=0)&(v>0);ids=mapping[idx[mask]];val=v[mask].astype(np.float32);assert len(ids)>0,('zero mapped genes',int(src[j]));nonzero[positions[j]]=len(ids);examples.append({'genes':torch.from_numpy(np.r_[np.int64(cls),ids.astype(np.int64)]),'expressions':torch.from_numpy(np.r_[np.float32(args['pad_value']),val])})
 batch=collator(examples);g=batch['gene'].numpy();v=batch['expr'].numpy();width=g.shape[1];assert v.min()>=-2 and v.max()<=50;genes[positions,:width]=g.astype(np.int32);values[positions,:width]=v.astype(np.int8);lengths[positions]=(g!=pad).sum(1)
 if offset%10240==0:print('TOKENIZED',offset+len(positions),'/',n,'seconds',round(time.time()-start,1),flush=True);genes.flush();values.flush()
 if psutil.virtual_memory().available<12*2**30:raise RuntimeError('CPU preparation stopped below12GiB available')
genes.flush();values.flush();lengths.flush();nonzero.flush();np.save(P/'feature_vocab_ids.npy',mapping);audit={'source_shape':list(a.shape),'matrix_source':'raw.X' if a.raw is not None else 'X','gene_name_column':column,'matched_unique_features':int((mapping>=0).sum()),'ambiguous_symbol_features_dropped':int(sum(counts[name]>1 for name in names)),'eligible_primary_human_cells':len(eligible),'selected_unique_cells':n,'source_obs_id_unique':bool(a.obs_names.is_unique),'sample_expression':sample_stats,'seed':42,'preprocessing':'official51-bin DataCollator; nonzero genes; actual per-cell sampling to1200 incl CLS; cached once for all runs; duplicate gene symbols dropped and counted','tokenization_seconds':time.time()-start,'subsets':{str(k):{'cells':k,'tokens_min':int(lengths[:k].min()),'tokens_mean':float(lengths[:k].mean()),'tokens_max':int(lengths[:k].max()),'nonzero_genes_mean':float(nonzero[:k].mean()),'truncated_cells':int((nonzero[:k]+1>1200).sum()),'distributions':{col:{str(x):int(y) for x,y in meta.iloc[:k][col].value_counts().items()} for col in ['assay','tissue','cell_type','donor_id'] if col in meta}} for k in [10000,100000,300000]},'source_download':json.loads((D/'download_source.json').read_text()),'interpretation':'actual human cell embeddings resource test; cell-type labels are metadata only; no pancreas classifier or accuracy claim'};(P/'input_audit.json').write_text(json.dumps(audit,indent=2));(P/'complete.json').write_text(json.dumps({'cells':n,'finished':time.time()}));a.file.close();print('PREPARED',n,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/report.py — 원문 전문 (6,551 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/report.py", "sha256": "7e4412df1ca29c40cc86a43bdf84af035efae38af8a79f7c4ba7b9ca14633286", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,csv,sys,time
R=Path('/home/hyein/ai-benchmark/scgpt');run=sys.argv[1];D=R/'results'/run;P=R/'data_external/tabula_sapiens/prepared_300k_seed42';s=json.loads((D/'summary.json').read_text());a=json.loads((P/'input_audit.json').read_text());out=[]
def add(x=''):out.append(x)
add('# scGPT 실제30만 세포 임베딩 및 자원 측정');add();add(f"상태: **{s['status']}**. 실행 `{run}`. [원본 summary](../results/{run}/summary.json) · [입력 검사](../data_external/tabula_sapiens/prepared_300k_seed42/input_audit.json)");add();add('## 실제 입력과 전처리');add();add(f"Tabula Sapiens 고정 Census2024-07-01 원본 {a['source_shape'][0]:,}세포 × {a['source_shape'][1]:,} features에서 seed42로 **300,000개 고유 실제 세포**를 선택했다. 반복 복제한 입력이 아니다. source raw.X를 사용하며 vocabulary와 일치한 유전자 {a['matched_unique_features']:,}개를 처리한다. 원본 파일 SHA256은 download_source.json에 저장했다. raw.X의 의미와 count provenance는 원본 metadata와 구분해 확인하며, 정수형 값 표본만으로 원시 count를 확정하지 않는다.");add();add('공식51-bin 전처리, 비영 유전자, 최대1200토큰(CLS포함). 선택·유전자 sampling·binning 결과는 한번 cache해 두 배치 조건에 동일하게 제공한다. 실제 GPU 입력은 배치 내 최대 유효길이로 padding한다. 1만/10만/30만은 동일 순서의 중첩 부분집합이며, 각 batch의30만세포 단일 pass에서 누적 지점을 측정한다. 독립적인3회 실행이 아니다.');add();add('|고유 세포|평균 실제 토큰|최소/최대 토큰|길이 제한으로 sampling된 세포|');add('|---|---|---|---|')
for k,v in a['subsets'].items():add(f"|{k}|{v['tokens_mean']:.2f}|{v['tokens_min']}/{v['tokens_max']}|{v['truncated_cells']}|")
add();add('tissue·assay·cell_type 분포와 실제 cell ID/원본 행은 input_audit.json 및 cells.tsv에 보존했다. cell_type은 metadata이며 이번 임베딩 추론의 정답 분류 평가가 아니다.');add();add('재개 실행인 경우 기존 완료 지점의 출력을 재사용하고 나머지만 새 구간으로 측정한다. 해당 행의 시간/처리량은 이번 구간 처리 세포 수 기준이며 전체 누적 세포 처리 시간으로 해석하지 않는다.');add();add('## 모델과 측정');add();add(f"whole-human pretrained scGPT, 구성 파라미터 {s.get('parameters','N/A'):,}, FP32, TF32=False, seed42. 동일 모델을 유지하며 batch8→256 순차 실행한다. classifier fine-tuning은 수행하지 않는다. pretrained encoder/value encoder/transformer의 필수 가중치 이름·shape를 검증해 적재했다. 모델 출처와 가중치 파일은 기존 기본 보고서 및 models/whole-human에 있다.");add();add('계측 범위: cache의 배치 구성·전송·forward·정규화CLS 반환·CPU memmap 저장. 다운로드/입력검사/tokenization/model loading/warmup 제외. CUDA 동기화 경계를 사용한다. 누적 지점의 JSON/flush 일시 정지는 시간에서 제외하지만 GPU 표본 구간에는 포함될 수 있다. RSS와 가용 RAM은0.5초 대기+조회 시간 간격의 sampled 값이다. GPU-Util은 nvidia-smi 장치 전체값으로 디스플레이 등도 포함될 수 있다. 다른 GPU compute job과 겹치지 않게 시작했다.');add();base=s.get('model_only_postload',{});add(f"모델 적재 직후: allocated {base.get('allocated_gib',0):.4f} / reserved {base.get('reserved_gib',0):.4f} GiB, RSS {base.get('rss_gib',0):.3f} GiB, 시스템 MemAvailable {base.get('system_available_gib',0):.3f} GiB. CUDA 초기화 및 CPU token cache mmap 열기를 포함하며 전체 cache를 GPU에 올리지 않았다.");add();add('|batch|누적 출력 세포|이번 측정 구간 처리 세포|초|세포/초|allocated peak GiB|추가 peak GiB|reserved peak GiB|GPU 평균/최대%|RSS sampled peak GiB|가용 최소 GiB|');add('|---|---|---|---|---|---|---|---|---|---|---|');flat=[]
for c in s['conditions']:
 m=c['memory'];f={'batch':c['batch_size'],'cells':c['unique_cells'],'processed_cells_this_segment':c.get('processed_cells',c['unique_cells']),'seconds':c['seconds'],'cells_per_second':c['cells_per_second'],'allocated_peak_gib':m['peak_allocated_gib'],'additional_from_same_condition_baseline_gib':c['peak_delta_gib'],'reserved_peak_gib':m['peak_reserved_gib'],'gpu_util_avg':c['gpu_util_avg'],'gpu_util_max':c['gpu_util_max'],'rss_sampled_peak_gib':c['rss_sampled_peak_gib'],'available_sampled_min_gib':c['available_sampled_min_gib']};flat.append(f);add('|'+ '|'.join(str(round(v,4)) if isinstance(v,float) else str(v) for v in f.values())+'|')
add();add('추가 peak은 해당 조건 시작 baseline 대비 차이이며 모델 적재 직후 값과 동일하지 않을 수 있다.');add('CUDA allocated/reserved는 프로세스 allocator 수치다. 전체121.688GiB는 CPU/GPU 공유 RAM이며 별도 전용 VRAM이 아니다. RSS와 CUDA를 더하지 않는다. 80% 도달은 성공 조건이 아니며 실제 관측치를 보고한다. 세포 수 증가와 batch 증가는 다른 축이다.');add();add('## 출력 검증과 한계');add();add(json.dumps(s.get('output_verification',{'status':'pending'}),ensure_ascii=False));add();add('출력은 각 batch의 embeddings_batchB.npy, 300000×512 float32이다. 각 forward에서 유한값을 검사하며, 완료 후256개 실제 세포의 배치 간 embedding을 비교한다. 실제30만세포 임베딩 처리 결과이며 지도학습 정확도·donor 독립 평가·사전학습 데이터와의 독립성을 입증하지 않는다. 별도 pancreas 전체 학습 결과와 구분한다.');add();add('[공식 scGPT](https://github.com/bowang-lab/scGPT) · [공식 Census 원본 다운로드](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.download_source_h5ad.html) · [Tabula Sapiens collection](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5)');add();add(f"실행 코드: scripts/large_scale/prepare.py, inference.py. 로그·GPU 시계열·설정·예측은 results/{run}에 보존한다. import 경로 오류로 최초 시작 전 실패한 로그도 보존하고 retry1에서 실행했다.")
(R/'reports/scGPT_actual_300k.md').write_text('\n'.join(out)+'\n')
if flat:
 with (R/'reports/actual_300k_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(s['status'],len(flat),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/resumable_inference.py — 원문 전문 (6,223 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/resumable_inference.py", "sha256": "21a2fe15e04c4b1c6fffc7586bedc67b97c2919bbd5c02b23cc6822b828741bc", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,time,json,gc,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from scgpt.model import TransformerModel
from gpu_support import Meter,write_json
R=Path('/workspace/scgpt');P=R/'data_external/tabula_sapiens/prepared_300k_seed42';D=R/'results'/os.environ['RUN_ID'];D.mkdir(exist_ok=True);torch.set_num_threads(8);torch.manual_seed(42);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;meter=Meter(D);S={'status':'running','dataset':'Tabula Sapiens actual human cells, Census2024-07-01','conditions':[],'settings':{'batches':[8,256],'dtype':'FP32','tf32':False,'seed':42,'max_length':1200,'padding':'actual batch maximum cached nonpad length','task':'pretrained whole-human normalized CLS embeddings, no cell-type classifier and no test accuracy claim','count_points':[10000,100000,300000],'measurement':'cumulative nested prefix endpoints within one300k pass per batch, includes cached batch construction/transfers/forward/CPU outputs/memmap writes','excludes':'download, h5ad audit, tokenization cache, model load, warmup','input_policy':'same cached tokens/values and same original cells/order for both batch settings'}}
def save():write_json(D/'summary.json',S)
try:
 previous=json.loads((D/'resume_state.json').read_text()) if (D/'resume_state.json').exists() else None
 if previous:
  S['conditions']=previous['conditions'];S['resume_from_run']=previous['source_run'];S['settings']['measurement']='completed original prefix records plus separately timed resumed remainder; processed_cells and resume_start_cell define each measured interval, not uninterrupted cumulative timing'
 audit=json.loads((P/'input_audit.json').read_text());assert audit['selected_unique_cells']==300000;S['input_audit']=str(P/'input_audit.json');gcache=np.load(P/'gene_ids.npy',mmap_mode='r');vcache=np.load(P/'expression_bins.npy',mmap_mode='r');lengths=np.load(P/'lengths.npy',mmap_mode='r');source=R/'results'/(R/'results/SUPPLEMENT_LATEST').read_text().strip();bundle=R/'models/finetuned'/source.name/'epoch_001';vocab=json.loads((bundle/'vocab.json').read_text());config=json.loads((bundle/'model_config.json').read_text());config['n_cls']=1;config['do_mvc']=True;meter.phase='model_loading';t=time.time();model=TransformerModel(**config,vocab=vocab);state=torch.load(R/'models/whole-human/best_model.pt',map_location='cpu',weights_only=True);mapped={}
 for key,value in state.items():
  for a,b in [(r'self_attn\._impl\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\._impl\.out_proj\.','self_attn.out_proj.')]:key=re.sub(a,b,key)
  mapped[key]=value
 current=model.state_dict();compatible={k:v for k,v in mapped.items() if k in current and current[k].shape==v.shape};required=[k for k in current if k.startswith(('encoder.','value_encoder.','transformer_encoder.'))];assert set(required)<=set(compatible);model.load_state_dict(compatible,strict=False);del state,mapped,current,compatible;gc.collect();model.float().cuda().eval();torch.cuda.synchronize();S['model_loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']=sum(p.numel() for p in model.parameters());S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'scgpt':__import__('scgpt').__version__,'gpu':torch.cuda.get_device_name(0)};pad=vocab[config['pad_token']];save()
 def forward(start,end):
  width=int(lengths[start:end].max());g=torch.from_numpy(np.array(gcache[start:end,:width],dtype=np.int64)).cuda();v=torch.from_numpy(np.array(vcache[start:end,:width],dtype=np.float32)).cuda();h=model._encode(g,v,src_key_padding_mask=g.eq(pad));z=torch.nn.functional.normalize(h[:,0,:],p=2,dim=1).cpu().numpy();assert np.isfinite(z).all();return z,width
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():z,_=forward(0,8)
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del z;model.cur_gene_token_embs=None
 for batch in [8,256]:
  checkpoint=max([c['unique_cells'] for c in S['conditions'] if c['batch_size']==batch],default=0)
  if checkpoint==300000:continue
  target=D/f'embeddings_batch{batch}.npy';emb=np.lib.format.open_memmap(target,mode='r+' if checkpoint else 'w+',dtype=np.float32,shape=(300000,512));model.cur_gene_token_embs=None;gc.collect();t,base=meter.begin(f'actual_cells_batch{batch}');start=checkpoint;shapes={};log_next=checkpoint+4096;paused=0.
  with torch.inference_mode():
   for endpoint in [10000,100000,300000]:
    if endpoint<=checkpoint:continue
    while start<endpoint:
     end=min(start+batch,endpoint);z,width=forward(start,end);emb[start:end]=z;shapes[f'{end-start}x{width}']=shapes.get(f'{end-start}x{width}',0)+1;start=end
     if start>=log_next:print('REAL_CELLS',batch,start,'/',300000,'seconds',round(time.time()-t-paused,1),flush=True);log_next=start+4096
    r=meter.end(t,base,name=f'actual_cells_n{endpoint}_batch{batch}',batch_size=batch,unique_cells=endpoint,processed_cells=endpoint-checkpoint,resume_start_cell=checkpoint,measurement_repeats=1,actual_token_shapes=dict(shapes));r['seconds']-=paused;r['cells_per_second']=(endpoint-checkpoint)/r['seconds'];S['conditions'].append(r);print('PREFIX_COMPLETE',json.dumps({k:v for k,v in r.items() if k not in ['actual_token_shapes','memory','baseline']}),flush=True);p=time.time();save();emb.flush();paused+=time.time()-p
  emb.flush();del emb;model.cur_gene_token_embs=None;gc.collect();torch.cuda.empty_cache()
 a=np.load(D/'embeddings_batch8.npy',mmap_mode='r');b=np.load(D/'embeddings_batch256.npy',mmap_mode='r');ix=np.random.default_rng(42).choice(300000,256,replace=False);x=np.array(a[ix]);y=np.array(b[ix]);S['output_verification']={'sampled_actual_cells':256,'max_abs_embedding_difference_between_batches':float(np.abs(x-y).max()),'minimum_cosine_similarity':float((x*y).sum(1).min()),'output_shape':[300000,512],'all_outputs_finite_checked_during_forward':True};assert S['output_verification']['minimum_cosine_similarity']>.999;S['status']='complete';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/resume.py — 원문 전문 (1,680 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/resume.py", "sha256": "880321b874d8e549d78d4693d210d37341923c53a9767fe8e82f109eee566d21", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Resume an explicitly paused run into a new folder without overwriting its results."""
from pathlib import Path
import json,sys,datetime,shutil,subprocess,shlex,time
R=Path('/home/hyein/ai-benchmark/scgpt');old=sys.argv[1] if len(sys.argv)>1 else (R/'results/LARGE_LATEST').read_text().strip();source=R/'results'/old;state=json.loads((source/'pause_checkpoint.json').read_text());assert json.loads((source/'summary.json').read_text())['status']=='paused';assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU compute job active'
run='actual_300k_resume_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir()
for b,n in state['completed_prefixes'].items():
 if n:shutil.copy2(source/f'embeddings_batch{b}.npy',D/f'embeddings_batch{b}.npy')
(D/'resume_state.json').write_text(json.dumps(state,indent=2));(R/'results/LARGE_LATEST').write_text(run+'\n');cmd=['docker','exec','--user','1000:1000','-e','MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba','-e','RUN_ID='+run,'scgpt-spark','python','-u','/workspace/scgpt/scripts/large_scale/resumable_inference.py'];(D/'inference_command.json').write_text(json.dumps(cmd,indent=2));print('RESUME',old,'->',run,state['completed_prefixes'],flush=True)
with (D/'inference.log').open('w') as f:code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
(D/'inference_exit.json').write_text(json.dumps({'exit_code':code,'ended_at':time.time()}));subprocess.run([sys.executable,str(R/'scripts/large_scale/report.py'),run],check=True);sys.exit(code)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/large_scale/watch.py — 원문 전문 (753 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/large_scale/watch.py", "sha256": "408105c2ba3904123322025616bed935a9c291943ee728de3670ece5167fd5ac", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import subprocess,time,json,sys
R=Path('/home/hyein/ai-benchmark');run=sys.argv[1];D=R/'scgpt/results'/run
while True:
 try:s=json.loads((D/'summary.json').read_text())
 except Exception:time.sleep(30);continue
 subprocess.run([sys.executable,str(R/'scgpt/scripts/large_scale/report.py'),run],check=True)
 if s['status'] in ['complete','failed','paused']:
  (D/'final_status.json').write_text(json.dumps({'status':s['status'],'observed_at':time.time(),'completed_conditions':len(s['conditions'])}))
  with (R/'STATUS.md').open('a') as f:f.write('\n실제30만세포 실행 '+run+' 최종 상태: '+s['status']+'; 조건수 '+str(len(s['conditions']))+'; 보고서 scgpt/reports/scGPT_actual_300k.md.\n')
  break
 time.sleep(30)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/launch_full_realdata.py — 원문 전문 (2,504 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/launch_full_realdata.py", "sha256": "7fcd5a15c2e5b884fa5e84086dbec1e43df91dcfbe2e733aef689be6891fc299", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import subprocess,datetime,json,time,sys
R=Path('/home/hyein/ai-benchmark/scgpt');run='realdata_full_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/REALDATA_LATEST').write_text(run+'\n');(D/'protocol.json').write_text(json.dumps({'dataset':'existing actual TOSICA pancreas demo_train10600/demo_test4218','split':'training8480/validation2120/test4218, same seed42','batch_cases':[8,284],'epochs':1,'max_length':1200,'padding':'original official collator dynamic max-in-batch, no forced1200 padding','precision':'FP32, TF32False','optimizer':'same original AdamW/lr1e-4/wd.01 clip1.0','initialization':'same fresh pretrained encoder + seed42 classifier per case','test_usage':'separate4218 evaluated from saved model; no model selection on test','warning':'different batch sizes imply different optimizer step counts, accuracy is not a batch-only causal effect'},indent=2));print(run,flush=True)
for batch in [8,284]:
 for mode in ['train','test']:
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'another GPU computation is active'
  out=f'/workspace/scgpt/results/{run}/batch_{batch}/'+('train' if mode=='train' else 'test');bundle=f'/workspace/scgpt/models/finetuned/{run}/batch_{batch}';args=f'--mode train --n-cells 0 --epochs 1 --model-out {bundle}' if mode=='train' else f'--mode predict --bundle {bundle}/epoch_001 --input /workspace/scgpt/data/pancreas_demo/demo_test.h5ad';cmd=['sg','docker','-c',f'docker exec --user 1000:1000 -e SCGPT_GUARD_DIR=/workspace/scgpt/results/{run} -e MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib -e NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba scgpt-spark python -u /workspace/scgpt/scripts/full_realdata.py {args} --out {out} --batch-size {batch} --max-length 1200 --seed 42'];(D/f'batch_{batch}').mkdir(exist_ok=True);(D/f'batch_{batch}/{mode}_command.json').write_text(json.dumps(cmd,indent=2));start=time.time();print('START',batch,mode,flush=True)
  with (D/f'batch_{batch}/{mode}.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
  (D/f'batch_{batch}/{mode}_exit.json').write_text(json.dumps({'exit_code':code,'total_subprocess_wall_seconds':time.time()-start,'finished':datetime.datetime.now().isoformat()}));print('EXIT',batch,mode,code,flush=True)
  if code:sys.exit(code)
(D/'exit_code.json').write_text(json.dumps({'exit_code':0}));print('ALL_FINISHED',flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/launch_memory_refine.py — 원문 전문 (799 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/launch_memory_refine.py", "sha256": "9593eb00e81a5c6a9efa072f1466e9155331b8e0069c6231060e44f81f59cba3", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,datetime,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/scgpt');assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();run='memory_batch_refine_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/MEMORY_REFINE_LATEST').write_text(run+'\n');cmd=['sg','docker','-c',f'docker exec --user 1000:1000 -e RUN_ID={run} scgpt-spark python -u /workspace/scgpt/scripts/memory_batch_refine.py'];(D/'command.json').write_text(json.dumps(cmd,indent=2));print(run,flush=True)
with (D/'run.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'exit_code.json').write_text(json.dumps({'exit_code':code}));print('EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/launch_memory_sweep.py — 원문 전문 (784 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/launch_memory_sweep.py", "sha256": "a599f7b196a46781b22f4d84def09d382b03541b0dd35d0d305cb935751a505f", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,datetime,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/scgpt');assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();run='memory_batch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/MEMORY_LATEST').write_text(run+'\n');cmd=['sg','docker','-c',f'docker exec --user 1000:1000 -e RUN_ID={run} scgpt-spark python -u /workspace/scgpt/scripts/memory_batch_sweep.py'];(D/'command.json').write_text(json.dumps(cmd,indent=2));print(run,flush=True)
with (D/'run.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'exit_code.json').write_text(json.dumps({'exit_code':code}));print('EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/launch_reload_supplement.py — 원문 전문 (695 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/launch_reload_supplement.py", "sha256": "3acde93e464dbbeb01f510748deaa21899d21fb40ff2e4444b4cec09caba5465", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/scgpt');run=(R/'results/SUPPLEMENT_LATEST').read_text().strip();D=R/'results'/run;assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();cmd=['sg','docker','-c',f'docker exec --user 1000:1000 -e RUN_ID={run} scgpt-spark python -u /workspace/scgpt/scripts/reload_supplement.py'];(D/'reload_command.json').write_text(json.dumps(cmd,indent=2))
with (D/'reload.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'reload_exit.json').write_text(json.dumps({'exit_code':code}));print('RELOAD_EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/launch_supplement.py — 원문 전문 (878 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/launch_supplement.py", "sha256": "d324d418971d6c647398cdd366b761f84d3a87fc52895ca441caddd5df338d5b", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,datetime,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/scgpt');assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();run='supplement_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/SUPPLEMENT_LATEST').write_text(run+'\n');cmd=['sg','docker','-c',f'docker exec --user 1000:1000 -e RUN_ID={run} -e MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib -e NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba scgpt-spark python -u /workspace/scgpt/scripts/supplement.py'];(D/'command.json').write_text(json.dumps(cmd,indent=2));print(run,flush=True)
with (D/'run.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'exit_code.json').write_text(json.dumps({'exit_code':code}));print('EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/launch_train_util.py — 원문 전문 (950 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/launch_train_util.py", "sha256": "cd056e3375c7f3cd147c8a0225ea702be574e0c2f97aa2f4fdd6d43e4709e24d", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/scgpt');run=(R/'results/SUPPLEMENT_LATEST').read_text().strip();D=R/'results'/run;assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();cmd=['sg','docker','-c',f'docker exec --user 1000:1000 -e MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib -e NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba scgpt-spark python -u /workspace/scgpt/scripts/train_util_supplement.py --out /workspace/scgpt/results/{run}/training_util --model-out /workspace/scgpt/models/finetuned/{run} --n-cells 1000 --epochs 1 --batch-size 8 --max-length 1200'];(D/'training_command.json').write_text(json.dumps(cmd,indent=2))
with (D/'training.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'training_exit.json').write_text(json.dumps({'exit_code':code}));print('TRAIN_EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/memory_batch_refine.py — 원문 전문 (7,341 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/memory_batch_refine.py", "sha256": "71bd415ddfdbcd1f646e42a990904af5f13da38867a4b138612c0a2932d37606", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Actual batch capacity probe; fixed FP32/1200 cached gene tokens, fresh start per batch."""
from pathlib import Path
import os,time,json,gc,random,re
import numpy as np,torch,psutil
from scgpt.model import TransformerModel
from gpu_support import Meter,write_json
R=Path('/workspace/scgpt');D=R/'results'/os.environ['RUN_ID'];D.mkdir(exist_ok=True);source=R/'results'/(R/'results/SUPPLEMENT_LATEST').read_text().strip();T=source/'training_util';bundle=R/'models/finetuned'/source.name/'epoch_001';torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.manual_seed(42);np.random.seed(42);random.seed(42);meter=Meter(D);S={'status':'running','experiment':'followup_batch_memory_capacity_80pct_boundary_refinement','conditions':[],'source_basic_run':source.name,'settings':{'mode':'training_resource_probe','dtype':'FP32','tf32':False,'max_length_including_CLS':1200,'optimizer':'AdamW lr1e-4 wd.01 betas.9/.999 eps1e-8 clip1.0','initialization':'same pretrained encoder and seed42 fresh13-class classifier restored before every batch','min_seconds_per_condition':15,'min_steps':3,'target_allocated_fraction_of_shared_RAM':[.8,.9],'minimum_system_available_gib':8,'swap_increase_stop_gib':2,'batch_only_variable':True,'quality_training':'not an accuracy experiment; updated probe weights saved separately','batch_repetition':'same nested firstB real training cells across steps, no replicated cells inside a batch'}}
def save():write_json(D/'summary.json',S)
try:
 config=json.loads((bundle/'model_config.json').read_text());vocab=json.loads((bundle/'vocab.json').read_text());model=TransformerModel(**config,vocab=vocab);raw=torch.load(R/'models/whole-human/best_model.pt',map_location='cpu',weights_only=True);mapped={}
 for key,value in raw.items():
  for a,b in [(r'self_attn\._impl\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\._impl\.out_proj\.','self_attn.out_proj.')]:key=re.sub(a,b,key)
  mapped[key]=value
 current=model.state_dict();core=('encoder.','value_encoder.','transformer_encoder.');required=[k for k in current if k.startswith(core)];loaded={k:mapped[k] for k in required if k in mapped and mapped[k].shape==current[k].shape};assert len(loaded)==len(required);model.load_state_dict(loaded,strict=False);del raw,mapped,current,loaded
 for name,p in model.named_parameters():p.requires_grad_(name.startswith(core+('cls_decoder.',)))
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.float().cuda();torch.cuda.synchronize();S['model_only_postload']=meter.memory();S['parameters']={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)};cache=np.load(T/'epoch_001/training_tokenized_inputs.npz');genes=cache['gene_ids'];values=cache['expression_bins'];labels=cache['labels'];assert genes.shape==(1000,1200) and values.shape==genes.shape and len(labels)==1000;pad=vocab[config['pad_token']];S['input_audit']={'unique_available_cells':1000,'gene_values_shape':list(genes.shape),'labels':{str(int(k)):int(n) for k,n in zip(*np.unique(labels,return_counts=True))},'source':str(T/'epoch_001/training_tokenized_inputs.npz'),'token_min':int((genes!=pad).sum(1).min()),'token_max':int((genes!=pad).sum(1).max()),'cache_is_real_training_cells':True};write_json(D/'model_config.json',config);(D/'probe_models').mkdir(exist_ok=True);torch.save(initial,D/'initial_state.pt');save()
 batches=[284,292];stop_reason='available_unique_input_limit';previous=None
 for proposed in batches:
  model.zero_grad(set_to_none=True);model.cur_gene_token_embs=None;gc.collect();torch.cuda.empty_cache();budget=meter.memory()['reserved_gib']+psutil.virtual_memory().available/2**30-8
  b=proposed
  if previous:
   projected=previous['memory']['peak_reserved_gib']*b/previous['batch_size']*1.02
   if projected>budget:
    cap=int(previous['batch_size']*budget/(previous['memory']['peak_reserved_gib']*1.02));cap=min(b,cap)
    if cap<=previous['batch_size']+max(1,previous['batch_size']//30):stop_reason='predicted_reservation_would_violate_8GiB_available_guard';S['stopped_before_batch']={'proposed':b,'projected_reserved_gib':projected,'reservation_budget_gib':budget,'safe_cap_estimate':cap};save();break
    b=cap;S['next_batch_capped_by_memory_prediction']={'proposed':proposed,'tested':b,'reservation_budget_gib':budget}
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);model.load_state_dict(initial);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8);torch.cuda.synchronize();t,base=meter.begin(f'train_batch{b}');steps=0;losses=[];print('START_BATCH',b,'availableGiB',base['system_available_gib'],flush=True);save()
  try:
   while steps<3 or time.time()-t<15:
    opt.zero_grad(set_to_none=True);g=torch.from_numpy(genes[:b].copy()).cuda();v=torch.from_numpy(values[:b].copy()).cuda();y=torch.from_numpy(labels[:b].copy()).cuda();h=model._encode(g,v,src_key_padding_mask=g.eq(pad));logits=model.cls_decoder(h[:,0,:]);loss=torch.nn.functional.cross_entropy(logits,y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();losses.append(float(loss.detach()));steps+=1;del g,v,y,h,logits,loss
   row=meter.end(t,base,name=f'train_batch{b}',batch_size=b,unique_inputs=b,steps=steps,processed=b*steps,loss_first=losses[0],loss_last=losses[-1],actual_input_shape=[b,1200],actual_mean_nonpad_tokens=float((genes[:b]!=pad).sum(1).mean()));row['throughput_cells_per_second']=row['processed']/row['seconds'];row['allocated_fraction_of_shared_RAM']=row['memory']['peak_allocated_gib']/row['memory']['system_total_gib'];row['reserved_fraction_of_shared_RAM']=row['memory']['peak_reserved_gib']/row['memory']['system_total_gib'];row['reached_80pct_allocated']=row['allocated_fraction_of_shared_RAM']>=.8;row['reached_90pct_allocated']=row['allocated_fraction_of_shared_RAM']>=.9;S['conditions'].append(row);print('RESULT',json.dumps(row),flush=True);save();meter.phase='save_probe';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},D/'probe_models'/f'batch_{b}.pt');previous=row
  except torch.cuda.OutOfMemoryError as e:
   S['oom']={'batch':b,'error':str(e)};stop_reason='CUDA_OOM';save();break
  finally:
   opt.zero_grad(set_to_none=True);del opt;model.zero_grad(set_to_none=True);model.cur_gene_token_embs=None;gc.collect();torch.cuda.empty_cache()
  if row['reached_90pct_allocated']:stop_reason='90pct_allocated_target_observed';break
  if b<proposed:stop_reason='tested_predicted_safe_cap';break
 S['status']='complete';S['stop_reason']=stop_reason;S['largest_successful_batch']=max(r['batch_size'] for r in S['conditions']);S['maximum_peak_allocated_gib']=max(r['memory']['peak_allocated_gib'] for r in S['conditions']);S['reached_80pct_allocated']=any(r['reached_80pct_allocated'] for r in S['conditions']);S['reached_90pct_allocated']=any(r['reached_90pct_allocated'] for r in S['conditions']);save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/memory_batch_sweep.py — 원문 전문 (7,381 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/memory_batch_sweep.py", "sha256": "8c2ad5b07edfd7a4290562891973db3ad1afdb4e65cf1f50f731a9896c4c9f80", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Actual batch capacity probe; fixed FP32/1200 cached gene tokens, fresh start per batch."""
from pathlib import Path
import os,time,json,gc,random,re
import numpy as np,torch,psutil
from scgpt.model import TransformerModel
from gpu_support import Meter,write_json
R=Path('/workspace/scgpt');D=R/'results'/os.environ['RUN_ID'];D.mkdir(exist_ok=True);source=R/'results'/(R/'results/SUPPLEMENT_LATEST').read_text().strip();T=source/'training_util';bundle=R/'models/finetuned'/source.name/'epoch_001';torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.manual_seed(42);np.random.seed(42);random.seed(42);meter=Meter(D);S={'status':'running','experiment':'followup_batch_memory_capacity','conditions':[],'source_basic_run':source.name,'settings':{'mode':'training_resource_probe','dtype':'FP32','tf32':False,'max_length_including_CLS':1200,'optimizer':'AdamW lr1e-4 wd.01 betas.9/.999 eps1e-8 clip1.0','initialization':'same pretrained encoder and seed42 fresh13-class classifier restored before every batch','min_seconds_per_condition':15,'min_steps':3,'target_allocated_fraction_of_shared_RAM':[.8,.9],'minimum_system_available_gib':8,'swap_increase_stop_gib':2,'batch_only_variable':True,'quality_training':'not an accuracy experiment; updated probe weights saved separately','batch_repetition':'same nested firstB real training cells across steps, no replicated cells inside a batch'}}
def save():write_json(D/'summary.json',S)
try:
 config=json.loads((bundle/'model_config.json').read_text());vocab=json.loads((bundle/'vocab.json').read_text());model=TransformerModel(**config,vocab=vocab);raw=torch.load(R/'models/whole-human/best_model.pt',map_location='cpu',weights_only=True);mapped={}
 for key,value in raw.items():
  for a,b in [(r'self_attn\._impl\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\.Wqkv\.','self_attn.in_proj_'),(r'self_attn\._impl\.out_proj\.','self_attn.out_proj.')]:key=re.sub(a,b,key)
  mapped[key]=value
 current=model.state_dict();core=('encoder.','value_encoder.','transformer_encoder.');required=[k for k in current if k.startswith(core)];loaded={k:mapped[k] for k in required if k in mapped and mapped[k].shape==current[k].shape};assert len(loaded)==len(required);model.load_state_dict(loaded,strict=False);del raw,mapped,current,loaded
 for name,p in model.named_parameters():p.requires_grad_(name.startswith(core+('cls_decoder.',)))
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.float().cuda();torch.cuda.synchronize();S['model_only_postload']=meter.memory();S['parameters']={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)};cache=np.load(T/'epoch_001/training_tokenized_inputs.npz');genes=cache['gene_ids'];values=cache['expression_bins'];labels=cache['labels'];assert genes.shape==(1000,1200) and values.shape==genes.shape and len(labels)==1000;pad=vocab[config['pad_token']];S['input_audit']={'unique_available_cells':1000,'gene_values_shape':list(genes.shape),'labels':{str(int(k)):int(n) for k,n in zip(*np.unique(labels,return_counts=True))},'source':str(T/'epoch_001/training_tokenized_inputs.npz'),'token_min':int((genes!=pad).sum(1).min()),'token_max':int((genes!=pad).sum(1).max()),'cache_is_real_training_cells':True};write_json(D/'model_config.json',config);(D/'probe_models').mkdir(exist_ok=True);torch.save(initial,D/'initial_state.pt');save()
 batches=[8,16,32,48,64,96,128,160,192,224,256,288,320,384,448,512,640,768,896,1000];stop_reason='available_unique_input_limit';previous=None
 for proposed in batches:
  model.zero_grad(set_to_none=True);model.cur_gene_token_embs=None;gc.collect();torch.cuda.empty_cache();budget=meter.memory()['reserved_gib']+psutil.virtual_memory().available/2**30-8
  b=proposed
  if previous:
   projected=previous['memory']['peak_reserved_gib']*b/previous['batch_size']*1.08
   if projected>budget:
    cap=int(previous['batch_size']*budget/(previous['memory']['peak_reserved_gib']*1.08));cap=min(b,cap)
    if cap<=previous['batch_size']+max(1,previous['batch_size']//30):stop_reason='predicted_reservation_would_violate_8GiB_available_guard';S['stopped_before_batch']={'proposed':b,'projected_reserved_gib':projected,'reservation_budget_gib':budget,'safe_cap_estimate':cap};save();break
    b=cap;S['next_batch_capped_by_memory_prediction']={'proposed':proposed,'tested':b,'reservation_budget_gib':budget}
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);model.load_state_dict(initial);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8);torch.cuda.synchronize();t,base=meter.begin(f'train_batch{b}');steps=0;losses=[];print('START_BATCH',b,'availableGiB',base['system_available_gib'],flush=True);save()
  try:
   while steps<3 or time.time()-t<15:
    opt.zero_grad(set_to_none=True);g=torch.from_numpy(genes[:b].copy()).cuda();v=torch.from_numpy(values[:b].copy()).cuda();y=torch.from_numpy(labels[:b].copy()).cuda();h=model._encode(g,v,src_key_padding_mask=g.eq(pad));logits=model.cls_decoder(h[:,0,:]);loss=torch.nn.functional.cross_entropy(logits,y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();losses.append(float(loss.detach()));steps+=1;del g,v,y,h,logits,loss
   row=meter.end(t,base,name=f'train_batch{b}',batch_size=b,unique_inputs=b,steps=steps,processed=b*steps,loss_first=losses[0],loss_last=losses[-1],actual_input_shape=[b,1200],actual_mean_nonpad_tokens=float((genes[:b]!=pad).sum(1).mean()));row['throughput_cells_per_second']=row['processed']/row['seconds'];row['allocated_fraction_of_shared_RAM']=row['memory']['peak_allocated_gib']/row['memory']['system_total_gib'];row['reserved_fraction_of_shared_RAM']=row['memory']['peak_reserved_gib']/row['memory']['system_total_gib'];row['reached_80pct_allocated']=row['allocated_fraction_of_shared_RAM']>=.8;row['reached_90pct_allocated']=row['allocated_fraction_of_shared_RAM']>=.9;S['conditions'].append(row);print('RESULT',json.dumps(row),flush=True);save();meter.phase='save_probe';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},D/'probe_models'/f'batch_{b}.pt');previous=row
  except torch.cuda.OutOfMemoryError as e:
   S['oom']={'batch':b,'error':str(e)};stop_reason='CUDA_OOM';save();break
  finally:
   opt.zero_grad(set_to_none=True);del opt;model.zero_grad(set_to_none=True);model.cur_gene_token_embs=None;gc.collect();torch.cuda.empty_cache()
  if row['reached_90pct_allocated']:stop_reason='90pct_allocated_target_observed';break
  if b<proposed:stop_reason='tested_predicted_safe_cap';break
 S['status']='complete';S['stop_reason']=stop_reason;S['largest_successful_batch']=max(r['batch_size'] for r in S['conditions']);S['maximum_peak_allocated_gib']=max(r['memory']['peak_allocated_gib'] for r in S['conditions']);S['reached_80pct_allocated']=any(r['reached_80pct_allocated'] for r in S['conditions']);S['reached_90pct_allocated']=any(r['reached_90pct_allocated'] for r in S['conditions']);save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/reload_supplement.py — 원문 전문 (1,848 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/reload_supplement.py", "sha256": "b732330bec4e8271c9c08f9ada25d3a531798873cdcf827439b26e59e76e5911", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,numpy as np,torch
from scgpt.model import TransformerModel
R=Path('/workspace/scgpt');D=R/'results'/os.environ['RUN_ID'];T=D/'training_util';s=json.loads((T/'summary.json').read_text());bundle=Path(s['best_checkpoint']['model_bundle']);torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;config=json.loads((bundle/'model_config.json').read_text());vocab=json.loads((bundle/'vocab.json').read_text());model=TransformerModel(**config,vocab=vocab).float().cuda().eval();model.load_state_dict(torch.load(bundle/'model.pt',map_location='cuda',weights_only=True));v=np.load(T/'validation_tokenized_inputs.npz');pred=[];pad=vocab[config['pad_token']]
with torch.inference_mode():
 for i in range(0,len(v['gene_ids']),8):
  g=v['gene_ids'][i:i+8];l=int((g!=pad).sum(1).max());genes=torch.from_numpy(g[:,:l].copy()).cuda();values=torch.from_numpy(v['expression_bins'][i:i+8,:l].copy()).cuda();h=model._encode(genes,values,src_key_padding_mask=genes.eq(pad));pred.append(model.cls_decoder(h[:,0,:]).cpu().numpy())
y=np.concatenate(pred);ref=np.load(T/'epoch_001/validation/logits_probabilities.npz')['logits'];r={'independent_process':True,'validation_cells':len(y),'max_abs_difference':float(np.abs(y-ref).max()),'predictions_match':bool(np.array_equal(y.argmax(1),ref.argmax(1)))};r['passed']=r['max_abs_difference']<=1e-5 and r['predictions_match'];(D/'reload.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);assert r['passed'];S=json.loads((D/'summary.json').read_text());S['training_resource_supplement']=s;S['training_loading_baseline']=json.loads((T/'model_loading_baseline.json').read_text());S['reload']=r;S['status']='complete';(D/'summary.json').write_text(json.dumps(S,indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/report_full_realdata.py — 원문 전문 (10,539 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/report_full_realdata.py", "sha256": "c53d84d82f5bd0e60d9e65b3f36670591d13250bc328b65311392871d9cb5a46", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Summarize both completed full-real-data cases and verify their actual inputs."""
from pathlib import Path
import csv,json,hashlib
import numpy as np,pandas as pd
R=Path('/workspace/scgpt');run=(R/'results/REALDATA_LATEST').read_text().strip();D=R/'results'/run;cases={};audit={};P=R/'reports';P.mkdir(exist_ok=True)
for batch in [8,284]:
 C=D/f'batch_{batch}';tr=json.loads((C/'train/summary.json').read_text());te=json.loads((C/'test/summary.json').read_text());assert tr['status']=='complete' and te['status']=='complete';assert tr['training_cells']==8480 and tr['validation_cells']==2120 and te['cells_total']==4218 and te['cells_evaluated_known_labels']==4213;assert tr['saved_model_reload_verified'];e=tr['epochs_detail'][0];assert e['strict_reload_passed'] and e['reload_max_abs_logit_error']<=1e-5
 for mode in ['train','test']:assert json.loads((C/f'{mode}_exit.json').read_text())['exit_code']==0
 shape=pd.read_csv(C/'train/epoch_001/training_steps.csv');resource=pd.read_csv(C/'train/epoch_001/training_resources.csv');steps=len(shape);padded=shape['padded_token_length'] if 'padded_token_length' in shape else shape['padded_token_length_including_cls'] if 'padded_token_length_including_cls' in shape else None
 cases[batch]={'train':tr,'test':te,'steps':steps,'padded_token_min':int(padded.min()) if padded is not None else None,'padded_token_mean_unweighted':float(padded.mean()) if padded is not None else None,'padded_token_max':int(padded.max()) if padded is not None else None,'available_sampled_min_gib':float(resource.system_available_bytes.min()/2**30),'swap_max_gib':float(resource.system_swap_used_bytes.max()/2**30),'train_total_wall_seconds':json.loads((C/'train_exit.json').read_text())['total_subprocess_wall_seconds'],'test_total_wall_seconds':json.loads((C/'test_exit.json').read_text())['total_subprocess_wall_seconds']}
# Exact comparisons use the full recorded per-cell arrays after dynamic batch padding.
for mode,relative in [('train','train/epoch_001/training_tokenized_inputs.npz'),('test','test/tokenized_inputs.npz')]:
 files=[D/f'batch_{b}'/relative for b in [8,284]];aa=np.load(files[0]);bb=np.load(files[1]);common=set(aa.files)&set(bb.files);checks={k:bool(np.array_equal(aa[k],bb[k])) for k in common};audit[mode]={'input_array_shapes':{k:list(aa[k].shape) for k in common},'exact_input_equality':checks};assert checks.get('gene_ids') and checks.get('expression_bins') and checks.get('labels'),audit[mode]
 if mode=='train':assert checks.get('original_row_indices')
 del aa,bb
# Same original train/validation cell assignments, same checkpoint/structure/labels.
for name in ['split_assignments.tsv','model_config.json','labels.json','genes.json']:
 a=D/'batch_8/train'/name;b=D/'batch_284/train'/name;audit[name+'_identical']=hashlib.sha256(a.read_bytes()).hexdigest()==hashlib.sha256(b.read_bytes()).hexdigest();assert audit[name+'_identical']
(D/'input_equality_verification.json').write_text(json.dumps(audit,indent=2));summary={'status':'complete','run_id':run,'protocol':json.loads((D/'protocol.json').read_text()),'source_data_audit':json.loads((D/'source_data_audit.json').read_text()),'cases':{str(b):v for b,v in cases.items()},'input_verification':audit,'interpretation':'real14818cells, original dynamic padding; same one epoch gives1060 vs30 optimizer updates; not a controlled convergence-quality ablation'};(D/'summary.json').write_text(json.dumps(summary,indent=2));out=[]
def a(s=''):out.append(s)
def table(h,rows):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rows:a('| '+' | '.join(map(str,r))+' |')
 a()
a('# scGPT: 만 단위 실제 전체 데이터에서 batch 8과284 비교');a();a(f'실행 `{run}` · [원본 summary](../results/{run}/summary.json) · [입력 일치 검증](../results/{run}/input_equality_verification.json)');a();a('## 실제 데이터 전체 구성');a();a('공개 TOSICA pancreas demo_train10,600세포와 별도 demo_test4,218세포, **총14,818개 실제 세포**를 사용했다. 두 파일의 cell ID는 각각 고유하며 파일 간 겹치는 ID도0개임을 다시 확인했다. 원본은 세포×3,000유전자, scGPT vocabulary와 일치하는2,999유전자를 사용했다. 이전1,000세포 capacity probe와 구분되는 전체 데이터 처리 실험이다.');a();table(['역할','세포 수','사용'],[['학습',8480,'demo_train의80%,13유형'],['검증',2120,'demo_train의20%,seed42 stratified 고정'],['독립test',4218,'학습과분리,학습라벨내4213세포로지표;test-only5세포별도보고']]);a('cell-type 라벨은 실제 metadata를 사용한다. 원본값의 raw-count/정규화 provenance는확인되지않아 원시count라고단정하지않는다. 제공된표현값에추가normalize/log1p를하지않고공식51-bin전처리를적용했다.');a();a('## 이번 비교에서 고정한 조건');a();a('같은pretrainedwhole-humanencoder+seed42classifier, FP32/TF32=False,최대1200token(CLS포함),동일gene집합·분할·학습세포순서·AdamW1e-4/wd.01/gradientclip1.0,1epoch를유지했다. batch만8→284로바꿨다. 모든세포를GPU에한번에올리지않고정상미니배치로처리했다.');a();a('**원래 DataCollator의 배치 내 실제 최대 길이에 맞춘 dynamic padding**을사용했다. 앞선용량probe처럼모든배치를강제로1200열로채우지않았다. 실제geneIDs·bin값·라벨·학습행순서의전체저장배열이두조건에서정확히같음을검사했고모두통과했다. test입력배열도동일했다. 주의: 저장배열은비교/보관을위해1200열까지pad되지만실제GPU입력shape는각training_steps.csv에기록된dynamic길이다.');a();rows=[];flat=[]
for b,c in cases.items():
 tr=c['train'];e=tr['epochs_detail'][0];te=c['test'];rows.append([b,tr['training_cells'],c['steps'],f"{tr['training_seconds_total']:.2f}",f"{e['training_cells_per_second']:.2f}",f"{e['gpu_peak_allocated_gib']:.3f}",f"{e['gpu_peak_reserved_gib']:.3f}",f"{e['gpu_peak_allocated_gib']/121.68751525878906*100:.2f}",f"{e['gpu_util_mean_percent']:.1f}/{e['gpu_util_max_percent']:.0f}",f"{c['available_sampled_min_gib']:.3f}"]);flat.append({'batch':b,'train_cells':tr['training_cells'],'optimizer_steps':c['steps'],'train_seconds':tr['training_seconds_total'],'train_cells_per_second':e['training_cells_per_second'],'train_peak_allocated_gib':e['gpu_peak_allocated_gib'],'train_peak_reserved_gib':e['gpu_peak_reserved_gib'],'gpu_util_avg':e['gpu_util_mean_percent'],'gpu_util_max':e['gpu_util_max_percent'],'available_sampled_min_gib':c['available_sampled_min_gib'],'train_subprocess_wall_seconds':c['train_total_wall_seconds'],'test_subprocess_wall_seconds':c['test_total_wall_seconds'],'test_cells':te['cells_total'],'test_forward_seconds':te['forward_seconds'],'test_accuracy':te['accuracy'],'test_macro_f1_13classes':te['macro_f1_all_training_classes'],'test_weighted_f1':te['weighted_f1'],'dynamic_token_min':c['padded_token_min'],'dynamic_token_mean_unweighted':c['padded_token_mean_unweighted'],'dynamic_token_max':c['padded_token_max']})
a('## 실제 전체 학습의 시간·GPU·메모리');a();table(['batch','고유학습세포','optimizer step','학습초','세포/초','allocated peak GiB','reserved peak GiB','공유RAM대비%','GPU평균/최대%','가용RAM sampled최소GiB'],rows);a('학습시간은원본script와같이tokenization·dynamic batch구성·전송·forward/loss/backward/gradientclip/optimizer를포함하고 validation·모델저장·독립test는제외한다. subprocess전체wall시간은CSV에따로기록해 imports·준비·validation·저장까지포함하는시간과구분했다. GPU비동기경계를동기화했다.');a();table(['batch','실제padded길이최소','평균(batch비가중)','최대'],[[b,c['padded_token_min'],c['padded_token_mean_unweighted'],c['padded_token_max']] for b,c in cases.items()]);a('큰batch는실제길이가긴세포하나가같은batch의padding길이를늘릴수있다. 전체세포수1만개와한번의GPUbatch크기는다른축이다. dataset개수는전체실행시간에,순간메모리는모델·정밀도·실제batch/길이에영향을준다.');a();a('## 저장 모델의 독립 test');a();table(['batch','test고유세포','알려진라벨평가수','forward초','Accuracy','macroF1(학습13유형)','weightedF1'],[[b,c['test']['cells_total'],c['test']['cells_evaluated_known_labels'],f"{c['test']['forward_seconds']:.2f}",f"{c['test']['accuracy']:.5f}",f"{c['test']['macro_f1_all_training_classes']:.5f}",f"{c['test']['weighted_f1']:.5f}"] for b,c in cases.items()]);a('각모델은별도경로에저장하고새프로세스로strict load해test4,218세포를평가했다. 학습직후저장전후validationlogits재현도검증했다. test-only MHC classII5세포는closed-set정확도에서제외했으며원본prediction파일에포함했다.');a();a('**1epoch가같아도optimizerupdate수는batch8의1060회와batch284의30회로다르다.** 학습률을batch에비례해조정하지않았고같은초기화/학습률을고정했다. test지표차이를batch크기만의순수인과효과나큰batch의최적도달성능으로해석하지않는다. donor독립/homology검증도미수행이다.');a();a('## 입력·저장·안전성');a();a('각case의train/split_assignments.tsv·split_class_counts.csv·run_config.json·preprocessing.json·source_snapshot과epoch_001/training_tokenized_inputs.npz·training_steps.csv·training_resources.csv를보존했다. 모델은models/finetuned/'+run+'/batch_B/epoch_001에있으며가중치·optimizer/RNG·라벨·vocab·실행코드·엄격재로드manifest를포함한다. 학습재개는검증하지않았다.');a();a('가용RAM8GiB미만또는swap2GiB증가시중단하도록했고GPU작업은한번에하나씩실행했다. 메모리비율은CUDAallocatedpeak/전체공유RAM121.688GiB참고값이며전용VRAM점유율이나시스템전체사용률과같지않다. RSS와CUDA는더하지않고reserved는cache/단편화포함예약으로구분한다. RSS/가용량은0.2초대기+조회시계열sampled값이다.');a();a('[앞선고정1200padding용량실험](scGPT_memory_batch.md) · [기본scGPT결과](scGPT.md) · [공식scGPT소스](https://github.com/bowang-lab/scGPT)');(P/'scGPT_realdata_full.md').write_text('\n'.join(out)+'\n')
with (P/'realdata_full_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(P/'scGPT_realdata_full.md');print(json.dumps({'input_equality':audit,'comparison':flat},indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/report_memory_sweep.py — 원문 전문 (7,745 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/report_memory_sweep.py", "sha256": "fbedd2d0f6fcb119f5b76624661f60b966f0d299e4a483f8f13f665b49d97836", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,csv
R=Path('/workspace/scgpt');base=(R/'results/MEMORY_LATEST').read_text().strip();names=[base]
if (R/'results/MEMORY_REFINE_LATEST').exists():names.append((R/'results/MEMORY_REFINE_LATEST').read_text().strip())
summaries=[json.loads((R/'results'/name/'summary.json').read_text()) for name in names if (R/'results'/name/'summary.json').exists()];rows=[{**r,'run':sname} for sname,s in zip(names,summaries) for r in s['conditions']];out=[]
def a(t=''):out.append(t)
def table(h,rs):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rs:a('| '+' | '.join(map(str,r))+' |')
 a()
if not rows:raise RuntimeError('no finished batches')
rows.sort(key=lambda r:r['batch_size']);best=max(rows,key=lambda r:r['memory']['peak_allocated_gib']);total=best['memory']['system_total_gib'];reached80=any(r['reached_80pct_allocated'] for r in rows);reached90=any(r['reached_90pct_allocated'] for r in rows)
a('# scGPT 후속: batch 확대와 실제 메모리 용량');a();a('기본10종 완료 이후 별도 승인으로 실행한 **학습 자원용량 실험**이다. FP32·최대1200token·동일모델·전처리·optimizer를 고정하고 batch를 확대했다. 모델정확도 개선 실험이 아니다.');a();a(f'관측 최대 **batch{best["batch_size"]}: CUDA allocated peak {best["memory"]["peak_allocated_gib"]:.3f}GiB / 공유RAM {total:.3f}GiB = {best["allocated_fraction_of_shared_RAM"]*100:.2f}%**. 80% 초과: {reached80}; 90% 도달: {reached90}.');a();a(f'80% 기준 {total*.8:.3f}GiB,90% 기준 {total*.9:.3f}GiB다. 공유메모리에서이비율은전용VRAM점유율이나시스템전체사용률과같지않다. 최대성공batch는시험범위중최대이며이론적한계가아니다.');a();table(['batch','고유세포','실제입력shape','step/반복','초','세포/초','allocatedpeakGiB','reservedpeakGiB','할당/공유RAM%','GPU평균/최대%','가용RAM sampled최소GiB'],[[r['batch_size'],r['unique_inputs'],str(r['actual_input_shape']),r['steps'],f"{r['seconds']:.2f}",f"{r['throughput_cells_per_second']:.2f}",f"{r['memory']['peak_allocated_gib']:.3f}",f"{r['memory']['peak_reserved_gib']:.3f}",f"{r['allocated_fraction_of_shared_RAM']*100:.2f}",f"{r['gpu_util_avg']:.1f}/{r['gpu_util_max']:.0f}",f"{r['available_sampled_min_gib']:.3f}" if r['available_sampled_min_gib'] is not None else 'N/A'] for r in rows]);first=rows[0];ratio=best['throughput_cells_per_second']/first['throughput_cells_per_second'];a(f'batch{first["batch_size"]} 대비 최대메모리 조건의처리량은약{ratio:.2f}배다. 메모리는 {first["memory"]["peak_allocated_gib"]:.2f}→{best["memory"]["peak_allocated_gib"]:.2f}GiB로늘었다. 중간batch에서이미처리량이비슷하면더큰batch의추가메모리효용은작다.');a();a('![batch별메모리와처리량](memory_batch_curve.png)');a();a('## 정확한 입력·학습·시간 범위');a();a('기존실제train1000세포의학습token화cache(gene IDs,expression bins,13개세포유형정답)를사용했다. 한batch내중복세포는없으며firstB중첩부분집합을같은조건에서여러step반복한다. 각condition은최소3step/15초이고실제step수를표에썼다. 작은batch는더많은측정step을반복하므로다른학습량의정확도비교로쓰지않는다.');a();a('각batch는동일pretrainedencoder·seed42새classifier·새AdamW에서시작한다. FP32/TF32=False,AdamWlr1e-4/wd.01/betas.9,.999/eps1e-8/gradientclip1.0을고정했다. 원본51bin·CLS전처리이며가중치이론량·실측적재량과실행중activation/gradient/optimizer증가를구분한다. 각run summary의parameters와model_only_postload에있다.');a();a('**이용량실험은모든batch를1200열까지padding한cache를그대로사용한다.** 원본일반학습은batch내최대실제길이에맞춘dynamicpadding을사용하므로평균시간을원본실험과직접비교하지않는다. 실제nonpad token평균은약300대이며,나머지padding도dense학습의메모리/계산비용을발생시킨다. 따라서90%에가까운할당이모두유효한유전자계산량이라는뜻은아니다. 일관된1200길이조건의capacity상한조사다.');a();a('시간에는cached batch구성·CPU→GPU·forward/classifier/loss·backward·gradientclip·optimizer와step경계CUDA동기화를포함한다. imports·모델적재·H5읽기·token화·probe가중치저장은제외했다. 각조건allocatorpeak는reset하고현재baseline도별도로기록했다. GPU-Util은활동시간비율로FLOPS/메모리%가아니다.');a();a('원본기본결과는보존했다. capacity probe후가중치는각run/probe_models/batch_B.pt에별도저장했으며모델초기화는initial_state.pt,구조는model_config.json이다. 이probe가중치를검증된생물학적최종모델로채택하지않았고,학습재개/정확도평가를주장하지않는다.');a();a('## 중단·측정 한계');a();a('시스템MemAvailable8GiB미만또는swap+2GiB에서자체프로세스를중단한다. 다음batch예상reserved가현재예약량+가용량−8GiB예산을넘으면확대전에중단하거나예상안전batch로축소한다. 첫coarse sweep은8%예측여유,80%경계refine은이전실측에대해2%여유를적용했다. 중단은90%불가능증명이나모델절대최대가아니다.');a()
for name,s in zip(names,summaries):a('- `'+name+'`: '+s['status']+' / '+str(s.get('stop_reason','running')))
a();a('reserved는cache/단편화포함이며실제살아있는tensor할당과다르다. nvidia-smi는SparkVRAM용량을제공하지않으므로PyTorchcounter와전체공유RAM·시스템가용량을구분했다. RSS와CUDA는더하지않는다. RSS/가용량은0.5초대기+조회시계열sampled값이며순간최저를보장하지않는다.');a();a('## 원본 경로');a()
for name in names:a(f'- [run {name}](../results/{name}/summary.json) · [로그](../results/{name}/run.log) · [시계열](../results/{name}/resources.csv)')
a();a('[기본 scGPT 보고서](scGPT.md) · [기존모델·공식소스](https://github.com/bowang-lab/scGPT)');(R/'reports/scGPT_memory_batch.md').write_text('\n'.join(out)+'\n');keys=['run','batch_size','unique_inputs','steps','seconds','throughput_cells_per_second','peak_allocated_gib','peak_reserved_gib','allocated_percent_shared','gpu_util_avg','gpu_util_max','available_sampled_min_gib'];flat=[]
for r in rows:flat.append({'run':r['run'],**{k:r[k] for k in ['batch_size','unique_inputs','steps','seconds','throughput_cells_per_second','gpu_util_avg','gpu_util_max','available_sampled_min_gib']},'peak_allocated_gib':r['memory']['peak_allocated_gib'],'peak_reserved_gib':r['memory']['peak_reserved_gib'],'allocated_percent_shared':r['allocated_fraction_of_shared_RAM']*100})
with (R/'reports/memory_batch_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(11,4));x=[r['batch_size'] for r in rows];axs[0].plot(x,[r['memory']['peak_allocated_gib'] for r in rows],'o-',label='Allocated peak');axs[0].plot(x,[r['memory']['peak_reserved_gib'] for r in rows],label='Reserved peak');axs[0].axhline(.8*total,ls='--',color='orange',label='80% shared RAM');axs[0].axhline(.9*total,ls='--',color='red',label='90% shared RAM');axs[0].set_ylabel('GiB');axs[0].legend(fontsize=8);axs[1].plot(x,[r['throughput_cells_per_second'] for r in rows],'o-');axs[1].set_ylabel('Cells / second')
for ax in axs:ax.set_xlabel('Batch size, fixed padded1200 tokens, FP32');ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(R/'reports/memory_batch_curve.png',dpi=160);plt.close(fig);print(R/'reports/scGPT_memory_batch.md')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/report_supplement.py — 원문 전문 (7,846 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/report_supplement.py", "sha256": "65b76ad7e9d5f118d2411e8a4f04d488a8835588462c8112540677a4e14f6ba9", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,csv
R=Path('/home/hyein/ai-benchmark/scgpt');run=(R/'results/SUPPLEMENT_LATEST').read_text().strip();D=R/'results'/run;s=json.loads((D/'summary.json').read_text());old=json.loads((R/'results/finetuned_test_20261023_034345_99523/summary.json').read_text());out=[]
def a(t=''):out.append(t)
def table(h,rows):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rows:a('| '+' | '.join(map(str,r))+' |')
 a()
def f(v):return 'N/A' if v is None else f'{v:.4f}'
a('# scGPT 기존 완료 결과와 입력 규모·GPU 사용률 보완');a();a(f'보완run `{run}` · 상태 **{s["status"]}** · [원본summary](../results/{run}/summary.json) · [자원시계열](../results/{run}/resources.csv)');a();a('## 기존 기본 실험은 이미 완료');a();a('2026-10-23에입력검사·사전학습임베딩추론·1,000세포파인튜닝·저장/재로드·독립test·최대길이256/512/1024/1200비교를완료했다. 원본파일과학습모델을보존했고이번보완값을과거GPU-Util실측처럼소급해붙이지않는다.');a();a('[기존test](../results/finetuned_test_20261023_034345_99523/summary.json) · [기존길이비교](../results/length_compare_20261023_040313_501423/comparison.csv)');a();a('기존저장모델test4218세포중학습라벨이있는4213세포의Accuracy '+f(old['accuracy'])+',학습13클래스macroF1 '+f(old['macro_f1_all_training_classes'])+',weightedF1 '+f(old['weighted_f1'])+'.test-onlyMHCclassII5세포는해당지표에서제외했다. 기존1epoch파인튜닝CUDApeak3.3741GiB,저장모델test추론peak0.3811GiB다.');a();a('## 동일 환경·모델·실제 입력');a();a('기존scgpt-spark컨테이너와whole-human pretrained checkpoint/args/vocab를그대로재사용했다. 이전컨테이너는sleep infinity상태로중지돼있어다시시작했으며기존실험을자동재실행하는명령은아니었다.');a();a('환경 '+json.dumps(s['environment']));a();a('파라미터 '+json.dumps(s['parameters'])+'.추론구성전체와실제로쓰는encoder를구분한다. 원본checkpointSHA-256/소스snapshot/누락weight검사는보완run의setup/provenance.json·weight_loading.json에있다. nativeTransformer(use_fast_transformer=False),FP32,TF32=False,batch8,seed42,max_length1200(CLS포함)를유지했다.');a();b=s['model_only_postload'];table(['첫forward전','allocatedGiB','reservedGiB','RSSGiB','시스템가용GiB'],[['추론모델적재직후',*[f(b[k]) for k in ['allocated_gib','reserved_gib','rss_gib','system_available_gib']]]]);a('FP32이론값은파라미터수×4byte이며실측과구분한다. 적재 '+f(s['model_loading_seconds'])+'초,입력준비 '+f(s['input_preparation_seconds'])+'초,한번수행한token화cache '+f(s['tokenization_cache_seconds'])+'초,warmup '+f(s['warmup_seconds'])+'초다.');a();a('실제입력은TOSICA저자배포pancreas demo_train10600×3000/demo_test4218×3000 h5ad.사용가능vocab유전자2999개,Celltype은임베딩입력에넣지않는metadata다. raw/layers및정규화provenance가없어원시count라고단정하지않고제공값그대로51bin으로변환했다.');a();a('입력감사 '+json.dumps(s['input_audit']));a();a('실제test세포ID·nonzero유전자수·원래행·Celltype은input_cells.tsv,첫세포원표현은raw_first_cell_expression.npy,실제geneIDs/binned값은tokenized_all.npz다. source/setup/selected_cells.tsv·matched_genes.tsv도저장했다. 모델입력은B×최대token수geneID(int64)와binvalue(float32),출력은L2정규화된CLS B×512embedding이다.');a();a('## 개수·길이별 GPU 사용률과 메모리');a();rows=[];flat=[]
for r in s['conditions']:
 rows.append([r['name'],r['unique_inputs'],r['length'],f(r['tokens_mean']),r['repeats'],f(r['seconds']),f(r['throughput']),f(r['gpu_util_avg'])+'/'+f(r['gpu_util_max']),f(r['baseline']['allocated_gib']),f(r['memory']['peak_allocated_gib']),f(r['peak_delta_gib'])]);d={k:v for k,v in r.items() if k not in ['baseline','memory']};d.update(allocated_baseline_gib=r['baseline']['allocated_gib'],peak_allocated_gib=r['memory']['peak_allocated_gib'],peak_reserved_gib=r['memory']['peak_reserved_gib']);flat.append(d)
table(['조건','고유세포','cap','평균실제token','반복','초','세포/초','GPU평균/최대%','시작GiB','peakGiB','증가GiB'],rows)
with (R/'reports/comparison_supplement.csv').open('w',newline='') as f0:w=csv.DictWriter(f0,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
a('개수는같은cached1200cap의중첩128/1024/4218세포다. 세포별실제nonzero유전자수분포가달라subset별평균token도달라지며이를표에기록했다. 길이는동일128세포의한번샘플링한1200tokencache앞256/512/1024개를사용해중첩유전자집합을만들었다. 과거길이실험의cap별독립randomsampling과정책이다르므로속도향상/정확도변화로직접비교하지않는다.');a();a('각추론조건최소15초반복,고유입력과반복처리를구분했다. 시간은cached배치구성·CPU→GPU·encoder·CLS정규화·GPU→CPU반환을포함하며원본h5ad읽기·token화·모델적재·warmup·저장은제외했다. 각조건CUDApeak reset/empty_cache/synchronize,같은run실제시작allocated대비delta다. GPU-Util은활동시간비율로FLOPS나메모리%가아니다.0.5초대기+조회가조건안에포함된표본만집계했다.');a();a('## 학습 GPU 사용률 보완');a()
if 'training_resource_supplement' in s:
 t=s['training_resource_supplement'];e=t['epochs_detail'][0];b=s['training_loading_baseline'];table(['조건','train세포','epochs','batch','초','GPU평균/최대%','모델적재GiB','학습peakGiB'],[['누락사용률계측보완',t['training_cells'],t['epochs'],t['batch_size'],f(t['training_seconds_total']),f(e['gpu_util_mean_percent'])+'/'+f(e['gpu_util_max_percent']),f(b['allocated_gib']),f(e['gpu_peak_allocated_gib'])]]);a('GPU사용률이없었던기존학습조건을계측하기위해동일1000세포/1epoch를별도run에서실행했다. 원본모델/결과는유지했고새모델은별도경로에저장했다. 원본02_finetune.py복사본에GPU-Util시계열과적재baseline만추가해학습알고리즘·split·optimizer를유지했다.');a();a('전체train10600에서동일seed42cell단위validation2120,별도train1000을골랐으며13유형정답으로encoder+classifier를갱신했다. 저장모델bundle '+str(t['best_checkpoint']['model_bundle'])+'.새보완모델은validation과재로드를검증했고독립test를새로평가한것은아니다. 위기존testAccuracy와새모델을혼동하지않는다.');a();a('학습sampler는0.2초대기+조회시간과시작/끝표본을포함하므로첫/끝nvidia-smi관측창이주변활동을포함할수있다. resource CSV는training_util/epoch_001/training_resources.csv다. GPU메모리는fit구간allocatorpeak,validation/저장은시간에서제외했다. optimizer/RNG/epoch저장,encoder/head변경,원본strict reload검증을수행했다. 독립프로세스validation재로드 '+json.dumps(s.get('reload',{})));a()
else:a('학습사용률보완실행중.');a()
a('## 해석과 한계');a();a('Spark는공유RAM121.688GiB이므로RSS와CUDA를더하지않는다. allocated는사용tensor, reserved는cache포함allocator예약이며장치전체점유율과다르다. rawcount유래·donor독립검증은확인되지않았다. 개수증가가peak를비례증가시킨다고가정하지않으며batch8이고정돼메모리증가는작았다. 메모리90%도달은별도승인된후속batch확대실험에서검토한다.');a();a('공식소스: [scGPT](https://github.com/bowang-lab/scGPT). 정확한설치리비전·원본모델hash와코드는각기존/보완run의provenance와environment에보존했다.');(R/'reports/scGPT.md').write_text('\n'.join(out)+'\n');print(R/'reports/scGPT.md')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/resume_full_realdata.py — 원문 전문 (1,795 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/resume_full_realdata.py", "sha256": "50356d8db966815221816ce92d63434e30c341d8b8c3dc89298a278eba5a544a", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import subprocess,json,time,shlex
R=Path('/home/hyein/ai-benchmark/scgpt');run=(R/'results/REALDATA_LATEST').read_text().strip();D=R/'results'/run
for batch in [256,224]:
 success=True
 for mode in ['train','test']:
  C=D/f'batch_{batch}';C.mkdir(exist_ok=True);done=C/f'{mode}_exit.json'
  if done.exists() and json.loads(done.read_text())['exit_code']==0:continue
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  out=f'/workspace/scgpt/results/{run}/batch_{batch}/'+mode;bundle=f'/workspace/scgpt/models/finetuned/{run}/batch_{batch}';args=['--mode','train','--n-cells','0','--epochs','1','--model-out',bundle] if mode=='train' else ['--mode','predict','--bundle',bundle+'/epoch_001','--input','/workspace/scgpt/data/pancreas_demo/demo_test.h5ad'];cmd=['docker','exec','--user','1000:1000','-e','SCGPT_GUARD_DIR='+f'/workspace/scgpt/results/{run}/batch_{batch}','-e','MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba','scgpt-spark','python','-u','/workspace/scgpt/scripts/full_realdata.py',*args,'--out',out,'--batch-size',str(batch),'--max-length','1200','--seed','42'];(C/f'{mode}_command.json').write_text(json.dumps(cmd,indent=2));start=time.time();print('START',batch,mode,flush=True)
  with (C/f'{mode}.log').open('w') as f:code=subprocess.call(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  done.write_text(json.dumps({'exit_code':code,'total_subprocess_wall_seconds':time.time()-start}));print('EXIT',batch,mode,code,flush=True)
  if code:success=False;break
 if success:(D/'comparison_cases.json').write_text(json.dumps([8,batch]));print('COMPLETED_CASES',8,batch,flush=True);break
 time.sleep(5)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/supplement.py — 원문 전문 (3,922 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/supplement.py", "sha256": "b903ef892ac0ccb8d759b02e72225adc41f41229974937b36c1728faee6a1b8c", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,json,os,time,gc
from gpu_support import Meter,write_json
import numpy as np,torch
R=Path('/workspace/scgpt');D=R/'results'/os.environ['RUN_ID'];meter=Meter(D);S={'status':'running','conditions':[],'settings':{'batch':8,'max_length':1200,'dtype':'FP32','tf32':False,'seed':42,'inference':'same native pretrained whole-human encoder; CPU cache tokenization once, normalized CLS embeddings','input_count':'nested first128/1024/all4218 test cells, same cached genes/values','length':'same128cells, first256/512/1024/1200 tokens of once-sampled1200 token cache; nested controlled supplement, differs from original independent random sampling by length'}}
def save():write_json(D/'summary.json',S)
try:
 source=R/'scripts/01_inference.py';prefix=source.read_text().split('# Warm the CUDA execution path, then reset random seeds before the measured run.')[0];sys.argv=[str(source),'--split','test','--n-cells','0','--batch-size','8','--max-length','1200','--out',str(D/'setup')];meter.phase='setup';ns={'__file__':str(source),'__name__':'__scgpt_setup__'};exec(compile(prefix,str(source),'exec'),ns);model=ns['model'];S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'scgpt':__import__('scgpt').__version__};S['parameters']={'total':ns['parameter_count_total'],'encoder':ns['parameter_count_encoder']};S['model_only_postload']=meter.memory();S['model_loading_seconds']=ns['model_loading_seconds'];S['input_preparation_seconds']=ns['input_preparation_seconds'];save()
 meter.phase='tokenize_cache';t=time.time();n=ns['n'];pad=ns['pad_id'];genes=np.full((n,1200),pad,np.int64);values=np.full((n,1200),ns['pad_value'],np.float32);ns['seed_all'](42)
 for i in range(0,n,8):
  b=ns['make_batch'](i,min(i+8,n));l=b['gene'].shape[1];genes[i:i+len(b['gene']),:l]=b['gene'].numpy();values[i:i+len(b['gene']),:l]=b['expr'].numpy()
 S['tokenization_cache_seconds']=time.time()-t;np.savez_compressed(D/'tokenized_all.npz',gene_ids=genes,expression_bins=values);ns['cell_metadata'].to_csv(D/'input_cells.tsv',sep='\t');np.save(D/'raw_first_cell_expression.npy',ns['matrix'][0]);S['input_audit']={'cells':n,'source_genes':3000,'matched_genes':2999,'matrix_dtype':'float32','tokens_min':int((genes!=pad).sum(1).min()),'tokens_mean':float((genes!=pad).sum(1).mean()),'tokens_max':int((genes!=pad).sum(1).max()),'raw_count_provenance':'not verified, values used as supplied, official51-bin preprocessing','labels':'Celltype metadata not provided to embedding encoder'};save()
 def batch(i,end,length):return {'gene':torch.from_numpy(genes[i:end,:length].copy()),'expr':torch.from_numpy(values[i:end,:length].copy())}
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():z=ns['encode_batch'](batch(0,8,1200))
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;np.save(D/'example_embeddings.npy',z);model.cur_gene_token_embs=None
 for nsel,length in [(128,1200),(1024,1200),(4218,1200),(128,256),(128,512),(128,1024)]:
  model.cur_gene_token_embs=None;gc.collect();t,base=meter.begin(f'inference_n{nsel}_cap{length}');repeats=0
  with torch.inference_mode():
   while repeats==0 or time.time()-t<15:
    for i in range(0,nsel,8):z=ns['encode_batch'](batch(i,min(i+8,nsel),length))
    torch.cuda.synchronize();repeats+=1
  counts=(genes[:nsel,:length]!=pad).sum(1);r=meter.end(t,base,name=f'inference_n{nsel}_cap{length}',mode='inference',length=length,unique_inputs=nsel,repeats=repeats,processed=nsel*repeats,tokens_min=int(counts.min()),tokens_mean=float(counts.mean()),tokens_max=int(counts.max()));r['throughput']=r['processed']/r['seconds'];S['conditions'].append(r);print('CONDITION',r['name'],r['seconds'],r['gpu_util_avg'],r['memory']['peak_allocated_gib'],flush=True);save()
 S['status']='inference_complete';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/scripts/train_util_supplement.py — 원문 전문 (38,533 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/scripts/train_util_supplement.py", "sha256": "d2019b4af252c4e04b4da9afc9606a036f76ac9158bf48357880070f91b7e3cc", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""scGPT pancreas classification: supervised fine-tuning and saved-bundle prediction.

Run train through scgpt_finetune.sh. Reload example (inside the same container):
python predict.py --mode predict --bundle /path/to/epoch_001 \
    --input /path/to/demo_test.h5ad --out /path/to/new_prediction_results
No package installation or test-set training is performed.
"""
import argparse
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.preprocess import binning
from scgpt.tokenizer import GeneVocab

ROOT = Path('/workspace/scgpt')
GIB = 1024 ** 3
CORE = ('encoder.', 'value_encoder.', 'transformer_encoder.')
TRAINABLE = CORE + ('cls_decoder.',)
MEMORY_NOTES = ('CUDA peaks are process-local PyTorch allocator counters, not total device usage. '
                'RSS is sampled every 0.2 s, not an exact peak. Spark uses shared memory: '
                'do not add RSS and CUDA as independent physical memory pools.')


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResourceSampler:
    def __init__(self):
        self.process = psutil.Process()
        self.rows = []
        self.event = threading.Event()
        self.lock = threading.Lock()

    def sample(self):
        with self.lock:
            self.rows.append({
                'elapsed_seconds': time.perf_counter() - self.start_time,
                'gpu_util_percent': float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]),
                'process_rss_bytes': self.process.memory_info().rss,
                'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'cuda_reserved_bytes': torch.cuda.memory_reserved(),
                'system_available_bytes': psutil.virtual_memory().available,
            })

    def loop(self):
        while not self.event.wait(0.2):
            self.sample()

    def start(self):
        self.start_time = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()
        self.sample()

    def metrics(self):
        return {
            'gpu_util_mean_percent': float(np.mean([r['gpu_util_percent'] for r in self.rows])),
            'gpu_util_max_percent': max(r['gpu_util_percent'] for r in self.rows),
            'gpu_util_samples': len(self.rows),
            'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated() / GIB,
            'gpu_peak_reserved_gib': torch.cuda.max_memory_reserved() / GIB,
            'python_process_rss_peak_observed_gib': max(r['process_rss_bytes'] for r in self.rows) / GIB,
        }


def preserve_environment(out):
    source = out / 'source_snapshot'
    source.mkdir(exist_ok=True)
    for name, obj in [('model.py', TransformerModel), ('data_collator.py', DataCollator),
                      ('preprocess.py', binning), ('gene_tokenizer.py', GeneVocab)]:
        shutil.copy2(inspect.getfile(obj), source / name)
    packages = {p: metadata.version(p) for p in
                ['torch', 'scgpt', 'numpy', 'anndata', 'scanpy', 'scipy', 'pandas', 'scikit-learn', 'psutil']}
    direct_url = metadata.distribution('scgpt').read_text('direct_url.json')
    device = torch.cuda.get_device_properties(0)
    write_json(out / 'environment.json', {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'architecture': platform.machine(), 'packages': packages,
        'scgpt_install_source': json.loads(direct_url) if direct_url else None,
        'cuda_version': torch.version.cuda, 'gpu_name': device.name,
        'gpu_compute_capability': [device.major, device.minor],
        'cuda_reported_total_memory_gib': device.total_memory / GIB,
        'system_total_memory_gib': psutil.virtual_memory().total / GIB,
        'torch_cpu_threads': torch.get_num_threads(),
    })
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (out / 'requirements_freeze.txt').write_text(freeze.stdout)


def choose_training_subset(pool, y, requested, seed):
    """Proportional subset with at least one cell from every training class."""
    if requested == 0 or requested == len(pool):
        return np.sort(pool)
    classes, counts = np.unique(y[pool], return_counts=True)
    if requested < len(classes) or requested > len(pool):
        raise ValueError(f'n-cells must be 0 (all) or {len(classes)}..{len(pool)}')
    ideal = counts * (requested / len(pool))
    quota = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while quota.sum() < requested:
        available = np.flatnonzero(quota < counts)
        k = available[np.argmax((ideal - quota)[available])]
        quota[k] += 1
    while quota.sum() > requested:
        available = np.flatnonzero(quota > 1)
        k = available[np.argmax((quota - ideal)[available])]
        quota[k] -= 1
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(pool[y[pool] == c], int(q), replace=False)
                               for c, q in zip(classes, quota)])
    return np.sort(selected)


def extract_matrix(adata, genes):
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Cell IDs and gene names must be unique')
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise ValueError(f'Missing {len(missing)} training genes, examples: {sorted(missing)[:5]}')
    selected = adata[:, genes]
    matrix = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
    matrix = matrix.astype(np.float32, copy=True)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError('Expression values must be finite and nonnegative')
    if np.any(np.count_nonzero(matrix, axis=1) == 0):
        raise ValueError('A cell has no nonzero expression after vocabulary filtering')
    return matrix


class Tokenizer:
    def __init__(self, matrix, gene_names, vocab, config):
        self.matrix = matrix
        self.gene_ids = np.asarray(vocab(gene_names), dtype=np.int64)
        self.pad_id = vocab[config['pad_token']]
        self.cls_id = vocab['<cls>']
        self.pad_value = config['pad_value']
        self.max_length = config['max_length_including_cls']
        self.collator = DataCollator(do_padding=True, pad_token_id=self.pad_id,
            pad_value=self.pad_value, do_mlm=False, do_binning=True,
            max_length=self.max_length, sampling=True, keep_first_n_tokens=1)

    def batch(self, rows):
        examples = []
        for i in rows:
            nz = np.flatnonzero(self.matrix[i])
            genes = np.concatenate([np.array([self.cls_id], dtype=np.int64), self.gene_ids[nz]])
            values = np.concatenate([np.array([self.pad_value], dtype=np.float32), self.matrix[i, nz]])
            examples.append({'genes': torch.from_numpy(genes), 'expressions': torch.from_numpy(values)})
        return self.collator(examples)

    def empty_recording(self, n):
        return (np.full((n, self.max_length), self.pad_id, dtype=np.int64),
                np.full((n, self.max_length), self.pad_value, dtype=np.float32))

    def cache(self, rows, batch_size):
        genes, values = self.empty_recording(len(rows))
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch = self.batch(rows[start:end])
            length = batch['gene'].shape[1]
            genes[start:end, :length] = batch['gene'].numpy()
            values[start:end, :length] = batch['expr'].numpy()
        return genes, values


def cached_batch(cache, start, end, pad_id):
    genes, values = cache
    length = int(np.count_nonzero(genes[start:end] != pad_id, axis=1).max())
    return {'gene': torch.from_numpy(genes[start:end, :length]),
            'expr': torch.from_numpy(values[start:end, :length])}


def classify(model, batch, pad_id):
    genes = batch['gene'].to('cuda')
    values = batch['expr'].to('cuda')
    encoded = model._encode(genes, values, src_key_padding_mask=genes.eq(pad_id))
    # Official classification path uses raw CLS, not L2-normalized embeddings.
    return model.cls_decoder(encoded[:, 0, :])


def load_pretrained_backbone(model, checkpoint, out):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError('Expected a plain tensor state_dict')
    converted = {}
    for key, value in state.items():
        for pattern, replacement in [
            (r'self_attn\._impl\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\.Wqkv\.', 'self_attn.in_proj_'),
            (r'self_attn\._impl\.out_proj\.', 'self_attn.out_proj.'),
        ]:
            key = re.sub(pattern, replacement, key)
        if key in converted:
            raise ValueError(f'Duplicate renamed checkpoint key: {key}')
        converted[key] = value
    current = model.state_dict()
    required = [k for k in current if k.startswith(CORE)]
    loaded = {k: converted[k] for k in required
              if k in converted and converted[k].shape == current[k].shape}
    missing = sorted(set(required) - set(loaded))
    write_json(out / 'weight_loading.json', {
        'required_backbone_keys': len(required), 'loaded_backbone_keys': len(loaded),
        'missing_backbone_keys': missing, 'loaded_keys': sorted(loaded),
        'classifier': 'new random initialization; pretrained classifier not used',
        'unused_checkpoint_keys': sorted(set(converted) - set(loaded)),
    })
    if missing:
        raise RuntimeError(f'Incomplete pretrained backbone: {missing[:10]}')
    model.load_state_dict(loaded, strict=False)


def evaluate(model, cache, y, cell_ids, labels, pad_id, batch_size, out):
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    logits_all = np.empty((len(y), len(labels)), dtype=np.float32)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sampler = ResourceSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(y), batch_size):
                end = min(start + batch_size, len(y))
                batch = cached_batch(cache, start, end, pad_id)
                logits_all[start:end] = classify(model, batch, pad_id).cpu().numpy()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
    finally:
        sampler.stop()
        pd.DataFrame(sampler.rows).to_csv(out / 'resources.csv', index=False)
    if not np.isfinite(logits_all).all():
        raise FloatingPointError('Non-finite prediction logits')
    probabilities = torch.from_numpy(logits_all).softmax(dim=1).numpy()
    predicted = logits_all.argmax(axis=1)
    known = y >= 0
    metrics = {
        'cells_total': len(y), 'cells_evaluated_known_labels': int(known.sum()),
        'cells_without_known_training_label': int((~known).sum()),
        'forward_seconds': seconds, 'cells_per_second': len(y) / seconds,
        'timing_scope': 'cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving',
        **sampler.metrics(),
    }
    if known.any():
        true, pred = y[known], predicted[known]
        metrics.update({
            'loss': float(torch.nn.functional.cross_entropy(torch.from_numpy(logits_all[known]),
                                                           torch.from_numpy(true)).item()),
            'accuracy': float(accuracy_score(true, pred)),
            'macro_f1_all_training_classes': float(f1_score(true, pred, labels=np.arange(len(labels)), average='macro', zero_division=0)),
            'macro_f1_present_true_classes': float(f1_score(true, pred, labels=np.unique(true), average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(true, pred, average='weighted', zero_division=0)),
        })
        write_json(out / 'classification_report.json', classification_report(
            true, pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0))
        pd.DataFrame(confusion_matrix(true, pred, labels=np.arange(len(labels))),
                     index=labels, columns=labels).rename_axis('true_label').to_csv(out / 'confusion_matrix.csv')
    pd.DataFrame({
        'cell_id': cell_ids, 'true_label_id': y,
        'true_label': [labels[i] if i >= 0 else '<unknown_or_unlabeled>' for i in y],
        'predicted_label_id': predicted, 'predicted_label': [labels[i] for i in predicted],
        'max_probability': probabilities.max(axis=1), 'included_in_metrics': known,
    }).to_csv(out / 'predictions.tsv', sep='\t', index=False)
    np.savez_compressed(out / 'logits_probabilities.npz', logits=logits_all, probabilities=probabilities)
    write_json(out / 'metrics.json', metrics)
    model.cur_gene_token_embs = None
    return metrics, logits_all


def rng_state():
    np_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {'algorithm': np_state[0], 'keys': np_state[1].tolist(),
                  'position': int(np_state[2]), 'has_gauss': int(np_state[3]), 'cached_gaussian': float(np_state[4])},
        'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
    }


def train(opt):
    out = opt.out
    pretrained = ROOT / 'models/whole-human'
    args = json.loads((pretrained / 'args.json').read_text())
    if (args.get('n_bins') != 51 or args.get('input_style') != 'binned'
            or args.get('input_emb_style', 'continuous') != 'continuous' or args.get('pre_norm', False)):
        raise ValueError('Requires the inspected 51-bin continuous-value post-norm checkpoint')
    vocab = GeneVocab.from_file(pretrained / 'vocab.json')
    for token in [args['pad_token'], '<cls>', '<eoc>']:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[args['pad_token']])
    input_path = ROOT / 'data/pancreas_demo/demo_train.h5ad'
    t0 = time.perf_counter()
    data = ad.read_h5ad(input_path)
    if data.shape != (10600, 3000) or 'Celltype' not in data.obs or data.obs['Celltype'].isna().any():
        raise ValueError('Input differs from the inspected 10600 x 3000 labeled training file')
    label_values = data.obs['Celltype'].astype(str).to_numpy()
    labels = sorted(set(label_values))
    if len(labels) != 13:
        raise ValueError(f'Expected 13 training classes; found {len(labels)}')
    label_to_id = {name: i for i, name in enumerate(labels)}
    y = np.asarray([label_to_id[x] for x in label_values], dtype=np.int64)
    all_rows = np.arange(data.n_obs)
    pool, validation = train_test_split(all_rows, test_size=opt.validation_fraction,
                                        random_state=opt.seed, stratify=y)
    validation = np.sort(validation)
    training = choose_training_subset(pool, y, opt.n_cells, opt.seed)
    if len(np.intersect1d(training, validation)):
        raise RuntimeError('Training and validation overlap')
    if set(y[training]) != set(range(len(labels))) or set(y[validation]) != set(range(len(labels))):
        raise ValueError('Every class must be represented in training and validation')
    genes = [str(g) for g in data.var_names if g in vocab]
    if len(genes) != 2999:
        raise ValueError(f'Expected 2999 vocabulary-matched genes, found {len(genes)}')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    roles = np.full(len(y), 'train_pool_unused', dtype=object)
    roles[validation] = 'validation'
    roles[training] = 'train_used'
    pd.DataFrame({'original_row_index': all_rows, 'cell_id': cell_ids,
                  'Celltype': label_values, 'label_id': y, 'role': roles}).to_csv(
                      out / 'split_assignments.tsv', sep='\t', index=False)
    pd.crosstab(pd.Series(label_values, name='Celltype'), pd.Series(roles, name='role')).to_csv(
        out / 'split_class_counts.csv')
    input_seconds = time.perf_counter() - t0
    del data
    preprocess = {
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'], 'n_bins': 51,
        'max_length_including_cls': opt.max_length,
        'additional_normalize_total': False, 'additional_log1p': False,
        'expression_provenance': 'Supplied h5ad values; no raw layer or preprocessing metadata',
        'tokenization': 'nonzero genes only; prepend CLS; official per-cell quantile binning; random overlength sampling; dynamic padding',
        'pooling': 'raw CLS features into official cls_decoder; no L2 normalization',
        'prediction_genes': 'require every saved training gene; reorder to training gene order; ignore additional genes',
    }
    constructor = {
        'ntoken': len(vocab), 'd_model': args['embsize'], 'nhead': args['nheads'],
        'd_hid': args['d_hid'], 'nlayers': args['nlayers'],
        'nlayers_cls': args['n_layers_cls'], 'n_cls': len(labels), 'dropout': args['dropout'],
        'pad_token': args['pad_token'], 'pad_value': args['pad_value'],
        'do_mvc': False, 'do_dab': False, 'use_batch_labels': False,
        'domain_spec_batchnorm': False, 'input_emb_style': 'continuous', 'cell_emb_style': 'cls',
        'explicit_zero_prob': False, 'use_fast_transformer': False, 'pre_norm': False,
    }
    config = {
        'task': 'supervised cell-type classification', 'seed': opt.seed,
        'train_source_cells': len(y), 'training_pool_cells': len(pool), 'training_cells': len(training),
        'validation_cells': len(validation), 'classes': len(labels), 'epochs': opt.epochs,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'autocast': False, 'tf32': False,
        'optimizer': {'name': 'AdamW', 'lr': opt.lr, 'weight_decay': opt.weight_decay,
                      'betas': [0.9, 0.999], 'eps': 1e-8},
        'scheduler': None, 'gradient_clip_max_norm': 1.0,
        'loss': 'unweighted cross entropy', 'trainable_prefixes': list(TRAINABLE),
        'unused_expression_decoder': 'frozen and bypassed',
        'classifier': 'new official ClsDecoder; all classifier layers trainable',
        'split': 'stratified by Celltype before subsampling; no donor/batch metadata available',
        'validation_limit': 'within-file cell split; not donor-independent validation',
        'validation_tokens': 'generated once with seed+10000, saved and reused',
        'epoch_seed': 'seed+epoch; reinitializes RNG before each epoch',
        'checkpoint_selection': 'highest validation macro-F1 over all 13 training classes',
        'test_used_for_training_or_selection': False,
        'training_warmup_steps': 0,
        'training_timing_scope': 'tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization',
        'training_timer_excludes': 'imports, h5ad/model loading, validation token cache, validation, disk saving, reload verification',
        'memory_notes': MEMORY_NOTES, 'preprocessing': preprocess,
        'reproducibility': 'Seeds, exact tokens and environment saved; bitwise reproducibility across platforms is not guaranteed',
    }
    for filename, obj in [('run_config.json', config), ('preprocessing.json', preprocess),
                          ('model_config.json', constructor), ('labels.json', {'labels': labels, 'label_to_id': label_to_id}),
                          ('genes.json', genes), ('vocab.json', vocab.get_stoi())]:
        write_json(out / filename, obj)
    shutil.copy2(pretrained / 'args.json', out / 'pretrained_args.json')
    write_json(out / 'provenance.json', {
        'input': {'path': str(input_path), 'sha256': sha256(input_path)},
        'pretrained_checkpoint': {'path': str(pretrained / 'best_model.pt'), 'sha256': sha256(pretrained / 'best_model.pt')},
        'script_sha256': sha256(__file__),
        'tutorial': 'https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html',
        'dataset_source': 'TOSICA author-distributed Figshare demo_train.h5ad (file 39009503)',
    })
    print(f'[Split] train={len(training)}, validation={len(validation)}, classes={len(labels)}, matched_genes={len(genes)}', flush=True)
    print('[Input] expression values as supplied -> 51 bins; no additional normalization/log1p', flush=True)
    t0 = time.perf_counter()
    seed_all(opt.seed)
    model = TransformerModel(**constructor, vocab=vocab)
    load_pretrained_backbone(model, pretrained / 'best_model.pt', out)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE))
    model.float().to('cuda')
    torch.cuda.synchronize()
    write_json(out/'model_loading_baseline.json', {'allocated_gib':torch.cuda.memory_allocated()/GIB,'reserved_gib':torch.cuda.memory_reserved()/GIB,'rss_gib':psutil.Process().memory_info().rss/GIB,'system_available_gib':psutil.virtual_memory().available/GIB})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=opt.lr, weight_decay=opt.weight_decay,
                                 betas=(0.9, 0.999), eps=1e-8)
    torch.cuda.synchronize()
    model_loading_seconds = time.perf_counter() - t0
    probes = ['transformer_encoder.layers.0.self_attn.in_proj_weight', 'cls_decoder.out_layer.weight']
    initial = {k: model.state_dict()[k].detach().cpu().clone() for k in probes}
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed + 10000)
    t0 = time.perf_counter()
    val_cache = tokenizer.cache(validation, opt.batch_size)
    cache_seconds = time.perf_counter() - t0
    np.savez_compressed(out / 'validation_tokenized_inputs.npz', gene_ids=val_cache[0],
                        expression_bins=val_cache[1], original_row_indices=validation, labels=y[validation])
    print('[Validation] fixed input tokens saved', flush=True)
    opt.model_out.mkdir(parents=True, exist_ok=False)
    epoch_summaries = []
    best = None
    for epoch in range(1, opt.epochs + 1):
        epoch_out = out / f'epoch_{epoch:03d}'
        epoch_out.mkdir()
        seed_all(opt.seed + epoch)
        order = np.random.default_rng(opt.seed + epoch).permutation(training)
        saved_genes, saved_values = tokenizer.empty_recording(len(order))
        steps = []
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler = ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        loss_sum, correct = 0.0, 0
        try:
            for step, start in enumerate(range(0, len(order), opt.batch_size), 1):
                end = min(start + opt.batch_size, len(order))
                rows = order[start:end]
                batch = tokenizer.batch(rows)
                length = batch['gene'].shape[1]
                saved_genes[start:end, :length] = batch['gene'].numpy()
                saved_values[start:end, :length] = batch['expr'].numpy()
                target = torch.from_numpy(y[rows]).to('cuda')
                optimizer.zero_grad(set_to_none=True)
                logits = classify(model, batch, tokenizer.pad_id)
                loss = torch.nn.functional.cross_entropy(logits, target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Non-finite training loss at epoch {epoch}, step {step}')
                loss.backward()
                if epoch == 1 and step == 1:
                    missing_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
                    if missing_grad:
                        raise RuntimeError(f'Trainable parameters without gradients: {missing_grad}')
                    print(f'[Model input] gene/value={tuple(batch["gene"].shape)}, labels={tuple(target.shape)}, logits={tuple(logits.shape)}', flush=True)
                    print('[Backward] all trainable parameters received gradients', flush=True)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_value = float(loss.detach().item())
                batch_correct = int((logits.detach().argmax(dim=1) == target).sum().item())
                loss_sum += loss_value * len(rows)
                correct += batch_correct
                steps.append({'epoch': epoch, 'step': step, 'cells': len(rows), 'padded_length': length,
                              'loss': loss_value, 'correct': batch_correct, 'grad_norm_before_clip': float(grad_norm.item())})
                if step % 25 == 0 or end == len(order):
                    print(f'[Train {epoch}/{opt.epochs}] {end}/{len(order)} cells, online_loss={loss_sum/end:.4f}', flush=True)
                del logits, loss, target
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - t0
        finally:
            sampler.stop()
            pd.DataFrame(sampler.rows).to_csv(epoch_out / 'training_resources.csv', index=False)
        training_metrics = sampler.metrics()
        optimizer.zero_grad(set_to_none=True)
        model.cur_gene_token_embs = None
        pd.DataFrame(steps).to_csv(epoch_out / 'training_steps.csv', index=False)
        np.savez_compressed(epoch_out / 'training_tokenized_inputs.npz', gene_ids=saved_genes,
                            expression_bins=saved_values, original_row_indices=order, labels=y[order])
        del saved_genes, saved_values
        deltas = {k: float((model.state_dict()[k].detach().cpu() - initial[k]).abs().max().item()) for k in probes}
        if not all(np.isfinite(v) and v > 0 for v in deltas.values()):
            raise RuntimeError(f'Expected encoder and classifier weights to change: {deltas}')
        validation_metrics, val_logits = evaluate(model, val_cache, y[validation], cell_ids[validation],
            labels, tokenizer.pad_id, opt.batch_size, epoch_out / 'validation')
        bundle = opt.model_out / f'epoch_{epoch:03d}'
        bundle.mkdir()
        save_start = time.perf_counter()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, bundle / 'model.pt')
        torch.save({'epoch_completed': epoch, 'optimizer': optimizer.state_dict(),
                    'rng_state': rng_state(), 'run_config': config}, bundle / 'training_state.pt')
        for filename in ['model_config.json', 'preprocessing.json', 'labels.json', 'genes.json', 'vocab.json',
                         'run_config.json', 'pretrained_args.json', 'environment.json', 'requirements_freeze.txt']:
            shutil.copy2(out / filename, bundle / filename)
        shutil.copy2(__file__, bundle / 'predict.py')
        (bundle / 'README.txt').write_text(
            'Saved scGPT encoder + trained cell-type classifier. model.pt is a plain state_dict.\n'
            'Run prediction inside the scgpt-spark container using its existing environment:\n'
            'python /path/to/epoch_001/predict.py --mode predict --bundle /path/to/epoch_001 '
            '--input /workspace/scgpt/data/pancreas_demo/demo_test.h5ad --out /workspace/scgpt/results/NEW_NAME\n'
            'All input cells are retained. Unknown labels are excluded from closed-set metrics and reported separately.\n'
            'No unknown-class detector is trained; softmax confidence is not an unknown detector.\n'
            'training_state.pt stores optimizer and RNG states for recovery/development; this script has no resume CLI.\n'
            'Validation is a cell split within the supplied train file, not a donor-independent benchmark.\n')
        save_seconds = time.perf_counter() - save_start
        # Verify strict reload and identical predictions from the saved weights.
        reloaded = TransformerModel(**constructor, vocab=vocab)
        reloaded.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        reloaded.float().to('cuda').eval()
        end = min(opt.batch_size, len(validation))
        with torch.inference_mode():
            reload_logits = classify(reloaded, cached_batch(val_cache, 0, end, tokenizer.pad_id), tokenizer.pad_id).cpu().numpy()
        reload_error = float(np.max(np.abs(reload_logits - val_logits[:end])))
        if not np.allclose(reload_logits, val_logits[:end], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f'Saved-model reload mismatch: max error {reload_error}')
        del reloaded
        gc.collect()
        write_json(bundle / 'manifest.json', {
            'model_sha256': sha256(bundle / 'model.pt'),
            'training_state_sha256': sha256(bundle / 'training_state.pt'),
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
            'results_directory': str(out),
        })
        record = {
            'epoch': epoch, 'training_cells': len(training), 'training_seconds': training_seconds,
            'training_cells_per_second': len(training) / training_seconds,
            'online_training_loss': loss_sum / len(training), 'online_training_accuracy': correct / len(training),
            **training_metrics, 'validation': validation_metrics, 'weight_probe_max_abs_deltas': deltas,
            'model_bundle': str(bundle), 'checkpoint_save_seconds': save_seconds,
            'strict_reload_passed': True, 'reload_max_abs_logit_error': reload_error,
        }
        write_json(epoch_out / 'summary.json', record)
        epoch_summaries.append(record)
        score = validation_metrics['macro_f1_all_training_classes']
        if best is None or score > best['validation_macro_f1']:
            best = {'epoch': epoch, 'validation_macro_f1': score, 'model_bundle': str(bundle)}
            write_json(opt.model_out / 'best.json', best)
        print(f'[Validation] accuracy={validation_metrics["accuracy"]:.4f}, macro-F1={score:.4f}', flush=True)
        print(f'[Saved] {bundle}; reload check passed', flush=True)
    summary = {
        'status': 'complete', 'task': 'scGPT supervised cell-type fine-tuning',
        'training_cells': len(training), 'validation_cells': len(validation), 'classes': len(labels),
        'matched_genes': len(genes), 'epochs': opt.epochs, 'batch_size': opt.batch_size,
        'max_length_including_cls': opt.max_length, 'precision': 'FP32', 'seed': opt.seed,
        'parameter_count_constructed_model': sum(p.numel() for p in model.parameters()),
        'parameter_count_trainable': sum(p.numel() for p in trainable),
        'parameter_count_encoder': sum(p.numel() for name, p in model.named_parameters() if name.startswith(CORE)),
        'input_preparation_seconds': input_seconds, 'model_loading_seconds': model_loading_seconds,
        'validation_tokenization_seconds': cache_seconds,
        'training_seconds_total': sum(e['training_seconds'] for e in epoch_summaries),
        'training_gpu_peak_allocated_gib': max(e['gpu_peak_allocated_gib'] for e in epoch_summaries),
        'training_gpu_peak_reserved_gib': max(e['gpu_peak_reserved_gib'] for e in epoch_summaries),
        'training_process_rss_peak_observed_gib': max(e['python_process_rss_peak_observed_gib'] for e in epoch_summaries),
        'best_checkpoint': best, 'validation': epoch_summaries[best['epoch'] - 1]['validation'],
        'weights_updated': True, 'encoder_and_classifier_updated': True, 'saved_model_reload_verified': True,
        'test_evaluated': False, 'epochs_detail': epoch_summaries,
        'timing_scope': config['training_timing_scope'], 'memory_notes': MEMORY_NOTES,
        'interpretation': 'Small-run fine-tuning and validation check; not a full-data or donor-independent accuracy benchmark',
    }
    write_json(out / 'summary.json', summary)
    print('\n' + json.dumps({k: summary[k] for k in [
        'status', 'training_cells', 'validation_cells', 'epochs', 'training_seconds_total',
        'training_gpu_peak_allocated_gib', 'training_process_rss_peak_observed_gib',
        'best_checkpoint', 'saved_model_reload_verified', 'test_evaluated']}, indent=2), flush=True)
    print('SCGPT_FINETUNE_COMPLETE', flush=True)


def predict(opt):
    if opt.bundle is None or opt.input is None:
        raise ValueError('Prediction requires --bundle and --input')
    bundle = opt.bundle
    constructor = json.loads((bundle / 'model_config.json').read_text())
    preprocess = json.loads((bundle / 'preprocessing.json').read_text())
    labels = json.loads((bundle / 'labels.json').read_text())['labels']
    label_to_id = {name: i for i, name in enumerate(labels)}
    genes = json.loads((bundle / 'genes.json').read_text())
    vocab = GeneVocab.from_file(bundle / 'vocab.json')
    vocab.set_default_index(vocab[preprocess['pad_token']])
    data = ad.read_h5ad(opt.input)
    if data.n_obs == 0:
        raise ValueError('Input has no cells')
    matrix = extract_matrix(data, genes)
    cell_ids = data.obs_names.astype(str).to_numpy()
    originals = (data.obs['Celltype'].astype('string').fillna('<unlabeled>').astype(str).to_numpy()
                 if 'Celltype' in data.obs else np.full(data.n_obs, '<unlabeled>'))
    y = np.asarray([label_to_id.get(label, -1) for label in originals], dtype=np.int64)
    pd.DataFrame({'cell_id': cell_ids, 'original_label': originals, 'known_training_label': y >= 0}).to_csv(
        opt.out / 'input_cells.tsv', sep='\t', index=False)
    write_json(opt.out / 'prediction_config.json', {
        'bundle': str(bundle), 'model_sha256': sha256(bundle / 'model.pt'),
        'input': str(opt.input), 'input_sha256': sha256(opt.input), 'seed': opt.seed,
        'batch_size': opt.batch_size, 'precision': 'FP32', 'preprocessing': preprocess,
        'unknown_label_counts': pd.Series(originals[y < 0]).value_counts().astype(int).to_dict(),
        'unknown_policy': 'retain all predictions; evaluate known ground-truth labels only; no unknown-class detector',
    })
    model = TransformerModel(**constructor, vocab=vocab)
    model.load_state_dict(torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    model.float().to('cuda').eval()
    tokenizer = Tokenizer(matrix, genes, vocab, preprocess)
    seed_all(opt.seed)
    t0 = time.perf_counter()
    cache = tokenizer.cache(np.arange(len(y)), opt.batch_size)
    tokenization_seconds = time.perf_counter() - t0
    np.savez_compressed(opt.out / 'tokenized_inputs.npz', gene_ids=cache[0], expression_bins=cache[1], labels=y)
    metrics, _ = evaluate(model, cache, y, cell_ids, labels, tokenizer.pad_id, opt.batch_size, opt.out)
    summary = {'status': 'complete', 'task': 'fine-tuned cell-type predictions',
               'tokenization_seconds': tokenization_seconds, 'warmup_steps': 0,
               'weights_updated': False, 'model_bundle': str(bundle), 'memory_notes': MEMORY_NOTES, **metrics}
    write_json(opt.out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print('SCGPT_PREDICTION_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-out', type=Path)
    parser.add_argument('--n-cells', type=int, default=1000, help='training subset size; 0=all training-pool cells')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=1200)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--input', type=Path)
    opt = parser.parse_args()
    if (opt.batch_size < 1 or opt.epochs < 1 or opt.n_cells < 0 or opt.max_length < 2
            or not 0 < opt.validation_fraction < 1 or not np.isfinite(opt.lr) or opt.lr <= 0
            or not np.isfinite(opt.weight_decay) or opt.weight_decay < 0 or not 0 <= opt.seed < 2**32 - 10000):
        parser.error('Invalid numeric option')
    if opt.mode == 'train' and opt.model_out is None:
        parser.error('Training requires --model-out')
    opt.out.mkdir(parents=True, exist_ok=True)
    if (opt.out / 'run_config.json').exists() or (opt.out / 'summary.json').exists() or (opt.out / 'prediction_config.json').exists():
        raise FileExistsError(f'Use a new output directory: {opt.out}')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; stopping instead of running on CPU')
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        seed_all(opt.seed)
        preserve_environment(opt.out)
        train(opt) if opt.mode == 'train' else predict(opt)
    except Exception as exc:
        write_json(opt.out / 'failure.json', {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scgpt/results/length_compare_20261023_040313_501423/compare_lengths.py — 원문 전문 (19,900 bytes)</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/length_compare_20261023_040313_501423/compare_lengths.py", "sha256": "647d484d773956dc5f6591aa8db8de3ce6721b57fe08a6b6d97be476ff3ebb71", "encoding": "utf-8", "added_final_newline": false} -->
```python
"""Run existing, GPU-validated scGPT scripts at four maximum token lengths.

Each case runs in a fresh process. Cell IDs, order, checkpoint, batch size,
precision and random seed are fixed within each task. The existing official
random gene sampling is preserved; nested gene subsets are not enforced.
This is a resource benchmark of maximum-length settings, not a controlled
biological accuracy ablation or a fixed actual-length benchmark.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/scgpt')
LENGTHS = [256, 512, 1024, 1200]
SEED = 42
N_CELLS = 1000
BATCH = 8


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(values):
    import numpy as np
    return {'min': int(np.min(values)), 'mean': float(np.mean(values)),
            'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
            'max': int(np.max(values))}


def prepare(out):
    """Inspect the real input and exact subsets before launching any benchmark."""
    import numpy as np
    import pandas as pd
    import anndata as ad
    from scipy import sparse
    from sklearn.model_selection import train_test_split
    from scgpt.tokenizer import GeneVocab

    spec = importlib.util.spec_from_file_location('baseline_finetune', out / 'source/02_finetune.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)  # guarded main: does not train or initialize a model
    args = read_json(ROOT / 'models/whole-human/args.json')
    if args.get('n_bins') != 51 or args.get('input_style') != 'binned':
        raise ValueError('Unexpected pretrained input configuration')
    vocab = GeneVocab.from_file(ROOT / 'models/whole-human/vocab.json')
    for token in [args['pad_token'], '<cls>', '<eoc>']:
        if token not in vocab:
            vocab.append_token(token)
    report = {'length_caps_including_cls': LENGTHS, 'pad_token_id': int(vocab[args['pad_token']]),
              'checkpoint_sha256': sha256(ROOT / 'models/whole-human/best_model.pt'),
              'input_files': {}, 'tasks': {}}
    gene_reference = None
    for task, split, expected in [('inference', 'test', (4218, 3000)),
                                  ('finetune', 'train', (10600, 3000))]:
        path = ROOT / f'data/pancreas_demo/demo_{split}.h5ad'
        data = ad.read_h5ad(path)
        if data.shape != expected or 'Celltype' not in data.obs:
            raise ValueError(f'Unexpected {split} data: {data.shape}, {data.obs.columns.tolist()}')
        if not data.var_names.is_unique or not data.obs_names.is_unique or data.obs['Celltype'].isna().any():
            raise ValueError('Cell IDs/genes must be unique and labels must be present')
        genes = np.asarray(data.var_names.astype(str))
        if gene_reference is not None and not np.array_equal(genes, gene_reference):
            raise ValueError('Training and test gene order differs from the inspected data')
        gene_reference = genes.copy()
        mask = np.asarray([g in vocab for g in genes])
        if int(mask.sum()) != 2999:
            raise ValueError(f'Expected 2999 matched genes, got {mask.sum()}')
        label_values = data.obs['Celltype'].astype(str).to_numpy()
        if task == 'inference':
            selected = np.sort(np.random.default_rng(SEED).choice(data.n_obs, N_CELLS, replace=False))
            order = selected.copy()
        else:
            labels = sorted(set(label_values))
            if len(labels) != 13:
                raise ValueError('Expected 13 training classes')
            label_map = {label: i for i, label in enumerate(labels)}
            y = np.asarray([label_map[label] for label in label_values], dtype=np.int64)
            pool, validation = train_test_split(np.arange(data.n_obs), test_size=.2,
                                               random_state=SEED, stratify=y)
            selected = baseline.choose_training_subset(pool, y, N_CELLS, SEED)
            order = np.random.default_rng(SEED + 1).permutation(selected)
            if np.intersect1d(selected, validation).size:
                raise RuntimeError('Train/validation overlap')
        subset = data[order, mask]
        matrix = subset.X.toarray() if sparse.issparse(subset.X) else np.asarray(subset.X)
        matrix = matrix.astype(np.float32, copy=False)
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError('Nonfinite or negative expression values')
        raw_lengths = np.count_nonzero(matrix, axis=1) + 1
        if np.any(raw_lengths < 2):
            raise ValueError('A selected cell has no nonzero matched expression')
        cell_ids = data.obs_names[order].astype(str).to_numpy()
        pd.DataFrame({'original_row_index': order, 'cell_id': cell_ids,
                      'Celltype': label_values[order], 'tokens_before_sampling_including_cls': raw_lengths}).to_csv(
            out / f'{task}_input_cells.tsv', sep='\t', index=False)
        profile = {'source_split': split, 'source_shape': list(data.shape),
                   'selected_cells': len(order), 'matched_genes': int(mask.sum()),
                   'cell_order_original_row_indices': order.tolist(),
                   'tokens_before_sampling_including_cls': stats(raw_lengths), 'caps': {}}
        for cap in LENGTHS:
            lengths = np.minimum(raw_lengths, cap)
            padded = np.asarray([lengths[start:start + BATCH].max() for start in range(0, len(order), BATCH)])
            profile['caps'][str(cap)] = {
                'actual_nonpadding_tokens': stats(lengths), 'batch_padded_tokens': stats(padded),
                'cells_reaching_cap': int((raw_lengths >= cap).sum()),
                'cells_shortened': int((raw_lengths > cap).sum()),
                'batches_reaching_cap': int((padded == cap).sum()),
            }
        report['input_files'][split] = {'path': str(path), 'sha256': sha256(path)}
        report['tasks'][task] = profile
        print(f'[Input check] {task}: source={data.shape}, selected={matrix.shape}', flush=True)
        print(f'  Actual tokens including CLS: {stats(raw_lengths)}', flush=True)
        for cap in LENGTHS:
            c = profile['caps'][str(cap)]
            print(f'  cap={cap:4d}: mean actual={c["actual_nonpadding_tokens"]["mean"]:.1f}, '
                  f'mean padded={c["batch_padded_tokens"]["mean"]:.1f}, '
                  f'cells reaching cap={c["cells_reaching_cap"]}/{len(order)}', flush=True)
        del data, subset, matrix
        gc.collect()
    write_json(out / 'input_length_report.json', report)
    print('LENGTH_INPUT_CHECK_COMPLETE', flush=True)


def run_logged(command, log_path):
    t0 = time.perf_counter()
    with Path(log_path).open('w') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    return {'returncode': status, 'whole_process_seconds': time.perf_counter() - t0}


def collect(out):
    import numpy as np
    import pandas as pd

    plan = read_json(out / 'experiment_config.json')
    profile = read_json(out / 'input_length_report.json')
    records = read_json(out / 'process_records.json') if (out / 'process_records.json').exists() else {}
    rows = []
    for case in plan['cases']:
        case_dir = out / case['name']
        if not (case_dir / 'summary.json').exists():
            continue
        result = read_json(case_dir / 'summary.json')
        if result.get('status') != 'complete':
            continue
        task, cap = case['task'], case['max_length']
        assert result['batch_size'] == BATCH and result['precision'] == 'FP32' and result['seed'] == SEED
        assert result['max_length_including_cls'] == cap
        expected = profile['tasks'][task]
        env = read_json(case_dir / 'environment.json')
        provenance = read_json(case_dir / 'provenance.json')
        assert provenance['input']['sha256'] == profile['input_files'][expected['source_split']]['sha256']
        checkpoint = provenance['checkpoint' if task == 'inference' else 'pretrained_checkpoint']
        assert checkpoint['sha256'] == profile['checkpoint_sha256']
        if task == 'inference':
            order = pd.read_csv(case_dir / 'selected_cells.tsv', sep='\t')['original_row_index'].to_numpy()
            token_path = case_dir / 'tokenized_inputs.npz'
            shapes = pd.read_csv(case_dir / 'batch_shapes.tsv', sep='\t')['padded_token_length'].to_numpy()
            elapsed = result['inference_seconds']
            allocated, reserved, rss = (result[k] for k in ['gpu_peak_allocated_gib',
                'gpu_peak_reserved_gib', 'python_process_rss_peak_observed_gib'])
            reload_ok, model_bundle = None, None
        else:
            assert result['epochs'] == 1 and result['training_cells'] == N_CELLS
            token_path = case_dir / 'epoch_001/training_tokenized_inputs.npz'
            with np.load(token_path, allow_pickle=False) as tokens:
                order = tokens['original_row_indices'].copy()
            shapes = pd.read_csv(case_dir / 'epoch_001/training_steps.csv')['padded_length'].to_numpy()
            elapsed = result['training_seconds_total']
            allocated, reserved, rss = (result[k] for k in ['training_gpu_peak_allocated_gib',
                'training_gpu_peak_reserved_gib', 'training_process_rss_peak_observed_gib'])
            reload_ok = result['saved_model_reload_verified']
            model_bundle = result['best_checkpoint']['model_bundle']
            assert reload_ok and result['encoder_and_classifier_updated']
        np.testing.assert_array_equal(order, expected['cell_order_original_row_indices'])
        with np.load(token_path, allow_pickle=False) as tokens:
            ids = tokens['gene_ids']
            assert ids.shape == (N_CELLS, cap)
            used = np.count_nonzero(ids != profile['pad_token_id'], axis=1)
        expected_lengths = pd.read_csv(out / f'{task}_input_cells.tsv', sep='\t')['tokens_before_sampling_including_cls'].to_numpy()
        np.testing.assert_array_equal(used, np.minimum(expected_lengths, cap))
        expected_padded = np.asarray([used[start:start + BATCH].max() for start in range(0, N_CELLS, BATCH)])
        np.testing.assert_array_equal(shapes, expected_padded)
        shared_ram = env['system_total_memory_gib']
        rows.append({
            'task': task, 'source_split': expected['source_split'], 'cells': N_CELLS,
            'max_length_including_cls': cap, 'actual_tokens_min': int(used.min()),
            'actual_tokens_mean': float(used.mean()), 'actual_tokens_max': int(used.max()),
            'batch_padded_tokens_mean': float(shapes.mean()), 'batch_padded_tokens_max': int(shapes.max()),
            'time_seconds': elapsed, 'cells_per_second': N_CELLS / elapsed,
            'gpu_peak_allocated_gib': allocated, 'gpu_peak_reserved_gib': reserved,
            'process_rss_peak_observed_gib': rss,
            'gpu_allocation_pct_of_system_ram': allocated / shared_ram * 100,
            'system_total_memory_gib': shared_ram,
            'whole_process_seconds': records.get(case['name'], {}).get('whole_process_seconds'),
            'same_cells_and_order_verified': True, 'checkpoint_and_input_hashes_verified': True,
            'saved_model_reload_verified': reload_ok, 'model_bundle': model_bundle,
            'timing_scope': result['timing_scope'], 'results_directory': str(case_dir),
        })
    if not rows:
        raise RuntimeError('No completed measurement cases to aggregate')
    frame = pd.DataFrame(rows).sort_values(['task', 'max_length_including_cls'])
    frame.to_csv(out / 'comparison.csv', index=False)
    frame.to_csv(out / 'comparison.tsv', sep='\t', index=False)
    display = frame[['task', 'max_length_including_cls', 'actual_tokens_mean', 'batch_padded_tokens_mean',
                     'time_seconds', 'gpu_peak_allocated_gib', 'gpu_peak_reserved_gib', 'process_rss_peak_observed_gib']]
    display = display.rename(columns={'max_length_including_cls': 'cap', 'actual_tokens_mean': 'actual_mean',
        'batch_padded_tokens_mean': 'padded_mean', 'time_seconds': 'seconds',
        'gpu_peak_allocated_gib': 'GPU_alloc_GiB', 'gpu_peak_reserved_gib': 'GPU_res_GiB',
        'process_rss_peak_observed_gib': 'RSS_GiB'})
    table = display.to_string(index=False, float_format=lambda v: f'{v:.3f}')
    (out / 'comparison_table.txt').write_text(table + '\n')
    write_json(out / 'summary.json', {
        'status': 'complete' if len(rows) == len(plan['cases']) else 'partial',
        'completed_cases': len(rows), 'expected_cases': len(plan['cases']), 'measurements': rows,
        'design': plan, 'resource_notes': [
            'Maximum-length caps include CLS and are not fixed actual token lengths.',
            'Inference uses 1000 test cells; fine-tuning uses 1000 training cells. Compare length effects within each task.',
            'Cell identities, order, input hashes and initial checkpoint hashes are verified across length conditions.',
            'Official random gene sampling is unchanged; subsets are not guaranteed to be nested between caps.',
            'FP32, batch 8, one measurement per condition; no GPU utilization, temperature or power monitoring.',
            'Inference excludes one-batch warmup. Fine-tuning includes first-step initialization and excludes validation/saving.',
            'Fresh process for every case. Reserved memory may include within-case warmup caches.',
            'PyTorch CUDA allocator counters and process RSS are separate measurements on shared physical memory; do not sum them.',
            'Whole-process time includes imports, input/model load, validation, saving and reload checks where applicable.',
        ],
    })
    print('\n' + table + '\n', flush=True)
    # A shareable scientific figure; optional failure does not discard completed measurements.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        for axis, field, title in zip(axes,
            ['time_seconds', 'gpu_peak_allocated_gib', 'process_rss_peak_observed_gib'],
            ['Measured time (s)', 'Peak GPU allocation (GiB)', 'Observed process RSS peak (GiB)']):
            for task, group in frame.groupby('task', sort=False):
                group = group.sort_values('max_length_including_cls')
                axis.plot(group['max_length_including_cls'], group[field], marker='o', label=task)
            axis.set_xlabel('Maximum tokens including CLS')
            axis.set_ylabel(title)
            axis.set_xticks(LENGTHS)
            axis.grid(alpha=.25)
            axis.set_ylim(bottom=0)
        axes[0].legend()
        fig.suptitle('scGPT on Spark: maximum input length vs resource use', fontsize=14)
        fig.text(.5, .01, '1,000 fixed cells per task | batch 8 | FP32 | 1 run per cap | inference: test cells; fine-tuning: train cells',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=[0, .05, 1, .93])
        fig.savefig(out / 'comparison.png', dpi=180)
        fig.savefig(out / 'comparison.svg')
        plt.close(fig)
    except Exception as exc:
        (out / 'plot_warning.txt').write_text(str(exc) + '\n')
        print(f'[Plot] Could not render figure; CSV/JSON results are saved: {exc}', flush=True)
    print('SCGPT_LENGTH_COMPARISON_COMPLETE' if len(rows) == len(plan['cases']) else 'SCGPT_LENGTH_COMPARISON_PARTIAL', flush=True)


def run(out):
    if (out / 'experiment_config.json').exists():
        raise FileExistsError('Use a new comparison directory')
    for source in ['01_inference.py', '02_finetune.py']:
        if not (out / 'source' / source).is_file():
            raise FileNotFoundError(out / 'source' / source)
    cases = [{'name': f'{task}_L{cap:04d}', 'task': task, 'max_length': cap}
             for task in ['inference', 'finetune'] for cap in LENGTHS]
    plan = {'cases': cases, 'seed': SEED, 'cells_per_task': N_CELLS, 'batch_size': BATCH,
            'precision': 'FP32', 'epochs_per_finetune': 1, 'repeats_per_condition': 1,
            'length_policy': 'official random nonzero-gene sampling with dynamic batch padding; cap includes CLS',
            'inference_split': 'test', 'finetune_split': 'train; stratified 20% validation held out',
            'source_sha256': {name: sha256(out / 'source' / name) for name in ['01_inference.py', '02_finetune.py']}}
    write_json(out / 'experiment_config.json', plan)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'prepare', '--out', str(out)]
    preflight = run_logged(command, out / 'input_check.log')
    if preflight['returncode']:
        raise RuntimeError('Input check failed; no GPU benchmark was started. See input_check.log')
    process_records = {}
    for case in cases:
        task, cap = case['task'], case['max_length']
        case_dir = out / case['name']
        case_dir.mkdir()
        name = '01_inference.py' if task == 'inference' else '02_finetune.py'
        script = case_dir / name
        shutil.copy2(out / 'source' / name, script)
        command = [sys.executable, '-u', str(script), '--out', str(case_dir),
                   '--n-cells', str(N_CELLS), '--batch-size', str(BATCH),
                   '--max-length', str(cap), '--seed', str(SEED)]
        if task == 'inference':
            command += ['--split', 'test']
        else:
            command += ['--mode', 'train', '--epochs', '1', '--validation-fraction', '0.2',
                        '--lr', '0.0001', '--weight-decay', '0.01',
                        '--model-out', str(ROOT / 'models/finetuned' / out.name / f'L{cap:04d}')]
        (case_dir / 'run_command.sh').write_text('#!/usr/bin/env bash\n' + shlex.join(command) + '\n')
        print(f'\n[Case] {case["name"]}: {N_CELLS} cells, batch={BATCH}, FP32', flush=True)
        process_records[case['name']] = run_logged(command, case_dir / 'run.log')
        write_json(out / 'process_records.json', process_records)
        if process_records[case['name']]['returncode']:
            write_json(out / 'failure.json', {'failed_case': case, 'process': process_records[case['name']]})
            raise RuntimeError(f'{case["name"]} failed; prior outputs are preserved. See {case_dir}/run.log')
    collect_command = [sys.executable, '-u', str(Path(__file__).resolve()), 'collect', '--out', str(out)]
    aggregation = run_logged(collect_command, out / 'collection.log')
    if aggregation['returncode']:
        raise RuntimeError('Aggregation failed; measurement files remain saved in each case directory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'prepare', 'collect'])
    parser.add_argument('--out', type=Path, required=True)
    opt = parser.parse_args()
    {'run': run, 'prepare': prepare, 'collect': collect}[opt.mode](opt.out)

```
<!-- END_RESTORE_FILE -->


</details>
