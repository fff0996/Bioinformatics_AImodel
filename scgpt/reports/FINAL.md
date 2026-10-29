# scGPT 최종 통합 보고서

기본 실험과 모든 후속 실험을 이 문서에 함께 수록했다. 다른 기본/후속 문서를 따로 읽을 필요 없이 아래 요약과 본문을 보면 된다.

## 1. 입력과 메모리 결론

세포×유전자발현 행렬 → 세포별유전자ID·발현bin값 → 임베딩/세포유형. batch는세포행수. 최대1200토큰은CLS+유전자+padding자리.

**목표는 메모리 할당80% 조건 확인**이다. 전체공유RAM121.688GiB의80%는약97.35GiB다. GPU-Util은별도연산활동지표이며이목표의성공판정에쓰지않는다.

|작업|관측메모리 peak GiB|공유RAM대비|해당 입력·배치·정밀도|80% 확인/제약|
|---|---|---|---|---|---|
|실제30만세포 추론|**5.477**|4.50%|고유30만세포를256개씩처리. 세포당최대1200토큰,배치내실제최대길이padding,FP32.|80% 미확인; 30만세포를동시에GPU에올린값아님|
|학습 용량 probe|**100.750**|82.79%|실제1000학습세포cache 중292세포×1200고정padding,FP32·AdamW·3step. 실제평균nonpad약306자리.|80% 도달 확인(82.79%). padding포함용량probe이며전체데이터파인튜닝과구분|
|실제 전체 pancreas 파인튜닝|**3.374**|2.77%|train8480/val2120/test4218세포,학습batch8·FP32·최대1200·원래dynamic padding·1epoch.|큰batch284/256/224는메모리guard중단. 이조건에서80%성공을입증하지못함|

메모리는PyTorch의경우CUDA allocated peak다. reserved는예약량으로별도다. **AlphaFold2는컴파일·warmup등을포함한JAX프로세스누적peak**이므로CUDA구간peak와같은정의로비교하지않는다. RSS와CUDA를더하지않는다. 표는시험한조건의관측값이며모델의절대최대나모든모델의80%달성을의미하지않는다.

## 2. 기본 정보·입력 검사·추론·학습·평가

## scGPT 기존 완료 결과와 입력 규모·GPU 사용률 보완


보완run `supplement_20261027_211638_936991` · 상태 **complete** · [원본summary](../results/supplement_20261027_211638_936991/summary.json) · [자원시계열](../results/supplement_20261027_211638_936991/resources.csv)

### 기존 기본 실험은 이미 완료

2026-10-23에입력검사·사전학습임베딩추론·1,000세포파인튜닝·저장/재로드·독립test·최대길이256/512/1024/1200비교를완료했다. 원본파일과학습모델을보존했고이번보완값을과거GPU-Util실측처럼소급해붙이지않는다.

[기존test](../results/finetuned_test_20261023_034345_99523/summary.json) · [기존길이비교](../results/length_compare_20261023_040313_501423/comparison.csv)

기존저장모델test4218세포중학습라벨이있는4213세포의Accuracy 0.9380,학습13클래스macroF1 0.5350,weightedF1 0.9312.test-onlyMHCclassII5세포는해당지표에서제외했다. 기존1epoch파인튜닝CUDApeak3.3741GiB,저장모델test추론peak0.3811GiB다.

### 동일 환경·모델·실제 입력

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

### 개수·길이별 GPU 사용률과 메모리

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

### 학습 GPU 사용률 보완

| 조건 | train세포 | epochs | batch | 초 | GPU평균/최대% | 모델적재GiB | 학습peakGiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 누락사용률계측보완 | 1000 | 1 | 8 | 24.1723 | 88.6887/95.0000 | 0.1913 | 3.3741 |

GPU사용률이없었던기존학습조건을계측하기위해동일1000세포/1epoch를별도run에서실행했다. 원본모델/결과는유지했고새모델은별도경로에저장했다. 원본02_finetune.py복사본에GPU-Util시계열과적재baseline만추가해학습알고리즘·split·optimizer를유지했다.

전체train10600에서동일seed42cell단위validation2120,별도train1000을골랐으며13유형정답으로encoder+classifier를갱신했다. 저장모델bundle /workspace/scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001.새보완모델은validation과재로드를검증했고독립test를새로평가한것은아니다. 위기존testAccuracy와새모델을혼동하지않는다.

학습sampler는0.2초대기+조회시간과시작/끝표본을포함하므로첫/끝nvidia-smi관측창이주변활동을포함할수있다. resource CSV는training_util/epoch_001/training_resources.csv다. GPU메모리는fit구간allocatorpeak,validation/저장은시간에서제외했다. optimizer/RNG/epoch저장,encoder/head변경,원본strict reload검증을수행했다. 독립프로세스validation재로드 {"independent_process": true, "validation_cells": 2120, "max_abs_difference": 0.0, "predictions_match": true, "passed": true}

### 해석과 한계

Spark는공유RAM121.688GiB이므로RSS와CUDA를더하지않는다. allocated는사용tensor, reserved는cache포함allocator예약이며장치전체점유율과다르다. rawcount유래·donor독립검증은확인되지않았다. 개수증가가peak를비례증가시킨다고가정하지않으며batch8이고정돼메모리증가는작았다. 메모리90%도달은별도승인된후속batch확대실험에서검토한다.

공식소스: [scGPT](https://github.com/bowang-lab/scGPT). 정확한설치리비전·원본모델hash와코드는각기존/보완run의provenance와environment에보존했다.

## 3. 후속 실험 통합 — 학습 배치 확대

## scGPT 후속: batch 확대와 실제 메모리 용량


### 입력을 읽는 법: 세포 행렬, 배치, 유전자 토큰

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

#### 왜 전체 세포 수보다 배치가 순간 메모리에 영향을 주는가?

전체30만세포를 한꺼번에 GPU에 올리지 않고, `세포8개 전송 → 계산 → 결과를 CPU/파일에 저장 → 다음8개`처럼 나눠 처리한다. 배치와 실제 토큰 길이 등 조건이 같으면 전체 세포 수 증가는 주로 처리 횟수·총시간·출력 파일 크기를 늘린다. 순간 GPU 메모리가 세포 수에 비례해 증가하지는 않는다. 세포별 길이 분포가 달라지면 dynamic padding을 통해 메모리가 달라질 수 있다.

#### 추론 결과와 학습 용량 결과를 구분하기

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

![batch별메모리와처리량](memory_batch_curve.png)

### 정확한 입력·학습·시간 범위

기존실제train1000세포의학습token화cache(gene IDs,expression bins,13개세포유형정답)를사용했다. 한batch내중복세포는없으며firstB중첩부분집합을같은조건에서여러step반복한다. 각condition은최소3step/15초이고실제step수를표에썼다. 작은batch는더많은측정step을반복하므로다른학습량의정확도비교로쓰지않는다.

각batch는동일pretrainedencoder·seed42새classifier·새AdamW에서시작한다. FP32/TF32=False,AdamWlr1e-4/wd.01/betas.9,.999/eps1e-8/gradientclip1.0을고정했다. 원본51bin·CLS전처리이며가중치이론량·실측적재량과실행중activation/gradient/optimizer증가를구분한다. 각run summary의parameters와model_only_postload에있다.

**이용량실험은모든batch를1200열까지padding한cache를그대로사용한다.** 원본일반학습은batch내최대실제길이에맞춘dynamicpadding을사용하므로평균시간을원본실험과직접비교하지않는다. 실제nonpad token평균은약300대이며,나머지padding도dense학습의메모리/계산비용을발생시킨다. 따라서90%에가까운할당이모두유효한유전자계산량이라는뜻은아니다. 일관된1200길이조건의capacity상한조사다.

시간에는cached batch구성·CPU→GPU·forward/classifier/loss·backward·gradientclip·optimizer와step경계CUDA동기화를포함한다. imports·모델적재·H5읽기·token화·probe가중치저장은제외했다. 각조건allocatorpeak는reset하고현재baseline도별도로기록했다. GPU-Util은활동시간비율로FLOPS/메모리%가아니다.

원본기본결과는보존했다. capacity probe후가중치는각run/probe_models/batch_B.pt에별도저장했으며모델초기화는initial_state.pt,구조는model_config.json이다. 이probe가중치를검증된생물학적최종모델로채택하지않았고,학습재개/정확도평가를주장하지않는다.

### 중단·측정 한계

시스템MemAvailable8GiB미만또는swap+2GiB에서자체프로세스를중단한다. 다음batch예상reserved가현재예약량+가용량−8GiB예산을넘으면확대전에중단하거나예상안전batch로축소한다. 첫coarse sweep은8%예측여유,80%경계refine은이전실측에대해2%여유를적용했다. 중단은90%불가능증명이나모델절대최대가아니다.

- `memory_batch_20261027_213310_982032`: complete / tested_predicted_safe_cap
- `memory_batch_refine_20261027_214026_294469`: complete / available_unique_input_limit

reserved는cache/단편화포함이며실제살아있는tensor할당과다르다. nvidia-smi는SparkVRAM용량을제공하지않으므로PyTorchcounter와전체공유RAM·시스템가용량을구분했다. RSS와CUDA는더하지않는다. RSS/가용량은0.5초대기+조회시계열sampled값이며순간최저를보장하지않는다.

### 원본 경로

- [run memory_batch_20261027_213310_982032](../results/memory_batch_20261027_213310_982032/summary.json) · 로그 (로컬 보관: `../results/memory_batch_20261027_213310_982032/run.log`) · [시계열](../results/memory_batch_20261027_213310_982032/resources.csv)
- [run memory_batch_refine_20261027_214026_294469](../results/memory_batch_refine_20261027_214026_294469/summary.json) · 로그 (로컬 보관: `../results/memory_batch_refine_20261027_214026_294469/run.log`) · [시계열](../results/memory_batch_refine_20261027_214026_294469/resources.csv)

[기본 scGPT 보고서](scGPT.md) · [기존모델·공식소스](https://github.com/bowang-lab/scGPT)

## 4. 후속 실험 통합 — 실제 전체세포 학습

## scGPT 실제 pancreas 전체 데이터 처리


실행: `realdata_full_20261027_215022_050816`. 총 14,818개 실제 세포: 학습 8,480 / validation 2,120 / 독립 test 4,218. FP32, batch8, seed42, 최대1,200토큰, 원래 dynamic padding, 1epoch. 입력 ID 중복·파일 간 겹침 없음은 source_data_audit.json에서 확인했다. 원시 count provenance는 확인되지 않았다.

|조건|학습시간|allocated peak|GPU 평균/최대|독립 test Accuracy|
|---|---|---|---|---|
|batch8|202.88초|3.374 GiB|93.02/95%|0.92215|

모델 저장 및 validation strict reload 최대 logit 오차0. 별도 프로세스에서 test4,218세포를 예측했고, 학습 라벨에 속한4,213세포로 정확도를 계산했다. test-only5세포는 알려진 클래스 지표에서 제외했다. macroF1(학습13클래스) 0.52963, weightedF1 0.91658. 학습1epoch/1060 optimizer updates이며 donor-independent 검증은 아니다.

batch284·256·224 학습은 시스템 가용 RAM8GiB guard가 발동해 중단됐다. 완료된 학습·배치 비교 결과로 쓰지 않는다. 각각 실패 로그 및 memory_guard.json을 보존했다. 샘플링 감지 지연으로224 조건은 가용4.424GiB에서 중단됐으며, 안전한 최대 배치가 검증됐다는 의미가 아니다. 추가 대형 학습 배치 재시도 대신 사용자 우선 요청인 실제30만세포 임베딩 추론을 진행한다.

[원본 결과](../results/realdata_full_20261027_215022_050816/summary.json) · [실행·입력·저장 모델 코드](../scripts/full_realdata.py)

CUDA allocated와 reserved는 프로세스 allocator 값이다. Spark 공유 RAM에서 RSS와 CUDA를 독립 물리 메모리처럼 더하지 않는다. GPU-Util은 메모리 점유율이 아니다. 학습시간은 전처리/전송/forward/backward/optimizer를 포함하며 validation·모델저장은 제외한다.

## 5. 후속 실험 통합 — 실제30만세포 추론

## scGPT 실제30만 세포 임베딩 및 자원 측정


### 입력을 읽는 법: 세포 행렬, 배치, 유전자 토큰

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

#### 왜 전체 세포 수보다 배치가 순간 메모리에 영향을 주는가?

전체30만세포를 한꺼번에 GPU에 올리지 않고, `세포8개 전송 → 계산 → 결과를 CPU/파일에 저장 → 다음8개`처럼 나눠 처리한다. 배치와 실제 토큰 길이 등 조건이 같으면 전체 세포 수 증가는 주로 처리 횟수·총시간·출력 파일 크기를 늘린다. 순간 GPU 메모리가 세포 수에 비례해 증가하지는 않는다. 세포별 길이 분포가 달라지면 dynamic padding을 통해 메모리가 달라질 수 있다.

#### 추론 결과와 학습 용량 결과를 구분하기

| 실험 | 배치(세포 수) | 입력 길이 정책 | CUDA allocated peak | 전체 공유RAM121.688GiB 대비 |
|---|---|---|---|---|
| 실제30만 고유세포 임베딩 추론 | 8 | 최대1200, 배치 내 실제 최대길이로 padding | 0.389GiB | 약0.32% |
| 같은 실제30만 고유세포 임베딩 추론 | 256 | 위와 동일한 cache·길이 정책 | 5.477GiB | 약4.50% |
| 별도 학습 용량 probe | 292 | 모든 입력을1200자리로 고정 padding | 100.750GiB | 약82.79% |

**82.79%는 batch292의 고정 padding 학습 용량 실험에서 관측한 값**이다. batch256 실제데이터 추론에서80%를 사용한 것이 아니다. 서로 다른 학습 상태·padding 정책·데이터 조건이므로 위 행 사이의 차이를 배치만의 효과로 해석하지 않는다. 학습 용량 probe는 실제1000학습세포 cache의 부분집합을 측정 step 동안 반복한 실험이며30만세포 학습 실험이 아니다.

GPU 연산 사용률(예:95%)은 GPU 커널이 활동한 시간 비율이다. 메모리 비율은 여기서 CUDA allocated peak를 CPU/GPU 공유RAM121.688GiB로 나눈 참고값이다. 둘은 다른 지표이며 별도 전용VRAM 점유율로 해석하지 않는다. reserved는 allocator가 확보한 예약 공간으로 allocated와 구분한다.

상태: **complete**. 실행 `actual_300k_resume_20261028_093701_696230`. [원본 summary](../results/actual_300k_resume_20261028_093701_696230/summary.json) · 입력 검사 (로컬 보관: `../data_external/tabula_sapiens/prepared_300k_seed42/input_audit.json`)

### 실제 입력과 전처리

Tabula Sapiens 고정 Census2024-07-01 원본 483,152세포 × 58,482 features에서 seed42로 **300,000개 고유 실제 세포**를 선택했다. 반복 복제한 입력이 아니다. source raw.X를 사용하며 vocabulary와 일치한 유전자 40,474개를 처리한다. 원본 파일 SHA256은 download_source.json에 저장했다. raw.X의 의미와 count provenance는 원본 metadata와 구분해 확인하며, 정수형 값 표본만으로 원시 count를 확정하지 않는다.

공식51-bin 전처리, 비영 유전자, 최대1200토큰(CLS포함). 선택·유전자 sampling·binning 결과는 한번 cache해 두 배치 조건에 동일하게 제공한다. 실제 GPU 입력은 배치 내 최대 유효길이로 padding한다. 1만/10만/30만은 동일 순서의 중첩 부분집합이며, 각 batch의30만세포 단일 pass에서 누적 지점을 측정한다. 독립적인3회 실행이 아니다.

|고유 세포|평균 실제 토큰|최소/최대 토큰|길이 제한으로 sampling된 세포|
|---|---|---|---|
|10000|1159.37|189/1200|8960|
|100000|1160.36|180/1200|89708|
|300000|1160.89|176/1200|269172|

tissue·assay·cell_type 분포와 실제 cell ID/원본 행은 input_audit.json 및 cells.tsv에 보존했다. cell_type은 metadata이며 이번 임베딩 추론의 정답 분류 평가가 아니다.

재개 실행인 경우 기존 완료 지점의 출력을 재사용하고 나머지만 새 구간으로 측정한다. 해당 행의 시간/처리량은 이번 구간 처리 세포 수 기준이며 전체 누적 세포 처리 시간으로 해석하지 않는다.

### 모델과 측정

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

### 출력 검증과 한계

{"sampled_actual_cells": 256, "max_abs_embedding_difference_between_batches": 2.682209014892578e-07, "minimum_cosine_similarity": 0.9999998211860657, "output_shape": [300000, 512], "all_outputs_finite_checked_during_forward": true}

출력은 각 batch의 embeddings_batchB.npy, 300000×512 float32이다. 각 forward에서 유한값을 검사하며, 완료 후256개 실제 세포의 배치 간 embedding을 비교한다. 실제30만세포 임베딩 처리 결과이며 지도학습 정확도·donor 독립 평가·사전학습 데이터와의 독립성을 입증하지 않는다. 별도 pancreas 전체 학습 결과와 구분한다.

[공식 scGPT](https://github.com/bowang-lab/scGPT) · [공식 Census 원본 다운로드](https://chanzuckerberg.github.io/cellxgene-census/_autosummary/cellxgene_census.download_source_h5ad.html) · [Tabula Sapiens collection](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5)

실행 코드: scripts/large_scale/prepare.py, resume.py, resumable_inference.py, finalize.py. 로그·GPU 시계열·설정·예측은 results/actual_300k_resume_20261028_093701_696230에 보존한다. 전날 최초 시도는 import 경로 오류로 실패해 retry1로 실행했고, 10만세포 지점에서 사용자 요청으로 중단한 뒤 오늘 새 run에서 나머지부터 재개했다. 최초 실패·중단·재개 로그를 모두 보존했다.

### 30만 고유세포 전체 처리 최종 비교

batch8은 전날10만+재개20만의 유효 처리시간 합이며 야간 중단·모델 재적재·warmup은 제외한다. batch256은 한 세션의30만 누적시간이다. 두 조건의 시간 범위는 전처리cache 이후 GPU처리/반환/기록이며, 세션 분할에 따른 차이는 남는다. 메모리는 각 구간의 reset peak 중 최대이고 baseline 차이는 원본 행에서 확인한다. GPU 평균은 표본 수로 가중했다.

|batch|고유세포|유효초 합|세포/초|allocated peak GiB|reserved peak GiB|GPU 평균/최대%|세션 구간 수|
|---|---|---|---|---|---|---|---|
|8|300000|2663.19|112.65|0.389|0.404|94.89/96|2|
|256|300000|2628.60|114.13|5.477|21.475|95.10/96|1|

[전체30만세포 출력 검증](../results/actual_300k_resume_20261028_093701_696230/final_output_verification.json): 각 조건300000×512, 전체finite·단위norm·세포별batch간cosine 검사 및 전날10만출력의 정확한 보존 확인. 새로운 GPU실험은 이어서 시작하지 않는다.

## 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.
