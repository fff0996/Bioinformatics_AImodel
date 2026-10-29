# scGPT 기존 완료 결과와 입력 규모·GPU 사용률 보완

보완run `supplement_20261027_211638_936991` · 상태 **complete** · [원본summary](../results/supplement_20261027_211638_936991/summary.json) · [자원시계열](../results/supplement_20261027_211638_936991/resources.csv)

## 기존 기본 실험은 이미 완료

2026-10-23에입력검사·사전학습임베딩추론·1,000세포파인튜닝·저장/재로드·독립test·최대길이256/512/1024/1200비교를완료했다. 원본파일과학습모델을보존했고이번보완값을과거GPU-Util실측처럼소급해붙이지않는다.

[기존test](../results/finetuned_test_20261023_034345_99523/summary.json) · [기존길이비교](../results/length_compare_20261023_040313_501423/comparison.csv)

기존저장모델test4218세포중학습라벨이있는4213세포의Accuracy 0.9380,학습13클래스macroF1 0.5350,weightedF1 0.9312.test-onlyMHCclassII5세포는해당지표에서제외했다. 기존1epoch파인튜닝CUDApeak3.3741GiB,저장모델test추론peak0.3811GiB다.

## 동일 환경·모델·실제 입력

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

## 개수·길이별 GPU 사용률과 메모리

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

## 학습 GPU 사용률 보완

| 조건 | train세포 | epochs | batch | 초 | GPU평균/최대% | 모델적재GiB | 학습peakGiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 누락사용률계측보완 | 1000 | 1 | 8 | 24.1723 | 88.6887/95.0000 | 0.1913 | 3.3741 |

GPU사용률이없었던기존학습조건을계측하기위해동일1000세포/1epoch를별도run에서실행했다. 원본모델/결과는유지했고새모델은별도경로에저장했다. 원본02_finetune.py복사본에GPU-Util시계열과적재baseline만추가해학습알고리즘·split·optimizer를유지했다.

전체train10600에서동일seed42cell단위validation2120,별도train1000을골랐으며13유형정답으로encoder+classifier를갱신했다. 저장모델bundle /workspace/scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001.새보완모델은validation과재로드를검증했고독립test를새로평가한것은아니다. 위기존testAccuracy와새모델을혼동하지않는다.

학습sampler는0.2초대기+조회시간과시작/끝표본을포함하므로첫/끝nvidia-smi관측창이주변활동을포함할수있다. resource CSV는training_util/epoch_001/training_resources.csv다. GPU메모리는fit구간allocatorpeak,validation/저장은시간에서제외했다. optimizer/RNG/epoch저장,encoder/head변경,원본strict reload검증을수행했다. 독립프로세스validation재로드 {"independent_process": true, "validation_cells": 2120, "max_abs_difference": 0.0, "predictions_match": true, "passed": true}

## 해석과 한계

Spark는공유RAM121.688GiB이므로RSS와CUDA를더하지않는다. allocated는사용tensor, reserved는cache포함allocator예약이며장치전체점유율과다르다. rawcount유래·donor독립검증은확인되지않았다. 개수증가가peak를비례증가시킨다고가정하지않으며batch8이고정돼메모리증가는작았다. 메모리90%도달은별도승인된후속batch확대실험에서검토한다.

공식소스: [scGPT](https://github.com/bowang-lab/scGPT). 정확한설치리비전·원본모델hash와코드는각기존/보완run의provenance와environment에보존했다.
