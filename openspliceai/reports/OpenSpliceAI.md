# OpenSpliceAI — DGX Spark splice-site 추론·파인튜닝

> 최신 통합판: [기본·모든 후속 실험을 합친 최종 보고서](FINAL.md). 입력·추론·학습별 메모리와 80% 목표 상태는 최종 보고서 첫 표를 확인한다.

run `openspliceai_20261027_205554_995618` · 상태 **complete** · [summary](../results/openspliceai_20261027_205554_995618/RECORDS.md#file-e3051816aa6e) · 로그 (로컬 보관: `../results/openspliceai_20261027_205554_995618/benchmark.log`) · [GPU·메모리 시계열](../results/openspliceai_20261027_205554_995618/RECORDS.md#file-ee59b8737e84)

## 공식 모델과 환경

공식OpenSpliceAI0.0.7, source commit `369ec06283af59447195283a57d0384f4448ad03`. OSAI_MANE-10000nt의공식5개seed10–14 checkpoint를모두불러와순차ensemble했다. checkpoint경로·SHA-256은summary.checkpoint_sources에있다. TensorFlow/Keras원본SpliceAI와구분한다.

실제파라미터 {"per_model": 696739, "ensemble_total": 3483695, "trainable_single": 696739, "observed_gradient": 696739}

각모델32채널,16개residual unit,dilatedConv1d이며입력B×4×(출력길이+10000),출력B×3×출력길이다. 총context10000nt는양쪽5000nt씩제거되며출력3채널은non-splice/acceptor/donor확률이다. 입력크기비교에서context·가중치를고정하고중앙출력구간길이만변경했다.

환경 {"torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "gpu": "NVIDIA GB10"}

FP32,TF32=False,seed42,8CPUthreads. 공식predict/transfer소스의context10000설정은BATCH_SIZE=6*N_GPUS와N_GPUS=2상수여서실제batch12를그대로채택했다. 실험GPU는1개이며2개GPU를썼다는뜻이아니다. 파인튜닝은seed10단일모델의전파라미터,AdamW1e-4/weight_decay.01,CosineAnnealingWarmRestarts(T0=5,Tmult1,eta_min1e-5),10epochs다. 공식categorical_crossentropy_2d와같은확률log-loss를계산했다.

| 입력forward전 | allocatedGiB | reservedGiB | RSSGiB | 시스템가용GiB |
| --- | --- | --- | --- | --- |
| 5모델적재직후 | 0.0134 | 0.0176 | 0.7642 | 113.5768 |

이론FP32파라미터수×4byte와위실측은다르다. 적재 0.2750초,3배치warmup 1.2730초. CPU입력JSON은적재전에준비했다.

## 실제 서열·splice-site 정답

기존검증된GRCh38.primary_assembly참조와공식패키지의grch38 transcript/exon표를사용했다. 표의SHA-256과전처리정책은 [input_audit.json](../data/RECORDS.md#file-246beda1e980),실제서열·염기별라벨은 inputs.json (로컬 보관: `../data/inputs.json`)에있다. pretrained모델은MANE모델이지만이평가주석표가동일MANE원본이라고주장하지않는다. 주석기반정답이며새로운실험검증라벨은아니다.

| split | 구간수 | 허용염색체 | 원본입력nt | 보관라벨nt |
| --- | --- | --- | --- | --- |
| train | 96 | ['2', '4'] | 20000 | 10000 |
| validation | 24 | ['6'] | 20000 | 10000 |
| test | 36 | ['1', '3'] | 20000 | 10000 |

염색체단위로학습/검증/test를분리하고각구간20kb는같은염색체의다른선택구간과겹치지않게했다. 실제사용염색체는각inputs레코드에있으며위표는허용목록이다. 원래사전학습모델이인간유전체를봤으므로이번test는사전학습독립외부일반화시험이아니다.

plus strand donor는exon_end−1,acceptor는next_exon_start의exon염기좌표다. minus strand는역할을바꾸고서열·라벨을reverse-complement방향으로정렬했다. 같은염색체/strand의제공된모든주석site를라벨로합쳐인접유전자site누락을줄였다. 모호한좌표는−1로무시하며A/C/G/Tone-hot,N은모두0이다.

주석과좌표검증용실제motif:

| split | donor motif counts | acceptor motif counts |
| --- | --- | --- |
| train | {'GT': 286, 'CT': 1, 'AT': 2, 'GC': 1} | {'AG': 304, 'AC': 2} |
| validation | {'GT': 56, 'GC': 1} | {'AG': 63} |
| test | {'GT': 154, 'GC': 2, 'AT': 1} | {'AG': 149, 'AA': 1} |

각split의GTdonor/AGacceptor비율이75%를넘는지assert했고대부분canonicalmotif다. 입력한구간의정답행렬B×출력길이와모델출력B×3×출력길이일치를검사했다. 학습은원본20kb중중앙15kb입력→5kb출력,실제B×4×15000FP32와B×3×5000one-hot라벨을사용했다.

실제학습예 FAM110C_2_43532_- / - / 20kb중앞120nt:

```text
AGCGATCCTCCTGCCTCAGTCTCCCAAAGTGCTCAGATTAAGATATGAGCCATTGCACCCAGCCTAGGGTTTTGTCTCTAATTGCATTGAGTTCCTCTTTAAAAAAAATTTAAGACGATT
```

라벨수 {'0': 9998, '2': 1, '1': 1}

## 입력별 GPU 사용률·메모리

| 조건 | 입력nt | 출력nt | 고유구간 | 반복 | 초 | 구간/초 | GPU평균/최대% | 시작GiB | peakGiB | 증가GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ensemble_output5000_n12 | 15000 | 5000 | 12 | 48 | 15.2936 | 37.6627 | 94.1034/95.0000 | 0.0134 | 0.1248 | 0.1114 |
| ensemble_output5000_n24 | 15000 | 5000 | 24 | 24 | 15.3384 | 37.5527 | 93.9655/95.0000 | 0.0134 | 0.1248 | 0.1114 |
| ensemble_output5000_n36 | 15000 | 5000 | 36 | 16 | 15.3107 | 37.6207 | 94.6552/95.0000 | 0.0134 | 0.1248 | 0.1114 |
| ensemble_output1000_n12 | 11000 | 1000 | 12 | 70 | 15.0066 | 55.9754 | 94.6429/95.0000 | 0.0134 | 0.0943 | 0.0809 |
| ensemble_output10000_n12 | 20000 | 10000 | 12 | 35 | 15.4010 | 27.2710 | 94.4138/95.0000 | 0.0134 | 0.1661 | 0.1527 |
| training_epoch1 | 15000 | 5000 | 96 | 1 | 3.9867 | 24.0803 | 75.6250/95.0000 | 0.0027 | 2.1603 | 2.1576 |
| training_epoch2 | 15000 | 5000 | 96 | 1 | 3.7874 | 25.3469 | 86.0000/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch3 | 15000 | 5000 | 96 | 1 | 3.7764 | 25.4212 | 77.2857/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch4 | 15000 | 5000 | 96 | 1 | 3.8076 | 25.2130 | 82.3750/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch5 | 15000 | 5000 | 96 | 1 | 3.7894 | 25.3340 | 81.2857/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch6 | 15000 | 5000 | 96 | 1 | 3.8006 | 25.2589 | 83.5714/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch7 | 15000 | 5000 | 96 | 1 | 3.7831 | 25.3763 | 88.7143/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch8 | 15000 | 5000 | 96 | 1 | 3.7392 | 25.6740 | 86.4286/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch9 | 15000 | 5000 | 96 | 1 | 3.6669 | 26.1801 | 92.4286/95.0000 | 0.0106 | 2.1603 | 2.1496 |
| training_epoch10 | 15000 | 5000 | 96 | 1 | 3.6711 | 26.1505 | 94.0000/95.0000 | 0.0106 | 2.1603 | 2.1496 |

추론5조건최소15초반복이며개수12/24/36은같은15000nt입력중첩subset이다. 크기비교는같은12구간에서입력11000/15000/20000nt(출력1000/5000/10000)를비교한다.5모델은같은입력으로순차forward후확률평균한다. epoch행은단일모델학습으로ensemble추론과조건이다르다.

시간에는메모리에준비된서열의one-hot/라벨구성·GPU전송·모델연산과추론확률평균을포함한다. 추론출력은GPUtensor이며CPU반환·디스크준비/저장·모델적재·warmup은제외했다. 학습은forward/loss/backward/optimizer까지이며epoch별validation은제외했다. CUDA synchronize와조건별peak reset을적용하고증가량은같은run의실제시작allocated대비다. gradient·optimizer·중간값도포함한다.

GPU-Util은0.5초대기+조회전체가조건안인장치활동표본으로FLOPS/메모리%가아니다. epoch가짧으면표본수한계가있어단독평균대표성이낮다. 공유RAM121.688GiB에서allocated와RSS를더하지않는다. allocated/reserved/RSS/가용량시계열은sampled값이며장치전체물리점유량과구분한다.

## 주석 test·학습·저장·재로드

| 모델 | site | 양성수 | AUPRC(AP) | top-k recall |
| --- | --- | --- | --- | --- |
| 사전학습5ensemble | acceptor | 107 | 0.9121 | 0.8879 |
| 사전학습5ensemble | donor | 107 | 0.9537 | 0.9065 |
| 사전학습seed10 | acceptor | 107 | 0.9179 | 0.8879 |
| 사전학습seed10 | donor | 107 | 0.9410 | 0.8692 |
| 파인튜닝seed10 | acceptor | 107 | 0.9148 | 0.8785 |
| 파인튜닝seed10 | donor | 107 | 0.9472 | 0.8785 |

top-k는주석양성수k만큼가장높은score를뽑았을때정답비율이다. nonsplice가압도적으로많아일반accuracy보다donor/acceptorAUPRC를보고한다. 5모델ensemble과단일파인튜닝결과를동일조건향상으로주장하지않으며seed10전후를비교한다.

고정10epoch후test를평가했고test로epoch를선택하지않았다. 초기conv가중치최대변화 0.0032로실제갱신을확인했다. 저장 `models/openspliceai_20261027_205554_995618/model_seed10_finetuned.pt`,optimizer/scheduler/RNG/epoch와task.json도보관했다. 저장 0.0223초. 학습재개는검증하지않았다.

독립프로세스재로드 {"independent_process": true, "test_windows": 36, "input_shape": [12, 4, 15000], "max_abs_difference": 0.0, "all_base_classes_match": true, "passed": true}

## 한계·공식 출처

작은염색체별주석구간에대한동작·학습·계측검증이다. curatedannotation의완전성및사전학습중복/동족성이한계이며전체인간유전체평가를대체하지않는다. 이번과제는splice-site예측이며변이delta scoreCLI평가는수행하지않았다.

- [공식코드·모델](https://github.com/Kuanhao-Chao/OpenSpliceAI)
- [공식transfer예제](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/transfer/transfer_example.sh)
- [공식모델구조](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/openspliceai/train_base/openspliceai.py)
