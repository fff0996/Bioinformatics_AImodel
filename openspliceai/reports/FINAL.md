# OpenSpliceAI 최종 통합 보고서

기본 실험과 모든 후속 실험을 이 문서에 함께 수록했다. 다른 기본/후속 문서를 따로 읽을 필요 없이 아래 요약과 본문을 보면 된다.

## 1. 입력과 메모리 결론

DNA구간을4채널one-hot로변환 → 위치별splice site확률. 같은5개MANE10000 모델ensemble.

**목표는 메모리 할당80% 조건 확인**이다. 전체공유RAM121.688GiB의80%는약97.35GiB다. GPU-Util은별도연산활동지표이며이목표의성공판정에쓰지않는다.

|작업|관측메모리 peak GiB|공유RAM대비|해당 입력·배치·정밀도|80% 확인/제약|
|---|---|---|---|---|---|
|5모델 ensemble 추론|**1.463**|1.20%|실제15000nt 구간156개. GPU입력156×4×15000,출력156×3×5000,FP32. 5모델순차계산/평균.|80% 미확인; 확보한고유구간156개한계|
|단일 seed10 전체 파인튜닝|**2.160**|1.78%|실제train96/val24/test36구간. 12×4×15000 입력,정답12×5000 위치라벨,FP32·10epoch.|5모델을함께학습한값아님; 학습batch확대 미실시|

메모리는PyTorch의경우CUDA allocated peak다. reserved는예약량으로별도다. **AlphaFold2는컴파일·warmup등을포함한JAX프로세스누적peak**이므로CUDA구간peak와같은정의로비교하지않는다. RSS와CUDA를더하지않는다. 표는시험한조건의관측값이며모델의절대최대나모든모델의80%달성을의미하지않는다.

## 2. 기본 정보·입력 검사·추론·학습·평가

## OpenSpliceAI — DGX Spark splice-site 추론·파인튜닝


run `openspliceai_20261027_205554_995618` · 상태 **complete** · [summary](../results/openspliceai_20261027_205554_995618/summary.json) · 로그 (로컬 보관: `../results/openspliceai_20261027_205554_995618/benchmark.log`) · [GPU·메모리 시계열](../results/openspliceai_20261027_205554_995618/resources.csv)

### 공식 모델과 환경

공식OpenSpliceAI0.0.7, source commit `369ec06283af59447195283a57d0384f4448ad03`. OSAI_MANE-10000nt의공식5개seed10–14 checkpoint를모두불러와순차ensemble했다. checkpoint경로·SHA-256은summary.checkpoint_sources에있다. TensorFlow/Keras원본SpliceAI와구분한다.

실제파라미터 {"per_model": 696739, "ensemble_total": 3483695, "trainable_single": 696739, "observed_gradient": 696739}

각모델32채널,16개residual unit,dilatedConv1d이며입력B×4×(출력길이+10000),출력B×3×출력길이다. 총context10000nt는양쪽5000nt씩제거되며출력3채널은non-splice/acceptor/donor확률이다. 입력크기비교에서context·가중치를고정하고중앙출력구간길이만변경했다.

환경 {"torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "gpu": "NVIDIA GB10"}

FP32,TF32=False,seed42,8CPUthreads. 공식predict/transfer소스의context10000설정은BATCH_SIZE=6*N_GPUS와N_GPUS=2상수여서실제batch12를그대로채택했다. 실험GPU는1개이며2개GPU를썼다는뜻이아니다. 파인튜닝은seed10단일모델의전파라미터,AdamW1e-4/weight_decay.01,CosineAnnealingWarmRestarts(T0=5,Tmult1,eta_min1e-5),10epochs다. 공식categorical_crossentropy_2d와같은확률log-loss를계산했다.

| 입력forward전 | allocatedGiB | reservedGiB | RSSGiB | 시스템가용GiB |
| --- | --- | --- | --- | --- |
| 5모델적재직후 | 0.0134 | 0.0176 | 0.7642 | 113.5768 |

이론FP32파라미터수×4byte와위실측은다르다. 적재 0.2750초,3배치warmup 1.2730초. CPU입력JSON은적재전에준비했다.

### 실제 서열·splice-site 정답

기존검증된GRCh38.primary_assembly참조와공식패키지의grch38 transcript/exon표를사용했다. 표의SHA-256과전처리정책은 [input_audit.json](../data/input_audit.json),실제서열·염기별라벨은 inputs.json (로컬 보관: `../data/inputs.json`)에있다. pretrained모델은MANE모델이지만이평가주석표가동일MANE원본이라고주장하지않는다. 주석기반정답이며새로운실험검증라벨은아니다.

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

### 입력별 GPU 사용률·메모리

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

### 주석 test·학습·저장·재로드

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

### 한계·공식 출처

작은염색체별주석구간에대한동작·학습·계측검증이다. curatedannotation의완전성및사전학습중복/동족성이한계이며전체인간유전체평가를대체하지않는다. 이번과제는splice-site예측이며변이delta scoreCLI평가는수행하지않았다.

- [공식코드·모델](https://github.com/Kuanhao-Chao/OpenSpliceAI)
- [공식transfer예제](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/transfer/transfer_example.sh)
- [공식모델구조](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/openspliceai/train_base/openspliceai.py)

## 3. 후속 실험 통합 — 배치 확대 및 자원 검증

## openspliceai 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_144737_167765` · 종료사유 `available_unique_input_limit`

### 입력과 고정 조건

{
  "source": "actual GRCh38 curated regions; input15000nt -> output5000nt; original five-model MANE10000 ensemble",
  "unique_available_inputs": 156,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

### 배치별 측정

측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.

|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|791|791|15.0172|52.6729|0.0134|0.0227|0.0093|0.0352|0.0186|92.3103|96.0|1.0836|112.8657|
|2|378|756|15.0195|50.3345|0.0134|0.0319|0.0186|0.0352|0.0263|92.1071|94.0|1.0863|112.8673|
|4|179|716|15.0013|47.7292|0.0134|0.0505|0.0371|0.0762|0.0415|90.1429|96.0|1.0791|112.8994|
|8|83|664|15.0214|44.2037|0.0134|0.0876|0.0742|0.1133|0.072|93.3571|96.0|1.0862|112.8627|
|16|36|576|15.0106|38.3728|0.0134|0.1652|0.1518|0.1836|0.1358|91.9286|96.0|1.0865|112.8381|
|32|19|608|15.8191|38.4346|0.0134|0.3102|0.2968|0.3262|0.2549|94.9667|96.0|1.091|112.5204|
|64|10|640|16.6657|38.4022|0.0134|0.611|0.5976|0.625|0.5021|93.2258|96.0|1.0958|112.3472|
|96|7|672|17.3643|38.7|0.0134|0.905|0.8916|0.9355|0.7437|88.5758|96.0|1.0964|111.9866|
|128|5|640|16.6323|38.4793|0.0134|1.2021|1.1887|1.2441|0.9878|93.3226|96.0|1.1011|111.6641|
|156|4|624|16.3604|38.1408|0.0134|1.4628|1.4494|1.5|1.2021|95.6452|96.0|1.1071|111.4199|

### 결과 해석과 검증

최대시험성공batch 156. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://github.com/Kuanhao-Chao/OpenSpliceAI) · [기존 모델 보고서 디렉터리](.)

## 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.
