# OpenSpliceAI 최종 통합 보고서

## 1. 모델과 실제 입력

DNA구간을4채널one-hot로변환 → 위치별splice site확률. 같은5개MANE10000 모델ensemble.

실제파일·입력/정답예시·전처리와모델구조설명은이문서5절에전문을포함했다. 추론데이터와지도학습label을구분하며,세포/유전자·서열/토큰·이미지/타일의단위를섞지않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|배치 추론|고유1개; GRCh38 실제15000nt 구간,5모델ensemble|batch1, onehot:[1, 4, 15000],FP32|791회/총791건|15.02초|52.67입력/초|0.023 GiB / 0.02%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|배치 추론|고유156개; GRCh38 실제15000nt 구간,5모델ensemble|batch156, onehot:[156, 4, 15000],FP32|4회/총624건|16.36초|38.14입력/초|1.463 GiB / 1.20%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|단일seed10전체파인튜닝|train96/val24/test36;실제splice 위치라벨|12×4×15000,FP32|10epoch/80step|37.81초(학습구간합)|25.39구간/초|2.160 GiB / 1.78%|동일저장모델test지표본문;80%미도달|

## 4. 결과 해석

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### OpenSpliceAI — DGX Spark splice-site 추론·파인튜닝


run `openspliceai_20261027_205554_995618` · 상태 **complete** · [summary](../results/openspliceai_20261027_205554_995618/RECORDS.md#file-e3051816aa6e) · 로그 (로컬 보관: `../results/openspliceai_20261027_205554_995618/benchmark.log`) · [GPU·메모리 시계열](../results/openspliceai_20261027_205554_995618/RECORDS.md#file-ee59b8737e84)

#### 공식 모델과 환경

공식OpenSpliceAI0.0.7, source commit `369ec06283af59447195283a57d0384f4448ad03`. OSAI_MANE-10000nt의공식5개seed10–14 checkpoint를모두불러와순차ensemble했다. checkpoint경로·SHA-256은summary.checkpoint_sources에있다. TensorFlow/Keras원본SpliceAI와구분한다.

실제파라미터 {"per_model": 696739, "ensemble_total": 3483695, "trainable_single": 696739, "observed_gradient": 696739}

각모델32채널,16개residual unit,dilatedConv1d이며입력B×4×(출력길이+10000),출력B×3×출력길이다. 총context10000nt는양쪽5000nt씩제거되며출력3채널은non-splice/acceptor/donor확률이다. 입력크기비교에서context·가중치를고정하고중앙출력구간길이만변경했다.

환경 {"torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "gpu": "NVIDIA GB10"}

FP32,TF32=False,seed42,8CPUthreads. 공식predict/transfer소스의context10000설정은BATCH_SIZE=6*N_GPUS와N_GPUS=2상수여서실제batch12를그대로채택했다. 실험GPU는1개이며2개GPU를썼다는뜻이아니다. 파인튜닝은seed10단일모델의전파라미터,AdamW1e-4/weight_decay.01,CosineAnnealingWarmRestarts(T0=5,Tmult1,eta_min1e-5),10epochs다. 공식categorical_crossentropy_2d와같은확률log-loss를계산했다.

| 입력forward전 | allocatedGiB | reservedGiB | RSSGiB | 시스템가용GiB |
| --- | --- | --- | --- | --- |
| 5모델적재직후 | 0.0134 | 0.0176 | 0.7642 | 113.5768 |

이론FP32파라미터수×4byte와위실측은다르다. 적재 0.2750초,3배치warmup 1.2730초. CPU입력JSON은적재전에준비했다.

#### 실제 서열·splice-site 정답

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

#### 입력별 GPU 사용률·메모리

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

#### 주석 test·학습·저장·재로드

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

#### 한계·공식 출처

작은염색체별주석구간에대한동작·학습·계측검증이다. curatedannotation의완전성및사전학습중복/동족성이한계이며전체인간유전체평가를대체하지않는다. 이번과제는splice-site예측이며변이delta scoreCLI평가는수행하지않았다.

- [공식코드·모델](https://github.com/Kuanhao-Chao/OpenSpliceAI)
- [공식transfer예제](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/transfer/transfer_example.sh)
- [공식모델구조](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/openspliceai/train_base/openspliceai.py)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### openspliceai 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_144737_167765` · 종료사유 `available_unique_input_limit`

#### 입력과 고정 조건

{
  "source": "actual GRCh38 curated regions; input15000nt -> output5000nt; original five-model MANE10000 ensemble",
  "unique_available_inputs": 156,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

#### 배치별 측정

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

#### 결과 해석과 검증

최대시험성공batch 156. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://github.com/Kuanhao-Chao/OpenSpliceAI) · [기존 모델 보고서 디렉터리](.)

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
|`openspliceai/scripts/benchmark.py`|`35c95257b0d5eeb42fcd31019f1dc360e3b0c745535035074fa7e050f0107861`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`openspliceai/scripts/prepare.py`|`e38ffac7abb52de06b6439d9a614c9c02d1bbbead1255611528fad8b6f38327f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`openspliceai/scripts/reload.py`|`215a839e220cff1598db44d3078c65f2d5998947d281d809d851550184940401`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`openspliceai/scripts/report.py`|`468acc5b8012f133bbc1421c28ca7d6f003db0153bc44ff3838a33be75a3d8be`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`openspliceai/scripts/setup.py`|`34c5931018bf7c317205782f0379ef7592b9458706a69397443e186ede95ae48`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|
|`scripts/remaining_followups/launch.py`|`84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/report.py`|`8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/worker.py`|`e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|

<details>
<summary>openspliceai/scripts/benchmark.py — 원문 전문 (7,294 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/benchmark.py", "sha256": "35c95257b0d5eeb42fcd31019f1dc360e3b0c745535035074fa7e050f0107861", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,os,time,json,random,gc,hashlib
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import torch,numpy as np
from sklearn.metrics import average_precision_score
from openspliceai.train_base.openspliceai import SpliceAI
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'source/models/openspliceai-mane/10000nt';W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'checkpoint':'official OSAI_MANE10000 seeds10..14 five-model ensemble','batch':12,'basis':'official predict/transfer CL10000 code BATCH_SIZE=6*N_GPUS with hardcoded N_GPUS2; run uses single GPU with batch12, not two GPUs','dtype':'FP32','tf32':False,'seed':42,'context_nt':10000,'normal_output_nt':5000,'fine_tune_checkpoint':'seed10 only, compare single-seed10 baseline','fine_tune_epochs':10,'lr':1e-4,'optimizer':'AdamW default weight_decay.01; cosine restart T0=5,Tmult1,eta_min1e-5','train_n':96,'validation_n':24,'test_n':36,'fine_tune':'all parameters, curated packaged GRCh38 annotation'} }
def save():write_json(D/'summary.json',S)
def crop(r,sl):return r['sequence'][10000-(sl+10000)//2:10000+(sl+10000)//2],np.array(r['labels_10000'][5000-sl//2:5000+sl//2],dtype=np.int64)
def encode(rows,sl):
 seqs=[crop(r,sl)[0] for r in rows];tab=np.zeros((256,4),dtype=np.float32)
 for j,c in enumerate('ACGT'):tab[ord(c),j]=1
 x=np.stack([tab[np.frombuffer(seq.encode(),dtype=np.uint8)].T for seq in seqs]);y=np.stack([crop(r,sl)[1] for r in rows]);return torch.from_numpy(x).cuda(),y
try:
 data=json.loads((R/'data/inputs.json').read_text());meter.phase='loading';t=time.time();models=[];sources=[]
 for seed in range(10,15):
  p=P/f'model_10000nt_rs{seed}.pt';m=SpliceAI(32,W,AR).float().cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True));m.eval();models.append(m);sources.append({'checkpoint':str(p.relative_to(R)),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
 torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'per_model':sum(p.numel() for p in models[0].parameters()),'ensemble_total':sum(p.numel() for m in models for p in m.parameters()),'trainable_single':sum(p.numel() for p in models[0].parameters() if p.requires_grad)};S['checkpoint_sources']=sources;save()
 def predict(ms,rows,sl):
  x,y=encode(rows,sl);out=None
  for m in ms:
   z=m(x);out=z if out is None else out+z
  return out/len(ms),y
 meter.phase='warmup';t=time.time()
 with torch.no_grad():
  for _ in range(3):z,y=predict(models,data['test'][:12],5000);del z,y
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t
 for sl,n in [(5000,12),(5000,24),(5000,36),(1000,12),(10000,12)]:
  name=f'ensemble_output{sl}_n{n}';t,base=meter.begin(name);repeats=0
  with torch.no_grad():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,12):z,y=predict(models,data['test'][i:i+12],sl);assert z.shape==(min(12,n-i),3,sl);del z,y
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='ensemble_inference',bp=sl+10000,output_nt=sl,unique_inputs=n,repeats=repeats,processed=n*repeats);row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',name,row['seconds'],row['memory']['peak_allocated_gib'],flush=True);save()
 def score(ms,rows):
  arrays=[];labels=[]
  for m in ms:m.eval()
  with torch.no_grad():
   for i in range(0,len(rows),12):z,y=predict(ms,rows[i:i+12],5000);arrays.append(z.cpu().numpy());labels.append(y)
  z=np.concatenate(arrays);y=np.concatenate(labels);res={}
  for k,name in [(1,'acceptor'),(2,'donor')]:
   valid=y>=0;true=(y==k)[valid];probs=z[:,k,:][valid];num=int(true.sum());idx=np.argpartition(probs,-num)[-num:] if num else [];res[name]={'positive_sites':num,'auprc':float(average_precision_score(true,probs)),'top_k_recall':float(true[idx].mean()) if num else None}
  return res,z
 meter.phase='initial_test';S['ensemble_test_before'],_=score(models,data['test']);S['single_seed10_test_before'],_=score(models[:1],data['test']);del models,m;gc.collect();torch.cuda.empty_cache();meter.phase='classifier_loading';model=SpliceAI(32,W,AR).float().cuda();model.load_state_dict(torch.load(P/'model_10000nt_rs10.pt',map_location='cuda',weights_only=True));S['single_model_only']=meter.memory();before=model.initial_conv.weight.detach().cpu().clone();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=5,T_mult=1,eta_min=1e-5);history=[];grad=None
 for epoch in range(10):
  model.train();order=np.random.default_rng(42+epoch).permutation(96);t,base=meter.begin(f'training_epoch{epoch+1}');losses=[]
  for step in range(0,96,12):
   rows=[data['train'][int(i)] for i in order[step:step+12]];x,y=encode(rows,5000);yy=torch.from_numpy(y).cuda();valid=yy>=0;labels=torch.nn.functional.one_hot(yy.clamp_min(0),3).permute(0,2,1).float()*valid.unsqueeze(1);opt.zero_grad(set_to_none=True);z=model(x);loss=-(labels*torch.log(z+1e-10)).sum(1).mean();assert torch.isfinite(loss);loss.backward()
   if grad is None:grad=sum(p.numel() for p in model.parameters() if p.grad is not None)
   opt.step();losses.append(float(loss.detach()));del x,yy,labels,z,loss
  sched.step(epoch+1);row=meter.end(t,base,name=f'training_epoch{epoch+1}',mode='single_model_training',bp=15000,output_nt=5000,unique_inputs=96,repeats=1,processed=96,steps=8,loss_mean=float(np.mean(losses)));row['throughput']=96/row['seconds'];S['conditions'].append(row);meter.phase='validation';val,_=score([model],data['validation']);history.append({'epoch':epoch+1,'loss':row['loss_mean'],'validation':val});print('EPOCH',epoch+1,row['loss_mean'],val,flush=True);S['epochs']=history;save()
 S['parameters']['observed_gradient']=grad;S['weight_change_max_abs']=float((model.initial_conv.weight.detach().cpu()-before).abs().max());assert S['weight_change_max_abs']>0;meter.phase='saving';t=time.time();model.eval();torch.save(model.state_dict(),M/'model_seed10_finetuned.pt');torch.save({'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':10,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},M/'training_state.pt');write_json(M/'task.json',{'L':32,'W':W.tolist(),'AR':AR.tolist(),'context':10000,'output':5000,'input_channels':['A','C','G','T'],'labels':['nonsplice','acceptor','donor'],'annotation':'source/openspliceai/variant/annotations/grch38.txt','resume_tested':False});S['save_seconds']=time.time()-t;S['saved_model']=str((M/'model_seed10_finetuned.pt').relative_to(R));meter.phase='final_test';S['finetuned_test'],z=score([model],data['test']);np.save(D/'test_probabilities.npy',z);write_json(D/'test_window_ids.json',[r['id'] for r in data['test']]);S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>openspliceai/scripts/prepare.py — 원문 전문 (3,950 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/prepare.py", "sha256": "e38ffac7abb52de06b6439d9a614c9c02d1bbbead1255611528fad8b6f38327f", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import csv,json,hashlib,bisect,collections
import pysam,numpy as np
R=Path('/work');fa=pysam.FastaFile('/reference/GRCh38.primary_assembly.genome.fa');annotation=R/'source/openspliceai/variant/annotations/grch38.txt';rows=list(csv.DictReader(annotation.open(),delimiter='\t'));sites=collections.defaultdict(lambda:{1:set(),2:set()})
for row in rows:
 starts=[int(v) for v in row['EXON_START'].split(',') if v];ends=[int(v) for v in row['EXON_END'].split(',') if v];assert len(starts)==len(ends)
 if len(starts)<2:continue
 key=(row['CHROM'],row['STRAND']);donors=[e-1 for e in ends[:-1]];acceptors=starts[1:]
 if row['STRAND']=='-':donors,acceptors=acceptors,donors
 sites[key][2].update(donors);sites[key][1].update(acceptors)
sites={k:{t:sorted(v) for t,v in x.items()} for k,x in sites.items()};out={};audit={'reference':'GRCh38.primary_assembly.genome.fa (existing BEND reference)','annotation':str(annotation.relative_to(R)),'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),'coordinate_policy':'genomic0based; plus donor exon_end-1, acceptor next_exon_start; minus swapped then reverse-complement; labeled exon bases per official create-data motif convention','source_checkpoint':'official OSAI_MANE10000 ensemble seeds10..14; annotation here is bundled grch38 transcript table, not asserted identical MANE training annotation','splits':{}}
for split,chroms,n in [('train',['2','4'],96),('validation',['6'],24),('test',['1','3'],36)]:
 selected=[];lastend={};seen=set()
 for row in rows:
  chrom=row['CHROM'];strand=row['STRAND'];key=(chrom,strand)
  if chrom not in chroms or key not in sites or row['#NAME'] in seen:continue
  starts=[int(v) for v in row['EXON_START'].split(',') if v];ends=[int(v) for v in row['EXON_END'].split(',') if v]
  if len(starts)<2:continue
  center=(ends[0]-1+starts[1])//2;start=center-10000;end=center+10000;ctg='chr'+chrom
  if start<0 or end>fa.get_reference_length(ctg) or start<lastend.get(ctg,-1):continue
  seq=fa.fetch(ctg,start,end).upper()
  if not set(seq)<=set('ACGTN') or seq.count('N')>100:continue
  labels=np.zeros(10000,dtype=np.int64);out_start=center-5000;out_end=center+5000
  for kind,coords in sites[key].items():
   for pos in coords[bisect.bisect_left(coords,out_start):bisect.bisect_left(coords,out_end)]:
    idx=pos-out_start if strand=='+' else out_end-1-pos
    if labels[idx] not in [0,kind]:labels[idx]=-1
    else:labels[idx]=kind
  if not np.any(labels==1) or not np.any(labels==2):continue
  if strand=='-':seq=seq.translate(str.maketrans('ACGTN','TGCAN'))[::-1]
  rec={'id':row['#NAME']+'_'+chrom+'_'+str(center)+'_'+strand,'gene':row['#NAME'],'chromosome':ctg,'strand':strand,'genomic_input_start':start,'genomic_input_end':end,'sequence':seq,'labels_10000':labels.tolist(),'label_counts':dict(collections.Counter(map(int,labels)))};selected.append(rec);seen.add(row['#NAME']);lastend[ctg]=end
  if len(selected)==n:break
 assert len(selected)==n,(split,len(selected));out[split]=selected;motifs=collections.defaultdict(collections.Counter)
 for r in selected:
  seq=r['sequence'][5000:-5000]
  for i,v in enumerate(r['labels_10000']):
   if v==2:motifs['donor'][seq[i+1:i+3]]+=1
   if v==1:motifs['acceptor'][seq[max(0,i-2):i]]+=1
 audit['splits'][split]={'n':len(selected),'chromosomes':chroms,'input_nt':20000,'output_label_nt':10000,'motifs':{k:dict(v) for k,v in motifs.items()},'first_example':{k:v for k,v in selected[0].items() if k not in ['sequence','labels_10000']}}
 for kind,canonical in [('donor','GT'),('acceptor','AG')]:assert motifs[kind][canonical]/sum(motifs[kind].values())>.75,(split,kind,motifs[kind])
allrows=[r for rs in out.values() for r in rs];assert len({hashlib.sha256(r['sequence'].encode()).hexdigest() for r in allrows})==len(allrows);(R/'data/inputs.json').write_text(json.dumps(out));(R/'data/input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit['splits'],indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>openspliceai/scripts/reload.py — 원문 전문 (1,449 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/reload.py", "sha256": "215a839e220cff1598db44d3078c65f2d5998947d281d809d851550184940401", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,json
sys.path.insert(0,'/support')
from gpu_support import write_json
import numpy as np,torch
from openspliceai.train_base.openspliceai import SpliceAI
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);m=SpliceAI(32,W,AR).float().cuda().eval();m.load_state_dict(torch.load(R/s['saved_model'],map_location='cuda',weights_only=True));data=json.loads((R/'data/inputs.json').read_text())['test'];tab=np.zeros((256,4),np.float32)
for j,c in enumerate('ACGT'):tab[ord(c),j]=1
pred=[]
with torch.no_grad():
 for i in range(0,len(data),12):
  x=np.stack([tab[np.frombuffer(r['sequence'][2500:17500].encode(),np.uint8)].T for r in data[i:i+12]]);pred.append(m(torch.from_numpy(x).cuda()).cpu().numpy())
y=np.concatenate(pred);ref=np.load(D/'test_probabilities.npy');r={'independent_process':True,'test_windows':len(data),'input_shape':[12,4,15000],'max_abs_difference':float(np.abs(y-ref).max()),'all_base_classes_match':bool(np.array_equal(y.argmax(1),ref.argmax(1)))};r['passed']=r['max_abs_difference']<=1e-5 and r['all_base_classes_match'];write_json(D/'reload.json',r);assert r['passed'];s['reload']=r;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(r),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>openspliceai/scripts/report.py — 원문 전문 (8,518 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/report.py", "sha256": "468acc5b8012f133bbc1421c28ca7d6f003db0153bc44ff3838a33be75a3d8be", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import json,csv
R=Path('/home/hyein/ai-benchmark/openspliceai');run=(R/'results/LATEST').read_text().strip();D=R/'results'/run;s=json.loads((D/'summary.json').read_text());audit=json.loads((R/'data/input_audit.json').read_text());out=[]
def a(t=''):out.append(t)
def table(h,rows):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rows:a('| '+' | '.join(map(str,r))+' |')
 a()
def f(v):return 'N/A' if v is None else f'{v:.4f}'
a('# OpenSpliceAI — DGX Spark splice-site 추론·파인튜닝');a();a(f'run `{run}` · 상태 **{s["status"]}** · [summary](../results/{run}/summary.json) · [로그](../results/{run}/benchmark.log) · [GPU·메모리 시계열](../results/{run}/resources.csv)');a();a('## 공식 모델과 환경');a();a('공식OpenSpliceAI0.0.7, source commit `369ec06283af59447195283a57d0384f4448ad03`. OSAI_MANE-10000nt의공식5개seed10–14 checkpoint를모두불러와순차ensemble했다. checkpoint경로·SHA-256은summary.checkpoint_sources에있다. TensorFlow/Keras원본SpliceAI와구분한다.');a();a('실제파라미터 '+json.dumps(s.get('parameters',{})));a();a('각모델32채널,16개residual unit,dilatedConv1d이며입력B×4×(출력길이+10000),출력B×3×출력길이다. 총context10000nt는양쪽5000nt씩제거되며출력3채널은non-splice/acceptor/donor확률이다. 입력크기비교에서context·가중치를고정하고중앙출력구간길이만변경했다.');a();a('환경 '+json.dumps(s['environment']));a();a('FP32,TF32=False,seed42,8CPUthreads. 공식predict/transfer소스의context10000설정은BATCH_SIZE=6*N_GPUS와N_GPUS=2상수여서실제batch12를그대로채택했다. 실험GPU는1개이며2개GPU를썼다는뜻이아니다. 파인튜닝은seed10단일모델의전파라미터,AdamW1e-4/weight_decay.01,CosineAnnealingWarmRestarts(T0=5,Tmult1,eta_min1e-5),10epochs다. 공식categorical_crossentropy_2d와같은확률log-loss를계산했다.');a();b=s['model_only_postload'];table(['입력forward전','allocatedGiB','reservedGiB','RSSGiB','시스템가용GiB'],[['5모델적재직후',*[f(b[k]) for k in ['allocated_gib','reserved_gib','rss_gib','system_available_gib']]]]);a('이론FP32파라미터수×4byte와위실측은다르다. 적재 '+f(s['load_seconds'])+'초,3배치warmup '+f(s.get('warmup_seconds'))+'초. CPU입력JSON은적재전에준비했다.');a();a('## 실제 서열·splice-site 정답');a();a('기존검증된GRCh38.primary_assembly참조와공식패키지의grch38 transcript/exon표를사용했다. 표의SHA-256과전처리정책은 [input_audit.json](../data/input_audit.json),실제서열·염기별라벨은 [inputs.json](../data/inputs.json)에있다. pretrained모델은MANE모델이지만이평가주석표가동일MANE원본이라고주장하지않는다. 주석기반정답이며새로운실험검증라벨은아니다.');a();table(['split','구간수','허용염색체','원본입력nt','보관라벨nt'],[[k,v['n'],v['chromosomes'],v['input_nt'],v['output_label_nt']] for k,v in audit['splits'].items()]);a('염색체단위로학습/검증/test를분리하고각구간20kb는같은염색체의다른선택구간과겹치지않게했다. 실제사용염색체는각inputs레코드에있으며위표는허용목록이다. 원래사전학습모델이인간유전체를봤으므로이번test는사전학습독립외부일반화시험이아니다.');a();a('plus strand donor는exon_end−1,acceptor는next_exon_start의exon염기좌표다. minus strand는역할을바꾸고서열·라벨을reverse-complement방향으로정렬했다. 같은염색체/strand의제공된모든주석site를라벨로합쳐인접유전자site누락을줄였다. 모호한좌표는−1로무시하며A/C/G/Tone-hot,N은모두0이다.');a();a('주석과좌표검증용실제motif:');a();table(['split','donor motif counts','acceptor motif counts'],[[k,v['motifs']['donor'],v['motifs']['acceptor']] for k,v in audit['splits'].items()]);a('각split의GTdonor/AGacceptor비율이75%를넘는지assert했고대부분canonicalmotif다. 입력한구간의정답행렬B×출력길이와모델출력B×3×출력길이일치를검사했다. 학습은원본20kb중중앙15kb입력→5kb출력,실제B×4×15000FP32와B×3×5000one-hot라벨을사용했다.');a();ex=json.loads((R/'data/inputs.json').read_text())['train'][0];a('실제학습예 '+ex['id']+' / '+ex['strand']+' / 20kb중앞120nt:');a();a('```text\n'+ex['sequence'][:120]+'\n```');a();a('라벨수 '+str(ex['label_counts']));a();a('## 입력별 GPU 사용률·메모리');a();rows=[];flat=[]
for r in s['conditions']:
 rows.append([r['name'],r['bp'],r['output_nt'],r['unique_inputs'],r['repeats'],f(r['seconds']),f(r['throughput']),f(r['gpu_util_avg'])+'/'+f(r['gpu_util_max']),f(r['baseline']['allocated_gib']),f(r['memory']['peak_allocated_gib']),f(r['peak_delta_gib'])]);d={k:v for k,v in r.items() if k not in ['baseline','memory']};d.update(baseline_allocated_gib=r['baseline']['allocated_gib'],peak_allocated_gib=r['memory']['peak_allocated_gib'],peak_reserved_gib=r['memory']['peak_reserved_gib']);flat.append(d)
table(['조건','입력nt','출력nt','고유구간','반복','초','구간/초','GPU평균/최대%','시작GiB','peakGiB','증가GiB'],rows)
if flat:
 keys=list(dict.fromkeys(k for r in flat for k in r))
 with (R/'reports/comparison.csv').open('w',newline='') as f0:w=csv.DictWriter(f0,fieldnames=keys);w.writeheader();w.writerows(flat)
a('추론5조건최소15초반복이며개수12/24/36은같은15000nt입력중첩subset이다. 크기비교는같은12구간에서입력11000/15000/20000nt(출력1000/5000/10000)를비교한다.5모델은같은입력으로순차forward후확률평균한다. epoch행은단일모델학습으로ensemble추론과조건이다르다.');a();a('시간에는메모리에준비된서열의one-hot/라벨구성·GPU전송·모델연산과추론확률평균을포함한다. 추론출력은GPUtensor이며CPU반환·디스크준비/저장·모델적재·warmup은제외했다. 학습은forward/loss/backward/optimizer까지이며epoch별validation은제외했다. CUDA synchronize와조건별peak reset을적용하고증가량은같은run의실제시작allocated대비다. gradient·optimizer·중간값도포함한다.');a();a('GPU-Util은0.5초대기+조회전체가조건안인장치활동표본으로FLOPS/메모리%가아니다. epoch가짧으면표본수한계가있어단독평균대표성이낮다. 공유RAM121.688GiB에서allocated와RSS를더하지않는다. allocated/reserved/RSS/가용량시계열은sampled값이며장치전체물리점유량과구분한다.');a()
if 'finetuned_test' in s:
 a('## 주석 test·학습·저장·재로드');a();rows=[]
 for name,key in [('사전학습5ensemble','ensemble_test_before'),('사전학습seed10','single_seed10_test_before'),('파인튜닝seed10','finetuned_test')]:
  for kind,v in s[key].items():rows.append([name,kind,v['positive_sites'],f(v['auprc']),f(v['top_k_recall'])])
 table(['모델','site','양성수','AUPRC(AP)','top-k recall'],rows);a('top-k는주석양성수k만큼가장높은score를뽑았을때정답비율이다. nonsplice가압도적으로많아일반accuracy보다donor/acceptorAUPRC를보고한다. 5모델ensemble과단일파인튜닝결과를동일조건향상으로주장하지않으며seed10전후를비교한다.');a();a('고정10epoch후test를평가했고test로epoch를선택하지않았다. 초기conv가중치최대변화 '+f(s['weight_change_max_abs'])+'로실제갱신을확인했다. 저장 `'+s['saved_model']+'`,optimizer/scheduler/RNG/epoch와task.json도보관했다. 저장 '+f(s['save_seconds'])+'초. 학습재개는검증하지않았다.');a();a('독립프로세스재로드 '+json.dumps(s.get('reload',{'status':'pending'})));a()
a('## 한계·공식 출처');a();a('작은염색체별주석구간에대한동작·학습·계측검증이다. curatedannotation의완전성및사전학습중복/동족성이한계이며전체인간유전체평가를대체하지않는다. 이번과제는splice-site예측이며변이delta scoreCLI평가는수행하지않았다.');a();a('- [공식코드·모델](https://github.com/Kuanhao-Chao/OpenSpliceAI)\n- [공식transfer예제](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/transfer/transfer_example.sh)\n- [공식모델구조](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/openspliceai/train_base/openspliceai.py)');(R/'reports/OpenSpliceAI.md').write_text('\n'.join(out)+'\n');print(R/'reports/OpenSpliceAI.md')
````
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>openspliceai/scripts/setup.py — 원문 전문 (534 bytes)</summary>


<!-- RESTORE_FILE {"path": "openspliceai/scripts/setup.py", "sha256": "34c5931018bf7c317205782f0379ef7592b9458706a69397443e186ede95ae48", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,subprocess,importlib.metadata,json
from pathlib import Path
R=Path('/work');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True);subprocess.run(['/work/.venv/bin/pip','install','--no-deps','/work/source'],check=True);(R/'constraints.txt').write_text('torch=='+importlib.metadata.version('torch')+'\n');subprocess.run(['/work/.venv/bin/pip','install','-c','/work/constraints.txt','--cache-dir','/work/.cache/pip','scikit-learn','pysam','gffutils','pyfaidx','biopython'],check=True)
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

<details>
<summary>scripts/remaining_followups/launch.py — 원문 전문 (7,883 bytes)</summary>


<!-- RESTORE_FILE {"path": "scripts/remaining_followups/launch.py", "sha256": "84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,subprocess,time,datetime,sys,os,shlex,csv,hashlib
import psutil
ROOT=Path('/home/hyein/ai-benchmark');plan=['scvi','dnabert2','ntv2','rnafm','openspliceai'];stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');Q={'status':'running','queue_id':stamp,'order':plan,'models':{},'scope':'inference batch expansion, same existing weights; sequential GPU only','applicability_notes':{'AlphaFold2':'ColabFold query list is processed sequentially; no independent-query GPU batch axis exposed in existing official path. Complex chain count changes biological input, not batch. No artificial concurrent GPU jobs or replacement model. Existing baseline retained; dedicated batch sweep not performed.','Clair3':'Existing official calling pipeline controls batch internally (predictBatchSize and GPU multiplier), no explicit CLI minibatch argument in installed path. Do not confuse candidate count, threads/workers and batch. Dedicated source-level candidate-cache batch sweep not performed; existing measured GPU batch1000 and coverage/region results retained.'}}
qp=ROOT/'FOLLOWUP_QUEUE.json'
def queue_save():
 qp.write_text(json.dumps(Q,indent=2));lines=['# 후속 배치 실험 진행','',Q['scope'],'','|모델|상태|최대 성공batch|보고서|','|---|---|---|---|']
 for model in plan:
  s=Q['models'].get(model,{});lines.append('|'+model+'|'+s.get('status','queued')+'|'+str(s.get('largest_successful_batch',''))+'|['+model+']('+model+'/reports/Batch_memory_followup.md)|')
 lines+=['','## 배치 확대 적용 범위','']+[k+': '+v for k,v in Q['applicability_notes'].items()];(ROOT/'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines)+'\n')
queue_save()
for model in plan:
 R=ROOT/model;run='batch_followup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/BATCH_LATEST').write_text(run+'\n');records=[]
 if model=='dnabert2':records=list(csv.DictReader(Path('/home/hyein/dnabert2-benchmark/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv').open()));source='existing BEND prepared n1000, fixed4000bp, actual native tokenizer length verified worker'
 elif model=='ntv2':
  original=json.loads((R/'data/inputs.json').read_text())['long_inference'];records=[{**r,'sequence':r['sequence'][(len(r['sequence'])-6000)//2:(len(r['sequence'])-6000)//2+6000]} for r in original];source='existing real GRCh38 regions, central6000nt crops -> native ~1001tokens'
 elif model=='rnafm':
  original=json.loads((R/'data/inputs.json').read_text());records=[{**r,'sequence':r['sequence'][:512]} for split in ['train','validation','test'] for r in original[split] if len(r['sequence'])>=512];source='official tutorial RNAcentral actual unique sequences >=512nt, first512nt explicit crop; pool limit retained'
 elif model=='openspliceai':
  original=json.loads((R/'data/inputs.json').read_text());records=[{**r,'sequence':r['sequence'][2500:17500]} for split in ['train','validation','test'] for r in original[split]];source='actual GRCh38 curated regions; input15000nt -> output5000nt; original five-model MANE10000 ensemble'
 else:source='saved full10176cell SCVI model and its raw-count AnnData, fixed2000genes; latent posterior mean'
 if records:
  seen=set();unique=[]
  for r in records:
   assert r['sequence'] and set(r['sequence'])<=set('ACGTUNRYKMSWBDHVX')
   if r['sequence'] not in seen:unique.append(r);seen.add(r['sequence'])
  records=unique;assert len({r['id'] for r in records})==len(records);(D/'inputs.json').write_text(json.dumps(records,indent=2));pool=len(records)
 else:pool=10176
 S={'status':'running','model':model,'run_id':run,'conditions':[],'failed':[],'input_audit':{'source':source,'unique_available_inputs':pool,'selection':'nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained','token_policy':'fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers'},'largest_successful_batch':None};swap0=psutil.swap_memory().used
 def save():(D/'summary.json').write_text(json.dumps(S,indent=2));Q['models'][model]={'status':S['status'],'run_id':run,'largest_successful_batch':S.get('largest_successful_batch')};queue_save()
 save();print('MODEL_START',model,run,pool,flush=True)
 sizes=[1,2,4,8,16,32,64,96,128,192,256,384,512,768,1024,2048,4096,8192] if model!='scvi' else [128,256,512,1024,2048,4096,8192]
 sizes=sorted(set([b for b in sizes if b<=pool]+[pool]))
 for batch in sizes:
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if active:S['stop_reason']='other_GPU_job_present';S['status']='waiting_external_gpu';save();raise RuntimeError('Do not overlap existing GPU job '+active)
  available=psutil.virtual_memory().available/2**30
  if available<16 or psutil.swap_memory().used>swap0+2*2**30:S['stop_reason']='system_memory_guard';break
  if S['conditions']:
   last=S['conditions'][-1];base=last['model_only_allocated_gib'];estimate=base+(last['memory']['peak_reserved_gib']-base)*batch/last['batch_size']*1.25+3
   if estimate>available-16:S['stop_reason']='predicted_reservation_exceeds_available_budget';S['skipped_next_batch']={'batch':batch,'estimated_reserved_gib':estimate,'available_gib':available};break
  name='followup-'+model+'-'+run.lower()+'-'+str(batch);cmd=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--ipc=host','--user','1000:1000','-v',str(R)+':/work','-v',str(ROOT/'scripts')+':/support:ro','-e','RUN_ID='+run,'-e','MODEL_KIND='+model,'-e','BATCH='+str(batch),'-e','MPLCONFIGDIR=/work/.cache/matplotlib','-e','NUMBA_CACHE_DIR=/work/.cache/numba','-e','HF_HOME=/work/.cache/huggingface','-e','HF_MODULES_CACHE=/work/.cache/modules','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1']
  python='/work/.venv/bin/python'
  if model=='dnabert2':cmd+=['-v','/home/hyein/dnabert2-benchmark:/workspace:ro'];python='/workspace/miniforge/envs/dnabert2/bin/python'
  cmd+=['nvcr.io/nvidia/pytorch:26.06-py3',python,'-u','/support/remaining_followups/worker.py'];(D/f'batch{batch}_command.json').write_text(json.dumps(cmd,indent=2));start=time.time();reason=None
  with (D/f'batch{batch}.log').open('w') as f:
   p=subprocess.Popen(['sg','docker','-c',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
   while p.poll() is None:
    if psutil.virtual_memory().available<16*2**30 or psutil.swap_memory().used>swap0+2*2**30:reason='parent_memory_guard';subprocess.run(['sg','docker','-c',shlex.join(['docker','stop','-t','2',name])],stdout=f,stderr=subprocess.STDOUT);break
    time.sleep(.5)
   code=p.wait()
  (D/f'batch{batch}_exit.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.time()-start,'reason':reason}));print('BATCH_EXIT',model,batch,code,flush=True)
  if code:S['failed'].append({'batch':batch,'exit_code':code,'reason':reason or 'worker_error_see_log'});S['stop_reason']='worker_stopped_see_log';break
  worker=json.loads((D/f'batch_{batch}/summary.json').read_text());assert worker['status']=='complete';S['conditions'].append(worker['conditions'][0]);S['largest_successful_batch']=batch;save()
 else:S['stop_reason']='available_unique_input_limit'
 S['status']='complete' if not S['failed'] else 'stopped';S['reached80']=any(c['allocated_fraction_shared_ram']>=.8 for c in S['conditions']);S['reached90']=any(c['allocated_fraction_shared_ram']>=.9 for c in S['conditions']);save();subprocess.run([sys.executable,str(ROOT/'scripts/remaining_followups/report.py'),model,str(D)],check=True);print('MODEL_FINISH',model,S['status'],S.get('stop_reason'),flush=True)
Q['status']='finished_with_unresolved_items' if any(s['status']!='complete' for s in Q['models'].values()) else 'complete_for_supported_sweeps';queue_save();print('QUEUE_FINISHED',Q['status'],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/remaining_followups/report.py — 원문 전문 (5,292 bytes)</summary>


<!-- RESTORE_FILE {"path": "scripts/remaining_followups/report.py", "sha256": "8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,sys,csv
ROOT=Path('/home/hyein/ai-benchmark');model=sys.argv[1];D=Path(sys.argv[2]);S=json.loads((D/'summary.json').read_text());R=ROOT/model;lines=[f'# {model} 실제 입력의 배치별 추론 자원','',f"상태 **{S['status']}** · run `{D.name}` · 종료사유 `{S.get('stop_reason','running')}`",'','## 입력과 고정 조건','',json.dumps(S['input_audit'],ensure_ascii=False,indent=2),'','배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.','','FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.','','## 배치별 측정','','측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.','', '|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|','|---|---|---|---|---|---|---|---|---|---|---|---|---|'];flat=[]
for c in S['conditions']:
 m=c['memory'];b=c['model_only_allocated_gib'];r={'batch':c['batch_size'],'repeats':c['repeats'],'processed_inputs':c['processed_inputs'],'seconds':c['seconds'],'inputs_per_second':c['inputs_per_second'],'loaded_allocated_gib':b,'allocated_peak_gib':m['peak_allocated_gib'],'additional_from_same_run_load_gib':m['peak_allocated_gib']-b,'reserved_peak_gib':m['peak_reserved_gib'],'percent_shared_ram':100*c['allocated_fraction_shared_ram'],'gpu_mean':c['gpu_util_avg'],'gpu_max':c['gpu_util_max'],'rss_sampled_peak_gib':c['rss_sampled_peak_gib'],'available_sampled_min_gib':c['available_sampled_min_gib']};flat.append(r);lines.append('|'+ '|'.join(str(round(v,4)) if isinstance(v,float) else str(v) for v in r.values())+'|')
lines+=['','## 결과 해석과 검증','',f"최대시험성공batch {S.get('largest_successful_batch')}. allocated80%도달 {S.get('reached80')},90%도달 {S.get('reached90')}. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.",'','출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.','','GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.','','원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.']
sources={'scvi':'https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html','dnabert2':'https://github.com/MAGICS-LAB/DNABERT_2','ntv2':'https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species','rnafm':'https://github.com/ml4bio/RNA-FM','openspliceai':'https://github.com/Kuanhao-Chao/OpenSpliceAI'}
lines+=['','[공식 출처]('+sources[model]+') · [기존 모델 보고서 디렉터리](.)']
(R/'reports/Batch_memory_followup.md').write_text('\n'.join(lines)+'\n')
if flat:
 with (R/'reports/batch_memory_followup.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for name in ['STATUS.md','RESULTS.md','MEMORY_RESULTS.md']:
 with (ROOT/name).open('a') as f:f.write('\n'+model+' 배치후속 '+S['status']+', 최대성공batch '+str(S.get('largest_successful_batch'))+'. [보고서]('+model+'/reports/Batch_memory_followup.md). 종료사유 '+S.get('stop_reason','running')+'.\n')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/remaining_followups/worker.py — 원문 전문 (6,920 bytes)</summary>


<!-- RESTORE_FILE {"path": "scripts/remaining_followups/worker.py", "sha256": "e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,time,json,gc,argparse
import numpy as np,torch
sys.path.insert(0,'/support');from gpu_support import Meter,write_json
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];kind=os.environ['MODEL_KIND'];B=int(os.environ['BATCH']);C=D/f'batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'batch':B,'task':'inference resource batch sweep only, no training','conditions':[]}
def save():write_json(C/'summary.json',S)
try:
 torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='load';t=time.time();tokenizer=None
 if kind=='scvi':
  import scvi
  P=R/'models/scvi_20261027_185327_767988/train_10176';m=scvi.model.SCVI.load(str(P),accelerator='cpu');m.to_device('cuda:0');m.module.eval();nets=[m.module];ad=m.adata;assert ad.n_vars==2000 and ad.n_obs>=B;S['checkpoint']=str(P);S['input']={'unique_cells':B,'genes':ad.n_vars,'raw_counts_source':'saved adata.h5ad layers counts, actual 10x PBMC4k+8k','original_pool_cells':ad.n_obs,'batch_covariate':'sample_id, biological batch distinct from GPU batch'}
  x=ad[:B].copy()
  def forward():return m.get_latent_representation(adata=x,batch_size=B,give_mean=True)
 else:
  records=json.loads((D/'inputs.json').read_text());assert len(records)>=B
  if kind=='dnabert2':
   from transformers import AutoModel,AutoTokenizer,BertConfig
   P=Path('/workspace/dnabert2_eval/model');tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);m=AutoModel.from_pretrained(P,config=BertConfig.from_pretrained(P),trust_remote_code=True,local_files_only=True).float().cuda().eval();nets=[m]
  elif kind=='ntv2':
   from transformers import AutoModelForMaskedLM,AutoTokenizer
   P=R/'models/checkpoint';tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);m=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True).float().cuda().eval();nets=[m]
  elif kind=='rnafm':
   import fm
   torch.serialization.add_safe_globals([argparse.Namespace]);m,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');m=m.float().cuda().eval();nets=[m];converter=alphabet.get_batch_converter()
  elif kind=='openspliceai':
   from openspliceai.train_base.openspliceai import SpliceAI
   W=np.array([11]*8+[21]*4+[41]*4);AR=np.array([1]*4+[4]*4+[10]*4+[25]*4);nets=[]
   for seed in range(10,15):
    net=SpliceAI(32,W,AR).float().cuda();net.load_state_dict(torch.load(R/f'source/models/openspliceai-mane/10000nt/model_10000nt_rs{seed}.pt',map_location='cpu',weights_only=True));net.eval();nets.append(net)
  else:raise ValueError(kind)
 torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters_total']=sum(p.numel() for net in nets for p in net.parameters());S['parameters_trained_in_this_run']=0;S['theoretical_weight_storage_gib']=sum(p.numel()*p.element_size() for net in nets for p in net.parameters())/2**30;S['parameters_used_backbone']=sum(p.numel() for p in m.esm.parameters()) if kind=='ntv2' else S['parameters_total'];S['checkpoint_reference']={'scvi':'models/scvi_20261027_185327_767988/train_10176','dnabert2':'/workspace/dnabert2_eval/model + model_revision.txt','ntv2':'models/checkpoint; existing setup/download provenance','rnafm':'models/RNA-FM_pretrained.pth; existing data/sources.json','openspliceai':'source/models/openspliceai-mane/10000nt seeds10..14'}[kind];S['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'precision':'FP32','tf32':False};save()
 if kind!='scvi':
  cache=D/'token_cache.pt'
  if not cache.exists():
   if tokenizer is not None:
    encoded=tokenizer([r['sequence'] for r in records],padding=True,truncation=False,return_tensors='pt');encoded={k:encoded[k] for k in ['input_ids','attention_mask']}
   elif kind=='rnafm':
    _,_,tok=converter([(r['id'],r['sequence']) for r in records]);encoded={'input_ids':tok}
   else:
    tab=np.zeros((256,4),np.uint8)
    for j,c in enumerate('ACGT'):tab[ord(c),j]=1
    encoded={'onehot':torch.from_numpy(np.stack([tab[np.frombuffer(r['sequence'].encode(),np.uint8)].T for r in records]))}
   torch.save(encoded,cache)
  encoded=torch.load(cache,map_location='cpu',weights_only=True);cpu={k:v[:B] for k,v in encoded.items()};S['input']={'unique_sequences':B,'pool_sequences':len(records),'raw_length_min':min(len(r['sequence']) for r in records[:B]),'raw_length_max':max(len(r['sequence']) for r in records[:B]),'input_shapes':{k:list(v.shape) for k,v in cpu.items()},'tokenizer_truncation':False,'padding':'same pool-wide longest token width fixed across batches; explicit crop lengths in inputs provenance','ids':[r['id'] for r in records[:B]]}
  if kind=='ntv2':assert cpu['input_ids'].shape[1]<=2048
  if kind=='rnafm':assert cpu['input_ids'].shape[1]<=1024
  def forward():
   x={k:(v.cuda().float() if k=='onehot' else v.cuda()) for k,v in cpu.items()}
   if kind in ['dnabert2','ntv2']:
    h=m(**x)[0] if kind=='dnabert2' else m.esm(**x,return_dict=True).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1).to(h.dtype);out=(h*mask).sum(1)/mask.sum(1)
   elif kind=='rnafm':
    h=m(x['input_ids'],repr_layers=[12],return_contacts=False)['representations'][12];out=torch.stack([h[i,1:len(r['sequence'])+1].mean(0) for i,r in enumerate(records[:B])])
   else:
    out=None
    for net in nets:
     z=net(x['onehot']);out=z if out is None else out+z
    out=out/len(nets)
   return out.cpu().numpy()
 meter.phase='warmup';t=time.time()
 with torch.no_grad():out=forward()
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;assert out.shape[0]==B and np.isfinite(out).all();del out;gc.collect();t,base=meter.begin('inference');steps=0
 with torch.no_grad():
  while steps<2 or time.time()-t<15:out=forward();assert np.isfinite(out).all();torch.cuda.synchronize();steps+=1
 r=meter.end(t,base,batch_size=B,unique_inputs=B,processed_inputs=B*steps,repeats=steps);r['inputs_per_second']=B*steps/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];r['model_only_allocated_gib']=S['model_only_postload']['allocated_gib'];S['conditions']=[r];np.save(C/'outputs.npy',out);assert np.array_equal(np.load(C/'outputs.npy'),out);S['output_shape']=list(out.shape);S['finite_and_exact_save_reload']=True
 first=D/'reference_first.npy'
 if not first.exists():np.save(first,out[0]);S['reference_first_created']=True
 else:
  ref=np.load(first);diff=float(np.abs(ref-out[0]).max());S['batch_invariance_max_abs']=diff;assert np.allclose(ref,out[0],atol=1e-4,rtol=1e-4),(kind,diff)
 S['status']='complete';save();print('BATCH_COMPLETE',kind,B,r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->


</details>
