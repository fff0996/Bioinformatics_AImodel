# RNA-FM 최종 통합 보고서

## 1. 모델과 실제 입력

RNA 염기서열 → 염기토큰 → 640차원 임베딩. 분류실험은본체를동결하고선형head만학습.

실제파일·입력/정답예시·전처리와모델구조설명은이문서5절에전문을포함했다. 추론데이터와지도학습label을구분하며,세포/유전자·서열/토큰·이미지/타일의단위를섞지않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|배치 추론|고유1개; 실제RNAcentral512nt RNA|batch1, input_ids:[1, 514],FP32|667회/총667건|15.00초|44.45입력/초|0.448 GiB / 0.37%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|배치 추론|고유76개; 실제RNAcentral512nt RNA|batch76, input_ids:[76, 514],FP32|9회/총684건|16.63초|41.13입력/초|3.844 GiB / 3.16%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|기존실제길이혼합 임베딩추출|고유896RNA;원래길이혼합(최대1020nt)|최대batch50,배치내길이padding(max1022토큰),FP32|전체896개1회|46.98초|19.07서열/초|8.870 GiB / 7.29%|상기고정512nt후속과조건다름;80%미도달|
|분류head만학습(본체동결)|train896/val224/test280;실제RNA label|batch8×640차원임베딩,FP32|100epoch/11200step|10.66초(validation포함)|8404.04벡터/초(범위주의)|0.434 GiB / 0.36%|본체파인튜닝아님;test지표본문;80%미도달|

## 4. 결과 해석

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### RNA-FM — DGX Spark


run `rnafm_20261027_204440_142840` · 상태 **complete** · summary (원본 기록 로컬 보관: `rnafm/results/rnafm_20261027_204440_142840/summary.json`) · 로그 (로컬 보관: `../results/rnafm_20261027_204440_142840/benchmark.log`) · 자원 시계열 (원본 기록 로컬 보관: `rnafm/results/rnafm_20261027_204440_142840/resources.csv`)

#### 체크포인트와 실행 구성

공식RNA-FM ncRNA 12-layer checkpoint `RNA-FM_pretrained.pth`, mRNA-FM과구분한다. 소스 commit `348951516e0963d22bbb33b3c9fc18c89081d38e`, RNA-FM0.2.2. 공식 Hugging Face cuhkaih/rnafm의 고정revision/SHA-256은 models/repository.json과 models/sources.json에보존했고LFS해시를검증했다.

실제 파라미터: {"backbone_total": 99521546, "backbone_trainable": 0, "head": 17948}

전체RNA-FM99,521,546개파라미터의FP32이론용량은약0.371GiB다. 배포checkpoint1.19GB와실제GPU모델할당량은같지않다. forward는기본LM logits와12층representation을계산하지만contact attention/structure predictor는계산하지않는다. 공식RNAtype튜토리얼처럼backbone은동결하고640→28선형classifier만학습했다. RNA-FM backbone파인튜닝완료라고표현하지않는다.

환경 {"torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "rna_fm": "0.2.2", "gpu": "NVIDIA GB10"}

FP32,TF32=False,seed42,8CPUthreads. 공식RNAtype추론chunk50을배치상한으로,분류기학습batch8/lr1e-3/Adam/100epochs를고정했다. 튜토리얼의1024zero-pad평균은사용하지않고실제염기token만mean pooling했다(BOS/EOS/PAD제외).

| 입력forward전 | allocatedGiB | reservedGiB | RSSGiB | system availableGiB |
| --- | --- | --- | --- | --- |
| RNA-FM 적재 직후 | 0.371 | 0.420 | 1.221 | 112.919 |

적재 0.526초,3배치warmup 0.847초. CPU입력JSON은적재전에준비했다.

#### 실제 RNA와 유형 라벨

공식RNA-FM튜토리얼이제공한RNAcentral1400서열/28유형을사용했다. 원본FASTA와SHA-256은input_audit에기록했다. ACGUN을검사하고N은그대로지원token으로유지했다. exact sequence중복과상충라벨을검사했으며이번데이터는1400개모두고유했다. 각유형별seed42로train32/validation8/test10개를분리해896/224/280개다. 동족성cluster분할은하지않았고RNAcentral이사전학습에도쓰여완전독립외부일반화는미확인이다.

| split | n | 길이nt |
| --- | --- | --- |
| train | 896 | 20–1020 |
| validation | 224 | 21–974 |
| test | 280 | 19–953 |
| long | 7 | 967–1020 |
| count512 | 64 | 521–1020 |

라벨매핑 {"RNase_MRP_RNA": 0, "RNase_P_RNA": 1, "SRP_RNA": 2, "Y_RNA": 3, "antisense_RNA": 4, "autocatalytically_spliced_intron": 5, "guide_RNA": 6, "hammerhead_ribozyme": 7, "lncRNA": 8, "miRNA": 9, "misc_RNA": 10, "ncRNA": 11, "other": 12, "piRNA": 13, "pre_miRNA": 14, "precursor_RNA": 15, "rRNA": 16, "ribozyme": 17, "sRNA": 18, "scRNA": 19, "scaRNA": 20, "siRNA": 21, "snRNA": 22, "snoRNA": 23, "tRNA": 24, "telomerase_RNA": 25, "tmRNA": 26, "vault_RNA": 27}

실제학습예: URS00020832CC / ribozyme / label17

```text
UUUUUUUUUUUUUUUUUUUUUUUUUUGGUUCUAGUAACUAACAGGAAAGCCCAGAAAAACUGCUGCGUUGUUCCAUUUUCAAUUAUCGCGU
```

RNA-FM은염기당1token+BOS/EOS로변환한다.원본최대1020nt→1022tokens로max_positions1024범위내다. 분류기학습입력은동결RNA-FM에서미리추출한B×640 FP32 embedding과B정수label이다. 원본서열·라벨과split은 inputs.json (로컬 보관: `../data/inputs.json`),검사는 [input_audit](RNA-FM.md#record-d4c1c95d3dfb)에있다.

개수비교는512nt이상인실제서열64개를동일512ntcrop으로만들어중첩8/32/64개,길이비교는960nt이상인동일실제7개에서128/256/512/960nt앞부분crop이다. 반복은계측반복이며고유입력으로세지않는다. 최대길이비교에일곱개만있다는데이터한계를명시한다.

#### 입력별 GPU 사용률·메모리

| 조건 | nt | 고유입력 | 반복/epoch | 초 | 건/초 | GPU평균/최대% | 시작GiB | peakGiB | 증가GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inference_nt512_n8 | 512 | 8 | 74 | 15.001 | 39.465 | 91.964/95.000 | 0.402 | 0.766 | 0.364 |
| inference_nt512_n32 | 512 | 32 | 19 | 15.597 | 38.983 | 94.724/95.000 | 0.402 | 1.852 | 1.450 |
| inference_nt512_n64 | 512 | 64 | 10 | 16.468 | 38.864 | 94.839/95.000 | 0.402 | 2.665 | 2.263 |
| inference_nt128_n7 | 128 | 7 | 453 | 15.019 | 211.127 | 94.357/95.000 | 0.402 | 0.479 | 0.077 |
| inference_nt256_n7 | 256 | 7 | 195 | 15.029 | 90.827 | 94.786/95.000 | 0.402 | 0.561 | 0.159 |
| inference_nt512_n7 | 512 | 7 | 84 | 15.038 | 39.100 | 94.464/95.000 | 0.402 | 0.721 | 0.319 |
| inference_nt960_n7 | 960 | 7 | 35 | 15.333 | 15.979 | 94.759/95.000 | 0.402 | 1.456 | 1.054 |
| extract_train | native | 896 | 1 | 46.985 | 19.070 | 94.605/95.000 | 0.402 | 8.870 | 8.468 |
| extract_validation | native | 224 | 1 | 12.297 | 18.215 | 87.591/95.000 | 0.402 | 8.147 | 7.745 |
| extract_test | native | 280 | 1 | 14.930 | 18.755 | 91.643/95.000 | 0.402 | 7.828 | 7.426 |
| linear_head_training_100epochs | native | 896 | 100 | 10.662 | 8404.036 | 19.250/95.000 | 0.402 | 0.434 | 0.032 |

추론규모표시간은CPUtoken화·GPU전송·forward·pooling까지이며출력은GPUtensor다. native_embedding행은CPUembedding반환까지포함한다. 원본파일준비·모델적재·warmup·저장은제외하고별도기록한다. 추론7조건최소15초,각조건allocator peak를reset하고CUDA경계를synchronize했다. GPU-Util은활동시간비율이며FLOPS나memory%가아니다.0.5초대기+조회완료가조건안에포함된표본만집계했다. 공유RAM121.688GiB에서RSS와CUDA는더하지않는다. RSS/available은시계열sampled값이다.

#### 분류기 학습·저장·test

선형head만100epoch학습하고validation Accuracy가가장높은epoch를선택했다. test로epoch를선택하지않았다. 선택epoch 100; test {"accuracy": 0.7678571428571429, "macro_f1": 0.7662770895636066}

headweight최대변화 2.795; frozenbackbone확인tensor변화 0.0. 전체backbone은requires_grad=False라optimizer대상에없다.

저장head `models/rnafm_20261027_204440_142840/best_head.pt`,frozenbackbone은`models/RNA-FM_pretrained.pth`와함께쓴다. task.json에라벨/전처리/원본모델을저장했고최종epoch100optimizer/RNG상태와선택besthead를구분했다. 학습재개는검증하지않았다. 저장시간 0.001초. 독립새프로세스RNA-FM+head재로드: {"independent_process": true, "inputs": 280, "embedding_max_abs_difference": 0.0, "logits_max_abs_difference": 0.0, "predictions_match": true, "passed": true}

head학습GPU구간에는epoch별validation이포함돼있으며동결backbone은GPU에유지했지만재계산하지않았다. 작은선형head의GPU-Util이낮아도실패가아니다. 이결과는RNA구조예측정확도나backbone파인튜닝성능과다르다.

#### 한계·공식 출처

RNA유형간길이분포차이가있고동족서열·사전학습중복가능성이있어소규모튜토리얼재현수준의분류결과다. source setup.py의누락README_backup.md참조만README.md로고쳐설치했다. PyTorchcheckpoint의argparse.Namespace는명시safe_globals로허용하고공식해시검증파일만불러왔다.

- [RNA-FM공식소스](https://github.com/ml4bio/RNA-FM)
- [공식RNAtype튜토리얼·데이터](https://github.com/ml4bio/RNA-FM/tree/main/tutorials/rna_family-clustering_type-classification)
- [공식checkpoint](https://huggingface.co/cuhkaih/rnafm)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### rnafm 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_144452_703996` · 종료사유 `available_unique_input_limit`

#### 입력과 고정 조건

{
  "source": "official tutorial RNAcentral actual unique sequences >=512nt, first512nt explicit crop; pool limit retained",
  "unique_available_inputs": 76,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

#### 배치별 측정

측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.

|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|667|667|15.0043|44.4538|0.3707|0.4485|0.0777|0.4941|0.3685|91.8276|95.0|1.4783|111.8161|
|2|304|608|15.0237|40.4695|0.3707|0.493|0.1223|0.5742|0.4051|92.0|96.0|1.4772|111.6005|
|4|157|628|15.0625|41.6929|0.3707|0.5839|0.2131|0.6309|0.4798|92.25|96.0|1.4775|111.5708|
|8|77|616|15.031|40.982|0.3707|0.7655|0.3948|0.8262|0.6291|95.1852|96.0|1.454|111.3744|
|16|39|624|15.1835|41.0972|0.3707|1.1261|0.7554|1.2148|0.9254|91.9643|96.0|1.4529|110.9596|
|32|20|640|15.6186|40.9767|0.3707|1.8522|1.4815|1.959|1.5221|95.4828|96.0|1.4529|110.2438|
|64|10|640|15.618|40.9784|0.3707|3.2985|2.9278|3.4551|2.7107|95.4483|96.0|1.4528|108.684|
|76|9|684|16.6293|41.1321|0.3707|3.8443|3.4735|4.0059|3.1591|95.4194|96.0|1.4527|108.1993|

#### 결과 해석과 검증

최대시험성공batch 76. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://github.com/ml4bio/RNA-FM) · [기존 모델 보고서 디렉터리](RNA-FM.md)

### 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.


## 6. 실제 실행 코드·명령어 전문

<!-- EXECUTED_SCRIPTS_START -->

### 원문과 출처 검증

아래 코드 블록은 원본 내용을 생략하거나 재작성하지 않았다. 각 파일의 원래 경로·원문 SHA256·출처 상태를 함께 기록했다. `실행 run 저장 snapshot`을 우선하고, 현재 소스만 남은 경우에는 당시 파일과 같다고 단정하지 않는다. 동일 바이트 버전은 한 번만 수록하고 해당 경로들을 함께 표시한다.

GitHub에는 `.py`/`.sh`를 별도 파일로 두지 않는다. 코드 블록을 원래 경로에 저장하거나 [복원 안내](RNA-FM.md)의 코드로 로컬에서 복원한다. 라이브러리 전체 소스는 이 사용자 실행 스크립트와 구분하며 패키지 버전/출처를 따른다.

### 코드 목록

|원래 파일|원문 SHA256|출처 상태|
|---|---|---|
|`rnafm/scripts/benchmark.py`|`16bd21a5b8334af05e0fe06f437a974ac83e51a2d20bfd66421838cc770bea58`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`rnafm/scripts/download.py`|`f366bdfe5347a48730184673b7fe2067180f7f0aead1dd04f7ba3b800399071f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`rnafm/scripts/prepare.py`|`4ba91d09440c7414653ea63e786d207629ce4ccc5905cac9a56dffa8eca917fb`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`rnafm/scripts/reload.py`|`2126d29afe17eaf672647be834c0a2c2550c1bc203f4d36434801003cc41533b`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`rnafm/scripts/report.py`|`7477fe25e592335212712226fd9a9d7b375137072dc8a587ed451e8c1629a3f2`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`rnafm/scripts/setup.py`|`677fbc25a9369717a1b850aa32d18fdf99d2c1f441f245db86c3653f37025b8a`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|
|`scripts/remaining_followups/launch.py`|`84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/report.py`|`8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/worker.py`|`e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|

<details>
<summary>rnafm/scripts/benchmark.py — 원문 전문 (7,009 bytes)</summary>


<!-- RESTORE_FILE {"path": "rnafm/scripts/benchmark.py", "sha256": "16bd21a5b8334af05e0fe06f437a974ac83e51a2d20bfd66421838cc770bea58", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,os,time,json,random,argparse
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import torch,numpy as np,fm
from sklearn.metrics import accuracy_score,f1_score
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.serialization.add_safe_globals([argparse.Namespace]);meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'rna_fm':fm.__version__,'gpu':torch.cuda.get_device_name(0)},'settings':{'inference_batch_cap':50,'head_training_batch':8,'dtype':'FP32','tf32':False,'seed':42,'epochs':100,'lr':.001,'optimizer':'Adam','training':'frozen pretrained RNA-FM + linear640->28 classifier; backbone NOT finetuned','pooling':'mean only actual nucleotide tokens, excludes BOS/EOS/PAD','reference':'official RNA type tutorial chunk50/head batch8/lr1e-3/100epochs; corrected padded-mean pooling'}}
def save():write_json(D/'summary.json',S)
try:
 data=json.loads((R/'data/inputs.json').read_text());meter.phase='loading';t=time.time();backbone,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');backbone=backbone.float().cuda().eval();backbone.requires_grad_(False);convert=alphabet.get_batch_converter();torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'backbone_total':sum(p.numel() for p in backbone.parameters()),'backbone_trainable':sum(p.numel() for p in backbone.parameters() if p.requires_grad),'head':640*28+28};save()
 def embed(rows):
  pairs=[(r['id'],r['sequence']) for r in rows];_,_,tokens=convert(pairs);assert tokens.shape[1]<=1024;tokens=tokens.cuda();result=backbone(tokens,repr_layers=[12],return_contacts=False);h=result['representations'][12];z=torch.stack([h[i,1:len(r['sequence'])+1].mean(0) for i,r in enumerate(rows)]);return z,h,tokens
 meter.phase='warmup';t=time.time()
 with torch.no_grad():
  for _ in range(3):z,h,tok=embed(data['train'][:8]);del z,h,tok
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t
 for length,n,group in [(512,8,'count512'),(512,32,'count512'),(512,64,'count512'),(128,7,'long'),(256,7,'long'),(512,7,'long'),(960,7,'long')]:
  rows=[{**r,'sequence':r['sequence'][:length]} for r in data[group][:n]];name=f'inference_nt{length}_n{n}';t,base=meter.begin(name);repeats=0;shapes=set()
  with torch.no_grad():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,50):z,h,tok=embed(rows[i:i+50]);shapes.add(str(list(tok.shape)));del z,h,tok
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='inference',length=length,unique_inputs=n,repeats=repeats,processed=n*repeats,actual_token_shapes=sorted(shapes));row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',name,row['seconds'],row['memory']['peak_allocated_gib'],flush=True);save()
 features={}
 for split in ['train','validation','test']:
  rows=data[split];t,base=meter.begin('extract_'+split);parts=[];shapes=[]
  with torch.no_grad():
   for i in range(0,len(rows),50):
    z,h,tok=embed(rows[i:i+50]);parts.append(z.cpu().numpy());shapes.append(list(tok.shape))
    if split=='train' and i==0:np.save(D/'example_token_embedding.npy',h[0,:len(rows[0]['sequence'])+2].cpu().numpy())
    del z,h,tok
  row=meter.end(t,base,name='extract_'+split,mode='native_embedding',unique_inputs=len(rows),repeats=1,processed=len(rows),actual_token_shapes=shapes);row['throughput']=len(rows)/row['seconds'];S['conditions'].append(row);features[split]=np.concatenate(parts);np.save(D/(split+'_embeddings.npy'),features[split]);save()
 head=torch.nn.Linear(640,28).float().cuda();opt=torch.optim.Adam(head.parameters(),lr=.001);criterion=torch.nn.CrossEntropyLoss();Ys={k:np.array([r['label'] for r in data[k]]) for k in features};before=head.weight.detach().cpu().clone();originalbackbone=next(backbone.parameters()).detach().cpu().clone();history=[];best=-1;beststate=None;bestepoch=None
 def evaluate(split):
  head.eval();parts=[]
  with torch.no_grad():
   for i in range(0,len(features[split]),8):parts.append(head(torch.from_numpy(features[split][i:i+8]).cuda()).cpu().numpy())
  logits=np.concatenate(parts);pred=logits.argmax(1);return {'accuracy':float(accuracy_score(Ys[split],pred)),'macro_f1':float(f1_score(Ys[split],pred,average='macro',labels=list(range(28)),zero_division=0))},logits
 t,base=meter.begin('linear_head_training_100epochs')
 for epoch in range(100):
  head.train();order=np.random.default_rng(42+epoch).permutation(len(features['train']));losses=[]
  for i in range(0,len(order),8):
   ids=order[i:i+8];x=torch.from_numpy(features['train'][ids]).cuda();y=torch.from_numpy(Ys['train'][ids]).cuda();opt.zero_grad(set_to_none=True);loss=criterion(head(x),y);loss.backward();opt.step();losses.append(float(loss.detach()))
  val,_=evaluate('validation');history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':val})
  if val['accuracy']>best:best=val['accuracy'];bestepoch=epoch+1;beststate={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
  if (epoch+1)%20==0:print('HEAD_EPOCH',epoch+1,val,flush=True)
 row=meter.end(t,base,name='linear_head_training_100epochs',mode='linear_head_training_with_validation',unique_inputs=len(features['train']),repeats=100,processed=len(features['train'])*100,steps=int(np.ceil(len(features['train'])/8))*100);row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);S['training_history']=history;S['best_epoch']=bestepoch;S['head_weight_change_max_abs']=float((head.weight.detach().cpu()-before).abs().max());S['backbone_weight_change_max_abs']=float((next(backbone.parameters()).detach().cpu()-originalbackbone).abs().max());assert S['backbone_weight_change_max_abs']==0;meter.phase='saving';t=time.time();torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'epoch':100,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},M/'final_training_state.pt');torch.save(beststate,M/'best_head.pt');write_json(M/'task.json',{'input_dim':640,'labels':json.loads((R/'data/input_audit.json').read_text())['labels'],'backbone_checkpoint':'models/RNA-FM_pretrained.pth','pooling':S['settings']['pooling'],'selected_epoch':bestepoch,'resume_tested':False});S['save_seconds']=time.time()-t;head.load_state_dict(beststate);meter.phase='test';S['test'],logits=evaluate('test');np.save(D/'test_logits.npy',logits);write_json(D/'test_predictions.json',[{'id':r['id'],'label':r['label'],'prediction':int(v.argmax())} for r,v in zip(data['test'],logits)]);S['saved_head']=str((M/'best_head.pt').relative_to(R));S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>rnafm/scripts/download.py — 원문 전문 (756 bytes)</summary>


<!-- RESTORE_FILE {"path": "rnafm/scripts/download.py", "sha256": "f366bdfe5347a48730184673b7fe2067180f7f0aead1dd04f7ba3b800399071f", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,json,urllib.request
sys.path.insert(0,'/home/hyein/ai-benchmark/scripts')
from fetch_hf_files import fetch
R=Path('/home/hyein/ai-benchmark/rnafm');repo='cuhkaih/rnafm';m=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo));rev=m['sha'];tree=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'/tree/'+rev));(R/'models/repository.json').write_text(json.dumps({'meta':m,'tree':tree},indent=2));f=next(r for r in tree if r['path']=='RNA-FM_pretrained.pth');row=fetch('https://huggingface.co/'+repo+'/resolve/'+rev+'/'+f['path'],R/'models'/f['path'],f['size'],f.get('lfs',{}).get('oid'));(R/'models/sources.json').write_text(json.dumps([row],indent=2));print(row,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>rnafm/scripts/prepare.py — 원문 전문 (2,381 bytes)</summary>


<!-- RESTORE_FILE {"path": "rnafm/scripts/prepare.py", "sha256": "4ba91d09440c7414653ea63e786d207629ce4ccc5905cac9a56dffa8eca917fb", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,hashlib,collections,random
R=Path('/home/hyein/ai-benchmark/rnafm');p=R/'source/tutorials/tutorial_data/format_rnacentral_active.100.sample-Max50.fasta';rows=[]
for part in p.read_text().split('>')[1:]:
 ls=part.splitlines();head=ls[0];seq=''.join(ls[1:]).upper();assert seq and set(seq)<=set('ACGUN');rows.append({'id':head.split()[0],'description':head,'type':head.split()[1],'sequence':seq,'sha256':hashlib.sha256(seq.encode()).hexdigest()})
byseq=collections.defaultdict(list)
for r in rows:byseq[r['sha256']].append(r)
unique=[];conflicts=0
for rs in byseq.values():
 if len({r['type'] for r in rs})>1:conflicts+=len(rs);continue
 unique.append(rs[0])
labels=sorted({r['type'] for r in unique});mapping={v:i for i,v in enumerate(labels)};groups=collections.defaultdict(list)
for r in unique:r['label']=mapping[r['type']];groups[r['type']].append(r)
rng=random.Random(42);out={'train':[],'validation':[],'test':[]}
for name,rs in sorted(groups.items()):
 rng.shuffle(rs);n=len(rs);nt=max(1,round(n*.2));nv=max(1,round(n*.16));out['test']+=rs[:nt];out['validation']+=rs[nt:nt+nv];out['train']+=rs[nt+nv:]
for rs in out.values():rng.shuffle(rs)
out['long']=[r for r in unique if len(r['sequence'])>=960];out['count512']=[r for r in unique if len(r['sequence'])>=512][:64];assert len(out['long'])==7 and len(out['count512'])==64
sets=[{r['sha256'] for r in out[k]} for k in ['train','validation','test']];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
audit={'source_file':str(p.relative_to(R)),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'raw_rows':len(rows),'unique_unambiguous':len(unique),'conflicting_label_rows_removed':conflicts,'labels':mapping,'splits':{k:{'n':len(rs),'length_min':min(len(r['sequence']) for r in rs),'length_max':max(len(r['sequence']) for r in rs),'labels':dict(collections.Counter(r['type'] for r in rs)),'first_example':rs[0]} for k,rs in out.items()},'policy':'exact-sequence dedup and conflicting-label removal before stratified per-type seed42 split; homologs not clustered; RNACentral may overlap pretraining'};(R/'data/inputs.json').write_text(json.dumps(out,indent=2));(R/'data/input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v['n'] for k,v in audit['splits'].items()}));print('labels',len(labels),'dedup',len(unique),'conflicts',conflicts)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>rnafm/scripts/reload.py — 원문 전문 (1,831 bytes)</summary>


<!-- RESTORE_FILE {"path": "rnafm/scripts/reload.py", "sha256": "2126d29afe17eaf672647be834c0a2c2550c1bc203f4d36434801003cc41533b", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,json,os,argparse
sys.path.insert(0,'/support')
from gpu_support import write_json
import torch,numpy as np,fm
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());torch.serialization.add_safe_globals([argparse.Namespace]);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model,alphabet=fm.pretrained.rna_fm_t12(R/'models/RNA-FM_pretrained.pth');model=model.float().cuda().eval();convert=alphabet.get_batch_converter();head=torch.nn.Linear(640,28).cuda();head.load_state_dict(torch.load(R/s['saved_head'],weights_only=True,map_location='cuda'));head.eval();rows=json.loads((R/'data/inputs.json').read_text())['test'];parts=[]
with torch.no_grad():
 for i in range(0,len(rows),50):
  rs=rows[i:i+50];_,_,tokens=convert([(r['id'],r['sequence']) for r in rs]);h=model(tokens.cuda(),repr_layers=[12],return_contacts=False)['representations'][12];z=torch.stack([h[j,1:len(r['sequence'])+1].mean(0) for j,r in enumerate(rs)]);parts.append(z.cpu().numpy())
 X=np.concatenate(parts);refX=np.load(D/'test_embeddings.npy');pred=[]
with torch.no_grad():
 for i in range(0,len(X),8):pred.append(head(torch.from_numpy(X[i:i+8]).cuda()).cpu().numpy())
y=np.concatenate(pred);ref=np.load(D/'test_logits.npy');r={'independent_process':True,'inputs':len(rows),'embedding_max_abs_difference':float(np.abs(X-refX).max()),'logits_max_abs_difference':float(np.abs(y-ref).max()),'predictions_match':bool(np.array_equal(y.argmax(1),ref.argmax(1)))};r['passed']=r['embedding_max_abs_difference']<=1e-5 and r['logits_max_abs_difference']<=1e-5 and r['predictions_match'];write_json(D/'reload.json',r);assert r['passed'];s['reload']=r;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(r),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>rnafm/scripts/report.py — 원문 전문 (7,204 bytes)</summary>


<!-- RESTORE_FILE {"path": "rnafm/scripts/report.py", "sha256": "7477fe25e592335212712226fd9a9d7b375137072dc8a587ed451e8c1629a3f2", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import json,csv
R=Path('/home/hyein/ai-benchmark/rnafm');run=(R/'results/LATEST').read_text().strip();D=R/'results'/run;s=json.loads((D/'summary.json').read_text());audit=json.loads((R/'data/input_audit.json').read_text());out=[]
def a(t=''):out.append(t)
def table(h,rows):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rows:a('| '+' | '.join(map(str,r))+' |')
 a()
def f(v):return 'N/A' if v is None else f'{v:.3f}'
a('# RNA-FM — DGX Spark');a();a(f'run `{run}` · 상태 **{s["status"]}** · [summary](../results/{run}/summary.json) · [로그](../results/{run}/benchmark.log) · [자원 시계열](../results/{run}/resources.csv)');a();a('## 체크포인트와 실행 구성');a();a('공식RNA-FM ncRNA 12-layer checkpoint `RNA-FM_pretrained.pth`, mRNA-FM과구분한다. 소스 commit `348951516e0963d22bbb33b3c9fc18c89081d38e`, RNA-FM0.2.2. 공식 Hugging Face cuhkaih/rnafm의 고정revision/SHA-256은 models/repository.json과 models/sources.json에보존했고LFS해시를검증했다.');a();a('실제 파라미터: '+json.dumps(s.get('parameters',{})));a();a('전체RNA-FM99,521,546개파라미터의FP32이론용량은약0.371GiB다. 배포checkpoint1.19GB와실제GPU모델할당량은같지않다. forward는기본LM logits와12층representation을계산하지만contact attention/structure predictor는계산하지않는다. 공식RNAtype튜토리얼처럼backbone은동결하고640→28선형classifier만학습했다. RNA-FM backbone파인튜닝완료라고표현하지않는다.');a();a('환경 '+json.dumps(s['environment']));a();a('FP32,TF32=False,seed42,8CPUthreads. 공식RNAtype추론chunk50을배치상한으로,분류기학습batch8/lr1e-3/Adam/100epochs를고정했다. 튜토리얼의1024zero-pad평균은사용하지않고실제염기token만mean pooling했다(BOS/EOS/PAD제외).');a();b=s['model_only_postload'];table(['입력forward전','allocatedGiB','reservedGiB','RSSGiB','system availableGiB'],[['RNA-FM 적재 직후',*[f(b[k]) for k in ['allocated_gib','reserved_gib','rss_gib','system_available_gib']]]]);a('적재 '+f(s['load_seconds'])+'초,3배치warmup '+f(s.get('warmup_seconds'))+'초. CPU입력JSON은적재전에준비했다.');a();a('## 실제 RNA와 유형 라벨');a();a('공식RNA-FM튜토리얼이제공한RNAcentral1400서열/28유형을사용했다. 원본FASTA와SHA-256은input_audit에기록했다. ACGUN을검사하고N은그대로지원token으로유지했다. exact sequence중복과상충라벨을검사했으며이번데이터는1400개모두고유했다. 각유형별seed42로train32/validation8/test10개를분리해896/224/280개다. 동족성cluster분할은하지않았고RNAcentral이사전학습에도쓰여완전독립외부일반화는미확인이다.');a();table(['split','n','길이nt'],[[k,v['n'],str(v['length_min'])+'–'+str(v['length_max'])] for k,v in audit['splits'].items()]);a('라벨매핑 '+json.dumps(audit['labels'],ensure_ascii=False));a();ex=audit['splits']['train']['first_example'];a('실제학습예: '+ex['id']+' / '+ex['type']+' / label'+str(ex['label']));a();a('```text\n'+ex['sequence']+'\n```');a();a('RNA-FM은염기당1token+BOS/EOS로변환한다.원본최대1020nt→1022tokens로max_positions1024범위내다. 분류기학습입력은동결RNA-FM에서미리추출한B×640 FP32 embedding과B정수label이다. 원본서열·라벨과split은 [inputs.json](../data/inputs.json),검사는 [input_audit](../data/input_audit.json)에있다.');a();a('개수비교는512nt이상인실제서열64개를동일512ntcrop으로만들어중첩8/32/64개,길이비교는960nt이상인동일실제7개에서128/256/512/960nt앞부분crop이다. 반복은계측반복이며고유입력으로세지않는다. 최대길이비교에일곱개만있다는데이터한계를명시한다.');a();a('## 입력별 GPU 사용률·메모리');a();rs=[];flat=[]
for r in s['conditions']:
 rs.append([r['name'],r.get('length','native'),r['unique_inputs'],r['repeats'],f(r['seconds']),f(r['throughput']),f(r['gpu_util_avg'])+'/'+f(r['gpu_util_max']),f(r['baseline']['allocated_gib']),f(r['memory']['peak_allocated_gib']),f(r['peak_delta_gib'])]);d={k:v for k,v in r.items() if k not in ['memory','baseline']};d.update(allocated_baseline_gib=r['baseline']['allocated_gib'],peak_allocated_gib=r['memory']['peak_allocated_gib'],peak_reserved_gib=r['memory']['peak_reserved_gib']);flat.append(d)
table(['조건','nt','고유입력','반복/epoch','초','건/초','GPU평균/최대%','시작GiB','peakGiB','증가GiB'],rs)
if flat:
 keys=list(dict.fromkeys(k for r in flat for k in r))
 with (R/'reports/comparison.csv').open('w',newline='') as f0:w=csv.DictWriter(f0,fieldnames=keys);w.writeheader();w.writerows(flat)
a('추론규모표시간은CPUtoken화·GPU전송·forward·pooling까지이며출력은GPUtensor다. native_embedding행은CPUembedding반환까지포함한다. 원본파일준비·모델적재·warmup·저장은제외하고별도기록한다. 추론7조건최소15초,각조건allocator peak를reset하고CUDA경계를synchronize했다. GPU-Util은활동시간비율이며FLOPS나memory%가아니다.0.5초대기+조회완료가조건안에포함된표본만집계했다. 공유RAM121.688GiB에서RSS와CUDA는더하지않는다. RSS/available은시계열sampled값이다.');a()
if 'test' in s:
 a('## 분류기 학습·저장·test');a();a('선형head만100epoch학습하고validation Accuracy가가장높은epoch를선택했다. test로epoch를선택하지않았다. 선택epoch '+str(s['best_epoch'])+'; test '+json.dumps(s['test']));a();a('headweight최대변화 '+f(s['head_weight_change_max_abs'])+'; frozenbackbone확인tensor변화 '+str(s['backbone_weight_change_max_abs'])+'. 전체backbone은requires_grad=False라optimizer대상에없다.');a();a('저장head `'+s['saved_head']+'`,frozenbackbone은`models/RNA-FM_pretrained.pth`와함께쓴다. task.json에라벨/전처리/원본모델을저장했고최종epoch100optimizer/RNG상태와선택besthead를구분했다. 학습재개는검증하지않았다. 저장시간 '+f(s['save_seconds'])+'초. 독립새프로세스RNA-FM+head재로드: '+json.dumps(s.get('reload',{'status':'pending'})));a();a('head학습GPU구간에는epoch별validation이포함돼있으며동결backbone은GPU에유지했지만재계산하지않았다. 작은선형head의GPU-Util이낮아도실패가아니다. 이결과는RNA구조예측정확도나backbone파인튜닝성능과다르다.');a()
a('## 한계·공식 출처');a();a('RNA유형간길이분포차이가있고동족서열·사전학습중복가능성이있어소규모튜토리얼재현수준의분류결과다. source setup.py의누락README_backup.md참조만README.md로고쳐설치했다. PyTorchcheckpoint의argparse.Namespace는명시safe_globals로허용하고공식해시검증파일만불러왔다.');a();a('- [RNA-FM공식소스](https://github.com/ml4bio/RNA-FM)\n- [공식RNAtype튜토리얼·데이터](https://github.com/ml4bio/RNA-FM/tree/main/tutorials/rna_family-clustering_type-classification)\n- [공식checkpoint](https://huggingface.co/cuhkaih/rnafm)');(R/'reports/RNAFM.md').write_text('\n'.join(out)+'\n');print(R/'reports/RNAFM.md')
````
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>rnafm/scripts/setup.py — 원문 전문 (511 bytes)</summary>


<!-- RESTORE_FILE {"path": "rnafm/scripts/setup.py", "sha256": "677fbc25a9369717a1b850aa32d18fdf99d2c1f441f245db86c3653f37025b8a", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,sys,json,importlib.metadata
from pathlib import Path
R=Path('/work');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True);subprocess.run(['/work/.venv/bin/pip','install','--cache-dir','/work/.cache/pip','--no-deps','/work/source'],check=True);subprocess.run(['/work/.venv/bin/pip','install','scikit-learn','biopython','ptflops'],check=True);(R/'results/setup_done.json').write_text(json.dumps({'torch_preserved':importlib.metadata.version('torch')}))
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

### record-d4c1c95d3dfb

원래 파일: `rnafm/data/input_audit.json` · 원문 SHA256: `e4504c185f7e0db4b151cf4da90cc735c3e79cfc2188fc939ab7d1dad3f77d42`

<details>
<summary>rnafm/data/input_audit.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/data/input_audit.json", "sha256": "e4504c185f7e0db4b151cf4da90cc735c3e79cfc2188fc939ab7d1dad3f77d42", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "source_file": "source/tutorials/tutorial_data/format_rnacentral_active.100.sample-Max50.fasta",
  "source_sha256": "4f45f90112a7ecf005c5435943da81a68fcd057635cde3c701f7ace490949be6",
  "raw_rows": 1400,
  "unique_unambiguous": 1400,
  "conflicting_label_rows_removed": 0,
  "labels": {
    "RNase_MRP_RNA": 0,
    "RNase_P_RNA": 1,
    "SRP_RNA": 2,
    "Y_RNA": 3,
    "antisense_RNA": 4,
    "autocatalytically_spliced_intron": 5,
    "guide_RNA": 6,
    "hammerhead_ribozyme": 7,
    "lncRNA": 8,
    "miRNA": 9,
    "misc_RNA": 10,
    "ncRNA": 11,
    "other": 12,
    "piRNA": 13,
    "pre_miRNA": 14,
    "precursor_RNA": 15,
    "rRNA": 16,
    "ribozyme": 17,
    "sRNA": 18,
    "scRNA": 19,
    "scaRNA": 20,
    "siRNA": 21,
    "snRNA": 22,
    "snoRNA": 23,
    "tRNA": 24,
    "telomerase_RNA": 25,
    "tmRNA": 26,
    "vault_RNA": 27
  },
  "splits": {
    "train": {
      "n": 896,
      "length_min": 20,
      "length_max": 1020,
      "labels": {
        "ribozyme": 32,
        "precursor_RNA": 32,
        "piRNA": 32,
        "autocatalytically_spliced_intron": 32,
        "RNase_P_RNA": 32,
        "pre_miRNA": 32,
        "scRNA": 32,
        "other": 32,
        "RNase_MRP_RNA": 32,
        "guide_RNA": 32,
        "Y_RNA": 32,
        "hammerhead_ribozyme": 32,
        "rRNA": 32,
        "antisense_RNA": 32,
        "vault_RNA": 32,
        "ncRNA": 32,
        "lncRNA": 32,
        "miRNA": 32,
        "snRNA": 32,
        "tmRNA": 32,
        "scaRNA": 32,
        "tRNA": 32,
        "siRNA": 32,
        "telomerase_RNA": 32,
        "sRNA": 32,
        "SRP_RNA": 32,
        "misc_RNA": 32,
        "snoRNA": 32
      },
      "first_example": {
        "id": "URS00020832CC",
        "description": "URS00020832CC ribozyme from 14 species",
        "type": "ribozyme",
        "sequence": "UUUUUUUUUUUUUUUUUUUUUUUUUUGGUUCUAGUAACUAACAGGAAAGCCCAGAAAAACUGCUGCGUUGUUCCAUUUUCAAUUAUCGCGU",
        "sha256": "0c55774d77e2322a51cf3b4786e10509742455f64421c79bf9584fb111d5bef9",
        "label": 17
      }
    },
    "validation": {
      "n": 224,
      "length_min": 21,
      "length_max": 974,
      "labels": {
        "piRNA": 8,
        "hammerhead_ribozyme": 8,
        "guide_RNA": 8,
        "SRP_RNA": 8,
        "lncRNA": 8,
        "autocatalytically_spliced_intron": 8,
        "snoRNA": 8,
        "rRNA": 8,
        "RNase_P_RNA": 8,
        "Y_RNA": 8,
        "tmRNA": 8,
        "snRNA": 8,
        "vault_RNA": 8,
        "siRNA": 8,
        "pre_miRNA": 8,
        "scaRNA": 8,
        "scRNA": 8,
        "RNase_MRP_RNA": 8,
        "precursor_RNA": 8,
        "tRNA": 8,
        "ncRNA": 8,
        "miRNA": 8,
        "telomerase_RNA": 8,
        "other": 8,
        "antisense_RNA": 8,
        "sRNA": 8,
        "ribozyme": 8,
        "misc_RNA": 8
      },
      "first_example": {
        "id": "URS00000DAEB7",
        "description": "URS00000DAEB7 piRNA from 15 species",
        "type": "piRNA",
        "sequence": "UUUGAUGAUUAUGUUGACCAA",
        "sha256": "166b2ee3ace1a3879956a6c2e204c761f12c0284db77b5f32c63cda0f30a9877",
        "label": 13
      }
    },
    "test": {
      "n": 280,
      "length_min": 19,
      "length_max": 953,
      "labels": {
        "SRP_RNA": 10,
        "Y_RNA": 10,
        "telomerase_RNA": 10,
        "hammerhead_ribozyme": 10,
        "rRNA": 10,
        "piRNA": 10,
        "snoRNA": 10,
        "other": 10,
        "precursor_RNA": 10,
        "snRNA": 10,
        "siRNA": 10,
        "lncRNA": 10,
        "scaRNA": 10,
        "tRNA": 10,
        "misc_RNA": 10,
        "ncRNA": 10,
        "miRNA": 10,
        "vault_RNA": 10,
        "pre_miRNA": 10,
        "RNase_P_RNA": 10,
        "antisense_RNA": 10,
        "RNase_MRP_RNA": 10,
        "scRNA": 10,
        "tmRNA": 10,
        "ribozyme": 10,
        "guide_RNA": 10,
        "sRNA": 10,
        "autocatalytically_spliced_intron": 10
      },
      "first_example": {
        "id": "URS0000919068",
        "description": "URS0000919068 SRP_RNA from 13 species",
        "type": "SRP_RNA",
        "sequence": "GCUGCAUAUGUAGGCGUACCAGUAGUCCCAUCUACCAGGAAGGCUGAGGCAGGAGGAUCACUUGAGCCUAGGAGUUCUGGACUGUAGUGCUGAUUGGGUGUCCUAAGUUCCGAUUCAAUAUGGUGACAUUCUGGGAAUGGGGACCCUAGGUUGCCUAAAGAGGGGUGAACUGGCUCAGGUUGGAAAUGGAGCAGCUCAAAACUCCUGUACUAAUCAGUAGUGGUAUACCCUGUGCUAAUCAGUAGUGGUAUAUCCGAUGCUAAUCCAGUAGUGGUACAUCCCUUGC",
        "sha256": "4a47d343b061721e16c2d3a8e5e9ae604cbaff44b95a7963955aac465c94c7dc",
        "label": 2
      }
    },
    "long": {
      "n": 7,
      "length_min": 967,
      "length_max": 1020,
      "labels": {
        "rRNA": 1,
        "sRNA": 2,
        "lncRNA": 2,
        "antisense_RNA": 2
      },
      "first_example": {
        "id": "URS00021DACA6",
        "description": "URS00021DACA6 rRNA from 12 species",
        "type": "rRNA",
        "sequence": "AGAGACUUAGUCCUAACCUUACUGUUGGUUUUUGCCAGACAUAUACAUGCAAGUAUCCGCAUUCCAGUGUAAAAGCCCUAGGCACCCUUAAAUAAAAGUCGAUAGGAGCGGGUAUCAGACUCACCCAUUAACCGGUAGUCCAAGACGCCUUGCACUUGCCACGCCCCCACGGGUACUCAGCAGNNGUUAACAUUAAGCAAUGAGUGUAAACUUGACUUAGUCAUGGCAACUGAGGGUCGGUAAAUCCUGUGCCAGCCACCGCGGUCAUACAGGAGACCCAAAUCAACAUUACUUGACGGCGUAAAGAGUGGUCACAUGUUAUCAAAGUAACUAAGAUUAAAAUGCAACUGAGCUGUCAUAAGCCCAAGAUGCCCAUAAGGCCACCUACCAAAGAAGAUCUUAGACUAACGACCAAUUGAAAUCCACGAAAGCCAGGGCCCAAACUGGGAUUAGAUACCCCACUAUGCCUGGCCCUAAAUAUUGAUGUUUAAUACUACCUAAACAUCCGCCUGGGAACUACGAGCACAAACGCUUAAAACUCUAAGGACUUGGCGGUGCCCCAAACCCACCUAGAGGAGCCUGUUCUAUAAUCGAUAAUCCACGCUACACCUUACCAUUCCUUGCCAAGUCAGCCUACAUACCGCCGUCCCCAGCCCACCCAUCUGAAGGUUCAGCAGUGAGCGCAAUAGUCCAAUCACACUAGUAAGACAGGUCAAGGUAUAGCCUAUGGAAUGGGAGUAAUGGGCUACAUUUUCUAGAUUAGAACAUCACGGCAAAGGGGCAUGAAACUGCCCUUGGAAGGCGGAUUUAGCAGUAAAGUGGGACAAUCGAGCCCUCUUUAAGCCGGCUCUGGGGCACGUACAUACCGCCCGUCACCCUCAUCAAAAGCGACCAAAACCCCAAAUAACUAAUAAGCUACUCAGCCAAAGAUGAGGUAAGUCGUAACAAGGUAAGUGUACCGGAAGGUGCACUUAGACAAU",
        "sha256": "941d608fb96ec260a5b0976d5f186a8bf468a767e4e83a22804e8e796d24c6ca",
        "label": 16
      }
    },
    "count512": {
      "n": 64,
      "length_min": 521,
      "length_max": 1020,
      "labels": {
        "rRNA": 11,
        "sRNA": 11,
        "lncRNA": 26,
        "misc_RNA": 1,
        "ncRNA": 2,
        "precursor_RNA": 6,
        "antisense_RNA": 7
      },
      "first_example": {
        "id": "URS00003F8AE0",
        "description": "URS00003F8AE0 rRNA from 14 species",
        "type": "rRNA",
        "sequence": "UGNNUCAGGAUGAACGCUGACAGAAUGCUUAACACAUGCAAGUCGAUUCGAUUUACCUUCGGGUAUUGAGGAUGGCGGACGGGUGAGUAACGCGUAAGGAACUUGCCUCUUGGUCUGGGACAACUGUUGGAAACGACAGCUAAUACCGGAUAUUAUGAGAUUCUCGCAUGGGAAACUUAUGAAAGCUAUAUGCGCCAAGAGAGAGCCUUGCGUUCCAUUAGCUAGUUGGUGGGGUAACGGCCCACCAAGGCGACGAUGGAUAGCCGGCCUGAGAGGGUGAACGGCCACAAGGGGACUGAGACACGGCCCUUACUCCUACGGGAGGCAGCAGUGGGGAAUAUUGGACAAUGGGCCACAAGCCUGAUCCAGCAAUUCUGUGUGCACGAUGAAGGUCUUCGGAUUGUAAAGUGCUUUCAGUUGGGAAGAAGAAAGUGACGGUACCAACAGAAGAAGCGACGGCUAAAUACGUGCCAGCAGCCGCGGUAAUACGUAUGUCGCAAGCGUUAUCCGGAUUUAUUGGGCGUAAAGCGCGUCUAGGCGGAAAAAUAAGUCUGAUGUUAAAAUGCGGGGCUCAACUCCGUAUUGCGUUGGAAACUGUUUUUCUAGAGUACUGGAGAGGUGGGCGGAACUACNAGUGUAGAGGUGAAAUUCGUAGAUAUUUGUAGGAAUGCCGAUGGAGAAGUCAGCUCACUGGACAGAUACUGACGCUAAANCGCGAAAGC",
        "sha256": "a801c805a577cb63dd0af1b5f71c0e8a011936d582d77a6491e6eb65931fd877",
        "label": 16
      }
    }
  },
  "policy": "exact-sequence dedup and conflicting-label removal before stratified per-type seed42 split; homologs not clustered; RNACentral may overlap pretraining"
}
```
<!-- END_RESTORE_FILE -->


</details>

### record-786daf7a0c4f

원래 파일: `rnafm/models/sources.json` · 원문 SHA256: `b56b800b348c79b640c36d38d1f1e342bcd8597891af226e78e11491c2ef6067`

<details>
<summary>rnafm/models/sources.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/models/sources.json", "sha256": "b56b800b348c79b640c36d38d1f1e342bcd8597891af226e78e11491c2ef6067", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "path": "/home/hyein/ai-benchmark/rnafm/models/RNA-FM_pretrained.pth",
    "url": "https://huggingface.co/cuhkaih/rnafm/resolve/91d4a46d28d8054a7b429955e8fc0c253ba0afd6/RNA-FM_pretrained.pth",
    "sha256": "5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99",
    "bytes": 1194424423
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-6942963462d5

원래 파일: `rnafm/results/batch_followup_20261028_144452_703996/batch16_command.json` · 원문 SHA256: `57b48e587c1fad0b009bd89509b371113c709c42d775d3b2ab35d2cac1ce9363`

<details>
<summary>rnafm/results/batch_followup_20261028_144452_703996/batch16_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch16_command.json", "sha256": "57b48e587c1fad0b009bd89509b371113c709c42d775d3b2ab35d2cac1ce9363", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-16",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=16",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-97f5e3868ce1

원래 파일: `rnafm/results/batch_followup_20261028_144452_703996/batch1_command.json` · 원문 SHA256: `e71d34b318fa5fdff5e507b38ab49191be2d44a236943538964c934e4de98894`

<details>
<summary>rnafm/results/batch_followup_20261028_144452_703996/batch1_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch1_command.json", "sha256": "e71d34b318fa5fdff5e507b38ab49191be2d44a236943538964c934e4de98894", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=1",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-d2c7fe7761b0

원래 파일: `rnafm/results/batch_followup_20261028_144452_703996/batch2_command.json` · 원문 SHA256: `d57ea84b522c6f3d69549feed7e38d436cd3fc85e25fcecdecc0e58caa9ab087`

<details>
<summary>rnafm/results/batch_followup_20261028_144452_703996/batch2_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch2_command.json", "sha256": "d57ea84b522c6f3d69549feed7e38d436cd3fc85e25fcecdecc0e58caa9ab087", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-2",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=2",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-0fafa31587fb

원래 파일: `rnafm/results/batch_followup_20261028_144452_703996/batch32_command.json` · 원문 SHA256: `23ef4ef4ab2703e877178f209024f01415487c907bf70768864cc61e56a54af4`

<details>
<summary>rnafm/results/batch_followup_20261028_144452_703996/batch32_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch32_command.json", "sha256": "23ef4ef4ab2703e877178f209024f01415487c907bf70768864cc61e56a54af4", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-32",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=32",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-b0f9eb79f667

원래 파일: `rnafm/results/batch_followup_20261028_144452_703996/batch4_command.json` · 원문 SHA256: `9d56661fd0c3df65487807d8cb46a03faf2d7e142c6f020748ba43b7b482baef`

<details>
<summary>rnafm/results/batch_followup_20261028_144452_703996/batch4_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch4_command.json", "sha256": "9d56661fd0c3df65487807d8cb46a03faf2d7e142c6f020748ba43b7b482baef", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-4",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=4",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-8b4e8d75faff

원래 파일: `rnafm/results/batch_followup_20261028_144452_703996/batch64_command.json` · 원문 SHA256: `5e5cd4725cb07c9667a267c9fc63e1fad2b5defe673c69db5e5e2f9d50a6abe8`

<details>
<summary>rnafm/results/batch_followup_20261028_144452_703996/batch64_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch64_command.json", "sha256": "5e5cd4725cb07c9667a267c9fc63e1fad2b5defe673c69db5e5e2f9d50a6abe8", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-64",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=64",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-611328585f4c

원래 파일: `rnafm/results/batch_followup_20261028_144452_703996/batch76_command.json` · 원문 SHA256: `d9865f13ac7d6b4c26617d746968a7040a58fa06e9cd53a64c85607b6adb7085`

<details>
<summary>rnafm/results/batch_followup_20261028_144452_703996/batch76_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch76_command.json", "sha256": "d9865f13ac7d6b4c26617d746968a7040a58fa06e9cd53a64c85607b6adb7085", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-76",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=76",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-4c5f6204fea6

원래 파일: `rnafm/results/batch_followup_20261028_144452_703996/batch8_command.json` · 원문 SHA256: `cd0e22588a1f3473921237e7d6002cd95863624a6d83a565116917b513c18a40`

<details>
<summary>rnafm/results/batch_followup_20261028_144452_703996/batch8_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch8_command.json", "sha256": "cd0e22588a1f3473921237e7d6002cd95863624a6d83a565116917b513c18a40", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-8",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=8",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-f63139a5082d

원래 파일: `rnafm/results/rnafm_20261027_204440_142840/benchmark_command.json` · 원문 SHA256: `822c49e4b29625ea4e92c30abdced79e3aa2d30748c7186808e1bee58500c7e4`

<details>
<summary>rnafm/results/rnafm_20261027_204440_142840/benchmark_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/rnafm_20261027_204440_142840/benchmark_command.json", "sha256": "822c49e4b29625ea4e92c30abdced79e3aa2d30748c7186808e1bee58500c7e4", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/rnafm:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=rnafm_20261027_204440_142840 -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/benchmark.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### record-306973ffeecb

원래 파일: `rnafm/results/rnafm_20261027_204440_142840/reload_command.json` · 원문 SHA256: `efe732f30f2bd2e2abb8e1d9158055b8a2326a76c2da1c639d5ed3b6a4b34f69`

<details>
<summary>rnafm/results/rnafm_20261027_204440_142840/reload_command.json — 명령/설정/환경 원문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/rnafm_20261027_204440_142840/reload_command.json", "sha256": "efe732f30f2bd2e2abb8e1d9158055b8a2326a76c2da1c639d5ed3b6a4b34f69", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/rnafm:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=rnafm_20261027_204440_142840 -e HF_HOME=/work/.cache/huggingface -e HF_MODULES_CACHE=/work/.cache/modules -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/reload.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

### 원문 확인·로컬 복원

아래 코드는실제로해시검증·독립임시폴더복원검증에사용한도구전문이다. 필요할때로컬의restore.py에저장하여저장소루트에서`python3 restore.py`(검증),`python3 restore.py --write`(복원)로사용한다. 실제모델실험은자동실행하지않는다. 코드/설정/명령은본문코드블록에수록했고원본데이터·모델가중치·원시자원시계열/대형예측은장비로컬에보존했다. 원본로그를모두보고서본문으로옮긴것은아니다.

```python
from pathlib import Path
import argparse,base64,hashlib,json,re,sys
p=argparse.ArgumentParser(description='Verify or restore original files embedded in Markdown. No benchmark is executed.');p.add_argument('--write',action='store_true',help='restore verified original files under repository root');args=p.parse_args();root=Path.cwd().resolve();files={};aliases=[];blocks=0
for md in root.rglob('*.md'):
 if '.git' in md.parts:continue
 lines=md.read_text(encoding='utf-8').splitlines(keepends=True);i=0
 while i<len(lines):
  line=lines[i].strip()
  if line.startswith('<!-- RESTORE_ALIAS ') and line.endswith(' -->'):aliases.append(json.loads(line[len('<!-- RESTORE_ALIAS '):-len(' -->')]))
  if not line.startswith('<!-- RESTORE_FILE ') or not line.endswith(' -->'):i+=1;continue
  meta=json.loads(line[len('<!-- RESTORE_FILE '):-len(' -->')]);i+=1;opening=re.match(r'^(`{3,})[^\n]*\n?$',lines[i]);assert opening,(md,i);fence=opening.group(1);i+=1;content=[]
  while i<len(lines) and lines[i].strip()!=fence:content.append(lines[i]);i+=1
  assert i<len(lines),(md,'missing closing fence');payload=''.join(content)
  if meta['encoding']=='base64':data=base64.b64decode(payload.strip(),validate=True)
  else:
   if meta.get('added_final_newline'):assert payload.endswith('\n');payload=payload[:-1]
   data=payload.encode('utf-8')
  actual=hashlib.sha256(data).hexdigest();assert actual==meta['sha256'],(md,meta['path'],'SHA256 mismatch');path=(root/meta['path']).resolve();assert root in path.parents,(meta['path'],'path outside repository');previous=files.get(meta['path']);assert previous is None or previous['data']==data,(meta['path'],'conflicting code versions');files[meta['path']]={'data':data,'sha256':actual};blocks+=1;i+=1
pending=aliases[:]
while pending:
 progress=False
 for meta in pending[:]:
  if meta['source'] not in files:continue
  src=files[meta['source']];assert src['sha256']==meta['sha256'];previous=files.get(meta['path']);assert previous is None or previous['data']==src['data'];files[meta['path']]=src;pending.remove(meta);progress=True
 assert progress,('unresolved aliases',pending)
if args.write:
 for name,item in files.items():
  dest=(root/name).resolve();assert root in dest.parents;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and dest.read_bytes()!=item['data']:raise RuntimeError('Refusing to overwrite different existing file: '+name)
  dest.write_bytes(item['data'])
print(json.dumps({'verified_embedded_blocks':blocks,'restorable_original_paths':len(files),'restored':args.write},ensure_ascii=False))

```
