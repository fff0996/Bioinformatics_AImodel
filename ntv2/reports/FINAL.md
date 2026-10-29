# Nucleotide Transformer V2 최종 통합 보고서

## 1. 모델과 실제 입력

DNA서열 → 주로6-mer 토큰 → 임베딩 또는enhancer 분류. 공개500M checkpoint.

실제파일·입력/정답예시·전처리와모델구조설명은이문서5절에전문을포함했다. 추론데이터와지도학습label을구분하며,세포/유전자·서열/토큰·이미지/타일의단위를섞지않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|배치 추론|고유1개; GRCh38 실제6000nt DNA|batch1, input_ids:[1, 1001] / attention_mask:[1, 1001],FP32|78회/총78건|15.12초|5.16입력/초|2.058 GiB / 1.69%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|배치 추론|고유128개; GRCh38 실제6000nt DNA|batch128, input_ids:[128, 1001] / attention_mask:[128, 1001],FP32|2회/총256건|56.80초|4.51입력/초|21.592 GiB / 17.74%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|전체backbone 파인튜닝|실제enhancer train512/val128/test400;199~200nt|batch8,FP32|2epoch/128step|38.13초(학습구간합)|26.86서열/초|9.252 GiB / 7.60%|동일저장모델test 지표본문;80%미도달|

## 4. 결과 해석

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### Nucleotide Transformer V2 — DGX Spark


실행 `ntv2_20261027_201332_251243` · 상태 **complete** · [원본 summary](../results/ntv2_20261027_201332_251243/RECORDS.md#file-88011100c0fb) · 실행 로그 (로컬 보관: `../results/ntv2_20261027_201332_251243/benchmark.log`) · [GPU·메모리 시계열](../results/ntv2_20261027_201332_251243/RECORDS.md#file-53bccab8ae73)

#### 모델과 기준 조건

공식 `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`, 리비전 `06615c1660c892fc199840c18123f8385b3542a8`. V2 500M을 대표로 선택했으며 V1·V3로 바꾸지 않았다. 다운로드된 model.safetensors의 SHA-256은 공식 LFS oid와 일치함을 검증했고 전체 출처는 `results/sources.json`에 보관했다.

실제 파라미터: {"mlm_total": 498345436, "embedding_backbone": 493084113, "mlm_head": 5261323, "classifier_total": 494135763, "classifier_trainable": 494135763}

MLM 전체 모델을 로드하되 임베딩 측정은 `model.esm` backbone의 최종 hidden state만 계산해 masked mean(특수토큰 포함)으로B×1024를 만든다. MLM head와 모든 층 hidden state를 계산/저장하는 경로와 메모리·시간이 다르다. 분류는 공식 custom EsmForSequenceClassification의 CLS MLP head를 사용하고 전 학습 가능 backbone·head를 함께 갱신했다.

환경: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "4.44.2", "cuda": "13.3", "gpu": "NVIDIA GB10"}

batch8,FP32,TF32=False,seed42,8CPU threads. 공식 fine-tuning notebook의 batch8/lr1e-5/2epoch를 참고하되 NT-v2와 enhancer binary 소규모 subset에 적용했다. notebook의1000maxsteps나 전체 데이터 재현 실험이 아니다. 추론에서도 batch8을 고정하며 배치 증가 sweep을 수행하지 않았다.

| 상태 | allocated GiB | reserved GiB | RSS GiB | system available GiB |
| --- | --- | --- | --- | --- |
| MLM 적재 직후/첫 입력 전 | 1.857 | 1.873 | 1.125 | 111.229 |

이론 FP32 가중치 용량은 MLM 파라미터수×4byte/2³⁰이며 실측과 구분한다. 공유 RAM121.688GiB에서 RSS와CUDA를 더하지 않는다. 적재 11.959초; warmup 0.596초. CPU 입력 JSON 로딩은 적재 전에 수행했다.

#### 실제 입력·라벨·분할

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

입력은6-mer 중심 토큰화 후 int64 input_ids/attention_mask의B×실제token수, 라벨은B 정수다. [검사 결과](../data/RECORDS.md#file-5c86518a2a1e)와 선택 서열·split·해시 `data/inputs.json`을 보관했다.

길이/개수 실험은 별도의 GRCh38 참조 유전체 동일 구간128개를 사용한다. BEND에서 이미 확인한 좌표 중심으로12,282nt를 추출하고 ACGT 외 문자가 없는 실제 구간을 택했다. 같은 구간의 중심 crop600/3000/6000/12276nt를 사용하며 무라벨 추론 전용이다. 음의 가닥은 reverse-complement했다. 짧은 enhancer 학습 데이터의 정확도와 이 긴 무라벨 서열을 혼동하지 않는다.

#### 입력별 GPU 사용률·메모리

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

#### 파인튜닝·저장·독립 test

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

#### 해석과 공식 출처

길이가 커질 때 attention 중간값 등 추가 메모리가 커질 수 있지만, 데이터 개수가 늘어도 batch정책이 같으면 순간 peak가비례 증가하지 않는다. 공유RAM90% 채우기는실험목표가아니다.

- [공식 NT-v2 모델 카드](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)
- [공식 소스·학습 예제 링크](https://github.com/instadeepai/nucleotide-transformer)
- [공식 enhancer 데이터](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)
- [공식 분류 fine-tuning notebook](https://github.com/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### ntv2 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_143713_869625` · 종료사유 `available_unique_input_limit`

#### 입력과 고정 조건

{
  "source": "existing real GRCh38 regions, central6000nt crops -> native ~1001tokens",
  "unique_available_inputs": 128,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

#### 배치별 측정

측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.

|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|78|78|15.1193|5.159|1.8565|2.0578|0.2013|2.0781|1.691|90.7931|96.0|1.3639|109.9901|
|2|36|72|15.3145|4.7014|1.8565|2.2095|0.3529|2.2578|1.8157|95.4286|96.0|1.3637|109.5496|
|4|18|72|15.7872|4.5607|1.8565|2.5201|0.6636|2.5547|2.071|95.5333|96.0|1.3639|109.1396|
|8|9|72|16.122|4.466|1.8565|3.14|1.2835|3.2051|2.5804|95.4667|96.0|1.3638|108.3269|
|16|5|80|17.8771|4.475|1.8565|4.3629|2.5064|4.4902|3.5854|95.3636|96.0|1.3638|107.0482|
|32|3|96|21.3241|4.5019|1.8565|6.8339|4.9773|7.0371|5.6159|95.4872|96.0|1.3637|104.5106|
|64|2|128|28.8515|4.4365|1.8565|11.7469|9.8904|12.1543|9.6534|95.3585|96.0|1.3638|99.4116|
|96|2|192|42.5332|4.5141|1.8565|16.6768|14.8203|17.2461|13.7046|93.1772|96.0|1.364|94.6979|
|128|2|256|56.8039|4.5067|1.8565|21.5923|19.7357|22.3652|17.744|94.8269|96.0|1.364|89.7241|

#### 결과 해석과 검증

최대시험성공batch 128. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) · [기존 모델 보고서 디렉터리](.)

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
|`ntv2/scripts/benchmark.py`|`be49bb893e908f3438d91e2b6431a2a25782031c258e8a4c72824f4f4060102b`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/discover.py`|`a9fbd2115bef84a39db1d46e118746f0c59ae67016760723252cb8fadc381572`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/finetune.py`|`8b2557f1f9c116146dbb262496379dac60f5adcd1395e9b890a34eba2d9f0e87`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/inference_remeasure.py`|`16cbf5cc2d7f96d71cd1d2a2dc37a347d9d5fc4c9077ff7a3614a651aa95ebc4`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/prepare.py`|`5fb03badb6d1462dbf126094b4ba43d5d6d43465a4c42a5fb3dec95eeaff3fb2`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/reload.py`|`4cd7bfc5003bbc767a0c501defc4fbf5e416f1cd1d109a7bf2a6583ad2821dc4`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/report.py`|`6dedc5616c522f8c1e0ce15ade7421f98ca9ea361787d373e8525a9b0f905657`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`ntv2/scripts/setup.py`|`a366ab833d086bdeffe8b27c8dae210a62f53c2721339fbd02dce08ff0218cdb`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|
|`scripts/remaining_followups/launch.py`|`84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/report.py`|`8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/worker.py`|`e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|

<details>
<summary>ntv2/scripts/benchmark.py — 원문 전문 (7,687 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/benchmark.py", "sha256": "be49bb893e908f3438d91e2b6431a2a25782031c258e8a4c72824f4f4060102b", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,os,time,json,gc,random,hashlib
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,torch
from transformers import AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'models/checkpoint';torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'batch':8,'dtype':'FP32','tf32':False,'seed':42,'lr':1e-5,'weight_decay':0.,'epochs':2,'train':512,'validation':128,'test':400,'inference':'MLM model.esm backbone, masked mean including special tokens; MLM logits not computed','finetuning':'official custom EsmForSequenceClassification CLS MLP head; all trainable parameters'}}
def save():write_json(D/'summary.json',S)
def encode(seqs):
 a=tokenizer(seqs,padding=True,truncation=False,return_tensors='pt');lens=a['attention_mask'].sum(1).tolist();assert max(lens)<=2048,(max(lens),len(seqs[0]));return {k:a[k].cuda() for k in ['input_ids','attention_mask']},lens
try:
 inputs=json.loads((R/'data/inputs.json').read_text());tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);tok={}
 for bp in [600,3000,6000,12276]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference']];v=tokenizer(seqs,padding=False,truncation=False)['input_ids'];lens=list(map(len,v));assert max(lens)<=2048;tok[str(bp)]={'min':min(lens),'max':max(lens),'mean':float(np.mean(lens))}
 S['token_audit']=tok;S['input_audit']='data/input_audit.json';save();meter.phase='loading_mlm';t=time.time();model=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True).float().cuda().eval();torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'mlm_total':sum(p.numel() for p in model.parameters()),'embedding_backbone':sum(p.numel() for p in model.esm.parameters()),'mlm_head':sum(p.numel() for p in model.lm_head.parameters())};save()
 def embedding(seqs):
  x,lens=encode(seqs);h=model.esm(**x,return_dict=True).last_hidden_state;m=x['attention_mask'].unsqueeze(-1);z=(h*m).sum(1)/m.sum(1);return z,lens
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():
  for _ in range(3):z,_=embedding([r['sequence'][:600] for r in inputs['long_inference'][:8]])
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;np.save(D/'example_embeddings.npy',z.cpu().numpy());del z
 for bp,n in [(3000,8),(3000,32),(3000,128),(600,32),(6000,32),(12276,32)]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference'][:n]];name=f'inference_bp{bp}_n{n}';t,base=meter.begin(name);repeats=0;lengths=[]
  with torch.inference_mode():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,8):z,lens=embedding(seqs[i:i+8]);lengths.extend(lens);del z
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='inference',bp=bp,unique_inputs=n,repeats=repeats,processed=n*repeats,tokens_min=min(lengths),tokens_mean=float(np.mean(lengths)),tokens_max=max(lengths));row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',json.dumps(row),flush=True);save()
 del model;gc.collect();torch.cuda.empty_cache();torch.manual_seed(42);meter.phase='loading_classifier';t=time.time();model=AutoModelForSequenceClassification.from_pretrained(P,trust_remote_code=True,local_files_only=True,num_labels=2).float().cuda();torch.cuda.synchronize();S['classifier_loading_seconds']=time.time()-t;S['classifier_model_only']=meter.memory();S['parameters']['classifier_total']=sum(p.numel() for p in model.parameters());S['parameters']['classifier_trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad);save()
 def evaluate(rows):
  model.eval();logits=[]
  with torch.no_grad():
   for i in range(0,len(rows),8):
    x,_=encode([r['sequence'] for r in rows[i:i+8]]);z=model(**x).logits;logits.append(z.cpu().numpy())
  p=np.concatenate(logits);y=np.array([r['label'] for r in rows]);pred=p.argmax(1);prob=torch.from_numpy(p).softmax(-1)[:,1].numpy();return {'accuracy':float(accuracy_score(y,pred)),'f1':float(f1_score(y,pred)),'mcc':float(matthews_corrcoef(y,pred)),'auroc':float(roc_auc_score(y,prob))},p
 meter.phase='pretrain_validation';S['initial_validation'],_=evaluate(inputs['validation']);before={'backbone':model.esm.encoder.layer[-1].output.dense.weight.detach().cpu().clone(),'head':next(model.classifier.parameters()).detach().cpu().clone()};opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.);S['epochs']=[];grad=None
 for epoch in range(2):
  model.train();order=np.random.default_rng(42+epoch).permutation(512);losses=[];t,base=meter.begin(f'train_epoch{epoch+1}')
  for step in range(0,512,8):
   rows=[inputs['train'][int(i)] for i in order[step:step+8]];x,lens=encode([r['sequence'] for r in rows]);labels=torch.tensor([r['label'] for r in rows],device='cuda');opt.zero_grad(set_to_none=True);result=model(**x,labels=labels);loss=result.loss;assert torch.isfinite(loss);loss.backward()
   if grad is None:grad={'numel_with_gradient':sum(p.numel() for p in model.parameters() if p.grad is not None),'no_gradient_names':[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]}
   opt.step();losses.append(float(loss.detach()));del result,loss,x,labels
   if step%128==0:print('TRAIN',epoch+1,step,'loss',losses[-1],flush=True)
  row=meter.end(t,base,name=f'train_epoch{epoch+1}',mode='training',unique_inputs=512,repeats=1,processed=512,steps=64,loss_mean=float(np.mean(losses)),losses=losses);row['throughput']=512/row['seconds'];S['conditions'].append(row);meter.phase='validation';val,log=evaluate(inputs['validation']);S['epochs'].append({'epoch':epoch+1,'training':row,'validation':val});print('EPOCH',epoch+1,val,flush=True);save()
 S['gradient_observation']=grad;S['parameter_changes']={'backbone_max_abs':float((model.esm.encoder.layer[-1].output.dense.weight.detach().cpu()-before['backbone']).abs().max()),'head_max_abs':float((next(model.classifier.parameters()).detach().cpu()-before['head']).abs().max())};meter.phase='saving';t=time.time();dest=M/'classifier';model.save_pretrained(dest,safe_serialization=True);tokenizer.save_pretrained(dest);torch.save({'optimizer':opt.state_dict(),'epoch':2,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state()},M/'optimizer_rng.pt');S['save_seconds']=time.time()-t;S['saved_model']=str(dest.relative_to(R));write_json(M/'task.json',{'labels':{'0':'non-enhancer','1':'enhancer'},'settings':S['settings'],'resume_tested':False,'source_checkpoint':'models/checkpoint','input_split':'data/inputs.json'});meter.phase='test';t=time.time();S['test'],logits=evaluate(inputs['test']);torch.cuda.synchronize();S['test_seconds']=time.time()-t;np.save(D/'test_logits.npy',logits);write_json(D/'test_predictions.json',[{'id':r['id'],'label':r['label'],'prediction':int(p.argmax())} for r,p in zip(inputs['test'],logits)]);S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['error']=repr(e);S['status']='failed';save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/discover.py — 원문 전문 (725 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/discover.py", "sha256": "a9fbd2115bef84a39db1d46e118746f0c59ae67016760723252cb8fadc381572", "encoding": "utf-8", "added_final_newline": false} -->
```python
import json,urllib.request
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/ntv2')
for kind,repo in [('models','InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'),('datasets','InstaDeepAI/nucleotide_transformer_downstream_tasks')]:
 meta=json.load(urllib.request.urlopen('https://huggingface.co/api/'+kind+'/'+repo,timeout=60));tree=json.load(urllib.request.urlopen('https://huggingface.co/api/'+kind+'/'+repo+'/tree/'+meta['sha']+'?recursive=true',timeout=60));(R/('models' if kind=='models' else 'data')/'repository.json').write_text(json.dumps({'meta':meta,'tree':tree},indent=2));print(kind,meta['sha'],[(r['path'],r.get('size')) for r in tree if kind=='models' or 'enhancers/' in r['path']],flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/finetune.py — 원문 전문 (5,939 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/finetune.py", "sha256": "8b2557f1f9c116146dbb262496379dac60f5adcd1395e9b890a34eba2d9f0e87", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,os,time,json,gc,random,hashlib
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,torch
from transformers import AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'models/checkpoint';torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'batch':8,'dtype':'FP32','tf32':False,'seed':42,'lr':1e-5,'weight_decay':0.,'epochs':2,'train':512,'validation':128,'test':400,'inference':'MLM model.esm backbone, masked mean including special tokens; MLM logits not computed','finetuning':'official custom EsmForSequenceClassification CLS MLP head; all trainable parameters'}}
def save():write_json(D/'summary.json',S)
def encode(seqs):
 a=tokenizer(seqs,padding=True,truncation=False,return_tensors='pt');lens=a['attention_mask'].sum(1).tolist();assert max(lens)<=2048,(max(lens),len(seqs[0]));return {k:a[k].cuda() for k in ['input_ids','attention_mask']},lens
try:
 inputs=json.loads((R/'data/inputs.json').read_text());tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);old=json.loads((D/'summary.json').read_text());write_json(D/'summary_before_recovery.json',old);S=old;S['status']='recovering_finetuning';S['recovery']='rotary cache created under inference_mode cannot be reused for autograd; validation now uses no_grad; six completed inference conditions retained';S.pop('error',None)
 gc.collect();torch.cuda.empty_cache();torch.manual_seed(42);meter.phase='loading_classifier';t=time.time();model=AutoModelForSequenceClassification.from_pretrained(P,trust_remote_code=True,local_files_only=True,num_labels=2).float().cuda();torch.cuda.synchronize();S['classifier_loading_seconds']=time.time()-t;S['classifier_model_only']=meter.memory();S['parameters']['classifier_total']=sum(p.numel() for p in model.parameters());S['parameters']['classifier_trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad);save()
 def evaluate(rows):
  model.eval();logits=[]
  with torch.no_grad():
   for i in range(0,len(rows),8):
    x,_=encode([r['sequence'] for r in rows[i:i+8]]);z=model(**x).logits;logits.append(z.cpu().numpy())
  p=np.concatenate(logits);y=np.array([r['label'] for r in rows]);pred=p.argmax(1);prob=torch.from_numpy(p).softmax(-1)[:,1].numpy();return {'accuracy':float(accuracy_score(y,pred)),'f1':float(f1_score(y,pred)),'mcc':float(matthews_corrcoef(y,pred)),'auroc':float(roc_auc_score(y,prob))},p
 meter.phase='pretrain_validation';S['initial_validation'],_=evaluate(inputs['validation']);before={'backbone':model.esm.encoder.layer[-1].output.dense.weight.detach().cpu().clone(),'head':next(model.classifier.parameters()).detach().cpu().clone()};opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.);S['epochs']=[];grad=None
 for epoch in range(2):
  model.train();order=np.random.default_rng(42+epoch).permutation(512);losses=[];t,base=meter.begin(f'train_epoch{epoch+1}')
  for step in range(0,512,8):
   rows=[inputs['train'][int(i)] for i in order[step:step+8]];x,lens=encode([r['sequence'] for r in rows]);labels=torch.tensor([r['label'] for r in rows],device='cuda');opt.zero_grad(set_to_none=True);result=model(**x,labels=labels);loss=result.loss;assert torch.isfinite(loss);loss.backward()
   if grad is None:grad={'numel_with_gradient':sum(p.numel() for p in model.parameters() if p.grad is not None),'no_gradient_names':[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]}
   opt.step();losses.append(float(loss.detach()));del result,loss,x,labels
   if step%128==0:print('TRAIN',epoch+1,step,'loss',losses[-1],flush=True)
  row=meter.end(t,base,name=f'train_epoch{epoch+1}',mode='training',unique_inputs=512,repeats=1,processed=512,steps=64,loss_mean=float(np.mean(losses)),losses=losses);row['throughput']=512/row['seconds'];S['conditions'].append(row);meter.phase='validation';val,log=evaluate(inputs['validation']);S['epochs'].append({'epoch':epoch+1,'training':row,'validation':val});print('EPOCH',epoch+1,val,flush=True);save()
 S['gradient_observation']=grad;S['parameter_changes']={'backbone_max_abs':float((model.esm.encoder.layer[-1].output.dense.weight.detach().cpu()-before['backbone']).abs().max()),'head_max_abs':float((next(model.classifier.parameters()).detach().cpu()-before['head']).abs().max())};meter.phase='saving';t=time.time();dest=M/'classifier';model.save_pretrained(dest,safe_serialization=True);tokenizer.save_pretrained(dest);torch.save({'optimizer':opt.state_dict(),'epoch':2,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state()},M/'optimizer_rng.pt');S['save_seconds']=time.time()-t;S['saved_model']=str(dest.relative_to(R));write_json(M/'task.json',{'labels':{'0':'non-enhancer','1':'enhancer'},'settings':S['settings'],'resume_tested':False,'source_checkpoint':'models/checkpoint','input_split':'data/inputs.json'});meter.phase='test';t=time.time();S['test'],logits=evaluate(inputs['test']);torch.cuda.synchronize();S['test_seconds']=time.time()-t;np.save(D/'test_logits.npy',logits);write_json(D/'test_predictions.json',[{'id':r['id'],'label':r['label'],'prediction':int(p.argmax())} for r,p in zip(inputs['test'],logits)]);S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['error']=repr(e);S['status']='failed';save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/inference_remeasure.py — 원문 전문 (4,418 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/inference_remeasure.py", "sha256": "16cbf5cc2d7f96d71cd1d2a2dc37a347d9d5fc4c9077ff7a3614a651aa95ebc4", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,os,time,json,gc,random,hashlib
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,torch
from transformers import AutoModelForMaskedLM,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];M=R/'models'/os.environ['RUN_ID'];M.mkdir(exist_ok=True);P=R/'models/checkpoint';torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
meter=Meter(D);S={'status':'running','conditions':[],'environment':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'settings':{'batch':8,'dtype':'FP32','tf32':False,'seed':42,'lr':1e-5,'weight_decay':0.,'epochs':2,'train':512,'validation':128,'test':400,'inference':'MLM model.esm backbone, masked mean including special tokens; MLM logits not computed','finetuning':'official custom EsmForSequenceClassification CLS MLP head; all trainable parameters'}}
def save():write_json(D/'summary.json',S)
def encode(seqs):
 a=tokenizer(seqs,padding=True,truncation=False,return_tensors='pt');lens=a['attention_mask'].sum(1).tolist();assert max(lens)<=2048,(max(lens),len(seqs[0]));return {k:a[k].cuda() for k in ['input_ids','attention_mask']},lens
try:
 previous=json.loads((D/'summary.json').read_text());write_json(D/'summary_before_remeasurement.json',previous);S=previous;oldparams=dict(S['parameters']);S['conditions']=[r for r in S['conditions'] if r['mode']=='training'];S['status']='remeasuring_inference';
 inputs=json.loads((R/'data/inputs.json').read_text());tokenizer=AutoTokenizer.from_pretrained(P,trust_remote_code=True,local_files_only=True);tok={}
 for bp in [600,3000,6000,12276]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference']];v=tokenizer(seqs,padding=False,truncation=False)['input_ids'];lens=list(map(len,v));assert max(lens)<=2048;tok[str(bp)]={'min':min(lens),'max':max(lens),'mean':float(np.mean(lens))}
 S['token_audit']=tok;S['input_audit']='data/input_audit.json';save();meter.phase='loading_mlm';t=time.time();model=AutoModelForMaskedLM.from_pretrained(P,trust_remote_code=True,local_files_only=True).float().cuda().eval();torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']={'mlm_total':sum(p.numel() for p in model.parameters()),'embedding_backbone':sum(p.numel() for p in model.esm.parameters()),'mlm_head':sum(p.numel() for p in model.lm_head.parameters())};save()
 def embedding(seqs):
  x,lens=encode(seqs);h=model.esm(**x,return_dict=True).last_hidden_state;m=x['attention_mask'].unsqueeze(-1);z=(h*m).sum(1)/m.sum(1);return z,lens
 meter.phase='warmup';t=time.time()
 with torch.inference_mode():
  for _ in range(3):z,_=embedding([r['sequence'][:600] for r in inputs['long_inference'][:8]])
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;np.save(D/'example_embeddings.npy',z.cpu().numpy());del z
 for bp,n in [(3000,8),(3000,32),(3000,128),(600,32),(6000,32),(12276,32)]:
  seqs=[r['sequence'][(12282-bp)//2:(12282-bp)//2+bp] for r in inputs['long_inference'][:n]];name=f'inference_bp{bp}_n{n}';t,base=meter.begin(name);repeats=0;lengths=[]
  with torch.inference_mode():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,8):z,lens=embedding(seqs[i:i+8]);lengths.extend(lens);del z
    torch.cuda.synchronize();repeats+=1
  row=meter.end(t,base,name=name,mode='inference',bp=bp,unique_inputs=n,repeats=repeats,processed=n*repeats,tokens_min=min(lengths),tokens_mean=float(np.mean(lengths)),tokens_max=max(lengths));row['throughput']=row['processed']/row['seconds'];S['conditions'].append(row);print('CONDITION',json.dumps(row),flush=True);save()
 S['parameters']={**oldparams,**S['parameters']};S['resource_files']=[p.name for p in D.glob('resources*.csv')];S['recovery_note']='Initial inference CSV was overwritten by first recovery. Inference six conditions remeasured in fresh process, training retained, each CSV now unique.';S['status']='saved_evaluated_pending_reload';save()
except BaseException as e:S['error']=repr(e);S['status']='failed';save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/prepare.py — 원문 전문 (3,379 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/prepare.py", "sha256": "5fb03badb6d1462dbf126094b4ba43d5d6d43465a4c42a5fb3dec95eeaff3fb2", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,hashlib,collections
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
R=Path('/work');src=Path('/source/external_data/BEND');out={};audit={};official={}
for split in ['train','test']:
 df=pd.read_parquet(R/f'data/enhancers/{split}.parquet');print(split,df.shape,df.columns.tolist(),df.iloc[0].to_dict(),flush=True);official[split]=df
tr=official['train'];te=official['test'];assert {'sequence','label'}<=set(tr.columns)
for df in [tr,te]:assert not df[['sequence','label']].isnull().any().any() and set(df.label.unique())=={0,1}
testseq=set(te.sequence);tr=tr.loc[~tr.sequence.isin(testseq)].drop_duplicates('sequence').copy();te=te.drop_duplicates('sequence').copy();trainids,validids=train_test_split(np.arange(len(tr)),test_size=.15,stratify=tr.label,random_state=42)
def subset(df,indices,n):
 ids,_=train_test_split(indices,train_size=n,stratify=df.iloc[indices].label,random_state=42) if len(indices)>n else (indices,[])
 return [{'id':str(df.iloc[i].get('id',df.index[i])),'sequence':df.iloc[i].sequence,'label':int(df.iloc[i].label),'sequence_sha256':hashlib.sha256(df.iloc[i].sequence.encode()).hexdigest()} for i in ids]
out['train']=subset(tr,trainids,512);out['validation']=subset(tr,validids,128);out['test']=subset(te,np.arange(len(te)),min(400,len(te)))
sets=[set(r['sequence_sha256'] for r in out[k]) for k in out];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
manifest=json.loads((src/'prepared/20261012_001503_n1000_seed42/manifest.json').read_text());fa=src/'genomes/GRCh38.primary_assembly.genome.fa';idx={}
for line in Path(str(fa)+'.fai').read_text().splitlines():
 name,length,offset,bases,width,*_=line.split('\t');idx[name]=tuple(map(int,[length,offset,bases,width]))
def getseq(f,chrom,start,end):
 if chrom not in idx:chrom=chrom.removeprefix('chr')
 length,off,bases,width=idx[chrom];assert start>=0 and end<=length
 lo=off+start//bases*width+start%bases;hi=off+(end-1)//bases*width+(end-1)%bases+1;f.seek(lo);return f.read(hi-lo).replace(b'\n',b'').replace(b'\r',b'').decode().upper()
long=[]
with fa.open('rb') as f:
 for pos in manifest['positions']:
  center=pos['center'];seq=getseq(f,pos['chromosome'],center-6141,center+6141)
  if len(seq)!=12282 or not set(seq)<=set('ACGT'):continue
  if pos['strand']=='-':seq=seq.translate(str.maketrans('ACGT','TGCA'))[::-1]
  long.append(dict(id=pos['id'],sequence=seq,chromosome=pos['chromosome'],center=center,strand=pos['strand']))
  if len(long)==128:break
assert len(long)==128;out['long_inference']=long
for k,rows in out.items():audit[k]={'n':len(rows),'length_min':min(len(r['sequence']) for r in rows),'length_max':max(len(r['sequence']) for r in rows),'labels':dict(collections.Counter(r.get('label','unlabeled') for r in rows)),'first_example':rows[0]}
(R/'data/inputs.json').write_text(json.dumps(out,indent=2));(R/'data/input_audit.json').write_text(json.dumps({'splits':audit,'official_train_rows':len(official['train']),'official_test_rows':len(official['test']),'train_after_test_overlap_and_duplicates_removed':len(tr),'policy':'official test untouched for selection; train/val stratified seed42; exact sequence hash disjoint; long unlabeled reference regions inference only'},indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a!='first_example'} for k,v in audit.items()},indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/reload.py — 원문 전문 (1,382 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/reload.py", "sha256": "4cd7bfc5003bbc767a0c501defc4fbf5e416f1cd1d109a7bf2a6583ad2821dc4", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,sys,time
sys.path.insert(0,'/support')
from gpu_support import write_json
import torch,numpy as np
from transformers import AutoModelForSequenceClassification,AutoTokenizer
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());p=R/s['saved_model'];torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;m=AutoModelForSequenceClassification.from_pretrained(p,trust_remote_code=True,local_files_only=True).float().cuda().eval();tok=AutoTokenizer.from_pretrained(p,trust_remote_code=True,local_files_only=True);rows=json.loads((R/'data/inputs.json').read_text())['test'];pred=[]
with torch.inference_mode():
 for i in range(0,len(rows),8):
  x=tok([r['sequence'] for r in rows[i:i+8]],padding=True,truncation=False,return_tensors='pt');pred.append(m(**{k:x[k].cuda() for k in ['input_ids','attention_mask']}).logits.cpu().numpy())
y=np.concatenate(pred);ref=np.load(D/'test_logits.npy');diff=float(np.abs(y-ref).max());r={'independent_process':True,'test_inputs':len(rows),'max_abs_difference':diff,'predictions_exact_match':bool(np.array_equal(y.argmax(-1),ref.argmax(-1))),'passed':bool(diff<=1e-5)};write_json(D/'reload.json',r);assert r['passed'];s['reload']=r;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(r),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/report.py — 원문 전문 (9,157 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/report.py", "sha256": "6dedc5616c522f8c1e0ce15ade7421f98ca9ea361787d373e8525a9b0f905657", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import json,csv,hashlib
R=Path('/home/hyein/ai-benchmark/ntv2');run=(R/'results/LATEST').read_text().strip();D=R/'results'/run;s=json.loads((D/'summary.json').read_text());audit=json.loads((R/'data/input_audit.json').read_text());meta=json.loads((R/'models/repository.json').read_text());out=[]
def a(t=''):out.append(t)
def table(h,rows):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rows:a('| '+' | '.join(map(str,r))+' |')
 a()
def f(x):return 'N/A' if x is None else f'{x:.3f}'
a('# Nucleotide Transformer V2 — DGX Spark');a();a(f'실행 `{run}` · 상태 **{s["status"]}** · [원본 summary](../results/{run}/summary.json) · [실행 로그](../results/{run}/benchmark.log) · [GPU·메모리 시계열](../results/{run}/resources.csv)');a();a('## 모델과 기준 조건');a();a('공식 `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`, 리비전 `'+meta['meta']['sha']+'`. V2 500M을 대표로 선택했으며 V1·V3로 바꾸지 않았다. 다운로드된 model.safetensors의 SHA-256은 공식 LFS oid와 일치함을 검증했고 전체 출처는 `results/sources.json`에 보관했다.');a();a('실제 파라미터: '+json.dumps(s.get('parameters',{}),ensure_ascii=False));a();a('MLM 전체 모델을 로드하되 임베딩 측정은 `model.esm` backbone의 최종 hidden state만 계산해 masked mean(특수토큰 포함)으로B×1024를 만든다. MLM head와 모든 층 hidden state를 계산/저장하는 경로와 메모리·시간이 다르다. 분류는 공식 custom EsmForSequenceClassification의 CLS MLP head를 사용하고 전 학습 가능 backbone·head를 함께 갱신했다.');a();a('환경: '+json.dumps(s['environment']));a();a('batch8,FP32,TF32=False,seed42,8CPU threads. 공식 fine-tuning notebook의 batch8/lr1e-5/2epoch를 참고하되 NT-v2와 enhancer binary 소규모 subset에 적용했다. notebook의1000maxsteps나 전체 데이터 재현 실험이 아니다. 추론에서도 batch8을 고정하며 배치 증가 sweep을 수행하지 않았다.');a();b=s['model_only_postload'];table(['상태','allocated GiB','reserved GiB','RSS GiB','system available GiB'],[['MLM 적재 직후/첫 입력 전',*[f(b[k]) for k in ['allocated_gib','reserved_gib','rss_gib','system_available_gib']]]]);a('이론 FP32 가중치 용량은 MLM 파라미터수×4byte/2³⁰이며 실측과 구분한다. 공유 RAM'+f(b['system_total_gib'])+'GiB에서 RSS와CUDA를 더하지 않는다. 적재 '+f(s['loading_seconds'])+'초; warmup '+f(s.get('warmup_seconds'))+'초. CPU 입력 JSON 로딩은 적재 전에 수행했다.');a();a('## 실제 입력·라벨·분할');a();rows=[]
for k,v in audit['splits'].items():rows.append([k,v['n'],f"{v['length_min']}–{v['length_max']}nt",v['labels']])
table(['split','고유 서열','실제 길이','라벨 분포'],rows);a('공식 NT downstream enhancer train에서 seed42 stratified 학습/검증을 분리하고 공식 test400개는 평가 전용으로 유지했다. train/test exact sequence 중복과 train 내 중복을 제거한 뒤 부분집합을 선택했다. genomic homology 및 사전학습 genome 중복까지 제거한 것은 아니다.0=non-enhancer,1=enhancer. 원본 길이는199–200nt로 임의 정규화나 길이 증폭을 하지 않았다.');a();r=audit['splits']['train']['first_example'];a('실제 학습 예: ID `'+r['id']+'`, label `'+str(r['label'])+'`, '+str(len(r['sequence']))+'nt');a();a('```text\n'+r['sequence']+'\n```');a();a('입력은6-mer 중심 토큰화 후 int64 input_ids/attention_mask의B×실제token수, 라벨은B 정수다. [검사 결과](../data/input_audit.json)와 선택 서열·split·해시 `data/inputs.json`을 보관했다.');a();a('길이/개수 실험은 별도의 GRCh38 참조 유전체 동일 구간128개를 사용한다. BEND에서 이미 확인한 좌표 중심으로12,282nt를 추출하고 ACGT 외 문자가 없는 실제 구간을 택했다. 같은 구간의 중심 crop600/3000/6000/12276nt를 사용하며 무라벨 추론 전용이다. 음의 가닥은 reverse-complement했다. 짧은 enhancer 학습 데이터의 정확도와 이 긴 무라벨 서열을 혼동하지 않는다.');a();a('## 입력별 GPU 사용률·메모리');a();rs=[];flat=[]
for r in s['conditions']:
 bp=r.get('bp','199–200');n=r['unique_inputs'];m=r['memory'];rs.append([r['name'],bp,n,r.get('tokens_mean','학습 입력'),r['repeats'],f(r['seconds']),f(r['throughput']),f(r['gpu_util_avg'])+'/'+f(r['gpu_util_max']),f(r['baseline']['allocated_gib']),f(m['peak_allocated_gib']),f(r['peak_delta_gib'])]);flat.append({k:v for k,v in r.items() if k not in ['memory','baseline','losses']});flat[-1].update(allocated_baseline_gib=r['baseline']['allocated_gib'],peak_allocated_gib=m['peak_allocated_gib'],peak_reserved_gib=m['peak_reserved_gib'])
table(['조건','nt','고유 개수','평균token','반복','초','건/초','GPU 평균/최대%','시작GiB','peakGiB','증가GiB'],rs)
if flat:
 keys=list(dict.fromkeys(k for r in flat for k in r))
 with (R/'reports/comparison.csv').open('w',newline='') as f0:w=csv.DictWriter(f0,fieldnames=keys);w.writeheader();w.writerows(flat)
a('실측 tokenizer 결과: '+json.dumps(s.get('token_audit',{})));a();a('추론은 최소15초 조건 반복, 고유8/32/128개와 총처리수는 구분했다. 길이 비교는같은32개, 개수 비교는같은3000nt의 중첩 부분집합이다. truncation=False이며 실제 token이2048이하임을 assert했다. 각 조건 CUDA peak를 reset하고 cache를 비운 뒤 같은 run의 시작 allocated를 기준으로 추가량을 계산했다.');a();a('시간은 메모리에 준비한 원본 서열의 CPU 토큰화·GPU 전송·forward·pooling까지 포함하며 결과는 GPU tensor다. 반복 구간의 CPU 결과 반환/디스크 저장은 제외한다. 학습은 토큰화·전송·forward·loss·backward·AdamW까지 포함하고 epoch별 validation은 제외했다. CUDA 경계를 synchronize했다. 디스크 입력 준비와 파일 저장은 제외하며 저장은 별도 시간으로 기록했다. compile은 warmup에 섞여 별도값N/A다.');a();a('GPU-Util은 장치 활동시간 비율로 FLOPS/메모리 비율이 아니다.0.5초 대기+조회 구간이 측정 안에 완전히 포함된 표본만 집계하며 디스플레이 영향이 포함될 수 있다. RSS/available/allocated/reserved 시계열은 sampled값이다. 짧은 epoch의 평균은 표본수와 함께 해석한다.');a()
if 'test' in s:
 a('## 파인튜닝·저장·독립 test');a();table(['epoch','train loss','validation Accuracy','F1','MCC','AUROC'],[[r['epoch'],f(r['training']['loss_mean']),*[f(r['validation'][k]) for k in ['accuracy','f1','mcc','auroc']]] for r in s['epochs']]);a('최종 epoch2를 고정 평가했고 test로 epoch/조건을 선택하지 않았다. test400개: '+json.dumps(s['test']));a();a('실제 gradient 관찰: '+json.dumps(s['gradient_observation']));a();a('backbone과head 변경 검증: '+json.dumps(s['parameter_changes']));a();a('모델 저장: `'+s['saved_model']+'`; 저장 '+f(s['save_seconds'])+'초. optimizer/RNG/epoch도 저장했지만 학습 재개는 검증하지 않았다. test logits와 실제 예측 라벨을 보존했다. 독립 프로세스 재로드: '+json.dumps(s.get('reload',{'status':'pending'})));a();a('검증 Accuracy0.8906에 비해 공식 test Accuracy0.6975로 낮아, 검증 성능을 외부 일반화 성능으로 대체해 보고하지 않는다. 첫 독립 재로드에서 TF32 기본 설정 불일치로logits차이0.00575가 나타났으나 분류 라벨은 모두 일치했고, 원 FP32/TF32=False로 맞춰 재검증했다.');a();a('소규모2epoch 실행 검증으로, 전체 benchmark 재현이나 최고의 enhancer 정확도를 주장하지 않는다.');a()
a('복구 이력: 첫 학습 진입에서 rotary cache/inference_mode 충돌이 발생해 validation을 no_grad로 바꾸고 학습을 새 체크포인트에서 복구했다. 첫 복구 시 초기 추론 시계열 파일을 덮어쓴 문제가 있어, 추론6조건을 별도 프로세스로 재계측하고 학습은 재실행하지 않았다. 최종 표는 재계측 추론과 완료된2epoch 학습이다. 자원 CSV는 단계마다 별도 파일을 보존한다. 실패 로그와 복구 전 summary도 남겼다.');a();a('## 해석과 공식 출처');a();a('길이가 커질 때 attention 중간값 등 추가 메모리가 커질 수 있지만, 데이터 개수가 늘어도 batch정책이 같으면 순간 peak가비례 증가하지 않는다. 공유RAM90% 채우기는실험목표가아니다.');a();a('- [공식 NT-v2 모델 카드](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)\n- [공식 소스·학습 예제 링크](https://github.com/instadeepai/nucleotide-transformer)\n- [공식 enhancer 데이터](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)\n- [공식 분류 fine-tuning notebook](https://github.com/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb)');(R/'reports/NTv2.md').write_text('\n'.join(out)+'\n');print(R/'reports/NTv2.md')
````
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>ntv2/scripts/setup.py — 원문 전문 (538 bytes)</summary>


<!-- RESTORE_FILE {"path": "ntv2/scripts/setup.py", "sha256": "a366ab833d086bdeffe8b27c8dae210a62f53c2721339fbd02dce08ff0218cdb", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,subprocess,json,importlib.metadata
from pathlib import Path
R=Path('/work');(R/'constraints.txt').write_text('torch=='+importlib.metadata.version('torch')+'\n');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True)
cmd=['/work/.venv/bin/pip','install','--cache-dir','/work/.cache/pip','-c','/work/constraints.txt','transformers==4.44.2','pyarrow','scikit-learn','psutil'];code=subprocess.call(cmd);(R/'results/setup_exit.json').write_text(json.dumps({'exit_code':code}));sys.exit(code)
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
