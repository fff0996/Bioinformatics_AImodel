# Nucleotide Transformer V2 최종 통합 보고서

기본 실험과 모든 후속 실험을 이 문서에 함께 수록했다. 다른 기본/후속 문서를 따로 읽을 필요 없이 아래 요약과 본문을 보면 된다.

## 1. 입력과 메모리 결론

DNA서열 → 주로6-mer 토큰 → 임베딩 또는enhancer 분류. 공개500M checkpoint.

**목표는 메모리 할당80% 조건 확인**이다. 전체공유RAM121.688GiB의80%는약97.35GiB다. GPU-Util은별도연산활동지표이며이목표의성공판정에쓰지않는다.

|작업|관측메모리 peak GiB|공유RAM대비|해당 입력·배치·정밀도|80% 확인/제약|
|---|---|---|---|---|---|
|추론|**21.592**|17.74%|실제6000nt 서열128개. GPU입력128×1001토큰,FP32.|80% 미확인; 확보한고유서열128개한계|
|전체 backbone 파인튜닝|**9.252**|7.60%|199~200nt enhancer서열,train512/val128/test400. batch8·FP32·2epoch,분류head와backbone학습.|학습batch확대 미실시|

메모리는PyTorch의경우CUDA allocated peak다. reserved는예약량으로별도다. **AlphaFold2는컴파일·warmup등을포함한JAX프로세스누적peak**이므로CUDA구간peak와같은정의로비교하지않는다. RSS와CUDA를더하지않는다. 표는시험한조건의관측값이며모델의절대최대나모든모델의80%달성을의미하지않는다.

## 2. 기본 정보·입력 검사·추론·학습·평가

## Nucleotide Transformer V2 — DGX Spark


실행 `ntv2_20261027_201332_251243` · 상태 **complete** · [원본 summary](../results/ntv2_20261027_201332_251243/summary.json) · 실행 로그 (로컬 보관: `../results/ntv2_20261027_201332_251243/benchmark.log`) · [GPU·메모리 시계열](../results/ntv2_20261027_201332_251243/resources.csv)

### 모델과 기준 조건

공식 `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`, 리비전 `06615c1660c892fc199840c18123f8385b3542a8`. V2 500M을 대표로 선택했으며 V1·V3로 바꾸지 않았다. 다운로드된 model.safetensors의 SHA-256은 공식 LFS oid와 일치함을 검증했고 전체 출처는 `results/sources.json`에 보관했다.

실제 파라미터: {"mlm_total": 498345436, "embedding_backbone": 493084113, "mlm_head": 5261323, "classifier_total": 494135763, "classifier_trainable": 494135763}

MLM 전체 모델을 로드하되 임베딩 측정은 `model.esm` backbone의 최종 hidden state만 계산해 masked mean(특수토큰 포함)으로B×1024를 만든다. MLM head와 모든 층 hidden state를 계산/저장하는 경로와 메모리·시간이 다르다. 분류는 공식 custom EsmForSequenceClassification의 CLS MLP head를 사용하고 전 학습 가능 backbone·head를 함께 갱신했다.

환경: {"torch": "2.13.0a0+8145d630e8.nv26.06", "transformers": "4.44.2", "cuda": "13.3", "gpu": "NVIDIA GB10"}

batch8,FP32,TF32=False,seed42,8CPU threads. 공식 fine-tuning notebook의 batch8/lr1e-5/2epoch를 참고하되 NT-v2와 enhancer binary 소규모 subset에 적용했다. notebook의1000maxsteps나 전체 데이터 재현 실험이 아니다. 추론에서도 batch8을 고정하며 배치 증가 sweep을 수행하지 않았다.

| 상태 | allocated GiB | reserved GiB | RSS GiB | system available GiB |
| --- | --- | --- | --- | --- |
| MLM 적재 직후/첫 입력 전 | 1.857 | 1.873 | 1.125 | 111.229 |

이론 FP32 가중치 용량은 MLM 파라미터수×4byte/2³⁰이며 실측과 구분한다. 공유 RAM121.688GiB에서 RSS와CUDA를 더하지 않는다. 적재 11.959초; warmup 0.596초. CPU 입력 JSON 로딩은 적재 전에 수행했다.

### 실제 입력·라벨·분할

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

입력은6-mer 중심 토큰화 후 int64 input_ids/attention_mask의B×실제token수, 라벨은B 정수다. [검사 결과](../data/input_audit.json)와 선택 서열·split·해시 `data/inputs.json`을 보관했다.

길이/개수 실험은 별도의 GRCh38 참조 유전체 동일 구간128개를 사용한다. BEND에서 이미 확인한 좌표 중심으로12,282nt를 추출하고 ACGT 외 문자가 없는 실제 구간을 택했다. 같은 구간의 중심 crop600/3000/6000/12276nt를 사용하며 무라벨 추론 전용이다. 음의 가닥은 reverse-complement했다. 짧은 enhancer 학습 데이터의 정확도와 이 긴 무라벨 서열을 혼동하지 않는다.

### 입력별 GPU 사용률·메모리

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

### 파인튜닝·저장·독립 test

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

### 해석과 공식 출처

길이가 커질 때 attention 중간값 등 추가 메모리가 커질 수 있지만, 데이터 개수가 늘어도 batch정책이 같으면 순간 peak가비례 증가하지 않는다. 공유RAM90% 채우기는실험목표가아니다.

- [공식 NT-v2 모델 카드](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)
- [공식 소스·학습 예제 링크](https://github.com/instadeepai/nucleotide-transformer)
- [공식 enhancer 데이터](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks)
- [공식 분류 fine-tuning notebook](https://github.com/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb)

## 3. 후속 실험 통합 — 배치 확대 및 자원 검증

## ntv2 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_143713_869625` · 종료사유 `available_unique_input_limit`

### 입력과 고정 조건

{
  "source": "existing real GRCh38 regions, central6000nt crops -> native ~1001tokens",
  "unique_available_inputs": 128,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

### 배치별 측정

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

### 결과 해석과 검증

최대시험성공batch 128. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) · [기존 모델 보고서 디렉터리](.)

## 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.
