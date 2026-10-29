# DNABERT-2 최종 통합 보고서

## 1. 모델과 실제 입력

DNA 염기서열 → BPE토큰 → 임베딩/18개histone 표지 예측. 염기수와토큰수는다르다.

실제파일·입력/정답예시·전처리와모델구조설명은이문서5절에전문을포함했다. 추론데이터와지도학습label을구분하며,세포/유전자·서열/토큰·이미지/타일의단위를섞지않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|배치 추론|고유1개; BEND 실제4000bp DNA|batch1, input_ids:[1, 851] / attention_mask:[1, 851],FP32|359회/총359건|15.01초|23.92입력/초|0.619 GiB / 0.51%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|배치 추론|고유512개; BEND 실제4000bp DNA|batch512, input_ids:[512, 851] / attention_mask:[512, 851],FP32|2회/총1024건|46.60초|21.98입력/초|60.007 GiB / 49.31%|출력finite·저장재로드·첫입력일치검증;80%미도달|
|기존전체파인튜닝|실제BEND1000개,4000bp,18개multi-label|8×697~851토큰,FP32|1epoch;총1000서열처리|116.87초|8.56서열/초|10.571 GiB / 8.69%|해당실행독립정확도미평가;80%미도달|

## 4. 결과 해석

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### DNABERT-2 기존 결과와 GPU 사용률 보완


보완 run: `supplement_20261027_195301_706540`; 상태: complete. 기존 원본은 `/home/hyein/dnabert2-benchmark/dnabert2_eval`에 보존했다.

#### 모델·실행 조건

공식 DNABERT-2-117M, revision `7bce263b15377fc15361f52cfab88f8b586abda0`, 실제 encoder 파라미터 117,068,544. FP32, batch8, TF32=False, seed42,8 CPU threads. 토큰화는 padding=True/truncation=False, masked mean pooling(특수 토큰 포함)이다. 18개 출력의 선형 head는 13,842개 파라미터를 추가한다. Bert pooler가 초기화되었다는 공식 loader 경고가 있으나 이 과제는 hidden-state masked mean을 사용해 pooler 출력은 사용하지 않는다. MLM head도 임베딩 모델에는 적재하지 않는다.

환경: {"torch": "2.14.1+cu130", "transformers": "4.29.2", "cuda": "13.0", "gpu": "NVIDIA GB10"}

기존 conda 환경을 원래 /workspace 경로로 읽기 전용 mount하여 사용했다. Triton/FlashAttention이 없어 기존과 같은 PyTorch attention fallback으로 실행됐다. 다른 모델/환경과의 속도 차이를 입력 길이의 효과로만 해석하면 안 된다.

입력 forward 전 모델만 GPU에 올린 allocated **0.436118GiB**, reserved 0.479GiB, RSS 1.100GiB, 시스템 available 112.949GiB. 적재 0.888초. 이론 FP32 파라미터 용량 0.436114GiB와 실측은 구분한다.

#### 실제 입력과 정답

기존 확보한 BEND histone_modification train의 동일1,000개 genomic region을 재사용했다. 참조 GRCh38, 음의 가닥 reverse-complement, A/C/G/T 및 길이·고유 ID·길이별 동일 ID 순서를 검사했다. 원본 train/validation/test를 바꾸지 않았다. 원본 region의 18개 히스톤 표지 multi-label을 0/1 타깃으로 사용한다. bp 수는 토큰 수가 아니며 BPE 결과는 아래에 별도 표기한다. 길이별 crop의 라벨은 원본 region에서 상속하므로 잘라낸 범위의 염기별 정답이라고 주장하지 않는다.

실제2,000bp 입력의 앞120bp:

```text
CTTTTGAATATGAATGTCCTAATTTAATTCTTTGGACCTTTGAGGGGAGGACACTATCACTTCTACATATGTAGAGAAGTAAAAGTCTCATAGATCCATCTTGTCTTTAAAAATATATGG
```

ID: `chr3:152302071-152302583:+`; 양성 라벨 인덱스: `[4, 16]`. 전체 서열·라벨 예시는 input_examples.json (로컬 보관: `../results/supplement_20261027_195301_706540/input_examples.json`). 학습 입력은 input_ids/attention_mask의 B×실제토큰길이, 정답은 B×18 float32 multi-hot이다.

#### 입력별 GPU 사용률·메모리 동시 측정

| 모드 | bp | 고유 개수 | token 최소/평균/최대 | 반복 | 초 | 건/초 | GPU 평균/최대% | 시작 GiB | peak GiB | 증가 GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inference | 2000 | 32 | 352/387.0/419 | 29 | 15.25 | 60.83 | 94.1/95 | 0.479 | 0.745 | 0.266 |
| inference | 2000 | 128 | 352/391.0/429 | 8 | 17.08 | 59.94 | 94.7/95 | 0.479 | 0.756 | 0.277 |
| inference | 2000 | 1000 | 349/391.8/438 | 1 | 16.71 | 59.85 | 94.4/95 | 0.479 | 0.765 | 0.286 |
| inference | 512 | 128 | 87/103.4/118 | 36 | 15.34 | 300.45 | 89.2/94 | 0.479 | 0.532 | 0.053 |
| inference | 1000 | 128 | 173/198.2/222 | 17 | 15.19 | 143.28 | 92.9/95 | 0.479 | 0.589 | 0.110 |
| inference | 4000 | 128 | 709/777.9/846 | 3 | 16.94 | 22.66 | 94.6/95 | 0.479 | 1.420 | 0.941 |
| training_resource_probe | 512 | 128 | 87/103.4/118 | 1 | 2.35 | 54.48 | 69.0/91 | 0.467 | 2.261 | 1.793 |
| training_resource_probe | 1000 | 128 | 173/198.2/222 | 1 | 3.62 | 35.32 | 86.0/94 | 0.499 | 3.095 | 2.596 |
| training_resource_probe | 2000 | 128 | 352/391.0/429 | 1 | 7.31 | 17.50 | 89.5/95 | 0.499 | 5.118 | 4.620 |
| training_resource_probe | 4000 | 128 | 709/777.9/846 | 1 | 16.30 | 7.85 | 92.9/95 | 0.499 | 10.376 | 9.878 |

추론은 최소15초 동안 전체 subset을 반복했다. 개수는2,000bp에서32/128/1000 중첩 부분집합, 길이는 동일128개에서512/1000/2000/4000bp를 비교했다. 토큰 길이 분포는 subset마다 조금 다르므로 개수만의 완전한 통제 실험은 아니다. batch8을 고정하고 데이터를 통째로 GPU에 올리지 않았다.

학습 행은 기존 완료된1,000개·1epoch 학습에 빠져 있던 사용률을 보완하는 **별도128개·16step 자원 probe**다. 각 길이는 동일 초기 체크포인트·seed·AdamW(lr2e-5,weight_decay.01)로 시작한다. 1,000개 학습이나 새로운 test 평가를 완료했다는 뜻은 아니다. probe 가중치는 최종 모델로 채택하지 않고 기존 저장·재로드 검증된 학습 모델을 보존했다. 짧은 학습 구간은 반복하지 않았으므로 표본 수와 초기 optimizer 할당 영향을 고려한다.

512bp 학습2.35초/4표본,1000bp 학습3.62초/7표본은 짧은 관찰로 평균 GPU 사용률의 대표성이 낮다. nvidia-smi 자체 관측창 때문에 앞 구간 활동이 섞일 수 있다. 2000bp7.31초/13표본,4000bp16.30초/30표본으로 구간 길이도 다르다. 이 값들을 일반적인 학습 사용률의 확정 비교로 쓰지 않는다.

추론 warmup 뒤 baseline이 모델 적재량보다 큰 것은 실제 실행 후 유지되는 버퍼/상태를 포함한 관찰값이다. 정확한 내부 원인은 계측하지 않았다. 입력당 순수 버퍼 증가량이라고 표현하지 않는다. 체크포인트와 코드 SHA-256은 verification.json에 저장했다.

시계열은 [resources.csv](../results/supplement_20261027_195301_706540/RECORDS.md#file-6fb2fed4477e)에 저장했다. nvidia-smi GPU 활동시간 비율을0.5초 대기+조회 간격으로 표본화하며 최대 FLOPS/메모리%가 아니다. 장치 전체에 디스플레이 영향이 포함될 수 있다. 같은 run의 현재 allocated에서 peak 증가량을 계산했고 peak는 조건마다 reset했다. 학습 증가는 gradient·optimizer·중간값을 포함한다. RSS/available/swap은 시계열 표본이다. Spark 공유 RAM121.688GiB에서 CUDA와RSS를 더하면 안 된다.

시간은 CPU메모리에 준비한 데이터의 토큰화·전송·forward·pooling, 학습은 backward·optimizer까지 포함하며 CUDA 경계는 동기화했다. 디스크 읽기/저장 시간은 제외하고 별도 미측정(N/A), compile은 분리되지 않았다. 추론3배치 warmup은 별도 수행했다. 학습은 초기 optimizer 생성을 포함한 실제 첫 step부터 측정한다.

#### 기존 완료 결과의 위치

기존2,000bp 학습1,000개·batch8 FP32는 평균391.834 tokens(349–438), peak5.172566GiB였다. 4,000bp 학습 peak10.5712GiB(공유 RAM 약8.69%)였다. 이 과거 run에는 GPU-Util이 없어서 이번 별도 probe 사용률을 과거 값처럼 붙이지 않는다.

원본 `bend_length_results/*training*/saved_model` 및 `finetune_results/*/saved_model`에 기존 학습 모델과 save/reload 결과가 있다. 이번 보완은 새로운 정확도 향상이나 독립 test 성능을 주장하지 않는다. 전체 기존 결과는 삭제하거나 덮어쓰지 않았다.

#### scGPT도 이미 완료

scGPT 기본 추론·파인튜닝·저장·재로드·test·길이 비교는 완료돼 있다. 새로 필요한 것은 입력 개수와 GPU 사용률의 보완이다.

[scGPT 저장 모델 test summary](../../scgpt/results/finetuned_test_20261023_034345_99523/RECORDS.md#file-9cd02c9a4f70) · [기존 길이 비교](../../scgpt/results/length_compare_20261023_040313_501423/RECORDS.md#file-d50c252c8230)

test4,218세포 중 알려진 라벨4,213세포 기준 Accuracy0.9380489, 학습13클래스 macroF1 0.5349519, weightedF1 0.9311712. test-only 라벨5세포는 해당 지표에서 제외했다.

#### 공식 출처

- [DNABERT-2 공식 모델 카드](https://huggingface.co/zhihan1996/DNABERT-2-117M)
- [공식 구현](https://github.com/MAGICS-LAB/DNABERT_2)
- [BEND 공식 저장소](https://github.com/nyuolab/BEND)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### dnabert2 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_143041_359149` · 종료사유 `predicted_reservation_exceeds_available_budget`

#### 입력과 고정 조건

{
  "source": "existing BEND prepared n1000, fixed4000bp, actual native tokenizer length verified worker",
  "unique_available_inputs": 1000,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

#### 배치별 측정

측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.

|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|359|359|15.0071|23.922|0.4361|0.6192|0.1831|0.6641|0.5089|90.7143|94.0|1.8413|111.3421|
|2|174|348|15.0025|23.1962|0.4361|0.7334|0.2973|0.8438|0.6027|91.7143|95.0|1.5495|111.4481|
|4|86|344|15.0109|22.9167|0.4361|0.9664|0.5302|1.1504|0.7941|95.0|96.0|1.5807|111.1316|
|8|43|344|15.0924|22.7929|0.4361|1.4328|0.9967|1.5137|1.1774|95.3571|96.0|1.5808|110.813|
|16|22|352|15.3937|22.8665|0.4361|2.3592|1.9231|2.4961|1.9387|95.2857|96.0|1.5779|109.7566|
|32|11|352|15.2773|23.0408|0.4361|4.2193|3.7832|4.4277|3.4673|95.3929|96.0|1.5804|107.7379|
|64|6|384|16.6337|23.0856|0.4361|7.939|7.5029|8.375|6.5241|95.5|96.0|1.5816|103.6364|
|96|4|384|16.8844|22.7429|0.4361|11.6582|11.2221|12.3262|9.5804|89.7667|96.0|1.5821|99.7311|
|128|3|384|17.1004|22.4556|0.4361|15.3807|14.9446|16.2695|12.6395|93.9032|96.0|1.5508|95.6986|
|192|2|384|17.51|21.9304|0.4361|22.8197|22.3836|24.6309|18.7527|93.1935|96.0|1.5529|87.2993|
|256|2|512|23.2079|22.0615|0.4361|30.2544|29.8182|32.6816|24.8623|94.7619|96.0|1.576|79.0801|
|384|2|768|34.6935|22.1367|0.4361|45.1315|44.6953|48.7656|37.088|92.0635|96.0|1.5468|62.9258|
|512|2|1024|46.5984|21.975|0.4361|60.0071|59.571|64.8574|49.3124|92.1566|96.0|1.5761|46.4286|

#### 결과 해석과 검증

최대시험성공batch 512. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://github.com/MAGICS-LAB/DNABERT_2) · [기존 모델 보고서 디렉터리](.)

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
|`dnabert2/scripts/launch.py`|`50ac752612bb8c985280ea32f5c56930c31b862ab10e9bd1a488aa64333d7122`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`dnabert2/scripts/report.py`|`a5c859f864dae6c0ca6c46831fc6619b44cca201f037c2d59dc5eaf30d13ba50`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`dnabert2/scripts/supplement.py`|`02b6e2f8ab6f9560bfa47dfddbd74b2f8d4ac6ef2e7c03a7b0b28331e10ad95c`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|
|`scripts/remaining_followups/launch.py`|`84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/report.py`|`8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`scripts/remaining_followups/worker.py`|`e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|

<details>
<summary>dnabert2/scripts/launch.py — 원문 전문 (998 bytes)</summary>


<!-- RESTORE_FILE {"path": "dnabert2/scripts/launch.py", "sha256": "50ac752612bb8c985280ea32f5c56930c31b862ab10e9bd1a488aa64333d7122", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,datetime,json
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/dnabert2');assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();run='supplement_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LATEST').write_text(run+'\n');cmd=['sg','docker','-c',f'docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/dnabert2-benchmark:/workspace:ro -v {R}:/work -e RUN_ID={run} -e HF_MODULES_CACHE=/work/.cache/modules -e HF_HOME=/work/.cache/huggingface nvcr.io/nvidia/pytorch:26.06-py3 /workspace/miniforge/envs/dnabert2/bin/python -u /work/scripts/supplement.py'];(D/'command.json').write_text(json.dumps(cmd,indent=2));print(run,flush=True)
with (D/'run.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/'exit_code.json').write_text(json.dumps({'exit_code':code}));print('EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>dnabert2/scripts/report.py — 원문 전문 (8,429 bytes)</summary>


<!-- RESTORE_FILE {"path": "dnabert2/scripts/report.py", "sha256": "a5c859f864dae6c0ca6c46831fc6619b44cca201f037c2d59dc5eaf30d13ba50", "encoding": "utf-8", "added_final_newline": false} -->
````python
from pathlib import Path
import json,csv
R=Path('/home/hyein/ai-benchmark');B=R/'dnabert2';run=(B/'results/LATEST').read_text().strip();D=B/'results'/run;s=json.loads((D/'summary.json').read_text());lines=[]
def a(x=''):lines.append(x)
def table(h,rs):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rs:a('| '+' | '.join(map(str,r))+' |')
 a()
a('# DNABERT-2 기존 결과와 GPU 사용률 보완');a();a(f'보완 run: `{run}`; 상태: {s.get("status","running")}. 기존 원본은 `/home/hyein/dnabert2-benchmark/dnabert2_eval`에 보존했다.');a();a('## 모델·실행 조건');a();a(f'공식 DNABERT-2-117M, revision `{s["model_revision"].strip()}`, 실제 encoder 파라미터 {s["parameters"]:,}. FP32, batch8, TF32=False, seed42,8 CPU threads. 토큰화는 padding=True/truncation=False, masked mean pooling(특수 토큰 포함)이다. 18개 출력의 선형 head는 13,842개 파라미터를 추가한다. Bert pooler가 초기화되었다는 공식 loader 경고가 있으나 이 과제는 hidden-state masked mean을 사용해 pooler 출력은 사용하지 않는다. MLM head도 임베딩 모델에는 적재하지 않는다.');a();a('환경: '+json.dumps(s['environment']));a();a('기존 conda 환경을 원래 /workspace 경로로 읽기 전용 mount하여 사용했다. Triton/FlashAttention이 없어 기존과 같은 PyTorch attention fallback으로 실행됐다. 다른 모델/환경과의 속도 차이를 입력 길이의 효과로만 해석하면 안 된다.');a();b=s['model_only_postload'];a(f'입력 forward 전 모델만 GPU에 올린 allocated **{b["allocated_gib"]:.6f}GiB**, reserved {b["reserved_gib"]:.3f}GiB, RSS {b["rss_gib"]:.3f}GiB, 시스템 available {b["available_gib"]:.3f}GiB. 적재 {s["load_seconds"]:.3f}초. 이론 FP32 파라미터 용량 {s["parameters"]*4/2**30:.6f}GiB와 실측은 구분한다.');a();a('## 실제 입력과 정답');a();a('기존 확보한 BEND histone_modification train의 동일1,000개 genomic region을 재사용했다. 참조 GRCh38, 음의 가닥 reverse-complement, A/C/G/T 및 길이·고유 ID·길이별 동일 ID 순서를 검사했다. 원본 train/validation/test를 바꾸지 않았다. 원본 region의 18개 히스톤 표지 multi-label을 0/1 타깃으로 사용한다. bp 수는 토큰 수가 아니며 BPE 결과는 아래에 별도 표기한다. 길이별 crop의 라벨은 원본 region에서 상속하므로 잘라낸 범위의 염기별 정답이라고 주장하지 않는다.');a();examples=json.loads((D/'input_examples.json').read_text());ex=examples['2000'];a('실제2,000bp 입력의 앞120bp:');a();a('```text\n'+ex['sequence'][:120]+'\n```');a();a('ID: `'+ex['id']+'`; 양성 라벨 인덱스: `'+ex['labels']+'`. 전체 서열·라벨 예시는 [input_examples.json](../results/'+run+'/input_examples.json). 학습 입력은 input_ids/attention_mask의 B×실제토큰길이, 정답은 B×18 float32 multi-hot이다.');a();a('## 입력별 GPU 사용률·메모리 동시 측정');a();rows=[];flat=[]
for r in s['conditions']:
 base=r['baseline']['allocated_gib'];peak=r['memory']['peak_allocated_gib'];u='N/A' if r['gpu_util_avg'] is None else f'{r["gpu_util_avg"]:.1f}/{r["gpu_util_max"]:.0f}';rows.append([r['mode'],r['bp'],r['unique_inputs'],f'{r["tokens_min"]}/{r["tokens_mean"]:.1f}/{r["tokens_max"]}',r['repeats'],f'{r["seconds"]:.2f}',f'{r["throughput"]:.2f}',u,f'{base:.3f}',f'{peak:.3f}',f'{peak-base:.3f}']);flat.append({'mode':r['mode'],'bp':r['bp'],'unique_inputs':r['unique_inputs'],'repeats':r['repeats'],'processed':r['processed'],'seconds':r['seconds'],'throughput':r['throughput'],'tokens_mean':r['tokens_mean'],'gpu_util_avg':r['gpu_util_avg'],'gpu_util_max':r['gpu_util_max'],'gpu_baseline_allocated_gib':base,'gpu_peak_allocated_gib':peak,'gpu_peak_delta_gib':peak-base,'gpu_peak_reserved_gib':r['memory']['peak_reserved_gib'],'rss_endpoint_gib':r['memory']['rss_gib']})
table(['모드','bp','고유 개수','token 최소/평균/최대','반복','초','건/초','GPU 평균/최대%','시작 GiB','peak GiB','증가 GiB'],rows)
with (B/'reports/comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
a('추론은 최소15초 동안 전체 subset을 반복했다. 개수는2,000bp에서32/128/1000 중첩 부분집합, 길이는 동일128개에서512/1000/2000/4000bp를 비교했다. 토큰 길이 분포는 subset마다 조금 다르므로 개수만의 완전한 통제 실험은 아니다. batch8을 고정하고 데이터를 통째로 GPU에 올리지 않았다.');a();a('학습 행은 기존 완료된1,000개·1epoch 학습에 빠져 있던 사용률을 보완하는 **별도128개·16step 자원 probe**다. 각 길이는 동일 초기 체크포인트·seed·AdamW(lr2e-5,weight_decay.01)로 시작한다. 1,000개 학습이나 새로운 test 평가를 완료했다는 뜻은 아니다. probe 가중치는 최종 모델로 채택하지 않고 기존 저장·재로드 검증된 학습 모델을 보존했다. 짧은 학습 구간은 반복하지 않았으므로 표본 수와 초기 optimizer 할당 영향을 고려한다.');a();a('512bp 학습2.35초/4표본,1000bp 학습3.62초/7표본은 짧은 관찰로 평균 GPU 사용률의 대표성이 낮다. nvidia-smi 자체 관측창 때문에 앞 구간 활동이 섞일 수 있다. 2000bp7.31초/13표본,4000bp16.30초/30표본으로 구간 길이도 다르다. 이 값들을 일반적인 학습 사용률의 확정 비교로 쓰지 않는다.');a();a('추론 warmup 뒤 baseline이 모델 적재량보다 큰 것은 실제 실행 후 유지되는 버퍼/상태를 포함한 관찰값이다. 정확한 내부 원인은 계측하지 않았다. 입력당 순수 버퍼 증가량이라고 표현하지 않는다. 체크포인트와 코드 SHA-256은 verification.json에 저장했다.');a();a('시계열은 [resources.csv](../results/'+run+'/resources.csv)에 저장했다. nvidia-smi GPU 활동시간 비율을0.5초 대기+조회 간격으로 표본화하며 최대 FLOPS/메모리%가 아니다. 장치 전체에 디스플레이 영향이 포함될 수 있다. 같은 run의 현재 allocated에서 peak 증가량을 계산했고 peak는 조건마다 reset했다. 학습 증가는 gradient·optimizer·중간값을 포함한다. RSS/available/swap은 시계열 표본이다. Spark 공유 RAM121.688GiB에서 CUDA와RSS를 더하면 안 된다.');a();a('시간은 CPU메모리에 준비한 데이터의 토큰화·전송·forward·pooling, 학습은 backward·optimizer까지 포함하며 CUDA 경계는 동기화했다. 디스크 읽기/저장 시간은 제외하고 별도 미측정(N/A), compile은 분리되지 않았다. 추론3배치 warmup은 별도 수행했다. 학습은 초기 optimizer 생성을 포함한 실제 첫 step부터 측정한다.');a();a('## 기존 완료 결과의 위치');a();a('기존2,000bp 학습1,000개·batch8 FP32는 평균391.834 tokens(349–438), peak5.172566GiB였다. 4,000bp 학습 peak10.5712GiB(공유 RAM 약8.69%)였다. 이 과거 run에는 GPU-Util이 없어서 이번 별도 probe 사용률을 과거 값처럼 붙이지 않는다.');a();a('원본 `bend_length_results/*training*/saved_model` 및 `finetune_results/*/saved_model`에 기존 학습 모델과 save/reload 결과가 있다. 이번 보완은 새로운 정확도 향상이나 독립 test 성능을 주장하지 않는다. 전체 기존 결과는 삭제하거나 덮어쓰지 않았다.');a();a('## scGPT도 이미 완료');a();a('scGPT 기본 추론·파인튜닝·저장·재로드·test·길이 비교는 완료돼 있다. 새로 필요한 것은 입력 개수와 GPU 사용률의 보완이다.');a();a('[scGPT 저장 모델 test summary](../../scgpt/results/finetuned_test_20261023_034345_99523/summary.json) · [기존 길이 비교](../../scgpt/results/length_compare_20261023_040313_501423/comparison.csv)');a();a('test4,218세포 중 알려진 라벨4,213세포 기준 Accuracy0.9380489, 학습13클래스 macroF1 0.5349519, weightedF1 0.9311712. test-only 라벨5세포는 해당 지표에서 제외했다.');a();a('## 공식 출처');a();a('- [DNABERT-2 공식 모델 카드](https://huggingface.co/zhihan1996/DNABERT-2-117M)\n- [공식 구현](https://github.com/MAGICS-LAB/DNABERT_2)\n- [BEND 공식 저장소](https://github.com/nyuolab/BEND)');(B/'reports/DNABERT2.md').write_text('\n'.join(lines)+'\n');print(B/'reports/DNABERT2.md')
````
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>dnabert2/scripts/supplement.py — 원문 전문 (6,662 bytes)</summary>


<!-- RESTORE_FILE {"path": "dnabert2/scripts/supplement.py", "sha256": "02b6e2f8ab6f9560bfa47dfddbd74b2f8d4ac6ef2e7c03a7b0b28331e10ad95c", "encoding": "utf-8", "added_final_newline": false} -->
```python
import csv,json,time,threading,subprocess,os,gc,hashlib
from pathlib import Path
import numpy as np,psutil,torch
from transformers import AutoModel,AutoTokenizer,BertConfig
R=Path('/work');SRC=Path('/workspace/dnabert2_eval');D=R/'results'/os.environ['RUN_ID'];D.mkdir(exist_ok=True);G=2**30;torch.set_num_threads(8);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
modelpath=SRC/'model';tokenizer=AutoTokenizer.from_pretrained(str(modelpath),trust_remote_code=True,local_files_only=True);config=BertConfig.from_pretrained(str(modelpath),local_files_only=True)
S={'environment':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'model_revision':(SRC/'model_revision.txt').read_text(),'settings':{'batch':8,'dtype':'FP32','tf32':False,'seed':42,'pooling':'masked mean including special tokens','training':'resource probe only; original completed saved training retained','learning_rate':2e-5,'weight_decay':.01},'conditions':[]};samples=[];phase='initial';stop=threading.Event();swap0=psutil.swap_memory().used

def mem():return dict(allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,peak_allocated_gib=torch.cuda.max_memory_allocated()/G,peak_reserved_gib=torch.cuda.max_memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,available_gib=psutil.virtual_memory().available/G,total_gib=psutil.virtual_memory().total/G)
def monitor():
 with (D/'resources.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['time_unix','phase','gpu_util','rss_gib','available_gib','swap_gib'])
  while not stop.is_set():
   t=time.time();p=phase
   try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
   except Exception:u=None
   v=psutil.virtual_memory();row=[t,p,u,psutil.Process().memory_info().rss/G,v.available/G,psutil.swap_memory().used/G];samples.append(row);w.writerow(row);f.flush()
   if v.available<8*G or psutil.swap_memory().used>swap0+2*G:os._exit(86)
   stop.wait(.5)
def save(): (D/'summary.json').write_text(json.dumps(S,indent=2))
th=threading.Thread(target=monitor,daemon=True);th.start()
try:
 phase='loading';t=time.time();encoder=AutoModel.from_pretrained(str(modelpath),config=config,trust_remote_code=True,local_files_only=True).float().cuda();torch.cuda.synchronize();S['load_seconds']=time.time()-t;S['model_only_postload']=mem();S['parameters']=sum(p.numel() for p in encoder.parameters());save()
 prepared=SRC/'external_data/BEND/prepared/20261012_001503_n1000_seed42';data={}
 for bp in [512,1000,2000,4000]:
  rows=list(csv.DictReader((prepared/f'train_n1000_bp{bp}.csv').open()));assert len(rows)==1000 and len({r['id'] for r in rows})==1000 and all(len(r['sequence'])==bp and set(r['sequence'])<=set('ACGT') for r in rows);data[bp]=rows
 assert all([r['id'] for r in data[b]]==[r['id'] for r in data[512]] for b in data)
 (D/'input_examples.json').write_text(json.dumps({str(bp):rows[0] for bp,rows in data.items()},indent=2));S['input_source']=str(prepared);S['input_ids_sha256']=hashlib.sha256('\n'.join(r['id'] for r in data[512]).encode()).hexdigest()
 def batch(rows):
  x=tokenizer([r['sequence'] for r in rows],padding=True,truncation=False,return_tensors='pt');lens=x['attention_mask'].sum(1).tolist();return {k:x[k].cuda() for k in ['input_ids','attention_mask']},lens
 def pool(x):
  y=encoder(**x)[0];m=x['attention_mask'].unsqueeze(-1).to(y.dtype);return (y*m).sum(1)/m.sum(1).clamp_min(1)
 encoder.eval();phase='warmup';t=time.time()
 with torch.inference_mode():
  for _ in range(3):x,_=batch(data[2000][:8]);z=pool(x);del x,z
 torch.cuda.synchronize();S['warmup_seconds']=time.time()-t
 for bp,n in [(2000,32),(2000,128),(2000,1000),(512,128),(1000,128),(4000,128)]:
  phase=f'inference_bp{bp}_n{n}';torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=mem();t=time.time();repeats=0;lengths=[]
  with torch.inference_mode():
   while repeats==0 or time.time()-t<15:
    for i in range(0,n,8):
     x,lens=batch(data[bp][i:i+8]);z=pool(x);lengths.extend(lens);del x,z
    torch.cuda.synchronize();repeats+=1
  end=time.time();us=[r[2] for r in samples if r[1]==phase and r[0]>=t and r[2] is not None];row={'condition':phase,'mode':'inference','bp':bp,'unique_inputs':n,'repeats':repeats,'processed':n*repeats,'seconds':end-t,'throughput':n*repeats/(end-t),'tokens_min':min(lengths),'tokens_mean':float(np.mean(lengths)),'tokens_max':max(lengths),'baseline':base,'memory':mem(),'gpu_util_avg':float(np.mean(us)) if us else None,'gpu_util_max':max(us) if us else None,'samples':len(us)};S['conditions'].append(row);print(json.dumps(row),flush=True);save()
 # Fresh initialization for each training resource probe; saved full training results are reused.
 del encoder;gc.collect();torch.cuda.empty_cache()
 for bp in [512,1000,2000,4000]:
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);phase=f'training_load_{bp}';encoder=AutoModel.from_pretrained(str(modelpath),config=config,trust_remote_code=True,local_files_only=True).float().cuda();head=torch.nn.Linear(config.hidden_size,18).float().cuda();encoder.train();opt=torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()),lr=2e-5,weight_decay=.01);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();base=mem();phase=f'training_bp{bp}_n128';t=time.time();losses=[];lengths=[]
  for i in range(0,128,8):
   rows=data[bp][i:i+8];x,lens=batch(rows);lengths.extend(lens);targets=torch.zeros(len(rows),18,device='cuda')
   for j,r in enumerate(rows):targets[j,json.loads(r['labels'])]=1
   opt.zero_grad(set_to_none=True);y=head(pool(x));loss=torch.nn.functional.binary_cross_entropy_with_logits(y,targets);loss.backward();opt.step();losses.append(float(loss.detach()));del x,y,loss,targets
  torch.cuda.synchronize();end=time.time();us=[r[2] for r in samples if r[1]==phase and r[0]>=t and r[2] is not None];row={'condition':phase,'mode':'training_resource_probe','bp':bp,'unique_inputs':128,'repeats':1,'processed':128,'steps':16,'seconds':end-t,'throughput':128/(end-t),'tokens_min':min(lengths),'tokens_mean':float(np.mean(lengths)),'tokens_max':max(lengths),'baseline':base,'memory':mem(),'gpu_util_avg':float(np.mean(us)) if us else None,'gpu_util_max':max(us) if us else None,'samples':len(us),'losses':losses};S['conditions'].append(row);print(json.dumps(row),flush=True);save();del encoder,head,opt;gc.collect();torch.cuda.empty_cache()
 S['status']='complete';save()
finally:stop.set();th.join(timeout=5);save()
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


### 참고 PDF의 기존 DNABERT-2 실행 코드 원문

아래는PDF에서명시한기존실행코드의현재보관원문이다. 원래호스트위치는`/home/hyein/dnabert2-benchmark/dnabert2_eval/`,컨테이너위치는`/workspace/dnabert2_eval/`다. Git복원경로는dnabert2/legacy_scripts로표시하며실행시원래layout/mount에맞춰둔다. 실행당시snapshot해시가확인되지않은경우당시와바이트동일하다고단정하지않는다.

<details>
<summary>benchmark.py — 원래 보관 코드 전문</summary>

원문SHA256: `7504c537f3238f0718e729618ba1abd47d75dc8e7851a872a3020e2394e432cb`

<!-- RESTORE_FILE {"path": "dnabert2/legacy_scripts/benchmark.py", "sha256": "7504c537f3238f0718e729618ba1abd47d75dc8e7851a872a3020e2394e432cb", "encoding": "utf-8", "added_final_newline": false} -->
```python
import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1000)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--bp", type=int, default=300)
p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
p.add_argument("--save-embeddings", action="store_true")
a = p.parse_args()
assert a.n > 0 and a.batch > 0 and a.bp > 0

root = Path("/workspace/dnabert2_eval")
run = root / "results" / (
    f"{time.strftime('%Y%m%d_%H%M%S')}_n{a.n}_b{a.batch}"
    f"_bp{a.bp}_{a.precision}_{time.time_ns()}"
)
run.mkdir(parents=True)
proc = psutil.Process()
proc.cpu_percent()
stop = threading.Event()
state = {"phase": "startup", "done": 0}
started = time.perf_counter()
peaks = {}
fields = [
    "seconds", "phase", "done", "cpu_percent", "rss_gib",
    "system_used_gib", "system_available_gib",
    "gpu_allocated_gib", "gpu_reserved_gib",
    "gpu_util_percent", "gpu_temp_c", "gpu_power_w"
]

def gpu_stats():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, timeout=3, check=True
        )
        return [float(x.strip()) for x in r.stdout.strip().split(",")]
    except Exception:
        return ["N/A"] * 3

def sample(writer):
    vm = psutil.virtual_memory()
    util, temp, power = gpu_stats()
    row = dict(zip(fields, [
        round(time.perf_counter() - started, 3), state["phase"],
        state["done"], proc.cpu_percent(), proc.memory_info().rss / 2**30,
        vm.used / 2**30, vm.available / 2**30,
        torch.cuda.memory_allocated() / 2**30,
        torch.cuda.memory_reserved() / 2**30, util, temp, power
    ]))
    writer.writerow(row)
    phase_peaks = peaks.setdefault(row["phase"], {})
    for k in fields[3:]:
        if isinstance(row[k], (int, float)):
            phase_peaks[k] = max(phase_peaks.get(k, 0), row[k])
    print(
        f"[{row['phase']}] {row['done']:,}/{a.n:,} "
        f"| elapsed={row['seconds']:.1f}s "
        f"| CPU={row['cpu_percent']:.0f}% "
        f"| RAM={row['rss_gib']:.2f}GiB "
        f"| GPU alloc/reserved={row['gpu_allocated_gib']:.2f}/"
        f"{row['gpu_reserved_gib']:.2f}GiB "
        f"| GPU={util}% {temp}C {power}W", flush=True
    )

def monitor():
    with (run / "resources.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            sample(writer)
            f.flush()
            stop.wait(2)
        sample(writer)

summary = {
    "settings": vars(a),
    "memory_note": "Shared system memory; do not add RSS and GPU allocation.",
    "cpu_note": "Process CPU percent can exceed 100 on multiple cores.",
    "resource_sampling_seconds": 2,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
}
thread = threading.Thread(target=monitor, daemon=True)
thread.start()

try:
    state["phase"] = "data_loading"
    ds = load_from_disk(str(root / "raw_dataset"))
    data = ds["train"].filter(lambda r: r["task"] == "promoter_all")
    del ds
    assert len(data) > 0
    assert {"sequence", "label"}.issubset(data.column_names)
    assert all(len(s) == 300 for s in data["sequence"])
    summary["source_rows"] = len(data)
    summary["repeated_input"] = a.n > len(data)
    summary["length_mode"] = (
        "native" if a.bp == 300 else
        "cropped" if a.bp < 300 else "synthetically_repeated"
    )
    summary["input_bases"] = a.n * a.bp

    state["phase"] = "model_loading"
    torch.cuda.synchronize()
    rss_before = proc.memory_info().rss
    t = time.perf_counter()
    model_path = str(root / "model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    config = BertConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path, config=config,
        trust_remote_code=True, local_files_only=True
    ).to("cuda").eval()
    torch.cuda.synchronize()
    summary["model_load_seconds"] = time.perf_counter() - t
    summary["parameters"] = sum(x.numel() for x in model.parameters())
    summary["post_load_rss_gib"] = proc.memory_info().rss / 2**30
    summary["load_rss_delta_gib"] = (
        proc.memory_info().rss - rss_before
    ) / 2**30
    summary["post_load_gpu_allocated_gib"] = (
        torch.cuda.memory_allocated() / 2**30
    )
    summary["loading_gpu_peak_gib"] = (
        torch.cuda.max_memory_allocated() / 2**30
    )

    def encode(offset, count):
        idx = [(offset + j) % len(data) for j in range(count)]
        seqs = data.select(idx)["sequence"]
        seqs = [(s * ((a.bp + len(s) - 1) // len(s)))[:a.bp]
                for s in seqs]
        return tokenizer(
            seqs, padding=True, truncation=False, return_tensors="pt"
        )

    def forward(tokens):
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=a.precision == "bf16"
        ):
            outputs = model(
                input_ids=tokens["input_ids"].to("cuda"),
                attention_mask=tokens["attention_mask"].to("cuda")
            )
            hidden = outputs[0]
            mask = tokens["attention_mask"].to("cuda").unsqueeze(-1)
            return (hidden * mask).sum(1) / mask.sum(1)

    state["phase"] = "warmup"
    for _ in range(3):
        emb = forward(encode(0, min(a.batch, a.n)))
        del emb
    torch.cuda.synchronize()

    saved = None
    if a.save_embeddings:
        saved = np.lib.format.open_memmap(
            run / "embeddings.npy", mode="w+", dtype="float32",
            shape=(a.n, config.hidden_size)
        )

    state["phase"] = "inference"
    torch.cuda.reset_peak_memory_stats()
    token_min, token_max, token_total = float("inf"), 0, 0
    t = time.perf_counter()
    for offset in range(0, a.n, a.batch):
        count = min(a.batch, a.n - offset)
        tokens = encode(offset, count)
        lengths = tokens["attention_mask"].sum(1)
        token_min = min(token_min, int(lengths.min()))
        token_max = max(token_max, int(lengths.max()))
        token_total += int(lengths.sum())
        emb = forward(tokens)
        if saved is not None:
            saved[offset:offset + count] = emb.float().cpu().numpy()
        del emb, tokens
        state["done"] = offset + count
    if saved is not None:
        saved.flush()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t

    summary.update(
        status="success",
        inference_seconds=elapsed,
        sequences_per_second=a.n / elapsed,
        tokens_min=token_min,
        tokens_max=token_max,
        tokens_mean=token_total / a.n,
        inference_gpu_peak_allocated_gib=(
            torch.cuda.max_memory_allocated() / 2**30
        ),
        inference_gpu_peak_reserved_gib=(
            torch.cuda.max_memory_reserved() / 2**30
        ),
        embeddings_saved=a.save_embeddings,
        timing_scope="tokenization + transfer + forward + optional output write",
    )
except Exception as e:
    summary.update(status="failed", error=f"{type(e).__name__}: {e}")
    raise
finally:
    stop.set()
    thread.join()
    summary["sampled_phase_peaks"] = peaks
    summary["output_bytes"] = sum(
        f.stat().st_size for f in run.iterdir() if f.is_file()
    )
    (run / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("Results:", run, flush=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details>
<summary>finetune_benchmark.py — 원래 보관 코드 전문</summary>

원문SHA256: `6d97afd1b90e7ba0fa5f2e0feacc6bce19e4f7fdf0868b9561306f2a9dddf56f`

<!-- RESTORE_FILE {"path": "dnabert2/legacy_scripts/finetune_benchmark.py", "sha256": "6d97afd1b90e7ba0fa5f2e0feacc6bce19e4f7fdf0868b9561306f2a9dddf56f", "encoding": "utf-8", "added_final_newline": false} -->
```python
import argparse
import csv
import json
import random
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import psutil
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer, BertConfig

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1000)
args = parser.parse_args()
assert args.n > 0 and args.n % 2 == 0, "--n은 양의 짝수여야 합니다."

root = Path("/workspace/dnabert2_eval")
out = root / "finetune_results" / (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + f"_n{args.n}_b8_bp300_fp32_{time.time_ns()}"
)
out.mkdir(parents=True)

random.seed(42)
torch.manual_seed(42)
assert torch.cuda.is_available(), "CUDA를 사용할 수 없습니다."
torch.cuda.manual_seed_all(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 먼저 실제 데이터 형식과 길이·라벨을 확인합니다.
ds = load_from_disk(str(root / "raw_dataset"))["train"]
assert {"sequence", "label", "task"}.issubset(ds.column_names)
ds = ds.filter(lambda row: row["task"] == "promoter_all")

lengths = Counter(map(len, ds["sequence"]))
counts = Counter(ds["label"])
print("컬럼:", ds.column_names, flush=True)
print("서열 길이:", dict(lengths), flush=True)
print("라벨별 개수:", dict(counts), flush=True)
assert set(lengths) == {300}
assert set(counts) == {0, 1}

groups = {
    label: [i for i, value in enumerate(ds["label"]) if value == label]
    for label in (0, 1)
}
indices = []
for label in (0, 1):
    random.shuffle(groups[label])
    assert len(groups[label]) >= args.n // 2, "보유 데이터가 부족합니다."
    indices.extend(groups[label][:args.n // 2])
random.shuffle(indices)

selected = ds.select(indices)
sequences = selected["sequence"]
labels = selected["label"]
print("이번 학습:", len(selected), "개 /", dict(Counter(labels)), flush=True)

model_path = str(root / "model")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, local_files_only=True
)
config = BertConfig.from_pretrained(model_path, local_files_only=True)
encoder = AutoModel.from_pretrained(
    model_path, config=config,
    trust_remote_code=True, local_files_only=True
)

class Classifier(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled)

model = Classifier(encoder).float().cuda().train()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-5, weight_decay=0.01
)
criterion = torch.nn.CrossEntropyLoss()

summary = {
    "settings": {
        "n": args.n, "batch": 8, "bp": 300,
        "precision": "fp32", "epochs": 1,
        "learning_rate": 2e-5, "seed": 42,
        "mode": "full_finetuning",
        "pooling": "masked_mean_including_special_tokens",
        "tf32": False,
    },
    "gpu": torch.cuda.get_device_name(0),
    "parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ),
    "timing_scope": "tokenization + transfer + forward + backward + optimizer",
    "ram_sampling_interval_seconds": 0.2,
}

process = psutil.Process()
stop = threading.Event()
samples = []

def monitor():
    while not stop.is_set():
        samples.append({
            "seconds": time.perf_counter() - started,
            "rss_gib": process.memory_info().rss / 2**30,
        })
        stop.wait(0.2)

thread = None
loss_sum = 0.0
done = 0
token_lengths = []
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()

try:
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    for start in range(0, args.n, 8):
        texts = sequences[start:start + 8]
        batch = tokenizer(
            texts, padding=True, truncation=False, return_tensors="pt"
        )
        token_lengths.extend(batch["attention_mask"].sum(1).tolist())
        inputs = {
            key: batch[key].cuda()
            for key in ("input_ids", "attention_mask")
        }
        targets = torch.tensor(
            labels[start:start + 8], dtype=torch.long, device="cuda"
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach().item() * len(texts)
        done += len(texts)
        if done % 200 == 0 or done == args.n:
            print(
                f"학습 {done}/{args.n} | loss={loss.item():.4f}",
                flush=True
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary.update({
        "status": "success",
        "training_seconds": elapsed,
        "sequences_per_second": done / elapsed,
        "mean_training_loss": loss_sum / done,
        "tokens_min": min(token_lengths),
        "tokens_max": max(token_lengths),
    })

except Exception as error:
    summary.update({"status": "failed", "error": repr(error)})
    raise

finally:
    stop.set()
    if thread is not None:
        thread.join()
    samples.append({
        "seconds": time.perf_counter() - started,
        "rss_gib": process.memory_info().rss / 2**30,
    })
    summary.update({
        "processed_sequences": done,
        "training_gpu_peak_allocated_gib":
            torch.cuda.max_memory_allocated() / 2**30,
        "training_gpu_peak_reserved_gib":
            torch.cuda.max_memory_reserved() / 2**30,
        "training_ram_peak_sampled_gib":
            max(row["rss_gib"] for row in samples),
    })
    with (out / "resources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seconds", "rss_gib"])
        writer.writeheader()
        writer.writerows(samples)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("결과 폴더:", out, flush=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details>
<summary>bend_length_benchmark.py — 원래 보관 코드 전문</summary>

원문SHA256: `b3a77a777a2df4c405ec9b88ab4518ad9c9802d2aec0fad16f9153fa8116cd5a`

<!-- RESTORE_FILE {"path": "dnabert2/legacy_scripts/bend_length_benchmark.py", "sha256": "b3a77a777a2df4c405ec9b88ab4518ad9c9802d2aec0fad16f9153fa8116cd5a", "encoding": "utf-8", "added_final_newline": false} -->
```python
import argparse
import csv
import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
from transformers import AutoModel, AutoTokenizer, BertConfig

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--bp", type=int, required=True)
parser.add_argument("--mode", choices=["inference", "training"], required=True)
args = parser.parse_args()

root = Path("/workspace/dnabert2_eval")
data_dir = Path(args.data_dir)
path = data_dir / f"train_n1000_bp{args.bp}.csv"

# 학습 전에 실제 입력 모양을 확인합니다.
with path.open() as f:
    reader = csv.DictReader(f)
    assert {"id", "sequence", "labels"}.issubset(reader.fieldnames)
    rows = list(reader)

assert len(rows) == 1000
assert len({r["id"] for r in rows}) == 1000
sequences = [r["sequence"] for r in rows]
label_lists = [json.loads(r["labels"]) for r in rows]
assert all(len(s) == args.bp for s in sequences)
assert all(set(s) <= set("ACGT") for s in sequences)
assert all(
    labels and all(isinstance(x, int) and 0 <= x < 18 for x in labels)
    for labels in label_lists
)

manifest = json.loads((data_dir / "manifest.json").read_text())
assert [r["id"] for r in rows] == [
    r["id"] for r in manifest["positions"]
]
assert label_lists == [r["labels"] for r in manifest["positions"]]

targets_cpu = torch.zeros(1000, 18, dtype=torch.float32)
for i, labels in enumerate(label_lists):
    targets_cpu[i, labels] = 1

print(
    f"입력 확인: {len(rows)}개 / {args.bp}bp / "
    f"라벨 행렬 {tuple(targets_cpu.shape)} / {args.mode}",
    flush=True
)

assert torch.cuda.is_available()
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model_path = str(root / "model")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, local_files_only=True
)
config = BertConfig.from_pretrained(model_path, local_files_only=True)
encoder = AutoModel.from_pretrained(
    model_path, config=config,
    trust_remote_code=True, local_files_only=True
)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        if args.mode == "training":
            self.head = torch.nn.Linear(config.hidden_size, 18)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled) if args.mode == "training" else pooled

model = Model().float().cuda()
training = args.mode == "training"
model.train(training)
optimizer = (
    torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    if training else None
)
criterion = torch.nn.BCEWithLogitsLoss()

def encode(texts):
    batch = tokenizer(
        texts, padding=True, truncation=False, return_tensors="pt"
    )
    lengths = batch["attention_mask"].sum(1).tolist()
    inputs = {
        k: batch[k].cuda() for k in ("input_ids", "attention_mask")
    }
    return inputs, lengths

# 추론만 준비 실행합니다. 학습은 가중치 변경 없이 바로 측정합니다.
if not training:
    def warmup():
        with torch.inference_mode():
            for _ in range(3):
                inputs, _ = encode(sequences[:8])
                model(**inputs)
        torch.cuda.synchronize()
    warmup()

out = root / "bend_length_results" / (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + f"_{args.mode}_n1000_b8_bp{args.bp}_fp32_{time.time_ns()}"
)
out.mkdir(parents=True)

summary = {
    "settings": {
        "dataset": "BEND histone_modification",
        "mode": args.mode,
        "n": 1000,
        "batch": 8,
        "bp": args.bp,
        "precision": "fp32",
        "tf32": False,
        "epochs": 1 if training else None,
        "learning_rate": 2e-5 if training else None,
        "seed": 42,
        "num_labels": 18,
        "pooling": "masked_mean_including_special_tokens",
    },
    "gpu": torch.cuda.get_device_name(0),
    "parameters": sum(p.numel() for p in model.parameters()),
    "data_file": str(path),
    "sample_ids_sha256": hashlib.sha256(
        "\n".join(r["id"] for r in rows).encode()
    ).hexdigest(),
    "timing_scope": (
        "tokenization + transfer + forward + backward + optimizer"
        if training else "tokenization + transfer + forward + pooling"
    ),
    "inference_warmup_batches": 0 if training else 3,
    "ram_sampling_interval_seconds": 0.2,
    "shared_memory_denominator_gib": 121,
}

process = psutil.Process()
stop = threading.Event()
samples, losses, token_lengths = [], [], []

def monitor():
    while not stop.is_set():
        samples.append({
            "seconds": time.perf_counter() - started,
            "rss_gib": process.memory_info().rss / 2**30,
        })
        stop.wait(0.2)

done = 0
loss_sum = 0.0
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
thread = threading.Thread(target=monitor, daemon=True)
thread.start()

try:
    for start in range(0, 1000, 8):
        inputs, lengths = encode(sequences[start:start + 8])
        token_lengths.extend(lengths)
        count = len(lengths)

        if training:
            targets = targets_cpu[start:start + count].cuda()
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            value = loss.detach().item()
            loss_sum += value * count
            losses.append({
                "step": len(losses) + 1,
                "processed_sequences": done + count,
                "loss": value,
            })
            del logits, loss, targets
        else:
            with torch.inference_mode():
                embeddings = model(**inputs)
            del embeddings

        del inputs
        done += count
        if done % 200 == 0:
            print(f"{args.mode} {args.bp}bp: {done}/1000", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary.update({
        "status": "success",
        "seconds": elapsed,
        "sequences_per_second": done / elapsed,
        "tokens_min": min(token_lengths),
        "tokens_mean": sum(token_lengths) / len(token_lengths),
        "tokens_max": max(token_lengths),
    })
    if training:
        summary["mean_training_loss"] = loss_sum / done

except Exception as error:
    summary.update({"status": "failed", "error": repr(error)})
    raise

finally:
    stop.set()
    thread.join()
    samples.append({
        "seconds": time.perf_counter() - started,
        "rss_gib": process.memory_info().rss / 2**30,
    })
    peak = torch.cuda.max_memory_allocated() / 2**30
    summary.update({
        "processed_sequences": done,
        "gpu_peak_allocated_gib": peak,
        "gpu_allocated_percent_of_shared_memory": peak / 121 * 100,
        "ram_peak_sampled_gib": max(r["rss_gib"] for r in samples),
    })
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    with (out / "resources.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seconds", "rss_gib"])
        writer.writeheader()
        writer.writerows(samples)
    if training:
        with (out / "loss.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "processed_sequences", "loss"]
            )
            writer.writeheader()
            writer.writerows(losses)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("결과 폴더:", out, flush=True)

# 그래프 생성 시간은 자원 측정 구간에서 제외합니다.
if training and losses:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [r["step"] for r in losses]
        y = [r["loss"] for r in losses]
        smooth = [
            sum(y[max(0, i - 9):i + 1]) / len(y[max(0, i - 9):i + 1])
            for i in range(len(y))
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, y, alpha=0.4, linewidth=0.8, label="Batch loss")
        ax.plot(x, smooth, linewidth=2, label="Moving average (10 batches)")
        ax.set(
            xlabel="Optimizer step",
            ylabel="Training loss (BCE)",
            title=f"DNABERT-2 | BEND | {args.bp} bp | n=1,000 | 1 epoch",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "training_loss.png", dpi=200)
        fig.savefig(out / "training_loss.svg")
        plt.close(fig)
        print("Loss 그래프 저장 완료", flush=True)
    except ImportError:
        print("matplotlib 없음: loss.csv는 저장됐습니다.", flush=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details>
<summary>finetune_benchmark_with_save.py — 원래 보관 코드 전문</summary>

원문SHA256: `923c4098bf3c9f39ea3a7644a301cd88463e3cd5295cf4678217d2d328308f2b`

<!-- RESTORE_FILE {"path": "dnabert2/legacy_scripts/finetune_benchmark_with_save.py", "sha256": "923c4098bf3c9f39ea3a7644a301cd88463e3cd5295cf4678217d2d328308f2b", "encoding": "utf-8", "added_final_newline": false} -->
```python
import argparse
import csv
import json
import random
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import psutil
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer, BertConfig

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1000)
args = parser.parse_args()
assert args.n > 0 and args.n % 2 == 0, "--n은 양의 짝수여야 합니다."

root = Path("/workspace/dnabert2_eval")
out = root / "finetune_results" / (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + f"_n{args.n}_b8_bp300_fp32_{time.time_ns()}"
)
out.mkdir(parents=True)

random.seed(42)
torch.manual_seed(42)
assert torch.cuda.is_available(), "CUDA를 사용할 수 없습니다."
torch.cuda.manual_seed_all(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 먼저 실제 데이터 형식과 길이·라벨을 확인합니다.
ds = load_from_disk(str(root / "raw_dataset"))["train"]
assert {"sequence", "label", "task"}.issubset(ds.column_names)
ds = ds.filter(lambda row: row["task"] == "promoter_all")

lengths = Counter(map(len, ds["sequence"]))
counts = Counter(ds["label"])
print("컬럼:", ds.column_names, flush=True)
print("서열 길이:", dict(lengths), flush=True)
print("라벨별 개수:", dict(counts), flush=True)
assert set(lengths) == {300}
assert set(counts) == {0, 1}

groups = {
    label: [i for i, value in enumerate(ds["label"]) if value == label]
    for label in (0, 1)
}
indices = []
for label in (0, 1):
    random.shuffle(groups[label])
    assert len(groups[label]) >= args.n // 2, "보유 데이터가 부족합니다."
    indices.extend(groups[label][:args.n // 2])
random.shuffle(indices)

selected = ds.select(indices)
sequences = selected["sequence"]
labels = selected["label"]
print("이번 학습:", len(selected), "개 /", dict(Counter(labels)), flush=True)

model_path = str(root / "model")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, local_files_only=True
)
config = BertConfig.from_pretrained(model_path, local_files_only=True)
encoder = AutoModel.from_pretrained(
    model_path, config=config,
    trust_remote_code=True, local_files_only=True
)

class Classifier(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled)

model = Classifier(encoder).float().cuda().train()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-5, weight_decay=0.01
)
criterion = torch.nn.CrossEntropyLoss()

summary = {
    "settings": {
        "n": args.n, "batch": 8, "bp": 300,
        "precision": "fp32", "epochs": 1,
        "learning_rate": 2e-5, "seed": 42,
        "mode": "full_finetuning",
        "pooling": "masked_mean_including_special_tokens",
        "tf32": False,
    },
    "gpu": torch.cuda.get_device_name(0),
    "parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ),
    "timing_scope": "tokenization + transfer + forward + backward + optimizer",
    "ram_sampling_interval_seconds": 0.2,
}

process = psutil.Process()
stop = threading.Event()
samples = []

def monitor():
    while not stop.is_set():
        samples.append({
            "seconds": time.perf_counter() - started,
            "rss_gib": process.memory_info().rss / 2**30,
        })
        stop.wait(0.2)

thread = None
loss_sum = 0.0
done = 0
token_lengths = []
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()

try:
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    for start in range(0, args.n, 8):
        texts = sequences[start:start + 8]
        batch = tokenizer(
            texts, padding=True, truncation=False, return_tensors="pt"
        )
        token_lengths.extend(batch["attention_mask"].sum(1).tolist())
        inputs = {
            key: batch[key].cuda()
            for key in ("input_ids", "attention_mask")
        }
        targets = torch.tensor(
            labels[start:start + 8], dtype=torch.long, device="cuda"
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach().item() * len(texts)
        done += len(texts)
        if done % 200 == 0 or done == args.n:
            print(
                f"학습 {done}/{args.n} | loss={loss.item():.4f}",
                flush=True
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary.update({
        "status": "success",
        "training_seconds": elapsed,
        "sequences_per_second": done / elapsed,
        "mean_training_loss": loss_sum / done,
        "tokens_min": min(token_lengths),
        "tokens_max": max(token_lengths),
    })

except Exception as error:
    summary.update({"status": "failed", "error": repr(error)})
    raise

finally:
    stop.set()
    if thread is not None:
        thread.join()
    samples.append({
        "seconds": time.perf_counter() - started,
        "rss_gib": process.memory_info().rss / 2**30,
    })
    summary.update({
        "processed_sequences": done,
        "training_gpu_peak_allocated_gib":
            torch.cuda.max_memory_allocated() / 2**30,
        "training_gpu_peak_reserved_gib":
            torch.cuda.max_memory_reserved() / 2**30,
        "training_ram_peak_sampled_gib":
            max(row["rss_gib"] for row in samples),
    })
    with (out / "resources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seconds", "rss_gib"])
        writer.writeheader()
        writer.writerows(samples)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("결과 폴더:", out, flush=True)


# --- 학습과 기존 자원 측정이 끝난 뒤 저장한다 ---
if summary.get("status") == "success" and hasattr(model, "head"):
    import hashlib
    import shutil
    save_started = time.perf_counter()
    saved = out / "saved_model"
    saved.mkdir(exist_ok=False)
    model.eval()
    model.encoder.save_pretrained(str(saved / "encoder"), safe_serialization=False)
    # DNABERT-2의 실제 실행 모듈과 상대 import 파일도 함께 보존한다.
    import inspect
    implementation_dir = Path(inspect.getfile(type(model.encoder))).parent
    assert (implementation_dir / "bert_layers.py").is_file()
    for implementation in implementation_dir.rglob("*.py"):
        destination = saved / "encoder" / implementation.relative_to(implementation_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(implementation, destination)
    tokenizer.save_pretrained(str(saved / "tokenizer"))
    torch.save(model.head.state_dict(), saved / "classifier.pt")
    metadata = {
        "num_labels": model.head.out_features,
        "problem_type": "single_label_classification" if model.head.out_features == 2
                        else "multi_label_classification",
        "pooling": "masked_mean_including_special_tokens",
        "tokenization": {"padding": True, "truncation": False, "max_length": None},
        "settings": summary["settings"],
        "optimizer_saved": False,
        "purpose": "inference; not a training-resume checkpoint",
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (saved / "task_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (saved / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy2(__file__, saved / "training_script.py")
    save_seconds = time.perf_counter() - save_started

    # 저장 전후 동일 입력의 logits가 일치하는지 확인한다.
    check = {"status": "pending", "save_seconds": save_seconds}
    try:
        check_tokenizer = AutoTokenizer.from_pretrained(
            str(saved / "tokenizer"), local_files_only=True, trust_remote_code=True)
        check_config = BertConfig.from_pretrained(str(saved / "encoder"), local_files_only=True)
        check_encoder = AutoModel.from_pretrained(
            str(saved / "encoder"), config=check_config,
            local_files_only=True, trust_remote_code=True).float().cuda().eval()
        check_head = torch.nn.Linear(check_config.hidden_size, metadata["num_labels"])
        check_head.load_state_dict(torch.load(
            saved / "classifier.pt", map_location="cpu", weights_only=True), strict=True)
        check_head = check_head.float().cuda().eval()
        before = tokenizer(sequences[:2], padding=True, truncation=False, return_tensors="pt")
        after = check_tokenizer(sequences[:2], padding=True, truncation=False, return_tensors="pt")
        for key in ("input_ids", "attention_mask"):
            assert torch.equal(before[key], after[key]), f"Tokenizer mismatch: {key}"
        before = {k: before[k].cuda() for k in ("input_ids", "attention_mask")}
        after = {k: after[k].cuda() for k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            expected = model(**before)
            hidden = check_encoder(**after)[0]
            mask = after["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            actual = check_head(pooled)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        check.update(status="passed", checked_sequences=2,
                     output_shape=list(actual.shape),
                     max_abs_logit_difference=(actual - expected).abs().max().item())
    except Exception as error:
        check.update(status="failed", error=repr(error))
        raise
    finally:
        (saved / "reload_check.json").write_text(json.dumps(check, indent=2) + "\n")
    print("Saved model:", saved, flush=True)
    print("Reload check:", json.dumps(check), flush=True)
    for item in sorted(saved.rglob("*")):
        if item.is_file():
            print(item.relative_to(saved), item.stat().st_size, "bytes", flush=True)
```
<!-- END_RESTORE_FILE -->

</details>

<details>
<summary>bend_length_benchmark_with_save.py — 원래 보관 코드 전문</summary>

원문SHA256: `e85381dbc12089ffaf05ec1f3c0f88de1cf501ae9e8285b9b8ce0f81fa114be1`

<!-- RESTORE_FILE {"path": "dnabert2/legacy_scripts/bend_length_benchmark_with_save.py", "sha256": "e85381dbc12089ffaf05ec1f3c0f88de1cf501ae9e8285b9b8ce0f81fa114be1", "encoding": "utf-8", "added_final_newline": false} -->
```python
import argparse
import csv
import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
from transformers import AutoModel, AutoTokenizer, BertConfig

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--bp", type=int, required=True)
parser.add_argument("--mode", choices=["inference", "training"], required=True)
args = parser.parse_args()

root = Path("/workspace/dnabert2_eval")
data_dir = Path(args.data_dir)
path = data_dir / f"train_n1000_bp{args.bp}.csv"

# 학습 전에 실제 입력 모양을 확인합니다.
with path.open() as f:
    reader = csv.DictReader(f)
    assert {"id", "sequence", "labels"}.issubset(reader.fieldnames)
    rows = list(reader)

assert len(rows) == 1000
assert len({r["id"] for r in rows}) == 1000
sequences = [r["sequence"] for r in rows]
label_lists = [json.loads(r["labels"]) for r in rows]
assert all(len(s) == args.bp for s in sequences)
assert all(set(s) <= set("ACGT") for s in sequences)
assert all(
    labels and all(isinstance(x, int) and 0 <= x < 18 for x in labels)
    for labels in label_lists
)

manifest = json.loads((data_dir / "manifest.json").read_text())
assert [r["id"] for r in rows] == [
    r["id"] for r in manifest["positions"]
]
assert label_lists == [r["labels"] for r in manifest["positions"]]

targets_cpu = torch.zeros(1000, 18, dtype=torch.float32)
for i, labels in enumerate(label_lists):
    targets_cpu[i, labels] = 1

print(
    f"입력 확인: {len(rows)}개 / {args.bp}bp / "
    f"라벨 행렬 {tuple(targets_cpu.shape)} / {args.mode}",
    flush=True
)

assert torch.cuda.is_available()
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model_path = str(root / "model")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, local_files_only=True
)
config = BertConfig.from_pretrained(model_path, local_files_only=True)
encoder = AutoModel.from_pretrained(
    model_path, config=config,
    trust_remote_code=True, local_files_only=True
)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        if args.mode == "training":
            self.head = torch.nn.Linear(config.hidden_size, 18)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled) if args.mode == "training" else pooled

model = Model().float().cuda()
training = args.mode == "training"
model.train(training)
optimizer = (
    torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    if training else None
)
criterion = torch.nn.BCEWithLogitsLoss()

def encode(texts):
    batch = tokenizer(
        texts, padding=True, truncation=False, return_tensors="pt"
    )
    lengths = batch["attention_mask"].sum(1).tolist()
    inputs = {
        k: batch[k].cuda() for k in ("input_ids", "attention_mask")
    }
    return inputs, lengths

# 추론만 준비 실행합니다. 학습은 가중치 변경 없이 바로 측정합니다.
if not training:
    def warmup():
        with torch.inference_mode():
            for _ in range(3):
                inputs, _ = encode(sequences[:8])
                model(**inputs)
        torch.cuda.synchronize()
    warmup()

out = root / "bend_length_results" / (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + f"_{args.mode}_n1000_b8_bp{args.bp}_fp32_{time.time_ns()}"
)
out.mkdir(parents=True)

summary = {
    "settings": {
        "dataset": "BEND histone_modification",
        "mode": args.mode,
        "n": 1000,
        "batch": 8,
        "bp": args.bp,
        "precision": "fp32",
        "tf32": False,
        "epochs": 1 if training else None,
        "learning_rate": 2e-5 if training else None,
        "seed": 42,
        "num_labels": 18,
        "pooling": "masked_mean_including_special_tokens",
    },
    "gpu": torch.cuda.get_device_name(0),
    "parameters": sum(p.numel() for p in model.parameters()),
    "data_file": str(path),
    "sample_ids_sha256": hashlib.sha256(
        "\n".join(r["id"] for r in rows).encode()
    ).hexdigest(),
    "timing_scope": (
        "tokenization + transfer + forward + backward + optimizer"
        if training else "tokenization + transfer + forward + pooling"
    ),
    "inference_warmup_batches": 0 if training else 3,
    "ram_sampling_interval_seconds": 0.2,
    "shared_memory_denominator_gib": 121,
}

process = psutil.Process()
stop = threading.Event()
samples, losses, token_lengths = [], [], []

def monitor():
    while not stop.is_set():
        samples.append({
            "seconds": time.perf_counter() - started,
            "rss_gib": process.memory_info().rss / 2**30,
        })
        stop.wait(0.2)

done = 0
loss_sum = 0.0
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
thread = threading.Thread(target=monitor, daemon=True)
thread.start()

try:
    for start in range(0, 1000, 8):
        inputs, lengths = encode(sequences[start:start + 8])
        token_lengths.extend(lengths)
        count = len(lengths)

        if training:
            targets = targets_cpu[start:start + count].cuda()
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            value = loss.detach().item()
            loss_sum += value * count
            losses.append({
                "step": len(losses) + 1,
                "processed_sequences": done + count,
                "loss": value,
            })
            del logits, loss, targets
        else:
            with torch.inference_mode():
                embeddings = model(**inputs)
            del embeddings

        del inputs
        done += count
        if done % 200 == 0:
            print(f"{args.mode} {args.bp}bp: {done}/1000", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary.update({
        "status": "success",
        "seconds": elapsed,
        "sequences_per_second": done / elapsed,
        "tokens_min": min(token_lengths),
        "tokens_mean": sum(token_lengths) / len(token_lengths),
        "tokens_max": max(token_lengths),
    })
    if training:
        summary["mean_training_loss"] = loss_sum / done

except Exception as error:
    summary.update({"status": "failed", "error": repr(error)})
    raise

finally:
    stop.set()
    thread.join()
    samples.append({
        "seconds": time.perf_counter() - started,
        "rss_gib": process.memory_info().rss / 2**30,
    })
    peak = torch.cuda.max_memory_allocated() / 2**30
    summary.update({
        "processed_sequences": done,
        "gpu_peak_allocated_gib": peak,
        "gpu_allocated_percent_of_shared_memory": peak / 121 * 100,
        "ram_peak_sampled_gib": max(r["rss_gib"] for r in samples),
    })
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    with (out / "resources.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seconds", "rss_gib"])
        writer.writeheader()
        writer.writerows(samples)
    if training:
        with (out / "loss.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "processed_sequences", "loss"]
            )
            writer.writeheader()
            writer.writerows(losses)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("결과 폴더:", out, flush=True)

# 그래프 생성 시간은 자원 측정 구간에서 제외합니다.
if training and losses:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [r["step"] for r in losses]
        y = [r["loss"] for r in losses]
        smooth = [
            sum(y[max(0, i - 9):i + 1]) / len(y[max(0, i - 9):i + 1])
            for i in range(len(y))
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, y, alpha=0.4, linewidth=0.8, label="Batch loss")
        ax.plot(x, smooth, linewidth=2, label="Moving average (10 batches)")
        ax.set(
            xlabel="Optimizer step",
            ylabel="Training loss (BCE)",
            title=f"DNABERT-2 | BEND | {args.bp} bp | n=1,000 | 1 epoch",
        )
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "training_loss.png", dpi=200)
        fig.savefig(out / "training_loss.svg")
        plt.close(fig)
        print("Loss 그래프 저장 완료", flush=True)
    except ImportError:
        print("matplotlib 없음: loss.csv는 저장됐습니다.", flush=True)


# --- 학습과 기존 자원 측정이 끝난 뒤 저장한다 ---
if summary.get("status") == "success" and hasattr(model, "head"):
    import hashlib
    import shutil
    save_started = time.perf_counter()
    saved = out / "saved_model"
    saved.mkdir(exist_ok=False)
    model.eval()
    model.encoder.save_pretrained(str(saved / "encoder"), safe_serialization=False)
    # DNABERT-2의 실제 실행 모듈과 상대 import 파일도 함께 보존한다.
    import inspect
    implementation_dir = Path(inspect.getfile(type(model.encoder))).parent
    assert (implementation_dir / "bert_layers.py").is_file()
    for implementation in implementation_dir.rglob("*.py"):
        destination = saved / "encoder" / implementation.relative_to(implementation_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(implementation, destination)
    tokenizer.save_pretrained(str(saved / "tokenizer"))
    torch.save(model.head.state_dict(), saved / "classifier.pt")
    metadata = {
        "num_labels": model.head.out_features,
        "problem_type": "single_label_classification" if model.head.out_features == 2
                        else "multi_label_classification",
        "pooling": "masked_mean_including_special_tokens",
        "tokenization": {"padding": True, "truncation": False, "max_length": None},
        "settings": summary["settings"],
        "optimizer_saved": False,
        "purpose": "inference; not a training-resume checkpoint",
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (saved / "task_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (saved / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy2(__file__, saved / "training_script.py")
    save_seconds = time.perf_counter() - save_started

    # 저장 전후 동일 입력의 logits가 일치하는지 확인한다.
    check = {"status": "pending", "save_seconds": save_seconds}
    try:
        check_tokenizer = AutoTokenizer.from_pretrained(
            str(saved / "tokenizer"), local_files_only=True, trust_remote_code=True)
        check_config = BertConfig.from_pretrained(str(saved / "encoder"), local_files_only=True)
        check_encoder = AutoModel.from_pretrained(
            str(saved / "encoder"), config=check_config,
            local_files_only=True, trust_remote_code=True).float().cuda().eval()
        check_head = torch.nn.Linear(check_config.hidden_size, metadata["num_labels"])
        check_head.load_state_dict(torch.load(
            saved / "classifier.pt", map_location="cpu", weights_only=True), strict=True)
        check_head = check_head.float().cuda().eval()
        before = tokenizer(sequences[:2], padding=True, truncation=False, return_tensors="pt")
        after = check_tokenizer(sequences[:2], padding=True, truncation=False, return_tensors="pt")
        for key in ("input_ids", "attention_mask"):
            assert torch.equal(before[key], after[key]), f"Tokenizer mismatch: {key}"
        before = {k: before[k].cuda() for k in ("input_ids", "attention_mask")}
        after = {k: after[k].cuda() for k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            expected = model(**before)
            hidden = check_encoder(**after)[0]
            mask = after["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            actual = check_head(pooled)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        check.update(status="passed", checked_sequences=2,
                     output_shape=list(actual.shape),
                     max_abs_logit_difference=(actual - expected).abs().max().item())
    except Exception as error:
        check.update(status="failed", error=repr(error))
        raise
    finally:
        (saved / "reload_check.json").write_text(json.dumps(check, indent=2) + "\n")
    print("Saved model:", saved, flush=True)
    print("Reload check:", json.dumps(check), flush=True)
    for item in sorted(saved.rglob("*")):
        if item.is_file():
            print(item.relative_to(saved), item.stat().st_size, "bytes", flush=True)
```
<!-- END_RESTORE_FILE -->

</details>
