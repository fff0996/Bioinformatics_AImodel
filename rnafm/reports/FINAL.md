# RNA-FM 최종 통합 보고서

기본 실험과 모든 후속 실험을 이 문서에 함께 수록했다. 다른 기본/후속 문서를 따로 읽을 필요 없이 아래 요약과 본문을 보면 된다.

## 1. 입력과 메모리 결론

RNA 염기서열 → 염기토큰 → 640차원 임베딩. 분류실험은본체를동결하고선형head만학습.

**목표는 메모리 할당80% 조건 확인**이다. 전체공유RAM121.688GiB의80%는약97.35GiB다. GPU-Util은별도연산활동지표이며이목표의성공판정에쓰지않는다.

|작업|관측메모리 peak GiB|공유RAM대비|해당 입력·배치·정밀도|80% 확인/제약|
|---|---|---|---|---|---|
|기존 실제길이 혼합 추론|**8.870**|7.29%|train896개 RNA를한번에최대50개씩임베딩추출. 입력은배치별길이로padding,관측shape중최대50×1022토큰(최대1020nt). FP32.|이전체추출구간의peak; 어느개별batch에서peak인지별도분리하지않음. 80% 미확인|
|고정512nt 배치 후속 추론|**3.844**|3.16%|512nt 실제RNA76개,입력76×514토큰(BOS/EOS포함),FP32.|고유76서열풀끝. 이값이전체실험최대라는뜻아님|
|동결 임베딩의 분류head 학습|**0.434**|0.36%|추출된640차원벡터를batch8×640으로입력,28종RNA label,100epoch. 본체는학습하지않고GPU에남아있었음.|본체파인튜닝 미실시. 메모리에상주한동결본체도포함|

메모리는PyTorch의경우CUDA allocated peak다. reserved는예약량으로별도다. **AlphaFold2는컴파일·warmup등을포함한JAX프로세스누적peak**이므로CUDA구간peak와같은정의로비교하지않는다. RSS와CUDA를더하지않는다. 표는시험한조건의관측값이며모델의절대최대나모든모델의80%달성을의미하지않는다.

## 2. 기본 정보·입력 검사·추론·학습·평가

## RNA-FM — DGX Spark


run `rnafm_20261027_204440_142840` · 상태 **complete** · [summary](../results/rnafm_20261027_204440_142840/summary.json) · 로그 (로컬 보관: `../results/rnafm_20261027_204440_142840/benchmark.log`) · [자원 시계열](../results/rnafm_20261027_204440_142840/resources.csv)

### 체크포인트와 실행 구성

공식RNA-FM ncRNA 12-layer checkpoint `RNA-FM_pretrained.pth`, mRNA-FM과구분한다. 소스 commit `348951516e0963d22bbb33b3c9fc18c89081d38e`, RNA-FM0.2.2. 공식 Hugging Face cuhkaih/rnafm의 고정revision/SHA-256은 models/repository.json과 models/sources.json에보존했고LFS해시를검증했다.

실제 파라미터: {"backbone_total": 99521546, "backbone_trainable": 0, "head": 17948}

전체RNA-FM99,521,546개파라미터의FP32이론용량은약0.371GiB다. 배포checkpoint1.19GB와실제GPU모델할당량은같지않다. forward는기본LM logits와12층representation을계산하지만contact attention/structure predictor는계산하지않는다. 공식RNAtype튜토리얼처럼backbone은동결하고640→28선형classifier만학습했다. RNA-FM backbone파인튜닝완료라고표현하지않는다.

환경 {"torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "rna_fm": "0.2.2", "gpu": "NVIDIA GB10"}

FP32,TF32=False,seed42,8CPUthreads. 공식RNAtype추론chunk50을배치상한으로,분류기학습batch8/lr1e-3/Adam/100epochs를고정했다. 튜토리얼의1024zero-pad평균은사용하지않고실제염기token만mean pooling했다(BOS/EOS/PAD제외).

| 입력forward전 | allocatedGiB | reservedGiB | RSSGiB | system availableGiB |
| --- | --- | --- | --- | --- |
| RNA-FM 적재 직후 | 0.371 | 0.420 | 1.221 | 112.919 |

적재 0.526초,3배치warmup 0.847초. CPU입력JSON은적재전에준비했다.

### 실제 RNA와 유형 라벨

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

RNA-FM은염기당1token+BOS/EOS로변환한다.원본최대1020nt→1022tokens로max_positions1024범위내다. 분류기학습입력은동결RNA-FM에서미리추출한B×640 FP32 embedding과B정수label이다. 원본서열·라벨과split은 inputs.json (로컬 보관: `../data/inputs.json`),검사는 [input_audit](../data/input_audit.json)에있다.

개수비교는512nt이상인실제서열64개를동일512ntcrop으로만들어중첩8/32/64개,길이비교는960nt이상인동일실제7개에서128/256/512/960nt앞부분crop이다. 반복은계측반복이며고유입력으로세지않는다. 최대길이비교에일곱개만있다는데이터한계를명시한다.

### 입력별 GPU 사용률·메모리

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

### 분류기 학습·저장·test

선형head만100epoch학습하고validation Accuracy가가장높은epoch를선택했다. test로epoch를선택하지않았다. 선택epoch 100; test {"accuracy": 0.7678571428571429, "macro_f1": 0.7662770895636066}

headweight최대변화 2.795; frozenbackbone확인tensor변화 0.0. 전체backbone은requires_grad=False라optimizer대상에없다.

저장head `models/rnafm_20261027_204440_142840/best_head.pt`,frozenbackbone은`models/RNA-FM_pretrained.pth`와함께쓴다. task.json에라벨/전처리/원본모델을저장했고최종epoch100optimizer/RNG상태와선택besthead를구분했다. 학습재개는검증하지않았다. 저장시간 0.001초. 독립새프로세스RNA-FM+head재로드: {"independent_process": true, "inputs": 280, "embedding_max_abs_difference": 0.0, "logits_max_abs_difference": 0.0, "predictions_match": true, "passed": true}

head학습GPU구간에는epoch별validation이포함돼있으며동결backbone은GPU에유지했지만재계산하지않았다. 작은선형head의GPU-Util이낮아도실패가아니다. 이결과는RNA구조예측정확도나backbone파인튜닝성능과다르다.

### 한계·공식 출처

RNA유형간길이분포차이가있고동족서열·사전학습중복가능성이있어소규모튜토리얼재현수준의분류결과다. source setup.py의누락README_backup.md참조만README.md로고쳐설치했다. PyTorchcheckpoint의argparse.Namespace는명시safe_globals로허용하고공식해시검증파일만불러왔다.

- [RNA-FM공식소스](https://github.com/ml4bio/RNA-FM)
- [공식RNAtype튜토리얼·데이터](https://github.com/ml4bio/RNA-FM/tree/main/tutorials/rna_family-clustering_type-classification)
- [공식checkpoint](https://huggingface.co/cuhkaih/rnafm)

## 3. 후속 실험 통합 — 배치 확대 및 자원 검증

## rnafm 실제 입력의 배치별 추론 자원


상태 **complete** · run `batch_followup_20261028_144452_703996` · 종료사유 `available_unique_input_limit`

### 입력과 고정 조건

{
  "source": "official tutorial RNAcentral actual unique sequences >=512nt, first512nt explicit crop; pool limit retained",
  "unique_available_inputs": 76,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

### 배치별 측정

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

### 결과 해석과 검증

최대시험성공batch 76. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://github.com/ml4bio/RNA-FM) · [기존 모델 보고서 디렉터리](.)

## 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.
