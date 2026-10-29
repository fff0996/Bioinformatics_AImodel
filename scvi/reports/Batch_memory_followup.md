# scvi 실제 입력의 배치별 추론 자원

> 최신 통합판: [기본·모든 후속 실험을 합친 최종 보고서](FINAL.md). 입력·추론·학습별 메모리와 80% 목표 상태는 최종 보고서 첫 표를 확인한다.

상태 **complete** · run `batch_followup_20261028_142752_517769` · 종료사유 `available_unique_input_limit`

## 입력과 고정 조건

{
  "source": "saved full10176cell SCVI model and its raw-count AnnData, fixed2000genes; latent posterior mean",
  "unique_available_inputs": 10176,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

## 배치별 측정

측정은cache 이후배치구성/전송·forward·pooling/latent계산·CPU반환을포함한다. 다운로드·입력선택/crop·tokenization·모델로딩·warmup·파일저장은제외한다. scVI공식latent API는내부데이터로더구성을포함한다. OpenSpliceAI는동일5개모델ensemble을순차평가한다. 최솟값2회·15초이상반복이며실제횟수를기록한다.

|batch/고유입력|반복|총처리|초|입력/초|모델적재GiB|allocated peak GiB|적재대비추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|RSS sampled peak GiB|가용 sampled 최소GiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|128|13607|1741696|15.001|116105.6941|0.0048|0.039|0.0342|0.041|0.032|15.6897|19.0|1.5399|112.4895|
|256|11926|3053056|15.0007|203528.1284|0.0048|0.0419|0.0371|0.0566|0.0345|17.4138|20.0|1.5383|112.371|
|512|10058|5149696|15.0006|343299.507|0.0048|0.0478|0.0429|0.0566|0.0393|20.1724|24.0|1.5434|112.4101|
|1024|7756|7942144|15.0008|529446.5185|0.0048|0.0595|0.0546|0.0762|0.0489|26.1724|29.0|1.5465|112.4435|
|2048|4718|9662464|15.0018|644088.376|0.0048|0.084|0.0791|0.0859|0.069|24.2414|26.0|1.534|112.3655|
|4096|2211|9056256|15.0009|603715.4809|0.0048|0.1318|0.127|0.1504|0.1083|24.8621|28.0|1.5626|112.3741|
|8192|154|1261568|15.0271|83952.6489|0.0048|0.2232|0.2184|0.2461|0.1834|5.7931|7.0|1.6075|112.1286|
|10176|122|1241472|15.0815|82317.5007|0.0048|0.2696|0.2647|0.2871|0.2215|4.7241|8.0|1.6315|112.0164|

## 결과 해석과 검증

최대시험성공batch 10176. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html) · [기존 모델 보고서 디렉터리](.)
