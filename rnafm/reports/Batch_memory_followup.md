# rnafm 실제 입력의 배치별 추론 자원

> 최신 통합판: [기본·모든 후속 실험을 합친 최종 보고서](FINAL.md). 입력·추론·학습별 메모리와 80% 목표 상태는 최종 보고서 첫 표를 확인한다.

상태 **complete** · run `batch_followup_20261028_144452_703996` · 종료사유 `available_unique_input_limit`

## 입력과 고정 조건

{
  "source": "official tutorial RNAcentral actual unique sequences >=512nt, first512nt explicit crop; pool limit retained",
  "unique_available_inputs": 76,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

## 배치별 측정

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

## 결과 해석과 검증

최대시험성공batch 76. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://github.com/ml4bio/RNA-FM) · [기존 모델 보고서 디렉터리](.)
