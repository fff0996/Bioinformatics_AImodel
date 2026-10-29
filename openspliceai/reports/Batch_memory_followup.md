# openspliceai 실제 입력의 배치별 추론 자원

상태 **complete** · run `batch_followup_20261028_144737_167765` · 종료사유 `available_unique_input_limit`

## 입력과 고정 조건

{
  "source": "actual GRCh38 curated regions; input15000nt -> output5000nt; original five-model MANE10000 ensemble",
  "unique_available_inputs": 156,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

## 배치별 측정

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

## 결과 해석과 검증

최대시험성공batch 156. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://github.com/Kuanhao-Chao/OpenSpliceAI) · [기존 모델 보고서 디렉터리](.)
