# ntv2 실제 입력의 배치별 추론 자원

> 최신 통합판: [기본·모든 후속 실험을 합친 최종 보고서](FINAL.md). 입력·추론·학습별 메모리와 80% 목표 상태는 최종 보고서 첫 표를 확인한다.

상태 **complete** · run `batch_followup_20261028_143713_869625` · 종료사유 `available_unique_input_limit`

## 입력과 고정 조건

{
  "source": "existing real GRCh38 regions, central6000nt crops -> native ~1001tokens",
  "unique_available_inputs": 128,
  "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
  "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
}

배치는한번에GPU에서계산하는독립입력개수다. scVI는세포수,서열모델은서열수이며원본전체데이터개수와다르다. 실제기존공개데이터의중첩고유부분집합을사용한다. 반복처리는관측시간확보용이며고유데이터를늘렸다는뜻이아니다. 길이crop·tokenization·dtype·체크포인트를고정하고batch만확대한다. 실제input_shapes와ID는각batch/summary에있고원본입력선택은inputs.json또는저장scVIadata에있다.

FP32/TF32=False/seed42,각조건fresh container. 기존정상환경·모델·데이터를보존하고로컬가중치만재사용한다. 전체/실제사용backbone 파라미터수·모델출처·기본학습/정확도평가는기본모델보고서를참조한다. 각새run의전체파라미터수·적재기준값·모델로딩/warmup시간·Torch/CUDA·저장출력은batch별summary에있다. 이번실험은추론만수행하므로학습파라미터0이다.

## 배치별 측정

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

## 결과 해석과 검증

최대시험성공batch 128. allocated80%도달 False,90%도달 False. 실제고유입력pool한계/안전guard로중단할수있으며모델절대한계를뜻하지않는다. 다음예약량예측은보수적안전추정이며실측실패와구분한다.

출력shape·전체finite·저장후정확재로드·동일첫입력의배치별예측일치검사를수행한다. 결과가안맞거나실패하면성공행에넣지않고batch로그/exit/summary에보관한다. 독립생물학적정확도평가/학습을추가수행한결과가아니다. 원래모델의기본학습/재로드/test결과를재사용한다.

GPU-Util은계산활동시간비율이며메모리비율이아니다. CUDA allocated/reserved는프로세스allocator이고공유RAM약121.688GiB대비비율은전용VRAM점유율이아니다. RSS와CUDA를더하지않는다. 모델적재와추가메모리는같은worker에서측정했다. RSS/MemAvailable은0.5초대기+조회간격sampled값이다. 가용16GiB·swap증가2GiB guard와다음batch예약예측으로확대를중단한다.

원본설정·코드·환경·input IDs·cache·출력·GPU시계열·명령·종료코드는실행폴더와scripts/remaining_followups에보관한다. 새학습은수행하지않아분류라벨을자원실험의정답으로사용하지않는다.

[공식 출처](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) · [기존 모델 보고서 디렉터리](.)
