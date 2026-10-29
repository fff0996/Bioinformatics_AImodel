# 10종 후속 실험 최종 결과

10종 모두 후속 실행과 각 범위의 출력 검증·저장·보고서를 완료했다. 모델별 후속 범위와 정밀도·길이·메모리 정의가 다르며 같은 과제/조건의 모델 간 성능 순위로 해석하지 않는다. 아래 배치는 **시험한 최대 성공 조건**이며 이론적 최대 지원 배치가 아니다.

|모델|후속 작업|최대 시험 성공batch|메모리 peak GiB|공유RAM대비%|GPU평균%|종료/해석 제한|상세|
|---|---|---|---|---|---|---|---|
|ESM-2|추론|384|64.416|52.94|94.79|다음 예약량 예측 guard; 실제 OOM 아님|[보고서](esm2/reports/ESM2_batch_memory.md)|
|AlphaFold2 — ColabFold|실험적 FP32 JAX vmap|4|2.308|1.90|94.48|실제4query 끝; JAX peak는컴파일/warmup포함 누적값|[보고서](colabfold/reports/Batch_memory_followup.md)|
|scvi|추론|10176|0.270|0.22|4.72|available_unique_input_limit|[보고서](scvi/reports/Batch_memory_followup.md)|
|Cellpose|추론|384|43.397|35.66|89.84|다음 예약량 예측 guard; 실제 OOM 아님|[보고서](cellpose/reports/Cellpose_batch_memory.md)|
|dnabert2|추론|512|60.007|49.31|92.16|predicted_reservation_exceeds_available_budget|[보고서](dnabert2/reports/Batch_memory_followup.md)|
|ntv2|추론|128|21.592|17.74|94.83|available_unique_input_limit|[보고서](ntv2/reports/Batch_memory_followup.md)|
|Clair3 pileup|실제 후보 network 추론|5708|4.111|3.38|88.48|확보한 고유feature pool 끝; 최대한계 아님|[보고서](clair3/reports/Batch_memory_followup.md)|
|Clair3 full-alignment|실제 후보 network 추론|894|1.499|1.23|91.68|확보한 고유feature pool 끝; 최대한계 아님|[보고서](clair3/reports/Batch_memory_followup.md)|
|rnafm|추론|76|3.844|3.16|95.42|available_unique_input_limit|[보고서](rnafm/reports/Batch_memory_followup.md)|
|openspliceai|추론|156|1.463|1.20|95.65|available_unique_input_limit|[보고서](openspliceai/reports/Batch_memory_followup.md)|
|scGPT|실제30만세포 임베딩 추론|256|5.477|4.50|95.10|같은30만세포 batch8/256 비교|[보고서](scgpt/reports/scGPT_actual_300k.md)|

메모리 비율의 분모는CPU/GPU공유RAM121.688GiB다. GPU 연산사용률과 메모리 비율은 다른 지표이고, 전용VRAM점유율이 아니며RSS와CUDA를더하지않는다. PyTorch행은CUDAallocated peak다. AF2는JAX프로세스누적allocator peak로컴파일/warmup을포함하므로실행구간peak와직접비교하지않는다. 예약reserved와sampled RSS/가용량도각보고서에별도로있다.

## scGPT 별도 학습 용량 결과

FP32/1200열고정padding 학습probe의최대성공batch292: allocated100.750GiB,전체공유RAM대비82.79%. 이는위표의추론batch256·5.477GiB와다른실험이다. 90%는관측하지못했다. [학습용량보고서](scgpt/reports/scGPT_memory_batch.md)

## 마지막 두 모델의 구현과 검증

Clair3는실제HG003 BAM에서전체candidate tensor를추출해pileup/full-alignment를따로배치확대했다. 원래native pipeline의동일입력/확률대조와저장·재로드를검증했다. 후보좌표를내보내지못해생물학적고유위치수가아닌중복제거feature수로보고한다. VCF호출전체시간과candidate network시간은구분한다.

AlphaFold2는CLI기본배치가아닌JAX vmap 별도실험이다. 단량체model_1/115AA실제4query/기존MSA/3recycle을사용했다. BF16벡터화참조검증은구조차이로실패해보존했고,FP32고정별도조건의batch1/2/4로실행했다. batch1과native참조CA정렬RMSD0.1795Å/pLDDT최대차0.1875로사전0.5Å기준통과. 출력PDB·좌표·pLDDT·query수검증을수행했고batch별수치차이도보존했다. 기존기본BF16·5모델순차예측과동일조건으로표현하지않는다.

기존미실시/중단/실패기록은인수인계이력으로남기며현재최신상태는이최종표를우선한다. 원본run/명령/로그/출력/환경/입력선택/자원CSV는각모델results와보고서에있다. 추가GPU모델작업은자동시작하지않는다.

뇌이미지 추가 Cellpose NucMM-M brain_nucmm_20261029_110603_965442 상태 complete. [보고서](cellpose/reports/Cellpose_brain_NucMM.md). 일반뇌MRI가아닌세포핵micro-CT·정답instance검증,기본/배치실험과별도run.
