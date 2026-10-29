# scGPT 실제 pancreas 전체 데이터 처리

실행: `realdata_full_20261027_215022_050816`. 총 14,818개 실제 세포: 학습 8,480 / validation 2,120 / 독립 test 4,218. FP32, batch8, seed42, 최대1,200토큰, 원래 dynamic padding, 1epoch. 입력 ID 중복·파일 간 겹침 없음은 source_data_audit.json에서 확인했다. 원시 count provenance는 확인되지 않았다.

|조건|학습시간|allocated peak|GPU 평균/최대|독립 test Accuracy|
|---|---|---|---|---|
|batch8|202.88초|3.374 GiB|93.02/95%|0.92215|

모델 저장 및 validation strict reload 최대 logit 오차0. 별도 프로세스에서 test4,218세포를 예측했고, 학습 라벨에 속한4,213세포로 정확도를 계산했다. test-only5세포는 알려진 클래스 지표에서 제외했다. macroF1(학습13클래스) 0.52963, weightedF1 0.91658. 학습1epoch/1060 optimizer updates이며 donor-independent 검증은 아니다.

batch284·256·224 학습은 시스템 가용 RAM8GiB guard가 발동해 중단됐다. 완료된 학습·배치 비교 결과로 쓰지 않는다. 각각 실패 로그 및 memory_guard.json을 보존했다. 샘플링 감지 지연으로224 조건은 가용4.424GiB에서 중단됐으며, 안전한 최대 배치가 검증됐다는 의미가 아니다. 추가 대형 학습 배치 재시도 대신 사용자 우선 요청인 실제30만세포 임베딩 추론을 진행한다.

[원본 결과](../results/realdata_full_20261027_215022_050816/summary.json) · [실행·입력·저장 모델 코드](../scripts/full_realdata.py)

CUDA allocated와 reserved는 프로세스 allocator 값이다. Spark 공유 RAM에서 RSS와 CUDA를 독립 물리 메모리처럼 더하지 않는다. GPU-Util은 메모리 점유율이 아니다. 학습시간은 전처리/전송/forward/backward/optimizer를 포함하며 validation·모델저장은 제외한다.
