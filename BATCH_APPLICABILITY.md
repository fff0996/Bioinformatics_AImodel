# 배치 실험 적용 범위 — 최신 상태

2026-10-28 추가 검토 후 AlphaFold2·Clair3 모두 별도 후속 실행을 완료했다. Clair3는 실제candidate tensor cache에서native model batch를조절했고,AF2는FP32 고정JAX vmap별도경로로실제4query를검증했다. 구현·정밀도·검증제약은 [전체최종결과](FOLLOWUP_RESULTS.md), [Clair3](clair3/reports/Batch_memory_followup.md), [AF2](colabfold/reports/Batch_memory_followup.md)를우선한다.

아래는 추가실험 전에 작성한 초기검토이력이다. 현재미실시상태를뜻하지않는다.

# 후속 배치 실험의 적용 범위

ESM-2·Cellpose·scGPT 완료 결과는 보존한다. 남은 scVI→DNABERT-2→NT-v2→RNA-FM→OpenSpliceAI는 기존 가중치와 실제 입력으로 GPU 하나씩 추론 배치 확대를 진행한다. 학습 확대를 수행한 것처럼 표현하지 않는다. 데이터 고유 개수가 최대 시험 배치를 제한할 수 있다.

AlphaFold2 — ColabFold: 현재 경로의 여러 query 목록은 차례로 처리하는 작업 목록이다. 서로 독립적인 query를 한 forward의 batch로 묶는 별도 지원 경로를 확인하지 못했으므로 전용 batch 스윕을 수행하지 않는다. 복합체의 사슬 수 확대는 다른 생물학적 입력이며 batch 확대가 아니다. 기존 구조예측 결과는 보존한다. [공식 ColabFold](https://github.com/sokrypton/ColabFold)

Clair3: 설치 소스 clair3/CallVariantsFromCffi.py의266~269행에서 batch는CPU경로 predictBatchSize×threads, GPU경로 predictBatchSize×5로 결정된다. 기존GPU 측정 기본batch는1000이었다. 공식 현재CLI에서 독립 minibatch 인자를 확인하지 못했다. threads/worker 확대를 batch 확대라고 표현하지 않으며, 내부 source-level 설정 변경과 충분한 실제 후보 tensor cache를 이용한 별도 전용실험은 이번큐에서 수행하지 않았다. 따라서 Clair3의 최대 지원batch나 메모리80/90% 가능성을 검증한 것으로 쓰지 않는다. 기존 variant calling 결과는 보존한다. [공식 Clair3](https://github.com/HKU-BAL/Clair3)

현재 단계의 제한이지 모든 모델 구현에서 batch 자체가 불가능하다는 뜻은 아니다. 최신 실제 실행상태는 FOLLOWUP_RESULTS.md 및 FOLLOWUP_QUEUE.json에서 확인한다.
