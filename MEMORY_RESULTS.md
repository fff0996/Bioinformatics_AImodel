# scGPT 후속 실험 결과

> 최신 읽기 경로: [기본·후속을 합친 최종 보고](FINAL_REPORT.md). 이 문서는 이전 실행/계측 기록이며, 80% 메모리 조건 확인은 scGPT 학습용량probe에 한정된다. 다른 모델은 미확인이다.

최신 사용자 범위: 후속 실험은 **scGPT만**, 나머지9종의 추가 sweep은 진행하지 않는다.

- [배치별 학습 용량 실험](scgpt/reports/scGPT_memory_batch.md): 완료. FP32/고정1200열 padding에서 시험한 최대 성공batch292, CUDA allocated peak100.750GiB / 전체공유RAM121.688GiB =82.79%. 90% 미도달. 정상 dynamic padding과 다른 capacity probe다.
- [pancreas 실제14,818세포 전체 처리](scgpt/reports/scGPT_realdata_full.md): batch8 학습8480/validation2120/test4218 완료·저장·재로드 검증. 큰 학습batch284/256/224는 메모리guard 중단, 성공 비교로 해석하지 않는다.
- [Tabula Sapiens 실제30만세포 최종 비교](scgpt/reports/scGPT_actual_300k.md): **2026-10-28 완료**. 같은300000개 고유세포·FP32·max1200 입력으로 batch8/256 비교 및 전체출력검증 통과. batch8 allocated peak0.389GiB·GPU평균94.89%·유효2663.19초, batch256 allocated peak5.477GiB·reserved peak21.475GiB·GPU평균95.10%·2628.60초. batch8은 전날10만+재개20만의 시간 합으로 야간중단 제외. 배치32배 확대에도 처리량은112.65→114.13세포/초로 유사했다(단일 실행·세션 차이 포함).

최종 run: `scgpt/results/actual_300k_resume_20261028_093701_696230`. inference exit0, 전체300000×512 출력 두 조건 모두finite·단위norm 검증, 배치간cosine 최소0.99999964, 전날10만출력 정확히 보존. 중단·재개 실제 실행도 검증됐다.

**후속 실험과 보고서 반영을 완료했고 GPU 계산 프로세스가 없음을 확인했다. 추가 모델 실험은 시작하지 않는다.**

Spark 메모리는CPU/GPU공유다. allocated/전체공유RAM 비율은전용VRAM점유율이나시스템전체사용률이아니다. GPU-Util과메모리점유를구분하고RSS와CUDA를더하지않는다.

2026-10-28 최종 완료: scGPT 실제30만 고유세포 batch8/256 추론 및 전체출력검증 완료. 실행 `actual_300k_resume_20261028_093701_696230`, [최종 보고서](scgpt/reports/scGPT_actual_300k.md). 후속 실험 범위는 scGPT만이며 다른 모델은 진행하지 않는다. 최신 상태는 이 항목을 우선한다.

2026-10-28 사용자 추가승인: ESM-2 긴 실제서열900~950AA 배치확대 **추론** 실험 종료. run esm2/results/batch_inference_20261028_112900_601721. 결과는 [배치 메모리 보고서](esm2/reports/ESM2_batch_memory.md)에정리한다. scGPT결과보존.

ESM-2 후속 최종 상태 complete, 최대성공batch 384,80%도달 False,90%도달 False. [상세](esm2/reports/ESM2_batch_memory.md). 새로운 다른 모델 실험은 자동 시작하지 않는다.

Cellpose 타일배치후속 tile_batch_20261028_134130_265230 상태 running, 최대성공타일batch 8,80%도달 None,90%도달 None. [보고서](cellpose/reports/Cellpose_batch_memory.md). 추가모델실험자동시작없음.

Cellpose 타일배치후속 tile_batch_20261028_134130_265230 상태 complete, 최대성공타일batch 384,80%도달 False,90%도달 False. [보고서](cellpose/reports/Cellpose_batch_memory.md). 추가모델실험자동시작없음.

남은 후속 모델은사용자승인으로자동순차진행중이다. [통합후속진행](FOLLOWUP_RESULTS.md)에서모델별실제상태·결과와배치확대미실시항목을확인한다.

scvi 배치후속 complete, 최대성공batch 10176. [보고서](scvi/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

dnabert2 배치후속 complete, 최대성공batch 512. [보고서](dnabert2/reports/Batch_memory_followup.md). 종료사유 predicted_reservation_exceeds_available_budget.

ntv2 배치후속 complete, 최대성공batch 128. [보고서](ntv2/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

rnafm 배치후속 complete, 최대성공batch 76. [보고서](rnafm/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

openspliceai 배치후속 complete, 최대성공batch 156. [보고서](openspliceai/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

clair3 전용추가후속 complete; [결과](clair3/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

af2 전용추가후속 stopped; [결과](colabfold/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

clair3 전용추가후속 complete; [결과](clair3/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

af2 전용추가후속 stopped; [결과](colabfold/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

clair3 전용추가후속 complete; [결과](clair3/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

af2 전용추가후속 complete; [결과](colabfold/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

10종 후속최종정리완료. 이전AlphaFold2/Clair3미실시상태는새실행으로갱신됐다. 구현·정밀도·검증범위차이를포함한최신전체표: [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md). 추가GPU작업없음.
