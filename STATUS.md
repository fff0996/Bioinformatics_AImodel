# DGX Spark 바이오 AI 진행 상태

갱신: 2026-10-27T21:28:47+09:00 KST. GPU 실험은 한 번에 하나씩 실행한다.

| 순서 | 모델 | 상태 | 최신 경로 / 제약 |
| --- | --- | --- | --- |
| 1 | ESM-2 | 입력·추론·개수/길이 비교·부분 backbone 학습·저장·독립 재로드·test·보고서 완료 | esm2/reports/ESM2_20261027_164300_125163.md |
| 2 | AlphaFold2 (ColabFold) | 환경·입력·단일체 35구조·복합체 5구조·규모 계측·저장 구조 검증·보고서 완료; 학습 N/A | colabfold/reports/AlphaFold2_ColabFold.md |
| 3 | scVI | raw count 입력·3개 규모 400epoch 학습·latent 추론·batch 진단·저장·독립 재로드·보고서 완료 | scvi/reports/scVI.md |
| 4 | Cellpose | Cellpose4.2.1 cpsam_v2 입력·추론6조건·100epoch 파인튜닝·test·저장·독립 재로드·보고서 완료 | cellpose/reports/Cellpose.md |
| 5 | DNABERT-2 | 기존 기본 결과 보존; 추론6조건·학습 자원 probe4조건·사용률/메모리 보완·보고서 완료 | dnabert2/reports/DNABERT2.md; 원본 모델 보존, 짧은 학습 probe 사용률은 표본 한계 명시 |
| 6 | Nucleotide Transformer V2 | 공식500M 입력·추론6조건·전backbone2epoch 학습·test400·저장·독립 재로드·보고서 완료 | ntv2/reports/NTv2.md; 오류 복구 이력 보존 |
| 7 | Clair3 | v2.0.4 공식 ONT 두모델·구간3조건/coverage1조건·GIAB exact allele 비교·VCF저장·독립 재로드·보고서 완료; 학습 미수행 | clair3/reports/Clair3.md; ARM64 및 worker1개 호환 patch 기록 |
| 8 | RNA-FM | 공식99.5M 입력·추론7조건·1400RNA 임베딩·동결backbone+선형head100epoch·test280·저장·독립재로드·보고서 완료 | rnafm/reports/RNAFM.md; backbone 파인튜닝 아님 |
| 9 | OpenSpliceAI | 공식MANE10000nt 5ensemble·추론5조건·주석입력·seed10 전파라미터10epoch·test36·저장·독립재로드·보고서 완료 | openspliceai/reports/OpenSpliceAI.md; 변이delta CLI 미실행 |
| 10 | scGPT | 기존 추론·파인튜닝·test·길이 비교 보존; 새 추론6조건·학습GPU사용률 보완·모델저장·독립재로드·보고서 완료 | scgpt/reports/scGPT.md; 추가 학습run은계측목적, 기존test성능과구분 |

ESM-2 기본 run: esm2/results/20261027_164300_125163 (종료 코드 0).
보완 run: esm2/results/supplement_20261027_165443_595867 (완료 및 exit_code.json 확인).
기존 ESM-2 작업은 세션 중단 후에도 정상 완료됐고 중복 실행하지 않았다. 보완은 기존 작업 종료 확인 후 실시했다.
부분 파인튜닝 정확도: test n=128, Spearman 0.0241; 동작 검증이며 높은 예측 성능 완료로 표현하지 않는다.

Docker 접근 차단은 해소됐다. 사용자가 hyein을 docker 그룹에 등록했고 sg docker로 실제 이미지·GPU 실행을 확인했다.

AlphaFold2 단일체 run: colabfold/results/af2_20261027_172813_247681 (종료 코드 0, 구조 35개 검증).
AlphaFold2 복합체 run: colabfold/results/af2_multimer_20261027_181640_431467 (종료 코드 0, 구조 5개 검증).
AlphaFold2는 추론만 수행했다. ESM-2·AlphaFold2·scVI의 새 검증을 완료했고 Cellpose까지 완료했고 DNABERT-2 보완도 완료했다. 다음 순서는 Nucleotide Transformer V2다.

scVI 완료 run: scvi/results/scvi_20261027_185901_460537 (종료 코드 0, 세 모델 독립 재로드 최대 차이 0).
scVI 학습 source: 1024세포=scvi/results/scvi_20261027_184921_028161, 4096/10176세포=scvi/results/scvi_20261027_185327_767988. 학습을 중복 실행하지 않고 후속 로딩/추론 오류만 복구했다.
추가 보고서 기준: AGENTS.md에 모델 적재 메모리·입력 예시·GPU 사용률과 메모리 동시 표를 명시했다. 결과 안내: RESULTS.md.

Cellpose 완료 run: cellpose/results/cellpose_20261027_193331_947064 (학습·평가 exit0, 독립 재로드 exit0; flow/cellprob 최대 차이0, mask 일치).
Cellpose 학습16/validation8/test16 BBBC039 공식 분할 subset, 100epochs1600steps. AP50 0.9054→0.9138; AP75/90은 소폭 하락. 추론 peak1.491GiB, 학습 peak7.985GiB, GPU 최대95%. 결과: cellpose/reports/Cellpose.md.

DNABERT-2 보완: dnabert2/results/supplement_20261027_195301_706540 (exit0, 10조건). scGPT는 기존 완료 상태를 재확인했고 기본 결과를 재실행하지 않았다. scGPT test·길이 비교 링크는 RESULTS.md에 추가했다.

NT-v2 완료: ntv2/results/ntv2_20261027_201332_251243; finetune/inference_remeasure/reload exit0, logits차이0. test Accuracy0.6975/AUROC0.786175; 학습 peak9.252GiB, 추론12276nt=2047tokens peak6.475GiB. 다음 Clair3.

Clair3 완료: clair3/results/clair3_20261027_203339_768667; benchmark/reload exit0, 두모델 예측차이0. 200kb전체:17.93초/worker peak0.684GiB/GPU평균11.3%. half coverage peak0.763GiB. 다음 RNA-FM.

후속 실험 추가 승인: 2026-10-27 사용자 요청. 기본10종 완료 후 모델별 batch 확대를 통해 메모리90% 도달 가능성과 처리량 증가/포화·안전한 최대 성공 배치를 확인한다. 모델·길이·정밀도는고정하고GPU사용률/allocated/reserved/공유RAM가용량을동시에기록. GPU는하나씩,메모리압박시중단. 아직이후속실험은실행하지않았다.

RNA-FM 완료: rnafm/results/rnafm_20261027_204440_142840; embedding/logits재로드차이0, test Accuracy0.767857/macroF1 0.766277. 다음 OpenSpliceAI,이후scGPT보완. 후속배치확대별도승인유지.

OpenSpliceAI 완료: openspliceai/results/openspliceai_20261027_205554_995618; benchmark/reload exit0, test확률차이0. 다음scGPT사용률/개수보완,그후승인된배치확대실험.

기본10종 검증/범위별보고서 완료. scGPT 보완: scgpt/results/supplement_20261027_211638_936991; 추론/학습/독립재로드 exit0, validation2120 logits차이0. 학습1000세포batch8 FP32 peak3.374GiB/GPU평균88.69%.
현재 사용자가 진행시점을 재확인했다. 후속 메모리80~90% 배치 확대 실험은 **아직 실행하지 않았으며 계획만 추가**했다. 현재까지 run은기존고정batch 기본/보완작업이다. 기본10종 결과정리와 후속실험을구분해보고한다.

후속 실제 시작: 사용자 "이제 ㅇㅇㅇ 후속 ㄱㄱ" 승인에따라 scGPT batch 확대 실행 중. run: scgpt/results/memory_batch_20261027_213310_982032. 기존기본/보완은전부완료상태유지. FP32/1200token/전파라미터중동일학습대상/optimizer고정,실제1000학습세포중첩batch. allocated80/90%와처리량을측정하며가용RAM8GiB·swap+2GiBguard와추정예약량에따라확대를중단한다.

사용자 추가 요청: 실제 만 단위 세포 전체 처리 검증을 진행한다. scGPT demo_train10600 중 기존분할대로train8480/validation2120,별도test4218을사용. batch8과284를동일초기화·FP32·max_length1200·원래dynamicpadding·1epoch로순차비교한다. 앞선1200열고정padding capacity실험과구분한다. GPU는하나씩,가용8GiB/swap+2GiBguard유지.

최신 우선순위: 실제 대규모 세포 처리. Tabula Sapiens Census2024-07-01 원본483152×58482 다운로드/SHA256 완료, raw.X에서300000개 고유 실제 세포 입력 cache 검증 완료. scgpt/results/actual_300k_20261027_221433_021330 에서 batch8→256 순차 추론 진행(1만/10만/30만 누적 지점 기록). GPU 작업 중복 없음. 최초 inference는 gpu_support import 오류로 GPU 시작 전 종료, 경로 수정 후 inference_retry1.log로 재실행. pancreas 전체 batch8 학습/재로드/test 완료;284/256/224 학습은 메모리 guard 중단, 비교 미완료.

사용자 승인으로 일시중단 완료. scGPT actual_300k_20261027_221433_021330: batch8 고유100000세포/903.027초/GPU평균94.772%·최대95%/allocated peak0.389380GiB. 출력100000×512 전부finite·정규화검증 통과, pause_checkpoint.json 저장. retry1 종료코드143(의도적SIGTERM), 다른 GPU compute process 없음 확인. batch8 나머지200000 및batch256 미완료. 다른모델후속실험은취소(범위scGPT만). 사용자재개요청전새GPU작업없음. 재개: python3 scgpt/scripts/large_scale/resume.py actual_300k_20261027_221433_021330. 재개코드는구문검사통과,실제재개검증은아직.

2026-10-28 사용자 출근 후 재개 승인. actual_300k_resume_20261028_093701_696230: 전날batch8 완료100000세포 출력 복사 보존 후100000행부터 재개,남은200000세포 처리 중. 이후같은실제300000세포 batch256 순차비교. resume timing은별도구간으로명시. 기본9종 추가후속없음. 최종출력CPU검증·보고서통합후종료하는finalize.py 대기 중.

2026-10-28 최종 완료: scGPT 실제30만 고유세포 batch8/256 추론 및 전체출력검증 완료. 실행 `actual_300k_resume_20261028_093701_696230`, [최종 보고서](scgpt/reports/scGPT_actual_300k.md). 후속 실험 범위는 scGPT만이며 다른 모델은 진행하지 않는다. 최신 상태는 이 항목을 우선한다.

최종 수동 확인: inference_exit0·finalization complete·전체30만출력검증 통과, nvidia-smi compute process 없음. 사용자 요청 범위 scGPT 후속 완료 및 보고서 반영 완료.

보고서 독자 설명 보완: scGPT_actual_300k.md 및 scGPT_memory_batch.md에 세포×유전자 행렬의 행/열, batch=세포행 수, 유전자ID·발현값 대응, CLS/padding 포함1200토큰 의미, 개수 증가와 순간메모리 차이, 추론batch256 대 학습probe292의 메모리 비교를 추가했다. 추가GPU실험없음.

ESM-2 추가후속 실행중: batch_inference_20261028_112900_601721. UniProt reviewed500서열900~950AA 확보,고유서열중첩prefix batch확대,최대서열950을첫행고정하여실제입력952token길이유지. esm2_t33_650M_UR50D FP32추론,학습아님. 각조건fresh process,가용16GiB·swap+2GiBguard.

ESM-2 후속 긴서열배치추론 batch_inference_20261028_112900_601721 상태 running, 최대성공batch 2, 종료사유 running. 보고서 esm2/reports/ESM2_batch_memory.md.

ESM-2 후속 긴서열배치추론 batch_inference_20261028_112900_601721 상태 complete, 최대성공batch 384, 종료사유 predicted_reservation_exceeds_available_budget. 보고서 esm2/reports/ESM2_batch_memory.md.

Cellpose 후속진행: tile_batch_20261028_134130_265230. 실제BBBC039200이미지520×696 uint16검사·SHA검증후CPU공식정규화cache생성완료,이미지당12개256×256타일. GPU별도fresh container에서BF16 core.run_net 추론배치8→12→24…순차확대. GPU작업하나씩,가용16GiB/swap+2GiBguard.

Cellpose 타일배치후속 tile_batch_20261028_134130_265230 상태 running, 최대성공타일batch 8,80%도달 None,90%도달 None. [보고서](cellpose/reports/Cellpose_batch_memory.md). 추가모델실험자동시작없음.

Cellpose 타일배치후속 tile_batch_20261028_134130_265230 상태 complete, 최대성공타일batch 384,80%도달 False,90%도달 False. [보고서](cellpose/reports/Cellpose_batch_memory.md). 추가모델실험자동시작없음.

사용자자동연속진행승인반영: scripts/remaining_followups/launch.py 실행중, scVI부터시작. 큐 FOLLOWUP_QUEUE.json 및 FOLLOWUP_RESULTS.md, 로그 remaining_followups.log. 모델마다추론batch스윕→출력검증→보고서→다음모델. GPU한개작업씩,실제데이터pool부족은그대로제한으로보고한다. AlphaFold2/Clair3 전용batch스윕미실시사항은적용범위에명시.

scvi 배치후속 complete, 최대성공batch 10176. [보고서](scvi/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

dnabert2 배치후속 complete, 최대성공batch 512. [보고서](dnabert2/reports/Batch_memory_followup.md). 종료사유 predicted_reservation_exceeds_available_budget.

ntv2 배치후속 complete, 최대성공batch 128. [보고서](ntv2/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

rnafm 배치후속 complete, 최대성공batch 76. [보고서](rnafm/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

openspliceai 배치후속 complete, 최대성공batch 156. [보고서](openspliceai/reports/Batch_memory_followup.md). 종료사유 available_unique_input_limit.

남은2종후속실제실행: Clair3 candidate_batch_20261028_185641_089149 기존실제HG003 ONT200kb재호출및전체featurecache성공(P5708/F894 고유feature),batch스윕실행중. AF2 vmap_batch_20261028_185758_495006의115AA실제4query·MSAprocessedfeature CPUcache완료,Clair3 GPU종료후JAXvmap검증및batch1/2/4실행대기. CLI미노출을불가능으로판단했던이전기록을새실험이갱신한다.

clair3 전용추가후속 complete; [결과](clair3/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

af2 전용추가후속 stopped; [결과](colabfold/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

clair3 전용추가후속 complete; [결과](clair3/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

Clair3후속완료: P5708 고유feature batch/peak4.111GiB, F894/peak1.499GiB,양쪽batch별finite·저장재로드·확률일치검증통과. AF2첫시도중첩jit compiler_options 충돌기록보존후 vmap_batch_retry_20261028_190537_289704 새run에서동일Haiku apply의기존tuned옵션을outerjit으로이동해재검증중.

af2 전용추가후속 stopped; [결과](colabfold/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

clair3 전용추가후속 complete; [결과](clair3/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

af2 전용추가후속 complete; [결과](colabfold/reports/Batch_memory_followup.md). 기존미실시기록이후새run으로진행한결과이다.

10종 후속최종정리완료. 이전AlphaFold2/Clair3미실시상태는새실행으로갱신됐다. 구현·정밀도·검증범위차이를포함한최신전체표: [FOLLOWUP_RESULTS.md](FOLLOWUP_RESULTS.md). 추가GPU작업없음.

최종수동검증완료: Clair3 P5708/F894 실제feature 추론·native pipeline 확률일치, AF2 FP32 vmap batch1/2/4 native대조 및7개PDB·finite·CA잔기수 검증완료. AF2 batch4 누적JAX allocator peak2.308GiB,평균GPU94.476%. 원래BF16실패두run보존,현재FP32조건과명확히구분. nvidia-smi compute process 없음 확인. 통합10종 FOLLOWUP_RESULTS.md 재생성완료.

뇌이미지 추가 Cellpose NucMM-M brain_nucmm_20261029_110603_965442 상태 complete. [보고서](cellpose/reports/Cellpose_brain_NucMM.md). 일반뇌MRI가아닌세포핵micro-CT·정답instance검증,기본/배치실험과별도run.

최종보고구조변경: 모델별reports/FINAL.md에기본·후속·추가데이터실험본문을통합했고 README/FINAL_REPORT.md의주요읽기경로도단일최종보고로변경했다. 모든모델80%도달목표는완료되지않았으며scGPT학습용량probe만확인된것으로표기했다. 추가GPU실험없음.
