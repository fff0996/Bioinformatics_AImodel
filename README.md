# DGX Spark 바이오 AI 최종 보고

**먼저 [최종 통합 결과](FINAL_REPORT.md)를 읽으면 된다.** 아래 모델별 최종 보고서에는 기본 조사와 모든 후속 실험이 한 문서에 합쳐져 있다.

## 모델별 최종 보고서

|모델|읽을 파일|
|---|---|
|ESM-2|[최종 보고서](esm2/reports/FINAL.md)|
|AlphaFold2 — ColabFold|[최종 보고서](colabfold/reports/FINAL.md)|
|scVI|[최종 보고서](scvi/reports/FINAL.md)|
|Cellpose|[최종 보고서](cellpose/reports/FINAL.md)|
|DNABERT-2|[최종 보고서](dnabert2/reports/FINAL.md)|
|Nucleotide Transformer V2|[최종 보고서](ntv2/reports/FINAL.md)|
|Clair3|[최종 보고서](clair3/reports/FINAL.md)|
|RNA-FM|[최종 보고서](rnafm/reports/FINAL.md)|
|OpenSpliceAI|[최종 보고서](openspliceai/reports/FINAL.md)|
|scGPT|[최종 보고서](scgpt/reports/FINAL.md)|

## 무엇을 확인할 수 있나?

각 최종 보고서 첫 표에서 **추론/학습 구분 → 입력 모양·배치·정밀도 → 최대 메모리 할당 GiB·공유RAM대비% → 80% 도달 여부와 중단 이유**를 확인한다. 이어지는 본문에 모델정보·데이터출처·전처리·입출력예시·실행시간·GPU-Util·배치실험·정답평가·저장/재로드검증을 함께 수록했다.

**메모리80% 목표는약97.35GiB다. 지금 확인된것은scGPT의고정1200padding 학습용량probe(batch292,100.75GiB/82.79%)뿐이고, 다른모델의80%조건은미확인이다.** 후속실험실행이끝난것과이목표를달성한것은다르다.

GPU-Util은연산활동시간비율,CUDAallocated는실제프로세스메모리할당,CUDAreserved는예약량이다. Spark약121.7GiB는CPU/GPU공유RAM으로전용VRAM이아니다. RSS와CUDA를더하지않는다. AF2는JAX누적peak를사용하며모델별조건이달라숫자만으로성능순위를매기지않는다.

## 원본 기록과 재현

- [측정 기록 및 이전 전체 결과](RESULTS.md)
- [후속 실행 내역](FOLLOWUP_RESULTS.md)
- [저장 범위와 재현 안내](docs/REPRODUCIBILITY.md)
- [실행 이력](STATUS.md)

원본 데이터·가중치·대형배열·가상환경·캐시는Git에올리지않았다. 작은설정/지표JSON·자원CSV·코드·예시그림·출처와해시를포함한다. 이전기본/후속파일은추적용으로보존했지만결과를읽을때는각모델최종보고서하나만보면된다.

## 코드와 모든 기록의 Markdown 보관

실행스크립트는독립파일로배포하지않고모델별최종보고서에전문을수록했다. 설정·자원기록도RECORDS.md의코드블록으로보존한다. [복원 및 원문 해시검증](docs/RESTORE_FILES.md)을통해로컬에서원래파일을다시만들수있다.
