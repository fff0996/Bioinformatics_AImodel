# 최종 통합 보고 — 모델별 입력·추론·학습 메모리

**이 표에서 모델을 고른 뒤 최종 보고서 하나만 읽으면 된다. 각 문서에 기본 조사·입력 형태·추론·학습·배치 후속·평가·검증을 모두 합쳤다.**

## 먼저 알아야 할 결론

메모리80% 목표는allocated약97.35GiB다. 현재이조건을입증한것은scGPT의FP32/1200자리padding/학습용량probe(batch284약80.54%,batch292약82.79%)뿐이다. 일반적인전체세포파인튜닝에서같이달성했다는뜻은아니다. **나머지모델의80%조건은미확인**이다. 후속run을실행했다는것과80%목표를해결했다는것을구분한다.

## 모델별 최종 보고서

|모델|추론 관측최대 GiB|학습 관측최대 GiB|메모리80% 조건|읽을 파일|
|---|---|---|---|---|
|ESM-2|64.416|3.623|미확인|[최종 보고서](esm2/reports/FINAL.md)|
|AlphaFold2 — ColabFold|2.706|미실시|미확인|[최종 보고서](colabfold/reports/FINAL.md)|
|scVI|0.270|0.097|미확인|[최종 보고서](scvi/reports/FINAL.md)|
|Cellpose|43.397|7.985|미확인|[최종 보고서](cellpose/reports/FINAL.md)|
|DNABERT-2|60.007|10.571|미확인|[최종 보고서](dnabert2/reports/FINAL.md)|
|Nucleotide Transformer V2|21.592|9.252|미확인|[최종 보고서](ntv2/reports/FINAL.md)|
|Clair3|4.111|미실시|미확인|[최종 보고서](clair3/reports/FINAL.md)|
|RNA-FM|8.870|0.434|미확인|[최종 보고서](rnafm/reports/FINAL.md)|
|OpenSpliceAI|1.463|2.160|미확인|[최종 보고서](openspliceai/reports/FINAL.md)|
|scGPT|5.477|100.750|학습용량probe만 확인|[최종 보고서](scgpt/reports/FINAL.md)|

학습열에는작업방식이서로다른scVI처음부터학습·ESM부분파인튜닝·RNA-FM선형head학습·scGPT용량probe가함께있다. 숫자만으로학습메모리순위를매기지말고각최종보고서첫표에서**무엇을학습했는지·입력·배치**를확인한다. AF2메모리는JAX프로세스누적peak라PyTorch실행구간allocated와정의가다르다.

원본증거를더확인할때만최종보고서안의run JSON/CSV링크를이용하면된다. 이전기본/후속파일은추적용으로보존하되주요읽기경로는이문서와각모델FINAL.md다.
