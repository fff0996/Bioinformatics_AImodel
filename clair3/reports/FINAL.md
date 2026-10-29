# Clair3 최종 통합 보고서

## 1. 모델과 실제 입력

long-read BAM+참조유전체 → 변이후보특징배열 → 변이확률/VCF. batch는read수가아닌후보feature 배열수.

실제파일·입력/정답예시·전처리와모델구조설명은이문서5절에전문을포함했다. 추론데이터와지도학습label을구분하며,세포/유전자·서열/토큰·이미지/타일의단위를섞지않는다.

## 2. 실행 조건과 측정 기준

표의실제batch·입력shape·정밀도·epoch/step를먼저확인한다. GPU-Util은연산활동시간비율이고,메모리목표80%는약97.35GiB allocated다. 모델별학습범위(전체/부분/head-only/처음부터학습/용량probe)를같은학습으로가정하지않는다. 같은run의시간·메모리를사용하고예측정확도가미평가이면그대로표시한다.

## 3. 조건별 추론·학습 결과 — 한 표

아래시간은각원본조건의실측값이다. 조건별포함/제외범위는4~5절과원본기록에서확인한다. 대부분PyTorch CUDAallocated peak이나AlphaFold2는컴파일/warmup포함JAX프로세스누적peak다. RSS와CUDA를독립메모리처럼더하지않는다.

|작업|실제 입력·고유 데이터 수|동시 GPU batch·모양|학습량/반복|측정시간|처리량/step 참고|최대 메모리 / 공유RAM대비|성능·80% 상태|
|---|---|---|---|---|---|---|---|
|Clair3_P 후보추론|실제HG003 ONT 후보feature고유1개|batch1×[33, 18],FP32모델|18096회/총18096후보|15.00초|1206.37후보/초|0.057 GiB / 0.05%|native pipeline확률대조통과;새variant정확도평가아님;80%미도달|
|Clair3_P 후보추론|실제HG003 ONT 후보feature고유5708개|batch5708×[33, 18],FP32모델|25회/총142700후보|15.10초|9447.45후보/초|4.111 GiB / 3.38%|native pipeline확률대조통과;새variant정확도평가아님;80%미도달|
|Clair3_F 후보추론|실제HG003 ONT 후보feature고유1개|batch1×[89, 33, 8],FP32모델|7026회/총7026후보|15.00초|468.34후보/초|0.178 GiB / 0.15%|native pipeline확률대조통과;새variant정확도평가아님;80%미도달|
|Clair3_F 후보추론|실제HG003 ONT 후보feature고유894개|batch894×[89, 33, 8],FP32모델|116회/총103704후보|15.04초|6896.99후보/초|1.499 GiB / 1.23%|native pipeline확률대조통과;새variant정확도평가아님;80%미도달|
|학습|이번작업미실시|—|—|미실시|—|미측정|학습지원기능과이번실행을구분|

## 4. 결과 해석

같은실행경로에서batch를늘린행끼리메모리와처리량을비교한다. 전체고유데이터수와반복처리건수는다르다. 추론·학습·후처리의측정시간범위가다르면단순속도순위로해석하지않는다. 현재80%에도달한조건을이모델에서확인한것은아니며데이터pool한계·예상guard와실제실패를구분해본문에남겼다. 위주표에는동일조건의시간·메모리만연결했고미실시평가를성능성공으로표현하지않는다.

## 5. 실험 방법·입력 예시·기본/후속·평가·저장 상세


### Clair3 — DGX Spark long-read 변이 호출


run `clair3_20261027_203339_768667` · 상태 **complete** · [summary](../results/clair3_20261027_203339_768667/RECORDS.md#file-ccc0aa43f73b) · 로그 (로컬 보관: `../results/clair3_20261027_203339_768667/benchmark.log`) · [자원 시계열](../results/clair3_20261027_203339_768667/RECORDS.md#file-11cc0da66d78)

#### 모델과 실제 실행 환경

Clair3 공식 소스 v2.0.4, commit `9d1dd3f2475072b6255900e406c34c6b0545fc88`. v2.0.4 이름의 git tag는 없어 실제 main commit과 shared/version.py를 고정했다. 공식 PyTorch `ont/pileup.pt`와 `ont/full_alignment.pt`를 사용하며 Guppy3.6 ONT 데모에 맞춘 구형 ONT 모델이다. 최신 R10.4.1/Dorado 모델과 혼용하지 않는다.

환경 {"torch": "2.13.0a0+8145d630e8.nv26.06", "cuda": "13.3", "gpu": "NVIDIA GB10"}

원래 NGC PyTorch/CUDA 이미지를 보존하고 별도 ARM64 이미지를 만들었다. CFFI native 전처리는 시스템htslib으로 컴파일, phasing은공식 기본whatshap, 선택적 가속용PyPy 대신 CPython3.12를 썼다. Spark의 nvidia-smi memory.free=N/A 때문에 공식 VRAM 잔량 기반 worker 산정을 그대로 쓸 수 없어, GPU predictor worker1개로 명시하는 호환 변경을 적용했다. 모형 forward·배치·변이 호출 알고리즘은 유지했다. 원본·patch는 `results/benchmark_compatibility.patch`에 저장했다.

공식 GPU 배치 정책 predictBatchSize200×5를 고정하며 마지막 batch는 후보 수에 따라 작다. CPU threads4, GPU0, FP32가중치, 실제 입력 특징은 pileup int32/full-alignment int8이며 모델 내부에서 float로 변환/스케일한다. GPU 다른 실험은동시에 실행하지 않았다.

| 모델 | 실제파라미터 | 적재 allocatedGiB | reservedGiB | RSSGiB | 가용GiB |
| --- | --- | --- | --- | --- | --- |
| Clair3_F | 2986522 | 0.011 | 0.025 | 0.619 | 113.028 |
| Clair3_P | 2074520 | 0.008 | 0.023 | 0.698 | 112.746 |

이론 FP32가중치 용량은각 파라미터수×4byte다. 위는 각 worker의 checkpoint 적재를 완료하고 첫forward전의 실측이며 입력tensor/optimizer는없다. 모델 두개는순차적으로 쓰며 파라미터합을한번에 올린 크기로 가정하지 않는다.

#### 실제 입력과 정답

공식 ONT quick demo: GIAB HG003, Guppy3.6, GRCh38_no_alt chr20, chr20:100000–300000(BED0-based half-open). 원본 BAM은coordinate정렬·index·참조 contig길이를 확인했다. GIAB HG003 v4.2.1 VCF와 confident BED도 확보했다. 데이터·모델 URL/바이트/SHA-256은 `data/sources.json`과 `models/ont/sources.json`.

첫 실제 read: {"name": "dc9c6024-1ac4-499b-87b8-2d7e64475b46", "reference_start_0based": 60000, "cigar": "22735S14M1D32M1I6M3D4M1D13M1D21M1D103M1I79M1I80M1D54M1D12M1I19M1I11M1I25M1D79M1D6M1D18M2D88M2D6M1D45M1D9M1D46M1D24M1D9M1I59M1I2M1D32M1D38M1D31M1D46M1D45M1D29M1D5M2D67M1D57M2D11M2D52M1I7M1D24M1D129M1D9M1D13M1I9M3D11M1529I31M1D27M3D4M1D19M1D3M1I9M2D23M1D21M3I11M2D12M1D2M1D50M2D27M1D99M1D10M1D84M1D14M1D29M3D33M2D68M2D49M1D19M1D27M2D28M1D26M1D11M1D45M1D22M1D42M2D47M1I6M2D34M1D40M1D39M1I5M1D40M2D9M1I9M3D37M1D4M1D24M3D42M2D5M1D44M1D2M1I5M1I7M1D4M1D15M1I15M1D23M1I1M1I12M1D9M1I47M1I69M1D19M5I35M1D70M5D63M45D590M1I28M1I46M1D2M2D96M1D1M2D4M2D19M1I11M1D50M2D25M2D4M1I6M1D54M2D58M1D1M2D39M1D1M3D1M1D28M7D72M1D71M1D23M3I4M1I5M1I77M1I2M2D15M1I37M1I82M2I7M1D32M1D134M3D48M1D1M1D19M1I8M1D4M1I115M1D52M2D10M1D15M1D24M1D2M1D77M2D4M1D90M1D30M1I39M1D2M2I5M4I34M1I99M2D24M1D49M2I20M1D35M2D73M2D39M1D4M1D4M1D63M2D47M1D11M1D4M1D4M1D38M1D153M1D30M9I146M2D12M1D7M1D3M1I27M1I41M1I2M1I28M2D25M1D40M1D9M1I4M1I59M1D59M1D4M1D33M2D4M1D2M1I42M1D32M1D19M1D4M1I30M1D10M2D4M1D3M2D19M1I18M1D34M1D4M1I5M1D6M1D4M1D3M2D3M2D8M2D14M3D2M1D18M2D2M1D6M1D4M1D37M1D13M1D3M1I38M2D4M1D22M1D81M1D2M1I18M3D5M1D47M1D74M1D18M1D76M1I31M3I17M2D9M1I32M1I54M1D8M2D92M1I8M1D109M1I50M1I32M1D10M3D6M1D13M2D37M1D112M1I14M1I8M1I16M2D120M1D1M2D51M1D5M2I67M2D6M1D53M1D95M3D33M1D41M1D14M3D5M1D47M2D91M1I31M2D13M1D76M1D1M1D63M1D19M1D92M1D7M5D11M1I10M1D11M1I87M1D68M2D6M1D32M1D3M1D8M3I5M1I37M1D58M3I76M2I30M3I9M2I39M1I15M2I189M1D17M2D6M1D4M1I2M2D9M1I55M2I43M2D150M1D28M1D68M2D1M2D47M1D30M1I126M1I13M1D28M1I12M1I50M1D62M2I53M1D43M1D29M1I37M1D9M2D17M2D13M1D6M1D141M1I13M1D11M2D33M3D210M1D7M2D11M2D10M3I11M4D88M1D40M1D52M1I22M1D66M1I52M2I12M3I33M1D110M1D46M1D1M1D2M1D23M1I5M1I43M1I57M1I49M1D172M1D49M1I105M1I89M1D27M1I2M1I83M1D68M1D54M1I19M1D52M1D6M1D18M2D6M2D42M1I49M1D20M1I81M1D146M2D65M1I138M1I2M1I78M2I19M1I45M1D19M2I76M2D17M1I19M1I48M1I14M1D292M1I4M1D5M1D27M1I64M1I29M1D72M1I77M1D78M2D20M1D30M1D10M1I29M4D9M1I47M1D68M1I70M2I2M2D94M1D54M3D125M1D169M1D30M1D151M2D5M2I2M2D30M1I52M1D140M1D41M1D134M1D60M3D4M2D39M1D27M1I66M2D19M1D3M1D18M1I48M1D2M2I11M1D66M1I110M1I69M2I83M2D24M1I103M1D117M1D118M2I28M1D10M2D35M2I21M1I7M1D33M2D92M2D19M2D74M1I108M1D24M1D76M1I98M1I8M1D84M1D97M2D93M2I68M1I3M1D38M1D15M2D21M2D35M2D35M1I142M1D17M2I7M1D35M1D20M1D49M1D34M1D117M1D15M1D52M1D159M1D34M1D173M1I70M1D2M1D93M1D14M1D4M1I53M1I18M1D78M1D22M1D157M1D4M1D33M4D3M2D5M3I5M4D20M2I51M1D10M2I102M1I5M1D14M1I6M2I7M1I6M1D3M1D31M2D38M3D11M1D14M1D13M1D35M1I12M1D72M2I82M1D31M2I10M2D30M1D37M1D29M4D32M1D28M1D2M1I4M4D4M1D16M2D40M2D2M1D2M1D2M1D5M1D13M2D20M1D32M1D146M8I9M1I9M1I7M2I27M1D3M1D56M2D15M1D38M1I15M5D123M1I9M2I12M1I223M1D268M1I22M2D70M1D3M1D65M1D29M1I29M1D10M1D42M1D2M1D6M3D2M1D29M1D33M2I30M1D88M2D47M1I7M1I31M1D3M4D3M3D14M1I20M2D68M1D13M1D3M1D20M3I82M1I15M1D48M3I18M1I39M2D125M1I3M1I2M1I36M1I2M1I1M3I15M2D63M1D10M1D11M4D62M1D36M1D3M1D10M2D47M2D29M2D47M2D2M4D2M3I3M1D18M1D2M1D18M1D8M2I5M1D9M2D4M2D25M1D12M2D48M1I53M2D97M1D33M1D78M1D15M1D24M1D59M1D24M1D19M4D181M1I15M4I25M1I8M1D28M1D102M2D22M1D2M1D10M1D12M6D85M1I81M1D119M1I23M1D11M2I17M5D3M1D4M2D9M1D6M1D7M2D90M1D8M5D33M2I38M1I6M1D17M2D1M1I87M2D29M1I31M1I44M2I14M2D8M2D48M1D75M1I50M1D6M1I47M1D32M1D38M2D33M1D50M2D17M3I6M1D33M1I29M1I64M1I1M1I6M5I15M2D14M2D12M4D4M2D16M1D7M1I96M1D108M1D22M3I14M2D22M1D19M1D3M1I47M2D29M2D20M3D82M1D15M1D4M4D33M2D80M2I70M1D24M1I21M3I50M1I4M1I5M1I35M1D73M1D58M1D5M3I114M1D4M1D18M1I37M2D18M1I25M2D34M2I51M1I35M2D4M1I9M1D7M1I8M2D24M2I9M1I33M2D4M1I1M2I33M1D56M1I3M1I5M1D38M2I16M1D34M1I23M1D56M1I76M1I32M3D29M4D4M1D7M1D10M10I68M1I32M1D90M1I48M2D21M1D5M1D7M1D2M3D23M1I14M1D40M1I58M3D21M1I85M3D138M1D71M7D4M1I5M1D6M2I11M2I29M2I3M2I26M3D5M1D11M1I17M2I52M3D4M1I60M3D50M1D6M1I85M1D5M1I46M1D61M1I61M1D5M1D59M3D20M2D138M1D40M2I120M1D52M1D133M1D24M1I43M1D21M2I6M1D24M6I128M3D13M4D87M1D10M1D10M2D4M1D14M1D1M1D27M1D12M1D144M1I1M1I29M1D28M1I6M1D8M3D4M1I69M1I33M1D36M1I2M1I11M1I53M1I83M1D13M1I9M2D10M2D16M1D7M2D28M1I2M2D4M1D51M1I69M1D88M2D71M2D59M1I56M2I21M1D131M1D31M3D6M1I96M4D46M1I29M1D5M1D73M1I150M1I40M1I77M1D115M3D9M1D43M1I46M1D37M1D10M1I89M3I27M1D4M1D51M1D52M1D54M1D113M1D29M1I131M1D5M1D43M3D41M3I34M1D30M1I89M4D13M1I4M1D8M1I85M1D130M1I15M1D40M1I15M1D116M4D36M1D21M1D101M1D37M1D77M1I8M1I46M3I4M1D6M1D63M1D62M1I46M2D14M2D13M3D173M1D47M1D72M3I43M1D100M1D4M1D32M1I33M1D86M2D52M4I64M1D1M1D108M1D36M3D18M1D53M4D38M1I55M2D16M1D5M1D49M1D34M1D54M1D46M1I27M2I8M2D126M1D24M1D129M1D61M2D57M1D25M2D21M2D15M1I44M1D84M3D4M2I25M3D1M1I2M1I3M1I26M1D4M1I145M1I28M1D43M2I14M1D12M1D85M2I10M1I43M2I96M1I2M2D20M3I28M1D33M1D49M1I12M2D13M1D36M1I95M1I24M1D28M1I51M1D27M2D4M2D49M1D45M1D35M1D50M2D150M1D24M2D11M3D5M1D31M1I26M1I45M1D12M1D37M1I98M1D30M1D101M2D31M1D9M1D229M1I62M1I135M2D44M2D45M1I16M1I55M1D130M1I198M2D5M2I55M2D28M1D15M2D8M1I23M3D45M1I34M1D3M1D23M1D4M1D30M1D6M1I3M1D3M7D3M1I4M2I10M1I7M3D3M1I215M1D8M1D42M2I31M2D9M1D17M1I73M1D33M1D1M1D64M1I32M1I35M4D11M1D117M1D112M1D17M1I1M2D7M1D13M1I63M3I1M1I7M1I261M2D186M2D9M1D74M1I20M1D52M1I159M1D6M1D28M1D19M1D53M1I8M1I63M1D45M1D61M1D158M1D23M1I8M1D4M2D10M3D160M2D98M1D203M5I2M3I225M1D121M1D149M1I180M1I25M1I7M2I23M1D21M2D108M1I88M3D122M1D8M1I28M1D12M3D6M1D8M3D12M1I13M2D20M1I10M1D9M1D5M1D2M1D2M2I18M1I5M2D4M2I10M2D9M1D6M4D6M2D5M2D4M1D30M1I17M2D8M1I7M1I2M1D8M1D17M1I15M2D7M1D6M1D3M1D4M1D3M1I4M1I120M1D7M1D18M1D42M3D74M1I5M1I26M2I11M2D35M1D13M1D6M1D3M3I1M1D220M8D61M1I24M1D134M4I40M3D60M1I98M1D25M1D56M1D166M1D56M1D50M1I9M1D42M1D2M1D157M2D6M1I26M1D31M2I19M2I16M1I91M1D57M3I30M1D102M1I21M1D2M1D5M1I33M1D4M2I8M1D7M1D20M1D31M1D31M1D51M1D30M1I41M2D39M1D56M1D12M3D60M1D38M1I18M2I62M1D11M2I49M1D29M1D40M2I136M1D14M2D10M2D79M1I48M3D47M1D116M3D25M2D3M1D24M2D50M2D87M2I149M2D3M1D4M1D10M1I3M1D53M2D15M1D24M1D86M3D127M1D28M1D24M2D19M2D120M2I16M1D23M1D5M1D4M1I3M1D14M2D7M1D18M1D31M2I2M1D170M1I76M1D8M2D19M4I1M1I14M1I29M3D1M1D70M1D10M1D14M5D7M1D4M2I5M1D19M3I12M1D37M3I11M1D4M1D19M1D36M3D23M2I41M1D57M1D41M1D20M1I58M3I15M1D14M1D17M1D6M2I3M1I25M3D8M2I12M1D11M1I18M4I6M4D16M1D10M2D7M1D10M1D18M1D32M4D9M2I16M1I10M1I23M1I23M1I25M1D4M2D4M2D24M3D6M2D16M1D69M1D97M1D45M1D9M1D8M1D25M1D39M1D16M1I6M2D85M1D40M2I81M1D48M4D6M1I18M2I122M1D11M1I38M1D14M2D49M2D1M1D20M1I65M1D7M2D8M1D23M2D49M2D9M1I64M1I29M2I19M1D9M2I13M1D175M2D4M2I42M3D7M7D6M1D38M1D69M1I27M1D3M1D5M1I9M1D163M1I21M1I6M1I1M2D59M2I20M4D3M3D1M2I89M1I43M2D35M1D7M3D21M1D24M1D2M2D11M2I5M3D3M2I37M1D5M1D31M1I24M2I3M1I55M1D37M1D30M1I16M2D3M1D37M1I17M1D3M1D15M1I2M1D5M2I12M1I60M1D11M3I2M1I2M4D27M2D43M1D19M1D13M3D10M1I23M1I16M2I33M2D2M2D17M2I19M1I8M2D35M1D14M1D41M1I48M1D84M1I31M1I49M1I21M1D37M1D17M1I1M1I62M3I4M37I22M1D74M1D136M1D82M2I35M1D7M1I3M1I63M1I56M1D26M1I4M1D4M1D7M1D20M1I10M1D5M1D1M1D47M1D44M2I15M1I63M2D16M1D28M1I25M1D27M2D50M2I93M1D55M1D5M3D17M2D66M1D6M1D17M2I13M1D13M1I4M1I38M1D10M1D9M2D47M1D32M1D6M1D5M2I32M1D77M2I5M2D16M1D4M1D22M2D7M1I32M1D1M1D4M2D18M1I42M1D7M1D3M1D25M1D7M1D16M1I29M1I2M1D28M1D17M1I10M1I7M1D11M1D3M3D28M1I15M1D24M1D41M1D7M1I28M3I5M1D9M1D36M1D85M4D62M2D7M1I5M1I42M1D28M2I6M1D3M1D36M1I61M1I16M9I82M1D112M1I3M1D8M1I84M1I2M1I2M6D152M2I14M2D4M1D21M1I102M1D1M2I25M1I36M1I98M1I16M1I26M1I5M1I44M1D14M2I28M1D56M1D127M3I1M2D13M1D86M1I12M1D41M1I63M2I98M1I86M1D15M2D19M1I3M1D13M2D35M2D87M1I10M1I28M1I24M1D2M1D1M3I36M1D27M1D21M2D35M1D63M1D14M1D3M2I11M2D8M2D79M1D8M1D17M1D7M1D13M1D3M1I3M1D18M3D37M1I4M1D4M1D7M1D1M1D15M2D13M1D76M1D1M1D31M3D6M1I13M1I4M1I30M1D93M1D2M1D5M1I39M3D7M1D20M1I92M2I1M1D7M2I28M1D47M1I53M1D6M2D12M1D3M1D52M1I6M1I29M1I47M1D10M2I8M1D36M1I14M1D3M1D152M4I17M2D21M1D6M2I4M2I7M1I21M3D21M1D7M1I46M1D87M2I40M1I14M3I12M2D4M1D6M1I12M1I6M1D7M2D16M1D22M1D5M1I7M1D16M2D18M2D34M6D26M1I58M1D12M5D11M1I51M2I10M1I13M2I10M1I31M1I63M1I51M1D42M1I50M2I2M1D61M1D23M1I4M1I2M1D7M4I4M3D14M2I67M1I2M2D9M1D35M1D3M1I3M1D96M1D126M1I89M1I18M1D195M1I8M1D19M1I47M1I22M2D12M2D24M1D35M1D17M1I31M1I96M1D6M1D34M1D26M1D39M2D27M6D45M1I2M1D22M1I36M3D77M2D22M1D30M1D61M1D38M1D15M1D27M1D22M1I17M2D23M1D80M1I84M3I7M1D39M1D200M1D37M1I198M1D58M1D68M1I23M2D26M5D138M1I58M1D9M1D3M1I52M2I45M2D131M1I39M2I22M1D8M1I37M1I175M1D108M2D4M1I56M1D11M2D214M1I11M2I47M1D197M1I54M2I21M1D57M1I14M2D51M3I12M1D6M3D44M1I62M1D44M1D50M2D3M1I17M1D36M2D32M1I75M1I100M1I34M3I2M1D40M1I54M1D64M1D96M1D155M2I55M1D163M1I5M1I17M1D89M1I16M1I24M2I7M1D126M1D29M1D19M1I9M2D34M2D45M2D71M2D53M1I42M1I25M1D18M1I10M3D76M2D3M2D36M1I19M2D78M2I67M4D5M1D100M2D48M2D32M1I43M1D131M3I12M1D67M1I78M1D8M1I6M1D132M1I73M2I14M1D19M1D20M3D5M1I19M3D16M2I7M2D61M1D6M2I15M2I22M1D6M1D65M5D23M3I42M4D64M2D13M1D17M1D51M3I99M2I77M1D90M1I8M1D71M4D3M2I32M1D73M1D9M1D98M1D14M2D20M2I44M1D1M2I110M1D21M1I43M1I5M1D67M1D11M1I21M1D21M2D9M1I61M1D14M1D39M1D45M2I40M1D43M1D204M3I2M1D71M1I63M1D135M1D144M1D49M1D10M1D149M1I74M1I42M3D49M1D26M1D41M1D93M3D57M5I41M1I135M1I16M1D252M2I84M1D34M1I19M1D16M1D76M1D13M2I14M1I2M1I18M1I27M3I1M1I6M1D3M1D15M1D201M1I33M1D8M1I106M1D48M3D87M1D101M1I5M1I58M1D12M1I3M1I6M1D7M4D41M1D5M1I30M4D54M1I27M2D107M1D72M1D67M1D13M2D48M2D33M1D8M1I68M1D28M1D13M1D47M1I4M2I121M1D128M3D80M1D13M1D209M3I1M1I14M2D52M1I7M1D9M1D16M1I9M2D10M2D48M1D82M1D74M2D16M1I46M1I30M1D8M2I19M1D25M1I41M1D21M1I11M1I23M1I18M3D9M1D4M1D22M1I4M1D5M1D8M1I2M2D57M2D41M1I22M1I7M1D22M3I48M2D36M1I78M1D2M1D3M1D65M2D6M2I23M1I3M1I15M1I17M2I34M2I68M1I15M1I21M2D8M1D1M1D118M1D55M3D9M1D14M1I53M1I18M1D12M1D24M1D3M2D16M1D109M1D66M1I20M1D1M2D15M1D9M1D81M3I12M1I242M1D17M1D2M2D37M2I26M2I42M2D30M1I96M1I72M1D11M2I48M1D40M2D26M2D50M1D16M2D135M1D24M1D72M2I63M1I38M1I39M1D47M2I51M1D84M1D94M1D10M1I15M1D32M1D17M1D41M1D20M1D28M1I5M5D8M2D2M1I72M1D37M3D59M1D27M1I27M1D26M1D15M1I2M1D46M5D81M1D91M1I16M1I116M1D136M1D151M1D13M1I98M4D50M2D29M1D88M1I4M1I51M2I39M1I79M1D41M1D2M1D30M1I13M1I35M1D68M4I38M1D62M2D53M1I23M1I140M1D64M2D46M1I39M2D81M1I11M2D69M2D12M1I89M2I108M2D13M2I3M1D1M1D4M1D16M2I27M3D47M1I3M1D62M3D77M1I31M1I89M3D1M1D38M2D39M6D7M2I41M2D1M1D1M1D86M2I188M1D243M11I75M2I3M1D15M1D169M1I40M1D18M1D157M1D14M2D86M1I27M1I160M2D10M2D12M1I78M1I47M1D1M1D41M2I27M1I20M1I14M1I12M4D35M2I15M1I31M1I69M1I3M1I37M1D37M2D8M6D9M1D17M1D55M1I75M2D27M2D19M1I57M2I13M1D17M1D33M1D119M2D14M2I2M2D66M1D3M1D3M1I37M2D54M1I5M1D3M1D15M1D19M1I22M1D24M1I41M1I31M1D62M1D65M2D50M1D16M1D44M1I148M2I24M5D300M1I63M1D4M1D111M1I106M1I3M1D14M1D54M1D40M1D58M2D37M1D30M1I83M1D4M1D13M3I22M1I82M2I16M1I27M1I52M1I26M1I184M2I37M1D59M1D2M1I13M2I9M1D19M2D43M1I32M2I18M1D46M1D5M1D9M2I11M2D17M1D7M6D37M3D97M1D3M1D75M1I53M2D4M2D2M1D15M1I237M1I1M1I22M2I31M1I6M2I176M1D18M1D29M3D3M2D11M1D6M2D24M1I2M1I26M2D4M2D11M1I27M1D15M4D14M2D5M2I16M1D4M1I30M1D21M2D2M2I2M1I131M1D6M2I11M1D45M2D14M1D36M2I25M1D11M1D15M1I2M1I7M1D19M1D50M1D3M2D59M1D19M1I7M1I16M1D102M1I4M1I17M1D55M2D18M1I17M2D29M1D8M2D13M2D52M1D69M2D33M1D37M2I12M1D8M1D20M3I19M1I20M1D41M4D14M1I55M2I16M1D36M1I27M1I25M1D15M1D8M1D32M1I16M1D6M1D8M2D2M2I12M4I5M2D9M1D47M1D9M2D71M1I26M1D46M4D19M1D102M2D7M2D5M1I150M1D2M1D11M2I27M5D64M1D30M2D2M1D95M1D8M1D16M1D2M2I40M2I42M1I10M2I24M1I27M1D75M1I26M1D86M1D4M1I29M1I35M1I34M1I5M1I36M2I9M1D44M1D4M2D8M2D1M1D17M1I3M1D1M2D7M1D24M2D14M2D21M1I4M1I31M1D14M2I44M1I15M1D33M1D79M1I106M1D41M1D31M4D12M1D7M1I15M2D12M1D83M2D8M1D4M1I2M3D6M1I62M2D73M2D30M3D3M2D16M1D1M1D31M4D31M1I42M1D38M3D24M1D3M1I5M2D1M3D25M1D19M1D5M3D16M1I3M1D7M1I9M3I1M1I14M1D6M3D5M1D21M1I25M1I8M1D35M1D5M1I10M1D55M1D2M1D10M1I79M1D5M1I80M2D13M1I11M2D32M1D29M2D24M1D7M1I49M1D1M1D16M2D13M5D32M2I18M1I12M1D52M1I11M1D13M1I85M1D37M1I12M1I7M1D38M8D1M1D3M1D7M1D46M1D4M2I5M1I4M2D15M1D90M1I64M1D4M1D3M1D5M2D7M4I66M1I15M1I35M1D10M1D22M1D12M1I7M1D13M1D44M1D25M1I4M1D3M1D41M1D24M1I9M2I60M1D6M2D130M2I3M1I7M1D3M1D39M2D6M1I4M1D11M1I15M1D44M5D4M2D5M1I25M1I25M1I115M1I30M1D2M2I28M3I17M2I21M1I130M1D5M2D18M2D14M1D5M1I92M1I25M2D9M1I5M1I7M1D2M1D40M4D85M1I5M2I2M4D4M1D60M1I44M1D11M1I108M2I59M1D39M3D8M1D4M3D78M1D6M1I24M1I78M1D2M2I48M1I13M2I12M1D13M3D54M1I14M72I10M3I4M3I17M2D26M1D21M1I6M1D35M1I6M2D21M1D50M1D3M2I10M1D6M1I17M1D51M1D128M1D17M1D23M4D14M1D5M2D64M1D10M1D11M1I90M3D32M2D4M1I21M3D2M2D5M3I14M1D25M1D44M1D6M1I4M1D14M2D28M2D6M1D51M1I3M2D2M1I3M1D12M2D19M1I61M1I13M1I56M2I19M2D12M1D21M1D4M1I39M6D1M1D12M3D1M1D8M1I9M3D18M1D3M2D67M1D14M2D7M1D12M1D23M2D21M1D4M1D15M1D2M2I12M1I14M4D5M2I21M2I31M3D66M1I34M2I6M1D43M1D23M1D14M1I13M1I6M1D32M4D4M1D4M2I1M1D3M1D40M1D10M2I2M2D3M1D60M1I28M2I10M1D9M3I14M3D32M3D35M2I32M1D86M1I7M1D20M1D25M1D96M1D24M1I80M1I10M4D30M2D12M1D9M1I98M3D81M2I18M1D22M2D51M1I69M2D37M1I26M1D34M4D44M1I7M2I22M2I12M1D10M1D4M1D11M1D103M1I20M1I20M1D16M1D8M1D98M2D50M16D12M2D9M1D63M1D11M1D76M2I33M5D25M3D63M1D18M1I7M2D31M1D21M1D7M2D15M1I82M1D59M1I66M1I13M1D73M1D8M1D13M2D11M1I7M1I65M1I31M1D8M1D18M1I72M1I31M1I191M1D80M1D9M1D65M13D81M1I90M1D5M1I4M1D2M1D6M1D49M1D15M1D31M1D5M1I33M2I74M1D10M3I46M3I22M2D59M1D5M1D84M2I2M1D13M1I18M2D5M2I3M2I2M20I117M1I13M5D41M2D1M1D16M1S", "length": 117851, "mapping_quality": 60, "sequence_first100": "TTACTTCGTTCAGTTACGTATTGCTTTGAAGGCTGGGCTACCTCAGTGTGGAAGGTGGGCAGTTCCACAGGAATGGTGCCAGGGCCAGAGGGCAATACCA", "tags": [["NM", 7874], ["ms", 151392], ["AS", 152519], ["nn", 725], ["tp", "P"], ["cm", 8416], ["s1", 58736], ["s2", 625], ["dv", 0.041600000113248825], ["SA", "chr12,16970,+,21S18762M358D99068S,52,2368;chr3,76029363,+,18278S3125M142I96306S,1,555;chr19,75654,+,21822S1087M10D94942S,0,83;"]]}

| 구간bp | 겹치는read | 평균coverage | 중앙coverage | 최소/최대 | 평균read길이 |
| --- | --- | --- | --- | --- | --- |
| 50000 | 381 | 102.974 | 103.000 | 93/122 | 27879.129 |
| 100000 | 682 | 107.131 | 106.000 | 93/126 | 24254.419 |
| 200000 | 1305 | 107.528 | 107.000 | 84/129 | 20632.951 |

분석 구간 확대와 coverage 확대는 별개다. coverage조건은동일200kb에서 samtools view -s42.5(seed42,fraction0.5)로 read를 downsample했다. 실제 coverage는 각 실행후 다시 기록했다. 긴read는여러구간에겹칠수있어 read개수가단순구간길이에비례하지않는다.

모델은 BAM자체를tensor로보지않는다. pileup후보의33위치×채널특징과 full-alignment후보의read-depth×33위치×채널 특징으로바뀐다. 실제shape와 첫후보tensor/예측을 저장했다. 최종출력은좌표·REF/ALT·genotype·quality를 가진VCF다.

#### 분석 입력별 GPU 사용률·메모리

| 조건 | bp | coverage | read | 후보tensor처리수 | 전체초 | bp/초 | GPU평균/최대% | 시작GiB | worker peakGiB | 증가GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| region_50000_full | 50000 | 102.974 | 381 | 1486 | 16.475 | 3034.986 | 7.290/18.000 | 0.011 | 0.148 | 0.137 |
| region_100000_full | 100000 | 107.131 | 682 | 3134 | 16.695 | 5989.694 | 7.188/20.000 | 0.011 | 0.347 | 0.336 |
| region_200000_full | 200000 | 107.528 | 1305 | 6611 | 17.932 | 11153.156 | 11.265/88.000 | 0.011 | 0.684 | 0.673 |
| region_200000_half | 200000 | 50.804 | 629 | 9308 | 17.233 | 11605.571 | 12.091/58.000 | 0.011 | 0.763 | 0.752 |

- region_50000_full 실제batch입력: `['[102, 33, 18]', '[103, 33, 18]', '[105, 33, 18]', '[108, 33, 18]', '[126, 33, 18]', '[181, 89, 33, 8]', '[89, 33, 18]', '[90, 33, 18]', '[92, 33, 18]', '[93, 33, 18]', '[97, 33, 18]']`; GPU forward+전송/CPU출력 합 0.702초.
- region_100000_full 실제batch입력: `['[178, 33, 18]', '[183, 33, 18]', '[187, 33, 18]', '[192, 33, 18]', '[208, 33, 18]', '[210, 33, 18]', '[215, 33, 18]', '[216, 33, 18]', '[218, 33, 18]', '[221, 33, 18]', '[232, 33, 18]', '[238, 33, 18]', '[444, 89, 33, 8]']`; GPU forward+전송/CPU출력 합 0.788초.
- region_200000_full 실제batch입력: `['[376, 33, 18]', '[384, 33, 18]', '[388, 33, 18]', '[392, 33, 18]', '[398, 33, 18]', '[399, 33, 18]', '[403, 33, 18]', '[414, 33, 18]', '[426, 33, 18]', '[468, 33, 18]', '[528, 33, 18]', '[554, 33, 18]', '[587, 33, 18]', '[894, 89, 33, 8]']`; GPU forward+전송/CPU출력 합 1.196초.
- region_200000_half 실제batch입력: `['[1000, 89, 33, 8]', '[141, 89, 33, 8]', '[453, 33, 18]', '[471, 33, 18]', '[493, 33, 18]', '[545, 33, 18]', '[579, 33, 18]', '[585, 33, 18]', '[595, 33, 18]', '[603, 33, 18]', '[667, 33, 18]', '[670, 33, 18]', '[775, 33, 18]', '[813, 33, 18]', '[918, 33, 18]']`; GPU forward+전송/CPU출력 합 1.552초.

전체시간은worker시작·모델적재·CPU후보생성·I/O·GPU추론·whatshap·후처리·VCF저장을포함하는end-to-end다. GIAB평가·다운샘플·입력감사·정규화는제외했다. 독립warmup/compile시간은N/A이며worker시작비용을전체에포함한다. 순수forward시간은별도기록하고CUDA synchronize했다. 표의 처리량은bp/초이며 후보수분포/coverage가다르면엄밀한비교에한계가있다.

GPU평균/최대는전체pipeline구간의활동시간비율이다. CPU전처리·phasing·I/O기간이포함돼평균이낮을수있다. GPU커널구간만의FLOPS나memory비율이아니다. 전체RAM121.688GiB를90%채우는것은목표가아니다. RSS와CUDA를더하지않는다.

CUDA peak는여러subprocess 중 **가장큰단일worker allocator peak**이며프로세스합계/장치전체점유율이아니다. 각worker모델적재후peak를reset했다. baseline과delta는동일peakworker로맞춰기록했다. 시계열CSV allocated/reserved는감시parent값이므로모델메모리로쓰지않는다. 실제worker메모리는각조건instrumentation/worker_*.jsonl에있다. RSS는parent시계열과worker스냅샷으로구분한다.

#### 결과 검증

호출VCF와truth를bcftools norm(-f참조,-m-any)으로left-normalize/split한뒤,해당분석구간및GIABconfident BED 안의PASS실제ALTallele exact matching을계산했다. genotype포함haplotype-aware hap.py/vcfeval평가가아니며복잡한동등표현변이는FP/FN으로나뉠수있다. 공식quick demo의hap.py수치와직접비교하지않는다.

| 조건 | 종류 | TP | FP | FN | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| region_50000_full | SNP | 57 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| region_50000_full | INDEL | 12 | 0 | 3 | 1.000 | 0.800 | 0.889 |
| region_100000_full | SNP | 208 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| region_100000_full | INDEL | 28 | 0 | 5 | 1.000 | 0.848 | 0.918 |
| region_200000_full | SNP | 404 | 1 | 0 | 0.998 | 1.000 | 0.999 |
| region_200000_full | INDEL | 46 | 4 | 15 | 0.920 | 0.754 | 0.829 |
| region_200000_half | SNP | 404 | 1 | 0 | 0.998 | 1.000 | 0.999 |
| region_200000_half | INDEL | 41 | 1 | 20 | 0.976 | 0.672 | 0.796 |

원본VCF·normalizedVCF·metrics.json·명령·exit code·예측tensor·실제입력예시를각조건폴더에저장했다. 독립프로세스checkpoint reload: [{"model": "Clair3_F", "actual_input_shape": [32, 89, 33, 8], "independent_process": true, "max_abs_difference": 0.0, "passed": true, "original_batch_shape": [181, 89, 33, 8], "note": "first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4"}, {"model": "Clair3_P", "actual_input_shape": [32, 33, 18], "independent_process": true, "max_abs_difference": 0.0, "passed": true, "original_batch_shape": [103, 33, 18], "note": "first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4"}]

공식Train은PyTorch학습을지원하지만이번Clair3과제범위는long-read germline호출·자원측정이다. 파인튜닝은수행하지않았으며학습완료라고쓰지않는다. GIAB단일샘플의작은구간실험이므로새샘플일반화나WGS전체성능은미확인이다.

설치시curl개발라이브러리충돌을전용이미지내에서정리했고,첫호출전처리의pigz누락을보완했다. 실패run과로그를보존했다.

#### 공식 출처

- [Clair3 공식소스·설치·모델](https://github.com/HKU-BAL/Clair3)
- [공식 ONT quick demo](https://github.com/HKU-BAL/Clair3/blob/main/docs/quick_demo/ont_quick_demo.md)
- [공식 PyTorch ONT 가중치](https://www.bio8.cs.hku.hk/clair3/clair3_models_pytorch/ont/)

### 3. 후속 실험 통합 — 배치 확대 및 자원 검증

### Clair3 실제 후보 tensor의 배치 확대 후속 실험


상태 **complete** · candidate_batch_20261028_185641_089149

기존HG003 ONT BAM/GRCh38 chr20:100000~300000/threads4/한GPUworker의공식variant calling을다시실행해실제후보feature tensor 전체를cache했다. 모델입력이나호출결과를바꾸지않는opt-in capture hook을사용했다. CLI옵션이없더라도모델의실제forward 입력을분리해batch를조절할수있음을확인한실험이다. 후보생성과VCF호출은별도단계이며후속시간은CPUcache→GPU nativeforward→CPU확률반환만포함한다.

pileup과full-alignment는다른입력형태/모델로따로측정한다. 같은feature값은중복제거했고가짜zero row/반복복제row를batch에채우지않았다. 원본candidate좌표는capturehook으로내보내지못했으므로고유생물학적위치수대신고유feature row수를보고한다. cache파일SHA256·원본shape/dtype·선택pool은input_audit.json에있다.

|모델|고유feature pool|각후보shape|dtype|
|---|---|---|---|
|Clair3_P|5708|[33, 18]|int8|
|Clair3_F|894|[89, 33, 8]|int8|

|모델|batch|반복|고유후보|총처리|초|후보/초|모델적재GiB|allocated peak GiB|추가GiB|reserved peak GiB|공유RAM대비%|GPU평균/최대%|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Clair3_P|1|18096|1|18096|15.0004|1206.3659|0.0077|0.0573|0.0496|0.0898|0.0471|84.4828|89.0|
|Clair3_P|8|16249|8|129992|15.0008|8665.6579|0.0077|0.063|0.0553|0.0977|0.0518|86.1724|90.0|
|Clair3_P|32|13518|32|432576|15.0008|28836.7784|0.0077|0.0803|0.0725|0.127|0.066|87.3103|91.0|
|Clair3_P|128|1345|128|172160|15.0071|11471.893|0.0077|0.1475|0.1398|0.2773|0.1212|89.1429|95.0|
|Clair3_P|512|293|512|150016|15.0105|9994.0699|0.0077|0.4207|0.413|0.8945|0.3457|92.3571|96.0|
|Clair3_P|1000|147|1000|147000|15.1032|9733.0169|0.0077|0.7687|0.761|1.2422|0.6317|91.2963|96.0|
|Clair3_P|2048|72|2048|147456|15.0313|9809.9265|0.0077|1.5113|1.5036|2.4746|1.242|89.2963|96.0|
|Clair3_P|4096|36|4096|147456|15.2014|9700.1604|0.0077|2.9666|2.9589|4.8828|2.4379|92.1429|96.0|
|Clair3_P|5708|25|5708|142700|15.1046|9447.4542|0.0077|4.1114|4.1037|6.7832|3.3787|88.4815|96.0|
|Clair3_F|1|7026|1|7026|15.002|468.3375|0.0111|0.1784|0.1673|0.2969|0.1466|89.8571|93.0|
|Clair3_F|8|9756|8|78048|15.0008|5202.9243|0.0111|0.0476|0.0365|0.0566|0.0391|87.0|93.0|
|Clair3_F|32|3205|32|102560|15.0042|6835.4007|0.0111|0.1232|0.1121|0.1484|0.1013|91.75|95.0|
|Clair3_F|128|823|128|105344|15.0178|7014.5928|0.0111|0.3766|0.3655|0.5156|0.3095|91.8929|96.0|
|Clair3_F|512|196|512|100352|15.0386|6672.9585|0.0111|0.972|0.9609|1.4902|0.7988|88.1071|96.0|
|Clair3_F|894|116|894|103704|15.0361|6896.9851|0.0111|1.4991|1.488|2.4219|1.2319|91.6786|96.0|

FP32/TF32=False/seed42, 기존공식ont가중치를고정했다. 각batch는새프로세스로실행한다. 전체파라미터/적재기준값/실제입력dtype·shape/출력shape/표본메모리는각worker summary에있다.
원래공식pipeline에서기록한동일후보의출력확률과batch1재추론대조도통과했다(P최대오차4.17e-6,F1.06e-5). native_pipeline_candidate_equivalence.json에저장했다.
전체출력finite·저장재로드정확일치·같은첫후보의batch간확률일치(atol/rtol1e-4)를검증했다. 실제BAM호출VCF도별도보존했다. 이단계는새로운variant accuracy평가나학습이아니며기존truth정확도결과를참조한다.

종료이유: {"Clair3_P": "available_real_candidate_cache_limit", "Clair3_F": "available_real_candidate_cache_limit"}
확보한실제후보pool에서측정한최대batch이며모델절대한계가아니다. GPU사용률은메모리비율과다르다. CUDAallocated/reserved는프로세스allocator,분모는CPU/GPU공유RAM121.688GiB다. RSS와CUDA를더하지않는다. 가용16GiB·swap+2GiB guard,0.5초대기+조회표본간격. RSS/가용량sampled값과CUDApeak를구분한다.

[공식 Clair3](https://github.com/HKU-BAL/Clair3) · [기본 결과](Clair3.md)

원본실행: results/candidate_batch_20261028_185641_089149; 명령·소스·가중치출처·inputcache·출력·자원시계열·exit·검증값을보존한다.

### 기록 해석

본문의과거완료표시는해당run의실행·저장상태이다. 사용자목표인80%메모리도달여부는이문서첫표를우선한다. 미도달/미확인을불가능으로해석하지않으며,학습미실시·데이터pool제한·예상예약guard·실패run을완료된성공조건과구분한다.


## 6. 실제 실행 코드·명령어 전문

<!-- EXECUTED_SCRIPTS_START -->

### 원문과 출처 검증

아래 코드 블록은 원본 내용을 생략하거나 재작성하지 않았다. 각 파일의 원래 경로·원문 SHA256·출처 상태를 함께 기록했다. `실행 run 저장 snapshot`을 우선하고, 현재 소스만 남은 경우에는 당시 파일과 같다고 단정하지 않는다. 동일 바이트 버전은 한 번만 수록하고 해당 경로들을 함께 표시한다.

GitHub에는 `.py`/`.sh`를 별도 파일로 두지 않는다. 코드 블록을 원래 경로에 저장하거나 [복원 안내](../../docs/RESTORE_FILES.md)의 코드로 로컬에서 복원한다. 라이브러리 전체 소스는 이 사용자 실행 스크립트와 구분하며 패키지 버전/출처를 따른다.

### 코드 목록

|원래 파일|원문 SHA256|출처 상태|
|---|---|---|
|`clair3/Dockerfile`|`042a78fd10943f77204509c11fc0da187828bed569ef522383d9f045d363152e`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/audit.py`|`60de3f0256af3c437887d44ec911f949547b5d311f6cc525095d23361b9ca4a0`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/batch_followup/capture.py`|`2d5c7a4f07920135dd918e1602af7e1c00f448aa2504961479429874d7eee39e`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/candidate_batch_20261028_185641_089149/source_capture.py`|`2d5c7a4f07920135dd918e1602af7e1c00f448aa2504961479429874d7eee39e`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/scripts/batch_followup/prepare_cache.py`|`6bf8fe7d725872c0e6220d4b3a66beb60ecd2a745e4290bdda49547c33452a2f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/candidate_batch_20261028_185641_089149/source_prepare_cache.py`|`6bf8fe7d725872c0e6220d4b3a66beb60ecd2a745e4290bdda49547c33452a2f`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/scripts/batch_followup/worker.py`|`e860b7101a462406f629583836ca9ae616b7fd98101aa92c9fd27fd8d06b832c`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/candidate_batch_20261028_185641_089149/source_worker.py`|`e860b7101a462406f629583836ca9ae616b7fd98101aa92c9fd27fd8d06b832c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/scripts/benchmark.py`|`4e1312610a86ee145ac0707ba0a3cb26d0e355a492de7851ea4525cef140439f`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/discover_download.py`|`c80f3c16ef6b26b3629d404a42669612e8fa76b72e9e2a563709914f276ab47d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/instrument.py`|`9130c01daabcb4faca2e3d4437182ab38fa3449ab71d1baf07946acd875aa84c`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/candidate_batch_20261028_185641_089149/source_instrument.py`|`9130c01daabcb4faca2e3d4437182ab38fa3449ab71d1baf07946acd875aa84c`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/scripts/launch.py`|`032dc38ce8ceda6481c9938e762b6cfa50d33fac9c04b53e3e389d4773296e74`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/reload.py`|`1311d548e5698ddb3f660e2c7ed232adc657d8908750554b063598da1d4c60b9`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/report.py`|`1d088fa9f4579dafd0ff4edae1d86c28658db578ae4248497a691388092e6763`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/scripts/setup.py`|`deb5e12b435a19d7e8c8083fb70136fcece1a56cdba24734202356bd70c4b10d`|현재 보관 소스; 실행 당시와의 바이트 동일성은 별도 근거가 없는 경우 미검증|
|`clair3/results/CallVariantsFromCffiGPU.original.py`|`5c55410ab45d1b1ea012ac204b9fd374c46fb2bbbab1dcd0c49353c4a40a3e72`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`clair3/results/CallVariantsFromCffi.original.py`|`024fe816b279a416d84ab487b6948ad2899088fcca607bb431264e0e89a96ace`|실행 run 저장 snapshot; 실패·재시도 버전도 경로 그대로 보존|
|`scripts/gpu_support.py`|`349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2`|실행 계측 공통 helper 현재 보관본; 버전은 각 run 기록과 대조|

<details>
<summary>clair3/Dockerfile — 원문 전문 (280 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/Dockerfile", "sha256": "042a78fd10943f77204509c11fc0da187828bed569ef522383d9f045d363152e", "encoding": "utf-8", "added_final_newline": false} -->
```dockerfile
FROM nvcr.io/nvidia/pytorch:26.06-py3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends samtools bcftools tabix parallel pigz libhts-dev libcurl4-gnutls-dev libssl-dev liblzma-dev libbz2-dev zlib1g-dev libdeflate-dev && apt-get clean
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/audit.py — 원문 전문 (1,716 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/audit.py", "sha256": "60de3f0256af3c437887d44ec911f949547b5d311f6cc525095d23361b9ca4a0", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,pysam,numpy as np
R=Path('/work');bam=R/'data/HG003_chr20_demo.bam';fa=R/'data/GRCh38_no_alt_chr20.fa';b=pysam.AlignmentFile(bam,'rb');f=pysam.FastaFile(fa);assert b.has_index() and f.get_reference_length('chr20')==b.get_reference_length('chr20');rows=[]
for size in [50000,100000,200000]:
 start=100000;end=start+size;(R/f'data/region_{size}.bed').write_text(f'chr20\t{start}\t{end}\n');depth=np.zeros(size,dtype=np.int32)
 for col in b.pileup('chr20',start,end,truncate=True,stepper='samtools',min_base_quality=0,max_depth=10000):depth[col.reference_pos-start]=col.nsegments
 reads=list(b.fetch('chr20',start,end));rows.append({'region_0based_halfopen':['chr20',start,end],'length_bp':size,'reads_overlapping':len(reads),'coverage_mean':float(depth.mean()),'coverage_median':float(np.median(depth)),'coverage_min':int(depth.min()),'coverage_max':int(depth.max()),'read_length_mean':float(np.mean([r.query_length for r in reads]))})
r=next(b.fetch('chr20',100000,300000));out={'bam_header':b.header.to_dict(),'sort_order':b.header.to_dict().get('HD',{}).get('SO'),'reference_length':f.get_reference_length('chr20'),'regions':rows,'first_read':{'name':r.query_name,'reference_start_0based':r.reference_start,'cigar':r.cigarstring,'length':r.query_length,'mapping_quality':r.mapping_quality,'sequence_first100':r.query_sequence[:100],'tags':r.get_tags()},'source':'official Clair3 ONT quick demo HG003 Guppy3.6 ~85x; ONT non-dwell model ont','truth':'GIAB HG003 GRCh38 chr20 v4.2.1 VCF and confident BED'};(R/'data/input_audit.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps({k:v for k,v in out.items() if k not in ['bam_header','first_read']},indent=2))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/batch_followup/capture.py — 원문 전문 (996 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/batch_followup/capture.py", "sha256": "2d5c7a4f07920135dd918e1602af7e1c00f448aa2504961479429874d7eee39e", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,subprocess,json,time
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];env=os.environ.copy();env['CLAIR3_METRICS_DIR']=str(D/'instrumentation');env['CLAIR3_FULL_CANDIDATE_CACHE']=str(D/'candidate_cache');env['PYTHONPATH']='/work/scripts:/work/source';env['PATH']='/work/.venv/bin:'+env['PATH'];cmd=['/work/.venv/bin/python','/work/source/run_clair3.py','--bam_fn=/work/data/HG003_chr20_demo.bam','--ref_fn=/work/data/GRCh38_no_alt_chr20.fa','--threads=4','--platform=ont','--model_path=/work/models/ont','--output='+str(D/'calls'),'--bed_fn=/work/data/region_200000.bed','--python=/work/.venv/bin/python','--pypy=/work/.venv/bin/python','--whatshap=/work/.venv/bin/whatshap','--device=cuda:0','--use_gpu'];(D/'capture_command.json').write_text(json.dumps(cmd,indent=2));t=time.time();p=subprocess.run(cmd,env=env,cwd='/work/source');(D/'capture_exit.json').write_text(json.dumps({'exit_code':p.returncode,'wall_seconds':time.time()-t}));assert p.returncode==0
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "clair3/results/candidate_batch_20261028_185641_089149/source_capture.py", "source": "clair3/scripts/batch_followup/capture.py", "sha256": "2d5c7a4f07920135dd918e1602af7e1c00f448aa2504961479429874d7eee39e"} -->

</details>

<details>
<summary>clair3/scripts/batch_followup/prepare_cache.py — 원문 전문 (1,271 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/batch_followup/prepare_cache.py", "sha256": "6bf8fe7d725872c0e6220d4b3a66beb60ecd2a745e4290bdda49547c33452a2f", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,json,hashlib
import numpy as np
D=Path('/work/results')/os.environ['RUN_ID'];audit={}
for kind in ['Clair3_P','Clair3_F']:
 files=sorted((D/'candidate_cache').glob(kind+'_*.npz'));assert files;rows=[];sources=[];seen=set()
 for file in files:
  v=np.load(file);x=v['X'];sources.append({'file':file.name,'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.file_digest(file.open('rb'),'sha256').hexdigest()})
  for r in x:
   digest=hashlib.sha256(r.tobytes()).hexdigest()
   if digest not in seen:rows.append(r);seen.add(digest)
 a=np.stack(rows);np.save(D/(kind+'_inputs.npy'),a);audit[kind]={'actual_feature_tensor_rows_before_deduplication':sum(s['shape'][0] for s in sources),'unique_feature_rows':len(a),'feature_shape':list(a.shape[1:]),'dtype':str(a.dtype),'source_caches':sources,'identity_limitation':'deduplicated exact feature tensors; genomic candidate coordinates not exported by this capture hook, no invented position IDs','source':'real HG003 ONT BAM, GRCh38 chr20 100000..300000, existing official Clair3 extraction; no zero-filled or synthetic performance inputs'}
(D/'input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v['unique_feature_rows'] for k,v in audit.items()}),flush=True)
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "clair3/results/candidate_batch_20261028_185641_089149/source_prepare_cache.py", "source": "clair3/scripts/batch_followup/prepare_cache.py", "sha256": "6bf8fe7d725872c0e6220d4b3a66beb60ecd2a745e4290bdda49547c33452a2f"} -->

</details>

<details>
<summary>clair3/scripts/batch_followup/worker.py — 원문 전문 (2,680 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/batch_followup/worker.py", "sha256": "e860b7101a462406f629583836ca9ae616b7fd98101aa92c9fd27fd8d06b832c", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import os,sys,time,json,gc
import numpy as np,torch
sys.path[:0]=['/support','/work/source'];from gpu_support import Meter,write_json
from clair3.model import Clair3_P,Clair3_F
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];kind=os.environ['CANDIDATE_MODEL'];B=int(os.environ['BATCH']);C=D/f'{kind}_batch_{B}';C.mkdir();meter=Meter(C);S={'status':'running','model':kind,'batch':B,'task':'actual candidate tensor native model inference, excludes BAM preprocessing/VCF calling'}
try:
 torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.init();meter.phase='loading';t=time.time();model=(Clair3_P if kind=='Clair3_P' else Clair3_F)(predict=True,add_indel_length=kind=='Clair3_F').float().cuda().eval();p=R/'models/ont'/('pileup.pt' if kind=='Clair3_P' else 'full_alignment.pt');ck=torch.load(p,map_location='cpu',weights_only=False);model.load_state_dict(ck.get('state_dict',ck));del ck;torch.cuda.synchronize();S['loading_seconds']=time.time()-t;S['model_only_postload']=meter.memory();S['parameters']=sum(p.numel() for p in model.parameters());S['parameters_trained_in_run']=0
 x=np.load(D/(kind+'_inputs.npy'),mmap_mode='r');assert len(x)>=B;values=np.array(x[:B]);S['input_shape']=list(values.shape);S['input_dtype']=str(values.dtype);S['unique_candidates']=B
 def forward():
  with torch.inference_mode():return model(torch.from_numpy(values).cuda()).cpu().numpy()
 meter.phase='warmup';t=time.time();out=forward();torch.cuda.synchronize();S['warmup_seconds']=time.time()-t;del out;gc.collect();t,base=meter.begin('candidate_inference');repeats=0
 while repeats<2 or time.time()-t<15:out=forward();assert np.isfinite(out).all();torch.cuda.synchronize();repeats+=1
 r=meter.end(t,base,batch_size=B,unique_candidates=B,repeats=repeats,processed_candidates=B*repeats);r['candidates_per_second']=B*repeats/r['seconds'];r['allocated_fraction_shared_ram']=r['memory']['peak_allocated_gib']/r['memory']['system_total_gib'];S['condition']=r;np.save(C/'predictions.npy',out);assert np.array_equal(np.load(C/'predictions.npy'),out);S['output_shape']=list(out.shape);ref=D/(kind+'_reference_first.npy')
 if ref.exists():S['batch_invariance_max_abs']=float(np.abs(np.load(ref)-out[0]).max());assert np.allclose(np.load(ref),out[0],atol=1e-4,rtol=1e-4)
 else:np.save(ref,out[0])
 S['finite_and_exact_save_reload']=True;S['status']='complete';write_json(C/'summary.json',S);print('BATCH_COMPLETE',kind,B,r['memory']['peak_allocated_gib'],flush=True)
except BaseException as e:S['status']='failed';S['error']=repr(e);write_json(C/'summary.json',S);raise
finally:meter.close()
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "clair3/results/candidate_batch_20261028_185641_089149/source_worker.py", "source": "clair3/scripts/batch_followup/worker.py", "sha256": "e860b7101a462406f629583836ca9ae616b7fd98101aa92c9fd27fd8d06b832c"} -->

</details>

<details>
<summary>clair3/scripts/benchmark.py — 원문 전문 (6,709 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/benchmark.py", "sha256": "4e1312610a86ee145ac0707ba0a3cb26d0e355a492de7851ea4525cef140439f", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys,os,json,time,subprocess,hashlib,collections
from pathlib import Path
sys.path.insert(0,'/support')
from gpu_support import Meter,write_json
import numpy as np,pysam,torch
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];meter=Meter(D);S={'status':'running','conditions':[],'settings':{'version':'2.0.4','commit':'9d1dd3f2475072b6255900e406c34c6b0545fc88','threads':4,'gpu_predictor_workers':1,'batch_policy':'official predictBatchSize200*5 for GPU, actual batches logged','platform':'ont','checkpoint':'models/ont/{pileup,full_alignment}.pt','precision':'FP32 weights, original integer input features; model converts/scales input','phasing':'default whatshap intermediate phasing','preprocessing':'native ARM64 CFFI; CPython3.12 replaces optional PyPy accelerator','warmup':'N/A, end-to-end includes each worker load and startup','training':'supported by official Train; this run performs variant calling only'},'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}}
def save():write_json(D/'summary.json',S)
def must(cmd,**kw):
 p=subprocess.run(cmd,**kw);assert p.returncode==0,(cmd,p.returncode)
try:
 # Actual input and truth files were audited before GPU calling.
 S['input_audit']=json.loads((R/'data/input_audit.json').read_text());save();half=R/'data/HG003_half.bam'
 if not half.exists():
  must(['samtools','view','-s','42.5','-b','-o',str(half),str(R/'data/HG003_chr20_demo.bam')]);must(['samtools','index',str(half)])
 normtruth=D/'truth_normalized.vcf.gz';must(['bcftools','norm','-f',str(R/'data/GRCh38_no_alt_chr20.fa'),'-m','-any','-Oz','-o',str(normtruth),str(R/'data/HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz')]);must(['bcftools','index','-t',str(normtruth)])
 for length,bamname in [(50000,'HG003_chr20_demo.bam'),(100000,'HG003_chr20_demo.bam'),(200000,'HG003_chr20_demo.bam'),(200000,'HG003_half.bam')]:
  name=f'region_{length}_'+('half' if 'half' in bamname else 'full');dest=D/name;dest.mkdir();env=os.environ.copy();env['CLAIR3_METRICS_DIR']=str(dest/'instrumentation');env['PYTHONPATH']='/work/scripts:/work/source';env['PATH']='/work/.venv/bin:'+env['PATH'];cmd=['/work/.venv/bin/python','/work/source/run_clair3.py','--bam_fn='+str(R/'data'/bamname),'--ref_fn=/work/data/GRCh38_no_alt_chr20.fa','--threads=4','--platform=ont','--model_path=/work/models/ont','--output='+str(dest/'calls'),'--bed_fn='+str(R/f'data/region_{length}.bed'),'--python=/work/.venv/bin/python','--pypy=/work/.venv/bin/python','--whatshap=/work/.venv/bin/whatshap','--device=cuda:0','--use_gpu'];write_json(dest/'command.json',cmd);t,base=meter.begin(name)
  with (dest/'run.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd='/work/source')
  end=time.time();write_json(dest/'exit_code.json',{'exit_code':code});assert code==0,(name,code);workers=[]
  for p in (dest/'instrumentation').glob('worker_*.jsonl'):workers.extend(json.loads(line) for line in p.read_text().splitlines())
  loads=[r for r in workers if r['event']=='model_only_postload'];forwards=[r for r in workers if r['event']=='forward'];assert loads and forwards and all(r['memory']['allocated_gib']>0 for r in loads)
  if 'model_only_postload' not in S:S['model_only_postload']={**loads[0]['memory'],'system_available_gib':loads[0]['memory']['available_gib'],'system_total_gib':meter.memory()['system_total_gib']};S['parameters']={r['model']:r['total_parameters'] for r in loads}
  row=meter.end(t,base,name=name,mode='variant_calling',bp=length,unique_inputs=1,repeats=1,processed=length,bam=bamname,worker_loads=loads,actual_input_shapes=sorted({str(r['shape']) for r in forwards}),forward_calls=len(forwards),forward_seconds=sum(r['seconds'] for r in forwards),candidate_tensors=sum(r['shape'][0] for r in forwards));row['seconds']=end-t;row['throughput']=length/row['seconds'];row['memory']=dict(peak_allocated_gib=max(r['memory']['peak_allocated_gib'] for r in forwards),peak_reserved_gib=max(r['memory']['peak_reserved_gib'] for r in forwards));row['baseline']={'allocated_gib':max(r['memory']['allocated_gib'] for r in loads)};row['peak_delta_gib']=row['memory']['peak_allocated_gib']-row['baseline']['allocated_gib'];row['memory_scope']='maximum per-worker allocator peak, not sum of processes; multiple models run serially';S['conditions'].append(row);save();print('CONDITION',name,'seconds',row['seconds'],'peak',row['memory']['peak_allocated_gib'],flush=True)
  calls=dest/'calls/merge_output.vcf.gz';assert calls.exists();norm=dest/'calls_normalized.vcf.gz';must(['bcftools','norm','-f',str(R/'data/GRCh38_no_alt_chr20.fa'),'-m','-any','-Oz','-o',str(norm),str(calls)]);must(['bcftools','index','-t',str(norm)])
  # Genomic region vs coverage experiments remain separate; audit actual depth.
  b=pysam.AlignmentFile(str(R/'data'/bamname));depth=np.zeros(length,dtype=np.int32)
  for col in b.pileup('chr20',100000,100000+length,truncate=True,stepper='samtools',min_base_quality=0,max_depth=10000):depth[col.reference_pos-100000]=col.nsegments
  row['coverage_mean']=float(depth.mean());row['reads_overlapping']=b.count('chr20',100000,100000+length);b.close()
  beds=[]
  for line in (R/'data/HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed').read_text().splitlines():
   if not line or line.startswith('#'):continue
   c,a,z,*_=line.split()
   if c=='chr20':beds.append((int(a),int(z)))
  def keys(path):
   out={'SNP':set(),'INDEL':set()};v=pysam.VariantFile(str(path))
   for rec in v.fetch('chr20',100000,100000+length):
    if rec.pos-1<100000 or rec.stop>100000+length or not any(a<=rec.pos-1 and rec.stop<=z for a,z in beds):continue
    if set(rec.filter.keys())-{'PASS','.'}:continue
    gt=next(iter(rec.samples.values())).get('GT',None)
    for i,alt in enumerate(rec.alts or [],1):
     if gt and i not in gt:continue
     if not alt or alt.startswith('<') or alt=='*':continue
     kind='SNP' if len(rec.ref)==len(alt)==1 else 'INDEL';out[kind].add((rec.contig,rec.pos,rec.ref,alt))
   return out
  truth=keys(normtruth);query=keys(norm);accuracy={}
  for k in truth:
   tp=len(truth[k]&query[k]);fp=len(query[k]-truth[k]);fn=len(truth[k]-query[k]);accuracy[k]={'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
  row['normalized_exact_allele_metrics']=accuracy;write_json(dest/'metrics.json',accuracy);row['output_vcf']=str(calls.relative_to(R));row['output_vcf_sha256']=hashlib.file_digest(calls.open('rb'),'sha256').hexdigest();save()
 S['status']='calling_evaluated_pending_reload';save()
except BaseException as e:S['status']='failed';S['error']=repr(e);save();raise
finally:meter.close();save()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/discover_download.py — 원문 전문 (1,169 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/discover_download.py", "sha256": "c80f3c16ef6b26b3629d404a42669612e8fa76b72e9e2a563709914f276ab47d", "encoding": "utf-8", "added_final_newline": false} -->
```python
import urllib.request,json,re,hashlib
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/clair3');base='https://www.bio8.cs.hku.hk/clair3/'
for path in ['clair3_models_pytorch/','clair3_models_pytorch/ont/']:
 try:
  b=urllib.request.urlopen(base+path,timeout=45).read();(R/'results'/('index_'+path.strip('/').replace('/','_')+'.html')).write_bytes(b);print(path,b.decode()[:10000],flush=True)
 except Exception as e:print(path,repr(e),flush=True)
base+='demo/quick_demo/ont/';rows=[]
files=['GRCh38_no_alt_chr20.fa','GRCh38_no_alt_chr20.fa.fai','HG003_chr20_demo.bam','HG003_chr20_demo.bam.bai','HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz','HG003_GRCh38_chr20_v4.2.1_benchmark.vcf.gz.tbi','HG003_GRCh38_chr20_v4.2.1_benchmark_noinconsistent.bed']
for name in files:
 p=R/'data'/name;print('DOWNLOAD',name,flush=True)
 if not p.exists():
  with urllib.request.urlopen(base+name,timeout=120) as f:p.write_bytes(f.read())
 rows.append({'file':name,'url':base+name,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()});(R/'data/sources.json').write_text(json.dumps(rows,indent=2));print('DONE',name,p.stat().st_size,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/instrument.py — 원문 전문 (2,499 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/instrument.py", "sha256": "9130c01daabcb4faca2e3d4437182ab38fa3449ab71d1baf07946acd875aa84c", "encoding": "utf-8", "added_final_newline": false} -->
```python
import os,json,time,hashlib
from pathlib import Path
import numpy as np,torch,psutil
G=2**30

def install(namespace):
 original_load=namespace['_load_torch_checkpoint'];original_predict=namespace['_torch_predict'];state={'first':True}
 def emit(r):
  root=Path(os.environ['CLAIR3_METRICS_DIR']);root.mkdir(parents=True,exist_ok=True)
  with (root/f'worker_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps({'time_unix':time.time(),'pid':os.getpid(),**r})+'\n')
 def mem():return dict(allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,peak_allocated_gib=torch.cuda.max_memory_allocated()/G,peak_reserved_gib=torch.cuda.max_memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,available_gib=psutil.virtual_memory().available/G)
 def load(model,path,device):
  t=time.time();original_load(model,path,device)
  if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  state['model']=type(model).__name__;state['checkpoint']=path;emit(dict(event='model_only_postload',model=state['model'],checkpoint=path,total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),load_seconds=time.time()-t,dtype=str(next(model.parameters()).dtype),tf32=torch.backends.cuda.matmul.allow_tf32,memory=mem()))
 def predict(model,device,X):
  if device.type=='cuda':torch.cuda.synchronize()
  t=time.time();Y=original_predict(model,device,X)
  if device.type=='cuda':torch.cuda.synchronize()
  emit(dict(event='forward',model=state.get('model'),shape=list(X.shape),dtype=str(X.dtype),seconds=time.time()-t,memory=mem()))
  cache_dir=os.environ.get('CLAIR3_FULL_CANDIDATE_CACHE')
  if cache_dir:
   dest=Path(cache_dir);dest.mkdir(parents=True,exist_ok=True);idx=state.get('cache_index',0);state['cache_index']=idx+1;name=f"{state.get('model')}_{os.getpid()}_{idx}.npz";np.savez_compressed(dest/name,X=X,Y=Y);emit(dict(event='full_actual_candidate_cache',file=str(dest/name),model=state.get('model'),shape=list(X.shape),synthetic_rows=0))

  if state['first']:
   root=Path(os.environ['CLAIR3_METRICS_DIR']);np.savez_compressed(root/f'reload_input_{os.getpid()}.npz',X=X[:min(32,len(X))],Y=Y[:min(32,len(Y))]);emit(dict(event='reload_sample',file=f'reload_input_{os.getpid()}.npz',model=state.get('model'),checkpoint=state.get('checkpoint')));state['first']=False
  return Y
 namespace['_load_torch_checkpoint']=load;namespace['_torch_predict']=predict
```
<!-- END_RESTORE_FILE -->

<!-- RESTORE_ALIAS {"path": "clair3/results/candidate_batch_20261028_185641_089149/source_instrument.py", "source": "clair3/scripts/instrument.py", "sha256": "9130c01daabcb4faca2e3d4437182ab38fa3449ab71d1baf07946acd875aa84c"} -->

</details>

<details>
<summary>clair3/scripts/launch.py — 원문 전문 (1,172 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/launch.py", "sha256": "032dc38ce8ceda6481c9938e762b6cfa50d33fac9c04b53e3e389d4773296e74", "encoding": "utf-8", "added_final_newline": false} -->
```python
import subprocess,datetime,json,sys
from pathlib import Path
R=Path('/home/hyein/ai-benchmark/clair3');script=sys.argv[1] if len(sys.argv)>1 else 'benchmark.py';assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
if script=='benchmark.py':run='clair3_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');D=R/'results'/run;D.mkdir();(R/'results/LATEST').write_text(run+'\n')
else:run=(R/'results/LATEST').read_text().strip();D=R/'results'/run
cmd=['sg','docker','-c',f'docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v {R}:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID={run} -e PYTHONPATH=/work/scripts:/work/source -e MPLCONFIGDIR=/work/.cache/matplotlib local/clair3-spark:9d1dd3f /work/.venv/bin/python -u /work/scripts/{script}'];stem=script.removesuffix('.py');(D/(stem+'_command.json')).write_text(json.dumps(cmd,indent=2));print(run,script,flush=True)
with (D/(stem+'.log')).open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
(D/(stem+'_exit.json')).write_text(json.dumps({'exit_code':code}));print('EXIT',code,flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/reload.py — 원문 전문 (2,072 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/reload.py", "sha256": "1311d548e5698ddb3f660e2c7ed232adc657d8908750554b063598da1d4c60b9", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,os,json,gc
sys.path[:0]=['/support','/work/source']
from gpu_support import write_json
import numpy as np,torch
from clair3.model import Clair3_P,Clair3_F
R=Path('/work');D=R/'results'/os.environ['RUN_ID'];s=json.loads((D/'summary.json').read_text());torch.set_num_threads(1);results=[];tested=set()
for row in s['conditions']:
 dest=D/row['name']/'instrumentation'
 for file in dest.glob('worker_*.jsonl'):
  events=[json.loads(l) for l in file.read_text().splitlines()];load=next((e for e in events if e['event']=='model_only_postload'),None);sample=next((e for e in events if e['event']=='reload_sample'),None)
  if not load or not sample or load['model'] in tested:continue
  cl=load['model'];p=load['checkpoint'];p=p if p.endswith('.pt') else p+'.pt';v=np.load(dest/sample['file']);m=(Clair3_P if cl=='Clair3_P' else Clair3_F)(predict=True,add_indel_length=cl=='Clair3_F').cuda().eval();checkpoint=torch.load(p,map_location='cuda');m.load_state_dict(checkpoint.get('state_dict',checkpoint));torch.backends.cuda.matmul.allow_tf32=load['tf32']
  full_shape=next(e['shape'] for e in events if e['event']=='forward');x=np.zeros(full_shape,dtype=v['X'].dtype);x[:len(v['X'])]=v['X']
  with torch.inference_mode():y=m(torch.from_numpy(x).cuda()).cpu().numpy()[:len(v['X'])]
  diff=float(np.abs(y-v['Y']).max());r={'model':cl,'actual_input_shape':list(v['X'].shape),'independent_process':True,'max_abs_difference':diff,'passed':diff<=1e-4,'original_batch_shape':full_shape,'note':'first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4'};results.append(r);write_json(D/'reload.json',results);print(json.dumps(r),flush=True);assert r['passed'];tested.add(cl);del m,checkpoint;gc.collect();torch.cuda.empty_cache()
assert tested=={'Clair3_P','Clair3_F'},tested;write_json(D/'reload.json',results);s['reload']=results;s['status']='complete';write_json(D/'summary.json',s);print(json.dumps(results),flush=True)
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/report.py — 원문 전문 (8,455 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/report.py", "sha256": "1d088fa9f4579dafd0ff4edae1d86c28658db578ae4248497a691388092e6763", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import json,csv
R=Path('/home/hyein/ai-benchmark/clair3');run=(R/'results/LATEST').read_text().strip();D=R/'results'/run;s=json.loads((D/'summary.json').read_text());out=[]
def a(t=''):out.append(t)
def table(h,rows):
 a('| '+' | '.join(h)+' |');a('| '+' | '.join(['---']*len(h))+' |')
 for r in rows:a('| '+' | '.join(map(str,r))+' |')
 a()
def f(x):return 'N/A' if x is None else f'{x:.3f}'
a('# Clair3 — DGX Spark long-read 변이 호출');a();a(f'run `{run}` · 상태 **{s["status"]}** · [summary](../results/{run}/summary.json) · [로그](../results/{run}/benchmark.log) · [자원 시계열](../results/{run}/resources.csv)');a();a('## 모델과 실제 실행 환경');a();a('Clair3 공식 소스 v2.0.4, commit `9d1dd3f2475072b6255900e406c34c6b0545fc88`. v2.0.4 이름의 git tag는 없어 실제 main commit과 shared/version.py를 고정했다. 공식 PyTorch `ont/pileup.pt`와 `ont/full_alignment.pt`를 사용하며 Guppy3.6 ONT 데모에 맞춘 구형 ONT 모델이다. 최신 R10.4.1/Dorado 모델과 혼용하지 않는다.');a();a('환경 '+json.dumps(s['environment']));a();a('원래 NGC PyTorch/CUDA 이미지를 보존하고 별도 ARM64 이미지를 만들었다. CFFI native 전처리는 시스템htslib으로 컴파일, phasing은공식 기본whatshap, 선택적 가속용PyPy 대신 CPython3.12를 썼다. Spark의 nvidia-smi memory.free=N/A 때문에 공식 VRAM 잔량 기반 worker 산정을 그대로 쓸 수 없어, GPU predictor worker1개로 명시하는 호환 변경을 적용했다. 모형 forward·배치·변이 호출 알고리즘은 유지했다. 원본·patch는 `results/benchmark_compatibility.patch`에 저장했다.');a();a('공식 GPU 배치 정책 predictBatchSize200×5를 고정하며 마지막 batch는 후보 수에 따라 작다. CPU threads4, GPU0, FP32가중치, 실제 입력 특징은 pileup int32/full-alignment int8이며 모델 내부에서 float로 변환/스케일한다. GPU 다른 실험은동시에 실행하지 않았다.');a();loads={}
for r in s['conditions']:
 for v in r['worker_loads']:loads.setdefault(v['model'],v)
if loads:table(['모델','실제파라미터','적재 allocatedGiB','reservedGiB','RSSGiB','가용GiB'],[[k,v['total_parameters'],*[f(v['memory'][x]) for x in ['allocated_gib','reserved_gib','rss_gib','available_gib']]] for k,v in loads.items()])
a('이론 FP32가중치 용량은각 파라미터수×4byte다. 위는 각 worker의 checkpoint 적재를 완료하고 첫forward전의 실측이며 입력tensor/optimizer는없다. 모델 두개는순차적으로 쓰며 파라미터합을한번에 올린 크기로 가정하지 않는다.');a();a('## 실제 입력과 정답');a();audit=s['input_audit'];a('공식 ONT quick demo: GIAB HG003, Guppy3.6, GRCh38_no_alt chr20, chr20:100000–300000(BED0-based half-open). 원본 BAM은coordinate정렬·index·참조 contig길이를 확인했다. GIAB HG003 v4.2.1 VCF와 confident BED도 확보했다. 데이터·모델 URL/바이트/SHA-256은 `data/sources.json`과 `models/ont/sources.json`.');a();a('첫 실제 read: '+json.dumps(audit['first_read'],ensure_ascii=False));a();table(['구간bp','겹치는read','평균coverage','중앙coverage','최소/최대','평균read길이'],[[r['length_bp'],r['reads_overlapping'],f(r['coverage_mean']),f(r['coverage_median']),str(r['coverage_min'])+'/'+str(r['coverage_max']),f(r['read_length_mean'])] for r in audit['regions']]);a('분석 구간 확대와 coverage 확대는 별개다. coverage조건은동일200kb에서 samtools view -s42.5(seed42,fraction0.5)로 read를 downsample했다. 실제 coverage는 각 실행후 다시 기록했다. 긴read는여러구간에겹칠수있어 read개수가단순구간길이에비례하지않는다.');a();a('모델은 BAM자체를tensor로보지않는다. pileup후보의33위치×채널특징과 full-alignment후보의read-depth×33위치×채널 특징으로바뀐다. 실제shape와 첫후보tensor/예측을 저장했다. 최종출력은좌표·REF/ALT·genotype·quality를 가진VCF다.');a();a('## 분석 입력별 GPU 사용률·메모리');a();rows=[];flat=[]
for r in s['conditions']:
 rows.append([r['name'],r['bp'],f(r.get('coverage_mean')),r.get('reads_overlapping','N/A'),r['candidate_tensors'],f(r['seconds']),f(r['throughput']),f(r['gpu_util_avg'])+'/'+f(r['gpu_util_max']),f(r['baseline']['allocated_gib']),f(r['memory']['peak_allocated_gib']),f(r['peak_delta_gib'])]);flat.append({'name':r['name'],'region_bp':r['bp'],'coverage_mean':r.get('coverage_mean'),'reads':r.get('reads_overlapping'),'candidates':r['candidate_tensors'],'seconds':r['seconds'],'bp_per_second':r['throughput'],'gpu_util_avg':r['gpu_util_avg'],'gpu_util_max':r['gpu_util_max'],'baseline_allocated_gib':r['baseline']['allocated_gib'],'peak_worker_allocated_gib':r['memory']['peak_allocated_gib'],'peak_worker_reserved_gib':r['memory']['peak_reserved_gib'],'peak_delta_gib':r['peak_delta_gib'],'forward_seconds':r['forward_seconds'],'samples':r['samples']})
table(['조건','bp','coverage','read','후보tensor처리수','전체초','bp/초','GPU평균/최대%','시작GiB','worker peakGiB','증가GiB'],rows)
if flat:
 with (R/'reports/comparison.csv').open('w',newline='') as f0:w=csv.DictWriter(f0,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for r in s['conditions']:a('- '+r['name']+' 실제batch입력: `'+str(r['actual_input_shapes'])+'`; GPU forward+전송/CPU출력 합 '+f(r['forward_seconds'])+'초.')
a();a('전체시간은worker시작·모델적재·CPU후보생성·I/O·GPU추론·whatshap·후처리·VCF저장을포함하는end-to-end다. GIAB평가·다운샘플·입력감사·정규화는제외했다. 독립warmup/compile시간은N/A이며worker시작비용을전체에포함한다. 순수forward시간은별도기록하고CUDA synchronize했다. 표의 처리량은bp/초이며 후보수분포/coverage가다르면엄밀한비교에한계가있다.');a();a('GPU평균/최대는전체pipeline구간의활동시간비율이다. CPU전처리·phasing·I/O기간이포함돼평균이낮을수있다. GPU커널구간만의FLOPS나memory비율이아니다. 전체RAM121.688GiB를90%채우는것은목표가아니다. RSS와CUDA를더하지않는다.');a();a('CUDA peak는여러subprocess 중 **가장큰단일worker allocator peak**이며프로세스합계/장치전체점유율이아니다. 각worker모델적재후peak를reset했다. baseline과delta는동일peakworker로맞춰기록했다. 시계열CSV allocated/reserved는감시parent값이므로모델메모리로쓰지않는다. 실제worker메모리는각조건instrumentation/worker_*.jsonl에있다. RSS는parent시계열과worker스냅샷으로구분한다.');a();a('## 결과 검증');a();a('호출VCF와truth를bcftools norm(-f참조,-m-any)으로left-normalize/split한뒤,해당분석구간및GIABconfident BED 안의PASS실제ALTallele exact matching을계산했다. genotype포함haplotype-aware hap.py/vcfeval평가가아니며복잡한동등표현변이는FP/FN으로나뉠수있다. 공식quick demo의hap.py수치와직접비교하지않는다.');a();metrics=[]
for r in s['conditions']:
 for k,v in r.get('normalized_exact_allele_metrics',{}).items():metrics.append([r['name'],k,v['TP'],v['FP'],v['FN'],f(v['precision']),f(v['recall']),f(v['f1'])])
table(['조건','종류','TP','FP','FN','precision','recall','F1'],metrics);a('원본VCF·normalizedVCF·metrics.json·명령·exit code·예측tensor·실제입력예시를각조건폴더에저장했다. 독립프로세스checkpoint reload: '+json.dumps(s.get('reload',{'status':'pending'})));a();a('공식Train은PyTorch학습을지원하지만이번Clair3과제범위는long-read germline호출·자원측정이다. 파인튜닝은수행하지않았으며학습완료라고쓰지않는다. GIAB단일샘플의작은구간실험이므로새샘플일반화나WGS전체성능은미확인이다.');a();a('설치시curl개발라이브러리충돌을전용이미지내에서정리했고,첫호출전처리의pigz누락을보완했다. 실패run과로그를보존했다.');a();a('## 공식 출처');a();a('- [Clair3 공식소스·설치·모델](https://github.com/HKU-BAL/Clair3)\n- [공식 ONT quick demo](https://github.com/HKU-BAL/Clair3/blob/main/docs/quick_demo/ont_quick_demo.md)\n- [공식 PyTorch ONT 가중치](https://www.bio8.cs.hku.hk/clair3/clair3_models_pytorch/ont/)');(R/'reports/Clair3.md').write_text('\n'.join(out)+'\n');print(R/'reports/Clair3.md')
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/scripts/setup.py — 원문 전문 (842 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/scripts/setup.py", "sha256": "deb5e12b435a19d7e8c8083fb70136fcece1a56cdba24734202356bd70c4b10d", "encoding": "utf-8", "added_final_newline": false} -->
```python
from pathlib import Path
import sys,subprocess,importlib.metadata,json
R=Path('/work');(R/'constraints.txt').write_text('torch=='+importlib.metadata.version('torch')+'\n');subprocess.run([sys.executable,'-m','venv','--system-site-packages','/work/.venv'],check=True)
subprocess.run(['/work/.venv/bin/pip','install','-c','/work/constraints.txt','--cache-dir','/work/.cache/pip','hdf5plugin','numexpr','tqdm','cffi','torchmetrics','pysam','whatshap','mpmath==1.2.1','intervaltree'],check=True)
subprocess.run(['/work/.venv/bin/python','build.py'],cwd='/work/source',check=True)
(R/'results/setup_done.json').write_text(json.dumps({'torch':importlib.metadata.version('torch'),'source_commit':'9d1dd3f2475072b6255900e406c34c6b0545fc88','native':'CFFI ARM64 compiled with system htslib; whatshap phasing; CPython preprocessing instead of PyPy'}))
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/results/CallVariantsFromCffiGPU.original.py — 원문 전문 (32,174 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/results/CallVariantsFromCffiGPU.original.py", "sha256": "5c55410ab45d1b1ea012ac204b9fd374c46fb2bbbab1dcd0c49353c4a40a3e72", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys
import logging
from argparse import ArgumentParser, SUPPRESS
import subprocess
import os

from shared.utils import str2bool, log_error, log_warning, str_none
logging.basicConfig(format='%(message)s', level=logging.INFO)

file_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
main_entry = os.path.join(file_directory, "clair3.py")

def rocm_smi_free_memory():
    """Free VRAM in MB per physical GPU id from rocm-smi, or {} when it is unavailable."""
    try:
        output = subprocess.check_output(
            "rocm-smi --showmeminfo vram --csv".split(), stderr=subprocess.DEVNULL).decode('ascii')
    except (OSError, subprocess.CalledProcessError):
        return {}

    rows = [line.split(',') for line in output.splitlines() if line.strip()]
    if not rows:
        return {}
    header = [cell.strip().lower() for cell in rows[0]]
    total_idx = next((i for i, cell in enumerate(header) if 'total memory' in cell), None)
    used_idx = next((i for i, cell in enumerate(header) if 'used memory' in cell), None)
    if total_idx is None or used_idx is None:
        return {}

    free_memory = {}
    for row in rows[1:]:
        if len(row) <= max(total_idx, used_idx):
            continue
        # rocm-smi labels rows 'card0', 'card1', ...; the suffix is the physical id.
        name = row[0].strip()
        gpu_id = name[len('card'):] if name.startswith('card') else name
        if not gpu_id.isdigit():
            continue
        try:
            free_memory[int(gpu_id)] = (int(row[total_idx]) - int(row[used_idx])) // 1024 // 1024
        except ValueError:
            continue
    return free_memory

def get_gpu_memory(gpu_id):
    """Free VRAM in MB for a *physical* GPU id, via nvidia-smi (CUDA) or rocm-smi (ROCm).

    Queried out of process on purpose: creating a GPU context here would both consume
    VRAM and skew the very reading used to size the workers.
    """
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv "
        if gpu_id is not None:
            command += f" --id={gpu_id}"
        memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        return [int(x.split()[0]) for x in memory_free_info]
    except (OSError, subprocess.CalledProcessError):
        pass

    free_memory = rocm_smi_free_memory()
    if not free_memory:
        raise RuntimeError("Neither nvidia-smi nor rocm-smi is available to query GPU memory")
    if gpu_id is None:
        return [free_memory[key] for key in sorted(free_memory)]
    return [free_memory[gpu_id]]

def get_all_gpu_ids():
    """Physical GPU ids on this host, as reported by nvidia-smi (CUDA) or rocm-smi (ROCm).

    Both tools ignore CUDA_VISIBLE_DEVICES, so these are physical indices -- the same
    numbering that `--device=cuda:N` and CUDA_VISIBLE_DEVICES themselves refer to.
    """
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=index --format=csv,noheader".split()).decode('ascii')
        nvidia_ids = sorted(int(token) for token in output.split() if token.strip().isdigit())
    except (OSError, subprocess.CalledProcessError):
        nvidia_ids = []
    return nvidia_ids or sorted(rocm_smi_free_memory())

def parse_device_ids(device):
    """Turn a `--device` value such as 'cuda:3' or 'cuda:0,1' into physical GPU ids."""
    parts = [part.strip() for part in device.split("cuda:")[-1].split(",")]
    if not parts or any(not part.isdigit() for part in parts):
        print(log_error("Please specify the device as '--device=cuda:0' or '--device=cuda:0,1'"), file=sys.stderr)
        sys.exit(1)
    return [int(part) for part in parts]

def check_gpu_memory(memory, device_ids=None, print_log=True):
    """Work out how many worker slots each GPU can take.

    device_ids are *physical* GPU ids, as requested through --device or
    CUDA_VISIBLE_DEVICES; None means every GPU on the host. The ids handed back are
    physical too, because that is what the workers put into CUDA_VISIBLE_DEVICES and
    what `nvidia-smi --id=` and `rocm-smi` report. Mixing those physical ids with the
    0-based indices of the visible devices is what used to send every worker to GPU 0
    (issue #472).
    """
    all_gpu_ids = get_all_gpu_ids()
    if not all_gpu_ids:
        print(log_error("No GPU available, Please disabling --use_gpu for variant calling, exiting."), file=sys.stderr)
        sys.exit(1)

    if device_ids is None:
        device_ids = all_gpu_ids
    unknown_ids = [gpu_id for gpu_id in device_ids if gpu_id not in all_gpu_ids]
    if unknown_ids:
        print(log_error("Requested GPU id(s) {} not found on this host (available: {}), exiting.".format(
            unknown_ids, all_gpu_ids)), file=sys.stderr)
        sys.exit(1)

    gpu_id_list = []
    for gpu_id in device_ids:
        free_mem = get_gpu_memory(gpu_id=gpu_id)[0]  # Convert to MB
        gpu_threads = free_mem // memory
        gpu_id_list += [gpu_id] * gpu_threads
        if print_log:
            print(f"GPU {gpu_id} free memory: {free_mem} MB, assigning {memory} MB per thread, {gpu_threads} threads available")

    if len(gpu_id_list) == 0:
        print(log_error("No memory in GPU, Please assign GPU memory first, exiting."), file=sys.stderr)
        sys.exit(1)
    return gpu_id_list

def Run(args):
    try:
        rc = subprocess.check_call('time', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time = 'time '
    except subprocess.CalledProcessError as e:
        time = ''
    full_alignment_cpu_threads = max(args.cpu_threads//4, 1)
    prefix = "full_alignment" if not args.pileup else "pileup"
    dwell_flag = ' --enable_dwell_time True' if args.enable_dwell_time else ''

    pileup_per_thread_gpu_memory = 5000  # GB
    full_alignment_per_thread_gpu_memory = 8000  # GB

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")

    if args.device and args.device != "EMPTY":
        if not args.device.startswith('cuda:'):
            print(log_error("Please specify the device as '--device=cuda:0' or '--device=cuda:0,1'"), file=sys.stderr)
            sys.exit(1)
        device_ids = parse_device_ids(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(item) for item in device_ids])
    elif cuda_visible_devices not in ("-1", ""):
        # No --device given: honour a preset CUDA_VISIBLE_DEVICES instead of quietly
        # using every GPU on the host.
        device_ids = [int(x) for x in cuda_visible_devices.split(",") if x.strip().isdigit()] or None
        if device_ids:
            print(f"[INFO] ENV 'CUDA_VISIBLE_DEVICES' is set to {device_ids}, using these GPUs only.")
    else:
        device_ids = None

    gpu_id_list = check_gpu_memory(pileup_per_thread_gpu_memory if args.pileup else full_alignment_per_thread_gpu_memory, device_ids, print_log=False)

    if args.pileup:
        cp_command = time + args.parallel + ' -C " " '
        cp_command += ' --joblog ' + args.output_dir + '/log/parallel_1_pileup_create_tensor.log'
        cp_command += ' -j ' + str(args.cpu_threads*2)
        cp_command += ' --retries 4'
        cp_command += ' ' + args.python + ' ' + main_entry + " CallVariantsFromCffi"
        cp_command += ' --chkpnt_fn ' + args.chkpnt_fn
        cp_command += ' --bam_fn ' + args.bam_fn
        cp_command += " --call_fn " + os.path.join(args.call_fn, "pileup_{1}_{2}_call.vcf.gz")
        cp_command += ' --sampleName ' + args.sampleName
        cp_command += ' --ref_fn ' + args.ref_fn if args.ref_fn else ''
        cp_command += ' --vcf_fn ' + args.vcf_fn if args.vcf_fn != "EMPTY" else ''
        cp_command += ' --extend_bed ' + args.output_dir + "/tmp/split_beds/{1}"
        if args.bed_fn != "" and args.bed_fn is not None:
            cp_command += ' --bed_fn ' + str(args.bed_fn)
        cp_command += ' --threads ' + str(args.internal_threads)
        cp_command += ' --ctgName {1} '
        cp_command += ' --chunk_id {2} '
        cp_command += ' --chunk_num {3} '
        cp_command += ' --platform ' + args.platform
        cp_command += ' --fast_mode ' + str(args.fast_mode)
        cp_command += ' --snp_min_af ' + str(args.snp_min_af)
        cp_command += ' --indel_min_af ' + str(args.indel_min_af)
        cp_command += ' --minMQ ' + str(args.minMQ)
        cp_command += ' --minCoverage ' + str(args.minCoverage)
        cp_command += ' --call_snp_only ' + str(args.call_snp_only)
        cp_command += ' --enable_variant_calling_at_sequence_head_and_tail ' + str(
            args.enable_variant_calling_at_sequence_head_and_tail)
        cp_command += ' --gvcf ' + str(args.gvcf)
        cp_command += ' --base_err ' + str(args.base_err)
        cp_command += ' --gq_bin_size ' + str(args.gq_bin_size)
        cp_command += ' --enable_long_indel ' + str(args.enable_long_indel)
        cp_command += ' --samtools ' + str(args.samtools)
        cp_command += ' --temp_file_dir ' + args.output_dir + "/tmp/gvcf_tmp_output"
        cp_command += ' --pileup '
        cp_command += ' --keep_iupac_bases ' + str(args.keep_iupac_bases)
        cp_command += ' --cmd_fn ' + str(args.cmd_fn)
        cp_command += ' --use_gpu ' + str(args.use_gpu)
        cp_command += " --tensor_can_fn " + args.output_dir + f"/tmp/pileup_output/{prefix}" + "_{1}_{2}"
        cp_command += ' :::: ' + args.output_dir + "/tmp/CHUNK_LIST"

        try:
            return_code = subprocess.check_call(cp_command, shell=True, stdout=sys.stdout)
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                "ERROR in Pileup tensor generation, THE FOLLOWING COMMAND FAILED: {}\n".format(cp_command))
            exit(1)

        file_list = []
        ctg_set = set()
        with open(args.output_dir + "/tmp/CHUNK_LIST") as f:
            for row in f:
                ctg, chunk_id, chunk_num = row.rstrip().split(' ')
                if os.path.exists(
                        os.path.join(args.output_dir, 'tmp', 'pileup_output', f'{prefix}_{ctg}_{chunk_id}.npy')):
                    file_list.append(
                        os.path.join(args.output_dir, 'tmp', 'pileup_output', f'{prefix}_{ctg}_{chunk_id}'))
                    ctg_set.add(ctg)
        if len(file_list) == 0:
            return
        logging.info("[INFO] Found {} pileup tensor files to process".format(len(file_list)))


        # check gpu resource again, split the file based on the GPU resource
        gpu_id_list = check_gpu_memory(pileup_per_thread_gpu_memory, device_ids)

        total_gpu_treads = len(gpu_id_list)
        each_gpu_file_count = len(file_list) // total_gpu_treads if len(file_list) % total_gpu_treads == 0 else len(
            file_list) // total_gpu_treads + 1

        all_gpu_chunk_file_list = os.path.join(args.output_dir, 'tmp', f'{prefix}_all_gpu_chunk')
        with open(all_gpu_chunk_file_list, 'w') as all_f:
            for i in range(total_gpu_treads):
                gpu_file_list = file_list[i * each_gpu_file_count: (i + 1) * each_gpu_file_count]
                if len(gpu_file_list) == 0:
                    continue

                gpu_chunk_num_fn = os.path.join(args.output_dir, 'tmp', f'{prefix}_gpu_chunk_{i}')
                with open(gpu_chunk_num_fn, 'w') as f:
                    for fn in gpu_file_list:
                        f.write(fn + '\n')
                all_f.write(' '.join([str(i), str(gpu_id_list[i]), gpu_chunk_num_fn]) + '\n')


        for f in os.listdir(args.call_fn):
            if f.startswith(prefix) and f.endswith('.vcf'):
                os.remove(os.path.join(args.call_fn, f))

        cp_command = time + args.parallel + ' -C " " '
        cp_command += ' --joblog ' + args.output_dir + '/log/parallel_1_pileup_call_variant.log'
        cp_command += ' -j ' + str(total_gpu_treads)
        cp_command += ' --retries 4'
        cp_command += ' ' + args.python + ' ' + main_entry + " CallVariantsFromCffi"
        cp_command += ' --chkpnt_fn ' + args.chkpnt_fn
        cp_command += ' --bam_fn ' + args.bam_fn
        cp_command += " --call_fn " + os.path.join(args.call_fn, f"{prefix}_" + "{1}.vcf")
        cp_command += ' --sampleName ' + args.sampleName
        cp_command += ' --ref_fn ' + args.ref_fn if args.ref_fn else ''
        cp_command += ' --vcf_fn ' + args.vcf_fn if args.vcf_fn != "EMPTY" else ''
        if args.bed_fn != "" and args.bed_fn is not None:
            cp_command += ' --bed_fn ' + str(args.bed_fn)
        cp_command += ' --threads ' + str(args.internal_threads)
        cp_command += ' --platform ' + args.platform
        cp_command += ' --fast_mode ' + str(args.fast_mode)
        cp_command += ' --snp_min_af ' + str(args.snp_min_af)
        cp_command += ' --indel_min_af ' + str(args.indel_min_af)
        cp_command += ' --minMQ ' + str(args.minMQ)
        cp_command += ' --minCoverage ' + str(args.minCoverage)
        cp_command += ' --call_snp_only ' + str(args.call_snp_only)
        cp_command += ' --enable_variant_calling_at_sequence_head_and_tail ' + str(
            args.enable_variant_calling_at_sequence_head_and_tail)
        cp_command += ' --gvcf ' + str(args.gvcf)
        cp_command += ' --base_err ' + str(args.base_err)
        cp_command += ' --gq_bin_size ' + str(args.gq_bin_size)
        cp_command += ' --enable_long_indel ' + str(args.enable_long_indel)
        cp_command += ' --samtools ' + str(args.samtools)
        cp_command += ' --temp_file_dir ' + args.output_dir + "/tmp/gvcf_tmp_output"
        cp_command += ' --pileup '
        cp_command += ' --keep_iupac_bases ' + str(args.keep_iupac_bases)
        cp_command += ' --cmd_fn ' + str(args.cmd_fn)
        cp_command += ' --gpu_id {2} '
        cp_command += ' --cpu_threads ' + str(args.cpu_threads)
        cp_command += ' --use_gpu ' + str(args.use_gpu)
        cp_command += " --output_tensor_can_fn_list {3} "
        cp_command += ' :::: ' + all_gpu_chunk_file_list

        try:
            return_code = subprocess.check_call(cp_command, shell=True, stdout=sys.stdout)
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                "ERROR in Pileup model calling, THE FOLLOWING COMMAND FAILED: {}\n".format(cp_command))
            exit(1)

        print("[INFO] Removing temporary tensor files...")
        for f in file_list:
            if os.path.exists(f + ".npy"):
                os.remove(f + ".npy")
            if os.path.exists(f + ".info"):
                os.remove(f + ".info")

    else:
        # create full-alignment tensor
        ct_command = '(' + time + args.parallel + ' -C " "'
        ct_command += ' --joblog ' + args.output_dir + '/log/parallel_6_full_alignment_create_tensor.log'
        ct_command += ' -j ' + str(args.threads)
        ct_command += ' --retries 4'
        ct_command += ' ' + args.python + ' ' + main_entry + " CallVariantsFromCffi"
        ct_command += ' --chkpnt_fn ' + args.chkpnt_fn
        ct_command += ' --bam_fn ' + args.bam_fn
        ct_command += ' --sampleName ' + args.sampleName
        ct_command += ' --vcf_fn ' + args.vcf_fn if args.vcf_fn != "EMPTY" else ''
        ct_command += ' --ref_fn ' + args.ref_fn if args.ref_fn else ''
        ct_command += ' --full_aln_regions {1}'
        ct_command += ' --ctgName {1/.} '
        ct_command += " --add_indel_length" if args.add_indel_length else ''
        ct_command += ' --no_phasing_for_fa ' + str(args.no_phasing_for_fa) if args.no_phasing_for_fa else ''
        ct_command += ' --minMQ ' + str(args.minMQ)
        ct_command += ' --minCoverage ' + str(args.minCoverage)
        ct_command += ' --phased_vcf_fn ' + os.path.join(args.phased_vcf_fn,
                                                         "phased_{1/.}.vcf.gz") if args.phased_vcf_fn else ''
        ct_command += ' --gvcf ' + str(args.gvcf)
        ct_command += ' --enable_long_indel ' + str(args.enable_long_indel)
        ct_command += ' --samtools ' + args.samtools
        ct_command += ' --use_gpu ' + str(args.use_gpu)
        ct_command += ' --keep_iupac_bases ' + str(args.keep_iupac_bases)
        ct_command += ' --cmd_fn ' + args.cmd_fn if args.cmd_fn else ''
        ct_command += ' --platform ' + args.platform
        ct_command += dwell_flag
        ct_command += " --tensor_can_fn " + args.output_dir + "/tmp/full_alignment_output/{1/}"
        ct_command += ' :::: ' + args.full_aln_files
        ct_command += ' ) 2>&1 | tee ' + args.output_dir + '/log/6_full_alignment_create_tensor.log'

        try:
            return_code = subprocess.check_call(ct_command, shell=True, stdout=sys.stdout)
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                "ERROR in Full-alignment create tensor, THE FOLLOWING COMMAND FAILED: {}\n".format(ct_command))
            exit(1)

        # check gpu resource again, split the file based on the GPU resource
        gpu_id_list = check_gpu_memory(full_alignment_per_thread_gpu_memory, device_ids)


        file_list = []
        for f in open(args.full_aln_files, 'r').readlines():
            f = f.strip()
            if not f:
                continue
            suffix = f.split('/')[-1]
            if os.path.exists(args.output_dir + f"/tmp/full_alignment_output/{suffix}.npy"):
                file_list.append(args.output_dir + f"/tmp/full_alignment_output/{suffix}")

        total_gpu_treads = len(gpu_id_list)
        each_gpu_file_count = len(file_list) // total_gpu_treads if len(file_list) % total_gpu_treads == 0 else len(
            file_list) // total_gpu_treads + 1

        all_gpu_chunk_file_list = os.path.join(args.output_dir, 'tmp', f'{prefix}_all_gpu_chunk')
        with open(all_gpu_chunk_file_list, 'w') as all_f:
            for i in range(total_gpu_treads):
                gpu_file_list = file_list[i * each_gpu_file_count: (i + 1) * each_gpu_file_count]
                if len(gpu_file_list) == 0:
                    continue

                gpu_chunk_num_fn = os.path.join(args.output_dir, 'tmp', f'{prefix}_gpu_chunk_{i}')
                with open(gpu_chunk_num_fn, 'w') as f:
                    for fn in gpu_file_list:
                        f.write(fn + '\n')
                all_f.write(' '.join([str(i), str(gpu_id_list[i]), gpu_chunk_num_fn]) + '\n')

        for f in os.listdir(args.call_fn):
            if f.startswith(prefix) and f.endswith('.vcf'):
                os.remove(os.path.join(args.call_fn, f))

        #full-alignment variant calling
        ct_command = '(' + time + args.parallel + ' -C " "'
        ct_command += ' --joblog ' + args.output_dir + '/log/parallel_6_full_alignment_call_variant.log'
        ct_command += ' -j ' + str(total_gpu_treads)
        ct_command += ' --retries 4'
        ct_command += ' ' + args.python + ' ' + main_entry + " CallVariantsFromCffi"
        ct_command += ' --chkpnt_fn ' + args.chkpnt_fn
        ct_command += ' --bam_fn ' + args.bam_fn
        ct_command += ' --call_fn ' + os.path.join(args.call_fn, f"{prefix}_" + "{1}.vcf")
        ct_command += ' --sampleName ' + args.sampleName
        ct_command += ' --vcf_fn ' + args.vcf_fn if args.vcf_fn != "EMPTY" else ''
        ct_command += ' --ref_fn ' + args.ref_fn if args.ref_fn else ''
        ct_command += " --add_indel_length" if args.add_indel_length else ''
        ct_command += ' --no_phasing_for_fa ' + str(args.no_phasing_for_fa) if args.no_phasing_for_fa else ''
        ct_command += ' --minMQ ' + str(args.minMQ)
        ct_command += ' --minCoverage ' + str(args.minCoverage)
        ct_command += ' --phased_vcf_fn ' + os.path.join(args.phased_vcf_fn,
                                                         "phased_{2}.vcf.gz") if args.phased_vcf_fn else ''
        ct_command += ' --gvcf ' + str(args.gvcf)
        ct_command += ' --enable_long_indel ' + str(args.enable_long_indel)
        ct_command += ' --samtools ' + args.samtools
        ct_command += ' --use_gpu ' + str(args.use_gpu)
        ct_command += ' --keep_iupac_bases ' + str(args.keep_iupac_bases)
        ct_command += ' --cmd_fn ' + args.cmd_fn if args.cmd_fn else ''
        ct_command += ' --platform ' + args.platform
        ct_command += ' --gpu_id {2}'
        ct_command += ' --cpu_threads ' + str(full_alignment_cpu_threads)
        ct_command += ' --use_gpu ' + str(args.use_gpu)
        ct_command += dwell_flag
        ct_command += ' --output_tensor_can_fn_list {3} '
        ct_command += ' :::: ' + all_gpu_chunk_file_list
        ct_command += ' ) 2>&1 | tee ' + args.output_dir + '/log/6_full_alignment_call_variant.log'

        try:
            return_code = subprocess.check_call(ct_command, shell=True, stdout=sys.stdout)
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                "ERROR in Full-alignment model calling, THE FOLLOWING COMMAND FAILED: {}\n".format(ct_command))
            exit(1)

        print("[INFO] Removing temporary tensor files...")
        for f in file_list:
            if os.path.exists(f + ".npy"):
                os.remove(f + ".npy")
            if os.path.exists(f + ".info"):
                os.remove(f + ".info")

def main():
    parser = ArgumentParser(description="The wrapper of GPU Call variants")

    parser.add_argument('--platform', type=str, default="ont",
                        help="Sequencing platform of the input. Options: 'ont,hifi,ilmn', default: %(default)s")

    parser.add_argument('--tensor_fn', type=str, default="PIPE",
                        help="Tensor input filename, or stdin if not set")

    parser.add_argument('--full_aln_files', type=str, default=None,
                        help="Tensor input filename, or stdin if not set")

    parser.add_argument('--output_dir', type=str, default=None,
                        help="Tensor input filename, or stdin if not set")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a trained model for variant calling, required")

    parser.add_argument('--call_fn', type=str, default="clair3",
                        help="VCF output filename, or stdout if not set")

    parser.add_argument('--gvcf', type=str2bool, default=False,
                        help="Enable GVCF output, default: disabled")

    parser.add_argument('--ref_fn', type=str, default=None,
                        help="Reference fasta file input, required if --gvcf is enabled")

    parser.add_argument('--ctgName', type=str, default=None,
                        help="The name of the sequence to be processed")

    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-based starting position of the sequence to be processed, optional, will process the whole --ctgName if not set")

    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The 1-based inclusive ending position of the sequence to be processed, optional, will process the whole --ctgName if not set")

    parser.add_argument('--sampleName', type=str, default="SAMPLE",
                        help="Define the sample name to be shown in the VCF file, optional")

    parser.add_argument('--qual', type=int, default=2,
                        help="If set, variants with >=$qual will be marked 'PASS', or 'LowQual' otherwise, optional")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', samtools version >= 1.10 is required, default: %(default)s")

    # options for advanced users
    parser.add_argument('--temp_file_dir', type=str, default='./',
                        help="EXPERIMENTAL: The cache directory for storing temporary non-variant information if --gvcf is enabled, default: %(default)s")

    parser.add_argument('--haploid_precise', action='store_true',
                        help="EXPERIMENTAL: Enable haploid calling mode. Only 1/1 is considered as a variant")

    parser.add_argument('--haploid_sensitive', action='store_true',
                        help="EXPERIMENTAL: Enable haploid calling mode. 0/1 and 1/1 are considered as a variant")

    parser.add_argument('--enable_long_indel', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable long Indel variants(>50 bp) calling")

    parser.add_argument('--keep_iupac_bases', type=str2bool, default=False,
                        help="EXPERIMENTAL: Keep IUPAC (non ACGTN) reference and alternate bases, default: convert all IUPAC bases to N")

    parser.add_argument('--enable_dwell_time', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable dwell time channel for full-alignment tensors")

    # options for debug purpose
    parser.add_argument('--use_gpu', type=str2bool, default=True,
                        help="Use GPU for calling")

    parser.add_argument('--threads', type=int, default=1,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument('--predict_fn', type=str, default=None,
                        help="DEBUG: Output network output probabilities for further analysis")

    parser.add_argument('--input_probabilities', action='store_true',
                        help="DEBUG: Use network probability outputs as input and generate variants from them")

    parser.add_argument('--output_probabilities', action='store_true',
                        help="DEBUG: Output the network probabilities of gt21, genotype, indel_length_1 and indel_length_2")

    parser.add_argument('--enable_variant_calling_at_sequence_head_and_tail', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable variant calling in sequence head and tail start or end regions that flanking 16bp windows having no read support. Default: disable.")

    # options for internal process control
    ## In pileup mode or not (full alignment mode), default: False
    parser.add_argument('--pileup', action='store_true',
                        help=SUPPRESS)

    ## Include indel length in training and calling, false for pileup and true for raw alignment
    parser.add_argument('--add_indel_length', action='store_true',
                        help=SUPPRESS)

    ## The number of chucks to be divided into for parallel processing
    parser.add_argument('--chunk_num', type=int, default=None,
                        help=SUPPRESS)

    ## The chuck ID to work on
    parser.add_argument('--chunk_id', type=int, default=None,
                        help=SUPPRESS)

    ## Enable debug mode, default: False, optional
    parser.add_argument('--debug', action='store_true',
                        help=SUPPRESS)

    ## Generating outputs for ensemble model calling
    parser.add_argument('--output_for_ensemble', action='store_true',
                        help=SUPPRESS)

    ## Use bin file from HDF5 to speed up calling.
    parser.add_argument('--is_from_tables', action='store_true',
                        help=SUPPRESS)

    ## Output reference calls
    parser.add_argument('--showRef', type=str2bool, default=True,
                        help=SUPPRESS)

    # Pileup create tensor options for pileup calling
    parser.add_argument('--bam_fn', type=str, default="input.bam", required=True,
                        help="Sorted BAM file input, required")

    parser.add_argument('--bed_fn', type=str, nargs='?', action="store", default=None,
                        help="Call variant only in the provided regions. Will take an intersection if --ctgName and/or (--ctgStart, --ctgEnd) are set")

    parser.add_argument('--snp_min_af', type=float, default=0.08,
                        help="Minimum snp allele frequency for a site to be considered as a candidate site, default: %(default)f")

    parser.add_argument('--indel_min_af', type=float, default=0.15,
                        help="Minimum indel allele frequency for a site to be considered as a candidate site, default: %(default)f")

    parser.add_argument('--extend_bed', nargs='?', action="store", type=str, default=None,
                        help="DEBUG: Extend the regions in the --bed_fn by a few bp for tensor creation, default extend 16bp")

    parser.add_argument('--vcf_fn', type=str, default=None,
                        help="Candidate sites VCF file input, if provided, variants will only be called at the sites in the VCF file,  default: %(default)s")

    parser.add_argument('--minCoverage', type=int, default=2,
                        help="EXPERIMENTAL: Minimum coverage required to call a variant, default: %(default)f")

    parser.add_argument('--minMQ', type=int, default=5,
                        help="EXPERIMENTAL: If set, reads with mapping quality with <$minMQ are filtered, default: %(default)d")

    parser.add_argument('--minBQ', type=int, default=0,
                        help="EXPERIMENTAL: If set, bases with base quality with <$minBQ are filtered, default: %(default)d")

    parser.add_argument('--max_depth', type=int, default=144,
                        help="EXPERIMENTAL: Maximum full alignment depth to be processed. default: %(default)s")

    parser.add_argument('--fast_mode', type=str2bool, default=False,
                        help="EXPERIMENTAL: Skip variant candidates with AF <= 0.15, default: %(default)s")

    parser.add_argument('--call_snp_only', type=str2bool, default=False,
                        help="EXPERIMENTAL: Call candidates pass snp minimum AF only, ignore Indel candidates")

    parser.add_argument('--base_err', default=0.001, type=float,
                        help='DEBUG: Estimated base error rate in gvcf option, default: %(default)f')

    parser.add_argument('--gq_bin_size', default=5, type=int,
                        help='DEBUG: Default gq bin size for merge non-variant block in gvcf option, default: %(default)d')

    parser.add_argument('--bp_resolution', action='store_true',
                        help="DEBUG: Enable bp resolution for GVCF, default: disabled")

    # Full-alignment create tensor options for full-alignment calling
    parser.add_argument('--phased_vcf_fn', type=str, default=None,
                        help="Use heterozygous SNP variants in phased vcf file for haplotaging")

    parser.add_argument('--no_phasing_for_fa', type=str2bool, default=False,
                        help="EXPERIMENTAL: Call variants without whatshap or longphase phasing in full alignment calling")

    ## Provide the regions to be included in full-alignment based calling
    parser.add_argument('--full_aln_regions', type=str, default=None,
                        help=SUPPRESS)

    ## If defined, added command line into VCF header
    parser.add_argument('--cmd_fn', type=str_none, default=None,
                        help=SUPPRESS)

    parser.add_argument(
        "--parallel", type=str, default="parallel",
                        help="Absolute path of parallel, parallel >= 20191122 is required."
    )

    parser.add_argument(
        "--pypy", type=str, default="pypy3",
                        help="Absolute path of pypy3, pypy3 >= 3.6 is required."
    )

    parser.add_argument(
        "--python", type=str, default="python3",
                        help="Absolute path of pypy3, pypy3 >= 3.6 is required."
    )

    parser.add_argument(
        '--cpu_threads', type=int, default=4,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument(
        '--gpu_threads', type=int, default=4,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument(
        '--internal_threads', type=int, default=1,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument(
        '--device', type=str, default=None,
                        help="Specify the GPUs for calling default: 'cuda:auto'")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)


if __name__ == "__main__":
    main()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>clair3/results/CallVariantsFromCffi.original.py — 원문 전문 (26,510 bytes)</summary>


<!-- RESTORE_FILE {"path": "clair3/results/CallVariantsFromCffi.original.py", "sha256": "024fe816b279a416d84ab487b6948ad2899088fcca607bb431264e0e89a96ace", "encoding": "utf-8", "added_final_newline": false} -->
```python
import sys
import os
import copy
import csv
import torch
import logging
from time import time
import numpy as np

from argparse import ArgumentParser, SUPPRESS
from concurrent.futures import ProcessPoolExecutor, as_completed

from shared.utils import str2bool, log_error, log_warning, str_none, get_header
from shared.interval_tree import bed_tree_from
from clair3.CallVariants import OutputConfig, output_utilties_from, batch_output

logging.basicConfig(format='%(message)s', level=logging.INFO)

def _load_torch_checkpoint(model, checkpoint_path, device):
    #add .pt extension if not present
    if not checkpoint_path.endswith('.pt'):
        checkpoint_path = checkpoint_path + '.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def _select_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _limit_gpu_memory(memory_mb, device):
    if device.type != "cuda":
        return
    total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
    fraction = min(1.0, memory_mb / total_mb)
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    except Exception:
        pass


def _torch_predict(model, device, X):
    with torch.inference_mode():
        X_tensor = torch.from_numpy(X).to(device)
        Y = model(X_tensor)
    return Y.detach().cpu().numpy()


def Run(args):
    threads = args.threads
    os.environ["OMP_NUM_THREADS"] = f"{threads}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{threads}"
    os.environ["MKL_NUM_THREADS"] = f"{threads}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{threads}"

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(2 if threads > 1 else 1)

    global test_pos
    test_pos = None
    global param
    if args.pileup:
        import shared.param_p as param
    else:
        import shared.param_f as param

    if args.enable_long_indel:
        maximum_variant_length_that_need_infer = param.maximum_variant_length_that_need_infer_include_long_indel
    else:
        maximum_variant_length_that_need_infer = param.maximum_variant_length_that_need_infer

    output_config = OutputConfig(
        is_show_reference=args.showRef,
        is_debug=args.debug,
        is_haploid_precise_mode_enabled=args.haploid_precise,
        is_haploid_sensitive_mode_enabled=args.haploid_sensitive,
        is_output_for_ensemble=args.output_for_ensemble,
        quality_score_for_pass=args.qual,
        tensor_fn=args.tensor_fn,
        input_probabilities=args.input_probabilities,
        add_indel_length=args.add_indel_length,
        gvcf=args.gvcf,
        pileup=args.pileup,
        enable_long_indel=args.enable_long_indel,
        maximum_variant_length_that_need_infer=maximum_variant_length_that_need_infer,
        keep_iupac_bases=args.keep_iupac_bases
    )
    output_utilities = output_utilties_from(
        sample_name=args.sampleName,
        is_debug=args.debug,
        is_output_for_ensemble=args.output_for_ensemble,
        reference_file_path=args.ref_fn,
        output_file_path=args.call_fn,
        output_probabilities=args.output_probabilities,
        cmd_fn=args.cmd_fn
    )

    call_variants_from_cffi(args=args, output_config=output_config, output_utilities=output_utilities)

def tensor_generator_for_chunk(gen_cls, args, batch_size=50):
    if args.output_tensor_can_fn_list is not None:
        f_list = open(args.output_tensor_can_fn_list, 'r').read()
        f_list = f_list.strip().split('\n')
        import numpy as np
        for f in f_list:
            if f == '': continue
            parent_dir = os.path.dirname(args.output_tensor_can_fn_list)
            tensor = np.load(os.path.join(parent_dir, f +'.npy'))
            pos_alt = os.path.join(parent_dir, f + '.info')
            pos_alt_info = open(pos_alt, 'r').read().strip().split('\n')
            all_alt_info = []
            all_position = []
            for pos_alt in pos_alt_info:
                pos_alt = pos_alt.split('\t')
                all_position.append(pos_alt[0])
                all_alt_info.append(pos_alt[1])
            total_data = len(tensor)
            assert total_data == len(all_alt_info)
            assert total_data == len(all_position)
            batch_size = batch_size
            total_chunk = total_data // batch_size if total_data % batch_size == 0 else total_data // batch_size + 1
            for chunk_id in range(total_chunk):
                chunk_start = chunk_id * batch_size
                chunk_end = (chunk_id + 1) * batch_size if chunk_id < total_chunk - 1 else total_data
                yield (tensor[chunk_start:chunk_end],
                       all_position[chunk_start:chunk_end],
                       all_alt_info[chunk_start:chunk_end])
    else:
        tensor, all_position, all_alt_info = gen_cls(args)
        if tensor is None:
            return
        total_data = len(tensor)
        assert total_data == len(all_alt_info)
        assert total_data == len(all_position)
        batch_size = batch_size
        total_chunk = total_data // batch_size if total_data % batch_size == 0 else total_data // batch_size + 1
        for chunk_id in range(total_chunk):
            chunk_start = chunk_id * batch_size
            chunk_end = (chunk_id + 1) * batch_size if chunk_id < total_chunk - 1 else total_data
            yield (tensor[chunk_start:chunk_end],
                   all_position[chunk_start:chunk_end],
                   all_alt_info[chunk_start:chunk_end])

def tensor_generator_for_unchunked(gen_cls, args, batch_size=200):
    if args.bed_fn:
        tree = bed_tree_from(args.bed_fn)
        for contig in tree.values():
            contig.merge_overlaps()
        chunks = [(ctg_name, I.begin, I.end) for ctg_name, ctg_tree in tree.items() for I in ctg_tree.all_intervals]
    else:
        fai = f"{args.ref_fn}.fai"
        chunks = [(ctg, 0, ctg_len) for ctg, ctg_len, *_ in csv.reader(open(fai), delimiter='\t')]
    for ctg, start, end in chunks:
        temp_args = copy.copy(args)
        temp_args.ctgName = ctg
        temp_args.ctgStart = int(start)
        temp_args.ctgEnd = int(end)
        for batch in tensor_generator_for_chunk(gen_cls, temp_args, batch_size=batch_size):
            yield batch




def batch_output_worker(position, alt_info_list, shm_name, shape, dtype,
                        output_config, output_utilities):
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        Y = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()

        result = batch_output(position, alt_info_list, Y, output_config, output_utilities)

        return result
    finally:
        if 'shm' in locals():
            shm.close()
            shm.unlink()

def call_variants_from_cffi(args, output_config, output_utilities):
    if args.enable_dwell_time and args.pileup:
        sys.exit(log_error("[ERROR] --enable_dwell_time is only supported in full-alignment mode"))

    if args.pileup:
        from preprocess.CreateTensorPileupFromCffi import CreateTensorPileup as CT
    else:
        if args.enable_dwell_time:
            from preprocess.CreateTensorFullAlignmentFromCffiWithDwell import CreateTensorFullAlignment as CT
        else:
            from preprocess.CreateTensorFullAlignmentFromCffi import CreateTensorFullAlignment as CT

    if args.output_tensor_can_fn_list is None and args.use_gpu:
        CT(args)
        return

    use_triton_gpu = args.use_triton_gpu
    triton_client = None
    device = torch.device("cpu")
    if use_triton_gpu:
        import tritonclient.grpc as tritongrpcclient
        server_url = 'localhost:8001'
        try:
            triton_client = tritongrpcclient.InferenceServerClient(
                url=server_url,
                verbose=False
            )
        except Exception as e:
            print("channel creation failed: " + str(e), file=sys.stderr)
            sys.exit()
    elif args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.gpu_id else ""
        device = _select_device(True)
        _limit_gpu_memory(args.max_gpu_memory, device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = _select_device(False)

    global param
    if args.pileup:
        import shared.param_p as param
        if use_triton_gpu:
            model_name = 'pileup'
            input_dtype = 'INT32'
        else:
            from clair3.model import Clair3_P
            input_channels = (param.ont_input_shape if args.platform == 'ont' else param.input_shape)[-1]
            m = Clair3_P(add_indel_length=args.add_indel_length, predict=True, input_channels=input_channels)
    else:
        import shared.param_f as param
        if use_triton_gpu:
            model_name = 'alignment'
            input_dtype = 'INT8'
        else:
            from clair3.model import Clair3_F
            input_channels = (param.ont_input_shape if args.platform == 'ont' else param.input_shape)[-1]
            if args.enable_dwell_time:
                input_channels += 1
            m = Clair3_F(add_indel_length=args.add_indel_length, predict=True, input_channels=input_channels)

    if not use_triton_gpu:
        m.to(device)
        m.eval()
        _load_torch_checkpoint(m, args.chkpnt_fn, device)
    args.output_file = open(args.call_fn, 'w') if args.call_fn != 'PIPE' else sys.stdout

    header_str = get_header(reference_file_path=args.ref_fn, cmd_fn=args.cmd_fn, sample_name=args.sampleName)
    header_str += '\n'
    args.output_file.write(header_str)

    chunk_id = args.chunk_id - 1 if args.chunk_id else None  # 1-base to 0-base
    chunk_num = args.chunk_num
    full_alignment_mode = not args.pileup

    logging.info("Calling variants ...")
    variant_call_start_time = time()

    batch_output_method = batch_output
    total = 0

    if not args.use_gpu:
        batch_size = param.predictBatchSize * args.threads
    else:
        # increase bs in gpu for speed up
        batch_size = param.predictBatchSize * 5
    if args.ctgName == 'None' and args.pileup:
        batch_gen = tensor_generator_for_unchunked(CT, args, batch_size=batch_size)
    else:
        batch_gen = tensor_generator_for_chunk(CT, args, batch_size=batch_size)

    if not args.use_gpu:
        for (X, position, alt_info_list) in batch_gen:
            total += len(X)
            if args.pileup:
                for alt_idx, alt_info in enumerate(alt_info_list):
                    depth = int(alt_info.split('-', maxsplit=1)[0])
                    max_depth = param.max_depth_dict[args.platform]
                    # for extreme high coverage data, make sure we could have a truncated coverage
                    if depth > 0 and depth > max_depth * 1.5:
                        scale_factor = depth / max_depth
                        X[alt_idx] = X[alt_idx] / scale_factor

            if use_triton_gpu:
                inputs = [];
                outputs = []
                inputs.append(tritongrpcclient.InferInput('input_1', X.shape, input_dtype))
                outputs.append(tritongrpcclient.InferRequestedOutput('output_1'))
                inputs[0].set_data_from_numpy(X)
                results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
                Y = results.as_numpy('output_1')
            else:
                Y = _torch_predict(m, device, X)

            output = batch_output_method(position, alt_info_list, Y, output_config, output_utilities, args)
    else:
        from multiprocessing import shared_memory
        max_pending = args.cpu_threads * 2  # Limit concurrent tasks to bound memory
        with ProcessPoolExecutor(max_workers=args.cpu_threads) as executor:
            pending_futures = {}  # future -> shm_name for cleanup tracking
            batch_iter = iter(batch_gen)
            finished = False

            while pending_futures or not finished:
                # Submit new tasks up to max_pending limit
                while len(pending_futures) < max_pending and not finished:
                    try:
                        X, position, alt_info_list = next(batch_iter)
                    except StopIteration:
                        finished = True
                        break

                    total += len(X)
                    Y = _torch_predict(m, device, X)
                    shm = shared_memory.SharedMemory(create=True, size=Y.nbytes)
                    shared_Y = np.ndarray(Y.shape, dtype=Y.dtype, buffer=shm.buf)
                    np.copyto(shared_Y, Y)

                    future = executor.submit(
                        batch_output_worker,
                        position, alt_info_list,
                        shm.name, Y.shape, Y.dtype,
                        output_config, output_utilities
                    )
                    pending_futures[future] = shm.name
                    shm.close()

                if not pending_futures:
                    break

                # Wait for at least one task to complete
                done_futures = []
                for future in as_completed(pending_futures.keys()):
                    done_futures.append(future)
                    break  # Process one at a time to submit new tasks promptly

                for future in done_futures:
                    shm_name = pending_futures.pop(future)
                    try:
                        result = future.result()
                        args.output_file.write(result)
                    except Exception as e:
                        print(f"[Error] Task failed: {e}", flush=True, file=sys.stderr)
                        # Cleanup shared memory if worker failed before unlink
                        try:
                            orphan_shm = shared_memory.SharedMemory(name=shm_name)
                            orphan_shm.close()
                            orphan_shm.unlink()
                        except FileNotFoundError:
                            pass  # Already cleaned up by worker

    if chunk_id is not None:
        logging.info("Total processed positions in {} (chunk {}/{}) : {}".format(args.ctgName, chunk_id+1, chunk_num, total))
    elif full_alignment_mode:
        try:
            chunk_infos = args.call_fn.split('.')[-2]
            c_id, c_num = chunk_infos.split('_')
            c_id = int(c_id) + 1 # 0-index to 1-index
            logging.info("Total processed positions in {} (chunk {}/{}) : {}".format(args.ctgName, c_id, c_num, total))
        except:
            logging.info("Total processed positions {}: {}".format(args.ctgName if args.ctgName else "", total))
    else:
        logging.info("Total processed positions {}: {}".format(args.ctgName if args.ctgName else "", total))

    if full_alignment_mode and total == 0:
        logging.info(log_warning("[WARNING] No full-alignment output for file {}/{}".format(args.ctgName, args.call_fn)))

    logging.info("Total time elapsed: %.2f s" % (time() - variant_call_start_time))

    args.output_file.close()
    # remove file if on variant in output
    if os.path.exists(args.call_fn):
        for row in open(args.call_fn, 'r'):
            if row[0] != '#':
                return
        logging.info("[INFO] No vcf output for file {}, remove empty file".format(args.call_fn))
        os.remove(args.call_fn)


def main():
    parser = ArgumentParser(description="Call variants using a trained model and tensors of candidate variants")

    parser.add_argument('--platform', type=str, default="ont",
                        help="Sequencing platform of the input. Options: 'ont,hifi,ilmn', default: %(default)s")

    parser.add_argument('--tensor_fn', type=str, default="PIPE",
                        help="Tensor input filename, or stdin if not set")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a trained model for variant calling, required")

    parser.add_argument('--call_fn', type=str, default="clair3",
                        help="VCF output filename, or stdout if not set")

    parser.add_argument('--gvcf', type=str2bool, default=False,
                        help="Enable GVCF output, default: disabled")

    parser.add_argument('--ref_fn', type=str, default=None,
                        help="Reference fasta file input, required if --gvcf is enabled")

    parser.add_argument('--ctgName', type=str, default=None,
                        help="The name of the sequence to be processed")

    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-based starting position of the sequence to be processed, optional, will process the whole --ctgName if not set")

    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The 1-based inclusive ending position of the sequence to be processed, optional, will process the whole --ctgName if not set")

    parser.add_argument('--sampleName', type=str, default="SAMPLE",
                        help="Define the sample name to be shown in the VCF file, optional")

    parser.add_argument('--qual', type=int, default=2,
                        help="If set, variants with >=$qual will be marked 'PASS', or 'LowQual' otherwise, optional")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', samtools version >= 1.10 is required, default: %(default)s")

    # options for advanced users
    parser.add_argument('--temp_file_dir', type=str, default='./',
                        help="EXPERIMENTAL: The cache directory for storing temporary non-variant information if --gvcf is enabled, default: %(default)s")

    parser.add_argument('--haploid_precise', action='store_true',
                        help="EXPERIMENTAL: Enable haploid calling mode. Only 1/1 is considered as a variant")

    parser.add_argument('--haploid_sensitive', action='store_true',
                        help="EXPERIMENTAL: Enable haploid calling mode. 0/1 and 1/1 are considered as a variant")

    parser.add_argument('--enable_long_indel', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable long Indel variants(>50 bp) calling")

    parser.add_argument('--keep_iupac_bases', type=str2bool, default=False,
                        help="EXPERIMENTAL: Keep IUPAC (non ACGTN) reference and alternate bases, default: convert all IUPAC bases to N")

    # options for debug purpose
    parser.add_argument('--use_gpu', type=str2bool, default=False,
                        help="DEBUG: Use GPU for calling. Speed up is mostly insignficiant. Only use this for building your own pipeline")

    parser.add_argument('--threads', type=int, default=1,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument('--predict_fn', type=str, default=None,
                        help="DEBUG: Output network output probabilities for further analysis")

    parser.add_argument('--input_probabilities', action='store_true',
                        help="DEBUG: Use network probability outputs as input and generate variants from them")

    parser.add_argument('--output_probabilities', action='store_true',
                        help="DEBUG: Output the network probabilities of gt21, genotype, indel_length_1 and indel_length_2")

    parser.add_argument('--enable_variant_calling_at_sequence_head_and_tail', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable variant calling in sequence head and tail start or end regions that flanking 16bp windows having no read support. Default: disable.")

    # options for internal process control
    ## In pileup mode or not (full alignment mode), default: False
    parser.add_argument('--pileup', action='store_true',
                        help=SUPPRESS)

    ## Include indel length in training and calling, false for pileup and true for raw alignment
    parser.add_argument('--add_indel_length', action='store_true',
                        help=SUPPRESS)

    ## The number of chucks to be divided into for parallel processing
    parser.add_argument('--chunk_num', type=int, default=None,
                        help=SUPPRESS)

    ## The chuck ID to work on
    parser.add_argument('--chunk_id', type=int, default=None,
                        help=SUPPRESS)

    ## Enable debug mode, default: False, optional
    parser.add_argument('--debug', action='store_true',
                        help=SUPPRESS)

    ## Generating outputs for ensemble model calling
    parser.add_argument('--output_for_ensemble', action='store_true',
                        help=SUPPRESS)

    ## Use bin file from HDF5 to speed up calling.
    parser.add_argument('--is_from_tables', action='store_true',
                        help=SUPPRESS)

    ## Output reference calls
    parser.add_argument('--showRef', type=str2bool, default=True,
                        help=SUPPRESS)

    # Pileup create tensor options for pileup calling
    parser.add_argument('--bam_fn', type=str, default="input.bam", required=True,
                        help="Sorted BAM file input, required")

    parser.add_argument('--bed_fn', type=str, nargs='?', action="store", default=None,
                        help="Call variant only in the provided regions. Will take an intersection if --ctgName and/or (--ctgStart, --ctgEnd) are set")

    parser.add_argument('--snp_min_af', type=float, default=0.08,
                        help="Minimum snp allele frequency for a site to be considered as a candidate site, default: %(default)f")

    parser.add_argument('--indel_min_af', type=float, default=0.15,
                        help="Minimum indel allele frequency for a site to be considered as a candidate site, default: %(default)f")

    parser.add_argument('--extend_bed', nargs='?', action="store", type=str, default=None,
                        help="DEBUG: Extend the regions in the --bed_fn by a few bp for tensor creation, default extend 16bp")

    parser.add_argument('--vcf_fn', type=str, default=None,
                        help="Candidate sites VCF file input, if provided, variants will only be called at the sites in the VCF file,  default: %(default)s")

    parser.add_argument('--minCoverage', type=int, default=2,
                        help="EXPERIMENTAL: Minimum coverage required to call a variant, default: %(default)f")

    parser.add_argument('--minMQ', type=int, default=5,
                        help="EXPERIMENTAL: If set, reads with mapping quality with <$minMQ are filtered, default: %(default)d")

    parser.add_argument('--minBQ', type=int, default=0,
                        help="EXPERIMENTAL: If set, bases with base quality with <$minBQ are filtered, default: %(default)d")

    parser.add_argument('--max_depth', type=int, default=144,
                        help="EXPERIMENTAL: Maximum full alignment depth to be processed. default: %(default)s")

    parser.add_argument('--fast_mode', type=str2bool, default=False,
                        help="EXPERIMENTAL: Skip variant candidates with AF <= 0.15, default: %(default)s")

    parser.add_argument('--call_snp_only', type=str2bool, default=False,
                        help="EXPERIMENTAL: Call candidates pass snp minimum AF only, ignore Indel candidates")

    parser.add_argument('--base_err', default=0.001, type=float,
                        help='DEBUG: Estimated base error rate in gvcf option, default: %(default)f')

    parser.add_argument('--gq_bin_size', default=5, type=int,
                        help='DEBUG: Default gq bin size for merge non-variant block in gvcf option, default: %(default)d')

    parser.add_argument('--bp_resolution', action='store_true',
                        help="DEBUG: Enable bp resolution for GVCF, default: disabled")

    # Full-alignment create tensor options for full-alignment calling
    parser.add_argument('--phased_vcf_fn', type=str, default=None,
                        help="Use heterozygous SNP variants in phased vcf file for haplotaging")

    parser.add_argument('--no_phasing_for_fa', type=str2bool, default=False,
                        help="EXPERIMENTAL: Call variants without whatshap or longphase phasing in full alignment calling")

    parser.add_argument('--enable_dwell_time', type=str2bool, default=False,
                        help="EXPERIMENTAL: Enable dwell time channel when generating full-alignment tensors")

    ## Provide the regions to be included in full-alignment based calling
    parser.add_argument('--full_aln_regions', type=str, default=None,
                        help=SUPPRESS)

    ## If defined, added command line into VCF header
    parser.add_argument('--cmd_fn', type=str_none, default=None,
                        help=SUPPRESS)

    parser.add_argument('--tensor_can_fn', type=str, default="PIPE",
                        help="Tensor output, stdout by default, default: %(default)s")

    parser.add_argument('--gpu_id', type=str, default=None,
                        help="Tensor output, stdout by default, default: %(default)s")

    parser.add_argument('--max_gpu_memory', type=int, default=5000,
                        help="Tensor output, stdout by default, default: %(default)s")

    parser.add_argument('--use_triton_gpu', type=str2bool, default=False,
                        help="DEBUG: Use GPU for calling. Speed up is mostly insignficiant. Only use this for building your own pipeline")

    parser.add_argument('--output_tensor_can_fn_list', type=str, default=None,
                        help="Tensor output, stdout by default, default: %(default)s")

    parser.add_argument('--cpu_threads', type=int, default=4,
                        help="How many threads for python to use for parallelization default: %(default)f")

    parser.add_argument('--output_file', type=str, default=None,
                        help="How many threads for python to use for parallelization default: %(default)f")


    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)


if __name__ == "__main__":
    main()
```
<!-- END_RESTORE_FILE -->


</details>

<details>
<summary>scripts/gpu_support.py — 원문 전문 (2,858 bytes)</summary>


<!-- RESTORE_FILE {"path": "scripts/gpu_support.py", "sha256": "349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2", "encoding": "utf-8", "added_final_newline": false} -->
```python
import time,json,csv,threading,subprocess,os
from pathlib import Path
import numpy as np,psutil,torch
G=2**30
class Meter:
 def __init__(self,root):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.csv_path=self.root/'resources.csv';
  if self.csv_path.exists():self.csv_path=self.root/('resources_'+str(time.time_ns())+'.csv')
  self.phase='initial';self.rows=[];self.stop=threading.Event();self.swap0=psutil.swap_memory().used;self.th=threading.Thread(target=self.monitor,daemon=True);self.th.start()
 def memory(self):
  v=psutil.virtual_memory();return dict(allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,peak_allocated_gib=torch.cuda.max_memory_allocated()/G,peak_reserved_gib=torch.cuda.max_memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,system_available_gib=v.available/G,system_total_gib=v.total/G)
 def monitor(self):
  with self.csv_path.open('w',newline='') as f:
   keys=['query_start','query_end','phase','gpu_util','allocated_gib','reserved_gib','rss_gib','available_gib','swap_gib'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
   while not self.stop.is_set():
    t=time.time();phase=self.phase
    try:u=float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).strip().splitlines()[0])
    except Exception:u=None
    v=psutil.virtual_memory();r=dict(query_start=t,query_end=time.time(),phase=phase,gpu_util=u,allocated_gib=torch.cuda.memory_allocated()/G,reserved_gib=torch.cuda.memory_reserved()/G,rss_gib=psutil.Process().memory_info().rss/G,available_gib=v.available/G,swap_gib=psutil.swap_memory().used/G);self.rows.append(r);w.writerow(r);f.flush()
    if v.available<8*G or psutil.swap_memory().used>self.swap0+2*G:(self.root/'memory_guard.json').write_text(json.dumps(r));os._exit(86)
    self.stop.wait(.5)
 def stats(self,t,end):
  rows=[r for r in self.rows if r['query_start']>=t and r['query_end']<=end];u=[r['gpu_util'] for r in rows if r['gpu_util'] is not None];return dict(samples=len(u),gpu_util_avg=float(np.mean(u)) if u else None,gpu_util_max=max(u) if u else None,rss_sampled_peak_gib=max((r['rss_gib'] for r in rows),default=None),available_sampled_min_gib=min((r['available_gib'] for r in rows),default=None))
 def begin(self,name):
  self.phase=name;torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();return time.time(),self.memory()
 def end(self,t,base,**kw):
  torch.cuda.synchronize();end=time.time();m=self.memory();return dict(**kw,seconds=end-t,baseline=base,memory=m,peak_delta_gib=m['peak_allocated_gib']-base['allocated_gib'],**self.stats(t,end))
 def close(self):self.stop.set();self.th.join(timeout=5)
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str))
```
<!-- END_RESTORE_FILE -->


</details>
