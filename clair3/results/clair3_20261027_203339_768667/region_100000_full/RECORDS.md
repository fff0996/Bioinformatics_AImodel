# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-c04f0ce190c3

원래 경로: `clair3/results/clair3_20261027_203339_768667/region_100000_full/command.json` · SHA256: `597deb7e845b2d4e2aa0dab872effcddc07daafa692a549a7a679bd148eb315d`

<details>
<summary>command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_100000_full/command.json", "sha256": "597deb7e845b2d4e2aa0dab872effcddc07daafa692a549a7a679bd148eb315d", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_chr20_demo.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/clair3_20261027_203339_768667/region_100000_full/calls",
  "--bed_fn=/work/data/region_100000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-5c5cc8f5488a

원래 경로: `clair3/results/clair3_20261027_203339_768667/region_100000_full/exit_code.json` · SHA256: `8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd`

<details>
<summary>exit_code.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_100000_full/exit_code.json", "sha256": "8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "exit_code": 0
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-d307b652c3e6

원래 경로: `clair3/results/clair3_20261027_203339_768667/region_100000_full/metrics.json` · SHA256: `d93f201151657db3e64c40b8c4baabbbe92b999b51b7b5c0980a216578cec48e`

<details>
<summary>metrics.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_100000_full/metrics.json", "sha256": "d93f201151657db3e64c40b8c4baabbbe92b999b51b7b5c0980a216578cec48e", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "SNP": {
    "TP": 208,
    "FP": 0,
    "FN": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "INDEL": {
    "TP": 28,
    "FP": 0,
    "FN": 5,
    "precision": 1.0,
    "recall": 0.8484848484848485,
    "f1": 0.9180327868852459
  }
}
```
<!-- END_RESTORE_FILE -->


</details>
