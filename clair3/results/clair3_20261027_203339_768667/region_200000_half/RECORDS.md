# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-cf5df12c8701

원래 경로: `clair3/results/clair3_20261027_203339_768667/region_200000_half/command.json` · SHA256: `c509b115d1d876f16b5e112f9046ec4bf69dbc977b7c1c02db2ad269e108b227`

<details>
<summary>command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_200000_half/command.json", "sha256": "c509b115d1d876f16b5e112f9046ec4bf69dbc977b7c1c02db2ad269e108b227", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_half.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/clair3_20261027_203339_768667/region_200000_half/calls",
  "--bed_fn=/work/data/region_200000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-2b4ff1d7115a

원래 경로: `clair3/results/clair3_20261027_203339_768667/region_200000_half/exit_code.json` · SHA256: `8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd`

<details>
<summary>exit_code.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_200000_half/exit_code.json", "sha256": "8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "exit_code": 0
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-d6479611f2fa

원래 경로: `clair3/results/clair3_20261027_203339_768667/region_200000_half/metrics.json` · SHA256: `a0d5fdb1c8f66d63acfefd1b6b541d9c1839769ffa2c613428b7cfcfc776a79c`

<details>
<summary>metrics.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/region_200000_half/metrics.json", "sha256": "a0d5fdb1c8f66d63acfefd1b6b541d9c1839769ffa2c613428b7cfcfc776a79c", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "SNP": {
    "TP": 404,
    "FP": 1,
    "FN": 0,
    "precision": 0.9975308641975309,
    "recall": 1.0,
    "f1": 0.9987639060568603
  },
  "INDEL": {
    "TP": 41,
    "FP": 1,
    "FN": 20,
    "precision": 0.9761904761904762,
    "recall": 0.6721311475409836,
    "f1": 0.7961165048543689
  }
}
```
<!-- END_RESTORE_FILE -->


</details>
