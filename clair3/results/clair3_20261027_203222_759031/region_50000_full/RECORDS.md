# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-8560a2f733b0

원래 경로: `clair3/results/clair3_20261027_203222_759031/region_50000_full/command.json` · SHA256: `c1aa7986a959a7086934382c993b12f49e7b5c812ac247a3943c093c2f43345f`

<details>
<summary>command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203222_759031/region_50000_full/command.json", "sha256": "c1aa7986a959a7086934382c993b12f49e7b5c812ac247a3943c093c2f43345f", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_chr20_demo.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/clair3_20261027_203222_759031/region_50000_full/calls",
  "--bed_fn=/work/data/region_50000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-da84fdccff6e

원래 경로: `clair3/results/clair3_20261027_203222_759031/region_50000_full/exit_code.json` · SHA256: `953b9538e7e86b5237db3d2db46f7166ba7d2bfa599b53461f6440175b7783a1`

<details>
<summary>exit_code.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203222_759031/region_50000_full/exit_code.json", "sha256": "953b9538e7e86b5237db3d2db46f7166ba7d2bfa599b53461f6440175b7783a1", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "exit_code": 1
}
```
<!-- END_RESTORE_FILE -->


</details>
