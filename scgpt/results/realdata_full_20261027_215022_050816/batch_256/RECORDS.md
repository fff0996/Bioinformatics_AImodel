# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-bf4f10b41951

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_256/memory_guard.json` · SHA256: `174b8527a7c247d1d487a5a87326e063af451b5a1028c20877288324150c5a92`

<details>
<summary>memory_guard.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/memory_guard.json", "sha256": "174b8527a7c247d1d487a5a87326e063af451b5a1028c20877288324150c5a92", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "time_unix": 1793106687.1003087,
  "available_gib": 7.816379547119141,
  "swap_gib": 0.00069427490234375,
  "cuda_allocated_gib": 59.276695728302,
  "cuda_reserved_gib": 103.662109375
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-75a30842618b

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_256/train_command.json` · SHA256: `3184297eefbd16218b5d841b6d683bff580cb1d16d6c6df3c7b1f8b93a47d71b`

<details>
<summary>train_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train_command.json", "sha256": "3184297eefbd16218b5d841b6d683bff580cb1d16d6c6df3c7b1f8b93a47d71b", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "exec",
  "--user",
  "1000:1000",
  "-e",
  "SCGPT_GUARD_DIR=/workspace/scgpt/results/realdata_full_20261027_215022_050816/batch_256",
  "-e",
  "MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba",
  "scgpt-spark",
  "python",
  "-u",
  "/workspace/scgpt/scripts/full_realdata.py",
  "--mode",
  "train",
  "--n-cells",
  "0",
  "--epochs",
  "1",
  "--model-out",
  "/workspace/scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_256",
  "--out",
  "/workspace/scgpt/results/realdata_full_20261027_215022_050816/batch_256/train",
  "--batch-size",
  "256",
  "--max-length",
  "1200",
  "--seed",
  "42"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-2692046877c6

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_256/train_exit.json` · SHA256: `5a927ac8b080ef0319f262efae055dfe1851d9579b750dc0b824f83c41322cd5`

<details>
<summary>train_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train_exit.json", "sha256": "5a927ac8b080ef0319f262efae055dfe1851d9579b750dc0b824f83c41322cd5", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 86, "total_subprocess_wall_seconds": 116.2026743888855}
```
<!-- END_RESTORE_FILE -->


</details>
