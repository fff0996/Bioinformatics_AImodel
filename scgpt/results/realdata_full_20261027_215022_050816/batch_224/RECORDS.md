# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-b74ef0d5daf1

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_224/memory_guard.json` · SHA256: `b00a27796d01511ab6ff336560fb7bd0b405a22f55f9e9a122967e4909eeab28`

<details>
<summary>memory_guard.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/memory_guard.json", "sha256": "b00a27796d01511ab6ff336560fb7bd0b405a22f55f9e9a122967e4909eeab28", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "time_unix": 1793106811.8309977,
  "available_gib": 4.424278259277344,
  "swap_gib": 0.003997802734375,
  "cuda_allocated_gib": 69.09493350982666,
  "cuda_reserved_gib": 106.6015625
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-540181e7851a

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_224/train_command.json` · SHA256: `8349d619f0b582e9cb6e140d753a3b30ac884582238443e9c702b783272dcc86`

<details>
<summary>train_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train_command.json", "sha256": "8349d619f0b582e9cb6e140d753a3b30ac884582238443e9c702b783272dcc86", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "exec",
  "--user",
  "1000:1000",
  "-e",
  "SCGPT_GUARD_DIR=/workspace/scgpt/results/realdata_full_20261027_215022_050816/batch_224",
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
  "/workspace/scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_224",
  "--out",
  "/workspace/scgpt/results/realdata_full_20261027_215022_050816/batch_224/train",
  "--batch-size",
  "224",
  "--max-length",
  "1200",
  "--seed",
  "42"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-da88d9f590ed

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_224/train_exit.json` · SHA256: `18d49b683008137d5577ead7016929e88fbbbbd9b24f8660e589d2b3716f1af2`

<details>
<summary>train_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train_exit.json", "sha256": "18d49b683008137d5577ead7016929e88fbbbbd9b24f8660e589d2b3716f1af2", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 86, "total_subprocess_wall_seconds": 119.76167154312134}
```
<!-- END_RESTORE_FILE -->


</details>
