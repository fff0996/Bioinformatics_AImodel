# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-4c42cf7f3ac3

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_284/train_command.json` · SHA256: `c9a7bd8d24e4a48a166eaf2eeda3f29f25cfe2888b2141a58e981ef760b35a5f`

<details>
<summary>train_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train_command.json", "sha256": "c9a7bd8d24e4a48a166eaf2eeda3f29f25cfe2888b2141a58e981ef760b35a5f", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker exec --user 1000:1000 -e SCGPT_GUARD_DIR=/workspace/scgpt/results/realdata_full_20261027_215022_050816 -e MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib -e NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba scgpt-spark python -u /workspace/scgpt/scripts/full_realdata.py --mode train --n-cells 0 --epochs 1 --model-out /workspace/scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_284 --out /workspace/scgpt/results/realdata_full_20261027_215022_050816/batch_284/train --batch-size 284 --max-length 1200 --seed 42"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-d5ac35823f14

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_284/train_exit.json` · SHA256: `7795d606dfc25543ac083d42cb76eb1bcaf0cdeaa3211d3128b77927f54046e3`

<details>
<summary>train_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train_exit.json", "sha256": "7795d606dfc25543ac083d42cb76eb1bcaf0cdeaa3211d3128b77927f54046e3", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 86, "total_subprocess_wall_seconds": 110.33346772193909, "finished": "2026-10-27T21:59:10.452825"}
```
<!-- END_RESTORE_FILE -->


</details>
