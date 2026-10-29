# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-a8217f1c2e63

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_8/test_command.json` · SHA256: `f88f5c717c8ce99d209809b411a0cd3bca63933f6ee724c4ed199aafa8bfb642`

<details>
<summary>test_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test_command.json", "sha256": "f88f5c717c8ce99d209809b411a0cd3bca63933f6ee724c4ed199aafa8bfb642", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker exec --user 1000:1000 -e SCGPT_GUARD_DIR=/workspace/scgpt/results/realdata_full_20261027_215022_050816 -e MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib -e NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba scgpt-spark python -u /workspace/scgpt/scripts/full_realdata.py --mode predict --bundle /workspace/scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001 --input /workspace/scgpt/data/pancreas_demo/demo_test.h5ad --out /workspace/scgpt/results/realdata_full_20261027_215022_050816/batch_8/test --batch-size 8 --max-length 1200 --seed 42"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-bf98ded5f664

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_8/test_exit.json` · SHA256: `6eaa16d1e6820ea84f970ce175fdf74206ec7713c24288a21dd63509216f2a94`

<details>
<summary>test_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test_exit.json", "sha256": "6eaa16d1e6820ea84f970ce175fdf74206ec7713c24288a21dd63509216f2a94", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "total_subprocess_wall_seconds": 103.54626226425171, "finished": "2026-10-27T21:57:20.088440"}
```
<!-- END_RESTORE_FILE -->


</details>

## file-7f8a8c5c7ac1

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_8/train_command.json` · SHA256: `fe1cf87e76a2fb179e1625fce7d88e911f13ff46e56eb293fc2f6b616c1fe948`

<details>
<summary>train_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train_command.json", "sha256": "fe1cf87e76a2fb179e1625fce7d88e911f13ff46e56eb293fc2f6b616c1fe948", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker exec --user 1000:1000 -e SCGPT_GUARD_DIR=/workspace/scgpt/results/realdata_full_20261027_215022_050816 -e MPLCONFIGDIR=/workspace/scgpt/.cache/matplotlib -e NUMBA_CACHE_DIR=/workspace/scgpt/.cache/numba scgpt-spark python -u /workspace/scgpt/scripts/full_realdata.py --mode train --n-cells 0 --epochs 1 --model-out /workspace/scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8 --out /workspace/scgpt/results/realdata_full_20261027_215022_050816/batch_8/train --batch-size 8 --max-length 1200 --seed 42"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-130773255cd2

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/batch_8/train_exit.json` · SHA256: `75b1dd1b5ae95cddd9cb949ce875c68e20faab7cf2788934528bc0a2fb7952c0`

<details>
<summary>train_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train_exit.json", "sha256": "75b1dd1b5ae95cddd9cb949ce875c68e20faab7cf2788934528bc0a2fb7952c0", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "total_subprocess_wall_seconds": 314.4460768699646, "finished": "2026-10-27T21:55:36.521035"}
```
<!-- END_RESTORE_FILE -->


</details>
