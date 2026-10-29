# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-1c87cb4eba51

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch10176_command.json` · SHA256: `ec3117e67d4d4a41d8863cf05dba48b211b95ef586183dfd697a8fad32578459`

<details>
<summary>batch10176_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch10176_command.json", "sha256": "ec3117e67d4d4a41d8863cf05dba48b211b95ef586183dfd697a8fad32578459", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-scvi-batch_followup_20261028_142752_517769-10176",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/scvi:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_142752_517769",
  "-e",
  "MODEL_KIND=scvi",
  "-e",
  "BATCH=10176",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-0a712d2dbc92

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch10176_exit.json` · SHA256: `ae8ea08947da4008712f898f753d49e481f5e36a0c41b71f529d6dfe6c0e934d`

<details>
<summary>batch10176_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch10176_exit.json", "sha256": "ae8ea08947da4008712f898f753d49e481f5e36a0c41b71f529d6dfe6c0e934d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 21.019511461257935, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-f5d1a7b52e2b

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch1024_command.json` · SHA256: `ecdc5e2e73958ad025de782d07682d3ec2ee234b2c2d859296ee14218f6da874`

<details>
<summary>batch1024_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch1024_command.json", "sha256": "ecdc5e2e73958ad025de782d07682d3ec2ee234b2c2d859296ee14218f6da874", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-scvi-batch_followup_20261028_142752_517769-1024",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/scvi:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_142752_517769",
  "-e",
  "MODEL_KIND=scvi",
  "-e",
  "BATCH=1024",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-156e25541260

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch1024_exit.json` · SHA256: `4f9baea30a6fcf892d1bf1e5f63d66c3cd8e1c1cde4fc5d2a50a4fff019bfefa`

<details>
<summary>batch1024_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch1024_exit.json", "sha256": "4f9baea30a6fcf892d1bf1e5f63d66c3cd8e1c1cde4fc5d2a50a4fff019bfefa", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 20.519676685333252, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-298caf261fbf

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch128_command.json` · SHA256: `3cef9c3c169f9156652de75d14a1bab1bd0b9c53c197362ede74eea5ce392ca7`

<details>
<summary>batch128_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch128_command.json", "sha256": "3cef9c3c169f9156652de75d14a1bab1bd0b9c53c197362ede74eea5ce392ca7", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-scvi-batch_followup_20261028_142752_517769-128",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/scvi:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_142752_517769",
  "-e",
  "MODEL_KIND=scvi",
  "-e",
  "BATCH=128",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-df438598e52b

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch128_exit.json` · SHA256: `dd800e5d04a6bc752a1f5b76d8010f95be61d301b000bf94e4a9cceab0fec4bd`

<details>
<summary>batch128_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch128_exit.json", "sha256": "dd800e5d04a6bc752a1f5b76d8010f95be61d301b000bf94e4a9cceab0fec4bd", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 22.018195390701294, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-a4a5c6a11254

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch2048_command.json` · SHA256: `13280797ceb97f40e710634174ba93b29308adf98d33bb8e15837ef992228a80`

<details>
<summary>batch2048_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch2048_command.json", "sha256": "13280797ceb97f40e710634174ba93b29308adf98d33bb8e15837ef992228a80", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-scvi-batch_followup_20261028_142752_517769-2048",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/scvi:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_142752_517769",
  "-e",
  "MODEL_KIND=scvi",
  "-e",
  "BATCH=2048",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-df1ba247dc26

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch2048_exit.json` · SHA256: `2d2fe289158f4ccbb0b272c778dfcb11fd2618fe3b5d0ebefec9e499d5e2e42d`

<details>
<summary>batch2048_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch2048_exit.json", "sha256": "2d2fe289158f4ccbb0b272c778dfcb11fd2618fe3b5d0ebefec9e499d5e2e42d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 21.016561269760132, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-6d3b1f439b27

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch256_command.json` · SHA256: `bfe33b67fc4837ebc62e34360c25da85ac96c730d94fd0f787b7b2489a580490`

<details>
<summary>batch256_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch256_command.json", "sha256": "bfe33b67fc4837ebc62e34360c25da85ac96c730d94fd0f787b7b2489a580490", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-scvi-batch_followup_20261028_142752_517769-256",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/scvi:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_142752_517769",
  "-e",
  "MODEL_KIND=scvi",
  "-e",
  "BATCH=256",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-c6a5e2d0bc96

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch256_exit.json` · SHA256: `0682ad9a8b048c5e0c49a706da24f331f28f834f5f57f073de6d3e1a42838039`

<details>
<summary>batch256_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch256_exit.json", "sha256": "0682ad9a8b048c5e0c49a706da24f331f28f834f5f57f073de6d3e1a42838039", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 21.016342639923096, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-bf95a00d530b

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch4096_command.json` · SHA256: `93a2745e486ced3cefb0b83e3b2c2fb1d56a9b1586f3ba710d3be1fb5fba0e10`

<details>
<summary>batch4096_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch4096_command.json", "sha256": "93a2745e486ced3cefb0b83e3b2c2fb1d56a9b1586f3ba710d3be1fb5fba0e10", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-scvi-batch_followup_20261028_142752_517769-4096",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/scvi:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_142752_517769",
  "-e",
  "MODEL_KIND=scvi",
  "-e",
  "BATCH=4096",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-474af3a4954c

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch4096_exit.json` · SHA256: `c9eeb3f36ea0280f8f0e0a69770452008e41efbe495b714b0a9627c1693a32f3`

<details>
<summary>batch4096_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch4096_exit.json", "sha256": "c9eeb3f36ea0280f8f0e0a69770452008e41efbe495b714b0a9627c1693a32f3", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 21.0191593170166, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-c5cb7f050105

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch512_command.json` · SHA256: `5009f68f2ec4bb9e7a4b54ae21c53c791590d7cf7f5459a265e4600435f17949`

<details>
<summary>batch512_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch512_command.json", "sha256": "5009f68f2ec4bb9e7a4b54ae21c53c791590d7cf7f5459a265e4600435f17949", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-scvi-batch_followup_20261028_142752_517769-512",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/scvi:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_142752_517769",
  "-e",
  "MODEL_KIND=scvi",
  "-e",
  "BATCH=512",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-f4280ed6dfcc

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch512_exit.json` · SHA256: `943b7672e07da728d134d36ed090ab98d9845618aa24422ab5fc59c4473c4907`

<details>
<summary>batch512_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch512_exit.json", "sha256": "943b7672e07da728d134d36ed090ab98d9845618aa24422ab5fc59c4473c4907", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 21.018600702285767, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-b2306e38e45e

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch8192_command.json` · SHA256: `a40b1709553e325ab113512e1524bc70069aefb5db7a41f0c561e4cab6e88e64`

<details>
<summary>batch8192_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch8192_command.json", "sha256": "a40b1709553e325ab113512e1524bc70069aefb5db7a41f0c561e4cab6e88e64", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-scvi-batch_followup_20261028_142752_517769-8192",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/scvi:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_142752_517769",
  "-e",
  "MODEL_KIND=scvi",
  "-e",
  "BATCH=8192",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "-e",
  "NUMBA_CACHE_DIR=/work/.cache/numba",
  "-e",
  "HF_HOME=/work/.cache/huggingface",
  "-e",
  "HF_MODULES_CACHE=/work/.cache/modules",
  "-e",
  "HF_HUB_OFFLINE=1",
  "-e",
  "TRANSFORMERS_OFFLINE=1",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/work/.venv/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-5bee17abc586

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/batch8192_exit.json` · SHA256: `7c8172d1e231be3c07dd024c64cd89e65beeaa9e5eda5ee820aa6509cf9aa9df`

<details>
<summary>batch8192_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/batch8192_exit.json", "sha256": "7c8172d1e231be3c07dd024c64cd89e65beeaa9e5eda5ee820aa6509cf9aa9df", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 21.01873207092285, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-62ca31083f04

원래 경로: `scvi/results/batch_followup_20261028_142752_517769/summary.json` · SHA256: `09791a73bf2815d7740ec8c313f795628175cf1ba4eb0a427fcd088c012ed2a9`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/results/batch_followup_20261028_142752_517769/summary.json", "sha256": "09791a73bf2815d7740ec8c313f795628175cf1ba4eb0a427fcd088c012ed2a9", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "complete",
  "model": "scvi",
  "run_id": "batch_followup_20261028_142752_517769",
  "conditions": [
    {
      "batch_size": 128,
      "unique_inputs": 128,
      "processed_inputs": 1741696,
      "repeats": 13607,
      "seconds": 15.00095248222351,
      "baseline": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.037109375,
        "peak_allocated_gib": 0.036075592041015625,
        "peak_reserved_gib": 0.037109375,
        "rss_gib": 1.5389595031738281,
        "system_available_gib": 112.54663848876953,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.041015625,
        "peak_allocated_gib": 0.038999080657958984,
        "peak_reserved_gib": 0.041015625,
        "rss_gib": 1.5359611511230469,
        "system_available_gib": 112.52811813354492,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.0029234886169433594,
      "samples": 29,
      "gpu_util_avg": 15.689655172413794,
      "gpu_util_max": 19.0,
      "rss_sampled_peak_gib": 1.5399284362792969,
      "available_sampled_min_gib": 112.48947525024414,
      "inputs_per_second": 116105.69409268856,
      "allocated_fraction_shared_ram": 0.00032048547112676967,
      "model_only_allocated_gib": 0.004825592041015625
    },
    {
      "batch_size": 256,
      "unique_inputs": 256,
      "processed_inputs": 3053056,
      "repeats": 11926,
      "seconds": 15.000658750534058,
      "baseline": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.037109375,
        "peak_allocated_gib": 0.036075592041015625,
        "peak_reserved_gib": 0.037109375,
        "rss_gib": 1.5364151000976562,
        "system_available_gib": 112.5346450805664,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.056640625,
        "peak_allocated_gib": 0.041922569274902344,
        "peak_reserved_gib": 0.056640625,
        "rss_gib": 1.5383377075195312,
        "system_available_gib": 112.41555404663086,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.005846977233886719,
      "samples": 29,
      "gpu_util_avg": 17.413793103448278,
      "gpu_util_max": 20.0,
      "rss_sampled_peak_gib": 1.5383377075195312,
      "available_sampled_min_gib": 112.37102508544922,
      "inputs_per_second": 203528.12838244883,
      "allocated_fraction_shared_ram": 0.0003445100278836898,
      "model_only_allocated_gib": 0.004825592041015625
    },
    {
      "batch_size": 512,
      "unique_inputs": 512,
      "processed_inputs": 5149696,
      "repeats": 10058,
      "seconds": 15.000592470169067,
      "baseline": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.037109375,
        "peak_allocated_gib": 0.036075592041015625,
        "peak_reserved_gib": 0.037109375,
        "rss_gib": 1.5353012084960938,
        "system_available_gib": 112.4703483581543,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.056640625,
        "peak_allocated_gib": 0.04776954650878906,
        "peak_reserved_gib": 0.056640625,
        "rss_gib": 1.5434341430664062,
        "system_available_gib": 112.51658248901367,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.011693954467773438,
      "samples": 29,
      "gpu_util_avg": 20.17241379310345,
      "gpu_util_max": 24.0,
      "rss_sampled_peak_gib": 1.5434341430664062,
      "available_sampled_min_gib": 112.41012191772461,
      "inputs_per_second": 343299.50701886904,
      "allocated_fraction_shared_ram": 0.00039255914139753,
      "model_only_allocated_gib": 0.004825592041015625
    },
    {
      "batch_size": 1024,
      "unique_inputs": 1024,
      "processed_inputs": 7942144,
      "repeats": 7756,
      "seconds": 15.000842809677124,
      "baseline": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.037109375,
        "peak_allocated_gib": 0.036075592041015625,
        "peak_reserved_gib": 0.037109375,
        "rss_gib": 1.5372390747070312,
        "system_available_gib": 112.51875686645508,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.076171875,
        "peak_allocated_gib": 0.0594635009765625,
        "peak_reserved_gib": 0.076171875,
        "rss_gib": 1.5464630126953125,
        "system_available_gib": 112.47011184692383,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.023387908935546875,
      "samples": 29,
      "gpu_util_avg": 26.17241379310345,
      "gpu_util_max": 29.0,
      "rss_sampled_peak_gib": 1.5464630126953125,
      "available_sampled_min_gib": 112.4435043334961,
      "inputs_per_second": 529446.5184900465,
      "allocated_fraction_shared_ram": 0.0004886573684252105,
      "model_only_allocated_gib": 0.004825592041015625
    },
    {
      "batch_size": 2048,
      "unique_inputs": 2048,
      "processed_inputs": 9662464,
      "repeats": 4718,
      "seconds": 15.00176739692688,
      "baseline": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.037109375,
        "peak_allocated_gib": 0.036075592041015625,
        "peak_reserved_gib": 0.037109375,
        "rss_gib": 1.5156974792480469,
        "system_available_gib": 112.42618560791016,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.0859375,
        "peak_allocated_gib": 0.08395004272460938,
        "peak_reserved_gib": 0.0859375,
        "rss_gib": 1.5340042114257812,
        "system_available_gib": 112.47159194946289,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.04787445068359375,
      "samples": 29,
      "gpu_util_avg": 24.24137931034483,
      "gpu_util_max": 26.0,
      "rss_sampled_peak_gib": 1.5340042114257812,
      "available_sampled_min_gib": 112.3655014038086,
      "inputs_per_second": 644088.3760122398,
      "allocated_fraction_shared_ram": 0.0006898821341373881,
      "model_only_allocated_gib": 0.004825592041015625
    },
    {
      "batch_size": 4096,
      "unique_inputs": 4096,
      "processed_inputs": 9056256,
      "repeats": 2211,
      "seconds": 15.00086760520935,
      "baseline": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.037109375,
        "peak_allocated_gib": 0.036075592041015625,
        "peak_reserved_gib": 0.037109375,
        "rss_gib": 1.5226860046386719,
        "system_available_gib": 112.51547241210938,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.150390625,
        "peak_allocated_gib": 0.13182449340820312,
        "peak_reserved_gib": 0.150390625,
        "rss_gib": 1.5625953674316406,
        "system_available_gib": 112.4080581665039,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.0957489013671875,
      "samples": 29,
      "gpu_util_avg": 24.862068965517242,
      "gpu_util_max": 28.0,
      "rss_sampled_peak_gib": 1.5625953674316406,
      "available_sampled_min_gib": 112.37411880493164,
      "inputs_per_second": 603715.4808869212,
      "allocated_fraction_shared_ram": 0.0010833033539049267,
      "model_only_allocated_gib": 0.004825592041015625
    },
    {
      "batch_size": 8192,
      "unique_inputs": 8192,
      "processed_inputs": 1261568,
      "repeats": 154,
      "seconds": 15.027137517929077,
      "baseline": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.037109375,
        "peak_allocated_gib": 0.036075592041015625,
        "peak_reserved_gib": 0.037109375,
        "rss_gib": 1.5337028503417969,
        "system_available_gib": 112.45479202270508,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.24609375,
        "peak_allocated_gib": 0.22317886352539062,
        "peak_reserved_gib": 0.24609375,
        "rss_gib": 1.5465431213378906,
        "system_available_gib": 112.2279167175293,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.187103271484375,
      "samples": 29,
      "gpu_util_avg": 5.793103448275862,
      "gpu_util_max": 7.0,
      "rss_sampled_peak_gib": 1.6074943542480469,
      "available_sampled_min_gib": 112.12862777709961,
      "inputs_per_second": 83952.64889902063,
      "allocated_fraction_shared_ram": 0.0018340325468127363,
      "model_only_allocated_gib": 0.004825592041015625
    },
    {
      "batch_size": 10176,
      "unique_inputs": 10176,
      "processed_inputs": 1241472,
      "repeats": 122,
      "seconds": 15.081507444381714,
      "baseline": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.037109375,
        "peak_allocated_gib": 0.036075592041015625,
        "peak_reserved_gib": 0.037109375,
        "rss_gib": 1.540008544921875,
        "system_available_gib": 112.4418830871582,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.036075592041015625,
        "reserved_gib": 0.287109375,
        "peak_allocated_gib": 0.26955747604370117,
        "peak_reserved_gib": 0.287109375,
        "rss_gib": 1.5558509826660156,
        "system_available_gib": 112.08157348632812,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.23348188400268555,
      "samples": 29,
      "gpu_util_avg": 4.724137931034483,
      "gpu_util_max": 8.0,
      "rss_sampled_peak_gib": 1.6315498352050781,
      "available_sampled_min_gib": 112.01637268066406,
      "inputs_per_second": 82317.500725863,
      "allocated_fraction_shared_ram": 0.002215161312731562,
      "model_only_allocated_gib": 0.004825592041015625
    }
  ],
  "failed": [],
  "input_audit": {
    "source": "saved full10176cell SCVI model and its raw-count AnnData, fixed2000genes; latent posterior mean",
    "unique_available_inputs": 10176,
    "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
    "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
  },
  "largest_successful_batch": 10176,
  "stop_reason": "available_unique_input_limit",
  "reached80": false,
  "reached90": false
}
```
<!-- END_RESTORE_FILE -->


</details>
