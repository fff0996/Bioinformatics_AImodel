# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-6942963462d5

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch16_command.json` · SHA256: `57b48e587c1fad0b009bd89509b371113c709c42d775d3b2ab35d2cac1ce9363`

<details>
<summary>batch16_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch16_command.json", "sha256": "57b48e587c1fad0b009bd89509b371113c709c42d775d3b2ab35d2cac1ce9363", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-16",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=16",
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

## file-960a11627633

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch16_exit.json` · SHA256: `d02138c99a2d5625cb42e946db1356f6cdf76e474be4f967fb304f64d9229e76`

<details>
<summary>batch16_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch16_exit.json", "sha256": "d02138c99a2d5625cb42e946db1356f6cdf76e474be4f967fb304f64d9229e76", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 20.03323793411255, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-97f5e3868ce1

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch1_command.json` · SHA256: `e71d34b318fa5fdff5e507b38ab49191be2d44a236943538964c934e4de98894`

<details>
<summary>batch1_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch1_command.json", "sha256": "e71d34b318fa5fdff5e507b38ab49191be2d44a236943538964c934e4de98894", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=1",
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

## file-a8637146bbce

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch1_exit.json` · SHA256: `8213ea08111732ff90c8c5e60532bcc59927aafb234cf38557cd6b9b6833ddf3`

<details>
<summary>batch1_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch1_exit.json", "sha256": "8213ea08111732ff90c8c5e60532bcc59927aafb234cf38557cd6b9b6833ddf3", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 20.52164626121521, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-d2c7fe7761b0

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch2_command.json` · SHA256: `d57ea84b522c6f3d69549feed7e38d436cd3fc85e25fcecdecc0e58caa9ab087`

<details>
<summary>batch2_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch2_command.json", "sha256": "d57ea84b522c6f3d69549feed7e38d436cd3fc85e25fcecdecc0e58caa9ab087", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-2",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=2",
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

## file-9f452ec03983

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch2_exit.json` · SHA256: `b625c2293fb40e593ee8f4409ac024e326b1c61c8070ac6d68a3e641ff7ff9a1`

<details>
<summary>batch2_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch2_exit.json", "sha256": "b625c2293fb40e593ee8f4409ac024e326b1c61c8070ac6d68a3e641ff7ff9a1", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 19.52936625480652, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-0fafa31587fb

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch32_command.json` · SHA256: `23ef4ef4ab2703e877178f209024f01415487c907bf70768864cc61e56a54af4`

<details>
<summary>batch32_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch32_command.json", "sha256": "23ef4ef4ab2703e877178f209024f01415487c907bf70768864cc61e56a54af4", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-32",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=32",
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

## file-480b6493222b

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch32_exit.json` · SHA256: `8fd674c9cfa70155f1aded873aa595c384f8db83276337daba74b1693543530f`

<details>
<summary>batch32_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch32_exit.json", "sha256": "8fd674c9cfa70155f1aded873aa595c384f8db83276337daba74b1693543530f", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 20.53589653968811, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-b0f9eb79f667

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch4_command.json` · SHA256: `9d56661fd0c3df65487807d8cb46a03faf2d7e142c6f020748ba43b7b482baef`

<details>
<summary>batch4_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch4_command.json", "sha256": "9d56661fd0c3df65487807d8cb46a03faf2d7e142c6f020748ba43b7b482baef", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-4",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=4",
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

## file-6a794341f0d1

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch4_exit.json` · SHA256: `70e28e15154426817a1ae5b92b67f24a8c39da0db8b62e5d97281f52eef8ccef`

<details>
<summary>batch4_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch4_exit.json", "sha256": "70e28e15154426817a1ae5b92b67f24a8c39da0db8b62e5d97281f52eef8ccef", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 19.525622367858887, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-8b4e8d75faff

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch64_command.json` · SHA256: `5e5cd4725cb07c9667a267c9fc63e1fad2b5defe673c69db5e5e2f9d50a6abe8`

<details>
<summary>batch64_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch64_command.json", "sha256": "5e5cd4725cb07c9667a267c9fc63e1fad2b5defe673c69db5e5e2f9d50a6abe8", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-64",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=64",
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

## file-e40a114fdd1f

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch64_exit.json` · SHA256: `34464736248c4a0128dbb20f3c4f53dc749cee31bebbb17b7300968617d35256`

<details>
<summary>batch64_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch64_exit.json", "sha256": "34464736248c4a0128dbb20f3c4f53dc749cee31bebbb17b7300968617d35256", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 21.53317427635193, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-611328585f4c

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch76_command.json` · SHA256: `d9865f13ac7d6b4c26617d746968a7040a58fa06e9cd53a64c85607b6adb7085`

<details>
<summary>batch76_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch76_command.json", "sha256": "d9865f13ac7d6b4c26617d746968a7040a58fa06e9cd53a64c85607b6adb7085", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-76",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=76",
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

## file-cef7b9b234d5

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch76_exit.json` · SHA256: `9b790d67afea295ec0f6e23784a634d161223939b06ad000655c5f41679f1d64`

<details>
<summary>batch76_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch76_exit.json", "sha256": "9b790d67afea295ec0f6e23784a634d161223939b06ad000655c5f41679f1d64", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 23.042716026306152, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-4c5f6204fea6

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch8_command.json` · SHA256: `cd0e22588a1f3473921237e7d6002cd95863624a6d83a565116917b513c18a40`

<details>
<summary>batch8_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch8_command.json", "sha256": "cd0e22588a1f3473921237e7d6002cd95863624a6d83a565116917b513c18a40", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-rnafm-batch_followup_20261028_144452_703996-8",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/rnafm:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144452_703996",
  "-e",
  "MODEL_KIND=rnafm",
  "-e",
  "BATCH=8",
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

## file-d28807f9904d

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/batch8_exit.json` · SHA256: `b58e45be1f0769948d7ab8d5e0f0f5f4a6d4ec873780e3ae6cbbe069e9928a8a`

<details>
<summary>batch8_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/batch8_exit.json", "sha256": "b58e45be1f0769948d7ab8d5e0f0f5f4a6d4ec873780e3ae6cbbe069e9928a8a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 19.535288095474243, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-f6d38fd6ff14

원래 경로: `rnafm/results/batch_followup_20261028_144452_703996/summary.json` · SHA256: `9be3163318b89cfa7deec9a9b122363d17e3663ef6b01b261c33cbac6615c3a3`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "rnafm/results/batch_followup_20261028_144452_703996/summary.json", "sha256": "9be3163318b89cfa7deec9a9b122363d17e3663ef6b01b261c33cbac6615c3a3", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "complete",
  "model": "rnafm",
  "run_id": "batch_followup_20261028_144452_703996",
  "conditions": [
    {
      "batch_size": 1,
      "unique_inputs": 1,
      "processed_inputs": 667,
      "repeats": 667,
      "seconds": 15.004332065582275,
      "baseline": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.451171875,
        "peak_allocated_gib": 0.40199756622314453,
        "peak_reserved_gib": 0.451171875,
        "rss_gib": 1.4755935668945312,
        "system_available_gib": 111.98821640014648,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.494140625,
        "peak_allocated_gib": 0.4484577178955078,
        "peak_reserved_gib": 0.494140625,
        "rss_gib": 1.4783821105957031,
        "system_available_gib": 111.90662002563477,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.04646015167236328,
      "samples": 29,
      "gpu_util_avg": 91.82758620689656,
      "gpu_util_max": 95.0,
      "rss_sampled_peak_gib": 1.4782981872558594,
      "available_sampled_min_gib": 111.81607818603516,
      "inputs_per_second": 44.45382820672169,
      "allocated_fraction_shared_ram": 0.0036853223351778256,
      "model_only_allocated_gib": 0.37074756622314453
    },
    {
      "batch_size": 2,
      "unique_inputs": 2,
      "processed_inputs": 608,
      "repeats": 304,
      "seconds": 15.023655652999878,
      "baseline": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.451171875,
        "peak_allocated_gib": 0.40199756622314453,
        "peak_reserved_gib": 0.451171875,
        "rss_gib": 1.4759597778320312,
        "system_available_gib": 111.76081848144531,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.57421875,
        "peak_allocated_gib": 0.49300479888916016,
        "peak_reserved_gib": 0.57421875,
        "rss_gib": 1.4772415161132812,
        "system_available_gib": 111.79847717285156,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.09100723266601562,
      "samples": 28,
      "gpu_util_avg": 92.0,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.4771728515625,
      "available_sampled_min_gib": 111.60046768188477,
      "inputs_per_second": 40.469511152473494,
      "allocated_fraction_shared_ram": 0.004051399996463911,
      "model_only_allocated_gib": 0.37074756622314453
    },
    {
      "batch_size": 4,
      "unique_inputs": 4,
      "processed_inputs": 628,
      "repeats": 157,
      "seconds": 15.062520503997803,
      "baseline": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.451171875,
        "peak_allocated_gib": 0.40199756622314453,
        "peak_reserved_gib": 0.451171875,
        "rss_gib": 1.4764289855957031,
        "system_available_gib": 111.75282287597656,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.630859375,
        "peak_allocated_gib": 0.5838923454284668,
        "peak_reserved_gib": 0.630859375,
        "rss_gib": 1.4776229858398438,
        "system_available_gib": 111.79141998291016,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.18189477920532227,
      "samples": 28,
      "gpu_util_avg": 92.25,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.4775390625,
      "available_sampled_min_gib": 111.57084655761719,
      "inputs_per_second": 41.69288930317605,
      "allocated_fraction_shared_ram": 0.004798292940625183,
      "model_only_allocated_gib": 0.37074756622314453
    },
    {
      "batch_size": 8,
      "unique_inputs": 8,
      "processed_inputs": 616,
      "repeats": 77,
      "seconds": 15.030980110168457,
      "baseline": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.451171875,
        "peak_allocated_gib": 0.40199756622314453,
        "peak_reserved_gib": 0.451171875,
        "rss_gib": 1.452362060546875,
        "system_available_gib": 111.77954864501953,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.826171875,
        "peak_allocated_gib": 0.765531063079834,
        "peak_reserved_gib": 0.826171875,
        "rss_gib": 1.4540634155273438,
        "system_available_gib": 111.55071258544922,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.36353349685668945,
      "samples": 27,
      "gpu_util_avg": 95.18518518518519,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.4539833068847656,
      "available_sampled_min_gib": 111.37438201904297,
      "inputs_per_second": 40.982024823735614,
      "allocated_fraction_shared_ram": 0.006290958127066715,
      "model_only_allocated_gib": 0.37074756622314453
    },
    {
      "batch_size": 16,
      "unique_inputs": 16,
      "processed_inputs": 624,
      "repeats": 39,
      "seconds": 15.183516263961792,
      "baseline": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.451171875,
        "peak_allocated_gib": 0.40199756622314453,
        "peak_reserved_gib": 0.451171875,
        "rss_gib": 1.450653076171875,
        "system_available_gib": 111.74444580078125,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 1.21484375,
        "peak_allocated_gib": 1.126136302947998,
        "peak_reserved_gib": 1.21484375,
        "rss_gib": 1.4529609680175781,
        "system_available_gib": 111.2033576965332,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.7241387367248535,
      "samples": 28,
      "gpu_util_avg": 91.96428571428571,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.4528846740722656,
      "available_sampled_min_gib": 110.95959091186523,
      "inputs_per_second": 41.097199696823154,
      "allocated_fraction_shared_ram": 0.009254329012742835,
      "model_only_allocated_gib": 0.37074756622314453
    },
    {
      "batch_size": 32,
      "unique_inputs": 32,
      "processed_inputs": 640,
      "repeats": 20,
      "seconds": 15.618630170822144,
      "baseline": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.451171875,
        "peak_allocated_gib": 0.40199756622314453,
        "peak_reserved_gib": 0.451171875,
        "rss_gib": 1.4502372741699219,
        "system_available_gib": 111.78114700317383,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 1.958984375,
        "peak_allocated_gib": 1.852200984954834,
        "peak_reserved_gib": 1.958984375,
        "rss_gib": 1.4528694152832031,
        "system_available_gib": 110.4385871887207,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 1.4502034187316895,
      "samples": 29,
      "gpu_util_avg": 95.48275862068965,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.452911376953125,
      "available_sampled_min_gib": 110.24377059936523,
      "inputs_per_second": 40.97670493508531,
      "allocated_fraction_shared_ram": 0.015220961501398197,
      "model_only_allocated_gib": 0.37074756622314453
    },
    {
      "batch_size": 64,
      "unique_inputs": 64,
      "processed_inputs": 640,
      "repeats": 10,
      "seconds": 15.61797285079956,
      "baseline": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.451171875,
        "peak_allocated_gib": 0.40199756622314453,
        "peak_reserved_gib": 0.451171875,
        "rss_gib": 1.451385498046875,
        "system_available_gib": 111.81875610351562,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 3.455078125,
        "peak_allocated_gib": 3.2985291481018066,
        "peak_reserved_gib": 3.455078125,
        "rss_gib": 1.4525527954101562,
        "system_available_gib": 108.84786605834961,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 2.896531581878662,
      "samples": 29,
      "gpu_util_avg": 95.44827586206897,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.4527587890625,
      "available_sampled_min_gib": 108.68402862548828,
      "inputs_per_second": 40.97842953845545,
      "allocated_fraction_shared_ram": 0.027106553544847447,
      "model_only_allocated_gib": 0.37074756622314453
    },
    {
      "batch_size": 76,
      "unique_inputs": 76,
      "processed_inputs": 684,
      "repeats": 9,
      "seconds": 16.62934923171997,
      "baseline": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 0.451171875,
        "peak_allocated_gib": 0.40199756622314453,
        "peak_reserved_gib": 0.451171875,
        "rss_gib": 1.4512710571289062,
        "system_available_gib": 111.7845687866211,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.40199756622314453,
        "reserved_gib": 4.005859375,
        "peak_allocated_gib": 3.8442835807800293,
        "peak_reserved_gib": 4.005859375,
        "rss_gib": 1.4525260925292969,
        "system_available_gib": 108.41837692260742,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 3.4422860145568848,
      "samples": 31,
      "gpu_util_avg": 95.41935483870968,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.4527091979980469,
      "available_sampled_min_gib": 108.19926834106445,
      "inputs_per_second": 41.132096660480926,
      "allocated_fraction_shared_ram": 0.031591437894056024,
      "model_only_allocated_gib": 0.37074756622314453
    }
  ],
  "failed": [],
  "input_audit": {
    "source": "official tutorial RNAcentral actual unique sequences >=512nt, first512nt explicit crop; pool limit retained",
    "unique_available_inputs": 76,
    "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
    "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
  },
  "largest_successful_batch": 76,
  "stop_reason": "available_unique_input_limit",
  "reached80": false,
  "reached90": false
}
```
<!-- END_RESTORE_FILE -->


</details>
