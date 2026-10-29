# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-4bdfabf0fbea

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch128_command.json` · SHA256: `64195cfd88aef528aef25c321a553dffd729015c3007f3a29f727f9c6678cb8a`

<details>
<summary>batch128_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch128_command.json", "sha256": "64195cfd88aef528aef25c321a553dffd729015c3007f3a29f727f9c6678cb8a", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-128",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
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

## file-aa88a0058d3b

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch128_exit.json` · SHA256: `011499c5509cca87d19241b3c6f932d69e0d1994ef15b794af4f5e8f9c011f4f`

<details>
<summary>batch128_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch128_exit.json", "sha256": "011499c5509cca87d19241b3c6f932d69e0d1994ef15b794af4f5e8f9c011f4f", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 101.6248550415039, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-70fae65b73ec

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch16_command.json` · SHA256: `926d30380833c37c983526f916308871f1f1cd9e77193a658bbcd634fc50ab71`

<details>
<summary>batch16_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch16_command.json", "sha256": "926d30380833c37c983526f916308871f1f1cd9e77193a658bbcd634fc50ab71", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-16",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
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

## file-15b4923dbefb

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch16_exit.json` · SHA256: `b00fc18a35f51440fa9094cb320f3a2fbbfdae074324a320578249399958d3ce`

<details>
<summary>batch16_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch16_exit.json", "sha256": "b00fc18a35f51440fa9094cb320f3a2fbbfdae074324a320578249399958d3ce", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 39.55798268318176, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-3ebcaf44da60

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch1_command.json` · SHA256: `60ec8ce27554da1d6c1d9184e79cbb2869dfb62fb798f43914ebb9d7c0887deb`

<details>
<summary>batch1_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch1_command.json", "sha256": "60ec8ce27554da1d6c1d9184e79cbb2869dfb62fb798f43914ebb9d7c0887deb", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
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

## file-cf01185a0e3d

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch1_exit.json` · SHA256: `854154bb82c5a6531e1d330ef7d10ac2019191725a32917cb4fdd40fbde4df2b`

<details>
<summary>batch1_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch1_exit.json", "sha256": "854154bb82c5a6531e1d330ef7d10ac2019191725a32917cb4fdd40fbde4df2b", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 33.55159115791321, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-933e30c7320a

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch2_command.json` · SHA256: `62aed622624751d90e4ded758d73a5e647abf40dd7089d4a7de4cb59a675b908`

<details>
<summary>batch2_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch2_command.json", "sha256": "62aed622624751d90e4ded758d73a5e647abf40dd7089d4a7de4cb59a675b908", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-2",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
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

## file-b67a43830adc

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch2_exit.json` · SHA256: `84309e6f043a753a8bdebb33e4336bedcb98e2281c3835e65f88e0b46ac4efe6`

<details>
<summary>batch2_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch2_exit.json", "sha256": "84309e6f043a753a8bdebb33e4336bedcb98e2281c3835e65f88e0b46ac4efe6", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 31.036025047302246, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-9bfea73d4c4f

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch32_command.json` · SHA256: `a8342d0a3b194126fce49311bba2211d8b4742acd99db35cd868a7fec5369694`

<details>
<summary>batch32_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch32_command.json", "sha256": "a8342d0a3b194126fce49311bba2211d8b4742acd99db35cd868a7fec5369694", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-32",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
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

## file-10fd24d51f84

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch32_exit.json` · SHA256: `f73374be02c5354fc34c9c33b19572f8d651753f9f379fc0db6a5b9595471935`

<details>
<summary>batch32_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch32_exit.json", "sha256": "f73374be02c5354fc34c9c33b19572f8d651753f9f379fc0db6a5b9595471935", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 44.054346323013306, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-a5fab56f8d30

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch4_command.json` · SHA256: `3d9411f0ed2dafe31b3d768dfbd91d7416b193c74afb16ada01fff531ec6590f`

<details>
<summary>batch4_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch4_command.json", "sha256": "3d9411f0ed2dafe31b3d768dfbd91d7416b193c74afb16ada01fff531ec6590f", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-4",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
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

## file-39035684484e

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch4_exit.json` · SHA256: `876a59587fe96ef4d5a79be976576b375689ec19e9e31c05da0418e072ac4a7a`

<details>
<summary>batch4_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch4_exit.json", "sha256": "876a59587fe96ef4d5a79be976576b375689ec19e9e31c05da0418e072ac4a7a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 32.035914182662964, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-dbff3d56a698

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch64_command.json` · SHA256: `01e1ecaa85dd479825577aea3322153312f9ec466a103c86eeb532667876fdca`

<details>
<summary>batch64_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch64_command.json", "sha256": "01e1ecaa85dd479825577aea3322153312f9ec466a103c86eeb532667876fdca", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-64",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
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

## file-b0acf34bdd5e

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch64_exit.json` · SHA256: `28a61ffc6e2b77c23efc1fd0377266eec0b9981329a1c1cd7a7684c1437f5c76`

<details>
<summary>batch64_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch64_exit.json", "sha256": "28a61ffc6e2b77c23efc1fd0377266eec0b9981329a1c1cd7a7684c1437f5c76", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 59.07036995887756, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-179e55ddf098

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch8_command.json` · SHA256: `341519bf05c3674abaf27a0ff8feee2ff9b66b8768e8c55f346f2795155c8f1a`

<details>
<summary>batch8_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch8_command.json", "sha256": "341519bf05c3674abaf27a0ff8feee2ff9b66b8768e8c55f346f2795155c8f1a", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-8",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
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

## file-1046d6d30473

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch8_exit.json` · SHA256: `ac7289fa5195797256359be67e6407313a170b0059668f4810dbb544ee8ef92f`

<details>
<summary>batch8_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch8_exit.json", "sha256": "ac7289fa5195797256359be67e6407313a170b0059668f4810dbb544ee8ef92f", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 35.546064376831055, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-7ebeb2b48e67

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch96_command.json` · SHA256: `6b7a9c747e67f2f89fc8b98869e784fcb8013a1988e66b9bb4effffda17ed0ea`

<details>
<summary>batch96_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch96_command.json", "sha256": "6b7a9c747e67f2f89fc8b98869e784fcb8013a1988e66b9bb4effffda17ed0ea", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-ntv2-batch_followup_20261028_143713_869625-96",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/ntv2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143713_869625",
  "-e",
  "MODEL_KIND=ntv2",
  "-e",
  "BATCH=96",
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

## file-d542bb67d7d0

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/batch96_exit.json` · SHA256: `fc83cb5baf073f2a6e011ba15af2f30a71c460cdb11ed4361a0a4f40e873e0b6`

<details>
<summary>batch96_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/batch96_exit.json", "sha256": "fc83cb5baf073f2a6e011ba15af2f30a71c460cdb11ed4361a0a4f40e873e0b6", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 82.12491416931152, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-0dec98df55b1

원래 경로: `ntv2/results/batch_followup_20261028_143713_869625/summary.json` · SHA256: `55686eb892a1e0f0a1c8f919330914ae3d12bf915e7bb1e9b709daee5e6910bd`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/batch_followup_20261028_143713_869625/summary.json", "sha256": "55686eb892a1e0f0a1c8f919330914ae3d12bf915e7bb1e9b709daee5e6910bd", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "complete",
  "model": "ntv2",
  "run_id": "batch_followup_20261028_143713_869625",
  "conditions": [
    {
      "batch_size": 1,
      "unique_inputs": 1,
      "processed_inputs": 78,
      "repeats": 78,
      "seconds": 15.119301557540894,
      "baseline": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 1.91796875,
        "peak_allocated_gib": 1.9016180038452148,
        "peak_reserved_gib": 1.91796875,
        "rss_gib": 1.3638572692871094,
        "system_available_gib": 110.14234924316406,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 2.078125,
        "peak_allocated_gib": 2.0577917098999023,
        "peak_reserved_gib": 2.078125,
        "rss_gib": 1.3638916015625,
        "system_available_gib": 110.14033889770508,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.1561737060546875,
      "samples": 29,
      "gpu_util_avg": 90.79310344827586,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.3638725280761719,
      "available_sampled_min_gib": 109.99013137817383,
      "inputs_per_second": 5.158968468427483,
      "allocated_fraction_shared_ram": 0.01691045877240291,
      "model_only_allocated_gib": 1.8565120697021484
    },
    {
      "batch_size": 2,
      "unique_inputs": 2,
      "processed_inputs": 72,
      "repeats": 36,
      "seconds": 15.314484119415283,
      "baseline": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 1.91796875,
        "peak_allocated_gib": 1.9016180038452148,
        "peak_reserved_gib": 1.91796875,
        "rss_gib": 1.3636474609375,
        "system_available_gib": 109.87678146362305,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 2.2578125,
        "peak_allocated_gib": 2.209461212158203,
        "peak_reserved_gib": 2.2578125,
        "rss_gib": 1.3637199401855469,
        "system_available_gib": 109.64832305908203,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.3078432083129883,
      "samples": 28,
      "gpu_util_avg": 95.42857142857143,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.3637046813964844,
      "available_sampled_min_gib": 109.5495719909668,
      "inputs_per_second": 4.701431627639378,
      "allocated_fraction_shared_ram": 0.01815684384268518,
      "model_only_allocated_gib": 1.8565120697021484
    },
    {
      "batch_size": 4,
      "unique_inputs": 4,
      "processed_inputs": 72,
      "repeats": 18,
      "seconds": 15.787213563919067,
      "baseline": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 1.91796875,
        "peak_allocated_gib": 1.9016180038452148,
        "peak_reserved_gib": 1.91796875,
        "rss_gib": 1.3637962341308594,
        "system_available_gib": 109.78800964355469,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 2.5546875,
        "peak_allocated_gib": 2.5201101303100586,
        "peak_reserved_gib": 2.5546875,
        "rss_gib": 1.3638763427734375,
        "system_available_gib": 109.26117324829102,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.6184921264648438,
      "samples": 30,
      "gpu_util_avg": 95.53333333333333,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.3638648986816406,
      "available_sampled_min_gib": 109.13958358764648,
      "inputs_per_second": 4.560652816184904,
      "allocated_fraction_shared_ram": 0.020709685171487136,
      "model_only_allocated_gib": 1.8565120697021484
    },
    {
      "batch_size": 8,
      "unique_inputs": 8,
      "processed_inputs": 72,
      "repeats": 9,
      "seconds": 16.121955633163452,
      "baseline": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 1.919921875,
        "peak_allocated_gib": 1.9016180038452148,
        "peak_reserved_gib": 1.919921875,
        "rss_gib": 1.3637351989746094,
        "system_available_gib": 109.6255989074707,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 3.205078125,
        "peak_allocated_gib": 3.140049457550049,
        "peak_reserved_gib": 3.205078125,
        "rss_gib": 1.3637809753417969,
        "system_available_gib": 108.60356140136719,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 1.238431453704834,
      "samples": 30,
      "gpu_util_avg": 95.46666666666667,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.36376953125,
      "available_sampled_min_gib": 108.32693481445312,
      "inputs_per_second": 4.465959443027703,
      "allocated_fraction_shared_ram": 0.02580420391419944,
      "model_only_allocated_gib": 1.8565120697021484
    },
    {
      "batch_size": 16,
      "unique_inputs": 16,
      "processed_inputs": 80,
      "repeats": 5,
      "seconds": 17.877100944519043,
      "baseline": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 1.919921875,
        "peak_allocated_gib": 1.9016180038452148,
        "peak_reserved_gib": 1.919921875,
        "rss_gib": 1.3637428283691406,
        "system_available_gib": 109.62878036499023,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 4.490234375,
        "peak_allocated_gib": 4.362948417663574,
        "peak_reserved_gib": 4.490234375,
        "rss_gib": 1.3637809753417969,
        "system_available_gib": 107.41394805908203,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 2.4613304138183594,
      "samples": 33,
      "gpu_util_avg": 95.36363636363636,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.36376953125,
      "available_sampled_min_gib": 107.04817581176758,
      "inputs_per_second": 4.474998504974447,
      "allocated_fraction_shared_ram": 0.035853706178362074,
      "model_only_allocated_gib": 1.8565120697021484
    },
    {
      "batch_size": 32,
      "unique_inputs": 32,
      "processed_inputs": 96,
      "repeats": 3,
      "seconds": 21.324120044708252,
      "baseline": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 1.919921875,
        "peak_allocated_gib": 1.9016180038452148,
        "peak_reserved_gib": 1.919921875,
        "rss_gib": 1.3636398315429688,
        "system_available_gib": 109.67229461669922,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 7.037109375,
        "peak_allocated_gib": 6.833856105804443,
        "peak_reserved_gib": 7.037109375,
        "rss_gib": 1.3544731140136719,
        "system_available_gib": 104.80695724487305,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 4.9322381019592285,
      "samples": 39,
      "gpu_util_avg": 95.48717948717949,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.363677978515625,
      "available_sampled_min_gib": 104.51055526733398,
      "inputs_per_second": 4.501944267745912,
      "allocated_fraction_shared_ram": 0.05615905700162497,
      "model_only_allocated_gib": 1.8565120697021484
    },
    {
      "batch_size": 64,
      "unique_inputs": 64,
      "processed_inputs": 128,
      "repeats": 2,
      "seconds": 28.851468086242676,
      "baseline": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 1.919921875,
        "peak_allocated_gib": 1.9016180038452148,
        "peak_reserved_gib": 1.919921875,
        "rss_gib": 1.3637351989746094,
        "system_available_gib": 109.73995590209961,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 12.154296875,
        "peak_allocated_gib": 11.746935367584229,
        "peak_reserved_gib": 12.154296875,
        "rss_gib": 1.3532638549804688,
        "system_available_gib": 99.7784423828125,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 9.845317363739014,
      "samples": 53,
      "gpu_util_avg": 95.35849056603773,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.3638229370117188,
      "available_sampled_min_gib": 99.4116439819336,
      "inputs_per_second": 4.4365160073443395,
      "allocated_fraction_shared_ram": 0.09653361187137716,
      "model_only_allocated_gib": 1.8565120697021484
    },
    {
      "batch_size": 96,
      "unique_inputs": 96,
      "processed_inputs": 192,
      "repeats": 2,
      "seconds": 42.53317928314209,
      "baseline": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 1.919921875,
        "peak_allocated_gib": 1.9016180038452148,
        "peak_reserved_gib": 1.919921875,
        "rss_gib": 1.3639068603515625,
        "system_available_gib": 110.21672821044922,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 17.24609375,
        "peak_allocated_gib": 16.676846027374268,
        "peak_reserved_gib": 17.24609375,
        "rss_gib": 1.3535728454589844,
        "system_available_gib": 95.03275299072266,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 14.775228023529053,
      "samples": 79,
      "gpu_util_avg": 93.17721518987342,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.3640098571777344,
      "available_sampled_min_gib": 94.6978874206543,
      "inputs_per_second": 4.514122932637172,
      "allocated_fraction_shared_ram": 0.13704648329705915,
      "model_only_allocated_gib": 1.8565120697021484
    },
    {
      "batch_size": 128,
      "unique_inputs": 128,
      "processed_inputs": 256,
      "repeats": 2,
      "seconds": 56.803858041763306,
      "baseline": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 1.921875,
        "peak_allocated_gib": 1.9016180038452148,
        "peak_reserved_gib": 1.921875,
        "rss_gib": 1.36383056640625,
        "system_available_gib": 110.43696594238281,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 1.9016180038452148,
        "reserved_gib": 22.365234375,
        "peak_allocated_gib": 21.592252254486084,
        "peak_reserved_gib": 22.365234375,
        "rss_gib": 1.3540916442871094,
        "system_available_gib": 89.91479110717773,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 19.69063425064087,
      "samples": 104,
      "gpu_util_avg": 94.82692307692308,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.3639793395996094,
      "available_sampled_min_gib": 89.72407150268555,
      "inputs_per_second": 4.506736141263218,
      "allocated_fraction_shared_ram": 0.17744016063247336,
      "model_only_allocated_gib": 1.8565120697021484
    }
  ],
  "failed": [],
  "input_audit": {
    "source": "existing real GRCh38 regions, central6000nt crops -> native ~1001tokens",
    "unique_available_inputs": 128,
    "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
    "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
  },
  "largest_successful_batch": 128,
  "stop_reason": "available_unique_input_limit",
  "reached80": false,
  "reached90": false
}
```
<!-- END_RESTORE_FILE -->


</details>
