# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-f5d3d77156e5

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch128_command.json` · SHA256: `536cd529787b5caa7a24cc3c6127f4cac35edbb928b3a00d151be2bfbdb782e0`

<details>
<summary>batch128_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch128_command.json", "sha256": "536cd529787b5caa7a24cc3c6127f4cac35edbb928b3a00d151be2bfbdb782e0", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-128",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-76710d3c0a5e

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch128_exit.json` · SHA256: `a8ee0683d32f4814d486687b1981278263f609042662d197987c971773e6c517`

<details>
<summary>batch128_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch128_exit.json", "sha256": "a8ee0683d32f4814d486687b1981278263f609042662d197987c971773e6c517", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 26.552683115005493, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-963f0a1d0f91

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch16_command.json` · SHA256: `b31b16f01d4174dba19516f3d52d382879565c5768cb8dd32634d574ab5aae4c`

<details>
<summary>batch16_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch16_command.json", "sha256": "b31b16f01d4174dba19516f3d52d382879565c5768cb8dd32634d574ab5aae4c", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-16",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-bf3f20dcdccc

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch16_exit.json` · SHA256: `912b16a286abb6f12c162d3bb3d111018ac4dbab354fe97dce86ba22e6eb4734`

<details>
<summary>batch16_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch16_exit.json", "sha256": "912b16a286abb6f12c162d3bb3d111018ac4dbab354fe97dce86ba22e6eb4734", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 19.530046463012695, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-4bfc78ff5801

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch192_command.json` · SHA256: `5d4ba48ea477f886070f62deb652652b83cb7560dd1e82f68c9b7b469598a351`

<details>
<summary>batch192_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch192_command.json", "sha256": "5d4ba48ea477f886070f62deb652652b83cb7560dd1e82f68c9b7b469598a351", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-192",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
  "-e",
  "BATCH=192",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-c2d02da79d1a

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch192_exit.json` · SHA256: `a22310b02cf7a998cdf848a96e5e3b63569d7589301d993313a4373197e51502`

<details>
<summary>batch192_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch192_exit.json", "sha256": "a22310b02cf7a998cdf848a96e5e3b63569d7589301d993313a4373197e51502", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 30.557506322860718, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-9f5aada46d8b

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch1_command.json` · SHA256: `b4d28840daf2fd8e38a58b0283180301ea2c26cff98c8cdfa3330603d3ec0a4d`

<details>
<summary>batch1_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch1_command.json", "sha256": "b4d28840daf2fd8e38a58b0283180301ea2c26cff98c8cdfa3330603d3ec0a4d", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-272eb567ec32

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch1_exit.json` · SHA256: `0a7d8e8974c43310f01ef1f5d266a2a3612429be42ea1396ceb4ee534c4b57be`

<details>
<summary>batch1_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch1_exit.json", "sha256": "0a7d8e8974c43310f01ef1f5d266a2a3612429be42ea1396ceb4ee534c4b57be", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 19.525418519973755, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-5434914db69a

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch256_command.json` · SHA256: `5d9ea74a1cdadc52987965c02cd15a6d3b2f138e6a9b7cafa1d43eb265af4ae7`

<details>
<summary>batch256_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch256_command.json", "sha256": "5d9ea74a1cdadc52987965c02cd15a6d3b2f138e6a9b7cafa1d43eb265af4ae7", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-256",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-47d95e36b7f3

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch256_exit.json` · SHA256: `8635f0d68f2cacead6fc7014dfab52d8f9af9cd7e8c4c5279fe76c83744c1b22`

<details>
<summary>batch256_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch256_exit.json", "sha256": "8635f0d68f2cacead6fc7014dfab52d8f9af9cd7e8c4c5279fe76c83744c1b22", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 39.06377959251404, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-e590528b2a9e

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch2_command.json` · SHA256: `7276df22bf7530a4c5c12fb84ab2eb5e5860326258f01fd53874c11943535b31`

<details>
<summary>batch2_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch2_command.json", "sha256": "7276df22bf7530a4c5c12fb84ab2eb5e5860326258f01fd53874c11943535b31", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-2",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-505f264c64e9

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch2_exit.json` · SHA256: `1ae7622c3e8d34956a85612a2cee78346850c8c572b741eeafb734464ddcca42`

<details>
<summary>batch2_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch2_exit.json", "sha256": "1ae7622c3e8d34956a85612a2cee78346850c8c572b741eeafb734464ddcca42", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.523372173309326, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-a6e6f3b57790

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch32_command.json` · SHA256: `f76f92e90ab72a49435e81c6d878e203f0cb447bf2637250c3f43cea3c74fd61`

<details>
<summary>batch32_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch32_command.json", "sha256": "f76f92e90ab72a49435e81c6d878e203f0cb447bf2637250c3f43cea3c74fd61", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-32",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-553b66b930aa

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch32_exit.json` · SHA256: `66dcec88c5315e5a557609b7a79c97189de35b16f27d3dd90d2cfe720c1eddd3`

<details>
<summary>batch32_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch32_exit.json", "sha256": "66dcec88c5315e5a557609b7a79c97189de35b16f27d3dd90d2cfe720c1eddd3", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 20.030255556106567, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-ed8225249ebe

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch384_command.json` · SHA256: `7130681e2a8a455c8aba62afd7424896e9fe7f772f719fea6d1b88388679677f`

<details>
<summary>batch384_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch384_command.json", "sha256": "7130681e2a8a455c8aba62afd7424896e9fe7f772f719fea6d1b88388679677f", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-384",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
  "-e",
  "BATCH=384",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-72457a0ef2e1

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch384_exit.json` · SHA256: `594ae36e4ffab49992ec8c23ea385e689db9fe36ca34a7576907e55c6ae33416`

<details>
<summary>batch384_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch384_exit.json", "sha256": "594ae36e4ffab49992ec8c23ea385e689db9fe36ca34a7576907e55c6ae33416", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 57.07663059234619, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-8a9587d01566

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch4_command.json` · SHA256: `821b01455563f32bd306cf3a1befcb86854562112b80510d95e3f31cbe106556`

<details>
<summary>batch4_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch4_command.json", "sha256": "821b01455563f32bd306cf3a1befcb86854562112b80510d95e3f31cbe106556", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-4",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-3bc1b3fc4037

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch4_exit.json` · SHA256: `0ce19470b659b8103331ad6b5f3c60721d1bd1d4f2a03993579e6c6d742ecb42`

<details>
<summary>batch4_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch4_exit.json", "sha256": "0ce19470b659b8103331ad6b5f3c60721d1bd1d4f2a03993579e6c6d742ecb42", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.525993585586548, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-16e63bb4b6ab

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch512_command.json` · SHA256: `035fea871abbad94a516b086d69a95217595c7ea0f5511ede969e1b41144e934`

<details>
<summary>batch512_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch512_command.json", "sha256": "035fea871abbad94a516b086d69a95217595c7ea0f5511ede969e1b41144e934", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-512",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-8dffe03d9ccd

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch512_exit.json` · SHA256: `67d9b14c30491efe02fc10d44646ee9922787ed343feea9c478d6d4f89678e9a`

<details>
<summary>batch512_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch512_exit.json", "sha256": "67d9b14c30491efe02fc10d44646ee9922787ed343feea9c478d6d4f89678e9a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 75.64113092422485, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-2fdea473efab

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch64_command.json` · SHA256: `755ed66922cea3331089e80dfa0520506e23634ebd1d463b02e09abdaa83b4d1`

<details>
<summary>batch64_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch64_command.json", "sha256": "755ed66922cea3331089e80dfa0520506e23634ebd1d463b02e09abdaa83b4d1", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-64",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-dc24595309d6

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch64_exit.json` · SHA256: `a7ca84781febd5db8ccc4332f55951dc98fad333ea5ae2c674c42675facfb57d`

<details>
<summary>batch64_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch64_exit.json", "sha256": "a7ca84781febd5db8ccc4332f55951dc98fad333ea5ae2c674c42675facfb57d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 23.04512929916382, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-f724ae72c57e

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch8_command.json` · SHA256: `83e52f5118f838e46d71aaa569368f2bfbd2d0b7ee2540d163e309cdd08e03cf`

<details>
<summary>batch8_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch8_command.json", "sha256": "83e52f5118f838e46d71aaa569368f2bfbd2d0b7ee2540d163e309cdd08e03cf", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-8",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-4219577db149

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch8_exit.json` · SHA256: `74c20978aca58c52b3bde7fcf0fae45da1a7e8f34b3d82e2bb78fa3d1342be34`

<details>
<summary>batch8_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch8_exit.json", "sha256": "74c20978aca58c52b3bde7fcf0fae45da1a7e8f34b3d82e2bb78fa3d1342be34", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 19.025710105895996, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-fe0f9a3b64c3

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch96_command.json` · SHA256: `dee2de496c6be15adb459c1532fec37ff8ecec43ece6d8568f9c7508f5cbf53f`

<details>
<summary>batch96_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch96_command.json", "sha256": "dee2de496c6be15adb459c1532fec37ff8ecec43ece6d8568f9c7508f5cbf53f", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-dnabert2-batch_followup_20261028_143041_359149-96",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/dnabert2:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_143041_359149",
  "-e",
  "MODEL_KIND=dnabert2",
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
  "-v",
  "/home/hyein/dnabert2-benchmark:/workspace:ro",
  "nvcr.io/nvidia/pytorch:26.06-py3",
  "/workspace/miniforge/envs/dnabert2/bin/python",
  "-u",
  "/support/remaining_followups/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-c02d0cb68ad3

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/batch96_exit.json` · SHA256: `ef26633723bce46b638e82c01cc8517dd907ee537e91ef108ca805614e724ee7`

<details>
<summary>batch96_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/batch96_exit.json", "sha256": "ef26633723bce46b638e82c01cc8517dd907ee537e91ef108ca805614e724ee7", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 25.051310539245605, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-f9b8e802200c

원래 경로: `dnabert2/results/batch_followup_20261028_143041_359149/summary.json` · SHA256: `f3bd49e7e15a64c345d6bef0d39721604d1827186867483273898fd88f7c10f9`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/batch_followup_20261028_143041_359149/summary.json", "sha256": "f3bd49e7e15a64c345d6bef0d39721604d1827186867483273898fd88f7c10f9", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "complete",
  "model": "dnabert2",
  "run_id": "batch_followup_20261028_143041_359149",
  "conditions": [
    {
      "batch_size": 1,
      "unique_inputs": 1,
      "processed_inputs": 359,
      "repeats": 359,
      "seconds": 15.007117986679077,
      "baseline": {
        "allocated_gib": 0.5005712509155273,
        "reserved_gib": 0.578125,
        "peak_allocated_gib": 0.5005712509155273,
        "peak_reserved_gib": 0.578125,
        "rss_gib": 1.8384170532226562,
        "system_available_gib": 111.43684768676758,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.5005712509155273,
        "reserved_gib": 0.6640625,
        "peak_allocated_gib": 0.6192150115966797,
        "peak_reserved_gib": 0.6640625,
        "rss_gib": 1.8413009643554688,
        "system_available_gib": 111.49503707885742,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.11864376068115234,
      "samples": 28,
      "gpu_util_avg": 90.71428571428571,
      "gpu_util_max": 94.0,
      "rss_sampled_peak_gib": 1.8413009643554688,
      "available_sampled_min_gib": 111.34211730957031,
      "inputs_per_second": 23.92198157691989,
      "allocated_fraction_shared_ram": 0.005088566483421198,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 2,
      "unique_inputs": 2,
      "processed_inputs": 348,
      "repeats": 174,
      "seconds": 15.002455472946167,
      "baseline": {
        "allocated_gib": 0.5005712509155273,
        "reserved_gib": 0.54296875,
        "peak_allocated_gib": 0.5005712509155273,
        "peak_reserved_gib": 0.54296875,
        "rss_gib": 1.544952392578125,
        "system_available_gib": 111.76980209350586,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.5005712509155273,
        "reserved_gib": 0.84375,
        "peak_allocated_gib": 0.733433723449707,
        "peak_reserved_gib": 0.84375,
        "rss_gib": 1.5495071411132812,
        "system_available_gib": 111.54414749145508,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.2328624725341797,
      "samples": 28,
      "gpu_util_avg": 91.71428571428571,
      "gpu_util_max": 95.0,
      "rss_sampled_peak_gib": 1.5495071411132812,
      "available_sampled_min_gib": 111.44814682006836,
      "inputs_per_second": 23.196202823434216,
      "allocated_fraction_shared_ram": 0.0060271895756104175,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 4,
      "unique_inputs": 4,
      "processed_inputs": 344,
      "repeats": 86,
      "seconds": 15.010905027389526,
      "baseline": {
        "allocated_gib": 0.5005712509155273,
        "reserved_gib": 0.578125,
        "peak_allocated_gib": 0.5005712509155273,
        "peak_reserved_gib": 0.578125,
        "rss_gib": 1.5761070251464844,
        "system_available_gib": 111.74344635009766,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.5005712509155273,
        "reserved_gib": 1.150390625,
        "peak_allocated_gib": 0.9663629531860352,
        "peak_reserved_gib": 1.150390625,
        "rss_gib": 1.5806655883789062,
        "system_available_gib": 111.2230453491211,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.4657917022705078,
      "samples": 27,
      "gpu_util_avg": 95.0,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.5806655883789062,
      "available_sampled_min_gib": 111.13162231445312,
      "inputs_per_second": 22.916672870311494,
      "allocated_fraction_shared_ram": 0.007941348388377402,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 8,
      "unique_inputs": 8,
      "processed_inputs": 344,
      "repeats": 43,
      "seconds": 15.092406272888184,
      "baseline": {
        "allocated_gib": 0.5005712509155273,
        "reserved_gib": 0.54296875,
        "peak_allocated_gib": 0.5005712509155273,
        "peak_reserved_gib": 0.54296875,
        "rss_gib": 1.5763320922851562,
        "system_available_gib": 111.83835983276367,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.5005712509155273,
        "reserved_gib": 1.513671875,
        "peak_allocated_gib": 1.4327831268310547,
        "peak_reserved_gib": 1.513671875,
        "rss_gib": 1.580841064453125,
        "system_available_gib": 110.93497085571289,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.9322118759155273,
      "samples": 28,
      "gpu_util_avg": 95.35714285714286,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.580841064453125,
      "available_sampled_min_gib": 110.81303787231445,
      "inputs_per_second": 22.792919417890136,
      "allocated_fraction_shared_ram": 0.011774282051728965,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 16,
      "unique_inputs": 16,
      "processed_inputs": 352,
      "repeats": 22,
      "seconds": 15.39372992515564,
      "baseline": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 0.556640625,
        "peak_allocated_gib": 0.4997425079345703,
        "peak_reserved_gib": 0.556640625,
        "rss_gib": 1.5762901306152344,
        "system_available_gib": 111.73517227172852,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 2.49609375,
        "peak_allocated_gib": 2.359175205230713,
        "peak_reserved_gib": 2.49609375,
        "rss_gib": 1.5779495239257812,
        "system_available_gib": 109.83547592163086,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 1.8594326972961426,
      "samples": 28,
      "gpu_util_avg": 95.28571428571429,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.5779495239257812,
      "available_sampled_min_gib": 109.75655746459961,
      "inputs_per_second": 22.86645288123314,
      "allocated_fraction_shared_ram": 0.019387158988442885,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 32,
      "unique_inputs": 32,
      "processed_inputs": 352,
      "repeats": 11,
      "seconds": 15.277253866195679,
      "baseline": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 0.556640625,
        "peak_allocated_gib": 0.4997425079345703,
        "peak_reserved_gib": 0.556640625,
        "rss_gib": 1.5787239074707031,
        "system_available_gib": 111.72293853759766,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 4.427734375,
        "peak_allocated_gib": 4.21929407119751,
        "peak_reserved_gib": 4.427734375,
        "rss_gib": 1.5803871154785156,
        "system_available_gib": 107.82423782348633,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 3.7195515632629395,
      "samples": 28,
      "gpu_util_avg": 95.39285714285714,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.5803871154785156,
      "available_sampled_min_gib": 107.73794174194336,
      "inputs_per_second": 23.040790123863705,
      "allocated_fraction_shared_ram": 0.0346731878140865,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 64,
      "unique_inputs": 64,
      "processed_inputs": 384,
      "repeats": 6,
      "seconds": 16.633729934692383,
      "baseline": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 0.634765625,
        "peak_allocated_gib": 0.4997425079345703,
        "peak_reserved_gib": 0.634765625,
        "rss_gib": 1.5785713195800781,
        "system_available_gib": 111.47253799438477,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 8.375,
        "peak_allocated_gib": 7.939034461975098,
        "peak_reserved_gib": 8.375,
        "rss_gib": 1.5811996459960938,
        "system_available_gib": 103.83219528198242,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 7.439291954040527,
      "samples": 30,
      "gpu_util_avg": 95.5,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.5816268920898438,
      "available_sampled_min_gib": 103.63642883300781,
      "inputs_per_second": 23.085621896451787,
      "allocated_fraction_shared_ram": 0.06524115843019228,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 96,
      "unique_inputs": 96,
      "processed_inputs": 384,
      "repeats": 4,
      "seconds": 16.884371280670166,
      "baseline": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 0.712890625,
        "peak_allocated_gib": 0.4997425079345703,
        "peak_reserved_gib": 0.712890625,
        "rss_gib": 1.5784873962402344,
        "system_available_gib": 111.4747200012207,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 12.326171875,
        "peak_allocated_gib": 11.658183097839355,
        "peak_reserved_gib": 12.326171875,
        "rss_gib": 1.5816192626953125,
        "system_available_gib": 99.87912368774414,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 11.158440589904785,
      "samples": 30,
      "gpu_util_avg": 89.76666666666667,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.5821380615234375,
      "available_sampled_min_gib": 99.73106384277344,
      "inputs_per_second": 22.74292560953199,
      "allocated_fraction_shared_ram": 0.0958042661405836,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 128,
      "unique_inputs": 128,
      "processed_inputs": 384,
      "repeats": 3,
      "seconds": 17.100414276123047,
      "baseline": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 0.791015625,
        "peak_allocated_gib": 0.4997425079345703,
        "peak_reserved_gib": 0.791015625,
        "rss_gib": 1.5472068786621094,
        "system_available_gib": 111.3838882446289,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 16.26953125,
        "peak_allocated_gib": 15.380672931671143,
        "peak_reserved_gib": 16.26953125,
        "rss_gib": 1.550201416015625,
        "system_available_gib": 95.81088638305664,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 14.880930423736572,
      "samples": 31,
      "gpu_util_avg": 93.90322580645162,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.550811767578125,
      "available_sampled_min_gib": 95.6985969543457,
      "inputs_per_second": 22.455596326468605,
      "allocated_fraction_shared_ram": 0.1263948310470597,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 192,
      "unique_inputs": 192,
      "processed_inputs": 384,
      "repeats": 2,
      "seconds": 17.509963750839233,
      "baseline": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 1.416015625,
        "peak_allocated_gib": 0.4997425079345703,
        "peak_reserved_gib": 1.416015625,
        "rss_gib": 1.5494651794433594,
        "system_available_gib": 110.75351333618164,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 24.630859375,
        "peak_allocated_gib": 22.81971502304077,
        "peak_reserved_gib": 24.630859375,
        "rss_gib": 1.555084228515625,
        "system_available_gib": 87.41374588012695,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 22.3199725151062,
      "samples": 31,
      "gpu_util_avg": 93.19354838709677,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.552886962890625,
      "available_sampled_min_gib": 87.29928207397461,
      "inputs_per_second": 21.930370928471813,
      "allocated_fraction_shared_ram": 0.1875271672242694,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 256,
      "unique_inputs": 256,
      "processed_inputs": 512,
      "repeats": 2,
      "seconds": 23.207858085632324,
      "baseline": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 1.728515625,
        "peak_allocated_gib": 0.4997425079345703,
        "peak_reserved_gib": 1.728515625,
        "rss_gib": 1.5702133178710938,
        "system_available_gib": 110.55771255493164,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 32.681640625,
        "peak_allocated_gib": 30.254366397857666,
        "peak_reserved_gib": 32.681640625,
        "rss_gib": 1.5760154724121094,
        "system_available_gib": 79.23941421508789,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 29.754623889923096,
      "samples": 42,
      "gpu_util_avg": 94.76190476190476,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.5760154724121094,
      "available_sampled_min_gib": 79.08007431030273,
      "inputs_per_second": 22.061493055965055,
      "allocated_fraction_shared_ram": 0.24862342150315622,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 384,
      "unique_inputs": 384,
      "processed_inputs": 768,
      "repeats": 2,
      "seconds": 34.69348669052124,
      "baseline": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 2.349609375,
        "peak_allocated_gib": 0.4997425079345703,
        "peak_reserved_gib": 2.349609375,
        "rss_gib": 1.5437698364257812,
        "system_available_gib": 109.99192810058594,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 48.765625,
        "peak_allocated_gib": 45.13145065307617,
        "peak_reserved_gib": 48.765625,
        "rss_gib": 1.5468215942382812,
        "system_available_gib": 63.02089309692383,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 44.6317081451416,
      "samples": 63,
      "gpu_util_avg": 92.06349206349206,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.5468215942382812,
      "available_sampled_min_gib": 62.92582702636719,
      "inputs_per_second": 22.13671998005979,
      "allocated_fraction_shared_ram": 0.37087987668329425,
      "model_only_allocated_gib": 0.43611812591552734
    },
    {
      "batch_size": 512,
      "unique_inputs": 512,
      "processed_inputs": 1024,
      "repeats": 2,
      "seconds": 46.59835147857666,
      "baseline": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 2.974609375,
        "peak_allocated_gib": 0.4997425079345703,
        "peak_reserved_gib": 2.974609375,
        "rss_gib": 1.5729560852050781,
        "system_available_gib": 109.38874435424805,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.4997425079345703,
        "reserved_gib": 64.857421875,
        "peak_allocated_gib": 60.007078647613525,
        "peak_reserved_gib": 64.857421875,
        "rss_gib": 1.5763740539550781,
        "system_available_gib": 46.63058853149414,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 59.507336139678955,
      "samples": 83,
      "gpu_util_avg": 92.1566265060241,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.5761299133300781,
      "available_sampled_min_gib": 46.42860412597656,
      "inputs_per_second": 21.975026315486257,
      "allocated_fraction_shared_ram": 0.49312436464824133,
      "model_only_allocated_gib": 0.43611812591552734
    }
  ],
  "failed": [],
  "input_audit": {
    "source": "existing BEND prepared n1000, fixed4000bp, actual native tokenizer length verified worker",
    "unique_available_inputs": 1000,
    "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
    "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
  },
  "largest_successful_batch": 512,
  "stop_reason": "predicted_reservation_exceeds_available_budget",
  "skipped_next_batch": {
    "batch": 768,
    "estimated_reserved_gib": 124.22606265544891,
    "available_gib": 113.97422409057617
  },
  "reached80": false,
  "reached90": false
}
```
<!-- END_RESTORE_FILE -->


</details>
