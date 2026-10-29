# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-73963a404d36

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch128_command.json` · SHA256: `509e0dd2626d8714f7162a5c1c8faf9c00ffa9b73f07c77a43fd4c4ff7790912`

<details>
<summary>batch128_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch128_command.json", "sha256": "509e0dd2626d8714f7162a5c1c8faf9c00ffa9b73f07c77a43fd4c4ff7790912", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-128",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

## file-09e5d2adeb4d

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch128_exit.json` · SHA256: `6a19481bb00cc66956028d659ea26a7a9ff4579bc37772218e308ee78910a195`

<details>
<summary>batch128_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch128_exit.json", "sha256": "6a19481bb00cc66956028d659ea26a7a9ff4579bc37772218e308ee78910a195", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 23.53055214881897, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-793f93f8c8ac

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch156_command.json` · SHA256: `4f64ec299f1d44fb7fcfcbee1c57527306252f06e1db1583593651fee65c11f2`

<details>
<summary>batch156_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch156_command.json", "sha256": "4f64ec299f1d44fb7fcfcbee1c57527306252f06e1db1583593651fee65c11f2", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-156",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
  "-e",
  "BATCH=156",
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

## file-7e99857510a4

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch156_exit.json` · SHA256: `8dec0bfe2b4c7417f45757372c133bbd5cf8ab139762f5ea8f038abe8bc31e7e`

<details>
<summary>batch156_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch156_exit.json", "sha256": "8dec0bfe2b4c7417f45757372c133bbd5cf8ab139762f5ea8f038abe8bc31e7e", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 23.54096221923828, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-aecf0be5edb0

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch16_command.json` · SHA256: `f44b35a2884c74723f87fc7a37360aad440828686844fad64a5ed9cde56c3b5c`

<details>
<summary>batch16_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch16_command.json", "sha256": "f44b35a2884c74723f87fc7a37360aad440828686844fad64a5ed9cde56c3b5c", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-16",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

## file-db6f29e92d36

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch16_exit.json` · SHA256: `f9d0a8ad1d5b1bf2176096bfb56aa31d4af486d7602f63f0dbad3c4fd0adad29`

<details>
<summary>batch16_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch16_exit.json", "sha256": "f9d0a8ad1d5b1bf2176096bfb56aa31d4af486d7602f63f0dbad3c4fd0adad29", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.52858018875122, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-abdde78862ed

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch1_command.json` · SHA256: `4ab0d538c90fdeb24e73ba46a747dc85fe9bc5695f109b2fc2e0903f6c6ca367`

<details>
<summary>batch1_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch1_command.json", "sha256": "4ab0d538c90fdeb24e73ba46a747dc85fe9bc5695f109b2fc2e0903f6c6ca367", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

## file-55ff91809b61

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch1_exit.json` · SHA256: `706be3b64f7f9535ec2e16c82757421b7f828ac4100740db55dd848cb3669256`

<details>
<summary>batch1_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch1_exit.json", "sha256": "706be3b64f7f9535ec2e16c82757421b7f828ac4100740db55dd848cb3669256", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.517630100250244, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-9db12ec6d98a

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch2_command.json` · SHA256: `d5cf5779121fd9553ae002fb3ac278dde509621c105bdfced5ddb2465c58e5c0`

<details>
<summary>batch2_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch2_command.json", "sha256": "d5cf5779121fd9553ae002fb3ac278dde509621c105bdfced5ddb2465c58e5c0", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-2",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

## file-cdb3ba925566

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch2_exit.json` · SHA256: `a345831e4c5cb658138de82a7828718be773eb8ac729392519af8b2ea5d13fde`

<details>
<summary>batch2_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch2_exit.json", "sha256": "a345831e4c5cb658138de82a7828718be773eb8ac729392519af8b2ea5d13fde", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.517088651657104, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-8f32c23253c9

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch32_command.json` · SHA256: `7bac3e531e025c718400e7800a4b49d3ad90a527ac3624f7b195993e089839c1`

<details>
<summary>batch32_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch32_command.json", "sha256": "7bac3e531e025c718400e7800a4b49d3ad90a527ac3624f7b195993e089839c1", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-32",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

## file-6f20bd189ea1

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch32_exit.json` · SHA256: `13096839a7a3eaf8f3bb67b72c3f2ee574e8951505d66019da859339d18b5c75`

<details>
<summary>batch32_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch32_exit.json", "sha256": "13096839a7a3eaf8f3bb67b72c3f2ee574e8951505d66019da859339d18b5c75", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 20.03227162361145, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-e95bcbe9d3b2

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch4_command.json` · SHA256: `8cbf74294d02b62e5a522b0e58cd4b59219776cffe822801a204a26804afc005`

<details>
<summary>batch4_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch4_command.json", "sha256": "8cbf74294d02b62e5a522b0e58cd4b59219776cffe822801a204a26804afc005", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-4",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

## file-f1fa7aced049

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch4_exit.json` · SHA256: `36067e793c90bcaf0ccfff2f0c659d577f427ee57ff718dd05f41f0fa4be0a36`

<details>
<summary>batch4_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch4_exit.json", "sha256": "36067e793c90bcaf0ccfff2f0c659d577f427ee57ff718dd05f41f0fa4be0a36", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.51852512359619, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-12e765c2e921

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch64_command.json` · SHA256: `79b9c434ecc55a6a98a964479ff5cc96eec790e8ec44646dca00d06715f2664a`

<details>
<summary>batch64_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch64_command.json", "sha256": "79b9c434ecc55a6a98a964479ff5cc96eec790e8ec44646dca00d06715f2664a", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-64",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

## file-12eb77de8a90

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch64_exit.json` · SHA256: `177efc247b90ba7c72baf0c1ed9643476115f7b5fa72eee7f8663d77b9d2ab2b`

<details>
<summary>batch64_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch64_exit.json", "sha256": "177efc247b90ba7c72baf0c1ed9643476115f7b5fa72eee7f8663d77b9d2ab2b", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 21.527767181396484, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-a561c102fa27

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch8_command.json` · SHA256: `be04a54132547eff58f545302cfc9f876bb90d4ddfd4ec151a777f324b09dd14`

<details>
<summary>batch8_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch8_command.json", "sha256": "be04a54132547eff58f545302cfc9f876bb90d4ddfd4ec151a777f324b09dd14", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-8",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

## file-e2d3dac6c7cd

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch8_exit.json` · SHA256: `73a63811e1422cb4088baf54097d294f5c4e5b02868310502df222f7ee93a1ef`

<details>
<summary>batch8_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch8_exit.json", "sha256": "73a63811e1422cb4088baf54097d294f5c4e5b02868310502df222f7ee93a1ef", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.53093719482422, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-0db4a85fc2bb

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch96_command.json` · SHA256: `aab03c1e4834fc673a8bb460c6a53a6ecccdaf06b43ce9b944e97a3ad79d7288`

<details>
<summary>batch96_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch96_command.json", "sha256": "aab03c1e4834fc673a8bb460c6a53a6ecccdaf06b43ce9b944e97a3ad79d7288", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "followup-openspliceai-batch_followup_20261028_144737_167765-96",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/openspliceai:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=batch_followup_20261028_144737_167765",
  "-e",
  "MODEL_KIND=openspliceai",
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

## file-ebaed3da7d3a

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/batch96_exit.json` · SHA256: `85a14e329212ea2539fc3c76e2813007bbad925869e147ed72c3f4919a21a12a`

<details>
<summary>batch96_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/batch96_exit.json", "sha256": "85a14e329212ea2539fc3c76e2813007bbad925869e147ed72c3f4919a21a12a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 23.04099130630493, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-96bac6b13614

원래 경로: `openspliceai/results/batch_followup_20261028_144737_167765/summary.json` · SHA256: `5b4dc11d3a389e99de2da11c1c7c770b9baeeef02ee6206d95aeb195b4832b36`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/results/batch_followup_20261028_144737_167765/summary.json", "sha256": "5b4dc11d3a389e99de2da11c1c7c770b9baeeef02ee6206d95aeb195b4832b36", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "complete",
  "model": "openspliceai",
  "run_id": "batch_followup_20261028_144737_167765",
  "conditions": [
    {
      "batch_size": 1,
      "unique_inputs": 1,
      "processed_inputs": 791,
      "repeats": 791,
      "seconds": 15.017197132110596,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.0833816528320312,
        "system_available_gib": 112.88924026489258,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.03515625,
        "peak_allocated_gib": 0.022668838500976562,
        "peak_reserved_gib": 0.03515625,
        "rss_gib": 1.0836563110351562,
        "system_available_gib": 112.97844314575195,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.009276866912841797,
      "samples": 29,
      "gpu_util_avg": 92.3103448275862,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.0835723876953125,
      "available_sampled_min_gib": 112.86565399169922,
      "inputs_per_second": 52.672945093638035,
      "allocated_fraction_shared_ram": 0.0001862872986827568,
      "model_only_allocated_gib": 0.013391971588134766
    },
    {
      "batch_size": 2,
      "unique_inputs": 2,
      "processed_inputs": 756,
      "repeats": 378,
      "seconds": 15.019516229629517,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.0860481262207031,
        "system_available_gib": 112.966552734375,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.03515625,
        "peak_allocated_gib": 0.03194475173950195,
        "peak_reserved_gib": 0.03515625,
        "rss_gib": 1.0855712890625,
        "system_available_gib": 112.88580322265625,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.018552780151367188,
      "samples": 28,
      "gpu_util_avg": 92.10714285714286,
      "gpu_util_max": 94.0,
      "rss_sampled_peak_gib": 1.0862617492675781,
      "available_sampled_min_gib": 112.86729431152344,
      "inputs_per_second": 50.3345106754246,
      "allocated_fraction_shared_ram": 0.00026251461928174013,
      "model_only_allocated_gib": 0.013391971588134766
    },
    {
      "batch_size": 4,
      "unique_inputs": 4,
      "processed_inputs": 716,
      "repeats": 179,
      "seconds": 15.001300811767578,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.0788536071777344,
        "system_available_gib": 112.98145294189453,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.076171875,
        "peak_allocated_gib": 0.05049610137939453,
        "peak_reserved_gib": 0.076171875,
        "rss_gib": 1.0790901184082031,
        "system_available_gib": 112.95174026489258,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.037104129791259766,
      "samples": 28,
      "gpu_util_avg": 90.14285714285714,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.0790786743164062,
      "available_sampled_min_gib": 112.89942169189453,
      "inputs_per_second": 47.72919422016676,
      "allocated_fraction_shared_ram": 0.0004149653419416613,
      "model_only_allocated_gib": 0.013391971588134766
    },
    {
      "batch_size": 8,
      "unique_inputs": 8,
      "processed_inputs": 664,
      "repeats": 83,
      "seconds": 15.021369934082031,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.0860748291015625,
        "system_available_gib": 113.02488708496094,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.11328125,
        "peak_allocated_gib": 0.0876002311706543,
        "peak_reserved_gib": 0.11328125,
        "rss_gib": 1.0849685668945312,
        "system_available_gib": 112.8900375366211,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.07420825958251953,
      "samples": 28,
      "gpu_util_avg": 93.35714285714286,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.086181640625,
      "available_sampled_min_gib": 112.86271286010742,
      "inputs_per_second": 44.2036913353321,
      "allocated_fraction_shared_ram": 0.0007198785428756401,
      "model_only_allocated_gib": 0.013391971588134766
    },
    {
      "batch_size": 16,
      "unique_inputs": 16,
      "processed_inputs": 576,
      "repeats": 36,
      "seconds": 15.010629653930664,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.0863037109375,
        "system_available_gib": 113.03339004516602,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.18359375,
        "peak_allocated_gib": 0.16524076461791992,
        "peak_reserved_gib": 0.18359375,
        "rss_gib": 1.0842628479003906,
        "system_available_gib": 112.85663986206055,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.15184879302978516,
      "samples": 28,
      "gpu_util_avg": 91.92857142857143,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.0865287780761719,
      "available_sampled_min_gib": 112.838134765625,
      "inputs_per_second": 38.37280735583063,
      "allocated_fraction_shared_ram": 0.0013579105815950594,
      "model_only_allocated_gib": 0.013391971588134766
    },
    {
      "batch_size": 32,
      "unique_inputs": 32,
      "processed_inputs": 608,
      "repeats": 19,
      "seconds": 15.819069385528564,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.0830154418945312,
        "system_available_gib": 112.92901611328125,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.326171875,
        "peak_allocated_gib": 0.3102231025695801,
        "peak_reserved_gib": 0.326171875,
        "rss_gib": 1.08795166015625,
        "system_available_gib": 112.60932922363281,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.2968311309814453,
      "samples": 30,
      "gpu_util_avg": 94.96666666666667,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.0909538269042969,
      "available_sampled_min_gib": 112.52041625976562,
      "inputs_per_second": 38.43462502011681,
      "allocated_fraction_shared_ram": 0.002549342074327331,
      "model_only_allocated_gib": 0.013391971588134766
    },
    {
      "batch_size": 64,
      "unique_inputs": 64,
      "processed_inputs": 640,
      "repeats": 10,
      "seconds": 16.665730714797974,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.0834693908691406,
        "system_available_gib": 113.00634002685547,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.625,
        "peak_allocated_gib": 0.6110215187072754,
        "peak_reserved_gib": 0.625,
        "rss_gib": 1.0948600769042969,
        "system_available_gib": 112.3832893371582,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.5976295471191406,
      "samples": 31,
      "gpu_util_avg": 93.2258064516129,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.0957565307617188,
      "available_sampled_min_gib": 112.34719467163086,
      "inputs_per_second": 38.402156554211324,
      "allocated_fraction_shared_ram": 0.0050212342441854855,
      "model_only_allocated_gib": 0.013391971588134766
    },
    {
      "batch_size": 96,
      "unique_inputs": 96,
      "processed_inputs": 672,
      "repeats": 7,
      "seconds": 17.364341259002686,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.083953857421875,
        "system_available_gib": 112.97438430786133,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.935546875,
        "peak_allocated_gib": 0.9049801826477051,
        "peak_reserved_gib": 0.935546875,
        "rss_gib": 1.0908737182617188,
        "system_available_gib": 112.00393676757812,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.8915882110595703,
      "samples": 33,
      "gpu_util_avg": 88.57575757575758,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.0963668823242188,
      "available_sampled_min_gib": 111.98659133911133,
      "inputs_per_second": 38.69999961280397,
      "allocated_fraction_shared_ram": 0.007436918904319081,
      "model_only_allocated_gib": 0.013391971588134766
    },
    {
      "batch_size": 128,
      "unique_inputs": 128,
      "processed_inputs": 640,
      "repeats": 5,
      "seconds": 16.63230800628662,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.0843772888183594,
        "system_available_gib": 112.95397186279297,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 1.244140625,
        "peak_allocated_gib": 1.202089786529541,
        "peak_reserved_gib": 1.244140625,
        "rss_gib": 1.0936965942382812,
        "system_available_gib": 111.70052719116211,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 1.1886978149414062,
      "samples": 31,
      "gpu_util_avg": 93.3225806451613,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.1011199951171875,
      "available_sampled_min_gib": 111.66407775878906,
      "inputs_per_second": 38.479325885384945,
      "allocated_fraction_shared_ram": 0.009878497263857299,
      "model_only_allocated_gib": 0.013391971588134766
    },
    {
      "batch_size": 156,
      "unique_inputs": 156,
      "processed_inputs": 624,
      "repeats": 4,
      "seconds": 16.360412120819092,
      "baseline": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 0.013671875,
        "peak_allocated_gib": 0.013391971588134766,
        "peak_reserved_gib": 0.013671875,
        "rss_gib": 1.0848236083984375,
        "system_available_gib": 112.923828125,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.013391971588134766,
        "reserved_gib": 1.5,
        "peak_allocated_gib": 1.4627537727355957,
        "peak_reserved_gib": 1.5,
        "rss_gib": 1.1072425842285156,
        "system_available_gib": 111.45221710205078,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 1.449361801147461,
      "samples": 31,
      "gpu_util_avg": 95.64516129032258,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.107147216796875,
      "available_sampled_min_gib": 111.41990661621094,
      "inputs_per_second": 38.14084849402676,
      "allocated_fraction_shared_ram": 0.012020573923502363,
      "model_only_allocated_gib": 0.013391971588134766
    }
  ],
  "failed": [],
  "input_audit": {
    "source": "actual GRCh38 curated regions; input15000nt -> output5000nt; original five-model MANE10000 ensemble",
    "unique_available_inputs": 156,
    "selection": "nested original order, exact sequence deduplication, no synthetic copies; inference only, original split metadata retained",
    "token_policy": "fixed CPU pool-wide padding width across batch conditions, no tokenizer truncation; actual shapes in workers"
  },
  "largest_successful_batch": 156,
  "stop_reason": "available_unique_input_limit",
  "reached80": false,
  "reached90": false
}
```
<!-- END_RESTORE_FILE -->


</details>
