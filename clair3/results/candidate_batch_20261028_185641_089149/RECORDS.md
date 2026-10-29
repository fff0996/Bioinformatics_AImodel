# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-94a05e390c73

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/capture_command.json` · SHA256: `949cf824181683569eccb94e6c7158457e6a18721ece95cf8a91eed98082859b`

<details>
<summary>capture_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/capture_command.json", "sha256": "949cf824181683569eccb94e6c7158457e6a18721ece95cf8a91eed98082859b", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "/work/.venv/bin/python",
  "/work/source/run_clair3.py",
  "--bam_fn=/work/data/HG003_chr20_demo.bam",
  "--ref_fn=/work/data/GRCh38_no_alt_chr20.fa",
  "--threads=4",
  "--platform=ont",
  "--model_path=/work/models/ont",
  "--output=/work/results/candidate_batch_20261028_185641_089149/calls",
  "--bed_fn=/work/data/region_200000.bed",
  "--python=/work/.venv/bin/python",
  "--pypy=/work/.venv/bin/python",
  "--whatshap=/work/.venv/bin/whatshap",
  "--device=cuda:0",
  "--use_gpu"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-2d22729369c0

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/capture_exit.json` · SHA256: `27a00348126d43b1cd1a97bbfe79bea21508f5f00ddf54b1a8fca9d110f781f5`

<details>
<summary>capture_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/capture_exit.json", "sha256": "27a00348126d43b1cd1a97bbfe79bea21508f5f00ddf54b1a8fca9d110f781f5", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 19.57185697555542}
```
<!-- END_RESTORE_FILE -->


</details>

## file-6ff66c650b61

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-128_command.json` · SHA256: `1d8572f5403c8f3617ad7c92e57d49c31fc91066234d7e3a2a5aba68c99aa6c6`

<details>
<summary>clair3_f-128_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-128_command.json", "sha256": "1d8572f5403c8f3617ad7c92e57d49c31fc91066234d7e3a2a5aba68c99aa6c6", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-128-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=128", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-106ee2e87e96

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-128_exit.json` · SHA256: `16f0fe9d5555782d17ecf829bb3846797a796e6c0c467954648df726b100cf4a`

<details>
<summary>clair3_f-128_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-128_exit.json", "sha256": "16f0fe9d5555782d17ecf829bb3846797a796e6c0c467954648df726b100cf4a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.528987407684326, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-814f36b7357d

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-1_command.json` · SHA256: `1d96bcc893c60356ec312399349c3043191e3f1829dc149739eee6406b4a1dc3`

<details>
<summary>clair3_f-1_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-1_command.json", "sha256": "1d96bcc893c60356ec312399349c3043191e3f1829dc149739eee6406b4a1dc3", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-1-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=1", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-ca2c1a7d5eed

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-1_exit.json` · SHA256: `e339a3e8de3887b8728c2ced6b2b8cf73237d7ac606314f5cb2a46b2b252af1d`

<details>
<summary>clair3_f-1_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-1_exit.json", "sha256": "e339a3e8de3887b8728c2ced6b2b8cf73237d7ac606314f5cb2a46b2b252af1d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.025683879852295, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-74aac22b890a

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-32_command.json` · SHA256: `c8ecf0b51fdf8170f588a59a4975837b33bdd2804ec6f3fcc7e38b6ec73a999b`

<details>
<summary>clair3_f-32_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-32_command.json", "sha256": "c8ecf0b51fdf8170f588a59a4975837b33bdd2804ec6f3fcc7e38b6ec73a999b", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-32-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=32", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-4bd91bc7ca45

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-32_exit.json` · SHA256: `ad500b939a33016ec77ebac2f327b28308ac1987bad62b2d4203fdff730e1e70`

<details>
<summary>clair3_f-32_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-32_exit.json", "sha256": "ad500b939a33016ec77ebac2f327b28308ac1987bad62b2d4203fdff730e1e70", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.030561685562134, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-881edfd6a0b0

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-512_command.json` · SHA256: `62d319b5ce384ab7a8ac2d7814224cc9335ff26c53d2ecf956def8f245ba219d`

<details>
<summary>clair3_f-512_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-512_command.json", "sha256": "62d319b5ce384ab7a8ac2d7814224cc9335ff26c53d2ecf956def8f245ba219d", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-512-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=512", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-f1cddcc016d3

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-512_exit.json` · SHA256: `6a201a9d11551b84c9182c2df5b147fd07db7f8aa001de05a85b5bc4cb7c6d0b`

<details>
<summary>clair3_f-512_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-512_exit.json", "sha256": "6a201a9d11551b84c9182c2df5b147fd07db7f8aa001de05a85b5bc4cb7c6d0b", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.531967639923096, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-af7acbf1b9a8

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-894_command.json` · SHA256: `5c86cea7c2039efe3b1ea1dff47af9993ca9c0c6bb604faf0406d84cc104f76f`

<details>
<summary>clair3_f-894_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-894_command.json", "sha256": "5c86cea7c2039efe3b1ea1dff47af9993ca9c0c6bb604faf0406d84cc104f76f", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-894-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=894", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-a5d6ddf103e9

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-894_exit.json` · SHA256: `f87f203debc57d8461e1c7fe96101c956639d094645bde398c9b67699d225a09`

<details>
<summary>clair3_f-894_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-894_exit.json", "sha256": "f87f203debc57d8461e1c7fe96101c956639d094645bde398c9b67699d225a09", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.532811880111694, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-bf343cc19e42

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-8_command.json` · SHA256: `6ba5f550df8017c782834c7ce3588199e51d4e1ed031f22acb48e5c8b96c190f`

<details>
<summary>clair3_f-8_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-8_command.json", "sha256": "6ba5f550df8017c782834c7ce3588199e51d4e1ed031f22acb48e5c8b96c190f", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_f-8-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_F", "-e", "BATCH=8", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-491240a9ffc4

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_f-8_exit.json` · SHA256: `ad452bcc89e0735488fd65a47a49d582267e921cd81ddf00dddc8e55e56fc52d`

<details>
<summary>clair3_f-8_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-8_exit.json", "sha256": "ad452bcc89e0735488fd65a47a49d582267e921cd81ddf00dddc8e55e56fc52d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.024916648864746, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-01beba8af330

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1000_command.json` · SHA256: `368e91ae82641349cb30f2671c85010d6677fb592c0f18cedca44d73c1590f28`

<details>
<summary>clair3_p-1000_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1000_command.json", "sha256": "368e91ae82641349cb30f2671c85010d6677fb592c0f18cedca44d73c1590f28", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-1000-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=1000", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-e2dba5cd1e37

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1000_exit.json` · SHA256: `aa1a0842d63b1f1a2630560972037753bcb55c5830e021b11e5a9fa095ddb4d3`

<details>
<summary>clair3_p-1000_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1000_exit.json", "sha256": "aa1a0842d63b1f1a2630560972037753bcb55c5830e021b11e5a9fa095ddb4d3", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.532365083694458, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-e2a28053273d

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-128_command.json` · SHA256: `baba4ff65e2e6872eb49be7b1139dd73af8a130f24fef43b0f6af8b90abe9afe`

<details>
<summary>clair3_p-128_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-128_command.json", "sha256": "baba4ff65e2e6872eb49be7b1139dd73af8a130f24fef43b0f6af8b90abe9afe", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-128-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=128", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-5ea62d723b94

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-128_exit.json` · SHA256: `6e4cfe2ec79e1bff9b71a6de93ae0e4d56f4bdfc04a78a9c8c5812a120e63ba8`

<details>
<summary>clair3_p-128_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-128_exit.json", "sha256": "6e4cfe2ec79e1bff9b71a6de93ae0e4d56f4bdfc04a78a9c8c5812a120e63ba8", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.028083086013794, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-106e7c500508

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1_command.json` · SHA256: `cabadd14ce2deb154356b248cdc0ce49a687f0ee963243e5c5840aba87f080af`

<details>
<summary>clair3_p-1_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1_command.json", "sha256": "cabadd14ce2deb154356b248cdc0ce49a687f0ee963243e5c5840aba87f080af", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-1-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=1", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-bde4899fc836

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1_exit.json` · SHA256: `aba9d0bcdb5b6703a2589fc7674577788849f69f536bb7e8ab3912b7f18b3ef6`

<details>
<summary>clair3_p-1_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1_exit.json", "sha256": "aba9d0bcdb5b6703a2589fc7674577788849f69f536bb7e8ab3912b7f18b3ef6", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.0213360786438, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-4a220d77dfe6

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-2048_command.json` · SHA256: `18fa2b832c2cff24e9054d1c11ccce292f1aeec0ce26596513ae6631ccf4c181`

<details>
<summary>clair3_p-2048_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-2048_command.json", "sha256": "18fa2b832c2cff24e9054d1c11ccce292f1aeec0ce26596513ae6631ccf4c181", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-2048-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=2048", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-de3d70e1d59b

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-2048_exit.json` · SHA256: `e4201bb4a65e82c3ae454fd2b85d952b6ec509a71fb931aced5449cad34d9f0a`

<details>
<summary>clair3_p-2048_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-2048_exit.json", "sha256": "e4201bb4a65e82c3ae454fd2b85d952b6ec509a71fb931aced5449cad34d9f0a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.52994394302368, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-5f125c013de2

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-32_command.json` · SHA256: `9f2403182aef7010b37f05c7feb27e23f2aa00a3d5e3e79b963b10f5bb4ea783`

<details>
<summary>clair3_p-32_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-32_command.json", "sha256": "9f2403182aef7010b37f05c7feb27e23f2aa00a3d5e3e79b963b10f5bb4ea783", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-32-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=32", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-22ad4b9bc111

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-32_exit.json` · SHA256: `38cc59f394ecdb4282036c023ea6ee2c0678aec62c0a1473726b681a7bbfd0fa`

<details>
<summary>clair3_p-32_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-32_exit.json", "sha256": "38cc59f394ecdb4282036c023ea6ee2c0678aec62c0a1473726b681a7bbfd0fa", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.026323318481445, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-1ed92db4fc50

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-4096_command.json` · SHA256: `eebeeb98d0d87efd0970092ede2b8b3a3aa461488b77f821a49e970e6c3c9d56`

<details>
<summary>clair3_p-4096_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-4096_command.json", "sha256": "eebeeb98d0d87efd0970092ede2b8b3a3aa461488b77f821a49e970e6c3c9d56", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-4096-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=4096", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-8f4735fa9446

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-4096_exit.json` · SHA256: `a2e76577de2a1486c5ac876a3721a0418a58f8a023dd2966b73e2bed2d7bd6d4`

<details>
<summary>clair3_p-4096_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-4096_exit.json", "sha256": "a2e76577de2a1486c5ac876a3721a0418a58f8a023dd2966b73e2bed2d7bd6d4", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 19.032165050506592, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-73df5b9a7fd4

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-512_command.json` · SHA256: `d98e82d23d095c2b49756ca5b9cd9cd7d4b73ed6048800f0fd9ffa74a3c34ad5`

<details>
<summary>clair3_p-512_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-512_command.json", "sha256": "d98e82d23d095c2b49756ca5b9cd9cd7d4b73ed6048800f0fd9ffa74a3c34ad5", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-512-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=512", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-438bb324d958

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-512_exit.json` · SHA256: `18bd1558a57ef44adc100b4b51020c7b7f60382d3ffc9c8697873ede0cfad4e9`

<details>
<summary>clair3_p-512_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-512_exit.json", "sha256": "18bd1558a57ef44adc100b4b51020c7b7f60382d3ffc9c8697873ede0cfad4e9", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.032227993011475, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-b5b62109f993

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-5708_command.json` · SHA256: `ef8c0c305cee19f98e5a6af54f8a639195d85f41d390daadfd0067c9b4cec748`

<details>
<summary>clair3_p-5708_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-5708_command.json", "sha256": "ef8c0c305cee19f98e5a6af54f8a639195d85f41d390daadfd0067c9b4cec748", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-5708-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=5708", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-51b60bfdd2d5

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-5708_exit.json` · SHA256: `fc5e6afab3476cd4fcc123cb5238ae7452b84bfd86422ba3357d99ba30581fc1`

<details>
<summary>clair3_p-5708_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-5708_exit.json", "sha256": "fc5e6afab3476cd4fcc123cb5238ae7452b84bfd86422ba3357d99ba30581fc1", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 19.02671766281128, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-3f77f1f2cb64

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-8_command.json` · SHA256: `3abecbbfcbbb88cbc8a2093c5fd60fff08d7b95a33c1b2abdc35d2cf61d72eec`

<details>
<summary>clair3_p-8_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-8_command.json", "sha256": "3abecbbfcbbb88cbc8a2093c5fd60fff08d7b95a33c1b2abdc35d2cf61d72eec", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-clair3_p-8-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "-e", "CANDIDATE_MODEL=Clair3_P", "-e", "BATCH=8", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/worker.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-2910a870ab04

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/clair3_p-8_exit.json` · SHA256: `864f6acae984581df8477c76311fa351eab2b9e54c7c5ac6f9a8908fbe5672a5`

<details>
<summary>clair3_p-8_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-8_exit.json", "sha256": "864f6acae984581df8477c76311fa351eab2b9e54c7c5ac6f9a8908fbe5672a5", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 18.023452043533325, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-6e10b0702df4

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/host_capture_command.json` · SHA256: `c7a879beaacc81e1faaddc3eb97eda0227fecd8020629d6b7c23fd17ca8d1f99`

<details>
<summary>host_capture_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/host_capture_command.json", "sha256": "c7a879beaacc81e1faaddc3eb97eda0227fecd8020629d6b7c23fd17ca8d1f99", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--gpus",
  "all",
  "--ipc=host",
  "--network",
  "none",
  "--user",
  "1000:1000",
  "-v",
  "/home/hyein/ai-benchmark/clair3:/work",
  "-v",
  "/home/hyein/ai-benchmark/scripts:/support:ro",
  "-e",
  "RUN_ID=candidate_batch_20261028_185641_089149",
  "-e",
  "PYTHONPATH=/work/scripts:/work/source",
  "-e",
  "MPLCONFIGDIR=/work/.cache/matplotlib",
  "local/clair3-spark:9d1dd3f",
  "/work/.venv/bin/python",
  "-u",
  "/work/scripts/batch_followup/capture.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-1208cced6bbf

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/input_audit.json` · SHA256: `1f0d7bc8fa275ab93e2d356ca758d15eb7adcf9f5c695c19d5e947397271c22d`

<details>
<summary>input_audit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/input_audit.json", "sha256": "1f0d7bc8fa275ab93e2d356ca758d15eb7adcf9f5c695c19d5e947397271c22d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "Clair3_P": {
    "actual_feature_tensor_rows_before_deduplication": 5717,
    "unique_feature_rows": 5708,
    "feature_shape": [
      33,
      18
    ],
    "dtype": "int8",
    "source_caches": [
      {
        "file": "Clair3_P_364_0.npz",
        "shape": [
          384,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "d51110780bb2baa652943b335daaa63feaebd3c0f6df5603d07a712e6dbb9d5d"
      },
      {
        "file": "Clair3_P_364_1.npz",
        "shape": [
          426,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "101702cc5b9774b13cf07055dabfdd16fdd0074d0bb68af6dcc7b6bbb3cdaef5"
      },
      {
        "file": "Clair3_P_364_10.npz",
        "shape": [
          528,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "e6a1ab8d7e88158279a029783af4518300fcb1c59238e0c04a64e0d6a7b228ec"
      },
      {
        "file": "Clair3_P_364_11.npz",
        "shape": [
          554,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "1276ae724fe1be162945a11d0d5d9d591969b068de77fbba00c012ffb8c27ee7"
      },
      {
        "file": "Clair3_P_364_12.npz",
        "shape": [
          587,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "ce9bdc29c5ebff159f37d64dba63a56f11e3162530b3398ab71c19afdfbcbfde"
      },
      {
        "file": "Clair3_P_364_2.npz",
        "shape": [
          398,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "90e4dc800c0c50fc5e6303e8c6ef24f18ccc8d791bc5ce99fac713558627e399"
      },
      {
        "file": "Clair3_P_364_3.npz",
        "shape": [
          403,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "bb70e0f17a2fd8891d8d1f2571242584f68f776500dc98c00b8f5b7f126a56b8"
      },
      {
        "file": "Clair3_P_364_4.npz",
        "shape": [
          468,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "c606cb062ba8dbc537fc52318aaf2ba2975fd9f82c779cad57be5ecf6f9b2f56"
      },
      {
        "file": "Clair3_P_364_5.npz",
        "shape": [
          399,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "206c94596948941d11aec2b82c2090a8886d4a9953b4eac4d85b38a9de858156"
      },
      {
        "file": "Clair3_P_364_6.npz",
        "shape": [
          392,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "22a00c0619d232b524542dc43a1a3e5450c72a84411675f8b61e12272f3c8fb6"
      },
      {
        "file": "Clair3_P_364_7.npz",
        "shape": [
          376,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "a2f70d834b2e66403fb9d3984d96883cc47699a0f7e83995ea9576bd97334063"
      },
      {
        "file": "Clair3_P_364_8.npz",
        "shape": [
          388,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "34823de50bbff28a79e96763f70ad626881d2b8fb930d83d78911b34d9423178"
      },
      {
        "file": "Clair3_P_364_9.npz",
        "shape": [
          414,
          33,
          18
        ],
        "dtype": "int8",
        "sha256": "f84e002b1fa4d1ee0a94d66bf1446548d7e0f3755ad7da984c25ad871f4d5c22"
      }
    ],
    "identity_limitation": "deduplicated exact feature tensors; genomic candidate coordinates not exported by this capture hook, no invented position IDs",
    "source": "real HG003 ONT BAM, GRCh38 chr20 100000..300000, existing official Clair3 extraction; no zero-filled or synthetic performance inputs"
  },
  "Clair3_F": {
    "actual_feature_tensor_rows_before_deduplication": 894,
    "unique_feature_rows": 894,
    "feature_shape": [
      89,
      33,
      8
    ],
    "dtype": "int8",
    "source_caches": [
      {
        "file": "Clair3_F_562_0.npz",
        "shape": [
          894,
          89,
          33,
          8
        ],
        "dtype": "int8",
        "sha256": "fcb088913ed72ce7479b546432381f80948ba4da186e109c580fee135d6fa872"
      }
    ],
    "identity_limitation": "deduplicated exact feature tensors; genomic candidate coordinates not exported by this capture hook, no invented position IDs",
    "source": "real HG003 ONT BAM, GRCh38 chr20 100000..300000, existing official Clair3 extraction; no zero-filled or synthetic performance inputs"
  }
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-0e127dce7b4b

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/native_pipeline_candidate_equivalence.json` · SHA256: `a789c7891a732bf03801fc25ed5648498c24c4128dc650b027c42371c66d996a`

<details>
<summary>native_pipeline_candidate_equivalence.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/native_pipeline_candidate_equivalence.json", "sha256": "a789c7891a732bf03801fc25ed5648498c24c4128dc650b027c42371c66d996a", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "model": "Clair3_P",
    "original_actual_candidate_input_exact": true,
    "probability_max_abs_vs_native_pipeline": 4.172325134277344e-06,
    "passed": true
  },
  {
    "model": "Clair3_F",
    "original_actual_candidate_input_exact": true,
    "probability_max_abs_vs_native_pipeline": 1.055002212524414e-05,
    "passed": true
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-e2002103343c

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/prepare_command.json` · SHA256: `c3efc3e1993a9f254815cf3fa16ea5ca15c855060a326f7d58fdf6be2560837d`

<details>
<summary>prepare_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/prepare_command.json", "sha256": "c3efc3e1993a9f254815cf3fa16ea5ca15c855060a326f7d58fdf6be2560837d", "encoding": "utf-8", "added_final_newline": true} -->
```json
["docker", "run", "--rm", "--name", "clair3-prepare-candidate_batch_20261028_185641_089149", "--gpus", "all", "--ipc=host", "--network", "none", "--user", "1000:1000", "-v", "/home/hyein/ai-benchmark/clair3:/work", "-v", "/home/hyein/ai-benchmark/scripts:/support:ro", "-e", "RUN_ID=candidate_batch_20261028_185641_089149", "-e", "PYTHONPATH=/work/source:/work/scripts", "local/clair3-spark:9d1dd3f", "/work/.venv/bin/python", "-u", "/work/scripts/batch_followup/prepare_cache.py"]
```
<!-- END_RESTORE_FILE -->


</details>

## file-d60a464112fa

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/prepare_exit.json` · SHA256: `b3c20e7cabc6bc9769441cf7592bffe6c92f7699bfd5f615467c82ed179eba84`

<details>
<summary>prepare_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/prepare_exit.json", "sha256": "b3c20e7cabc6bc9769441cf7592bffe6c92f7699bfd5f615467c82ed179eba84", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 1.5039732456207275, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-8f543e99e509

원래 경로: `clair3/results/candidate_batch_20261028_185641_089149/summary.json` · SHA256: `0e17689c9503b10902d5bc717b82baafa87bbddb902819002bc55fd43a19ee7e`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/candidate_batch_20261028_185641_089149/summary.json", "sha256": "0e17689c9503b10902d5bc717b82baafa87bbddb902819002bc55fd43a19ee7e", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "complete",
  "run_id": "candidate_batch_20261028_185641_089149",
  "conditions": [
    {
      "batch_size": 1,
      "unique_candidates": 1,
      "repeats": 18096,
      "processed_candidates": 18096,
      "seconds": 15.000423431396484,
      "baseline": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.052734375,
        "peak_allocated_gib": 0.03897905349731445,
        "peak_reserved_gib": 0.052734375,
        "rss_gib": 0.9525146484375,
        "system_available_gib": 112.76077651977539,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.08984375,
        "peak_allocated_gib": 0.05730009078979492,
        "peak_reserved_gib": 0.08984375,
        "rss_gib": 0.9534530639648438,
        "system_available_gib": 112.7430305480957,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.01832103729248047,
      "samples": 29,
      "gpu_util_avg": 84.48275862068965,
      "gpu_util_max": 89.0,
      "rss_sampled_peak_gib": 0.9533958435058594,
      "available_sampled_min_gib": 112.69229507446289,
      "candidates_per_second": 1206.365945785527,
      "allocated_fraction_shared_ram": 0.0004708789613128068,
      "model": "Clair3_P",
      "model_only_allocated_gib": 0.007729053497314453
    },
    {
      "batch_size": 8,
      "unique_candidates": 8,
      "repeats": 16249,
      "processed_candidates": 129992,
      "seconds": 15.000823020935059,
      "baseline": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.052734375,
        "peak_allocated_gib": 0.03897905349731445,
        "peak_reserved_gib": 0.052734375,
        "rss_gib": 0.9091110229492188,
        "system_available_gib": 112.83369445800781,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.09765625,
        "peak_allocated_gib": 0.06304121017456055,
        "peak_reserved_gib": 0.09765625,
        "rss_gib": 0.909576416015625,
        "system_available_gib": 112.75636291503906,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.024062156677246094,
      "samples": 29,
      "gpu_util_avg": 86.17241379310344,
      "gpu_util_max": 90.0,
      "rss_sampled_peak_gib": 0.9095115661621094,
      "available_sampled_min_gib": 112.6187515258789,
      "candidates_per_second": 8665.657865477377,
      "allocated_fraction_shared_ram": 0.0005180581593805474,
      "model": "Clair3_P",
      "model_only_allocated_gib": 0.007729053497314453
    },
    {
      "batch_size": 32,
      "unique_candidates": 32,
      "repeats": 13518,
      "processed_candidates": 432576,
      "seconds": 15.000843524932861,
      "baseline": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.0546875,
        "peak_allocated_gib": 0.03897905349731445,
        "peak_reserved_gib": 0.0546875,
        "rss_gib": 0.9294357299804688,
        "system_available_gib": 112.76288604736328,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.126953125,
        "peak_allocated_gib": 0.08026504516601562,
        "peak_reserved_gib": 0.126953125,
        "rss_gib": 0.93572998046875,
        "system_available_gib": 112.71959686279297,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.04128599166870117,
      "samples": 29,
      "gpu_util_avg": 87.3103448275862,
      "gpu_util_max": 91.0,
      "rss_sampled_peak_gib": 0.9357032775878906,
      "available_sampled_min_gib": 112.62613296508789,
      "candidates_per_second": 28836.778363897778,
      "allocated_fraction_shared_ram": 0.000659599672121815,
      "model": "Clair3_P",
      "model_only_allocated_gib": 0.007729053497314453
    },
    {
      "batch_size": 128,
      "unique_candidates": 128,
      "repeats": 1345,
      "processed_candidates": 172160,
      "seconds": 15.007113456726074,
      "baseline": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.099609375,
        "peak_allocated_gib": 0.03897905349731445,
        "peak_reserved_gib": 0.099609375,
        "rss_gib": 0.9448699951171875,
        "system_available_gib": 112.78902053833008,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.27734375,
        "peak_allocated_gib": 0.14751100540161133,
        "peak_reserved_gib": 0.27734375,
        "rss_gib": 0.9457206726074219,
        "system_available_gib": 112.58818817138672,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.10853195190429688,
      "samples": 28,
      "gpu_util_avg": 89.14285714285714,
      "gpu_util_max": 95.0,
      "rss_sampled_peak_gib": 0.9457130432128906,
      "available_sampled_min_gib": 112.57790756225586,
      "candidates_per_second": 11471.893012366025,
      "allocated_fraction_shared_ram": 0.0012122114999875233,
      "model": "Clair3_P",
      "model_only_allocated_gib": 0.007729053497314453
    },
    {
      "batch_size": 512,
      "unique_candidates": 512,
      "repeats": 293,
      "processed_candidates": 150016,
      "seconds": 15.010501384735107,
      "baseline": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.28515625,
        "peak_allocated_gib": 0.03897905349731445,
        "peak_reserved_gib": 0.28515625,
        "rss_gib": 0.9453620910644531,
        "system_available_gib": 112.51150894165039,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.89453125,
        "peak_allocated_gib": 0.4207320213317871,
        "peak_reserved_gib": 0.89453125,
        "rss_gib": 0.94622802734375,
        "system_available_gib": 111.94770050048828,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.38175296783447266,
      "samples": 28,
      "gpu_util_avg": 92.35714285714286,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 0.9462242126464844,
      "available_sampled_min_gib": 111.87937927246094,
      "candidates_per_second": 9994.069895130779,
      "allocated_fraction_shared_ram": 0.0034574789405226116,
      "model": "Clair3_P",
      "model_only_allocated_gib": 0.007729053497314453
    },
    {
      "batch_size": 1000,
      "unique_candidates": 1000,
      "repeats": 147,
      "processed_candidates": 147000,
      "seconds": 15.103230714797974,
      "baseline": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.0546875,
        "peak_allocated_gib": 0.03897905349731445,
        "peak_reserved_gib": 0.0546875,
        "rss_gib": 0.9311027526855469,
        "system_available_gib": 112.7935676574707,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 1.2421875,
        "peak_allocated_gib": 0.7687101364135742,
        "peak_reserved_gib": 1.2421875,
        "rss_gib": 0.9317626953125,
        "system_available_gib": 111.5389518737793,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.7297310829162598,
      "samples": 27,
      "gpu_util_avg": 91.29629629629629,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 0.9317512512207031,
      "available_sampled_min_gib": 111.50748443603516,
      "candidates_per_second": 9733.016913789914,
      "allocated_fraction_shared_ram": 0.00631708302021602,
      "model": "Clair3_P",
      "model_only_allocated_gib": 0.007729053497314453
    },
    {
      "batch_size": 2048,
      "unique_candidates": 2048,
      "repeats": 72,
      "processed_candidates": 147456,
      "seconds": 15.031305313110352,
      "baseline": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.0859375,
        "peak_allocated_gib": 0.03897905349731445,
        "peak_reserved_gib": 0.0859375,
        "rss_gib": 0.9070892333984375,
        "system_available_gib": 112.7509994506836,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 2.474609375,
        "peak_allocated_gib": 1.5113496780395508,
        "peak_reserved_gib": 2.474609375,
        "rss_gib": 0.9113960266113281,
        "system_available_gib": 110.36069869995117,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 1.4723706245422363,
      "samples": 27,
      "gpu_util_avg": 89.29629629629629,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 0.9128875732421875,
      "available_sampled_min_gib": 110.31960678100586,
      "candidates_per_second": 9809.926478666388,
      "allocated_fraction_shared_ram": 0.012419923891332733,
      "model": "Clair3_P",
      "model_only_allocated_gib": 0.007729053497314453
    },
    {
      "batch_size": 4096,
      "unique_candidates": 4096,
      "repeats": 36,
      "processed_candidates": 147456,
      "seconds": 15.201398134231567,
      "baseline": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.150390625,
        "peak_allocated_gib": 0.03897905349731445,
        "peak_reserved_gib": 0.150390625,
        "rss_gib": 0.9243507385253906,
        "system_available_gib": 112.65579986572266,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 4.8828125,
        "peak_allocated_gib": 2.9665822982788086,
        "peak_reserved_gib": 4.8828125,
        "rss_gib": 0.9137840270996094,
        "system_available_gib": 107.89741897583008,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 2.927603244781494,
      "samples": 28,
      "gpu_util_avg": 92.14285714285714,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 0.9244575500488281,
      "available_sampled_min_gib": 107.84325408935547,
      "candidates_per_second": 9700.160386428424,
      "allocated_fraction_shared_ram": 0.024378690714243527,
      "model": "Clair3_P",
      "model_only_allocated_gib": 0.007729053497314453
    },
    {
      "batch_size": 5708,
      "unique_candidates": 5708,
      "repeats": 25,
      "processed_candidates": 142700,
      "seconds": 15.1045982837677,
      "baseline": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 0.201171875,
        "peak_allocated_gib": 0.03897905349731445,
        "peak_reserved_gib": 0.201171875,
        "rss_gib": 0.9256553649902344,
        "system_available_gib": 112.51725006103516,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.03897905349731445,
        "reserved_gib": 6.783203125,
        "peak_allocated_gib": 4.1114301681518555,
        "peak_reserved_gib": 6.783203125,
        "rss_gib": 0.9138107299804688,
        "system_available_gib": 105.96167755126953,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 4.072451114654541,
      "samples": 27,
      "gpu_util_avg": 88.48148148148148,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 0.9257965087890625,
      "available_sampled_min_gib": 105.74536514282227,
      "candidates_per_second": 9447.454167209062,
      "allocated_fraction_shared_ram": 0.03378678707842957,
      "model": "Clair3_P",
      "model_only_allocated_gib": 0.007729053497314453
    },
    {
      "batch_size": 1,
      "unique_candidates": 1,
      "repeats": 7026,
      "processed_candidates": 7026,
      "seconds": 15.002003192901611,
      "baseline": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.16015625,
        "peak_allocated_gib": 0.04239511489868164,
        "peak_reserved_gib": 0.16015625,
        "rss_gib": 1.2043190002441406,
        "system_available_gib": 112.49554061889648,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.296875,
        "peak_allocated_gib": 0.17840909957885742,
        "peak_reserved_gib": 0.296875,
        "rss_gib": 1.2047119140625,
        "system_available_gib": 112.44034194946289,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.13601398468017578,
      "samples": 28,
      "gpu_util_avg": 89.85714285714286,
      "gpu_util_max": 93.0,
      "rss_sampled_peak_gib": 1.2046279907226562,
      "available_sampled_min_gib": 112.34796905517578,
      "candidates_per_second": 468.33745531559686,
      "allocated_fraction_shared_ram": 0.0014661249282594054,
      "model": "Clair3_F",
      "model_only_allocated_gib": 0.01114511489868164
    },
    {
      "batch_size": 8,
      "unique_candidates": 8,
      "repeats": 9756,
      "processed_candidates": 78048,
      "seconds": 15.000794887542725,
      "baseline": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.0546875,
        "peak_allocated_gib": 0.04239511489868164,
        "peak_reserved_gib": 0.0546875,
        "rss_gib": 1.1487770080566406,
        "system_available_gib": 112.67323684692383,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.056640625,
        "peak_allocated_gib": 0.0476222038269043,
        "peak_reserved_gib": 0.056640625,
        "rss_gib": 1.1316909790039062,
        "system_available_gib": 112.69027709960938,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.005227088928222656,
      "samples": 29,
      "gpu_util_avg": 87.0,
      "gpu_util_max": 93.0,
      "rss_sampled_peak_gib": 1.1491737365722656,
      "available_sampled_min_gib": 112.63233947753906,
      "candidates_per_second": 5202.9242840200595,
      "allocated_fraction_shared_ram": 0.0003913483131414725,
      "model": "Clair3_F",
      "model_only_allocated_gib": 0.01114511489868164
    },
    {
      "batch_size": 32,
      "unique_candidates": 32,
      "repeats": 3205,
      "processed_candidates": 102560,
      "seconds": 15.004240989685059,
      "baseline": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.0859375,
        "peak_allocated_gib": 0.04239511489868164,
        "peak_reserved_gib": 0.0859375,
        "rss_gib": 1.208953857421875,
        "system_available_gib": 112.61082077026367,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.1484375,
        "peak_allocated_gib": 0.12324237823486328,
        "peak_reserved_gib": 0.1484375,
        "rss_gib": 1.2094268798828125,
        "system_available_gib": 112.55369186401367,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.08084726333618164,
      "samples": 28,
      "gpu_util_avg": 91.75,
      "gpu_util_max": 95.0,
      "rss_sampled_peak_gib": 1.2093582153320312,
      "available_sampled_min_gib": 112.52131652832031,
      "candidates_per_second": 6835.400742397217,
      "allocated_fraction_shared_ram": 0.0010127775061621362,
      "model": "Clair3_F",
      "model_only_allocated_gib": 0.01114511489868164
    },
    {
      "batch_size": 128,
      "unique_candidates": 128,
      "repeats": 823,
      "processed_candidates": 105344,
      "seconds": 15.01783537864685,
      "baseline": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.0703125,
        "peak_allocated_gib": 0.04239511489868164,
        "peak_reserved_gib": 0.0703125,
        "rss_gib": 1.2192153930664062,
        "system_available_gib": 112.63789367675781,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.515625,
        "peak_allocated_gib": 0.3766169548034668,
        "peak_reserved_gib": 0.515625,
        "rss_gib": 1.2195816040039062,
        "system_available_gib": 112.19988632202148,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.33422183990478516,
      "samples": 28,
      "gpu_util_avg": 91.89285714285714,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.219573974609375,
      "available_sampled_min_gib": 112.14381408691406,
      "candidates_per_second": 7014.59280541746,
      "allocated_fraction_shared_ram": 0.003094951474705743,
      "model": "Clair3_F",
      "model_only_allocated_gib": 0.01114511489868164
    },
    {
      "batch_size": 512,
      "unique_candidates": 512,
      "repeats": 196,
      "processed_candidates": 100352,
      "seconds": 15.038606882095337,
      "baseline": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.068359375,
        "peak_allocated_gib": 0.04239511489868164,
        "peak_reserved_gib": 0.068359375,
        "rss_gib": 1.2353363037109375,
        "system_available_gib": 112.63330459594727,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 1.490234375,
        "peak_allocated_gib": 0.9719953536987305,
        "peak_reserved_gib": 1.490234375,
        "rss_gib": 1.2217445373535156,
        "system_available_gib": 111.18709945678711,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 0.9296002388000488,
      "samples": 28,
      "gpu_util_avg": 88.10714285714286,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.2219009399414062,
      "available_sampled_min_gib": 111.15742492675781,
      "candidates_per_second": 6672.9585251328745,
      "allocated_fraction_shared_ram": 0.007987634159770772,
      "model": "Clair3_F",
      "model_only_allocated_gib": 0.01114511489868164
    },
    {
      "batch_size": 894,
      "unique_candidates": 894,
      "repeats": 116,
      "processed_candidates": 103704,
      "seconds": 15.036135196685791,
      "baseline": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 0.103515625,
        "peak_allocated_gib": 0.04239511489868164,
        "peak_reserved_gib": 0.103515625,
        "rss_gib": 1.2515068054199219,
        "system_available_gib": 112.52983856201172,
        "system_total_gib": 121.68751525878906
      },
      "memory": {
        "allocated_gib": 0.04239511489868164,
        "reserved_gib": 2.421875,
        "peak_allocated_gib": 1.499110221862793,
        "peak_reserved_gib": 2.421875,
        "rss_gib": 1.2341423034667969,
        "system_available_gib": 110.14111709594727,
        "system_total_gib": 121.68751525878906
      },
      "peak_delta_gib": 1.4567151069641113,
      "samples": 28,
      "gpu_util_avg": 91.67857142857143,
      "gpu_util_max": 96.0,
      "rss_sampled_peak_gib": 1.2346954345703125,
      "available_sampled_min_gib": 110.13618850708008,
      "candidates_per_second": 6896.985072524358,
      "allocated_fraction_shared_ram": 0.012319342856781008,
      "model": "Clair3_F",
      "model_only_allocated_gib": 0.01114511489868164
    }
  ],
  "failed": [],
  "settings": {
    "task": "native candidate model inference only",
    "threads": 4,
    "precision": "FP32",
    "tf32": false,
    "guard_available_gib": 16,
    "guard_swap_increase_gib": 2,
    "baseline_calling_batch": 1000,
    "note": "BAM candidate extraction completed once; actual cached candidate rows reused; no artificial repeated rows within batch"
  },
  "input_audit": {
    "Clair3_P": {
      "actual_feature_tensor_rows_before_deduplication": 5717,
      "unique_feature_rows": 5708,
      "feature_shape": [
        33,
        18
      ],
      "dtype": "int8",
      "source_caches": [
        {
          "file": "Clair3_P_364_0.npz",
          "shape": [
            384,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "d51110780bb2baa652943b335daaa63feaebd3c0f6df5603d07a712e6dbb9d5d"
        },
        {
          "file": "Clair3_P_364_1.npz",
          "shape": [
            426,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "101702cc5b9774b13cf07055dabfdd16fdd0074d0bb68af6dcc7b6bbb3cdaef5"
        },
        {
          "file": "Clair3_P_364_10.npz",
          "shape": [
            528,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "e6a1ab8d7e88158279a029783af4518300fcb1c59238e0c04a64e0d6a7b228ec"
        },
        {
          "file": "Clair3_P_364_11.npz",
          "shape": [
            554,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "1276ae724fe1be162945a11d0d5d9d591969b068de77fbba00c012ffb8c27ee7"
        },
        {
          "file": "Clair3_P_364_12.npz",
          "shape": [
            587,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "ce9bdc29c5ebff159f37d64dba63a56f11e3162530b3398ab71c19afdfbcbfde"
        },
        {
          "file": "Clair3_P_364_2.npz",
          "shape": [
            398,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "90e4dc800c0c50fc5e6303e8c6ef24f18ccc8d791bc5ce99fac713558627e399"
        },
        {
          "file": "Clair3_P_364_3.npz",
          "shape": [
            403,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "bb70e0f17a2fd8891d8d1f2571242584f68f776500dc98c00b8f5b7f126a56b8"
        },
        {
          "file": "Clair3_P_364_4.npz",
          "shape": [
            468,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "c606cb062ba8dbc537fc52318aaf2ba2975fd9f82c779cad57be5ecf6f9b2f56"
        },
        {
          "file": "Clair3_P_364_5.npz",
          "shape": [
            399,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "206c94596948941d11aec2b82c2090a8886d4a9953b4eac4d85b38a9de858156"
        },
        {
          "file": "Clair3_P_364_6.npz",
          "shape": [
            392,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "22a00c0619d232b524542dc43a1a3e5450c72a84411675f8b61e12272f3c8fb6"
        },
        {
          "file": "Clair3_P_364_7.npz",
          "shape": [
            376,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "a2f70d834b2e66403fb9d3984d96883cc47699a0f7e83995ea9576bd97334063"
        },
        {
          "file": "Clair3_P_364_8.npz",
          "shape": [
            388,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "34823de50bbff28a79e96763f70ad626881d2b8fb930d83d78911b34d9423178"
        },
        {
          "file": "Clair3_P_364_9.npz",
          "shape": [
            414,
            33,
            18
          ],
          "dtype": "int8",
          "sha256": "f84e002b1fa4d1ee0a94d66bf1446548d7e0f3755ad7da984c25ad871f4d5c22"
        }
      ],
      "identity_limitation": "deduplicated exact feature tensors; genomic candidate coordinates not exported by this capture hook, no invented position IDs",
      "source": "real HG003 ONT BAM, GRCh38 chr20 100000..300000, existing official Clair3 extraction; no zero-filled or synthetic performance inputs"
    },
    "Clair3_F": {
      "actual_feature_tensor_rows_before_deduplication": 894,
      "unique_feature_rows": 894,
      "feature_shape": [
        89,
        33,
        8
      ],
      "dtype": "int8",
      "source_caches": [
        {
          "file": "Clair3_F_562_0.npz",
          "shape": [
            894,
            89,
            33,
            8
          ],
          "dtype": "int8",
          "sha256": "fcb088913ed72ce7479b546432381f80948ba4da186e109c580fee135d6fa872"
        }
      ],
      "identity_limitation": "deduplicated exact feature tensors; genomic candidate coordinates not exported by this capture hook, no invented position IDs",
      "source": "real HG003 ONT BAM, GRCh38 chr20 100000..300000, existing official Clair3 extraction; no zero-filled or synthetic performance inputs"
    }
  },
  "stop_reasons": {
    "Clair3_P": "available_real_candidate_cache_limit",
    "Clair3_F": "available_real_candidate_cache_limit"
  }
}
```
<!-- END_RESTORE_FILE -->


</details>
