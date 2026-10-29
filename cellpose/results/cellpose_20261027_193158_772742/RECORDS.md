# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-c8c964ad5d14

원래 경로: `cellpose/results/cellpose_20261027_193158_772742/command.json` · SHA256: `fb93bcdd3e813e1c786353ac8c2e75d151c88fcb4faeb532f9187808c811b36a`

<details>
<summary>command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "cellpose/results/cellpose_20261027_193158_772742/command.json", "sha256": "fb93bcdd3e813e1c786353ac8c2e75d151c88fcb4faeb532f9187808c811b36a", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/cellpose:/work -e RUN_ID=cellpose_20261027_193158_772742 -e MPLCONFIGDIR=/work/.cache/matplotlib -e NUMBA_CACHE_DIR=/work/.cache/numba -e CELLPOSE_LOCAL_MODELS_PATH=/work/models nvcr.io/nvidia/pytorch:26.06-py3 /work/.venv/bin/python -u /work/scripts/benchmark.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-192fd39a6269

원래 경로: `cellpose/results/cellpose_20261027_193158_772742/events.json` · SHA256: `6b4fab138f02b50f16758dcaaa6960e86bbb4a78992db78cad3d43ec29d6db4c`

<details>
<summary>events.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "cellpose/results/cellpose_20261027_193158_772742/events.json", "sha256": "6b4fab138f02b50f16758dcaaa6960e86bbb4a78992db78cad3d43ec29d6db4c", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "time_unix": 1793097122.9192595,
    "phase": "loading",
    "condition": "",
    "event": "model_loaded",
    "total": 304628930,
    "requires_grad": 304592064,
    "parameter_dtypes": {
      "torch.bfloat16": 304628930
    },
    "theoretical_current_storage_gib": 0.5674155987799168,
    "memory": {
      "allocated_gib": 0.5682768821716309,
      "reserved_gib": 1.740234375,
      "peak_allocated_gib": 1.7182645797729492,
      "peak_reserved_gib": 1.740234375,
      "rss_gib": 1.0139617919921875,
      "system_available_gib": 111.8888053894043,
      "system_total_gib": 121.68751525878906,
      "swap_used_gib": 0.0
    }
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-ff0f1e2bb919

원래 경로: `cellpose/results/cellpose_20261027_193158_772742/exit_code.json` · SHA256: `b9d31493cc3c49e8ae6f8938034a925ff9cb307f0b18eab8191106a00d9f6663`

<details>
<summary>exit_code.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "cellpose/results/cellpose_20261027_193158_772742/exit_code.json", "sha256": "b9d31493cc3c49e8ae6f8938034a925ff9cb307f0b18eab8191106a00d9f6663", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 1, "finished": "2026-10-27T19:32:21.132118"}
```
<!-- END_RESTORE_FILE -->


</details>

## file-0beb66f310ec

원래 경로: `cellpose/results/cellpose_20261027_193158_772742/forward_calls.json` · SHA256: `1bcbdd503785878ea9ac981d5259863382932c134fa6ad9cc0f96f737ed76c28`

<details>
<summary>forward_calls.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "cellpose/results/cellpose_20261027_193158_772742/forward_calls.json", "sha256": "1bcbdd503785878ea9ac981d5259863382932c134fa6ad9cc0f96f737ed76c28", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "start": 1793097122.9269342,
    "end": 1793097124.3326585,
    "phase": "warmup",
    "condition": "native_one",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097124.334165,
    "end": 1793097124.5966277,
    "phase": "warmup",
    "condition": "native_one",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097124.8910854,
    "end": 1793097125.3348584,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097125.3355386,
    "end": 1793097125.5387502,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097125.5917656,
    "end": 1793097126.0159028,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097126.01639,
    "end": 1793097126.2581656,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097126.3101757,
    "end": 1793097126.7148027,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097126.715151,
    "end": 1793097126.9478383,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097126.9959497,
    "end": 1793097127.40688,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097127.4072084,
    "end": 1793097127.6156664,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097127.6633177,
    "end": 1793097128.0665214,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097128.0669398,
    "end": 1793097128.2752385,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097128.316615,
    "end": 1793097128.7164748,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097128.7168093,
    "end": 1793097128.9238825,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097128.9659142,
    "end": 1793097129.3813417,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097129.3816407,
    "end": 1793097129.58778,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097129.6293478,
    "end": 1793097130.0529764,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097130.0533106,
    "end": 1793097130.2674468,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097130.309373,
    "end": 1793097130.7180915,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097130.7184305,
    "end": 1793097130.9257932,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097130.9671836,
    "end": 1793097131.3803027,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097131.380618,
    "end": 1793097131.5819263,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097131.6238785,
    "end": 1793097132.0295706,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097132.0298748,
    "end": 1793097132.233523,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097132.2753663,
    "end": 1793097132.6775591,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097132.6778836,
    "end": 1793097132.8816752,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097132.9238873,
    "end": 1793097133.3331008,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097133.3334265,
    "end": 1793097133.5414248,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097133.5832012,
    "end": 1793097134.0133753,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097134.0137117,
    "end": 1793097134.2264292,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097134.268547,
    "end": 1793097134.6841366,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097134.6845424,
    "end": 1793097134.8897543,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097134.932177,
    "end": 1793097135.3461814,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097135.346588,
    "end": 1793097135.5479739,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097135.58989,
    "end": 1793097135.9981709,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097135.9984958,
    "end": 1793097136.204475,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097136.2461433,
    "end": 1793097136.6450741,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097136.6453898,
    "end": 1793097136.8527436,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097136.894665,
    "end": 1793097137.3072484,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097137.307553,
    "end": 1793097137.5136356,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097137.5559175,
    "end": 1793097137.987004,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097137.987325,
    "end": 1793097138.1980712,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097138.2416062,
    "end": 1793097138.657201,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097138.6575108,
    "end": 1793097138.8601081,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097138.9040775,
    "end": 1793097139.3006208,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097139.3009439,
    "end": 1793097139.5070617,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097139.550559,
    "end": 1793097139.9535677,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      8,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  },
  {
    "start": 1793097139.953974,
    "end": 1793097140.149301,
    "phase": "inference",
    "condition": "count_1",
    "shape": [
      4,
      1,
      256,
      256
    ],
    "dtype": "torch.bfloat16"
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-7d2a0c708f7d

원래 경로: `cellpose/results/cellpose_20261027_193158_772742/resources.csv` · SHA256: `9c0d029919c9734b01df340e87316bbee84540263f64ca763931df992bd6acc8`

<details>
<summary>resources.csv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "cellpose/results/cellpose_20261027_193158_772742/resources.csv", "sha256": "9c0d029919c9734b01df340e87316bbee84540263f64ca763931df992bd6acc8", "encoding": "utf-8", "added_final_newline": false} -->
```csv
query_start,query_end,phase,condition,gpu_util_percent,rss_gib,available_gib,swap_gib
1793097121.7246404,1793097121.7491193,initialization,,8.0,0.7450332641601562,114.26982498168945,0.0
1793097122.250412,1793097122.2710073,loading,,8.0,1.6089515686035156,113.25855255126953,0.0
1793097122.771752,1793097122.796098,loading,,5.0,0.9901084899902344,112.49572372436523,0.0
1793097123.2968652,1793097123.3166802,warmup,native_one,5.0,1.1525459289550781,111.83823776245117,0.0
1793097123.8171973,1793097123.8410273,warmup,native_one,7.0,1.554290771484375,110.49091720581055,0.0
1793097124.3418221,1793097124.376259,warmup,native_one,7.0,1.7337799072265625,110.31405639648438,0.0
1793097124.877062,1793097124.899673,inference,count_1,95.0,2.29510498046875,111.7072982788086,0.0
1793097125.4006362,1793097125.4367433,inference,count_1,95.0,2.3066673278808594,110.59902954101562,0.0
1793097125.937531,1793097125.9705093,inference,count_1,95.0,2.3118667602539062,110.57793426513672,0.0
1793097126.471428,1793097126.5088625,inference,count_1,95.0,2.320659637451172,110.56506729125977,0.0
1793097127.010331,1793097127.0532632,inference,count_1,95.0,2.3362083435058594,110.58036804199219,0.0
1793097127.5540395,1793097127.5869927,inference,count_1,95.0,2.3366966247558594,110.60123443603516,0.0
1793097128.087706,1793097128.1209064,inference,count_1,95.0,2.3410720825195312,110.58334732055664,0.0
1793097128.621647,1793097128.6602833,inference,count_1,95.0,2.3410720825195312,110.58938598632812,0.0
1793097129.1610265,1793097129.200703,inference,count_1,95.0,2.3410720825195312,110.58152770996094,0.0
1793097129.7011976,1793097129.7359414,inference,count_1,95.0,2.3410720825195312,110.58261108398438,0.0
1793097130.236294,1793097130.2701821,inference,count_1,95.0,2.3410720825195312,110.56685256958008,0.0
1793097130.770711,1793097130.8057752,inference,count_1,95.0,2.3410720825195312,110.61671829223633,0.0
1793097131.306394,1793097131.351348,inference,count_1,95.0,2.3410720825195312,110.61314010620117,0.0
1793097131.8521314,1793097131.8841238,inference,count_1,95.0,2.3410720825195312,110.61690902709961,0.0
1793097132.3851717,1793097132.4224184,inference,count_1,95.0,2.34326171875,110.60504913330078,0.0
1793097132.9234512,1793097132.958585,inference,count_1,95.0,2.34326171875,110.63092803955078,0.0
1793097133.4590504,1793097133.4977686,inference,count_1,95.0,2.34326171875,110.63552856445312,0.0
1793097133.998531,1793097134.0367224,inference,count_1,95.0,2.34326171875,110.6463623046875,0.0
1793097134.5376184,1793097134.5811834,inference,count_1,95.0,2.34326171875,110.64558792114258,0.0
1793097135.0819666,1793097135.1181493,inference,count_1,95.0,2.34326171875,110.65660095214844,0.0
1793097135.6192315,1793097135.659945,inference,count_1,95.0,2.34326171875,110.6468734741211,0.0
1793097136.1606526,1793097136.1935687,inference,count_1,95.0,2.34326171875,110.64909744262695,0.0
1793097136.6941454,1793097136.7263532,inference,count_1,95.0,2.34326171875,110.65600967407227,0.0
1793097137.2272062,1793097137.263339,inference,count_1,95.0,2.3432655334472656,110.6552505493164,0.0
1793097137.764568,1793097137.8005795,inference,count_1,85.0,2.3432655334472656,110.70174789428711,0.0
1793097138.3011878,1793097138.3364367,inference,count_1,85.0,2.3432655334472656,110.70531845092773,0.0
1793097138.8373103,1793097138.867881,inference,count_1,95.0,2.3432655334472656,110.70582962036133,0.0
1793097139.3686185,1793097139.4047964,inference,count_1,95.0,2.3432655334472656,110.7054443359375,0.0
1793097139.905349,1793097139.9502676,inference,count_1,80.0,2.3432655334472656,110.70421981811523,0.0
```
<!-- END_RESTORE_FILE -->


</details>

## file-cafd90d6f18e

원래 경로: `cellpose/results/cellpose_20261027_193158_772742/summary.json` · SHA256: `9bde3e8a4f0ba431967b5c070944429ee71ff68ffa8678de44be48bd501d6040`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "cellpose/results/cellpose_20261027_193158_772742/summary.json", "sha256": "9bde3e8a4f0ba431967b5c070944429ee71ff68ffa8678de44be48bd501d6040", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "run_id": "cellpose_20261027_193158_772742",
  "seed": 42,
  "environment": {
    "python": "3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]",
    "torch": "2.13.0a0+8145d630e8.nv26.06",
    "cuda": "13.3",
    "cellpose": "4.2.1",
    "gpu": "NVIDIA GB10",
    "platform": "Linux-7.0.0-1019-nvidia-aarch64-with-glibc2.39",
    "tf32": true
  },
  "conditions": [
    {
      "name": "count_1",
      "unique_images": 1,
      "raw_shapes": [
        [
          520,
          696
        ]
      ],
      "repeats": 23,
      "processed_images": 23,
      "seconds": 15.301878929138184,
      "images_per_second": 1.5030833864593505,
      "forward_including_transfers_seconds": 14.289727210998535,
      "tile_forward_calls": 46,
      "actual_forward_shapes": [
        "[4, 1, 256, 256]",
        "[8, 1, 256, 256]"
      ],
      "processed_tiles": 276,
      "baseline": {
        "allocated_gib": 0.5995268821716309,
        "reserved_gib": 0.642578125,
        "peak_allocated_gib": 0.5995268821716309,
        "peak_reserved_gib": 0.642578125,
        "rss_gib": 2.2950439453125,
        "system_available_gib": 111.77925872802734,
        "system_total_gib": 121.68751525878906,
        "swap_used_gib": 0.0
      },
      "memory": {
        "allocated_gib": 0.5995268821716309,
        "reserved_gib": 1.783203125,
        "peak_allocated_gib": 1.4913725852966309,
        "peak_reserved_gib": 1.783203125,
        "rss_gib": 2.3432655334472656,
        "system_available_gib": 110.70425796508789,
        "system_total_gib": 121.68751525878906,
        "swap_used_gib": 0.0
      },
      "sample_count": 28,
      "gpu_util_avg": 93.75,
      "gpu_util_max": 95.0,
      "rss_sampled_peak_gib": 2.3432655334472656,
      "available_sampled_min_gib": 110.56506729125977,
      "peak_delta_allocated_gib": 0.891845703125
    }
  ],
  "input_audit": "data/input_audit.json",
  "settings": {
    "inference_dtype": "bfloat16",
    "tile_batch_size": 8,
    "tile_size": 256,
    "tile_overlap": 0.1,
    "augment": false,
    "normalize": "official default 1st/99th percentile, grayscale + two zero channels",
    "diameter": null,
    "rescale": null,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "min_size": 15,
    "max_size_fraction": 0.4,
    "training_batch": 1,
    "training_epochs": 100,
    "learning_rate": 1e-05,
    "weight_decay": 0.1,
    "training_dtype": "float32 (official automatic conversion)",
    "training_augmentation": "official random rotate/resize, scale_range=.5, rescale=False, 256 crop",
    "seed_note": "global42; official trainer resets numpy seed to epoch index and validation42"
  },
  "before_load": {
    "allocated_gib": 0.0,
    "reserved_gib": 0.0,
    "peak_allocated_gib": 0.0,
    "peak_reserved_gib": 0.0,
    "rss_gib": 0.7740745544433594,
    "system_available_gib": 114.2389144897461,
    "system_total_gib": 121.68751525878906,
    "swap_used_gib": 0.0
  },
  "loading_seconds": 1.1581833362579346,
  "model_only_postload": {
    "allocated_gib": 0.5682768821716309,
    "reserved_gib": 1.740234375,
    "peak_allocated_gib": 1.7182645797729492,
    "peak_reserved_gib": 1.740234375,
    "rss_gib": 1.0139617919921875,
    "system_available_gib": 111.8888053894043,
    "system_total_gib": 121.68751525878906,
    "swap_used_gib": 0.0
  },
  "parameters": {
    "total": 304628930,
    "requires_grad": 304592064,
    "parameter_dtypes": {
      "torch.bfloat16": 304628930
    },
    "theoretical_current_storage_gib": 0.5674155987799168
  },
  "warmup_seconds": 1.9381251335144043,
  "error": "TypeError(\"event() got multiple values for argument 'name'\")",
  "status": "failed"
}
```
<!-- END_RESTORE_FILE -->


</details>
