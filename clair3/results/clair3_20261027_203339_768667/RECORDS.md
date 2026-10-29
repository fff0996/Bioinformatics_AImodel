# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-5d08764c1b00

원래 경로: `clair3/results/clair3_20261027_203339_768667/benchmark_command.json` · SHA256: `63b2d5d641ce3e45020d7e9f6bf523732b4b447c9da0a4f1c8a1811ce3b61887`

<details>
<summary>benchmark_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/benchmark_command.json", "sha256": "63b2d5d641ce3e45020d7e9f6bf523732b4b447c9da0a4f1c8a1811ce3b61887", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/clair3:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=clair3_20261027_203339_768667 -e PYTHONPATH=/work/scripts:/work/source -e MPLCONFIGDIR=/work/.cache/matplotlib local/clair3-spark:9d1dd3f /work/.venv/bin/python -u /work/scripts/benchmark.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-74371155a587

원래 경로: `clair3/results/clair3_20261027_203339_768667/benchmark_exit.json` · SHA256: `124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01`

<details>
<summary>benchmark_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/benchmark_exit.json", "sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0}
```
<!-- END_RESTORE_FILE -->


</details>

## file-80c9df8addf5

원래 경로: `clair3/results/clair3_20261027_203339_768667/different_batch_reload_command.json` · SHA256: `c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3`

<details>
<summary>different_batch_reload_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/different_batch_reload_command.json", "sha256": "c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/clair3:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=clair3_20261027_203339_768667 -e PYTHONPATH=/work/scripts:/work/source -e MPLCONFIGDIR=/work/.cache/matplotlib local/clair3-spark:9d1dd3f /work/.venv/bin/python -u /work/scripts/reload.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-53af65096e48

원래 경로: `clair3/results/clair3_20261027_203339_768667/different_batch_reload_exit.json` · SHA256: `2618b95f1f6be6bfde8537756f85b83845d039001563135b9d59d817446733eb`

<details>
<summary>different_batch_reload_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/different_batch_reload_exit.json", "sha256": "2618b95f1f6be6bfde8537756f85b83845d039001563135b9d59d817446733eb", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 1}
```
<!-- END_RESTORE_FILE -->


</details>

## file-f435a0fc6859

원래 경로: `clair3/results/clair3_20261027_203339_768667/reload.json` · SHA256: `aa1f550d92a6105e0024745e1b2d921b63b78651dc0441f3ee5d5628443ca863`

<details>
<summary>reload.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/reload.json", "sha256": "aa1f550d92a6105e0024745e1b2d921b63b78651dc0441f3ee5d5628443ca863", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "model": "Clair3_F",
    "actual_input_shape": [
      32,
      89,
      33,
      8
    ],
    "independent_process": true,
    "max_abs_difference": 0.0,
    "passed": true,
    "original_batch_shape": [
      181,
      89,
      33,
      8
    ],
    "note": "first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4"
  },
  {
    "model": "Clair3_P",
    "actual_input_shape": [
      32,
      33,
      18
    ],
    "independent_process": true,
    "max_abs_difference": 0.0,
    "passed": true,
    "original_batch_shape": [
      103,
      33,
      18
    ],
    "note": "first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4"
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-01b19b43542f

원래 경로: `clair3/results/clair3_20261027_203339_768667/reload_command.json` · SHA256: `c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3`

<details>
<summary>reload_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/reload_command.json", "sha256": "c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --ipc=host --network none --user 1000:1000 -v /home/hyein/ai-benchmark/clair3:/work -v /home/hyein/ai-benchmark/scripts:/support:ro -e RUN_ID=clair3_20261027_203339_768667 -e PYTHONPATH=/work/scripts:/work/source -e MPLCONFIGDIR=/work/.cache/matplotlib local/clair3-spark:9d1dd3f /work/.venv/bin/python -u /work/scripts/reload.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-735bff91ec98

원래 경로: `clair3/results/clair3_20261027_203339_768667/reload_exit.json` · SHA256: `124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01`

<details>
<summary>reload_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/reload_exit.json", "sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0}
```
<!-- END_RESTORE_FILE -->


</details>

## file-11cc0da66d78

원래 경로: `clair3/results/clair3_20261027_203339_768667/resources.csv` · SHA256: `b888d94af1da192f85a2b110588edb38b05044c3f081edc4d3d1b62fa84c1e83`

<details>
<summary>resources.csv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/resources.csv", "sha256": "b888d94af1da192f85a2b110588edb38b05044c3f081edc4d3d1b62fa84c1e83", "encoding": "utf-8", "added_final_newline": false} -->
```csv
query_start,query_end,phase,gpu_util,allocated_gib,reserved_gib,rss_gib,available_gib,swap_gib
1793100821.792909,1793100821.8114681,initial,7.0,0.0,0.0,0.5194625854492188,114.33468627929688,0.0
1793100822.3130362,1793100822.3462267,region_50000_full,4.0,0.0,0.0,0.6044387817382812,114.10403442382812,0.0
1793100822.8469722,1793100822.8658803,region_50000_full,4.0,0.0,0.0,0.6046524047851562,114.05775833129883,0.0
1793100823.3670537,1793100823.3892055,region_50000_full,7.0,0.0,0.0,0.604705810546875,113.74217987060547,0.0
1793100823.8901894,1793100823.9116025,region_50000_full,7.0,0.0,0.0,0.6047325134277344,113.50503158569336,0.0
1793100824.4125333,1793100824.4344614,region_50000_full,8.0,0.0,0.0,0.6047325134277344,113.86121368408203,0.0
1793100824.935172,1793100824.9550068,region_50000_full,8.0,0.0,0.0,0.6047325134277344,113.4230842590332,0.0
1793100825.456107,1793100825.4784262,region_50000_full,7.0,0.0,0.0,0.6047401428222656,113.82160186767578,0.0
1793100825.9792824,1793100826.001132,region_50000_full,7.0,0.0,0.0,0.6047477722167969,113.54137420654297,0.0
1793100826.5022902,1793100826.5227206,region_50000_full,6.0,0.0,0.0,0.6047859191894531,113.34702682495117,0.0
1793100827.0236151,1793100827.0426505,region_50000_full,5.0,0.0,0.0,0.6048164367675781,113.7358283996582,0.0
1793100827.543525,1793100827.5635896,region_50000_full,5.0,0.0,0.0,0.6048316955566406,113.27073669433594,0.0
1793100828.064422,1793100828.0866508,region_50000_full,7.0,0.0,0.0,0.6048316955566406,113.49285507202148,0.0
1793100828.587307,1793100828.6070626,region_50000_full,7.0,0.0,0.0,0.6048316955566406,113.21068572998047,0.0
1793100829.1079187,1793100829.129333,region_50000_full,4.0,0.0,0.0,0.6048316955566406,113.12943649291992,0.0
1793100829.6301768,1793100829.6492121,region_50000_full,4.0,0.0,0.0,0.6048431396484375,113.54788589477539,0.0
1793100830.1498194,1793100830.1687186,region_50000_full,6.0,0.0,0.0,0.6048431396484375,113.22654342651367,0.0
1793100830.669188,1793100830.6915035,region_50000_full,6.0,0.0,0.0,0.6048583984375,113.38715744018555,0.0
1793100831.1919887,1793100831.2132192,region_50000_full,7.0,0.0,0.0,0.604888916015625,113.22965240478516,0.0
1793100831.7140257,1793100831.7340636,region_50000_full,7.0,0.0,0.0,0.6049156188964844,113.18831634521484,0.0
1793100832.234852,1793100832.2554994,region_50000_full,7.0,0.0,0.0,0.6049156188964844,113.05806350708008,0.0
1793100832.7562413,1793100832.8296635,region_50000_full,7.0,0.0,0.0,0.6049156188964844,113.01465225219727,0.0
1793100833.3306859,1793100833.3714106,region_50000_full,11.0,0.0,0.0,0.60491943359375,112.09645462036133,0.0
1793100833.8721707,1793100833.8941147,region_50000_full,11.0,0.0,0.0,0.60491943359375,113.38706970214844,0.0
1793100834.3948696,1793100834.4162588,region_50000_full,6.0,0.0,0.0,0.6049232482910156,113.25873947143555,0.0
1793100834.9167142,1793100834.9355059,region_50000_full,6.0,0.0,0.0,0.6049270629882812,113.27686309814453,0.0
1793100835.4363992,1793100835.4570677,region_50000_full,5.0,0.0,0.0,0.6049308776855469,113.23141860961914,0.0
1793100835.957538,1793100835.9794738,region_50000_full,5.0,0.0,0.0,0.6049461364746094,113.17924118041992,0.0
1793100836.4806008,1793100836.5029998,region_50000_full,6.0,0.0,0.0,0.6049728393554688,113.44955444335938,0.0
1793100837.003779,1793100837.023705,region_50000_full,6.0,0.0,0.0,0.6050148010253906,113.33272933959961,0.0
1793100837.5248413,1793100837.5511203,region_50000_full,8.0,0.0,0.0,0.6050262451171875,113.2401237487793,0.0
1793100838.052155,1793100838.0711505,region_50000_full,18.0,0.0,0.0,0.6050262451171875,112.3393440246582,0.0
1793100838.5716987,1793100838.5908182,region_50000_full,18.0,0.0,0.0,0.6050262451171875,113.79098510742188,0.0
1793100839.091313,1793100839.111965,region_100000_full,6.0,0.0,0.0,0.6180953979492188,113.77774429321289,0.0
1793100839.612511,1793100839.632013,region_100000_full,6.0,0.0,0.0,0.6180953979492188,113.75415802001953,0.0
1793100840.1326246,1793100840.154694,region_100000_full,5.0,0.0,0.0,0.6180953979492188,113.28158187866211,0.0
1793100840.655175,1793100840.676983,region_100000_full,5.0,0.0,0.0,0.6180953979492188,113.50551223754883,0.0
1793100841.1775706,1793100841.1993454,region_100000_full,7.0,0.0,0.0,0.6180953979492188,113.1543960571289,0.0
1793100841.6998215,1793100841.7193363,region_100000_full,7.0,0.0,0.0,0.6180953979492188,112.98871231079102,0.0
1793100842.2198746,1793100842.2438846,region_100000_full,10.0,0.0,0.0,0.6180953979492188,113.26860809326172,0.0
1793100842.7447748,1793100842.7641108,region_100000_full,10.0,0.0,0.0,0.6180953979492188,112.82489395141602,0.0
1793100843.2646394,1793100843.286085,region_100000_full,6.0,0.0,0.0,0.6180953979492188,112.83617782592773,0.0
1793100843.7866561,1793100843.8098907,region_100000_full,6.0,0.0,0.0,0.6180953979492188,112.84054946899414,0.0
1793100844.3104656,1793100844.3301682,region_100000_full,5.0,0.0,0.0,0.6180953979492188,112.90464782714844,0.0
1793100844.8307211,1793100844.851708,region_100000_full,5.0,0.0,0.0,0.6180953979492188,112.79553985595703,0.0
1793100845.3522735,1793100845.373395,region_100000_full,6.0,0.0,0.0,0.6180953979492188,112.48673629760742,0.0
1793100845.8738651,1793100845.8953474,region_100000_full,6.0,0.0,0.0,0.6180953979492188,112.9950065612793,0.0
1793100846.3960426,1793100846.4167495,region_100000_full,7.0,0.0,0.0,0.6180953979492188,112.58740615844727,0.0
1793100846.917262,1793100846.9364142,region_100000_full,7.0,0.0,0.0,0.6180953979492188,112.8412857055664,0.0
1793100847.4368863,1793100847.4557765,region_100000_full,6.0,0.0,0.0,0.6180953979492188,112.65103149414062,0.0
1793100847.961815,1793100847.981395,region_100000_full,6.0,0.0,0.0,0.6180953979492188,112.59735870361328,0.0
1793100848.4820604,1793100848.5009086,region_100000_full,5.0,0.0,0.0,0.6180953979492188,112.64190673828125,0.0
1793100849.0014722,1793100849.0204816,region_100000_full,5.0,0.0,0.0,0.6180953979492188,112.46801376342773,0.0
1793100849.5209424,1793100849.5405264,region_100000_full,7.0,0.0,0.0,0.6180953979492188,112.11354064941406,0.0
1793100850.0411057,1793100850.0630448,region_100000_full,20.0,0.0,0.0,0.6180953979492188,111.18148040771484,0.0
1793100850.5634809,1793100850.5850196,region_100000_full,20.0,0.0,0.0,0.6180953979492188,113.22174072265625,0.0
1793100851.0856297,1793100851.1050608,region_100000_full,6.0,0.0,0.0,0.6180953979492188,113.21158218383789,0.0
1793100851.6056783,1793100851.6274483,region_100000_full,6.0,0.0,0.0,0.6180953979492188,113.22649002075195,0.0
1793100852.1281004,1793100852.1506593,region_100000_full,4.0,0.0,0.0,0.6180953979492188,113.20238876342773,0.0
1793100852.65135,1793100852.6745098,region_100000_full,4.0,0.0,0.0,0.6180953979492188,113.00073623657227,0.0
1793100853.175353,1793100853.1973014,region_100000_full,7.0,0.0,0.0,0.6180953979492188,112.98268508911133,0.0
1793100853.6982229,1793100853.7211072,region_100000_full,7.0,0.0,0.0,0.6180953979492188,113.25857925415039,0.0
1793100854.2216806,1793100854.24157,region_100000_full,9.0,0.0,0.0,0.6180953979492188,113.08549880981445,0.0
1793100854.7422945,1793100854.7639341,region_100000_full,9.0,0.0,0.0,0.6180953979492188,112.18712615966797,0.0
1793100855.2644064,1793100855.293997,region_100000_full,5.0,0.0,0.0,0.6180953979492188,112.09986114501953,0.0
1793100855.7997873,1793100855.8205295,region_100000_full,5.0,0.0,0.0,0.6255378723144531,113.54296493530273,0.0
1793100856.3213005,1793100856.3441224,region_200000_full,5.0,0.0,0.0,0.6187591552734375,113.54630279541016,0.0
1793100856.8451388,1793100856.865002,region_200000_full,5.0,0.0,0.0,0.6187591552734375,113.26305389404297,0.0
1793100857.3655922,1793100857.387915,region_200000_full,5.0,0.0,0.0,0.6187591552734375,113.02604675292969,0.0
1793100857.8887954,1793100857.9095602,region_200000_full,5.0,0.0,0.0,0.6187591552734375,113.35954284667969,0.0
1793100858.4104137,1793100858.431864,region_200000_full,8.0,0.0,0.0,0.6187591552734375,113.09206008911133,0.0
1793100858.9325056,1793100858.9524088,region_200000_full,8.0,0.0,0.0,0.6187591552734375,113.09955215454102,0.0
1793100859.4529777,1793100859.4735544,region_200000_full,4.0,0.0,0.0,0.6187591552734375,112.70495223999023,0.0
1793100859.9743109,1793100859.996595,region_200000_full,4.0,0.0,0.0,0.6187591552734375,112.6318244934082,0.0
1793100860.4973714,1793100860.5194569,region_200000_full,5.0,0.0,0.0,0.6187591552734375,112.93080520629883,0.0
1793100861.02054,1793100861.0436714,region_200000_full,5.0,0.0,0.0,0.6187591552734375,112.5153923034668,0.0
1793100861.5447962,1793100861.5674164,region_200000_full,7.0,0.0,0.0,0.6187591552734375,113.08443450927734,0.0
1793100862.068365,1793100862.0908248,region_200000_full,8.0,0.0,0.0,0.6187591552734375,112.91900253295898,0.0
1793100862.5923588,1793100862.6126137,region_200000_full,8.0,0.0,0.0,0.6187591552734375,112.8952407836914,0.0
1793100863.1137795,1793100863.134507,region_200000_full,7.0,0.0,0.0,0.6187591552734375,113.46052169799805,0.0
1793100863.6350725,1793100863.6575975,region_200000_full,7.0,0.0,0.0,0.6187591552734375,113.23172760009766,0.0
1793100864.1584415,1793100864.178476,region_200000_full,5.0,0.0,0.0,0.6187591552734375,113.23036193847656,0.0
1793100864.6794753,1793100864.702118,region_200000_full,5.0,0.0,0.0,0.6187591552734375,113.10835647583008,0.0
1793100865.20322,1793100865.2248318,region_200000_full,6.0,0.0,0.0,0.6187591552734375,113.03062438964844,0.0
1793100865.7259145,1793100865.74642,region_200000_full,6.0,0.0,0.0,0.6187591552734375,113.16769790649414,0.0
1793100866.2471426,1793100866.269401,region_200000_full,7.0,0.0,0.0,0.6187629699707031,113.1238784790039,0.0
1793100866.7711067,1793100866.7913737,region_200000_full,7.0,0.0,0.0,0.6187629699707031,112.40898132324219,0.0
1793100867.2926786,1793100867.345478,region_200000_full,88.0,0.0,0.0,0.6187629699707031,111.37965393066406,0.0
1793100867.8471036,1793100867.8682692,region_200000_full,88.0,0.0,0.0,0.6187629699707031,110.28792572021484,0.0
1793100868.369361,1793100868.389542,region_200000_full,12.0,0.0,0.0,0.6187629699707031,113.43142700195312,0.0
1793100868.8905392,1793100868.912739,region_200000_full,12.0,0.0,0.0,0.6187629699707031,113.3463020324707,0.0
1793100869.4137568,1793100869.4364085,region_200000_full,6.0,0.0,0.0,0.6187629699707031,113.34590911865234,0.0
1793100869.937649,1793100869.9585447,region_200000_full,6.0,0.0,0.0,0.6187629699707031,113.35767364501953,0.0
1793100870.4595594,1793100870.4819465,region_200000_full,8.0,0.0,0.0,0.6187667846679688,113.14227676391602,0.0
1793100870.9827285,1793100871.0030854,region_200000_full,8.0,0.0,0.0,0.6187744140625,113.16736221313477,0.0
1793100871.5038402,1793100871.5245895,region_200000_full,6.0,0.0,0.0,0.6187744140625,113.49450302124023,0.0
1793100872.0257328,1793100872.048181,region_200000_full,5.0,0.0,0.0,0.6187782287597656,113.34012603759766,0.0
1793100872.5496166,1793100872.582601,region_200000_full,5.0,0.0,0.0,0.6187782287597656,113.17886734008789,0.0
1793100873.0838196,1793100873.1053793,region_200000_full,6.0,0.0,0.0,0.6187782287597656,111.61587524414062,0.0
1793100873.606115,1793100873.6548598,region_200000_full,6.0,0.0,0.0,0.6187782287597656,113.62643051147461,0.0
1793100874.1612568,1793100874.1935205,region_200000_full,7.0,0.0,0.0,0.6317520141601562,113.60247039794922,0.0
1793100874.721058,1793100874.7404919,region_200000_half,7.0,0.0,0.0,0.6192665100097656,113.6201400756836,0.0
1793100875.241084,1793100875.2638035,region_200000_half,6.0,0.0,0.0,0.6192665100097656,113.61714553833008,0.0
1793100875.7645116,1793100875.7846136,region_200000_half,6.0,0.0,0.0,0.6192665100097656,113.19136428833008,0.0
1793100876.2853491,1793100876.308455,region_200000_half,5.0,0.0,0.0,0.6192665100097656,113.62661361694336,0.0
1793100876.8092997,1793100876.8298542,region_200000_half,5.0,0.0,0.0,0.6192665100097656,113.20734405517578,0.0
1793100877.3306577,1793100877.3528705,region_200000_half,7.0,0.0,0.0,0.6192665100097656,113.1219711303711,0.0
1793100877.8541415,1793100877.8746655,region_200000_half,7.0,0.0,0.0,0.6192665100097656,113.02897262573242,0.0
1793100878.3757288,1793100878.398423,region_200000_half,8.0,0.0,0.0,0.6192665100097656,112.91584014892578,0.0
1793100878.8992264,1793100878.919635,region_200000_half,8.0,0.0,0.0,0.6192665100097656,113.19828414916992,0.0
1793100879.4203901,1793100879.4424553,region_200000_half,6.0,0.0,0.0,0.6192665100097656,112.9594955444336,0.0
1793100879.9432926,1793100879.9645383,region_200000_half,6.0,0.0,0.0,0.6192665100097656,113.4015998840332,0.0
1793100880.4649801,1793100880.4843209,region_200000_half,4.0,0.0,0.0,0.6192665100097656,113.06871795654297,0.0
1793100880.9850512,1793100881.008172,region_200000_half,4.0,0.0,0.0,0.6192665100097656,112.96016311645508,0.0
1793100881.5092885,1793100881.5310287,region_200000_half,7.0,0.0,0.0,0.6192665100097656,113.00936889648438,0.0
1793100882.0314813,1793100882.051792,region_200000_half,8.0,0.0,0.0,0.6192665100097656,112.72863006591797,0.0
1793100882.553178,1793100882.5758789,region_200000_half,8.0,0.0,0.0,0.6192665100097656,113.03198623657227,0.0
1793100883.0766323,1793100883.097262,region_200000_half,5.0,0.0,0.0,0.6192665100097656,112.84536743164062,0.0
1793100883.5977328,1793100883.620405,region_200000_half,5.0,0.0,0.0,0.6192665100097656,112.95943832397461,0.0
1793100884.1215773,1793100884.1436148,region_200000_half,4.0,0.0,0.0,0.6192665100097656,112.94293594360352,0.0
1793100884.644791,1793100884.6881132,region_200000_half,4.0,0.0,0.0,0.6192665100097656,112.81188583374023,0.0
1793100885.1893363,1793100885.2312567,region_200000_half,58.0,0.0,0.0,0.6192741394042969,110.80221176147461,0.0
1793100885.732532,1793100885.7730937,region_200000_half,58.0,0.0,0.0,0.6192741394042969,109.82160186767578,0.0
1793100886.2748544,1793100886.2955155,region_200000_half,51.0,0.0,0.0,0.6192741394042969,109.27793884277344,0.0
1793100886.7964382,1793100886.81875,region_200000_half,51.0,0.0,0.0,0.6192741394042969,113.30151748657227,0.0
1793100887.3198586,1793100887.3394334,region_200000_half,6.0,0.0,0.0,0.6192741394042969,113.17866516113281,0.0
1793100887.8400333,1793100887.861587,region_200000_half,6.0,0.0,0.0,0.6192741394042969,113.19239044189453,0.0
1793100888.3621812,1793100888.3837934,region_200000_half,5.0,0.0,0.0,0.6192741394042969,113.09033584594727,0.0
1793100888.8842914,1793100888.9060528,region_200000_half,5.0,0.0,0.0,0.6192741394042969,112.94960021972656,0.0
1793100889.4066353,1793100889.4271317,region_200000_half,6.0,0.0,0.0,0.6192741394042969,113.24396133422852,0.0
1793100889.927601,1793100889.9471204,region_200000_half,6.0,0.0,0.0,0.6192741394042969,113.0415267944336,0.0
1793100890.4478185,1793100890.4681115,region_200000_half,10.0,0.0,0.0,0.6192779541015625,112.9831657409668,0.0
1793100890.9685714,1793100890.9910877,region_200000_half,10.0,0.0,0.0,0.6192779541015625,111.39725875854492,0.0
1793100891.491542,1793100891.5113258,region_200000_half,7.0,0.0,0.0,0.6192779541015625,113.4204330444336,0.0
1793100892.012491,1793100892.03275,region_200000_half,7.0,0.0,0.0,0.6193504333496094,113.46590805053711,0.0
```
<!-- END_RESTORE_FILE -->


</details>

## file-ccc0aa43f73b

원래 경로: `clair3/results/clair3_20261027_203339_768667/summary.json` · SHA256: `c18630e3bd08e3d956ee17a13a274f12aaad89337b565e42a6ec2d4c67494bfe`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "clair3/results/clair3_20261027_203339_768667/summary.json", "sha256": "c18630e3bd08e3d956ee17a13a274f12aaad89337b565e42a6ec2d4c67494bfe", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "complete",
  "conditions": [
    {
      "name": "region_50000_full",
      "mode": "variant_calling",
      "bp": 50000,
      "unique_inputs": 1,
      "repeats": 1,
      "processed": 50000,
      "bam": "HG003_chr20_demo.bam",
      "worker_loads": [
        {
          "time_unix": 1793100837.7206154,
          "pid": 608,
          "event": "model_only_postload",
          "model": "Clair3_F",
          "checkpoint": "/work/models/ont/full_alignment.pt",
          "total_parameters": 2986522,
          "trainable_parameters": 2986522,
          "load_seconds": 0.007873773574829102,
          "dtype": "torch.float32",
          "tf32": true,
          "memory": {
            "allocated_gib": 0.01114511489868164,
            "reserved_gib": 0.025390625,
            "peak_allocated_gib": 0.01114511489868164,
            "peak_reserved_gib": 0.025390625,
            "rss_gib": 0.6190414428710938,
            "available_gib": 113.02769470214844
          }
        },
        {
          "time_unix": 1793100832.9234774,
          "pid": 430,
          "event": "model_only_postload",
          "model": "Clair3_P",
          "checkpoint": "/work/models/ont/pileup.pt",
          "total_parameters": 2074520,
          "trainable_parameters": 2074520,
          "load_seconds": 0.004019498825073242,
          "dtype": "torch.float32",
          "tf32": true,
          "memory": {
            "allocated_gib": 0.007729053497314453,
            "reserved_gib": 0.0234375,
            "peak_allocated_gib": 0.007729053497314453,
            "peak_reserved_gib": 0.0234375,
            "rss_gib": 0.698028564453125,
            "available_gib": 112.74616241455078
          }
        }
      ],
      "actual_input_shapes": [
        "[102, 33, 18]",
        "[103, 33, 18]",
        "[105, 33, 18]",
        "[108, 33, 18]",
        "[126, 33, 18]",
        "[181, 89, 33, 8]",
        "[89, 33, 18]",
        "[90, 33, 18]",
        "[92, 33, 18]",
        "[93, 33, 18]",
        "[97, 33, 18]"
      ],
      "forward_calls": 14,
      "forward_seconds": 0.7019891738891602,
      "candidate_tensors": 1486,
      "seconds": 16.474541425704956,
      "baseline": {
        "allocated_gib": 0.01114511489868164,
        "reserved_gib": 0.025390625,
        "peak_allocated_gib": 0.01114511489868164,
        "peak_reserved_gib": 0.025390625,
        "rss_gib": 0.6190414428710938,
        "available_gib": 113.02769470214844
      },
      "memory": {
        "peak_allocated_gib": 0.1480560302734375,
        "peak_reserved_gib": 0.357421875
      },
      "peak_delta_gib": 0.13691091537475586,
      "samples": 31,
      "gpu_util_avg": 7.290322580645161,
      "gpu_util_max": 18.0,
      "rss_sampled_peak_gib": 0.6050262451171875,
      "available_sampled_min_gib": 112.09645462036133,
      "throughput": 3034.985843186252,
      "memory_scope": "maximum per-worker allocator peak, not sum of processes; multiple models run serially",
      "coverage_mean": 102.97378,
      "reads_overlapping": 381,
      "normalized_exact_allele_metrics": {
        "SNP": {
          "TP": 57,
          "FP": 0,
          "FN": 0,
          "precision": 1.0,
          "recall": 1.0,
          "f1": 1.0
        },
        "INDEL": {
          "TP": 12,
          "FP": 0,
          "FN": 3,
          "precision": 1.0,
          "recall": 0.8,
          "f1": 0.8888888888888888
        }
      },
      "output_vcf": "results/clair3_20261027_203339_768667/region_50000_full/calls/merge_output.vcf.gz",
      "output_vcf_sha256": "be2698ba0dc635f08fe24d8193e18d02001aa99c982700b689fd9799301654ae",
      "peak_worker_model": "Clair3_F",
      "peak_worker_pid": 608
    },
    {
      "name": "region_100000_full",
      "mode": "variant_calling",
      "bp": 100000,
      "unique_inputs": 1,
      "repeats": 1,
      "processed": 100000,
      "bam": "HG003_chr20_demo.bam",
      "worker_loads": [
        {
          "time_unix": 1793100849.4098637,
          "pid": 876,
          "event": "model_only_postload",
          "model": "Clair3_P",
          "checkpoint": "/work/models/ont/pileup.pt",
          "total_parameters": 2074520,
          "trainable_parameters": 2074520,
          "load_seconds": 0.004170656204223633,
          "dtype": "torch.float32",
          "tf32": true,
          "memory": {
            "allocated_gib": 0.007729053497314453,
            "reserved_gib": 0.0234375,
            "peak_allocated_gib": 0.007729053497314453,
            "peak_reserved_gib": 0.0234375,
            "rss_gib": 0.67950439453125,
            "available_gib": 112.21522903442383
          }
        },
        {
          "time_unix": 1793100854.5220366,
          "pid": 1062,
          "event": "model_only_postload",
          "model": "Clair3_F",
          "checkpoint": "/work/models/ont/full_alignment.pt",
          "total_parameters": 2986522,
          "trainable_parameters": 2986522,
          "load_seconds": 0.00786137580871582,
          "dtype": "torch.float32",
          "tf32": true,
          "memory": {
            "allocated_gib": 0.01114511489868164,
            "reserved_gib": 0.025390625,
            "peak_allocated_gib": 0.01114511489868164,
            "peak_reserved_gib": 0.025390625,
            "rss_gib": 0.6190910339355469,
            "available_gib": 112.8741340637207
          }
        }
      ],
      "actual_input_shapes": [
        "[178, 33, 18]",
        "[183, 33, 18]",
        "[187, 33, 18]",
        "[192, 33, 18]",
        "[208, 33, 18]",
        "[210, 33, 18]",
        "[215, 33, 18]",
        "[216, 33, 18]",
        "[218, 33, 18]",
        "[221, 33, 18]",
        "[232, 33, 18]",
        "[238, 33, 18]",
        "[444, 89, 33, 8]"
      ],
      "forward_calls": 14,
      "forward_seconds": 0.7881958484649658,
      "candidate_tensors": 3134,
      "seconds": 16.69534397125244,
      "baseline": {
        "allocated_gib": 0.01114511489868164,
        "reserved_gib": 0.025390625,
        "peak_allocated_gib": 0.01114511489868164,
        "peak_reserved_gib": 0.025390625,
        "rss_gib": 0.6190910339355469,
        "available_gib": 112.8741340637207
      },
      "memory": {
        "peak_allocated_gib": 0.3468775749206543,
        "peak_reserved_gib": 0.904296875
      },
      "peak_delta_gib": 0.33573246002197266,
      "samples": 32,
      "gpu_util_avg": 7.1875,
      "gpu_util_max": 20.0,
      "rss_sampled_peak_gib": 0.6180953979492188,
      "available_sampled_min_gib": 111.18148040771484,
      "throughput": 5989.693903413376,
      "memory_scope": "maximum per-worker allocator peak, not sum of processes; multiple models run serially",
      "coverage_mean": 107.1313,
      "reads_overlapping": 682,
      "normalized_exact_allele_metrics": {
        "SNP": {
          "TP": 208,
          "FP": 0,
          "FN": 0,
          "precision": 1.0,
          "recall": 1.0,
          "f1": 1.0
        },
        "INDEL": {
          "TP": 28,
          "FP": 0,
          "FN": 5,
          "precision": 1.0,
          "recall": 0.8484848484848485,
          "f1": 0.9180327868852459
        }
      },
      "output_vcf": "results/clair3_20261027_203339_768667/region_100000_full/calls/merge_output.vcf.gz",
      "output_vcf_sha256": "29b30bdd70a192a96aa44d8ea9153b233241ce59fdf3fca6c177c8cac39c43bc",
      "peak_worker_model": "Clair3_F",
      "peak_worker_pid": 1062
    },
    {
      "name": "region_200000_full",
      "mode": "variant_calling",
      "bp": 200000,
      "unique_inputs": 1,
      "repeats": 1,
      "processed": 200000,
      "bam": "HG003_chr20_demo.bam",
      "worker_loads": [
        {
          "time_unix": 1793100866.6704588,
          "pid": 1334,
          "event": "model_only_postload",
          "model": "Clair3_P",
          "checkpoint": "/work/models/ont/pileup.pt",
          "total_parameters": 2074520,
          "trainable_parameters": 2074520,
          "load_seconds": 0.0058138370513916016,
          "dtype": "torch.float32",
          "tf32": true,
          "memory": {
            "allocated_gib": 0.007729053497314453,
            "reserved_gib": 0.0234375,
            "peak_allocated_gib": 0.007729053497314453,
            "peak_reserved_gib": 0.0234375,
            "rss_gib": 0.6795082092285156,
            "available_gib": 112.76848983764648
          }
        },
        {
          "time_unix": 1793100872.6980748,
          "pid": 1556,
          "event": "model_only_postload",
          "model": "Clair3_F",
          "checkpoint": "/work/models/ont/full_alignment.pt",
          "total_parameters": 2986522,
          "trainable_parameters": 2986522,
          "load_seconds": 0.007401704788208008,
          "dtype": "torch.float32",
          "tf32": true,
          "memory": {
            "allocated_gib": 0.01114511489868164,
            "reserved_gib": 0.025390625,
            "peak_allocated_gib": 0.01114511489868164,
            "peak_reserved_gib": 0.025390625,
            "rss_gib": 0.6191215515136719,
            "available_gib": 112.97261428833008
          }
        }
      ],
      "actual_input_shapes": [
        "[376, 33, 18]",
        "[384, 33, 18]",
        "[388, 33, 18]",
        "[392, 33, 18]",
        "[398, 33, 18]",
        "[399, 33, 18]",
        "[403, 33, 18]",
        "[414, 33, 18]",
        "[426, 33, 18]",
        "[468, 33, 18]",
        "[528, 33, 18]",
        "[554, 33, 18]",
        "[587, 33, 18]",
        "[894, 89, 33, 8]"
      ],
      "forward_calls": 14,
      "forward_seconds": 1.1960506439208984,
      "candidate_tensors": 6611,
      "seconds": 17.932143688201904,
      "baseline": {
        "allocated_gib": 0.01114511489868164,
        "reserved_gib": 0.025390625,
        "peak_allocated_gib": 0.01114511489868164,
        "peak_reserved_gib": 0.025390625,
        "rss_gib": 0.6191215515136719,
        "available_gib": 112.97261428833008
      },
      "memory": {
        "peak_allocated_gib": 0.6840786933898926,
        "peak_reserved_gib": 2.205078125
      },
      "peak_delta_gib": 0.6729335784912109,
      "samples": 34,
      "gpu_util_avg": 11.264705882352942,
      "gpu_util_max": 88.0,
      "rss_sampled_peak_gib": 0.6187782287597656,
      "available_sampled_min_gib": 110.28792572021484,
      "throughput": 11153.1562247957,
      "memory_scope": "maximum per-worker allocator peak, not sum of processes; multiple models run serially",
      "coverage_mean": 107.527935,
      "reads_overlapping": 1305,
      "normalized_exact_allele_metrics": {
        "SNP": {
          "TP": 404,
          "FP": 1,
          "FN": 0,
          "precision": 0.9975308641975309,
          "recall": 1.0,
          "f1": 0.9987639060568603
        },
        "INDEL": {
          "TP": 46,
          "FP": 4,
          "FN": 15,
          "precision": 0.92,
          "recall": 0.7540983606557377,
          "f1": 0.8288288288288288
        }
      },
      "output_vcf": "results/clair3_20261027_203339_768667/region_200000_full/calls/merge_output.vcf.gz",
      "output_vcf_sha256": "77110dda695d8cdb28c2a18583fee7db181d1ed3a2c3a02287f0ffe82daf8318",
      "peak_worker_model": "Clair3_F",
      "peak_worker_pid": 1556
    },
    {
      "name": "region_200000_half",
      "mode": "variant_calling",
      "bp": 200000,
      "unique_inputs": 1,
      "repeats": 1,
      "processed": 200000,
      "bam": "HG003_half.bam",
      "worker_loads": [
        {
          "time_unix": 1793100890.500904,
          "pid": 2070,
          "event": "model_only_postload",
          "model": "Clair3_F",
          "checkpoint": "/work/models/ont/full_alignment.pt",
          "total_parameters": 2986522,
          "trainable_parameters": 2986522,
          "load_seconds": 0.010561466217041016,
          "dtype": "torch.float32",
          "tf32": true,
          "memory": {
            "allocated_gib": 0.01114511489868164,
            "reserved_gib": 0.025390625,
            "peak_allocated_gib": 0.01114511489868164,
            "peak_reserved_gib": 0.025390625,
            "rss_gib": 0.6191177368164062,
            "available_gib": 112.81945419311523
          }
        },
        {
          "time_unix": 1793100884.7679694,
          "pid": 1834,
          "event": "model_only_postload",
          "model": "Clair3_P",
          "checkpoint": "/work/models/ont/pileup.pt",
          "total_parameters": 2074520,
          "trainable_parameters": 2074520,
          "load_seconds": 0.005149364471435547,
          "dtype": "torch.float32",
          "tf32": true,
          "memory": {
            "allocated_gib": 0.007729053497314453,
            "reserved_gib": 0.0234375,
            "peak_allocated_gib": 0.007729053497314453,
            "peak_reserved_gib": 0.0234375,
            "rss_gib": 0.6795005798339844,
            "available_gib": 112.54525375366211
          }
        }
      ],
      "actual_input_shapes": [
        "[1000, 89, 33, 8]",
        "[141, 89, 33, 8]",
        "[453, 33, 18]",
        "[471, 33, 18]",
        "[493, 33, 18]",
        "[545, 33, 18]",
        "[579, 33, 18]",
        "[585, 33, 18]",
        "[595, 33, 18]",
        "[603, 33, 18]",
        "[667, 33, 18]",
        "[670, 33, 18]",
        "[775, 33, 18]",
        "[813, 33, 18]",
        "[918, 33, 18]"
      ],
      "forward_calls": 15,
      "forward_seconds": 1.552288293838501,
      "candidate_tensors": 9308,
      "seconds": 17.233102321624756,
      "baseline": {
        "allocated_gib": 0.01114511489868164,
        "reserved_gib": 0.025390625,
        "peak_allocated_gib": 0.01114511489868164,
        "peak_reserved_gib": 0.025390625,
        "rss_gib": 0.6191177368164062,
        "available_gib": 112.81945419311523
      },
      "memory": {
        "peak_allocated_gib": 0.7628636360168457,
        "peak_reserved_gib": 2.98046875
      },
      "peak_delta_gib": 0.7517185211181641,
      "samples": 33,
      "gpu_util_avg": 12.090909090909092,
      "gpu_util_max": 58.0,
      "rss_sampled_peak_gib": 0.6192779541015625,
      "available_sampled_min_gib": 109.27793884277344,
      "throughput": 11605.571432662611,
      "memory_scope": "maximum per-worker allocator peak, not sum of processes; multiple models run serially",
      "coverage_mean": 50.804275,
      "reads_overlapping": 629,
      "normalized_exact_allele_metrics": {
        "SNP": {
          "TP": 404,
          "FP": 1,
          "FN": 0,
          "precision": 0.9975308641975309,
          "recall": 1.0,
          "f1": 0.9987639060568603
        },
        "INDEL": {
          "TP": 41,
          "FP": 1,
          "FN": 20,
          "precision": 0.9761904761904762,
          "recall": 0.6721311475409836,
          "f1": 0.7961165048543689
        }
      },
      "output_vcf": "results/clair3_20261027_203339_768667/region_200000_half/calls/merge_output.vcf.gz",
      "output_vcf_sha256": "b54bcf6a61f7f68f564bb5da83ac2a312d49aac456c43d89057b4f146bf0e344",
      "peak_worker_model": "Clair3_F",
      "peak_worker_pid": 2070
    }
  ],
  "settings": {
    "version": "2.0.4",
    "commit": "9d1dd3f2475072b6255900e406c34c6b0545fc88",
    "threads": 4,
    "gpu_predictor_workers": 1,
    "batch_policy": "official predictBatchSize200*5 for GPU, actual batches logged",
    "platform": "ont",
    "checkpoint": "models/ont/{pileup,full_alignment}.pt",
    "precision": "FP32 weights, original integer input features; model converts/scales input",
    "phasing": "default whatshap intermediate phasing",
    "preprocessing": "native ARM64 CFFI; CPython3.12 replaces optional PyPy accelerator",
    "warmup": "N/A, end-to-end includes each worker load and startup",
    "training": "supported by official Train; this run performs variant calling only"
  },
  "environment": {
    "torch": "2.13.0a0+8145d630e8.nv26.06",
    "cuda": "13.3",
    "gpu": "NVIDIA GB10"
  },
  "input_audit": {
    "bam_header": {
      "HD": {
        "VN": "1.6",
        "SO": "coordinate"
      },
      "SQ": [
        {
          "SN": "chr1",
          "LN": 248956422
        },
        {
          "SN": "chr2",
          "LN": 242193529
        },
        {
          "SN": "chr3",
          "LN": 198295559
        },
        {
          "SN": "chr4",
          "LN": 190214555
        },
        {
          "SN": "chr5",
          "LN": 181538259
        },
        {
          "SN": "chr6",
          "LN": 170805979
        },
        {
          "SN": "chr7",
          "LN": 159345973
        },
        {
          "SN": "chr8",
          "LN": 145138636
        },
        {
          "SN": "chr9",
          "LN": 138394717
        },
        {
          "SN": "chr10",
          "LN": 133797422
        },
        {
          "SN": "chr11",
          "LN": 135086622
        },
        {
          "SN": "chr12",
          "LN": 133275309
        },
        {
          "SN": "chr13",
          "LN": 114364328
        },
        {
          "SN": "chr14",
          "LN": 107043718
        },
        {
          "SN": "chr15",
          "LN": 101991189
        },
        {
          "SN": "chr16",
          "LN": 90338345
        },
        {
          "SN": "chr17",
          "LN": 83257441
        },
        {
          "SN": "chr18",
          "LN": 80373285
        },
        {
          "SN": "chr19",
          "LN": 58617616
        },
        {
          "SN": "chr20",
          "LN": 64444167
        },
        {
          "SN": "chr21",
          "LN": 46709983
        },
        {
          "SN": "chr22",
          "LN": 50818468
        },
        {
          "SN": "chrX",
          "LN": 156040895
        },
        {
          "SN": "chrY",
          "LN": 57227415
        },
        {
          "SN": "chrM",
          "LN": 16569
        },
        {
          "SN": "chr1_KI270706v1_random",
          "LN": 175055
        },
        {
          "SN": "chr1_KI270707v1_random",
          "LN": 32032
        },
        {
          "SN": "chr1_KI270708v1_random",
          "LN": 127682
        },
        {
          "SN": "chr1_KI270709v1_random",
          "LN": 66860
        },
        {
          "SN": "chr1_KI270710v1_random",
          "LN": 40176
        },
        {
          "SN": "chr1_KI270711v1_random",
          "LN": 42210
        },
        {
          "SN": "chr1_KI270712v1_random",
          "LN": 176043
        },
        {
          "SN": "chr1_KI270713v1_random",
          "LN": 40745
        },
        {
          "SN": "chr1_KI270714v1_random",
          "LN": 41717
        },
        {
          "SN": "chr2_KI270715v1_random",
          "LN": 161471
        },
        {
          "SN": "chr2_KI270716v1_random",
          "LN": 153799
        },
        {
          "SN": "chr3_GL000221v1_random",
          "LN": 155397
        },
        {
          "SN": "chr4_GL000008v2_random",
          "LN": 209709
        },
        {
          "SN": "chr5_GL000208v1_random",
          "LN": 92689
        },
        {
          "SN": "chr9_KI270717v1_random",
          "LN": 40062
        },
        {
          "SN": "chr9_KI270718v1_random",
          "LN": 38054
        },
        {
          "SN": "chr9_KI270719v1_random",
          "LN": 176845
        },
        {
          "SN": "chr9_KI270720v1_random",
          "LN": 39050
        },
        {
          "SN": "chr11_KI270721v1_random",
          "LN": 100316
        },
        {
          "SN": "chr14_GL000009v2_random",
          "LN": 201709
        },
        {
          "SN": "chr14_GL000225v1_random",
          "LN": 211173
        },
        {
          "SN": "chr14_KI270722v1_random",
          "LN": 194050
        },
        {
          "SN": "chr14_GL000194v1_random",
          "LN": 191469
        },
        {
          "SN": "chr14_KI270723v1_random",
          "LN": 38115
        },
        {
          "SN": "chr14_KI270724v1_random",
          "LN": 39555
        },
        {
          "SN": "chr14_KI270725v1_random",
          "LN": 172810
        },
        {
          "SN": "chr14_KI270726v1_random",
          "LN": 43739
        },
        {
          "SN": "chr15_KI270727v1_random",
          "LN": 448248
        },
        {
          "SN": "chr16_KI270728v1_random",
          "LN": 1872759
        },
        {
          "SN": "chr17_GL000205v2_random",
          "LN": 185591
        },
        {
          "SN": "chr17_KI270729v1_random",
          "LN": 280839
        },
        {
          "SN": "chr17_KI270730v1_random",
          "LN": 112551
        },
        {
          "SN": "chr22_KI270731v1_random",
          "LN": 150754
        },
        {
          "SN": "chr22_KI270732v1_random",
          "LN": 41543
        },
        {
          "SN": "chr22_KI270733v1_random",
          "LN": 179772
        },
        {
          "SN": "chr22_KI270734v1_random",
          "LN": 165050
        },
        {
          "SN": "chr22_KI270735v1_random",
          "LN": 42811
        },
        {
          "SN": "chr22_KI270736v1_random",
          "LN": 181920
        },
        {
          "SN": "chr22_KI270737v1_random",
          "LN": 103838
        },
        {
          "SN": "chr22_KI270738v1_random",
          "LN": 99375
        },
        {
          "SN": "chr22_KI270739v1_random",
          "LN": 73985
        },
        {
          "SN": "chrY_KI270740v1_random",
          "LN": 37240
        },
        {
          "SN": "chrUn_KI270302v1",
          "LN": 2274
        },
        {
          "SN": "chrUn_KI270304v1",
          "LN": 2165
        },
        {
          "SN": "chrUn_KI270303v1",
          "LN": 1942
        },
        {
          "SN": "chrUn_KI270305v1",
          "LN": 1472
        },
        {
          "SN": "chrUn_KI270322v1",
          "LN": 21476
        },
        {
          "SN": "chrUn_KI270320v1",
          "LN": 4416
        },
        {
          "SN": "chrUn_KI270310v1",
          "LN": 1201
        },
        {
          "SN": "chrUn_KI270316v1",
          "LN": 1444
        },
        {
          "SN": "chrUn_KI270315v1",
          "LN": 2276
        },
        {
          "SN": "chrUn_KI270312v1",
          "LN": 998
        },
        {
          "SN": "chrUn_KI270311v1",
          "LN": 12399
        },
        {
          "SN": "chrUn_KI270317v1",
          "LN": 37690
        },
        {
          "SN": "chrUn_KI270412v1",
          "LN": 1179
        },
        {
          "SN": "chrUn_KI270411v1",
          "LN": 2646
        },
        {
          "SN": "chrUn_KI270414v1",
          "LN": 2489
        },
        {
          "SN": "chrUn_KI270419v1",
          "LN": 1029
        },
        {
          "SN": "chrUn_KI270418v1",
          "LN": 2145
        },
        {
          "SN": "chrUn_KI270420v1",
          "LN": 2321
        },
        {
          "SN": "chrUn_KI270424v1",
          "LN": 2140
        },
        {
          "SN": "chrUn_KI270417v1",
          "LN": 2043
        },
        {
          "SN": "chrUn_KI270422v1",
          "LN": 1445
        },
        {
          "SN": "chrUn_KI270423v1",
          "LN": 981
        },
        {
          "SN": "chrUn_KI270425v1",
          "LN": 1884
        },
        {
          "SN": "chrUn_KI270429v1",
          "LN": 1361
        },
        {
          "SN": "chrUn_KI270442v1",
          "LN": 392061
        },
        {
          "SN": "chrUn_KI270466v1",
          "LN": 1233
        },
        {
          "SN": "chrUn_KI270465v1",
          "LN": 1774
        },
        {
          "SN": "chrUn_KI270467v1",
          "LN": 3920
        },
        {
          "SN": "chrUn_KI270435v1",
          "LN": 92983
        },
        {
          "SN": "chrUn_KI270438v1",
          "LN": 112505
        },
        {
          "SN": "chrUn_KI270468v1",
          "LN": 4055
        },
        {
          "SN": "chrUn_KI270510v1",
          "LN": 2415
        },
        {
          "SN": "chrUn_KI270509v1",
          "LN": 2318
        },
        {
          "SN": "chrUn_KI270518v1",
          "LN": 2186
        },
        {
          "SN": "chrUn_KI270508v1",
          "LN": 1951
        },
        {
          "SN": "chrUn_KI270516v1",
          "LN": 1300
        },
        {
          "SN": "chrUn_KI270512v1",
          "LN": 22689
        },
        {
          "SN": "chrUn_KI270519v1",
          "LN": 138126
        },
        {
          "SN": "chrUn_KI270522v1",
          "LN": 5674
        },
        {
          "SN": "chrUn_KI270511v1",
          "LN": 8127
        },
        {
          "SN": "chrUn_KI270515v1",
          "LN": 6361
        },
        {
          "SN": "chrUn_KI270507v1",
          "LN": 5353
        },
        {
          "SN": "chrUn_KI270517v1",
          "LN": 3253
        },
        {
          "SN": "chrUn_KI270529v1",
          "LN": 1899
        },
        {
          "SN": "chrUn_KI270528v1",
          "LN": 2983
        },
        {
          "SN": "chrUn_KI270530v1",
          "LN": 2168
        },
        {
          "SN": "chrUn_KI270539v1",
          "LN": 993
        },
        {
          "SN": "chrUn_KI270538v1",
          "LN": 91309
        },
        {
          "SN": "chrUn_KI270544v1",
          "LN": 1202
        },
        {
          "SN": "chrUn_KI270548v1",
          "LN": 1599
        },
        {
          "SN": "chrUn_KI270583v1",
          "LN": 1400
        },
        {
          "SN": "chrUn_KI270587v1",
          "LN": 2969
        },
        {
          "SN": "chrUn_KI270580v1",
          "LN": 1553
        },
        {
          "SN": "chrUn_KI270581v1",
          "LN": 7046
        },
        {
          "SN": "chrUn_KI270579v1",
          "LN": 31033
        },
        {
          "SN": "chrUn_KI270589v1",
          "LN": 44474
        },
        {
          "SN": "chrUn_KI270590v1",
          "LN": 4685
        },
        {
          "SN": "chrUn_KI270584v1",
          "LN": 4513
        },
        {
          "SN": "chrUn_KI270582v1",
          "LN": 6504
        },
        {
          "SN": "chrUn_KI270588v1",
          "LN": 6158
        },
        {
          "SN": "chrUn_KI270593v1",
          "LN": 3041
        },
        {
          "SN": "chrUn_KI270591v1",
          "LN": 5796
        },
        {
          "SN": "chrUn_KI270330v1",
          "LN": 1652
        },
        {
          "SN": "chrUn_KI270329v1",
          "LN": 1040
        },
        {
          "SN": "chrUn_KI270334v1",
          "LN": 1368
        },
        {
          "SN": "chrUn_KI270333v1",
          "LN": 2699
        },
        {
          "SN": "chrUn_KI270335v1",
          "LN": 1048
        },
        {
          "SN": "chrUn_KI270338v1",
          "LN": 1428
        },
        {
          "SN": "chrUn_KI270340v1",
          "LN": 1428
        },
        {
          "SN": "chrUn_KI270336v1",
          "LN": 1026
        },
        {
          "SN": "chrUn_KI270337v1",
          "LN": 1121
        },
        {
          "SN": "chrUn_KI270363v1",
          "LN": 1803
        },
        {
          "SN": "chrUn_KI270364v1",
          "LN": 2855
        },
        {
          "SN": "chrUn_KI270362v1",
          "LN": 3530
        },
        {
          "SN": "chrUn_KI270366v1",
          "LN": 8320
        },
        {
          "SN": "chrUn_KI270378v1",
          "LN": 1048
        },
        {
          "SN": "chrUn_KI270379v1",
          "LN": 1045
        },
        {
          "SN": "chrUn_KI270389v1",
          "LN": 1298
        },
        {
          "SN": "chrUn_KI270390v1",
          "LN": 2387
        },
        {
          "SN": "chrUn_KI270387v1",
          "LN": 1537
        },
        {
          "SN": "chrUn_KI270395v1",
          "LN": 1143
        },
        {
          "SN": "chrUn_KI270396v1",
          "LN": 1880
        },
        {
          "SN": "chrUn_KI270388v1",
          "LN": 1216
        },
        {
          "SN": "chrUn_KI270394v1",
          "LN": 970
        },
        {
          "SN": "chrUn_KI270386v1",
          "LN": 1788
        },
        {
          "SN": "chrUn_KI270391v1",
          "LN": 1484
        },
        {
          "SN": "chrUn_KI270383v1",
          "LN": 1750
        },
        {
          "SN": "chrUn_KI270393v1",
          "LN": 1308
        },
        {
          "SN": "chrUn_KI270384v1",
          "LN": 1658
        },
        {
          "SN": "chrUn_KI270392v1",
          "LN": 971
        },
        {
          "SN": "chrUn_KI270381v1",
          "LN": 1930
        },
        {
          "SN": "chrUn_KI270385v1",
          "LN": 990
        },
        {
          "SN": "chrUn_KI270382v1",
          "LN": 4215
        },
        {
          "SN": "chrUn_KI270376v1",
          "LN": 1136
        },
        {
          "SN": "chrUn_KI270374v1",
          "LN": 2656
        },
        {
          "SN": "chrUn_KI270372v1",
          "LN": 1650
        },
        {
          "SN": "chrUn_KI270373v1",
          "LN": 1451
        },
        {
          "SN": "chrUn_KI270375v1",
          "LN": 2378
        },
        {
          "SN": "chrUn_KI270371v1",
          "LN": 2805
        },
        {
          "SN": "chrUn_KI270448v1",
          "LN": 7992
        },
        {
          "SN": "chrUn_KI270521v1",
          "LN": 7642
        },
        {
          "SN": "chrUn_GL000195v1",
          "LN": 182896
        },
        {
          "SN": "chrUn_GL000219v1",
          "LN": 179198
        },
        {
          "SN": "chrUn_GL000220v1",
          "LN": 161802
        },
        {
          "SN": "chrUn_GL000224v1",
          "LN": 179693
        },
        {
          "SN": "chrUn_KI270741v1",
          "LN": 157432
        },
        {
          "SN": "chrUn_GL000226v1",
          "LN": 15008
        },
        {
          "SN": "chrUn_GL000213v1",
          "LN": 164239
        },
        {
          "SN": "chrUn_KI270743v1",
          "LN": 210658
        },
        {
          "SN": "chrUn_KI270744v1",
          "LN": 168472
        },
        {
          "SN": "chrUn_KI270745v1",
          "LN": 41891
        },
        {
          "SN": "chrUn_KI270746v1",
          "LN": 66486
        },
        {
          "SN": "chrUn_KI270747v1",
          "LN": 198735
        },
        {
          "SN": "chrUn_KI270748v1",
          "LN": 93321
        },
        {
          "SN": "chrUn_KI270749v1",
          "LN": 158759
        },
        {
          "SN": "chrUn_KI270750v1",
          "LN": 148850
        },
        {
          "SN": "chrUn_KI270751v1",
          "LN": 150742
        },
        {
          "SN": "chrUn_KI270752v1",
          "LN": 27745
        },
        {
          "SN": "chrUn_KI270753v1",
          "LN": 62944
        },
        {
          "SN": "chrUn_KI270754v1",
          "LN": 40191
        },
        {
          "SN": "chrUn_KI270755v1",
          "LN": 36723
        },
        {
          "SN": "chrUn_KI270756v1",
          "LN": 79590
        },
        {
          "SN": "chrUn_KI270757v1",
          "LN": 71251
        },
        {
          "SN": "chrUn_GL000214v1",
          "LN": 137718
        },
        {
          "SN": "chrUn_KI270742v1",
          "LN": 186739
        },
        {
          "SN": "chrUn_GL000216v2",
          "LN": 176608
        },
        {
          "SN": "chrUn_GL000218v1",
          "LN": 161147
        },
        {
          "SN": "chrEBV",
          "LN": 171823
        }
      ],
      "PG": [
        {
          "ID": "minimap2",
          "PN": "minimap2",
          "VN": "2.14-r883",
          "CL": "minimap2 -t 40 -aL -z 600,200 -x map-ont /nas4/yswong/Clair_multi/fasta/GRCh38_no_alt_analysis_set.fasta /bal1-1/rbluo/Clairvoyante/pfda2/hg003/HG003_GM24149_1_2_3_Guppy_3.6.0_prom.fastq.gz"
        },
        {
          "ID": "samtools",
          "PN": "samtools",
          "PP": "minimap2",
          "VN": "1.10",
          "CL": "samtools view -@ 12 -bh -o HG003_chr20_demo.bam /mnt/bal36/zxzheng/testData/ont/data/HG003_GM24149_1_2_3_Guppy_3.6.0_prom.bam chr20:100000-300000"
        }
      ]
    },
    "sort_order": "coordinate",
    "reference_length": 64444167,
    "regions": [
      {
        "region_0based_halfopen": [
          "chr20",
          100000,
          150000
        ],
        "length_bp": 50000,
        "reads_overlapping": 381,
        "coverage_mean": 102.97378,
        "coverage_median": 103.0,
        "coverage_min": 93,
        "coverage_max": 122,
        "read_length_mean": 27879.128608923886
      },
      {
        "region_0based_halfopen": [
          "chr20",
          100000,
          200000
        ],
        "length_bp": 100000,
        "reads_overlapping": 682,
        "coverage_mean": 107.1313,
        "coverage_median": 106.0,
        "coverage_min": 93,
        "coverage_max": 126,
        "read_length_mean": 24254.41935483871
      },
      {
        "region_0based_halfopen": [
          "chr20",
          100000,
          300000
        ],
        "length_bp": 200000,
        "reads_overlapping": 1305,
        "coverage_mean": 107.527935,
        "coverage_median": 107.0,
        "coverage_min": 84,
        "coverage_max": 129,
        "read_length_mean": 20632.950957854406
      }
    ],
    "first_read": {
      "name": "dc9c6024-1ac4-499b-87b8-2d7e64475b46",
      "reference_start_0based": 60000,
      "cigar": "22735S14M1D32M1I6M3D4M1D13M1D21M1D103M1I79M1I80M1D54M1D12M1I19M1I11M1I25M1D79M1D6M1D18M2D88M2D6M1D45M1D9M1D46M1D24M1D9M1I59M1I2M1D32M1D38M1D31M1D46M1D45M1D29M1D5M2D67M1D57M2D11M2D52M1I7M1D24M1D129M1D9M1D13M1I9M3D11M1529I31M1D27M3D4M1D19M1D3M1I9M2D23M1D21M3I11M2D12M1D2M1D50M2D27M1D99M1D10M1D84M1D14M1D29M3D33M2D68M2D49M1D19M1D27M2D28M1D26M1D11M1D45M1D22M1D42M2D47M1I6M2D34M1D40M1D39M1I5M1D40M2D9M1I9M3D37M1D4M1D24M3D42M2D5M1D44M1D2M1I5M1I7M1D4M1D15M1I15M1D23M1I1M1I12M1D9M1I47M1I69M1D19M5I35M1D70M5D63M45D590M1I28M1I46M1D2M2D96M1D1M2D4M2D19M1I11M1D50M2D25M2D4M1I6M1D54M2D58M1D1M2D39M1D1M3D1M1D28M7D72M1D71M1D23M3I4M1I5M1I77M1I2M2D15M1I37M1I82M2I7M1D32M1D134M3D48M1D1M1D19M1I8M1D4M1I115M1D52M2D10M1D15M1D24M1D2M1D77M2D4M1D90M1D30M1I39M1D2M2I5M4I34M1I99M2D24M1D49M2I20M1D35M2D73M2D39M1D4M1D4M1D63M2D47M1D11M1D4M1D4M1D38M1D153M1D30M9I146M2D12M1D7M1D3M1I27M1I41M1I2M1I28M2D25M1D40M1D9M1I4M1I59M1D59M1D4M1D33M2D4M1D2M1I42M1D32M1D19M1D4M1I30M1D10M2D4M1D3M2D19M1I18M1D34M1D4M1I5M1D6M1D4M1D3M2D3M2D8M2D14M3D2M1D18M2D2M1D6M1D4M1D37M1D13M1D3M1I38M2D4M1D22M1D81M1D2M1I18M3D5M1D47M1D74M1D18M1D76M1I31M3I17M2D9M1I32M1I54M1D8M2D92M1I8M1D109M1I50M1I32M1D10M3D6M1D13M2D37M1D112M1I14M1I8M1I16M2D120M1D1M2D51M1D5M2I67M2D6M1D53M1D95M3D33M1D41M1D14M3D5M1D47M2D91M1I31M2D13M1D76M1D1M1D63M1D19M1D92M1D7M5D11M1I10M1D11M1I87M1D68M2D6M1D32M1D3M1D8M3I5M1I37M1D58M3I76M2I30M3I9M2I39M1I15M2I189M1D17M2D6M1D4M1I2M2D9M1I55M2I43M2D150M1D28M1D68M2D1M2D47M1D30M1I126M1I13M1D28M1I12M1I50M1D62M2I53M1D43M1D29M1I37M1D9M2D17M2D13M1D6M1D141M1I13M1D11M2D33M3D210M1D7M2D11M2D10M3I11M4D88M1D40M1D52M1I22M1D66M1I52M2I12M3I33M1D110M1D46M1D1M1D2M1D23M1I5M1I43M1I57M1I49M1D172M1D49M1I105M1I89M1D27M1I2M1I83M1D68M1D54M1I19M1D52M1D6M1D18M2D6M2D42M1I49M1D20M1I81M1D146M2D65M1I138M1I2M1I78M2I19M1I45M1D19M2I76M2D17M1I19M1I48M1I14M1D292M1I4M1D5M1D27M1I64M1I29M1D72M1I77M1D78M2D20M1D30M1D10M1I29M4D9M1I47M1D68M1I70M2I2M2D94M1D54M3D125M1D169M1D30M1D151M2D5M2I2M2D30M1I52M1D140M1D41M1D134M1D60M3D4M2D39M1D27M1I66M2D19M1D3M1D18M1I48M1D2M2I11M1D66M1I110M1I69M2I83M2D24M1I103M1D117M1D118M2I28M1D10M2D35M2I21M1I7M1D33M2D92M2D19M2D74M1I108M1D24M1D76M1I98M1I8M1D84M1D97M2D93M2I68M1I3M1D38M1D15M2D21M2D35M2D35M1I142M1D17M2I7M1D35M1D20M1D49M1D34M1D117M1D15M1D52M1D159M1D34M1D173M1I70M1D2M1D93M1D14M1D4M1I53M1I18M1D78M1D22M1D157M1D4M1D33M4D3M2D5M3I5M4D20M2I51M1D10M2I102M1I5M1D14M1I6M2I7M1I6M1D3M1D31M2D38M3D11M1D14M1D13M1D35M1I12M1D72M2I82M1D31M2I10M2D30M1D37M1D29M4D32M1D28M1D2M1I4M4D4M1D16M2D40M2D2M1D2M1D2M1D5M1D13M2D20M1D32M1D146M8I9M1I9M1I7M2I27M1D3M1D56M2D15M1D38M1I15M5D123M1I9M2I12M1I223M1D268M1I22M2D70M1D3M1D65M1D29M1I29M1D10M1D42M1D2M1D6M3D2M1D29M1D33M2I30M1D88M2D47M1I7M1I31M1D3M4D3M3D14M1I20M2D68M1D13M1D3M1D20M3I82M1I15M1D48M3I18M1I39M2D125M1I3M1I2M1I36M1I2M1I1M3I15M2D63M1D10M1D11M4D62M1D36M1D3M1D10M2D47M2D29M2D47M2D2M4D2M3I3M1D18M1D2M1D18M1D8M2I5M1D9M2D4M2D25M1D12M2D48M1I53M2D97M1D33M1D78M1D15M1D24M1D59M1D24M1D19M4D181M1I15M4I25M1I8M1D28M1D102M2D22M1D2M1D10M1D12M6D85M1I81M1D119M1I23M1D11M2I17M5D3M1D4M2D9M1D6M1D7M2D90M1D8M5D33M2I38M1I6M1D17M2D1M1I87M2D29M1I31M1I44M2I14M2D8M2D48M1D75M1I50M1D6M1I47M1D32M1D38M2D33M1D50M2D17M3I6M1D33M1I29M1I64M1I1M1I6M5I15M2D14M2D12M4D4M2D16M1D7M1I96M1D108M1D22M3I14M2D22M1D19M1D3M1I47M2D29M2D20M3D82M1D15M1D4M4D33M2D80M2I70M1D24M1I21M3I50M1I4M1I5M1I35M1D73M1D58M1D5M3I114M1D4M1D18M1I37M2D18M1I25M2D34M2I51M1I35M2D4M1I9M1D7M1I8M2D24M2I9M1I33M2D4M1I1M2I33M1D56M1I3M1I5M1D38M2I16M1D34M1I23M1D56M1I76M1I32M3D29M4D4M1D7M1D10M10I68M1I32M1D90M1I48M2D21M1D5M1D7M1D2M3D23M1I14M1D40M1I58M3D21M1I85M3D138M1D71M7D4M1I5M1D6M2I11M2I29M2I3M2I26M3D5M1D11M1I17M2I52M3D4M1I60M3D50M1D6M1I85M1D5M1I46M1D61M1I61M1D5M1D59M3D20M2D138M1D40M2I120M1D52M1D133M1D24M1I43M1D21M2I6M1D24M6I128M3D13M4D87M1D10M1D10M2D4M1D14M1D1M1D27M1D12M1D144M1I1M1I29M1D28M1I6M1D8M3D4M1I69M1I33M1D36M1I2M1I11M1I53M1I83M1D13M1I9M2D10M2D16M1D7M2D28M1I2M2D4M1D51M1I69M1D88M2D71M2D59M1I56M2I21M1D131M1D31M3D6M1I96M4D46M1I29M1D5M1D73M1I150M1I40M1I77M1D115M3D9M1D43M1I46M1D37M1D10M1I89M3I27M1D4M1D51M1D52M1D54M1D113M1D29M1I131M1D5M1D43M3D41M3I34M1D30M1I89M4D13M1I4M1D8M1I85M1D130M1I15M1D40M1I15M1D116M4D36M1D21M1D101M1D37M1D77M1I8M1I46M3I4M1D6M1D63M1D62M1I46M2D14M2D13M3D173M1D47M1D72M3I43M1D100M1D4M1D32M1I33M1D86M2D52M4I64M1D1M1D108M1D36M3D18M1D53M4D38M1I55M2D16M1D5M1D49M1D34M1D54M1D46M1I27M2I8M2D126M1D24M1D129M1D61M2D57M1D25M2D21M2D15M1I44M1D84M3D4M2I25M3D1M1I2M1I3M1I26M1D4M1I145M1I28M1D43M2I14M1D12M1D85M2I10M1I43M2I96M1I2M2D20M3I28M1D33M1D49M1I12M2D13M1D36M1I95M1I24M1D28M1I51M1D27M2D4M2D49M1D45M1D35M1D50M2D150M1D24M2D11M3D5M1D31M1I26M1I45M1D12M1D37M1I98M1D30M1D101M2D31M1D9M1D229M1I62M1I135M2D44M2D45M1I16M1I55M1D130M1I198M2D5M2I55M2D28M1D15M2D8M1I23M3D45M1I34M1D3M1D23M1D4M1D30M1D6M1I3M1D3M7D3M1I4M2I10M1I7M3D3M1I215M1D8M1D42M2I31M2D9M1D17M1I73M1D33M1D1M1D64M1I32M1I35M4D11M1D117M1D112M1D17M1I1M2D7M1D13M1I63M3I1M1I7M1I261M2D186M2D9M1D74M1I20M1D52M1I159M1D6M1D28M1D19M1D53M1I8M1I63M1D45M1D61M1D158M1D23M1I8M1D4M2D10M3D160M2D98M1D203M5I2M3I225M1D121M1D149M1I180M1I25M1I7M2I23M1D21M2D108M1I88M3D122M1D8M1I28M1D12M3D6M1D8M3D12M1I13M2D20M1I10M1D9M1D5M1D2M1D2M2I18M1I5M2D4M2I10M2D9M1D6M4D6M2D5M2D4M1D30M1I17M2D8M1I7M1I2M1D8M1D17M1I15M2D7M1D6M1D3M1D4M1D3M1I4M1I120M1D7M1D18M1D42M3D74M1I5M1I26M2I11M2D35M1D13M1D6M1D3M3I1M1D220M8D61M1I24M1D134M4I40M3D60M1I98M1D25M1D56M1D166M1D56M1D50M1I9M1D42M1D2M1D157M2D6M1I26M1D31M2I19M2I16M1I91M1D57M3I30M1D102M1I21M1D2M1D5M1I33M1D4M2I8M1D7M1D20M1D31M1D31M1D51M1D30M1I41M2D39M1D56M1D12M3D60M1D38M1I18M2I62M1D11M2I49M1D29M1D40M2I136M1D14M2D10M2D79M1I48M3D47M1D116M3D25M2D3M1D24M2D50M2D87M2I149M2D3M1D4M1D10M1I3M1D53M2D15M1D24M1D86M3D127M1D28M1D24M2D19M2D120M2I16M1D23M1D5M1D4M1I3M1D14M2D7M1D18M1D31M2I2M1D170M1I76M1D8M2D19M4I1M1I14M1I29M3D1M1D70M1D10M1D14M5D7M1D4M2I5M1D19M3I12M1D37M3I11M1D4M1D19M1D36M3D23M2I41M1D57M1D41M1D20M1I58M3I15M1D14M1D17M1D6M2I3M1I25M3D8M2I12M1D11M1I18M4I6M4D16M1D10M2D7M1D10M1D18M1D32M4D9M2I16M1I10M1I23M1I23M1I25M1D4M2D4M2D24M3D6M2D16M1D69M1D97M1D45M1D9M1D8M1D25M1D39M1D16M1I6M2D85M1D40M2I81M1D48M4D6M1I18M2I122M1D11M1I38M1D14M2D49M2D1M1D20M1I65M1D7M2D8M1D23M2D49M2D9M1I64M1I29M2I19M1D9M2I13M1D175M2D4M2I42M3D7M7D6M1D38M1D69M1I27M1D3M1D5M1I9M1D163M1I21M1I6M1I1M2D59M2I20M4D3M3D1M2I89M1I43M2D35M1D7M3D21M1D24M1D2M2D11M2I5M3D3M2I37M1D5M1D31M1I24M2I3M1I55M1D37M1D30M1I16M2D3M1D37M1I17M1D3M1D15M1I2M1D5M2I12M1I60M1D11M3I2M1I2M4D27M2D43M1D19M1D13M3D10M1I23M1I16M2I33M2D2M2D17M2I19M1I8M2D35M1D14M1D41M1I48M1D84M1I31M1I49M1I21M1D37M1D17M1I1M1I62M3I4M37I22M1D74M1D136M1D82M2I35M1D7M1I3M1I63M1I56M1D26M1I4M1D4M1D7M1D20M1I10M1D5M1D1M1D47M1D44M2I15M1I63M2D16M1D28M1I25M1D27M2D50M2I93M1D55M1D5M3D17M2D66M1D6M1D17M2I13M1D13M1I4M1I38M1D10M1D9M2D47M1D32M1D6M1D5M2I32M1D77M2I5M2D16M1D4M1D22M2D7M1I32M1D1M1D4M2D18M1I42M1D7M1D3M1D25M1D7M1D16M1I29M1I2M1D28M1D17M1I10M1I7M1D11M1D3M3D28M1I15M1D24M1D41M1D7M1I28M3I5M1D9M1D36M1D85M4D62M2D7M1I5M1I42M1D28M2I6M1D3M1D36M1I61M1I16M9I82M1D112M1I3M1D8M1I84M1I2M1I2M6D152M2I14M2D4M1D21M1I102M1D1M2I25M1I36M1I98M1I16M1I26M1I5M1I44M1D14M2I28M1D56M1D127M3I1M2D13M1D86M1I12M1D41M1I63M2I98M1I86M1D15M2D19M1I3M1D13M2D35M2D87M1I10M1I28M1I24M1D2M1D1M3I36M1D27M1D21M2D35M1D63M1D14M1D3M2I11M2D8M2D79M1D8M1D17M1D7M1D13M1D3M1I3M1D18M3D37M1I4M1D4M1D7M1D1M1D15M2D13M1D76M1D1M1D31M3D6M1I13M1I4M1I30M1D93M1D2M1D5M1I39M3D7M1D20M1I92M2I1M1D7M2I28M1D47M1I53M1D6M2D12M1D3M1D52M1I6M1I29M1I47M1D10M2I8M1D36M1I14M1D3M1D152M4I17M2D21M1D6M2I4M2I7M1I21M3D21M1D7M1I46M1D87M2I40M1I14M3I12M2D4M1D6M1I12M1I6M1D7M2D16M1D22M1D5M1I7M1D16M2D18M2D34M6D26M1I58M1D12M5D11M1I51M2I10M1I13M2I10M1I31M1I63M1I51M1D42M1I50M2I2M1D61M1D23M1I4M1I2M1D7M4I4M3D14M2I67M1I2M2D9M1D35M1D3M1I3M1D96M1D126M1I89M1I18M1D195M1I8M1D19M1I47M1I22M2D12M2D24M1D35M1D17M1I31M1I96M1D6M1D34M1D26M1D39M2D27M6D45M1I2M1D22M1I36M3D77M2D22M1D30M1D61M1D38M1D15M1D27M1D22M1I17M2D23M1D80M1I84M3I7M1D39M1D200M1D37M1I198M1D58M1D68M1I23M2D26M5D138M1I58M1D9M1D3M1I52M2I45M2D131M1I39M2I22M1D8M1I37M1I175M1D108M2D4M1I56M1D11M2D214M1I11M2I47M1D197M1I54M2I21M1D57M1I14M2D51M3I12M1D6M3D44M1I62M1D44M1D50M2D3M1I17M1D36M2D32M1I75M1I100M1I34M3I2M1D40M1I54M1D64M1D96M1D155M2I55M1D163M1I5M1I17M1D89M1I16M1I24M2I7M1D126M1D29M1D19M1I9M2D34M2D45M2D71M2D53M1I42M1I25M1D18M1I10M3D76M2D3M2D36M1I19M2D78M2I67M4D5M1D100M2D48M2D32M1I43M1D131M3I12M1D67M1I78M1D8M1I6M1D132M1I73M2I14M1D19M1D20M3D5M1I19M3D16M2I7M2D61M1D6M2I15M2I22M1D6M1D65M5D23M3I42M4D64M2D13M1D17M1D51M3I99M2I77M1D90M1I8M1D71M4D3M2I32M1D73M1D9M1D98M1D14M2D20M2I44M1D1M2I110M1D21M1I43M1I5M1D67M1D11M1I21M1D21M2D9M1I61M1D14M1D39M1D45M2I40M1D43M1D204M3I2M1D71M1I63M1D135M1D144M1D49M1D10M1D149M1I74M1I42M3D49M1D26M1D41M1D93M3D57M5I41M1I135M1I16M1D252M2I84M1D34M1I19M1D16M1D76M1D13M2I14M1I2M1I18M1I27M3I1M1I6M1D3M1D15M1D201M1I33M1D8M1I106M1D48M3D87M1D101M1I5M1I58M1D12M1I3M1I6M1D7M4D41M1D5M1I30M4D54M1I27M2D107M1D72M1D67M1D13M2D48M2D33M1D8M1I68M1D28M1D13M1D47M1I4M2I121M1D128M3D80M1D13M1D209M3I1M1I14M2D52M1I7M1D9M1D16M1I9M2D10M2D48M1D82M1D74M2D16M1I46M1I30M1D8M2I19M1D25M1I41M1D21M1I11M1I23M1I18M3D9M1D4M1D22M1I4M1D5M1D8M1I2M2D57M2D41M1I22M1I7M1D22M3I48M2D36M1I78M1D2M1D3M1D65M2D6M2I23M1I3M1I15M1I17M2I34M2I68M1I15M1I21M2D8M1D1M1D118M1D55M3D9M1D14M1I53M1I18M1D12M1D24M1D3M2D16M1D109M1D66M1I20M1D1M2D15M1D9M1D81M3I12M1I242M1D17M1D2M2D37M2I26M2I42M2D30M1I96M1I72M1D11M2I48M1D40M2D26M2D50M1D16M2D135M1D24M1D72M2I63M1I38M1I39M1D47M2I51M1D84M1D94M1D10M1I15M1D32M1D17M1D41M1D20M1D28M1I5M5D8M2D2M1I72M1D37M3D59M1D27M1I27M1D26M1D15M1I2M1D46M5D81M1D91M1I16M1I116M1D136M1D151M1D13M1I98M4D50M2D29M1D88M1I4M1I51M2I39M1I79M1D41M1D2M1D30M1I13M1I35M1D68M4I38M1D62M2D53M1I23M1I140M1D64M2D46M1I39M2D81M1I11M2D69M2D12M1I89M2I108M2D13M2I3M1D1M1D4M1D16M2I27M3D47M1I3M1D62M3D77M1I31M1I89M3D1M1D38M2D39M6D7M2I41M2D1M1D1M1D86M2I188M1D243M11I75M2I3M1D15M1D169M1I40M1D18M1D157M1D14M2D86M1I27M1I160M2D10M2D12M1I78M1I47M1D1M1D41M2I27M1I20M1I14M1I12M4D35M2I15M1I31M1I69M1I3M1I37M1D37M2D8M6D9M1D17M1D55M1I75M2D27M2D19M1I57M2I13M1D17M1D33M1D119M2D14M2I2M2D66M1D3M1D3M1I37M2D54M1I5M1D3M1D15M1D19M1I22M1D24M1I41M1I31M1D62M1D65M2D50M1D16M1D44M1I148M2I24M5D300M1I63M1D4M1D111M1I106M1I3M1D14M1D54M1D40M1D58M2D37M1D30M1I83M1D4M1D13M3I22M1I82M2I16M1I27M1I52M1I26M1I184M2I37M1D59M1D2M1I13M2I9M1D19M2D43M1I32M2I18M1D46M1D5M1D9M2I11M2D17M1D7M6D37M3D97M1D3M1D75M1I53M2D4M2D2M1D15M1I237M1I1M1I22M2I31M1I6M2I176M1D18M1D29M3D3M2D11M1D6M2D24M1I2M1I26M2D4M2D11M1I27M1D15M4D14M2D5M2I16M1D4M1I30M1D21M2D2M2I2M1I131M1D6M2I11M1D45M2D14M1D36M2I25M1D11M1D15M1I2M1I7M1D19M1D50M1D3M2D59M1D19M1I7M1I16M1D102M1I4M1I17M1D55M2D18M1I17M2D29M1D8M2D13M2D52M1D69M2D33M1D37M2I12M1D8M1D20M3I19M1I20M1D41M4D14M1I55M2I16M1D36M1I27M1I25M1D15M1D8M1D32M1I16M1D6M1D8M2D2M2I12M4I5M2D9M1D47M1D9M2D71M1I26M1D46M4D19M1D102M2D7M2D5M1I150M1D2M1D11M2I27M5D64M1D30M2D2M1D95M1D8M1D16M1D2M2I40M2I42M1I10M2I24M1I27M1D75M1I26M1D86M1D4M1I29M1I35M1I34M1I5M1I36M2I9M1D44M1D4M2D8M2D1M1D17M1I3M1D1M2D7M1D24M2D14M2D21M1I4M1I31M1D14M2I44M1I15M1D33M1D79M1I106M1D41M1D31M4D12M1D7M1I15M2D12M1D83M2D8M1D4M1I2M3D6M1I62M2D73M2D30M3D3M2D16M1D1M1D31M4D31M1I42M1D38M3D24M1D3M1I5M2D1M3D25M1D19M1D5M3D16M1I3M1D7M1I9M3I1M1I14M1D6M3D5M1D21M1I25M1I8M1D35M1D5M1I10M1D55M1D2M1D10M1I79M1D5M1I80M2D13M1I11M2D32M1D29M2D24M1D7M1I49M1D1M1D16M2D13M5D32M2I18M1I12M1D52M1I11M1D13M1I85M1D37M1I12M1I7M1D38M8D1M1D3M1D7M1D46M1D4M2I5M1I4M2D15M1D90M1I64M1D4M1D3M1D5M2D7M4I66M1I15M1I35M1D10M1D22M1D12M1I7M1D13M1D44M1D25M1I4M1D3M1D41M1D24M1I9M2I60M1D6M2D130M2I3M1I7M1D3M1D39M2D6M1I4M1D11M1I15M1D44M5D4M2D5M1I25M1I25M1I115M1I30M1D2M2I28M3I17M2I21M1I130M1D5M2D18M2D14M1D5M1I92M1I25M2D9M1I5M1I7M1D2M1D40M4D85M1I5M2I2M4D4M1D60M1I44M1D11M1I108M2I59M1D39M3D8M1D4M3D78M1D6M1I24M1I78M1D2M2I48M1I13M2I12M1D13M3D54M1I14M72I10M3I4M3I17M2D26M1D21M1I6M1D35M1I6M2D21M1D50M1D3M2I10M1D6M1I17M1D51M1D128M1D17M1D23M4D14M1D5M2D64M1D10M1D11M1I90M3D32M2D4M1I21M3D2M2D5M3I14M1D25M1D44M1D6M1I4M1D14M2D28M2D6M1D51M1I3M2D2M1I3M1D12M2D19M1I61M1I13M1I56M2I19M2D12M1D21M1D4M1I39M6D1M1D12M3D1M1D8M1I9M3D18M1D3M2D67M1D14M2D7M1D12M1D23M2D21M1D4M1D15M1D2M2I12M1I14M4D5M2I21M2I31M3D66M1I34M2I6M1D43M1D23M1D14M1I13M1I6M1D32M4D4M1D4M2I1M1D3M1D40M1D10M2I2M2D3M1D60M1I28M2I10M1D9M3I14M3D32M3D35M2I32M1D86M1I7M1D20M1D25M1D96M1D24M1I80M1I10M4D30M2D12M1D9M1I98M3D81M2I18M1D22M2D51M1I69M2D37M1I26M1D34M4D44M1I7M2I22M2I12M1D10M1D4M1D11M1D103M1I20M1I20M1D16M1D8M1D98M2D50M16D12M2D9M1D63M1D11M1D76M2I33M5D25M3D63M1D18M1I7M2D31M1D21M1D7M2D15M1I82M1D59M1I66M1I13M1D73M1D8M1D13M2D11M1I7M1I65M1I31M1D8M1D18M1I72M1I31M1I191M1D80M1D9M1D65M13D81M1I90M1D5M1I4M1D2M1D6M1D49M1D15M1D31M1D5M1I33M2I74M1D10M3I46M3I22M2D59M1D5M1D84M2I2M1D13M1I18M2D5M2I3M2I2M20I117M1I13M5D41M2D1M1D16M1S",
      "length": 117851,
      "mapping_quality": 60,
      "sequence_first100": "TTACTTCGTTCAGTTACGTATTGCTTTGAAGGCTGGGCTACCTCAGTGTGGAAGGTGGGCAGTTCCACAGGAATGGTGCCAGGGCCAGAGGGCAATACCA",
      "tags": [
        [
          "NM",
          7874
        ],
        [
          "ms",
          151392
        ],
        [
          "AS",
          152519
        ],
        [
          "nn",
          725
        ],
        [
          "tp",
          "P"
        ],
        [
          "cm",
          8416
        ],
        [
          "s1",
          58736
        ],
        [
          "s2",
          625
        ],
        [
          "dv",
          0.041600000113248825
        ],
        [
          "SA",
          "chr12,16970,+,21S18762M358D99068S,52,2368;chr3,76029363,+,18278S3125M142I96306S,1,555;chr19,75654,+,21822S1087M10D94942S,0,83;"
        ]
      ]
    },
    "source": "official Clair3 ONT quick demo HG003 Guppy3.6 ~85x; ONT non-dwell model ont",
    "truth": "GIAB HG003 GRCh38 chr20 v4.2.1 VCF and confident BED"
  },
  "model_only_postload": {
    "allocated_gib": 0.01114511489868164,
    "reserved_gib": 0.025390625,
    "peak_allocated_gib": 0.01114511489868164,
    "peak_reserved_gib": 0.025390625,
    "rss_gib": 0.6190414428710938,
    "available_gib": 113.02769470214844,
    "system_available_gib": 113.02769470214844,
    "system_total_gib": 121.68751525878906
  },
  "parameters": {
    "Clair3_F": 2986522,
    "Clair3_P": 2074520
  },
  "reload": [
    {
      "model": "Clair3_F",
      "actual_input_shape": [
        32,
        89,
        33,
        8
      ],
      "independent_process": true,
      "max_abs_difference": 0.0,
      "passed": true,
      "original_batch_shape": [
        181,
        89,
        33,
        8
      ],
      "note": "first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4"
    },
    {
      "model": "Clair3_P",
      "actual_input_shape": [
        32,
        33,
        18
      ],
      "independent_process": true,
      "max_abs_difference": 0.0,
      "passed": true,
      "original_batch_shape": [
        103,
        33,
        18
      ],
      "note": "first32 actual candidates retained; other independent batch rows zero-filled only for reload check, excluded from performance measurement; original batch shape fixed; FP32 tolerance1e-4"
    }
  ]
}
```
<!-- END_RESTORE_FILE -->


</details>
