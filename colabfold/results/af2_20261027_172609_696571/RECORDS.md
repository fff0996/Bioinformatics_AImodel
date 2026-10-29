# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-d891102ff9d3

원래 경로: `colabfold/results/af2_20261027_172609_696571/command.json` · SHA256: `0f3d43ff40cc0197ce7c2f1973d7fd25e24d62e4d3aa93318391e7072ebe21d3`

<details>
<summary>command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172609_696571/command.json", "sha256": "0f3d43ff40cc0197ce7c2f1973d7fd25e24d62e4d3aa93318391e7072ebe21d3", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --name af2_20261027_172609_696571 --gpus all --network none --user 1000:1000 -e JAX_PLATFORMS=cuda -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e MPLCONFIGDIR=/work/results/af2_20261027_172609_696571/mplcache -v /home/hyein/ai-benchmark/colabfold:/work -v /home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 python -u /work/scripts/benchmark_af2.py af2_20261027_172609_696571 msa_20261027_172359_775627"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-98aa2cbf1221

원래 경로: `colabfold/results/af2_20261027_172609_696571/exit_code.json` · SHA256: `0ffa99ba8d9353f2f7c12ed7789b2a0eb6737813c19a6fa9e0fc123c4c3fa2d5`

<details>
<summary>exit_code.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172609_696571/exit_code.json", "sha256": "0ffa99ba8d9353f2f7c12ed7789b2a0eb6737813c19a6fa9e0fc123c4c3fa2d5", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "exit_code": 1,
  "start_utc": "2026-10-27T08:26:09.696861+00:00",
  "end_utc": "2026-10-27T08:26:18.359902+00:00"
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-5408125f0590

원래 경로: `colabfold/results/af2_20261027_172609_696571/input_report.json` · SHA256: `bd3245f28e156f683af563f24049a2fbde69c6da5bf0287cde7db53f2c4b70aa`

<details>
<summary>input_report.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172609_696571/input_report.json", "sha256": "bd3245f28e156f683af563f24049a2fbde69c6da5bf0287cde7db53f2c4b70aa", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "id": "1UBQ",
    "residues": 76,
    "msa_depth": 21577,
    "msa_aligned_columns": 76,
    "msa_characters": [
      "-",
      "A",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "K",
      "L",
      "M",
      "N",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "V",
      "W",
      "X",
      "Y"
    ],
    "msa_sha256": "91df444d5eb1acf30c9bde7f43495f1191e8452191924c178e2e6161a056a049",
    "msa_bytes": 3001455,
    "token_dtype": "int32 features",
    "truncated": false,
    "templates": false
  },
  {
    "id": "count1_A2RU14",
    "residues": 115,
    "msa_depth": 568,
    "msa_aligned_columns": 115,
    "msa_characters": [
      "-",
      "A",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "K",
      "L",
      "M",
      "N",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "V",
      "W",
      "X",
      "Y"
    ],
    "msa_sha256": "65b27286d30511570201072127c6c4bf4deb6d09d4b4f1fbc948e7d7faaaf5ea",
    "msa_bytes": 101931,
    "token_dtype": "int32 features",
    "truncated": false,
    "templates": false
  },
  {
    "id": "count2_A6NFY7",
    "residues": 115,
    "msa_depth": 2382,
    "msa_aligned_columns": 115,
    "msa_characters": [
      "-",
      "A",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "K",
      "L",
      "M",
      "N",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "V",
      "W",
      "X",
      "Y"
    ],
    "msa_sha256": "50133c0bebf212ec0461ad8d17aa6b90256b188aec35e7b01738c23761643560",
    "msa_bytes": 426524,
    "token_dtype": "int32 features",
    "truncated": false,
    "templates": false
  },
  {
    "id": "count3_O14519",
    "residues": 115,
    "msa_depth": 1295,
    "msa_aligned_columns": 115,
    "msa_characters": [
      "-",
      "A",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "K",
      "L",
      "M",
      "N",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "V",
      "W",
      "X",
      "Y"
    ],
    "msa_sha256": "eda7682338958cb5814920cb653b6c7b32d974298f72fada8d1100f371bfc76d",
    "msa_bytes": 233005,
    "token_dtype": "int32 features",
    "truncated": false,
    "templates": false
  },
  {
    "id": "count4_P01270",
    "residues": 115,
    "msa_depth": 971,
    "msa_aligned_columns": 115,
    "msa_characters": [
      "-",
      "A",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "K",
      "L",
      "M",
      "N",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "V",
      "W",
      "X",
      "Y"
    ],
    "msa_sha256": "b08538aaaf14fecf190c2233a0e9adc1e887b79cef16364fc0c0134d14af6c53",
    "msa_bytes": 174827,
    "token_dtype": "int32 features",
    "truncated": false,
    "templates": false
  },
  {
    "id": "1LYZ",
    "residues": 129,
    "msa_depth": 5508,
    "msa_aligned_columns": 129,
    "msa_characters": [
      "-",
      "A",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "K",
      "L",
      "M",
      "N",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "V",
      "W",
      "X",
      "Y"
    ],
    "msa_sha256": "b507eac565c22c29e71aeb6db6280c372dcb493b3bc8a834ff9bd44bdfdbff54",
    "msa_bytes": 1074611,
    "token_dtype": "int32 features",
    "truncated": false,
    "templates": false
  },
  {
    "id": "1TIM",
    "residues": 247,
    "msa_depth": 16469,
    "msa_aligned_columns": 247,
    "msa_characters": [
      "-",
      "A",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "K",
      "L",
      "M",
      "N",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "V",
      "W",
      "X",
      "Y"
    ],
    "msa_sha256": "7cb301e50dc423142584a60ac65cbf3b8275d0761477a39acd0ddcc35831d3c0",
    "msa_bytes": 5232605,
    "token_dtype": "int32 features",
    "truncated": false,
    "templates": false
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-eb0a006d15e3

원래 경로: `colabfold/results/af2_20261027_172609_696571/log.txt` · SHA256: `0939cf8d25f5cdc7f950130787bd67b29e7ba564f2e9294d6830711ef34a90e5`

<details>
<summary>log.txt — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172609_696571/log.txt", "sha256": "0939cf8d25f5cdc7f950130787bd67b29e7ba564f2e9294d6830711ef34a90e5", "encoding": "utf-8", "added_final_newline": false} -->
```txt
2026-10-27 08:26:11,063 Running on GPU
2026-10-27 08:26:11,065 Found 5 citations for tools or databases
2026-10-27 08:26:11,065 Query 1/7: 1UBQ (length 76)
2026-10-27 08:26:11,525 generated new fontManager
2026-10-27 08:26:15,612 Could not predict 1UBQ. Not Enough GPU memory? Unknown backend cpu. Available backends are ['cuda']
2026-10-27 08:26:15,612 Query 2/7: count1_A2RU14 (length 115)
2026-10-27 08:26:15,772 Could not predict count1_A2RU14. Not Enough GPU memory? Unknown backend cpu. Available backends are ['cuda']
2026-10-27 08:26:15,772 Query 3/7: count2_A6NFY7 (length 115)
2026-10-27 08:26:16,022 Could not predict count2_A6NFY7. Not Enough GPU memory? Unknown backend cpu. Available backends are ['cuda']
2026-10-27 08:26:16,022 Query 4/7: count3_O14519 (length 115)
2026-10-27 08:26:16,203 Could not predict count3_O14519. Not Enough GPU memory? Unknown backend cpu. Available backends are ['cuda']
2026-10-27 08:26:16,203 Query 5/7: count4_P01270 (length 115)
2026-10-27 08:26:16,373 Could not predict count4_P01270. Not Enough GPU memory? Unknown backend cpu. Available backends are ['cuda']
2026-10-27 08:26:16,373 Query 6/7: 1LYZ (length 129)
2026-10-27 08:26:16,701 Could not predict 1LYZ. Not Enough GPU memory? Unknown backend cpu. Available backends are ['cuda']
2026-10-27 08:26:16,701 Query 7/7: 1TIM (length 247)
2026-10-27 08:26:17,786 Could not predict 1TIM. Not Enough GPU memory? Unknown backend cpu. Available backends are ['cuda']
2026-10-27 08:26:17,787 Done
```
<!-- END_RESTORE_FILE -->


</details>

## file-b983c645d1e4

원래 경로: `colabfold/results/af2_20261027_172609_696571/resources.csv` · SHA256: `95e23da7f17f3c961c8772b6605bb09f1fd9c8f67767372701fb4859407abaa2`

<details>
<summary>resources.csv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172609_696571/resources.csv", "sha256": "95e23da7f17f3c961c8772b6605bb09f1fd9c8f67767372701fb4859407abaa2", "encoding": "utf-8", "added_final_newline": false} -->
```csv
query_start,query_end,query,phase,gpu_util_percent,rss_bytes,system_available_bytes,swap_used_bytes
1793089570.7929766,1793089570.8608603,setup,setup,5.0,816222208,122783629312,0
1793089571.298256,1793089571.3504033,setup,preprocessing,5.0,851439616,122733486080,0
1793089571.7988915,1793089571.8199434,setup,pipeline,5.0,912650240,122542977024,0
1793089572.2989936,1793089572.3208535,setup,model_loading,7.0,1143734272,122373279744,0
1793089572.799176,1793089572.8190033,setup,model_loading,7.0,1503674368,122008477696,0
1793089573.299344,1793089573.3198416,setup,model_loading,6.0,2320842752,121190596608,0
1793089573.7995386,1793089573.820753,setup,model_loading,6.0,2617839616,120891637760,0
1793089574.2996342,1793089574.321822,setup,model_loading,35.0,3005079552,120501088256,0
1793089574.7998416,1793089574.8238149,setup,model_loading,35.0,3300888576,120206774272,0
1793089575.299943,1793089575.322556,setup,model_loading,10.0,3763220480,119727063040,0
1793089575.8054552,1793089575.8730536,count1_A2RU14,preprocessing,10.0,3879636992,119545102336,0
1793089576.3055534,1793089576.3266733,count3_O14519,preprocessing,5.0,3879641088,119525855232,0
1793089576.811541,1793089576.9604127,1LYZ,preprocessing,5.0,3881631744,119532134400,0
1793089577.324011,1793089577.3916044,1LYZ,preprocessing,5.0,3942776832,119459385344,0
```
<!-- END_RESTORE_FILE -->


</details>

## file-514e27bffd38

원래 경로: `colabfold/results/af2_20261027_172609_696571/summary.json` · SHA256: `b8e8c44e2079ade1ce413d2e57159c17a3406f87be7c3f669e7ba9c0e44cc8ec`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172609_696571/summary.json", "sha256": "b8e8c44e2079ade1ce413d2e57159c17a3406f87be7c3f669e7ba9c0e44cc8ec", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "failed",
  "run": "/work/results/af2_20261027_172609_696571",
  "settings": {
    "model_type": "alphafold2_ptm",
    "num_models": 5,
    "model_order": [
      1,
      2,
      3,
      4,
      5
    ],
    "num_recycles": 3,
    "recycle_early_stop_tolerance": 0.0,
    "num_ensemble": 1,
    "num_seeds": 1,
    "random_seed": 42,
    "use_templates": false,
    "num_relax": 0,
    "dropout": false,
    "msa_mode": "mmseqs2_uniref_env",
    "max_seq": 512,
    "max_extra_seq": 5120,
    "use_bfloat16": true,
    "use_pallas": false,
    "compile_mode": "tuned",
    "recompile_padding": 10,
    "batch": "one protein at a time, five weight sets sequentially",
    "preallocation": false
  },
  "predictions": [],
  "loaded_models": [
    {
      "model": "model_1",
      "parameters": 92907930,
      "parameter_dtypes": [
        "float32"
      ]
    },
    {
      "model": "model_2",
      "parameters": 92907930,
      "parameter_dtypes": [
        "float32"
      ]
    },
    {
      "model": "model_3",
      "parameters": 92907930,
      "parameter_dtypes": [
        "float32"
      ]
    },
    {
      "model": "model_4",
      "parameters": 92907930,
      "parameter_dtypes": [
        "float32"
      ]
    },
    {
      "model": "model_5",
      "parameters": 92907930,
      "parameter_dtypes": [
        "float32"
      ]
    }
  ],
  "model_loading_seconds": 3.6904985750006745,
  "pipeline_wall_seconds": 6.72560959600014,
  "error": "(0, 7)",
  "traceback": "Traceback (most recent call last):\n  File \"/work/scripts/benchmark_af2.py\", line 111, in <module>\n    summary['pipeline_wall_seconds']=time.perf_counter()-start;assert len(predictions)==5*len(queries),(len(predictions),len(queries));summary['status']='success';summary['gpu_apply_seconds']=sum(c['seconds'] for c in calls);summary['compilation_seconds']=sum(e['seconds'] for e in events if e['kind']=='compilation');summary['final_device_memory']=dmem();summary['wall_seconds']=time.perf_counter()-allstart;write('summary.json',summary);write('apply_calls.json',calls);event('success',summary_path=str(O/'summary.json'))\n                                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nAssertionError: (0, 7)\n"
}
```
<!-- END_RESTORE_FILE -->


</details>
