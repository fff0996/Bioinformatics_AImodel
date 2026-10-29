# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-db67334301c8

원래 경로: `colabfold/results/vmap_batch_20261028_185758_495006/batch_1/resources.csv` · SHA256: `e305366d32b4cd406d4e321b4c596d4deb199958fe42d9346c9435cdf9ef0425`

<details>
<summary>resources.csv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/batch_1/resources.csv", "sha256": "e305366d32b4cd406d4e321b4c596d4deb199958fe42d9346c9435cdf9ef0425", "encoding": "utf-8", "added_final_newline": false} -->
```csv
query_start,query_end,phase,gpu_util,rss_gib,available_gib,swap_gib,jax_bytes_in_use
1793181891.3527555,1793181891.3721619,loading,3.0,0.7529983520507812,113.27378463745117,0.00743865966796875,0
1793181891.8728304,1793181891.8927574,loading,3.0,1.252288818359375,112.75790023803711,0.00743865966796875,28763392
1793181892.3957,1793181892.4177644,loading,3.0,1.3835411071777344,112.09453582763672,0.00743865966796875,385484288
1793181892.9181588,1793181892.937406,loading,2.0,1.4644889831542969,111.47036361694336,0.00743865966796875,761517312
1793181893.4375455,1793181893.4602084,loading,2.0,1.4962577819824219,110.3868293762207,0.00743865966796875,1556313344
1793181893.9605465,1793181893.980913,loading,4.0,1.5148429870605469,110.33793258666992,0.00743865966796875,2069435904
1793181894.4810598,1793181894.5050273,loading,4.0,1.5868453979492188,108.19768142700195,0.00743865966796875,443757568
1793181895.0920997,1793181895.5679908,compilation,2.0,1.632476806640625,108.16742706298828,0.00743865966796875,461792768
```
<!-- END_RESTORE_FILE -->


</details>

## file-24853f59a3c1

원래 경로: `colabfold/results/vmap_batch_20261028_185758_495006/batch_1/summary.json` · SHA256: `648bbf91c3d148888dbb679892308ad6d5a2e9037d6e94d100bfe228c1928925`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/batch_1/summary.json", "sha256": "648bbf91c3d148888dbb679892308ad6d5a2e9037d6e94d100bfe228c1928925", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "failed",
  "batch": 1,
  "implementation": "experimental jax.vmap over unmodified ColabFold RunModel.apply; shared fixed model_1 weights, independent actual query features and RNG; not a native CLI batch option",
  "conditions": [],
  "input": {
    "unique_queries": 1,
    "query_ids": [
      "count1_A2RU14"
    ],
    "actual_residues": [
      115
    ],
    "feature_shapes_per_query": {
      "aatype": [
        4,
        115
      ],
      "residue_index": [
        4,
        115
      ],
      "seq_length": [
        4
      ],
      "is_distillation": [
        4
      ],
      "seq_mask": [
        4,
        115
      ],
      "msa_mask": [
        4,
        512,
        115
      ],
      "msa_row_mask": [
        4,
        512
      ],
      "random_crop_to_size_seed": [
        4,
        2
      ],
      "atom14_atom_exists": [
        4,
        115,
        14
      ],
      "residx_atom14_to_atom37": [
        4,
        115,
        14
      ],
      "residx_atom37_to_atom14": [
        4,
        115,
        37
      ],
      "atom37_atom_exists": [
        4,
        115,
        37
      ],
      "extra_msa": [
        4,
        1869,
        115
      ],
      "extra_msa_mask": [
        4,
        1869,
        115
      ],
      "extra_msa_row_mask": [
        4,
        1869
      ],
      "bert_mask": [
        4,
        512,
        115
      ],
      "true_msa": [
        4,
        512,
        115
      ],
      "extra_has_deletion": [
        4,
        1869,
        115
      ],
      "extra_deletion_value": [
        4,
        1869,
        115
      ],
      "msa_feat": [
        4,
        512,
        115,
        49
      ],
      "target_feat": [
        4,
        115,
        22
      ],
      "asym_id": [
        4,
        115
      ]
    },
    "padding": "pool-wide max feature shapes; zero-padded extra-MSA/cluster entries masked by zero masks; no sequence duplication",
    "actual_msa_valid_rows": [
      512
    ]
  },
  "loading_seconds": 2.6969761848449707,
  "parameters": 92907930,
  "weights_dtype": [
    "float32"
  ],
  "model_only_postload": {
    "jax": {
      "num_allocs": 3600,
      "bytes_in_use": 372437504,
      "peak_bytes_in_use": 2314587904,
      "largest_alloc_size": 99614720,
      "bytes_limit": 97997815808,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    },
    "system": {
      "rss_gib": 1.5804710388183594,
      "available_gib": 108.20450210571289,
      "total_gib": 121.68751525878906,
      "swap_gib": 0.00743865966796875
    }
  },
  "environment": {
    "jax": "0.10.2",
    "device": "cuda:0",
    "preallocate": "false",
    "compute": "original BF16 config, FP32 weights",
    "recycles": 3,
    "ensembles": 1,
    "templates": false,
    "relaxation": false,
    "seed_per_query": 42
  },
  "error": "ValueError(\"`compiler_options` can only be passed to top-level `jax.jit`. Got compiler_options={'xla_gpu_override_gemm_autotuner': 'block_m: 64 block_n: 64 block_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1'} specified on a nested jit with name: apply_fn and source info: \")"
}
```
<!-- END_RESTORE_FILE -->


</details>
