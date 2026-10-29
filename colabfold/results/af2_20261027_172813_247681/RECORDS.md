# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-13957c07372d

원래 경로: `colabfold/results/af2_20261027_172813_247681/aggregated_metrics.json` · SHA256: `7eccb49ae5973915b96b1a695c656be1b4ebce7631c01f7def2630dbf340a014`

<details>
<summary>aggregated_metrics.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/aggregated_metrics.json", "sha256": "7eccb49ae5973915b96b1a695c656be1b4ebce7631c01f7def2630dbf340a014", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "run": "af2_20261027_172813_247681",
    "query": "1UBQ",
    "unique_inputs": 1,
    "total_processed_proteins": 1,
    "predicted_models": 5,
    "gpu_apply_calls": 20,
    "original_residues": 76,
    "actual_residues": 76,
    "padded_residue_axis": 86,
    "original_a3m_rows": 21577,
    "actual_feature_msa_shape": [
      4,
      512,
      86
    ],
    "extra_msa_shape": [
      4,
      5120,
      86
    ],
    "truncated": false,
    "compute_seconds": 33.6369067659989,
    "compilation_seconds": 8.366444768000292,
    "query_wall_seconds": 43.25274384499971,
    "protein_throughput_per_second": 0.029729249688643402,
    "model_throughput_per_second": 0.148646248443217,
    "gpu_util_mean_percent": 91.84615384615384,
    "gpu_util_max_percent": 94.0,
    "sample_count": 65,
    "rss_sampled_peak_gib": 4.9677734375,
    "available_min_gib": 109.90876770019531,
    "swap_max_bytes": 0,
    "jax_allocator_peak_gib_since_process_start": 2.165952205657959,
    "jax_live_bytes_max_at_apply_return_gib": 1.7936582565307617,
    "query_start": 1793089697.9430606,
    "query_end": 1793089741.1958034
  },
  {
    "run": "af2_20261027_172813_247681",
    "query": "count1_A2RU14",
    "unique_inputs": 1,
    "total_processed_proteins": 1,
    "predicted_models": 5,
    "gpu_apply_calls": 20,
    "original_residues": 115,
    "actual_residues": 115,
    "padded_residue_axis": 125,
    "original_a3m_rows": 568,
    "actual_feature_msa_shape": [
      4,
      512,
      125
    ],
    "extra_msa_shape": [
      4,
      5120,
      125
    ],
    "truncated": false,
    "compute_seconds": 49.70025126600103,
    "compilation_seconds": 7.710386016000484,
    "query_wall_seconds": 58.28870832299981,
    "protein_throughput_per_second": 0.02012062262317133,
    "model_throughput_per_second": 0.10060311311585667,
    "gpu_util_mean_percent": 92.88421052631578,
    "gpu_util_max_percent": 95.0,
    "sample_count": 95,
    "rss_sampled_peak_gib": 5.419635772705078,
    "available_min_gib": 109.22414779663086,
    "swap_max_bytes": 0,
    "jax_allocator_peak_gib_since_process_start": 2.273073673248291,
    "jax_live_bytes_max_at_apply_return_gib": 1.799004316329956,
    "query_start": 1793089741.6190722,
    "query_end": 1793089799.9077797
  },
  {
    "run": "af2_20261027_172813_247681",
    "query": "count2_A6NFY7",
    "unique_inputs": 1,
    "total_processed_proteins": 1,
    "predicted_models": 5,
    "gpu_apply_calls": 20,
    "original_residues": 115,
    "actual_residues": 115,
    "padded_residue_axis": 125,
    "original_a3m_rows": 2382,
    "actual_feature_msa_shape": [
      4,
      512,
      125
    ],
    "extra_msa_shape": [
      4,
      5120,
      125
    ],
    "truncated": false,
    "compute_seconds": 49.61643685000399,
    "compilation_seconds": 0,
    "query_wall_seconds": 50.44888227699994,
    "protein_throughput_per_second": 0.02015461132412856,
    "model_throughput_per_second": 0.1007730566206428,
    "gpu_util_mean_percent": 92.82,
    "gpu_util_max_percent": 94.0,
    "sample_count": 100,
    "rss_sampled_peak_gib": 5.575614929199219,
    "available_min_gib": 109.49588394165039,
    "swap_max_bytes": 0,
    "jax_allocator_peak_gib_since_process_start": 2.273073673248291,
    "jax_live_bytes_max_at_apply_return_gib": 1.799004316329956,
    "query_start": 1793089800.548487,
    "query_end": 1793089850.9973686
  },
  {
    "run": "af2_20261027_172813_247681",
    "query": "count3_O14519",
    "unique_inputs": 1,
    "total_processed_proteins": 1,
    "predicted_models": 5,
    "gpu_apply_calls": 20,
    "original_residues": 115,
    "actual_residues": 115,
    "padded_residue_axis": 125,
    "original_a3m_rows": 1295,
    "actual_feature_msa_shape": [
      4,
      512,
      125
    ],
    "extra_msa_shape": [
      4,
      5120,
      125
    ],
    "truncated": false,
    "compute_seconds": 47.709969164999166,
    "compilation_seconds": 0,
    "query_wall_seconds": 48.58623763699961,
    "protein_throughput_per_second": 0.020959980010500968,
    "model_throughput_per_second": 0.10479990005250484,
    "gpu_util_mean_percent": 91.54838709677419,
    "gpu_util_max_percent": 95.0,
    "sample_count": 93,
    "rss_sampled_peak_gib": 5.603511810302734,
    "available_min_gib": 109.13208770751953,
    "swap_max_bytes": 0,
    "jax_allocator_peak_gib_since_process_start": 2.273073673248291,
    "jax_live_bytes_max_at_apply_return_gib": 1.799004316329956,
    "query_start": 1793089851.526802,
    "query_end": 1793089900.1130393
  },
  {
    "run": "af2_20261027_172813_247681",
    "query": "count4_P01270",
    "unique_inputs": 1,
    "total_processed_proteins": 1,
    "predicted_models": 5,
    "gpu_apply_calls": 20,
    "original_residues": 115,
    "actual_residues": 115,
    "padded_residue_axis": 125,
    "original_a3m_rows": 971,
    "actual_feature_msa_shape": [
      4,
      512,
      125
    ],
    "extra_msa_shape": [
      4,
      5120,
      125
    ],
    "truncated": false,
    "compute_seconds": 47.64036477599984,
    "compilation_seconds": 0,
    "query_wall_seconds": 48.568828466000014,
    "protein_throughput_per_second": 0.020990603340295535,
    "model_throughput_per_second": 0.10495301670147768,
    "gpu_util_mean_percent": 91.3578947368421,
    "gpu_util_max_percent": 95.0,
    "sample_count": 95,
    "rss_sampled_peak_gib": 5.583984375,
    "available_min_gib": 109.2553596496582,
    "swap_max_bytes": 0,
    "jax_allocator_peak_gib_since_process_start": 2.273073673248291,
    "jax_live_bytes_max_at_apply_return_gib": 1.799004316329956,
    "query_start": 1793089900.6306448,
    "query_end": 1793089949.1994724
  },
  {
    "run": "af2_20261027_172813_247681",
    "query": "1LYZ",
    "unique_inputs": 1,
    "total_processed_proteins": 1,
    "predicted_models": 5,
    "gpu_apply_calls": 20,
    "original_residues": 129,
    "actual_residues": 129,
    "padded_residue_axis": 139,
    "original_a3m_rows": 5508,
    "actual_feature_msa_shape": [
      4,
      512,
      139
    ],
    "extra_msa_shape": [
      4,
      5120,
      139
    ],
    "truncated": false,
    "compute_seconds": 59.394848060002005,
    "compilation_seconds": 9.291720894999344,
    "query_wall_seconds": 69.70946580100008,
    "protein_throughput_per_second": 0.016836477113129033,
    "model_throughput_per_second": 0.08418238556564515,
    "gpu_util_mean_percent": 93.325,
    "gpu_util_max_percent": 95.0,
    "sample_count": 120,
    "rss_sampled_peak_gib": 5.883522033691406,
    "available_min_gib": 108.88226318359375,
    "swap_max_bytes": 0,
    "jax_allocator_peak_gib_since_process_start": 2.307253360748291,
    "jax_live_bytes_max_at_apply_return_gib": 1.8042609691619873,
    "query_start": 1793089949.8723326,
    "query_end": 1793090019.5817976
  },
  {
    "run": "af2_20261027_172813_247681",
    "query": "1TIM",
    "unique_inputs": 1,
    "total_processed_proteins": 1,
    "predicted_models": 5,
    "gpu_apply_calls": 20,
    "original_residues": 247,
    "actual_residues": 247,
    "padded_residue_axis": 247,
    "original_a3m_rows": 16469,
    "actual_feature_msa_shape": [
      4,
      512,
      247
    ],
    "extra_msa_shape": [
      4,
      5120,
      247
    ],
    "truncated": false,
    "compute_seconds": 114.7147885530012,
    "compilation_seconds": 11.483351171000322,
    "query_wall_seconds": 127.51820730599866,
    "protein_throughput_per_second": 0.008717271875874785,
    "model_throughput_per_second": 0.04358635937937392,
    "gpu_util_mean_percent": 93.66814159292035,
    "gpu_util_max_percent": 95.0,
    "sample_count": 226,
    "rss_sampled_peak_gib": 6.861560821533203,
    "available_min_gib": 107.21379089355469,
    "swap_max_bytes": 0,
    "jax_allocator_peak_gib_since_process_start": 2.705740213394165,
    "jax_live_bytes_max_at_apply_return_gib": 1.836754560470581,
    "query_start": 1793090021.0439842,
    "query_end": 1793090148.5621908
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-2b04da4121ac

원래 경로: `colabfold/results/af2_20261027_172813_247681/apply_calls.json` · SHA256: `0e9320f4d1c11cfeda74382107090e6198f7b4bd05405461adb8b9085779d86d`

<details>
<summary>apply_calls.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/apply_calls.json", "sha256": "0e9320f4d1c11cfeda74382107090e6198f7b4bd05405461adb8b9085779d86d", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "query": "1UBQ",
    "start_unix": 1793089707.0367782,
    "end_unix": 1793089708.7980728,
    "seconds": 1.761290886000097,
    "device_memory": {
      "num_allocs": 6579,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089708.8272276,
    "end_unix": 1793089710.4853253,
    "seconds": 1.6580952750000506,
    "device_memory": {
      "num_allocs": 6616,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089710.5033252,
    "end_unix": 1793089712.1974175,
    "seconds": 1.6940893780001716,
    "device_memory": {
      "num_allocs": 6653,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089712.2111228,
    "end_unix": 1793089713.86504,
    "seconds": 1.6539154270003564,
    "device_memory": {
      "num_allocs": 6690,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089713.8943553,
    "end_unix": 1793089715.566145,
    "seconds": 1.67178795899963,
    "device_memory": {
      "num_allocs": 6730,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089715.589507,
    "end_unix": 1793089717.2709172,
    "seconds": 1.6814077039998665,
    "device_memory": {
      "num_allocs": 6767,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089717.2996452,
    "end_unix": 1793089718.9560034,
    "seconds": 1.6563556809996953,
    "device_memory": {
      "num_allocs": 6804,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089718.9762714,
    "end_unix": 1793089720.6817646,
    "seconds": 1.705490787000599,
    "device_memory": {
      "num_allocs": 6841,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089720.716,
    "end_unix": 1793089722.3752782,
    "seconds": 1.659274717999324,
    "device_memory": {
      "num_allocs": 6881,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089722.4049957,
    "end_unix": 1793089724.13174,
    "seconds": 1.7267421140004444,
    "device_memory": {
      "num_allocs": 6918,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089724.1581116,
    "end_unix": 1793089725.825866,
    "seconds": 1.667750769000122,
    "device_memory": {
      "num_allocs": 6955,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089725.8492224,
    "end_unix": 1793089727.5307596,
    "seconds": 1.6815347159999874,
    "device_memory": {
      "num_allocs": 6992,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089727.5672226,
    "end_unix": 1793089729.2615576,
    "seconds": 1.6943323750001582,
    "device_memory": {
      "num_allocs": 7032,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089729.291419,
    "end_unix": 1793089730.9507198,
    "seconds": 1.6592981439998766,
    "device_memory": {
      "num_allocs": 7069,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089730.9722219,
    "end_unix": 1793089732.6799893,
    "seconds": 1.707764947999749,
    "device_memory": {
      "num_allocs": 7106,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089732.7157476,
    "end_unix": 1793089734.3764153,
    "seconds": 1.6606642389997432,
    "device_memory": {
      "num_allocs": 7143,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089734.408656,
    "end_unix": 1793089736.1005275,
    "seconds": 1.6918693429997802,
    "device_memory": {
      "num_allocs": 7183,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089736.1289396,
    "end_unix": 1793089737.7782364,
    "seconds": 1.6492945779991715,
    "device_memory": {
      "num_allocs": 7220,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089737.7966106,
    "end_unix": 1793089739.472123,
    "seconds": 1.6755105539996293,
    "device_memory": {
      "num_allocs": 7257,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1UBQ",
    "start_unix": 1793089739.4883351,
    "end_unix": 1793089741.1687758,
    "seconds": 1.6804371710004489,
    "device_memory": {
      "num_allocs": 7294,
      "bytes_in_use": 1925925888,
      "peak_bytes_in_use": 2325673472,
      "largest_alloc_size": 322914560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089749.6604185,
    "end_unix": 1793089752.225559,
    "seconds": 2.565137978000166,
    "device_memory": {
      "num_allocs": 9587,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089752.2603953,
    "end_unix": 1793089754.7052264,
    "seconds": 2.444828843000323,
    "device_memory": {
      "num_allocs": 9624,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089754.7364938,
    "end_unix": 1793089757.2347274,
    "seconds": 2.498231608000424,
    "device_memory": {
      "num_allocs": 9661,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089757.2639468,
    "end_unix": 1793089759.6443264,
    "seconds": 2.380378419000408,
    "device_memory": {
      "num_allocs": 9698,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089759.6663058,
    "end_unix": 1793089762.1197646,
    "seconds": 2.4534558880004624,
    "device_memory": {
      "num_allocs": 9738,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089762.1440992,
    "end_unix": 1793089764.6495783,
    "seconds": 2.5054768159998275,
    "device_memory": {
      "num_allocs": 9775,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089764.6672451,
    "end_unix": 1793089767.1244776,
    "seconds": 2.457230652999897,
    "device_memory": {
      "num_allocs": 9812,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089767.1575494,
    "end_unix": 1793089769.6748395,
    "seconds": 2.517288911999458,
    "device_memory": {
      "num_allocs": 9849,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089769.7011561,
    "end_unix": 1793089772.2015865,
    "seconds": 2.500427659999332,
    "device_memory": {
      "num_allocs": 9889,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089772.2307322,
    "end_unix": 1793089774.6706822,
    "seconds": 2.439947435999784,
    "device_memory": {
      "num_allocs": 9926,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089774.6892843,
    "end_unix": 1793089777.2058332,
    "seconds": 2.5165471250002156,
    "device_memory": {
      "num_allocs": 9963,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089777.2296736,
    "end_unix": 1793089779.7145133,
    "seconds": 2.484838151999611,
    "device_memory": {
      "num_allocs": 10000,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089779.7432914,
    "end_unix": 1793089782.2319021,
    "seconds": 2.488608867000039,
    "device_memory": {
      "num_allocs": 10040,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089782.2624137,
    "end_unix": 1793089784.7649498,
    "seconds": 2.5025327659996037,
    "device_memory": {
      "num_allocs": 10077,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089784.7969422,
    "end_unix": 1793089787.257922,
    "seconds": 2.460977329999878,
    "device_memory": {
      "num_allocs": 10114,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089787.274857,
    "end_unix": 1793089789.796302,
    "seconds": 2.521442611000566,
    "device_memory": {
      "num_allocs": 10151,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089789.8284097,
    "end_unix": 1793089792.3429813,
    "seconds": 2.5145698390006146,
    "device_memory": {
      "num_allocs": 10191,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089792.3723786,
    "end_unix": 1793089794.840984,
    "seconds": 2.4686033369998768,
    "device_memory": {
      "num_allocs": 10228,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089794.8676665,
    "end_unix": 1793089797.3587437,
    "seconds": 2.49107475800065,
    "device_memory": {
      "num_allocs": 10265,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count1_A2RU14",
    "start_unix": 1793089797.38721,
    "end_unix": 1793089799.875866,
    "seconds": 2.488652267999896,
    "device_memory": {
      "num_allocs": 10302,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089800.808537,
    "end_unix": 1793089803.2706742,
    "seconds": 2.4621333500008404,
    "device_memory": {
      "num_allocs": 10342,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089803.3096154,
    "end_unix": 1793089805.8120825,
    "seconds": 2.502463288999934,
    "device_memory": {
      "num_allocs": 10379,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089805.839845,
    "end_unix": 1793089808.3577292,
    "seconds": 2.51788190499974,
    "device_memory": {
      "num_allocs": 10416,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089808.391939,
    "end_unix": 1793089810.8407776,
    "seconds": 2.448836720000145,
    "device_memory": {
      "num_allocs": 10453,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089810.870778,
    "end_unix": 1793089813.3766878,
    "seconds": 2.5059071430005133,
    "device_memory": {
      "num_allocs": 10493,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089813.3976135,
    "end_unix": 1793089815.8792953,
    "seconds": 2.481679389999954,
    "device_memory": {
      "num_allocs": 10530,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089815.9026144,
    "end_unix": 1793089818.3715448,
    "seconds": 2.4689280829998097,
    "device_memory": {
      "num_allocs": 10567,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089818.3933847,
    "end_unix": 1793089820.9036083,
    "seconds": 2.5102212350002446,
    "device_memory": {
      "num_allocs": 10604,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089820.931894,
    "end_unix": 1793089823.41257,
    "seconds": 2.4806734590001724,
    "device_memory": {
      "num_allocs": 10644,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089823.444157,
    "end_unix": 1793089825.939963,
    "seconds": 2.4958039160001135,
    "device_memory": {
      "num_allocs": 10681,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089825.9626808,
    "end_unix": 1793089828.4772806,
    "seconds": 2.514597295000385,
    "device_memory": {
      "num_allocs": 10718,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089828.5002897,
    "end_unix": 1793089830.9569962,
    "seconds": 2.456704043999707,
    "device_memory": {
      "num_allocs": 10755,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089830.9928548,
    "end_unix": 1793089833.4620602,
    "seconds": 2.4692034170002444,
    "device_memory": {
      "num_allocs": 10795,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089833.4977741,
    "end_unix": 1793089835.920932,
    "seconds": 2.423155714999666,
    "device_memory": {
      "num_allocs": 10832,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089835.9521441,
    "end_unix": 1793089838.3805842,
    "seconds": 2.4284374030003164,
    "device_memory": {
      "num_allocs": 10869,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089838.4037607,
    "end_unix": 1793089840.8259816,
    "seconds": 2.4222186940005486,
    "device_memory": {
      "num_allocs": 10906,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089840.859763,
    "end_unix": 1793089843.3880856,
    "seconds": 2.5283201930005816,
    "device_memory": {
      "num_allocs": 10946,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089843.4080045,
    "end_unix": 1793089845.904434,
    "seconds": 2.4964272680008435,
    "device_memory": {
      "num_allocs": 10983,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089845.9361818,
    "end_unix": 1793089848.4175565,
    "seconds": 2.481372284999452,
    "device_memory": {
      "num_allocs": 11020,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count2_A6NFY7",
    "start_unix": 1793089848.4467642,
    "end_unix": 1793089850.9682386,
    "seconds": 2.521472046000781,
    "device_memory": {
      "num_allocs": 11057,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089851.7800896,
    "end_unix": 1793089854.1491446,
    "seconds": 2.369053150999207,
    "device_memory": {
      "num_allocs": 11097,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089854.1765747,
    "end_unix": 1793089856.56402,
    "seconds": 2.38744165099979,
    "device_memory": {
      "num_allocs": 11134,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089856.596444,
    "end_unix": 1793089858.991261,
    "seconds": 2.394814760000372,
    "device_memory": {
      "num_allocs": 11171,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089859.0139642,
    "end_unix": 1793089861.3922553,
    "seconds": 2.3782888509995246,
    "device_memory": {
      "num_allocs": 11208,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089861.4257145,
    "end_unix": 1793089863.8093283,
    "seconds": 2.383611195999947,
    "device_memory": {
      "num_allocs": 11248,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089863.8511329,
    "end_unix": 1793089866.257677,
    "seconds": 2.4065426810002464,
    "device_memory": {
      "num_allocs": 11285,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089866.2974205,
    "end_unix": 1793089868.6889472,
    "seconds": 2.391524306000065,
    "device_memory": {
      "num_allocs": 11322,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089868.7104468,
    "end_unix": 1793089871.098904,
    "seconds": 2.388454983999509,
    "device_memory": {
      "num_allocs": 11359,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089871.1432211,
    "end_unix": 1793089873.5272665,
    "seconds": 2.3840432150000197,
    "device_memory": {
      "num_allocs": 11399,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089873.554629,
    "end_unix": 1793089875.9543295,
    "seconds": 2.3996986790007213,
    "device_memory": {
      "num_allocs": 11436,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089875.9912348,
    "end_unix": 1793089878.3905058,
    "seconds": 2.399268600999676,
    "device_memory": {
      "num_allocs": 11473,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089878.4163551,
    "end_unix": 1793089880.7978575,
    "seconds": 2.381500373999188,
    "device_memory": {
      "num_allocs": 11510,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089880.8344455,
    "end_unix": 1793089883.2046,
    "seconds": 2.3701514809999935,
    "device_memory": {
      "num_allocs": 11550,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089883.2316148,
    "end_unix": 1793089885.597642,
    "seconds": 2.366025055999671,
    "device_memory": {
      "num_allocs": 11587,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089885.6246479,
    "end_unix": 1793089888.023944,
    "seconds": 2.39929391700025,
    "device_memory": {
      "num_allocs": 11624,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089888.0545266,
    "end_unix": 1793089890.4371085,
    "seconds": 2.382579720000649,
    "device_memory": {
      "num_allocs": 11661,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089890.4741528,
    "end_unix": 1793089892.8453555,
    "seconds": 2.371201003000351,
    "device_memory": {
      "num_allocs": 11701,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089892.8731406,
    "end_unix": 1793089895.260957,
    "seconds": 2.3878150900000037,
    "device_memory": {
      "num_allocs": 11738,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089895.2856202,
    "end_unix": 1793089897.671222,
    "seconds": 2.385599526000078,
    "device_memory": {
      "num_allocs": 11775,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count3_O14519",
    "start_unix": 1793089897.694153,
    "end_unix": 1793089900.077216,
    "seconds": 2.3830609229999027,
    "device_memory": {
      "num_allocs": 11812,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089900.8750484,
    "end_unix": 1793089903.2719045,
    "seconds": 2.396855255000446,
    "device_memory": {
      "num_allocs": 11852,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089903.3071175,
    "end_unix": 1793089905.6790302,
    "seconds": 2.3719093420004356,
    "device_memory": {
      "num_allocs": 11889,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089905.7139294,
    "end_unix": 1793089908.098843,
    "seconds": 2.3849112110001442,
    "device_memory": {
      "num_allocs": 11926,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089908.1384165,
    "end_unix": 1793089910.5451834,
    "seconds": 2.406763338000019,
    "device_memory": {
      "num_allocs": 11963,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089910.584251,
    "end_unix": 1793089912.9868567,
    "seconds": 2.402603984999587,
    "device_memory": {
      "num_allocs": 12003,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089913.0125518,
    "end_unix": 1793089915.381029,
    "seconds": 2.368473991000428,
    "device_memory": {
      "num_allocs": 12040,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089915.4176755,
    "end_unix": 1793089917.7927954,
    "seconds": 2.375117629999295,
    "device_memory": {
      "num_allocs": 12077,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089917.8116975,
    "end_unix": 1793089920.1813123,
    "seconds": 2.3696124869993582,
    "device_memory": {
      "num_allocs": 12114,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089920.2132783,
    "end_unix": 1793089922.5935512,
    "seconds": 2.3802694630003316,
    "device_memory": {
      "num_allocs": 12154,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089922.6348393,
    "end_unix": 1793089925.0048735,
    "seconds": 2.370032039999387,
    "device_memory": {
      "num_allocs": 12191,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089925.0476506,
    "end_unix": 1793089927.4457865,
    "seconds": 2.3981334049994985,
    "device_memory": {
      "num_allocs": 12228,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089927.4775462,
    "end_unix": 1793089929.8641856,
    "seconds": 2.386636848999842,
    "device_memory": {
      "num_allocs": 12265,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089929.9035413,
    "end_unix": 1793089932.293992,
    "seconds": 2.3904473869997673,
    "device_memory": {
      "num_allocs": 12305,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089932.3184633,
    "end_unix": 1793089934.7058861,
    "seconds": 2.3874207369999567,
    "device_memory": {
      "num_allocs": 12342,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089934.739378,
    "end_unix": 1793089937.105703,
    "seconds": 2.3663216500008275,
    "device_memory": {
      "num_allocs": 12379,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089937.14534,
    "end_unix": 1793089939.5148726,
    "seconds": 2.3695304780003426,
    "device_memory": {
      "num_allocs": 12416,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089939.5543027,
    "end_unix": 1793089941.9325204,
    "seconds": 2.378215777999685,
    "device_memory": {
      "num_allocs": 12456,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089941.968453,
    "end_unix": 1793089944.362337,
    "seconds": 2.3938820260000284,
    "device_memory": {
      "num_allocs": 12493,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089944.3891206,
    "end_unix": 1793089946.7653987,
    "seconds": 2.376274934000321,
    "device_memory": {
      "num_allocs": 12530,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "count4_P01270",
    "start_unix": 1793089946.7922294,
    "end_unix": 1793089949.1591842,
    "seconds": 2.3669527900001412,
    "device_memory": {
      "num_allocs": 12567,
      "bytes_in_use": 1931666176,
      "peak_bytes_in_use": 2440694272,
      "largest_alloc_size": 451678720,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089959.576239,
    "end_unix": 1793089962.6087942,
    "seconds": 3.0325523370001974,
    "device_memory": {
      "num_allocs": 15019,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089962.6468918,
    "end_unix": 1793089965.5886827,
    "seconds": 2.941788670999813,
    "device_memory": {
      "num_allocs": 15056,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089965.6225314,
    "end_unix": 1793089968.579238,
    "seconds": 2.956704154000363,
    "device_memory": {
      "num_allocs": 15093,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089968.6076555,
    "end_unix": 1793089971.5922468,
    "seconds": 2.9845890870001313,
    "device_memory": {
      "num_allocs": 15130,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089971.621326,
    "end_unix": 1793089974.5827153,
    "seconds": 2.9613872370000536,
    "device_memory": {
      "num_allocs": 15170,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089974.6122725,
    "end_unix": 1793089977.5563862,
    "seconds": 2.9441102410000894,
    "device_memory": {
      "num_allocs": 15207,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089977.5873513,
    "end_unix": 1793089980.5483284,
    "seconds": 2.9609747759996026,
    "device_memory": {
      "num_allocs": 15244,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089980.5790882,
    "end_unix": 1793089983.575035,
    "seconds": 2.9959433470003205,
    "device_memory": {
      "num_allocs": 15281,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089983.6073544,
    "end_unix": 1793089986.573931,
    "seconds": 2.9665730900005656,
    "device_memory": {
      "num_allocs": 15321,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089986.6054168,
    "end_unix": 1793089989.536483,
    "seconds": 2.9310637690005024,
    "device_memory": {
      "num_allocs": 15358,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089989.5619302,
    "end_unix": 1793089992.5309885,
    "seconds": 2.9690546239999094,
    "device_memory": {
      "num_allocs": 15395,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089992.5598147,
    "end_unix": 1793089995.5613103,
    "seconds": 3.001493359000051,
    "device_memory": {
      "num_allocs": 15432,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089995.5968606,
    "end_unix": 1793089998.569439,
    "seconds": 2.972575949000202,
    "device_memory": {
      "num_allocs": 15472,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793089998.5982487,
    "end_unix": 1793090001.5492406,
    "seconds": 2.950989567999386,
    "device_memory": {
      "num_allocs": 15509,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793090001.575626,
    "end_unix": 1793090004.53343,
    "seconds": 2.957800502000282,
    "device_memory": {
      "num_allocs": 15546,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793090004.5652483,
    "end_unix": 1793090007.558609,
    "seconds": 2.9933570240000336,
    "device_memory": {
      "num_allocs": 15583,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793090007.5969586,
    "end_unix": 1793090010.5787504,
    "seconds": 2.9817882109991842,
    "device_memory": {
      "num_allocs": 15623,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793090010.6128635,
    "end_unix": 1793090013.54479,
    "seconds": 2.931924305001303,
    "device_memory": {
      "num_allocs": 15660,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793090013.5821104,
    "end_unix": 1793090016.5490322,
    "seconds": 2.9669194519992743,
    "device_memory": {
      "num_allocs": 15697,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1LYZ",
    "start_unix": 1793090016.5627208,
    "end_unix": 1793090019.5559814,
    "seconds": 2.993258357000741,
    "device_memory": {
      "num_allocs": 15734,
      "bytes_in_use": 1937310464,
      "peak_bytes_in_use": 2477394432,
      "largest_alloc_size": 499425792,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090033.066909,
    "end_unix": 1793090038.747856,
    "seconds": 5.68094231699979,
    "device_memory": {
      "num_allocs": 18049,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090038.788963,
    "end_unix": 1793090044.394786,
    "seconds": 5.605819045000317,
    "device_memory": {
      "num_allocs": 18086,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090044.439109,
    "end_unix": 1793090050.040281,
    "seconds": 5.6011697909998475,
    "device_memory": {
      "num_allocs": 18123,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090050.064731,
    "end_unix": 1793090055.6336029,
    "seconds": 5.568869605000145,
    "device_memory": {
      "num_allocs": 18160,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090055.6730323,
    "end_unix": 1793090061.254319,
    "seconds": 5.581284693000271,
    "device_memory": {
      "num_allocs": 18200,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090061.2811863,
    "end_unix": 1793090066.860724,
    "seconds": 5.579535275001035,
    "device_memory": {
      "num_allocs": 18237,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090066.895462,
    "end_unix": 1793090072.513944,
    "seconds": 5.618479619999562,
    "device_memory": {
      "num_allocs": 18274,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090072.5452633,
    "end_unix": 1793090078.4230192,
    "seconds": 5.877753529000984,
    "device_memory": {
      "num_allocs": 18311,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090078.4840975,
    "end_unix": 1793090084.3097057,
    "seconds": 5.825605403999361,
    "device_memory": {
      "num_allocs": 18351,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090084.3472497,
    "end_unix": 1793090090.222541,
    "seconds": 5.875289187999442,
    "device_memory": {
      "num_allocs": 18388,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090090.2593138,
    "end_unix": 1793090095.9819076,
    "seconds": 5.722591489999104,
    "device_memory": {
      "num_allocs": 18425,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090096.0263188,
    "end_unix": 1793090101.6878028,
    "seconds": 5.661481938999714,
    "device_memory": {
      "num_allocs": 18462,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090101.7385046,
    "end_unix": 1793090107.500865,
    "seconds": 5.762358092000795,
    "device_memory": {
      "num_allocs": 18502,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090107.5440426,
    "end_unix": 1793090113.3776474,
    "seconds": 5.833602641998368,
    "device_memory": {
      "num_allocs": 18539,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090113.409224,
    "end_unix": 1793090119.1776683,
    "seconds": 5.768442570000843,
    "device_memory": {
      "num_allocs": 18576,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090119.2148948,
    "end_unix": 1793090125.0474098,
    "seconds": 5.83251247500084,
    "device_memory": {
      "num_allocs": 18613,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090125.0934162,
    "end_unix": 1793090130.881715,
    "seconds": 5.78829673799919,
    "device_memory": {
      "num_allocs": 18653,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090130.9192262,
    "end_unix": 1793090136.7792091,
    "seconds": 5.859980871000516,
    "device_memory": {
      "num_allocs": 18690,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090136.8178098,
    "end_unix": 1793090142.6215975,
    "seconds": 5.803785953001352,
    "device_memory": {
      "num_allocs": 18727,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  },
  {
    "query": "1TIM",
    "start_unix": 1793090142.6444025,
    "end_unix": 1793090148.511392,
    "seconds": 5.866987315999722,
    "device_memory": {
      "num_allocs": 18764,
      "bytes_in_use": 1972200192,
      "peak_bytes_in_use": 2905266432,
      "largest_alloc_size": 857794560,
      "bytes_limit": 522645929984,
      "bytes_reserved": 0,
      "peak_bytes_reserved": 0,
      "largest_free_block_bytes": 0,
      "pool_bytes": 4292870144,
      "peak_pool_bytes": 4292870144
    }
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-96e58e815995

원래 경로: `colabfold/results/af2_20261027_172813_247681/command.json` · SHA256: `f6a82754f3156ad61690f725fd155583ddc8135c5218b470ca22b5c50359dab8`

<details>
<summary>command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/command.json", "sha256": "f6a82754f3156ad61690f725fd155583ddc8135c5218b470ca22b5c50359dab8", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --name af2_20261027_172813_247681 --gpus all --network none --user 1000:1000 -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e MPLCONFIGDIR=/work/results/af2_20261027_172813_247681/mplcache -v /home/hyein/ai-benchmark/colabfold:/work -v /home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 python -u /work/scripts/benchmark_af2.py af2_20261027_172813_247681 msa_20261027_172359_775627"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-153a2d64020c

원래 경로: `colabfold/results/af2_20261027_172813_247681/exit_code.json` · SHA256: `2a1c761937db8baa9fa8a27e05fc8a7995114b989b8b63a94b793f04652556be`

<details>
<summary>exit_code.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/exit_code.json", "sha256": "2a1c761937db8baa9fa8a27e05fc8a7995114b989b8b63a94b793f04652556be", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "exit_code": 0,
  "start_utc": "2026-10-27T08:28:13.247997+00:00",
  "end_utc": "2026-10-27T08:35:49.819901+00:00"
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-f80bcbbe57ea

원래 경로: `colabfold/results/af2_20261027_172813_247681/input_report.json` · SHA256: `bd3245f28e156f683af563f24049a2fbde69c6da5bf0287cde7db53f2c4b70aa`

<details>
<summary>input_report.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/input_report.json", "sha256": "bd3245f28e156f683af563f24049a2fbde69c6da5bf0287cde7db53f2c4b70aa", "encoding": "utf-8", "added_final_newline": true} -->
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

## file-25c540c7b967

원래 경로: `colabfold/results/af2_20261027_172813_247681/log.txt` · SHA256: `2904f540324d31524ef0c865c21c799a826ec3d7bdbc682a5fb599e883b25e6a`

<details>
<summary>log.txt — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/log.txt", "sha256": "2904f540324d31524ef0c865c21c799a826ec3d7bdbc682a5fb599e883b25e6a", "encoding": "utf-8", "added_final_newline": false} -->
```txt
2026-10-27 08:28:14,697 Running on GPU
2026-10-27 08:28:14,699 Found 5 citations for tools or databases
2026-10-27 08:28:14,699 Query 1/7: 1UBQ (length 76)
2026-10-27 08:28:15,170 generated new fontManager
2026-10-27 08:28:18,490 Padding length to 86
2026-10-27 08:28:28,825 alphafold2_ptm_model_1_seed_042 recycle=0 pLDDT=94.5 pTM=0.81
2026-10-27 08:28:30,502 alphafold2_ptm_model_1_seed_042 recycle=1 pLDDT=95.1 pTM=0.82 tol=0.0638
2026-10-27 08:28:32,209 alphafold2_ptm_model_1_seed_042 recycle=2 pLDDT=95.1 pTM=0.82 tol=0.0501
2026-10-27 08:28:33,889 alphafold2_ptm_model_1_seed_042 recycle=3 pLDDT=95 pTM=0.817 tol=0.0258
2026-10-27 08:28:33,890 alphafold2_ptm_model_1_seed_042 took 15.4s (3 recycles)
2026-10-27 08:28:35,588 alphafold2_ptm_model_2_seed_042 recycle=0 pLDDT=94.9 pTM=0.822
2026-10-27 08:28:37,298 alphafold2_ptm_model_2_seed_042 recycle=1 pLDDT=95.4 pTM=0.831 tol=0.115
2026-10-27 08:28:38,974 alphafold2_ptm_model_2_seed_042 recycle=2 pLDDT=95.2 pTM=0.827 tol=0.0447
2026-10-27 08:28:40,711 alphafold2_ptm_model_2_seed_042 recycle=3 pLDDT=95.2 pTM=0.827 tol=0.0306
2026-10-27 08:28:40,711 alphafold2_ptm_model_2_seed_042 took 6.8s (3 recycles)
2026-10-27 08:28:42,403 alphafold2_ptm_model_3_seed_042 recycle=0 pLDDT=95.2 pTM=0.829
2026-10-27 08:28:44,156 alphafold2_ptm_model_3_seed_042 recycle=1 pLDDT=95.6 pTM=0.834 tol=0.0864
2026-10-27 08:28:45,847 alphafold2_ptm_model_3_seed_042 recycle=2 pLDDT=95.4 pTM=0.832 tol=0.0526
2026-10-27 08:28:47,562 alphafold2_ptm_model_3_seed_042 recycle=3 pLDDT=95.4 pTM=0.834 tol=0.0306
2026-10-27 08:28:47,562 alphafold2_ptm_model_3_seed_042 took 6.8s (3 recycles)
2026-10-27 08:28:49,290 alphafold2_ptm_model_4_seed_042 recycle=0 pLDDT=95.7 pTM=0.837
2026-10-27 08:28:50,970 alphafold2_ptm_model_4_seed_042 recycle=1 pLDDT=96.1 pTM=0.844 tol=0.0592
2026-10-27 08:28:52,714 alphafold2_ptm_model_4_seed_042 recycle=2 pLDDT=95.9 pTM=0.84 tol=0.0568
2026-10-27 08:28:54,399 alphafold2_ptm_model_4_seed_042 recycle=3 pLDDT=95.9 pTM=0.841 tol=0.0268
2026-10-27 08:28:54,400 alphafold2_ptm_model_4_seed_042 took 6.8s (3 recycles)
2026-10-27 08:28:56,127 alphafold2_ptm_model_5_seed_042 recycle=0 pLDDT=93.9 pTM=0.802
2026-10-27 08:28:57,795 alphafold2_ptm_model_5_seed_042 recycle=1 pLDDT=94.4 pTM=0.811 tol=0.0694
2026-10-27 08:28:59,487 alphafold2_ptm_model_5_seed_042 recycle=2 pLDDT=93.8 pTM=0.798 tol=0.0589
2026-10-27 08:29:01,192 alphafold2_ptm_model_5_seed_042 recycle=3 pLDDT=94.2 pTM=0.808 tol=0.0516
2026-10-27 08:29:01,192 alphafold2_ptm_model_5_seed_042 took 6.8s (3 recycles)
2026-10-27 08:29:01,195 reranking models by 'plddt' metric
2026-10-27 08:29:01,195 rank_001_alphafold2_ptm_model_4_seed_042 pLDDT=95.9 pTM=0.841
2026-10-27 08:29:01,195 rank_002_alphafold2_ptm_model_3_seed_042 pLDDT=95.4 pTM=0.834
2026-10-27 08:29:01,195 rank_003_alphafold2_ptm_model_2_seed_042 pLDDT=95.2 pTM=0.827
2026-10-27 08:29:01,195 rank_004_alphafold2_ptm_model_1_seed_042 pLDDT=95 pTM=0.817
2026-10-27 08:29:01,195 rank_005_alphafold2_ptm_model_5_seed_042 pLDDT=94.2 pTM=0.808
2026-10-27 08:29:01,465 Query 2/7: count1_A2RU14 (length 115)
2026-10-27 08:29:01,929 Padding length to 125
2026-10-27 08:29:12,258 alphafold2_ptm_model_1_seed_042 recycle=0 pLDDT=83.6 pTM=0.451
2026-10-27 08:29:14,734 alphafold2_ptm_model_1_seed_042 recycle=1 pLDDT=85.1 pTM=0.462 tol=5.48
2026-10-27 08:29:17,262 alphafold2_ptm_model_1_seed_042 recycle=2 pLDDT=85.1 pTM=0.467 tol=4.68
2026-10-27 08:29:19,660 alphafold2_ptm_model_1_seed_042 recycle=3 pLDDT=84.1 pTM=0.47 tol=21.2
2026-10-27 08:29:19,660 alphafold2_ptm_model_1_seed_042 took 17.7s (3 recycles)
2026-10-27 08:29:22,142 alphafold2_ptm_model_2_seed_042 recycle=0 pLDDT=79.6 pTM=0.408
2026-10-27 08:29:24,665 alphafold2_ptm_model_2_seed_042 recycle=1 pLDDT=81.9 pTM=0.434 tol=2.36
2026-10-27 08:29:27,155 alphafold2_ptm_model_2_seed_042 recycle=2 pLDDT=81 pTM=0.443 tol=4.52
2026-10-27 08:29:29,695 alphafold2_ptm_model_2_seed_042 recycle=3 pLDDT=78.9 pTM=0.444 tol=2.91
2026-10-27 08:29:29,695 alphafold2_ptm_model_2_seed_042 took 10.0s (3 recycles)
2026-10-27 08:29:32,229 alphafold2_ptm_model_3_seed_042 recycle=0 pLDDT=80.6 pTM=0.437
2026-10-27 08:29:34,688 alphafold2_ptm_model_3_seed_042 recycle=1 pLDDT=84.9 pTM=0.458 tol=6.45
2026-10-27 08:29:37,228 alphafold2_ptm_model_3_seed_042 recycle=2 pLDDT=85.1 pTM=0.458 tol=1.78
2026-10-27 08:29:39,737 alphafold2_ptm_model_3_seed_042 recycle=3 pLDDT=84.5 pTM=0.451 tol=0.475
2026-10-27 08:29:39,738 alphafold2_ptm_model_3_seed_042 took 10.0s (3 recycles)
2026-10-27 08:29:42,260 alphafold2_ptm_model_4_seed_042 recycle=0 pLDDT=80.4 pTM=0.409
2026-10-27 08:29:44,795 alphafold2_ptm_model_4_seed_042 recycle=1 pLDDT=83.2 pTM=0.448 tol=1.39
2026-10-27 08:29:47,273 alphafold2_ptm_model_4_seed_042 recycle=2 pLDDT=83.1 pTM=0.469 tol=1.23
2026-10-27 08:29:49,822 alphafold2_ptm_model_4_seed_042 recycle=3 pLDDT=82 pTM=0.464 tol=0.636
2026-10-27 08:29:49,823 alphafold2_ptm_model_4_seed_042 took 10.1s (3 recycles)
2026-10-27 08:29:52,370 alphafold2_ptm_model_5_seed_042 recycle=0 pLDDT=81.8 pTM=0.434
2026-10-27 08:29:54,866 alphafold2_ptm_model_5_seed_042 recycle=1 pLDDT=86.1 pTM=0.49 tol=3.09
2026-10-27 08:29:57,385 alphafold2_ptm_model_5_seed_042 recycle=2 pLDDT=86.1 pTM=0.507 tol=1.44
2026-10-27 08:29:59,903 alphafold2_ptm_model_5_seed_042 recycle=3 pLDDT=86.1 pTM=0.521 tol=0.75
2026-10-27 08:29:59,903 alphafold2_ptm_model_5_seed_042 took 10.1s (3 recycles)
2026-10-27 08:29:59,907 reranking models by 'plddt' metric
2026-10-27 08:29:59,907 rank_001_alphafold2_ptm_model_5_seed_042 pLDDT=86.1 pTM=0.521
2026-10-27 08:29:59,907 rank_002_alphafold2_ptm_model_3_seed_042 pLDDT=84.5 pTM=0.451
2026-10-27 08:29:59,907 rank_003_alphafold2_ptm_model_1_seed_042 pLDDT=84.1 pTM=0.47
2026-10-27 08:29:59,907 rank_004_alphafold2_ptm_model_4_seed_042 pLDDT=82 pTM=0.464
2026-10-27 08:29:59,907 rank_005_alphafold2_ptm_model_2_seed_042 pLDDT=78.9 pTM=0.444
2026-10-27 08:30:00,355 Query 3/7: count2_A6NFY7 (length 115)
2026-10-27 08:30:00,806 Padding length to 125
2026-10-27 08:30:03,307 alphafold2_ptm_model_1_seed_042 recycle=0 pLDDT=73.4 pTM=0.508
2026-10-27 08:30:05,838 alphafold2_ptm_model_1_seed_042 recycle=1 pLDDT=72.4 pTM=0.51 tol=7.7
2026-10-27 08:30:08,390 alphafold2_ptm_model_1_seed_042 recycle=2 pLDDT=71.4 pTM=0.498 tol=5.96
2026-10-27 08:30:10,864 alphafold2_ptm_model_1_seed_042 recycle=3 pLDDT=71.3 pTM=0.498 tol=5.02
2026-10-27 08:30:10,865 alphafold2_ptm_model_1_seed_042 took 10.1s (3 recycles)
2026-10-27 08:30:13,396 alphafold2_ptm_model_2_seed_042 recycle=0 pLDDT=70.4 pTM=0.501
2026-10-27 08:30:15,901 alphafold2_ptm_model_2_seed_042 recycle=1 pLDDT=70.4 pTM=0.524 tol=4.85
2026-10-27 08:30:18,392 alphafold2_ptm_model_2_seed_042 recycle=2 pLDDT=70.1 pTM=0.523 tol=1.18
2026-10-27 08:30:20,925 alphafold2_ptm_model_2_seed_042 recycle=3 pLDDT=70.3 pTM=0.522 tol=1.73
2026-10-27 08:30:20,926 alphafold2_ptm_model_2_seed_042 took 10.1s (3 recycles)
2026-10-27 08:30:23,442 alphafold2_ptm_model_3_seed_042 recycle=0 pLDDT=73.5 pTM=0.525
2026-10-27 08:30:25,961 alphafold2_ptm_model_3_seed_042 recycle=1 pLDDT=73.1 pTM=0.53 tol=5.48
2026-10-27 08:30:28,499 alphafold2_ptm_model_3_seed_042 recycle=2 pLDDT=72.1 pTM=0.522 tol=2.24
2026-10-27 08:30:30,987 alphafold2_ptm_model_3_seed_042 recycle=3 pLDDT=72.4 pTM=0.524 tol=1.85
2026-10-27 08:30:30,987 alphafold2_ptm_model_3_seed_042 took 10.1s (3 recycles)
2026-10-27 08:30:33,496 alphafold2_ptm_model_4_seed_042 recycle=0 pLDDT=73.9 pTM=0.523
2026-10-27 08:30:35,950 alphafold2_ptm_model_4_seed_042 recycle=1 pLDDT=73.5 pTM=0.523 tol=4.71
2026-10-27 08:30:38,402 alphafold2_ptm_model_4_seed_042 recycle=2 pLDDT=73.6 pTM=0.524 tol=1.73
2026-10-27 08:30:40,854 alphafold2_ptm_model_4_seed_042 recycle=3 pLDDT=74 pTM=0.526 tol=1.28
2026-10-27 08:30:40,854 alphafold2_ptm_model_4_seed_042 took 9.9s (3 recycles)
2026-10-27 08:30:43,406 alphafold2_ptm_model_5_seed_042 recycle=0 pLDDT=71.5 pTM=0.523
2026-10-27 08:30:45,934 alphafold2_ptm_model_5_seed_042 recycle=1 pLDDT=72.2 pTM=0.533 tol=3.34
2026-10-27 08:30:48,445 alphafold2_ptm_model_5_seed_042 recycle=2 pLDDT=72.1 pTM=0.534 tol=7.33
2026-10-27 08:30:50,991 alphafold2_ptm_model_5_seed_042 recycle=3 pLDDT=71.8 pTM=0.529 tol=4.61
2026-10-27 08:30:50,991 alphafold2_ptm_model_5_seed_042 took 10.1s (3 recycles)
2026-10-27 08:30:50,997 reranking models by 'plddt' metric
2026-10-27 08:30:50,997 rank_001_alphafold2_ptm_model_4_seed_042 pLDDT=74 pTM=0.526
2026-10-27 08:30:50,997 rank_002_alphafold2_ptm_model_3_seed_042 pLDDT=72.4 pTM=0.524
2026-10-27 08:30:50,997 rank_003_alphafold2_ptm_model_5_seed_042 pLDDT=71.8 pTM=0.529
2026-10-27 08:30:50,997 rank_004_alphafold2_ptm_model_1_seed_042 pLDDT=71.3 pTM=0.498
2026-10-27 08:30:50,997 rank_005_alphafold2_ptm_model_2_seed_042 pLDDT=70.3 pTM=0.522
2026-10-27 08:30:51,358 Query 4/7: count3_O14519 (length 115)
2026-10-27 08:30:51,778 Padding length to 125
2026-10-27 08:30:54,175 alphafold2_ptm_model_1_seed_042 recycle=0 pLDDT=70.7 pTM=0.445
2026-10-27 08:30:56,595 alphafold2_ptm_model_1_seed_042 recycle=1 pLDDT=72.1 pTM=0.446 tol=5.02
2026-10-27 08:30:59,012 alphafold2_ptm_model_1_seed_042 recycle=2 pLDDT=71.4 pTM=0.449 tol=5.77
2026-10-27 08:31:01,420 alphafold2_ptm_model_1_seed_042 recycle=3 pLDDT=71.4 pTM=0.452 tol=3.97
2026-10-27 08:31:01,420 alphafold2_ptm_model_1_seed_042 took 9.6s (3 recycles)
2026-10-27 08:31:03,849 alphafold2_ptm_model_2_seed_042 recycle=0 pLDDT=69.4 pTM=0.438
2026-10-27 08:31:06,296 alphafold2_ptm_model_2_seed_042 recycle=1 pLDDT=69.8 pTM=0.441 tol=5.55
2026-10-27 08:31:08,709 alphafold2_ptm_model_2_seed_042 recycle=2 pLDDT=69.6 pTM=0.442 tol=1.76
2026-10-27 08:31:11,137 alphafold2_ptm_model_2_seed_042 recycle=3 pLDDT=69.9 pTM=0.442 tol=4.27
2026-10-27 08:31:11,137 alphafold2_ptm_model_2_seed_042 took 9.7s (3 recycles)
2026-10-27 08:31:13,553 alphafold2_ptm_model_3_seed_042 recycle=0 pLDDT=71.8 pTM=0.453
2026-10-27 08:31:15,989 alphafold2_ptm_model_3_seed_042 recycle=1 pLDDT=70.8 pTM=0.456 tol=4.21
2026-10-27 08:31:18,415 alphafold2_ptm_model_3_seed_042 recycle=2 pLDDT=70.6 pTM=0.46 tol=6.14
2026-10-27 08:31:20,828 alphafold2_ptm_model_3_seed_042 recycle=3 pLDDT=70.3 pTM=0.46 tol=5.46
2026-10-27 08:31:20,829 alphafold2_ptm_model_3_seed_042 took 9.7s (3 recycles)
2026-10-27 08:31:23,230 alphafold2_ptm_model_4_seed_042 recycle=0 pLDDT=69.1 pTM=0.452
2026-10-27 08:31:25,623 alphafold2_ptm_model_4_seed_042 recycle=1 pLDDT=69.6 pTM=0.451 tol=2.51
2026-10-27 08:31:28,053 alphafold2_ptm_model_4_seed_042 recycle=2 pLDDT=69.8 pTM=0.453 tol=1.97
2026-10-27 08:31:30,468 alphafold2_ptm_model_4_seed_042 recycle=3 pLDDT=70.2 pTM=0.456 tol=1.81
2026-10-27 08:31:30,468 alphafold2_ptm_model_4_seed_042 took 9.6s (3 recycles)
2026-10-27 08:31:32,871 alphafold2_ptm_model_5_seed_042 recycle=0 pLDDT=68.8 pTM=0.443
2026-10-27 08:31:35,284 alphafold2_ptm_model_5_seed_042 recycle=1 pLDDT=68.8 pTM=0.443 tol=9.43
2026-10-27 08:31:37,692 alphafold2_ptm_model_5_seed_042 recycle=2 pLDDT=68.8 pTM=0.444 tol=2.78
2026-10-27 08:31:40,108 alphafold2_ptm_model_5_seed_042 recycle=3 pLDDT=69.1 pTM=0.446 tol=4.33
2026-10-27 08:31:40,108 alphafold2_ptm_model_5_seed_042 took 9.6s (3 recycles)
2026-10-27 08:31:40,112 reranking models by 'plddt' metric
2026-10-27 08:31:40,112 rank_001_alphafold2_ptm_model_1_seed_042 pLDDT=71.4 pTM=0.452
2026-10-27 08:31:40,112 rank_002_alphafold2_ptm_model_3_seed_042 pLDDT=70.3 pTM=0.46
2026-10-27 08:31:40,112 rank_003_alphafold2_ptm_model_4_seed_042 pLDDT=70.2 pTM=0.456
2026-10-27 08:31:40,112 rank_004_alphafold2_ptm_model_2_seed_042 pLDDT=69.9 pTM=0.442
2026-10-27 08:31:40,112 rank_005_alphafold2_ptm_model_5_seed_042 pLDDT=69.1 pTM=0.446
2026-10-27 08:31:40,474 Query 5/7: count4_P01270 (length 115)
2026-10-27 08:31:40,873 Padding length to 125
2026-10-27 08:31:43,305 alphafold2_ptm_model_1_seed_042 recycle=0 pLDDT=67.9 pTM=0.301
2026-10-27 08:31:45,712 alphafold2_ptm_model_1_seed_042 recycle=1 pLDDT=69 pTM=0.33 tol=4.79
2026-10-27 08:31:48,137 alphafold2_ptm_model_1_seed_042 recycle=2 pLDDT=69.2 pTM=0.335 tol=2.4
2026-10-27 08:31:50,578 alphafold2_ptm_model_1_seed_042 recycle=3 pLDDT=69.1 pTM=0.329 tol=3.42
2026-10-27 08:31:50,578 alphafold2_ptm_model_1_seed_042 took 9.7s (3 recycles)
2026-10-27 08:31:53,011 alphafold2_ptm_model_2_seed_042 recycle=0 pLDDT=61.7 pTM=0.241
2026-10-27 08:31:55,416 alphafold2_ptm_model_2_seed_042 recycle=1 pLDDT=64.8 pTM=0.255 tol=7.64
2026-10-27 08:31:57,810 alphafold2_ptm_model_2_seed_042 recycle=2 pLDDT=64.6 pTM=0.255 tol=2.38
2026-10-27 08:32:00,207 alphafold2_ptm_model_2_seed_042 recycle=3 pLDDT=63.8 pTM=0.248 tol=1.47
2026-10-27 08:32:00,208 alphafold2_ptm_model_2_seed_042 took 9.6s (3 recycles)
2026-10-27 08:32:02,633 alphafold2_ptm_model_3_seed_042 recycle=0 pLDDT=69.4 pTM=0.296
2026-10-27 08:32:05,046 alphafold2_ptm_model_3_seed_042 recycle=1 pLDDT=69.9 pTM=0.31 tol=3.34
2026-10-27 08:32:07,476 alphafold2_ptm_model_3_seed_042 recycle=2 pLDDT=70.5 pTM=0.312 tol=2.12
2026-10-27 08:32:09,897 alphafold2_ptm_model_3_seed_042 recycle=3 pLDDT=70.2 pTM=0.308 tol=1.19
2026-10-27 08:32:09,898 alphafold2_ptm_model_3_seed_042 took 9.7s (3 recycles)
2026-10-27 08:32:12,317 alphafold2_ptm_model_4_seed_042 recycle=0 pLDDT=57.7 pTM=0.236
2026-10-27 08:32:14,738 alphafold2_ptm_model_4_seed_042 recycle=1 pLDDT=60.2 pTM=0.249 tol=9.2
2026-10-27 08:32:17,144 alphafold2_ptm_model_4_seed_042 recycle=2 pLDDT=58.9 pTM=0.245 tol=1.57
2026-10-27 08:32:19,548 alphafold2_ptm_model_4_seed_042 recycle=3 pLDDT=59.2 pTM=0.247 tol=1.45
2026-10-27 08:32:19,548 alphafold2_ptm_model_4_seed_042 took 9.6s (3 recycles)
2026-10-27 08:32:21,967 alphafold2_ptm_model_5_seed_042 recycle=0 pLDDT=62.2 pTM=0.26
2026-10-27 08:32:24,387 alphafold2_ptm_model_5_seed_042 recycle=1 pLDDT=63.9 pTM=0.26 tol=6.73
2026-10-27 08:32:26,790 alphafold2_ptm_model_5_seed_042 recycle=2 pLDDT=64.3 pTM=0.258 tol=3.35
2026-10-27 08:32:29,195 alphafold2_ptm_model_5_seed_042 recycle=3 pLDDT=64.1 pTM=0.258 tol=1.67
2026-10-27 08:32:29,195 alphafold2_ptm_model_5_seed_042 took 9.6s (3 recycles)
2026-10-27 08:32:29,199 reranking models by 'plddt' metric
2026-10-27 08:32:29,199 rank_001_alphafold2_ptm_model_3_seed_042 pLDDT=70.2 pTM=0.308
2026-10-27 08:32:29,199 rank_002_alphafold2_ptm_model_1_seed_042 pLDDT=69.1 pTM=0.329
2026-10-27 08:32:29,199 rank_003_alphafold2_ptm_model_5_seed_042 pLDDT=64.1 pTM=0.258
2026-10-27 08:32:29,199 rank_004_alphafold2_ptm_model_2_seed_042 pLDDT=63.8 pTM=0.248
2026-10-27 08:32:29,199 rank_005_alphafold2_ptm_model_4_seed_042 pLDDT=59.2 pTM=0.247
2026-10-27 08:32:29,565 Query 6/7: 1LYZ (length 129)
2026-10-27 08:32:30,263 Padding length to 139
2026-10-27 08:32:42,645 alphafold2_ptm_model_1_seed_042 recycle=0 pLDDT=96.6 pTM=0.882
2026-10-27 08:32:45,621 alphafold2_ptm_model_1_seed_042 recycle=1 pLDDT=97.2 pTM=0.892 tol=0.14
2026-10-27 08:32:48,606 alphafold2_ptm_model_1_seed_042 recycle=2 pLDDT=97.3 pTM=0.891 tol=0.0999
2026-10-27 08:32:51,615 alphafold2_ptm_model_1_seed_042 recycle=3 pLDDT=97.4 pTM=0.893 tol=0.0334
2026-10-27 08:32:51,615 alphafold2_ptm_model_1_seed_042 took 21.4s (3 recycles)
2026-10-27 08:32:54,610 alphafold2_ptm_model_2_seed_042 recycle=0 pLDDT=96.8 pTM=0.891
2026-10-27 08:32:57,585 alphafold2_ptm_model_2_seed_042 recycle=1 pLDDT=97.4 pTM=0.898 tol=0.159
2026-10-27 08:33:00,577 alphafold2_ptm_model_2_seed_042 recycle=2 pLDDT=97.5 pTM=0.9 tol=0.0559
2026-10-27 08:33:03,601 alphafold2_ptm_model_2_seed_042 recycle=3 pLDDT=97.6 pTM=0.9 tol=0.0271
2026-10-27 08:33:03,601 alphafold2_ptm_model_2_seed_042 took 12.0s (3 recycles)
2026-10-27 08:33:06,604 alphafold2_ptm_model_3_seed_042 recycle=0 pLDDT=97.1 pTM=0.89
2026-10-27 08:33:09,560 alphafold2_ptm_model_3_seed_042 recycle=1 pLDDT=97.6 pTM=0.899 tol=0.126
2026-10-27 08:33:12,558 alphafold2_ptm_model_3_seed_042 recycle=2 pLDDT=97.7 pTM=0.901 tol=0.0594
2026-10-27 08:33:15,590 alphafold2_ptm_model_3_seed_042 recycle=3 pLDDT=97.8 pTM=0.902 tol=0.0326
2026-10-27 08:33:15,590 alphafold2_ptm_model_3_seed_042 took 12.0s (3 recycles)
2026-10-27 08:33:18,596 alphafold2_ptm_model_4_seed_042 recycle=0 pLDDT=97.1 pTM=0.894
2026-10-27 08:33:21,574 alphafold2_ptm_model_4_seed_042 recycle=1 pLDDT=97.8 pTM=0.903 tol=0.181
2026-10-27 08:33:24,563 alphafold2_ptm_model_4_seed_042 recycle=2 pLDDT=97.8 pTM=0.904 tol=0.0488
2026-10-27 08:33:27,590 alphafold2_ptm_model_4_seed_042 recycle=3 pLDDT=97.9 pTM=0.904 tol=0.0288
2026-10-27 08:33:27,590 alphafold2_ptm_model_4_seed_042 took 12.0s (3 recycles)
2026-10-27 08:33:30,611 alphafold2_ptm_model_5_seed_042 recycle=0 pLDDT=97.3 pTM=0.9
2026-10-27 08:33:33,580 alphafold2_ptm_model_5_seed_042 recycle=1 pLDDT=98 pTM=0.909 tol=0.135
2026-10-27 08:33:36,561 alphafold2_ptm_model_5_seed_042 recycle=2 pLDDT=98.1 pTM=0.91 tol=0.0447
2026-10-27 08:33:39,576 alphafold2_ptm_model_5_seed_042 recycle=3 pLDDT=98.2 pTM=0.912 tol=0.0254
2026-10-27 08:33:39,576 alphafold2_ptm_model_5_seed_042 took 12.0s (3 recycles)
2026-10-27 08:33:39,581 reranking models by 'plddt' metric
2026-10-27 08:33:39,581 rank_001_alphafold2_ptm_model_5_seed_042 pLDDT=98.2 pTM=0.912
2026-10-27 08:33:39,581 rank_002_alphafold2_ptm_model_4_seed_042 pLDDT=97.9 pTM=0.904
2026-10-27 08:33:39,581 rank_003_alphafold2_ptm_model_3_seed_042 pLDDT=97.8 pTM=0.902
2026-10-27 08:33:39,581 rank_004_alphafold2_ptm_model_2_seed_042 pLDDT=97.6 pTM=0.9
2026-10-27 08:33:39,581 rank_005_alphafold2_ptm_model_1_seed_042 pLDDT=97.4 pTM=0.893
2026-10-27 08:33:40,010 Query 7/7: 1TIM (length 247)
2026-10-27 08:33:58,787 alphafold2_ptm_model_1_seed_042 recycle=0 pLDDT=95.7 pTM=0.925
2026-10-27 08:34:04,437 alphafold2_ptm_model_1_seed_042 recycle=1 pLDDT=96.1 pTM=0.931 tol=0.194
2026-10-27 08:34:10,063 alphafold2_ptm_model_1_seed_042 recycle=2 pLDDT=96.1 pTM=0.931 tol=0.0917
2026-10-27 08:34:15,662 alphafold2_ptm_model_1_seed_042 recycle=3 pLDDT=96.1 pTM=0.931 tol=0.0342
2026-10-27 08:34:15,663 alphafold2_ptm_model_1_seed_042 took 34.1s (3 recycles)
2026-10-27 08:34:21,279 alphafold2_ptm_model_2_seed_042 recycle=0 pLDDT=95.2 pTM=0.929
2026-10-27 08:34:26,893 alphafold2_ptm_model_2_seed_042 recycle=1 pLDDT=95.4 pTM=0.932 tol=0.186
2026-10-27 08:34:32,543 alphafold2_ptm_model_2_seed_042 recycle=2 pLDDT=95.5 pTM=0.933 tol=0.0643
2026-10-27 08:34:38,472 alphafold2_ptm_model_2_seed_042 recycle=3 pLDDT=95.6 pTM=0.934 tol=0.037
2026-10-27 08:34:38,473 alphafold2_ptm_model_2_seed_042 took 22.8s (3 recycles)
2026-10-27 08:34:44,345 alphafold2_ptm_model_3_seed_042 recycle=0 pLDDT=95.8 pTM=0.933
2026-10-27 08:34:50,257 alphafold2_ptm_model_3_seed_042 recycle=1 pLDDT=96.1 pTM=0.937 tol=0.18
2026-10-27 08:34:56,024 alphafold2_ptm_model_3_seed_042 recycle=2 pLDDT=96.4 pTM=0.938 tol=0.0734
2026-10-27 08:35:01,728 alphafold2_ptm_model_3_seed_042 recycle=3 pLDDT=96.4 pTM=0.938 tol=0.038
2026-10-27 08:35:01,728 alphafold2_ptm_model_3_seed_042 took 23.2s (3 recycles)
2026-10-27 08:35:07,542 alphafold2_ptm_model_4_seed_042 recycle=0 pLDDT=95.9 pTM=0.934
2026-10-27 08:35:13,407 alphafold2_ptm_model_4_seed_042 recycle=1 pLDDT=96.1 pTM=0.937 tol=0.208
2026-10-27 08:35:19,213 alphafold2_ptm_model_4_seed_042 recycle=2 pLDDT=96.3 pTM=0.938 tol=0.0995
2026-10-27 08:35:25,083 alphafold2_ptm_model_4_seed_042 recycle=3 pLDDT=96.4 pTM=0.938 tol=0.0531
2026-10-27 08:35:25,083 alphafold2_ptm_model_4_seed_042 took 23.3s (3 recycles)
2026-10-27 08:35:30,917 alphafold2_ptm_model_5_seed_042 recycle=0 pLDDT=96.2 pTM=0.936
2026-10-27 08:35:36,816 alphafold2_ptm_model_5_seed_042 recycle=1 pLDDT=96.6 pTM=0.938 tol=0.175
2026-10-27 08:35:42,643 alphafold2_ptm_model_5_seed_042 recycle=2 pLDDT=96.6 pTM=0.938 tol=0.0895
2026-10-27 08:35:48,548 alphafold2_ptm_model_5_seed_042 recycle=3 pLDDT=96.5 pTM=0.938 tol=0.0374
2026-10-27 08:35:48,548 alphafold2_ptm_model_5_seed_042 took 23.5s (3 recycles)
2026-10-27 08:35:48,561 reranking models by 'plddt' metric
2026-10-27 08:35:48,561 rank_001_alphafold2_ptm_model_5_seed_042 pLDDT=96.5 pTM=0.938
2026-10-27 08:35:48,561 rank_002_alphafold2_ptm_model_3_seed_042 pLDDT=96.4 pTM=0.938
2026-10-27 08:35:48,561 rank_003_alphafold2_ptm_model_4_seed_042 pLDDT=96.4 pTM=0.938
2026-10-27 08:35:48,561 rank_004_alphafold2_ptm_model_1_seed_042 pLDDT=96.1 pTM=0.931
2026-10-27 08:35:48,561 rank_005_alphafold2_ptm_model_2_seed_042 pLDDT=95.6 pTM=0.934
2026-10-27 08:35:48,959 Done
```
<!-- END_RESTORE_FILE -->


</details>

## file-412d7c599c46

원래 경로: `colabfold/results/af2_20261027_172813_247681/predictions.json` · SHA256: `82ad0f9844e58cd4d13671758362561d8be0e367c11dd93fdc07bc954de905f0`

<details>
<summary>predictions.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/predictions.json", "sha256": "82ad0f9844e58cd4d13671758362561d8be0e367c11dd93fdc07bc954de905f0", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "query": "1UBQ",
    "tag": "alphafold2_ptm_model_1_seed_042",
    "lengths": [
      76
    ],
    "mean_plddt": 95.0,
    "ptm": 0.81689453125,
    "plddt_shape": [
      86
    ],
    "pae_shape": [
      86,
      86
    ],
    "coordinates_shape": [
      86,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1UBQ",
    "tag": "alphafold2_ptm_model_2_seed_042",
    "lengths": [
      76
    ],
    "mean_plddt": 95.1875,
    "ptm": 0.8271484375,
    "plddt_shape": [
      86
    ],
    "pae_shape": [
      86,
      86
    ],
    "coordinates_shape": [
      86,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1UBQ",
    "tag": "alphafold2_ptm_model_3_seed_042",
    "lengths": [
      76
    ],
    "mean_plddt": 95.4375,
    "ptm": 0.833984375,
    "plddt_shape": [
      86
    ],
    "pae_shape": [
      86,
      86
    ],
    "coordinates_shape": [
      86,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1UBQ",
    "tag": "alphafold2_ptm_model_4_seed_042",
    "lengths": [
      76
    ],
    "mean_plddt": 95.9375,
    "ptm": 0.84130859375,
    "plddt_shape": [
      86
    ],
    "pae_shape": [
      86,
      86
    ],
    "coordinates_shape": [
      86,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1UBQ",
    "tag": "alphafold2_ptm_model_5_seed_042",
    "lengths": [
      76
    ],
    "mean_plddt": 94.1875,
    "ptm": 0.8076171875,
    "plddt_shape": [
      86
    ],
    "pae_shape": [
      86,
      86
    ],
    "coordinates_shape": [
      86,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count1_A2RU14",
    "tag": "alphafold2_ptm_model_1_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 84.125,
    "ptm": 0.4697265625,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count1_A2RU14",
    "tag": "alphafold2_ptm_model_2_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 78.9375,
    "ptm": 0.44384765625,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count1_A2RU14",
    "tag": "alphafold2_ptm_model_3_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 84.5,
    "ptm": 0.450927734375,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count1_A2RU14",
    "tag": "alphafold2_ptm_model_4_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 82.0,
    "ptm": 0.4638671875,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count1_A2RU14",
    "tag": "alphafold2_ptm_model_5_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 86.0625,
    "ptm": 0.5205078125,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count2_A6NFY7",
    "tag": "alphafold2_ptm_model_1_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 71.3125,
    "ptm": 0.49755859375,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count2_A6NFY7",
    "tag": "alphafold2_ptm_model_2_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 70.3125,
    "ptm": 0.5224609375,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count2_A6NFY7",
    "tag": "alphafold2_ptm_model_3_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 72.375,
    "ptm": 0.5244140625,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count2_A6NFY7",
    "tag": "alphafold2_ptm_model_4_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 74.0,
    "ptm": 0.5263671875,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count2_A6NFY7",
    "tag": "alphafold2_ptm_model_5_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 71.75,
    "ptm": 0.529296875,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count3_O14519",
    "tag": "alphafold2_ptm_model_1_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 71.375,
    "ptm": 0.452392578125,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count3_O14519",
    "tag": "alphafold2_ptm_model_2_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 69.9375,
    "ptm": 0.4423828125,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count3_O14519",
    "tag": "alphafold2_ptm_model_3_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 70.3125,
    "ptm": 0.460205078125,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count3_O14519",
    "tag": "alphafold2_ptm_model_4_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 70.25,
    "ptm": 0.45556640625,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count3_O14519",
    "tag": "alphafold2_ptm_model_5_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 69.125,
    "ptm": 0.446044921875,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count4_P01270",
    "tag": "alphafold2_ptm_model_1_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 69.0625,
    "ptm": 0.32861328125,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count4_P01270",
    "tag": "alphafold2_ptm_model_2_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 63.8125,
    "ptm": 0.24755859375,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count4_P01270",
    "tag": "alphafold2_ptm_model_3_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 70.25,
    "ptm": 0.308349609375,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count4_P01270",
    "tag": "alphafold2_ptm_model_4_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 59.15625,
    "ptm": 0.2474365234375,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "count4_P01270",
    "tag": "alphafold2_ptm_model_5_seed_042",
    "lengths": [
      115
    ],
    "mean_plddt": 64.0625,
    "ptm": 0.258056640625,
    "plddt_shape": [
      125
    ],
    "pae_shape": [
      125,
      125
    ],
    "coordinates_shape": [
      125,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1LYZ",
    "tag": "alphafold2_ptm_model_1_seed_042",
    "lengths": [
      129
    ],
    "mean_plddt": 97.375,
    "ptm": 0.892578125,
    "plddt_shape": [
      139
    ],
    "pae_shape": [
      139,
      139
    ],
    "coordinates_shape": [
      139,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1LYZ",
    "tag": "alphafold2_ptm_model_2_seed_042",
    "lengths": [
      129
    ],
    "mean_plddt": 97.5625,
    "ptm": 0.900390625,
    "plddt_shape": [
      139
    ],
    "pae_shape": [
      139,
      139
    ],
    "coordinates_shape": [
      139,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1LYZ",
    "tag": "alphafold2_ptm_model_3_seed_042",
    "lengths": [
      129
    ],
    "mean_plddt": 97.8125,
    "ptm": 0.90185546875,
    "plddt_shape": [
      139
    ],
    "pae_shape": [
      139,
      139
    ],
    "coordinates_shape": [
      139,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1LYZ",
    "tag": "alphafold2_ptm_model_4_seed_042",
    "lengths": [
      129
    ],
    "mean_plddt": 97.875,
    "ptm": 0.904296875,
    "plddt_shape": [
      139
    ],
    "pae_shape": [
      139,
      139
    ],
    "coordinates_shape": [
      139,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1LYZ",
    "tag": "alphafold2_ptm_model_5_seed_042",
    "lengths": [
      129
    ],
    "mean_plddt": 98.1875,
    "ptm": 0.912109375,
    "plddt_shape": [
      139
    ],
    "pae_shape": [
      139,
      139
    ],
    "coordinates_shape": [
      139,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1TIM",
    "tag": "alphafold2_ptm_model_1_seed_042",
    "lengths": [
      247
    ],
    "mean_plddt": 96.125,
    "ptm": 0.9306640625,
    "plddt_shape": [
      247
    ],
    "pae_shape": [
      247,
      247
    ],
    "coordinates_shape": [
      247,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1TIM",
    "tag": "alphafold2_ptm_model_2_seed_042",
    "lengths": [
      247
    ],
    "mean_plddt": 95.625,
    "ptm": 0.93359375,
    "plddt_shape": [
      247
    ],
    "pae_shape": [
      247,
      247
    ],
    "coordinates_shape": [
      247,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1TIM",
    "tag": "alphafold2_ptm_model_3_seed_042",
    "lengths": [
      247
    ],
    "mean_plddt": 96.4375,
    "ptm": 0.9384765625,
    "plddt_shape": [
      247
    ],
    "pae_shape": [
      247,
      247
    ],
    "coordinates_shape": [
      247,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1TIM",
    "tag": "alphafold2_ptm_model_4_seed_042",
    "lengths": [
      247
    ],
    "mean_plddt": 96.375,
    "ptm": 0.9375,
    "plddt_shape": [
      247
    ],
    "pae_shape": [
      247,
      247
    ],
    "coordinates_shape": [
      247,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  },
  {
    "query": "1TIM",
    "tag": "alphafold2_ptm_model_5_seed_042",
    "lengths": [
      247
    ],
    "mean_plddt": 96.5,
    "ptm": 0.9384765625,
    "plddt_shape": [
      247
    ],
    "pae_shape": [
      247,
      247
    ],
    "coordinates_shape": [
      247,
      37,
      3
    ],
    "finite_coordinates": true,
    "finite_plddt": true
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-286f3b77bd4c

원래 경로: `colabfold/results/af2_20261027_172813_247681/query_stats.json` · SHA256: `1ba3cca0457de03f7ccd48fc1538a128271e629037bca92044602e1843cab3e9`

<details>
<summary>query_stats.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/query_stats.json", "sha256": "1ba3cca0457de03f7ccd48fc1538a128271e629037bca92044602e1843cab3e9", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "query": "1UBQ",
    "wall_seconds": 43.25274384499971,
    "start_unix": 1793089697.9430606,
    "end_unix": 1793089741.1958034
  },
  {
    "query": "count1_A2RU14",
    "wall_seconds": 58.28870832299981,
    "start_unix": 1793089741.6190722,
    "end_unix": 1793089799.9077797
  },
  {
    "query": "count2_A6NFY7",
    "wall_seconds": 50.44888227699994,
    "start_unix": 1793089800.548487,
    "end_unix": 1793089850.9973686
  },
  {
    "query": "count3_O14519",
    "wall_seconds": 48.58623763699961,
    "start_unix": 1793089851.526802,
    "end_unix": 1793089900.1130393
  },
  {
    "query": "count4_P01270",
    "wall_seconds": 48.568828466000014,
    "start_unix": 1793089900.6306448,
    "end_unix": 1793089949.1994724
  },
  {
    "query": "1LYZ",
    "wall_seconds": 69.70946580100008,
    "start_unix": 1793089949.8723326,
    "end_unix": 1793090019.5817976
  },
  {
    "query": "1TIM",
    "wall_seconds": 127.51820730599866,
    "start_unix": 1793090021.0439842,
    "end_unix": 1793090148.5621908
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-f52d22086381

원래 경로: `colabfold/results/af2_20261027_172813_247681/resources.csv` · SHA256: `bba4a56a888ce41136ebfe65672448ef0c5be635690e7ac8c5f687519286f027`

<details>
<summary>resources.csv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/resources.csv", "sha256": "bba4a56a888ce41136ebfe65672448ef0c5be635690e7ac8c5f687519286f027", "encoding": "utf-8", "added_final_newline": false} -->
```csv
query_start,query_end,query,phase,gpu_util_percent,rss_bytes,system_available_bytes,swap_used_bytes
1793089694.4214492,1793089694.4899368,setup,setup,8.0,821256192,122782015488,0
1793089694.927042,1793089695.068015,1UBQ,preprocessing,8.0,871002112,122758574080,0
1793089695.4273129,1793089695.4499068,1UBQ,pipeline,4.0,918577152,122631696384,0
1793089695.927433,1793089695.9494286,1UBQ,model_loading,4.0,1260486656,122382913536,0
1793089696.4275687,1793089696.4471643,1UBQ,model_loading,28.0,1585860608,121949487104,0
1793089696.9276798,1793089696.9474401,1UBQ,model_loading,28.0,2527035392,120893808640,0
1793089697.4282157,1793089697.449155,1UBQ,model_loading,59.0,3189321728,120216059904,0
1793089697.9283404,1793089697.9520235,1UBQ,model_loading,59.0,3762372608,119533555712,0
1793089698.429722,1793089698.450814,1UBQ,preprocessing,5.0,4106928128,119285817344,0
1793089698.935608,1793089699.2977495,1UBQ,compilation,5.0,4109369344,119295897600,0
1793089699.4428678,1793089699.469758,1UBQ,compilation,5.0,4112908288,119275065344,0
1793089699.9430602,1793089699.9647439,1UBQ,compilation,5.0,4117364736,119271231488,0
1793089700.443243,1793089700.462588,1UBQ,compilation,5.0,4121325568,119253245952,0
1793089700.9437869,1793089700.9647942,1UBQ,compilation,5.0,4125237248,119258689536,0
1793089701.4439073,1793089701.465919,1UBQ,compilation,6.0,4128018432,119240294400,0
1793089701.9441764,1793089701.9653785,1UBQ,compilation,6.0,4129767424,119228182528,0
1793089702.444286,1793089702.4653986,1UBQ,compilation,6.0,4131401728,119218950144,0
1793089702.9446776,1793089702.9659026,1UBQ,compilation,6.0,4131971072,119211462656,0
1793089703.4447887,1793089703.4634197,1UBQ,compilation,6.0,4132638720,119203729408,0
1793089703.944893,1793089703.965569,1UBQ,compilation,6.0,4132642816,119196872704,0
1793089704.4450767,1793089704.4714148,1UBQ,compilation,8.0,4695572480,118612410368,0
1793089704.945178,1793089704.9665408,1UBQ,compilation,8.0,4776574976,118583111680,0
1793089705.4452987,1793089705.4673061,1UBQ,compilation,70.0,4796796928,118563041280,0
1793089705.945406,1793089705.977573,1UBQ,compilation,70.0,5026045952,118373564416,0
1793089706.4455075,1793089706.4668922,1UBQ,compilation,6.0,5156438016,118193324032,0
1793089706.9457,1793089706.9689631,1UBQ,compilation,6.0,5183467520,118148976640,0
1793089707.4458845,1793089707.479656,1UBQ,execution,66.0,5323165696,118027743232,0
1793089707.9462402,1793089707.972211,1UBQ,execution,66.0,5323165696,118014033920,0
1793089708.4466496,1793089708.474233,1UBQ,execution,94.0,5323165696,118013640704,0
1793089708.9469655,1793089708.9713192,1UBQ,execution,94.0,5332131840,118016172032,0
1793089709.4474168,1793089709.4779959,1UBQ,execution,90.0,5332131840,118014480384,0
1793089709.9476323,1793089709.979129,1UBQ,execution,90.0,5332131840,118026874880,0
1793089710.448201,1793089710.4781766,1UBQ,execution,94.0,5332131840,118044585984,0
1793089710.9483025,1793089710.980346,1UBQ,execution,94.0,5333966848,118045855744,0
1793089711.4487944,1793089711.4758599,1UBQ,execution,94.0,5333966848,118049329152,0
1793089711.9490614,1793089711.980451,1UBQ,execution,94.0,5333966848,118068887552,0
1793089712.4494653,1793089712.4720614,1UBQ,execution,94.0,5333975040,118078197760,0
1793089712.949745,1793089712.9770005,1UBQ,execution,94.0,5333975040,118079864832,0
1793089713.450289,1793089713.4775848,1UBQ,execution,94.0,5333975040,118076473344,0
1793089713.9507914,1793089713.9776244,1UBQ,execution,94.0,5334016000,118089072640,0
1793089714.4510894,1793089714.481896,1UBQ,execution,90.0,5334016000,118081994752,0
1793089714.9517167,1793089714.9757862,1UBQ,execution,90.0,5334016000,118099046400,0
1793089715.4519708,1793089715.4798343,1UBQ,execution,94.0,5334020096,118108405760,0
1793089715.9526064,1793089715.979334,1UBQ,execution,94.0,5334020096,118103502848,0
1793089716.4529424,1793089716.483938,1UBQ,execution,94.0,5334020096,118086934528,0
1793089716.9531307,1793089716.9806693,1UBQ,execution,94.0,5334020096,118141190144,0
1793089717.4533148,1793089717.4754493,1UBQ,execution,94.0,5334024192,118148272128,0
1793089717.9534345,1793089717.9864635,1UBQ,execution,94.0,5334024192,118173593600,0
1793089718.4542315,1793089718.4814084,1UBQ,execution,94.0,5334024192,118181924864,0
1793089718.954348,1793089718.9784527,1UBQ,execution,94.0,5334024192,118193987584,0
1793089719.4544804,1793089719.4854298,1UBQ,execution,90.0,5334024192,118182359040,0
1793089719.9552553,1793089719.983066,1UBQ,execution,90.0,5334024192,118183059456,0
1793089720.4560385,1793089720.4846544,1UBQ,execution,94.0,5334024192,118184038400,0
1793089720.9561565,1793089720.9864845,1UBQ,execution,94.0,5334032384,118192439296,0
1793089721.4563477,1793089721.486322,1UBQ,execution,94.0,5334032384,118194925568,0
1793089721.9564962,1793089721.983855,1UBQ,execution,94.0,5334032384,118195437568,0
1793089722.4572196,1793089722.4826403,1UBQ,execution,94.0,5334040576,118196113408,0
1793089722.9578478,1793089722.985566,1UBQ,execution,94.0,5334040576,118212665344,0
1793089723.4579592,1793089723.488253,1UBQ,execution,94.0,5334040576,118227714048,0
1793089723.958157,1793089723.9819748,1UBQ,execution,94.0,5334040576,118227234816,0
1793089724.4582767,1793089724.491323,1UBQ,execution,94.0,5334048768,118255677440,0
1793089724.9584231,1793089724.9870229,1UBQ,execution,94.0,5334048768,118258511872,0
1793089725.458545,1793089725.4867494,1UBQ,execution,94.0,5334048768,118254235648,0
1793089725.9590461,1793089725.982259,1UBQ,execution,94.0,5334048768,118254600192,0
1793089726.4593213,1793089726.4952338,1UBQ,execution,90.0,5334048768,118272532480,0
1793089726.959615,1793089726.9870594,1UBQ,execution,90.0,5334048768,118283280384,0
1793089727.4597294,1793089727.4910223,1UBQ,execution,94.0,5334048768,118279655424,0
1793089727.9600732,1793089727.9892306,1UBQ,execution,94.0,5334048768,118281895936,0
1793089728.4605944,1793089728.4871948,1UBQ,execution,94.0,5334048768,118284926976,0
1793089728.960792,1793089728.99228,1UBQ,execution,94.0,5334048768,118268231680,0
1793089729.4610174,1793089729.485309,1UBQ,execution,94.0,5334061056,118289268736,0
1793089729.9614947,1793089729.986146,1UBQ,execution,94.0,5334061056,118293307392,0
1793089730.4616961,1793089730.4915085,1UBQ,execution,94.0,5334065152,118284075008,0
1793089730.961805,1793089730.9868937,1UBQ,prediction,94.0,5334069248,118287020032,0
1793089731.4623353,1793089731.493484,1UBQ,execution,90.0,5334069248,118280609792,0
1793089731.962891,1793089731.9908752,1UBQ,execution,90.0,5334069248,118285647872,0
1793089732.463525,1793089732.4929447,1UBQ,execution,94.0,5334069248,118277791744,0
1793089732.9636288,1793089732.9868543,1UBQ,execution,94.0,5334073344,118283337728,0
1793089733.4640658,1793089733.493532,1UBQ,execution,94.0,5334073344,118280900608,0
1793089733.9648983,1793089733.990312,1UBQ,execution,94.0,5334073344,118265540608,0
1793089734.4650028,1793089734.4907439,1UBQ,execution,94.0,5334081536,118261760000,0
1793089734.9651184,1793089734.9976895,1UBQ,execution,94.0,5334081536,118282760192,0
1793089735.4654377,1793089735.4958072,1UBQ,execution,94.0,5334081536,118294433792,0
1793089735.9656894,1793089735.9948933,1UBQ,execution,94.0,5334081536,118296997888,0
1793089736.4657989,1793089736.5030851,1UBQ,execution,74.0,5334093824,118316929024,0
1793089736.96591,1793089736.993986,1UBQ,execution,74.0,5334093824,118299852800,0
1793089737.466065,1793089737.4928315,1UBQ,execution,94.0,5334093824,118299533312,0
1793089737.9666352,1793089737.9912355,1UBQ,execution,94.0,5334097920,118282645504,0
1793089738.4674013,1793089738.5037072,1UBQ,execution,92.0,5334106112,118266023936,0
1793089738.9679143,1793089738.9953039,1UBQ,execution,92.0,5334106112,118266023936,0
1793089739.4682863,1793089739.4894612,1UBQ,execution,94.0,5334106112,118260789248,0
1793089739.968519,1793089739.997855,1UBQ,execution,94.0,5334106112,118273449984,0
1793089740.4688282,1793089740.4928107,1UBQ,execution,94.0,5334106112,118290804736,0
1793089740.9692335,1793089740.995177,1UBQ,execution,94.0,5334106112,118290960384,0
1793089741.4715161,1793089741.4935737,1UBQ,between_queries,94.0,5290237952,118352867328,0
1793089742.1150587,1793089742.4920216,count1_A2RU14,compilation,94.0,5453832192,118172876800,0
1793089742.6207616,1793089742.6870553,count1_A2RU14,compilation,5.0,5455757312,118175150080,0
1793089743.120995,1793089743.142044,count1_A2RU14,compilation,5.0,5458984960,118155767808,0
1793089743.6210992,1793089743.6449256,count1_A2RU14,compilation,7.0,5458984960,118169673728,0
1793089744.121193,1793089744.141608,count1_A2RU14,compilation,7.0,5458984960,118167998464,0
1793089744.6213942,1793089744.640828,count1_A2RU14,compilation,6.0,5458984960,118095732736,0
1793089745.1215909,1793089745.1424158,count1_A2RU14,compilation,6.0,5458984960,117971582976,0
1793089745.6217034,1793089745.6426322,count1_A2RU14,compilation,6.0,5458984960,117906513920,0
1793089746.1218152,1793089746.1411414,count1_A2RU14,compilation,6.0,5463764992,117787058176,0
1793089746.6220105,1793089746.6412303,count1_A2RU14,compilation,5.0,5472817152,117783023616,0
1793089747.1224594,1793089747.1421509,count1_A2RU14,compilation,5.0,5485699072,117762940928,0
1793089747.6226382,1793089747.6447084,count1_A2RU14,compilation,6.0,5530664960,117566099456,0
1793089748.1229692,1793089748.1445663,count1_A2RU14,compilation,6.0,5586653184,117493166080,0
1793089748.6230717,1793089748.6493146,count1_A2RU14,compilation,84.0,5603000320,117470150656,0
1793089749.1231778,1793089749.1486,count1_A2RU14,compilation,84.0,5666430976,117365186560,0
1793089749.6232815,1793089749.6449292,count1_A2RU14,compilation,6.0,5690200064,117398966272,0
1793089750.1234274,1793089750.1515644,count1_A2RU14,execution,6.0,5795180544,117278535680,0
1793089750.6235673,1793089750.6577623,count1_A2RU14,execution,93.0,5795184640,117284765696,0
1793089751.123702,1793089751.1575356,count1_A2RU14,execution,93.0,5795184640,117288943616,0
1793089751.6242511,1793089751.6508572,count1_A2RU14,execution,94.0,5795184640,117289193472,0
1793089752.1243806,1793089752.1575627,count1_A2RU14,execution,94.0,5795184640,117331144704,0
1793089752.6248138,1793089752.6489446,count1_A2RU14,execution,94.0,5813559296,117314301952,0
1793089753.1253736,1793089753.1581604,count1_A2RU14,execution,94.0,5813559296,117341941760,0
1793089753.6259716,1793089753.6595228,count1_A2RU14,execution,94.0,5813559296,117343305728,0
1793089754.1261902,1793089754.153169,count1_A2RU14,execution,94.0,5813559296,117354536960,0
1793089754.6270497,1793089754.6595023,count1_A2RU14,execution,94.0,5813559296,117368610816,0
1793089755.1274102,1793089755.160549,count1_A2RU14,execution,94.0,5817450496,117371498496,0
1793089755.62753,1793089755.661964,count1_A2RU14,execution,93.0,5817450496,117372284928,0
1793089756.1276505,1793089756.162382,count1_A2RU14,execution,94.0,5817450496,117373190144,0
1793089756.6282768,1793089756.6610293,count1_A2RU14,execution,94.0,5817450496,117378187264,0
1793089757.1285815,1793089757.1579008,count1_A2RU14,execution,94.0,5817450496,117375586304,0
1793089757.6294267,1793089757.6703146,count1_A2RU14,execution,94.0,5817466880,117369270272,0
1793089758.129877,1793089758.1532152,count1_A2RU14,execution,94.0,5817466880,117372895232,0
1793089758.630331,1793089758.6540174,count1_A2RU14,execution,94.0,5817466880,117387771904,0
1793089759.1305997,1793089759.1572776,count1_A2RU14,execution,94.0,5817466880,117379637248,0
1793089759.63101,1793089759.6534808,count1_A2RU14,execution,94.0,5817466880,117389447168,0
1793089760.1314676,1793089760.1644962,count1_A2RU14,execution,94.0,5817495552,117383151616,0
1793089760.6318426,1793089760.6655316,count1_A2RU14,execution,94.0,5817495552,117421010944,0
1793089761.132116,1793089761.1637144,count1_A2RU14,execution,94.0,5817495552,117438009344,0
1793089761.6322196,1793089761.6635492,count1_A2RU14,execution,94.0,5817499648,117465579520,0
1793089762.132478,1793089762.1624174,count1_A2RU14,prediction,94.0,5817499648,117457059840,0
1793089762.6325934,1793089762.658095,count1_A2RU14,execution,92.0,5817499648,117460344832,0
1793089763.1328115,1793089763.1710503,count1_A2RU14,execution,94.0,5817499648,117486243840,0
1793089763.6330912,1793089763.6687777,count1_A2RU14,execution,94.0,5817499648,117511163904,0
1793089764.1333756,1793089764.1714356,count1_A2RU14,execution,94.0,5817499648,117502951424,0
1793089764.633868,1793089764.658475,count1_A2RU14,execution,94.0,5817499648,117510455296,0
1793089765.133968,1793089765.173847,count1_A2RU14,execution,94.0,5819248640,117510680576,0
1793089765.634328,1793089765.6623006,count1_A2RU14,execution,94.0,5819248640,117522186240,0
1793089766.134644,1793089766.1683993,count1_A2RU14,execution,94.0,5819248640,117519843328,0
1793089766.6352139,1793089766.6682937,count1_A2RU14,execution,94.0,5819248640,117524508672,0
1793089767.1354933,1793089767.1600876,count1_A2RU14,prediction,94.0,5819248640,117747564544,0
1793089767.6356883,1793089767.6676803,count1_A2RU14,execution,94.0,5819248640,117733507072,0
1793089768.1358883,1793089768.1628535,count1_A2RU14,execution,94.0,5819248640,117751918592,0
1793089768.6361039,1793089768.666125,count1_A2RU14,execution,94.0,5819248640,117763379200,0
1793089769.136394,1793089769.1761997,count1_A2RU14,execution,94.0,5819248640,117760188416,0
1793089769.6372309,1793089769.6703498,count1_A2RU14,execution,94.0,5819248640,117777199104,0
1793089770.1374485,1793089770.1672082,count1_A2RU14,execution,94.0,5819256832,117774827520,0
1793089770.6379387,1793089770.6700544,count1_A2RU14,execution,93.0,5819256832,117762351104,0
1793089771.1384578,1793089771.1640873,count1_A2RU14,execution,93.0,5819256832,117756379136,0
1793089771.6386542,1793089771.6635487,count1_A2RU14,execution,94.0,5819256832,117760172032,0
1793089772.139005,1793089772.165321,count1_A2RU14,execution,94.0,5819260928,117760368640,0
1793089772.639222,1793089772.676768,count1_A2RU14,execution,94.0,5819265024,117763043328,0
1793089773.139798,1793089773.1645942,count1_A2RU14,execution,94.0,5819265024,117755084800,0
1793089773.6399786,1793089773.6797342,count1_A2RU14,execution,94.0,5819265024,117743943680,0
1793089774.140295,1793089774.1654913,count1_A2RU14,execution,94.0,5819265024,117746716672,0
1793089774.640947,1793089774.6738513,count1_A2RU14,execution,93.0,5819265024,117768433664,0
1793089775.1413748,1793089775.1658554,count1_A2RU14,execution,93.0,5819265024,117765287936,0
1793089775.6416953,1793089775.6679611,count1_A2RU14,execution,93.0,5819265024,117774327808,0
1793089776.1420386,1793089776.167244,count1_A2RU14,execution,93.0,5819265024,117793685504,0
1793089776.6422858,1793089776.6776283,count1_A2RU14,execution,94.0,5819265024,117779664896,0
1793089777.1426554,1793089777.1746037,count1_A2RU14,execution,94.0,5819265024,117784231936,0
1793089777.642879,1793089777.6681218,count1_A2RU14,execution,94.0,5819269120,117788323840,0
1793089778.1430216,1793089778.173311,count1_A2RU14,execution,94.0,5819269120,117789265920,0
1793089778.6435654,1793089778.6739962,count1_A2RU14,execution,94.0,5819269120,117775179776,0
1793089779.143894,1793089779.1693594,count1_A2RU14,execution,94.0,5819269120,117790339072,0
1793089779.6440122,1793089779.671044,count1_A2RU14,execution,94.0,5819269120,117782753280,0
1793089780.144503,1793089780.1799145,count1_A2RU14,execution,94.0,5819273216,117774594048,0
1793089780.6446407,1793089780.669222,count1_A2RU14,execution,93.0,5819281408,117774692352,0
1793089781.1449938,1793089781.1775513,count1_A2RU14,execution,93.0,5819281408,117783117824,0
1793089781.6452947,1793089781.6714458,count1_A2RU14,execution,94.0,5819281408,117782249472,0
1793089782.1457577,1793089782.1899014,count1_A2RU14,execution,94.0,5819281408,117765427200,0
1793089782.645941,1793089782.6837285,count1_A2RU14,execution,94.0,5819285504,117762039808,0
1793089783.1464918,1793089783.18577,count1_A2RU14,execution,94.0,5819285504,117798113280,0
1793089783.6469805,1793089783.6765282,count1_A2RU14,execution,94.0,5819285504,117795426304,0
1793089784.147247,1793089784.1802514,count1_A2RU14,execution,94.0,5819285504,117778878464,0
1793089784.6477158,1793089784.6800115,count1_A2RU14,execution,94.0,5819285504,117790892032,0
1793089785.148016,1793089785.1766508,count1_A2RU14,execution,94.0,5819285504,117801840640,0
1793089785.6481404,1793089785.6779072,count1_A2RU14,execution,93.0,5819285504,117801304064,0
1793089786.1484742,1793089786.175084,count1_A2RU14,execution,93.0,5819285504,117801496576,0
1793089786.6487074,1793089786.6827333,count1_A2RU14,execution,94.0,5819285504,117792677888,0
1793089787.1490672,1793089787.1760309,count1_A2RU14,execution,94.0,5819285504,117785616384,0
1793089787.6495712,1793089787.6743803,count1_A2RU14,execution,94.0,5819285504,117786464256,0
1793089788.1498744,1793089788.1853428,count1_A2RU14,execution,94.0,5819285504,117787832320,0
1793089788.6500676,1793089788.6838715,count1_A2RU14,execution,93.0,5819285504,117791424512,0
1793089789.1501613,1793089789.1787274,count1_A2RU14,execution,93.0,5819285504,117801172992,0
1793089789.6506267,1793089789.6810386,count1_A2RU14,execution,94.0,5819285504,117805260800,0
1793089790.1511838,1793089790.1764932,count1_A2RU14,execution,94.0,5819285504,117807124480,0
1793089790.651675,1793089790.682306,count1_A2RU14,execution,93.0,5819285504,117796995072,0
1793089791.152202,1793089791.1812313,count1_A2RU14,execution,93.0,5819285504,117813538816,0
1793089791.6523888,1793089791.690316,count1_A2RU14,execution,94.0,5819285504,117816504320,0
1793089792.1526866,1793089792.1848447,count1_A2RU14,execution,94.0,5819285504,117800366080,0
1793089792.6531749,1793089792.6747499,count1_A2RU14,execution,94.0,5819289600,117784985600,0
1793089793.1536283,1793089793.1845334,count1_A2RU14,execution,94.0,5819289600,117775937536,0
1793089793.6537817,1793089793.6822796,count1_A2RU14,execution,95.0,5819289600,117766410240,0
1793089794.1540039,1793089794.1862319,count1_A2RU14,execution,95.0,5819289600,117768110080,0
1793089794.6542969,1793089794.6859376,count1_A2RU14,execution,94.0,5819289600,117775962112,0
1793089795.1549792,1793089795.176755,count1_A2RU14,execution,94.0,5819289600,117779075072,0
1793089795.6552732,1793089795.6913173,count1_A2RU14,execution,93.0,5819289600,117767221248,0
1793089796.1555648,1793089796.1947598,count1_A2RU14,execution,93.0,5819289600,117751099392,0
1793089796.6558619,1793089796.7025518,count1_A2RU14,execution,94.0,5819289600,117748199424,0
1793089797.1564667,1793089797.18478,count1_A2RU14,execution,94.0,5819289600,117740298240,0
1793089797.6570542,1793089797.684596,count1_A2RU14,execution,94.0,5819289600,117738082304,0
1793089798.1572025,1793089798.1927738,count1_A2RU14,execution,94.0,5819289600,117737082880,0
1793089798.6577537,1793089798.689828,count1_A2RU14,execution,94.0,5819289600,117742129152,0
1793089799.15787,1793089799.1887286,count1_A2RU14,execution,94.0,5819289600,117742055424,0
1793089799.6584425,1793089799.6888978,count1_A2RU14,execution,94.0,5819289600,117745713152,0
1793089800.1585515,1793089800.1806116,count1_A2RU14,between_queries,94.0,5821509632,117750202368,0
1793089800.6586502,1793089800.6807559,count2_A6NFY7,preprocessing,8.0,5837553664,117759557632,0
1793089801.1592894,1793089801.1837816,count2_A6NFY7,execution,8.0,5986566144,117597216768,0
1793089801.659404,1793089801.690082,count2_A6NFY7,execution,92.0,5986566144,117603037184,0
1793089802.1597211,1793089802.1932218,count2_A6NFY7,execution,92.0,5986566144,117618094080,0
1793089802.6604662,1793089802.6915255,count2_A6NFY7,execution,94.0,5986566144,117622681600,0
1793089803.1610208,1793089803.19395,count2_A6NFY7,execution,94.0,5986566144,117625880576,0
1793089803.6611724,1793089803.6875932,count2_A6NFY7,execution,94.0,5986570240,117638180864,0
1793089804.161298,1793089804.1940157,count2_A6NFY7,execution,94.0,5986570240,117622267904,0
1793089804.6618578,1793089804.6847768,count2_A6NFY7,execution,94.0,5986570240,117620703232,0
1793089805.161985,1793089805.1953943,count2_A6NFY7,execution,94.0,5986570240,117622149120,0
1793089805.6621106,1793089805.6855152,count2_A6NFY7,execution,94.0,5986570240,117624885248,0
1793089806.1625612,1793089806.1942906,count2_A6NFY7,execution,94.0,5986578432,117621997568,0
1793089806.6626801,1793089806.7007275,count2_A6NFY7,execution,93.0,5986578432,117629808640,0
1793089807.1628108,1793089807.1926064,count2_A6NFY7,execution,93.0,5986578432,117628452864,0
1793089807.6630359,1793089807.6906815,count2_A6NFY7,execution,94.0,5986578432,117621587968,0
1793089808.1631851,1793089808.1874,count2_A6NFY7,execution,94.0,5986578432,117623267328,0
1793089808.6636484,1793089808.6888883,count2_A6NFY7,execution,94.0,5986578432,117607440384,0
1793089809.1640918,1793089809.1956022,count2_A6NFY7,execution,94.0,5986578432,117591785472,0
1793089809.6642363,1793089809.6881764,count2_A6NFY7,execution,94.0,5986578432,117598441472,0
1793089810.1643546,1793089810.1977942,count2_A6NFY7,execution,94.0,5986578432,117600382976,0
1793089810.664459,1793089810.7060266,count2_A6NFY7,execution,94.0,5986578432,117601345536,0
1793089811.1645606,1793089811.1853015,count2_A6NFY7,execution,94.0,5986578432,117606375424,0
1793089811.6647096,1793089811.6941688,count2_A6NFY7,execution,92.0,5986578432,117615980544,0
1793089812.1648197,1793089812.1993437,count2_A6NFY7,execution,92.0,5986578432,117596647424,0
1793089812.665011,1793089812.690359,count2_A6NFY7,execution,94.0,5986578432,117587283968,0
1793089813.1653118,1793089813.2044253,count2_A6NFY7,execution,94.0,5986578432,117599084544,0
1793089813.6654286,1793089813.6933372,count2_A6NFY7,execution,94.0,5986578432,117588471808,0
1793089814.1655285,1793089814.2001004,count2_A6NFY7,execution,94.0,5986578432,117586837504,0
1793089814.66596,1793089814.6907086,count2_A6NFY7,execution,94.0,5986582528,117628227584,0
1793089815.1660688,1793089815.2016437,count2_A6NFY7,execution,94.0,5986582528,117633961984,0
1793089815.6661804,1793089815.7011974,count2_A6NFY7,execution,94.0,5986582528,117628186624,0
1793089816.166406,1793089816.193475,count2_A6NFY7,execution,94.0,5986582528,117629120512,0
1793089816.6665232,1793089816.6975088,count2_A6NFY7,execution,92.0,5986582528,117640785920,0
1793089817.1666346,1793089817.2026198,count2_A6NFY7,execution,92.0,5986582528,117641805824,0
1793089817.6667295,1793089817.6964612,count2_A6NFY7,execution,94.0,5986582528,117643902976,0
1793089818.1672945,1793089818.1978421,count2_A6NFY7,execution,94.0,5986582528,117648195584,0
1793089818.6678112,1793089818.6926696,count2_A6NFY7,execution,94.0,5986582528,117634097152,0
1793089819.1679397,1793089819.204997,count2_A6NFY7,execution,94.0,5986582528,117623406592,0
1793089819.6687925,1793089819.702373,count2_A6NFY7,execution,94.0,5986582528,117613355008,0
1793089820.1695285,1793089820.2028806,count2_A6NFY7,execution,94.0,5986582528,117623230464,0
1793089820.6702898,1793089820.7067256,count2_A6NFY7,execution,94.0,5986582528,117614804992,0
1793089821.1704085,1793089821.1961887,count2_A6NFY7,execution,94.0,5986684928,117613969408,0
1793089821.6706889,1793089821.7064438,count2_A6NFY7,execution,93.0,5986684928,117619150848,0
1793089822.1712565,1793089822.2009418,count2_A6NFY7,execution,93.0,5986684928,117603880960,0
1793089822.6718068,1793089822.702456,count2_A6NFY7,execution,94.0,5986684928,117597798400,0
1793089823.1719239,1793089823.2014556,count2_A6NFY7,execution,94.0,5986684928,117588320256,0
1793089823.672029,1793089823.692979,count2_A6NFY7,execution,94.0,5986684928,117594816512,0
1793089824.1721451,1793089824.2033365,count2_A6NFY7,execution,94.0,5986684928,117591080960,0
1793089824.6722803,1793089824.7033355,count2_A6NFY7,execution,94.0,5986684928,117578510336,0
1793089825.1724143,1793089825.2025726,count2_A6NFY7,execution,94.0,5986684928,117570310144,0
1793089825.6726294,1793089825.7165992,count2_A6NFY7,execution,94.0,5986684928,117588881408,0
1793089826.1727464,1793089826.1950274,count2_A6NFY7,execution,94.0,5986693120,117590155264,0
1793089826.6730194,1793089826.7022629,count2_A6NFY7,execution,93.0,5986693120,117597462528,0
1793089827.1735406,1793089827.2096941,count2_A6NFY7,execution,93.0,5986693120,117611757568,0
1793089827.6739478,1793089827.7113183,count2_A6NFY7,execution,94.0,5986693120,117621993472,0
1793089828.174053,1793089828.2100503,count2_A6NFY7,execution,94.0,5986693120,117604433920,0
1793089828.6743734,1793089828.7014644,count2_A6NFY7,execution,94.0,5986693120,117619683328,0
1793089829.174898,1793089829.198453,count2_A6NFY7,execution,94.0,5986693120,117619859456,0
1793089829.6754196,1793089829.710918,count2_A6NFY7,execution,94.0,5986697216,117617602560,0
1793089830.175867,1793089830.2071311,count2_A6NFY7,execution,94.0,5986697216,117602988032,0
1793089830.6760056,1793089830.7016253,count2_A6NFY7,execution,94.0,5986697216,117606977536,0
1793089831.1764247,1793089831.2016847,count2_A6NFY7,execution,94.0,5986729984,117605724160,0
1793089831.6769757,1793089831.7038155,count2_A6NFY7,execution,93.0,5986729984,117605941248,0
1793089832.1777716,1793089832.203396,count2_A6NFY7,execution,93.0,5986729984,117605871616,0
1793089832.6783683,1793089832.7138839,count2_A6NFY7,execution,94.0,5986729984,117614063616,0
1793089833.1787138,1793089833.2069967,count2_A6NFY7,execution,94.0,5986729984,117614288896,0
1793089833.679184,1793089833.7106936,count2_A6NFY7,execution,94.0,5986729984,117624152064,0
1793089834.1797545,1793089834.2095592,count2_A6NFY7,execution,94.0,5986729984,117636354048,0
1793089834.6805162,1793089834.7124527,count2_A6NFY7,execution,94.0,5986729984,117642366976,0
1793089835.1806018,1793089835.2124424,count2_A6NFY7,execution,94.0,5986729984,117634007040,0
1793089835.6811607,1793089835.711387,count2_A6NFY7,execution,94.0,5986729984,117651255296,0
1793089836.1813731,1793089836.2061176,count2_A6NFY7,execution,94.0,5986729984,117634932736,0
1793089836.6816545,1793089836.71332,count2_A6NFY7,execution,92.0,5986729984,117635391488,0
1793089837.1820552,1793089837.2182033,count2_A6NFY7,execution,92.0,5986729984,117645058048,0
1793089837.682176,1793089837.708058,count2_A6NFY7,execution,94.0,5986729984,117645029376,0
1793089838.1823063,1793089838.2079487,count2_A6NFY7,execution,94.0,5986729984,117645234176,0
1793089838.6824455,1793089838.7052739,count2_A6NFY7,execution,94.0,5986734080,117647138816,0
1793089839.1825502,1793089839.2190478,count2_A6NFY7,execution,94.0,5986734080,117646970880,0
1793089839.6826577,1793089839.7152693,count2_A6NFY7,execution,94.0,5986734080,117645012992,0
1793089840.1827617,1793089840.2152455,count2_A6NFY7,execution,94.0,5986734080,117642649600,0
1793089840.6828654,1793089840.7124736,count2_A6NFY7,execution,94.0,5986734080,117639495680,0
1793089841.1830637,1793089841.2084439,count2_A6NFY7,execution,94.0,5986770944,117644615680,0
1793089841.683182,1793089841.7082434,count2_A6NFY7,execution,92.0,5986770944,117653164032,0
1793089842.1833043,1793089842.2089336,count2_A6NFY7,execution,92.0,5986770944,117636751360,0
1793089842.6835427,1793089842.7165399,count2_A6NFY7,execution,94.0,5986770944,117655068672,0
1793089843.183774,1793089843.2133386,count2_A6NFY7,execution,94.0,5986770944,117645021184,0
1793089843.6838858,1793089843.7122695,count2_A6NFY7,execution,94.0,5986770944,117641138176,0
1793089844.1840775,1793089844.218024,count2_A6NFY7,execution,94.0,5986770944,117638524928,0
1793089844.6843991,1793089844.7175143,count2_A6NFY7,execution,94.0,5986770944,117623238656,0
1793089845.1847324,1793089845.2162578,count2_A6NFY7,execution,94.0,5986770944,117639839744,0
1793089845.6849642,1793089845.7175434,count2_A6NFY7,execution,94.0,5986770944,117641781248,0
1793089846.1850734,1793089846.2135453,count2_A6NFY7,execution,94.0,5986770944,117635493888,0
1793089846.6853547,1793089846.7200048,count2_A6NFY7,execution,92.0,5986770944,117618724864,0
1793089847.18565,1793089847.2187548,count2_A6NFY7,execution,92.0,5986770944,117606146048,0
1793089847.6858475,1793089847.7202585,count2_A6NFY7,execution,94.0,5986770944,117589311488,0
1793089848.186088,1793089848.2114131,count2_A6NFY7,execution,94.0,5986770944,117594509312,0
1793089848.686267,1793089848.715793,count2_A6NFY7,execution,94.0,5986770944,117601624064,0
1793089849.1864648,1793089849.2198794,count2_A6NFY7,execution,94.0,5986770944,117600837632,0
1793089849.6866803,1793089849.7152367,count2_A6NFY7,execution,94.0,5986770944,117604298752,0
1793089850.1868985,1793089850.223332,count2_A6NFY7,execution,94.0,5986770944,117587451904,0
1793089850.6870427,1793089850.7147787,count2_A6NFY7,execution,94.0,5986770944,117584965632,0
1793089851.1873338,1793089851.208594,count2_A6NFY7,between_queries,94.0,5869756416,117706948608,0
1793089851.6876278,1793089851.7109408,count3_O14519,preprocessing,6.0,5882376192,117710036992,0
1793089852.187936,1793089852.217352,count3_O14519,execution,6.0,6016307200,117570228224,0
1793089852.688223,1793089852.7155404,count3_O14519,execution,94.0,6016307200,117553143808,0
1793089853.1884265,1793089853.220079,count3_O14519,execution,94.0,6016307200,117536043008,0
1793089853.6889362,1793089853.7218673,count3_O14519,execution,94.0,6016307200,117520596992,0
1793089854.1894035,1793089854.2102273,count3_O14519,execution,94.0,6016311296,117537918976,0
1793089854.689992,1793089854.723789,count3_O14519,execution,67.0,6016311296,117533003776,0
1793089855.1902301,1793089855.2255437,count3_O14519,execution,67.0,6016311296,117533204480,0
1793089855.6907585,1793089855.7260904,count3_O14519,execution,94.0,6016311296,117541527552,0
1793089856.1910577,1793089856.2218773,count3_O14519,execution,94.0,6016311296,117539721216,0
1793089856.6915677,1793089856.7144341,count3_O14519,execution,94.0,6016315392,117557047296,0
1793089857.1919274,1793089857.2235932,count3_O14519,execution,94.0,6016315392,117564772352,0
1793089857.6922624,1793089857.7339635,count3_O14519,execution,94.0,6016319488,117563232256,0
1793089858.1924,1793089858.2287421,count3_O14519,execution,94.0,6016319488,117564497920,0
1793089858.692915,1793089858.7239325,count3_O14519,execution,95.0,6016319488,117578526720,0
1793089859.1934361,1793089859.2212884,count3_O14519,execution,95.0,6016327680,117586317312,0
1793089859.6939998,1793089859.724917,count3_O14519,execution,93.0,6016327680,117578776576,0
1793089860.1946816,1793089860.2233338,count3_O14519,execution,93.0,6016327680,117586857984,0
1793089860.6950135,1793089860.7200484,count3_O14519,execution,94.0,6016327680,117593567232,0
1793089861.195356,1793089861.2217307,count3_O14519,execution,94.0,6016327680,117593817088,0
1793089861.6957123,1793089861.7240837,count3_O14519,execution,94.0,6016557056,117594865664,0
1793089862.1961675,1793089862.2274666,count3_O14519,execution,94.0,6016557056,117597581312,0
1793089862.6962826,1793089862.7293804,count3_O14519,execution,94.0,6016557056,117603733504,0
1793089863.1967561,1793089863.2268648,count3_O14519,execution,94.0,6016557056,117590958080,0
1793089863.6972063,1793089863.7244644,count3_O14519,execution,94.0,6016557056,117591592960,0
1793089864.1974463,1793089864.2242415,count3_O14519,execution,94.0,6016557056,117591162880,0
1793089864.6979008,1793089864.7343526,count3_O14519,execution,92.0,6016557056,117602861056,0
1793089865.198212,1793089865.2302265,count3_O14519,execution,92.0,6016557056,117586038784,0
1793089865.6986403,1793089865.7361355,count3_O14519,execution,95.0,6016557056,117598814208,0
1793089866.199088,1793089866.2242577,count3_O14519,execution,95.0,6016557056,117620830208,0
1793089866.6992924,1793089866.7275796,count3_O14519,execution,94.0,6016561152,117612335104,0
1793089867.1996665,1793089867.2242298,count3_O14519,execution,94.0,6016561152,117604712448,0
1793089867.7004094,1793089867.737094,count3_O14519,execution,94.0,6016561152,117597548544,0
1793089868.2011743,1793089868.2287772,count3_O14519,execution,94.0,6016561152,117555507200,0
1793089868.7017307,1793089868.7253401,count3_O14519,prediction,94.0,6016565248,117431844864,0
1793089869.2025638,1793089869.2380047,count3_O14519,execution,94.0,6016565248,117357907968,0
1793089869.7027965,1793089869.7265413,count3_O14519,execution,94.0,6016565248,117217783808,0
1793089870.2030969,1793089870.2274075,count3_O14519,execution,94.0,6016565248,117179686912,0
1793089870.703216,1793089870.727978,count3_O14519,execution,94.0,6016565248,117271470080,0
1793089871.2039979,1793089871.226256,count3_O14519,execution,94.0,6016671744,117292593152,0
1793089871.7047503,1793089871.7307668,count3_O14519,execution,72.0,6016671744,117293617152,0
1793089872.205095,1793089872.2317255,count3_O14519,execution,72.0,6016671744,117293559808,0
1793089872.7052512,1793089872.7349093,count3_O14519,execution,95.0,6016671744,117315411968,0
1793089873.2056942,1793089873.237105,count3_O14519,execution,95.0,6016671744,117304332288,0
1793089873.7064617,1793089873.7331233,count3_O14519,execution,94.0,6016675840,117307355136,0
1793089874.2072365,1793089874.2380877,count3_O14519,execution,94.0,6016675840,117304455168,0
1793089874.7076929,1793089874.7346842,count3_O14519,execution,94.0,6016675840,117307195392,0
1793089875.208208,1793089875.242452,count3_O14519,execution,94.0,6016675840,117309087744,0
1793089875.7084727,1793089875.7373548,count3_O14519,execution,94.0,6016675840,117317189632,0
1793089876.2090223,1793089876.2308779,count3_O14519,execution,94.0,6016679936,117317652480,0
1793089876.709387,1793089876.741858,count3_O14519,execution,92.0,6016679936,117323792384,0
1793089877.2101712,1793089877.2402158,count3_O14519,execution,92.0,6016679936,117323448320,0
1793089877.710413,1793089877.7437744,count3_O14519,execution,94.0,6016679936,117334024192,0
1793089878.211068,1793089878.2399874,count3_O14519,execution,94.0,6016679936,117315956736,0
1793089878.711636,1793089878.7356412,count3_O14519,execution,94.0,6016688128,117324820480,0
1793089879.21219,1793089879.246291,count3_O14519,execution,94.0,6016688128,117310099456,0
1793089879.7124615,1793089879.7475991,count3_O14519,execution,94.0,6016688128,117313519616,0
1793089880.2131789,1793089880.2478106,count3_O14519,execution,94.0,6016688128,117297197056,0
1793089880.713727,1793089880.7463863,count3_O14519,execution,94.0,6016688128,117296779264,0
1793089881.2142937,1793089881.2452192,count3_O14519,execution,94.0,6016692224,117301833728,0
1793089881.7149618,1793089881.7404175,count3_O14519,execution,93.0,6016692224,117304672256,0
1793089882.2152748,1793089882.2470016,count3_O14519,execution,93.0,6016692224,117289226240,0
1793089882.7154849,1793089882.745331,count3_O14519,execution,94.0,6016692224,117291171840,0
1793089883.215583,1793089883.2420092,count3_O14519,prediction,94.0,6016700416,117282791424,0
1793089883.7156818,1793089883.745432,count3_O14519,execution,78.0,6016700416,117267398656,0
1793089884.2164314,1793089884.2434876,count3_O14519,execution,78.0,6016700416,117250600960,0
1793089884.71701,1793089884.7514212,count3_O14519,execution,94.0,6016700416,117255421952,0
1793089885.2175536,1793089885.251902,count3_O14519,execution,94.0,6016700416,117263695872,0
1793089885.7177324,1793089885.740872,count3_O14519,execution,94.0,6016700416,117262385152,0
1793089886.2178502,1793089886.2520556,count3_O14519,execution,94.0,6016700416,117243715584,0
1793089886.7179785,1793089886.753451,count3_O14519,execution,95.0,6016700416,117240795136,0
1793089887.2180946,1793089887.2523873,count3_O14519,execution,95.0,6016700416,117244637184,0
1793089887.7182379,1793089887.761584,count3_O14519,execution,94.0,6016700416,117248749568,0
1793089888.2185395,1793089888.246531,count3_O14519,execution,94.0,6016700416,117264183296,0
1793089888.718891,1793089888.7528439,count3_O14519,execution,92.0,6016700416,117285244928,0
1793089889.2194664,1793089889.2464583,count3_O14519,execution,92.0,6016700416,117291175936,0
1793089889.7200313,1793089889.7470298,count3_O14519,execution,94.0,6016700416,117291094016,0
1793089890.22054,1793089890.2528567,count3_O14519,execution,94.0,6016700416,117296611328,0
1793089890.7210996,1793089890.7460253,count3_O14519,execution,94.0,6016720896,117308559360,0
1793089891.2218332,1793089891.2627723,count3_O14519,execution,94.0,6016720896,117290905600,0
1793089891.7223778,1793089891.757829,count3_O14519,execution,94.0,6016720896,117293404160,0
1793089892.2229362,1793089892.2582052,count3_O14519,execution,94.0,6016720896,117292761088,0
1793089892.7236564,1793089892.7569497,count3_O14519,execution,94.0,6016720896,117305954304,0
1793089893.2244065,1793089893.2544732,count3_O14519,execution,94.0,6016720896,117305315328,0
1793089893.7250683,1793089893.7613928,count3_O14519,execution,93.0,6016720896,117305708544,0
1793089894.2254171,1793089894.2554076,count3_O14519,execution,93.0,6016720896,117310214144,0
1793089894.7257373,1793089894.7554772,count3_O14519,execution,94.0,6016720896,117313064960,0
1793089895.2263653,1793089895.2611203,count3_O14519,execution,94.0,6016720896,117309980672,0
1793089895.7265854,1793089895.7618108,count3_O14519,execution,94.0,6016720896,117319364608,0
1793089896.2269747,1793089896.251838,count3_O14519,execution,94.0,6016720896,117319819264,0
1793089896.7272072,1793089896.7524958,count3_O14519,execution,94.0,6016720896,117326262272,0
1793089897.2275665,1793089897.2531922,count3_O14519,execution,94.0,6016720896,117328666624,0
1793089897.728003,1793089897.7501814,count3_O14519,execution,94.0,6016724992,117315686400,0
1793089898.2286596,1793089898.2562873,count3_O14519,execution,94.0,6016724992,117334327296,0
1793089898.729596,1793089898.766452,count3_O14519,execution,94.0,6016724992,117325963264,0
1793089899.2305477,1793089899.2639606,count3_O14519,execution,94.0,6016724992,117342928896,0
1793089899.7307503,1793089899.7701337,count3_O14519,execution,94.0,6016724992,117349797888,0
1793089900.231706,1793089900.2539384,count3_O14519,between_queries,94.0,5870911488,117480079360,0
1793089900.7318037,1793089900.752217,count4_P01270,preprocessing,3.0,5897695232,117434998784,0
1793089901.2322643,1793089901.2762766,count4_P01270,execution,3.0,5995044864,117312049152,0
1793089901.7323847,1793089901.7556639,count4_P01270,execution,92.0,5995048960,117327015936,0
1793089902.2325199,1793089902.2582948,count4_P01270,execution,92.0,5995048960,117322305536,0
1793089902.7326798,1793089902.758931,count4_P01270,execution,94.0,5995048960,117333475328,0
1793089903.2328024,1793089903.2682438,count4_P01270,execution,94.0,5995048960,117330997248,0
1793089903.7329986,1793089903.7669702,count4_P01270,execution,94.0,5995053056,117345259520,0
1793089904.23314,1793089904.2645628,count4_P01270,execution,94.0,5995053056,117347078144,0
1793089904.7333593,1793089904.7650464,count4_P01270,execution,94.0,5995053056,117348503552,0
1793089905.2335832,1793089905.2647848,count4_P01270,execution,94.0,5995053056,117349507072,0
1793089905.7338593,1793089905.757649,count4_P01270,execution,94.0,5995061248,117351936000,0
1793089906.2342083,1793089906.263596,count4_P01270,execution,94.0,5995061248,117369589760,0
1793089906.7346153,1793089906.7681596,count4_P01270,execution,94.0,5995061248,117369339904,0
1793089907.2349002,1793089907.2642891,count4_P01270,execution,94.0,5995061248,117353168896,0
1793089907.7351642,1793089907.764306,count4_P01270,execution,94.0,5995061248,117335932928,0
1793089908.2353055,1793089908.2620516,count4_P01270,execution,94.0,5995065344,117537476608,0
1793089908.7355373,1793089908.770143,count4_P01270,execution,76.0,5995069440,117549731840,0
1793089909.2359388,1793089909.27971,count4_P01270,execution,76.0,5995073536,117557891072,0
1793089909.736214,1793089909.7739034,count4_P01270,execution,94.0,5995073536,117584420864,0
1793089910.236527,1793089910.2716665,count4_P01270,execution,94.0,5995073536,117590294528,0
1793089910.7366643,1793089910.761876,count4_P01270,execution,94.0,5995491328,117573591040,0
1793089911.237115,1793089911.2628903,count4_P01270,execution,94.0,5995495424,117557817344,0
1793089911.7373838,1793089911.7660394,count4_P01270,execution,94.0,5995495424,117581025280,0
1793089912.2375977,1793089912.2703032,count4_P01270,execution,94.0,5995495424,117596811264,0
1793089912.7379239,1793089912.766625,count4_P01270,execution,94.0,5995495424,117597536256,0
1793089913.2384481,1793089913.2613633,count4_P01270,execution,94.0,5995499520,117580984320,0
1793089913.7390184,1793089913.7643633,count4_P01270,execution,93.0,5995499520,117586595840,0
1793089914.239348,1793089914.2689135,count4_P01270,execution,93.0,5995499520,117603635200,0
1793089914.739828,1793089914.765615,count4_P01270,execution,94.0,5995503616,117616881664,0
1793089915.240372,1793089915.2708037,count4_P01270,execution,94.0,5995503616,117600346112,0
1793089915.7406738,1793089915.764448,count4_P01270,execution,94.0,5995507712,117589385216,0
1793089916.2410948,1793089916.2786996,count4_P01270,execution,94.0,5995507712,117617086464,0
1793089916.7416666,1793089916.7720368,count4_P01270,execution,94.0,5995507712,117600763904,0
1793089917.2421064,1793089917.2805934,count4_P01270,execution,94.0,5995507712,117616812032,0
1793089917.7426956,1793089917.7757778,count4_P01270,execution,94.0,5995507712,117631111168,0
1793089918.2432337,1793089918.2734022,count4_P01270,execution,94.0,5995511808,117632593920,0
1793089918.7434995,1793089918.7713933,count4_P01270,execution,93.0,5995511808,117632229376,0
1793089919.2438064,1793089919.26978,count4_P01270,execution,93.0,5995511808,117623152640,0
1793089919.7441163,1793089919.7699127,count4_P01270,execution,94.0,5995511808,117623574528,0
1793089920.2444155,1793089920.2663343,count4_P01270,execution,94.0,5995634688,117610053632,0
1793089920.7446983,1793089920.7768867,count4_P01270,execution,63.0,5995634688,117597900800,0
1793089921.2449677,1793089921.2722912,count4_P01270,execution,63.0,5995634688,117597581312,0
1793089921.7454827,1793089921.7727757,count4_P01270,execution,94.0,5995634688,117583699968,0
1793089922.2457783,1793089922.280062,count4_P01270,execution,94.0,5995634688,117592076288,0
1793089922.7460809,1793089922.766633,count4_P01270,execution,94.0,5995638784,117588967424,0
1793089923.2466867,1793089923.288568,count4_P01270,execution,94.0,5995638784,117604360192,0
1793089923.7472215,1793089923.7827199,count4_P01270,execution,94.0,5995638784,117607714816,0
1793089924.2473507,1793089924.2804809,count4_P01270,execution,94.0,5995638784,117610291200,0
1793089924.7475414,1793089924.7787914,count4_P01270,execution,94.0,5995638784,117591261184,0
1793089925.2477398,1793089925.2682338,count4_P01270,execution,94.0,5995638784,117582868480,0
1793089925.7480412,1793089925.7756357,count4_P01270,execution,92.0,5995638784,117586104320,0
1793089926.2482555,1793089926.2729251,count4_P01270,execution,92.0,5995638784,117582217216,0
1793089926.7484734,1793089926.774477,count4_P01270,execution,94.0,5995638784,117588353024,0
1793089927.2485752,1793089927.2841694,count4_P01270,execution,94.0,5995638784,117588017152,0
1793089927.7489877,1793089927.7742648,count4_P01270,execution,94.0,5995638784,117581053952,0
1793089928.2497296,1793089928.280131,count4_P01270,execution,94.0,5995638784,117582548992,0
1793089928.7499738,1793089928.7830222,count4_P01270,execution,95.0,5995638784,117588189184,0
1793089929.250497,1793089929.2837312,count4_P01270,execution,95.0,5995638784,117582999552,0
1793089929.7508583,1793089929.7841556,count4_P01270,execution,94.0,5995638784,117588393984,0
1793089930.251155,1793089930.2853715,count4_P01270,execution,94.0,5995675648,117580820480,0
1793089930.7513664,1793089930.7821443,count4_P01270,execution,93.0,5995675648,117608144896,0
1793089931.251856,1793089931.2843044,count4_P01270,execution,93.0,5995675648,117622112256,0
1793089931.7524328,1793089931.782462,count4_P01270,execution,94.0,5995675648,117608935424,0
1793089932.2530313,1793089932.288875,count4_P01270,execution,94.0,5995675648,117617311744,0
1793089932.7536466,1793089932.782702,count4_P01270,execution,94.0,5995683840,117600555008,0
1793089933.2538729,1793089933.2856498,count4_P01270,execution,94.0,5995683840,117599604736,0
1793089933.7544606,1793089933.786511,count4_P01270,execution,94.0,5995683840,117607346176,0
1793089934.2549877,1793089934.289202,count4_P01270,execution,94.0,5995683840,117590491136,0
1793089934.7552378,1793089934.7794056,count4_P01270,execution,94.0,5995692032,117585190912,0
1793089935.2553847,1793089935.2838688,count4_P01270,execution,94.0,5995692032,117594873856,0
1793089935.7554927,1793089935.781792,count4_P01270,execution,95.0,5995692032,117595742208,0
1793089936.2558687,1793089936.2830312,count4_P01270,execution,95.0,5995692032,117591453696,0
1793089936.7564142,1793089936.7802088,count4_P01270,execution,94.0,5995692032,117594697728,0
1793089937.2569802,1793089937.2791615,count4_P01270,execution,94.0,5995692032,117613191168,0
1793089937.7575085,1793089937.793948,count4_P01270,execution,69.0,5995692032,117616521216,0
1793089938.258448,1793089938.2911506,count4_P01270,execution,69.0,5995692032,117617221632,0
1793089938.758887,1793089938.7969925,count4_P01270,execution,94.0,5995692032,117641031680,0
1793089939.2592793,1793089939.2946136,count4_P01270,execution,94.0,5995692032,117647900672,0
1793089939.7597294,1793089939.7815175,count4_P01270,execution,94.0,5995749376,117639888896,0
1793089940.2601328,1793089940.2950397,count4_P01270,execution,94.0,5995749376,117623570432,0
1793089940.7606013,1793089940.794147,count4_P01270,execution,94.0,5995749376,117621846016,0
1793089941.2609708,1793089941.303337,count4_P01270,execution,94.0,5995749376,117621002240,0
1793089941.7614536,1793089941.8005614,count4_P01270,execution,94.0,5995749376,117617819648,0
1793089942.2617846,1793089942.2863173,count4_P01270,execution,94.0,5995749376,117616975872,0
1793089942.76235,1793089942.7970023,count4_P01270,execution,93.0,5995757568,117602369536,0
1793089943.2629175,1793089943.301273,count4_P01270,execution,93.0,5995757568,117623271424,0
1793089943.763459,1793089943.801891,count4_P01270,execution,94.0,5995757568,117623205888,0
1793089944.2641995,1793089944.3029053,count4_P01270,execution,94.0,5995757568,117620977664,0
1793089944.7646058,1793089944.8003109,count4_P01270,execution,94.0,5995757568,117611573248,0
1793089945.2647567,1793089945.2998328,count4_P01270,execution,94.0,5995757568,117615710208,0
1793089945.7649882,1793089945.7985392,count4_P01270,execution,94.0,5995757568,117599563776,0
1793089946.2655532,1793089946.3002145,count4_P01270,execution,94.0,5995757568,117599498240,0
1793089946.765766,1793089946.7883847,count4_P01270,execution,94.0,5995757568,117608542208,0
1793089947.2660236,1793089947.293767,count4_P01270,execution,94.0,5995757568,117611421696,0
1793089947.7665827,1793089947.7927434,count4_P01270,execution,94.0,5995757568,117603278848,0
1793089948.266695,1793089948.294086,count4_P01270,execution,94.0,5995757568,117616295936,0
1793089948.7668014,1793089948.7963657,count4_P01270,execution,94.0,5995757568,117599576064,0
1793089949.2669146,1793089949.2954397,count4_P01270,between_queries,94.0,5877293056,117734629376,0
1793089949.7670255,1793089949.7903051,1LYZ,between_queries,36.0,5891923968,117662515200,0
1793089950.2723296,1793089950.878261,1LYZ,compilation,4.0,6118744064,117442793472,0
1793089950.8837752,1793089951.05652,1LYZ,compilation,4.0,6118744064,117425467392,0
1793089951.3839023,1793089951.4063616,1LYZ,compilation,4.0,6120308736,117427634176,0
1793089951.8840265,1793089951.9052114,1LYZ,compilation,4.0,6120308736,117410467840,0
1793089952.3841321,1793089952.4039142,1LYZ,compilation,4.0,6120308736,117407346688,0
1793089952.8842237,1793089952.9058752,1LYZ,compilation,4.0,6120308736,117410947072,0
1793089953.384435,1793089953.406507,1LYZ,compilation,4.0,6120308736,117402972160,0
1793089953.8846323,1793089953.907264,1LYZ,compilation,4.0,6120308736,117415899136,0
1793089954.3851452,1793089954.407064,1LYZ,compilation,4.0,6120308736,117405716480,0
1793089954.8853226,1793089954.907784,1LYZ,compilation,4.0,6120308736,117420175360,0
1793089955.3855155,1793089955.4066358,1LYZ,compilation,4.0,6120308736,117428789248,0
1793089955.8856337,1793089955.907139,1LYZ,compilation,4.0,6120308736,117433147392,0
1793089956.3857431,1793089956.4087718,1LYZ,compilation,3.0,6120308736,117429813248,0
1793089956.885847,1793089956.9100378,1LYZ,compilation,3.0,6128742400,117267222528,0
1793089957.3860154,1793089957.4116983,1LYZ,compilation,50.0,6157279232,117265219584,0
1793089957.8862455,1793089957.911816,1LYZ,compilation,50.0,6183804928,117244235776,0
1793089958.3863595,1793089958.4314759,1LYZ,compilation,4.0,6212517888,117183688704,0
1793089958.8864737,1793089958.9074943,1LYZ,compilation,4.0,6248710144,117002293248,0
1793089959.386966,1793089959.4063418,1LYZ,compilation,7.0,6268477440,117021003776,0
1793089959.8871498,1793089959.9094377,1LYZ,execution,7.0,6317350912,116991676416,0
1793089960.3877301,1793089960.4110684,1LYZ,execution,94.0,6317350912,116993388544,0
1793089960.8879793,1793089960.9203415,1LYZ,execution,94.0,6317350912,116983693312,0
1793089961.38817,1793089961.4160306,1LYZ,execution,94.0,6317350912,116991135744,0
1793089961.8884175,1793089961.9243872,1LYZ,execution,94.0,6317350912,116986978304,0
1793089962.3887846,1793089962.4145548,1LYZ,execution,94.0,6317350912,116981968896,0
1793089962.8889759,1793089962.911723,1LYZ,execution,94.0,6317355008,117003988992,0
1793089963.3890874,1793089963.4295883,1LYZ,execution,94.0,6317355008,117020135424,0
1793089963.8893883,1793089963.9131029,1LYZ,execution,94.0,6317355008,117032267776,0
1793089964.3895903,1793089964.4168794,1LYZ,execution,94.0,6317355008,117052264448,0
1793089964.8903449,1793089964.9173245,1LYZ,execution,94.0,6317355008,117084876800,0
1793089965.3908782,1793089965.4227843,1LYZ,execution,94.0,6317355008,117083607040,0
1793089965.8916068,1793089965.912239,1LYZ,execution,94.0,6317359104,117112197120,0
1793089966.3921278,1793089966.4314334,1LYZ,execution,94.0,6317359104,117128593408,0
1793089966.8926702,1793089966.939602,1LYZ,execution,94.0,6317359104,117145464832,0
1793089967.3931305,1793089967.4216766,1LYZ,execution,94.0,6317359104,117127884800,0
1793089967.8938334,1793089967.9245067,1LYZ,execution,94.0,6317359104,117138030592,0
1793089968.394621,1793089968.4264257,1LYZ,execution,94.0,6317359104,117119881216,0
1793089968.895163,1793089968.9195285,1LYZ,execution,94.0,6317359104,117133680640,0
1793089969.3952968,1793089969.4215355,1LYZ,execution,94.0,6317359104,117151522816,0
1793089969.8955975,1793089969.9294336,1LYZ,execution,94.0,6317359104,117182468096,0
1793089970.3957984,1793089970.420891,1LYZ,execution,95.0,6317359104,117199040512,0
1793089970.8963408,1793089970.9227607,1LYZ,execution,95.0,6317359104,117187592192,0
1793089971.3964813,1793089971.42575,1LYZ,execution,94.0,6317359104,117196591104,0
1793089971.897216,1793089971.9196439,1LYZ,execution,94.0,6317363200,117210861568,0
1793089972.3975034,1793089972.4346857,1LYZ,execution,94.0,6317363200,117203025920,0
1793089972.8977213,1793089972.9245005,1LYZ,execution,94.0,6317363200,117207724032,0
1793089973.39803,1793089973.4261029,1LYZ,execution,94.0,6317363200,117207588864,0
1793089973.8983355,1793089973.9242182,1LYZ,execution,94.0,6317363200,117216940032,0
1793089974.3986146,1793089974.431283,1LYZ,execution,94.0,6317363200,117209436160,0
1793089974.8990407,1793089974.9204237,1LYZ,execution,94.0,6317363200,117221052416,0
1793089975.3991475,1793089975.436148,1LYZ,execution,94.0,6317363200,117204553728,0
1793089975.8992872,1793089975.9223027,1LYZ,execution,94.0,6317363200,117229182976,0
1793089976.3993914,1793089976.4236712,1LYZ,execution,94.0,6317363200,117228253184,0
1793089976.8999443,1793089976.9274898,1LYZ,execution,94.0,6317363200,117230088192,0
1793089977.4000497,1793089977.4425504,1LYZ,execution,94.0,6317363200,117239713792,0
1793089977.9003696,1793089977.9257443,1LYZ,execution,94.0,6317363200,117238505472,0
1793089978.4004977,1793089978.4489007,1LYZ,execution,94.0,6317363200,117251284992,0
1793089978.9006298,1793089978.9284406,1LYZ,execution,94.0,6317363200,117263478784,0
1793089979.4010522,1793089979.4387505,1LYZ,execution,94.0,6317363200,117261365248,0
1793089979.901168,1793089979.9367988,1LYZ,execution,94.0,6317363200,117266092032,0
1793089980.4012973,1793089980.4357936,1LYZ,execution,94.0,6317363200,117249925120,0
1793089980.9014256,1793089980.9282813,1LYZ,execution,94.0,6317363200,117253058560,0
1793089981.4015467,1793089981.4366527,1LYZ,execution,94.0,6317363200,117234061312,0
1793089981.9017537,1793089981.9263556,1LYZ,execution,94.0,6317363200,117248385024,0
1793089982.4018793,1793089982.4417748,1LYZ,execution,94.0,6317363200,117231489024,0
1793089982.9020088,1793089982.9385264,1LYZ,execution,94.0,6317363200,117234302976,0
1793089983.402213,1793089983.4337873,1LYZ,execution,94.0,6317363200,117222125568,0
1793089983.9023283,1793089983.9264333,1LYZ,execution,94.0,6317363200,117227581440,0
1793089984.402509,1793089984.4441724,1LYZ,execution,95.0,6317363200,117224120320,0
1793089984.9032943,1793089984.9374764,1LYZ,execution,95.0,6317363200,117243645952,0
1793089985.4036775,1793089985.4342413,1LYZ,execution,94.0,6317363200,117228421120,0
1793089985.9037948,1793089985.937741,1LYZ,execution,94.0,6317367296,117248827392,0
1793089986.4039164,1793089986.4295654,1LYZ,execution,94.0,6317367296,117248585728,0
1793089986.9046907,1793089986.9280453,1LYZ,execution,94.0,6317371392,117239316480,0
1793089987.4048045,1793089987.441869,1LYZ,execution,94.0,6317371392,117246767104,0
1793089987.904962,1793089987.9367104,1LYZ,execution,94.0,6317371392,117234413568,0
1793089988.4057026,1793089988.436428,1LYZ,execution,94.0,6317371392,117226229760,0
1793089988.9064896,1793089988.9439816,1LYZ,execution,94.0,6317371392,117227270144,0
1793089989.4070327,1793089989.435166,1LYZ,execution,94.0,6317371392,117222236160,0
1793089989.9078012,1793089989.9427829,1LYZ,execution,94.0,6317371392,117225623552,0
1793089990.4080412,1793089990.4413733,1LYZ,execution,94.0,6317371392,117228322816,0
1793089990.9087338,1793089990.9428132,1LYZ,execution,94.0,6317371392,117233512448,0
1793089991.408997,1793089991.448607,1LYZ,execution,94.0,6317371392,117237243904,0
1793089991.9092276,1793089991.9474354,1LYZ,execution,94.0,6317371392,117227544576,0
1793089992.409438,1793089992.4365294,1LYZ,execution,94.0,6317371392,117095235584,0
1793089992.9099429,1793089992.9357233,1LYZ,execution,94.0,6317371392,117043240960,0
1793089993.4103968,1793089993.441534,1LYZ,execution,94.0,6317371392,116936179712,0
1793089993.910927,1793089993.94907,1LYZ,execution,94.0,6317371392,116937474048,0
1793089994.411654,1793089994.439025,1LYZ,execution,94.0,6317371392,116917116928,0
1793089994.9125142,1793089994.9378517,1LYZ,execution,94.0,6317371392,116911439872,0
1793089995.413004,1793089995.4471889,1LYZ,execution,94.0,6317371392,117101240320,0
1793089995.9134195,1793089995.9396336,1LYZ,execution,94.0,6317371392,117117927424,0
1793089996.413774,1793089996.4372416,1LYZ,execution,94.0,6317371392,117119213568,0
1793089996.9141536,1793089996.9422877,1LYZ,execution,94.0,6317371392,117131776000,0
1793089997.4142785,1793089997.4424732,1LYZ,execution,94.0,6317371392,117115019264,0
1793089997.9145353,1793089997.9426787,1LYZ,execution,94.0,6317371392,117138378752,0
1793089998.4149303,1793089998.460001,1LYZ,execution,94.0,6317371392,117144559616,0
1793089998.9151268,1793089998.9428651,1LYZ,execution,94.0,6317375488,117164957696,0
1793089999.4156938,1793089999.449646,1LYZ,execution,94.0,6317375488,117163921408,0
1793089999.9159083,1793089999.9444134,1LYZ,execution,94.0,6317375488,117163229184,0
1793090000.416686,1793090000.4526587,1LYZ,execution,94.0,6317375488,117144584192,0
1793090000.917063,1793090000.9521625,1LYZ,execution,94.0,6317375488,117148798976,0
1793090001.4174588,1793090001.4495986,1LYZ,execution,94.0,6317375488,117152989184,0
1793090001.9177392,1793090001.9415014,1LYZ,execution,94.0,6317383680,117170647040,0
1793090002.4181488,1793090002.4499462,1LYZ,execution,94.0,6317383680,117164720128,0
1793090002.9182522,1793090002.9523256,1LYZ,execution,94.0,6317383680,117166018560,0
1793090003.4185436,1793090003.4536145,1LYZ,execution,94.0,6317383680,117181157376,0
1793090003.9190998,1793090003.942441,1LYZ,execution,94.0,6317383680,117194055680,0
1793090004.4193835,1793090004.4524379,1LYZ,execution,94.0,6317383680,117177282560,0
1793090004.9198263,1793090004.946457,1LYZ,execution,94.0,6317383680,117186396160,0
1793090005.4203556,1793090005.4533324,1LYZ,execution,94.0,6317383680,117194027008,0
1793090005.9204938,1793090005.9580243,1LYZ,execution,94.0,6317383680,117180375040,0
1793090006.4205995,1793090006.450252,1LYZ,execution,94.0,6317383680,117199446016,0
1793090006.920774,1793090006.9475744,1LYZ,execution,94.0,6317383680,117210210304,0
1793090007.4210694,1793090007.4600582,1LYZ,execution,94.0,6317383680,117218459648,0
1793090007.9213881,1793090007.9468236,1LYZ,execution,94.0,6317383680,117238263808,0
1793090008.4219887,1793090008.445544,1LYZ,execution,94.0,6317383680,117245841408,0
1793090008.9221067,1793090008.957006,1LYZ,execution,94.0,6317383680,117248454656,0
1793090009.4227257,1793090009.4599135,1LYZ,execution,94.0,6317383680,117248585728,0
1793090009.922976,1793090009.954859,1LYZ,execution,94.0,6317383680,117256163328,0
1793090010.423386,1793090010.4550319,1LYZ,execution,94.0,6317383680,117247901696,0
1793090010.9239078,1793090010.949273,1LYZ,execution,94.0,6317383680,117266104320,0
1793090011.4247386,1793090011.4621382,1LYZ,execution,94.0,6317383680,117273976832,0
1793090011.9249933,1793090011.9673042,1LYZ,execution,94.0,6317383680,117281312768,0
1793090012.4254622,1793090012.456415,1LYZ,execution,94.0,6317383680,117289394176,0
1793090012.925785,1793090012.9602501,1LYZ,execution,94.0,6317383680,117275643904,0
1793090013.4261281,1793090013.460504,1LYZ,execution,94.0,6317383680,117276979200,0
1793090013.9265916,1793090013.9529912,1LYZ,execution,94.0,6317383680,117268496384,0
1793090014.4267094,1793090014.4587655,1LYZ,execution,95.0,6317383680,117259800576,0
1793090014.9270704,1793090014.9678369,1LYZ,execution,95.0,6317383680,117280243712,0
1793090015.4275877,1793090015.4579554,1LYZ,execution,94.0,6317383680,117276401664,0
1793090015.9281313,1793090015.9578202,1LYZ,execution,94.0,6317383680,117290442752,0
1793090016.4285195,1793090016.4542046,1LYZ,execution,94.0,6317383680,117289902080,0
1793090016.9287176,1793090016.9518948,1LYZ,execution,94.0,6317383680,117292306432,0
1793090017.4295068,1793090017.4605951,1LYZ,execution,94.0,6317383680,117291511808,0
1793090017.9298224,1793090017.9574027,1LYZ,execution,94.0,6317383680,117290713088,0
1793090018.4300337,1793090018.461533,1LYZ,execution,94.0,6317383680,117274177536,0
1793090018.9301596,1793090018.968994,1LYZ,execution,94.0,6317383680,117274910720,0
1793090019.4304745,1793090019.4565442,1LYZ,execution,94.0,6317383680,117273792512,0
1793090019.9354072,1793090019.9592106,1LYZ,between_queries,94.0,6317383680,117285142528,0
1793090020.4407444,1793090020.6954644,1TIM,preprocessing,5.0,6318673920,117282856960,0
1793090020.9414923,1793090020.963772,1TIM,between_queries,5.0,6401810432,117144653824,0
1793090021.442252,1793090021.4678996,1TIM,preprocessing,9.0,6667431936,116898603008,0
1793090022.0007672,1793090022.2604935,1TIM,compilation,9.0,6870376448,116720623616,0
1793090022.5009007,1793090022.5212433,1TIM,compilation,8.0,6871171072,116709928960,0
1793090023.0011098,1793090023.021294,1TIM,compilation,8.0,6871171072,116693835776,0
1793090023.501212,1793090023.5213444,1TIM,compilation,6.0,6871171072,116706603008,0
1793090024.0013177,1793090024.0219452,1TIM,compilation,6.0,6871171072,116720611328,0
1793090024.5019953,1793090024.5231752,1TIM,compilation,5.0,6871171072,116730982400,0
1793090025.0022013,1793090025.023625,1TIM,compilation,5.0,6871171072,116729139200,0
1793090025.5024252,1793090025.5242832,1TIM,compilation,4.0,6871171072,116740743168,0
1793090026.002531,1793090026.024582,1TIM,compilation,4.0,6871171072,116738252800,0
1793090026.5027237,1793090026.5254083,1TIM,compilation,4.0,6871171072,116730769408,0
1793090027.0028243,1793090027.0232291,1TIM,compilation,4.0,6871171072,116734242816,0
1793090027.5029268,1793090027.5252185,1TIM,compilation,4.0,6871171072,116700626944,0
1793090028.0029988,1793090028.0301309,1TIM,compilation,4.0,6875738112,116427919360,0
1793090028.5031085,1793090028.530061,1TIM,compilation,78.0,6883749888,116355080192,0
1793090029.0036263,1793090029.032282,1TIM,compilation,78.0,7105568768,116059160576,0
1793090029.5040958,1793090029.5241926,1TIM,compilation,5.0,7145848832,115905929216,0
1793090030.0045335,1793090030.0263543,1TIM,compilation,5.0,7145848832,115913318400,0
1793090030.5049896,1793090030.5249925,1TIM,compilation,4.0,7145848832,115826094080,0
1793090031.0053952,1793090031.027946,1TIM,compilation,4.0,7135891456,115749318656,0
1793090031.5059125,1793090031.5263383,1TIM,compilation,4.0,7123271680,115632582656,0
1793090032.006112,1793090032.0282068,1TIM,compilation,4.0,7145635840,115453001728,0
1793090032.5062342,1793090032.5347724,1TIM,compilation,5.0,7179538432,115372675072,0
1793090033.0064516,1793090033.026404,1TIM,compilation,5.0,7183302656,115410952192,0
1793090033.507004,1793090033.540854,1TIM,execution,94.0,7366479872,115119931392,0
1793090034.007566,1793090034.0527568,1TIM,execution,94.0,7366479872,115125444608,0
1793090034.5080872,1793090034.5513313,1TIM,execution,94.0,7366479872,115150729216,0
1793090035.0084658,1793090035.0423973,1TIM,execution,94.0,7366479872,115150737408,0
1793090035.5087852,1793090035.547891,1TIM,execution,95.0,7366479872,115382108160,0
1793090036.009347,1793090036.046237,1TIM,execution,95.0,7366479872,115376795648,0
1793090036.5100384,1793090036.544866,1TIM,execution,95.0,7366479872,115406094336,0
1793090037.0103579,1793090037.041041,1TIM,execution,95.0,7366479872,115389984768,0
1793090037.5106428,1793090037.5424128,1TIM,execution,95.0,7366479872,115396116480,0
1793090038.0111947,1793090038.0407102,1TIM,execution,95.0,7366479872,115419385856,0
1793090038.5117266,1793090038.5377913,1TIM,execution,95.0,7366479872,115408048128,0
1793090039.0122948,1793090039.039796,1TIM,execution,95.0,7366492160,115436617728,0
1793090039.5128644,1793090039.540096,1TIM,execution,94.0,7366492160,115451305984,0
1793090040.013106,1793090040.0584543,1TIM,execution,94.0,7366492160,115495157760,0
1793090040.5136998,1793090040.5561144,1TIM,execution,94.0,7366492160,115488444416,0
1793090041.0139482,1793090041.0475879,1TIM,execution,94.0,7366492160,115488587776,0
1793090041.514413,1793090041.550707,1TIM,execution,95.0,7366492160,115490824192,0
1793090042.014624,1793090042.051033,1TIM,execution,95.0,7366492160,115514163200,0
1793090042.5150623,1793090042.5525022,1TIM,execution,95.0,7366492160,115542650880,0
1793090043.0153637,1793090043.0530562,1TIM,execution,95.0,7366492160,115524681728,0
1793090043.515671,1793090043.5590146,1TIM,execution,95.0,7366492160,115544879104,0
1793090044.0162818,1793090044.0534902,1TIM,execution,95.0,7366492160,115547971584,0
1793090044.5166774,1793090044.5414386,1TIM,execution,94.0,7366492160,115556032512,0
1793090045.017425,1793090045.0483563,1TIM,execution,94.0,7366492160,115564322816,0
1793090045.5179956,1793090045.546369,1TIM,execution,95.0,7366492160,115581313024,0
1793090046.0184436,1793090046.049057,1TIM,execution,95.0,7366492160,115605110784,0
1793090046.518562,1793090046.5455787,1TIM,execution,95.0,7366492160,115613683712,0
1793090047.0190291,1793090047.0462122,1TIM,execution,95.0,7366492160,115648774144,0
1793090047.5194306,1793090047.547784,1TIM,execution,95.0,7366492160,115659337728,0
1793090048.0198858,1793090048.049804,1TIM,execution,95.0,7366492160,115674419200,0
1793090048.5201974,1793090048.5477266,1TIM,execution,94.0,7366492160,115684536320,0
1793090049.0207636,1793090049.0471885,1TIM,execution,94.0,7366492160,115717914624,0
1793090049.5208793,1793090049.5471315,1TIM,execution,95.0,7366492160,115728048128,0
1793090050.0218036,1793090050.048174,1TIM,execution,95.0,7366492160,115743162368,0
1793090050.5219114,1793090050.5484242,1TIM,execution,94.0,7366492160,115767021568,0
1793090051.0224605,1793090051.061565,1TIM,execution,94.0,7366492160,115780993024,0
1793090051.5230453,1793090051.5527852,1TIM,execution,95.0,7366492160,115789615104,0
1793090052.0234084,1793090052.0541363,1TIM,execution,95.0,7366492160,115800645632,0
1793090052.5237575,1793090052.5492678,1TIM,execution,95.0,7366492160,115810848768,0
1793090053.024062,1793090053.052786,1TIM,execution,95.0,7366492160,115815661568,0
1793090053.5246825,1793090053.553159,1TIM,execution,95.0,7366492160,115823456256,0
1793090054.025069,1793090054.0516827,1TIM,execution,95.0,7366492160,115810529280,0
1793090054.525689,1793090054.5559762,1TIM,execution,95.0,7366492160,115853733888,0
1793090055.0259466,1793090055.0560937,1TIM,execution,95.0,7366492160,115844694016,0
1793090055.5265503,1793090055.576236,1TIM,execution,95.0,7366492160,115863904256,0
1793090056.0266778,1793090056.0647557,1TIM,execution,95.0,7367376896,115865899008,0
1793090056.527024,1793090056.5600548,1TIM,execution,95.0,7367376896,115900243968,0
1793090057.0272927,1793090057.0595822,1TIM,execution,95.0,7367376896,115918237696,0
1793090057.5274067,1793090057.5562186,1TIM,execution,95.0,7367376896,115946467328,0
1793090058.0277636,1793090058.0589478,1TIM,execution,95.0,7367376896,115933388800,0
1793090058.5280821,1793090058.562303,1TIM,execution,94.0,7367376896,115949576192,0
1793090059.0282974,1793090059.064506,1TIM,execution,94.0,7367376896,115958099968,0
1793090059.5288126,1793090059.5610807,1TIM,execution,94.0,7367380992,115953975296,0
1793090060.0292907,1793090060.0644512,1TIM,execution,94.0,7367380992,115970371584,0
1793090060.5296507,1793090060.5661163,1TIM,execution,94.0,7367380992,115986579456,0
1793090061.0297794,1793090061.0585856,1TIM,execution,94.0,7367380992,115985354752,0
1793090061.5302408,1793090061.5562508,1TIM,execution,68.0,7367380992,115994116096,0
1793090062.0307093,1793090062.0690134,1TIM,execution,68.0,7367380992,115992166400,0
1793090062.530899,1793090062.5606031,1TIM,execution,94.0,7367380992,115981742080,0
1793090063.031205,1793090063.061579,1TIM,execution,94.0,7367380992,115987836928,0
1793090063.5313075,1793090063.5661314,1TIM,execution,95.0,7367380992,115971829760,0
1793090064.0314934,1793090064.0628688,1TIM,execution,95.0,7367380992,115978620928,0
1793090064.5316863,1793090064.5586767,1TIM,execution,94.0,7367380992,115995131904,0
1793090065.0317957,1793090065.0654874,1TIM,execution,94.0,7367380992,115985690624,0
1793090065.5319202,1793090065.5587845,1TIM,execution,94.0,7367380992,116004773888,0
1793090066.0323272,1793090066.0611959,1TIM,execution,94.0,7367380992,116004352000,0
1793090066.5327911,1793090066.5629733,1TIM,execution,95.0,7367380992,116005859328,0
1793090067.0329235,1793090067.0657887,1TIM,execution,95.0,7367380992,116014309376,0
1793090067.5330365,1793090067.5588038,1TIM,execution,93.0,7367380992,116014809088,0
1793090068.0335011,1793090068.071226,1TIM,execution,93.0,7367380992,116004786176,0
1793090068.5336037,1793090068.5709362,1TIM,execution,95.0,7367380992,115990093824,0
1793090069.033856,1793090069.0705316,1TIM,execution,95.0,7367380992,115986599936,0
1793090069.5343661,1793090069.5707953,1TIM,execution,94.0,7367380992,115999199232,0
1793090070.0348046,1793090070.0709407,1TIM,execution,94.0,7367385088,116000047104,0
1793090070.5353622,1793090070.571322,1TIM,execution,95.0,7367385088,116011282432,0
1793090071.0358057,1793090071.0693989,1TIM,execution,95.0,7367389184,115994554368,0
1793090071.5362895,1793090071.570804,1TIM,execution,94.0,7367389184,115991232512,0
1793090072.0368578,1793090072.0691094,1TIM,execution,94.0,7367389184,115997487104,0
1793090072.5369732,1793090072.5706723,1TIM,prediction,94.0,7367397376,116002308096,0
1793090073.0374904,1793090073.0618827,1TIM,execution,94.0,7367397376,116009230336,0
1793090073.537816,1793090073.5685594,1TIM,execution,93.0,7367397376,116019752960,0
1793090074.038299,1793090074.0738652,1TIM,execution,93.0,7367397376,116015980544,0
1793090074.5388548,1793090074.5833094,1TIM,execution,94.0,7367397376,116022669312,0
1793090075.0394847,1793090075.0697494,1TIM,execution,94.0,7367397376,116007383040,0
1793090075.5397198,1793090075.5717769,1TIM,execution,94.0,7367397376,116020502528,0
1793090076.040118,1793090076.0725272,1TIM,execution,94.0,7367397376,116017905664,0
1793090076.5406833,1793090076.5755782,1TIM,execution,95.0,7367397376,116009787392,0
1793090077.0411837,1793090077.0887923,1TIM,execution,95.0,7367397376,116026945536,0
1793090077.5414886,1793090077.5682197,1TIM,execution,94.0,7367397376,116034641920,0
1793090078.0420732,1793090078.0761538,1TIM,execution,94.0,7367397376,116032454656,0
1793090078.5427732,1793090078.568905,1TIM,execution,94.0,7367479296,116044517376,0
1793090079.0433176,1793090079.0849652,1TIM,execution,94.0,7367479296,116028878848,0
1793090079.543893,1793090079.5880175,1TIM,execution,93.0,7367479296,116018630656,0
1793090080.0444398,1793090080.0893898,1TIM,execution,93.0,7367479296,116018311168,0
1793090080.5449402,1793090080.5928981,1TIM,execution,94.0,7367479296,116037271552,0
1793090081.0452623,1793090081.0719438,1TIM,execution,94.0,7367479296,116020695040,0
1793090081.5458124,1793090081.5791936,1TIM,execution,94.0,7367479296,116032643072,0
1793090082.046385,1793090082.088727,1TIM,execution,94.0,7367479296,116016504832,0
1793090082.5466983,1793090082.5923471,1TIM,execution,94.0,7367479296,116036132864,0
1793090083.0472398,1793090083.0755754,1TIM,execution,94.0,7367479296,116036194304,0
1793090083.5475435,1793090083.5804245,1TIM,execution,94.0,7367479296,116044451840,0
1793090084.0477355,1793090084.0873039,1TIM,execution,94.0,7367479296,116049379328,0
1793090084.5481977,1793090084.5807548,1TIM,execution,82.0,7367479296,116051705856,0
1793090085.0487542,1793090085.0745556,1TIM,execution,82.0,7367479296,116034994176,0
1793090085.5492985,1793090085.5833452,1TIM,execution,94.0,7367479296,116049854464,0
1793090086.0496502,1793090086.0938601,1TIM,execution,94.0,7367479296,116030328832,0
1793090086.5502155,1793090086.5893457,1TIM,execution,94.0,7367479296,116030742528,0
1793090087.050572,1793090087.0826666,1TIM,execution,94.0,7367479296,116028743680,0
1793090087.5506918,1793090087.5798907,1TIM,execution,94.0,7367479296,116038889472,0
1793090088.0508482,1793090088.0876966,1TIM,execution,94.0,7367479296,116017803264,0
1793090088.551316,1793090088.5799005,1TIM,execution,94.0,7367479296,115950596096,0
1793090089.0514276,1793090089.0928671,1TIM,execution,94.0,7367479296,115877724160,0
1793090089.5519168,1793090089.5941763,1TIM,execution,95.0,7367479296,115806285824,0
1793090090.0524452,1793090090.088681,1TIM,execution,95.0,7367479296,115664515072,0
1793090090.5525587,1793090090.57657,1TIM,execution,62.0,7367479296,115665448960,0
1793090091.0531085,1793090091.0838256,1TIM,execution,62.0,7367479296,115676229632,0
1793090091.5537143,1793090091.5857885,1TIM,execution,95.0,7367479296,115667460096,0
1793090092.05428,1793090092.0913877,1TIM,execution,95.0,7367479296,115676725248,0
1793090092.5545752,1793090092.5965593,1TIM,execution,95.0,7367479296,115672666112,0
1793090093.0551226,1793090093.106917,1TIM,execution,95.0,7367479296,115813384192,0
1793090093.5554917,1793090093.5934823,1TIM,execution,94.0,7367479296,115822145536,0
1793090094.056059,1793090094.1069765,1TIM,execution,94.0,7367479296,115823980544,0
1793090094.556599,1793090094.598024,1TIM,execution,95.0,7367479296,115852251136,0
1793090095.0571506,1793090095.0856826,1TIM,execution,95.0,7367479296,115864682496,0
1793090095.5576503,1793090095.5867739,1TIM,execution,94.0,7367479296,115861323776,0
1793090096.0584042,1793090096.0862944,1TIM,execution,94.0,7367479296,115853643776,0
1793090096.5587075,1793090096.5810194,1TIM,execution,94.0,7367479296,115840098304,0
1793090097.0594637,1793090097.1011915,1TIM,execution,94.0,7367479296,115847536640,0
1793090097.5598233,1793090097.6041431,1TIM,execution,95.0,7367479296,115848019968,0
1793090098.0601475,1793090098.1060035,1TIM,execution,95.0,7367479296,115853852672,0
1793090098.5604775,1793090098.6098907,1TIM,execution,95.0,7367479296,115886006272,0
1793090099.0610242,1793090099.1123607,1TIM,execution,95.0,7367479296,115897700352,0
1793090099.5618207,1793090099.6102917,1TIM,execution,94.0,7367479296,115928207360,0
1793090100.0624213,1793090100.1085966,1TIM,execution,94.0,7367479296,115945316352,0
1793090100.5630696,1793090100.6090128,1TIM,execution,94.0,7367479296,115945730048,0
1793090101.0631561,1793090101.095752,1TIM,execution,94.0,7367479296,115928182784,0
1793090101.5632663,1793090101.5912669,1TIM,execution,94.0,7367479296,115939975168,0
1793090102.063482,1793090102.0917654,1TIM,execution,94.0,7367512064,115945607168,0
1793090102.5636759,1793090102.6007528,1TIM,execution,94.0,7367512064,115957432320,0
1793090103.064111,1793090103.1087666,1TIM,execution,94.0,7367512064,115972554752,0
1793090103.5643797,1793090103.6069825,1TIM,execution,94.0,7367512064,115985047552,0
1793090104.0645714,1793090104.0992699,1TIM,execution,94.0,7367512064,115983978496,0
1793090104.564778,1793090104.6010964,1TIM,execution,94.0,7367512064,115992641536,0
1793090105.065003,1793090105.1141202,1TIM,execution,94.0,7367512064,116009988096,0
1793090105.5653005,1793090105.6048698,1TIM,execution,95.0,7367512064,116008591360,0
1793090106.0657525,1793090106.100407,1TIM,execution,95.0,7367512064,115996913664,0
1793090106.566301,1793090106.6073241,1TIM,execution,94.0,7367512064,116003651584,0
1793090107.0665615,1793090107.0947638,1TIM,execution,94.0,7367512064,116011597824,0
1793090107.56713,1793090107.5972588,1TIM,execution,94.0,7367512064,116012556288,0
1793090108.0674493,1793090108.0980675,1TIM,execution,94.0,7367512064,116014120960,0
1793090108.5677373,1793090108.6007564,1TIM,execution,94.0,7367512064,115996225536,0
1793090109.0680263,1793090109.0993588,1TIM,execution,94.0,7367512064,115998507008,0
1793090109.5685797,1793090109.6067545,1TIM,execution,94.0,7367512064,115986083840,0
1793090110.0690997,1793090110.1185803,1TIM,execution,94.0,7367512064,115990884352,0
1793090110.5694652,1793090110.6106708,1TIM,execution,94.0,7367512064,115996528640,0
1793090111.0697072,1793090111.1041565,1TIM,execution,94.0,7367512064,116001095680,0
1793090111.5702176,1793090111.5980608,1TIM,execution,94.0,7367512064,116007329792,0
1793090112.0710142,1793090112.1065445,1TIM,execution,94.0,7367512064,116007698432,0
1793090112.5712874,1793090112.6085258,1TIM,execution,95.0,7367512064,116007890944,0
1793090113.0718422,1793090113.1143198,1TIM,execution,95.0,7367512064,116012781568,0
1793090113.5721316,1793090113.6064353,1TIM,execution,94.0,7367516160,116013768704,0
1793090114.0726924,1793090114.102771,1TIM,execution,94.0,7367520256,116028661760,0
1793090114.5730245,1793090114.6127431,1TIM,execution,94.0,7367520256,116036014080,0
1793090115.0735664,1793090115.1082492,1TIM,execution,94.0,7367520256,116032225280,0
1793090115.573879,1793090115.6172252,1TIM,execution,94.0,7367520256,116028174336,0
1793090116.0744162,1793090116.102113,1TIM,execution,94.0,7367520256,116029538304,0
1793090116.5749326,1793090116.6042318,1TIM,execution,94.0,7367520256,116034301952,0
1793090117.0752354,1793090117.1130717,1TIM,execution,94.0,7367520256,116036231168,0
1793090117.575928,1793090117.6088164,1TIM,execution,95.0,7367520256,116043448320,0
1793090118.0766943,1793090118.1142836,1TIM,execution,95.0,7367520256,116048654336,0
1793090118.576825,1793090118.612756,1TIM,execution,95.0,7367520256,116051492864,0
1793090119.0776856,1793090119.135787,1TIM,execution,95.0,7367520256,116035469312,0
1793090119.578167,1793090119.6199107,1TIM,execution,94.0,7367520256,116017643520,0
1793090120.078452,1793090120.120569,1TIM,execution,94.0,7367520256,116034572288,0
1793090120.5788548,1793090120.6246064,1TIM,execution,94.0,7367520256,116027645952,0
1793090121.0791807,1793090121.107044,1TIM,execution,94.0,7367520256,116009197568,0
1793090121.579502,1793090121.6084964,1TIM,execution,94.0,7367520256,115992313856,0
1793090122.0800557,1793090122.1134624,1TIM,execution,94.0,7367520256,115975745536,0
1793090122.5804064,1793090122.619072,1TIM,execution,95.0,7367520256,115967090688,0
1793090123.080911,1793090123.119677,1TIM,execution,95.0,7367520256,115966828544,0
1793090123.581025,1793090123.6147487,1TIM,execution,94.0,7367520256,115968999424,0
1793090124.0811725,1793090124.1091058,1TIM,execution,94.0,7367520256,115952349184,0
1793090124.581395,1793090124.6140957,1TIM,execution,94.0,7367520256,115955912704,0
1793090125.0814972,1793090125.1140323,1TIM,prediction,94.0,7367540736,115974012928,0
1793090125.5817096,1793090125.6070735,1TIM,execution,93.0,7367540736,115975499776,0
1793090126.0819252,1793090126.1120126,1TIM,execution,93.0,7367540736,115983794176,0
1793090126.5823917,1793090126.6159778,1TIM,execution,94.0,7367540736,115986644992,0
1793090127.0829794,1793090127.1177726,1TIM,execution,94.0,7367540736,115968249856,0
1793090127.5833795,1793090127.6235287,1TIM,execution,94.0,7367540736,115968114688,0
1793090128.0835922,1793090128.1246412,1TIM,execution,94.0,7367540736,115982356480,0
1793090128.5840778,1793090128.6202343,1TIM,execution,94.0,7367540736,116003401728,0
1793090129.0846615,1793090129.1184173,1TIM,execution,94.0,7367540736,116008042496,0
1793090129.5854347,1793090129.6224797,1TIM,execution,94.0,7367540736,116005408768,0
1793090130.085721,1793090130.1298146,1TIM,execution,94.0,7367540736,116006207488,0
1793090130.5865,1793090130.6367948,1TIM,execution,94.0,7367540736,116016226304,0
1793090131.0866833,1793090131.1248515,1TIM,execution,94.0,7367540736,115999522816,0
1793090131.5872622,1793090131.6143014,1TIM,execution,93.0,7367540736,115988975616,0
1793090132.087561,1793090132.124018,1TIM,execution,93.0,7367540736,115975491584,0
1793090132.5883327,1793090132.6182477,1TIM,execution,94.0,7367540736,115988148224,0
1793090133.0886533,1793090133.1176214,1TIM,execution,94.0,7367540736,115997028352,0
1793090133.5894437,1793090133.6311984,1TIM,execution,95.0,7367540736,116000043008,0
1793090134.090088,1793090134.1305602,1TIM,execution,95.0,7367540736,116004433920,0
1793090134.590452,1793090134.6311066,1TIM,execution,94.0,7367540736,116011819008,0
1793090135.091018,1793090135.1282,1TIM,execution,94.0,7367540736,116026281984,0
1793090135.5915287,1793090135.6202905,1TIM,execution,94.0,7367540736,116015173632,0
1793090136.0920732,1793090136.1255224,1TIM,execution,94.0,7367540736,116022784000,0
1793090136.592939,1793090136.6359968,1TIM,execution,94.0,7367540736,116040241152,0
1793090137.0936704,1793090137.119841,1TIM,execution,94.0,7367540736,116030910464,0
1793090137.594173,1793090137.6263433,1TIM,execution,94.0,7367540736,116028620800,0
1793090138.0947292,1793090138.1282914,1TIM,execution,94.0,7367540736,116020813824,0
1793090138.595498,1793090138.6284988,1TIM,execution,94.0,7367540736,116023762944,0
1793090139.0962462,1793090139.13312,1TIM,execution,94.0,7367540736,116015779840,0
1793090139.5965521,1793090139.644575,1TIM,execution,94.0,7367540736,116009336832,0
1793090140.0971003,1793090140.1261299,1TIM,execution,94.0,7367540736,116016541696,0
1793090140.597631,1793090140.6259341,1TIM,execution,95.0,7367540736,116016812032,0
1793090141.097795,1793090141.14083,1TIM,execution,95.0,7367540736,116013711360,0
1793090141.5981007,1793090141.6355174,1TIM,execution,95.0,7367540736,116016635904,0
1793090142.0988994,1793090142.1327558,1TIM,execution,95.0,7367540736,116024500224,0
1793090142.5990064,1793090142.6285138,1TIM,execution,94.0,7367540736,116031455232,0
1793090143.0994744,1793090143.1257193,1TIM,execution,94.0,7367544832,116040560640,0
1793090143.6002512,1793090143.6515334,1TIM,execution,94.0,7367544832,116052418560,0
1793090144.1006944,1793090144.1297324,1TIM,execution,94.0,7367544832,116048441344,0
1793090144.6009946,1793090144.6372483,1TIM,execution,95.0,7367544832,116046520320,0
1793090145.1015778,1793090145.137301,1TIM,execution,95.0,7367544832,116041711616,0
1793090145.602168,1793090145.6478517,1TIM,execution,94.0,7367544832,116047593472,0
1793090146.102273,1793090146.156792,1TIM,execution,94.0,7367544832,116069617664,0
1793090146.602376,1793090146.6483748,1TIM,execution,94.0,7367544832,116054487040,0
1793090147.1026664,1793090147.1311498,1TIM,execution,94.0,7367544832,116038574080,0
1793090147.602938,1793090147.6322587,1TIM,execution,94.0,7367544832,116038197248,0
1793090148.103245,1793090148.1485653,1TIM,execution,94.0,7367544832,116036943872,0
1793090148.6187854,1793090148.6427083,1TIM,between_queries,94.0,6978940928,116440014848,0
```
<!-- END_RESTORE_FILE -->


</details>

## file-515cad2b623a

원래 경로: `colabfold/results/af2_20261027_172813_247681/structure_validation.json` · SHA256: `f793e8cb2a383d3b7f9a6fe36f7272cbdf7d8e03930cd897778043b4c130e890`

<details>
<summary>structure_validation.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/structure_validation.json", "sha256": "f793e8cb2a383d3b7f9a6fe36f7272cbdf7d8e03930cd897778043b4c130e890", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "passed": true,
  "files_validated": 35,
  "results": [
    {
      "query": "1UBQ",
      "rank": 1,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_unrelaxed_rank_001_alphafold2_ptm_model_4_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_001_alphafold2_ptm_model_4_seed_042.json",
      "residues": 76,
      "mean_plddt": 95.95881578947369,
      "ptm": 0.84,
      "pdb_sha256": "67c7f47eeb9c385821841972034a9294cf4a9d2494a016c473f32fe76a31adea",
      "finite": true,
      "reference": "/work/data/1UBQ.pdb",
      "aligned_ca": 76,
      "predicted_ca": 76,
      "reference_ca": 76,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.8007980998342538,
      "ca_lddt": 0.9818149717514124,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1UBQ",
      "rank": 2,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_unrelaxed_rank_002_alphafold2_ptm_model_3_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "residues": 76,
      "mean_plddt": 95.43447368421052,
      "ptm": 0.83,
      "pdb_sha256": "bc9218a0b26e21a5b44975adab705a6c64b48016eaab30051fee3d6c388c7097",
      "finite": true,
      "reference": "/work/data/1UBQ.pdb",
      "aligned_ca": 76,
      "predicted_ca": 76,
      "reference_ca": 76,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.7840603889336422,
      "ca_lddt": 0.975635593220339,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1UBQ",
      "rank": 3,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_unrelaxed_rank_003_alphafold2_ptm_model_2_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_003_alphafold2_ptm_model_2_seed_042.json",
      "residues": 76,
      "mean_plddt": 95.20223684210528,
      "ptm": 0.83,
      "pdb_sha256": "320685e514bdac73e3f1ac918f07218f9eaf4630ed09eff4882731e38efb74b3",
      "finite": true,
      "reference": "/work/data/1UBQ.pdb",
      "aligned_ca": 76,
      "predicted_ca": 76,
      "reference_ca": 76,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.8804810051753925,
      "ca_lddt": 0.983933615819209,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1UBQ",
      "rank": 4,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_unrelaxed_rank_004_alphafold2_ptm_model_1_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_004_alphafold2_ptm_model_1_seed_042.json",
      "residues": 76,
      "mean_plddt": 94.98881578947368,
      "ptm": 0.82,
      "pdb_sha256": "7c9e6105f0942207b3d24d84f5d774e4110ede8a95fda4a5a730ff9e2bcb7905",
      "finite": true,
      "reference": "/work/data/1UBQ.pdb",
      "aligned_ca": 76,
      "predicted_ca": 76,
      "reference_ca": 76,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.764832589317443,
      "ca_lddt": 0.9743997175141242,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1UBQ",
      "rank": 5,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_unrelaxed_rank_005_alphafold2_ptm_model_5_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_005_alphafold2_ptm_model_5_seed_042.json",
      "residues": 76,
      "mean_plddt": 94.19986842105263,
      "ptm": 0.81,
      "pdb_sha256": "89a5f17dc39c38acc9ff75c34463bb462fcd7c988c14e4549b1b12dba94f0d31",
      "finite": true,
      "reference": "/work/data/1UBQ.pdb",
      "aligned_ca": 76,
      "predicted_ca": 76,
      "reference_ca": 76,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.7312163814922874,
      "ca_lddt": 0.9791666666666666,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "count1_A2RU14",
      "rank": 1,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_001_alphafold2_ptm_model_5_seed_042.json",
      "residues": 115,
      "mean_plddt": 86.07573913043477,
      "ptm": 0.52,
      "pdb_sha256": "354c0354c3d446feeedc385b59d44311528b9a3621225bf7b5174b00611eb0c1",
      "finite": true
    },
    {
      "query": "count1_A2RU14",
      "rank": 2,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_unrelaxed_rank_002_alphafold2_ptm_model_3_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "residues": 115,
      "mean_plddt": 84.52713043478263,
      "ptm": 0.45,
      "pdb_sha256": "3d8b796077de58544dc7b40a339ec090bd1912a6fc44aa32af1f51e03a53ad02",
      "finite": true
    },
    {
      "query": "count1_A2RU14",
      "rank": 3,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_unrelaxed_rank_003_alphafold2_ptm_model_1_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_003_alphafold2_ptm_model_1_seed_042.json",
      "residues": 115,
      "mean_plddt": 84.12852173913046,
      "ptm": 0.47,
      "pdb_sha256": "95dcd7bc9aeb5d1eb7752085d9057c7be537785fde592b2df7fe94859457c105",
      "finite": true
    },
    {
      "query": "count1_A2RU14",
      "rank": 4,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_unrelaxed_rank_004_alphafold2_ptm_model_4_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_004_alphafold2_ptm_model_4_seed_042.json",
      "residues": 115,
      "mean_plddt": 82.02782608695651,
      "ptm": 0.46,
      "pdb_sha256": "66aa186b5af485815154439f4053c70ba96012dd3a45db03a58f4c18bf88f09b",
      "finite": true
    },
    {
      "query": "count1_A2RU14",
      "rank": 5,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_unrelaxed_rank_005_alphafold2_ptm_model_2_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_005_alphafold2_ptm_model_2_seed_042.json",
      "residues": 115,
      "mean_plddt": 78.95852173913043,
      "ptm": 0.44,
      "pdb_sha256": "256da5fb2f78c991ccdfbeee6ce615fc1fc25df2c39260048fc6dddf59719be1",
      "finite": true
    },
    {
      "query": "count2_A6NFY7",
      "rank": 1,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_unrelaxed_rank_001_alphafold2_ptm_model_4_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_001_alphafold2_ptm_model_4_seed_042.json",
      "residues": 115,
      "mean_plddt": 74.014,
      "ptm": 0.53,
      "pdb_sha256": "a7ddddc2ed6f078819667a163a61e2e0b120551eef55dd51bf8f7c45281fb946",
      "finite": true
    },
    {
      "query": "count2_A6NFY7",
      "rank": 2,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_unrelaxed_rank_002_alphafold2_ptm_model_3_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "residues": 115,
      "mean_plddt": 72.35078260869564,
      "ptm": 0.52,
      "pdb_sha256": "b06eec7230bd63037822369a3ce5eedc64be1eb6ba29a9aa253a74d2ed7226d2",
      "finite": true
    },
    {
      "query": "count2_A6NFY7",
      "rank": 3,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_unrelaxed_rank_003_alphafold2_ptm_model_5_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_003_alphafold2_ptm_model_5_seed_042.json",
      "residues": 115,
      "mean_plddt": 71.7791304347826,
      "ptm": 0.53,
      "pdb_sha256": "a1d40e3f1372fba4c52765e3d01633b9437f6387993c890b6f3159071293b53e",
      "finite": true
    },
    {
      "query": "count2_A6NFY7",
      "rank": 4,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_unrelaxed_rank_004_alphafold2_ptm_model_1_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_004_alphafold2_ptm_model_1_seed_042.json",
      "residues": 115,
      "mean_plddt": 71.34217391304348,
      "ptm": 0.5,
      "pdb_sha256": "c9baebc973f40b7dec670b0ca9bd405e0892e1861e147fcd40041d323db54934",
      "finite": true
    },
    {
      "query": "count2_A6NFY7",
      "rank": 5,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_unrelaxed_rank_005_alphafold2_ptm_model_2_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_005_alphafold2_ptm_model_2_seed_042.json",
      "residues": 115,
      "mean_plddt": 70.32104347826088,
      "ptm": 0.52,
      "pdb_sha256": "d5789f6dfb52e488c66e8720554392eabce4ff5f716fbcd51a34baf239e2b352",
      "finite": true
    },
    {
      "query": "count3_O14519",
      "rank": 1,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_001_alphafold2_ptm_model_1_seed_042.json",
      "residues": 115,
      "mean_plddt": 71.36591304347826,
      "ptm": 0.45,
      "pdb_sha256": "092d03b5decc3f698bd246da78401d8d216f42f5d5083fbe4750234592833411",
      "finite": true
    },
    {
      "query": "count3_O14519",
      "rank": 2,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_unrelaxed_rank_002_alphafold2_ptm_model_3_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "residues": 115,
      "mean_plddt": 70.28773913043479,
      "ptm": 0.46,
      "pdb_sha256": "efc5d9c036bfa4731db21eb4cb9aa28ac302832a0dd3425d301a8bcb0ca4d3b1",
      "finite": true
    },
    {
      "query": "count3_O14519",
      "rank": 3,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_unrelaxed_rank_003_alphafold2_ptm_model_4_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_003_alphafold2_ptm_model_4_seed_042.json",
      "residues": 115,
      "mean_plddt": 70.248,
      "ptm": 0.46,
      "pdb_sha256": "ba1f3721a0b45f400e33903ae6f4a1f86ef5f514ebd679a51e0d85f90596417f",
      "finite": true
    },
    {
      "query": "count3_O14519",
      "rank": 4,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_unrelaxed_rank_004_alphafold2_ptm_model_2_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_004_alphafold2_ptm_model_2_seed_042.json",
      "residues": 115,
      "mean_plddt": 69.95426086956522,
      "ptm": 0.44,
      "pdb_sha256": "bb6c48a0fe1ccb86a3e9a6c6e8a0008a116fb86ba18e5e30b237d529f1151464",
      "finite": true
    },
    {
      "query": "count3_O14519",
      "rank": 5,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_unrelaxed_rank_005_alphafold2_ptm_model_5_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_005_alphafold2_ptm_model_5_seed_042.json",
      "residues": 115,
      "mean_plddt": 69.13730434782607,
      "ptm": 0.45,
      "pdb_sha256": "50a9417a4f0913b39f1e54a3a61d3ab32577f05e0e52c2e0d12931901fbb0491",
      "finite": true
    },
    {
      "query": "count4_P01270",
      "rank": 1,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_001_alphafold2_ptm_model_3_seed_042.json",
      "residues": 115,
      "mean_plddt": 70.2611304347826,
      "ptm": 0.31,
      "pdb_sha256": "5de81314b5963b23d9acce2b28abe7c30c176ac132f6d14f1f7a7508d6fb1e50",
      "finite": true
    },
    {
      "query": "count4_P01270",
      "rank": 2,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_unrelaxed_rank_002_alphafold2_ptm_model_1_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_002_alphafold2_ptm_model_1_seed_042.json",
      "residues": 115,
      "mean_plddt": 69.03426086956522,
      "ptm": 0.33,
      "pdb_sha256": "4fd8dbff888a3fe8a26127c258073d5228de438785e54b5813ffc23a6cd55114",
      "finite": true
    },
    {
      "query": "count4_P01270",
      "rank": 3,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_unrelaxed_rank_003_alphafold2_ptm_model_5_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_003_alphafold2_ptm_model_5_seed_042.json",
      "residues": 115,
      "mean_plddt": 64.03260869565219,
      "ptm": 0.26,
      "pdb_sha256": "e2670d91e5e024a87f377599af4f4ea1af80b9951fe705fe291709ea00186f90",
      "finite": true
    },
    {
      "query": "count4_P01270",
      "rank": 4,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_unrelaxed_rank_004_alphafold2_ptm_model_2_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_004_alphafold2_ptm_model_2_seed_042.json",
      "residues": 115,
      "mean_plddt": 63.79973913043479,
      "ptm": 0.25,
      "pdb_sha256": "31d7676e450ae374b50490861c95c2886e10973b573c5b83f08d14eb0850cf77",
      "finite": true
    },
    {
      "query": "count4_P01270",
      "rank": 5,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_unrelaxed_rank_005_alphafold2_ptm_model_4_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_005_alphafold2_ptm_model_4_seed_042.json",
      "residues": 115,
      "mean_plddt": 59.158,
      "ptm": 0.25,
      "pdb_sha256": "84cdf8fe750d5eda65977df1f17f3a1c1c1820afb86ddb4d45ba68bf4c820bcd",
      "finite": true
    },
    {
      "query": "1LYZ",
      "rank": 1,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_001_alphafold2_ptm_model_5_seed_042.json",
      "residues": 129,
      "mean_plddt": 98.19992248062015,
      "ptm": 0.91,
      "pdb_sha256": "f4de7c562c30d6a001b41dbaa4f0cc89cfc472d17459e4ad4de8272e925dd305",
      "finite": true,
      "reference": "/work/data/1LYZ.pdb",
      "aligned_ca": 129,
      "predicted_ca": 129,
      "reference_ca": 129,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.5459262682162478,
      "ca_lddt": 0.949366032699366,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1LYZ",
      "rank": 2,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_unrelaxed_rank_002_alphafold2_ptm_model_4_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_002_alphafold2_ptm_model_4_seed_042.json",
      "residues": 129,
      "mean_plddt": 97.84589147286822,
      "ptm": 0.9,
      "pdb_sha256": "1d0e341dae81c631a694c03c1ca45cf115d00dfce772eb88e94822c3fe97cadc",
      "finite": true,
      "reference": "/work/data/1LYZ.pdb",
      "aligned_ca": 129,
      "predicted_ca": 129,
      "reference_ca": 129,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.5251691972532709,
      "ca_lddt": 0.9530363697030364,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1LYZ",
      "rank": 3,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_unrelaxed_rank_003_alphafold2_ptm_model_3_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_003_alphafold2_ptm_model_3_seed_042.json",
      "residues": 129,
      "mean_plddt": 97.79372093023255,
      "ptm": 0.9,
      "pdb_sha256": "44f829c346fb97e831238dae3b789d917ab12937a99e5a993b2d20eb52fac07e",
      "finite": true,
      "reference": "/work/data/1LYZ.pdb",
      "aligned_ca": 129,
      "predicted_ca": 129,
      "reference_ca": 129,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.5100524240163119,
      "ca_lddt": 0.9557057057057057,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1LYZ",
      "rank": 4,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_unrelaxed_rank_004_alphafold2_ptm_model_2_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_004_alphafold2_ptm_model_2_seed_042.json",
      "residues": 129,
      "mean_plddt": 97.56542635658914,
      "ptm": 0.9,
      "pdb_sha256": "e37fe941b5c7ab3aae3df131011db586d1d635d706abe3ed061e629b8ac8da7a",
      "finite": true,
      "reference": "/work/data/1LYZ.pdb",
      "aligned_ca": 129,
      "predicted_ca": 129,
      "reference_ca": 129,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.5593399968825035,
      "ca_lddt": 0.950367033700367,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1LYZ",
      "rank": 5,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_unrelaxed_rank_005_alphafold2_ptm_model_1_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_005_alphafold2_ptm_model_1_seed_042.json",
      "residues": 129,
      "mean_plddt": 97.39062015503876,
      "ptm": 0.89,
      "pdb_sha256": "110db8830de0ca1365358901c6486250208f850274a66fc4d93b22465f43cd4a",
      "finite": true,
      "reference": "/work/data/1LYZ.pdb",
      "aligned_ca": 129,
      "predicted_ca": 129,
      "reference_ca": 129,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 0.5583791005192437,
      "ca_lddt": 0.9476976976976977,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1TIM",
      "rank": 1,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_unrelaxed_rank_001_alphafold2_ptm_model_5_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_001_alphafold2_ptm_model_5_seed_042.json",
      "residues": 247,
      "mean_plddt": 96.48862348178139,
      "ptm": 0.94,
      "pdb_sha256": "9043169c0530de18a4d0c139fd95365ee01fd0fb6aee919e39fbe13dc7af658b",
      "finite": true,
      "reference": "/work/data/1TIM.pdb",
      "aligned_ca": 247,
      "predicted_ca": 247,
      "reference_ca": 247,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 1.0435786951475245,
      "ca_lddt": 0.8845340582993505,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1TIM",
      "rank": 2,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_unrelaxed_rank_002_alphafold2_ptm_model_3_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "residues": 247,
      "mean_plddt": 96.45481781376519,
      "ptm": 0.94,
      "pdb_sha256": "406dee9a74662cf5e65f53d4d27bb546e66cb2466e3f1a7bbf595240122384e4",
      "finite": true,
      "reference": "/work/data/1TIM.pdb",
      "aligned_ca": 247,
      "predicted_ca": 247,
      "reference_ca": 247,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 1.0609301118904182,
      "ca_lddt": 0.8855913004077934,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1TIM",
      "rank": 3,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_unrelaxed_rank_003_alphafold2_ptm_model_4_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_003_alphafold2_ptm_model_4_seed_042.json",
      "residues": 247,
      "mean_plddt": 96.35720647773277,
      "ptm": 0.94,
      "pdb_sha256": "90cf1e42ab772e1a56ff60c24a6e49cd1e3cc92a7f19c0002c96dbbcee38e8ae",
      "finite": true,
      "reference": "/work/data/1TIM.pdb",
      "aligned_ca": 247,
      "predicted_ca": 247,
      "reference_ca": 247,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 1.0861333979019798,
      "ca_lddt": 0.8803428485123094,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1TIM",
      "rank": 4,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_unrelaxed_rank_004_alphafold2_ptm_model_1_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_004_alphafold2_ptm_model_1_seed_042.json",
      "residues": 247,
      "mean_plddt": 96.12174089068826,
      "ptm": 0.93,
      "pdb_sha256": "cc11391216828bafe56400c33b51d8d32427bc9a93dddd539a127abccfea499a",
      "finite": true,
      "reference": "/work/data/1TIM.pdb",
      "aligned_ca": 247,
      "predicted_ca": 247,
      "reference_ca": 247,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 1.1793652740975218,
      "ca_lddt": 0.8793233650505966,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    },
    {
      "query": "1TIM",
      "rank": 5,
      "pdb_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_unrelaxed_rank_005_alphafold2_ptm_model_2_seed_042.pdb",
      "scores_path": "/work/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_005_alphafold2_ptm_model_2_seed_042.json",
      "residues": 247,
      "mean_plddt": 95.61805668016193,
      "ptm": 0.93,
      "pdb_sha256": "3197c50584899c3876432fdb72fa60ee3dbf4f6077803c4c6e5dcb86af6fa6b3",
      "finite": true,
      "reference": "/work/data/1TIM.pdb",
      "aligned_ca": 247,
      "predicted_ca": 247,
      "reference_ca": 247,
      "sequence_identity": 1.0,
      "ca_rmsd_angstrom": 1.1613022138413016,
      "ca_lddt": 0.876491466545839,
      "definition": "Global sequence alignment; optimal Kabsch CA RMSD and CA-only lDDT, reference distance cutoff 15A. Not official all-atom lDDT."
    }
  ]
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-ca3c0f6d9da8

원래 경로: `colabfold/results/af2_20261027_172813_247681/summary.json` · SHA256: `1d72cd1c81341bd327f7e8ee4f3b4b9cdce6b62c4dbaf32ff7c8afbb5ac73f75`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172813_247681/summary.json", "sha256": "1d72cd1c81341bd327f7e8ee4f3b4b9cdce6b62c4dbaf32ff7c8afbb5ac73f75", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "success",
  "run": "/work/results/af2_20261027_172813_247681",
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
  "predictions": [
    {
      "query": "1UBQ",
      "tag": "alphafold2_ptm_model_1_seed_042",
      "lengths": [
        76
      ],
      "mean_plddt": 95.0,
      "ptm": 0.81689453125,
      "plddt_shape": [
        86
      ],
      "pae_shape": [
        86,
        86
      ],
      "coordinates_shape": [
        86,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1UBQ",
      "tag": "alphafold2_ptm_model_2_seed_042",
      "lengths": [
        76
      ],
      "mean_plddt": 95.1875,
      "ptm": 0.8271484375,
      "plddt_shape": [
        86
      ],
      "pae_shape": [
        86,
        86
      ],
      "coordinates_shape": [
        86,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1UBQ",
      "tag": "alphafold2_ptm_model_3_seed_042",
      "lengths": [
        76
      ],
      "mean_plddt": 95.4375,
      "ptm": 0.833984375,
      "plddt_shape": [
        86
      ],
      "pae_shape": [
        86,
        86
      ],
      "coordinates_shape": [
        86,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1UBQ",
      "tag": "alphafold2_ptm_model_4_seed_042",
      "lengths": [
        76
      ],
      "mean_plddt": 95.9375,
      "ptm": 0.84130859375,
      "plddt_shape": [
        86
      ],
      "pae_shape": [
        86,
        86
      ],
      "coordinates_shape": [
        86,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1UBQ",
      "tag": "alphafold2_ptm_model_5_seed_042",
      "lengths": [
        76
      ],
      "mean_plddt": 94.1875,
      "ptm": 0.8076171875,
      "plddt_shape": [
        86
      ],
      "pae_shape": [
        86,
        86
      ],
      "coordinates_shape": [
        86,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count1_A2RU14",
      "tag": "alphafold2_ptm_model_1_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 84.125,
      "ptm": 0.4697265625,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count1_A2RU14",
      "tag": "alphafold2_ptm_model_2_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 78.9375,
      "ptm": 0.44384765625,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count1_A2RU14",
      "tag": "alphafold2_ptm_model_3_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 84.5,
      "ptm": 0.450927734375,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count1_A2RU14",
      "tag": "alphafold2_ptm_model_4_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 82.0,
      "ptm": 0.4638671875,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count1_A2RU14",
      "tag": "alphafold2_ptm_model_5_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 86.0625,
      "ptm": 0.5205078125,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count2_A6NFY7",
      "tag": "alphafold2_ptm_model_1_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 71.3125,
      "ptm": 0.49755859375,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count2_A6NFY7",
      "tag": "alphafold2_ptm_model_2_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 70.3125,
      "ptm": 0.5224609375,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count2_A6NFY7",
      "tag": "alphafold2_ptm_model_3_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 72.375,
      "ptm": 0.5244140625,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count2_A6NFY7",
      "tag": "alphafold2_ptm_model_4_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 74.0,
      "ptm": 0.5263671875,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count2_A6NFY7",
      "tag": "alphafold2_ptm_model_5_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 71.75,
      "ptm": 0.529296875,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count3_O14519",
      "tag": "alphafold2_ptm_model_1_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 71.375,
      "ptm": 0.452392578125,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count3_O14519",
      "tag": "alphafold2_ptm_model_2_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 69.9375,
      "ptm": 0.4423828125,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count3_O14519",
      "tag": "alphafold2_ptm_model_3_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 70.3125,
      "ptm": 0.460205078125,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count3_O14519",
      "tag": "alphafold2_ptm_model_4_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 70.25,
      "ptm": 0.45556640625,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count3_O14519",
      "tag": "alphafold2_ptm_model_5_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 69.125,
      "ptm": 0.446044921875,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count4_P01270",
      "tag": "alphafold2_ptm_model_1_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 69.0625,
      "ptm": 0.32861328125,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count4_P01270",
      "tag": "alphafold2_ptm_model_2_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 63.8125,
      "ptm": 0.24755859375,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count4_P01270",
      "tag": "alphafold2_ptm_model_3_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 70.25,
      "ptm": 0.308349609375,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count4_P01270",
      "tag": "alphafold2_ptm_model_4_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 59.15625,
      "ptm": 0.2474365234375,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "count4_P01270",
      "tag": "alphafold2_ptm_model_5_seed_042",
      "lengths": [
        115
      ],
      "mean_plddt": 64.0625,
      "ptm": 0.258056640625,
      "plddt_shape": [
        125
      ],
      "pae_shape": [
        125,
        125
      ],
      "coordinates_shape": [
        125,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1LYZ",
      "tag": "alphafold2_ptm_model_1_seed_042",
      "lengths": [
        129
      ],
      "mean_plddt": 97.375,
      "ptm": 0.892578125,
      "plddt_shape": [
        139
      ],
      "pae_shape": [
        139,
        139
      ],
      "coordinates_shape": [
        139,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1LYZ",
      "tag": "alphafold2_ptm_model_2_seed_042",
      "lengths": [
        129
      ],
      "mean_plddt": 97.5625,
      "ptm": 0.900390625,
      "plddt_shape": [
        139
      ],
      "pae_shape": [
        139,
        139
      ],
      "coordinates_shape": [
        139,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1LYZ",
      "tag": "alphafold2_ptm_model_3_seed_042",
      "lengths": [
        129
      ],
      "mean_plddt": 97.8125,
      "ptm": 0.90185546875,
      "plddt_shape": [
        139
      ],
      "pae_shape": [
        139,
        139
      ],
      "coordinates_shape": [
        139,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1LYZ",
      "tag": "alphafold2_ptm_model_4_seed_042",
      "lengths": [
        129
      ],
      "mean_plddt": 97.875,
      "ptm": 0.904296875,
      "plddt_shape": [
        139
      ],
      "pae_shape": [
        139,
        139
      ],
      "coordinates_shape": [
        139,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1LYZ",
      "tag": "alphafold2_ptm_model_5_seed_042",
      "lengths": [
        129
      ],
      "mean_plddt": 98.1875,
      "ptm": 0.912109375,
      "plddt_shape": [
        139
      ],
      "pae_shape": [
        139,
        139
      ],
      "coordinates_shape": [
        139,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1TIM",
      "tag": "alphafold2_ptm_model_1_seed_042",
      "lengths": [
        247
      ],
      "mean_plddt": 96.125,
      "ptm": 0.9306640625,
      "plddt_shape": [
        247
      ],
      "pae_shape": [
        247,
        247
      ],
      "coordinates_shape": [
        247,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1TIM",
      "tag": "alphafold2_ptm_model_2_seed_042",
      "lengths": [
        247
      ],
      "mean_plddt": 95.625,
      "ptm": 0.93359375,
      "plddt_shape": [
        247
      ],
      "pae_shape": [
        247,
        247
      ],
      "coordinates_shape": [
        247,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1TIM",
      "tag": "alphafold2_ptm_model_3_seed_042",
      "lengths": [
        247
      ],
      "mean_plddt": 96.4375,
      "ptm": 0.9384765625,
      "plddt_shape": [
        247
      ],
      "pae_shape": [
        247,
        247
      ],
      "coordinates_shape": [
        247,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1TIM",
      "tag": "alphafold2_ptm_model_4_seed_042",
      "lengths": [
        247
      ],
      "mean_plddt": 96.375,
      "ptm": 0.9375,
      "plddt_shape": [
        247
      ],
      "pae_shape": [
        247,
        247
      ],
      "coordinates_shape": [
        247,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    },
    {
      "query": "1TIM",
      "tag": "alphafold2_ptm_model_5_seed_042",
      "lengths": [
        247
      ],
      "mean_plddt": 96.5,
      "ptm": 0.9384765625,
      "plddt_shape": [
        247
      ],
      "pae_shape": [
        247,
        247
      ],
      "coordinates_shape": [
        247,
        37,
        3
      ],
      "finite_coordinates": true,
      "finite_plddt": true
    }
  ],
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
  "model_loading_seconds": 2.395940059000168,
  "pipeline_wall_seconds": 454.26669272099934,
  "gpu_apply_seconds": 402.41356543600614,
  "compilation_seconds": 36.85190285000044,
  "final_device_memory": {
    "num_allocs": 18764,
    "bytes_in_use": 374628608,
    "peak_bytes_in_use": 2905266432,
    "largest_alloc_size": 857794560,
    "bytes_limit": 522645929984,
    "bytes_reserved": 0,
    "peak_bytes_reserved": 0,
    "largest_free_block_bytes": 0,
    "pool_bytes": 4292870144,
    "peak_pool_bytes": 4292870144
  },
  "wall_seconds": 454.9514712900009
}
```
<!-- END_RESTORE_FILE -->


</details>
