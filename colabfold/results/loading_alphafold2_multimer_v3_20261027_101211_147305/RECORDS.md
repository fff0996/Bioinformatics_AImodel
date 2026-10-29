# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-17f7637f2e9c

원래 경로: `colabfold/results/loading_alphafold2_multimer_v3_20261027_101211_147305/loading.json` · SHA256: `629dc1cde95b914ca9de61173755e546906d3b933b7ff35ed8d04a2172b9da3a`

<details>
<summary>loading.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/loading_alphafold2_multimer_v3_20261027_101211_147305/loading.json", "sha256": "629dc1cde95b914ca9de61173755e546906d3b933b7ff35ed8d04a2172b9da3a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "model_type": "alphafold2_multimer_v3",
  "weight_sets": 5,
  "parameters_per_model": [
    {
      "name": "model_1",
      "parameters": 93223705
    },
    {
      "name": "model_2",
      "parameters": 93223705
    },
    {
      "name": "model_3",
      "parameters": 93223705
    },
    {
      "name": "model_4",
      "parameters": 93223705
    },
    {
      "name": "model_5",
      "parameters": 93223705
    }
  ],
  "parameter_dtype": "float32",
  "compute_configuration": "bfloat16",
  "load_seconds": 2.3331397519996244,
  "device_memory_before": {
    "num_allocs": 0,
    "bytes_in_use": 0,
    "peak_bytes_in_use": 0,
    "largest_alloc_size": 0,
    "bytes_limit": 522645929984,
    "bytes_reserved": 0,
    "peak_bytes_reserved": 0,
    "largest_free_block_bytes": 0,
    "pool_bytes": 0,
    "peak_pool_bytes": 0
  },
  "device_memory_after": {
    "num_allocs": 3960,
    "bytes_in_use": 2338916608,
    "peak_bytes_in_use": 2381549824,
    "largest_alloc_size": 60407808,
    "bytes_limit": 522645929984,
    "bytes_reserved": 0,
    "peak_bytes_reserved": 0,
    "largest_free_block_bytes": 0,
    "pool_bytes": 4297064448,
    "peak_pool_bytes": 4297064448
  },
  "after_load_allocated_gib": 2.178285837173462,
  "before_first_forward_ready_allocated_gib": 1.800525426864624,
  "device_memory_ready": {
    "num_allocs": 3960,
    "bytes_in_use": 1933299456,
    "peak_bytes_in_use": 2381549824,
    "largest_alloc_size": 60407808,
    "bytes_limit": 522645929984,
    "bytes_reserved": 0,
    "peak_bytes_reserved": 0,
    "largest_free_block_bytes": 0,
    "pool_bytes": 4297064448,
    "peak_pool_bytes": 4297064448
  },
  "environment": {
    "TF_FORCE_UNIFIED_MEMORY": "1",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0"
  },
  "loading_peak_gib": 2.2179911136627197,
  "rss_before_load_gib": 0.7538948059082031,
  "rss_after_load_gib": 3.37939453125,
  "scope": "Fresh process and same image; five official weight sets loaded, no input features, forward, compile or optimizer; loader snapshot may keep an extra initial runner parameter reference; ready snapshot selects model_1 as normal batch.run does, then releases unused references"
}
```
<!-- END_RESTORE_FILE -->


</details>
