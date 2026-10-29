# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-5e06ebb979b6

원래 경로: `colabfold/results/loading_alphafold2_ptm_20261027_100705_136907/loading.json` · SHA256: `a8c1ba3a5e1cf47e845f703bc6ffc1027a0ebf2ad9e7c420d3aadda1b2310dd3`

<details>
<summary>loading.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/loading_alphafold2_ptm_20261027_100705_136907/loading.json", "sha256": "a8c1ba3a5e1cf47e845f703bc6ffc1027a0ebf2ad9e7c420d3aadda1b2310dd3", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "model_type": "alphafold2_ptm",
  "weight_sets": 5,
  "parameters_per_model": [
    {
      "name": "model_1",
      "parameters": 92907930
    },
    {
      "name": "model_2",
      "parameters": 92907930
    },
    {
      "name": "model_3",
      "parameters": 92907930
    },
    {
      "name": "model_4",
      "parameters": 92907930
    },
    {
      "name": "model_5",
      "parameters": 92907930
    }
  ],
  "parameter_dtype": "float32",
  "compute_configuration": "bfloat16",
  "load_seconds": 2.0962006720001227,
  "device_memory_before": {
    "num_allocs": 0,
    "bytes_in_use": 0,
    "peak_bytes_in_use": 0,
    "largest_alloc_size": 0,
    "bytes_limit": 97997815808,
    "bytes_reserved": 0,
    "peak_bytes_reserved": 0,
    "largest_free_block_bytes": 0,
    "pool_bytes": 0,
    "peak_pool_bytes": 0
  },
  "device_memory_after": {
    "num_allocs": 3600,
    "bytes_in_use": 2316957696,
    "peak_bytes_in_use": 2327255296,
    "largest_alloc_size": 99614720,
    "bytes_limit": 97997815808,
    "bytes_reserved": 0,
    "peak_bytes_reserved": 0,
    "largest_free_block_bytes": 0,
    "pool_bytes": 4292870144,
    "peak_pool_bytes": 4292870144
  },
  "after_load_allocated_gib": 2.157835006713867,
  "loading_peak_gib": 2.1674253940582275,
  "rss_before_load_gib": 0.7162284851074219,
  "rss_after_load_gib": 1.1455917358398438,
  "scope": "Fresh process and same image; five official weight sets loaded, no input features, forward, compile or optimizer; loader may keep additional runner references"
}
```
<!-- END_RESTORE_FILE -->


</details>
