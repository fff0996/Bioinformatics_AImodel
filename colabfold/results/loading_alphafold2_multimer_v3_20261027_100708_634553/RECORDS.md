# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-d0e920bdfc7a

원래 경로: `colabfold/results/loading_alphafold2_multimer_v3_20261027_100708_634553/loading.json` · SHA256: `7b0ce7e0be7caa2d56d10a4aabe01bb7069067b78e079f084ff374e0caa35cf2`

<details>
<summary>loading.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/loading_alphafold2_multimer_v3_20261027_100708_634553/loading.json", "sha256": "7b0ce7e0be7caa2d56d10a4aabe01bb7069067b78e079f084ff374e0caa35cf2", "encoding": "utf-8", "added_final_newline": true} -->
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
  "load_seconds": 2.2020127559990215,
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
    "num_allocs": 3960,
    "bytes_in_use": 2338604800,
    "peak_bytes_in_use": 2339190528,
    "largest_alloc_size": 98713600,
    "bytes_limit": 97997815808,
    "bytes_reserved": 0,
    "peak_bytes_reserved": 0,
    "largest_free_block_bytes": 0,
    "pool_bytes": 4297064448,
    "peak_pool_bytes": 4297064448
  },
  "after_load_allocated_gib": 2.177995443344116,
  "loading_peak_gib": 2.1785409450531006,
  "rss_before_load_gib": 0.7161979675292969,
  "rss_after_load_gib": 1.1208572387695312,
  "scope": "Fresh process and same image; five official weight sets loaded, no input features, forward, compile or optimizer; loader may keep additional runner references"
}
```
<!-- END_RESTORE_FILE -->


</details>
