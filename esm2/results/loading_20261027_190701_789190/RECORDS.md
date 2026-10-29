# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-6fadfb0c7605

원래 경로: `esm2/results/loading_20261027_190701_789190/loading.json` · SHA256: `1955cb20caaed6caf18c534d772456bb88fc0caa9dd47893cbca78ba8ffbee3d`

<details>
<summary>loading.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "esm2/results/loading_20261027_190701_789190/loading.json", "sha256": "1955cb20caaed6caf18c534d772456bb88fc0caa9dd47893cbca78ba8ffbee3d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "checkpoint": "esm2_t33_650M_UR50D",
  "total_parameters": 651043254,
  "parameter_dtype": "torch.float32",
  "torch": "2.14.1+cu130",
  "cuda": "13.0",
  "host": "spark-e4b6",
  "load_seconds": 2.3110589169991727,
  "cuda_after_load_allocated_gib": 2.489795207977295,
  "cuda_after_load_reserved_gib": 2.55859375,
  "cuda_loading_peak_allocated_gib": 2.489795207977295,
  "rss_before_load_gib": 0.4947967529296875,
  "rss_after_load_gib": 2.087055206298828,
  "system_available_gib": 109.84812545776367,
  "scope": "Fresh process, CUDA initialized before load; original pretrained weights only; no tokenization, forward or optimizer"
}
```
<!-- END_RESTORE_FILE -->


</details>
