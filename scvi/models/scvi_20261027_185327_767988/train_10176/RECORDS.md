# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-559b317ec60c

원래 경로: `scvi/models/scvi_20261027_185327_767988/train_10176/benchmark_metadata.json` · SHA256: `6edc30ad47c7371e1254e98699d454e1bad4ae5c3f36069fd003d71f5be083ad`

<details>
<summary>benchmark_metadata.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scvi/models/scvi_20261027_185327_767988/train_10176/benchmark_metadata.json", "sha256": "6edc30ad47c7371e1254e98699d454e1bad4ae5c3f36069fd003d71f5be083ad", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "condition": "train_10176",
  "unique_training_cells": 10176,
  "processed_training_cells": 4070400,
  "validation_cells": 1272,
  "test_cells": 1273,
  "epochs": 400,
  "optimizer_steps": 32000,
  "genes": 2000,
  "latent_dim": 10,
  "total_parameters": 1293526,
  "configured_trainable_parameters": 1293526,
  "encoder_parameters": 258964,
  "observed_gradient_parameters": 778884,
  "updated_parameter_tensors": 17,
  "initialization_sha256": "a1a88a8f3b1b835b70a4716a83fe2013d646d8c0ef2bfff34afc5165b4bf4926",
  "preprocess_seconds": 0.01748456499990425,
  "initialization_seconds": 0.009538583999528782,
  "training_loop_seconds": 159.0651606000174,
  "validation_loop_seconds": 0.4153169350029202,
  "fit_wall_seconds": 159.90338336199966,
  "evaluation_seconds": 0.031916108999212156,
  "save_seconds": 0.06125996800074063,
  "condition_wall_seconds": 160.06057625499852,
  "training_cells_per_second": 25589.513031300172,
  "test_elbo": -536.2740478515625,
  "validation_elbo": -554.1507568359375,
  "elbo_definition": "scVI 1.5.1 get_elbo returns ELBO; higher is better",
  "cuda_peak_allocated_gib": 0.09743928909301758,
  "cuda_peak_reserved_gib": 0.099609375,
  "fit_profile": {
    "gpu_util_mean_percent": 25.715625,
    "gpu_util_max_percent": 33.0,
    "sample_count": 320,
    "rss_sampled_peak_gib": 2.060474395751953,
    "available_min_gib": 112.38762283325195,
    "swap_max_bytes": 0
  },
  "training_phase_profile": {
    "gpu_util_mean_percent": 25.7626582278481,
    "gpu_util_max_percent": 33.0,
    "sample_count": 316,
    "rss_sampled_peak_gib": 2.0569610595703125,
    "available_min_gib": 112.38762283325195,
    "swap_max_bytes": 0
  },
  "validation_phase_profile": {
    "gpu_util_mean_percent": 24.0,
    "gpu_util_max_percent": 24.0,
    "sample_count": 2,
    "rss_sampled_peak_gib": 2.0569419860839844,
    "available_min_gib": 112.57820129394531,
    "swap_max_bytes": 0
  },
  "model_path": "/work/models/scvi_20261027_185327_767988/train_10176",
  "training_success": true,
  "model_saved": true,
  "reload_verified": false,
  "training_resume_verified": false
}
```
<!-- END_RESTORE_FILE -->


</details>
