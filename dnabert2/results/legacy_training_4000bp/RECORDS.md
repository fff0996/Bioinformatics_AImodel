# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-dd75133f627c

원래 경로: `dnabert2/results/legacy_training_4000bp/provenance.json` · SHA256: `974292ae76b0c97fe445fb5e77b89fdff842945684acd008f585bc99af8509bb`

<details>
<summary>provenance.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/legacy_training_4000bp/provenance.json", "sha256": "974292ae76b0c97fe445fb5e77b89fdff842945684acd008f585bc99af8509bb", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "source_path": "/home/hyein/dnabert2-benchmark/dnabert2_eval/bend_length_results/20261011_151930_training_n1000_b8_bp4000_fp32_1791731970455157330/summary.json",
  "note": "Existing completed BEND n1000 batch8 FP32 4000bp training reference; copied summary only, no new experiment and no model weights copied"
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-24532bfc6cab

원래 경로: `dnabert2/results/legacy_training_4000bp/summary.json` · SHA256: `9b0ad1c82bae00270401e963ab57e3fd4eb7af36c08d6e0b17218d20563484cc`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "dnabert2/results/legacy_training_4000bp/summary.json", "sha256": "9b0ad1c82bae00270401e963ab57e3fd4eb7af36c08d6e0b17218d20563484cc", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "settings": {
    "dataset": "BEND histone_modification",
    "mode": "training",
    "n": 1000,
    "batch": 8,
    "bp": 4000,
    "precision": "fp32",
    "tf32": false,
    "epochs": 1,
    "learning_rate": 2e-05,
    "seed": 42,
    "num_labels": 18,
    "pooling": "masked_mean_including_special_tokens"
  },
  "gpu": "NVIDIA GB10",
  "parameters": 117082386,
  "data_file": "/workspace/dnabert2_eval/external_data/BEND/prepared/20261012_001503_n1000_seed42/train_n1000_bp4000.csv",
  "sample_ids_sha256": "258e72eef8b16d69f22b84e1d5b02154557b6b9fd3853c879c3d5f61df5ba51c",
  "timing_scope": "tokenization + transfer + forward + backward + optimizer",
  "inference_warmup_batches": 0,
  "ram_sampling_interval_seconds": 0.2,
  "shared_memory_denominator_gib": 121,
  "status": "success",
  "seconds": 116.86838827491738,
  "sequences_per_second": 8.556633789178582,
  "tokens_min": 697,
  "tokens_mean": 778.855,
  "tokens_max": 851,
  "mean_training_loss": 0.471063063621521,
  "processed_sequences": 1000,
  "gpu_peak_allocated_gib": 10.571216583251953,
  "gpu_allocated_percent_of_shared_memory": 8.736542630786738,
  "ram_peak_sampled_gib": 1.8029937744140625
}
```
<!-- END_RESTORE_FILE -->


</details>
