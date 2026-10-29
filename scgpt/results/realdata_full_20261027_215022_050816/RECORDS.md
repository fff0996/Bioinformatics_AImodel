# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-5533080949cc

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/memory_guard.json` · SHA256: `9c0d9aba43c517d293f80d1b0be639f87a0177cc707b132bc441cdd198e9a076`

<details>
<summary>memory_guard.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/memory_guard.json", "sha256": "9c0d9aba43c517d293f80d1b0be639f87a0177cc707b132bc441cdd198e9a076", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "time_unix": 1793105948.6204712,
  "available_gib": 7.705135345458984,
  "swap_gib": 0.000698089599609375,
  "cuda_allocated_gib": 53.944210052490234,
  "cuda_reserved_gib": 103.806640625
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-814883bdc5be

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/protocol.json` · SHA256: `db84f91926add35797c51721ae17f861c1240659b5f6e2b1865ba5d37b597dfb`

<details>
<summary>protocol.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/protocol.json", "sha256": "db84f91926add35797c51721ae17f861c1240659b5f6e2b1865ba5d37b597dfb", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "dataset": "existing actual TOSICA pancreas demo_train10600/demo_test4218",
  "split": "training8480/validation2120/test4218, same seed42",
  "batch_cases": [
    8,
    284
  ],
  "epochs": 1,
  "max_length": 1200,
  "padding": "original official collator dynamic max-in-batch, no forced1200 padding",
  "precision": "FP32, TF32False",
  "optimizer": "same original AdamW/lr1e-4/wd.01 clip1.0",
  "initialization": "same fresh pretrained encoder + seed42 classifier per case",
  "test_usage": "separate4218 evaluated from saved model; no model selection on test",
  "warning": "different batch sizes imply different optimizer step counts, accuracy is not a batch-only causal effect"
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-6ca19a7cee88

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/source_data_audit.json` · SHA256: `c0ab0e3db0c31e9196e75b27e1dfe74f96e3135cf4923c740aa821fa445de3ef`

<details>
<summary>source_data_audit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/source_data_audit.json", "sha256": "c0ab0e3db0c31e9196e75b27e1dfe74f96e3135cf4923c740aa821fa445de3ef", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "train_file_shape": [
    10600,
    3000
  ],
  "test_file_shape": [
    4218,
    3000
  ],
  "train_obs_id_unique": true,
  "test_obs_id_unique": true,
  "shared_cell_ids_between_files": 0,
  "train_obs_columns": [
    "Celltype"
  ],
  "test_obs_columns": [
    "Celltype"
  ],
  "train_label_counts": {
    "alpha": 3136,
    "beta": 2966,
    "ductal": 1290,
    "acinar": 1144,
    "delta": 793,
    "PSC": 524,
    "PP": 356,
    "endothelial": 273,
    "macrophage": 52,
    "mast": 25,
    "epsilon": 21,
    "schwann": 13,
    "t_cell": 7
  },
  "test_label_counts": {
    "alpha": 2011,
    "beta": 1006,
    "ductal": 414,
    "PP": 282,
    "acinar": 209,
    "delta": 188,
    "PSC": 73,
    "endothelial": 16,
    "epsilon": 7,
    "mast": 7,
    "MHC class II": 5
  }
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-def4ed1b1494

원래 경로: `scgpt/results/realdata_full_20261027_215022_050816/summary.json` · SHA256: `9cf5227045994fdcfbfc9ef840f235f780b89cb5894487583ab144b0c3758a1f`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/realdata_full_20261027_215022_050816/summary.json", "sha256": "9cf5227045994fdcfbfc9ef840f235f780b89cb5894487583ab144b0c3758a1f", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "partial",
  "completed_batch8": {
    "train": {
      "status": "complete",
      "task": "scGPT supervised cell-type fine-tuning",
      "training_cells": 8480,
      "validation_cells": 2120,
      "classes": 13,
      "matched_genes": 2999,
      "epochs": 1,
      "batch_size": 8,
      "max_length_including_cls": 1200,
      "precision": "FP32",
      "seed": 42,
      "parameter_count_constructed_model": 51338254,
      "parameter_count_trainable": 50812429,
      "parameter_count_encoder": 50278400,
      "input_preparation_seconds": 0.3290299939981196,
      "model_loading_seconds": 0.999307805999706,
      "validation_tokenization_seconds": 0.2001921030023368,
      "training_seconds_total": 202.88499515999865,
      "training_gpu_peak_allocated_gib": 3.374143600463867,
      "training_gpu_peak_reserved_gib": 4.693359375,
      "training_process_rss_peak_observed_gib": 2.77008056640625,
      "best_checkpoint": {
        "epoch": 1,
        "validation_macro_f1": 0.6764915422035108,
        "model_bundle": "/workspace/scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001"
      },
      "validation": {
        "cells_total": 2120,
        "cells_evaluated_known_labels": 2120,
        "cells_without_known_training_label": 0,
        "forward_seconds": 3.8487031789991306,
        "cells_per_second": 550.8348920145393,
        "timing_scope": "cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving",
        "gpu_util_mean_percent": 74.26315789473684,
        "gpu_util_max_percent": 94.0,
        "gpu_util_samples": 19,
        "gpu_peak_allocated_gib": 0.7571239471435547,
        "gpu_peak_reserved_gib": 4.693359375,
        "python_process_rss_peak_observed_gib": 2.3290977478027344,
        "loss": 0.3006041944026947,
        "accuracy": 0.9320754716981132,
        "macro_f1_all_training_classes": 0.6764915422035108,
        "macro_f1_present_true_classes": 0.6764915422035108,
        "weighted_f1": 0.9283806673159697
      },
      "weights_updated": true,
      "encoder_and_classifier_updated": true,
      "saved_model_reload_verified": true,
      "test_evaluated": false,
      "epochs_detail": [
        {
          "epoch": 1,
          "training_cells": 8480,
          "training_seconds": 202.88499515999865,
          "training_cells_per_second": 41.79707815904534,
          "online_training_loss": 0.32035121583014087,
          "online_training_accuracy": 0.9320754716981132,
          "gpu_util_mean_percent": 93.0183908045977,
          "gpu_util_max_percent": 95.0,
          "gpu_util_samples": 870,
          "gpu_peak_allocated_gib": 3.374143600463867,
          "gpu_peak_reserved_gib": 4.693359375,
          "python_process_rss_peak_observed_gib": 2.77008056640625,
          "validation": {
            "cells_total": 2120,
            "cells_evaluated_known_labels": 2120,
            "cells_without_known_training_label": 0,
            "forward_seconds": 3.8487031789991306,
            "cells_per_second": 550.8348920145393,
            "timing_scope": "cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving",
            "gpu_util_mean_percent": 74.26315789473684,
            "gpu_util_max_percent": 94.0,
            "gpu_util_samples": 19,
            "gpu_peak_allocated_gib": 0.7571239471435547,
            "gpu_peak_reserved_gib": 4.693359375,
            "python_process_rss_peak_observed_gib": 2.3290977478027344,
            "loss": 0.3006041944026947,
            "accuracy": 0.9320754716981132,
            "macro_f1_all_training_classes": 0.6764915422035108,
            "macro_f1_present_true_classes": 0.6764915422035108,
            "weighted_f1": 0.9283806673159697
          },
          "weight_probe_max_abs_deltas": {
            "transformer_encoder.layers.0.self_attn.in_proj_weight": 0.01844094693660736,
            "cls_decoder.out_layer.weight": 0.017070859670639038
          },
          "model_bundle": "/workspace/scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001",
          "checkpoint_save_seconds": 2.2351961559979827,
          "strict_reload_passed": true,
          "reload_max_abs_logit_error": 0.0
        }
      ],
      "timing_scope": "tokenization/binning/sampling/padding, transfers, encoder/classifier, cross-entropy, backward, gradient clipping, AdamW, in-memory input/metric recording; includes first-step initialization",
      "memory_notes": "CUDA peaks are process-local PyTorch allocator counters, not total device usage. RSS is sampled every 0.2 s, not an exact peak. Spark uses shared memory: do not add RSS and CUDA as independent physical memory pools.",
      "interpretation": "Small-run fine-tuning and validation check; not a full-data or donor-independent accuracy benchmark"
    },
    "test": {
      "status": "complete",
      "task": "fine-tuned cell-type predictions",
      "tokenization_seconds": 0.4637714349992166,
      "warmup_steps": 0,
      "weights_updated": false,
      "model_bundle": "/workspace/scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001",
      "memory_notes": "CUDA peaks are process-local PyTorch allocator counters, not total device usage. RSS is sampled every 0.2 s, not an exact peak. Spark uses shared memory: do not add RSS and CUDA as independent physical memory pools.",
      "cells_total": 4218,
      "cells_evaluated_known_labels": 4213,
      "cells_without_known_training_label": 5,
      "forward_seconds": 16.655858064001222,
      "cells_per_second": 253.24423297749414,
      "timing_scope": "cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving",
      "gpu_util_mean_percent": 87.98684210526316,
      "gpu_util_max_percent": 95.0,
      "gpu_util_samples": 76,
      "gpu_peak_allocated_gib": 0.3811001777648926,
      "gpu_peak_reserved_gib": 0.6171875,
      "python_process_rss_peak_observed_gib": 1.7402877807617188,
      "loss": 0.34234651923179626,
      "accuracy": 0.9221457393781154,
      "macro_f1_all_training_classes": 0.5296341992072012,
      "macro_f1_present_true_classes": 0.6885244589693615,
      "weighted_f1": 0.9165802290481048
    }
  },
  "larger_batches": {
    "284": {
      "status": "memory_guard_stopped",
      "guard": {
        "time_unix": 1793105948.6204712,
        "available_gib": 7.705135345458984,
        "swap_gib": 0.000698089599609375,
        "cuda_allocated_gib": 53.944210052490234,
        "cuda_reserved_gib": 103.806640625
      }
    },
    "256": {
      "status": "memory_guard_stopped",
      "guard": {
        "time_unix": 1793106687.1003087,
        "available_gib": 7.816379547119141,
        "swap_gib": 0.00069427490234375,
        "cuda_allocated_gib": 59.276695728302,
        "cuda_reserved_gib": 103.662109375
      }
    },
    "224": {
      "status": "memory_guard_stopped",
      "guard": {
        "time_unix": 1793106811.8309977,
        "available_gib": 4.424278259277344,
        "swap_gib": 0.003997802734375,
        "cuda_allocated_gib": 69.09493350982666,
        "cuda_reserved_gib": 106.6015625
      }
    }
  },
  "note": "Large batch training comparisons stopped for shared-memory safety. Prioritized separate300k real-cell embedding inference."
}
```
<!-- END_RESTORE_FILE -->


</details>
