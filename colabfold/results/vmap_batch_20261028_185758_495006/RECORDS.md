# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-4bb8424a9b6d

원래 경로: `colabfold/results/vmap_batch_20261028_185758_495006/batch1_command.json` · SHA256: `57590c787aadfcd194e74c93df19506c8b2c8e4d5f35af183c47d468faeb9d2f`

<details>
<summary>batch1_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/batch1_command.json", "sha256": "57590c787aadfcd194e74c93df19506c8b2c8e4d5f35af183c47d468faeb9d2f", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "af2-vmap-vmap_batch_20261028_185758_495006-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "RUN_ID=vmap_batch_20261028_185758_495006",
  "-e",
  "BATCH=1",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_batch_20261028_185758_495006/mplcache",
  "-v",
  "/home/hyein/ai-benchmark/colabfold:/work",
  "-v",
  "/home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro",
  "ghcr.io/sokrypton/colabfold:1.6.2-cuda13",
  "python",
  "-u",
  "/work/scripts/batch_followup/worker.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-802193f95a4b

원래 경로: `colabfold/results/vmap_batch_20261028_185758_495006/batch1_exit.json` · SHA256: `bdec275bbb902bac5c6478f31945aeff3e79af5d0aff5ee641a53761ac892eef`

<details>
<summary>batch1_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/batch1_exit.json", "sha256": "bdec275bbb902bac5c6478f31945aeff3e79af5d0aff5ee641a53761ac892eef", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 1, "wall_seconds": 6.5079662799835205, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-cb0e0c6f0f0e

원래 경로: `colabfold/results/vmap_batch_20261028_185758_495006/capture_command.json` · SHA256: `16fefa7b2e2667dc7cc32e01c60a60d2e01f742eff1c54df62a859b7622cca37`

<details>
<summary>capture_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/capture_command.json", "sha256": "16fefa7b2e2667dc7cc32e01c60a60d2e01f742eff1c54df62a859b7622cca37", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "JAX_PLATFORMS=cpu",
  "-e",
  "RUN_ID=vmap_batch_20261028_185758_495006",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_batch_20261028_185758_495006/mplcache",
  "-v",
  "/home/hyein/ai-benchmark/colabfold:/work",
  "-v",
  "/home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro",
  "ghcr.io/sokrypton/colabfold:1.6.2-cuda13",
  "python",
  "-u",
  "/work/scripts/batch_followup/capture.py"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-d727bc711ffa

원래 경로: `colabfold/results/vmap_batch_20261028_185758_495006/capture_exit.json` · SHA256: `124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01`

<details>
<summary>capture_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/capture_exit.json", "sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0}
```
<!-- END_RESTORE_FILE -->


</details>

## file-038429aff83f

원래 경로: `colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json` · SHA256: `34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d`

<details>
<summary>input_audit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json", "sha256": "34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "queries": [
    {
      "id": "count1_A2RU14",
      "sequence": "MAGTVLGVGAGVFILALLWVAVLLLCVLLSRASGAARFSVIFLFFGAVIITSVLLLFPRAGEFPAPEVEVKIVDDFFIGRYVLLAFLSAIFLGGLFLVLIHYVLEPIYAKPLHSY",
      "length": 115,
      "msa_depth_before_feature_sampling": 568,
      "msa_sha256": "65b27286d30511570201072127c6c4bf4deb6d09d4b4f1fbc948e7d7faaaf5ea"
    },
    {
      "id": "count2_A6NFY7",
      "sequence": "MSRHSRLQRQVLSLYRDLLRAGRGKPGAEARVRAEFRQHAGLPRSDVLRIEYLYRRGRRQLQLLRSGHATAMGAFVRPRAPTGEPGGVGCQPDDGDSPRNPHDSTGAPETRPDGR",
      "length": 115,
      "msa_depth_before_feature_sampling": 2382,
      "msa_sha256": "50133c0bebf212ec0461ad8d17aa6b90256b188aec35e7b01738c23761643560"
    },
    {
      "id": "count3_O14519",
      "sequence": "MSYKPNLAAHMPAAALNAAGSVHSPSTSMATSSQYRQLLSDYGPPSLGYTQGTGNSQVPQSKYAELLAIIEELGKEIRPTYAGSKSAMERLKRGIIHARGLVRECLAETERNARS",
      "length": 115,
      "msa_depth_before_feature_sampling": 1295,
      "msa_sha256": "eda7682338958cb5814920cb653b6c7b32d974298f72fada8d1100f371bfc76d"
    },
    {
      "id": "count4_P01270",
      "sequence": "MIPAKDMAKVMIVMLAICFLTKSDGKSVKKRSVSEIQLMHNLGKHLNSMERVEWLRKKLQDVHNFVALGAPLAPRDAGSQRPRKKEDNVLVESHEKSLGEADKADVNVLTKAKSQ",
      "length": 115,
      "msa_depth_before_feature_sampling": 971,
      "msa_sha256": "b08538aaaf14fecf190c2233a0e9adc1e887b79cef16364fc0c0134d14af6c53"
    }
  ],
  "unique_queries": 4,
  "features": "official ColabFold processed features captured before predict, JAX_PLATFORMS=cpu during capture",
  "preset": "alphafold2_ptm model_1, actual runner shared model3 architecture without templates",
  "num_recycles": 3,
  "num_ensemble": 1,
  "max_seq": 512,
  "max_extra_seq": 5120,
  "templates": false,
  "relaxation": false,
  "seed": 42
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-72d2cd0f5cdf

원래 경로: `colabfold/results/vmap_batch_20261028_185758_495006/runner_config.json` · SHA256: `707eb227ce413194fb0bd2a7a6f44bbb0199b5ca4fb7ac8a2a0c8cc4becbdf7a`

<details>
<summary>runner_config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/runner_config.json", "sha256": "707eb227ce413194fb0bd2a7a6f44bbb0199b5ca4fb7ac8a2a0c8cc4becbdf7a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "data": {
    "common": {
      "masked_msa": {
        "profile_prob": 0.1,
        "same_prob": 0.1,
        "uniform_prob": 0.1
      },
      "max_extra_msa": 458,
      "msa_cluster_features": true,
      "num_recycle": 3,
      "reduce_msa_clusters_by_max_templates": false,
      "resample_msa_in_recycling": true,
      "template_features": [
        "template_all_atom_positions",
        "template_sum_probs",
        "template_aatype",
        "template_all_atom_masks",
        "template_domain_names"
      ],
      "unsupervised_features": [
        "aatype",
        "residue_index",
        "sequence",
        "msa",
        "domain_name",
        "num_alignments",
        "seq_length",
        "between_segment_residues",
        "deletion_matrix"
      ],
      "use_templates": false
    },
    "eval": {
      "feat": {
        "aatype": [
          "num residues placeholder"
        ],
        "all_atom_mask": [
          "num residues placeholder",
          null
        ],
        "all_atom_positions": [
          "num residues placeholder",
          null,
          null
        ],
        "alt_chi_angles": [
          "num residues placeholder",
          null
        ],
        "asym_id": [
          "num residues placeholder"
        ],
        "atom14_alt_gt_exists": [
          "num residues placeholder",
          null
        ],
        "atom14_alt_gt_positions": [
          "num residues placeholder",
          null,
          null
        ],
        "atom14_atom_exists": [
          "num residues placeholder",
          null
        ],
        "atom14_atom_is_ambiguous": [
          "num residues placeholder",
          null
        ],
        "atom14_gt_exists": [
          "num residues placeholder",
          null
        ],
        "atom14_gt_positions": [
          "num residues placeholder",
          null,
          null
        ],
        "atom37_atom_exists": [
          "num residues placeholder",
          null
        ],
        "backbone_affine_mask": [
          "num residues placeholder"
        ],
        "backbone_affine_tensor": [
          "num residues placeholder",
          null
        ],
        "bert_mask": [
          "msa placeholder",
          "num residues placeholder"
        ],
        "chi_angles": [
          "num residues placeholder",
          null
        ],
        "chi_mask": [
          "num residues placeholder",
          null
        ],
        "extra_deletion_value": [
          "extra msa placeholder",
          "num residues placeholder"
        ],
        "extra_has_deletion": [
          "extra msa placeholder",
          "num residues placeholder"
        ],
        "extra_msa": [
          "extra msa placeholder",
          "num residues placeholder"
        ],
        "extra_msa_mask": [
          "extra msa placeholder",
          "num residues placeholder"
        ],
        "extra_msa_row_mask": [
          "extra msa placeholder"
        ],
        "is_distillation": [],
        "msa_feat": [
          "msa placeholder",
          "num residues placeholder",
          null
        ],
        "msa_mask": [
          "msa placeholder",
          "num residues placeholder"
        ],
        "msa_row_mask": [
          "msa placeholder"
        ],
        "pseudo_beta": [
          "num residues placeholder",
          null
        ],
        "pseudo_beta_mask": [
          "num residues placeholder"
        ],
        "random_crop_to_size_seed": [
          null
        ],
        "residue_index": [
          "num residues placeholder"
        ],
        "residx_atom14_to_atom37": [
          "num residues placeholder",
          null
        ],
        "residx_atom37_to_atom14": [
          "num residues placeholder",
          null
        ],
        "resolution": [],
        "rigidgroups_alt_gt_frames": [
          "num residues placeholder",
          null,
          null
        ],
        "rigidgroups_group_exists": [
          "num residues placeholder",
          null
        ],
        "rigidgroups_group_is_ambiguous": [
          "num residues placeholder",
          null
        ],
        "rigidgroups_gt_exists": [
          "num residues placeholder",
          null
        ],
        "rigidgroups_gt_frames": [
          "num residues placeholder",
          null,
          null
        ],
        "seq_length": [],
        "seq_mask": [
          "num residues placeholder"
        ],
        "target_feat": [
          "num residues placeholder",
          null
        ],
        "template_aatype": [
          "num templates placeholder",
          "num residues placeholder"
        ],
        "template_all_atom_masks": [
          "num templates placeholder",
          "num residues placeholder",
          null
        ],
        "template_all_atom_positions": [
          "num templates placeholder",
          "num residues placeholder",
          null,
          null
        ],
        "template_backbone_affine_mask": [
          "num templates placeholder",
          "num residues placeholder"
        ],
        "template_backbone_affine_tensor": [
          "num templates placeholder",
          "num residues placeholder",
          null
        ],
        "template_mask": [
          "num templates placeholder"
        ],
        "template_pseudo_beta": [
          "num templates placeholder",
          "num residues placeholder",
          null
        ],
        "template_pseudo_beta_mask": [
          "num templates placeholder",
          "num residues placeholder"
        ],
        "template_sum_probs": [
          "num templates placeholder",
          null
        ],
        "true_msa": [
          "msa placeholder",
          "num residues placeholder"
        ]
      },
      "fixed_size": true,
      "masked_msa_replace_fraction": 0.15,
      "max_msa_clusters": 512,
      "max_templates": 4,
      "num_ensemble": 1,
      "subsample_templates": false
    }
  },
  "model": {
    "calc_extended_ptm": false,
    "embeddings_and_evoformer": {
      "evoformer": {
        "msa_column_attention": {
          "dropout_rate": 0.0,
          "gating": true,
          "num_head": 8,
          "orientation": "per_column",
          "shared_dropout": true
        },
        "msa_row_attention_with_pair_bias": {
          "dropout_rate": 0.15,
          "gating": true,
          "num_head": 8,
          "orientation": "per_row",
          "shared_dropout": true
        },
        "msa_transition": {
          "dropout_rate": 0.0,
          "num_intermediate_factor": 4,
          "orientation": "per_row",
          "shared_dropout": true
        },
        "outer_product_mean": {
          "chunk_size": 128,
          "dropout_rate": 0.0,
          "first": false,
          "num_outer_channel": 32,
          "orientation": "per_row",
          "shared_dropout": true
        },
        "pair_transition": {
          "dropout_rate": 0.0,
          "num_intermediate_factor": 4,
          "orientation": "per_row",
          "shared_dropout": true
        },
        "triangle_attention_ending_node": {
          "dropout_rate": 0.25,
          "gating": true,
          "num_head": 4,
          "orientation": "per_column",
          "shared_dropout": true
        },
        "triangle_attention_starting_node": {
          "dropout_rate": 0.25,
          "gating": true,
          "num_head": 4,
          "orientation": "per_row",
          "shared_dropout": true
        },
        "triangle_multiplication_incoming": {
          "dropout_rate": 0.25,
          "equation": "kjc,kic->ijc",
          "fuse_projection_weights": true,
          "num_intermediate_channel": 128,
          "orientation": "per_row",
          "shared_dropout": true
        },
        "triangle_multiplication_outgoing": {
          "dropout_rate": 0.25,
          "equation": "ikc,jkc->ijc",
          "fuse_projection_weights": true,
          "num_intermediate_channel": 128,
          "orientation": "per_row",
          "shared_dropout": true
        }
      },
      "evoformer_num_block": 48,
      "extra_msa_channel": 64,
      "extra_msa_stack_num_block": 4,
      "max_relative_feature": 32,
      "msa_channel": 256,
      "pair_channel": 128,
      "prev_pos": {
        "max_bin": 20.75,
        "min_bin": 3.25,
        "num_bins": 15
      },
      "recycle_features": true,
      "recycle_pos": true,
      "seq_channel": 384,
      "template": {
        "attention": {
          "gating": false,
          "key_dim": 64,
          "num_head": 4,
          "value_dim": 64
        },
        "dgram_features": {
          "max_bin": 50.75,
          "min_bin": 3.25,
          "num_bins": 39
        },
        "embed_torsion_angles": false,
        "enabled": false,
        "max_templates": 4,
        "subbatch_size": 128,
        "template_pair_stack": {
          "num_block": 2,
          "pair_transition": {
            "dropout_rate": 0.0,
            "num_intermediate_factor": 2,
            "orientation": "per_row",
            "shared_dropout": true
          },
          "triangle_attention_ending_node": {
            "dropout_rate": 0.25,
            "gating": true,
            "key_dim": 64,
            "num_head": 4,
            "orientation": "per_column",
            "shared_dropout": true,
            "value_dim": 64
          },
          "triangle_attention_starting_node": {
            "dropout_rate": 0.25,
            "gating": true,
            "key_dim": 64,
            "num_head": 4,
            "orientation": "per_row",
            "shared_dropout": true,
            "value_dim": 64
          },
          "triangle_multiplication_incoming": {
            "dropout_rate": 0.25,
            "equation": "kjc,kic->ijc",
            "fuse_projection_weights": true,
            "num_intermediate_channel": 64,
            "orientation": "per_row",
            "shared_dropout": true
          },
          "triangle_multiplication_outgoing": {
            "dropout_rate": 0.25,
            "equation": "ikc,jkc->ijc",
            "fuse_projection_weights": true,
            "num_intermediate_channel": 64,
            "orientation": "per_row",
            "shared_dropout": true
          }
        },
        "use_template_unit_vector": false
      }
    },
    "global_config": {
      "bfloat16": true,
      "bfloat16_output": false,
      "deterministic": false,
      "eval_dropout": false,
      "multimer_mode": false,
      "subbatch_size": 4,
      "use_pallas": false,
      "use_remat": false,
      "zero_init": true
    },
    "heads": {
      "distogram": {
        "first_break": 2.3125,
        "last_break": 21.6875,
        "num_bins": 64,
        "weight": 0.0
      },
      "experimentally_resolved": {
        "filter_by_resolution": true,
        "max_resolution": 3.0,
        "min_resolution": 0.1,
        "weight": 0.0
      },
      "masked_msa": {
        "num_output": 23,
        "weight": 0.0
      },
      "predicted_aligned_error": {
        "filter_by_resolution": true,
        "max_error_bin": 31.0,
        "max_resolution": 3.0,
        "min_resolution": 0.1,
        "num_bins": 64,
        "num_channels": 128,
        "weight": 0.1
      },
      "predicted_lddt": {
        "filter_by_resolution": true,
        "max_resolution": 3.0,
        "min_resolution": 0.1,
        "num_bins": 50,
        "num_channels": 128,
        "weight": 0.01
      },
      "structure_module": {
        "angle_norm_weight": 0.01,
        "chi_weight": 0.5,
        "clash_overlap_tolerance": 1.5,
        "compute_in_graph_metrics": true,
        "dropout": 0.1,
        "fape": {
          "clamp_distance": 10.0,
          "clamp_type": "relu",
          "loss_unit_distance": 10.0
        },
        "num_channel": 384,
        "num_head": 12,
        "num_layer": 8,
        "num_layer_in_transition": 3,
        "num_point_qk": 4,
        "num_point_v": 8,
        "num_scalar_qk": 16,
        "num_scalar_v": 16,
        "position_scale": 10.0,
        "sidechain": {
          "atom_clamp_distance": 10.0,
          "length_scale": 10.0,
          "num_channel": 128,
          "num_residual_block": 2,
          "weight_frac": 0.5
        },
        "structural_violation_loss_weight": 1.0,
        "violation_tolerance_factor": 12.0,
        "weight": 1.0
      }
    },
    "num_recycle": 3,
    "rank_by": "plddt",
    "recycle_early_stop_tolerance": 0.0,
    "resample_msa_in_recycling": true,
    "stop_at_score": 100.0,
    "use_probs_extended": true
  }
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-4a2548d8f634

원래 경로: `colabfold/results/vmap_batch_20261028_185758_495006/summary.json` · SHA256: `afafa1ea1f383449e97d09dea06f0b9e2eaa0962154d1396e7208b1f1d5c3d88`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_batch_20261028_185758_495006/summary.json", "sha256": "afafa1ea1f383449e97d09dea06f0b9e2eaa0962154d1396e7208b1f1d5c3d88", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "stopped",
  "run_id": "vmap_batch_20261028_185758_495006",
  "workers": [
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
  ],
  "failed": [
    {
      "batch": 1,
      "exit_code": 1
    }
  ],
  "scope": "experimental JAX vmap independent monomer batch axis on existing ColabFold model_1, no simultaneous GPU jobs",
  "settings": {
    "guard_available_gib": 16,
    "guard_swap_increase_gib": 2
  },
  "stop_reason": "worker_error_or_guard_see_log"
}
```
<!-- END_RESTORE_FILE -->


</details>
