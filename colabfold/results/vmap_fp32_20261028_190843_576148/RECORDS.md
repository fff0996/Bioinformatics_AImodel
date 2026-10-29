# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-830df92fc0fd

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/batch1_command.json` · SHA256: `ad73ef573ce45307ddd3c0386e5374c451da1e20908f849c99560468707f9f14`

<details>
<summary>batch1_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch1_command.json", "sha256": "ad73ef573ce45307ddd3c0386e5374c451da1e20908f849c99560468707f9f14", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "af2-vmap-vmap_fp32_20261028_190843_576148-1",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "RUN_ID=vmap_fp32_20261028_190843_576148",
  "-e",
  "BATCH=1",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_fp32_20261028_190843_576148/mplcache",
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

## file-71a9e200f78d

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/batch1_exit.json` · SHA256: `7eb82672f3705de0cbe750d83fedab0a43da9476163acd65dc11cdb95c8418d4`

<details>
<summary>batch1_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch1_exit.json", "sha256": "7eb82672f3705de0cbe750d83fedab0a43da9476163acd65dc11cdb95c8418d4", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 64.58476543426514, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-36fa58f4b3fe

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/batch2_command.json` · SHA256: `08e32def6516646237a84e9558052a62015dfb3d7031a3e8ca41e08dc4ab0ce8`

<details>
<summary>batch2_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch2_command.json", "sha256": "08e32def6516646237a84e9558052a62015dfb3d7031a3e8ca41e08dc4ab0ce8", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "af2-vmap-vmap_fp32_20261028_190843_576148-2",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "RUN_ID=vmap_fp32_20261028_190843_576148",
  "-e",
  "BATCH=2",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_fp32_20261028_190843_576148/mplcache",
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

## file-a48c5b3fd4ec

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/batch2_exit.json` · SHA256: `9ee5e4a3fbb2949b24ea4efd96e8cb67ea37bc6e1498a9b3d4399dd80628e751`

<details>
<summary>batch2_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch2_exit.json", "sha256": "9ee5e4a3fbb2949b24ea4efd96e8cb67ea37bc6e1498a9b3d4399dd80628e751", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 82.10873079299927, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-4e2b87343e37

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/batch4_command.json` · SHA256: `74e4c1cf908f63d2330b27060a571896d7c79e36aa3337e345ff1154d2c8ff78`

<details>
<summary>batch4_command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch4_command.json", "sha256": "74e4c1cf908f63d2330b27060a571896d7c79e36aa3337e345ff1154d2c8ff78", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "docker",
  "run",
  "--rm",
  "--name",
  "af2-vmap-vmap_fp32_20261028_190843_576148-4",
  "--gpus",
  "all",
  "--network",
  "none",
  "--ipc=host",
  "--user",
  "1000:1000",
  "-e",
  "RUN_ID=vmap_fp32_20261028_190843_576148",
  "-e",
  "BATCH=4",
  "-e",
  "JAX_PLATFORMS=cuda,cpu",
  "-e",
  "XLA_PYTHON_CLIENT_PREALLOCATE=false",
  "-e",
  "MPLCONFIGDIR=/work/results/vmap_fp32_20261028_190843_576148/mplcache",
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

## file-7acdc5afe1d1

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/batch4_exit.json` · SHA256: `d06b36c239f0a6e1880f1f99318fc27c063d5041da8d383123dcd936f0d7a55b`

<details>
<summary>batch4_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch4_exit.json", "sha256": "d06b36c239f0a6e1880f1f99318fc27c063d5041da8d383123dcd936f0d7a55b", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0, "wall_seconds": 154.21510648727417, "reason": null}
```
<!-- END_RESTORE_FILE -->


</details>

## file-978117bf8270

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/capture_exit.json` · SHA256: `124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01`

<details>
<summary>capture_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/capture_exit.json", "sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0}
```
<!-- END_RESTORE_FILE -->


</details>

## file-89f7010b0c5c

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/experiment_protocol.json` · SHA256: `81cd4a4720a2e6bb66efb2ad1b7efc7e2cd2504dc4e578eacf24e5b5d0445b6a`

<details>
<summary>experiment_protocol.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/experiment_protocol.json", "sha256": "81cd4a4720a2e6bb66efb2ad1b7efc7e2cd2504dc4e578eacf24e5b5d0445b6a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "runtime_precision": "FP32 activations and parameters, fixed across batch1/2/4; recycle boundary output/prev FP16 conversion matches native RunModel.predict",
  "feature_capture_runner_config": "runner_config.json was captured during CPU preprocessing under original BF16 config; not the runtime precision",
  "bf16_reference_failed_run": "vmap_batch_retry_20261028_190537_289704",
  "batch1_reference_ca_rmsd_gate_angstrom": 0.5,
  "gpu_order": "Clair3 completed before AlphaFold2 GPU jobs started"
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-16252fd1c1bd

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/input_audit.json` · SHA256: `34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d`

<details>
<summary>input_audit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/input_audit.json", "sha256": "34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d", "encoding": "utf-8", "added_final_newline": true} -->
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

## file-97a7bb41b74a

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/parent_run.json` · SHA256: `50f5eddfefb44456fa3a430f36ff322d5ac59bc079f2f3dd8dcd181fb6d33dd4`

<details>
<summary>parent_run.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/parent_run.json", "sha256": "50f5eddfefb44456fa3a430f36ff322d5ac59bc079f2f3dd8dcd181fb6d33dd4", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "parent_run": "vmap_batch_retry_20261028_190537_289704",
  "reason": "BF16 vmap vs native reference CA alignedRMSD1.075A and pLDDT max2.781 failed preset0.5A gate. Try FP32 fixed consistently across batch1/2/4 to test numeric sensitivity; not tuned to accuracy or memory target."
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-7c084fc05944

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/runner_config.json` · SHA256: `707eb227ce413194fb0bd2a7a6f44bbb0199b5ca4fb7ac8a2a0c8cc4becbdf7a`

<details>
<summary>runner_config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/runner_config.json", "sha256": "707eb227ce413194fb0bd2a7a6f44bbb0199b5ca4fb7ac8a2a0c8cc4becbdf7a", "encoding": "utf-8", "added_final_newline": true} -->
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

## file-ff8c01654321

원래 경로: `colabfold/results/vmap_fp32_20261028_190843_576148/summary.json` · SHA256: `aa0105d06562fc9ba562769b4d637acf865004fa070a584995bc28a07109516b`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/vmap_fp32_20261028_190843_576148/summary.json", "sha256": "aa0105d06562fc9ba562769b4d637acf865004fa070a584995bc28a07109516b", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "complete",
  "run_id": "vmap_fp32_20261028_190843_576148",
  "workers": [
    {
      "status": "complete",
      "batch": 1,
      "implementation": "experimental jax.vmap over unmodified ColabFold RunModel.apply; shared fixed model_1 weights, independent actual query features and RNG; not a native CLI batch option",
      "conditions": [
        {
          "batch": 1,
          "unique_queries": 1,
          "repeats": 2,
          "processed_queries": 2,
          "seconds": 22.847734451293945,
          "queries_per_second": 0.08753603138479812,
          "gpu_util_avg": 93.37209302325581,
          "gpu_util_max": 95.0,
          "sample_count": 43,
          "jax_memory_stats": {
            "num_allocs": 7811,
            "bytes_in_use": 463533056,
            "peak_bytes_in_use": 2321053696,
            "largest_alloc_size": 381945600,
            "bytes_limit": 97997815808,
            "bytes_reserved": 0,
            "peak_bytes_reserved": 0,
            "largest_free_block_bytes": 0,
            "pool_bytes": 4292870144,
            "peak_pool_bytes": 4292870144
          },
          "memory_definition": "peak_bytes_in_use is cumulative process allocator peak including compilation and warmup; execution bytes_in_use samples separate, no unsupported reset",
          "execution_sampled_peak_bytes_in_use": 881670656,
          "rss_sampled_peak_gib": 2.31463623046875,
          "available_sampled_min_gib": 107.46133804321289
        }
      ],
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
      "loading_seconds": 2.432121515274048,
      "parameters": 92907930,
      "weights_dtype": [
        "float32"
      ],
      "model_only_postload": {
        "jax": {
          "num_allocs": 3600,
          "bytes_in_use": 373869824,
          "peak_bytes_in_use": 2321053696,
          "largest_alloc_size": 99614720,
          "bytes_limit": 97997815808,
          "bytes_reserved": 0,
          "peak_bytes_reserved": 0,
          "largest_free_block_bytes": 0,
          "pool_bytes": 4292870144,
          "peak_pool_bytes": 4292870144
        },
        "system": {
          "rss_gib": 1.5773162841796875,
          "available_gib": 108.31254959106445,
          "total_gib": 121.68751525878906,
          "swap_gib": 0.00743865966796875
        }
      },
      "environment": {
        "jax": "0.10.2",
        "device": "cuda:0",
        "preallocate": "false",
        "compute": "FP32 activations/weights, explicit separate validation condition after BF16 vmap reference drift",
        "recycles": 3,
        "ensembles": 1,
        "templates": false,
        "relaxation": false,
        "seed_per_query": 42
      },
      "compilation_seconds": 7.9508092403411865,
      "compiled_memory_analysis": "CompiledMemoryStats(generated_code_size_in_bytes=25196, argument_size_in_bytes=390249042, output_size_in_bytes=10470204, alias_size_in_bytes=0, temp_size_in_bytes=381945560, host_generated_code_size_in_bytes=0, host_argument_size_in_bytes=0, host_output_size_in_bytes=0, host_alias_size_in_bytes=0, host_temp_size_in_bytes=0)",
      "warmup_seconds": 11.607905626296997,
      "predictions": [
        {
          "id": "count1_A2RU14",
          "residues": 115,
          "mean_plddt": 83.9375,
          "finite_and_pdb_valid": true
        }
      ],
      "native_reference_seconds_excluded": 17.070082187652588,
      "native_reference_comparison": {
        "ca_aligned_rmsd_angstrom": 0.17949736570387045,
        "plddt_max_abs": 0.1875,
        "native_recycles": 3
      }
    },
    {
      "status": "complete",
      "batch": 2,
      "implementation": "experimental jax.vmap over unmodified ColabFold RunModel.apply; shared fixed model_1 weights, independent actual query features and RNG; not a native CLI batch option",
      "conditions": [
        {
          "batch": 2,
          "unique_queries": 2,
          "repeats": 2,
          "processed_queries": 4,
          "seconds": 44.38909840583801,
          "queries_per_second": 0.09011221546851521,
          "gpu_util_avg": 93.6951219512195,
          "gpu_util_max": 96.0,
          "sample_count": 82,
          "jax_memory_stats": {
            "num_allocs": 7946,
            "bytes_in_use": 594572800,
            "peak_bytes_in_use": 2316647424,
            "largest_alloc_size": 762401280,
            "bytes_limit": 97997815808,
            "bytes_reserved": 0,
            "peak_bytes_reserved": 0,
            "largest_free_block_bytes": 0,
            "pool_bytes": 4292870144,
            "peak_pool_bytes": 4292870144
          },
          "memory_definition": "peak_bytes_in_use is cumulative process allocator peak including compilation and warmup; execution bytes_in_use samples separate, no unsupported reset",
          "execution_sampled_peak_bytes_in_use": 1441412096,
          "rss_sampled_peak_gib": 2.546154022216797,
          "available_sampled_min_gib": 107.30325317382812
        }
      ],
      "input": {
        "unique_queries": 2,
        "query_ids": [
          "count1_A2RU14",
          "count2_A6NFY7"
        ],
        "actual_residues": [
          115,
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
          512,
          512
        ]
      },
      "loading_seconds": 2.38616943359375,
      "parameters": 92907930,
      "weights_dtype": [
        "float32"
      ],
      "model_only_postload": {
        "jax": {
          "num_allocs": 3600,
          "bytes_in_use": 372647168,
          "peak_bytes_in_use": 2316647424,
          "largest_alloc_size": 99614720,
          "bytes_limit": 97997815808,
          "bytes_reserved": 0,
          "peak_bytes_reserved": 0,
          "largest_free_block_bytes": 0,
          "pool_bytes": 4292870144,
          "peak_pool_bytes": 4292870144
        },
        "system": {
          "rss_gib": 1.5784873962402344,
          "available_gib": 108.37319564819336,
          "total_gib": 121.68751525878906,
          "swap_gib": 0.00743865966796875
        }
      },
      "environment": {
        "jax": "0.10.2",
        "device": "cuda:0",
        "preallocate": "false",
        "compute": "FP32 activations/weights, explicit separate validation condition after BF16 vmap reference drift",
        "recycles": 3,
        "ensembles": 1,
        "templates": false,
        "relaxation": false,
        "seed_per_query": 42
      },
      "compilation_seconds": 10.201570510864258,
      "compiled_memory_analysis": "CompiledMemoryStats(generated_code_size_in_bytes=25196, argument_size_in_bytes=408980012, output_size_in_bytes=20940288, alias_size_in_bytes=0, temp_size_in_bytes=762401240, host_generated_code_size_in_bytes=0, host_argument_size_in_bytes=0, host_output_size_in_bytes=0, host_alias_size_in_bytes=0, host_temp_size_in_bytes=0)",
      "warmup_seconds": 22.34554100036621,
      "predictions": [
        {
          "id": "count1_A2RU14",
          "residues": 115,
          "mean_plddt": 83.875,
          "finite_and_pdb_valid": true
        },
        {
          "id": "count2_A6NFY7",
          "residues": 115,
          "mean_plddt": 71.4375,
          "finite_and_pdb_valid": true
        }
      ],
      "first_query_batch_comparison": {
        "ca_aligned_rmsd_angstrom": 0.0659733595834261,
        "plddt_max_abs": 0.125
      }
    },
    {
      "status": "complete",
      "batch": 4,
      "implementation": "experimental jax.vmap over unmodified ColabFold RunModel.apply; shared fixed model_1 weights, independent actual query features and RNG; not a native CLI batch option",
      "conditions": [
        {
          "batch": 4,
          "unique_queries": 4,
          "repeats": 2,
          "processed_queries": 8,
          "seconds": 90.66219711303711,
          "queries_per_second": 0.08823964402744008,
          "gpu_util_avg": 94.47590361445783,
          "gpu_util_max": 96.0,
          "sample_count": 166,
          "jax_memory_stats": {
            "num_allocs": 8017,
            "bytes_in_use": 796945408,
            "peak_bytes_in_use": 2478557696,
            "largest_alloc_size": 1523517696,
            "bytes_limit": 97997815808,
            "bytes_reserved": 0,
            "peak_bytes_reserved": 0,
            "largest_free_block_bytes": 0,
            "pool_bytes": 4292870144,
            "peak_pool_bytes": 4292870144
          },
          "memory_definition": "peak_bytes_in_use is cumulative process allocator peak including compilation and warmup; execution bytes_in_use samples separate, no unsupported reset",
          "execution_sampled_peak_bytes_in_use": 2478557696,
          "rss_sampled_peak_gib": 2.784984588623047,
          "available_sampled_min_gib": 106.94659042358398
        }
      ],
      "input": {
        "unique_queries": 4,
        "query_ids": [
          "count1_A2RU14",
          "count2_A6NFY7",
          "count3_O14519",
          "count4_P01270"
        ],
        "actual_residues": [
          115,
          115,
          115,
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
          512,
          512,
          512,
          512
        ]
      },
      "loading_seconds": 2.50765323638916,
      "parameters": 92907930,
      "weights_dtype": [
        "float32"
      ],
      "model_only_postload": {
        "jax": {
          "num_allocs": 3600,
          "bytes_in_use": 373954048,
          "peak_bytes_in_use": 2317922048,
          "largest_alloc_size": 99614720,
          "bytes_limit": 97997815808,
          "bytes_reserved": 0,
          "peak_bytes_reserved": 0,
          "largest_free_block_bytes": 0,
          "pool_bytes": 4292870144,
          "peak_pool_bytes": 4292870144
        },
        "system": {
          "rss_gib": 1.5867691040039062,
          "available_gib": 108.34519958496094,
          "total_gib": 121.68751525878906,
          "swap_gib": 0.00743865966796875
        }
      },
      "environment": {
        "jax": "0.10.2",
        "device": "cuda:0",
        "preallocate": "false",
        "compute": "FP32 activations/weights, explicit separate validation condition after BF16 vmap reference drift",
        "recycles": 3,
        "ensembles": 1,
        "templates": false,
        "relaxation": false,
        "seed_per_query": 42
      },
      "compilation_seconds": 12.51270866394043,
      "compiled_memory_analysis": "CompiledMemoryStats(generated_code_size_in_bytes=25196, argument_size_in_bytes=446441952, output_size_in_bytes=41880456, alias_size_in_bytes=0, temp_size_in_bytes=1523517656, host_generated_code_size_in_bytes=0, host_argument_size_in_bytes=0, host_output_size_in_bytes=0, host_alias_size_in_bytes=0, host_temp_size_in_bytes=0)",
      "warmup_seconds": 45.50552153587341,
      "predictions": [
        {
          "id": "count1_A2RU14",
          "residues": 115,
          "mean_plddt": 83.875,
          "finite_and_pdb_valid": true
        },
        {
          "id": "count2_A6NFY7",
          "residues": 115,
          "mean_plddt": 71.4375,
          "finite_and_pdb_valid": true
        },
        {
          "id": "count3_O14519",
          "residues": 115,
          "mean_plddt": 71.3125,
          "finite_and_pdb_valid": true
        },
        {
          "id": "count4_P01270",
          "residues": 115,
          "mean_plddt": 68.875,
          "finite_and_pdb_valid": true
        }
      ],
      "first_query_batch_comparison": {
        "ca_aligned_rmsd_angstrom": 0.18623998936360886,
        "plddt_max_abs": 0.1875
      }
    }
  ],
  "failed": [],
  "scope": "experimental JAX vmap independent monomer batch axis on existing ColabFold model_1, no simultaneous GPU jobs",
  "settings": {
    "guard_available_gib": 16,
    "guard_swap_increase_gib": 2
  },
  "stop_reason": "actual_four_query_pool_complete"
}
```
<!-- END_RESTORE_FILE -->


</details>
