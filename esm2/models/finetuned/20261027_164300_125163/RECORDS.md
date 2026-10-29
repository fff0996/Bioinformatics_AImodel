# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-ce28c386189b

원래 경로: `esm2/models/finetuned/20261027_164300_125163/metadata.json` · SHA256: `d05d1be4796a26a411c91a0491ceca0b48e0f488cc52646249fbe9bd0e53a78d`

<details>
<summary>metadata.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "esm2/models/finetuned/20261027_164300_125163/metadata.json", "sha256": "d05d1be4796a26a411c91a0491ceca0b48e0f488cc52646249fbe9bd0e53a78d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "target_mean": 3.173424863256514,
  "target_std": 0.8272688422178766,
  "base_checkpoint": "esm2_t33_650M_UR50D.pt",
  "base_sha256": "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
  "epoch": 1,
  "seed": 42,
  "training_scope": "final transformer block + final layer norm + linear regression head",
  "resume_training_verified": false
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-03d7122ffc65

원래 경로: `esm2/models/finetuned/20261027_164300_125163/preprocessing.json` · SHA256: `56a6c2288bd2b753f8b7ba30650efbca8666c114a9aa38a77686a01dff861d6d`

<details>
<summary>preprocessing.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "esm2/models/finetuned/20261027_164300_125163/preprocessing.json", "sha256": "56a6c2288bd2b753f8b7ba30650efbca8666c114a9aa38a77686a01dff861d6d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "architecture": "ESM-1b",
  "tokens": [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "<null_1>",
    "<mask>"
  ],
  "residue_limit": 1022,
  "prepend_bos": true,
  "append_eos": true,
  "pooling": "residue mean; exclude BOS/EOS/padding",
  "label": "continuous TAPE log_fluorescence",
  "label_mapping": null
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-0e477d9a95d1

원래 경로: `esm2/models/finetuned/20261027_164300_125163/training_config.json` · SHA256: `8108b77c4416fb5129d9c62da99ff076ea9b0903c53ad2901bc585310378279b`

<details>
<summary>training_config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "esm2/models/finetuned/20261027_164300_125163/training_config.json", "sha256": "8108b77c4416fb5129d9c62da99ff076ea9b0903c53ad2901bc585310378279b", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "optimizer": "AdamW",
  "learning_rate_backbone": 0.0001,
  "learning_rate_head": 0.001,
  "weight_decay": 0.01,
  "epochs": 1,
  "steps": 8,
  "token_budget": 4096,
  "precision": "FP32",
  "tf32": false,
  "seed": 42,
  "trainable_blocks": [
    "layers.32",
    "emb_layer_norm_after",
    "regression_head"
  ],
  "training_inputs": "/home/hyein/ai-benchmark/esm2/results/20261027_164300_125163/training_inputs.json",
  "base_environment": "/home/hyein/ai-benchmark/esm2/.venv/pyvenv.cfg"
}
```
<!-- END_RESTORE_FILE -->


</details>
