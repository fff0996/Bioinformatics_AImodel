# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-c7b820f8e0ae

원래 경로: `ntv2/models/ntv2_20261027_201332_251243/classifier/config.json` · SHA256: `bfe3ca2332354c0cb7a6bb0ad26973a336a754521fcc0bb42684a4403e98e6f5`

<details>
<summary>config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/models/ntv2_20261027_201332_251243/classifier/config.json", "sha256": "bfe3ca2332354c0cb7a6bb0ad26973a336a754521fcc0bb42684a4403e98e6f5", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "_name_or_path": "/work/models/checkpoint",
  "add_bias_fnn": false,
  "architectures": [
    "EsmForSequenceClassification"
  ],
  "attention_probs_dropout_prob": 0.0,
  "auto_map": {
    "AutoConfig": "esm_config.EsmConfig",
    "AutoModelForMaskedLM": "modeling_esm.EsmForMaskedLM",
    "AutoModelForSequenceClassification": "modeling_esm.EsmForSequenceClassification",
    "AutoModelForTokenClassification": "modeling_esm.EsmForTokenClassification"
  },
  "emb_layer_norm_before": false,
  "esmfold_config": null,
  "hidden_dropout_prob": 0.0,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "is_folding_model": false,
  "layer_norm_eps": 1e-12,
  "mask_token_id": 2,
  "max_position_embeddings": 2050,
  "model_type": "esm",
  "num_attention_heads": 16,
  "num_hidden_layers": 29,
  "pad_token_id": 1,
  "position_embedding_type": "rotary",
  "problem_type": "single_label_classification",
  "tie_word_embeddings": false,
  "token_dropout": false,
  "torch_dtype": "float32",
  "transformers_version": "4.44.2",
  "use_cache": false,
  "vocab_list": null,
  "vocab_size": 4107
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-83415068a0e3

원래 경로: `ntv2/models/ntv2_20261027_201332_251243/classifier/tokenizer_config.json` · SHA256: `546e0b71a1729d81a378c192fa670bf23896c3e3d76cec6e53734d276362af75`

<details>
<summary>tokenizer_config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/models/ntv2_20261027_201332_251243/classifier/tokenizer_config.json", "sha256": "546e0b71a1729d81a378c192fa670bf23896c3e3d76cec6e53734d276362af75", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "added_tokens_decoder": {
    "0": {
      "content": "<unk>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "1": {
      "content": "<pad>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "2": {
      "content": "<mask>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "3": {
      "content": "<cls>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
  },
  "clean_up_tokenization_spaces": true,
  "cls_token": "<cls>",
  "eos_token": null,
  "mask_token": "<mask>",
  "model_max_length": 2048,
  "pad_token": "<pad>",
  "tokenizer_class": "EsmTokenizer",
  "unk_token": "<unk>"
}
```
<!-- END_RESTORE_FILE -->


</details>
