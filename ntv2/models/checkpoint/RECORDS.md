# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-e647f95574d9

원래 경로: `ntv2/models/checkpoint/config.json` · SHA256: `27e19a9e3fbba65b933bde793b503734744b5c6e376c74dcd1cb7317876efab8`

<details>
<summary>config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/models/checkpoint/config.json", "sha256": "27e19a9e3fbba65b933bde793b503734744b5c6e376c74dcd1cb7317876efab8", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "add_bias_fnn": false,
  "architectures": [
    "EsmForMaskedLM",
    "EsmForTokenClassification",
    "EsmForSequenceClassification"
  ],
  "attention_probs_dropout_prob": 0.0,
  "auto_map": {
    "AutoConfig": "esm_config.EsmConfig",
    "AutoModelForMaskedLM": "modeling_esm.EsmForMaskedLM",
    "AutoModelForTokenClassification": "modeling_esm.EsmForTokenClassification",
    "AutoModelForSequenceClassification": "modeling_esm.EsmForSequenceClassification"
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
  "num_attention_heads": 16,
  "num_hidden_layers": 29,
  "pad_token_id": 1,
  "position_embedding_type": "rotary",
  "tie_word_embeddings": false,
  "token_dropout": false,
  "torch_dtype": "float32",
  "transformers_version": "4.32.0.dev0",
  "use_cache": false,
  "vocab_list": null,
  "vocab_size": 4107
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-d39519f0198c

원래 경로: `ntv2/models/checkpoint/tokenizer_config.json` · SHA256: `253d338919eba938e50b776f3243cc739462c207fe64e3d2e81cc5e681bee45b`

<details>
<summary>tokenizer_config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/models/checkpoint/tokenizer_config.json", "sha256": "253d338919eba938e50b776f3243cc739462c207fe64e3d2e81cc5e681bee45b", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "clean_up_tokenization_spaces": true,
  "eos_token": null,
  "model_max_length": 2048,
  "tokenizer_class": "EsmTokenizer"
}
```
<!-- END_RESTORE_FILE -->


</details>
