# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-4a66ef9cde8f

원래 경로: `scgpt/models/whole-human/args.json` · SHA256: `c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60`

<details>
<summary>args.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/models/whole-human/args.json", "sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "data_source": "/scratch/ssd004/datasets/cellxgene/scb_strict/human",
  "save_dir": "/scratch/ssd004/datasets/cellxgene/save/cellxgene_census_human-May23-08-36-2023",
  "load_model": null,
  "n_hvg": null,
  "valid_size_or_ratio": 0.003,
  "dist_backend": "nccl",
  "grad_accu_steps": 1,
  "pad_token": "<pad>",
  "input_style": "binned",
  "input_emb_style": "continuous",
  "n_bins": 51,
  "max_seq_len": 1200,
  "training_tasks": "both",
  "dist_url": "tcp://gpu188.cluster.local:53833",
  "mask_ratio": [
    0.25,
    0.5,
    0.75
  ],
  "trunc_by_sample": true,
  "vocab_path": "/scratch/ssd004/datasets/cellxgene/scFormer/scformer/tokenizer/default_census_vocab.json",
  "rank": 0,
  "batch_size": 32,
  "eval_batch_size": 64,
  "epochs": 6,
  "lr": 0.0001,
  "scheduler_interval": 100,
  "scheduler_factor": 0.99,
  "warmup_ratio_or_step": 10000.0,
  "no_cls": true,
  "no_cce": true,
  "fp16": true,
  "fast_transformer": true,
  "nlayers": 12,
  "nheads": 8,
  "embsize": 512,
  "d_hid": 512,
  "dropout": 0.2,
  "n_layers_cls": 3,
  "log_interval": 9000,
  "save_interval": 27000,
  "mask_value": -1,
  "pad_value": -2,
  "USE_CLS": false,
  "USE_CCE": false,
  "MVC": true,
  "USE_GENERATIVE_TRAINING": true,
  "world_size": 16,
  "distributed": true,
  "local_rank": 0,
  "gpu": 0
}
```
<!-- END_RESTORE_FILE -->


</details>
