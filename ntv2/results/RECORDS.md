# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-f8ca57f195c7

원래 경로: `ntv2/results/setup_exit.json` · SHA256: `124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01`

<details>
<summary>setup_exit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/setup_exit.json", "sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01", "encoding": "utf-8", "added_final_newline": true} -->
```json
{"exit_code": 0}
```
<!-- END_RESTORE_FILE -->


</details>

## file-777e836fcf8a

원래 경로: `ntv2/results/sources.json` · SHA256: `ec9d718268e93f262d932a04fc41c4a1946062bf9ce8a1ed886734cb65d4a740`

<details>
<summary>sources.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "ntv2/results/sources.json", "sha256": "ec9d718268e93f262d932a04fc41c4a1946062bf9ce8a1ed886734cb65d4a740", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/README.md",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/README.md",
    "sha256": "16352a70466c2eac7db56fde1c18bdf24f53515c06c7d55237f3428a032286c7",
    "bytes": 6346
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/config.json",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/config.json",
    "sha256": "27e19a9e3fbba65b933bde793b503734744b5c6e376c74dcd1cb7317876efab8",
    "bytes": 1065
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/esm_config.py",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/esm_config.py",
    "sha256": "a44e859baa08465ecdcd76b0a73f6e4fe011245a212931c403de1149ae9613ec",
    "bytes": 14876
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/model.safetensors",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/model.safetensors",
    "sha256": "7a00ffaaf8f3ca627741a61e568d7e550b5c79aada70c76c8cb861192fbe7c59",
    "bytes": 1993438552
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/modeling_esm.py",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/modeling_esm.py",
    "sha256": "f2b003f45d2fa4f94e92d8bbef927b719dfdd7ab7c7a95743daa2b10ab140cb3",
    "bytes": 58205
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/special_tokens_map.json",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/special_tokens_map.json",
    "sha256": "d6dc30bf018166daab248b0abf7efda6fd1b1e0a2d1bee5b31b23db2ebdaee77",
    "bytes": 101
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/tokenizer_config.json",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/tokenizer_config.json",
    "sha256": "253d338919eba938e50b776f3243cc739462c207fe64e3d2e81cc5e681bee45b",
    "bytes": 129
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/models/checkpoint/vocab.txt",
    "url": "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species/resolve/06615c1660c892fc199840c18123f8385b3542a8/vocab.txt",
    "sha256": "c00e0ad166d6ab3f7540ebc92270392e581bb3106763412f29034b49323e1052",
    "bytes": 28718
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/data/enhancers/test.parquet",
    "url": "https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks/resolve/96d86d567d4cd33536e49b429dc7983121619a08/enhancers/test.parquet",
    "sha256": "0e1862a1809124051014b60ea9021e9d414561c194af44fde0ce0eac5ac220e1",
    "bytes": 41136
  },
  {
    "path": "/home/hyein/ai-benchmark/ntv2/data/enhancers/train.parquet",
    "url": "https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks/resolve/96d86d567d4cd33536e49b429dc7983121619a08/enhancers/train.parquet",
    "sha256": "698167c5c09b1f98e240560c51e81e540b27aa91e162e3ff1cfcc204ae1ee6e8",
    "bytes": 1465411
  }
]
```
<!-- END_RESTORE_FILE -->


</details>
