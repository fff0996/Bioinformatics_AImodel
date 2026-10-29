# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-8d03be18c0a9

원래 경로: `FOLLOWUP_QUEUE.json` · SHA256: `ec0aab0130dff2a9b351a406dd1af4f2c679e9d562a2c5236d9f044820aef1a3`

<details>
<summary>FOLLOWUP_QUEUE.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "FOLLOWUP_QUEUE.json", "sha256": "ec0aab0130dff2a9b351a406dd1af4f2c679e9d562a2c5236d9f044820aef1a3", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "status": "all_ten_followups_completed_with_scope_differences",
  "queue_id": "20261028_142752_517613",
  "order": [
    "scvi",
    "dnabert2",
    "ntv2",
    "rnafm",
    "openspliceai"
  ],
  "models": {
    "scvi": {
      "status": "complete",
      "run_id": "batch_followup_20261028_142752_517769",
      "largest_successful_batch": 10176
    },
    "dnabert2": {
      "status": "complete",
      "run_id": "batch_followup_20261028_143041_359149",
      "largest_successful_batch": 512
    },
    "ntv2": {
      "status": "complete",
      "run_id": "batch_followup_20261028_143713_869625",
      "largest_successful_batch": 128
    },
    "rnafm": {
      "status": "complete",
      "run_id": "batch_followup_20261028_144452_703996",
      "largest_successful_batch": 76
    },
    "openspliceai": {
      "status": "complete",
      "run_id": "batch_followup_20261028_144737_167765",
      "largest_successful_batch": 156
    }
  },
  "scope": "inference batch expansion, same existing weights; sequential GPU only",
  "applicability_notes": {
    "AlphaFold2": "후속 실제 실행 결과가 기존 미실시 기록을 갱신함: complete. CLI 기본 배치와 별도 구현/계측 범위는 새 보고서 참조.",
    "Clair3": "후속 실제 실행 결과가 기존 미실시 기록을 갱신함: complete. CLI 기본 배치와 별도 구현/계측 범위는 새 보고서 참조."
  },
  "additional_models": {
    "AlphaFold2": {
      "status": "complete",
      "run_id": "vmap_fp32_20261028_190843_576148",
      "report": "colabfold/reports/Batch_memory_followup.md"
    },
    "Clair3": {
      "status": "complete",
      "run_id": "candidate_batch_20261028_185641_089149",
      "report": "clair3/reports/Batch_memory_followup.md"
    }
  },
  "completion_definition": "execution/output/report completion only; does not mean 80 percent allocated memory goal reached",
  "memory_80percent_goal": {
    "esm2": "unconfirmed",
    "colabfold": "unconfirmed",
    "scvi": "unconfirmed",
    "cellpose": "unconfirmed",
    "dnabert2": "unconfirmed",
    "ntv2": "unconfirmed",
    "clair3": "unconfirmed",
    "rnafm": "unconfirmed",
    "openspliceai": "unconfirmed",
    "scgpt": "confirmed only in fixed1200padding training capacity probe; regular full-data training unconfirmed"
  }
}
```
<!-- END_RESTORE_FILE -->


</details>
