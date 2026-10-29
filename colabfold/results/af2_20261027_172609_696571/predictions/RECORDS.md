# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-0f80f272f05f

원래 경로: `colabfold/results/af2_20261027_172609_696571/predictions/config.json` · SHA256: `e82886c416da5105bd891dc47ca493e9f4136aae0dd38faeafb9306d30fa6207`

<details>
<summary>config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/af2_20261027_172609_696571/predictions/config.json", "sha256": "e82886c416da5105bd891dc47ca493e9f4136aae0dd38faeafb9306d30fa6207", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
    "num_queries": 7,
    "use_templates": false,
    "num_relax": 0,
    "relax_max_iterations": 0,
    "relax_tolerance": 2.39,
    "relax_stiffness": 10.0,
    "relax_max_outer_iterations": 3,
    "msa_mode": "mmseqs2_uniref_env",
    "model_type": "alphafold2_ptm",
    "num_models": 5,
    "num_recycles": 3,
    "recycle_early_stop_tolerance": 0.0,
    "num_ensemble": 1,
    "model_order": [
        1,
        2,
        3,
        4,
        5
    ],
    "initial_guess": null,
    "keep_existing_results": false,
    "rank_by": "plddt",
    "max_seq": 512,
    "max_extra_seq": 5120,
    "pair_mode": "unpaired_paired",
    "pairing_strategy": "greedy",
    "host_url": "https://api.colabfold.com",
    "user_agent": "",
    "stop_at_score": 100,
    "random_seed": 42,
    "num_seeds": 1,
    "recompile_padding": 10,
    "commit": null,
    "use_dropout": false,
    "use_cluster_profile": true,
    "use_fuse": true,
    "use_bfloat16": true,
    "use_pallas": false,
    "compile_mode": "tuned",
    "version": "1.6.2",
    "calc_extra_ptm": false,
    "use_probs_extra": true,
    "max_template_date": "2100-01-01",
    "max_template_hits": 20
}
```
<!-- END_RESTORE_FILE -->


</details>
