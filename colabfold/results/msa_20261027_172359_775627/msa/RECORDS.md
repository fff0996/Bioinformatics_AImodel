# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-b0a4a1a98d19

원래 경로: `colabfold/results/msa_20261027_172359_775627/msa/config.json` · SHA256: `1f89e453012a50e90c1c56b85bd9e46af13becf500958c9d6abbadab02df692d`

<details>
<summary>config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_20261027_172359_775627/msa/config.json", "sha256": "1f89e453012a50e90c1c56b85bd9e46af13becf500958c9d6abbadab02df692d", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
    "num_queries": 7,
    "use_templates": false,
    "num_relax": 0,
    "relax_max_iterations": 2000,
    "relax_tolerance": 2.39,
    "relax_stiffness": 10.0,
    "relax_max_outer_iterations": 3,
    "msa_mode": "mmseqs2_uniref_env",
    "model_type": "alphafold2_ptm",
    "num_models": 0,
    "num_recycles": null,
    "recycle_early_stop_tolerance": null,
    "num_ensemble": 1,
    "model_order": [
        1,
        2,
        3,
        4,
        5
    ],
    "initial_guess": null,
    "keep_existing_results": true,
    "rank_by": "plddt",
    "max_seq": 512,
    "max_extra_seq": 5120,
    "pair_mode": "unpaired_paired",
    "pairing_strategy": "greedy",
    "host_url": "https://api.colabfold.com",
    "user_agent": "colabfold/1.6.2",
    "stop_at_score": 100,
    "random_seed": 0,
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

## file-cd89d79bb9bd

원래 경로: `colabfold/results/msa_20261027_172359_775627/msa/log.txt` · SHA256: `79706534b7e8cf572e0bd47ac43919a807253fc7617f756dc0d00cc6a8d37093`

<details>
<summary>log.txt — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_20261027_172359_775627/msa/log.txt", "sha256": "79706534b7e8cf572e0bd47ac43919a807253fc7617f756dc0d00cc6a8d37093", "encoding": "utf-8", "added_final_newline": false} -->
```txt
2026-10-27 08:24:00,446 Running colabfold 1.6.2
2026-10-27 08:24:00,840 Running on GPU
2026-10-27 08:24:00,890 Found 4 citations for tools or databases
2026-10-27 08:24:00,890 Query 1/7: 1UBQ (length 76)
2026-10-27 08:24:01,002 /cache is not a writable directory
2026-10-27 08:24:01,002 Matplotlib created a temporary cache directory at /tmp/matplotlib-_iwj4duh because there was an issue with the default path ({configdir}); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
2026-10-27 08:24:01,140 generated new fontManager
2026-10-27 08:24:12,929 Saved /work/results/msa_20261027_172359_775627/msa/1UBQ.pickle
2026-10-27 08:24:13,453 Query 2/7: count1_A2RU14 (length 115)
2026-10-27 08:24:15,156 Saved /work/results/msa_20261027_172359_775627/msa/count1_A2RU14.pickle
2026-10-27 08:24:15,303 Query 3/7: count2_A6NFY7 (length 115)
2026-10-27 08:24:20,282 Saved /work/results/msa_20261027_172359_775627/msa/count2_A6NFY7.pickle
2026-10-27 08:24:20,462 Query 4/7: count3_O14519 (length 115)
2026-10-27 08:24:22,807 Saved /work/results/msa_20261027_172359_775627/msa/count3_O14519.pickle
2026-10-27 08:24:22,972 Query 5/7: count4_P01270 (length 115)
2026-10-27 08:24:24,977 Saved /work/results/msa_20261027_172359_775627/msa/count4_P01270.pickle
2026-10-27 08:24:25,134 Query 6/7: 1LYZ (length 129)
2026-10-27 08:24:31,220 Saved /work/results/msa_20261027_172359_775627/msa/1LYZ.pickle
2026-10-27 08:24:31,477 Query 7/7: 1TIM (length 247)
2026-10-27 08:24:55,558 Saved /work/results/msa_20261027_172359_775627/msa/1TIM.pickle
2026-10-27 08:24:56,355 Done
```
<!-- END_RESTORE_FILE -->


</details>
