# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-739209c3e5c2

원래 경로: `colabfold/results/msa_complex_20261027_173649_931405/msa/config.json` · SHA256: `90899c0db89ff7e9509c661616a41d9d6521e03e3207f6849a93f4830a2cc49a`

<details>
<summary>config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_complex_20261027_173649_931405/msa/config.json", "sha256": "90899c0db89ff7e9509c661616a41d9d6521e03e3207f6849a93f4830a2cc49a", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
    "num_queries": 1,
    "use_templates": false,
    "num_relax": 0,
    "relax_max_iterations": 2000,
    "relax_tolerance": 2.39,
    "relax_stiffness": 10.0,
    "relax_max_outer_iterations": 3,
    "msa_mode": "mmseqs2_uniref_env",
    "model_type": "alphafold2_multimer_v3",
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
    "rank_by": "multimer",
    "max_seq": 508,
    "max_extra_seq": 2048,
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

## file-8cfe9abd6d0e

원래 경로: `colabfold/results/msa_complex_20261027_173649_931405/msa/log.txt` · SHA256: `0cb3a6327497d1cc90edfbe1ab97cec7907f83cef8a75f357826965899d15cd5`

<details>
<summary>log.txt — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_complex_20261027_173649_931405/msa/log.txt", "sha256": "0cb3a6327497d1cc90edfbe1ab97cec7907f83cef8a75f357826965899d15cd5", "encoding": "utf-8", "added_final_newline": false} -->
```txt
2026-10-27 08:36:50,592 Running colabfold 1.6.2
2026-10-27 08:36:50,651 WARNING: no GPU detected, will be using CPU
2026-10-27 08:36:50,704 Found 4 citations for tools or databases
2026-10-27 08:36:50,704 Query 1/1: 3G5O_A_3G5O_B (length 180)
2026-10-27 08:36:50,763 /cache is not a writable directory
2026-10-27 08:36:50,763 Matplotlib created a temporary cache directory at /tmp/matplotlib-gfyu3_cq because there was an issue with the default path ({configdir}); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
2026-10-27 08:36:50,836 generated new fontManager
2026-10-27 08:37:09,298 Saved /work/results/msa_complex_20261027_173649_931405/msa/3G5O_A_3G5O_B.pickle
2026-10-27 08:37:09,796 Done
```
<!-- END_RESTORE_FILE -->


</details>
