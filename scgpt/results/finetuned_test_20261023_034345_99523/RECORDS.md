# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-47d3efcf1621

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/classification_report.json` · SHA256: `20296547027fa585a67015aaf4b01205bd03ba395f9bc33a0fb20430bc9de177`

<details>
<summary>classification_report.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/classification_report.json", "sha256": "20296547027fa585a67015aaf4b01205bd03ba395f9bc33a0fb20430bc9de177", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "PP": {
    "precision": 1.0,
    "recall": 0.49645390070921985,
    "f1-score": 0.6635071090047393,
    "support": 282.0
  },
  "PSC": {
    "precision": 0.9861111111111112,
    "recall": 0.9726027397260274,
    "f1-score": 0.9793103448275862,
    "support": 73.0
  },
  "acinar": {
    "precision": 1.0,
    "recall": 0.7081339712918661,
    "f1-score": 0.8291316526610645,
    "support": 209.0
  },
  "alpha": {
    "precision": 0.938288920056101,
    "recall": 0.9980109398309299,
    "f1-score": 0.9672289156626506,
    "support": 2011.0
  },
  "beta": {
    "precision": 0.9727095516569201,
    "recall": 0.9920477137176938,
    "f1-score": 0.9822834645669292,
    "support": 1006.0
  },
  "delta": {
    "precision": 0.8655913978494624,
    "recall": 0.8563829787234043,
    "f1-score": 0.8609625668449198,
    "support": 188.0
  },
  "ductal": {
    "precision": 0.8707627118644068,
    "recall": 0.9927536231884058,
    "f1-score": 0.927765237020316,
    "support": 414.0
  },
  "endothelial": {
    "precision": 0.5925925925925926,
    "recall": 1.0,
    "f1-score": 0.7441860465116279,
    "support": 16.0
  },
  "epsilon": {
    "precision": 0.0,
    "recall": 0.0,
    "f1-score": 0.0,
    "support": 7.0
  },
  "macrophage": {
    "precision": 0.0,
    "recall": 0.0,
    "f1-score": 0.0,
    "support": 0.0
  },
  "mast": {
    "precision": 0.0,
    "recall": 0.0,
    "f1-score": 0.0,
    "support": 7.0
  },
  "schwann": {
    "precision": 0.0,
    "recall": 0.0,
    "f1-score": 0.0,
    "support": 0.0
  },
  "t_cell": {
    "precision": 0.0,
    "recall": 0.0,
    "f1-score": 0.0,
    "support": 0.0
  },
  "accuracy": 0.9380488962734393,
  "macro avg": {
    "precision": 0.5558504834715842,
    "recall": 0.5397219897836575,
    "f1-score": 0.5349519490076795,
    "support": 4213.0
  },
  "weighted avg": {
    "precision": 0.9402182210538421,
    "recall": 0.9380488962734393,
    "f1-score": 0.9311712408040055,
    "support": 4213.0
  }
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-23c9b1412178

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/confusion_matrix.csv` · SHA256: `ef5429cee7c86c6a6e618fbb1be2cb37bfb4126fb818f85b327745f4520ed5b8`

<details>
<summary>confusion_matrix.csv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/confusion_matrix.csv", "sha256": "ef5429cee7c86c6a6e618fbb1be2cb37bfb4126fb818f85b327745f4520ed5b8", "encoding": "utf-8", "added_final_newline": false} -->
```csv
true_label,PP,PSC,acinar,alpha,beta,delta,ductal,endothelial,epsilon,macrophage,mast,schwann,t_cell
PP,140,0,0,118,1,23,0,0,0,0,0,0,0
PSC,0,71,0,0,0,0,0,2,0,0,0,0,0
acinar,0,0,148,1,1,0,59,0,0,0,0,0,0
alpha,0,1,0,2007,0,0,0,3,0,0,0,0,0
beta,0,0,0,3,998,2,0,3,0,0,0,0,0
delta,0,0,0,2,21,161,2,2,0,0,0,0,0
ductal,0,0,0,1,1,0,411,1,0,0,0,0,0
endothelial,0,0,0,0,0,0,0,16,0,0,0,0,0
epsilon,0,0,0,7,0,0,0,0,0,0,0,0,0
macrophage,0,0,0,0,0,0,0,0,0,0,0,0,0
mast,0,0,0,0,4,0,0,0,0,3,0,0,0
schwann,0,0,0,0,0,0,0,0,0,0,0,0,0
t_cell,0,0,0,0,0,0,0,0,0,0,0,0,0
```
<!-- END_RESTORE_FILE -->


</details>

## file-10756eed1b51

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/environment.json` · SHA256: `e39bbbb96c8d97c80b93a592bd0af3b04baee3030ab6eaed8f4eda073d68fe2d`

<details>
<summary>environment.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/environment.json", "sha256": "e39bbbb96c8d97c80b93a592bd0af3b04baee3030ab6eaed8f4eda073d68fe2d", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "started_at_utc": "2026-10-22T18:43:47.695933+00:00",
  "python": "3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]",
  "architecture": "aarch64",
  "packages": {
    "torch": "2.13.0a0+8145d630e8.nv26.6.54250401",
    "scgpt": "0.2.5",
    "numpy": "2.1.0",
    "anndata": "0.12.19",
    "scanpy": "1.12.4",
    "scipy": "1.17.1",
    "pandas": "2.3.3",
    "scikit-learn": "1.9.1",
    "psutil": "7.2.2"
  },
  "scgpt_install_source": {
    "url": "https://github.com/bowang-lab/scGPT.git",
    "vcs_info": {
      "commit_id": "cebd6fae655b9c585a4807daa3ac31bb764f06b4",
      "requested_revision": "cebd6fae655b9c585a4807daa3ac31bb764f06b4",
      "vcs": "git"
    }
  },
  "cuda_version": "13.3",
  "gpu_name": "NVIDIA GB10",
  "gpu_compute_capability": [
    12,
    1
  ],
  "cuda_reported_total_memory_gib": 121.68751525878906,
  "system_total_memory_gib": 121.68751525878906,
  "torch_cpu_threads": 8
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-8ba00634d3bd

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/input_cells.tsv` · SHA256: `fcba1693abc4f3aec88b7a0c296a1cdc8985141efdc444f4809cffcca8b60662`

<details>
<summary>input_cells.tsv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/input_cells.tsv", "sha256": "fcba1693abc4f3aec88b7a0c296a1cdc8985141efdc444f4809cffcca8b60662", "encoding": "utf-8", "added_final_newline": false} -->
```tsv
cell_id	original_label	known_training_label
10th_C11_S96	beta	True
10th_C13_S61	beta	True
10th_C14_S53	beta	True
10th_C17_S97	beta	True
10th_C19_S62	PSC	True
10th_C1_S59	beta	True
10th_C20_S54	beta	True
10th_C23_S98	beta	True
10th_C24_S90	ductal	True
10th_C28_S91	alpha	True
10th_C30_S107	alpha	True
10th_C31_S48	alpha	True
10th_C32_S56	beta	True
10th_C33_S64	beta	True
10th_C34_S92	alpha	True
10th_C37_S49	beta	True
10th_C41_S101	alpha	True
10th_C42_S109	acinar	True
10th_C43_S50	alpha	True
10th_C45_S66	alpha	True
10th_C4_S103	ductal	True
10th_C50_S73	alpha	True
10th_C51_S80	beta	True
10th_C52_S110	alpha	True
10th_C53_S118	alpha	True
10th_C54_S124	ductal	True
10th_C56_S74	PSC	True
10th_C57_S81	alpha	True
10th_C5_S95	alpha	True
10th_C61_S68	beta	True
10th_C63_S82	acinar	True
10th_C64_S112	beta	True
10th_C66_S126	beta	True
10th_C74_S76	alpha	True
10th_C76_S128	delta	True
10th_C77_S120	PSC	True
10th_C7_S60	ductal	True
10th_C81_S70	beta	True
10th_C84_S115	beta	True
10th_C85_S86	beta	True
10th_C86_S78	acinar	True
10th_C87_S71	acinar	True
10th_C88_S130	ductal	True
10th_C90_S116	alpha	True
10th_C95_S123	alpha	True
10th_C96_S117	alpha	True
11th-C12_S47	alpha	True
11th-C16_S64	PSC	True
11th-C21_S4	beta	True
11th-C22_S65	beta	True
11th-C24_S49	ductal	True
11th-C26_S12	ductal	True
11th-C27_S19	beta	True
11th-C2_S8	PSC	True
11th-C30_S66	beta	True
11th-C33_S20	delta	True
11th-C36_S67	alpha	True
11th-C37_S6	alpha	True
11th-C40_S52	PSC	True
11th-C41_S60	beta	True
11th-C43_S7	ductal	True
11th-C44_S15	beta	True
11th-C49_S23	beta	True
11th-C4_S62	delta	True
11th-C51_S39	beta	True
11th-C53_S78	beta	True
11th-C55_S24	ductal	True
11th-C56_S32	PSC	True
11th-C57_S40	alpha	True
11th-C58_S71	ductal	True
11th-C5_S54	alpha	True
11th-C61_S25	alpha	True
11th-C62_S33	alpha	True
11th-C64_S72	acinar	True
11th-C66_S88	beta	True
11th-C67_S26	alpha	True
11th-C68_S34	delta	True
11th-C77_S82	delta	True
11th-C7_S16	ductal	True
11th-C93_S30	alpha	True
12th-C10_S155	beta	True
12th-C11_S147	alpha	True
12th-C12_S139	beta	True
12th-C14_S190	beta	True
12th-C15_S183	beta	True
12th-C1_S195	alpha	True
12th-C21_S184	beta	True
12th-C23_S149	alpha	True
12th-C24_S141	beta	True
12th-C25_S185	beta	True
12th-C26_S192	beta	True
12th-C28_S142	alpha	True
12th-C29_S150	alpha	True
12th-C30_S157	beta	True
12th-C33_S200	beta	True
12th-C35_S151	beta	True
12th-C36_S158	PSC	True
12th-C37_S187	alpha	True
12th-C38_S194	beta	True
12th-C39_S201	alpha	True
12th-C3_S181	beta	True
12th-C40_S144	beta	True
12th-C42_S159	beta	True
12th-C46_S145	PSC	True
12th-C47_S153	beta	True
12th-C48_S160	beta	True
12th-C4_S154	beta	True
12th-C50_S210	alpha	True
12th-C51_S217	beta	True
12th-C53_S168	acinar	True
12th-C55_S203	beta	True
12th-C56_S211	ductal	True
12th-C57_S218	alpha	True
12th-C58_S162	alpha	True
12th-C59_S169	beta	True
12th-C60_S176	beta	True
12th-C63_S219	alpha	True
12th-C64_S163	beta	True
12th-C65_S170	beta	True
12th-C67_S205	beta	True
12th-C71_S171	alpha	True
12th-C74_S214	beta	True
12th-C75_S206	alpha	True
12th-C76_S177	beta	True
12th-C7_S196	beta	True
12th-C80_S215	alpha	True
12th-C81_S207	alpha	True
12th-C84_S165	beta	True
12th-C85_S137	beta	True
12th-C8_S189	beta	True
12th-C90_S166	beta	True
12th-C93_S209	beta	True
12th-C94_S180	PSC	True
12th-C95_S174	alpha	True
12th-C96_S167	beta	True
13th_C12_S178	beta	True
13th_C13_S149	alpha	True
13th_C15_S134	beta	True
13th_C16_S193	beta	True
13th_C17_S186	beta	True
13th_C18_S179	alpha	True
13th_C1_S147	alpha	True
13th_C20_S142	alpha	True
13th_C21_S135	beta	True
13th_C23_S187	alpha	True
13th_C27_S151	acinar	True
13th_C29_S188	alpha	True
13th_C31_S137	alpha	True
13th_C34_S181	beta	True
13th_C36_S196	alpha	True
13th_C38_S145	beta	True
13th_C39_S153	beta	True
13th_C3_S132	alpha	True
13th_C46_S183	beta	True
13th_C49_S155	ductal	True
13th_C4_S191	alpha	True
13th_C51_S170	delta	True
13th_C52_S198	beta	True
13th_C53_S205	alpha	True
13th_C54_S212	alpha	True
13th_C55_S156	alpha	True
13th_C57_S171	beta	True
13th_C5_S184	beta	True
13th_C60_S213	alpha	True
13th_C63_S172	beta	True
13th_C66_S214	alpha	True
13th_C68_S165	beta	True
13th_C69_S173	alpha	True
13th_C6_S177	alpha	True
13th_C70_S201	beta	True
13th_C71_S208	beta	True
13th_C72_S215	beta	True
13th_C75_S159	beta	True
13th_C76_S216	alpha	True
13th_C77_S209	alpha	True
13th_C79_S174	alpha	True
13th_C80_S167	beta	True
13th_C81_S160	beta	True
13th_C86_S168	alpha	True
13th_C88_S218	alpha	True
13th_C89_S211	beta	True
13th_C8_S140	ductal	True
13th_C92_S169	alpha	True
13th_C93_S162	alpha	True
13th_C94_S219	beta	True
13th_C96_S204	beta	True
1st-61_S27	beta	True
1st-C11_S58	beta	True
1st-C13_S19	beta	True
1st-C15_S3	alpha	True
1st-C18_S51	alpha	True
1st-C19_S20	alpha	True
1st-C20_S12	alpha	True
1st-C21_S4	alpha	True
1st-C32_S14	beta	True
1st-C34_S54	PSC	True
1st-C39_S23	beta	True
1st-C42_S71	PSC	True
1st-C50_S33	alpha	True
1st-C51_S41	alpha	True
1st-C53_S74	alpha	True
1st-C56_S34	alpha	True
1st-C57_S42	alpha	True
1st-C58_S76	beta	True
1st-C59_S77	alpha	True
1st-C62_S35	alpha	True
1st-C64_S79	beta	True
1st-C68_S36	alpha	True
1st-C69_S44	delta	True
1st-C71_S83	alpha	True
1st-C73_S45	alpha	True
1st-C74_S37	alpha	True
1st-C75_S29	alpha	True
1st-C77_S86	beta	True
1st-C78_S85	alpha	True
1st-C79_S46	beta	True
1st-C80_S38	alpha	True
1st-C8_S10	alpha	True
1st-C90_S91	beta	True
1st-C96_S94	beta	True
2nd-C11_S20	alpha	True
2nd-C15_S25	beta	True
2nd-C16_S33	alpha	True
2nd-C17_S32	beta	True
2nd-C19_S39	beta	True
2nd-C1_S3	alpha	True
2nd-C21_S37	beta	True
2nd-C25_S49	delta	True
2nd-C26_S50	alpha	True
2nd-C27_S51	beta	True
2nd-C28_S55	alpha	True
2nd-C29_S56	alpha	True
2nd-C2_S2	PSC	True
2nd-C31_S61	alpha	True
2nd-C32_S62	beta	True
2nd-C36_S69	alpha	True
2nd-C37_S73	beta	True
2nd-C38_S74	beta	True
2nd-C39_S75	alpha	True
2nd-C3_S1	alpha	True
2nd-C47_S92	alpha	True
2nd-C49_S4	beta	True
2nd-C4_S9	alpha	True
2nd-C51_S6	beta	True
2nd-C52_S10	beta	True
2nd-C54_S12	PSC	True
2nd-C57_S18	alpha	True
2nd-C58_S22	alpha	True
2nd-C5_S8	beta	True
2nd-C62_S29	beta	True
2nd-C67_S40	alpha	True
2nd-C69_S42	alpha	True
2nd-C6_S7	alpha	True
2nd-C72_S48	beta	True
2nd-C74_S53	alpha	True
2nd-C75_S52	alpha	True
2nd-C76_S60	beta	True
2nd-C77_S59	alpha	True
2nd-C7_S15	beta	True
2nd-C80_S65	alpha	True
2nd-C81_S64	beta	True
2nd-C83_S71	alpha	True
2nd-C86_S77	beta	True
2nd-C87_S76	beta	True
2nd-C88_S84	beta	True
2nd-C8_S14	alpha	True
2nd-C93_S88	beta	True
2nd-C94_S96	alpha	True
2nd-C95_S95	beta	True
3rd-C11_S58	beta	True
3rd-C12_S68	alpha	True
3rd-C18_S80	alpha	True
3rd-C1_S39	delta	True
3rd-C20_S22	ductal	True
3rd-C21_S11	alpha	True
3rd-C23_S94	alpha	True
3rd-C25_S15	beta	True
3rd-C26_S28	alpha	True
3rd-C27_S49	alpha	True
3rd-C32_S31	alpha	True
3rd-C33_S50	beta	True
3rd-C35_S23	PSC	True
3rd-C36_S2	beta	True
3rd-C3_S2	alpha	True
3rd-C42_S74	PSC	True
3rd-C46_S93	alpha	True
3rd-C49_S60	alpha	True
3rd-C4_S25	beta	True
3rd-C51_S87	PSC	True
3rd-C52_S86	alpha	True
3rd-C56_S90	beta	True
3rd-C57_S5	alpha	True
3rd-C58_S3	beta	True
3rd-C59_S91	beta	True
3rd-C60_S10	beta	True
3rd-C61_S64	alpha	True
3rd-C63_S54	alpha	True
3rd-C65_S20	beta	True
3rd-C67_S69	alpha	True
3rd-C68_S13	delta	True
3rd-C69_S78	alpha	True
3rd-C6_S44	delta	True
3rd-C71_S80	beta	True
3rd-C76_S22	alpha	True
3rd-C79_S67	alpha	True
3rd-C7_S40	alpha	True
3rd-C82_S11	beta	True
3rd-C84_S75	beta	True
3rd-C85_S79	alpha	True
3rd-C85_S83	alpha	True
3rd-C86_S85	delta	True
3rd-C87_S88	beta	True
3rd-C88_S24	beta	True
3rd-C91_S91	alpha	True
3rd-C92_S86	beta	True
3rd-C95_S81	alpha	True
3rd-C96_S79	beta	True
3rd-C9_S5	alpha	True
4th-61_S28	beta	True
4th-C12_S19	alpha	True
4th-C14_S26	beta	True
4th-C18_S31	alpha	True
4th-C22_S45	alpha	True
4th-C23_S44	alpha	True
4th-C24_S43	ductal	True
4th-C26_S50	beta	True
4th-C27_S51	acinar	True
4th-C28_S55	beta	True
4th-C29_S56	alpha	True
4th-C32_S62	delta	True
4th-C33_S63	alpha	True
4th-C35_S68	alpha	True
4th-C36_S69	alpha	True
4th-C37_S73	alpha	True
4th-C3_S1	alpha	True
4th-C40_S79	alpha	True
4th-C45_S87	beta	True
4th-C49_S4	beta	True
4th-C4_S9	alpha	True
4th-C50_S5	alpha	True
4th-C51_S6	alpha	True
4th-C56_S17	alpha	True
4th-C57_S18	alpha	True
4th-C59_S23	alpha	True
4th-C5_S8	beta	True
4th-C62_S29	alpha	True
4th-C63_S30	alpha	True
4th-C64_S34	alpha	True
4th-C66_S36	alpha	True
4th-C67_S40	beta	True
4th-C68_S41	alpha	True
4th-C69_S42	alpha	True
4th-C73_S54	alpha	True
4th-C75_S52	alpha	True
4th-C78_S58	alpha	True
4th-C79_S66	alpha	True
4th-C7_S15	beta	True
4th-C80_S65	alpha	True
4th-C81_S64	ductal	True
4th-C82_S72	alpha	True
4th-C83_S71	alpha	True
4th-C85_S78	delta	True
4th-C86_S77	alpha	True
4th-C87_S76	delta	True
4th-C88_S84	alpha	True
4th-C89_S83	beta	True
4th-C8_S14	beta	True
4th-C92_S89	alpha	True
4th-C93_S88	beta	True
4th-C94_S96	alpha	True
4th-C95_S95	acinar	True
4th-C96_S94	alpha	True
4th-C9_S13	alpha	True
5th-C10_S92	acinar	True
5th-C11_S91	ductal	True
5th-C12_S90	acinar	True
5th-C14_S97	beta	True
5th-C16_S98	beta	True
5th-C18_S96	beta	True
5th-C19_S104	acinar	True
5th-C21_S102	alpha	True
5th-C25_S108	beta	True
5th-C30_S105	ductal	True
5th-C34_S108	alpha	True
5th-C35_S109	delta	True
5th-C40_S112	acinar	True
5th-C41_S113	ductal	True
5th-C42_S114	acinar	True
5th-C43_S121	ductal	True
5th-C45_S123	acinar	True
5th-C47_S118	delta	True
5th-C49_S89	beta	True
5th-C50_S90	acinar	True
5th-C54_S89	acinar	True
5th-C56_S95	acinar	True
5th-C58_S93	acinar	True
5th-C59_S94	alpha	True
5th-C5_S85	beta	True
5th-C64_S99	beta	True
5th-C66_S101	alpha	True
5th-C67_S105	acinar	True
5th-C69_S107	alpha	True
5th-C6_S84	alpha	True
5th-C70_S103	alpha	True
5th-C73_S111	acinar	True
5th-C74_S110	alpha	True
5th-C7_S93	beta	True
5th-C80_S116	beta	True
5th-C81_S115	acinar	True
5th-C83_S111	delta	True
5th-C87_S119	alpha	True
5th-C88_S117	acinar	True
5th-C90_S115	beta	True
5th-C92_S124	alpha	True
6th-C12_S18	beta	True
6th-C15_S22	beta	True
6th-C16_S29	beta	True
6th-C19_S35	beta	True
6th-C1_S3	beta	True
6th-C20_S34	beta	True
6th-C21_S33	alpha	True
6th-C22_S39	beta	True
6th-C23_S38	beta	True
6th-C24_S37	delta	True
6th-C25_S43	beta	True
6th-C28_S49	beta	True
6th-C29_S50	beta	True
6th-C2_S2	beta	True
6th-C30_S51	beta	True
6th-C34_S61	alpha	True
6th-C40_S73	beta	True
6th-C42_S75	beta	True
6th-C45_S81	alpha	True
6th-C51_S6	beta	True
6th-C54_S11	beta	True
6th-C57_S17	beta	True
6th-C58_S20	beta	True
6th-C59_S21	beta	True
6th-C5_S8	alpha	True
6th-C62_S25	beta	True
6th-C63_S26	beta	True
6th-C64_S30	beta	True
6th-C65_S31	alpha	True
6th-C66_S32	beta	True
6th-C67_S36	beta	True
6th-C71_S41	beta	True
6th-C73_S48	beta	True
6th-C75_S46	beta	True
6th-C77_S53	beta	True
6th-C79_S60	beta	True
6th-C80_S59	beta	True
6th-C82_S66	beta	True
6th-C83_S65	alpha	True
6th-C84_S64	beta	True
6th-C85_S72	beta	True
6th-C87_S70	alpha	True
6th-C89_S77	beta	True
6th-C90_S76	beta	True
6th-C91_S84	beta	True
6th-C93_S82	beta	True
6th-C9_S12	alpha	True
7th-C10_S19	beta	True
7th-C11_S18	alpha	True
7th-C12_S17	beta	True
7th-C14_S24	alpha	True
7th-C15_S23	delta	True
7th-C19_S36	beta	True
7th-C1_S3	beta	True
7th-C20_S35	beta	True
7th-C21_S34	beta	True
7th-C25_S43	beta	True
7th-C28_S48	beta	True
7th-C2_S2	beta	True
7th-C31_S53	beta	True
7th-C32_S54	alpha	True
7th-C34_S59	alpha	True
7th-C36_S61	alpha	True
7th-C3_S1	beta	True
7th-C44_S76	acinar	True
7th-C45_S77	beta	True
7th-C47_S81	ductal	True
7th-C49_S4	alpha	True
7th-C4_S8	beta	True
7th-C50_S5	beta	True
7th-C52_S9	beta	True
7th-C53_S10	beta	True
7th-C56_S15	delta	True
7th-C57_S16	beta	True
7th-C58_S20	alpha	True
7th-C59_S21	beta	True
7th-C60_S22	beta	True
7th-C62_S27	beta	True
7th-C64_S31	beta	True
7th-C65_S32	beta	True
7th-C66_S33	beta	True
7th-C68_S38	beta	True
7th-C69_S39	alpha	True
7th-C72_S42	delta	True
7th-C73_S47	beta	True
7th-C76_S52	ductal	True
7th-C77_S51	beta	True
7th-C79_S58	alpha	True
7th-C80_S57	beta	True
7th-C81_S56	ductal	True
7th-C84_S62	beta	True
7th-C85_S69	alpha	True
7th-C89_S74	beta	True
7th-C8_S13	beta	True
7th-C90_S73	beta	True
7th-C94_S83	beta	True
7th-C96_S82	beta	True
8th-C10_S100	beta	True
8th-C11_S99	alpha	True
8th-C18_S110	beta	True
8th-C20_S116	alpha	True
8th-C21_S115	alpha	True
8th-C23_S121	alpha	True
8th-C24_S120	beta	True
8th-C26_S126	alpha	True
8th-C28_S131	alpha	True
8th-C2_S85	beta	True
8th-C34_S141	alpha	True
8th-C37_S146	alpha	True
8th-C38_S147	beta	True
8th-C44_S155	beta	True
8th-C47_S160	beta	True
8th-C55_S96	alpha	True
8th-C57_S98	alpha	True
8th-C58_S101	alpha	True
8th-C59_S102	alpha	True
8th-C61_S107	beta	True
8th-C64_S112	alpha	True
8th-C65_S113	beta	True
8th-C67_S117	ductal	True
8th-C6_S88	delta	True
8th-C70_S123	beta	True
8th-C73_S130	ductal	True
8th-C74_S129	alpha	True
8th-C75_S128	beta	True
8th-C79_S140	alpha	True
8th-C7_S95	alpha	True
8th-C84_S143	delta	True
8th-C85_S150	alpha	True
8th-C87_S148	beta	True
8th-C90_S152	alpha	True
8th-C94_S162	alpha	True
8th-C95_S161	alpha	True
9th-C10_S18	beta	True
9th-C11_S17	beta	True
9th-C13_S24	beta	True
9th-C14_S23	beta	True
9th-C15_S22	alpha	True
9th-C16_S30	beta	True
9th-C17_S29	alpha	True
9th-C19_S36	alpha	True
9th-C1_S3	beta	True
9th-C20_S35	alpha	True
9th-C22_S41	PSC	True
9th-C25_S45	beta	True
9th-C28_S50	delta	True
9th-C29_S51	alpha	True
9th-C2_S2	PSC	True
9th-C30_S52	alpha	True
9th-C38_S64	beta	True
9th-C3_S1	beta	True
9th-C41_S70	beta	True
9th-C46_S79	beta	True
9th-C49_S4	alpha	True
9th-C51_S6	beta	True
9th-C52_S10	alpha	True
9th-C57_S16	alpha	True
9th-C59_S20	beta	True
9th-C5_S8	alpha	True
9th-C61_S25	ductal	True
9th-C62_S26	alpha	True
9th-C64_S31	alpha	True
9th-C65_S32	beta	True
9th-C67_S37	alpha	True
9th-C69_S39	beta	True
9th-C6_S7	beta	True
9th-C70_S42	beta	True
9th-C74_S48	alpha	True
9th-C75_S47	alpha	True
9th-C77_S53	alpha	True
9th-C80_S56	alpha	True
9th-C81_S55	beta	True
9th-C83_S61	alpha	True
9th-C84_S60	ductal	True
9th-C85_S68	alpha	True
9th-C86_S67	beta	True
9th-C87_S66	alpha	True
9th-C88_S73	beta	True
9th-C89_S72	beta	True
9th-C8_S14	beta	True
9th-C91_S78	beta	True
9th-C92_S77	alpha	True
9th-C93_S76	beta	True
9th-C94_S83	beta	True
9th-C9_S13	alpha	True
AZ-A10	delta	True
AZ-A11	alpha	True
AZ-A12	delta	True
AZ-A2	PP	True
AZ-A5	ductal	True
AZ-A6	alpha	True
AZ-A7	acinar	True
AZ-A8	beta	True
AZ-B1	delta	True
AZ-B10	delta	True
AZ-B12	alpha	True
AZ-B2	alpha	True
AZ-B3	alpha	True
AZ-B5	alpha	True
AZ-B8	acinar	True
AZ-B9	alpha	True
AZ-C1	alpha	True
AZ-C10	beta	True
AZ-C11	delta	True
AZ-C2	alpha	True
AZ-C3	ductal	True
AZ-C4	beta	True
AZ-C5	alpha	True
AZ-C8	alpha	True
AZ-C9	ductal	True
AZ-D12	PP	True
AZ-D2	beta	True
AZ-D3	alpha	True
AZ-D4	alpha	True
AZ-D5	delta	True
AZ-D6	beta	True
AZ-D7	alpha	True
AZ-D8	PP	True
AZ-D9	alpha	True
AZ-E12	alpha	True
AZ-E2	alpha	True
AZ-E3	MHC class II	False
AZ-E4	alpha	True
AZ-E5	alpha	True
AZ-E7	beta	True
AZ-E8	alpha	True
AZ-E9	alpha	True
AZ-F1	beta	True
AZ-F2	beta	True
AZ-F4	alpha	True
AZ-F5	PSC	True
AZ-F6	acinar	True
AZ-F8	alpha	True
AZ-F9	ductal	True
AZ-G1	alpha	True
AZ-G10	alpha	True
AZ-G2	beta	True
AZ-G3	alpha	True
AZ-G4	alpha	True
AZ-G5	PP	True
AZ-H1	delta	True
AZ-H10	acinar	True
AZ-H11	beta	True
AZ-H12	PP	True
AZ-H2	beta	True
AZ-H3	PP	True
AZ-H4	endothelial	True
AZ-H5	PP	True
AZ-H6	beta	True
AZ-H9	alpha	True
HP1502401-A10	beta	True
HP1502401-A12	alpha	True
HP1502401-A14	acinar	True
HP1502401-A15	alpha	True
HP1502401-A17	alpha	True
HP1502401-A18	delta	True
HP1502401-A2	delta	True
HP1502401-A22	acinar	True
HP1502401-A24	delta	True
HP1502401-A4	alpha	True
HP1502401-A6	beta	True
HP1502401-A7	beta	True
HP1502401-B1	beta	True
HP1502401-B10	alpha	True
HP1502401-B11	PP	True
HP1502401-B14	beta	True
HP1502401-B15	PP	True
HP1502401-B16	alpha	True
HP1502401-B17	acinar	True
HP1502401-B18	PP	True
HP1502401-B20	alpha	True
HP1502401-B21	beta	True
HP1502401-B23	beta	True
HP1502401-B24	beta	True
HP1502401-B5	alpha	True
HP1502401-B7	beta	True
HP1502401-B8	delta	True
HP1502401-C1	acinar	True
HP1502401-C10	beta	True
HP1502401-C12	ductal	True
HP1502401-C14	alpha	True
HP1502401-C16	alpha	True
HP1502401-C18	beta	True
HP1502401-C20	delta	True
HP1502401-C21	beta	True
HP1502401-C4	alpha	True
HP1502401-C5	delta	True
HP1502401-C6	beta	True
HP1502401-C7	beta	True
HP1502401-C8	delta	True
HP1502401-D10	acinar	True
HP1502401-D12	delta	True
HP1502401-D13	beta	True
HP1502401-D16	alpha	True
HP1502401-D17	delta	True
HP1502401-D18	ductal	True
HP1502401-D2	acinar	True
HP1502401-D20	alpha	True
HP1502401-D22	beta	True
HP1502401-D24	alpha	True
HP1502401-D3	beta	True
HP1502401-D4	ductal	True
HP1502401-D5	alpha	True
HP1502401-D6	beta	True
HP1502401-D7	beta	True
HP1502401-D8	alpha	True
HP1502401-E10	beta	True
HP1502401-E11	alpha	True
HP1502401-E14	epsilon	True
HP1502401-E16	alpha	True
HP1502401-E17	alpha	True
HP1502401-E19	beta	True
HP1502401-E2	acinar	True
HP1502401-E20	acinar	True
HP1502401-E21	alpha	True
HP1502401-E23	acinar	True
HP1502401-E24	acinar	True
HP1502401-E3	delta	True
HP1502401-E5	PP	True
HP1502401-E6	alpha	True
HP1502401-E7	PP	True
HP1502401-E9	delta	True
HP1502401-F10	alpha	True
HP1502401-F11	delta	True
HP1502401-F12	alpha	True
HP1502401-F13	acinar	True
HP1502401-F15	PP	True
HP1502401-F16	beta	True
HP1502401-F17	PP	True
HP1502401-F18	PP	True
HP1502401-F19	alpha	True
HP1502401-F20	alpha	True
HP1502401-F21	beta	True
HP1502401-F22	beta	True
HP1502401-F23	alpha	True
HP1502401-F24	beta	True
HP1502401-F4	alpha	True
HP1502401-F9	beta	True
HP1502401-G10	alpha	True
HP1502401-G11	PP	True
HP1502401-G12	PP	True
HP1502401-G14	beta	True
HP1502401-G18	acinar	True
HP1502401-G20	delta	True
HP1502401-G22	beta	True
HP1502401-G23	ductal	True
HP1502401-G24	beta	True
HP1502401-G3	delta	True
HP1502401-G4	beta	True
HP1502401-G5	beta	True
HP1502401-G6	beta	True
HP1502401-G7	alpha	True
HP1502401-G8	alpha	True
HP1502401-H1	alpha	True
HP1502401-H11	beta	True
HP1502401-H12	beta	True
HP1502401-H13	PP	True
HP1502401-H14	alpha	True
HP1502401-H15	alpha	True
HP1502401-H16	alpha	True
HP1502401-H18	PSC	True
HP1502401-H19	beta	True
HP1502401-H2	alpha	True
HP1502401-H20	alpha	True
HP1502401-H21	ductal	True
HP1502401-H22	beta	True
HP1502401-H23	ductal	True
HP1502401-H24	alpha	True
HP1502401-H3	alpha	True
HP1502401-H4	alpha	True
HP1502401-H5	alpha	True
HP1502401-H6	delta	True
HP1502401-H7	alpha	True
HP1502401-H8	alpha	True
HP1502401-I1	alpha	True
HP1502401-I12	endothelial	True
HP1502401-I14	acinar	True
HP1502401-I15	alpha	True
HP1502401-I16	alpha	True
HP1502401-I18	alpha	True
HP1502401-I2	ductal	True
HP1502401-I21	ductal	True
HP1502401-I22	alpha	True
HP1502401-I23	alpha	True
HP1502401-I24	beta	True
HP1502401-I4	alpha	True
HP1502401-I6	beta	True
HP1502401-I7	delta	True
HP1502401-I8	PP	True
HP1502401-I9	delta	True
HP1502401-J1	alpha	True
HP1502401-J10	beta	True
HP1502401-J11	alpha	True
HP1502401-J12	beta	True
HP1502401-J14	alpha	True
HP1502401-J15	ductal	True
HP1502401-J16	mast	True
HP1502401-J17	ductal	True
HP1502401-J18	alpha	True
HP1502401-J19	ductal	True
HP1502401-J2	acinar	True
HP1502401-J20	alpha	True
HP1502401-J21	alpha	True
HP1502401-J22	beta	True
HP1502401-J23	alpha	True
HP1502401-J24	ductal	True
HP1502401-J4	alpha	True
HP1502401-J6	alpha	True
HP1502401-J7	mast	True
HP1502401-J8	alpha	True
HP1502401-K1	alpha	True
HP1502401-K10	acinar	True
HP1502401-K11	alpha	True
HP1502401-K12	acinar	True
HP1502401-K14	PP	True
HP1502401-K15	alpha	True
HP1502401-K16	alpha	True
HP1502401-K2	alpha	True
HP1502401-K20	alpha	True
HP1502401-K21	alpha	True
HP1502401-K22	alpha	True
HP1502401-K23	PP	True
HP1502401-K24	PP	True
HP1502401-K3	alpha	True
HP1502401-K5	alpha	True
HP1502401-K6	alpha	True
HP1502401-K9	alpha	True
HP1502401-L1	alpha	True
HP1502401-L10	beta	True
HP1502401-L11	alpha	True
HP1502401-L12	alpha	True
HP1502401-L13	alpha	True
HP1502401-L14	PP	True
HP1502401-L15	alpha	True
HP1502401-L16	alpha	True
HP1502401-L17	alpha	True
HP1502401-L18	ductal	True
HP1502401-L2	alpha	True
HP1502401-L20	alpha	True
HP1502401-L21	alpha	True
HP1502401-L22	alpha	True
HP1502401-L23	alpha	True
HP1502401-L3	alpha	True
HP1502401-L4	alpha	True
HP1502401-L5	ductal	True
HP1502401-L7	alpha	True
HP1502401-L9	alpha	True
HP1502401-M1	alpha	True
HP1502401-M10	alpha	True
HP1502401-M11	alpha	True
HP1502401-M12	beta	True
HP1502401-M14	alpha	True
HP1502401-M15	alpha	True
HP1502401-M16	ductal	True
HP1502401-M17	mast	True
HP1502401-M19	alpha	True
HP1502401-M2	beta	True
HP1502401-M21	alpha	True
HP1502401-M22	acinar	True
HP1502401-M23	alpha	True
HP1502401-M24	alpha	True
HP1502401-M3	alpha	True
HP1502401-M4	alpha	True
HP1502401-M5	alpha	True
HP1502401-M6	delta	True
HP1502401-M7	alpha	True
HP1502401-M8	acinar	True
HP1502401-N1	beta	True
HP1502401-N10	ductal	True
HP1502401-N11	PP	True
HP1502401-N12	alpha	True
HP1502401-N15	alpha	True
HP1502401-N16	ductal	True
HP1502401-N17	alpha	True
HP1502401-N18	ductal	True
HP1502401-N2	acinar	True
HP1502401-N20	alpha	True
HP1502401-N21	beta	True
HP1502401-N23	PP	True
HP1502401-N3	alpha	True
HP1502401-N4	alpha	True
HP1502401-N7	delta	True
HP1502401-N8	beta	True
HP1502401-N9	PP	True
HP1502401-O1	alpha	True
HP1502401-O10	ductal	True
HP1502401-O14	acinar	True
HP1502401-O16	beta	True
HP1502401-O17	alpha	True
HP1502401-O18	delta	True
HP1502401-O2	alpha	True
HP1502401-O20	alpha	True
HP1502401-O21	alpha	True
HP1502401-O22	alpha	True
HP1502401-O23	alpha	True
HP1502401-O24	alpha	True
HP1502401-O3	alpha	True
HP1502401-O4	alpha	True
HP1502401-O6	delta	True
HP1502401-O8	alpha	True
HP1502401-O9	mast	True
HP1504101T2D-A10	acinar	True
HP1504101T2D-A12	alpha	True
HP1504101T2D-A13	PP	True
HP1504101T2D-A15	PP	True
HP1504101T2D-A17	PP	True
HP1504101T2D-A18	PP	True
HP1504101T2D-A19	alpha	True
HP1504101T2D-A20	PP	True
HP1504101T2D-A23	alpha	True
HP1504101T2D-A24	delta	True
HP1504101T2D-A4	PP	True
HP1504101T2D-A5	alpha	True
HP1504101T2D-A6	alpha	True
HP1504101T2D-A8	alpha	True
HP1504101T2D-B10	PP	True
HP1504101T2D-B11	alpha	True
HP1504101T2D-B12	PP	True
HP1504101T2D-B13	alpha	True
HP1504101T2D-B16	alpha	True
HP1504101T2D-B17	PP	True
HP1504101T2D-B18	PP	True
HP1504101T2D-B21	alpha	True
HP1504101T2D-B22	PP	True
HP1504101T2D-B24	PP	True
HP1504101T2D-B4	PP	True
HP1504101T2D-B5	PP	True
HP1504101T2D-B6	beta	True
HP1504101T2D-B7	alpha	True
HP1504101T2D-B8	ductal	True
HP1504101T2D-C1	endothelial	True
HP1504101T2D-C10	delta	True
HP1504101T2D-C11	acinar	True
HP1504101T2D-C12	PP	True
HP1504101T2D-C14	alpha	True
HP1504101T2D-C15	alpha	True
HP1504101T2D-C17	alpha	True
HP1504101T2D-C18	beta	True
HP1504101T2D-C20	alpha	True
HP1504101T2D-C21	PP	True
HP1504101T2D-C22	alpha	True
HP1504101T2D-C4	alpha	True
HP1504101T2D-C9	delta	True
HP1504101T2D-D1	alpha	True
HP1504101T2D-D11	PP	True
HP1504101T2D-D12	alpha	True
HP1504101T2D-D13	alpha	True
HP1504101T2D-D15	PP	True
HP1504101T2D-D17	PP	True
HP1504101T2D-D18	PP	True
HP1504101T2D-D19	alpha	True
HP1504101T2D-D2	alpha	True
HP1504101T2D-D21	PP	True
HP1504101T2D-D23	PP	True
HP1504101T2D-D24	PP	True
HP1504101T2D-D5	alpha	True
HP1504101T2D-D6	alpha	True
HP1504101T2D-D7	alpha	True
HP1504101T2D-D9	beta	True
HP1504101T2D-E1	alpha	True
HP1504101T2D-E10	alpha	True
HP1504101T2D-E11	PP	True
HP1504101T2D-E14	PP	True
HP1504101T2D-E15	beta	True
HP1504101T2D-E16	beta	True
HP1504101T2D-E18	PP	True
HP1504101T2D-E19	alpha	True
HP1504101T2D-E20	PP	True
HP1504101T2D-E21	alpha	True
HP1504101T2D-E22	alpha	True
HP1504101T2D-E23	alpha	True
HP1504101T2D-E24	beta	True
HP1504101T2D-E3	alpha	True
HP1504101T2D-E4	PP	True
HP1504101T2D-E6	alpha	True
HP1504101T2D-E7	PP	True
HP1504101T2D-E8	PP	True
HP1504101T2D-E9	PP	True
HP1504101T2D-F1	acinar	True
HP1504101T2D-F10	alpha	True
HP1504101T2D-F11	PP	True
HP1504101T2D-F12	alpha	True
HP1504101T2D-F13	PP	True
HP1504101T2D-F14	PP	True
HP1504101T2D-F15	alpha	True
HP1504101T2D-F16	PP	True
HP1504101T2D-F17	delta	True
HP1504101T2D-F18	PSC	True
HP1504101T2D-F19	alpha	True
HP1504101T2D-F2	PP	True
HP1504101T2D-F20	PP	True
HP1504101T2D-F22	alpha	True
HP1504101T2D-F3	alpha	True
HP1504101T2D-F7	alpha	True
HP1504101T2D-G1	alpha	True
HP1504101T2D-G10	alpha	True
HP1504101T2D-G12	alpha	True
HP1504101T2D-G14	alpha	True
HP1504101T2D-G16	PP	True
HP1504101T2D-G17	alpha	True
HP1504101T2D-G19	alpha	True
HP1504101T2D-G2	alpha	True
HP1504101T2D-G21	PP	True
HP1504101T2D-G22	alpha	True
HP1504101T2D-G23	alpha	True
HP1504101T2D-G24	alpha	True
HP1504101T2D-G3	PP	True
HP1504101T2D-G4	PP	True
HP1504101T2D-G8	PP	True
HP1504101T2D-G9	PP	True
HP1504101T2D-H1	PP	True
HP1504101T2D-H14	PP	True
HP1504101T2D-H15	alpha	True
HP1504101T2D-H17	alpha	True
HP1504101T2D-H18	ductal	True
HP1504101T2D-H2	alpha	True
HP1504101T2D-H20	alpha	True
HP1504101T2D-H21	alpha	True
HP1504101T2D-H22	alpha	True
HP1504101T2D-H23	beta	True
HP1504101T2D-H24	PP	True
HP1504101T2D-H4	alpha	True
HP1504101T2D-H5	delta	True
HP1504101T2D-H6	beta	True
HP1504101T2D-H9	alpha	True
HP1504101T2D-I1	alpha	True
HP1504101T2D-I10	PP	True
HP1504101T2D-I14	alpha	True
HP1504101T2D-I15	PP	True
HP1504101T2D-I16	PP	True
HP1504101T2D-I17	alpha	True
HP1504101T2D-I18	alpha	True
HP1504101T2D-I2	alpha	True
HP1504101T2D-I20	alpha	True
HP1504101T2D-I22	alpha	True
HP1504101T2D-I23	alpha	True
HP1504101T2D-I24	alpha	True
HP1504101T2D-I5	alpha	True
HP1504101T2D-I6	acinar	True
HP1504101T2D-I7	PP	True
HP1504101T2D-I8	alpha	True
HP1504101T2D-J1	alpha	True
HP1504101T2D-J11	alpha	True
HP1504101T2D-J13	alpha	True
HP1504101T2D-J14	delta	True
HP1504101T2D-J15	alpha	True
HP1504101T2D-J16	acinar	True
HP1504101T2D-J18	alpha	True
HP1504101T2D-J2	alpha	True
HP1504101T2D-J20	alpha	True
HP1504101T2D-J22	PP	True
HP1504101T2D-J24	alpha	True
HP1504101T2D-J3	PP	True
HP1504101T2D-J5	PP	True
HP1504101T2D-J7	alpha	True
HP1504101T2D-J9	PP	True
HP1504101T2D-K10	acinar	True
HP1504101T2D-K12	alpha	True
HP1504101T2D-K17	alpha	True
HP1504101T2D-K18	alpha	True
HP1504101T2D-K19	acinar	True
HP1504101T2D-K2	PP	True
HP1504101T2D-K20	alpha	True
HP1504101T2D-K22	alpha	True
HP1504101T2D-K23	alpha	True
HP1504101T2D-K24	alpha	True
HP1504101T2D-K4	alpha	True
HP1504101T2D-K8	PP	True
HP1504101T2D-L1	PP	True
HP1504101T2D-L10	alpha	True
HP1504101T2D-L11	delta	True
HP1504101T2D-L13	PP	True
HP1504101T2D-L14	acinar	True
HP1504101T2D-L15	PP	True
HP1504101T2D-L18	alpha	True
HP1504101T2D-L19	alpha	True
HP1504101T2D-L2	alpha	True
HP1504101T2D-L21	alpha	True
HP1504101T2D-L23	alpha	True
HP1504101T2D-L24	PP	True
HP1504101T2D-L3	alpha	True
HP1504101T2D-L4	alpha	True
HP1504101T2D-L5	alpha	True
HP1504101T2D-L7	alpha	True
HP1504101T2D-L8	alpha	True
HP1504101T2D-M1	PP	True
HP1504101T2D-M10	alpha	True
HP1504101T2D-M15	alpha	True
HP1504101T2D-M18	alpha	True
HP1504101T2D-M19	alpha	True
HP1504101T2D-M2	alpha	True
HP1504101T2D-M21	alpha	True
HP1504101T2D-M23	alpha	True
HP1504101T2D-M24	alpha	True
HP1504101T2D-M3	PP	True
HP1504101T2D-M4	alpha	True
HP1504101T2D-M5	alpha	True
HP1504101T2D-M6	alpha	True
HP1504101T2D-M7	alpha	True
HP1504101T2D-M8	epsilon	True
HP1504101T2D-M9	alpha	True
HP1504101T2D-N1	alpha	True
HP1504101T2D-N10	beta	True
HP1504101T2D-N12	alpha	True
HP1504101T2D-N13	alpha	True
HP1504101T2D-N15	alpha	True
HP1504101T2D-N16	PP	True
HP1504101T2D-N17	alpha	True
HP1504101T2D-N19	alpha	True
HP1504101T2D-N2	alpha	True
HP1504101T2D-N20	PP	True
HP1504101T2D-N21	delta	True
HP1504101T2D-N22	alpha	True
HP1504101T2D-N24	alpha	True
HP1504101T2D-N5	alpha	True
HP1504101T2D-N6	alpha	True
HP1504101T2D-N7	alpha	True
HP1504101T2D-N8	alpha	True
HP1504101T2D-N9	alpha	True
HP1504101T2D-O10	alpha	True
HP1504101T2D-O11	PP	True
HP1504101T2D-O12	beta	True
HP1504101T2D-O13	alpha	True
HP1504101T2D-O14	alpha	True
HP1504101T2D-O16	alpha	True
HP1504101T2D-O20	alpha	True
HP1504101T2D-O22	alpha	True
HP1504101T2D-O23	PP	True
HP1504101T2D-O24	ductal	True
HP1504101T2D-O3	alpha	True
HP1504101T2D-O4	PP	True
HP1504101T2D-O6	alpha	True
HP1504101T2D-O9	alpha	True
HP1504101T2D-P11	alpha	True
HP1504101T2D-P12	PP	True
HP1504101T2D-P14	alpha	True
HP1504101T2D-P17	alpha	True
HP1504101T2D-P18	alpha	True
HP1504101T2D-P2	alpha	True
HP1504101T2D-P21	alpha	True
HP1504101T2D-P22	alpha	True
HP1504101T2D-P24	PP	True
HP1504101T2D-P3	PSC	True
HP1504101T2D-P5	delta	True
HP1504101T2D-P7	alpha	True
HP1504101T2D-P9	PP	True
HP1504901-A10	delta	True
HP1504901-A14	alpha	True
HP1504901-A16	beta	True
HP1504901-A19	alpha	True
HP1504901-A2	alpha	True
HP1504901-A21	delta	True
HP1504901-A22	PSC	True
HP1504901-A23	endothelial	True
HP1504901-A3	beta	True
HP1504901-A5	PP	True
HP1504901-A9	alpha	True
HP1504901-B1	alpha	True
HP1504901-B10	beta	True
HP1504901-B11	beta	True
HP1504901-B14	ductal	True
HP1504901-B15	alpha	True
HP1504901-B16	alpha	True
HP1504901-B18	alpha	True
HP1504901-B20	PP	True
HP1504901-B21	alpha	True
HP1504901-B22	beta	True
HP1504901-B23	PP	True
HP1504901-B24	PSC	True
HP1504901-B3	alpha	True
HP1504901-B4	delta	True
HP1504901-B6	PP	True
HP1504901-B8	PP	True
HP1504901-B9	delta	True
HP1504901-C10	delta	True
HP1504901-C11	alpha	True
HP1504901-C13	epsilon	True
HP1504901-C14	PP	True
HP1504901-C16	alpha	True
HP1504901-C17	beta	True
HP1504901-C23	PP	True
HP1504901-D1	alpha	True
HP1504901-D12	alpha	True
HP1504901-D13	acinar	True
HP1504901-D14	alpha	True
HP1504901-D15	alpha	True
HP1504901-D16	alpha	True
HP1504901-D17	alpha	True
HP1504901-D18	epsilon	True
HP1504901-D2	alpha	True
HP1504901-D21	alpha	True
HP1504901-D22	alpha	True
HP1504901-D23	alpha	True
HP1504901-D24	PP	True
HP1504901-D5	alpha	True
HP1504901-D6	delta	True
HP1504901-D7	beta	True
HP1504901-D9	PP	True
HP1504901-E14	alpha	True
HP1504901-E16	beta	True
HP1504901-E18	alpha	True
HP1504901-E2	endothelial	True
HP1504901-E21	beta	True
HP1504901-E23	PP	True
HP1504901-E24	alpha	True
HP1504901-E6	ductal	True
HP1504901-E9	PSC	True
HP1504901-F1	PP	True
HP1504901-F10	alpha	True
HP1504901-F11	beta	True
HP1504901-F12	alpha	True
HP1504901-F13	beta	True
HP1504901-F14	epsilon	True
HP1504901-F15	PP	True
HP1504901-F16	endothelial	True
HP1504901-F17	beta	True
HP1504901-F18	PP	True
HP1504901-F19	alpha	True
HP1504901-F2	delta	True
HP1504901-F20	alpha	True
HP1504901-F23	acinar	True
HP1504901-F24	alpha	True
HP1504901-F3	alpha	True
HP1504901-F4	alpha	True
HP1504901-F5	alpha	True
HP1504901-F8	ductal	True
HP1504901-F9	alpha	True
HP1504901-G10	endothelial	True
HP1504901-G11	ductal	True
HP1504901-G12	beta	True
HP1504901-G13	alpha	True
HP1504901-G15	alpha	True
HP1504901-G18	beta	True
HP1504901-G19	alpha	True
HP1504901-G2	alpha	True
HP1504901-G20	PP	True
HP1504901-G21	beta	True
HP1504901-G24	alpha	True
HP1504901-G3	alpha	True
HP1504901-G4	beta	True
HP1504901-G6	alpha	True
HP1504901-G8	beta	True
HP1504901-G9	alpha	True
HP1504901-H1	alpha	True
HP1504901-H10	alpha	True
HP1504901-H11	alpha	True
HP1504901-H12	alpha	True
HP1504901-H14	PP	True
HP1504901-H16	acinar	True
HP1504901-H17	beta	True
HP1504901-H18	beta	True
HP1504901-H19	PP	True
HP1504901-H2	alpha	True
HP1504901-H22	beta	True
HP1504901-H4	alpha	True
HP1504901-H5	beta	True
HP1504901-H6	alpha	True
HP1504901-I10	ductal	True
HP1504901-I12	alpha	True
HP1504901-I13	endothelial	True
HP1504901-I17	endothelial	True
HP1504901-I18	PP	True
HP1504901-I19	PSC	True
HP1504901-I2	alpha	True
HP1504901-I21	alpha	True
HP1504901-I22	PP	True
HP1504901-I24	alpha	True
HP1504901-I3	alpha	True
HP1504901-I7	delta	True
HP1504901-I8	alpha	True
HP1504901-J12	alpha	True
HP1504901-J17	alpha	True
HP1504901-J18	PSC	True
HP1504901-J20	alpha	True
HP1504901-J22	alpha	True
HP1504901-J23	alpha	True
HP1504901-J24	PSC	True
HP1504901-J3	beta	True
HP1504901-J4	delta	True
HP1504901-J5	ductal	True
HP1504901-J6	alpha	True
HP1504901-J7	beta	True
HP1504901-J8	alpha	True
HP1504901-J9	PP	True
HP1504901-K1	alpha	True
HP1504901-K12	alpha	True
HP1504901-K14	PP	True
HP1504901-K19	beta	True
HP1504901-K20	delta	True
HP1504901-K22	beta	True
HP1504901-K23	ductal	True
HP1504901-K5	alpha	True
HP1504901-K6	alpha	True
HP1504901-L1	alpha	True
HP1504901-L13	alpha	True
HP1504901-L14	alpha	True
HP1504901-L16	ductal	True
HP1504901-L17	beta	True
HP1504901-L18	PP	True
HP1504901-L20	beta	True
HP1504901-L21	alpha	True
HP1504901-L22	beta	True
HP1504901-L3	PP	True
HP1504901-L8	beta	True
HP1504901-L9	PP	True
HP1504901-M11	PSC	True
HP1504901-M12	alpha	True
HP1504901-M18	alpha	True
HP1504901-M19	PSC	True
HP1504901-M21	delta	True
HP1504901-M24	alpha	True
HP1504901-M3	alpha	True
HP1504901-M4	PP	True
HP1504901-N1	alpha	True
HP1504901-N12	alpha	True
HP1504901-N13	beta	True
HP1504901-N15	PP	True
HP1504901-N16	alpha	True
HP1504901-N17	alpha	True
HP1504901-N18	ductal	True
HP1504901-N19	alpha	True
HP1504901-N2	alpha	True
HP1504901-N21	PSC	True
HP1504901-N22	alpha	True
HP1504901-N23	PP	True
HP1504901-N24	delta	True
HP1504901-N7	alpha	True
HP1504901-N8	beta	True
HP1504901-N9	PP	True
HP1504901-O1	PP	True
HP1504901-O10	alpha	True
HP1504901-O11	ductal	True
HP1504901-O14	ductal	True
HP1504901-O16	alpha	True
HP1504901-O17	alpha	True
HP1504901-O18	beta	True
HP1504901-O20	beta	True
HP1504901-O21	PP	True
HP1504901-O24	PP	True
HP1504901-O4	ductal	True
HP1504901-O5	PP	True
HP1504901-O6	alpha	True
HP1504901-O7	alpha	True
HP1504901-O8	beta	True
HP1504901-O9	alpha	True
HP1504901-P10	alpha	True
HP1504901-P11	ductal	True
HP1504901-P14	beta	True
HP1504901-P18	endothelial	True
HP1504901-P20	alpha	True
HP1504901-P21	alpha	True
HP1504901-P3	endothelial	True
HP1504901-P7	ductal	True
HP1504901-P9	PSC	True
HP1506401-A12	ductal	True
HP1506401-A15	beta	True
HP1506401-A17	beta	True
HP1506401-A18	ductal	True
HP1506401-A20	ductal	True
HP1506401-A21	alpha	True
HP1506401-A24	beta	True
HP1506401-A4	ductal	True
HP1506401-A5	alpha	True
HP1506401-A6	acinar	True
HP1506401-A7	acinar	True
HP1506401-A9	acinar	True
HP1506401-B1	acinar	True
HP1506401-B11	beta	True
HP1506401-B14	acinar	True
HP1506401-B15	PSC	True
HP1506401-B16	acinar	True
HP1506401-B17	ductal	True
HP1506401-B20	PSC	True
HP1506401-B22	PP	True
HP1506401-B23	ductal	True
HP1506401-B24	ductal	True
HP1506401-B3	PP	True
HP1506401-B4	PP	True
HP1506401-B5	acinar	True
HP1506401-B6	acinar	True
HP1506401-C13	acinar	True
HP1506401-C15	beta	True
HP1506401-C16	ductal	True
HP1506401-C17	ductal	True
HP1506401-C23	acinar	True
HP1506401-C24	acinar	True
HP1506401-C4	beta	True
HP1506401-C5	alpha	True
HP1506401-C7	acinar	True
HP1506401-C8	beta	True
HP1506401-C9	beta	True
HP1506401-D1	ductal	True
HP1506401-D12	acinar	True
HP1506401-D13	beta	True
HP1506401-D14	acinar	True
HP1506401-D15	ductal	True
HP1506401-D17	ductal	True
HP1506401-D2	ductal	True
HP1506401-D24	acinar	True
HP1506401-D3	acinar	True
HP1506401-D7	ductal	True
HP1506401-D8	ductal	True
HP1506401-D9	PP	True
HP1506401-E11	alpha	True
HP1506401-E12	PP	True
HP1506401-E15	alpha	True
HP1506401-E16	delta	True
HP1506401-E17	ductal	True
HP1506401-E18	acinar	True
HP1506401-E20	acinar	True
HP1506401-E23	acinar	True
HP1506401-E24	acinar	True
HP1506401-E6	ductal	True
HP1506401-E8	acinar	True
HP1506401-F10	ductal	True
HP1506401-F11	beta	True
HP1506401-F12	beta	True
HP1506401-F13	alpha	True
HP1506401-F16	acinar	True
HP1506401-F17	alpha	True
HP1506401-F18	beta	True
HP1506401-F19	ductal	True
HP1506401-F2	acinar	True
HP1506401-F21	ductal	True
HP1506401-F23	ductal	True
HP1506401-F24	beta	True
HP1506401-F3	acinar	True
HP1506401-F5	acinar	True
HP1506401-F8	beta	True
HP1506401-F9	beta	True
HP1506401-G1	alpha	True
HP1506401-G11	delta	True
HP1506401-G12	beta	True
HP1506401-G13	alpha	True
HP1506401-G14	acinar	True
HP1506401-G16	acinar	True
HP1506401-G18	ductal	True
HP1506401-G19	ductal	True
HP1506401-G2	ductal	True
HP1506401-G20	ductal	True
HP1506401-G21	ductal	True
HP1506401-G23	ductal	True
HP1506401-G24	alpha	True
HP1506401-G3	alpha	True
HP1506401-G4	acinar	True
HP1506401-G5	alpha	True
HP1506401-G6	beta	True
HP1506401-G8	PP	True
HP1506401-G9	beta	True
HP1506401-H1	acinar	True
HP1506401-H13	alpha	True
HP1506401-H14	alpha	True
HP1506401-H15	acinar	True
HP1506401-H16	acinar	True
HP1506401-H18	acinar	True
HP1506401-H19	PP	True
HP1506401-H2	ductal	True
HP1506401-H22	alpha	True
HP1506401-H23	ductal	True
HP1506401-H3	acinar	True
HP1506401-H4	ductal	True
HP1506401-H6	ductal	True
HP1506401-H7	ductal	True
HP1506401-H9	ductal	True
HP1506401-I1	PP	True
HP1506401-I10	acinar	True
HP1506401-I11	beta	True
HP1506401-I12	beta	True
HP1506401-I13	acinar	True
HP1506401-I14	PP	True
HP1506401-I15	acinar	True
HP1506401-I17	PP	True
HP1506401-I18	acinar	True
HP1506401-I2	PP	True
HP1506401-I21	ductal	True
HP1506401-I23	beta	True
HP1506401-I24	ductal	True
HP1506401-I4	ductal	True
HP1506401-I6	acinar	True
HP1506401-I8	alpha	True
HP1506401-J11	acinar	True
HP1506401-J14	acinar	True
HP1506401-J15	acinar	True
HP1506401-J16	epsilon	True
HP1506401-J17	beta	True
HP1506401-J19	beta	True
HP1506401-J2	acinar	True
HP1506401-J20	ductal	True
HP1506401-J21	ductal	True
HP1506401-J22	ductal	True
HP1506401-J23	acinar	True
HP1506401-J24	acinar	True
HP1506401-J3	acinar	True
HP1506401-J4	alpha	True
HP1506401-J5	ductal	True
HP1506401-J6	acinar	True
HP1506401-J7	ductal	True
HP1506401-J8	ductal	True
HP1506401-J9	acinar	True
HP1506401-K1	beta	True
HP1506401-K12	alpha	True
HP1506401-K14	acinar	True
HP1506401-K15	acinar	True
HP1506401-K16	ductal	True
HP1506401-K17	ductal	True
HP1506401-K19	ductal	True
HP1506401-K2	acinar	True
HP1506401-K21	beta	True
HP1506401-K22	PP	True
HP1506401-K23	PP	True
HP1506401-K4	ductal	True
HP1506401-K6	acinar	True
HP1506401-L1	ductal	True
HP1506401-L10	ductal	True
HP1506401-L11	acinar	True
HP1506401-L13	ductal	True
HP1506401-L17	PP	True
HP1506401-L18	acinar	True
HP1506401-L19	acinar	True
HP1506401-L20	ductal	True
HP1506401-L21	ductal	True
HP1506401-L22	acinar	True
HP1506401-L23	alpha	True
HP1506401-L3	acinar	True
HP1506401-L4	ductal	True
HP1506401-L5	beta	True
HP1506401-L7	acinar	True
HP1506401-L9	acinar	True
HP1506401-M1	acinar	True
HP1506401-M11	ductal	True
HP1506401-M12	alpha	True
HP1506401-M16	ductal	True
HP1506401-M17	acinar	True
HP1506401-M18	acinar	True
HP1506401-M19	acinar	True
HP1506401-M21	ductal	True
HP1506401-M4	acinar	True
HP1506401-M6	PP	True
HP1506401-M8	acinar	True
HP1506401-N1	acinar	True
HP1506401-N10	ductal	True
HP1506401-N11	beta	True
HP1506401-N15	acinar	True
HP1506401-N16	ductal	True
HP1506401-N17	ductal	True
HP1506401-N19	alpha	True
HP1506401-N20	beta	True
HP1506401-N22	acinar	True
HP1506401-N23	acinar	True
HP1506401-N24	acinar	True
HP1506401-N4	alpha	True
HP1506401-N5	acinar	True
HP1506401-N9	acinar	True
HP1506401-O10	beta	True
HP1506401-O11	beta	True
HP1506401-O12	acinar	True
HP1506401-O16	acinar	True
HP1506401-O18	acinar	True
HP1506401-O2	ductal	True
HP1506401-O21	acinar	True
HP1506401-O23	alpha	True
HP1506401-O24	acinar	True
HP1506401-O3	alpha	True
HP1506401-O4	ductal	True
HP1506401-O7	acinar	True
HP1506401-O8	ductal	True
HP1506401-P1	ductal	True
HP1506401-P10	acinar	True
HP1506401-P11	ductal	True
HP1506401-P12	beta	True
HP1506401-P14	beta	True
HP1506401-P17	acinar	True
HP1506401-P19	alpha	True
HP1506401-P21	ductal	True
HP1506401-P22	ductal	True
HP1506401-P23	acinar	True
HP1506401-P3	alpha	True
HP1506401-P4	ductal	True
HP1506401-P5	ductal	True
HP1507101-A12	beta	True
HP1507101-A14	alpha	True
HP1507101-A20	alpha	True
HP1507101-A22	alpha	True
HP1507101-A4	alpha	True
HP1507101-A5	delta	True
HP1507101-A6	beta	True
HP1507101-B1	alpha	True
HP1507101-B12	alpha	True
HP1507101-B13	alpha	True
HP1507101-B14	alpha	True
HP1507101-B16	alpha	True
HP1507101-B18	alpha	True
HP1507101-B19	alpha	True
HP1507101-B20	PP	True
HP1507101-B23	alpha	True
HP1507101-B24	alpha	True
HP1507101-B3	alpha	True
HP1507101-B4	alpha	True
HP1507101-B5	beta	True
HP1507101-B6	alpha	True
HP1507101-B7	alpha	True
HP1507101-C12	beta	True
HP1507101-C14	alpha	True
HP1507101-C18	alpha	True
HP1507101-C2	alpha	True
HP1507101-C4	alpha	True
HP1507101-C5	alpha	True
HP1507101-C7	beta	True
HP1507101-C8	alpha	True
HP1507101-D1	alpha	True
HP1507101-D10	endothelial	True
HP1507101-D12	alpha	True
HP1507101-D14	alpha	True
HP1507101-D15	alpha	True
HP1507101-D17	alpha	True
HP1507101-D18	alpha	True
HP1507101-D2	alpha	True
HP1507101-D22	beta	True
HP1507101-D23	beta	True
HP1507101-D3	beta	True
HP1507101-D5	alpha	True
HP1507101-D6	beta	True
HP1507101-D8	alpha	True
HP1507101-D9	PSC	True
HP1507101-E10	alpha	True
HP1507101-E12	alpha	True
HP1507101-E14	beta	True
HP1507101-E2	alpha	True
HP1507101-E20	alpha	True
HP1507101-E22	acinar	True
HP1507101-E24	alpha	True
HP1507101-E6	alpha	True
HP1507101-E7	alpha	True
HP1507101-E8	alpha	True
HP1507101-E9	alpha	True
HP1507101-F1	alpha	True
HP1507101-F10	delta	True
HP1507101-F11	beta	True
HP1507101-F12	delta	True
HP1507101-F13	alpha	True
HP1507101-F14	beta	True
HP1507101-F15	ductal	True
HP1507101-F16	alpha	True
HP1507101-F18	alpha	True
HP1507101-F19	alpha	True
HP1507101-F2	alpha	True
HP1507101-F20	alpha	True
HP1507101-F22	alpha	True
HP1507101-F23	beta	True
HP1507101-F3	alpha	True
HP1507101-F4	beta	True
HP1507101-F5	alpha	True
HP1507101-F7	alpha	True
HP1507101-F8	alpha	True
HP1507101-G10	ductal	True
HP1507101-G12	alpha	True
HP1507101-G13	alpha	True
HP1507101-G14	alpha	True
HP1507101-G15	alpha	True
HP1507101-G16	alpha	True
HP1507101-G18	alpha	True
HP1507101-G2	PSC	True
HP1507101-G22	alpha	True
HP1507101-G23	alpha	True
HP1507101-G6	alpha	True
HP1507101-G8	alpha	True
HP1507101-H1	alpha	True
HP1507101-H11	alpha	True
HP1507101-H12	alpha	True
HP1507101-H14	alpha	True
HP1507101-H15	alpha	True
HP1507101-H16	ductal	True
HP1507101-H17	beta	True
HP1507101-H22	alpha	True
HP1507101-H23	alpha	True
HP1507101-H3	alpha	True
HP1507101-H4	beta	True
HP1507101-H5	alpha	True
HP1507101-H6	alpha	True
HP1507101-H7	alpha	True
HP1507101-H9	alpha	True
HP1507101-I11	alpha	True
HP1507101-I12	beta	True
HP1507101-I13	alpha	True
HP1507101-I16	alpha	True
HP1507101-I17	alpha	True
HP1507101-I18	beta	True
HP1507101-I19	alpha	True
HP1507101-I2	alpha	True
HP1507101-I20	alpha	True
HP1507101-I22	alpha	True
HP1507101-I4	alpha	True
HP1507101-I5	alpha	True
HP1507101-I6	alpha	True
HP1507101-I8	alpha	True
HP1507101-J1	alpha	True
HP1507101-J11	alpha	True
HP1507101-J12	PSC	True
HP1507101-J13	alpha	True
HP1507101-J15	ductal	True
HP1507101-J17	alpha	True
HP1507101-J18	alpha	True
HP1507101-J19	beta	True
HP1507101-J2	alpha	True
HP1507101-J20	beta	True
HP1507101-J21	alpha	True
HP1507101-J22	alpha	True
HP1507101-J23	alpha	True
HP1507101-J24	alpha	True
HP1507101-J3	alpha	True
HP1507101-J5	alpha	True
HP1507101-J7	acinar	True
HP1507101-K13	PSC	True
HP1507101-K15	beta	True
HP1507101-K16	ductal	True
HP1507101-K20	PSC	True
HP1507101-K23	beta	True
HP1507101-K3	alpha	True
HP1507101-K4	alpha	True
HP1507101-K8	alpha	True
HP1507101-L1	PSC	True
HP1507101-L11	alpha	True
HP1507101-L14	alpha	True
HP1507101-L16	beta	True
HP1507101-L17	acinar	True
HP1507101-L18	alpha	True
HP1507101-L19	alpha	True
HP1507101-L2	alpha	True
HP1507101-L21	alpha	True
HP1507101-L22	beta	True
HP1507101-L23	delta	True
HP1507101-L24	beta	True
HP1507101-L3	alpha	True
HP1507101-L4	delta	True
HP1507101-L5	alpha	True
HP1507101-L7	alpha	True
HP1507101-M12	alpha	True
HP1507101-M18	alpha	True
HP1507101-M2	alpha	True
HP1507101-M22	PP	True
HP1507101-M4	alpha	True
HP1507101-M8	beta	True
HP1507101-N1	alpha	True
HP1507101-N13	beta	True
HP1507101-N16	alpha	True
HP1507101-N19	alpha	True
HP1507101-N2	alpha	True
HP1507101-N20	alpha	True
HP1507101-N4	alpha	True
HP1507101-N5	beta	True
HP1507101-N6	beta	True
HP1507101-N8	ductal	True
HP1507101-N9	alpha	True
HP1507101-O1	alpha	True
HP1507101-O11	alpha	True
HP1507101-O12	ductal	True
HP1507101-O13	alpha	True
HP1507101-O14	beta	True
HP1507101-O16	alpha	True
HP1507101-O19	beta	True
HP1507101-O20	alpha	True
HP1507101-O22	alpha	True
HP1507101-O23	delta	True
HP1507101-O3	beta	True
HP1507101-O5	alpha	True
HP1507101-O8	beta	True
HP1507101-P1	ductal	True
HP1507101-P10	alpha	True
HP1507101-P12	alpha	True
HP1507101-P2	beta	True
HP1507101-P21	delta	True
HP1507101-P23	alpha	True
HP1507101-P3	alpha	True
HP1507101-P4	alpha	True
HP1507101-P5	alpha	True
HP1507101-P8	alpha	True
HP1508501T2D-A10	beta	True
HP1508501T2D-A11	acinar	True
HP1508501T2D-A12	alpha	True
HP1508501T2D-A13	alpha	True
HP1508501T2D-A14	alpha	True
HP1508501T2D-A15	alpha	True
HP1508501T2D-A16	alpha	True
HP1508501T2D-A17	ductal	True
HP1508501T2D-A19	PP	True
HP1508501T2D-A23	acinar	True
HP1508501T2D-A3	alpha	True
HP1508501T2D-A4	alpha	True
HP1508501T2D-A7	ductal	True
HP1508501T2D-A8	PSC	True
HP1508501T2D-B10	PP	True
HP1508501T2D-B11	alpha	True
HP1508501T2D-B12	alpha	True
HP1508501T2D-B14	acinar	True
HP1508501T2D-B15	alpha	True
HP1508501T2D-B16	beta	True
HP1508501T2D-B17	ductal	True
HP1508501T2D-B18	ductal	True
HP1508501T2D-B19	ductal	True
HP1508501T2D-B20	beta	True
HP1508501T2D-B23	alpha	True
HP1508501T2D-B3	alpha	True
HP1508501T2D-B4	alpha	True
HP1508501T2D-B5	alpha	True
HP1508501T2D-B8	alpha	True
HP1508501T2D-B9	alpha	True
HP1508501T2D-C10	alpha	True
HP1508501T2D-C11	ductal	True
HP1508501T2D-C12	PSC	True
HP1508501T2D-C13	alpha	True
HP1508501T2D-C14	beta	True
HP1508501T2D-C15	ductal	True
HP1508501T2D-C16	delta	True
HP1508501T2D-C17	beta	True
HP1508501T2D-C18	ductal	True
HP1508501T2D-C19	ductal	True
HP1508501T2D-C2	ductal	True
HP1508501T2D-C20	alpha	True
HP1508501T2D-C23	alpha	True
HP1508501T2D-C4	beta	True
HP1508501T2D-C8	delta	True
HP1508501T2D-C9	MHC class II	False
HP1508501T2D-D1	alpha	True
HP1508501T2D-D10	alpha	True
HP1508501T2D-D11	alpha	True
HP1508501T2D-D13	alpha	True
HP1508501T2D-D14	alpha	True
HP1508501T2D-D15	alpha	True
HP1508501T2D-D16	alpha	True
HP1508501T2D-D17	alpha	True
HP1508501T2D-D18	acinar	True
HP1508501T2D-D2	alpha	True
HP1508501T2D-D20	alpha	True
HP1508501T2D-D21	alpha	True
HP1508501T2D-D22	ductal	True
HP1508501T2D-D3	alpha	True
HP1508501T2D-D5	alpha	True
HP1508501T2D-D6	alpha	True
HP1508501T2D-D8	alpha	True
HP1508501T2D-D9	acinar	True
HP1508501T2D-E1	PSC	True
HP1508501T2D-E10	alpha	True
HP1508501T2D-E11	ductal	True
HP1508501T2D-E12	alpha	True
HP1508501T2D-E13	alpha	True
HP1508501T2D-E14	alpha	True
HP1508501T2D-E15	ductal	True
HP1508501T2D-E16	ductal	True
HP1508501T2D-E17	ductal	True
HP1508501T2D-E18	alpha	True
HP1508501T2D-E19	alpha	True
HP1508501T2D-E2	alpha	True
HP1508501T2D-E20	PSC	True
HP1508501T2D-E21	alpha	True
HP1508501T2D-E22	alpha	True
HP1508501T2D-E23	alpha	True
HP1508501T2D-E4	acinar	True
HP1508501T2D-E6	alpha	True
HP1508501T2D-E7	ductal	True
HP1508501T2D-E8	alpha	True
HP1508501T2D-E9	ductal	True
HP1508501T2D-F1	beta	True
HP1508501T2D-F11	PP	True
HP1508501T2D-F12	ductal	True
HP1508501T2D-F14	MHC class II	False
HP1508501T2D-F15	ductal	True
HP1508501T2D-F16	alpha	True
HP1508501T2D-F17	alpha	True
HP1508501T2D-F18	alpha	True
HP1508501T2D-F2	alpha	True
HP1508501T2D-F20	alpha	True
HP1508501T2D-F21	acinar	True
HP1508501T2D-F23	alpha	True
HP1508501T2D-F3	beta	True
HP1508501T2D-F4	ductal	True
HP1508501T2D-F5	ductal	True
HP1508501T2D-F7	beta	True
HP1508501T2D-F8	ductal	True
HP1508501T2D-F9	alpha	True
HP1508501T2D-G1	ductal	True
HP1508501T2D-G10	alpha	True
HP1508501T2D-G11	alpha	True
HP1508501T2D-G12	acinar	True
HP1508501T2D-G13	ductal	True
HP1508501T2D-G14	ductal	True
HP1508501T2D-G15	alpha	True
HP1508501T2D-G17	acinar	True
HP1508501T2D-G18	acinar	True
HP1508501T2D-G19	acinar	True
HP1508501T2D-G2	ductal	True
HP1508501T2D-G20	alpha	True
HP1508501T2D-G21	alpha	True
HP1508501T2D-G22	PP	True
HP1508501T2D-G23	acinar	True
HP1508501T2D-G3	ductal	True
HP1508501T2D-G4	alpha	True
HP1508501T2D-G5	beta	True
HP1508501T2D-G6	alpha	True
HP1508501T2D-H10	alpha	True
HP1508501T2D-H11	delta	True
HP1508501T2D-H12	mast	True
HP1508501T2D-H13	ductal	True
HP1508501T2D-H14	ductal	True
HP1508501T2D-H15	alpha	True
HP1508501T2D-H16	acinar	True
HP1508501T2D-H17	ductal	True
HP1508501T2D-H18	PP	True
HP1508501T2D-H21	alpha	True
HP1508501T2D-H22	alpha	True
HP1508501T2D-H23	alpha	True
HP1508501T2D-H3	alpha	True
HP1508501T2D-H4	alpha	True
HP1508501T2D-H5	PSC	True
HP1508501T2D-H6	alpha	True
HP1508501T2D-I1	ductal	True
HP1508501T2D-I10	acinar	True
HP1508501T2D-I11	ductal	True
HP1508501T2D-I12	ductal	True
HP1508501T2D-I13	ductal	True
HP1508501T2D-I14	alpha	True
HP1508501T2D-I15	alpha	True
HP1508501T2D-I16	alpha	True
HP1508501T2D-I17	ductal	True
HP1508501T2D-I18	delta	True
HP1508501T2D-I19	PSC	True
HP1508501T2D-I2	alpha	True
HP1508501T2D-I20	alpha	True
HP1508501T2D-I21	ductal	True
HP1508501T2D-I22	alpha	True
HP1508501T2D-I23	delta	True
HP1508501T2D-I3	PSC	True
HP1508501T2D-I4	ductal	True
HP1508501T2D-I7	acinar	True
HP1508501T2D-I8	beta	True
HP1508501T2D-J1	alpha	True
HP1508501T2D-J11	ductal	True
HP1508501T2D-J12	alpha	True
HP1508501T2D-J13	ductal	True
HP1508501T2D-J14	ductal	True
HP1508501T2D-J15	ductal	True
HP1508501T2D-J16	ductal	True
HP1508501T2D-J17	alpha	True
HP1508501T2D-J18	delta	True
HP1508501T2D-J2	PSC	True
HP1508501T2D-J20	alpha	True
HP1508501T2D-J21	alpha	True
HP1508501T2D-J22	ductal	True
HP1508501T2D-J23	acinar	True
HP1508501T2D-J3	alpha	True
HP1508501T2D-J4	alpha	True
HP1508501T2D-J5	alpha	True
HP1508501T2D-J6	beta	True
HP1508501T2D-J8	alpha	True
HP1508501T2D-J9	alpha	True
HP1508501T2D-K1	alpha	True
HP1508501T2D-K12	alpha	True
HP1508501T2D-K14	alpha	True
HP1508501T2D-K15	ductal	True
HP1508501T2D-K16	PP	True
HP1508501T2D-K17	alpha	True
HP1508501T2D-K18	endothelial	True
HP1508501T2D-K19	alpha	True
HP1508501T2D-K2	ductal	True
HP1508501T2D-K21	alpha	True
HP1508501T2D-K22	ductal	True
HP1508501T2D-K23	ductal	True
HP1508501T2D-K3	ductal	True
HP1508501T2D-K4	alpha	True
HP1508501T2D-K6	acinar	True
HP1508501T2D-K7	ductal	True
HP1508501T2D-K9	alpha	True
HP1508501T2D-L10	PSC	True
HP1508501T2D-L11	ductal	True
HP1508501T2D-L13	alpha	True
HP1508501T2D-L14	alpha	True
HP1508501T2D-L15	PSC	True
HP1508501T2D-L16	alpha	True
HP1508501T2D-L17	alpha	True
HP1508501T2D-L18	alpha	True
HP1508501T2D-L2	acinar	True
HP1508501T2D-L20	ductal	True
HP1508501T2D-L21	ductal	True
HP1508501T2D-L23	ductal	True
HP1508501T2D-L5	ductal	True
HP1508501T2D-L6	alpha	True
HP1508501T2D-L7	alpha	True
HP1508501T2D-L9	ductal	True
HP1508501T2D-M1	mast	True
HP1508501T2D-M10	PSC	True
HP1508501T2D-M11	acinar	True
HP1508501T2D-M12	alpha	True
HP1508501T2D-M14	beta	True
HP1508501T2D-M15	ductal	True
HP1508501T2D-M16	PSC	True
HP1508501T2D-M17	ductal	True
HP1508501T2D-M18	acinar	True
HP1508501T2D-M19	alpha	True
HP1508501T2D-M20	alpha	True
HP1508501T2D-M22	ductal	True
HP1508501T2D-M23	ductal	True
HP1508501T2D-M3	ductal	True
HP1508501T2D-M4	alpha	True
HP1508501T2D-M5	alpha	True
HP1508501T2D-M7	acinar	True
HP1508501T2D-M8	alpha	True
HP1508501T2D-M9	alpha	True
HP1508501T2D-N1	ductal	True
HP1508501T2D-N10	ductal	True
HP1508501T2D-N11	alpha	True
HP1508501T2D-N13	acinar	True
HP1508501T2D-N15	ductal	True
HP1508501T2D-N17	ductal	True
HP1508501T2D-N18	PP	True
HP1508501T2D-N19	ductal	True
HP1508501T2D-N2	ductal	True
HP1508501T2D-N20	alpha	True
HP1508501T2D-N21	acinar	True
HP1508501T2D-N22	ductal	True
HP1508501T2D-N23	ductal	True
HP1508501T2D-N4	alpha	True
HP1508501T2D-N5	alpha	True
HP1508501T2D-N6	ductal	True
HP1508501T2D-N7	ductal	True
HP1508501T2D-N8	ductal	True
HP1508501T2D-N9	ductal	True
HP1508501T2D-O1	alpha	True
HP1508501T2D-O11	acinar	True
HP1508501T2D-O12	ductal	True
HP1508501T2D-O13	acinar	True
HP1508501T2D-O14	alpha	True
HP1508501T2D-O3	ductal	True
HP1508501T2D-O4	beta	True
HP1508501T2D-O7	acinar	True
HP1508501T2D-O8	ductal	True
HP1508501T2D-O9	ductal	True
HP1508501T2D-P11	alpha	True
HP1508501T2D-P14	acinar	True
HP1508501T2D-P2	acinar	True
HP1508501T2D-P3	alpha	True
HP1508501T2D-P4	PP	True
HP1508501T2D-P6	alpha	True
HP1508501T2D-P7	alpha	True
HP1508501T2D-P8	alpha	True
HP1508501T2D-P9	alpha	True
HP1509101-A3	alpha	True
HP1509101-A7	alpha	True
HP1509101-A8	ductal	True
HP1509101-B10	alpha	True
HP1509101-B13	beta	True
HP1509101-B16	acinar	True
HP1509101-B17	PSC	True
HP1509101-B19	ductal	True
HP1509101-B20	alpha	True
HP1509101-B23	ductal	True
HP1509101-C21	ductal	True
HP1509101-C22	alpha	True
HP1509101-C23	beta	True
HP1509101-D17	ductal	True
HP1509101-D18	beta	True
HP1509101-D19	alpha	True
HP1509101-D22	delta	True
HP1509101-D23	ductal	True
HP1509101-D5	alpha	True
HP1509101-D8	alpha	True
HP1509101-E19	alpha	True
HP1509101-E22	beta	True
HP1509101-E3	ductal	True
HP1509101-E4	delta	True
HP1509101-F18	acinar	True
HP1509101-F19	alpha	True
HP1509101-F21	PP	True
HP1509101-F22	ductal	True
HP1509101-F23	endothelial	True
HP1509101-F4	alpha	True
HP1509101-F7	ductal	True
HP1509101-G16	alpha	True
HP1509101-G18	alpha	True
HP1509101-G19	alpha	True
HP1509101-G9	beta	True
HP1509101-H1	alpha	True
HP1509101-H13	delta	True
HP1509101-H20	ductal	True
HP1509101-H21	alpha	True
HP1509101-H22	alpha	True
HP1509101-H23	alpha	True
HP1509101-H7	delta	True
HP1509101-I10	alpha	True
HP1509101-I12	ductal	True
HP1509101-I17	endothelial	True
HP1509101-I21	alpha	True
HP1509101-I23	alpha	True
HP1509101-I5	alpha	True
HP1509101-J12	alpha	True
HP1509101-J15	alpha	True
HP1509101-J16	delta	True
HP1509101-J19	beta	True
HP1509101-J20	alpha	True
HP1509101-J22	ductal	True
HP1509101-J23	ductal	True
HP1509101-J4	alpha	True
HP1509101-J8	alpha	True
HP1509101-J9	ductal	True
HP1509101-K10	ductal	True
HP1509101-K15	ductal	True
HP1509101-K19	ductal	True
HP1509101-K22	alpha	True
HP1509101-K9	beta	True
HP1509101-L11	alpha	True
HP1509101-L13	ductal	True
HP1509101-L14	beta	True
HP1509101-L16	ductal	True
HP1509101-L17	alpha	True
HP1509101-L18	PSC	True
HP1509101-L19	delta	True
HP1509101-L23	delta	True
HP1509101-L8	alpha	True
HP1509101-M10	alpha	True
HP1509101-M12	alpha	True
HP1509101-N17	alpha	True
HP1509101-N18	alpha	True
HP1509101-N20	alpha	True
HP1509101-N23	beta	True
HP1509101-N9	alpha	True
HP1509101-O1	delta	True
HP1509101-O13	alpha	True
HP1509101-O14	ductal	True
HP1509101-O16	alpha	True
HP1509101-O18	beta	True
HP1509101-O4	delta	True
HP1509101-O7	alpha	True
HP1509101-O9	alpha	True
HP1509101-P12	delta	True
HP1509101-P17	ductal	True
HP1509101-P18	alpha	True
HP1509101-P20	alpha	True
HP1509101-P23	PSC	True
HP1509101-P4	alpha	True
HP1509101-P8	ductal	True
HP1509101-P9	ductal	True
HP1525301T2D-A1	ductal	True
HP1525301T2D-A11	alpha	True
HP1525301T2D-A12	ductal	True
HP1525301T2D-A13	ductal	True
HP1525301T2D-A14	ductal	True
HP1525301T2D-A15	ductal	True
HP1525301T2D-A16	PP	True
HP1525301T2D-A17	ductal	True
HP1525301T2D-A19	ductal	True
HP1525301T2D-A3	alpha	True
HP1525301T2D-A5	alpha	True
HP1525301T2D-A6	alpha	True
HP1525301T2D-A7	delta	True
HP1525301T2D-A9	alpha	True
HP1525301T2D-B1	alpha	True
HP1525301T2D-B10	ductal	True
HP1525301T2D-B11	alpha	True
HP1525301T2D-B12	acinar	True
HP1525301T2D-B13	alpha	True
HP1525301T2D-B14	ductal	True
HP1525301T2D-B15	alpha	True
HP1525301T2D-B18	alpha	True
HP1525301T2D-B21	ductal	True
HP1525301T2D-B24	ductal	True
HP1525301T2D-B3	ductal	True
HP1525301T2D-B4	alpha	True
HP1525301T2D-B5	PP	True
HP1525301T2D-B6	PSC	True
HP1525301T2D-B7	alpha	True
HP1525301T2D-B8	PP	True
HP1525301T2D-C1	alpha	True
HP1525301T2D-C10	alpha	True
HP1525301T2D-C11	ductal	True
HP1525301T2D-C13	ductal	True
HP1525301T2D-C14	acinar	True
HP1525301T2D-C15	ductal	True
HP1525301T2D-C16	alpha	True
HP1525301T2D-C17	ductal	True
HP1525301T2D-C19	ductal	True
HP1525301T2D-C2	alpha	True
HP1525301T2D-C21	ductal	True
HP1525301T2D-C22	ductal	True
HP1525301T2D-C23	acinar	True
HP1525301T2D-C24	ductal	True
HP1525301T2D-C3	ductal	True
HP1525301T2D-C4	ductal	True
HP1525301T2D-C5	alpha	True
HP1525301T2D-C6	ductal	True
HP1525301T2D-C7	acinar	True
HP1525301T2D-C8	alpha	True
HP1525301T2D-C9	ductal	True
HP1525301T2D-D1	ductal	True
HP1525301T2D-D10	acinar	True
HP1525301T2D-D11	alpha	True
HP1525301T2D-D12	PP	True
HP1525301T2D-D14	ductal	True
HP1525301T2D-D15	ductal	True
HP1525301T2D-D19	delta	True
HP1525301T2D-D2	ductal	True
HP1525301T2D-D21	PSC	True
HP1525301T2D-D22	ductal	True
HP1525301T2D-D23	beta	True
HP1525301T2D-D3	ductal	True
HP1525301T2D-D4	ductal	True
HP1525301T2D-D6	acinar	True
HP1525301T2D-D8	ductal	True
HP1525301T2D-D9	alpha	True
HP1525301T2D-E1	alpha	True
HP1525301T2D-E11	ductal	True
HP1525301T2D-E12	PSC	True
HP1525301T2D-E13	alpha	True
HP1525301T2D-E15	ductal	True
HP1525301T2D-E16	alpha	True
HP1525301T2D-E17	ductal	True
HP1525301T2D-E19	PP	True
HP1525301T2D-E2	acinar	True
HP1525301T2D-E22	alpha	True
HP1525301T2D-E23	alpha	True
HP1525301T2D-E24	PSC	True
HP1525301T2D-E3	PP	True
HP1525301T2D-E5	PSC	True
HP1525301T2D-E7	alpha	True
HP1525301T2D-E8	alpha	True
HP1525301T2D-F10	acinar	True
HP1525301T2D-F12	ductal	True
HP1525301T2D-F15	beta	True
HP1525301T2D-F16	ductal	True
HP1525301T2D-F19	alpha	True
HP1525301T2D-F2	alpha	True
HP1525301T2D-F21	ductal	True
HP1525301T2D-F23	beta	True
HP1525301T2D-F24	PSC	True
HP1525301T2D-F3	acinar	True
HP1525301T2D-F4	alpha	True
HP1525301T2D-F5	acinar	True
HP1525301T2D-F7	PSC	True
HP1525301T2D-F8	ductal	True
HP1525301T2D-G1	alpha	True
HP1525301T2D-G10	ductal	True
HP1525301T2D-G11	alpha	True
HP1525301T2D-G13	ductal	True
HP1525301T2D-G14	ductal	True
HP1525301T2D-G15	alpha	True
HP1525301T2D-G17	acinar	True
HP1525301T2D-G18	alpha	True
HP1525301T2D-G19	alpha	True
HP1525301T2D-G20	ductal	True
HP1525301T2D-G21	ductal	True
HP1525301T2D-G22	alpha	True
HP1525301T2D-G23	PP	True
HP1525301T2D-G3	ductal	True
HP1525301T2D-G4	alpha	True
HP1525301T2D-G5	ductal	True
HP1525301T2D-G8	beta	True
HP1525301T2D-G9	ductal	True
HP1525301T2D-H1	ductal	True
HP1525301T2D-H10	beta	True
HP1525301T2D-H11	ductal	True
HP1525301T2D-H12	alpha	True
HP1525301T2D-H14	ductal	True
HP1525301T2D-H16	ductal	True
HP1525301T2D-H17	ductal	True
HP1525301T2D-H18	ductal	True
HP1525301T2D-H19	alpha	True
HP1525301T2D-H2	ductal	True
HP1525301T2D-H20	alpha	True
HP1525301T2D-H23	alpha	True
HP1525301T2D-H24	ductal	True
HP1525301T2D-H3	ductal	True
HP1525301T2D-H6	alpha	True
HP1525301T2D-H7	alpha	True
HP1525301T2D-H8	acinar	True
HP1525301T2D-H9	acinar	True
HP1525301T2D-I1	ductal	True
HP1525301T2D-I10	ductal	True
HP1525301T2D-I11	MHC class II	False
HP1525301T2D-I12	alpha	True
HP1525301T2D-I15	ductal	True
HP1525301T2D-I16	ductal	True
HP1525301T2D-I17	ductal	True
HP1525301T2D-I18	ductal	True
HP1525301T2D-I19	ductal	True
HP1525301T2D-I2	alpha	True
HP1525301T2D-I21	alpha	True
HP1525301T2D-I22	ductal	True
HP1525301T2D-I23	ductal	True
HP1525301T2D-I24	ductal	True
HP1525301T2D-I3	ductal	True
HP1525301T2D-I4	alpha	True
HP1525301T2D-I5	alpha	True
HP1525301T2D-I6	beta	True
HP1525301T2D-I7	alpha	True
HP1525301T2D-I8	ductal	True
HP1525301T2D-J1	alpha	True
HP1525301T2D-J10	delta	True
HP1525301T2D-J13	alpha	True
HP1525301T2D-J14	alpha	True
HP1525301T2D-J15	ductal	True
HP1525301T2D-J16	acinar	True
HP1525301T2D-J17	PSC	True
HP1525301T2D-J18	PP	True
HP1525301T2D-J19	alpha	True
HP1525301T2D-J2	alpha	True
HP1525301T2D-J20	ductal	True
HP1525301T2D-J21	ductal	True
HP1525301T2D-J23	acinar	True
HP1525301T2D-J4	alpha	True
HP1525301T2D-J5	acinar	True
HP1525301T2D-J7	ductal	True
HP1525301T2D-J8	alpha	True
HP1525301T2D-J9	ductal	True
HP1525301T2D-K1	ductal	True
HP1525301T2D-K10	PSC	True
HP1525301T2D-K11	ductal	True
HP1525301T2D-K12	ductal	True
HP1525301T2D-K14	ductal	True
HP1525301T2D-K15	alpha	True
HP1525301T2D-K16	ductal	True
HP1525301T2D-K17	ductal	True
HP1525301T2D-K18	ductal	True
HP1525301T2D-K19	alpha	True
HP1525301T2D-K2	ductal	True
HP1525301T2D-K21	ductal	True
HP1525301T2D-K22	alpha	True
HP1525301T2D-K23	alpha	True
HP1525301T2D-K3	alpha	True
HP1525301T2D-K5	acinar	True
HP1525301T2D-K6	ductal	True
HP1525301T2D-K7	acinar	True
HP1525301T2D-K8	acinar	True
HP1525301T2D-L1	ductal	True
HP1525301T2D-L13	ductal	True
HP1525301T2D-L14	beta	True
HP1525301T2D-L15	alpha	True
HP1525301T2D-L16	ductal	True
HP1525301T2D-L17	beta	True
HP1525301T2D-L2	ductal	True
HP1525301T2D-L20	ductal	True
HP1525301T2D-L22	alpha	True
HP1525301T2D-L23	ductal	True
HP1525301T2D-L24	alpha	True
HP1525301T2D-L4	ductal	True
HP1525301T2D-L5	acinar	True
HP1525301T2D-L6	ductal	True
HP1525301T2D-L8	PSC	True
HP1525301T2D-L9	alpha	True
HP1525301T2D-M10	alpha	True
HP1525301T2D-M11	ductal	True
HP1525301T2D-M12	ductal	True
HP1525301T2D-M13	alpha	True
HP1525301T2D-M15	ductal	True
HP1525301T2D-M16	alpha	True
HP1525301T2D-M17	delta	True
HP1525301T2D-M2	PSC	True
HP1525301T2D-M20	ductal	True
HP1525301T2D-M22	ductal	True
HP1525301T2D-M5	acinar	True
HP1525301T2D-M6	ductal	True
HP1525301T2D-M7	beta	True
HP1525301T2D-M8	alpha	True
HP1525301T2D-M9	ductal	True
HP1525301T2D-N1	ductal	True
HP1525301T2D-N10	acinar	True
HP1525301T2D-N11	ductal	True
HP1525301T2D-N12	ductal	True
HP1525301T2D-N13	alpha	True
HP1525301T2D-N14	ductal	True
HP1525301T2D-N16	ductal	True
HP1525301T2D-N18	beta	True
HP1525301T2D-N20	ductal	True
HP1525301T2D-N21	PP	True
HP1525301T2D-N23	alpha	True
HP1525301T2D-N24	alpha	True
HP1525301T2D-N3	beta	True
HP1525301T2D-N4	ductal	True
HP1525301T2D-N5	ductal	True
HP1525301T2D-N6	ductal	True
HP1525301T2D-N7	alpha	True
HP1525301T2D-N8	alpha	True
HP1525301T2D-N9	alpha	True
HP1525301T2D-O1	acinar	True
HP1525301T2D-O11	PP	True
HP1525301T2D-O13	alpha	True
HP1525301T2D-O14	ductal	True
HP1525301T2D-O16	ductal	True
HP1525301T2D-O17	ductal	True
HP1525301T2D-O18	alpha	True
HP1525301T2D-O2	delta	True
HP1525301T2D-O20	ductal	True
HP1525301T2D-O22	ductal	True
HP1525301T2D-O23	ductal	True
HP1525301T2D-O24	ductal	True
HP1525301T2D-O3	acinar	True
HP1525301T2D-O5	alpha	True
HP1525301T2D-O6	alpha	True
HP1525301T2D-O7	ductal	True
HP1525301T2D-O9	PSC	True
HP1525301T2D-P1	ductal	True
HP1525301T2D-P11	alpha	True
HP1525301T2D-P13	ductal	True
HP1525301T2D-P14	ductal	True
HP1525301T2D-P15	endothelial	True
HP1525301T2D-P16	PSC	True
HP1525301T2D-P17	alpha	True
HP1525301T2D-P18	alpha	True
HP1525301T2D-P19	ductal	True
HP1525301T2D-P2	ductal	True
HP1525301T2D-P20	ductal	True
HP1525301T2D-P21	ductal	True
HP1525301T2D-P22	alpha	True
HP1525301T2D-P23	ductal	True
HP1525301T2D-P24	alpha	True
HP1525301T2D-P4	ductal	True
HP1525301T2D-P5	ductal	True
HP1525301T2D-P6	alpha	True
HP1525301T2D-P8	alpha	True
HP1525301T2D-P9	alpha	True
HP1526901T2D-A1	beta	True
HP1526901T2D-A10	delta	True
HP1526901T2D-A11	ductal	True
HP1526901T2D-A12	alpha	True
HP1526901T2D-A14	beta	True
HP1526901T2D-A15	PP	True
HP1526901T2D-A16	delta	True
HP1526901T2D-A17	alpha	True
HP1526901T2D-A19	beta	True
HP1526901T2D-A2	beta	True
HP1526901T2D-A20	PP	True
HP1526901T2D-A21	ductal	True
HP1526901T2D-A22	delta	True
HP1526901T2D-A24	ductal	True
HP1526901T2D-A3	alpha	True
HP1526901T2D-A4	delta	True
HP1526901T2D-A5	beta	True
HP1526901T2D-A6	acinar	True
HP1526901T2D-A7	alpha	True
HP1526901T2D-A8	alpha	True
HP1526901T2D-A9	alpha	True
HP1526901T2D-B1	beta	True
HP1526901T2D-B10	beta	True
HP1526901T2D-B11	alpha	True
HP1526901T2D-B12	alpha	True
HP1526901T2D-B13	PP	True
HP1526901T2D-B14	alpha	True
HP1526901T2D-B17	alpha	True
HP1526901T2D-B18	delta	True
HP1526901T2D-B20	alpha	True
HP1526901T2D-B23	ductal	True
HP1526901T2D-B24	ductal	True
HP1526901T2D-B3	ductal	True
HP1526901T2D-B4	beta	True
HP1526901T2D-B5	ductal	True
HP1526901T2D-B6	delta	True
HP1526901T2D-B7	delta	True
HP1526901T2D-B9	beta	True
HP1526901T2D-C1	delta	True
HP1526901T2D-C10	beta	True
HP1526901T2D-C11	beta	True
HP1526901T2D-C13	PP	True
HP1526901T2D-C14	ductal	True
HP1526901T2D-C15	delta	True
HP1526901T2D-C16	alpha	True
HP1526901T2D-C17	beta	True
HP1526901T2D-C19	PP	True
HP1526901T2D-C2	alpha	True
HP1526901T2D-C20	ductal	True
HP1526901T2D-C22	ductal	True
HP1526901T2D-C23	ductal	True
HP1526901T2D-C24	delta	True
HP1526901T2D-C3	beta	True
HP1526901T2D-C4	alpha	True
HP1526901T2D-C6	alpha	True
HP1526901T2D-C8	acinar	True
HP1526901T2D-D1	alpha	True
HP1526901T2D-D10	PP	True
HP1526901T2D-D12	ductal	True
HP1526901T2D-D14	PP	True
HP1526901T2D-D16	alpha	True
HP1526901T2D-D17	beta	True
HP1526901T2D-D18	alpha	True
HP1526901T2D-D19	beta	True
HP1526901T2D-D2	ductal	True
HP1526901T2D-D21	delta	True
HP1526901T2D-D22	ductal	True
HP1526901T2D-D24	delta	True
HP1526901T2D-D3	ductal	True
HP1526901T2D-D4	ductal	True
HP1526901T2D-D5	alpha	True
HP1526901T2D-D7	acinar	True
HP1526901T2D-D8	alpha	True
HP1526901T2D-D9	alpha	True
HP1526901T2D-E1	ductal	True
HP1526901T2D-E10	PP	True
HP1526901T2D-E11	beta	True
HP1526901T2D-E14	ductal	True
HP1526901T2D-E15	PSC	True
HP1526901T2D-E16	acinar	True
HP1526901T2D-E18	alpha	True
HP1526901T2D-E19	delta	True
HP1526901T2D-E2	beta	True
HP1526901T2D-E20	delta	True
HP1526901T2D-E21	delta	True
HP1526901T2D-E22	ductal	True
HP1526901T2D-E23	ductal	True
HP1526901T2D-E3	delta	True
HP1526901T2D-E4	PP	True
HP1526901T2D-E5	PP	True
HP1526901T2D-E6	alpha	True
HP1526901T2D-E8	alpha	True
HP1526901T2D-E9	ductal	True
HP1526901T2D-F1	alpha	True
HP1526901T2D-F10	alpha	True
HP1526901T2D-F12	PP	True
HP1526901T2D-F15	ductal	True
HP1526901T2D-F16	ductal	True
HP1526901T2D-F17	beta	True
HP1526901T2D-F18	beta	True
HP1526901T2D-F19	alpha	True
HP1526901T2D-F2	alpha	True
HP1526901T2D-F20	alpha	True
HP1526901T2D-F21	alpha	True
HP1526901T2D-F22	ductal	True
HP1526901T2D-F23	delta	True
HP1526901T2D-F24	ductal	True
HP1526901T2D-F3	alpha	True
HP1526901T2D-F5	beta	True
HP1526901T2D-F7	PP	True
HP1526901T2D-F8	ductal	True
HP1526901T2D-F9	alpha	True
HP1526901T2D-G1	PP	True
HP1526901T2D-G10	alpha	True
HP1526901T2D-G11	MHC class II	False
HP1526901T2D-G13	alpha	True
HP1526901T2D-G14	alpha	True
HP1526901T2D-G16	alpha	True
HP1526901T2D-G17	beta	True
HP1526901T2D-G18	alpha	True
HP1526901T2D-G19	PP	True
HP1526901T2D-G2	delta	True
HP1526901T2D-G21	beta	True
HP1526901T2D-G22	ductal	True
HP1526901T2D-G23	delta	True
HP1526901T2D-G24	ductal	True
HP1526901T2D-G4	beta	True
HP1526901T2D-G7	alpha	True
HP1526901T2D-G8	alpha	True
HP1526901T2D-G9	PP	True
HP1526901T2D-H10	beta	True
HP1526901T2D-H12	alpha	True
HP1526901T2D-H16	PP	True
HP1526901T2D-H17	alpha	True
HP1526901T2D-H18	delta	True
HP1526901T2D-H19	alpha	True
HP1526901T2D-H2	acinar	True
HP1526901T2D-H20	beta	True
HP1526901T2D-H21	ductal	True
HP1526901T2D-H22	beta	True
HP1526901T2D-H23	PP	True
HP1526901T2D-H24	beta	True
HP1526901T2D-H3	PP	True
HP1526901T2D-H4	beta	True
HP1526901T2D-H5	ductal	True
HP1526901T2D-H6	alpha	True
HP1526901T2D-H8	ductal	True
HP1526901T2D-H9	ductal	True
HP1526901T2D-I1	ductal	True
HP1526901T2D-I10	delta	True
HP1526901T2D-I11	delta	True
HP1526901T2D-I12	acinar	True
HP1526901T2D-I13	beta	True
HP1526901T2D-I14	alpha	True
HP1526901T2D-I15	alpha	True
HP1526901T2D-I16	alpha	True
HP1526901T2D-I17	acinar	True
HP1526901T2D-I18	delta	True
HP1526901T2D-I19	delta	True
HP1526901T2D-I20	beta	True
HP1526901T2D-I21	alpha	True
HP1526901T2D-I22	beta	True
HP1526901T2D-I24	PP	True
HP1526901T2D-I3	beta	True
HP1526901T2D-I4	alpha	True
HP1526901T2D-I6	alpha	True
HP1526901T2D-I7	alpha	True
HP1526901T2D-I8	alpha	True
HP1526901T2D-I9	PSC	True
HP1526901T2D-J1	delta	True
HP1526901T2D-J11	ductal	True
HP1526901T2D-J12	ductal	True
HP1526901T2D-J13	alpha	True
HP1526901T2D-J14	alpha	True
HP1526901T2D-J17	PP	True
HP1526901T2D-J18	ductal	True
HP1526901T2D-J19	alpha	True
HP1526901T2D-J2	alpha	True
HP1526901T2D-J20	delta	True
HP1526901T2D-J21	beta	True
HP1526901T2D-J22	delta	True
HP1526901T2D-J24	alpha	True
HP1526901T2D-J3	ductal	True
HP1526901T2D-J4	alpha	True
HP1526901T2D-J5	beta	True
HP1526901T2D-J6	beta	True
HP1526901T2D-J8	alpha	True
HP1526901T2D-K1	ductal	True
HP1526901T2D-K10	alpha	True
HP1526901T2D-K11	PP	True
HP1526901T2D-K12	PP	True
HP1526901T2D-K14	alpha	True
HP1526901T2D-K15	alpha	True
HP1526901T2D-K17	acinar	True
HP1526901T2D-K18	alpha	True
HP1526901T2D-K19	ductal	True
HP1526901T2D-K2	alpha	True
HP1526901T2D-K20	beta	True
HP1526901T2D-K21	PP	True
HP1526901T2D-K22	delta	True
HP1526901T2D-K23	beta	True
HP1526901T2D-K24	delta	True
HP1526901T2D-K3	beta	True
HP1526901T2D-K4	ductal	True
HP1526901T2D-K6	alpha	True
HP1526901T2D-K7	beta	True
HP1526901T2D-L1	beta	True
HP1526901T2D-L11	alpha	True
HP1526901T2D-L12	PP	True
HP1526901T2D-L13	alpha	True
HP1526901T2D-L14	PP	True
HP1526901T2D-L15	acinar	True
HP1526901T2D-L16	alpha	True
HP1526901T2D-L17	alpha	True
HP1526901T2D-L19	alpha	True
HP1526901T2D-L2	alpha	True
HP1526901T2D-L21	alpha	True
HP1526901T2D-L23	alpha	True
HP1526901T2D-L24	delta	True
HP1526901T2D-L4	beta	True
HP1526901T2D-L5	beta	True
HP1526901T2D-L6	beta	True
HP1526901T2D-L7	PP	True
HP1526901T2D-M1	alpha	True
HP1526901T2D-M13	epsilon	True
HP1526901T2D-M14	ductal	True
HP1526901T2D-M15	beta	True
HP1526901T2D-M16	alpha	True
HP1526901T2D-M18	acinar	True
HP1526901T2D-M19	delta	True
HP1526901T2D-M2	beta	True
HP1526901T2D-M20	ductal	True
HP1526901T2D-M21	alpha	True
HP1526901T2D-M22	PP	True
HP1526901T2D-M23	alpha	True
HP1526901T2D-M3	PP	True
HP1526901T2D-M4	alpha	True
HP1526901T2D-M7	alpha	True
HP1526901T2D-M8	ductal	True
HP1526901T2D-N1	beta	True
HP1526901T2D-N10	alpha	True
HP1526901T2D-N11	ductal	True
HP1526901T2D-N12	beta	True
HP1526901T2D-N13	beta	True
HP1526901T2D-N14	alpha	True
HP1526901T2D-N15	alpha	True
HP1526901T2D-N16	delta	True
HP1526901T2D-N18	PP	True
HP1526901T2D-N19	PP	True
HP1526901T2D-N2	alpha	True
HP1526901T2D-N20	PP	True
HP1526901T2D-N21	ductal	True
HP1526901T2D-N23	beta	True
HP1526901T2D-N3	beta	True
HP1526901T2D-N4	acinar	True
HP1526901T2D-N6	alpha	True
HP1526901T2D-N7	acinar	True
HP1526901T2D-N8	alpha	True
HP1526901T2D-N9	alpha	True
HP1526901T2D-O1	beta	True
HP1526901T2D-O10	PP	True
HP1526901T2D-O12	mast	True
HP1526901T2D-O13	beta	True
HP1526901T2D-O14	beta	True
HP1526901T2D-O15	beta	True
HP1526901T2D-O17	beta	True
HP1526901T2D-O19	acinar	True
HP1526901T2D-O2	PSC	True
HP1526901T2D-O21	alpha	True
HP1526901T2D-O22	alpha	True
HP1526901T2D-O23	alpha	True
HP1526901T2D-O24	beta	True
HP1526901T2D-O3	PP	True
HP1526901T2D-O4	ductal	True
HP1526901T2D-O5	ductal	True
HP1526901T2D-O6	alpha	True
HP1526901T2D-O7	delta	True
HP1526901T2D-O8	beta	True
HP1526901T2D-O9	delta	True
HP1526901T2D-P1	alpha	True
HP1526901T2D-P10	beta	True
HP1526901T2D-P11	beta	True
HP1526901T2D-P12	alpha	True
HP1526901T2D-P13	PSC	True
HP1526901T2D-P14	alpha	True
HP1526901T2D-P17	beta	True
HP1526901T2D-P18	beta	True
HP1526901T2D-P19	alpha	True
HP1526901T2D-P2	alpha	True
HP1526901T2D-P20	alpha	True
HP1526901T2D-P21	alpha	True
HP1526901T2D-P22	delta	True
HP1526901T2D-P23	PP	True
HP1526901T2D-P4	beta	True
HP1526901T2D-P7	beta	True
HP1526901T2D-P9	beta	True
Sample_1	beta	True
Sample_2	beta	True
Sample_3	beta	True
Sample_4	beta	True
Sample_5	beta	True
Sample_6	beta	True
Sample_7	beta	True
Sample_8	beta	True
Sample_9	beta	True
Sample_10	beta	True
Sample_11	beta	True
Sample_12	beta	True
Sample_13	beta	True
Sample_14	beta	True
Sample_15	beta	True
Sample_16	beta	True
Sample_17	beta	True
Sample_18	beta	True
Sample_19	beta	True
Sample_20	beta	True
Sample_21	beta	True
Sample_22	beta	True
Sample_23	beta	True
Sample_24	beta	True
Sample_25	beta	True
Sample_26	beta	True
Sample_27	beta	True
Sample_28	beta	True
Sample_29	beta	True
Sample_30	beta	True
Sample_32	beta	True
Sample_33	beta	True
Sample_34	beta	True
Sample_35	beta	True
Sample_36	beta	True
Sample_37	beta	True
Sample_38	beta	True
Sample_39	beta	True
Sample_40	beta	True
Sample_41	beta	True
Sample_42	beta	True
Sample_44	beta	True
Sample_45	beta	True
Sample_46	beta	True
Sample_47	beta	True
Sample_48	beta	True
Sample_49	beta	True
Sample_50	beta	True
Sample_51	beta	True
Sample_52	beta	True
Sample_53	beta	True
Sample_54	beta	True
Sample_55	beta	True
Sample_56	beta	True
Sample_57	beta	True
Sample_58	beta	True
Sample_59	beta	True
Sample_60	beta	True
Sample_61	beta	True
Sample_62	beta	True
Sample_63	beta	True
Sample_64	beta	True
Sample_65	beta	True
Sample_66	beta	True
Sample_67	beta	True
Sample_68	beta	True
Sample_69	beta	True
Sample_71	beta	True
Sample_72	beta	True
Sample_74	beta	True
Sample_75	beta	True
Sample_76	beta	True
Sample_77	beta	True
Sample_78	beta	True
Sample_79	beta	True
Sample_80	beta	True
Sample_81	beta	True
Sample_82	beta	True
Sample_83	beta	True
Sample_84	beta	True
Sample_85	beta	True
Sample_86	beta	True
Sample_87	beta	True
Sample_88	beta	True
Sample_89	beta	True
Sample_90	beta	True
Sample_91	beta	True
Sample_92	beta	True
Sample_93	beta	True
Sample_95	beta	True
Sample_96	beta	True
Sample_97	beta	True
Sample_98	beta	True
Sample_99	beta	True
Sample_100	beta	True
Sample_101	beta	True
Sample_102	beta	True
Sample_103	beta	True
Sample_104	beta	True
Sample_105	beta	True
Sample_106	beta	True
Sample_107	beta	True
Sample_108	beta	True
Sample_109	beta	True
Sample_110	beta	True
Sample_111	beta	True
Sample_112	beta	True
Sample_113	beta	True
Sample_115	beta	True
Sample_116	beta	True
Sample_117	beta	True
Sample_119	beta	True
Sample_120	beta	True
Sample_121	beta	True
Sample_122	beta	True
Sample_124	beta	True
Sample_125	beta	True
Sample_126	beta	True
Sample_127	beta	True
Sample_128	beta	True
Sample_129	beta	True
Sample_130	beta	True
Sample_131	beta	True
Sample_132	beta	True
Sample_133	beta	True
Sample_134	beta	True
Sample_135	beta	True
Sample_136	beta	True
Sample_137	beta	True
Sample_138	beta	True
Sample_139	beta	True
Sample_140	beta	True
Sample_141	beta	True
Sample_142	beta	True
Sample_143	beta	True
Sample_145	beta	True
Sample_147	beta	True
Sample_148	beta	True
Sample_149	beta	True
Sample_150	beta	True
Sample_151	beta	True
Sample_152	beta	True
Sample_153	beta	True
Sample_154	beta	True
Sample_155	beta	True
Sample_156	beta	True
Sample_157	beta	True
Sample_158	beta	True
Sample_159	beta	True
Sample_160	beta	True
Sample_161	beta	True
Sample_162	beta	True
Sample_163	beta	True
Sample_164	beta	True
Sample_165	beta	True
Sample_166	beta	True
Sample_167	beta	True
Sample_168	beta	True
Sample_169	beta	True
Sample_170	beta	True
Sample_172	beta	True
Sample_174	beta	True
Sample_175	beta	True
Sample_176	beta	True
Sample_177	beta	True
Sample_178	beta	True
Sample_179	beta	True
Sample_180	beta	True
Sample_181	beta	True
Sample_182	beta	True
Sample_183	beta	True
Sample_184	beta	True
Sample_185	beta	True
Sample_186	beta	True
Sample_187	beta	True
Sample_188	beta	True
Sample_189	beta	True
Sample_190	beta	True
Sample_191	beta	True
Sample_192	beta	True
Sample_193	beta	True
Sample_194	beta	True
Sample_195	beta	True
Sample_196	beta	True
Sample_197	beta	True
Sample_198	beta	True
Sample_199	beta	True
Sample_200	beta	True
Sample_201	beta	True
Sample_202	beta	True
Sample_203	beta	True
Sample_205	beta	True
Sample_206	beta	True
Sample_207	beta	True
Sample_208	alpha	True
Sample_209	alpha	True
Sample_211	alpha	True
Sample_212	alpha	True
Sample_213	alpha	True
Sample_214	alpha	True
Sample_215	alpha	True
Sample_216	alpha	True
Sample_217	alpha	True
Sample_218	alpha	True
Sample_219	alpha	True
Sample_220	alpha	True
Sample_221	alpha	True
Sample_222	alpha	True
Sample_223	alpha	True
Sample_224	alpha	True
Sample_225	alpha	True
Sample_226	alpha	True
Sample_227	alpha	True
Sample_228	alpha	True
Sample_229	alpha	True
Sample_230	alpha	True
Sample_231	alpha	True
Sample_232	alpha	True
Sample_234	alpha	True
Sample_235	alpha	True
Sample_236	alpha	True
Sample_237	alpha	True
Sample_238	alpha	True
Sample_239	alpha	True
Sample_240	alpha	True
Sample_241	alpha	True
Sample_242	alpha	True
Sample_243	alpha	True
Sample_244	alpha	True
Sample_245	alpha	True
Sample_246	alpha	True
Sample_247	alpha	True
Sample_248	alpha	True
Sample_249	alpha	True
Sample_250	alpha	True
Sample_251	alpha	True
Sample_252	alpha	True
Sample_253	alpha	True
Sample_255	alpha	True
Sample_256	alpha	True
Sample_257	alpha	True
Sample_258	alpha	True
Sample_259	alpha	True
Sample_260	alpha	True
Sample_261	alpha	True
Sample_262	alpha	True
Sample_263	alpha	True
Sample_264	alpha	True
Sample_265	alpha	True
Sample_266	alpha	True
Sample_267	alpha	True
Sample_268	alpha	True
Sample_269	alpha	True
Sample_270	alpha	True
Sample_272	alpha	True
Sample_273	alpha	True
Sample_274	alpha	True
Sample_275	alpha	True
Sample_276	alpha	True
Sample_277	alpha	True
Sample_279	alpha	True
Sample_280	alpha	True
Sample_281	alpha	True
Sample_282	alpha	True
Sample_283	alpha	True
Sample_284	alpha	True
Sample_285	alpha	True
Sample_286	alpha	True
Sample_287	alpha	True
Sample_290	alpha	True
Sample_291	alpha	True
Sample_292	alpha	True
Sample_293	alpha	True
Sample_294	alpha	True
Sample_295	alpha	True
Sample_296	alpha	True
Sample_297	alpha	True
Sample_298	alpha	True
Sample_299	alpha	True
Sample_300	alpha	True
Sample_301	alpha	True
Sample_302	alpha	True
Sample_303	alpha	True
Sample_304	alpha	True
Sample_305	alpha	True
Sample_306	alpha	True
Sample_307	alpha	True
Sample_308	alpha	True
Sample_309	alpha	True
Sample_310	alpha	True
Sample_311	alpha	True
Sample_312	alpha	True
Sample_313	alpha	True
Sample_314	alpha	True
Sample_315	alpha	True
Sample_316	alpha	True
Sample_317	alpha	True
Sample_318	alpha	True
Sample_319	alpha	True
Sample_320	alpha	True
Sample_321	alpha	True
Sample_322	alpha	True
Sample_323	alpha	True
Sample_324	alpha	True
Sample_325	alpha	True
Sample_326	alpha	True
Sample_327	alpha	True
Sample_328	alpha	True
Sample_329	alpha	True
Sample_330	alpha	True
Sample_332	alpha	True
Sample_333	alpha	True
Sample_334	alpha	True
Sample_335	alpha	True
Sample_336	alpha	True
Sample_337	alpha	True
Sample_338	alpha	True
Sample_340	alpha	True
Sample_341	alpha	True
Sample_342	alpha	True
Sample_343	alpha	True
Sample_344	alpha	True
Sample_345	alpha	True
Sample_346	alpha	True
Sample_347	alpha	True
Sample_348	alpha	True
Sample_349	alpha	True
Sample_350	alpha	True
Sample_351	alpha	True
Sample_352	alpha	True
Sample_353	alpha	True
Sample_354	alpha	True
Sample_355	alpha	True
Sample_356	alpha	True
Sample_357	alpha	True
Sample_358	alpha	True
Sample_359	alpha	True
Sample_360	alpha	True
Sample_361	alpha	True
Sample_362	alpha	True
Sample_363	alpha	True
Sample_364	alpha	True
Sample_365	alpha	True
Sample_366	alpha	True
Sample_367	alpha	True
Sample_368	alpha	True
Sample_369	alpha	True
Sample_370	alpha	True
Sample_372	alpha	True
Sample_373	alpha	True
Sample_374	alpha	True
Sample_375	alpha	True
Sample_376	alpha	True
Sample_377	alpha	True
Sample_378	alpha	True
Sample_379	alpha	True
Sample_380	alpha	True
Sample_381	alpha	True
Sample_382	alpha	True
Sample_383	alpha	True
Sample_384	alpha	True
Sample_385	alpha	True
Sample_386	alpha	True
Sample_387	alpha	True
Sample_388	alpha	True
Sample_389	alpha	True
Sample_390	alpha	True
Sample_391	alpha	True
Sample_392	alpha	True
Sample_393	alpha	True
Sample_394	alpha	True
Sample_395	alpha	True
Sample_396	alpha	True
Sample_397	alpha	True
Sample_398	alpha	True
Sample_399	alpha	True
Sample_400	alpha	True
Sample_401	alpha	True
Sample_402	alpha	True
Sample_403	alpha	True
Sample_404	alpha	True
Sample_405	alpha	True
Sample_406	alpha	True
Sample_407	alpha	True
Sample_408	alpha	True
Sample_409	alpha	True
Sample_410	alpha	True
Sample_411	alpha	True
Sample_412	alpha	True
Sample_413	alpha	True
Sample_414	alpha	True
Sample_416	alpha	True
Sample_417	alpha	True
Sample_418	alpha	True
Sample_419	alpha	True
Sample_421	alpha	True
Sample_422	alpha	True
Sample_423	alpha	True
Sample_424	alpha	True
Sample_425	alpha	True
Sample_426	alpha	True
Sample_427	alpha	True
Sample_428	alpha	True
Sample_429	alpha	True
Sample_430	alpha	True
Sample_431	alpha	True
Sample_432	alpha	True
Sample_433	alpha	True
Sample_434	alpha	True
Sample_435	alpha	True
Sample_438	alpha	True
Sample_439	alpha	True
Sample_440	alpha	True
Sample_441	alpha	True
Sample_442	alpha	True
Sample_443	alpha	True
Sample_444	alpha	True
Sample_445	alpha	True
Sample_446	alpha	True
Sample_447	alpha	True
Sample_448	alpha	True
Sample_449	alpha	True
Sample_450	alpha	True
Sample_451	alpha	True
Sample_452	alpha	True
Sample_453	alpha	True
Sample_454	alpha	True
Sample_455	alpha	True
Sample_456	alpha	True
Sample_457	alpha	True
Sample_458	alpha	True
Sample_459	alpha	True
Sample_460	alpha	True
Sample_461	alpha	True
Sample_462	alpha	True
Sample_463	alpha	True
Sample_464	alpha	True
Sample_465	alpha	True
Sample_466	alpha	True
Sample_467	alpha	True
Sample_468	alpha	True
Sample_469	alpha	True
Sample_470	alpha	True
Sample_471	alpha	True
Sample_472	alpha	True
Sample_473	alpha	True
Sample_474	alpha	True
Sample_475	alpha	True
Sample_476	alpha	True
Sample_477	alpha	True
Sample_478	alpha	True
Sample_479	alpha	True
Sample_480	alpha	True
Sample_481	alpha	True
Sample_482	alpha	True
Sample_483	alpha	True
Sample_484	alpha	True
Sample_485	alpha	True
Sample_486	alpha	True
Sample_487	alpha	True
Sample_489	alpha	True
Sample_491	alpha	True
Sample_492	alpha	True
Sample_493	alpha	True
Sample_494	alpha	True
Sample_495	alpha	True
Sample_496	alpha	True
Sample_497	alpha	True
Sample_498	alpha	True
Sample_499	alpha	True
Sample_500	alpha	True
Sample_501	alpha	True
Sample_502	alpha	True
Sample_503	alpha	True
Sample_504	alpha	True
Sample_505	alpha	True
Sample_506	alpha	True
Sample_507	alpha	True
Sample_508	alpha	True
Sample_509	alpha	True
Sample_510	alpha	True
Sample_511	alpha	True
Sample_513	alpha	True
Sample_514	alpha	True
Sample_515	alpha	True
Sample_516	alpha	True
Sample_517	alpha	True
Sample_518	alpha	True
Sample_519	alpha	True
Sample_520	alpha	True
Sample_521	alpha	True
Sample_522	alpha	True
Sample_523	alpha	True
Sample_524	alpha	True
Sample_525	alpha	True
Sample_526	alpha	True
Sample_527	alpha	True
Sample_528	alpha	True
Sample_530	alpha	True
Sample_532	alpha	True
Sample_533	alpha	True
Sample_534	alpha	True
Sample_535	alpha	True
Sample_536	alpha	True
Sample_537	alpha	True
Sample_538	alpha	True
Sample_539	alpha	True
Sample_540	alpha	True
Sample_541	alpha	True
Sample_542	alpha	True
Sample_543	alpha	True
Sample_544	alpha	True
Sample_546	alpha	True
Sample_547	alpha	True
Sample_548	alpha	True
Sample_549	alpha	True
Sample_550	alpha	True
Sample_551	alpha	True
Sample_552	alpha	True
Sample_554	alpha	True
Sample_555	alpha	True
Sample_556	alpha	True
Sample_557	alpha	True
Sample_558	alpha	True
Sample_559	alpha	True
Sample_560	alpha	True
Sample_561	alpha	True
Sample_562	alpha	True
Sample_563	alpha	True
Sample_564	alpha	True
Sample_565	alpha	True
Sample_566	alpha	True
Sample_567	alpha	True
Sample_568	alpha	True
Sample_570	alpha	True
Sample_571	alpha	True
Sample_572	alpha	True
Sample_573	alpha	True
Sample_575	alpha	True
Sample_576	alpha	True
Sample_577	alpha	True
Sample_578	alpha	True
Sample_579	alpha	True
Sample_580	alpha	True
Sample_581	alpha	True
Sample_582	alpha	True
Sample_583	alpha	True
Sample_584	alpha	True
Sample_585	delta	True
Sample_586	delta	True
Sample_587	delta	True
Sample_588	delta	True
Sample_589	delta	True
Sample_590	delta	True
Sample_591	delta	True
Sample_592	delta	True
Sample_593	delta	True
Sample_594	delta	True
Sample_595	delta	True
Sample_596	delta	True
Sample_597	delta	True
Sample_598	delta	True
Sample_599	delta	True
Sample_600	delta	True
Sample_601	delta	True
Sample_602	delta	True
Sample_603	delta	True
Sample_604	delta	True
Sample_605	delta	True
Sample_606	delta	True
Sample_607	delta	True
Sample_608	delta	True
Sample_609	delta	True
Sample_610	delta	True
Sample_611	delta	True
Sample_612	delta	True
Sample_613	PP	True
Sample_614	PP	True
Sample_615	PP	True
Sample_616	PP	True
Sample_617	PP	True
Sample_618	PP	True
Sample_619	PP	True
Sample_620	PP	True
Sample_621	PP	True
Sample_622	PP	True
Sample_623	PP	True
Sample_624	PP	True
Sample_625	PP	True
Sample_626	PP	True
Sample_627	PP	True
Sample_628	PP	True
Sample_629	PP	True
Sample_630	PP	True
Sample_631	PP	True
Sample_633	PP	True
Sample_634	PP	True
Sample_635	PP	True
Sample_636	PP	True
Sample_637	PP	True
Sample_638	PP	True
Sample_639	PP	True
Sample_641	PP	True
Sample_643	PP	True
Sample_644	PP	True
Sample_646	PP	True
Sample_647	PP	True
Sample_650	PP	True
Sample_651	PP	True
Sample_652	beta	True
Sample_653	beta	True
Sample_654	beta	True
Sample_655	beta	True
Sample_656	beta	True
Sample_657	beta	True
Sample_658	beta	True
Sample_659	beta	True
Sample_660	beta	True
Sample_661	beta	True
Sample_662	beta	True
Sample_663	beta	True
Sample_664	beta	True
Sample_666	beta	True
Sample_668	beta	True
Sample_669	beta	True
Sample_670	beta	True
Sample_671	beta	True
Sample_672	beta	True
Sample_673	beta	True
Sample_674	beta	True
Sample_675	beta	True
Sample_676	beta	True
Sample_677	beta	True
Sample_678	beta	True
Sample_679	beta	True
Sample_680	beta	True
Sample_682	beta	True
Sample_683	beta	True
Sample_684	beta	True
Sample_685	beta	True
Sample_686	beta	True
Sample_687	beta	True
Sample_688	beta	True
Sample_689	beta	True
Sample_690	beta	True
Sample_691	beta	True
Sample_692	beta	True
Sample_693	beta	True
Sample_694	beta	True
Sample_695	beta	True
Sample_696	beta	True
Sample_697	beta	True
Sample_698	beta	True
Sample_699	beta	True
Sample_700	beta	True
Sample_701	beta	True
Sample_702	beta	True
Sample_703	beta	True
Sample_704	beta	True
Sample_705	beta	True
Sample_706	beta	True
Sample_707	beta	True
Sample_708	beta	True
Sample_709	beta	True
Sample_711	beta	True
Sample_712	beta	True
Sample_713	beta	True
Sample_714	beta	True
Sample_715	beta	True
Sample_716	beta	True
Sample_717	beta	True
Sample_718	beta	True
Sample_719	beta	True
Sample_721	beta	True
Sample_722	beta	True
Sample_723	beta	True
Sample_724	beta	True
Sample_725	beta	True
Sample_726	beta	True
Sample_727	beta	True
Sample_728	beta	True
Sample_729	beta	True
Sample_730	beta	True
Sample_732	beta	True
Sample_733	beta	True
Sample_734	beta	True
Sample_736	beta	True
Sample_737	beta	True
Sample_739	beta	True
Sample_740	beta	True
Sample_741	beta	True
Sample_742	beta	True
Sample_743	beta	True
Sample_744	beta	True
Sample_745	beta	True
Sample_746	beta	True
Sample_747	beta	True
Sample_748	beta	True
Sample_749	beta	True
Sample_750	beta	True
Sample_751	beta	True
Sample_752	beta	True
Sample_753	beta	True
Sample_754	beta	True
Sample_755	beta	True
Sample_756	beta	True
Sample_757	beta	True
Sample_758	beta	True
Sample_759	beta	True
Sample_760	beta	True
Sample_761	beta	True
Sample_762	beta	True
Sample_763	beta	True
Sample_764	beta	True
Sample_765	beta	True
Sample_766	beta	True
Sample_767	beta	True
Sample_769	beta	True
Sample_770	beta	True
Sample_771	beta	True
Sample_772	beta	True
Sample_773	beta	True
Sample_774	beta	True
Sample_775	beta	True
Sample_776	beta	True
Sample_777	beta	True
Sample_778	beta	True
Sample_779	beta	True
Sample_780	beta	True
Sample_781	beta	True
Sample_782	beta	True
Sample_783	beta	True
Sample_784	beta	True
Sample_785	beta	True
Sample_786	beta	True
Sample_787	beta	True
Sample_788	beta	True
Sample_789	beta	True
Sample_790	beta	True
Sample_791	beta	True
Sample_792	beta	True
Sample_793	beta	True
Sample_794	beta	True
Sample_795	beta	True
Sample_796	beta	True
Sample_797	beta	True
Sample_798	beta	True
Sample_799	beta	True
Sample_800	beta	True
Sample_801	beta	True
Sample_802	beta	True
Sample_803	beta	True
Sample_804	beta	True
Sample_805	beta	True
Sample_806	beta	True
Sample_807	beta	True
Sample_808	beta	True
Sample_809	beta	True
Sample_810	beta	True
Sample_811	beta	True
Sample_812	beta	True
Sample_813	beta	True
Sample_814	beta	True
Sample_815	beta	True
Sample_816	beta	True
Sample_817	beta	True
Sample_818	beta	True
Sample_819	beta	True
Sample_821	beta	True
Sample_823	beta	True
Sample_824	beta	True
Sample_825	beta	True
Sample_826	beta	True
Sample_827	beta	True
Sample_828	beta	True
Sample_829	beta	True
Sample_830	beta	True
Sample_831	beta	True
Sample_832	beta	True
Sample_833	beta	True
Sample_834	beta	True
Sample_835	beta	True
Sample_837	beta	True
Sample_838	beta	True
Sample_839	beta	True
Sample_840	beta	True
Sample_841	beta	True
Sample_842	beta	True
Sample_843	beta	True
Sample_844	beta	True
Sample_845	beta	True
Sample_846	beta	True
Sample_847	beta	True
Sample_848	beta	True
Sample_849	beta	True
Sample_850	beta	True
Sample_851	beta	True
Sample_852	beta	True
Sample_853	beta	True
Sample_854	beta	True
Sample_855	beta	True
Sample_856	beta	True
Sample_857	beta	True
Sample_858	beta	True
Sample_859	beta	True
Sample_860	beta	True
Sample_861	beta	True
Sample_862	beta	True
Sample_863	beta	True
Sample_864	beta	True
Sample_866	beta	True
Sample_868	beta	True
Sample_869	beta	True
Sample_870	beta	True
Sample_871	beta	True
Sample_872	beta	True
Sample_873	beta	True
Sample_874	beta	True
Sample_875	beta	True
Sample_876	beta	True
Sample_877	beta	True
Sample_879	beta	True
Sample_880	beta	True
Sample_881	beta	True
Sample_883	beta	True
Sample_884	beta	True
Sample_885	beta	True
Sample_886	beta	True
Sample_887	beta	True
Sample_888	beta	True
Sample_889	beta	True
Sample_890	beta	True
Sample_891	beta	True
Sample_892	beta	True
Sample_893	beta	True
Sample_895	beta	True
Sample_896	beta	True
Sample_897	beta	True
Sample_898	beta	True
Sample_899	beta	True
Sample_901	beta	True
Sample_902	beta	True
Sample_903	beta	True
Sample_904	beta	True
Sample_905	beta	True
Sample_906	beta	True
Sample_907	beta	True
Sample_908	beta	True
Sample_909	beta	True
Sample_910	beta	True
Sample_911	beta	True
Sample_912	beta	True
Sample_913	beta	True
Sample_914	beta	True
Sample_915	beta	True
Sample_916	beta	True
Sample_917	beta	True
Sample_918	beta	True
Sample_919	beta	True
Sample_920	beta	True
Sample_921	beta	True
Sample_922	beta	True
Sample_923	beta	True
Sample_924	beta	True
Sample_925	beta	True
Sample_926	beta	True
Sample_927	beta	True
Sample_928	beta	True
Sample_929	beta	True
Sample_930	beta	True
Sample_931	beta	True
Sample_932	beta	True
Sample_933	beta	True
Sample_934	beta	True
Sample_935	beta	True
Sample_936	beta	True
Sample_937	beta	True
Sample_938	beta	True
Sample_939	beta	True
Sample_940	beta	True
Sample_941	beta	True
Sample_942	beta	True
Sample_943	beta	True
Sample_944	beta	True
Sample_945	beta	True
Sample_946	beta	True
Sample_947	beta	True
Sample_948	alpha	True
Sample_949	alpha	True
Sample_950	alpha	True
Sample_951	alpha	True
Sample_952	alpha	True
Sample_953	alpha	True
Sample_954	alpha	True
Sample_955	alpha	True
Sample_956	alpha	True
Sample_957	alpha	True
Sample_958	alpha	True
Sample_959	alpha	True
Sample_960	alpha	True
Sample_961	alpha	True
Sample_962	alpha	True
Sample_963	alpha	True
Sample_964	alpha	True
Sample_965	alpha	True
Sample_966	alpha	True
Sample_967	alpha	True
Sample_968	alpha	True
Sample_969	alpha	True
Sample_970	alpha	True
Sample_971	alpha	True
Sample_972	alpha	True
Sample_973	alpha	True
Sample_974	alpha	True
Sample_976	alpha	True
Sample_977	alpha	True
Sample_978	alpha	True
Sample_979	alpha	True
Sample_980	alpha	True
Sample_981	alpha	True
Sample_982	alpha	True
Sample_983	alpha	True
Sample_984	alpha	True
Sample_985	alpha	True
Sample_987	alpha	True
Sample_988	alpha	True
Sample_989	alpha	True
Sample_990	alpha	True
Sample_991	alpha	True
Sample_992	alpha	True
Sample_993	alpha	True
Sample_994	alpha	True
Sample_995	alpha	True
Sample_996	alpha	True
Sample_997	alpha	True
Sample_998	alpha	True
Sample_999	alpha	True
Sample_1000	alpha	True
Sample_1001	alpha	True
Sample_1002	alpha	True
Sample_1003	alpha	True
Sample_1005	alpha	True
Sample_1006	alpha	True
Sample_1007	alpha	True
Sample_1008	alpha	True
Sample_1009	alpha	True
Sample_1010	alpha	True
Sample_1011	alpha	True
Sample_1012	alpha	True
Sample_1013	alpha	True
Sample_1014	alpha	True
Sample_1015	alpha	True
Sample_1016	alpha	True
Sample_1017	alpha	True
Sample_1018	alpha	True
Sample_1020	alpha	True
Sample_1022	alpha	True
Sample_1023	alpha	True
Sample_1024	alpha	True
Sample_1025	alpha	True
Sample_1026	alpha	True
Sample_1027	alpha	True
Sample_1028	alpha	True
Sample_1029	alpha	True
Sample_1030	alpha	True
Sample_1032	alpha	True
Sample_1033	alpha	True
Sample_1034	alpha	True
Sample_1035	alpha	True
Sample_1036	alpha	True
Sample_1037	alpha	True
Sample_1038	alpha	True
Sample_1039	alpha	True
Sample_1040	alpha	True
Sample_1041	alpha	True
Sample_1042	alpha	True
Sample_1043	alpha	True
Sample_1044	alpha	True
Sample_1046	alpha	True
Sample_1047	alpha	True
Sample_1049	alpha	True
Sample_1050	alpha	True
Sample_1051	alpha	True
Sample_1052	alpha	True
Sample_1053	alpha	True
Sample_1054	alpha	True
Sample_1055	alpha	True
Sample_1056	alpha	True
Sample_1057	alpha	True
Sample_1058	alpha	True
Sample_1059	alpha	True
Sample_1060	alpha	True
Sample_1061	alpha	True
Sample_1062	alpha	True
Sample_1063	alpha	True
Sample_1064	alpha	True
Sample_1065	alpha	True
Sample_1066	alpha	True
Sample_1067	alpha	True
Sample_1068	alpha	True
Sample_1069	alpha	True
Sample_1070	alpha	True
Sample_1071	alpha	True
Sample_1072	alpha	True
Sample_1073	alpha	True
Sample_1074	alpha	True
Sample_1075	alpha	True
Sample_1076	alpha	True
Sample_1077	alpha	True
Sample_1078	alpha	True
Sample_1079	alpha	True
Sample_1080	alpha	True
Sample_1081	alpha	True
Sample_1082	alpha	True
Sample_1083	alpha	True
Sample_1084	alpha	True
Sample_1085	alpha	True
Sample_1087	alpha	True
Sample_1090	alpha	True
Sample_1091	alpha	True
Sample_1092	alpha	True
Sample_1093	alpha	True
Sample_1094	alpha	True
Sample_1095	alpha	True
Sample_1096	alpha	True
Sample_1097	alpha	True
Sample_1098	alpha	True
Sample_1099	alpha	True
Sample_1100	alpha	True
Sample_1101	alpha	True
Sample_1102	alpha	True
Sample_1103	alpha	True
Sample_1104	alpha	True
Sample_1105	alpha	True
Sample_1106	alpha	True
Sample_1107	alpha	True
Sample_1108	alpha	True
Sample_1109	alpha	True
Sample_1110	alpha	True
Sample_1111	alpha	True
Sample_1112	alpha	True
Sample_1113	alpha	True
Sample_1114	alpha	True
Sample_1115	alpha	True
Sample_1116	alpha	True
Sample_1117	alpha	True
Sample_1118	alpha	True
Sample_1119	alpha	True
Sample_1121	alpha	True
Sample_1123	alpha	True
Sample_1124	alpha	True
Sample_1125	alpha	True
Sample_1126	alpha	True
Sample_1127	alpha	True
Sample_1128	alpha	True
Sample_1129	alpha	True
Sample_1130	alpha	True
Sample_1131	alpha	True
Sample_1133	alpha	True
Sample_1135	alpha	True
Sample_1136	alpha	True
Sample_1138	alpha	True
Sample_1139	alpha	True
Sample_1141	alpha	True
Sample_1142	alpha	True
Sample_1143	alpha	True
Sample_1144	alpha	True
Sample_1145	alpha	True
Sample_1146	alpha	True
Sample_1147	alpha	True
Sample_1148	alpha	True
Sample_1149	alpha	True
Sample_1150	alpha	True
Sample_1151	alpha	True
Sample_1152	alpha	True
Sample_1153	alpha	True
Sample_1154	alpha	True
Sample_1155	alpha	True
Sample_1156	alpha	True
Sample_1157	alpha	True
Sample_1158	alpha	True
Sample_1159	alpha	True
Sample_1160	alpha	True
Sample_1161	alpha	True
Sample_1162	alpha	True
Sample_1163	alpha	True
Sample_1164	alpha	True
Sample_1165	alpha	True
Sample_1166	alpha	True
Sample_1167	alpha	True
Sample_1168	alpha	True
Sample_1169	alpha	True
Sample_1170	alpha	True
Sample_1171	alpha	True
Sample_1172	alpha	True
Sample_1173	alpha	True
Sample_1174	alpha	True
Sample_1175	alpha	True
Sample_1176	alpha	True
Sample_1177	alpha	True
Sample_1178	alpha	True
Sample_1179	alpha	True
Sample_1180	alpha	True
Sample_1181	alpha	True
Sample_1182	alpha	True
Sample_1185	alpha	True
Sample_1186	alpha	True
Sample_1187	alpha	True
Sample_1188	alpha	True
Sample_1189	alpha	True
Sample_1190	alpha	True
Sample_1191	alpha	True
Sample_1192	alpha	True
Sample_1193	alpha	True
Sample_1194	alpha	True
Sample_1195	alpha	True
Sample_1196	alpha	True
Sample_1197	alpha	True
Sample_1198	alpha	True
Sample_1199	alpha	True
Sample_1200	alpha	True
Sample_1201	alpha	True
Sample_1202	alpha	True
Sample_1203	alpha	True
Sample_1204	alpha	True
Sample_1205	alpha	True
Sample_1206	alpha	True
Sample_1207	alpha	True
Sample_1208	alpha	True
Sample_1209	alpha	True
Sample_1210	alpha	True
Sample_1211	alpha	True
Sample_1212	alpha	True
Sample_1213	alpha	True
Sample_1214	alpha	True
Sample_1215	alpha	True
Sample_1216	alpha	True
Sample_1217	alpha	True
Sample_1218	alpha	True
Sample_1219	alpha	True
Sample_1220	alpha	True
Sample_1221	alpha	True
Sample_1222	alpha	True
Sample_1223	alpha	True
Sample_1224	alpha	True
Sample_1225	alpha	True
Sample_1226	alpha	True
Sample_1227	alpha	True
Sample_1228	alpha	True
Sample_1229	alpha	True
Sample_1230	alpha	True
Sample_1231	alpha	True
Sample_1232	alpha	True
Sample_1233	alpha	True
Sample_1234	alpha	True
Sample_1235	alpha	True
Sample_1236	alpha	True
Sample_1237	alpha	True
Sample_1238	alpha	True
Sample_1239	alpha	True
Sample_1240	alpha	True
Sample_1241	alpha	True
Sample_1242	alpha	True
Sample_1243	alpha	True
Sample_1244	alpha	True
Sample_1245	alpha	True
Sample_1246	alpha	True
Sample_1247	alpha	True
Sample_1248	alpha	True
Sample_1249	alpha	True
Sample_1250	alpha	True
Sample_1251	alpha	True
Sample_1252	alpha	True
Sample_1253	alpha	True
Sample_1254	alpha	True
Sample_1256	alpha	True
Sample_1257	alpha	True
Sample_1258	alpha	True
Sample_1259	alpha	True
Sample_1260	alpha	True
Sample_1261	alpha	True
Sample_1262	alpha	True
Sample_1263	alpha	True
Sample_1264	alpha	True
Sample_1265	alpha	True
Sample_1266	alpha	True
Sample_1267	alpha	True
Sample_1269	alpha	True
Sample_1270	alpha	True
Sample_1271	alpha	True
Sample_1272	alpha	True
Sample_1273	alpha	True
Sample_1274	alpha	True
Sample_1275	alpha	True
Sample_1276	alpha	True
Sample_1277	alpha	True
Sample_1278	alpha	True
Sample_1279	alpha	True
Sample_1280	alpha	True
Sample_1281	alpha	True
Sample_1282	alpha	True
Sample_1283	alpha	True
Sample_1284	alpha	True
Sample_1285	alpha	True
Sample_1286	alpha	True
Sample_1287	alpha	True
Sample_1288	alpha	True
Sample_1289	alpha	True
Sample_1290	alpha	True
Sample_1291	alpha	True
Sample_1292	alpha	True
Sample_1293	alpha	True
Sample_1294	alpha	True
Sample_1295	alpha	True
Sample_1297	alpha	True
Sample_1298	alpha	True
Sample_1299	alpha	True
Sample_1300	alpha	True
Sample_1301	alpha	True
Sample_1302	alpha	True
Sample_1303	alpha	True
Sample_1304	alpha	True
Sample_1305	alpha	True
Sample_1307	alpha	True
Sample_1308	alpha	True
Sample_1309	alpha	True
Sample_1311	alpha	True
Sample_1312	alpha	True
Sample_1313	alpha	True
Sample_1314	alpha	True
Sample_1315	alpha	True
Sample_1316	alpha	True
Sample_1317	alpha	True
Sample_1318	alpha	True
Sample_1319	alpha	True
Sample_1320	alpha	True
Sample_1321	alpha	True
Sample_1322	alpha	True
Sample_1323	alpha	True
Sample_1324	alpha	True
Sample_1325	alpha	True
Sample_1326	alpha	True
Sample_1327	alpha	True
Sample_1328	alpha	True
Sample_1329	alpha	True
Sample_1330	alpha	True
Sample_1331	alpha	True
Sample_1332	alpha	True
Sample_1333	alpha	True
Sample_1334	alpha	True
Sample_1335	alpha	True
Sample_1336	alpha	True
Sample_1338	alpha	True
Sample_1340	alpha	True
Sample_1341	alpha	True
Sample_1342	alpha	True
Sample_1343	alpha	True
Sample_1344	alpha	True
Sample_1346	alpha	True
Sample_1347	alpha	True
Sample_1348	alpha	True
Sample_1349	alpha	True
Sample_1350	alpha	True
Sample_1351	alpha	True
Sample_1352	alpha	True
Sample_1353	alpha	True
Sample_1354	alpha	True
Sample_1355	alpha	True
Sample_1356	alpha	True
Sample_1357	alpha	True
Sample_1358	alpha	True
Sample_1359	alpha	True
Sample_1360	alpha	True
Sample_1361	alpha	True
Sample_1362	alpha	True
Sample_1363	alpha	True
Sample_1364	alpha	True
Sample_1365	alpha	True
Sample_1366	alpha	True
Sample_1367	alpha	True
Sample_1368	alpha	True
Sample_1369	alpha	True
Sample_1370	alpha	True
Sample_1371	alpha	True
Sample_1372	alpha	True
Sample_1373	alpha	True
Sample_1374	alpha	True
Sample_1375	alpha	True
Sample_1377	alpha	True
Sample_1378	alpha	True
Sample_1379	alpha	True
Sample_1380	alpha	True
Sample_1381	alpha	True
Sample_1382	alpha	True
Sample_1383	alpha	True
Sample_1384	alpha	True
Sample_1385	alpha	True
Sample_1386	alpha	True
Sample_1387	alpha	True
Sample_1388	alpha	True
Sample_1389	alpha	True
Sample_1390	alpha	True
Sample_1391	alpha	True
Sample_1392	alpha	True
Sample_1393	alpha	True
Sample_1394	alpha	True
Sample_1395	alpha	True
Sample_1396	alpha	True
Sample_1397	alpha	True
Sample_1398	alpha	True
Sample_1399	alpha	True
Sample_1401	alpha	True
Sample_1402	alpha	True
Sample_1403	alpha	True
Sample_1404	alpha	True
Sample_1405	alpha	True
Sample_1406	alpha	True
Sample_1407	alpha	True
Sample_1408	alpha	True
Sample_1409	alpha	True
Sample_1410	alpha	True
Sample_1411	alpha	True
Sample_1412	alpha	True
Sample_1414	alpha	True
Sample_1415	alpha	True
Sample_1417	alpha	True
Sample_1418	alpha	True
Sample_1419	alpha	True
Sample_1420	alpha	True
Sample_1421	alpha	True
Sample_1422	alpha	True
Sample_1423	alpha	True
Sample_1424	alpha	True
Sample_1425	alpha	True
Sample_1426	alpha	True
Sample_1427	alpha	True
Sample_1428	alpha	True
Sample_1429	alpha	True
Sample_1430	alpha	True
Sample_1431	alpha	True
Sample_1432	alpha	True
Sample_1433	alpha	True
Sample_1434	alpha	True
Sample_1435	alpha	True
Sample_1436	alpha	True
Sample_1437	alpha	True
Sample_1438	alpha	True
Sample_1439	alpha	True
Sample_1440	alpha	True
Sample_1441	alpha	True
Sample_1442	alpha	True
Sample_1443	alpha	True
Sample_1444	alpha	True
Sample_1445	alpha	True
Sample_1446	alpha	True
Sample_1447	alpha	True
Sample_1448	alpha	True
Sample_1449	alpha	True
Sample_1450	alpha	True
Sample_1451	alpha	True
Sample_1452	alpha	True
Sample_1453	alpha	True
Sample_1454	alpha	True
Sample_1455	alpha	True
Sample_1456	alpha	True
Sample_1457	alpha	True
Sample_1458	alpha	True
Sample_1459	alpha	True
Sample_1461	alpha	True
Sample_1462	alpha	True
Sample_1463	alpha	True
Sample_1464	alpha	True
Sample_1465	alpha	True
Sample_1466	alpha	True
Sample_1467	alpha	True
Sample_1468	alpha	True
Sample_1469	alpha	True
Sample_1470	alpha	True
Sample_1471	alpha	True
Sample_1472	alpha	True
Sample_1473	alpha	True
Sample_1474	alpha	True
Sample_1475	alpha	True
Sample_1476	alpha	True
Sample_1478	alpha	True
Sample_1479	alpha	True
Sample_1480	alpha	True
Sample_1481	alpha	True
Sample_1482	alpha	True
Sample_1483	alpha	True
Sample_1484	alpha	True
Sample_1485	alpha	True
Sample_1487	alpha	True
Sample_1488	alpha	True
Sample_1489	alpha	True
Sample_1490	alpha	True
Sample_1491	alpha	True
Sample_1492	alpha	True
Sample_1493	alpha	True
Sample_1495	alpha	True
Sample_1496	alpha	True
Sample_1497	alpha	True
Sample_1498	alpha	True
Sample_1499	alpha	True
Sample_1500	alpha	True
Sample_1501	alpha	True
Sample_1502	alpha	True
Sample_1503	alpha	True
Sample_1506	alpha	True
Sample_1507	alpha	True
Sample_1508	alpha	True
Sample_1509	alpha	True
Sample_1510	alpha	True
Sample_1511	alpha	True
Sample_1512	alpha	True
Sample_1513	alpha	True
Sample_1514	alpha	True
Sample_1515	alpha	True
Sample_1516	alpha	True
Sample_1517	delta	True
Sample_1518	delta	True
Sample_1519	delta	True
Sample_1522	delta	True
Sample_1523	delta	True
Sample_1525	delta	True
Sample_1526	delta	True
Sample_1527	delta	True
Sample_1528	delta	True
Sample_1529	delta	True
Sample_1533	delta	True
Sample_1534	delta	True
Sample_1535	delta	True
Sample_1536	delta	True
Sample_1537	delta	True
Sample_1538	delta	True
Sample_1539	delta	True
Sample_1540	delta	True
Sample_1541	delta	True
Sample_1542	delta	True
Sample_1546	delta	True
Sample_1547	PP	True
Sample_1548	PP	True
Sample_1549	PP	True
Sample_1550	PP	True
Sample_1551	PP	True
Sample_1552	PP	True
Sample_1553	PP	True
Sample_1554	PP	True
Sample_1555	PP	True
Sample_1556	PP	True
Sample_1557	PP	True
Sample_1558	PP	True
Sample_1559	PP	True
Sample_1560	PP	True
Sample_1561	PP	True
Sample_1562	PP	True
Sample_1563	PP	True
Sample_1564	PP	True
Sample_1565	PP	True
Sample_1566	PP	True
Sample_1567	PP	True
Sample_1568	PP	True
Sample_1569	PP	True
Sample_1570	PP	True
Sample_1571	PP	True
Sample_1572	PP	True
Sample_1573	PP	True
Sample_1574	PP	True
Sample_1575	PP	True
Sample_1576	PP	True
Sample_1577	PP	True
Sample_1578	PP	True
Sample_1579	PP	True
Sample_1580	PP	True
Sample_1581	PP	True
Sample_1582	PP	True
Sample_1583	PP	True
Sample_1584	PP	True
Sample_1585	PP	True
Sample_1586	PP	True
Sample_1587	PP	True
Sample_1588	PP	True
Sample_1589	PP	True
Sample_1590	PP	True
Sample_1591	PP	True
Sample_1592	PP	True
Sample_1593	PP	True
Sample_1594	PP	True
Sample_1595	PP	True
Sample_1597	PP	True
Sample_1598	PP	True
Sample_1600	PP	True
```
<!-- END_RESTORE_FILE -->


</details>

## file-fb70be9a84cd

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/metrics.json` · SHA256: `5a2a0488d8bbb0a6ea55c2c6db6a04b97e0c6ad1819a6bc0c3434a276a47f2c8`

<details>
<summary>metrics.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/metrics.json", "sha256": "5a2a0488d8bbb0a6ea55c2c6db6a04b97e0c6ad1819a6bc0c3434a276a47f2c8", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "cells_total": 4218,
  "cells_evaluated_known_labels": 4213,
  "cells_without_known_training_label": 5,
  "forward_seconds": 15.474318906897679,
  "cells_per_second": 272.5806560778469,
  "timing_scope": "cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving",
  "gpu_peak_allocated_gib": 0.3811001777648926,
  "gpu_peak_reserved_gib": 0.6171875,
  "python_process_rss_peak_observed_gib": 1.7534446716308594,
  "loss": 0.23071691393852234,
  "accuracy": 0.9380488962734393,
  "macro_f1_all_training_classes": 0.5349519490076795,
  "macro_f1_present_true_classes": 0.6954375337099833,
  "weighted_f1": 0.9311712408040055
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-fb90e4941c29

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/prediction_config.json` · SHA256: `697e88dc28849383487c72aa401016d1e6424796ea332b6a6b44b0451f6e2c13`

<details>
<summary>prediction_config.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/prediction_config.json", "sha256": "697e88dc28849383487c72aa401016d1e6424796ea332b6a6b44b0451f6e2c13", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "bundle": "/workspace/scgpt/models/finetuned/finetune_20261023_033701_500560/epoch_001",
  "model_sha256": "4e7881b22185b278a44f6052ce478cc95a0c7a57249de48f047b830eea190e70",
  "input": "/workspace/scgpt/data/pancreas_demo/demo_test.h5ad",
  "input_sha256": "6fd33b1d8e5a9b3eb515ce3252869e32db7e5088edf8b9302b0ab698fdab5b29",
  "seed": 42,
  "batch_size": 8,
  "precision": "FP32",
  "preprocessing": {
    "pad_token": "<pad>",
    "pad_value": -2,
    "n_bins": 51,
    "max_length_including_cls": 1200,
    "additional_normalize_total": false,
    "additional_log1p": false,
    "expression_provenance": "Supplied h5ad values; no raw layer or preprocessing metadata",
    "tokenization": "nonzero genes only; prepend CLS; official per-cell quantile binning; random overlength sampling; dynamic padding",
    "pooling": "raw CLS features into official cls_decoder; no L2 normalization",
    "prediction_genes": "require every saved training gene; reorder to training gene order; ignore additional genes"
  },
  "unknown_label_counts": {
    "MHC class II": 5
  },
  "unknown_policy": "retain all predictions; evaluate known ground-truth labels only; no unknown-class detector"
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-30bb869dc571

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/predictions.tsv` · SHA256: `c12fb3537eb706b91164b8dc216b44ffc306187a37312d064c6a0acc77168008`

<details>
<summary>predictions.tsv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/predictions.tsv", "sha256": "c12fb3537eb706b91164b8dc216b44ffc306187a37312d064c6a0acc77168008", "encoding": "utf-8", "added_final_newline": false} -->
```tsv
cell_id	true_label_id	true_label	predicted_label_id	predicted_label	max_probability	included_in_metrics
10th_C11_S96	4	beta	4	beta	0.9980323	True
10th_C13_S61	4	beta	4	beta	0.99796224	True
10th_C14_S53	4	beta	4	beta	0.998028	True
10th_C17_S97	4	beta	4	beta	0.9978405	True
10th_C19_S62	1	PSC	1	PSC	0.99367684	True
10th_C1_S59	4	beta	4	beta	0.99773383	True
10th_C20_S54	4	beta	4	beta	0.99767953	True
10th_C23_S98	4	beta	4	beta	0.99779797	True
10th_C24_S90	6	ductal	6	ductal	0.96797913	True
10th_C28_S91	3	alpha	3	alpha	0.99775416	True
10th_C30_S107	3	alpha	3	alpha	0.9975872	True
10th_C31_S48	3	alpha	3	alpha	0.9976356	True
10th_C32_S56	4	beta	4	beta	0.9968836	True
10th_C33_S64	4	beta	4	beta	0.99791664	True
10th_C34_S92	3	alpha	3	alpha	0.9972203	True
10th_C37_S49	4	beta	4	beta	0.9980453	True
10th_C41_S101	3	alpha	3	alpha	0.9975841	True
10th_C42_S109	2	acinar	2	acinar	0.9925478	True
10th_C43_S50	3	alpha	3	alpha	0.99776065	True
10th_C45_S66	3	alpha	3	alpha	0.9977343	True
10th_C4_S103	6	ductal	6	ductal	0.9941964	True
10th_C50_S73	3	alpha	3	alpha	0.9977793	True
10th_C51_S80	4	beta	4	beta	0.9980197	True
10th_C52_S110	3	alpha	3	alpha	0.99774206	True
10th_C53_S118	3	alpha	3	alpha	0.9976936	True
10th_C54_S124	6	ductal	6	ductal	0.9698197	True
10th_C56_S74	1	PSC	1	PSC	0.9925694	True
10th_C57_S81	3	alpha	3	alpha	0.9976659	True
10th_C5_S95	3	alpha	3	alpha	0.99778616	True
10th_C61_S68	4	beta	4	beta	0.99757713	True
10th_C63_S82	2	acinar	2	acinar	0.9943908	True
10th_C64_S112	4	beta	4	beta	0.9978666	True
10th_C66_S126	4	beta	4	beta	0.9979802	True
10th_C74_S76	3	alpha	3	alpha	0.9976986	True
10th_C76_S128	5	delta	4	beta	0.8545319	True
10th_C77_S120	1	PSC	1	PSC	0.99359363	True
10th_C7_S60	6	ductal	6	ductal	0.992667	True
10th_C81_S70	4	beta	4	beta	0.9978168	True
10th_C84_S115	4	beta	4	beta	0.99775	True
10th_C85_S86	4	beta	4	beta	0.997891	True
10th_C86_S78	2	acinar	2	acinar	0.99418145	True
10th_C87_S71	2	acinar	2	acinar	0.9911185	True
10th_C88_S130	6	ductal	6	ductal	0.9937596	True
10th_C90_S116	3	alpha	3	alpha	0.9973568	True
10th_C95_S123	3	alpha	3	alpha	0.9977507	True
10th_C96_S117	3	alpha	3	alpha	0.99770725	True
11th-C12_S47	3	alpha	3	alpha	0.9977335	True
11th-C16_S64	1	PSC	1	PSC	0.9929368	True
11th-C21_S4	4	beta	4	beta	0.9978497	True
11th-C22_S65	4	beta	4	beta	0.9978079	True
11th-C24_S49	6	ductal	6	ductal	0.9941248	True
11th-C26_S12	6	ductal	6	ductal	0.9917167	True
11th-C27_S19	4	beta	4	beta	0.9980246	True
11th-C2_S8	1	PSC	1	PSC	0.99365944	True
11th-C30_S66	4	beta	4	beta	0.9977786	True
11th-C33_S20	5	delta	4	beta	0.5130793	True
11th-C36_S67	3	alpha	3	alpha	0.9975969	True
11th-C37_S6	3	alpha	3	alpha	0.9977168	True
11th-C40_S52	1	PSC	1	PSC	0.9915731	True
11th-C41_S60	4	beta	4	beta	0.9979401	True
11th-C43_S7	6	ductal	6	ductal	0.9924271	True
11th-C44_S15	4	beta	4	beta	0.99784887	True
11th-C49_S23	4	beta	4	beta	0.9980773	True
11th-C4_S62	5	delta	6	ductal	0.9876429	True
11th-C51_S39	4	beta	4	beta	0.9980399	True
11th-C53_S78	4	beta	4	beta	0.9977017	True
11th-C55_S24	6	ductal	6	ductal	0.98838335	True
11th-C56_S32	1	PSC	1	PSC	0.9933437	True
11th-C57_S40	3	alpha	3	alpha	0.99774253	True
11th-C58_S71	6	ductal	6	ductal	0.9936162	True
11th-C5_S54	3	alpha	3	alpha	0.9969926	True
11th-C61_S25	3	alpha	3	alpha	0.9975805	True
11th-C62_S33	3	alpha	3	alpha	0.99759704	True
11th-C64_S72	2	acinar	2	acinar	0.99375737	True
11th-C66_S88	4	beta	4	beta	0.99792	True
11th-C67_S26	3	alpha	3	alpha	0.9977558	True
11th-C68_S34	5	delta	6	ductal	0.98223025	True
11th-C77_S82	5	delta	7	endothelial	0.99195427	True
11th-C7_S16	6	ductal	6	ductal	0.99356705	True
11th-C93_S30	3	alpha	3	alpha	0.9976871	True
12th-C10_S155	4	beta	4	beta	0.9976897	True
12th-C11_S147	3	alpha	3	alpha	0.9977054	True
12th-C12_S139	4	beta	4	beta	0.99741566	True
12th-C14_S190	4	beta	4	beta	0.997672	True
12th-C15_S183	4	beta	4	beta	0.99784756	True
12th-C1_S195	3	alpha	3	alpha	0.9976452	True
12th-C21_S184	4	beta	4	beta	0.9977896	True
12th-C23_S149	3	alpha	3	alpha	0.99764603	True
12th-C24_S141	4	beta	4	beta	0.9977848	True
12th-C25_S185	4	beta	4	beta	0.9977767	True
12th-C26_S192	4	beta	4	beta	0.99771786	True
12th-C28_S142	3	alpha	3	alpha	0.9974921	True
12th-C29_S150	3	alpha	3	alpha	0.9976513	True
12th-C30_S157	4	beta	4	beta	0.997712	True
12th-C33_S200	4	beta	4	beta	0.9978186	True
12th-C35_S151	4	beta	4	beta	0.9981078	True
12th-C36_S158	1	PSC	1	PSC	0.99379575	True
12th-C37_S187	3	alpha	3	alpha	0.9978115	True
12th-C38_S194	4	beta	4	beta	0.99793774	True
12th-C39_S201	3	alpha	3	alpha	0.9976023	True
12th-C3_S181	4	beta	4	beta	0.9977214	True
12th-C40_S144	4	beta	4	beta	0.99788266	True
12th-C42_S159	4	beta	4	beta	0.997908	True
12th-C46_S145	1	PSC	1	PSC	0.99349314	True
12th-C47_S153	4	beta	4	beta	0.9978003	True
12th-C48_S160	4	beta	4	beta	0.9978434	True
12th-C4_S154	4	beta	4	beta	0.997825	True
12th-C50_S210	3	alpha	3	alpha	0.99768794	True
12th-C51_S217	4	beta	4	beta	0.99774724	True
12th-C53_S168	2	acinar	2	acinar	0.99476385	True
12th-C55_S203	4	beta	4	beta	0.9978783	True
12th-C56_S211	6	ductal	6	ductal	0.9940783	True
12th-C57_S218	3	alpha	3	alpha	0.9965661	True
12th-C58_S162	3	alpha	3	alpha	0.99773383	True
12th-C59_S169	4	beta	4	beta	0.99762946	True
12th-C60_S176	4	beta	4	beta	0.9978746	True
12th-C63_S219	3	alpha	3	alpha	0.9970879	True
12th-C64_S163	4	beta	4	beta	0.99768627	True
12th-C65_S170	4	beta	4	beta	0.997701	True
12th-C67_S205	4	beta	4	beta	0.99785405	True
12th-C71_S171	3	alpha	3	alpha	0.99773395	True
12th-C74_S214	4	beta	4	beta	0.99795353	True
12th-C75_S206	3	alpha	3	alpha	0.9975744	True
12th-C76_S177	4	beta	7	endothelial	0.9648519	True
12th-C7_S196	4	beta	4	beta	0.9979388	True
12th-C80_S215	3	alpha	3	alpha	0.997619	True
12th-C81_S207	3	alpha	3	alpha	0.9976802	True
12th-C84_S165	4	beta	4	beta	0.99782807	True
12th-C85_S137	4	beta	4	beta	0.99770975	True
12th-C8_S189	4	beta	4	beta	0.9978283	True
12th-C90_S166	4	beta	4	beta	0.99790883	True
12th-C93_S209	4	beta	4	beta	0.997672	True
12th-C94_S180	1	PSC	1	PSC	0.9936047	True
12th-C95_S174	3	alpha	3	alpha	0.9976065	True
12th-C96_S167	4	beta	4	beta	0.99779797	True
13th_C12_S178	4	beta	4	beta	0.9978992	True
13th_C13_S149	3	alpha	3	alpha	0.99775094	True
13th_C15_S134	4	beta	4	beta	0.9977786	True
13th_C16_S193	4	beta	4	beta	0.99789363	True
13th_C17_S186	4	beta	4	beta	0.9978752	True
13th_C18_S179	3	alpha	3	alpha	0.9976037	True
13th_C1_S147	3	alpha	3	alpha	0.9976826	True
13th_C20_S142	3	alpha	3	alpha	0.9976743	True
13th_C21_S135	4	beta	4	beta	0.997858	True
13th_C23_S187	3	alpha	7	endothelial	0.96821433	True
13th_C27_S151	2	acinar	2	acinar	0.9946571	True
13th_C29_S188	3	alpha	3	alpha	0.9977526	True
13th_C31_S137	3	alpha	3	alpha	0.9977035	True
13th_C34_S181	4	beta	4	beta	0.99771214	True
13th_C36_S196	3	alpha	1	PSC	0.81445074	True
13th_C38_S145	4	beta	4	beta	0.99786735	True
13th_C39_S153	4	beta	4	beta	0.9976714	True
13th_C3_S132	3	alpha	3	alpha	0.99768424	True
13th_C46_S183	4	beta	4	beta	0.99759275	True
13th_C49_S155	6	ductal	6	ductal	0.9931918	True
13th_C4_S191	3	alpha	3	alpha	0.997624	True
13th_C51_S170	5	delta	5	delta	0.98070955	True
13th_C52_S198	4	beta	4	beta	0.9977063	True
13th_C53_S205	3	alpha	3	alpha	0.99760497	True
13th_C54_S212	3	alpha	3	alpha	0.99767107	True
13th_C55_S156	3	alpha	3	alpha	0.99610436	True
13th_C57_S171	4	beta	4	beta	0.9978048	True
13th_C5_S184	4	beta	4	beta	0.9976931	True
13th_C60_S213	3	alpha	3	alpha	0.9977858	True
13th_C63_S172	4	beta	4	beta	0.9976642	True
13th_C66_S214	3	alpha	3	alpha	0.9963574	True
13th_C68_S165	4	beta	4	beta	0.9981608	True
13th_C69_S173	3	alpha	7	endothelial	0.9565916	True
13th_C6_S177	3	alpha	3	alpha	0.99754864	True
13th_C70_S201	4	beta	4	beta	0.9977786	True
13th_C71_S208	4	beta	4	beta	0.9981502	True
13th_C72_S215	4	beta	7	endothelial	0.8464734	True
13th_C75_S159	4	beta	4	beta	0.9976217	True
13th_C76_S216	3	alpha	3	alpha	0.99767345	True
13th_C77_S209	3	alpha	3	alpha	0.99751365	True
13th_C79_S174	3	alpha	3	alpha	0.99734336	True
13th_C80_S167	4	beta	4	beta	0.9978066	True
13th_C81_S160	4	beta	4	beta	0.99794	True
13th_C86_S168	3	alpha	3	alpha	0.99769646	True
13th_C88_S218	3	alpha	3	alpha	0.99715865	True
13th_C89_S211	4	beta	4	beta	0.99780935	True
13th_C8_S140	6	ductal	6	ductal	0.99325484	True
13th_C92_S169	3	alpha	3	alpha	0.9977189	True
13th_C93_S162	3	alpha	3	alpha	0.9977581	True
13th_C94_S219	4	beta	4	beta	0.99798656	True
13th_C96_S204	4	beta	4	beta	0.9977175	True
1st-61_S27	4	beta	4	beta	0.99796224	True
1st-C11_S58	4	beta	4	beta	0.9978089	True
1st-C13_S19	4	beta	4	beta	0.9959896	True
1st-C15_S3	3	alpha	3	alpha	0.997764	True
1st-C18_S51	3	alpha	3	alpha	0.9975383	True
1st-C19_S20	3	alpha	3	alpha	0.9976363	True
1st-C20_S12	3	alpha	3	alpha	0.99757904	True
1st-C21_S4	3	alpha	3	alpha	0.99772376	True
1st-C32_S14	4	beta	4	beta	0.997979	True
1st-C34_S54	1	PSC	1	PSC	0.9938783	True
1st-C39_S23	4	beta	4	beta	0.99807227	True
1st-C42_S71	1	PSC	1	PSC	0.9931912	True
1st-C50_S33	3	alpha	3	alpha	0.9976883	True
1st-C51_S41	3	alpha	3	alpha	0.99775356	True
1st-C53_S74	3	alpha	3	alpha	0.99710006	True
1st-C56_S34	3	alpha	3	alpha	0.99743974	True
1st-C57_S42	3	alpha	3	alpha	0.9976937	True
1st-C58_S76	4	beta	4	beta	0.9970611	True
1st-C59_S77	3	alpha	3	alpha	0.99766684	True
1st-C62_S35	3	alpha	3	alpha	0.997764	True
1st-C64_S79	4	beta	4	beta	0.9980458	True
1st-C68_S36	3	alpha	3	alpha	0.9977582	True
1st-C69_S44	5	delta	5	delta	0.9344657	True
1st-C71_S83	3	alpha	3	alpha	0.9977018	True
1st-C73_S45	3	alpha	3	alpha	0.99775475	True
1st-C74_S37	3	alpha	3	alpha	0.99778783	True
1st-C75_S29	3	alpha	3	alpha	0.9911533	True
1st-C77_S86	4	beta	4	beta	0.97630674	True
1st-C78_S85	3	alpha	3	alpha	0.9974801	True
1st-C79_S46	4	beta	4	beta	0.99793196	True
1st-C80_S38	3	alpha	3	alpha	0.997771	True
1st-C8_S10	3	alpha	3	alpha	0.99774534	True
1st-C90_S91	4	beta	4	beta	0.99771667	True
1st-C96_S94	4	beta	4	beta	0.99792886	True
2nd-C11_S20	3	alpha	3	alpha	0.99769187	True
2nd-C15_S25	4	beta	4	beta	0.99796665	True
2nd-C16_S33	3	alpha	3	alpha	0.99743396	True
2nd-C17_S32	4	beta	4	beta	0.9956642	True
2nd-C19_S39	4	beta	4	beta	0.99810165	True
2nd-C1_S3	3	alpha	3	alpha	0.99780065	True
2nd-C21_S37	4	beta	7	endothelial	0.99160236	True
2nd-C25_S49	5	delta	5	delta	0.96450734	True
2nd-C26_S50	3	alpha	3	alpha	0.9976774	True
2nd-C27_S51	4	beta	4	beta	0.9977939	True
2nd-C28_S55	3	alpha	3	alpha	0.9976763	True
2nd-C29_S56	3	alpha	3	alpha	0.9966665	True
2nd-C2_S2	1	PSC	7	endothelial	0.991247	True
2nd-C31_S61	3	alpha	3	alpha	0.99774534	True
2nd-C32_S62	4	beta	4	beta	0.9976712	True
2nd-C36_S69	3	alpha	3	alpha	0.9976707	True
2nd-C37_S73	4	beta	4	beta	0.99769574	True
2nd-C38_S74	4	beta	4	beta	0.9981567	True
2nd-C39_S75	3	alpha	3	alpha	0.99743396	True
2nd-C3_S1	3	alpha	3	alpha	0.9978103	True
2nd-C47_S92	3	alpha	3	alpha	0.9977131	True
2nd-C49_S4	4	beta	4	beta	0.9977869	True
2nd-C4_S9	3	alpha	3	alpha	0.99726355	True
2nd-C51_S6	4	beta	4	beta	0.9979366	True
2nd-C52_S10	4	beta	4	beta	0.99770147	True
2nd-C54_S12	1	PSC	1	PSC	0.99266666	True
2nd-C57_S18	3	alpha	3	alpha	0.9971123	True
2nd-C58_S22	3	alpha	3	alpha	0.9975777	True
2nd-C5_S8	4	beta	4	beta	0.9979196	True
2nd-C62_S29	4	beta	4	beta	0.9979809	True
2nd-C67_S40	3	alpha	3	alpha	0.9977585	True
2nd-C69_S42	3	alpha	3	alpha	0.99756855	True
2nd-C6_S7	3	alpha	3	alpha	0.9976285	True
2nd-C72_S48	4	beta	4	beta	0.9976236	True
2nd-C74_S53	3	alpha	3	alpha	0.99760735	True
2nd-C75_S52	3	alpha	3	alpha	0.99759334	True
2nd-C76_S60	4	beta	4	beta	0.99756694	True
2nd-C77_S59	3	alpha	3	alpha	0.99771833	True
2nd-C7_S15	4	beta	4	beta	0.99799865	True
2nd-C80_S65	3	alpha	3	alpha	0.99766135	True
2nd-C81_S64	4	beta	4	beta	0.9980229	True
2nd-C83_S71	3	alpha	3	alpha	0.9952939	True
2nd-C86_S77	4	beta	4	beta	0.99782914	True
2nd-C87_S76	4	beta	4	beta	0.9980945	True
2nd-C88_S84	4	beta	4	beta	0.9978269	True
2nd-C8_S14	3	alpha	3	alpha	0.99766123	True
2nd-C93_S88	4	beta	4	beta	0.99793893	True
2nd-C94_S96	3	alpha	3	alpha	0.9974954	True
2nd-C95_S95	4	beta	4	beta	0.9980988	True
3rd-C11_S58	4	beta	4	beta	0.9978523	True
3rd-C12_S68	3	alpha	3	alpha	0.9976	True
3rd-C18_S80	3	alpha	3	alpha	0.9977182	True
3rd-C1_S39	5	delta	5	delta	0.9625106	True
3rd-C20_S22	6	ductal	6	ductal	0.9906998	True
3rd-C21_S11	3	alpha	3	alpha	0.99752325	True
3rd-C23_S94	3	alpha	3	alpha	0.9972537	True
3rd-C25_S15	4	beta	4	beta	0.9978503	True
3rd-C26_S28	3	alpha	3	alpha	0.9961255	True
3rd-C27_S49	3	alpha	3	alpha	0.99768305	True
3rd-C32_S31	3	alpha	3	alpha	0.99746275	True
3rd-C33_S50	4	beta	4	beta	0.99761665	True
3rd-C35_S23	1	PSC	1	PSC	0.9920134	True
3rd-C36_S2	4	beta	4	beta	0.9976665	True
3rd-C3_S2	3	alpha	3	alpha	0.9977393	True
3rd-C42_S74	1	PSC	1	PSC	0.98227274	True
3rd-C46_S93	3	alpha	3	alpha	0.9977188	True
3rd-C49_S60	3	alpha	3	alpha	0.9977723	True
3rd-C4_S25	4	beta	4	beta	0.99799794	True
3rd-C51_S87	1	PSC	1	PSC	0.9924015	True
3rd-C52_S86	3	alpha	3	alpha	0.9977209	True
3rd-C56_S90	4	beta	4	beta	0.9977164	True
3rd-C57_S5	3	alpha	3	alpha	0.99751437	True
3rd-C58_S3	4	beta	4	beta	0.9979898	True
3rd-C59_S91	4	beta	4	beta	0.99807644	True
3rd-C60_S10	4	beta	4	beta	0.99809295	True
3rd-C61_S64	3	alpha	3	alpha	0.9978218	True
3rd-C63_S54	3	alpha	3	alpha	0.9975568	True
3rd-C65_S20	4	beta	4	beta	0.998085	True
3rd-C67_S69	3	alpha	3	alpha	0.99736077	True
3rd-C68_S13	5	delta	5	delta	0.9755078	True
3rd-C69_S78	3	alpha	3	alpha	0.99773693	True
3rd-C6_S44	5	delta	5	delta	0.8988864	True
3rd-C71_S80	4	beta	4	beta	0.99802977	True
3rd-C76_S22	3	alpha	3	alpha	0.99769753	True
3rd-C79_S67	3	alpha	3	alpha	0.99766064	True
3rd-C7_S40	3	alpha	3	alpha	0.997635	True
3rd-C82_S11	4	beta	4	beta	0.99818087	True
3rd-C84_S75	4	beta	4	beta	0.9979936	True
3rd-C85_S79	3	alpha	3	alpha	0.99752825	True
3rd-C85_S83	3	alpha	3	alpha	0.99752265	True
3rd-C86_S85	5	delta	5	delta	0.8283023	True
3rd-C87_S88	4	beta	4	beta	0.99791163	True
3rd-C88_S24	4	beta	4	beta	0.9978823	True
3rd-C91_S91	3	alpha	3	alpha	0.9977125	True
3rd-C92_S86	4	beta	4	beta	0.99802065	True
3rd-C95_S81	3	alpha	3	alpha	0.9976528	True
3rd-C96_S79	4	beta	4	beta	0.9981223	True
3rd-C9_S5	3	alpha	3	alpha	0.9975769	True
4th-61_S28	4	beta	4	beta	0.9981505	True
4th-C12_S19	3	alpha	3	alpha	0.99766976	True
4th-C14_S26	4	beta	4	beta	0.99811137	True
4th-C18_S31	3	alpha	3	alpha	0.997656	True
4th-C22_S45	3	alpha	3	alpha	0.9977702	True
4th-C23_S44	3	alpha	3	alpha	0.99754435	True
4th-C24_S43	6	ductal	4	beta	0.9974884	True
4th-C26_S50	4	beta	4	beta	0.99636334	True
4th-C27_S51	2	acinar	2	acinar	0.974361	True
4th-C28_S55	4	beta	4	beta	0.9975891	True
4th-C29_S56	3	alpha	3	alpha	0.99754494	True
4th-C32_S62	5	delta	5	delta	0.95296025	True
4th-C33_S63	3	alpha	3	alpha	0.9976446	True
4th-C35_S68	3	alpha	3	alpha	0.997672	True
4th-C36_S69	3	alpha	3	alpha	0.99554354	True
4th-C37_S73	3	alpha	3	alpha	0.9975973	True
4th-C3_S1	3	alpha	3	alpha	0.88844943	True
4th-C40_S79	3	alpha	3	alpha	0.99776995	True
4th-C45_S87	4	beta	4	beta	0.9978194	True
4th-C49_S4	4	beta	4	beta	0.9970439	True
4th-C4_S9	3	alpha	3	alpha	0.99774796	True
4th-C50_S5	3	alpha	3	alpha	0.9976046	True
4th-C51_S6	3	alpha	3	alpha	0.99654883	True
4th-C56_S17	3	alpha	3	alpha	0.9976731	True
4th-C57_S18	3	alpha	3	alpha	0.97992146	True
4th-C59_S23	3	alpha	3	alpha	0.9972698	True
4th-C5_S8	4	beta	4	beta	0.9979417	True
4th-C62_S29	3	alpha	3	alpha	0.9976799	True
4th-C63_S30	3	alpha	3	alpha	0.99727577	True
4th-C64_S34	3	alpha	3	alpha	0.997747	True
4th-C66_S36	3	alpha	3	alpha	0.99746716	True
4th-C67_S40	4	beta	4	beta	0.99797016	True
4th-C68_S41	3	alpha	3	alpha	0.99762875	True
4th-C69_S42	3	alpha	3	alpha	0.99659663	True
4th-C73_S54	3	alpha	3	alpha	0.9977787	True
4th-C75_S52	3	alpha	3	alpha	0.9977126	True
4th-C78_S58	3	alpha	7	endothelial	0.9926785	True
4th-C79_S66	3	alpha	3	alpha	0.99766773	True
4th-C7_S15	4	beta	4	beta	0.99803	True
4th-C80_S65	3	alpha	3	alpha	0.99688244	True
4th-C81_S64	6	ductal	6	ductal	0.9940871	True
4th-C82_S72	3	alpha	3	alpha	0.99746084	True
4th-C83_S71	3	alpha	3	alpha	0.9977932	True
4th-C85_S78	5	delta	5	delta	0.82415587	True
4th-C86_S77	3	alpha	3	alpha	0.99781513	True
4th-C87_S76	5	delta	5	delta	0.92381686	True
4th-C88_S84	3	alpha	3	alpha	0.9975197	True
4th-C89_S83	4	beta	4	beta	0.99592364	True
4th-C8_S14	4	beta	4	beta	0.9977769	True
4th-C92_S89	3	alpha	3	alpha	0.9977235	True
4th-C93_S88	4	beta	4	beta	0.99808335	True
4th-C94_S96	3	alpha	3	alpha	0.9966413	True
4th-C95_S95	2	acinar	6	ductal	0.9873886	True
4th-C96_S94	3	alpha	3	alpha	0.9977469	True
4th-C9_S13	3	alpha	3	alpha	0.99769706	True
5th-C10_S92	2	acinar	2	acinar	0.99410796	True
5th-C11_S91	6	ductal	6	ductal	0.9903807	True
5th-C12_S90	2	acinar	2	acinar	0.9650191	True
5th-C14_S97	4	beta	4	beta	0.9979296	True
5th-C16_S98	4	beta	3	alpha	0.993393	True
5th-C18_S96	4	beta	4	beta	0.9976547	True
5th-C19_S104	2	acinar	2	acinar	0.9943039	True
5th-C21_S102	3	alpha	3	alpha	0.9970841	True
5th-C25_S108	4	beta	4	beta	0.99634606	True
5th-C30_S105	6	ductal	6	ductal	0.9879731	True
5th-C34_S108	3	alpha	3	alpha	0.9976672	True
5th-C35_S109	5	delta	5	delta	0.967282	True
5th-C40_S112	2	acinar	2	acinar	0.97257406	True
5th-C41_S113	6	ductal	7	endothelial	0.9549876	True
5th-C42_S114	2	acinar	2	acinar	0.9905704	True
5th-C43_S121	6	ductal	6	ductal	0.99293643	True
5th-C45_S123	2	acinar	2	acinar	0.9938666	True
5th-C47_S118	5	delta	7	endothelial	0.98830193	True
5th-C49_S89	4	beta	4	beta	0.9979085	True
5th-C50_S90	2	acinar	2	acinar	0.99396616	True
5th-C54_S89	2	acinar	2	acinar	0.9941286	True
5th-C56_S95	2	acinar	2	acinar	0.99302214	True
5th-C58_S93	2	acinar	2	acinar	0.99315614	True
5th-C59_S94	3	alpha	3	alpha	0.99621445	True
5th-C5_S85	4	beta	4	beta	0.9978624	True
5th-C64_S99	4	beta	4	beta	0.9918168	True
5th-C66_S101	3	alpha	3	alpha	0.98778695	True
5th-C67_S105	2	acinar	2	acinar	0.7178036	True
5th-C69_S107	3	alpha	3	alpha	0.99681526	True
5th-C6_S84	3	alpha	3	alpha	0.99774206	True
5th-C70_S103	3	alpha	3	alpha	0.99768853	True
5th-C73_S111	2	acinar	2	acinar	0.993834	True
5th-C74_S110	3	alpha	3	alpha	0.99732	True
5th-C7_S93	4	beta	3	alpha	0.85735804	True
5th-C80_S116	4	beta	4	beta	0.9980414	True
5th-C81_S115	2	acinar	2	acinar	0.9938083	True
5th-C83_S111	5	delta	5	delta	0.9761697	True
5th-C87_S119	3	alpha	3	alpha	0.99774855	True
5th-C88_S117	2	acinar	2	acinar	0.9932678	True
5th-C90_S115	4	beta	4	beta	0.99782336	True
5th-C92_S124	3	alpha	3	alpha	0.99645567	True
6th-C12_S18	4	beta	4	beta	0.99808854	True
6th-C15_S22	4	beta	4	beta	0.99798787	True
6th-C16_S29	4	beta	4	beta	0.9977763	True
6th-C19_S35	4	beta	4	beta	0.99803394	True
6th-C1_S3	4	beta	4	beta	0.9979102	True
6th-C20_S34	4	beta	4	beta	0.9978975	True
6th-C21_S33	3	alpha	3	alpha	0.99773717	True
6th-C22_S39	4	beta	4	beta	0.99784255	True
6th-C23_S38	4	beta	4	beta	0.9981121	True
6th-C24_S37	5	delta	5	delta	0.97939134	True
6th-C25_S43	4	beta	4	beta	0.9979097	True
6th-C28_S49	4	beta	4	beta	0.9980469	True
6th-C29_S50	4	beta	4	beta	0.99804735	True
6th-C2_S2	4	beta	4	beta	0.9980101	True
6th-C30_S51	4	beta	4	beta	0.9979463	True
6th-C34_S61	3	alpha	3	alpha	0.99771726	True
6th-C40_S73	4	beta	4	beta	0.9980064	True
6th-C42_S75	4	beta	4	beta	0.99795	True
6th-C45_S81	3	alpha	3	alpha	0.99773884	True
6th-C51_S6	4	beta	4	beta	0.99806374	True
6th-C54_S11	4	beta	4	beta	0.99804366	True
6th-C57_S17	4	beta	4	beta	0.998075	True
6th-C58_S20	4	beta	4	beta	0.9978669	True
6th-C59_S21	4	beta	4	beta	0.9980525	True
6th-C5_S8	3	alpha	3	alpha	0.99774605	True
6th-C62_S25	4	beta	4	beta	0.99778336	True
6th-C63_S26	4	beta	4	beta	0.7853176	True
6th-C64_S30	4	beta	4	beta	0.9978375	True
6th-C65_S31	3	alpha	3	alpha	0.99781513	True
6th-C66_S32	4	beta	4	beta	0.9974578	True
6th-C67_S36	4	beta	4	beta	0.9979693	True
6th-C71_S41	4	beta	4	beta	0.9976941	True
6th-C73_S48	4	beta	4	beta	0.9979108	True
6th-C75_S46	4	beta	4	beta	0.9979765	True
6th-C77_S53	4	beta	4	beta	0.9981341	True
6th-C79_S60	4	beta	4	beta	0.99762183	True
6th-C80_S59	4	beta	4	beta	0.99802125	True
6th-C82_S66	4	beta	4	beta	0.99813485	True
6th-C83_S65	3	alpha	3	alpha	0.9977264	True
6th-C84_S64	4	beta	4	beta	0.9980344	True
6th-C85_S72	4	beta	4	beta	0.9977957	True
6th-C87_S70	3	alpha	3	alpha	0.9972907	True
6th-C89_S77	4	beta	4	beta	0.997761	True
6th-C90_S76	4	beta	4	beta	0.99805605	True
6th-C91_S84	4	beta	4	beta	0.99771464	True
6th-C93_S82	4	beta	4	beta	0.998094	True
6th-C9_S12	3	alpha	3	alpha	0.9977405	True
7th-C10_S19	4	beta	4	beta	0.99811697	True
7th-C11_S18	3	alpha	3	alpha	0.9973809	True
7th-C12_S17	4	beta	4	beta	0.99805766	True
7th-C14_S24	3	alpha	3	alpha	0.9977303	True
7th-C15_S23	5	delta	5	delta	0.97400624	True
7th-C19_S36	4	beta	4	beta	0.998072	True
7th-C1_S3	4	beta	4	beta	0.9979734	True
7th-C20_S35	4	beta	4	beta	0.9978847	True
7th-C21_S34	4	beta	4	beta	0.9978148	True
7th-C25_S43	4	beta	4	beta	0.99803823	True
7th-C28_S48	4	beta	4	beta	0.9981111	True
7th-C2_S2	4	beta	4	beta	0.99797374	True
7th-C31_S53	4	beta	4	beta	0.9981055	True
7th-C32_S54	3	alpha	3	alpha	0.99776995	True
7th-C34_S59	3	alpha	3	alpha	0.9976617	True
7th-C36_S61	3	alpha	3	alpha	0.99766684	True
7th-C3_S1	4	beta	4	beta	0.9977381	True
7th-C44_S76	2	acinar	4	beta	0.60046256	True
7th-C45_S77	4	beta	4	beta	0.9980475	True
7th-C47_S81	6	ductal	6	ductal	0.985458	True
7th-C49_S4	3	alpha	3	alpha	0.99777645	True
7th-C4_S8	4	beta	4	beta	0.99801815	True
7th-C50_S5	4	beta	4	beta	0.99787354	True
7th-C52_S9	4	beta	4	beta	0.9977118	True
7th-C53_S10	4	beta	4	beta	0.9981229	True
7th-C56_S15	5	delta	4	beta	0.85367244	True
7th-C57_S16	4	beta	4	beta	0.9980628	True
7th-C58_S20	3	alpha	3	alpha	0.99778324	True
7th-C59_S21	4	beta	4	beta	0.9980293	True
7th-C60_S22	4	beta	4	beta	0.99762696	True
7th-C62_S27	4	beta	4	beta	0.9975962	True
7th-C64_S31	4	beta	4	beta	0.9981674	True
7th-C65_S32	4	beta	4	beta	0.9979882	True
7th-C66_S33	4	beta	4	beta	0.9979522	True
7th-C68_S38	4	beta	4	beta	0.9740699	True
7th-C69_S39	3	alpha	3	alpha	0.997595	True
7th-C72_S42	5	delta	5	delta	0.9462095	True
7th-C73_S47	4	beta	4	beta	0.99808145	True
7th-C76_S52	6	ductal	6	ductal	0.99295956	True
7th-C77_S51	4	beta	4	beta	0.99804026	True
7th-C79_S58	3	alpha	3	alpha	0.99775034	True
7th-C80_S57	4	beta	4	beta	0.998089	True
7th-C81_S56	6	ductal	6	ductal	0.9898656	True
7th-C84_S62	4	beta	4	beta	0.99778926	True
7th-C85_S69	3	alpha	3	alpha	0.9976737	True
7th-C89_S74	4	beta	4	beta	0.9979963	True
7th-C8_S13	4	beta	4	beta	0.99784136	True
7th-C90_S73	4	beta	4	beta	0.99815696	True
7th-C94_S83	4	beta	4	beta	0.99807775	True
7th-C96_S82	4	beta	4	beta	0.99791473	True
8th-C10_S100	4	beta	4	beta	0.9979253	True
8th-C11_S99	3	alpha	3	alpha	0.997773	True
8th-C18_S110	4	beta	4	beta	0.9980387	True
8th-C20_S116	3	alpha	3	alpha	0.9975714	True
8th-C21_S115	3	alpha	3	alpha	0.99764246	True
8th-C23_S121	3	alpha	3	alpha	0.9976368	True
8th-C24_S120	4	beta	4	beta	0.9978947	True
8th-C26_S126	3	alpha	3	alpha	0.9977229	True
8th-C28_S131	3	alpha	3	alpha	0.99769443	True
8th-C2_S85	4	beta	4	beta	0.9978077	True
8th-C34_S141	3	alpha	3	alpha	0.99720687	True
8th-C37_S146	3	alpha	3	alpha	0.9977228	True
8th-C38_S147	4	beta	4	beta	0.9978764	True
8th-C44_S155	4	beta	4	beta	0.9978477	True
8th-C47_S160	4	beta	4	beta	0.99811685	True
8th-C55_S96	3	alpha	3	alpha	0.99780506	True
8th-C57_S98	3	alpha	3	alpha	0.9977049	True
8th-C58_S101	3	alpha	3	alpha	0.9977416	True
8th-C59_S102	3	alpha	3	alpha	0.9976676	True
8th-C61_S107	4	beta	4	beta	0.99795985	True
8th-C64_S112	3	alpha	3	alpha	0.9976545	True
8th-C65_S113	4	beta	4	beta	0.9978836	True
8th-C67_S117	6	ductal	6	ductal	0.99359185	True
8th-C6_S88	5	delta	5	delta	0.795216	True
8th-C70_S123	4	beta	4	beta	0.9980804	True
8th-C73_S130	6	ductal	6	ductal	0.992747	True
8th-C74_S129	3	alpha	3	alpha	0.99770904	True
8th-C75_S128	4	beta	4	beta	0.9981073	True
8th-C79_S140	3	alpha	3	alpha	0.9976896	True
8th-C7_S95	3	alpha	3	alpha	0.9975823	True
8th-C84_S143	5	delta	5	delta	0.9851059	True
8th-C85_S150	3	alpha	3	alpha	0.99777406	True
8th-C87_S148	4	beta	4	beta	0.998058	True
8th-C90_S152	3	alpha	3	alpha	0.9977585	True
8th-C94_S162	3	alpha	3	alpha	0.99773383	True
8th-C95_S161	3	alpha	3	alpha	0.9976357	True
9th-C10_S18	4	beta	4	beta	0.9978701	True
9th-C11_S17	4	beta	4	beta	0.998085	True
9th-C13_S24	4	beta	4	beta	0.99808574	True
9th-C14_S23	4	beta	4	beta	0.99777067	True
9th-C15_S22	3	alpha	3	alpha	0.99758327	True
9th-C16_S30	4	beta	4	beta	0.9981483	True
9th-C17_S29	3	alpha	3	alpha	0.9975805	True
9th-C19_S36	3	alpha	3	alpha	0.9975369	True
9th-C1_S3	4	beta	4	beta	0.9978815	True
9th-C20_S35	3	alpha	3	alpha	0.9976864	True
9th-C22_S41	1	PSC	1	PSC	0.9194327	True
9th-C25_S45	4	beta	4	beta	0.9980823	True
9th-C28_S50	5	delta	5	delta	0.9677723	True
9th-C29_S51	3	alpha	3	alpha	0.9976623	True
9th-C2_S2	1	PSC	1	PSC	0.99378324	True
9th-C30_S52	3	alpha	3	alpha	0.9976617	True
9th-C38_S64	4	beta	4	beta	0.99819416	True
9th-C3_S1	4	beta	4	beta	0.99803036	True
9th-C41_S70	4	beta	4	beta	0.9981231	True
9th-C46_S79	4	beta	4	beta	0.9979462	True
9th-C49_S4	3	alpha	3	alpha	0.99769235	True
9th-C51_S6	4	beta	4	beta	0.99784994	True
9th-C52_S10	3	alpha	3	alpha	0.9976586	True
9th-C57_S16	3	alpha	3	alpha	0.9971572	True
9th-C59_S20	4	beta	4	beta	0.99809736	True
9th-C5_S8	3	alpha	3	alpha	0.9977423	True
9th-C61_S25	6	ductal	6	ductal	0.9941004	True
9th-C62_S26	3	alpha	3	alpha	0.99762446	True
9th-C64_S31	3	alpha	3	alpha	0.99758196	True
9th-C65_S32	4	beta	4	beta	0.99771595	True
9th-C67_S37	3	alpha	3	alpha	0.99770206	True
9th-C69_S39	4	beta	4	beta	0.9977512	True
9th-C6_S7	4	beta	4	beta	0.99777704	True
9th-C70_S42	4	beta	4	beta	0.9971559	True
9th-C74_S48	3	alpha	3	alpha	0.99735755	True
9th-C75_S47	3	alpha	3	alpha	0.99774396	True
9th-C77_S53	3	alpha	3	alpha	0.99771225	True
9th-C80_S56	3	alpha	3	alpha	0.9976077	True
9th-C81_S55	4	beta	4	beta	0.9981117	True
9th-C83_S61	3	alpha	3	alpha	0.9977703	True
9th-C84_S60	6	ductal	6	ductal	0.47346744	True
9th-C85_S68	3	alpha	3	alpha	0.9975816	True
9th-C86_S67	4	beta	4	beta	0.9977943	True
9th-C87_S66	3	alpha	3	alpha	0.9973972	True
9th-C88_S73	4	beta	4	beta	0.99788815	True
9th-C89_S72	4	beta	4	beta	0.99814343	True
9th-C8_S14	4	beta	4	beta	0.9980082	True
9th-C91_S78	4	beta	4	beta	0.9980489	True
9th-C92_S77	3	alpha	3	alpha	0.99762315	True
9th-C93_S76	4	beta	4	beta	0.9974928	True
9th-C94_S83	4	beta	4	beta	0.99801576	True
9th-C9_S13	3	alpha	3	alpha	0.9976642	True
AZ-A10	5	delta	5	delta	0.95738715	True
AZ-A11	3	alpha	3	alpha	0.99761707	True
AZ-A12	5	delta	5	delta	0.6153716	True
AZ-A2	0	PP	3	alpha	0.9830035	True
AZ-A5	6	ductal	6	ductal	0.9933426	True
AZ-A6	3	alpha	3	alpha	0.9975858	True
AZ-A7	2	acinar	2	acinar	0.99394035	True
AZ-A8	4	beta	4	beta	0.997948	True
AZ-B1	5	delta	5	delta	0.9821593	True
AZ-B10	5	delta	4	beta	0.8952071	True
AZ-B12	3	alpha	3	alpha	0.9976411	True
AZ-B2	3	alpha	3	alpha	0.9975153	True
AZ-B3	3	alpha	3	alpha	0.99770796	True
AZ-B5	3	alpha	3	alpha	0.99770665	True
AZ-B8	2	acinar	6	ductal	0.9702527	True
AZ-B9	3	alpha	3	alpha	0.9977435	True
AZ-C1	3	alpha	3	alpha	0.99748445	True
AZ-C10	4	beta	4	beta	0.9980767	True
AZ-C11	5	delta	4	beta	0.9633989	True
AZ-C2	3	alpha	3	alpha	0.9977036	True
AZ-C3	6	ductal	6	ductal	0.99442863	True
AZ-C4	4	beta	4	beta	0.99775547	True
AZ-C5	3	alpha	3	alpha	0.9975351	True
AZ-C8	3	alpha	3	alpha	0.9976884	True
AZ-C9	6	ductal	6	ductal	0.99473	True
AZ-D12	0	PP	0	PP	0.73893356	True
AZ-D2	4	beta	4	beta	0.9980311	True
AZ-D3	3	alpha	3	alpha	0.9976342	True
AZ-D4	3	alpha	3	alpha	0.99764204	True
AZ-D5	5	delta	5	delta	0.98576444	True
AZ-D6	4	beta	4	beta	0.9977908	True
AZ-D7	3	alpha	3	alpha	0.9976829	True
AZ-D8	0	PP	3	alpha	0.8670224	True
AZ-D9	3	alpha	3	alpha	0.9975581	True
AZ-E12	3	alpha	3	alpha	0.9976356	True
AZ-E2	3	alpha	3	alpha	0.99764156	True
AZ-E3	-1	<unknown_or_unlabeled>	9	macrophage	0.89272755	False
AZ-E4	3	alpha	3	alpha	0.99748933	True
AZ-E5	3	alpha	3	alpha	0.997681	True
AZ-E7	4	beta	4	beta	0.9979518	True
AZ-E8	3	alpha	3	alpha	0.9976406	True
AZ-E9	3	alpha	3	alpha	0.99770015	True
AZ-F1	4	beta	4	beta	0.99799746	True
AZ-F2	4	beta	4	beta	0.9980876	True
AZ-F4	3	alpha	3	alpha	0.99736845	True
AZ-F5	1	PSC	1	PSC	0.99418074	True
AZ-F6	2	acinar	2	acinar	0.99340034	True
AZ-F8	3	alpha	3	alpha	0.9976326	True
AZ-F9	6	ductal	6	ductal	0.9887551	True
AZ-G1	3	alpha	3	alpha	0.9963232	True
AZ-G10	3	alpha	3	alpha	0.9977297	True
AZ-G2	4	beta	4	beta	0.9980704	True
AZ-G3	3	alpha	3	alpha	0.9976883	True
AZ-G4	3	alpha	3	alpha	0.9974826	True
AZ-G5	0	PP	3	alpha	0.82706016	True
AZ-H1	5	delta	5	delta	0.6198652	True
AZ-H10	2	acinar	2	acinar	0.96580726	True
AZ-H11	4	beta	4	beta	0.99796873	True
AZ-H12	0	PP	0	PP	0.71914387	True
AZ-H2	4	beta	4	beta	0.9978981	True
AZ-H3	0	PP	0	PP	0.7876638	True
AZ-H4	7	endothelial	7	endothelial	0.99270284	True
AZ-H5	0	PP	0	PP	0.5850692	True
AZ-H6	4	beta	4	beta	0.9979614	True
AZ-H9	3	alpha	3	alpha	0.9976968	True
HP1502401-A10	4	beta	4	beta	0.9978662	True
HP1502401-A12	3	alpha	3	alpha	0.9975788	True
HP1502401-A14	2	acinar	6	ductal	0.9820402	True
HP1502401-A15	3	alpha	3	alpha	0.99765813	True
HP1502401-A17	3	alpha	3	alpha	0.9977005	True
HP1502401-A18	5	delta	4	beta	0.96165293	True
HP1502401-A2	5	delta	5	delta	0.9702398	True
HP1502401-A22	2	acinar	6	ductal	0.99444467	True
HP1502401-A24	5	delta	5	delta	0.98360795	True
HP1502401-A4	3	alpha	3	alpha	0.9976088	True
HP1502401-A6	4	beta	4	beta	0.9980028	True
HP1502401-A7	4	beta	4	beta	0.997652	True
HP1502401-B1	4	beta	4	beta	0.99813074	True
HP1502401-B10	3	alpha	3	alpha	0.9976096	True
HP1502401-B11	0	PP	5	delta	0.92711365	True
HP1502401-B14	4	beta	4	beta	0.9980229	True
HP1502401-B15	0	PP	3	alpha	0.58478713	True
HP1502401-B16	3	alpha	3	alpha	0.99775606	True
HP1502401-B17	2	acinar	6	ductal	0.9756226	True
HP1502401-B18	0	PP	3	alpha	0.8981988	True
HP1502401-B20	3	alpha	3	alpha	0.9976095	True
HP1502401-B21	4	beta	4	beta	0.99795735	True
HP1502401-B23	4	beta	4	beta	0.9979329	True
HP1502401-B24	4	beta	4	beta	0.99803454	True
HP1502401-B5	3	alpha	3	alpha	0.9975617	True
HP1502401-B7	4	beta	4	beta	0.9979972	True
HP1502401-B8	5	delta	5	delta	0.9860967	True
HP1502401-C1	2	acinar	6	ductal	0.994697	True
HP1502401-C10	4	beta	4	beta	0.9978806	True
HP1502401-C12	6	ductal	6	ductal	0.9940726	True
HP1502401-C14	3	alpha	3	alpha	0.99773264	True
HP1502401-C16	3	alpha	3	alpha	0.99756634	True
HP1502401-C18	4	beta	4	beta	0.99801517	True
HP1502401-C20	5	delta	5	delta	0.97778463	True
HP1502401-C21	4	beta	4	beta	0.99806947	True
HP1502401-C4	3	alpha	3	alpha	0.99774843	True
HP1502401-C5	5	delta	5	delta	0.78574675	True
HP1502401-C6	4	beta	4	beta	0.9980331	True
HP1502401-C7	4	beta	4	beta	0.99796677	True
HP1502401-C8	5	delta	5	delta	0.9753228	True
HP1502401-D10	2	acinar	6	ductal	0.9762807	True
HP1502401-D12	5	delta	5	delta	0.89679223	True
HP1502401-D13	4	beta	4	beta	0.99790204	True
HP1502401-D16	3	alpha	3	alpha	0.99773383	True
HP1502401-D17	5	delta	5	delta	0.96208054	True
HP1502401-D18	6	ductal	6	ductal	0.9941077	True
HP1502401-D2	2	acinar	2	acinar	0.99239254	True
HP1502401-D20	3	alpha	3	alpha	0.99772006	True
HP1502401-D22	4	beta	4	beta	0.9980829	True
HP1502401-D24	3	alpha	3	alpha	0.99757165	True
HP1502401-D3	4	beta	4	beta	0.99787426	True
HP1502401-D4	6	ductal	6	ductal	0.9945899	True
HP1502401-D5	3	alpha	3	alpha	0.9976974	True
HP1502401-D6	4	beta	4	beta	0.9979595	True
HP1502401-D7	4	beta	4	beta	0.9980027	True
HP1502401-D8	3	alpha	3	alpha	0.97481495	True
HP1502401-E10	4	beta	4	beta	0.99796885	True
HP1502401-E11	3	alpha	3	alpha	0.9976204	True
HP1502401-E14	8	epsilon	3	alpha	0.99707484	True
HP1502401-E16	3	alpha	3	alpha	0.9971783	True
HP1502401-E17	3	alpha	3	alpha	0.8920444	True
HP1502401-E19	4	beta	4	beta	0.99811566	True
HP1502401-E2	2	acinar	6	ductal	0.4947898	True
HP1502401-E20	2	acinar	6	ductal	0.95542616	True
HP1502401-E21	3	alpha	3	alpha	0.9976987	True
HP1502401-E23	2	acinar	2	acinar	0.9944324	True
HP1502401-E24	2	acinar	6	ductal	0.81506646	True
HP1502401-E3	5	delta	5	delta	0.97215784	True
HP1502401-E5	0	PP	0	PP	0.623282	True
HP1502401-E6	3	alpha	3	alpha	0.9971564	True
HP1502401-E7	0	PP	0	PP	0.43792996	True
HP1502401-E9	5	delta	5	delta	0.91213447	True
HP1502401-F10	3	alpha	3	alpha	0.9976618	True
HP1502401-F11	5	delta	5	delta	0.97066146	True
HP1502401-F12	3	alpha	3	alpha	0.99776316	True
HP1502401-F13	2	acinar	6	ductal	0.9810508	True
HP1502401-F15	0	PP	3	alpha	0.98371834	True
HP1502401-F16	4	beta	4	beta	0.9971493	True
HP1502401-F17	0	PP	3	alpha	0.7056498	True
HP1502401-F18	0	PP	0	PP	0.45713368	True
HP1502401-F19	3	alpha	3	alpha	0.99763167	True
HP1502401-F20	3	alpha	3	alpha	0.9977483	True
HP1502401-F21	4	beta	4	beta	0.9979235	True
HP1502401-F22	4	beta	4	beta	0.9977191	True
HP1502401-F23	3	alpha	3	alpha	0.99760324	True
HP1502401-F24	4	beta	4	beta	0.9980196	True
HP1502401-F4	3	alpha	3	alpha	0.99773467	True
HP1502401-F9	4	beta	4	beta	0.9977192	True
HP1502401-G10	3	alpha	3	alpha	0.997617	True
HP1502401-G11	0	PP	3	alpha	0.6263868	True
HP1502401-G12	0	PP	5	delta	0.5708221	True
HP1502401-G14	4	beta	4	beta	0.9979978	True
HP1502401-G18	2	acinar	6	ductal	0.88720065	True
HP1502401-G20	5	delta	5	delta	0.9651781	True
HP1502401-G22	4	beta	4	beta	0.9980804	True
HP1502401-G23	6	ductal	6	ductal	0.9945104	True
HP1502401-G24	4	beta	4	beta	0.9977779	True
HP1502401-G3	5	delta	5	delta	0.86045736	True
HP1502401-G4	4	beta	4	beta	0.99802256	True
HP1502401-G5	4	beta	4	beta	0.99810576	True
HP1502401-G6	4	beta	4	beta	0.9980514	True
HP1502401-G7	3	alpha	3	alpha	0.9977689	True
HP1502401-G8	3	alpha	3	alpha	0.9976507	True
HP1502401-H1	3	alpha	3	alpha	0.9972262	True
HP1502401-H11	4	beta	4	beta	0.99779886	True
HP1502401-H12	4	beta	4	beta	0.997974	True
HP1502401-H13	0	PP	3	alpha	0.729136	True
HP1502401-H14	3	alpha	3	alpha	0.99773395	True
HP1502401-H15	3	alpha	3	alpha	0.99770075	True
HP1502401-H16	3	alpha	3	alpha	0.99767274	True
HP1502401-H18	1	PSC	1	PSC	0.99466527	True
HP1502401-H19	4	beta	4	beta	0.9977034	True
HP1502401-H2	3	alpha	3	alpha	0.9976555	True
HP1502401-H20	3	alpha	3	alpha	0.99772125	True
HP1502401-H21	6	ductal	6	ductal	0.8847917	True
HP1502401-H22	4	beta	4	beta	0.9980361	True
HP1502401-H23	6	ductal	6	ductal	0.9946561	True
HP1502401-H24	3	alpha	3	alpha	0.9976369	True
HP1502401-H3	3	alpha	3	alpha	0.99768317	True
HP1502401-H4	3	alpha	3	alpha	0.9976897	True
HP1502401-H5	3	alpha	3	alpha	0.9974113	True
HP1502401-H6	5	delta	5	delta	0.98583543	True
HP1502401-H7	3	alpha	3	alpha	0.9974577	True
HP1502401-H8	3	alpha	3	alpha	0.99764866	True
HP1502401-I1	3	alpha	3	alpha	0.9977585	True
HP1502401-I12	7	endothelial	7	endothelial	0.9924966	True
HP1502401-I14	2	acinar	6	ductal	0.9880108	True
HP1502401-I15	3	alpha	3	alpha	0.9974916	True
HP1502401-I16	3	alpha	3	alpha	0.99762934	True
HP1502401-I18	3	alpha	3	alpha	0.9975834	True
HP1502401-I2	6	ductal	6	ductal	0.9938047	True
HP1502401-I21	6	ductal	6	ductal	0.994406	True
HP1502401-I22	3	alpha	3	alpha	0.99766845	True
HP1502401-I23	3	alpha	3	alpha	0.99759066	True
HP1502401-I24	4	beta	4	beta	0.9979012	True
HP1502401-I4	3	alpha	3	alpha	0.9963372	True
HP1502401-I6	4	beta	4	beta	0.99721575	True
HP1502401-I7	5	delta	5	delta	0.9827533	True
HP1502401-I8	0	PP	3	alpha	0.98894995	True
HP1502401-I9	5	delta	5	delta	0.9748064	True
HP1502401-J1	3	alpha	3	alpha	0.9976671	True
HP1502401-J10	4	beta	4	beta	0.9979869	True
HP1502401-J11	3	alpha	3	alpha	0.9976928	True
HP1502401-J12	4	beta	4	beta	0.997879	True
HP1502401-J14	3	alpha	3	alpha	0.9976241	True
HP1502401-J15	6	ductal	6	ductal	0.99449843	True
HP1502401-J16	10	mast	9	macrophage	0.259063	True
HP1502401-J17	6	ductal	6	ductal	0.9947371	True
HP1502401-J18	3	alpha	3	alpha	0.99770606	True
HP1502401-J19	6	ductal	6	ductal	0.99465746	True
HP1502401-J2	2	acinar	6	ductal	0.9909272	True
HP1502401-J20	3	alpha	3	alpha	0.99774384	True
HP1502401-J21	3	alpha	3	alpha	0.9977291	True
HP1502401-J22	4	beta	4	beta	0.9980398	True
HP1502401-J23	3	alpha	3	alpha	0.9977114	True
HP1502401-J24	6	ductal	6	ductal	0.9943106	True
HP1502401-J4	3	alpha	3	alpha	0.9977083	True
HP1502401-J6	3	alpha	3	alpha	0.99775785	True
HP1502401-J7	10	mast	4	beta	0.23789103	True
HP1502401-J8	3	alpha	3	alpha	0.9975804	True
HP1502401-K1	3	alpha	3	alpha	0.99773633	True
HP1502401-K10	2	acinar	6	ductal	0.98798954	True
HP1502401-K11	3	alpha	3	alpha	0.9974577	True
HP1502401-K12	2	acinar	6	ductal	0.99414694	True
HP1502401-K14	0	PP	3	alpha	0.7043462	True
HP1502401-K15	3	alpha	3	alpha	0.99766195	True
HP1502401-K16	3	alpha	3	alpha	0.99768674	True
HP1502401-K2	3	alpha	3	alpha	0.9975921	True
HP1502401-K20	3	alpha	3	alpha	0.99766195	True
HP1502401-K21	3	alpha	3	alpha	0.9976515	True
HP1502401-K22	3	alpha	3	alpha	0.9977451	True
HP1502401-K23	0	PP	0	PP	0.6666859	True
HP1502401-K24	0	PP	5	delta	0.90745646	True
HP1502401-K3	3	alpha	3	alpha	0.9977155	True
HP1502401-K5	3	alpha	3	alpha	0.9976618	True
HP1502401-K6	3	alpha	3	alpha	0.9977143	True
HP1502401-K9	3	alpha	3	alpha	0.99773514	True
HP1502401-L1	3	alpha	3	alpha	0.99774927	True
HP1502401-L10	4	beta	4	beta	0.9980234	True
HP1502401-L11	3	alpha	3	alpha	0.9977335	True
HP1502401-L12	3	alpha	3	alpha	0.99770135	True
HP1502401-L13	3	alpha	3	alpha	0.9977582	True
HP1502401-L14	0	PP	5	delta	0.8577065	True
HP1502401-L15	3	alpha	3	alpha	0.9976815	True
HP1502401-L16	3	alpha	3	alpha	0.99773115	True
HP1502401-L17	3	alpha	3	alpha	0.9977125	True
HP1502401-L18	6	ductal	6	ductal	0.99453765	True
HP1502401-L2	3	alpha	3	alpha	0.9976961	True
HP1502401-L20	3	alpha	3	alpha	0.9976814	True
HP1502401-L21	3	alpha	3	alpha	0.9977564	True
HP1502401-L22	3	alpha	3	alpha	0.9975394	True
HP1502401-L23	3	alpha	3	alpha	0.9976361	True
HP1502401-L3	3	alpha	3	alpha	0.9977138	True
HP1502401-L4	3	alpha	3	alpha	0.9976361	True
HP1502401-L5	6	ductal	6	ductal	0.9941724	True
HP1502401-L7	3	alpha	3	alpha	0.99774027	True
HP1502401-L9	3	alpha	3	alpha	0.99694014	True
HP1502401-M1	3	alpha	3	alpha	0.9977018	True
HP1502401-M10	3	alpha	3	alpha	0.99772626	True
HP1502401-M11	3	alpha	3	alpha	0.9977155	True
HP1502401-M12	4	beta	4	beta	0.9980394	True
HP1502401-M14	3	alpha	3	alpha	0.99764997	True
HP1502401-M15	3	alpha	3	alpha	0.9976908	True
HP1502401-M16	6	ductal	6	ductal	0.994253	True
HP1502401-M17	10	mast	4	beta	0.25721994	True
HP1502401-M19	3	alpha	3	alpha	0.99770194	True
HP1502401-M2	4	beta	4	beta	0.9980142	True
HP1502401-M21	3	alpha	3	alpha	0.9976737	True
HP1502401-M22	2	acinar	6	ductal	0.99486095	True
HP1502401-M23	3	alpha	3	alpha	0.99768853	True
HP1502401-M24	3	alpha	3	alpha	0.99770737	True
HP1502401-M3	3	alpha	3	alpha	0.99769235	True
HP1502401-M4	3	alpha	3	alpha	0.9977787	True
HP1502401-M5	3	alpha	3	alpha	0.99766976	True
HP1502401-M6	5	delta	5	delta	0.97831285	True
HP1502401-M7	3	alpha	3	alpha	0.99757546	True
HP1502401-M8	2	acinar	6	ductal	0.98664933	True
HP1502401-N1	4	beta	4	beta	0.9979736	True
HP1502401-N10	6	ductal	6	ductal	0.99465054	True
HP1502401-N11	0	PP	3	alpha	0.36783218	True
HP1502401-N12	3	alpha	3	alpha	0.9976888	True
HP1502401-N15	3	alpha	3	alpha	0.9977957	True
HP1502401-N16	6	ductal	6	ductal	0.99345464	True
HP1502401-N17	3	alpha	3	alpha	0.997703	True
HP1502401-N18	6	ductal	6	ductal	0.99402314	True
HP1502401-N2	2	acinar	2	acinar	0.6648553	True
HP1502401-N20	3	alpha	3	alpha	0.9976707	True
HP1502401-N21	4	beta	4	beta	0.99795073	True
HP1502401-N23	0	PP	5	delta	0.91562986	True
HP1502401-N3	3	alpha	3	alpha	0.9976901	True
HP1502401-N4	3	alpha	3	alpha	0.99770516	True
HP1502401-N7	5	delta	5	delta	0.9691121	True
HP1502401-N8	4	beta	4	beta	0.99793845	True
HP1502401-N9	0	PP	3	alpha	0.99088293	True
HP1502401-O1	3	alpha	3	alpha	0.99768054	True
HP1502401-O10	6	ductal	6	ductal	0.99415576	True
HP1502401-O14	2	acinar	6	ductal	0.9862757	True
HP1502401-O16	4	beta	4	beta	0.99799144	True
HP1502401-O17	3	alpha	3	alpha	0.9977652	True
HP1502401-O18	5	delta	5	delta	0.74481696	True
HP1502401-O2	3	alpha	3	alpha	0.99770516	True
HP1502401-O20	3	alpha	3	alpha	0.9976956	True
HP1502401-O21	3	alpha	3	alpha	0.9975994	True
HP1502401-O22	3	alpha	3	alpha	0.9977385	True
HP1502401-O23	3	alpha	3	alpha	0.9974402	True
HP1502401-O24	3	alpha	3	alpha	0.99764293	True
HP1502401-O3	3	alpha	3	alpha	0.99771476	True
HP1502401-O4	3	alpha	3	alpha	0.9977336	True
HP1502401-O6	5	delta	5	delta	0.9541695	True
HP1502401-O8	3	alpha	3	alpha	0.99764234	True
HP1502401-O9	10	mast	4	beta	0.2154083	True
HP1504101T2D-A10	2	acinar	6	ductal	0.94200414	True
HP1504101T2D-A12	3	alpha	3	alpha	0.9977325	True
HP1504101T2D-A13	0	PP	0	PP	0.62173444	True
HP1504101T2D-A15	0	PP	3	alpha	0.9081296	True
HP1504101T2D-A17	0	PP	0	PP	0.6324642	True
HP1504101T2D-A18	0	PP	3	alpha	0.99656445	True
HP1504101T2D-A19	3	alpha	3	alpha	0.99755657	True
HP1504101T2D-A20	0	PP	0	PP	0.7554019	True
HP1504101T2D-A23	3	alpha	3	alpha	0.9977723	True
HP1504101T2D-A24	5	delta	5	delta	0.93825096	True
HP1504101T2D-A4	0	PP	3	alpha	0.993786	True
HP1504101T2D-A5	3	alpha	3	alpha	0.99771214	True
HP1504101T2D-A6	3	alpha	3	alpha	0.99775726	True
HP1504101T2D-A8	3	alpha	3	alpha	0.9977094	True
HP1504101T2D-B10	0	PP	0	PP	0.7608734	True
HP1504101T2D-B11	3	alpha	3	alpha	0.9976833	True
HP1504101T2D-B12	0	PP	5	delta	0.84805405	True
HP1504101T2D-B13	3	alpha	3	alpha	0.99773705	True
HP1504101T2D-B16	3	alpha	3	alpha	0.99777824	True
HP1504101T2D-B17	0	PP	3	alpha	0.9964371	True
HP1504101T2D-B18	0	PP	0	PP	0.72089845	True
HP1504101T2D-B21	3	alpha	3	alpha	0.99781775	True
HP1504101T2D-B22	0	PP	0	PP	0.64527637	True
HP1504101T2D-B24	0	PP	5	delta	0.85502595	True
HP1504101T2D-B4	0	PP	5	delta	0.6398653	True
HP1504101T2D-B5	0	PP	0	PP	0.637656	True
HP1504101T2D-B6	4	beta	4	beta	0.9979153	True
HP1504101T2D-B7	3	alpha	3	alpha	0.9977749	True
HP1504101T2D-B8	6	ductal	6	ductal	0.99483925	True
HP1504101T2D-C1	7	endothelial	7	endothelial	0.99247664	True
HP1504101T2D-C10	5	delta	5	delta	0.96422136	True
HP1504101T2D-C11	2	acinar	2	acinar	0.93384165	True
HP1504101T2D-C12	0	PP	3	alpha	0.5181452	True
HP1504101T2D-C14	3	alpha	3	alpha	0.9977266	True
HP1504101T2D-C15	3	alpha	3	alpha	0.99776673	True
HP1504101T2D-C17	3	alpha	3	alpha	0.9975431	True
HP1504101T2D-C18	4	beta	4	beta	0.9980252	True
HP1504101T2D-C20	3	alpha	3	alpha	0.99759287	True
HP1504101T2D-C21	0	PP	0	PP	0.4958199	True
HP1504101T2D-C22	3	alpha	3	alpha	0.9977443	True
HP1504101T2D-C4	3	alpha	3	alpha	0.9976057	True
HP1504101T2D-C9	5	delta	5	delta	0.9534227	True
HP1504101T2D-D1	3	alpha	3	alpha	0.9976706	True
HP1504101T2D-D11	0	PP	3	alpha	0.9965553	True
HP1504101T2D-D12	3	alpha	3	alpha	0.99767536	True
HP1504101T2D-D13	3	alpha	3	alpha	0.99764925	True
HP1504101T2D-D15	0	PP	0	PP	0.47909436	True
HP1504101T2D-D17	0	PP	0	PP	0.511498	True
HP1504101T2D-D18	0	PP	5	delta	0.48830277	True
HP1504101T2D-D19	3	alpha	3	alpha	0.9978058	True
HP1504101T2D-D2	3	alpha	3	alpha	0.9977336	True
HP1504101T2D-D21	0	PP	3	alpha	0.79086703	True
HP1504101T2D-D23	0	PP	0	PP	0.7077333	True
HP1504101T2D-D24	0	PP	5	delta	0.48691508	True
HP1504101T2D-D5	3	alpha	3	alpha	0.99779284	True
HP1504101T2D-D6	3	alpha	3	alpha	0.9976472	True
HP1504101T2D-D7	3	alpha	3	alpha	0.9977562	True
HP1504101T2D-D9	4	beta	4	beta	0.99796355	True
HP1504101T2D-E1	3	alpha	3	alpha	0.9973508	True
HP1504101T2D-E10	3	alpha	3	alpha	0.99765235	True
HP1504101T2D-E11	0	PP	3	alpha	0.97183627	True
HP1504101T2D-E14	0	PP	5	delta	0.4735505	True
HP1504101T2D-E15	4	beta	4	beta	0.9977914	True
HP1504101T2D-E16	4	beta	4	beta	0.9979557	True
HP1504101T2D-E18	0	PP	3	alpha	0.99324787	True
HP1504101T2D-E19	3	alpha	3	alpha	0.9974474	True
HP1504101T2D-E20	0	PP	3	alpha	0.99153614	True
HP1504101T2D-E21	3	alpha	3	alpha	0.99775535	True
HP1504101T2D-E22	3	alpha	3	alpha	0.99771947	True
HP1504101T2D-E23	3	alpha	3	alpha	0.99764025	True
HP1504101T2D-E24	4	beta	4	beta	0.9977429	True
HP1504101T2D-E3	3	alpha	3	alpha	0.9976482	True
HP1504101T2D-E4	0	PP	3	alpha	0.7414316	True
HP1504101T2D-E6	3	alpha	3	alpha	0.9977386	True
HP1504101T2D-E7	0	PP	3	alpha	0.5698889	True
HP1504101T2D-E8	0	PP	0	PP	0.6528844	True
HP1504101T2D-E9	0	PP	3	alpha	0.78882915	True
HP1504101T2D-F1	2	acinar	6	ductal	0.93557554	True
HP1504101T2D-F10	3	alpha	3	alpha	0.9977464	True
HP1504101T2D-F11	0	PP	0	PP	0.7762509	True
HP1504101T2D-F12	3	alpha	3	alpha	0.99766695	True
HP1504101T2D-F13	0	PP	0	PP	0.77322996	True
HP1504101T2D-F14	0	PP	0	PP	0.64285445	True
HP1504101T2D-F15	3	alpha	3	alpha	0.99766123	True
HP1504101T2D-F16	0	PP	3	alpha	0.9877556	True
HP1504101T2D-F17	5	delta	5	delta	0.95856285	True
HP1504101T2D-F18	1	PSC	1	PSC	0.9942128	True
HP1504101T2D-F19	3	alpha	3	alpha	0.9976661	True
HP1504101T2D-F2	0	PP	3	alpha	0.9633452	True
HP1504101T2D-F20	0	PP	3	alpha	0.8664154	True
HP1504101T2D-F22	3	alpha	3	alpha	0.99767894	True
HP1504101T2D-F3	3	alpha	3	alpha	0.997595	True
HP1504101T2D-F7	3	alpha	3	alpha	0.9976	True
HP1504101T2D-G1	3	alpha	3	alpha	0.9975956	True
HP1504101T2D-G10	3	alpha	3	alpha	0.99764425	True
HP1504101T2D-G12	3	alpha	3	alpha	0.9977336	True
HP1504101T2D-G14	3	alpha	3	alpha	0.99733305	True
HP1504101T2D-G16	0	PP	0	PP	0.70681274	True
HP1504101T2D-G17	3	alpha	3	alpha	0.99768555	True
HP1504101T2D-G19	3	alpha	3	alpha	0.99768853	True
HP1504101T2D-G2	3	alpha	3	alpha	0.99773693	True
HP1504101T2D-G21	0	PP	3	alpha	0.99673283	True
HP1504101T2D-G22	3	alpha	3	alpha	0.9976908	True
HP1504101T2D-G23	3	alpha	3	alpha	0.99771404	True
HP1504101T2D-G24	3	alpha	3	alpha	0.9976815	True
HP1504101T2D-G3	0	PP	0	PP	0.79069984	True
HP1504101T2D-G4	0	PP	0	PP	0.7797574	True
HP1504101T2D-G8	0	PP	0	PP	0.54190624	True
HP1504101T2D-G9	0	PP	5	delta	0.7045833	True
HP1504101T2D-H1	0	PP	3	alpha	0.87578213	True
HP1504101T2D-H14	0	PP	3	alpha	0.86032283	True
HP1504101T2D-H15	3	alpha	3	alpha	0.9974796	True
HP1504101T2D-H17	3	alpha	3	alpha	0.99611014	True
HP1504101T2D-H18	6	ductal	6	ductal	0.9936633	True
HP1504101T2D-H2	3	alpha	3	alpha	0.99767023	True
HP1504101T2D-H20	3	alpha	3	alpha	0.99769443	True
HP1504101T2D-H21	3	alpha	3	alpha	0.99776435	True
HP1504101T2D-H22	3	alpha	3	alpha	0.99765897	True
HP1504101T2D-H23	4	beta	4	beta	0.9980458	True
HP1504101T2D-H24	0	PP	0	PP	0.5087171	True
HP1504101T2D-H4	3	alpha	3	alpha	0.9977303	True
HP1504101T2D-H5	5	delta	5	delta	0.98719364	True
HP1504101T2D-H6	4	beta	4	beta	0.99787164	True
HP1504101T2D-H9	3	alpha	3	alpha	0.9976336	True
HP1504101T2D-I1	3	alpha	3	alpha	0.99466246	True
HP1504101T2D-I10	0	PP	0	PP	0.6551409	True
HP1504101T2D-I14	3	alpha	3	alpha	0.99772376	True
HP1504101T2D-I15	0	PP	0	PP	0.7336478	True
HP1504101T2D-I16	0	PP	0	PP	0.79806644	True
HP1504101T2D-I17	3	alpha	3	alpha	0.99742895	True
HP1504101T2D-I18	3	alpha	3	alpha	0.9977095	True
HP1504101T2D-I2	3	alpha	3	alpha	0.9975223	True
HP1504101T2D-I20	3	alpha	3	alpha	0.99773395	True
HP1504101T2D-I22	3	alpha	3	alpha	0.997682	True
HP1504101T2D-I23	3	alpha	3	alpha	0.997621	True
HP1504101T2D-I24	3	alpha	3	alpha	0.99764097	True
HP1504101T2D-I5	3	alpha	3	alpha	0.9976585	True
HP1504101T2D-I6	2	acinar	6	ductal	0.92183477	True
HP1504101T2D-I7	0	PP	3	alpha	0.9868015	True
HP1504101T2D-I8	3	alpha	3	alpha	0.99773043	True
HP1504101T2D-J1	3	alpha	3	alpha	0.9977591	True
HP1504101T2D-J11	3	alpha	3	alpha	0.99775463	True
HP1504101T2D-J13	3	alpha	3	alpha	0.9977836	True
HP1504101T2D-J14	5	delta	5	delta	0.9816048	True
HP1504101T2D-J15	3	alpha	3	alpha	0.9977047	True
HP1504101T2D-J16	2	acinar	6	ductal	0.99266195	True
HP1504101T2D-J18	3	alpha	3	alpha	0.99775153	True
HP1504101T2D-J2	3	alpha	3	alpha	0.9976305	True
HP1504101T2D-J20	3	alpha	3	alpha	0.9977301	True
HP1504101T2D-J22	0	PP	0	PP	0.65397185	True
HP1504101T2D-J24	3	alpha	3	alpha	0.99777764	True
HP1504101T2D-J3	0	PP	0	PP	0.5537811	True
HP1504101T2D-J5	0	PP	3	alpha	0.8011789	True
HP1504101T2D-J7	3	alpha	3	alpha	0.9977285	True
HP1504101T2D-J9	0	PP	0	PP	0.51833576	True
HP1504101T2D-K10	2	acinar	2	acinar	0.96790445	True
HP1504101T2D-K12	3	alpha	3	alpha	0.997675	True
HP1504101T2D-K17	3	alpha	3	alpha	0.99771047	True
HP1504101T2D-K18	3	alpha	3	alpha	0.99775666	True
HP1504101T2D-K19	2	acinar	6	ductal	0.92981285	True
HP1504101T2D-K2	0	PP	3	alpha	0.7468118	True
HP1504101T2D-K20	3	alpha	3	alpha	0.9976299	True
HP1504101T2D-K22	3	alpha	3	alpha	0.9976922	True
HP1504101T2D-K23	3	alpha	3	alpha	0.9975629	True
HP1504101T2D-K24	3	alpha	3	alpha	0.99773467	True
HP1504101T2D-K4	3	alpha	3	alpha	0.997769	True
HP1504101T2D-K8	0	PP	0	PP	0.6040942	True
HP1504101T2D-L1	0	PP	0	PP	0.5857246	True
HP1504101T2D-L10	3	alpha	3	alpha	0.9977276	True
HP1504101T2D-L11	5	delta	5	delta	0.9833961	True
HP1504101T2D-L13	0	PP	5	delta	0.3864745	True
HP1504101T2D-L14	2	acinar	2	acinar	0.99453235	True
HP1504101T2D-L15	0	PP	3	alpha	0.4374669	True
HP1504101T2D-L18	3	alpha	3	alpha	0.99764997	True
HP1504101T2D-L19	3	alpha	3	alpha	0.9977452	True
HP1504101T2D-L2	3	alpha	3	alpha	0.9977221	True
HP1504101T2D-L21	3	alpha	3	alpha	0.99772733	True
HP1504101T2D-L23	3	alpha	3	alpha	0.99774253	True
HP1504101T2D-L24	0	PP	3	alpha	0.97776264	True
HP1504101T2D-L3	3	alpha	3	alpha	0.99763966	True
HP1504101T2D-L4	3	alpha	3	alpha	0.9976998	True
HP1504101T2D-L5	3	alpha	3	alpha	0.9976967	True
HP1504101T2D-L7	3	alpha	3	alpha	0.9976865	True
HP1504101T2D-L8	3	alpha	3	alpha	0.9975412	True
HP1504101T2D-M1	0	PP	0	PP	0.72987676	True
HP1504101T2D-M10	3	alpha	3	alpha	0.9976488	True
HP1504101T2D-M15	3	alpha	3	alpha	0.9976801	True
HP1504101T2D-M18	3	alpha	3	alpha	0.9976991	True
HP1504101T2D-M19	3	alpha	3	alpha	0.9976834	True
HP1504101T2D-M2	3	alpha	3	alpha	0.9977124	True
HP1504101T2D-M21	3	alpha	3	alpha	0.9976046	True
HP1504101T2D-M23	3	alpha	3	alpha	0.99767226	True
HP1504101T2D-M24	3	alpha	3	alpha	0.9977277	True
HP1504101T2D-M3	0	PP	0	PP	0.7404348	True
HP1504101T2D-M4	3	alpha	3	alpha	0.9974974	True
HP1504101T2D-M5	3	alpha	3	alpha	0.99775535	True
HP1504101T2D-M6	3	alpha	3	alpha	0.99772257	True
HP1504101T2D-M7	3	alpha	3	alpha	0.9977119	True
HP1504101T2D-M8	8	epsilon	3	alpha	0.9969241	True
HP1504101T2D-M9	3	alpha	3	alpha	0.9976156	True
HP1504101T2D-N1	3	alpha	3	alpha	0.9975666	True
HP1504101T2D-N10	4	beta	4	beta	0.99814904	True
HP1504101T2D-N12	3	alpha	3	alpha	0.99775237	True
HP1504101T2D-N13	3	alpha	3	alpha	0.99774593	True
HP1504101T2D-N15	3	alpha	3	alpha	0.9976895	True
HP1504101T2D-N16	0	PP	0	PP	0.6078987	True
HP1504101T2D-N17	3	alpha	3	alpha	0.9977658	True
HP1504101T2D-N19	3	alpha	3	alpha	0.99779797	True
HP1504101T2D-N2	3	alpha	3	alpha	0.997718	True
HP1504101T2D-N20	0	PP	3	alpha	0.9703119	True
HP1504101T2D-N21	5	delta	5	delta	0.9869076	True
HP1504101T2D-N22	3	alpha	3	alpha	0.9977228	True
HP1504101T2D-N24	3	alpha	3	alpha	0.9977447	True
HP1504101T2D-N5	3	alpha	3	alpha	0.9977458	True
HP1504101T2D-N6	3	alpha	3	alpha	0.9977841	True
HP1504101T2D-N7	3	alpha	3	alpha	0.9977062	True
HP1504101T2D-N8	3	alpha	3	alpha	0.9977774	True
HP1504101T2D-N9	3	alpha	3	alpha	0.9977545	True
HP1504101T2D-O10	3	alpha	3	alpha	0.99777216	True
HP1504101T2D-O11	0	PP	3	alpha	0.9699469	True
HP1504101T2D-O12	4	beta	4	beta	0.9976421	True
HP1504101T2D-O13	3	alpha	3	alpha	0.9976999	True
HP1504101T2D-O14	3	alpha	3	alpha	0.997799	True
HP1504101T2D-O16	3	alpha	3	alpha	0.9977483	True
HP1504101T2D-O20	3	alpha	3	alpha	0.997715	True
HP1504101T2D-O22	3	alpha	3	alpha	0.99769247	True
HP1504101T2D-O23	0	PP	3	alpha	0.9606701	True
HP1504101T2D-O24	6	ductal	6	ductal	0.9939685	True
HP1504101T2D-O3	3	alpha	3	alpha	0.99765515	True
HP1504101T2D-O4	0	PP	0	PP	0.58004355	True
HP1504101T2D-O6	3	alpha	3	alpha	0.9975677	True
HP1504101T2D-O9	3	alpha	3	alpha	0.9977093	True
HP1504101T2D-P11	3	alpha	3	alpha	0.9977319	True
HP1504101T2D-P12	0	PP	0	PP	0.6570859	True
HP1504101T2D-P14	3	alpha	3	alpha	0.99151444	True
HP1504101T2D-P17	3	alpha	3	alpha	0.99770916	True
HP1504101T2D-P18	3	alpha	3	alpha	0.99776256	True
HP1504101T2D-P2	3	alpha	3	alpha	0.99767345	True
HP1504101T2D-P21	3	alpha	3	alpha	0.99770445	True
HP1504101T2D-P22	3	alpha	3	alpha	0.997729	True
HP1504101T2D-P24	0	PP	3	alpha	0.99388605	True
HP1504101T2D-P3	1	PSC	1	PSC	0.99370825	True
HP1504101T2D-P5	5	delta	4	beta	0.68437123	True
HP1504101T2D-P7	3	alpha	3	alpha	0.99761885	True
HP1504101T2D-P9	0	PP	3	alpha	0.7917148	True
HP1504901-A10	5	delta	5	delta	0.9726902	True
HP1504901-A14	3	alpha	3	alpha	0.9976915	True
HP1504901-A16	4	beta	4	beta	0.9980122	True
HP1504901-A19	3	alpha	3	alpha	0.997592	True
HP1504901-A2	3	alpha	3	alpha	0.9976774	True
HP1504901-A21	5	delta	5	delta	0.9771721	True
HP1504901-A22	1	PSC	1	PSC	0.98856676	True
HP1504901-A23	7	endothelial	7	endothelial	0.9927515	True
HP1504901-A3	4	beta	4	beta	0.9980679	True
HP1504901-A5	0	PP	3	alpha	0.92338204	True
HP1504901-A9	3	alpha	3	alpha	0.9976827	True
HP1504901-B1	3	alpha	3	alpha	0.99770075	True
HP1504901-B10	4	beta	4	beta	0.9919416	True
HP1504901-B11	4	beta	4	beta	0.998085	True
HP1504901-B14	6	ductal	6	ductal	0.993437	True
HP1504901-B15	3	alpha	3	alpha	0.99773765	True
HP1504901-B16	3	alpha	3	alpha	0.9977609	True
HP1504901-B18	3	alpha	3	alpha	0.9977476	True
HP1504901-B20	0	PP	3	alpha	0.4879424	True
HP1504901-B21	3	alpha	3	alpha	0.997626	True
HP1504901-B22	4	beta	4	beta	0.9979875	True
HP1504901-B23	0	PP	3	alpha	0.9318428	True
HP1504901-B24	1	PSC	1	PSC	0.99058914	True
HP1504901-B3	3	alpha	3	alpha	0.9977748	True
HP1504901-B4	5	delta	5	delta	0.98587775	True
HP1504901-B6	0	PP	0	PP	0.54827315	True
HP1504901-B8	0	PP	3	alpha	0.9455036	True
HP1504901-B9	5	delta	5	delta	0.9791989	True
HP1504901-C10	5	delta	5	delta	0.98597705	True
HP1504901-C11	3	alpha	3	alpha	0.997724	True
HP1504901-C13	8	epsilon	3	alpha	0.9977616	True
HP1504901-C14	0	PP	0	PP	0.49827215	True
HP1504901-C16	3	alpha	3	alpha	0.9976978	True
HP1504901-C17	4	beta	4	beta	0.99796116	True
HP1504901-C23	0	PP	3	alpha	0.84307957	True
HP1504901-D1	3	alpha	3	alpha	0.9636686	True
HP1504901-D12	3	alpha	3	alpha	0.99770325	True
HP1504901-D13	2	acinar	2	acinar	0.95921767	True
HP1504901-D14	3	alpha	3	alpha	0.9976617	True
HP1504901-D15	3	alpha	3	alpha	0.9976554	True
HP1504901-D16	3	alpha	3	alpha	0.9976344	True
HP1504901-D17	3	alpha	3	alpha	0.9977701	True
HP1504901-D18	8	epsilon	3	alpha	0.99779403	True
HP1504901-D2	3	alpha	3	alpha	0.9976628	True
HP1504901-D21	3	alpha	3	alpha	0.9976178	True
HP1504901-D22	3	alpha	3	alpha	0.99759185	True
HP1504901-D23	3	alpha	3	alpha	0.99770904	True
HP1504901-D24	0	PP	3	alpha	0.98814166	True
HP1504901-D5	3	alpha	3	alpha	0.99767953	True
HP1504901-D6	5	delta	5	delta	0.9869703	True
HP1504901-D7	4	beta	4	beta	0.9976598	True
HP1504901-D9	0	PP	3	alpha	0.7336611	True
HP1504901-E14	3	alpha	3	alpha	0.9977307	True
HP1504901-E16	4	beta	4	beta	0.9980288	True
HP1504901-E18	3	alpha	3	alpha	0.9976374	True
HP1504901-E2	7	endothelial	7	endothelial	0.9925997	True
HP1504901-E21	4	beta	4	beta	0.9978696	True
HP1504901-E23	0	PP	0	PP	0.7860515	True
HP1504901-E24	3	alpha	3	alpha	0.997781	True
HP1504901-E6	6	ductal	6	ductal	0.99322563	True
HP1504901-E9	1	PSC	1	PSC	0.9930298	True
HP1504901-F1	0	PP	0	PP	0.4435912	True
HP1504901-F10	3	alpha	3	alpha	0.99749255	True
HP1504901-F11	4	beta	4	beta	0.9980982	True
HP1504901-F12	3	alpha	3	alpha	0.9977315	True
HP1504901-F13	4	beta	4	beta	0.9979919	True
HP1504901-F14	8	epsilon	3	alpha	0.99776983	True
HP1504901-F15	0	PP	3	alpha	0.9922833	True
HP1504901-F16	7	endothelial	7	endothelial	0.9925795	True
HP1504901-F17	4	beta	4	beta	0.9980696	True
HP1504901-F18	0	PP	0	PP	0.70141363	True
HP1504901-F19	3	alpha	3	alpha	0.9977442	True
HP1504901-F2	5	delta	5	delta	0.96261585	True
HP1504901-F20	3	alpha	3	alpha	0.99773216	True
HP1504901-F23	2	acinar	2	acinar	0.9917162	True
HP1504901-F24	3	alpha	3	alpha	0.9972204	True
HP1504901-F3	3	alpha	3	alpha	0.9977435	True
HP1504901-F4	3	alpha	3	alpha	0.99773884	True
HP1504901-F5	3	alpha	3	alpha	0.9977196	True
HP1504901-F8	6	ductal	6	ductal	0.9947418	True
HP1504901-F9	3	alpha	3	alpha	0.9977175	True
HP1504901-G10	7	endothelial	7	endothelial	0.99253196	True
HP1504901-G11	6	ductal	6	ductal	0.99384874	True
HP1504901-G12	4	beta	4	beta	0.99803156	True
HP1504901-G13	3	alpha	3	alpha	0.99766064	True
HP1504901-G15	3	alpha	3	alpha	0.99771094	True
HP1504901-G18	4	beta	4	beta	0.9743711	True
HP1504901-G19	3	alpha	3	alpha	0.9976331	True
HP1504901-G2	3	alpha	3	alpha	0.99751055	True
HP1504901-G20	0	PP	0	PP	0.78137803	True
HP1504901-G21	4	beta	4	beta	0.99785537	True
HP1504901-G24	3	alpha	3	alpha	0.99766904	True
HP1504901-G3	3	alpha	3	alpha	0.9976802	True
HP1504901-G4	4	beta	4	beta	0.99788815	True
HP1504901-G6	3	alpha	3	alpha	0.99775416	True
HP1504901-G8	4	beta	4	beta	0.9979966	True
HP1504901-G9	3	alpha	3	alpha	0.9976357	True
HP1504901-H1	3	alpha	3	alpha	0.9976349	True
HP1504901-H10	3	alpha	3	alpha	0.99768054	True
HP1504901-H11	3	alpha	3	alpha	0.99773407	True
HP1504901-H12	3	alpha	3	alpha	0.9977399	True
HP1504901-H14	0	PP	3	alpha	0.9965641	True
HP1504901-H16	2	acinar	6	ductal	0.99433327	True
HP1504901-H17	4	beta	4	beta	0.9977391	True
HP1504901-H18	4	beta	4	beta	0.99772793	True
HP1504901-H19	0	PP	0	PP	0.74177074	True
HP1504901-H2	3	alpha	3	alpha	0.9976393	True
HP1504901-H22	4	beta	4	beta	0.9980813	True
HP1504901-H4	3	alpha	3	alpha	0.99772996	True
HP1504901-H5	4	beta	4	beta	0.99815196	True
HP1504901-H6	3	alpha	3	alpha	0.9977144	True
HP1504901-I10	6	ductal	6	ductal	0.99356174	True
HP1504901-I12	3	alpha	3	alpha	0.99750096	True
HP1504901-I13	7	endothelial	7	endothelial	0.99225736	True
HP1504901-I17	7	endothelial	7	endothelial	0.9928999	True
HP1504901-I18	0	PP	3	alpha	0.8558447	True
HP1504901-I19	1	PSC	1	PSC	0.9932793	True
HP1504901-I2	3	alpha	3	alpha	0.99759775	True
HP1504901-I21	3	alpha	3	alpha	0.99771214	True
HP1504901-I22	0	PP	3	alpha	0.74988717	True
HP1504901-I24	3	alpha	3	alpha	0.99756783	True
HP1504901-I3	3	alpha	3	alpha	0.99771774	True
HP1504901-I7	5	delta	5	delta	0.9787663	True
HP1504901-I8	3	alpha	3	alpha	0.9976682	True
HP1504901-J12	3	alpha	3	alpha	0.9977575	True
HP1504901-J17	3	alpha	3	alpha	0.9977024	True
HP1504901-J18	1	PSC	1	PSC	0.9933357	True
HP1504901-J20	3	alpha	3	alpha	0.9977168	True
HP1504901-J22	3	alpha	3	alpha	0.9977022	True
HP1504901-J23	3	alpha	3	alpha	0.9977628	True
HP1504901-J24	1	PSC	1	PSC	0.992762	True
HP1504901-J3	4	beta	4	beta	0.9972978	True
HP1504901-J4	5	delta	5	delta	0.98564225	True
HP1504901-J5	6	ductal	6	ductal	0.99395365	True
HP1504901-J6	3	alpha	3	alpha	0.9976972	True
HP1504901-J7	4	beta	4	beta	0.99756396	True
HP1504901-J8	3	alpha	3	alpha	0.99772614	True
HP1504901-J9	0	PP	3	alpha	0.6799668	True
HP1504901-K1	3	alpha	3	alpha	0.99772674	True
HP1504901-K12	3	alpha	3	alpha	0.99766064	True
HP1504901-K14	0	PP	3	alpha	0.84484416	True
HP1504901-K19	4	beta	4	beta	0.99800986	True
HP1504901-K20	5	delta	5	delta	0.9846734	True
HP1504901-K22	4	beta	4	beta	0.9980895	True
HP1504901-K23	6	ductal	6	ductal	0.993067	True
HP1504901-K5	3	alpha	3	alpha	0.997718	True
HP1504901-K6	3	alpha	3	alpha	0.9977381	True
HP1504901-L1	3	alpha	3	alpha	0.9977022	True
HP1504901-L13	3	alpha	3	alpha	0.9977602	True
HP1504901-L14	3	alpha	3	alpha	0.99760485	True
HP1504901-L16	6	ductal	6	ductal	0.9936965	True
HP1504901-L17	4	beta	4	beta	0.9979569	True
HP1504901-L18	0	PP	3	alpha	0.98771054	True
HP1504901-L20	4	beta	4	beta	0.9979943	True
HP1504901-L21	3	alpha	3	alpha	0.9973665	True
HP1504901-L22	4	beta	4	beta	0.9979995	True
HP1504901-L3	0	PP	0	PP	0.71832824	True
HP1504901-L8	4	beta	4	beta	0.99805695	True
HP1504901-L9	0	PP	3	alpha	0.9878945	True
HP1504901-M11	1	PSC	1	PSC	0.98708725	True
HP1504901-M12	3	alpha	3	alpha	0.9974989	True
HP1504901-M18	3	alpha	3	alpha	0.997766	True
HP1504901-M19	1	PSC	1	PSC	0.99354	True
HP1504901-M21	5	delta	5	delta	0.98630595	True
HP1504901-M24	3	alpha	3	alpha	0.99768	True
HP1504901-M3	3	alpha	3	alpha	0.9976204	True
HP1504901-M4	0	PP	3	alpha	0.74232227	True
HP1504901-N1	3	alpha	3	alpha	0.9977029	True
HP1504901-N12	3	alpha	3	alpha	0.997696	True
HP1504901-N13	4	beta	4	beta	0.9979912	True
HP1504901-N15	0	PP	3	alpha	0.6075625	True
HP1504901-N16	3	alpha	3	alpha	0.99769515	True
HP1504901-N17	3	alpha	3	alpha	0.997618	True
HP1504901-N18	6	ductal	6	ductal	0.99399155	True
HP1504901-N19	3	alpha	3	alpha	0.99762744	True
HP1504901-N2	3	alpha	3	alpha	0.997586	True
HP1504901-N21	1	PSC	1	PSC	0.9931851	True
HP1504901-N22	3	alpha	3	alpha	0.9976004	True
HP1504901-N23	0	PP	0	PP	0.58388186	True
HP1504901-N24	5	delta	5	delta	0.9849829	True
HP1504901-N7	3	alpha	3	alpha	0.9976998	True
HP1504901-N8	4	beta	4	beta	0.99797577	True
HP1504901-N9	0	PP	0	PP	0.5709503	True
HP1504901-O1	0	PP	3	alpha	0.91973954	True
HP1504901-O10	3	alpha	3	alpha	0.9976809	True
HP1504901-O11	6	ductal	6	ductal	0.99423426	True
HP1504901-O14	6	ductal	6	ductal	0.99232376	True
HP1504901-O16	3	alpha	3	alpha	0.9977068	True
HP1504901-O17	3	alpha	3	alpha	0.9977082	True
HP1504901-O18	4	beta	4	beta	0.9980305	True
HP1504901-O20	4	beta	4	beta	0.9979201	True
HP1504901-O21	0	PP	0	PP	0.53728354	True
HP1504901-O24	0	PP	0	PP	0.5845267	True
HP1504901-O4	6	ductal	6	ductal	0.99497706	True
HP1504901-O5	0	PP	3	alpha	0.993501	True
HP1504901-O6	3	alpha	3	alpha	0.9977241	True
HP1504901-O7	3	alpha	3	alpha	0.9976623	True
HP1504901-O8	4	beta	4	beta	0.99786735	True
HP1504901-O9	3	alpha	3	alpha	0.99770665	True
HP1504901-P10	3	alpha	3	alpha	0.99774987	True
HP1504901-P11	6	ductal	6	ductal	0.9945186	True
HP1504901-P14	4	beta	4	beta	0.9980501	True
HP1504901-P18	7	endothelial	7	endothelial	0.9928758	True
HP1504901-P20	3	alpha	3	alpha	0.9976979	True
HP1504901-P21	3	alpha	3	alpha	0.99761045	True
HP1504901-P3	7	endothelial	7	endothelial	0.9928231	True
HP1504901-P7	6	ductal	6	ductal	0.9939826	True
HP1504901-P9	1	PSC	1	PSC	0.992959	True
HP1506401-A12	6	ductal	6	ductal	0.9922029	True
HP1506401-A15	4	beta	4	beta	0.997968	True
HP1506401-A17	4	beta	4	beta	0.9976857	True
HP1506401-A18	6	ductal	6	ductal	0.9943386	True
HP1506401-A20	6	ductal	6	ductal	0.9932515	True
HP1506401-A21	3	alpha	3	alpha	0.99759585	True
HP1506401-A24	4	beta	4	beta	0.99785185	True
HP1506401-A4	6	ductal	6	ductal	0.9899635	True
HP1506401-A5	3	alpha	3	alpha	0.99762887	True
HP1506401-A6	2	acinar	2	acinar	0.9927008	True
HP1506401-A7	2	acinar	2	acinar	0.9941182	True
HP1506401-A9	2	acinar	2	acinar	0.7199956	True
HP1506401-B1	2	acinar	2	acinar	0.9944078	True
HP1506401-B11	4	beta	4	beta	0.99769396	True
HP1506401-B14	2	acinar	2	acinar	0.9923935	True
HP1506401-B15	1	PSC	1	PSC	0.9940387	True
HP1506401-B16	2	acinar	2	acinar	0.9939028	True
HP1506401-B17	6	ductal	6	ductal	0.9914141	True
HP1506401-B20	1	PSC	1	PSC	0.9936341	True
HP1506401-B22	0	PP	3	alpha	0.8460381	True
HP1506401-B23	6	ductal	6	ductal	0.992975	True
HP1506401-B24	6	ductal	6	ductal	0.9902615	True
HP1506401-B3	0	PP	3	alpha	0.9846405	True
HP1506401-B4	0	PP	3	alpha	0.8746999	True
HP1506401-B5	2	acinar	2	acinar	0.9915138	True
HP1506401-B6	2	acinar	2	acinar	0.7579201	True
HP1506401-C13	2	acinar	2	acinar	0.9926501	True
HP1506401-C15	4	beta	4	beta	0.9979107	True
HP1506401-C16	6	ductal	6	ductal	0.9927434	True
HP1506401-C17	6	ductal	6	ductal	0.99145913	True
HP1506401-C23	2	acinar	2	acinar	0.9931768	True
HP1506401-C24	2	acinar	6	ductal	0.7746302	True
HP1506401-C4	4	beta	4	beta	0.9978523	True
HP1506401-C5	3	alpha	3	alpha	0.997684	True
HP1506401-C7	2	acinar	2	acinar	0.9635987	True
HP1506401-C8	4	beta	4	beta	0.9979419	True
HP1506401-C9	4	beta	4	beta	0.997617	True
HP1506401-D1	6	ductal	6	ductal	0.9923534	True
HP1506401-D12	2	acinar	2	acinar	0.99444515	True
HP1506401-D13	4	beta	4	beta	0.99777776	True
HP1506401-D14	2	acinar	2	acinar	0.99419457	True
HP1506401-D15	6	ductal	6	ductal	0.9928087	True
HP1506401-D17	6	ductal	6	ductal	0.99234277	True
HP1506401-D2	6	ductal	6	ductal	0.9888587	True
HP1506401-D24	2	acinar	2	acinar	0.9888459	True
HP1506401-D3	2	acinar	2	acinar	0.99284583	True
HP1506401-D7	6	ductal	6	ductal	0.9937063	True
HP1506401-D8	6	ductal	6	ductal	0.9916432	True
HP1506401-D9	0	PP	5	delta	0.9071473	True
HP1506401-E11	3	alpha	3	alpha	0.99761814	True
HP1506401-E12	0	PP	4	beta	0.55793613	True
HP1506401-E15	3	alpha	3	alpha	0.997535	True
HP1506401-E16	5	delta	4	beta	0.9778416	True
HP1506401-E17	6	ductal	6	ductal	0.9926495	True
HP1506401-E18	2	acinar	2	acinar	0.9946243	True
HP1506401-E20	2	acinar	2	acinar	0.99300647	True
HP1506401-E23	2	acinar	2	acinar	0.99268085	True
HP1506401-E24	2	acinar	2	acinar	0.9931102	True
HP1506401-E6	6	ductal	6	ductal	0.9944746	True
HP1506401-E8	2	acinar	2	acinar	0.99404705	True
HP1506401-F10	6	ductal	6	ductal	0.9920642	True
HP1506401-F11	4	beta	4	beta	0.99792093	True
HP1506401-F12	4	beta	4	beta	0.9979225	True
HP1506401-F13	3	alpha	3	alpha	0.99750334	True
HP1506401-F16	2	acinar	6	ductal	0.81554264	True
HP1506401-F17	3	alpha	3	alpha	0.9976017	True
HP1506401-F18	4	beta	4	beta	0.9978079	True
HP1506401-F19	6	ductal	6	ductal	0.99343514	True
HP1506401-F2	2	acinar	6	ductal	0.94285107	True
HP1506401-F21	6	ductal	6	ductal	0.9931554	True
HP1506401-F23	6	ductal	6	ductal	0.99244714	True
HP1506401-F24	4	beta	4	beta	0.9980134	True
HP1506401-F3	2	acinar	2	acinar	0.96953994	True
HP1506401-F5	2	acinar	2	acinar	0.9821566	True
HP1506401-F8	4	beta	4	beta	0.9979528	True
HP1506401-F9	4	beta	4	beta	0.9978124	True
HP1506401-G1	3	alpha	3	alpha	0.99749064	True
HP1506401-G11	5	delta	4	beta	0.7858204	True
HP1506401-G12	4	beta	4	beta	0.99799263	True
HP1506401-G13	3	alpha	3	alpha	0.997729	True
HP1506401-G14	2	acinar	2	acinar	0.99322045	True
HP1506401-G16	2	acinar	2	acinar	0.99384034	True
HP1506401-G18	6	ductal	6	ductal	0.9915541	True
HP1506401-G19	6	ductal	6	ductal	0.98967	True
HP1506401-G2	6	ductal	6	ductal	0.9931015	True
HP1506401-G20	6	ductal	6	ductal	0.99295396	True
HP1506401-G21	6	ductal	6	ductal	0.9932755	True
HP1506401-G23	6	ductal	6	ductal	0.9916583	True
HP1506401-G24	3	alpha	3	alpha	0.9977277	True
HP1506401-G3	3	alpha	3	alpha	0.9977769	True
HP1506401-G4	2	acinar	2	acinar	0.9792092	True
HP1506401-G5	3	alpha	3	alpha	0.99771345	True
HP1506401-G6	4	beta	4	beta	0.9978276	True
HP1506401-G8	0	PP	3	alpha	0.90620196	True
HP1506401-G9	4	beta	4	beta	0.99783117	True
HP1506401-H1	2	acinar	2	acinar	0.9944159	True
HP1506401-H13	3	alpha	3	alpha	0.9975842	True
HP1506401-H14	3	alpha	3	alpha	0.9974969	True
HP1506401-H15	2	acinar	2	acinar	0.99468416	True
HP1506401-H16	2	acinar	2	acinar	0.994437	True
HP1506401-H18	2	acinar	2	acinar	0.9607973	True
HP1506401-H19	0	PP	5	delta	0.9257159	True
HP1506401-H2	6	ductal	6	ductal	0.99427104	True
HP1506401-H22	3	alpha	3	alpha	0.99746335	True
HP1506401-H23	6	ductal	6	ductal	0.99334395	True
HP1506401-H3	2	acinar	2	acinar	0.9944113	True
HP1506401-H4	6	ductal	6	ductal	0.99258703	True
HP1506401-H6	6	ductal	6	ductal	0.99447095	True
HP1506401-H7	6	ductal	6	ductal	0.9926813	True
HP1506401-H9	6	ductal	6	ductal	0.99217516	True
HP1506401-I1	0	PP	0	PP	0.5729935	True
HP1506401-I10	2	acinar	2	acinar	0.9941373	True
HP1506401-I11	4	beta	4	beta	0.99784076	True
HP1506401-I12	4	beta	4	beta	0.99781525	True
HP1506401-I13	2	acinar	2	acinar	0.9912928	True
HP1506401-I14	0	PP	3	alpha	0.97377545	True
HP1506401-I15	2	acinar	6	ductal	0.9686701	True
HP1506401-I17	0	PP	3	alpha	0.7197878	True
HP1506401-I18	2	acinar	2	acinar	0.9930078	True
HP1506401-I2	0	PP	3	alpha	0.9730672	True
HP1506401-I21	6	ductal	6	ductal	0.99111325	True
HP1506401-I23	4	beta	4	beta	0.9979481	True
HP1506401-I24	6	ductal	6	ductal	0.9931365	True
HP1506401-I4	6	ductal	6	ductal	0.9912772	True
HP1506401-I6	2	acinar	2	acinar	0.99318063	True
HP1506401-I8	3	alpha	3	alpha	0.9975727	True
HP1506401-J11	2	acinar	2	acinar	0.9873384	True
HP1506401-J14	2	acinar	6	ductal	0.942955	True
HP1506401-J15	2	acinar	2	acinar	0.94997036	True
HP1506401-J16	8	epsilon	3	alpha	0.9974475	True
HP1506401-J17	4	beta	4	beta	0.9978282	True
HP1506401-J19	4	beta	4	beta	0.99781966	True
HP1506401-J2	2	acinar	2	acinar	0.9944101	True
HP1506401-J20	6	ductal	6	ductal	0.992569	True
HP1506401-J21	6	ductal	6	ductal	0.99391234	True
HP1506401-J22	6	ductal	6	ductal	0.9940049	True
HP1506401-J23	2	acinar	2	acinar	0.992234	True
HP1506401-J24	2	acinar	2	acinar	0.99425465	True
HP1506401-J3	2	acinar	2	acinar	0.99237883	True
HP1506401-J4	3	alpha	3	alpha	0.9970521	True
HP1506401-J5	6	ductal	6	ductal	0.9919722	True
HP1506401-J6	2	acinar	2	acinar	0.9942437	True
HP1506401-J7	6	ductal	6	ductal	0.9904136	True
HP1506401-J8	6	ductal	6	ductal	0.99327457	True
HP1506401-J9	2	acinar	2	acinar	0.9927946	True
HP1506401-K1	4	beta	4	beta	0.9979651	True
HP1506401-K12	3	alpha	3	alpha	0.99775594	True
HP1506401-K14	2	acinar	2	acinar	0.99315083	True
HP1506401-K15	2	acinar	2	acinar	0.9924625	True
HP1506401-K16	6	ductal	6	ductal	0.99152875	True
HP1506401-K17	6	ductal	6	ductal	0.99252313	True
HP1506401-K19	6	ductal	6	ductal	0.99099696	True
HP1506401-K2	2	acinar	2	acinar	0.9729815	True
HP1506401-K21	4	beta	4	beta	0.9978523	True
HP1506401-K22	0	PP	3	alpha	0.98746467	True
HP1506401-K23	0	PP	3	alpha	0.9963315	True
HP1506401-K4	6	ductal	6	ductal	0.99388313	True
HP1506401-K6	2	acinar	2	acinar	0.95728076	True
HP1506401-L1	6	ductal	6	ductal	0.9938241	True
HP1506401-L10	6	ductal	6	ductal	0.994468	True
HP1506401-L11	2	acinar	2	acinar	0.9933809	True
HP1506401-L13	6	ductal	6	ductal	0.99393976	True
HP1506401-L17	0	PP	3	alpha	0.9079214	True
HP1506401-L18	2	acinar	2	acinar	0.9934918	True
HP1506401-L19	2	acinar	2	acinar	0.9929542	True
HP1506401-L20	6	ductal	6	ductal	0.9930703	True
HP1506401-L21	6	ductal	6	ductal	0.9936832	True
HP1506401-L22	2	acinar	2	acinar	0.9898538	True
HP1506401-L23	3	alpha	3	alpha	0.9976605	True
HP1506401-L3	2	acinar	2	acinar	0.9941797	True
HP1506401-L4	6	ductal	6	ductal	0.99396646	True
HP1506401-L5	4	beta	4	beta	0.9978783	True
HP1506401-L7	2	acinar	2	acinar	0.99421084	True
HP1506401-L9	2	acinar	2	acinar	0.9884072	True
HP1506401-M1	2	acinar	2	acinar	0.95066243	True
HP1506401-M11	6	ductal	6	ductal	0.9930478	True
HP1506401-M12	3	alpha	3	alpha	0.99777335	True
HP1506401-M16	6	ductal	6	ductal	0.9937854	True
HP1506401-M17	2	acinar	2	acinar	0.9932672	True
HP1506401-M18	2	acinar	2	acinar	0.9936458	True
HP1506401-M19	2	acinar	2	acinar	0.99323094	True
HP1506401-M21	6	ductal	6	ductal	0.9908183	True
HP1506401-M4	2	acinar	2	acinar	0.98051023	True
HP1506401-M6	0	PP	3	alpha	0.47294998	True
HP1506401-M8	2	acinar	2	acinar	0.9938929	True
HP1506401-N1	2	acinar	2	acinar	0.8769819	True
HP1506401-N10	6	ductal	6	ductal	0.993133	True
HP1506401-N11	4	beta	4	beta	0.9980089	True
HP1506401-N15	2	acinar	2	acinar	0.9939972	True
HP1506401-N16	6	ductal	6	ductal	0.99358785	True
HP1506401-N17	6	ductal	6	ductal	0.9927649	True
HP1506401-N19	3	alpha	3	alpha	0.99760276	True
HP1506401-N20	4	beta	4	beta	0.9979297	True
HP1506401-N22	2	acinar	2	acinar	0.99375397	True
HP1506401-N23	2	acinar	2	acinar	0.9936054	True
HP1506401-N24	2	acinar	6	ductal	0.8262187	True
HP1506401-N4	3	alpha	3	alpha	0.99744886	True
HP1506401-N5	2	acinar	2	acinar	0.9922232	True
HP1506401-N9	2	acinar	2	acinar	0.992607	True
HP1506401-O10	4	beta	4	beta	0.99755585	True
HP1506401-O11	4	beta	4	beta	0.99782	True
HP1506401-O12	2	acinar	2	acinar	0.99411386	True
HP1506401-O16	2	acinar	2	acinar	0.9935288	True
HP1506401-O18	2	acinar	2	acinar	0.9929717	True
HP1506401-O2	6	ductal	6	ductal	0.99397165	True
HP1506401-O21	2	acinar	2	acinar	0.99409276	True
HP1506401-O23	3	alpha	3	alpha	0.99765706	True
HP1506401-O24	2	acinar	2	acinar	0.9939277	True
HP1506401-O3	3	alpha	3	alpha	0.9977247	True
HP1506401-O4	6	ductal	6	ductal	0.9923969	True
HP1506401-O7	2	acinar	6	ductal	0.98887384	True
HP1506401-O8	6	ductal	6	ductal	0.9919586	True
HP1506401-P1	6	ductal	6	ductal	0.990063	True
HP1506401-P10	2	acinar	2	acinar	0.993802	True
HP1506401-P11	6	ductal	6	ductal	0.99253297	True
HP1506401-P12	4	beta	4	beta	0.9978871	True
HP1506401-P14	4	beta	4	beta	0.9979213	True
HP1506401-P17	2	acinar	2	acinar	0.9919626	True
HP1506401-P19	3	alpha	3	alpha	0.9975751	True
HP1506401-P21	6	ductal	6	ductal	0.99425346	True
HP1506401-P22	6	ductal	6	ductal	0.99413955	True
HP1506401-P23	2	acinar	2	acinar	0.9936009	True
HP1506401-P3	3	alpha	3	alpha	0.99764305	True
HP1506401-P4	6	ductal	6	ductal	0.99310464	True
HP1506401-P5	6	ductal	6	ductal	0.99207217	True
HP1507101-A12	4	beta	4	beta	0.99787235	True
HP1507101-A14	3	alpha	3	alpha	0.99770194	True
HP1507101-A20	3	alpha	3	alpha	0.9976417	True
HP1507101-A22	3	alpha	3	alpha	0.99775785	True
HP1507101-A4	3	alpha	3	alpha	0.99765754	True
HP1507101-A5	5	delta	5	delta	0.9503846	True
HP1507101-A6	4	beta	4	beta	0.99760455	True
HP1507101-B1	3	alpha	3	alpha	0.9977246	True
HP1507101-B12	3	alpha	3	alpha	0.99768317	True
HP1507101-B13	3	alpha	3	alpha	0.9977168	True
HP1507101-B14	3	alpha	3	alpha	0.9976114	True
HP1507101-B16	3	alpha	3	alpha	0.9977005	True
HP1507101-B18	3	alpha	3	alpha	0.9976822	True
HP1507101-B19	3	alpha	3	alpha	0.9977471	True
HP1507101-B20	0	PP	5	delta	0.7781623	True
HP1507101-B23	3	alpha	3	alpha	0.99763703	True
HP1507101-B24	3	alpha	3	alpha	0.9975962	True
HP1507101-B3	3	alpha	3	alpha	0.99773276	True
HP1507101-B4	3	alpha	3	alpha	0.9977748	True
HP1507101-B5	4	beta	4	beta	0.9977284	True
HP1507101-B6	3	alpha	3	alpha	0.9977458	True
HP1507101-B7	3	alpha	3	alpha	0.99774146	True
HP1507101-C12	4	beta	4	beta	0.9975937	True
HP1507101-C14	3	alpha	3	alpha	0.99758565	True
HP1507101-C18	3	alpha	3	alpha	0.99765706	True
HP1507101-C2	3	alpha	3	alpha	0.9973066	True
HP1507101-C4	3	alpha	3	alpha	0.9972154	True
HP1507101-C5	3	alpha	3	alpha	0.99774337	True
HP1507101-C7	4	beta	4	beta	0.9979259	True
HP1507101-C8	3	alpha	3	alpha	0.9976156	True
HP1507101-D1	3	alpha	3	alpha	0.99772996	True
HP1507101-D10	7	endothelial	7	endothelial	0.9916529	True
HP1507101-D12	3	alpha	3	alpha	0.997696	True
HP1507101-D14	3	alpha	3	alpha	0.9974753	True
HP1507101-D15	3	alpha	3	alpha	0.9975581	True
HP1507101-D17	3	alpha	3	alpha	0.9974942	True
HP1507101-D18	3	alpha	3	alpha	0.9975447	True
HP1507101-D2	3	alpha	3	alpha	0.99760985	True
HP1507101-D22	4	beta	4	beta	0.99741995	True
HP1507101-D23	4	beta	4	beta	0.99782336	True
HP1507101-D3	4	beta	4	beta	0.9973985	True
HP1507101-D5	3	alpha	3	alpha	0.9975726	True
HP1507101-D6	4	beta	4	beta	0.99794656	True
HP1507101-D8	3	alpha	3	alpha	0.9977335	True
HP1507101-D9	1	PSC	1	PSC	0.99401194	True
HP1507101-E10	3	alpha	3	alpha	0.9977648	True
HP1507101-E12	3	alpha	3	alpha	0.9974794	True
HP1507101-E14	4	beta	4	beta	0.99788326	True
HP1507101-E2	3	alpha	3	alpha	0.99736303	True
HP1507101-E20	3	alpha	3	alpha	0.9976006	True
HP1507101-E22	2	acinar	2	acinar	0.994662	True
HP1507101-E24	3	alpha	3	alpha	0.9975865	True
HP1507101-E6	3	alpha	3	alpha	0.99749905	True
HP1507101-E7	3	alpha	3	alpha	0.99755293	True
HP1507101-E8	3	alpha	3	alpha	0.997282	True
HP1507101-E9	3	alpha	3	alpha	0.9976586	True
HP1507101-F1	3	alpha	3	alpha	0.99773383	True
HP1507101-F10	5	delta	4	beta	0.9240397	True
HP1507101-F11	4	beta	4	beta	0.9974679	True
HP1507101-F12	5	delta	4	beta	0.9972899	True
HP1507101-F13	3	alpha	3	alpha	0.997491	True
HP1507101-F14	4	beta	4	beta	0.997598	True
HP1507101-F15	6	ductal	6	ductal	0.9942748	True
HP1507101-F16	3	alpha	3	alpha	0.99756324	True
HP1507101-F18	3	alpha	3	alpha	0.9977196	True
HP1507101-F19	3	alpha	3	alpha	0.99769837	True
HP1507101-F2	3	alpha	3	alpha	0.9975938	True
HP1507101-F20	3	alpha	3	alpha	0.9977259	True
HP1507101-F22	3	alpha	3	alpha	0.9976671	True
HP1507101-F23	4	beta	4	beta	0.9979206	True
HP1507101-F3	3	alpha	3	alpha	0.997652	True
HP1507101-F4	4	beta	4	beta	0.9977562	True
HP1507101-F5	3	alpha	3	alpha	0.9975872	True
HP1507101-F7	3	alpha	3	alpha	0.9976687	True
HP1507101-F8	3	alpha	3	alpha	0.99766856	True
HP1507101-G10	6	ductal	6	ductal	0.99298674	True
HP1507101-G12	3	alpha	3	alpha	0.9977519	True
HP1507101-G13	3	alpha	3	alpha	0.9976465	True
HP1507101-G14	3	alpha	3	alpha	0.9976998	True
HP1507101-G15	3	alpha	3	alpha	0.99757606	True
HP1507101-G16	3	alpha	3	alpha	0.9976548	True
HP1507101-G18	3	alpha	3	alpha	0.9976736	True
HP1507101-G2	1	PSC	1	PSC	0.9927529	True
HP1507101-G22	3	alpha	3	alpha	0.9975231	True
HP1507101-G23	3	alpha	3	alpha	0.9976235	True
HP1507101-G6	3	alpha	3	alpha	0.99713254	True
HP1507101-G8	3	alpha	3	alpha	0.9976342	True
HP1507101-H1	3	alpha	3	alpha	0.99756783	True
HP1507101-H11	3	alpha	3	alpha	0.99741673	True
HP1507101-H12	3	alpha	3	alpha	0.99775165	True
HP1507101-H14	3	alpha	3	alpha	0.9977137	True
HP1507101-H15	3	alpha	3	alpha	0.99770457	True
HP1507101-H16	6	ductal	6	ductal	0.9933949	True
HP1507101-H17	4	beta	4	beta	0.997673	True
HP1507101-H22	3	alpha	3	alpha	0.9976993	True
HP1507101-H23	3	alpha	3	alpha	0.9975688	True
HP1507101-H3	3	alpha	3	alpha	0.99752444	True
HP1507101-H4	4	beta	4	beta	0.9979886	True
HP1507101-H5	3	alpha	3	alpha	0.99762964	True
HP1507101-H6	3	alpha	3	alpha	0.99755245	True
HP1507101-H7	3	alpha	3	alpha	0.9977138	True
HP1507101-H9	3	alpha	3	alpha	0.99767476	True
HP1507101-I11	3	alpha	3	alpha	0.9977865	True
HP1507101-I12	4	beta	4	beta	0.99778545	True
HP1507101-I13	3	alpha	3	alpha	0.99754804	True
HP1507101-I16	3	alpha	3	alpha	0.9975587	True
HP1507101-I17	3	alpha	3	alpha	0.9976935	True
HP1507101-I18	4	beta	4	beta	0.99815863	True
HP1507101-I19	3	alpha	3	alpha	0.9976344	True
HP1507101-I2	3	alpha	3	alpha	0.99757665	True
HP1507101-I20	3	alpha	3	alpha	0.9975707	True
HP1507101-I22	3	alpha	3	alpha	0.99776757	True
HP1507101-I4	3	alpha	3	alpha	0.9976641	True
HP1507101-I5	3	alpha	3	alpha	0.9976713	True
HP1507101-I6	3	alpha	3	alpha	0.9975446	True
HP1507101-I8	3	alpha	3	alpha	0.9975079	True
HP1507101-J1	3	alpha	3	alpha	0.9976514	True
HP1507101-J11	3	alpha	3	alpha	0.997615	True
HP1507101-J12	1	PSC	1	PSC	0.99254507	True
HP1507101-J13	3	alpha	3	alpha	0.997619	True
HP1507101-J15	6	ductal	6	ductal	0.99315566	True
HP1507101-J17	3	alpha	3	alpha	0.99769825	True
HP1507101-J18	3	alpha	3	alpha	0.9975424	True
HP1507101-J19	4	beta	4	beta	0.9976756	True
HP1507101-J2	3	alpha	3	alpha	0.99685955	True
HP1507101-J20	4	beta	4	beta	0.99778265	True
HP1507101-J21	3	alpha	3	alpha	0.997664	True
HP1507101-J22	3	alpha	3	alpha	0.99761605	True
HP1507101-J23	3	alpha	3	alpha	0.99737966	True
HP1507101-J24	3	alpha	3	alpha	0.99778724	True
HP1507101-J3	3	alpha	3	alpha	0.99767345	True
HP1507101-J5	3	alpha	3	alpha	0.99739206	True
HP1507101-J7	2	acinar	6	ductal	0.9382187	True
HP1507101-K13	1	PSC	1	PSC	0.9929668	True
HP1507101-K15	4	beta	4	beta	0.99749726	True
HP1507101-K16	6	ductal	6	ductal	0.99311876	True
HP1507101-K20	1	PSC	1	PSC	0.9933581	True
HP1507101-K23	4	beta	4	beta	0.9976947	True
HP1507101-K3	3	alpha	3	alpha	0.9976666	True
HP1507101-K4	3	alpha	3	alpha	0.9977264	True
HP1507101-K8	3	alpha	3	alpha	0.9974727	True
HP1507101-L1	1	PSC	1	PSC	0.993317	True
HP1507101-L11	3	alpha	3	alpha	0.997532	True
HP1507101-L14	3	alpha	3	alpha	0.99758124	True
HP1507101-L16	4	beta	4	beta	0.99775994	True
HP1507101-L17	2	acinar	6	ductal	0.96973825	True
HP1507101-L18	3	alpha	3	alpha	0.9970968	True
HP1507101-L19	3	alpha	3	alpha	0.99754155	True
HP1507101-L2	3	alpha	3	alpha	0.9974527	True
HP1507101-L21	3	alpha	3	alpha	0.99749005	True
HP1507101-L22	4	beta	4	beta	0.9976368	True
HP1507101-L23	5	delta	4	beta	0.59131753	True
HP1507101-L24	4	beta	4	beta	0.9977355	True
HP1507101-L3	3	alpha	3	alpha	0.99751943	True
HP1507101-L4	5	delta	5	delta	0.9231331	True
HP1507101-L5	3	alpha	3	alpha	0.99742967	True
HP1507101-L7	3	alpha	3	alpha	0.9976375	True
HP1507101-M12	3	alpha	3	alpha	0.9976011	True
HP1507101-M18	3	alpha	3	alpha	0.9974904	True
HP1507101-M2	3	alpha	3	alpha	0.9975567	True
HP1507101-M22	0	PP	5	delta	0.9781357	True
HP1507101-M4	3	alpha	3	alpha	0.99767035	True
HP1507101-M8	4	beta	4	beta	0.99726534	True
HP1507101-N1	3	alpha	3	alpha	0.9951147	True
HP1507101-N13	4	beta	4	beta	0.99753225	True
HP1507101-N16	3	alpha	3	alpha	0.9975611	True
HP1507101-N19	3	alpha	3	alpha	0.997355	True
HP1507101-N2	3	alpha	3	alpha	0.99706215	True
HP1507101-N20	3	alpha	3	alpha	0.9976821	True
HP1507101-N4	3	alpha	3	alpha	0.9977451	True
HP1507101-N5	4	beta	4	beta	0.99775994	True
HP1507101-N6	4	beta	4	beta	0.99771774	True
HP1507101-N8	6	ductal	6	ductal	0.9933368	True
HP1507101-N9	3	alpha	3	alpha	0.99771285	True
HP1507101-O1	3	alpha	3	alpha	0.99764735	True
HP1507101-O11	3	alpha	3	alpha	0.99753773	True
HP1507101-O12	6	ductal	6	ductal	0.9938246	True
HP1507101-O13	3	alpha	3	alpha	0.9976271	True
HP1507101-O14	4	beta	4	beta	0.9976846	True
HP1507101-O16	3	alpha	3	alpha	0.99760616	True
HP1507101-O19	4	beta	4	beta	0.99785393	True
HP1507101-O20	3	alpha	3	alpha	0.99770796	True
HP1507101-O22	3	alpha	3	alpha	0.99750537	True
HP1507101-O23	5	delta	5	delta	0.8924309	True
HP1507101-O3	4	beta	4	beta	0.997649	True
HP1507101-O5	3	alpha	3	alpha	0.9971782	True
HP1507101-O8	4	beta	4	beta	0.99773836	True
HP1507101-P1	6	ductal	6	ductal	0.99351597	True
HP1507101-P10	3	alpha	3	alpha	0.99759454	True
HP1507101-P12	3	alpha	3	alpha	0.9976834	True
HP1507101-P2	4	beta	4	beta	0.9977633	True
HP1507101-P21	5	delta	4	beta	0.9521366	True
HP1507101-P23	3	alpha	3	alpha	0.99760675	True
HP1507101-P3	3	alpha	3	alpha	0.99769527	True
HP1507101-P4	3	alpha	3	alpha	0.997718	True
HP1507101-P5	3	alpha	3	alpha	0.99774504	True
HP1507101-P8	3	alpha	3	alpha	0.9975841	True
HP1508501T2D-A10	4	beta	4	beta	0.99781704	True
HP1508501T2D-A11	2	acinar	2	acinar	0.99089324	True
HP1508501T2D-A12	3	alpha	3	alpha	0.9974827	True
HP1508501T2D-A13	3	alpha	3	alpha	0.9977812	True
HP1508501T2D-A14	3	alpha	3	alpha	0.99758697	True
HP1508501T2D-A15	3	alpha	3	alpha	0.997818	True
HP1508501T2D-A16	3	alpha	3	alpha	0.9977246	True
HP1508501T2D-A17	6	ductal	6	ductal	0.99381554	True
HP1508501T2D-A19	0	PP	3	alpha	0.93302876	True
HP1508501T2D-A23	2	acinar	2	acinar	0.98511016	True
HP1508501T2D-A3	3	alpha	3	alpha	0.99771935	True
HP1508501T2D-A4	3	alpha	3	alpha	0.9975908	True
HP1508501T2D-A7	6	ductal	6	ductal	0.9942099	True
HP1508501T2D-A8	1	PSC	1	PSC	0.9937668	True
HP1508501T2D-B10	0	PP	3	alpha	0.99595284	True
HP1508501T2D-B11	3	alpha	3	alpha	0.99773943	True
HP1508501T2D-B12	3	alpha	3	alpha	0.99754155	True
HP1508501T2D-B14	2	acinar	2	acinar	0.9947305	True
HP1508501T2D-B15	3	alpha	3	alpha	0.99754333	True
HP1508501T2D-B16	4	beta	4	beta	0.99766195	True
HP1508501T2D-B17	6	ductal	6	ductal	0.99278873	True
HP1508501T2D-B18	6	ductal	6	ductal	0.99363834	True
HP1508501T2D-B19	6	ductal	6	ductal	0.99341583	True
HP1508501T2D-B20	4	beta	4	beta	0.9978452	True
HP1508501T2D-B23	3	alpha	3	alpha	0.997799	True
HP1508501T2D-B3	3	alpha	3	alpha	0.9977992	True
HP1508501T2D-B4	3	alpha	3	alpha	0.99767977	True
HP1508501T2D-B5	3	alpha	3	alpha	0.99773574	True
HP1508501T2D-B8	3	alpha	3	alpha	0.9977679	True
HP1508501T2D-B9	3	alpha	3	alpha	0.9977754	True
HP1508501T2D-C10	3	alpha	3	alpha	0.9977189	True
HP1508501T2D-C11	6	ductal	6	ductal	0.99410206	True
HP1508501T2D-C12	1	PSC	1	PSC	0.9931705	True
HP1508501T2D-C13	3	alpha	3	alpha	0.9978041	True
HP1508501T2D-C14	4	beta	4	beta	0.9975568	True
HP1508501T2D-C15	6	ductal	6	ductal	0.9940029	True
HP1508501T2D-C16	5	delta	4	beta	0.62561077	True
HP1508501T2D-C17	4	beta	4	beta	0.99803597	True
HP1508501T2D-C18	6	ductal	6	ductal	0.9936326	True
HP1508501T2D-C19	6	ductal	6	ductal	0.9942987	True
HP1508501T2D-C2	6	ductal	6	ductal	0.993948	True
HP1508501T2D-C20	3	alpha	3	alpha	0.9977035	True
HP1508501T2D-C23	3	alpha	3	alpha	0.9973201	True
HP1508501T2D-C4	4	beta	4	beta	0.9974982	True
HP1508501T2D-C8	5	delta	5	delta	0.9611207	True
HP1508501T2D-C9	-1	<unknown_or_unlabeled>	9	macrophage	0.5932636	False
HP1508501T2D-D1	3	alpha	3	alpha	0.9977381	True
HP1508501T2D-D10	3	alpha	3	alpha	0.9977902	True
HP1508501T2D-D11	3	alpha	3	alpha	0.9976793	True
HP1508501T2D-D13	3	alpha	3	alpha	0.9977562	True
HP1508501T2D-D14	3	alpha	3	alpha	0.99779344	True
HP1508501T2D-D15	3	alpha	3	alpha	0.9971226	True
HP1508501T2D-D16	3	alpha	3	alpha	0.99783355	True
HP1508501T2D-D17	3	alpha	3	alpha	0.9978053	True
HP1508501T2D-D18	2	acinar	6	ductal	0.80803436	True
HP1508501T2D-D2	3	alpha	3	alpha	0.99767524	True
HP1508501T2D-D20	3	alpha	3	alpha	0.9977048	True
HP1508501T2D-D21	3	alpha	3	alpha	0.9953549	True
HP1508501T2D-D22	6	ductal	6	ductal	0.9941531	True
HP1508501T2D-D3	3	alpha	3	alpha	0.9977997	True
HP1508501T2D-D5	3	alpha	3	alpha	0.9977678	True
HP1508501T2D-D6	3	alpha	3	alpha	0.99774384	True
HP1508501T2D-D8	3	alpha	3	alpha	0.99767834	True
HP1508501T2D-D9	2	acinar	2	acinar	0.9892664	True
HP1508501T2D-E1	1	PSC	1	PSC	0.9926368	True
HP1508501T2D-E10	3	alpha	3	alpha	0.99775285	True
HP1508501T2D-E11	6	ductal	6	ductal	0.99396545	True
HP1508501T2D-E12	3	alpha	3	alpha	0.9976616	True
HP1508501T2D-E13	3	alpha	3	alpha	0.99773943	True
HP1508501T2D-E14	3	alpha	3	alpha	0.9976204	True
HP1508501T2D-E15	6	ductal	6	ductal	0.9942416	True
HP1508501T2D-E16	6	ductal	6	ductal	0.99058306	True
HP1508501T2D-E17	6	ductal	6	ductal	0.9927965	True
HP1508501T2D-E18	3	alpha	3	alpha	0.99771535	True
HP1508501T2D-E19	3	alpha	3	alpha	0.99768853	True
HP1508501T2D-E2	3	alpha	3	alpha	0.99759513	True
HP1508501T2D-E20	1	PSC	1	PSC	0.9937743	True
HP1508501T2D-E21	3	alpha	3	alpha	0.99778396	True
HP1508501T2D-E22	3	alpha	3	alpha	0.9977264	True
HP1508501T2D-E23	3	alpha	3	alpha	0.99777216	True
HP1508501T2D-E4	2	acinar	2	acinar	0.9912156	True
HP1508501T2D-E6	3	alpha	3	alpha	0.9978027	True
HP1508501T2D-E7	6	ductal	6	ductal	0.99434644	True
HP1508501T2D-E8	3	alpha	3	alpha	0.99776435	True
HP1508501T2D-E9	6	ductal	6	ductal	0.9944405	True
HP1508501T2D-F1	4	beta	4	beta	0.9980281	True
HP1508501T2D-F11	0	PP	3	alpha	0.7546065	True
HP1508501T2D-F12	6	ductal	6	ductal	0.993588	True
HP1508501T2D-F14	-1	<unknown_or_unlabeled>	9	macrophage	0.79260683	False
HP1508501T2D-F15	6	ductal	6	ductal	0.9942251	True
HP1508501T2D-F16	3	alpha	3	alpha	0.9978116	True
HP1508501T2D-F17	3	alpha	3	alpha	0.9977254	True
HP1508501T2D-F18	3	alpha	3	alpha	0.9976967	True
HP1508501T2D-F2	3	alpha	3	alpha	0.99757284	True
HP1508501T2D-F20	3	alpha	3	alpha	0.9975835	True
HP1508501T2D-F21	2	acinar	6	ductal	0.9570114	True
HP1508501T2D-F23	3	alpha	3	alpha	0.99774086	True
HP1508501T2D-F3	4	beta	4	beta	0.99776363	True
HP1508501T2D-F4	6	ductal	6	ductal	0.99216366	True
HP1508501T2D-F5	6	ductal	6	ductal	0.98085964	True
HP1508501T2D-F7	4	beta	4	beta	0.99765587	True
HP1508501T2D-F8	6	ductal	6	ductal	0.9933606	True
HP1508501T2D-F9	3	alpha	3	alpha	0.99773455	True
HP1508501T2D-G1	6	ductal	6	ductal	0.99343586	True
HP1508501T2D-G10	3	alpha	3	alpha	0.9977805	True
HP1508501T2D-G11	3	alpha	3	alpha	0.99779665	True
HP1508501T2D-G12	2	acinar	2	acinar	0.99184316	True
HP1508501T2D-G13	6	ductal	6	ductal	0.9929275	True
HP1508501T2D-G14	6	ductal	6	ductal	0.9925171	True
HP1508501T2D-G15	3	alpha	3	alpha	0.9977404	True
HP1508501T2D-G17	2	acinar	2	acinar	0.9939667	True
HP1508501T2D-G18	2	acinar	2	acinar	0.8997127	True
HP1508501T2D-G19	2	acinar	2	acinar	0.9939114	True
HP1508501T2D-G2	6	ductal	6	ductal	0.99426526	True
HP1508501T2D-G20	3	alpha	3	alpha	0.997661	True
HP1508501T2D-G21	3	alpha	3	alpha	0.9977641	True
HP1508501T2D-G22	0	PP	0	PP	0.52557003	True
HP1508501T2D-G23	2	acinar	2	acinar	0.9624259	True
HP1508501T2D-G3	6	ductal	6	ductal	0.99356717	True
HP1508501T2D-G4	3	alpha	3	alpha	0.99772996	True
HP1508501T2D-G5	4	beta	4	beta	0.9978606	True
HP1508501T2D-G6	3	alpha	3	alpha	0.99754155	True
HP1508501T2D-H10	3	alpha	3	alpha	0.9976598	True
HP1508501T2D-H11	5	delta	5	delta	0.62076277	True
HP1508501T2D-H12	10	mast	4	beta	0.28693014	True
HP1508501T2D-H13	6	ductal	6	ductal	0.99350137	True
HP1508501T2D-H14	6	ductal	6	ductal	0.99277335	True
HP1508501T2D-H15	3	alpha	3	alpha	0.9977824	True
HP1508501T2D-H16	2	acinar	2	acinar	0.9943604	True
HP1508501T2D-H17	6	ductal	6	ductal	0.9937822	True
HP1508501T2D-H18	0	PP	3	alpha	0.7014665	True
HP1508501T2D-H21	3	alpha	3	alpha	0.9977024	True
HP1508501T2D-H22	3	alpha	3	alpha	0.99708825	True
HP1508501T2D-H23	3	alpha	3	alpha	0.99772626	True
HP1508501T2D-H3	3	alpha	3	alpha	0.99775654	True
HP1508501T2D-H4	3	alpha	3	alpha	0.99770075	True
HP1508501T2D-H5	1	PSC	1	PSC	0.9938567	True
HP1508501T2D-H6	3	alpha	3	alpha	0.99772054	True
HP1508501T2D-I1	6	ductal	6	ductal	0.993375	True
HP1508501T2D-I10	2	acinar	2	acinar	0.9834589	True
HP1508501T2D-I11	6	ductal	6	ductal	0.9930275	True
HP1508501T2D-I12	6	ductal	6	ductal	0.99381703	True
HP1508501T2D-I13	6	ductal	6	ductal	0.993769	True
HP1508501T2D-I14	3	alpha	3	alpha	0.99776983	True
HP1508501T2D-I15	3	alpha	3	alpha	0.99767023	True
HP1508501T2D-I16	3	alpha	3	alpha	0.99760616	True
HP1508501T2D-I17	6	ductal	6	ductal	0.9917278	True
HP1508501T2D-I18	5	delta	5	delta	0.54111606	True
HP1508501T2D-I19	1	PSC	1	PSC	0.99349517	True
HP1508501T2D-I2	3	alpha	3	alpha	0.99761707	True
HP1508501T2D-I20	3	alpha	3	alpha	0.9977627	True
HP1508501T2D-I21	6	ductal	6	ductal	0.9928121	True
HP1508501T2D-I22	3	alpha	3	alpha	0.99770796	True
HP1508501T2D-I23	5	delta	5	delta	0.93760306	True
HP1508501T2D-I3	1	PSC	1	PSC	0.9934963	True
HP1508501T2D-I4	6	ductal	6	ductal	0.9937517	True
HP1508501T2D-I7	2	acinar	2	acinar	0.99331623	True
HP1508501T2D-I8	4	beta	4	beta	0.9977666	True
HP1508501T2D-J1	3	alpha	3	alpha	0.99750465	True
HP1508501T2D-J11	6	ductal	6	ductal	0.99306536	True
HP1508501T2D-J12	3	alpha	3	alpha	0.9976883	True
HP1508501T2D-J13	6	ductal	6	ductal	0.99374324	True
HP1508501T2D-J14	6	ductal	6	ductal	0.9934164	True
HP1508501T2D-J15	6	ductal	6	ductal	0.99395657	True
HP1508501T2D-J16	6	ductal	6	ductal	0.9938917	True
HP1508501T2D-J17	3	alpha	3	alpha	0.99767095	True
HP1508501T2D-J18	5	delta	4	beta	0.89904964	True
HP1508501T2D-J2	1	PSC	1	PSC	0.99319494	True
HP1508501T2D-J20	3	alpha	3	alpha	0.99773693	True
HP1508501T2D-J21	3	alpha	3	alpha	0.9977507	True
HP1508501T2D-J22	6	ductal	6	ductal	0.994128	True
HP1508501T2D-J23	2	acinar	2	acinar	0.6823825	True
HP1508501T2D-J3	3	alpha	3	alpha	0.99779785	True
HP1508501T2D-J4	3	alpha	3	alpha	0.99775535	True
HP1508501T2D-J5	3	alpha	3	alpha	0.9976943	True
HP1508501T2D-J6	4	beta	4	beta	0.9978962	True
HP1508501T2D-J8	3	alpha	3	alpha	0.9977621	True
HP1508501T2D-J9	3	alpha	3	alpha	0.99774075	True
HP1508501T2D-K1	3	alpha	3	alpha	0.9977963	True
HP1508501T2D-K12	3	alpha	3	alpha	0.99773955	True
HP1508501T2D-K14	3	alpha	3	alpha	0.9977629	True
HP1508501T2D-K15	6	ductal	6	ductal	0.99258304	True
HP1508501T2D-K16	0	PP	3	alpha	0.8421382	True
HP1508501T2D-K17	3	alpha	3	alpha	0.99783605	True
HP1508501T2D-K18	7	endothelial	7	endothelial	0.99218255	True
HP1508501T2D-K19	3	alpha	3	alpha	0.9977774	True
HP1508501T2D-K2	6	ductal	6	ductal	0.99422014	True
HP1508501T2D-K21	3	alpha	3	alpha	0.997766	True
HP1508501T2D-K22	6	ductal	6	ductal	0.99311584	True
HP1508501T2D-K23	6	ductal	6	ductal	0.99371743	True
HP1508501T2D-K3	6	ductal	6	ductal	0.9931773	True
HP1508501T2D-K4	3	alpha	3	alpha	0.9978619	True
HP1508501T2D-K6	2	acinar	2	acinar	0.9756218	True
HP1508501T2D-K7	6	ductal	6	ductal	0.9939095	True
HP1508501T2D-K9	3	alpha	3	alpha	0.9977875	True
HP1508501T2D-L10	1	PSC	1	PSC	0.9939534	True
HP1508501T2D-L11	6	ductal	6	ductal	0.9937232	True
HP1508501T2D-L13	3	alpha	3	alpha	0.99768615	True
HP1508501T2D-L14	3	alpha	3	alpha	0.9976283	True
HP1508501T2D-L15	1	PSC	1	PSC	0.9942795	True
HP1508501T2D-L16	3	alpha	3	alpha	0.9977569	True
HP1508501T2D-L17	3	alpha	3	alpha	0.99766123	True
HP1508501T2D-L18	3	alpha	3	alpha	0.99773186	True
HP1508501T2D-L2	2	acinar	2	acinar	0.82624453	True
HP1508501T2D-L20	6	ductal	6	ductal	0.99391806	True
HP1508501T2D-L21	6	ductal	6	ductal	0.99342525	True
HP1508501T2D-L23	6	ductal	6	ductal	0.9942259	True
HP1508501T2D-L5	6	ductal	6	ductal	0.9934906	True
HP1508501T2D-L6	3	alpha	3	alpha	0.9975713	True
HP1508501T2D-L7	3	alpha	3	alpha	0.9977406	True
HP1508501T2D-L9	6	ductal	6	ductal	0.9935807	True
HP1508501T2D-M1	10	mast	9	macrophage	0.24464571	True
HP1508501T2D-M10	1	PSC	1	PSC	0.9939492	True
HP1508501T2D-M11	2	acinar	2	acinar	0.9941414	True
HP1508501T2D-M12	3	alpha	3	alpha	0.99775475	True
HP1508501T2D-M14	4	beta	4	beta	0.9978719	True
HP1508501T2D-M15	6	ductal	6	ductal	0.991503	True
HP1508501T2D-M16	1	PSC	1	PSC	0.9935104	True
HP1508501T2D-M17	6	ductal	6	ductal	0.9936	True
HP1508501T2D-M18	2	acinar	2	acinar	0.99477166	True
HP1508501T2D-M19	3	alpha	3	alpha	0.9977157	True
HP1508501T2D-M20	3	alpha	3	alpha	0.9977155	True
HP1508501T2D-M22	6	ductal	6	ductal	0.9938765	True
HP1508501T2D-M23	6	ductal	6	ductal	0.9947541	True
HP1508501T2D-M3	6	ductal	6	ductal	0.9924735	True
HP1508501T2D-M4	3	alpha	3	alpha	0.9977157	True
HP1508501T2D-M5	3	alpha	3	alpha	0.99775237	True
HP1508501T2D-M7	2	acinar	2	acinar	0.9926311	True
HP1508501T2D-M8	3	alpha	3	alpha	0.99773407	True
HP1508501T2D-M9	3	alpha	3	alpha	0.9977981	True
HP1508501T2D-N1	6	ductal	6	ductal	0.9932087	True
HP1508501T2D-N10	6	ductal	6	ductal	0.9936376	True
HP1508501T2D-N11	3	alpha	3	alpha	0.9976871	True
HP1508501T2D-N13	2	acinar	2	acinar	0.99133986	True
HP1508501T2D-N15	6	ductal	6	ductal	0.993302	True
HP1508501T2D-N17	6	ductal	6	ductal	0.9940142	True
HP1508501T2D-N18	0	PP	5	delta	0.9547226	True
HP1508501T2D-N19	6	ductal	6	ductal	0.9943581	True
HP1508501T2D-N2	6	ductal	6	ductal	0.9934695	True
HP1508501T2D-N20	3	alpha	3	alpha	0.99768865	True
HP1508501T2D-N21	2	acinar	6	ductal	0.8395761	True
HP1508501T2D-N22	6	ductal	6	ductal	0.99426126	True
HP1508501T2D-N23	6	ductal	6	ductal	0.99345994	True
HP1508501T2D-N4	3	alpha	3	alpha	0.9977678	True
HP1508501T2D-N5	3	alpha	3	alpha	0.99772435	True
HP1508501T2D-N6	6	ductal	6	ductal	0.99426633	True
HP1508501T2D-N7	6	ductal	6	ductal	0.99381834	True
HP1508501T2D-N8	6	ductal	6	ductal	0.9934221	True
HP1508501T2D-N9	6	ductal	6	ductal	0.99406403	True
HP1508501T2D-O1	3	alpha	3	alpha	0.9977174	True
HP1508501T2D-O11	2	acinar	6	ductal	0.9635454	True
HP1508501T2D-O12	6	ductal	6	ductal	0.9942503	True
HP1508501T2D-O13	2	acinar	2	acinar	0.98303854	True
HP1508501T2D-O14	3	alpha	3	alpha	0.9977055	True
HP1508501T2D-O3	6	ductal	6	ductal	0.99505913	True
HP1508501T2D-O4	4	beta	4	beta	0.9978422	True
HP1508501T2D-O7	2	acinar	2	acinar	0.98910564	True
HP1508501T2D-O8	6	ductal	6	ductal	0.9930902	True
HP1508501T2D-O9	6	ductal	6	ductal	0.99379784	True
HP1508501T2D-P11	3	alpha	3	alpha	0.99759054	True
HP1508501T2D-P14	2	acinar	3	alpha	0.9807989	True
HP1508501T2D-P2	2	acinar	6	ductal	0.9932069	True
HP1508501T2D-P3	3	alpha	3	alpha	0.9977284	True
HP1508501T2D-P4	0	PP	3	alpha	0.91903394	True
HP1508501T2D-P6	3	alpha	3	alpha	0.9977291	True
HP1508501T2D-P7	3	alpha	3	alpha	0.997718	True
HP1508501T2D-P8	3	alpha	3	alpha	0.99774987	True
HP1508501T2D-P9	3	alpha	3	alpha	0.99746007	True
HP1509101-A3	3	alpha	3	alpha	0.99740964	True
HP1509101-A7	3	alpha	3	alpha	0.9971825	True
HP1509101-A8	6	ductal	6	ductal	0.99458635	True
HP1509101-B10	3	alpha	3	alpha	0.99764156	True
HP1509101-B13	4	beta	4	beta	0.9978643	True
HP1509101-B16	2	acinar	6	ductal	0.9096279	True
HP1509101-B17	1	PSC	1	PSC	0.9937738	True
HP1509101-B19	6	ductal	6	ductal	0.9925564	True
HP1509101-B20	3	alpha	3	alpha	0.99748445	True
HP1509101-B23	6	ductal	6	ductal	0.9914881	True
HP1509101-C21	6	ductal	6	ductal	0.9936429	True
HP1509101-C22	3	alpha	3	alpha	0.99762964	True
HP1509101-C23	4	beta	4	beta	0.9977448	True
HP1509101-D17	6	ductal	6	ductal	0.99316543	True
HP1509101-D18	4	beta	4	beta	0.99779177	True
HP1509101-D19	3	alpha	3	alpha	0.9976063	True
HP1509101-D22	5	delta	5	delta	0.72832626	True
HP1509101-D23	6	ductal	6	ductal	0.99358904	True
HP1509101-D5	3	alpha	3	alpha	0.99764913	True
HP1509101-D8	3	alpha	3	alpha	0.9977921	True
HP1509101-E19	3	alpha	3	alpha	0.997551	True
HP1509101-E22	4	beta	4	beta	0.99795216	True
HP1509101-E3	6	ductal	6	ductal	0.99321824	True
HP1509101-E4	5	delta	5	delta	0.92910606	True
HP1509101-F18	2	acinar	6	ductal	0.9939062	True
HP1509101-F19	3	alpha	3	alpha	0.99759007	True
HP1509101-F21	0	PP	3	alpha	0.9929289	True
HP1509101-F22	6	ductal	6	ductal	0.99243927	True
HP1509101-F23	7	endothelial	7	endothelial	0.99166447	True
HP1509101-F4	3	alpha	3	alpha	0.99761534	True
HP1509101-F7	6	ductal	6	ductal	0.9937477	True
HP1509101-G16	3	alpha	3	alpha	0.99729115	True
HP1509101-G18	3	alpha	3	alpha	0.99774665	True
HP1509101-G19	3	alpha	3	alpha	0.99774075	True
HP1509101-G9	4	beta	4	beta	0.9979213	True
HP1509101-H1	3	alpha	3	alpha	0.9968353	True
HP1509101-H13	5	delta	5	delta	0.9403182	True
HP1509101-H20	6	ductal	6	ductal	0.9928763	True
HP1509101-H21	3	alpha	3	alpha	0.9971904	True
HP1509101-H22	3	alpha	3	alpha	0.99753475	True
HP1509101-H23	3	alpha	3	alpha	0.9975777	True
HP1509101-H7	5	delta	4	beta	0.94838965	True
HP1509101-I10	3	alpha	3	alpha	0.9975776	True
HP1509101-I12	6	ductal	6	ductal	0.99228036	True
HP1509101-I17	7	endothelial	7	endothelial	0.99160916	True
HP1509101-I21	3	alpha	3	alpha	0.99761236	True
HP1509101-I23	3	alpha	3	alpha	0.9976223	True
HP1509101-I5	3	alpha	3	alpha	0.9976792	True
HP1509101-J12	3	alpha	3	alpha	0.99753124	True
HP1509101-J15	3	alpha	3	alpha	0.99762815	True
HP1509101-J16	5	delta	4	beta	0.5204521	True
HP1509101-J19	4	beta	4	beta	0.9976936	True
HP1509101-J20	3	alpha	3	alpha	0.99746203	True
HP1509101-J22	6	ductal	6	ductal	0.9924495	True
HP1509101-J23	6	ductal	6	ductal	0.9937163	True
HP1509101-J4	3	alpha	3	alpha	0.99735415	True
HP1509101-J8	3	alpha	3	alpha	0.9975399	True
HP1509101-J9	6	ductal	6	ductal	0.9928624	True
HP1509101-K10	6	ductal	6	ductal	0.9916854	True
HP1509101-K15	6	ductal	6	ductal	0.9940392	True
HP1509101-K19	6	ductal	6	ductal	0.99388593	True
HP1509101-K22	3	alpha	3	alpha	0.9975979	True
HP1509101-K9	4	beta	4	beta	0.9977191	True
HP1509101-L11	3	alpha	3	alpha	0.99769104	True
HP1509101-L13	6	ductal	6	ductal	0.9931096	True
HP1509101-L14	4	beta	4	beta	0.9979386	True
HP1509101-L16	6	ductal	6	ductal	0.9913658	True
HP1509101-L17	3	alpha	3	alpha	0.9418067	True
HP1509101-L18	1	PSC	7	endothelial	0.9866818	True
HP1509101-L19	5	delta	3	alpha	0.95055026	True
HP1509101-L23	5	delta	5	delta	0.84330255	True
HP1509101-L8	3	alpha	3	alpha	0.99771404	True
HP1509101-M10	3	alpha	3	alpha	0.99768174	True
HP1509101-M12	3	alpha	3	alpha	0.99770147	True
HP1509101-N17	3	alpha	3	alpha	0.9975183	True
HP1509101-N18	3	alpha	3	alpha	0.9973091	True
HP1509101-N20	3	alpha	3	alpha	0.9976439	True
HP1509101-N23	4	beta	4	beta	0.997895	True
HP1509101-N9	3	alpha	3	alpha	0.99766624	True
HP1509101-O1	5	delta	5	delta	0.96653426	True
HP1509101-O13	3	alpha	3	alpha	0.9976051	True
HP1509101-O14	6	ductal	6	ductal	0.99249345	True
HP1509101-O16	3	alpha	3	alpha	0.9975352	True
HP1509101-O18	4	beta	4	beta	0.9980045	True
HP1509101-O4	5	delta	4	beta	0.6396552	True
HP1509101-O7	3	alpha	3	alpha	0.99766076	True
HP1509101-O9	3	alpha	3	alpha	0.9973814	True
HP1509101-P12	5	delta	4	beta	0.7531132	True
HP1509101-P17	6	ductal	6	ductal	0.9934777	True
HP1509101-P18	3	alpha	3	alpha	0.9977543	True
HP1509101-P20	3	alpha	3	alpha	0.9976554	True
HP1509101-P23	1	PSC	1	PSC	0.7447795	True
HP1509101-P4	3	alpha	3	alpha	0.9975988	True
HP1509101-P8	6	ductal	3	alpha	0.4813582	True
HP1509101-P9	6	ductal	6	ductal	0.8992027	True
HP1525301T2D-A1	6	ductal	6	ductal	0.9946325	True
HP1525301T2D-A11	3	alpha	3	alpha	0.99776983	True
HP1525301T2D-A12	6	ductal	6	ductal	0.9936504	True
HP1525301T2D-A13	6	ductal	6	ductal	0.994468	True
HP1525301T2D-A14	6	ductal	6	ductal	0.994949	True
HP1525301T2D-A15	6	ductal	6	ductal	0.9948585	True
HP1525301T2D-A16	0	PP	3	alpha	0.9872639	True
HP1525301T2D-A17	6	ductal	6	ductal	0.99439174	True
HP1525301T2D-A19	6	ductal	6	ductal	0.9945833	True
HP1525301T2D-A3	3	alpha	3	alpha	0.9977228	True
HP1525301T2D-A5	3	alpha	3	alpha	0.9977876	True
HP1525301T2D-A6	3	alpha	3	alpha	0.9977946	True
HP1525301T2D-A7	5	delta	5	delta	0.98049396	True
HP1525301T2D-A9	3	alpha	3	alpha	0.9977519	True
HP1525301T2D-B1	3	alpha	3	alpha	0.99777645	True
HP1525301T2D-B10	6	ductal	6	ductal	0.99420434	True
HP1525301T2D-B11	3	alpha	3	alpha	0.997775	True
HP1525301T2D-B12	2	acinar	6	ductal	0.9626578	True
HP1525301T2D-B13	3	alpha	3	alpha	0.9978059	True
HP1525301T2D-B14	6	ductal	6	ductal	0.9943968	True
HP1525301T2D-B15	3	alpha	3	alpha	0.9976156	True
HP1525301T2D-B18	3	alpha	3	alpha	0.9978217	True
HP1525301T2D-B21	6	ductal	6	ductal	0.9933348	True
HP1525301T2D-B24	6	ductal	6	ductal	0.9950034	True
HP1525301T2D-B3	6	ductal	6	ductal	0.9948532	True
HP1525301T2D-B4	3	alpha	3	alpha	0.9977404	True
HP1525301T2D-B5	0	PP	3	alpha	0.99606776	True
HP1525301T2D-B6	1	PSC	1	PSC	0.9934094	True
HP1525301T2D-B7	3	alpha	3	alpha	0.99774534	True
HP1525301T2D-B8	0	PP	3	alpha	0.98145914	True
HP1525301T2D-C1	3	alpha	3	alpha	0.99778867	True
HP1525301T2D-C10	3	alpha	3	alpha	0.9977781	True
HP1525301T2D-C11	6	ductal	6	ductal	0.99404955	True
HP1525301T2D-C13	6	ductal	6	ductal	0.99466056	True
HP1525301T2D-C14	2	acinar	2	acinar	0.9672946	True
HP1525301T2D-C15	6	ductal	6	ductal	0.9941186	True
HP1525301T2D-C16	3	alpha	3	alpha	0.9977943	True
HP1525301T2D-C17	6	ductal	6	ductal	0.99438214	True
HP1525301T2D-C19	6	ductal	6	ductal	0.99462974	True
HP1525301T2D-C2	3	alpha	3	alpha	0.9977552	True
HP1525301T2D-C21	6	ductal	6	ductal	0.99331975	True
HP1525301T2D-C22	6	ductal	6	ductal	0.99479526	True
HP1525301T2D-C23	2	acinar	2	acinar	0.96128243	True
HP1525301T2D-C24	6	ductal	6	ductal	0.993439	True
HP1525301T2D-C3	6	ductal	6	ductal	0.9949767	True
HP1525301T2D-C4	6	ductal	6	ductal	0.9946179	True
HP1525301T2D-C5	3	alpha	3	alpha	0.9977907	True
HP1525301T2D-C6	6	ductal	6	ductal	0.9946213	True
HP1525301T2D-C7	2	acinar	6	ductal	0.9283377	True
HP1525301T2D-C8	3	alpha	3	alpha	0.9977348	True
HP1525301T2D-C9	6	ductal	6	ductal	0.9946649	True
HP1525301T2D-D1	6	ductal	6	ductal	0.9924034	True
HP1525301T2D-D10	2	acinar	6	ductal	0.96973175	True
HP1525301T2D-D11	3	alpha	3	alpha	0.9978098	True
HP1525301T2D-D12	0	PP	0	PP	0.665569	True
HP1525301T2D-D14	6	ductal	6	ductal	0.9929128	True
HP1525301T2D-D15	6	ductal	6	ductal	0.9945913	True
HP1525301T2D-D19	5	delta	5	delta	0.98478186	True
HP1525301T2D-D2	6	ductal	6	ductal	0.99488515	True
HP1525301T2D-D21	1	PSC	1	PSC	0.9906849	True
HP1525301T2D-D22	6	ductal	6	ductal	0.99409676	True
HP1525301T2D-D23	4	beta	4	beta	0.9980458	True
HP1525301T2D-D3	6	ductal	6	ductal	0.99308527	True
HP1525301T2D-D4	6	ductal	6	ductal	0.99511945	True
HP1525301T2D-D6	2	acinar	6	ductal	0.91767186	True
HP1525301T2D-D8	6	ductal	6	ductal	0.99467045	True
HP1525301T2D-D9	3	alpha	3	alpha	0.99779236	True
HP1525301T2D-E1	3	alpha	3	alpha	0.9977162	True
HP1525301T2D-E11	6	ductal	6	ductal	0.9901424	True
HP1525301T2D-E12	1	PSC	1	PSC	0.99343395	True
HP1525301T2D-E13	3	alpha	3	alpha	0.9977731	True
HP1525301T2D-E15	6	ductal	6	ductal	0.99388134	True
HP1525301T2D-E16	3	alpha	3	alpha	0.9976387	True
HP1525301T2D-E17	6	ductal	6	ductal	0.9949773	True
HP1525301T2D-E19	0	PP	3	alpha	0.5733523	True
HP1525301T2D-E2	2	acinar	2	acinar	0.99214643	True
HP1525301T2D-E22	3	alpha	3	alpha	0.9978064	True
HP1525301T2D-E23	3	alpha	3	alpha	0.9977126	True
HP1525301T2D-E24	1	PSC	1	PSC	0.993468	True
HP1525301T2D-E3	0	PP	3	alpha	0.99347746	True
HP1525301T2D-E5	1	PSC	1	PSC	0.9938327	True
HP1525301T2D-E7	3	alpha	3	alpha	0.9977679	True
HP1525301T2D-E8	3	alpha	3	alpha	0.9977156	True
HP1525301T2D-F10	2	acinar	6	ductal	0.9714679	True
HP1525301T2D-F12	6	ductal	6	ductal	0.99499243	True
HP1525301T2D-F15	4	beta	4	beta	0.997903	True
HP1525301T2D-F16	6	ductal	6	ductal	0.9918937	True
HP1525301T2D-F19	3	alpha	3	alpha	0.9977386	True
HP1525301T2D-F2	3	alpha	3	alpha	0.99776757	True
HP1525301T2D-F21	6	ductal	6	ductal	0.9943211	True
HP1525301T2D-F23	4	beta	4	beta	0.99791414	True
HP1525301T2D-F24	1	PSC	1	PSC	0.9938153	True
HP1525301T2D-F3	2	acinar	6	ductal	0.8995023	True
HP1525301T2D-F4	3	alpha	3	alpha	0.99776745	True
HP1525301T2D-F5	2	acinar	6	ductal	0.9048075	True
HP1525301T2D-F7	1	PSC	1	PSC	0.993785	True
HP1525301T2D-F8	6	ductal	6	ductal	0.99528474	True
HP1525301T2D-G1	3	alpha	3	alpha	0.99776745	True
HP1525301T2D-G10	6	ductal	6	ductal	0.993658	True
HP1525301T2D-G11	3	alpha	3	alpha	0.99772733	True
HP1525301T2D-G13	6	ductal	6	ductal	0.99414355	True
HP1525301T2D-G14	6	ductal	6	ductal	0.9948413	True
HP1525301T2D-G15	3	alpha	3	alpha	0.9977316	True
HP1525301T2D-G17	2	acinar	6	ductal	0.97793394	True
HP1525301T2D-G18	3	alpha	3	alpha	0.9977544	True
HP1525301T2D-G19	3	alpha	3	alpha	0.9977278	True
HP1525301T2D-G20	6	ductal	6	ductal	0.99470985	True
HP1525301T2D-G21	6	ductal	6	ductal	0.99503124	True
HP1525301T2D-G22	3	alpha	3	alpha	0.9977812	True
HP1525301T2D-G23	0	PP	3	alpha	0.8572085	True
HP1525301T2D-G3	6	ductal	6	ductal	0.99401045	True
HP1525301T2D-G4	3	alpha	3	alpha	0.9977884	True
HP1525301T2D-G5	6	ductal	6	ductal	0.994676	True
HP1525301T2D-G8	4	beta	4	beta	0.9981712	True
HP1525301T2D-G9	6	ductal	6	ductal	0.99430096	True
HP1525301T2D-H1	6	ductal	6	ductal	0.9947897	True
HP1525301T2D-H10	4	beta	5	delta	0.52550715	True
HP1525301T2D-H11	6	ductal	6	ductal	0.99385095	True
HP1525301T2D-H12	3	alpha	3	alpha	0.9977329	True
HP1525301T2D-H14	6	ductal	6	ductal	0.99534154	True
HP1525301T2D-H16	6	ductal	6	ductal	0.9942099	True
HP1525301T2D-H17	6	ductal	6	ductal	0.99459755	True
HP1525301T2D-H18	6	ductal	6	ductal	0.9950605	True
HP1525301T2D-H19	3	alpha	3	alpha	0.99771523	True
HP1525301T2D-H2	6	ductal	6	ductal	0.99451643	True
HP1525301T2D-H20	3	alpha	3	alpha	0.9977769	True
HP1525301T2D-H23	3	alpha	3	alpha	0.99778616	True
HP1525301T2D-H24	6	ductal	6	ductal	0.9946715	True
HP1525301T2D-H3	6	ductal	6	ductal	0.99395233	True
HP1525301T2D-H6	3	alpha	3	alpha	0.99777204	True
HP1525301T2D-H7	3	alpha	3	alpha	0.9977227	True
HP1525301T2D-H8	2	acinar	2	acinar	0.6288912	True
HP1525301T2D-H9	2	acinar	2	acinar	0.5593709	True
HP1525301T2D-I1	6	ductal	6	ductal	0.99470335	True
HP1525301T2D-I10	6	ductal	6	ductal	0.9950406	True
HP1525301T2D-I11	-1	<unknown_or_unlabeled>	9	macrophage	0.9217883	False
HP1525301T2D-I12	3	alpha	3	alpha	0.99778336	True
HP1525301T2D-I15	6	ductal	6	ductal	0.99496233	True
HP1525301T2D-I16	6	ductal	6	ductal	0.9942578	True
HP1525301T2D-I17	6	ductal	6	ductal	0.9936945	True
HP1525301T2D-I18	6	ductal	6	ductal	0.9942615	True
HP1525301T2D-I19	6	ductal	6	ductal	0.9946867	True
HP1525301T2D-I2	3	alpha	3	alpha	0.9977549	True
HP1525301T2D-I21	3	alpha	3	alpha	0.9977295	True
HP1525301T2D-I22	6	ductal	6	ductal	0.9938437	True
HP1525301T2D-I23	6	ductal	6	ductal	0.9943772	True
HP1525301T2D-I24	6	ductal	6	ductal	0.9945023	True
HP1525301T2D-I3	6	ductal	6	ductal	0.99339676	True
HP1525301T2D-I4	3	alpha	3	alpha	0.9977794	True
HP1525301T2D-I5	3	alpha	3	alpha	0.9977658	True
HP1525301T2D-I6	4	beta	4	beta	0.9978897	True
HP1525301T2D-I7	3	alpha	3	alpha	0.9977424	True
HP1525301T2D-I8	6	ductal	6	ductal	0.99451244	True
HP1525301T2D-J1	3	alpha	3	alpha	0.99772733	True
HP1525301T2D-J10	5	delta	5	delta	0.98209643	True
HP1525301T2D-J13	3	alpha	3	alpha	0.997789	True
HP1525301T2D-J14	3	alpha	3	alpha	0.99768555	True
HP1525301T2D-J15	6	ductal	6	ductal	0.99432665	True
HP1525301T2D-J16	2	acinar	2	acinar	0.93466115	True
HP1525301T2D-J17	1	PSC	1	PSC	0.9937896	True
HP1525301T2D-J18	0	PP	0	PP	0.48009825	True
HP1525301T2D-J19	3	alpha	3	alpha	0.99775726	True
HP1525301T2D-J2	3	alpha	3	alpha	0.9977835	True
HP1525301T2D-J20	6	ductal	6	ductal	0.99360245	True
HP1525301T2D-J21	6	ductal	6	ductal	0.994491	True
HP1525301T2D-J23	2	acinar	2	acinar	0.8871405	True
HP1525301T2D-J4	3	alpha	3	alpha	0.99773216	True
HP1525301T2D-J5	2	acinar	2	acinar	0.97286624	True
HP1525301T2D-J7	6	ductal	6	ductal	0.99438196	True
HP1525301T2D-J8	3	alpha	3	alpha	0.9977628	True
HP1525301T2D-J9	6	ductal	6	ductal	0.9950663	True
HP1525301T2D-K1	6	ductal	6	ductal	0.994743	True
HP1525301T2D-K10	1	PSC	1	PSC	0.99366885	True
HP1525301T2D-K11	6	ductal	6	ductal	0.9941989	True
HP1525301T2D-K12	6	ductal	6	ductal	0.9916141	True
HP1525301T2D-K14	6	ductal	6	ductal	0.9943246	True
HP1525301T2D-K15	3	alpha	3	alpha	0.997673	True
HP1525301T2D-K16	6	ductal	6	ductal	0.9933444	True
HP1525301T2D-K17	6	ductal	6	ductal	0.9941293	True
HP1525301T2D-K18	6	ductal	6	ductal	0.99399877	True
HP1525301T2D-K19	3	alpha	3	alpha	0.99773204	True
HP1525301T2D-K2	6	ductal	6	ductal	0.99392927	True
HP1525301T2D-K21	6	ductal	6	ductal	0.9941521	True
HP1525301T2D-K22	3	alpha	3	alpha	0.9977591	True
HP1525301T2D-K23	3	alpha	3	alpha	0.9977028	True
HP1525301T2D-K3	3	alpha	3	alpha	0.9976826	True
HP1525301T2D-K5	2	acinar	2	acinar	0.9222495	True
HP1525301T2D-K6	6	ductal	6	ductal	0.9931237	True
HP1525301T2D-K7	2	acinar	2	acinar	0.9910971	True
HP1525301T2D-K8	2	acinar	6	ductal	0.97468716	True
HP1525301T2D-L1	6	ductal	6	ductal	0.99482405	True
HP1525301T2D-L13	6	ductal	6	ductal	0.9945568	True
HP1525301T2D-L14	4	beta	4	beta	0.99757093	True
HP1525301T2D-L15	3	alpha	3	alpha	0.9977824	True
HP1525301T2D-L16	6	ductal	6	ductal	0.9940348	True
HP1525301T2D-L17	4	beta	4	beta	0.9980399	True
HP1525301T2D-L2	6	ductal	6	ductal	0.99449146	True
HP1525301T2D-L20	6	ductal	6	ductal	0.9944665	True
HP1525301T2D-L22	3	alpha	3	alpha	0.9977361	True
HP1525301T2D-L23	6	ductal	6	ductal	0.99321824	True
HP1525301T2D-L24	3	alpha	3	alpha	0.99779475	True
HP1525301T2D-L4	6	ductal	6	ductal	0.9946036	True
HP1525301T2D-L5	2	acinar	6	ductal	0.9388287	True
HP1525301T2D-L6	6	ductal	6	ductal	0.9940088	True
HP1525301T2D-L8	1	PSC	1	PSC	0.9938547	True
HP1525301T2D-L9	3	alpha	3	alpha	0.99779093	True
HP1525301T2D-M10	3	alpha	3	alpha	0.9977812	True
HP1525301T2D-M11	6	ductal	6	ductal	0.9942668	True
HP1525301T2D-M12	6	ductal	6	ductal	0.9946668	True
HP1525301T2D-M13	3	alpha	3	alpha	0.9977702	True
HP1525301T2D-M15	6	ductal	6	ductal	0.9950706	True
HP1525301T2D-M16	3	alpha	3	alpha	0.9977738	True
HP1525301T2D-M17	5	delta	5	delta	0.9841315	True
HP1525301T2D-M2	1	PSC	1	PSC	0.9940714	True
HP1525301T2D-M20	6	ductal	6	ductal	0.99350995	True
HP1525301T2D-M22	6	ductal	6	ductal	0.9940481	True
HP1525301T2D-M5	2	acinar	6	ductal	0.99236417	True
HP1525301T2D-M6	6	ductal	6	ductal	0.9949726	True
HP1525301T2D-M7	4	beta	4	beta	0.99798393	True
HP1525301T2D-M8	3	alpha	3	alpha	0.9977397	True
HP1525301T2D-M9	6	ductal	6	ductal	0.9939627	True
HP1525301T2D-N1	6	ductal	6	ductal	0.9948872	True
HP1525301T2D-N10	2	acinar	6	ductal	0.9947207	True
HP1525301T2D-N11	6	ductal	6	ductal	0.9934516	True
HP1525301T2D-N12	6	ductal	6	ductal	0.99483335	True
HP1525301T2D-N13	3	alpha	3	alpha	0.9977818	True
HP1525301T2D-N14	6	ductal	6	ductal	0.99487925	True
HP1525301T2D-N16	6	ductal	6	ductal	0.9950612	True
HP1525301T2D-N18	4	beta	4	beta	0.9981019	True
HP1525301T2D-N20	6	ductal	6	ductal	0.99361414	True
HP1525301T2D-N21	0	PP	3	alpha	0.99656624	True
HP1525301T2D-N23	3	alpha	3	alpha	0.9978309	True
HP1525301T2D-N24	3	alpha	3	alpha	0.9977779	True
HP1525301T2D-N3	4	beta	4	beta	0.9980286	True
HP1525301T2D-N4	6	ductal	6	ductal	0.9951298	True
HP1525301T2D-N5	6	ductal	6	ductal	0.99469924	True
HP1525301T2D-N6	6	ductal	6	ductal	0.9940309	True
HP1525301T2D-N7	3	alpha	3	alpha	0.997595	True
HP1525301T2D-N8	3	alpha	3	alpha	0.9977608	True
HP1525301T2D-N9	3	alpha	3	alpha	0.9978169	True
HP1525301T2D-O1	2	acinar	2	acinar	0.9780018	True
HP1525301T2D-O11	0	PP	3	alpha	0.53779685	True
HP1525301T2D-O13	3	alpha	3	alpha	0.9977736	True
HP1525301T2D-O14	6	ductal	6	ductal	0.99485296	True
HP1525301T2D-O16	6	ductal	6	ductal	0.9938258	True
HP1525301T2D-O17	6	ductal	6	ductal	0.9935882	True
HP1525301T2D-O18	3	alpha	3	alpha	0.99775237	True
HP1525301T2D-O2	5	delta	5	delta	0.98154885	True
HP1525301T2D-O20	6	ductal	6	ductal	0.9933822	True
HP1525301T2D-O22	6	ductal	6	ductal	0.9946479	True
HP1525301T2D-O23	6	ductal	6	ductal	0.9939194	True
HP1525301T2D-O24	6	ductal	6	ductal	0.99485886	True
HP1525301T2D-O3	2	acinar	2	acinar	0.9373694	True
HP1525301T2D-O5	3	alpha	3	alpha	0.9977404	True
HP1525301T2D-O6	3	alpha	3	alpha	0.99758387	True
HP1525301T2D-O7	6	ductal	6	ductal	0.98990005	True
HP1525301T2D-O9	1	PSC	1	PSC	0.9930716	True
HP1525301T2D-P1	6	ductal	6	ductal	0.994636	True
HP1525301T2D-P11	3	alpha	3	alpha	0.9977291	True
HP1525301T2D-P13	6	ductal	6	ductal	0.994636	True
HP1525301T2D-P14	6	ductal	6	ductal	0.9948237	True
HP1525301T2D-P15	7	endothelial	7	endothelial	0.9926745	True
HP1525301T2D-P16	1	PSC	1	PSC	0.993658	True
HP1525301T2D-P17	3	alpha	3	alpha	0.9977939	True
HP1525301T2D-P18	3	alpha	3	alpha	0.99776804	True
HP1525301T2D-P19	6	ductal	6	ductal	0.9945885	True
HP1525301T2D-P2	6	ductal	6	ductal	0.9940347	True
HP1525301T2D-P20	6	ductal	6	ductal	0.9939148	True
HP1525301T2D-P21	6	ductal	6	ductal	0.99396175	True
HP1525301T2D-P22	3	alpha	3	alpha	0.9978364	True
HP1525301T2D-P23	6	ductal	6	ductal	0.9951213	True
HP1525301T2D-P24	3	alpha	3	alpha	0.9978077	True
HP1525301T2D-P4	6	ductal	6	ductal	0.99287826	True
HP1525301T2D-P5	6	ductal	6	ductal	0.99473083	True
HP1525301T2D-P6	3	alpha	3	alpha	0.997681	True
HP1525301T2D-P8	3	alpha	3	alpha	0.99777657	True
HP1525301T2D-P9	3	alpha	3	alpha	0.99775827	True
HP1526901T2D-A1	4	beta	4	beta	0.99797493	True
HP1526901T2D-A10	5	delta	5	delta	0.96782714	True
HP1526901T2D-A11	6	ductal	6	ductal	0.99413705	True
HP1526901T2D-A12	3	alpha	3	alpha	0.9975407	True
HP1526901T2D-A14	4	beta	4	beta	0.99790984	True
HP1526901T2D-A15	0	PP	0	PP	0.6142338	True
HP1526901T2D-A16	5	delta	5	delta	0.94148386	True
HP1526901T2D-A17	3	alpha	3	alpha	0.9975975	True
HP1526901T2D-A19	4	beta	4	beta	0.99793744	True
HP1526901T2D-A2	4	beta	4	beta	0.9981566	True
HP1526901T2D-A20	0	PP	0	PP	0.54639846	True
HP1526901T2D-A21	6	ductal	6	ductal	0.99175733	True
HP1526901T2D-A22	5	delta	4	beta	0.9903702	True
HP1526901T2D-A24	6	ductal	6	ductal	0.9939521	True
HP1526901T2D-A3	3	alpha	3	alpha	0.9975969	True
HP1526901T2D-A4	5	delta	5	delta	0.50610006	True
HP1526901T2D-A5	4	beta	4	beta	0.997954	True
HP1526901T2D-A6	2	acinar	2	acinar	0.9944786	True
HP1526901T2D-A7	3	alpha	3	alpha	0.99773514	True
HP1526901T2D-A8	3	alpha	3	alpha	0.9977476	True
HP1526901T2D-A9	3	alpha	3	alpha	0.9976809	True
HP1526901T2D-B1	4	beta	4	beta	0.99795735	True
HP1526901T2D-B10	4	beta	4	beta	0.9979856	True
HP1526901T2D-B11	3	alpha	3	alpha	0.9977598	True
HP1526901T2D-B12	3	alpha	3	alpha	0.99768066	True
HP1526901T2D-B13	0	PP	3	alpha	0.87851226	True
HP1526901T2D-B14	3	alpha	3	alpha	0.99767846	True
HP1526901T2D-B17	3	alpha	3	alpha	0.99761885	True
HP1526901T2D-B18	5	delta	5	delta	0.98246336	True
HP1526901T2D-B20	3	alpha	3	alpha	0.99769634	True
HP1526901T2D-B23	6	ductal	6	ductal	0.99327636	True
HP1526901T2D-B24	6	ductal	6	ductal	0.99434435	True
HP1526901T2D-B3	6	ductal	6	ductal	0.9940759	True
HP1526901T2D-B4	4	beta	4	beta	0.99803	True
HP1526901T2D-B5	6	ductal	6	ductal	0.9932045	True
HP1526901T2D-B6	5	delta	5	delta	0.85608715	True
HP1526901T2D-B7	5	delta	5	delta	0.94237155	True
HP1526901T2D-B9	4	beta	4	beta	0.9980288	True
HP1526901T2D-C1	5	delta	5	delta	0.9797737	True
HP1526901T2D-C10	4	beta	4	beta	0.9980077	True
HP1526901T2D-C11	4	beta	4	beta	0.9980653	True
HP1526901T2D-C13	0	PP	3	alpha	0.99736756	True
HP1526901T2D-C14	6	ductal	6	ductal	0.99440235	True
HP1526901T2D-C15	5	delta	5	delta	0.98246187	True
HP1526901T2D-C16	3	alpha	3	alpha	0.99772686	True
HP1526901T2D-C17	4	beta	4	beta	0.9980621	True
HP1526901T2D-C19	0	PP	0	PP	0.76993483	True
HP1526901T2D-C2	3	alpha	3	alpha	0.99761057	True
HP1526901T2D-C20	6	ductal	6	ductal	0.99484384	True
HP1526901T2D-C22	6	ductal	6	ductal	0.99437875	True
HP1526901T2D-C23	6	ductal	6	ductal	0.994124	True
HP1526901T2D-C24	5	delta	5	delta	0.9829534	True
HP1526901T2D-C3	4	beta	4	beta	0.9981584	True
HP1526901T2D-C4	3	alpha	3	alpha	0.9976413	True
HP1526901T2D-C6	3	alpha	3	alpha	0.99764997	True
HP1526901T2D-C8	2	acinar	2	acinar	0.99213225	True
HP1526901T2D-D1	3	alpha	3	alpha	0.9977774	True
HP1526901T2D-D10	0	PP	3	alpha	0.8838557	True
HP1526901T2D-D12	6	ductal	6	ductal	0.99375135	True
HP1526901T2D-D14	0	PP	3	alpha	0.76680577	True
HP1526901T2D-D16	3	alpha	3	alpha	0.9976178	True
HP1526901T2D-D17	4	beta	4	beta	0.9981324	True
HP1526901T2D-D18	3	alpha	3	alpha	0.9977647	True
HP1526901T2D-D19	4	beta	4	beta	0.9978108	True
HP1526901T2D-D2	6	ductal	6	ductal	0.9939334	True
HP1526901T2D-D21	5	delta	5	delta	0.9481053	True
HP1526901T2D-D22	6	ductal	6	ductal	0.993518	True
HP1526901T2D-D24	5	delta	5	delta	0.812115	True
HP1526901T2D-D3	6	ductal	6	ductal	0.9939468	True
HP1526901T2D-D4	6	ductal	6	ductal	0.99443096	True
HP1526901T2D-D5	3	alpha	3	alpha	0.9977289	True
HP1526901T2D-D7	2	acinar	2	acinar	0.7744915	True
HP1526901T2D-D8	3	alpha	3	alpha	0.9977101	True
HP1526901T2D-D9	3	alpha	3	alpha	0.9976342	True
HP1526901T2D-E1	6	ductal	6	ductal	0.99287534	True
HP1526901T2D-E10	0	PP	3	alpha	0.85887164	True
HP1526901T2D-E11	4	beta	4	beta	0.99797934	True
HP1526901T2D-E14	6	ductal	6	ductal	0.9918475	True
HP1526901T2D-E15	1	PSC	1	PSC	0.99344856	True
HP1526901T2D-E16	2	acinar	6	ductal	0.81647325	True
HP1526901T2D-E18	3	alpha	3	alpha	0.9976578	True
HP1526901T2D-E19	5	delta	5	delta	0.98335105	True
HP1526901T2D-E2	4	beta	4	beta	0.9978897	True
HP1526901T2D-E20	5	delta	5	delta	0.8947517	True
HP1526901T2D-E21	5	delta	5	delta	0.96939564	True
HP1526901T2D-E22	6	ductal	6	ductal	0.99369955	True
HP1526901T2D-E23	6	ductal	6	ductal	0.99353975	True
HP1526901T2D-E3	5	delta	5	delta	0.9820857	True
HP1526901T2D-E4	0	PP	3	alpha	0.86555064	True
HP1526901T2D-E5	0	PP	0	PP	0.81918186	True
HP1526901T2D-E6	3	alpha	3	alpha	0.99763036	True
HP1526901T2D-E8	3	alpha	3	alpha	0.99777466	True
HP1526901T2D-E9	6	ductal	6	ductal	0.9939016	True
HP1526901T2D-F1	3	alpha	3	alpha	0.99771607	True
HP1526901T2D-F10	3	alpha	3	alpha	0.9977071	True
HP1526901T2D-F12	0	PP	3	alpha	0.87552774	True
HP1526901T2D-F15	6	ductal	6	ductal	0.9935976	True
HP1526901T2D-F16	6	ductal	6	ductal	0.99420124	True
HP1526901T2D-F17	4	beta	4	beta	0.9981086	True
HP1526901T2D-F18	4	beta	4	beta	0.9980665	True
HP1526901T2D-F19	3	alpha	3	alpha	0.9976356	True
HP1526901T2D-F2	3	alpha	3	alpha	0.9977526	True
HP1526901T2D-F20	3	alpha	3	alpha	0.9976301	True
HP1526901T2D-F21	3	alpha	3	alpha	0.99767226	True
HP1526901T2D-F22	6	ductal	6	ductal	0.9913306	True
HP1526901T2D-F23	5	delta	5	delta	0.97469044	True
HP1526901T2D-F24	6	ductal	6	ductal	0.9938252	True
HP1526901T2D-F3	3	alpha	3	alpha	0.997773	True
HP1526901T2D-F5	4	beta	4	beta	0.99813807	True
HP1526901T2D-F7	0	PP	3	alpha	0.5500831	True
HP1526901T2D-F8	6	ductal	6	ductal	0.9922496	True
HP1526901T2D-F9	3	alpha	3	alpha	0.99773705	True
HP1526901T2D-G1	0	PP	5	delta	0.569812	True
HP1526901T2D-G10	3	alpha	3	alpha	0.99764353	True
HP1526901T2D-G11	-1	<unknown_or_unlabeled>	9	macrophage	0.8088644	False
HP1526901T2D-G13	3	alpha	3	alpha	0.9971392	True
HP1526901T2D-G14	3	alpha	3	alpha	0.9977315	True
HP1526901T2D-G16	3	alpha	3	alpha	0.99775463	True
HP1526901T2D-G17	4	beta	4	beta	0.9981128	True
HP1526901T2D-G18	3	alpha	3	alpha	0.99764746	True
HP1526901T2D-G19	0	PP	0	PP	0.67809033	True
HP1526901T2D-G2	5	delta	5	delta	0.9831156	True
HP1526901T2D-G21	4	beta	4	beta	0.99816376	True
HP1526901T2D-G22	6	ductal	6	ductal	0.9932058	True
HP1526901T2D-G23	5	delta	5	delta	0.95069355	True
HP1526901T2D-G24	6	ductal	6	ductal	0.9940849	True
HP1526901T2D-G4	4	beta	4	beta	0.9974929	True
HP1526901T2D-G7	3	alpha	3	alpha	0.99770445	True
HP1526901T2D-G8	3	alpha	3	alpha	0.99776876	True
HP1526901T2D-G9	0	PP	0	PP	0.5279238	True
HP1526901T2D-H10	4	beta	4	beta	0.99794203	True
HP1526901T2D-H12	3	alpha	3	alpha	0.9977452	True
HP1526901T2D-H16	0	PP	3	alpha	0.55721515	True
HP1526901T2D-H17	3	alpha	3	alpha	0.9975345	True
HP1526901T2D-H18	5	delta	5	delta	0.6472519	True
HP1526901T2D-H19	3	alpha	3	alpha	0.997712	True
HP1526901T2D-H2	2	acinar	6	ductal	0.93288743	True
HP1526901T2D-H20	4	beta	4	beta	0.9977609	True
HP1526901T2D-H21	6	ductal	6	ductal	0.9929516	True
HP1526901T2D-H22	4	beta	4	beta	0.9981103	True
HP1526901T2D-H23	0	PP	5	delta	0.6717264	True
HP1526901T2D-H24	4	beta	4	beta	0.9980301	True
HP1526901T2D-H3	0	PP	3	alpha	0.9741019	True
HP1526901T2D-H4	4	beta	4	beta	0.998083	True
HP1526901T2D-H5	6	ductal	6	ductal	0.99385035	True
HP1526901T2D-H6	3	alpha	3	alpha	0.997767	True
HP1526901T2D-H8	6	ductal	6	ductal	0.9921293	True
HP1526901T2D-H9	6	ductal	6	ductal	0.9933548	True
HP1526901T2D-I1	6	ductal	6	ductal	0.9947548	True
HP1526901T2D-I10	5	delta	5	delta	0.9704194	True
HP1526901T2D-I11	5	delta	5	delta	0.96057135	True
HP1526901T2D-I12	2	acinar	6	ductal	0.9761267	True
HP1526901T2D-I13	4	beta	4	beta	0.9979851	True
HP1526901T2D-I14	3	alpha	3	alpha	0.99774003	True
HP1526901T2D-I15	3	alpha	3	alpha	0.99768865	True
HP1526901T2D-I16	3	alpha	3	alpha	0.99770135	True
HP1526901T2D-I17	2	acinar	6	ductal	0.9864521	True
HP1526901T2D-I18	5	delta	5	delta	0.9795675	True
HP1526901T2D-I19	5	delta	5	delta	0.9539952	True
HP1526901T2D-I20	4	beta	4	beta	0.99793553	True
HP1526901T2D-I21	3	alpha	3	alpha	0.9977137	True
HP1526901T2D-I22	4	beta	4	beta	0.99816364	True
HP1526901T2D-I24	0	PP	3	alpha	0.7912836	True
HP1526901T2D-I3	4	beta	4	beta	0.9979013	True
HP1526901T2D-I4	3	alpha	3	alpha	0.9976736	True
HP1526901T2D-I6	3	alpha	3	alpha	0.99761873	True
HP1526901T2D-I7	3	alpha	3	alpha	0.9976568	True
HP1526901T2D-I8	3	alpha	3	alpha	0.9976088	True
HP1526901T2D-I9	1	PSC	1	PSC	0.99420047	True
HP1526901T2D-J1	5	delta	5	delta	0.93865764	True
HP1526901T2D-J11	6	ductal	6	ductal	0.9923307	True
HP1526901T2D-J12	6	ductal	6	ductal	0.9939767	True
HP1526901T2D-J13	3	alpha	3	alpha	0.997729	True
HP1526901T2D-J14	3	alpha	3	alpha	0.9976901	True
HP1526901T2D-J17	0	PP	3	alpha	0.9603544	True
HP1526901T2D-J18	6	ductal	6	ductal	0.9941174	True
HP1526901T2D-J19	3	alpha	3	alpha	0.9976125	True
HP1526901T2D-J2	3	alpha	3	alpha	0.9975969	True
HP1526901T2D-J20	5	delta	5	delta	0.9719516	True
HP1526901T2D-J21	4	beta	4	beta	0.9979824	True
HP1526901T2D-J22	5	delta	5	delta	0.9737492	True
HP1526901T2D-J24	3	alpha	3	alpha	0.9977592	True
HP1526901T2D-J3	6	ductal	6	ductal	0.99053484	True
HP1526901T2D-J4	3	alpha	3	alpha	0.99764687	True
HP1526901T2D-J5	4	beta	4	beta	0.9980622	True
HP1526901T2D-J6	4	beta	4	beta	0.9981298	True
HP1526901T2D-J8	3	alpha	3	alpha	0.997698	True
HP1526901T2D-K1	6	ductal	6	ductal	0.9935446	True
HP1526901T2D-K10	3	alpha	3	alpha	0.99771285	True
HP1526901T2D-K11	0	PP	3	alpha	0.50331694	True
HP1526901T2D-K12	0	PP	3	alpha	0.95719784	True
HP1526901T2D-K14	3	alpha	3	alpha	0.9977514	True
HP1526901T2D-K15	3	alpha	3	alpha	0.9977354	True
HP1526901T2D-K17	2	acinar	2	acinar	0.9340239	True
HP1526901T2D-K18	3	alpha	3	alpha	0.9977208	True
HP1526901T2D-K19	6	ductal	6	ductal	0.9942163	True
HP1526901T2D-K2	3	alpha	3	alpha	0.99767035	True
HP1526901T2D-K20	4	beta	4	beta	0.9979102	True
HP1526901T2D-K21	0	PP	3	alpha	0.9841578	True
HP1526901T2D-K22	5	delta	5	delta	0.9546316	True
HP1526901T2D-K23	4	beta	4	beta	0.9980804	True
HP1526901T2D-K24	5	delta	5	delta	0.92593807	True
HP1526901T2D-K3	4	beta	4	beta	0.99808794	True
HP1526901T2D-K4	6	ductal	6	ductal	0.9933561	True
HP1526901T2D-K6	3	alpha	3	alpha	0.99777955	True
HP1526901T2D-K7	4	beta	4	beta	0.9979292	True
HP1526901T2D-L1	4	beta	4	beta	0.99799937	True
HP1526901T2D-L11	3	alpha	3	alpha	0.9977629	True
HP1526901T2D-L12	0	PP	3	alpha	0.6172862	True
HP1526901T2D-L13	3	alpha	3	alpha	0.9977035	True
HP1526901T2D-L14	0	PP	0	PP	0.566645	True
HP1526901T2D-L15	2	acinar	6	ductal	0.9560592	True
HP1526901T2D-L16	3	alpha	3	alpha	0.9977068	True
HP1526901T2D-L17	3	alpha	3	alpha	0.9977514	True
HP1526901T2D-L19	3	alpha	3	alpha	0.99767214	True
HP1526901T2D-L2	3	alpha	3	alpha	0.9976042	True
HP1526901T2D-L21	3	alpha	3	alpha	0.9977616	True
HP1526901T2D-L23	3	alpha	3	alpha	0.9977151	True
HP1526901T2D-L24	5	delta	5	delta	0.9795004	True
HP1526901T2D-L4	4	beta	4	beta	0.99809724	True
HP1526901T2D-L5	4	beta	4	beta	0.9938465	True
HP1526901T2D-L6	4	beta	4	beta	0.99793005	True
HP1526901T2D-L7	0	PP	3	alpha	0.77965796	True
HP1526901T2D-M1	3	alpha	3	alpha	0.99771667	True
HP1526901T2D-M13	8	epsilon	3	alpha	0.9917698	True
HP1526901T2D-M14	6	ductal	6	ductal	0.9912281	True
HP1526901T2D-M15	4	beta	4	beta	0.9979913	True
HP1526901T2D-M16	3	alpha	3	alpha	0.99765223	True
HP1526901T2D-M18	2	acinar	2	acinar	0.9942931	True
HP1526901T2D-M19	5	delta	5	delta	0.9822682	True
HP1526901T2D-M2	4	beta	4	beta	0.9980823	True
HP1526901T2D-M20	6	ductal	6	ductal	0.9929039	True
HP1526901T2D-M21	3	alpha	3	alpha	0.9976891	True
HP1526901T2D-M22	0	PP	3	alpha	0.7997104	True
HP1526901T2D-M23	3	alpha	3	alpha	0.99776196	True
HP1526901T2D-M3	0	PP	3	alpha	0.7552003	True
HP1526901T2D-M4	3	alpha	3	alpha	0.99765897	True
HP1526901T2D-M7	3	alpha	3	alpha	0.99773765	True
HP1526901T2D-M8	6	ductal	6	ductal	0.99430907	True
HP1526901T2D-N1	4	beta	4	beta	0.9979578	True
HP1526901T2D-N10	3	alpha	3	alpha	0.99777466	True
HP1526901T2D-N11	6	ductal	6	ductal	0.9944431	True
HP1526901T2D-N12	4	beta	4	beta	0.99817944	True
HP1526901T2D-N13	4	beta	4	beta	0.9979213	True
HP1526901T2D-N14	3	alpha	3	alpha	0.9976598	True
HP1526901T2D-N15	3	alpha	3	alpha	0.9977525	True
HP1526901T2D-N16	5	delta	5	delta	0.90898293	True
HP1526901T2D-N18	0	PP	3	alpha	0.98859763	True
HP1526901T2D-N19	0	PP	3	alpha	0.75619763	True
HP1526901T2D-N2	3	alpha	3	alpha	0.99752194	True
HP1526901T2D-N20	0	PP	3	alpha	0.7132102	True
HP1526901T2D-N21	6	ductal	6	ductal	0.9931375	True
HP1526901T2D-N23	4	beta	4	beta	0.99808997	True
HP1526901T2D-N3	4	beta	4	beta	0.9981047	True
HP1526901T2D-N4	2	acinar	2	acinar	0.99427575	True
HP1526901T2D-N6	3	alpha	3	alpha	0.99769115	True
HP1526901T2D-N7	2	acinar	2	acinar	0.8707717	True
HP1526901T2D-N8	3	alpha	3	alpha	0.99772805	True
HP1526901T2D-N9	3	alpha	3	alpha	0.99771416	True
HP1526901T2D-O1	4	beta	4	beta	0.9979746	True
HP1526901T2D-O10	0	PP	3	alpha	0.9950427	True
HP1526901T2D-O12	10	mast	9	macrophage	0.29885665	True
HP1526901T2D-O13	4	beta	4	beta	0.99808806	True
HP1526901T2D-O14	4	beta	4	beta	0.9980247	True
HP1526901T2D-O15	4	beta	4	beta	0.9980154	True
HP1526901T2D-O17	4	beta	4	beta	0.99808973	True
HP1526901T2D-O19	2	acinar	6	ductal	0.97523355	True
HP1526901T2D-O2	1	PSC	1	PSC	0.99404496	True
HP1526901T2D-O21	3	alpha	3	alpha	0.997775	True
HP1526901T2D-O22	3	alpha	3	alpha	0.997766	True
HP1526901T2D-O23	3	alpha	3	alpha	0.99770606	True
HP1526901T2D-O24	4	beta	4	beta	0.9980325	True
HP1526901T2D-O3	0	PP	3	alpha	0.50495416	True
HP1526901T2D-O4	6	ductal	6	ductal	0.99422497	True
HP1526901T2D-O5	6	ductal	6	ductal	0.993881	True
HP1526901T2D-O6	3	alpha	3	alpha	0.99764997	True
HP1526901T2D-O7	5	delta	5	delta	0.9649401	True
HP1526901T2D-O8	4	beta	4	beta	0.9980253	True
HP1526901T2D-O9	5	delta	5	delta	0.9741504	True
HP1526901T2D-P1	3	alpha	3	alpha	0.9977877	True
HP1526901T2D-P10	4	beta	4	beta	0.9981798	True
HP1526901T2D-P11	4	beta	4	beta	0.9981154	True
HP1526901T2D-P12	3	alpha	3	alpha	0.9977596	True
HP1526901T2D-P13	1	PSC	1	PSC	0.99245524	True
HP1526901T2D-P14	3	alpha	3	alpha	0.9977424	True
HP1526901T2D-P17	4	beta	4	beta	0.9981	True
HP1526901T2D-P18	4	beta	4	beta	0.9979436	True
HP1526901T2D-P19	3	alpha	3	alpha	0.99749666	True
HP1526901T2D-P2	3	alpha	3	alpha	0.9976853	True
HP1526901T2D-P20	3	alpha	3	alpha	0.9977937	True
HP1526901T2D-P21	3	alpha	3	alpha	0.99770576	True
HP1526901T2D-P22	5	delta	5	delta	0.9410834	True
HP1526901T2D-P23	0	PP	0	PP	0.50546575	True
HP1526901T2D-P4	4	beta	4	beta	0.9979603	True
HP1526901T2D-P7	4	beta	4	beta	0.9980033	True
HP1526901T2D-P9	4	beta	4	beta	0.9979752	True
Sample_1	4	beta	4	beta	0.9979049	True
Sample_2	4	beta	4	beta	0.9979273	True
Sample_3	4	beta	4	beta	0.9980463	True
Sample_4	4	beta	4	beta	0.9979791	True
Sample_5	4	beta	4	beta	0.997945	True
Sample_6	4	beta	4	beta	0.9980478	True
Sample_7	4	beta	4	beta	0.99793833	True
Sample_8	4	beta	4	beta	0.9978393	True
Sample_9	4	beta	4	beta	0.9980812	True
Sample_10	4	beta	4	beta	0.9979978	True
Sample_11	4	beta	4	beta	0.9980611	True
Sample_12	4	beta	4	beta	0.9980806	True
Sample_13	4	beta	4	beta	0.9980836	True
Sample_14	4	beta	4	beta	0.99816424	True
Sample_15	4	beta	4	beta	0.99805874	True
Sample_16	4	beta	4	beta	0.9978005	True
Sample_17	4	beta	4	beta	0.99734145	True
Sample_18	4	beta	4	beta	0.99714166	True
Sample_19	4	beta	4	beta	0.9980484	True
Sample_20	4	beta	4	beta	0.99797136	True
Sample_21	4	beta	4	beta	0.99763787	True
Sample_22	4	beta	4	beta	0.99808145	True
Sample_23	4	beta	4	beta	0.9981384	True
Sample_24	4	beta	4	beta	0.997954	True
Sample_25	4	beta	4	beta	0.9980192	True
Sample_26	4	beta	4	beta	0.99785197	True
Sample_27	4	beta	4	beta	0.9979607	True
Sample_28	4	beta	4	beta	0.9980849	True
Sample_29	4	beta	4	beta	0.9981053	True
Sample_30	4	beta	4	beta	0.9965642	True
Sample_32	4	beta	4	beta	0.99816436	True
Sample_33	4	beta	4	beta	0.9979673	True
Sample_34	4	beta	4	beta	0.9981908	True
Sample_35	4	beta	4	beta	0.9982674	True
Sample_36	4	beta	4	beta	0.99806887	True
Sample_37	4	beta	4	beta	0.9980117	True
Sample_38	4	beta	4	beta	0.99816483	True
Sample_39	4	beta	4	beta	0.9980654	True
Sample_40	4	beta	4	beta	0.9980336	True
Sample_41	4	beta	4	beta	0.99806255	True
Sample_42	4	beta	4	beta	0.9980982	True
Sample_44	4	beta	4	beta	0.99804664	True
Sample_45	4	beta	4	beta	0.9980489	True
Sample_46	4	beta	4	beta	0.9979335	True
Sample_47	4	beta	4	beta	0.9159143	True
Sample_48	4	beta	4	beta	0.9979189	True
Sample_49	4	beta	4	beta	0.9980307	True
Sample_50	4	beta	4	beta	0.998102	True
Sample_51	4	beta	4	beta	0.99801683	True
Sample_52	4	beta	4	beta	0.9981456	True
Sample_53	4	beta	4	beta	0.99767345	True
Sample_54	4	beta	4	beta	0.9980786	True
Sample_55	4	beta	4	beta	0.997886	True
Sample_56	4	beta	4	beta	0.99802965	True
Sample_57	4	beta	4	beta	0.99782026	True
Sample_58	4	beta	4	beta	0.99807525	True
Sample_59	4	beta	4	beta	0.99783546	True
Sample_60	4	beta	4	beta	0.9980507	True
Sample_61	4	beta	4	beta	0.998085	True
Sample_62	4	beta	4	beta	0.9977464	True
Sample_63	4	beta	4	beta	0.9978896	True
Sample_64	4	beta	4	beta	0.99800366	True
Sample_65	4	beta	4	beta	0.9980275	True
Sample_66	4	beta	4	beta	0.99806505	True
Sample_67	4	beta	4	beta	0.99788564	True
Sample_68	4	beta	4	beta	0.99806947	True
Sample_69	4	beta	4	beta	0.9980386	True
Sample_71	4	beta	4	beta	0.9980805	True
Sample_72	4	beta	4	beta	0.99819654	True
Sample_74	4	beta	4	beta	0.9979639	True
Sample_75	4	beta	4	beta	0.99790776	True
Sample_76	4	beta	4	beta	0.9979663	True
Sample_77	4	beta	4	beta	0.9978289	True
Sample_78	4	beta	4	beta	0.9980453	True
Sample_79	4	beta	4	beta	0.9963888	True
Sample_80	4	beta	4	beta	0.99778014	True
Sample_81	4	beta	4	beta	0.9980274	True
Sample_82	4	beta	4	beta	0.9978636	True
Sample_83	4	beta	4	beta	0.9981451	True
Sample_84	4	beta	4	beta	0.99740475	True
Sample_85	4	beta	4	beta	0.99796945	True
Sample_86	4	beta	4	beta	0.99798834	True
Sample_87	4	beta	4	beta	0.99809545	True
Sample_88	4	beta	4	beta	0.99816495	True
Sample_89	4	beta	4	beta	0.9981263	True
Sample_90	4	beta	4	beta	0.997624	True
Sample_91	4	beta	4	beta	0.99777454	True
Sample_92	4	beta	4	beta	0.9978363	True
Sample_93	4	beta	4	beta	0.99803466	True
Sample_95	4	beta	4	beta	0.99799585	True
Sample_96	4	beta	4	beta	0.99803907	True
Sample_97	4	beta	4	beta	0.9979754	True
Sample_98	4	beta	4	beta	0.99811894	True
Sample_99	4	beta	4	beta	0.99819154	True
Sample_100	4	beta	4	beta	0.9980903	True
Sample_101	4	beta	4	beta	0.9980428	True
Sample_102	4	beta	4	beta	0.9980388	True
Sample_103	4	beta	4	beta	0.99814856	True
Sample_104	4	beta	4	beta	0.998131	True
Sample_105	4	beta	4	beta	0.9978508	True
Sample_106	4	beta	4	beta	0.9979572	True
Sample_107	4	beta	4	beta	0.9980465	True
Sample_108	4	beta	4	beta	0.9976751	True
Sample_109	4	beta	4	beta	0.99800056	True
Sample_110	4	beta	4	beta	0.99789184	True
Sample_111	4	beta	4	beta	0.99796885	True
Sample_112	4	beta	4	beta	0.99795204	True
Sample_113	4	beta	4	beta	0.9980538	True
Sample_115	4	beta	4	beta	0.99809724	True
Sample_116	4	beta	4	beta	0.9952956	True
Sample_117	4	beta	4	beta	0.99803144	True
Sample_119	4	beta	4	beta	0.9979407	True
Sample_120	4	beta	4	beta	0.9977597	True
Sample_121	4	beta	4	beta	0.9979785	True
Sample_122	4	beta	4	beta	0.9977683	True
Sample_124	4	beta	4	beta	0.9979651	True
Sample_125	4	beta	4	beta	0.99758387	True
Sample_126	4	beta	4	beta	0.99602085	True
Sample_127	4	beta	4	beta	0.9979685	True
Sample_128	4	beta	4	beta	0.99784124	True
Sample_129	4	beta	4	beta	0.99796486	True
Sample_130	4	beta	4	beta	0.9980241	True
Sample_131	4	beta	4	beta	0.9979893	True
Sample_132	4	beta	4	beta	0.99814653	True
Sample_133	4	beta	4	beta	0.9979062	True
Sample_134	4	beta	4	beta	0.9981225	True
Sample_135	4	beta	4	beta	0.99805486	True
Sample_136	4	beta	4	beta	0.99798775	True
Sample_137	4	beta	4	beta	0.9981363	True
Sample_138	4	beta	4	beta	0.99798834	True
Sample_139	4	beta	4	beta	0.99772984	True
Sample_140	4	beta	4	beta	0.99800843	True
Sample_141	4	beta	4	beta	0.99812585	True
Sample_142	4	beta	4	beta	0.9980357	True
Sample_143	4	beta	4	beta	0.99808	True
Sample_145	4	beta	4	beta	0.9979354	True
Sample_147	4	beta	4	beta	0.9981047	True
Sample_148	4	beta	4	beta	0.9978587	True
Sample_149	4	beta	4	beta	0.9981396	True
Sample_150	4	beta	4	beta	0.99130684	True
Sample_151	4	beta	4	beta	0.99791485	True
Sample_152	4	beta	4	beta	0.997931	True
Sample_153	4	beta	4	beta	0.9980546	True
Sample_154	4	beta	4	beta	0.99807227	True
Sample_155	4	beta	4	beta	0.9979931	True
Sample_156	4	beta	4	beta	0.99809796	True
Sample_157	4	beta	4	beta	0.997979	True
Sample_158	4	beta	4	beta	0.9980691	True
Sample_159	4	beta	4	beta	0.9978854	True
Sample_160	4	beta	4	beta	0.99795073	True
Sample_161	4	beta	4	beta	0.9978923	True
Sample_162	4	beta	4	beta	0.9980598	True
Sample_163	4	beta	4	beta	0.9979327	True
Sample_164	4	beta	4	beta	0.9977349	True
Sample_165	4	beta	4	beta	0.9980946	True
Sample_166	4	beta	4	beta	0.9921565	True
Sample_167	4	beta	4	beta	0.99803585	True
Sample_168	4	beta	4	beta	0.9981772	True
Sample_169	4	beta	4	beta	0.99803394	True
Sample_170	4	beta	4	beta	0.99806184	True
Sample_172	4	beta	4	beta	0.9975193	True
Sample_174	4	beta	4	beta	0.9980197	True
Sample_175	4	beta	4	beta	0.9980354	True
Sample_176	4	beta	3	alpha	0.99769115	True
Sample_177	4	beta	4	beta	0.9980185	True
Sample_178	4	beta	4	beta	0.9979673	True
Sample_179	4	beta	4	beta	0.9977608	True
Sample_180	4	beta	4	beta	0.9982204	True
Sample_181	4	beta	4	beta	0.9980623	True
Sample_182	4	beta	4	beta	0.99800664	True
Sample_183	4	beta	4	beta	0.99799585	True
Sample_184	4	beta	4	beta	0.9980228	True
Sample_185	4	beta	4	beta	0.9980698	True
Sample_186	4	beta	4	beta	0.9980013	True
Sample_187	4	beta	4	beta	0.99784577	True
Sample_188	4	beta	4	beta	0.99759287	True
Sample_189	4	beta	4	beta	0.9979145	True
Sample_190	4	beta	4	beta	0.9980065	True
Sample_191	4	beta	4	beta	0.9981548	True
Sample_192	4	beta	4	beta	0.9980469	True
Sample_193	4	beta	4	beta	0.99807703	True
Sample_194	4	beta	4	beta	0.9980702	True
Sample_195	4	beta	4	beta	0.99720025	True
Sample_196	4	beta	4	beta	0.998083	True
Sample_197	4	beta	4	beta	0.9980724	True
Sample_198	4	beta	4	beta	0.99810046	True
Sample_199	4	beta	4	beta	0.9980793	True
Sample_200	4	beta	4	beta	0.9980761	True
Sample_201	4	beta	4	beta	0.99803394	True
Sample_202	4	beta	4	beta	0.9982022	True
Sample_203	4	beta	4	beta	0.9982185	True
Sample_205	4	beta	4	beta	0.998054	True
Sample_206	4	beta	4	beta	0.9981279	True
Sample_207	4	beta	4	beta	0.99808705	True
Sample_208	3	alpha	3	alpha	0.99740595	True
Sample_209	3	alpha	3	alpha	0.9975019	True
Sample_211	3	alpha	3	alpha	0.997542	True
Sample_212	3	alpha	3	alpha	0.9976649	True
Sample_213	3	alpha	3	alpha	0.9977301	True
Sample_214	3	alpha	3	alpha	0.9977209	True
Sample_215	3	alpha	3	alpha	0.99770725	True
Sample_216	3	alpha	3	alpha	0.9976731	True
Sample_217	3	alpha	3	alpha	0.9976545	True
Sample_218	3	alpha	3	alpha	0.9975489	True
Sample_219	3	alpha	3	alpha	0.99767715	True
Sample_220	3	alpha	3	alpha	0.9975921	True
Sample_221	3	alpha	3	alpha	0.9977265	True
Sample_222	3	alpha	3	alpha	0.99775034	True
Sample_223	3	alpha	3	alpha	0.99752814	True
Sample_224	3	alpha	3	alpha	0.9972179	True
Sample_225	3	alpha	3	alpha	0.9976325	True
Sample_226	3	alpha	3	alpha	0.9976586	True
Sample_227	3	alpha	3	alpha	0.99773103	True
Sample_228	3	alpha	3	alpha	0.997523	True
Sample_229	3	alpha	3	alpha	0.997581	True
Sample_230	3	alpha	3	alpha	0.9976828	True
Sample_231	3	alpha	3	alpha	0.99770063	True
Sample_232	3	alpha	3	alpha	0.9976368	True
Sample_234	3	alpha	3	alpha	0.9977047	True
Sample_235	3	alpha	3	alpha	0.9976526	True
Sample_236	3	alpha	3	alpha	0.9975172	True
Sample_237	3	alpha	3	alpha	0.9976888	True
Sample_238	3	alpha	3	alpha	0.99769455	True
Sample_239	3	alpha	3	alpha	0.99768615	True
Sample_240	3	alpha	3	alpha	0.9974107	True
Sample_241	3	alpha	3	alpha	0.99765515	True
Sample_242	3	alpha	3	alpha	0.9975883	True
Sample_243	3	alpha	3	alpha	0.99771523	True
Sample_244	3	alpha	3	alpha	0.9975649	True
Sample_245	3	alpha	3	alpha	0.9974802	True
Sample_246	3	alpha	3	alpha	0.9976151	True
Sample_247	3	alpha	3	alpha	0.99741805	True
Sample_248	3	alpha	3	alpha	0.99768186	True
Sample_249	3	alpha	3	alpha	0.9977041	True
Sample_250	3	alpha	3	alpha	0.9976726	True
Sample_251	3	alpha	3	alpha	0.99772805	True
Sample_252	3	alpha	3	alpha	0.99739695	True
Sample_253	3	alpha	3	alpha	0.997647	True
Sample_255	3	alpha	3	alpha	0.99749786	True
Sample_256	3	alpha	3	alpha	0.9976497	True
Sample_257	3	alpha	3	alpha	0.997074	True
Sample_258	3	alpha	3	alpha	0.9976922	True
Sample_259	3	alpha	3	alpha	0.99772304	True
Sample_260	3	alpha	3	alpha	0.99763584	True
Sample_261	3	alpha	3	alpha	0.99780077	True
Sample_262	3	alpha	3	alpha	0.9977047	True
Sample_263	3	alpha	3	alpha	0.9970541	True
Sample_264	3	alpha	3	alpha	0.99715936	True
Sample_265	3	alpha	3	alpha	0.9974942	True
Sample_266	3	alpha	3	alpha	0.99771947	True
Sample_267	3	alpha	3	alpha	0.9966276	True
Sample_268	3	alpha	3	alpha	0.99770796	True
Sample_269	3	alpha	3	alpha	0.9977627	True
Sample_270	3	alpha	3	alpha	0.99778724	True
Sample_272	3	alpha	3	alpha	0.9977399	True
Sample_273	3	alpha	3	alpha	0.9976579	True
Sample_274	3	alpha	3	alpha	0.9977174	True
Sample_275	3	alpha	3	alpha	0.9977309	True
Sample_276	3	alpha	3	alpha	0.9975721	True
Sample_277	3	alpha	3	alpha	0.99775946	True
Sample_279	3	alpha	3	alpha	0.99766773	True
Sample_280	3	alpha	3	alpha	0.9977316	True
Sample_281	3	alpha	3	alpha	0.9976445	True
Sample_282	3	alpha	3	alpha	0.99751365	True
Sample_283	3	alpha	3	alpha	0.9977533	True
Sample_284	3	alpha	3	alpha	0.99747723	True
Sample_285	3	alpha	3	alpha	0.9976254	True
Sample_286	3	alpha	3	alpha	0.99774003	True
Sample_287	3	alpha	3	alpha	0.99777055	True
Sample_290	3	alpha	3	alpha	0.9976859	True
Sample_291	3	alpha	3	alpha	0.99773645	True
Sample_292	3	alpha	3	alpha	0.99773943	True
Sample_293	3	alpha	3	alpha	0.99761355	True
Sample_294	3	alpha	3	alpha	0.9975556	True
Sample_295	3	alpha	3	alpha	0.997701	True
Sample_296	3	alpha	3	alpha	0.99769956	True
Sample_297	3	alpha	3	alpha	0.9974935	True
Sample_298	3	alpha	3	alpha	0.99748504	True
Sample_299	3	alpha	3	alpha	0.9977203	True
Sample_300	3	alpha	3	alpha	0.9977895	True
Sample_301	3	alpha	3	alpha	0.99776495	True
Sample_302	3	alpha	3	alpha	0.9976841	True
Sample_303	3	alpha	3	alpha	0.99772435	True
Sample_304	3	alpha	3	alpha	0.99757093	True
Sample_305	3	alpha	3	alpha	0.99773127	True
Sample_306	3	alpha	3	alpha	0.9977164	True
Sample_307	3	alpha	3	alpha	0.9978114	True
Sample_308	3	alpha	3	alpha	0.9977124	True
Sample_309	3	alpha	3	alpha	0.9977925	True
Sample_310	3	alpha	3	alpha	0.99777514	True
Sample_311	3	alpha	3	alpha	0.997761	True
Sample_312	3	alpha	3	alpha	0.99769527	True
Sample_313	3	alpha	3	alpha	0.9977563	True
Sample_314	3	alpha	3	alpha	0.99772865	True
Sample_315	3	alpha	3	alpha	0.9976705	True
Sample_316	3	alpha	3	alpha	0.99779785	True
Sample_317	3	alpha	3	alpha	0.9977914	True
Sample_318	3	alpha	3	alpha	0.9976301	True
Sample_319	3	alpha	3	alpha	0.9977604	True
Sample_320	3	alpha	3	alpha	0.99761826	True
Sample_321	3	alpha	3	alpha	0.997773	True
Sample_322	3	alpha	3	alpha	0.9976578	True
Sample_323	3	alpha	3	alpha	0.9976356	True
Sample_324	3	alpha	3	alpha	0.9976799	True
Sample_325	3	alpha	3	alpha	0.99774545	True
Sample_326	3	alpha	3	alpha	0.9975702	True
Sample_327	3	alpha	3	alpha	0.997738	True
Sample_328	3	alpha	3	alpha	0.99772626	True
Sample_329	3	alpha	3	alpha	0.997726	True
Sample_330	3	alpha	3	alpha	0.9977575	True
Sample_332	3	alpha	3	alpha	0.9976579	True
Sample_333	3	alpha	3	alpha	0.9976788	True
Sample_334	3	alpha	3	alpha	0.9978179	True
Sample_335	3	alpha	3	alpha	0.9977835	True
Sample_336	3	alpha	3	alpha	0.9977964	True
Sample_337	3	alpha	3	alpha	0.9977623	True
Sample_338	3	alpha	3	alpha	0.9977556	True
Sample_340	3	alpha	3	alpha	0.99778855	True
Sample_341	3	alpha	3	alpha	0.9977202	True
Sample_342	3	alpha	3	alpha	0.99766266	True
Sample_343	3	alpha	3	alpha	0.9977337	True
Sample_344	3	alpha	3	alpha	0.9977017	True
Sample_345	3	alpha	3	alpha	0.9977297	True
Sample_346	3	alpha	3	alpha	0.99772245	True
Sample_347	3	alpha	3	alpha	0.99776566	True
Sample_348	3	alpha	3	alpha	0.9977132	True
Sample_349	3	alpha	3	alpha	0.99770397	True
Sample_350	3	alpha	3	alpha	0.99773127	True
Sample_351	3	alpha	3	alpha	0.9977241	True
Sample_352	3	alpha	3	alpha	0.9977399	True
Sample_353	3	alpha	3	alpha	0.9976235	True
Sample_354	3	alpha	3	alpha	0.99774784	True
Sample_355	3	alpha	3	alpha	0.9977683	True
Sample_356	3	alpha	3	alpha	0.9977616	True
Sample_357	3	alpha	3	alpha	0.9977609	True
Sample_358	3	alpha	3	alpha	0.9976661	True
Sample_359	3	alpha	3	alpha	0.98804665	True
Sample_360	3	alpha	3	alpha	0.99761814	True
Sample_361	3	alpha	3	alpha	0.9977436	True
Sample_362	3	alpha	3	alpha	0.99774194	True
Sample_363	3	alpha	3	alpha	0.99773216	True
Sample_364	3	alpha	3	alpha	0.9977584	True
Sample_365	3	alpha	3	alpha	0.9977596	True
Sample_366	3	alpha	3	alpha	0.9973839	True
Sample_367	3	alpha	3	alpha	0.9977406	True
Sample_368	3	alpha	3	alpha	0.9977951	True
Sample_369	3	alpha	3	alpha	0.99774843	True
Sample_370	3	alpha	3	alpha	0.9973443	True
Sample_372	3	alpha	3	alpha	0.99771994	True
Sample_373	3	alpha	3	alpha	0.99750584	True
Sample_374	3	alpha	3	alpha	0.99771476	True
Sample_375	3	alpha	3	alpha	0.99767584	True
Sample_376	3	alpha	3	alpha	0.9971431	True
Sample_377	3	alpha	3	alpha	0.9976985	True
Sample_378	3	alpha	3	alpha	0.9977405	True
Sample_379	3	alpha	3	alpha	0.99774444	True
Sample_380	3	alpha	3	alpha	0.9977568	True
Sample_381	3	alpha	3	alpha	0.9977404	True
Sample_382	3	alpha	3	alpha	0.99775237	True
Sample_383	3	alpha	3	alpha	0.9977465	True
Sample_384	3	alpha	3	alpha	0.99769104	True
Sample_385	3	alpha	3	alpha	0.9977216	True
Sample_386	3	alpha	3	alpha	0.99775034	True
Sample_387	3	alpha	3	alpha	0.9978011	True
Sample_388	3	alpha	3	alpha	0.99774194	True
Sample_389	3	alpha	3	alpha	0.9977343	True
Sample_390	3	alpha	3	alpha	0.99770606	True
Sample_391	3	alpha	3	alpha	0.9976689	True
Sample_392	3	alpha	3	alpha	0.9977639	True
Sample_393	3	alpha	3	alpha	0.99773705	True
Sample_394	3	alpha	3	alpha	0.9976132	True
Sample_395	3	alpha	3	alpha	0.9977397	True
Sample_396	3	alpha	3	alpha	0.9977609	True
Sample_397	3	alpha	3	alpha	0.9977336	True
Sample_398	3	alpha	3	alpha	0.99776196	True
Sample_399	3	alpha	3	alpha	0.99749845	True
Sample_400	3	alpha	3	alpha	0.99777126	True
Sample_401	3	alpha	3	alpha	0.9977755	True
Sample_402	3	alpha	3	alpha	0.99774545	True
Sample_403	3	alpha	3	alpha	0.9977144	True
Sample_404	3	alpha	3	alpha	0.9977551	True
Sample_405	3	alpha	3	alpha	0.99771166	True
Sample_406	3	alpha	3	alpha	0.9977629	True
Sample_407	3	alpha	3	alpha	0.9977119	True
Sample_408	3	alpha	3	alpha	0.99770796	True
Sample_409	3	alpha	3	alpha	0.99769115	True
Sample_410	3	alpha	3	alpha	0.99777055	True
Sample_411	3	alpha	3	alpha	0.9977755	True
Sample_412	3	alpha	3	alpha	0.99776196	True
Sample_413	3	alpha	3	alpha	0.99759454	True
Sample_414	3	alpha	3	alpha	0.997366	True
Sample_416	3	alpha	3	alpha	0.99772555	True
Sample_417	3	alpha	3	alpha	0.9976858	True
Sample_418	3	alpha	3	alpha	0.9977487	True
Sample_419	3	alpha	3	alpha	0.9978066	True
Sample_421	3	alpha	3	alpha	0.9977366	True
Sample_422	3	alpha	3	alpha	0.9966126	True
Sample_423	3	alpha	3	alpha	0.99763834	True
Sample_424	3	alpha	3	alpha	0.9977387	True
Sample_425	3	alpha	3	alpha	0.99774027	True
Sample_426	3	alpha	3	alpha	0.9977495	True
Sample_427	3	alpha	3	alpha	0.99773407	True
Sample_428	3	alpha	3	alpha	0.9976966	True
Sample_429	3	alpha	3	alpha	0.9968335	True
Sample_430	3	alpha	3	alpha	0.99774665	True
Sample_431	3	alpha	3	alpha	0.9977768	True
Sample_432	3	alpha	3	alpha	0.9977203	True
Sample_433	3	alpha	3	alpha	0.9977399	True
Sample_434	3	alpha	3	alpha	0.9978021	True
Sample_435	3	alpha	3	alpha	0.99767464	True
Sample_438	3	alpha	3	alpha	0.9977246	True
Sample_439	3	alpha	3	alpha	0.9976642	True
Sample_440	3	alpha	3	alpha	0.99778974	True
Sample_441	3	alpha	3	alpha	0.997738	True
Sample_442	3	alpha	3	alpha	0.9976891	True
Sample_443	3	alpha	3	alpha	0.9974796	True
Sample_444	3	alpha	3	alpha	0.99777067	True
Sample_445	3	alpha	3	alpha	0.9976913	True
Sample_446	3	alpha	3	alpha	0.9977812	True
Sample_447	3	alpha	3	alpha	0.9977075	True
Sample_448	3	alpha	3	alpha	0.9976864	True
Sample_449	3	alpha	3	alpha	0.99775034	True
Sample_450	3	alpha	3	alpha	0.99774206	True
Sample_451	3	alpha	3	alpha	0.99778914	True
Sample_452	3	alpha	3	alpha	0.9976762	True
Sample_453	3	alpha	3	alpha	0.9977603	True
Sample_454	3	alpha	3	alpha	0.99779046	True
Sample_455	3	alpha	3	alpha	0.9977429	True
Sample_456	3	alpha	3	alpha	0.99764365	True
Sample_457	3	alpha	3	alpha	0.9976567	True
Sample_458	3	alpha	3	alpha	0.99768245	True
Sample_459	3	alpha	3	alpha	0.9977196	True
Sample_460	3	alpha	3	alpha	0.99773514	True
Sample_461	3	alpha	3	alpha	0.99772257	True
Sample_462	3	alpha	3	alpha	0.9977831	True
Sample_463	3	alpha	3	alpha	0.9976229	True
Sample_464	3	alpha	3	alpha	0.9977507	True
Sample_465	3	alpha	3	alpha	0.9977063	True
Sample_466	3	alpha	3	alpha	0.99766845	True
Sample_467	3	alpha	3	alpha	0.99776757	True
Sample_468	3	alpha	3	alpha	0.99773896	True
Sample_469	3	alpha	3	alpha	0.9977174	True
Sample_470	3	alpha	3	alpha	0.99777514	True
Sample_471	3	alpha	3	alpha	0.997769	True
Sample_472	3	alpha	3	alpha	0.9977691	True
Sample_473	3	alpha	3	alpha	0.99766505	True
Sample_474	3	alpha	3	alpha	0.997735	True
Sample_475	3	alpha	3	alpha	0.997682	True
Sample_476	3	alpha	3	alpha	0.9977774	True
Sample_477	3	alpha	3	alpha	0.9977654	True
Sample_478	3	alpha	3	alpha	0.9977246	True
Sample_479	3	alpha	3	alpha	0.99767286	True
Sample_480	3	alpha	3	alpha	0.9971154	True
Sample_481	3	alpha	3	alpha	0.99777466	True
Sample_482	3	alpha	3	alpha	0.99774796	True
Sample_483	3	alpha	3	alpha	0.99764913	True
Sample_484	3	alpha	3	alpha	0.9976915	True
Sample_485	3	alpha	3	alpha	0.99775153	True
Sample_486	3	alpha	3	alpha	0.9974011	True
Sample_487	3	alpha	3	alpha	0.9976937	True
Sample_489	3	alpha	3	alpha	0.99774337	True
Sample_491	3	alpha	3	alpha	0.99777466	True
Sample_492	3	alpha	3	alpha	0.9977452	True
Sample_493	3	alpha	3	alpha	0.9977089	True
Sample_494	3	alpha	3	alpha	0.99770856	True
Sample_495	3	alpha	3	alpha	0.9976545	True
Sample_496	3	alpha	3	alpha	0.997747	True
Sample_497	3	alpha	3	alpha	0.9977049	True
Sample_498	3	alpha	3	alpha	0.9976164	True
Sample_499	3	alpha	3	alpha	0.9976775	True
Sample_500	3	alpha	3	alpha	0.9976209	True
Sample_501	3	alpha	3	alpha	0.99773407	True
Sample_502	3	alpha	3	alpha	0.9977665	True
Sample_503	3	alpha	3	alpha	0.99775875	True
Sample_504	3	alpha	3	alpha	0.99773115	True
Sample_505	3	alpha	3	alpha	0.99768865	True
Sample_506	3	alpha	3	alpha	0.99774015	True
Sample_507	3	alpha	3	alpha	0.99765164	True
Sample_508	3	alpha	3	alpha	0.99740547	True
Sample_509	3	alpha	3	alpha	0.997693	True
Sample_510	3	alpha	3	alpha	0.9977221	True
Sample_511	3	alpha	3	alpha	0.99761873	True
Sample_513	3	alpha	3	alpha	0.99775034	True
Sample_514	3	alpha	3	alpha	0.99763274	True
Sample_515	3	alpha	3	alpha	0.9977259	True
Sample_516	3	alpha	3	alpha	0.99712545	True
Sample_517	3	alpha	3	alpha	0.99770594	True
Sample_518	3	alpha	3	alpha	0.99768686	True
Sample_519	3	alpha	3	alpha	0.9977309	True
Sample_520	3	alpha	3	alpha	0.9976799	True
Sample_521	3	alpha	3	alpha	0.99772173	True
Sample_522	3	alpha	3	alpha	0.9977556	True
Sample_523	3	alpha	3	alpha	0.99775416	True
Sample_524	3	alpha	3	alpha	0.9977155	True
Sample_525	3	alpha	3	alpha	0.9976688	True
Sample_526	3	alpha	3	alpha	0.9977289	True
Sample_527	3	alpha	3	alpha	0.99765694	True
Sample_528	3	alpha	3	alpha	0.9977125	True
Sample_530	3	alpha	3	alpha	0.9977418	True
Sample_532	3	alpha	3	alpha	0.99764025	True
Sample_533	3	alpha	3	alpha	0.99770576	True
Sample_534	3	alpha	3	alpha	0.9977188	True
Sample_535	3	alpha	3	alpha	0.99758077	True
Sample_536	3	alpha	3	alpha	0.9976162	True
Sample_537	3	alpha	3	alpha	0.99770904	True
Sample_538	3	alpha	3	alpha	0.997727	True
Sample_539	3	alpha	3	alpha	0.99768806	True
Sample_540	3	alpha	3	alpha	0.9977518	True
Sample_541	3	alpha	3	alpha	0.99771214	True
Sample_542	3	alpha	3	alpha	0.9972792	True
Sample_543	3	alpha	3	alpha	0.99764544	True
Sample_544	3	alpha	3	alpha	0.99772984	True
Sample_546	3	alpha	3	alpha	0.99770147	True
Sample_547	3	alpha	3	alpha	0.9976908	True
Sample_548	3	alpha	3	alpha	0.9977381	True
Sample_549	3	alpha	3	alpha	0.99771416	True
Sample_550	3	alpha	3	alpha	0.9977087	True
Sample_551	3	alpha	3	alpha	0.9977463	True
Sample_552	3	alpha	3	alpha	0.9977584	True
Sample_554	3	alpha	3	alpha	0.9975726	True
Sample_555	3	alpha	3	alpha	0.99771476	True
Sample_556	3	alpha	3	alpha	0.99778205	True
Sample_557	3	alpha	3	alpha	0.99767095	True
Sample_558	3	alpha	3	alpha	0.9976707	True
Sample_559	3	alpha	3	alpha	0.99774665	True
Sample_560	3	alpha	3	alpha	0.9977658	True
Sample_561	3	alpha	3	alpha	0.9977596	True
Sample_562	3	alpha	3	alpha	0.9976115	True
Sample_563	3	alpha	3	alpha	0.9977488	True
Sample_564	3	alpha	3	alpha	0.99772376	True
Sample_565	3	alpha	3	alpha	0.9977666	True
Sample_566	3	alpha	3	alpha	0.9976736	True
Sample_567	3	alpha	3	alpha	0.9977335	True
Sample_568	3	alpha	3	alpha	0.99776757	True
Sample_570	3	alpha	3	alpha	0.99773407	True
Sample_571	3	alpha	3	alpha	0.9976968	True
Sample_572	3	alpha	3	alpha	0.9976865	True
Sample_573	3	alpha	3	alpha	0.99766916	True
Sample_575	3	alpha	3	alpha	0.99773186	True
Sample_576	3	alpha	3	alpha	0.9977266	True
Sample_577	3	alpha	3	alpha	0.9977672	True
Sample_578	3	alpha	3	alpha	0.9977337	True
Sample_579	3	alpha	3	alpha	0.997801	True
Sample_580	3	alpha	3	alpha	0.9974062	True
Sample_581	3	alpha	3	alpha	0.9977742	True
Sample_582	3	alpha	3	alpha	0.9975439	True
Sample_583	3	alpha	3	alpha	0.997735	True
Sample_584	3	alpha	3	alpha	0.9976419	True
Sample_585	5	delta	5	delta	0.7968056	True
Sample_586	5	delta	5	delta	0.71941227	True
Sample_587	5	delta	5	delta	0.9761178	True
Sample_588	5	delta	5	delta	0.9861434	True
Sample_589	5	delta	3	alpha	0.9975998	True
Sample_590	5	delta	5	delta	0.981645	True
Sample_591	5	delta	5	delta	0.84267867	True
Sample_592	5	delta	5	delta	0.9796499	True
Sample_593	5	delta	5	delta	0.98449534	True
Sample_594	5	delta	5	delta	0.98488367	True
Sample_595	5	delta	5	delta	0.98524654	True
Sample_596	5	delta	5	delta	0.98264575	True
Sample_597	5	delta	5	delta	0.9856464	True
Sample_598	5	delta	5	delta	0.9849954	True
Sample_599	5	delta	5	delta	0.96505296	True
Sample_600	5	delta	5	delta	0.98131657	True
Sample_601	5	delta	5	delta	0.9756285	True
Sample_602	5	delta	5	delta	0.9866432	True
Sample_603	5	delta	5	delta	0.98632205	True
Sample_604	5	delta	5	delta	0.98685616	True
Sample_605	5	delta	5	delta	0.9855167	True
Sample_606	5	delta	5	delta	0.98713416	True
Sample_607	5	delta	5	delta	0.98451483	True
Sample_608	5	delta	5	delta	0.8998414	True
Sample_609	5	delta	5	delta	0.97823596	True
Sample_610	5	delta	5	delta	0.98299384	True
Sample_611	5	delta	5	delta	0.9869915	True
Sample_612	5	delta	5	delta	0.9835735	True
Sample_613	0	PP	0	PP	0.53211766	True
Sample_614	0	PP	5	delta	0.5750271	True
Sample_615	0	PP	0	PP	0.5838679	True
Sample_616	0	PP	0	PP	0.8386169	True
Sample_617	0	PP	0	PP	0.8050739	True
Sample_618	0	PP	0	PP	0.68828505	True
Sample_619	0	PP	0	PP	0.7741948	True
Sample_620	0	PP	0	PP	0.7569607	True
Sample_621	0	PP	5	delta	0.48419228	True
Sample_622	0	PP	0	PP	0.73397464	True
Sample_623	0	PP	0	PP	0.83683205	True
Sample_624	0	PP	3	alpha	0.990767	True
Sample_625	0	PP	0	PP	0.83092463	True
Sample_626	0	PP	0	PP	0.8389634	True
Sample_627	0	PP	0	PP	0.7915242	True
Sample_628	0	PP	0	PP	0.7881388	True
Sample_629	0	PP	0	PP	0.8475106	True
Sample_630	0	PP	3	alpha	0.5337762	True
Sample_631	0	PP	0	PP	0.87255657	True
Sample_633	0	PP	0	PP	0.82755685	True
Sample_634	0	PP	0	PP	0.8510661	True
Sample_635	0	PP	0	PP	0.8646854	True
Sample_636	0	PP	0	PP	0.8263182	True
Sample_637	0	PP	3	alpha	0.9031071	True
Sample_638	0	PP	0	PP	0.7019772	True
Sample_639	0	PP	0	PP	0.8400209	True
Sample_641	0	PP	0	PP	0.7561258	True
Sample_643	0	PP	0	PP	0.87294173	True
Sample_644	0	PP	0	PP	0.8141541	True
Sample_646	0	PP	0	PP	0.8292182	True
Sample_647	0	PP	0	PP	0.8439001	True
Sample_650	0	PP	0	PP	0.7212314	True
Sample_651	0	PP	0	PP	0.6744912	True
Sample_652	4	beta	4	beta	0.99804246	True
Sample_653	4	beta	4	beta	0.99795806	True
Sample_654	4	beta	4	beta	0.99809843	True
Sample_655	4	beta	4	beta	0.9980154	True
Sample_656	4	beta	4	beta	0.9980051	True
Sample_657	4	beta	4	beta	0.9980222	True
Sample_658	4	beta	4	beta	0.99806327	True
Sample_659	4	beta	4	beta	0.9981164	True
Sample_660	4	beta	4	beta	0.99793845	True
Sample_661	4	beta	4	beta	0.99808383	True
Sample_662	4	beta	4	beta	0.99815875	True
Sample_663	4	beta	4	beta	0.9981316	True
Sample_664	4	beta	4	beta	0.9977877	True
Sample_666	4	beta	4	beta	0.9978217	True
Sample_668	4	beta	4	beta	0.9924819	True
Sample_669	4	beta	4	beta	0.99799407	True
Sample_670	4	beta	4	beta	0.9978046	True
Sample_671	4	beta	4	beta	0.99777716	True
Sample_672	4	beta	4	beta	0.99796045	True
Sample_673	4	beta	4	beta	0.99778026	True
Sample_674	4	beta	4	beta	0.9980269	True
Sample_675	4	beta	4	beta	0.99780816	True
Sample_676	4	beta	4	beta	0.99806005	True
Sample_677	4	beta	4	beta	0.99800843	True
Sample_678	4	beta	4	beta	0.9981996	True
Sample_679	4	beta	4	beta	0.9980888	True
Sample_680	4	beta	4	beta	0.99789715	True
Sample_682	4	beta	4	beta	0.9977169	True
Sample_683	4	beta	4	beta	0.9974235	True
Sample_684	4	beta	4	beta	0.9979741	True
Sample_685	4	beta	4	beta	0.99774504	True
Sample_686	4	beta	4	beta	0.9979242	True
Sample_687	4	beta	4	beta	0.9978714	True
Sample_688	4	beta	4	beta	0.9980819	True
Sample_689	4	beta	4	beta	0.9975957	True
Sample_690	4	beta	4	beta	0.99775046	True
Sample_691	4	beta	4	beta	0.998061	True
Sample_692	4	beta	4	beta	0.99794966	True
Sample_693	4	beta	4	beta	0.9976921	True
Sample_694	4	beta	4	beta	0.997615	True
Sample_695	4	beta	4	beta	0.99791163	True
Sample_696	4	beta	4	beta	0.9978277	True
Sample_697	4	beta	4	beta	0.9979625	True
Sample_698	4	beta	4	beta	0.99785125	True
Sample_699	4	beta	4	beta	0.99815875	True
Sample_700	4	beta	4	beta	0.99794763	True
Sample_701	4	beta	4	beta	0.99787784	True
Sample_702	4	beta	4	beta	0.99793065	True
Sample_703	4	beta	4	beta	0.9979558	True
Sample_704	4	beta	4	beta	0.998075	True
Sample_705	4	beta	4	beta	0.99790823	True
Sample_706	4	beta	4	beta	0.99782985	True
Sample_707	4	beta	4	beta	0.99778545	True
Sample_708	4	beta	4	beta	0.9977545	True
Sample_709	4	beta	4	beta	0.9977017	True
Sample_711	4	beta	4	beta	0.9980488	True
Sample_712	4	beta	4	beta	0.9979079	True
Sample_713	4	beta	4	beta	0.9976676	True
Sample_714	4	beta	4	beta	0.99806803	True
Sample_715	4	beta	4	beta	0.9978573	True
Sample_716	4	beta	4	beta	0.99775237	True
Sample_717	4	beta	4	beta	0.9981223	True
Sample_718	4	beta	4	beta	0.98077947	True
Sample_719	4	beta	4	beta	0.99810034	True
Sample_721	4	beta	4	beta	0.99754554	True
Sample_722	4	beta	4	beta	0.99806875	True
Sample_723	4	beta	4	beta	0.99763334	True
Sample_724	4	beta	4	beta	0.99796593	True
Sample_725	4	beta	4	beta	0.9968086	True
Sample_726	4	beta	4	beta	0.99778074	True
Sample_727	4	beta	4	beta	0.99787784	True
Sample_728	4	beta	4	beta	0.9977176	True
Sample_729	4	beta	4	beta	0.9980318	True
Sample_730	4	beta	4	beta	0.9980184	True
Sample_732	4	beta	4	beta	0.9980439	True
Sample_733	4	beta	4	beta	0.9979005	True
Sample_734	4	beta	4	beta	0.9981674	True
Sample_736	4	beta	4	beta	0.99781716	True
Sample_737	4	beta	4	beta	0.9973248	True
Sample_739	4	beta	4	beta	0.9979236	True
Sample_740	4	beta	4	beta	0.9980038	True
Sample_741	4	beta	4	beta	0.9981188	True
Sample_742	4	beta	4	beta	0.9975916	True
Sample_743	4	beta	4	beta	0.9977343	True
Sample_744	4	beta	4	beta	0.99778867	True
Sample_745	4	beta	4	beta	0.9979309	True
Sample_746	4	beta	4	beta	0.9978296	True
Sample_747	4	beta	4	beta	0.99794906	True
Sample_748	4	beta	4	beta	0.9981135	True
Sample_749	4	beta	4	beta	0.9979565	True
Sample_750	4	beta	4	beta	0.99807054	True
Sample_751	4	beta	4	beta	0.998071	True
Sample_752	4	beta	4	beta	0.998041	True
Sample_753	4	beta	4	beta	0.99793494	True
Sample_754	4	beta	4	beta	0.99806076	True
Sample_755	4	beta	4	beta	0.9980484	True
Sample_756	4	beta	4	beta	0.997698	True
Sample_757	4	beta	4	beta	0.99790406	True
Sample_758	4	beta	4	beta	0.9977756	True
Sample_759	4	beta	4	beta	0.9979494	True
Sample_760	4	beta	4	beta	0.9979202	True
Sample_761	4	beta	4	beta	0.997755	True
Sample_762	4	beta	4	beta	0.9978605	True
Sample_763	4	beta	4	beta	0.99800855	True
Sample_764	4	beta	4	beta	0.99791473	True
Sample_765	4	beta	4	beta	0.99421495	True
Sample_766	4	beta	4	beta	0.99804	True
Sample_767	4	beta	4	beta	0.99764293	True
Sample_769	4	beta	4	beta	0.9976515	True
Sample_770	4	beta	4	beta	0.9979613	True
Sample_771	4	beta	4	beta	0.9978409	True
Sample_772	4	beta	4	beta	0.99798644	True
Sample_773	4	beta	4	beta	0.9978249	True
Sample_774	4	beta	4	beta	0.997914	True
Sample_775	4	beta	4	beta	0.9980386	True
Sample_776	4	beta	4	beta	0.99795663	True
Sample_777	4	beta	4	beta	0.9976774	True
Sample_778	4	beta	4	beta	0.99794334	True
Sample_779	4	beta	4	beta	0.99776745	True
Sample_780	4	beta	4	beta	0.99801123	True
Sample_781	4	beta	4	beta	0.99788827	True
Sample_782	4	beta	4	beta	0.9979942	True
Sample_783	4	beta	4	beta	0.99812645	True
Sample_784	4	beta	5	delta	0.7645294	True
Sample_785	4	beta	4	beta	0.99745625	True
Sample_786	4	beta	4	beta	0.9980135	True
Sample_787	4	beta	4	beta	0.9982083	True
Sample_788	4	beta	4	beta	0.9978301	True
Sample_789	4	beta	4	beta	0.9980586	True
Sample_790	4	beta	4	beta	0.9978656	True
Sample_791	4	beta	4	beta	0.9979311	True
Sample_792	4	beta	4	beta	0.99816895	True
Sample_793	4	beta	4	beta	0.99812573	True
Sample_794	4	beta	4	beta	0.9978908	True
Sample_795	4	beta	4	beta	0.99807334	True
Sample_796	4	beta	4	beta	0.9978503	True
Sample_797	4	beta	4	beta	0.99805427	True
Sample_798	4	beta	4	beta	0.9976427	True
Sample_799	4	beta	4	beta	0.99775237	True
Sample_800	4	beta	4	beta	0.99797803	True
Sample_801	4	beta	4	beta	0.9979867	True
Sample_802	4	beta	4	beta	0.99797994	True
Sample_803	4	beta	4	beta	0.9976145	True
Sample_804	4	beta	4	beta	0.9977138	True
Sample_805	4	beta	4	beta	0.99793506	True
Sample_806	4	beta	4	beta	0.9981092	True
Sample_807	4	beta	4	beta	0.9980804	True
Sample_808	4	beta	4	beta	0.9980348	True
Sample_809	4	beta	4	beta	0.99790984	True
Sample_810	4	beta	4	beta	0.99356824	True
Sample_811	4	beta	4	beta	0.9979831	True
Sample_812	4	beta	4	beta	0.99798787	True
Sample_813	4	beta	4	beta	0.99790645	True
Sample_814	4	beta	4	beta	0.99803084	True
Sample_815	4	beta	4	beta	0.99814546	True
Sample_816	4	beta	4	beta	0.9981263	True
Sample_817	4	beta	4	beta	0.9978079	True
Sample_818	4	beta	4	beta	0.9981255	True
Sample_819	4	beta	4	beta	0.9977113	True
Sample_821	4	beta	4	beta	0.9980531	True
Sample_823	4	beta	4	beta	0.99785596	True
Sample_824	4	beta	4	beta	0.99714655	True
Sample_825	4	beta	4	beta	0.9979646	True
Sample_826	4	beta	4	beta	0.9980761	True
Sample_827	4	beta	4	beta	0.9979862	True
Sample_828	4	beta	4	beta	0.9979526	True
Sample_829	4	beta	4	beta	0.99789965	True
Sample_830	4	beta	4	beta	0.99795204	True
Sample_831	4	beta	4	beta	0.99807847	True
Sample_832	4	beta	4	beta	0.9979773	True
Sample_833	4	beta	4	beta	0.99793434	True
Sample_834	4	beta	4	beta	0.9979442	True
Sample_835	4	beta	4	beta	0.99816555	True
Sample_837	4	beta	4	beta	0.9982008	True
Sample_838	4	beta	4	beta	0.99780196	True
Sample_839	4	beta	4	beta	0.99800855	True
Sample_840	4	beta	4	beta	0.99793494	True
Sample_841	4	beta	4	beta	0.9981634	True
Sample_842	4	beta	4	beta	0.9979284	True
Sample_843	4	beta	4	beta	0.9981171	True
Sample_844	4	beta	4	beta	0.99784935	True
Sample_845	4	beta	4	beta	0.9979721	True
Sample_846	4	beta	4	beta	0.99787426	True
Sample_847	4	beta	4	beta	0.99784243	True
Sample_848	4	beta	4	beta	0.91236675	True
Sample_849	4	beta	4	beta	0.99441516	True
Sample_850	4	beta	4	beta	0.9978219	True
Sample_851	4	beta	4	beta	0.99804026	True
Sample_852	4	beta	4	beta	0.9979977	True
Sample_853	4	beta	4	beta	0.99807656	True
Sample_854	4	beta	4	beta	0.9979327	True
Sample_855	4	beta	4	beta	0.9975695	True
Sample_856	4	beta	4	beta	0.99787056	True
Sample_857	4	beta	4	beta	0.99808145	True
Sample_858	4	beta	4	beta	0.99781954	True
Sample_859	4	beta	4	beta	0.9942719	True
Sample_860	4	beta	4	beta	0.99775046	True
Sample_861	4	beta	4	beta	0.9981078	True
Sample_862	4	beta	4	beta	0.9980716	True
Sample_863	4	beta	4	beta	0.99809104	True
Sample_864	4	beta	4	beta	0.997985	True
Sample_866	4	beta	4	beta	0.99807787	True
Sample_868	4	beta	4	beta	0.99807006	True
Sample_869	4	beta	4	beta	0.9979001	True
Sample_870	4	beta	4	beta	0.9979796	True
Sample_871	4	beta	4	beta	0.99812394	True
Sample_872	4	beta	4	beta	0.99802846	True
Sample_873	4	beta	4	beta	0.99797803	True
Sample_874	4	beta	4	beta	0.9980939	True
Sample_875	4	beta	4	beta	0.9979668	True
Sample_876	4	beta	4	beta	0.9980368	True
Sample_877	4	beta	4	beta	0.9980788	True
Sample_879	4	beta	4	beta	0.99800617	True
Sample_880	4	beta	4	beta	0.9978789	True
Sample_881	4	beta	4	beta	0.99806815	True
Sample_883	4	beta	4	beta	0.9979912	True
Sample_884	4	beta	4	beta	0.9980411	True
Sample_885	4	beta	4	beta	0.99813336	True
Sample_886	4	beta	4	beta	0.99773693	True
Sample_887	4	beta	4	beta	0.99803954	True
Sample_888	4	beta	4	beta	0.99779934	True
Sample_889	4	beta	4	beta	0.9978077	True
Sample_890	4	beta	4	beta	0.9979693	True
Sample_891	4	beta	4	beta	0.9980538	True
Sample_892	4	beta	4	beta	0.9978963	True
Sample_893	4	beta	4	beta	0.9979905	True
Sample_895	4	beta	4	beta	0.99815077	True
Sample_896	4	beta	4	beta	0.9980494	True
Sample_897	4	beta	4	beta	0.99802005	True
Sample_898	4	beta	4	beta	0.9979949	True
Sample_899	4	beta	4	beta	0.9981207	True
Sample_901	4	beta	4	beta	0.9980457	True
Sample_902	4	beta	4	beta	0.99802655	True
Sample_903	4	beta	4	beta	0.99803597	True
Sample_904	4	beta	4	beta	0.99812454	True
Sample_905	4	beta	4	beta	0.997996	True
Sample_906	4	beta	4	beta	0.997971	True
Sample_907	4	beta	4	beta	0.998014	True
Sample_908	4	beta	4	beta	0.9977658	True
Sample_909	4	beta	4	beta	0.99819773	True
Sample_910	4	beta	4	beta	0.9980268	True
Sample_911	4	beta	4	beta	0.99804485	True
Sample_912	4	beta	4	beta	0.99808013	True
Sample_913	4	beta	4	beta	0.9981505	True
Sample_914	4	beta	4	beta	0.99807215	True
Sample_915	4	beta	4	beta	0.99809927	True
Sample_916	4	beta	4	beta	0.9981477	True
Sample_917	4	beta	4	beta	0.99797946	True
Sample_918	4	beta	4	beta	0.99795294	True
Sample_919	4	beta	4	beta	0.9980976	True
Sample_920	4	beta	4	beta	0.9980574	True
Sample_921	4	beta	4	beta	0.9981225	True
Sample_922	4	beta	4	beta	0.997743	True
Sample_923	4	beta	4	beta	0.9980557	True
Sample_924	4	beta	4	beta	0.997787	True
Sample_925	4	beta	4	beta	0.99807113	True
Sample_926	4	beta	4	beta	0.99801576	True
Sample_927	4	beta	4	beta	0.99805534	True
Sample_928	4	beta	4	beta	0.9978695	True
Sample_929	4	beta	4	beta	0.9955081	True
Sample_930	4	beta	4	beta	0.9980032	True
Sample_931	4	beta	4	beta	0.99801093	True
Sample_932	4	beta	4	beta	0.99808973	True
Sample_933	4	beta	4	beta	0.99799216	True
Sample_934	4	beta	4	beta	0.99804026	True
Sample_935	4	beta	4	beta	0.9978789	True
Sample_936	4	beta	4	beta	0.99804366	True
Sample_937	4	beta	4	beta	0.9972915	True
Sample_938	4	beta	4	beta	0.9974905	True
Sample_939	4	beta	4	beta	0.99811506	True
Sample_940	4	beta	4	beta	0.9975005	True
Sample_941	4	beta	4	beta	0.9978929	True
Sample_942	4	beta	4	beta	0.9978398	True
Sample_943	4	beta	4	beta	0.99796295	True
Sample_944	4	beta	4	beta	0.9976683	True
Sample_945	4	beta	4	beta	0.99807906	True
Sample_946	4	beta	4	beta	0.99803215	True
Sample_947	4	beta	4	beta	0.99804425	True
Sample_948	3	alpha	3	alpha	0.99772483	True
Sample_949	3	alpha	3	alpha	0.99777967	True
Sample_950	3	alpha	3	alpha	0.99776924	True
Sample_951	3	alpha	3	alpha	0.9978132	True
Sample_952	3	alpha	3	alpha	0.9977769	True
Sample_953	3	alpha	3	alpha	0.9977527	True
Sample_954	3	alpha	3	alpha	0.9977277	True
Sample_955	3	alpha	3	alpha	0.99773264	True
Sample_956	3	alpha	3	alpha	0.99775153	True
Sample_957	3	alpha	3	alpha	0.99782604	True
Sample_958	3	alpha	3	alpha	0.99777466	True
Sample_959	3	alpha	3	alpha	0.99772984	True
Sample_960	3	alpha	3	alpha	0.99778086	True
Sample_961	3	alpha	3	alpha	0.9977673	True
Sample_962	3	alpha	3	alpha	0.99774796	True
Sample_963	3	alpha	3	alpha	0.99775034	True
Sample_964	3	alpha	3	alpha	0.99773526	True
Sample_965	3	alpha	3	alpha	0.997663	True
Sample_966	3	alpha	3	alpha	0.9977373	True
Sample_967	3	alpha	3	alpha	0.9977164	True
Sample_968	3	alpha	3	alpha	0.99776995	True
Sample_969	3	alpha	3	alpha	0.9977616	True
Sample_970	3	alpha	3	alpha	0.99771214	True
Sample_971	3	alpha	3	alpha	0.99776757	True
Sample_972	3	alpha	3	alpha	0.9976949	True
Sample_973	3	alpha	3	alpha	0.9976937	True
Sample_974	3	alpha	3	alpha	0.99772054	True
Sample_976	3	alpha	3	alpha	0.99774796	True
Sample_977	3	alpha	3	alpha	0.99783105	True
Sample_978	3	alpha	3	alpha	0.9977283	True
Sample_979	3	alpha	3	alpha	0.9977871	True
Sample_980	3	alpha	3	alpha	0.99773127	True
Sample_981	3	alpha	3	alpha	0.99780446	True
Sample_982	3	alpha	3	alpha	0.9977958	True
Sample_983	3	alpha	3	alpha	0.9977022	True
Sample_984	3	alpha	3	alpha	0.9977337	True
Sample_985	3	alpha	3	alpha	0.9977216	True
Sample_987	3	alpha	3	alpha	0.99769944	True
Sample_988	3	alpha	3	alpha	0.9976852	True
Sample_989	3	alpha	3	alpha	0.9977095	True
Sample_990	3	alpha	3	alpha	0.9977647	True
Sample_991	3	alpha	3	alpha	0.9977443	True
Sample_992	3	alpha	3	alpha	0.99776757	True
Sample_993	3	alpha	3	alpha	0.9977653	True
Sample_994	3	alpha	3	alpha	0.9976768	True
Sample_995	3	alpha	3	alpha	0.99769765	True
Sample_996	3	alpha	3	alpha	0.9976745	True
Sample_997	3	alpha	3	alpha	0.9977239	True
Sample_998	3	alpha	3	alpha	0.9977227	True
Sample_999	3	alpha	3	alpha	0.99768806	True
Sample_1000	3	alpha	3	alpha	0.997449	True
Sample_1001	3	alpha	3	alpha	0.9977634	True
Sample_1002	3	alpha	3	alpha	0.9975369	True
Sample_1003	3	alpha	3	alpha	0.9975388	True
Sample_1005	3	alpha	3	alpha	0.9976714	True
Sample_1006	3	alpha	3	alpha	0.99779403	True
Sample_1007	3	alpha	3	alpha	0.9977163	True
Sample_1008	3	alpha	3	alpha	0.99779415	True
Sample_1009	3	alpha	3	alpha	0.99761814	True
Sample_1010	3	alpha	3	alpha	0.9973023	True
Sample_1011	3	alpha	3	alpha	0.9976851	True
Sample_1012	3	alpha	3	alpha	0.9977539	True
Sample_1013	3	alpha	3	alpha	0.9976549	True
Sample_1014	3	alpha	3	alpha	0.9977314	True
Sample_1015	3	alpha	3	alpha	0.99775785	True
Sample_1016	3	alpha	3	alpha	0.99777156	True
Sample_1017	3	alpha	3	alpha	0.9977208	True
Sample_1018	3	alpha	3	alpha	0.99777323	True
Sample_1020	3	alpha	3	alpha	0.99775654	True
Sample_1022	3	alpha	3	alpha	0.99772173	True
Sample_1023	3	alpha	3	alpha	0.9977119	True
Sample_1024	3	alpha	3	alpha	0.9977545	True
Sample_1025	3	alpha	3	alpha	0.99771273	True
Sample_1026	3	alpha	3	alpha	0.9977469	True
Sample_1027	3	alpha	3	alpha	0.9976635	True
Sample_1028	3	alpha	3	alpha	0.9975363	True
Sample_1029	3	alpha	3	alpha	0.99776554	True
Sample_1030	3	alpha	3	alpha	0.9977043	True
Sample_1032	3	alpha	3	alpha	0.9977139	True
Sample_1033	3	alpha	3	alpha	0.9977406	True
Sample_1034	3	alpha	3	alpha	0.9976931	True
Sample_1035	3	alpha	3	alpha	0.99771273	True
Sample_1036	3	alpha	3	alpha	0.9976835	True
Sample_1037	3	alpha	3	alpha	0.99758625	True
Sample_1038	3	alpha	3	alpha	0.9977946	True
Sample_1039	3	alpha	3	alpha	0.9977543	True
Sample_1040	3	alpha	3	alpha	0.9976955	True
Sample_1041	3	alpha	3	alpha	0.9977938	True
Sample_1042	3	alpha	3	alpha	0.9977223	True
Sample_1043	3	alpha	3	alpha	0.9977908	True
Sample_1044	3	alpha	3	alpha	0.997733	True
Sample_1046	3	alpha	3	alpha	0.99770856	True
Sample_1047	3	alpha	3	alpha	0.997626	True
Sample_1049	3	alpha	3	alpha	0.9977435	True
Sample_1050	3	alpha	3	alpha	0.99770063	True
Sample_1051	3	alpha	3	alpha	0.9977189	True
Sample_1052	3	alpha	3	alpha	0.9977664	True
Sample_1053	3	alpha	3	alpha	0.997629	True
Sample_1054	3	alpha	3	alpha	0.99769247	True
Sample_1055	3	alpha	3	alpha	0.9977264	True
Sample_1056	3	alpha	3	alpha	0.9976786	True
Sample_1057	3	alpha	3	alpha	0.9977144	True
Sample_1058	3	alpha	3	alpha	0.9977112	True
Sample_1059	3	alpha	3	alpha	0.9977272	True
Sample_1060	3	alpha	3	alpha	0.99771476	True
Sample_1061	3	alpha	3	alpha	0.9977424	True
Sample_1062	3	alpha	3	alpha	0.9977957	True
Sample_1063	3	alpha	3	alpha	0.99775356	True
Sample_1064	3	alpha	3	alpha	0.997701	True
Sample_1065	3	alpha	3	alpha	0.99775887	True
Sample_1066	3	alpha	3	alpha	0.9977779	True
Sample_1067	3	alpha	3	alpha	0.9977729	True
Sample_1068	3	alpha	3	alpha	0.9975291	True
Sample_1069	3	alpha	3	alpha	0.9977869	True
Sample_1070	3	alpha	3	alpha	0.9976942	True
Sample_1071	3	alpha	3	alpha	0.9977399	True
Sample_1072	3	alpha	3	alpha	0.99778324	True
Sample_1073	3	alpha	3	alpha	0.9976724	True
Sample_1074	3	alpha	3	alpha	0.9977551	True
Sample_1075	3	alpha	3	alpha	0.99774003	True
Sample_1076	3	alpha	3	alpha	0.9977532	True
Sample_1077	3	alpha	3	alpha	0.99762696	True
Sample_1078	3	alpha	3	alpha	0.9977391	True
Sample_1079	3	alpha	3	alpha	0.99774736	True
Sample_1080	3	alpha	3	alpha	0.99770504	True
Sample_1081	3	alpha	3	alpha	0.99772865	True
Sample_1082	3	alpha	3	alpha	0.9977406	True
Sample_1083	3	alpha	3	alpha	0.99763477	True
Sample_1084	3	alpha	3	alpha	0.9974189	True
Sample_1085	3	alpha	3	alpha	0.9978041	True
Sample_1087	3	alpha	3	alpha	0.9974189	True
Sample_1090	3	alpha	3	alpha	0.99771094	True
Sample_1091	3	alpha	3	alpha	0.9977932	True
Sample_1092	3	alpha	3	alpha	0.99770916	True
Sample_1093	3	alpha	3	alpha	0.9977552	True
Sample_1094	3	alpha	3	alpha	0.9977582	True
Sample_1095	3	alpha	3	alpha	0.99771583	True
Sample_1096	3	alpha	3	alpha	0.99772877	True
Sample_1097	3	alpha	3	alpha	0.9977314	True
Sample_1098	3	alpha	3	alpha	0.99778074	True
Sample_1099	3	alpha	3	alpha	0.9977349	True
Sample_1100	3	alpha	3	alpha	0.99777067	True
Sample_1101	3	alpha	3	alpha	0.9972857	True
Sample_1102	3	alpha	3	alpha	0.9977423	True
Sample_1103	3	alpha	3	alpha	0.997703	True
Sample_1104	3	alpha	3	alpha	0.99771166	True
Sample_1105	3	alpha	3	alpha	0.9977241	True
Sample_1106	3	alpha	3	alpha	0.99768734	True
Sample_1107	3	alpha	3	alpha	0.99777573	True
Sample_1108	3	alpha	3	alpha	0.99774975	True
Sample_1109	3	alpha	3	alpha	0.99772733	True
Sample_1110	3	alpha	3	alpha	0.9978078	True
Sample_1111	3	alpha	3	alpha	0.99776125	True
Sample_1112	3	alpha	3	alpha	0.9977884	True
Sample_1113	3	alpha	3	alpha	0.99768853	True
Sample_1114	3	alpha	3	alpha	0.99762315	True
Sample_1115	3	alpha	3	alpha	0.99768114	True
Sample_1116	3	alpha	3	alpha	0.99777883	True
Sample_1117	3	alpha	3	alpha	0.9976712	True
Sample_1118	3	alpha	3	alpha	0.99772316	True
Sample_1119	3	alpha	3	alpha	0.99778986	True
Sample_1121	3	alpha	3	alpha	0.99778026	True
Sample_1123	3	alpha	3	alpha	0.9978288	True
Sample_1124	3	alpha	3	alpha	0.99773693	True
Sample_1125	3	alpha	3	alpha	0.99772936	True
Sample_1126	3	alpha	3	alpha	0.99776304	True
Sample_1127	3	alpha	3	alpha	0.99771845	True
Sample_1128	3	alpha	3	alpha	0.99774194	True
Sample_1129	3	alpha	3	alpha	0.997712	True
Sample_1130	3	alpha	3	alpha	0.997482	True
Sample_1131	3	alpha	3	alpha	0.9977404	True
Sample_1133	3	alpha	3	alpha	0.9977272	True
Sample_1135	3	alpha	3	alpha	0.997757	True
Sample_1136	3	alpha	3	alpha	0.9975439	True
Sample_1138	3	alpha	3	alpha	0.9977049	True
Sample_1139	3	alpha	3	alpha	0.9971482	True
Sample_1141	3	alpha	3	alpha	0.99781334	True
Sample_1142	3	alpha	3	alpha	0.9977652	True
Sample_1143	3	alpha	3	alpha	0.99777764	True
Sample_1144	3	alpha	3	alpha	0.9977592	True
Sample_1145	3	alpha	3	alpha	0.997755	True
Sample_1146	3	alpha	3	alpha	0.99770606	True
Sample_1147	3	alpha	3	alpha	0.99767965	True
Sample_1148	3	alpha	3	alpha	0.99772626	True
Sample_1149	3	alpha	3	alpha	0.99675864	True
Sample_1150	3	alpha	3	alpha	0.9975867	True
Sample_1151	3	alpha	3	alpha	0.99771106	True
Sample_1152	3	alpha	3	alpha	0.9977253	True
Sample_1153	3	alpha	3	alpha	0.99771345	True
Sample_1154	3	alpha	3	alpha	0.9977355	True
Sample_1155	3	alpha	3	alpha	0.997689	True
Sample_1156	3	alpha	3	alpha	0.9976332	True
Sample_1157	3	alpha	3	alpha	0.9976125	True
Sample_1158	3	alpha	3	alpha	0.9977042	True
Sample_1159	3	alpha	3	alpha	0.9977584	True
Sample_1160	3	alpha	3	alpha	0.99760485	True
Sample_1161	3	alpha	3	alpha	0.99777395	True
Sample_1162	3	alpha	3	alpha	0.9977149	True
Sample_1163	3	alpha	3	alpha	0.99772424	True
Sample_1164	3	alpha	3	alpha	0.99771047	True
Sample_1165	3	alpha	3	alpha	0.99768066	True
Sample_1166	3	alpha	3	alpha	0.997598	True
Sample_1167	3	alpha	3	alpha	0.9977278	True
Sample_1168	3	alpha	3	alpha	0.9976713	True
Sample_1169	3	alpha	3	alpha	0.9975284	True
Sample_1170	3	alpha	3	alpha	0.9976421	True
Sample_1171	3	alpha	3	alpha	0.99768	True
Sample_1172	3	alpha	3	alpha	0.99774665	True
Sample_1173	3	alpha	3	alpha	0.9977386	True
Sample_1174	3	alpha	3	alpha	0.99764484	True
Sample_1175	3	alpha	3	alpha	0.99770856	True
Sample_1176	3	alpha	3	alpha	0.9975042	True
Sample_1177	3	alpha	3	alpha	0.9975552	True
Sample_1178	3	alpha	3	alpha	0.9974643	True
Sample_1179	3	alpha	3	alpha	0.99769527	True
Sample_1180	3	alpha	3	alpha	0.9977481	True
Sample_1181	3	alpha	3	alpha	0.9977229	True
Sample_1182	3	alpha	3	alpha	0.9972568	True
Sample_1185	3	alpha	3	alpha	0.99767035	True
Sample_1186	3	alpha	3	alpha	0.99763715	True
Sample_1187	3	alpha	3	alpha	0.9977848	True
Sample_1188	3	alpha	3	alpha	0.9977048	True
Sample_1189	3	alpha	3	alpha	0.99765176	True
Sample_1190	3	alpha	3	alpha	0.99771166	True
Sample_1191	3	alpha	3	alpha	0.99775106	True
Sample_1192	3	alpha	3	alpha	0.997741	True
Sample_1193	3	alpha	3	alpha	0.99759066	True
Sample_1194	3	alpha	3	alpha	0.997459	True
Sample_1195	3	alpha	3	alpha	0.99768734	True
Sample_1196	3	alpha	3	alpha	0.9977714	True
Sample_1197	3	alpha	3	alpha	0.9977603	True
Sample_1198	3	alpha	3	alpha	0.99771655	True
Sample_1199	3	alpha	3	alpha	0.99776316	True
Sample_1200	3	alpha	3	alpha	0.99769765	True
Sample_1201	3	alpha	3	alpha	0.99779034	True
Sample_1202	3	alpha	3	alpha	0.9977744	True
Sample_1203	3	alpha	3	alpha	0.9977581	True
Sample_1204	3	alpha	3	alpha	0.9977804	True
Sample_1205	3	alpha	3	alpha	0.99767274	True
Sample_1206	3	alpha	3	alpha	0.99782914	True
Sample_1207	3	alpha	3	alpha	0.9978004	True
Sample_1208	3	alpha	3	alpha	0.99782336	True
Sample_1209	3	alpha	3	alpha	0.99782157	True
Sample_1210	3	alpha	3	alpha	0.99778974	True
Sample_1211	3	alpha	3	alpha	0.9977881	True
Sample_1212	3	alpha	3	alpha	0.99770063	True
Sample_1213	3	alpha	3	alpha	0.99777067	True
Sample_1214	3	alpha	3	alpha	0.99780554	True
Sample_1215	3	alpha	3	alpha	0.9977986	True
Sample_1216	3	alpha	3	alpha	0.9977805	True
Sample_1217	3	alpha	3	alpha	0.9975931	True
Sample_1218	3	alpha	3	alpha	0.9977514	True
Sample_1219	3	alpha	3	alpha	0.99777275	True
Sample_1220	3	alpha	3	alpha	0.99778557	True
Sample_1221	3	alpha	3	alpha	0.99779046	True
Sample_1222	3	alpha	3	alpha	0.99772567	True
Sample_1223	3	alpha	3	alpha	0.99776804	True
Sample_1224	3	alpha	3	alpha	0.99778074	True
Sample_1225	3	alpha	3	alpha	0.99776125	True
Sample_1226	3	alpha	3	alpha	0.9977884	True
Sample_1227	3	alpha	3	alpha	0.99774456	True
Sample_1228	3	alpha	3	alpha	0.99780065	True
Sample_1229	3	alpha	3	alpha	0.9976828	True
Sample_1230	3	alpha	3	alpha	0.9977028	True
Sample_1231	3	alpha	3	alpha	0.9977373	True
Sample_1232	3	alpha	3	alpha	0.9977412	True
Sample_1233	3	alpha	3	alpha	0.997718	True
Sample_1234	3	alpha	3	alpha	0.9977575	True
Sample_1235	3	alpha	3	alpha	0.99775535	True
Sample_1236	3	alpha	3	alpha	0.99773896	True
Sample_1237	3	alpha	3	alpha	0.9977581	True
Sample_1238	3	alpha	3	alpha	0.99761754	True
Sample_1239	3	alpha	3	alpha	0.99764997	True
Sample_1240	3	alpha	3	alpha	0.9976593	True
Sample_1241	3	alpha	3	alpha	0.99781334	True
Sample_1242	3	alpha	3	alpha	0.9977392	True
Sample_1243	3	alpha	3	alpha	0.99772865	True
Sample_1244	3	alpha	3	alpha	0.9976672	True
Sample_1245	3	alpha	3	alpha	0.9977477	True
Sample_1246	3	alpha	3	alpha	0.9977469	True
Sample_1247	3	alpha	3	alpha	0.9977348	True
Sample_1248	3	alpha	3	alpha	0.9977596	True
Sample_1249	3	alpha	3	alpha	0.99764246	True
Sample_1250	3	alpha	3	alpha	0.99773276	True
Sample_1251	3	alpha	3	alpha	0.997769	True
Sample_1252	3	alpha	3	alpha	0.9977786	True
Sample_1253	3	alpha	3	alpha	0.99776185	True
Sample_1254	3	alpha	3	alpha	0.99750346	True
Sample_1256	3	alpha	3	alpha	0.99775356	True
Sample_1257	3	alpha	3	alpha	0.99770564	True
Sample_1258	3	alpha	3	alpha	0.9977658	True
Sample_1259	3	alpha	3	alpha	0.99775356	True
Sample_1260	3	alpha	3	alpha	0.99766684	True
Sample_1261	3	alpha	3	alpha	0.99770737	True
Sample_1262	3	alpha	3	alpha	0.9977708	True
Sample_1263	3	alpha	3	alpha	0.99773574	True
Sample_1264	3	alpha	3	alpha	0.9977429	True
Sample_1265	3	alpha	3	alpha	0.9977533	True
Sample_1266	3	alpha	3	alpha	0.9976687	True
Sample_1267	3	alpha	3	alpha	0.9977214	True
Sample_1269	3	alpha	3	alpha	0.9977405	True
Sample_1270	3	alpha	3	alpha	0.9977702	True
Sample_1271	3	alpha	3	alpha	0.9977494	True
Sample_1272	3	alpha	3	alpha	0.99776673	True
Sample_1273	3	alpha	3	alpha	0.9977441	True
Sample_1274	3	alpha	3	alpha	0.99770904	True
Sample_1275	3	alpha	3	alpha	0.9977701	True
Sample_1276	3	alpha	3	alpha	0.99776006	True
Sample_1277	3	alpha	3	alpha	0.99773276	True
Sample_1278	3	alpha	3	alpha	0.9977329	True
Sample_1279	3	alpha	3	alpha	0.9976119	True
Sample_1280	3	alpha	3	alpha	0.9977082	True
Sample_1281	3	alpha	3	alpha	0.99773633	True
Sample_1282	3	alpha	3	alpha	0.9976839	True
Sample_1283	3	alpha	3	alpha	0.9977648	True
Sample_1284	3	alpha	3	alpha	0.9977513	True
Sample_1285	3	alpha	3	alpha	0.99775606	True
Sample_1286	3	alpha	3	alpha	0.99768996	True
Sample_1287	3	alpha	3	alpha	0.99761873	True
Sample_1288	3	alpha	3	alpha	0.99774605	True
Sample_1289	3	alpha	3	alpha	0.99776113	True
Sample_1290	3	alpha	3	alpha	0.99774927	True
Sample_1291	3	alpha	3	alpha	0.9977652	True
Sample_1292	3	alpha	3	alpha	0.997752	True
Sample_1293	3	alpha	3	alpha	0.9977494	True
Sample_1294	3	alpha	3	alpha	0.9977799	True
Sample_1295	3	alpha	3	alpha	0.9977137	True
Sample_1297	3	alpha	3	alpha	0.9976745	True
Sample_1298	3	alpha	3	alpha	0.9977143	True
Sample_1299	3	alpha	3	alpha	0.99775153	True
Sample_1300	3	alpha	3	alpha	0.99773216	True
Sample_1301	3	alpha	3	alpha	0.9974752	True
Sample_1302	3	alpha	3	alpha	0.9977787	True
Sample_1303	3	alpha	3	alpha	0.9977627	True
Sample_1304	3	alpha	3	alpha	0.99774003	True
Sample_1305	3	alpha	3	alpha	0.99771273	True
Sample_1307	3	alpha	3	alpha	0.99777836	True
Sample_1308	3	alpha	3	alpha	0.99768794	True
Sample_1309	3	alpha	3	alpha	0.9977087	True
Sample_1311	3	alpha	3	alpha	0.9977265	True
Sample_1312	3	alpha	3	alpha	0.99774724	True
Sample_1313	3	alpha	3	alpha	0.99758995	True
Sample_1314	3	alpha	3	alpha	0.9977272	True
Sample_1315	3	alpha	3	alpha	0.9977539	True
Sample_1316	3	alpha	3	alpha	0.9977683	True
Sample_1317	3	alpha	3	alpha	0.99767774	True
Sample_1318	3	alpha	3	alpha	0.99766207	True
Sample_1319	3	alpha	3	alpha	0.9977563	True
Sample_1320	3	alpha	3	alpha	0.9977316	True
Sample_1321	3	alpha	3	alpha	0.99777466	True
Sample_1322	3	alpha	3	alpha	0.99750924	True
Sample_1323	3	alpha	3	alpha	0.9975981	True
Sample_1324	3	alpha	3	alpha	0.99769694	True
Sample_1325	3	alpha	3	alpha	0.997792	True
Sample_1326	3	alpha	3	alpha	0.9977762	True
Sample_1327	3	alpha	3	alpha	0.9977622	True
Sample_1328	3	alpha	3	alpha	0.997726	True
Sample_1329	3	alpha	3	alpha	0.9977393	True
Sample_1330	3	alpha	3	alpha	0.9976956	True
Sample_1331	3	alpha	3	alpha	0.9977367	True
Sample_1332	3	alpha	3	alpha	0.9977937	True
Sample_1333	3	alpha	3	alpha	0.9977449	True
Sample_1334	3	alpha	3	alpha	0.9977258	True
Sample_1335	3	alpha	3	alpha	0.99775594	True
Sample_1336	3	alpha	3	alpha	0.997733	True
Sample_1338	3	alpha	3	alpha	0.99709237	True
Sample_1340	3	alpha	3	alpha	0.9977502	True
Sample_1341	3	alpha	3	alpha	0.99774265	True
Sample_1342	3	alpha	3	alpha	0.9977081	True
Sample_1343	3	alpha	3	alpha	0.99771166	True
Sample_1344	3	alpha	3	alpha	0.99775034	True
Sample_1346	3	alpha	3	alpha	0.99776745	True
Sample_1347	3	alpha	3	alpha	0.99765104	True
Sample_1348	3	alpha	3	alpha	0.9977192	True
Sample_1349	3	alpha	3	alpha	0.9977441	True
Sample_1350	3	alpha	3	alpha	0.99755585	True
Sample_1351	3	alpha	3	alpha	0.9977494	True
Sample_1352	3	alpha	3	alpha	0.99773127	True
Sample_1353	3	alpha	3	alpha	0.997687	True
Sample_1354	3	alpha	3	alpha	0.9976761	True
Sample_1355	3	alpha	3	alpha	0.99777156	True
Sample_1356	3	alpha	3	alpha	0.9977514	True
Sample_1357	3	alpha	3	alpha	0.99774456	True
Sample_1358	3	alpha	3	alpha	0.9976835	True
Sample_1359	3	alpha	3	alpha	0.9977228	True
Sample_1360	3	alpha	3	alpha	0.99765044	True
Sample_1361	3	alpha	3	alpha	0.99776137	True
Sample_1362	3	alpha	3	alpha	0.99773645	True
Sample_1363	3	alpha	3	alpha	0.99770385	True
Sample_1364	3	alpha	3	alpha	0.99773407	True
Sample_1365	3	alpha	3	alpha	0.99776495	True
Sample_1366	3	alpha	3	alpha	0.9976584	True
Sample_1367	3	alpha	3	alpha	0.99776375	True
Sample_1368	3	alpha	3	alpha	0.9977544	True
Sample_1369	3	alpha	3	alpha	0.9977385	True
Sample_1370	3	alpha	3	alpha	0.997759	True
Sample_1371	3	alpha	3	alpha	0.99774605	True
Sample_1372	3	alpha	3	alpha	0.99770504	True
Sample_1373	3	alpha	3	alpha	0.9976902	True
Sample_1374	3	alpha	3	alpha	0.9977094	True
Sample_1375	3	alpha	3	alpha	0.9977324	True
Sample_1377	3	alpha	3	alpha	0.9977379	True
Sample_1378	3	alpha	3	alpha	0.9977604	True
Sample_1379	3	alpha	3	alpha	0.99771607	True
Sample_1380	3	alpha	3	alpha	0.99774736	True
Sample_1381	3	alpha	3	alpha	0.997681	True
Sample_1382	3	alpha	3	alpha	0.9977108	True
Sample_1383	3	alpha	3	alpha	0.9976382	True
Sample_1384	3	alpha	3	alpha	0.9976928	True
Sample_1385	3	alpha	3	alpha	0.9977106	True
Sample_1386	3	alpha	3	alpha	0.99770004	True
Sample_1387	3	alpha	3	alpha	0.99774903	True
Sample_1388	3	alpha	3	alpha	0.99777067	True
Sample_1389	3	alpha	3	alpha	0.99756336	True
Sample_1390	3	alpha	3	alpha	0.9976313	True
Sample_1391	3	alpha	3	alpha	0.9977368	True
Sample_1392	3	alpha	3	alpha	0.9975218	True
Sample_1393	3	alpha	3	alpha	0.9976553	True
Sample_1394	3	alpha	3	alpha	0.99771166	True
Sample_1395	3	alpha	3	alpha	0.99749124	True
Sample_1396	3	alpha	3	alpha	0.99769586	True
Sample_1397	3	alpha	3	alpha	0.9977041	True
Sample_1398	3	alpha	3	alpha	0.99770254	True
Sample_1399	3	alpha	3	alpha	0.9975085	True
Sample_1401	3	alpha	3	alpha	0.99769574	True
Sample_1402	3	alpha	3	alpha	0.99769896	True
Sample_1403	3	alpha	3	alpha	0.9978084	True
Sample_1404	3	alpha	3	alpha	0.99769115	True
Sample_1405	3	alpha	3	alpha	0.9977404	True
Sample_1406	3	alpha	3	alpha	0.99772865	True
Sample_1407	3	alpha	3	alpha	0.9976993	True
Sample_1408	3	alpha	3	alpha	0.9977272	True
Sample_1409	3	alpha	3	alpha	0.99774384	True
Sample_1410	3	alpha	3	alpha	0.9976979	True
Sample_1411	3	alpha	3	alpha	0.99767107	True
Sample_1412	3	alpha	3	alpha	0.9977563	True
Sample_1414	3	alpha	3	alpha	0.9977482	True
Sample_1415	3	alpha	3	alpha	0.9977375	True
Sample_1417	3	alpha	3	alpha	0.99776447	True
Sample_1418	3	alpha	3	alpha	0.99775237	True
Sample_1419	3	alpha	3	alpha	0.99769235	True
Sample_1420	3	alpha	3	alpha	0.9976139	True
Sample_1421	3	alpha	3	alpha	0.9976419	True
Sample_1422	3	alpha	3	alpha	0.9975605	True
Sample_1423	3	alpha	3	alpha	0.99765694	True
Sample_1424	3	alpha	3	alpha	0.9977296	True
Sample_1425	3	alpha	3	alpha	0.9977422	True
Sample_1426	3	alpha	3	alpha	0.9977354	True
Sample_1427	3	alpha	3	alpha	0.9976724	True
Sample_1428	3	alpha	3	alpha	0.99769384	True
Sample_1429	3	alpha	3	alpha	0.9977283	True
Sample_1430	3	alpha	3	alpha	0.99775714	True
Sample_1431	3	alpha	3	alpha	0.99770004	True
Sample_1432	3	alpha	3	alpha	0.9977465	True
Sample_1433	3	alpha	3	alpha	0.99776614	True
Sample_1434	3	alpha	3	alpha	0.9976574	True
Sample_1435	3	alpha	3	alpha	0.9977379	True
Sample_1436	3	alpha	3	alpha	0.99776924	True
Sample_1437	3	alpha	3	alpha	0.9977016	True
Sample_1438	3	alpha	3	alpha	0.99774003	True
Sample_1439	3	alpha	3	alpha	0.99775165	True
Sample_1440	3	alpha	3	alpha	0.9977168	True
Sample_1441	3	alpha	3	alpha	0.9975914	True
Sample_1442	3	alpha	3	alpha	0.99769336	True
Sample_1443	3	alpha	3	alpha	0.9977375	True
Sample_1444	3	alpha	3	alpha	0.9976827	True
Sample_1445	3	alpha	3	alpha	0.9976985	True
Sample_1446	3	alpha	3	alpha	0.9976909	True
Sample_1447	3	alpha	3	alpha	0.9977774	True
Sample_1448	3	alpha	3	alpha	0.9974558	True
Sample_1449	3	alpha	3	alpha	0.99767023	True
Sample_1450	3	alpha	3	alpha	0.9977289	True
Sample_1451	3	alpha	3	alpha	0.9976714	True
Sample_1452	3	alpha	3	alpha	0.9977107	True
Sample_1453	3	alpha	3	alpha	0.9977411	True
Sample_1454	3	alpha	3	alpha	0.99767977	True
Sample_1455	3	alpha	3	alpha	0.997736	True
Sample_1456	3	alpha	3	alpha	0.9977755	True
Sample_1457	3	alpha	3	alpha	0.99759173	True
Sample_1458	3	alpha	3	alpha	0.9977464	True
Sample_1459	3	alpha	3	alpha	0.99770564	True
Sample_1461	3	alpha	3	alpha	0.99776137	True
Sample_1462	3	alpha	3	alpha	0.9977362	True
Sample_1463	3	alpha	3	alpha	0.99763584	True
Sample_1464	3	alpha	3	alpha	0.99777395	True
Sample_1465	3	alpha	3	alpha	0.99771786	True
Sample_1466	3	alpha	3	alpha	0.9977264	True
Sample_1467	3	alpha	3	alpha	0.9977209	True
Sample_1468	3	alpha	3	alpha	0.9976381	True
Sample_1469	3	alpha	3	alpha	0.99774736	True
Sample_1470	3	alpha	3	alpha	0.99770975	True
Sample_1471	3	alpha	3	alpha	0.99770904	True
Sample_1472	3	alpha	3	alpha	0.99767727	True
Sample_1473	3	alpha	3	alpha	0.99766636	True
Sample_1474	3	alpha	3	alpha	0.9977416	True
Sample_1475	3	alpha	3	alpha	0.9977337	True
Sample_1476	3	alpha	3	alpha	0.99771726	True
Sample_1478	3	alpha	3	alpha	0.9977398	True
Sample_1479	3	alpha	3	alpha	0.99743795	True
Sample_1480	3	alpha	3	alpha	0.9976693	True
Sample_1481	3	alpha	3	alpha	0.9978029	True
Sample_1482	3	alpha	3	alpha	0.9975484	True
Sample_1483	3	alpha	3	alpha	0.997735	True
Sample_1484	3	alpha	3	alpha	0.9977781	True
Sample_1485	3	alpha	3	alpha	0.99769235	True
Sample_1487	3	alpha	3	alpha	0.9977203	True
Sample_1488	3	alpha	3	alpha	0.9976826	True
Sample_1489	3	alpha	3	alpha	0.99764997	True
Sample_1490	3	alpha	3	alpha	0.9974745	True
Sample_1491	3	alpha	3	alpha	0.99772424	True
Sample_1492	3	alpha	3	alpha	0.9976804	True
Sample_1493	3	alpha	3	alpha	0.9975993	True
Sample_1495	3	alpha	3	alpha	0.9977264	True
Sample_1496	3	alpha	3	alpha	0.99773175	True
Sample_1497	3	alpha	3	alpha	0.99773055	True
Sample_1498	3	alpha	3	alpha	0.9977277	True
Sample_1499	3	alpha	3	alpha	0.99772877	True
Sample_1500	3	alpha	3	alpha	0.9976937	True
Sample_1501	3	alpha	3	alpha	0.99773407	True
Sample_1502	3	alpha	3	alpha	0.99773836	True
Sample_1503	3	alpha	3	alpha	0.9977101	True
Sample_1506	3	alpha	3	alpha	0.9976084	True
Sample_1507	3	alpha	3	alpha	0.99770653	True
Sample_1508	3	alpha	3	alpha	0.9976562	True
Sample_1509	3	alpha	3	alpha	0.997658	True
Sample_1510	3	alpha	3	alpha	0.99764436	True
Sample_1511	3	alpha	3	alpha	0.9976853	True
Sample_1512	3	alpha	3	alpha	0.9977616	True
Sample_1513	3	alpha	3	alpha	0.9977164	True
Sample_1514	3	alpha	3	alpha	0.99768984	True
Sample_1515	3	alpha	3	alpha	0.99766076	True
Sample_1516	3	alpha	3	alpha	0.9977133	True
Sample_1517	5	delta	5	delta	0.9871105	True
Sample_1518	5	delta	5	delta	0.98683923	True
Sample_1519	5	delta	5	delta	0.9667513	True
Sample_1522	5	delta	5	delta	0.98122185	True
Sample_1523	5	delta	5	delta	0.9515418	True
Sample_1525	5	delta	4	beta	0.74455655	True
Sample_1526	5	delta	5	delta	0.95934284	True
Sample_1527	5	delta	5	delta	0.96391	True
Sample_1528	5	delta	5	delta	0.9785436	True
Sample_1529	5	delta	5	delta	0.9835824	True
Sample_1533	5	delta	5	delta	0.98607635	True
Sample_1534	5	delta	5	delta	0.9848375	True
Sample_1535	5	delta	5	delta	0.9872371	True
Sample_1536	5	delta	5	delta	0.97975725	True
Sample_1537	5	delta	5	delta	0.9409061	True
Sample_1538	5	delta	5	delta	0.987613	True
Sample_1539	5	delta	5	delta	0.9874899	True
Sample_1540	5	delta	5	delta	0.98285204	True
Sample_1541	5	delta	5	delta	0.86187136	True
Sample_1542	5	delta	5	delta	0.98373187	True
Sample_1546	5	delta	5	delta	0.98728406	True
Sample_1547	0	PP	0	PP	0.8398278	True
Sample_1548	0	PP	0	PP	0.83850676	True
Sample_1549	0	PP	3	alpha	0.7886169	True
Sample_1550	0	PP	0	PP	0.80755746	True
Sample_1551	0	PP	0	PP	0.68826866	True
Sample_1552	0	PP	0	PP	0.4077359	True
Sample_1553	0	PP	0	PP	0.71985435	True
Sample_1554	0	PP	0	PP	0.84189034	True
Sample_1555	0	PP	0	PP	0.6745167	True
Sample_1556	0	PP	0	PP	0.8919428	True
Sample_1557	0	PP	0	PP	0.7180516	True
Sample_1558	0	PP	0	PP	0.89131147	True
Sample_1559	0	PP	0	PP	0.7842758	True
Sample_1560	0	PP	0	PP	0.7956324	True
Sample_1561	0	PP	0	PP	0.72982496	True
Sample_1562	0	PP	0	PP	0.7760563	True
Sample_1563	0	PP	0	PP	0.8167157	True
Sample_1564	0	PP	0	PP	0.71232766	True
Sample_1565	0	PP	0	PP	0.7993338	True
Sample_1566	0	PP	0	PP	0.73575884	True
Sample_1567	0	PP	0	PP	0.85816103	True
Sample_1568	0	PP	0	PP	0.8233444	True
Sample_1569	0	PP	0	PP	0.6645971	True
Sample_1570	0	PP	0	PP	0.70186645	True
Sample_1571	0	PP	0	PP	0.8262265	True
Sample_1572	0	PP	0	PP	0.8442991	True
Sample_1573	0	PP	0	PP	0.80615324	True
Sample_1574	0	PP	0	PP	0.8294894	True
Sample_1575	0	PP	0	PP	0.9002562	True
Sample_1576	0	PP	0	PP	0.50418884	True
Sample_1577	0	PP	0	PP	0.5432356	True
Sample_1578	0	PP	0	PP	0.7700356	True
Sample_1579	0	PP	3	alpha	0.4804823	True
Sample_1580	0	PP	0	PP	0.85129654	True
Sample_1581	0	PP	0	PP	0.76598823	True
Sample_1582	0	PP	0	PP	0.8596778	True
Sample_1583	0	PP	3	alpha	0.9520527	True
Sample_1584	0	PP	0	PP	0.7197355	True
Sample_1585	0	PP	5	delta	0.49523503	True
Sample_1586	0	PP	0	PP	0.7306246	True
Sample_1587	0	PP	0	PP	0.86506957	True
Sample_1588	0	PP	0	PP	0.8146716	True
Sample_1589	0	PP	0	PP	0.834919	True
Sample_1590	0	PP	0	PP	0.6896411	True
Sample_1591	0	PP	0	PP	0.7707605	True
Sample_1592	0	PP	0	PP	0.8861513	True
Sample_1593	0	PP	0	PP	0.8559807	True
Sample_1594	0	PP	0	PP	0.84134156	True
Sample_1595	0	PP	0	PP	0.9116589	True
Sample_1597	0	PP	0	PP	0.6678551	True
Sample_1598	0	PP	3	alpha	0.7348347	True
Sample_1600	0	PP	0	PP	0.8647606	True
```
<!-- END_RESTORE_FILE -->


</details>

## file-14d94b84e650

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/requirements_freeze.txt` · SHA256: `432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d`

<details>
<summary>requirements_freeze.txt — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/requirements_freeze.txt", "sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d", "encoding": "utf-8", "added_final_newline": false} -->
```txt
absl-py==2.4.0
aiohappyeyeballs==2.6.2
aiohttp==3.14.1
aiosignal==1.4.0
anndata==0.12.19
annotated-doc==0.0.4
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.13.0
apex @ file:///opt/pytorch/apex/dist/apex-0.1-cp312-cp312-linux_aarch64.whl#sha256=da723b7848638dfc7e1e58b8e27bd37237b093e57700154de8ed8dc61d9d10e6
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
array-api-compat==1.15.0
arrow==1.4.0
asttokens==3.0.1
astunparse==1.6.3
async-lru==2.3.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.15.0
black==26.5.1
bleach==6.4.0
certifi==2026.5.20
cffi==2.0.0
charset-normalizer==3.4.7
click==8.4.1
cloudpickle==3.1.2
cmake==3.31.6
comm==0.2.3
contourpy==1.3.3
cuda-bindings==13.3.1
cuda-core==1.0.1
cuda-pathfinder==1.5.5
cuda-python==13.3.1
cuda-tile==1.4.0
cycler==0.12.1
Cython==3.2.5
datasets==2.21.0
debugpy==1.8.21
decorator==5.3.1
defusedxml==0.7.1
dill==0.3.8
dllist==2.0.0
docstring_parser==0.18.0
donfig==0.8.1.post1
einops==0.8.2
executing==2.2.1
fast-array-utils==1.5.1
fastjsonschema==2.21.2
filelock==3.29.1
flash_attn @ file:///opt/transfer/flash_attn-2.7.4.post1%2Bgit5231d95fe1.54157505-cp312-cp312-linux_aarch64.whl#sha256=c582b983568ec7175eba847adef22b6b380efbe6c2ca6c4103ed778ed014613f
fonttools==4.63.0
formulaic==1.2.2
fqdn==1.5.1
frozenlist==1.8.0
fsspec==2024.6.1
gast==0.7.0
gdown==6.4.2
gitdb==4.0.12
GitPython==3.1.50
google-crc32c==1.9.0
grpcio==1.81.0
grpcio-tools==1.81.0
h11==0.16.0
h5py==3.16.0
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.18.0
idna==3.18
importlib_metadata==9.0.0
interface_meta==2.0.1
ipykernel==7.2.0
ipython==9.14.1
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
isort==8.0.1
jedi==0.20.0
Jinja2==3.1.6
joblib==1.6.0
json5==0.14.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_client==8.9.0
jupyter_core==5.9.1
jupyter_server==2.19.0
jupyter_server_terminals==0.5.4
jupyterlab==4.5.8
jupyterlab_code_formatter==3.0.3
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
jupyterlab_tensorboard_pro @ file:///builder/jupyterlab_tensorboard_pro-4.0.0-py2.py3-none-any.whl#sha256=b639e0d57174368cba55246fb2b225bc5eb99065b2f7e04366d10fdb8281f660
jupytext==1.19.3
kiwisolver==1.5.0
lark==1.3.1
legacy-api-wrap==1.5
llvmlite==0.46.0
makefun==1.16.0
Markdown==3.10.2
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.9
matplotlib-inline==0.2.2
mdit-py-plugins==0.6.1
mdurl==0.1.2
mistune==3.2.1
ml_dtypes==0.5.4
mpmath==1.3.0
msgspec==0.22.0
multidict==6.7.1
multiprocess==0.70.16
mypy_extensions==1.1.0
narwhals==2.27.1
natsort==8.4.0
nbclient==0.11.0
nbconvert==7.17.1
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.6.1
ninja==1.13.0
notebook==7.5.7
notebook_shim==0.2.4
numba==0.64.0
numcodecs==0.17.1
numpy==2.1.0
nvdlfw_inspect==0.2.2
nvidia-cutlass-dsl==4.5.1
nvidia-cutlass-dsl-libs-base==4.5.1
nvidia-dali-cuda130==2.1.1
nvidia-libnvcomp-cu13==5.1.0.21
nvidia-ml-py==13.610.43
nvidia-modelopt @ file:///tmp/modelopt/nvidia_modelopt-0.44.0-py3-none-any.whl#sha256=9b54a853dfda161db97a0dfce4d7c24269d1d19966b5e2026094186af897f74d
nvidia-nvimgcodec-cu13==0.7.0.49
nvidia-nvjpeg==13.2.0.21
nvidia-nvjpeg2k-cu13==0.10.0.49
nvidia-nvtiff-cu13==0.7.0.79
nvidia-resiliency-ext @ file:///tmp/nvrx/nvidia_resiliency_ext-0.6.0-cp312-cp312-manylinux_2_39_aarch64.whl#sha256=a71c1a1650b32c72f7fdf7d6f73b4d275961ce5d68bd68efaff9e0d7099031cb
nvtx==0.2.15
omegaconf==2.3.0
onnx @ file:///opt/pytorch/pytorch/third_party/onnx
onnx-ir==0.2.1
onnxscript==0.7.0
optree==0.19.1
packaging==26.0
pandas==2.3.3
pandocfilters==1.5.1
parso==0.8.7
pathspec==1.1.1
patsy==1.0.3
pexpect==4.9.0
pillow==12.2.0
platformdirs==4.10.0
polygraphy==0.50.3
prometheus_client==0.25.0
prompt_toolkit==3.0.52
propcache==0.5.2
protobuf==6.33.6
psutil==7.2.2
ptyprocess==0.7.0
PuLP==3.3.2
pure_eval==0.2.3
pyarrow==24.0.0
pybind11==3.0.4
pycocotools @ git+https://github.com/nvidia/cocoapi.git@08350a157b41f0755a536441b5abec89ab934c5b#subdirectory=PythonAPI
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
Pygments==2.17.2
pynndescent==0.6.0
pyparsing==3.3.2
PySocks==1.7.1
python-dateutil==2.9.0.post0
python-json-logger==4.1.0
python_hostlist==2.3.0
pytokens==0.4.1
pytz==2026.5
PyYAML==6.0.1
pyzmq==27.1.0
referencing==0.37.0
regex==2026.5.9
requests==2.34.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rich==15.0.0
rpds-py==2026.5.1
safetensors==0.8.0
scanpy==1.12.4
scGPT @ git+https://github.com/bowang-lab/scGPT.git@cebd6fae655b9c585a4807daa3ac31bb764f06b4
scikit-learn==1.9.1
scikit-misc==0.5.3
scipy==1.17.1
scverse-misc==0.1.7
seaborn==0.13.2
Send2Trash==2.1.0
sentry-sdk==2.62.0
session-info2==0.4.2
setuptools==81.0.0
shellingham==1.5.4
six==1.16.0
smmap==5.0.3
soupsieve==2.8.4
stack-data==0.6.3
statsmodels==0.15.0
sympy==1.14.0
tabulate==0.10.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorrt @ file:///workspace/TensorRT-11.0.0.114/python/tensorrt-11.0.0.114-cp312-none-linux_aarch64.whl#sha256=b02d6572dbd05c0aa215a5772816eee18c17d3f3324c7094d8b932d3e92f4689
terminado==0.18.1
threadpoolctl==3.7.0
tinycss2==1.5.1
tokenizers==0.23.1
torch @ file:///opt/transfer/torch-2.13.0a0%2B8145d630e8.nv26.6.54250401-cp312-cp312-linux_aarch64.whl#sha256=8a7efd12b08642a772ec7fd4dc2ab8ea88dc964323e0aca3a0f329a662193cfa
torch_tensorrt @ file:///opt/pytorch/torch_tensorrt/dist/torch_tensorrt-2.13.0a0-cp312-cp312-linux_aarch64.whl#sha256=2f42d582051a1d3948e74c4f3dc8447599594d6a96fae38d714ab84e9a6b73b7
torchao @ file:///opt/pytorch/ao
torchdata==0.11.0
torchtitan @ file:///opt/pytorch/torchtitan
torchvision @ file:///opt/transfer/torchvision-0.27.0a0%2B499ca510.nv26.6.54250401-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=8c9341542e4b370b041bb8a0be1722b202abcf07fba94d762b2835e114fee7a5
tornado==6.5.7
tqdm==4.68.2
traitlets==5.15.1
transformer_engine @ file:///workspace/TransformerEngine
triton @ file:///tmp/dist/triton-3.7.0%2Bgitb7fa781f.nv26.6-cp312-cp312-linux_aarch64.whl#sha256=ac76a8804e3c7883c603679839f5db8b694366d50c8d2ab929fe309c63261827
triton_kernels @ file:///tmp/dist/triton_kernels-1.0.0%2Bgitb7fa781f.nv26.6-py3-none-any.whl#sha256=517938a42b240c5ad4bf5dce7720512e29cd408b2d58e2367f9f5448bdf13737
typeguard==4.5.2
typer==0.25.1
typing-inspection==0.4.2
typing_extensions==4.15.0
tyro==1.0.13
tzdata==2026.2
umap-learn==0.5.12
uri-template==1.3.0
urllib3==2.7.0
wandb==0.27.2
wcwidth==0.8.1
webcolors==25.10.0
webencodings==0.5.1
websocket-client==1.9.0
Werkzeug==3.1.8
wheel==0.47.0
wrapt==2.5.1
xxhash==3.7.0
yarl==1.24.2
zarr==3.4.1
zipp==4.1.0
```
<!-- END_RESTORE_FILE -->


</details>

## file-b414bb9f8efe

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/resources.csv` · SHA256: `b1990bb8267312666e747961c53f828975c061f175fdc0fc9a0bbf97aae63131`

<details>
<summary>resources.csv — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/resources.csv", "sha256": "b1990bb8267312666e747961c53f828975c061f175fdc0fc9a0bbf97aae63131", "encoding": "utf-8", "added_final_newline": false} -->
```csv
elapsed_seconds,process_rss_bytes,cuda_allocated_bytes,cuda_reserved_bytes,system_available_bytes
1.8719583749771118e-06,1234788352,205353984,211812352,119785013248
0.2017278210259974,1589026816,295688192,301989888,119260790784
0.4029984420631081,1882333184,350140928,390070272,118927024128
0.6041085021570325,1882394624,325041664,505413632,118811369472
0.8051842590793967,1882423296,349185536,505413632,118811201536
1.0060337909962982,1882443776,351057920,505413632,118811455488
1.2073311789426953,1882451968,352457216,505413632,118811324416
1.4084579111076891,1882460160,359545856,505413632,118811324416
1.6097157001495361,1882472448,351588864,505413632,118811324416
1.8113551039714366,1882484736,350512640,505413632,118811226112
2.012162795988843,1882492928,352960512,505413632,118811226112
2.21319808694534,1882501120,354971648,505413632,118812442624
2.413953874958679,1882513408,345900544,555745280,118761455616
2.6153675511013716,1882529792,347914240,608174080,118708699136
2.816825947025791,1882537984,325041664,608174080,118634311680
3.018252630950883,1882550272,310361600,608174080,118634311680
3.2197658109944314,1882562560,312883712,608174080,118634311680
3.4210842540487647,1882570752,355189760,608174080,118634311680
3.6228507950436324,1882587136,316926464,608174080,118634311680
3.824213925981894,1882619904,305179648,608174080,118634311680
4.025639105122536,1882648576,321266688,608174080,118636380160
4.22703202906996,1882669056,308956160,608174080,118635925504
4.42840332002379,1877557248,299722752,608174080,118635925504
4.629759652074426,1879703552,316704768,608174080,118635925504
4.830609070137143,1879937024,296197632,608174080,118635868160
5.03168028104119,1879605248,292995072,608174080,118635868160
5.233121620025486,1879658496,299166720,608174080,118635868160
5.434410670073703,1881849856,356917248,608174080,118636617728
5.635850778082386,1881878528,351065088,662700032,118581571584
5.837179539958015,1881894912,362633728,662700032,118581440512
6.038593807024881,1881919488,359809024,662700032,118583250944
6.24004938500002,1881952256,353661440,662700032,118583250944
6.441827221075073,1881972736,359373824,662700032,118689361920
6.643097406951711,1881972736,339612672,662700032,118688894976
6.845043738139793,1881972736,333974528,662700032,118704902144
7.04637120408006,1881972736,335472128,662700032,118705192960
7.247551949927583,1881976832,337766400,662700032,118705192960
7.449252824997529,1881976832,333074432,662700032,118705405952
7.6505220991093665,1881976832,349735424,662700032,118704947200
7.851741148158908,1881985024,342045696,662700032,118704308224
8.052874582121149,1881993216,357192704,662700032,118713987072
8.254358144011348,1882001408,349643264,662700032,118712696832
8.455186857143417,1882046464,347140096,662700032,118712926208
8.65656697913073,1882058752,347250688,662700032,118727417856
8.857857820112258,1882066944,344812032,662700032,118741463040
9.058976548956707,1882075136,373502464,662700032,118740926464
9.259919981937855,1882087424,349333504,662700032,118744236032
9.460717175155878,1882091520,343175168,662700032,118743318528
9.661653680028394,1882099712,350948352,662700032,118741917696
9.863167769042775,1882107904,341508096,662700032,118749790208
10.06394981010817,1882148864,354779648,662700032,118751830016
10.264692186145112,1882161152,351965184,662700032,118752276480
10.465657650027424,1882165248,345817600,662700032,118752276480
10.667542875977233,1882243072,346134528,662700032,118752276480
10.868683508131653,1882288128,336438784,662700032,118763192320
11.069775037001818,1882341376,283133952,662700032,118762930176
11.271064101019874,1882394624,347026432,662700032,118764961792
11.471912461100146,1882415104,313373696,662700032,118763585536
11.673182758037001,1882435584,345227264,662700032,118764765184
11.874526974046603,1882472448,344897024,662700032,118773760000
12.07556875795126,1882521600,287331840,662700032,118791471104
12.27681775810197,1882562560,297158656,662700032,118791340032
12.477434454020113,1882595328,286138880,662700032,118766784512
12.678275901125744,1882619904,300354048,662700032,118758793216
12.87975338101387,1882644480,294201856,662700032,118752444416
13.080516109010205,1882660864,287329280,662700032,118724816896
13.281533428002149,1882681344,296096768,662700032,118716379136
13.482300091069192,1882697728,272328704,662700032,118710722560
13.683745315065607,1882714112,295916032,662700032,118710726656
13.884764186106622,1882730496,287510016,662700032,118710464512
14.086073633981869,1882746880,292038144,662700032,118710235136
14.287149001145735,1880236032,287692288,662700032,118710005760
14.487842143978924,1880272896,289809920,662700032,118690136064
14.688757399097085,1880297472,307665408,662700032,118664245248
14.890021918108687,1880326144,288853504,662700032,118666551296
15.090764180058613,1880338432,294139904,662700032,118639484928
15.291836651042104,1880371200,284123648,662700032,118639063040
15.476342637091875,1880850432,241005568,662700032,118605869056
```
<!-- END_RESTORE_FILE -->


</details>

## file-9cd02c9a4f70

원래 경로: `scgpt/results/finetuned_test_20261023_034345_99523/summary.json` · SHA256: `a0aed1f9174799fc2dac7437418024793e20188aea1685025610612870b1fa89`

<details>
<summary>summary.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "scgpt/results/finetuned_test_20261023_034345_99523/summary.json", "sha256": "a0aed1f9174799fc2dac7437418024793e20188aea1685025610612870b1fa89", "encoding": "utf-8", "added_final_newline": false} -->
```json
{
  "status": "complete",
  "task": "fine-tuned cell-type predictions",
  "tokenization_seconds": 0.4721863870508969,
  "warmup_steps": 0,
  "weights_updated": false,
  "model_bundle": "/workspace/scgpt/models/finetuned/finetune_20261023_033701_500560/epoch_001",
  "memory_notes": "CUDA peaks are process-local PyTorch allocator counters, not total device usage. RSS is sampled every 0.2 s, not an exact peak. Spark uses shared memory: do not add RSS and CUDA as independent physical memory pools.",
  "cells_total": 4218,
  "cells_evaluated_known_labels": 4213,
  "cells_without_known_training_label": 5,
  "forward_seconds": 15.474318906897679,
  "cells_per_second": 272.5806560778469,
  "timing_scope": "cached input batching, CPU-to-GPU, encoder/classifier, GPU-to-CPU logits; excludes tokenization and file saving",
  "gpu_peak_allocated_gib": 0.3811001777648926,
  "gpu_peak_reserved_gib": 0.6171875,
  "python_process_rss_peak_observed_gib": 1.7534446716308594,
  "loss": 0.23071691393852234,
  "accuracy": 0.9380488962734393,
  "macro_f1_all_training_classes": 0.5349519490076795,
  "macro_f1_present_true_classes": 0.6954375337099833,
  "weighted_f1": 0.9311712408040055
}
```
<!-- END_RESTORE_FILE -->


</details>
