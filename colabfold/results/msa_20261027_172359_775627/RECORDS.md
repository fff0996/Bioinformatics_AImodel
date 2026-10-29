# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-7db6d4397295

원래 경로: `colabfold/results/msa_20261027_172359_775627/command.json` · SHA256: `380b7ba942ec309bc1be2b1d7aef81ee6563075845936312da6ba2df64d297e5`

<details>
<summary>command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_20261027_172359_775627/command.json", "sha256": "380b7ba942ec309bc1be2b1d7aef81ee6563075845936312da6ba2df64d297e5", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --gpus all --user 1000:1000 -e JAX_PLATFORMS=cuda -e SSL_CERT_FILE=/etc/ssl/certs/host-ca.crt -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/host-ca.crt -v /etc/ssl/certs/ca-certificates.crt:/etc/ssl/certs/host-ca.crt:ro -v /home/hyein/ai-benchmark/colabfold:/work -v /home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 colabfold_batch /work/data/queries.fasta /work/results/msa_20261027_172359_775627/msa --msa-only --data /cache --sort-queries-by length"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-683b3b128ce1

원래 경로: `colabfold/results/msa_20261027_172359_775627/exit_code.json` · SHA256: `a360ba8b2e28b6f42ab27f5d8d253a66826a0bed401570a7397eb18e614fc7c9`

<details>
<summary>exit_code.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_20261027_172359_775627/exit_code.json", "sha256": "a360ba8b2e28b6f42ab27f5d8d253a66826a0bed401570a7397eb18e614fc7c9", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "exit_code": 0,
  "start_utc": "2026-10-27T08:23:59.775980+00:00",
  "end_utc": "2026-10-27T08:24:56.796060+00:00"
}
```
<!-- END_RESTORE_FILE -->


</details>
