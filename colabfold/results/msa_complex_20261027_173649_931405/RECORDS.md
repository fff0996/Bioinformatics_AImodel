# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-4321b4b89e25

원래 경로: `colabfold/results/msa_complex_20261027_173649_931405/command.json` · SHA256: `ca377a53d691f51d0df0c294eae298b2993d8093e619de949b8c38f1cd8a70bf`

<details>
<summary>command.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_complex_20261027_173649_931405/command.json", "sha256": "ca377a53d691f51d0df0c294eae298b2993d8093e619de949b8c38f1cd8a70bf", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  "sg",
  "docker",
  "-c",
  "docker run --rm --network bridge --user 1000:1000 -e JAX_PLATFORMS=cpu -e SSL_CERT_FILE=/etc/ssl/certs/host-ca.crt -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/host-ca.crt -v /etc/ssl/certs/ca-certificates.crt:/etc/ssl/certs/host-ca.crt:ro -v /home/hyein/ai-benchmark/colabfold:/work -v /home/hyein/ai-benchmark/colabfold/cache/colabfold:/cache:ro ghcr.io/sokrypton/colabfold:1.6.2-cuda13 colabfold_batch /work/data/complex.fasta /work/results/msa_complex_20261027_173649_931405/msa --msa-only --model-type alphafold2_multimer_v3 --data /cache"
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-1a30bdef36e0

원래 경로: `colabfold/results/msa_complex_20261027_173649_931405/exit_code.json` · SHA256: `7ab25aaeedd36ca96f59eb6eec343444fbf5435fcc934eca2a640c7b091c26df`

<details>
<summary>exit_code.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/msa_complex_20261027_173649_931405/exit_code.json", "sha256": "7ab25aaeedd36ca96f59eb6eec343444fbf5435fcc934eca2a640c7b091c26df", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "exit_code": 0,
  "start_utc": "2026-10-27T08:36:49.931603+00:00",
  "end_utc": "2026-10-27T08:37:10.118975+00:00"
}
```
<!-- END_RESTORE_FILE -->


</details>
