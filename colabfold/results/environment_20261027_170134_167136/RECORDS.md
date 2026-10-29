# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-a4815a33eb6a

원래 경로: `colabfold/results/environment_20261027_170134_167136/environment_check.json` · SHA256: `240685315786d26d7d38745ad4bc98a13dfc74de09aef782edf105307896952e`

<details>
<summary>environment_check.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/results/environment_20261027_170134_167136/environment_check.json", "sha256": "240685315786d26d7d38745ad4bc98a13dfc74de09aef782edf105307896952e", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "command": [
      "id"
    ],
    "exit_code": 0,
    "stdout": "uid=1000(hyein) gid=1000(hyein) groups=1000(hyein),4(adm),27(sudo),29(audio),30(dip),46(plugdev),100(users),122(lpadmin)\n",
    "stderr": ""
  },
  {
    "command": [
      "stat",
      "-c",
      "%A %U %G %u %g %n",
      "/var/run/docker.sock"
    ],
    "exit_code": 0,
    "stdout": "srw-rw---- root docker 0 988 /var/run/docker.sock\n",
    "stderr": ""
  },
  {
    "command": [
      "docker",
      "ps",
      "-a"
    ],
    "exit_code": 1,
    "stdout": "",
    "stderr": "permission denied while trying to connect to the docker API at unix:///var/run/docker.sock\n"
  },
  {
    "command": [
      "docker",
      "images"
    ],
    "exit_code": 1,
    "stdout": "",
    "stderr": "permission denied while trying to connect to the docker API at unix:///var/run/docker.sock\n"
  },
  {
    "command": [
      "docker",
      "inspect",
      "scgpt-spark"
    ],
    "exit_code": 1,
    "stdout": "[]\n",
    "stderr": "permission denied while trying to connect to the docker API at unix:///var/run/docker.sock\n"
  }
]
```
<!-- END_RESTORE_FILE -->


</details>
