# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-a522ccb739d9

원래 경로: `docs/LOCAL_ARTIFACT_LINKS.json` · SHA256: `16ec9e55b161e01491406e1ed7041ca1c8e8db17b6a3f6ee52f19393b2bd6a56`

<details>
<summary>LOCAL_ARTIFACT_LINKS.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "docs/LOCAL_ARTIFACT_LINKS.json", "sha256": "16ec9e55b161e01491406e1ed7041ca1c8e8db17b6a3f6ee52f19393b2bd6a56", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "document": "ntv2/reports/NTv2.md",
    "target": "../results/ntv2_20261027_201332_251243/benchmark.log"
  },
  {
    "document": "rnafm/reports/RNAFM.md",
    "target": "../results/rnafm_20261027_204440_142840/benchmark.log"
  },
  {
    "document": "rnafm/reports/RNAFM.md",
    "target": "../data/inputs.json"
  },
  {
    "document": "dnabert2/reports/DNABERT2.md",
    "target": "../results/supplement_20261027_195301_706540/input_examples.json"
  },
  {
    "document": "openspliceai/reports/OpenSpliceAI.md",
    "target": "../results/openspliceai_20261027_205554_995618/benchmark.log"
  },
  {
    "document": "openspliceai/reports/OpenSpliceAI.md",
    "target": "../data/inputs.json"
  },
  {
    "document": "scgpt/reports/scGPT_actual_300k.md",
    "target": "../data_external/tabula_sapiens/prepared_300k_seed42/input_audit.json"
  },
  {
    "document": "scgpt/reports/scGPT_memory_batch.md",
    "target": "../results/memory_batch_20261027_213310_982032/run.log"
  },
  {
    "document": "scgpt/reports/scGPT_memory_batch.md",
    "target": "../results/memory_batch_refine_20261027_214026_294469/run.log"
  },
  {
    "document": "clair3/reports/Clair3.md",
    "target": "../results/clair3_20261027_203339_768667/benchmark.log"
  },
  {
    "document": "cellpose/reports/Cellpose.md",
    "target": "../results/cellpose_20261027_193331_947064/run.log"
  },
  {
    "document": "cellpose/reports/Cellpose_cellpose_20261027_193331_947064.md",
    "target": "../results/cellpose_20261027_193331_947064/run.log"
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-e2a43c9d43e3

원래 경로: `docs/PUBLICATION_MANIFEST.json` · SHA256: `1a22412f5eb482add5d4644675ece41274072a1200630648b7695f79e3bdf2de`

<details>
<summary>PUBLICATION_MANIFEST.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "docs/PUBLICATION_MANIFEST.json", "sha256": "1a22412f5eb482add5d4644675ece41274072a1200630648b7695f79e3bdf2de", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "source_workspace": "/home/hyein/ai-benchmark",
  "original_dataset_and_weights": "not included",
  "selected_files": [
    {
      "path": "RESULTS.md",
      "bytes": 20199,
      "source_sha256": "5349296467c0aa439514a5e270f46d87a3a34d5c777c7ad3b251fdbd47582c0b"
    },
    {
      "path": "FOLLOWUP_RESULTS.md",
      "bytes": 4316,
      "source_sha256": "7e893da0670e75f1838ddadbd774e24d01d6de894c62cb0800d4dd9b102de01e"
    },
    {
      "path": "MEMORY_RESULTS.md",
      "bytes": 5270,
      "source_sha256": "b7c5a2155de3bcb7ad6488665961a85df356b7bca23c1b7e4c9dc0c9d0742133"
    },
    {
      "path": "STATUS.md",
      "bytes": 14845,
      "source_sha256": "3ec15bceb1cde3240bd874960d27ff46eb65f0f80fd4f73535c80e12161c39ff"
    },
    {
      "path": "BATCH_APPLICABILITY.md",
      "bytes": 2359,
      "source_sha256": "5c087966747d3e826c95aee9ce7d919bd44002214683905456636bdc29b05e00"
    },
    {
      "path": "FOLLOWUP_QUEUE.json",
      "bytes": 1773,
      "source_sha256": "ae6e95b2ec45545f7fd156af9c225a30dd958cc014baee160c92c41003baf092"
    },
    {
      "path": "scripts/build_followup_overview.py",
      "bytes": 5554,
      "source_sha256": "758bc4fe7956cd039e5bf1577f6fa641306bb72bbbcf8e4ae279e1d23cf26425"
    },
    {
      "path": "scripts/clair3_batch_launch.py",
      "bytes": 3282,
      "source_sha256": "b8724c07ea44c668b75692f6f11f1c62090676709f754db4e207018948f8e0f9"
    },
    {
      "path": "scripts/launch_model.py",
      "bytes": 1350,
      "source_sha256": "bfc0e857b4135402be2fd0335265b13299bf5234449a467e9fd2d6ff78f6fdeb"
    },
    {
      "path": "scripts/00_prepare_scgpt_inputs.sh",
      "bytes": 10476,
      "source_sha256": "8b07ce164fc598f2319e368bb70c855640ffaf52f8a2f1fc3f5628ef38a1d7c9"
    },
    {
      "path": "scripts/gpu_support.py",
      "bytes": 2858,
      "source_sha256": "349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2"
    },
    {
      "path": "scripts/af2_vmap_launch.py",
      "bytes": 2829,
      "source_sha256": "68c706f5a2fcb6da7c57452936689f1e79ca218d52e2020548915db603480c01"
    },
    {
      "path": "scripts/two_batch_report.py",
      "bytes": 8497,
      "source_sha256": "a4fa6195788965f1fd6a0599718a24fcc35feebdc681fa10f26bcab070c22277"
    },
    {
      "path": "scripts/two_remaining_launch.py",
      "bytes": 1052,
      "source_sha256": "a93cb7bf32db3e72a391b9a3f420c4cd040776644a5c1c25850e0ee5b2d970d2"
    },
    {
      "path": "scripts/build_results_index.py",
      "bytes": 12505,
      "source_sha256": "6cd748826101815906a53f5ec4775e62b1906afaf3c121d493cb767ee3b530e1"
    },
    {
      "path": "scripts/fetch_hf_files.py",
      "bytes": 2439,
      "source_sha256": "9b3766589dc4ac5d21f8f97cc7c18767209bd0101365a9f9d08f7e2b65709489"
    },
    {
      "path": "scripts/af2_feature_capture_launch.py",
      "bytes": 919,
      "source_sha256": "2d62f809b99dd4b97b04835333d35d25b3eb86c98d7ee514a7df20f422b6e6da"
    },
    {
      "path": "scripts/remaining_followups/launch.py",
      "bytes": 7883,
      "source_sha256": "84ea6bdb09c57582e25bfc2adf9da1ba72bcc66836af78386532e5869488af63"
    },
    {
      "path": "scripts/remaining_followups/report.py",
      "bytes": 5292,
      "source_sha256": "8b67b28dbc254521328fff13f7ad3b5abcef84f687eff838616e741ab957d700"
    },
    {
      "path": "scripts/remaining_followups/worker.py",
      "bytes": 6920,
      "source_sha256": "e47c0fd347e235ec6529c9052da478986e5515c1d9b72dad653c2ac421ddff74"
    },
    {
      "path": "esm2/scripts/measure_loading.py",
      "bytes": 1578,
      "source_sha256": "1c021da223b492c31233ca4c23267eb7ffc4a61b5616fcd0257bf8477e53d5a1"
    },
    {
      "path": "esm2/scripts/benchmark.py",
      "bytes": 15891,
      "source_sha256": "20863f34231101de503f46b8b5ef61c1ea1c775eb0ff7ebb49320c5548f9e94d"
    },
    {
      "path": "esm2/scripts/report.py",
      "bytes": 23131,
      "source_sha256": "13e34868ad22d1e2c0e83c3fb84e93c10ed055452992983f57b0bb73a2d44b0b"
    },
    {
      "path": "esm2/scripts/predict.py",
      "bytes": 2224,
      "source_sha256": "1ff266dbfc0eebeb26c4e8f9b6995529c8c75b59b25c43c70e1ba10bd19bda9b"
    },
    {
      "path": "esm2/scripts/supplement.py",
      "bytes": 8873,
      "source_sha256": "ceac1a1593c7aaa06e2ded3fff60a8b930858d9c5ef8a4508baec89a324ae266"
    },
    {
      "path": "esm2/scripts/prepare.py",
      "bytes": 1851,
      "source_sha256": "29f09e4a7d6cc687bbb14ddd3a2fd8cb8f7cfec87e2c61d682b3b57238c6a154"
    },
    {
      "path": "esm2/scripts/batch_followup/launch.py",
      "bytes": 5161,
      "source_sha256": "894b721991490d3b66a0c80b641a2ddad27fb2109efabbe80d7b2dd4d9f9885b"
    },
    {
      "path": "esm2/scripts/batch_followup/report.py",
      "bytes": 5693,
      "source_sha256": "66216a98513911eb4413bdc08aad2dc418908c2fe3e9a03df2754e7ac5bb45a6"
    },
    {
      "path": "esm2/scripts/batch_followup/worker.py",
      "bytes": 3958,
      "source_sha256": "02bf8c3072ca143a9b0765278e62cf1380434c78a240008cf214e0a59ab02f89"
    },
    {
      "path": "esm2/reports/comparison_20261027_164300_125163.csv",
      "bytes": 3689,
      "source_sha256": "826fb95a4d445c8b8b103d7c4cdb937eb42bd0f2ae409418e45c8e6b4c8e9e27"
    },
    {
      "path": "esm2/reports/batch_memory_comparison.csv",
      "bytes": 3334,
      "source_sha256": "256fe38ddb854c02bbda06ba5d2cc30ad36f05322c59b51a4094a899452d0d7c"
    },
    {
      "path": "esm2/reports/ESM2_batch_memory.md",
      "bytes": 4845,
      "source_sha256": "a134131429c078a7510619329a840203be9a741bb01c317e6b1e560c1ee6ec01"
    },
    {
      "path": "esm2/reports/artifacts_20261027_164300_125163.json",
      "bytes": 19414,
      "source_sha256": "7529bbb3eaa63b4bd4d7a4506f800dc88bece0458db0e1f15d301020f0967250"
    },
    {
      "path": "esm2/reports/ESM2_20261027_164300_125163.md",
      "bytes": 19614,
      "source_sha256": "57bed5e8853b018a10aa7bd1f2bdb608065160032d1559d79851d26a227bb009"
    },
    {
      "path": "esm2/results/20261027_164300_125163/input_report.json",
      "bytes": 2154,
      "source_sha256": "1e54d6180f5ead2157742cb5bb7f2b209828d4180c4c3c3bc419cbddde54279c"
    },
    {
      "path": "esm2/results/20261027_164300_125163/test_original.json",
      "bytes": 8477,
      "source_sha256": "d49d26a5edf8a03e0de7bf8936ef26cced1c4b26c3bd02e9151314a27701afcb"
    },
    {
      "path": "esm2/results/20261027_164300_125163/count_160_metrics.json",
      "bytes": 807,
      "source_sha256": "538b6d3e71331c078be2215c4d834458a888f7d617c547bc90bd52ddee6bd337"
    },
    {
      "path": "esm2/results/20261027_164300_125163/environment.json",
      "bytes": 2408,
      "source_sha256": "674c9c318509d61bf417ad6423fbc7c4873b24fbab2f9e4befd70aed032ecad9"
    },
    {
      "path": "esm2/results/20261027_164300_125163/baseline_validation.json",
      "bytes": 4334,
      "source_sha256": "8e47602d7eecc5f4ce5618e6adbc6515eaf8fe4ca1a8ae91196bddff4a4f04da"
    },
    {
      "path": "esm2/results/20261027_164300_125163/summary.json",
      "bytes": 8788,
      "source_sha256": "a50738734670647de052692c129ff7a92832af344d49fe878df5f060e50f68f5"
    },
    {
      "path": "esm2/results/20261027_164300_125163/pip_freeze.txt",
      "bytes": 1678,
      "source_sha256": "e98aad249434c5f10cb3f48cd2fa6ddbc996d34e3f4c179e0c60eff1716fa803"
    },
    {
      "path": "esm2/results/20261027_164300_125163/length_257_512_metrics.json",
      "bytes": 794,
      "source_sha256": "e6558a0e568f427172305704dbe65188fff483e63317310c90f2560edfe8960e"
    },
    {
      "path": "esm2/results/20261027_164300_125163/resources.csv",
      "bytes": 18236,
      "source_sha256": "27621acc770e18c283bb94b6f2ee01352506f4f8a9da0ecefa6392c5560682b6"
    },
    {
      "path": "esm2/results/20261027_164300_125163/process_recovery.json",
      "bytes": 463,
      "source_sha256": "94a10a85722404e73a02959723246f4e17e31c6f91ee13ddc39896513f069160"
    },
    {
      "path": "esm2/results/20261027_164300_125163/length_129_256_metrics.json",
      "bytes": 794,
      "source_sha256": "1370778d7050d07709a799f18158c096ba5088edf8f19a9b49deb4f8eeec2023"
    },
    {
      "path": "esm2/results/20261027_164300_125163/sources.json",
      "bytes": 6137,
      "source_sha256": "e7ebd5ffacbf41ccbd2decc584223c4ce9cd442ef1feef7559022c9c52a80f71"
    },
    {
      "path": "esm2/results/20261027_164300_125163/count_96_metrics.json",
      "bytes": 810,
      "source_sha256": "70f55b7a2b275c736b47892cf2b9ff416883b19f2cf4f4d7fee5bf8cff6a619f"
    },
    {
      "path": "esm2/results/20261027_164300_125163/count_32_metrics.json",
      "bytes": 805,
      "source_sha256": "a21367a2e07e029e80f135bef755c4bb3946edf1edd5443ca0a6ee3431495fcf"
    },
    {
      "path": "esm2/results/20261027_164300_125163/smoke.json",
      "bytes": 150,
      "source_sha256": "2964a6f77b001ee3873541c3bee1b2f0adca52e939d6cf1ad429fa0d967515f7"
    },
    {
      "path": "esm2/results/20261027_164300_125163/training_metrics.json",
      "bytes": 1589,
      "source_sha256": "1caac58d7aeb800795a255c5b0288ed027fe1a88b9112002d11023fbb172de95"
    },
    {
      "path": "esm2/results/20261027_164300_125163/length_513_1022_metrics.json",
      "bytes": 813,
      "source_sha256": "40e166d9a4ec29cc0fff5dbe2c40db2d971c3475abad0b94d4cdf14f990d38e7"
    },
    {
      "path": "esm2/results/20261027_164300_125163/validation.json",
      "bytes": 4331,
      "source_sha256": "5503291ac5b4ca8eef57185637bd77c3160dc89e2836cfa47fe38477f5697d42"
    },
    {
      "path": "esm2/results/20261027_164300_125163/test_reloaded.json",
      "bytes": 8477,
      "source_sha256": "d49d26a5edf8a03e0de7bf8936ef26cced1c4b26c3bd02e9151314a27701afcb"
    },
    {
      "path": "esm2/results/20261027_164300_125163/length_50_128_metrics.json",
      "bytes": 814,
      "source_sha256": "c7a85039494889077d7ae0eaecebfa4251c3811b76e8d2fd2ecd70cbda1cd323"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit.json",
      "bytes": 89,
      "source_sha256": "d6bada1aa960e75a7030f14a2145316ea13e8bec339604e82f3e0d471a184e7f"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch128.json",
      "bytes": 73,
      "source_sha256": "4c2a971109230fdc3f67af7de9278d7f09455196ef5d2fc1a248a16c3c254abf"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch2.json",
      "bytes": 207,
      "source_sha256": "377b8b8f0a30a72c02d513eda4e92f06eb3fae6acae497d10e86b5263a4de6d3"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch256.json",
      "bytes": 73,
      "source_sha256": "d82db0796cad926c8e3e56ced185fd863a84b330da15f8d13da4da7d2399fa3a"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch1.json",
      "bytes": 73,
      "source_sha256": "3727a0186c4fb569e9c6d9d24faba35d7a216e172a9eea510de6ac1a7e6df162"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch160.json",
      "bytes": 209,
      "source_sha256": "8787007fdf02b7fdd7dd16a3c24cbc9bf5b5cbcde3498ecdeac674c712588fbe"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch224.json",
      "bytes": 73,
      "source_sha256": "9d98565039d3e2a1c94b126aba189019b9b4c9f581bae624aba00da216b4c338"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/summary.json",
      "bytes": 23625,
      "source_sha256": "672854a258afca4b7b6cbe77803333ec5b17e6a549e1a7505f2be0fbc3e1193c"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch192.json",
      "bytes": 209,
      "source_sha256": "2c25bbad309678d22a78e35f1b1f21d1a4305cad17c57acd11460be193aa97b0"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch160.json",
      "bytes": 73,
      "source_sha256": "9206af581c3f5520f082758f1037e822ca209a0bbf86aafe04d4c00c79dd3215"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch224.json",
      "bytes": 209,
      "source_sha256": "19604ae1e252f678cc3db60078ed7077d41d0ac36fe80b4ab1d55a9461077c38"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch288.json",
      "bytes": 209,
      "source_sha256": "83f8934843bc436360476cc945ee4e3cd5bc504fbe08f242474b0e9dfb34a7e6"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch32.json",
      "bytes": 73,
      "source_sha256": "8f29a73748a7fa1f3ddf7ccf131ecdfa1d42c68bc11d3f11d1c068682a4212cc"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch64.json",
      "bytes": 71,
      "source_sha256": "0ee2d455d61d723e933f12090f4e55c186b79952ebb495b8a1d98b9717e11b98"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch2.json",
      "bytes": 72,
      "source_sha256": "b6d0d8b32dff560d84dddd59d47798a2b3ff11c82db1ee5270be829452a8c6bd"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch128.json",
      "bytes": 209,
      "source_sha256": "30a90523cbaf2d9a89ccbf69cec3b295e9e72dc4c6674427d9f10cf033753e7d"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch16.json",
      "bytes": 208,
      "source_sha256": "8a4c5619e0b6d036b70f9f2b21e73320004b9e0d7789625412d8191ee1c92453"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch32.json",
      "bytes": 208,
      "source_sha256": "70c8a26886f3d241a99ca129bf301b51f94349660f4590add70abdd5024a3c73"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch4.json",
      "bytes": 72,
      "source_sha256": "38d071e5436cf301073e6a888610f6fb00974d13ee000e086c39fa352b7bbeba"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch16.json",
      "bytes": 73,
      "source_sha256": "4038f0e87428eb7e60e532ebb599126d3d691f0263ca61e238fed4e262ee790b"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch384.json",
      "bytes": 209,
      "source_sha256": "f92e6c3b61a974496b932187a33472a45dd85e0366ba1bab1adc5f5e6642c45b"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch320.json",
      "bytes": 72,
      "source_sha256": "cef342951302998885e73caf90c6af7e7be579dee20cf97d8c1358f5b58e86dd"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch192.json",
      "bytes": 73,
      "source_sha256": "371684fc3d7a7eef3b2ea31243535ad46df28e02225b273809ad3bba08655448"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch8.json",
      "bytes": 73,
      "source_sha256": "826ca275b8befad39f93279cbe892766edb3b3dd2ffe9ecc61e7fa6d00000736"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch384.json",
      "bytes": 72,
      "source_sha256": "c4ec0e118ea6e18ef680cb20d983672e5efdb91df3d9a8763a07f5be9f3115f5"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch64.json",
      "bytes": 208,
      "source_sha256": "39ab190a390436b76f2c787243a9d33cbe990bad1c94ecbeaab5a68905be14cb"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch4.json",
      "bytes": 207,
      "source_sha256": "e255a052d455c1fe6c454a73f231b444290c12282e5bbca8034a6352b5266b4d"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch288.json",
      "bytes": 73,
      "source_sha256": "4d28adcb735cda17c44c311c87853887413c8c5768ce6fe406f684395568fa76"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch8.json",
      "bytes": 207,
      "source_sha256": "ecf2e756275d0177bb9aa5d6d79e1204552d537e4b7519b55dd0a974560f20fa"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/exit_batch96.json",
      "bytes": 72,
      "source_sha256": "cab61000de387eef2a526b9dfe2c466ced17cbb6ec26b04e2026ab853776b507"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch96.json",
      "bytes": 208,
      "source_sha256": "5329a5ddee6993c627c8683700fbc228792703c0a51012ff0a9e81c912aa610b"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch320.json",
      "bytes": 209,
      "source_sha256": "b2aa6570eb08f39501f1b294a66790b5ce42ef0c4f35e8e29bafc89854e8a448"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch256.json",
      "bytes": 209,
      "source_sha256": "d8a6f691138fca1c508d10fcb2d2dbfef8664b89ec2608d25c97adabbcc7f82c"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/command_batch1.json",
      "bytes": 207,
      "source_sha256": "d02372c7d22821361b4496efc2b2f01993300bcea790ca4eba9480a0e3722993"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_32/summary.json",
      "bytes": 2559,
      "source_sha256": "ef3a805a086b70a7d08e5fbe028e1365eed247c9886c74fd3628b4a237c1cde7"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_32/resources.csv",
      "bytes": 9026,
      "source_sha256": "95d9ec3df6ca655b2511f68dd2f0d3e15371331e995fcb8c2fb240455c2067ac"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_8/summary.json",
      "bytes": 2548,
      "source_sha256": "8d3cf94969b806adbfe9a1ced747c2d0b0388bc4f882028ccc2e0091ae8d9a9e"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_8/resources.csv",
      "bytes": 5804,
      "source_sha256": "07c28412f6824c1eba15d4a9034c71e5423d3cebe8c3a199e9fd2e8fd5177890"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_320/summary.json",
      "bytes": 2563,
      "source_sha256": "eea7c8ca6242bedad98537a998700402439935ac2795c146fa71421093192450"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_320/resources.csv",
      "bytes": 74331,
      "source_sha256": "74cc2f846fa8fe19fc49cb862df32f8b548c6fe4bef69786b346a9ae60da442f"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_16/summary.json",
      "bytes": 2555,
      "source_sha256": "4f1dab6b53dee83d0c52f1a31b689bce79e372929c3802e4c35113b224f141bc"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_16/resources.csv",
      "bytes": 6499,
      "source_sha256": "c2f8d58111bc96e3706894cbc4ded528015f522689f97a6f2527ae67bdc72cd1"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_64/summary.json",
      "bytes": 2563,
      "source_sha256": "08176d2c50dd7c2cdd7dfbb2f10b3133c6d6c54e0f1114c25e9afd1e3966c0e7"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_64/resources.csv",
      "bytes": 17518,
      "source_sha256": "1507a3c0e2983050512ffed20f85174d8c757972152c260ec3f2a14d65ebde88"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_256/summary.json",
      "bytes": 2568,
      "source_sha256": "24b99ab8eadb36e3ed21e07c9d962ca046a603dcefcb75858d632757f24b9990"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_256/resources.csv",
      "bytes": 60228,
      "source_sha256": "76d1836613ff1e2d82c48ca57b2bebfb2feba0d2efdbb24e03306d52af3b8755"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_288/summary.json",
      "bytes": 2575,
      "source_sha256": "cecf2782cf8f350de354e72f673abc0da20cc1ef94eb1f856f1516ac743b7c18"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_288/resources.csv",
      "bytes": 67772,
      "source_sha256": "d3d0fb3559dadd3264561c872e4d70e7b2502272ec96b164729967ba8311f0c1"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_2/summary.json",
      "bytes": 2529,
      "source_sha256": "7f731027aac1cc79d4453d361e8268d46977f3d7cd2feb1702f448d65facd09b"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_2/resources.csv",
      "bytes": 5349,
      "source_sha256": "df59b8e6e4f5c4fda5fa9ed8844bcb55e745df9adc9c130c0e9c139ac6fc2704"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_192/summary.json",
      "bytes": 2561,
      "source_sha256": "847fcaa6e8c36cf302746bb34813d01f8c870e15bf58d95598bef7400a6dce36"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_192/resources.csv",
      "bytes": 49989,
      "source_sha256": "dee2e3a55a0721975918308aab0fed7323ec08c7e1da2f06bd9b9e5821e004c0"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_160/summary.json",
      "bytes": 2563,
      "source_sha256": "7480cd7979f857f61aaaaa1d7af53f4f158e126c27643a4fcf35282f687452e6"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_160/resources.csv",
      "bytes": 42092,
      "source_sha256": "ab21429d1a585729dd86aabb017e9bc1d415d0598768343a16adce8170f5baa2"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_4/summary.json",
      "bytes": 2554,
      "source_sha256": "79ef7c514ac13ece981eb91b3415e78545987fb0e46ac3f3178d7e8d1be60a74"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_4/resources.csv",
      "bytes": 5542,
      "source_sha256": "5bdbf93581c784165eacd6ca26e63a27904a078a1b3412ad8d402ad6a5036da1"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_384/summary.json",
      "bytes": 2557,
      "source_sha256": "f7bfa3129d88ed91f528daf503e4f8b906437092adc020fbd3b3d2825147170e"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_384/resources.csv",
      "bytes": 89012,
      "source_sha256": "271019e98ba2609ad7f2ba97c906af31314568fbd1fb8bc9e02dffaaae20ce11"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_1/summary.json",
      "bytes": 2472,
      "source_sha256": "7e78698ba4dacf5274bde26c43614f2b8da27d5d631b8c3a11c944046586749d"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_1/resources.csv",
      "bytes": 5234,
      "source_sha256": "28d56644f6b32f21c0fb3335a6fad978ff6b8c0efffba9f10ad894fae7ab28e9"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_128/summary.json",
      "bytes": 2571,
      "source_sha256": "4e66858833675aa4d25f174a5e8301db569149a0a65ee2cc156ca1bccb00d3b2"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_128/resources.csv",
      "bytes": 33513,
      "source_sha256": "a077ac6a923e8d95ad8129650fc4db9abb3e115d04474164fc80b35625b8435c"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_96/summary.json",
      "bytes": 2566,
      "source_sha256": "3aacfa87d493ade359157c50bb11d087eed2237bed217f76b535271e8d185f4c"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_96/resources.csv",
      "bytes": 25910,
      "source_sha256": "8fbb791e6d18afbdeff61717edd4885c1b77a4ab842975e4aab764f84c8cef28"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_224/summary.json",
      "bytes": 2574,
      "source_sha256": "96ed5da8a4a4a4bdc5738dbde7095cf4e2f9e72b2e11769b65f07339f4ee5261"
    },
    {
      "path": "esm2/results/batch_inference_20261028_112900_601721/batch_224/resources.csv",
      "bytes": 54109,
      "source_sha256": "4778624831fde655d26ebadd38a0e45f80fb672f002cf9ee2316748f30a69154"
    },
    {
      "path": "esm2/results/loading_20261027_190701_789190/loading.json",
      "bytes": 640,
      "source_sha256": "1955cb20caaed6caf18c534d772456bb88fc0caa9dd47893cbca78ba8ffbee3d"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/fresh_reload_predictions.json",
      "bytes": 5299,
      "source_sha256": "82217bee01b0704c8301d057c7a4b9304e5f1d8e919b00df5ab02ce3425bfdac"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/summary.json",
      "bytes": 9341,
      "source_sha256": "bb2fd6c024b34a42b2f6c209045c383b1fe98586e784b3b685d956a3c8d88812"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/length_257_512_metrics.json",
      "bytes": 1874,
      "source_sha256": "0f77ee8ff6bca918e571c204d020f46997d7f0afa8065e91eced0de3c520dc83"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/resources.csv",
      "bytes": 56396,
      "source_sha256": "83595746dc5e94271a9884e59e614713571ecaf55578af89c547a7d7f8b37af6"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/length_129_256_metrics.json",
      "bytes": 1854,
      "source_sha256": "49d9ad59f53034143a2919a8a7e9874535cfb2856468b94c6026ad2791aa4cdb"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/exit_code.json",
      "bytes": 74,
      "source_sha256": "a127ea8a6baf0af1477db62f67524492e25db703318367f945a4b6aae5f0ca9f"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/cli_validation.json",
      "bytes": 82,
      "source_sha256": "8d3a618af244b0879728ab85071c839d1a1440207cd169d211494d1ff7ad221d"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/command.json",
      "bytes": 196,
      "source_sha256": "e50a382b05f7e96e148491848a8612d5470041ec85be1bca28aa542c438ca0b8"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/original_artifact_validation.json",
      "bytes": 545,
      "source_sha256": "91f57c500bf846efed47869369e5fb2f8b9c83046d38c8b5a7d9a9b0a0d3ffad"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/length_513_1022_metrics.json",
      "bytes": 1872,
      "source_sha256": "0e58e4ca9f6f927654a7eb087157ad2cdf68a2d8a986ce39cd2e8e0360d65d17"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/cli_predictions.csv",
      "bytes": 160,
      "source_sha256": "e302ba885de9eef2437f975ee3b0f7f42b3bccf88d0d8f72fa5f8a9d205d0999"
    },
    {
      "path": "esm2/results/supplement_20261027_165443_595867/length_50_128_metrics.json",
      "bytes": 1879,
      "source_sha256": "58ccdc862a8c1d9e9716e736c898636a047ceb1908ed1349352ba1bfd8ae9a0d"
    },
    {
      "path": "esm2/data/uniprot_reviewed_900_950_20261028.source.json",
      "bytes": 1561,
      "source_sha256": "9d7a8dc43fb26d110ff81f926fdc7b5852cf373007301e91aee277c504319440"
    },
    {
      "path": "esm2/data/sources.json",
      "bytes": 6137,
      "source_sha256": "e7ebd5ffacbf41ccbd2decc584223c4ce9cd442ef1feef7559022c9c52a80f71"
    },
    {
      "path": "esm2/models/finetuned/20261027_164300_125163/metadata.json",
      "bytes": 362,
      "source_sha256": "d05d1be4796a26a411c91a0491ceca0b48e0f488cc52646249fbe9bd0e53a78d"
    },
    {
      "path": "esm2/models/finetuned/20261027_164300_125163/training_config.json",
      "bytes": 498,
      "source_sha256": "8108b77c4416fb5129d9c62da99ff076ea9b0903c53ad2901bc585310378279b"
    },
    {
      "path": "esm2/models/finetuned/20261027_164300_125163/preprocessing.json",
      "bytes": 569,
      "source_sha256": "56a6c2288bd2b753f8b7ba30650efbca8666c114a9aa38a77686a01dff861d6d"
    },
    {
      "path": "colabfold/scripts/measure_loading.py",
      "bytes": 2257,
      "source_sha256": "39667da7763f11bbaacf0a1b1e1264bb28b641c51600b6ba4af63cf5355c1b50"
    },
    {
      "path": "colabfold/scripts/validate_structures.py",
      "bytes": 3770,
      "source_sha256": "8137f03a6ec35c630f5843b9fb7c61dfe97715857676cc0b562b3b6a07a45500"
    },
    {
      "path": "colabfold/scripts/launch_multimer.py",
      "bytes": 1590,
      "source_sha256": "ef4104935d86cc6e0f25ee6e08f4af0e1ce26f8494b77ef71650e950e8d6d73f"
    },
    {
      "path": "colabfold/scripts/launch.py",
      "bytes": 1438,
      "source_sha256": "c5a623b4a8a1e0efcf840b18ac32c52d82447095e594d328fcd34c2227a05f53"
    },
    {
      "path": "colabfold/scripts/prepare_inputs.py",
      "bytes": 1884,
      "source_sha256": "2a327312228528a70d4ba6e8003cee0096ce9059a1b0d67c510b05d7561b1e7f"
    },
    {
      "path": "colabfold/scripts/benchmark_af2.py",
      "bytes": 11120,
      "source_sha256": "f11c9e66c3b8f3514780db7513ff135ca9f8f9fdc706dc0884f8b06b648cdee3"
    },
    {
      "path": "colabfold/scripts/report_af2.py",
      "bytes": 24538,
      "source_sha256": "34c4855a7ffc5dd9b66aa88a94f1f69230d9a5d5a05917028eb48d667d6e3cb3"
    },
    {
      "path": "colabfold/scripts/inspect_environment.py",
      "bytes": 971,
      "source_sha256": "2fcfbabaa25bad65104e67a2c41bebaaa1ba77dd4b4696ff956fe36acf894c3d"
    },
    {
      "path": "colabfold/scripts/batch_followup/capture.py",
      "bytes": 2352,
      "source_sha256": "d9f662d352182b1ceaafeb7bc3aed41b58c83d54d0eb982b4a27d1a7e971c3ff"
    },
    {
      "path": "colabfold/scripts/batch_followup/worker.py",
      "bytes": 8561,
      "source_sha256": "3d83bfc23d5a62f73522ebc6ddd26ad91c3f0a0a666f7a76b97eb4da74da008a"
    },
    {
      "path": "colabfold/reports/AlphaFold2_ColabFold_af2_20261027_172813_247681.md",
      "bytes": 18407,
      "source_sha256": "4db5b915267b5776e0a175954e5b1669469f0bcf2b2998edd23ff82b6b511f62"
    },
    {
      "path": "colabfold/reports/weight_manifest.json",
      "bytes": 2302,
      "source_sha256": "d2ddaac544c6f602e673e18d3f5edb4a48c79241416f037239de56645d2ff25f"
    },
    {
      "path": "colabfold/reports/comparison.csv",
      "bytes": 2860,
      "source_sha256": "b5a2d3f4ea4619db64905a92545a7dfa171b3c22d456c87d30c6a945e9c56f30"
    },
    {
      "path": "colabfold/reports/metrics.json",
      "bytes": 9078,
      "source_sha256": "ac170b3f39958bd98aa1eadf3bb2ae3bb52436fe4d566f89046f4ad94829c225"
    },
    {
      "path": "colabfold/reports/AlphaFold2_ColabFold.md",
      "bytes": 20217,
      "source_sha256": "49da1f3033a6e9b3ee2dbe57f9e7c03c4a4274e862e2bc3a73093787cb71faa3"
    },
    {
      "path": "colabfold/reports/final_environment.json",
      "bytes": 5803,
      "source_sha256": "7023083f88856dacffe70f0d63d598b7a0d8918811640ddd3e68bb52f5d3e066"
    },
    {
      "path": "colabfold/reports/count_prefix_metrics.json",
      "bytes": 1780,
      "source_sha256": "d56eb0ef71327ce87cd32486ddf92d9a906c3ec703d7231aaf2fa9861b562722"
    },
    {
      "path": "colabfold/reports/Batch_memory_followup.md",
      "bytes": 3064,
      "source_sha256": "a0364436728d9aabbc6754708ea5cc9205c275ee480a48fae30e8fcd0b18597e"
    },
    {
      "path": "colabfold/reports/performance.png",
      "bytes": 151904,
      "source_sha256": "947297b572c700377ab9306ea698883fc71d0be19754fd066fa98013c59f2796"
    },
    {
      "path": "colabfold/reports/artifact_manifest.json",
      "bytes": 42945,
      "source_sha256": "11945a6e3b2b3bf35c7e8277de29215a26355e15caee76a59a89b01c7b0faa0c"
    },
    {
      "path": "colabfold/reports/batch_memory_followup.csv",
      "bytes": 495,
      "source_sha256": "474dbb8f0d6f87dcc31e04c447a8d8b2b1eb4a8db9fc36ccdb17fe11ebf2f585"
    },
    {
      "path": "colabfold/reports/mplcache/fontlist-v3.11.0.json",
      "bytes": 26961,
      "source_sha256": "f9996e005272d3edad82b7fc092602b9a6f2ee6a90f9849dd6b8d3ecc8a40485"
    },
    {
      "path": "colabfold/results/environment_20261027_170134_167136/environment_check.json",
      "bytes": 1111,
      "source_sha256": "240685315786d26d7d38745ad4bc98a13dfc74de09aef782edf105307896952e"
    },
    {
      "path": "colabfold/results/loading_alphafold2_multimer_v3_20261027_100708_634553/loading.json",
      "bytes": 1542,
      "source_sha256": "7b0ce7e0be7caa2d56d10a4aabe01bb7069067b78e079f084ff374e0caa35cf2"
    },
    {
      "path": "colabfold/results/loading_alphafold2_multimer_v3_20261027_101211_147305/loading.json",
      "bytes": 2170,
      "source_sha256": "629dc1cde95b914ca9de61173755e546906d3b933b7ff35ed8d04a2172b9da3a"
    },
    {
      "path": "colabfold/results/loading_alphafold2_ptm_20261027_101206_957503/loading.json",
      "bytes": 2159,
      "source_sha256": "9dc3721a40de5997c35adf40386b6589db3a3e1e5b478c4c6f76c0197fcf6330"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/input_audit.json",
      "bytes": 1741,
      "source_sha256": "34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/summary.json",
      "bytes": 4302,
      "source_sha256": "afafa1ea1f383449e97d09dea06f0b9e2eaa0962154d1396e7208b1f1d5c3d88"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/batch1_exit.json",
      "bytes": 68,
      "source_sha256": "bdec275bbb902bac5c6478f31945aeff3e79af5d0aff5ee641a53761ac892eef"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/runner_config.json",
      "bytes": 12560,
      "source_sha256": "707eb227ce413194fb0bd2a7a6f44bbb0199b5ca4fb7ac8a2a0c8cc4becbdf7a"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/capture_exit.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/batch1_command.json",
      "bytes": 665,
      "source_sha256": "57590c787aadfcd194e74c93df19506c8b2c8e4d5f35af183c47d468faeb9d2f"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/capture_command.json",
      "bytes": 508,
      "source_sha256": "16fefa7b2e2667dc7cc32e01c60a60d2e01f742eff1c54df62a859b7622cca37"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/capture_count1_A2RU14/config.json",
      "bytes": 1117,
      "source_sha256": "7308973cbbbd59668bae3e20d7ca01aeb787d80a455a790d01b8294dfe30899f"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/capture_count2_A6NFY7/config.json",
      "bytes": 1117,
      "source_sha256": "7308973cbbbd59668bae3e20d7ca01aeb787d80a455a790d01b8294dfe30899f"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/mplcache/fontlist-v3.11.0.json",
      "bytes": 26961,
      "source_sha256": "607a422b5a35515c7265ee87c7e9e42d68d58bc443af0252f2e8bdac311ad20a"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/capture_count4_P01270/config.json",
      "bytes": 1117,
      "source_sha256": "7308973cbbbd59668bae3e20d7ca01aeb787d80a455a790d01b8294dfe30899f"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/capture_count3_O14519/config.json",
      "bytes": 1117,
      "source_sha256": "7308973cbbbd59668bae3e20d7ca01aeb787d80a455a790d01b8294dfe30899f"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/batch_1/summary.json",
      "bytes": 3247,
      "source_sha256": "648bbf91c3d148888dbb679892308ad6d5a2e9037d6e94d100bfe228c1928925"
    },
    {
      "path": "colabfold/results/vmap_batch_20261028_185758_495006/batch_1/resources.csv",
      "bytes": 1027,
      "source_sha256": "fecc341c984e327dde97c89f299ff508d076a13f9e6c86bb0c64d8f9c8bc7a39"
    },
    {
      "path": "colabfold/results/environment_verified/environment.json",
      "bytes": 3159,
      "source_sha256": "a083282cdcbcb0e329f3b5ef6cc9d1c058e8f97aa44bfa51b5f08c829c7ef47f"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/input_report.json",
      "bytes": 4029,
      "source_sha256": "bd3245f28e156f683af563f24049a2fbde69c6da5bf0287cde7db53f2c4b70aa"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/summary.json",
      "bytes": 16537,
      "source_sha256": "1d72cd1c81341bd327f7e8ee4f3b4b9cdce6b62c4dbaf32ff7c8afbb5ac73f75"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/resources.csv",
      "bytes": 80730,
      "source_sha256": "48e8e79e2865416f8a9a8b817d47486782a1ee5b522262c5953a41a390d3d0cf"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/aggregated_metrics.json",
      "bytes": 7952,
      "source_sha256": "7eccb49ae5973915b96b1a695c656be1b4ebce7631c01f7def2630dbf340a014"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/exit_code.json",
      "bytes": 120,
      "source_sha256": "2a1c761937db8baa9fa8a27e05fc8a7995114b989b8b63a94b793f04652556be"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/command.json",
      "bytes": 503,
      "source_sha256": "f6a82754f3156ad61690f725fd155583ddc8135c5218b470ca22b5c50359dab8"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/structure_validation.json",
      "bytes": 24249,
      "source_sha256": "f793e8cb2a383d3b7f9a6fe36f7272cbdf7d8e03930cd897778043b4c130e890"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/query_stats.json",
      "bytes": 1032,
      "source_sha256": "1ba3cca0457de03f7ccd48fc1538a128271e629037bca92044602e1843cab3e9"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/apply_calls.json",
      "bytes": 71356,
      "source_sha256": "0e9320f4d1c11cfeda74382107090e6198f7b4bd05405461adb8b9085779d86d"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions.json",
      "bytes": 12982,
      "source_sha256": "82ad0f9844e58cd4d13671758362561d8be0e367c11dd93fdc07bc954de905f0"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/log.txt",
      "bytes": 20227,
      "source_sha256": "2904f540324d31524ef0c865c21c799a826ec3d7bdbc682a5fb599e883b25e6a"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/mplcache/fontlist-v3.11.0.json",
      "bytes": 26961,
      "source_sha256": "a6c190ec34f6f5a7ce5240d9cd476da7e265b2ce7026cfd974b31da3ff067c94"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1LYZ.done.txt",
      "bytes": 0,
      "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_001_alphafold2_ptm_model_5_seed_042.json",
      "bytes": 73185,
      "source_sha256": "dbec8bc5b9fff30b18b796028c1d12a9edc74f27fdff5a97bcb13712af6302ce"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1UBQ.done.txt",
      "bytes": 0,
      "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "bytes": 74054,
      "source_sha256": "3cdf95362cb4a9903fcb5362f4b6716fa95127750f74bd0cffac634bf44820e2"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "bytes": 29222,
      "source_sha256": "18602362625b3c85b648b30bcde5a6ad09c2d023ecef2f7755faeb7dcb00cab3"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_001_alphafold2_ptm_model_3_seed_042.json",
      "bytes": 76363,
      "source_sha256": "d37d983fb446a5380c6da14974d785bc40f61546de9a39dbfef923c81936c390"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1TIM.done.txt",
      "bytes": 0,
      "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_001_alphafold2_ptm_model_4_seed_042.json",
      "bytes": 74130,
      "source_sha256": "17a69652b88841c533532c894b88f7c964128a8afb6b27e0baed86e54555041c"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_004_alphafold2_ptm_model_1_seed_042.json",
      "bytes": 302298,
      "source_sha256": "eb71745e82c071c80a4a0d9e2a277410807e79eba0d35819a15ee40054aacad4"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_005_alphafold2_ptm_model_1_seed_042.json",
      "bytes": 82710,
      "source_sha256": "91e21d2d3e07280b088286b13df3f0fca50ff4f912faecf088f4d7656f00cd85"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_003_alphafold2_ptm_model_4_seed_042.json",
      "bytes": 302532,
      "source_sha256": "88d31cd143d86e150d2ae192f7dd3a8d3da0072fa8781b4d51a05861ca5c4920"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1UBQ_predicted_aligned_error_v1.json",
      "bytes": 34552,
      "source_sha256": "7da61e78202bd8bf5fbe40056a90a05a2a2eb1d896e27d9b90f6cc267ddf0c83"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_005_alphafold2_ptm_model_4_seed_042.json",
      "bytes": 77166,
      "source_sha256": "d622c3c5a3f0ef41390b046be28d2f25fa81304f9e11b954a9b112d158e5eff6"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_003_alphafold2_ptm_model_2_seed_042.json",
      "bytes": 29206,
      "source_sha256": "e8af55f39fa6607d4329b52269b948ecb71659d2cf78db4bf2871986f6acb158"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_004_alphafold2_ptm_model_2_seed_042.json",
      "bytes": 75318,
      "source_sha256": "e80b915f31e7ab8f8dfbf979fd50df0e8341b64acf6cc41e909125458bd00d99"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_003_alphafold2_ptm_model_4_seed_042.json",
      "bytes": 75228,
      "source_sha256": "96400d7d4909637993f94d9b7d2f9077293757c3fe4cac37b349164200951b86"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_005_alphafold2_ptm_model_5_seed_042.json",
      "bytes": 29187,
      "source_sha256": "61926c1c59ed49a11b7de466559e5e76aa36cce2119339f6b1ecf71e78383f41"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count1_A2RU14.done.txt",
      "bytes": 0,
      "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_predicted_aligned_error_v1.json",
      "bytes": 86696,
      "source_sha256": "61dcadcdb882107154587ea9bec884511b363eec85702bea6cbac8954a05aade"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count4_P01270_predicted_aligned_error_v1.json",
      "bytes": 88925,
      "source_sha256": "561d3005a1bd4379e8eb44417dbed7a12cb5ff6f5e1cce28cfcb270b4327303c"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_002_alphafold2_ptm_model_4_seed_042.json",
      "bytes": 82650,
      "source_sha256": "8e05e0f1b758e7fa399e072e3693f2cb28247b3486d2fd63bf6298c0f261f535"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_005_alphafold2_ptm_model_2_seed_042.json",
      "bytes": 74310,
      "source_sha256": "44cdaa4dd86b9003aff44484816335c4bb3a7d30ff001578bde830730b364b92"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count4_P01270.done.txt",
      "bytes": 0,
      "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_005_alphafold2_ptm_model_2_seed_042.json",
      "bytes": 73916,
      "source_sha256": "8005bd4462d3cefc68101d95c7d6056f0a35ace1be8a1113a78342e84eb59234"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_004_alphafold2_ptm_model_1_seed_042.json",
      "bytes": 74426,
      "source_sha256": "b7e7858ce4ecb3f51a08b701ac24a7d65331aa233ce18141eaccda6da76c6777"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_004_alphafold2_ptm_model_4_seed_042.json",
      "bytes": 74155,
      "source_sha256": "8bb379103cd695cbbfafcbb6b972b9614c69d97ace5f6f39d10575c9c0361c6e"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_005_alphafold2_ptm_model_5_seed_042.json",
      "bytes": 75270,
      "source_sha256": "772daec272e83c2b5faa13add5b65777eca69c8e7fe166ae494bb3c22302f348"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_004_alphafold2_ptm_model_2_seed_042.json",
      "bytes": 82633,
      "source_sha256": "c5fc9e9bf85650dff960e05a5362b07e63b9ba452198af68d9efd3f204c0def9"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_002_alphafold2_ptm_model_1_seed_042.json",
      "bytes": 76311,
      "source_sha256": "3d72984db480bcaaca32b3cf2c29c057db772231b536564790fbceb04953ddf0"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/config.json",
      "bytes": 1162,
      "source_sha256": "e82886c416da5105bd891dc47ca493e9f4136aae0dd38faeafb9306d30fa6207"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_003_alphafold2_ptm_model_1_seed_042.json",
      "bytes": 73893,
      "source_sha256": "a6399fab5ebdd17988b385b2dd1cdcf76b1745634de6d995a88f541410f48a6d"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count3_O14519.done.txt",
      "bytes": 0,
      "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "bytes": 75288,
      "source_sha256": "39b655736aeb64350209ab7aa58b1e838a7a759ee24fba26b2e10ccd81e4deec"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count2_A6NFY7_scores_rank_003_alphafold2_ptm_model_5_seed_042.json",
      "bytes": 74169,
      "source_sha256": "a99fab0040179abdec533573e7e99f9907e0f6f9ad883010f76b09dc53a061e9"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_003_alphafold2_ptm_model_5_seed_042.json",
      "bytes": 76698,
      "source_sha256": "103090c5bb820e369ab7162853c54fd93bf6be71a8da7dbf52c50690e8a199a2"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_001_alphafold2_ptm_model_4_seed_042.json",
      "bytes": 29200,
      "source_sha256": "75d93e720494adef62f2dd456495c6fe32fdb9c42f119b24bf527ca858c37945"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_005_alphafold2_ptm_model_2_seed_042.json",
      "bytes": 302979,
      "source_sha256": "a81bc8397985d38969f3765bc4c536cf07af38cb39495590ff34672024f147c4"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_001_alphafold2_ptm_model_5_seed_042.json",
      "bytes": 82663,
      "source_sha256": "9c959d989cba97ac8c9982364a443e4c07f6cffa2dc4b36564cdeb4f90a04aac"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count1_A2RU14_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "bytes": 74169,
      "source_sha256": "9e38de55fe7d4b7d26469feb9832fbb7af9fa210d55dc96181483a173af44b98"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count3_O14519_scores_rank_001_alphafold2_ptm_model_1_seed_042.json",
      "bytes": 75336,
      "source_sha256": "f0f29630580545126661f51627181097d0c3fae152295ed4d097ec8e23c97694"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1UBQ_scores_rank_004_alphafold2_ptm_model_1_seed_042.json",
      "bytes": 29192,
      "source_sha256": "ea9f469bf97f0aacd5602496c36b8a2b0d2a0df1ade61a6e0fabe83e6018a526"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count3_O14519_predicted_aligned_error_v1.json",
      "bytes": 87905,
      "source_sha256": "b74c99f10a791102274aac3f91217e2ad3d5f55bc4da9db874e8a6cabb78e140"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1TIM_predicted_aligned_error_v1.json",
      "bytes": 362101,
      "source_sha256": "08dfc49cc6070dd327b85b7f559c7593fd2657ab4e975ed7affa6e8e403c1842"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_001_alphafold2_ptm_model_5_seed_042.json",
      "bytes": 302532,
      "source_sha256": "baca0a075c53a45c7430bc61fdff548c43b9c84191aafd4a482a2907a5dd98d7"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count4_P01270_scores_rank_004_alphafold2_ptm_model_2_seed_042.json",
      "bytes": 76949,
      "source_sha256": "830a44b72c99b56f60ead51615b4eb94e70315a5f3a53ced01d2d3f953507522"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1TIM_scores_rank_002_alphafold2_ptm_model_3_seed_042.json",
      "bytes": 302286,
      "source_sha256": "f4324b388f8b91b17de9db330dc5045b1bcfbb75e76859b0b8e1b6b1e6cdcd58"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1LYZ_scores_rank_003_alphafold2_ptm_model_3_seed_042.json",
      "bytes": 82617,
      "source_sha256": "568be74cc0bcbdb58490a30441483f1112eedd19d68f8fc71e01768b7b853513"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count2_A6NFY7.done.txt",
      "bytes": 0,
      "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/count1_A2RU14_predicted_aligned_error_v1.json",
      "bytes": 85753,
      "source_sha256": "4ec977aff3a6d5e4665b9e850d123bbe0629fc490d21216d6c1d2ebe4ff201a9"
    },
    {
      "path": "colabfold/results/af2_20261027_172813_247681/predictions/1LYZ_predicted_aligned_error_v1.json",
      "bytes": 98562,
      "source_sha256": "4af2639dd6f21e25dcaf78a69986d68cd2d77d95fed6cb23eeb9b595bee3cf63"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/input_audit.json",
      "bytes": 1741,
      "source_sha256": "34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/parent_run.json",
      "bytes": 285,
      "source_sha256": "50f5eddfefb44456fa3a430f36ff322d5ac59bc079f2f3dd8dcd181fb6d33dd4"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch2_command.json",
      "bytes": 662,
      "source_sha256": "08e32def6516646237a84e9558052a62015dfb3d7031a3e8ca41e08dc4ab0ce8"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch2_exit.json",
      "bytes": 67,
      "source_sha256": "9ee5e4a3fbb2949b24ea4efd96e8cb67ea37bc6e1498a9b3d4399dd80628e751"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/summary.json",
      "bytes": 17978,
      "source_sha256": "aa0105d06562fc9ba562769b4d637acf865004fa070a584995bc28a07109516b"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch1_exit.json",
      "bytes": 67,
      "source_sha256": "7eb82672f3705de0cbe750d83fedab0a43da9476163acd65dc11cdb95c8418d4"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch4_exit.json",
      "bytes": 68,
      "source_sha256": "d06b36c239f0a6e1880f1f99318fc27c063d5041da8d383123dcd936f0d7a55b"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/runner_config.json",
      "bytes": 12560,
      "source_sha256": "707eb227ce413194fb0bd2a7a6f44bbb0199b5ca4fb7ac8a2a0c8cc4becbdf7a"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/capture_exit.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/experiment_protocol.json",
      "bytes": 505,
      "source_sha256": "81cd4a4720a2e6bb66efb2ad1b7efc7e2cd2504dc4e578eacf24e5b5d0445b6a"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch4_command.json",
      "bytes": 662,
      "source_sha256": "74e4c1cf908f63d2330b27060a571896d7c79e36aa3337e345ff1154d2c8ff78"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch1_command.json",
      "bytes": 662,
      "source_sha256": "ad73ef573ce45307ddd3c0386e5374c451da1e20908f849c99560468707f9f14"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch_2/summary.json",
      "bytes": 4928,
      "source_sha256": "582be95983c4c80b69107a856f545fe58b85a2f079f9887e374b687b33be088e"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch_2/resources.csv",
      "bytes": 18067,
      "source_sha256": "8c6a98cecb3d0531c8f200436e0d148cce1a5952bce28c868d6491489ce6c780"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch_4/summary.json",
      "bytes": 5280,
      "source_sha256": "fe0a217ce7e50944d2739df2203cc75b57f8dade754c2a0b0da570198a2f0ced"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch_4/resources.csv",
      "bytes": 34002,
      "source_sha256": "1bafa53e7c6785491df0da8ebd989f2b7e0361348d9a29eac52396ea85cc3406"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch_1/summary.json",
      "bytes": 4843,
      "source_sha256": "d28e2930c3c298e008a1f47873a94acd717d34d25e0661e4e5527517d137fbd4"
    },
    {
      "path": "colabfold/results/vmap_fp32_20261028_190843_576148/batch_1/resources.csv",
      "bytes": 14648,
      "source_sha256": "1e855f30d6afdf1007a38366b732d7dbd8683ec9d79007f7675355b1df1b1e95"
    },
    {
      "path": "colabfold/results/af2_20261027_172609_696571/input_report.json",
      "bytes": 4029,
      "source_sha256": "bd3245f28e156f683af563f24049a2fbde69c6da5bf0287cde7db53f2c4b70aa"
    },
    {
      "path": "colabfold/results/af2_20261027_172609_696571/summary.json",
      "bytes": 2264,
      "source_sha256": "b8e8c44e2079ade1ce413d2e57159c17a3406f87be7c3f669e7ba9c0e44cc8ec"
    },
    {
      "path": "colabfold/results/af2_20261027_172609_696571/resources.csv",
      "bytes": 1340,
      "source_sha256": "56f8c6167a71452006dc17345aa967be46055ea323b0cc138aa06e6f8d408216"
    },
    {
      "path": "colabfold/results/af2_20261027_172609_696571/exit_code.json",
      "bytes": 120,
      "source_sha256": "0ffa99ba8d9353f2f7c12ed7789b2a0eb6737813c19a6fa9e0fc123c4c3fa2d5"
    },
    {
      "path": "colabfold/results/af2_20261027_172609_696571/command.json",
      "bytes": 499,
      "source_sha256": "0f3d43ff40cc0197ce7c2f1973d7fd25e24d62e4d3aa93318391e7072ebe21d3"
    },
    {
      "path": "colabfold/results/af2_20261027_172609_696571/log.txt",
      "bytes": 1493,
      "source_sha256": "0939cf8d25f5cdc7f950130787bd67b29e7ba564f2e9294d6830711ef34a90e5"
    },
    {
      "path": "colabfold/results/af2_20261027_172609_696571/mplcache/fontlist-v3.11.0.json",
      "bytes": 26961,
      "source_sha256": "4692c99435f2d2ee69575d9316c47aa70aa6868dcd8c980d2dcd32357f85e672"
    },
    {
      "path": "colabfold/results/af2_20261027_172609_696571/predictions/config.json",
      "bytes": 1162,
      "source_sha256": "e82886c416da5105bd891dc47ca493e9f4136aae0dd38faeafb9306d30fa6207"
    },
    {
      "path": "colabfold/results/loading_alphafold2_ptm_20261027_100705_136907/loading.json",
      "bytes": 1534,
      "source_sha256": "a8c1ba3a5e1cf47e845f703bc6ffc1027a0ebf2ad9e7c420d3aadda1b2310dd3"
    },
    {
      "path": "colabfold/results/msa_complex_20261027_173649_931405/exit_code.json",
      "bytes": 120,
      "source_sha256": "7ab25aaeedd36ca96f59eb6eec343444fbf5435fcc934eca2a640c7b091c26df"
    },
    {
      "path": "colabfold/results/msa_complex_20261027_173649_931405/command.json",
      "bytes": 571,
      "source_sha256": "ca377a53d691f51d0df0c294eae298b2993d8093e619de949b8c38f1cd8a70bf"
    },
    {
      "path": "colabfold/results/msa_complex_20261027_173649_931405/msa/config.json",
      "bytes": 1193,
      "source_sha256": "90899c0db89ff7e9509c661616a41d9d6521e03e3207f6849a93f4830a2cc49a"
    },
    {
      "path": "colabfold/results/msa_complex_20261027_173649_931405/msa/log.txt",
      "bytes": 833,
      "source_sha256": "0cb3a6327497d1cc90edfbe1ab97cec7907f83cef8a75f357826965899d15cd5"
    },
    {
      "path": "colabfold/results/msa_20261027_172359_775627/exit_code.json",
      "bytes": 120,
      "source_sha256": "a360ba8b2e28b6f42ab27f5d8d253a66826a0bed401570a7397eb18e614fc7c9"
    },
    {
      "path": "colabfold/results/msa_20261027_172359_775627/command.json",
      "bytes": 547,
      "source_sha256": "380b7ba942ec309bc1be2b1d7aef81ee6563075845936312da6ba2df64d297e5"
    },
    {
      "path": "colabfold/results/msa_20261027_172359_775627/msa/config.json",
      "bytes": 1182,
      "source_sha256": "1f89e453012a50e90c1c56b85bd9e46af13becf500958c9d6abbadab02df692d"
    },
    {
      "path": "colabfold/results/msa_20261027_172359_775627/msa/log.txt",
      "bytes": 1689,
      "source_sha256": "79706534b7e8cf572e0bd47ac43919a807253fc7617f756dc0d00cc6a8d37093"
    },
    {
      "path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/input_audit.json",
      "bytes": 1741,
      "source_sha256": "34491f5107a33365f60297e4afb5e06e7d93319d446bc6ba129587b17a32c75d"
    },
    {
      "path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/parent_run.json",
      "bytes": 200,
      "source_sha256": "6b22dc5fe584c5a43f1b4daa52376503d26a87bc6db040e2eccb439f54ff08d9"
    },
    {
      "path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/summary.json",
      "bytes": 6052,
      "source_sha256": "98019c9ab27efc01110cb4f9bf3370c833a7c2bd20a98fbee5437fd21f40ccc5"
    },
    {
      "path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/batch1_exit.json",
      "bytes": 65,
      "source_sha256": "2028dac98989b2bd0cf71e919a0c55149f0c2ed5ad476d9737505e70cf1b446f"
    },
    {
      "path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/runner_config.json",
      "bytes": 12560,
      "source_sha256": "707eb227ce413194fb0bd2a7a6f44bbb0199b5ca4fb7ac8a2a0c8cc4becbdf7a"
    },
    {
      "path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/capture_exit.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/batch1_command.json",
      "bytes": 683,
      "source_sha256": "51ee18f08e6f1d05d975f908caaf4ed2f3ea9a816aa4ed03f480655df1556dbc"
    },
    {
      "path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/batch_1/summary.json",
      "bytes": 4811,
      "source_sha256": "c3ccc330fc83106cba75c4283b9e2136f07c1c55ca88261d12c07c98973a9480"
    },
    {
      "path": "colabfold/results/vmap_batch_retry_20261028_190537_289704/batch_1/resources.csv",
      "bytes": 11462,
      "source_sha256": "40251881c094a8cf8ce51176ae4a645e67002c0c3d60535d75b65cccee8a7f54"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/input_report.json",
      "bytes": 583,
      "source_sha256": "4ea493dc9aeda0a38a7fa184aa3724dddc624433201ace856879f50e54eabd05"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/summary.json",
      "bytes": 4304,
      "source_sha256": "e74f0d9c341599b2f3c3567828e79823659b8f429a235addc8842f86e9474f66"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/resources.csv",
      "bytes": 67919,
      "source_sha256": "643c70e81c8385786b3528589d8365329dea827088e324047586add7518445f7"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/aggregated_metrics.json",
      "bytes": 1128,
      "source_sha256": "f7fbe077de1a9fd98172b6bd9cf70552724cb94bcbb26b6ab93b74094c623d7c"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/exit_code.json",
      "bytes": 120,
      "source_sha256": "8f53062da0f07eabab47b582e9088c2582fc1df131186964065c5875a1582e62"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/command.json",
      "bytes": 547,
      "source_sha256": "05ecf1ac194852ce6d9973180e81edf806414c530c1411c34089922f87780eb1"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/structure_validation.json",
      "bytes": 9514,
      "source_sha256": "e330f16f7ea62486a1dcdc5b186baf706a03e88b97ca8905c32af1825a0d39d0"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/query_stats.json",
      "bytes": 154,
      "source_sha256": "cbf6a3a6f8b31a485ad2e917c073e8b0b3f1034d76ce18b6e01e82f81fb78159"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/apply_calls.json",
      "bytes": 53883,
      "source_sha256": "3fc11100da16bf2329024113a65fba6069d71f4acf88817f3dc83a1c20fa3ab2"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/predictions.json",
      "bytes": 2090,
      "source_sha256": "80ec4824a7e9374698c442a02906f34d27cfbe17b44ce06e6de4a0373b4b537a"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/log.txt",
      "bytes": 13535,
      "source_sha256": "58955071e4db8d11e01082de661793b2767a6bd7dc0a43df154fcda6e08dc3fd"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/mplcache/fontlist-v3.11.0.json",
      "bytes": 26961,
      "source_sha256": "2c17ab036d2afd0b23ce7d18b4e28d435d8e0833a120995e75472f0c410fbffd"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/predictions/3G5O_A_3G5O_B_scores_rank_003_alphafold2_multimer_v3_model_5_seed_042.json",
      "bytes": 163126,
      "source_sha256": "310ea3fec3f6ab4e0546995e6158a97c17a1cf919979c616f613188c21186b2f"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/predictions/3G5O_A_3G5O_B_scores_rank_002_alphafold2_multimer_v3_model_3_seed_042.json",
      "bytes": 163018,
      "source_sha256": "869b47a88ac741ab9aef8b38d434c41468117e09477c8e3238c9c23aecd4dd16"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/predictions/3G5O_A_3G5O_B_scores_rank_001_alphafold2_multimer_v3_model_4_seed_042.json",
      "bytes": 163051,
      "source_sha256": "e73dd1fe19dd67990dc938172c991269f9a36bfa1a1675240889717b0b6b7fb4"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/predictions/3G5O_A_3G5O_B_scores_rank_004_alphafold2_multimer_v3_model_2_seed_042.json",
      "bytes": 163066,
      "source_sha256": "f532f1193df163cf2b294747220554f31b0c2d48dddd36c4f155b1665e00cd2a"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/predictions/config.json",
      "bytes": 1174,
      "source_sha256": "7bfbb3261d7c2bbed3c42f6177b0d0311ed95938a5dd8729a7f0e665bb300181"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/predictions/3G5O_A_3G5O_B_scores_rank_005_alphafold2_multimer_v3_model_1_seed_042.json",
      "bytes": 163244,
      "source_sha256": "2ba8aa0dea148b2815bd4b4053acf7de16a9facbbbc096f21b23c2555392fabf"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/predictions/3G5O_A_3G5O_B_predicted_aligned_error_v1.json",
      "bytes": 194402,
      "source_sha256": "60412a8c111d0cbd917bcc2ae000de784cd1150aecab2772a2d7b3b03654ff06"
    },
    {
      "path": "colabfold/results/af2_multimer_20261027_181640_431467/predictions/3G5O_A_3G5O_B.done.txt",
      "bytes": 0,
      "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "path": "colabfold/data/complex_manifest.json",
      "bytes": 763,
      "source_sha256": "fa58017a4f6964f391a0782e93ea1a60a580b73145250fc103380fa319c23593"
    },
    {
      "path": "colabfold/data/sources.json",
      "bytes": 6116,
      "source_sha256": "1004dd2b9badd383344dcfaefdc80a44cab1f70cd29ddbe1fd064fa764dc72e6"
    },
    {
      "path": "colabfold/data/input_manifest.json",
      "bytes": 2679,
      "source_sha256": "4984b41e06e061e5bbfac95bd693c73e080c2be5b825c71d21d79d9c5a237ba9"
    },
    {
      "path": "scvi/scripts/benchmark.py",
      "bytes": 18015,
      "source_sha256": "a3ca63437414a72298ab23f77f7fe17e30b98a57bbe3138398dcb9a08b62a3fc"
    },
    {
      "path": "scvi/scripts/setup_env.py",
      "bytes": 1202,
      "source_sha256": "61feab41808751423b12941a6b0ac7b20705ac6a08208a347e73580f13068414"
    },
    {
      "path": "scvi/scripts/launch.py",
      "bytes": 1280,
      "source_sha256": "55e8b33b69a2a08ed83d2d236d1aae2b5df23dbc6296cd55328ec6b68393462e"
    },
    {
      "path": "scvi/scripts/report.py",
      "bytes": 21865,
      "source_sha256": "f04419f65c34b21978e746673be92a3c41c8f61aaaacdd5e3f6ca09aa975f687"
    },
    {
      "path": "scvi/scripts/prepare_inputs.py",
      "bytes": 5191,
      "source_sha256": "285efb06f82f3f740e5f78a42714849b3dbe91612e3262c4d6a1743cc10f4c51"
    },
    {
      "path": "scvi/scripts/verify_reload.py",
      "bytes": 2854,
      "source_sha256": "77b5a2a1b19102254f898dc7630109ecbcf6d7bcd1dab50f8568b5202ff46d72"
    },
    {
      "path": "scvi/scripts/predict.py",
      "bytes": 1069,
      "source_sha256": "77bc0219030fc8fbd543deebcf2a91b503cd461b8465336c0f786516feaec832"
    },
    {
      "path": "scvi/scripts/download_data.py",
      "bytes": 1381,
      "source_sha256": "c9180afb6d36ddb7be409efb35c6a1356f60cb2bd73c931f80352204b14d9b8f"
    },
    {
      "path": "scvi/reports/comparison.csv",
      "bytes": 1544,
      "source_sha256": "ac1db2acfe50123f8c9531631e10739ad3dae8462eba3f6a66e281b1faf69a71"
    },
    {
      "path": "scvi/reports/metrics.json",
      "bytes": 3247,
      "source_sha256": "ee6ee43fd5385fe254ad33756a3775707cd55d368dbbab225a5151501aa2005b"
    },
    {
      "path": "scvi/reports/scvi_data_and_training.png",
      "bytes": 754405,
      "source_sha256": "ffd5712a1d1616437b742857eff9cac4dce49feb0670db75d0bb307e7781ead0"
    },
    {
      "path": "scvi/reports/final_environment.json",
      "bytes": 1981,
      "source_sha256": "41b8fb2b88a70a6b83d51bd7f7519716fb664068b3f6f44273a2e86431f0c40c"
    },
    {
      "path": "scvi/reports/Batch_memory_followup.md",
      "bytes": 4568,
      "source_sha256": "bcff2433f124896d4aca4abd74336922e39d3f5f1db00a1ad61c5e30cd120cf7"
    },
    {
      "path": "scvi/reports/scVI.md",
      "bytes": 16879,
      "source_sha256": "0128f9a73b313d699bae323ebb224faf08a2f70c7455467b382edb7a0db00a72"
    },
    {
      "path": "scvi/reports/scVI_scvi_20261027_185901_460537.md",
      "bytes": 16879,
      "source_sha256": "0128f9a73b313d699bae323ebb224faf08a2f70c7455467b382edb7a0db00a72"
    },
    {
      "path": "scvi/reports/training_input_example_metadata.csv",
      "bytes": 474,
      "source_sha256": "1cbe4538715964ae9e1cf1f15be348174330c1e8df047694ab70de56ba4bdf3b"
    },
    {
      "path": "scvi/reports/artifact_manifest.json",
      "bytes": 19757,
      "source_sha256": "4c6502fb599321ce89aa636a6f3dd2a97fa45202a260ec5849628b160f20943a"
    },
    {
      "path": "scvi/reports/latent_example.csv",
      "bytes": 830,
      "source_sha256": "f77a3e73f38938e4db8b463f6badab212fe10cd0cdb5b032207a36f47a0da597"
    },
    {
      "path": "scvi/reports/training_input_example.csv",
      "bytes": 340,
      "source_sha256": "a024784f7076f3bff66beac6d26f0307ddce1aa162185713dfc321938bd398b5"
    },
    {
      "path": "scvi/reports/batch_memory_followup.csv",
      "bytes": 1905,
      "source_sha256": "6f91b6748ebcdda430d00d8e43b13e2760e87eef2fc24872618b46e2462a732d"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/input_report.json",
      "bytes": 2779,
      "source_sha256": "2ee1f155054b2d36e14512d86551544c0679e4fd496242cadeffdf37674331ff"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_reconstruction_loss_train.csv",
      "bytes": 8566,
      "source_sha256": "6a41d4780ea797da57b00290b48f6b1bcaeae706c92ce12677939ed8c66afa3f"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_elbo_validation.csv",
      "bytes": 445,
      "source_sha256": "45c369c0fc0f0f49c673888f4f3086c8aa3feccb2ad81f97205fe7c0ed27223c"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/recovery.json",
      "bytes": 2203,
      "source_sha256": "918a6f3bf70ed5aa095d07fb759452d0f5a67e968c19c592860d60535ec929cf"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_elbo_validation.csv",
      "bytes": 453,
      "source_sha256": "8d837caa4ef5453c83c1f24d50026c3b1297d58eeb484c8d1a76112f9cf62b60"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_train_loss.csv",
      "bytes": 8534,
      "source_sha256": "7d0220c5a64112a2dc2ff96fd4ca9ba134d538519d3fb9b2f374d2b5a279f2fd"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/environment.json",
      "bytes": 2027,
      "source_sha256": "2ee81805d92336023249407207fc6bfb066255412a4e98125f95995bea0bf086"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_elbo_train.csv",
      "bytes": 8536,
      "source_sha256": "bfafb762a08ab036e86b29ac4612f08ae20b7c9ca4da1a5f5a3a69a8cb4f22bc"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_elbo_validation.csv",
      "bytes": 449,
      "source_sha256": "dbbfea3a93aacc06f83e7ee25aab959d44918b44b6022522af25420f20286e46"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_train_loss.csv",
      "bytes": 8544,
      "source_sha256": "64e0951a4760c6e098a7eba91850750c828c413151ba3027d9997b1d430da18b"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/summary.json",
      "bytes": 9762,
      "source_sha256": "b6cc488e5561a098adf1f702cd0a0a029c2c6024f210bb5abb19a4fe86ab4721"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_validation_loss.csv",
      "bytes": 447,
      "source_sha256": "f3118353aeecd5759e6b2a36b6759d5091d86607809aa6731567b0137569d982"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/pip_freeze.txt",
      "bytes": 6710,
      "source_sha256": "6d77870d299fe68c4f65c06e459c304cf2ba4edccc4a32c650168aa4d721785e"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_metrics.json",
      "bytes": 2046,
      "source_sha256": "b3ff95be8bfe13055faf668c2f9b646f8d9956a0f7a4af338076901ecf4ff842"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_kl_local_train.csv",
      "bytes": 8914,
      "source_sha256": "4ac55b13644aa1063dfd4cb3e1306895e9fb754b42880ecd87426cd31001a8fd"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_kl_global_validation.csv",
      "bytes": 182,
      "source_sha256": "baa28bc10aa8ba80a2b0a5f99491f55a465543b888bcf81cb0475cf0dc1a5dc9"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/resources.csv",
      "bytes": 40137,
      "source_sha256": "e8ff3d909f419c463639725c9daaeba366319a9495bcef95289ff52df7d60f16"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_kl_global_validation.csv",
      "bytes": 182,
      "source_sha256": "baa28bc10aa8ba80a2b0a5f99491f55a465543b888bcf81cb0475cf0dc1a5dc9"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_kl_global_train.csv",
      "bytes": 3112,
      "source_sha256": "699630259a316ad26940bf1133c23a3537ac37a9d676949c0488192ec0ac910e"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_reconstruction_loss_train.csv",
      "bytes": 8524,
      "source_sha256": "09f59e70383fdf519be3cd0a1389eb2ddc561b632f830a63abb2b62afb7d4b8b"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_kl_local_validation.csv",
      "bytes": 471,
      "source_sha256": "9a5f87382f1ca744a42a043f0abdae054cbf1d6f426800ca04689f25bfdd0007"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_test_latent.csv",
      "bytes": 171981,
      "source_sha256": "949be70be94a368797d7dbc338bba6673a0066bdd5061707ee7d68a4cd8cb01d"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_validation_loss.csv",
      "bytes": 449,
      "source_sha256": "33f417a31a3be1509e55f7e260a78ae2d3d5dd277ba447be43bf72c8bee2b75d"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/exit_code.json",
      "bytes": 120,
      "source_sha256": "729b0faa7c34db672c154e202baea476c55ebacd2ca78499e1b68c054ebc28db"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_kl_local_validation.csv",
      "bytes": 476,
      "source_sha256": "aa71a53b5942cf5f5335adf302b03c156beb2daba98aab30d076a3e08e31fab7"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/sources.json",
      "bytes": 2930,
      "source_sha256": "41f27de5778ecdabf1be3e1c81d4d21c5c8f8d1250ba9c7b8d35e989fadf5555"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/command.json",
      "bytes": 423,
      "source_sha256": "6b6118d327a7aa9e1b0635246dd0ea02247eae2d4095e04a0d15d2c04fc4d538"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_kl_local_validation.csv",
      "bytes": 468,
      "source_sha256": "fde13800cab26f0bb5dfdcf70121b461765912f09b5d4aabb7f92b4a93c55960"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_metrics.json",
      "bytes": 2108,
      "source_sha256": "fa097928f60aa3261d36b764de1ce43016759ece4682dc99cca4b2a43b8fa780"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_test_latent.csv",
      "bytes": 172928,
      "source_sha256": "254cc8512504c50665861363d922c790afa54bf889cc24c6d32cc74c95ec5579"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_epoch_timings.json",
      "bytes": 53857,
      "source_sha256": "71fb443045aa825d2714612ffe817a2669a28848d38d2411ae02448db75e0f4f"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_reconstruction_loss_validation.csv",
      "bytes": 460,
      "source_sha256": "d130ea243ec35d1f5e12a8aaabe955404a35925c0feaaa8a7ca41e765924b9bc"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_epoch_timings.json",
      "bytes": 54184,
      "source_sha256": "b7e8126330df7cf090c1b863c75fb16946232de51ba6c2b8b4f04a23bbb80249"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_kl_weight.csv",
      "bytes": 9029,
      "source_sha256": "2694594d3585e5faada04e691f56710f2ac0526af588d03946c678bb0c2934b6"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_reconstruction_loss_validation.csv",
      "bytes": 468,
      "source_sha256": "2ab4355ccd1301a8c4106d7195026565adfaeb171ea4accac7a554f81c8509e6"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_validation_loss.csv",
      "bytes": 445,
      "source_sha256": "d854b0d7bca30adf05a037edc16d8b82e89f3f6768afed8d9d91c954eb71144f"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_kl_global_train.csv",
      "bytes": 3112,
      "source_sha256": "699630259a316ad26940bf1133c23a3537ac37a9d676949c0488192ec0ac910e"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_elbo_train.csv",
      "bytes": 8518,
      "source_sha256": "b115f3968dac01801ae1178b41d622851ca4f5934d69323b5a9cbeeed87af68c"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/splits.json",
      "bytes": 180822,
      "source_sha256": "ce80f266283a1271e221f539aac7bdaa5b46b0203f19468b37e078fbb9864850"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_train_loss.csv",
      "bytes": 8510,
      "source_sha256": "3ad639efbe0d67895b073d564b12cbc8a707c9f691739b2df1bea9f26a59402c"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_kl_local_train.csv",
      "bytes": 8961,
      "source_sha256": "b2b692f36ee23695f3230f9d1cb98fdb7651ca458728c47fb8f946e1222c8963"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_metrics.json",
      "bytes": 2107,
      "source_sha256": "6edc30ad47c7371e1254e98699d454e1bad4ae5c3f36069fd003d71f5be083ad"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_kl_global_train.csv",
      "bytes": 3112,
      "source_sha256": "699630259a316ad26940bf1133c23a3537ac37a9d676949c0488192ec0ac910e"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_reconstruction_loss_validation.csv",
      "bytes": 459,
      "source_sha256": "618db6b1eaa56c95a04b1276df7ecc9745dc5343d454dd5cc5544495fdb5643a"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_kl_weight.csv",
      "bytes": 7408,
      "source_sha256": "8414d2c7cd92994c5f106bab2dcdb6cd61728d2646a4e1c8adba431052acd39f"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_4096_epoch_timings.json",
      "bytes": 53971,
      "source_sha256": "801e5d2a1b42866ae04e3efe4b429cbd5f85fe03522e6963cecbff99ab448e4d"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_test_latent.csv",
      "bytes": 172383,
      "source_sha256": "cc71912f434e75df580a3c3c433d10f4b679482a2b7181ec8b0876dad3017abc"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_kl_local_train.csv",
      "bytes": 8980,
      "source_sha256": "1344e9536c5b803c5e077e75162344e72e591a7a8eb7e58064ab6a868d20b951"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_1024_kl_global_validation.csv",
      "bytes": 182,
      "source_sha256": "baa28bc10aa8ba80a2b0a5f99491f55a465543b888bcf81cb0475cf0dc1a5dc9"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_elbo_train.csv",
      "bytes": 8520,
      "source_sha256": "e410ede8866b9c0d7229c0553d5e6013ab77cf790a7487a591477c304ad82c79"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_kl_weight.csv",
      "bytes": 9029,
      "source_sha256": "2694594d3585e5faada04e691f56710f2ac0526af588d03946c678bb0c2934b6"
    },
    {
      "path": "scvi/results/scvi_20261027_185327_767988/train_10176_reconstruction_loss_train.csv",
      "bytes": 8480,
      "source_sha256": "11b4d8de68bca76e177b18ae02e858a66a3ae69d7917551d32f962be1b13d757"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/input_report.json",
      "bytes": 2779,
      "source_sha256": "2ee1f155054b2d36e14512d86551544c0679e4fd496242cadeffdf37674331ff"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_reconstruction_loss_train.csv",
      "bytes": 8566,
      "source_sha256": "6a41d4780ea797da57b00290b48f6b1bcaeae706c92ce12677939ed8c66afa3f"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_elbo_validation.csv",
      "bytes": 445,
      "source_sha256": "45c369c0fc0f0f49c673888f4f3086c8aa3feccb2ad81f97205fe7c0ed27223c"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/recovery.json",
      "bytes": 133,
      "source_sha256": "31ec822aa422c9679dc02c82ea207928e4386b03399093183b33d5880ae10e2e"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_elbo_validation.csv",
      "bytes": 453,
      "source_sha256": "8d837caa4ef5453c83c1f24d50026c3b1297d58eeb484c8d1a76112f9cf62b60"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_train_loss.csv",
      "bytes": 8534,
      "source_sha256": "7d0220c5a64112a2dc2ff96fd4ca9ba134d538519d3fb9b2f374d2b5a279f2fd"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/inference_1024_cells.csv",
      "bytes": 76217,
      "source_sha256": "8b5f797a29a3e0b100c1e432573131626e1e210b4a124da2b75248051475ad2e"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/environment.json",
      "bytes": 2027,
      "source_sha256": "f0a0882b4b1a0adfaa51db97ef36905578c48d067a35a97f64f2a5a9ed7a13b1"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_elbo_train.csv",
      "bytes": 8536,
      "source_sha256": "bfafb762a08ab036e86b29ac4612f08ae20b7c9ca4da1a5f5a3a69a8cb4f22bc"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_elbo_validation.csv",
      "bytes": 449,
      "source_sha256": "dbbfea3a93aacc06f83e7ee25aab959d44918b44b6022522af25420f20286e46"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_train_loss.csv",
      "bytes": 8544,
      "source_sha256": "64e0951a4760c6e098a7eba91850750c828c413151ba3027d9997b1d430da18b"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/summary.json",
      "bytes": 159236,
      "source_sha256": "d19228269ecd85a013080d7faa7f9b65a16ba1d1b03821b97dd38184ba978578"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_validation_loss.csv",
      "bytes": 447,
      "source_sha256": "f3118353aeecd5759e6b2a36b6759d5091d86607809aa6731567b0137569d982"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/pip_freeze.txt",
      "bytes": 6710,
      "source_sha256": "6d77870d299fe68c4f65c06e459c304cf2ba4edccc4a32c650168aa4d721785e"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_metrics.json",
      "bytes": 2046,
      "source_sha256": "b3ff95be8bfe13055faf668c2f9b646f8d9956a0f7a4af338076901ecf4ff842"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/inference_4096_metrics.json",
      "bytes": 26994,
      "source_sha256": "eae7f80beaa2d2f81413b6aa1a2c0038ef70d44fcf6974b3bf751b7d8c6b08c9"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_kl_local_train.csv",
      "bytes": 8914,
      "source_sha256": "4ac55b13644aa1063dfd4cb3e1306895e9fb754b42880ecd87426cd31001a8fd"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_kl_global_validation.csv",
      "bytes": 182,
      "source_sha256": "baa28bc10aa8ba80a2b0a5f99491f55a465543b888bcf81cb0475cf0dc1a5dc9"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/resources.csv",
      "bytes": 9212,
      "source_sha256": "74c51671d277feba5c82aa22861dfca65b9f617d07d3227fc1868e81b6581b20"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_kl_global_validation.csv",
      "bytes": 182,
      "source_sha256": "baa28bc10aa8ba80a2b0a5f99491f55a465543b888bcf81cb0475cf0dc1a5dc9"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_kl_global_train.csv",
      "bytes": 3112,
      "source_sha256": "699630259a316ad26940bf1133c23a3537ac37a9d676949c0488192ec0ac910e"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_reconstruction_loss_train.csv",
      "bytes": 8524,
      "source_sha256": "09f59e70383fdf519be3cd0a1389eb2ddc561b632f830a63abb2b62afb7d4b8b"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_kl_local_validation.csv",
      "bytes": 471,
      "source_sha256": "9a5f87382f1ca744a42a043f0abdae054cbf1d6f426800ca04689f25bfdd0007"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/inference_12721_cells.csv",
      "bytes": 945020,
      "source_sha256": "219f00e7629e485b42a39ed734cc9af769e2b725650af615c2b3d1700a69027e"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/loading_baselines.json",
      "bytes": 2360,
      "source_sha256": "a5737433b931be98d98acb3ece8876acc83341a175d0524ca39b9aa93fd694bc"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_test_latent.csv",
      "bytes": 171981,
      "source_sha256": "949be70be94a368797d7dbc338bba6673a0066bdd5061707ee7d68a4cd8cb01d"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_validation_loss.csv",
      "bytes": 449,
      "source_sha256": "33f417a31a3be1509e55f7e260a78ae2d3d5dd277ba447be43bf72c8bee2b75d"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/exit_code.json",
      "bytes": 120,
      "source_sha256": "07ba8422e2586dff778e3cbd41521f20a523ab661a1688ef8615d65e170fe6ac"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/inference_4096_cells.csv",
      "bytes": 304241,
      "source_sha256": "9935c6e66006786baaaa81b52f205cc06141d2c7f76dceaf3ce6025d6386aa02"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_kl_local_validation.csv",
      "bytes": 476,
      "source_sha256": "aa71a53b5942cf5f5335adf302b03c156beb2daba98aab30d076a3e08e31fab7"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/sources.json",
      "bytes": 2930,
      "source_sha256": "41f27de5778ecdabf1be3e1c81d4d21c5c8f8d1250ba9c7b8d35e989fadf5555"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/command.json",
      "bytes": 423,
      "source_sha256": "816c995f9f111b8e249819531dfb04a95357df47e321d4549f4a66b428a1e904"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_kl_local_validation.csv",
      "bytes": 468,
      "source_sha256": "fde13800cab26f0bb5dfdcf70121b461765912f09b5d4aabb7f92b4a93c55960"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_metrics.json",
      "bytes": 2108,
      "source_sha256": "fa097928f60aa3261d36b764de1ce43016759ece4682dc99cca4b2a43b8fa780"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/integration_metrics.json",
      "bytes": 734,
      "source_sha256": "534cc987a759ab3507865d4b0a8b9aa642c6b15966839f43b9231c736081184e"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_test_latent.csv",
      "bytes": 172928,
      "source_sha256": "254cc8512504c50665861363d922c790afa54bf889cc24c6d32cc74c95ec5579"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_epoch_timings.json",
      "bytes": 53857,
      "source_sha256": "71fb443045aa825d2714612ffe817a2669a28848d38d2411ae02448db75e0f4f"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/loading_baselines_annotated.json",
      "bytes": 2522,
      "source_sha256": "3f5cc869ff1726241325e03e8df31969bd960a490c11ba353891fef98ad2fcad"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_reconstruction_loss_validation.csv",
      "bytes": 460,
      "source_sha256": "d130ea243ec35d1f5e12a8aaabe955404a35925c0feaaa8a7ca41e765924b9bc"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_epoch_timings.json",
      "bytes": 54184,
      "source_sha256": "b7e8126330df7cf090c1b863c75fb16946232de51ba6c2b8b4f04a23bbb80249"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/inference_12721_metrics.json",
      "bytes": 9563,
      "source_sha256": "105f601522d4a7d4b5d7f8a72976e8f3812758489ae751e83bdcb96369ed8711"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_kl_weight.csv",
      "bytes": 9029,
      "source_sha256": "2694594d3585e5faada04e691f56710f2ac0526af588d03946c678bb0c2934b6"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_reconstruction_loss_validation.csv",
      "bytes": 468,
      "source_sha256": "2ab4355ccd1301a8c4106d7195026565adfaeb171ea4accac7a554f81c8509e6"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_validation_loss.csv",
      "bytes": 445,
      "source_sha256": "d854b0d7bca30adf05a037edc16d8b82e89f3f6768afed8d9d91c954eb71144f"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_kl_global_train.csv",
      "bytes": 3112,
      "source_sha256": "699630259a316ad26940bf1133c23a3537ac37a9d676949c0488192ec0ac910e"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_elbo_train.csv",
      "bytes": 8518,
      "source_sha256": "b115f3968dac01801ae1178b41d622851ca4f5934d69323b5a9cbeeed87af68c"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/splits.json",
      "bytes": 180822,
      "source_sha256": "ce80f266283a1271e221f539aac7bdaa5b46b0203f19468b37e078fbb9864850"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_train_loss.csv",
      "bytes": 8510,
      "source_sha256": "3ad639efbe0d67895b073d564b12cbc8a707c9f691739b2df1bea9f26a59402c"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/inference_1024_metrics.json",
      "bytes": 93427,
      "source_sha256": "9554ed87f8ada1ba5eb9063b5c6571746666c92b48edfe1307882a1729aef805"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_kl_local_train.csv",
      "bytes": 8961,
      "source_sha256": "b2b692f36ee23695f3230f9d1cb98fdb7651ca458728c47fb8f946e1222c8963"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_metrics.json",
      "bytes": 2107,
      "source_sha256": "6edc30ad47c7371e1254e98699d454e1bad4ae5c3f36069fd003d71f5be083ad"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_kl_global_train.csv",
      "bytes": 3112,
      "source_sha256": "699630259a316ad26940bf1133c23a3537ac37a9d676949c0488192ec0ac910e"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_reconstruction_loss_validation.csv",
      "bytes": 459,
      "source_sha256": "618db6b1eaa56c95a04b1276df7ecc9745dc5343d454dd5cc5544495fdb5643a"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_kl_weight.csv",
      "bytes": 7408,
      "source_sha256": "8414d2c7cd92994c5f106bab2dcdb6cd61728d2646a4e1c8adba431052acd39f"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_4096_epoch_timings.json",
      "bytes": 53971,
      "source_sha256": "801e5d2a1b42866ae04e3efe4b429cbd5f85fe03522e6963cecbff99ab448e4d"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_test_latent.csv",
      "bytes": 172383,
      "source_sha256": "cc71912f434e75df580a3c3c433d10f4b679482a2b7181ec8b0876dad3017abc"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_kl_local_train.csv",
      "bytes": 8980,
      "source_sha256": "1344e9536c5b803c5e077e75162344e72e591a7a8eb7e58064ab6a868d20b951"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_1024_kl_global_validation.csv",
      "bytes": 182,
      "source_sha256": "baa28bc10aa8ba80a2b0a5f99491f55a465543b888bcf81cb0475cf0dc1a5dc9"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_elbo_train.csv",
      "bytes": 8520,
      "source_sha256": "e410ede8866b9c0d7229c0553d5e6013ab77cf790a7487a591477c304ad82c79"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_kl_weight.csv",
      "bytes": 9029,
      "source_sha256": "2694594d3585e5faada04e691f56710f2ac0526af588d03946c678bb0c2934b6"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/train_10176_reconstruction_loss_train.csv",
      "bytes": 8480,
      "source_sha256": "11b4d8de68bca76e177b18ae02e858a66a3ae69d7917551d32f962be1b13d757"
    },
    {
      "path": "scvi/results/scvi_20261027_185901_460537/independent_reload.json",
      "bytes": 803,
      "source_sha256": "fe62cfc14cf6bec4b4da9f177725b2fa56d20539198486267b04ce881864c687"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/input_report.json",
      "bytes": 2779,
      "source_sha256": "2ee1f155054b2d36e14512d86551544c0679e4fd496242cadeffdf37674331ff"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_reconstruction_loss_train.csv",
      "bytes": 8566,
      "source_sha256": "6a41d4780ea797da57b00290b48f6b1bcaeae706c92ce12677939ed8c66afa3f"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_train_loss.csv",
      "bytes": 8534,
      "source_sha256": "7d0220c5a64112a2dc2ff96fd4ca9ba134d538519d3fb9b2f374d2b5a279f2fd"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/environment.json",
      "bytes": 2027,
      "source_sha256": "bf837d930aec612aa6d17db74b2a7d4cbb28f27f33d096cced75bedf39baad94"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_elbo_train.csv",
      "bytes": 8536,
      "source_sha256": "bfafb762a08ab036e86b29ac4612f08ae20b7c9ca4da1a5f5a3a69a8cb4f22bc"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_elbo_validation.csv",
      "bytes": 449,
      "source_sha256": "dbbfea3a93aacc06f83e7ee25aab959d44918b44b6022522af25420f20286e46"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/summary.json",
      "bytes": 4026,
      "source_sha256": "75ec92ad4075fb3523267c56bf4ea98ed7be7d4b9cb48d04d44dba42d3b94538"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/pip_freeze.txt",
      "bytes": 6710,
      "source_sha256": "6d77870d299fe68c4f65c06e459c304cf2ba4edccc4a32c650168aa4d721785e"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_metrics.json",
      "bytes": 2046,
      "source_sha256": "b3ff95be8bfe13055faf668c2f9b646f8d9956a0f7a4af338076901ecf4ff842"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_kl_local_train.csv",
      "bytes": 8914,
      "source_sha256": "4ac55b13644aa1063dfd4cb3e1306895e9fb754b42880ecd87426cd31001a8fd"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/resources.csv",
      "bytes": 4267,
      "source_sha256": "ad2b668862a6b22adc3d90064973050cc8d7235df9a4d6124ac45b4251edeaf3"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_kl_local_validation.csv",
      "bytes": 471,
      "source_sha256": "9a5f87382f1ca744a42a043f0abdae054cbf1d6f426800ca04689f25bfdd0007"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_test_latent.csv",
      "bytes": 171981,
      "source_sha256": "949be70be94a368797d7dbc338bba6673a0066bdd5061707ee7d68a4cd8cb01d"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/exit_code.json",
      "bytes": 120,
      "source_sha256": "411edb8519d3ea3944de66fb643f09bfd8ed2620fa7df43cbb5fcb36fcf777bc"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/sources.json",
      "bytes": 2930,
      "source_sha256": "41f27de5778ecdabf1be3e1c81d4d21c5c8f8d1250ba9c7b8d35e989fadf5555"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/command.json",
      "bytes": 395,
      "source_sha256": "261aa52bed0679e89a25047e5521169ca40b97da6ab63224b6ed3668e8bd2ce4"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_reconstruction_loss_validation.csv",
      "bytes": 460,
      "source_sha256": "d130ea243ec35d1f5e12a8aaabe955404a35925c0feaaa8a7ca41e765924b9bc"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_epoch_timings.json",
      "bytes": 54184,
      "source_sha256": "b7e8126330df7cf090c1b863c75fb16946232de51ba6c2b8b4f04a23bbb80249"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_validation_loss.csv",
      "bytes": 445,
      "source_sha256": "d854b0d7bca30adf05a037edc16d8b82e89f3f6768afed8d9d91c954eb71144f"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_kl_global_train.csv",
      "bytes": 3112,
      "source_sha256": "699630259a316ad26940bf1133c23a3537ac37a9d676949c0488192ec0ac910e"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/splits.json",
      "bytes": 180822,
      "source_sha256": "ce80f266283a1271e221f539aac7bdaa5b46b0203f19468b37e078fbb9864850"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/smoke.json",
      "bytes": 151,
      "source_sha256": "2f3b69f4def141e1bd7cf19926fee73578d53a01297ee21902bab7d00c19114f"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_kl_weight.csv",
      "bytes": 7408,
      "source_sha256": "8414d2c7cd92994c5f106bab2dcdb6cd61728d2646a4e1c8adba431052acd39f"
    },
    {
      "path": "scvi/results/scvi_20261027_184921_028161/train_1024_kl_global_validation.csv",
      "bytes": 182,
      "source_sha256": "baa28bc10aa8ba80a2b0a5f99491f55a465543b888bcf81cb0475cf0dc1a5dc9"
    },
    {
      "path": "scvi/results/setup_20261027_093024/base_versions.json",
      "bytes": 168,
      "source_sha256": "688cb2909954a54aa124eb71f6fd106a9bfec6022f30f2c735237bf74b4e3227"
    },
    {
      "path": "scvi/results/setup_20261027_093024/exit_code.json",
      "bytes": 20,
      "source_sha256": "8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd"
    },
    {
      "path": "scvi/results/setup_20261027_093024/install_command.json",
      "bytes": 169,
      "source_sha256": "02a89568a00cc9c2dfe1e4ea1e7ddfbb7ae40dbd35dbd1e4d5153c9d3be2d05b"
    },
    {
      "path": "scvi/results/inputs_20261027_093652_887319/input_report.json",
      "bytes": 2779,
      "source_sha256": "2ee1f155054b2d36e14512d86551544c0679e4fd496242cadeffdf37674331ff"
    },
    {
      "path": "scvi/results/inputs_20261027_093652_887319/cells.csv",
      "bytes": 894111,
      "source_sha256": "e1be5e8eb44f8264890d0825d0b8d506264f9759a3ef75b004cedfe1664b9f05"
    },
    {
      "path": "scvi/results/inputs_20261027_093652_887319/environment.json",
      "bytes": 1235,
      "source_sha256": "d0a10c4a73473d6b3f5bf1ea4842d2b4fba6572b4e80f91b9007ebfea7de45e1"
    },
    {
      "path": "scvi/results/inputs_20261027_093652_887319/pip_freeze.txt",
      "bytes": 6710,
      "source_sha256": "6d77870d299fe68c4f65c06e459c304cf2ba4edccc4a32c650168aa4d721785e"
    },
    {
      "path": "scvi/results/inputs_20261027_093652_887319/genes.csv",
      "bytes": 54700,
      "source_sha256": "e02802eaa9147682dfe2ca12c460ff2b89d9468e892d90331ef70c94dbdf3a85"
    },
    {
      "path": "scvi/results/inputs_20261027_093652_887319/splits.json",
      "bytes": 180822,
      "source_sha256": "ce80f266283a1271e221f539aac7bdaa5b46b0203f19468b37e078fbb9864850"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch256_command.json",
      "bytes": 791,
      "source_sha256": "bfe33b67fc4837ebc62e34360c25da85ac96c730d94fd0f787b7b2489a580490"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch4096_command.json",
      "bytes": 793,
      "source_sha256": "93a2745e486ced3cefb0b83e3b2c2fb1d56a9b1586f3ba710d3be1fb5fba0e10"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/summary.json",
      "bytes": 10722,
      "source_sha256": "09791a73bf2815d7740ec8c313f795628175cf1ba4eb0a427fcd088c012ed2a9"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch8192_command.json",
      "bytes": 793,
      "source_sha256": "a40b1709553e325ab113512e1524bc70069aefb5db7a41f0c561e4cab6e88e64"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch512_exit.json",
      "bytes": 68,
      "source_sha256": "943b7672e07da728d134d36ed090ab98d9845618aa24422ab5fc59c4473c4907"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch128_command.json",
      "bytes": 791,
      "source_sha256": "3cef9c3c169f9156652de75d14a1bab1bd0b9c53c197362ede74eea5ce392ca7"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch10176_exit.json",
      "bytes": 68,
      "source_sha256": "ae8ea08947da4008712f898f753d49e481f5e36a0c41b71f529d6dfe6c0e934d"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch512_command.json",
      "bytes": 791,
      "source_sha256": "5009f68f2ec4bb9e7a4b54ae21c53c791590d7cf7f5459a265e4600435f17949"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch1024_exit.json",
      "bytes": 68,
      "source_sha256": "4f9baea30a6fcf892d1bf1e5f63d66c3cd8e1c1cde4fc5d2a50a4fff019bfefa"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch4096_exit.json",
      "bytes": 66,
      "source_sha256": "c9eeb3f36ea0280f8f0e0a69770452008e41efbe495b714b0a9627c1693a32f3"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch128_exit.json",
      "bytes": 68,
      "source_sha256": "dd800e5d04a6bc752a1f5b76d8010f95be61d301b000bf94e4a9cceab0fec4bd"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch256_exit.json",
      "bytes": 68,
      "source_sha256": "0682ad9a8b048c5e0c49a706da24f331f28f834f5f57f073de6d3e1a42838039"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch10176_command.json",
      "bytes": 795,
      "source_sha256": "ec3117e67d4d4a41d8863cf05dba48b211b95ef586183dfd697a8fad32578459"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch2048_exit.json",
      "bytes": 68,
      "source_sha256": "2d2fe289158f4ccbb0b272c778dfcb11fd2618fe3b5d0ebefec9e499d5e2e42d"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch8192_exit.json",
      "bytes": 67,
      "source_sha256": "7c8172d1e231be3c07dd024c64cd89e65beeaa9e5eda5ee820aa6509cf9aa9df"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch1024_command.json",
      "bytes": 793,
      "source_sha256": "ecdc5e2e73958ad025de782d07682d3ec2ee234b2c2d859296ee14218f6da874"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch2048_command.json",
      "bytes": 793,
      "source_sha256": "13280797ceb97f40e710634174ba93b29308adf98d33bb8e15837ef992228a80"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_8192/summary.json",
      "bytes": 2650,
      "source_sha256": "9cfcc7098a1b97e12404279bacf4ca3a7d9651919a816dca3bc975115365386e"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_8192/resources.csv",
      "bytes": 4836,
      "source_sha256": "55c5993b3c41cb9f7c94025a4794975e41a54b6273bef3faa5e43517ba314d96"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_512/summary.json",
      "bytes": 2480,
      "source_sha256": "141090e28a1e9b22cb57e41e2116fe096847cd8f65a30785c33d5c870979706f"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_512/resources.csv",
      "bytes": 4899,
      "source_sha256": "acacd799b934ee65114067916fc02e20d3dea421330db0022aa66e9eaebca302"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_4096/summary.json",
      "bytes": 2656,
      "source_sha256": "0e2fad0f3ed2e7501f14cc8cb5c887376386d83d8d3808768d396b4c24518abe"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_4096/resources.csv",
      "bytes": 4895,
      "source_sha256": "9a2856a3d490d254ff812b2616a9a0064455cdb8ef94a937423a12c466cb9b55"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_1024/summary.json",
      "bytes": 2484,
      "source_sha256": "b01dfbbd41bed4ca858dfae2b0c9936e7194d0e1aa8f285fbc30535dd6715d7a"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_1024/resources.csv",
      "bytes": 4875,
      "source_sha256": "b8513d540729716c8709196ee9a9e4d1acde310b0ffe44bb278b5c456cfc8687"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_2048/summary.json",
      "bytes": 2480,
      "source_sha256": "952681db7240f5842b514ccdeb97e6647dc369c57dccc98ad36576111da31834"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_2048/resources.csv",
      "bytes": 4842,
      "source_sha256": "4c58401cdabce8566cbde67a45c497fda5eda931cd5e177f6fb27afdab887a36"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_256/summary.json",
      "bytes": 2465,
      "source_sha256": "9a76e5d42e0bdba29b0ade68b3c4d02a159e4a01aec970e66a6d6c9cc239fbbf"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_256/resources.csv",
      "bytes": 4903,
      "source_sha256": "02694f967f59917d3aa3406cd57dbccb40a91619b730d68eac604cb94f14a739"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_128/summary.json",
      "bytes": 2470,
      "source_sha256": "6a13585cda4dfe5f4b7fb501bde674a808e5cb426e6e9e02ad9bf8997b8da0c7"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_128/resources.csv",
      "bytes": 5128,
      "source_sha256": "ceb637be5a0a6f4b9d41bc72a94c11c595d647f5a589a3caa37b9627e1ea9e09"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_10176/summary.json",
      "bytes": 2656,
      "source_sha256": "637136fbd23a14163cafa980a64db5ecb35f3ea3a34fe59b7980df108ccb61c0"
    },
    {
      "path": "scvi/results/batch_followup_20261028_142752_517769/batch_10176/resources.csv",
      "bytes": 4879,
      "source_sha256": "442df7bfe9a155c6e3898969c56b90872dabd603ac24f687389364b261e0bb8e"
    },
    {
      "path": "scvi/data/sources.json",
      "bytes": 2930,
      "source_sha256": "41f27de5778ecdabf1be3e1c81d4d21c5c8f8d1250ba9c7b8d35e989fadf5555"
    },
    {
      "path": "scvi/models/scvi_20261027_185327_767988/train_4096/benchmark_metadata.json",
      "bytes": 2108,
      "source_sha256": "fa097928f60aa3261d36b764de1ce43016759ece4682dc99cca4b2a43b8fa780"
    },
    {
      "path": "scvi/models/scvi_20261027_185327_767988/train_10176/benchmark_metadata.json",
      "bytes": 2107,
      "source_sha256": "6edc30ad47c7371e1254e98699d454e1bad4ae5c3f36069fd003d71f5be083ad"
    },
    {
      "path": "scvi/models/scvi_20261027_184921_028161/train_1024/benchmark_metadata.json",
      "bytes": 2046,
      "source_sha256": "b3ff95be8bfe13055faf668c2f9b646f8d9956a0f7a4af338076901ecf4ff842"
    },
    {
      "path": "cellpose/scripts/benchmark.py",
      "bytes": 11779,
      "source_sha256": "92bb0bc8aa6660bb49907f9a7489e0fb0186a1e9534a4c2ac61773ffe2122064"
    },
    {
      "path": "cellpose/scripts/setup.py",
      "bytes": 914,
      "source_sha256": "60b6d7fa24925ede16a3e8d4f3d52b8d96a807e369475641e53ac88d2ccee005"
    },
    {
      "path": "cellpose/scripts/reload.py",
      "bytes": 1590,
      "source_sha256": "bd6883884c6fdf28df35379367163b07c5b50f00df9cff6348ebaa5a61b3ff38"
    },
    {
      "path": "cellpose/scripts/download_ranges.py",
      "bytes": 2053,
      "source_sha256": "e8e12fd27fdc81583f69f09f8e73cc0140214f8d206e78588938ac41dd1bbf51"
    },
    {
      "path": "cellpose/scripts/download.py",
      "bytes": 1687,
      "source_sha256": "d41375c0f1b41314cca8065c9d538d6529ac44a03675b83e054957ea8753e89d"
    },
    {
      "path": "cellpose/scripts/launch.py",
      "bytes": 1042,
      "source_sha256": "c51e34d1f8266b54455565b6225b61c3ad566af1e0c14d008bc9ae6737613fe1"
    },
    {
      "path": "cellpose/scripts/report.py",
      "bytes": 14017,
      "source_sha256": "1418a73e6c2424df48d44c6488993adb772aeccd29bfe8a6a14f6cfe93eb9d44"
    },
    {
      "path": "cellpose/scripts/prepare_inputs.py",
      "bytes": 2561,
      "source_sha256": "fc95d36e4d480267e1581dfb7a22ad837005d9913b0cd74b3cf690e312e8dcd2"
    },
    {
      "path": "cellpose/scripts/brain/benchmark.py",
      "bytes": 6940,
      "source_sha256": "bf1a7ffbbd62f210a25901f2a3fc63c2ba6ea95a41e886c33cc980856e13cf79"
    },
    {
      "path": "cellpose/scripts/brain/audit.py",
      "bytes": 3345,
      "source_sha256": "0800b4eba9381146a96f8f3a83e7ba6fb13e4d16f8af2c46e58bf7621a27b9f4"
    },
    {
      "path": "cellpose/scripts/brain/download.py",
      "bytes": 1912,
      "source_sha256": "db6a3d031b4a69c6d478c98d47005a8c4ebb6d86dba455332c89d203975e0f4d"
    },
    {
      "path": "cellpose/scripts/brain/launch.py",
      "bytes": 1424,
      "source_sha256": "6a129f4bbd1b4f9c7b53a8b296bfa8dbf587258701fd8380bec6cdea03366bbf"
    },
    {
      "path": "cellpose/scripts/brain/report.py",
      "bytes": 8033,
      "source_sha256": "f450e53ccb5d0f977deb9c331c02639f62e1328d085b4a3f4af18a5be87a7566"
    },
    {
      "path": "cellpose/scripts/brain/plot.py",
      "bytes": 1194,
      "source_sha256": "e312391b7ab33c4d293fc352b41898d1cd88561c5d3daad6dbb496f696cec6f4"
    },
    {
      "path": "cellpose/scripts/batch_followup/launch.py",
      "bytes": 4273,
      "source_sha256": "e33d79d4d385e1b831c4991c024816c49c041892427e7eabb8db1b2bcb5a99f2"
    },
    {
      "path": "cellpose/scripts/batch_followup/report.py",
      "bytes": 5963,
      "source_sha256": "1696a39a4bf09cd93f4be02df32e86efb9d5940501b91f2d77fc7dc6e8f8f3c3"
    },
    {
      "path": "cellpose/scripts/batch_followup/worker.py",
      "bytes": 3893,
      "source_sha256": "dbaf2034b557ac3f6e7b44c1af00054f597788f14e9d2ff53d6f42a80c7ce154"
    },
    {
      "path": "cellpose/scripts/batch_followup/prepare.py",
      "bytes": 2155,
      "source_sha256": "18524351c8e37842e56f80af19fad5f73724eec977c39b41d9d03e7f796b81c5"
    },
    {
      "path": "cellpose/reports/Cellpose.md",
      "bytes": 11615,
      "source_sha256": "e0bcd57a23f84fba0ece3e4b17bf3c3d32c100049b76d967217fc9490d3aa357"
    },
    {
      "path": "cellpose/reports/comparison.csv",
      "bytes": 1479,
      "source_sha256": "f9dacc8d3cbcae0e9dc1b5de5db3c5985c7bee5a3078f4cf4881acc7a2474001"
    },
    {
      "path": "cellpose/reports/prediction_example.png",
      "bytes": 116214,
      "source_sha256": "e6c4cfa7e1159ea407bd14b2ced6f810804d9e29bfae8795525e7f4c1d3c6f91"
    },
    {
      "path": "cellpose/reports/Cellpose_brain_NucMM.md",
      "bytes": 6277,
      "source_sha256": "ddf65faa3b6e71de4d0b6d1db0b87a9e6f44847979da2bd5e9d99c09996d46cd"
    },
    {
      "path": "cellpose/reports/input_example.png",
      "bytes": 613808,
      "source_sha256": "41c01e3f00b35294ff7173d295a177fee1d90454e83ea438f00c0264b8541e26"
    },
    {
      "path": "cellpose/reports/brain_nucmm_comparison.csv",
      "bytes": 1046,
      "source_sha256": "7785d48fc15c844a055fdd19240d7e94c4f84cabe2b708ec4f64ed3c11272e60"
    },
    {
      "path": "cellpose/reports/Cellpose_cellpose_20261027_193331_947064.md",
      "bytes": 11615,
      "source_sha256": "e0bcd57a23f84fba0ece3e4b17bf3c3d32c100049b76d967217fc9490d3aa357"
    },
    {
      "path": "cellpose/reports/batch_memory_comparison.csv",
      "bytes": 1666,
      "source_sha256": "9906c7ca3bef3a46fedb93066cc9e66e74a208f95a8dcb7d2ba0a405d30fb986"
    },
    {
      "path": "cellpose/reports/Cellpose_batch_memory.md",
      "bytes": 4781,
      "source_sha256": "77a713dcb61fda6994ec5f4cf6ec80a56bb4d94781b5469a8f93e9ba42d746ca"
    },
    {
      "path": "cellpose/reports/training_loss.png",
      "bytes": 94175,
      "source_sha256": "61d15e740d88a7dc50d2db7655c9ac7832628d655d259653c7970e2ae7914bdf"
    },
    {
      "path": "cellpose/results/environment_packages.txt",
      "bytes": 8016,
      "source_sha256": "e98e9dbc88d2c40c1d60ec56dde406c5269e5f1dd6a5ce216bfb5c9f10163ab0"
    },
    {
      "path": "cellpose/results/brain_nucmm_20261029_110603_965442/exit.json",
      "bytes": 52,
      "source_sha256": "fd37030662d79d677652eb9fc92e685277be2020ce4bcb59bbb6492013bc9be2"
    },
    {
      "path": "cellpose/results/brain_nucmm_20261029_110603_965442/input_audit.json",
      "bytes": 6828,
      "source_sha256": "c78d401249331ebad5ccfb96bb07f7a538797d2efe37499ef5236468aa40879c"
    },
    {
      "path": "cellpose/results/brain_nucmm_20261029_110603_965442/forward_calls.json",
      "bytes": 96205,
      "source_sha256": "d9ccf25472233334b83bbbb5fcd84fcbd5f77382ce70827beebd36af86a9e95b"
    },
    {
      "path": "cellpose/results/brain_nucmm_20261029_110603_965442/brain_prediction_examples.png",
      "bytes": 795767,
      "source_sha256": "af94068781df90545d588312aa33407737e2c0a6e87af21f5c17354f83455c9c"
    },
    {
      "path": "cellpose/results/brain_nucmm_20261029_110603_965442/summary.json",
      "bytes": 33541,
      "source_sha256": "d032706d30d39c0714418d0206dc127bbb35a3ef73e8d2b6d8ee54686a9193f8"
    },
    {
      "path": "cellpose/results/brain_nucmm_20261029_110603_965442/resources.csv",
      "bytes": 49985,
      "source_sha256": "36132e79f03890482c1f8b8cbd98aa3bdcc15f9b9ffa8f04a7e1f491d5b876ad"
    },
    {
      "path": "cellpose/results/brain_nucmm_20261029_110603_965442/plot_exit.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "cellpose/results/brain_nucmm_20261029_110603_965442/command.json",
      "bytes": 603,
      "source_sha256": "219d58cf42de64cce836d967a518f1052a112cc92773d02eafa9f010607d15fc"
    },
    {
      "path": "cellpose/results/brain_nucmm_20261029_110603_965442/stage_times.json",
      "bytes": 56592,
      "source_sha256": "42780c6a08ff3a5c6cab6a8ce5afd536ef5a21eb6f9831db6203fb59d9140fea"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193331_947064/forward_calls.json",
      "bytes": 126213,
      "source_sha256": "e93e4ebf5fa79181d6351499e5f07135dadeaba28e1f595e76c8e43c32e6a1e3"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193331_947064/summary.json",
      "bytes": 23437,
      "source_sha256": "02ecf105c8fb80d110007f456a2a0d1fd3b770abe5dd893019a68fada5e2e3d1"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193331_947064/reload.json",
      "bytes": 314,
      "source_sha256": "f98a6e6d5ee655bcf5d364da6d55f3ee656c6b865b9e3e27b05c36e9cd3065bd"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193331_947064/reload_exit_code.json",
      "bytes": 58,
      "source_sha256": "70d2385400b7130f9f9235f77a24f3674e09cfba834af2e20882c0bee277c62c"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193331_947064/resources.csv",
      "bytes": 151390,
      "source_sha256": "51f12102c3af36b93eb6526d273be8e6d36d5ecd6c43dd4184eea4c3ccce1768"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193331_947064/reload_command.json",
      "bytes": 394,
      "source_sha256": "b37754806a246ac1970bcb28dd808ade28203c1d4f28767f1e7d350c8fe28ba1"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193331_947064/exit_code.json",
      "bytes": 58,
      "source_sha256": "03a72811033bc348b29b28bb7adaa547da3d065bd9fb67a7b1f026e4e48cc225"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193331_947064/command.json",
      "bytes": 397,
      "source_sha256": "e16fd8eb865a73c4e5b369d4411dc48c4dc22d08169b1732444d1fd2b2c4394f"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193331_947064/events.json",
      "bytes": 10989,
      "source_sha256": "b32af92c2b1b2c49b233896d66d322c528d217a9f23c332cd93b64defe71d879"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193158_772742/forward_calls.json",
      "bytes": 10440,
      "source_sha256": "1bcbdd503785878ea9ac981d5259863382932c134fa6ad9cc0f96f737ed76c28"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193158_772742/summary.json",
      "bytes": 3665,
      "source_sha256": "9bde3e8a4f0ba431967b5c070944429ee71ff68ffa8678de44be48bd501d6040"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193158_772742/resources.csv",
      "bytes": 3624,
      "source_sha256": "e274ab66a3564325534f54287f1fef6ba34ae0d7f7524d2b7e3a8ac3e8cff0e6"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193158_772742/exit_code.json",
      "bytes": 58,
      "source_sha256": "b9d31493cc3c49e8ae6f8938034a925ff9cb307f0b18eab8191106a00d9f6663"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193158_772742/command.json",
      "bytes": 397,
      "source_sha256": "fb93bcdd3e813e1c786353ac8c2e75d151c88fcb4faeb532f9187808c811b36a"
    },
    {
      "path": "cellpose/results/cellpose_20261027_193158_772742/events.json",
      "bytes": 651,
      "source_sha256": "6b4fab138f02b50f16758dcaaa6960e86bbb4a78992db78cad3d43ec29d6db4c"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch48_exit.json",
      "bytes": 73,
      "source_sha256": "c75032c43eca473e43de622e3ddf35ec7795b7fd8c542d8b24bd27b176a03ba2"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/exit.json",
      "bytes": 85,
      "source_sha256": "02dcfd872efb66e69004da158fa4047360903b6e283f35307233c69be962de07"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch384_command.json",
      "bytes": 646,
      "source_sha256": "19a4caa649738dc0e2659c8d8599016dbf33dbe7192e1cac28358c7975180d75"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/input_audit.json",
      "bytes": 94025,
      "source_sha256": "38ac5c793332d826aaa5737099e7f14113c7e49043c569da91ae7a78ea0c8c89"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/prepare_command.json",
      "bytes": 597,
      "source_sha256": "a1c2e0a2578be4be678623b12cbdb5ec60232dba8b1a914f4a1d1ecb8b7f6b75"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch12_exit.json",
      "bytes": 72,
      "source_sha256": "45c9f8edaaa0d8994cdd4dba3fce6ab5692d2816147c859fec46a9533bfdba8e"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/prepare_exit.json",
      "bytes": 72,
      "source_sha256": "c49aa2b420c09d443049a147102cbb475244a6322640b1df621481d9d04b1157"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/summary.json",
      "bytes": 18397,
      "source_sha256": "cf3926acc58a92753cf9afce2c2e505ae57c176d626b35debd3bae1001ec2ac8"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch192_exit.json",
      "bytes": 70,
      "source_sha256": "9796cee3606954b669b2badfb4fd7cf399c771f464ab1c86bc5e6d89268baabe"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch192_command.json",
      "bytes": 646,
      "source_sha256": "8d936ea50e42dba98fda52120ec8665bdbf6a190a1196a992782222160f4cd80"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch96_command.json",
      "bytes": 644,
      "source_sha256": "bc413dc16a5500ad0bd310f827a8f2b458d1fe21ef1917afaf0168f46e9fafed"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch384_exit.json",
      "bytes": 72,
      "source_sha256": "ce7aa061735def53e2bd20a1a5b7a1d5ced3dd2e11e6d908d01a66c6ca36c4f0"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch24_exit.json",
      "bytes": 73,
      "source_sha256": "8f614498cc61c5a1d1c55a9c94180f8c1804a722595e80e1fcb26129d69f258a"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch48_command.json",
      "bytes": 644,
      "source_sha256": "742ff1d1b7c41ccf6a2a2fa8fb43b5c0856a9e1d1d1ecb9794a4c9a585c651b3"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch8_command.json",
      "bytes": 642,
      "source_sha256": "75d7a000410ab3e1f9508ae2db4103c8527d5de6c90e33bc6ef322be564e74ae"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch8_exit.json",
      "bytes": 72,
      "source_sha256": "f8349a393684da800ad5ccb8d0be5d2451c9def90cf852132f84f2b5d60a653a"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch96_exit.json",
      "bytes": 73,
      "source_sha256": "f99e6e4fb12676aaf29a8b23df0775a0674ecde7da50b8f5991027787ae18d94"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch12_command.json",
      "bytes": 644,
      "source_sha256": "ddb3cf53afa4c3afe398323847482b84d928fbcd50f17359740ee70dbaf57bee"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch24_command.json",
      "bytes": 644,
      "source_sha256": "890dc66fd010ad1238f62f20ca9ba8d41e06a27c7568e3b3fd60c1fab2e1a01f"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_8/summary.json",
      "bytes": 5647,
      "source_sha256": "ad23ee28a2ad0411c92eff6634f390f071435f2f805f63d3122c231c0d6778a2"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_8/resources.csv",
      "bytes": 5863,
      "source_sha256": "246cc591d9a0891d5cfe2ed1991f43d17ccedc9c652187bf05db34eeaa11fa18"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_12/summary.json",
      "bytes": 3922,
      "source_sha256": "d3c317f57be30e83b2de5ccbd1906169f15027ee665eed1385192955e4cfff06"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_12/resources.csv",
      "bytes": 5244,
      "source_sha256": "01d25f11ec8b7f3ae71a82a21d97a9b418fe87f70d4332718cb8991411178789"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_192/summary.json",
      "bytes": 2542,
      "source_sha256": "0b727ec7072fb1afda2c93e516cb5c9fad92d949ffc237d218c5f7e9480b55a0"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_192/resources.csv",
      "bytes": 9156,
      "source_sha256": "5077d568263759ce4663ac5e957a5cc4a468b82eac1f40562f367c63c6cf1480"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_384/summary.json",
      "bytes": 2539,
      "source_sha256": "cb069695b2672260d9abea9d5fd88eebdcf25d0eaa025ab4f384f650d4732d88"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_384/resources.csv",
      "bytes": 18255,
      "source_sha256": "398a030f425cc3a073a1893815ac0da04fddca893eaf58e961f29212c8200f69"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_24/summary.json",
      "bytes": 3203,
      "source_sha256": "aade59c82fa69ef93066b8a6cc8cd6976b9f5188846ac25a191b1783af2aea22"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_24/resources.csv",
      "bytes": 5527,
      "source_sha256": "aa5d761c0b596ba742a71bb34c239da73cd78afc334095a01ee5be30481fbfc8"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_96/summary.json",
      "bytes": 2621,
      "source_sha256": "549d463e5affc4288a3cc9f071f3b1c6b32e9c3082a283ecd6ac7098321adb0a"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_96/resources.csv",
      "bytes": 6930,
      "source_sha256": "9a0e3bb3359647799b6a0e5bb3e91c78c2d68f02c34231bd7a301c43d6de1e83"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_48/summary.json",
      "bytes": 2811,
      "source_sha256": "886fa2524449f284b63a6bd30fcf14007cece000d584d8c18c77565d51963222"
    },
    {
      "path": "cellpose/results/tile_batch_20261028_134130_265230/batch_48/resources.csv",
      "bytes": 6117,
      "source_sha256": "1173a55cbcfa8d5bc33082b84f196e7890c9b6f5791903c6ae08ba1550bb010d"
    },
    {
      "path": "cellpose/results/setup_20261027_102051/exit_code.json",
      "bytes": 20,
      "source_sha256": "8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd"
    },
    {
      "path": "cellpose/results/setup_20261027_102051/command.json",
      "bytes": 167,
      "source_sha256": "ff2291d8a93156be58481bb4fbea47c3f4b6551fd44cc84ded1da33887f4084d"
    },
    {
      "path": "cellpose/data/input_audit.json",
      "bytes": 141990,
      "source_sha256": "82ab441581106a6e4b99e5aa44cc0fde1ca278a0d93cfc61737734b22d004c61"
    },
    {
      "path": "cellpose/data/sources.json",
      "bytes": 2672,
      "source_sha256": "df6b3fe619e00362866aa3c2182fe4f2f8e49464ee4eb7dfe25f604f510f9722"
    },
    {
      "path": "cellpose/data/brain_nucmm/input_audit.json",
      "bytes": 6828,
      "source_sha256": "c78d401249331ebad5ccfb96bb07f7a538797d2efe37499ef5236468aa40879c"
    },
    {
      "path": "cellpose/data/brain_nucmm/repository_metadata.json",
      "bytes": 3119,
      "source_sha256": "455dc22a95f08d4a95763e57ffb34f89eacb840a58e049e64a022a1974cf2c06"
    },
    {
      "path": "cellpose/data/brain_nucmm/download_manifest.json",
      "bytes": 6222,
      "source_sha256": "d8ff865456b772b62f3877a759f8f165dde02446edab30352399a7a8638a195b"
    },
    {
      "path": "cellpose/data/brain_nucmm/input_preview.png",
      "bytes": 578847,
      "source_sha256": "1790206f6c052336d07c7ab8327aed13b9699732b7ab96dd4e186050a16fab10"
    },
    {
      "path": "dnabert2/scripts/launch.py",
      "bytes": 998,
      "source_sha256": "50ac752612bb8c985280ea32f5c56930c31b862ab10e9bd1a488aa64333d7122"
    },
    {
      "path": "dnabert2/scripts/report.py",
      "bytes": 8429,
      "source_sha256": "a5c859f864dae6c0ca6c46831fc6619b44cca201f037c2d59dc5eaf30d13ba50"
    },
    {
      "path": "dnabert2/scripts/supplement.py",
      "bytes": 6662,
      "source_sha256": "02b6e2f8ab6f9560bfa47dfddbd74b2f8d4ac6ef2e7c03a7b0b28331e10ad95c"
    },
    {
      "path": "dnabert2/reports/comparison.csv",
      "bytes": 2069,
      "source_sha256": "9b371d0e2677282011c332a0024b8b24ac9b12a28de321e007767d02af109694"
    },
    {
      "path": "dnabert2/reports/Batch_memory_followup.md",
      "bytes": 4913,
      "source_sha256": "aae42baf7d3e58824bef953ae62d958daaec9aeadcb0231240150ee3db907653"
    },
    {
      "path": "dnabert2/reports/DNABERT2.md",
      "bytes": 7784,
      "source_sha256": "9d54e184582cf2f9fd2d0655359308f06cb856e33f93ebbb18626dfb770d1b85"
    },
    {
      "path": "dnabert2/reports/batch_memory_followup.csv",
      "bytes": 2742,
      "source_sha256": "7df0409677034e35196371c6a1b0a45a969a9496d2757f4506a649639fcd93a9"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch256_command.json",
      "bytes": 884,
      "source_sha256": "5d9ea74a1cdadc52987965c02cd15a6d3b2f138e6a9b7cafa1d43eb265af4ae7"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch384_command.json",
      "bytes": 884,
      "source_sha256": "7130681e2a8a455c8aba62afd7424896e9fe7f772f719fea6d1b88388679677f"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch32_command.json",
      "bytes": 882,
      "source_sha256": "f76f92e90ab72a49435e81c6d878e203f0cb447bf2637250c3f43cea3c74fd61"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch2_command.json",
      "bytes": 880,
      "source_sha256": "7276df22bf7530a4c5c12fb84ab2eb5e5860326258f01fd53874c11943535b31"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch2_exit.json",
      "bytes": 68,
      "source_sha256": "1ae7622c3e8d34956a85612a2cee78346850c8c572b741eeafb734464ddcca42"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/summary.json",
      "bytes": 16790,
      "source_sha256": "f3bd49e7e15a64c345d6bef0d39721604d1827186867483273898fd88f7c10f9"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch192_exit.json",
      "bytes": 68,
      "source_sha256": "a22310b02cf7a998cdf848a96e5e3b63569d7589301d993313a4373197e51502"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch1_exit.json",
      "bytes": 68,
      "source_sha256": "0a7d8e8974c43310f01ef1f5d266a2a3612429be42ea1396ceb4ee534c4b57be"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch192_command.json",
      "bytes": 884,
      "source_sha256": "5d4ba48ea477f886070f62deb652652b83cb7560dd1e82f68c9b7b469598a351"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch512_exit.json",
      "bytes": 67,
      "source_sha256": "67d9b14c30491efe02fc10d44646ee9922787ed343feea9c478d6d4f89678e9a"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch4_exit.json",
      "bytes": 68,
      "source_sha256": "0ce19470b659b8103331ad6b5f3c60721d1bd1d4f2a03993579e6c6d742ecb42"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch96_command.json",
      "bytes": 882,
      "source_sha256": "dee2de496c6be15adb459c1532fec37ff8ecec43ece6d8568f9c7508f5cbf53f"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch128_command.json",
      "bytes": 884,
      "source_sha256": "536cd529787b5caa7a24cc3c6127f4cac35edbb928b3a00d151be2bfbdb782e0"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch384_exit.json",
      "bytes": 67,
      "source_sha256": "594ae36e4ffab49992ec8c23ea385e689db9fe36ca34a7576907e55c6ae33416"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch512_command.json",
      "bytes": 884,
      "source_sha256": "035fea871abbad94a516b086d69a95217595c7ea0f5511ede969e1b41144e934"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch32_exit.json",
      "bytes": 68,
      "source_sha256": "66dcec88c5315e5a557609b7a79c97189de35b16f27d3dd90d2cfe720c1eddd3"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch128_exit.json",
      "bytes": 68,
      "source_sha256": "a8ee0683d32f4814d486687b1981278263f609042662d197987c971773e6c517"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch256_exit.json",
      "bytes": 67,
      "source_sha256": "8635f0d68f2cacead6fc7014dfab52d8f9af9cd7e8c4c5279fe76c83744c1b22"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch4_command.json",
      "bytes": 880,
      "source_sha256": "821b01455563f32bd306cf3a1befcb86854562112b80510d95e3f31cbe106556"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch64_exit.json",
      "bytes": 67,
      "source_sha256": "a7ca84781febd5db8ccc4332f55951dc98fad333ea5ae2c674c42675facfb57d"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch1_command.json",
      "bytes": 880,
      "source_sha256": "b4d28840daf2fd8e38a58b0283180301ea2c26cff98c8cdfa3330603d3ec0a4d"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch8_command.json",
      "bytes": 880,
      "source_sha256": "83e52f5118f838e46d71aaa569368f2bfbd2d0b7ee2540d163e309cdd08e03cf"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch8_exit.json",
      "bytes": 68,
      "source_sha256": "74c20978aca58c52b3bde7fcf0fae45da1a7e8f34b3d82e2bb78fa3d1342be34"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch96_exit.json",
      "bytes": 68,
      "source_sha256": "ef26633723bce46b638e82c01cc8517dd907ee537e91ef108ca805614e724ee7"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch16_exit.json",
      "bytes": 68,
      "source_sha256": "912b16a286abb6f12c162d3bb3d111018ac4dbab354fe97dce86ba22e6eb4734"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch64_command.json",
      "bytes": 882,
      "source_sha256": "755ed66922cea3331089e80dfa0520506e23634ebd1d463b02e09abdaa83b4d1"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch16_command.json",
      "bytes": 882,
      "source_sha256": "b31b16f01d4174dba19516f3d52d382879565c5768cb8dd32634d574ab5aae4c"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_32/summary.json",
      "bytes": 3863,
      "source_sha256": "0524fb893c13d4231d207adfb556550a88237e78d349694277865eef65a361c2"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_32/resources.csv",
      "bytes": 4747,
      "source_sha256": "0b9accc1d841662a82ce801e4d6108464f79e8387f1a450d65014becbcd28dc6"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_8/summary.json",
      "bytes": 3013,
      "source_sha256": "dd303037f4e3a24dad09a51fffcf66db71fe8507cf1392aeab787ac4c58b5237"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_8/resources.csv",
      "bytes": 4483,
      "source_sha256": "155d1afb97d320d5c1de7b7906bf54abb7267448ef79e8a75fb15cf6ba207f9b"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_512/summary.json",
      "bytes": 20628,
      "source_sha256": "7b973480e78e3e8a1fe7705089b1f5c1e617a0e38382c48bdcccb531a1ad6675"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_512/resources.csv",
      "bytes": 18056,
      "source_sha256": "2677e6f6fe8b860175e01db0c047145e14147e83fcde80f678534b979a7f6a4a"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_16/summary.json",
      "bytes": 3297,
      "source_sha256": "306627931a05e8e1d638e15d55e96939495a55a2d6b1da0433f05d2c82d577f6"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_16/resources.csv",
      "bytes": 4570,
      "source_sha256": "bbe44e598d906043dcd0e53b60101885e450ccc7edd72ce95ea9aecb1ce26394"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_64/summary.json",
      "bytes": 4956,
      "source_sha256": "88db6a453cac3ef32f69d3315ade9b88accc141e464aac90a2b7de041c8a7222"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_64/resources.csv",
      "bytes": 5275,
      "source_sha256": "58705086cb60686e91dc4a2eb89823749e4341634e6749abd3e8dcb8674f18c2"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_256/summary.json",
      "bytes": 11703,
      "source_sha256": "363049e0a262cee0cb8345d699a07b1ea5cd938e9e9718ace96edbbd64f0e4cb"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_256/resources.csv",
      "bytes": 9572,
      "source_sha256": "556644dc6f1c89e5417d1790a6fae6edcb0beca5a77a681cf0e98a576478dd06"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_2/summary.json",
      "bytes": 2800,
      "source_sha256": "79528b79d1b1833c35ee72ae45583778394cf5f2b2d24fc4790f4ac94de2e0cf"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_2/resources.csv",
      "bytes": 4236,
      "source_sha256": "fc2382828481b39605cc869477822b987f3701ee65f76eeffd0f1460a8ee822e"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_192/summary.json",
      "bytes": 9465,
      "source_sha256": "10a9b2c5c0a5fb4bfc7d2efa3fa7171fb79c53a1c504a042add712f9735c0938"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_192/resources.csv",
      "bytes": 7120,
      "source_sha256": "c723d74f172cd6c6035e45f429ddf67ffa6106d9fb8e5b4182363ffc34a01aef"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_4/summary.json",
      "bytes": 2861,
      "source_sha256": "dbbc3915ad6358b6e5cc2b4380c05820404e2e748d50809f69d04fcf7ce72376"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_4/resources.csv",
      "bytes": 4355,
      "source_sha256": "7d29ebc122c8631c1fea378816023ba51ce205d55945dd1b0d300784ce4a9846"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_384/summary.json",
      "bytes": 16155,
      "source_sha256": "a1ab499b9377977dae35b28c602760e89ade0a69b7f938ad794cc3d0e7c7e865"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_384/resources.csv",
      "bytes": 13674,
      "source_sha256": "7559a5c0ed8ab0d8a7684abe103c624203f43f6550d86c5518f364c1f80d43b3"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_1/summary.json",
      "bytes": 2749,
      "source_sha256": "3944354c7b2cbc29abbc8209c888fd48cd5a14b6207f8a9bf2620e68cd233d49"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_1/resources.csv",
      "bytes": 4667,
      "source_sha256": "6aedd4354764f6ac39febdf77adcd5a6bea8f4dd4839780121cc180d3143292f"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_128/summary.json",
      "bytes": 7228,
      "source_sha256": "ad65ea9fe4217b89dd02c5f498af9ee399204efd422f913c3711433d28b90e1f"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_128/resources.csv",
      "bytes": 6397,
      "source_sha256": "1dea79e8c4dbbd400d0bc9062a7a5a57937807911c76c5b089c99c8ae10e09de"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_96/summary.json",
      "bytes": 6099,
      "source_sha256": "6403d53a0a11c4394ecfe94aaf7153c4718daf4677f7a94d13368065bfdbfa5a"
    },
    {
      "path": "dnabert2/results/batch_followup_20261028_143041_359149/batch_96/resources.csv",
      "bytes": 5908,
      "source_sha256": "921d264eccedbfc4272fd70862a9c33c281a1bbf1b2a4f2a5444c0e55c1d0cf7"
    },
    {
      "path": "dnabert2/results/supplement_20261027_195301_706540/summary.json",
      "bytes": 13819,
      "source_sha256": "3b6c6a49b64b33d3398cea4a4a383347acba01b24c81f23e2a444a5b91797a5d"
    },
    {
      "path": "dnabert2/results/supplement_20261027_195301_706540/resources.csv",
      "bytes": 21409,
      "source_sha256": "0b13828309aea20bc8135b920fddff778632096202346ba35de84af59033a6b9"
    },
    {
      "path": "dnabert2/results/supplement_20261027_195301_706540/exit_code.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "dnabert2/results/supplement_20261027_195301_706540/command.json",
      "bytes": 427,
      "source_sha256": "28fce39ba5a7d5387c14ca1a21254981968d3df4c5cf1716948139d742afa706"
    },
    {
      "path": "dnabert2/results/supplement_20261027_195301_706540/verification.json",
      "bytes": 674,
      "source_sha256": "5f9e68e335ece002c58fac1367a4b27fc8f9006d099d78c614fa378cf92202e7"
    },
    {
      "path": "ntv2/scripts/benchmark.py",
      "bytes": 7687,
      "source_sha256": "be49bb893e908f3438d91e2b6431a2a25782031c258e8a4c72824f4f4060102b"
    },
    {
      "path": "ntv2/scripts/setup.py",
      "bytes": 538,
      "source_sha256": "a366ab833d086bdeffe8b27c8dae210a62f53c2721339fbd02dce08ff0218cdb"
    },
    {
      "path": "ntv2/scripts/reload.py",
      "bytes": 1382,
      "source_sha256": "4cd7bfc5003bbc767a0c501defc4fbf5e416f1cd1d109a7bf2a6583ad2821dc4"
    },
    {
      "path": "ntv2/scripts/report.py",
      "bytes": 9157,
      "source_sha256": "6dedc5616c522f8c1e0ce15ade7421f98ca9ea361787d373e8525a9b0f905657"
    },
    {
      "path": "ntv2/scripts/inference_remeasure.py",
      "bytes": 4418,
      "source_sha256": "16cbf5cc2d7f96d71cd1d2a2dc37a347d9d5fc4c9077ff7a3614a651aa95ebc4"
    },
    {
      "path": "ntv2/scripts/discover.py",
      "bytes": 725,
      "source_sha256": "a9fbd2115bef84a39db1d46e118746f0c59ae67016760723252cb8fadc381572"
    },
    {
      "path": "ntv2/scripts/finetune.py",
      "bytes": 5939,
      "source_sha256": "8b2557f1f9c116146dbb262496379dac60f5adcd1395e9b890a34eba2d9f0e87"
    },
    {
      "path": "ntv2/scripts/prepare.py",
      "bytes": 3379,
      "source_sha256": "5fb03badb6d1462dbf126094b4ba43d5d6d43465a4c42a5fb3dec95eeaff3fb2"
    },
    {
      "path": "ntv2/reports/comparison.csv",
      "bytes": 2076,
      "source_sha256": "794870509fe56a93342ac8fae98fcad70bd4986fb0dd1814b984ae6bc0dcaa66"
    },
    {
      "path": "ntv2/reports/Batch_memory_followup.md",
      "bytes": 4518,
      "source_sha256": "ce660f60e21b9cf1a5ef6eabbea3fc09a17f332bb34b3211cddca02f4439440c"
    },
    {
      "path": "ntv2/reports/NTv2.md",
      "bytes": 9274,
      "source_sha256": "1bab40e225d59a85dcce1dca88a0c2213d0775a045784e8b9ef34e455d58e2a7"
    },
    {
      "path": "ntv2/reports/batch_memory_followup.csv",
      "bytes": 1956,
      "source_sha256": "4fd3076ab50feb497167f75c08be6f37396dda861654d687009c750e130ee46f"
    },
    {
      "path": "ntv2/results/sources.json",
      "bytes": 3491,
      "source_sha256": "ec9d718268e93f262d932a04fc41c4a1946062bf9ce8a1ed886734cb65d4a740"
    },
    {
      "path": "ntv2/results/setup_exit.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch32_command.json",
      "bytes": 789,
      "source_sha256": "a8342d0a3b194126fce49311bba2211d8b4742acd99db35cd868a7fec5369694"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch2_command.json",
      "bytes": 787,
      "source_sha256": "62aed622624751d90e4ded758d73a5e647abf40dd7089d4a7de4cb59a675b908"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch2_exit.json",
      "bytes": 68,
      "source_sha256": "84309e6f043a753a8bdebb33e4336bedcb98e2281c3835e65f88e0b46ac4efe6"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/summary.json",
      "bytes": 11692,
      "source_sha256": "55686eb892a1e0f0a1c8f919330914ae3d12bf915e7bb1e9b709daee5e6910bd"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch1_exit.json",
      "bytes": 67,
      "source_sha256": "854154bb82c5a6531e1d330ef7d10ac2019191725a32917cb4fdd40fbde4df2b"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch4_exit.json",
      "bytes": 68,
      "source_sha256": "876a59587fe96ef4d5a79be976576b375689ec19e9e31c05da0418e072ac4a7a"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch96_command.json",
      "bytes": 789,
      "source_sha256": "6b7a9c747e67f2f89fc8b98869e784fcb8013a1988e66b9bb4effffda17ed0ea"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch128_command.json",
      "bytes": 791,
      "source_sha256": "64195cfd88aef528aef25c321a553dffd729015c3007f3a29f727f9c6678cb8a"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch32_exit.json",
      "bytes": 68,
      "source_sha256": "f73374be02c5354fc34c9c33b19572f8d651753f9f379fc0db6a5b9595471935"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch128_exit.json",
      "bytes": 67,
      "source_sha256": "011499c5509cca87d19241b3c6f932d69e0d1994ef15b794af4f5e8f9c011f4f"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch4_command.json",
      "bytes": 787,
      "source_sha256": "3d9411f0ed2dafe31b3d768dfbd91d7416b193c74afb16ada01fff531ec6590f"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch64_exit.json",
      "bytes": 67,
      "source_sha256": "28a61ffc6e2b77c23efc1fd0377266eec0b9981329a1c1cd7a7684c1437f5c76"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch1_command.json",
      "bytes": 787,
      "source_sha256": "60ec8ce27554da1d6c1d9184e79cbb2869dfb62fb798f43914ebb9d7c0887deb"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch8_command.json",
      "bytes": 787,
      "source_sha256": "341519bf05c3674abaf27a0ff8feee2ff9b66b8768e8c55f346f2795155c8f1a"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch8_exit.json",
      "bytes": 68,
      "source_sha256": "ac7289fa5195797256359be67e6407313a170b0059668f4810dbb544ee8ef92f"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch96_exit.json",
      "bytes": 67,
      "source_sha256": "fc83cb5baf073f2a6e011ba15af2f30a71c460cdb11ed4361a0a4f40e873e0b6"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch16_exit.json",
      "bytes": 67,
      "source_sha256": "b00fc18a35f51440fa9094cb320f3a2fbbfdae074324a320578249399958d3ce"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch64_command.json",
      "bytes": 789,
      "source_sha256": "01e1ecaa85dd479825577aea3322153312f9ec466a103c86eeb532667876fdca"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch16_command.json",
      "bytes": 789,
      "source_sha256": "926d30380833c37c983526f916308871f1f1cd9e77193a658bbcd634fc50ab71"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_32/summary.json",
      "bytes": 3869,
      "source_sha256": "124e85292253c2f872b8e708d5180a14ec290860ca4bed290c5373fbefa42fe8"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_32/resources.csv",
      "bytes": 11000,
      "source_sha256": "da49975193588d13388896a6be5933034bcd918d43cf9357bdbfa82ed5908460"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_8/summary.json",
      "bytes": 3013,
      "source_sha256": "2f994f512433af961c9a0a8ef32c31265efb4e18f3d1ec4b85bc22759113c76b"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_8/resources.csv",
      "bytes": 8661,
      "source_sha256": "ae7fc8c2ca44e1af048fd892aadc70d4fea00293693a47fdf385cc630ac6d37b"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_16/summary.json",
      "bytes": 3299,
      "source_sha256": "f3663a1f193f600cca5f673c1dd9047015db5c0bb986d9adf077e04e82ad4117"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_16/resources.csv",
      "bytes": 9686,
      "source_sha256": "16bdcc1813d6279d4357bfaeb2049c1e939d2c28c47640deff9638ec5b183e07"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_64/summary.json",
      "bytes": 4989,
      "source_sha256": "ba38588243b2c961da7fc20a055cdd4e917d7b8c9175eacec2f9aba9e6ad2aad"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_64/resources.csv",
      "bytes": 14783,
      "source_sha256": "6abb8e71868c3b3f2ef1c2703fe88b34d2f4a82adf3c132c8f7f6c38b138e5cb"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_2/summary.json",
      "bytes": 2804,
      "source_sha256": "d1f986928582d7b9ea713ea70db5320b68f1ec6afa53f9981fcf904306a1a67f"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_2/resources.csv",
      "bytes": 7549,
      "source_sha256": "c9004a254359bdb62d04a7939ad47aadea9215a5163899bae0caaee70eef864d"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_4/summary.json",
      "bytes": 2881,
      "source_sha256": "0512a37a1400478a9d23268497ef6c59392644d35baefeec5e680922cd78d5e4"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_4/resources.csv",
      "bytes": 7821,
      "source_sha256": "fb42424c76943faee9457714284b2cebe914c5e685f491fc1ef090823ce96f64"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_1/summary.json",
      "bytes": 2752,
      "source_sha256": "435df6719ae874a6a2c45e52b803449a9a8d07e6cb0f80ef6b43f650aab22592"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_1/resources.csv",
      "bytes": 8167,
      "source_sha256": "41674a50a3138e9652f2bdd317703e4e82bd02a1978c18c4593364781c0a5441"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_128/summary.json",
      "bytes": 7236,
      "source_sha256": "8439fafa33db52c2dea44afc890cad533dd35f781749a90c3f817039cbe25799"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_128/resources.csv",
      "bytes": 25712,
      "source_sha256": "b1c247e535e2b92da3174d142f12490771d95847f8d3ae62e765c75b75840775"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_96/summary.json",
      "bytes": 6106,
      "source_sha256": "3a42b0fafd3966734157c2b73bb875565c183a8fe3ae5dba1b993e339692b36b"
    },
    {
      "path": "ntv2/results/batch_followup_20261028_143713_869625/batch_96/resources.csv",
      "bytes": 20620,
      "source_sha256": "308f4ceaebb250ea242bc6a8fee1edd60f62df6a22f5cbb956e49a8b177e1cf9"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/tf32_mismatch_reload.json",
      "bytes": 155,
      "source_sha256": "92f8a1ed67e0639b5dd23f367982e4348001ac40718bd0cc2f5d59e5b7a1ffd2"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/inference_remeasure_exit.json",
      "bytes": 58,
      "source_sha256": "129517e0169e9722d97e5ecbbaab495b4395876eb18b82a4078a64d86ddc90df"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/finetune_exit.json",
      "bytes": 58,
      "source_sha256": "1de2464160fd870edc58eb6235a0e3c7bc9e417c932ba7e0d4ff54bc4d143f6f"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/inference_remeasure_command.json",
      "bytes": 481,
      "source_sha256": "2d75e42b9c9caed3793c6d253446c02f1530da14ff7b5e3537ff2f9a81bddeab"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/summary.json",
      "bytes": 23602,
      "source_sha256": "1d967cb6694a287f6294860235388636448d2baa7a0a2475539b504153dfd02d"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/resources_1793099926040094418.csv",
      "bytes": 31911,
      "source_sha256": "72f94bed83bd56400ce11e991a97424ba17ef9842815373aafcb93405632de32"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/reload.json",
      "bytes": 137,
      "source_sha256": "18d000dc6f55e611e24777972096b0da618aaf89f7f13c1a442c717de2a4143d"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/tf32_mismatch_reload_command.json",
      "bytes": 468,
      "source_sha256": "d8acd57a0875e2f5ea5eabbf285fea45472ef7c54e717369a8c4d5413064f7f1"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/test_predictions.json",
      "bytes": 24692,
      "source_sha256": "7040169e35f34992cd8c694d266c086fb49ff7d2172839211f1e49e63dae0bb7"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/resources.csv",
      "bytes": 19409,
      "source_sha256": "1fd8a7cd74735cfcaf9c717e03ce4ba5bd904ebb450f5102d3212ac21908d7ed"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/benchmark_command.json",
      "bytes": 471,
      "source_sha256": "4deee39ebeab2df553341f013f453111138a5b5b1865badc972d495ad7f47225"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/finetune_command.json",
      "bytes": 470,
      "source_sha256": "c041f16fd03aaa324cb121679ba5726d1155b620644dca792c4f7d871ef3b646"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/reload_command.json",
      "bytes": 468,
      "source_sha256": "d8acd57a0875e2f5ea5eabbf285fea45472ef7c54e717369a8c4d5413064f7f1"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/tf32_mismatch_reload_exit.json",
      "bytes": 58,
      "source_sha256": "c2659d105e568c208d36520236ba8579d1e42ad41c1151f2cbdb2ba1e1951b81"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/reload_exit.json",
      "bytes": 58,
      "source_sha256": "7cab41167633a097eb75a3dc7781167bbf069dc66639b2a6ed90a6e7a27a5044"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/benchmark_exit.json",
      "bytes": 58,
      "source_sha256": "23eb40ad01f53c75eeb3539e3e7e6ad6f89d9a6507cc211f6dd442e79b2f879c"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/summary_before_recovery.json",
      "bytes": 9822,
      "source_sha256": "4cc1dd971b8afe51e9d0376b306d1bfb855080f891f1b88dea403df2baab5921"
    },
    {
      "path": "ntv2/results/ntv2_20261027_201332_251243/summary_before_remeasurement.json",
      "bytes": 23206,
      "source_sha256": "eb11fd6f5b270961796baf24ce2105e2ebc97bb6206c9b3c31be603227a94ff3"
    },
    {
      "path": "ntv2/data/input_audit.json",
      "bytes": 14546,
      "source_sha256": "f604a2c6ed67c8c07f92702f0f98736b4165c7b83819dfb4ce7e8606ada2f37d"
    },
    {
      "path": "ntv2/models/ntv2_20261027_201332_251243/classifier/tokenizer_config.json",
      "bytes": 928,
      "source_sha256": "546e0b71a1729d81a378c192fa670bf23896c3e3d76cec6e53734d276362af75"
    },
    {
      "path": "ntv2/models/ntv2_20261027_201332_251243/classifier/config.json",
      "bytes": 1123,
      "source_sha256": "bfe3ca2332354c0cb7a6bb0ad26973a336a754521fcc0bb42684a4403e98e6f5"
    },
    {
      "path": "ntv2/models/checkpoint/tokenizer_config.json",
      "bytes": 129,
      "source_sha256": "253d338919eba938e50b776f3243cc739462c207fe64e3d2e81cc5e681bee45b"
    },
    {
      "path": "ntv2/models/checkpoint/config.json",
      "bytes": 1065,
      "source_sha256": "27e19a9e3fbba65b933bde793b503734744b5c6e376c74dcd1cb7317876efab8"
    },
    {
      "path": "clair3/scripts/benchmark.py",
      "bytes": 6709,
      "source_sha256": "4e1312610a86ee145ac0707ba0a3cb26d0e355a492de7851ea4525cef140439f"
    },
    {
      "path": "clair3/scripts/setup.py",
      "bytes": 842,
      "source_sha256": "deb5e12b435a19d7e8c8083fb70136fcece1a56cdba24734202356bd70c4b10d"
    },
    {
      "path": "clair3/scripts/audit.py",
      "bytes": 1716,
      "source_sha256": "60de3f0256af3c437887d44ec911f949547b5d311f6cc525095d23361b9ca4a0"
    },
    {
      "path": "clair3/scripts/reload.py",
      "bytes": 2072,
      "source_sha256": "1311d548e5698ddb3f660e2c7ed232adc657d8908750554b063598da1d4c60b9"
    },
    {
      "path": "clair3/scripts/launch.py",
      "bytes": 1172,
      "source_sha256": "032dc38ce8ceda6481c9938e762b6cfa50d33fac9c04b53e3e389d4773296e74"
    },
    {
      "path": "clair3/scripts/report.py",
      "bytes": 8455,
      "source_sha256": "1d088fa9f4579dafd0ff4edae1d86c28658db578ae4248497a691388092e6763"
    },
    {
      "path": "clair3/scripts/instrument.py",
      "bytes": 2499,
      "source_sha256": "9130c01daabcb4faca2e3d4437182ab38fa3449ab71d1baf07946acd875aa84c"
    },
    {
      "path": "clair3/scripts/discover_download.py",
      "bytes": 1169,
      "source_sha256": "c80f3c16ef6b26b3629d404a42669612e8fa76b72e9e2a563709914f276ab47d"
    },
    {
      "path": "clair3/scripts/batch_followup/capture.py",
      "bytes": 996,
      "source_sha256": "2d5c7a4f07920135dd918e1602af7e1c00f448aa2504961479429874d7eee39e"
    },
    {
      "path": "clair3/scripts/batch_followup/worker.py",
      "bytes": 2680,
      "source_sha256": "e860b7101a462406f629583836ca9ae616b7fd98101aa92c9fd27fd8d06b832c"
    },
    {
      "path": "clair3/scripts/batch_followup/prepare_cache.py",
      "bytes": 1271,
      "source_sha256": "6bf8fe7d725872c0e6220d4b3a66beb60ecd2a745e4290bdda49547c33452a2f"
    },
    {
      "path": "clair3/reports/comparison.csv",
      "bytes": 989,
      "source_sha256": "bb541baf3984df117af1815195811b69318fda6849902b06246d5a7b4a6e5d1b"
    },
    {
      "path": "clair3/reports/Clair3.md",
      "bytes": 22222,
      "source_sha256": "5c82e3d58b3e2bec80d9474c17cd72cff9af5540f76cc12898e1a35499b250ec"
    },
    {
      "path": "clair3/reports/Batch_memory_followup.md",
      "bytes": 4177,
      "source_sha256": "35d0e4a1a0032ea6836e9e8af237c7866fd7b6199f47679bd9d2fe2343bb0d54"
    },
    {
      "path": "clair3/reports/batch_memory_followup.csv",
      "bytes": 2851,
      "source_sha256": "07eb11a52d784e72e38fce212c1385d4245f762889c6dbda1ff7f182c565a480"
    },
    {
      "path": "clair3/results/setup_done.json",
      "bytes": 218,
      "source_sha256": "b2e304a35d4ac101639d4cc45086eb2b72b0775834013b86fbdc7a4f13c8f168"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/summary.json",
      "bytes": 48122,
      "source_sha256": "c18630e3bd08e3d956ee17a13a274f12aaad89337b565e42a6ec2d4c67494bfe"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/reload.json",
      "bytes": 918,
      "source_sha256": "aa1f550d92a6105e0024745e1b2d921b63b78651dc0441f3ee5d5628443ca863"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/resources.csv",
      "bytes": 15081,
      "source_sha256": "b36d8665a16a8af03c9638020a6f86d278b663ce41c20ded8c2aeb1b2bf43967"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/benchmark_command.json",
      "bytes": 395,
      "source_sha256": "63b2d5d641ce3e45020d7e9f6bf523732b4b447c9da0a4f1c8a1811ce3b61887"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/different_batch_reload_command.json",
      "bytes": 392,
      "source_sha256": "c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/reload_command.json",
      "bytes": 392,
      "source_sha256": "c29ea80d2f14e0b7d1df73b2a3318af7bd93a4ec639768642b76c880e3d61dc3"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/different_batch_reload_exit.json",
      "bytes": 16,
      "source_sha256": "2618b95f1f6be6bfde8537756f85b83845d039001563135b9d59d817446733eb"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/reload_exit.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/benchmark_exit.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_200000_full/metrics.json",
      "bytes": 289,
      "source_sha256": "7da775eaad454a3c54f0731afe0e901b963c710c6d5831793459186458487522"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_200000_full/exit_code.json",
      "bytes": 20,
      "source_sha256": "8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_200000_full/command.json",
      "bytes": 503,
      "source_sha256": "d7397e0f3db35c331c171335765b85ea58aa7c63a3d69be54c06971d64478dcb"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_200000_half/metrics.json",
      "bytes": 303,
      "source_sha256": "a0d5fdb1c8f66d63acfefd1b6b541d9c1839769ffa2c613428b7cfcfc776a79c"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_200000_half/exit_code.json",
      "bytes": 20,
      "source_sha256": "8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_200000_half/command.json",
      "bytes": 497,
      "source_sha256": "c509b115d1d876f16b5e112f9046ec4bf69dbc977b7c1c02db2ad269e108b227"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_100000_full/metrics.json",
      "bytes": 257,
      "source_sha256": "d93f201151657db3e64c40b8c4baabbbe92b999b51b7b5c0980a216578cec48e"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_100000_full/exit_code.json",
      "bytes": 20,
      "source_sha256": "8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_100000_full/command.json",
      "bytes": 503,
      "source_sha256": "597deb7e845b2d4e2aa0dab872effcddc07daafa692a549a7a679bd148eb315d"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_50000_full/metrics.json",
      "bytes": 241,
      "source_sha256": "99730408dae0c373ec97ba7b60adeaacea320b9d49e71c2dcb519a50d1ea202a"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_50000_full/exit_code.json",
      "bytes": 20,
      "source_sha256": "8f0c1c520d4343f54805dcdaa74287fb690f4d610186020a9343b1407654bdbd"
    },
    {
      "path": "clair3/results/clair3_20261027_203339_768667/region_50000_full/command.json",
      "bytes": 501,
      "source_sha256": "152c0bb041222e6731f4632148bbb358aeaeea1375e9693b0803079350a407cf"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-8_exit.json",
      "bytes": 68,
      "source_sha256": "ad452bcc89e0735488fd65a47a49d582267e921cd81ddf00dddc8e55e56fc52d"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/input_audit.json",
      "bytes": 4381,
      "source_sha256": "1f0d7bc8fa275ab93e2d356ca758d15eb7adcf9f5c695c19d5e947397271c22d"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1000_command.json",
      "bytes": 534,
      "source_sha256": "368e91ae82641349cb30f2671c85010d6677fb592c0f18cedca44d73c1590f28"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/prepare_command.json",
      "bytes": 481,
      "source_sha256": "c3efc3e1993a9f254815cf3fa16ea5ca15c855060a326f7d58fdf6be2560837d"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-32_exit.json",
      "bytes": 68,
      "source_sha256": "38cc59f394ecdb4282036c023ea6ee2c0678aec62c0a1473726b681a7bbfd0fa"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-8_exit.json",
      "bytes": 68,
      "source_sha256": "864f6acae984581df8477c76311fa351eab2b9e54c7c5ac6f9a8908fbe5672a5"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1_command.json",
      "bytes": 528,
      "source_sha256": "cabadd14ce2deb154356b248cdc0ce49a687f0ee963243e5c5840aba87f080af"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/prepare_exit.json",
      "bytes": 68,
      "source_sha256": "b3c20e7cabc6bc9769441cf7592bffe6c92f7699bfd5f615467c82ed179eba84"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/summary.json",
      "bytes": 24484,
      "source_sha256": "0e17689c9503b10902d5bc717b82baafa87bbddb902819002bc55fd43a19ee7e"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-4096_exit.json",
      "bytes": 68,
      "source_sha256": "a2e76577de2a1486c5ac876a3721a0418a58f8a023dd2966b73e2bed2d7bd6d4"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-128_command.json",
      "bytes": 532,
      "source_sha256": "baba4ff65e2e6872eb49be7b1139dd73af8a130f24fef43b0f6af8b90abe9afe"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-1_command.json",
      "bytes": 528,
      "source_sha256": "1d96bcc893c60356ec312399349c3043191e3f1829dc149739eee6406b4a1dc3"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-8_command.json",
      "bytes": 528,
      "source_sha256": "3abecbbfcbbb88cbc8a2093c5fd60fff08d7b95a33c1b2abdc35d2cf61d72eec"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-512_exit.json",
      "bytes": 68,
      "source_sha256": "18bd1558a57ef44adc100b4b51020c7b7f60382d3ffc9c8697873ede0cfad4e9"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-8_command.json",
      "bytes": 528,
      "source_sha256": "6ba5f550df8017c782834c7ce3588199e51d4e1ed031f22acb48e5c8b96c190f"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-512_command.json",
      "bytes": 532,
      "source_sha256": "d98e82d23d095c2b49756ca5b9cd9cd7d4b73ed6048800f0fd9ffa74a3c34ad5"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-5708_command.json",
      "bytes": 534,
      "source_sha256": "ef8c0c305cee19f98e5a6af54f8a639195d85f41d390daadfd0067c9b4cec748"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/native_pipeline_candidate_equivalence.json",
      "bytes": 348,
      "source_sha256": "a789c7891a732bf03801fc25ed5648498c24c4128dc650b027c42371c66d996a"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-32_exit.json",
      "bytes": 68,
      "source_sha256": "ad500b939a33016ec77ebac2f327b28308ac1987bad62b2d4203fdff730e1e70"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1000_exit.json",
      "bytes": 68,
      "source_sha256": "aa1a0842d63b1f1a2630560972037753bcb55c5830e021b11e5a9fa095ddb4d3"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/capture_exit.json",
      "bytes": 51,
      "source_sha256": "27a00348126d43b1cd1a97bbfe79bea21508f5f00ddf54b1a8fca9d110f781f5"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-32_command.json",
      "bytes": 530,
      "source_sha256": "9f2403182aef7010b37f05c7feb27e23f2aa00a3d5e3e79b963b10f5bb4ea783"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-2048_command.json",
      "bytes": 534,
      "source_sha256": "18fa2b832c2cff24e9054d1c11ccce292f1aeec0ce26596513ae6631ccf4c181"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-1_exit.json",
      "bytes": 66,
      "source_sha256": "aba9d0bcdb5b6703a2589fc7674577788849f69f536bb7e8ab3912b7f18b3ef6"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-128_exit.json",
      "bytes": 68,
      "source_sha256": "6e4cfe2ec79e1bff9b71a6de93ae0e4d56f4bdfc04a78a9c8c5812a120e63ba8"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-894_command.json",
      "bytes": 532,
      "source_sha256": "5c86cea7c2039efe3b1ea1dff47af9993ca9c0c6bb604faf0406d84cc104f76f"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-2048_exit.json",
      "bytes": 67,
      "source_sha256": "e4201bb4a65e82c3ae454fd2b85d952b6ec509a71fb931aced5449cad34d9f0a"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-1_exit.json",
      "bytes": 68,
      "source_sha256": "e339a3e8de3887b8728c2ced6b2b8cf73237d7ac606314f5cb2a46b2b252af1d"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-128_exit.json",
      "bytes": 68,
      "source_sha256": "16f0fe9d5555782d17ecf829bb3846797a796e6c0c467954648df726b100cf4a"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/capture_command.json",
      "bytes": 493,
      "source_sha256": "949cf824181683569eccb94e6c7158457e6a18721ece95cf8a91eed98082859b"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/host_capture_command.json",
      "bytes": 504,
      "source_sha256": "c7a879beaacc81e1faaddc3eb97eda0227fecd8020629d6b7c23fd17ca8d1f99"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-894_exit.json",
      "bytes": 68,
      "source_sha256": "f87f203debc57d8461e1c7fe96101c956639d094645bde398c9b67699d225a09"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-128_command.json",
      "bytes": 532,
      "source_sha256": "1d8572f5403c8f3617ad7c92e57d49c31fc91066234d7e3a2a5aba68c99aa6c6"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-32_command.json",
      "bytes": 530,
      "source_sha256": "c8ecf0b51fdf8170f588a59a4975837b33bdd2804ec6f3fcc7e38b6ec73a999b"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-512_exit.json",
      "bytes": 68,
      "source_sha256": "6a201a9d11551b84c9182c2df5b147fd07db7f8aa001de05a85b5bc4cb7c6d0b"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_f-512_command.json",
      "bytes": 532,
      "source_sha256": "62d319b5ce384ab7a8ac2d7814224cc9335ff26c53d2ecf956def8f245ba219d"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-4096_command.json",
      "bytes": 534,
      "source_sha256": "eebeeb98d0d87efd0970092ede2b8b3a3aa461488b77f821a49e970e6c3c9d56"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/clair3_p-5708_exit.json",
      "bytes": 67,
      "source_sha256": "fc5e6afab3476cd4fcc123cb5238ae7452b84bfd86422ba3357d99ba30581fc1"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_512/summary.json",
      "bytes": 2011,
      "source_sha256": "8fc57039093f68c0677e57ccb3b5abc8ecf2677deb610dce8ca7f9ea4f4bac34"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_512/resources.csv",
      "bytes": 4663,
      "source_sha256": "3b68ae7056f0287b3caa0c9c7873a6354bc497cb815969d52cc54841e0544fb6"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_5708/summary.json",
      "bytes": 2004,
      "source_sha256": "098f20d6a7c86a6376fc7598f18b265e91a4a35d1772eb67ffca30fc24581623"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_5708/resources.csv",
      "bytes": 4533,
      "source_sha256": "3d1bb48316d32857b04f160605f954775b3c0533d06954da9599a32ef8b0ff0c"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_894/summary.json",
      "bytes": 2001,
      "source_sha256": "4aabda174348d210e76302a6d7d207ced35e625971d56f07db51dd9bc308dc6f"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_894/resources.csv",
      "bytes": 4576,
      "source_sha256": "d1d7613dfe914ddb42fdf68204838e56a30c8c7eaf4fc6c1137c3a9f0bb15e1e"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_1/summary.json",
      "bytes": 1936,
      "source_sha256": "c4275b3ffc2b64fd31b65f234debd6d74d4843b35e9db39bda6fdbb879b5fab5"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_1/resources.csv",
      "bytes": 4574,
      "source_sha256": "399c7639793aae922b8ec3895571862c04ea48b5bafb6ebed670d0500952aef5"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_8/summary.json",
      "bytes": 1992,
      "source_sha256": "7a820c7b091325ccc02ac3b4b5c52ac8104910dce849e68a339077253ffaf6c1"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_8/resources.csv",
      "bytes": 4628,
      "source_sha256": "7f500e1d5b1148be02199b4ac2dce1c8790e3c4f64565ade547e6460eeda7a6a"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_8/summary.json",
      "bytes": 1983,
      "source_sha256": "ab918ec5aeb9e4d14bac4d29e5ca3aa5cfe0b5f42fa5b4bff27e54f00d816094"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_8/resources.csv",
      "bytes": 4659,
      "source_sha256": "325baf9325c761d86b347a4aa6d67be6c4691af3b1ed526fa438851344249a81"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_128/summary.json",
      "bytes": 2008,
      "source_sha256": "51fc5807e96d4c296bfc453b80ab7d37ccc7e628eaa2a178fb99f6974f2e2973"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_128/resources.csv",
      "bytes": 4472,
      "source_sha256": "b767ec7d39ca74c31ab06d67280ab4f82b2cf70469a395e327dde718534997f0"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_1000/summary.json",
      "bytes": 1997,
      "source_sha256": "e4ac83fe4aed61b461191019218306c966ee1bdfc2e4fd0e79e66532241a46d2"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_1000/resources.csv",
      "bytes": 4434,
      "source_sha256": "5b392e8b3d5457afd25a7da6bb2de5ba1e933d203eb7b6516a1e0cf162a48ad0"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_32/summary.json",
      "bytes": 1997,
      "source_sha256": "89d0c5a14045647a0c16e174cf34c0d51b079477769744462e9cb3afebbbf2db"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_32/resources.csv",
      "bytes": 4647,
      "source_sha256": "71f6cb0cc7f626778aa38a19d5ccb42e0c3e1bbbd44d9124e7a1296a097860e4"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_2048/summary.json",
      "bytes": 2001,
      "source_sha256": "a851b1026e8023cc91cc64ecd2078587b938de89149c485107e7c77633d66545"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_2048/resources.csv",
      "bytes": 4481,
      "source_sha256": "00a12685db3d4e4a50e0e2a8e9db2979e3cca652d251af5486941fc07bf9797e"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_512/summary.json",
      "bytes": 1999,
      "source_sha256": "d2eb97fd92a6eb49d07ac92f848c882038cfbda639d0cbb406856a141a5c27ba"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_512/resources.csv",
      "bytes": 4476,
      "source_sha256": "d6db1e4573db219faa33e437b0fd8b8a69e2f39b773ad829c366c92c9fceda5f"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_4096/summary.json",
      "bytes": 2002,
      "source_sha256": "f5f08b49acc4e5f625c2cc73802bf4733f0fd660eab134d8f3734214de202215"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_4096/resources.csv",
      "bytes": 4582,
      "source_sha256": "13720f139a3e14d724733262288bcf789fe22e262b53fd2bfadcbaa1a0267060"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_1/summary.json",
      "bytes": 1935,
      "source_sha256": "538a2749dd9eeff5d172552907511fd40507b07c0b26f41eaf2cb3d2b1323427"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_P_batch_1/resources.csv",
      "bytes": 4637,
      "source_sha256": "9ba9d3d68b45baf3051e09e58e12e92bfda6e17d015968ceb7e904c77e462003"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_128/summary.json",
      "bytes": 1998,
      "source_sha256": "43f86b0b7c1b81e80544f40c915e0a3d17982f6806845c5ef48a37165ea23cfe"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_128/resources.csv",
      "bytes": 4543,
      "source_sha256": "806980462a482bd8ae5851fb23c73ded12fb1d5e083678d132d033152173bcd9"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_32/summary.json",
      "bytes": 1987,
      "source_sha256": "7fff1fa840cff3c4c0705c1a7f50d9af07047734680812d374c35d1bd01f46c6"
    },
    {
      "path": "clair3/results/candidate_batch_20261028_185641_089149/Clair3_F_batch_32/resources.csv",
      "bytes": 4573,
      "source_sha256": "182ca6d900d27627910f1699fb674e1a8d38dd28a03a405893b9af3d562cf29e"
    },
    {
      "path": "clair3/results/clair3_20261027_203222_759031/summary.json",
      "bytes": 31743,
      "source_sha256": "01ca76f1da3fe87ceaf0b704b5c64e1e775c6e89f5add2ba368d29af64e1a031"
    },
    {
      "path": "clair3/results/clair3_20261027_203222_759031/resources.csv",
      "bytes": 400,
      "source_sha256": "7b28d48dadfd36cb0b5ca24b609b98fc8603722fc21249cd4817886e81e2be83"
    },
    {
      "path": "clair3/results/clair3_20261027_203222_759031/benchmark_command.json",
      "bytes": 395,
      "source_sha256": "7a415096730e28a352362d3ef7a6f5df4292d68de28f14f35e99906b60db749b"
    },
    {
      "path": "clair3/results/clair3_20261027_203222_759031/benchmark_exit.json",
      "bytes": 16,
      "source_sha256": "2618b95f1f6be6bfde8537756f85b83845d039001563135b9d59d817446733eb"
    },
    {
      "path": "clair3/results/clair3_20261027_203222_759031/region_50000_full/exit_code.json",
      "bytes": 20,
      "source_sha256": "953b9538e7e86b5237db3d2db46f7166ba7d2bfa599b53461f6440175b7783a1"
    },
    {
      "path": "clair3/results/clair3_20261027_203222_759031/region_50000_full/command.json",
      "bytes": 501,
      "source_sha256": "c1aa7986a959a7086934382c993b12f49e7b5c812ac247a3943c093c2f43345f"
    },
    {
      "path": "clair3/data/input_audit.json",
      "bytes": 28990,
      "source_sha256": "c19b4ddf8961699e445671cc09bef83fbd18d3294d28ed4b26c1c62ee56d68be"
    },
    {
      "path": "clair3/data/sources.json",
      "bytes": 1840,
      "source_sha256": "8b43ce5b945d14288e46ca0e2217b32cf271698abfa0ba666472a343ec67e478"
    },
    {
      "path": "clair3/models/ont/sources.json",
      "bytes": 461,
      "source_sha256": "4014c39ae3e46003e5bb3c6086578c55dbe63e411e4b495981da67462feb6dc1"
    },
    {
      "path": "rnafm/scripts/benchmark.py",
      "bytes": 7009,
      "source_sha256": "16bd21a5b8334af05e0fe06f437a974ac83e51a2d20bfd66421838cc770bea58"
    },
    {
      "path": "rnafm/scripts/setup.py",
      "bytes": 511,
      "source_sha256": "677fbc25a9369717a1b850aa32d18fdf99d2c1f441f245db86c3653f37025b8a"
    },
    {
      "path": "rnafm/scripts/reload.py",
      "bytes": 1831,
      "source_sha256": "2126d29afe17eaf672647be834c0a2c2550c1bc203f4d36434801003cc41533b"
    },
    {
      "path": "rnafm/scripts/download.py",
      "bytes": 756,
      "source_sha256": "f366bdfe5347a48730184673b7fe2067180f7f0aead1dd04f7ba3b800399071f"
    },
    {
      "path": "rnafm/scripts/report.py",
      "bytes": 7204,
      "source_sha256": "7477fe25e592335212712226fd9a9d7b375137072dc8a587ed451e8c1629a3f2"
    },
    {
      "path": "rnafm/scripts/prepare.py",
      "bytes": 2381,
      "source_sha256": "4ba91d09440c7414653ea63e786d207629ce4ccc5905cac9a56dffa8eca917fb"
    },
    {
      "path": "rnafm/reports/comparison.csv",
      "bytes": 3101,
      "source_sha256": "9e61ba100325575417fef73f99c6f12acb25904bdc16fb0e4b8dd0680319349b"
    },
    {
      "path": "rnafm/reports/RNAFM.md",
      "bytes": 7449,
      "source_sha256": "a518114bf82cbfb3769e6adb2fbe607a8f2192bd52734c1e9abada7343015acd"
    },
    {
      "path": "rnafm/reports/Batch_memory_followup.md",
      "bytes": 4428,
      "source_sha256": "8c767e2198c309aa05df3d500a95acb8e1d6e0de004240f8efd52d9a4729cb9c"
    },
    {
      "path": "rnafm/reports/batch_memory_followup.csv",
      "bytes": 1779,
      "source_sha256": "d1f7d13f2435808561300705f477cf2b241de26d2b2b753c73a54470132ca868"
    },
    {
      "path": "rnafm/results/setup_done.json",
      "bytes": 58,
      "source_sha256": "4e535f6906440ddf1e4a1780a1bae628eedbbd2ab13f4337bc3c65c16661838f"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch76_exit.json",
      "bytes": 68,
      "source_sha256": "9b790d67afea295ec0f6e23784a634d161223939b06ad000655c5f41679f1d64"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch32_command.json",
      "bytes": 792,
      "source_sha256": "23ef4ef4ab2703e877178f209024f01415487c907bf70768864cc61e56a54af4"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch2_command.json",
      "bytes": 790,
      "source_sha256": "d57ea84b522c6f3d69549feed7e38d436cd3fc85e25fcecdecc0e58caa9ab087"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch2_exit.json",
      "bytes": 67,
      "source_sha256": "b625c2293fb40e593ee8f4409ac024e326b1c61c8070ac6d68a3e641ff7ff9a1"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/summary.json",
      "bytes": 10560,
      "source_sha256": "9be3163318b89cfa7deec9a9b122363d17e3663ef6b01b261c33cbac6615c3a3"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch76_command.json",
      "bytes": 792,
      "source_sha256": "d9865f13ac7d6b4c26617d746968a7040a58fa06e9cd53a64c85607b6adb7085"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch1_exit.json",
      "bytes": 67,
      "source_sha256": "8213ea08111732ff90c8c5e60532bcc59927aafb234cf38557cd6b9b6833ddf3"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch4_exit.json",
      "bytes": 68,
      "source_sha256": "70e28e15154426817a1ae5b92b67f24a8c39da0db8b62e5d97281f52eef8ccef"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch32_exit.json",
      "bytes": 67,
      "source_sha256": "8fd674c9cfa70155f1aded873aa595c384f8db83276337daba74b1693543530f"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch4_command.json",
      "bytes": 790,
      "source_sha256": "9d56661fd0c3df65487807d8cb46a03faf2d7e142c6f020748ba43b7b482baef"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch64_exit.json",
      "bytes": 67,
      "source_sha256": "34464736248c4a0128dbb20f3c4f53dc749cee31bebbb17b7300968617d35256"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch1_command.json",
      "bytes": 790,
      "source_sha256": "e71d34b318fa5fdff5e507b38ab49191be2d44a236943538964c934e4de98894"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch8_command.json",
      "bytes": 790,
      "source_sha256": "cd0e22588a1f3473921237e7d6002cd95863624a6d83a565116917b513c18a40"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch8_exit.json",
      "bytes": 68,
      "source_sha256": "b58e45be1f0769948d7ab8d5e0f0f5f4a6d4ec873780e3ae6cbbe069e9928a8a"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch16_exit.json",
      "bytes": 67,
      "source_sha256": "d02138c99a2d5625cb42e946db1356f6cdf76e474be4f967fb304f64d9229e76"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch64_command.json",
      "bytes": 792,
      "source_sha256": "5e5cd4725cb07c9667a267c9fc63e1fad2b5defe673c69db5e5e2f9d50a6abe8"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch16_command.json",
      "bytes": 792,
      "source_sha256": "57b48e587c1fad0b009bd89509b371113c709c42d775d3b2ab35d2cac1ce9363"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_32/summary.json",
      "bytes": 3434,
      "source_sha256": "a186b9b905c4c1724d662f8d22e66b69017cc29aad6811ca6faa527939f00de6"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_32/resources.csv",
      "bytes": 4921,
      "source_sha256": "2ec94bbabd0f09ddeec31705734b5d9f45c64429641e7475e56a9b5c2589126e"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_8/summary.json",
      "bytes": 2878,
      "source_sha256": "8bea00647e04d755a42a7de4c34fb19f12e4235f2ba3bad220d0c0fb3508b945"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_8/resources.csv",
      "bytes": 4458,
      "source_sha256": "67958da6ccada301a5eecd4b6380f791de5aa2feb2fbe8a2c0ce18ef549d41f8"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_16/summary.json",
      "bytes": 3065,
      "source_sha256": "e6b375539f8fc6e9018e7659b8f01a37beb91b8e1fc7b12f4ca748f435181475"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_16/resources.csv",
      "bytes": 4573,
      "source_sha256": "5cf8ff71af705c9a720bbebd032bf4c4433a37d2eafd01fa3878658e1cbefc63"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_64/summary.json",
      "bytes": 4166,
      "source_sha256": "9157765cd746d887ac9087168fa190cf67ef1fefc832ae4974f6e5978c3e8a7c"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_64/resources.csv",
      "bytes": 5052,
      "source_sha256": "ba088f69c28c5a6ca2866e1e8a6b9aa9b70bba8be8e100f5c3cca6b2ce9bdd72"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_76/summary.json",
      "bytes": 4447,
      "source_sha256": "4b5cd532f33618ce0eab27476e78ecef4562632b8cd3a0a40456496efd6ec8ec"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_76/resources.csv",
      "bytes": 5496,
      "source_sha256": "c2556338c6a08acd0d763b0aa6659329bc36faabdd939600ca2da130d365adfa"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_2/summary.json",
      "bytes": 2726,
      "source_sha256": "10b8266d49464530ed67125051d3302f9b06d44b7920ec260f00bdbbb0c74cda"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_2/resources.csv",
      "bytes": 4446,
      "source_sha256": "939052d797f3fc02d5afa7f59856d4405ff4b22f2428c5d7e309b26446632ce0"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_4/summary.json",
      "bytes": 2770,
      "source_sha256": "a0515c5203ab709e51457a0dd46b4dbfa11c93e9fd1997081905086277108f3a"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_4/resources.csv",
      "bytes": 4435,
      "source_sha256": "30a492efd45893df3bfc9f4c09142060600deeade52573816065c9ce2f5fd704"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_1/summary.json",
      "bytes": 2703,
      "source_sha256": "34e971b6f25b0e9d0d97a4e2b9efb8936c44fecf6b49e159e10bfff7fbb42eab"
    },
    {
      "path": "rnafm/results/batch_followup_20261028_144452_703996/batch_1/resources.csv",
      "bytes": 4817,
      "source_sha256": "04adeb7e72336a42be65644889481ae6848d84b4f04e4df57b3347794e22f394"
    },
    {
      "path": "rnafm/results/rnafm_20261027_204440_142840/summary.json",
      "bytes": 33810,
      "source_sha256": "5ecfa4b0e41acea8bcb59cc328c764f6bad0f30706c3139ba1b35dac41de0306"
    },
    {
      "path": "rnafm/results/rnafm_20261027_204440_142840/reload.json",
      "bytes": 172,
      "source_sha256": "37076632caa45826a556c37dc572f888aec99339821894be109e89abcfe62990"
    },
    {
      "path": "rnafm/results/rnafm_20261027_204440_142840/test_predictions.json",
      "bytes": 20532,
      "source_sha256": "f923af0692dbed3ddbe9c0683304af2336e82916cae83bafceb26b79b7944565"
    },
    {
      "path": "rnafm/results/rnafm_20261027_204440_142840/resources.csv",
      "bytes": 48317,
      "source_sha256": "2543e884db18cc5e8ab259511b61fca17aa1606cfd0f905c3d126ddbd61d2c3d"
    },
    {
      "path": "rnafm/results/rnafm_20261027_204440_142840/benchmark_command.json",
      "bytes": 473,
      "source_sha256": "822c49e4b29625ea4e92c30abdced79e3aa2d30748c7186808e1bee58500c7e4"
    },
    {
      "path": "rnafm/results/rnafm_20261027_204440_142840/reload_command.json",
      "bytes": 470,
      "source_sha256": "efe732f30f2bd2e2abb8e1d9158055b8a2326a76c2da1c639d5ed3b6a4b34f69"
    },
    {
      "path": "rnafm/results/rnafm_20261027_204440_142840/reload_exit.json",
      "bytes": 58,
      "source_sha256": "fdd1c9353ff497f2aaa9a2d84b007121117597491162f063a0ef19e28d4fdec8"
    },
    {
      "path": "rnafm/results/rnafm_20261027_204440_142840/benchmark_exit.json",
      "bytes": 58,
      "source_sha256": "1d591447eecb4bb34969a6463022cf183c653cecf88f63ef39223c941eb14295"
    },
    {
      "path": "rnafm/data/input_audit.json",
      "bytes": 7412,
      "source_sha256": "e4504c185f7e0db4b151cf4da90cc735c3e79cfc2188fc939ab7d1dad3f77d42"
    },
    {
      "path": "rnafm/models/sources.json",
      "bytes": 314,
      "source_sha256": "b56b800b348c79b640c36d38d1f1e342bcd8597891af226e78e11491c2ef6067"
    },
    {
      "path": "openspliceai/scripts/benchmark.py",
      "bytes": 7294,
      "source_sha256": "35c95257b0d5eeb42fcd31019f1dc360e3b0c745535035074fa7e050f0107861"
    },
    {
      "path": "openspliceai/scripts/setup.py",
      "bytes": 534,
      "source_sha256": "34c5931018bf7c317205782f0379ef7592b9458706a69397443e186ede95ae48"
    },
    {
      "path": "openspliceai/scripts/reload.py",
      "bytes": 1449,
      "source_sha256": "215a839e220cff1598db44d3078c65f2d5998947d281d809d851550184940401"
    },
    {
      "path": "openspliceai/scripts/report.py",
      "bytes": 8518,
      "source_sha256": "468acc5b8012f133bbc1421c28ca7d6f003db0153bc44ff3838a33be75a3d8be"
    },
    {
      "path": "openspliceai/scripts/prepare.py",
      "bytes": 3950,
      "source_sha256": "e38ffac7abb52de06b6439d9a614c9c02d1bbbead1255611528fad8b6f38327f"
    },
    {
      "path": "openspliceai/reports/comparison.csv",
      "bytes": 3925,
      "source_sha256": "dbb22800765f25c1d5641b2a4faef3d46c94202706de82f580cd64ea3f5576fd"
    },
    {
      "path": "openspliceai/reports/Batch_memory_followup.md",
      "bytes": 4626,
      "source_sha256": "24c501a395a5c41ee358cfe2c55b6db013f15526e348ca11221b2e5187d08b2a"
    },
    {
      "path": "openspliceai/reports/OpenSpliceAI.md",
      "bytes": 9204,
      "source_sha256": "10e1a6f8db2a9cb79aba150ab914a4b515ca55b8e407e0f770a96704bf8940c4"
    },
    {
      "path": "openspliceai/reports/batch_memory_followup.csv",
      "bytes": 2210,
      "source_sha256": "3e6ce7a58acc38e6d19c9816e0e4a3065589c7ef78f449b29a8860c3e51eda49"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch32_command.json",
      "bytes": 813,
      "source_sha256": "7bac3e531e025c718400e7800a4b49d3ad90a527ac3624f7b195993e089839c1"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch2_command.json",
      "bytes": 811,
      "source_sha256": "d5cf5779121fd9553ae002fb3ac278dde509621c105bdfced5ddb2465c58e5c0"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch2_exit.json",
      "bytes": 68,
      "source_sha256": "a345831e4c5cb658138de82a7828718be773eb8ac729392519af8b2ea5d13fde"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/summary.json",
      "bytes": 13073,
      "source_sha256": "5b4dc11d3a389e99de2da11c1c7c770b9baeeef02ee6206d95aeb195b4832b36"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch1_exit.json",
      "bytes": 68,
      "source_sha256": "706be3b64f7f9535ec2e16c82757421b7f828ac4100740db55dd848cb3669256"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch4_exit.json",
      "bytes": 67,
      "source_sha256": "36067e793c90bcaf0ccfff2f0c659d577f427ee57ff718dd05f41f0fa4be0a36"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch96_command.json",
      "bytes": 813,
      "source_sha256": "aab03c1e4834fc673a8bb460c6a53a6ecccdaf06b43ce9b944e97a3ad79d7288"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch128_command.json",
      "bytes": 815,
      "source_sha256": "509e0dd2626d8714f7162a5c1c8faf9c00ffa9b73f07c77a43fd4c4ff7790912"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch32_exit.json",
      "bytes": 67,
      "source_sha256": "13096839a7a3eaf8f3bb67b72c3f2ee574e8951505d66019da859339d18b5c75"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch128_exit.json",
      "bytes": 67,
      "source_sha256": "6a19481bb00cc66956028d659ea26a7a9ff4579bc37772218e308ee78910a195"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch4_command.json",
      "bytes": 811,
      "source_sha256": "8cbf74294d02b62e5a522b0e58cd4b59219776cffe822801a204a26804afc005"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch156_command.json",
      "bytes": 815,
      "source_sha256": "4f64ec299f1d44fb7fcfcbee1c57527306252f06e1db1583593651fee65c11f2"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch64_exit.json",
      "bytes": 68,
      "source_sha256": "177efc247b90ba7c72baf0c1ed9643476115f7b5fa72eee7f8663d77b9d2ab2b"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch1_command.json",
      "bytes": 811,
      "source_sha256": "4ab0d538c90fdeb24e73ba46a747dc85fe9bc5695f109b2fc2e0903f6c6ca367"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch8_command.json",
      "bytes": 811,
      "source_sha256": "be04a54132547eff58f545302cfc9f876bb90d4ddfd4ec151a777f324b09dd14"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch8_exit.json",
      "bytes": 67,
      "source_sha256": "73a63811e1422cb4088baf54097d294f5c4e5b02868310502df222f7ee93a1ef"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch156_exit.json",
      "bytes": 67,
      "source_sha256": "8dec0bfe2b4c7417f45757372c133bbd5cf8ab139762f5ea8f038abe8bc31e7e"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch96_exit.json",
      "bytes": 67,
      "source_sha256": "85a14e329212ea2539fc3c76e2813007bbad925869e147ed72c3f4919a21a12a"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch16_exit.json",
      "bytes": 67,
      "source_sha256": "f9d0a8ad1d5b1bf2176096bfb56aa31d4af486d7602f63f0dbad3c4fd0adad29"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch64_command.json",
      "bytes": 813,
      "source_sha256": "79b9c434ecc55a6a98a964479ff5cc96eec790e8ec44646dca00d06715f2664a"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch16_command.json",
      "bytes": 813,
      "source_sha256": "f44b35a2884c74723f87fc7a37360aad440828686844fad64a5ed9cde56c3b5c"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_32/summary.json",
      "bytes": 3591,
      "source_sha256": "3ac6e454a38d8d27bd020ea63d19656e76b74bcf8f5ca9309f8b60fdd47139c2"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_32/resources.csv",
      "bytes": 4848,
      "source_sha256": "54a8bafdaff736df4b55e3d55f6bd213cf595b33fd6c94c9c9729c9f4ea1605a"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_8/summary.json",
      "bytes": 2913,
      "source_sha256": "a04e1cd67b55d3d11ec7361c641181d83992bf35d30ec8f0e56f105a9730a5cf"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_8/resources.csv",
      "bytes": 4383,
      "source_sha256": "97991a4711340fbd0503d83080bee6531a8d10d2bbd684c194f47fba1817e425"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_16/summary.json",
      "bytes": 3142,
      "source_sha256": "632114239f13e87cbe2641bb3cb387d9a84c1e31b418894e320e4bf6f7df2e95"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_16/resources.csv",
      "bytes": 4517,
      "source_sha256": "e085066f4bac5b11703e5d60a8e9bfccad763d80d2420bf3558ce451ca32297d"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_64/summary.json",
      "bytes": 4486,
      "source_sha256": "21b78dea0fca9efaf384007dd8138ab7604d61b7df68f58bd225e01dd40d537c"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_64/resources.csv",
      "bytes": 5020,
      "source_sha256": "542588eddb3c57194c986accffa48cbc23ea3b9dd07cd9f24312d3e4b2881983"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_2/summary.json",
      "bytes": 2760,
      "source_sha256": "ff8c4fd6646be62016cd66e44699c42acc14849a1f2b9917c4ae4fe1fbd058bc"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_2/resources.csv",
      "bytes": 4391,
      "source_sha256": "e8762ab9109e659fb9209bb57d54a83966b123c8ac328746ffe68c0a62c688cb"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_156/summary.json",
      "bytes": 7037,
      "source_sha256": "c7b20ea4bcffa82f4943b4ba43b25c173a26a5764647143406c265454a37b16e"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_156/resources.csv",
      "bytes": 5501,
      "source_sha256": "e9b638472d5b666ae8711ebf4ca33242099e62a41a05a636641fa8c20c2b8ddc"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_4/summary.json",
      "bytes": 2816,
      "source_sha256": "81a1668ec54b3b708001248430701846d52a914bdbe13ef0525cc51907fe6b78"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_4/resources.csv",
      "bytes": 4410,
      "source_sha256": "1aaaa5934df8c1b459db7298ac1f939d129dc0ac826548c44f4a97745b169552"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_1/summary.json",
      "bytes": 2736,
      "source_sha256": "a97bc0eeab43a5da67caba39f87be458fc745a35fbdf61267f9da6feb88bf6f2"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_1/resources.csv",
      "bytes": 4527,
      "source_sha256": "f7bf1a0d7f11d98ca9e774bb78c1750430732230ee7a07a44bfebf93b06e358a"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_128/summary.json",
      "bytes": 6281,
      "source_sha256": "61c63de949373adc69cd68540e75ec6a3ae3bb60512cecd072a720df3f13add4"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_128/resources.csv",
      "bytes": 5683,
      "source_sha256": "7a0bd1657791a95127ae875cb5b2ebcad0c4740f5a0401d170cbc752ff21fa4f"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_96/summary.json",
      "bytes": 5404,
      "source_sha256": "c08383c77e8756bd9d364e9d67be56e74380cec195b7b903fec79a48969c42ee"
    },
    {
      "path": "openspliceai/results/batch_followup_20261028_144737_167765/batch_96/resources.csv",
      "bytes": 5638,
      "source_sha256": "8958365cf7b4893814df5040bd5317c08952042d938028b27b1480e320d368d1"
    },
    {
      "path": "openspliceai/results/openspliceai_20261027_205554_995618/summary.json",
      "bytes": 26414,
      "source_sha256": "db59432b8110e89cad161c170a110dfc872d0ea414f36b46081ac3a4ab238c92"
    },
    {
      "path": "openspliceai/results/openspliceai_20261027_205554_995618/reload.json",
      "bytes": 185,
      "source_sha256": "928967b4513d4467a2e81d8209931b3e215f4964b6203d4ac06338cd6d69245a"
    },
    {
      "path": "openspliceai/results/openspliceai_20261027_205554_995618/resources.csv",
      "bytes": 31227,
      "source_sha256": "a51b1e52d28e9cbc100e3a0f02bdb7ad317c9b02e68335c1a45fcd40ab4404e3"
    },
    {
      "path": "openspliceai/results/openspliceai_20261027_205554_995618/benchmark_command.json",
      "bytes": 487,
      "source_sha256": "9bb11141e70c4ac9bd19a55d1e8623e655e56edfe6e1d905cd731425bce05796"
    },
    {
      "path": "openspliceai/results/openspliceai_20261027_205554_995618/reload_command.json",
      "bytes": 484,
      "source_sha256": "b5bc083c3a7d068b4fdab4827af8302f533f90cc4e57c7ab921da2ab5f3b18f6"
    },
    {
      "path": "openspliceai/results/openspliceai_20261027_205554_995618/test_window_ids.json",
      "bytes": 856,
      "source_sha256": "a50ea7e7b8b339281343170e6feaeff2baa10790d86b560c2542ab5fff5076b0"
    },
    {
      "path": "openspliceai/results/openspliceai_20261027_205554_995618/reload_exit.json",
      "bytes": 58,
      "source_sha256": "3b52a9f5dc9ec559fbf0b6969bf91ba622f55cc92f2bbf1e4c54051a8e0b4081"
    },
    {
      "path": "openspliceai/results/openspliceai_20261027_205554_995618/benchmark_exit.json",
      "bytes": 58,
      "source_sha256": "d4278d63c536813cc6f9a51d45d6a83fd0d28f4972c765435fa2aec61e745955"
    },
    {
      "path": "openspliceai/data/input_audit.json",
      "bytes": 2542,
      "source_sha256": "967ab007ff41bf8c6f7796afd752b6e246ce9c12dc6b84634ca961f2f205fb23"
    },
    {
      "path": "scgpt/scripts/launch_memory_refine.py",
      "bytes": 799,
      "source_sha256": "9593eb00e81a5c6a9efa072f1466e9155331b8e0069c6231060e44f81f59cba3"
    },
    {
      "path": "scgpt/scripts/launch_supplement.py",
      "bytes": 878,
      "source_sha256": "d324d418971d6c647398cdd366b761f84d3a87fc52895ca441caddd5df338d5b"
    },
    {
      "path": "scgpt/scripts/memory_batch_refine.py",
      "bytes": 7341,
      "source_sha256": "71bd415ddfdbcd1f646e42a990904af5f13da38867a4b138612c0a2932d37606"
    },
    {
      "path": "scgpt/scripts/launch_full_realdata.py",
      "bytes": 2504,
      "source_sha256": "7fcd5a15c2e5b884fa5e84086dbec1e43df91dcfbe2e733aef689be6891fc299"
    },
    {
      "path": "scgpt/scripts/launch_memory_sweep.py",
      "bytes": 784,
      "source_sha256": "a599f7b196a46781b22f4d84def09d382b03541b0dd35d0d305cb935751a505f"
    },
    {
      "path": "scgpt/scripts/01_inference.py",
      "bytes": 19350,
      "source_sha256": "2f7cef694a53112817e070530144a51f1ebf965670a07ac3b6377b7e0e41698c"
    },
    {
      "path": "scgpt/scripts/02_run_finetune.sh",
      "bytes": 39641,
      "source_sha256": "22f965af22beafe155a54c31a5ecc0cd24a7190b8c1a67043f98cfa8be4e0f5a"
    },
    {
      "path": "scgpt/scripts/01_run_inference.sh",
      "bytes": 20910,
      "source_sha256": "07f6bef99e3ce0c503f4a937ecd1ea04af324b7b3c9baf11a20cfd1058b86005"
    },
    {
      "path": "scgpt/scripts/02_finetune.py",
      "bytes": 37835,
      "source_sha256": "e5e6c06b7d8997df87a120dafd4a553f23882180478f554da8cf5bec39efa6bc"
    },
    {
      "path": "scgpt/scripts/00_prepare_inputs.py",
      "bytes": 8171,
      "source_sha256": "d02cbe419cb536c00582aa100374e99467da69a5091f224f7ae860b7fb7e8dc9"
    },
    {
      "path": "scgpt/scripts/launch_train_util.py",
      "bytes": 950,
      "source_sha256": "cd056e3375c7f3cd147c8a0225ea702be574e0c2f97aa2f4fdd6d43e4709e24d"
    },
    {
      "path": "scgpt/scripts/launch_reload_supplement.py",
      "bytes": 695,
      "source_sha256": "3acde93e464dbbeb01f510748deaa21899d21fb40ff2e4444b4cec09caba5465"
    },
    {
      "path": "scgpt/scripts/full_realdata.py",
      "bytes": 39178,
      "source_sha256": "9368e97d5d003484531e9f52b6ef93e01836ed89eb9f986dd25b330a95450ab9"
    },
    {
      "path": "scgpt/scripts/train_util_supplement.py",
      "bytes": 38533,
      "source_sha256": "d2019b4af252c4e04b4da9afc9606a036f76ac9158bf48357880070f91b7e3cc"
    },
    {
      "path": "scgpt/scripts/resume_full_realdata.py",
      "bytes": 1795,
      "source_sha256": "50356d8db966815221816ce92d63434e30c341d8b8c3dc89298a278eba5a544a"
    },
    {
      "path": "scgpt/scripts/report_supplement.py",
      "bytes": 7846,
      "source_sha256": "65b76ad7e9d5f118d2411e8a4f04d488a8835588462c8112540677a4e14f6ba9"
    },
    {
      "path": "scgpt/scripts/reload_supplement.py",
      "bytes": 1848,
      "source_sha256": "b732330bec4e8271c9c08f9ada25d3a531798873cdcf827439b26e59e76e5911"
    },
    {
      "path": "scgpt/scripts/gpu_support.py",
      "bytes": 2858,
      "source_sha256": "349a7dc8855cbf32ad999ac113f97c4cf496b0b0154dd7b8e54fe00af58632d2"
    },
    {
      "path": "scgpt/scripts/report_memory_sweep.py",
      "bytes": 7745,
      "source_sha256": "fbedd2d0f6fcb119f5b76624661f60b966f0d299e4a483f8f13f665b49d97836"
    },
    {
      "path": "scgpt/scripts/supplement.py",
      "bytes": 3922,
      "source_sha256": "b903ef892ac0ccb8d759b02e72225adc41f41229974937b36c1728faee6a1b8c"
    },
    {
      "path": "scgpt/scripts/memory_batch_sweep.py",
      "bytes": 7381,
      "source_sha256": "8c2ad5b07edfd7a4290562891973db3ad1afdb4e65cf1f50f731a9896c4c9f80"
    },
    {
      "path": "scgpt/scripts/report_full_realdata.py",
      "bytes": 10539,
      "source_sha256": "c53d84d82f5bd0e60d9e65b3f36670591d13250bc328b65311392871d9cb5a46"
    },
    {
      "path": "scgpt/scripts/large_scale/inference.py",
      "bytes": 5543,
      "source_sha256": "67dd1ecf01629fa5d7f0668cbf84a11abe049e294694a58ff4a6339c7cb5fe3a"
    },
    {
      "path": "scgpt/scripts/large_scale/watch.py",
      "bytes": 753,
      "source_sha256": "408105c2ba3904123322025616bed935a9c291943ee728de3670ece5167fd5ac"
    },
    {
      "path": "scgpt/scripts/large_scale/resume.py",
      "bytes": 1680,
      "source_sha256": "880321b874d8e549d78d4693d210d37341923c53a9767fe8e82f109eee566d21"
    },
    {
      "path": "scgpt/scripts/large_scale/download.py",
      "bytes": 2100,
      "source_sha256": "4b991acb386eb2afd20cf69b398709754363756212d536249a9053d57199dea0"
    },
    {
      "path": "scgpt/scripts/large_scale/launch.py",
      "bytes": 1990,
      "source_sha256": "3488ebd0224604a3f2e0454a220e2ab1831b211d2741a84772083fbd613f382d"
    },
    {
      "path": "scgpt/scripts/large_scale/pause.py",
      "bytes": 2009,
      "source_sha256": "3e2ff120f03a8a44ee41e32f6d0e0029321039e0ac4835e0873d619d92326dbb"
    },
    {
      "path": "scgpt/scripts/large_scale/report.py",
      "bytes": 6551,
      "source_sha256": "7e4412df1ca29c40cc86a43bdf84af035efae38af8a79f7c4ba7b9ca14633286"
    },
    {
      "path": "scgpt/scripts/large_scale/discover.py",
      "bytes": 617,
      "source_sha256": "78bfd3cc3bfcc861d3e8f9955a259bdc86185b17775ba2654ff1f894f784ffad"
    },
    {
      "path": "scgpt/scripts/large_scale/resumable_inference.py",
      "bytes": 6223,
      "source_sha256": "21a2fe15e04c4b1c6fffc7586bedc67b97c2919bbd5c02b23cc6822b828741bc"
    },
    {
      "path": "scgpt/scripts/large_scale/finalize.py",
      "bytes": 5520,
      "source_sha256": "af0870b22df477c08140f85d6d3809bc97b0bf2d83db2746015771fe218f548a"
    },
    {
      "path": "scgpt/scripts/large_scale/prepare.py",
      "bytes": 5474,
      "source_sha256": "f5a10495f6d64b55f763bd3048475ac2cad61ecad2af64c391ab92484da22b0a"
    },
    {
      "path": "scgpt/reports/scGPT_actual_300k.md",
      "bytes": 10987,
      "source_sha256": "ede47e1f26d3f65b1803b5455b34b298e718054aa341dc19880752ff78a2fdac"
    },
    {
      "path": "scgpt/reports/actual_300k_comparison.csv",
      "bytes": 1204,
      "source_sha256": "713b5249ecb549dc23e783f2266e81944cf5070929bd74298a866e771bea9a47"
    },
    {
      "path": "scgpt/reports/scGPT_realdata_full.md",
      "bytes": 1877,
      "source_sha256": "2c83444478a471caefc9d60c44616117b8ff9ebeff1b9bd4b35ef4b05c0e0e0c"
    },
    {
      "path": "scgpt/reports/memory_batch_comparison.csv",
      "bytes": 2604,
      "source_sha256": "4b9704bb306e651e377401236050cd5476b519c6b37eba88c09632e7cb019916"
    },
    {
      "path": "scgpt/reports/memory_batch_curve.png",
      "bytes": 86402,
      "source_sha256": "9d15fc14941c76cfc3cec111608babb3594ee220287d76ea2af66218bbea4e45"
    },
    {
      "path": "scgpt/reports/scGPT.md",
      "bytes": 7202,
      "source_sha256": "2bad9741bb3ea8827f311cb2449ce8ce8bcf422ae4a4034a0e415c27dec138d8"
    },
    {
      "path": "scgpt/reports/scGPT_memory_batch.md",
      "bytes": 10174,
      "source_sha256": "9b2f0676055da1e58408f04e9d1dca5e44604753b70957621326d8880a782883"
    },
    {
      "path": "scgpt/reports/comparison_supplement.csv",
      "bytes": 1692,
      "source_sha256": "a27dfbfd88f75045248573572fda0befbdc45d7e93b607647d995da9c8a7c135"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/memory_guard.json",
      "bytes": 194,
      "source_sha256": "9c0d9aba43c517d293f80d1b0be639f87a0177cc707b132bc441cdd198e9a076"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/summary.json",
      "bytes": 7384,
      "source_sha256": "9cf5227045994fdcfbfc9ef840f235f780b89cb5894487583ab144b0c3758a1f"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/protocol.json",
      "bytes": 689,
      "source_sha256": "db84f91926add35797c51721ae17f861c1240659b5f6e2b1865ba5d37b597dfb"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/source_data_audit.json",
      "bytes": 790,
      "source_sha256": "c0ab0e3db0c31e9196e75b27e1dfe74f96e3135cf4923c740aa821fa445de3ef"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train_exit.json",
      "bytes": 110,
      "source_sha256": "75b1dd1b5ae95cddd9cb949ce875c68e20faab7cf2788934528bc0a2fb7952c0"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test_exit.json",
      "bytes": 111,
      "source_sha256": "6eaa16d1e6820ea84f970ce175fdf74206ec7713c24288a21dd63509216f2a94"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train_command.json",
      "bytes": 561,
      "source_sha256": "fe1cf87e76a2fb179e1625fce7d88e911f13ff46e56eb293fc2f6b616c1fe948"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test_command.json",
      "bytes": 605,
      "source_sha256": "f88f5c717c8ce99d209809b411a0cd3bca63933f6ee724c4ed199aafa8bfb642"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/input_cells.tsv",
      "bytes": 101283,
      "source_sha256": "fcba1693abc4f3aec88b7a0c296a1cdc8985141efdc444f4809cffcca8b60662"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/environment.json",
      "bytes": 917,
      "source_sha256": "a7c1e54055b04e3fda07cce58e25bd63f1a65d8c4ec46a993be54a3ca2af90e7"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/summary.json",
      "bytes": 1301,
      "source_sha256": "71edba5cf9fc8b7d6f36ed7875486c21cbfa9a6503d8920b397a5041701ccae7"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/classification_report.json",
      "bytes": 2015,
      "source_sha256": "37305f2c007be56aea99083d2544460fd926d8aea60f317bb180659ca064b354"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/resources.csv",
      "bytes": 5792,
      "source_sha256": "4cd7c34f5650e1c606dca2dddb6e20405673c5ca664d03730410e45b9cce633a"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/confusion_matrix.csv",
      "bytes": 544,
      "source_sha256": "a73502a3f271bea49fb61d33064731fd94497d9a688eaa22d7ba1f91a00a2dc8"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/metrics.json",
      "bytes": 789,
      "source_sha256": "bc48269760e113a4b1594488cdb1416d57d8be6f45c6ead0e61fbc42817ef887"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/predictions.tsv",
      "bytes": 186247,
      "source_sha256": "d1a1a8a4257d2db4848400802d1fc00887f5768a75843564a9e9200fd53cb17b"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/test/prediction_config.json",
      "bytes": 1198,
      "source_sha256": "6fe55f2dcaa5c8357366b7f787ad5b86e53acd479ebeb0875cb1ce9075cb0573"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/provenance.json",
      "bytes": 607,
      "source_sha256": "4658153ab18b783ba33071b76d9a0251d36fa68d45e06ab585b4d7503f20437a"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/environment.json",
      "bytes": 917,
      "source_sha256": "b7f410b362fb385fb845e4d6c4cad82024b6b7778b9eb0dd3e01477865610264"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/summary.json",
      "bytes": 4452,
      "source_sha256": "0b22bad01f3d14cc851afe25c2f8e9fd569cae5684f0626562af965d733b9a0e"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/split_class_counts.csv",
      "bytes": 210,
      "source_sha256": "4556b47a1de4ab47fee0e7b054b0e337c1f219d62b24b7e4b887c4235a7b62a6"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/model_loading_baseline.json",
      "bytes": 153,
      "source_sha256": "ad519f9c711a3c7f534cd243eba77064c98477477e387ae1f5b7aad4181c4fde"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/run_config.json",
      "bytes": 2602,
      "source_sha256": "4e3f60131b23a6ec6abda16fa18221cfcf9a4a160f493db088989b38e0309885"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/split_assignments.tsv",
      "bytes": 509577,
      "source_sha256": "b43302670c0d6b7362ae41a8b70e88f7ae6dfba32a3ee23a62064147198f83ba"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/epoch_001/training_steps.csv",
      "bytes": 56551,
      "source_sha256": "2ba79a589810fc0f047a5b47366d05131a61b417c1e87794d0cb435a0e6b68d4"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/epoch_001/summary.json",
      "bytes": 1732,
      "source_sha256": "af43eb14d23bd284f87ad7f579d426b31df2963d0cf9852a737489c985de7c0b"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/epoch_001/training_resources.csv",
      "bytes": 66115,
      "source_sha256": "af459671efe63e0d6feccb9602c9fb377d507153eb01e669924ce7f8c941588c"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/epoch_001/validation/classification_report.json",
      "bytes": 2017,
      "source_sha256": "e18ee94478d514699fcc647e76371d6c1d9542bb08ce329bcefaca6c5c503bb0"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/epoch_001/validation/resources.csv",
      "bytes": 1577,
      "source_sha256": "a05908110e91980fdd809bc956d3a742d3886005d995037ee53bbaa046cc109f"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/epoch_001/validation/confusion_matrix.csv",
      "bytes": 544,
      "source_sha256": "5c68d5326dc52c5812b35341ff5b8dbb2782192da2249237a1b6180a3b670433"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/epoch_001/validation/metrics.json",
      "bytes": 789,
      "source_sha256": "7b9a615a162637c133634efbf3ec7f3577aa6c296737b7470cab461d9f894f70"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_8/train/epoch_001/validation/predictions.tsv",
      "bytes": 116693,
      "source_sha256": "6e66097ed403ebcfe4f288bdfa83767eeeedde65e6322afbc69a652ddf4445c2"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train_exit.json",
      "bytes": 112,
      "source_sha256": "7795d606dfc25543ac083d42cb76eb1bcaf0cdeaa3211d3128b77927f54046e3"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train_command.json",
      "bytes": 567,
      "source_sha256": "c9a7bd8d24e4a48a166eaf2eeda3f29f25cfe2888b2141a58e981ef760b35a5f"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/provenance.json",
      "bytes": 607,
      "source_sha256": "4658153ab18b783ba33071b76d9a0251d36fa68d45e06ab585b4d7503f20437a"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/environment.json",
      "bytes": 917,
      "source_sha256": "a9d78db366c5b30e1e4f021c01c19e7eaede51d37a3be41e57b23c724849b6c9"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/split_class_counts.csv",
      "bytes": 210,
      "source_sha256": "4556b47a1de4ab47fee0e7b054b0e337c1f219d62b24b7e4b887c4235a7b62a6"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/model_loading_baseline.json",
      "bytes": 151,
      "source_sha256": "5165742fd0f303d8ec31583ba64004b6298e0c89bd6e9a4c51ad5c9c1a6c4b34"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/run_config.json",
      "bytes": 2604,
      "source_sha256": "ed43f80f026ae872f7d0eb49e0421fa19956c6c14d9d9d54d1dbda6a9c373e46"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/split_assignments.tsv",
      "bytes": 509577,
      "source_sha256": "b43302670c0d6b7362ae41a8b70e88f7ae6dfba32a3ee23a62064147198f83ba"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_284/train/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train_exit.json",
      "bytes": 69,
      "source_sha256": "5a927ac8b080ef0319f262efae055dfe1851d9579b750dc0b824f83c41322cd5"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/memory_guard.json",
      "bytes": 190,
      "source_sha256": "174b8527a7c247d1d487a5a87326e063af451b5a1028c20877288324150c5a92"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train_command.json",
      "bytes": 694,
      "source_sha256": "3184297eefbd16218b5d841b6d683bff580cb1d16d6c6df3c7b1f8b93a47d71b"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/provenance.json",
      "bytes": 607,
      "source_sha256": "4658153ab18b783ba33071b76d9a0251d36fa68d45e06ab585b4d7503f20437a"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/environment.json",
      "bytes": 917,
      "source_sha256": "1553dd30e0a27e4016704cf0bd6f8b8c95137403c5b965282886856f113071d5"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/split_class_counts.csv",
      "bytes": 210,
      "source_sha256": "4556b47a1de4ab47fee0e7b054b0e337c1f219d62b24b7e4b887c4235a7b62a6"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/model_loading_baseline.json",
      "bytes": 153,
      "source_sha256": "2c5b9ee399e384181be9099e80a1885427b3126eb7e67e0623194f8a7fcf9eba"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/run_config.json",
      "bytes": 2604,
      "source_sha256": "99b4e7eb7d963922815d4c4d3106eac2afb74c14a0e3f2c4305d9ed54fc50248"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/split_assignments.tsv",
      "bytes": 509577,
      "source_sha256": "b43302670c0d6b7362ae41a8b70e88f7ae6dfba32a3ee23a62064147198f83ba"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_256/train/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train_exit.json",
      "bytes": 70,
      "source_sha256": "18d49b683008137d5577ead7016929e88fbbbbd9b24f8660e589d2b3716f1af2"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/memory_guard.json",
      "bytes": 188,
      "source_sha256": "b00a27796d01511ab6ff336560fb7bd0b405a22f55f9e9a122967e4909eeab28"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train_command.json",
      "bytes": 694,
      "source_sha256": "8349d619f0b582e9cb6e140d753a3b30ac884582238443e9c702b783272dcc86"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/provenance.json",
      "bytes": 607,
      "source_sha256": "4658153ab18b783ba33071b76d9a0251d36fa68d45e06ab585b4d7503f20437a"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/environment.json",
      "bytes": 917,
      "source_sha256": "56d78ba8354ba557d169cf4429d39aca0dd12fd13450b458c1af21b7d463c68f"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/split_class_counts.csv",
      "bytes": 210,
      "source_sha256": "4556b47a1de4ab47fee0e7b054b0e337c1f219d62b24b7e4b887c4235a7b62a6"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/model_loading_baseline.json",
      "bytes": 153,
      "source_sha256": "951d1e1d9f9164b5b3c48cce567b6eb283086b458c17d3cac38e5637c814be62"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/run_config.json",
      "bytes": 2604,
      "source_sha256": "09258eaeb41d1828a898987b65d8b3f43f19697168dbcdd596602d883498c60b"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/split_assignments.tsv",
      "bytes": 509577,
      "source_sha256": "b43302670c0d6b7362ae41a8b70e88f7ae6dfba32a3ee23a62064147198f83ba"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/realdata_full_20261027_215022_050816/batch_224/train/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/input_report.json",
      "bytes": 4362,
      "source_sha256": "59bb7cab2b4f3bdc302dbd17b234670ec63cfa3a2b100dda802a47f28b566c0b"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/environment.json",
      "bytes": 287,
      "source_sha256": "c89df6bc5626aa53da4df88f015f6683bd39f4495d949b4d0cb95f6d33c5089e"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/demo_train_var_head.tsv",
      "bytes": 139,
      "source_sha256": "6b6b5ed5aa2da9077519838f6c38234901cea149f18fde3296f24d4088dc6744"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/demo_train_obs_head.tsv",
      "bytes": 360,
      "source_sha256": "c6c6d8570d9f6e05aed3dacb9263ddacf6296edc09129db44a02ca444feb1541"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/demo_train_expression_5x5.tsv",
      "bytes": 359,
      "source_sha256": "14ea74f5368e039eb1511e702b7247b20222b9a98222649d0325d2db40682f0f"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/demo_test_obs_head.tsv",
      "bytes": 191,
      "source_sha256": "af1d5c1fe11d4c2d8d40910a57045923f432b203594004b3568e298cdcd4824e"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/run_command.txt",
      "bytes": 162,
      "source_sha256": "4f401b8ddb8da986a358fbcf0e8abeec7b9934d342d1a91df79e21f1589724e8"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/container_image.txt",
      "bytes": 119,
      "source_sha256": "ed859ddfe22da12dea67d60b1deffc2b5cf0b9ae058e70dce307289fd8b3b85d"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/demo_test_var_head.tsv",
      "bytes": 139,
      "source_sha256": "6b6b5ed5aa2da9077519838f6c38234901cea149f18fde3296f24d4088dc6744"
    },
    {
      "path": "scgpt/results/input_check_20261023_030617_498960/demo_test_expression_5x5.tsv",
      "bytes": 270,
      "source_sha256": "b43507d13a2e13b1299f5ef6f2b30f2eaf88bf3f8a1a199ee98451799d0f17b3"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/provenance.json",
      "bytes": 651,
      "source_sha256": "299f3fb563a9b674d8ec2ce72cde51210c0dc02c2f3433ae37625006fc945e45"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/excluded_genes.txt",
      "bytes": 5,
      "source_sha256": "da223486f4863d881c93ea5657493c8567afd82ca21388a42033e12045e35491"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/environment.json",
      "bytes": 832,
      "source_sha256": "1d4e32b8e904f87c91c907911be3d6fc05566e37ce46a721fa9296e3391beed6"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/matched_genes.tsv",
      "bytes": 35396,
      "source_sha256": "c04b6b3cb62f3336d56a596ce2b5a81bbd68ba28b325fe62fd51174e7283c5d6"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/summary.json",
      "bytes": 1625,
      "source_sha256": "bb9c4e9631a3054532489b5df7985d46b04077b5547dd50bc2f4fa594928a815"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/weight_loading.json",
      "bytes": 8847,
      "source_sha256": "d71ad993c0c4251bba97858a2e93e7799e1492dbb75fede6d9264f7c2af58817"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/resources.csv",
      "bytes": 1552,
      "source_sha256": "2659a447cc78b68ce06554b292deedf10ca63f971c077a65d7e5bd0e25d2873a"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/first_cell_token_example.tsv",
      "bytes": 418,
      "source_sha256": "a0c90bd76d022a68eb6f1f6c13c9bed8932ec46c3c7da39d866cb182de66848b"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/selected_cells.tsv",
      "bytes": 46108,
      "source_sha256": "c08d6c017d7ad7a3b88dd27149e638b89c075eed3920bc5ec0904ee30a0b3cd8"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/batch_shapes.tsv",
      "bytes": 2213,
      "source_sha256": "b77b96cbaceb0936ff1906648166da0b05af1fb077899d87b5b24e6e1c1a3e1a"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/run_config.json",
      "bytes": 1231,
      "source_sha256": "fb978ede8689146426269ba68c4215dab978a21e0cb22bffd8f551aef2e5a447"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/pretrained_vocab.json",
      "bytes": 1317639,
      "source_sha256": "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/container_image.txt",
      "bytes": 119,
      "source_sha256": "ed859ddfe22da12dea67d60b1deffc2b5cf0b9ae058e70dce307289fd8b3b85d"
    },
    {
      "path": "scgpt/results/inference_20261023_031632_499710/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/provenance.json",
      "bytes": 607,
      "source_sha256": "188b6369ffb7e5ff9d4267f84abc2614f25a527744ecb560155aaf28c0d3817b"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/environment.json",
      "bytes": 917,
      "source_sha256": "be90871c7f751ebe9164796485380b15603edcab5bf453844b8d1bcb86e8320c"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/summary.json",
      "bytes": 4055,
      "source_sha256": "7a1fd17d40eaa50b0ef4bf9b44509ad24d2160ecc03e1f8f405c151da536afbf"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/split_class_counts.csv",
      "bytes": 264,
      "source_sha256": "3c96e7a29f9a9fe2181055e019149aa55bb5a86c43734efd5b301095e11b6e96"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/run_config.json",
      "bytes": 2602,
      "source_sha256": "ebca3341cd7b6a7709c0bf7fb9edc2f2de17b10d87854a87667dbfe2964b42ad"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/container_image.txt",
      "bytes": 119,
      "source_sha256": "ed859ddfe22da12dea67d60b1deffc2b5cf0b9ae058e70dce307289fd8b3b85d"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/split_assignments.tsv",
      "bytes": 561937,
      "source_sha256": "5e67cb073008fadcbb8f6b228358f9c649b917e45a6d7350f9278e475c26608a"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/epoch_001/training_steps.csv",
      "bytes": 6457,
      "source_sha256": "84272d2c4f50ee46b484722eebb441fe7eb6e71c3c8ec093815ed822e35af921"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/epoch_001/summary.json",
      "bytes": 1486,
      "source_sha256": "f1e6f125820550208d887a0dd1e42c4667de5dcf05a166305b9a0bf51a1ed385"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/epoch_001/training_resources.csv",
      "bytes": 6989,
      "source_sha256": "6b7cb2dc0f4e56fd7bfa0191b4ab22afd46f2f2279aba101824d8062b12ae44c"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/epoch_001/validation/classification_report.json",
      "bytes": 2014,
      "source_sha256": "815f47ae7ff49c72f4ff941fcad07c8a5e4bc742736dc5eb48262506a136d2b7"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/epoch_001/validation/resources.csv",
      "bytes": 1250,
      "source_sha256": "3262e85237cc6bfc306c63181aefc81d8842b7e4211e68723bfbdca9938fe527"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/epoch_001/validation/confusion_matrix.csv",
      "bytes": 545,
      "source_sha256": "0040b48a16bb4c3abb0bebf5aab275fb7efd6c9980701f12192d6224b16d853b"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/epoch_001/validation/metrics.json",
      "bytes": 680,
      "source_sha256": "927a88e642ab3db49b8b2ed861fc4a3bca18e184a99320971c8da5fd50e82de1"
    },
    {
      "path": "scgpt/results/finetune_20261023_033701_500560/epoch_001/validation/predictions.tsv",
      "bytes": 116955,
      "source_sha256": "ff732a2f858006bde4199cdbd1d90857ee148fdfaff66a8c9b71373a46f79a82"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/input_cells.tsv",
      "bytes": 101283,
      "source_sha256": "fcba1693abc4f3aec88b7a0c296a1cdc8985141efdc444f4809cffcca8b60662"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/environment.json",
      "bytes": 917,
      "source_sha256": "e39bbbb96c8d97c80b93a592bd0af3b04baee3030ab6eaed8f4eda073d68fe2d"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/summary.json",
      "bytes": 1183,
      "source_sha256": "a0aed1f9174799fc2dac7437418024793e20188aea1685025610612870b1fa89"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/classification_report.json",
      "bytes": 1970,
      "source_sha256": "20296547027fa585a67015aaf4b01205bd03ba395f9bc33a0fb20430bc9de177"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/resources.csv",
      "bytes": 4970,
      "source_sha256": "b1990bb8267312666e747961c53f828975c061f175fdc0fc9a0bbf97aae63131"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/confusion_matrix.csv",
      "bytes": 547,
      "source_sha256": "ef5429cee7c86c6a6e618fbb1be2cb37bfb4126fb818f85b327745f4520ed5b8"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/metrics.json",
      "bytes": 684,
      "source_sha256": "5a2a0488d8bbb0a6ea55c2c6db6a04b97e0c6ad1819a6bc0c3434a276a47f2c8"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/predictions.tsv",
      "bytes": 186446,
      "source_sha256": "c12fb3537eb706b91164b8dc216b44ffc306187a37312d064c6a0acc77168008"
    },
    {
      "path": "scgpt/results/finetuned_test_20261023_034345_99523/prediction_config.json",
      "bytes": 1185,
      "source_sha256": "697e88dc28849383487c72aa401016d1e6424796ea332b6a6b44b0451f6e2c13"
    },
    {
      "path": "scgpt/results/memory_batch_20261027_213310_982032/summary.json",
      "bytes": 20654,
      "source_sha256": "eafc88da60ced5967b68423c492631603f0f88be319c2d877ac026e96313b2fe"
    },
    {
      "path": "scgpt/results/memory_batch_20261027_213310_982032/resources.csv",
      "bytes": 70918,
      "source_sha256": "d508fd39fea822254a9dd5f2bdf42ec39a44ca0ff61ee3288ec15dec9ba6ba30"
    },
    {
      "path": "scgpt/results/memory_batch_20261027_213310_982032/exit_code.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "scgpt/results/memory_batch_20261027_213310_982032/command.json",
      "bytes": 179,
      "source_sha256": "4b3950cf74cb49f3ead6c7159c3dcda4140e091bd04e2c0ff194ab4f76cb9b34"
    },
    {
      "path": "scgpt/results/memory_batch_20261027_213310_982032/model_config.json",
      "bytes": 430,
      "source_sha256": "14eedc0514e813a820ed37ddff5dd474d8974b1b9a26a62c10909b3b27e2aace"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_command.json",
      "bytes": 454,
      "source_sha256": "e2bf266effcb64dfc20183528b532973be869204ddc6317fe92f222dadefa69a"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/input_cells.tsv",
      "bytes": 197318,
      "source_sha256": "8eec8757f7eed608aaef5920f61893d6eab15ceeb07cb8984e791a667efb8620"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/summary.json",
      "bytes": 14138,
      "source_sha256": "3196b3bf7aa7712d7b330bcc86d32cf61d259f434916e49cadd7ee875ea6d6fb"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/reload.json",
      "bytes": 137,
      "source_sha256": "e4b1bb1358307f499221817bf0b5986f72215790ba87eee62bca8bce2b217d21"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/resources.csv",
      "bytes": 38596,
      "source_sha256": "b34a693ceeae01f3bb624e9be735e18985e0b8cbd75f3375e193b0c76a34e455"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/reload_command.json",
      "bytes": 176,
      "source_sha256": "54d081360b2f6b6447fde0f526f49688f86321340eb4c19dc24e858545a91d02"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/exit_code.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/command.json",
      "bytes": 269,
      "source_sha256": "addcc2b6b57ee7dc116b671857de0d1af1754809d724f81fe460cba8edd19bdd"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/reload_exit.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_exit.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/provenance.json",
      "bytes": 607,
      "source_sha256": "3185543207046d7e66a1f79fac2a5fc59db9086bb9d81809600e727c25b196b5"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/environment.json",
      "bytes": 917,
      "source_sha256": "b7cdd31b3b5ec5087d27b6299ae8a30a810c38abf8b745a666e4c9882769af55"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/summary.json",
      "bytes": 4414,
      "source_sha256": "dc4b1ac79557994f44d36d75b5756a383dcad17f09b8943f20842457848b7b5d"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/split_class_counts.csv",
      "bytes": 264,
      "source_sha256": "3c96e7a29f9a9fe2181055e019149aa55bb5a86c43734efd5b301095e11b6e96"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/model_loading_baseline.json",
      "bytes": 153,
      "source_sha256": "65907c82e85ba3b843c1fd64b151092805aeb7fd6241c6f13469745a5e5bbf87"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/run_config.json",
      "bytes": 2602,
      "source_sha256": "ebca3341cd7b6a7709c0bf7fb9edc2f2de17b10d87854a87667dbfe2964b42ad"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/split_assignments.tsv",
      "bytes": 561937,
      "source_sha256": "5e67cb073008fadcbb8f6b228358f9c649b917e45a6d7350f9278e475c26608a"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/epoch_001/training_steps.csv",
      "bytes": 6455,
      "source_sha256": "2081fe206a3aa37e5bcffcba8c47b4e60745fcd2899995684421f1f1bb48cf5b"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/epoch_001/summary.json",
      "bytes": 1705,
      "source_sha256": "14e3bc51c6a240d4ae995cb84abf8baa8050ea4865e0bef8e94cd220f07ed34b"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/epoch_001/training_resources.csv",
      "bytes": 7414,
      "source_sha256": "b16ede3cf2671a5c03778d39ad2d6355d3ae704cf302b1d5d40bee34365ac1b9"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/epoch_001/validation/classification_report.json",
      "bytes": 2002,
      "source_sha256": "1a86605910ca9f136e7d0fc4b3c73166cead7edc84b2aeda7ab7c06725b80c4f"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/epoch_001/validation/resources.csv",
      "bytes": 1358,
      "source_sha256": "affef9ca63ace0d46754b09d9a97c1c26bab8eb82d463b32ccbe81469a0d62fc"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/epoch_001/validation/confusion_matrix.csv",
      "bytes": 547,
      "source_sha256": "a4ebcef1f962d90ff4cf682bfd22cd359472e717b9818d8464ab124faff7cf5e"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/epoch_001/validation/metrics.json",
      "bytes": 787,
      "source_sha256": "c8452e74ecc24c0bef25049db43fc6e9d679bc0f8869c531ac5963f149e8f332"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/training_util/epoch_001/validation/predictions.tsv",
      "bytes": 116719,
      "source_sha256": "04abe841eaa584257dd956b511873cd7f52d8dccae7a43d956aeebee14b3b878"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/provenance.json",
      "bytes": 651,
      "source_sha256": "299f3fb563a9b674d8ec2ce72cde51210c0dc02c2f3433ae37625006fc945e45"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/excluded_genes.txt",
      "bytes": 5,
      "source_sha256": "da223486f4863d881c93ea5657493c8567afd82ca21388a42033e12045e35491"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/environment.json",
      "bytes": 832,
      "source_sha256": "1d4e32b8e904f87c91c907911be3d6fc05566e37ce46a721fa9296e3391beed6"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/matched_genes.tsv",
      "bytes": 35396,
      "source_sha256": "c04b6b3cb62f3336d56a596ce2b5a81bbd68ba28b325fe62fd51174e7283c5d6"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/weight_loading.json",
      "bytes": 8847,
      "source_sha256": "d71ad993c0c4251bba97858a2e93e7799e1492dbb75fede6d9264f7c2af58817"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/selected_cells.tsv",
      "bytes": 197318,
      "source_sha256": "8eec8757f7eed608aaef5920f61893d6eab15ceeb07cb8984e791a667efb8620"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/run_config.json",
      "bytes": 1231,
      "source_sha256": "d2fc9e193d53c263f2bca5a302074b30679cae4fddd92e4df94e5a4c74354552"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/pretrained_vocab.json",
      "bytes": 1317639,
      "source_sha256": "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f"
    },
    {
      "path": "scgpt/results/supplement_20261027_211638_936991/setup/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_input_cells.tsv",
      "bytes": 39169,
      "source_sha256": "27f9974e7eea2b5e98494d9604ed5cc114b96bc4bdd7c71492db96026bc6372b"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/summary.json",
      "bytes": 12519,
      "source_sha256": "20ab82fcc7ea4e54e4a0595abb46788daaa4e4712ef3838b281626cf93d8059a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_input_cells.tsv",
      "bytes": 27937,
      "source_sha256": "3e2165db6ac90ff77b64dec618d9855cb2ee5806ce22f60bddeeb8f9ca18ac84"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/comparison_table.txt",
      "bytes": 783,
      "source_sha256": "4d6025e22148acb19f2b0b0d2b4100bdbb9efd6888356fe020db6268ee65adad"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/comparison.csv",
      "bytes": 4599,
      "source_sha256": "bedb062478bb520a6c779660686c281f262c1c03bea50c653a648b9b73de88ba"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/process_records.json",
      "bytes": 774,
      "source_sha256": "761f016f891438558556a98045afac892a94f16a111767a11784354048b3443f"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/experiment_config.json",
      "bytes": 1337,
      "source_sha256": "33c468aed1f549756d6add2b5648b032872634848dff522f2f381711696a9400"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/input_length_report.json",
      "bytes": 33141,
      "source_sha256": "6f8cb6e8bdd2bf27fc2f3da92fdd18eae5636be08ad7f1bf18e756a953f9e5ae"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/comparison.tsv",
      "bytes": 4583,
      "source_sha256": "f25e04f03d24b391c24bec08a6dc4e8d52398c59d84a1ae37ae6f33258e9138c"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/container_image.txt",
      "bytes": 119,
      "source_sha256": "ed859ddfe22da12dea67d60b1deffc2b5cf0b9ae058e70dce307289fd8b3b85d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/provenance.json",
      "bytes": 607,
      "source_sha256": "188b6369ffb7e5ff9d4267f84abc2614f25a527744ecb560155aaf28c0d3817b"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/environment.json",
      "bytes": 917,
      "source_sha256": "8acf7c818c2e2fdb8be422ac1047c6071a3bd4d1f5f4882af92967d8accb71b4"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/summary.json",
      "bytes": 4084,
      "source_sha256": "29b77d1407f480a70f9ff907934369f84eb547ef0cccff13bbc9cec5b971fff8"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/split_class_counts.csv",
      "bytes": 264,
      "source_sha256": "3c96e7a29f9a9fe2181055e019149aa55bb5a86c43734efd5b301095e11b6e96"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/run_config.json",
      "bytes": 2601,
      "source_sha256": "9682f7ea4e7c7112690de4e7e99be52ccdc0f00de1ded74255b0ff188658eb5d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/split_assignments.tsv",
      "bytes": 561937,
      "source_sha256": "5e67cb073008fadcbb8f6b228358f9c649b917e45a6d7350f9278e475c26608a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/preprocessing.json",
      "bytes": 586,
      "source_sha256": "b8b9fd6d2b1ddf715565074030c026b3aedeb183a914227b62c572116a9f21b8"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/epoch_001/training_steps.csv",
      "bytes": 6458,
      "source_sha256": "80333c2072f50336cc1360128f913024028cbe9293cb2127fc1b235f8a30c3ce"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/epoch_001/summary.json",
      "bytes": 1500,
      "source_sha256": "06bd15c11a799381e4794e5574014b59004150f148db20bd1a23a4255acd8d5a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/epoch_001/training_resources.csv",
      "bytes": 5199,
      "source_sha256": "3c37c370fc71efd690e03adbb48ab3f92974bebbabd7f1306defe121c31963ee"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/epoch_001/validation/classification_report.json",
      "bytes": 1995,
      "source_sha256": "e60056257d73932e6fbc2905884c7eb069cde6a189404cda11d51477036dcd06"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/epoch_001/validation/resources.csv",
      "bytes": 1186,
      "source_sha256": "79d0d2322d4cf6bbf72418055e105d610a3924516fe61c32bded7f8993e8aa98"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/epoch_001/validation/confusion_matrix.csv",
      "bytes": 546,
      "source_sha256": "55371d2ec6c035cea22312a54762137fdf17c0be4ffa9dcd9db193b9bb151421"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/epoch_001/validation/metrics.json",
      "bytes": 680,
      "source_sha256": "2b234aff6e19fd68dec77d3be9ff146fe6ced3c5b080ba3495095501cb59b777"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0512/epoch_001/validation/predictions.tsv",
      "bytes": 116721,
      "source_sha256": "0f7a39332dd28cd4a47cf494109f8405a2192b6f2c6bf9e94e357c91d5a97273"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/provenance.json",
      "bytes": 651,
      "source_sha256": "299f3fb563a9b674d8ec2ce72cde51210c0dc02c2f3433ae37625006fc945e45"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/excluded_genes.txt",
      "bytes": 5,
      "source_sha256": "da223486f4863d881c93ea5657493c8567afd82ca21388a42033e12045e35491"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/environment.json",
      "bytes": 832,
      "source_sha256": "1d4e32b8e904f87c91c907911be3d6fc05566e37ce46a721fa9296e3391beed6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/matched_genes.tsv",
      "bytes": 35396,
      "source_sha256": "c04b6b3cb62f3336d56a596ce2b5a81bbd68ba28b325fe62fd51174e7283c5d6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/summary.json",
      "bytes": 1623,
      "source_sha256": "1cc40ec7aae79137208f9cc050f1c284eb18b601d833c9b6a23ce9cce9cacee2"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/weight_loading.json",
      "bytes": 8847,
      "source_sha256": "d71ad993c0c4251bba97858a2e93e7799e1492dbb75fede6d9264f7c2af58817"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/resources.csv",
      "bytes": 1408,
      "source_sha256": "535411000e67476562cb3039df80e2912c2fe847c2889e72d619136a02cd7259"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/first_cell_token_example.tsv",
      "bytes": 418,
      "source_sha256": "a0c90bd76d022a68eb6f1f6c13c9bed8932ec46c3c7da39d866cb182de66848b"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/selected_cells.tsv",
      "bytes": 46108,
      "source_sha256": "c08d6c017d7ad7a3b88dd27149e638b89c075eed3920bc5ec0904ee30a0b3cd8"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/batch_shapes.tsv",
      "bytes": 2213,
      "source_sha256": "b77b96cbaceb0936ff1906648166da0b05af1fb077899d87b5b24e6e1c1a3e1a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/run_config.json",
      "bytes": 1231,
      "source_sha256": "48e54f27d7a2f91a900a3c9b68d80f7b6552ffca76a458587b58b4f9d7856a30"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/pretrained_vocab.json",
      "bytes": 1317639,
      "source_sha256": "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1200/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/provenance.json",
      "bytes": 607,
      "source_sha256": "188b6369ffb7e5ff9d4267f84abc2614f25a527744ecb560155aaf28c0d3817b"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/environment.json",
      "bytes": 917,
      "source_sha256": "91063582293265cefef5740c2e37211d57b4915f1a8e95fda5060c4b17fffb2a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/summary.json",
      "bytes": 4098,
      "source_sha256": "e009619a4411d48219f9095879906f266921cbbcbb627c357ef35db5ede845d4"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/split_class_counts.csv",
      "bytes": 264,
      "source_sha256": "3c96e7a29f9a9fe2181055e019149aa55bb5a86c43734efd5b301095e11b6e96"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/run_config.json",
      "bytes": 2602,
      "source_sha256": "9425e4f2e43229b3cb899c0da73e66fb034ef5b9ee9f89288d21cfce925f7e8c"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/split_assignments.tsv",
      "bytes": 561937,
      "source_sha256": "5e67cb073008fadcbb8f6b228358f9c649b917e45a6d7350f9278e475c26608a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/preprocessing.json",
      "bytes": 587,
      "source_sha256": "cb7c30419cf9112a35ec603d10b5929c834f14fd8e9371d2d092d3b09106d627"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/epoch_001/training_steps.csv",
      "bytes": 6488,
      "source_sha256": "41c1443c6e2a9457bf5f1fec2dacb1acbe46f1d5247737c893c121cdb4d3e428"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/epoch_001/summary.json",
      "bytes": 1507,
      "source_sha256": "aefdd00ad4c0ec520b3931ef6e5bb1c88505478a91d9600930383307a42ecfc2"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/epoch_001/training_resources.csv",
      "bytes": 6980,
      "source_sha256": "1024d881c7deee3f441483b7c5157dfa204c5cc2e71531ae4d66d2621d77abcb"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/epoch_001/validation/classification_report.json",
      "bytes": 1985,
      "source_sha256": "bdc78a3073f303438a34a32fe42c2c6411d606b9813d3c6b88f04f610cc4c8e3"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/epoch_001/validation/resources.csv",
      "bytes": 1249,
      "source_sha256": "1fa8804b7bf1b9e47aec0f83832d6f7f3ed82f6e7cbf5ac87de1daf789591567"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/epoch_001/validation/confusion_matrix.csv",
      "bytes": 545,
      "source_sha256": "b19438895c60d3456d4e1d6fdc8e3ab9261311a1f18f07194fbd368225c85ecf"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/epoch_001/validation/metrics.json",
      "bytes": 685,
      "source_sha256": "08b66f0c340761e9b35909a0d901daefc0d85021d791c4947f23b93fdc096669"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1024/epoch_001/validation/predictions.tsv",
      "bytes": 116778,
      "source_sha256": "b6c89ff390fa9054f6ebec6ee17388236175c28338848dcc6cfcc726f4dcbea9"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/provenance.json",
      "bytes": 651,
      "source_sha256": "299f3fb563a9b674d8ec2ce72cde51210c0dc02c2f3433ae37625006fc945e45"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/excluded_genes.txt",
      "bytes": 5,
      "source_sha256": "da223486f4863d881c93ea5657493c8567afd82ca21388a42033e12045e35491"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/environment.json",
      "bytes": 832,
      "source_sha256": "1d4e32b8e904f87c91c907911be3d6fc05566e37ce46a721fa9296e3391beed6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/matched_genes.tsv",
      "bytes": 35396,
      "source_sha256": "c04b6b3cb62f3336d56a596ce2b5a81bbd68ba28b325fe62fd51174e7283c5d6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/summary.json",
      "bytes": 1624,
      "source_sha256": "94ef71bace1efdc97a1b74ef86025dada5b35f51cff8d2c25372cb3e5b0d9f3a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/weight_loading.json",
      "bytes": 8847,
      "source_sha256": "d71ad993c0c4251bba97858a2e93e7799e1492dbb75fede6d9264f7c2af58817"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/resources.csv",
      "bytes": 1412,
      "source_sha256": "d620254ea12cbec080985b9a6e682d3484cffa36db47780b0cd4f83f930e3ee7"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/first_cell_token_example.tsv",
      "bytes": 418,
      "source_sha256": "a0c90bd76d022a68eb6f1f6c13c9bed8932ec46c3c7da39d866cb182de66848b"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/selected_cells.tsv",
      "bytes": 46029,
      "source_sha256": "e9ce5bbef4cc8ec13b32278a84b99c50e1431051ec711795d722a35ae6113fd2"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/batch_shapes.tsv",
      "bytes": 2213,
      "source_sha256": "583abdbb7fab2bcfde43b05efc885af49f5035822ec89d225b6dddbc0ea92535"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/run_config.json",
      "bytes": 1231,
      "source_sha256": "07fb231b1ae0a3458c81011f881e68225689dadeaaa9a3490e396c14d956cfb8"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/pretrained_vocab.json",
      "bytes": 1317639,
      "source_sha256": "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L1024/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/provenance.json",
      "bytes": 607,
      "source_sha256": "188b6369ffb7e5ff9d4267f84abc2614f25a527744ecb560155aaf28c0d3817b"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/environment.json",
      "bytes": 917,
      "source_sha256": "4ee7166b4f9a65782a5940fb6d0579b72cfa61cb5be964ff34f669ee1f1ee0f1"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/summary.json",
      "bytes": 4089,
      "source_sha256": "c9023e0b3d9a145783dd18142a33ac4e036a3778f7a7e434359df5311c5c742a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/split_class_counts.csv",
      "bytes": 264,
      "source_sha256": "3c96e7a29f9a9fe2181055e019149aa55bb5a86c43734efd5b301095e11b6e96"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/run_config.json",
      "bytes": 2601,
      "source_sha256": "4a5eadbf2e18433e11afa381bc855b27cc840884fa60c3fab7d82afb7d302816"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/split_assignments.tsv",
      "bytes": 561937,
      "source_sha256": "5e67cb073008fadcbb8f6b228358f9c649b917e45a6d7350f9278e475c26608a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/preprocessing.json",
      "bytes": 586,
      "source_sha256": "f62eebb58282fb9d7d2eae3063aef867356a7cac0a14287901119a68093d343d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/epoch_001/training_steps.csv",
      "bytes": 6420,
      "source_sha256": "f694882943731313ed1557ebdb2d229739a8e00f836b5cd52fff6039a2aacf8f"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/epoch_001/summary.json",
      "bytes": 1503,
      "source_sha256": "2b6738e5cdff8c6a06ed2324bdfe3fd95cf80b3dee670f7ccbf32b46fac3acf0"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/epoch_001/training_resources.csv",
      "bytes": 3082,
      "source_sha256": "59dd0b003497336ef6c1fef2c5d8dfdd318c680c8fb548086c8012bfb41282ce"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/epoch_001/validation/classification_report.json",
      "bytes": 2029,
      "source_sha256": "c6174bb68313b8c54facd69ddc309521f039a11182a422fc1f52b4adb8243ac2"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/epoch_001/validation/resources.csv",
      "bytes": 933,
      "source_sha256": "8a5e72778988bfbdb9b0673ad0789b09b499aad2fbf4b85d3fa8f465cedc935a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/epoch_001/validation/confusion_matrix.csv",
      "bytes": 547,
      "source_sha256": "c8d7c97cbe8d8ef260c0210f0b1348f0bd27678ec5c68a325b620f1ea4e0a706"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/epoch_001/validation/metrics.json",
      "bytes": 683,
      "source_sha256": "cc7acc3e691b4527a2019061a45c7de11a7f9f54add2f46eac8aa45dca3e104d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L0256/epoch_001/validation/predictions.tsv",
      "bytes": 116896,
      "source_sha256": "5b6ee3359cdf5de32d1b6ca11b5e9b7e8dd5c34d4bb42f14dc23c6285499fed1"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/provenance.json",
      "bytes": 651,
      "source_sha256": "299f3fb563a9b674d8ec2ce72cde51210c0dc02c2f3433ae37625006fc945e45"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/excluded_genes.txt",
      "bytes": 5,
      "source_sha256": "da223486f4863d881c93ea5657493c8567afd82ca21388a42033e12045e35491"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/environment.json",
      "bytes": 832,
      "source_sha256": "1d4e32b8e904f87c91c907911be3d6fc05566e37ce46a721fa9296e3391beed6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/matched_genes.tsv",
      "bytes": 35396,
      "source_sha256": "c04b6b3cb62f3336d56a596ce2b5a81bbd68ba28b325fe62fd51174e7283c5d6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/summary.json",
      "bytes": 1624,
      "source_sha256": "dd3f9b41003cd24922ff6ea8e2a4a3e079133431619ff66071fb183ed07c6627"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/weight_loading.json",
      "bytes": 8847,
      "source_sha256": "d71ad993c0c4251bba97858a2e93e7799e1492dbb75fede6d9264f7c2af58817"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/resources.csv",
      "bytes": 663,
      "source_sha256": "2e84d95e6439f113b4d979cea6f93bccab8b1418aa9dc249f1b4947ae27ef287"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/first_cell_token_example.tsv",
      "bytes": 433,
      "source_sha256": "ab7684315b4cd46fd599614e9c09a0ed0c7056b09b5d61abe97b5801ad170732"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/selected_cells.tsv",
      "bytes": 45026,
      "source_sha256": "c0d6dd6aa093d2f0ed4258558c39968ec3375644558c3dc1177de6ef3c264ecc"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/batch_shapes.tsv",
      "bytes": 2160,
      "source_sha256": "5fe9149e5b97385b2ea2e439d26c5fb253b15eb348f5abfe1310a6b026553f91"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/run_config.json",
      "bytes": 1230,
      "source_sha256": "a43cb71664932991ed6d9d0059829aea013edbf008783c7d6c3217760cfc72d0"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/pretrained_vocab.json",
      "bytes": 1317639,
      "source_sha256": "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0256/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/provenance.json",
      "bytes": 651,
      "source_sha256": "299f3fb563a9b674d8ec2ce72cde51210c0dc02c2f3433ae37625006fc945e45"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/excluded_genes.txt",
      "bytes": 5,
      "source_sha256": "da223486f4863d881c93ea5657493c8567afd82ca21388a42033e12045e35491"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/environment.json",
      "bytes": 832,
      "source_sha256": "1d4e32b8e904f87c91c907911be3d6fc05566e37ce46a721fa9296e3391beed6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/matched_genes.tsv",
      "bytes": 35396,
      "source_sha256": "c04b6b3cb62f3336d56a596ce2b5a81bbd68ba28b325fe62fd51174e7283c5d6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/summary.json",
      "bytes": 1622,
      "source_sha256": "25e728120efeaf77f5f8e6108640217f767125d737d097364fa1ef3ece53d7b6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/weight_loading.json",
      "bytes": 8847,
      "source_sha256": "d71ad993c0c4251bba97858a2e93e7799e1492dbb75fede6d9264f7c2af58817"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/resources.csv",
      "bytes": 1073,
      "source_sha256": "dafc26855d9d25087d9c22947bdc353314d44367bf22a28c7279c6cde3130dad"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/first_cell_token_example.tsv",
      "bytes": 433,
      "source_sha256": "ab7684315b4cd46fd599614e9c09a0ed0c7056b09b5d61abe97b5801ad170732"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/selected_cells.tsv",
      "bytes": 45461,
      "source_sha256": "7bbf372fb3a14632ef0c13483adb7e9d0ee140486c2dc4910686ecee9cd94d93"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/batch_shapes.tsv",
      "bytes": 2160,
      "source_sha256": "c4dea69b8f79dd0634d4519c82d8dc9b2c002dcc656490593ce73b08cfa01c9e"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/run_config.json",
      "bytes": 1230,
      "source_sha256": "376baa7d8a2e9797f2f33181df13aaef8c6e26451267b3729eeff865afc0bd9e"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/pretrained_vocab.json",
      "bytes": 1317639,
      "source_sha256": "acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/inference_L0512/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/provenance.json",
      "bytes": 607,
      "source_sha256": "188b6369ffb7e5ff9d4267f84abc2614f25a527744ecb560155aaf28c0d3817b"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/environment.json",
      "bytes": 917,
      "source_sha256": "3978ac5310697dfca81fe4a7e46cc08397adb08aa6b709ae239bdd7ddd27584e"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/summary.json",
      "bytes": 4086,
      "source_sha256": "f1b8523dd0cef34b1ae5949b435454550ed792bfdf3f80d16a8eda8915e1b0cb"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/weight_loading.json",
      "bytes": 8443,
      "source_sha256": "15d319e75c1aa88cfe9c5c20d30ae0bcc1410c8c999865a8efb63e864cb42679"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/requirements_freeze.txt",
      "bytes": 6559,
      "source_sha256": "432f1ccfd6a5f036d7b974ac56fc8dc3aa5fea447fc0c30c41d68abbb47f5e4d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/split_class_counts.csv",
      "bytes": 264,
      "source_sha256": "3c96e7a29f9a9fe2181055e019149aa55bb5a86c43734efd5b301095e11b6e96"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/vocab.json",
      "bytes": 1317640,
      "source_sha256": "f38ca7bcb270e12b0b54fba634d3662428ebc28e201f8e938482dd1e734cb570"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/run_config.json",
      "bytes": 2602,
      "source_sha256": "ebca3341cd7b6a7709c0bf7fb9edc2f2de17b10d87854a87667dbfe2964b42ad"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/split_assignments.tsv",
      "bytes": 561937,
      "source_sha256": "5e67cb073008fadcbb8f6b228358f9c649b917e45a6d7350f9278e475c26608a"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/epoch_001/training_steps.csv",
      "bytes": 6464,
      "source_sha256": "6b4bb9e43d1109893f01c39ce85d182e44c78670109b662e92aaae797fe40220"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/epoch_001/summary.json",
      "bytes": 1501,
      "source_sha256": "9ff259d6c2ce34d1bf51e2123a21bc46219fb43419174c85f285ae956d5a1532"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/epoch_001/training_resources.csv",
      "bytes": 7053,
      "source_sha256": "0bae70a2366489a10fda3c0a43c28056e48c32ef67ad70708f0b3a80486c2e03"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/epoch_001/validation/classification_report.json",
      "bytes": 2002,
      "source_sha256": "4c04a410241f6f11c0830fcf0248aac54c88e83e7856634095ace121732b2a09"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/epoch_001/validation/resources.csv",
      "bytes": 1246,
      "source_sha256": "d4c94d8d45ff9524538e0b5beacaf789d2c218ceb670193f1e4cb0601dd76e95"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/epoch_001/validation/confusion_matrix.csv",
      "bytes": 545,
      "source_sha256": "e9cb8d6755b31cfd79657b3d96478e26af1c331629326368b60317632ab3748d"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/epoch_001/validation/metrics.json",
      "bytes": 684,
      "source_sha256": "0f32115af40a8f2bf11468a305b565232175789d5259e7a9f8a6fa72ba0402a5"
    },
    {
      "path": "scgpt/results/length_compare_20261023_040313_501423/finetune_L1200/epoch_001/validation/predictions.tsv",
      "bytes": 116903,
      "source_sha256": "52e1840ee96383766a5e92d82d9618903ec46a019d6892f5a1bce26011cca8a1"
    },
    {
      "path": "scgpt/results/memory_batch_refine_20261027_214026_294469/summary.json",
      "bytes": 5189,
      "source_sha256": "2efee00a5bf34867437995f9ceadc80a002499fee39c26b6aa7992e96c5297be"
    },
    {
      "path": "scgpt/results/memory_batch_refine_20261027_214026_294469/resources.csv",
      "bytes": 24294,
      "source_sha256": "b42e22ea06214761bd0d856b7ad7b67e67c20ab7cd8e663681f5c2357dcc64a6"
    },
    {
      "path": "scgpt/results/memory_batch_refine_20261027_214026_294469/exit_code.json",
      "bytes": 16,
      "source_sha256": "124fd50cb6d223cdd18fedc166d00af7254199b6e6dedf9be82f71db87af6d01"
    },
    {
      "path": "scgpt/results/memory_batch_refine_20261027_214026_294469/command.json",
      "bytes": 187,
      "source_sha256": "8a0efcafcf7f096391639683706e0c7aa0511c83d29af8b9d4a87415dc6bd60e"
    },
    {
      "path": "scgpt/results/memory_batch_refine_20261027_214026_294469/model_config.json",
      "bytes": 430,
      "source_sha256": "14eedc0514e813a820ed37ddff5dd474d8974b1b9a26a62c10909b3b27e2aace"
    },
    {
      "path": "scgpt/results/actual_300k_resume_20261028_093701_696230/summary.json",
      "bytes": 9622,
      "source_sha256": "69463e067678423c40176f7556f3c5066a438af93e02d3fa69092a5dd5af62c8"
    },
    {
      "path": "scgpt/results/actual_300k_resume_20261028_093701_696230/resources.csv",
      "bytes": 1233570,
      "source_sha256": "bb26aa4c533a1742df9fd084cabed2654e3fa9f2fa40d94f53b195eb4a9e5ccc"
    },
    {
      "path": "scgpt/results/actual_300k_resume_20261028_093701_696230/resume_state.json",
      "bytes": 2850,
      "source_sha256": "bccaaf8b620d5be030a5279cb429031721eda787a5b0203e1d2a86d98c2d0310"
    },
    {
      "path": "scgpt/results/actual_300k_resume_20261028_093701_696230/inference_exit.json",
      "bytes": 48,
      "source_sha256": "84eac1bd721d29379b15b74e75963ddf4e11bb098b33119cc2919a4a1b34b221"
    },
    {
      "path": "scgpt/results/actual_300k_resume_20261028_093701_696230/inference_command.json",
      "bytes": 335,
      "source_sha256": "5a61e731110d427c80d00ffd672b9448789bd213ae95d0e06416bd61f0fdd9c3"
    },
    {
      "path": "scgpt/results/actual_300k_resume_20261028_093701_696230/whole_dataset_comparison.json",
      "bytes": 710,
      "source_sha256": "3517e2b41e7eb2cfd426f3334d2e186f4b536d4f4ce19d5ae34c89813f6dc9b8"
    },
    {
      "path": "scgpt/results/actual_300k_resume_20261028_093701_696230/final_output_verification.json",
      "bytes": 421,
      "source_sha256": "4dc930f278c857c5c963b8fa750a03936c8090f4aaba0c7676b9be9a8d817163"
    },
    {
      "path": "scgpt/results/actual_300k_resume_20261028_093701_696230/finalization_status.json",
      "bytes": 121,
      "source_sha256": "8460fa84c8a469766869808e659e8335bcd8644d6326f04275904ea8c9852339"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/prepare_command.json",
      "bytes": 296,
      "source_sha256": "2969f6fbc763ea21d80cd3cad31d58d6dcf0c3ed41d2436c349e9085b5b37b38"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/prepare_exit.json",
      "bytes": 52,
      "source_sha256": "7803f9893c81ea2cb8c5267a75519dc950b142e59dce08eb3aeed5649e01b433"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/summary.json",
      "bytes": 7040,
      "source_sha256": "98d345ecbe4d888538522de0bd2cb0a0a1285463830a3cbdc2ef9571c244d613"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/final_status.json",
      "bytes": 182,
      "source_sha256": "64bbcd907de0d4872bb87510ff84314beb3304d322baef99e460944c8932004d"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/pause_output_verification.json",
      "bytes": 183,
      "source_sha256": "2bdce0c6d30738d3922ec4baa712095c7172910f7263a00db7f904d1ac8b2ed4"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/inference_retry1_exit.json",
      "bytes": 145,
      "source_sha256": "d20aff265d0269798815901debab01960e9a960ff20e4c992d6aefb8bf2b33d1"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/resources.csv",
      "bytes": 283557,
      "source_sha256": "c6cf4bd9ce08becd357877a7dea23cf0ba6b2c3d872ca8d4b10045a6fee82f78"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/inference_exit.json",
      "bytes": 52,
      "source_sha256": "2c7bb794ac03df12f267e3d47125605a3ad01903cce654b6801836dbf2c61648"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/inference_command.json",
      "bytes": 318,
      "source_sha256": "2020ba6849ab446d2d54654ed548df3f0b681e0e655ecdfdb9d94c6b47b1e79e"
    },
    {
      "path": "scgpt/results/actual_300k_20261027_221433_021330/pause_checkpoint.json",
      "bytes": 2850,
      "source_sha256": "bccaaf8b620d5be030a5279cb429031721eda787a5b0203e1d2a86d98c2d0310"
    },
    {
      "path": "scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001/manifest.json",
      "bytes": 355,
      "source_sha256": "6bf8288aefad65d2b3cad1c7e5f89bea5e23bb93e980f0f4b63ee4aad8b53f25"
    },
    {
      "path": "scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001/run_config.json",
      "bytes": 2602,
      "source_sha256": "4e3f60131b23a6ec6abda16fa18221cfcf9a4a160f493db088989b38e0309885"
    },
    {
      "path": "scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/models/finetuned/realdata_full_20261027_215022_050816/batch_8/epoch_001/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/models/finetuned/finetune_20261023_033701_500560/epoch_001/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/models/finetuned/finetune_20261023_033701_500560/epoch_001/manifest.json",
      "bytes": 336,
      "source_sha256": "a1b81cd9dacdbe45a58217dfde8558b2ce984d3e3e14cd29a2ed95e1e3927f2a"
    },
    {
      "path": "scgpt/models/finetuned/finetune_20261023_033701_500560/epoch_001/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/models/finetuned/finetune_20261023_033701_500560/epoch_001/run_config.json",
      "bytes": 2602,
      "source_sha256": "ebca3341cd7b6a7709c0bf7fb9edc2f2de17b10d87854a87667dbfe2964b42ad"
    },
    {
      "path": "scgpt/models/finetuned/finetune_20261023_033701_500560/epoch_001/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/models/finetuned/finetune_20261023_033701_500560/epoch_001/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/models/finetuned/finetune_20261023_033701_500560/epoch_001/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001/manifest.json",
      "bytes": 352,
      "source_sha256": "72cbaa072b5b6c1d5cf69aed0c2068d7fb83aa29598a9435f9bcd54059043553"
    },
    {
      "path": "scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001/run_config.json",
      "bytes": 2602,
      "source_sha256": "ebca3341cd7b6a7709c0bf7fb9edc2f2de17b10d87854a87667dbfe2964b42ad"
    },
    {
      "path": "scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/models/finetuned/supplement_20261027_211638_936991/epoch_001/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001/manifest.json",
      "bytes": 357,
      "source_sha256": "86527bbd9326e11b48918e824929035339196cd358458aad65aaf99851772259"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001/run_config.json",
      "bytes": 2602,
      "source_sha256": "ebca3341cd7b6a7709c0bf7fb9edc2f2de17b10d87854a87667dbfe2964b42ad"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001/preprocessing.json",
      "bytes": 587,
      "source_sha256": "23b107866a21a2bc2b45a35cbd79d769a5f6414652248b9c14c12a72bebc9aef"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1200/epoch_001/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001/manifest.json",
      "bytes": 357,
      "source_sha256": "b5c4d35b8e1086992eb1de9d4ce30cf38ee66ce31a74039172dc9eb421409b2d"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001/run_config.json",
      "bytes": 2601,
      "source_sha256": "4a5eadbf2e18433e11afa381bc855b27cc840884fa60c3fab7d82afb7d302816"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001/preprocessing.json",
      "bytes": 586,
      "source_sha256": "f62eebb58282fb9d7d2eae3063aef867356a7cac0a14287901119a68093d343d"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0256/epoch_001/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001/manifest.json",
      "bytes": 357,
      "source_sha256": "b2fa7f51af5e714e14db50c2faea1cc1a71d3a9f8c66272eb56ab754739c2eeb"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001/run_config.json",
      "bytes": 2602,
      "source_sha256": "9425e4f2e43229b3cb899c0da73e66fb034ef5b9ee9f89288d21cfce925f7e8c"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001/preprocessing.json",
      "bytes": 587,
      "source_sha256": "cb7c30419cf9112a35ec603d10b5929c834f14fd8e9371d2d092d3b09106d627"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L1024/epoch_001/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001/labels.json",
      "bytes": 446,
      "source_sha256": "4daf09326888da5af8173335d3026a0d236aa52b47f32698fb844663cfdf83f6"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001/manifest.json",
      "bytes": 357,
      "source_sha256": "1cb978b49b711dbaf8b6f969293d2648d2cc543ad44ed13edccb9a8a0b62287c"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001/genes.json",
      "bytes": 33494,
      "source_sha256": "fd86a15d742edcd7c2103425c4404439f5014c4bd8d9ca46b9b2a140b8572b92"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001/run_config.json",
      "bytes": 2601,
      "source_sha256": "9682f7ea4e7c7112690de4e7e99be52ccdc0f00de1ded74255b0ff188658eb5d"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001/preprocessing.json",
      "bytes": 586,
      "source_sha256": "b8b9fd6d2b1ddf715565074030c026b3aedeb183a914227b62c572116a9f21b8"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001/model_config.json",
      "bytes": 431,
      "source_sha256": "3c519ef10ace7348a7aafc52ba6b26ee9ba11df9f7ea65681f4bd10489e59aa5"
    },
    {
      "path": "scgpt/models/finetuned/length_compare_20261023_040313_501423/L0512/epoch_001/pretrained_args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scgpt/models/whole-human/args.json",
      "bytes": 1300,
      "source_sha256": "c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60"
    },
    {
      "path": "scvi/constraints.txt",
      "bytes": 90,
      "source_sha256": "df79342bbebeaa4b61ad0af476674c5068aaba817a92b3258f79fe9a3c3ff183"
    },
    {
      "path": "cellpose/constraints.txt",
      "bytes": 90,
      "source_sha256": "df79342bbebeaa4b61ad0af476674c5068aaba817a92b3258f79fe9a3c3ff183"
    },
    {
      "path": "ntv2/constraints.txt",
      "bytes": 43,
      "source_sha256": "0682fa586a739091d790d29324bff7de38f467ff1389bb10ed8310c6a57545ee"
    },
    {
      "path": "clair3/Dockerfile",
      "bytes": 280,
      "source_sha256": "042a78fd10943f77204509c11fc0da187828bed569ef522383d9f045d363152e"
    },
    {
      "path": "clair3/constraints.txt",
      "bytes": 43,
      "source_sha256": "0682fa586a739091d790d29324bff7de38f467ff1389bb10ed8310c6a57545ee"
    },
    {
      "path": "openspliceai/constraints.txt",
      "bytes": 43,
      "source_sha256": "0682fa586a739091d790d29324bff7de38f467ff1389bb10ed8310c6a57545ee"
    }
  ],
  "credential_headers_redacted": [],
  "markdown_paths_adapted": [
    "ntv2/reports/NTv2.md",
    "rnafm/reports/RNAFM.md",
    "dnabert2/reports/DNABERT2.md",
    "openspliceai/reports/OpenSpliceAI.md",
    "scgpt/reports/scGPT_actual_300k.md",
    "scgpt/reports/scGPT_memory_batch.md",
    "clair3/reports/Clair3.md",
    "cellpose/reports/Cellpose.md",
    "cellpose/reports/Cellpose_cellpose_20261027_193331_947064.md"
  ],
  "local_artifact_links": 12,
  "publication_transformations": [
    "Text line endings normalized to LF",
    "Python trailing whitespace removed",
    "Local artifact Markdown links adapted for GitHub",
    "Original source hashes refer to local originals before these publication transformations"
  ],
  "integrated_report_update": {
    "entrypoint": "FINAL_REPORT.md",
    "models": 10,
    "each_model": "<model>/reports/FINAL.md includes base and followup bodies",
    "memory_80percent_goal": "only scGPT fixed-padding training capacity probe confirmed; other model conditions unconfirmed"
  }
}
```
<!-- END_RESTORE_FILE -->


</details>
