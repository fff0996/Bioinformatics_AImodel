# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-4c66cd9b8b0d

원래 경로: `cellpose/data/input_audit.json` · SHA256: `82ab441581106a6e4b99e5aa44cc0fde1ca278a0d93cfc61737734b22d004c61`

<details>
<summary>input_audit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "cellpose/data/input_audit.json", "sha256": "82ab441581106a6e4b99e5aa44cc0fde1ca278a0d93cfc61737734b22d004c61", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "all": [
    {
      "id": "IXMtest_A06_s6_w1B9577918-4973-4A87-BA73-A168AA755527",
      "split": "training",
      "image": "data/images/images/IXMtest_A06_s6_w1B9577918-4973-4A87-BA73-A168AA755527.tif",
      "source_mask": "data/masks/masks/IXMtest_A06_s6_w1B9577918-4973-4A87-BA73-A168AA755527.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 1998,
      "instances": 71,
      "image_sha256": "f9a51ae558e53416980ca152c740ad9dced85eedbd105261841139aada16b77d",
      "selected": false
    },
    {
      "id": "IXMtest_A15_s5_w1825174D4-ED30-490C-9635-6196417D6C9D",
      "split": "training",
      "image": "data/images/images/IXMtest_A15_s5_w1825174D4-ED30-490C-9635-6196417D6C9D.tif",
      "source_mask": "data/masks/masks/IXMtest_A15_s5_w1825174D4-ED30-490C-9635-6196417D6C9D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 2224,
      "instances": 129,
      "image_sha256": "79d8d52851dd1fdff52bc8b480e5201415c124f2793c6e3a7c9c9926cf5658fa",
      "selected": false
    },
    {
      "id": "IXMtest_A16_s3_w1032BE329-E21B-4E1B-B4B8-58700685EE0C",
      "split": "training",
      "image": "data/images/images/IXMtest_A16_s3_w1032BE329-E21B-4E1B-B4B8-58700685EE0C.tif",
      "source_mask": "data/masks/masks/IXMtest_A16_s3_w1032BE329-E21B-4E1B-B4B8-58700685EE0C.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 3409,
      "instances": 117,
      "image_sha256": "31d2264d2dcc27bb2c4aedd06805c10f9cd953e2f0893824fd070f29524e3a2a",
      "selected": false
    },
    {
      "id": "IXMtest_A18_s1_w1BFDF1C94-9C1F-4F5F-BBC1-05196333B1BF",
      "split": "training",
      "image": "data/images/images/IXMtest_A18_s1_w1BFDF1C94-9C1F-4F5F-BBC1-05196333B1BF.tif",
      "source_mask": "data/masks/masks/IXMtest_A18_s1_w1BFDF1C94-9C1F-4F5F-BBC1-05196333B1BF.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 129,
      "max": 2505,
      "instances": 102,
      "image_sha256": "c118bfba48188ad76d17c36135384f7658bf98c4399d97a037b3cd0623ce78ff",
      "selected": false
    },
    {
      "id": "IXMtest_A20_s4_w153DE191F-B112-471B-AC0A-FAD27990714B",
      "split": "training",
      "image": "data/images/images/IXMtest_A20_s4_w153DE191F-B112-471B-AC0A-FAD27990714B.tif",
      "source_mask": "data/masks/masks/IXMtest_A20_s4_w153DE191F-B112-471B-AC0A-FAD27990714B.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 115,
      "max": 1526,
      "instances": 133,
      "image_sha256": "588808cacb2985263edd7afff831e1391b37b83f796a496d124b1cc3ebde59e6",
      "selected": false
    },
    {
      "id": "IXMtest_A21_s1_w1D8BF2790-C628-4109-94AF-DF6D8A3480B6",
      "split": "training",
      "image": "data/images/images/IXMtest_A21_s1_w1D8BF2790-C628-4109-94AF-DF6D8A3480B6.tif",
      "source_mask": "data/masks/masks/IXMtest_A21_s1_w1D8BF2790-C628-4109-94AF-DF6D8A3480B6.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 128,
      "max": 1915,
      "instances": 75,
      "image_sha256": "5029f62a7cd06ad30d528ac22fa83a94ac0f2aef9d01f041ff96e80ea621ae9d",
      "selected": false
    },
    {
      "id": "IXMtest_A24_s9_w152CD6793-DA45-4975-869A-CBD49D645E37",
      "split": "training",
      "image": "data/images/images/IXMtest_A24_s9_w152CD6793-DA45-4975-869A-CBD49D645E37.tif",
      "source_mask": "data/masks/masks/IXMtest_A24_s9_w152CD6793-DA45-4975-869A-CBD49D645E37.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 1362,
      "instances": 53,
      "image_sha256": "be0de5924e22ab3f7ff7e39cc041b8f3c71d4f0ee2f3deff9a29f7ce102133e1",
      "selected": false
    },
    {
      "id": "IXMtest_B12_s9_w1ABE4C6B8-6110-4D7D-B782-94C43FDC2E1F",
      "split": "training",
      "image": "data/images/images/IXMtest_B12_s9_w1ABE4C6B8-6110-4D7D-B782-94C43FDC2E1F.tif",
      "source_mask": "data/masks/masks/IXMtest_B12_s9_w1ABE4C6B8-6110-4D7D-B782-94C43FDC2E1F.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 3398,
      "instances": 109,
      "image_sha256": "6d9c79e14473510fce59f9634e25eca437dff8893de5143f8d27c3a9ece321fc",
      "selected": true,
      "mask": "data/prepared/IXMtest_B12_s9_w1ABE4C6B8-6110-4D7D-B782-94C43FDC2E1F_mask.npy"
    },
    {
      "id": "IXMtest_B20_s2_w159B9FE71-035A-4DED-B0CA-C76916C968BC",
      "split": "training",
      "image": "data/images/images/IXMtest_B20_s2_w159B9FE71-035A-4DED-B0CA-C76916C968BC.tif",
      "source_mask": "data/masks/masks/IXMtest_B20_s2_w159B9FE71-035A-4DED-B0CA-C76916C968BC.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 1986,
      "instances": 88,
      "image_sha256": "cf3e628f13040792d3ac0c6b07103ac9a8084afc2b01063e93867b553cb97d4b",
      "selected": true,
      "mask": "data/prepared/IXMtest_B20_s2_w159B9FE71-035A-4DED-B0CA-C76916C968BC_mask.npy"
    },
    {
      "id": "IXMtest_B21_s3_w141E785B1-44FE-4ED0-9CCE-6FF076EFE9FE",
      "split": "training",
      "image": "data/images/images/IXMtest_B21_s3_w141E785B1-44FE-4ED0-9CCE-6FF076EFE9FE.tif",
      "source_mask": "data/masks/masks/IXMtest_B21_s3_w141E785B1-44FE-4ED0-9CCE-6FF076EFE9FE.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 2141,
      "instances": 191,
      "image_sha256": "ca881a90b4820efb9c29a3e21acf72809f105ba32823a0760265bd70b1643083",
      "selected": false
    },
    {
      "id": "IXMtest_B21_s4_w1521471E0-9BD7-492A-8739-9C782C2585B0",
      "split": "training",
      "image": "data/images/images/IXMtest_B21_s4_w1521471E0-9BD7-492A-8739-9C782C2585B0.tif",
      "source_mask": "data/masks/masks/IXMtest_B21_s4_w1521471E0-9BD7-492A-8739-9C782C2585B0.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 2648,
      "instances": 107,
      "image_sha256": "d57aa7128c0bbac8538ab132b25329f2367363419498e4a6560b21b81e75a9b8",
      "selected": false
    },
    {
      "id": "IXMtest_B21_s7_w1276972A2-1794-4A98-8E76-876A597C346B",
      "split": "training",
      "image": "data/images/images/IXMtest_B21_s7_w1276972A2-1794-4A98-8E76-876A597C346B.tif",
      "source_mask": "data/masks/masks/IXMtest_B21_s7_w1276972A2-1794-4A98-8E76-876A597C346B.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 114,
      "max": 1967,
      "instances": 112,
      "image_sha256": "67fb11766acfe130d72bfe2dab43b070613c2ef95b6b8e18697208f46b285147",
      "selected": false
    },
    {
      "id": "IXMtest_B22_s6_w19397283D-2426-4942-8A0F-EB75E1D48C2D",
      "split": "training",
      "image": "data/images/images/IXMtest_B22_s6_w19397283D-2426-4942-8A0F-EB75E1D48C2D.tif",
      "source_mask": "data/masks/masks/IXMtest_B22_s6_w19397283D-2426-4942-8A0F-EB75E1D48C2D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 124,
      "max": 2951,
      "instances": 83,
      "image_sha256": "c6f2d56a484d38e4eb59424d44ccd14e064a8a7f88e954fd96f7a08fa2307431",
      "selected": false
    },
    {
      "id": "IXMtest_B22_s8_w10754C18F-B059-47B4-A423-FF429B984D80",
      "split": "training",
      "image": "data/images/images/IXMtest_B22_s8_w10754C18F-B059-47B4-A423-FF429B984D80.tif",
      "source_mask": "data/masks/masks/IXMtest_B22_s8_w10754C18F-B059-47B4-A423-FF429B984D80.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 3264,
      "instances": 74,
      "image_sha256": "8158a8253e9a5d040b339123c117248259569765aa0f4296ca1b2ebab5564c0e",
      "selected": false
    },
    {
      "id": "IXMtest_B23_s1_w152C21D3B-75A9-4FF9-839A-23E6CE19AEEC",
      "split": "training",
      "image": "data/images/images/IXMtest_B23_s1_w152C21D3B-75A9-4FF9-839A-23E6CE19AEEC.tif",
      "source_mask": "data/masks/masks/IXMtest_B23_s1_w152C21D3B-75A9-4FF9-839A-23E6CE19AEEC.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 993,
      "instances": 86,
      "image_sha256": "3dfe1d6214ce55eb2b3f288bc55b43d73574811e5c7b7c1618ac816842093f82",
      "selected": false
    },
    {
      "id": "IXMtest_C02_s4_w1661FF37C-AB9B-4038-BDF3-D21DFD2811BE",
      "split": "training",
      "image": "data/images/images/IXMtest_C02_s4_w1661FF37C-AB9B-4038-BDF3-D21DFD2811BE.tif",
      "source_mask": "data/masks/masks/IXMtest_C02_s4_w1661FF37C-AB9B-4038-BDF3-D21DFD2811BE.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 1034,
      "instances": 49,
      "image_sha256": "da771f81e81847163adf4c9810baddf28e19cf221f5f7d015c20c1684ddf71bb",
      "selected": false
    },
    {
      "id": "IXMtest_C05_s7_w1F71963FB-8F29-41CB-A5F5-07CB9584BBC5",
      "split": "training",
      "image": "data/images/images/IXMtest_C05_s7_w1F71963FB-8F29-41CB-A5F5-07CB9584BBC5.tif",
      "source_mask": "data/masks/masks/IXMtest_C05_s7_w1F71963FB-8F29-41CB-A5F5-07CB9584BBC5.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 4095,
      "instances": 144,
      "image_sha256": "fa938c297a652ff50892b7935259fc3aa7288b809215cc7080cf85ab63ec667e",
      "selected": false
    },
    {
      "id": "IXMtest_C09_s7_w1768A3B0D-47FE-4D77-B1C6-46018E29486F",
      "split": "training",
      "image": "data/images/images/IXMtest_C09_s7_w1768A3B0D-47FE-4D77-B1C6-46018E29486F.tif",
      "source_mask": "data/masks/masks/IXMtest_C09_s7_w1768A3B0D-47FE-4D77-B1C6-46018E29486F.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 1501,
      "instances": 164,
      "image_sha256": "7a5b200b3b8e0c6901c28e1602abb7bed046018d692c6bc3fac2aa7f6cc9cb89",
      "selected": false
    },
    {
      "id": "IXMtest_C19_s4_w1AD3DC23E-B9B5-46E9-9716-26096C672A13",
      "split": "training",
      "image": "data/images/images/IXMtest_C19_s4_w1AD3DC23E-B9B5-46E9-9716-26096C672A13.tif",
      "source_mask": "data/masks/masks/IXMtest_C19_s4_w1AD3DC23E-B9B5-46E9-9716-26096C672A13.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 2967,
      "instances": 124,
      "image_sha256": "04b171051f9062520d57d042be72499be9d28aed486a95b664e0f35e089247ed",
      "selected": true,
      "mask": "data/prepared/IXMtest_C19_s4_w1AD3DC23E-B9B5-46E9-9716-26096C672A13_mask.npy"
    },
    {
      "id": "IXMtest_C21_s9_w1289444BE-8623-4867-86E7-01E155C6538D",
      "split": "training",
      "image": "data/images/images/IXMtest_C21_s9_w1289444BE-8623-4867-86E7-01E155C6538D.tif",
      "source_mask": "data/masks/masks/IXMtest_C21_s9_w1289444BE-8623-4867-86E7-01E155C6538D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 3899,
      "instances": 157,
      "image_sha256": "ee339763471da5af6d9c4204e183d1592a603327fc385c760ee67a4f6b211baf",
      "selected": false
    },
    {
      "id": "IXMtest_D04_s2_w1E05C134E-BAFE-4FD7-8116-2D05E0839879",
      "split": "training",
      "image": "data/images/images/IXMtest_D04_s2_w1E05C134E-BAFE-4FD7-8116-2D05E0839879.tif",
      "source_mask": "data/masks/masks/IXMtest_D04_s2_w1E05C134E-BAFE-4FD7-8116-2D05E0839879.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 124,
      "max": 3121,
      "instances": 144,
      "image_sha256": "e0f0d573f8f1aecd4d74ae681705b19c408ccc18d8c6cb66bdc60a81709569d3",
      "selected": false
    },
    {
      "id": "IXMtest_D04_s9_w17B6268DB-8215-4DC6-943C-CC009A8A5732",
      "split": "training",
      "image": "data/images/images/IXMtest_D04_s9_w17B6268DB-8215-4DC6-943C-CC009A8A5732.tif",
      "source_mask": "data/masks/masks/IXMtest_D04_s9_w17B6268DB-8215-4DC6-943C-CC009A8A5732.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 3251,
      "instances": 117,
      "image_sha256": "8cf0b9b7ed93431b71f4bdac3e0f59fc1f92a8c4c7fba4469d5b728258b3a838",
      "selected": false
    },
    {
      "id": "IXMtest_D06_s5_w13C67AAA9-6E81-42DB-AC5F-7126602F3607",
      "split": "training",
      "image": "data/images/images/IXMtest_D06_s5_w13C67AAA9-6E81-42DB-AC5F-7126602F3607.tif",
      "source_mask": "data/masks/masks/IXMtest_D06_s5_w13C67AAA9-6E81-42DB-AC5F-7126602F3607.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 113,
      "max": 3086,
      "instances": 160,
      "image_sha256": "a770c48972924d9040d10a383c91e8e6108596fd731948681bf8b0561d7fc9e7",
      "selected": false
    },
    {
      "id": "IXMtest_D20_s9_w1588B4C42-5A0D-4B93-9A80-A6527CC1C411",
      "split": "training",
      "image": "data/images/images/IXMtest_D20_s9_w1588B4C42-5A0D-4B93-9A80-A6527CC1C411.tif",
      "source_mask": "data/masks/masks/IXMtest_D20_s9_w1588B4C42-5A0D-4B93-9A80-A6527CC1C411.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 2374,
      "instances": 154,
      "image_sha256": "7cdb4040907032a489f5b31457f615fdf573d166bd881073596fee852c3d2848",
      "selected": false
    },
    {
      "id": "IXMtest_E04_s4_w1F5ABC12B-FABD-4E33-AB99-82271209853E",
      "split": "training",
      "image": "data/images/images/IXMtest_E04_s4_w1F5ABC12B-FABD-4E33-AB99-82271209853E.tif",
      "source_mask": "data/masks/masks/IXMtest_E04_s4_w1F5ABC12B-FABD-4E33-AB99-82271209853E.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 124,
      "max": 3250,
      "instances": 135,
      "image_sha256": "6b698da8f363a03508b4ef54bc4912c161077b3c64763aa5acb8ba047e759dd1",
      "selected": false
    },
    {
      "id": "IXMtest_E05_s2_w15CCE97F8-3F02-412E-8BF2-FC92972DDA1A",
      "split": "training",
      "image": "data/images/images/IXMtest_E05_s2_w15CCE97F8-3F02-412E-8BF2-FC92972DDA1A.tif",
      "source_mask": "data/masks/masks/IXMtest_E05_s2_w15CCE97F8-3F02-412E-8BF2-FC92972DDA1A.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 129,
      "max": 3401,
      "instances": 180,
      "image_sha256": "aa61ad91c499ed2134718bfee54ac0084f043246d6f6649fe4f804a3f915a710",
      "selected": false
    },
    {
      "id": "IXMtest_E08_s2_w1588EDEBA-1BBA-4E49-810F-2EF2064C8846",
      "split": "training",
      "image": "data/images/images/IXMtest_E08_s2_w1588EDEBA-1BBA-4E49-810F-2EF2064C8846.tif",
      "source_mask": "data/masks/masks/IXMtest_E08_s2_w1588EDEBA-1BBA-4E49-810F-2EF2064C8846.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 2573,
      "instances": 147,
      "image_sha256": "db4d20917008c454457d906ac748aaa23df7a7c81b4eb9b8a7b6b177c05422db",
      "selected": false
    },
    {
      "id": "IXMtest_E15_s5_w1419F04CC-50C3-41FB-A87A-FB17647A2E33",
      "split": "training",
      "image": "data/images/images/IXMtest_E15_s5_w1419F04CC-50C3-41FB-A87A-FB17647A2E33.tif",
      "source_mask": "data/masks/masks/IXMtest_E15_s5_w1419F04CC-50C3-41FB-A87A-FB17647A2E33.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 115,
      "max": 3964,
      "instances": 158,
      "image_sha256": "18d6e8b506f0a911d1cb1cf32ca8156b8486c2e02f9adc7b78cab90d604453b9",
      "selected": false
    },
    {
      "id": "IXMtest_E19_s2_w1752F162C-2C91-4A67-BF4F-D073C3D88291",
      "split": "training",
      "image": "data/images/images/IXMtest_E19_s2_w1752F162C-2C91-4A67-BF4F-D073C3D88291.tif",
      "source_mask": "data/masks/masks/IXMtest_E19_s2_w1752F162C-2C91-4A67-BF4F-D073C3D88291.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 126,
      "max": 4095,
      "instances": 149,
      "image_sha256": "a724dbd8e9323f81ff59cbb5602e6cbb40d1e663e21a21924f655a02ac1eac87",
      "selected": false
    },
    {
      "id": "IXMtest_E19_s4_w129545707-4CD3-4498-AC27-E4AE24D0253C",
      "split": "training",
      "image": "data/images/images/IXMtest_E19_s4_w129545707-4CD3-4498-AC27-E4AE24D0253C.tif",
      "source_mask": "data/masks/masks/IXMtest_E19_s4_w129545707-4CD3-4498-AC27-E4AE24D0253C.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 1577,
      "instances": 127,
      "image_sha256": "e4d6d989bfe071457d1d9b3c41a03e9dc968347fb1f096c9ec97188af0df1980",
      "selected": false
    },
    {
      "id": "IXMtest_E23_s8_w10F1F106B-057A-482C-93BB-C9A0B044C054",
      "split": "training",
      "image": "data/images/images/IXMtest_E23_s8_w10F1F106B-057A-482C-93BB-C9A0B044C054.tif",
      "source_mask": "data/masks/masks/IXMtest_E23_s8_w10F1F106B-057A-482C-93BB-C9A0B044C054.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 114,
      "max": 1110,
      "instances": 96,
      "image_sha256": "50b05db02d4646ad6d95ad4d309a5aafe04d40be016f22e0ce349972fa4db893",
      "selected": false
    },
    {
      "id": "IXMtest_F01_s2_w189F7DC13-4572-43C9-8C23-CD9AEB41B24A",
      "split": "training",
      "image": "data/images/images/IXMtest_F01_s2_w189F7DC13-4572-43C9-8C23-CD9AEB41B24A.tif",
      "source_mask": "data/masks/masks/IXMtest_F01_s2_w189F7DC13-4572-43C9-8C23-CD9AEB41B24A.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 1774,
      "instances": 102,
      "image_sha256": "547f66e0cabbc70ebf187fde28e52647d50ba64b63e1990824b3016182e67f8d",
      "selected": false
    },
    {
      "id": "IXMtest_F07_s5_w15678F19D-6F23-4FB4-8851-5B57CF8CCCF8",
      "split": "training",
      "image": "data/images/images/IXMtest_F07_s5_w15678F19D-6F23-4FB4-8851-5B57CF8CCCF8.tif",
      "source_mask": "data/masks/masks/IXMtest_F07_s5_w15678F19D-6F23-4FB4-8851-5B57CF8CCCF8.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 3189,
      "instances": 133,
      "image_sha256": "d802552c7cb02829dad8f6dee2f0020af71c54dd5c1709ef66a9247cbcf0d0fe",
      "selected": false
    },
    {
      "id": "IXMtest_F12_s9_w1CD127D58-4641-41D3-88CC-E12B7FE77D78",
      "split": "training",
      "image": "data/images/images/IXMtest_F12_s9_w1CD127D58-4641-41D3-88CC-E12B7FE77D78.tif",
      "source_mask": "data/masks/masks/IXMtest_F12_s9_w1CD127D58-4641-41D3-88CC-E12B7FE77D78.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 4095,
      "instances": 86,
      "image_sha256": "ede3024d112c350aaac7f2cf8a92d79c0ddb58623fa53c7ad7558cf89d982192",
      "selected": false
    },
    {
      "id": "IXMtest_F13_s7_w13C1B1D8C-293E-454F-B0FD-6C2C3F9F5173",
      "split": "training",
      "image": "data/images/images/IXMtest_F13_s7_w13C1B1D8C-293E-454F-B0FD-6C2C3F9F5173.tif",
      "source_mask": "data/masks/masks/IXMtest_F13_s7_w13C1B1D8C-293E-454F-B0FD-6C2C3F9F5173.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 208,
      "instances": 0,
      "image_sha256": "97ecd7bcff6e74cc92e7d945f13adddb218776b0de1ac0a5bfac8bcf1a42fc95",
      "selected": false
    },
    {
      "id": "IXMtest_F21_s1_w1E2770CF6-417B-4804-9B45-6EE7D783D8CA",
      "split": "training",
      "image": "data/images/images/IXMtest_F21_s1_w1E2770CF6-417B-4804-9B45-6EE7D783D8CA.tif",
      "source_mask": "data/masks/masks/IXMtest_F21_s1_w1E2770CF6-417B-4804-9B45-6EE7D783D8CA.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 128,
      "max": 3314,
      "instances": 148,
      "image_sha256": "15e1a158c75840cc90a19a5c4e8b03ab41641cdfab0a389bf8f85025eb589053",
      "selected": false
    },
    {
      "id": "IXMtest_F22_s6_w1D6C33377-3947-4DAC-B62D-531C2565F150",
      "split": "training",
      "image": "data/images/images/IXMtest_F22_s6_w1D6C33377-3947-4DAC-B62D-531C2565F150.tif",
      "source_mask": "data/masks/masks/IXMtest_F22_s6_w1D6C33377-3947-4DAC-B62D-531C2565F150.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 2885,
      "instances": 127,
      "image_sha256": "33c7137902b0841e597d87cc2dbe2fee5230534d113d9b0363dfb4f931e99b74",
      "selected": false
    },
    {
      "id": "IXMtest_G05_s6_w1AA0013DE-ED7A-428E-A17A-09959E7DC1A2",
      "split": "training",
      "image": "data/images/images/IXMtest_G05_s6_w1AA0013DE-ED7A-428E-A17A-09959E7DC1A2.tif",
      "source_mask": "data/masks/masks/IXMtest_G05_s6_w1AA0013DE-ED7A-428E-A17A-09959E7DC1A2.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 1702,
      "instances": 136,
      "image_sha256": "da89e4025aebadf44ccd279aebf4a6b66c3114676718fbc1b551e259bde1e969",
      "selected": false
    },
    {
      "id": "IXMtest_G06_s3_w19ABE30DE-4BFC-4CD5-AD16-6C8BD6E69F9D",
      "split": "training",
      "image": "data/images/images/IXMtest_G06_s3_w19ABE30DE-4BFC-4CD5-AD16-6C8BD6E69F9D.tif",
      "source_mask": "data/masks/masks/IXMtest_G06_s3_w19ABE30DE-4BFC-4CD5-AD16-6C8BD6E69F9D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 3936,
      "instances": 169,
      "image_sha256": "079b69b374ece48120645e4c3b88a532a4be2c8bed16258f8621ecc3ac60bae5",
      "selected": true,
      "mask": "data/prepared/IXMtest_G06_s3_w19ABE30DE-4BFC-4CD5-AD16-6C8BD6E69F9D_mask.npy"
    },
    {
      "id": "IXMtest_G08_s6_w1823194BE-DE9A-4638-BACD-CABE5FA4C89C",
      "split": "training",
      "image": "data/images/images/IXMtest_G08_s6_w1823194BE-DE9A-4638-BACD-CABE5FA4C89C.tif",
      "source_mask": "data/masks/masks/IXMtest_G08_s6_w1823194BE-DE9A-4638-BACD-CABE5FA4C89C.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 3086,
      "instances": 126,
      "image_sha256": "5d09f5913592f964e6a0acaa9132dd388032006b6148d837bb7123a7db840f81",
      "selected": false
    },
    {
      "id": "IXMtest_G16_s3_w1B4690283-D75E-4DFB-92B0-29419E8292C6",
      "split": "training",
      "image": "data/images/images/IXMtest_G16_s3_w1B4690283-D75E-4DFB-92B0-29419E8292C6.tif",
      "source_mask": "data/masks/masks/IXMtest_G16_s3_w1B4690283-D75E-4DFB-92B0-29419E8292C6.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 131,
      "max": 3565,
      "instances": 128,
      "image_sha256": "2fb7f07b5eb2083704bb1f353e39e30c2eb314d998166f978df2337cbeb3f480",
      "selected": false
    },
    {
      "id": "IXMtest_G16_s6_w1C3F175E2-0AF5-401C-AC5C-8E128A237B22",
      "split": "training",
      "image": "data/images/images/IXMtest_G16_s6_w1C3F175E2-0AF5-401C-AC5C-8E128A237B22.tif",
      "source_mask": "data/masks/masks/IXMtest_G16_s6_w1C3F175E2-0AF5-401C-AC5C-8E128A237B22.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 130,
      "max": 4095,
      "instances": 185,
      "image_sha256": "bf4d24c9061b14f289aa09d30f262161dd506db80a2822eb2a7d816d79f14005",
      "selected": false
    },
    {
      "id": "IXMtest_G23_s4_w126F855D5-89CE-432B-9592-684B6E5EEF24",
      "split": "training",
      "image": "data/images/images/IXMtest_G23_s4_w126F855D5-89CE-432B-9592-684B6E5EEF24.tif",
      "source_mask": "data/masks/masks/IXMtest_G23_s4_w126F855D5-89CE-432B-9592-684B6E5EEF24.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 2846,
      "instances": 148,
      "image_sha256": "f7f8ead96da6ad1d390e478c012d93fca062c0c0bd530fca365c354d29bab379",
      "selected": false
    },
    {
      "id": "IXMtest_G23_s8_w12910C37A-C52C-45B2-A796-7C60A2247C32",
      "split": "training",
      "image": "data/images/images/IXMtest_G23_s8_w12910C37A-C52C-45B2-A796-7C60A2247C32.tif",
      "source_mask": "data/masks/masks/IXMtest_G23_s8_w12910C37A-C52C-45B2-A796-7C60A2247C32.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 2916,
      "instances": 88,
      "image_sha256": "70594c9f86df856c3b658fc3c9238cd173a557849bf187282af12d213fbf5153",
      "selected": false
    },
    {
      "id": "IXMtest_H01_s8_w13C7E7AAD-FC32-42F8-88C5-DDEA4BD8832C",
      "split": "training",
      "image": "data/images/images/IXMtest_H01_s8_w13C7E7AAD-FC32-42F8-88C5-DDEA4BD8832C.tif",
      "source_mask": "data/masks/masks/IXMtest_H01_s8_w13C7E7AAD-FC32-42F8-88C5-DDEA4BD8832C.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 1242,
      "instances": 92,
      "image_sha256": "ffcd46ca096acdf4079e1c38f015b656fcb42c80401cdfe0578d4ee1bd7625b4",
      "selected": false
    },
    {
      "id": "IXMtest_H01_s9_w1237AA324-894F-4C97-99F3-FEF47F52A550",
      "split": "training",
      "image": "data/images/images/IXMtest_H01_s9_w1237AA324-894F-4C97-99F3-FEF47F52A550.tif",
      "source_mask": "data/masks/masks/IXMtest_H01_s9_w1237AA324-894F-4C97-99F3-FEF47F52A550.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 2332,
      "instances": 104,
      "image_sha256": "69ca4ae509e51c290bc6e164db805297916978032a9a502d9c0456b10b3836cf",
      "selected": false
    },
    {
      "id": "IXMtest_H06_s4_w16261E805-AF5F-4785-BB65-C77B61FF39F7",
      "split": "training",
      "image": "data/images/images/IXMtest_H06_s4_w16261E805-AF5F-4785-BB65-C77B61FF39F7.tif",
      "source_mask": "data/masks/masks/IXMtest_H06_s4_w16261E805-AF5F-4785-BB65-C77B61FF39F7.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 114,
      "max": 1545,
      "instances": 170,
      "image_sha256": "5f0adac2f536dd5c856ab999790bd38eef31c97886d7704bb516ebe6540f07ec",
      "selected": false
    },
    {
      "id": "IXMtest_H06_s6_w1C3C98946-0210-4D6A-831E-5681452CCD1D",
      "split": "training",
      "image": "data/images/images/IXMtest_H06_s6_w1C3C98946-0210-4D6A-831E-5681452CCD1D.tif",
      "source_mask": "data/masks/masks/IXMtest_H06_s6_w1C3C98946-0210-4D6A-831E-5681452CCD1D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 3706,
      "instances": 154,
      "image_sha256": "46f8ff1f494d5f4d1010e37006722240ee0a46b0a3aea4a762b10aed8e5bd121",
      "selected": false
    },
    {
      "id": "IXMtest_H07_s2_w1D8F30687-16A2-4A0D-BDD5-377257089055",
      "split": "training",
      "image": "data/images/images/IXMtest_H07_s2_w1D8F30687-16A2-4A0D-BDD5-377257089055.tif",
      "source_mask": "data/masks/masks/IXMtest_H07_s2_w1D8F30687-16A2-4A0D-BDD5-377257089055.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 1651,
      "instances": 115,
      "image_sha256": "f92b53194712e088ec2e6636394338555dbda4d8401f1e5d2338984accf56532",
      "selected": false
    },
    {
      "id": "IXMtest_H08_s8_w1418B93F8-515E-49A7-A9AE-CAF7A1589A38",
      "split": "training",
      "image": "data/images/images/IXMtest_H08_s8_w1418B93F8-515E-49A7-A9AE-CAF7A1589A38.tif",
      "source_mask": "data/masks/masks/IXMtest_H08_s8_w1418B93F8-515E-49A7-A9AE-CAF7A1589A38.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 2264,
      "instances": 110,
      "image_sha256": "d8e0fe62bf709f798822d07ac76f8b209c78e0aea0fbc4db570b88213984b9cd",
      "selected": false
    },
    {
      "id": "IXMtest_H11_s5_w1612B789B-6279-41E1-B4C6-009BF5772334",
      "split": "training",
      "image": "data/images/images/IXMtest_H11_s5_w1612B789B-6279-41E1-B4C6-009BF5772334.tif",
      "source_mask": "data/masks/masks/IXMtest_H11_s5_w1612B789B-6279-41E1-B4C6-009BF5772334.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 124,
      "max": 2171,
      "instances": 153,
      "image_sha256": "b88f1807ec029c5294cebf00230ba533343375e5838b78f8ec8994f93b6cc28e",
      "selected": true,
      "mask": "data/prepared/IXMtest_H11_s5_w1612B789B-6279-41E1-B4C6-009BF5772334_mask.npy"
    },
    {
      "id": "IXMtest_H16_s4_w16207B133-B3F5-4C39-87A6-D71359B27581",
      "split": "training",
      "image": "data/images/images/IXMtest_H16_s4_w16207B133-B3F5-4C39-87A6-D71359B27581.tif",
      "source_mask": "data/masks/masks/IXMtest_H16_s4_w16207B133-B3F5-4C39-87A6-D71359B27581.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 112,
      "max": 3152,
      "instances": 88,
      "image_sha256": "a5d0ae2b4561e6c6a8e243212d775511b06f2752a76becd7ee3502c8e0b8cd87",
      "selected": false
    },
    {
      "id": "IXMtest_H17_s1_w10A751E6C-5D06-4147-AB73-7FFAE0B57CC5",
      "split": "training",
      "image": "data/images/images/IXMtest_H17_s1_w10A751E6C-5D06-4147-AB73-7FFAE0B57CC5.tif",
      "source_mask": "data/masks/masks/IXMtest_H17_s1_w10A751E6C-5D06-4147-AB73-7FFAE0B57CC5.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 1693,
      "instances": 136,
      "image_sha256": "3ab43e93497fcf4e9fa209c8d88fcba90ce325c542d6c83c61b23e3257ac2fea",
      "selected": false
    },
    {
      "id": "IXMtest_H18_s7_w1D78A9181-AF11-4219-9C1E-2DB2684E948E",
      "split": "training",
      "image": "data/images/images/IXMtest_H18_s7_w1D78A9181-AF11-4219-9C1E-2DB2684E948E.tif",
      "source_mask": "data/masks/masks/IXMtest_H18_s7_w1D78A9181-AF11-4219-9C1E-2DB2684E948E.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 2335,
      "instances": 160,
      "image_sha256": "f51dd75323764f869ef4eb0e112852068fa9e5d37013d40eb8243462b4547a6d",
      "selected": false
    },
    {
      "id": "IXMtest_H24_s6_w15034AA9F-051A-4BF0-A62F-ABAE599C4867",
      "split": "training",
      "image": "data/images/images/IXMtest_H24_s6_w15034AA9F-051A-4BF0-A62F-ABAE599C4867.tif",
      "source_mask": "data/masks/masks/IXMtest_H24_s6_w15034AA9F-051A-4BF0-A62F-ABAE599C4867.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 3460,
      "instances": 136,
      "image_sha256": "78619f3f8db166cf5c3abf2972893602fb19c5f8b01d648e86532b311dc0f012",
      "selected": false
    },
    {
      "id": "IXMtest_I15_s5_w11265571E-AA95-4E58-82CF-A75E3F240956",
      "split": "training",
      "image": "data/images/images/IXMtest_I15_s5_w11265571E-AA95-4E58-82CF-A75E3F240956.tif",
      "source_mask": "data/masks/masks/IXMtest_I15_s5_w11265571E-AA95-4E58-82CF-A75E3F240956.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 3024,
      "instances": 185,
      "image_sha256": "c04fe31ef02382f9a93abcc63edec4ae374e1caa662be15b53984a420391e1a2",
      "selected": false
    },
    {
      "id": "IXMtest_I15_s8_w196F01FA4-CE06-44AE-AAA5-ADEC35BAE513",
      "split": "training",
      "image": "data/images/images/IXMtest_I15_s8_w196F01FA4-CE06-44AE-AAA5-ADEC35BAE513.tif",
      "source_mask": "data/masks/masks/IXMtest_I15_s8_w196F01FA4-CE06-44AE-AAA5-ADEC35BAE513.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 1576,
      "instances": 178,
      "image_sha256": "9ecef49eebc26318faca8e3093aa66841ec0ded7e0b94c9263ccbe3cae3827f2",
      "selected": true,
      "mask": "data/prepared/IXMtest_I15_s8_w196F01FA4-CE06-44AE-AAA5-ADEC35BAE513_mask.npy"
    },
    {
      "id": "IXMtest_J02_s5_w1F53DDD35-C0B2-4E39-BA3B-9F66D289AB02",
      "split": "training",
      "image": "data/images/images/IXMtest_J02_s5_w1F53DDD35-C0B2-4E39-BA3B-9F66D289AB02.tif",
      "source_mask": "data/masks/masks/IXMtest_J02_s5_w1F53DDD35-C0B2-4E39-BA3B-9F66D289AB02.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 2531,
      "instances": 155,
      "image_sha256": "0f436c4e26e8718d79977aaed3fb9034bcfe21fc8c7b1ba0695ae26c1ad4685c",
      "selected": false
    },
    {
      "id": "IXMtest_J03_s4_w1472990BF-A442-497B-8A43-8EE065002EEB",
      "split": "training",
      "image": "data/images/images/IXMtest_J03_s4_w1472990BF-A442-497B-8A43-8EE065002EEB.tif",
      "source_mask": "data/masks/masks/IXMtest_J03_s4_w1472990BF-A442-497B-8A43-8EE065002EEB.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 1712,
      "instances": 131,
      "image_sha256": "f4fd461f351f556d174fe9c72bfa7cb8b4b2d74c0b0c7bedb7eecb01cbcfc84b",
      "selected": false
    },
    {
      "id": "IXMtest_J07_s9_w16921AB3C-1E9C-41C5-ADDA-7DE5CBF9CCBB",
      "split": "training",
      "image": "data/images/images/IXMtest_J07_s9_w16921AB3C-1E9C-41C5-ADDA-7DE5CBF9CCBB.tif",
      "source_mask": "data/masks/masks/IXMtest_J07_s9_w16921AB3C-1E9C-41C5-ADDA-7DE5CBF9CCBB.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 129,
      "max": 3727,
      "instances": 194,
      "image_sha256": "221384503a7766713648117e1fd3571557e8f573b329f893734eb72d134eceba",
      "selected": false
    },
    {
      "id": "IXMtest_J11_s2_w1177A125B-DEBE-4CB7-8F26-53A871785BA5",
      "split": "training",
      "image": "data/images/images/IXMtest_J11_s2_w1177A125B-DEBE-4CB7-8F26-53A871785BA5.tif",
      "source_mask": "data/masks/masks/IXMtest_J11_s2_w1177A125B-DEBE-4CB7-8F26-53A871785BA5.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 128,
      "max": 2155,
      "instances": 135,
      "image_sha256": "a0eca2e1b2ae9a1865a0765e3c30ff9a7b67a9d566ac79ff7bca168156218376",
      "selected": false
    },
    {
      "id": "IXMtest_J15_s9_w1857B395F-AE7A-4319-BC26-3A041C0E31E9",
      "split": "training",
      "image": "data/images/images/IXMtest_J15_s9_w1857B395F-AE7A-4319-BC26-3A041C0E31E9.tif",
      "source_mask": "data/masks/masks/IXMtest_J15_s9_w1857B395F-AE7A-4319-BC26-3A041C0E31E9.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 4095,
      "instances": 151,
      "image_sha256": "42c084e69358e35994077b9c6c454185998688517bda4accb3c641966b16e62a",
      "selected": false
    },
    {
      "id": "IXMtest_J20_s5_w15709AA91-01B9-4959-963A-9D1D574D12E4",
      "split": "training",
      "image": "data/images/images/IXMtest_J20_s5_w15709AA91-01B9-4959-963A-9D1D574D12E4.tif",
      "source_mask": "data/masks/masks/IXMtest_J20_s5_w15709AA91-01B9-4959-963A-9D1D574D12E4.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 1876,
      "instances": 143,
      "image_sha256": "b3cc4cbefefec1dcbc1a715b8216777a97a26d5c752db37afb3d25239a5c1eaa",
      "selected": false
    },
    {
      "id": "IXMtest_K02_s3_w1FE99F6CA-8D35-4496-8F52-DBE55EFAB7E6",
      "split": "training",
      "image": "data/images/images/IXMtest_K02_s3_w1FE99F6CA-8D35-4496-8F52-DBE55EFAB7E6.tif",
      "source_mask": "data/masks/masks/IXMtest_K02_s3_w1FE99F6CA-8D35-4496-8F52-DBE55EFAB7E6.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 115,
      "max": 1937,
      "instances": 68,
      "image_sha256": "df4106241b2d9903813a8759b2799f5da77ca80c02b0da6d3477bab415cc6adc",
      "selected": false
    },
    {
      "id": "IXMtest_K05_s7_w1816D5E2A-B9D8-48DA-9FE6-CBC35FE38C8D",
      "split": "training",
      "image": "data/images/images/IXMtest_K05_s7_w1816D5E2A-B9D8-48DA-9FE6-CBC35FE38C8D.tif",
      "source_mask": "data/masks/masks/IXMtest_K05_s7_w1816D5E2A-B9D8-48DA-9FE6-CBC35FE38C8D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 2836,
      "instances": 155,
      "image_sha256": "50f5fd45edf3d9287743299dfc7c4fe6770580a4015ba3c85283311048c4de0c",
      "selected": true,
      "mask": "data/prepared/IXMtest_K05_s7_w1816D5E2A-B9D8-48DA-9FE6-CBC35FE38C8D_mask.npy"
    },
    {
      "id": "IXMtest_K06_s9_w1787DFDA5-E2CE-40DB-9430-CC3766599BCC",
      "split": "training",
      "image": "data/images/images/IXMtest_K06_s9_w1787DFDA5-E2CE-40DB-9430-CC3766599BCC.tif",
      "source_mask": "data/masks/masks/IXMtest_K06_s9_w1787DFDA5-E2CE-40DB-9430-CC3766599BCC.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 3328,
      "instances": 126,
      "image_sha256": "be8d0bbe1c9217d4c711c752c611906545ef39f56b86a2fe244f5393f0129f76",
      "selected": false
    },
    {
      "id": "IXMtest_K11_s4_w139A2D71E-EC27-49CC-BDA0-1118747BDC76",
      "split": "training",
      "image": "data/images/images/IXMtest_K11_s4_w139A2D71E-EC27-49CC-BDA0-1118747BDC76.tif",
      "source_mask": "data/masks/masks/IXMtest_K11_s4_w139A2D71E-EC27-49CC-BDA0-1118747BDC76.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 2985,
      "instances": 104,
      "image_sha256": "d7810d35d1d8975cb9eb2544c295fcf794dd9a16a5a06950fc6ea16f040ede3f",
      "selected": true,
      "mask": "data/prepared/IXMtest_K11_s4_w139A2D71E-EC27-49CC-BDA0-1118747BDC76_mask.npy"
    },
    {
      "id": "IXMtest_K12_s6_w160D86D6B-648B-433E-9776-8A42DF40E5FB",
      "split": "training",
      "image": "data/images/images/IXMtest_K12_s6_w160D86D6B-648B-433E-9776-8A42DF40E5FB.tif",
      "source_mask": "data/masks/masks/IXMtest_K12_s6_w160D86D6B-648B-433E-9776-8A42DF40E5FB.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 128,
      "max": 2508,
      "instances": 137,
      "image_sha256": "e88ce30fa0e7e7d5907659e7788fb6ca69ddf5b55639855c54671fd7ee137944",
      "selected": false
    },
    {
      "id": "IXMtest_K18_s8_w11FC8D3EA-E30D-46B8-A45E-0399327AA8AC",
      "split": "training",
      "image": "data/images/images/IXMtest_K18_s8_w11FC8D3EA-E30D-46B8-A45E-0399327AA8AC.tif",
      "source_mask": "data/masks/masks/IXMtest_K18_s8_w11FC8D3EA-E30D-46B8-A45E-0399327AA8AC.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 1900,
      "instances": 143,
      "image_sha256": "d2159fba3ea488b2ac79ea51f39c1594779ac3b50f0568c998f0793a0c8e496e",
      "selected": false
    },
    {
      "id": "IXMtest_L01_s2_w1E5038251-DBA3-44D0-BC37-E43E2FC8C174",
      "split": "training",
      "image": "data/images/images/IXMtest_L01_s2_w1E5038251-DBA3-44D0-BC37-E43E2FC8C174.tif",
      "source_mask": "data/masks/masks/IXMtest_L01_s2_w1E5038251-DBA3-44D0-BC37-E43E2FC8C174.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 170,
      "instances": 0,
      "image_sha256": "fbfe08ac0acda0a753544c77c2634919f33978aead5d51d3c2e3f45f4382c99f",
      "selected": false
    },
    {
      "id": "IXMtest_L01_s3_w1E7E0D198-5FB4-4E10-A27C-C46463DA9E06",
      "split": "training",
      "image": "data/images/images/IXMtest_L01_s3_w1E7E0D198-5FB4-4E10-A27C-C46463DA9E06.tif",
      "source_mask": "data/masks/masks/IXMtest_L01_s3_w1E7E0D198-5FB4-4E10-A27C-C46463DA9E06.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 4095,
      "instances": 101,
      "image_sha256": "182c4afb0b564785e9e9cb5e72c9c1e758c01e6a7bf40dabe3f07d37c53812d2",
      "selected": false
    },
    {
      "id": "IXMtest_L03_s6_w1BE79472D-5E2C-422F-A16E-7AC0691C0FD8",
      "split": "training",
      "image": "data/images/images/IXMtest_L03_s6_w1BE79472D-5E2C-422F-A16E-7AC0691C0FD8.tif",
      "source_mask": "data/masks/masks/IXMtest_L03_s6_w1BE79472D-5E2C-422F-A16E-7AC0691C0FD8.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 3595,
      "instances": 146,
      "image_sha256": "2273b39d4ba8dde9037b312db00e98f309fea058f6c3cdc5c6854eaa14eb2584",
      "selected": true,
      "mask": "data/prepared/IXMtest_L03_s6_w1BE79472D-5E2C-422F-A16E-7AC0691C0FD8_mask.npy"
    },
    {
      "id": "IXMtest_L06_s4_w14DD3575A-627B-4A71-91F0-3396DE0B33C4",
      "split": "training",
      "image": "data/images/images/IXMtest_L06_s4_w14DD3575A-627B-4A71-91F0-3396DE0B33C4.tif",
      "source_mask": "data/masks/masks/IXMtest_L06_s4_w14DD3575A-627B-4A71-91F0-3396DE0B33C4.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 2921,
      "instances": 82,
      "image_sha256": "5697a648b4f6d1106bb8a8477b9fda7a05699a282c044fc0e2cc0d449165db09",
      "selected": false
    },
    {
      "id": "IXMtest_L13_s5_w14B7DEC0F-735B-428F-B8B4-F4FF32B6ED4A",
      "split": "training",
      "image": "data/images/images/IXMtest_L13_s5_w14B7DEC0F-735B-428F-B8B4-F4FF32B6ED4A.tif",
      "source_mask": "data/masks/masks/IXMtest_L13_s5_w14B7DEC0F-735B-428F-B8B4-F4FF32B6ED4A.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 1733,
      "instances": 131,
      "image_sha256": "041c7ea55e38f9438cde5d7fe6c1b3e4f7dec5ff8b12ff4d95b6656886a3ea6d",
      "selected": false
    },
    {
      "id": "IXMtest_L17_s1_w1DDC627E5-ADF1-441C-A437-D1D91CC0D498",
      "split": "training",
      "image": "data/images/images/IXMtest_L17_s1_w1DDC627E5-ADF1-441C-A437-D1D91CC0D498.tif",
      "source_mask": "data/masks/masks/IXMtest_L17_s1_w1DDC627E5-ADF1-441C-A437-D1D91CC0D498.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 4095,
      "instances": 144,
      "image_sha256": "9d1160e8c59032261c52dcdd7b0e155c43ce036399862181cc4162de5dd6abe8",
      "selected": true,
      "mask": "data/prepared/IXMtest_L17_s1_w1DDC627E5-ADF1-441C-A437-D1D91CC0D498_mask.npy"
    },
    {
      "id": "IXMtest_L21_s5_w122478CD2-80DC-4B4E-9BC8-A6F6239F4103",
      "split": "training",
      "image": "data/images/images/IXMtest_L21_s5_w122478CD2-80DC-4B4E-9BC8-A6F6239F4103.tif",
      "source_mask": "data/masks/masks/IXMtest_L21_s5_w122478CD2-80DC-4B4E-9BC8-A6F6239F4103.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 1619,
      "instances": 73,
      "image_sha256": "4195358c75873d18d9ec193821878f0c90482b74e70270ff0ccc31dabab08c65",
      "selected": false
    },
    {
      "id": "IXMtest_M04_s3_w1A599DF67-1E7F-4A09-84FD-0080767A735C",
      "split": "training",
      "image": "data/images/images/IXMtest_M04_s3_w1A599DF67-1E7F-4A09-84FD-0080767A735C.tif",
      "source_mask": "data/masks/masks/IXMtest_M04_s3_w1A599DF67-1E7F-4A09-84FD-0080767A735C.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 128,
      "max": 4095,
      "instances": 166,
      "image_sha256": "0642abf507476e7555f676f908dfa9826a13ac628b4e4013ad840d3452f3da25",
      "selected": false
    },
    {
      "id": "IXMtest_M12_s5_w16D817EFA-C3C9-45DB-AC15-61BB143DEC62",
      "split": "training",
      "image": "data/images/images/IXMtest_M12_s5_w16D817EFA-C3C9-45DB-AC15-61BB143DEC62.tif",
      "source_mask": "data/masks/masks/IXMtest_M12_s5_w16D817EFA-C3C9-45DB-AC15-61BB143DEC62.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 126,
      "max": 4095,
      "instances": 157,
      "image_sha256": "08fcf84fdc946526b2d1e2072126b7f89cc1ebd0f70b9f965c54e4c26527a587",
      "selected": true,
      "mask": "data/prepared/IXMtest_M12_s5_w16D817EFA-C3C9-45DB-AC15-61BB143DEC62_mask.npy"
    },
    {
      "id": "IXMtest_M15_s8_w113217199-6161-475C-A786-265D0444C84B",
      "split": "training",
      "image": "data/images/images/IXMtest_M15_s8_w113217199-6161-475C-A786-265D0444C84B.tif",
      "source_mask": "data/masks/masks/IXMtest_M15_s8_w113217199-6161-475C-A786-265D0444C84B.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 126,
      "max": 3555,
      "instances": 120,
      "image_sha256": "312fa71552be2600659d9fd9e3b3d465f2a39ec52ce81645c4c1b35c46ca29f9",
      "selected": true,
      "mask": "data/prepared/IXMtest_M15_s8_w113217199-6161-475C-A786-265D0444C84B_mask.npy"
    },
    {
      "id": "IXMtest_N02_s8_w1B14E70A6-CACF-425C-A958-1285482357F3",
      "split": "training",
      "image": "data/images/images/IXMtest_N02_s8_w1B14E70A6-CACF-425C-A958-1285482357F3.tif",
      "source_mask": "data/masks/masks/IXMtest_N02_s8_w1B14E70A6-CACF-425C-A958-1285482357F3.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 3441,
      "instances": 172,
      "image_sha256": "c4b6565eb879d908382bc730791e23619e8a045815b1bdd21b31f35f8a2b068a",
      "selected": false
    },
    {
      "id": "IXMtest_N15_s3_w148785C31-5B78-47BA-8802-FB8F2CEA6BE2",
      "split": "training",
      "image": "data/images/images/IXMtest_N15_s3_w148785C31-5B78-47BA-8802-FB8F2CEA6BE2.tif",
      "source_mask": "data/masks/masks/IXMtest_N15_s3_w148785C31-5B78-47BA-8802-FB8F2CEA6BE2.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 126,
      "max": 4095,
      "instances": 111,
      "image_sha256": "13ca348ba4d30c83936680f9ee7899acd084b868771b7ca039edf710fdde8444",
      "selected": false
    },
    {
      "id": "IXMtest_N15_s8_w16DBEFA45-FE5A-4233-A1B9-89CA8CF81FB8",
      "split": "training",
      "image": "data/images/images/IXMtest_N15_s8_w16DBEFA45-FE5A-4233-A1B9-89CA8CF81FB8.tif",
      "source_mask": "data/masks/masks/IXMtest_N15_s8_w16DBEFA45-FE5A-4233-A1B9-89CA8CF81FB8.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 114,
      "max": 3504,
      "instances": 106,
      "image_sha256": "cba67f627f80977fa85442d7f509aae59eb783b1c8e5882e5f21c89fb5721cf0",
      "selected": false
    },
    {
      "id": "IXMtest_N18_s5_w1A2FA7E05-3145-461D-9111-FAB41A7F6D6F",
      "split": "training",
      "image": "data/images/images/IXMtest_N18_s5_w1A2FA7E05-3145-461D-9111-FAB41A7F6D6F.tif",
      "source_mask": "data/masks/masks/IXMtest_N18_s5_w1A2FA7E05-3145-461D-9111-FAB41A7F6D6F.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 4095,
      "instances": 130,
      "image_sha256": "3e9085463e5a40f237164a63a8822db2cfb056f5fc5ce16a1faebb071d0c5b1e",
      "selected": false
    },
    {
      "id": "IXMtest_N23_s2_w16380EDDA-E5B8-4168-8BE7-F83AD9810680",
      "split": "training",
      "image": "data/images/images/IXMtest_N23_s2_w16380EDDA-E5B8-4168-8BE7-F83AD9810680.tif",
      "source_mask": "data/masks/masks/IXMtest_N23_s2_w16380EDDA-E5B8-4168-8BE7-F83AD9810680.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 1247,
      "instances": 118,
      "image_sha256": "b205b43fe3e8e47607312c229172e8319df07b510e84b74008cfed2f5f86b122",
      "selected": false
    },
    {
      "id": "IXMtest_N24_s7_w1AC0733FE-E0FD-45B2-930B-FEEAB052DB36",
      "split": "training",
      "image": "data/images/images/IXMtest_N24_s7_w1AC0733FE-E0FD-45B2-930B-FEEAB052DB36.tif",
      "source_mask": "data/masks/masks/IXMtest_N24_s7_w1AC0733FE-E0FD-45B2-930B-FEEAB052DB36.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 1511,
      "instances": 60,
      "image_sha256": "cad70ce52eae471f989f2221d24cd7b7c79c0b8daf7b332e54fe57edaa81ef5a",
      "selected": false
    },
    {
      "id": "IXMtest_O07_s2_w148B3D2F0-D4D6-4F1A-88D3-F18574F52153",
      "split": "training",
      "image": "data/images/images/IXMtest_O07_s2_w148B3D2F0-D4D6-4F1A-88D3-F18574F52153.tif",
      "source_mask": "data/masks/masks/IXMtest_O07_s2_w148B3D2F0-D4D6-4F1A-88D3-F18574F52153.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 1866,
      "instances": 94,
      "image_sha256": "71e425f04e5fd001f3c0b34cadfbeaf0baffb0a26ba0228d3257f273bd9760ce",
      "selected": false
    },
    {
      "id": "IXMtest_O07_s6_w17884A281-85EA-4DD8-9D5A-42427988AE36",
      "split": "training",
      "image": "data/images/images/IXMtest_O07_s6_w17884A281-85EA-4DD8-9D5A-42427988AE36.tif",
      "source_mask": "data/masks/masks/IXMtest_O07_s6_w17884A281-85EA-4DD8-9D5A-42427988AE36.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 2953,
      "instances": 92,
      "image_sha256": "c198903feaef39c7ea963fe9536aea18d2ee8ab90adcf5f907b6ac40e87f6e60",
      "selected": false
    },
    {
      "id": "IXMtest_O15_s6_w1A4844AB3-6D66-4B9A-8212-758E061C349F",
      "split": "training",
      "image": "data/images/images/IXMtest_O15_s6_w1A4844AB3-6D66-4B9A-8212-758E061C349F.tif",
      "source_mask": "data/masks/masks/IXMtest_O15_s6_w1A4844AB3-6D66-4B9A-8212-758E061C349F.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 4095,
      "instances": 33,
      "image_sha256": "23b8285e37bcec8a96764719a3ea62f0fafcd96b525f01329002b7cb90efaa08",
      "selected": false
    },
    {
      "id": "IXMtest_O16_s3_w129E61E12-E398-496E-9408-092D2D70E74B",
      "split": "training",
      "image": "data/images/images/IXMtest_O16_s3_w129E61E12-E398-496E-9408-092D2D70E74B.tif",
      "source_mask": "data/masks/masks/IXMtest_O16_s3_w129E61E12-E398-496E-9408-092D2D70E74B.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 1374,
      "instances": 128,
      "image_sha256": "7fa38ac3b3b708df5e0b42886fc4c9ad95c18a0a939abad1f198a137c5db9834",
      "selected": true,
      "mask": "data/prepared/IXMtest_O16_s3_w129E61E12-E398-496E-9408-092D2D70E74B_mask.npy"
    },
    {
      "id": "IXMtest_O18_s1_w1C2EE9CCF-5055-4E56-BB3A-212A63545A90",
      "split": "training",
      "image": "data/images/images/IXMtest_O18_s1_w1C2EE9CCF-5055-4E56-BB3A-212A63545A90.tif",
      "source_mask": "data/masks/masks/IXMtest_O18_s1_w1C2EE9CCF-5055-4E56-BB3A-212A63545A90.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 3290,
      "instances": 130,
      "image_sha256": "6c2004c1ecf54041833bfe92404f747541624f73ac5f6bd50b0e7accff1ba2f8",
      "selected": false
    },
    {
      "id": "IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A",
      "split": "training",
      "image": "data/images/images/IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A.tif",
      "source_mask": "data/masks/masks/IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 3490,
      "instances": 122,
      "image_sha256": "c6ed48d778e7df5677ef9481f26d81c81ba20ffda807f9df2291cc7c8d278ffb",
      "selected": true,
      "mask": "data/prepared/IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A_mask.npy"
    },
    {
      "id": "IXMtest_O24_s5_w175443699-7C4C-4E5F-9C5F-57A342CC87DE",
      "split": "training",
      "image": "data/images/images/IXMtest_O24_s5_w175443699-7C4C-4E5F-9C5F-57A342CC87DE.tif",
      "source_mask": "data/masks/masks/IXMtest_O24_s5_w175443699-7C4C-4E5F-9C5F-57A342CC87DE.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 128,
      "max": 3771,
      "instances": 119,
      "image_sha256": "9fd46cb1a58508f58c15ef20d2a9417d2c89877311de0582fd18156e40192ee6",
      "selected": false
    },
    {
      "id": "IXMtest_P05_s5_w18FC141F8-8BCA-4851-9000-31D080922BDD",
      "split": "training",
      "image": "data/images/images/IXMtest_P05_s5_w18FC141F8-8BCA-4851-9000-31D080922BDD.tif",
      "source_mask": "data/masks/masks/IXMtest_P05_s5_w18FC141F8-8BCA-4851-9000-31D080922BDD.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 1668,
      "instances": 160,
      "image_sha256": "3a7ca22b14f76d3db0cb7b32b9eaea188c1595f790d5bc8545e19d202db6c8d5",
      "selected": false
    },
    {
      "id": "IXMtest_P09_s4_w11E382363-8C26-4A73-9D8A-8193D66E5785",
      "split": "training",
      "image": "data/images/images/IXMtest_P09_s4_w11E382363-8C26-4A73-9D8A-8193D66E5785.tif",
      "source_mask": "data/masks/masks/IXMtest_P09_s4_w11E382363-8C26-4A73-9D8A-8193D66E5785.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 114,
      "max": 3709,
      "instances": 17,
      "image_sha256": "0df845d706ea4f0fb9ab8cb148017f75a333369c249da96b1911b5281f2896ec",
      "selected": true,
      "mask": "data/prepared/IXMtest_P09_s4_w11E382363-8C26-4A73-9D8A-8193D66E5785_mask.npy"
    },
    {
      "id": "IXMtest_P10_s2_w130B9EBAE-801C-4AAC-A3CE-1C96EE2E6FF9",
      "split": "training",
      "image": "data/images/images/IXMtest_P10_s2_w130B9EBAE-801C-4AAC-A3CE-1C96EE2E6FF9.tif",
      "source_mask": "data/masks/masks/IXMtest_P10_s2_w130B9EBAE-801C-4AAC-A3CE-1C96EE2E6FF9.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 115,
      "max": 2202,
      "instances": 82,
      "image_sha256": "6c3b78729b0fa7ca38fabca48da788edb9d995a3a0b33448ad56eae695fe99c1",
      "selected": false
    },
    {
      "id": "IXMtest_P10_s7_w1F78192AF-7D11-4D41-80F3-8CD6DB05AB57",
      "split": "training",
      "image": "data/images/images/IXMtest_P10_s7_w1F78192AF-7D11-4D41-80F3-8CD6DB05AB57.tif",
      "source_mask": "data/masks/masks/IXMtest_P10_s7_w1F78192AF-7D11-4D41-80F3-8CD6DB05AB57.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 4095,
      "instances": 138,
      "image_sha256": "7088a3570627d321804a6b2d868f3e87da5bfd1cf31fac702348bf001a167ab3",
      "selected": false
    },
    {
      "id": "IXMtest_P13_s6_w1270AEE06-D31F-421C-8147-13644F4C5B94",
      "split": "training",
      "image": "data/images/images/IXMtest_P13_s6_w1270AEE06-D31F-421C-8147-13644F4C5B94.tif",
      "source_mask": "data/masks/masks/IXMtest_P13_s6_w1270AEE06-D31F-421C-8147-13644F4C5B94.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 1312,
      "instances": 92,
      "image_sha256": "d9071939690f818c355e7491dc96ded09915c4ec34f66e4941e28e9856d0a386",
      "selected": false
    },
    {
      "id": "IXMtest_P23_s7_w13B627CB9-6C57-4049-AAD7-6468A051DD24",
      "split": "training",
      "image": "data/images/images/IXMtest_P23_s7_w13B627CB9-6C57-4049-AAD7-6468A051DD24.tif",
      "source_mask": "data/masks/masks/IXMtest_P23_s7_w13B627CB9-6C57-4049-AAD7-6468A051DD24.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 3841,
      "instances": 12,
      "image_sha256": "74b21a3855f7039a6599a740370fab0c4367f3eaf1ba7f85ba219234ac477459",
      "selected": false
    },
    {
      "id": "IXMtest_P23_s9_w1CA355675-EEC7-40B6-82C8-DC7E52A7A825",
      "split": "training",
      "image": "data/images/images/IXMtest_P23_s9_w1CA355675-EEC7-40B6-82C8-DC7E52A7A825.tif",
      "source_mask": "data/masks/masks/IXMtest_P23_s9_w1CA355675-EEC7-40B6-82C8-DC7E52A7A825.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 113,
      "max": 2115,
      "instances": 140,
      "image_sha256": "15839e3214ee8dc2bceffea7812bbd08a9f50c4e2879cd9ab34842c6c81bc241",
      "selected": true,
      "mask": "data/prepared/IXMtest_P23_s9_w1CA355675-EEC7-40B6-82C8-DC7E52A7A825_mask.npy"
    },
    {
      "id": "IXMtest_P24_s9_w13AC6C03C-E8D7-4A23-B649-514BB4052F52",
      "split": "training",
      "image": "data/images/images/IXMtest_P24_s9_w13AC6C03C-E8D7-4A23-B649-514BB4052F52.tif",
      "source_mask": "data/masks/masks/IXMtest_P24_s9_w13AC6C03C-E8D7-4A23-B649-514BB4052F52.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 1853,
      "instances": 89,
      "image_sha256": "cc06216bade34d55db98dc5729ea3d5c503cd1bedff0ba1c34f177717377f375",
      "selected": false
    },
    {
      "id": "IXMtest_A02_s1_w1051DAA7C-7042-435F-99F0-1E847D9B42CB",
      "split": "validation",
      "image": "data/images/images/IXMtest_A02_s1_w1051DAA7C-7042-435F-99F0-1E847D9B42CB.tif",
      "source_mask": "data/masks/masks/IXMtest_A02_s1_w1051DAA7C-7042-435F-99F0-1E847D9B42CB.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 4095,
      "instances": 110,
      "image_sha256": "9eacee0e5a4c29af70307af4d310320632e31ed84b4d8b5fd4fce42930f780ec",
      "selected": false
    },
    {
      "id": "IXMtest_B04_s4_w1F6AEFA0F-AF87-4B3B-A334-698647CFE043",
      "split": "validation",
      "image": "data/images/images/IXMtest_B04_s4_w1F6AEFA0F-AF87-4B3B-A334-698647CFE043.tif",
      "source_mask": "data/masks/masks/IXMtest_B04_s4_w1F6AEFA0F-AF87-4B3B-A334-698647CFE043.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 2921,
      "instances": 104,
      "image_sha256": "050eb9075826c493e83496821a9d6fdcb552661fb97ca1881342c635fcbf370e",
      "selected": false
    },
    {
      "id": "IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78",
      "split": "validation",
      "image": "data/images/images/IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78.tif",
      "source_mask": "data/masks/masks/IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 3271,
      "instances": 120,
      "image_sha256": "d7888558da0167835ee5fd0455f16316f131ec2056a37927ce240ba71ce91d59",
      "selected": true,
      "mask": "data/prepared/IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78_mask.npy"
    },
    {
      "id": "IXMtest_B17_s7_w1215A0A98-4A76-4846-B54A-F7C1EAF84E02",
      "split": "validation",
      "image": "data/images/images/IXMtest_B17_s7_w1215A0A98-4A76-4846-B54A-F7C1EAF84E02.tif",
      "source_mask": "data/masks/masks/IXMtest_B17_s7_w1215A0A98-4A76-4846-B54A-F7C1EAF84E02.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 3428,
      "instances": 106,
      "image_sha256": "a5f3fadb772aed2c0fef49c9bd02f748d250b96476b4aafb7d97044911ea8863",
      "selected": false
    },
    {
      "id": "IXMtest_B19_s7_w1E43B84DB-39E2-4BFB-8CB4-554B32098C75",
      "split": "validation",
      "image": "data/images/images/IXMtest_B19_s7_w1E43B84DB-39E2-4BFB-8CB4-554B32098C75.tif",
      "source_mask": "data/masks/masks/IXMtest_B19_s7_w1E43B84DB-39E2-4BFB-8CB4-554B32098C75.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 2019,
      "instances": 93,
      "image_sha256": "c983a40730eceae5668a088ae6e93cf1e5daabd2ef43483c14575ed54d0d5b58",
      "selected": true,
      "mask": "data/prepared/IXMtest_B19_s7_w1E43B84DB-39E2-4BFB-8CB4-554B32098C75_mask.npy"
    },
    {
      "id": "IXMtest_B24_s9_w18C4FE0DD-12CA-4711-9722-3E3105D1E691",
      "split": "validation",
      "image": "data/images/images/IXMtest_B24_s9_w18C4FE0DD-12CA-4711-9722-3E3105D1E691.tif",
      "source_mask": "data/masks/masks/IXMtest_B24_s9_w18C4FE0DD-12CA-4711-9722-3E3105D1E691.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 2758,
      "instances": 43,
      "image_sha256": "09576736335f5dff3de7be6de40162ce36d63d30a88c3bc0d312304e4b2a20b6",
      "selected": false
    },
    {
      "id": "IXMtest_C18_s1_w11C16FC59-2E29-496A-803A-89581FDF538A",
      "split": "validation",
      "image": "data/images/images/IXMtest_C18_s1_w11C16FC59-2E29-496A-803A-89581FDF538A.tif",
      "source_mask": "data/masks/masks/IXMtest_C18_s1_w11C16FC59-2E29-496A-803A-89581FDF538A.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 1827,
      "instances": 142,
      "image_sha256": "08217bfcf100d86492bce0aac3af11ec376ed4b6090c87b58c56285d8227f56b",
      "selected": false
    },
    {
      "id": "IXMtest_D02_s8_w1AC6783DF-ED35-4818-8091-E6D02AF4BFBD",
      "split": "validation",
      "image": "data/images/images/IXMtest_D02_s8_w1AC6783DF-ED35-4818-8091-E6D02AF4BFBD.tif",
      "source_mask": "data/masks/masks/IXMtest_D02_s8_w1AC6783DF-ED35-4818-8091-E6D02AF4BFBD.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 2522,
      "instances": 113,
      "image_sha256": "93939f6ac7685ff69ba3b8cf08b268f1bb4f263e657cbe98894a30c9fc47b5aa",
      "selected": false
    },
    {
      "id": "IXMtest_D07_s4_w16CF58D03-0B05-41FE-AE73-7298887DEBB1",
      "split": "validation",
      "image": "data/images/images/IXMtest_D07_s4_w16CF58D03-0B05-41FE-AE73-7298887DEBB1.tif",
      "source_mask": "data/masks/masks/IXMtest_D07_s4_w16CF58D03-0B05-41FE-AE73-7298887DEBB1.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 1939,
      "instances": 131,
      "image_sha256": "050e1dd986cdc6b20ff32085334b4e662aea509ae1e248097b31df14aa751e9f",
      "selected": false
    },
    {
      "id": "IXMtest_D19_s6_w1EB1F11AE-4FB6-481F-94D9-40246870F0CB",
      "split": "validation",
      "image": "data/images/images/IXMtest_D19_s6_w1EB1F11AE-4FB6-481F-94D9-40246870F0CB.tif",
      "source_mask": "data/masks/masks/IXMtest_D19_s6_w1EB1F11AE-4FB6-481F-94D9-40246870F0CB.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 3475,
      "instances": 152,
      "image_sha256": "63168d8d7c9b9551866c5846fc32d3f53597d03211a2c3b8c207c61782e3d596",
      "selected": true,
      "mask": "data/prepared/IXMtest_D19_s6_w1EB1F11AE-4FB6-481F-94D9-40246870F0CB_mask.npy"
    },
    {
      "id": "IXMtest_E06_s3_w1701573EB-CE9A-4D76-8668-0416996E1DCD",
      "split": "validation",
      "image": "data/images/images/IXMtest_E06_s3_w1701573EB-CE9A-4D76-8668-0416996E1DCD.tif",
      "source_mask": "data/masks/masks/IXMtest_E06_s3_w1701573EB-CE9A-4D76-8668-0416996E1DCD.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 3008,
      "instances": 56,
      "image_sha256": "13a1850f8034683184c0e07000a314350b084d415925bd65537366346cb0f86d",
      "selected": false
    },
    {
      "id": "IXMtest_E12_s9_w1A811DEC0-ADD9-411A-B5D5-A654C70F253D",
      "split": "validation",
      "image": "data/images/images/IXMtest_E12_s9_w1A811DEC0-ADD9-411A-B5D5-A654C70F253D.tif",
      "source_mask": "data/masks/masks/IXMtest_E12_s9_w1A811DEC0-ADD9-411A-B5D5-A654C70F253D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 1635,
      "instances": 128,
      "image_sha256": "d2b35ecc6958aab89b08db6816890176b06a4357a191798137f7585374cb06de",
      "selected": false
    },
    {
      "id": "IXMtest_F08_s1_w144C3056F-C4DD-4D39-A40F-4F4576A6DBD8",
      "split": "validation",
      "image": "data/images/images/IXMtest_F08_s1_w144C3056F-C4DD-4D39-A40F-4F4576A6DBD8.tif",
      "source_mask": "data/masks/masks/IXMtest_F08_s1_w144C3056F-C4DD-4D39-A40F-4F4576A6DBD8.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 2536,
      "instances": 199,
      "image_sha256": "12c8e3d0e446b3680b49594b5f5dc5d45adf0777c49b21259aac1df16d8da22d",
      "selected": false
    },
    {
      "id": "IXMtest_F12_s5_w17F3E9DFC-6705-40A9-B5FE-C60261D73052",
      "split": "validation",
      "image": "data/images/images/IXMtest_F12_s5_w17F3E9DFC-6705-40A9-B5FE-C60261D73052.tif",
      "source_mask": "data/masks/masks/IXMtest_F12_s5_w17F3E9DFC-6705-40A9-B5FE-C60261D73052.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 4095,
      "instances": 138,
      "image_sha256": "0ffa34379cf2c3edb21feed483ac7ec45c26e62c6b8a5585e17656ddd801152c",
      "selected": true,
      "mask": "data/prepared/IXMtest_F12_s5_w17F3E9DFC-6705-40A9-B5FE-C60261D73052_mask.npy"
    },
    {
      "id": "IXMtest_F22_s6_w1F4C7ADE4-B68D-4D30-A063-722B87AA2DA1",
      "split": "validation",
      "image": "data/images/images/IXMtest_F22_s6_w1F4C7ADE4-B68D-4D30-A063-722B87AA2DA1.tif",
      "source_mask": "data/masks/masks/IXMtest_F22_s6_w1F4C7ADE4-B68D-4D30-A063-722B87AA2DA1.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 4095,
      "instances": 152,
      "image_sha256": "73fd4a27b9d45424c75154db4240d8338e6a446a7c1b488bad20b8a11294d160",
      "selected": false
    },
    {
      "id": "IXMtest_G10_s3_w1C1257E17-1DBA-4619-B06E-D6DBB8A53088",
      "split": "validation",
      "image": "data/images/images/IXMtest_G10_s3_w1C1257E17-1DBA-4619-B06E-D6DBB8A53088.tif",
      "source_mask": "data/masks/masks/IXMtest_G10_s3_w1C1257E17-1DBA-4619-B06E-D6DBB8A53088.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 112,
      "max": 2052,
      "instances": 140,
      "image_sha256": "2443042238953c07a92ad2b0f7ce24e4039ae6cf688177c518e8536408f16979",
      "selected": false
    },
    {
      "id": "IXMtest_G12_s6_w16850371E-A405-4D73-9816-F5F68F885D38",
      "split": "validation",
      "image": "data/images/images/IXMtest_G12_s6_w16850371E-A405-4D73-9816-F5F68F885D38.tif",
      "source_mask": "data/masks/masks/IXMtest_G12_s6_w16850371E-A405-4D73-9816-F5F68F885D38.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 3384,
      "instances": 156,
      "image_sha256": "9cc56b55ad06165bdbdf7e1c4b25e4e7cc1328c0aa8423ddc25f6555c3bb11bb",
      "selected": false
    },
    {
      "id": "IXMtest_G13_s9_w19606195E-12B4-46FF-9B83-1F7FE11B3AFB",
      "split": "validation",
      "image": "data/images/images/IXMtest_G13_s9_w19606195E-12B4-46FF-9B83-1F7FE11B3AFB.tif",
      "source_mask": "data/masks/masks/IXMtest_G13_s9_w19606195E-12B4-46FF-9B83-1F7FE11B3AFB.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 124,
      "max": 3491,
      "instances": 73,
      "image_sha256": "a9b65ce7310dfa949dbc2068eecfb9862833fe9613577a75f9c5caba9322aaf4",
      "selected": false
    },
    {
      "id": "IXMtest_G18_s9_w17F495AF2-03D2-4683-BD37-56887F4A3A84",
      "split": "validation",
      "image": "data/images/images/IXMtest_G18_s9_w17F495AF2-03D2-4683-BD37-56887F4A3A84.tif",
      "source_mask": "data/masks/masks/IXMtest_G18_s9_w17F495AF2-03D2-4683-BD37-56887F4A3A84.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 3120,
      "instances": 104,
      "image_sha256": "18f8643fd303ff61f59517a4444f7b63b148b07419a39f758acf4fc787ee8ff3",
      "selected": false
    },
    {
      "id": "IXMtest_G22_s3_w157F2847B-C953-410E-8F60-956A6023AED4",
      "split": "validation",
      "image": "data/images/images/IXMtest_G22_s3_w157F2847B-C953-410E-8F60-956A6023AED4.tif",
      "source_mask": "data/masks/masks/IXMtest_G22_s3_w157F2847B-C953-410E-8F60-956A6023AED4.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 128,
      "max": 4095,
      "instances": 146,
      "image_sha256": "a461edd3f4a08f8a84f9f1867c7dfbcd53e619d2a3cd7705f0557dd2e7efd05f",
      "selected": false
    },
    {
      "id": "IXMtest_G22_s5_w1CE8AEFCD-7739-4D60-B112-9D2D73EE05E5",
      "split": "validation",
      "image": "data/images/images/IXMtest_G22_s5_w1CE8AEFCD-7739-4D60-B112-9D2D73EE05E5.tif",
      "source_mask": "data/masks/masks/IXMtest_G22_s5_w1CE8AEFCD-7739-4D60-B112-9D2D73EE05E5.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 1969,
      "instances": 138,
      "image_sha256": "8b4f667ab3dafe143c1d8efebb9f422fc1be61df90a30408bec9900c0f9da23e",
      "selected": false
    },
    {
      "id": "IXMtest_H11_s6_w19DF4E879-8DE4-45D6-840B-305BDDB27076",
      "split": "validation",
      "image": "data/images/images/IXMtest_H11_s6_w19DF4E879-8DE4-45D6-840B-305BDDB27076.tif",
      "source_mask": "data/masks/masks/IXMtest_H11_s6_w19DF4E879-8DE4-45D6-840B-305BDDB27076.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 126,
      "max": 2671,
      "instances": 127,
      "image_sha256": "5b90aeb50afde2d834a6e5a3f007ec232102fd0f26f833170a7d193fd3876190",
      "selected": false
    },
    {
      "id": "IXMtest_I01_s4_w1218CC565-C87E-4390-936A-4D3E51BC10DB",
      "split": "validation",
      "image": "data/images/images/IXMtest_I01_s4_w1218CC565-C87E-4390-936A-4D3E51BC10DB.tif",
      "source_mask": "data/masks/masks/IXMtest_I01_s4_w1218CC565-C87E-4390-936A-4D3E51BC10DB.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 1611,
      "instances": 69,
      "image_sha256": "486f9c2e22e297b75a9bd14a806a583fa063fa550606d7b2f16b0312e9fe790c",
      "selected": false
    },
    {
      "id": "IXMtest_I04_s9_w16A5CC270-8B92-42EE-AA4A-855776F7D46B",
      "split": "validation",
      "image": "data/images/images/IXMtest_I04_s9_w16A5CC270-8B92-42EE-AA4A-855776F7D46B.tif",
      "source_mask": "data/masks/masks/IXMtest_I04_s9_w16A5CC270-8B92-42EE-AA4A-855776F7D46B.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 3734,
      "instances": 163,
      "image_sha256": "1c3c92a12924dba60e947692d3eddc236148ba2f6610c4f09d9a7f3799e2fe7b",
      "selected": false
    },
    {
      "id": "IXMtest_I07_s4_w1F156255A-3842-46FB-ABF2-9D041E523F86",
      "split": "validation",
      "image": "data/images/images/IXMtest_I07_s4_w1F156255A-3842-46FB-ABF2-9D041E523F86.tif",
      "source_mask": "data/masks/masks/IXMtest_I07_s4_w1F156255A-3842-46FB-ABF2-9D041E523F86.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 115,
      "max": 1820,
      "instances": 118,
      "image_sha256": "0001556fe12ae4b5a6c24edcd522c4033f7d375ba457c105e4d62d05ffbca078",
      "selected": true,
      "mask": "data/prepared/IXMtest_I07_s4_w1F156255A-3842-46FB-ABF2-9D041E523F86_mask.npy"
    },
    {
      "id": "IXMtest_I08_s2_w11996D679-5D76-4FB8-A681-2014A8999EC8",
      "split": "validation",
      "image": "data/images/images/IXMtest_I08_s2_w11996D679-5D76-4FB8-A681-2014A8999EC8.tif",
      "source_mask": "data/masks/masks/IXMtest_I08_s2_w11996D679-5D76-4FB8-A681-2014A8999EC8.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 3847,
      "instances": 105,
      "image_sha256": "d73070d6ccdd07dfa6f333ceb856a3e137e64ac6664bf6529c5ca3fdcf9df146",
      "selected": false
    },
    {
      "id": "IXMtest_I11_s6_w1B2DC04C7-2D7D-45C6-9DC2-66D8605FBE63",
      "split": "validation",
      "image": "data/images/images/IXMtest_I11_s6_w1B2DC04C7-2D7D-45C6-9DC2-66D8605FBE63.tif",
      "source_mask": "data/masks/masks/IXMtest_I11_s6_w1B2DC04C7-2D7D-45C6-9DC2-66D8605FBE63.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 1678,
      "instances": 70,
      "image_sha256": "919ebf76ecdf10999dd8194c2507e8218b14e308050e014ec132938e3190163a",
      "selected": false
    },
    {
      "id": "IXMtest_I16_s9_w1582FD22A-5270-4FDC-868F-5F75808E2321",
      "split": "validation",
      "image": "data/images/images/IXMtest_I16_s9_w1582FD22A-5270-4FDC-868F-5F75808E2321.tif",
      "source_mask": "data/masks/masks/IXMtest_I16_s9_w1582FD22A-5270-4FDC-868F-5F75808E2321.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 126,
      "max": 1883,
      "instances": 123,
      "image_sha256": "63eb63b8baa8830ad778165746606ca71f8a43d27da42520cb586705b3b12e66",
      "selected": false
    },
    {
      "id": "IXMtest_I18_s3_w1544C8B7A-E092-4F9D-B8D3-C489638D770F",
      "split": "validation",
      "image": "data/images/images/IXMtest_I18_s3_w1544C8B7A-E092-4F9D-B8D3-C489638D770F.tif",
      "source_mask": "data/masks/masks/IXMtest_I18_s3_w1544C8B7A-E092-4F9D-B8D3-C489638D770F.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 1481,
      "instances": 160,
      "image_sha256": "b4c7879051b7044b1528186396c7b929adee3f53d1b810b1ebe9951534ca955e",
      "selected": false
    },
    {
      "id": "IXMtest_J02_s8_w1D9C198F9-ECF0-4EF7-848D-AC7782CD3C28",
      "split": "validation",
      "image": "data/images/images/IXMtest_J02_s8_w1D9C198F9-ECF0-4EF7-848D-AC7782CD3C28.tif",
      "source_mask": "data/masks/masks/IXMtest_J02_s8_w1D9C198F9-ECF0-4EF7-848D-AC7782CD3C28.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 115,
      "max": 1169,
      "instances": 108,
      "image_sha256": "da8efba4b9147e82055ddaf823973c47e3e6904452321e65783e74e40c64af83",
      "selected": false
    },
    {
      "id": "IXMtest_J08_s2_w1C146DB1C-05B3-49EF-9C62-1185FD9897AC",
      "split": "validation",
      "image": "data/images/images/IXMtest_J08_s2_w1C146DB1C-05B3-49EF-9C62-1185FD9897AC.tif",
      "source_mask": "data/masks/masks/IXMtest_J08_s2_w1C146DB1C-05B3-49EF-9C62-1185FD9897AC.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 129,
      "max": 4095,
      "instances": 175,
      "image_sha256": "aee45f7af7fb7ae7402cb98e1e91411e60860c18254511e25bb500d95a7a3de6",
      "selected": false
    },
    {
      "id": "IXMtest_J20_s1_w1EEE65E52-7AD8-47C7-A286-6E84C5D77953",
      "split": "validation",
      "image": "data/images/images/IXMtest_J20_s1_w1EEE65E52-7AD8-47C7-A286-6E84C5D77953.tif",
      "source_mask": "data/masks/masks/IXMtest_J20_s1_w1EEE65E52-7AD8-47C7-A286-6E84C5D77953.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 124,
      "max": 1972,
      "instances": 142,
      "image_sha256": "9ea71ec947d38b393adb87ce22dba89b45783cfa8d80bcf21b2329cf6bbc106f",
      "selected": false
    },
    {
      "id": "IXMtest_K12_s1_w193D6C057-1AA9-4E2F-86EA-2E71961BE68B",
      "split": "validation",
      "image": "data/images/images/IXMtest_K12_s1_w193D6C057-1AA9-4E2F-86EA-2E71961BE68B.tif",
      "source_mask": "data/masks/masks/IXMtest_K12_s1_w193D6C057-1AA9-4E2F-86EA-2E71961BE68B.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 2331,
      "instances": 86,
      "image_sha256": "1a179d3815932ba4b693c267b09dda50deccb1c176ee45e1bd7e55f7d77ce78b",
      "selected": false
    },
    {
      "id": "IXMtest_K12_s7_w12A7857A5-3C92-4A08-8E81-2CA8A99F67AE",
      "split": "validation",
      "image": "data/images/images/IXMtest_K12_s7_w12A7857A5-3C92-4A08-8E81-2CA8A99F67AE.tif",
      "source_mask": "data/masks/masks/IXMtest_K12_s7_w12A7857A5-3C92-4A08-8E81-2CA8A99F67AE.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 3303,
      "instances": 231,
      "image_sha256": "59203d2e3f7c5c99ffb6794af165d5d4f3dfc056933822c087b89e89f078f595",
      "selected": false
    },
    {
      "id": "IXMtest_L03_s2_w1AC4550E2-F824-4A58-9CC5-952AD9ECE76A",
      "split": "validation",
      "image": "data/images/images/IXMtest_L03_s2_w1AC4550E2-F824-4A58-9CC5-952AD9ECE76A.tif",
      "source_mask": "data/masks/masks/IXMtest_L03_s2_w1AC4550E2-F824-4A58-9CC5-952AD9ECE76A.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 2149,
      "instances": 119,
      "image_sha256": "dc9e102b6c0718aa10fb22dbb353ad151bb66d1bec22483a0d8db0fda15c3ec6",
      "selected": false
    },
    {
      "id": "IXMtest_L05_s2_w1B9C6FAC9-9D48-4184-8D9B-ABFC3BEC1125",
      "split": "validation",
      "image": "data/images/images/IXMtest_L05_s2_w1B9C6FAC9-9D48-4184-8D9B-ABFC3BEC1125.tif",
      "source_mask": "data/masks/masks/IXMtest_L05_s2_w1B9C6FAC9-9D48-4184-8D9B-ABFC3BEC1125.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 2226,
      "instances": 148,
      "image_sha256": "01d1aa68f5bb1decebec548d739d220e617e540218f6807444b6efb8c0f4a645",
      "selected": false
    },
    {
      "id": "IXMtest_L10_s6_w12D12D64C-2639-4CA8-9BB4-99F92C9B7068",
      "split": "validation",
      "image": "data/images/images/IXMtest_L10_s6_w12D12D64C-2639-4CA8-9BB4-99F92C9B7068.tif",
      "source_mask": "data/masks/masks/IXMtest_L10_s6_w12D12D64C-2639-4CA8-9BB4-99F92C9B7068.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 185,
      "instances": 0,
      "image_sha256": "9cfed38c3581400464c6d29d73880b70e3fffdcd921ba6923b9788d2d0f232bf",
      "selected": false
    },
    {
      "id": "IXMtest_L11_s4_w13C057BB5-9CFB-471F-84B5-72F80654CF81",
      "split": "validation",
      "image": "data/images/images/IXMtest_L11_s4_w13C057BB5-9CFB-471F-84B5-72F80654CF81.tif",
      "source_mask": "data/masks/masks/IXMtest_L11_s4_w13C057BB5-9CFB-471F-84B5-72F80654CF81.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 4095,
      "instances": 169,
      "image_sha256": "a950895ed1d5f9e94ea75b9b71f5ed9bb936878f78c583dd9dde4f2454aa9323",
      "selected": false
    },
    {
      "id": "IXMtest_L14_s5_w14B42C89E-7650-44AC-9D7B-50BE61EA307E",
      "split": "validation",
      "image": "data/images/images/IXMtest_L14_s5_w14B42C89E-7650-44AC-9D7B-50BE61EA307E.tif",
      "source_mask": "data/masks/masks/IXMtest_L14_s5_w14B42C89E-7650-44AC-9D7B-50BE61EA307E.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 2865,
      "instances": 160,
      "image_sha256": "bc4b16c17f8b2438dc78704daf6c51d8d6a274f258d2717e2e0680d3bf58fb91",
      "selected": false
    },
    {
      "id": "IXMtest_M20_s3_w15C73A7C7-F81B-4583-AB8F-0A64336AF070",
      "split": "validation",
      "image": "data/images/images/IXMtest_M20_s3_w15C73A7C7-F81B-4583-AB8F-0A64336AF070.tif",
      "source_mask": "data/masks/masks/IXMtest_M20_s3_w15C73A7C7-F81B-4583-AB8F-0A64336AF070.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 1129,
      "instances": 63,
      "image_sha256": "4ca8afb63077139fe9ae5d61dacb7a1f69734e441e9f7791da0059ed0f3e2e16",
      "selected": false
    },
    {
      "id": "IXMtest_M23_s8_w118BC311D-A998-4161-8256-22839B2421F2",
      "split": "validation",
      "image": "data/images/images/IXMtest_M23_s8_w118BC311D-A998-4161-8256-22839B2421F2.tif",
      "source_mask": "data/masks/masks/IXMtest_M23_s8_w118BC311D-A998-4161-8256-22839B2421F2.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 4095,
      "instances": 162,
      "image_sha256": "98833468663e3b473ee718fb837c2d73069dc5c1afdcc565ea7c498d5c08e00c",
      "selected": true,
      "mask": "data/prepared/IXMtest_M23_s8_w118BC311D-A998-4161-8256-22839B2421F2_mask.npy"
    },
    {
      "id": "IXMtest_N12_s7_w166EF3FAB-EA33-4B28-91E3-034A1654BAAE",
      "split": "validation",
      "image": "data/images/images/IXMtest_N12_s7_w166EF3FAB-EA33-4B28-91E3-034A1654BAAE.tif",
      "source_mask": "data/masks/masks/IXMtest_N12_s7_w166EF3FAB-EA33-4B28-91E3-034A1654BAAE.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 2766,
      "instances": 202,
      "image_sha256": "d0073eed82bfb885c193b131118a5609d026fbe46068ffbed346ac3cce4549cf",
      "selected": true,
      "mask": "data/prepared/IXMtest_N12_s7_w166EF3FAB-EA33-4B28-91E3-034A1654BAAE_mask.npy"
    },
    {
      "id": "IXMtest_N18_s2_w1CC5ED51D-86C5-437D-8EDD-E56E4C949B3B",
      "split": "validation",
      "image": "data/images/images/IXMtest_N18_s2_w1CC5ED51D-86C5-437D-8EDD-E56E4C949B3B.tif",
      "source_mask": "data/masks/masks/IXMtest_N18_s2_w1CC5ED51D-86C5-437D-8EDD-E56E4C949B3B.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 146,
      "max": 3073,
      "instances": 128,
      "image_sha256": "a9cd80fd5f704b7bda8ccf1986314eab42bf0c9b23a0c5efcecb7a1297b4073a",
      "selected": false
    },
    {
      "id": "IXMtest_O09_s2_w133C7EDCE-1C7C-41A6-9E52-7AD499E7CDC8",
      "split": "validation",
      "image": "data/images/images/IXMtest_O09_s2_w133C7EDCE-1C7C-41A6-9E52-7AD499E7CDC8.tif",
      "source_mask": "data/masks/masks/IXMtest_O09_s2_w133C7EDCE-1C7C-41A6-9E52-7AD499E7CDC8.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 126,
      "max": 1575,
      "instances": 79,
      "image_sha256": "f32e0594abe3d8c2ce2b8e987bba67ddf736163b3c89a9fb9e8c578cd934108f",
      "selected": false
    },
    {
      "id": "IXMtest_O10_s8_w18F4DB020-BFB7-4F13-B99C-C39F8E54F85D",
      "split": "validation",
      "image": "data/images/images/IXMtest_O10_s8_w18F4DB020-BFB7-4F13-B99C-C39F8E54F85D.tif",
      "source_mask": "data/masks/masks/IXMtest_O10_s8_w18F4DB020-BFB7-4F13-B99C-C39F8E54F85D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 4095,
      "instances": 20,
      "image_sha256": "8a6a917ed6b9aabccdaf42de755d0e5a4b9eba3e64157f06a15c7208cd0f7b06",
      "selected": false
    },
    {
      "id": "IXMtest_O13_s3_w12D9C1C9C-C582-4080-B9BE-4807FA3E0843",
      "split": "validation",
      "image": "data/images/images/IXMtest_O13_s3_w12D9C1C9C-C582-4080-B9BE-4807FA3E0843.tif",
      "source_mask": "data/masks/masks/IXMtest_O13_s3_w12D9C1C9C-C582-4080-B9BE-4807FA3E0843.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 1908,
      "instances": 24,
      "image_sha256": "a19f70dc3b4b65491199c2af35286f171d34a7b51cc2d4081a60f604dfabb39f",
      "selected": false
    },
    {
      "id": "IXMtest_O16_s2_w1F6F6A3A1-99E4-4029-B734-022806CF6D42",
      "split": "validation",
      "image": "data/images/images/IXMtest_O16_s2_w1F6F6A3A1-99E4-4029-B734-022806CF6D42.tif",
      "source_mask": "data/masks/masks/IXMtest_O16_s2_w1F6F6A3A1-99E4-4029-B734-022806CF6D42.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 128,
      "max": 4095,
      "instances": 128,
      "image_sha256": "321ed5a5e44cd614849f0c1d3e2fd517008cb457e629db5e97eaf90dafb78b40",
      "selected": false
    },
    {
      "id": "IXMtest_P01_s3_w1A7DC2612-9C11-4656-B100-102AF8FE8B43",
      "split": "validation",
      "image": "data/images/images/IXMtest_P01_s3_w1A7DC2612-9C11-4656-B100-102AF8FE8B43.tif",
      "source_mask": "data/masks/masks/IXMtest_P01_s3_w1A7DC2612-9C11-4656-B100-102AF8FE8B43.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 128,
      "max": 3449,
      "instances": 65,
      "image_sha256": "204cda9a3df39092632014da5ea7024230987487517b2470a1934f847a0a2c16",
      "selected": false
    },
    {
      "id": "IXMtest_P15_s3_w10F5E9699-743C-4177-93CE-27CFD65A925E",
      "split": "validation",
      "image": "data/images/images/IXMtest_P15_s3_w10F5E9699-743C-4177-93CE-27CFD65A925E.tif",
      "source_mask": "data/masks/masks/IXMtest_P15_s3_w10F5E9699-743C-4177-93CE-27CFD65A925E.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 2884,
      "instances": 105,
      "image_sha256": "bce6322539875bad32697c6a69cad6728f410fbbe904378bf51ad4e289655f6c",
      "selected": true,
      "mask": "data/prepared/IXMtest_P15_s3_w10F5E9699-743C-4177-93CE-27CFD65A925E_mask.npy"
    },
    {
      "id": "IXMtest_P21_s5_w1ACEBEE91-BAFA-49E6-9D97-D07197400A15",
      "split": "validation",
      "image": "data/images/images/IXMtest_P21_s5_w1ACEBEE91-BAFA-49E6-9D97-D07197400A15.tif",
      "source_mask": "data/masks/masks/IXMtest_P21_s5_w1ACEBEE91-BAFA-49E6-9D97-D07197400A15.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 112,
      "max": 2472,
      "instances": 103,
      "image_sha256": "80631b15a48e9d32591e2725090cff3e3ad211281a9d96ed40d82630dfde6a62",
      "selected": false
    },
    {
      "id": "IXMtest_A09_s1_w1CE70AD49-290D-4312-82E6-CDC717F32637",
      "split": "test",
      "image": "data/images/images/IXMtest_A09_s1_w1CE70AD49-290D-4312-82E6-CDC717F32637.tif",
      "source_mask": "data/masks/masks/IXMtest_A09_s1_w1CE70AD49-290D-4312-82E6-CDC717F32637.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 1720,
      "instances": 156,
      "image_sha256": "51787b533f38efc0d934a9cc0f760e704ef5fca97d561aa868f2099f2e6bb1ce",
      "selected": false
    },
    {
      "id": "IXMtest_A12_s7_w1EAEEA614-51ED-43B3-A4FF-088730911E4C",
      "split": "test",
      "image": "data/images/images/IXMtest_A12_s7_w1EAEEA614-51ED-43B3-A4FF-088730911E4C.tif",
      "source_mask": "data/masks/masks/IXMtest_A12_s7_w1EAEEA614-51ED-43B3-A4FF-088730911E4C.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 112,
      "max": 3885,
      "instances": 69,
      "image_sha256": "d49d90f12cde7176191b0acbf2ed783c39fe8bdf0caf968ca5f8a1627429b8f1",
      "selected": true,
      "mask": "data/prepared/IXMtest_A12_s7_w1EAEEA614-51ED-43B3-A4FF-088730911E4C_mask.npy"
    },
    {
      "id": "IXMtest_A16_s2_w15AF20A10-82AE-48FA-AC50-7AE8AC3AA544",
      "split": "test",
      "image": "data/images/images/IXMtest_A16_s2_w15AF20A10-82AE-48FA-AC50-7AE8AC3AA544.tif",
      "source_mask": "data/masks/masks/IXMtest_A16_s2_w15AF20A10-82AE-48FA-AC50-7AE8AC3AA544.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 1357,
      "instances": 93,
      "image_sha256": "69329412d32aff2e6ce272576062f54d43453d4126313bae405389091c79843a",
      "selected": false
    },
    {
      "id": "IXMtest_A22_s8_w1E2AFE190-831D-4D9C-961E-3AA2ECB3599D",
      "split": "test",
      "image": "data/images/images/IXMtest_A22_s8_w1E2AFE190-831D-4D9C-961E-3AA2ECB3599D.tif",
      "source_mask": "data/masks/masks/IXMtest_A22_s8_w1E2AFE190-831D-4D9C-961E-3AA2ECB3599D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 3617,
      "instances": 95,
      "image_sha256": "fbc992000fe626926d56ece5f1731af9284add61d1b8096b3d99e943a47435e5",
      "selected": false
    },
    {
      "id": "IXMtest_B02_s9_w124B5080D-EBE1-47D2-B147-C0F342039EDF",
      "split": "test",
      "image": "data/images/images/IXMtest_B02_s9_w124B5080D-EBE1-47D2-B147-C0F342039EDF.tif",
      "source_mask": "data/masks/masks/IXMtest_B02_s9_w124B5080D-EBE1-47D2-B147-C0F342039EDF.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 2083,
      "instances": 66,
      "image_sha256": "f6c668b78aa2f19af2ed936b758e024d00ffc1a2db091ee04d8f6c6ced342003",
      "selected": false
    },
    {
      "id": "IXMtest_B04_s2_w17C6C7F8D-98F7-422B-92CD-EA61EE813325",
      "split": "test",
      "image": "data/images/images/IXMtest_B04_s2_w17C6C7F8D-98F7-422B-92CD-EA61EE813325.tif",
      "source_mask": "data/masks/masks/IXMtest_B04_s2_w17C6C7F8D-98F7-422B-92CD-EA61EE813325.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 1584,
      "instances": 122,
      "image_sha256": "a28f5fe95e775e88271527f6fe1c5562f9e94db2da93ac4cf0dc7a09c086fbd2",
      "selected": false
    },
    {
      "id": "IXMtest_B05_s5_w10296BEF4-0F48-4EC8-BC24-6C604871398B",
      "split": "test",
      "image": "data/images/images/IXMtest_B05_s5_w10296BEF4-0F48-4EC8-BC24-6C604871398B.tif",
      "source_mask": "data/masks/masks/IXMtest_B05_s5_w10296BEF4-0F48-4EC8-BC24-6C604871398B.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 4095,
      "instances": 115,
      "image_sha256": "9789dd00d103bbfc9356d1f6503a9c1210b12abaefa6d6050fb3fd74f6338cd6",
      "selected": false
    },
    {
      "id": "IXMtest_C07_s5_w12C9F2926-A017-4962-8660-72C9C20C86E4",
      "split": "test",
      "image": "data/images/images/IXMtest_C07_s5_w12C9F2926-A017-4962-8660-72C9C20C86E4.tif",
      "source_mask": "data/masks/masks/IXMtest_C07_s5_w12C9F2926-A017-4962-8660-72C9C20C86E4.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 1273,
      "instances": 156,
      "image_sha256": "3ebabf72d4a0d737cdf345f291cb67c5ce8fcdb7ccaaa52c874d8abdc1413143",
      "selected": true,
      "mask": "data/prepared/IXMtest_C07_s5_w12C9F2926-A017-4962-8660-72C9C20C86E4_mask.npy"
    },
    {
      "id": "IXMtest_C14_s8_w1612E3D43-FA7E-4FFB-BE53-F34A3A0CAFCD",
      "split": "test",
      "image": "data/images/images/IXMtest_C14_s8_w1612E3D43-FA7E-4FFB-BE53-F34A3A0CAFCD.tif",
      "source_mask": "data/masks/masks/IXMtest_C14_s8_w1612E3D43-FA7E-4FFB-BE53-F34A3A0CAFCD.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 3818,
      "instances": 137,
      "image_sha256": "09363558d0fd0aa848e28f79dfc2b130f2bcc67208acab9741193afe76811b83",
      "selected": false
    },
    {
      "id": "IXMtest_C23_s2_w10533E965-3DC3-4556-9C26-EA8DD4F8DE61",
      "split": "test",
      "image": "data/images/images/IXMtest_C23_s2_w10533E965-3DC3-4556-9C26-EA8DD4F8DE61.tif",
      "source_mask": "data/masks/masks/IXMtest_C23_s2_w10533E965-3DC3-4556-9C26-EA8DD4F8DE61.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 3567,
      "instances": 119,
      "image_sha256": "033b59cb26728242bfec49319c15a3784a825b4d5c5d5b5ecf60b760ecad9924",
      "selected": false
    },
    {
      "id": "IXMtest_D01_s1_w181AE787C-B5EE-4150-A90E-2FE43165C32E",
      "split": "test",
      "image": "data/images/images/IXMtest_D01_s1_w181AE787C-B5EE-4150-A90E-2FE43165C32E.tif",
      "source_mask": "data/masks/masks/IXMtest_D01_s1_w181AE787C-B5EE-4150-A90E-2FE43165C32E.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 127,
      "max": 1966,
      "instances": 100,
      "image_sha256": "62f8532495bcae6829667866b376d9fc484039278427b67781ce9e441073a5d5",
      "selected": false
    },
    {
      "id": "IXMtest_D06_s8_w1BE84C8EF-4CD0-4B56-8267-4E662F57AC25",
      "split": "test",
      "image": "data/images/images/IXMtest_D06_s8_w1BE84C8EF-4CD0-4B56-8267-4E662F57AC25.tif",
      "source_mask": "data/masks/masks/IXMtest_D06_s8_w1BE84C8EF-4CD0-4B56-8267-4E662F57AC25.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 2717,
      "instances": 167,
      "image_sha256": "02a0f641f54f4dec9d0cac45a16f531721a2916ed74cdba7f2678e968e88f3d8",
      "selected": false
    },
    {
      "id": "IXMtest_D08_s7_w1F476A544-A07C-4E9B-A6CB-BAEF4CD6F64E",
      "split": "test",
      "image": "data/images/images/IXMtest_D08_s7_w1F476A544-A07C-4E9B-A6CB-BAEF4CD6F64E.tif",
      "source_mask": "data/masks/masks/IXMtest_D08_s7_w1F476A544-A07C-4E9B-A6CB-BAEF4CD6F64E.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 1803,
      "instances": 49,
      "image_sha256": "6d403228631aa2ffdc3307dc35c6bd268903be699af19656790010b717d28794",
      "selected": false
    },
    {
      "id": "IXMtest_D10_s2_w1BB72A093-72AB-476C-9513-2FF43BFB2134",
      "split": "test",
      "image": "data/images/images/IXMtest_D10_s2_w1BB72A093-72AB-476C-9513-2FF43BFB2134.tif",
      "source_mask": "data/masks/masks/IXMtest_D10_s2_w1BB72A093-72AB-476C-9513-2FF43BFB2134.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 2110,
      "instances": 141,
      "image_sha256": "5e5a9ce5cb911d222bffcd9c82ed129a561d1513a8eea0473d7e83c0d356774e",
      "selected": false
    },
    {
      "id": "IXMtest_D20_s3_w19D371AF3-0189-48A3-AF3B-D108DE6A017F",
      "split": "test",
      "image": "data/images/images/IXMtest_D20_s3_w19D371AF3-0189-48A3-AF3B-D108DE6A017F.tif",
      "source_mask": "data/masks/masks/IXMtest_D20_s3_w19D371AF3-0189-48A3-AF3B-D108DE6A017F.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 133,
      "max": 1918,
      "instances": 169,
      "image_sha256": "9a5a09ec913e10abeecf8cdfa3f467c6f414505553375d0db887bb5fea80fc72",
      "selected": true,
      "mask": "data/prepared/IXMtest_D20_s3_w19D371AF3-0189-48A3-AF3B-D108DE6A017F_mask.npy"
    },
    {
      "id": "IXMtest_E07_s1_w1641C6847-142B-463F-8B08-5B3296615572",
      "split": "test",
      "image": "data/images/images/IXMtest_E07_s1_w1641C6847-142B-463F-8B08-5B3296615572.tif",
      "source_mask": "data/masks/masks/IXMtest_E07_s1_w1641C6847-142B-463F-8B08-5B3296615572.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 129,
      "max": 4095,
      "instances": 170,
      "image_sha256": "2de13faf836648622bee872a0106047221dd6b2eceb529fcfc1a60a9e16f3451",
      "selected": true,
      "mask": "data/prepared/IXMtest_E07_s1_w1641C6847-142B-463F-8B08-5B3296615572_mask.npy"
    },
    {
      "id": "IXMtest_F03_s5_w1FE22EC7D-720A-4848-888A-6AFE1E1C85B0",
      "split": "test",
      "image": "data/images/images/IXMtest_F03_s5_w1FE22EC7D-720A-4848-888A-6AFE1E1C85B0.tif",
      "source_mask": "data/masks/masks/IXMtest_F03_s5_w1FE22EC7D-720A-4848-888A-6AFE1E1C85B0.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 1414,
      "instances": 152,
      "image_sha256": "a4674bae1ed2bb2eef5266bab4af90aac50158fa46dd49999bfa5cff57b25b27",
      "selected": false
    },
    {
      "id": "IXMtest_F04_s5_w1D94DA1A2-873C-44B3-80EB-36DC2A97E9A3",
      "split": "test",
      "image": "data/images/images/IXMtest_F04_s5_w1D94DA1A2-873C-44B3-80EB-36DC2A97E9A3.tif",
      "source_mask": "data/masks/masks/IXMtest_F04_s5_w1D94DA1A2-873C-44B3-80EB-36DC2A97E9A3.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 3673,
      "instances": 116,
      "image_sha256": "1c87ab7ba25d93df33136d7bba9a49d316f6920f8ca96ed4429c66f8a8c1b8b4",
      "selected": false
    },
    {
      "id": "IXMtest_F08_s7_w117D78658-1BA7-495D-9EAA-E4FA155861AC",
      "split": "test",
      "image": "data/images/images/IXMtest_F08_s7_w117D78658-1BA7-495D-9EAA-E4FA155861AC.tif",
      "source_mask": "data/masks/masks/IXMtest_F08_s7_w117D78658-1BA7-495D-9EAA-E4FA155861AC.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 111,
      "max": 1607,
      "instances": 65,
      "image_sha256": "68bff7687a8dc6d1ed720e50452d59140ac2700e6ed3c7197cd73012510e47d0",
      "selected": false
    },
    {
      "id": "IXMtest_F12_s8_w1DBD80811-5297-4415-ACD1-EC9286BE76A4",
      "split": "test",
      "image": "data/images/images/IXMtest_F12_s8_w1DBD80811-5297-4415-ACD1-EC9286BE76A4.tif",
      "source_mask": "data/masks/masks/IXMtest_F12_s8_w1DBD80811-5297-4415-ACD1-EC9286BE76A4.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 115,
      "max": 2148,
      "instances": 7,
      "image_sha256": "5c34b879cb7103c334cfc9de3f8ff21d128d09504771de8f8cf8b0250b9a46d4",
      "selected": true,
      "mask": "data/prepared/IXMtest_F12_s8_w1DBD80811-5297-4415-ACD1-EC9286BE76A4_mask.npy"
    },
    {
      "id": "IXMtest_F14_s7_w1E67F2A8A-EEC1-4F01-A1E2-0ECFC639637F",
      "split": "test",
      "image": "data/images/images/IXMtest_F14_s7_w1E67F2A8A-EEC1-4F01-A1E2-0ECFC639637F.tif",
      "source_mask": "data/masks/masks/IXMtest_F14_s7_w1E67F2A8A-EEC1-4F01-A1E2-0ECFC639637F.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 4095,
      "instances": 118,
      "image_sha256": "7f03b873fe9039afec4f4ce2cbff9cbcd2bc7855d9bf8ea0149ef59153c023c0",
      "selected": false
    },
    {
      "id": "IXMtest_G01_s2_w1FBE52723-8BDF-4346-89BB-216A4A69ED1C",
      "split": "test",
      "image": "data/images/images/IXMtest_G01_s2_w1FBE52723-8BDF-4346-89BB-216A4A69ED1C.tif",
      "source_mask": "data/masks/masks/IXMtest_G01_s2_w1FBE52723-8BDF-4346-89BB-216A4A69ED1C.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 116,
      "max": 1364,
      "instances": 92,
      "image_sha256": "206e9a06614db3a70ddfaafa412436cc92ce6e148e2d2f07e6179594a435e62f",
      "selected": true,
      "mask": "data/prepared/IXMtest_G01_s2_w1FBE52723-8BDF-4346-89BB-216A4A69ED1C_mask.npy"
    },
    {
      "id": "IXMtest_G06_s7_w19444140C-EF22-42DE-863A-514D836BE850",
      "split": "test",
      "image": "data/images/images/IXMtest_G06_s7_w19444140C-EF22-42DE-863A-514D836BE850.tif",
      "source_mask": "data/masks/masks/IXMtest_G06_s7_w19444140C-EF22-42DE-863A-514D836BE850.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 124,
      "max": 2940,
      "instances": 126,
      "image_sha256": "3824cb0bd3a56359e8832b2bb9f4a7128af94d5b3ee9e35d564c0f83a5c46ceb",
      "selected": false
    },
    {
      "id": "IXMtest_I03_s4_w1C724E24D-7AD5-4166-A810-DC75A825495D",
      "split": "test",
      "image": "data/images/images/IXMtest_I03_s4_w1C724E24D-7AD5-4166-A810-DC75A825495D.tif",
      "source_mask": "data/masks/masks/IXMtest_I03_s4_w1C724E24D-7AD5-4166-A810-DC75A825495D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 117,
      "max": 1422,
      "instances": 130,
      "image_sha256": "90f02786a40ac470f05e5876165544ce2d710f324b7eb0f6cca616dd75dcc46e",
      "selected": false
    },
    {
      "id": "IXMtest_I12_s1_w18A06F07F-CD8F-4056-B041-3967BE48AD78",
      "split": "test",
      "image": "data/images/images/IXMtest_I12_s1_w18A06F07F-CD8F-4056-B041-3967BE48AD78.tif",
      "source_mask": "data/masks/masks/IXMtest_I12_s1_w18A06F07F-CD8F-4056-B041-3967BE48AD78.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 126,
      "max": 3856,
      "instances": 202,
      "image_sha256": "cc9232820fb734b61ffd37dc1c5d3f867f5ef19723950707f466b7e58940df30",
      "selected": true,
      "mask": "data/prepared/IXMtest_I12_s1_w18A06F07F-CD8F-4056-B041-3967BE48AD78_mask.npy"
    },
    {
      "id": "IXMtest_I17_s6_w1EB7CA00F-DF95-466C-BF8C-6304B6A4974E",
      "split": "test",
      "image": "data/images/images/IXMtest_I17_s6_w1EB7CA00F-DF95-466C-BF8C-6304B6A4974E.tif",
      "source_mask": "data/masks/masks/IXMtest_I17_s6_w1EB7CA00F-DF95-466C-BF8C-6304B6A4974E.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 3448,
      "instances": 154,
      "image_sha256": "f11d4a8a586126d11128fd5b5f991399787ca7850d47c1df3947c764c65165de",
      "selected": true,
      "mask": "data/prepared/IXMtest_I17_s6_w1EB7CA00F-DF95-466C-BF8C-6304B6A4974E_mask.npy"
    },
    {
      "id": "IXMtest_I17_s7_w1BB3D534E-DF08-4F21-BAC6-6B105BB40AF9",
      "split": "test",
      "image": "data/images/images/IXMtest_I17_s7_w1BB3D534E-DF08-4F21-BAC6-6B105BB40AF9.tif",
      "source_mask": "data/masks/masks/IXMtest_I17_s7_w1BB3D534E-DF08-4F21-BAC6-6B105BB40AF9.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 1526,
      "instances": 131,
      "image_sha256": "f9675e76849ce6a71e0aeb68df5bba314922637c5e92a054570583246a81a146",
      "selected": false
    },
    {
      "id": "IXMtest_I23_s5_w1E3053D6B-8CEF-48E3-A6A5-2F0D7C1AA177",
      "split": "test",
      "image": "data/images/images/IXMtest_I23_s5_w1E3053D6B-8CEF-48E3-A6A5-2F0D7C1AA177.tif",
      "source_mask": "data/masks/masks/IXMtest_I23_s5_w1E3053D6B-8CEF-48E3-A6A5-2F0D7C1AA177.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 1641,
      "instances": 36,
      "image_sha256": "c314b3466af5e61c25c45778a84d8b861bbcd6ed0fd9f2f497e7510366b2a2e0",
      "selected": false
    },
    {
      "id": "IXMtest_J16_s8_w1AD7BC3EB-6A29-4A53-AA59-E39653839B0D",
      "split": "test",
      "image": "data/images/images/IXMtest_J16_s8_w1AD7BC3EB-6A29-4A53-AA59-E39653839B0D.tif",
      "source_mask": "data/masks/masks/IXMtest_J16_s8_w1AD7BC3EB-6A29-4A53-AA59-E39653839B0D.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 124,
      "max": 2940,
      "instances": 145,
      "image_sha256": "ae80797cae04102d7ba232d414a77b606287cab6fe2c4f8769c32fbf7497080f",
      "selected": false
    },
    {
      "id": "IXMtest_J17_s1_w19E7BCD2F-6D7E-4B39-9154-F93E4D539183",
      "split": "test",
      "image": "data/images/images/IXMtest_J17_s1_w19E7BCD2F-6D7E-4B39-9154-F93E4D539183.tif",
      "source_mask": "data/masks/masks/IXMtest_J17_s1_w19E7BCD2F-6D7E-4B39-9154-F93E4D539183.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 1626,
      "instances": 82,
      "image_sha256": "48773bf2733767c4d50ec853bb12e92641215b5810a1a38c60e8272e4e7a05d2",
      "selected": true,
      "mask": "data/prepared/IXMtest_J17_s1_w19E7BCD2F-6D7E-4B39-9154-F93E4D539183_mask.npy"
    },
    {
      "id": "IXMtest_K01_s3_w1054711D8-52CF-4672-B3A0-EC9EC2E681C4",
      "split": "test",
      "image": "data/images/images/IXMtest_K01_s3_w1054711D8-52CF-4672-B3A0-EC9EC2E681C4.tif",
      "source_mask": "data/masks/masks/IXMtest_K01_s3_w1054711D8-52CF-4672-B3A0-EC9EC2E681C4.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 1578,
      "instances": 102,
      "image_sha256": "7780fe8000c3a4898621bbb5fdd8d8dbfcb2f550e92453a6af177c2a30ab64a9",
      "selected": false
    },
    {
      "id": "IXMtest_K01_s5_w1A3DE001A-72D6-4321-8B25-4300AB0207AC",
      "split": "test",
      "image": "data/images/images/IXMtest_K01_s5_w1A3DE001A-72D6-4321-8B25-4300AB0207AC.tif",
      "source_mask": "data/masks/masks/IXMtest_K01_s5_w1A3DE001A-72D6-4321-8B25-4300AB0207AC.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 113,
      "max": 3150,
      "instances": 119,
      "image_sha256": "0361885ce4eccfd71b7834433f7982e40fb313af95dc5ae05e8144757469c040",
      "selected": false
    },
    {
      "id": "IXMtest_K03_s5_w1DC4CE558-042C-482E-8CAE-FCCB57AA9A55",
      "split": "test",
      "image": "data/images/images/IXMtest_K03_s5_w1DC4CE558-042C-482E-8CAE-FCCB57AA9A55.tif",
      "source_mask": "data/masks/masks/IXMtest_K03_s5_w1DC4CE558-042C-482E-8CAE-FCCB57AA9A55.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 125,
      "max": 2375,
      "instances": 139,
      "image_sha256": "43fa2c513feb0812c341c120b8860be14c6b28b7d68a5dd98c95c35ad627fd43",
      "selected": true,
      "mask": "data/prepared/IXMtest_K03_s5_w1DC4CE558-042C-482E-8CAE-FCCB57AA9A55_mask.npy"
    },
    {
      "id": "IXMtest_K05_s9_w1647AEF60-D7B0-4D83-AFA5-2B05E968468F",
      "split": "test",
      "image": "data/images/images/IXMtest_K05_s9_w1647AEF60-D7B0-4D83-AFA5-2B05E968468F.tif",
      "source_mask": "data/masks/masks/IXMtest_K05_s9_w1647AEF60-D7B0-4D83-AFA5-2B05E968468F.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 121,
      "max": 3089,
      "instances": 142,
      "image_sha256": "0d93fbe37721d75105462100d1aceee5ae89285f95a0faf8049bf462ca8ff381",
      "selected": false
    },
    {
      "id": "IXMtest_K17_s5_w1BB3AD9D7-3A85-458C-AB1C-07A816C2C203",
      "split": "test",
      "image": "data/images/images/IXMtest_K17_s5_w1BB3AD9D7-3A85-458C-AB1C-07A816C2C203.tif",
      "source_mask": "data/masks/masks/IXMtest_K17_s5_w1BB3AD9D7-3A85-458C-AB1C-07A816C2C203.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 3112,
      "instances": 149,
      "image_sha256": "42ed7955651a252795b5d0bc78a69ef6c2949383966a8d30fbfbf43f1de10fb8",
      "selected": true,
      "mask": "data/prepared/IXMtest_K17_s5_w1BB3AD9D7-3A85-458C-AB1C-07A816C2C203_mask.npy"
    },
    {
      "id": "IXMtest_L06_s5_w1610AE208-D6E8-4DC4-AC19-D26284AA8556",
      "split": "test",
      "image": "data/images/images/IXMtest_L06_s5_w1610AE208-D6E8-4DC4-AC19-D26284AA8556.tif",
      "source_mask": "data/masks/masks/IXMtest_L06_s5_w1610AE208-D6E8-4DC4-AC19-D26284AA8556.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 3009,
      "instances": 198,
      "image_sha256": "6b17fb5e036d1f622578bdd87fad26a64f14634399e7919fbddd868b152eaf49",
      "selected": false
    },
    {
      "id": "IXMtest_M12_s7_w193E9BA0F-274A-4EE8-A0F0-6A58471A333A",
      "split": "test",
      "image": "data/images/images/IXMtest_M12_s7_w193E9BA0F-274A-4EE8-A0F0-6A58471A333A.tif",
      "source_mask": "data/masks/masks/IXMtest_M12_s7_w193E9BA0F-274A-4EE8-A0F0-6A58471A333A.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 2410,
      "instances": 197,
      "image_sha256": "301b13eaf968e56bce7065c21f9319d91719847b5873ae27ae398e2ba4d968a3",
      "selected": false
    },
    {
      "id": "IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2",
      "split": "test",
      "image": "data/images/images/IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2.tif",
      "source_mask": "data/masks/masks/IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 120,
      "max": 1305,
      "instances": 152,
      "image_sha256": "72aa16357727f76578b5c44a5167caf5284045c3e2fd4d75648363240153d160",
      "selected": true,
      "mask": "data/prepared/IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2_mask.npy"
    },
    {
      "id": "IXMtest_N08_s2_w159DEEDD5-2882-4F9E-9048-F5BD2B918651",
      "split": "test",
      "image": "data/images/images/IXMtest_N08_s2_w159DEEDD5-2882-4F9E-9048-F5BD2B918651.tif",
      "source_mask": "data/masks/masks/IXMtest_N08_s2_w159DEEDD5-2882-4F9E-9048-F5BD2B918651.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 1773,
      "instances": 127,
      "image_sha256": "0d943128297bc37671c31a2624e2a8ec60ccdc235541c7f9cd0ccd8775538af8",
      "selected": true,
      "mask": "data/prepared/IXMtest_N08_s2_w159DEEDD5-2882-4F9E-9048-F5BD2B918651_mask.npy"
    },
    {
      "id": "IXMtest_N11_s4_w142A84EA3-47C3-4B49-B6CA-BBC6685BBE1E",
      "split": "test",
      "image": "data/images/images/IXMtest_N11_s4_w142A84EA3-47C3-4B49-B6CA-BBC6685BBE1E.tif",
      "source_mask": "data/masks/masks/IXMtest_N11_s4_w142A84EA3-47C3-4B49-B6CA-BBC6685BBE1E.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 115,
      "max": 3197,
      "instances": 93,
      "image_sha256": "2b2262a733fbe76f73e31bea2a36656a01daddaf47abc0ed49e167219d9de963",
      "selected": false
    },
    {
      "id": "IXMtest_N12_s9_w1C98EC3CC-2F8E-4761-A21A-EBF14469AD40",
      "split": "test",
      "image": "data/images/images/IXMtest_N12_s9_w1C98EC3CC-2F8E-4761-A21A-EBF14469AD40.tif",
      "source_mask": "data/masks/masks/IXMtest_N12_s9_w1C98EC3CC-2F8E-4761-A21A-EBF14469AD40.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 3188,
      "instances": 155,
      "image_sha256": "014c672520887cff1b1ba62612daee4c9301d56bdbc64698d562b5e6c87113f4",
      "selected": false
    },
    {
      "id": "IXMtest_N21_s1_w14F264B20-62D6-47EC-9BCD-B9416AB302D9",
      "split": "test",
      "image": "data/images/images/IXMtest_N21_s1_w14F264B20-62D6-47EC-9BCD-B9416AB302D9.tif",
      "source_mask": "data/masks/masks/IXMtest_N21_s1_w14F264B20-62D6-47EC-9BCD-B9416AB302D9.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 110,
      "max": 2854,
      "instances": 26,
      "image_sha256": "e58bc57a1e790f095e0da44d17eaf7e8eeeff1dc14c6d7bb87d630684e8d1bf3",
      "selected": false
    },
    {
      "id": "IXMtest_O01_s6_w11A23978B-BAAD-4287-B1F6-FFBCF45C5E2F",
      "split": "test",
      "image": "data/images/images/IXMtest_O01_s6_w11A23978B-BAAD-4287-B1F6-FFBCF45C5E2F.tif",
      "source_mask": "data/masks/masks/IXMtest_O01_s6_w11A23978B-BAAD-4287-B1F6-FFBCF45C5E2F.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 1314,
      "instances": 82,
      "image_sha256": "1b4d4ef302cb7af542f9319ae95fcfbf8cd6b43e8835bcc93de600d72c2a00d0",
      "selected": false
    },
    {
      "id": "IXMtest_O02_s4_w1814603DA-DD56-44EF-98C5-AFF3B636F986",
      "split": "test",
      "image": "data/images/images/IXMtest_O02_s4_w1814603DA-DD56-44EF-98C5-AFF3B636F986.tif",
      "source_mask": "data/masks/masks/IXMtest_O02_s4_w1814603DA-DD56-44EF-98C5-AFF3B636F986.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 1901,
      "instances": 74,
      "image_sha256": "79e75ca5dc49dff1583bc675ba5ccced6cf33535832a9555aa856b829278291d",
      "selected": false
    },
    {
      "id": "IXMtest_O04_s2_w19A18B0D8-8E8B-4572-A5C0-5C678F3AD54C",
      "split": "test",
      "image": "data/images/images/IXMtest_O04_s2_w19A18B0D8-8E8B-4572-A5C0-5C678F3AD54C.tif",
      "source_mask": "data/masks/masks/IXMtest_O04_s2_w19A18B0D8-8E8B-4572-A5C0-5C678F3AD54C.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 123,
      "max": 1517,
      "instances": 89,
      "image_sha256": "aefa533820425d9a1f9e5b94261b8f9c5959e3276aa8cd793ea01e8ae3139db0",
      "selected": false
    },
    {
      "id": "IXMtest_P07_s8_w144364F25-950A-472C-A529-1A9AD0433B6C",
      "split": "test",
      "image": "data/images/images/IXMtest_P07_s8_w144364F25-950A-472C-A529-1A9AD0433B6C.tif",
      "source_mask": "data/masks/masks/IXMtest_P07_s8_w144364F25-950A-472C-A529-1A9AD0433B6C.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 4095,
      "instances": 113,
      "image_sha256": "0a3a070ed15b242a3b8557d4533fb9fdb030577bd70686668f6deee0695fa10b",
      "selected": true,
      "mask": "data/prepared/IXMtest_P07_s8_w144364F25-950A-472C-A529-1A9AD0433B6C_mask.npy"
    },
    {
      "id": "IXMtest_P07_s8_w19D682C29-5685-4A33-8CAA-F0797DD7F021",
      "split": "test",
      "image": "data/images/images/IXMtest_P07_s8_w19D682C29-5685-4A33-8CAA-F0797DD7F021.tif",
      "source_mask": "data/masks/masks/IXMtest_P07_s8_w19D682C29-5685-4A33-8CAA-F0797DD7F021.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 122,
      "max": 3180,
      "instances": 50,
      "image_sha256": "6d977fceac2c7aedd9502a281d3f59939d051b78d720440d189f61da03f5ab0c",
      "selected": false
    },
    {
      "id": "IXMtest_P15_s8_w1D26A5BC7-CF59-4027-B785-85AD33773609",
      "split": "test",
      "image": "data/images/images/IXMtest_P15_s8_w1D26A5BC7-CF59-4027-B785-85AD33773609.tif",
      "source_mask": "data/masks/masks/IXMtest_P15_s8_w1D26A5BC7-CF59-4027-B785-85AD33773609.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 118,
      "max": 3449,
      "instances": 82,
      "image_sha256": "cbf81c613d35a1918ba7734e795241a93d75f445e70664208a6aee5b3e949e53",
      "selected": true,
      "mask": "data/prepared/IXMtest_P15_s8_w1D26A5BC7-CF59-4027-B785-85AD33773609_mask.npy"
    },
    {
      "id": "IXMtest_P19_s6_w10EBAD15B-2822-48B9-B9C8-4DF5D80AB940",
      "split": "test",
      "image": "data/images/images/IXMtest_P19_s6_w10EBAD15B-2822-48B9-B9C8-4DF5D80AB940.tif",
      "source_mask": "data/masks/masks/IXMtest_P19_s6_w10EBAD15B-2822-48B9-B9C8-4DF5D80AB940.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 119,
      "max": 1633,
      "instances": 89,
      "image_sha256": "c71d72360df1ca6cd67c6fef964787c3e4918c877e2ee98928255bb20c0039e2",
      "selected": true,
      "mask": "data/prepared/IXMtest_P19_s6_w10EBAD15B-2822-48B9-B9C8-4DF5D80AB940_mask.npy"
    },
    {
      "id": "IXMtest_P21_s4_w19B7DC88B-10BB-4B51-9A10-9E30285688C1",
      "split": "test",
      "image": "data/images/images/IXMtest_P21_s4_w19B7DC88B-10BB-4B51-9A10-9E30285688C1.tif",
      "source_mask": "data/masks/masks/IXMtest_P21_s4_w19B7DC88B-10BB-4B51-9A10-9E30285688C1.png",
      "shape": [
        520,
        696
      ],
      "dtype": "uint16",
      "min": 113,
      "max": 1208,
      "instances": 62,
      "image_sha256": "ab4513ed247e94c79cadd2425db46ad6c2fc9434f79ac8837828e94775ca7698",
      "selected": false
    }
  ],
  "selected": {
    "training": [
      {
        "id": "IXMtest_B12_s9_w1ABE4C6B8-6110-4D7D-B782-94C43FDC2E1F",
        "split": "training",
        "image": "data/images/images/IXMtest_B12_s9_w1ABE4C6B8-6110-4D7D-B782-94C43FDC2E1F.tif",
        "source_mask": "data/masks/masks/IXMtest_B12_s9_w1ABE4C6B8-6110-4D7D-B782-94C43FDC2E1F.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 119,
        "max": 3398,
        "instances": 109,
        "image_sha256": "6d9c79e14473510fce59f9634e25eca437dff8893de5143f8d27c3a9ece321fc",
        "selected": true,
        "mask": "data/prepared/IXMtest_B12_s9_w1ABE4C6B8-6110-4D7D-B782-94C43FDC2E1F_mask.npy"
      },
      {
        "id": "IXMtest_B20_s2_w159B9FE71-035A-4DED-B0CA-C76916C968BC",
        "split": "training",
        "image": "data/images/images/IXMtest_B20_s2_w159B9FE71-035A-4DED-B0CA-C76916C968BC.tif",
        "source_mask": "data/masks/masks/IXMtest_B20_s2_w159B9FE71-035A-4DED-B0CA-C76916C968BC.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 117,
        "max": 1986,
        "instances": 88,
        "image_sha256": "cf3e628f13040792d3ac0c6b07103ac9a8084afc2b01063e93867b553cb97d4b",
        "selected": true,
        "mask": "data/prepared/IXMtest_B20_s2_w159B9FE71-035A-4DED-B0CA-C76916C968BC_mask.npy"
      },
      {
        "id": "IXMtest_C19_s4_w1AD3DC23E-B9B5-46E9-9716-26096C672A13",
        "split": "training",
        "image": "data/images/images/IXMtest_C19_s4_w1AD3DC23E-B9B5-46E9-9716-26096C672A13.tif",
        "source_mask": "data/masks/masks/IXMtest_C19_s4_w1AD3DC23E-B9B5-46E9-9716-26096C672A13.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 123,
        "max": 2967,
        "instances": 124,
        "image_sha256": "04b171051f9062520d57d042be72499be9d28aed486a95b664e0f35e089247ed",
        "selected": true,
        "mask": "data/prepared/IXMtest_C19_s4_w1AD3DC23E-B9B5-46E9-9716-26096C672A13_mask.npy"
      },
      {
        "id": "IXMtest_G06_s3_w19ABE30DE-4BFC-4CD5-AD16-6C8BD6E69F9D",
        "split": "training",
        "image": "data/images/images/IXMtest_G06_s3_w19ABE30DE-4BFC-4CD5-AD16-6C8BD6E69F9D.tif",
        "source_mask": "data/masks/masks/IXMtest_G06_s3_w19ABE30DE-4BFC-4CD5-AD16-6C8BD6E69F9D.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 117,
        "max": 3936,
        "instances": 169,
        "image_sha256": "079b69b374ece48120645e4c3b88a532a4be2c8bed16258f8621ecc3ac60bae5",
        "selected": true,
        "mask": "data/prepared/IXMtest_G06_s3_w19ABE30DE-4BFC-4CD5-AD16-6C8BD6E69F9D_mask.npy"
      },
      {
        "id": "IXMtest_H11_s5_w1612B789B-6279-41E1-B4C6-009BF5772334",
        "split": "training",
        "image": "data/images/images/IXMtest_H11_s5_w1612B789B-6279-41E1-B4C6-009BF5772334.tif",
        "source_mask": "data/masks/masks/IXMtest_H11_s5_w1612B789B-6279-41E1-B4C6-009BF5772334.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 124,
        "max": 2171,
        "instances": 153,
        "image_sha256": "b88f1807ec029c5294cebf00230ba533343375e5838b78f8ec8994f93b6cc28e",
        "selected": true,
        "mask": "data/prepared/IXMtest_H11_s5_w1612B789B-6279-41E1-B4C6-009BF5772334_mask.npy"
      },
      {
        "id": "IXMtest_I15_s8_w196F01FA4-CE06-44AE-AAA5-ADEC35BAE513",
        "split": "training",
        "image": "data/images/images/IXMtest_I15_s8_w196F01FA4-CE06-44AE-AAA5-ADEC35BAE513.tif",
        "source_mask": "data/masks/masks/IXMtest_I15_s8_w196F01FA4-CE06-44AE-AAA5-ADEC35BAE513.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 119,
        "max": 1576,
        "instances": 178,
        "image_sha256": "9ecef49eebc26318faca8e3093aa66841ec0ded7e0b94c9263ccbe3cae3827f2",
        "selected": true,
        "mask": "data/prepared/IXMtest_I15_s8_w196F01FA4-CE06-44AE-AAA5-ADEC35BAE513_mask.npy"
      },
      {
        "id": "IXMtest_K05_s7_w1816D5E2A-B9D8-48DA-9FE6-CBC35FE38C8D",
        "split": "training",
        "image": "data/images/images/IXMtest_K05_s7_w1816D5E2A-B9D8-48DA-9FE6-CBC35FE38C8D.tif",
        "source_mask": "data/masks/masks/IXMtest_K05_s7_w1816D5E2A-B9D8-48DA-9FE6-CBC35FE38C8D.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 125,
        "max": 2836,
        "instances": 155,
        "image_sha256": "50f5fd45edf3d9287743299dfc7c4fe6770580a4015ba3c85283311048c4de0c",
        "selected": true,
        "mask": "data/prepared/IXMtest_K05_s7_w1816D5E2A-B9D8-48DA-9FE6-CBC35FE38C8D_mask.npy"
      },
      {
        "id": "IXMtest_K11_s4_w139A2D71E-EC27-49CC-BDA0-1118747BDC76",
        "split": "training",
        "image": "data/images/images/IXMtest_K11_s4_w139A2D71E-EC27-49CC-BDA0-1118747BDC76.tif",
        "source_mask": "data/masks/masks/IXMtest_K11_s4_w139A2D71E-EC27-49CC-BDA0-1118747BDC76.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 119,
        "max": 2985,
        "instances": 104,
        "image_sha256": "d7810d35d1d8975cb9eb2544c295fcf794dd9a16a5a06950fc6ea16f040ede3f",
        "selected": true,
        "mask": "data/prepared/IXMtest_K11_s4_w139A2D71E-EC27-49CC-BDA0-1118747BDC76_mask.npy"
      },
      {
        "id": "IXMtest_L03_s6_w1BE79472D-5E2C-422F-A16E-7AC0691C0FD8",
        "split": "training",
        "image": "data/images/images/IXMtest_L03_s6_w1BE79472D-5E2C-422F-A16E-7AC0691C0FD8.tif",
        "source_mask": "data/masks/masks/IXMtest_L03_s6_w1BE79472D-5E2C-422F-A16E-7AC0691C0FD8.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 121,
        "max": 3595,
        "instances": 146,
        "image_sha256": "2273b39d4ba8dde9037b312db00e98f309fea058f6c3cdc5c6854eaa14eb2584",
        "selected": true,
        "mask": "data/prepared/IXMtest_L03_s6_w1BE79472D-5E2C-422F-A16E-7AC0691C0FD8_mask.npy"
      },
      {
        "id": "IXMtest_L17_s1_w1DDC627E5-ADF1-441C-A437-D1D91CC0D498",
        "split": "training",
        "image": "data/images/images/IXMtest_L17_s1_w1DDC627E5-ADF1-441C-A437-D1D91CC0D498.tif",
        "source_mask": "data/masks/masks/IXMtest_L17_s1_w1DDC627E5-ADF1-441C-A437-D1D91CC0D498.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 120,
        "max": 4095,
        "instances": 144,
        "image_sha256": "9d1160e8c59032261c52dcdd7b0e155c43ce036399862181cc4162de5dd6abe8",
        "selected": true,
        "mask": "data/prepared/IXMtest_L17_s1_w1DDC627E5-ADF1-441C-A437-D1D91CC0D498_mask.npy"
      },
      {
        "id": "IXMtest_M12_s5_w16D817EFA-C3C9-45DB-AC15-61BB143DEC62",
        "split": "training",
        "image": "data/images/images/IXMtest_M12_s5_w16D817EFA-C3C9-45DB-AC15-61BB143DEC62.tif",
        "source_mask": "data/masks/masks/IXMtest_M12_s5_w16D817EFA-C3C9-45DB-AC15-61BB143DEC62.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 126,
        "max": 4095,
        "instances": 157,
        "image_sha256": "08fcf84fdc946526b2d1e2072126b7f89cc1ebd0f70b9f965c54e4c26527a587",
        "selected": true,
        "mask": "data/prepared/IXMtest_M12_s5_w16D817EFA-C3C9-45DB-AC15-61BB143DEC62_mask.npy"
      },
      {
        "id": "IXMtest_M15_s8_w113217199-6161-475C-A786-265D0444C84B",
        "split": "training",
        "image": "data/images/images/IXMtest_M15_s8_w113217199-6161-475C-A786-265D0444C84B.tif",
        "source_mask": "data/masks/masks/IXMtest_M15_s8_w113217199-6161-475C-A786-265D0444C84B.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 126,
        "max": 3555,
        "instances": 120,
        "image_sha256": "312fa71552be2600659d9fd9e3b3d465f2a39ec52ce81645c4c1b35c46ca29f9",
        "selected": true,
        "mask": "data/prepared/IXMtest_M15_s8_w113217199-6161-475C-A786-265D0444C84B_mask.npy"
      },
      {
        "id": "IXMtest_O16_s3_w129E61E12-E398-496E-9408-092D2D70E74B",
        "split": "training",
        "image": "data/images/images/IXMtest_O16_s3_w129E61E12-E398-496E-9408-092D2D70E74B.tif",
        "source_mask": "data/masks/masks/IXMtest_O16_s3_w129E61E12-E398-496E-9408-092D2D70E74B.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 116,
        "max": 1374,
        "instances": 128,
        "image_sha256": "7fa38ac3b3b708df5e0b42886fc4c9ad95c18a0a939abad1f198a137c5db9834",
        "selected": true,
        "mask": "data/prepared/IXMtest_O16_s3_w129E61E12-E398-496E-9408-092D2D70E74B_mask.npy"
      },
      {
        "id": "IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A",
        "split": "training",
        "image": "data/images/images/IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A.tif",
        "source_mask": "data/masks/masks/IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 121,
        "max": 3490,
        "instances": 122,
        "image_sha256": "c6ed48d778e7df5677ef9481f26d81c81ba20ffda807f9df2291cc7c8d278ffb",
        "selected": true,
        "mask": "data/prepared/IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A_mask.npy"
      },
      {
        "id": "IXMtest_P09_s4_w11E382363-8C26-4A73-9D8A-8193D66E5785",
        "split": "training",
        "image": "data/images/images/IXMtest_P09_s4_w11E382363-8C26-4A73-9D8A-8193D66E5785.tif",
        "source_mask": "data/masks/masks/IXMtest_P09_s4_w11E382363-8C26-4A73-9D8A-8193D66E5785.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 114,
        "max": 3709,
        "instances": 17,
        "image_sha256": "0df845d706ea4f0fb9ab8cb148017f75a333369c249da96b1911b5281f2896ec",
        "selected": true,
        "mask": "data/prepared/IXMtest_P09_s4_w11E382363-8C26-4A73-9D8A-8193D66E5785_mask.npy"
      },
      {
        "id": "IXMtest_P23_s9_w1CA355675-EEC7-40B6-82C8-DC7E52A7A825",
        "split": "training",
        "image": "data/images/images/IXMtest_P23_s9_w1CA355675-EEC7-40B6-82C8-DC7E52A7A825.tif",
        "source_mask": "data/masks/masks/IXMtest_P23_s9_w1CA355675-EEC7-40B6-82C8-DC7E52A7A825.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 113,
        "max": 2115,
        "instances": 140,
        "image_sha256": "15839e3214ee8dc2bceffea7812bbd08a9f50c4e2879cd9ab34842c6c81bc241",
        "selected": true,
        "mask": "data/prepared/IXMtest_P23_s9_w1CA355675-EEC7-40B6-82C8-DC7E52A7A825_mask.npy"
      }
    ],
    "validation": [
      {
        "id": "IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78",
        "split": "validation",
        "image": "data/images/images/IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78.tif",
        "source_mask": "data/masks/masks/IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 125,
        "max": 3271,
        "instances": 120,
        "image_sha256": "d7888558da0167835ee5fd0455f16316f131ec2056a37927ce240ba71ce91d59",
        "selected": true,
        "mask": "data/prepared/IXMtest_B12_s2_w19F7E0279-D087-4B5E-9899-61971C29CB78_mask.npy"
      },
      {
        "id": "IXMtest_B19_s7_w1E43B84DB-39E2-4BFB-8CB4-554B32098C75",
        "split": "validation",
        "image": "data/images/images/IXMtest_B19_s7_w1E43B84DB-39E2-4BFB-8CB4-554B32098C75.tif",
        "source_mask": "data/masks/masks/IXMtest_B19_s7_w1E43B84DB-39E2-4BFB-8CB4-554B32098C75.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 118,
        "max": 2019,
        "instances": 93,
        "image_sha256": "c983a40730eceae5668a088ae6e93cf1e5daabd2ef43483c14575ed54d0d5b58",
        "selected": true,
        "mask": "data/prepared/IXMtest_B19_s7_w1E43B84DB-39E2-4BFB-8CB4-554B32098C75_mask.npy"
      },
      {
        "id": "IXMtest_D19_s6_w1EB1F11AE-4FB6-481F-94D9-40246870F0CB",
        "split": "validation",
        "image": "data/images/images/IXMtest_D19_s6_w1EB1F11AE-4FB6-481F-94D9-40246870F0CB.tif",
        "source_mask": "data/masks/masks/IXMtest_D19_s6_w1EB1F11AE-4FB6-481F-94D9-40246870F0CB.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 127,
        "max": 3475,
        "instances": 152,
        "image_sha256": "63168d8d7c9b9551866c5846fc32d3f53597d03211a2c3b8c207c61782e3d596",
        "selected": true,
        "mask": "data/prepared/IXMtest_D19_s6_w1EB1F11AE-4FB6-481F-94D9-40246870F0CB_mask.npy"
      },
      {
        "id": "IXMtest_F12_s5_w17F3E9DFC-6705-40A9-B5FE-C60261D73052",
        "split": "validation",
        "image": "data/images/images/IXMtest_F12_s5_w17F3E9DFC-6705-40A9-B5FE-C60261D73052.tif",
        "source_mask": "data/masks/masks/IXMtest_F12_s5_w17F3E9DFC-6705-40A9-B5FE-C60261D73052.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 127,
        "max": 4095,
        "instances": 138,
        "image_sha256": "0ffa34379cf2c3edb21feed483ac7ec45c26e62c6b8a5585e17656ddd801152c",
        "selected": true,
        "mask": "data/prepared/IXMtest_F12_s5_w17F3E9DFC-6705-40A9-B5FE-C60261D73052_mask.npy"
      },
      {
        "id": "IXMtest_I07_s4_w1F156255A-3842-46FB-ABF2-9D041E523F86",
        "split": "validation",
        "image": "data/images/images/IXMtest_I07_s4_w1F156255A-3842-46FB-ABF2-9D041E523F86.tif",
        "source_mask": "data/masks/masks/IXMtest_I07_s4_w1F156255A-3842-46FB-ABF2-9D041E523F86.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 115,
        "max": 1820,
        "instances": 118,
        "image_sha256": "0001556fe12ae4b5a6c24edcd522c4033f7d375ba457c105e4d62d05ffbca078",
        "selected": true,
        "mask": "data/prepared/IXMtest_I07_s4_w1F156255A-3842-46FB-ABF2-9D041E523F86_mask.npy"
      },
      {
        "id": "IXMtest_M23_s8_w118BC311D-A998-4161-8256-22839B2421F2",
        "split": "validation",
        "image": "data/images/images/IXMtest_M23_s8_w118BC311D-A998-4161-8256-22839B2421F2.tif",
        "source_mask": "data/masks/masks/IXMtest_M23_s8_w118BC311D-A998-4161-8256-22839B2421F2.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 125,
        "max": 4095,
        "instances": 162,
        "image_sha256": "98833468663e3b473ee718fb837c2d73069dc5c1afdcc565ea7c498d5c08e00c",
        "selected": true,
        "mask": "data/prepared/IXMtest_M23_s8_w118BC311D-A998-4161-8256-22839B2421F2_mask.npy"
      },
      {
        "id": "IXMtest_N12_s7_w166EF3FAB-EA33-4B28-91E3-034A1654BAAE",
        "split": "validation",
        "image": "data/images/images/IXMtest_N12_s7_w166EF3FAB-EA33-4B28-91E3-034A1654BAAE.tif",
        "source_mask": "data/masks/masks/IXMtest_N12_s7_w166EF3FAB-EA33-4B28-91E3-034A1654BAAE.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 119,
        "max": 2766,
        "instances": 202,
        "image_sha256": "d0073eed82bfb885c193b131118a5609d026fbe46068ffbed346ac3cce4549cf",
        "selected": true,
        "mask": "data/prepared/IXMtest_N12_s7_w166EF3FAB-EA33-4B28-91E3-034A1654BAAE_mask.npy"
      },
      {
        "id": "IXMtest_P15_s3_w10F5E9699-743C-4177-93CE-27CFD65A925E",
        "split": "validation",
        "image": "data/images/images/IXMtest_P15_s3_w10F5E9699-743C-4177-93CE-27CFD65A925E.tif",
        "source_mask": "data/masks/masks/IXMtest_P15_s3_w10F5E9699-743C-4177-93CE-27CFD65A925E.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 123,
        "max": 2884,
        "instances": 105,
        "image_sha256": "bce6322539875bad32697c6a69cad6728f410fbbe904378bf51ad4e289655f6c",
        "selected": true,
        "mask": "data/prepared/IXMtest_P15_s3_w10F5E9699-743C-4177-93CE-27CFD65A925E_mask.npy"
      }
    ],
    "test": [
      {
        "id": "IXMtest_A12_s7_w1EAEEA614-51ED-43B3-A4FF-088730911E4C",
        "split": "test",
        "image": "data/images/images/IXMtest_A12_s7_w1EAEEA614-51ED-43B3-A4FF-088730911E4C.tif",
        "source_mask": "data/masks/masks/IXMtest_A12_s7_w1EAEEA614-51ED-43B3-A4FF-088730911E4C.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 112,
        "max": 3885,
        "instances": 69,
        "image_sha256": "d49d90f12cde7176191b0acbf2ed783c39fe8bdf0caf968ca5f8a1627429b8f1",
        "selected": true,
        "mask": "data/prepared/IXMtest_A12_s7_w1EAEEA614-51ED-43B3-A4FF-088730911E4C_mask.npy"
      },
      {
        "id": "IXMtest_C07_s5_w12C9F2926-A017-4962-8660-72C9C20C86E4",
        "split": "test",
        "image": "data/images/images/IXMtest_C07_s5_w12C9F2926-A017-4962-8660-72C9C20C86E4.tif",
        "source_mask": "data/masks/masks/IXMtest_C07_s5_w12C9F2926-A017-4962-8660-72C9C20C86E4.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 116,
        "max": 1273,
        "instances": 156,
        "image_sha256": "3ebabf72d4a0d737cdf345f291cb67c5ce8fcdb7ccaaa52c874d8abdc1413143",
        "selected": true,
        "mask": "data/prepared/IXMtest_C07_s5_w12C9F2926-A017-4962-8660-72C9C20C86E4_mask.npy"
      },
      {
        "id": "IXMtest_D20_s3_w19D371AF3-0189-48A3-AF3B-D108DE6A017F",
        "split": "test",
        "image": "data/images/images/IXMtest_D20_s3_w19D371AF3-0189-48A3-AF3B-D108DE6A017F.tif",
        "source_mask": "data/masks/masks/IXMtest_D20_s3_w19D371AF3-0189-48A3-AF3B-D108DE6A017F.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 133,
        "max": 1918,
        "instances": 169,
        "image_sha256": "9a5a09ec913e10abeecf8cdfa3f467c6f414505553375d0db887bb5fea80fc72",
        "selected": true,
        "mask": "data/prepared/IXMtest_D20_s3_w19D371AF3-0189-48A3-AF3B-D108DE6A017F_mask.npy"
      },
      {
        "id": "IXMtest_E07_s1_w1641C6847-142B-463F-8B08-5B3296615572",
        "split": "test",
        "image": "data/images/images/IXMtest_E07_s1_w1641C6847-142B-463F-8B08-5B3296615572.tif",
        "source_mask": "data/masks/masks/IXMtest_E07_s1_w1641C6847-142B-463F-8B08-5B3296615572.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 129,
        "max": 4095,
        "instances": 170,
        "image_sha256": "2de13faf836648622bee872a0106047221dd6b2eceb529fcfc1a60a9e16f3451",
        "selected": true,
        "mask": "data/prepared/IXMtest_E07_s1_w1641C6847-142B-463F-8B08-5B3296615572_mask.npy"
      },
      {
        "id": "IXMtest_F12_s8_w1DBD80811-5297-4415-ACD1-EC9286BE76A4",
        "split": "test",
        "image": "data/images/images/IXMtest_F12_s8_w1DBD80811-5297-4415-ACD1-EC9286BE76A4.tif",
        "source_mask": "data/masks/masks/IXMtest_F12_s8_w1DBD80811-5297-4415-ACD1-EC9286BE76A4.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 115,
        "max": 2148,
        "instances": 7,
        "image_sha256": "5c34b879cb7103c334cfc9de3f8ff21d128d09504771de8f8cf8b0250b9a46d4",
        "selected": true,
        "mask": "data/prepared/IXMtest_F12_s8_w1DBD80811-5297-4415-ACD1-EC9286BE76A4_mask.npy"
      },
      {
        "id": "IXMtest_G01_s2_w1FBE52723-8BDF-4346-89BB-216A4A69ED1C",
        "split": "test",
        "image": "data/images/images/IXMtest_G01_s2_w1FBE52723-8BDF-4346-89BB-216A4A69ED1C.tif",
        "source_mask": "data/masks/masks/IXMtest_G01_s2_w1FBE52723-8BDF-4346-89BB-216A4A69ED1C.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 116,
        "max": 1364,
        "instances": 92,
        "image_sha256": "206e9a06614db3a70ddfaafa412436cc92ce6e148e2d2f07e6179594a435e62f",
        "selected": true,
        "mask": "data/prepared/IXMtest_G01_s2_w1FBE52723-8BDF-4346-89BB-216A4A69ED1C_mask.npy"
      },
      {
        "id": "IXMtest_I12_s1_w18A06F07F-CD8F-4056-B041-3967BE48AD78",
        "split": "test",
        "image": "data/images/images/IXMtest_I12_s1_w18A06F07F-CD8F-4056-B041-3967BE48AD78.tif",
        "source_mask": "data/masks/masks/IXMtest_I12_s1_w18A06F07F-CD8F-4056-B041-3967BE48AD78.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 126,
        "max": 3856,
        "instances": 202,
        "image_sha256": "cc9232820fb734b61ffd37dc1c5d3f867f5ef19723950707f466b7e58940df30",
        "selected": true,
        "mask": "data/prepared/IXMtest_I12_s1_w18A06F07F-CD8F-4056-B041-3967BE48AD78_mask.npy"
      },
      {
        "id": "IXMtest_I17_s6_w1EB7CA00F-DF95-466C-BF8C-6304B6A4974E",
        "split": "test",
        "image": "data/images/images/IXMtest_I17_s6_w1EB7CA00F-DF95-466C-BF8C-6304B6A4974E.tif",
        "source_mask": "data/masks/masks/IXMtest_I17_s6_w1EB7CA00F-DF95-466C-BF8C-6304B6A4974E.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 122,
        "max": 3448,
        "instances": 154,
        "image_sha256": "f11d4a8a586126d11128fd5b5f991399787ca7850d47c1df3947c764c65165de",
        "selected": true,
        "mask": "data/prepared/IXMtest_I17_s6_w1EB7CA00F-DF95-466C-BF8C-6304B6A4974E_mask.npy"
      },
      {
        "id": "IXMtest_J17_s1_w19E7BCD2F-6D7E-4B39-9154-F93E4D539183",
        "split": "test",
        "image": "data/images/images/IXMtest_J17_s1_w19E7BCD2F-6D7E-4B39-9154-F93E4D539183.tif",
        "source_mask": "data/masks/masks/IXMtest_J17_s1_w19E7BCD2F-6D7E-4B39-9154-F93E4D539183.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 122,
        "max": 1626,
        "instances": 82,
        "image_sha256": "48773bf2733767c4d50ec853bb12e92641215b5810a1a38c60e8272e4e7a05d2",
        "selected": true,
        "mask": "data/prepared/IXMtest_J17_s1_w19E7BCD2F-6D7E-4B39-9154-F93E4D539183_mask.npy"
      },
      {
        "id": "IXMtest_K03_s5_w1DC4CE558-042C-482E-8CAE-FCCB57AA9A55",
        "split": "test",
        "image": "data/images/images/IXMtest_K03_s5_w1DC4CE558-042C-482E-8CAE-FCCB57AA9A55.tif",
        "source_mask": "data/masks/masks/IXMtest_K03_s5_w1DC4CE558-042C-482E-8CAE-FCCB57AA9A55.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 125,
        "max": 2375,
        "instances": 139,
        "image_sha256": "43fa2c513feb0812c341c120b8860be14c6b28b7d68a5dd98c95c35ad627fd43",
        "selected": true,
        "mask": "data/prepared/IXMtest_K03_s5_w1DC4CE558-042C-482E-8CAE-FCCB57AA9A55_mask.npy"
      },
      {
        "id": "IXMtest_K17_s5_w1BB3AD9D7-3A85-458C-AB1C-07A816C2C203",
        "split": "test",
        "image": "data/images/images/IXMtest_K17_s5_w1BB3AD9D7-3A85-458C-AB1C-07A816C2C203.tif",
        "source_mask": "data/masks/masks/IXMtest_K17_s5_w1BB3AD9D7-3A85-458C-AB1C-07A816C2C203.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 118,
        "max": 3112,
        "instances": 149,
        "image_sha256": "42ed7955651a252795b5d0bc78a69ef6c2949383966a8d30fbfbf43f1de10fb8",
        "selected": true,
        "mask": "data/prepared/IXMtest_K17_s5_w1BB3AD9D7-3A85-458C-AB1C-07A816C2C203_mask.npy"
      },
      {
        "id": "IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2",
        "split": "test",
        "image": "data/images/images/IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2.tif",
        "source_mask": "data/masks/masks/IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 120,
        "max": 1305,
        "instances": 152,
        "image_sha256": "72aa16357727f76578b5c44a5167caf5284045c3e2fd4d75648363240153d160",
        "selected": true,
        "mask": "data/prepared/IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2_mask.npy"
      },
      {
        "id": "IXMtest_N08_s2_w159DEEDD5-2882-4F9E-9048-F5BD2B918651",
        "split": "test",
        "image": "data/images/images/IXMtest_N08_s2_w159DEEDD5-2882-4F9E-9048-F5BD2B918651.tif",
        "source_mask": "data/masks/masks/IXMtest_N08_s2_w159DEEDD5-2882-4F9E-9048-F5BD2B918651.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 118,
        "max": 1773,
        "instances": 127,
        "image_sha256": "0d943128297bc37671c31a2624e2a8ec60ccdc235541c7f9cd0ccd8775538af8",
        "selected": true,
        "mask": "data/prepared/IXMtest_N08_s2_w159DEEDD5-2882-4F9E-9048-F5BD2B918651_mask.npy"
      },
      {
        "id": "IXMtest_P07_s8_w144364F25-950A-472C-A529-1A9AD0433B6C",
        "split": "test",
        "image": "data/images/images/IXMtest_P07_s8_w144364F25-950A-472C-A529-1A9AD0433B6C.tif",
        "source_mask": "data/masks/masks/IXMtest_P07_s8_w144364F25-950A-472C-A529-1A9AD0433B6C.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 122,
        "max": 4095,
        "instances": 113,
        "image_sha256": "0a3a070ed15b242a3b8557d4533fb9fdb030577bd70686668f6deee0695fa10b",
        "selected": true,
        "mask": "data/prepared/IXMtest_P07_s8_w144364F25-950A-472C-A529-1A9AD0433B6C_mask.npy"
      },
      {
        "id": "IXMtest_P15_s8_w1D26A5BC7-CF59-4027-B785-85AD33773609",
        "split": "test",
        "image": "data/images/images/IXMtest_P15_s8_w1D26A5BC7-CF59-4027-B785-85AD33773609.tif",
        "source_mask": "data/masks/masks/IXMtest_P15_s8_w1D26A5BC7-CF59-4027-B785-85AD33773609.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 118,
        "max": 3449,
        "instances": 82,
        "image_sha256": "cbf81c613d35a1918ba7734e795241a93d75f445e70664208a6aee5b3e949e53",
        "selected": true,
        "mask": "data/prepared/IXMtest_P15_s8_w1D26A5BC7-CF59-4027-B785-85AD33773609_mask.npy"
      },
      {
        "id": "IXMtest_P19_s6_w10EBAD15B-2822-48B9-B9C8-4DF5D80AB940",
        "split": "test",
        "image": "data/images/images/IXMtest_P19_s6_w10EBAD15B-2822-48B9-B9C8-4DF5D80AB940.tif",
        "source_mask": "data/masks/masks/IXMtest_P19_s6_w10EBAD15B-2822-48B9-B9C8-4DF5D80AB940.png",
        "shape": [
          520,
          696
        ],
        "dtype": "uint16",
        "min": 119,
        "max": 1633,
        "instances": 89,
        "image_sha256": "c71d72360df1ca6cd67c6fef964787c3e4918c877e2ee98928255bb20c0039e2",
        "selected": true,
        "mask": "data/prepared/IXMtest_P19_s6_w10EBAD15B-2822-48B9-B9C8-4DF5D80AB940_mask.npy"
      }
    ]
  },
  "seed": 42,
  "pixel_size_um": null,
  "mask_decode": "skimage.measure.label(PNG first channel, background=0), official BBBC039 decoding",
  "channels": "raw grayscale; actual inference forward1channel; official training preprocessing adds two zero channels",
  "split_policy": "official original-image split, deterministic subset before crops"
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-a4a8ad74e74f

원래 경로: `cellpose/data/sources.json` · SHA256: `df6b3fe619e00362866aa3c2182fe4f2f8e49464ee4eb7dfe25f604f510f9722`

<details>
<summary>sources.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "cellpose/data/sources.json", "sha256": "df6b3fe619e00362866aa3c2182fe4f2f8e49464ee4eb7dfe25f604f510f9722", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "path": "data/images.zip",
    "url": "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
    "revision": "BBBC039v1",
    "bytes": 77915748,
    "sha256": "6f30a5d4fe38c928ded972704f085975f8dc0d65d9aa366df00e5a9d449fddd7",
    "headers": {
      "date": "Tue, 27 Oct 2026 10:22:08 GMT",
      "server": "Apache",
      "last-modified": "Fri, 08 Jun 2018 16:52:44 GMT",
      "etag": "\"4a4e664-56e243a90cf00\"",
      "accept-ranges": "bytes",
      "content-length": "77915748",
      "access-control-allow-origin": "*",
      "content-type": "application/zip",
      "connection": "close"
    },
    "checked_utc": "2026-10-27T10:22:27.223774+00:00",
    "license_note": "BBBC039v1 CC0; model license notes preserved from official source"
  },
  {
    "path": "data/masks.zip",
    "url": "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
    "revision": "BBBC039v1",
    "bytes": 2753811,
    "sha256": "f9e6043d8ca56344a4886f96a700d804d6ee982f31e2b2cd3194af2a053c2710",
    "headers": {
      "date": "Tue, 27 Oct 2026 10:22:28 GMT",
      "server": "Apache",
      "last-modified": "Fri, 08 Jun 2018 16:52:46 GMT",
      "etag": "\"2a0513-56e243aaf5380\"",
      "accept-ranges": "bytes",
      "content-length": "2753811",
      "access-control-allow-origin": "*",
      "content-type": "application/zip",
      "connection": "close"
    },
    "checked_utc": "2026-10-27T10:22:29.750559+00:00",
    "license_note": "BBBC039v1 CC0; model license notes preserved from official source"
  },
  {
    "path": "data/metadata.zip",
    "url": "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
    "revision": "BBBC039v1",
    "bytes": 17816,
    "sha256": "a2c1f900bed9ba92a99553efd4c2ae98598433691c7401d818653ab61110deb2",
    "headers": {
      "date": "Tue, 27 Oct 2026 10:22:30 GMT",
      "server": "Apache",
      "last-modified": "Fri, 08 Jun 2018 16:52:52 GMT",
      "etag": "\"4598-56e243b0ae100\"",
      "accept-ranges": "bytes",
      "content-length": "17816",
      "access-control-allow-origin": "*",
      "content-type": "application/zip",
      "connection": "close"
    },
    "checked_utc": "2026-10-27T10:22:30.461570+00:00",
    "license_note": "BBBC039v1 CC0; model license notes preserved from official source"
  },
  {
    "path": "models/cpsam_v2",
    "revision": "7c61431b5fbb078f3296754bd15d9f51b320f837",
    "sha256": "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667",
    "bytes": 1233586851,
    "url": "https://huggingface.co/mouseland/cellpose-sam/resolve/7c61431b5fbb078f3296754bd15d9f51b320f837/cpsam_v2",
    "method": "Content-Range verified chunks after preserved prefix"
  }
]
```
<!-- END_RESTORE_FILE -->


</details>
