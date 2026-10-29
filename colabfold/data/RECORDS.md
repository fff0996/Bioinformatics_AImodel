# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-f432058ad404

원래 경로: `colabfold/data/complex_manifest.json` · SHA256: `fa58017a4f6964f391a0782e93ea1a60a580b73145250fc103380fa319c23593`

<details>
<summary>complex_manifest.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/data/complex_manifest.json", "sha256": "fa58017a4f6964f391a0782e93ea1a60a580b73145250fc103380fa319c23593", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "queries": [
    {
      "id": "3G5O_A_3G5O_B",
      "sequence": "MRILPISTIKGKLNEFVDAVSSTQDQITITKNGAPAAVLVGADEWESLQETLYWLAQPGIRESIAEADADIASGRTYGEDEIRAEFGVPRRPHMPYTVRFTTTARRDLHKLPPRILAAVVEFAFGDLSREPLRVGKPLRRELAGTFSARRGTYRLLYRIDDEHTTVVILRVDHRADIYRR",
      "chains": [
        "MRILPISTIKGKLNEFVDAVSSTQDQITITKNGAPAAVLVGADEWESLQETLYWLAQPGIRESIAEADADIASGRTYGEDEIRAEFGVPRRPH",
        "MPYTVRFTTTARRDLHKLPPRILAAVVEFAFGDLSREPLRVGKPLRRELAGTFSARRGTYRLLYRIDDEHTTVVILRVDHRADIYRR"
      ],
      "lengths": [
        93,
        87
      ],
      "length": 180,
      "source": "https://github.com/sokrypton/ColabFold/blob/v1.6.2/test-data/complex/input.csv",
      "kind": "complex"
    }
  ],
  "format": "FASTA with colon-separated chains",
  "unique_complexes": 1
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-61742f1279c9

원래 경로: `colabfold/data/input_manifest.json` · SHA256: `4984b41e06e061e5bbfac95bd693c73e080c2be5b825c71d21d79d9c5a237ba9`

<details>
<summary>input_manifest.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/data/input_manifest.json", "sha256": "4984b41e06e061e5bbfac95bd693c73e080c2be5b825c71d21d79d9c5a237ba9", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "created_utc": "2026-10-27T08:23:59.745490+00:00",
  "unique_sequences": 7,
  "queries": [
    {
      "id": "1UBQ",
      "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
      "length": 76,
      "kind": "length",
      "source": "https://www.rcsb.org/fasta/entry/1UBQ",
      "reference_chain": "A"
    },
    {
      "id": "count1_A2RU14",
      "sequence": "MAGTVLGVGAGVFILALLWVAVLLLCVLLSRASGAARFSVIFLFFGAVIITSVLLLFPRAGEFPAPEVEVKIVDDFFIGRYVLLAFLSAIFLGGLFLVLIHYVLEPIYAKPLHSY",
      "length": 115,
      "kind": "count",
      "source": "UniProtKB reviewed 2026_03; reused ESM-2 source snapshot",
      "source_id": "sp|A2RU14|TM218_HUMAN"
    },
    {
      "id": "count2_A6NFY7",
      "sequence": "MSRHSRLQRQVLSLYRDLLRAGRGKPGAEARVRAEFRQHAGLPRSDVLRIEYLYRRGRRQLQLLRSGHATAMGAFVRPRAPTGEPGGVGCQPDDGDSPRNPHDSTGAPETRPDGR",
      "length": 115,
      "kind": "count",
      "source": "UniProtKB reviewed 2026_03; reused ESM-2 source snapshot",
      "source_id": "sp|A6NFY7|SDHF1_HUMAN"
    },
    {
      "id": "count3_O14519",
      "sequence": "MSYKPNLAAHMPAAALNAAGSVHSPSTSMATSSQYRQLLSDYGPPSLGYTQGTGNSQVPQSKYAELLAIIEELGKEIRPTYAGSKSAMERLKRGIIHARGLVRECLAETERNARS",
      "length": 115,
      "kind": "count",
      "source": "UniProtKB reviewed 2026_03; reused ESM-2 source snapshot",
      "source_id": "sp|O14519|CDKA1_HUMAN"
    },
    {
      "id": "count4_P01270",
      "sequence": "MIPAKDMAKVMIVMLAICFLTKSDGKSVKKRSVSEIQLMHNLGKHLNSMERVEWLRKKLQDVHNFVALGAPLAPRDAGSQRPRKKEDNVLVESHEKSLGEADKADVNVLTKAKSQ",
      "length": 115,
      "kind": "count",
      "source": "UniProtKB reviewed 2026_03; reused ESM-2 source snapshot",
      "source_id": "sp|P01270|PTHY_HUMAN"
    },
    {
      "id": "1LYZ",
      "sequence": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
      "length": 129,
      "kind": "length",
      "source": "https://www.rcsb.org/fasta/entry/1LYZ",
      "reference_chain": "A"
    },
    {
      "id": "1TIM",
      "sequence": "APRKFFVGGNWKMNGKRKSLGELIHTLDGAKLSADTEVVCGAPSIYLDFARQKLDAKIGVAAQNCYKVPKGAFTGEISPAMIKDIGAAWVILGHSERRHVFGESDELIGQKVAHALAEGLGVIACIGEKLDEREAGITEKVVFQETKAIADNVKDWSKVVLAYEPVWAIGTGKTATPQQAQEVHEKLRGWLKTHVSDAVAVQSRIIYGGSVTGGNCKELASQHDVDGFLVGGASLKPEFVDIINAKH",
      "length": 247,
      "kind": "length",
      "source": "https://www.rcsb.org/fasta/entry/1TIM",
      "reference_chain": "A"
    }
  ],
  "format": "FASTA amino-acid sequences; PDB reference chain A; no labels for count proteins",
  "missing_sequences": 0,
  "sequence_dtype": "text",
  "preprocessed": false,
  "msa": "not yet generated"
}
```
<!-- END_RESTORE_FILE -->


</details>

## file-f519caea368d

원래 경로: `colabfold/data/sources.json` · SHA256: `1004dd2b9badd383344dcfaefdc80a44cab1f70cd29ddbe1fd064fa764dc72e6`

<details>
<summary>sources.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "colabfold/data/sources.json", "sha256": "1004dd2b9badd383344dcfaefdc80a44cab1f70cd29ddbe1fd064fa764dc72e6", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "files": [
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1LYZ.fasta",
      "bytes": 190,
      "sha256": "bd71bd464c5247807690c913b1183b196c3ac97c76175d6c00a56a98cf6a8c50",
      "url": "https://www.rcsb.org/fasta/entry/1LYZ",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1LYZ.pdb",
      "bytes": 145962,
      "sha256": "28e95e819d4e0fc48acdf44209e37ba06f5c9d5d93d9950d67a760b32964326e",
      "url": "https://files.rcsb.org/download/1LYZ.pdb",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1TIM.fasta",
      "bytes": 315,
      "sha256": "4bd6b869b65f358356d971ffd95e2ce8bccf332c19446f67b124f35c71e4045d",
      "url": "https://www.rcsb.org/fasta/entry/1TIM",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1TIM.pdb",
      "bytes": 351783,
      "sha256": "79b35b419696a1e368aad2121ae2712e46444b5978de1e86bdc286f482222cdc",
      "url": "https://files.rcsb.org/download/1TIM.pdb",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1UBQ.fasta",
      "bytes": 123,
      "sha256": "f00b1a97f51c157133e1e2b3f3ed868aa2edf79591e8b5206a1bdd0f3b7c94d2",
      "url": "https://www.rcsb.org/fasta/entry/1UBQ",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/1UBQ.pdb",
      "bytes": 78570,
      "sha256": "d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161",
      "url": "https://files.rcsb.org/download/1UBQ.pdb",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/3G5O.fasta",
      "bytes": 384,
      "sha256": "683c5fb83e2d8a7d2ce9d1ab790d362cbbd2d769b2a5c3ce68136cddd1db0821",
      "url": "https://www.rcsb.org/fasta/entry/3G5O",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/3G5O.pdb",
      "bytes": 294273,
      "sha256": "b18e40f567838e810fea65b1c83f93b8bd156e774705d89ba265ce0ef1164798",
      "url": "https://files.rcsb.org/download/3G5O.pdb",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/a3m_listing.json",
      "bytes": 2784,
      "sha256": "43525ad6c3c61f572677d162448ebb9272e57fb1f4b7078345764a66109ef5e6",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/batch_listing.json",
      "bytes": 2498,
      "sha256": "ac38f4f512acdc6ad60014197f231ec39e469c2e00a17ee609a7eab818645c8c",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/complex.fasta",
      "bytes": 197,
      "sha256": "b5e3fcbd9d0327489b05906830279b7f0356eaaec34ac97192827a25c1021702",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/complex_input.csv",
      "bytes": 225,
      "sha256": "847500a212bcd8a20c4737c5fd908e46af81a74eb7037032930a54176ffb295c",
      "url": "https://raw.githubusercontent.com/sokrypton/ColabFold/v1.6.2/test-data/complex/input.csv",
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/complex_listing.json",
      "bytes": 1847,
      "sha256": "1310d8c8f8858e5b46f03e54f03cc8800e9b3344d7a22f07c57f62437708a333",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/complex_manifest.json",
      "bytes": 763,
      "sha256": "fa58017a4f6964f391a0782e93ea1a60a580b73145250fc103380fa319c23593",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/input_manifest.json",
      "bytes": 2679,
      "sha256": "4984b41e06e061e5bbfac95bd693c73e080c2be5b825c71d21d79d9c5a237ba9",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/queries.fasta",
      "bytes": 997,
      "sha256": "ad75f1a3dd58b303cf99ada9c2a42a8823acc2c86093ec746f4b000ae4ce9294",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_batch_5AWL_1_listing.json",
      "bytes": 2087,
      "sha256": "fe9283e08f5d623171aa55180ecf1dcfbd62946a2dacec91fc0e3ed0c7bbb118",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_batch_input_listing.json",
      "bytes": 2977,
      "sha256": "6c71103bb1c40b7efdcb04a27a146d293ae9af574bf83ca00fa26d3cb4ccd20b",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_complex_3G5O_A_3G5O_B_listing.json",
      "bytes": 2201,
      "sha256": "69c2bc9e0cc70893f5622ad490adec41cee737c56d57a84681e10af8dc4aadfa",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_mmseqs-api-reponses_listing.json",
      "bytes": 7616,
      "sha256": "1beee2fedfcdb9832f9975065d94d458a6598b84fff6a899500c5c0d1238b8c9",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    },
    {
      "file": "/home/hyein/ai-benchmark/colabfold/data/test-data_single_5AWL_1_listing.json",
      "bytes": 2099,
      "sha256": "ef33c852576bdff2f72d1dbb9840a091ac7d6f854a8566f6d152e62d309eee77",
      "url": null,
      "checked_kst": "2026-10-27T18:25:48+09:00"
    }
  ],
  "uniprot_parent_source": "/home/hyein/ai-benchmark/esm2/data/sources.json",
  "msa_service": "https://api.colabfold.com",
  "msa_mode": "mmseqs2_uniref_env",
  "server_database_revision": null
}
```
<!-- END_RESTORE_FILE -->


</details>
