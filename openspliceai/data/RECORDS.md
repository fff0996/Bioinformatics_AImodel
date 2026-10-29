# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-246beda1e980

원래 경로: `openspliceai/data/input_audit.json` · SHA256: `967ab007ff41bf8c6f7796afd752b6e246ce9c12dc6b84634ca961f2f205fb23`

<details>
<summary>input_audit.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "openspliceai/data/input_audit.json", "sha256": "967ab007ff41bf8c6f7796afd752b6e246ce9c12dc6b84634ca961f2f205fb23", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "reference": "GRCh38.primary_assembly.genome.fa (existing BEND reference)",
  "annotation": "source/openspliceai/variant/annotations/grch38.txt",
  "annotation_sha256": "565c0ea83f66b2210182364c3d626fef526229e3d02f444bab247329b6695f2c",
  "coordinate_policy": "genomic0based; plus donor exon_end-1, acceptor next_exon_start; minus swapped then reverse-complement; labeled exon bases per official create-data motif convention",
  "source_checkpoint": "official OSAI_MANE10000 ensemble seeds10..14; annotation here is bundled grch38 transcript table, not asserted identical MANE training annotation",
  "splits": {
    "train": {
      "n": 96,
      "chromosomes": [
        "2",
        "4"
      ],
      "input_nt": 20000,
      "output_label_nt": 10000,
      "motifs": {
        "donor": {
          "GT": 286,
          "CT": 1,
          "AT": 2,
          "GC": 1
        },
        "acceptor": {
          "AG": 304,
          "AC": 2
        }
      },
      "first_example": {
        "id": "FAM110C_2_43532_-",
        "gene": "FAM110C",
        "chromosome": "chr2",
        "strand": "-",
        "genomic_input_start": 33532,
        "genomic_input_end": 53532,
        "label_counts": {
          "0": 9998,
          "2": 1,
          "1": 1
        }
      }
    },
    "validation": {
      "n": 24,
      "chromosomes": [
        "6"
      ],
      "input_nt": 20000,
      "output_label_nt": 10000,
      "motifs": {
        "donor": {
          "GT": 56,
          "GC": 1
        },
        "acceptor": {
          "AG": 63
        }
      },
      "first_example": {
        "id": "IRF4_6_392452_+",
        "gene": "IRF4",
        "chromosome": "chr6",
        "strand": "+",
        "genomic_input_start": 382452,
        "genomic_input_end": 402452,
        "label_counts": {
          "0": 9991,
          "2": 5,
          "1": 4
        }
      }
    },
    "test": {
      "n": 36,
      "chromosomes": [
        "1",
        "3"
      ],
      "input_nt": 20000,
      "output_label_nt": 10000,
      "motifs": {
        "donor": {
          "GT": 154,
          "GC": 2,
          "AT": 1
        },
        "acceptor": {
          "AG": 149,
          "AA": 1
        }
      },
      "first_example": {
        "id": "SAMD11_1_925860_+",
        "gene": "SAMD11",
        "chromosome": "chr1",
        "strand": "+",
        "genomic_input_start": 915860,
        "genomic_input_end": 935860,
        "label_counts": {
          "0": 9995,
          "2": 3,
          "1": 2
        }
      }
    }
  }
}
```
<!-- END_RESTORE_FILE -->


</details>
