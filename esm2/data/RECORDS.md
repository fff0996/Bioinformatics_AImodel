# 원본 설정·측정 기록

파일 내용을 Markdown 코드 블록에 그대로 보존했다. 본문 결과 해석은 모델별 `reports/FINAL.md`를 우선한다. CSV/JSON 원본 파일이 필요하면 `docs/RESTORE_FILES.md`의 복원 코드를 사용한다. PNG는 base64로 보관하며 본문 그림은 이전 커밋의 고정 원본 URL로 표시한다.

## file-d065fc584239

원래 경로: `esm2/data/sources.json` · SHA256: `e7ebd5ffacbf41ccbd2decc584223c4ce9cd442ef1feef7559022c9c52a80f71`

<details>
<summary>sources.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "esm2/data/sources.json", "sha256": "e7ebd5ffacbf41ccbd2decc584223c4ce9cd442ef1feef7559022c9c52a80f71", "encoding": "utf-8", "added_final_newline": true} -->
```json
[
  {
    "path": "data/uniprot_reviewed.fasta",
    "url": "https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20length%3A%5B50%20TO%201022%5D&format=fasta&size=500",
    "bytes": 259221,
    "sha256": "37222bb198046868760dd240413b9dd46fcd44aefc0ea68d67929f9f43a8c7d3",
    "headers": {
      "Vary": "accept,accept-encoding,x-uniprot-release,x-api-deployment-date, User-Agent",
      "Cache-Control": "public, max-age=43200",
      "x-cache": "miss cached",
      "Content-Type": "text/plain;format=fasta",
      "Access-Control-Allow-Credentials": "true",
      "Access-Control-Expose-Headers": "Link, X-Total-Results, X-UniProt-Release, X-UniProt-Release-Date, X-API-Deployment-Date",
      "X-API-Deployment-Date": "07-October-2026",
      "strict-transport-security": "max-age=31536000; includeSubDomains; preload",
      "Date": "Tue, 27 Oct 2026 07:40:19 GMT",
      "Access-Control-Max-Age": "1728000",
      "X-UniProt-Release": "2026_03",
      "Link": "<https://rest.uniprot.org/uniprotkb/search?format=fasta&query=reviewed%3Atrue%20AND%20length%3A%5B50%20TO%201022%5D&cursor=c9bacmxsqhkqgdwiuhth8jj7to99sscq47ho0&size=500>; rel=\"next\"",
      "X-Total-Results": "544243",
      "Transfer-Encoding": "chunked",
      "Access-Control-Allow-Origin": "*",
      "Connection": "close",
      "Access-Control-Allow-Methods": "GET, PUT, POST, DELETE, PATCH, OPTIONS",
      "Access-Control-Allow-Headers": "DNT,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization",
      "X-UniProt-Release-Date": "02-September-2026"
    },
    "checked_utc": "2026-10-27T07:40:20.661126+00:00"
  },
  {
    "path": "data/fluorescence.tar.gz",
    "url": "https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz",
    "bytes": 1635678,
    "sha256": "20c9d114234a5136c4087920480e7fa2d586a653061f0cd141f626a984ede8bd",
    "headers": {
      "x-amz-id-2": "oO4IwE+XhyJAc2m+yxgfS8OMDfIPWKIfh9F/LmyoAgsMMm859bgl92/nQBHcb2k2UOSVt/RDOZ4=",
      "x-amz-request-id": "Y3QREGECEFEMDCWG",
      "Server": "AmazonS3",
      "Date": "Tue, 27 Oct 2026 07:40:22 GMT",
      "Last-Modified": "Mon, 02 Aug 2021 21:35:32 GMT",
      "ETag": "\"d63d1d51ec8c20ff0d981e4cbd67457a\"",
      "x-amz-server-side-encryption": "AES256",
      "x-amz-version-id": "XaIPt6wWSWvLXtAM7KoacyXabIlxP6bz",
      "Accept-Ranges": "bytes",
      "Content-Type": "application/x-tar",
      "Content-Length": "1635678",
      "Connection": "close"
    },
    "checked_utc": "2026-10-27T07:40:23.125362+00:00"
  },
  {
    "path": "models/esm2_t33_650M_UR50D.pt",
    "url": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
    "bytes": 2604537549,
    "sha256": "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
    "headers": {
      "Content-Type": "binary/octet-stream",
      "Content-Length": "2604537549",
      "Connection": "close",
      "Accept-Ranges": "bytes",
      "Server": "AmazonS3",
      "Last-Modified": "Thu, 18 Aug 2022 13:26:44 GMT",
      "x-amz-version-id": "2G4typsUSwOKQLH35sqPMJ27ZLX1AEJG",
      "x-amz-server-side-encryption": "AES256",
      "x-amz-storage-class": "INTELLIGENT_TIERING",
      "Date": "Tue, 27 Oct 2026 07:40:24 GMT",
      "ETag": "\"12a18098227c0ff911354647d25d494d-311\"",
      "X-Cache": "RefreshHit from cloudfront",
      "Via": "1.1 540723249b706f7cf8fcf2324ac3eb4c.cloudfront.net (CloudFront)",
      "X-Amz-Cf-Pop": "ICN57-P4",
      "X-Amz-Cf-Id": "RWk98Ya6B5Jbk-isYd-U9BFY_lei3kuOkQVcbRh-WEY9MWDsKaEP5g=="
    },
    "checked_utc": "2026-10-27T07:40:47.438005+00:00"
  },
  {
    "path": "models/esm2_t33_650M_UR50D-contact-regression.pt",
    "url": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt",
    "bytes": 3687,
    "sha256": "8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575",
    "headers": {
      "Content-Type": "binary/octet-stream",
      "Content-Length": "3687",
      "Connection": "close",
      "Accept-Ranges": "bytes",
      "Server": "AmazonS3",
      "Last-Modified": "Thu, 18 Aug 2022 13:25:01 GMT",
      "x-amz-version-id": "cEcpCgy3_xlt3NCWsoXOzeIrh7.EDOBu",
      "x-amz-server-side-encryption": "AES256",
      "Date": "Tue, 27 Oct 2026 07:40:48 GMT",
      "ETag": "\"743354dd846c9460fd2fe50892033543\"",
      "X-Cache": "RefreshHit from cloudfront",
      "Via": "1.1 7685cf04fa239254cb7d83b3bae5a1ec.cloudfront.net (CloudFront)",
      "X-Amz-Cf-Pop": "ICN57-P4",
      "X-Amz-Cf-Id": "TJN0wv0A9IFE3WwZ2GmEqHCg6O7Gx64gVnqWZUmbVj2M6y3KxT3d0Q=="
    },
    "checked_utc": "2026-10-27T07:40:47.889818+00:00"
  },
  {
    "path": "data/official_extract.py",
    "url": "https://raw.githubusercontent.com/facebookresearch/esm/main/scripts/extract.py",
    "bytes": 5161,
    "sha256": "9bcd76f69d434af390e9ac27f9cc1f09e701bfdfc7836a1409679cdc24b9c348",
    "headers": {
      "Connection": "close",
      "Content-Length": "5161",
      "Cache-Control": "max-age=300",
      "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
      "Content-Type": "text/plain; charset=utf-8",
      "ETag": "\"2fdc61ed643211c77d3f2044d04429de35eec769ff8ce95fc922c9da099192a6\"",
      "Strict-Transport-Security": "max-age=31536000",
      "X-Content-Type-Options": "nosniff",
      "X-Frame-Options": "deny",
      "X-XSS-Protection": "1; mode=block",
      "X-GitHub-Request-Id": "B96C:95E3B:156E9:2A52B:6AE0557F",
      "x-github-edge-region": "koreacentral",
      "Accept-Ranges": "bytes",
      "Date": "Tue, 27 Oct 2026 07:40:48 GMT",
      "Via": "1.1 varnish",
      "X-Served-By": "cache-icn1450066-ICN",
      "X-Cache": "MISS",
      "X-Cache-Hits": "0",
      "X-Timer": "S1793086848.916004,VS0,VE203",
      "Vary": "Authorization,Accept-Encoding",
      "Access-Control-Allow-Origin": "*",
      "Cross-Origin-Resource-Policy": "cross-origin",
      "X-Fastly-Request-ID": "4b23bbad0924f8e2d25757d66817c83911d6bb2f",
      "Expires": "Tue, 27 Oct 2026 07:45:48 GMT",
      "Source-Age": "0"
    },
    "checked_utc": "2026-10-27T07:40:48.126357+00:00"
  }
]
```
<!-- END_RESTORE_FILE -->


</details>

## file-9d752d6d300c

원래 경로: `esm2/data/uniprot_reviewed_900_950_20261028.source.json` · SHA256: `9d7a8dc43fb26d110ff81f926fdc7b5852cf373007301e91aee277c504319440`

<details>
<summary>uniprot_reviewed_900_950_20261028.source.json — 원문 전문</summary>


<!-- RESTORE_FILE {"path": "esm2/data/uniprot_reviewed_900_950_20261028.source.json", "sha256": "9d7a8dc43fb26d110ff81f926fdc7b5852cf373007301e91aee277c504319440", "encoding": "utf-8", "added_final_newline": true} -->
```json
{
  "url": "https://rest.uniprot.org/uniprotkb/search?query=reviewed%3Atrue%20AND%20length%3A%5B900%20TO%20950%5D&format=fasta&size=500",
  "sha256": "d2a6773c09022e8ced6526fb8f38a0fab9bf344cdcc39bda42a58acb9fa16732",
  "bytes": 521317,
  "headers": {
    "Vary": "accept,accept-encoding,x-uniprot-release,x-api-deployment-date, User-Agent",
    "Cache-Control": "public, max-age=43200",
    "x-cache": "miss cached",
    "Content-Type": "text/plain;format=fasta",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Expose-Headers": "Link, X-Total-Results, X-UniProt-Release, X-UniProt-Release-Date, X-API-Deployment-Date",
    "X-API-Deployment-Date": "07-October-2026",
    "strict-transport-security": "max-age=31536000; includeSubDomains; preload",
    "Date": "Wed, 28 Oct 2026 02:27:02 GMT",
    "Access-Control-Max-Age": "1728000",
    "X-UniProt-Release": "2026_03",
    "Link": "<https://rest.uniprot.org/uniprotkb/search?format=fasta&query=reviewed%3Atrue%20AND%20length%3A%5B900%20TO%20950%5D&cursor=c9bacmxsqhkqgdwa659orqyetr0ii434c477c&size=500>; rel=\"next\"",
    "X-Total-Results": "4294",
    "Transfer-Encoding": "chunked",
    "Access-Control-Allow-Origin": "*",
    "Connection": "close",
    "Access-Control-Allow-Methods": "GET, PUT, POST, DELETE, PATCH, OPTIONS",
    "Access-Control-Allow-Headers": "DNT,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization",
    "X-UniProt-Release-Date": "02-September-2026"
  },
  "retrieved_utc": "2026-10-28T02:27:03.871516+00:00"
}
```
<!-- END_RESTORE_FILE -->


</details>
