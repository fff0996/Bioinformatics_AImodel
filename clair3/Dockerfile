FROM nvcr.io/nvidia/pytorch:26.06-py3
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends samtools bcftools tabix parallel pigz libhts-dev libcurl4-gnutls-dev libssl-dev liblzma-dev libbz2-dev zlib1g-dev libdeflate-dev && apt-get clean
