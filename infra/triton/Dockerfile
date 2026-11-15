# Triton + what our Python-backend models need (v8).
# Base pinned by digest like compose.yaml's other images (HLD §7). To bump:
#   docker buildx imagetools inspect nvcr.io/nvidia/tritonserver:<yy.mm>-py3
FROM nvcr.io/nvidia/tritonserver:26.10-py3@sha256:b404f7de5dbeeefe0b17d7d149745a50d0989a25d0d831433b5ba63c716c2fcd

# tokenizer model (Python backend). Same version as the local check script, so client
# and server tokenize identically. --break-system-packages: Ubuntu 24.04 marks the
# system Python as externally managed; this image is single-purpose, so that's fine.
RUN pip install --no-cache-dir --break-system-packages tokenizers==0.23.3
