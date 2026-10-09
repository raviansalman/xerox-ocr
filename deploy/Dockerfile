# syntax=docker/dockerfile:1.7
# API and worker image: the docintel package with OCR (Tesseract) and office conversion (LibreOffice).
#   docker build -f deploy/Dockerfile -t docintel:2.0.0 .
# Behind a TLS-intercepting proxy, pass its CA only for the build (it is not kept in the image):
#   docker build --secret id=ca,src=proxy-ca.crt --build-arg HTTPS_PROXY=... -f deploy/Dockerfile .
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_THREAD_LIMIT=1 PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3.12 python3.12-venv \
        tesseract-ocr tesseract-ocr-eng \
        libreoffice-writer-nogui libreoffice-calc-nogui libreoffice-impress-nogui \
        fonts-dejavu-core libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && useradd --create-home --uid 10001 docintel

# Arabic OCR: Tesseract's "best" Arabic model (tessdata_best 4.1.0); on Arabic scans it made 40% fewer character
# errors than the distribution's standard model. Taken from deploy/tessdata/ when present (offline builds), else
# downloaded; the checksum is verified either way.
ARG TESSDATA_ARA_URL=https://github.com/tesseract-ocr/tessdata_best/raw/4.1.0/ara.traineddata
ARG TESSDATA_ARA_SHA256=ab9d157d8e38ca00e7e39c7d5363a5239e053f5b0dbdb3167dde9d8124335896
RUN --mount=type=bind,source=deploy/tessdata,target=/tessdata-src \
    f=/usr/share/tesseract-ocr/5/tessdata/ara.traineddata \
    && if [ -f /tessdata-src/ara.traineddata ]; then cp /tessdata-src/ara.traineddata "$f"; \
       else curl -fsSL "$TESSDATA_ARA_URL" -o "$f"; fi \
    && echo "$TESSDATA_ARA_SHA256  $f" | sha256sum -c -

WORKDIR /app
COPY pyproject.toml README.md ./
COPY docintel ./docintel
RUN --mount=type=secret,id=ca,required=false \
    if [ -s /run/secrets/ca ]; then cat /etc/ssl/certs/ca-certificates.crt /run/secrets/ca > /tmp/ca.pem; \
        export PIP_CERT=/tmp/ca.pem; fi \
    && pip install ".[answer]" && rm -f /tmp/ca.pem \
    && mkdir -p /data/objects && chown -R docintel:docintel /data     # a new named volume copies this ownership

USER docintel
ENV DOCINTEL_STORAGE_DIR=/data/objects HOME=/home/docintel
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fs http://127.0.0.1:8000/health/live || exit 1
CMD ["docintel", "serve", "--host", "0.0.0.0", "--port", "8000"]
