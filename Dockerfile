# syntax=docker/dockerfile:1

# ── Build stage ───────────────────────────────────────────────────────────
FROM ghcr.io/hybridgroup/gocv:4.12.0 AS builder

# Install ONNX Runtime shared library.
ARG ORT_VERSION=1.19.2
ARG TARGETARCH=amd64

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
         -O /tmp/ort.tgz && \
    mkdir -p /opt/onnxruntime && \
    tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 && \
    rm /tmp/ort.tgz && \
    cp /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/ && \
    ldconfig && \
    apt-get purge -y wget && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY go.mod go.sum ./
COPY vendor/ vendor/
COPY . .

# Build the CLI binary.
RUN CGO_ENABLED=1 go build -o /ppocr ./cmd/ppocr

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM ubuntu:24.04

# Install only the runtime libraries needed by OpenCV and ONNX Runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy OpenCV shared libs from the build stage.
COPY --from=builder /usr/local/lib/libopencv* /usr/local/lib/
COPY --from=builder /usr/local/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

# Copy the built binary.
COPY --from=builder /ppocr /usr/local/bin/ppocr

WORKDIR /app

# Models and images are expected to be mounted at runtime:
#   docker run -v ./models:/app/models -v ./images:/app/images ppocr -image /app/images/test.jpg
ENTRYPOINT ["ppocr"]
CMD ["-help"]
