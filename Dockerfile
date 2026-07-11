# syntax=docker/dockerfile:1

# ── Build stage ───────────────────────────────────────────────────────────
# Use the official OpenCV 4.12.0 image from the GoCV project.
FROM ghcr.io/hybridgroup/opencv:4.12.0 AS builder

# Install Go
# Install Go 1.25.4 directly (apt-get installs an older version)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://go.dev/dl/go1.25.4.linux-amd64.tar.gz" -O /tmp/go.tar.gz && \
    rm -rf /usr/local/go && \
    tar -C /usr/local -xzf /tmp/go.tar.gz && \
    rm /tmp/go.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Install ONNX Runtime shared library.
ARG ORT_VERSION=1.19.2
ARG TARGETARCH=amd64

RUN wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
         -O /tmp/ort.tgz && \
    mkdir -p /opt/onnxruntime && \
    tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 && \
    rm /tmp/ort.tgz && \
    cp /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/ && \
    ldconfig

WORKDIR /app
COPY go.mod go.sum ./
COPY vendor/ vendor/
COPY . .

# Build the CLI binary.
# Downgrade onnxruntime_go to match our C library version (1.19.2 / API 19)
RUN go get github.com/yalue/onnxruntime_go@v1.16.0 && go mod tidy
RUN CGO_ENABLED=1 go build -o /ppocr ./cmd/ppocr

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM ghcr.io/hybridgroup/opencv:4.12.0

# Copy ONNX Runtime shared libs from the build stage.
COPY --from=builder /usr/local/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

# Copy the built binary.
COPY --from=builder /ppocr /usr/local/bin/ppocr

WORKDIR /app

# Models and images are expected to be mounted at runtime:
#   docker run -v ./models:/app/models -v ./images:/app/images ppocr -image /app/images/test.jpg
ENTRYPOINT ["ppocr"]
CMD ["-help"]
