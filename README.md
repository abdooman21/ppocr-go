# go-ppocr

Native Go runtime for PaddleOCR's PP-OCRv5/v6 models — detection, text-line
orientation classification, and recognition — via `onnxruntime_go` for
inference and `gocv` for image processing. No Python, no compiled Paddle
C++ inference library.

## Features

- **Pure Go orchestration** — only cgo dependency is OpenCV (via gocv) and
  the onnxruntime shared library (loaded at runtime, not linked).
- **PP-OCRv6 defaults** — works out of the box with PP-OCRv6 medium models.
  PP-OCRv5 and v3/v4 models are supported by adjusting options.
- **Functional options API** — clean, extensible configuration.
- **Orientation-aware** — optional text-line orientation classifier corrects
  upside-down text before recognition.

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Go | 1.21+ | |
| OpenCV | 4.12.0 | Required by gocv at build time |
| ONNX Runtime | 1.17+ | Shared library (`.dll`/`.so`/`.dylib`) must be available at runtime |

## Installation

```bash
go get github.com/abdooman21/ppocr-go@latest
```

## Getting Models

Pull PP-OCRv6 models via the PaddleOCR Python CLI, then convert to ONNX:

```bash
pip install paddleocr paddlex
python3 -c "from paddleocr import TextDetection; TextDetection(model_name='PP-OCRv6_medium_det')"
python3 -c "from paddleocr import TextRecognition; TextRecognition(model_name='PP-OCRv6_medium_rec')"
python3 -c "from paddleocr import TextLineOrientation; TextLineOrientation()"
paddlex --install paddle2onnx
paddlex --paddle2onnx --paddle_model_dir <cached_det_dir> --onnx_model_dir ./models/det_onnx
paddlex --paddle2onnx --paddle_model_dir <cached_rec_dir> --onnx_model_dir ./models/rec_onnx
paddlex --paddle2onnx --paddle_model_dir <cached_cls_dir> --onnx_model_dir ./models/cls_onnx
```

Drop the resulting `.onnx` files and dictionary under `models/`
(already gitignored). See `ROADMAP.md` for details.

## Usage — Go Library

```go
package main

import (
    "fmt"
    "log"

    ppocr "github.com/abdooman21/ppocr-go"
    "gocv.io/x/gocv"
)

func main() {
    engine, err := ppocr.NewEngine(
        ppocr.WithDetModel("models/det_onnx/inference.onnx"),
        ppocr.WithRecModel("models/rec_onnx/inference.onnx"),
        ppocr.WithClsModel("models/cls_onnx/inference.onnx"), // optional
        ppocr.WithDict("models/ppocr_keys_v1.txt"),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer engine.Close()

    img := gocv.IMRead("document.jpg", gocv.IMReadColor)
    defer img.Close()

    lines, err := engine.Recognize(img)
    if err != nil {
        log.Fatal(err)
    }

    for _, l := range lines {
        fmt.Printf("[%.2f] %s\n", l.Confidence, l.Text)
        fmt.Printf("  box: %v\n", l.Box)
    }
}
```

## Usage — CLI

```bash
go run ./cmd/ppocr \
    -image photo.jpg \
    -det models/det_onnx/inference.onnx \
    -rec models/rec_onnx/inference.onnx \
    -dict models/ppocr_keys_v1.txt

# JSON output:
go run ./cmd/ppocr -image photo.jpg -json
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `-image` | *(required)* | Path to image file |
| `-det` | `models/det_onnx/inference.onnx` | Detection model |
| `-rec` | `models/rec_onnx/inference.onnx` | Recognition model |
| `-cls` | *(none)* | Orientation classifier (optional) |
| `-dict` | `models/ppocr_keys_v1.txt` | Dictionary file |
| `-ort-lib` | *(system default)* | Path to onnxruntime shared library |
| `-json` | `false` | Output as JSON |
| `-limit-type` | `min` (v6) | Resize strategy: `max` or `min` |
| `-limit-side-len` | `64` (v6) | Resize limit |
| `-thresh` | `0.3` | DB binarization threshold |
| `-box-thresh` | `0.6` | Minimum box score |
| `-unclip-ratio` | `1.5` | Box expansion ratio |
| `-rec-img-h` | `48` | Recognition input height |

## Configuration Options

All options have sensible PP-OCRv6 defaults. Override only what you need:

```go
engine, err := ppocr.NewEngine(
    // Required
    ppocr.WithDetModel("models/det.onnx"),
    ppocr.WithRecModel("models/rec.onnx"),
    ppocr.WithDict("models/dict.txt"),

    // Optional — override defaults
    ppocr.WithClsModel("models/cls.onnx"),         // enable orientation fix
    ppocr.WithOrtLib("/usr/local/lib/libonnxruntime.so"), // explicit lib path
    ppocr.WithLimitType("max"),                     // v3/v4/v5 resize mode
    ppocr.WithLimitSideLen(960),                    // v3/v4/v5 limit
    ppocr.WithThresh(0.3),                          // DB binarization
    ppocr.WithBoxThresh(0.6),                       // min box score
    ppocr.WithUnclipRatio(1.5),                     // box expansion
    ppocr.WithRecImgH(48),                          // rec model input height
    ppocr.WithRecMaxW(320),                         // rec max padded width
    ppocr.WithClsThresh(0.9),                       // cls confidence threshold
    ppocr.WithLineThreshold(10),                    // reading-order grouping
)
```

## Layout

```
ppocr.go             public API (Engine, Line, NewEngine, Recognize)
options.go           functional options (WithDetModel, WithRecModel, ...)
internal/onnxrt/     generic ONNX Runtime session wrapper          (Phase 1)
internal/preprocess/ resize/normalize/crop, PaddleOCR-exact        (Phase 2)
internal/detect/     DB detection postprocessing                   (Phase 3)
internal/recognize/  orientation classify + CTC decode + dict      (Phase 4)
internal/pipeline/   orchestrates the above end to end             (Phase 5)
cmd/ppocr/           example CLI                                   (Phase 6)
models/              .onnx + dictionary files (gitignored)
testdata/            parity-test fixtures
```

## Docker

A Dockerfile is provided for building and running in environments where
OpenCV and ONNX Runtime are not installed:

```bash
docker build -t ppocr .
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/images:/app/images \
    ppocr -image /app/images/test.jpg
```

## License

Apache 2.0 — see `LICENSE`.
