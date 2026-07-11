# go-ppocr

Native Go runtime for PaddleOCR's PP-OCRv5/v6 models — detection, text-line
orientation classification, and recognition — via `onnxruntime_go` for
inference and `gocv` for image processing. No Python, no compiled Paddle
C++ inference library.

**Status: early scaffold.** Preprocessing (resize/normalize/crop) and the
ONNX runtime wrapper are implemented. Detection postprocessing, CTC
decoding, and pipeline orchestration are stubbed (they `panic` if called).
See `ROADMAP.md` for the full phase-by-phase plan and current status —
that file is the source of truth, update it as phases complete.

## Setup

    go get github.com/yalue/onnxruntime_go@latest
    go get gocv.io/x/gocv@latest

You'll also need the `onnxruntime` and OpenCV (4.12.0) shared libraries
installed on the system running this — see the deployment note in
ROADMAP.md.

## Getting models

See "Model acquisition" in `ROADMAP.md` — pull PP-OCRv6 models via the
`paddleocr`/`paddlex` Python CLI, then convert to ONNX with `paddle2onnx`.
Drop the resulting files under `models/` (already gitignored).

## Layout

    internal/onnxrt/     generic ONNX Runtime session wrapper       (Phase 1 — done)
    internal/preprocess/ resize/normalize/crop, PaddleOCR-exact     (Phase 2 — done)
    internal/detect/     DB detection postprocessing                (Phase 3 — stub)
    internal/recognize/  orientation classify + CTC decode          (Phase 4 — stub)
    internal/pipeline/   orchestrates the above end to end          (Phase 5 — stub)
    ppocr.go             public API                                 (Phase 6 — stub)
    cmd/ppocr/           example CLI                                (Phase 6 — stub)
    models/              .onnx + dictionary files go here (gitignored)
    testdata/            parity-test fixtures (Phase 2/8)

## License

Apache 2.0 — see `LICENSE`.
