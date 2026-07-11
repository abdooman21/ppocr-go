// go-ppocr runs PaddleOCR's PP-OCRv5/v6 detection + recognition models
// natively in Go, using onnxruntime_go for inference and gocv for image
// processing — no Python, no compiled Paddle C++ library required.
//
// Public API surface is intentionally thin and NOT YET STABLE (Phase 6).
// Everything currently routes through internal/pipeline, which is itself
// still a stub pending Phases 3-5. See ROADMAP.md for full status.
package ppocr

import "gocv.io/x/gocv"

// Config configures an Engine: model file paths, dictionary path, and
// detection/recognition thresholds.
type Config struct {
	DetModelPath string
	RecModelPath string
	ClsModelPath string
	DictPath     string
	// TODO(phase 6): thresholds, execution provider (CPU/CUDA/CoreML),
	// batch size, functional options
}

// Line is one recognized, positioned line of text.
type Line struct {
	Box        [4][2]int
	Text       string
	Confidence float64
}

// Engine is the top-level entry point: load once, call Recognize per image.
type Engine struct {
	// TODO(phase 6): wraps internal/pipeline.Pipeline
}

// NewEngine loads all three models and prepares an Engine for use.
func NewEngine(cfg Config) (*Engine, error) {
	// TODO(phase 6)
	panic("ppocr.NewEngine: not implemented yet — see ROADMAP.md Phase 6")
}

// Recognize runs the full OCR pipeline on one image and returns
// recognized, positioned lines of text.
func (e *Engine) Recognize(img gocv.Mat) ([]Line, error) {
	// TODO(phase 6)
	panic("Engine.Recognize: not implemented yet — see ROADMAP.md Phase 6")
}
