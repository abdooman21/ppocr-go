// go-ppocr runs PaddleOCR's PP-OCRv5/v6 detection + recognition models
// natively in Go, using onnxruntime_go for inference and gocv for image
// processing — no Python, no compiled Paddle C++ library required.
//
// Create an [Engine] with [NewEngine], then call [Engine.Recognize] on each
// image. The engine is safe for concurrent use from multiple goroutines.
//
// Basic usage:
//
//	engine, err := ppocr.NewEngine(
//	    ppocr.WithDetModel("models/det.onnx"),
//	    ppocr.WithRecModel("models/rec.onnx"),
//	    ppocr.WithDict("models/ppocr_keys_v1.txt"),
//	)
//	if err != nil { log.Fatal(err) }
//	defer engine.Close()
//
//	img := gocv.IMRead("document.jpg", gocv.IMReadColor)
//	defer img.Close()
//
//	lines, err := engine.Recognize(img)
//	for _, l := range lines {
//	    fmt.Printf("%s (%.2f)\n", l.Text, l.Confidence)
//	}
package ppocr

import (
	"fmt"

	"gocv.io/x/gocv"

	"github.com/abdooman21/ppocr-go/internal/onnxrt"
	"github.com/abdooman21/ppocr-go/internal/pipeline"
	"github.com/abdooman21/ppocr-go/internal/preprocess"
	"github.com/abdooman21/ppocr-go/internal/recognize"
)

// Line is one recognized, positioned line of text.
type Line struct {
	// Box is the four corner points of the text region in the original
	// image's coordinate space, ordered: top-left, top-right, bottom-right,
	// bottom-left.
	Box [4][2]int

	// Text is the recognized string.
	Text string

	// Confidence is the average per-character softmax probability from the
	// recognition model, in [0, 1].
	Confidence float64
}

// Engine is the top-level entry point. Create one with [NewEngine], then
// call [Recognize] per image. Safe for concurrent use — each Recognize
// call is independent.
type Engine struct {
	pipe   *pipeline.Pipeline
	detMdl *onnxrt.Model
	clsMdl *onnxrt.Model // nil if no cls model configured
	recMdl *onnxrt.Model
}

// NewEngine initializes the ONNX Runtime environment, loads the models
// specified by the given options, and returns a ready-to-use Engine.
//
// At minimum you must provide [WithDetModel], [WithRecModel], and
// [WithDict]. Use [WithClsModel] to enable orientation classification
// (recommended but optional). Use [WithOrtLib] if the onnxruntime shared
// library is not on the default search path.
//
// Call [Engine.Close] when done to release all resources.
func NewEngine(opts ...Option) (*Engine, error) {
	cfg := &config{
		// I/O names matching PP-OCRv5/v6 exported ONNX models.
		detInputName:  "x",
		detOutputName: "sigmoid_0.tmp_0",
		clsInputName:  "x",
		clsOutputName: "softmax_0.tmp_0",
		recInputName:  "x",
		recOutputName: "softmax_0.tmp_0",
	}
	for _, o := range opts {
		o(cfg)
	}

	// Validate required fields.
	if cfg.detModelPath == "" {
		return nil, fmt.Errorf("ppocr.NewEngine: detection model path is required (use WithDetModel)")
	}
	if cfg.recModelPath == "" {
		return nil, fmt.Errorf("ppocr.NewEngine: recognition model path is required (use WithRecModel)")
	}
	if cfg.dictPath == "" {
		return nil, fmt.Errorf("ppocr.NewEngine: dictionary path is required (use WithDict)")
	}

	// ── Initialize ONNX Runtime ──────────────────────────────────────
	if err := onnxrt.Init(cfg.ortLibPath); err != nil {
		return nil, fmt.Errorf("ppocr: init onnxruntime: %w", err)
	}

	// ── Load models ──────────────────────────────────────────────────
	detMdl, err := onnxrt.Load(cfg.detModelPath, cfg.detInputName, cfg.detOutputName)
	if err != nil {
		return nil, fmt.Errorf("ppocr: load detection model: %w", err)
	}

	var clsMdl *onnxrt.Model
	if cfg.clsModelPath != "" {
		m, err := onnxrt.Load(cfg.clsModelPath, cfg.clsInputName, cfg.clsOutputName)
		if err != nil {
			detMdl.Close()
			return nil, fmt.Errorf("ppocr: load classifier model: %w", err)
		}
		clsMdl = m
	}

	recMdl, err := onnxrt.Load(cfg.recModelPath, cfg.recInputName, cfg.recOutputName)
	if err != nil {
		detMdl.Close()
		if clsMdl != nil {
			clsMdl.Close()
		}
		return nil, fmt.Errorf("ppocr: load recognition model: %w", err)
	}

	// ── Load dictionary ──────────────────────────────────────────────
	dict, err := recognize.LoadDictionary(cfg.dictPath)
	if err != nil {
		detMdl.Close()
		if clsMdl != nil {
			clsMdl.Close()
		}
		recMdl.Close()
		return nil, fmt.Errorf("ppocr: load dictionary: %w", err)
	}

	// ── Build pipeline ───────────────────────────────────────────────
	pipeCfg := pipeline.Config{
		DetModel:      detMdl,
		ClsModel:      clsMdl,
		RecModel:      recMdl,
		Dict:          dict,
		LimitType:     preprocess.LimitType(cfg.limitType),
		LimitSideLen:  cfg.limitSideLen,
		MaxSideLimit:  cfg.maxSideLimit,
		Thresh:        cfg.thresh,
		BoxThresh:     cfg.boxThresh,
		UnclipRatio:   cfg.unclipRatio,
		RecImgH:       cfg.recImgH,
		RecMaxW:       cfg.recMaxW,
		ClsThresh:     cfg.clsThresh,
		LineThreshold: cfg.lineThreshold,
	}
	pipe, err := pipeline.New(pipeCfg)
	if err != nil {
		detMdl.Close()
		if clsMdl != nil {
			clsMdl.Close()
		}
		recMdl.Close()
		return nil, fmt.Errorf("ppocr: build pipeline: %w", err)
	}

	return &Engine{
		pipe:   pipe,
		detMdl: detMdl,
		clsMdl: clsMdl,
		recMdl: recMdl,
	}, nil
}

// Recognize runs the full OCR pipeline on one image and returns the
// recognized, positioned lines of text in reading order.
//
// The input must be a BGR image loaded via gocv.IMRead or equivalent.
// Do NOT convert to RGB — PaddleOCR's models expect BGR input.
func (e *Engine) Recognize(img gocv.Mat) ([]Line, error) {
	pipeLines, err := e.pipe.Run(img)
	if err != nil {
		return nil, err
	}

	lines := make([]Line, len(pipeLines))
	for i, pl := range pipeLines {
		lines[i] = Line{
			Box:        pl.Quad,
			Text:       pl.Text,
			Confidence: pl.Confidence,
		}
	}
	return lines, nil
}

// Close releases all loaded ONNX models and shuts down the runtime.
// After Close returns, the Engine must not be used.
func (e *Engine) Close() error {
	var firstErr error
	capture := func(err error) {
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}

	capture(e.detMdl.Close())
	if e.clsMdl != nil {
		capture(e.clsMdl.Close())
	}
	capture(e.recMdl.Close())
	capture(onnxrt.Shutdown())
	return firstErr
}
