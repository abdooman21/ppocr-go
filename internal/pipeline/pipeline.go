// Package pipeline wires preprocess → detect → classify → recognize
// into one call per image: detect boxes, sort into reading order,
// crop/warp each, classify + correct orientation, recognize (batched),
// assemble the result.
package pipeline

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"

	"github.com/abdooman21/ppocr-go/internal/detect"
	"github.com/abdooman21/ppocr-go/internal/onnxrt"
	"github.com/abdooman21/ppocr-go/internal/preprocess"
	"github.com/abdooman21/ppocr-go/internal/recognize"
)

// Line is one recognized, positioned line of text.
type Line struct {
	Quad       [4][2]int
	Text       string
	Confidence float64
}

// Config holds the loaded models and thresholds a Pipeline needs.
type Config struct {
	// Models — all three loaded via onnxrt.Load before calling New.
	DetModel *onnxrt.Model
	ClsModel *onnxrt.Model // nil to skip orientation classification
	RecModel *onnxrt.Model

	// Dictionary for CTC decode (loaded via recognize.LoadDictionary).
	Dict recognize.Dictionary

	// Detection params — zero values get PaddleOCR v6 defaults.
	LimitType    preprocess.LimitType // "max" or "min" (default "min" for v6)
	LimitSideLen int                  // default 64 for v6, 960 for v3/v4/v5
	MaxSideLimit int                  // default 4000 for v6 (ignored for "max" mode)
	Thresh       float64              // prob map binarization, default 0.3
	BoxThresh    float64              // min box score, default 0.6
	UnclipRatio  float64              // DB box expansion, default 1.5

	// Recognition params.
	RecImgH int // model input height, default 48
	RecMaxW int // max padded width, default 320

	// Classification params.
	ClsThresh float64 // min confidence to accept 180° rotation, default 0.9

	// LineThreshold controls reading-order sort: boxes whose Y centroids
	// differ by ≤ this many pixels are treated as the same visual line.
	// Default 10.
	LineThreshold int
}

// Pipeline runs the full detect → classify → recognize flow on one image.
type Pipeline struct {
	cfg Config
}

// New builds a Pipeline from a loaded Config. Applies defaults for any
// zero-valued threshold/size fields.
func New(cfg Config) (*Pipeline, error) {
	if cfg.DetModel == nil {
		return nil, fmt.Errorf("pipeline.New: DetModel is required")
	}
	if cfg.RecModel == nil {
		return nil, fmt.Errorf("pipeline.New: RecModel is required")
	}
	if len(cfg.Dict) == 0 {
		return nil, fmt.Errorf("pipeline.New: Dict is required (use recognize.LoadDictionary)")
	}

	// Apply PaddleOCR v6 defaults.
	if cfg.LimitType == "" {
		cfg.LimitType = preprocess.LimitMin
	}
	if cfg.LimitSideLen <= 0 {
		cfg.LimitSideLen = 64
	}
	if cfg.MaxSideLimit <= 0 {
		cfg.MaxSideLimit = 4000
	}
	if cfg.Thresh <= 0 {
		cfg.Thresh = 0.3
	}
	if cfg.BoxThresh <= 0 {
		cfg.BoxThresh = 0.6
	}
	if cfg.UnclipRatio <= 0 {
		cfg.UnclipRatio = 1.5
	}
	if cfg.RecImgH <= 0 {
		cfg.RecImgH = 48
	}
	if cfg.RecMaxW <= 0 {
		cfg.RecMaxW = 320
	}
	if cfg.ClsThresh <= 0 {
		cfg.ClsThresh = recognize.DefaultClsThresh
	}
	if cfg.LineThreshold <= 0 {
		cfg.LineThreshold = 10
	}

	return &Pipeline{cfg: cfg}, nil
}

// Run executes the full pipeline on one image:
//
//  1. Preprocess + run detection model → probability map.
//  2. Postprocess probability map → bounding boxes.
//  3. Sort boxes into reading order.
//  4. For each box: crop → (optionally) classify orientation → recognize.
//  5. Assemble and return results.
func (p *Pipeline) Run(img gocv.Mat) ([]Line, error) {
	// ── Step 1: Detection preprocessing ──────────────────────────────
	resized, ratioH, ratioW := preprocess.ResizeForDetection(
		img, p.cfg.LimitSideLen, p.cfg.LimitType, p.cfg.MaxSideLimit,
	)
	defer resized.Close()

	detInput := preprocess.NormalizeDetInput(resized)
	detH, detW := resized.Rows(), resized.Cols()
	detShape := []int64{1, 3, int64(detH), int64(detW)}

	// ── Step 2: Run detection model ──────────────────────────────────
	probMap, outShape, err := p.cfg.DetModel.Run(detInput, detShape)
	if err != nil {
		return nil, fmt.Errorf("pipeline: detection inference: %w", err)
	}

	// The det model outputs [1, 1, H, W]. Extract mapH/mapW from shape.
	mapH, mapW := int(outShape[2]), int(outShape[3])

	// ── Step 3: Detection postprocessing ─────────────────────────────
	boxes, err := detect.PostProcess(
		probMap, mapH, mapW,
		ratioH, ratioW,
		p.cfg.Thresh, p.cfg.BoxThresh, p.cfg.UnclipRatio,
	)
	if err != nil {
		return nil, fmt.Errorf("pipeline: detection postprocess: %w", err)
	}
	if len(boxes) == 0 {
		return nil, nil
	}

	// ── Step 4: Sort into reading order ──────────────────────────────
	sortReadingOrder(boxes, p.cfg.LineThreshold)

	// ── Step 5: Crop → Classify → Recognize each box ────────────────
	lines := make([]Line, 0, len(boxes))
	for _, box := range boxes {
		line, err := p.processBox(img, box)
		if err != nil {
			// Skip boxes that fail (e.g. degenerate crops) rather than
			// aborting the entire image.
			continue
		}
		if line.Text == "" {
			continue
		}
		lines = append(lines, line)
	}

	return lines, nil
}

// processBox handles a single detected box: crop, classify, recognize.
func (p *Pipeline) processBox(img gocv.Mat, box detect.Box) (Line, error) {
	// Convert detect.Box quad (image.Point) to the format RotateCrop expects.
	crop := preprocess.RotateCrop(img, box.Quad)
	defer crop.Close()

	if crop.Empty() {
		return Line{}, fmt.Errorf("empty crop")
	}

	// ── Optional: orientation classification ─────────────────────────
	if p.cfg.ClsModel != nil {
		clsData := prepareClsInput(crop)
		clsShape := []int64{1, 3, int64(clsInputH), int64(clsInputW)}

		clsOut, _, err := p.cfg.ClsModel.Run(clsData, clsShape)
		if err == nil {
			rotate, _ := recognize.ClassifyOrientation(clsOut, p.cfg.ClsThresh)
			if rotate {
				rotated := gocv.NewMat()
				gocv.Rotate(crop, &rotated, gocv.Rotate180Clockwise)
				crop.Close()
				crop = rotated
			}
		}
		// If cls inference fails, just proceed without rotation — it's
		// better to try recognition on a possibly-flipped crop than to
		// skip the box entirely.
	}

	// ── Recognition ──────────────────────────────────────────────────
	recData := preprocess.NormalizeRecInput(crop, p.cfg.RecImgH, p.cfg.RecMaxW)
	recShape := []int64{1, 3, int64(p.cfg.RecImgH), int64(p.cfg.RecMaxW)}

	recOut, recOutShape, err := p.cfg.RecModel.Run(recData, recShape)
	if err != nil {
		return Line{}, fmt.Errorf("recognition inference: %w", err)
	}

	// Rec model outputs [1, timesteps, numClasses].
	timesteps := int(recOutShape[1])
	numClasses := int(recOutShape[2])

	result, err := recognize.Decode(recOut, timesteps, numClasses, p.cfg.Dict)
	if err != nil {
		return Line{}, fmt.Errorf("CTC decode: %w", err)
	}

	// ── Assemble Line ────────────────────────────────────────────────
	quad := boxQuadToArray(box.Quad)
	return Line{
		Quad:       quad,
		Text:       result.Text,
		Confidence: result.Confidence,
	}, nil
}

// boxQuadToArray converts detect.Box's [4]image.Point to [4][2]int for the
// public Line type.
func boxQuadToArray(q [4]image.Point) [4][2]int {
	return [4][2]int{
		{q[0].X, q[0].Y},
		{q[1].X, q[1].Y},
		{q[2].X, q[2].Y},
		{q[3].X, q[3].Y},
	}
}
