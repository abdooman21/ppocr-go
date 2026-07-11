// Package pipeline wires preprocess -> detect -> classify -> recognize
// into one call per image: detect boxes, sort into reading order,
// crop/warp each, classify + correct orientation, recognize (batched),
// assemble the result.
//
// NOT YET IMPLEMENTED — this is Phase 5, and depends on Phases 3 and 4
// (internal/detect, internal/recognize are still stubs). Stubbed here so
// ppocr.go has a concrete thing to call.
package pipeline

import "gocv.io/x/gocv"

// Line is one recognized, positioned line of text.
type Line struct {
	Quad       [4][2]int
	Text       string
	Confidence float64
}

// Config holds the loaded models and thresholds a Pipeline needs.
type Config struct {
	// TODO(phase 5): *onnxrt.Model for det/cls/rec, recognize.Dictionary,
	// detection thresholds (thresh/box_thresh/unclip_ratio), rec image height
}

// Pipeline runs the full detect -> classify -> recognize flow on one image.
type Pipeline struct {
	cfg Config
}

// New builds a Pipeline from a loaded Config.
func New(cfg Config) (*Pipeline, error) {
	// TODO(phase 5)
	panic("pipeline.New: not implemented yet — see ROADMAP.md Phase 5")
}

// Run executes the full pipeline on one image.
func (p *Pipeline) Run(img gocv.Mat) ([]Line, error) {
	// TODO(phase 5): detect -> sort reading order -> crop -> classify ->
	//   recognize (batched) -> assemble
	panic("pipeline.Run: not implemented yet — see ROADMAP.md Phase 5")
}
