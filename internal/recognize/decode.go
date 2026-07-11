// Package recognize turns the recognition model's per-timestep character
// probabilities into text, plus the small orientation classifier that runs
// on each crop first.
//
// NOT YET IMPLEMENTED — this is Phase 4.
package recognize

// Dictionary maps model output indices to characters. Must match the exact
// model/language variant loaded — a mismatch here produces confident-looking
// garbage with no error thrown.
type Dictionary []string

// Result is one recognized text line.
type Result struct {
	Text       string
	Confidence float64
}

// Decode greedily CTC-decodes a recognition model's raw output: argmax per
// timestep, collapse repeated characters, drop the blank token, map indices
// through dict.
func Decode(logits []float32, timesteps, numClasses int, dict Dictionary) (Result, error) {
	// TODO(phase 4): argmax per timestep -> collapse repeats -> drop blanks
	//   -> map indices through dict -> average per-char confidence
	panic("recognize.Decode: not implemented yet — see ROADMAP.md Phase 4")
}

// ClassifyOrientation runs the small angle-classifier model's output and
// reports whether the crop is upside down (needs a 180° rotation before
// recognition).
func ClassifyOrientation(logits []float32) (rotate180 bool, confidence float64) {
	// TODO(phase 4): softmax over the 2-class output, threshold
	panic("recognize.ClassifyOrientation: not implemented yet — see ROADMAP.md Phase 4")
}
