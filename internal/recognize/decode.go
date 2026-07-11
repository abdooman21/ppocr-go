// Package recognize turns the recognition model's per-timestep character
// probabilities into text, plus the small orientation classifier that runs
// on each crop first.
package recognize

import (
	"errors"
	"math"
)

// Dictionary maps model output indices to characters. Must match the exact
// model/language variant loaded — a mismatch here produces confident-looking
// garbage with no error thrown.
//
// Index 0 in the model output is the CTC blank token and is NOT stored in
// this slice. Model output index i maps to dict[i-1]. This matches PaddleOCR's
// convention: the dictionary file has no blank entry.
type Dictionary []string

// Result is one recognized text line.
type Result struct {
	Text       string
	Confidence float64
}

// DefaultClsThresh is PaddleOCR's default classification threshold. Only
// rotate 180° if the classifier's confidence exceeds this.
const DefaultClsThresh = 0.9

// ClassifyOrientation interprets the 2-class angle-classifier output and
// reports whether the crop is upside down (needs a 180° rotation before
// recognition).
//
// logits is expected to be [prob_0°, prob_180°] — the raw outputs from the
// cls model. We apply softmax ourselves, so pre-softmaxed or
// post-softmaxed inputs both work (softmax is idempotent on already-
// normalized probabilities within floating-point tolerance).
func ClassifyOrientation(logits []float32, clsThresh float64) (rotate180 bool, confidence float64) {
	if len(logits) < 2 {
		return false, 0
	}
	if clsThresh <= 0 {
		clsThresh = DefaultClsThresh
	}

	// Softmax over the 2 logits.
	maxVal := logits[0]
	if logits[1] > maxVal {
		maxVal = logits[1]
	}
	exp0 := math.Exp(float64(logits[0] - maxVal))
	exp1 := math.Exp(float64(logits[1] - maxVal))
	sum := exp0 + exp1

	prob0 := exp0 / sum // probability of 0° (upright)
	prob1 := exp1 / sum // probability of 180° (upside down)

	if prob1 > prob0 && prob1 > clsThresh {
		return true, prob1
	}
	return false, prob0
}

// Decode greedily CTC-decodes a recognition model's raw output: argmax per
// timestep, collapse repeated characters, drop the blank token (index 0),
// map indices through dict, average per-character confidence.
//
// logits is the flattened [timesteps × numClasses] tensor from the rec model.
// dict must match the model — see Dictionary's doc comment for the index
// mapping convention.
func Decode(logits []float32, timesteps, numClasses int, dict Dictionary) (Result, error) {
	if timesteps <= 0 || numClasses <= 0 {
		return Result{}, errors.New("recognize.Decode: timesteps and numClasses must be positive")
	}
	if len(logits) < timesteps*numClasses {
		return Result{}, errors.New("recognize.Decode: logits length does not match timesteps × numClasses")
	}

	// 1. Argmax + softmax confidence per timestep.
	indices := make([]int, timesteps)
	confs := make([]float64, timesteps)

	for t := 0; t < timesteps; t++ {
		offset := t * numClasses
		row := logits[offset : offset+numClasses]

		// Find argmax and max value for numerically stable softmax.
		bestIdx := 0
		bestVal := row[0]
		for i := 1; i < numClasses; i++ {
			if row[i] > bestVal {
				bestVal = row[i]
				bestIdx = i
			}
		}

		// Softmax denominator (for the winning class's confidence only).
		var expSum float64
		for i := 0; i < numClasses; i++ {
			expSum += math.Exp(float64(row[i] - bestVal))
		}
		confs[t] = 1.0 / expSum // = exp(0) / sum = the argmax class probability
		indices[t] = bestIdx
	}

	// 2. Collapse consecutive duplicates and drop blank (index 0).
	var chars []string
	var charConfs []float64
	prevIdx := -1

	for t := 0; t < timesteps; t++ {
		idx := indices[t]
		if idx == prevIdx {
			prevIdx = idx
			continue
		}
		prevIdx = idx

		if idx == 0 {
			// CTC blank token — skip.
			continue
		}

		// Map model index to dictionary character.
		// Model index 0 = blank, model index i ≥ 1 → dict[i-1].
		dictIdx := idx - 1
		if dictIdx < 0 || dictIdx >= len(dict) {
			// Out-of-dict index — skip rather than crash. This can happen
			// if the model and dictionary file are mismatched.
			continue
		}
		chars = append(chars, dict[dictIdx])
		charConfs = append(charConfs, confs[t])
	}

	// 3. Assemble result.
	if len(chars) == 0 {
		return Result{Text: "", Confidence: 0}, nil
	}

	text := ""
	for _, c := range chars {
		text += c
	}

	var confSum float64
	for _, c := range charConfs {
		confSum += c
	}
	avgConf := confSum / float64(len(charConfs))

	return Result{Text: text, Confidence: avgConf}, nil
}
