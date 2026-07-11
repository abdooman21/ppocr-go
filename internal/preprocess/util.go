// Package preprocess implements the exact image transforms PaddleOCR's own
// pipeline applies before handing pixels to the detection and recognition
// models: resize, normalize, and crop/rectify detected text regions. See
// ROADMAP.md, "Preprocessing reference constants", for where these values
// come from — don't re-derive them from memory.
package preprocess

import "math"

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func roundToMultipleOf32(x int) int {
	v := int(math.Round(float64(x)/32.0)) * 32
	return maxInt(v, 32)
}
