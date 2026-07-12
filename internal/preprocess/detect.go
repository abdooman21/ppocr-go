package preprocess

import (
	"image"

	"gocv.io/x/gocv"
)

// LimitType controls how ResizeForDetection scales the input before it's
// fed to the DB detection model. "max" (PP-OCRv3/v4/v5's default) shrinks
// large images down to limitSideLen. "min" (PP-OCRv6's actual default,
// limitSideLen=64) only upscales images smaller than limitSideLen, capped
// by maxSideLimit. Match whichever your model version actually ships
// with — using the wrong one changes the effective resolution the model
// sees, not just performance.
type LimitType string

const (
	LimitMax LimitType = "max"
	LimitMin LimitType = "min"
)

// ResizeForDetection reproduces PaddleOCR's DetResizeForTest: it resizes
// directly to a target size that's already a multiple of 32, rather than
// resizing then padding. Returns the resized image and the height/width
// ratios needed to map detected box coordinates back to the original image.
func ResizeForDetection(img gocv.Mat, limitSideLen int, limitType LimitType, maxSideLimit int) (resized gocv.Mat, ratioH, ratioW float64) {
	h, w := img.Rows(), img.Cols()
	ratio := 1.0
	switch limitType {
	case LimitMax:
		if maxInt(h, w) > limitSideLen {
			ratio = float64(limitSideLen) / float64(maxInt(h, w))
		}
	case LimitMin:
		if minInt(h, w) < limitSideLen {
			ratio = float64(limitSideLen) / float64(minInt(h, w))
		}
	}
	resizeH := roundToMultipleOf32(int(float64(h) * ratio))
	resizeW := roundToMultipleOf32(int(float64(w) * ratio))
	if maxSideLimit > 0 {
		resizeH = minInt(resizeH, maxSideLimit)
		resizeW = minInt(resizeW, maxSideLimit)
	}
	resized = gocv.NewMat()
	gocv.Resize(img, &resized, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)
	return resized, float64(resizeH) / float64(h), float64(resizeW) / float64(w)
}

// NormalizeDetInput converts a resized BGR image into the NCHW float32
// tensor the detection model expects: scale=1/255, mean=[0.485,0.456,0.406],
// std=[0.229,0.224,0.225], applied positionally — no RGB conversion, since
// PaddleOCR never converts to RGB either.
//
// gocv.BlobFromImage is deliberately not used here: its scalefactor is a
// single scalar for all channels, but std differs per channel, so it can't
// reproduce this exactly. See ROADMAP.md, "Gotcha 2."
func NormalizeDetInput(resized gocv.Mat) []float32 {
	f32 := gocv.NewMat()
	resized.ConvertToWithParams(&f32, gocv.MatTypeCV32FC3, 1.0/255.0, 0)
	defer f32.Close()

	channels := gocv.Split(f32)
	defer func() {
		for _, c := range channels {
			c.Close()
		}
	}()

	mean := [3]float32{0.485, 0.456, 0.406}
	std := [3]float32{0.229, 0.224, 0.225}
	h, w := resized.Rows(), resized.Cols()
	out := make([]float32, 3*h*w) // NCHW: plane 0, plane 1, plane 2

	for c := 0; c < 3; c++ {
		channels[c].SubtractFloat(mean[c])
		channels[c].DivideFloat(std[c])
		data, _ := channels[c].DataPtrFloat32()
		copy(out[c*h*w:(c+1)*h*w], data)
	}
	return out
}
