package pipeline

import (
	"image"

	"gocv.io/x/gocv"
)

// clsInputH and clsInputW are the fixed input dimensions for the
// PaddleOCR angle-classifier model (TextLineOrientation).
const (
	clsInputH = 48
	clsInputW = 192
)

// prepareClsInput resizes a cropped text-line image to the classifier's
// fixed input size and normalizes it (scale=1/255, mean=0.5, std=0.5 —
// same constants as the recognition model, range [-1, 1]).
//
// Returns a flat NCHW float32 slice with shape [1, 3, clsInputH, clsInputW].
func prepareClsInput(crop gocv.Mat) []float32 {
	var resized gocv.Mat
	gocv.Resize(crop, &resized, image.Pt(clsInputW, clsInputH), 0, 0, gocv.InterpolationLinear)
	defer resized.Close()

	var f32 gocv.Mat
	resized.ConvertToWithParams(&f32, gocv.MatTypeCV32FC3, 1.0/255.0, 0)
	defer f32.Close()
	f32.SubtractFloat(0.5)
	f32.DivideFloat(0.5)

	channels := gocv.Split(f32)
	defer func() {
		for _, c := range channels {
			c.Close()
		}
	}()

	h, w := clsInputH, clsInputW
	out := make([]float32, 3*h*w)
	for c := 0; c < 3; c++ {
		data, _ := channels[c].DataPtrFloat32()
		copy(out[c*h*w:(c+1)*h*w], data)
	}
	return out
}
