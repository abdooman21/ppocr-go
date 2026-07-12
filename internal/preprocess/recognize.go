package preprocess

import (
	"image"
	"math"

	"gocv.io/x/gocv"
)

// NormalizeRecInput resizes a rectified text crop to a fixed height,
// keeping aspect ratio, then zero-pads the width up to maxW so a batch of
// crops can share one tensor. Normalization is scale=1/255, mean=0.5,
// std=0.5 (range [-1,1]) — deliberately different constants from
// detection, not a typo.
func NormalizeRecInput(crop gocv.Mat, imgH, maxW int) []float32 {
	h, w := crop.Rows(), crop.Cols()
	resizedW := minInt(int(math.Ceil(float64(imgH)*float64(w)/float64(h))), maxW)

	resized := gocv.NewMat()
	gocv.Resize(crop, &resized, image.Pt(resizedW, imgH), 0, 0, gocv.InterpolationLinear)
	defer resized.Close()

	f32 := gocv.NewMat()
	resized.ConvertToWithParams(&f32, gocv.MatTypeCV32FC3, 1.0/255.0, 0)
	defer f32.Close()
	channels := gocv.Split(f32)
	defer func() {
		for _, c := range channels {
			c.Close()
		}
	}()

	for c := 0; c < 3; c++ {
		channels[c].SubtractFloat(0.5)
		channels[c].DivideFloat(0.5)
	}

	out := make([]float32, 3*imgH*maxW) // zero-initialized — that zero IS the padding
	for c := 0; c < 3; c++ {
		data, _ := channels[c].DataPtrFloat32()
		base := c * imgH * maxW
		for row := 0; row < imgH; row++ {
			copy(out[base+row*maxW:base+row*maxW+resizedW], data[row*resizedW:(row+1)*resizedW])
		}
	}
	return out
}
