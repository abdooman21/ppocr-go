package preprocess

import (
	"image"
	"math"

	"gocv.io/x/gocv"
)

// RotateCrop reproduces PaddleOCR's get_rotate_crop_image: warps a
// detected quadrilateral to an upright rectangle via a perspective
// transform, then rotates 90° if the result comes out taller than wide
// (catches vertical text lines). Run this once per detected box, before
// recognition.
func RotateCrop(img gocv.Mat, quad [4]image.Point) gocv.Mat {
	dist := func(a, b image.Point) float64 {
		dx, dy := float64(a.X-b.X), float64(a.Y-b.Y)
		return math.Sqrt(dx*dx + dy*dy)
	}
	cropW := int(math.Max(dist(quad[0], quad[1]), dist(quad[2], quad[3])))
	cropH := int(math.Max(dist(quad[0], quad[3]), dist(quad[1], quad[2])))

	toPt2f := func(p image.Point) gocv.Point2f { return gocv.Point2f{X: float32(p.X), Y: float32(p.Y)} }
	src := gocv.NewPoint2fVectorFromPoints([]gocv.Point2f{toPt2f(quad[0]), toPt2f(quad[1]), toPt2f(quad[2]), toPt2f(quad[3])})
	dst := gocv.NewPoint2fVectorFromPoints([]gocv.Point2f{
		{X: 0, Y: 0}, {X: float32(cropW), Y: 0}, {X: float32(cropW), Y: float32(cropH)}, {X: 0, Y: float32(cropH)},
	})
	m := gocv.GetPerspectiveTransform2f(src, dst)
	defer m.Close()

	var cropped gocv.Mat
	// Python also sets flags=INTER_CUBIC, borderMode=BORDER_REPLICATE —
	// use gocv.WarpPerspectiveWithParams if you need bit-exact parity here.
	gocv.WarpPerspective(img, &cropped, m, image.Pt(cropW, cropH))

	if float64(cropped.Rows())/float64(cropped.Cols()) >= 1.5 {
		var rotated gocv.Mat
		gocv.Rotate(cropped, &rotated, gocv.Rotate90Clockwise)
		cropped.Close()
		return rotated
	}
	return cropped
}
