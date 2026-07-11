// Package detect turns the DB detection model's raw probability-map output
// into a list of text-region quadrilaterals: threshold, contour extraction
// (gocv), min-area-rect fitting, "unclip" expansion, and box
// scoring/filtering against box_thresh.
package detect

import (
	"errors"
	"image"
	"math"
	"sort"

	"gocv.io/x/gocv"
)

// Box is one detected text region, in the original (pre-resize) image's
// coordinate space.
type Box struct {
	Quad  [4]image.Point
	Score float64
}

// PostProcess turns a detection model's raw output tensor into boxes.
// probMap is the model's single-channel output; ratioH/ratioW are the
// resize ratios from preprocess.ResizeForDetection, used to map coordinates
// back to the original image. thresh/boxThresh/unclipRatio default to
// 0.3/0.6/1.5 for PP-OCRv6.
func PostProcess(probMap []float32, mapH, mapW int, ratioH, ratioW, thresh, boxThresh, unclipRatio float64) ([]Box, error) {
	if mapH <= 0 || mapW <= 0 || len(probMap) != mapH*mapW {
		return nil, errors.New("detect.PostProcess: invalid probability map shape")
	}
	if ratioH <= 0 || ratioW <= 0 {
		return nil, errors.New("detect.PostProcess: resize ratios must be positive")
	}
	if thresh <= 0 {
		thresh = 0.3
	}
	if boxThresh <= 0 {
		boxThresh = 0.6
	}
	if unclipRatio <= 0 {
		unclipRatio = 1.5
	}

	maskData := make([]byte, len(probMap))
	for i, p := range probMap {
		if float64(p) > thresh {
			maskData[i] = 255
		}
	}

	mask, err := gocv.NewMatFromBytes(mapH, mapW, gocv.MatTypeCV8UC1, maskData)
	if err != nil {
		return nil, err
	}
	defer mask.Close()

	contours := gocv.FindContours(mask, gocv.RetrievalList, gocv.ChainApproxSimple)
	defer contours.Close()

	boxes := make([]Box, 0, contours.Size())
	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		if contour.Size() < 3 {
			continue
		}

		rect := gocv.MinAreaRect(contour)
		if rect.Width < 3 || rect.Height < 3 || len(rect.Points) != 4 {
			continue
		}

		quad := orderQuad(rect.Points)
		score := boxScore(probMap, mapH, mapW, quad)
		if score < boxThresh {
			continue
		}

		quad = unclipQuad(quad, unclipRatio)
		for j := range quad {
			quad[j].X = int(math.Round(float64(quad[j].X) / ratioW))
			quad[j].Y = int(math.Round(float64(quad[j].Y) / ratioH))
		}

		boxes = append(boxes, Box{Quad: quad, Score: score})
	}

	return boxes, nil
}

func orderQuad(points []image.Point) [4]image.Point {
	pts := append([]image.Point(nil), points...)
	cx, cy := 0.0, 0.0
	for _, p := range pts {
		cx += float64(p.X)
		cy += float64(p.Y)
	}
	cx /= float64(len(pts))
	cy /= float64(len(pts))

	sort.Slice(pts, func(i, j int) bool {
		return math.Atan2(float64(pts[i].Y)-cy, float64(pts[i].X)-cx) < math.Atan2(float64(pts[j].Y)-cy, float64(pts[j].X)-cx)
	})

	start := 0
	for i := 1; i < len(pts); i++ {
		if pts[i].X+pts[i].Y < pts[start].X+pts[start].Y {
			start = i
		}
	}

	return [4]image.Point{pts[start], pts[(start+1)&3], pts[(start+2)&3], pts[(start+3)&3]}
}

func boxScore(probMap []float32, h, w int, quad [4]image.Point) float64 {
	minX, maxX := clampRange(min4(quad[0].X, quad[1].X, quad[2].X, quad[3].X), max4(quad[0].X, quad[1].X, quad[2].X, quad[3].X), w)
	minY, maxY := clampRange(min4(quad[0].Y, quad[1].Y, quad[2].Y, quad[3].Y), max4(quad[0].Y, quad[1].Y, quad[2].Y, quad[3].Y), h)
	if minX > maxX || minY > maxY {
		return 0
	}

	var sum float64
	count := 0
	for y := minY; y <= maxY; y++ {
		row := y * w
		for x := minX; x <= maxX; x++ {
			if pointInQuad(float64(x)+0.5, float64(y)+0.5, quad) {
				sum += float64(probMap[row+x])
				count++
			}
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func unclipQuad(quad [4]image.Point, ratio float64) [4]image.Point {
	area := polygonArea(quad)
	perimeter := polygonPerimeter(quad)
	if area == 0 || perimeter == 0 {
		return quad
	}

	cx := float64(quad[0].X+quad[1].X+quad[2].X+quad[3].X) / 4
	cy := float64(quad[0].Y+quad[1].Y+quad[2].Y+quad[3].Y) / 4
	dist := area * ratio / perimeter

	for i, p := range quad {
		dx, dy := float64(p.X)-cx, float64(p.Y)-cy
		r := math.Hypot(dx, dy)
		if r == 0 {
			continue
		}
		scale := (r + dist) / r
		quad[i] = image.Pt(int(math.Round(cx+dx*scale)), int(math.Round(cy+dy*scale)))
	}
	return quad
}

func pointInQuad(x, y float64, quad [4]image.Point) bool {
	var pos, neg bool
	for i := 0; i < 4; i++ {
		a, b := quad[i], quad[(i+1)&3]
		cross := (float64(b.X-a.X) * (y - float64(a.Y))) - (float64(b.Y-a.Y) * (x - float64(a.X)))
		if cross < 0 {
			neg = true
		} else if cross > 0 {
			pos = true
		}
		if pos && neg {
			return false
		}
	}
	return true
}

func polygonArea(quad [4]image.Point) float64 {
	sum := 0
	for i := 0; i < 4; i++ {
		a, b := quad[i], quad[(i+1)&3]
		sum += a.X*b.Y - b.X*a.Y
	}
	return math.Abs(float64(sum)) * 0.5
}

func polygonPerimeter(quad [4]image.Point) float64 {
	var sum float64
	for i := 0; i < 4; i++ {
		a, b := quad[i], quad[(i+1)&3]
		sum += math.Hypot(float64(a.X-b.X), float64(a.Y-b.Y))
	}
	return sum
}

func clampRange(minVal, maxVal, limit int) (int, int) {
	if minVal < 0 {
		minVal = 0
	}
	if maxVal >= limit {
		maxVal = limit - 1
	}
	return minVal, maxVal
}

func min4(a, b, c, d int) int {
	return min(min(a, b), min(c, d))
}

func max4(a, b, c, d int) int {
	return max(max(a, b), max(c, d))
}
