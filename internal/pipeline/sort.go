package pipeline

import (
	"image"
	"sort"

	"github.com/abdooman21/ppocr-go/internal/detect"
)

// sortReadingOrder sorts detected boxes into natural reading order:
// top-to-bottom by centroid Y, then left-to-right by centroid X for boxes
// on roughly the same horizontal line.
//
// Two boxes are considered "on the same line" if their Y centroids are
// within lineThreshold pixels of each other. This prevents a slight vertical
// offset from splitting a visual line into two.
func sortReadingOrder(boxes []detect.Box, lineThreshold int) {
	centroid := func(q [4]image.Point) (int, int) {
		cx := (q[0].X + q[1].X + q[2].X + q[3].X) / 4
		cy := (q[0].Y + q[1].Y + q[2].Y + q[3].Y) / 4
		return cx, cy
	}

	sort.SliceStable(boxes, func(i, j int) bool {
		_, yi := centroid(boxes[i].Quad)
		_, yj := centroid(boxes[j].Quad)

		// If the Y centroids are close enough, treat them as same line
		// and sort left-to-right instead.
		if abs(yi-yj) <= lineThreshold {
			xi, _ := centroid(boxes[i].Quad)
			xj, _ := centroid(boxes[j].Quad)
			return xi < xj
		}
		return yi < yj
	})
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
