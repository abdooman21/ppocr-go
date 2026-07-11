// Command ppocr is a minimal CLI wrapping the go-ppocr engine — useful as
// a smoke test and a usage example once Phase 6 lands.
package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	imgPath := flag.String("image", "", "path to an image file to OCR")
	detModel := flag.String("det", "models/det_onnx/inference.onnx", "detection model path")
	recModel := flag.String("rec", "models/rec_onnx/inference.onnx", "recognition model path")
	flag.Parse()

	if *imgPath == "" {
		fmt.Fprintln(os.Stderr, "usage: ppocr -image path/to/file.jpg")
		os.Exit(1)
	}

	fmt.Printf("would OCR %s using det=%s rec=%s\n", *imgPath, *detModel, *recModel)
	// TODO(phase 6): wire up ppocr.NewEngine + Engine.Recognize once
	// Phases 3-5 are implemented.
}
