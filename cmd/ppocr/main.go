// Command ppocr is a CLI for the go-ppocr engine. It reads an image,
// runs the full OCR pipeline, and prints recognized text lines to stdout.
//
// Usage:
//
//	ppocr -image photo.jpg -det models/det.onnx -rec models/rec.onnx -dict models/ppocr_keys_v1.txt
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	ppocr "github.com/abdooman21/ppocr-go"
	"gocv.io/x/gocv"
)

func main() {
	imgPath := flag.String("image", "", "path to an image file to OCR (required)")
	detModel := flag.String("det", "models/det_onnx/inference.onnx", "detection model path")
	recModel := flag.String("rec", "models/rec_onnx/inference.onnx", "recognition model path")
	clsModel := flag.String("cls", "", "orientation classifier model path (optional)")
	dictPath := flag.String("dict", "models/ppocr_keys_v1.txt", "dictionary file path")
	ortLib := flag.String("ort-lib", "", "path to onnxruntime shared library (optional, uses system default)")
	jsonOut := flag.Bool("json", false, "output results as JSON instead of plain text")

	// Tuning knobs — zero means "use default".
	limitType := flag.String("limit-type", "", `detection resize strategy: "max" or "min" (default: v6 "min")`)
	limitSideLen := flag.Int("limit-side-len", 0, "detection resize limit side length (default: 64 for v6)")
	thresh := flag.Float64("thresh", 0, "DB binarization threshold (default: 0.3)")
	boxThresh := flag.Float64("box-thresh", 0, "minimum box score (default: 0.6)")
	unclipRatio := flag.Float64("unclip-ratio", 0, "DB box expansion ratio (default: 1.5)")
	recImgH := flag.Int("rec-img-h", 0, "recognition model input height (default: 48)")

	flag.Parse()

	if *imgPath == "" {
		fmt.Fprintln(os.Stderr, "error: -image is required")
		fmt.Fprintln(os.Stderr, "usage: ppocr -image path/to/file.jpg [-det ...] [-rec ...] [-dict ...]")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Build options.
	opts := []ppocr.Option{
		ppocr.WithDetModel(*detModel),
		ppocr.WithRecModel(*recModel),
		ppocr.WithDict(*dictPath),
	}
	if *clsModel != "" {
		opts = append(opts, ppocr.WithClsModel(*clsModel))
	}
	if *ortLib != "" {
		opts = append(opts, ppocr.WithOrtLib(*ortLib))
	}
	if *limitType != "" {
		opts = append(opts, ppocr.WithLimitType(*limitType))
	}
	if *limitSideLen > 0 {
		opts = append(opts, ppocr.WithLimitSideLen(*limitSideLen))
	}
	if *thresh > 0 {
		opts = append(opts, ppocr.WithThresh(*thresh))
	}
	if *boxThresh > 0 {
		opts = append(opts, ppocr.WithBoxThresh(*boxThresh))
	}
	if *unclipRatio > 0 {
		opts = append(opts, ppocr.WithUnclipRatio(*unclipRatio))
	}
	if *recImgH > 0 {
		opts = append(opts, ppocr.WithRecImgH(*recImgH))
	}

	// Initialize engine.
	engine, err := ppocr.NewEngine(opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()

	// Read image.
	img := gocv.IMRead(*imgPath, gocv.IMReadColor)
	if img.Empty() {
		fmt.Fprintf(os.Stderr, "error: could not read image %s\n", *imgPath)
		os.Exit(1)
	}
	defer img.Close()

	// Run OCR.
	lines, err := engine.Recognize(img)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	// Output results.
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(lines); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	} else {
		if len(lines) == 0 {
			fmt.Println("(no text detected)")
			return
		}
		for i, l := range lines {
			fmt.Printf("[%d] (%.2f) %s\n", i+1, l.Confidence, l.Text)
		}
	}
}
