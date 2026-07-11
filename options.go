package ppocr

// config holds all user-facing configuration values. Zero values mean
// "use PaddleOCR v6 defaults" for all thresholds and sizes.
type config struct {
	// Paths.
	ortLibPath   string
	detModelPath string
	clsModelPath string
	recModelPath string
	dictPath     string

	// ONNX I/O tensor names — these vary by model export but are stable
	// across the PP-OCRv5/v6 family. Overridable for custom models.
	detInputName  string
	detOutputName string
	clsInputName  string
	clsOutputName string
	recInputName  string
	recOutputName string

	// Detection.
	limitType    string
	limitSideLen int
	maxSideLimit int
	thresh       float64
	boxThresh    float64
	unclipRatio  float64

	// Recognition.
	recImgH int
	recMaxW int

	// Classification.
	clsThresh float64

	// Pipeline.
	lineThreshold int
}

// Option configures an [Engine] via [NewEngine].
type Option func(*config)

// ── Required options ─────────────────────────────────────────────────────

// WithDetModel sets the path to the detection ONNX model file.
func WithDetModel(path string) Option {
	return func(c *config) { c.detModelPath = path }
}

// WithRecModel sets the path to the recognition ONNX model file.
func WithRecModel(path string) Option {
	return func(c *config) { c.recModelPath = path }
}

// WithDict sets the path to the PaddleOCR dictionary text file (one
// character per line, no blank token).
func WithDict(path string) Option {
	return func(c *config) { c.dictPath = path }
}

// ── Optional options ─────────────────────────────────────────────────────

// WithClsModel sets the path to the orientation-classifier ONNX model.
// If not set, orientation classification is skipped.
func WithClsModel(path string) Option {
	return func(c *config) { c.clsModelPath = path }
}

// WithOrtLib sets the path to the onnxruntime shared library
// (onnxruntime.dll / libonnxruntime.so / libonnxruntime.dylib). If not
// set, the platform default search path is used.
func WithOrtLib(path string) Option {
	return func(c *config) { c.ortLibPath = path }
}

// WithDetIONames overrides the default detection model input/output tensor
// names ("x" / "sigmoid_0.tmp_0"). Only needed if your ONNX export uses
// different names.
func WithDetIONames(input, output string) Option {
	return func(c *config) {
		c.detInputName = input
		c.detOutputName = output
	}
}

// WithClsIONames overrides the default classifier model input/output
// tensor names ("x" / "softmax_0.tmp_0").
func WithClsIONames(input, output string) Option {
	return func(c *config) {
		c.clsInputName = input
		c.clsOutputName = output
	}
}

// WithRecIONames overrides the default recognition model input/output
// tensor names ("x" / "softmax_0.tmp_0").
func WithRecIONames(input, output string) Option {
	return func(c *config) {
		c.recInputName = input
		c.recOutputName = output
	}
}

// WithLimitType sets the detection resize strategy: "max" (shrink large
// images, PP-OCRv3/v4/v5 default) or "min" (upscale tiny images, PP-OCRv6
// default). Default: "min".
func WithLimitType(t string) Option {
	return func(c *config) { c.limitType = t }
}

// WithLimitSideLen sets the detection resize limit side length. Default 64
// for v6, use 960 for v3/v4/v5.
func WithLimitSideLen(n int) Option {
	return func(c *config) { c.limitSideLen = n }
}

// WithMaxSideLimit sets the maximum side length after detection resize.
// Only applies when LimitType is "min". Default 4000.
func WithMaxSideLimit(n int) Option {
	return func(c *config) { c.maxSideLimit = n }
}

// WithThresh sets the DB probability map binarization threshold.
// Default 0.3.
func WithThresh(v float64) Option {
	return func(c *config) { c.thresh = v }
}

// WithBoxThresh sets the minimum box score to keep a detection.
// Default 0.6.
func WithBoxThresh(v float64) Option {
	return func(c *config) { c.boxThresh = v }
}

// WithUnclipRatio sets the DB box expansion ratio. Default 1.5.
func WithUnclipRatio(v float64) Option {
	return func(c *config) { c.unclipRatio = v }
}

// WithRecImgH sets the recognition model input height in pixels.
// Default 48.
func WithRecImgH(h int) Option {
	return func(c *config) { c.recImgH = h }
}

// WithRecMaxW sets the maximum padded width for recognition input.
// Default 320.
func WithRecMaxW(w int) Option {
	return func(c *config) { c.recMaxW = w }
}

// WithClsThresh sets the minimum confidence to accept a 180° rotation
// from the orientation classifier. Default 0.9.
func WithClsThresh(v float64) Option {
	return func(c *config) { c.clsThresh = v }
}

// WithLineThreshold sets the reading-order sort line grouping threshold
// in pixels: boxes whose Y centroids differ by ≤ this value are treated
// as the same visual line. Default 10.
func WithLineThreshold(px int) Option {
	return func(c *config) { c.lineThreshold = px }
}
