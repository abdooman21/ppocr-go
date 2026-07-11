// Package onnxrt is a thin, PaddleOCR-agnostic wrapper around
// onnxruntime_go. It knows how to load a single ONNX graph and run tensors
// through it — nothing about detection, recognition, or PaddleOCR at all.
// That logic lives one layer up, in internal/preprocess, internal/detect,
// and internal/recognize. See ROADMAP.md, Phase 1.
package onnxrt

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var initOnce sync.Once
var initErr error

// Init loads the onnxruntime shared library and creates the global
// environment. Call exactly once at process startup, before any Load().
// sharedLibPath may be "" to use the platform default (onnxruntime.so /
// onnxruntime.dll), but setting it explicitly is more predictable in Docker.
func Init(sharedLibPath string) error {
	initOnce.Do(func() {
		if sharedLibPath != "" {
			ort.SetSharedLibraryPath(sharedLibPath)
		}
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

// Shutdown tears down the global onnxruntime environment. Call once at
// process exit.
func Shutdown() error {
	return ort.DestroyEnvironment()
}

// Model wraps exactly one ONNX graph — a detector, a classifier, or a
// recognizer.
type Model struct {
	session    *ort.DynamicAdvancedSession
	inputName  string
	outputName string
}

// Load opens a single-input, single-output ONNX model. All three PaddleOCR
// stages (det/cls/rec) fit this shape. inputName/outputName come from
// inspecting the model — either via Python (`onnx` package: sess.get_inputs())
// or onnxruntime_go's own GetInputOutputInfo(path) helper.
func Load(modelPath, inputName, outputName string) (*Model, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{inputName},
		[]string{outputName},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", modelPath, err)
	}
	return &Model{session: session, inputName: inputName, outputName: outputName}, nil
}

// Close releases the underlying onnxruntime session.
func (m *Model) Close() error {
	return m.session.Destroy()
}

// Run executes the graph on one NCHW float32 tensor and returns the raw
// output data plus its shape. The caller is responsible for building
// data/shape and interpreting whatever comes back.
//
// NOTE: DynamicAdvancedSession exists specifically because PaddleOCR's
// input AND output shapes both vary per image. Double-check the exact
// Run() signature against the onnxruntime_go version you pin — it has
// shifted across releases — but this overall shape (pass input tensor(s)
// in, get output tensor(s) with library-determined shape back) is stable.
func (m *Model) Run(data []float32, shape []int64) (outData []float32, outShape []int64, err error) {
	input, err := ort.NewTensor(ort.NewShape(shape...), data)
	if err != nil {
		return nil, nil, fmt.Errorf("input tensor: %w", err)
	}
	defer input.Destroy()

	outputs := []ort.Value{nil} // nil => let onnxruntime allocate the output
	if err := m.session.Run([]ort.Value{input}, outputs); err != nil {
		return nil, nil, fmt.Errorf("run: %w", err)
	}
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("unexpected output type %T", outputs[0])
	}
	defer outTensor.Destroy()

	return outTensor.GetData(), outTensor.GetShape(), nil
}
