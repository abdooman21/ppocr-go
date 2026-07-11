package recognize

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// LoadDictionary reads a PaddleOCR dictionary file (one character per line)
// and returns a Dictionary. The file must NOT include a blank token — that's
// implicit as model output index 0 per PaddleOCR convention.
//
// Typical dictionary files have ~6,000–15,000 entries depending on the
// language pack. The returned slice's length determines the maximum valid
// model output index (index i maps to dict[i-1], so the model may output
// indices in [0, len(dict)] where 0 is blank).
func LoadDictionary(path string) (Dictionary, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("recognize.LoadDictionary: %w", err)
	}
	defer f.Close()

	var dict Dictionary
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		// Strip UTF-8 BOM if present (common in Windows-edited files).
		line = strings.TrimPrefix(line, "\xef\xbb\xbf")
		// Each line is one character/token. Don't trim further — spaces
		// and special characters are valid dictionary entries.
		if line == "" {
			continue
		}
		dict = append(dict, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("recognize.LoadDictionary: scan: %w", err)
	}
	if len(dict) == 0 {
		return nil, fmt.Errorf("recognize.LoadDictionary: %s is empty or has no valid entries", path)
	}
	return dict, nil
}
