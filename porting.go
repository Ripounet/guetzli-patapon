package guetzli_patapon

import (
	"fmt"
	"io"
	"math"
	"os"
)

func assert(b bool) {
	if !b {
		panic("Assertion failed :(")
	}
}

func memcmpInt(a, b []int, n int) bool {
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func std_min(a, b int) int {
	if b < a {
		return b
	}
	return a
}

func std_max(a, b int) int {
	if b > a {
		return b
	}
	return a
}

func std_minFloat64(a, b float64) float64 {
	if b < a {
		return b
	}
	return a
}

func std_maxFloat64(a, b float64) float64 {
	if b > a {
		return b
	}
	return a
}

func std_minFloat32(a, b float32) float32 {
	if b < a {
		return b
	}
	return a
}

func std_maxFloa32(a, b float32) float32 {
	if b > a {
		return b
	}
	return a
}

func std_minUint32(a, b uint32) uint32 {
	if b < a {
		return b
	}
	return a
}

func std_maxUint32(a, b uint32) uint32 {
	if b > a {
		return b
	}
	return a
}

func std_abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func std_round(a float64) int {
	// TODO PATAPON what about negative numbers?
	return int(a + 0.5)
}

func minusOr0(a, b int) int {
	if a < b {
		return 0
	}
	return a - b
}

func cloneMatrixFloat64(src [][]float64) (dst [][]float64) {
	dst = make([][]float64, len(src))
	for i := range src {
		dst[i] = make([]float64, len(src[i]))
		copy(dst[i], src[i])
	}
	return dst
}

func cloneMatrixFloat32(src [][]float32) (dst [][]float32) {
	dst = make([][]float32, len(src))
	for i := range src {
		dst[i] = make([]float32, len(src[i]))
		copy(dst[i], src[i])
	}
	return dst
}

func cloneSliceFloat32(src []float32) (dst []float32) {
	dst = make([]float32, len(src))
	copy(dst, src)
	return dst
}

func cloneSliceBool(src []bool) (dst []bool) {
	dst = make([]bool, len(src))
	copy(dst, src)
	return dst
}

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func one(b bool) int {
	if b {
		return 1
	}
	return 0
}

func fprintf(w io.Writer, format string, a ...interface{}) {
	_, _ = fmt.Fprintf(w, format, a...)
}

var stderr = os.Stderr

//
// TODO
// TODO
// TODO
//

var (
	ComputeBlockIDCT  func(block []coeff_t, out []byte)
	RGBToYUV420       func(rgb_in []byte, width, height int) [][]float32
	PreProcessChannel func(w, h, channel int, sigma, amount float32, blur, sharpen bool, image [][]float32) [][]float32
)
