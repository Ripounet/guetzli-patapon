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

func std_maxFloat32(a, b float32) float32 {
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

func cloneSliceByte(src []byte) (dst []byte) {
	dst = make([]byte, len(src))
	copy(dst, src)
	return dst
}

// Mimicks std::vector.resize()
func resizeSliceFloat32(a []float32, n int) []float32 {
	if len(a) >= n {
		return a[:n:n]
	}
	grown := make([]float32, n)
	copy(grown, a)
	return grown
}

// Mimicks std::vector<std::vector>.resize()
func resizeMatrixFloat32(a [][]float32, n int) [][]float32 {
	if len(a) >= n {
		return a[:n:n]
	}
	grown := make([][]float32, n)
	copy(grown, a)
	return grown
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

// tern mimicks ternary operator (b?x:y), only for int type.
func tern(b bool, x, y int) int {
	if b {
		return x
	}
	return y
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
	GUETZLI_LOG = func(x ...interface{}) { fmt.Println(x...) }
	logx        = func(x ...interface{}) { fmt.Fprintln(os.Stderr, x...) }
	logf        = func(format string, x ...interface{}) { fmt.Fprintf(os.Stderr, format, x...) }
)
