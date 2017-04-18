package guetzli_patapon

import (
	"fmt"
	"io"
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

func std_abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func cloneMatrixFloat64(src [][]float64) (dst [][]float64) {
	dst = make([][]float64, len(src))
	for i := range src {
		dst[i] = make([]float64, len(src[i]))
		copy(dst[i], src[i])
	}
	return dst
}

func fprintf(w io.Writer, format string, a ...interface{}) {
	_, _ = fmt.Fprintf(w, format, a...)
}

var stderr = os.Stderr
