package guetzli_patapon

import (
	"fmt"
	"os"
)

func GUETZLI_LOG_QUANT(stats *ProcessStats, q [][kDCTBlockSize]int) {
	print := func(format string, x ...interface{}) {
		fmt.Fprintf(os.Stderr, format, x...)
	}

	for y := 0; y < 8; y++ {
		for c := 0; c < 3; c++ {
			for x := 0; x < 8; x++ {
				print(" %2d", (q)[c][8*y+x])
			}
			print("   ")
		}
		print("\n")
	}
}
