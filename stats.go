package guetzli_patapon

import "io"

const (
	kNumItersCnt     = "number of iterations"
	kNumItersUpCnt   = "number of iterations up"
	kNumItersDownCnt = "number of iterations down"
)

type ProcessStats struct {
	counters          map[string]int
	debug_output      string
	debug_output_file io.Writer

	filename string
}
