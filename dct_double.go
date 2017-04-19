package guetzli_patapon

// kDCTMatrix[8*u+x] = 0.5*alpha(u)*cos((2*x+1)*u*M_PI/16),
// where alpha(0) = 1/sqrt(2) and alpha(u) = 1 for u > 0.
var kDCTMatrix = [64]float64{
	0.3535533906, 0.3535533906, 0.3535533906, 0.3535533906,
	0.3535533906, 0.3535533906, 0.3535533906, 0.3535533906,
	0.4903926402, 0.4157348062, 0.2777851165, 0.0975451610,
	-0.0975451610, -0.2777851165, -0.4157348062, -0.4903926402,
	0.4619397663, 0.1913417162, -0.1913417162, -0.4619397663,
	-0.4619397663, -0.1913417162, 0.1913417162, 0.4619397663,
	0.4157348062, -0.0975451610, -0.4903926402, -0.2777851165,
	0.2777851165, 0.4903926402, 0.0975451610, -0.4157348062,
	0.3535533906, -0.3535533906, -0.3535533906, 0.3535533906,
	0.3535533906, -0.3535533906, -0.3535533906, 0.3535533906,
	0.2777851165, -0.4903926402, 0.0975451610, 0.4157348062,
	-0.4157348062, -0.0975451610, 0.4903926402, -0.2777851165,
	0.1913417162, -0.4619397663, 0.4619397663, -0.1913417162,
	-0.1913417162, 0.4619397663, -0.4619397663, 0.1913417162,
	0.0975451610, -0.2777851165, 0.4157348062, -0.4903926402,
	0.4903926402, -0.4157348062, 0.2777851165, -0.0975451610,
}

func DCT1d(in []float64, stride int, out []float64) {
	for x := 0; x < 8; x++ {
		out[x*stride] = 0.0
		for u := 0; u < 8; u++ {
			out[x*stride] += kDCTMatrix[8*x+u] * in[u*stride]
		}
	}
}

func IDCT1d(in []float64, stride int, out []float64) {
	for x := 0; x < 8; x++ {
		out[x*stride] = 0.0
		for u := 0; u < 8; u++ {
			out[x*stride] += kDCTMatrix[8*u+x] * in[u*stride]
		}
	}
}

type Transform1d func(in []float64, stride int, out []float64)

func TransformBlock(block []float64, f Transform1d) {
	var tmp [64]float64
	for x := 0; x < 8; x++ {
		f(block[x:], 8, tmp[x:])
	}
	for y := 0; y < 8; y++ {
		f(tmp[8*y:], 1, block[8*y:])
	}
}

func ComputeBlockDCTDouble(block []float64) {
	TransformBlock(block, DCT1d)
}

func ComputeBlockIDCTDouble(block []float64) {
	TransformBlock(block, IDCT1d)
}
