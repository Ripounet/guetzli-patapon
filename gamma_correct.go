package guetzli_patapon

import "math"

func NewSrgb8ToLinearTable() []float64 {
	table := make([]float64, 256)
	i := 0
	for ; i < 11; i++ {
		table[i] = float64(i) / 12.92
	}
	for ; i < 256; i++ {
		table[i] = 255.0 * math.Pow(((float64(i)/255.0)+0.055)/1.055, 2.4)
	}
	return table
}

var Srgb8ToLinearTable = NewSrgb8ToLinearTable()
