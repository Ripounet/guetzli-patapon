package guetzli_patapon

func Quantize(raw_coeff coeff_t, quant int) coeff_t {
	r := int(raw_coeff) % quant
	var delta coeff_t
	switch {
	case 2*r > quant:
		delta = coeff_t(quant - r)
	case -2*r > quant:
		delta = coeff_t(-quant - r)
	default:
		delta = coeff_t(-r)
	}
	return raw_coeff + delta
}

func QuantizeBlock(block []coeff_t, q []int) bool {
	changed := false
	for k := 0; k < kDCTBlockSize; k++ {
		coeff := Quantize(block[k], q[k])
		changed = changed || (coeff != block[k])
		block[k] = coeff
	}
	return changed
}
