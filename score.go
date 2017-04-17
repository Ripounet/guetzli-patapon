package guetzli_patapon

import "math"

func ScoreJPEG(butteraugli_distance float64, size int, butteraugli_target float64) float64 {
	const (
		kScale       = 50
		kMaxExponent = 10
		kLargeSize   = 1e30
	)

	// TODO(user): The score should also depend on distance below target (and be
	// smooth).
	diff := butteraugli_distance - butteraugli_target
	if diff <= 0.0 {
		return float64(size)
	}
	exponent := kScale * diff
	if exponent > kMaxExponent {
		return kLargeSize*math.Exp(kMaxExponent)*diff + float64(size)
	}
	return math.Exp(exponent) * float64(size)
}
