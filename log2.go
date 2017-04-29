package guetzli_patapon

func Log2FloorNonZero(n uint32) int {
	result := 0
	for {
		n >>= 1
		if n == 0 {
			break
		}
		result++
	}
	return result
}

func Log2Floor(n uint32) int {
	if n == 0 {
		return -1
	} else {
		return Log2FloorNonZero(n)
	}
}
