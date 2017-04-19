package guetzli_patapon

// Integer implementation of the Discrete Cosine Transform (DCT)
//
// Note! DCT output is kept scaled by 16, to retain maximum 16bit precision

///////////////////////////////////////////////////////////////////////////////
var (
	// Cosine table: C(k) = cos(k.pi/16)/sqrt(2), k = 1..7 using 15 bits signed
	kTable04 = [7]coeff_t{22725, 21407, 19266, 16384, 12873, 8867, 4520}
	// rows #1 and #7 are pre-multiplied by 2.C(1) before the 2nd pass.
	// This multiply is merged in the table of constants used during 1st pass:
	kTable17 = [7]coeff_t{31521, 29692, 26722, 22725, 17855, 12299, 6270}
	// rows #2 and #6 are pre-multiplied by 2.C(2):
	kTable26 = [7]coeff_t{29692, 27969, 25172, 21407, 16819, 11585, 5906}
	// rows #3 and #5 are pre-multiplied by 2.C(3):
	kTable35 = [7]coeff_t{26722, 25172, 22654, 19266, 15137, 10426, 5315}
)

///////////////////////////////////////////////////////////////////////////////
// Constants (15bit precision) and C macros for IDCT vertical pass

const (
	kTan1   = 13036  // = tan(pi/16)
	kTan2   = 27146  // = tan(2.pi/16) = sqrt(2) - 1.
	kTan3m1 = -21746 // = tan(3.pi/16) - 1
	k2Sqrt2 = 23170  // = 1 / 2.sqrt(2)
)

///////////////////////////////////////////////////////////////////////////////
// Constants for DCT horizontal pass

// Note about the CORRECT_LSB macro:
// using 16bit fixed-point constants, we often compute products like:
// p = (A*x + B*y + 32768) >> 16 by adding two sub-terms q = (A*x) >> 16
// and r = (B*y) >> 16 together. Statistically, we have p = q + r + 1
// in 3/4 of the cases. This can be easily seen from the relation:
//   (a + b + 1) >> 1 = (a >> 1) + (b >> 1) + ((a|b)&1)
// The approximation we are doing is replacing ((a|b)&1) by 1.
// In practice, this is a slightly more involved because the constants A and B
// have also been rounded compared to their exact floating point value.
// However, all in all the correction is quite small, and CORRECT_LSB can
// be defined empty if needed.

func CORRECT_LSB(a *int) {
	*a++
}

// DCT vertical pass

func ColumnDct(in []coeff_t) {
	for i := 0; i < 8; i++ {
		var m0, m1, m2, m3, m4, m5, m6, m7 int

		// COLUMN_DCT8:
		{
			in := in[i:]

			m0 = int(in[0*8])
			m2 = int(in[2*8])
			m7 = int(in[7*8])
			m5 = int(in[5*8])

			m0, m7 = (m0 - m7), (m0 + m7)
			m2, m5 = (m2 - m5), (m2 + m5)

			m3 = int(in[3*8])
			m4 = int(in[4*8])
			m3, m4 = (m3 - m4), (m3 + m4)

			m6 = int(in[6*8])
			m1 = int(in[1*8])
			m1, m6 = (m1 - m6), (m1 + m6)
			m7, m4 = (m7 - m4), (m7 + m4)
			m6, m5 = (m6 - m5), (m6 + m5)

			/* RowIdct() needs 15bits fixed-point input, when the output from   */
			/* ColumnIdct() would be 12bits. We are better doing the shift by 3 */
			/* now instead of in RowIdct(), because we have some multiplies to  */
			/* perform, that can take advantage of the extra 3bits precision.   */
			m4 = m4 << 3
			m5 = m5 << 3
			m4, m5 = (m4 - m5), (m4 + m5)
			in[0*8] = coeff_t(m5)
			in[4*8] = coeff_t(m4)

			m7 = m7 << 3
			m6 = m6 << 3
			m3 = m3 << 3
			m0 = m0 << 3

			m4 = kTan2
			m5 = m4
			m4 = (m4 * m7) >> 16
			m5 = (m5 * m6) >> 16
			m4 -= m6
			m5 += m7
			in[2*8] = coeff_t(m5)
			in[6*8] = coeff_t(m4)

			/* We should be multiplying m6 by C4 = 1/sqrt(2) here, but we only have */
			/* the k2Sqrt2 = 1/(2.sqrt(2)) constant that fits into 15bits. So we    */
			/* shift by 4 instead of 3 to compensate for the additional 1/2 factor. */
			m6 = k2Sqrt2
			m2 = m2<<3 + 1
			m1 = m1<<3 + 1
			m1, m2 = (m1 - m2), (m1 + m2)
			m2 = (m2 * m6) >> 16
			m1 = (m1 * m6) >> 16
			m3, m1 = (m3 - m1), (m3 + m1)
			m0, m2 = (m0 - m2), (m0 + m2)

			m4 = kTan3m1
			m5 = kTan1
			m7 = m3
			m6 = m1
			m3 = (m3 * m4) >> 16
			m1 = (m1 * m5) >> 16

			m3 += m7
			m1 += m2
			CORRECT_LSB(&m1)
			CORRECT_LSB(&m3)
			m4 = (m4 * m0) >> 16
			m5 = (m5 * m2) >> 16
			m4 += m0
			m0 -= m3
			m7 += m4
			m5 -= m6

			in[1*8] = coeff_t(m1)
			in[3*8] = coeff_t(m0)
			in[5*8] = coeff_t(m7)
			in[7*8] = coeff_t(m5)
		}

	}
}

// DCT horizontal pass

// We don't really need to round before descaling, since we
// still have 4 bits of precision left as final scaled output.
func DESCALE(a int) coeff_t {
	return coeff_t(a >> 16)
}

func RowDct(in, table []coeff_t) {
	// The Fourier transform is an unitary operator, so we're basically
	// doing the transpose of RowIdct()
	a0 := int(in[0] + in[7])
	b0 := int(in[0] - in[7])
	a1 := int(in[1] + in[6])
	b1 := int(in[1] - in[6])
	a2 := int(in[2] + in[5])
	b2 := int(in[2] - in[5])
	a3 := int(in[3] + in[4])
	b3 := int(in[3] - in[4])

	// even part
	C2 := int(table[1])
	C4 := int(table[3])
	C6 := int(table[5])
	c0 := int(a0 + a3)
	c1 := int(a0 - a3)
	c2 := int(a1 + a2)
	c3 := int(a1 - a2)

	in[0] = DESCALE(C4 * (c0 + c2))
	in[4] = DESCALE(C4 * (c0 - c2))
	in[2] = DESCALE(C2*c1 + C6*c3)
	in[6] = DESCALE(C6*c1 - C2*c3)

	// odd part
	C1 := int(table[0])
	C3 := int(table[2])
	C5 := int(table[4])
	C7 := int(table[6])
	in[1] = DESCALE(C1*b0 + C3*b1 + C5*b2 + C7*b3)
	in[3] = DESCALE(C3*b0 - C7*b1 - C1*b2 - C5*b3)
	in[5] = DESCALE(C5*b0 - C1*b1 + C7*b2 + C3*b3)
	in[7] = DESCALE(C7*b0 - C5*b1 + C3*b2 - C1*b3)
}

///////////////////////////////////////////////////////////////////////////////
// visible FDCT callable functions

func ComputeBlockDCT(coeffs []coeff_t) {
	ColumnDct(coeffs)
	RowDct(coeffs[0*8:], kTable04[:])
	RowDct(coeffs[1*8:], kTable17[:])
	RowDct(coeffs[2*8:], kTable26[:])
	RowDct(coeffs[3*8:], kTable35[:])
	RowDct(coeffs[4*8:], kTable04[:])
	RowDct(coeffs[5*8:], kTable35[:])
	RowDct(coeffs[6*8:], kTable26[:])
	RowDct(coeffs[7*8:], kTable17[:])
}
