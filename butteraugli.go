package guetzli_patapon

import "math"

type ButteraugliButteraugliComparator struct {
	xsize_, ysize_         int
	num_pixels_            int
	step_                  int
	res_xsize_, res_ysize_ int
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

const (
	kInternalGoodQualityThreshold = 14.921561160295326
	kGlobalScale                  = 1.0 / kInternalGoodQualityThreshold
)

func DotProduct(u, v []float64) float64 {
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
}

// weird variant of DotProduct
func DotProduct_(u []float32, v []float64) float64 {
	return float64(u[0])*v[0] + float64(u[1])*v[1] + float64(u[2])*v[2]
}

// Computes a horizontal convolution and transposes the result.
func Convolution(xsize, ysize int,
	xstep int,
	length, offset int,
	multipliers []float32,
	inp []float32,
	border_ratio float64,
	result []float32) {
	//PROFILER_FUNC;
	var weight_no_border float32
	for j := 0; j <= 2*offset; j++ {
		weight_no_border += multipliers[j]
	}
	for x, ox := 0, 0; x < xsize; x, ox = x+xstep, ox+1 {
		minx := minusOr0(x, offset)
		maxx := std_min(xsize, x+length-offset) - 1
		var weight float32
		for j := minx; j <= maxx; j++ {
			weight += multipliers[j-x+offset]
		}
		// Interpolate linearly between the no-border scaling and border scaling.
		weight = (1.0-float32(border_ratio))*weight + float32(border_ratio)*weight_no_border
		scale := 1.0 / weight
		for y := 0; y < ysize; y++ {
			var sum float32
			for j := minx; j <= maxx; j++ {
				sum += inp[y*xsize+j] * multipliers[j-x+offset]
			}
			result[ox*ysize+y] = float32(sum * scale)
		}
	}
}

func Blur(xsize, ysize int, channel []float32, sigma, border_ratio float64) {
	//PROFILER_FUNC;
	m := 2.25 // Accuracy increases when m is increased.
	scaler := -1.0 / (2 * sigma * sigma)
	// For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
	diff := std_max(1, int(m*math.Abs(sigma)))
	expn_size := 2*diff + 1
	expn := make([]float32, expn_size)
	for i := -diff; i <= diff; i++ {
		expn[i+diff] = float32(math.Exp(scaler * float64(i*i)))
	}
	xstep := std_max(1, int(sigma/3))
	ystep := xstep
	dxsize := (xsize + xstep - 1) / xstep
	dysize := (ysize + ystep - 1) / ystep
	tmp := make([]float32, dxsize*ysize)
	Convolution(xsize, ysize, xstep, expn_size, diff, expn, channel,
		border_ratio,
		tmp)
	output := channel
	var downsampled_output []float32
	if xstep > 1 {
		downsampled_output = make([]float32, dxsize*dysize)
		output = downsampled_output
	}
	Convolution(ysize, dxsize, ystep, expn_size, diff, expn, tmp,
		border_ratio, output)
	if xstep > 1 {
		for y := 0; y < ysize; y++ {
			for x := 0; x < xsize; x++ {
				// TODO: Use correct rounding.
				channel[y*xsize+x] =
					downsampled_output[(y/ystep)*dxsize+(x/xstep)]
			}
		}
	}
}

// To change this to n, add the relevant FFTn function and kFFTnMapIndexTable.
const (
	kBlockEdge     = 8
	kBlockSize_    = kBlockEdge * kBlockEdge
	kBlockEdgeHalf = kBlockEdge / 2
	kBlockHalf     = kBlockEdge * kBlockEdgeHalf
)

// Contrast sensitivity related weights.
var csf8x8 = [kBlockHalf + kBlockEdgeHalf + 1]float64{
	5.28270670524,
	0.0,
	0.0,
	0.0,
	0.3831134973,
	0.676303603859,
	3.58927792424,
	18.6104367002,
	18.6104367002,
	3.09093131948,
	1.0,
	0.498250875965,
	0.36198671102,
	0.308982169883,
	0.1312701920435,
	2.37370549629,
	3.58927792424,
	1.0,
	2.37370549629,
	0.991205724152,
	1.05178802919,
	0.627264168628,
	0.4,
	0.1312701920435,
	0.676303603859,
	0.498250875965,
	0.991205724152,
	0.5,
	0.3831134973,
	0.349686450518,
	0.627264168628,
	0.308982169883,
	0.3831134973,
	0.36198671102,
	1.05178802919,
	0.3831134973,
	0.12,
}

func GetContrastSensitivityMatrix() []float64 {
	return csf8x8[:]
}

func MakeHighFreqColorDiffDx() []float64 {
	lut := make([]float64, 21)
	const off = 11.38708334481672
	const inc = 14.550189611520716
	lut[0] = 0.0
	lut[1] = off
	for i := 2; i < 21; i++ {
		lut[i] = lut[i-1] + inc
	}
	return lut
}

func GetHighFreqColorDiffDx() []float64 {
	kLut := MakeHighFreqColorDiffDx()
	// TODO make only once?
	return kLut
}

func MakeHighFreqColorDiffDy() []float64 {
	lut := make([]float64, 21)
	const off = 1.4103373714040413
	const inc = 0.7084088867024
	lut[0] = 0.0
	lut[1] = off
	for i := 2; i < 21; i++ {
		lut[i] = lut[i-1] + inc
	}
	return lut
}

func GetHighFreqColorDiffDy() []float64 {
	kLut := MakeHighFreqColorDiffDy()
	// TODO make only once?
	return kLut
}

func MakeLowFreqColorDiffDy() []float64 {
	lut := make([]float64, 21)
	const inc = 5.2511644570349185
	lut[0] = 0.0
	for i := 1; i < 21; i++ {
		lut[i] = lut[i-1] + inc
	}
	return lut
}

func GetLowFreqColorDiffDy() []float64 {
	kLut := MakeLowFreqColorDiffDy()
	// TODO make only once?
	return kLut
}

func Interpolate(array []float64, size int, sx float64) float64 {
	ix := math.Abs(sx)
	assert(ix < 10000)
	baseix := int(ix)
	var res float64
	if baseix >= size-1 {
		res = array[size-1]
	} else {
		mix := ix - float64(baseix)
		nextix := baseix + 1
		res = array[baseix] + mix*(array[nextix]-array[baseix])
	}
	if sx < 0 {
		res = -res
	}
	return res
}

func InterpolateClampNegative(array []float64, size int, sx float64) float64 {
	if sx < 0 {
		sx = 0
	}
	ix := math.Abs(sx)
	baseix := int(ix)
	var res float64
	if baseix >= size-1 {
		res = array[size-1]
	} else {
		mix := ix - float64(baseix)
		nextix := baseix + 1
		res = array[baseix] + mix*(array[nextix]-array[baseix])
	}
	return res
}

func RgbToXyb(r, g, b float64, valx, valy, valz *float64) {
	const a0 = 1.01611726948
	const a1 = 0.982482243696
	const a2 = 1.43571362627
	const a3 = 0.896039849412
	*valx = a0*r - a1*g
	*valy = a2*r + a3*g
	*valz = b
}

func XybToVals(x, y, z float64, valx, valy, valz *float64) {
	const xmul = 0.758304045695
	const ymul = 2.28148649801
	const zmul = 1.87816926918
	*valx = Interpolate(GetHighFreqColorDiffDx(), 21, x*xmul)
	*valy = Interpolate(GetHighFreqColorDiffDy(), 21, y*ymul)
	*valz = zmul * z
}

// Rough psychovisual distance to gray for low frequency colors.
func XybLowFreqToVals(x, y, z float64, valx, valy, valz *float64) {
	const xmul = 6.64482198135
	const ymul = 0.837846224276
	const zmul = 7.34905756986
	const y_to_z_mul = 0.0812519812628
	z += y_to_z_mul * y
	*valz = z * zmul
	*valx = x * xmul
	*valy = Interpolate(GetLowFreqColorDiffDy(), 21, y*ymul)
}

func RemoveRangeAroundZero(v, Range float64) float64 {
	if v >= -Range && v < Range {
		return 0
	}
	if v < 0 {
		return v + Range
	} else {
		return v - Range
	}
}

func XybDiffLowFreqSquaredAccumulate(r0, g0, b0,
	r1, g1, b1,
	factor float64, res []float64) {
	var valx0, valy0, valz0 float64
	var valx1, valy1, valz1 float64
	XybLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0)
	if r1 == 0.0 && g1 == 0.0 && b1 == 0.0 {
		// PROFILER_ZONE("XybDiff r1=g1=b1=0")
		res[0] += factor * valx0 * valx0
		res[1] += factor * valy0 * valy0
		res[2] += factor * valz0 * valz0
		return
	}
	XybLowFreqToVals(r1, g1, b1, &valx1, &valy1, &valz1)
	// Approximate the distance of the colors by their respective distances
	// to gray.
	valx := valx0 - valx1
	valy := valy0 - valy1
	valz := valz0 - valz1
	res[0] += factor * valx * valx
	res[1] += factor * valy * valy
	res[2] += factor * valz * valz
}

type Complex complex128

func (c Complex) withReal(r float64) Complex {
	return Complex(complex(r, imag(c)))
}

func (c Complex) withImag(i float64) Complex {
	return Complex(complex(real(c), i))
}

func NewComplex(r, i float64) Complex {
	return Complex(complex(r, i))
}

func abssq(c Complex) float64 {
	return real(c)*real(c) + imag(c)*imag(c)
}

func TransposeBlock(data []Complex) {
	for i := 0; i < kBlockEdge; i++ {
		for j := 0; j < i; j++ {
			data[kBlockEdge*i+j], data[kBlockEdge*j+i] = data[kBlockEdge*j+i], data[kBlockEdge*i+j]
		}
	}
}

//  D. J. Bernstein's Fast Fourier Transform algorithm on 4 elements.
func FFT4(a []Complex) {
	var t1, t2, t3, t4, t5, t6, t7, t8 float64
	t5 = real(a[2])
	t1 = real(a[0]) - t5
	t7 = real(a[3])
	t5 += real(a[0])
	t3 = real(a[1]) - t7
	t7 += real(a[1])
	t8 = t5 + t7
	a[0] = a[0].withReal(t8)
	t5 -= t7
	a[1] = a[1].withReal(t5)
	t6 = imag(a[2])
	t2 = imag(a[0]) - t6
	t6 += imag(a[0])
	t5 = imag(a[3])
	a[2] = a[2].withImag(t2 + t3)
	t2 -= t3
	a[3] = a[3].withImag(t2)
	t4 = imag(a[1]) - t5
	a[3] = a[3].withReal(t1 + t4)
	t1 -= t4
	a[2] = a[2].withReal(t1)
	t5 += imag(a[1])
	a[0] = a[0].withImag(t6 + t5)
	t6 -= t5
	a[1] = a[1].withImag(t6)
}

const kSqrtHalf = 0.70710678118654752440084436210484903

//  D. J. Bernstein's Fast Fourier Transform algorithm on 8 elements.
func FFT8(a []Complex) {
	var t1, t2, t3, t4, t5, t6, t7, t8 float64

	t7 = imag(a[4])
	t4 = imag(a[0]) - t7
	t7 += imag(a[0])
	a[0] = a[0].withImag(t7)

	t8 = real(a[6])
	t5 = real(a[2]) - t8
	t8 += real(a[2])
	a[2] = a[2].withReal(t8)

	t7 = imag(a[6])
	a[6] = a[6].withImag(t4 - t5)
	t4 += t5

	t6 = imag(a[2]) - t7
	t7 += imag(a[2])
	a[2] = a[2].withImag(t7)

	t8 = real(a[4])
	t3 = real(a[0]) - t8
	t8 += real(a[0])
	a[0] = a[0].withReal(t8)

	a[4] = a[4].withReal(t3 - t6)
	t3 += t6
	a[6] = a[6].withReal(t3)

	t7 = real(a[5])
	t3 = real(a[1]) - t7
	t7 += real(a[1])
	a[1] = a[1].withReal(t7)

	t8 = imag(a[7])
	t6 = imag(a[3]) - t8
	t8 += imag(a[3])
	a[3] = a[3].withImag(t8)
	t1 = t3 - t6
	t3 += t6

	t7 = imag(a[5])
	t4 = imag(a[1]) - t7
	t7 += imag(a[1])
	a[1] = a[1].withImag(t7)

	t8 = real(a[7])
	t5 = real(a[3]) - t8
	t8 += real(a[3])
	a[3] = a[3].withReal(t8)

	t2 = t4 - t5
	t4 += t5

	t6 = t1 - t4
	t8 = kSqrtHalf
	t6 *= t8
	a[5] -= NewComplex(t6, 0)
	t1 += t4
	t1 *= t8
	a[5] -= NewComplex(0, t1)
	t6 += real(a[4])
	a[4] = a[4].withReal(t6)
	t1 += imag(a[4])
	a[4] = a[4].withImag(t1)

	t5 = t2 - t3
	t5 *= t8
	a[7] = a[7].withImag(imag(a[6]) - t5)
	t2 += t3
	t2 *= t8
	a[7] = a[7].withReal(real(a[6]) - t2)
	t2 += real(a[6])
	a[6] = a[6].withReal(t2)
	t5 += imag(a[6])
	a[6] = a[6].withImag(t5)

	FFT4(a)

	// Reorder to the correct output order.
	// TODO: Modify the above computation so that this is not needed.
	tmp := a[2]
	a[2] = a[3]
	a[3] = a[5]
	a[5] = a[7]
	a[7] = a[4]
	a[4] = a[1]
	a[1] = a[6]
	a[6] = tmp
}

// Same as FFT8, but all inputs are real.
// TODO: Since this does not need to be in-place, maybe there is a
// faster FFT than this one, which is derived from DJB's in-place complex FFT.
func RealFFT8(in []float64, out []Complex) {
	var t1, t2, t3, t5, t6, t7, t8 float64
	t8 = in[6]
	t5 = in[2] - t8
	t8 += in[2]
	out[2] = out[2].withReal(t8)
	out[6] = out[6].withImag(-t5)
	out[4] = out[4].withImag(t5)
	t8 = in[4]
	t3 = in[0] - t8
	t8 += in[0]
	out[0] = out[0].withReal(t8)
	out[4] = out[4].withReal(t3)
	out[6] = out[6].withReal(t3)
	t7 = in[5]
	t3 = in[1] - t7
	t7 += in[1]
	out[1] = out[1].withReal(t7)
	t8 = in[7]
	t5 = in[3] - t8
	t8 += in[3]
	out[3] = out[3].withReal(t8)
	t2 = -t5
	t6 = t3 - t5
	t8 = kSqrtHalf
	t6 *= t8
	out[5] = out[5].withReal(real(out[4]) - t6)
	t1 = t3 + t5
	t1 *= t8
	out[5] = out[5].withImag(imag(out[4]) - t1)
	t6 += real(out[4])
	out[4] = out[4].withReal(t6)
	t1 += imag(out[4])
	out[4] = out[4].withImag(t1)
	t5 = t2 - t3
	t5 *= t8
	out[7] = out[7].withImag(imag(out[6]) - t5)
	t2 += t3
	t2 *= t8
	out[7] = out[7].withReal(real(out[6]) - t2)
	t2 += real(out[6])
	out[6] = out[6].withReal(t2)
	t5 += imag(out[6])
	out[6] = out[6].withImag(t5)
	t5 = real(out[2])
	t1 = real(out[0]) - t5
	t7 = real(out[3])
	t5 += real(out[0])
	t3 = real(out[1]) - t7
	t7 += real(out[1])
	t8 = t5 + t7
	out[0] = out[0].withReal(t8)
	t5 -= t7
	out[1] = out[1].withReal(t5)
	out[2] = out[2].withImag(t3)
	out[3] = out[3].withImag(-t3)
	out[3] = out[3].withReal(t1)
	out[2] = out[2].withReal(t1)
	out[0] = out[0].withImag(0)
	out[1] = out[1].withImag(0)

	// Reorder to the correct output order.
	// TODO: Modify the above computation so that this is not needed.
	tmp := out[2]
	out[2] = out[3]
	out[3] = out[5]
	out[5] = out[7]
	out[7] = out[4]
	out[4] = out[1]
	out[1] = out[6]
	out[6] = tmp
}

// Fills in block[kBlockEdgeHalf..(kBlockHalf+kBlockEdgeHalf)], and leaves the
// rest unmodified.
func ButteraugliFFTSquared(block []float64) {
	global_mul := 0.000064
	block_c := make([]Complex, kBlockSize_)
	assert(kBlockEdge == 8)
	for y := 0; y < kBlockEdge; y++ {
		RealFFT8(block[y*kBlockEdge:], block_c[y*kBlockEdge:])
	}
	TransposeBlock(block_c)
	r0 := make([]float64, kBlockEdge)
	r1 := make([]float64, kBlockEdge)
	for x := 0; x < kBlockEdge; x++ {
		r0[x] = real(block_c[x])
		r1[x] = real(block_c[kBlockHalf+x])
	}
	RealFFT8(r0, block_c)
	RealFFT8(r1, block_c[kBlockHalf:])
	for y := 1; y < kBlockEdgeHalf; y++ {
		FFT8(block_c[y*kBlockEdge:])
	}
	for i := kBlockEdgeHalf; i < kBlockHalf+kBlockEdgeHalf+1; i++ {
		block[i] = abssq(block_c[i])
		block[i] *= global_mul
	}
}

// Computes 8x8 FFT of each channel of xyb0 and xyb1 and adds the total squared
// 3-dimensional xybdiff of the two blocks to diff_xyb_{dc,ac} and the average
// diff on the edges to diff_xyb_edge_dc.
func ButteraugliBlockDiff(xyb0, xyb1 []float64, diff_xyb_dc, diff_xyb_ac, diff_xyb_edge_dc []float64) {
	// PROFILER_FUNC;
	csf8x8 := GetContrastSensitivityMatrix()

	var avgdiff_xyb [3]float64
	var avgdiff_edge [3][4]float64
	for i := 0; i < 3*kBlockSize_; i++ {
		diff_xyb := xyb0[i] - xyb1[i]
		c := i / kBlockSize_
		avgdiff_xyb[c] += diff_xyb / kBlockSize_
		k := i % kBlockSize_
		kx := k % kBlockEdge
		ky := k / kBlockEdge
		var h_edge_idx, v_edge_idx int
		if ky == 0 {
			h_edge_idx = 1
		} else if ky == 7 {
			h_edge_idx = 3
		} else {
			h_edge_idx = -1
		}
		if kx == 0 {
			v_edge_idx = 1
		} else if kx == 7 {
			v_edge_idx = 3
		} else {
			v_edge_idx = -1
		}
		if h_edge_idx >= 0 {
			avgdiff_edge[c][h_edge_idx] += diff_xyb / kBlockEdge
		}
		if v_edge_idx >= 0 {
			avgdiff_edge[c][v_edge_idx] += diff_xyb / kBlockEdge
		}
	}
	XybDiffLowFreqSquaredAccumulate(avgdiff_xyb[0],
		avgdiff_xyb[1],
		avgdiff_xyb[2],
		0, 0, 0, csf8x8[0],
		diff_xyb_dc)
	for i := 0; i < 4; i++ {
		XybDiffLowFreqSquaredAccumulate(avgdiff_edge[0][i],
			avgdiff_edge[1][i],
			avgdiff_edge[2][i],
			0, 0, 0, csf8x8[0],
			diff_xyb_edge_dc)
	}

	xyb_avg := xyb0
	xyb_halfdiff := xyb1
	for i := 0; i < 3*kBlockSize_; i++ {
		avg := (xyb0[i] + xyb1[i]) / 2
		halfdiff := (xyb0[i] - xyb1[i]) / 2
		xyb_avg[i] = avg
		xyb_halfdiff[i] = halfdiff
	}
	y_avg := xyb_avg[kBlockSize_:]
	x_halfdiff_squared := xyb_halfdiff[0:]
	y_halfdiff := xyb_halfdiff[kBlockSize_:]
	z_halfdiff_squared := xyb_halfdiff[2*kBlockSize_:]
	ButteraugliFFTSquared(y_avg)
	ButteraugliFFTSquared(x_halfdiff_squared)
	ButteraugliFFTSquared(y_halfdiff)
	ButteraugliFFTSquared(z_halfdiff_squared)

	const xmul = 64.8
	const ymul = 1.753123908348329
	const ymul2 = 1.51983458269
	const zmul = 2.4

	for i := kBlockEdgeHalf; i < kBlockHalf+kBlockEdgeHalf+1; i++ {
		d := csf8x8[i]
		diff_xyb_ac[0] += d * xmul * x_halfdiff_squared[i]
		diff_xyb_ac[2] += d * zmul * z_halfdiff_squared[i]

		y_avg[i] = math.Sqrt(y_avg[i])
		y_halfdiff[i] = math.Sqrt(y_halfdiff[i])
		y0 := y_avg[i] - y_halfdiff[i]
		y1 := y_avg[i] + y_halfdiff[i]
		// Remove the impact of small absolute values.
		// This improves the behavior with flat noise.
		const ylimit = 0.04
		y0 = RemoveRangeAroundZero(y0, ylimit)
		y1 = RemoveRangeAroundZero(y1, ylimit)
		if y0 != y1 {
			valy0 := Interpolate(GetHighFreqColorDiffDy(), 21, y0*ymul2)
			valy1 := Interpolate(GetHighFreqColorDiffDy(), 21, y1*ymul2)
			valy := ymul * (valy0 - valy1)
			diff_xyb_ac[1] += d * valy * valy
		}
	}
}

// Low frequency edge detectors.
// Two edge detectors are applied in each corner of the 8x8 square.
// The squared 3-dimensional error vector is added to diff_xyb.
func Butteraugli8x8CornerEdgeDetectorDiff(
	pos_x, pos_y, xsize, ysize int,
	blurred0, blurred1 [][]float32,
	diff_xyb []float64) {
	// PROFILER_FUNC;
	local_count := 0.0
	var local_xyb [3]float64
	const w = 0.711100840192
	for k := 0; k < 4; k++ {
		step := 3
		offset := [4][2]int{{0, 0}, {0, 7}, {7, 0}, {7, 7}}
		x := pos_x + offset[k][0]
		y := pos_y + offset[k][1]
		if x >= step && x+step < xsize {
			ix := y*xsize + (x - step)
			ix2 := ix + 2*step
			XybDiffLowFreqSquaredAccumulate(
				w*float64(blurred0[0][ix]-blurred0[0][ix2]),
				w*float64(blurred0[1][ix]-blurred0[1][ix2]),
				w*float64(blurred0[2][ix]-blurred0[2][ix2]),
				w*float64(blurred1[0][ix]-blurred1[0][ix2]),
				w*float64(blurred1[1][ix]-blurred1[1][ix2]),
				w*float64(blurred1[2][ix]-blurred1[2][ix2]),
				1.0, local_xyb[:])
			local_count++
		}
		if y >= step && y+step < ysize {
			ix := (y-step)*xsize + x
			ix2 := ix + 2*step*xsize
			XybDiffLowFreqSquaredAccumulate(
				w*float64(blurred0[0][ix]-blurred0[0][ix2]),
				w*float64(blurred0[1][ix]-blurred0[1][ix2]),
				w*float64(blurred0[2][ix]-blurred0[2][ix2]),
				w*float64(blurred1[0][ix]-blurred1[0][ix2]),
				w*float64(blurred1[1][ix]-blurred1[1][ix2]),
				w*float64(blurred1[2][ix]-blurred1[2][ix2]),
				1.0, local_xyb[:])
			local_count++
		}
	}
	const weight = 0.01617112696
	mul := weight * 8.0 / local_count
	for i := 0; i < 3; i++ {
		diff_xyb[i] += mul * local_xyb[i]
	}
}

// https://en.wikipedia.org/wiki/Photopsin absordance modeling.
var opsinAbsorbance = []float64{
	0.348036746003,
	0.577814843137,
	0.0544556093735,
	0.774145581713,
	0.26922717275,
	0.767247733938,
	0.0366922708552,
	0.920130265014,
	0.0882062883536,
	0.158581714673,
	0.712857943858,
	10.6524069248,
}

func OpsinAbsorbance(in, out []float64) {
	mix := opsinAbsorbance
	out[0] = mix[0]*in[0] + mix[1]*in[1] + mix[2]*in[2] + mix[3]
	out[1] = mix[4]*in[0] + mix[5]*in[1] + mix[6]*in[2] + mix[7]
	out[2] = mix[8]*in[0] + mix[9]*in[1] + mix[10]*in[2] + mix[11]
}

func GammaMinArg() float64 {
	var in, out [3]float64
	OpsinAbsorbance(in[:], out[:])
	return std_minFloat64(out[0], std_minFloat64(out[1], out[2]))
}

func GammaMaxArg() float64 {
	in := [3]float64{255.0, 255.0, 255.0}
	var out [3]float64
	OpsinAbsorbance(in[:], out[:])
	return std_maxFloat64(out[0], std_maxFloat64(out[1], out[2]))
}

func NewButteraugliButteraugliComparator(xsize, ysize, step int) *ButteraugliButteraugliComparator {
	assert(step <= 4)
	return &ButteraugliButteraugliComparator{
		xsize_:      xsize,
		ysize_:      ysize,
		num_pixels_: xsize * ysize,
		step_:       step,
		res_xsize_:  (xsize + step - 1) / step,
		res_ysize_:  (ysize + step - 1) / step,
	}
}

func MaskHighIntensityChange(xsize, ysize int, c0, c1, xyb0, xyb1 [][]float32) {
	// PROFILER_FUNC;
	for y := 0; y < ysize; y++ {
		for x := 0; x < xsize; x++ {
			ix := y*xsize + x
			ave := [3]float64{
				float64(c0[0][ix]+c1[0][ix]) * 0.5,
				float64(c0[1][ix]+c1[1][ix]) * 0.5,
				float64(c0[2][ix]+c1[2][ix]) * 0.5,
			}
			sqr_max_diff := -1.0
			{
				offset := [4]int{-1, 1, -int(xsize), int(xsize)}
				border := [4]bool{x == 0, x+1 == xsize, y == 0, y+1 == ysize}
				for dir := 0; dir < 4; dir++ {
					if border[dir] {
						continue
					}
					ix2 := ix + offset[dir]
					diff := 0.5*float64(c0[1][ix2]+c1[1][ix2]) - ave[1]
					diff *= diff
					if sqr_max_diff < diff {
						sqr_max_diff = diff
					}
				}
			}
			const kReductionX = 275.19165240059317
			const kReductionY = 18599.41286306991
			const kReductionZ = 410.8995306951065
			const kChromaBalance = 106.95800948271017
			chroma_scale := kChromaBalance / (ave[1] + kChromaBalance)

			mix := [3]float64{
				chroma_scale * kReductionX / (sqr_max_diff + kReductionX),
				kReductionY / (sqr_max_diff + kReductionY),
				chroma_scale * kReductionZ / (sqr_max_diff + kReductionZ),
			}
			// Interpolate lineraly between the average color and the actual
			// color -- to reduce the importance of this pixel.
			for i := 0; i < 3; i++ {
				xyb0[i][ix] = float32(mix[i])*c0[i][ix] + (1 - float32(mix[i])*float32(ave[i]))
				xyb1[i][ix] = float32(mix[i])*c1[i][ix] + (1 - float32(mix[i])*float32(ave[i]))
			}
		}
	}
}

func SimpleGamma(v float64) float64 {
	const kGamma = 0.387494322593
	const limit = 43.01745241042018
	bright := v - limit
	if bright >= 0 {
		const mul = 0.0383723643799
		v -= bright * mul
	}
	const limit2 = 94.68634353321337
	bright2 := v - limit2
	if bright2 >= 0 {
		const mul = 0.22885405968
		v -= bright2 * mul
	}
	const offset = 0.156775786057
	const scale = 8.898059160493739
	retval := scale * (offset + math.Pow(v, kGamma))
	return retval
}

// Polynomial evaluation via Clenshaw's scheme (similar to Horner's).
// Template enables compile-time unrolling of the recursion, but must reside
// outside of a class due to the specialization.
func ClenshawRecursion(INDEX int, x float64, coefficients []float64, b1, b2 *float64) {
	if INDEX == 0 {
		ClenshawRecursion0(x, coefficients, b1, b2)
		return
	}

	x_b1 := x * (*b1)
	t := (x_b1 + x_b1) - (*b2) + coefficients[INDEX]
	*b2 = *b1
	*b1 = t

	ClenshawRecursion(INDEX-1, x, coefficients, b1, b2)
}

func ClenshawRecursion0(x float64, coefficients []float64, b1, b2 *float64) {
	x_b1 := x * (*b1)
	// The final iteration differs - no 2 * x_b1 here.
	*b1 = x_b1 - (*b2) + coefficients[0]
}

// Rational polynomial := dividing two polynomial evaluations. These are easier
// to find than minimax polynomials.
type RationalPolynomial struct {
	// Domain of the polynomials; they are undefined elsewhere.
	min_value, max_value float64

	// Coefficients of T_n (Chebyshev polynomials of the first kind).
	// Degree 5/5 is a compromise between accuracy (0.1%) and numerical stability.
	p, q [5 + 1]float64
}

// Evaluates the polynomial at x (in [min_value, max_value]).
func (rp *RationalPolynomial) EvalAt(x float32) float64 {
	// First normalize to [0, 1].
	x01 := (float64(x) - rp.min_value) / (rp.max_value - rp.min_value)
	// And then to [-1, 1] domain of Chebyshev polynomials.
	xc := 2.0*x01 - 1.0

	yp := EvaluatePolynomial(xc, rp.p[:], len(rp.p))
	yq := EvaluatePolynomial(xc, rp.q[:], len(rp.q))
	if yq == 0.0 {
		return 0.0
	}
	return yp / yq
}

func EvaluatePolynomial(x float64, coefficients []float64, N int) float64 {
	b1 := 0.0
	b2 := 0.0
	ClenshawRecursion(N-1, x, coefficients, &b1, &b2)
	return b1
}

func GammaPolynomial(value float32) float32 {
	// Generated by gamma_polynomial.m from equispaced x/gamma(x) samples.
	r := RationalPolynomial{
		min_value: 0.770000000000000,
		max_value: 274.579999999999984,
		p: [5 + 1]float64{
			881.979476556478289, 1496.058452015812463, 908.662212739659481,
			373.566100223287378, 85.840860336314364, 6.683258861509244,
		},
		q: [5 + 1]float64{
			12.262350348616792, 20.557285797683576, 12.161463238367844,
			4.711532733641639, 0.899112889751053, 0.035662329617191,
		},
	}
	return float32(r.EvalAt(value))
}

func Gamma(v float64) float64 {
	// return SimpleGamma(v);
	return float64(GammaPolynomial(float32(v)))
}

func OpsinDynamicsImage(xsize, ysize int, rgb [][]float32) {
	// PROFILER_FUNC;
	blurred := cloneMatrixFloat32(rgb) // TODO PATAPON is this intended to be a copy?
	const kSigma = 1.1
	for i := 0; i < 3; i++ {
		Blur(xsize, ysize, blurred[i], kSigma, 0.0)
	}
	for i := 0; i < len(rgb[0]); i++ {
		var sensitivity [3]float64
		{
			// Calculate sensitivity[3] based on the smoothed image gamma derivative.
			pre_rgb := [3]float64{
				float64(blurred[0][i]),
				float64(blurred[1][i]),
				float64(blurred[2][i]),
			}
			var pre_mixed [3]float64
			OpsinAbsorbance(pre_rgb[:], pre_mixed[:])
			sensitivity[0] = Gamma(pre_mixed[0]) / pre_mixed[0]
			sensitivity[1] = Gamma(pre_mixed[1]) / pre_mixed[1]
			sensitivity[2] = Gamma(pre_mixed[2]) / pre_mixed[2]
		}
		cur_rgb := [3]float64{
			float64(rgb[0][i]),
			float64(rgb[1][i]),
			float64(rgb[2][i]),
		}
		var cur_mixed [3]float64
		OpsinAbsorbance(cur_rgb[:], cur_mixed[:])
		cur_mixed[0] *= sensitivity[0]
		cur_mixed[1] *= sensitivity[1]
		cur_mixed[2] *= sensitivity[2]
		var x, y, z float64
		RgbToXyb(cur_mixed[0], cur_mixed[1], cur_mixed[2], &x, &y, &z)
		rgb[0][i] = float32(x)
		rgb[1][i] = float32(y)
		rgb[2][i] = float32(z)
	}
}

func ScaleImage(scale float64, result []float32) {
	// PROFILER_FUNC;
	for i := 0; i < len(result); i++ {
		result[i] *= float32(scale)
	}
}

// Making a cluster of local errors to be more impactful than
// just a single error.
func CalculateDiffmap(xsize, ysize, step int, diffmap []float32) {
	// PROFILER_FUNC;

	// Shift the diffmap more correctly above the pixels, from 2.5 pixels to 0.5
	// pixels distance over the original image. The border of 2 pixels on top and
	// left side and 3 pixels on right and bottom side are zeroed, but these
	// values have no meaning, they only exist to keep the result map the same
	// size as the input images.
	s2 := (8 - step) / 2
	{
		// Upsample and take square root.
		diffmap_out := make([]float32, xsize*ysize)
		res_xsize := (xsize + step - 1) / step
		for res_y := 0; res_y+8-step < ysize; res_y += step {
			for res_x := 0; res_x+8-step < xsize; res_x += step {
				res_ix := (res_y*res_xsize + res_x) / step
				orig_val := diffmap[res_ix]
				const kInitialSlope = 100.0
				// TODO(b/29974893): Until that is fixed do not call sqrt on very small
				// numbers.
				var val float32
				if orig_val < (1.0 / (kInitialSlope * kInitialSlope)) {
					val = kInitialSlope * orig_val
				} else {
					val = sqrt32(orig_val)
				}

				for off_y := 0; off_y < step; off_y++ {
					for off_x := 0; off_x < step; off_x++ {
						diffmap_out[(res_y+off_y+s2)*xsize+res_x+off_x+s2] = val
					}
				}
			}
		}
		copy(diffmap, diffmap_out)
	}
	{
		const kSigma = 8.8510880283
		const mul1 = 24.8235314874
		const scale = 1.0 / (1.0 + mul1)
		s := 8 - step
		blurred := make([]float32, (xsize-s)*(ysize-s))
		for y := 0; y < ysize-s; y++ {
			for x := 0; x < xsize-s; x++ {
				blurred[y*(xsize-s)+x] = diffmap[(y+s2)*xsize+x+s2]
			}
		}
		const border_ratio = 0.03027655136
		Blur(xsize-s, ysize-s, blurred, kSigma, border_ratio)
		for y := 0; y < ysize-s; y++ {
			for x := 0; x < xsize-s; x++ {
				diffmap[(y+s2)*xsize+x+s2] += float32(mul1) * blurred[y*(xsize-s)+x]
			}
		}
		ScaleImage(scale, diffmap)
	}
}

func (bbc *ButteraugliButteraugliComparator) DiffmapOpsinDynamicsImage(xyb0_arg, xyb1 [][]float32) (result []float32) {
	if bbc.xsize_ < 8 || bbc.ysize_ < 8 {
		return
	}
	xyb0 := xyb0_arg
	{
		xyb1_c := xyb1
		MaskHighIntensityChange(bbc.xsize_, bbc.ysize_, xyb0_arg, xyb1_c, xyb0, xyb1)
	}
	assert(8 <= bbc.xsize_)
	for i := 0; i < 3; i++ {
		assert(len(xyb0[i]) == bbc.num_pixels_)
		assert(len(xyb1[i]) == bbc.num_pixels_)
	}
	edge_detector_map := make([]float32, 3*bbc.res_xsize_*bbc.res_ysize_)
	bbc.EdgeDetectorMap(xyb0, xyb1, edge_detector_map)
	block_diff_dc := make([]float32, 3*bbc.res_xsize_*bbc.res_ysize_)
	block_diff_ac := make([]float32, 3*bbc.res_xsize_*bbc.res_ysize_)
	bbc.BlockDiffMap(xyb0, xyb1, block_diff_dc, block_diff_ac)
	bbc.EdgeDetectorLowFreq(xyb0, xyb1, block_diff_ac)
	{
		mask_xyb := make([][]float32, 3)
		mask_xyb_dc := make([][]float32, 3)
		Mask(xyb0, xyb1, bbc.xsize_, bbc.ysize_, mask_xyb, mask_xyb_dc)
		result = bbc.CombineChannels(mask_xyb, mask_xyb_dc, block_diff_dc, block_diff_ac, edge_detector_map)
	}
	CalculateDiffmap(bbc.xsize_, bbc.ysize_, bbc.step_, result)
	return result
}

func (bbc *ButteraugliButteraugliComparator) BlockDiffMap(xyb0, xyb1 [][]float32, block_diff_dc, block_diff_ac []float32) {
	// PROFILER_FUNC;
	for res_y := 0; res_y+(kBlockEdge-bbc.step_-1) < bbc.ysize_; res_y += bbc.step_ {
		for res_x := 0; res_x+(kBlockEdge-bbc.step_-1) < bbc.xsize_; res_x += bbc.step_ {
			res_ix := (res_y*bbc.res_xsize_ + res_x) / bbc.step_
			offset := (std_min(res_y, bbc.ysize_-8)*bbc.xsize_ + std_min(res_x, bbc.xsize_-8))
			var block0, block1 [3 * kBlockEdge * kBlockEdge]float64
			for i := 0; i < 3; i++ {
				m0 := block0[i*kBlockEdge*kBlockEdge:]
				m1 := block1[i*kBlockEdge*kBlockEdge:]
				for y := 0; y < kBlockEdge; y++ {
					for x := 0; x < kBlockEdge; x++ {
						m0[kBlockEdge*y+x] = float64(xyb0[i][offset+y*bbc.xsize_+x])
						m1[kBlockEdge*y+x] = float64(xyb1[i][offset+y*bbc.xsize_+x])
					}
				}
			}
			var diff_xyb_dc, diff_xyb_ac, diff_xyb_edge_dc [3]float64
			ButteraugliBlockDiff(block0[:], block1[:], diff_xyb_dc[:], diff_xyb_ac[:], diff_xyb_edge_dc[:])
			for i := 0; i < 3; i++ {
				block_diff_dc[3*res_ix+i] = float32(diff_xyb_dc[i])
				block_diff_ac[3*res_ix+i] = float32(diff_xyb_ac[i])
			}
		}
	}
}

func (bbc *ButteraugliButteraugliComparator) EdgeDetectorMap(xyb0, xyb1 [][]float32, edge_detector_map []float32) {
	// PROFILER_FUNC;
	kSigma := [3]float64{
		1.5,
		0.586,
		0.4,
	}
	blurred0 := cloneMatrixFloat32(xyb0) // TODO PATAPON is this intended to be a copy?
	blurred1 := cloneMatrixFloat32(xyb1) // TODO PATAPON is this intended to be a copy?
	for i := 0; i < 3; i++ {
		Blur(bbc.xsize_, bbc.ysize_, blurred0[i], kSigma[i], 0.0)
		Blur(bbc.xsize_, bbc.ysize_, blurred1[i], kSigma[i], 0.0)
	}
	for res_y := 0; res_y+(8-bbc.step_) < bbc.ysize_; res_y += bbc.step_ {
		for res_x := 0; res_x+(8-bbc.step_) < bbc.xsize_; res_x += bbc.step_ {
			res_ix := (res_y*bbc.res_xsize_ + res_x) / bbc.step_
			var diff_xyb [3]float64
			Butteraugli8x8CornerEdgeDetectorDiff(std_min(res_x, bbc.xsize_-8),
				std_min(res_y, bbc.ysize_-8),
				bbc.xsize_, bbc.ysize_,
				blurred0, blurred1,
				diff_xyb[:])
			for i := 0; i < 3; i++ {
				edge_detector_map[3*res_ix+i] = float32(diff_xyb[i])
			}
		}
	}
}

func (bbc *ButteraugliButteraugliComparator) EdgeDetectorLowFreq(xyb0, xyb1 [][]float32, block_diff_ac []float32) {
	// PROFILER_FUNC;
	const kSigma = 14
	const kMul = 10
	blurred0 := cloneMatrixFloat32(xyb0) // TODO PATAPON is this intended to be a copy?
	blurred1 := cloneMatrixFloat32(xyb1) // TODO PATAPON is this intended to be a copy?
	for i := 0; i < 3; i++ {
		Blur(bbc.xsize_, bbc.ysize_, blurred0[i], kSigma, 0.0)
		Blur(bbc.xsize_, bbc.ysize_, blurred1[i], kSigma, 0.0)
	}
	step := 8
	for y := 0; y+step < bbc.ysize_; y += bbc.step_ {
		resy := y / bbc.step_
		resx := step / bbc.step_
		for x := 0; x+step < bbc.xsize_; x, resx = x+bbc.step_, resx+1 {
			ix := y*bbc.xsize_ + x
			res_ix := resy*bbc.res_xsize_ + resx
			var diff [4][3]float64
			for i := 0; i < 3; i++ {
				ix2 := ix + 8
				diff[0][i] =
					float64((blurred1[i][ix] - blurred0[i][ix]) +
						(blurred0[i][ix2] - blurred1[i][ix2]))
				ix2 = ix + 8*bbc.xsize_
				diff[1][i] =
					float64((blurred1[i][ix] - blurred0[i][ix]) +
						(blurred0[i][ix2] - blurred1[i][ix2]))
				ix2 = ix + 6*bbc.xsize_ + 6
				diff[2][i] =
					float64((blurred1[i][ix] - blurred0[i][ix]) +
						(blurred0[i][ix2] - blurred1[i][ix2]))
				ix2 = ix + 6*bbc.xsize_ - 6
				diff[3][i] = 0
				if x >= step {
					diff[3][i] = float64((blurred1[i][ix] - blurred0[i][ix]) + (blurred0[i][ix2] - blurred1[i][ix2]))
				}
			}
			var max_diff_xyb [3]float64
			for k := 0; k < 4; k++ {
				var diff_xyb [3]float64
				XybDiffLowFreqSquaredAccumulate(diff[k][0], diff[k][1], diff[k][2],
					0, 0, 0, 1.0,
					diff_xyb[:])
				for i := 0; i < 3; i++ {
					max_diff_xyb[i] = std_maxFloat64(max_diff_xyb[i], diff_xyb[i])
				}
			}
			for i := 0; i < 3; i++ {
				block_diff_ac[3*res_ix+i] += float32(kMul * max_diff_xyb[i])
			}
		}
	}
}

func (bbc *ButteraugliButteraugliComparator) CombineChannels(
	mask_xyb, mask_xyb_dc [][]float32,
	block_diff_dc, block_diff_ac, edge_detector_map []float32) (result []float32) {
	// PROFILER_FUNC;
	result = make([]float32, bbc.res_xsize_*bbc.res_ysize_)
	for res_y := 0; res_y+(8-bbc.step_) < bbc.ysize_; res_y += bbc.step_ {
		for res_x := 0; res_x+(8-bbc.step_) < bbc.xsize_; res_x += bbc.step_ {
			res_ix := (res_y*bbc.res_xsize_ + res_x) / bbc.step_
			var mask, dc_mask [3]float64
			for i := 0; i < 3; i++ {
				mask[i] = float64(mask_xyb[i][(res_y+3)*bbc.xsize_+(res_x+3)])
				dc_mask[i] = float64(mask_xyb_dc[i][(res_y+3)*bbc.xsize_+(res_x+3)])
			}
			result[res_ix] = float32(
				DotProduct_(block_diff_dc[3*res_ix:], dc_mask[:]) +
					DotProduct_(block_diff_ac[3*res_ix:], mask[:]) +
					DotProduct_(edge_detector_map[3*res_ix:], mask[:]))
		}
	}
	return result
}

func ButteraugliScoreFromDiffmap(diffmap []float32) float64 {
	// PROFILER_FUNC;
	var retval float64
	for ix := 0; ix < len(diffmap); ix++ {
		retval = std_maxFloat64(retval, float64(diffmap[ix]))
	}
	return retval
}

func MakeMask(extmul, extoff, mul, offset, scaler float64) []float64 {
	lut := make([]float64, 512)
	for i := 0; i < len(lut); i++ {
		c := mul / ((0.01 * scaler * float64(i)) + offset)
		lut[i] = 1.0 + extmul*(c+extoff)
		assert(lut[i] >= 0.0)
		lut[i] *= lut[i]
	}
	return lut
}

func MaskX(delta float64) float64 {
	// PROFILER_FUNC;
	const extmul = 0.975741017749
	const extoff = -4.25328244168
	const offset = 0.454909521427
	const scaler = 0.0738288224836
	const mul = 20.8029176447
	lut := MakeMask(extmul, extoff, mul, offset, scaler)
	return InterpolateClampNegative(lut, len(lut), delta)
}

func MaskY(delta float64) float64 {
	//PROFILER_FUNC;
	const extmul = 0.373995618954
	const extoff = 1.5307267433
	const offset = 0.911952641929
	const scaler = 1.1731667845
	const mul = 16.2447033988
	lut := MakeMask(extmul, extoff, mul, offset, scaler)
	return InterpolateClampNegative(lut, len(lut), delta)
}

func MaskB(delta float64) float64 {
	//PROFILER_FUNC;
	const extmul = 0.61582234137
	const extoff = -4.25376118646
	const offset = 1.05105070921
	const scaler = 0.47434643535
	const mul = 31.1444967089
	lut := MakeMask(extmul, extoff, mul, offset, scaler)
	return InterpolateClampNegative(lut, len(lut), delta)
}

func MaskDcX(delta float64) float64 {
	//PROFILER_FUNC;
	const extmul = 1.79116943438
	const extoff = -3.86797479189
	const offset = 0.670960225853
	const scaler = 0.486575865525
	const mul = 20.4563479139
	lut := MakeMask(extmul, extoff, mul, offset, scaler)
	return InterpolateClampNegative(lut, len(lut), delta)
}

func MaskDcY(delta float64) float64 {
	//PROFILER_FUNC;
	const extmul = 0.212223514236
	const extoff = -3.65647120524
	const offset = 1.73396799447
	const scaler = 0.170392660501
	const mul = 21.6566724788
	lut := MakeMask(extmul, extoff, mul, offset, scaler)
	return InterpolateClampNegative(lut, len(lut), delta)
}

func MaskDcB(delta float64) float64 {
	//PROFILER_FUNC;
	const extmul = 0.349376011816
	const extoff = -0.894711072781
	const offset = 0.901647926679
	const scaler = 0.380086095024
	const mul = 18.0373825149
	lut := MakeMask(extmul, extoff, mul, offset, scaler)
	return InterpolateClampNegative(lut, len(lut), delta)
}

// Replaces values[x + y * xsize] with the minimum of the values in the
// square_size square with coordinates
//   x - offset .. x + square_size - offset - 1,
//   y - offset .. y + square_size - offset - 1.
func MinSquareVal(square_size, offset, xsize, ysize int, values []float32) {
	//PROFILER_FUNC;
	// offset is not negative and smaller than square_size.
	assert(offset < square_size)
	tmp := make([]float32, xsize*ysize)
	for y := 0; y < ysize; y++ {
		minh := 0
		if offset <= y {
			minh = y - offset
		}
		maxh := std_min(ysize, y+square_size-offset)
		for x := 0; x < xsize; x++ {
			min := values[x+minh*xsize]
			for j := minh + 1; j < maxh; j++ {
				min = std_minFloat32(min, values[x+j*xsize])
			}
			tmp[x+y*xsize] = float32(min)
		}
	}
	for x := 0; x < xsize; x++ {
		minw := 0
		if offset <= x {
			minw = x - offset
		}
		maxw := std_min(xsize, x+square_size-offset)
		for y := 0; y < ysize; y++ {
			min := tmp[minw+y*xsize]
			for j := minw + 1; j < maxw; j++ {
				min = std_minFloat32(min, tmp[j+y*xsize])
			}
			values[x+y*xsize] = float32(min)
		}
	}
}

// ===== Functions used by Mask only =====
func Average5x5(xsize, ysize int, diffs []float32) {
	//PROFILER_FUNC;
	if xsize < 4 || ysize < 4 {
		// TODO: Make this work for small dimensions as well.
		return
	}
	w := float32(0.679144890667)
	scale := float32(1.0 / (5.0 + 4*w))
	result := cloneSliceFloat32(diffs) // TODO PATAPON: is this intended as a copy?
	tmp0 := cloneSliceFloat32(diffs)   // TODO PATAPON: is this intended as a copy?
	tmp1 := cloneSliceFloat32(diffs)   // TODO PATAPON: is this intended as a copy?
	ScaleImage(float64(w), tmp1)
	for y := 0; y < ysize; y++ {
		row0 := y * xsize
		result[row0+1] += tmp0[row0]
		result[row0+0] += tmp0[row0+1]
		result[row0+2] += tmp0[row0+1]
		for x := 2; x < xsize-2; x++ {
			result[row0+x-1] += tmp0[row0+x]
			result[row0+x+1] += tmp0[row0+x]
		}
		result[row0+xsize-3] += tmp0[row0+xsize-2]
		result[row0+xsize-1] += tmp0[row0+xsize-2]
		result[row0+xsize-2] += tmp0[row0+xsize-1]
		if y > 0 {
			rowd1 := row0 - xsize
			result[rowd1+1] += tmp1[row0]
			result[rowd1+0] += tmp0[row0]
			for x := 1; x < xsize-1; x++ {
				result[rowd1+x+1] += tmp1[row0+x]
				result[rowd1+x+0] += tmp0[row0+x]
				result[rowd1+x-1] += tmp1[row0+x]
			}
			result[rowd1+xsize-1] += tmp0[row0+xsize-1]
			result[rowd1+xsize-2] += tmp1[row0+xsize-1]
		}
		if y+1 < ysize {
			rowu1 := row0 + xsize
			result[rowu1+1] += tmp1[row0]
			result[rowu1+0] += tmp0[row0]
			for x := 1; x < xsize-1; x++ {
				result[rowu1+x+1] += tmp1[row0+x]
				result[rowu1+x+0] += tmp0[row0+x]
				result[rowu1+x-1] += tmp1[row0+x]
			}
			result[rowu1+xsize-1] += tmp0[row0+xsize-1]
			result[rowu1+xsize-2] += tmp1[row0+xsize-1]
		}
	}
	copy(diffs, result)
	ScaleImage(float64(scale), diffs)
}

func DiffPrecompute(xyb0, xyb1 [][]float32, xsize, ysize int, mask [][]float32) {
	//PROFILER_FUNC;
	assert(len(mask) == 3)
	mask[0] = make([]float32, len(xyb0[0]))
	mask[1] = make([]float32, len(xyb0[0]))
	mask[2] = make([]float32, len(xyb0[0]))
	var valsh0, valsv0, valsh1, valsv1 [3]float64
	var ix2 int
	for y := 0; y < ysize; y++ {
		for x := 0; x < xsize; x++ {
			ix := x + xsize*y
			if x+1 < xsize {
				ix2 = ix + 1
			} else {
				ix2 = ix - 1
			}
			{
				x0 := float64(xyb0[0][ix] - xyb0[0][ix2])
				y0 := float64(xyb0[1][ix] - xyb0[1][ix2])
				z0 := float64(xyb0[2][ix] - xyb0[2][ix2])
				XybToVals(x0, y0, z0, &valsh0[0], &valsh0[1], &valsh0[2])
				x1 := float64(xyb1[0][ix] - xyb1[0][ix2])
				y1 := float64(xyb1[1][ix] - xyb1[1][ix2])
				z1 := float64(xyb1[2][ix] - xyb1[2][ix2])
				XybToVals(x1, y1, z1, &valsh1[0], &valsh1[1], &valsh1[2])
			}
			if y+1 < ysize {
				ix2 = ix + xsize
			} else {
				ix2 = ix - xsize
			}
			{
				x0 := float64(xyb0[0][ix] - xyb0[0][ix2])
				y0 := float64(xyb0[1][ix] - xyb0[1][ix2])
				z0 := float64(xyb0[2][ix] - xyb0[2][ix2])
				XybToVals(x0, y0, z0, &valsv0[0], &valsv0[1], &valsv0[2])
				x1 := float64(xyb1[0][ix] - xyb1[0][ix2])
				y1 := float64(xyb1[1][ix] - xyb1[1][ix2])
				z1 := float64(xyb1[2][ix] - xyb1[2][ix2])
				XybToVals(x1, y1, z1, &valsv1[0], &valsv1[1], &valsv1[2])
			}
			for i := 0; i < 3; i++ {
				sup0 := math.Abs(valsh0[i]) + math.Abs(valsv0[i])
				sup1 := math.Abs(valsh1[i]) + math.Abs(valsv1[i])
				m := std_minFloat64(sup0, sup1)
				mask[i][ix] = float32(m)
			}
		}
	}
}

func Mask(xyb0, xyb1 [][]float32, xsize, ysize int, mask, mask_dc [][]float32) {
	//PROFILER_FUNC;
	assert(len(mask) == 3)    // TODO PATAPON: always?
	assert(len(mask_dc) == 3) // TODO PATAPON: always?

	mask = mask[:3]
	for i := 0; i < 3; i++ {
		mask[i] = resizeSliceFloat32(mask[i], xsize*ysize)
	}
	DiffPrecompute(xyb0, xyb1, xsize, ysize, mask)
	for i := 0; i < 3; i++ {
		Average5x5(xsize, ysize, mask[i])
		MinSquareVal(4, 0, xsize, ysize, mask[i])
		sigma := [3]float64{
			9.65781083553,
			14.2644604355,
			4.53358927369,
		}
		Blur(xsize, ysize, mask[i], sigma[i], 0.0)
	}
	const w00 = 232.206464018
	const w11 = 22.9455222245
	const w22 = 503.962310606

	mask_dc = mask_dc[:3]
	for i := 0; i < 3; i++ {
		mask_dc[i] = make([]float32, xsize*ysize)
	}
	for y := 0; y < ysize; y++ {
		for x := 0; x < xsize; x++ {
			idx := y*xsize + x
			s0 := mask[0][idx]
			s1 := mask[1][idx]
			s2 := mask[2][idx]
			p0 := w00 * float64(s0)
			p1 := w11 * float64(s1)
			p2 := w22 * float64(s2)

			mask[0][idx] = float32(MaskX(p0))
			mask[1][idx] = float32(MaskY(p1))
			mask[2][idx] = float32(MaskB(p2))
			mask_dc[0][idx] = float32(MaskDcX(p0))
			mask_dc[1][idx] = float32(MaskDcY(p1))
			mask_dc[2][idx] = float32(MaskDcB(p2))
		}
	}
	for i := 0; i < 3; i++ {
		ScaleImage(kGlobalScale*kGlobalScale, mask[i])
		ScaleImage(kGlobalScale*kGlobalScale, mask_dc[i])
	}
}
