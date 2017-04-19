package guetzli_patapon

// TODO
// TODO
// TODO

type ButteraugliButteraugliComparator struct {
	xsize_, ysize_         int
	num_pixels_            int
	step_                  int
	res_xsize_, res_ysize_ int
}

func NewButteraugliButteraugliComparator(width, height, step int) *ButteraugliButteraugliComparator {
	return nil
}

func ButteraugliScoreFromDiffmap(distmap []float64) float64 {
	return -1
}

// Computes the butteraugli map between xyb0 and xyb1 and updates result.
// Both xyb0 and xyb1 are in opsin-dynamics space.
// NOTE: The xyb1 image is mutated by this function in-place.
func (*ButteraugliButteraugliComparator) DiffmapOpsinDynamicsImage(xyb0, xyb1 [][]float64, result []float64) {

}

// Compute values of local frequency and dc masking based on the activity
// in the two images.
func Mask(rgb0, rgb1 [][]float64,
	xsize, ysize int,
	mask, mask_dc [][]float64) {

}

// Computes difference metrics for one 8x8 block.
func ButteraugliBlockDiff(rgb0, rgb1 [192]float64,
	diff_xyb_dc, diff_xyb_ac, diff_xyb_edge_dc [3]float64) {

}

func OpsinDynamicsImage(xsize, ysize int, rgb [][]float64) {

}

func MaskHighIntensityChange(xsize, ysize int,
	c0, c1 [][]float64,
	rgb0, rgb1 [][]float64) {

}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


const(
 kInternalGoodQualityThreshold = 14.921561160295326
 kGlobalScale = 1.0 / kInternalGoodQualityThreshold
)

func DotProduct(u, v [3]float64) float64 {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

// Computes a horizontal convolution and transposes the result.
func Convolution(xsize, ysize int,
                        xstep int,
                        length, offset int,
                        multipliers []float32,
                        inp []float32,
                        border_ratio float64,
                        result []float32) {
  PROFILER_FUNC;
  weight_no_border := 0.0;
  for j := 0; j <= 2 * offset; j++ {
    weight_no_border += multipliers[j];
  }
  for x,ox := 0, 0; x < xsize; x,ox = x+xstep, ox+1 {
    minx := minusOr0(x, offset)
    maxx := std_min(xsize, x + length - offset) - 1;
    weight := 0.0;
    for j := minx; j <= maxx; j++ {
      weight += multipliers[j - x + offset];
    }
    // Interpolate linearly between the no-border scaling and border scaling.
    weight = (1.0 - border_ratio) * weight + border_ratio * weight_no_border;
    scale := 1.0 / weight;
    for y := 0; y < ysize; y++ {
      sum := 0.0;
      for j := minx; j <= maxx; j++ {
        sum += inp[y * xsize + j] * multipliers[j - x + offset];
      }
      result[ox * ysize + y] = static_cast<float>(sum * scale);
    }
  }
}

func Blur(xsize, ysize int, channel []float32, sigma, border_ratio float64) {
  PROFILER_FUNC;
  m := 2.25;  // Accuracy increases when m is increased.
  scaler := -1.0 / (2 * sigma * sigma);
  // For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
  diff := std_max(1, m * fabs(sigma));
  expn_size := 2 * diff + 1;
  expn := make([]float32, expn_size);
  for i := -diff; i <= diff; i++ {
    expn[i + diff] = float32(exp(scaler * i * i));
  }
  xstep := std_max(1, int(sigma / 3));
  ystep := xstep;
  dxsize := (xsize + xstep - 1) / xstep;
  dysize := (ysize + ystep - 1) / ystep;
  tmp := make([]float32, dxsize * ysize);
  Convolution(xsize, ysize, xstep, expn_size, diff, expn.data(), channel,
              border_ratio,
              tmp.data());
  float* output = channel;
  var downsampled_output []float32;
  if (xstep > 1) {
    downsampled_output = make(X, dxsize * dysize);
    output = downsampled_output.data();
  }
  Convolution(ysize, dxsize, ystep, expn_size, diff, expn.data(), tmp.data(),
              border_ratio, output);
  if (xstep > 1) {
    for y := 0; y < ysize; y++ {
      for x := 0; x < xsize; x++ {
        // TODO: Use correct rounding.
        channel[y * xsize + x] =
            downsampled_output[(y / ystep) * dxsize + (x / xstep)];
      }
    }
  }
}

// To change this to n, add the relevant FFTn function and kFFTnMapIndexTable.
const(
  kBlockEdge = 8
  kBlockSize = kBlockEdge * kBlockEdge
  kBlockEdgeHalf = kBlockEdge / 2
  kBlockHalf = kBlockEdge * kBlockEdgeHalf
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
  return csf8x8[:];
}

func MakeHighFreqColorDiffDx() []float64 {
  lut := make([]float64, 21);
  const off = 11.38708334481672
  const inc = 14.550189611520716
  lut[0] = 0.0;
  lut[1] = off;
  for i := 2; i < 21; i++ {
    lut[i] = lut[i - 1] + inc;
  }
  return lut;
}

func GetHighFreqColorDiffDx() []float64 {
  kLut := MakeHighFreqColorDiffDx();
  // TODO make only once?
  return kLut
}

func MakeHighFreqColorDiffDy() []float64 {
  lut := make([]float64, 21);
  const off = 1.4103373714040413;
  const inc = 0.7084088867024;
  lut[0] = 0.0;
  lut[1] = off;
  for i := 2; i < 21; i++ {
    lut[i] = lut[i - 1] + inc;
  }
  return lut;
}

func GetHighFreqColorDiffDy()  []float64  {
  kLut := MakeHighFreqColorDiffDy();
  // TODO make only once?
  return kLut
}

func MakeLowFreqColorDiffDy() []float64 {
  lut := make([]float64, 21);
  const inc = 5.2511644570349185;
  lut[0] = 0.0;
  for i := 1; i < 21; i++ {
    lut[i] = lut[i - 1] + inc;
  }
  return lut;
}

func GetLowFreqColorDiffDy() []float64 {
  kLut := MakeLowFreqColorDiffDy();
  // TODO make only once?
  return kLut.data();
}

func Interpolate(array []float64, size int, sx float64) float64 {
  ix := fabs(sx);
  assert(ix < 10000);
  baseix := static_cast<int>(ix);
  var res float64;
  if (baseix >= size - 1) {
    res = array[size - 1];
  } else {
    mix := ix - baseix;
    nextix := baseix + 1;
    res = array[baseix] + mix * (array[nextix] - array[baseix]);
  }
  if (sx < 0) {
  	res = -res;
  }
  return res;
}

func InterpolateClampNegative(array []float64, size int, sx float64) float64 {
  if (sx < 0) {
    sx = 0;
  }
  ix := fabs(sx);
  baseix := static_cast<int>(ix);
  var res float64;
  if (baseix >= size - 1) {
    res = array[size - 1];
  } else {
    mix := ix - baseix;
    nextix := baseix + 1;
    res = array[baseix] + mix * (array[nextix] - array[baseix]);
  }
  return res;
}

func RgbToXyb( r,  g,  b float64, valx, valy, valz *float64) {
  const a0 = 1.01611726948;
  const a1 = 0.982482243696;
  const a2 = 1.43571362627;
  const a3 = 0.896039849412;
  *valx = a0 * r - a1 * g;
  *valy = a2 * r + a3 * g;
  *valz = b;
}

func XybToVals(r,  g,  b float64, valx, valy, valz *float64) {
  const xmul = 0.758304045695;
  const ymul = 2.28148649801;
  const zmul = 1.87816926918;
  *valx = Interpolate(GetHighFreqColorDiffDx(), 21, x * xmul);
  *valy = Interpolate(GetHighFreqColorDiffDy(), 21, y * ymul);
  *valz = zmul * z;
}

// Rough psychovisual distance to gray for low frequency colors.
func XybLowFreqToVals(r,  g,  b float64, valx, valy, valz *float64) {
  const xmul = 6.64482198135;
  const ymul = 0.837846224276;
  const zmul = 7.34905756986;
  const y_to_z_mul = 0.0812519812628;
  z += y_to_z_mul * y;
  *valz = z * zmul;
  *valx = x * xmul;
  *valy = Interpolate(GetLowFreqColorDiffDy(), 21, y * ymul);
}

func RemoveRangeAroundZero(v, Range float64) float64 {
  if (v >= -Range && v < Range) {
    return 0;
  }
  if (v < 0) {
    return v + Range;
  } else {
    return v - Range;
  }
}

func XybDiffLowFreqSquaredAccumulate( r0,  g0,  b0,
                                      r1,  g1,  b1,
                                      factor float64, res []float64) {
  var valx0, valy0, valz0 float64;
  var valx1, valy1, valz1 float64;
  XybLowFreqToVals(r0, g0, b0, &valx0, &valy0, &valz0);
  if (r1 == 0.0 && g1 == 0.0 && b1 == 0.0) {
    PROFILER_ZONE("XybDiff r1=g1=b1=0");
    res[0] += factor * valx0 * valx0;
    res[1] += factor * valy0 * valy0;
    res[2] += factor * valz0 * valz0;
    return;
  }
  XybLowFreqToVals(r1, g1, b1, &valx1, &valy1, &valz1);
  // Approximate the distance of the colors by their respective distances
  // to gray.
  valx := valx0 - valx1;
  valy := valy0 - valy1;
  valz := valz0 - valz1;
  res[0] += factor * valx * valx;
  res[1] += factor * valy * valy;
  res[2] += factor * valz * valz;
}

type Complex complex128

func abssq(c Complex) float64 {
  return c.real * c.real + c.imag * c.imag;
}

func TransposeBlock(data []Complex) {
  for i := 0; i < kBlockEdge; i++ {
    for j := 0; j < i; j++ {
      data[kBlockEdge * i + j], data[kBlockEdge * j + i] = data[kBlockEdge * j + i], data[kBlockEdge * i + j]
    }
  }
}

//  D. J. Bernstein's Fast Fourier Transform algorithm on 4 elements.
func FFT4(a []Complex) {
  var t1, t2, t3, t4, t5, t6, t7, t8 float64;
  t5 = a[2].real;
  t1 = a[0].real - t5;
  t7 = a[3].real;
  t5 += a[0].real;
  t3 = a[1].real - t7;
  t7 += a[1].real;
  t8 = t5 + t7;
  a[0].real = t8;
  t5 -= t7;
  a[1].real = t5;
  t6 = a[2].imag;
  t2 = a[0].imag - t6;
  t6 += a[0].imag;
  t5 = a[3].imag;
  a[2].imag = t2 + t3;
  t2 -= t3;
  a[3].imag = t2;
  t4 = a[1].imag - t5;
  a[3].real = t1 + t4;
  t1 -= t4;
  a[2].real = t1;
  t5 += a[1].imag;
  a[0].imag = t6 + t5;
  t6 -= t5;
  a[1].imag = t6;
}

const kSqrtHalf = 0.70710678118654752440084436210484903;

//  D. J. Bernstein's Fast Fourier Transform algorithm on 8 elements.
func FFT8(a []Complex) {
  var t1, t2, t3, t4, t5, t6, t7, t8 float64;

  t7 = a[4].imag;
  t4 = a[0].imag - t7;
  t7 += a[0].imag;
  a[0].imag = t7;

  t8 = a[6].real;
  t5 = a[2].real - t8;
  t8 += a[2].real;
  a[2].real = t8;

  t7 = a[6].imag;
  a[6].imag = t4 - t5;
  t4 += t5;
  a[4].imag = t4;

  t6 = a[2].imag - t7;
  t7 += a[2].imag;
  a[2].imag = t7;

  t8 = a[4].real;
  t3 = a[0].real - t8;
  t8 += a[0].real;
  a[0].real = t8;

  a[4].real = t3 - t6;
  t3 += t6;
  a[6].real = t3;

  t7 = a[5].real;
  t3 = a[1].real - t7;
  t7 += a[1].real;
  a[1].real = t7;

  t8 = a[7].imag;
  t6 = a[3].imag - t8;
  t8 += a[3].imag;
  a[3].imag = t8;
  t1 = t3 - t6;
  t3 += t6;

  t7 = a[5].imag;
  t4 = a[1].imag - t7;
  t7 += a[1].imag;
  a[1].imag = t7;

  t8 = a[7].real;
  t5 = a[3].real - t8;
  t8 += a[3].real;
  a[3].real = t8;

  t2 = t4 - t5;
  t4 += t5;

  t6 = t1 - t4;
  t8 = kSqrtHalf;
  t6 *= t8;
  a[5].real = a[4].real - t6;
  t1 += t4;
  t1 *= t8;
  a[5].imag = a[4].imag - t1;
  t6 += a[4].real;
  a[4].real = t6;
  t1 += a[4].imag;
  a[4].imag = t1;

  t5 = t2 - t3;
  t5 *= t8;
  a[7].imag = a[6].imag - t5;
  t2 += t3;
  t2 *= t8;
  a[7].real = a[6].real - t2;
  t2 += a[6].real;
  a[6].real = t2;
  t5 += a[6].imag;
  a[6].imag = t5;

  FFT4(a);

  // Reorder to the correct output order.
  // TODO: Modify the above computation so that this is not needed.
  tmp := a[2];
  a[2] = a[3];
  a[3] = a[5];
  a[5] = a[7];
  a[7] = a[4];
  a[4] = a[1];
  a[1] = a[6];
  a[6] = tmp;
}

// Same as FFT8, but all inputs are real.
// TODO: Since this does not need to be in-place, maybe there is a
// faster FFT than this one, which is derived from DJB's in-place complex FFT.
func RealFFT8(in []float64, out []Complex) {
  var t1, t2, t3, t5, t6, t7, t8 float64;
  t8 = in[6];
  t5 = in[2] - t8;
  t8 += in[2];
  out[2].real = t8;
  out[6].imag = -t5;
  out[4].imag = t5;
  t8 = in[4];
  t3 = in[0] - t8;
  t8 += in[0];
  out[0].real = t8;
  out[4].real = t3;
  out[6].real = t3;
  t7 = in[5];
  t3 = in[1] - t7;
  t7 += in[1];
  out[1].real = t7;
  t8 = in[7];
  t5 = in[3] - t8;
  t8 += in[3];
  out[3].real = t8;
  t2 = -t5;
  t6 = t3 - t5;
  t8 = kSqrtHalf;
  t6 *= t8;
  out[5].real = out[4].real - t6;
  t1 = t3 + t5;
  t1 *= t8;
  out[5].imag = out[4].imag - t1;
  t6 += out[4].real;
  out[4].real = t6;
  t1 += out[4].imag;
  out[4].imag = t1;
  t5 = t2 - t3;
  t5 *= t8;
  out[7].imag = out[6].imag - t5;
  t2 += t3;
  t2 *= t8;
  out[7].real = out[6].real - t2;
  t2 += out[6].real;
  out[6].real = t2;
  t5 += out[6].imag;
  out[6].imag = t5;
  t5 = out[2].real;
  t1 = out[0].real - t5;
  t7 = out[3].real;
  t5 += out[0].real;
  t3 = out[1].real - t7;
  t7 += out[1].real;
  t8 = t5 + t7;
  out[0].real = t8;
  t5 -= t7;
  out[1].real = t5;
  out[2].imag = t3;
  out[3].imag = -t3;
  out[3].real = t1;
  out[2].real = t1;
  out[0].imag = 0;
  out[1].imag = 0;

  // Reorder to the correct output order.
  // TODO: Modify the above computation so that this is not needed.
  tmp := out[2];
  out[2] = out[3];
  out[3] = out[5];
  out[5] = out[7];
  out[7] = out[4];
  out[4] = out[1];
  out[1] = out[6];
  out[6] = tmp;
}

// Fills in block[kBlockEdgeHalf..(kBlockHalf+kBlockEdgeHalf)], and leaves the
// rest unmodified.
func ButteraugliFFTSquared(block []float64) {
  global_mul := 0.000064;
  block_c := make([]Complex, kBlockSize);
  assert(kBlockEdge == 8);
  for y := 0; y < kBlockEdge; y++ {
    RealFFT8(block + y * kBlockEdge, block_c + y * kBlockEdge);
  }
  TransposeBlock(block_c);
  r0 := make(float64, kBlockEdge);
  r1 := make(float64, kBlockEdge);
  for x := 0; x < kBlockEdge; x++ {
    r0[x] = block_c[x].real;
    r1[x] = block_c[kBlockHalf + x].real;
  }
  RealFFT8(r0, block_c);
  RealFFT8(r1, block_c + kBlockHalf);
  for y := 1; y < kBlockEdgeHalf; y++ {
    FFT8(block_c + y * kBlockEdge);
  }
  for i := kBlockEdgeHalf; i < kBlockHalf + kBlockEdgeHalf + 1; i++ {
    block[i] = abssq(block_c[i]);
    block[i] *= global_mul;
  }
}

// Computes 8x8 FFT of each channel of xyb0 and xyb1 and adds the total squared
// 3-dimensional xybdiff of the two blocks to diff_xyb_{dc,ac} and the average
// diff on the edges to diff_xyb_edge_dc.
func ButteraugliBlockDiff(xyb0, xyb1 []float64,
                          diff_xyb_dc, diff_xyb_ac, diff_xyb_edge_dc []float64) {
  // PROFILER_FUNC;
  csf8x8 := GetContrastSensitivityMatrix();

  var avgdiff_xyb [3]float64
  var avgdiff_edge [3][4]float64
  for i := 0; i < 3 * kBlockSize; i++ {
    diff_xyb := xyb0[i] - xyb1[i];
    c := i / kBlockSize;
    avgdiff_xyb[c] += diff_xyb / kBlockSize;
    k := i % kBlockSize;
    kx := k % kBlockEdge;
    ky := k / kBlockEdge;
    var h_edge_idx, v_edge_idx int 
    if ky == 0 {
		h_edge_idx = 1
    }else if ky == 7 {
		h_edge_idx = 3
    }else{
		h_edge_idx = -1
    }
    if kx == 0 {
		v_edge_idx = 1
    }else if kx == 7 {
		v_edge_idx = 3
    }else{
		v_edge_idx = -1
    }
    if (h_edge_idx >= 0) {
      avgdiff_edge[c][h_edge_idx] += diff_xyb / kBlockEdge;
    }
    if (v_edge_idx >= 0) {
      avgdiff_edge[c][v_edge_idx] += diff_xyb / kBlockEdge;
    }
  }
  XybDiffLowFreqSquaredAccumulate(avgdiff_xyb[0],
                                  avgdiff_xyb[1],
                                  avgdiff_xyb[2],
                                  0, 0, 0, csf8x8[0],
                                  diff_xyb_dc);
  for i := 0; i < 4; i++ {
    XybDiffLowFreqSquaredAccumulate(avgdiff_edge[0][i],
                                    avgdiff_edge[1][i],
                                    avgdiff_edge[2][i],
                                    0, 0, 0, csf8x8[0],
                                    diff_xyb_edge_dc);
  }

  double* xyb_avg = xyb0;
  double* xyb_halfdiff = xyb1;
  for i := 0; i < 3 * kBlockSize; i++ {
    avg := (xyb0[i] + xyb1[i])/2;
    halfdiff := (xyb0[i] - xyb1[i])/2;
    xyb_avg[i] = avg;
    xyb_halfdiff[i] = halfdiff;
  }
  double *y_avg = &xyb_avg[kBlockSize];
  double *x_halfdiff_squared = &xyb_halfdiff[0];
  double *y_halfdiff = &xyb_halfdiff[kBlockSize];
  double *z_halfdiff_squared = &xyb_halfdiff[2 * kBlockSize];
  ButteraugliFFTSquared(y_avg);
  ButteraugliFFTSquared(x_halfdiff_squared);
  ButteraugliFFTSquared(y_halfdiff);
  ButteraugliFFTSquared(z_halfdiff_squared);

  const xmul = 64.8;
  const ymul = 1.753123908348329;
  const ymul2 = 1.51983458269;
  const zmul = 2.4;

  for i := kBlockEdgeHalf; i < kBlockHalf + kBlockEdgeHalf + 1; i++ {
    d := csf8x8[i];
    diff_xyb_ac[0] += d * xmul * x_halfdiff_squared[i];
    diff_xyb_ac[2] += d * zmul * z_halfdiff_squared[i];

    y_avg[i] = sqrt(y_avg[i]);
    y_halfdiff[i] = sqrt(y_halfdiff[i]);
    y0 := y_avg[i] - y_halfdiff[i];
    y1 := y_avg[i] + y_halfdiff[i];
    // Remove the impact of small absolute values.
    // This improves the behavior with flat noise.
    const ylimit = 0.04;
    y0 = RemoveRangeAroundZero(y0, ylimit);
    y1 = RemoveRangeAroundZero(y1, ylimit);
    if (y0 != y1) {
      valy0 := Interpolate(GetHighFreqColorDiffDy(), 21, y0 * ymul2);
      valy1 := Interpolate(GetHighFreqColorDiffDy(), 21, y1 * ymul2);
      valy := ymul * (valy0 - valy1);
      diff_xyb_ac[1] += d * valy * valy;
    }
  }
}

// Low frequency edge detectors.
// Two edge detectors are applied in each corner of the 8x8 square.
// The squared 3-dimensional error vector is added to diff_xyb.
func Butteraugli8x8CornerEdgeDetectorDiff(
    pos_x,    pos_y,    xsize,    ysize int,
    blurred0, blurred1 [][]float32,
    diff_xyb []float64) {
  // PROFILER_FUNC;
  local_count := 0;
  var local_xyb [3]float64
  const w = 0.711100840192;
  for k := 0; k < 4; k++ {
    step := 3;
    offset := [4][2]int{ { 0, 0 }, { 0, 7 }, { 7, 0 }, { 7, 7 } };
    x := pos_x + offset[k][0];
    y := pos_y + offset[k][1];
    if (x >= step && x + step < xsize) {
      ix := y * xsize + (x - step);
      ix2 := ix + 2 * step;
      XybDiffLowFreqSquaredAccumulate(
          w * (blurred0[0][ix] - blurred0[0][ix2]),
          w * (blurred0[1][ix] - blurred0[1][ix2]),
          w * (blurred0[2][ix] - blurred0[2][ix2]),
          w * (blurred1[0][ix] - blurred1[0][ix2]),
          w * (blurred1[1][ix] - blurred1[1][ix2]),
          w * (blurred1[2][ix] - blurred1[2][ix2]),
          1.0, local_xyb);
      local_count++;
    }
    if (y >= step && y + step < ysize) {
      ix := (y - step) * xsize + x;
      ix2 := ix + 2 * step * xsize;
      XybDiffLowFreqSquaredAccumulate(
          w * (blurred0[0][ix] - blurred0[0][ix2]),
          w * (blurred0[1][ix] - blurred0[1][ix2]),
          w * (blurred0[2][ix] - blurred0[2][ix2]),
          w * (blurred1[0][ix] - blurred1[0][ix2]),
          w * (blurred1[1][ix] - blurred1[1][ix2]),
          w * (blurred1[2][ix] - blurred1[2][ix2]),
          1.0, local_xyb);
      local_count++;
    }
  }
  const weight = 0.01617112696;
  mul := weight * 8.0 / local_count;
  for i := 0; i < 3; i++ {
    diff_xyb[i] += mul * local_xyb[i];
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
  };

func OpsinAbsorbance(in, out []float64) {
  mix := opsinAbsorbance;
  out[0] = mix[0] * in[0] + mix[1] * in[1] + mix[2] * in[2] + mix[3];
  out[1] = mix[4] * in[0] + mix[5] * in[1] + mix[6] * in[2] + mix[7];
  out[2] = mix[8] * in[0] + mix[9] * in[1] + mix[10] * in[2] + mix[11];
}

func GammaMinArg() float64 {
  var in, out [3]float64
  OpsinAbsorbance(in[:], out[:]);
  return std_min(out[0], std_min(out[1], out[2]));
}

func GammaMaxArg() float64 {
  in := [3]float64{ 255.0, 255.0, 255.0 };
  var out [3]float64
  OpsinAbsorbance(in[:], out[:]);
  return std_max(out[0], std_max(out[1], out[2]));
}

func NewButteraugliComparator(xsize, ysize, step int) *ButteraugliButteraugliComparator {
	return &ButteraugliButteraugliComparator{
      xsize_: xsize,
      ysize_: ysize,
      num_pixels_: xsize * ysize,
      step_: step,
      res_xsize_: (xsize + step - 1) / step,
      res_ysize_: (ysize + step - 1) / step,
    }
  	assert(step <= 4);
}

func MaskHighIntensityChange(xsize, ysize int, c0,c1,xyb0,xyb1 [][]float32) {
  // PROFILER_FUNC;
  for y := 0; y < ysize; y++ {
    for x := 0; x < xsize; x++ {
      ix := y * xsize + x;
      ave := [3]float64{
        (c0[0][ix] + c1[0][ix]) * 0.5,
        (c0[1][ix] + c1[1][ix]) * 0.5,
        (c0[2][ix] + c1[2][ix]) * 0.5,
      };
      sqr_max_diff := -1;
      {
        offset := [4]int{ -1, 1, -static_cast<int>(xsize), static_cast<int>(xsize) };
        border := [4]int{ x == 0, x + 1 == xsize, y == 0, y + 1 == ysize };
        for dir := 0; dir < 4; dir++ {
          if (border[dir]) {
            continue;
          }
          ix2 := ix + offset[dir];
          diff := 0.5 * (c0[1][ix2] + c1[1][ix2]) - ave[1];
          diff *= diff;
          if (sqr_max_diff < diff) {
            sqr_max_diff = diff;
          }
        }
      }
      const kReductionX = 275.19165240059317;
      const kReductionY = 18599.41286306991;
      const kReductionZ = 410.8995306951065;
      const kChromaBalance = 106.95800948271017;
      chroma_scale := kChromaBalance / (ave[1] + kChromaBalance);

      mix := [3]float64{
        chroma_scale * kReductionX / (sqr_max_diff + kReductionX),
        kReductionY / (sqr_max_diff + kReductionY),
        chroma_scale * kReductionZ / (sqr_max_diff + kReductionZ),
      };
      // Interpolate lineraly between the average color and the actual
      // color -- to reduce the importance of this pixel.
      for i := 0; i < 3; i++ {
        xyb0[i][ix] = static_cast<float>(mix[i] * c0[i][ix] + (1 - mix[i]) * ave[i]);
        xyb1[i][ix] = static_cast<float>(mix[i] * c1[i][ix] + (1 - mix[i]) * ave[i]);
      }
    }
  }
}

func SimpleGamma(v float64) float64 {
  const kGamma = 0.387494322593;
  const limit = 43.01745241042018;
  bright := v - limit;
  if (bright >= 0) {
    const mul = 0.0383723643799;
    v -= bright * mul;
  }
  const limit2 = 94.68634353321337;
  bright2 := v - limit2;
  if (bright2 >= 0) {
    const mul = 0.22885405968;
    v -= bright2 * mul;
  }
  const offset = 0.156775786057;
  const scale = 8.898059160493739;
  retval := scale * (offset + pow(v, kGamma));
  return retval;
}

// Polynomial evaluation via Clenshaw's scheme (similar to Horner's).
// Template enables compile-time unrolling of the recursion, but must reside
// outside of a class due to the specialization.
template <int INDEX>
func ClenshawRecursion(const double x, const double *coefficients,
                                     double *b1, double *b2) {
  const x_b1 := x * (*b1);
  const t := (x_b1 + x_b1) - (*b2) + coefficients[INDEX];
  *b2 = *b1;
  *b1 = t;

  ClenshawRecursion<INDEX - 1>(x, coefficients, b1, b2);
}

// Base case
template <>
inline func ClenshawRecursion<0>(const double x, const double *coefficients,
                                 double *b1, double *b2) {
  const x_b1 := x * (*b1);
  // The final iteration differs - no 2 * x_b1 here.
  *b1 = x_b1 - (*b2) + coefficients[0];
}

// Rational polynomial := dividing two polynomial evaluations. These are easier
// to find than minimax polynomials.
struct RationalPolynomial {
  template <int N>
  static double EvaluatePolynomial(const double x,
                                   const double (&coefficients)[N]) {
    b1 := 0.0;
    b2 := 0.0;
    ClenshawRecursion<N - 1>(x, coefficients, &b1, &b2);
    return b1;
  }

  // Evaluates the polynomial at x (in [min_value, max_value]).
  inline double operator()(const float x) const {
    // First normalize to [0, 1].
    const x01 := (x - min_value) / (max_value - min_value);
    // And then to [-1, 1] domain of Chebyshev polynomials.
    const xc := 2.0 * x01 - 1.0;

    const yp := EvaluatePolynomial(xc, p);
    const yq := EvaluatePolynomial(xc, q);
    if (yq == 0.0) return 0.0;
    return static_cast<float>(yp / yq);
  }

  // Domain of the polynomials; they are undefined elsewhere.
  double min_value;
  double max_value;

  // Coefficients of T_n (Chebyshev polynomials of the first kind).
  // Degree 5/5 is a compromise between accuracy (0.1%) and numerical stability.
  double p[5 + 1];
  double q[5 + 1];
};

float GammaPolynomial(float value) {
  // Generated by gamma_polynomial.m from equispaced x/gamma(x) samples.
  static const RationalPolynomial r = {
  0.770000000000000, 274.579999999999984,
  {
    881.979476556478289, 1496.058452015812463, 908.662212739659481,
    373.566100223287378, 85.840860336314364, 6.683258861509244,
  },
  {
    12.262350348616792, 20.557285797683576, 12.161463238367844,
    4.711532733641639, 0.899112889751053, 0.035662329617191,
  }};
  return static_cast<float>(r(value));
}

double Gamma(double v) {
  // return SimpleGamma(v);
  return GammaPolynomial(static_cast<float>(v));
}

func OpsinDynamicsImage(size_t xsize, size_t ysize,
                        [][]float32 &rgb) {
  PROFILER_FUNC;
  [][]float32 blurred = rgb;
  const kSigma = 1.1;
  for i := 0; i < 3; i++ {
    Blur(xsize, ysize, blurred[i].data(), kSigma, 0.0);
  }
  for i := 0; i < rgb[0].size(); i++ {
    double sensitivity[3];
    {
      // Calculate sensitivity[3] based on the smoothed image gamma derivative.
      double pre_rgb[3] = { blurred[0][i], blurred[1][i], blurred[2][i] };
      double pre_mixed[3];
      OpsinAbsorbance(pre_rgb, pre_mixed);
      sensitivity[0] = Gamma(pre_mixed[0]) / pre_mixed[0];
      sensitivity[1] = Gamma(pre_mixed[1]) / pre_mixed[1];
      sensitivity[2] = Gamma(pre_mixed[2]) / pre_mixed[2];
    }
    double cur_rgb[3] = { rgb[0][i],  rgb[1][i],  rgb[2][i] };
    double cur_mixed[3];
    OpsinAbsorbance(cur_rgb, cur_mixed);
    cur_mixed[0] *= sensitivity[0];
    cur_mixed[1] *= sensitivity[1];
    cur_mixed[2] *= sensitivity[2];
    double x, y, z;
    RgbToXyb(cur_mixed[0], cur_mixed[1], cur_mixed[2], &x, &y, &z);
    rgb[0][i] = static_cast<float>(x);
    rgb[1][i] = static_cast<float>(y);
    rgb[2][i] = static_cast<float>(z);
  }
}

func ScaleImage(double scale, []float32 *result) {
  PROFILER_FUNC;
  for i := 0; i < result.size(); i++ {
    (*result)[i] *= static_cast<float>(scale);
  }
}

// Making a cluster of local errors to be more impactful than
// just a single error.
func CalculateDiffmap(const size_t xsize, const size_t ysize,
                      const size_t step,
                      []float32* diffmap) {
  PROFILER_FUNC;
  // Shift the diffmap more correctly above the pixels, from 2.5 pixels to 0.5
  // pixels distance over the original image. The border of 2 pixels on top and
  // left side and 3 pixels on right and bottom side are zeroed, but these
  // values have no meaning, they only exist to keep the result map the same
  // size as the input images.
  s2 := (8 - step) / 2;
  {
    // Upsample and take square root.
    []float32 diffmap_out(xsize * ysize);
    const size_t res_xsize = (xsize + step - 1) / step;
    for res_y := 0; res_y + 8 - step < ysize; res_y += step) {
      for res_x := 0; res_x + 8 - step < xsize; res_x += step) {
        size_t res_ix = (res_y * res_xsize + res_x) / step;
        float orig_val = (*diffmap)[res_ix];
        constexpr float kInitialSlope = 100;
        // TODO(b/29974893): Until that is fixed do not call sqrt on very small
        // numbers.
        val := orig_val < (1.0 / (kInitialSlope * kInitialSlope))
                                ? kInitialSlope * orig_val
                                : std::sqrt(orig_val);
        for off_y := 0; off_y < step; off_y++ {
          for off_x := 0; off_x < step; off_x++ {
            diffmap_out[(res_y + off_y + s2) * xsize +
                        res_x + off_x + s2] = val;
          }
        }
      }
    }
    *diffmap = diffmap_out;
  }
  {
    const kSigma = 8.8510880283;
    const mul1 = 24.8235314874;
    const scale = 1.0 / (1.0 + mul1);
    s := 8 - step;
    []float32 blurred((xsize - s) * (ysize - s));
    for y := 0; y < ysize - s; y++ {
      for x := 0; x < xsize - s; x++ {
        blurred[y * (xsize - s) + x] = (*diffmap)[(y + s2) * xsize + x + s2];
      }
    }
    const border_ratio = 0.03027655136;
    Blur(xsize - s, ysize - s, blurred.data(), kSigma, border_ratio);
    for y := 0; y < ysize - s; y++ {
      for x := 0; x < xsize - s; x++ {
        (*diffmap)[(y + s2) * xsize + x + s2]
            += static_cast<float>(mul1) * blurred[y * (xsize - s) + x];
      }
    }
    ScaleImage(scale, diffmap);
  }
}

func ButteraugliComparator::DiffmapOpsinDynamicsImage(
    const std::vector<[]float32> &xyb0_arg,
    std::vector<[]float32> &xyb1,
    []float32 &result) {
  if (xsize_ < 8 || ysize_ < 8) return;
  auto xyb0 = xyb0_arg;
  {
    auto xyb1_c = xyb1;
    MaskHighIntensityChange(xsize_, ysize_, xyb0_arg, xyb1_c, xyb0, xyb1);
  }
  assert(8 <= xsize_);
  for i := 0; i < 3; i++) {
    assert(xyb0[i].size() == num_pixels_);
    assert(xyb1[i].size() == num_pixels_);
  }
  []float32 edge_detector_map(3 * res_xsize_ * res_ysize_);
  EdgeDetectorMap(xyb0, xyb1, &edge_detector_map);
  []float32 block_diff_dc(3 * res_xsize_ * res_ysize_);
  []float32 block_diff_ac(3 * res_xsize_ * res_ysize_);
  BlockDiffMap(xyb0, xyb1, &block_diff_dc, &block_diff_ac);
  EdgeDetectorLowFreq(xyb0, xyb1, &block_diff_ac);
  {
    [][]float32 mask_xyb(3);
    [][]float32 mask_xyb_dc(3);
    Mask(xyb0, xyb1, xsize_, ysize_, &mask_xyb, &mask_xyb_dc);
    CombineChannels(mask_xyb, mask_xyb_dc, block_diff_dc, block_diff_ac,
                    edge_detector_map, &result);
  }
  CalculateDiffmap(xsize_, ysize_, step_, &result);
}

func ButteraugliComparator::BlockDiffMap(
    [][]float32 &xyb0,
    [][]float32 &xyb1,
    []float32* block_diff_dc,
    []float32* block_diff_ac) {
  PROFILER_FUNC;
  for res_y := 0; res_y + (kBlockEdge - step_ - 1) < ysize_;
       res_y += step_) {
    for res_x := 0; res_x + (kBlockEdge - step_ - 1) < xsize_;
         res_x += step_) {
      size_t res_ix = (res_y * res_xsize_ + res_x) / step_;
      size_t offset = (std_min(res_y, ysize_ - 8) * xsize_ +
                       std_min(res_x, xsize_ - 8));
      double block0[3 * kBlockEdge * kBlockEdge];
      double block1[3 * kBlockEdge * kBlockEdge];
      for i := 0; i < 3; i++ {
        double *m0 = &block0[i * kBlockEdge * kBlockEdge];
        double *m1 = &block1[i * kBlockEdge * kBlockEdge];
        for y := 0; y < kBlockEdge; y++) {
          for x := 0; x < kBlockEdge; x++) {
            m0[kBlockEdge * y + x] = xyb0[i][offset + y * xsize_ + x];
            m1[kBlockEdge * y + x] = xyb1[i][offset + y * xsize_ + x];
          }
        }
      }
      double diff_xyb_dc[3] = { 0.0 };
      double diff_xyb_ac[3] = { 0.0 };
      double diff_xyb_edge_dc[3] = { 0.0 };
      ButteraugliBlockDiff(block0, block1,
                           diff_xyb_dc, diff_xyb_ac, diff_xyb_edge_dc);
      for i := 0; i < 3; i++ {
        (*block_diff_dc)[3 * res_ix + i] = static_cast<float>(diff_xyb_dc[i]);
        (*block_diff_ac)[3 * res_ix + i] = static_cast<float>(diff_xyb_ac[i]);
      }
    }
  }
}

func ButteraugliComparator::EdgeDetectorMap(
    [][]float32 &xyb0,
    [][]float32 &xyb1,
    []float32* edge_detector_map) {
  PROFILER_FUNC;
  static const double kSigma[3] = {
    1.5,
    0.586,
    0.4,
  };
  [][]float32 blurred0(xyb0);
  [][]float32 blurred1(xyb1);
  for i := 0; i < 3; i++) {
    Blur(xsize_, ysize_, blurred0[i].data(), kSigma[i], 0.0);
    Blur(xsize_, ysize_, blurred1[i].data(), kSigma[i], 0.0);
  }
  for res_y := 0; res_y + (8 - step_) < ysize_; res_y += step_) {
    for res_x := 0; res_x + (8 - step_) < xsize_; res_x += step_) {
      size_t res_ix = (res_y * res_xsize_ + res_x) / step_;
      double diff_xyb[3] = { 0.0 };
      Butteraugli8x8CornerEdgeDetectorDiff(std_min(res_x, xsize_ - 8),
                                           std_min(res_y, ysize_ - 8),
                                           xsize_, ysize_,
                                           blurred0, blurred1,
                                           diff_xyb);
      for i := 0; i < 3; i++ {
        (*edge_detector_map)[3 * res_ix + i] = static_cast<float>(diff_xyb[i]);
      }
    }
  }
}

func ButteraugliComparator::EdgeDetectorLowFreq(
     xyb0, xyb1 [][]float32,
    []float32* block_diff_ac) {
  PROFILER_FUNC;
  const kSigma = 14;
  const kMul = 10;
  [][]float32 blurred0(xyb0);
  [][]float32 blurred1(xyb1);
  for i := 0; i < 3; i++) {
    Blur(xsize_, ysize_, blurred0[i].data(), kSigma, 0.0);
    Blur(xsize_, ysize_, blurred1[i].data(), kSigma, 0.0);
  }
  step := 8;
  for y := 0; y + step < ysize_; y += step_) {
    resy := y / step_;
    resx := step / step_;
    for x := 0; x + step < xsize_; x += step_, resx++) {
      ix := y * xsize_ + x;
      res_ix := resy * res_xsize_ + resx;
      double diff[4][3];
      for i := 0; i < 3; i++ {
        ix2 := ix + 8;
        diff[0][i] =
            ((blurred1[i][ix] - blurred0[i][ix]) +
             (blurred0[i][ix2] - blurred1[i][ix2]));
        ix2 = ix + 8 * xsize_;
        diff[1][i] =
            ((blurred1[i][ix] - blurred0[i][ix]) +
             (blurred0[i][ix2] - blurred1[i][ix2]));
        ix2 = ix + 6 * xsize_ + 6;
        diff[2][i] =
            ((blurred1[i][ix] - blurred0[i][ix]) +
             (blurred0[i][ix2] - blurred1[i][ix2]));
        ix2 = ix + 6 * xsize_ - 6;
        diff[3][i] = x < step ? 0 :
            ((blurred1[i][ix] - blurred0[i][ix]) +
             (blurred0[i][ix2] - blurred1[i][ix2]));
      }
      double max_diff_xyb[3] = { 0 };
      for k := 0; k < 4; k++ {
        double diff_xyb[3] = { 0 };
        XybDiffLowFreqSquaredAccumulate(diff[k][0], diff[k][1], diff[k][2],
                                        0, 0, 0, 1.0,
                                        diff_xyb);
        for i := 0; i < 3; i++ {
          max_diff_xyb[i] = std_max<double>(max_diff_xyb[i], diff_xyb[i]);
        }
      }
      for i := 0; i < 3; i++ {
        (*block_diff_ac)[3 * res_ix + i] += static_cast<float>(kMul * max_diff_xyb[i]);
      }
    }
  }
}

func ButteraugliComparator::CombineChannels(
    mask_xyb, mask_xyb_dc [][]float32,
    block_diff_dc, block_diff_ac, edge_detector_map []float32,
    result []float32) {
  // PROFILER_FUNC;
  result = make([]X, res_xsize_ * res_ysize_);
  for res_y := 0; res_y + (8 - step_) < ysize_; res_y += step_) {
    for res_x := 0; res_x + (8 - step_) < xsize_; res_x += step_) {
      size_t res_ix = (res_y * res_xsize_ + res_x) / step_;
      double mask[3];
      double dc_mask[3];
      for i := 0; i < 3; i++ {
        mask[i] = mask_xyb[i][(res_y + 3) * xsize_ + (res_x + 3)];
        dc_mask[i] = mask_xyb_dc[i][(res_y + 3) * xsize_ + (res_x + 3)];
      }
      (*result)[res_ix] = static_cast<float>(
           DotProduct(&block_diff_dc[3 * res_ix], dc_mask) +
           DotProduct(&block_diff_ac[3 * res_ix], mask) +
           DotProduct(&edge_detector_map[3 * res_ix], mask));
    }
  }
}

double ButteraugliScoreFromDiffmap([]float32 diffmap) {
  PROFILER_FUNC;
  float retval = 0.0f;
  for ix := 0; ix < diffmap.size(); ix++ {
    retval = std_max(retval, diffmap[ix]);
  }
  return retval;
}

static std::array<double, 512> MakeMask(
    double extmul, double extoff,
    double mul, double offset,
    double scaler) {
  std::array<double, 512> lut;
  for i := 0; i < lut.size(); i++ {
    const c := mul / ((0.01 * scaler * i) + offset);
    lut[i] = 1.0 + extmul * (c + extoff);
    assert(lut[i] >= 0.0);
    lut[i] *= lut[i];
  }
  return lut;
}

double MaskX(double delta) {
  PROFILER_FUNC;
  const extmul = 0.975741017749;
  const extoff = -4.25328244168;
  const offset = 0.454909521427;
  const scaler = 0.0738288224836;
  const mul = 20.8029176447;
  static const std::array<double, 512> lut =
                MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskY(double delta) {
  PROFILER_FUNC;
  const extmul = 0.373995618954;
  const extoff = 1.5307267433;
  const offset = 0.911952641929;
  const scaler = 1.1731667845;
  const mul = 16.2447033988;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskB(double delta) {
  PROFILER_FUNC;
  const extmul = 0.61582234137;
  const extoff = -4.25376118646;
  const offset = 1.05105070921;
  const scaler = 0.47434643535;
  const mul = 31.1444967089;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcX(double delta) {
  PROFILER_FUNC;
  const extmul = 1.79116943438;
  const extoff = -3.86797479189;
  const offset = 0.670960225853;
  const scaler = 0.486575865525;
  const mul = 20.4563479139;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcY(double delta) {
  PROFILER_FUNC;
  const extmul = 0.212223514236;
  const extoff = -3.65647120524;
  const offset = 1.73396799447;
  const scaler = 0.170392660501;
  const mul = 21.6566724788;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

double MaskDcB(double delta) {
  PROFILER_FUNC;
  const extmul = 0.349376011816;
  const extoff = -0.894711072781;
  const offset = 0.901647926679;
  const scaler = 0.380086095024;
  const mul = 18.0373825149;
  static const std::array<double, 512> lut =
      MakeMask(extmul, extoff, mul, offset, scaler);
  return InterpolateClampNegative(lut.data(), lut.size(), delta);
}

// Replaces values[x + y * xsize] with the minimum of the values in the
// square_size square with coordinates
//   x - offset .. x + square_size - offset - 1,
//   y - offset .. y + square_size - offset - 1.
func MinSquareVal(size_t square_size, size_t offset,
                  size_t xsize, size_t ysize,
                  float *values) {
  PROFILER_FUNC;
  // offset is not negative and smaller than square_size.
  assert(offset < square_size);
  []float32 tmp(xsize * ysize);
  for y := 0; y < ysize; y++ {
    const size_t minh = offset > y ? 0 : y - offset;
    const size_t maxh = std_min<size_t>(ysize, y + square_size - offset);
    for x := 0; x < xsize; x++ {
      min := values[x + minh * xsize];
      for j := minh + 1; j < maxh; j++ {
        min = fmin(min, values[x + j * xsize]);
      }
      tmp[x + y * xsize] = static_cast<float>(min);
    }
  }
  for x := 0; x < xsize; x++ {
    const size_t minw = offset > x ? 0 : x - offset;
    const size_t maxw = std_min<size_t>(xsize, x + square_size - offset);
    for y := 0; y < ysize; y++ {
      min := tmp[minw + y * xsize];
      for j := minw + 1; j < maxw; j++ {
        min = fmin(min, tmp[j + y * xsize]);
      }
      values[x + y * xsize] = static_cast<float>(min);
    }
  }
}

// ===== Functions used by Mask only =====
func Average5x5(int xsize, int ysize, []float32* diffs) {
  PROFILER_FUNC;
  if (xsize < 4 || ysize < 4) {
    // TODO: Make this work for small dimensions as well.
    return;
  }
  static const float w = 0.679144890667f;
  static const float scale = 1.0f / (5.0f + 4 * w);
  []float32 result = *diffs;
  []float32 tmp0 = *diffs;
  []float32 tmp1 = *diffs;
  ScaleImage(w, &tmp1);
  for y := 0; y < ysize; y++) {
    row0 := y * xsize;
    result[row0 + 1] += tmp0[row0];
    result[row0 + 0] += tmp0[row0 + 1];
    result[row0 + 2] += tmp0[row0 + 1];
    for x := 2; x < xsize - 2; x++ {
      result[row0 + x - 1] += tmp0[row0 + x];
      result[row0 + x + 1] += tmp0[row0 + x];
    }
    result[row0 + xsize - 3] += tmp0[row0 + xsize - 2];
    result[row0 + xsize - 1] += tmp0[row0 + xsize - 2];
    result[row0 + xsize - 2] += tmp0[row0 + xsize - 1];
    if (y > 0) {
      rowd1 := row0 - xsize;
      result[rowd1 + 1] += tmp1[row0];
      result[rowd1 + 0] += tmp0[row0];
      for x := 1; x < xsize - 1; x++ {
        result[rowd1 + x + 1] += tmp1[row0 + x];
        result[rowd1 + x + 0] += tmp0[row0 + x];
        result[rowd1 + x - 1] += tmp1[row0 + x];
      }
      result[rowd1 + xsize - 1] += tmp0[row0 + xsize - 1];
      result[rowd1 + xsize - 2] += tmp1[row0 + xsize - 1];
    }
    if (y + 1 < ysize) {
      rowu1 := row0 + xsize;
      result[rowu1 + 1] += tmp1[row0];
      result[rowu1 + 0] += tmp0[row0];
      for x := 1; x < xsize - 1; x++ {
        result[rowu1 + x + 1] += tmp1[row0 + x];
        result[rowu1 + x + 0] += tmp0[row0 + x];
        result[rowu1 + x - 1] += tmp1[row0 + x];
      }
      result[rowu1 + xsize - 1] += tmp0[row0 + xsize - 1];
      result[rowu1 + xsize - 2] += tmp1[row0 + xsize - 1];
    }
  }
  *diffs = result;
  ScaleImage(scale, diffs);
}

func DiffPrecompute(
    [][]float32 &xyb0,
    [][]float32 &xyb1,
    size_t xsize, size_t ysize,
    [][]float32 *mask) {
  PROFILER_FUNC;
  mask = [][]float32{
  	make([]float32, len(xyb0[0])),
  	make([]float32, len(xyb0[0])),
  	make([]float32, len(xyb0[0])),
  }
  var valsh0, valsv0, valsh1, valsv1 [3]float64
  var ix2 int;
  for y := 0; y < ysize; y++ {
    for x := 0; x < xsize; x++ {
      size_t ix = x + xsize * y;
      if (x + 1 < xsize) {
        ix2 = ix + 1;
      } else {
        ix2 = ix - 1;
      }
      {
        x0 := (xyb0[0][ix] - xyb0[0][ix2]);
        y0 := (xyb0[1][ix] - xyb0[1][ix2]);
        z0 := (xyb0[2][ix] - xyb0[2][ix2]);
        XybToVals(x0, y0, z0, &valsh0[0], &valsh0[1], &valsh0[2]);
        x1 := (xyb1[0][ix] - xyb1[0][ix2]);
        y1 := (xyb1[1][ix] - xyb1[1][ix2]);
        z1 := (xyb1[2][ix] - xyb1[2][ix2]);
        XybToVals(x1, y1, z1, &valsh1[0], &valsh1[1], &valsh1[2]);
      }
      if (y + 1 < ysize) {
        ix2 = ix + xsize;
      } else {
        ix2 = ix - xsize;
      }
      {
        x0 := (xyb0[0][ix] - xyb0[0][ix2]);
        y0 := (xyb0[1][ix] - xyb0[1][ix2]);
        z0 := (xyb0[2][ix] - xyb0[2][ix2]);
        XybToVals(x0, y0, z0, &valsv0[0], &valsv0[1], &valsv0[2]);
        x1 := (xyb1[0][ix] - xyb1[0][ix2]);
        y1 := (xyb1[1][ix] - xyb1[1][ix2]);
        z1 := (xyb1[2][ix] - xyb1[2][ix2]);
        XybToVals(x1, y1, z1, &valsv1[0], &valsv1[1], &valsv1[2]);
      }
      for i := 0; i < 3; i++ {
        sup0 := fabs(valsh0[i]) + fabs(valsv0[i]);
        sup1 := fabs(valsh1[i]) + fabs(valsv1[i]);
        m := std_min(sup0, sup1);
        (*mask)[i][ix] = static_cast<float>(m);
      }
    }
  }
}

func Mask([][]float32 &xyb0,
          [][]float32 &xyb1,
          size_t xsize, size_t ysize,
          [][]float32 *mask,
          [][]float32 *mask_dc) {
   PROFILER_FUNC;
  mask.resize(3);
  for i := 0; i < 3; i++ {
    (*mask)[i].resize(xsize * ysize);
  }
  DiffPrecompute(xyb0, xyb1, xsize, ysize, mask);
  for i := 0; i < 3; i++ {
    Average5x5(xsize, ysize, &(*mask)[i]);
    MinSquareVal(4, 0, xsize, ysize, (*mask)[i].data());
    static const double sigma[3] = {
      9.65781083553,
      14.2644604355,
      4.53358927369,
    };
    Blur(xsize, ysize, (*mask)[i].data(), sigma[i], 0.0);
  }
  const w00 = 232.206464018;
  const w11 = 22.9455222245;
  const w22 = 503.962310606;

  mask_dc.resize(3);
  for i := 0; i < 3; i++ {
    (*mask_dc)[i].resize(xsize * ysize);
  }
  for y := 0; y < ysize; y++ {
    for x := 0; x < xsize; x++ {
      const size_t idx = y * xsize + x;
      const s0 := (*mask)[0][idx];
      const s1 := (*mask)[1][idx];
      const s2 := (*mask)[2][idx];
      const p0 := w00 * s0;
      const p1 := w11 * s1;
      const p2 := w22 * s2;

      (*mask)[0][idx] = static_cast<float>(MaskX(p0));
      (*mask)[1][idx] = static_cast<float>(MaskY(p1));
      (*mask)[2][idx] = static_cast<float>(MaskB(p2));
      (*mask_dc)[0][idx] = static_cast<float>(MaskDcX(p0));
      (*mask_dc)[1][idx] = static_cast<float>(MaskDcY(p1));
      (*mask_dc)[2][idx] = static_cast<float>(MaskDcB(p2));
    }
  }
  for i := 0; i < 3; i++ {
    ScaleImage(kGlobalScale * kGlobalScale, &(*mask)[i]);
    ScaleImage(kGlobalScale * kGlobalScale, &(*mask_dc)[i]);
  }
}
