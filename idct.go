package guetzli_patapon

// Integer implementation of the Inverse Discrete Cosine Transform (IDCT).

// kIDCTMatrix[8*x+u] = alpha(u)*cos((2*x+1)*u*M_PI/16)*sqrt(2), with fixed 13
// bit precision, where alpha(0) = 1/sqrt(2) and alpha(u) = 1 for u > 0.
// Some coefficients are off by +-1 to mimick libjpeg's behaviour.
var kIDCTMatrix = [kDCTBlockSize]int{
  8192,  11363,  10703,   9633,   8192,   6437,   4433,   2260,
  8192,   9633,   4433,  -2259,  -8192, -11362, -10704,  -6436,
  8192,   6437,  -4433, -11362,  -8192,   2261,  10704,   9633,
  8192,   2260, -10703,  -6436,   8192,   9633,  -4433, -11363,
  8192,  -2260, -10703,   6436,   8192,  -9633,  -4433,  11363,
  8192,  -6437,  -4433,  11362,  -8192,  -2261,  10704,  -9633,
  8192,  -9633,   4433,   2259,  -8192,  11362, -10704,   6436,
  8192, -11363,  10703,  -9633,   8192,  -6437,   4433,  -2260,
};

// Computes out[x] = sum{kIDCTMatrix[8*x+u]*in[u*stride]; for u in [0..7]}
func Compute1dIDCT(in []coeff_t, stride int, out []int) {
  var tmp0, tmp1, tmp2, tmp3, tmp4 int

  tmp1 = kIDCTMatrix[0] * int(in[0]);
  for i:=0;i<8;i++ {
     out[i] = tmp1
  }

  tmp0 = int(in[stride]);
  tmp1 = kIDCTMatrix[ 1] * tmp0;
  tmp2 = kIDCTMatrix[ 9] * tmp0;
  tmp3 = kIDCTMatrix[17] * tmp0;
  tmp4 = kIDCTMatrix[25] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] += tmp3;
  out[3] += tmp4;
  out[4] -= tmp4;
  out[5] -= tmp3;
  out[6] -= tmp2;
  out[7] -= tmp1;

  tmp0 = int(in[2 * stride]);
  tmp1 = kIDCTMatrix[ 2] * tmp0;
  tmp2 = kIDCTMatrix[10] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] -= tmp2;
  out[3] -= tmp1;
  out[4] -= tmp1;
  out[5] -= tmp2;
  out[6] += tmp2;
  out[7] += tmp1;

  tmp0 = int(in[3 * stride])
  tmp1 = kIDCTMatrix[ 3] * tmp0;
  tmp2 = kIDCTMatrix[11] * tmp0;
  tmp3 = kIDCTMatrix[19] * tmp0;
  tmp4 = kIDCTMatrix[27] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] += tmp3;
  out[3] += tmp4;
  out[4] -= tmp4;
  out[5] -= tmp3;
  out[6] -= tmp2;
  out[7] -= tmp1;

  tmp0 = int(in[4 * stride])
  tmp1 = kIDCTMatrix[ 4] * tmp0;
  out[0] += tmp1;
  out[1] -= tmp1;
  out[2] -= tmp1;
  out[3] += tmp1;
  out[4] += tmp1;
  out[5] -= tmp1;
  out[6] -= tmp1;
  out[7] += tmp1;

  tmp0 = int(in[5 * stride])
  tmp1 = kIDCTMatrix[ 5] * tmp0;
  tmp2 = kIDCTMatrix[13] * tmp0;
  tmp3 = kIDCTMatrix[21] * tmp0;
  tmp4 = kIDCTMatrix[29] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] += tmp3;
  out[3] += tmp4;
  out[4] -= tmp4;
  out[5] -= tmp3;
  out[6] -= tmp2;
  out[7] -= tmp1;

  tmp0 = int(in[6 * stride])
  tmp1 = kIDCTMatrix[ 6] * tmp0;
  tmp2 = kIDCTMatrix[14] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] -= tmp2;
  out[3] -= tmp1;
  out[4] -= tmp1;
  out[5] -= tmp2;
  out[6] += tmp2;
  out[7] += tmp1;

  tmp0 = int(in[7 * stride])
  tmp1 = kIDCTMatrix[ 7] * tmp0;
  tmp2 = kIDCTMatrix[15] * tmp0;
  tmp3 = kIDCTMatrix[23] * tmp0;
  tmp4 = kIDCTMatrix[31] * tmp0;
  out[0] += tmp1;
  out[1] += tmp2;
  out[2] += tmp3;
  out[3] += tmp4;
  out[4] -= tmp4;
  out[5] -= tmp3;
  out[6] -= tmp2;
  out[7] -= tmp1;
}

func ComputeBlockIDCT(block []coeff_t, out []byte) {
  var colidcts [kDCTBlockSize]coeff_t;
  const kColScale = 11;
  const kColRound = 1 << (kColScale - 1);
  for x := 0; x < 8; x++ {
    var colbuf [8]int
    Compute1dIDCT(block[x:], 8, colbuf[:]);
    for y := 0; y < 8; y++ {
      colidcts[8 * y + x] = coeff_t((colbuf[y] + kColRound) >> kColScale);
    }
  }
  const kRowScale = 18;
  const kRowRound = 257 << (kRowScale - 1);  // includes offset by 128
  for y := 0; y < 8; y++ {
    rowidx := 8 * y;
    var rowbuf [8]int
    Compute1dIDCT(colidcts[rowidx:], 1, rowbuf[:]);
    for x := 0; x < 8; x++ {
      out[rowidx + x] = byte(std_max(0, std_min(255, (rowbuf[x] + kRowRound) >> kRowScale)));
    }
  }
}
