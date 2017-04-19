package guetzli_patapon

const (
	kIQuantBits = 16
	// Output of the DCT is upscaled by 16.
	kDCTBits = kIQuantBits + 4
	kBias    = 0x80 << (kDCTBits - 8)
)

func QuantizeP(v *coeff_t, iquant int) {
	*v = coeff_t((int(*v)*iquant + kBias) >> kDCTBits)
}

// Single pixel rgb to 16-bit yuv conversion.
// The returned yuv values are signed integers in the
// range [-128, 127] inclusive.
func RGBToYUV16(rgb []byte, out []coeff_t) {
	const (
		FRAC = 16
		HALF = 1 << (FRAC - 1)
	)
	r, g, b := int(rgb[0]), int(rgb[1]), int(rgb[2])
	out[0] = coeff_t((19595*r + 38469*g + 7471*b - (128 << 16) + HALF) >> FRAC)
	out[64] = coeff_t((-11059*r - 21709*g + 32768*b + HALF - 1) >> FRAC)
	out[128] = coeff_t((32768*r - 27439*g - 5329*b + HALF - 1) >> FRAC)
}

func AddApp0Data(jpg *JPEGData) {
	kApp0Data := [...]byte{
		0xe0, 0x00, 0x10, // APP0
		0x4a, 0x46, 0x49, 0x46, 0x00, // 'JFIF'
		0x01, 0x01, // v1.01
		0x00, 0x00, 0x01, 0x00, 0x01, // aspect ratio = 1:1
		0x00, 0x00, // thumbnail width/height
	}
	jpg.app_data = append(jpg.app_data, string(kApp0Data[:]))
}

func EncodeRGBToJpegQ(rgb []byte, w, h int, quant []int, jpg *JPEGData) bool {
	if w < 0 || w >= 1<<16 || h < 0 || h >= 1<<16 || len(rgb) != 3*w*h {
		return false
	}
	InitJPEGDataForYUV444(w, h, jpg)
	AddApp0Data(jpg)

	var iquant [3 * kDCTBlockSize]int
	idx := 0
	for i := 0; i < 3; i++ {
		for j := 0; j < kDCTBlockSize; j++ {
			v := quant[idx]
			jpg.quant[i].values[j] = v
			iquant[idx] = ((1 << kIQuantBits) + 1) / v
			idx++
		}
	}

	// Compute YUV444 DCT coefficients.
	block_ix := 0
	for block_y := 0; block_y < jpg.MCU_rows; block_y++ {
		for block_x := 0; block_x < jpg.MCU_cols; block_x++ {
			var block [3 * kDCTBlockSize]coeff_t
			// RGB.YUV transform.
			for iy := 0; iy < 8; iy++ {
				for ix := 0; ix < 8; ix++ {
					y := std_min(h-1, 8*block_y+iy)
					x := std_min(w-1, 8*block_x+ix)
					p := y*w + x
					RGBToYUV16(rgb[3*p:], block[8*iy+ix:])
				}
			}
			// DCT
			for i := 0; i < 3; i++ {
				ComputeBlockDCT(block[i*kDCTBlockSize:])
			}
			// Quantization
			for i := 0; i < 3*64; i++ {
				QuantizeP(&block[i], iquant[i])
			}
			// Copy the resulting coefficients to *jpg.
			for i := 0; i < 3; i++ {
				copy(jpg.components[i].coeffs[block_ix*kDCTBlockSize:],
					block[i*kDCTBlockSize:(i+1)*kDCTBlockSize])
			}
			block_ix++
		}
	}

	return true
}

var quantOnes [3 * kDCTBlockSize]int

func init() {
	for i := range quantOnes {
		quantOnes[i] = 1
	}
}

func EncodeRGBToJpeg(rgb []byte, w, h int, jpg *JPEGData) bool {
	return EncodeRGBToJpegQ(rgb, w, h, quantOnes[:], jpg)
}
