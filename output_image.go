package guetzli_patapon

import (
	"fmt"
	"log"
)

type OutputImageComponent struct {
	width_            int
	height_           int
	factor_x_         int
	factor_y_         int
	width_in_blocks_  int
	height_in_blocks_ int
	num_blocks_       int
	coeffs_           []coeff_t
	pixels_           []uint16
	// Same as last argument of ApplyGlobalQuantization() (default is all 1s).
	quant_ [kDCTBlockSize]int
}

func (oic *OutputImageComponent) width() int            { return oic.width_ }
func (oic *OutputImageComponent) height() int           { return oic.height_ }
func (oic *OutputImageComponent) factor_x() int         { return oic.factor_x_ }
func (oic *OutputImageComponent) factor_y() int         { return oic.factor_y_ }
func (oic *OutputImageComponent) width_in_blocks() int  { return oic.width_in_blocks_ }
func (oic *OutputImageComponent) height_in_blocks() int { return oic.height_in_blocks_ }
func (oic *OutputImageComponent) coeffs() []coeff_t     { return oic.coeffs_ }
func (oic *OutputImageComponent) quant() []int          { return oic.quant_[:] }

type OutputImage struct {
	width_, height_ int
	components_     []OutputImageComponent
}

func (img *OutputImage) width() int                            { return img.width_ }
func (img *OutputImage) height() int                           { return img.height_ }
func (img *OutputImage) component(c int) *OutputImageComponent { return &img.components_[c] }

// If sharpen or blur are enabled, preprocesses image before downsampling U or
// V to improve butteraugli score and/or reduce file size.
// u_sharpen: sharpen the u channel in red areas to improve score (not as
// effective as v_sharpen, blue is not so important)
// u_blur: blur the u channel in some areas to reduce file size
// v_sharpen: sharpen the v channel in red areas to improve score
// v_blur: blur the v channel in some areas to reduce file size
type DownsampleConfig struct {
	// Default is YUV420.
	u_factor_x, u_factor_y int
	v_factor_x, v_factor_y int
	u_sharpen, u_blur      bool
	v_sharpen, v_blur      bool
	use_silver_screen      bool
}

func downsampleConfig() DownsampleConfig {
	return DownsampleConfig{
		u_factor_x:        2,
		u_factor_y:        2,
		v_factor_x:        2,
		v_factor_y:        2,
		u_sharpen:         true,
		u_blur:            true,
		v_sharpen:         true,
		v_blur:            true,
		use_silver_screen: false,
	}
}

//////////////////////////////////////////////////////////////

func (oic *OutputImageComponent) Reset(factor_x, factor_y int) {
	oic.factor_x_ = factor_x
	oic.factor_y_ = factor_y
	oic.width_in_blocks_ = (oic.width_ + 8*factor_x - 1) / (8 * factor_x)
	oic.height_in_blocks_ = (oic.height_ + 8*factor_y - 1) / (8 * factor_y)
	oic.num_blocks_ = oic.width_in_blocks_ * oic.height_in_blocks_
	oic.coeffs_ = make([]coeff_t, oic.num_blocks_*kDCTBlockSize)
	oic.pixels_ = make([]uint16, oic.width_*oic.height_)
	for i := range oic.pixels_ {
		oic.pixels_[i] = 128 << 4
	}
	for i := 0; i < kDCTBlockSize; i++ {
		oic.quant_[i] = 1
	}
}

func (oic *OutputImageComponent) IsAllZero() bool {
	numcoeffs := oic.num_blocks_ * kDCTBlockSize
	for i := 0; i < numcoeffs; i++ {
		if oic.coeffs_[i] != 0 {
			return false
		}
	}
	return true
}

func (oic *OutputImageComponent) GetCoeffBlock(block_x, block_y int, block []coeff_t) {
	assert(block_x < oic.width_in_blocks_)
	assert(block_y < oic.height_in_blocks_)
	offset := (block_y*oic.width_in_blocks_ + block_x) * kDCTBlockSize
	copy(block[:kDCTBlockSize], oic.coeffs_[offset:])
}

func (oic *OutputImageComponent) ToPixels(xmin, ymin, xsize, ysize int, out []byte, stride int) {
	assert(xmin >= 0)
	assert(ymin >= 0)
	assert(xmin < oic.width_)
	assert(ymin < oic.height_)
	yend1 := ymin + ysize
	yend0 := std_min(yend1, oic.height_)
	y := ymin
	for ; y < yend0; y++ {
		xend1 := xmin + xsize
		xend0 := std_min(xend1, oic.width_)
		x := xmin
		px := y*oic.width_ + xmin
		for ; x < xend0; x, px, out = x+1, px+1, out[stride:] {
			out[0] = byte((int(oic.pixels_[px]) + 8 - (x & 1)) >> 4)
		}
		offset := -stride
		for ; x < xend1; x++ {
			out = out[offset:]
			out = out[stride:]
		}
	}
	for ; y < yend1; y++ {
		offset := -stride * xsize
		for x := 0; x < xsize; x++ {
			out = out[offset:]
			out = out[stride:]
		}
	}
}

func (oic *OutputImageComponent) ToFloatPixels(out []float32, stride int) {
	assert(oic.factor_x_ == 1)
	assert(oic.factor_y_ == 1)
	for block_y := 0; block_y < oic.height_in_blocks_; block_y++ {
		for block_x := 0; block_x < oic.width_in_blocks_; block_x++ {
			var block [kDCTBlockSize]coeff_t
			oic.GetCoeffBlock(block_x, block_y, block[:])
			var blockd [kDCTBlockSize]float64
			for k := 0; k < kDCTBlockSize; k++ {
				blockd[k] = float64(block[k])
			}
			ComputeBlockIDCTDouble(blockd[:])
			for iy := 0; iy < 8; iy++ {
				for ix := 0; ix < 8; ix++ {
					y := block_y*8 + iy
					x := block_x*8 + ix
					if y >= oic.height_ || x >= oic.width_ {
						continue
					}
					out[(y*oic.width_+x)*stride] = float32(blockd[8*iy+ix] + 128.0)
				}
			}
		}
	}
}

func (oic *OutputImageComponent) SetCoeffBlock(block_x, block_y int, block []coeff_t) {
	assert(block_x < oic.width_in_blocks_)
	assert(block_y < oic.height_in_blocks_)
	offset := (block_y*oic.width_in_blocks_ + block_x) * kDCTBlockSize
	copy(oic.coeffs_[offset:], block[:kDCTBlockSize])
	var idct [kDCTBlockSize]byte
	ComputeBlockIDCT(oic.coeffs_[offset:], idct[:])
	oic.UpdatePixelsForBlock(block_x, block_y, idct[:])
}

func (oic *OutputImageComponent) UpdatePixelsForBlock(block_x, block_y int, idct []byte) {
	if oic.factor_x_ == 1 && oic.factor_y_ == 1 {
		for iy := 0; iy < 8; iy++ {
			for ix := 0; ix < 8; ix++ {
				x := 8*block_x + ix
				y := 8*block_y + iy
				if x >= oic.width_ || y >= oic.height_ {
					continue
				}
				p := y*oic.width_ + x
				oic.pixels_[p] = uint16(idct[8*iy+ix] << 4)
			}
		}
	} else if oic.factor_x_ == 2 && oic.factor_y_ == 2 {
		// Fill in the 10x10 pixel area in the subsampled image that will be the
		// basis of the upsampling. This area is enough to hold the 3x3 kernel of
		// the fancy upsampler around each pixel.
		const kSubsampledEdgeSize = 10
		var subsampled [kSubsampledEdgeSize * kSubsampledEdgeSize]uint16
		for j := 0; j < kSubsampledEdgeSize; j++ {
			// The order we fill in the rows is:
			//   8 rows intersecting the block, row below, row above
			y0 := block_y * 16
			if j < 9 {
				y0 += j * 2
			} else {
				y0 -= 2
			}
			for i := 0; i < kSubsampledEdgeSize; i++ {
				// The order we fill in each row is:
				//   8 pixels within the block, left edge, right edge
				ix := 0
				if j < 9 {
					ix = (j + 1) * kSubsampledEdgeSize
				}
				if i < 9 {
					ix += i + 1
				}
				x0 := block_x * 16
				if i < 9 {
					x0 += i * 2
				} else {
					x0 -= 2
				}
				switch {
				case x0 < 0:
					subsampled[ix] = subsampled[ix+1]
				case y0 < 0:
					subsampled[ix] = subsampled[ix+kSubsampledEdgeSize]
				case x0 >= oic.width_:
					subsampled[ix] = subsampled[ix-1]
				case y0 >= oic.height_:
					subsampled[ix] = subsampled[ix-kSubsampledEdgeSize]
				case i < 8 && j < 8:
					subsampled[ix] = uint16(idct[j*8+i] << 4)
				default:
					// Reconstruct the subsampled pixels around the edge of the current
					// block by computing the inverse of the fancy upsampler.
					y1 := std_max(y0-1, 0)
					x1 := std_max(x0-1, 0)
					subsampled[ix] = (oic.pixels_[y0*oic.width_+x0]*9 +
						oic.pixels_[y1*oic.width_+x1] -
						oic.pixels_[y0*oic.width_+x1]*3 -
						oic.pixels_[y1*oic.width_+x0]*3) >> 2
				}
			}
		}

		// Determine area to update.
		xmin := std_max(block_x*16-1, 0)
		xmax := std_min(block_x*16+16, oic.width_-1)
		ymin := std_max(block_y*16-1, 0)
		ymax := std_min(block_y*16+16, oic.height_-1)

		// Apply the fancy upsampler on the subsampled block.
		for y := ymin; y <= ymax; y++ {
			y0 := ((y & ^1)/2 - block_y*8 + 1) * kSubsampledEdgeSize
			dy := ((y&1)*2 - 1) * kSubsampledEdgeSize
			rowptr := oic.pixels_[y*oic.width_:]
			for x := xmin; x <= xmax; x++ {
				x0 := (x & ^1)/2 - block_x*8 + 1
				dx := (x&1)*2 - 1
				ix := x0 + y0
				rowptr[x] = (subsampled[ix]*9 + subsampled[ix+dy]*3 +
					subsampled[ix+dx]*3 + subsampled[ix+dx+dy]) >> 4
			}
		}
	} else {
		log.Fatalf("Sampling ratio not supported: factor_x = %d factor_y = %d\n",
			oic.factor_x_, oic.factor_y_)
	}
}

func (oic *OutputImageComponent) CopyFromJpegComponent(comp *JPEGComponent, factor_x, factor_y int, quant []int) {
	oic.Reset(factor_x, factor_y)
	assert(oic.width_in_blocks_ <= comp.width_in_blocks)
	assert(oic.height_in_blocks_ <= comp.height_in_blocks)
	src_row_size := comp.width_in_blocks * kDCTBlockSize
	for block_y := 0; block_y < oic.height_in_blocks_; block_y++ {
		src_coeffs := comp.coeffs[block_y*src_row_size:]
		for block_x := 0; block_x < oic.width_in_blocks_; block_x++ {
			var block [kDCTBlockSize]coeff_t
			for i := 0; i < kDCTBlockSize; i++ {
				block[i] = src_coeffs[i] * coeff_t(quant[i])
			}
			oic.SetCoeffBlock(block_x, block_y, block[:])
			src_coeffs = src_coeffs[kDCTBlockSize:]
		}
	}
	copy(oic.quant_[:], quant)
}

func (oic *OutputImageComponent) ApplyGlobalQuantization(q []int) {
	for block_y := 0; block_y < oic.height_in_blocks_; block_y++ {
		for block_x := 0; block_x < oic.width_in_blocks_; block_x++ {
			var block [kDCTBlockSize]coeff_t
			oic.GetCoeffBlock(block_x, block_y, block[:])
			if QuantizeBlock(block[:], q) {
				oic.SetCoeffBlock(block_x, block_y, block[:])
			}
		}
	}
	copy(oic.quant_[:], q)
}

func NewOutputImage(w, h int) *OutputImage {
	img := new(OutputImage)
	img.width_ = w
	img.height_ = h
	img.components_ = make([]OutputImageComponent, 3)
	for i := range img.components_ {
		img.components_[i].width_ = w
		img.components_[i].height_ = h
		img.components_[i].Reset(1, 1)
	}
	return img
}

func (img *OutputImage) CopyFromJpegData(jpg *JPEGData) {
	for i := 0; i < len(jpg.components); i++ {
		comp := &jpg.components[i]
		assert(jpg.max_h_samp_factor%comp.h_samp_factor == 0)
		assert(jpg.max_v_samp_factor%comp.v_samp_factor == 0)
		factor_x := jpg.max_h_samp_factor / comp.h_samp_factor
		factor_y := jpg.max_v_samp_factor / comp.v_samp_factor
		assert(comp.quant_idx < len(jpg.quant))
		img.components_[i].CopyFromJpegComponent(comp, factor_x, factor_y, jpg.quant[comp.quant_idx].values)
	}
}

func SetDownsampledCoefficients(pixels []float32, factor_x, factor_y int, comp *OutputImageComponent) {
	assert(len(pixels) == comp.width()*comp.height())
	comp.Reset(factor_x, factor_y)
	for block_y := 0; block_y < comp.height_in_blocks(); block_y++ {
		for block_x := 0; block_x < comp.width_in_blocks(); block_x++ {
			var blockd [kDCTBlockSize]float64
			x0 := 8 * block_x * factor_x
			y0 := 8 * block_y * factor_y
			assert(x0 < comp.width())
			assert(y0 < comp.height())
			for iy := 0; iy < 8; iy++ {
				for ix := 0; ix < 8; ix++ {
					var avg float32
					for j := 0; j < factor_y; j++ {
						for i := 0; i < factor_x; i++ {
							x := std_min(x0+ix*factor_x+i, comp.width()-1)
							y := std_min(y0+iy*factor_y+j, comp.height()-1)
							avg += pixels[y*comp.width()+x]
						}
					}
					avg /= float32(factor_x * factor_y)
					blockd[iy*8+ix] = float64(avg)
				}
			}
			ComputeBlockDCTDouble(blockd[:])
			blockd[0] -= 1024.0
			var block [kDCTBlockSize]coeff_t
			for k := 0; k < kDCTBlockSize; k++ {
				block[k] = coeff_t(std_round(blockd[k]))
			}
			comp.SetCoeffBlock(block_x, block_y, block[:])
		}
	}
}

func (img *OutputImage) Downsample(cfg *DownsampleConfig) {
	if img.components_[1].IsAllZero() && img.components_[2].IsAllZero() {
		// If the image is already grayscale, nothing to do.
		return
	}
	if cfg.use_silver_screen &&
		cfg.u_factor_x == 2 && cfg.u_factor_y == 2 &&
		cfg.v_factor_x == 2 && cfg.v_factor_y == 2 {
		rgb := img.ToSRGB_()
		yuv := RGBToYUV420(rgb, img.width_, img.height_)
		SetDownsampledCoefficients(yuv[0], 1, 1, &img.components_[0])
		SetDownsampledCoefficients(yuv[1], 2, 2, &img.components_[1])
		SetDownsampledCoefficients(yuv[2], 2, 2, &img.components_[2])
		return
	}
	// Get the floating-point precision YUV array represented by the set of
	// DCT coefficients.
	yuv := make([][]float32, 3)
	for i := range yuv {
		yuv[i] = make([]float32, img.width_*img.height_)
	}
	for c := 0; c < 3; c++ {
		img.components_[c].ToFloatPixels(yuv[c], 1)
	}

	yuv = PreProcessChannel(img.width_, img.height_, 2, 1.3, 0.5, cfg.u_sharpen, cfg.u_blur, yuv)
	yuv = PreProcessChannel(img.width_, img.height_, 1, 1.3, 0.5, cfg.v_sharpen, cfg.v_blur, yuv)

	// Do the actual downsampling (averaging) and forward-DCT.
	if cfg.u_factor_x != 1 || cfg.u_factor_y != 1 {
		SetDownsampledCoefficients(yuv[1], cfg.u_factor_x, cfg.u_factor_y, &img.components_[1])
	}
	if cfg.v_factor_x != 1 || cfg.v_factor_y != 1 {
		SetDownsampledCoefficients(yuv[2], cfg.v_factor_x, cfg.v_factor_y, &img.components_[2])
	}
}

func (img *OutputImage) ApplyGlobalQuantization(q [][kDCTBlockSize]int) {
	for c := 0; c < 3; c++ {
		img.components_[c].ApplyGlobalQuantization(q[c][:])
	}
}

func (img *OutputImage) SaveToJpegData(jpg *JPEGData) {
	assert(img.components_[0].factor_x() == 1)
	assert(img.components_[0].factor_y() == 1)
	jpg.width = img.width_
	jpg.height = img.height_
	jpg.max_h_samp_factor = 1
	jpg.max_v_samp_factor = 1
	jpg.MCU_cols = img.components_[0].width_in_blocks()
	jpg.MCU_rows = img.components_[0].height_in_blocks()
	ncomp := 3
	if img.components_[1].IsAllZero() && img.components_[2].IsAllZero() {
		ncomp = 1
	}
	for i := 1; i < ncomp; i++ {
		jpg.max_h_samp_factor = std_max(jpg.max_h_samp_factor,
			img.components_[i].factor_x())
		jpg.max_v_samp_factor = std_max(jpg.max_h_samp_factor,
			img.components_[i].factor_y())
		jpg.MCU_cols = std_min(jpg.MCU_cols, img.components_[i].width_in_blocks())
		jpg.MCU_rows = std_min(jpg.MCU_rows, img.components_[i].height_in_blocks())
	}
	jpg.components = make([]JPEGComponent, ncomp)
	var q [3][kDCTBlockSize]int
	for c := 0; c < 3; c++ {
		copy(q[c][:kDCTBlockSize], img.components_[c].quant())
	}
	for c := 0; c < ncomp; c++ {
		comp := &jpg.components[c]
		assert(jpg.max_h_samp_factor%img.components_[c].factor_x() == 0)
		assert(jpg.max_v_samp_factor%img.components_[c].factor_y() == 0)
		comp.id = c
		comp.h_samp_factor = jpg.max_h_samp_factor / img.components_[c].factor_x()
		comp.v_samp_factor = jpg.max_v_samp_factor / img.components_[c].factor_y()
		comp.width_in_blocks = jpg.MCU_cols * comp.h_samp_factor
		comp.height_in_blocks = jpg.MCU_rows * comp.v_samp_factor
		comp.num_blocks = comp.width_in_blocks * comp.height_in_blocks
		comp.coeffs = make([]coeff_t, kDCTBlockSize*comp.num_blocks)

		var last_dc coeff_t
		src_coeffs := img.components_[c].coeffs()
		dest_coeffs := comp.coeffs
		for block_y := 0; block_y < comp.height_in_blocks; block_y++ {
			for block_x := 0; block_x < comp.width_in_blocks; block_x++ {
				if block_y >= img.components_[c].height_in_blocks() ||
					block_x >= img.components_[c].width_in_blocks() {
					dest_coeffs[0] = last_dc
					for k := 1; k < kDCTBlockSize; k++ {
						dest_coeffs[k] = 0
					}
				} else {
					for k := 0; k < kDCTBlockSize; k++ {
						quant := q[c][k]
						coeff := src_coeffs[k]
						assert(int(coeff)%quant == 0)
						dest_coeffs[k] = coeff / coeff_t(quant)
					}
					src_coeffs = src_coeffs[kDCTBlockSize:]
				}
				last_dc = dest_coeffs[0]
				dest_coeffs = dest_coeffs[kDCTBlockSize:]
			}
		}
	}
	SaveQuantTables(q[:], jpg)
}

func (img *OutputImage) ToSRGB(xmin, ymin, xsize, ysize int) []byte {
	rgb := make([]byte, xsize*ysize*3)
	for c := 0; c < 3; c++ {
		img.components_[c].ToPixels(xmin, ymin, xsize, ysize, rgb[c:], 3)
	}
	for p := 0; p < len(rgb); p += 3 {
		ColorTransformYCbCrToRGB(rgb[p:])
	}
	return rgb
}

func (img *OutputImage) ToSRGB_() []byte {
	return img.ToSRGB(0, 0, img.width_, img.height_)
}

func (img *OutputImage) ToLinearRGB(xmin, ymin, xsize, ysize int, rgb [][]float32) {
	lut := Srgb8ToLinearTable
	rgb_pixels := img.ToSRGB(xmin, ymin, xsize, ysize)
	for p := 0; p < xsize*ysize; p++ {
		for i := 0; i < 3; i++ {
			rgb[i][p] = float32(lut[rgb_pixels[3*p+i]])
		}
	}
}

func (img *OutputImage) ToLinearRGB_(rgb [][]float32) {
	img.ToLinearRGB(0, 0, img.width_, img.height_, rgb)
}

func (img *OutputImage) FrameTypeStr() string {
	return fmt.Sprintf("f%d%d%d%d%d%d",
		img.component(0).factor_x(), img.component(0).factor_y(),
		img.component(1).factor_x(), img.component(1).factor_y(),
		img.component(2).factor_x(), img.component(2).factor_y())
}
