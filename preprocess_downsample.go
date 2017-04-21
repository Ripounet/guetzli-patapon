package guetzli_patapon

import "math"

// convolve with size*size kernel
func Convolve2D(image []float32, w, h int, kernel []float64, size int) []float32 {
	result := cloneSliceFloat32(image) // TODO PATAPON: is this intended to be a copy?
	size2 := size / 2
	for i := 0; i < len(image); i++ {
		x := i % w
		y := i / w
		// Avoid non-normalized results at boundary by skipping edges.
		if x < size2 || x+size-size2-1 >= w || y < size2 || y+size-size2-1 >= h {
			continue
		}
		var v float32
		for j := 0; j < size*size; j++ {
			x2 := x + j%size - size2
			y2 := y + j/size - size2
			v += float32(kernel[j]) * image[y2*w+x2]
		}
		result[i] = v
	}
	return result
}

// convolve horizontally and vertically with 1D kernel
func Convolve2X(image []float32, w, h int, kernel []float64, size int, mul float64) []float32 {
	temp := cloneSliceFloat32(image) // TODO PATAPON: is this intended to be a copy?
	size2 := size / 2
	for i := 0; i < len(image); i++ {
		x := i % w
		y := i / w
		// Avoid non-normalized results at boundary by skipping edges.
		if x < size2 || x+size-size2-1 >= w {
			continue
		}
		var v float32
		for j := 0; j < size; j++ {
			x2 := x + j - size2
			v += float32(kernel[j]) * image[y*w+x2]
		}
		temp[i] = v * float32(mul)
	}
	result := cloneSliceFloat32(temp) // TODO PATAPON: is this intended to be a copy?
	for i := 0; i < len(temp); i++ {
		x := i % w
		y := i / w
		// Avoid non-normalized results at boundary by skipping edges.
		if y < size2 || y+size-size2-1 >= h {
			continue
		}
		var v float32
		for j := 0; j < size; j++ {
			y2 := y + j - size2
			v += float32(kernel[j]) * temp[y2*w+x]
		}
		result[i] = v * float32(mul)
	}
	return result
}

func Normal(x, sigma float64) float64 {
	const kInvSqrt2Pi = 0.3989422804014327
	return math.Exp(-x*x/(2*sigma*sigma)) * kInvSqrt2Pi / sigma
}

func Sharpen(image []float32, w, h int, sigma, amount float64) []float32 {
	// This is only made for small sigma, e.g. 1.3.
	kernel := make([]float64, 5)
	for i := 0; i < len(kernel); i++ {
		kernel[i] = Normal(float64(i-len(kernel)/2), sigma)
	}

	sum := 0.0
	for i := 0; i < len(kernel); i++ {
		sum += kernel[i]
	}
	mul := 1.0 / sum

	result := Convolve2X(image, w, h, kernel, len(kernel), mul)
	for i := 0; i < len(image); i++ {
		result[i] = image[i] + (image[i]-result[i])*float32(amount)
	}
	return result
}

func Erode(w, h int, image []bool) {
	temp := cloneSliceBool(image)
	for y := 1; y+1 < h; y++ {
		for x := 1; x+1 < w; x++ {
			index := y*w + x
			if !(temp[index] && temp[index-1] && temp[index+1] && temp[index-w] && temp[index+w]) {
				image[index] = false
			}
		}
	}
}

func Dilate(w, h int, image []bool) {
	temp := cloneSliceBool(image)
	for y := 1; y+1 < h; y++ {
		for x := 1; x+1 < w; x++ {
			index := y*w + x
			if temp[index] || temp[index-1] || temp[index+1] || temp[index-w] || temp[index+w] {
				image[index] = true
			}
		}
	}
}

func Blur_(image []float32, w, h int) []float32 {
	// This is only made for small sigma, e.g. 1.3.
	const kSigma = 1.3
	kernel := make([]float64, 5)
	for i := 0; i < len(kernel); i++ {
		kernel[i] = Normal(float64(i-len(kernel)/2), kSigma)
	}

	sum := 0.0
	for i := 0; i < len(kernel); i++ {
		sum += kernel[i]
	}
	mul := 1.0 / sum

	return Convolve2X(image, w, h, kernel, len(kernel), mul)
}

// Do the sharpening to the v channel, but only in areas where it will help
// channel should be 2 for v sharpening, or 1 for less effective u sharpening
func PreProcessChannel(w, h, channel int, sigma, amount float32, blur, sharpen bool, image [][]float32) [][]float32 {
	if !blur && !sharpen {
		return image
	}

	// Bring in range 0.0-1.0 for Y, -0.5 - 0.5 for U and V
	yuv := cloneMatrixFloat32(image) // TODO PATAPON: is this intended to be a copy?
	for i := 0; i < len(yuv[0]); i++ {
		yuv[0][i] /= 255.0
		yuv[1][i] = yuv[1][i]/255.0 - 0.5
		yuv[2][i] = yuv[2][i]/255.0 - 0.5
	}

	// Map of areas where the image is not too bright to apply the effect.
	darkmap := make([]bool, len(image[0]))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			index := y*w + x
			y := yuv[0][index]
			u := yuv[1][index]
			v := yuv[2][index]

			r := y + 1.402*v
			g := y - 0.34414*u - 0.71414*v
			b := y + 1.772*u

			// Parameters tuned to avoid sharpening in too bright areas, where the
			// effect makes it worse instead of better.
			if channel == 2 && g < 0.85 && b < 0.85 && r < 0.9 {
				darkmap[index] = true
			}
			if channel == 1 && r < 0.85 && g < 0.85 && b < 0.9 {
				darkmap[index] = true
			}
		}
	}

	Erode(w, h, darkmap)
	Erode(w, h, darkmap)
	Erode(w, h, darkmap)

	// Map of areas where the image is red enough (blue in case of u channel).
	redmap := make([]bool, len(image[0]))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			index := y*w + x
			u := yuv[1][index]
			v := yuv[2][index]

			// Parameters tuned to allow only colors on which sharpening is useful.
			if channel == 2 && 2.116*v > -0.34414*u+0.2 && 1.402*v > 1.772*u+0.2 {
				redmap[index] = true
			}
			if channel == 1 && v < 1.263*u-0.1 && u > -0.33741*v {
				redmap[index] = true
			}
		}
	}

	Dilate(w, h, redmap)
	Dilate(w, h, redmap)
	Dilate(w, h, redmap)

	// Map of areas where to allow sharpening by combining red and dark areas
	sharpenmap := make([]bool, len(image[0]))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			index := y*w + x
			sharpenmap[index] = redmap[index] && darkmap[index]
		}
	}

	// Threshold for where considered an edge.
	threshold := 127.5
	if channel == 2 {
		threshold *= 0.02
	}

	var kEdgeMatrix = [9]float64{
		0, -1, 0,
		-1, 4, -1,
		0, -1, 0,
	}

	// Map of areas where to allow blurring, only where it is not too sharp
	blurmap := make([]bool, len(image[0]))
	edge := Convolve2D(yuv[channel], w, h, kEdgeMatrix[:], 3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			index := y*w + x
			u := yuv[1][index]
			v := yuv[2][index]
			if sharpenmap[index] {
				continue
			}
			if !darkmap[index] {
				continue
			}
			if math.Abs(float64(edge[index])) < threshold && v < -0.162*u {
				blurmap[index] = true
			}
		}
	}
	Erode(w, h, blurmap)
	Erode(w, h, blurmap)

	// Choose sharpened, blurred or original per pixel
	sharpened := Sharpen(yuv[channel], w, h, float64(sigma), float64(amount))
	blurred := Blur_(yuv[channel], w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			index := y*w + x

			if sharpenmap[index] {
				if sharpen {
					yuv[channel][index] = sharpened[index]
				}
			} else if blurmap[index] {
				if blur {
					yuv[channel][index] = blurred[index]
				}
			}
		}
	}

	// Bring back to range 0-255
	for i := 0; i < len(yuv[0]); i++ {
		yuv[0][i] *= 255.0
		yuv[1][i] = (yuv[1][i] + 0.5) * 255.0
		yuv[2][i] = (yuv[2][i] + 0.5) * 255.0
	}
	return yuv
}

func Clip(val float32) float32 {
	return std_maxFloa32(0.0, std_minFloat32(255.0, val))
}

func RGBToY(r, g, b float32) float32 {
	return 0.299*r + 0.587*g + 0.114*b
}

func RGBToU(r, g, b float32) float32 {
	return -0.16874*r - 0.33126*g + 0.5*b + 128.0
}

func RGBToV(r, g, b float32) float32 {
	return 0.5*r - 0.41869*g - 0.08131*b + 128.0
}

func YUVToR(y, u, v float32) float32 {
	return y + 1.402*(v-128.0)
}

func YUVToG(y, u, v float32) float32 {
	return y - 0.344136*(u-128.0) - 0.714136*(v-128.0)
}

func YUVToB(y, u, v float32) float32 {
	return y + 1.772*(u-128.0)
}

// TODO(user) Use SRGB->linear conversion and a lookup-table.
func GammaToLinear(x float32) float32 {
	return float32(math.Pow(float64(x/255.0), 2.2))
}

// TODO(user) Use linear->SRGB conversion and a lookup-table.
func LinearToGamma(x float32) float32 {
	return float32(255.0 * math.Pow(float64(x), 1.0/2.2))
}

func LinearlyAveragedLuma(rgb []float32) []float32 {
	assert(len(rgb)%3 == 0)
	y := make([]float32, len(rgb)/3)
	for i, p := 0, 0; p < len(rgb); i, p = i+1, p+3 {
		y[i] = LinearToGamma(RGBToY(GammaToLinear(rgb[p+0]),
			GammaToLinear(rgb[p+1]),
			GammaToLinear(rgb[p+2])))
	}
	return y
}

func LinearlyDownsample2x2(rgb_in []float32, width, height int) []float32 {
	assert(len(rgb_in) == 3*width*height)
	w := (width + 1) / 2
	h := (height + 1) / 2
	rgb_out := make([]float32, 3*w*h)
	for y, p := 0, 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for i := 0; i < 3; i, p = i+1, p+1 {
				rgb_out[p] = 0.0
				for iy := 0; iy < 2; iy++ {
					for ix := 0; ix < 2; ix++ {
						yy := std_min(height-1, 2*y+iy)
						xx := std_min(width-1, 2*x+ix)
						rgb_out[p] += GammaToLinear(rgb_in[3*(yy*width+xx)+i])
					}
				}
				rgb_out[p] = LinearToGamma(0.25 * rgb_out[p])
			}
		}
	}
	return rgb_out
}

func RGBToYUV(rgb []float32) [][]float32 {
	yuv := make([][]float32, 3)
	for i := range yuv {
		yuv[i] = make([]float32, len(rgb)/3)
	}
	for i, p := 0, 0; p < len(rgb); i, p = i+1, p+3 {
		r := rgb[p+0]
		g := rgb[p+1]
		b := rgb[p+2]
		yuv[0][i] = RGBToY(r, g, b)
		yuv[1][i] = RGBToU(r, g, b)
		yuv[2][i] = RGBToV(r, g, b)
	}
	return yuv
}

func YUVToRGB(yuv [][]float32) []float32 {
	rgb := make([]float32, 3*len(yuv[0]))
	for i, p := 0, 0; p < len(rgb); i, p = i+1, p+3 {
		y := yuv[0][i]
		u := yuv[1][i]
		v := yuv[2][i]
		rgb[p+0] = Clip(YUVToR(y, u, v))
		rgb[p+1] = Clip(YUVToG(y, u, v))
		rgb[p+2] = Clip(YUVToB(y, u, v))
	}
	return rgb
}

// Upsamples img_in with a box-filter, and returns an image with output
// dimensions width x height.
func Upsample2x2(img_in []float32, width, height int) []float32 {
	w := (width + 1) / 2
	h := (height + 1) / 2
	assert(len(img_in) == w*h)
	img_out := make([]float32, width*height)
	for y, p := 0, 0; y < h; y++ {
		for x := 0; x < w; x, p = x+1, p+1 {
			for iy := 0; iy < 2; iy++ {
				for ix := 0; ix < 2; ix++ {
					yy := std_min(height-1, 2*y+iy)
					xx := std_min(width-1, 2*x+ix)
					img_out[yy*width+xx] = img_in[p]
				}
			}
		}
	}
	return img_out
}

// Apply the "fancy upsample" filter used by libjpeg.
func Blur__(img []float32, width, height int) []float32 {
	img_out := make([]float32, width*height)
	for y0 := 0; y0 < height; y0 += 2 {
		for x0 := 0; x0 < width; x0 += 2 {
			for iy := 0; iy < 2 && y0+iy < height; iy++ {
				for ix := 0; ix < 2 && x0+ix < width; ix++ {
					dy := 4*iy - 2
					dx := 4*ix - 2
					x1 := std_min(width-1, std_max(0, x0+dx))
					y1 := std_min(height-1, std_max(0, y0+dy))
					img_out[(y0+iy)*width+x0+ix] =
						(9.0*img[y0*width+x0] +
							3.0*img[y0*width+x1] +
							3.0*img[y1*width+x0] +
							1.0*img[y1*width+x1]) / 16.0
				}
			}
		}
	}
	return img_out
}

func YUV420ToRGB(yuv420 [][]float32, width, height int) []float32 {
	var yuv [][]float32
	yuv = append(yuv, yuv420[0])
	u := Upsample2x2(yuv420[1], width, height)
	v := Upsample2x2(yuv420[2], width, height)
	yuv = append(yuv, Blur__(u, width, height))
	yuv = append(yuv, Blur__(v, width, height))
	return YUVToRGB(yuv)
}

func UpdateGuess(target []float32,
	reconstructed []float32,
	guess []float32) {
	assert(len(reconstructed) == len(guess))
	assert(len(target) == len(guess))
	for i := 0; i < len(guess); i++ {
		// TODO(user): Evaluate using a decaying constant here.
		guess[i] = Clip(guess[i] - (reconstructed[i] - target[i]))
	}
}

func RGBToYUV420(rgb_in []byte, width, height int) [][]float32 {
	rgbf := make([]float32, len(rgb_in))
	for i := 0; i < len(rgb_in); i++ {
		rgbf[i] = float32(rgb_in[i])
	}
	y_target := LinearlyAveragedLuma(rgbf)
	yuv_target := RGBToYUV(LinearlyDownsample2x2(rgbf, width, height))
	yuv_guess := cloneMatrixFloat32(yuv_target) // TODO PATAPON is this intended to be a copy?
	yuv_guess[0] = Upsample2x2(yuv_guess[0], width, height)
	// TODO(user): Stop early if the error is small enough.
	for iter := 0; iter < 20; iter++ {
		rgb_rec := YUV420ToRGB(yuv_guess, width, height)
		y_rec := LinearlyAveragedLuma(rgb_rec)
		yuv_rec := RGBToYUV(LinearlyDownsample2x2(rgb_rec, width, height))
		UpdateGuess(y_target, y_rec, yuv_guess[0])
		UpdateGuess(yuv_target[1], yuv_rec[1], yuv_guess[1])
		UpdateGuess(yuv_target[2], yuv_rec[2], yuv_guess[2])
	}
	yuv_guess[1] = Upsample2x2(yuv_guess[1], width, height)
	yuv_guess[2] = Upsample2x2(yuv_guess[2], width, height)
	return yuv_guess
}
