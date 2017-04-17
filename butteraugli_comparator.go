package guetzli_patapon

import "math"

const kButteraugliStep = 3

type ButteraugliComparator struct {
	width_               int
	height_              int
	target_distance_     float64
	rgb_orig_            []byte
	block_x_, block_y_   int
	factor_x_, factor_y_ int
	rgb_linear_pregamma_ [][]float64
	mask_xyz_            [][]float64
	per_block_pregamma_  [][][]float64
	comparator_          *ButteraugliButteraugliComparator
	distance_            float64
	distmap_             []float64
	stats_               *ProcessStats
}

func NewButteraugliComparator(width, height int,
	rgb []byte,
	target_distance float64,
	stats *ProcessStats) *ButteraugliComparator {
	bc := new(ButteraugliComparator)
	bc.width_ = width
	bc.height_ = height
	bc.target_distance_ = target_distance
	bc.rgb_orig_ = rgb
	bc.rgb_linear_pregamma_ = [][]float64{
		make([]float64, bc.width_*bc.height_),
		make([]float64, bc.width_*bc.height_),
		make([]float64, bc.width_*bc.height_),
	}
	bc.comparator_ = NewButteraugliButteraugliComparator(bc.width_, bc.height_, kButteraugliStep)
	bc.distance_ = 0.0
	bc.distmap_ = make([]float64, bc.width_)
	for i := range bc.distmap_ {
		bc.distmap_[i] = float64(bc.height_)
	}
	bc.stats_ = stats
	lut := Srgb8ToLinearTable
	for c := 0; c < 3; c++ {
		for y, ix := 0, 0; y < bc.height_; y++ {
			for x := 0; x < bc.width_; x, ix = x+1, ix+1 {
				bc.rgb_linear_pregamma_[c][ix] = lut[bc.rgb_orig_[3*ix+c]]
			}
		}
	}
	OpsinDynamicsImage(bc.width_, bc.height_, bc.rgb_linear_pregamma_)
	return bc
}

func (bc *ButteraugliComparator) Compare(img OutputImage) {
	rgb := [][]float64{
		make([]float64, bc.width_*bc.height_),
		make([]float64, bc.width_*bc.height_),
		make([]float64, bc.width_*bc.height_),
	}
	img.ToLinearRGB(rgb)
	OpsinDynamicsImage(bc.width_, bc.height_, rgb)
	bc.comparator_.DiffmapOpsinDynamicsImage(bc.rgb_linear_pregamma_, rgb, bc.distmap_)
	bc.distance_ = ButteraugliScoreFromDiffmap(bc.distmap_)
	// GUETZLI_LOG(stats_, " BA[100.00%%] D[%6.4f]", distance_);
}

func (bc *ButteraugliComparator) StartBlockComparisons() {
	Mask(bc.rgb_linear_pregamma_, bc.rgb_linear_pregamma_,
		bc.width_, bc.height_,
		bc.mask_xyz_, nil)
}

func (bc *ButteraugliComparator) FinishBlockComparisons() {
	bc.mask_xyz_ = nil
}

func (bc *ButteraugliComparator) SwitchBlock(block_x, block_y, factor_x, factor_y int) {
	bc.block_x_ = block_x
	bc.block_y_ = block_y
	bc.factor_x_ = factor_x
	bc.factor_y_ = factor_y
	bc.per_block_pregamma_ = make([][][]float64, bc.factor_x_*bc.factor_y_)
	lut := Srgb8ToLinearTable
	for off_y, bx := 0, 0; off_y < bc.factor_y_; off_y++ {
		for off_x := 0; off_x < bc.factor_x_; off_x, bx = off_x+1, bx+1 {
			bc.per_block_pregamma_[bx] = [][]float64{
				make([]float64, kDCTBlockSize),
				make([]float64, kDCTBlockSize),
				make([]float64, kDCTBlockSize),
			}
			block_xx := bc.block_x_*bc.factor_x_ + off_x
			block_yy := bc.block_y_*bc.factor_y_ + off_y
			for iy, i := 0, 0; iy < 8; iy++ {
				for ix := 0; ix < 8; ix, i = ix+1, i+1 {
					x := std_min(8*block_xx+ix, bc.width_-1)
					y := std_min(8*block_yy+iy, bc.height_-1)
					px := y*bc.width_ + x
					for c := 0; c < 3; c++ {
						bc.per_block_pregamma_[bx][c][i] = lut[bc.rgb_orig_[3*px+c]]
					}
				}
			}
			OpsinDynamicsImage(8, 8, bc.per_block_pregamma_[bx])
		}
	}
}

func (bc *ButteraugliComparator) CompareBlock(img *OutputImage, off_x, off_y int) float64 {
	block_x := bc.block_x_*bc.factor_x_ + off_x
	block_y := bc.block_y_*bc.factor_y_ + off_y
	xmin := 8 * block_x
	ymin := 8 * block_y
	block_ix := off_y*bc.factor_x_ + off_x
	rgb0_c := bc.per_block_pregamma_[block_ix]

	rgb1_c := make([][]float64, kDCTBlockSize)
	img.ToLinearRGB_(xmin, ymin, 8, 8, rgb1_c)
	OpsinDynamicsImage(8, 8, rgb1_c)

	rgb0 := cloneMatrixFloat64(rgb0_c)
	rgb1 := cloneMatrixFloat64(rgb1_c)

	MaskHighIntensityChange(8, 8, rgb0_c, rgb1_c, rgb0, rgb1)

	var b0, b1 [3 * kDCTBlockSize]float64
	for c := 0; c < 3; c++ {
		for ix := 0; ix < kDCTBlockSize; ix++ {
			b0[c*kDCTBlockSize+ix] = rgb0[c][ix]
			b1[c*kDCTBlockSize+ix] = rgb1[c][ix]
		}
	}
	var diff_xyz_dc, diff_xyz_ac, diff_xyz_edge_dc [3]float64
	ButteraugliBlockDiff(b0, b1, diff_xyz_dc, diff_xyz_ac, diff_xyz_edge_dc)

	var scale [3]float64
	for c := 0; c < 3; c++ {
		scale[c] = bc.mask_xyz_[c][ymin*bc.width_+xmin]
	}

	const kEdgeWeight = 0.05

	diff := 0.0
	diff_edge := 0.0
	for c := 0; c < 3; c++ {
		diff += diff_xyz_dc[c] * scale[c]
		diff += diff_xyz_ac[c] * scale[c]
		diff_edge += diff_xyz_edge_dc[c] * scale[c]
	}
	return math.Sqrt((1-kEdgeWeight)*diff + kEdgeWeight*diff_edge)
}

func (bc *ButteraugliComparator) BlockErrorLimit() float64 {
	return bc.target_distance_
}

func (bc *ButteraugliComparator) ComputeBlockErrorAdjustmentWeights(
	direction int,
	max_block_dist int,
	target_mul float64,
	factor_x, factor_y int,
	distmap []float64,
	block_weight []float64) {
	target_distance := bc.target_distance_ * target_mul
	sizex := 8 * factor_x
	sizey := 8 * factor_y
	block_width := (bc.width_ + sizex - 1) / sizex
	block_height := (bc.height_ + sizey - 1) / sizey
	max_dist_per_block := make([]float64, block_width*block_height)
	for block_y := 0; block_y < block_height; block_y++ {
		for block_x := 0; block_x < block_width; block_x++ {
			block_ix := block_y*block_width + block_x
			x_max := std_min(bc.width_, sizex*(block_x+1))
			y_max := std_min(bc.height_, sizey*(block_y+1))
			max_dist := 0.0
			for y := sizey * block_y; y < y_max; y++ {
				for x := sizex * block_x; x < x_max; x++ {
					max_dist = std_maxFloat64(max_dist, distmap[y*bc.width_+x])
				}
			}
			max_dist_per_block[block_ix] = max_dist
		}
	}
	for block_y := 0; block_y < block_height; block_y++ {
		for block_x := 0; block_x < block_width; block_x++ {
			block_ix := block_y*block_width + block_x
			max_local_dist := target_distance
			x_min := std_max(0, block_x-max_block_dist)
			y_min := std_max(0, block_y-max_block_dist)
			x_max := std_min(block_width, block_x+1+max_block_dist)
			y_max := std_min(block_height, block_y+1+max_block_dist)
			for y := y_min; y < y_max; y++ {
				for x := x_min; x < x_max; x++ {
					max_local_dist = std_maxFloat64(max_local_dist, max_dist_per_block[y*block_width+x])
				}
			}
			if direction > 0 {
				if max_dist_per_block[block_ix] <= target_distance && max_local_dist <= 1.1*target_distance {
					block_weight[block_ix] = 1.0
				}
			} else {
				const kLocalMaxWeight = 0.5
				if max_dist_per_block[block_ix] <= (1-kLocalMaxWeight)*target_distance+kLocalMaxWeight*max_local_dist {
					continue
				}
				for y := y_min; y < y_max; y++ {
					for x := x_min; x < x_max; x++ {
						d := std_max(std_abs(y-block_y), std_abs(x-block_x))
						ix := y*block_width + x
						block_weight[ix] = std_maxFloat64(block_weight[ix], 1.0/(float64(d)+1.0))
					}
				}
			}
		}
	}
}

func (bc *ButteraugliComparator) ScoreOutputSize(size int) float64 {
	return ScoreJPEG(bc.distance_, size, bc.target_distance_)
}
