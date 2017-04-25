package guetzli_patapon

import (
	"bytes"
	"log"
	"math"
	"sort"
)

type Params struct {
	butteraugli_target       float32
	clear_metadata           bool
	try_420                  bool
	force_420                bool
	use_silver_screen        bool
	zeroing_greedy_lookahead int
	new_zeroing_model        bool
}

func params() Params {
	return Params{
		butteraugli_target:       1.0,
		clear_metadata:           true,
		try_420:                  false,
		force_420:                false,
		use_silver_screen:        false,
		zeroing_greedy_lookahead: 3,
		new_zeroing_model:        true,
	}
}

type GuetzliOutput struct {
	jpeg_data         string
	distmap           []float32
	distmap_aggregate float64
	score             float64
}

//////////////////////////////////////////

const kBlockSize = 3 * kDCTBlockSize

type CoeffData struct {
	idx       int
	block_err float32
}

type QuantData struct {
	q        [3][kDCTBlockSize]int
	jpg_size int
	dist_ok  bool
}

type Processor struct {
	params_       Params
	comparator_   Comparator
	final_output_ *GuetzliOutput
	stats_        *ProcessStats
}

func RemoveOriginalQuantization(jpg *JPEGData, q_in [][kDCTBlockSize]int) {
	for i := 0; i < 3; i++ {
		c := jpg.components[i]
		q := jpg.quant[c.quant_idx].values
		copy(q_in[i][:kDCTBlockSize], q)
		for j := 0; j < len(c.coeffs); j++ {
			c.coeffs[j] *= coeff_t(q[j%kDCTBlockSize])
		}
	}
	var q [3][kDCTBlockSize]int
	for i := 0; i < 3; i++ {
		for j := 0; j < kDCTBlockSize; j++ {
			q[i][j] = 1
		}
	}
	SaveQuantTables(q[:], jpg)
}

func (p *Processor) DownsampleImage(img *OutputImage) {
	if img.component(1).factor_x() > 1 || img.component(1).factor_y() > 1 {
		return
	}
	var cfg DownsampleConfig
	cfg.use_silver_screen = p.params_.use_silver_screen
	img.Downsample(&cfg)
}

func CheckJpegSanity(jpg *JPEGData) bool {
	kMaxComponent := 1 << 12
	for _, comp := range jpg.components {
		quant_table := jpg.quant[comp.quant_idx]
		for i := 0; i < len(comp.coeffs); i++ {
			coeff := comp.coeffs[i]
			quant := quant_table.values[i%kDCTBlockSize]
			if std_abs(int(coeff)*quant) > kMaxComponent {
				return false
			}
		}
	}
	return true
}

func (p *Processor) OutputJpeg(jpg *JPEGData) []byte {
	var out bytes.Buffer
	if !WriteJpeg(jpg, p.params_.clear_metadata, &out) {
		panic("ouch")
	}
	return out.Bytes()
}

func (p *Processor) MaybeOutput(encoded_jpg string) {
	score := p.comparator_.ScoreOutputSize(len(encoded_jpg))
	GUETZLI_LOG(p.stats_, " Score[%.4f]", score)
	if score < p.final_output_.score || p.final_output_.score < 0 {
		p.final_output_.jpeg_data = encoded_jpg
		p.final_output_.distmap = p.comparator_.distmap()
		p.final_output_.distmap_aggregate = float64(p.comparator_.distmap_aggregate())
		p.final_output_.score = score
		GUETZLI_LOG(p.stats_, " (*)")
	}
	GUETZLI_LOG(p.stats_, "\n")
}

func CompareQuantData(a, b *QuantData) bool {
	if a.dist_ok && !b.dist_ok {
		return true
	}
	if !a.dist_ok && b.dist_ok {
		return false
	}
	return a.jpg_size < b.jpg_size
}

// Compares a[0..kBlockSize) and b[0..kBlockSize) vectors, and returns
//   0 : if they are equal
//  -1 : if a is everywhere <= than b and in at least one coordinate <
//   1 : if a is everywhere >= than b and in at least one coordinate >
//   2 : if a and b are uncomparable (some coordinate smaller and some greater)
func CompareQuantMatrices(a, b []int) int {
	i := 0
	for i < kBlockSize && a[i] == b[i] {
		i++
	}
	if i == kBlockSize {
		return 0
	}
	if a[i] < b[i] {
		i++
		for ; i < kBlockSize; i++ {
			if a[i] > b[i] {
				return 2
			}
		}
		return -1
	} else {
		i++
		for ; i < kBlockSize; i++ {
			if a[i] < b[i] {
				return 2
			}
		}
		return 1
	}
}

func ContrastSensitivity(k int) float64 {
	return 1.0 / (1.0 + float64(kJPEGZigZagOrder[k])/2.0)
}

func QuantMatrixHeuristicScore(q [][kDCTBlockSize]int) float64 {
	score := 0.0
	for c := 0; c < 3; c++ {
		for k := 0; k < kDCTBlockSize; k++ {
			score += (0.5 * (float64(q[c][k]) - 1.0) * ContrastSensitivity(k))
		}
	}
	return score
}

type QuantMatrixGenerator struct {
	downsample_ bool
	// Lower bound for quant matrix heuristic score used in binary search.
	hscore_a_ float64
	// Upper bound for quant matrix heuristic score used in binary search, or 0.0
	// if no upper bound is found yet.
	hscore_b_ float64
	// Cached value of the sum of all ContrastSensitivity() values over all
	// quant matrix elements.
	total_csf_ float64
	quants_    []QuantData

	stats_ *ProcessStats
}

func quantMatrixGenerator(downsample bool, stats *ProcessStats) QuantMatrixGenerator {
	gen := QuantMatrixGenerator{
		downsample_: downsample,
		hscore_a_:   -1.0,
		hscore_b_:   -1.0,
		total_csf_:  0.0,
		stats_:      stats,
	}
	for k := 0; k < kDCTBlockSize; k++ {
		gen.total_csf_ += 3.0 * ContrastSensitivity(k)
	}
	return gen
}

func (qmg *QuantMatrixGenerator) GetNext(q [][kDCTBlockSize]int) bool {
	// This loop should terminate by return. This 1000 iteration limit is just a
	// precaution.
	for iter := 0; iter < 1000; iter++ {
		var hscore float64
		if qmg.hscore_b_ == -1.0 {
			if qmg.hscore_a_ == -1.0 {
				if !qmg.downsample_ {
					hscore = qmg.total_csf_
				}
			} else {
				if qmg.hscore_a_ < 5.0*qmg.total_csf_ {
					hscore = qmg.hscore_a_ + qmg.total_csf_
				} else {
					hscore = 2 * (qmg.hscore_a_ + qmg.total_csf_)
				}
			}
			if hscore > 100*qmg.total_csf_ {
				// We could not find a quantization matrix that creates enough
				// butteraugli error. This can happen if all dct coefficients are
				// close to zero in the original image.
				return false
			}
		} else if qmg.hscore_b_ == 0.0 {
			return false
		} else if qmg.hscore_a_ == -1.0 {
			hscore = 0.0
		} else {
			var lower_q [3][kDCTBlockSize]int
			var upper_q [3][kDCTBlockSize]int
			const kEps = 0.05
			qmg.getQuantMatrixWithHeuristicScore(
				(1-kEps)*qmg.hscore_a_+kEps*0.5*(qmg.hscore_a_+qmg.hscore_b_),
				lower_q[:])
			qmg.getQuantMatrixWithHeuristicScore(
				(1-kEps)*qmg.hscore_b_+kEps*0.5*(qmg.hscore_a_+qmg.hscore_b_),
				upper_q[:])
			if CompareQuantMatrices(lower_q[0][:], upper_q[0][:]) == 0 {
				return false
			}
			hscore = (qmg.hscore_a_ + qmg.hscore_b_) * 0.5
		}
		qmg.getQuantMatrixWithHeuristicScore(hscore, q)
		retry := false
		for i := 0; i < len(qmg.quants_); i++ {
			if CompareQuantMatrices(q[0][:], qmg.quants_[i].q[0][:]) == 0 {
				if qmg.quants_[i].dist_ok {
					qmg.hscore_a_ = hscore
				} else {
					qmg.hscore_b_ = hscore
				}
				retry = true
				break
			}
		}
		if !retry {
			return true
		}
	}
	return false
}

func (qmg *QuantMatrixGenerator) Add(data *QuantData) {
	qmg.quants_ = append(qmg.quants_, *data)
	hscore := QuantMatrixHeuristicScore(data.q[:])
	if data.dist_ok {
		qmg.hscore_a_ = std_maxFloat64(qmg.hscore_a_, hscore)
	} else {
		qmg.hscore_b_ = hscore
		if qmg.hscore_b_ != -1.0 {
			qmg.hscore_b_ = std_minFloat64(qmg.hscore_b_, hscore)
		}
	}
}

func (qmg *QuantMatrixGenerator) getQuantMatrixWithHeuristicScore(score float64, q [][kDCTBlockSize]int) {
	level := int(score / qmg.total_csf_)
	score -= float64(level) * qmg.total_csf_
	for k := kDCTBlockSize - 1; k >= 0; k-- {
		for c := 0; c < 3; c++ {
			q[c][kJPEGNaturalOrder[k]] = 2*level + 1
			if score > 0.0 {
				q[c][kJPEGNaturalOrder[k]] += 2
			}
		}
		score -= 3.0 * ContrastSensitivity(kJPEGNaturalOrder[k])
	}
}

func (p *Processor) TryQuantMatrix(jpg_in *JPEGData,
	target_mul float32,
	q [][kDCTBlockSize]int,
	img *OutputImage) QuantData {
	var data QuantData
	copy(data.q[:], q)
	img.CopyFromJpegData(jpg_in)
	img.ApplyGlobalQuantization(data.q[:])
	var buf []byte
	{
		jpg_out := *jpg_in
		img.SaveToJpegData(&jpg_out)
		buf = p.OutputJpeg(&jpg_out)
	}
	encoded_jpg := string(buf)
	// GUETZLI_LOG(stats_, "Iter %2d: %s quantization matrix:\n",
	//             stats_.counters[kNumItersCnt] + 1,
	//             img.FrameTypeStr().c_str());
	// GUETZLI_LOG_QUANT(stats_, q);
	// GUETZLI_LOG(stats_, "Iter %2d: %s GQ[%5.2f] Out[%7zd]",
	//             stats_.counters[kNumItersCnt] + 1,
	//             img.FrameTypeStr().c_str(),
	//             QuantMatrixHeuristicScore(q), encoded_jpg.size());
	p.stats_.counters[kNumItersCnt]++
	p.comparator_.Compare(img)
	data.dist_ok = p.comparator_.DistanceOK(float64(target_mul))
	data.jpg_size = len(encoded_jpg)
	p.MaybeOutput(encoded_jpg)
	return data
}

func (p *Processor) SelectQuantMatrix(jpg_in *JPEGData, downsample bool,
	best_q [][kDCTBlockSize]int,
	img *OutputImage) bool {
	qgen := quantMatrixGenerator(downsample, p.stats_)
	// Don't try to go up to exactly the target distance when selecting a
	// quantization matrix, since we will need some slack to do the frequency
	// masking later.
	const target_mul_high float32 = 0.97
	const target_mul_low float32 = 0.95

	best := p.TryQuantMatrix(jpg_in, target_mul_high, best_q, img)
	for {
		var q_next [][kDCTBlockSize]int
		if !qgen.GetNext(q_next) {
			break
		}

		data := p.TryQuantMatrix(jpg_in, target_mul_high, q_next, img)
		qgen.Add(&data)
		if CompareQuantData(&data, &best) {
			best = data
			if data.dist_ok && !p.comparator_.DistanceOK(float64(target_mul_low)) {
				break
			}
		}
	}

	copy(best_q[0][:], best.q[0][:])
	// GUETZLI_LOG(stats_, "\n%s selected quantization matrix:\n",
	//             downsample ? "YUV420" : "YUV444");
	// GUETZLI_LOG_QUANT(stats_, best_q)
	return best.dist_ok
}

// pair is a sortable type containing an int and a float.
// Sort order is f ASC
type pair struct {
	i int
	f float32
}

type pairs []pair

func (p pairs) Len() int           { return len(p) }
func (p pairs) Less(i, j int) bool { return p[i].f < p[j].f }
func (p pairs) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// REQUIRES: block[c*64...(c*64+63)] is all zero if (comp_mask & (1<<c)) == 0.
func (p *Processor) ComputeBlockZeroingOrder(
	block, orig_block []coeff_t,
	block_x, block_y, factor_x int,
	factor_y int, comp_mask byte, img *OutputImage) (output_order []CoeffData) {
	oldCsf := [kDCTBlockSize]byte{
		10, 10, 20, 40, 60, 70, 80, 90,
		10, 20, 30, 60, 70, 80, 90, 90,
		20, 30, 60, 70, 80, 90, 90, 90,
		40, 60, 70, 80, 90, 90, 90, 90,
		60, 70, 80, 90, 90, 90, 90, 90,
		70, 80, 90, 90, 90, 90, 90, 90,
		80, 90, 90, 90, 90, 90, 90, 90,
		90, 90, 90, 90, 90, 90, 90, 90,
	}
	kWeight := [3]float64{1.0, 0.22, 0.20}
	var input_order pairs
	for c := 0; c < 3; c++ {
		if (comp_mask & (1 << uint(c))) == 0 {
			continue
		}
		for k := 1; k < kDCTBlockSize; k++ {
			idx := c*kDCTBlockSize + k
			if block[idx] != 0 {
				var score float32
				if p.params_.new_zeroing_model {
					score = float32(std_abs(int(orig_block[idx])))*csf[idx] + bias[idx]
				} else {
					score = float32(
						(math.Abs(float64(orig_block[idx])) - float64(kJPEGZigZagOrder[k])/64.0) *
							kWeight[c] /
							float64(oldCsf[k]))
				}
				input_order = append(input_order, pair{idx, score})
			}
		}
	}
	sort.Sort(input_order)
	var processed_block [kBlockSize]coeff_t
	copy(processed_block[:], block)
	p.comparator_.SwitchBlock(block_x, block_y, factor_x, factor_y)
	for len(input_order) > 0 {
		best_err := float32(1e17)
		best_i := 0
		for i := 0; i < std_min(p.params_.zeroing_greedy_lookahead, len(input_order)); i++ {
			var candidate_block [kBlockSize]coeff_t
			copy(candidate_block[:], processed_block[:])
			idx := input_order[i].i
			candidate_block[idx] = 0
			for c := 0; c < 3; c++ {
				if comp_mask&(1<<uint(c)) != 0 {
					img.component(c).SetCoeffBlock(
						block_x, block_y, candidate_block[c*kDCTBlockSize:])
				}
			}
			max_err := float32(0)
			for iy := 0; iy < factor_y; iy++ {
				for ix := 0; ix < factor_x; ix++ {
					block_xx := block_x*factor_x + ix
					block_yy := block_y*factor_y + iy
					if 8*block_xx < img.width() && 8*block_yy < img.height() {
						err := float32(p.comparator_.CompareBlock(img, ix, iy))
						max_err = std_maxFloat32(max_err, err)
					}
				}
			}
			if max_err < best_err {
				best_err = max_err
				best_i = i
			}
		}
		idx := input_order[best_i].i
		processed_block[idx] = 0
		input_order = input_order[best_i:]
		output_order = append(output_order, CoeffData{idx, best_err})
		for c := 0; c < 3; c++ {
			if comp_mask&(1<<uint(c)) != 0 {
				img.component(c).SetCoeffBlock(
					block_x, block_y, processed_block[c*kDCTBlockSize:])
			}
		}
	}
	// Make the block error values monotonic.
	min_err := float32(1e10)
	for i := len(output_order) - 1; i >= 0; i-- {
		min_err = std_minFloat32(min_err, output_order[i].block_err)
		output_order[i].block_err = min_err
	}
	// Cut off at the block error limit.
	num := 0
	for num < len(output_order) && output_order[num].block_err <= p.comparator_.BlockErrorLimit() {
		num++
	}
	output_order = output_order[:num]
	// Restore *img to the same state as it was at the start of this function.
	for c := 0; c < 3; c++ {
		if comp_mask&(1<<uint(c)) != 0 {
			img.component(c).SetCoeffBlock(
				block_x, block_y, block[c*kDCTBlockSize:])
		}
	}
	return
}

func UpdateACHistogram(weight int,
	coeffs []coeff_t,
	q []int,
	ac_histogram *JpegHistogram) {
	r := 0
	for k := 1; k < 64; k++ {
		k_nat := kJPEGNaturalOrder[k]
		coeff := coeffs[k_nat]
		if coeff == 0 {
			r++
			continue
		}
		for r > 15 {
			ac_histogram.AddW(0xf0, weight)
			r -= 16
		}
		nbits := Log2FloorNonZero(uint32(std_abs(int(coeff)/q[k_nat]))) + 1
		symbol := (r << 4) + nbits
		ac_histogram.AddW(symbol, weight)
		r = 0
	}
	if r > 0 {
		ac_histogram.AddW(0, weight)
	}
}

func ComputeEntropyCodes(histograms []JpegHistogram) (int, []byte) {
	clustered := make([]JpegHistogram, len(histograms))
	copy(clustered, histograms) // TODO PATAPON is this intended as a copy?
	num := len(histograms)
	indexes := make([]int, len(histograms))
	clustered_depths := make([]byte, len(histograms)*kSize)
	ClusterHistograms(clustered, &num, indexes, clustered_depths)
	depths := make([]byte, len(clustered_depths))
	for i := 0; i < len(histograms); i++ {
		copy(depths[i*kSize:(i+1)*kSize], clustered_depths[indexes[i]*kSize:])
	}
	var histogram_size int
	for i := 0; i < num; i++ {
		histogram_size += HistogramHeaderCost(&clustered[i]) / 8
	}
	return histogram_size, depths
}

func EntropyCodedDataSize(histograms []JpegHistogram, depths []byte) int {
	numbits := 0
	for i := 0; i < len(histograms); i++ {
		numbits += HistogramEntropyCost(&histograms[i], depths[i*kSize:])
	}
	return (numbits + 7) / 8
}

func EstimateDCSize(jpg *JPEGData) int {
	histograms := BuildDCHistograms(jpg)
	num := len(histograms)
	indexes := make([]int, num)
	depths := make([]byte, num*kSize)
	return ClusterHistograms(histograms, &num, indexes, depths)
}

func (p *Processor) SelectFrequencyMasking(jpg *JPEGData, img *OutputImage,
	comp_mask byte,
	target_mul float64,
	stop_early bool) {
	width := img.width()
	height := img.height()
	ncomp := len(jpg.components)
	last_c := Log2FloorNonZero(uint32(comp_mask))
	if int(last_c) >= len(jpg.components) {
		return
	}
	factor_x := img.component(last_c).factor_x()
	factor_y := img.component(last_c).factor_y()
	block_width := (width + 8*factor_x - 1) / (8 * factor_x)
	block_height := (height + 8*factor_y - 1) / (8 * factor_y)
	num_blocks := block_width * block_height

	candidate_coeff_offsets := make([]int, num_blocks+1)
	candidate_coeffs := make([]byte, 0, 60*num_blocks)
	candidate_coeff_errors := make([]float32, 0, 60*num_blocks)
	block_order := make([]CoeffData, 3*kDCTBlockSize)
	p.comparator_.StartBlockComparisons()
	for block_y, block_ix := 0, 0; block_y < block_height; block_y++ {
		for block_x := 0; block_x < block_width; block_x, block_ix = block_x+1, block_ix+1 {
			var block [kBlockSize]coeff_t
			var orig_block [kBlockSize]coeff_t
			for c := 0; c < 3; c++ {
				if comp_mask&(1<<uint(c)) != 0 {
					assert(img.component(c).factor_x() == factor_x)
					assert(img.component(c).factor_y() == factor_y)
					img.component(c).GetCoeffBlock(block_x, block_y, block[c*kDCTBlockSize:])
					comp := jpg.components[c]
					jpg_block_ix := block_y*comp.width_in_blocks + block_x
					copy(orig_block[c*kDCTBlockSize:(c+1)*kDCTBlockSize],
						comp.coeffs[jpg_block_ix*kDCTBlockSize:])
				}
			}
			block_order = p.ComputeBlockZeroingOrder(block[:], orig_block[:], block_x, block_y, factor_x, factor_y, comp_mask, img)
			candidate_coeff_offsets[block_ix] = len(candidate_coeffs)
			for i := 0; i < len(block_order); i++ {
				candidate_coeffs = append(candidate_coeffs, byte(block_order[i].idx))
				candidate_coeff_errors = append(candidate_coeff_errors, block_order[i].block_err)
			}
		}
	}
	p.comparator_.FinishBlockComparisons()
	candidate_coeff_offsets[num_blocks] = len(candidate_coeffs)

	ac_histograms := make([]JpegHistogram, ncomp)
	var jpg_header_size, dc_size int
	{
		jpg_out := *jpg
		img.SaveToJpegData(&jpg_out)
		jpg_header_size = JpegHeaderSize(&jpg_out, p.params_.clear_metadata)
		dc_size = EstimateDCSize(&jpg_out)
		BuildACHistograms(&jpg_out, ac_histograms)
	}
	ac_histogram_size, ac_depths := ComputeEntropyCodes(ac_histograms)
	base_size := jpg_header_size + dc_size + ac_histogram_size +
		EntropyCodedDataSize(ac_histograms, ac_depths)
	prev_size := base_size

	max_block_error := make([]float32, num_blocks)
	last_indexes := make([]int, num_blocks)

	first_up_iter := true
	for direction := range []int{1, -1} {
		for {
			if stop_early && direction == -1 {
				if 100*prev_size > 101*len(p.final_output_.jpeg_data) {
					// If we are down-adjusting the error, the output size will only keep
					// increasing.
					// TODO(user): Do this check always by comparing only the size
					// of the currently processed components.
					break
				}
			}
			var global_order pairs
			var blocks_to_change int
			var block_weight []float32
			for rblock := 1; rblock <= 4; rblock++ {
				block_weight = make([]float32, num_blocks)
				distmap := make([]float32, width*height)
				if !first_up_iter {
					distmap = p.comparator_.distmap()
				}
				p.comparator_.ComputeBlockErrorAdjustmentWeights(
					direction, rblock, target_mul, factor_x, factor_y, distmap, block_weight)
				global_order = nil
				blocks_to_change = 0
				for block_y, block_ix := 0, 0; block_y < block_height; block_y++ {
					for block_x := 0; block_x < block_width; block_x, block_ix = block_x+1, block_ix+1 {
						last_index := last_indexes[block_ix]
						offset := candidate_coeff_offsets[block_ix]
						num_candidates := candidate_coeff_offsets[block_ix+1] - offset
						candidate_errors := candidate_coeff_errors[offset:]
						max_err := max_block_error[block_ix]
						if block_weight[block_ix] == 0 {
							continue
						}
						if direction > 0 {
							for i := last_index; i < num_candidates; i++ {
								val := float32((candidate_errors[i] - max_err) /
									block_weight[block_ix])
								global_order = append(global_order, pair{block_ix, val})
							}
							if last_index < num_candidates {
								blocks_to_change++
							}
						} else {
							for i := last_index - 1; i >= 0; i-- {
								val := float32((max_err - candidate_errors[i]) /
									block_weight[block_ix])
								global_order = append(global_order, pair{block_ix, val})
							}
							if last_index > 0 {
								blocks_to_change++
							}
						}
					}
				}
				if len(global_order) > 0 {
					// If we found something to adjust with the current block adjustment
					// radius, we can stop and adjust the blocks we have.
					break
				}
			}

			if len(global_order) > 0 {
				break
			}

			sort.Sort(global_order)

			rel_size_delta := 0.0005
			if direction > 0 {
				rel_size_delta = 0.01
			}
			if direction > 0 && p.comparator_.DistanceOK(1.0) {
				rel_size_delta = 0.05
			}
			min_size_delta := float64(base_size) * rel_size_delta

			var coeffs_to_change_per_block float32
			if direction > 0 {
				coeffs_to_change_per_block = 2.0
			} else {
				coeffs_to_change_per_block = float32(factor_x*factor_y) * 0.2
			}
			min_coeffs_to_change := coeffs_to_change_per_block * float32(blocks_to_change)

			if first_up_iter {
				limit := float32(0.75 * p.comparator_.BlockErrorLimit())
				it := sort.Search(len(global_order), func(i int) bool {
					return global_order[i].f >= limit
				})
				min_coeffs_to_change = std_maxFloat32(min_coeffs_to_change, float32(it))
				first_up_iter = false
			}

			changed_blocks := map[int]bool{}
			val_threshold := float32(0.0)
			changed_coeffs := 0
			est_jpg_size := prev_size
			for i := 0; i < len(global_order); i++ {
				block_ix := global_order[i].i
				block_x := block_ix % block_width
				block_y := block_ix / block_width
				last_idx := last_indexes[block_ix]
				offset := candidate_coeff_offsets[block_ix]
				candidates := candidate_coeffs[offset:]
				idx := candidates[last_idx+std_min(direction, 0)]
				c := idx / kDCTBlockSize
				k := idx % kDCTBlockSize
				quant := img.component(int(c)).quant()
				comp := &jpg.components[c]
				jpg_block_ix := block_y*comp.width_in_blocks + block_x
				newval := 0
				if direction <= 0 {
					newval = int(Quantize(comp.coeffs[jpg_block_ix*kDCTBlockSize+int(k)], quant[k]))
				}
				var block [kDCTBlockSize]coeff_t
				img.component(int(c)).GetCoeffBlock(block_x, block_y, block[:])
				UpdateACHistogram(-1, block[:], quant, &ac_histograms[c])
				block[k] = coeff_t(newval)
				UpdateACHistogram(1, block[:], quant, &ac_histograms[c])
				img.component(int(c)).SetCoeffBlock(block_x, block_y, block[:])
				last_indexes[block_ix] += direction
				changed_blocks[block_ix] = true
				val_threshold = global_order[i].f
				changed_coeffs++
				const kEntropyCodeUpdateFreq = 10
				if i%kEntropyCodeUpdateFreq == 0 {
					ac_histogram_size, ac_depths = ComputeEntropyCodes(ac_histograms)
				}
				est_jpg_size = jpg_header_size + dc_size + ac_histogram_size +
					EntropyCodedDataSize(ac_histograms, ac_depths)
				if float32(changed_coeffs) > min_coeffs_to_change &&
					math.Abs(float64(est_jpg_size-prev_size)) > min_size_delta {
					break
				}
			}
			// global_order_size := len(global_order)
			global_order = nil

			for i := 0; i < num_blocks; i++ {
				max_block_error[i] += block_weight[i] * val_threshold * float32(direction)
			}

			p.stats_.counters[kNumItersCnt]++
			if direction > 0 {
				p.stats_.counters[kNumItersUpCnt]++
			} else {
				p.stats_.counters[kNumItersDownCnt]++
			}
			var encoded_jpg string
			{
				jpg_out := *jpg
				img.SaveToJpegData(&jpg_out)
				encoded_jpg_bytes := p.OutputJpeg(&jpg_out)
				encoded_jpg = string(encoded_jpg_bytes)
			}
			// GUETZLI_LOG(stats_,
			//             "Iter %2d: %s(%d) %s Coeffs[%d/%zd] "
			//             "Blocks[%zd/%d/%d] ValThres[%.4f] Out[%7zd] EstErr[%.2f%%]",
			//             stats_.counters[kNumItersCnt], img.FrameTypeStr().c_str(),
			//             comp_mask, direction > 0 ? "up" : "down", changed_coeffs,
			//             global_order_size, changed_blocks.size(),
			//             blocks_to_change, num_blocks, val_threshold,
			//             encoded_jpg.size(),
			//             100.0 - (100.0 * est_jpg_size) / encoded_jpg.size());
			p.comparator_.Compare(img)
			p.MaybeOutput(encoded_jpg)
			prev_size = est_jpg_size
		}
	}
}

func IsGrayscale(jpg *JPEGData) bool {
	for c := 1; c < 3; c++ {
		comp := jpg.components[c]
		for _, c := range comp.coeffs {
			if c != 0 {
				return false
			}
		}
	}
	return true
}

func (p *Processor) ProcessJpegData(params *Params, jpg_in *JPEGData,
	comparator Comparator, out *GuetzliOutput,
	stats *ProcessStats) bool {
	p.params_ = *params
	p.comparator_ = comparator
	p.final_output_ = out
	p.stats_ = stats

	if params.butteraugli_target > 2.0 {
		log.Println("Guetzli should be called with quality >= 84, otherwise the\n",
			"output will have noticeable artifacts. If you want to\n",
			"proceed anyway, please edit the source code.\n")
		return false
	}
	if len(jpg_in.components) != 3 || !HasYCbCrColorSpace(jpg_in) {
		log.Println("Only YUV color space input jpeg is supported\n")
		return false
	}
	input_is_420 := false
	switch {
	case jpg_in.Is444():
		input_is_420 = false
	case jpg_in.Is420():
		input_is_420 = true
	default:
		log.Println("Unsupported sampling factors:")
		for i := 0; i < len(jpg_in.components); i++ {
			log.Printf(" %dx%d", jpg_in.components[i].h_samp_factor,
				jpg_in.components[i].v_samp_factor)
		}
		log.Println()
		return false
	}
	var q_in [3][kDCTBlockSize]int
	// Output the original image, in case we do not manage to create anything
	// with a good enough quality.
	encoded_jpg := string(p.OutputJpeg(jpg_in))
	p.final_output_.score = -1
	GUETZLI_LOG(stats, "Original Out[%7zd]", len(encoded_jpg))
	if p.comparator_ == nil {
		GUETZLI_LOG(stats, " <image too small for Butteraugli>\n")
		p.final_output_.jpeg_data = encoded_jpg
		p.final_output_.distmap = make([]float32, jpg_in.width*jpg_in.height)
		p.final_output_.distmap_aggregate = 0
		p.final_output_.score = float64(len(encoded_jpg))
		// Butteraugli doesn't work with images this small.
		return true
	}
	{
		jpg := *jpg_in
		RemoveOriginalQuantization(&jpg, q_in[:])
		img := OutputImage{width_: jpg.width, height_: jpg.height}
		img.CopyFromJpegData(&jpg)
		p.comparator_.Compare(&img)
	}
	p.MaybeOutput(encoded_jpg)
	try_420 := 0
	if input_is_420 || p.params_.force_420 || (p.params_.try_420 && !IsGrayscale(jpg_in)) {
		try_420 = 1
	}
	force_420 := 0
	if input_is_420 || p.params_.force_420 {
		force_420 = 1
	}
	for downsample := force_420; downsample <= try_420; downsample++ {
		jpg := *jpg_in
		RemoveOriginalQuantization(&jpg, q_in[:])
		img := OutputImage{width_: jpg.width, height_: jpg.height}
		img.CopyFromJpegData(&jpg)
		if downsample != 0 {
			p.DownsampleImage(&img)
			img.SaveToJpegData(&jpg)
		}
		var best_q [3][kDCTBlockSize]int
		copy(best_q[:], q_in[:])
		if !p.SelectQuantMatrix(&jpg, downsample != 0, best_q[:], &img) {
			for c := 0; c < 3; c++ {
				for i := 0; i < kDCTBlockSize; i++ {
					best_q[c][i] = 1
				}
			}
		}
		img.CopyFromJpegData(&jpg)
		img.ApplyGlobalQuantization(best_q[:])

		if downsample == 0 {
			p.SelectFrequencyMasking(&jpg, &img, 7, 1.0, false)
		} else {
			ymul := 0.97
			if len(jpg.components) == 1 {
				ymul = 1.0
			}
			p.SelectFrequencyMasking(&jpg, &img, 1, ymul, false)
			p.SelectFrequencyMasking(&jpg, &img, 6, 1.0, true)
		}
	}

	return true
}

func ProcessJpegData(params *Params, jpg_in *JPEGData,
	comparator Comparator, out *GuetzliOutput,
	stats *ProcessStats) bool {
	var processor Processor
	return processor.ProcessJpegData(params, jpg_in, comparator, out, stats)
}

func Process_(params *Params, stats *ProcessStats, data []byte) (ok bool, jpg_out string) {
	var jpg JPEGData
	if !ReadJpeg(data, JPEG_READ_ALL, &jpg) {
		log.Printf("Can't read jpg data from input file\n")
		return false, ""
	}
	if !CheckJpegSanity(&jpg) {
		log.Printf("Unsupported input JPEG (unexpectedly large coefficient values).\n")
		return false, ""
	}
	rgb := DecodeJpegToRGB(&jpg)
	if len(rgb) == 0 {
		log.Printf("Unsupported input JPEG file (e.g. unsupported " +
			"downsampling mode).\nPlease provide the input image as " +
			"a PNG file.\n")
		return false, ""
	}
	var out GuetzliOutput
	var dummy_stats ProcessStats
	if stats == nil {
		stats = &dummy_stats
	}
	var comparator *ButteraugliComparator
	if jpg.width >= 32 && jpg.height >= 32 {
		comparator = NewButteraugliComparator(jpg.width, jpg.height, rgb, params.butteraugli_target, stats)
	}
	ok = ProcessJpegData(params, &jpg, comparator, &out, stats)
	jpg_out = out.jpeg_data
	return ok, jpg_out
}

func Process__(params *Params, stats *ProcessStats, rgb []byte, w, h int) (ok bool, jpg_out string) {
	var jpg JPEGData
	if !EncodeRGBToJpeg(rgb, w, h, &jpg) {
		log.Printf("Could not create jpg data from rgb pixels\n")
		return false, ""
	}
	var out GuetzliOutput
	var dummy_stats ProcessStats
	if stats == nil {
		stats = &dummy_stats
	}
	var comparator *ButteraugliComparator
	if jpg.width >= 32 && jpg.height >= 32 {
		comparator = NewButteraugliComparator(jpg.width, jpg.height, rgb, params.butteraugli_target, stats)
	}
	ok = ProcessJpegData(params, &jpg, comparator, &out, stats)
	jpg_out = out.jpeg_data
	return ok, jpg_out
}
