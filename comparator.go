package guetzli_patapon

// Represents a baseline image, a comparison metric and an image acceptance
// criteria based on this metric.
type Comparator interface {

	// Compares img with the baseline image and saves the resulting distance map
	// inside the object. The provided image must have the same dimensions as the
	// baseline image.
	Compare(img *OutputImage)

	// Must be called before any CompareBlock() calls can be called.
	StartBlockComparisons()
	// No more CompareBlock() calls can be called after this.
	FinishBlockComparisons()

	// Sets the coordinates of the current macro-block for the purpose of
	// CompareBlock() calls.
	SwitchBlock(block_x, block_y, factor_x, factor_y int)

	// Compares the 8x8 block with offsets (off_x, off_y) within the current
	// macro-block of the baseline image with the same block of img and returns
	// the resulting per-block distance. The interpretation of the returned
	// distance depends on the comparator used.
	CompareBlock(img *OutputImage, off_x, off_y int) float64

	// Returns the combined score of the output image in the last Compare() call
	// (or the baseline image, if Compare() was not called yet), based on output
	// size and the similarity metric.
	ScoreOutputSize(size int) float64

	// Returns true if the argument of the last Compare() call (or the baseline
	// image, if Compare() was not called yet) meets the image acceptance
	// criteria. The target_mul modifies the acceptance criteria used in this call
	// the following way:
	//    = 1.0 : the original acceptance criteria is used,
	//    < 1.0 : a more strict acceptance criteria is used,
	//    > 1.0 : a less strict acceptance criteria is used.
	DistanceOK(target_mul float64) bool

	// Returns the distance map between the baseline image and the image in the
	// last Compare() call (or the baseline image, if Compare() was not called
	// yet).
	// The dimensions of the distance map are the same as the baseline image.
	// The interpretation of the distance values depend on the comparator used.
	distmap() []float32

	// Returns an aggregate distance or similarity value between the baseline
	// image and the image in the last Compare() call (or the baseline image, if
	// Compare() was not called yet).
	// The interpretation of this aggregate value depends on the comparator used.
	distmap_aggregate() float32

	// Returns a heuristic cutoff on block errors in the sense that we won't
	// consider distortions where a block error is greater than this.
	BlockErrorLimit() float32
	// Given the search direction (+1 for upwards and -1 for downwards) and the
	// current distance map, fills in *block_weight image with the relative block
	// error adjustment weights.
	// The target_mul param has the same semantics as in DistanceOK().
	// Note that this is essentially a static function in the sense that it does
	// not depend on the last Compare() call.
	ComputeBlockErrorAdjustmentWeights(
		direction int, max_block_dist int, target_mul float64, factor_x int,
		factor_y int, distmap []float32,
		block_weight []float32)
}
