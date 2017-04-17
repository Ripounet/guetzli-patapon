package guetzli_patapon

// TODO
// TODO
// TODO

type ButteraugliButteraugliComparator struct{}

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
