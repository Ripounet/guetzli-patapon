package guetzli_patapon

// Mimic libjpeg's heuristics to guess jpeg color space.
// Requires that the jpg has 3 components.
func HasYCbCrColorSpace(jpg *JPEGData) bool {
	has_Adobe_marker := false
	Adobe_transform := byte(0)
	for _, app := range jpg.app_data {
		if app[0] == 0xe0 {
			return true
		}
		if app[0] == 0xee && len(app) >= 15 {
			has_Adobe_marker = true
			Adobe_transform = app[14]
		}
	}
	if has_Adobe_marker {
		return (Adobe_transform != 0)
	}
	cid0 := jpg.components[0].id
	cid1 := jpg.components[1].id
	cid2 := jpg.components[2].id
	return (cid0 != 'R' || cid1 != 'G' || cid2 != 'B')
}

func DecodeJpegToRGB(jpg *JPEGData) []byte {
	if len(jpg.components) == 1 ||
		(len(jpg.components) == 3 &&
			HasYCbCrColorSpace(jpg) && (jpg.Is420() || jpg.Is444())) {
		img := OutputImage{width_: jpg.width, height_: jpg.height}
		img.CopyFromJpegData(jpg)
		return img.ToSRGB_()
	}
	return nil
}
