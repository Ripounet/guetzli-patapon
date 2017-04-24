package guetzli_patapon

// Function pointer type used to write len bytes into buf. Returns the
// number of bytes written or -1 on error.
type JPEGOutputHook func(data interface{}, buf []byte) int

// Output callback function with associated data.
type JPEGOutput struct {
	cb   JPEGOutputHook
	data interface{}
}

func (out *JPEGOutput) Write(buf []byte) bool {
	return len(buf) == 0 || out.cb(out.data, buf) == len(buf)
}

type HuffmanCodeTable struct {
	depth [256]byte
	code  [256]int
}

const kSize = kJpegHuffmanAlphabetSize + 1
const JpegHistogram_kSize = kSize

type JpegHistogram struct {
	counts [kSize]uint32
}

func NewJpegHistogram() *JpegHistogram {
	h := new(JpegHistogram)
	h.Clear()
	return h
}

func (h *JpegHistogram) Clear() {
	for i := range h.counts {
		h.counts[i] = 0
	}
	h.counts[kSize-1] = 1
}

func (h *JpegHistogram) Add(symbol int) {
	h.counts[symbol] += 2
}

func (h *JpegHistogram) AddW(symbol int, weight int) {
	h.counts[symbol] += uint32(2 * weight)
}

func (h *JpegHistogram) AddHistogram(other *JpegHistogram) {
	for i := 0; i+1 < kSize; i++ {
		h.counts[i] += other.counts[i]
	}
	h.counts[kSize-1] = 1
}

func (h *JpegHistogram) NumSymbols() int {
	n := 0
	for i := 0; i+1 < kSize; i++ {
		if h.counts[i] > 0 {
			n++
		}
	}
	return n
}

////////////////////////////////

const kJpegPrecision = 8

// Writes len bytes from buf, using the out callback.
func JPEGWrite(out JPEGOutput, buf []byte) bool {
	const kBlockSize = 1 << 30
	pos := 0
	for len(buf)-pos > int(kBlockSize) {
		if !out.Write(buf[pos : pos+kBlockSize]) {
			return false
		}
		pos += kBlockSize
	}
	return out.Write(buf[pos:])
}

func EncodeMetadata(jpg *JPEGData, strip_metadata bool, out JPEGOutput) bool {
	if strip_metadata {
		kApp0Data := []byte{
			0xff, 0xe0, 0x00, 0x10, // APP0
			0x4a, 0x46, 0x49, 0x46, 0x00, // 'JFIF'
			0x01, 0x01, // v1.01
			0x00, 0x00, 0x01, 0x00, 0x01, // aspect ratio = 1:1
			0x00, 0x00, // thumbnail width/height
		}
		return JPEGWrite(out, kApp0Data)
	}
	ok := true
	for i := 0; i < len(jpg.app_data); i++ {
		data := []byte{0xff}
		ok = ok && JPEGWrite(out, data)
		ok = ok && JPEGWrite(out, []byte(jpg.app_data[i]))
	}
	for i := 0; i < len(jpg.com_data); i++ {
		data := []byte{0xff, 0xfe}
		ok = ok && JPEGWrite(out, data)
		ok = ok && JPEGWrite(out, []byte(jpg.com_data[i]))
	}
	return ok
}

func EncodeDQT(quant []JPEGQuantTable, out JPEGOutput) bool {
	marker_len := 2
	for i := 0; i < len(quant); i++ {
		marker_len += 1 + kDCTBlockSize
		if quant[i].precision != 0 {
			marker_len += kDCTBlockSize
		}
	}
	data := make([]byte, marker_len+2)
	pos := 0
	data[pos] = 0xff
	pos++
	data[pos] = 0xdb
	pos++
	data[pos] = byte(marker_len >> 8)
	pos++
	data[pos] = byte(marker_len & 0xff)
	pos++
	for i := 0; i < len(quant); i++ {
		table := &quant[i]
		data[pos] = byte((table.precision << 4) + table.index)
		pos++
		for k := 0; k < kDCTBlockSize; k++ {
			val := table.values[kJPEGNaturalOrder[k]]
			if table.precision != 0 {
				data[pos] = byte(val >> 8)
				pos++
			}
			data[pos] = byte(val & 0xff)
			pos++
		}
	}
	return JPEGWrite(out, data[:pos])
}

func EncodeSOF(jpg *JPEGData, out JPEGOutput) bool {
	ncomps := len(jpg.components)
	marker_len := 8 + 3*ncomps
	data := make([]byte, marker_len+2)
	pos := 0
	data[pos] = 0xff
	pos++
	data[pos] = 0xc1
	pos++
	data[pos] = byte(marker_len >> 8)
	pos++
	data[pos] = byte(marker_len & 0xff)
	pos++
	data[pos] = kJpegPrecision
	pos++
	data[pos] = byte(jpg.height >> 8)
	pos++
	data[pos] = byte(jpg.height & 0xff)
	pos++
	data[pos] = byte(jpg.width >> 8)
	pos++
	data[pos] = byte(jpg.width & 0xff)
	pos++
	data[pos] = byte(ncomps)
	pos++
	for i := 0; i < ncomps; i++ {
		data[pos] = byte(jpg.components[i].id)
		pos++
		data[pos] = byte((jpg.components[i].h_samp_factor << 4) | (jpg.components[i].v_samp_factor))
		pos++
		quant_idx := jpg.components[i].quant_idx
		if quant_idx >= len(jpg.quant) {
			return false
		}
		data[pos] = byte(jpg.quant[quant_idx].index)
		pos++
	}
	return JPEGWrite(out, data[:pos])
}

// Builds a JPEG-style huffman code from the given bit depths.
func BuildHuffmanCode(depth []byte, counts, values []int) {
	for i := 0; i < JpegHistogram_kSize; i++ {
		if depth[i] > 0 {
			counts[depth[i]]++
		}
	}
	var offset [kJpegHuffmanMaxBitLength + 1]int
	for i := 1; i <= kJpegHuffmanMaxBitLength; i++ {
		offset[i] = offset[i-1] + counts[i-1]
	}
	for i := 0; i < JpegHistogram_kSize; i++ {
		if depth[i] > 0 {
			values[offset[depth[i]]] = i
			offset[depth[i]]++
		}
	}
}

func BuildHuffmanCodeTable(counts, values []int, table *HuffmanCodeTable) {
	var huffcode [256]int
	var huffsize [256]int
	p := 0
	for l := 1; l <= kJpegHuffmanMaxBitLength; l++ {
		i := counts[l]
		for ; i > 0; i-- {
			huffsize[p] = l
			p++
		}
	}

	if p == 0 {
		return
	}

	huffsize[p-1] = 0
	lastp := p - 1

	code := 0
	si := huffsize[0]
	p = 0
	for huffsize[p] != 0 {
		for (huffsize[p]) == si {
			huffcode[p] = code
			p++
			code++
		}
		code <<= 1
		si++
	}
	for p = 0; p < lastp; p++ {
		i := values[p]
		table.depth[i] = byte(huffsize[p])
		table.code[i] = huffcode[p]
	}
}

// Updates ac_histogram with the counts of the AC symbols that will be added by
// a sequential jpeg encoder for this block. Every symbol is counted twice so
// that we can add a fake symbol at the end with count 1 to be the last (least
// frequent) symbol with the all 1 code.
func UpdateACHistogramForDCTBlock(coeffs []coeff_t, ac_histogram *JpegHistogram) {
	r := 0
	for k := 1; k < 64; k++ {
		coeff := coeffs[kJPEGNaturalOrder[k]]
		if coeff == 0 {
			r++
			continue
		}
		for r > 15 {
			ac_histogram.Add(0xf0)
			r -= 16
		}
		nbits := Log2FloorNonZero(uint32(std_abs(int(coeff)))) + 1
		symbol := (r << 4) + nbits
		ac_histogram.Add(symbol)
		r = 0
	}
	if r > 0 {
		ac_histogram.Add(0)
	}
}

func HistogramHeaderCost(histo *JpegHistogram) int {
	header_bits := 17 * 8
	for i := 0; i+1 < JpegHistogram_kSize; i++ {
		if histo.counts[i] > 0 {
			header_bits += 8
		}
	}
	return header_bits
}

func HistogramEntropyCost(histo *JpegHistogram, depths []byte) int {
	bits := 0
	for i := 0; i+1 < JpegHistogram_kSize; i++ {
		// JpegHistogram::Add() counts every symbol twice, so we have to divide by
		// two here.
		bits += int(histo.counts[i]/2) * int(depths[i]+byte(i&0xf))
	}
	// Estimate escape byte rate to be 0.75/256.
	bits += (bits*3 + 512) >> 10
	return bits
}

func BuildDCHistograms(jpg *JPEGData) (histo []JpegHistogram) {
	histo = make([]JpegHistogram, len(jpg.components))
	for i := range histo {
		histo[i].Clear()
	}

	for i := 0; i < len(jpg.components); i++ {
		c := &jpg.components[i]
		dc_histogram := &histo[i]
		var last_dc_coeff coeff_t
		for mcu_y := 0; mcu_y < jpg.MCU_rows; mcu_y++ {
			for mcu_x := 0; mcu_x < jpg.MCU_cols; mcu_x++ {
				for iy := 0; iy < c.v_samp_factor; iy++ {
					for ix := 0; ix < c.h_samp_factor; ix++ {
						block_y := mcu_y*c.v_samp_factor + iy
						block_x := mcu_x*c.h_samp_factor + ix
						block_idx := block_y*c.width_in_blocks + block_x
						dc_coeff := c.coeffs[block_idx<<6]
						diff := std_abs(int(dc_coeff - last_dc_coeff))
						nbits := Log2Floor(uint32(diff)) + 1
						dc_histogram.Add(nbits)
						last_dc_coeff = dc_coeff
					}
				}
			}
		}
	}
	return histo
}

func BuildACHistograms(jpg *JPEGData, histo []JpegHistogram) {
	for i := 0; i < len(jpg.components); i++ {
		c := &jpg.components[i]
		ac_histogram := &histo[i]
		for j := 0; j < len(c.coeffs); j += kDCTBlockSize {
			UpdateACHistogramForDCTBlock(c.coeffs[j:], ac_histogram)
		}
	}
}

// Size of everything except the Huffman codes and the entropy coded data.
func JpegHeaderSize(jpg *JPEGData, strip_metadata bool) int {
	num_bytes := 0
	num_bytes += 2 // SOI
	if strip_metadata {
		num_bytes += 18 // APP0
	} else {
		for i := 0; i < len(jpg.app_data); i++ {
			num_bytes += 1 + len(jpg.app_data[i])
		}
		for i := 0; i < len(jpg.com_data); i++ {
			num_bytes += 2 + len(jpg.com_data[i])
		}
	}
	// DQT
	num_bytes += 4
	for i := 0; i < len(jpg.quant); i++ {
		num_bytes += 1 + kDCTBlockSize
		if jpg.quant[i].precision != 0 {
			num_bytes += kDCTBlockSize
		}
	}
	num_bytes += 10 + 3*len(jpg.components) // SOF
	num_bytes += 4                          // DHT (w/o actual Huffman code data)
	num_bytes += 8 + 2*len(jpg.components)  // SOS
	num_bytes += 2                          // EOI
	num_bytes += len(jpg.tail_data)
	return num_bytes
}

func ClusterHistograms(histo []JpegHistogram, num *int, histo_indexes []int, depth []byte) int {
	for i := 0; i < *num*JpegHistogram_kSize; i++ {
		depth[0] = 0
	}
	var costs [kMaxComponents]int
	for i := 0; i < *num; i++ {
		histo_indexes[i] = i
		tree := make([]HuffmanTree, 2*JpegHistogram_kSize+1)
		CreateHuffmanTree(histo[i].counts[:], JpegHistogram_kSize,
			kJpegHuffmanMaxBitLength, tree,
			depth[i*JpegHistogram_kSize:])
		costs[i] = HistogramHeaderCost(&histo[i]) +
			HistogramEntropyCost(&histo[i], depth[i*JpegHistogram_kSize:])
	}
	orig_num := *num
	for *num > 1 {
		last := *num - 1
		second_last := *num - 2
		combined := histo[last] // this is a copy
		combined.AddHistogram(&histo[second_last])
		tree := make([]HuffmanTree, 2*JpegHistogram_kSize+1)
		var depth_combined [JpegHistogram_kSize]byte
		CreateHuffmanTree(combined.counts[:], JpegHistogram_kSize,
			kJpegHuffmanMaxBitLength, tree, depth_combined[:])
		cost_combined := (HistogramHeaderCost(&combined) +
			HistogramEntropyCost(&combined, depth_combined[:]))
		if cost_combined < costs[last]+costs[second_last] {
			histo[second_last] = combined
			histo[last].Clear()
			costs[second_last] = cost_combined
			copy(depth[second_last*JpegHistogram_kSize:], depth_combined[:])
			for i := 0; i < orig_num; i++ {
				if histo_indexes[i] == last {
					histo_indexes[i] = second_last
				}
			}
			(*num)--
		} else {
			break
		}
	}
	total_cost := 0
	for i := 0; i < *num; i++ {
		total_cost += costs[i]
	}
	return (total_cost + 7) / 8
}

func EstimateJpegDataSize(num_components int, histograms []JpegHistogram) int {
	assert(len(histograms) == 2*num_components)
	clustered := make([]JpegHistogram, len(histograms))
	copy(clustered, histograms)
	num_dc := num_components
	num_ac := num_components
	var indexes [kMaxComponents]int
	var depth [kMaxComponents * JpegHistogram_kSize]byte
	return (ClusterHistograms(clustered, &num_dc, indexes[:], depth[:]) +
		ClusterHistograms(clustered[num_components:], &num_ac, indexes[:], depth[:]))
}

// Writes DHT and SOS marker segments to out and fills in DC/AC Huffman tables
// for each component of the image.
func BuildAndEncodeHuffmanCodes(jpg *JPEGData, out JPEGOutput) (ok bool, dc_huff_tables, ac_huff_tables []HuffmanCodeTable) {
	ncomps := len(jpg.components)
	dc_huff_tables = make([]HuffmanCodeTable, ncomps)
	ac_huff_tables = make([]HuffmanCodeTable, ncomps)

	// Build separate DC histograms for each component.
	histograms := BuildDCHistograms(jpg)

	// Cluster DC histograms.
	num_dc_histo := ncomps
	var dc_histo_indexes [kMaxComponents]int
	depths := make([]byte, ncomps*JpegHistogram_kSize)
	ClusterHistograms(histograms, &num_dc_histo, dc_histo_indexes[:], depths[:])

	// Build separate AC histograms for each component.
	// histograms.resize(num_dc_histo + ncomps)
	for len(histograms) < num_dc_histo+ncomps {
		h := NewJpegHistogram()
		histograms = append(histograms, *h)
	}
	// depths.resize((num_dc_histo + ncomps) * JpegHistogram_kSize)
	for len(depths) < num_dc_histo+ncomps {
		depths = append(depths, 0)
	}
	BuildACHistograms(jpg, histograms[num_dc_histo:])

	// Cluster AC histograms.
	num_ac_histo := ncomps
	var ac_histo_indexes [kMaxComponents]int
	ClusterHistograms(histograms[num_dc_histo:], &num_ac_histo, ac_histo_indexes[:], depths[num_dc_histo*JpegHistogram_kSize:])

	// Compute DHT and SOS marker data sizes and start emitting DHT marker.
	num_histo := num_dc_histo + num_ac_histo
	// histograms.resize(num_histo)
	for len(histograms) <= num_histo {
		h := NewJpegHistogram()
		histograms = append(histograms, *h)
	}
	histograms = histograms[:num_histo]
	total_count := 0
	for i := 0; i < len(histograms); i++ {
		total_count += histograms[i].NumSymbols()
	}
	dht_marker_len := 2 + num_histo*(kJpegHuffmanMaxBitLength+1) + total_count
	sos_marker_len := 6 + 2*ncomps
	data := make([]byte, dht_marker_len+sos_marker_len+4)
	pos := 0
	data[pos] = 0xff
	pos++
	data[pos] = 0xc4
	pos++
	data[pos] = byte(dht_marker_len >> 8)
	pos++
	data[pos] = byte(dht_marker_len & 0xff)
	pos++

	// Compute Huffman codes for each histograms.
	for i := 0; i < num_histo; i++ {
		is_dc := i < num_dc_histo
		idx := tern(is_dc, i, i-num_dc_histo)
		var counts [kJpegHuffmanMaxBitLength + 1]int
		var values [JpegHistogram_kSize]int
		BuildHuffmanCode(depths[i*JpegHistogram_kSize:], counts[:], values[:])
		var table HuffmanCodeTable
		for j := 0; j < 256; j++ {
			table.depth[j] = 255
		}
		BuildHuffmanCodeTable(counts[:], values[:], &table)
		for c := 0; c < ncomps; c++ {
			if is_dc {
				if dc_histo_indexes[c] == idx {
					dc_huff_tables[c] = table
				}
			} else {
				if ac_histo_indexes[c] == idx {
					ac_huff_tables[c] = table
				}
			}
		}
		max_length := kJpegHuffmanMaxBitLength
		for max_length > 0 && counts[max_length] == 0 {
			max_length--
		}
		counts[max_length]--
		total_count := 0
		for j := 0; j <= max_length; j++ {
			total_count += counts[j]
		}

		data[pos] = byte(tern(is_dc, i, i-num_dc_histo+0x10))
		pos++
		for j := 1; j <= kJpegHuffmanMaxBitLength; j++ {
			data[pos] = byte(counts[j])
			pos++
		}
		for j := 0; j < total_count; j++ {
			data[pos] = byte(values[j])
			pos++
		}
	}

	// Emit SOS marker data.
	data[pos] = 0xff
	pos++
	data[pos] = 0xda
	pos++
	data[pos] = byte(sos_marker_len >> 8)
	pos++
	data[pos] = byte(sos_marker_len & 0xff)
	pos++
	data[pos] = byte(ncomps)
	pos++
	for i := 0; i < ncomps; i++ {
		data[pos] = byte(jpg.components[i].id)
		pos++
		data[pos] = byte((dc_histo_indexes[i] << 4) | ac_histo_indexes[i])
		pos++
	}
	data[pos] = 0
	pos++
	data[pos] = 63
	pos++
	data[pos] = 0
	pos++
	assert(pos == len(data))
	return JPEGWrite(out, data), dc_huff_tables, ac_huff_tables
}

func EncodeDCTBlockSequential(coeffs []coeff_t,
	dc_huff, ac_huff *HuffmanCodeTable,
	last_dc_coeff *int,
	bw *BitWriter) {
	var temp2, temp int
	temp2 = int(coeffs[0])
	temp = temp2 - *last_dc_coeff
	*last_dc_coeff = temp2
	temp2 = temp
	if temp < 0 {
		temp = -temp
		temp2--
	}
	nbits := Log2Floor(uint32(temp)) + 1
	bw.WriteBits(int(dc_huff.depth[nbits]), uint64(dc_huff.code[nbits]))
	if nbits > 0 {
		bw.WriteBits(nbits, uint64(temp2&((1<<uint(nbits))-1)))
	}
	r := 0
	for k := 1; k < 64; k++ {
		temp = int(coeffs[kJPEGNaturalOrder[k]])
		if temp == 0 {
			r++
			continue
		}
		if temp < 0 {
			temp = -temp
			temp2 = ^temp
		} else {
			temp2 = temp
		}
		for r > 15 {
			bw.WriteBits(int(ac_huff.depth[0xf0]), uint64(ac_huff.code[0xf0]))
			r -= 16
		}
		nbits := Log2FloorNonZero(uint32(temp)) + 1
		symbol := (r << 4) + nbits
		bw.WriteBits(int(ac_huff.depth[symbol]), uint64(ac_huff.code[symbol]))
		bw.WriteBits(nbits, uint64(temp2&((1<<uint(nbits))-1)))
		r = 0
	}
	if r > 0 {
		bw.WriteBits(int(ac_huff.depth[0]), uint64(ac_huff.code[0]))
	}
}

func EncodeScan(jpg *JPEGData,
	dc_huff_table, ac_huff_table []HuffmanCodeTable,
	out JPEGOutput) bool {
	var last_dc_coeff [kMaxComponents]int
	bw := NewBitWriter(1 << 17)
	for mcu_y := 0; mcu_y < jpg.MCU_rows; mcu_y++ {
		for mcu_x := 0; mcu_x < jpg.MCU_cols; mcu_x++ {
			// Encode one MCU
			for i := 0; i < len(jpg.components); i++ {
				c := &jpg.components[i]
				nblocks_y := c.v_samp_factor
				nblocks_x := c.h_samp_factor
				for iy := 0; iy < nblocks_y; iy++ {
					for ix := 0; ix < nblocks_x; ix++ {
						block_y := mcu_y*nblocks_y + iy
						block_x := mcu_x*nblocks_x + ix
						block_idx := block_y*c.width_in_blocks + block_x
						coeffs := c.coeffs[block_idx<<6:]
						EncodeDCTBlockSequential(coeffs, &dc_huff_table[i], &ac_huff_table[i],
							&last_dc_coeff[i], bw)
					}
				}
			}
			if bw.pos > (1 << 16) {
				if !JPEGWrite(out, bw.data[:bw.pos]) {
					return false
				}
				bw.pos = 0
			}
		}
	}
	bw.JumpToByteBoundary()
	return !bw.overflow && JPEGWrite(out, bw.data[:bw.pos])
}

func WriteJpeg(jpg *JPEGData, strip_metadata bool, out JPEGOutput) bool {
	kSOIMarker := [2]byte{0xff, 0xd8}
	kEOIMarker := [2]byte{0xff, 0xd9}
	var dc_codes, ac_codes []HuffmanCodeTable
	baehc := func() (ok bool) {
		ok, dc_codes, ac_codes = BuildAndEncodeHuffmanCodes(jpg, out)
		return ok
	}
	return (JPEGWrite(out, kSOIMarker[:]) &&
		EncodeMetadata(jpg, strip_metadata, out) &&
		EncodeDQT(jpg.quant, out) &&
		EncodeSOF(jpg, out) &&
		// BuildAndEncodeHuffmanCodes(jpg, out, &dc_codes, &ac_codes) &&
		baehc() &&
		EncodeScan(jpg, dc_codes, ac_codes, out) &&
		JPEGWrite(out, kEOIMarker[:]) &&
		(strip_metadata || JPEGWrite(out, []byte(jpg.tail_data))))
}

func NullOut(data interface{}, buf []byte) int {
	return len(buf)
}

// func BuildSequentialHuffmanCodes(jpg *JPEGData) (dc_huffman_code_tables, ac_huffman_code_tables *[]HuffmanCodeTable) {
// 	var out JPEGOutput
// 	return BuildAndEncodeHuffmanCodes(jpg, out)
// }
