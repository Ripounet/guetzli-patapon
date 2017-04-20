package guetzli_patapon

type JpegReadMode int

const (
	JPEG_READ_HEADER = iota // only basic headers
	JPEG_READ_TABLES        // headers and tables (quant, Huffman, ...)
	JPEG_READ_ALL           // everything
)

// Macros for commonly used error conditions.

func VERIFY_LEN(pos *int, n, length int) bool {
	if *pos+n > length {
		fprintf(stderr, "Unexpected end of input: pos=%d need=%d len=%d\n",
			int(*pos), int(n),
			int(length))
		// jpg.err = JPEG_UNEXPECTED_EOF
		panic(*pos + n) // TODO PATAPON not panic?
		return false
	}
	return true
}

func VERIFY_INPUT(var_, low, high int, code string) bool {
	if var_ < low || var_ > high {
		fprintf(stderr, "Invalid %s: %d\n", var_, int(var_))
		// jpg.err = JPEG_INVALID_ //## code;
		panic(code) // TODO PATAPON not panic?
		return false
	}
	return true
}

func VERIFY_MARKER_END(pos *int, start_pos, marker_len int) bool {
	if start_pos+marker_len != *pos {
		fprintf(stderr, "Invalid marker length: declared=%d actual=%d\n",
			int(marker_len),
			int(*pos-start_pos))
		// jpg.err = JPEG_WRONG_MARKER_SIZE
		panic("JPEG_WRONG_MARKER_SIZE") // TODO PATAPON not panic?
		return false
	}
	return true
}

func EXPECT_MARKER(pos int, length int, data []byte) bool {
	if pos+2 > length || data[pos] != 0xff {
		found := byte(0)
		if pos < length {
			found = data[pos]
		}
		fprintf(stderr, "Marker byte (0xff) expected, found: %d "+
			"pos=%d len=%d\n",
			found, int(pos),
			int(length))
		// jpg.err = JPEG_MARKER_BYTE_NOT_FOUND
		panic("JPEG_MARKER_BYTE_NOT_FOUND") // TODO not panic?
		return false
	}
	return true
}

func SignedLeftshift(v, s int) int {
	if v >= 0 {
		return (v << uint(s))
	} else {
		return -((-v) << uint(s))
	}
}

// Returns ceil(a/b).
func DivCeil(a, b int) int {
	return (a + b - 1) / b
}

func ReadUint8(data []byte, pos *int) int {
	v := data[*pos]
	(*pos)++
	return int(v)
}

func ReadUint16(data []byte, pos *int) int {
	v := (data[*pos] << 8) + data[*pos+1]
	*pos += 2
	return int(v)
}

// Reads the Start of Frame (SOF) marker segment and fills in *jpg with the
// parsed data.
func ProcessSOF(data []byte, length int,
	mode JpegReadMode, pos *int, jpg *JPEGData) bool {
	if jpg.width != 0 {
		fprintf(stderr, "Duplicate SOF marker.\n")
		jpg.err = JPEG_DUPLICATE_SOF
		return false
	}
	start_pos := *pos
	VERIFY_LEN(pos, 8, length)
	marker_len := ReadUint16(data, pos)
	precision := ReadUint8(data, pos)
	height := ReadUint16(data, pos)
	width := ReadUint16(data, pos)
	num_components := ReadUint8(data, pos)
	VERIFY_INPUT(precision, 8, 8, "PRECISION")
	VERIFY_INPUT(height, 1, 65535, "HEIGHT")
	VERIFY_INPUT(width, 1, 65535, "WIDTH")
	VERIFY_INPUT(num_components, 1, kMaxComponents, "NUMCOMP")
	VERIFY_LEN(pos, 3*num_components, length)
	jpg.height = height
	jpg.width = width
	jpg.components = make([]JPEGComponent, num_components)

	// Read sampling factors and quant table index for each component.
	ids_seen := make([]bool, 256)
	for i := 0; i < len(jpg.components); i++ {
		id := ReadUint8(data, pos)
		if ids_seen[id] { // (cf. section B.2.2, syntax of Ci)
			fprintf(stderr, "Duplicate ID %d in SOF.\n", id)
			jpg.err = JPEG_DUPLICATE_COMPONENT_ID
			return false
		}
		ids_seen[id] = true
		jpg.components[i].id = id
		factor := ReadUint8(data, pos)
		h_samp_factor := factor >> 4
		v_samp_factor := factor & 0xf
		VERIFY_INPUT(h_samp_factor, 1, 15, "SAMP_FACTOR")
		VERIFY_INPUT(v_samp_factor, 1, 15, "SAMP_FACTOR")
		jpg.components[i].h_samp_factor = h_samp_factor
		jpg.components[i].v_samp_factor = v_samp_factor
		jpg.components[i].quant_idx = ReadUint8(data, pos)
		jpg.max_h_samp_factor = std_max(jpg.max_h_samp_factor, h_samp_factor)
		jpg.max_v_samp_factor = std_max(jpg.max_v_samp_factor, v_samp_factor)
	}

	// We have checked above that none of the sampling factors are 0, so the max
	// sampling factors can not be 0.
	jpg.MCU_rows = DivCeil(jpg.height, jpg.max_v_samp_factor*8)
	jpg.MCU_cols = DivCeil(jpg.width, jpg.max_h_samp_factor*8)
	// Compute the block dimensions for each component.
	if mode == JPEG_READ_ALL {
		for i := 0; i < len(jpg.components); i++ {
			c := &jpg.components[i]
			if jpg.max_h_samp_factor%c.h_samp_factor != 0 ||
				jpg.max_v_samp_factor%c.v_samp_factor != 0 {
				fprintf(stderr, "Non-integral subsampling ratios.\n")
				jpg.err = JPEG_INVALID_SAMPLING_FACTORS
				return false
			}
			c.width_in_blocks = jpg.MCU_cols * c.h_samp_factor
			c.height_in_blocks = jpg.MCU_rows * c.v_samp_factor
			num_blocks := uint64(c.width_in_blocks * c.height_in_blocks)
			if num_blocks > (1 << 21) {
				// Refuse to allocate more than 1 GB of memory for the coefficients,
				// that is 2M blocks x 64 coeffs x 2 bytes per coeff x max 4 components.
				// TODO(user) Add this limit to a GuetzliParams struct.
				fprintf(stderr, "Image too large.\n")
				jpg.err = JPEG_IMAGE_TOO_LARGE
				return false
			}
			c.num_blocks = int(num_blocks)
			c.coeffs = make([]coeff_t, c.num_blocks*kDCTBlockSize)
		}
	}
	VERIFY_MARKER_END(pos, start_pos, marker_len)
	return true
}

// Reads the Start of Scan (SOS) marker segment and fills in *scan_info with the
// parsed data.
func ProcessSOS(data []byte, length int, pos *int, jpg *JPEGData) bool {
	start_pos := *pos
	VERIFY_LEN(pos, 3, length)
	marker_len := ReadUint16(data, pos)
	comps_in_scan := ReadUint8(data, pos)
	VERIFY_INPUT(comps_in_scan, 1, len(jpg.components), "COMPS_IN_SCAN")

	var scan_info JPEGScanInfo
	scan_info.components = make([]JPEGComponentScanInfo, comps_in_scan)
	VERIFY_LEN(pos, 2*comps_in_scan, length)
	ids_seen := make([]bool, 256)
	for i := 0; i < comps_in_scan; i++ {
		id := ReadUint8(data, pos)
		if ids_seen[id] { // (cf. section B.2.3, regarding CSj)
			fprintf(stderr, "Duplicate ID %d in SOS.\n", id)
			jpg.err = JPEG_DUPLICATE_COMPONENT_ID
			return false
		}
		ids_seen[id] = true
		found_index := false
		for j := 0; j < len(jpg.components); j++ {
			if jpg.components[j].id == id {
				scan_info.components[i].comp_idx = j
				found_index = true
			}
		}
		if !found_index {
			fprintf(stderr, "SOS marker: Could not find component with id %d\n", id)
			jpg.err = JPEG_COMPONENT_NOT_FOUND
			return false
		}
		c := ReadUint8(data, pos)
		dc_tbl_idx := c >> 4
		ac_tbl_idx := c & 0xf
		VERIFY_INPUT(dc_tbl_idx, 0, 3, "HUFFMAN_INDEX")
		VERIFY_INPUT(ac_tbl_idx, 0, 3, "HUFFMAN_INDEX")
		scan_info.components[i].dc_tbl_idx = dc_tbl_idx
		scan_info.components[i].ac_tbl_idx = ac_tbl_idx
	}
	VERIFY_LEN(pos, 3, length)
	scan_info.Ss = ReadUint8(data, pos)
	scan_info.Se = ReadUint8(data, pos)
	VERIFY_INPUT(scan_info.Ss, 0, 63, "START_OF_SCAN")
	VERIFY_INPUT(scan_info.Se, scan_info.Ss, 63, "END_OF_SCAN")
	c := ReadUint8(data, pos)
	scan_info.Ah = c >> 4
	scan_info.Al = c & 0xf
	// Check that all the Huffman tables needed for this scan are defined.
	for i := 0; i < comps_in_scan; i++ {
		found_dc_table := false
		found_ac_table := false
		for j := 0; j < len(jpg.huffman_code); j++ {
			slot_id := jpg.huffman_code[j].slot_id
			if slot_id == scan_info.components[i].dc_tbl_idx {
				found_dc_table = true
			} else if slot_id == scan_info.components[i].ac_tbl_idx+16 {
				found_ac_table = true
			}
		}
		if scan_info.Ss == 0 && !found_dc_table {
			fprintf(stderr, "SOS marker: Could not find DC Huffman table with index "+
				"%d\n", scan_info.components[i].dc_tbl_idx)
			jpg.err = JPEG_HUFFMAN_TABLE_NOT_FOUND
			return false
		}
		if scan_info.Se > 0 && !found_ac_table {
			fprintf(stderr, "SOS marker: Could not find AC Huffman table with index "+
				"%d\n", scan_info.components[i].ac_tbl_idx)
			jpg.err = JPEG_HUFFMAN_TABLE_NOT_FOUND
			return false
		}
	}
	jpg.scan_info = append(jpg.scan_info, scan_info)
	VERIFY_MARKER_END(pos, start_pos, marker_len)
	return true
}

// Reads the Define Huffman Table (DHT) marker segment and fills in *jpg with
// the parsed data. Builds the Huffman decoding table in either dc_huff_lut or
// ac_huff_lut, depending on the type and solt_id of Huffman code being read.
func ProcessDHT(data []byte, length int, mode JpegReadMode,
	dc_huff_lut, ac_huff_lut []HuffmanTableEntry,
	pos *int,
	jpg *JPEGData) bool {
	start_pos := *pos
	VERIFY_LEN(pos, 2, length)
	marker_len := ReadUint16(data, pos)
	if marker_len == 2 {
		fprintf(stderr, "DHT marker: no Huffman table found\n")
		jpg.err = JPEG_EMPTY_DHT
		return false
	}
	for *pos < start_pos+marker_len {
		VERIFY_LEN(pos, 1+kJpegHuffmanMaxBitLength, length)
		var huff JPEGHuffmanCode
		huff.slot_id = ReadUint8(data, pos)
		huffman_index := huff.slot_id
		is_ac_table := (huff.slot_id & 0x10) != 0
		var huff_lut []HuffmanTableEntry
		if is_ac_table {
			huffman_index -= 0x10
			VERIFY_INPUT(huffman_index, 0, 3, "HUFFMAN_INDEX")
			huff_lut = ac_huff_lut[huffman_index*kJpegHuffmanLutSize:]
		} else {
			VERIFY_INPUT(huffman_index, 0, 3, "HUFFMAN_INDEX")
			huff_lut = dc_huff_lut[huffman_index*kJpegHuffmanLutSize:]
		}
		huff.counts[0] = 0
		total_count := 0
		space := 1 << kJpegHuffmanMaxBitLength
		max_depth := 1
		for i := 1; i <= kJpegHuffmanMaxBitLength; i++ {
			count := ReadUint8(data, pos)
			if count != 0 {
				max_depth = i
			}
			huff.counts[i] = count
			total_count += count
			space -= count * (1 << uint(kJpegHuffmanMaxBitLength-i))
		}
		if is_ac_table {
			VERIFY_INPUT(total_count, 0, kJpegHuffmanAlphabetSize, "HUFFMAN_CODE")
		} else {
			VERIFY_INPUT(total_count, 0, kJpegDCAlphabetSize, "HUFFMAN_CODE")
		}
		VERIFY_LEN(pos, total_count, length)
		values_seen := make([]bool, 256)
		for i := 0; i < total_count; i++ {
			value := ReadUint8(data, pos)
			if !is_ac_table {
				VERIFY_INPUT(value, 0, kJpegDCAlphabetSize-1, "HUFFMAN_CODE")
			}
			if values_seen[value] {
				fprintf(stderr, "Duplicate Huffman code value %d\n", value)
				jpg.err = JPEG_INVALID_HUFFMAN_CODE
				return false
			}
			values_seen[value] = true
			huff.values[i] = value
		}
		// Add an invalid symbol that will have the all 1 code.
		huff.counts[max_depth]++
		huff.values[total_count] = kJpegHuffmanAlphabetSize
		space -= (1 << uint(kJpegHuffmanMaxBitLength-max_depth))
		if space < 0 {
			fprintf(stderr, "Invalid Huffman code lengths.\n")
			jpg.err = JPEG_INVALID_HUFFMAN_CODE
			return false
		} else if space > 0 && huff_lut[0].value != 0xffff {
			// Re-initialize the values to an invalid symbol so that we can recognize
			// it when reading the bit stream using a Huffman code with space > 0.
			for i := 0; i < kJpegHuffmanLutSize; i++ {
				huff_lut[i].bits = 0
				huff_lut[i].value = 0xffff
			}
		}
		huff.is_last = (*pos == start_pos+marker_len)
		if mode == JPEG_READ_ALL && BuildJpegHuffmanTable(&huff.counts[0], &huff.values[0], huff_lut) != 0 {
			fprintf(stderr, "Failed to build Huffman table.\n")
			jpg.err = JPEG_INVALID_HUFFMAN_CODE
			return false
		}
		jpg.huffman_code = append(jpg.huffman_code, huff)
	}
	VERIFY_MARKER_END(pos, start_pos, marker_len)
	return true
}

// Reads the Define Quantization Table (DQT) marker segment and fills in *jpg
// with the parsed data.
func ProcessDQT(data []byte, length int, pos *int, jpg *JPEGData) bool {
	start_pos := *pos
	VERIFY_LEN(pos, 2, length)
	marker_len := ReadUint16(data, pos)
	if marker_len == 2 {
		fprintf(stderr, "DQT marker: no quantization table found\n")
		jpg.err = JPEG_EMPTY_DQT
		return false
	}
	for *pos < start_pos+marker_len && len(jpg.quant) < kMaxQuantTables {
		VERIFY_LEN(pos, 1, length)
		quant_table_index := ReadUint8(data, pos)
		quant_table_precision := quant_table_index >> 4
		quant_table_index &= 0xf
		VERIFY_INPUT(quant_table_index, 0, 3, "QUANT_TBL_INDEX")
		if quant_table_precision != 0 {
			VERIFY_LEN(pos, 2*kDCTBlockSize, length)
		} else {
			VERIFY_LEN(pos, 1*kDCTBlockSize, length)
		}

		var table JPEGQuantTable
		table.index = quant_table_index
		table.precision = quant_table_precision
		for i := 0; i < kDCTBlockSize; i++ {
			var quant_val int
			if quant_table_precision != 0 {
				quant_val = ReadUint16(data, pos)
			} else {
				quant_val = ReadUint8(data, pos)
			}
			VERIFY_INPUT(quant_val, 1, 65535, "QUANT_VAL")
			table.values[kJPEGNaturalOrder[i]] = quant_val
		}
		table.is_last = (*pos == start_pos+marker_len)
		jpg.quant = append(jpg.quant, table)
	}
	VERIFY_MARKER_END(pos, start_pos, marker_len)
	return true
}

// Reads the DRI marker and saved the restart interval into *jpg.
func ProcessDRI(data []byte, length int, pos *int,
	jpg *JPEGData) bool {
	if jpg.restart_interval > 0 {
		fprintf(stderr, "Duplicate DRI marker.\n")
		jpg.err = JPEG_DUPLICATE_DRI
		return false
	}
	start_pos := *pos
	VERIFY_LEN(pos, 4, length)
	marker_len := ReadUint16(data, pos)
	restart_interval := ReadUint16(data, pos)
	jpg.restart_interval = restart_interval
	VERIFY_MARKER_END(pos, start_pos, marker_len)
	return true
}

// Saves the APP marker segment as a string to *jpg.
func ProcessAPP(data []byte, length int, pos *int,
	jpg *JPEGData) bool {
	VERIFY_LEN(pos, 2, length)
	marker_len := ReadUint16(data, pos)
	VERIFY_INPUT(marker_len, 2, 65535, "MARKER_LEN")
	VERIFY_LEN(pos, marker_len-2, length)
	var app_str string
	// Save the marker type together with the app data.
	//std::string app_str(reinterpret_cast<const char*>(&data[*pos - 3]), marker_len + 1);
	assert(false) // TODO above line
	*pos += marker_len - 2
	jpg.app_data = append(jpg.app_data, app_str)
	return true
}

// Saves the COM marker segment as a string to *jpg.
func ProcessCOM(data []byte, length int, pos *int, jpg *JPEGData) bool {
	VERIFY_LEN(pos, 2, length)
	marker_len := ReadUint16(data, pos)
	VERIFY_INPUT(marker_len, 2, 65535, "MARKER_LEN")
	VERIFY_LEN(pos, marker_len-2, length)
	var com_str string
	//std::string com_str(reinterpret_cast<const char*>(&data[*pos - 2]), marker_len);
	assert(false) // TODO above line
	*pos += marker_len - 2
	jpg.com_data = append(jpg.com_data, com_str)
	return true
}

// Helper structure to read bits from the entropy coded data segment.
type BitReaderState struct {
	data_            []byte
	len_             int
	pos_             int
	val_             uint64
	bits_left_       int
	next_marker_pos_ int
}

func NewBitReaderState(data []byte, length int, pos int) *BitReaderState {
	br := new(BitReaderState)
	br.data_ = data
	br.len_ = length
	br.Reset(pos)
	return br
}

func (br *BitReaderState) Reset(pos int) {
	br.pos_ = pos
	br.val_ = 0
	br.bits_left_ = 0
	br.next_marker_pos_ = br.len_ - 2
	br.FillBitWindow()
}

// Returns the next byte and skips the 0xff/0x00 escape sequences.
func (br *BitReaderState) GetNextByte() byte {
	if br.pos_ >= br.next_marker_pos_ {
		br.pos_++
		return 0
	}
	c := br.data_[br.pos_]
	br.pos_++
	if c == 0xff {
		escape := br.data_[br.pos_]
		if escape == 0 {
			br.pos_++
		} else {
			// 0xff was followed by a non-zero byte, which means that we found the
			// start of the next marker segment.
			br.next_marker_pos_ = br.pos_ - 1
		}
	}
	return c
}

func (br *BitReaderState) FillBitWindow() {
	if br.bits_left_ <= 16 {
		for br.bits_left_ <= 56 {
			br.val_ <<= 8
			br.val_ |= uint64(br.GetNextByte())
			br.bits_left_ += 8
		}
	}
}

func (br *BitReaderState) ReadBits(nbits int) int {
	br.FillBitWindow()
	val := uint64(br.val_>>uint(br.bits_left_-nbits)) & ((1 << uint(nbits)) - 1)
	br.bits_left_ -= nbits
	return int(val)
}

// Sets *pos to the next stream position where parsing should continue.
// Returns false if the stream ended too early.
func (br *BitReaderState) FinishStream(pos *int) bool {
	// Give back some bytes that we did not use.
	unused_bytes_left := br.bits_left_ >> 3
	for unused_bytes_left > 0 {
		unused_bytes_left--
		br.pos_--
		// If we give back a 0 byte, we need to check if it was a 0xff/0x00 escape
		// sequence, and if yes, we need to give back one more byte.
		if br.pos_ < br.next_marker_pos_ && br.data_[br.pos_] == 0 && br.data_[br.pos_-1] == 0xff {
			br.pos_--
		}
	}
	if br.pos_ > br.next_marker_pos_ {
		// Data ran out before the scan was complete.
		fprintf(stderr, "Unexpected end of scan.\n")
		return false
	}
	*pos = br.pos_
	return true
}

// Returns the next Huffman-coded symbol.
func ReadSymbol(table []HuffmanTableEntry, br *BitReaderState) int {
	var nbits int
	br.FillBitWindow()
	val := (br.val_ >> uint(br.bits_left_-8)) & 0xff
	table = table[val:]
	nbits = int(table[0].bits) - 8
	if nbits > 0 {
		br.bits_left_ -= 8
		table = table[table[0].value:]
		val = (br.val_ >> uint(br.bits_left_-nbits)) & ((1 << uint(nbits)) - 1)
		table = table[val:]
	}
	br.bits_left_ -= int(table[0].bits)
	return int(table[0].value)
}

// Returns the DC diff or AC value for extra bits value x and prefix code s.
// See Tables F.1 and F.2 of the spec.
func HuffExtend(x, s int) int {
	if x < (1 << uint(s-1)) {
		return x - (1 << uint(s)) + 1
	} else {
		return x
	}
}

// Decodes one 8x8 block of DCT coefficients from the bit stream.
func DecodeDCTBlock(dc_huff, ac_huff []HuffmanTableEntry,
	Ss, Se, Al int,
	eobrun *int,
	br *BitReaderState,
	jpg *JPEGData,
	last_dc_coeff, coeffs []coeff_t) bool {
	var s, r int
	eobrun_allowed := Ss > 0
	if Ss == 0 {
		s = ReadSymbol(dc_huff, br)
		if s >= kJpegDCAlphabetSize {
			fprintf(stderr, "Invalid Huffman symbol %d for DC coefficient.\n", s)
			jpg.err = JPEG_INVALID_SYMBOL
			return false
		}
		if s > 0 {
			r = br.ReadBits(s)
			s = HuffExtend(r, s)
		}
		s += int(last_dc_coeff[0])
		dc_coeff := coeff_t(SignedLeftshift(s, Al))
		coeffs[0] = dc_coeff
		if dc_coeff != coeffs[0] {
			fprintf(stderr, "Invalid DC coefficient %d\n", dc_coeff)
			jpg.err = JPEG_NON_REPRESENTABLE_DC_COEFF
			return false
		}
		last_dc_coeff[0] = coeff_t(s)
		Ss++
	}
	if Ss > Se {
		return true
	}
	if *eobrun > 0 {
		(*eobrun)--
		return true
	}
	for k := Ss; k <= Se; k++ {
		s = ReadSymbol(ac_huff, br)
		if s >= kJpegHuffmanAlphabetSize {
			fprintf(stderr, "Invalid Huffman symbol %d for AC coefficient %d\n",
				s, k)
			jpg.err = JPEG_INVALID_SYMBOL
			return false
		}
		r = s >> 4
		s &= 15
		if s > 0 {
			k += r
			if k > Se {
				fprintf(stderr, "Out-of-band coefficient %d band was %d-%d\n",
					k, Ss, Se)
				jpg.err = JPEG_OUT_OF_BAND_COEFF
				return false
			}
			if s+Al >= kJpegDCAlphabetSize {
				fprintf(stderr, "Out of range AC coefficient value: s=%d Al=%d k=%d\n",
					s, Al, k)
				jpg.err = JPEG_NON_REPRESENTABLE_AC_COEFF
				return false
			}
			r = br.ReadBits(s)
			s = HuffExtend(r, s)
			coeffs[kJPEGNaturalOrder[k]] = coeff_t(SignedLeftshift(s, Al))
		} else if r == 15 {
			k += 15
		} else {
			*eobrun = 1 << uint(r)
			if r > 0 {
				if !eobrun_allowed {
					fprintf(stderr, "End-of-block run crossing DC coeff.\n")
					jpg.err = JPEG_EOB_RUN_TOO_LONG
					return false
				}
				*eobrun += br.ReadBits(r)
			}
			break
		}
	}
	(*eobrun)--
	return true
}

func RefineDCTBlock(ac_huff []HuffmanTableEntry,
	Ss, Se, Al int,
	eobrun *int,
	br *BitReaderState,
	jpg *JPEGData,
	coeffs []coeff_t) bool {
	eobrun_allowed := Ss > 0
	if Ss == 0 {
		s := br.ReadBits(1)
		dc_coeff := coeffs[0]
		dc_coeff |= coeff_t(s << uint(Al))
		coeffs[0] = dc_coeff
		Ss++
	}
	if Ss > Se {
		return true
	}
	p1 := 1 << uint(Al)
	m1 := -(1 << uint(Al))
	k := Ss
	var r, s int
	in_zero_run := false
	if *eobrun <= 0 {
		for ; k <= Se; k++ {
			s = ReadSymbol(ac_huff, br)
			if s >= kJpegHuffmanAlphabetSize {
				fprintf(stderr, "Invalid Huffman symbol %d for AC coefficient %d\n",
					s, k)
				jpg.err = JPEG_INVALID_SYMBOL
				return false
			}
			r = s >> 4
			s &= 15
			if s != 0 {
				if s != 1 {
					fprintf(stderr, "Invalid Huffman symbol %d for AC coefficient %d\n",
						s, k)
					jpg.err = JPEG_INVALID_SYMBOL
					return false
				}
				s = m1
				if br.ReadBits(1) != 0 {
					s = p1
				}
				in_zero_run = false
			} else {
				if r != 15 {
					*eobrun = 1 << uint(r)
					if r > 0 {
						if !eobrun_allowed {
							fprintf(stderr, "End-of-block run crossing DC coeff.\n")
							jpg.err = JPEG_EOB_RUN_TOO_LONG
							return false
						}
						*eobrun += br.ReadBits(r)
					}
					break
				}
				in_zero_run = true
			}
			for {
				thiscoef := coeffs[kJPEGNaturalOrder[k]]
				if thiscoef != 0 {
					if br.ReadBits(1) != 0 {
						if (thiscoef & coeff_t(p1)) == 0 {
							if thiscoef >= 0 {
								thiscoef += coeff_t(p1)
							} else {
								thiscoef += coeff_t(m1)
							}
						}
					}
					coeffs[kJPEGNaturalOrder[k]] = thiscoef
				} else {
					r--
					if r < 0 {
						break
					}
				}
				k++
				if k > Se {
					break
				}
			}
			if s != 0 {
				if k > Se {
					fprintf(stderr, "Out-of-band coefficient %d band was %d-%d\n",
						k, Ss, Se)
					jpg.err = JPEG_OUT_OF_BAND_COEFF
					return false
				}
				coeffs[kJPEGNaturalOrder[k]] = coeff_t(s)
			}
		}
	}
	if in_zero_run {
		fprintf(stderr, "Extra zero run before end-of-block.\n")
		jpg.err = JPEG_EXTRA_ZERO_RUN
		return false
	}
	if *eobrun > 0 {
		for ; k <= Se; k++ {
			thiscoef := coeffs[kJPEGNaturalOrder[k]]
			if thiscoef != 0 {
				if br.ReadBits(1) != 0 {
					if (thiscoef & coeff_t(p1)) == 0 {
						if thiscoef >= 0 {
							thiscoef += coeff_t(p1)
						} else {
							thiscoef += coeff_t(m1)
						}
					}
				}
				coeffs[kJPEGNaturalOrder[k]] = thiscoef
			}
		}
	}
	(*eobrun)--
	return true
}

func ProcessRestart(data []byte, length int,
	next_restart_marker *int, br *BitReaderState,
	jpg *JPEGData) bool {
	pos := 0
	if !br.FinishStream(&pos) {
		jpg.err = JPEG_INVALID_SCAN
		return false
	}
	expected_marker := 0xd0 + *next_restart_marker
	EXPECT_MARKER(pos, length, data)
	marker := data[pos+1]
	if int(marker) != expected_marker {
		fprintf(stderr, "Did not find expected restart marker %d actual=%d\n",
			expected_marker, marker)
		jpg.err = JPEG_WRONG_RESTART_MARKER
		return false
	}
	br.Reset(pos + 2)
	*next_restart_marker += 1
	*next_restart_marker &= 0x7
	return true
}

func ProcessScan(data []byte, length int,
	dc_huff_lut, ac_huff_lut []HuffmanTableEntry,
	scan_progression *[kMaxComponents][kDCTBlockSize]uint16,
	is_progressive bool,
	pos *int,
	jpg *JPEGData) bool {
	if !ProcessSOS(data, length, pos, jpg) {
		return false
	}
	scan_info := &jpg.scan_info[len(jpg.scan_info)-1]
	is_interleaved := len(scan_info.components) > 1
	var MCUs_per_row, MCU_rows int
	if is_interleaved {
		MCUs_per_row = jpg.MCU_cols
		MCU_rows = jpg.MCU_rows
	} else {
		c := jpg.components[scan_info.components[0].comp_idx]
		MCUs_per_row = DivCeil(jpg.width*c.h_samp_factor, 8*jpg.max_h_samp_factor)
		MCU_rows = DivCeil(jpg.height*c.v_samp_factor, 8*jpg.max_v_samp_factor)
	}
	var last_dc_coeff [kMaxComponents]coeff_t
	br := NewBitReaderState(data, length, *pos)
	restarts_to_go := jpg.restart_interval
	next_restart_marker := 0
	eobrun := -1
	block_scan_index := 0
	var Al, Ah, Ss, Se int
	if is_progressive {
		Al, Ah, Ss, Se = scan_info.Al, scan_info.Ah, scan_info.Ss, scan_info.Se
	}
	var scan_bitmask uint16
	if Ah == 0 {
		scan_bitmask = 0xffff << uint(Al)
	} else {
		scan_bitmask = 1 << uint(Al)
	}
	refinement_bitmask := (uint16(1) << uint(Al)) - 1
	for i := 0; i < len(scan_info.components); i++ {
		comp_idx := scan_info.components[i].comp_idx
		for k := Ss; k <= Se; k++ {
			if (scan_progression[comp_idx][k] & scan_bitmask) != 0 {
				fprintf(stderr, "Overlapping scans: component=%d k=%d prev_mask=%d "+
					"cur_mask=%d\n", comp_idx, k, scan_progression[i][k],
					scan_bitmask)
				jpg.err = JPEG_OVERLAPPING_SCANS
				return false
			}
			if (scan_progression[comp_idx][k] & refinement_bitmask) != 0 {
				fprintf(stderr, "Invalid scan order, a more refined scan was already "+
					"done: component=%d k=%d prev_mask=%d cur_mask=%d\n", comp_idx,
					k, scan_progression[i][k], scan_bitmask)
				jpg.err = JPEG_INVALID_SCAN_ORDER
				return false
			}
			scan_progression[comp_idx][k] |= scan_bitmask
		}
	}
	if Al > 10 {
		fprintf(stderr, "Scan parameter Al=%d is not supported in guetzli.\n", Al)
		jpg.err = JPEG_NON_REPRESENTABLE_AC_COEFF
		return false
	}
	for mcu_y := 0; mcu_y < MCU_rows; mcu_y++ {
		for mcu_x := 0; mcu_x < MCUs_per_row; mcu_x++ {
			// Handle the restart intervals.
			if jpg.restart_interval > 0 {
				if restarts_to_go == 0 {
					if ProcessRestart(data, length, &next_restart_marker, br, jpg) {
						restarts_to_go = jpg.restart_interval
						zeroCoeffs(last_dc_coeff[:])
						if eobrun > 0 {
							fprintf(stderr, "End-of-block run too long.\n")
							jpg.err = JPEG_EOB_RUN_TOO_LONG
							return false
						}
						eobrun = -1 // fresh start
					} else {
						return false
					}
				}
				restarts_to_go--
			}
			// Decode one MCU.
			for i := 0; i < len(scan_info.components); i++ {
				si := &scan_info.components[i]
				c := &jpg.components[si.comp_idx]
				dc_lut := dc_huff_lut[si.dc_tbl_idx*kJpegHuffmanLutSize:]
				ac_lut := ac_huff_lut[si.ac_tbl_idx*kJpegHuffmanLutSize:]
				nblocks_y, nblocks_x := 1, 1
				if is_interleaved {
					nblocks_y, nblocks_x = c.v_samp_factor, c.h_samp_factor
				}
				for iy := 0; iy < nblocks_y; iy++ {
					for ix := 0; ix < nblocks_x; ix++ {
						block_y := mcu_y*nblocks_y + iy
						block_x := mcu_x*nblocks_x + ix
						block_idx := block_y*c.width_in_blocks + block_x
						coeffs := c.coeffs[block_idx*kDCTBlockSize:]
						if Ah == 0 {
							if !DecodeDCTBlock(dc_lut, ac_lut, Ss, Se, Al, &eobrun, br, jpg,
								last_dc_coeff[si.comp_idx:], coeffs) {
								return false
							}
						} else {
							if !RefineDCTBlock(ac_lut, Ss, Se, Al,
								&eobrun, br, jpg, coeffs) {
								return false
							}
						}
						block_scan_index++
					}
				}
			}
		}
	}
	if eobrun > 0 {
		fprintf(stderr, "End-of-block run too long.\n")
		jpg.err = JPEG_EOB_RUN_TOO_LONG
		return false
	}
	if !br.FinishStream(pos) {
		jpg.err = JPEG_INVALID_SCAN
		return false
	}
	if *pos > length {
		fprintf(stderr, "Unexpected end of file during scan. pos=%d len=%d\n",
			int(*pos), int(length))
		jpg.err = JPEG_UNEXPECTED_EOF
		return false
	}
	return true
}

// Changes the quant_idx field of the components to refer to the index of the
// quant table in the jpg.quant array.
func FixupIndexes(jpg *JPEGData) bool {
	for i := 0; i < len(jpg.components); i++ {
		c := &jpg.components[i]
		found_index := false
		for j := 0; j < len(jpg.quant); j++ {
			if jpg.quant[j].index == c.quant_idx {
				c.quant_idx = j
				found_index = true
				break
			}
		}
		if !found_index {
			fprintf(stderr, "Quantization table with index %zd not found\n",
				c.quant_idx)
			jpg.err = JPEG_QUANT_TABLE_NOT_FOUND
			return false
		}
	}
	return true
}

func FindNextMarker(data []byte, length, pos int) int {
	// kIsValidMarker[i] == 1 means (0xc0 + i) is a valid marker.
	kIsValidMarker := []uint8{
		1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
	}
	num_skipped := 0
	for pos+1 < length &&
		(data[pos] != 0xff || data[pos+1] < 0xc0 ||
			kIsValidMarker[data[pos+1]-0xc0] != 0) {
		pos++
		num_skipped++
	}
	return num_skipped
}

func ReadJpeg(data []byte, length int, mode JpegReadMode, jpg *JPEGData) bool {
	pos := 0
	// Check SOI marker.
	EXPECT_MARKER(pos, length, data)
	marker := data[pos+1]
	pos += 2
	if marker != 0xd8 {
		fprintf(stderr, "Did not find expected SOI marker, actual=%d\n", marker)
		jpg.err = JPEG_SOI_NOT_FOUND
		return false
	}
	lut_size := kMaxHuffmanTables * kJpegHuffmanLutSize
	dc_huff_lut := make([]HuffmanTableEntry, lut_size)
	ac_huff_lut := make([]HuffmanTableEntry, lut_size)
	found_sof := false
	var scan_progression [kMaxComponents][kDCTBlockSize]uint16

	is_progressive := false // default
	for {
		// Read next marker.
		num_skipped := FindNextMarker(data, length, pos)
		if num_skipped > 0 {
			// Add a fake marker to indicate arbitrary in-between-markers data.
			jpg.marker_order = append(jpg.marker_order, 0xff)
			jpg.inter_marker_data = append(jpg.inter_marker_data, string(data))
			//jpg.inter_marker_data.push_back(std::string(reinterpret_cast<const char*>(&data[pos]),num_skipped)); ??
			pos += num_skipped
		}
		EXPECT_MARKER(pos, length, data)
		marker = data[pos+1]
		pos += 2
		ok := true
		switch marker {
		case 0xc0, 0xc1, 0xc2:
			is_progressive = (marker == 0xc2)
			ok = ProcessSOF(data, length, mode, &pos, jpg)
			found_sof = true
			break
		case 0xc4:
			ok = ProcessDHT(data, length, mode, dc_huff_lut, ac_huff_lut, &pos, jpg)
			break
		case 0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7:
			// RST markers do not have any data.
			break
		case 0xd9:
			// Found end marker.
			break
		case 0xda:
			if mode == JPEG_READ_ALL {
				ok = ProcessScan(data, length, dc_huff_lut, ac_huff_lut,
					&scan_progression, is_progressive, &pos, jpg)
			}
			break
		case 0xdb:
			ok = ProcessDQT(data, length, &pos, jpg)
			break
		case 0xdd:
			ok = ProcessDRI(data, length, &pos, jpg)
			break
		case 0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef:
			if mode != JPEG_READ_TABLES {
				ok = ProcessAPP(data, length, &pos, jpg)
			}
			break
		case 0xfe:
			if mode != JPEG_READ_TABLES {
				ok = ProcessCOM(data, length, &pos, jpg)
			}
			break
		default:
			fprintf(stderr, "Unsupported marker: %d pos=%d len=%d\n",
				marker, int(pos), int(length))
			jpg.err = JPEG_UNSUPPORTED_MARKER
			ok = false
			break
		}
		if !ok {
			return false
		}
		jpg.marker_order = append(jpg.marker_order, marker)
		if mode == JPEG_READ_HEADER && found_sof {
			break
		}
		if marker == 0xd9 {
			break
		}
	}

	if !found_sof {
		fprintf(stderr, "Missing SOF marker.\n")
		jpg.err = JPEG_SOF_NOT_FOUND
		return false
	}

	// Supplemental checks.
	if mode == JPEG_READ_ALL {
		if pos < length {
			jpg.tail_data = string(data[pos : length-pos])
		}
		if !FixupIndexes(jpg) {
			return false
		}
		if len(jpg.huffman_code) == 0 {
			// Section B.2.4.2: "If a table has never been defined for a particular
			// destination, then when this destination is specified in a scan header,
			// the results are unpredictable."
			fprintf(stderr, "Need at least one Huffman code table.\n")
			jpg.err = JPEG_HUFFMAN_TABLE_ERROR
			return false
		}
		if len(jpg.huffman_code) >= kMaxDHTMarkers {
			fprintf(stderr, "Too many Huffman tables.\n")
			jpg.err = JPEG_HUFFMAN_TABLE_ERROR
			return false
		}
	}
	return true
}

// ??
// bool ReadJpeg(const std::string& data, mode JpegReadMode, jpg *JPEGData{
//   return ReadJpeg(reinterpret_cast<const uint8_t*>(data.data()),
//                   static_cast<const int>(len(data)),
//                   mode, jpg);
// }
