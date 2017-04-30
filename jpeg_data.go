package guetzli_patapon

// Data structures that represent the contents of a jpeg file.

const (
	kDCTBlockSize            = 64
	kMaxComponents           = 4
	kMaxQuantTables          = 4
	kMaxHuffmanTables        = 4
	kJpegHuffmanMaxBitLength = 16
	kJpegHuffmanAlphabetSize = 256
	kJpegDCAlphabetSize      = 12
	kMaxDHTMarkers           = 512
)

var kDefaultQuantMatrix = [2][64]uint8{
	{16, 11, 10, 16, 24, 40, 51, 61,
		12, 12, 14, 19, 26, 58, 60, 55,
		14, 13, 16, 24, 40, 57, 69, 56,
		14, 17, 22, 29, 51, 87, 80, 62,
		18, 22, 37, 56, 68, 109, 103, 77,
		24, 35, 55, 64, 81, 104, 113, 92,
		49, 64, 78, 87, 103, 121, 120, 101,
		72, 92, 95, 98, 112, 100, 103, 99},
	{17, 18, 24, 47, 99, 99, 99, 99,
		18, 21, 26, 66, 99, 99, 99, 99,
		24, 26, 56, 99, 99, 99, 99, 99,
		47, 66, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99,
		99, 99, 99, 99, 99, 99, 99, 99},
}

var kJPEGNaturalOrder = [80]int{
	0, 1, 8, 16, 9, 2, 3, 10,
	17, 24, 32, 25, 18, 11, 4, 5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13, 6, 7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63,
	// extra entries for safety in decoder
	63, 63, 63, 63, 63, 63, 63, 63,
	63, 63, 63, 63, 63, 63, 63, 63,
}

var kJPEGZigZagOrder = [64]int{
	0, 1, 5, 6, 14, 15, 27, 28,
	2, 4, 7, 13, 16, 26, 29, 42,
	3, 8, 12, 17, 25, 30, 41, 43,
	9, 11, 18, 24, 31, 40, 44, 53,
	10, 19, 23, 32, 39, 45, 52, 54,
	20, 22, 33, 38, 46, 51, 55, 60,
	21, 34, 37, 47, 50, 56, 59, 61,
	35, 36, 48, 49, 57, 58, 62, 63,
}

// Quantization values for an 8x8 pixel block.
type JPEGQuantTable struct {
	values    []int
	precision int
	// The index of this quantization table as it was parsed from the input JPEG.
	// Each DQT marker segment contains an 'index' field, and we save this index
	// here. Valid values are 0 to 3.
	index int
	// Set to true if this table is the last one within its marker segment.
	is_last bool
}

func NewJPEGQuantTable() JPEGQuantTable {
	return JPEGQuantTable{
		values:    make([]int, kDCTBlockSize),
		precision: 0,
		index:     0,
		is_last:   true,
	}
}

// Huffman code and decoding lookup table used for DC and AC coefficients.
type JPEGHuffmanCode struct {
	// Bit length histogram.
	counts []int
	// Symbol values sorted by increasing bit lengths.
	values []int
	// The index of the Huffman code in the current set of Huffman codes. For AC
	// component Huffman codes, 0x10 is added to the index.
	slot_id int
	// Set to true if this Huffman code is the last one within its marker segment.
	is_last bool
}

func NewJPEGHuffmanCode() JPEGHuffmanCode {
	return JPEGHuffmanCode{
		counts:  make([]int, kJpegHuffmanMaxBitLength+1),
		values:  make([]int, kJpegHuffmanAlphabetSize+1),
		slot_id: 0,
		is_last: true,
	}
}

// Huffman table indexes used for one component of one scan.
type JPEGComponentScanInfo struct {
	comp_idx   int
	dc_tbl_idx int
	ac_tbl_idx int
}

// Contains information that is used in one scan.
type JPEGScanInfo struct {
	// Parameters used for progressive scans (named the same way as in the spec):
	//   Ss : Start of spectral band in zig-zag sequence.
	//   Se : End of spectral band in zig-zag sequence.
	//   Ah : Successive approximation bit position, high.
	//   Al : Successive approximation bit position, low.
	Ss         int
	Se         int
	Ah         int
	Al         int
	components []JPEGComponentScanInfo
}

type coeff_t int16

func zeroCoeffs(c []coeff_t) {
	for i := range c {
		c[i] = 0
	}
}

// Represents one component of a jpeg file.
type JPEGComponent struct {
	// One-byte id of the component.
	id int
	// Horizontal and vertical sampling factors.
	// In interleaved mode, each minimal coded unit (MCU) has
	// h_samp_factor x v_samp_factor DCT blocks from this component.
	h_samp_factor int
	v_samp_factor int
	// The index of the quantization table used for this component.
	quant_idx int
	// The dimensions of the component measured in 8x8 blocks.
	width_in_blocks  int
	height_in_blocks int
	num_blocks       int
	// The DCT coefficients of this component, laid out block-by-block, divided
	// through the quantization matrix values.
	coeffs []coeff_t
}

func NewJPEGComponent() JPEGComponent {
	return JPEGComponent{
		id:               0,
		h_samp_factor:    1,
		v_samp_factor:    1,
		quant_idx:        0,
		width_in_blocks:  0,
		height_in_blocks: 0,
	}
}

// Represents a parsed jpeg file.
type JPEGData struct {
	width             int
	height            int
	version           int
	max_h_samp_factor int
	max_v_samp_factor int
	MCU_rows          int
	MCU_cols          int
	restart_interval  int
	app_data          []string
	com_data          []string
	quant             []JPEGQuantTable
	huffman_code      []JPEGHuffmanCode
	components        []JPEGComponent
	scan_info         []JPEGScanInfo
	marker_order      []byte
	inter_marker_data []string
	tail_data         string
	original_jpg      []byte
	original_jpg_size int
	err               JPEGReadError
}

func NewJPEGData() JPEGData {
	return JPEGData{
		width:             0,
		height:            0,
		version:           0,
		max_h_samp_factor: 1,
		max_v_samp_factor: 1,
		MCU_rows:          0,
		MCU_cols:          0,
		restart_interval:  0,
		original_jpg:      nil,
		original_jpg_size: 0,
		err:               JPEG_OK,
	}
}

// In C++,
//   jpg := *jpg_in
// would copy all the lists contents.
// But in go we must be explicit.
func (data *JPEGData) clone() *JPEGData {
	data2 := new(JPEGData)
	*data2 = *data
	data2.app_data = make([]string, len(data.app_data))
	data2.app_data = make([]string, len(data.app_data))
	data2.com_data = make([]string, len(data.com_data))
	data2.quant = make([]JPEGQuantTable, len(data.quant))
	data2.huffman_code = make([]JPEGHuffmanCode, len(data.huffman_code))
	data2.components = make([]JPEGComponent, len(data.components))
	data2.scan_info = make([]JPEGScanInfo, len(data.scan_info))
	data2.marker_order = make([]byte, len(data.marker_order))
	data2.inter_marker_data = make([]string, len(data.inter_marker_data))
	data2.original_jpg = make([]byte, len(data.original_jpg))
	return data2
}

func (data *JPEGData) Is420() bool {
	return len(data.components) == 3 &&
		data.max_h_samp_factor == 2 &&
		data.max_v_samp_factor == 2 &&
		data.components[0].h_samp_factor == 2 &&
		data.components[0].v_samp_factor == 2 &&
		data.components[1].h_samp_factor == 1 &&
		data.components[1].v_samp_factor == 1 &&
		data.components[2].h_samp_factor == 1 &&
		data.components[2].v_samp_factor == 1
}

func (data *JPEGData) Is444() bool {
	return len(data.components) == 3 &&
		data.max_h_samp_factor == 1 &&
		data.max_v_samp_factor == 1 &&
		data.components[0].h_samp_factor == 1 &&
		data.components[0].v_samp_factor == 1 &&
		data.components[1].h_samp_factor == 1 &&
		data.components[1].v_samp_factor == 1 &&
		data.components[2].h_samp_factor == 1 &&
		data.components[2].v_samp_factor == 1
}

func InitJPEGDataForYUV444(w, h int, jpg *JPEGData) {
	jpg.width = w
	jpg.height = h
	jpg.max_h_samp_factor = 1
	jpg.max_v_samp_factor = 1
	jpg.MCU_rows = (h + 7) >> 3
	jpg.MCU_cols = (w + 7) >> 3
	jpg.quant = make([]JPEGQuantTable, 3)
	jpg.components = make([]JPEGComponent, 3)
	for i := 0; i < 3; i++ {
		c := &jpg.components[i]
		c.id = i
		c.h_samp_factor = 1
		c.v_samp_factor = 1
		c.quant_idx = i
		c.width_in_blocks = jpg.MCU_cols
		c.height_in_blocks = jpg.MCU_rows
		c.num_blocks = c.width_in_blocks * c.height_in_blocks
		c.coeffs = make([]coeff_t, c.num_blocks*kDCTBlockSize)
	}
}

func SaveQuantTables(q [][kDCTBlockSize]int, jpg *JPEGData) {
	jpg.quant = nil
	num_tables := 0
	for i := 0; i < len(jpg.components); i++ {
		comp := &jpg.components[i]
		// Check if we have this quant table already.
		found := false
		for j := 0; j < num_tables; j++ {
			if memcmpInt(q[i][:], jpg.quant[j].values, kDCTBlockSize) {
				comp.quant_idx = j
				found = true
				break
			}
		}
		if !found {
			table := NewJPEGQuantTable()
			copy(table.values[:], q[i][:])
			table.precision = 0
			for k := 0; k < kDCTBlockSize; k++ {
				assert(table.values[k] >= 0)
				assert(table.values[k] < (1 << 16))
				if table.values[k] > 0xff {
					table.precision = 1
				}
			}
			table.index = num_tables
			comp.quant_idx = num_tables
			jpg.quant = append(jpg.quant, table)
			num_tables++
		}
	}
}
