package guetzli_patapon

const kJpegHuffmanRootTableBits = 8

// Maximum huffman lookup table size.
// According to zlib/examples/enough.c, 758 entries are always enough for
// an alphabet of 257 symbols (256 + 1 special symbol for the all 1s code) and
// max bit length 16 if the root table has 8 bits.
const kJpegHuffmanLutSize = 758

type HuffmanTableEntry struct {
	bits  uint8  // number of bits used for this symbol
	value uint16 // symbol value or table offset
}

func NewHuffmanTableEntry() HuffmanTableEntry {
	// Initialize the value to an invalid symbol so that we can recognize it
	// when reading the bit stream using a Huffman code with space > 0.
	return HuffmanTableEntry{
		bits:  0,
		value: 0xffff,
	}
}

// Builds jpeg-style Huffman lookup table from the given symbols.
// The symbols are in order of increasing bit lengths. The number of symbols
// with bit length n is given in counts[n] for each n >= 1.
// Returns the size of the lookup table.
var BuildJpegHuffmanTable func(counts *int, symbols *int, lut []HuffmanTableEntry) int // TODO func body!!!

// var BuildJpegHuffmanTable func(counts *int, symbols *int, lut *HuffmanTableEntry) int
