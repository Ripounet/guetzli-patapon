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

// Returns the table width of the next 2nd level table, count is the histogram
// of bit lengths for the remaining symbols, len is the code length of the next
// processed symbol.
func NextTableBitSize(count []int, length int) int {
	left := 1 << uint(length-kJpegHuffmanRootTableBits)
	for length < kJpegHuffmanMaxBitLength {
		left -= count[length]
		if left <= 0 {
			break
		}
		length++
		left <<= 1
	}
	return length - kJpegHuffmanRootTableBits
}

// Builds jpeg-style Huffman lookup table from the given symbols.
// The symbols are in order of increasing bit lengths. The number of symbols
// with bit length n is given in counts[n] for each n >= 1.
// Returns the size of the lookup table.
func BuildJpegHuffmanTable(count_in, symbols []int, lut []HuffmanTableEntry) int {
	var (
		code       HuffmanTableEntry   // current table entry
		table      []HuffmanTableEntry // next available space in table
		length     int                 // current code length
		idx        int                 // symbol index
		key        int                 // prefix code
		reps       int                 // number of replicate key values in current table
		low        int                 // low bits for current root entry
		table_bits int                 // key length of current table
		table_size int                 // size of current table
		total_size int                 // sum of root table size and 2nd level table sizes
	)

	// Make a local copy of the input bit length histogram.
	var count [kJpegHuffmanMaxBitLength + 1]int
	total_count := 0
	for length = 1; length <= kJpegHuffmanMaxBitLength; length++ {
		count[length] = count_in[length]
		total_count += count[length]
	}

	table = lut
	// table_delta used in go version, to work around pointer arithmetic
	table_delta := 0
	table_bits = kJpegHuffmanRootTableBits
	table_size = 1 << uint(table_bits)
	total_size = table_size

	// Special case code with only one value.
	if total_count == 1 {
		code.bits = 0
		code.value = uint16(symbols[0])
		for key = 0; key < total_size; key++ {
			table[key] = code
		}
		return total_size
	}

	// Fill in root table.
	key = 0
	idx = 0
	for length = 1; length <= kJpegHuffmanRootTableBits; length++ {
		for ; count[length] > 0; count[length]-- {
			code.bits = uint8(length)
			code.value = uint16(symbols[idx])
			idx++
			reps = 1 << uint(kJpegHuffmanRootTableBits-length)
			for ; reps > 0; reps-- {
				table[key] = code
				key++
			}
		}
	}

	// Fill in 2nd level tables and add pointers to root table.
	table = table[table_size:]
	table_delta += table_size
	table_size = 0
	low = 0
	for length = kJpegHuffmanRootTableBits + 1; length <= kJpegHuffmanMaxBitLength; length++ {
		for ; count[length] > 0; count[length]-- {
			// Start a new sub-table if the previous one is full.
			if low >= table_size {
				table = table[table_size:]
				table_delta += table_size
				table_bits = NextTableBitSize(count[:], length)
				table_size = 1 << uint(table_bits)
				total_size += table_size
				low = 0
				lut[key].bits = uint8(table_bits + kJpegHuffmanRootTableBits)
				lut[key].value = uint16(table_delta - key)
				key++
			}
			code.bits = uint8(length - kJpegHuffmanRootTableBits)
			code.value = uint16(symbols[idx])
			idx++
			reps = 1 << uint(table_bits-int(code.bits))
			for ; reps > 0; reps-- {
				table[low] = code
				low++
			}
		}
	}

	return total_size
}
