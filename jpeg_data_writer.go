package guetzli_patapon

// Function pointer type used to write len bytes into buf. Returns the
// number of bytes written or -1 on error.
type JPEGOutputHook func(data interface{}, byf []byte) int

// Output callback function with associated data.
type JPEGOutput struct {
	cb   JPEGOutputHook
	data interface{}
}

func (out *JPEGOutput) Write(buf []byte) bool {
	return len(buf) == 0 || cb(data, buf) == len(buf)
}

type HuffmanCodeTable struct {
	depth [256]byte
	code  [256]int
}

const kSize = kJpegHuffmanAlphabetSize + 1

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
	h.counts[symbol] += 2 * weight
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
		if counts[i] > 0 {
			n++
		}
	}
	return n
}

////////////////////////////////


const kJpegPrecision = 8;

// Writes len bytes from buf, using the out callback.
func JPEGWrite(JPEGOutput out, buf []byte) bool {
  const kBlockSize uint32 = 1 << 30;
  pos := 0;
  for (len(buf) - pos > kBlockSize) {
    if (!out.Write(buf + pos, kBlockSize)) {
      return false;
    }
    pos += kBlockSize;
  }
  return out.Write(buf + pos, len - pos);
}

func EncodeMetadata(jpg *JPEGData, strip_metadata bool, out JPEGOutput) bool {
  if (strip_metadata) {
    kApp0Data := []byte{
      0xff, 0xe0, 0x00, 0x10,        // APP0
      0x4a, 0x46, 0x49, 0x46, 0x00,  // 'JFIF'
      0x01, 0x01,                    // v1.01
      0x00, 0x00, 0x01, 0x00, 0x01,  // aspect ratio = 1:1
      0x00, 0x00,                     // thumbnail width/height
    };
    return JPEGWrite(out, kApp0Data);
  }
  ok := true;
  for i := 0; i < jpg.app_data.size(); i++ {
    data := []byte{ 0xff };
    ok = ok && JPEGWrite(out, data);
    ok = ok && JPEGWrite(out, jpg.app_data[i]);
  }
  for i := 0; i < jpg.com_data.size(); i++ {
    data := []byte{ 0xff, 0xfe };
    ok = ok && JPEGWrite(out, data);
    ok = ok && JPEGWrite(out, jpg.com_data[i]);
  }
  return ok;
}

func EncodeDQT(quant []JPEGQuantTable, out JPEGOutput) bool {
  marker_len := 2;
  for i := 0; i < len(quant); i++ {
    marker_len += 1 + kDCTBlockSize;
    if quant[i].precision {
      marker_len += kDCTBlockSize;   
    }
  }
  data := make([]byte, marker_len + 2);
  pos := 0;
  data[pos] = 0xff; pos++
  data[pos] = 0xdb; pos++
  data[pos] = marker_len >> 8; pos++
  data[pos] = marker_len & 0xff; pos++
  for i := 0; i < len(quant); i++ {
    table = &quant[i];
    data[pos] = (table.precision << 4) + table.index; pos++
    for k := 0; k < kDCTBlockSize; k++ {
      val := table.values[kJPEGNaturalOrder[k]];
      if (table.precision) {
        data[pos] = val >> 8; pos++
      }
      data[pos] = val & 0xff; pos++
    }
  }
  return JPEGWrite(out, data, pos);
}

func EncodeSOF(jpg *JPEGData, out JPEGOutput) bool {
  ncomps := jpg.components.size();
  marker_len := 8 + 3 * ncomps;
  data := make([]byte, marker_len + 2);
  pos := 0;
  data[pos] = 0xff; pos++
  data[pos] = 0xc1; pos++
  data[pos] = static_cast<uint8_t>(marker_len >> 8); pos++
  data[pos] = marker_len & 0xff; pos++
  data[pos] = kJpegPrecision; pos++
  data[pos] = jpg.height >> 8; pos++
  data[pos] = jpg.height & 0xff; pos++
  data[pos] = jpg.width >> 8; pos++
  data[pos] = jpg.width & 0xff; pos++
  data[pos] = byte(ncomps); pos++
  for i := 0; i < ncomps; i++ {
    data[pos] = jpg.components[i].id; pos++;
    data[pos] = ((jpg.components[i].h_samp_factor << 4) |
                      (jpg.components[i].v_samp_factor)); pos++;
    quant_idx := jpg.components[i].quant_idx;
    if (quant_idx >= len(jpg.quant)) {
      return false;
    }
    data[pos] = jpg.quant[quant_idx].index; pos++
  }
  return JPEGWrite(out, data, pos);
}

// Builds a JPEG-style huffman code from the given bit depths.
func BuildHuffmanCode(uint8_t* depth, int* counts, int* values) {
  for i := 0; i < JpegHistogram_kSize; i++ {
    if (depth[i] > 0) {
      counts[depth[i]]++
    }
  }
  var offset [kJpegHuffmanMaxBitLength + 1]int
  for i := 1; i <= kJpegHuffmanMaxBitLength; i++ {
    offset[i] = offset[i - 1] + counts[i - 1];
  }
  for i := 0; i < JpegHistogram_kSize; i++ {
    if (depth[i] > 0) {
      values[offset[depth[i]]] = i;
      offset[depth[i]]++
    }
  }
}

func BuildHuffmanCodeTable(counts, values []int, table []HuffmanCodeTable) {
  var huffcode [256]int
  var huffsize [256]int
  p := 0;
  for l := 1; l <= kJpegHuffmanMaxBitLength; l++ {
    i := counts[l];
    for ;i>0; i-- {
      huffsize[p] = l;
      p++
    }
  }

  if (p == 0) {
    return;
  }

  huffsize[p - 1] = 0;
  lastp := p - 1;

  code := 0;
  si := huffsize[0];
  p = 0;
  for (huffsize[p]) {
    for ((huffsize[p]) == si) {
      huffcode[p] = code;
      p++
      code++;
    }
    code <<= 1;
    si++;
  }
  for p = 0; p < lastp; p++ {
    i := values[p];
    table.depth[i] = huffsize[p];
    table.code[i] = huffcode[p];
  }
}

// Updates ac_histogram with the counts of the AC symbols that will be added by
// a sequential jpeg encoder for this block. Every symbol is counted twice so
// that we can add a fake symbol at the end with count 1 to be the last (least
// frequent) symbol with the all 1 code.
func UpdateACHistogramForDCTBlock(coeffs []coeff_t, ac_histogram *JpegHistogram) {
  r := 0;
  for k := 1; k < 64; k++ {
    coeff := coeffs[kJPEGNaturalOrder[k]];
    if (coeff == 0) {
      r++;
      continue;
    }
    for (r > 15) {
      ac_histogram.Add(0xf0);
      r -= 16;
    }
    nbits := Log2FloorNonZero(std_abs(coeff)) + 1;
    symbol := (r << 4) + nbits;
    ac_histogram.Add(symbol);
    r = 0;
  }
  if (r > 0) {
    ac_histogram.Add(0);
  }
}

size_t HistogramHeaderCost(const JpegHistogram& histo) {
  header_bits := 17 * 8;
  for i := 0; i + 1 < JpegHistogram::kSize; i++ {
    if (histo.counts[i] > 0) {
      header_bits += 8;
    }
  }
  return header_bits;
}

size_t HistogramEntropyCost(const JpegHistogram& histo,
                            const uint8_t depths[256]) {
  bits := 0;
  for i := 0; i + 1 < JpegHistogram::kSize; i++ {
    // JpegHistogram::Add() counts every symbol twice, so we have to divide by
    // two here.
    bits += (histo.counts[i] / 2) * (depths[i] + (i & 0xf));
  }
  // Estimate escape byte rate to be 0.75/256.
  bits += (bits * 3 + 512) >> 10;
  return bits;
}

func BuildDCHistograms(jpg *JPEGData) (histo []JpegHistogram) {
  histo = make([]JpegHistogram, len(jpg.components)
  for i := 0; i < jpg.components.size(); i++ {
    c := &jpg.components[i];
    JpegHistogram* dc_histogram = &histo[i];
    coeff_t last_dc_coeff = 0;
    for mcu_y := 0; mcu_y < jpg.MCU_rows; mcu_y++ {
      for mcu_x := 0; mcu_x < jpg.MCU_cols; mcu_x++ {
        for iy := 0; iy < c.v_samp_factor; iy++ {
          for ix := 0; ix < c.h_samp_factor; ix++ {
            block_y := mcu_y * c.v_samp_factor + iy;
            block_x := mcu_x * c.h_samp_factor + ix;
            block_idx := block_y * c.width_in_blocks + block_x;
            coeff_t dc_coeff = c.coeffs[block_idx << 6];
            diff := std::abs(dc_coeff - last_dc_coeff);
            nbits := Log2Floor(diff) + 1;
            dc_histogram.Add(nbits);
            last_dc_coeff = dc_coeff;
          }
        }
      }
    }
  }
}

func BuildACHistograms(jpg *JPEGData, JpegHistogram* histo) {
  for i := 0; i < jpg.components.size(); i++ {
    const JPEGComponent& c = jpg.components[i];
    JpegHistogram* ac_histogram = &histo[i];
    for j := 0; j < c.coeffs.size(); j += kDCTBlockSize) {
      UpdateACHistogramForDCTBlock(&c.coeffs[j], ac_histogram);
    }
  }
}

// Size of everything except the Huffman codes and the entropy coded data.
size_t JpegHeaderSize(jpg *JPEGData, bool strip_metadata) {
  num_bytes := 0;
  num_bytes += 2;  // SOI
  if (strip_metadata) {
    num_bytes += 18;  // APP0
  } else {
    for i := 0; i < jpg.app_data.size(); i++ {
      num_bytes += 1 + jpg.app_data[i].size();
    }
    for i := 0; i < jpg.com_data.size(); i++ {
      num_bytes += 2 + jpg.com_data[i].size();
    }
  }
  // DQT
  num_bytes += 4;
  for i := 0; i < len(jpg.quant); i++ {
    num_bytes += 1 + (jpg.quant[i].precision ? 2 : 1) * kDCTBlockSize;
  }
  num_bytes += 10 + 3 * jpg.components.size();  // SOF
  num_bytes += 4;  // DHT (w/o actual Huffman code data)
  num_bytes += 8 + 2 * jpg.components.size();  // SOS
  num_bytes += 2;  // EOI
  num_bytes += jpg.tail_data.size();
  return num_bytes;
}

size_t ClusterHistograms(JpegHistogram* histo, size_t* num,
                         int* histo_indexes, uint8_t* depth) {
  memset(depth, 0, *num * JpegHistogram::kSize);
  size_t costs[kMaxComponents];
  for i := 0; i < *num; i++ {
    histo_indexes[i] = i;
    std::vector<HuffmanTree> tree(2 * JpegHistogram::kSize + 1);
    CreateHuffmanTree(histo[i].counts, JpegHistogram::kSize,
                      kJpegHuffmanMaxBitLength, &tree[0],
                      &depth[i * JpegHistogram::kSize]);
    costs[i] = (HistogramHeaderCost(histo[i]) +
                HistogramEntropyCost(histo[i],
                                     &depth[i * JpegHistogram::kSize]));
  }
  const orig_num := *num;
  for (*num > 1) {
    last := *num - 1;
    second_last := *num - 2;
    JpegHistogram combined(histo[last]);
    combined.AddHistogram(histo[second_last]);
    std::vector<HuffmanTree> tree(2 * JpegHistogram::kSize + 1);
    uint8_t depth_combined[JpegHistogram::kSize] = { 0 };
    CreateHuffmanTree(combined.counts, JpegHistogram::kSize,
                      kJpegHuffmanMaxBitLength, &tree[0], depth_combined);
    cost_combined := (HistogramHeaderCost(combined) +
                            HistogramEntropyCost(combined, depth_combined));
    if (cost_combined < costs[last] + costs[second_last]) {
      histo[second_last] = combined;
      histo[last] = JpegHistogram();
      costs[second_last] = cost_combined;
      memcpy(&depth[second_last * JpegHistogram::kSize], depth_combined,
             sizeof(depth_combined));
      for i := 0; i < orig_num; i++ {
        if (histo_indexes[i] == last) {
          histo_indexes[i] = second_last;
        }
      }
      --(*num);
    } else {
      break;
    }
  }
  total_cost := 0;
  for i := 0; i < *num; i++ {
    total_cost += costs[i];
  }
  return (total_cost + 7) / 8;
}

func EstimateJpegDataSize(num_components int,
                            const std::vector<JpegHistogram>& histograms) int {
  assert(histograms.size() == 2 * num_components);
  std::vector<JpegHistogram> clustered = histograms;
  num_dc := num_components;
  num_ac := num_components;
  int indexes[kMaxComponents];
  uint8_t depth[kMaxComponents * JpegHistogram::kSize];
  return (ClusterHistograms(&clustered[0], &num_dc, indexes, depth) +
          ClusterHistograms(&clustered[num_components], &num_ac, indexes,
                            depth));
}

// Writes DHT and SOS marker segments to out and fills in DC/AC Huffman tables
// for each component of the image.
func BuildAndEncodeHuffmanCodes(jpg *JPEGData, out JPEGOutput) (ok bool, dc_huff_tables, ac_huff_tables []HuffmanCodeTable) {
  const ncomps := len(jpg.components);
  dc_huff_tables = make([]HuffmanCodeTable, ncomps);
  ac_huff_tables = make([]HuffmanCodeTable, ncomps);

  // Build separate DC histograms for each component.
  histograms := make([]JpegHistogram, ncomps);
  BuildDCHistograms(jpg, &histograms[0]);

  // Cluster DC histograms.
  num_dc_histo := ncomps;
  int dc_histo_indexes[kMaxComponents];
  std::vector<uint8_t> depths(ncomps * JpegHistogram::kSize);
  ClusterHistograms(&histograms[0], &num_dc_histo, dc_histo_indexes,
                    &depths[0]);

  // Build separate AC histograms for each component.
  histograms.resize(num_dc_histo + ncomps);
  depths.resize((num_dc_histo + ncomps) * JpegHistogram::kSize);
  BuildACHistograms(jpg, &histograms[num_dc_histo]);

  // Cluster AC histograms.
  num_ac_histo := ncomps;
  int ac_histo_indexes[kMaxComponents];
  ClusterHistograms(&histograms[num_dc_histo], &num_ac_histo, ac_histo_indexes,
                    &depths[num_dc_histo * JpegHistogram::kSize]);

  // Compute DHT and SOS marker data sizes and start emitting DHT marker.
  num_histo := num_dc_histo + num_ac_histo;
  histograms.resize(num_histo);
  total_count := 0;
  for i := 0; i < histograms.size(); i++ {
    total_count += histograms[i].NumSymbols();
  }
  const dht_marker_len :=
      2 + num_histo * (kJpegHuffmanMaxBitLength + 1) + total_count;
  const sos_marker_len := 6 + 2 * ncomps;
  std::vector<uint8_t> data(dht_marker_len + sos_marker_len + 4);
  pos := 0;
  data[pos++] = 0xff;
  data[pos++] = 0xc4;
  data[pos++] = static_cast<uint8_t>(dht_marker_len >> 8);
  data[pos++] = dht_marker_len & 0xff;

  // Compute Huffman codes for each histograms.
  for i := 0; i < num_histo; i++ {
    const bool is_dc = static_cast<size_t>(i) < num_dc_histo;
    const idx := is_dc ? i : i - num_dc_histo;
    int counts[kJpegHuffmanMaxBitLength + 1] = { 0 };
    int values[JpegHistogram::kSize] = { 0 };
    BuildHuffmanCode(&depths[i * JpegHistogram::kSize], counts, values);
    HuffmanCodeTable table;
    for j := 0; j < 256; j++ table.depth[j] = 255;
    BuildHuffmanCodeTable(counts, values, &table);
    for c := 0; c < ncomps; c++ {
      if (is_dc) {
        if (dc_histo_indexes[c] == idx) (*dc_huff_tables)[c] = table;
      } else {
        if (ac_histo_indexes[c] == idx) (*ac_huff_tables)[c] = table;
      }
    }
    max_length := kJpegHuffmanMaxBitLength;
    for (max_length > 0 && counts[max_length] == 0) --max_length;
    --counts[max_length];
    total_count := 0;
    for j := 0; j <= max_length; j++ total_count += counts[j];
    data[pos++] = is_dc ? i : static_cast<uint8_t>(i - num_dc_histo + 0x10);
    for j := 1; j <= kJpegHuffmanMaxBitLength; j++ {
      data[pos++] = counts[j];
    }
    for j := 0; j < total_count; j++ {
      data[pos++] = values[j];
    }
  }

  // Emit SOS marker data.
  data[pos++] = 0xff;
  data[pos++] = 0xda;
  data[pos++] = static_cast<uint8_t>(sos_marker_len >> 8);
  data[pos++] = sos_marker_len & 0xff;
  data[pos++] = ncomps;
  for i := 0; i < ncomps; i++ {
    data[pos++] = jpg.components[i].id;
    data[pos++] = (dc_histo_indexes[i] << 4) | ac_histo_indexes[i];
  }
  data[pos++] = 0;
  data[pos++] = 63;
  data[pos++] = 0;
  assert(pos == data.size());
  return JPEGWrite(out, &data[0], data.size());
}

func EncodeDCTBlockSequential(const coeff_t* coeffs,
                              const HuffmanCodeTable& dc_huff,
                              const HuffmanCodeTable& ac_huff,
                              coeff_t* last_dc_coeff,
                              BitWriter* bw) {
  coeff_t temp2;
  coeff_t temp;
  temp2 = coeffs[0];
  temp = temp2 - *last_dc_coeff;
  *last_dc_coeff = temp2;
  temp2 = temp;
  if (temp < 0) {
    temp = -temp;
    temp2--;
  }
  nbits := Log2Floor(temp) + 1;
  bw.WriteBits(dc_huff.depth[nbits], dc_huff.code[nbits]);
  if (nbits > 0) {
    bw.WriteBits(nbits, temp2 & ((1 << nbits) - 1));
  }
  r := 0;
  for k := 1; k < 64; k++ {
    if ((temp = coeffs[kJPEGNaturalOrder[k]]) == 0) {
      r++;
      continue;
    }
    if (temp < 0) {
      temp = -temp;
      temp2 = ~temp;
    } else {
      temp2 = temp;
    }
    for (r > 15) {
      bw.WriteBits(ac_huff.depth[0xf0], ac_huff.code[0xf0]);
      r -= 16;
    }
    nbits := Log2FloorNonZero(temp) + 1;
    symbol := (r << 4) + nbits;
    bw.WriteBits(ac_huff.depth[symbol], ac_huff.code[symbol]);
    bw.WriteBits(nbits, temp2 & ((1 << nbits) - 1));
    r = 0;
  }
  if (r > 0) {
    bw.WriteBits(ac_huff.depth[0], ac_huff.code[0]);
  }
}

func EncodeScan(jpg *JPEGData,
                const std::vector<HuffmanCodeTable>& dc_huff_table,
                const std::vector<HuffmanCodeTable>& ac_huff_table,
                out JPEGOutput) bool {
  coeff_t last_dc_coeff[kMaxComponents] = { 0 };
  BitWriter bw(1 << 17);
  for mcu_y := 0; mcu_y < jpg.MCU_rows; mcu_y++ {
    for mcu_x := 0; mcu_x < jpg.MCU_cols; mcu_x++ {
      // Encode one MCU
      for i := 0; i < jpg.components.size(); i++ {
        const JPEGComponent& c = jpg.components[i];
        nblocks_y := c.v_samp_factor;
        nblocks_x := c.h_samp_factor;
        for iy := 0; iy < nblocks_y; iy++ {
          for ix := 0; ix < nblocks_x; ix++ {
            block_y := mcu_y * nblocks_y + iy;
            block_x := mcu_x * nblocks_x + ix;
            block_idx := block_y * c.width_in_blocks + block_x;
            const coeff_t* coeffs = &c.coeffs[block_idx << 6];
            EncodeDCTBlockSequential(coeffs, dc_huff_table[i], ac_huff_table[i],
                                     &last_dc_coeff[i], &bw);
          }
        }
      }
      if (bw.pos > (1 << 16)) {
        if (!JPEGWrite(out, bw.data.get(), bw.pos)) {
          return false;
        }
        bw.pos = 0;
      }
    }
  }
  bw.JumpToByteBoundary();
  return !bw.overflow && JPEGWrite(out, bw.data.get(), bw.pos);
}

}  // namespace

func WriteJpeg(jpg *JPEGData, strip_metadata bool, out JPEGOutput) bool {
  static const uint8_t kSOIMarker[2] = { 0xff, 0xd8 };
  static const uint8_t kEOIMarker[2] = { 0xff, 0xd9 };
  std::vector<HuffmanCodeTable> dc_codes;
  std::vector<HuffmanCodeTable> ac_codes;
  return (JPEGWrite(out, kSOIMarker, sizeof(kSOIMarker)) &&
          EncodeMetadata(jpg, strip_metadata, out) &&
          EncodeDQT(jpg.quant, out) &&
          EncodeSOF(jpg, out) &&
          BuildAndEncodeHuffmanCodes(jpg, out, &dc_codes, &ac_codes) &&
          EncodeScan(jpg, dc_codes, ac_codes, out) &&
          JPEGWrite(out, kEOIMarker, sizeof(kEOIMarker)) &&
          (strip_metadata || JPEGWrite(out, jpg.tail_data)));
}

int NullOut(void* data, const uint8_t* buf, size_t count) {
  return count;
}

func BuildSequentialHuffmanCodes(
    jpg *JPEGData,
    std::vector<HuffmanCodeTable>* dc_huffman_code_tables,
    std::vector<HuffmanCodeTable>* ac_huffman_code_tables) {
  var out JPEGOutput
  BuildAndEncodeHuffmanCodes(jpg, out, dc_huffman_code_tables,
                             ac_huffman_code_tables);
}
