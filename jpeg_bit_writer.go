package guetzli_patapon

// Returns true if and only if x has a zero byte, i.e. one of
// x & 0xff, x & 0xff00, ..., x & 0xff00000000000000 is zero.
func HasZeroByte(x uint64) bool {
  return ((x - 0x0101010101010101) & ^x & 0x8080808080808080) != 0;
}

// Handles the packing of bits into output bytes.
type BitWriter struct {
  length int;
  data []byte;
  pos int;
  put_buffer uint64;
  put_bits int;
  overflow bool;
};

func NewBitWriter(length int) *BitWriter {
	return &BitWriter{
		length: length,
		data: make([]byte, length),
		pos: 0,
		put_buffer: 0,
		put_bits: 64,
		overflow: false,
	}
}

func (bw *BitWriter) WriteBits(nbits int, bits uint64) {
    bw.put_bits -= nbits;
    bw.put_buffer |= (bits << uint(bw.put_bits));
    if (bw.put_bits <= 16) {
      // At this point we are ready to emit the most significant 6 bytes of
      // put_buffer_ to the output.
      // The JPEG format requires that after every 0xff byte in the entropy
      // coded section, there is a zero byte, therefore we first check if any of
      // the 6 most significant bytes of put_buffer_ is 0xff.
      if (HasZeroByte(^bw.put_buffer | 0xffff)) {
        // We have a 0xff byte somewhere, examine each byte and append a zero
        // byte if necessary.
        bw.EmitByte(int((bw.put_buffer >> 56) & 0xff));
        bw.EmitByte(int((bw.put_buffer >> 48) & 0xff));
        bw.EmitByte(int((bw.put_buffer >> 40) & 0xff));
        bw.EmitByte(int((bw.put_buffer >> 32) & 0xff));
        bw.EmitByte(int((bw.put_buffer >> 24) & 0xff));
        bw.EmitByte(int((bw.put_buffer >> 16) & 0xff));
      } else if (bw.pos + 6 < bw.length) {
        // We don't have any 0xff bytes, output all 6 bytes without checking.
        bw.data[bw.pos] = byte((bw.put_buffer >> 56) & 0xff);
        bw.data[bw.pos + 1] = byte((bw.put_buffer >> 48) & 0xff);
        bw.data[bw.pos + 2] = byte((bw.put_buffer >> 40) & 0xff);
        bw.data[bw.pos + 3] = byte((bw.put_buffer >> 32) & 0xff);
        bw.data[bw.pos + 4] = byte((bw.put_buffer >> 24) & 0xff);
        bw.data[bw.pos + 5] = byte((bw.put_buffer >> 16) & 0xff);
        bw.pos += 6;
      } else {
        bw.overflow = true;
      }
      bw.put_buffer <<= 48;
      bw.put_bits += 48;
    }
  }

  // Writes the given byte to the output, writes an extra zero if byte is 0xff.
  func (bw *BitWriter) EmitByte(byte_ int) {
    if (bw.pos < bw.length) {
      bw.data[bw.pos] = byte(byte_);
      bw.pos++
    } else {
      bw.overflow = true;
    }
    if (byte_ == 0xff) {
      bw.EmitByte(0);
    }
  }

  func (bw *BitWriter) JumpToByteBoundary() {
    for (bw.put_bits <= 56) {
      c := (bw.put_buffer >> 56) & 0xff;
      bw.EmitByte(int(c));
      bw.put_buffer <<= 8;
      bw.put_bits += 8;
    }
    if (bw.put_bits < 64) {
      padmask := 0xff >> (64 - uint(bw.put_bits));
      c := (int(bw.put_buffer >> 56) & ^padmask) | padmask;
      bw.EmitByte(c);
    }
    bw.put_buffer = 0;
    bw.put_bits = 64;
  }
