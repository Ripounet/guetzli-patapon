package guetzli_patapon

// Returns non-zero if and only if x has a zero byte, i.e. one of
// x & 0xff, x & 0xff00, ..., x & 0xff00000000000000 is zero.
func HasZeroByte(x uint64) uint64 {
  return (x - 0x0101010101010101) & ^x & 0x8080808080808080;
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
		date: make([]byte, length),
		pos: 0,
		put_buffer: 0,
		put_bits: 64,
		overflow: false,
	}
}

func (bw *BitWriter) WriteBits(nbits int, bits uint64) {
    put_bits -= nbits;
    put_buffer |= (bits << put_bits);
    if (put_bits <= 16) {
      // At this point we are ready to emit the most significant 6 bytes of
      // put_buffer_ to the output.
      // The JPEG format requires that after every 0xff byte in the entropy
      // coded section, there is a zero byte, therefore we first check if any of
      // the 6 most significant bytes of put_buffer_ is 0xff.
      if (HasZeroByte(^put_buffer | 0xffff)) {
        // We have a 0xff byte somewhere, examine each byte and append a zero
        // byte if necessary.
        EmitByte((put_buffer >> 56) & 0xff);
        EmitByte((put_buffer >> 48) & 0xff);
        EmitByte((put_buffer >> 40) & 0xff);
        EmitByte((put_buffer >> 32) & 0xff);
        EmitByte((put_buffer >> 24) & 0xff);
        EmitByte((put_buffer >> 16) & 0xff);
      } else if (pos + 6 < len) {
        // We don't have any 0xff bytes, output all 6 bytes without checking.
        data[pos] = (put_buffer >> 56) & 0xff;
        data[pos + 1] = (put_buffer >> 48) & 0xff;
        data[pos + 2] = (put_buffer >> 40) & 0xff;
        data[pos + 3] = (put_buffer >> 32) & 0xff;
        data[pos + 4] = (put_buffer >> 24) & 0xff;
        data[pos + 5] = (put_buffer >> 16) & 0xff;
        pos += 6;
      } else {
        overflow = true;
      }
      put_buffer <<= 48;
      put_bits += 48;
    }
  }

  // Writes the given byte to the output, writes an extra zero if byte is 0xff.
  func (bw *BitWriter) EmitByte(byte_ int) {
    if (pos < len) {
      data[pos] = byte_;
      pos++
    } else {
      overflow = true;
    }
    if (byte_ == 0xff) {
      EmitByte(0);
    }
  }

  func (bw *BitWriter) JumpToByteBoundary() {
    for (put_bits <= 56) {
      c := (put_buffer >> 56) & 0xff;
      EmitByte(c);
      put_buffer <<= 8;
      put_bits += 8;
    }
    if (put_bits < 64) {
      padmask := 0xff >> (64 - put_bits);
      c := ((put_buffer >> 56) & ^padmask) | padmask;
      EmitByte(c);
    }
    put_buffer = 0;
    put_bits = 64;
  }
