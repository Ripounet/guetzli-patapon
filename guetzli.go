package guetzli_patapon

const (
	kDefaultJPEGQuality = 95

	// An upper estimate of memory usage of Guetzli. The bound is
	// max(kLowerMemusaeMB * 1<<20, pixel_count * kBytesPerPixel)
	kBytesPerPixel    = 125
	kLowestMemusageMB = 100 // in MB

	kDefaultMemlimitMB = 6000 // in MB
)

func BlendOnBlack(val, alpha byte) byte {
	return (int(val)*int(alpha) + 128) / 255
}

func ReadPNG(string data) (ok bool, xsize, ysize int, rgb []byte) {
	png_ptr := png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr)
	if png_ptr == nil {
		return
	}

	info_ptr := png_create_info_struct(png_ptr)
	if info_ptr == nil {
		png_destroy_read_struct(&png_ptr, nullptr, nullptr)
		return
	}

	if setjmp(png_jmpbuf(png_ptr)) != 0 {
		// Ok we are here because of the setjmp.
		png_destroy_read_struct(&png_ptr, &info_ptr, nullptr)
		return
	}

	panic("TODO")
	/*
	  std::istringstream memstream(data, std::ios::in | std::ios::binary);
	  png_set_read_fn(png_ptr, static_cast<void*>(&memstream), [](png_structp png_ptr, png_bytep outBytes, png_size_t byteCountToRead) {
	    std::istringstream& memstream = *static_cast<std::istringstream*>(png_get_io_ptr(png_ptr));

	    memstream.read(reinterpret_cast<char*>(outBytes), byteCountToRead);

	    if (memstream.eof()) png_error(png_ptr, "unexpected end of data");
	    if (memstream.fail()) png_error(png_ptr, "read from memory error");
	  });
	*/

	// The png_transforms flags are as follows:
	// packing == convert 1,2,4 bit images,
	// strip == 16 . 8 bits / channel,
	// shift == use sBIT dynamics, and
	// expand == palettes . rgb, grayscale . 8 bit images, tRNS . alpha.
	const png_transforms uint = PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND | PNG_TRANSFORM_STRIP_16

	png_read_png(png_ptr, info_ptr, png_transforms, nullptr)

	png_bytep * row_pointers = png_get_rows(png_ptr, info_ptr)

	*xsize = png_get_image_width(png_ptr, info_ptr)
	*ysize = png_get_image_height(png_ptr, info_ptr)
	rgb.resize(3 * (*xsize) * (*ysize))

	components := png_get_channels(png_ptr, info_ptr)
	switch components {
	case 1:
		// GRAYSCALE
		for y := 0; y < *ysize; y++ {
			row_in := row_pointers[y]
			row_out = rgb[3*y*(*xsize):]
			for x := 0; x < *xsize; x++ {
				gray := row_in[x]
				row_out[3*x+0] = gray
				row_out[3*x+1] = gray
				row_out[3*x+2] = gray
			}
		}
	case 2:
		// GRAYSCALE + ALPHA
		for y := 0; y < *ysize; y++ {
			row_in := row_pointers[y]
			row_out := rgb[3*y*(*xsize):]
			for x := 0; x < *xsize; x++ {
				gray := BlendOnBlack(row_in[2*x], row_in[2*x+1])
				row_out[3*x+0] = gray
				row_out[3*x+1] = gray
				row_out[3*x+2] = gray
			}
		}
	case 3:
		// RGB
		for y := 0; y < *ysize; y++ {
			row_in := row_pointers[y]
			row_out := rgb[3*y*(*xsize):]
			copy(row_out[:3*(*xsize)], row_in)
		}
	case 4:
		// RGBA
		for y := 0; y < *ysize; y++ {
			row_in := row_pointers[y]
			row_out := rgb[3*y*(*xsize):]
			for x := 0; x < *xsize; x++ {
				alpha := row_in[4*x+3]
				row_out[3*x+0] = BlendOnBlack(row_in[4*x+0], alpha)
				row_out[3*x+1] = BlendOnBlack(row_in[4*x+1], alpha)
				row_out[3*x+2] = BlendOnBlack(row_in[4*x+2], alpha)
			}
		}
	default:
		png_destroy_read_struct(&png_ptr, &info_ptr, nullptr)
		return
	}
	png_destroy_read_struct(&png_ptr, &info_ptr, nullptr)
	ok = true
	return
}

func ReadFileOrDie(filename string) []byte {
	read_from_stdin := filename == "-"
	var buffer []byte
	var err error

	if read_from_stdin {
		buffer, err = ioutil.ReadAll(os.Stdin)
	} else {
		buffer, err = ioutil.ReadFile(filename)
	}
	if err != nil {
		log.Fatalln("Can't open input file:", err)
	}
	return buffer
}

func WriteFileOrDie(filename string, contents []byte) {
	write_to_stdout := filename == "-"
	var err error

	if write_to_stdout {
		_, err = os.Stdout.Write(contents)
	} else {
		err = ioutil.WriteFile(filename, contents, 0666)
	}
	if err != nil {
		log.Fatalln("Can't write:", err)
	}
}

var (
	flagVerbose    = flag.Bool("verbose", false, "Print a verbose trace of all attempts to standard output")
	flagQuality    = flag.Int("quality", kDefaultJPEGQuality, "Visual quality to aim for, expressed as a JPEG quality value")
	flagMemLimit   = flag.iNT("memlimit", kDefaultMemlimitMB, "Memory limit in MB. Guetzli will fail if unable to stay under the limit")
	flagNoMemLimit = flag.Bool("nomemlimit", false, "Do not limit memory usage")
)

func usage() {
	fmt.Fprintln(os.Stderr,
		"Guetzli JPEG compressor. Usage: \n",
		"guetzli [flags] input_filename output_filename\n",
		"\n",
		"Flags:\n",
		// "  --verbose    - Print a verbose trace of all attempts to standard output.\n"
		// "  --quality Q  - Visual quality to aim for, expressed as a JPEG quality value.\n"
		// "                 Default value is %d.\n"
		// "  --memlimit M - Memory limit in MB. Guetzli will fail if unable to stay under\n"
		// "                 the limit. Default limit is %d MB.\n"
		// "  --nomemlimit - Do not limit memory usage.\n", kDefaultJPEGQuality, kDefaultMemlimitMB);
	)
	flag.PrintDefaults()
	os.Exit(1)
}

func main() {
	flag.Usage = usage
	flag.Parse()

	if len(flag.Args()) != 2 {
		usage()
	}

	in_data := ReadFileOrDie(argv[opt_idx])
	var out_data []byte

	var params Params
	params.butteraugli_target = float32(ButteraugliScoreForQuality(quality))

	var stats ProcessStats

	if *flagVerbose {
		stats.debug_output_file = os.Stderr
	}

	kPNGMagicBytes := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}
	if len(in_data) >= 8 && bytes.Equal(in_data[:8], kPNGMagicBytes) {
		var xsize, ysize int
		var rgb []byte
		var ok bool
		if ok, xsize, ysize, rgb := ReadPNG(in_data); !ok {
			log.Fatalln("Error reading PNG data from input file\n")
		}
		pixels := float64(xsize * ysize)
		if memlimit_mb != -1 &&
			(pixels*kBytesPerPixel/(1<<20) > memlimit_mb ||
				memlimit_mb < kLowestMemusageMB) {
			log.Fatalln("Memory limit would be exceeded. Failing.\n")
		}
		if !Process(params, &stats, rgb, xsize, ysize, &out_data) {
			log.Fatalln("Guetzli processing failed\n")
			return 1
		}
	} else {
		var jpg_header JPEGData
		if !ReadJpeg(in_data, JPEG_READ_HEADER, &jpg_header) {
			log.Fatalln("Error reading JPG data from input file\n")
			return 1
		}
		pixels := float64(jpg_header.width) * jpg_header.height
		if memlimit_mb != -1 &&
			(pixels*kBytesPerPixel/(1<<20) > memlimit_mb ||
				memlimit_mb < kLowestMemusageMB) {
			log.Fatalln("Memory limit would be exceeded. Failing.\n")
			return 1
		}
		if !Process(params, &stats, in_data, &out_data) {
			log.Fatalln("Guetzli processing failed\n")
			return 1
		}
	}

	WriteFileOrDie(argv[opt_idx+1], out_data)
	return 0
}
