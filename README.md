# Medical Image Codec

This project implements a simple medical image codec with DCT-based block
compression, zigzag + RLE, and optional Huffman entropy coding. It is built
around CT DICOM images and supports fixed low/high frequency quantization.

## Files

- `codec.py`: CLI entry for encode/decode.
- `bitstream.py`: bitstream I/O, header layout, and file save logic.
- `decode.py`: bitstream decoding, reconstruction, RMSE/PSNR, and DICOM output.
- `entropy_coding.py`: entropy coding (RLE + Huffman) implementation.
- `encode.py`: encoding core (DCT/quantization/zigzag/RLE) + encode-only CLI.
- `test/q_ratio_test.py`: RD curve comparison (split vs single quantization).
- `test/hu_error_stats.py`: HU-binned error stats and plots.
- `build_medical_huffman_table.py`: build a fixed Huffman table from dataset.
- `readdcm.py`: DICOM reading and windowed display utilities.

## Compression Pipeline

1. Read DICOM pixel data.
2. Optional level shift (default: `2^(B-1)`).
3. 8x8 block DCT.
4. Low/high frequency quantization (Qlow/Qhigh).
5. Zigzag scan.
6. DC DPCM + AC RLE.
7. Entropy coding (fixed medical table or adaptive).
8. Write header + bitstream (`.mic`).

## Quick Start

### Encode / Decode (single file)

Use `./test_data/1-010.dcm` for the test input.

1. Run encode:
```
python codec.py encode --input ./test_data/1-010.dcm --output output.mic --quality 50
```
2. Run decode:
```
python codec.py decode --input output.mic --output recon.dcm
```

3. Visualize (combined JPG):
```
python readdcm.py --inputs ./test_data/1-010.dcm recon.dcm
```

Quality (`--quality`) is the quantization step (Q). Larger values give more
compression but lower visual quality.

### Experiment: RD Curve Comparison

```
python test/q_ratio_test.py --entropy-method huffman_adapt
```

Usage options:
```
python test/q_ratio_test.py --q-high-list 80,60,40,30 --q-high-ratio 2 --entropy-method huffman_adapt
python test/q_ratio_test.py --q-low-list 80,60,40,30 --entropy-method huffman_adapt
```


### Entropy Coding Methods

- `only_RLE`: RLE with fixed fields.
- `huffman_std`: fixed medical table (with fallback when out-of-range).
- `huffman_adapt`: dataset-adaptive Huffman table.

## Build a Fixed "Medical" Huffman Table

This scans all DICOM files under CT_COLONOGRAPHY (excluding topo)
and prints a fixed Huffman length table you can hardcode.

```
python build_medical_huffman_table.py
```

For a faster approximate table:
```
python build_medical_huffman_table.py --max-files 200
```

## Notes

- Topogram/topo series are excluded because they are scout images with
  different acquisition geometry and intensity characteristics, which
  can skew symbol statistics for the main CT slices.
- PSNR uses MAX = 2^B - 1, where B is the DICOM bit depth.
- Windowed display and combined JPG output are handled in `readdcm.py`.
- The `.mic` extension stands for **Medical Image Codec**, used to label the
  custom compressed bitstream format in this project.
