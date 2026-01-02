# Medical Image Codec

This project implements a simple medical image codec with DCT-based block
compression, zigzag + RLE, and optional Huffman entropy coding. It is built
around CT DICOM images and supports fixed low/high frequency quantization.

## Files

- `codec.py`: DCT/IDCT, quantization, zigzag, RLE encoding, and encode driver.
- `bitstream.py`: bitstream I/O, header layout, and file save logic.
- `decode.py`: bitstream decoding, reconstruction, RMSE/PSNR, and display.
- `entropy_coding.py`: entropy coding (RLE + Huffman) implementation.
- `test.py`: experiment runner for Q-halving and RD curves.
- `build_medical_huffman_table.py`: build a fixed Huffman table from dataset.
- `readdcm.py`: DICOM reading and windowed display utilities.

## Quick Start

### Encode / Decode (single file)

1. Edit the DICOM path in `codec.py` or pass your own path.
2. Run encode:
```
~/miniconda3/envs/codec/bin/python codec.py
```
3. Run decode:
```
~/miniconda3/envs/codec/bin/python decode.py
```

### Experiment 3: Q Halving RD Curve

```
~/miniconda3/envs/codec/bin/python test.py --entropy-method only_RLE
```

Outputs are written under `experiment3/<method>/`:

- `rd_curve.csv` / `rd_curve.jpg`
- `recon_q*_window.jpg`
- `diff_q*.jpg`

Optional level shift:
```
~/miniconda3/envs/codec/bin/python test.py --level-shift yes
~/miniconda3/envs/codec/bin/python test.py --level-shift no
```

### Entropy Coding Methods

- `only_RLE`: RLE with fixed fields.
- `huffman_std`: JPEG standard AC table (may fall back if size>10 appears).
- `huffman_adapt`: dataset-adaptive Huffman table.

## Build a Fixed "Medical" Huffman Table

This scans all DICOM files under CT_COLONOGRAPHY (excluding topo)
and prints a fixed Huffman length table you can hardcode.

```
~/miniconda3/envs/codec/bin/python build_medical_huffman_table.py
```

For a faster approximate table:
```
~/miniconda3/envs/codec/bin/python build_medical_huffman_table.py --max-files 200
```

## Notes

- Topogram/topo series are excluded because they are scout images with
  different acquisition geometry and intensity characteristics, which
  can skew symbol statistics for the main CT slices.
- PSNR uses MAX = 2^B - 1, where B is the DICOM bit depth.
- Windowed display is aligned across `readdcm.py`, `test.py`, and `decode.py`.
