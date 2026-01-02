import argparse
import os

import numpy as np
import pydicom

from codec import pad_image, block_dct, quantize_band, zigzag_scan, encode_block_rle_ac
from entropy_coding import _build_huffman_from_freqs


def iter_dicom_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        if "topo" in dirpath.lower():
            continue
        for name in filenames:
            if not name.lower().endswith(".dcm"):
                continue
            path = os.path.join(dirpath, name)
            if "topo" in path.lower():
                continue
            print("\nFound DICOM file:", path, end=" ")
            yield path


def update_freqs_from_image(ac_freqs, dc_freqs, image, q_low, q_high, q_split, level_shift):
    padded_img, _, _ = pad_image(image)
    h, w = padded_img.shape

    max_size = 0
    over_15 = 0
    prev_dc = 0

    for r in range(0, h, 8):
        for c in range(0, w, 8):
            block = padded_img[r:r + 8, c:c + 8].astype(np.float64)
            if level_shift:
                block = block - level_shift
            coeff = block_dct(block)
            q_coeff = quantize_band(coeff, q_low, q_high, q_split)
            zigzag = zigzag_scan(q_coeff)
            dc = int(zigzag[0])
            dc_diff = dc - prev_dc
            prev_dc = dc
            dc_len = abs(dc_diff).bit_length()
            max_size = max(max_size, dc_len)
            if dc_len > 15:
                over_15 += 1
                dc_len = 15
            dc_freqs[dc_len] += 1

            rle = encode_block_rle_ac(zigzag)

            for run, val in rle:
                if run == 0 and val == 0:
                    ac_freqs[0x00] += 1
                    continue

                while run > 15:
                    ac_freqs[0xF0] += 1
                    run -= 16

                length = abs(int(val)).bit_length()
                max_size = max(max_size, length)
                if length > 15:
                    over_15 += 1
                    length = 15

                symbol = (run << 4) | length
                ac_freqs[symbol] += 1

    return max_size, over_15


def main():
    parser = argparse.ArgumentParser(
        description="Build a fixed Huffman table from CT_COLONOGRAPHY DICOM files"
    )
    parser.add_argument(
        "--root",
        default="/ssd7/jiakai/multimedia_hw2/CT_COLONOGRAPHY",
        help="Root directory for DICOM files",
    )
    parser.add_argument("--q-low", type=int, default=10, help="Low-frequency Q step")
    parser.add_argument("--q-high", type=int, default=30, help="High-frequency Q step")
    parser.add_argument("--q-split", type=int, default=4, help="Low/high split (u+v)")
    parser.add_argument(
        "--level-shift",
        default="yes",
        choices=["yes", "no"],
        help="Level shift (yes=auto 2^(B-1), no=disabled)",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Optional file limit")
    args = parser.parse_args()

    ac_freqs = [0] * 256
    dc_freqs = [0] * 16
    total_files = 0
    max_size_seen = 0
    over_15_total = 0

    for path in iter_dicom_files(args.root):
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array
        bit_depth = getattr(dcm, 'BitsStored', np.iinfo(image.dtype).bits)
        level_shift = (1 << (bit_depth - 1)) if args.level_shift == "yes" else 0
        max_size, over_15 = update_freqs_from_image(
            ac_freqs, dc_freqs, image, args.q_low, args.q_high, args.q_split, level_shift
        )
        max_size_seen = max(max_size_seen, max_size)
        over_15_total += over_15
        total_files += 1
        if args.max_files and total_files >= args.max_files:
            break

    _, ac_lengths = _build_huffman_from_freqs(ac_freqs)
    _, dc_lengths = _build_huffman_from_freqs(dc_freqs)

    print(f"Files used: {total_files}")
    print(f"Max size seen: {max_size_seen}")
    print(f"Count size>15 (clamped): {over_15_total}")
    print("MEDICAL_HUFFMAN_AC_LENGTHS = [")
    for i in range(0, 256, 16):
        chunk = ac_lengths[i:i + 16]
        print("    " + ", ".join(str(x) for x in chunk) + ",")
    print("]")
    print("MEDICAL_HUFFMAN_DC_LENGTHS = [")
    for i in range(0, 16, 16):
        chunk = dc_lengths[i:i + 16]
        print("    " + ", ".join(str(x) for x in chunk) + ",")
    print("]")


if __name__ == "__main__":
    main()
