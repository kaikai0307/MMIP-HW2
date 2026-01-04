import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from encode import test_codec_mvp, auto_level_shift
from decode import compute_rmse_psnr, load_compressed_file
from entropy_coding import EntropyCoder
from readdcm import analyze_dicom_file


def build_q_list(base_q, levels):
    q_list = []
    q = base_q
    for _ in range(levels):
        q = max(1, int(q))
        if q_list and q == q_list[-1]:
            break
        q_list.append(q)
        q = q // 2
    return q_list


def parse_q_list(arg):
    if not arg:
        return []
    items = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def run_series(
    raw_img,
    bit_depth,
    q_list,
    q_high_fn,
    q_high_list,
    q_split,
    out_dir,
    entropy_method,
    level_shift,
    pixel_repr,
    mode_label,
):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    if q_high_list is not None and len(q_high_list) != len(q_list):
        raise ValueError("q_high_list length must match q_low list length")

    for idx, q_step in enumerate(q_list):
        if q_high_list is not None:
            q_high = max(1, int(q_high_list[idx]))
        else:
            q_high = max(1, int(round(q_high_fn(q_step))))
        test_codec_mvp(
            raw_img,
            q_step=q_step,
            q_high=q_high,
            q_split=q_split,
            bit_depth=bit_depth,
            entropy_method=entropy_method,
            level_shift=level_shift,
            pixel_repr=pixel_repr,
        )

        mic_path = os.path.join(out_dir, f"output_q{q_step}.mic")
        if os.path.exists("output.mic"):
            os.replace("output.mic", mic_path)
        else:
            raise FileNotFoundError("output.mic not found after encoding.")

        recon_img, header_info = load_compressed_file(mic_path, return_header=True)
        rmse, psnr = compute_rmse_psnr(raw_img, recon_img, bit_depth)

        size_bytes = os.path.getsize(mic_path)
        bpp = (size_bytes * 8) / (header_info["h"] * header_info["w"])

        results.append(
            {
                "mode": mode_label,
                "method": header_info.get("method", entropy_method),
                "level_shift": header_info.get("level_shift", level_shift),
                "q_step": q_step,
                "q_high": q_high,
                "q_ratio": q_high / q_step if q_step else float("nan"),
                "q_split": q_split,
                "size_bytes": size_bytes,
                "bpp": bpp,
                "rmse": rmse,
                "psnr_db": psnr,
            }
        )

    return results


def plot_rd_curve(rows, x_axis, metric, label, color):
    rows = sorted(rows, key=lambda r: r["q_step"])
    if x_axis == "size":
        x_vals = [r["size_bytes"] for r in rows]
        x_label = "Size (bytes)"
    else:
        x_vals = [r["bpp"] for r in rows]
        x_label = "Bitrate (bpp)"

    if metric == "rmse":
        y_vals = [r["rmse"] for r in rows]
        y_label = "RMSE (lower is better)"
    else:
        y_vals = [r["psnr_db"] for r in rows]
        y_label = "PSNR (dB)"

    plt.plot(x_vals, y_vals, marker="o", label=label, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "method",
                "level_shift",
                "q_step",
                "q_high",
                "q_ratio",
                "q_split",
                "size_bytes",
                "bpp",
                "rmse",
                "psnr_db",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="RD curve comparison: split vs single Q")
    parser.add_argument(
        "--dicom",
        default="./test_data/1-010.dcm",
        help="Path to DICOM file",
    )
    parser.add_argument("--q-base", type=int, default=50, help="Base Qlow")
    parser.add_argument("--levels", type=int, default=4, help="Number of halving steps")
    parser.add_argument(
        "--q-low-list",
        default="",
        help="Comma-separated Qlow list (e.g., 50,40,30,20)",
    )
    parser.add_argument(
        "--q-high-list",
        default="",
        help="Comma-separated Qhigh list (also used as Qlow for single-quant curve)",
    )
    parser.add_argument(
        "--q-high-ratio",
        type=float,
        default=2.0,
        help="Qhigh = Qlow * ratio for split curve",
    )
    parser.add_argument("--q-split", type=int, default=4, help="Low/high split (u+v)")
    parser.add_argument("--out-dir", default="experiment/rd_compare", help="Output directory")
    parser.add_argument(
        "--metric",
        default="psnr",
        choices=["psnr", "rmse"],
        help="Quality metric for Y axis",
    )
    parser.add_argument(
        "--x-axis",
        default="bpp",
        choices=["bpp", "size"],
        help="X axis for RD curve",
    )
    parser.add_argument(
        "--level-shift",
        default="yes",
        choices=["yes", "no"],
        help="Level shift (yes=auto, no=disabled)",
    )
    parser.add_argument(
        "--entropy-method",
        default="huffman_adapt",
        choices=["only_RLE", "huffman_std", "huffman_adapt"],
        help="Entropy coding method",
    )
    args = parser.parse_args()

    raw_img, header = analyze_dicom_file(args.dicom)
    bit_depth = getattr(header, "BitsStored", np.iinfo(raw_img.dtype).bits)
    pixel_repr = getattr(header, "PixelRepresentation", None)
    if args.level_shift == "yes":
        level_shift = auto_level_shift(raw_img, bit_depth, pixel_repr)
    else:
        level_shift = 0
    entropy_method = EntropyCoder.normalize_method(args.entropy_method)
    q_low_list = parse_q_list(args.q_low_list)
    q_high_list = parse_q_list(args.q_high_list)
    if q_high_list:
        q_low_split = [max(1, int(round(qh / args.q_high_ratio))) for qh in q_high_list]
        q_low_single = list(q_high_list)
    else:
        if not q_low_list:
            q_low_list = build_q_list(args.q_base, args.levels)
        q_low_split = list(q_low_list)
        q_low_single = list(q_low_list)

    split_dir = os.path.join(args.out_dir, "split")
    single_dir = os.path.join(args.out_dir, "single")

    split_rows = run_series(
        raw_img,
        bit_depth,
        q_low_split,
        lambda q: q * args.q_high_ratio,
        q_high_list if q_high_list else None,
        args.q_split,
        split_dir,
        entropy_method,
        level_shift,
        pixel_repr,
        "split",
    )

    single_rows = run_series(
        raw_img,
        bit_depth,
        q_low_single,
        lambda q: q,
        None,
        args.q_split,
        single_dir,
        entropy_method,
        level_shift,
        pixel_repr,
        "single",
    )

    all_rows = split_rows + single_rows
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "rd_curve_compare.csv")
    write_csv(csv_path, all_rows)

    plt.figure(figsize=(6, 4))
    plot_rd_curve(
        split_rows,
        args.x_axis,
        args.metric,
        f"huffman_adapt split (ratio={args.q_high_ratio:g})",
        color="#d55e00",
    )
    plot_rd_curve(
        single_rows,
        args.x_axis,
        args.metric,
        "huffman_adapt single (Qhigh=Qlow)",
        color="#0072b2",
    )
    plt.title("Rate-Distortion Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, "rd_curve_compare.jpg")
    plt.savefig(plot_path, dpi=150)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
