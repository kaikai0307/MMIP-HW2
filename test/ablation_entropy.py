import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from decode import load_compressed_file
from encode import auto_level_shift, test_codec_mvp
from entropy_coding import EntropyCoder
from readdcm import analyze_dicom_file, to_hu


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


def parse_choice_list(arg):
    if not arg:
        return []
    items = []
    for part in arg.split(","):
        part = part.strip().lower()
        if part:
            items.append(part)
    return items


def compute_rmse_psnr_custom(original_img, reconstructed_img, max_val):
    h = min(original_img.shape[0], reconstructed_img.shape[0])
    w = min(original_img.shape[1], reconstructed_img.shape[1])
    orig = original_img[:h, :w].astype(np.float64)
    recon = reconstructed_img[:h, :w].astype(np.float64)

    mse = np.mean((orig - recon) ** 2)
    rmse = np.sqrt(mse)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(max_val / rmse)
    return rmse, psnr


def run_case(
    raw_img,
    bit_depth,
    pixel_repr,
    q_low,
    q_high,
    q_split,
    entropy_method,
    level_shift,
    metric_domain,
    ablation_type,
    dicom_header,
    out_dir,
):
    entropy_method = EntropyCoder.normalize_method(entropy_method)
    method_dir = os.path.join(
        out_dir,
        entropy_method,
        f"shift_{metric_domain}_{'on' if level_shift else 'off'}",
        f"split_{q_split}",
    )
    os.makedirs(method_dir, exist_ok=True)

    q_low = max(1, int(q_low))
    q_high = max(1, int(q_high))
    test_codec_mvp(
        raw_img,
        q_step=q_low,
        q_high=q_high,
        q_split=q_split,
        bit_depth=bit_depth,
        entropy_method=entropy_method,
        level_shift=level_shift,
        pixel_repr=pixel_repr,
    )

    mic_path = os.path.join(method_dir, f"output_q{q_low}_qh{q_high}.mic")
    if os.path.exists("output.mic"):
        os.replace("output.mic", mic_path)
    else:
        raise FileNotFoundError("output.mic not found after encoding.")

    recon_img, header_info = load_compressed_file(mic_path, return_header=True)
    if metric_domain == "hu":
        hu_raw = to_hu(raw_img.astype(np.float64), dicom_header)
        hu_recon = to_hu(recon_img.astype(np.float64), dicom_header)
        max_val = float(np.max(hu_raw) - np.min(hu_raw))
        if max_val <= 0:
            max_val = 1.0
        rmse, psnr = compute_rmse_psnr_custom(hu_raw, hu_recon, max_val)
    else:
        max_val = (1 << bit_depth) - 1
        rmse, psnr = compute_rmse_psnr_custom(raw_img, recon_img, max_val)

    size_bytes = os.path.getsize(mic_path)
    bpp = (size_bytes * 8) / (header_info["h"] * header_info["w"])

    return {
        "ablation": ablation_type,
        "method": entropy_method,
        "q_low": q_low,
        "q_high": q_high,
        "q_high_ratio": q_high / q_low if q_low else float("nan"),
        "q_split": q_split,
        "metric_domain": metric_domain,
        "level_shift": "yes" if level_shift else "no",
        "size_bytes": size_bytes,
        "bpp": bpp,
        "rmse": rmse,
        "psnr_db": psnr,
    }


def plot_curve(rows, x_axis, metric, label, color):
    rows = sorted(rows, key=lambda r: r["q_low"])
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
                "ablation",
                "method",
                "q_low",
                "q_high",
                "q_high_ratio",
                "q_split",
                "metric_domain",
                "level_shift",
                "size_bytes",
                "bpp",
                "rmse",
                "psnr_db",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Ablation: entropy + quantization sweep")
    parser.add_argument(
        "--dicom",
        default="./test_data/1-010.dcm",
        help="Path to DICOM file",
    )
    parser.add_argument("--q-low", type=int, default=25, help="Qlow for split case")
    parser.add_argument("--q-high", type=int, default=50, help="Qhigh for split case")
    parser.add_argument("--q-baseline", type=int, default=50, help="Baseline Q (high=low)")
    parser.add_argument(
        "--q-split-list",
        default="4",
        help="Comma-separated Qsplit list (u+v)",
    )
    parser.add_argument("--base-q-split", type=int, default=4, help="Baseline Qsplit")
    parser.add_argument(
        "--methods",
        default="only_RLE,huffman_adapt",
        help="Comma-separated entropy methods",
    )
    parser.add_argument("--base-method", default="huffman_adapt", help="Baseline method")
    parser.add_argument(
        "--metric-domain-list",
        default="raw",
        help="Comma-separated metric domains: raw,hu",
    )
    parser.add_argument("--base-metric-domain", default="raw", help="Baseline metric domain")
    parser.add_argument(
        "--level-shift-list",
        default="yes,no",
        help="Comma-separated level shift modes: yes,no",
    )
    parser.add_argument("--base-level-shift", default="yes", help="Baseline level shift")
    parser.add_argument("--out-dir", default="experiment/ablation_entropy", help="Output directory")
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
    args = parser.parse_args()

    raw_img, header = analyze_dicom_file(args.dicom)
    bit_depth = getattr(header, "BitsStored", np.iinfo(raw_img.dtype).bits)
    pixel_repr = getattr(header, "PixelRepresentation", None)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    q_split_list = [int(v) for v in parse_q_list(args.q_split_list)]
    metric_domains = parse_choice_list(args.metric_domain_list)
    level_shift_modes = parse_choice_list(args.level_shift_list)
    if not q_split_list:
        q_split_list = [4]
    if not metric_domains:
        metric_domains = ["raw"]
    if not level_shift_modes:
        level_shift_modes = ["yes"]
    all_rows = []

    base_method = EntropyCoder.normalize_method(args.base_method)
    base_q_split = args.base_q_split
    base_metric_domain = args.base_metric_domain
    base_level_shift = args.base_level_shift
    base_level_shift_value = auto_level_shift(raw_img, bit_depth, pixel_repr) if base_level_shift == "yes" else 0

    def level_shift_value(mode):
        return auto_level_shift(raw_img, bit_depth, pixel_repr) if mode == "yes" else 0

    # Ablation 1: entropy methods (others fixed to baseline)
    for method in methods:
        row = run_case(
            raw_img,
            bit_depth,
            pixel_repr,
            args.q_baseline,
            args.q_baseline,
            base_q_split,
            method,
            base_level_shift_value,
            base_metric_domain,
            "method",
            header,
            args.out_dir,
        )
        all_rows.append(row)

    # Ablation 2: level shift (others fixed)
    for shift_mode in level_shift_modes:
        row = run_case(
            raw_img,
            bit_depth,
            pixel_repr,
            args.q_baseline,
            args.q_baseline,
            base_q_split,
            base_method,
            level_shift_value(shift_mode),
            base_metric_domain,
            "level_shift",
            header,
            args.out_dir,
        )
        all_rows.append(row)

    # Ablation 3: q_split (others fixed)
    for q_split in q_split_list:
        row = run_case(
            raw_img,
            bit_depth,
            pixel_repr,
            args.q_baseline,
            args.q_baseline,
            q_split,
            base_method,
            base_level_shift_value,
            base_metric_domain,
            "q_split",
            header,
            args.out_dir,
        )
        all_rows.append(row)

    # Ablation 4: metric domain (others fixed)
    for domain in metric_domains:
        row = run_case(
            raw_img,
            bit_depth,
            pixel_repr,
            args.q_baseline,
            args.q_baseline,
            base_q_split,
            base_method,
            base_level_shift_value,
            domain,
            "metric_domain",
            header,
            args.out_dir,
        )
        all_rows.append(row)

    # Ablation 5: high/low Q (others fixed)
    for q_low, q_high in [
        (args.q_baseline, args.q_baseline),
        (args.q_low, args.q_high),
    ]:
        row = run_case(
            raw_img,
            bit_depth,
            pixel_repr,
            q_low,
            q_high,
            base_q_split,
            base_method,
            base_level_shift_value,
            base_metric_domain,
            "q_high_low",
            header,
            args.out_dir,
        )
        all_rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "ablation_entropy.csv")
    write_csv(csv_path, all_rows)

    plt.figure(figsize=(6, 4))
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#f0e442", "#000000"]
    plot_groups = []
    for method in methods:
        plot_groups.append((method, "method", method))
    for shift_mode in level_shift_modes:
        plot_groups.append((base_method, "level_shift", f"shift_{shift_mode}"))
    for q_split in q_split_list:
        plot_groups.append((base_method, "q_split", f"split_{q_split}"))
    for domain in metric_domains:
        plot_groups.append((base_method, "metric_domain", f"metric_{domain}"))
    plot_groups.append((base_method, "q_high_low", "q_high_low"))

    for idx, (method, ablation, label) in enumerate(plot_groups):
        rows = [
            r for r in all_rows
            if r["ablation"] == ablation
            and r["method"] == EntropyCoder.normalize_method(method)
        ]
        if rows:
            plot_curve(rows, args.x_axis, args.metric, label, colors[idx % len(colors)])
    plt.title("Entropy Coding Ablation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, "ablation_entropy.jpg")
    plt.savefig(plot_path, dpi=150)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
