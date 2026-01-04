import argparse
import csv
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from decode import load_compressed_file
from readdcm import analyze_dicom_file, to_hu


def parse_bins(bins_arg):
    if not bins_arg:
        boundaries = [-700, 300]
    else:
        boundaries = []
        for part in bins_arg.split(","):
            part = part.strip()
            if not part:
                continue
            lower = part.lower()
            if lower in ("inf", "+inf"):
                boundaries.append(float("inf"))
            elif lower == "-inf":
                boundaries.append(float("-inf"))
            else:
                boundaries.append(float(part))

    if not boundaries:
        raise ValueError("Empty HU bin list")

    edges = boundaries[:]
    if not np.isneginf(edges[0]):
        edges = [float("-inf")] + edges
    if not np.isposinf(edges[-1]):
        edges = edges + [float("inf")]

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            raise ValueError("HU bin edges must be strictly increasing")
    return edges


def compute_bin_stats(hu_raw, hu_recon, edges):
    err = hu_recon - hu_raw
    stats = []
    for i in range(len(edges) - 1):
        low = edges[i]
        high = edges[i + 1]
        mask = (hu_raw >= low) & (hu_raw < high)
        count = int(mask.sum())
        if count == 0:
            rmse = float("nan")
            mae = float("nan")
            mean_err = float("nan")
            max_abs = float("nan")
        else:
            diff = err[mask]
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))
            mean_err = float(np.mean(diff))
            max_abs = float(np.max(np.abs(diff)))

        stats.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "rmse": rmse,
                "mae": mae,
                "mean_err": mean_err,
                "max_abs_err": max_abs,
            }
        )
    return stats


def format_edge(value):
    if np.isneginf(value):
        return "-inf"
    if np.isposinf(value):
        return "inf"
    return f"{value:g}"


def write_csv(path, stats):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin_low",
                "bin_high",
                "count",
                "rmse",
                "mae",
                "mean_err",
                "max_abs_err",
            ],
        )
        writer.writeheader()
        writer.writerows(stats)


def parse_q_step(path):
    match = re.search(r"output_q(\d+)", os.path.basename(path))
    if match:
        return int(match.group(1))
    return None


def collect_mic_files(mic_dir, pattern, limit):
    paths = glob.glob(os.path.join(mic_dir, pattern))
    if not paths:
        raise ValueError(f"No files match {pattern} in {mic_dir}")

    def sort_key(p):
        q = parse_q_step(p)
        return (q is None, q if q is not None else 0, os.path.basename(p))

    paths = sorted(paths, key=sort_key)
    if limit and limit > 0:
        paths = paths[:limit]
    return paths


def plot_rmse_curve(x_values, rmse_by_bin, bin_labels, x_label, plot_path):
    plt.figure(figsize=(9, 5))
    for idx, label in enumerate(bin_labels):
        plt.plot(x_values, rmse_by_bin[idx], marker="o", label=label)
    plt.xlabel(x_label)
    plt.ylabel("RMSE (HU)")
    plt.title("HU-bin RMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="HU-binned error statistics")
    parser.add_argument(
        "--dicom",
        default="/ssd7/jiakai/multimedia_hw2/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04 Adult-0.4.1/3.000000-Colosupine  1.0  B30f-4.563/1-010.dcm",
        help="Path to DICOM file",
    )
    parser.add_argument("--mic", default="output.mic", help="Path to .mic file")
    parser.add_argument(
        "--mic-dir",
        default="./experiment3/rd_compare1.5/single",
        help="Directory with output*.mic files (process multiple files if set)",
    )
    parser.add_argument(
        "--pattern",
        default="output*.mic",
        help="Glob pattern inside --mic-dir",
    )
    parser.add_argument(
        "--limit",
        "--num",
        dest="limit",
        type=int,
        default=0,
        help="Process only first N files (0 = all)",
    )
    parser.add_argument(
        "--bins",
        default="",
        help="Comma-separated HU boundaries (auto adds -inf/inf, default: -700,300)",
    )
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    parser.add_argument("--plot", default="", help="Optional plot output path")
    args = parser.parse_args()

    raw_img, header = analyze_dicom_file(args.dicom)
    raw_full = raw_img.astype(np.float64)
    hu_raw_full = to_hu(raw_full, header)

    edges = parse_bins(args.bins)

    if args.mic_dir:
        mic_files = collect_mic_files(args.mic_dir, args.pattern, args.limit)
        all_stats = []
        rmse_by_bin = [[] for _ in range(len(edges) - 1)]
        x_values = []
        bin_labels = [
            f"[{format_edge(edges[i])}, {format_edge(edges[i + 1])})"
            for i in range(len(edges) - 1)
        ]

        print(f"Found {len(mic_files)} file(s) in {args.mic_dir}")
        for path in mic_files:
            recon_img, _ = load_compressed_file(path, return_header=True)
            h = min(hu_raw_full.shape[0], recon_img.shape[0])
            w = min(hu_raw_full.shape[1], recon_img.shape[1])
            hu_raw = hu_raw_full[:h, :w]
            recon = recon_img[:h, :w].astype(np.float64)
            hu_recon = to_hu(recon, header)
            stats = compute_bin_stats(hu_raw, hu_recon, edges)

            q_step = parse_q_step(path)
            x_values.append(q_step if q_step is not None else len(x_values) + 1)
            for idx, s in enumerate(stats):
                rmse_by_bin[idx].append(s["rmse"])
                row = {
                    "file": os.path.basename(path),
                    "q_step": q_step if q_step is not None else "",
                    "bin_low": s["bin_low"],
                    "bin_high": s["bin_high"],
                    "count": s["count"],
                    "rmse": s["rmse"],
                    "mae": s["mae"],
                    "mean_err": s["mean_err"],
                    "max_abs_err": s["max_abs_err"],
                }
                all_stats.append(row)

            print(f"{os.path.basename(path)} -> q={q_step}")

        if args.csv:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "file",
                        "q_step",
                        "bin_low",
                        "bin_high",
                        "count",
                        "rmse",
                        "mae",
                        "mean_err",
                        "max_abs_err",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_stats)
            print(f"Saved CSV: {args.csv}")

        plot_path = args.plot or os.path.join(args.mic_dir, "hu_bins_plot.png")
        x_label = "Q step" if any(parse_q_step(p) is not None for p in mic_files) else "Sample"
        plot_rmse_curve(x_values, rmse_by_bin, bin_labels, x_label, plot_path)
        print(f"Saved plot: {plot_path}")
        return

    recon_img, _ = load_compressed_file(args.mic, return_header=True)

    h = min(raw_full.shape[0], recon_img.shape[0])
    w = min(raw_full.shape[1], recon_img.shape[1])
    hu_raw = hu_raw_full[:h, :w]
    recon = recon_img[:h, :w].astype(np.float64)
    hu_recon = to_hu(recon, header)

    stats = compute_bin_stats(hu_raw, hu_recon, edges)

    print("HU bin error stats:")
    for s in stats:
        low = format_edge(s["bin_low"])
        high = format_edge(s["bin_high"])
        print(
            f"[{low}, {high}): n={s['count']}, "
            f"RMSE={s['rmse']:.4f}, MAE={s['mae']:.4f}, "
            f"Mean={s['mean_err']:.4f}, MaxAbs={s['max_abs_err']:.4f}"
        )

    if args.csv:
        write_csv(args.csv, stats)
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
