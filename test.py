import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from codec import test_codec_mvp
from decode import compute_rmse_psnr, load_compressed_file
from entropy_coding import EntropyCoder
from readdcm import analyze_dicom_file, to_hu, window_image


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


def run_experiment(
    dicom_path,
    base_q,
    levels,
    q_high,
    q_split,
    out_dir,
    window_center,
    window_width,
    entropy_method,
    level_shift_mode,
):
    entropy_method = EntropyCoder.normalize_method(entropy_method)
    if os.path.basename(out_dir) != entropy_method:
        out_dir = os.path.join(out_dir, entropy_method)
    os.makedirs(out_dir, exist_ok=True)

    raw_img, header = analyze_dicom_file(dicom_path)
    bit_depth = getattr(header, "BitsStored", np.iinfo(raw_img.dtype).bits)
    level_shift = (1 << (bit_depth - 1)) if level_shift_mode == "yes" else 0
    raw_hu = to_hu(raw_img, header)
    raw_window = window_image(raw_img, header, window_center, window_width)
    diff_range = window_width / 2

    raw_window_path = os.path.join(out_dir, "raw_window.jpg")
    plt.imsave(raw_window_path, raw_window, cmap="gray", vmin=0, vmax=255)

    q_list = build_q_list(base_q, levels)
    results = []

    for q_step in q_list:
        qh = q_step if q_high is None else q_high

        t0 = time.perf_counter()
        test_codec_mvp(
            raw_img,
            q_step=q_step,
            q_high=qh,
            q_split=q_split,
            bit_depth=bit_depth,
            entropy_method=entropy_method,
            level_shift=level_shift,
        )
        t1 = time.perf_counter()

        mic_path = os.path.join(out_dir, f"output_q{q_step}.mic")
        if os.path.exists("output.mic"):
            os.replace("output.mic", mic_path)
        t2 = time.perf_counter()

        recon_img, header_info = load_compressed_file(mic_path, return_header=True)
        t3 = time.perf_counter()
        rmse, psnr = compute_rmse_psnr(raw_img, recon_img, bit_depth)
        t4 = time.perf_counter()

        size_bytes = os.path.getsize(mic_path)
        bpp = (size_bytes * 8) / (header_info["h"] * header_info["w"])

        recon_window = window_image(recon_img, header, window_center, window_width)
        recon_path = os.path.join(out_dir, f"recon_q{q_step}_window.jpg")
        plt.imsave(recon_path, recon_window, cmap="gray", vmin=0, vmax=255)

        recon_hu = to_hu(recon_img, header)
        diff = recon_hu - raw_hu
        diff_path = os.path.join(out_dir, f"diff_q{q_step}.jpg")
        plt.imsave(diff_path, diff, cmap="bwr", vmin=-diff_range, vmax=diff_range)
        t5 = time.perf_counter()

        results.append(
            {
                "method": header_info.get("method", entropy_method),
                "level_shift": header_info.get("level_shift", level_shift),
                "q_step": q_step,
                "q_high": qh,
                "q_split": q_split,
                "size_bytes": size_bytes,
                "bpp": bpp,
                "rmse": rmse,
                "psnr_db": psnr,
                "t_encode_s": t1 - t0,
                "t_move_s": t2 - t1,
                "t_decode_s": t3 - t2,
                "t_metric_s": t4 - t3,
                "t_visual_s": t5 - t4,
            }
        )

    csv_path = os.path.join(out_dir, "rd_curve.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "level_shift",
                "q_step",
                "q_high",
                "q_split",
                "size_bytes",
                "bpp",
                "rmse",
                "psnr_db",
                "t_encode_s",
                "t_move_s",
                "t_decode_s",
                "t_metric_s",
                "t_visual_s",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    bpp_vals = [r["bpp"] for r in results]
    psnr_vals = [r["psnr_db"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(bpp_vals, psnr_vals, marker="o")
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title("Rate-Distortion Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rd_curve.jpg"), dpi=150)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: Q halving RD curve")
    parser.add_argument(
        "--dicom",
        default="/ssd7/jiakai/multimedia_hw2/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04 Adult-0.4.1/3.000000-Colosupine  1.0  B30f-4.563/1-010.dcm",
        help="Path to DICOM file",
    )
    parser.add_argument("--q-base", type=int, default=10, help="Base Q step")
    parser.add_argument("--levels", type=int, default=4, help="Number of halving steps")
    parser.add_argument("--q-high", type=int, default=None, help="High-frequency Q step")
    parser.add_argument("--q-split", type=int, default=4, help="Low/high split (u+v)")
    parser.add_argument("--out-dir", default="experiment3", help="Output directory")
    parser.add_argument("--window-center", type=float, default=40, help="CT window center")
    parser.add_argument("--window-width", type=float, default=400, help="CT window width")
    parser.add_argument(
        "--level-shift",
        default="yes",
        choices=["yes", "no"],
        help="Level shift (yes=auto 2^(B-1), no=disabled)",
    )
    parser.add_argument(
        "--entropy-method",
        default="only_RLE",
        choices=["only_RLE", "huffman_std", "huffman_adapt"],
        help="Entropy coding method",
    )
    args = parser.parse_args()

    results = run_experiment(
        args.dicom,
        args.q_base,
        args.levels,
        args.q_high,
        args.q_split,
        args.out_dir,
        args.window_center,
        args.window_width,
        args.entropy_method,
        args.level_shift,
    )

    for r in results:
        print(
            f"method={r['method']}, shift={r['level_shift']}, Q={r['q_step']}, "
            f"size={r['size_bytes']} bytes, bpp={r['bpp']:.4f}, "
            f"RMSE={r['rmse']:.4f}, PSNR={r['psnr_db']:.2f} dB, "
            f"t_enc={r['t_encode_s']:.3f}s, t_dec={r['t_decode_s']:.3f}s"
        )
