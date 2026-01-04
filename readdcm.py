import pydicom
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def analyze_dicom_file(filepath):
    # 1. 讀取 DICOM 檔案
    dcm = pydicom.dcmread(filepath)
    
    # 2. 提取關鍵壓縮參數 (Header Info)
    rows = dcm.Rows
    cols = dcm.Columns
    bits_stored = dcm.BitsStored        
    bits_allocated = dcm.BitsAllocated  # 通常是 16 (即使只用了 12 bit)
    pixel_repr = dcm.PixelRepresentation # 0=unsigned (無號), 1=signed (有號)
    
    print(f"--- DICOM Header Analysis ---")
    print(f"Dimensions: {rows} x {cols}")
    print(f"Bits Stored (有效位元): {bits_stored}")
    print(f"Bits Allocated (存儲位元): {bits_allocated}")
    print(f"Pixel Representation: {'Signed (有號)' if pixel_repr else 'Unsigned (無號)'}")
    
    # 3. 獲取原始像素數據 (Raw Pixel Data)
    raw_img = dcm.pixel_array
    
    print(f"Raw Min Value: {raw_img.min()}")
    print(f"Raw Max Value: {raw_img.max()}")
    print(f"Data Type: {raw_img.dtype}")
    
    return raw_img, dcm


def to_hu(raw_img, header):
    slope = getattr(header, 'RescaleSlope', 1)
    intercept = getattr(header, 'RescaleIntercept', 0)
    return raw_img * slope + intercept


def window_image(raw_img, header, window_center=40, window_width=400):
    """
    將 Raw 影像轉為 HU 後套用 window，輸出 0-255 的顯示影像
    """
    hu_img = to_hu(raw_img, header)

    min_visible = window_center - (window_width / 2)
    max_visible = window_center + (window_width / 2)

    display_img = np.clip(hu_img, min_visible, max_visible)
    display_img = ((display_img - min_visible) / window_width) * 255
    return display_img


def save_medical_image(raw_img, header, path):
    """
    將原始 16-bit 數據轉換為適合人類觀看的 8-bit 圖像 (僅供顯示用)
    """
    display_img = window_image(raw_img, header, window_center=40, window_width=400)
    
    plt.figure(figsize=(10, 5))
    plt.imsave(path, display_img, cmap='gray')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine DICOM images into one JPG")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input DICOM paths")
    parser.add_argument("--output-dir", default="result", help="Output JPG path")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, filepath in enumerate(args.inputs):
        filename = os.path.basename(filepath).split('.')[0]
        raw_img, header = analyze_dicom_file(filepath)
        save_path = os.path.join(args.output_dir, f"{filename}.jpg")
        save_medical_image(raw_img, header, save_path)

    # filepath_roi = "test_data/1-010.dcm"
    # filepath_recon = "recon.dcm"
