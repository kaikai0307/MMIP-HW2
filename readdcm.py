import pydicom
import numpy as np
import matplotlib.pyplot as plt

def analyze_dicom_file(filepath):
    # 1. 讀取 DICOM 檔案
    dcm = pydicom.dcmread(filepath)
    
    # 2. 提取關鍵壓縮參數 (Header Info)
    # 作業要求：必須處理 >8-bit 的資料 [cite: 25, 26]
    rows = dcm.Rows
    cols = dcm.Columns
    bits_stored = dcm.BitsStored        # 最重要：是 12 還是 16？
    bits_allocated = dcm.BitsAllocated  # 通常是 16 (即使只用了 12 bit)
    pixel_repr = dcm.PixelRepresentation # 0=unsigned (無號), 1=signed (有號)
    
    print(f"--- DICOM Header Analysis ---")
    print(f"Dimensions: {rows} x {cols}")
    print(f"Bits Stored (有效位元): {bits_stored}")
    print(f"Bits Allocated (存儲位元): {bits_allocated}")
    print(f"Pixel Representation: {'Signed (有號)' if pixel_repr else 'Unsigned (無號)'}")
    
    # 3. 獲取原始像素數據 (Raw Pixel Data)
    # 注意：這是您要送進 DCT 壓縮的數據，絕對不要轉成 uint8！
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


def show_medical_image(raw_img, header):
    """
    將原始 16-bit 數據轉換為適合人類觀看的 8-bit 圖像 (僅供顯示用)
    """
    # 設定腹部/結腸的視窗 (Abdomen Window)
    # 這是放射科醫師看的設定：窗位(Level) 40, 窗寬(Width) 400
    display_img = window_image(raw_img, header, window_center=40, window_width=400)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Raw Data (Hist Equ)")
    # 直接看 Raw data，不做特定 windowing，讓 matplotlib 自動對比
    plt.imshow(raw_img, cmap='gray') 
    
    plt.subplot(1, 2, 2)
    plt.title("Doctor View (Abdomen Window)")
    plt.imshow(display_img, cmap='gray')
    
    plt.show()
    plt.imsave("medical_image_display.jpg", display_img, cmap='gray')




# --- 執行讀取 (請修改為您的檔案路徑) ---
# 假設您手邊有從 Medimodel 或 OsiriX 下載的檔案
if __name__ == "__main__":
    filepath = "/ssd7/jiakai/multimedia_hw2/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04 Adult-0.4.1/3.000000-Colosupine  1.0  B30f-4.563/1-010.dcm"

    raw_img, header = analyze_dicom_file(filepath)
    show_medical_image(raw_img, header)
