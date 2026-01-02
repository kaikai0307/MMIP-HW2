# decode.py (建議新建這個檔案)
import struct
import numpy as np
import matplotlib.pyplot as plt
from bitstream import BitReader
from codec import inverse_zigzag_scan, dequantize_band, block_idct
from entropy_coding import EntropyCoder
from readdcm import analyze_dicom_file, window_image

def load_compressed_file(filepath, return_header=False):
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
        
    # 1. 解析 Header (根據您 encode 時寫入的格式)
    # v5: '>4sBHHBBBBBH' = Magic(4), Ver(1), H(2), W(2), Depth(1), Qlow(1), Qhigh(1), Qsplit(1), Method(1), LevelShift(2)
    header_prefix_size = struct.calcsize('>4sB')
    magic, ver = struct.unpack('>4sB', file_bytes[:header_prefix_size])
    if magic != b'MIPC':
        raise ValueError("Invalid file format!")

    method = 'only_RLE'
    coder = None

    if ver >= 5:
        header_size = struct.calcsize('>4sBHHBBBBBH')
        header_data = file_bytes[:header_size]
        magic, ver, h, w, depth, q_low, q_high, q_split, method_id, level_shift = struct.unpack('>4sBHHBBBBBH', header_data)
    elif ver == 4:
        header_size = struct.calcsize('>4sBHHBBBBB')
        header_data = file_bytes[:header_size]
        magic, ver, h, w, depth, q_low, q_high, q_split, method_id = struct.unpack('>4sBHHBBBBB', header_data)
        level_shift = 0
    else:
        raise ValueError("Unsupported file version (expected v4/v5). Please re-encode.")

    if ver >= 4:
        lengths_end = header_size
        if method_id == 1:
            method = 'huffman_std'
            coder = EntropyCoder.create_for_decoding(method)
        elif method_id == 2:
            method = 'huffman_adapt'
            lengths_start = header_size
            lengths_mid = lengths_start + 256
            lengths_end = lengths_mid + 16
            ac_lengths = list(file_bytes[lengths_start:lengths_mid])
            dc_lengths = list(file_bytes[lengths_mid:lengths_end])
            coder = EntropyCoder.create_for_decoding(
                method,
                lengths={"ac": ac_lengths, "dc": dc_lengths},
            )
        body_data = file_bytes[lengths_end:]
    
    if coder is None:
        coder = EntropyCoder.create_for_decoding(method)
    coder.reset_state()
    print(
        f"Header Info -> Size: {h}x{w}, Depth: {depth}, Qlow: {q_low}, "
        f"Qhigh: {q_high}, Qsplit: {q_split}, Method: {method}, Shift: {level_shift}"
    )
    
    # 2. 準備解碼
    reader = BitReader(body_data)
    reconstructed_img = np.zeros((h, w)) # 這裡先不考慮 padding 的邊緣，簡化處理
    
    # 計算有 padding 的尺寸 (因為編碼時是以 8x8 為單位)
    # 如果您編碼時有 padding，這裡也要模擬同樣的迴圈
    # 為了簡化，我們先假設圖是 512x512 (剛好整除)
    # 如果不是 8 的倍數，迴圈要跑到 padding 後的大小
    pad_h = ((h + 7) // 8) * 8
    pad_w = ((w + 7) // 8) * 8
    
    # 3. 逐區塊解碼
    for r in range(0, pad_h, 8):
        for c in range(0, pad_w, 8):
            # A. 從 bitstream 拉出係數
            zigzag = coder.decode_block(reader)
            
            # B. Inverse ZigZag
            q_coeff = inverse_zigzag_scan(zigzag)
            
            # C. Dequantize
            dct_coeff = dequantize_band(q_coeff, q_low, q_high, q_split)
            
            # D. IDCT
            block = block_idct(dct_coeff)
            if level_shift:
                block = block + level_shift
            
            # E. 填回影像 (注意邊界檢查，不要寫出界)
            # 這裡只填入有效範圍
            r_end = min(r+8, h)
            c_end = min(c+8, w)
            reconstructed_img[r:r_end, c:c_end] = block[:r_end-r, :c_end-c]
            
    if return_header:
        header_info = {
            "h": h,
            "w": w,
            "depth": depth,
            "q_low": q_low,
            "q_high": q_high,
            "q_split": q_split,
            "method": method,
            "level_shift": level_shift,
            "ver": ver,
        }
        return reconstructed_img, header_info
    return reconstructed_img

def compute_rmse_psnr(original_img, reconstructed_img, bit_depth):
    h = min(original_img.shape[0], reconstructed_img.shape[0])
    w = min(original_img.shape[1], reconstructed_img.shape[1])
    orig = original_img[:h, :w].astype(np.float64)
    recon = reconstructed_img[:h, :w].astype(np.float64)

    mse = np.mean((orig - recon) ** 2)
    rmse = np.sqrt(mse)

    max_val = (1 << bit_depth) - 1
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(max_val / rmse)
    return rmse, psnr

if __name__ == "__main__":

    # --- 測試解碼 ---
    filepath = "output.mic"
    img_recon, header_info = load_compressed_file(filepath, return_header=True)

    # --- RMSE / PSNR ---
    # 請填入原始 DICOM 檔案路徑以計算 RMSE/PSNR
    original_filepath = "/ssd7/jiakai/multimedia_hw2/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04 Adult-0.4.1/3.000000-Colosupine  1.0  B30f-4.563/1-010.dcm"
    raw_img, header = analyze_dicom_file(original_filepath)
    rmse, psnr = compute_rmse_psnr(raw_img, img_recon, header_info["depth"])
    print(f"RMSE: {rmse:.4f}")
    print(f"PSNR: {psnr:.2f} dB (MAX=2^{header_info['depth']}-1)")

    # --- 顯示結果 (Windowed) ---
    window_center = 40
    window_width = 400
    recon_window = window_image(img_recon, header, window_center, window_width)
    plt.imshow(recon_window, cmap="gray", vmin=0, vmax=255)
    plt.title("Reconstructed Image (Windowed)")
    plt.show()
    plt.imsave("reconstructed_image_window.jpg", recon_window, cmap="gray", vmin=0, vmax=255)
