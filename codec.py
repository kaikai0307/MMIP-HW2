import numpy as np
from readdcm import analyze_dicom_file
from bitstream import BitWriter, save_compressed_file


###附檔名 .mic (Medical Image Codec)

def pad_image(image, block_size=8):
    """
    將影像長寬補滿為 block_size 的倍數
    """
    h, w = image.shape
    # 計算需要補多少
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    
    # 使用 edge padding (複製邊緣像素) 比補 0 好，能減少邊緣的高頻偽影
    padded_img = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
    return padded_img, h, w  # 回傳原始長寬以便將來裁切回來

def get_dct_matrix(N=8):
    """
    產生 N x N 的 DCT 變換矩陣 T
    """
    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                alpha = np.sqrt(1 / N)
            else:
                alpha = np.sqrt(2 / N)
            T[i, j] = alpha * np.cos((np.pi * (2 * j + 1) * i) / (2 * N))
    return T

# 預先計算好 T 矩陣，全域共用以加速
DCT_MATRIX = get_dct_matrix(8)
IDCT_MATRIX = DCT_MATRIX.T  # DCT 矩陣是正交的，反變換就是轉置

def block_dct(block):
    """ 對 8x8 區塊執行 DCT """
    # 轉換為 float 避免溢位，並減去 128 (或 2^(B-1)) 讓數值中心化 (Level Shift)
    # 對於 12-bit 影像，中心化通常減 2048，這一步可選，但標準 JPEG 會做
    return np.dot(np.dot(DCT_MATRIX, block), IDCT_MATRIX)

def block_idct(coefficients):
    """ 對 8x8 係數執行反 DCT (IDCT) """
    return np.dot(np.dot(IDCT_MATRIX, coefficients), DCT_MATRIX)

def quantize(dct_coeffs, quality_step):
    """
    簡單的均勻量化
    Quality Step (Q) 越大 -> 壓縮率越高 -> 畫質越差
    """
    # 1. 除以 Q 並四捨五入 (Round)
    q_coeffs = np.round(dct_coeffs / quality_step)
    
    # 2. 轉為整數
    return q_coeffs.astype(int)

def dequantize(q_coeffs, quality_step):
    """
    反量化：乘回去
    """
    return q_coeffs * quality_step

_Q_MATRIX_CACHE = {}

def build_q_matrix(q_low, q_high, split=4):
    """
    低頻/高頻使用不同量化步階 (以 u+v 作為頻率分界)
    """
    key = (q_low, q_high, split)
    if key in _Q_MATRIX_CACHE:
        return _Q_MATRIX_CACHE[key]

    q_mat = np.zeros((8, 8), dtype=float)
    for u in range(8):
        for v in range(8):
            q_mat[u, v] = q_low if (u + v) <= split else q_high

    _Q_MATRIX_CACHE[key] = q_mat
    return q_mat

def quantize_band(dct_coeffs, q_low, q_high, split=4):
    """
    低頻/高頻不同 Q 的量化
    """
    q_mat = build_q_matrix(q_low, q_high, split)
    q_coeffs = np.round(dct_coeffs / q_mat)
    return q_coeffs.astype(int)

def dequantize_band(q_coeffs, q_low, q_high, split=4):
    """
    低頻/高頻不同 Q 的反量化
    """
    q_mat = build_q_matrix(q_low, q_high, split)
    return q_coeffs * q_mat

def test_codec_mvp(
    original_image,
    q_step=20,
    q_high=None,
    q_split=4,
    bit_depth=None,
    entropy_method='only_RLE',
    level_shift=None,
):
    q_low = q_step
    if q_high is None:
        q_high = q_step
    if bit_depth is None:
        bit_depth = np.iinfo(original_image.dtype).bits
    if level_shift is None:
        level_shift = 1 << (bit_depth - 1)
    # 1. Padding
    padded_img, orig_h, orig_w = pad_image(original_image)
    h, w = padded_img.shape
    
    # 準備容器
    reconstructed_img = np.zeros((h, w))
    all_blocks_rle = []
    prev_dc = 0
    
    # 2. 逐區塊處理 (Block Processing)
    # 這裡用雙層迴圈示範原理，之後可用 view_as_blocks 加速
    for r in range(0, h, 8):
        for c in range(0, w, 8):
            # 取出 8x8 區塊
            block = padded_img[r:r+8, c:c+8].astype(np.float64)
            if level_shift:
                block = block - level_shift
            
            # --- Encoder 端 ---
            # DCT 變換
            coeff = block_dct(block)
            # 量化 (這是主要失真來源)
            q_coeff = quantize_band(coeff, q_low, q_high, q_split)
            

            zigzag = zigzag_scan(q_coeff)
            dc = int(zigzag[0])
            dc_diff = dc - prev_dc
            prev_dc = dc

            rle = encode_block_rle_ac(zigzag)
            all_blocks_rle.append((dc_diff, rle))
            
            
            # --- Decoder 端 ---
            # 反量化
            rec_coeff = dequantize_band(q_coeff, q_low, q_high, q_split)
            # 反 DCT
            rec_block = block_idct(rec_coeff)
            if level_shift:
                rec_block = rec_block + level_shift
            
            # 放回大圖
            reconstructed_img[r:r+8, c:c+8] = rec_block

    estimated_bytes = estimate_compressed_size(all_blocks_rle)
    original_bytes = original_image.nbytes
    cr = original_bytes / estimated_bytes

    header_info = {
        'h': orig_h,
        'w': orig_w,
        'depth': bit_depth,
        'q_low': q_low,
        'q_high': q_high,
        'q_split': q_split,
        'level_shift': level_shift,
    }
    save_compressed_file("output.mic", header_info, all_blocks_rle, method=entropy_method)
    ##.mic (Medical Image Codec)
    
    print(f"Original Size: {original_bytes / 1024:.2f} KB")
    print(f"Estimated Compressed: {estimated_bytes / 1024:.2f} KB")
    print(f"Compression Ratio: {cr:.2f}:1")


    # 3. 裁切回原始大小並轉回正確型態 (Clip防止數值越界)
    final_img = reconstructed_img[:orig_h, :orig_w]
    final_img = np.clip(final_img, original_image.min(), original_image.max()) 
    
    return final_img

# 預先定義好的 ZigZag 索引表 (8x8 -> 64)
# 這能確保低頻係數在前，高頻(通常是0)在後
ZIGZAG_INDICES = np.array([
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
])

def zigzag_scan(block_8x8):
    """
    輸入: 8x8 量化後的區塊
    輸出: 64 元素的 1D 陣列
    """
    return block_8x8.ravel()[ZIGZAG_INDICES]


def encode_block_rle(zigzag_coeff):
    """
    將 64 個係數轉換為 RLE 符號列表
    格式: List of (run_zeros, value)
    """
    assert len(zigzag_coeff) == 64, f"Error: Block size is {len(zigzag_coeff)}, expected 64!"
    
    rle_symbols = []
    run_zeros = 0
    
    # 處理 DC (第一個係數通常單獨存，這裡簡化先一起做)
    # 實際 JPEG 會對 DC 做差分編碼 (DPCM)，作業若沒時間可略過
    
    for i in range(len(zigzag_coeff)):
        val = zigzag_coeff[i]
        
        if val == 0:
            run_zeros += 1
        else:
            # 遇到非 0，紀錄 (run, val) 並重置計數器
            rle_symbols.append((run_zeros, val))
            run_zeros = 0
            
    # 最後加上 EOB (End of Block) 標記
    # 只有當最後還有剩餘的 0 時才需要，避免解碼時錯位
    if run_zeros > 0:
        rle_symbols.append((0, 0)) 
    
    return rle_symbols

def encode_block_rle_ac(zigzag_coeff):
    """
    只針對 AC 係數做 RLE (跳過 DC)
    """
    rle_symbols = []
    run_zeros = 0

    for i in range(1, len(zigzag_coeff)):
        val = zigzag_coeff[i]
        if val == 0:
            run_zeros += 1
        else:
            rle_symbols.append((run_zeros, val))
            run_zeros = 0

    if run_zeros > 0:
        rle_symbols.append((0, 0))

    return rle_symbols

def decode_block_rle(rle_symbols):
    """
    將 RLE 符號還原回 64 個係數
    """
    coeff = np.zeros(64, dtype=int)
    idx = 0
    
    for run, val in rle_symbols:
        # 檢查是否為 EOB
        if run == 0 and val == 0:
            break # 後面全是 0，直接結束
            
        # 跳過 run 個 0
        idx += run
        
        # 填入數值
        if idx < 64:
            coeff[idx] = val
            idx += 1
            
    return coeff


def inverse_zigzag_scan(array_64):
    """
    輸入: 64 元素的 1D 陣列
    輸出: 8x8 區塊
    """
    block = np.zeros((8, 8), dtype=int)
    block.ravel()[ZIGZAG_INDICES] = array_64
    return block

def estimate_compressed_size(image_rle_data):
    """
    估算 bitstream 大小 (用於調試)
    假設簡單的編碼方式：
    - Run (計數): 用 4 bits (0-15)
    - Value (數值): 平均用 6 bits (這取決於數值大小)
    """
    total_bits = 0
    for dc_diff, block_symbols in image_rle_data:
        dc_len = abs(int(dc_diff)).bit_length()
        if dc_len > 0:
            total_bits += (4 + 1 + dc_len)  # DC length + sign + value
        else:
            total_bits += 4

        for run, val in block_symbols:
            # 每個 symbol 大約花費:
            # run: 4 bits
            # value: 變動長度 (數值越大 bit 越多)
            if val == 0: # EOB
                bits_for_val = 0
            else:
                val_native = int(val) 
                bits_for_val = abs(val_native).bit_length() + 1 # +1 是為了存正負號 (Sign bit)
                
            total_bits += (4 + bits_for_val)

            
    return total_bits / 8 # 回傳 Bytes



if __name__ == "__main__":
    # --- 執行測試 ---
    # 假設 raw_img 是您上一步讀出來的 CT 數據
    filepath = "/ssd7/jiakai/multimedia_hw2/CT_COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0001/01-01-2000-1-Abdomen24ACRINColoIRB2415-04 Adult-0.4.1/3.000000-Colosupine  1.0  B30f-4.563/1-010.dcm"
    raw_img, header = analyze_dicom_file(filepath)
    bit_depth = getattr(header, 'BitsStored', np.iinfo(raw_img.dtype).bits)
    level_shift = 1 << (bit_depth - 1)
    decoded_img = test_codec_mvp(
        raw_img,
        q_step=50,
        q_high=50,
        q_split=4,
        bit_depth=bit_depth,
        level_shift=level_shift,
    )

    # 計算 RMSE
    mse = np.mean((raw_img - decoded_img) ** 2)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")
