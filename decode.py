# decode.py (建議新建這個檔案)
import argparse
import struct
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from bitstream import BitReader
from encode import inverse_zigzag_scan, dequantize_band, block_idct
from entropy_coding import EntropyCoder
from readdcm import analyze_dicom_file


def _infer_pixel_repr(level_shift, array):
    if level_shift == 0:
        return 1
    if np.min(array) < 0:
        return 1
    return 0


def _build_minimal_dicom(rows, cols, header_info):
    meta = FileMetaDataset()
    meta.FileMetaInformationVersion = b"\x00\x01"
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OT"
    ds.PatientName = "Anon"
    ds.PatientID = "000000"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols

    bit_depth = int(header_info.get("depth", 16))
    bits_allocated = 16 if bit_depth > 8 else 8
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bit_depth
    ds.HighBit = bit_depth - 1
    pixel_repr = _infer_pixel_repr(header_info.get("level_shift", 0), np.zeros((1, 1)))
    ds.PixelRepresentation = pixel_repr

    ds.RescaleSlope = float(header_info.get("rescale_slope", 1.0))
    ds.RescaleIntercept = float(header_info.get("rescale_intercept", 0.0))
    ds.WindowCenter = float(header_info.get("window_center", 40.0))
    ds.WindowWidth = float(header_info.get("window_width", 400.0))
    return ds


def save_recon_dicom(path, array, header_info, template_path=None):
    if template_path:
        ds = pydicom.dcmread(template_path)
    else:
        rows, cols = array.shape
        ds = _build_minimal_dicom(rows, cols, header_info)
    rows, cols = array.shape
    ds.Rows = rows
    ds.Columns = cols

    bit_depth = int(getattr(ds, "BitsStored", 16))
    pixel_repr = int(getattr(ds, "PixelRepresentation", _infer_pixel_repr(header_info.get("level_shift", 0), array)))
    if pixel_repr == 1:
        min_val = -(1 << (bit_depth - 1))
        max_val = (1 << (bit_depth - 1)) - 1
        dtype = np.int16 if getattr(ds, "BitsAllocated", 16) > 8 else np.int8
    else:
        min_val = 0
        max_val = (1 << bit_depth) - 1
        dtype = np.uint16 if getattr(ds, "BitsAllocated", 16) > 8 else np.uint8

    recon = np.rint(array).astype(np.int64)
    recon = np.clip(recon, min_val, max_val).astype(dtype)
    ds.PixelData = recon.tobytes()

    new_uid = generate_uid()
    ds.SOPInstanceUID = new_uid
    if hasattr(ds, "file_meta") and ds.file_meta is not None:
        ds.file_meta.MediaStorageSOPInstanceUID = new_uid

    ds.save_as(path)
    return path

def load_compressed_file(filepath, return_header=False):
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
        
    # 1. 解析 Header (根據您 encode 時寫入的格式)
    # v6: '>4sBHHBBBBBHffff' = Magic(4), Ver(1), H(2), W(2), Depth(1), Qlow(1), Qhigh(1), Qsplit(1), Method(1), LevelShift(2),
    #       RescaleSlope(4), RescaleIntercept(4), WindowCenter(4), WindowWidth(4)
    header_prefix_size = struct.calcsize('>4sB')
    magic, ver = struct.unpack('>4sB', file_bytes[:header_prefix_size])
    if magic != b'MIPC':
        raise ValueError("Invalid file format!")

    method = 'only_RLE'
    coder = None

    if ver >= 6:
        header_size = struct.calcsize('>4sBHHBBBBBHffff')
        header_data = file_bytes[:header_size]
        (
            magic,
            ver,
            h,
            w,
            depth,
            q_low,
            q_high,
            q_split,
            method_id,
            level_shift,
            rescale_slope,
            rescale_intercept,
            window_center,
            window_width,
        ) = struct.unpack('>4sBHHBBBBBHffff', header_data)
    elif ver == 5:
        header_size = struct.calcsize('>4sBHHBBBBBH')
        header_data = file_bytes[:header_size]
        magic, ver, h, w, depth, q_low, q_high, q_split, method_id, level_shift = struct.unpack('>4sBHHBBBBBH', header_data)
        rescale_slope = None
        rescale_intercept = None
        window_center = None
        window_width = None
    elif ver == 4:
        header_size = struct.calcsize('>4sBHHBBBBB')
        header_data = file_bytes[:header_size]
        magic, ver, h, w, depth, q_low, q_high, q_split, method_id = struct.unpack('>4sBHHBBBBB', header_data)
        level_shift = 0
        rescale_slope = None
        rescale_intercept = None
        window_center = None
        window_width = None
    else:
        raise ValueError("Unsupported file version (expected v4/v5/v6). Please re-encode.")

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
            "rescale_slope": rescale_slope,
            "rescale_intercept": rescale_intercept,
            "window_center": window_center,
            "window_width": window_width,
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


def decode_cmd(args):
    img_recon, header_info = load_compressed_file(args.input, return_header=True)
    output_path = args.output
    if not output_path.lower().endswith(".dcm"):
        output_path = f"{output_path}.dcm"
    save_recon_dicom(output_path, img_recon, header_info, args.dicom or None)
    print(f"Saved decoded DICOM: {output_path}")

    if args.dicom:
        raw_img, header = analyze_dicom_file(args.dicom)
        rmse, psnr = compute_rmse_psnr(raw_img, img_recon, header_info["depth"])
        print(f"RMSE: {rmse:.4f}")
        print(f"PSNR: {psnr:.2f} dB (MAX=2^{header_info['depth']}-1)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode .mic to DICOM")
    parser.add_argument("--input", default="output.mic", help="Path to .mic file")
    parser.add_argument("--output", default="recon.dcm", help="Output DICOM path")
    parser.add_argument("--dicom", default="", help="Optional reference DICOM for metadata/RMSE")
    args = parser.parse_args()
    decode_cmd(args)
