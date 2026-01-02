import struct

from entropy_coding import EntropyCoder

class BitWriter:
    def __init__(self):
        self.buffer = 0       # 暫存器 (整數)
        self.count = 0        # 目前暫存了幾個 bit
        self.bytes_list = bytearray() # 最終輸出的 byte 陣列

    def write(self, value, num_bits):
        """
        寫入 num_bits 個位元
        例如: write(3, 4) -> 寫入二進制 '0011'
        """
        # 遮罩處理，確保不會寫入超過 num_bits 的數據
        value = value & ((1 << num_bits) - 1)
        
        # 把新數據推入暫存器 (左移)
        self.buffer = (self.buffer << num_bits) | value
        self.count += num_bits
        
        # 只要湊滿 8 bits，就切下來存入 bytes_list
        while self.count >= 8:
            self.count -= 8
            # 取出最左邊的 8 bits
            byte_val = (self.buffer >> self.count) & 0xFF
            self.bytes_list.append(byte_val)

    def flush(self):
        """
        結束時呼叫，把剩下不足 8 bits 的補 0 寫出
        """
        if self.count > 0:
            self.buffer = self.buffer << (8 - self.count)
            self.bytes_list.append(self.buffer & 0xFF)
            self.count = 0
            self.buffer = 0
            
    def get_bytes(self):
        return self.bytes_list


class BitReader:
    def __init__(self, bytes_data):
        self.data = bytes_data
        self.byte_idx = 0     # 目前讀到第幾個 byte
        self.buffer = 0       # 暫存區
        self.count = 0        # 暫存區剩幾個 bit

    def read(self, num_bits):
        """
        讀取 num_bits 個位元，回傳整數
        """
        # 如果暫存區的 bit 不夠，就從 data 載入新的 byte
        while self.count < num_bits:
            if self.byte_idx >= len(self.data):
                # 檔案讀完了，補 0 (或拋出錯誤)
                self.buffer = self.buffer << 8
                self.count += 8
            else:
                # 載入下一個 byte
                self.buffer = (self.buffer << 8) | self.data[self.byte_idx]
                self.byte_idx += 1
                self.count += 8

        # 從暫存區取出最左邊的 num_bits
        self.count -= num_bits
        result = (self.buffer >> self.count) & ((1 << num_bits) - 1)
        return result

def save_compressed_file(filepath, header_info, rle_data, method='only_RLE'):
    writer = BitWriter()
    
    # 1. 寫入 Header (使用 struct)
    # Magic(4s) + Ver(B) + H(H) + W(H) + Depth(B) + Q_Step(B)
    # H = unsigned short (2 bytes), B = unsigned char (1 byte)
    method_alias = EntropyCoder.normalize_method(method)
    method_map = {
        'only_RLE': 0,
        'huffman_std': 1,
        'huffman_adapt': 2,
    }

    if 'q_high' in header_info or 'q_low' in header_info or 'q_split' in header_info:
        q_low = header_info.get('q_low', header_info.get('q_step', 0))
        q_high = header_info.get('q_high', q_low)
        q_split = header_info.get('q_split', 7)
    else:
        q_low = header_info.get('q_step', 0)
        q_high = q_low
        q_split = 7
    level_shift = header_info.get('level_shift', 0)

    if method_alias not in method_map:
        raise ValueError(f"Unknown method: {method_alias}")

    coder = EntropyCoder.create_for_encoding(method_alias, rle_data=rle_data)
    method_alias = coder.method
    if method_alias not in method_map:
        raise ValueError(f"Unknown method after fallback: {method_alias}")

    method_id = method_map[method_alias]

    header_bytes = struct.pack('>4sBHHBBBBBH', 
                               b'MIPC',      # Magic
                               5,            # Version
                               header_info['h'], 
                               header_info['w'], 
                               header_info['depth'], 
                               q_low,
                               q_high,
                               q_split,
                               method_id,
                               level_shift)
    if method_id == 2:
        header_bytes += bytes(coder.lengths_ac)
        header_bytes += bytes(coder.lengths_dc)
    
    # 2. 寫入 Body (壓縮數據)
    coder.pack_rle_to_bits(rle_data, writer)
    writer.flush()
    
    # 3. 存檔
    with open(filepath, 'wb') as f:
        f.write(header_bytes)
        f.write(writer.get_bytes())
        
    print(f"Saved: {filepath}")
    print(f"Header Size: {len(header_bytes)} bytes")
    print(f"Body Size: {len(writer.get_bytes())} bytes")
