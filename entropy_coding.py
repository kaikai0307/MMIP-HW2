import heapq

MEDICAL_HUFFMAN_AC_LENGTHS = [
    5, 2, 3, 3, 4, 5, 7, 8, 9, 10, 16, 24, 0, 0, 0, 0,
    0, 3, 5, 6, 6, 8, 9, 10, 12, 15, 23, 0, 0, 0, 0, 0,
    0, 4, 6, 7, 8, 9, 11, 11, 13, 16, 24, 0, 0, 0, 0, 0,
    0, 6, 7, 9, 10, 11, 13, 14, 15, 18, 0, 0, 0, 0, 0, 0,
    0, 6, 8, 9, 10, 12, 14, 15, 18, 21, 0, 0, 0, 0, 0, 0,
    0, 7, 9, 11, 13, 15, 17, 19, 20, 0, 0, 0, 0, 0, 0, 0,
    0, 8, 9, 11, 14, 16, 20, 22, 23, 0, 0, 0, 0, 0, 0, 0,
    0, 9, 11, 14, 17, 20, 22, 24, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 10, 13, 16, 19, 22, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 11, 14, 18, 21, 22, 0, 24, 23, 0, 0, 0, 0, 0, 0, 0,
    0, 11, 14, 18, 19, 24, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 12, 15, 20, 22, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 12, 15, 22, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 12, 14, 19, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 13, 17, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    12, 13, 17, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

MEDICAL_HUFFMAN_DC_LENGTHS = [
    4, 4, 3, 3, 3, 3, 3, 3, 4, 5, 6, 6, 0, 0, 0, 0,
]



def _build_huffman_code_lengths(freqs):
    heap = []
    counter = 0
    for sym, freq in enumerate(freqs):
        if freq > 0:
            heapq.heappush(heap, (freq, counter, sym))
            counter += 1

    size = len(freqs)
    if not heap:
        return [0] * size
    if len(heap) == 1:
        lengths = [0] * size
        lengths[heap[0][2]] = 1
        return lengths

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        node = (n1, n2)
        heapq.heappush(heap, (f1 + f2, counter, node))
        counter += 1

    root = heap[0][2]
    lengths = [0] * size

    def assign_lengths(node, depth):
        if isinstance(node, int):
            lengths[node] = max(depth, 1)
            return
        left, right = node
        assign_lengths(left, depth + 1)
        assign_lengths(right, depth + 1)

    assign_lengths(root, 0)
    return lengths


def _build_canonical_codes(lengths):
    symbols = [(length, sym) for sym, length in enumerate(lengths) if length > 0]
    symbols.sort()
    codes = {}
    code = 0
    prev_len = 0
    for length, sym in symbols:
        code <<= (length - prev_len)
        codes[sym] = (code, length)
        code += 1
        prev_len = length
    return codes


MEDICAL_HUFFMAN_AC_CODES = _build_canonical_codes(MEDICAL_HUFFMAN_AC_LENGTHS)
MEDICAL_HUFFMAN_DC_CODES = _build_canonical_codes(MEDICAL_HUFFMAN_DC_LENGTHS)


def _build_huffman_from_freqs(freqs):
    lengths = _build_huffman_code_lengths(freqs)
    codes = _build_canonical_codes(lengths)
    return codes, lengths


def _build_decode_table(codes):
    table = {}
    max_len = 0
    for sym, (code, length) in codes.items():
        table[(length, code)] = sym
        if length > max_len:
            max_len = length
    return table, max_len


def _read_huffman_symbol(reader, decode_table, max_len):
    code = 0
    for length in range(1, max_len + 1):
        bit = reader.read(1)
        code = (code << 1) | bit
        symbol = decode_table.get((length, code))
        if symbol is not None:
            return symbol
    raise ValueError("Invalid Huffman code")


def _collect_ac_dc_freqs(all_blocks_rle):
    ac_freqs = [0] * 256
    dc_freqs = [0] * 16
    for dc_diff, block_symbols in all_blocks_rle:
        dc_len = abs(int(dc_diff)).bit_length()
        if dc_len > 15:
            dc_len = 15
        dc_freqs[dc_len] += 1

        for run, val in block_symbols:
            if run == 0 and val == 0:
                ac_freqs[0x00] += 1
                continue

            while run > 15:
                ac_freqs[0xF0] += 1
                run -= 16

            length = abs(int(val)).bit_length()
            if length > 15:
                length = 15
            symbol = (run << 4) | length
            ac_freqs[symbol] += 1
    return ac_freqs, dc_freqs


class EntropyCoder:
    def __init__(self, method="only_RLE", codes=None, lengths=None):
        self.method = self.normalize_method(method)
        self.codes_ac = None
        self.codes_dc = None
        self.lengths_ac = None
        self.lengths_dc = None
        self.decode_table_ac = None
        self.decode_table_dc = None
        self.max_len_ac = None
        self.max_len_dc = None
        self.prev_dc = 0

        if codes is not None:
            self.codes_ac = codes.get("ac")
            self.codes_dc = codes.get("dc")
        if lengths is not None:
            self.lengths_ac = lengths.get("ac")
            self.lengths_dc = lengths.get("dc")

        if self.method == "huffman_std":
            if self.codes_ac is None:
                self.codes_ac = MEDICAL_HUFFMAN_AC_CODES
            if self.codes_dc is None:
                self.codes_dc = MEDICAL_HUFFMAN_DC_CODES
            self.decode_table_ac, self.max_len_ac = _build_decode_table(self.codes_ac)
            self.decode_table_dc, self.max_len_dc = _build_decode_table(self.codes_dc)
        elif self.method == "huffman_adapt":
            if self.codes_ac is None and self.lengths_ac is not None:
                self.codes_ac = _build_canonical_codes(self.lengths_ac)
            if self.codes_dc is None and self.lengths_dc is not None:
                self.codes_dc = _build_canonical_codes(self.lengths_dc)
            if self.codes_ac is not None:
                self.decode_table_ac, self.max_len_ac = _build_decode_table(self.codes_ac)
            if self.codes_dc is not None:
                self.decode_table_dc, self.max_len_dc = _build_decode_table(self.codes_dc)

    @staticmethod
    def normalize_method(method):
        if method == "huffman":
            return "huffman_std"
        return method

    @classmethod
    def create_for_encoding(cls, method, rle_data=None):
        method = cls.normalize_method(method)
        if method == "only_RLE":
            return cls(method=method)
        if method == "huffman_std":
            if rle_data is not None:
                ac_freqs, dc_freqs = _collect_ac_dc_freqs(rle_data)
                unsupported_ac = [
                    sym for sym, freq in enumerate(ac_freqs)
                    if freq > 0 and sym not in MEDICAL_HUFFMAN_AC_CODES
                ]
                unsupported_dc = [
                    cat for cat, freq in enumerate(dc_freqs)
                    if freq > 0 and cat not in MEDICAL_HUFFMAN_DC_CODES
                ]
                if unsupported_ac or unsupported_dc:
                    max_len_ac = max((sym & 0x0F) for sym in unsupported_ac) if unsupported_ac else 0
                    max_len_dc = max(unsupported_dc) if unsupported_dc else 0
                    print(
                        "Warning: Huffman std table cannot encode "
                        f"AC size={max_len_ac} or DC size={max_len_dc}, "
                        "falling back to huffman_adapt."
                    )
                    ac_codes, ac_lengths = _build_huffman_from_freqs(ac_freqs)
                    dc_codes, dc_lengths = _build_huffman_from_freqs(dc_freqs)
                    return cls(
                        method="huffman_adapt",
                        codes={"ac": ac_codes, "dc": dc_codes},
                        lengths={"ac": ac_lengths, "dc": dc_lengths},
                    )
            return cls(
                method=method,
                codes={"ac": MEDICAL_HUFFMAN_AC_CODES, "dc": MEDICAL_HUFFMAN_DC_CODES},
            )
        if method == "huffman_adapt":
            if rle_data is None:
                raise ValueError("rle_data is required for huffman_adapt")
            ac_freqs, dc_freqs = _collect_ac_dc_freqs(rle_data)
            ac_codes, ac_lengths = _build_huffman_from_freqs(ac_freqs)
            dc_codes, dc_lengths = _build_huffman_from_freqs(dc_freqs)
            return cls(
                method=method,
                codes={"ac": ac_codes, "dc": dc_codes},
                lengths={"ac": ac_lengths, "dc": dc_lengths},
            )
        raise ValueError(f"Unknown method: {method}")

    @classmethod
    def create_for_decoding(cls, method, lengths=None):
        method = cls.normalize_method(method)
        if method == "only_RLE":
            return cls(method=method)
        if method == "huffman_std":
            return cls(
                method=method,
                codes={"ac": MEDICAL_HUFFMAN_AC_CODES, "dc": MEDICAL_HUFFMAN_DC_CODES},
            )
        if method == "huffman_adapt":
            if lengths is None:
                raise ValueError("lengths are required for huffman_adapt")
            ac_lengths = lengths.get("ac")
            dc_lengths = lengths.get("dc")
            ac_codes = _build_canonical_codes(ac_lengths)
            dc_codes = _build_canonical_codes(dc_lengths)
            return cls(
                method=method,
                codes={"ac": ac_codes, "dc": dc_codes},
                lengths={"ac": ac_lengths, "dc": dc_lengths},
            )
        raise ValueError(f"Unknown method: {method}")

    def reset_state(self):
        self.prev_dc = 0

    def pack_rle_to_bits(self, all_blocks_rle, writer):
        for dc_diff, block_symbols in all_blocks_rle:
            self._write_dc(dc_diff, writer)
            self._write_ac(block_symbols, writer)

    def _write_dc(self, dc_diff, writer):
        val_int = int(dc_diff)
        abs_val = abs(val_int)
        length = abs_val.bit_length()
        sign = 1 if val_int < 0 else 0

        if self.method == "only_RLE":
            if length > 15:
                raise ValueError(f"DC length too large for 4-bit field: {length}")
            writer.write(length, 4)
            if length > 0:
                writer.write(sign, 1)
                writer.write(abs_val, length)
            return

        if self.codes_dc is None:
            raise ValueError("DC Huffman codes not initialized")
        if length not in self.codes_dc:
            raise ValueError(f"DC length not in Huffman table: {length}")
        code, clen = self.codes_dc[length]
        writer.write(code, clen)
        if length > 0:
            writer.write(sign, 1)
            writer.write(abs_val, length)

    def _write_ac(self, block_symbols, writer):
        for run, val in block_symbols:
            if self.method == "only_RLE":
                if run == 0 and val == 0:  # EOB
                    writer.write(0, 4)
                    writer.write(0, 4)
                    continue

                while run > 15:
                    writer.write(15, 4)
                    writer.write(0, 4)
                    run -= 16

                val_int = int(val)
                sign = 1 if val_int < 0 else 0
                abs_val = abs(val_int)
                length = abs_val.bit_length()
                if length > 15:
                    raise ValueError(f"AC length too large for 4-bit field: {length}")

                writer.write(run, 4)
                writer.write(length, 4)
                writer.write(sign, 1)
                writer.write(abs_val, length)
            elif self.method in ("huffman_std", "huffman_adapt"):
                if self.codes_ac is None:
                    raise ValueError("AC Huffman codes not initialized")

                if run == 0 and val == 0:
                    code, length = self.codes_ac[0x00]
                    writer.write(code, length)
                    continue

                while run > 15:
                    code, length = self.codes_ac[0xF0]
                    writer.write(code, length)
                    run -= 16

                val_int = int(val)
                sign = 1 if val_int < 0 else 0
                abs_val = abs(val_int)
                value_len = abs_val.bit_length()
                if value_len > 15:
                    raise ValueError(f"AC length too large for 4-bit field: {value_len}")

                symbol = (run << 4) | value_len
                code, length = self.codes_ac[symbol]
                writer.write(code, length)
                writer.write(sign, 1)
                writer.write(abs_val, value_len)
            else:
                raise ValueError(f"Unknown method: {self.method}")

    def decode_block(self, reader):
        block_coeffs = [0] * 64
        dc_diff = self._read_dc(reader)
        dc_val = self.prev_dc + dc_diff
        self.prev_dc = dc_val
        block_coeffs[0] = dc_val

        idx = 1
        if self.method == "only_RLE":
            while idx < 64:
                run = reader.read(4)
                length = reader.read(4)

                if run == 0 and length == 0:
                    break
                if run == 15 and length == 0:
                    idx += 16
                    continue

                idx += run
                if idx >= 64:
                    break

                sign = reader.read(1)
                abs_val = reader.read(length) if length > 0 else 0
                val = -abs_val if sign == 1 else abs_val
                block_coeffs[idx] = val
                idx += 1
        else:
            if self.decode_table_ac is None or self.max_len_ac is None:
                raise ValueError("AC Huffman decode table not initialized")

            while idx < 64:
                symbol = _read_huffman_symbol(reader, self.decode_table_ac, self.max_len_ac)
                if symbol == 0x00:
                    break
                if symbol == 0xF0:
                    idx += 16
                    continue

                run = symbol >> 4
                length = symbol & 0x0F

                idx += run
                if idx >= 64:
                    break

                sign = reader.read(1)
                abs_val = reader.read(length) if length > 0 else 0
                val = -abs_val if sign == 1 else abs_val
                block_coeffs[idx] = val
                idx += 1

        return block_coeffs

    def _read_dc(self, reader):
        if self.method == "only_RLE":
            length = reader.read(4)
            if length == 0:
                return 0
            sign = reader.read(1)
            abs_val = reader.read(length) if length > 0 else 0
            return -abs_val if sign == 1 else abs_val

        if self.decode_table_dc is None or self.max_len_dc is None:
            raise ValueError("DC Huffman decode table not initialized")

        length = _read_huffman_symbol(reader, self.decode_table_dc, self.max_len_dc)
        if length == 0:
            return 0
        sign = reader.read(1)
        abs_val = reader.read(length) if length > 0 else 0
        return -abs_val if sign == 1 else abs_val
