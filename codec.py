import argparse

from decode import decode_cmd
from encode import encode_cmd


def main():
    parser = argparse.ArgumentParser(description="MIC encoder/decoder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enc = subparsers.add_parser("encode", help="Encode DICOM to .mic")
    enc.add_argument("--input", required=True, help="Path to DICOM file")
    enc.add_argument("--output", required=True, help="Output .mic path")
    enc.add_argument(
        "--quality",
        type=int,
        required=True,
        help="Quantization step (larger = more compression, lower quality)",
    )
    enc.add_argument("--q-split", type=int, default=4, help="Low/high split (u+v)")
    enc.add_argument("--window-center", type=float, default=40, help="Window center")
    enc.add_argument("--window-width", type=float, default=400, help="Window width")
    enc.set_defaults(func=encode_cmd)

    dec = subparsers.add_parser("decode", help="Decode .mic to DICOM")
    dec.add_argument("--input", required=True, help="Path to .mic file")
    dec.add_argument("--output", required=True, help="Output DICOM path (.dcm)")
    dec.add_argument(
        "--dicom",
        default="",
        help="Optional reference DICOM to preserve metadata",
    )
    dec.set_defaults(func=decode_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
