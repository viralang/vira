#!/usr/bin/env python3
"""
Generates a valid Hugging Face SafeTensors weight file (.safetensors)
without requiring any external dependencies (using pure Python standard library: struct & json).
"""

import json
import os
import struct

def export_safetensors():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.safetensors")

    # 1. Prepare raw tensor buffer: 4x4 matrix of float32 values (0.75)
    rows = 4
    cols = 4
    floats = [0.75 * (i + 1) for i in range(rows * cols)]
    tensor_bytes = struct.pack(f"{len(floats)}f", *floats)
    total_bytes = len(tensor_bytes)

    # 2. Prepare header metadata JSON
    header = {
        "fc1.weight": {
            "dtype": "F32",
            "shape": [rows, cols],
            "data_offsets": [0, total_bytes]
        },
        "__metadata__": {
            "format": "vex_pt",
            "author": "Vex AI Framework"
        }
    }
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_len = len(header_json)

    # 3. Write file: 8 bytes uint64 header length + JSON + binary tensor buffer
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(tensor_bytes)

    print(f"✓ Exported SafeTensors file: {out_path}")
    print(f"  • Header length: {header_len} bytes")
    print(f"  • Tensor 'fc1.weight': shape [{rows}, {cols}], size {total_bytes} bytes")
    print(f"  • First float: {floats[0]:.2f}, Last float: {floats[-1]:.2f}")

if __name__ == "__main__":
    export_safetensors()
