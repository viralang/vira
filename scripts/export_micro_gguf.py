#!/usr/bin/env python3
"""
Generates a real, valid GGUF v3 binary model file for testing Vex's native GGUF parser
and local LLM inference engine without requiring llama.cpp or any external C/Python libraries.
"""

import os
import struct

def write_gguf_string(f, s):
    data = s.encode('utf-8')
    f.write(struct.pack('<Q', len(data)))
    f.write(data)

def write_gguf_kv(f, key, val_type, pack_fmt, *val):
    write_gguf_string(f, key)
    f.write(struct.pack('<I', val_type))
    f.write(struct.pack(pack_fmt, *val))

def export_gguf():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "micro_llama.gguf")

    vocab_size = 8
    hidden_dim = 16
    alignment = 32

    # Prepare tensor list: (name, shape, data_bytes, dtype)
    # GGUF convention: shape is [cols, rows] (fastest dimension first)
    tensors = []

    def make_weight(name, rows, cols, val=0.1):
        total = rows * cols
        floats = [val * ((i % 7) + 1) for i in range(total)]
        raw = struct.pack(f'<{total}f', *floats)
        # In GGUF: shape = [cols, rows]
        tensors.append((name, [cols, rows], raw, 0)) # dtype 0 = F32

    make_weight("token_embd.weight", vocab_size, hidden_dim, 0.25)
    make_weight("blk.0.attn_norm.weight", 1, hidden_dim, 1.0)
    make_weight("blk.0.attn_q.weight", hidden_dim, hidden_dim, 0.15)
    make_weight("blk.0.attn_k.weight", hidden_dim, hidden_dim, 0.15)
    make_weight("blk.0.attn_v.weight", hidden_dim, hidden_dim, 0.20)
    make_weight("blk.0.attn_output.weight", hidden_dim, hidden_dim, 0.10)
    make_weight("blk.0.ffn_norm.weight", 1, hidden_dim, 1.0)
    make_weight("blk.0.ffn_gate.weight", hidden_dim, hidden_dim * 2, 0.08)
    make_weight("blk.0.ffn_up.weight", hidden_dim, hidden_dim * 2, 0.08)
    make_weight("blk.0.ffn_down.weight", hidden_dim * 2, hidden_dim, 0.05)
    make_weight("output_norm.weight", 1, hidden_dim, 1.0)
    make_weight("output.weight", hidden_dim, vocab_size, 0.30)

    # Calculate offsets
    tensor_offsets = []
    curr_off = 0
    for name, shape, data, dtype in tensors:
        tensor_offsets.append(curr_off)
        size = len(data)
        pad = (alignment - (size % alignment)) % alignment
        curr_off += size + pad

    with open(out_path, "wb") as f:
        # 1. Header: magic (4), version (4), tensor_count (8), kv_count (8)
        f.write(b"GGUF")
        f.write(struct.pack('<I', 3)) # Version 3
        f.write(struct.pack('<Q', len(tensors)))
        f.write(struct.pack('<Q', 4)) # 4 metadata KV pairs

        # 2. Metadata KV Pairs
        # general.architecture = "llama" (val_type 8 = STRING)
        write_gguf_string(f, "general.architecture")
        f.write(struct.pack('<I', 8))
        write_gguf_string(f, "llama")

        # general.alignment = 32 (val_type 4 = UINT32)
        write_gguf_kv(f, "general.alignment", 4, '<I', alignment)

        # llama.embedding_length = 16 (val_type 4 = UINT32)
        write_gguf_kv(f, "llama.embedding_length", 4, '<I', hidden_dim)

        # llama.vocab_size = 8 (val_type 4 = UINT32)
        write_gguf_kv(f, "llama.vocab_size", 4, '<I', vocab_size)

        # 3. Tensor Info Table
        for (name, shape, data, dtype), off in zip(tensors, tensor_offsets):
            write_gguf_string(f, name)
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', dtype))
            f.write(struct.pack('<Q', off))

        # 4. Alignment padding before data buffer
        cur_pos = f.tell()
        pad = (alignment - (cur_pos % alignment)) % alignment
        if pad > 0:
            f.write(b"\x00" * pad)

        # 5. Tensor Binary Data Buffers
        for name, shape, data, dtype in tensors:
            f.write(data)
            pad = (alignment - (len(data) % alignment)) % alignment
            if pad > 0:
                f.write(b"\x00" * pad)

    print(f"✓ Exported real GGUF v3 model file: {out_path}")
    print(f"  • Architecture: llama (vocab={vocab_size}, hidden_dim={hidden_dim})")
    print(f"  • Tensors count: {len(tensors)}")
    print(f"  • File size: {os.path.getsize(out_path)} bytes")

if __name__ == "__main__":
    export_gguf()
