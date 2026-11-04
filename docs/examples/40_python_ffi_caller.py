#!/usr/bin/env python3
"""
Vira v0.7 Native C & Python FFI Companion Script
------------------------------------------------
This script demonstrates how to load and invoke a high-performance compiled
Vira shared library (.dll / .so / .dylib) directly from Python using ctypes.

Prerequisites:
  Build the shared library from Vira source:
    vira build -shared -o build/vira_ffi_demo.dll docs/examples/40_native_c_and_python_ffi.vir
"""

import os
import sys
import ctypes
import platform
import subprocess

def get_shared_lib_filename(base_name="vira_ffi_demo"):
    system = platform.system()
    if system == "Windows":
        return f"{base_name}.dll"
    elif system == "Darwin":
        return f"lib{base_name}.dylib"
    else:
        return f"lib{base_name}.so"

def main():
    print("=== Python FFI Caller for Vira Native Shared Library ===")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    build_dir = os.path.join(workspace_root, "build")
    os.makedirs(build_dir, exist_ok=True)

    lib_filename = get_shared_lib_filename("vira_ffi_demo")
    lib_path = os.path.join(build_dir, lib_filename)
    vir_src = os.path.join(script_dir, "40_native_c_and_python_ffi.vir")

    # If the shared library does not exist yet, build it using vira CLI
    if not os.path.exists(lib_path):
        print(f"[Python FFI] Shared library not found at: {lib_path}")
        print(f"[Python FFI] Compiling Vira source into shared library...")
        vira_bin = os.path.join(workspace_root, "vira.exe" if platform.system() == "Windows" else "vira")
        cmd = [vira_bin, "build", "-shared", "-o", lib_path, vir_src]
        print(f"[Python FFI] Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=workspace_root)
        if res.returncode != 0:
            print("[Python FFI] Failed to compile Vira shared library!", file=sys.stderr)
            sys.exit(1)
        print(f"[Python FFI] Successfully compiled shared library.")

    # 1. Load shared library via ctypes
    print(f"[Python FFI] Loading shared library: {lib_path}")
    lib = ctypes.CDLL(lib_path)

    # 2. Bind `vira_add_integers(c_int, c_int) -> c_int`
    lib.vira_add_integers.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.vira_add_integers.restype = ctypes.c_int

    a, b = 15, 27
    sum_result = lib.vira_add_integers(a, b)
    print(f"[Python FFI] vira_add_integers({a}, {b}) = {sum_result}")
    assert sum_result == 42, f"Expected 42, got {sum_result}"

    # 3. Bind `vira_vector_norm(c_double, c_double) -> c_double`
    lib.vira_vector_norm.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.vira_vector_norm.restype = ctypes.c_double

    x, y = 3.0, 4.0
    norm_result = lib.vira_vector_norm(x, y)
    print(f"[Python FFI] vira_vector_norm({x}, {y}) = {norm_result}")
    assert abs(norm_result - 5.0) < 1e-6, f"Expected 5.0, got {norm_result}"

    # Another vector test (6.0, 8.0 -> 10.0)
    x2, y2 = 6.0, 8.0
    norm_result2 = lib.vira_vector_norm(x2, y2)
    print(f"[Python FFI] vira_vector_norm({x2}, {y2}) = {norm_result2}")
    assert abs(norm_result2 - 10.0) < 1e-6, f"Expected 10.0, got {norm_result2}"

    # 4. Demonstrate C-compatible struct mapping
    class NativeVec2(ctypes.Structure):
        _fields_ = [
            ("x", ctypes.c_double),
            ("y", ctypes.c_double),
        ]

    pt = NativeVec2(12.0, 5.0)
    pt_norm = lib.vira_vector_norm(pt.x, pt.y)
    print(f"[Python FFI] NativeVec2({pt.x}, {pt.y}) norm = {pt_norm}")
    assert abs(pt_norm - 13.0) < 1e-6, f"Expected 13.0, got {pt_norm}"

    print("=== All Python ctypes FFI assertions PASSED successfully! ===")

if __name__ == "__main__":
    main()

