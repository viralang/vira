"""Checks the kernels' index arithmetic on the CPU (no GPU needed).

    python3 runtime/cuda/test_emulate.py

Compiles the .cu sources as C++ with emulate.h and runs them on Q8_1 activations and
random weight blocks. Q8_0 is compared with a plain sum over the quantized values; every
format (Q8_0 included) is compared with the runtime's CPU dequantiser
(_vira_rt_dequant_blocks_to_fp32, checked against llama.cpp by ViraLama's reference
tests), so the kernels are not tested against themselves.
"""
import ctypes, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


RUNTIME = os.path.dirname(HERE)
# name, GGUF dtype, bytes per block, values per block, f16 fields (byte offsets)
FORMATS = [
    ('q4_0', 2, 18, 32, [0]),
    ('q4_1', 3, 20, 32, [0, 2]),
    ('q5_0', 6, 22, 32, [0]),
    ('q5_1', 7, 24, 32, [0, 2]),
    ('q8_0', 8, 34, 32, [0]),
    ('q4_k', 12, 144, 256, [0, 2]),
    ('q5_k', 13, 176, 256, [0, 2]),
    ('q6_k', 14, 210, 256, [208]),
]


def build_oracle():
    d = tempfile.mkdtemp()
    src = os.path.join(d, 'oracle.c')
    open(src, 'w').write('#include "vira_rt.h"\n'
                         'void oracle(unsigned dt, const unsigned char *raw, size_t nb, float *out)'
                         ' { _vira_rt_dequant_blocks_to_fp32(dt, raw, 0, nb, out); }\n')
    out = os.path.join(d, 'oracle.so')
    subprocess.check_call(['gcc', '-O1', '-w', '-shared', '-fPIC', '-I', RUNTIME, src, '-o', out, '-lm', '-lpthread'])
    return ctypes.CDLL(out)


def random_blocks(rng, name, nblocks, size, f16_at):
    raw = rng.integers(0, 256, size=(nblocks, size), dtype=np.uint8)
    for off in f16_at:
        v = rng.uniform(0.002, 0.05, size=nblocks).astype(np.float16)
        if name.endswith('_1') and off == 2:
            v = (-v).astype(np.float16)
        raw[:, off:off + 2] = v.view(np.uint8).reshape(nblocks, 2)
    return raw


def build_fused():
    out = os.path.join(tempfile.mkdtemp(), 'f.so')
    src = os.path.join(tempfile.mkdtemp(), 'f.cpp')
    open(src, 'w').write('#define VL_EMULATE 1\n#include "%s/fused.cu"\n' % HERE + r'''
extern "C" void run_rmsnorm_q8_1(const float *x, const float *w, float *y, unsigned char *q8, int rows, int cols, float eps) {
  VL_LAUNCH_THREADED(vira_cu_rmsnorm_q8_1_f32, rows, 256, x, w, y, q8, cols, eps);
}
''')
    subprocess.check_call(['g++', '-std=c++20', '-O2', '-shared', '-fPIC', '-pthread', '-I', HERE, src, '-o', out])
    return ctypes.CDLL(out)


def check_fused(rng):
    """rmsnorm + Q8_1 against NumPy: the rmsnorm, then the Q8_1 rule of vira_cu_quantize_q8_1."""
    lib = build_fused()
    ok = True
    for rows, cols in [(1, 64), (1, 4096), (2, 1024), (4, 2048), (3, 96), (1, 5120)]:
        x = rng.standard_normal((rows, cols)).astype(np.float32)
        x[0, :32] = 0.0  # an all-zero Q8_1 block (amax == 0)
        w = rng.uniform(0.5, 1.5, cols).astype(np.float32)
        eps = np.float32(1e-6)
        y = np.zeros((rows, cols), np.float32)
        q8 = np.zeros(rows * cols // 32 * 40, np.uint8)
        lib.run_rmsnorm_q8_1(x.ctypes.data_as(ctypes.c_void_p), w.ctypes.data_as(ctypes.c_void_p),
                             y.ctypes.data_as(ctypes.c_void_p), q8.ctypes.data_as(ctypes.c_void_p),
                             rows, cols, ctypes.c_float(eps))
        r = (1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(axis=1) + eps)).astype(np.float32)
        want = (r[:, None] * x) * w[None, :]
        e_y = np.abs(y - want).max() / np.abs(want).max()
        b = y.reshape(rows, cols // 32, 32)
        amax = np.abs(b).max(axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            qq = np.where(amax[..., None] > 0, np.clip(np.rint(b * (np.float32(127.0) / amax[..., None])), -127, 127), 0).astype(np.int32)
        d = (amax * np.float32(1.0 / 127.0)).astype(np.float32)
        blocks = q8.reshape(rows, cols // 32, 40)
        gd = blocks[..., 0:4].copy().view(np.float32)[..., 0]
        gs = blocks[..., 4:8].copy().view(np.float32)[..., 0]
        gq = blocks[..., 8:].view(np.int8).astype(np.int32)
        exact = bool(np.array_equal(gq, qq) and np.allclose(gd, d, rtol=1e-6, atol=0) and
                     np.allclose(gs, d * qq.sum(axis=2), rtol=1e-5, atol=1e-7))
        good = bool(e_y < 1e-5 and exact)
        ok &= good
        print('rmsnorm_q8_1 rows=%d cols=%5d  y rel err %.1e  q8 matches the quantize rule: %s  %s'
              % (rows, cols, e_y, exact, 'ok' if good else 'FAIL'))
    return ok


def build():
    out = os.path.join(tempfile.mkdtemp(), 'k.so')
    src = os.path.join(tempfile.mkdtemp(), 'k.cpp')
    open(src, 'w').write('#define VL_EMULATE 1\n#include "%s/mmvq2.cu"\n' % HERE + r'''
extern "C" void run_mmvq2_q8_0(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) {
  VL_LAUNCH(vira_cu_mmvq2_q8_0, N, M, THREADS, xq, w, y, M, nb, N);
}
#define VL_RUN_WARP(fmt) \
extern "C" void run_mmvq2w_##fmt(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) { \
  VL_LAUNCH2(vira_cu_mmvq2w_##fmt, (N + 3) / 4, M, 32, 4, xq, w, y, M, nb, N); \
}
VL_RUN_WARP(q8_0) VL_RUN_WARP(q4_0) VL_RUN_WARP(q4_1) VL_RUN_WARP(q5_0) VL_RUN_WARP(q5_1)
VL_RUN_WARP(q4_k) VL_RUN_WARP(q5_k) VL_RUN_WARP(q6_k)
''')
    subprocess.check_call(['g++', '-std=c++20', '-O2', '-shared', '-fPIC', '-pthread', '-I', HERE, src, '-o', out])
    return ctypes.CDLL(out)


def q8_0_weights(rng, N, K):
    """GGUF Q8_0: per 32 values {f16 d, int8 qs[32]} = 34 bytes."""
    w = rng.standard_normal((N, K)).astype(np.float32)
    b = w.reshape(N, K // 32, 32)
    d = np.abs(b).max(axis=2) / 127.0
    d[d == 0] = 1.0
    qs = np.clip(np.rint(b / d[..., None]), -127, 127).astype(np.int8)
    d16 = d.astype(np.float16)
    raw = np.zeros((N, K // 32, 34), np.uint8)
    raw[..., 0:2] = d16.view(np.uint8).reshape(N, K // 32, 2)
    raw[..., 2:] = qs.view(np.uint8)
    return raw.tobytes(), d16.astype(np.float32), qs.astype(np.int32)


def q8_1_acts(rng, M, K):
    """Q8_1 as vira_cu_quantize_q8_1 writes it: {f32 d, f32 sum, int8 qs[32]} = 40 bytes."""
    x = rng.standard_normal((M, K)).astype(np.float32)
    b = x.reshape(M, K // 32, 32)
    d = (np.abs(b).max(axis=2) / 127.0).astype(np.float32)
    d[d == 0] = 1.0
    qs = np.clip(np.rint(b / d[..., None]), -127, 127).astype(np.int8)
    raw = np.zeros((M, K // 32, 40), np.uint8)
    head = np.stack([d, (d * qs.sum(axis=2)).astype(np.float32)], axis=-1).astype(np.float32)
    raw[..., 0:8] = head.view(np.uint8).reshape(M, K // 32, 8)
    raw[..., 8:] = qs.view(np.uint8)
    return raw.tobytes(), d, qs.astype(np.int32)


def main():
    lib = build()
    rng = np.random.default_rng(0)
    cases = [(1, 64, 8), (1, 4096, 33), (2, 4096, 17), (4, 1024, 5), (3, 12288, 7), (1, 96, 151)]
    ok = True
    for M, K, N in cases:
        wraw, wd, wq = q8_0_weights(rng, N, K)
        xraw, xd, xq = q8_1_acts(rng, M, K)
        # reference: sum over blocks of dw * dx * dot(qw, qx)
        dots = np.einsum('nbk,mbk->mnb', wq, xq).astype(np.float64)
        want = (dots * wd[None, :, :] * xd[:, None, :]).sum(axis=2)
        for kname in ('mmvq2_q8_0', 'mmvq2w_q8_0'):
            y = np.full(M * N, np.nan, np.float32)
            getattr(lib, 'run_' + kname)(xraw, wraw, y.ctypes.data_as(ctypes.c_void_p), M, K // 32, N)
            err = np.abs(y.reshape(M, N) - want).max() / max(1e-9, np.abs(want).max())
            good = bool(err < 1e-5)
            ok &= good
            print('%-12s M=%d K=%5d N=%3d  max rel err %.2e  %s' % (kname, M, K, N, err, 'ok' if good else 'FAIL'))
    # every format against the runtime's CPU dequantiser
    oracle = build_oracle()
    for name, dt, size, qk, f16_at in FORMATS:
        for M, K, N in [(1, 256, 5), (1, 4096, 9), (2, 2048, 6), (4, 1536, 3), (1, 12288, 2)]:
            nbr = K // qk
            raw = random_blocks(rng, name, N * nbr, size, f16_at)
            wf = np.zeros(N * K, np.float32)
            oracle.oracle(dt, raw.tobytes(), ctypes.c_size_t(N * nbr), wf.ctypes.data_as(ctypes.c_void_p))
            xraw, xd, xq = q8_1_acts(rng, M, K)
            xf = (xq.astype(np.float64) * xd[..., None]).reshape(M, K)
            want = xf @ wf.reshape(N, K).astype(np.float64).T
            y = np.full(M * N, np.nan, np.float32)
            getattr(lib, 'run_mmvq2w_' + name)(xraw, raw.tobytes(), y.ctypes.data_as(ctypes.c_void_p), M, nbr, N)
            err = np.abs(y.reshape(M, N) - want).max() / max(1e-9, np.abs(want).max())
            good = bool(err < 1e-4)
            ok &= good
            print('mmvq2w_%-5s M=%d K=%5d N=%d  vs CPU dequant: max rel err %.2e  %s' % (name, M, K, N, err, 'ok' if good else 'FAIL'))
    ok &= check_fused(rng)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
