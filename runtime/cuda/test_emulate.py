"""Checks the kernels' index arithmetic on the CPU (no GPU needed).

    python3 runtime/cuda/test_emulate.py

Compiles the .cu sources as C++ with emulate.h, runs them on Q8_0 weights and Q8_1
activations built here, and compares with a plain sum over the same quantized values.
"""
import ctypes, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build():
    out = os.path.join(tempfile.mkdtemp(), 'k.so')
    src = os.path.join(tempfile.mkdtemp(), 'k.cpp')
    open(src, 'w').write('#define VL_EMULATE 1\n#include "%s/mmvq2.cu"\n' % HERE + r'''
extern "C" void run_mmvq2_q8_0(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) {
  VL_LAUNCH(vira_cu_mmvq2_q8_0, N, M, THREADS, xq, w, y, M, nb, N);
}
extern "C" void run_mmvq2w_q8_0(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) {
  VL_LAUNCH2(vira_cu_mmvq2w_q8_0, (N + 3) / 4, M, 32, 4, xq, w, y, M, nb, N);
}
''')
    subprocess.check_call(['g++', '-O2', '-shared', '-fPIC', '-I', HERE, src, '-o', out])
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
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
