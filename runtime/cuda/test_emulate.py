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
extern "C" void run_moe_act(float *g, const float *u, int n, int act, float lim) {
  VL_LAUNCH(vira_cu_moe_act_f32, (n + 255) / 256, 1, 256, g, u, n, act, lim);
}
extern "C" void run_moe_sum(const float *y, const float *wt, const int *pos, float *out, int M, int n_used, int D) {
  VL_LAUNCH(vira_cu_moe_sum_f32, (D + 255) / 256, M, 256, y, wt, pos, out, M, n_used, D);
}
extern "C" void run_rmsnorm_q8_1(const float *x, const float *w, float *y, unsigned char *q8, int rows, int cols, float eps) {
  VL_LAUNCH_THREADED(vira_cu_rmsnorm_q8_1_f32, (cols / 32 + 7) / 8, rows, 256, x, w, y, q8, cols, eps);
}
extern "C" void run_rmsnorm_x4(const float *x, const float *w, float *y, unsigned char *q4, int rows, int cols, float eps) {
  VL_LAUNCH_THREADED(vira_cu_rmsnorm_x4_f32, rows, 1, 256, x, w, y, q4, cols, eps);
}
extern "C" void run_silu_mul(const float *g, const float *u, float *o, int n) {
  VL_LAUNCH(vira_cu_silu_mul_f32, (n + 255) / 256, 1, 256, g, u, o, n);
}
''')
    subprocess.check_call(['g++', '-std=c++20', '-O2', '-shared', '-fPIC', '-pthread', '-I', HERE, src, '-o', out])
    return ctypes.CDLL(out)


def check_moe_act(lib, rng):
    """vira_cu_moe_act_f32 against the formulas of ViraLama's CPU MoE path (vl_moe_full)."""
    ok = True
    n = 1000
    g0 = (rng.standard_normal(n) * 4).astype(np.float32)
    u = (rng.standard_normal(n) * 4).astype(np.float32)
    for act, lim in [(0, 0.0), (1, 0.0), (2, 0.0), (3, 2.5), (3, 0.0)]:
        g = g0.copy()
        lib.run_moe_act(g.ctypes.data_as(ctypes.c_void_p), u.ctypes.data_as(ctypes.c_void_p), n, act, ctypes.c_float(lim))
        v, b = g0.astype(np.float64), u.astype(np.float64)
        if act == 3:
            a, c = (np.minimum(v, lim), np.clip(b, -lim, lim)) if lim > 1e-6 else (v, b)
            want = a / (1 + np.exp(-a)) * c
        elif act == 2:
            xg, yu = np.minimum(v, 7.0), np.clip(b, -7.0, 7.0)
            want = xg / (1 + np.exp(-1.702 * xg)) * (yu + 1)
        elif act == 1:
            want = 0.5 * v * (1 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3))) * b
        else:
            want = v / (1 + np.exp(-v)) * b
        err = np.abs(g - want).max() / np.abs(want).max()
        good = bool(err < 1e-5)
        ok &= good
        print('moe_act act=%d lim=%.1f  max rel err %.1e  %s' % (act, lim, err, 'ok' if good else 'FAIL'))
    return ok


def check_fused(rng):
    """rmsnorm + Q8_1 against NumPy: the rmsnorm, then the Q8_1 rule of vira_cu_quantize_q8_1."""
    lib = build_fused()
    ok = check_moe_act(lib, rng)
    M, nu, D, P = 3, 4, 300, 12
    yv = rng.standard_normal((P, D)).astype(np.float32)
    wt = rng.random(M * nu).astype(np.float32)
    pos = rng.permutation(P).astype(np.int32)
    out = np.zeros((M, D), np.float32)
    lib.run_moe_sum(yv.ctypes.data_as(ctypes.c_void_p), wt.ctypes.data_as(ctypes.c_void_p), pos.ctypes.data_as(ctypes.c_void_p),
                    out.ctypes.data_as(ctypes.c_void_p), M, nu, D)
    want = np.stack([sum(wt[t * nu + k] * yv[pos[t * nu + k]].astype(np.float64) for k in range(nu)) for t in range(M)])
    e = np.abs(out - want).max() / np.abs(want).max()
    ok &= bool(e < 1e-6)
    print('moe_sum  max rel err %.1e  %s' % (e, 'ok' if e < 1e-6 else 'FAIL'))
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
        if cols % 128 == 0:  # prompt rows: the same y and the same numbers in the x4 layout
            y4 = np.zeros((rows, cols), np.float32)
            q4 = np.zeros(rows * cols // 32 * 40, np.uint8)
            lib.run_rmsnorm_x4(x.ctypes.data_as(ctypes.c_void_p), w.ctypes.data_as(ctypes.c_void_p),
                               y4.ctypes.data_as(ctypes.c_void_p), q4.ctypes.data_as(ctypes.c_void_p),
                               rows, cols, ctypes.c_float(eps))
            same = bool(np.array_equal(y4, y) and np.array_equal(q4, np.frombuffer(q8_1_to_x4(q8.tobytes(), rows, cols), np.uint8)))
            ok &= same
            print('rmsnorm_x4   rows=%d cols=%5d  y and q bytes equal rmsnorm_q8_1 (x4 order): %s  %s'
                  % (rows, cols, same, 'ok' if same else 'FAIL'))
    for n in (1, 255, 1000):
        g = (3.0 * rng.standard_normal(n)).astype(np.float32)
        u = rng.standard_normal(n).astype(np.float32)
        o = np.zeros(n, np.float32)
        lib.run_silu_mul(g.ctypes.data_as(ctypes.c_void_p), u.ctypes.data_as(ctypes.c_void_p), o.ctypes.data_as(ctypes.c_void_p), n)
        e2 = np.exp2((g * np.array([0xBFB8AA3B], np.uint32).view(np.float32)[0]).astype(np.float32)).astype(np.float32)
        want = ((g / (e2 + np.float32(1.0))).astype(np.float32) * u).astype(np.float32)
        e = float(np.abs(o - want).max() / max(1e-30, np.abs(want).max()))
        same = e < 1e-6  # exp2 of NumPy vs the C library: an ulp apart at most
        ok &= same
        print('silu_mul n=%4d  vs silu then multiply in float32: rel err %.1e  %s' % (n, e, 'ok' if same else 'FAIL'))
    return ok


def build_mmq(kind='mmq'):
    """kind 'mmq' (dp4a, tile 64 x 32) or 'mmqt' (tensor cores, tile 64 x 64)."""
    out = os.path.join(tempfile.mkdtemp(), 'q.so')
    src = os.path.join(tempfile.mkdtemp(), 'q.cpp')
    body = '#define VL_EMULATE 1\n#include "%s/%s.cu"\n' % (HERE, kind)
    tm = 32 if kind == 'mmq' else 64
    for name, _, _, _, _ in FORMATS:
        if kind == 'mmqt':
            body += ('extern "C" void run_%s_%s(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) {\n'
                     '  VL_LAUNCH_THREADED(vira_cu_%s_%s, (M + %d) / %d, (N + 63) / 64, 128, xq, w, y, M, nb, N);\n}\n') % (
                         kind, name, kind, name, tm - 1, tm)
        else:
            body += ('extern "C" void run_%s_%s(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) {\n'
                     '  VL_LAUNCH_THREADED(vira_cu_%s_%s, (N + 63) / 64, (M + %d) / %d, 128, xq, w, y, M, nb, N);\n}\n') % (
                         kind, name, kind, name, tm - 1, tm)
    if kind == 'mmqt':
        for name, _, _, _, _ in FORMATS:
            for J in (128, 64):
                body += ('extern "C" void run_mmqb%d_%s(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) {\n'
                         '  VL_LAUNCH_THREADED(vira_cu_mmqb%d_%s, (M + %d) / %d, (N + 127) / 128, 256, xq, w, y, M, nb, N);\n}\n') % (
                             J, name, J, name, J - 1, J)
        for name, _, _, _, _ in FORMATS:
            body += ('extern "C" void run_mmqg_%s(const unsigned char *xq, const unsigned char *wg, const unsigned char *wu, float *y, int M, int nb, int N, unsigned char *q4) {\n'
                     '  VL_LAUNCH_THREADED(vira_cu_mmqg128_%s, (M + 127) / 128, (N + 63) / 64, 256, xq, wg, wu, y, M, nb, N, q4);\n}\n') % (name, name)
        body += ('extern "C" void run_quantize_x4(const float *x, unsigned char *q, int M, int K) {\n'
                 '  VL_LAUNCH_THREADED(vira_cu_quantize_q8_1x4, K / 128, M, 128, x, q, K);\n}\n')
    open(src, 'w').write(body)
    subprocess.check_call(['g++', '-std=c++20', '-O2', '-shared', '-fPIC', '-pthread', '-I', HERE, src, '-o', out])
    return ctypes.CDLL(out)


def q8_1_to_x4(raw, M, K):
    """40-byte Q8_1 blocks -> the x4 layout of mmqt.cu: per row and 128 values {d[4], s[4], q[128]}."""
    b = np.frombuffer(raw, np.uint8).reshape(M, K // 128, 4, 40)
    out = np.zeros((M, K // 128, 160), np.uint8)
    out[..., 0:16] = b[..., 0:4].reshape(M, K // 128, 16)
    out[..., 16:32] = b[..., 4:8].reshape(M, K // 128, 16)
    out[..., 32:] = b[..., 8:].reshape(M, K // 128, 128)
    return out.tobytes()


def check_mmqg(rng, oracle):
    """Fused gate/up + SwiGLU (vira_cu_mmqg128_<fmt>) against silu(x Wg) * (x Wu) from the CPU
    dequantiser (SiLU in float64)."""
    lib = build_mmq('mmqt')
    ok = True
    for name, dt, size, qk, f16_at in FORMATS:
        for M, K, N in [(70, 512, 66), (133, 256, 70), (77, 256, 256)]:
            nbr = K // qk
            ws = []
            for _ in range(2):
                raw = random_blocks(rng, name, N * nbr, size, f16_at)
                wf = np.zeros(N * K, np.float32)
                oracle.oracle(dt, raw.tobytes(), ctypes.c_size_t(N * nbr), wf.ctypes.data_as(ctypes.c_void_p))
                ws.append((raw, wf.reshape(N, K).astype(np.float64)))
            xraw, xd, xq = q8_1_acts(rng, M, K)
            xf = (xq.astype(np.float64) * xd[..., None]).reshape(M, K)
            g = xf @ ws[0][1].T
            u = xf @ ws[1][1].T
            want = g / (1.0 + np.exp(-g)) * u
            y = np.full(M * N, np.nan, np.float32)
            q4 = np.zeros(M * N // 32 * 40, np.uint8) if N % 128 == 0 else None
            getattr(lib, 'run_mmqg_' + name)(q8_1_to_x4(xraw, M, K), ws[0][0].tobytes(), ws[1][0].tobytes(), y.ctypes.data_as(ctypes.c_void_p), M, nbr, N,
                                            q4.ctypes.data_as(ctypes.c_void_p) if q4 is not None else None)
            err = np.abs(y.reshape(M, N) - want).max() / max(1e-9, np.abs(want).max())
            good = bool(err < 1e-4)
            if q4 is not None:  # the x4 copy must be the quantize rule applied to y
                qref = np.zeros(M * N // 32 * 40, np.uint8)
                lib.run_quantize_x4(y.ctypes.data_as(ctypes.c_void_p), qref.ctypes.data_as(ctypes.c_void_p), M, N)
                same = bool(np.array_equal(q4, qref))
                print('mmqg128_%-5s M=%3d N=%3d  x4 copy equals quantize_q8_1x4(y): %s' % (name, M, N, same))
                good &= same
            ok &= good
            print('mmqg128_%-5s M=%3d K=%4d N=%3d  vs silu(x Wg) * (x Wu): max rel err %.2e  %s' % (name, M, K, N, err, 'ok' if good else 'FAIL'))
    return ok


def check_quantize_x4(rng):
    """vira_cu_quantize_q8_1x4 against the GPU quantizer's arithmetic in float32."""
    lib = build_mmq('mmqt')
    ok = True
    for M, K in [(3, 256), (5, 1024)]:
        x = rng.standard_normal((M, K)).astype(np.float32)
        x[1, 128:160] = 0.0  # an all-zero block
        q = np.zeros(M * K // 128 * 160, np.uint8)
        lib.run_quantize_x4(x.ctypes.data_as(ctypes.c_void_p), q.ctypes.data_as(ctypes.c_void_p), M, K)
        b = x.reshape(M, K // 32, 32)
        amax = np.abs(b).max(axis=2).astype(np.float32)
        inv = np.where(amax > 0, np.float32(127.0) / np.where(amax > 0, amax, 1), 0).astype(np.float32)
        qs = np.clip(np.rint((b * inv[..., None]).astype(np.float32)), -127, 127).astype(np.int8)
        d = (amax * np.array([0x3C010204], np.uint32).view(np.float32)[0]).astype(np.float32)
        s = (d * qs.astype(np.int32).sum(axis=2).astype(np.float32)).astype(np.float32)
        raw = np.zeros((M, K // 32, 40), np.uint8)
        raw[..., 0:4] = d.view(np.uint8).reshape(M, K // 32, 4)
        raw[..., 4:8] = s.view(np.uint8).reshape(M, K // 32, 4)
        raw[..., 8:] = qs.view(np.uint8)
        want = np.frombuffer(q8_1_to_x4(raw.tobytes(), M, K), np.uint8)
        good = bool(np.array_equal(q, want))
        ok &= good
        print('quantize_q8_1x4 M=%d K=%4d  bytes equal %s  %s' % (M, K, good, 'ok' if good else 'FAIL'))
    return ok


def check_mmq(rng, oracle, kind='mmq'):
    """Prompt-size products (M > 4) of every format against the CPU dequantiser."""
    lib = build_mmq(kind)
    ok = True
    for name, dt, size, qk, f16_at in FORMATS:
        for M, K, N in [(5, 256, 7), (32, 512, 64), (40, 768, 70), (9, 1024, 130), (70, 512, 66), (133, 256, 135)]:
            nbr = K // qk
            raw = random_blocks(rng, name, N * nbr, size, f16_at)
            wf = np.zeros(N * K, np.float32)
            oracle.oracle(dt, raw.tobytes(), ctypes.c_size_t(N * nbr), wf.ctypes.data_as(ctypes.c_void_p))
            xraw, xd, xq = q8_1_acts(rng, M, K)
            xf = (xq.astype(np.float64) * xd[..., None]).reshape(M, K)
            want = xf @ wf.reshape(N, K).astype(np.float64).T
            if kind == 'mmqt':
                xraw = q8_1_to_x4(xraw, M, K)
            y = np.full(M * N, np.nan, np.float32)
            kernels = [kind] if kind == 'mmq' else ['mmqt', 'mmqb128', 'mmqb64']
            for kn in kernels:
              y = np.full(M * N, np.nan, np.float32)
              getattr(lib, 'run_%s_%s' % (kn, name))(xraw, raw.tobytes(), y.ctypes.data_as(ctypes.c_void_p), M, nbr, N)
              err = np.abs(y.reshape(M, N) - want).max() / max(1e-9, np.abs(want).max())
              good = bool(err < 1e-4)
              ok &= good
              print('%s_%-5s M=%2d K=%4d N=%3d  vs CPU dequant: max rel err %.2e  %s' % (kn, name, M, K, N, err, 'ok' if good else 'FAIL'))
    return ok


def check_attn(rng):
    """FP16-KV attention (decode and causal prompt rows, GQA) against NumPy on the same
    half-rounded cache; and the f32 -> f16 row store."""
    out = os.path.join(tempfile.mkdtemp(), 'a.so')
    src = os.path.join(tempfile.mkdtemp(), 'a.cpp')
    open(src, 'w').write('#define VL_EMULATE 1\n#include "%s/attn.cu"\n' % HERE + r'''
extern "C" void run_attn(const float *q, const unsigned short *k, const unsigned short *v, float *o, int nh, int kvm,
                         int hd, int kvd, int M, long long start, float scale) {
  VL_LAUNCH_THREADED(vira_cu_attn_f16kv, nh, M, 256, q, k, v, o, nh, kvm, hd, kvd, M, start, scale);
}
extern "C" void run_attn_tile(const float *q, const unsigned short *k, const unsigned short *v, float *o, int nh, int kvm,
                              int hd, int kvd, int M, long long start, float scale) {
  VL_LAUNCH_THREADED(vira_cu_attn_f16kv_tile, nh, (M + 15) / 16, 256, q, k, v, o, nh, kvm, hd, kvd, M, start, scale);
}
extern "C" void run_store(const float *s, unsigned short *d, long long row0, int cols, long long n) {
  VL_LAUNCH(vira_cu_kv_store_f16, (int)((n + 255) / 256), 1, 256, s, d, row0, cols, n);
}
''')
    subprocess.check_call(['g++', '-std=c++20', '-O2', '-shared', '-fPIC', '-pthread', '-I', HERE, src, '-o', out])
    lib = ctypes.CDLL(out)
    ok = True
    cases = [(4, 2, 64, 40, 1, 20, 'attn_f16kv'), (8, 2, 128, 64, 5, 30, 'attn_f16kv'), (2, 1, 256, 16, 3, 0, 'attn_f16kv'),
             (6, 3, 80, 33, 1, 0, 'attn_f16kv'), (4, 4, 32, 300, 1, 299, 'attn_f16kv'),
             (8, 2, 128, 64, 5, 30, 'attn_tile'), (4, 2, 64, 80, 37, 3, 'attn_tile'), (5, 1, 128, 70, 70, 0, 'attn_tile'),
             (2, 2, 96, 120, 17, 100, 'attn_tile'), (4, 4, 32, 40, 33, 0, 'attn_tile')]
    for nh, nkv, hd, ctx, M, start, kname in cases:
        kvd = nkv * hd
        kf = rng.standard_normal((ctx, kvd)).astype(np.float32)
        vf = rng.standard_normal((ctx, kvd)).astype(np.float32)
        k16 = np.zeros(ctx * kvd, np.uint16)
        lib.run_store(kf.ctypes.data_as(ctypes.c_void_p), k16.ctypes.data_as(ctypes.c_void_p), ctypes.c_longlong(0), kvd, ctypes.c_longlong(ctx * kvd))
        good_store = bool(np.array_equal(k16, kf.astype(np.float16).view(np.uint16).ravel()))
        v16 = vf.astype(np.float16)
        q = rng.standard_normal((M * nh, hd)).astype(np.float32)
        o = np.full((M * nh, hd), np.nan, np.float32)
        scale = 1.0 / np.sqrt(hd)
        getattr(lib, 'run_attn' if kname == 'attn_f16kv' else 'run_attn_tile')(q.ctypes.data_as(ctypes.c_void_p), k16.ctypes.data_as(ctypes.c_void_p),
                     v16.view(np.uint16).ctypes.data_as(ctypes.c_void_p), o.ctypes.data_as(ctypes.c_void_p),
                     nh, nh // nkv, hd, kvd, M, ctypes.c_longlong(start), ctypes.c_float(scale))
        K = kf.astype(np.float16).astype(np.float64)
        V = v16.astype(np.float64)
        want = np.zeros((M * nh, hd))
        for t in range(M):
            n = start + t + 1
            for h in range(nh):
                kh = (h // (nh // nkv)) * hd
                s = K[:n, kh:kh + hd] @ q[t * nh + h].astype(np.float64) * scale
                p = np.exp(s - s.max()); p /= p.sum()
                want[t * nh + h] = p @ V[:n, kh:kh + hd]
        err = np.abs(o - want).max() / np.abs(want).max()
        good = bool(err < 1e-5) and good_store
        ok &= good
        print('%-10s nh=%d kv=%d hd=%3d M=%2d start=%3d  rel err %.1e  store %s  %s' % (kname, nh, nkv, hd, M, start, err, good_store, 'ok' if good else 'FAIL'))
    return ok


def check_attn_fa(rng):
    """Tensor-core prompt attention (attnt.cu) against NumPy on the half-rounded cache. Q and P
    go through f16 inside the kernel, so the tolerance is f16-sized."""
    out = os.path.join(tempfile.mkdtemp(), 'fa.so')
    src = os.path.join(tempfile.mkdtemp(), 'fa.cpp')
    open(src, 'w').write('#define VL_EMULATE 1\n#include "%s/attnt.cu"\n' % HERE + r'''
extern "C" void run_fa(const float *q, const unsigned short *k, const unsigned short *v, float *o, int nh, int kvm,
                       int hd, int kvd, int M, long long start, float scale, unsigned char *q4) {
  if (hd == 128) { VL_LAUNCH_THREADED(vira_cu_attn_fa128, nh, (M + 63) / 64, 128, q, k, v, o, nh, kvm, hd, kvd, M, start, scale, q4); }
  else { VL_LAUNCH_THREADED(vira_cu_attn_fa64, nh, (M + 63) / 64, 128, q, k, v, o, nh, kvm, hd, kvd, M, start, scale, q4); }
}
''')
    subprocess.check_call(['g++', '-std=c++20', '-O2', '-shared', '-fPIC', '-pthread', '-I', HERE, src, '-o', out])
    lib = ctypes.CDLL(out)
    ok = True
    for nh, nkv, hd, ctx, M, start in [(4, 2, 128, 64, 5, 30), (5, 1, 128, 80, 80, 0), (4, 2, 64, 120, 70, 33),
                                       (2, 2, 128, 200, 37, 150), (8, 2, 64, 40, 33, 0)]:
        kvd = nkv * hd
        k16 = rng.standard_normal((ctx, kvd)).astype(np.float16)
        v16 = rng.standard_normal((ctx, kvd)).astype(np.float16)
        q = rng.standard_normal((M * nh, hd)).astype(np.float32)
        o = np.full((M * nh, hd), np.nan, np.float32)
        scale = 1.0 / np.sqrt(hd)
        kt = nh * hd
        q4 = np.zeros(M * kt // 32 * 40, np.uint8) if kt % 128 == 0 else None
        lib.run_fa(q.ctypes.data_as(ctypes.c_void_p), k16.view(np.uint16).ctypes.data_as(ctypes.c_void_p),
                   v16.view(np.uint16).ctypes.data_as(ctypes.c_void_p), o.ctypes.data_as(ctypes.c_void_p),
                   nh, nh // nkv, hd, kvd, M, ctypes.c_longlong(start), ctypes.c_float(scale),
                   q4.ctypes.data_as(ctypes.c_void_p) if q4 is not None else None)
        if q4 is not None:  # the x4 copy of o (rows of nh * hd values) by the quantize rule
            b = o.reshape(M, kt // 32, 32)
            amax = np.abs(b).max(axis=2).astype(np.float32)
            inv = np.where(amax > 0, np.float32(127.0) / np.where(amax > 0, amax, 1), 0).astype(np.float32)
            qs = np.clip(np.rint((b * inv[..., None]).astype(np.float32)), -127, 127).astype(np.int8)
            dd = (amax * np.array([0x3C010204], np.uint32).view(np.float32)[0]).astype(np.float32)
            ss = (dd * qs.astype(np.int32).sum(axis=2).astype(np.float32)).astype(np.float32)
            raw = np.zeros((M, kt // 32, 40), np.uint8)
            raw[..., 0:4] = dd.view(np.uint8).reshape(M, kt // 32, 4)
            raw[..., 4:8] = ss.view(np.uint8).reshape(M, kt // 32, 4)
            raw[..., 8:] = qs.view(np.uint8)
            same = bool(np.array_equal(q4, np.frombuffer(q8_1_to_x4(raw.tobytes(), M, kt), np.uint8)))
            print('attn_fa    x4 copy of the output equals the quantize rule: %s' % same)
            ok &= same
        K = k16.astype(np.float64)
        V = v16.astype(np.float64)
        want = np.zeros((M * nh, hd))
        for t in range(M):
            n = start + t + 1
            for h in range(nh):
                kh = (h // (nh // nkv)) * hd
                s = K[:n, kh:kh + hd] @ q[t * nh + h].astype(np.float64) * scale
                p = np.exp(s - s.max()); p /= p.sum()
                want[t * nh + h] = p @ V[:n, kh:kh + hd]
        err = np.abs(o - want).max() / np.abs(want).max()
        good = bool(err < 5e-3)
        ok &= good
        print('attn_fa    nh=%d kv=%d hd=%3d M=%2d start=%3d  rel err %.1e  %s' % (nh, nkv, hd, M, start, err, 'ok' if good else 'FAIL'))
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
#define VL_RUN_X(fmt) \
extern "C" void run_mmvq2x_##fmt(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) { \
  VL_LAUNCH2(vira_cu_mmvq2x_##fmt, (N + 7) / 8, M, 32, 4, xq, w, y, M, nb, N); \
}
VL_RUN_X(q4_0) VL_RUN_X(q4_1) VL_RUN_X(q5_0) VL_RUN_X(q5_1) VL_RUN_X(q4_k) VL_RUN_X(q5_k) VL_RUN_X(q6_k)
#define VL_RUN_Q(fmt) \
extern "C" void run_mmvq2q_##fmt(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) { \
  VL_LAUNCH2(vira_cu_mmvq2q_##fmt, (N + 15) / 16, M, 32, 4, xq, w, y, M, nb, N); \
}
#define VL_RUN_ID(fmt) \
extern "C" void run_mmvq_id_##fmt(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N, const int *ids, const int *xrow, long long es) { \
  VL_LAUNCH2(vira_cu_mmvq_id_##fmt, (N + 7) / 8, M, 32, 4, xq, w, y, M, nb, N, ids, xrow, es); \
}
#define VL_RUN_B(fmt) \
extern "C" void run_mmvqb_##fmt(const unsigned char *xq, const unsigned char *w, float *y, int M, int nb, int N) { \
  VL_LAUNCH2(vira_cu_mmvqb_##fmt, (N + 7) / 8, 1, 32, 4, xq, w, y, M, nb, N); \
}
VL_RUN_B(q4_0) VL_RUN_B(q4_1) VL_RUN_B(q5_0) VL_RUN_B(q5_1) VL_RUN_B(q8_0) VL_RUN_B(q4_k) VL_RUN_B(q5_k) VL_RUN_B(q6_k)
VL_RUN_ID(q4_0) VL_RUN_ID(q4_1) VL_RUN_ID(q5_0) VL_RUN_ID(q5_1) VL_RUN_ID(q8_0) VL_RUN_ID(q4_k) VL_RUN_ID(q5_k) VL_RUN_ID(q6_k)
VL_RUN_Q(q4_0) VL_RUN_Q(q4_1) VL_RUN_Q(q5_0) VL_RUN_Q(q5_1) VL_RUN_Q(q4_k) VL_RUN_Q(q5_k) VL_RUN_Q(q6_k)
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


def check_paged(rng):
    """PagedAttention FP16 kernels (paged.cu): fused per-token RoPE + scatter write, CoW block
    copy, row gather, and ragged multi-sequence GQA attention (with SWA, softcap, and sinks)
    compared against independent NumPy reference + deliberate planted break test."""
    ext = '.dll' if os.name == 'nt' else '.so'
    out = os.path.join(tempfile.mkdtemp(), 'pg' + ext)
    src = os.path.join(tempfile.mkdtemp(), 'pg.cpp')
    src_inc = HERE.replace('\\', '/')
    open(src, 'w').write('#define VL_EMULATE 1\n#include "%s/paged.cu"\n' % src_inc + r'''
extern "C" void run_rope_write(float *q, float *k, const float *v, unsigned short *kp, unsigned short *vp,
                               const int *pos, const int *smap, int T, int nh, int nkv, int hd, int nrot,
                               int neox, float base, float fscale) {
  VL_LAUNCH_THREADED(vira_cu_paged_rope_and_write_f16, T, 1, 256, q, k, v, kp, vp, pos, smap, T, nh, nkv, hd, nrot, neox, base, fscale);
}
extern "C" void run_copy_blk(unsigned short *kp, unsigned short *vp, int dst_b, int src_b, int ps, int kvd, int nr) {
  long long n = (long long)nr * kvd;
  VL_LAUNCH(vira_cu_paged_copy_block_f16, (int)((n + 255) / 256), 1, 256, kp, vp, dst_b, src_b, ps, kvd, nr);
}
extern "C" void run_gather(const float *s, float *d, const int *ridx, int nout, int cols) {
  long long n = (long long)nout * cols;
  VL_LAUNCH(vira_cu_paged_gather_rows_f32, (int)((n + 255) / 256), 1, 256, s, d, ridx, nout, cols);
}
extern "C" void run_paged_attn(const float *q, const unsigned short *kp, const unsigned short *vp, float *o,
                               const int *seq_id, const int *max_kv, const int *bt, const float *sinks,
                               int nh, int kvm, int hd, int kvd, int T, int max_b, int ps, float scale,
                               int win, float scap, int hs) {
  VL_LAUNCH_THREADED(vira_cu_paged_attn_ragged_f16, nh, T, 256, q, kp, vp, o, seq_id, max_kv, bt, sinks,
                     nh, kvm, hd, kvd, T, max_b, ps, scale, win, scap, hs);
}
''')
    cmd = ['g++', '-std=c++20', '-O2', '-shared', '-pthread', '-I', HERE, src, '-o', out]
    if os.name != 'nt':
        cmd.insert(4, '-fPIC')
    subprocess.check_call(cmd)
    lib = ctypes.CDLL(out)
    ok = True
    ps = 16
    for nh, nkv, hd, neox, win, scap, use_sinks in [
        (8, 2, 64, 1, 0, 0.0, 0),
        (6, 3, 128, 0, 19, 30.0, 0),
        (4, 1, 64, 1, 0, 0.0, 1),
        (4, 4, 32, 0, 0, 0.0, 0),
    ]:
        kvd = nkv * hd
        n_blocks = 12
        kp = np.zeros((n_blocks * ps, kvd), np.uint16)
        vp = np.zeros((n_blocks * ps, kvd), np.uint16)
        # Pre-populate non-contiguous blocks for 3 sequences:
        # Seq 0: length 20 (1 decode token at pos 19), blocks [5, 2]
        # Seq 1: length 22 (6 chunked-prefill tokens at pos 16..21), shares prefix block 5 with Seq 0, then forks block 9!
        # Seq 2: length 5 (5 prefill tokens at pos 0..4), block [7]
        bt = np.array([[5, 2, 0], [5, 9, 0], [7, 0, 0]], np.int32)
        # Fill historical KV rows: block 5 rows 0..15 (shared prefix!), block 2 rows 0..2 (pos 16..18 of seq 0)
        hist_k = rng.standard_normal((n_blocks * ps, kvd)).astype(np.float16)
        hist_v = rng.standard_normal((n_blocks * ps, kvd)).astype(np.float16)
        kp[:] = hist_k.view(np.uint16)
        vp[:] = hist_v.view(np.uint16)

        # Test CoW block copy: copy 3 rows from block 2 -> block 9, verify exact match
        lib.run_copy_blk(kp.ctypes.data_as(ctypes.c_void_p), vp.ctypes.data_as(ctypes.c_void_p),
                         9, 2, ps, kvd, 3)
        cow_ok = bool(np.array_equal(kp[9 * ps:9 * ps + 3], kp[2 * ps:2 * ps + 3]))
        ok &= cow_ok

        # Build ragged step of T = 1 + 6 + 5 = 12 tokens
        positions = np.array([19, 16, 17, 18, 19, 20, 21, 0, 1, 2, 3, 4], np.int32)
        seq_id = np.array([0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], np.int32)
        max_kv = positions + 1
        smap = np.array([
            2 * ps + 3,
            9 * ps + 0, 9 * ps + 1, 9 * ps + 2, 9 * ps + 3, 9 * ps + 4, 9 * ps + 5,
            7 * ps + 0, 7 * ps + 1, 7 * ps + 2, 7 * ps + 3, 7 * ps + 4,
        ], np.int32)
        T = len(positions)
        q0 = rng.standard_normal((T * nh, hd)).astype(np.float32)
        k0 = rng.standard_normal((T * nkv, hd)).astype(np.float32)
        v0 = rng.standard_normal((T, kvd)).astype(np.float32)
        q = q0.copy(); k = k0.copy()
        base, fscale = 10000.0, 1.0
        lib.run_rope_write(q.ctypes.data_as(ctypes.c_void_p), k.ctypes.data_as(ctypes.c_void_p),
                           v0.ctypes.data_as(ctypes.c_void_p), kp.ctypes.data_as(ctypes.c_void_p),
                           vp.ctypes.data_as(ctypes.c_void_p), positions.ctypes.data_as(ctypes.c_void_p),
                           smap.ctypes.data_as(ctypes.c_void_p), T, nh, nkv, hd, hd, neox,
                           ctypes.c_float(base), ctypes.c_float(fscale))

        # Reference RoPE + scatter write in NumPy
        def ref_rope(arr, n_h):
            res = arr.copy().reshape(T, n_h, hd)
            half = hd // 2
            freqs = (positions[:, None].astype(np.float32) * fscale) * (base ** (-2.0 * np.arange(half, dtype=np.float32) / hd))[None, :]
            c, s = np.cos(freqs)[:, None, :], np.sin(freqs)[:, None, :]
            if neox:
                x0, x1 = res[:, :, :half].copy(), res[:, :, half:].copy()
                res[:, :, :half] = x0 * c - x1 * s
                res[:, :, half:] = x0 * s + x1 * c
            else:
                x0, x1 = res[:, :, 0::2].copy(), res[:, :, 1::2].copy()
                res[:, :, 0::2] = x0 * c - x1 * s
                res[:, :, 1::2] = x0 * s + x1 * c
            return res.reshape(T * n_h, hd)

        q_ref = ref_rope(q0, nh)
        k_ref = ref_rope(k0, nkv).reshape(T, kvd)
        rope_err = max(np.abs(q - q_ref).max(), np.abs(k.reshape(T, kvd) - k_ref).max())
        K_pool_f = kp.view(np.float16).astype(np.float64)
        V_pool_f = vp.view(np.float16).astype(np.float64)

        scale = 1.0 / np.sqrt(hd)
        sinks = rng.standard_normal(nh).astype(np.float32) if use_sinks else np.zeros(nh, np.float32)
        o = np.full((T * nh, hd), np.nan, np.float32)
        lib.run_paged_attn(q.ctypes.data_as(ctypes.c_void_p), kp.ctypes.data_as(ctypes.c_void_p),
                           vp.ctypes.data_as(ctypes.c_void_p), o.ctypes.data_as(ctypes.c_void_p),
                           seq_id.ctypes.data_as(ctypes.c_void_p), max_kv.ctypes.data_as(ctypes.c_void_p),
                           bt.ctypes.data_as(ctypes.c_void_p),
                           sinks.ctypes.data_as(ctypes.c_void_p) if use_sinks else None,
                           nh, nh // nkv, hd, kvd, T, 3, ps, ctypes.c_float(scale), win,
                           ctypes.c_float(scap), use_sinks)

        want = np.zeros((T * nh, hd), np.float64)
        for t in range(T):
            b = int(seq_id[t])
            nk = int(max_kv[t])
            lo = max(0, nk - win) if win > 0 else 0
            phys_rows = [int(bt[b, j // ps]) * ps + (j % ps) for j in range(lo, nk)]
            K_seq = K_pool_f[phys_rows]
            V_seq = V_pool_f[phys_rows]
            for h in range(nh):
                kh = (h // (nh // nkv)) * hd
                sc = (K_seq[:, kh:kh + hd] @ q[t * nh + h].astype(np.float64)) * scale
                if scap > 0.0:
                    sc = scap * np.tanh(sc / scap)
                mx = max(float(sc.max()), float(sinks[h])) if use_sinks else float(sc.max())
                p = np.exp(sc - mx)
                denom = p.sum() + (np.exp(float(sinks[h]) - mx) if use_sinks else 0.0)
                want[t * nh + h] = (p / denom) @ V_seq[:, kh:kh + hd]

        err = np.abs(o - want).max() / max(1e-9, np.abs(want).max())
        # Verify selective gather_rows for logit_rows = [0, 6, 11]
        ridx = np.array([0, 6, 11], np.int32)
        gathered = np.zeros((3, nh * hd), np.float32)
        o_flat = o.reshape(T, nh * hd)
        lib.run_gather(o_flat.ctypes.data_as(ctypes.c_void_p), gathered.ctypes.data_as(ctypes.c_void_p),
                       ridx.ctypes.data_as(ctypes.c_void_p), 3, nh * hd)
        gather_ok = bool(np.array_equal(gathered, o_flat[ridx]))

        # Deliberate break proof (AGENTS.md Rule 1): corrupting block_table must move output
        bt_bad = bt.copy(); bt_bad[1, 0] = 7
        o_bad = np.zeros_like(o)
        lib.run_paged_attn(q.ctypes.data_as(ctypes.c_void_p), kp.ctypes.data_as(ctypes.c_void_p),
                           vp.ctypes.data_as(ctypes.c_void_p), o_bad.ctypes.data_as(ctypes.c_void_p),
                           seq_id.ctypes.data_as(ctypes.c_void_p), max_kv.ctypes.data_as(ctypes.c_void_p),
                           bt_bad.ctypes.data_as(ctypes.c_void_p),
                           sinks.ctypes.data_as(ctypes.c_void_p) if use_sinks else None,
                           nh, nh // nkv, hd, kvd, T, 3, ps, ctypes.c_float(scale), win,
                           ctypes.c_float(scap), use_sinks)
        break_caught = bool(np.abs(o_bad - want).max() > 1e-2)

        good = bool(err < 1e-5 and rope_err < 1e-5 and cow_ok and gather_ok and break_caught)
        ok &= good
        print('paged_attn nh=%d kv=%d hd=%3d neox=%d win=%2d scap=%.0f sink=%d  rel err %.1e  break=%s  %s' %
              (nh, nkv, hd, neox, win, scap, use_sinks, err, break_caught, 'ok' if good else 'FAIL'))
    return ok


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
    oracle = build_oracle()
    # MoE: pairs (expert ids[m], activation row xrow[m]) in one launch
    for name, dt, size, qk, f16_at in FORMATS:
        n_exp, K, N, T, P = 5, 512, 11, 3, 7
        nbr = K // qk
        raw = random_blocks(rng, name, n_exp * N * nbr, size, f16_at)
        wf = np.zeros(n_exp * N * K, np.float32)
        oracle.oracle(dt, raw.tobytes(), ctypes.c_size_t(n_exp * N * nbr), wf.ctypes.data_as(ctypes.c_void_p))
        wf = wf.reshape(n_exp, N, K).astype(np.float64)
        xraw, xd, xq = q8_1_acts(rng, T, K)
        xf = (xq.astype(np.float64) * xd[..., None]).reshape(T, K)
        ids = rng.integers(0, n_exp, P).astype(np.int32)
        xrow = rng.integers(0, T, P).astype(np.int32)
        want = np.stack([wf[ids[m]] @ xf[xrow[m]] for m in range(P)])
        y = np.full(P * N, np.nan, np.float32)
        getattr(lib, 'run_mmvq_id_' + name)(xraw, raw.tobytes(), y.ctypes.data_as(ctypes.c_void_p), P, nbr, N,
                                            ids.ctypes.data_as(ctypes.c_void_p), xrow.ctypes.data_as(ctypes.c_void_p),
                                            ctypes.c_longlong(N * nbr * size))
        err = np.abs(y.reshape(P, N) - want).max() / max(1e-9, np.abs(want).max())
        good = bool(err < 1e-4)
        ok &= good
        print('mmvq_id_%-5s pairs=%d  vs CPU dequant: max rel err %.2e  %s' % (name, P, err, 'ok' if good else 'FAIL'))
    # every format against the runtime's CPU dequantiser
    for name, dt, size, qk, f16_at in FORMATS:
        for M, K, N in [(1, 256, 5), (1, 4096, 9), (2, 2048, 6), (4, 1536, 3), (1, 12288, 2), (1, 768, 19)]:
            nbr = K // qk
            raw = random_blocks(rng, name, N * nbr, size, f16_at)
            wf = np.zeros(N * K, np.float32)
            oracle.oracle(dt, raw.tobytes(), ctypes.c_size_t(N * nbr), wf.ctypes.data_as(ctypes.c_void_p))
            xraw, xd, xq = q8_1_acts(rng, M, K)
            xf = (xq.astype(np.float64) * xd[..., None]).reshape(M, K)
            want = xf @ wf.reshape(N, K).astype(np.float64).T
            for kind in ('mmvq2w_', 'mmvq2x_', 'mmvq2q_', 'mmvqb_'):
                if kind in ('mmvq2x_', 'mmvq2q_') and name == 'q8_0':
                    continue
                if kind == 'mmvqb_' and M > 4:
                    continue
                y = np.full(M * N, np.nan, np.float32)
                getattr(lib, 'run_' + kind + name)(xraw, raw.tobytes(), y.ctypes.data_as(ctypes.c_void_p), M, nbr, N)
                err = np.abs(y.reshape(M, N) - want).max() / max(1e-9, np.abs(want).max())
                good = bool(err < 1e-4)
                ok &= good
                print('%s%-5s M=%d K=%5d N=%d  vs CPU dequant: max rel err %.2e  %s' % (kind, name, M, K, N, err, 'ok' if good else 'FAIL'))
    ok &= check_fused(rng)
    ok &= check_mmq(rng, oracle)
    ok &= check_mmq(rng, oracle, 'mmqt')
    ok &= check_quantize_x4(rng)
    ok &= check_mmqg(rng, oracle)
    ok &= check_attn(rng)
    ok &= check_attn_fa(rng)
    ok &= check_paged(rng)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
