// Matrix x vector kernels for decode (M <= 4 rows of activations), quantized weights.
//
// Source of runtime/vira_cuda_mmvq2_ptx.h. Regenerate after editing:
//   python3 runtime/cuda/build_ptx.py
// (NVRTC 11.8, target compute_61: dp4a; the PTX loads on any driver from CUDA 11.8 up,
// and the driver compiles it for the GPU it runs on.)
//
// Layout (one CUDA block per output row, 128 threads = 4 warps):
//   - Weights are read in their GGUF layout, untouched (the prefill GEMM kernels share
//     the same device buffer).
//   - Every 4 consecutive threads share one 32-weight block, 8 bytes of quants each, so
//     one warp reads 8 consecutive blocks (a contiguous stretch of the row) per step and
//     128 threads read 32 blocks. The old kernel gave each row 4 threads and each warp 8
//     rows, so a warp read 8 scattered places and one thread did a whole block alone.
//   - Activations come quantized to Q8_1 by vira_cu_quantize_q8_1:
//     40-byte blocks {f32 d, f32 sum, int8 qs[32]}.
//   - Partial sums: warp shuffle, then 4 values through shared memory.
//
// Signature matches vira_cu_mmvq_q8_0, so the host passes the same arguments:
//   (xq8_1, w, y, M, nb = K / 32, N); grid (N, M), block (128, 1, 1).

#define WARP 32
#define THREADS 128

#ifdef VL_EMULATE
#include "emulate.h" // CPU test harness: runtime/cuda/test_emulate.py
#else
#define VL_WRITER 0      // the thread that writes the row's result
#define VL_WARP_WRITER 0 // the lane that writes in the warp-per-row kernel

__device__ __forceinline__ int vl_dp4a(int a, int b, int c) {
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); // sm_61+
  return d;
}

// Two 16-bit loads: a Q8_0 block is 34 bytes, so its quants are only 2-byte aligned.
__device__ __forceinline__ int vl_load_int_b2(const unsigned char *p) {
  const unsigned short *s = (const unsigned short *)p;
  return (int)((unsigned int)__ldg(s) | ((unsigned int)__ldg(s + 1) << 16));
}

__device__ __forceinline__ float vl_half_to_float(unsigned short h) {
  float f;
  asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
  return f;
}

__device__ __forceinline__ float vl_block_reduce(float v) {
  __shared__ float part[THREADS / WARP];
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  const int tid = threadIdx.x;
  if ((tid & (WARP - 1)) == 0) part[tid / WARP] = v;
  __syncthreads();
  float s = 0.0f;
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < THREADS / WARP; i++) s += part[i];
  }
  return s;
}

__device__ __forceinline__ float vl_warp_reduce(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}
__device__ __forceinline__ float vl_warp_reduce2(float v) { return vl_warp_reduce(v); }
#endif // VL_EMULATE

extern "C" __global__ void __launch_bounds__(THREADS)
vira_cu_mmvq2_q8_0(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,
                   float *__restrict__ y, int M, int nb, int N) {
  const int row = blockIdx.x;
  const int m = blockIdx.y;
  if (row >= N || m >= M) return; // the whole block leaves together
  const int tid = threadIdx.x;
  const int part = tid & 3; // which 8 bytes of the block's 32 quants
  const unsigned char *wrow = w + (size_t)row * (size_t)nb * 34u;
  const unsigned char *xrow = xq + (size_t)m * (size_t)nb * 40u;
  float acc = 0.0f;
#pragma unroll 4
  for (int b = tid >> 2; b < nb; b += THREADS / 4) {
    const unsigned char *wb = wrow + (size_t)b * 34u;
    const unsigned char *xb = xrow + (size_t)b * 40u;
    const int w0 = vl_load_int_b2(wb + 2 + part * 8);
    const int w1 = vl_load_int_b2(wb + 6 + part * 8);
    const int *xi = (const int *)(xb + 8 + part * 8); // 40 * b + 8 + 8 * part: 4-byte aligned
    int s = vl_dp4a(w0, __ldg(xi), 0);
    s = vl_dp4a(w1, __ldg(xi + 1), s);
    const float dw = vl_half_to_float(__ldg((const unsigned short *)wb));
    const float dx = __ldg((const float *)xb);
    acc += dw * dx * (float)s;
  }
  const float total = vl_block_reduce(acc);
  if (tid == VL_WRITER) y[(size_t)m * (size_t)N + (size_t)row] = total;
}

// Default since the T4 A/B (Qwen3-8B Q8_0: warp 26 tok/s, row kernel above 25, old 13):
// one warp per output row, 4 rows per block of (32, 4). VIRA_GPU_MMVQ=row selects the one above.
// For K = 4096 a 128-thread row block gives each thread one step; here each thread does
// four, so more loads are in flight per thread. Same arguments; grid ((N + 3) / 4, M).
extern "C" __global__ void __launch_bounds__(THREADS)
vira_cu_mmvq2w_q8_0(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,
                    float *__restrict__ y, int M, int nb, int N) {
  const int row = blockIdx.x * (THREADS / WARP) + threadIdx.y;
  const int m = blockIdx.y;
  if (row >= N || m >= M) return; // a whole warp leaves together; no block barrier below
  const int lane = threadIdx.x;
  const int part = lane & 3;
  const unsigned char *wrow = w + (size_t)row * (size_t)nb * 34u;
  const unsigned char *xrow = xq + (size_t)m * (size_t)nb * 40u;
  float acc = 0.0f;
#pragma unroll 4
  for (int b = lane >> 2; b < nb; b += WARP / 4) {
    const unsigned char *wb = wrow + (size_t)b * 34u;
    const unsigned char *xb = xrow + (size_t)b * 40u;
    const int w0 = vl_load_int_b2(wb + 2 + part * 8);
    const int w1 = vl_load_int_b2(wb + 6 + part * 8);
    const int *xi = (const int *)(xb + 8 + part * 8);
    int s = vl_dp4a(w0, __ldg(xi), 0);
    s = vl_dp4a(w1, __ldg(xi + 1), s);
    const float dw = vl_half_to_float(__ldg((const unsigned short *)wb));
    const float dx = __ldg((const float *)xb);
    acc += dw * dx * (float)s;
  }
  const float total = vl_warp_reduce(acc);
  if (lane == VL_WARP_WRITER) y[(size_t)m * (size_t)N + (size_t)row] = total;
}

// ---------------------------------------------------------------------------------------
// Warp-per-row kernels for the other GGUF formats, same arguments as the old
// vira_cu_mmvq_<fmt> kernels: (xq8_1, w, y, M, nb, N), nb = blocks per row (K / 32 for the
// 32-value formats, K / 256 for the K-quants). Grid ((N + 3) / 4, M), block (32, 4).
// A "unit" is one weight block; TPU threads share a unit and the warp covers 32 / TPU
// units per step, reading them contiguously. Block layouts follow ggml (ggml-common.h).

__device__ __forceinline__ int vl_load_int_b4(const unsigned char *p) { // 4-byte aligned
  return __ldg((const int *)p);
}
__device__ __forceinline__ int vl_sum_bytes(int x) { return vl_dp4a(x, 0x01010101, 0); }
// bits base..base+3 of qh -> bit 4 of bytes 0..3
__device__ __forceinline__ int vl_qh_nibble(unsigned int qh, int base) {
  const unsigned int b = (qh >> base) & 0xFu;
  return (int)(((b & 1u) << 4) | ((b & 2u) << 11) | ((b & 4u) << 18) | ((b & 8u) << 25));
}
__device__ __forceinline__ float vl_xd(const unsigned char *xb) { return __ldg((const float *)xb); }
__device__ __forceinline__ float vl_xs(const unsigned char *xb) { return __ldg((const float *)(xb + 4)); }
__device__ __forceinline__ int vl_xq(const unsigned char *xb, int i) { // int i of the 32 quants
  return __ldg((const int *)(xb + 8) + i);
}

// Each format has a loader (vx_*) for the thread's slice of the Q8_1 activation and a dot
// (vd_*) that takes it, so a two-row warp loads the activation once for both rows.
struct vl_x8 {
  int q[4];   // up to 16 int8 activation values
  float d, s; // Q8_1 block scale and d * sum(q)
};
__device__ __forceinline__ vl_x8 vx_32(const unsigned char *xb, int p) { // 32-value formats
  vl_x8 X;
  X.q[0] = vl_xq(xb, 2 * p);
  X.q[1] = vl_xq(xb, 2 * p + 1);
  X.q[2] = X.q[3] = 0;
  X.d = vl_xd(xb);
  X.s = vl_xs(xb);
  return X;
}

// Q4_0 (18 B): f16 d, qs[16]; v = (nibble - 8) * d; values 0..15 low nibbles, 16..31 high.
// TPU 4: thread part p takes values 8p..8p+7.
__device__ __forceinline__ float vd_q4_0(const unsigned char *wb, const vl_x8 &X, int p) {
  const int sh = (p >> 1) * 4;
  const int q0 = (vl_load_int_b2(wb + 2 + (p & 1) * 8) >> sh) & 0x0F0F0F0F;
  const int q1 = (vl_load_int_b2(wb + 6 + (p & 1) * 8) >> sh) & 0x0F0F0F0F;
  const int dot = vl_dp4a(q1, X.q[1], vl_dp4a(q0, X.q[0], 0));
  const int sx = vl_sum_bytes(X.q[0]) + vl_sum_bytes(X.q[1]);
  return vl_half_to_float(__ldg((const unsigned short *)wb)) * X.d * (float)(dot - 8 * sx);
}
// Q4_1 (20 B): f16 d, f16 m, qs[16]; v = nibble * d + m.
__device__ __forceinline__ float vd_q4_1(const unsigned char *wb, const vl_x8 &X, int p) {
  const int sh = (p >> 1) * 4;
  const int q0 = (vl_load_int_b4(wb + 4 + (p & 1) * 8) >> sh) & 0x0F0F0F0F;
  const int q1 = (vl_load_int_b4(wb + 8 + (p & 1) * 8) >> sh) & 0x0F0F0F0F;
  const int dot = vl_dp4a(q1, X.q[1], vl_dp4a(q0, X.q[0], 0));
  float r = vl_half_to_float(__ldg((const unsigned short *)wb)) * X.d * (float)dot;
  if (p == 0) r += vl_half_to_float(__ldg((const unsigned short *)wb + 1)) * X.s;
  return r;
}
// Q5_0 (22 B): f16 d, u32 qh, qs[16]; v = (nibble | qh bit << 4) - 16, times d.
__device__ __forceinline__ float vd_q5_0(const unsigned char *wb, const vl_x8 &X, int p) {
  const int sh = (p >> 1) * 4;
  const unsigned int qh = (unsigned int)vl_load_int_b2(wb + 2);
  const int q0 = ((vl_load_int_b2(wb + 6 + (p & 1) * 8) >> sh) & 0x0F0F0F0F) | vl_qh_nibble(qh, 8 * p);
  const int q1 = ((vl_load_int_b2(wb + 10 + (p & 1) * 8) >> sh) & 0x0F0F0F0F) | vl_qh_nibble(qh, 8 * p + 4);
  const int dot = vl_dp4a(q1, X.q[1], vl_dp4a(q0, X.q[0], 0));
  const int sx = vl_sum_bytes(X.q[0]) + vl_sum_bytes(X.q[1]);
  return vl_half_to_float(__ldg((const unsigned short *)wb)) * X.d * (float)(dot - 16 * sx);
}
// Q5_1 (24 B): f16 d, f16 m, u32 qh, qs[16]; v = (nibble | qh bit << 4) * d + m.
__device__ __forceinline__ float vd_q5_1(const unsigned char *wb, const vl_x8 &X, int p) {
  const int sh = (p >> 1) * 4;
  const unsigned int qh = (unsigned int)vl_load_int_b4(wb + 4);
  const int q0 = ((vl_load_int_b4(wb + 8 + (p & 1) * 8) >> sh) & 0x0F0F0F0F) | vl_qh_nibble(qh, 8 * p);
  const int q1 = ((vl_load_int_b4(wb + 12 + (p & 1) * 8) >> sh) & 0x0F0F0F0F) | vl_qh_nibble(qh, 8 * p + 4);
  const int dot = vl_dp4a(q1, X.q[1], vl_dp4a(q0, X.q[0], 0));
  float r = vl_half_to_float(__ldg((const unsigned short *)wb)) * X.d * (float)dot;
  if (p == 0) r += vl_half_to_float(__ldg((const unsigned short *)wb + 1)) * X.s;
  return r;
}

// K-quant 6-bit scale and min of sub-block j (ggml get_scale_min_k4).
__device__ __forceinline__ void vl_k4_scale_min(const unsigned char *sc, int j, int *s, int *m) {
  if (j < 4) {
    *s = __ldg(sc + j) & 63;
    *m = __ldg(sc + j + 4) & 63;
  } else {
    *s = (__ldg(sc + j + 4) & 0xF) | ((__ldg(sc + j - 4) >> 6) << 4);
    *m = (__ldg(sc + j + 4) >> 4) | ((__ldg(sc + j) >> 6) << 4);
  }
}
// xs = the 8 Q8_1 blocks of one super-block; thread t of 16 reads 16 values of one block.
// Q4_K / Q5_K: block t / 2, values 16 (t & 1) ..; Q6_K: block 4 (t / 8) + (t / 2) % 4, same half.
__device__ __forceinline__ vl_x8 vx_k(const unsigned char *xs, int blk, int h) {
  const unsigned char *xb = xs + blk * 40;
  vl_x8 X;
#pragma unroll
  for (int i = 0; i < 4; i++) X.q[i] = vl_xq(xb, 4 * h + i);
  X.d = vl_xd(xb);
  X.s = vl_xs(xb);
  return X;
}
__device__ __forceinline__ vl_x8 vx_q4_k(const unsigned char *xs, int t) { return vx_k(xs, t >> 1, t & 1); }
__device__ __forceinline__ vl_x8 vx_q6_k(const unsigned char *xs, int t) {
  return vx_k(xs, 4 * (t >> 3) + ((t >> 1) & 3), t & 1);
}
// Q4_K (144 B): f16 d, f16 dmin, scales[12], qs[128]; 8 sub-blocks of 32, sub-block j uses
// the low (j even) or high (j odd) nibbles of qs[32 * (j / 2) ...]; v = d*s*q - dmin*m.
__device__ __forceinline__ float vd_q4_k(const unsigned char *wb, const vl_x8 &X, int t) {
  const int j = t >> 1, h = t & 1;
  const unsigned char *q = wb + 16 + (j >> 1) * 32 + h * 16;
  const int sh = (j & 1) * 4;
  int dot = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) dot = vl_dp4a((vl_load_int_b4(q + 4 * i) >> sh) & 0x0F0F0F0F, X.q[i], dot);
  int sc, mn;
  vl_k4_scale_min(wb + 4, j, &sc, &mn);
  const unsigned int dd = (unsigned int)vl_load_int_b4(wb);
  float r = vl_half_to_float((unsigned short)(dd & 0xFFFF)) * (float)sc * X.d * (float)dot;
  if (h == 0) r -= vl_half_to_float((unsigned short)(dd >> 16)) * (float)mn * X.s;
  return r;
}
// Q5_K (176 B): f16 d, f16 dmin, scales[12], qh[32], qs[128]; as Q4_K plus bit j of qh[l]
// as bit 4 of value l of sub-block j.
__device__ __forceinline__ float vd_q5_k(const unsigned char *wb, const vl_x8 &X, int t) {
  const int j = t >> 1, h = t & 1;
  const unsigned char *q = wb + 48 + (j >> 1) * 32 + h * 16;
  const unsigned char *qh = wb + 16 + h * 16;
  const int sh = (j & 1) * 4;
  int dot = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int lo = (vl_load_int_b4(q + 4 * i) >> sh) & 0x0F0F0F0F;
    const int hi = ((vl_load_int_b4(qh + 4 * i) >> j) & 0x01010101) << 4;
    dot = vl_dp4a(lo | hi, X.q[i], dot);
  }
  int sc, mn;
  vl_k4_scale_min(wb + 4, j, &sc, &mn);
  const unsigned int dd = (unsigned int)vl_load_int_b4(wb);
  float r = vl_half_to_float((unsigned short)(dd & 0xFFFF)) * (float)sc * X.d * (float)dot;
  if (h == 0) r -= vl_half_to_float((unsigned short)(dd >> 16)) * (float)mn * X.s;
  return r;
}
// Q6_K (210 B): ql[128], qh[64], int8 scales[16], f16 d at 208. Half ip (128 values) uses
// ql[64 ip ..], qh[32 ip ..], scales[8 ip ..]; its quarter k (32 values, Q8_1 block 4 ip + k)
// takes ql[(k & 1) * 32 + l] low (k < 2) or high nibble, bits 2k..2k+1 of qh[l], scale
// 2k + l / 16; v = d * scale * (q - 32). TPU 16: thread t = (ip, k, h), 16 values each.
__device__ __forceinline__ float vd_q6_k(const unsigned char *wb, const vl_x8 &X, int t) {
  const int ip = t >> 3, k = (t >> 1) & 3, h = t & 1;
  const unsigned char *ql = wb + 64 * ip + (k & 1) * 32 + h * 16;
  const unsigned char *qh = wb + 128 + 32 * ip + h * 16;
  const int shl = (k >> 1) * 4, shh = 2 * k;
  int dot = 0, sx = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int lo = (vl_load_int_b2(ql + 4 * i) >> shl) & 0x0F0F0F0F;
    const int hi = ((vl_load_int_b2(qh + 4 * i) >> shh) & 0x03030303) << 4;
    dot = vl_dp4a(lo | hi, X.q[i], dot);
    sx += vl_sum_bytes(X.q[i]);
  }
  const float sc = (float)(signed char)__ldg(wb + 192 + 8 * ip + 2 * k + h);
  const float d = vl_half_to_float(__ldg((const unsigned short *)(wb + 208)));
  return d * sc * X.d * (float)(dot - 32 * sx);
}

// TPU threads per unit, BYTES per unit, XSTRIDE = Q8_1 bytes per unit.
#define VL_MMVQ_WARP_KERNEL(NAME, VX, VD, TPU, BYTES, XSTRIDE)                                     \
  extern "C" __global__ void __launch_bounds__(THREADS)                                       \
  NAME(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,             \
       float *__restrict__ y, int M, int nb, int N) {                                         \
    const int row = blockIdx.x * (THREADS / WARP) + threadIdx.y;                               \
    const int m = blockIdx.y;                                                                  \
    if (row >= N || m >= M) return;                                                            \
    const int lane = threadIdx.x;                                                              \
    const int t = lane % (TPU);                                                                \
    const unsigned char *wrow = w + (size_t)row * (size_t)nb * (BYTES);                        \
    const unsigned char *xrow = xq + (size_t)m * (size_t)nb * (XSTRIDE);                       \
    float acc = 0.0f;                                                                          \
    _Pragma("unroll 2") for (int b = lane / (TPU); b < nb; b += WARP / (TPU)) {                \
      const vl_x8 X = VX(xrow + (size_t)b * (XSTRIDE), t);                                     \
      acc += VD(wrow + (size_t)b * (BYTES), X, t);                                             \
    }                                                                                          \
    const float total = vl_warp_reduce(acc);                                                   \
    if (lane == VL_WARP_WRITER) y[(size_t)m * (size_t)N + (size_t)row] = total;                \
  }

// Two rows per warp (VIRA_GPU_MMVQ=w1 keeps one): both rows read the same Q8_1 activation
// block, so it is loaded once (the second VD call's loads repeat read-only addresses and
// are merged). Q4_K weights are 18 bytes per 32 values but the activation is 40, so with
// one row per warp the activation re-reads outweighed the weights (T4, Qwen3-14B Q4_K_M:
// ~51% of bandwidth). Grid ((N + 7) / 8, M), block (32, 4).
#define VL_MMVQ_WARP2_KERNEL(NAME, VX, VD, TPU, BYTES, XSTRIDE)                                    \
  extern "C" __global__ void __launch_bounds__(THREADS)                                       \
  NAME(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,             \
       float *__restrict__ y, int M, int nb, int N) {                                         \
    const int row0 = (blockIdx.x * (THREADS / WARP) + threadIdx.y) * 2;                        \
    const int m = blockIdx.y;                                                                  \
    if (row0 >= N || m >= M) return;                                                           \
    const int lane = threadIdx.x;                                                              \
    const int t = lane % (TPU);                                                                \
    const bool two = row0 + 1 < N;                                                             \
    const unsigned char *w0 = w + (size_t)row0 * (size_t)nb * (BYTES);                         \
    const unsigned char *w1 = two ? w0 + (size_t)nb * (BYTES) : w0;                            \
    const unsigned char *xrow = xq + (size_t)m * (size_t)nb * (XSTRIDE);                       \
    float acc0 = 0.0f, acc1 = 0.0f;                                                            \
    _Pragma("unroll 2") for (int b = lane / (TPU); b < nb; b += WARP / (TPU)) {                \
      const vl_x8 X = VX(xrow + (size_t)b * (XSTRIDE), t);                                     \
      acc0 += VD(w0 + (size_t)b * (BYTES), X, t);                                              \
      acc1 += VD(w1 + (size_t)b * (BYTES), X, t);                                              \
    }                                                                                          \
    const float t0 = vl_warp_reduce(acc0);                                                     \
    const float t1 = vl_warp_reduce2(acc1);                                                    \
    if (lane == VL_WARP_WRITER) {                                                              \
      y[(size_t)m * (size_t)N + (size_t)row0] = t0;                                            \
      if (two) y[(size_t)m * (size_t)N + (size_t)row0 + 1] = t1;                               \
    }                                                                                          \
  }

VL_MMVQ_WARP2_KERNEL(vira_cu_mmvq2x_q4_0, vx_32, vd_q4_0, 4, 18u, 40u)
VL_MMVQ_WARP2_KERNEL(vira_cu_mmvq2x_q4_1, vx_32, vd_q4_1, 4, 20u, 40u)
VL_MMVQ_WARP2_KERNEL(vira_cu_mmvq2x_q5_0, vx_32, vd_q5_0, 4, 22u, 40u)
VL_MMVQ_WARP2_KERNEL(vira_cu_mmvq2x_q5_1, vx_32, vd_q5_1, 4, 24u, 40u)
VL_MMVQ_WARP2_KERNEL(vira_cu_mmvq2x_q4_k, vx_q4_k, vd_q4_k, 16, 144u, 320u)
VL_MMVQ_WARP2_KERNEL(vira_cu_mmvq2x_q5_k, vx_q4_k, vd_q5_k, 16, 176u, 320u)
VL_MMVQ_WARP2_KERNEL(vira_cu_mmvq2x_q6_k, vx_q6_k, vd_q6_k, 16, 210u, 320u)

VL_MMVQ_WARP_KERNEL(vira_cu_mmvq2w_q4_0, vx_32, vd_q4_0, 4, 18u, 40u)
VL_MMVQ_WARP_KERNEL(vira_cu_mmvq2w_q4_1, vx_32, vd_q4_1, 4, 20u, 40u)
VL_MMVQ_WARP_KERNEL(vira_cu_mmvq2w_q5_0, vx_32, vd_q5_0, 4, 22u, 40u)
VL_MMVQ_WARP_KERNEL(vira_cu_mmvq2w_q5_1, vx_32, vd_q5_1, 4, 24u, 40u)
VL_MMVQ_WARP_KERNEL(vira_cu_mmvq2w_q4_k, vx_q4_k, vd_q4_k, 16, 144u, 320u)
VL_MMVQ_WARP_KERNEL(vira_cu_mmvq2w_q5_k, vx_q4_k, vd_q5_k, 16, 176u, 320u)
VL_MMVQ_WARP_KERNEL(vira_cu_mmvq2w_q6_k, vx_q6_k, vd_q6_k, 16, 210u, 320u)
