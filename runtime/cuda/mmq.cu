// Quantized matrix x matrix for prompt processing (M > 4 activation rows), dp4a.
// Source of runtime/vira_cuda_mmq_ptx.h (python3 runtime/cuda/build_ptx.py).
//
// y[m, n] = sum_k x[m, k] * W[n, k], W in its GGUF layout (the decode kernels share the
// buffer), x quantized to Q8_1 by vira_cu_quantize_q8_1 (40-byte blocks {f32 d, f32 d*sum,
// int8 q[32]}). Block of 128 threads computes a tile of 64 weight rows x 32 activation rows;
// per step of 128 values (half of a 256-value slice) it decodes the weight slice into shared memory as int8 plus a float
// scale per 16 values and a bias per 32 (so every format meets the same inner loop):
//   y += sum_g  a[n, g] * dx[m, g / 2] * dot16(qw[n, g], qx[m, g])  +  sum_b  c[n, b] * sx[m, b]
// where sx = dx * sum(qx) is the Q8_1 block's second field. Each thread accumulates 4 rows x
// 4 columns. Weights are read from memory once per 32 activation rows (the decode kernel
// would read them once per row), activations once per 64 weight rows.
// Grid ((N + 63) / 64, (M + 31) / 32), block 128.

#define WARP 32
#define MQ_N 64   // weight rows per block
#define MQ_M 32   // activation rows per block
#define MQ_K 128  // values per step: half of a 256-value slice (shared memory ~17.5 KB, 3 blocks/SM)
#define MQ_I (MQ_K / 4)

#ifdef VL_EMULATE
#include "emulate.h"
#else
#define VL_SYNC() __syncthreads()
__device__ __forceinline__ int vl_dp4a(int a, int b, int c) {
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ int vl_load_int_b2(const unsigned char *p) {
  const unsigned short *s = (const unsigned short *)p;
  return (int)((unsigned int)__ldg(s) | ((unsigned int)__ldg(s + 1) << 16));
}
__device__ __forceinline__ float vl_half_to_float(unsigned short h) {
  float f;
  asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
  return f;
}
#endif

__device__ __forceinline__ float vl_h(const unsigned char *p) { return vl_half_to_float(__ldg((const unsigned short *)p)); }
__device__ __forceinline__ int vl_ld4(const unsigned char *p) { return __ldg((const int *)p); }
// bytes q in 0..127 minus o (o in 0..127) as signed bytes, no borrow between bytes
__device__ __forceinline__ int vl_sub_bytes(int q, unsigned int o4) {
  return (int)((((unsigned int)q | 0x80808080u) - o4) ^ 0x80808080u);
}
__device__ __forceinline__ int vl_qh4(unsigned int qh, int base) { // bits base..base+3 -> bit 4 of bytes
  const unsigned int b = (qh >> base) & 0xFu;
  return (int)(((b & 1u) << 4) | ((b & 2u) << 11) | ((b & 4u) << 18) | ((b & 8u) << 25));
}
__device__ __forceinline__ void vl_k4sm(const unsigned char *sc, int j, int *s, int *m) {
  if (j < 4) {
    *s = __ldg(sc + j) & 63;
    *m = __ldg(sc + j + 4) & 63;
  } else {
    *s = (__ldg(sc + j + 4) & 0xF) | ((__ldg(sc + j - 4) >> 6) << 4);
    *m = (__ldg(sc + j + 4) >> 4) | ((__ldg(sc + j) >> 6) << 4);
  }
}

// ---- per format: int i (values 4i..4i+3) of the 256-value slice starting at unit u of a
// row; and the slice's scales a[16] (per 16 values) and bias c[8] (per 32 values) ----
// 32-value formats: the slice is 8 blocks, block u + i / 8, int i % 8 of it.
#define VL_LEGACY(NAME, BYTES, QOFF, BODY, SC)                                               \
  __device__ __forceinline__ int NAME##_q(const unsigned char *row, int u, int i) {          \
    const unsigned char *b = row + (size_t)(u + (i >> 3)) * (BYTES);                         \
    const int ii = i & 7;                                                                    \
    BODY                                                                                     \
  }                                                                                          \
  __device__ __forceinline__ void NAME##_s(const unsigned char *row, int u, float *a, float *c) { \
    _Pragma("unroll") for (int j = 0; j < 8; j++) {                                          \
      const unsigned char *b = row + (size_t)(u + j) * (BYTES);                              \
      SC                                                                                     \
    }                                                                                        \
  }
// Q4_0: values 0..15 low nibbles of qs, 16..31 high; v = d * (q - 8)
VL_LEGACY(mq_q4_0, 18, 2,
  const int q = (vl_load_int_b2(b + 2 + 4 * (ii & 3)) >> ((ii >> 2) * 4)) & 0x0F0F0F0F;
  return vl_sub_bytes(q, 0x08080808u);,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = 0.0f;)
// Q4_1: v = d * q + m
VL_LEGACY(mq_q4_1, 20, 4,
  return (vl_ld4(b + 4 + 4 * (ii & 3)) >> ((ii >> 2) * 4)) & 0x0F0F0F0F;,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = vl_h(b + 2);)
// Q5_0: v = d * ((nibble | qh bit << 4) - 16)
VL_LEGACY(mq_q5_0, 22, 6,
  const int q = ((vl_load_int_b2(b + 6 + 4 * (ii & 3)) >> ((ii >> 2) * 4)) & 0x0F0F0F0F) |
                vl_qh4((unsigned int)vl_load_int_b2(b + 2), 4 * ii);
  return vl_sub_bytes(q, 0x10101010u);,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = 0.0f;)
// Q5_1: v = d * (nibble | qh bit << 4) + m
VL_LEGACY(mq_q5_1, 24, 8,
  return ((vl_ld4(b + 8 + 4 * (ii & 3)) >> ((ii >> 2) * 4)) & 0x0F0F0F0F) |
         vl_qh4((unsigned int)vl_ld4(b + 4), 4 * ii);,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = vl_h(b + 2);)
// Q8_0: v = d * q
VL_LEGACY(mq_q8_0, 34, 2,
  return vl_load_int_b2(b + 2 + 4 * ii);,
  a[2 * j] = a[2 * j + 1] = vl_h(b); c[j] = 0.0f;)

// K-quants: the slice is one super-block (unit u); sub-block j = i / 8 holds values 32j..
// Q4_K (144 B): d, dmin, scales[12], qs[128]; sub-block j = nibble (j & 1) of qs[32 (j/2)..]
__device__ __forceinline__ int mq_q4_k_q(const unsigned char *row, int u, int i) {
  const unsigned char *b = row + (size_t)u * 144u;
  const int j = i >> 3, ii = i & 7;
  return (vl_ld4(b + 16 + 32 * (j >> 1) + 4 * ii) >> (4 * (j & 1))) & 0x0F0F0F0F;
}
__device__ __forceinline__ void mq_q4_k_s(const unsigned char *row, int u, float *a, float *c) {
  const unsigned char *b = row + (size_t)u * 144u;
  const float d = vl_h(b), dm = vl_h(b + 2);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int s, m;
    vl_k4sm(b + 4, j, &s, &m);
    a[2 * j] = a[2 * j + 1] = d * (float)s;
    c[j] = -dm * (float)m;
  }
}
// Q5_K (176 B): d, dmin, scales[12], qh[32], qs[128]; bit j of qh[l] is bit 4 of value l
__device__ __forceinline__ int mq_q5_k_q(const unsigned char *row, int u, int i) {
  const unsigned char *b = row + (size_t)u * 176u;
  const int j = i >> 3, ii = i & 7;
  const int lo = (vl_ld4(b + 48 + 32 * (j >> 1) + 4 * ii) >> (4 * (j & 1))) & 0x0F0F0F0F;
  const int hi = ((vl_ld4(b + 16 + 4 * ii) >> j) & 0x01010101) << 4;
  return lo | hi;
}
__device__ __forceinline__ void mq_q5_k_s(const unsigned char *row, int u, float *a, float *c) {
  const unsigned char *b = row + (size_t)u * 176u;
  const float d = vl_h(b), dm = vl_h(b + 2);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int s, m;
    vl_k4sm(b + 4, j, &s, &m);
    a[2 * j] = a[2 * j + 1] = d * (float)s;
    c[j] = -dm * (float)m;
  }
}
// Q6_K (210 B): ql[128], qh[64], int8 scales[16], d at 208. Sub-block j: half ip = j / 4,
// quarter k = j % 4; value l (0..31): ql[64 ip + 32 (k & 1) + l] nibble (k >> 1), bits 2k of
// qh[32 ip + l]; v = d * scales[8 ip + 2k + l / 16] * (q - 32)
__device__ __forceinline__ int mq_q6_k_q(const unsigned char *row, int u, int i) {
  const unsigned char *b = row + (size_t)u * 210u;
  const int j = i >> 3, ii = i & 7, ip = j >> 2, k = j & 3;
  const int lo = (vl_load_int_b2(b + 64 * ip + 32 * (k & 1) + 4 * ii) >> (4 * (k >> 1))) & 0x0F0F0F0F;
  const int hi = ((vl_load_int_b2(b + 128 + 32 * ip + 4 * ii) >> (2 * k)) & 0x03030303) << 4;
  return vl_sub_bytes(lo | hi, 0x20202020u);
}
__device__ __forceinline__ void mq_q6_k_s(const unsigned char *row, int u, float *a, float *c) {
  const unsigned char *b = row + (size_t)u * 210u;
  const float d = vl_h(b + 208);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int ip = j >> 2, k = j & 3;
    a[2 * j] = d * (float)(signed char)__ldg(b + 192 + 8 * ip + 2 * k);
    a[2 * j + 1] = d * (float)(signed char)__ldg(b + 192 + 8 * ip + 2 * k + 1);
    c[j] = 0.0f;
  }
}

// UPS = units per 256-value step (8 for 32-value blocks, 1 for super-blocks); BYTES per unit.
#define VL_MMQ_KERNEL(NAME, F, UPS, BYTES)                                                     \
  extern "C" __global__ void __launch_bounds__(128)                                           \
  NAME(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,             \
       float *__restrict__ y, int M, int nb, int N) {                                         \
    __shared__ int sW[MQ_N][MQ_I + 1];                                                         \
    __shared__ float sA[MQ_N][9];                                                              \
    __shared__ float sC[MQ_N][5];                                                              \
    __shared__ int sX[MQ_M][MQ_I + 1];                                                         \
    __shared__ float sXd[MQ_M][5];                                                             \
    __shared__ float sXs[MQ_M][5];                                                             \
    const int tid = threadIdx.x;                                                               \
    const int n0 = blockIdx.x * MQ_N, m0 = blockIdx.y * MQ_M;                                  \
    const int rg = tid & 15, cg = tid >> 4; /* rows 4 rg.., columns 4 cg.. */                   \
    const int nunits = nb;                     /* units per row */                             \
    const int nxb = nb * (UPS == 8 ? 1 : 8);   /* Q8_1 blocks per activation row */            \
    float acc[4][4];                                                                           \
    _Pragma("unroll") for (int r = 0; r < 4; r++)                                              \
      _Pragma("unroll") for (int c = 0; c < 4; c++) acc[r][c] = 0.0f;                          \
    for (int step = 0; step < (nunits / UPS) * 2; step++) {                                    \
      const int u = (step >> 1) * UPS; /* first unit of the 256-value slice */                  \
      const int hs = step & 1;          /* which half of it */                                  \
      /* weights: warp k fills rows k, k + 4, ..; lane i reads int i of the row, so a warp  */ \
      /* reads 128 contiguous values (a thread per row made 16 scattered rows per load). */ \
      {                                                                                        \
        const int i = tid & 31;                                                                \
        _Pragma("unroll 4") for (int r = tid >> 5; r < MQ_N; r += 4) {                         \
          const int n = n0 + r;                                                                \
          sW[r][i] = n < N ? F##_q(w + (size_t)n * (size_t)nb * (BYTES), u, 32 * hs + i) : 0;  \
        }                                                                                      \
        if (tid < MQ_N) { /* scales of row tid */                                              \
          const int n = n0 + tid;                                                              \
          if (n < N) {                                                                         \
            float a[16], c[8];                                                                 \
            F##_s(w + (size_t)n * (size_t)nb * (BYTES), u, a, c);                              \
            if (hs) { /* constant indices keep a / c in registers */                           \
              _Pragma("unroll") for (int g = 0; g < 8; g++) sA[tid][g] = a[8 + g];             \
              _Pragma("unroll") for (int g = 0; g < 4; g++) sC[tid][g] = c[4 + g];             \
            } else {                                                                           \
              _Pragma("unroll") for (int g = 0; g < 8; g++) sA[tid][g] = a[g];                 \
              _Pragma("unroll") for (int g = 0; g < 4; g++) sC[tid][g] = c[g];                 \
            }                                                                                  \
          } else {                                                                             \
            for (int g = 0; g < 8; g++) sA[tid][g] = 0.0f;                                     \
            for (int g = 0; g < 4; g++) sC[tid][g] = 0.0f;                                     \
          }                                                                                    \
        }                                                                                      \
      }                                                                                        \
      /* activations: thread fills 8 ints (one Q8_1 block) of column tid / 4 */                \
      {                                                                                        \
        const int c = tid >> 2, part = tid & 3;                                                \
        const int m = m0 + c;                                                                  \
        const int xb0 = (u / UPS) * 8 + 4 * hs; /* first Q8_1 block of this step */           \
        if (m < M) {                                                                           \
          const unsigned char *xb = xq + ((size_t)m * (size_t)nxb + (size_t)xb0 + (size_t)part) * 40u; \
          _Pragma("unroll") for (int i = 0; i < 8; i++) sX[c][8 * part + i] = __ldg((const int *)(xb + 8u) + i); \
          sXd[c][part] = __ldg((const float *)xb);                                             \
          sXs[c][part] = __ldg((const float *)(xb + 4u));                                      \
        } else {                                                                               \
          for (int i = 0; i < 8; i++) sX[c][8 * part + i] = 0;                                 \
          sXd[c][part] = sXs[c][part] = 0.0f;                                                  \
        }                                                                                      \
      }                                                                                        \
      VL_SYNC();                                                                               \
      _Pragma("unroll 2") for (int g = 0; g < 8; g++) { /* 16-value group */                   \
        int dot[4][4];                                                                         \
        _Pragma("unroll") for (int r = 0; r < 4; r++)                                          \
          _Pragma("unroll") for (int c = 0; c < 4; c++) dot[r][c] = 0;                         \
        _Pragma("unroll") for (int i = 0; i < 4; i++) {                                        \
          int wv[4], xv[4];                                                                    \
          _Pragma("unroll") for (int r = 0; r < 4; r++) wv[r] = sW[4 * rg + r][4 * g + i];     \
          _Pragma("unroll") for (int c = 0; c < 4; c++) xv[c] = sX[4 * cg + c][4 * g + i];     \
          _Pragma("unroll") for (int r = 0; r < 4; r++)                                        \
            _Pragma("unroll") for (int c = 0; c < 4; c++) dot[r][c] = vl_dp4a(wv[r], xv[c], dot[r][c]); \
        }                                                                                      \
        _Pragma("unroll") for (int r = 0; r < 4; r++) {                                        \
          const float a = sA[4 * rg + r][g];                                                   \
          _Pragma("unroll") for (int c = 0; c < 4; c++)                                        \
            acc[r][c] += a * sXd[4 * cg + c][g >> 1] * (float)dot[r][c];                       \
        }                                                                                      \
        if (g & 1) { /* the 32-value block's bias term */                                      \
          _Pragma("unroll") for (int r = 0; r < 4; r++) {                                      \
            const float cb = sC[4 * rg + r][g >> 1];                                           \
            _Pragma("unroll") for (int c = 0; c < 4; c++) acc[r][c] += cb * sXs[4 * cg + c][g >> 1]; \
          }                                                                                    \
        }                                                                                      \
      }                                                                                        \
      VL_SYNC();                                                                               \
    }                                                                                          \
    _Pragma("unroll") for (int c = 0; c < 4; c++) {                                            \
      const int m = m0 + 4 * cg + c;                                                           \
      if (m >= M) continue;                                                                    \
      _Pragma("unroll") for (int r = 0; r < 4; r++) {                                          \
        const int n = n0 + 4 * rg + r;                                                         \
        if (n < N) y[(size_t)m * (size_t)N + (size_t)n] = acc[r][c];                           \
      }                                                                                        \
    }                                                                                          \
  }

// nb is in units: 32-value blocks for the first five, super-blocks for the K-quants
// (the same argument the decode kernels get). Rows must hold a whole number of steps:
// K % 256 == 0 (the host checks).
VL_MMQ_KERNEL(vira_cu_mmq_q4_0, mq_q4_0, 8, 18u)
VL_MMQ_KERNEL(vira_cu_mmq_q4_1, mq_q4_1, 8, 20u)
VL_MMQ_KERNEL(vira_cu_mmq_q5_0, mq_q5_0, 8, 22u)
VL_MMQ_KERNEL(vira_cu_mmq_q5_1, mq_q5_1, 8, 24u)
VL_MMQ_KERNEL(vira_cu_mmq_q8_0, mq_q8_0, 8, 34u)
VL_MMQ_KERNEL(vira_cu_mmq_q4_k, mq_q4_k, 1, 144u)
VL_MMQ_KERNEL(vira_cu_mmq_q5_k, mq_q5_k, 1, 176u)
VL_MMQ_KERNEL(vira_cu_mmq_q6_k, mq_q6_k, 1, 210u)
