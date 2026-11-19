// nvrtc-arch: compute_75
// Quantized matrix x matrix for prompt processing on int8 tensor cores (Turing and newer).
// Source of runtime/vira_cuda_mmqt_ptx.h (python3 runtime/cuda/build_ptx.py). Same contract
// as mmq.cu (the dp4a kernel, kept for Pascal and as VIRA_GPU_MMQ=1):
//
// y[m, n] = sum_k x[m, k] * W[n, k], W in its GGUF layout, x quantized to Q8_1 (40-byte
// blocks {f32 d, f32 d*sum, int8 q[32]}). The weight slice is decoded into shared memory as
// int8 plus a scale per 16 values and a bias per 32 (mmq_fmt.cuh), then
//   y += sum_g a[n, g] * dx[m, g / 2] * dot16(qw[n, g], qx[m, g]) + sum_b c[n, b] * sx[m, b]
// with each dot16 done by mma.sync.m8n8k16.s8 (8 weight rows x 8 activation rows x 16 values
// per warp instruction, 8x the work of a dp4a per thread). Formats with one scale per 32
// values add two mma into the same int32 before the float step; Q6_K (a scale per 16) does
// the float step after each.
//
// Block: 128 threads (4 warps), tile 64 weight rows x 64 activation rows; warp w computes
// rows 32 (w & 1).. x activation rows 32 (w >> 1).. as 4 x 4 mma tiles. Weights are read
// from memory once per 64 activation rows (twice less often than mmq.cu).
//
// Pipelining: the raw words of step s + 1 (weights, their scales, activations) are loaded
// into registers before step s is computed and decoded into shared memory after it, so the
// memory latency overlaps the mma work instead of stalling the block twice per step.
// int32 -> float without I2F (a quarter-rate instruction on Turing): the mma accumulators
// start at 0x4B400000, the bits of 1.5 * 2^23, so for |dot| < 2^22 (here |dot| <= 32 * 128
// * 127) the int bits read as a float are exactly 1.5 * 2^23 + dot, and one FADD removes it.
// Grid ((N + 63) / 64, (M + 63) / 64), block 128.
//
// Fragment layout of mma.m8n8k16.row.col.s32.s8.s8.s32 (PTX ISA; the same as llama.cpp's
// tile<8, 4, int> / tile<16, 8, int> get_i / get_j in ggml-cuda/mma.cuh), lane = 4 gr + tg:
//   A (8 x 16, row):  the lane holds row gr, values 4 tg..4 tg + 3 (one int)
//   B (16 x 8, col):  the lane holds column gr, values 4 tg..4 tg + 3 (one int)
//   D (8 x 8, s32):   the lane holds row gr, columns 2 tg and 2 tg + 1
// Rows of A are weight rows and columns of B activation rows, so both fragments are "int
// 4 g + tg of row gr" of the shared tiles. A shared row stores int 4 g + tg at 8 tg + g, so
// the lane's ints for g = 2 kb and 2 kb + 1 are adjacent (one 8-byte load per row and kb);
// rows are 34 ints apart (2 mod 32), so a half-warp's 8-byte loads hit all 32 banks once.
//
// Loads (T4 profile: the issue rate was limited by the number of load instructions): a
// thread loads 32 values of a weight row per job with 16-byte loads (Q4_K, Q5_K: the host
// keeps 16-byte aligned weights), aligned 4-byte words + funnel shifts (Q6_K, Q8_0, whose
// blocks are only 2-byte aligned), or int by int (the other formats). Activations come in
// the x4 layout of vira_cu_quantize_q8_1x4 (below): per row and 128 values {f32 d[4], f32
// s[4], int8 q[128]}, the same numbers as four Q8_1 blocks, so a row's step is 160
// contiguous 16-byte aligned bytes: 10 16-byte loads.

#define TQ_N 64   // weight rows per block
#define TQ_M 64   // activation rows per block
#define TQ_K 128  // values per step (half of a 256-value slice)
#define TQ_I (TQ_K / 4)
#define TQ_S 34   // shared row stride in ints (34 = 2 mod 32: conflict-free 8-byte fragment loads)

#include "mmq_fmt.cuh"

#ifndef VL_EMULATE
// d0, d1 += A x B for the lane's two D elements (warp-wide; every lane must call it)
__device__ __forceinline__ void vl_mma(int &d0, int &d1, int a, int b) {
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
      : "+r"(d0), "+r"(d1)
      : "r"(a), "r"(b));
}
// d0, d1 = A x B + c (c in both): the first mma of a group starts at the magic constant
// without a register move per accumulator
__device__ __forceinline__ void vl_mma_c(int &d0, int &d1, int a, int b, int c) {
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %4};"
      : "=r"(d0), "=r"(d1)
      : "r"(a), "r"(b), "r"(c));
}
#endif

#define VL_MAGIC 0x4B400000
#define VL_MAGIC_F 12582912.0f

#ifndef VL_EMULATE
__device__ __forceinline__ void vl_lds2(const int *p, int *o) {
  const int2 v = *(const int2 *)p;
  o[0] = v.x;
  o[1] = v.y;
}
__device__ __forceinline__ void vl_lds2f(const float *p, float *o) {
  const float2 v = *(const float2 *)p;
  o[0] = v.x;
  o[1] = v.y;
}
__device__ __forceinline__ void vl_sts2(int *p, int v0, int v1) {
  int2 v;
  v.x = v0;
  v.y = v1;
  *(int2 *)p = v;
}
__device__ __forceinline__ int vl_shfl_xor_i(int v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
#else
static inline void vl_lds2(const int *p, int *o) { memcpy(o, p, 8); }
static inline void vl_lds2f(const float *p, float *o) { memcpy(o, p, 8); }
static inline void vl_sts2(int *p, int v0, int v1) {
  p[0] = v0;
  p[1] = v1;
}
#endif
// 4x4 butterfly transpose across 4 consecutive lanes (c = lane & 3): lane c starts with
// (A[c][0..3]) and ends with (A[0..3][c]). Lets 4 lanes writing one shared row issue four
// conflict-free 8-byte stores at offsets 8*c + 0,2,4,6 (same bank-free layout as vl_lds2).
__device__ __forceinline__ void vl_tr4(int &v0, int &v1, int &v2, int &v3, int c) {
  const int b0 = c & 1;
  const int s1 = vl_shfl_xor_i(b0 ? v0 : v1, 1);
  const int s3 = vl_shfl_xor_i(b0 ? v2 : v3, 1);
  if (b0) { v0 = s1; v2 = s3; } else { v1 = s1; v3 = s3; }
  const int b1 = (c >> 1) & 1;
  const int t2 = vl_shfl_xor_i(b1 ? v0 : v2, 2);
  const int t3 = vl_shfl_xor_i(b1 ? v1 : v3, 2);
  if (b1) { v0 = t2; v1 = t3; } else { v2 = t2; v3 = t3; }
}
// Store 8 decoded ints of one shared row (4 lanes c = 0..3 per row) as four 8-byte STS.64
// with zero bank conflicts. ST: 0 = Q4_0..Q8_0 & xq, 1 = Q4_K/Q5_K, 2 = Q6_K.
template <int ST>
__device__ __forceinline__ void vl_store_row(int *row, int c, int *o) {
  vl_tr4(o[0], o[1], o[2], o[3], c);
  vl_tr4(o[4], o[5], o[6], o[7], c);
  if (ST == 0) {
    vl_sts2(row + 8 * c + 0, o[0], o[4]);
    vl_sts2(row + 8 * c + 2, o[1], o[5]);
    vl_sts2(row + 8 * c + 4, o[2], o[6]);
    vl_sts2(row + 8 * c + 6, o[3], o[7]);
  } else if (ST == 1) {
    vl_sts2(row + 8 * c + 0, o[0], o[1]);
    vl_sts2(row + 8 * c + 2, o[4], o[5]);
    vl_sts2(row + 8 * c + 4, o[2], o[3]);
    vl_sts2(row + 8 * c + 6, o[6], o[7]);
  } else {
    vl_sts2(row + 8 * c + 0, o[0], o[1]);
    vl_sts2(row + 8 * c + 2, o[2], o[3]);
    vl_sts2(row + 8 * c + 4, o[4], o[5]);
    vl_sts2(row + 8 * c + 6, o[6], o[7]);
  }
}
__device__ __forceinline__ void vl_ldg16(const unsigned char *p, int *o) { // p 16-byte aligned
  const int4 v = __ldg((const int4 *)p);
  o[0] = v.x;
  o[1] = v.y;
  o[2] = v.z;
  o[3] = v.w;
}
// n + 1 aligned words from p rounded down to 4; o[n + 1] = bit shift of p within its word
template <int NW>
__device__ __forceinline__ void vl_ldg_words(const unsigned char *p, int *o) {
  const unsigned char *q = (const unsigned char *)((size_t)p & ~(size_t)3);
#pragma unroll
  for (int k = 0; k < NW; k++) o[k] = __ldg((const int *)(q + 4 * k));
  o[NW] = 8 * (int)((size_t)p & 3);
}
// Q8_1 in the x4 layout: grid (K / 128, M), block 128; warp w quantizes values
// 32 w .. 32 w + 31 of the 128-value group, with the arithmetic of vira_cu_quantize_q8_1
// (warp absmax, q = rni(v * (127 / amax)) clamped to +-127, d = amax * (1 / 127), s = d *
// sum(q)), so the numbers equal its 40-byte blocks.
#ifndef VL_EMULATE
__device__ __forceinline__ float vl_wmaxf(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ int vl_wsumi(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
#endif
extern "C" __global__ void __launch_bounds__(128)
vira_cu_quantize_q8_1x4(const float *__restrict__ x, unsigned char *__restrict__ q, int K) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = blockIdx.x, m = blockIdx.y;
  const int ng = K / 128;
  const float v = __ldg(x + (size_t)m * (size_t)K + (size_t)g * 128u + (size_t)(32 * w + lane));
  const float amax = vl_wmaxf(fabsf(v));
  int qi = 0;
  if (amax > 0.0f) {
    qi = __float2int_rn(__fmul_rn(v, __fdiv_rn(127.0f, amax)));
    qi = max(min(qi, 127), -127);
  }
  const int sum = vl_wsumi(qi);
  unsigned char *rec = q + ((size_t)m * (size_t)ng + (size_t)g) * 160u;
  rec[32 + 32 * w + lane] = (unsigned char)(signed char)qi;
  if (lane == 0) {
    const float d = __fmul_rn(amax, __int_as_float(0x3C010204));
    ((float *)rec)[w] = d;
    ((float *)rec)[4 + w] = __fmul_rn(d, (float)sum);
  }
}

// slice-local int i (0..31 of a 128-value step) -> its place in a shared row
__device__ __forceinline__ int vl_perm(int i) { return (i & 3) * 8 + (i >> 2); }

// ---- per format: a job is 32 values (8 ints) of one weight row in one step (half hs of
// the 256-value slice at unit u), chunk c = 0..3. vload: raw words (RAW of them); vdec: the
// 8 ints; vpos(c, t): their places 0..31 in the step. sload / sdec: the scales of 32-value
// sub-block j (0..7) of the slice, as in mmq_fmt.cuh. ----
// UPS = units per slice; BYTES per unit; S16 = a scale per 16 values; HASC = has a bias;
// A16 = needs 16-byte aligned weights.
#define VL_SCALES(F)                                                                           \
  static __device__ __forceinline__ void sload(const unsigned char *row, int u, int j, int *r) { F##_sr(row, u, j, r); } \
  static __device__ __forceinline__ void sdec(const int *r, int j, float *a0, float *a1, float *c) { F##_sd(r, j, a0, a1, c); }
// formats loaded int by int (mmq_fmt.cuh decoders): chunk c = ints 8 c .. 8 c + 7
#define VL_TRAITS_INTS(T, F, BYTES_, HASC_)                                                    \
  struct T {                                                                                   \
    enum { UPS = 8, BYTES = BYTES_, S16 = 0, HASC = HASC_, A16 = 0, RAW = 16, ST = 0 };        \
    static __device__ __forceinline__ void vload(const unsigned char *row, int u, int hs, int c, int *r) { \
      _Pragma("unroll") for (int t = 0; t < 8; t++) F##_qr(row, u, 32 * hs + 8 * c + t, r + 2 * t); \
    }                                                                                          \
    static __device__ __forceinline__ void vdec(const int *r, int hs, int c, int *o) {         \
      _Pragma("unroll") for (int t = 0; t < 8; t++) o[t] = F##_qd(r + 2 * t, 32 * hs + 8 * c + t); \
    }                                                                                          \
    static __device__ __forceinline__ int vpos(int c, int t) { return 8 * c + t; }             \
    VL_SCALES(F)                                                                               \
  };
VL_TRAITS_INTS(vt_q4_0, mq_q4_0, 18, 0)
VL_TRAITS_INTS(vt_q4_1, mq_q4_1, 20, 1)
VL_TRAITS_INTS(vt_q5_0, mq_q5_0, 22, 0)
VL_TRAITS_INTS(vt_q5_1, mq_q5_1, 24, 1)

// Q8_0: chunk c is block u + 4 hs + c {f16 d, int8 qs[32]} (34 bytes, 2-byte aligned): 9
// aligned words; qs int t starts 2 + 4 t bytes after the block
struct vt_q8_0 {
  enum { UPS = 8, BYTES = 34, S16 = 0, HASC = 0, A16 = 0, RAW = 10, ST = 0 };
  static __device__ __forceinline__ void vload(const unsigned char *row, int u, int hs, int c, int *r) {
    vl_ldg_words<9>(row + (size_t)(u + 4 * hs + c) * 34u, r);
  }
  static __device__ __forceinline__ void vdec(const int *r, int hs, int c, int *o) {
    (void)hs;
    (void)c;
    const bool odd = r[9] != 0; // block at 2 mod 4: qs int t is word t + 1
#pragma unroll
    for (int t = 0; t < 8; t++) o[t] = odd ? r[t + 1] : __funnelshift_r(r[t], r[t + 1], 16);
  }
  static __device__ __forceinline__ int vpos(int c, int t) { return 8 * c + t; }
  VL_SCALES(mq_q8_0)
};
// Q4_K: qs bytes 64 hs + 16 c .. +15 of the super-block; low nibbles are sub-block
// 2 (2 hs + c / 2), high nibbles the next one
struct vt_q4_k {
  enum { UPS = 1, BYTES = 144, S16 = 0, HASC = 1, A16 = 1, RAW = 4, ST = 1 };
  static __device__ __forceinline__ void vload(const unsigned char *row, int u, int hs, int c, int *r) {
    vl_ldg16(row + (size_t)u * 144u + 16 + 64 * hs + 16 * c, r);
  }
  static __device__ __forceinline__ void vdec(const int *r, int hs, int c, int *o) {
    (void)hs;
    (void)c;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      o[t] = r[t] & 0x0F0F0F0F;
      o[4 + t] = (r[t] >> 4) & 0x0F0F0F0F;
    }
  }
  static __device__ __forceinline__ int vpos(int c, int t) { return 8 * (2 * (c >> 1) + (t >> 2)) + 4 * (c & 1) + (t & 3); }
  static __device__ __forceinline__ void sload(const unsigned char *row, int u, int j, int *r) {
    (void)j;
    vl_ldg16(row + (size_t)u * 144u, r);
  }
  static __device__ __forceinline__ void sdec(const int *r, int j, float *a0, float *a1, float *c) { vl_k4_sd(r, j, a0, a1, c); }
};
// Q5_K: as Q4_K with qs at 48; bit j of qh[l] (qh at 16) is bit 4 of value l of sub-block j
struct vt_q5_k {
  enum { UPS = 1, BYTES = 176, S16 = 0, HASC = 1, A16 = 1, RAW = 8, ST = 1 };
  static __device__ __forceinline__ void vload(const unsigned char *row, int u, int hs, int c, int *r) {
    const unsigned char *b = row + (size_t)u * 176u;
    vl_ldg16(b + 48 + 64 * hs + 16 * c, r);
    vl_ldg16(b + 16 + 16 * (c & 1), r + 4);
  }
  static __device__ __forceinline__ void vdec(const int *r, int hs, int c, int *o) {
    const int j = 2 * (2 * hs + (c >> 1));
#pragma unroll
    for (int t = 0; t < 4; t++) {
      o[t] = (r[t] & 0x0F0F0F0F) | (((r[4 + t] >> j) & 0x01010101) << 4);
      o[4 + t] = ((r[t] >> 4) & 0x0F0F0F0F) | (((r[4 + t] >> (j + 1)) & 0x01010101) << 4);
    }
  }
  static __device__ __forceinline__ int vpos(int c, int t) { return 8 * (2 * (c >> 1) + (t >> 2)) + 4 * (c & 1) + (t & 3); }
  static __device__ __forceinline__ void sload(const unsigned char *row, int u, int j, int *r) {
    (void)j;
    vl_ldg16(row + (size_t)u * 176u, r);
  }
  static __device__ __forceinline__ void sdec(const int *r, int j, float *a0, float *a1, float *c) { vl_k4_sd(r, j, a0, a1, c); }
};
// Q6_K (210 bytes, 2-byte aligned): ql bytes 64 hs + 16 c .. +15 (low nibbles: quarter
// k0 = c / 2, high: k0 + 2), qh bytes 128 + 32 hs + 16 (c & 1) .. +15 (bits 2 k)
struct vt_q6_k {
  enum { UPS = 1, BYTES = 210, S16 = 1, HASC = 0, A16 = 0, RAW = 12, ST = 2 };
  static __device__ __forceinline__ void vload(const unsigned char *row, int u, int hs, int c, int *r) {
    const unsigned char *b = row + (size_t)u * 210u;
    vl_ldg_words<5>(b + 64 * hs + 16 * c, r);
    vl_ldg_words<5>(b + 128 + 32 * hs + 16 * (c & 1), r + 6);
  }
  static __device__ __forceinline__ void vdec(const int *r, int hs, int c, int *o) {
    (void)hs;
    const int k0 = c >> 1;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      const int ql = __funnelshift_r(r[t], r[t + 1], r[5]);
      const int qh = __funnelshift_r(r[6 + t], r[7 + t], r[11]);
      o[t] = vl_sub_bytes((ql & 0x0F0F0F0F) | (((qh >> (2 * k0)) & 0x03030303) << 4), 0x20202020u);
      o[4 + t] = vl_sub_bytes(((ql >> 4) & 0x0F0F0F0F) | (((qh >> (2 * k0 + 4)) & 0x03030303) << 4), 0x20202020u);
    }
  }
  static __device__ __forceinline__ int vpos(int c, int t) { return 8 * ((c >> 1) + 2 * (t >> 2)) + 4 * (c & 1) + (t & 3); }
  VL_SCALES(mq_q6_k)
  // the int8 scales of the sub-block's two halves and the super-block's d (vl_mmqb's integer path)
  static __device__ __forceinline__ void sdec_int(const int *r, int *sc0, int *sc1, float *d) {
    *sc0 = (int)(signed char)(r[0] & 0xFF);
    *sc1 = (int)(signed char)((r[0] >> 8) & 0xFF);
    *d = vl_half_to_float((unsigned short)r[1]);
  }
};

template <class F>
__device__ __forceinline__ void vl_mmqt(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,
                                        float *__restrict__ y, int M, int nb, int N) {
  __shared__ __align__(16) int sW[TQ_N][TQ_S];
  __shared__ __align__(16) int sX[TQ_M][TQ_S];
  // scales: per 16 values (Q6_K) or per 32; biases per 32. Strides 9 / 5 / 4 keep the 8
  // rows a warp reads apart in the banks, and the total under 21.8 KB: 3 blocks per SM.
  __shared__ float sA[TQ_N][F::S16 ? 9 : 5];
  __shared__ float sC[TQ_N][4];
  __shared__ __align__(16) float sXd[4][TQ_M]; // [32-value block][activation row]
  __shared__ __align__(16) float sXs[4][TQ_M];
  const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  const int gr = lane >> 2, tg = lane & 3;
  const int wn = (wp & 1) * 32, wm = (wp >> 1) * 32;
  const int m0 = blockIdx.x * TQ_M, n0 = blockIdx.y * TQ_N;
  const int nxb = nb * (F::UPS == 8 ? 1 : 8); // Q8_1 blocks per activation row
  const size_t rowb = (size_t)nb * (size_t)F::BYTES;
  const int nsteps = (nb / F::UPS) * 2;
  // registers holding the next step's raw words: 2 weight jobs (row, chunk), 2 scale jobs
  // (row, sub-block), 1 activation scale job (tid < 64: d[4], tid >= 64: s[4]), 2 activation q rows
  int wr[2][F::RAW];
  int sr[2][4];
  int xsr[4];
  int xqr[2][8];
  const int ngx = nxb / 4; // 128-value records per activation row
  float acc[4][4][2];
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int b = 0; b < 4; b++) acc[a][b][0] = acc[a][b][1] = 0.0f;

#define VL_MMQT_LOAD(STEP)                                                                     \
  {                                                                                            \
    const int u_ = ((STEP) >> 1) * F::UPS, hs_ = (STEP) & 1;                                   \
    _Pragma("unroll") for (int q = 0; q < 2; q++) {                                            \
      const int job = tid + 128 * q, n = n0 + (job >> 2);                                      \
      if (n < N) {                                                                             \
        F::vload(w + (size_t)n * rowb, u_, hs_, job & 3, wr[q]);                               \
        F::sload(w + (size_t)n * rowb, u_, 4 * hs_ + (job & 3), sr[q]);                        \
      }                                                                                        \
    }                                                                                          \
    const int g_ = (u_ / F::UPS) * 2 + hs_; /* 128-value record of this step */               \
    {                                                                                          \
      const int r_ = tid & 63, m = m0 + r_;                                                    \
      if (m < M) vl_ldg16(xq + ((size_t)m * (size_t)ngx + (size_t)g_) * 160u + (tid >= 64 ? 16u : 0u), xsr); \
    }                                                                                          \
    _Pragma("unroll") for (int q = 0; q < 2; q++) {                                            \
      const int r_ = (tid >> 2) + 32 * q, c_ = tid & 3, m = m0 + r_;                           \
      if (m < M) {                                                                             \
        const unsigned char *rec_ = xq + ((size_t)m * (size_t)ngx + (size_t)g_) * 160u + 32u + 32u * (unsigned)c_; \
        vl_ldg16(rec_, xqr[q]);                                                                \
        vl_ldg16(rec_ + 16u, xqr[q] + 4);                                                      \
      }                                                                                        \
    }                                                                                          \
  }

  VL_MMQT_LOAD(0)
  for (int step = 0; step < nsteps; step++) {
    const int hs = step & 1;
    // decode the registers into shared memory
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int job = tid + 128 * q, r = job >> 2, c = job & 3;
      const bool ok = n0 + r < N;
      int o[8];
#pragma unroll
      for (int t = 0; t < 8; t++) o[t] = 0;
      float a0 = 0.0f, a1 = 0.0f, cb = 0.0f;
      if (ok) {
        F::vdec(wr[q], hs, c, o);
        F::sdec(sr[q], 4 * hs + c, &a0, &a1, &cb);
      }
      vl_store_row<F::ST>(sW[r], c, o);
      if (F::S16) {
        sA[r][2 * c] = a0;
        sA[r][2 * c + 1] = a1;
      } else {
        sA[r][c] = a0; // = a1
      }
      sC[r][c] = cb;
    }
    {
      const int r = tid & 63;
      const bool ok = m0 + r < M;
      if (tid < 64) {
#pragma unroll
        for (int t = 0; t < 4; t++) sXd[t][r] = ok ? __int_as_float(xsr[t]) : 0.0f;
      } else {
#pragma unroll
        for (int t = 0; t < 4; t++) sXs[t][r] = ok ? __int_as_float(xsr[t]) : 0.0f;
      }
    }
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int r = (tid >> 2) + 32 * q, c = tid & 3;
      const bool ok = m0 + r < M;
      int o[8];
#pragma unroll
      for (int t = 0; t < 8; t++) o[t] = ok ? xqr[q][t] : 0;
      vl_store_row<0>(sX[r], c, o);
    }
    VL_SYNC();
    if (step + 1 < nsteps) VL_MMQT_LOAD(step + 1)
#pragma unroll 1
    for (int kb = 0; kb < 4; kb++) { // 32-value block: 16-value groups 2 kb, 2 kb + 1
      float xd[4][2];
#pragma unroll
      for (int b = 0; b < 4; b++) vl_lds2f(&sXd[kb][wm + 8 * b + 2 * tg], xd[b]);
      int af[4][2], bf[4][2];
#pragma unroll
      for (int a = 0; a < 4; a++) vl_lds2(&sW[wn + 8 * a + gr][8 * tg + 2 * kb], af[a]);
#pragma unroll
      for (int b = 0; b < 4; b++) vl_lds2(&sX[wm + 8 * b + gr][8 * tg + 2 * kb], bf[b]);
      int dot[4][4][2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
          for (int b = 0; b < 4; b++) {
            if (h == 0 || F::S16) vl_mma_c(dot[a][b][0], dot[a][b][1], af[a][h], bf[b][h], VL_MAGIC);
            else vl_mma(dot[a][b][0], dot[a][b][1], af[a][h], bf[b][h]);
          }
        if (F::S16 || h == 1) { // float step: per 16 values (Q6_K) or per 32
#pragma unroll
          for (int a = 0; a < 4; a++) {
            const float s = F::S16 ? sA[wn + 8 * a + gr][2 * kb + h] : sA[wn + 8 * a + gr][kb];
            const float snm = -s * VL_MAGIC_F;
#pragma unroll
            for (int b = 0; b < 4; b++)
#pragma unroll
              for (int l = 0; l < 2; l++) {
                if (F::S16) acc[a][b][l] += s * xd[b][l] * (__int_as_float(dot[a][b][l]) - VL_MAGIC_F);
                else acc[a][b][l] = __fmaf_rn(__fmaf_rn(__int_as_float(dot[a][b][l]), s, snm), xd[b][l], acc[a][b][l]);
              }
          }
        }
      }
      if (F::HASC) { // the 32-value block's bias term
        float xs[4][2];
#pragma unroll
        for (int b = 0; b < 4; b++) vl_lds2f(&sXs[kb][wm + 8 * b + 2 * tg], xs[b]);
#pragma unroll
        for (int a = 0; a < 4; a++) {
          const float cb = sC[wn + 8 * a + gr][kb];
#pragma unroll
          for (int b = 0; b < 4; b++)
#pragma unroll
            for (int l = 0; l < 2; l++) acc[a][b][l] += cb * xs[b][l];
        }
      }
    }
    VL_SYNC();
  }
#undef VL_MMQT_LOAD
#pragma unroll
  for (int a = 0; a < 4; a++) {
    const int n = n0 + wn + 8 * a + gr;
    if (n >= N) continue;
#pragma unroll
    for (int b = 0; b < 4; b++)
#pragma unroll
      for (int l = 0; l < 2; l++) {
        const int m = m0 + wm + 8 * b + 2 * tg + l;
        if (m < M) y[(size_t)m * (size_t)N + (size_t)n] = acc[a][b][l];
      }
  }
}

// nb is in units: 32-value blocks for the first five, super-blocks for the K-quants.
// K % 256 == 0, activations in the x4 layout, 16-byte aligned; Q4_K / Q5_K weights 16-byte
// aligned (the host checks and uses mmq.cu otherwise).
#define VL_MMQT_KERNEL(NAME, T)                                                                \
  extern "C" __global__ void __launch_bounds__(128, 3)                                        \
  NAME(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,             \
       float *__restrict__ y, int M, int nb, int N) {                                         \
    vl_mmqt<T>(xq, w, y, M, nb, N);                                                            \
  }
VL_MMQT_KERNEL(vira_cu_mmqt_q4_0, vt_q4_0)
VL_MMQT_KERNEL(vira_cu_mmqt_q4_1, vt_q4_1)
VL_MMQT_KERNEL(vira_cu_mmqt_q5_0, vt_q5_0)
VL_MMQT_KERNEL(vira_cu_mmqt_q5_1, vt_q5_1)
VL_MMQT_KERNEL(vira_cu_mmqt_q8_0, vt_q8_0)
VL_MMQT_KERNEL(vira_cu_mmqt_q4_k, vt_q4_k)
VL_MMQT_KERNEL(vira_cu_mmqt_q5_k, vt_q5_k)
VL_MMQT_KERNEL(vira_cu_mmqt_q6_k, vt_q6_k)

// ---- big tile (llama.cpp's Turing configuration, ggml-cuda/mmq-config-ampere.cuh): ----
// 256 threads (8 warps), 128 weight rows x J activation rows (J = 128 or 64) per block, one
// block per SM. Each decoded weight value is used by J activation rows and each loaded
// activation by 128 weight rows (mmqt: 64 and 64), so the load and decode work per multiply
// halves. Warp w: weight rows 32 (w & 3).., activation rows (J / 2) (w >> 2)..; its A
// fragments (4 tiles of 8 rows) and scales stay in registers for a whole 32-value block while
// it walks its J / 16 tiles of 8 activation rows (as llama.cpp's vec_dot_q8_1_q8_1_mma).
// Loads, decoders, pipelining and the per-value arithmetic are those of vl_mmqt above, so
// both give the same sums in the same order.
// Grid ((N + 127) / 128, (M + J - 1) / J), block 256; ~46 KB shared (J = 128).
template <bool B> struct vl_enable_if {};
template <> struct vl_enable_if<true> { typedef void type; };
#define TB_N 128
// integer scales of an S16 format (only Q6_K); never called for the others
template <class F> struct vl_has_sdec_int { enum { v = 0 }; };
template <> struct vl_has_sdec_int<vt_q6_k> { enum { v = 1 }; };
template <class F>
__device__ __forceinline__ typename vl_enable_if<vl_has_sdec_int<F>::v>::type vl_sdec_int(const int *r, int *s0, int *s1, float *d) {
  F::sdec_int(r, s0, s1, d);
}
template <class F>
__device__ __forceinline__ typename vl_enable_if<!vl_has_sdec_int<F>::v>::type vl_sdec_int(const int *, int *, int *, float *) {}
#ifndef VL_EMULATE
__device__ __forceinline__ float vl_ex2a(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
#else
static inline float vl_ex2a(float x) { return exp2f(x); }
#endif
#ifndef VL_EMULATE
__device__ __forceinline__ float vl_shfl_xor(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
#endif
// silu(g) * u with the arithmetic of vira_cu_silu_mul_f32 (fused.cu)
__device__ __forceinline__ float vl_silu_mul(float g, float u) {
  const float e = vl_ex2a(__fmul_rn(g, __int_as_float(0xBFB8AA3B)));
  return __fmul_rn(__fdiv_rn(g, __fadd_rn(e, 1.0f)), u);
}

// GLU = false: y[m, n] = x . W[n] over 128 weight rows per block.
// GLU = true (gpu_linear_swiglu): w and w2 are the gate and up matrices; a block takes 64
// rows of each into its 128 shared rows (gate 0..63, up 64..127), warp w computes gate and up
// for 32 rows (w & 1) x J / 4 activation rows (w >> 1), and writes silu(gate) * up. The sums
// are those of the GLU = false kernel on each matrix (same order), so the result equals
// vira_cu_mmqb + vira_cu_silu_mul_f32.
// GLU with q4 != NULL (N % 128 == 0): the output is also written quantized in the x4 layout
// (vira_cu_quantize_q8_1x4's arithmetic), which the down projection reads; a warp holds whole
// 32-value blocks of an output row (its 32 rows of the 64), spread over the 8 lanes of a tg.
template <class F, int J, bool GLU>
__device__ __forceinline__ void vl_mmqb(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,
                                        const unsigned char *__restrict__ w2, float *__restrict__ y, int M, int nb,
                                        int N, unsigned char *__restrict__ q4 = NULL) {
  constexpr int NA = GLU ? 8 : 4;            // A tiles (8 weight rows) per warp
  constexpr int NBT = GLU ? J / 32 : J / 16; // activation tiles (8 rows) per warp
  constexpr int NXR = J / 64;                // activation q rows per 4-lane group (2 for J=128, 1 for J=64)
  __shared__ __align__(16) int sW[TB_N][TQ_S];
  __shared__ __align__(16) int sX[J][TQ_S];
  __shared__ float sA[TB_N][F::S16 ? 9 : 5];
  __shared__ float sC[TB_N][4];
  __shared__ __align__(16) float sXd[4][J];
  __shared__ __align__(16) float sXs[4][J];
  const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  const int gr = lane >> 2, tg = lane & 3;
  const int wn = GLU ? (wp & 1) * 32 : (wp & 3) * 32;
  const int wm = GLU ? (wp >> 1) * (J / 4) : (wp >> 2) * (J / 2);
  const int m0 = blockIdx.x * J, n0 = blockIdx.y * (GLU ? TB_N / 2 : TB_N);
  const int nxb = nb * (F::UPS == 8 ? 1 : 8);
  const int ngx = nxb / 4;
  const size_t rowb = (size_t)nb * (size_t)F::BYTES;
  const int nsteps = (nb / F::UPS) * 2;
  int wr[2][F::RAW];
  int sr[2][4];
  int xsr[4];
  int xqr[NXR][8];
  float acc[NA][NBT][2];
#pragma unroll
  for (int a = 0; a < NA; a++)
#pragma unroll
    for (int b = 0; b < NBT; b++) acc[a][b][0] = acc[a][b][1] = 0.0f;

  // shared row r -> its matrix row (GLU: rows 64.. are the up matrix)
#define VL_ROW_N(r) (n0 + (GLU ? ((r) & 63) : (r)))
#define VL_ROW_W(r) (GLU && (r) >= 64 ? w2 : w)
#define VL_MMQB_LOAD(STEP)                                                                     \
  {                                                                                            \
    const int u_ = ((STEP) >> 1) * F::UPS, hs_ = (STEP) & 1;                                   \
    _Pragma("unroll") for (int q = 0; q < 2; q++) {                                            \
      const int job = tid + 256 * q, r_ = job >> 2, n = VL_ROW_N(r_);                          \
      if (n < N) {                                                                             \
        const unsigned char *wrow_ = VL_ROW_W(r_) + (size_t)n * rowb;                          \
        F::vload(wrow_, u_, hs_, job & 3, wr[q]);                                              \
        F::sload(wrow_, u_, 4 * hs_ + (job & 3), sr[q]);                                       \
      }                                                                                        \
    }                                                                                          \
    const int g_ = (u_ / F::UPS) * 2 + hs_;                                                    \
    if (tid < 2 * J) {                                                                         \
      const int r_ = tid < J ? tid : tid - J, m = m0 + r_;                                     \
      if (m < M) vl_ldg16(xq + ((size_t)m * (size_t)ngx + (size_t)g_) * 160u + (tid >= J ? 16u : 0u), xsr); \
    }                                                                                          \
    _Pragma("unroll") for (int q = 0; q < NXR; q++) {                                          \
      const int r_ = (tid >> 2) + 64 * q, c_ = tid & 3, m = m0 + r_;                           \
      if (m < M) {                                                                             \
        const unsigned char *rec_ = xq + ((size_t)m * (size_t)ngx + (size_t)g_) * 160u + 32u + 32u * (unsigned)c_; \
        vl_ldg16(rec_, xqr[q]);                                                                \
        vl_ldg16(rec_ + 16u, xqr[q] + 4);                                                      \
      }                                                                                        \
    }                                                                                          \
  }

  VL_MMQB_LOAD(0)
  for (int step = 0; step < nsteps; step++) {
    const int hs = step & 1;
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int job = tid + 256 * q, r = job >> 2, c = job & 3;
      const bool ok = VL_ROW_N(r) < N;
      int o[8];
#pragma unroll
      for (int t = 0; t < 8; t++) o[t] = 0;
      float a0 = 0.0f, a1 = 0.0f, cb = 0.0f;
      int i0 = 0, i1 = 0;
      if (ok) {
        F::vdec(wr[q], hs, c, o);
        if (F::S16) vl_sdec_int<F>(sr[q], &i0, &i1, &cb); // cb = the super-block's d
        else F::sdec(sr[q], 4 * hs + c, &a0, &a1, &cb);
      }
      vl_store_row<F::ST>(sW[r], c, o);
      if (F::S16) { // integer scales (bit patterns) and d: see the S16 branch below
        sA[r][2 * c] = __int_as_float(i0);
        sA[r][2 * c + 1] = __int_as_float(i1);
      } else {
        sA[r][c] = a0;
      }
      sC[r][c] = cb;
    }
    if (tid < J) {
      const bool ok = m0 + tid < M;
#pragma unroll
      for (int t = 0; t < 4; t++) sXd[t][tid] = ok ? __int_as_float(xsr[t]) : 0.0f;
    } else if (tid < 2 * J) {
      const int r = tid - J;
      const bool ok = m0 + r < M;
#pragma unroll
      for (int t = 0; t < 4; t++) sXs[t][r] = ok ? __int_as_float(xsr[t]) : 0.0f;
    }
#pragma unroll
    for (int q = 0; q < NXR; q++) {
      const int r = (tid >> 2) + 64 * q, c = tid & 3;
      const bool ok = m0 + r < M;
      int o[8];
#pragma unroll
      for (int t = 0; t < 8; t++) o[t] = ok ? xqr[q][t] : 0;
      vl_store_row<0>(sX[r], c, o);
    }
    VL_SYNC();
    if (step + 1 < nsteps) VL_MMQB_LOAD(step + 1)
#pragma unroll 1
    for (int kb = 0; kb < 4; kb++) {
      int af[NA][2];
      float s0[NA], snm[NA], cbv[NA];
      int si0[NA], si1[NA];
#pragma unroll
      for (int a = 0; a < NA; a++) {
        const int r = (a >= 4 ? 64 : 0) + wn + 8 * (a & 3) + gr;
        vl_lds2(&sW[r][8 * tg + 2 * kb], af[a]);
        if (F::S16) {
          si0[a] = __float_as_int(sA[r][2 * kb]);
          si1[a] = __float_as_int(sA[r][2 * kb + 1]);
          s0[a] = sC[r][kb]; // d
          snm[a] = 0.0f;
        } else {
          s0[a] = sA[r][kb];
          snm[a] = -s0[a] * VL_MAGIC_F; // exact: s0 has <= 17 mantissa bits, VL_MAGIC_F = 3 * 2^22
        }
        cbv[a] = F::HASC ? sC[r][kb] : 0.0f;
      }
#pragma unroll
      for (int b = 0; b < NBT; b++) {
        int bf[2];
        float xd[2], xs[2];
        vl_lds2(&sX[wm + 8 * b + gr][8 * tg + 2 * kb], bf);
        vl_lds2f(&sXd[kb][wm + 8 * b + 2 * tg], xd);
        if (F::HASC) vl_lds2f(&sXs[kb][wm + 8 * b + 2 * tg], xs);
#pragma unroll
        for (int a = 0; a < NA; a++) {
          int d0, d1;
          if (F::S16) {
            // Q6_K, as llama.cpp's vec_dot_q6_K_q8_1_mma: the two 16-value dots times their
            // int8 scales summed in integers (|t| <= 2 * 16 * 32 * 127 * 128 < 2^24, exact as a
            // float), then one conversion and d * xd * t: 2 float ops instead of 6
            int e0, e1;
            vl_mma_c(d0, d1, af[a][0], bf[0], 0);
            vl_mma_c(e0, e1, af[a][1], bf[1], 0);
            const int t0 = d0 * si0[a] + e0 * si1[a], t1 = d1 * si0[a] + e1 * si1[a];
            acc[a][b][0] = __fmaf_rn(s0[a] * xd[0], (float)t0, acc[a][b][0]);
            acc[a][b][1] = __fmaf_rn(s0[a] * xd[1], (float)t1, acc[a][b][1]);
          } else {
            vl_mma_c(d0, d1, af[a][0], bf[0], VL_MAGIC);
            vl_mma(d0, d1, af[a][1], bf[1]);
            acc[a][b][0] = __fmaf_rn(__fmaf_rn(__int_as_float(d0), s0[a], snm[a]), xd[0], acc[a][b][0]);
            acc[a][b][1] = __fmaf_rn(__fmaf_rn(__int_as_float(d1), s0[a], snm[a]), xd[1], acc[a][b][1]);
          }
          if (F::HASC) {
            acc[a][b][0] = __fmaf_rn(cbv[a], xs[0], acc[a][b][0]);
            acc[a][b][1] = __fmaf_rn(cbv[a], xs[1], acc[a][b][1]);
          }
        }
      }
    }
    VL_SYNC();
  }
#undef VL_MMQB_LOAD
#undef VL_ROW_N
#undef VL_ROW_W
  if (GLU) {
#pragma unroll
    for (int b = 0; b < NBT; b++)
#pragma unroll
      for (int l = 0; l < 2; l++) {
        const int m = m0 + wm + 8 * b + 2 * tg + l;
        float v[4];
        float amax = 0.0f;
#pragma unroll
        for (int a = 0; a < 4; a++) {
          v[a] = vl_silu_mul(acc[a][b][l], acc[GLU ? a + 4 : a][b][l]);
          const int n = n0 + wn + 8 * a + gr;
          if (m < M && n < N) y[(size_t)m * (size_t)N + (size_t)n] = v[a];
          amax = fmaxf(amax, fabsf(v[a]));
        }
        if (q4 != NULL) { // uniform: every lane takes part in the shuffles
          amax = fmaxf(amax, vl_shfl_xor(amax, 4));
          amax = fmaxf(amax, vl_shfl_xor(amax, 8));
          amax = fmaxf(amax, vl_shfl_xor(amax, 16));
          int qv[4], qs = 0;
#pragma unroll
          for (int a = 0; a < 4; a++) {
            int q = 0;
            if (amax > 0.0f) {
              q = __float2int_rn(__fmul_rn(v[a], __fdiv_rn(127.0f, amax)));
              q = max(min(q, 127), -127);
            }
            qv[a] = q;
            qs += q;
          }
          float sum = (float)qs; // |sum| <= 32 * 127: exact as a float
          sum += vl_shfl_xor(sum, 4);
          sum += vl_shfl_xor(sum, 8);
          sum += vl_shfl_xor(sum, 16);
          if (m < M) {
            const int c0 = n0 + wn; // first column of this warp's 32-value block
            unsigned char *rec = q4 + ((size_t)m * (size_t)(N / 128) + (size_t)(c0 / 128)) * 160u;
            const int sub = (c0 % 128) / 32;
#pragma unroll
            for (int a = 0; a < 4; a++) rec[32 + 32 * sub + 8 * a + gr] = (unsigned char)(signed char)qv[a];
            if (gr == 0) {
              const float d = __fmul_rn(amax, __int_as_float(0x3C010204));
              ((float *)rec)[sub] = d;
              ((float *)rec)[4 + sub] = __fmul_rn(d, sum);
            }
          }
        }
      }
    return;
  }
#pragma unroll
  for (int a = 0; a < 4; a++) {
    const int n = n0 + wn + 8 * a + gr;
    if (n >= N) continue;
#pragma unroll
    for (int b = 0; b < NBT; b++)
#pragma unroll
      for (int l = 0; l < 2; l++) {
        const int m = m0 + wm + 8 * b + 2 * tg + l;
        if (m < M) y[(size_t)m * (size_t)N + (size_t)n] = GLU ? vl_silu_mul(acc[a][b][l], acc[GLU ? a + 4 : a][b][l]) : acc[a][b][l];
      }
  }
}
#define VL_MMQB_KERNEL(NAME, T, J)                                                             \
  extern "C" __global__ void __launch_bounds__(256, 1)                                        \
  NAME(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,             \
       float *__restrict__ y, int M, int nb, int N) {                                         \
    vl_mmqb<T, J, false>(xq, w, NULL, y, M, nb, N);                                            \
  }
// silu(x W_gate) * (x W_up) for gpu_linear_swiglu, and (q4 != NULL, N % 128 == 0) its x4 copy.
// Grid ((N + 63) / 64, (M + J - 1) / J), block 256.
#define VL_MMQG_KERNEL(NAME, T, J)                                                             \
  extern "C" __global__ void __launch_bounds__(256, 1)                                        \
  NAME(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ wg,            \
       const unsigned char *__restrict__ wu, float *__restrict__ y, int M, int nb, int N,      \
       unsigned char *__restrict__ q4) {                                                       \
    vl_mmqb<T, J, true>(xq, wg, wu, y, M, nb, N, q4);                                          \
  }
VL_MMQG_KERNEL(vira_cu_mmqg128_q4_0, vt_q4_0, 128)
VL_MMQG_KERNEL(vira_cu_mmqg128_q4_1, vt_q4_1, 128)
VL_MMQG_KERNEL(vira_cu_mmqg128_q5_0, vt_q5_0, 128)
VL_MMQG_KERNEL(vira_cu_mmqg128_q5_1, vt_q5_1, 128)
VL_MMQG_KERNEL(vira_cu_mmqg128_q8_0, vt_q8_0, 128)
VL_MMQG_KERNEL(vira_cu_mmqg128_q4_k, vt_q4_k, 128)
VL_MMQG_KERNEL(vira_cu_mmqg128_q5_k, vt_q5_k, 128)
VL_MMQG_KERNEL(vira_cu_mmqg128_q6_k, vt_q6_k, 128)
VL_MMQB_KERNEL(vira_cu_mmqb128_q4_0, vt_q4_0, 128)
VL_MMQB_KERNEL(vira_cu_mmqb128_q4_1, vt_q4_1, 128)
VL_MMQB_KERNEL(vira_cu_mmqb128_q5_0, vt_q5_0, 128)
VL_MMQB_KERNEL(vira_cu_mmqb128_q5_1, vt_q5_1, 128)
VL_MMQB_KERNEL(vira_cu_mmqb128_q8_0, vt_q8_0, 128)
VL_MMQB_KERNEL(vira_cu_mmqb128_q4_k, vt_q4_k, 128)
VL_MMQB_KERNEL(vira_cu_mmqb128_q5_k, vt_q5_k, 128)
VL_MMQB_KERNEL(vira_cu_mmqb128_q6_k, vt_q6_k, 128)
VL_MMQB_KERNEL(vira_cu_mmqb64_q4_0, vt_q4_0, 64)
VL_MMQB_KERNEL(vira_cu_mmqb64_q4_1, vt_q4_1, 64)
VL_MMQB_KERNEL(vira_cu_mmqb64_q5_0, vt_q5_0, 64)
VL_MMQB_KERNEL(vira_cu_mmqb64_q5_1, vt_q5_1, 64)
VL_MMQB_KERNEL(vira_cu_mmqb64_q8_0, vt_q8_0, 64)
VL_MMQB_KERNEL(vira_cu_mmqb64_q4_k, vt_q4_k, 64)
VL_MMQB_KERNEL(vira_cu_mmqb64_q5_k, vt_q5_k, 64)
VL_MMQB_KERNEL(vira_cu_mmqb64_q6_k, vt_q6_k, 64)
