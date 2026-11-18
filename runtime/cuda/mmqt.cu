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
// 4 g + tg of row gr" of the shared tiles: shared rows are 36 ints apart (36 = 4 mod 32), so
// the 32 lanes hit 32 different banks.

#define TQ_N 64   // weight rows per block
#define TQ_M 64   // activation rows per block
#define TQ_K 128  // values per step (half of a 256-value slice)
#define TQ_I (TQ_K / 4)
#define TQ_S 36   // shared row stride in ints

#include "mmq_fmt.cuh"

#ifndef VL_EMULATE
// d0, d1 += A x B for the lane's two D elements (warp-wide; every lane must call it)
__device__ __forceinline__ void vl_mma(int &d0, int &d1, int a, int b) {
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
      : "+r"(d0), "+r"(d1)
      : "r"(a), "r"(b));
}
#endif

#define VL_MAGIC 0x4B400000
#define VL_MAGIC_F 12582912.0f

// per-format traits: UPS = units per 256-value slice (8 for 32-value blocks, 1 for
// super-blocks); BYTES per unit; S16 = a scale per 16 values (Q6_K); HASC = has a bias
#define VL_TRAITS(T, F, UPS_, BYTES_, S16_, HASC_)                                            \
  struct T {                                                                                 \
    enum { UPS = UPS_, BYTES = BYTES_, S16 = S16_, HASC = HASC_ };                           \
    static __device__ __forceinline__ void qr(const unsigned char *row, int u, int i, int *r) { F##_qr(row, u, i, r); } \
    static __device__ __forceinline__ int qd(const int *r, int i) { return F##_qd(r, i); }   \
    static __device__ __forceinline__ void sr(const unsigned char *row, int u, int j, int *r) { F##_sr(row, u, j, r); } \
    static __device__ __forceinline__ void sd(const int *r, int j, float *a0, float *a1, float *c) { F##_sd(r, j, a0, a1, c); } \
  };
VL_TRAITS(vt_q4_0, mq_q4_0, 8, 18, 0, 0)
VL_TRAITS(vt_q4_1, mq_q4_1, 8, 20, 0, 1)
VL_TRAITS(vt_q5_0, mq_q5_0, 8, 22, 0, 0)
VL_TRAITS(vt_q5_1, mq_q5_1, 8, 24, 0, 1)
VL_TRAITS(vt_q8_0, mq_q8_0, 8, 34, 0, 0)
VL_TRAITS(vt_q4_k, mq_q4_k, 1, 144, 0, 1)
VL_TRAITS(vt_q5_k, mq_q5_k, 1, 176, 0, 1)
VL_TRAITS(vt_q6_k, mq_q6_k, 1, 210, 1, 0)

#define TQ_WJ (TQ_N / 4) // weight ints per thread per step (row wp + 4 k, int lane)

template <class F>
__device__ __forceinline__ void vl_mmqt(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,
                                        float *__restrict__ y, int M, int nb, int N) {
  __shared__ int sW[TQ_N][TQ_S];
  __shared__ int sX[TQ_M][TQ_S];
  __shared__ float sA[TQ_N][9];
  __shared__ float sC[TQ_N][5];
  __shared__ float sXd[TQ_M][5];
  __shared__ float sXs[TQ_M][5];
  const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  const int gr = lane >> 2, tg = lane & 3;
  const int wn = (wp & 1) * 32, wm = (wp >> 1) * 32;
  const int n0 = blockIdx.x * TQ_N, m0 = blockIdx.y * TQ_M;
  const int nxb = nb * (F::UPS == 8 ? 1 : 8); // Q8_1 blocks per activation row
  const size_t rowb = (size_t)nb * (size_t)F::BYTES;
  const int nsteps = (nb / F::UPS) * 2;
  // registers holding the next step's raw words
  int wr[TQ_WJ][2];
  int sr[2][4];
  int xr[2][8];
  float xf[2][2];
  float acc[4][4][2];
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int b = 0; b < 4; b++) acc[a][b][0] = acc[a][b][1] = 0.0f;

#define VL_MMQT_LOAD(STEP)                                                                     \
  {                                                                                            \
    const int u_ = ((STEP) >> 1) * F::UPS, hs_ = (STEP) & 1;                                   \
    _Pragma("unroll") for (int k = 0; k < TQ_WJ; k++) {                                        \
      const int n = n0 + wp + 4 * k;                                                           \
      if (n < N) F::qr(w + (size_t)n * rowb, u_, 32 * hs_ + lane, wr[k]);                      \
    }                                                                                          \
    _Pragma("unroll") for (int q = 0; q < 2; q++) {                                            \
      const int job = tid + 128 * q, n = n0 + (job >> 2);                                      \
      if (n < N) F::sr(w + (size_t)n * rowb, u_, 4 * hs_ + (job & 3), sr[q]);                  \
    }                                                                                          \
    const int xb0_ = (u_ / F::UPS) * 8 + 4 * hs_; /* first Q8_1 block of this step */         \
    _Pragma("unroll") for (int q = 0; q < 2; q++) {                                            \
      const int job = tid + 128 * q, m = m0 + (job >> 2);                                      \
      if (m < M) {                                                                             \
        const unsigned char *xb = xq + ((size_t)m * (size_t)nxb + (size_t)xb0_ + (size_t)(job & 3)) * 40u; \
        _Pragma("unroll") for (int i = 0; i < 8; i++) xr[q][i] = __ldg((const int *)(xb + 8u) + i); \
        xf[q][0] = __ldg((const float *)xb);                                                   \
        xf[q][1] = __ldg((const float *)(xb + 4u));                                            \
      }                                                                                        \
    }                                                                                          \
  }

  VL_MMQT_LOAD(0)
  for (int step = 0; step < nsteps; step++) {
    const int hs = step & 1;
    // decode the registers into shared memory
#pragma unroll
    for (int k = 0; k < TQ_WJ; k++) {
      const int r = wp + 4 * k;
      sW[r][lane] = n0 + r < N ? F::qd(wr[k], 32 * hs + lane) : 0;
    }
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int job = tid + 128 * q, r = job >> 2, jj = job & 3;
      float a0 = 0.0f, a1 = 0.0f, c = 0.0f;
      if (n0 + r < N) F::sd(sr[q], 4 * hs + jj, &a0, &a1, &c);
      sA[r][2 * jj] = a0;
      sA[r][2 * jj + 1] = a1;
      sC[r][jj] = c;
    }
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int job = tid + 128 * q, c = job >> 2, part = job & 3;
      const bool ok = m0 + c < M;
#pragma unroll
      for (int i = 0; i < 8; i++) sX[c][8 * part + i] = ok ? xr[q][i] : 0;
      sXd[c][part] = ok ? xf[q][0] : 0.0f;
      sXs[c][part] = ok ? xf[q][1] : 0.0f;
    }
    VL_SYNC();
    if (step + 1 < nsteps) VL_MMQT_LOAD(step + 1)
#pragma unroll 1
    for (int kb = 0; kb < 4; kb++) { // 32-value block
      float xd[4][2];
#pragma unroll
      for (int b = 0; b < 4; b++)
#pragma unroll
        for (int l = 0; l < 2; l++) xd[b][l] = sXd[wm + 8 * b + 2 * tg + l][kb];
      int dot[4][4][2];
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 4; b++) dot[a][b][0] = dot[a][b][1] = VL_MAGIC;
#pragma unroll
      for (int h = 0; h < 2; h++) { // 16-value group g = 2 kb + h
        const int col = 4 * (2 * kb + h) + tg;
        int af[4], bf[4];
#pragma unroll
        for (int a = 0; a < 4; a++) af[a] = sW[wn + 8 * a + gr][col];
#pragma unroll
        for (int b = 0; b < 4; b++) bf[b] = sX[wm + 8 * b + gr][col];
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
          for (int b = 0; b < 4; b++) vl_mma(dot[a][b][0], dot[a][b][1], af[a], bf[b]);
        if (F::S16 || h == 1) { // float step: per 16 values (Q6_K) or per 32
#pragma unroll
          for (int a = 0; a < 4; a++) {
            const float s = sA[wn + 8 * a + gr][2 * kb + (F::S16 ? h : 0)];
#pragma unroll
            for (int b = 0; b < 4; b++)
#pragma unroll
              for (int l = 0; l < 2; l++) {
                acc[a][b][l] += s * xd[b][l] * (__int_as_float(dot[a][b][l]) - VL_MAGIC_F);
                dot[a][b][l] = VL_MAGIC;
              }
          }
        }
      }
      if (F::HASC) { // the 32-value block's bias term
#pragma unroll
        for (int a = 0; a < 4; a++) {
          const float cb = sC[wn + 8 * a + gr][kb];
#pragma unroll
          for (int b = 0; b < 4; b++)
#pragma unroll
            for (int l = 0; l < 2; l++) acc[a][b][l] += cb * sXs[wm + 8 * b + 2 * tg + l][kb];
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
// K % 256 == 0 (the host checks).
#define VL_MMQT_KERNEL(NAME, T)                                                                \
  extern "C" __global__ void __launch_bounds__(128)                                           \
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
