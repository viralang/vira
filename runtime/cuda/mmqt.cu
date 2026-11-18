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

// UPS = units per 256-value slice (8 for 32-value blocks, 1 for super-blocks); BYTES per
// unit; S16 = a scale per 16 values (Q6_K); HASC = the format has a bias per 32 values.
#define VL_MMQT_KERNEL(NAME, F, UPS, BYTES, S16, HASC)                                         \
  extern "C" __global__ void __launch_bounds__(128)                                           \
  NAME(const unsigned char *__restrict__ xq, const unsigned char *__restrict__ w,             \
       float *__restrict__ y, int M, int nb, int N) {                                         \
    __shared__ int sW[TQ_N][TQ_S];                                                             \
    __shared__ int sX[TQ_M][TQ_S];                                                             \
    __shared__ float sA[TQ_N][9];                                                              \
    __shared__ float sC[TQ_N][5];                                                              \
    __shared__ float sXd[TQ_M][5];                                                             \
    __shared__ float sXs[TQ_M][5];                                                             \
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;                               \
    const int gr = lane >> 2, tg = lane & 3;                                                   \
    const int wn = (wp & 1) * 32, wm = (wp >> 1) * 32;                                         \
    const int n0 = blockIdx.x * TQ_N, m0 = blockIdx.y * TQ_M;                                  \
    const int nxb = nb * (UPS == 8 ? 1 : 8); /* Q8_1 blocks per activation row */              \
    float acc[4][4][2];                                                                        \
    _Pragma("unroll") for (int a = 0; a < 4; a++)                                              \
      _Pragma("unroll") for (int b = 0; b < 4; b++) acc[a][b][0] = acc[a][b][1] = 0.0f;        \
    for (int step = 0; step < (nb / UPS) * 2; step++) {                                        \
      const int u = (step >> 1) * UPS; /* first unit of the 256-value slice */                  \
      const int hs = step & 1;          /* which half of it */                                  \
      /* weights: warp k fills rows k, k + 4, ..; lane i reads int i of the row */             \
      _Pragma("unroll 4") for (int r = wp; r < TQ_N; r += 4) {                                 \
        const int n = n0 + r;                                                                  \
        sW[r][lane] = n < N ? F##_q(w + (size_t)n * (size_t)nb * (BYTES), u, 32 * hs + lane) : 0; \
      }                                                                                        \
      if (tid < TQ_N) { /* scales of row tid */                                                \
        const int n = n0 + tid;                                                                \
        if (n < N) {                                                                           \
          float a[16], c[8];                                                                   \
          F##_s(w + (size_t)n * (size_t)nb * (BYTES), u, a, c);                                \
          if (hs) { /* constant indices keep a / c in registers */                             \
            _Pragma("unroll") for (int g = 0; g < 8; g++) sA[tid][g] = a[8 + g];               \
            _Pragma("unroll") for (int g = 0; g < 4; g++) sC[tid][g] = c[4 + g];               \
          } else {                                                                             \
            _Pragma("unroll") for (int g = 0; g < 8; g++) sA[tid][g] = a[g];                   \
            _Pragma("unroll") for (int g = 0; g < 4; g++) sC[tid][g] = c[g];                   \
          }                                                                                    \
        } else {                                                                               \
          for (int g = 0; g < 8; g++) sA[tid][g] = 0.0f;                                       \
          for (int g = 0; g < 4; g++) sC[tid][g] = 0.0f;                                       \
        }                                                                                      \
      }                                                                                        \
      /* activations: job j = one Q8_1 block (8 ints) of activation row j / 4 */              \
      for (int j = tid; j < TQ_M * 4; j += 128) {                                              \
        const int c = j >> 2, part = j & 3;                                                    \
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
      _Pragma("unroll 1") for (int kb = 0; kb < 4; kb++) { /* 32-value block */                \
        float xd[4][2];                                                                        \
        _Pragma("unroll") for (int b = 0; b < 4; b++)                                          \
          _Pragma("unroll") for (int l = 0; l < 2; l++) xd[b][l] = sXd[wm + 8 * b + 2 * tg + l][kb]; \
        int dot[4][4][2];                                                                      \
        _Pragma("unroll") for (int a = 0; a < 4; a++)                                          \
          _Pragma("unroll") for (int b = 0; b < 4; b++) dot[a][b][0] = dot[a][b][1] = 0;       \
        _Pragma("unroll") for (int h = 0; h < 2; h++) { /* 16-value group g = 2 kb + h */     \
          const int col = 4 * (2 * kb + h) + tg;                                               \
          int af[4], bf[4];                                                                    \
          _Pragma("unroll") for (int a = 0; a < 4; a++) af[a] = sW[wn + 8 * a + gr][col];      \
          _Pragma("unroll") for (int b = 0; b < 4; b++) bf[b] = sX[wm + 8 * b + gr][col];      \
          _Pragma("unroll") for (int a = 0; a < 4; a++)                                        \
            _Pragma("unroll") for (int b = 0; b < 4; b++) vl_mma(dot[a][b][0], dot[a][b][1], af[a], bf[b]); \
          if (S16 || h == 1) { /* float step: per 16 values (Q6_K) or per 32 */                \
            _Pragma("unroll") for (int a = 0; a < 4; a++) {                                    \
              const float s = sA[wn + 8 * a + gr][2 * kb + (S16 ? h : 0)];                     \
              _Pragma("unroll") for (int b = 0; b < 4; b++)                                    \
                _Pragma("unroll") for (int l = 0; l < 2; l++) {                                \
                  acc[a][b][l] += s * xd[b][l] * (float)dot[a][b][l];                          \
                  dot[a][b][l] = 0;                                                            \
                }                                                                              \
            }                                                                                  \
          }                                                                                    \
        }                                                                                      \
        if (HASC) { /* the 32-value block's bias term */                                       \
          _Pragma("unroll") for (int a = 0; a < 4; a++) {                                      \
            const float cb = sC[wn + 8 * a + gr][kb];                                          \
            _Pragma("unroll") for (int b = 0; b < 4; b++)                                      \
              _Pragma("unroll") for (int l = 0; l < 2; l++)                                    \
                acc[a][b][l] += cb * sXs[wm + 8 * b + 2 * tg + l][kb];                         \
          }                                                                                    \
        }                                                                                      \
      }                                                                                        \
      VL_SYNC();                                                                               \
    }                                                                                          \
    _Pragma("unroll") for (int a = 0; a < 4; a++) {                                            \
      const int n = n0 + wn + 8 * a + gr;                                                      \
      if (n >= N) continue;                                                                    \
      _Pragma("unroll") for (int b = 0; b < 4; b++)                                            \
        _Pragma("unroll") for (int l = 0; l < 2; l++) {                                        \
          const int m = m0 + wm + 8 * b + 2 * tg + l;                                          \
          if (m < M) y[(size_t)m * (size_t)N + (size_t)n] = acc[a][b][l];                      \
        }                                                                                      \
    }                                                                                          \
  }

// nb is in units: 32-value blocks for the first five, super-blocks for the K-quants.
// K % 256 == 0 (the host checks).
VL_MMQT_KERNEL(vira_cu_mmqt_q4_0, mq_q4_0, 8, 18u, 0, 0)
VL_MMQT_KERNEL(vira_cu_mmqt_q4_1, mq_q4_1, 8, 20u, 0, 1)
VL_MMQT_KERNEL(vira_cu_mmqt_q5_0, mq_q5_0, 8, 22u, 0, 0)
VL_MMQT_KERNEL(vira_cu_mmqt_q5_1, mq_q5_1, 8, 24u, 0, 1)
VL_MMQT_KERNEL(vira_cu_mmqt_q8_0, mq_q8_0, 8, 34u, 0, 0)
VL_MMQT_KERNEL(vira_cu_mmqt_q4_k, mq_q4_k, 1, 144u, 0, 1)
VL_MMQT_KERNEL(vira_cu_mmqt_q5_k, mq_q5_k, 1, 176u, 0, 1)
VL_MMQT_KERNEL(vira_cu_mmqt_q6_k, mq_q6_k, 1, 210u, 1, 0)
