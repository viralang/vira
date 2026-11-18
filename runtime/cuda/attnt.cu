// nvrtc-arch: compute_75
// Prompt attention over the FP16 KV cache on tensor cores (Turing and newer).
// Source of runtime/vira_cuda_attnt_ptx.h (python3 runtime/cuda/build_ptx.py). Same contract
// as vira_cu_attn_f16kv / vira_cu_attn_f16kv_tile (attn.cu): out[t, h, :] = softmax_j(scale *
// q[t, h] . K[j, kvh]) V[j, kvh] over j <= start + t, kvh = h / kv_mul.
//
// FlashAttention-2 on mma.sync.m16n8k8 (f16 inputs, f32 accumulators), as llama.cpp's
// fattn-mma-f16.cuh does on Turing: a block of 4 warps takes 64 query rows of one head (16 per
// warp, held as f16 A fragments, already scaled); keys come in tiles of 32 through shared
// memory (K as rows, V transposed); S = Q K^T in f32, online softmax per row (a row lives in
// the 4 lanes of a quad), and the f32 accumulator layout of S is the A-fragment layout of
// P x V, so P goes to f16 without moving between lanes.
//
// Fragment layouts (PTX ISA; llama.cpp ggml-cuda/mma.cuh tile<16, 8, half2> / tile<8, 8,
// half2> / tile<16, 8, float>), lane = 4 gr + tg:
//   A (16 x 8 f16, row): a0 = {A[gr][2 tg], A[gr][2 tg + 1]}, a1 = the same for row gr + 8
//   B (8 x 8 f16, col):  b = {B[2 tg][gr], B[2 tg + 1][gr]}
//   D (16 x 8 f32):      d0, d1 = D[gr][2 tg], D[gr][2 tg + 1]; d2, d3 = row gr + 8
// Grid (n_heads, (M + 63) / 64), block 128. head_dim 64 or 128 (the host checks).

#define FA_Q 64   // query rows per block
#define FA_K 32   // keys per tile
#define FA_NEG_INF __int_as_float((int)0xff800000)

#ifdef VL_EMULATE
#include "emulate.h"
#else
#define VL_SYNC() __syncthreads()
__device__ __forceinline__ unsigned short vl_f2h(float f) {
  unsigned short h;
  asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float vl_h2f(unsigned short h) {
  float f;
  asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ float vl_shfl_xor(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
// d[0..3] += A x B
__device__ __forceinline__ void vl_mma_f16(float *d, unsigned int a0, unsigned int a1, unsigned int b) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b));
}
#endif

__device__ __forceinline__ unsigned int vl_pack_h2(float lo, float hi) {
  return (unsigned int)vl_f2h(lo) | ((unsigned int)vl_f2h(hi) << 16);
}

// q4 != NULL ((n_heads * HD) % 128 == 0): the output rows are also written quantized in the
// x4 layout of mmqt.cu (vira_cu_quantize_q8_1x4's arithmetic) for the output projection; a
// 32-value block of a row is dims 32 b.. of one head, held by the 4 lanes of a quad.
template <int HD>
__device__ __forceinline__ void vl_attn_fa(const float *__restrict__ q, const unsigned short *__restrict__ kc,
                                           const unsigned short *__restrict__ vc, float *__restrict__ out,
                                           int n_heads, int kv_mul, int kv_dim, int M, long long start, float scale,
                                           unsigned char *__restrict__ q4) {
  constexpr int KS = HD + 8;   // sK row stride in halves: (HD + 8) / 2 = 4 mod 32 words
  constexpr int VS = FA_K + 8; // sVt row stride in halves: 20 words
  constexpr int ND = HD / 8;   // 8-dim tiles of the output, 8-dim k-steps of Q K^T
  __shared__ __align__(16) unsigned short sK[FA_K * KS];
  __shared__ __align__(16) unsigned short sVt[HD * VS];
  const int h = blockIdx.x, t0 = blockIdx.y * FA_Q;
  const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  const int gr = lane >> 2, tg = lane & 3;
  const int kvh = h / kv_mul;
  const int r0 = t0 + 16 * wp + gr, r1 = r0 + 8; // this lane's two query rows
  // Q as A fragments (scaled, f16)
  unsigned int qa[ND][2];
#pragma unroll
  for (int kk = 0; kk < ND; kk++) {
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (r0 < M) {
      const float *qr = q + ((size_t)r0 * n_heads + h) * HD + 8 * kk + 2 * tg;
      v[0] = qr[0] * scale;
      v[1] = qr[1] * scale;
    }
    if (r1 < M) {
      const float *qr = q + ((size_t)r1 * n_heads + h) * HD + 8 * kk + 2 * tg;
      v[2] = qr[0] * scale;
      v[3] = qr[1] * scale;
    }
    qa[kk][0] = vl_pack_h2(v[0], v[1]);
    qa[kk][1] = vl_pack_h2(v[2], v[3]);
  }
  float o[ND][4];
#pragma unroll
  for (int d = 0; d < ND; d++) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.0f;
  float m0 = FA_NEG_INF, m1 = FA_NEG_INF, l0 = 0.0f, l1 = 0.0f;
  const int rows = M - t0 < FA_Q ? M - t0 : FA_Q;
  const long long last = start + t0 + rows - 1; // last key a row of the block sees
  for (long long j0 = 0; j0 <= last; j0 += FA_K) {
    VL_SYNC(); // the previous tile is used up
    // K rows and V transposed into shared memory, 2 halves per load
    for (int i = tid; i < FA_K * (HD / 2); i += 128) {
      const int jj = i / (HD / 2), d2 = i - jj * (HD / 2);
      const long long j = j0 + jj;
      unsigned int kw = 0u, vw = 0u;
      if (j <= last) {
        const size_t off = (size_t)j * (size_t)kv_dim + (size_t)kvh * HD + 2 * (size_t)d2;
        kw = __ldg((const unsigned int *)(kc + off));
        vw = __ldg((const unsigned int *)(vc + off));
      }
      *(unsigned int *)&sK[jj * KS + 2 * d2] = kw;
      sVt[(2 * d2) * VS + jj] = (unsigned short)(vw & 0xFFFFu);
      sVt[(2 * d2 + 1) * VS + jj] = (unsigned short)(vw >> 16);
    }
    VL_SYNC();
    // S = Q K^T for 16 rows x 32 keys: 4 tiles of 8 keys
    float s[4][4];
#pragma unroll
    for (int n = 0; n < 4; n++) {
      s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
      for (int kk = 0; kk < ND; kk++) {
        const unsigned int b = *(const unsigned int *)&sK[(8 * n + gr) * KS + 8 * kk + 2 * tg];
        vl_mma_f16(s[n], qa[kk][0], qa[kk][1], b);
      }
    }
    // causal mask, online softmax (a row is spread over the 4 lanes of a quad)
    float mx0 = FA_NEG_INF, mx1 = FA_NEG_INF;
#pragma unroll
    for (int n = 0; n < 4; n++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const long long key = j0 + 8 * n + 2 * tg + e;
        if (r0 >= M || key > start + r0) s[n][e] = FA_NEG_INF;
        if (r1 >= M || key > start + r1) s[n][2 + e] = FA_NEG_INF;
        mx0 = fmaxf(mx0, s[n][e]);
        mx1 = fmaxf(mx1, s[n][2 + e]);
      }
    mx0 = fmaxf(mx0, vl_shfl_xor(mx0, 1));
    mx0 = fmaxf(mx0, vl_shfl_xor(mx0, 2));
    mx1 = fmaxf(mx1, vl_shfl_xor(mx1, 1));
    mx1 = fmaxf(mx1, vl_shfl_xor(mx1, 2));
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float c0 = mn0 == FA_NEG_INF ? 1.0f : __expf(m0 - mn0);
    const float c1 = mn1 == FA_NEG_INF ? 1.0f : __expf(m1 - mn1);
    float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
    for (int n = 0; n < 4; n++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        s[n][e] = mn0 == FA_NEG_INF ? 0.0f : __expf(s[n][e] - mn0);
        s[n][2 + e] = mn1 == FA_NEG_INF ? 0.0f : __expf(s[n][2 + e] - mn1);
        ps0 += s[n][e];
        ps1 += s[n][2 + e];
      }
    l0 = l0 * c0 + ps0;
    l1 = l1 * c1 + ps1;
    m0 = mn0;
    m1 = mn1;
#pragma unroll
    for (int d = 0; d < ND; d++) {
      o[d][0] *= c0;
      o[d][1] *= c0;
      o[d][2] *= c1;
      o[d][3] *= c1;
    }
    // O += P V: P's accumulator layout is the A layout (keys 8 n .. 8 n + 7 = k-step n)
#pragma unroll
    for (int n = 0; n < 4; n++) {
      const unsigned int pa0 = vl_pack_h2(s[n][0], s[n][1]), pa1 = vl_pack_h2(s[n][2], s[n][3]);
#pragma unroll
      for (int d = 0; d < ND; d++) {
        const unsigned int b = *(const unsigned int *)&sVt[(8 * d + gr) * VS + 8 * n + 2 * tg];
        vl_mma_f16(o[d], pa0, pa1, b);
      }
    }
  }
  l0 += vl_shfl_xor(l0, 1);
  l0 += vl_shfl_xor(l0, 2);
  l1 += vl_shfl_xor(l1, 1);
  l1 += vl_shfl_xor(l1, 2);
#pragma unroll
  for (int rr = 0; rr < 2; rr++) {
    const int row = rr == 0 ? r0 : r1;
    const float inv = row < M ? 1.0f / (rr == 0 ? l0 : l1) : 0.0f;
    float v[ND][2];
#pragma unroll
    for (int d = 0; d < ND; d++) {
      v[d][0] = o[d][2 * rr] * inv;
      v[d][1] = o[d][2 * rr + 1] * inv;
    }
    if (row < M) {
      float *orow = out + ((size_t)row * n_heads + h) * HD + 2 * tg;
#pragma unroll
      for (int d = 0; d < ND; d++) {
        orow[8 * d] = v[d][0];
        orow[8 * d + 1] = v[d][1];
      }
    }
    if (q4 != NULL) { // uniform across the warp: every lane takes part in the shuffles
      const int kt = n_heads * HD; // row length of the projection's input
#pragma unroll
      for (int b = 0; b < HD / 32; b++) { // 32-value block: dims 32 b.., tiles d = 4 b .. 4 b + 3
        float amax = 0.0f;
#pragma unroll
        for (int dd = 0; dd < 4; dd++) amax = fmaxf(amax, fmaxf(fabsf(v[4 * b + dd][0]), fabsf(v[4 * b + dd][1])));
        amax = fmaxf(amax, vl_shfl_xor(amax, 1));
        amax = fmaxf(amax, vl_shfl_xor(amax, 2));
        int qv[8], qs = 0;
#pragma unroll
        for (int dd = 0; dd < 4; dd++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            int qq = 0;
            if (amax > 0.0f) {
              qq = __float2int_rn(__fmul_rn(v[4 * b + dd][e], __fdiv_rn(127.0f, amax)));
              qq = max(min(qq, 127), -127);
            }
            qv[2 * dd + e] = qq;
            qs += qq;
          }
        float sum = (float)qs; // |sum| <= 32 * 127: exact as a float
        sum += vl_shfl_xor(sum, 1);
        sum += vl_shfl_xor(sum, 2);
        if (row < M) {
          const int c0 = h * HD + 32 * b;
          unsigned char *rec = q4 + ((size_t)row * (size_t)(kt / 128) + (size_t)(c0 / 128)) * 160u;
          const int sub = (c0 % 128) / 32;
#pragma unroll
          for (int dd = 0; dd < 4; dd++)
#pragma unroll
            for (int e = 0; e < 2; e++) rec[32 + 32 * sub + 8 * dd + 2 * tg + e] = (unsigned char)(signed char)qv[2 * dd + e];
          if (tg == 0) {
            const float dsc = __fmul_rn(amax, __int_as_float(0x3C010204));
            ((float *)rec)[sub] = dsc;
            ((float *)rec)[4 + sub] = __fmul_rn(dsc, sum);
          }
        }
      }
    }
  }
}

// Arguments of vira_cu_attn_f16kv (head_dim fixed by the name) plus q4 (NULL: none).
#define VL_FA_KERNEL(NAME, HD)                                                                 \
  extern "C" __global__ void __launch_bounds__(128)                                           \
  NAME(const float *__restrict__ q, const unsigned short *__restrict__ kc, const unsigned short *__restrict__ vc, \
       float *__restrict__ out, int n_heads, int kv_mul, int head_dim, int kv_dim, int M, long long start, float scale, \
       unsigned char *__restrict__ q4) {                                                       \
    (void)head_dim;                                                                            \
    vl_attn_fa<HD>(q, kc, vc, out, n_heads, kv_mul, kv_dim, M, start, scale, q4);              \
  }
VL_FA_KERNEL(vira_cu_attn_fa64, 64)
VL_FA_KERNEL(vira_cu_attn_fa128, 128)
