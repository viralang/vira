// Attention over an FP16 KV cache, and the f32 <-> f16 row conversions it needs.
// Source of runtime/vira_cuda_attn_ptx.h (python3 runtime/cuda/build_ptx.py).
//
// Cache layout as the f32 kernels: K and V [ctx, kv_dim] row-major, kv_dim = n_kv_heads *
// head_dim, query head h reads kv head h / kv_mul. Stored as IEEE half (cvt.rn.f16.f32), so
// a cache costs half the memory of the f32 one.

#define WARP 32
#define AT_THREADS 256
#define AT_WARPS (AT_THREADS / WARP)
#define AT_MAX_D 8 // head_dim / 32 values per lane: head_dim <= 256
#define AT_NEG_INF __int_as_float((int)0xff800000)

#ifdef VL_EMULATE
#include "emulate.h"
#else
#define VL_SYNC() __syncthreads()
__device__ __forceinline__ float vl_h2f(unsigned short h) {
  float f;
  asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short vl_f2h(float f) {
  unsigned short h;
  asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float vl_wsum(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
#endif

// dst[(row0 + r) * cols + c] = half(src[r * cols + c]) for r < rows. Grid (n + 255) / 256.
extern "C" __global__ void vira_cu_kv_store_f16(const float *__restrict__ src, unsigned short *__restrict__ dst,
                                               long long row0, int cols, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[row0 * cols + i] = vl_f2h(src[i]);
}
// Same with row0 read from a 4-byte device pointer (CUDA Graph decode capture).
extern "C" __global__ void vira_cu_kv_store_f16_dpos(const float *__restrict__ src, unsigned short *__restrict__ dst,
                                                    const int *__restrict__ pos_ptr, int cols, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long row0 = (long long)(*pos_ptr);
  dst[row0 * cols + i] = vl_f2h(src[i]);
}
// dst[i] = float(src[i]) for i < n (cache read back to the host).
extern "C" __global__ void vira_cu_kv_load_f32(const unsigned short *__restrict__ src, float *__restrict__ dst,
                                              long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = vl_h2f(src[i]);
}

// out[t, h, :] = softmax_j(scale * q[t, h] . K[j, kvh]) V[j, kvh] over j <= start + t (causal),
// q / out rows t * n_heads + h (decode: one row of n_heads). One block per (head, query row):
// grid (n_heads, M), block 256. Each warp walks every 8th key with an online softmax (its own
// running max and sum), lanes hold head_dim / 32 dimensions; warps merge in shared memory.
__device__ __forceinline__ void vl_attn_f16kv_body(const float *__restrict__ q,
                                                    const unsigned short *__restrict__ kc,
                                                    const unsigned short *__restrict__ vc, float *__restrict__ out,
                                                    int n_heads, int kv_mul, int head_dim, int kv_dim, int M,
                                                    long long start, float scale) {
  __shared__ float s_m[AT_WARPS], s_l[AT_WARPS];
  __shared__ float s_acc[AT_WARPS][AT_MAX_D * WARP];
  const int h = blockIdx.x, t = blockIdx.y;
  if (h >= n_heads || t >= M) return; // whole block
  const int tid = threadIdx.x, lane = tid & (WARP - 1), wp = tid / WARP;
  const int nd = (head_dim + WARP - 1) / WARP; // values per lane
  const int kvh = h / kv_mul;
  const float *qr = q + ((size_t)t * n_heads + h) * head_dim;
  float qv[AT_MAX_D], acc[AT_MAX_D];
#pragma unroll
  for (int i = 0; i < AT_MAX_D; i++) {
    const int d = lane + WARP * i;
    qv[i] = (i < nd && d < head_dim) ? qr[d] * scale : 0.0f;
    acc[i] = 0.0f;
  }
  float m = AT_NEG_INF, l = 0.0f;
  const long long n_keys = start + t + 1;
  for (long long j = wp; j < n_keys; j += AT_WARPS) {
    const unsigned short *kr = kc + (size_t)j * kv_dim + (size_t)kvh * head_dim;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < AT_MAX_D; i++) {
      const int d = lane + WARP * i;
      if (i < nd && d < head_dim) s += qv[i] * vl_h2f(kr[d]);
    }
    s = vl_wsum(s);
    const float mn = fmaxf(m, s);
    const float corr = __expf(m - mn), p = __expf(s - mn);
    l = l * corr + p;
    const unsigned short *vr = vc + (size_t)j * kv_dim + (size_t)kvh * head_dim;
#pragma unroll
    for (int i = 0; i < AT_MAX_D; i++) {
      const int d = lane + WARP * i;
      if (i < nd && d < head_dim) acc[i] = acc[i] * corr + p * vl_h2f(vr[d]);
    }
    m = mn;
  }
  if (lane == 0) {
    s_m[wp] = m;
    s_l[wp] = l;
  }
#pragma unroll
  for (int i = 0; i < AT_MAX_D; i++) s_acc[wp][lane + WARP * i] = acc[i];
  VL_SYNC();
  if (wp == 0) {
    float M_ = AT_NEG_INF;
    for (int w = 0; w < AT_WARPS; w++) M_ = fmaxf(M_, s_m[w]);
    float L = 0.0f, o[AT_MAX_D];
#pragma unroll
    for (int i = 0; i < AT_MAX_D; i++) o[i] = 0.0f;
    for (int w = 0; w < AT_WARPS; w++) {
      if (s_l[w] == 0.0f) continue; // a warp with no keys (n_keys < 8)
      const float c = __expf(s_m[w] - M_);
      L += s_l[w] * c;
#pragma unroll
      for (int i = 0; i < AT_MAX_D; i++) o[i] += s_acc[w][lane + WARP * i] * c;
    }
    float *orow = out + ((size_t)t * n_heads + h) * head_dim;
    const float inv = 1.0f / L;
#pragma unroll
    for (int i = 0; i < AT_MAX_D; i++) {
      const int d = lane + WARP * i;
      if (i < nd && d < head_dim) orow[d] = o[i] * inv;
    }
  }
}
extern "C" __global__ void __launch_bounds__(AT_THREADS)
vira_cu_attn_f16kv(const float *__restrict__ q, const unsigned short *__restrict__ kc,
                   const unsigned short *__restrict__ vc, float *__restrict__ out, int n_heads, int kv_mul,
                   int head_dim, int kv_dim, int M, long long start, float scale) {
  vl_attn_f16kv_body(q, kc, vc, out, n_heads, kv_mul, head_dim, kv_dim, M, start, scale);
}
extern "C" __global__ void __launch_bounds__(AT_THREADS)
vira_cu_attn_f16kv_dpos(const float *__restrict__ q, const unsigned short *__restrict__ kc,
                        const unsigned short *__restrict__ vc, float *__restrict__ out, int n_heads, int kv_mul,
                        int head_dim, int kv_dim, int M, const int *__restrict__ pos_ptr, float scale) {
  vl_attn_f16kv_body(q, kc, vc, out, n_heads, kv_mul, head_dim, kv_dim, M, (long long)(*pos_ptr), scale);
}

// dst[i] = 0 for i < n (32-bit words): a new FP16 cache starts at zero without an upload.
extern "C" __global__ void vira_cu_kv_zero(unsigned int *__restrict__ dst, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = 0u;
}

// Prompt attention (M > 4 query rows), same contract as vira_cu_attn_f16kv. The decode
// kernel re-reads K and V for every query row; here a block of 256 threads takes 16 query
// rows of one head (warp w: rows 2w, 2w + 1) and walks the keys in tiles of 32: each tile of
// K and V is read from memory once (as half2 words), converted to float in shared memory,
// and used by all 16 rows. Scores: lane = key, both rows of the warp share the K reads;
// online softmax per tile (one max and one sum per row and tile); P x V: lanes hold
// head_dim / 32 dimensions. head_dim % 32 == 0 and <= 128 (the host checks).
// Grid (n_heads, (M + 15) / 16), block 256; shared 16.5 + 16 + 8 KB.
#define AQ 16  // query rows per block
#define AK 32  // keys per tile
#define AD 128 // max head_dim
#ifndef VL_EMULATE
__device__ __forceinline__ float vl_wmax_all(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float vl_shfl(float v, int src) { return __shfl_sync(0xffffffffu, v, src); }
#endif
extern "C" __global__ void __launch_bounds__(256)
vira_cu_attn_f16kv_tile(const float *__restrict__ q, const unsigned short *__restrict__ kc,
                        const unsigned short *__restrict__ vc, float *__restrict__ out, int n_heads, int kv_mul,
                        int head_dim, int kv_dim, int M, long long start, float scale) {
  __shared__ float sQ[AQ][AD];
  __shared__ float sK[AK][AD + 1]; // +1: lane j reads row j, all lanes in different banks
  __shared__ float sV[AK][AD];
  const int h = blockIdx.x, t0 = blockIdx.y * AQ;
  const int tid = threadIdx.x, lane = tid & (WARP - 1), wp = tid / WARP;
  const int kvh = h / kv_mul, nd = head_dim / WARP, hw = head_dim / 2;
  for (int i = tid; i < AQ * head_dim; i += 256) {
    const int r = i / head_dim, d = i - r * head_dim;
    sQ[r][d] = t0 + r < M ? q[((size_t)(t0 + r) * n_heads + h) * head_dim + d] * scale : 0.0f;
  }
  const int rows = M - t0 < AQ ? M - t0 : AQ;
  const long long last = start + t0 + rows - 1; // last key a row of the block sees
  float mx[2] = {AT_NEG_INF, AT_NEG_INF}, l[2] = {0.0f, 0.0f}, acc[2][AD / WARP];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int k = 0; k < AD / WARP; k++) acc[r][k] = 0.0f;
  for (long long j0 = 0; j0 <= last; j0 += AK) {
    VL_SYNC(); // the previous tile is used up (and sQ written)
    for (int i = tid; i < AK * hw; i += 256) {
      const int jj = i / hw, d2 = i - jj * hw;
      const long long j = j0 + jj;
      unsigned int kw = 0u, vw = 0u;
      if (j <= last) {
        const size_t off = (size_t)j * (size_t)kv_dim + (size_t)kvh * head_dim + 2 * (size_t)d2;
        kw = __ldg((const unsigned int *)(kc + off));
        vw = __ldg((const unsigned int *)(vc + off));
      }
      sK[jj][2 * d2] = vl_h2f((unsigned short)(kw & 0xFFFFu));
      sK[jj][2 * d2 + 1] = vl_h2f((unsigned short)(kw >> 16));
      sV[jj][2 * d2] = vl_h2f((unsigned short)(vw & 0xFFFFu));
      sV[jj][2 * d2 + 1] = vl_h2f((unsigned short)(vw >> 16));
    }
    VL_SYNC();
    float s[2] = {0.0f, 0.0f};
    for (int d = 0; d < head_dim; d++) {
      const float kd = sK[lane][d];
      s[0] += sQ[2 * wp][d] * kd;
      s[1] += sQ[2 * wp + 1][d] * kd;
    }
    float p[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int t = t0 + 2 * wp + r;
      if (t >= M || j0 + lane > start + t) s[r] = AT_NEG_INF; // causal (and rows past M)
      const float mn = fmaxf(mx[r], vl_wmax_all(s[r]));
      float corr = 1.0f;
      p[r] = 0.0f;
      if (mn != AT_NEG_INF) { // a row past M sees no key
        corr = __expf(mx[r] - mn);
        p[r] = __expf(s[r] - mn);
      }
      l[r] = l[r] * corr + vl_wsum(p[r]);
      mx[r] = mn;
#pragma unroll
      for (int k = 0; k < AD / WARP; k++) acc[r][k] *= corr;
    }
    for (int jj = 0; jj < AK; jj++) {
      const float p0 = vl_shfl(p[0], jj), p1 = vl_shfl(p[1], jj);
#pragma unroll
      for (int k = 0; k < AD / WARP; k++) {
        if (k < nd) {
          const float v = sV[jj][lane + WARP * k];
          acc[0][k] += p0 * v;
          acc[1][k] += p1 * v;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int t = t0 + 2 * wp + r;
    if (t >= M) continue;
    float *orow = out + ((size_t)t * n_heads + h) * head_dim;
    const float inv = 1.0f / l[r];
#pragma unroll
    for (int k = 0; k < AD / WARP; k++)
      if (k < nd) orow[lane + WARP * k] = acc[r][k] * inv;
  }
}
