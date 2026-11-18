// Fused small kernels for decode. Source of runtime/vira_cuda_fused_ptx.h
// (python3 runtime/cuda/build_ptx.py; same toolchain notes as mmvq2.cu).

#define WARP 32

#ifdef VL_EMULATE
#include "emulate.h"
#else
__device__ __forceinline__ float vl_rsqrt_approx(float x) {
  float r;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float vl_fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ float vl_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float vl_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float vl_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float vl_warp_max(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ int vl_warp_sum_i(int v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
#define VL_SYNC() __syncthreads()
#endif

// RMSNorm of one row that also writes the row
// as Q8_1 blocks {f32 d, f32 d * sum(q), int8 q[32]}, so the next matrix-vector product
// does not need its own quantize launch. Both halves repeat the arithmetic of the separate
// kernels exactly (vira_cu_rmsnorm_f32: strided fma sum, shared-memory tree, sum / cols,
// + eps, rsqrt.approx, (r * x) * w; vira_cu_quantize_q8_1: warp absmax, q = rni(v * (127 /
// amax)) clamped to +-127, d = amax * (1 / 127)), so results do not change. cols % 32 == 0.
// Grid ((cols / 32 + 7) / 8, rows), block 256: block x handles Q8_1 blocks 8x .. 8x + 7 of
// row y, one per warp. Every block repeats the row's sum of squares in the same order (the
// row is 20 KB at most, read from L2), so all of them get the same r; one block per row ran
// the whole decode norm on a single SM (T4: 28.8 us vs 10.4 + 9.4 for the two kernels).
extern "C" __global__ void __launch_bounds__(256)
vira_cu_rmsnorm_q8_1_f32(const float *__restrict__ x, const float *__restrict__ w,
                         float *__restrict__ y, unsigned char *__restrict__ q8, int cols, float eps) {
  __shared__ float sdata[256];
  const int tid = threadIdx.x;
  const int row = blockIdx.y;
  const float *xr = x + (size_t)row * (size_t)cols;
  float *yr = y + (size_t)row * (size_t)cols;
  float acc = 0.0f;
  for (int i = tid; i < cols; i += 256) {
    const float v = __ldg(xr + i);
    acc = vl_fma(v, v, acc);
  }
  sdata[tid] = acc;
  VL_SYNC();
  for (int o = 128; o > 0; o >>= 1) {
    if (tid < o) sdata[tid] = vl_add(sdata[tid + o], sdata[tid]);
    VL_SYNC();
  }
  const float r = vl_rsqrt_approx(vl_add(eps, vl_div(sdata[0], (float)cols)));
  // Q8_1: warp k quantises blocks k, k + 8, ...; lane l owns value 32 * blk + l
  const int lane = tid & (WARP - 1);
  const int nblk = cols / 32;
  unsigned char *qr = q8 + (size_t)row * (size_t)nblk * 40u;
  const int blk = blockIdx.x * (256 / WARP) + tid / WARP;
  if (blk < nblk) { // after the last barrier, so a warp may leave
    const int i = blk * 32 + lane;
    const float v = vl_mul(vl_mul(r, __ldg(xr + i)), __ldg(w + i));
    yr[i] = v;
    const float amax = vl_warp_max(fabsf(v));
    int q = 0;
    if (amax > 0.0f) {
      q = __float2int_rn(vl_mul(v, vl_div(127.0f, amax)));
      q = max(min(q, 127), -127);
    }
    const int sum = vl_warp_sum_i(q);
    unsigned char *qb = qr + (size_t)blk * 40u;
    qb[8 + lane] = (unsigned char)(signed char)q;
    if (lane == 0) {
      const float d = vl_mul(amax, __int_as_float(0x3C010204));
      ((float *)qb)[0] = d;
      ((float *)qb)[1] = vl_mul(d, (float)sum);
    }
  }
}

// MoE expert activation, in place: g[i] = act(g[i], u[i]) for i < n, the formulas of
// ViraLama's CPU path (vl_moe_full): 0 SiLU(g) * u, 1 GELU-tanh(g) * u, 2 gpt-oss
// (g clamped to 7, u to +-7: g * sigmoid(1.702 g) * (u + 1)), 3 DeepSeek-V4 clamped SwiGLU.
extern "C" __global__ void vira_cu_moe_act_f32(float *__restrict__ g, const float *__restrict__ u,
                                              int n, int act, float limit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = g[i], b = u[i];
  float r;
  if (act == 3) {
    float a = v, c = b;
    if (limit > 1e-6f) {
      a = a < limit ? a : limit;
      c = c < -limit ? -limit : (c > limit ? limit : c);
    }
    r = a / (1.0f + expf(-a)) * c;
  } else if (act == 2) {
    const float xg = v < 7.0f ? v : 7.0f;
    const float yu = b < -7.0f ? -7.0f : (b > 7.0f ? 7.0f : b);
    r = xg / (1.0f + expf(-1.702f * xg)) * (yu + 1.0f);
  } else if (act == 1) {
    const float k0 = 0.7978845608028654f;
    r = 0.5f * v * (1.0f + tanhf(k0 * (v + 0.044715f * v * v * v))) * b;
  } else {
    r = v / (1.0f + expf(-v)) * b;
  }
  g[i] = r;
}

// MoE weighted sum: out[t, d] = sum_k wt[t * n_used + k] * y[pos[t * n_used + k], d].
// Grid ((D + 255) / 256, M), block 256.
extern "C" __global__ void vira_cu_moe_sum_f32(const float *__restrict__ y, const float *__restrict__ wt,
                                              const int *__restrict__ pos, float *__restrict__ out,
                                              int M, int n_used, int D) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (d >= D || t >= M) return;
  float acc = 0.0f;
  for (int k = 0; k < n_used; k++) {
    const int i = t * n_used + k;
    acc += wt[i] * y[(size_t)pos[i] * (size_t)D + d];
  }
  out[(size_t)t * (size_t)D + d] = acc;
}
