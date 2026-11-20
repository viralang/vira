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
__device__ __forceinline__ float vl_warp_sum_f(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float vl_ex2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
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

// out[i] = silu(g[i]) * u[i], i < n: one launch instead of vira_cu_silu_f32 and
// vira_cu_elemwise_mul_f32 (gpu_silu_mul). Same arithmetic as the two (silu: x / (1 +
// ex2.approx(x * -log2 e)) with an IEEE division, then one IEEE multiply), so results do not
// change. Grid (n + 255) / 256, block 256.
extern "C" __global__ void vira_cu_silu_mul_f32(const float *__restrict__ g, const float *__restrict__ u,
                                               float *__restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = g[i];
  const float e = vl_ex2_approx(vl_mul(x, __int_as_float(0xBFB8AA3B)));
  out[i] = vl_mul(vl_div(x, vl_add(e, 1.0f)), u[i]);
}

// RMSNorm of prompt rows that also writes the rows quantized in the x4 layout of mmqt.cu
// (per row and 128 values {f32 d[4], f32 s[4], int8 q[128]}), so the next prompt-size
// product needs no quantize launch. Norm arithmetic as vira_cu_rmsnorm_f32 (strided fma sum,
// shared-memory tree, sum / cols, + eps, rsqrt.approx, (r * x) * w), quantization as
// vira_cu_quantize_q8_1x4 (warp absmax, q = rni(v * (127 / amax)) clamped to +-127, d = amax
// * (1 / 127), s = d * sum(q)). cols % 128 == 0. Grid rows, block 256: warp k quantizes the
// row's 32-value blocks k, k + 8, ...
extern "C" __global__ void __launch_bounds__(256)
vira_cu_rmsnorm_x4_f32(const float *__restrict__ x, const float *__restrict__ w, float *__restrict__ y,
                       unsigned char *__restrict__ q4, int cols, float eps) {
  __shared__ float sdata[256];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
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
  const int lane = tid & (WARP - 1), wp = tid / WARP;
  const int nblk = cols / 32;
  unsigned char *qr = q4 + (size_t)row * (size_t)(cols / 128) * 160u;
  for (int blk = wp; blk < nblk; blk += 256 / WARP) {
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
    unsigned char *rec = qr + (size_t)(blk / 4) * 160u;
    rec[32 + 32 * (blk % 4) + lane] = (unsigned char)(signed char)q;
    if (lane == 0) {
      const float d = vl_mul(amax, __int_as_float(0x3C010204));
      ((float *)rec)[blk % 4] = d;
      ((float *)rec)[4 + blk % 4] = vl_mul(d, (float)sum);
    }
  }
}

// RoPE (NeoX half-split and Norm adjacent-pair) reading pos from a 4-byte device pointer
// so CUDA Graphs can replay decode tokens without re-capturing when pos increments.
// Grid ((half_dim + 15) / 16, (rows + 15) / 16), block (16, 16).
extern "C" __global__ void vira_cu_rope_dpos_f32(const float *__restrict__ q, float *__restrict__ out,
                                                 int rows, int cols, int half_dim,
                                                 const int *__restrict__ pos_ptr, float freq_base) {
  const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int r = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  if (i >= half_dim || r >= rows) return;
  const float pos_f = (float)(*pos_ptr);
  const float angle = pos_f * powf(freq_base, -(2.0f * (float)i) / (float)cols);
  const float c = cosf(angle), s = sinf(angle);
  const size_t base = (size_t)r * (size_t)cols;
  const float q0 = q[base + (size_t)i];
  const float q1 = q[base + (size_t)(i + half_dim)];
  out[base + (size_t)i] = q0 * c - q1 * s;
  out[base + (size_t)(i + half_dim)] = q0 * s + q1 * c;
}
extern "C" __global__ void vira_cu_rope_norm_dpos_f32(const float *__restrict__ q, float *__restrict__ out,
                                                      int rows, int cols, int half_dim,
                                                      const int *__restrict__ pos_ptr, float freq_base) {
  const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int r = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
  if (i >= half_dim || r >= rows) return;
  const float pos_f = (float)(*pos_ptr);
  const float angle = pos_f * powf(freq_base, -(2.0f * (float)i) / (float)cols);
  const float c = cosf(angle), s = sinf(angle);
  const size_t base = (size_t)r * (size_t)cols + (size_t)(2 * i);
  const float q0 = q[base];
  const float q1 = q[base + 1u];
  out[base] = q0 * c - q1 * s;
  out[base + 1u] = q0 * s + q1 * c;
}

// Squared ReLU in place: x[i] = max(x[i], 0)^2 (Nemotron-H MLP). Grid (n + 255) / 256, block 256.
extern "C" __global__ void vira_cu_relu_sqr_f32(float *__restrict__ x, int n) {
  const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (i >= n) return;
  const float v = x[i] > 0.0f ? x[i] : 0.0f;
  x[i] = v * v;
}

// Mamba-2 stage 1: causal 1D depthwise conv + optional bias + SiLU across M tokens.
// One thread per channel c in [0, d_conv_ch); conv_state [d_conv - 1, d_conv_ch] stays in
// registers across all M tokens and is written back once. Requires 1 <= d_conv <= 8.
// Grid (d_conv_ch + 255) / 256, block 256.
extern "C" __global__ void vira_cu_mamba2_conv_f32(const float *__restrict__ zxBCdt,
                                                   const float *__restrict__ conv_w,
                                                   const float *__restrict__ conv_b, int has_conv_b,
                                                   float *__restrict__ conv_state,
                                                   float *__restrict__ xbc_act,
                                                   int M, int start_pos, int d_inner,
                                                   int d_conv_ch, int d_in_proj, int d_conv) {
  const int c = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (c >= d_conv_ch) return;
  const int hist = d_conv - 1;
  float cw[8];
  const float *cwp = conv_w + (size_t)c * (size_t)d_conv;
  for (int k = 0; k < d_conv; k++) cw[k] = cwp[k];
  const float bias = has_conv_b ? conv_b[c] : 0.0f;
  float st[8];
  for (int k = 0; k < hist; k++) {
    st[k] = (start_pos == 0) ? 0.0f : conv_state[(size_t)k * (size_t)d_conv_ch + (size_t)c];
  }
  for (int t = 0; t < M; t++) {
    const float x_in = zxBCdt[(size_t)t * (size_t)d_in_proj + (size_t)d_inner + (size_t)c];
    float acc = bias;
    for (int k = 0; k < hist; k++) acc += st[k] * cw[k];
    acc += x_in * cw[hist];
    xbc_act[(size_t)t * (size_t)d_conv_ch + (size_t)c] = acc / (1.0f + expf(-acc));
    for (int k = 0; k < hist - 1; k++) st[k] = st[k + 1];
    if (hist > 0) st[hist - 1] = x_in;
  }
  for (int k = 0; k < hist; k++) {
    conv_state[(size_t)k * (size_t)d_conv_ch + (size_t)c] = st[k];
  }
}

// Mamba-2 stage 2: SSM recurrence + D skip + z SwiGLU gate across M tokens.
// One warp (32 lanes) per output channel idx in [0, d_inner); block 128 = 4 warps.
// Each lane holds up to 8 state elements (d_state <= 256) in registers across all M tokens.
// Grid (d_inner + 3) / 4, block 128.
extern "C" __global__ void __launch_bounds__(128)
vira_cu_mamba2_ssm_f32(const float *__restrict__ zxBCdt, const float *__restrict__ xbc_act,
                       const float *__restrict__ dt_b, const float *__restrict__ ssm_a,
                       const float *__restrict__ ssm_d, float *__restrict__ ssm_state,
                       float *__restrict__ out, int M, int start_pos, int d_inner,
                       int d_state, int n_head, int n_group, int d_conv_ch, int d_in_proj) {
  const int wp = (int)threadIdx.x / WARP;
  const int lane = (int)threadIdx.x & (WARP - 1);
  const int idx = (int)blockIdx.x * 4 + wp;
  if (idx >= d_inner) return;

  const int head_dim = d_inner / n_head;
  const int h = idx / head_dim;
  const int g = h / (n_head / n_group);
  const float a_h = ssm_a[h];
  const float d_h = ssm_d[h];
  const float dtb_h = dt_b[h];
  const int n_chunks = (d_state + WARP - 1) / WARP;

  float s[8];
  float *st_row = ssm_state + (size_t)idx * (size_t)d_state;
  for (int k = 0; k < n_chunks; k++) {
    const int n = lane + k * WARP;
    s[k] = (start_pos == 0 || n >= d_state) ? 0.0f : st_row[n];
  }

  for (int t = 0; t < M; t++) {
    const float *zx_row = zxBCdt + (size_t)t * (size_t)d_in_proj;
    const float *act_row = xbc_act + (size_t)t * (size_t)d_conv_ch;
    const float dt_raw = zx_row[d_inner + d_conv_ch + h] + dtb_h;
    const float dt_h = dt_raw > 20.0f ? dt_raw : log1pf(expf(dt_raw));
    const float dA = expf(dt_h * a_h);
    const float x_hp = act_row[idx];
    const float dt_x = dt_h * x_hp;
    const float *Bg = act_row + d_inner + (size_t)g * (size_t)d_state;
    const float *Cg = act_row + d_inner + (size_t)(n_group + g) * (size_t)d_state;

    float acc = 0.0f;
    for (int k = 0; k < n_chunks; k++) {
      const int n = lane + k * WARP;
      if (n < d_state) {
        s[k] = s[k] * dA + dt_x * Bg[n];
        acc += s[k] * Cg[n];
      }
    }
    const float sum_y = vl_warp_sum_f(acc);
    if (lane == 0) {
      const float z_v = zx_row[idx];
      const float y_hp = sum_y + d_h * x_hp;
      out[(size_t)t * (size_t)d_inner + (size_t)idx] = (z_v / (1.0f + expf(-z_v))) * y_hp;
    }
  }

  for (int k = 0; k < n_chunks; k++) {
    const int n = lane + k * WARP;
    if (n < d_state) st_row[n] = s[k];
  }
}

// Mamba-2 stage 3: optional grouped RMSNorm in place on out [M, d_inner] with n_group groups.
// Grid (n_group, M), block 256.
extern "C" __global__ void __launch_bounds__(256)
vira_cu_mamba2_norm_f32(float *__restrict__ out, const float *__restrict__ w,
                        int d_inner, int n_group, float eps) {
  __shared__ float sdata[256];
  const int tid = (int)threadIdx.x;
  const int g = (int)blockIdx.x;
  const int t = (int)blockIdx.y;
  const int gsize = d_inner / n_group;
  float *yg = out + (size_t)t * (size_t)d_inner + (size_t)g * (size_t)gsize;
  const float *wg = w + (size_t)g * (size_t)gsize;

  float acc = 0.0f;
  for (int i = tid; i < gsize; i += 256) {
    const float v = yg[i];
    acc += v * v;
  }
  sdata[tid] = acc;
  VL_SYNC();
  for (int o = 128; o > 0; o >>= 1) {
    if (tid < o) sdata[tid] += sdata[tid + o];
    VL_SYNC();
  }
  const float inv = vl_rsqrt_approx(sdata[0] / (float)gsize + eps);
  for (int i = tid; i < gsize; i += 256) {
    yg[i] = yg[i] * inv * wg[i];
  }
}

// ============================================================================
// DeepSeek MLA attention kernels (T-128): keep latent cache + Q/K/V on GPU.
// ============================================================================

__device__ __forceinline__ void vl_mla_yarn_cs(int pair_idx, int pos, int n_rot,
                                               float freq_base, float freq_scale,
                                               float ext_factor, float attn_factor,
                                               float corr0, float corr1,
                                               float &cs, float &sn) {
  const float theta_extrap = (float)pos * powf(freq_base, -(2.0f * (float)pair_idx) / (float)n_rot);
  const float theta_interp = freq_scale * theta_extrap;
  float th = theta_interp;
  float mscale = attn_factor;
  if (ext_factor != 0.0f) {
    float denom = corr1 - corr0;
    if (denom < 0.001f) denom = 0.001f;
    float y = ((float)pair_idx - corr0) / denom;
    float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
    float mix = ramp * ext_factor;
    th = theta_interp * (1.0f - mix) + theta_extrap * mix;
    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
  }
  cs = cosf(th) * mscale;
  sn = sinf(th) * mscale;
}

// MLA stage 1: cache[pos0+t, 0..kv_lora] = rmsnorm(kvpe[t, 0..kv_lora], kv_norm_w, eps),
//              cache[pos0+t, kv_lora..stride] = rope_norm_yarn(kvpe[t, kv_lora..stride], pos0+t).
// Grid M, block 256.
extern "C" __global__ void __launch_bounds__(256)
vira_cu_mla_kv_store_f32(const float *__restrict__ kvpe, const float *__restrict__ kv_norm_w,
                         float *__restrict__ cache, int pos0, int kv_lora, int rope, float eps,
                         float freq_base, float freq_scale, float ext_factor, float attn_factor,
                         float corr0, float corr1) {
  __shared__ float sdata[256];
  const int tid = (int)threadIdx.x;
  const int t = (int)blockIdx.x;
  const int stride = kv_lora + rope;
  const float *src = kvpe + (size_t)t * (size_t)stride;
  float *dst = cache + (size_t)(pos0 + t) * (size_t)stride;

  float acc = 0.0f;
  for (int c = tid; c < kv_lora; c += 256) {
    const float v = src[c];
    acc += v * v;
  }
  sdata[tid] = acc;
  VL_SYNC();
  for (int o = 128; o > 0; o >>= 1) {
    if (tid < o) sdata[tid] += sdata[tid + o];
    VL_SYNC();
  }
  const float inv = vl_rsqrt_approx(sdata[0] / (float)kv_lora + eps);
  for (int c = tid; c < kv_lora; c += 256) {
    dst[c] = src[c] * inv * kv_norm_w[c];
  }
  const int half_r = rope / 2;
  for (int j = tid; j < half_r; j += 256) {
    float cs, sn;
    vl_mla_yarn_cs(j, pos0 + t, rope, freq_base, freq_scale, ext_factor, attn_factor, corr0, corr1, cs, sn);
    const float x0 = src[kv_lora + 2 * j];
    const float x1 = src[kv_lora + 2 * j + 1];
    dst[kv_lora + 2 * j]     = x0 * cs - x1 * sn;
    dst[kv_lora + 2 * j + 1] = x0 * sn + x1 * cs;
  }
}

// MLA stage 2: split q [M, nh, nope + rope] into qn [nh, M, nope] and roped qpe [M, nh, rope].
// Grid (nh, M), block 128.
extern "C" __global__ void __launch_bounds__(128)
vira_cu_mla_q_split_f32(const float *__restrict__ q, float *__restrict__ qn, float *__restrict__ qpe,
                        int M, int nh, int nope, int rope, int pos0,
                        float freq_base, float freq_scale, float ext_factor, float attn_factor,
                        float corr0, float corr1) {
  const int tid = (int)threadIdx.x;
  const int h = (int)blockIdx.x;
  const int t = (int)blockIdx.y;
  const int qd = nope + rope;
  const float *qr = q + ((size_t)t * (size_t)nh + (size_t)h) * (size_t)qd;
  float *qnr = qn + ((size_t)h * (size_t)M + (size_t)t) * (size_t)nope;
  float *qper = qpe + ((size_t)t * (size_t)nh + (size_t)h) * (size_t)rope;

  for (int i = tid; i < nope; i += 128) qnr[i] = qr[i];
  const int half_r = rope / 2;
  for (int j = tid; j < half_r; j += 128) {
    float cs, sn;
    vl_mla_yarn_cs(j, pos0 + t, rope, freq_base, freq_scale, ext_factor, attn_factor, corr0, corr1, cs, sn);
    const float x0 = qr[nope + 2 * j];
    const float x1 = qr[nope + 2 * j + 1];
    qper[2 * j]     = x0 * cs - x1 * sn;
    qper[2 * j + 1] = x0 * sn + x1 * cs;
  }
}

// MLA F32 per-head batched linear projection: x [nh, M, K] * w [nh, N, K]^T -> y.
// out_t_major == 0: y [nh, M, N]; out_t_major == 1: y [M, nh, N].
// Grid ((N + 3) / 4, nh * M), block 128 (4 warps).
extern "C" __global__ void __launch_bounds__(128)
vira_cu_mla_proj_f32(const float *__restrict__ x, const float *__restrict__ w, float *__restrict__ y,
                     int M, int nh, int K, int N, int out_t_major) {
  const int wp = (int)threadIdx.x / WARP;
  const int lane = (int)threadIdx.x & (WARP - 1);
  const int n = (int)blockIdx.x * 4 + wp;
  const int hm = (int)blockIdx.y;
  if (n >= N) return;
  const int h = hm / M;
  const int t = hm - h * M;
  const float *xr = x + (size_t)hm * (size_t)K;
  const float *wr = w + ((size_t)h * (size_t)N + (size_t)n) * (size_t)K;
  float acc = 0.0f;
  for (int k = lane; k < K; k += WARP) acc += xr[k] * wr[k];
  acc = vl_warp_sum_f(acc);
  if (lane == 0) {
    if (out_t_major) {
      y[((size_t)t * (size_t)nh + (size_t)h) * (size_t)N + (size_t)n] = acc;
    } else {
      y[(size_t)hm * (size_t)N + (size_t)n] = acc;
    }
  }
}

// MLA stage 4: absorbed latent attention over cache [ctx, kv_lora + rope].
// qa: [nh, M, kv_lora], qpe: [M, nh, rope], ctx_out: [nh, M, kv_lora].
// Requires kv_lora <= 1024, rope <= 256. Grid (nh, M), block 256 (8 warps).
extern "C" __global__ void __launch_bounds__(256)
vira_cu_mla_attn_f32(const float *__restrict__ qa, const float *__restrict__ qpe,
                     const float *__restrict__ cache, const float *__restrict__ mask, int has_mask,
                     float *__restrict__ ctx_out, int M, int pos0, int nh, int kv_lora, int rope,
                     float scale) {
  __shared__ float sqa[1024];
  __shared__ float sqpe[256];
  __shared__ float sscores[8];

  const int tid = (int)threadIdx.x;
  const int wp = tid / WARP;
  const int lane = tid & (WARP - 1);
  const int h = (int)blockIdx.x;
  const int t = (int)blockIdx.y;
  const int hm = h * M + t;
  const int stride = kv_lora + rope;
  const int n_keys = pos0 + t + 1;
  const int mask_stride = pos0 + M;

  const float *qar = qa + (size_t)hm * (size_t)kv_lora;
  const float *qper = qpe + ((size_t)t * (size_t)nh + (size_t)h) * (size_t)rope;
  for (int c = tid; c < kv_lora; c += 256) sqa[c] = qar[c];
  for (int c = tid; c < rope; c += 256) sqpe[c] = qper[c];
  VL_SYNC();

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float m_val = -1e30f;
  float l_val = 0.0f;

  for (int k0 = 0; k0 < n_keys; k0 += 8) {
    const int k = k0 + wp;
    float sc = -1e30f;
    if (k < n_keys) {
      bool masked = (has_mask && mask[(size_t)t * (size_t)mask_stride + (size_t)k] < -1e20f);
      if (!masked) {
        const float *cr = cache + (size_t)k * (size_t)stride;
        float dot = 0.0f;
        for (int c = lane; c < kv_lora; c += WARP) dot += sqa[c] * cr[c];
        for (int c = lane; c < rope; c += WARP) dot += sqpe[c] * cr[kv_lora + c];
        sc = vl_warp_sum_f(dot) * scale;
      }
    }
    if (lane == 0) sscores[wp] = sc;
    VL_SYNC();

    const int kn = (n_keys - k0) < 8 ? (n_keys - k0) : 8;
    float bmax = m_val;
    for (int j = 0; j < kn; j++) {
      if (sscores[j] > bmax) bmax = sscores[j];
    }
    const float rescale = (l_val > 0.0f) ? expf(m_val - bmax) : 0.0f;
    acc0 *= rescale;
    acc1 *= rescale;
    acc2 *= rescale;
    acc3 *= rescale;
    l_val *= rescale;
    m_val = bmax;

    for (int j = 0; j < kn; j++) {
      const float sj = sscores[j];
      if (sj <= -1e29f) continue;
      const float wj = expf(sj - m_val);
      l_val += wj;
      const float *cr = cache + (size_t)(k0 + j) * (size_t)stride;
      const int c0 = tid;
      const int c1 = tid + 256;
      const int c2 = tid + 512;
      const int c3 = tid + 768;
      if (c0 < kv_lora) acc0 += wj * cr[c0];
      if (c1 < kv_lora) acc1 += wj * cr[c1];
      if (c2 < kv_lora) acc2 += wj * cr[c2];
      if (c3 < kv_lora) acc3 += wj * cr[c3];
    }
    VL_SYNC();
  }

  const float inv_l = l_val > 0.0f ? (1.0f / l_val) : 0.0f;
  float *co = ctx_out + (size_t)hm * (size_t)kv_lora;
  if (tid < kv_lora) co[tid] = acc0 * inv_l;
  if (tid + 256 < kv_lora) co[tid + 256] = acc1 * inv_l;
  if (tid + 512 < kv_lora) co[tid + 512] = acc2 * inv_l;
  if (tid + 768 < kv_lora) co[tid + 768] = acc3 * inv_l;
}

// Transpose [nh, M, D] -> [M, nh, D] (used when M > 1 and quantized wvb wrote [nh, M, hdv]).
// Grid ((M * nh * D + 255) / 256), block 256.
extern "C" __global__ void vira_cu_mla_hmd_to_mhd_f32(const float *__restrict__ src,
                                                      float *__restrict__ dst,
                                                      int M, int nh, int D) {
  const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int total = M * nh * D;
  if (i >= total) return;
  const int d = i % D;
  const int r = i / D;
  const int h = r % nh;
  const int t = r / nh;
  dst[i] = src[((size_t)h * (size_t)M + (size_t)t) * (size_t)D + (size_t)d];
}
