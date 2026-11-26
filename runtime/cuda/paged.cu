// PagedAttention over an FP16 KV block pool, per-token RoPE + scatter write, CoW block copy,
// and row gather for selective LM-head projection.
// Source of runtime/vira_cuda_paged_ptx.h (python3 runtime/cuda/build_ptx.py).
//
// Block pool layout: k_pool and v_pool are [num_blocks * page_size, kv_dim] row-major in IEEE
// half (unsigned short), where physical row = block_id * page_size + page_offset.

#define PG_WARP 32
#define PG_THREADS 256
#define PG_WARPS (PG_THREADS / PG_WARP)
#define PG_MAX_D 8 // head_dim / 32 values per lane: head_dim <= 256
#define PG_NEG_INF __int_as_float((int)0xff800000)

#ifdef VL_EMULATE
#include "emulate.h"
#else
#define VL_SYNC() __syncthreads()
__device__ __forceinline__ float pg_h2f(unsigned short h) {
  float f;
  asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short pg_f2h(float f) {
  unsigned short h;
  asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float pg_wsum(float v) {
#pragma unroll
  for (int o = PG_WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
#endif

#ifdef VL_EMULATE
#define pg_h2f vl_h2f
#define pg_f2h vl_f2h
#define pg_wsum vl_wsum
#endif

// Fused per-token RoPE (positions[t], NeoX or normal adjacent pairs, partial n_rot <= head_dim)
// on q_heads [T * n_heads, head_dim] and k_heads [T * n_kv_heads, head_dim], plus scatter write
// of k_heads[t] and v[t, kv_dim] into physical slot slot_map[t] of FP16 k_pool and v_pool.
// Grid (T, 1), block 256.
extern "C" __global__ void __launch_bounds__(PG_THREADS)
vira_cu_paged_rope_and_write_f16(float *__restrict__ q_heads, float *__restrict__ k_heads,
                                 const float *__restrict__ v, unsigned short *__restrict__ k_pool,
                                 unsigned short *__restrict__ v_pool, const int *__restrict__ positions,
                                 const int *__restrict__ slot_map, int T, int n_heads, int n_kv_heads,
                                 int head_dim, int n_rot, int neox, float freq_base, float freq_scale) {
  const int t = blockIdx.x;
  if (t >= T) return;
  const int tid = threadIdx.x;
  const int pos = positions[t];
  const int phys_row = slot_map[t];
  const int kv_dim = n_kv_heads * head_dim;
  const int half_rot = n_rot / 2;

  // 1. RoPE on q_heads[t * n_heads .. (t + 1) * n_heads]
  if (half_rot > 0) {
    const int q_pairs = n_heads * half_rot;
    for (int idx = tid; idx < q_pairs; idx += PG_THREADS) {
      const int h = idx / half_rot;
      const int i = idx - h * half_rot;
      const float theta = ((float)pos * freq_scale) * powf(freq_base, -2.0f * (float)i / (float)n_rot);
      const float c = cosf(theta);
      const float s = sinf(theta);
      float *row = q_heads + ((size_t)t * n_heads + h) * head_dim;
      if (neox) {
        const float x0 = row[i], x1 = row[i + half_rot];
        row[i] = x0 * c - x1 * s;
        row[i + half_rot] = x0 * s + x1 * c;
      } else {
        const float x0 = row[2 * i], x1 = row[2 * i + 1];
        row[2 * i] = x0 * c - x1 * s;
        row[2 * i + 1] = x0 * s + x1 * c;
      }
    }

    // 2. RoPE on k_heads[t * n_kv_heads .. (t + 1) * n_kv_heads]
    const int k_pairs = n_kv_heads * half_rot;
    for (int idx = tid; idx < k_pairs; idx += PG_THREADS) {
      const int h = idx / half_rot;
      const int i = idx - h * half_rot;
      const float theta = ((float)pos * freq_scale) * powf(freq_base, -2.0f * (float)i / (float)n_rot);
      const float c = cosf(theta);
      const float s = sinf(theta);
      float *row = k_heads + ((size_t)t * n_kv_heads + h) * head_dim;
      if (neox) {
        const float x0 = row[i], x1 = row[i + half_rot];
        row[i] = x0 * c - x1 * s;
        row[i + half_rot] = x0 * s + x1 * c;
      } else {
        const float x0 = row[2 * i], x1 = row[2 * i + 1];
        row[2 * i] = x0 * c - x1 * s;
        row[2 * i + 1] = x0 * s + x1 * c;
      }
    }
  }
  VL_SYNC();

  // 3. Coalesced FP16 scatter write of rotated K and V into physical pool slot
  if (phys_row >= 0) {
    const float *k_src = k_heads + (size_t)t * kv_dim;
    const float *v_src = v + (size_t)t * kv_dim;
    unsigned short *k_dst = k_pool + (size_t)phys_row * kv_dim;
    unsigned short *v_dst = v_pool + (size_t)phys_row * kv_dim;
    for (int c = tid; c < kv_dim; c += PG_THREADS) {
      k_dst[c] = pg_f2h(k_src[c]);
      v_dst[c] = pg_f2h(v_src[c]);
    }
  }
}

// Copy-on-Write block clone: copies n_rows (1..page_size) rows from src_block to dst_block
// inside k_pool and v_pool. Grid ((n_elems + 255) / 256, 1), block 256.
extern "C" __global__ void vira_cu_paged_copy_block_f16(unsigned short *__restrict__ k_pool,
                                                        unsigned short *__restrict__ v_pool,
                                                        int dst_block, int src_block,
                                                        int page_size, int kv_dim, int n_rows) {
  const long long n = (long long)n_rows * kv_dim;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const size_t dst_off = (size_t)dst_block * page_size * kv_dim + (size_t)i;
  const size_t src_off = (size_t)src_block * page_size * kv_dim + (size_t)i;
  k_pool[dst_off] = k_pool[src_off];
  v_pool[dst_off] = v_pool[src_off];
}

// Selective row gather: dst[r, cols] = src[row_idx[r], cols] for r < n_out.
// Avoids projecting intermediate prefill tokens through the vocabulary LM head.
extern "C" __global__ void vira_cu_paged_gather_rows_f32(const float *__restrict__ src,
                                                         float *__restrict__ dst,
                                                         const int *__restrict__ row_idx,
                                                         int n_out, int cols) {
  const long long n = (long long)n_out * cols;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int r = (int)(i / cols);
  const int c = (int)(i - (long long)r * cols);
  const int src_r = row_idx[r];
  dst[i] = src[(size_t)src_r * cols + c];
}

// Unified Ragged + Paged GQA Attention over an FP16 block pool.
// Serves BOTH batched 1-token decode AND mixed ragged chunked-prefill + decode in one launch:
//   q:            [T * n_heads, head_dim] (float32)
//   k_pool/v_pool:[num_blocks * page_size, kv_dim] (FP16)
//   q_seq_id:     [T] sequence index b (0 .. B - 1) for query token t
//   q_max_kv:     [T] causal KV length visible to query token t (pos[t] + 1)
//   block_tables: [B * max_blocks] physical block IDs for sequence b
//   sinks:        optional [n_heads] attention sink logits (NULL when has_sinks == 0)
// Grid (n_heads, T), block 256.
extern "C" __global__ void __launch_bounds__(PG_THREADS)
vira_cu_paged_attn_ragged_f16(const float *__restrict__ q, const unsigned short *__restrict__ k_pool,
                              const unsigned short *__restrict__ v_pool, float *__restrict__ out,
                              const int *__restrict__ q_seq_id, const int *__restrict__ q_max_kv,
                              const int *__restrict__ block_tables, const float *__restrict__ sinks,
                              int n_heads, int kv_mul, int head_dim, int kv_dim, int T,
                              int max_blocks, int page_size, float scale, int window,
                              float softcap, int has_sinks) {
  __shared__ float s_m[PG_WARPS], s_l[PG_WARPS];
  __shared__ float s_acc[PG_WARPS][PG_MAX_D * PG_WARP];
  const int h = blockIdx.x, t = blockIdx.y;
  if (h >= n_heads || t >= T) return;
  const int tid = threadIdx.x, lane = tid & (PG_WARP - 1), wp = tid / PG_WARP;
  const int nd = (head_dim + PG_WARP - 1) / PG_WARP;
  const int kvh = h / kv_mul;
  const int seq_b = q_seq_id[t];
  const int n_keys = q_max_kv[t];
  int lo = 0;
  if (window > 0 && n_keys > window) lo = n_keys - window;
  const int *bt = block_tables + (size_t)seq_b * max_blocks;

  const float *qr = q + ((size_t)t * n_heads + h) * head_dim;
  const float q_scale = scale;
  float qv[PG_MAX_D], acc[PG_MAX_D];
#pragma unroll
  for (int i = 0; i < PG_MAX_D; i++) {
    const int d = lane + PG_WARP * i;
    qv[i] = (i < nd && d < head_dim) ? qr[d] * q_scale : 0.0f;
    acc[i] = 0.0f;
  }

  float m = PG_NEG_INF, l = 0.0f;
  for (int j = lo + wp; j < n_keys; j += PG_WARPS) {
    const int phys_blk = bt[j / page_size];
    const int blk_off = j % page_size;
    const size_t phys_row = (size_t)phys_blk * page_size + (size_t)blk_off;
    const unsigned short *kr = k_pool + phys_row * kv_dim + (size_t)kvh * head_dim;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < PG_MAX_D; i++) {
      const int d = lane + PG_WARP * i;
      if (i < nd && d < head_dim) s += qv[i] * pg_h2f(kr[d]);
    }
    s = pg_wsum(s);
    if (softcap > 0.0f) {
      s = softcap * tanhf(s / softcap);
    }
    const float mn = fmaxf(m, s);
    const float corr = __expf(m - mn), p = __expf(s - mn);
    l = l * corr + p;
    const unsigned short *vr = v_pool + phys_row * kv_dim + (size_t)kvh * head_dim;
#pragma unroll
    for (int i = 0; i < PG_MAX_D; i++) {
      const int d = lane + PG_WARP * i;
      if (i < nd && d < head_dim) acc[i] = acc[i] * corr + p * pg_h2f(vr[d]);
    }
    m = mn;
  }

  if (lane == 0) {
    s_m[wp] = m;
    s_l[wp] = l;
  }
#pragma unroll
  for (int i = 0; i < PG_MAX_D; i++) s_acc[wp][lane + PG_WARP * i] = acc[i];
  VL_SYNC();

  if (wp == 0) {
    float M_ = (has_sinks && sinks != NULL) ? sinks[h] : PG_NEG_INF;
    for (int w = 0; w < PG_WARPS; w++) M_ = fmaxf(M_, s_m[w]);
    float L = (has_sinks && sinks != NULL && M_ != PG_NEG_INF) ? __expf(sinks[h] - M_) : 0.0f;
    float o[PG_MAX_D];
#pragma unroll
    for (int i = 0; i < PG_MAX_D; i++) o[i] = 0.0f;
    for (int w = 0; w < PG_WARPS; w++) {
      if (s_l[w] == 0.0f) continue;
      const float c = __expf(s_m[w] - M_);
      L += s_l[w] * c;
#pragma unroll
      for (int i = 0; i < PG_MAX_D; i++) o[i] += s_acc[w][lane + PG_WARP * i] * c;
    }
    float *orow = out + ((size_t)t * n_heads + h) * head_dim;
    const float inv = L > 0.0f ? (1.0f / L) : 0.0f;
#pragma unroll
    for (int i = 0; i < PG_MAX_D; i++) {
      const int d = lane + PG_WARP * i;
      if (i < nd && d < head_dim) orow[d] = o[i] * inv;
    }
  }
}
