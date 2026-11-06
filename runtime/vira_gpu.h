#ifndef VIRA_GPU_H
#define VIRA_GPU_H

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "vira_rt.h"

// ============================================================================
// VIRA v0.9 HIGH-THROUGHPUT CROSS-PLATFORM GPU COMPUTE RUNTIME
// ============================================================================
// - Zero compile-time SDK dependencies: dynamically probes OpenCL drivers at
//   runtime via Win32 LoadLibraryA / POSIX dlopen.
// - Coalesced Warp/Wavefront GEMV (256 threads + float4 vectorized loads) for
//   LLM token generation (M=1..4) & 16x16 Shared-Memory Tiled GEMM for prefill.
// - Persistent Activation Scratch Buffers: zero per-layer clCreateBuffer /
//   clReleaseMemObject driver overhead during forward passes.
// - Non-Destructive Partial Offload Weight Cache (4096 slots, LRU): keeps as
//   many GGUF/SafeTensors layers pinned in VRAM as fit; overflow layers fall
//   back seamlessly to Vira's multi-threaded AVX2+FMA / NEON CPU pool.
// ============================================================================

#if defined(_WIN32) && !defined(_WIN64)
#define VIRA_CL_API __stdcall
#else
#define VIRA_CL_API
#endif

typedef int32_t vira_cl_int;
typedef uint32_t vira_cl_uint;
typedef uint64_t vira_cl_ulong;
typedef uint64_t vira_cl_bitfield;
typedef void *vira_cl_platform_id;
typedef void *vira_cl_device_id;
typedef void *vira_cl_context;
typedef void *vira_cl_command_queue;
typedef void *vira_cl_mem;
typedef void *vira_cl_program;
typedef void *vira_cl_kernel;
typedef void *vira_cl_event;

#define VIRA_CL_SUCCESS 0
#define VIRA_CL_DEVICE_TYPE_DEFAULT (1ULL << 0)
#define VIRA_CL_DEVICE_TYPE_CPU (1ULL << 1)
#define VIRA_CL_DEVICE_TYPE_GPU (1ULL << 2)
#define VIRA_CL_DEVICE_TYPE_ACCELERATOR (1ULL << 3)
#define VIRA_CL_DEVICE_TYPE_ALL 0xFFFFFFFFULL

#define VIRA_CL_DEVICE_TYPE_INFO 0x1000
#define VIRA_CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define VIRA_CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define VIRA_CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define VIRA_CL_DEVICE_NAME 0x102B
#define VIRA_CL_DEVICE_VENDOR 0x102C
#define VIRA_CL_DEVICE_VERSION 0x102F
#define VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035

#define VIRA_CL_MEM_READ_WRITE (1ULL << 0)
#define VIRA_CL_MEM_WRITE_ONLY (1ULL << 1)
#define VIRA_CL_MEM_READ_ONLY (1ULL << 2)
#define VIRA_CL_MEM_USE_HOST_PTR (1ULL << 3)
#define VIRA_CL_MEM_ALLOC_HOST_PTR (1ULL << 4)
#define VIRA_CL_MEM_COPY_HOST_PTR (1ULL << 5)

typedef vira_cl_int(VIRA_CL_API *pfn_clGetPlatformIDs)(vira_cl_uint,
                                                       vira_cl_platform_id *,
                                                       vira_cl_uint *);
typedef vira_cl_int(VIRA_CL_API *pfn_clGetDeviceIDs)(vira_cl_platform_id,
                                                     vira_cl_bitfield,
                                                     vira_cl_uint,
                                                     vira_cl_device_id *,
                                                     vira_cl_uint *);
typedef vira_cl_int(VIRA_CL_API *pfn_clGetDeviceInfo)(vira_cl_device_id,
                                                      vira_cl_uint, size_t,
                                                      void *, size_t *);
typedef vira_cl_context(VIRA_CL_API *pfn_clCreateContext)(
    const void *, vira_cl_uint, const vira_cl_device_id *, void *, void *,
    vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseContext)(vira_cl_context);
typedef vira_cl_command_queue(VIRA_CL_API *pfn_clCreateCommandQueue)(
    vira_cl_context, vira_cl_device_id, vira_cl_bitfield, vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseCommandQueue)(
    vira_cl_command_queue);
typedef vira_cl_program(VIRA_CL_API *pfn_clCreateProgramWithSource)(
    vira_cl_context, vira_cl_uint, const char **, const size_t *,
    vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clBuildProgram)(
    vira_cl_program, vira_cl_uint, const vira_cl_device_id *, const char *,
    void *, void *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseProgram)(vira_cl_program);
typedef vira_cl_kernel(VIRA_CL_API *pfn_clCreateKernel)(vira_cl_program,
                                                        const char *,
                                                        vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseKernel)(vira_cl_kernel);
typedef vira_cl_mem(VIRA_CL_API *pfn_clCreateBuffer)(vira_cl_context,
                                                     vira_cl_bitfield, size_t,
                                                     void *, vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseMemObject)(vira_cl_mem);
typedef vira_cl_int(VIRA_CL_API *pfn_clEnqueueWriteBuffer)(
    vira_cl_command_queue, vira_cl_mem, vira_cl_uint, size_t, size_t,
    const void *, vira_cl_uint, const vira_cl_event *, vira_cl_event *);
typedef vira_cl_int(VIRA_CL_API *pfn_clEnqueueReadBuffer)(
    vira_cl_command_queue, vira_cl_mem, vira_cl_uint, size_t, size_t, void *,
    vira_cl_uint, const vira_cl_event *, vira_cl_event *);
typedef vira_cl_int(VIRA_CL_API *pfn_clSetKernelArg)(vira_cl_kernel,
                                                     vira_cl_uint, size_t,
                                                     const void *);
typedef vira_cl_int(VIRA_CL_API *pfn_clEnqueueNDRangeKernel)(
    vira_cl_command_queue, vira_cl_kernel, vira_cl_uint, const size_t *,
    const size_t *, const size_t *, vira_cl_uint, const vira_cl_event *,
    vira_cl_event *);
typedef vira_cl_int(VIRA_CL_API *pfn_clFinish)(vira_cl_command_queue);

#define VIRA_GPU_WEIGHT_CACHE_CAP 4096
#define VIRA_GPU_WEIGHT_PROBE_LEN 64

typedef struct {
  const vira_f32 *host_ptr;
  size_t bytes;
  vira_cl_mem dev_buf;
  uint64_t last_used;
  bool uncacheable; // Set if VRAM is full so we don't retry uploading every token
} _vira_gpu_cache_entry;

typedef struct {
  volatile int32_t initialized;
  bool available;
  bool enabled;
  bool unified_mem;
  bool vram_saturated;
  volatile bool simulate_oom;
  char device_name[256];
  char backend_name[64];
  int64_t vram_bytes;
  int64_t vram_budget_bytes;
  volatile int64_t allocated_bytes;
  volatile int64_t oom_fallbacks;
  int32_t compute_units;
  size_t max_wg_size;
  uint64_t access_clock;
  pthread_mutex_t lock;

  void *lib_handle;
  pfn_clGetPlatformIDs clGetPlatformIDs;
  pfn_clGetDeviceIDs clGetDeviceIDs;
  pfn_clGetDeviceInfo clGetDeviceInfo;
  pfn_clCreateContext clCreateContext;
  pfn_clReleaseContext clReleaseContext;
  pfn_clCreateCommandQueue clCreateCommandQueue;
  pfn_clReleaseCommandQueue clReleaseCommandQueue;
  pfn_clCreateProgramWithSource clCreateProgramWithSource;
  pfn_clBuildProgram clBuildProgram;
  pfn_clReleaseProgram clReleaseProgram;
  pfn_clCreateKernel clCreateKernel;
  pfn_clReleaseKernel clReleaseKernel;
  pfn_clCreateBuffer clCreateBuffer;
  pfn_clReleaseMemObject clReleaseMemObject;
  pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer;
  pfn_clEnqueueReadBuffer clEnqueueReadBuffer;
  pfn_clSetKernelArg clSetKernelArg;
  pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
  pfn_clFinish clFinish;

  vira_cl_device_id device;
  vira_cl_context context;
  vira_cl_command_queue queue;
  vira_cl_program program;

  // High-performance compute kernels
  vira_cl_kernel kernel_gemv_vec4;
  vira_cl_kernel kernel_gemv_scalar;
  vira_cl_kernel kernel_linear_tiled;
  vira_cl_kernel kernel_matmul_tiled;
  vira_cl_kernel kernel_rmsnorm;
  vira_cl_kernel kernel_rope;
  vira_cl_kernel kernel_silu;

  // Persistent reusable device buffers for activations (zero per-op allocation)
  vira_cl_mem scratch_a;
  size_t scratch_a_cap;
  vira_cl_mem scratch_b;
  size_t scratch_b_cap;
  vira_cl_mem scratch_c;
  size_t scratch_c_cap;

  _vira_gpu_cache_entry weight_cache[VIRA_GPU_WEIGHT_CACHE_CAP];
} _vira_gpu_state;

static _vira_gpu_state _vira_gpu = {
    .initialized = 0,
    .available = false,
    .enabled = true,
    .unified_mem = false,
    .vram_saturated = false,
    .simulate_oom = false,
    .device_name = "Multi-Threaded CPU SIMD Fallback",
    .backend_name = "CPU-SIMD",
    .vram_bytes = 0,
    .vram_budget_bytes = 0,
    .allocated_bytes = 0,
    .oom_fallbacks = 0,
    .compute_units = 0,
    .max_wg_size = 256,
    .access_clock = 0,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .lib_handle = NULL,
};

// ============================================================================
// OPTIMIZED OPENCL KERNELS (NVIDIA / AMD / INTEL / APPLE COMPATIBLE)
// ============================================================================
static const char *_vira_opencl_kernel_src =
    "#define WG_SIZE 256\n"
    "#define TILE_DIM 16\n"
    "\n"
    "// 1. Fast Coalesced Float4 GEMV for LLM Decode (M <= 4, K % 4 == 0)\n"
    "// Each work-group of 256 threads computes 1 output element y[m, n].\n"
    "// Threads read adjacent float4 vectors across K (100% coalesced 128B transactions).\n"
    "__kernel void vira_cl_gemv_vec4(\n"
    "    __global const float4* restrict x,\n"
    "    __global const float4* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K4, const int N) {\n"
    "  const int lid = (int)get_local_id(0);\n"
    "  const int n   = (int)get_group_id(0);\n"
    "  const int m   = (int)get_group_id(1);\n"
    "  if (n >= N || m >= M) return;\n"
    "\n"
    "  __global const float4* x_row = x + (size_t)m * (size_t)K4;\n"
    "  __global const float4* w_row = w + (size_t)n * (size_t)K4;\n"
    "\n"
    "  float4 acc0 = (float4)(0.0f);\n"
    "  float4 acc1 = (float4)(0.0f);\n"
    "  int k = lid;\n"
    "  for (; k + WG_SIZE < K4; k += (WG_SIZE * 2)) {\n"
    "    acc0 = fma(x_row[k],           w_row[k],           acc0);\n"
    "    acc1 = fma(x_row[k + WG_SIZE], w_row[k + WG_SIZE], acc1);\n"
    "  }\n"
    "  if (k < K4) {\n"
    "    acc0 = fma(x_row[k], w_row[k], acc0);\n"
    "  }\n"
    "  float4 sum4 = acc0 + acc1;\n"
    "  float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);\n"
    "\n"
    "  __local float sdata[WG_SIZE];\n"
    "  sdata[lid] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {\n"
    "    if (lid < s) {\n"
    "      sdata[lid] += sdata[lid + s];\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  if (lid == 0) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] = sdata[0];\n"
    "  }\n"
    "}\n"
    "\n"
    "// 2. Coalesced Scalar GEMV Fallback (when K is not a multiple of 4)\n"
    "__kernel void vira_cl_gemv_scalar(\n"
    "    __global const float* restrict x,\n"
    "    __global const float* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int lid = (int)get_local_id(0);\n"
    "  const int n   = (int)get_group_id(0);\n"
    "  const int m   = (int)get_group_id(1);\n"
    "  if (n >= N || m >= M) return;\n"
    "\n"
    "  __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "  __global const float* w_row = w + (size_t)n * (size_t)K;\n"
    "  float sum = 0.0f;\n"
    "  for (int k = lid; k < K; k += WG_SIZE) {\n"
    "    sum = fma(x_row[k], w_row[k], sum);\n"
    "  }\n"
    "  __local float sdata[WG_SIZE];\n"
    "  sdata[lid] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {\n"
    "    if (lid < s) sdata[lid] += sdata[lid + s];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  if (lid == 0) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] = sdata[0];\n"
    "  }\n"
    "}\n"
    "\n"
    "// 3. 16x16 Shared-Memory Tiled Linear Projection: Y[M,N] = X[M,K] @ W[N,K]^T\n"
    "__kernel void vira_cl_linear_tiled_f32(\n"
    "    __global const float* restrict x,\n"
    "    __global const float* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int tx = (int)get_local_id(0);\n"
    "  const int ty = (int)get_local_id(1);\n"
    "  const int col = (int)get_group_id(0) * TILE_DIM + tx;\n"
    "  const int row = (int)get_group_id(1) * TILE_DIM + ty;\n"
    "\n"
    "  __local float tile_x[TILE_DIM][TILE_DIM];\n"
    "  __local float tile_w[TILE_DIM][TILE_DIM];\n"
    "\n"
    "  float acc = 0.0f;\n"
    "  const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;\n"
    "  for (int t = 0; t < num_tiles; ++t) {\n"
    "    const int kx = t * TILE_DIM + tx;\n"
    "    const int kw = t * TILE_DIM + ty;\n"
    "    tile_x[ty][tx] = (row < M && kx < K) ? x[(size_t)row * (size_t)K + (size_t)kx] : 0.0f;\n"
    "    // Coalesced load of W[col, kw]\n"
    "    const int w_row_idx = (int)get_group_id(0) * TILE_DIM + ty;\n"
    "    const int w_col_idx = t * TILE_DIM + tx;\n"
    "    tile_w[ty][tx] = (w_row_idx < N && w_col_idx < K)\n"
    "                         ? w[(size_t)w_row_idx * (size_t)K + (size_t)w_col_idx]\n"
    "                         : 0.0f;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    #pragma unroll\n"
    "    for (int k = 0; k < TILE_DIM; ++k) {\n"
    "      acc = fma(tile_x[ty][k], tile_w[tx][k], acc);\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  if (row < M && col < N) {\n"
    "    y[(size_t)row * (size_t)N + (size_t)col] = acc;\n"
    "  }\n"
    "}\n"
    "\n"
    "// 4. 16x16 Shared-Memory Tiled MatMul: C[M,N] = A[M,K] @ B[K,N]\n"
    "__kernel void vira_cl_matmul_tiled_f32(\n"
    "    __global const float* restrict a,\n"
    "    __global const float* restrict b,\n"
    "    __global float* restrict c,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int tx = (int)get_local_id(0);\n"
    "  const int ty = (int)get_local_id(1);\n"
    "  const int col = (int)get_group_id(0) * TILE_DIM + tx;\n"
    "  const int row = (int)get_group_id(1) * TILE_DIM + ty;\n"
    "\n"
    "  __local float tile_a[TILE_DIM][TILE_DIM];\n"
    "  __local float tile_b[TILE_DIM][TILE_DIM];\n"
    "\n"
    "  float acc = 0.0f;\n"
    "  const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;\n"
    "  for (int t = 0; t < num_tiles; ++t) {\n"
    "    const int ka = t * TILE_DIM + tx;\n"
    "    const int kb = t * TILE_DIM + ty;\n"
    "    tile_a[ty][tx] = (row < M && ka < K) ? a[(size_t)row * (size_t)K + (size_t)ka] : 0.0f;\n"
    "    tile_b[ty][tx] = (kb < K && col < N) ? b[(size_t)kb * (size_t)N + (size_t)col] : 0.0f;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    #pragma unroll\n"
    "    for (int k = 0; k < TILE_DIM; ++k) {\n"
    "      acc = fma(tile_a[ty][k], tile_b[k][tx], acc);\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  if (row < M && col < N) {\n"
    "    c[(size_t)row * (size_t)N + (size_t)col] = acc;\n"
    "  }\n"
    "}\n"
    "\n"
    "// 5. Work-Group RMSNorm Kernel\n"
    "__kernel void vira_cl_rmsnorm_f32(\n"
    "    __global const float* restrict x,\n"
    "    __global const float* restrict weight,\n"
    "    __global float* restrict out,\n"
    "    const int cols, const float eps) {\n"
    "  const int lid = (int)get_local_id(0);\n"
    "  const int row = (int)get_group_id(0);\n"
    "  __global const float* x_row = x + (size_t)row * (size_t)cols;\n"
    "  __global float* out_row = out + (size_t)row * (size_t)cols;\n"
    "  float sum_sq = 0.0f;\n"
    "  for (int c = lid; c < cols; c += WG_SIZE) {\n"
    "    float v = x_row[c];\n"
    "    sum_sq = fma(v, v, sum_sq);\n"
    "  }\n"
    "  __local float sdata[WG_SIZE];\n"
    "  sdata[lid] = sum_sq;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {\n"
    "    if (lid < s) sdata[lid] += sdata[lid + s];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  const float inv_rms = rsqrt(sdata[0] / (float)cols + eps);\n"
    "  for (int c = lid; c < cols; c += WG_SIZE) {\n"
    "    out_row[c] = x_row[c] * inv_rms * weight[c];\n"
    "  }\n"
    "}\n"
    "\n"
    "// 6. Rotary Position Embedding (RoPE) Kernel\n"
    "__kernel void vira_cl_rope_f32(\n"
    "    __global const float* restrict q,\n"
    "    __global float* restrict out,\n"
    "    const int rows, const int cols, const int half_dim,\n"
    "    const float pos, const float freq_base) {\n"
    "  const int i = (int)get_global_id(0);\n"
    "  const int r = (int)get_global_id(1);\n"
    "  if (i >= half_dim || r >= rows) return;\n"
    "  const float exponent = -(2.0f * (float)i) / (float)cols;\n"
    "  const float angle = pos * pow(freq_base, exponent);\n"
    "  const float cos_a = native_cos(angle);\n"
    "  const float sin_a = native_sin(angle);\n"
    "  __global const float* q_row = q + (size_t)r * (size_t)cols;\n"
    "  __global float* o_row = out + (size_t)r * (size_t)cols;\n"
    "  const float q0 = q_row[i];\n"
    "  const float q1 = q_row[i + half_dim];\n"
    "  o_row[i]            = fma(q0, cos_a, -q1 * sin_a);\n"
    "  o_row[i + half_dim] = fma(q0, sin_a,  q1 * cos_a);\n"
    "}\n"
    "\n"
    "// 7. Vectorized SiLU Activation Kernel\n"
    "__kernel void vira_cl_silu_f32(\n"
    "    __global const float* restrict x,\n"
    "    __global float* restrict out,\n"
    "    const int total) {\n"
    "  const int idx = (int)get_global_id(0);\n"
    "  if (idx >= total) return;\n"
    "  const float v = x[idx];\n"
    "  out[idx] = v / (1.0f + native_exp(-v));\n"
    "}\n";

static inline bool _vira_gpu_load_sym(void *lib, const char *name,
                                      void *fn_out) {
#ifdef _WIN32
  void *p = (void *)(uintptr_t)GetProcAddress((HMODULE)lib, name);
#else
  void *p = dlsym(lib, name);
#endif
  if (p == NULL) {
    return false;
  }
  memcpy(fn_out, &p, sizeof(void *));
  return true;
}

// Ensures a persistent device scratch buffer has at least `needed_bytes` capacity
static inline vira_cl_mem _vira_gpu_ensure_scratch_locked(vira_cl_mem *buf_ptr,
                                                          size_t *cap_ptr,
                                                          size_t needed_bytes,
                                                          vira_cl_bitfield flags,
                                                          vira_cl_int *out_err) {
  if (*buf_ptr != NULL && *cap_ptr >= needed_bytes) {
    *out_err = VIRA_CL_SUCCESS;
    return *buf_ptr;
  }
  size_t new_cap = (needed_bytes < (4ULL * 1024ULL * 1024ULL))
                       ? (4ULL * 1024ULL * 1024ULL)
                       : ((needed_bytes + 65535ULL) & ~65535ULL);
  if (*buf_ptr != NULL) {
    _vira_gpu.clReleaseMemObject(*buf_ptr);
    __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)(*cap_ptr),
                       __ATOMIC_RELAXED);
    *buf_ptr = NULL;
    *cap_ptr = 0;
  }
  vira_cl_int err = 0;
  vira_cl_mem new_buf =
      _vira_gpu.clCreateBuffer(_vira_gpu.context, flags, new_cap, NULL, &err);
  if (err != VIRA_CL_SUCCESS || new_buf == NULL) {
    *out_err = (err != VIRA_CL_SUCCESS) ? err : -4;
    return NULL;
  }
  *buf_ptr = new_buf;
  *cap_ptr = new_cap;
  __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)new_cap,
                     __ATOMIC_RELAXED);
  *out_err = VIRA_CL_SUCCESS;
  return new_buf;
}

static inline void vira_gpu_init(void) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) == 2) {
    return;
  }
  pthread_mutex_lock(&_vira_gpu.lock);
  if (_vira_gpu.initialized == 2) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return;
  }

  snprintf(_vira_gpu.device_name, sizeof(_vira_gpu.device_name),
           "Multi-Threaded CPU SIMD (%s)", vira_simd_backend_name());
  snprintf(_vira_gpu.backend_name, sizeof(_vira_gpu.backend_name), "CPU-SIMD");
  _vira_gpu.available = false;
  _vira_gpu.enabled = true;

  void *lib = NULL;
#ifdef _WIN32
  lib = (void *)LoadLibraryA("OpenCL.dll");
#elif defined(__APPLE__)
  lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_LAZY);
#else
  lib = dlopen("libOpenCL.so.1", RTLD_LAZY);
  if (!lib) {
    lib = dlopen("libOpenCL.so", RTLD_LAZY);
  }
#endif

  if (lib != NULL) {
    bool ok = true;
    ok = ok && _vira_gpu_load_sym(lib, "clGetPlatformIDs",
                                  &_vira_gpu.clGetPlatformIDs);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clGetDeviceIDs", &_vira_gpu.clGetDeviceIDs);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clGetDeviceInfo", &_vira_gpu.clGetDeviceInfo);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clCreateContext", &_vira_gpu.clCreateContext);
    ok = ok && _vira_gpu_load_sym(lib, "clReleaseContext",
                                  &_vira_gpu.clReleaseContext);
    ok = ok && _vira_gpu_load_sym(lib, "clCreateCommandQueue",
                                  &_vira_gpu.clCreateCommandQueue);
    ok = ok && _vira_gpu_load_sym(lib, "clReleaseCommandQueue",
                                  &_vira_gpu.clReleaseCommandQueue);
    ok = ok && _vira_gpu_load_sym(lib, "clCreateProgramWithSource",
                                  &_vira_gpu.clCreateProgramWithSource);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clBuildProgram", &_vira_gpu.clBuildProgram);
    ok = ok && _vira_gpu_load_sym(lib, "clReleaseProgram",
                                  &_vira_gpu.clReleaseProgram);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clCreateKernel", &_vira_gpu.clCreateKernel);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clReleaseKernel", &_vira_gpu.clReleaseKernel);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clCreateBuffer", &_vira_gpu.clCreateBuffer);
    ok = ok && _vira_gpu_load_sym(lib, "clReleaseMemObject",
                                  &_vira_gpu.clReleaseMemObject);
    ok = ok && _vira_gpu_load_sym(lib, "clEnqueueWriteBuffer",
                                  &_vira_gpu.clEnqueueWriteBuffer);
    ok = ok && _vira_gpu_load_sym(lib, "clEnqueueReadBuffer",
                                  &_vira_gpu.clEnqueueReadBuffer);
    ok = ok &&
         _vira_gpu_load_sym(lib, "clSetKernelArg", &_vira_gpu.clSetKernelArg);
    ok = ok && _vira_gpu_load_sym(lib, "clEnqueueNDRangeKernel",
                                  &_vira_gpu.clEnqueueNDRangeKernel);
    ok = ok && _vira_gpu_load_sym(lib, "clFinish", &_vira_gpu.clFinish);

    if (ok) {
      vira_cl_platform_id platforms[16];
      vira_cl_uint num_platforms = 0;
      if (_vira_gpu.clGetPlatformIDs(16, platforms, &num_platforms) ==
              VIRA_CL_SUCCESS &&
          num_platforms > 0) {
        // Select the GPU device with the highest score (prefers discrete high-VRAM GPUs)
        vira_cl_device_id best_dev = NULL;
        int64_t best_score = -1;

        for (vira_cl_uint p = 0; p < num_platforms; ++p) {
          vira_cl_device_id devs[8];
          vira_cl_uint num_devs = 0;
          if (_vira_gpu.clGetDeviceIDs(platforms[p], VIRA_CL_DEVICE_TYPE_GPU, 8,
                                       devs, &num_devs) != VIRA_CL_SUCCESS ||
              num_devs == 0) {
            continue;
          }
          for (vira_cl_uint d = 0; d < num_devs; ++d) {
            vira_cl_ulong gmem = 0;
            vira_cl_uint cu = 0;
            vira_cl_uint um = 0;
            _vira_gpu.clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_GLOBAL_MEM_SIZE,
                                      sizeof(gmem), &gmem, NULL);
            _vira_gpu.clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_MAX_COMPUTE_UNITS,
                                      sizeof(cu), &cu, NULL);
            _vira_gpu.clGetDeviceInfo(
                devs[d], VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(um), &um,
                NULL);
            // Discrete GPUs (unified_mem == 0) get a 4x priority multiplier over iGPUs
            int64_t score = (int64_t)(gmem >> 20) * (um == 0 ? 4LL : 1LL) +
                            (int64_t)cu * 64LL;
            if (score > best_score) {
              best_score = score;
              best_dev = devs[d];
            }
          }
        }

        if (best_dev != NULL) {
          vira_cl_int err = 0;
          vira_cl_context ctx =
              _vira_gpu.clCreateContext(NULL, 1, &best_dev, NULL, NULL, &err);
          if (err == VIRA_CL_SUCCESS && ctx != NULL) {
            vira_cl_command_queue q =
                _vira_gpu.clCreateCommandQueue(ctx, best_dev, 0, &err);
            if (err == VIRA_CL_SUCCESS && q != NULL) {
              size_t src_len = strlen(_vira_opencl_kernel_src);
              vira_cl_program prog = _vira_gpu.clCreateProgramWithSource(
                  ctx, 1, &_vira_opencl_kernel_src, &src_len, &err);
              if (err == VIRA_CL_SUCCESS && prog != NULL &&
                  _vira_gpu.clBuildProgram(
                      prog, 1, &best_dev,
                      "-cl-fast-relaxed-math -cl-mad-enable", NULL,
                      NULL) == VIRA_CL_SUCCESS) {
                vira_cl_int e1 = 0, e2 = 0, e3 = 0, e4 = 0, e5 = 0, e6 = 0,
                            e7 = 0;
                vira_cl_kernel k_gv4 = _vira_gpu.clCreateKernel(
                    prog, "vira_cl_gemv_vec4", &e1);
                vira_cl_kernel k_gvs = _vira_gpu.clCreateKernel(
                    prog, "vira_cl_gemv_scalar", &e2);
                vira_cl_kernel k_lin = _vira_gpu.clCreateKernel(
                    prog, "vira_cl_linear_tiled_f32", &e3);
                vira_cl_kernel k_mm = _vira_gpu.clCreateKernel(
                    prog, "vira_cl_matmul_tiled_f32", &e4);
                vira_cl_kernel k_rms = _vira_gpu.clCreateKernel(
                    prog, "vira_cl_rmsnorm_f32", &e5);
                vira_cl_kernel k_rope = _vira_gpu.clCreateKernel(
                    prog, "vira_cl_rope_f32", &e6);
                vira_cl_kernel k_silu = _vira_gpu.clCreateKernel(
                    prog, "vira_cl_silu_f32", &e7);

                if (e1 == VIRA_CL_SUCCESS && e2 == VIRA_CL_SUCCESS &&
                    e3 == VIRA_CL_SUCCESS && e4 == VIRA_CL_SUCCESS &&
                    k_gv4 && k_gvs && k_lin && k_mm) {
                  _vira_gpu.lib_handle = lib;
                  _vira_gpu.device = best_dev;
                  _vira_gpu.context = ctx;
                  _vira_gpu.queue = q;
                  _vira_gpu.program = prog;
                  _vira_gpu.kernel_gemv_vec4 = k_gv4;
                  _vira_gpu.kernel_gemv_scalar = k_gvs;
                  _vira_gpu.kernel_linear_tiled = k_lin;
                  _vira_gpu.kernel_matmul_tiled = k_mm;
                  _vira_gpu.kernel_rmsnorm = (e5 == VIRA_CL_SUCCESS) ? k_rms : NULL;
                  _vira_gpu.kernel_rope = (e6 == VIRA_CL_SUCCESS) ? k_rope : NULL;
                  _vira_gpu.kernel_silu = (e7 == VIRA_CL_SUCCESS) ? k_silu : NULL;
                  _vira_gpu.available = true;
                  snprintf(_vira_gpu.backend_name,
                           sizeof(_vira_gpu.backend_name), "OpenCL");

                  char dname[256] = {0};
                  if (_vira_gpu.clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_NAME,
                                                sizeof(dname) - 1, dname,
                                                NULL) == VIRA_CL_SUCCESS) {
                    snprintf(_vira_gpu.device_name,
                             sizeof(_vira_gpu.device_name), "%s", dname);
                  }
                  vira_cl_ulong gmem = 0;
                  if (_vira_gpu.clGetDeviceInfo(
                          best_dev, VIRA_CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(gmem), &gmem, NULL) == VIRA_CL_SUCCESS) {
                    _vira_gpu.vram_bytes = (int64_t)gmem;
                    // Reserve 85% of total VRAM for persistent weight caching
                    _vira_gpu.vram_budget_bytes =
                        (int64_t)((gmem * 85ULL) / 100ULL);
                  }
                  vira_cl_uint cu = 0;
                  if (_vira_gpu.clGetDeviceInfo(
                          best_dev, VIRA_CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(cu), &cu, NULL) == VIRA_CL_SUCCESS) {
                    _vira_gpu.compute_units = (int32_t)cu;
                  }
                  size_t mwg = 256;
                  if (_vira_gpu.clGetDeviceInfo(
                          best_dev, VIRA_CL_DEVICE_MAX_WORK_GROUP_SIZE,
                          sizeof(mwg), &mwg, NULL) == VIRA_CL_SUCCESS &&
                      mwg > 0) {
                    _vira_gpu.max_wg_size = mwg;
                  }
                  vira_cl_uint um = 0;
                  if (_vira_gpu.clGetDeviceInfo(
                          best_dev, VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY,
                          sizeof(um), &um, NULL) == VIRA_CL_SUCCESS) {
                    _vira_gpu.unified_mem = (um != 0);
                  }
                } else {
                  if (k_gv4) _vira_gpu.clReleaseKernel(k_gv4);
                  if (k_gvs) _vira_gpu.clReleaseKernel(k_gvs);
                  if (k_lin) _vira_gpu.clReleaseKernel(k_lin);
                  if (k_mm) _vira_gpu.clReleaseKernel(k_mm);
                  if (k_rms) _vira_gpu.clReleaseKernel(k_rms);
                  if (k_rope) _vira_gpu.clReleaseKernel(k_rope);
                  if (k_silu) _vira_gpu.clReleaseKernel(k_silu);
                  _vira_gpu.clReleaseProgram(prog);
                  _vira_gpu.clReleaseCommandQueue(q);
                  _vira_gpu.clReleaseContext(ctx);
                }
              } else {
                if (prog)
                  _vira_gpu.clReleaseProgram(prog);
                _vira_gpu.clReleaseCommandQueue(q);
                _vira_gpu.clReleaseContext(ctx);
              }
            } else {
              _vira_gpu.clReleaseContext(ctx);
            }
          }
        }
      }
    }
  }

  __atomic_store_n(&_vira_gpu.initialized, 2, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline void _vira_gpu_evict_ptr_locked(const vira_f32 *host_ptr) {
  if (!_vira_gpu.available || host_ptr == NULL) {
    return;
  }
  uint64_t h = ((uint64_t)(uintptr_t)host_ptr >> 6) * 11400714819323198485ULL;
  int32_t start_slot = (int32_t)(h & (VIRA_GPU_WEIGHT_CACHE_CAP - 1));

  for (int32_t probe = 0; probe < VIRA_GPU_WEIGHT_PROBE_LEN; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    if (_vira_gpu.weight_cache[idx].host_ptr == host_ptr) {
      if (_vira_gpu.weight_cache[idx].dev_buf != NULL) {
        _vira_gpu.clReleaseMemObject(_vira_gpu.weight_cache[idx].dev_buf);
        __atomic_sub_fetch(&_vira_gpu.allocated_bytes,
                           (int64_t)_vira_gpu.weight_cache[idx].bytes,
                           __ATOMIC_RELAXED);
        _vira_gpu.vram_saturated = false;
      }
      _vira_gpu.weight_cache[idx].host_ptr = NULL;
      _vira_gpu.weight_cache[idx].bytes = 0;
      _vira_gpu.weight_cache[idx].dev_buf = NULL;
      _vira_gpu.weight_cache[idx].last_used = 0;
      _vira_gpu.weight_cache[idx].uncacheable = false;
    }
  }
}

static inline void _vira_gpu_evict_ptr(const vira_f32 *host_ptr) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) != 2 ||
      !_vira_gpu.available || host_ptr == NULL) {
    return;
  }
  pthread_mutex_lock(&_vira_gpu.lock);
  _vira_gpu_evict_ptr_locked(host_ptr);
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline void _vira_gpu_evict_all_locked(void) {
  if (!_vira_gpu.available) {
    return;
  }
  for (int32_t i = 0; i < VIRA_GPU_WEIGHT_CACHE_CAP; ++i) {
    if (_vira_gpu.weight_cache[i].dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(_vira_gpu.weight_cache[i].dev_buf);
    }
    _vira_gpu.weight_cache[i].host_ptr = NULL;
    _vira_gpu.weight_cache[i].bytes = 0;
    _vira_gpu.weight_cache[i].dev_buf = NULL;
    _vira_gpu.weight_cache[i].last_used = 0;
    _vira_gpu.weight_cache[i].uncacheable = false;
  }
  if (_vira_gpu.scratch_a != NULL) {
    _vira_gpu.clReleaseMemObject(_vira_gpu.scratch_a);
    _vira_gpu.scratch_a = NULL;
    _vira_gpu.scratch_a_cap = 0;
  }
  if (_vira_gpu.scratch_b != NULL) {
    _vira_gpu.clReleaseMemObject(_vira_gpu.scratch_b);
    _vira_gpu.scratch_b = NULL;
    _vira_gpu.scratch_b_cap = 0;
  }
  if (_vira_gpu.scratch_c != NULL) {
    _vira_gpu.clReleaseMemObject(_vira_gpu.scratch_c);
    _vira_gpu.scratch_c = NULL;
    _vira_gpu.scratch_c_cap = 0;
  }
  _vira_gpu.vram_saturated = false;
  __atomic_store_n(&_vira_gpu.allocated_bytes, 0, __ATOMIC_RELAXED);
}

// Non-destructive persistent weight cache lookup/upload:
// - If already in VRAM: returns `dev_buf` in O(1) without touching PCIe.
// - If VRAM budget is full: marks `uncacheable = true` so subsequent tokens
//   immediately dispatch this overflow layer to CPU without thrashing VRAM!
static inline vira_cl_mem _vira_gpu_get_or_upload_weight_locked(
    const vira_f32 *host_ptr, size_t bytes, vira_cl_int *out_err) {
  uint64_t h = ((uint64_t)(uintptr_t)host_ptr >> 6) * 11400714819323198485ULL;
  int32_t start_slot = (int32_t)(h & (VIRA_GPU_WEIGHT_CACHE_CAP - 1));
  uint64_t now = ++_vira_gpu.access_clock;

  int32_t empty_slot = -1;
  for (int32_t probe = 0; probe < VIRA_GPU_WEIGHT_PROBE_LEN; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    if (_vira_gpu.weight_cache[idx].host_ptr == host_ptr &&
        _vira_gpu.weight_cache[idx].bytes == bytes) {
      if (_vira_gpu.weight_cache[idx].uncacheable) {
        *out_err = -4; // Pinned to CPU (VRAM full)
        return NULL;
      }
      if (_vira_gpu.weight_cache[idx].dev_buf != NULL) {
        _vira_gpu.weight_cache[idx].last_used = now;
        *out_err = VIRA_CL_SUCCESS;
        return _vira_gpu.weight_cache[idx].dev_buf;
      }
    }
    if (empty_slot < 0 && _vira_gpu.weight_cache[idx].host_ptr == NULL) {
      empty_slot = idx;
    }
  }

  // If VRAM is already saturated or this weight exceeds the 85% VRAM budget,
  // keep existing weights pinned in VRAM and run this overflow layer on CPU!
  int64_t cur_alloc = __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
  if (_vira_gpu.vram_saturated ||
      (_vira_gpu.vram_budget_bytes > 0 &&
       cur_alloc + (int64_t)bytes > _vira_gpu.vram_budget_bytes)) {
    if (empty_slot >= 0) {
      _vira_gpu.weight_cache[empty_slot].host_ptr = host_ptr;
      _vira_gpu.weight_cache[empty_slot].bytes = bytes;
      _vira_gpu.weight_cache[empty_slot].dev_buf = NULL;
      _vira_gpu.weight_cache[empty_slot].last_used = now;
      _vira_gpu.weight_cache[empty_slot].uncacheable = true;
    }
    _vira_gpu.vram_saturated = true;
    *out_err = -4;
    return NULL;
  }

  if (empty_slot < 0) {
    // Table window full: do not evict active weights; run overflow layer on CPU
    *out_err = -4;
    return NULL;
  }

  vira_cl_bitfield flags =
      VIRA_CL_MEM_READ_ONLY |
      (_vira_gpu.unified_mem ? VIRA_CL_MEM_USE_HOST_PTR
                             : VIRA_CL_MEM_COPY_HOST_PTR);
  vira_cl_int err = 0;
  vira_cl_mem buf = _vira_gpu.clCreateBuffer(_vira_gpu.context, flags, bytes,
                                             (void *)host_ptr, &err);
  if (err != VIRA_CL_SUCCESS || buf == NULL) {
    // Non-destructive OOM: mark this tensor uncacheable and preserve all prior layers!
    _vira_gpu.weight_cache[empty_slot].host_ptr = host_ptr;
    _vira_gpu.weight_cache[empty_slot].bytes = bytes;
    _vira_gpu.weight_cache[empty_slot].dev_buf = NULL;
    _vira_gpu.weight_cache[empty_slot].last_used = now;
    _vira_gpu.weight_cache[empty_slot].uncacheable = true;
    _vira_gpu.vram_saturated = true;
    *out_err = (err != VIRA_CL_SUCCESS) ? err : -4;
    return NULL;
  }

  _vira_gpu.weight_cache[empty_slot].host_ptr = host_ptr;
  _vira_gpu.weight_cache[empty_slot].bytes = bytes;
  _vira_gpu.weight_cache[empty_slot].dev_buf = buf;
  _vira_gpu.weight_cache[empty_slot].last_used = now;
  _vira_gpu.weight_cache[empty_slot].uncacheable = false;
  __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)bytes,
                     __ATOMIC_RELAXED);
  *out_err = VIRA_CL_SUCCESS;
  return buf;
}

static inline void vira_gpu_shutdown(void) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) != 2) {
    return;
  }
  pthread_mutex_lock(&_vira_gpu.lock);
  if (_vira_gpu.available) {
    _vira_gpu_evict_all_locked();
    if (_vira_gpu.queue)
      _vira_gpu.clFinish(_vira_gpu.queue);
    if (_vira_gpu.kernel_gemv_vec4)
      _vira_gpu.clReleaseKernel(_vira_gpu.kernel_gemv_vec4);
    if (_vira_gpu.kernel_gemv_scalar)
      _vira_gpu.clReleaseKernel(_vira_gpu.kernel_gemv_scalar);
    if (_vira_gpu.kernel_linear_tiled)
      _vira_gpu.clReleaseKernel(_vira_gpu.kernel_linear_tiled);
    if (_vira_gpu.kernel_matmul_tiled)
      _vira_gpu.clReleaseKernel(_vira_gpu.kernel_matmul_tiled);
    if (_vira_gpu.kernel_rmsnorm)
      _vira_gpu.clReleaseKernel(_vira_gpu.kernel_rmsnorm);
    if (_vira_gpu.kernel_rope)
      _vira_gpu.clReleaseKernel(_vira_gpu.kernel_rope);
    if (_vira_gpu.kernel_silu)
      _vira_gpu.clReleaseKernel(_vira_gpu.kernel_silu);
    if (_vira_gpu.program)
      _vira_gpu.clReleaseProgram(_vira_gpu.program);
    if (_vira_gpu.queue)
      _vira_gpu.clReleaseCommandQueue(_vira_gpu.queue);
    if (_vira_gpu.context)
      _vira_gpu.clReleaseContext(_vira_gpu.context);
    _vira_gpu.available = false;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline bool vira_gpu_is_available(void) {
  vira_gpu_init();
  return _vira_gpu.available;
}

static inline vira_string vira_gpu_device_name(void) {
  vira_gpu_init();
  return vira_str_lit(_vira_gpu.device_name,
                      (int32_t)strlen(_vira_gpu.device_name));
}

static inline vira_string vira_gpu_backend_name(void) {
  vira_gpu_init();
  return vira_str_lit(_vira_gpu.backend_name,
                      (int32_t)strlen(_vira_gpu.backend_name));
}

static inline int64_t vira_gpu_vram_bytes(void) {
  vira_gpu_init();
  return _vira_gpu.vram_bytes;
}

static inline int64_t vira_gpu_allocated_bytes(void) {
  vira_gpu_init();
  return __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
}

static inline int64_t vira_gpu_oom_fallbacks(void) {
  vira_gpu_init();
  return __atomic_load_n(&_vira_gpu.oom_fallbacks, __ATOMIC_RELAXED);
}

static inline void vira_gpu_set_enabled(bool enabled) {
  vira_gpu_init();
  _vira_gpu.enabled = enabled;
}

static inline bool vira_gpu_is_enabled(void) {
  vira_gpu_init();
  return _vira_gpu.enabled;
}

static inline void vira_gpu_simulate_oom(bool enable) {
  vira_gpu_init();
  _vira_gpu.simulate_oom = enable;
}

static inline void vira_gpu_sync(void) {
  vira_gpu_init();
  if (_vira_gpu.available && _vira_gpu.queue != NULL) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu.clFinish(_vira_gpu.queue);
    pthread_mutex_unlock(&_vira_gpu.lock);
  }
}

// ============================================================================
// GPU LINEAR PROJECTION: `y = x @ w^T` (WITH COALESCED GEMV & TILED GEMM)
// ============================================================================
static inline vira_tensor_f32 vira_gpu_linear(vira_arena *opt_arena,
                                              vira_tensor_f32 x,
                                              vira_tensor_f32 w) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&x);
  _vira_tensor_norm_shape(&w);
  int32_t M = x.rows;
  int32_t K = x.cols;
  int32_t N = w.rows;
  if (K != w.cols) {
    fprintf(stderr,
            "vira gpu_linear error: inner dimension mismatch x=(%dx%d) vs w=(%dx%d)\n",
            M, K, N, w.cols);
    exit(1);
  }

  if (_vira_gpu.simulate_oom) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }

  int64_t total_ops = (int64_t)M * (int64_t)N * (int64_t)K;
  if (!_vira_gpu.available || !_vira_gpu.enabled || total_ops < 65536LL) {
    return vira_tensor_linear(opt_arena, x, w);
  }

  size_t x_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  size_t w_bytes = (size_t)N * (size_t)K * sizeof(vira_f32);
  size_t y_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;

  // 1. Lookup or upload persistent weight matrix `w`
  vira_cl_mem w_buf =
      _vira_gpu_get_or_upload_weight_locked(w.data, w_bytes, &err);
  if (err != VIRA_CL_SUCCESS || w_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }

  // 2. Reuse persistent scratch buffers for input `x` and output `y`
  vira_cl_mem x_buf = _vira_gpu_ensure_scratch_locked(
      &_vira_gpu.scratch_a, &_vira_gpu.scratch_a_cap, x_bytes,
      VIRA_CL_MEM_READ_WRITE, &err);
  if (err != VIRA_CL_SUCCESS || x_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }

  vira_cl_mem y_buf = _vira_gpu_ensure_scratch_locked(
      &_vira_gpu.scratch_c, &_vira_gpu.scratch_c_cap, y_bytes,
      VIRA_CL_MEM_READ_WRITE, &err);
  if (err != VIRA_CL_SUCCESS || y_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }

  // Non-blocking write of activation `x` into persistent scratch buffer
  err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, x_buf, 0, 0, x_bytes,
                                       x.data, 0, NULL, NULL);

  if (err == VIRA_CL_SUCCESS) {
    if (M <= 4 && _vira_gpu.max_wg_size >= 256) {
      // Decode / small batch: dispatch coalesced 256-thread work-group per output row
      if ((K & 3) == 0) {
        int32_t K4 = K >> 2;
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_vec4, 0,
                                        sizeof(vira_cl_mem), &x_buf);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_vec4, 1,
                                        sizeof(vira_cl_mem), &w_buf);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_vec4, 2,
                                        sizeof(vira_cl_mem), &y_buf);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_vec4, 3,
                                        sizeof(int32_t), &M);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_vec4, 4,
                                        sizeof(int32_t), &K4);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_vec4, 5,
                                        sizeof(int32_t), &N);
        size_t l_work[2] = {256, 1};
        size_t g_work[2] = {(size_t)N * 256, (size_t)M};
        if (err == VIRA_CL_SUCCESS) {
          err = _vira_gpu.clEnqueueNDRangeKernel(
              _vira_gpu.queue, _vira_gpu.kernel_gemv_vec4, 2, NULL, g_work,
              l_work, 0, NULL, NULL);
        }
      } else {
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_scalar, 0,
                                        sizeof(vira_cl_mem), &x_buf);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_scalar, 1,
                                        sizeof(vira_cl_mem), &w_buf);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_scalar, 2,
                                        sizeof(vira_cl_mem), &y_buf);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_scalar, 3,
                                        sizeof(int32_t), &M);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_scalar, 4,
                                        sizeof(int32_t), &K);
        err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_gemv_scalar, 5,
                                        sizeof(int32_t), &N);
        size_t l_work[2] = {256, 1};
        size_t g_work[2] = {(size_t)N * 256, (size_t)M};
        if (err == VIRA_CL_SUCCESS) {
          err = _vira_gpu.clEnqueueNDRangeKernel(
              _vira_gpu.queue, _vira_gpu.kernel_gemv_scalar, 2, NULL, g_work,
              l_work, 0, NULL, NULL);
        }
      }
    } else {
      // Prefill / large batch (M > 4): dispatch 16x16 shared-memory tiled GEMM
      err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear_tiled, 0,
                                      sizeof(vira_cl_mem), &x_buf);
      err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear_tiled, 1,
                                      sizeof(vira_cl_mem), &w_buf);
      err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear_tiled, 2,
                                      sizeof(vira_cl_mem), &y_buf);
      err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear_tiled, 3,
                                      sizeof(int32_t), &M);
      err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear_tiled, 4,
                                      sizeof(int32_t), &K);
      err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_linear_tiled, 5,
                                      sizeof(int32_t), &N);
      size_t l_work[2] = {16, 16};
      size_t g_work[2] = {((size_t)N + 15) & ~15ULL, ((size_t)M + 15) & ~15ULL};
      if (err == VIRA_CL_SUCCESS) {
        err = _vira_gpu.clEnqueueNDRangeKernel(
            _vira_gpu.queue, _vira_gpu.kernel_linear_tiled, 2, NULL, g_work,
            l_work, 0, NULL, NULL);
      }
    }
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_buf, 1, 0, y_bytes,
                                        out.data, 0, NULL, NULL);
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_linear(opt_arena, x, w);
  }
  return out;
}

// ============================================================================
// GPU MATRIX MULTIPLICATION: `c = a @ b` (16x16 SHARED-MEMORY TILED GEMM)
// ============================================================================
static inline vira_tensor_f32 vira_gpu_matmul(vira_arena *opt_arena,
                                              vira_tensor_f32 a,
                                              vira_tensor_f32 b) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&a);
  _vira_tensor_norm_shape(&b);

  if (_vira_gpu.simulate_oom) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }

  int32_t M = a.rows;
  int32_t K = a.cols;
  int32_t N = b.cols;
  int64_t total_ops = (int64_t)M * (int64_t)K * (int64_t)N;
  if (!_vira_gpu.available || !_vira_gpu.enabled || a.n_dims > 2 ||
      b.n_dims > 2 || K != b.rows || total_ops < 65536LL) {
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }

  size_t a_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  size_t b_bytes = (size_t)K * (size_t)N * sizeof(vira_f32);
  size_t c_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;

  vira_cl_mem b_buf =
      _vira_gpu_get_or_upload_weight_locked(b.data, b_bytes, &err);
  if (err != VIRA_CL_SUCCESS || b_buf == NULL) {
    // If `b` isn't a cached weight (or VRAM is full), use persistent scratch_b
    b_buf = _vira_gpu_ensure_scratch_locked(
        &_vira_gpu.scratch_b, &_vira_gpu.scratch_b_cap, b_bytes,
        VIRA_CL_MEM_READ_WRITE, &err);
    if (err == VIRA_CL_SUCCESS && b_buf != NULL) {
      err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, b_buf, 0, 0,
                                           b_bytes, b.data, 0, NULL, NULL);
    }
  }
  if (err != VIRA_CL_SUCCESS || b_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }

  vira_cl_mem a_buf = _vira_gpu_ensure_scratch_locked(
      &_vira_gpu.scratch_a, &_vira_gpu.scratch_a_cap, a_bytes,
      VIRA_CL_MEM_READ_WRITE, &err);
  vira_cl_mem c_buf = (err == VIRA_CL_SUCCESS)
                          ? _vira_gpu_ensure_scratch_locked(
                                &_vira_gpu.scratch_c, &_vira_gpu.scratch_c_cap,
                                c_bytes, VIRA_CL_MEM_READ_WRITE, &err)
                          : NULL;
  if (err != VIRA_CL_SUCCESS || a_buf == NULL || c_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }

  err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, a_buf, 0, 0, a_bytes,
                                       a.data, 0, NULL, NULL);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul_tiled, 0,
                                  sizeof(vira_cl_mem), &a_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul_tiled, 1,
                                  sizeof(vira_cl_mem), &b_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul_tiled, 2,
                                  sizeof(vira_cl_mem), &c_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul_tiled, 3,
                                  sizeof(int32_t), &M);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul_tiled, 4,
                                  sizeof(int32_t), &K);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_matmul_tiled, 5,
                                  sizeof(int32_t), &N);

  size_t l_work[2] = {16, 16};
  size_t g_work[2] = {((size_t)N + 15) & ~15ULL, ((size_t)M + 15) & ~15ULL};
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(
        _vira_gpu.queue, _vira_gpu.kernel_matmul_tiled, 2, NULL, g_work, l_work,
        0, NULL, NULL);
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, c_buf, 1, 0, c_bytes,
                                        out.data, 0, NULL, NULL);
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_matmul_f32(opt_arena, a, b);
  }
  return out;
}

// ============================================================================
// GPU TRANSFORMER OPERATORS (RMSNORM, ROPE, SILU, SOFTMAX)
// ============================================================================
static inline vira_tensor_f32 vira_gpu_rmsnorm(vira_tensor_f32 x,
                                               vira_tensor_f32 weight,
                                               float eps) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&x);
  int32_t total = x.rows * x.cols;
  if (!_vira_gpu.available || !_vira_gpu.enabled ||
      _vira_gpu.kernel_rmsnorm == NULL || total < 16384 ||
      _vira_gpu.max_wg_size < 256) {
    return vira_tensor_rmsnorm(x, weight, eps);
  }

  size_t x_bytes = (size_t)total * sizeof(vira_f32);
  size_t w_bytes = (size_t)x.cols * sizeof(vira_f32);
  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  vira_cl_mem w_buf =
      _vira_gpu_get_or_upload_weight_locked(weight.data, w_bytes, &err);
  vira_cl_mem x_buf =
      (err == VIRA_CL_SUCCESS)
          ? _vira_gpu_ensure_scratch_locked(&_vira_gpu.scratch_a,
                                            &_vira_gpu.scratch_a_cap, x_bytes,
                                            VIRA_CL_MEM_READ_WRITE, &err)
          : NULL;
  vira_cl_mem o_buf =
      (err == VIRA_CL_SUCCESS)
          ? _vira_gpu_ensure_scratch_locked(&_vira_gpu.scratch_c,
                                            &_vira_gpu.scratch_c_cap, x_bytes,
                                            VIRA_CL_MEM_READ_WRITE, &err)
          : NULL;
  if (err != VIRA_CL_SUCCESS || !w_buf || !x_buf || !o_buf) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return vira_tensor_rmsnorm(x, weight, eps);
  }

  err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, x_buf, 0, 0, x_bytes,
                                       x.data, 0, NULL, NULL);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 0,
                                  sizeof(vira_cl_mem), &x_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 1,
                                  sizeof(vira_cl_mem), &w_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 2,
                                  sizeof(vira_cl_mem), &o_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 3, sizeof(int32_t),
                                  &x.cols);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 4, sizeof(float),
                                  &eps);
  size_t l_work = 256;
  size_t g_work = (size_t)x.rows * 256;
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_rmsnorm, 1, NULL,
                                           &g_work, &l_work, 0, NULL, NULL);
  }
  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, o_buf, 1, 0, x_bytes,
                                        out.data, 0, NULL, NULL);
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    return vira_tensor_rmsnorm(x, weight, eps);
  }
  return out;
}

static inline vira_tensor_f32 vira_gpu_rope(vira_tensor_f32 q, int64_t pos,
                                            float freq_base) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&q);
  int32_t total = q.rows * q.cols;
  if (!_vira_gpu.available || !_vira_gpu.enabled ||
      _vira_gpu.kernel_rope == NULL || total < 16384) {
    return vira_tensor_rope(q, pos, freq_base);
  }

  size_t q_bytes = (size_t)total * sizeof(vira_f32);
  int32_t half_dim = q.cols / 2;
  float fpos = (float)pos;

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  vira_cl_mem q_buf = _vira_gpu_ensure_scratch_locked(
      &_vira_gpu.scratch_a, &_vira_gpu.scratch_a_cap, q_bytes,
      VIRA_CL_MEM_READ_WRITE, &err);
  vira_cl_mem o_buf =
      (err == VIRA_CL_SUCCESS)
          ? _vira_gpu_ensure_scratch_locked(&_vira_gpu.scratch_c,
                                            &_vira_gpu.scratch_c_cap, q_bytes,
                                            VIRA_CL_MEM_READ_WRITE, &err)
          : NULL;
  if (err != VIRA_CL_SUCCESS || !q_buf || !o_buf) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return vira_tensor_rope(q, pos, freq_base);
  }

  err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, q_buf, 0, 0, q_bytes,
                                       q.data, 0, NULL, NULL);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 0,
                                  sizeof(vira_cl_mem), &q_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 1,
                                  sizeof(vira_cl_mem), &o_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 2, sizeof(int32_t),
                                  &q.rows);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 3, sizeof(int32_t),
                                  &q.cols);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 4, sizeof(int32_t),
                                  &half_dim);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 5, sizeof(float),
                                  &fpos);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 6, sizeof(float),
                                  &freq_base);
  size_t g_work[2] = {(size_t)half_dim, (size_t)q.rows};
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_rope, 2, NULL,
                                           g_work, NULL, 0, NULL, NULL);
  }
  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(q.n_dims, q.dims);
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, o_buf, 1, 0, q_bytes,
                                        out.data, 0, NULL, NULL);
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    return vira_tensor_rope(q, pos, freq_base);
  }
  return out;
}

static inline vira_tensor_f32 vira_gpu_softmax(vira_tensor_f32 logits) {
  return vira_tensor_softmax(logits);
}

static inline vira_tensor_f32 vira_gpu_silu(vira_tensor_f32 x) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&x);
  int32_t total = x.rows * x.cols;
  if (!_vira_gpu.available || !_vira_gpu.enabled ||
      _vira_gpu.kernel_silu == NULL || total < 32768) {
    return vira_tensor_silu(x);
  }

  size_t x_bytes = (size_t)total * sizeof(vira_f32);
  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  vira_cl_mem x_buf = _vira_gpu_ensure_scratch_locked(
      &_vira_gpu.scratch_a, &_vira_gpu.scratch_a_cap, x_bytes,
      VIRA_CL_MEM_READ_WRITE, &err);
  vira_cl_mem o_buf =
      (err == VIRA_CL_SUCCESS)
          ? _vira_gpu_ensure_scratch_locked(&_vira_gpu.scratch_c,
                                            &_vira_gpu.scratch_c_cap, x_bytes,
                                            VIRA_CL_MEM_READ_WRITE, &err)
          : NULL;
  if (err != VIRA_CL_SUCCESS || !x_buf || !o_buf) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return vira_tensor_silu(x);
  }

  err = _vira_gpu.clEnqueueWriteBuffer(_vira_gpu.queue, x_buf, 0, 0, x_bytes,
                                       x.data, 0, NULL, NULL);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_silu, 0,
                                  sizeof(vira_cl_mem), &x_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_silu, 1,
                                  sizeof(vira_cl_mem), &o_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_silu, 2, sizeof(int32_t),
                                  &total);
  size_t g_work = ((size_t)total + 255) & ~255ULL;
  size_t l_work = 256;
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_silu, 1, NULL,
                                           &g_work, &l_work, 0, NULL, NULL);
  }
  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, o_buf, 1, 0, x_bytes,
                                        out.data, 0, NULL, NULL);
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    return vira_tensor_silu(x);
  }
  return out;
}

#endif // VIRA_GPU_H