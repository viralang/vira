#ifndef VIRA_GPU_H
#define VIRA_GPU_H

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "vira_rt.h"

// ============================================================================
// VIRA v1.2 ULTRA-FAST PRODUCTION GPU RUNTIME
// ============================================================================
// Features:
// 1. Native GGUF Quantization in VRAM:
//    - Q4_0 (18B / 32 elems), Q4_1 (20B / 32 elems), Q5_0 (22B / 32 elems),
//      Q5_1 (24B / 32 elems), Q8_0 (34B / 32 elems)
//    - Q4_K (144B / 256 elems), Q5_K (176B / 256 elems), Q6_K (210B / 256 elems)
//    - Cuts VRAM consumption by up to 7.1x, enabling 3B-7B LLM inference on 4GB VRAM!
// 2. Hardware-Adaptive Warp GEMV:
//    - AMD Wavefront-64 (64 threads/warp, 4 rows/WG) vs NVIDIA/Intel/Apple Warp-32 (32 threads/warp, 8 rows/WG)
//    - Supports embedded/mobile GPUs with CL_DEVICE_MAX_WORK_GROUP_SIZE down to 64.
// 3. 8-Slot VRAM Activation Residency Ring (act_ring[8]):
//    - Eliminates redundant PCIe Host->Device re-uploads across chained Transformer blocks
//      (rmsnorm -> linear -> rope -> linear -> softmax -> matmul -> silu).
//    - O(1) epoch invalidation on arena reset/destroy and tensor mutation.
// 4. True LRU Weight Cache Eviction:
//    - Evicts oldest unused model weights under VRAM budget pressure.
// 5. Dynamic Activation Projection:
//    - Dedicated FP32 tiled linear kernel for arena-allocated activation tensors (e.g. Q @ K^T)
//      without weight-cache pollution.
// 6. Unified Memory Zero-Copy & Double-Buffered DMA Transfer Queue:
//    - Pipelined asynchronous FP16 conversion and transfer.
// 7. Multi-GPU & Environment Overrides:
//    - VIRA_OPENCL_LIB, VIRA_GPU_PLATFORM, VIRA_GPU_DEVICE, VIRA_GPU_WARP_SIZE,
//      VIRA_GPU_VRAM_BUDGET_MB, VIRA_GPU_ALLOW_CPU_CL.
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
#define VIRA_CL_DEVICE_TYPE_GPU (1ULL << 2)
#define VIRA_CL_DEVICE_TYPE_ALL 0xFFFFFFFFULL

#define VIRA_CL_PLATFORM_NAME 0x0902
#define VIRA_CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define VIRA_CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define VIRA_CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define VIRA_CL_DEVICE_NAME 0x102B
#define VIRA_CL_DEVICE_VENDOR 0x102C
#define VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#define VIRA_CL_PROGRAM_BUILD_LOG 0x1183

#define VIRA_CL_MEM_READ_WRITE (1ULL << 0)
#define VIRA_CL_MEM_WRITE_ONLY (1ULL << 1)
#define VIRA_CL_MEM_READ_ONLY (1ULL << 2)
#define VIRA_CL_MEM_USE_HOST_PTR (1ULL << 3)
#define VIRA_CL_MEM_ALLOC_HOST_PTR (1ULL << 4)
#define VIRA_CL_MEM_COPY_HOST_PTR (1ULL << 5)

#define VIRA_CL_MAP_READ (1ULL << 0)
#define VIRA_CL_MAP_WRITE (1ULL << 1)

typedef vira_cl_int(VIRA_CL_API *pfn_clGetPlatformIDs)(vira_cl_uint,
                                                       vira_cl_platform_id *,
                                                       vira_cl_uint *);
typedef vira_cl_int(VIRA_CL_API *pfn_clGetPlatformInfo)(vira_cl_platform_id,
                                                        vira_cl_uint, size_t,
                                                        void *, size_t *);
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
typedef vira_cl_int(VIRA_CL_API *pfn_clGetProgramBuildInfo)(
    vira_cl_program, vira_cl_device_id, vira_cl_uint, size_t, void *,
    size_t *);
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
typedef void *(VIRA_CL_API *pfn_clEnqueueMapBuffer)(
    vira_cl_command_queue, vira_cl_mem, vira_cl_uint, vira_cl_bitfield, size_t,
    size_t, vira_cl_uint, const vira_cl_event *, vira_cl_event *, vira_cl_int *);
typedef vira_cl_int(VIRA_CL_API *pfn_clEnqueueUnmapMemObject)(
    vira_cl_command_queue, vira_cl_mem, void *, vira_cl_uint,
    const vira_cl_event *, vira_cl_event *);
typedef vira_cl_int(VIRA_CL_API *pfn_clSetKernelArg)(vira_cl_kernel,
                                                     vira_cl_uint, size_t,
                                                     const void *);
typedef vira_cl_int(VIRA_CL_API *pfn_clEnqueueNDRangeKernel)(
    vira_cl_command_queue, vira_cl_kernel, vira_cl_uint, const size_t *,
    const size_t *, const size_t *, vira_cl_uint, const vira_cl_event *,
    vira_cl_event *);
typedef vira_cl_int(VIRA_CL_API *pfn_clFlush)(vira_cl_command_queue);
typedef vira_cl_int(VIRA_CL_API *pfn_clFinish)(vira_cl_command_queue);
typedef vira_cl_int(VIRA_CL_API *pfn_clWaitForEvents)(vira_cl_uint,
                                                      const vira_cl_event *);
typedef vira_cl_int(VIRA_CL_API *pfn_clReleaseEvent)(vira_cl_event);

#define VIRA_GPU_WEIGHT_CACHE_CAP 4096
#define VIRA_GPU_WEIGHT_PROBE_LEN 64
#define VIRA_GPU_ACT_RING_CAP 8
#define VIRA_GPU_NUM_KERNELS 18
#define VIRA_GPU_NUM_REQUIRED_KERNELS 12
#define VIRA_GPU_UPLOAD_CHUNK_ELEMS (1u << 22)

#define VIRA_CL_ERR_NO_ROOM (-4)
#define VIRA_CL_ERR_FP16_RANGE (-5)

typedef enum {
  VIRA_WFMT_F32 = 0,
  VIRA_WFMT_F16 = 1,
  VIRA_WFMT_Q4_0 = 2,
  VIRA_WFMT_Q4_1 = 3,
  VIRA_WFMT_Q5_0 = 6,
  VIRA_WFMT_Q5_1 = 7,
  VIRA_WFMT_Q8_0 = 8,
  VIRA_WFMT_Q4_K = 12,
  VIRA_WFMT_Q5_K = 13,
  VIRA_WFMT_Q6_K = 14,
} _vira_gpu_wfmt;

typedef struct {
  const vira_f32 *host_ptr;
  size_t elems;
  size_t bytes;
  vira_cl_mem dev_buf;
  uint64_t last_used;
  _vira_gpu_wfmt fmt;
  bool uncacheable;
} _vira_gpu_cache_entry;

typedef struct {
  const vira_f32 *host_ptr;
  size_t bytes;
  uint64_t epoch;
  vira_cl_mem dev_buf;
  size_t buf_cap;
  uint64_t last_used;
} _vira_gpu_act_entry;

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
  int32_t warp_size;
  int32_t rows_per_wg;
  int32_t wg_size;
  int32_t tile_dim;
  uint64_t access_clock;
  volatile uint64_t act_epoch;
  pthread_mutex_t lock;

  void *lib_handle;
  pfn_clGetPlatformIDs clGetPlatformIDs;
  pfn_clGetPlatformInfo clGetPlatformInfo;
  pfn_clGetDeviceIDs clGetDeviceIDs;
  pfn_clGetDeviceInfo clGetDeviceInfo;
  pfn_clCreateContext clCreateContext;
  pfn_clReleaseContext clReleaseContext;
  pfn_clCreateCommandQueue clCreateCommandQueue;
  pfn_clReleaseCommandQueue clReleaseCommandQueue;
  pfn_clCreateProgramWithSource clCreateProgramWithSource;
  pfn_clBuildProgram clBuildProgram;
  pfn_clGetProgramBuildInfo clGetProgramBuildInfo;
  pfn_clReleaseProgram clReleaseProgram;
  pfn_clCreateKernel clCreateKernel;
  pfn_clReleaseKernel clReleaseKernel;
  pfn_clCreateBuffer clCreateBuffer;
  pfn_clReleaseMemObject clReleaseMemObject;
  pfn_clEnqueueWriteBuffer clEnqueueWriteBuffer;
  pfn_clEnqueueReadBuffer clEnqueueReadBuffer;
  pfn_clEnqueueMapBuffer clEnqueueMapBuffer;
  pfn_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
  pfn_clSetKernelArg clSetKernelArg;
  pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
  pfn_clFlush clFlush;
  pfn_clFinish clFinish;
  pfn_clWaitForEvents clWaitForEvents;
  pfn_clReleaseEvent clReleaseEvent;

  vira_cl_device_id device;
  vira_cl_context context;
  vira_cl_command_queue queue;
  vira_cl_command_queue xfer_queue;
  vira_cl_program program;

  // Compute kernels
  vira_cl_kernel kernel_gemv_q4_0;
  vira_cl_kernel kernel_gemv_q4_1;
  vira_cl_kernel kernel_gemv_q5_0;
  vira_cl_kernel kernel_gemv_q5_1;
  vira_cl_kernel kernel_gemv_q8_0;
  vira_cl_kernel kernel_gemv_q4_k;
  vira_cl_kernel kernel_gemv_q5_k;
  vira_cl_kernel kernel_gemv_q6_k;
  vira_cl_kernel kernel_gemv_vec8;
  vira_cl_kernel kernel_gemv_vec4;
  vira_cl_kernel kernel_gemv_scalar;
  vira_cl_kernel kernel_linear_tiled_h;
  vira_cl_kernel kernel_linear_tiled_f32;
  vira_cl_kernel kernel_matmul_tiled;
  vira_cl_kernel kernel_rmsnorm;
  vira_cl_kernel kernel_rope;
  vira_cl_kernel kernel_softmax;
  vira_cl_kernel kernel_silu;

  // Activation Residency Ring
  _vira_gpu_act_entry act_ring[VIRA_GPU_ACT_RING_CAP];

  // Persistent weight cache
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
    .warp_size = 32,
    .rows_per_wg = 8,
    .wg_size = 256,
    .tile_dim = 16,
    .access_clock = 0,
    .act_epoch = 1,
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .lib_handle = NULL,
};

static inline void vira_gpu_init(void);
static inline void vira_gpu_shutdown(void);
static inline void _vira_gpu_evict_all_locked(void);
static inline void _vira_gpu_evict_ptr_locked(const vira_f32 *host_ptr);
static inline void _vira_gpu_evict_lru_bytes_locked(size_t needed_bytes);
static inline void _vira_gpu_invalidate_activations(void);

// ============================================================================
// OPENCL KERNELS: NATIVE GGUF (Q4_0..Q6_K) + FP16 + FP32 ACTIVATION OPERATORS
// ============================================================================
static const char *_vira_opencl_kernel_src =
    "#ifndef WARP_SIZE\n"
    "#define WARP_SIZE 32\n"
    "#endif\n"
    "#ifndef ROWS_PER_WG\n"
    "#define ROWS_PER_WG 8\n"
    "#endif\n"
    "#ifndef WG_SIZE\n"
    "#define WG_SIZE (WARP_SIZE * ROWS_PER_WG)\n"
    "#endif\n"
    "#ifndef TILE_DIM\n"
    "#define TILE_DIM 16\n"
    "#endif\n"
    "\n"
    "// 1. Native GGUF Q4_0 Multi-Row Warp GEMV (18 bytes per 32 weights)\n"
    "__kernel void vira_cl_gemv_q4_0(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_blocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_blocks_k * 32;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_blocks_k * 18ULL;\n"
    "    for (int b = lane; b < num_blocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 18ULL;\n"
    "      const float scale = vload_half(0, (__global const half*)blk);\n"
    "      __global const uchar* qs = blk + 2;\n"
    "      __global const float* xb = x_row + (size_t)b * 32ULL;\n"
    "      float bsum = 0.0f;\n"
    "      #pragma unroll 4\n"
    "      for (int j = 0; j < 16; ++j) {\n"
    "        const uchar byte = qs[j];\n"
    "        const float v0 = (float)((int)(byte & 0x0Fu) - 8);\n"
    "        const float v1 = (float)((int)(byte >> 4) - 8);\n"
    "        bsum = fma(xb[j],      v0, bsum);\n"
    "        bsum = fma(xb[j + 16], v1, bsum);\n"
    "      }\n"
    "      sum = fma(scale, bsum, sum);\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 2. Native GGUF Q4_1 Multi-Row Warp GEMV (20 bytes per 32 weights)\n"
    "__kernel void vira_cl_gemv_q4_1(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_blocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_blocks_k * 32;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_blocks_k * 20ULL;\n"
    "    for (int b = lane; b < num_blocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 20ULL;\n"
    "      const float d = vload_half(0, (__global const half*)blk);\n"
    "      const float m_val = vload_half(1, (__global const half*)blk);\n"
    "      __global const uchar* qs = blk + 4;\n"
    "      __global const float* xb = x_row + (size_t)b * 32ULL;\n"
    "      float bsum = 0.0f;\n"
    "      float xsum = 0.0f;\n"
    "      #pragma unroll 4\n"
    "      for (int j = 0; j < 16; ++j) {\n"
    "        const uchar byte = qs[j];\n"
    "        const float v0 = (float)(byte & 0x0Fu);\n"
    "        const float v1 = (float)(byte >> 4);\n"
    "        bsum = fma(xb[j],      v0, bsum);\n"
    "        bsum = fma(xb[j + 16], v1, bsum);\n"
    "        xsum += xb[j] + xb[j + 16];\n"
    "      }\n"
    "      sum = fma(d, bsum, fma(m_val, xsum, sum));\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 3. Native GGUF Q5_0 Multi-Row Warp GEMV (22 bytes per 32 weights)\n"
    "__kernel void vira_cl_gemv_q5_0(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_blocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_blocks_k * 32;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_blocks_k * 22ULL;\n"
    "    for (int b = lane; b < num_blocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 22ULL;\n"
    "      const float d = vload_half(0, (__global const half*)blk);\n"
    "      const uint qh = (uint)blk[2] | ((uint)blk[3] << 8) | ((uint)blk[4] << 16) | ((uint)blk[5] << 24);\n"
    "      __global const uchar* qs = blk + 6;\n"
    "      __global const float* xb = x_row + (size_t)b * 32ULL;\n"
    "      float bsum = 0.0f;\n"
    "      #pragma unroll 4\n"
    "      for (int j = 0; j < 16; ++j) {\n"
    "        const uchar byte = qs[j];\n"
    "        const float v0 = (float)((int)((byte & 0x0Fu) | (((qh >> j) & 1u) << 4)) - 16);\n"
    "        const float v1 = (float)((int)((byte >> 4) | (((qh >> (j + 16)) & 1u) << 4)) - 16);\n"
    "        bsum = fma(xb[j],      v0, bsum);\n"
    "        bsum = fma(xb[j + 16], v1, bsum);\n"
    "      }\n"
    "      sum = fma(d, bsum, sum);\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 4. Native GGUF Q5_1 Multi-Row Warp GEMV (24 bytes per 32 weights)\n"
    "__kernel void vira_cl_gemv_q5_1(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_blocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_blocks_k * 32;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_blocks_k * 24ULL;\n"
    "    for (int b = lane; b < num_blocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 24ULL;\n"
    "      const float d = vload_half(0, (__global const half*)blk);\n"
    "      const float m_val = vload_half(1, (__global const half*)blk);\n"
    "      const uint qh = (uint)blk[4] | ((uint)blk[5] << 8) | ((uint)blk[6] << 16) | ((uint)blk[7] << 24);\n"
    "      __global const uchar* qs = blk + 8;\n"
    "      __global const float* xb = x_row + (size_t)b * 32ULL;\n"
    "      float bsum = 0.0f;\n"
    "      float xsum = 0.0f;\n"
    "      #pragma unroll 4\n"
    "      for (int j = 0; j < 16; ++j) {\n"
    "        const uchar byte = qs[j];\n"
    "        const float v0 = (float)((byte & 0x0Fu) | (((qh >> j) & 1u) << 4));\n"
    "        const float v1 = (float)((byte >> 4) | (((qh >> (j + 16)) & 1u) << 4));\n"
    "        bsum = fma(xb[j],      v0, bsum);\n"
    "        bsum = fma(xb[j + 16], v1, bsum);\n"
    "        xsum += xb[j] + xb[j + 16];\n"
    "      }\n"
    "      sum = fma(d, bsum, fma(m_val, xsum, sum));\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 5. Native GGUF Q8_0 Multi-Row Warp GEMV (34 bytes per 32 weights)\n"
    "__kernel void vira_cl_gemv_q8_0(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_blocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_blocks_k * 32;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_blocks_k * 34ULL;\n"
    "    for (int b = lane; b < num_blocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 34ULL;\n"
    "      const float scale = vload_half(0, (__global const half*)blk);\n"
    "      __global const char* qs = (__global const char*)(blk + 2);\n"
    "      __global const float* xb = x_row + (size_t)b * 32ULL;\n"
    "      float bsum = 0.0f;\n"
    "      #pragma unroll 8\n"
    "      for (int j = 0; j < 32; ++j) {\n"
    "        bsum = fma(xb[j], (float)qs[j], bsum);\n"
    "      }\n"
    "      sum = fma(scale, bsum, sum);\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 6. Native GGUF Q4_K Multi-Row Warp GEMV (144 bytes per 256 weights)\n"
    "__kernel void vira_cl_gemv_q4_k(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_superblocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_superblocks_k * 256;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_superblocks_k * 144ULL;\n"
    "    for (int b = lane; b < num_superblocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 144ULL;\n"
    "      const float d    = vload_half(0, (__global const half*)blk);\n"
    "      const float dmin = vload_half(1, (__global const half*)blk);\n"
    "      __global const uchar* scales = blk + 4;\n"
    "      __global const uchar* qs     = blk + 16;\n"
    "      __global const float* xb     = x_row + (size_t)b * 256ULL;\n"
    "\n"
    "      float sb_sum = 0.0f;\n"
    "      for (int ib = 0; ib < 8; ++ib) {\n"
    "        uchar sc_raw, m_raw;\n"
    "        if (ib < 4) {\n"
    "          sc_raw = scales[ib] & 63u;\n"
    "          m_raw  = scales[ib + 4] & 63u;\n"
    "        } else {\n"
    "          sc_raw = (scales[ib + 4] & 0x0Fu) | ((scales[ib - 4] >> 6) << 4);\n"
    "          m_raw  = (scales[ib + 4] >> 4)    | ((scales[ib]     >> 6) << 4);\n"
    "        }\n"
    "        const float d_sub   = d * (float)sc_raw;\n"
    "        const float min_sub = dmin * (float)m_raw;\n"
    "        __global const uchar* qs_sub = qs + ((ib >> 1) * 32);\n"
    "        const int shift = (ib & 1) * 4;\n"
    "        __global const float* xb_sub = xb + ib * 32;\n"
    "\n"
    "        float sub_acc = 0.0f;\n"
    "        float sub_xsum = 0.0f;\n"
    "        #pragma unroll 8\n"
    "        for (int l = 0; l < 32; ++l) {\n"
    "          const float q = (float)((qs_sub[l] >> shift) & 0x0Fu);\n"
    "          sub_acc = fma(xb_sub[l], q, sub_acc);\n"
    "          sub_xsum += xb_sub[l];\n"
    "        }\n"
    "        sb_sum = fma(d_sub, sub_acc, sb_sum - min_sub * sub_xsum);\n"
    "      }\n"
    "      sum += sb_sum;\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 7. Native GGUF Q5_K Multi-Row Warp GEMV (176 bytes per 256 weights)\n"
    "__kernel void vira_cl_gemv_q5_k(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_superblocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_superblocks_k * 256;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_superblocks_k * 176ULL;\n"
    "    for (int b = lane; b < num_superblocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 176ULL;\n"
    "      const float d    = vload_half(0, (__global const half*)blk);\n"
    "      const float dmin = vload_half(1, (__global const half*)blk);\n"
    "      __global const uchar* scales = blk + 4;\n"
    "      __global const uchar* qh     = blk + 16;\n"
    "      __global const uchar* qs     = blk + 48;\n"
    "      __global const float* xb     = x_row + (size_t)b * 256ULL;\n"
    "\n"
    "      float sb_sum = 0.0f;\n"
    "      for (int ib = 0; ib < 8; ++ib) {\n"
    "        uchar sc_raw, m_raw;\n"
    "        if (ib < 4) {\n"
    "          sc_raw = scales[ib] & 63u;\n"
    "          m_raw  = scales[ib + 4] & 63u;\n"
    "        } else {\n"
    "          sc_raw = (scales[ib + 4] & 0x0Fu) | ((scales[ib - 4] >> 6) << 4);\n"
    "          m_raw  = (scales[ib + 4] >> 4)    | ((scales[ib]     >> 6) << 4);\n"
    "        }\n"
    "        const float d_sub   = d * (float)sc_raw;\n"
    "        const float min_sub = dmin * (float)m_raw;\n"
    "        __global const uchar* qs_sub = qs + ((ib >> 1) * 32);\n"
    "        const int shift = (ib & 1) * 4;\n"
    "        const uchar u_mask = (uchar)(1u << ib);\n"
    "        __global const float* xb_sub = xb + ib * 32;\n"
    "\n"
    "        float sub_acc = 0.0f;\n"
    "        float sub_xsum = 0.0f;\n"
    "        #pragma unroll 8\n"
    "        for (int l = 0; l < 32; ++l) {\n"
    "          const float q = (float)(((qs_sub[l] >> shift) & 0x0Fu) | ((qh[l] & u_mask) ? 16u : 0u));\n"
    "          sub_acc = fma(xb_sub[l], q, sub_acc);\n"
    "          sub_xsum += xb_sub[l];\n"
    "        }\n"
    "        sb_sum = fma(d_sub, sub_acc, sb_sum - min_sub * sub_xsum);\n"
    "      }\n"
    "      sum += sb_sum;\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 8. Native GGUF Q6_K Multi-Row Warp GEMV (210 bytes per 256 weights)\n"
    "__kernel void vira_cl_gemv_q6_k(\n"
    "    __global const float* restrict x,\n"
    "    __global const uchar* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int num_superblocks_k, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K    = num_superblocks_k * 256;\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    __global const uchar* w_row = w + (size_t)n * (size_t)num_superblocks_k * 210ULL;\n"
    "    for (int b = lane; b < num_superblocks_k; b += WARP_SIZE) {\n"
    "      __global const uchar* blk = w_row + (size_t)b * 210ULL;\n"
    "      const float d = vload_half(0, (__global const half*)(blk + 208));\n"
    "      __global const float* xb = x_row + (size_t)b * 256ULL;\n"
    "      float sb_sum = 0.0f;\n"
    "\n"
    "      for (int h = 0; h < 2; ++h) {\n"
    "        __global const uchar* ql_h = blk + h * 64;\n"
    "        __global const uchar* qh_h = blk + 128 + h * 32;\n"
    "        __global const char*  sc_h = (__global const char*)(blk + 192) + h * 8;\n"
    "        __global const float* xb_h = xb + h * 128;\n"
    "\n"
    "        #pragma unroll 4\n"
    "        for (int l = 0; l < 32; ++l) {\n"
    "          const int is = l >> 4;\n"
    "          const float q1 = (float)((int)((ql_h[l + 0]  & 0x0Fu) | (((qh_h[l] >> 0) & 3u) << 4)) - 32);\n"
    "          const float q2 = (float)((int)((ql_h[l + 32] & 0x0Fu) | (((qh_h[l] >> 2) & 3u) << 4)) - 32);\n"
    "          const float q3 = (float)((int)((ql_h[l + 0]  >> 4)    | (((qh_h[l] >> 4) & 3u) << 4)) - 32);\n"
    "          const float q4 = (float)((int)((ql_h[l + 32] >> 4)    | (((qh_h[l] >> 6) & 3u) << 4)) - 32);\n"
    "\n"
    "          const float w1 = d * (float)sc_h[is + 0] * q1;\n"
    "          const float w2 = d * (float)sc_h[is + 2] * q2;\n"
    "          const float w3 = d * (float)sc_h[is + 4] * q3;\n"
    "          const float w4 = d * (float)sc_h[is + 6] * q4;\n"
    "\n"
    "          sb_sum = fma(xb_h[l + 0],  w1, sb_sum);\n"
    "          sb_sum = fma(xb_h[l + 32], w2, sb_sum);\n"
    "          sb_sum = fma(xb_h[l + 64], w3, sb_sum);\n"
    "          sb_sum = fma(xb_h[l + 96], w4, sb_sum);\n"
    "        }\n"
    "      }\n"
    "      sum += sb_sum;\n"
    "    }\n"
    "  }\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 9. FP16-weight Multi-Row Warp GEMV, float4 loads (K % 8 == 0)\n"
    "__kernel void vira_cl_gemv_vec8_h(\n"
    "    __global const float4* restrict x,\n"
    "    __global const half* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K8, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "  const int K4   = K8 * 2;\n"
    "\n"
    "  float4 acc0 = (float4)(0.0f);\n"
    "  float4 acc1 = (float4)(0.0f);\n"
    "  if (n < N && m < M) {\n"
    "    __global const float4* x_row = x + (size_t)m * (size_t)K4;\n"
    "    const size_t w_off = (size_t)n * (size_t)K4;\n"
    "    int k = lane;\n"
    "    for (; k + WARP_SIZE < K4; k += (WARP_SIZE * 2)) {\n"
    "      acc0 = fma(x_row[k],             vload_half4(w_off + (size_t)k, w),             acc0);\n"
    "      acc1 = fma(x_row[k + WARP_SIZE], vload_half4(w_off + (size_t)(k + WARP_SIZE), w), acc1);\n"
    "    }\n"
    "    if (k < K4) {\n"
    "      acc0 = fma(x_row[k], vload_half4(w_off + (size_t)k, w), acc0);\n"
    "    }\n"
    "  }\n"
    "  const float4 s4 = acc0 + acc1;\n"
    "  const float sum = (s4.x + s4.y) + (s4.z + s4.w);\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 10. FP16-weight Multi-Row Warp GEMV, float4 loads (K % 4 == 0)\n"
    "__kernel void vira_cl_gemv_vec4_h(\n"
    "    __global const float4* restrict x,\n"
    "    __global const half* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K4, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "\n"
    "  float4 acc0 = (float4)(0.0f);\n"
    "  if (n < N && m < M) {\n"
    "    __global const float4* x_row = x + (size_t)m * (size_t)K4;\n"
    "    const size_t w_off = (size_t)n * (size_t)K4;\n"
    "    for (int k = lane; k < K4; k += WARP_SIZE) {\n"
    "      acc0 = fma(x_row[k], vload_half4(w_off + (size_t)k, w), acc0);\n"
    "    }\n"
    "  }\n"
    "  const float sum = (acc0.x + acc0.y) + (acc0.z + acc0.w);\n"
    "\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 11. Scalar FP16-weight Multi-Row Warp GEMV fallback\n"
    "__kernel void vira_cl_gemv_scalar_h(\n"
    "    __global const float* restrict x,\n"
    "    __global const half* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int lane = (int)get_local_id(0);\n"
    "  const int sg   = (int)get_local_id(1);\n"
    "  const int n    = (int)get_group_id(0) * ROWS_PER_WG + sg;\n"
    "  const int m    = (int)get_group_id(1);\n"
    "\n"
    "  float sum = 0.0f;\n"
    "  if (n < N && m < M) {\n"
    "    __global const float* x_row = x + (size_t)m * (size_t)K;\n"
    "    const size_t w_off = (size_t)n * (size_t)K;\n"
    "    for (int k = lane; k < K; k += WARP_SIZE) {\n"
    "      sum = fma(x_row[k], vload_half(w_off + (size_t)k, w), sum);\n"
    "    }\n"
    "  }\n"
    "  __local float sdata[ROWS_PER_WG][WARP_SIZE];\n"
    "  sdata[sg][lane] = sum;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#if WARP_SIZE >= 64\n"
    "  if (lane < 32) sdata[sg][lane] += sdata[sg][lane + 32];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#endif\n"
    "  if (lane < 16) sdata[sg][lane] += sdata[sg][lane + 16];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 8)  sdata[sg][lane] += sdata[sg][lane + 8];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane < 4)  sdata[sg][lane] += sdata[sg][lane + 4];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (lane == 0 && n < N && m < M) {\n"
    "    y[(size_t)m * (size_t)N + (size_t)n] =\n"
    "        (sdata[sg][0] + sdata[sg][1]) + (sdata[sg][2] + sdata[sg][3]);\n"
    "  }\n"
    "}\n"
    "\n"
    "// 12. Tiled linear projection with FP16 weights\n"
    "__kernel void vira_cl_linear_tiled_h(\n"
    "    __global const float* restrict x,\n"
    "    __global const half* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int tx = (int)get_local_id(0);\n"
    "  const int ty = (int)get_local_id(1);\n"
    "  const int col = (int)get_group_id(0) * TILE_DIM + tx;\n"
    "  const int row = (int)get_group_id(1) * TILE_DIM + ty;\n"
    "  __local float tile_x[TILE_DIM][TILE_DIM];\n"
    "  __local float tile_w[TILE_DIM][TILE_DIM + 1];\n"
    "  float acc = 0.0f;\n"
    "  const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;\n"
    "  for (int t = 0; t < num_tiles; ++t) {\n"
    "    const int kx = t * TILE_DIM + tx;\n"
    "    tile_x[ty][tx] = (row < M && kx < K) ? x[(size_t)row * (size_t)K + (size_t)kx] : 0.0f;\n"
    "    const int w_row_idx = (int)get_group_id(0) * TILE_DIM + ty;\n"
    "    const int w_col_idx = t * TILE_DIM + tx;\n"
    "    tile_w[ty][tx] = (w_row_idx < N && w_col_idx < K)\n"
    "                         ? vload_half((size_t)w_row_idx * (size_t)K + (size_t)w_col_idx, w)\n"
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
    "// 13. Tiled linear projection with FP32 weights (for dynamic activation matmul x @ w^T)\n"
    "__kernel void vira_cl_linear_tiled_f32(\n"
    "    __global const float* restrict x,\n"
    "    __global const float* restrict w,\n"
    "    __global float* restrict y,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int tx = (int)get_local_id(0);\n"
    "  const int ty = (int)get_local_id(1);\n"
    "  const int col = (int)get_group_id(0) * TILE_DIM + tx;\n"
    "  const int row = (int)get_group_id(1) * TILE_DIM + ty;\n"
    "  __local float tile_x[TILE_DIM][TILE_DIM];\n"
    "  __local float tile_w[TILE_DIM][TILE_DIM + 1];\n"
    "  float acc = 0.0f;\n"
    "  const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;\n"
    "  for (int t = 0; t < num_tiles; ++t) {\n"
    "    const int kx = t * TILE_DIM + tx;\n"
    "    tile_x[ty][tx] = (row < M && kx < K) ? x[(size_t)row * (size_t)K + (size_t)kx] : 0.0f;\n"
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
    "// 14. Tiled FP32 matmul (c = a @ b)\n"
    "__kernel void vira_cl_matmul_tiled_f32(\n"
    "    __global const float* restrict a,\n"
    "    __global const float* restrict b,\n"
    "    __global float* restrict c,\n"
    "    const int M, const int K, const int N) {\n"
    "  const int tx = (int)get_local_id(0);\n"
    "  const int ty = (int)get_local_id(1);\n"
    "  const int col = (int)get_group_id(0) * TILE_DIM + tx;\n"
    "  const int row = (int)get_group_id(1) * TILE_DIM + ty;\n"
    "  __local float tile_a[TILE_DIM][TILE_DIM];\n"
    "  __local float tile_b[TILE_DIM][TILE_DIM];\n"
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
    "// 15. Work-group RMSNorm\n"
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
    "    const float v = x_row[c];\n"
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
    "// 16. Rotary Position Embedding (RoPE)\n"
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
    "  const float cos_a = cos(angle);\n"
    "  const float sin_a = sin(angle);\n"
    "  __global const float* q_row = q + (size_t)r * (size_t)cols;\n"
    "  __global float* o_row = out + (size_t)r * (size_t)cols;\n"
    "  const float q0 = q_row[i];\n"
    "  const float q1 = q_row[i + half_dim];\n"
    "  o_row[i]            = fma(q0, cos_a, -q1 * sin_a);\n"
    "  o_row[i + half_dim] = fma(q0, sin_a,  q1 * cos_a);\n"
    "}\n"
    "\n"
    "// 17. Row-wise numerically stable Softmax\n"
    "__kernel void vira_cl_softmax_f32(\n"
    "    __global const float* restrict logits,\n"
    "    __global float* restrict out,\n"
    "    const int rows, const int cols) {\n"
    "  const int lid = (int)get_local_id(0);\n"
    "  const int row = (int)get_group_id(0);\n"
    "  if (row >= rows) return;\n"
    "  __global const float* in_row = logits + (size_t)row * (size_t)cols;\n"
    "  __global float* out_row = out + (size_t)row * (size_t)cols;\n"
    "\n"
    "  float max_v = -1e30f;\n"
    "  for (int c = lid; c < cols; c += WG_SIZE) {\n"
    "    if (in_row[c] > max_v) max_v = in_row[c];\n"
    "  }\n"
    "  __local float sdata[WG_SIZE];\n"
    "  sdata[lid] = max_v;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {\n"
    "    if (lid < s) {\n"
    "      if (sdata[lid + s] > sdata[lid]) sdata[lid] = sdata[lid + s];\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  const float row_max = sdata[0];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "  float sum_e = 0.0f;\n"
    "  for (int c = lid; c < cols; c += WG_SIZE) {\n"
    "    sum_e += native_exp(in_row[c] - row_max);\n"
    "  }\n"
    "  sdata[lid] = sum_e;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  for (int s = WG_SIZE >> 1; s > 0; s >>= 1) {\n"
    "    if (lid < s) sdata[lid] += sdata[lid + s];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  const float inv_sum = sdata[0] > 0.0f ? (1.0f / sdata[0]) : 1.0f;\n"
    "  for (int c = lid; c < cols; c += WG_SIZE) {\n"
    "    out_row[c] = native_exp(in_row[c] - row_max) * inv_sum;\n"
    "  }\n"
    "}\n"
    "\n"
    "// 18. SiLU activation\n"
    "__kernel void vira_cl_silu_f32(\n"
    "    __global const float* restrict x,\n"
    "    __global float* restrict out,\n"
    "    const int total) {\n"
    "  const int idx = (int)get_global_id(0);\n"
    "  if (idx >= total) return;\n"
    "  const float v = x[idx];\n"
    "  out[idx] = v / (1.0f + native_exp(-v));\n"
    "}\n";

static const char *_vira_opencl_kernel_names[VIRA_GPU_NUM_KERNELS] = {
    "vira_cl_gemv_q4_0",        "vira_cl_gemv_q4_1",
    "vira_cl_gemv_q5_0",        "vira_cl_gemv_q5_1",
    "vira_cl_gemv_q8_0",        "vira_cl_gemv_q4_k",
    "vira_cl_gemv_q5_k",        "vira_cl_gemv_q6_k",
    "vira_cl_gemv_vec8_h",      "vira_cl_gemv_vec4_h",
    "vira_cl_gemv_scalar_h",    "vira_cl_linear_tiled_h",
    "vira_cl_linear_tiled_f32", "vira_cl_matmul_tiled_f32",
    "vira_cl_rmsnorm_f32",      "vira_cl_rope_f32",
    "vira_cl_softmax_f32",      "vira_cl_silu_f32",
};

/**
 * Checks whether the `VIRA_GPU_DEBUG` environment variable is set to a truthy value.
 * @return True if GPU debug logging is enabled, false otherwise.
 * Usage: Controls verbose OpenCL build log output and debug diagnostics.
 */
static inline bool _vira_gpu_debug(void) {
  const char *e = getenv("VIRA_GPU_DEBUG");
  return e != NULL && e[0] != '\0' && e[0] != '0';
}

/**
 * Dynamically resolves an OpenCL symbol from the loaded shared library or DLL.
 * @param lib Handle to the loaded OpenCL dynamic library.
 * @param name C-string identifier of the API symbol (e.g. "clCreateBuffer").
 * @param fn_out Pointer to function pointer storage receiving the resolved address.
 * @return True if symbol was found and assigned, false otherwise.
 * Usage: Internal helper during dynamic OpenCL dispatch table binding.
 */
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

/**
 * Converts a 32-bit IEEE 754 single-precision float bit pattern to IEEE 754 half-precision float (FP16).
 * @param x 32-bit unsigned integer containing the raw bits of an f32.
 * @return 16-bit unsigned integer encoding the converted half-precision float.
 * Usage: Used to convert model weights to FP16 before uploading to GPU VRAM for 2x memory reduction and accelerated compute.
 */
static inline uint16_t _vira_f32_bits_to_f16(uint32_t x) {
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t exp = (x >> 23) & 0xFFu;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp == 0xFFu) {
    return (uint16_t)(sign | 0x7C00u | (mant ? (0x200u | (mant >> 13)) : 0u));
  }
  const int32_t e = (int32_t)exp - 127 + 15;
  if (e >= 31) {
    return (uint16_t)(sign | 0x7C00u);
  }
  if (e <= 0) {
    if (e < -10) {
      return (uint16_t)sign;
    }
    mant |= 0x800000u;
    const uint32_t shift = (uint32_t)(14 - e);
    uint32_t h = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (h & 1u))) {
      h++;
    }
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
    h++;
  }
  return (uint16_t)(sign | h);
}

/**
 * Batch converts an array of single-precision f32 values to half-precision FP16.
 * @param src Pointer to input array of 32-bit floats.
 * @param dst Pointer to destination buffer for 16-bit half floats.
 * @param n Number of elements to convert.
 * @return True if all values converted cleanly within range, false on non-finite overflow.
 * Usage: Quantizes float arrays in chunks during weight upload.
 */
static inline bool _vira_f32_to_f16_block(const vira_f32 *src, uint16_t *dst,
                                          size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits;
    memcpy(&bits, &src[i], sizeof(bits));
    const uint16_t h = _vira_f32_bits_to_f16(bits);
    if (((bits >> 23) & 0xFFu) != 0xFFu && (h & 0x7FFFu) == 0x7C00u) {
      return false;
    }
    dst[i] = h;
  }
  return true;
}

/**
 * Helper to bind standard matrix multiplication arguments to an OpenCL GEMM kernel.
 * @param k OpenCL kernel object.
 * @param a Pointer to matrix A GPU buffer.
 * @param b Pointer to matrix B GPU buffer.
 * @param c Pointer to matrix C GPU buffer.
 * @param d0 First dimension parameter (e.g. M).
 * @param d1 Second dimension parameter (e.g. K).
 * @param d2 Third dimension parameter (e.g. N).
 * @return OpenCL status code (0 on success).
 * Usage: Sets kernel arguments 0 through 5 for linear and matmul dispatch.
 */
static inline vira_cl_int _vira_gpu_set_gemm_args(vira_cl_kernel k,
                                                  vira_cl_mem *a,
                                                  vira_cl_mem *b,
                                                  vira_cl_mem *c, int32_t d0,
                                                  int32_t d1, int32_t d2) {
  vira_cl_int e = 0;
  e |= _vira_gpu.clSetKernelArg(k, 0, sizeof(vira_cl_mem), a);
  e |= _vira_gpu.clSetKernelArg(k, 1, sizeof(vira_cl_mem), b);
  e |= _vira_gpu.clSetKernelArg(k, 2, sizeof(vira_cl_mem), c);
  e |= _vira_gpu.clSetKernelArg(k, 3, sizeof(int32_t), &d0);
  e |= _vira_gpu.clSetKernelArg(k, 4, sizeof(int32_t), &d1);
  e |= _vira_gpu.clSetKernelArg(k, 5, sizeof(int32_t), &d2);
  return e;
}

/**
 * Retrieves and prints the OpenCL C program compilation log when build errors occur.
 * @param prog OpenCL program object that failed compilation.
 * @param dev Target OpenCL device ID.
 * Usage: Diagnostic logging when custom kernels or OpenCL drivers report syntax errors.
 */
static inline void _vira_gpu_dump_build_log(vira_cl_program prog,
                                            vira_cl_device_id dev) {
  if (!_vira_gpu_debug() || _vira_gpu.clGetProgramBuildInfo == NULL) {
    return;
  }
  char blog[8192] = {0};
  if (_vira_gpu.clGetProgramBuildInfo(prog, dev, VIRA_CL_PROGRAM_BUILD_LOG,
                                      sizeof(blog) - 1, blog,
                                      NULL) == VIRA_CL_SUCCESS) {
    fprintf(stderr, "vira gpu: OpenCL build failed:\n%s\n", blog);
  }
}

/**
 * Resets a weight cache entry to its initial empty state.
 * @param e Pointer to cache entry struct.
 * Usage: Cleans up cached tensor metadata when evicted from VRAM.
 */
static inline void _vira_gpu_cache_entry_reset(_vira_gpu_cache_entry *e) {
  e->host_ptr = NULL;
  e->elems = 0;
  e->bytes = 0;
  e->dev_buf = NULL;
  e->last_used = 0;
  e->fmt = VIRA_WFMT_F32;
  e->uncacheable = false;
}

/**
 * Resets an activation ring entry to its empty unmapped state.
 * @param e Pointer to activation ring entry.
 * Usage: Resets entry metadata during invalidation or eviction.
 */
static inline void _vira_gpu_act_entry_reset(_vira_gpu_act_entry *e) {
  e->host_ptr = NULL;
  e->bytes = 0;
  e->epoch = 0;
  e->last_used = 0;
}

/**
 * Increments the global GPU activation residency epoch counter in O(1).
 * Usage: Automatically invalidates all resident activation buffers when memory arenas reset or tensors mutate.
 */
static inline void _vira_gpu_invalidate_activations(void) {
  __atomic_add_fetch(&_vira_gpu.act_epoch, 1, __ATOMIC_RELEASE);
}

/**
 * Checks whether a given host memory address currently resides in the GPU activation residency ring.
 * @param ptr Host tensor buffer address.
 * @return True if pointer is an active activation buffer, false otherwise.
 * Usage: Distinguishes dynamic activations from static model weights in `vira_gpu_linear`.
 */
static inline bool _vira_gpu_is_activation_ptr_locked(const vira_f32 *ptr) {
  if (ptr == NULL) return false;
  uint64_t ep = __atomic_load_n(&_vira_gpu.act_epoch, __ATOMIC_ACQUIRE);
  for (int i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    if (_vira_gpu.act_ring[i].dev_buf != NULL &&
        _vira_gpu.act_ring[i].epoch == ep &&
        _vira_gpu.act_ring[i].host_ptr == ptr) {
      return true;
    }
  }
  return false;
}

/**
 * Evicts oldest cached weights until at least needed_bytes are freed under VRAM budget pressure.
 * @param needed_bytes Number of bytes required.
 * Usage: True LRU weight cache management.
 */
static inline void _vira_gpu_evict_lru_bytes_locked(size_t needed_bytes) {
  if (!_vira_gpu.available) return;
  int64_t cur_alloc =
      __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
  if (_vira_gpu.vram_budget_bytes <= 0 ||
      cur_alloc + (int64_t)needed_bytes <= _vira_gpu.vram_budget_bytes) {
    return;
  }

  while (cur_alloc + (int64_t)needed_bytes > _vira_gpu.vram_budget_bytes) {
    int32_t oldest_idx = -1;
    uint64_t oldest_clock = 0xFFFFFFFFFFFFFFFFULL;
    for (int32_t i = 0; i < VIRA_GPU_WEIGHT_CACHE_CAP; ++i) {
      _vira_gpu_cache_entry *e = &_vira_gpu.weight_cache[i];
      if (e->dev_buf != NULL && !e->uncacheable) {
        if (e->last_used < oldest_clock) {
          oldest_clock = e->last_used;
          oldest_idx = i;
        }
      }
    }
    if (oldest_idx < 0) {
      break;
    }
    _vira_gpu_cache_entry *slot = &_vira_gpu.weight_cache[oldest_idx];
    if (slot->dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(slot->dev_buf);
      __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)slot->bytes,
                         __ATOMIC_RELAXED);
    }
    _vira_gpu_cache_entry_reset(slot);
    cur_alloc = __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
  }
}

/**
 * Retrieves a resident activation buffer from the VRAM activation ring or uploads CPU tensor data into an LRU slot.
 * @param ptr Host float tensor data pointer.
 * @param bytes Size in bytes of tensor data.
 * @param out_hit Output boolean set to true if PCIe upload was avoided (zero-copy cache hit).
 * @param out_err Pointer to receive OpenCL status code.
 * @return OpenCL buffer handle containing tensor data in GPU VRAM, or NULL on error.
 * Usage: Chains Transformer operators with zero host->device copies when activations are chained.
 */
static inline vira_cl_mem _vira_gpu_get_or_upload_activation_locked(
    const vira_f32 *ptr, size_t bytes, bool *out_hit, vira_cl_int *out_err) {
  uint64_t ep = __atomic_load_n(&_vira_gpu.act_epoch, __ATOMIC_ACQUIRE);
  uint64_t now = ++_vira_gpu.access_clock;

  // 1. Check for residency hit
  for (int i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    _vira_gpu_act_entry *e = &_vira_gpu.act_ring[i];
    if (e->dev_buf != NULL && e->epoch == ep && e->host_ptr == ptr &&
        e->bytes == bytes) {
      e->last_used = now;
      if (out_hit) *out_hit = true;
      *out_err = VIRA_CL_SUCCESS;
      return e->dev_buf;
    }
  }

  // 2. Cache miss: find an empty, expired, or LRU slot
  if (out_hit) *out_hit = false;
  int best_slot = -1;
  uint64_t oldest_clock = 0xFFFFFFFFFFFFFFFFULL;
  for (int i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    _vira_gpu_act_entry *e = &_vira_gpu.act_ring[i];
    if (e->dev_buf == NULL || e->epoch != ep) {
      best_slot = i;
      break;
    }
    if (e->last_used < oldest_clock) {
      oldest_clock = e->last_used;
      best_slot = i;
    }
  }

  _vira_gpu_act_entry *slot = &_vira_gpu.act_ring[best_slot];
  if (slot->dev_buf == NULL || slot->buf_cap < bytes) {
    if (slot->dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(slot->dev_buf);
      __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)slot->buf_cap,
                         __ATOMIC_RELAXED);
      slot->dev_buf = NULL;
      slot->buf_cap = 0;
    }
    size_t new_cap = (bytes < (1024ULL * 1024ULL))
                         ? (1024ULL * 1024ULL)
                         : ((bytes + 65535ULL) & ~65535ULL);
    vira_cl_int err = 0;
    vira_cl_mem nb = _vira_gpu.clCreateBuffer(
        _vira_gpu.context, VIRA_CL_MEM_READ_WRITE, new_cap, NULL, &err);
    if (err != VIRA_CL_SUCCESS || nb == NULL) {
      *out_err = (err != VIRA_CL_SUCCESS) ? err : VIRA_CL_ERR_NO_ROOM;
      return NULL;
    }
    slot->dev_buf = nb;
    slot->buf_cap = new_cap;
    __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)new_cap,
                       __ATOMIC_RELAXED);
  }

  vira_cl_int err = _vira_gpu.clEnqueueWriteBuffer(
      _vira_gpu.queue, slot->dev_buf, 1, 0, bytes, ptr, 0, NULL, NULL);
  if (err != VIRA_CL_SUCCESS) {
    *out_err = err;
    return NULL;
  }
  slot->host_ptr = ptr;
  slot->bytes = bytes;
  slot->epoch = ep;
  slot->last_used = now;
  *out_err = VIRA_CL_SUCCESS;
  return slot->dev_buf;
}

/**
 * Allocates or reuses a destination VRAM activation slot to hold a GPU operator output.
 * @param needed_bytes Minimum byte capacity required for operator output.
 * @param avoid1 OpenCL buffer handle to avoid evicting (e.g. primary input).
 * @param avoid2 Secondary buffer handle to avoid evicting (or NULL).
 * @param out_err Pointer to receive OpenCL error code.
 * @return Pointer to acquired `_vira_gpu_act_entry` ring slot, or NULL on error.
 * Usage: Secures intermediate GPU output buffers while preserving active input operands.
 */
static inline _vira_gpu_act_entry *_vira_gpu_alloc_activation_slot_locked(
    size_t needed_bytes, vira_cl_mem avoid1, vira_cl_mem avoid2,
    vira_cl_int *out_err) {
  uint64_t ep = __atomic_load_n(&_vira_gpu.act_epoch, __ATOMIC_ACQUIRE);
  uint64_t now = ++_vira_gpu.access_clock;

  int best_slot = -1;
  uint64_t oldest_clock = 0xFFFFFFFFFFFFFFFFULL;
  for (int i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    _vira_gpu_act_entry *e = &_vira_gpu.act_ring[i];
    if (e->dev_buf != NULL && (e->dev_buf == avoid1 || e->dev_buf == avoid2)) {
      continue;
    }
    if (e->dev_buf == NULL || e->epoch != ep) {
      best_slot = i;
      break;
    }
    if (e->last_used < oldest_clock) {
      oldest_clock = e->last_used;
      best_slot = i;
    }
  }

  if (best_slot < 0) {
    *out_err = VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }

  _vira_gpu_act_entry *slot = &_vira_gpu.act_ring[best_slot];
  if (slot->dev_buf == NULL || slot->buf_cap < needed_bytes) {
    if (slot->dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(slot->dev_buf);
      __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)slot->buf_cap,
                         __ATOMIC_RELAXED);
      slot->dev_buf = NULL;
      slot->buf_cap = 0;
    }
    size_t new_cap = (needed_bytes < (1024ULL * 1024ULL))
                         ? (1024ULL * 1024ULL)
                         : ((needed_bytes + 65535ULL) & ~65535ULL);
    vira_cl_int err = 0;
    vira_cl_mem nb = _vira_gpu.clCreateBuffer(
        _vira_gpu.context, VIRA_CL_MEM_READ_WRITE, new_cap, NULL, &err);
    if (err != VIRA_CL_SUCCESS || nb == NULL) {
      *out_err = (err != VIRA_CL_SUCCESS) ? err : VIRA_CL_ERR_NO_ROOM;
      return NULL;
    }
    slot->dev_buf = nb;
    slot->buf_cap = new_cap;
    __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)new_cap,
                       __ATOMIC_RELAXED);
  }

  slot->host_ptr = NULL;
  slot->bytes = needed_bytes;
  slot->epoch = ep;
  slot->last_used = now;
  *out_err = VIRA_CL_SUCCESS;
  return slot;
}

/**
 * Evicts a specific tensor host pointer from the GPU VRAM cache while holding the GPU mutex.
 * @param host_ptr Memory address of the CPU tensor data to release.
 * Usage: Called when a tensor is deallocated or modified to invalidate its GPU VRAM copy.
 */
static inline void _vira_gpu_evict_ptr_locked(const vira_f32 *host_ptr) {
  if (!_vira_gpu.available || host_ptr == NULL) {
    return;
  }
  // Evict from activation ring
  for (int i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    if (_vira_gpu.act_ring[i].host_ptr == host_ptr) {
      _vira_gpu_act_entry_reset(&_vira_gpu.act_ring[i]);
    }
  }

  uint64_t h = ((uint64_t)(uintptr_t)host_ptr >> 6) * 11400714819323198485ULL;
  int32_t start_slot = (int32_t)(h & (VIRA_GPU_WEIGHT_CACHE_CAP - 1));

  for (int32_t probe = 0; probe < VIRA_GPU_WEIGHT_PROBE_LEN; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    _vira_gpu_cache_entry *e = &_vira_gpu.weight_cache[idx];
    if (e->host_ptr == host_ptr) {
      if (e->dev_buf != NULL) {
        _vira_gpu.clReleaseMemObject(e->dev_buf);
        __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)e->bytes,
                           __ATOMIC_RELAXED);
      }
      _vira_gpu_cache_entry_reset(e);
    }
  }
}

/**
 * Thread-safe wrapper to evict a specific host tensor from the GPU weight and activation caches.
 * @param host_ptr Memory address of the CPU tensor.
 * Usage: Invoked by tensor destructor `vira_tensor_f32_free()` to maintain GPU cache consistency.
 */
static inline void _vira_gpu_evict_ptr(const vira_f32 *host_ptr) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) != 2 ||
      !_vira_gpu.available || host_ptr == NULL) {
    return;
  }
  pthread_mutex_lock(&_vira_gpu.lock);
  _vira_gpu_evict_ptr_locked(host_ptr);
  pthread_mutex_unlock(&_vira_gpu.lock);
}

/**
 * Releases all cached weights and activation buffers from GPU VRAM while holding the GPU mutex.
 * Usage: Used during emergency OOM recovery or system shutdown to reclaim all device memory.
 */
static inline void _vira_gpu_evict_all_locked(void) {
  if (!_vira_gpu.available) {
    return;
  }
  for (int32_t i = 0; i < VIRA_GPU_WEIGHT_CACHE_CAP; ++i) {
    if (_vira_gpu.weight_cache[i].dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(_vira_gpu.weight_cache[i].dev_buf);
    }
    _vira_gpu_cache_entry_reset(&_vira_gpu.weight_cache[i]);
  }
  for (int32_t i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    if (_vira_gpu.act_ring[i].dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(_vira_gpu.act_ring[i].dev_buf);
      _vira_gpu.act_ring[i].dev_buf = NULL;
      _vira_gpu.act_ring[i].buf_cap = 0;
    }
    _vira_gpu_act_entry_reset(&_vira_gpu.act_ring[i]);
  }
  _vira_gpu.vram_saturated = false;
  __atomic_store_n(&_vira_gpu.allocated_bytes, 0, __ATOMIC_RELAXED);
  _vira_gpu_invalidate_activations();
}

/**
 * Directly registers and uploads raw quantized weight blocks (Q4_0..Q6_K) to GPU VRAM during model loading.
 * @param host_ptr CPU tensor data pointer serving as cache key.
 * @param gguf_dtype Quantization type from GGUF header (2, 3, 6, 7, 8, 12, 13, 14).
 * @param raw_blocks Pointer to raw quantized weight blocks.
 * @param raw_bytes Size in bytes of quantized weight data.
 * @param elems Total number of logical tensor elements.
 * Usage: Invoked by `vira_gguf_load_tensor` to stream quantized weights directly into VRAM for ultra-fast GPU GEMV inference.
 */
static inline void _vira_gpu_register_quant(const vira_f32 *host_ptr,
                                            uint32_t gguf_dtype,
                                            const void *raw_blocks,
                                            size_t raw_bytes, size_t elems) {
  vira_gpu_init();
  if (!_vira_gpu.available || !_vira_gpu.enabled || host_ptr == NULL ||
      raw_blocks == NULL || raw_bytes == 0 || (elems & 31) != 0) {
    return;
  }
  _vira_gpu_wfmt fmt;
  switch (gguf_dtype) {
  case 2:  fmt = VIRA_WFMT_Q4_0; break;
  case 3:  fmt = VIRA_WFMT_Q4_1; break;
  case 6:  fmt = VIRA_WFMT_Q5_0; break;
  case 7:  fmt = VIRA_WFMT_Q5_1; break;
  case 8:  fmt = VIRA_WFMT_Q8_0; break;
  case 12: fmt = VIRA_WFMT_Q4_K; break;
  case 13: fmt = VIRA_WFMT_Q5_K; break;
  case 14: fmt = VIRA_WFMT_Q6_K; break;
  default: return;
  }

  pthread_mutex_lock(&_vira_gpu.lock);
  _vira_gpu_evict_lru_bytes_locked(raw_bytes);
  int64_t cur_alloc =
      __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
  if (_vira_gpu.vram_budget_bytes > 0 &&
      cur_alloc + (int64_t)raw_bytes > _vira_gpu.vram_budget_bytes) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return;
  }

  uint64_t h = ((uint64_t)(uintptr_t)host_ptr >> 6) * 11400714819323198485ULL;
  int32_t start_slot = (int32_t)(h & (VIRA_GPU_WEIGHT_CACHE_CAP - 1));
  int32_t empty_slot = -1;
  int32_t oldest_probe = -1;
  uint64_t oldest_clock = 0xFFFFFFFFFFFFFFFFULL;

  for (int32_t probe = 0; probe < VIRA_GPU_WEIGHT_PROBE_LEN; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    if (_vira_gpu.weight_cache[idx].host_ptr == host_ptr) {
      pthread_mutex_unlock(&_vira_gpu.lock);
      return;
    }
    if (empty_slot < 0 && _vira_gpu.weight_cache[idx].host_ptr == NULL) {
      empty_slot = idx;
    }
    if (!_vira_gpu.weight_cache[idx].uncacheable &&
        _vira_gpu.weight_cache[idx].last_used < oldest_clock) {
      oldest_clock = _vira_gpu.weight_cache[idx].last_used;
      oldest_probe = idx;
    }
  }

  if (empty_slot < 0 && oldest_probe >= 0) {
    _vira_gpu_cache_entry *ev = &_vira_gpu.weight_cache[oldest_probe];
    if (ev->dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(ev->dev_buf);
      __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)ev->bytes,
                         __ATOMIC_RELAXED);
    }
    _vira_gpu_cache_entry_reset(ev);
    empty_slot = oldest_probe;
  }

  if (empty_slot < 0) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return;
  }

  vira_cl_int err = 0;
  vira_cl_mem buf = _vira_gpu.clCreateBuffer(
      _vira_gpu.context, VIRA_CL_MEM_READ_ONLY | VIRA_CL_MEM_COPY_HOST_PTR,
      raw_bytes, (void *)raw_blocks, &err);
  if (err == VIRA_CL_SUCCESS && buf != NULL) {
    _vira_gpu_cache_entry *slot = &_vira_gpu.weight_cache[empty_slot];
    slot->host_ptr = host_ptr;
    slot->elems = elems;
    slot->bytes = raw_bytes;
    slot->dev_buf = buf;
    slot->last_used = ++_vira_gpu.access_clock;
    slot->fmt = fmt;
    slot->uncacheable = false;
    __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)raw_bytes,
                       __ATOMIC_RELAXED);
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
}

/**
 * Converts an array of single-precision floats to half-precision (FP16) and uploads them to a read-only OpenCL buffer using double-buffering.
 * @param src Pointer to CPU float data.
 * @param elems Number of float elements.
 * @param out_err Pointer to receive error code.
 * @return OpenCL memory buffer handle, or NULL on error.
 * Usage: Pipelined DMA staging of FP16 model weights in GPU VRAM.
 */
static inline vira_cl_mem _vira_gpu_upload_half_locked(const vira_f32 *src,
                                                       size_t elems,
                                                       vira_cl_int *out_err) {
  const size_t bytes = elems * sizeof(uint16_t);
  vira_cl_int err = 0;
  vira_cl_mem buf = _vira_gpu.clCreateBuffer(
      _vira_gpu.context, VIRA_CL_MEM_READ_ONLY, bytes, NULL, &err);
  if (err != VIRA_CL_SUCCESS || buf == NULL) {
    *out_err = (err != VIRA_CL_SUCCESS) ? err : VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }

  const size_t chunk =
      elems < VIRA_GPU_UPLOAD_CHUNK_ELEMS ? elems : VIRA_GPU_UPLOAD_CHUNK_ELEMS;
  uint16_t *tmp[2];
  tmp[0] = (uint16_t *)malloc(chunk * sizeof(uint16_t));
  tmp[1] = (uint16_t *)malloc(chunk * sizeof(uint16_t));
  if (tmp[0] == NULL || tmp[1] == NULL) {
    if (tmp[0]) free(tmp[0]);
    if (tmp[1]) free(tmp[1]);
    _vira_gpu.clReleaseMemObject(buf);
    *out_err = VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }

  vira_cl_command_queue q =
      _vira_gpu.xfer_queue ? _vira_gpu.xfer_queue : _vira_gpu.queue;
  int cur_buf = 0;
  for (size_t off = 0; off < elems; off += chunk) {
    const size_t n = (elems - off < chunk) ? (elems - off) : chunk;
    if (!_vira_f32_to_f16_block(src + off, tmp[cur_buf], n)) {
      free(tmp[0]);
      free(tmp[1]);
      _vira_gpu.clReleaseMemObject(buf);
      *out_err = VIRA_CL_ERR_FP16_RANGE;
      return NULL;
    }
    err = _vira_gpu.clEnqueueWriteBuffer(q, buf, 1, off * sizeof(uint16_t),
                                         n * sizeof(uint16_t), tmp[cur_buf], 0,
                                         NULL, NULL);
    if (err != VIRA_CL_SUCCESS) {
      free(tmp[0]);
      free(tmp[1]);
      _vira_gpu.clReleaseMemObject(buf);
      *out_err = err;
      return NULL;
    }
    cur_buf ^= 1;
  }
  free(tmp[0]);
  free(tmp[1]);
  if (q != _vira_gpu.queue) {
    _vira_gpu.clFinish(q);
  }
  *out_err = VIRA_CL_SUCCESS;
  return buf;
}

/**
 * Looks up a tensor in the GPU weight cache, or uploads and caches it if not already present.
 * @param host_ptr Host data pointer as cache key.
 * @param elems Number of elements in the tensor.
 * @param prefer_half If true, attempts to upload as FP16 to save VRAM.
 * @param out_fmt Pointer to receive resulting weight format (FP16, Q4_0, etc.).
 * @param out_err Pointer to receive error code.
 * @return OpenCL device memory buffer, or NULL on failure.
 * Usage: Central lookup mechanism ensuring zero redundant PCIe transfers during repetitive LLM inference layers.
 */
static inline vira_cl_mem _vira_gpu_get_or_upload_weight_locked(
    const vira_f32 *host_ptr, size_t elems, bool prefer_half,
    _vira_gpu_wfmt *out_fmt, vira_cl_int *out_err) {
  uint64_t h = ((uint64_t)(uintptr_t)host_ptr >> 6) * 11400714819323198485ULL;
  int32_t start_slot = (int32_t)(h & (VIRA_GPU_WEIGHT_CACHE_CAP - 1));
  uint64_t now = ++_vira_gpu.access_clock;

  int32_t empty_slot = -1;
  int32_t oldest_probe = -1;
  uint64_t oldest_clock = 0xFFFFFFFFFFFFFFFFULL;

  for (int32_t probe = 0; probe < VIRA_GPU_WEIGHT_PROBE_LEN; ++probe) {
    int32_t idx = (start_slot + probe) & (VIRA_GPU_WEIGHT_CACHE_CAP - 1);
    _vira_gpu_cache_entry *e = &_vira_gpu.weight_cache[idx];
    if (e->host_ptr == host_ptr && e->elems == elems) {
      if (e->uncacheable) {
        *out_err = VIRA_CL_ERR_NO_ROOM;
        return NULL;
      }
      if (e->dev_buf != NULL) {
        // Format check: if caller requires FP32 (prefer_half == false), don't return half/quant
        if (!prefer_half && e->fmt != VIRA_WFMT_F32) {
          // Incompatible format cached, skip to upload as FP32
        } else {
          e->last_used = now;
          if (out_fmt) *out_fmt = e->fmt;
          *out_err = VIRA_CL_SUCCESS;
          return e->dev_buf;
        }
      }
    }
    if (empty_slot < 0 && e->host_ptr == NULL) {
      empty_slot = idx;
    }
    if (!e->uncacheable && e->last_used < oldest_clock) {
      oldest_clock = e->last_used;
      oldest_probe = idx;
    }
  }

  if (empty_slot < 0 && oldest_probe >= 0) {
    _vira_gpu_cache_entry *ev = &_vira_gpu.weight_cache[oldest_probe];
    if (ev->dev_buf != NULL) {
      _vira_gpu.clReleaseMemObject(ev->dev_buf);
      __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)ev->bytes,
                         __ATOMIC_RELAXED);
    }
    _vira_gpu_cache_entry_reset(ev);
    empty_slot = oldest_probe;
  }

  if (empty_slot < 0) {
    *out_err = VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }

  const size_t bytes =
      elems * (prefer_half ? sizeof(uint16_t) : sizeof(vira_f32));
  _vira_gpu_evict_lru_bytes_locked(bytes);

  _vira_gpu_cache_entry *slot = &_vira_gpu.weight_cache[empty_slot];
  slot->host_ptr = host_ptr;
  slot->elems = elems;
  slot->bytes = bytes;
  slot->dev_buf = NULL;
  slot->last_used = now;
  slot->fmt = prefer_half ? VIRA_WFMT_F16 : VIRA_WFMT_F32;
  slot->uncacheable = true;

  int64_t cur_alloc =
      __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
  if (_vira_gpu.vram_budget_bytes > 0 &&
      cur_alloc + (int64_t)bytes > _vira_gpu.vram_budget_bytes) {
    *out_err = VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }

  vira_cl_int err = 0;
  vira_cl_mem buf = NULL;
  if (prefer_half) {
    buf = _vira_gpu_upload_half_locked(host_ptr, elems, &err);
  } else {
    // Zero-copy host mapping on unified memory
    vira_cl_bitfield flags = VIRA_CL_MEM_READ_ONLY;
    if (_vira_gpu.unified_mem && (((uintptr_t)host_ptr & 63) == 0)) {
      flags |= VIRA_CL_MEM_USE_HOST_PTR;
    } else {
      flags |= VIRA_CL_MEM_COPY_HOST_PTR;
    }
    buf = _vira_gpu.clCreateBuffer(_vira_gpu.context, flags, bytes,
                                   (void *)host_ptr, &err);
    if (err == VIRA_CL_SUCCESS && buf == NULL) {
      err = VIRA_CL_ERR_NO_ROOM;
    }
  }
  if (err != VIRA_CL_SUCCESS || buf == NULL) {
    if (err != VIRA_CL_ERR_FP16_RANGE) {
      _vira_gpu.vram_saturated = true;
    }
    *out_err = (err != VIRA_CL_SUCCESS) ? err : VIRA_CL_ERR_NO_ROOM;
    return NULL;
  }

  slot->dev_buf = buf;
  slot->uncacheable = false;
  if (out_fmt) *out_fmt = slot->fmt;
  __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)bytes,
                     __ATOMIC_RELAXED);
  *out_err = VIRA_CL_SUCCESS;
  return buf;
}

/**
 * Case-insensitive string search helper for hardware platform/device filtering.
 * @param haystack Target string.
 * @param needle Substring to locate.
 * @return True if needle is found within haystack, false otherwise.
 * Usage: Internal helper for environment variable device matching.
 */
static inline bool _vira_strcasestr_match(const char *haystack,
                                          const char *needle) {
  if (!haystack || !needle) return false;
  size_t nlen = strlen(needle);
  if (nlen == 0) return true;
  size_t hlen = strlen(haystack);
  if (hlen < nlen) return false;
  for (size_t i = 0; i <= hlen - nlen; ++i) {
    bool match = true;
    for (size_t j = 0; j < nlen; ++j) {
      char c1 = haystack[i + j];
      char c2 = needle[j];
      if (c1 >= 'A' && c1 <= 'Z') c1 = (char)(c1 + ('a' - 'A'));
      if (c2 >= 'A' && c2 <= 'Z') c2 = (char)(c2 + ('a' - 'A'));
      if (c1 != c2) {
        match = false;
        break;
      }
    }
    if (match) return true;
  }
  return false;
}

/**
 * Dynamically resolves all OpenCL entry points, queries devices, creates contexts, command queues, and compiles embedded GEMV/GEMM kernels.
 * @param lib Dynamic library handle for OpenCL.
 * @return True if OpenCL hardware acceleration initialized successfully, false on error.
 * Usage: Invoked by `vira_gpu_init` when OpenCL library is located.
 */
static inline bool _vira_gpu_init_opencl_locked(void *lib) {
  _vira_gpu_state *g = &_vira_gpu;
  bool ok = true;
#define VIRA_GPU_LOAD(sym) ok = ok && _vira_gpu_load_sym(lib, #sym, &g->sym)
  VIRA_GPU_LOAD(clGetPlatformIDs);
  VIRA_GPU_LOAD(clGetPlatformInfo);
  VIRA_GPU_LOAD(clGetDeviceIDs);
  VIRA_GPU_LOAD(clGetDeviceInfo);
  VIRA_GPU_LOAD(clCreateContext);
  VIRA_GPU_LOAD(clReleaseContext);
  VIRA_GPU_LOAD(clCreateCommandQueue);
  VIRA_GPU_LOAD(clReleaseCommandQueue);
  VIRA_GPU_LOAD(clCreateProgramWithSource);
  VIRA_GPU_LOAD(clBuildProgram);
  VIRA_GPU_LOAD(clReleaseProgram);
  VIRA_GPU_LOAD(clCreateKernel);
  VIRA_GPU_LOAD(clReleaseKernel);
  VIRA_GPU_LOAD(clCreateBuffer);
  VIRA_GPU_LOAD(clReleaseMemObject);
  VIRA_GPU_LOAD(clEnqueueWriteBuffer);
  VIRA_GPU_LOAD(clEnqueueReadBuffer);
  VIRA_GPU_LOAD(clSetKernelArg);
  VIRA_GPU_LOAD(clEnqueueNDRangeKernel);
  VIRA_GPU_LOAD(clFinish);
#undef VIRA_GPU_LOAD
  if (!ok) {
    return false;
  }
  (void)_vira_gpu_load_sym(lib, "clGetProgramBuildInfo",
                           &g->clGetProgramBuildInfo);
  (void)_vira_gpu_load_sym(lib, "clEnqueueMapBuffer", &g->clEnqueueMapBuffer);
  (void)_vira_gpu_load_sym(lib, "clEnqueueUnmapMemObject",
                           &g->clEnqueueUnmapMemObject);
  (void)_vira_gpu_load_sym(lib, "clFlush", &g->clFlush);
  (void)_vira_gpu_load_sym(lib, "clWaitForEvents", &g->clWaitForEvents);
  (void)_vira_gpu_load_sym(lib, "clReleaseEvent", &g->clReleaseEvent);

  vira_cl_platform_id platforms[16];
  vira_cl_uint num_platforms = 0;
  if (g->clGetPlatformIDs(16, platforms, &num_platforms) != VIRA_CL_SUCCESS ||
      num_platforms == 0) {
    return false;
  }
  if (num_platforms > 16) {
    num_platforms = 16;
  }

  const char *env_plat = getenv("VIRA_GPU_PLATFORM");
  const char *env_dev = getenv("VIRA_GPU_DEVICE");
  const char *env_allow_cpu = getenv("VIRA_GPU_ALLOW_CPU_CL");
  vira_cl_bitfield dev_type = (env_allow_cpu && env_allow_cpu[0] == '1')
                                  ? VIRA_CL_DEVICE_TYPE_ALL
                                  : VIRA_CL_DEVICE_TYPE_GPU;

  vira_cl_device_id best_dev = NULL;
  int64_t best_score = -1;

  for (vira_cl_uint p = 0; p < num_platforms; ++p) {
    if (env_plat != NULL && env_plat[0] != '\0') {
      char plat_name[256] = {0};
      g->clGetPlatformInfo(platforms[p], VIRA_CL_PLATFORM_NAME,
                           sizeof(plat_name) - 1, plat_name, NULL);
      bool is_num = (env_plat[0] >= '0' && env_plat[0] <= '9');
      if (is_num) {
        if ((vira_cl_uint)atoi(env_plat) != p) continue;
      } else {
        if (!_vira_strcasestr_match(plat_name, env_plat)) continue;
      }
    }

    vira_cl_device_id devs[8];
    vira_cl_uint num_devs = 0;
    if (g->clGetDeviceIDs(platforms[p], dev_type, 8, devs, &num_devs) !=
            VIRA_CL_SUCCESS ||
        num_devs == 0) {
      continue;
    }
    if (num_devs > 8) {
      num_devs = 8;
    }

    for (vira_cl_uint d = 0; d < num_devs; ++d) {
      char cur_dname[256] = {0};
      g->clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_NAME, sizeof(cur_dname) - 1,
                         cur_dname, NULL);
      if (env_dev != NULL && env_dev[0] != '\0') {
        bool is_num = (env_dev[0] >= '0' && env_dev[0] <= '9');
        if (is_num) {
          if ((vira_cl_uint)atoi(env_dev) != d) continue;
        } else {
          if (!_vira_strcasestr_match(cur_dname, env_dev)) continue;
        }
      }

      size_t mwg = 0;
      g->clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof(mwg), &mwg, NULL);
      if (mwg < 64) {
        continue;
      }

      vira_cl_ulong gmem = 0;
      vira_cl_uint cu = 0;
      vira_cl_uint um = 0;
      g->clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem),
                         &gmem, NULL);
      g->clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu),
                         &cu, NULL);
      g->clGetDeviceInfo(devs[d], VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY,
                         sizeof(um), &um, NULL);

      int64_t score =
          (int64_t)(gmem >> 20) * (um == 0 ? 4LL : 1LL) + (int64_t)cu * 64LL;
      if (score > best_score) {
        best_score = score;
        best_dev = devs[d];
      }
    }
  }

  if (best_dev == NULL) {
    return false;
  }

  vira_cl_int err = 0;
  vira_cl_context ctx =
      g->clCreateContext(NULL, 1, &best_dev, NULL, NULL, &err);
  if (err != VIRA_CL_SUCCESS || ctx == NULL) {
    return false;
  }
  vira_cl_command_queue q = g->clCreateCommandQueue(ctx, best_dev, 0, &err);
  if (err != VIRA_CL_SUCCESS || q == NULL) {
    g->clReleaseContext(ctx);
    return false;
  }
  vira_cl_command_queue xq = g->clCreateCommandQueue(ctx, best_dev, 0, &err);
  if (err != VIRA_CL_SUCCESS || xq == NULL) {
    xq = q;
  }

  // Device geometry queries
  size_t mwg = 256;
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof(mwg), &mwg, NULL) == VIRA_CL_SUCCESS &&
      mwg > 0) {
    g->max_wg_size = mwg;
  }
  char vendor[256] = {0};
  g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_VENDOR, sizeof(vendor) - 1,
                     vendor, NULL);
  bool is_amd =
      (_vira_strcasestr_match(vendor, "AMD") ||
       _vira_strcasestr_match(vendor, "Advanced Micro Devices"));

  const char *env_warp = getenv("VIRA_GPU_WARP_SIZE");
  int32_t req_warp = (env_warp != NULL) ? atoi(env_warp) : 0;
  if (req_warp == 64 || (req_warp == 0 && is_amd && g->max_wg_size >= 128)) {
    g->warp_size = 64;
  } else {
    g->warp_size = 32;
  }

  if (g->max_wg_size >= 256) {
    g->rows_per_wg = (g->warp_size == 64) ? 4 : 8;
    g->tile_dim = 16;
  } else if (g->max_wg_size >= 128) {
    g->rows_per_wg = (g->warp_size == 64) ? 2 : 4;
    g->tile_dim = 8;
  } else {
    g->warp_size = 32;
    g->rows_per_wg = 2;
    g->tile_dim = 8;
  }
  g->wg_size = g->warp_size * g->rows_per_wg;

  size_t src_len = strlen(_vira_opencl_kernel_src);
  vira_cl_program prog = g->clCreateProgramWithSource(
      ctx, 1, &_vira_opencl_kernel_src, &src_len, &err);
  if (err != VIRA_CL_SUCCESS || prog == NULL) {
    if (xq && xq != q) g->clReleaseCommandQueue(xq);
    g->clReleaseCommandQueue(q);
    g->clReleaseContext(ctx);
    return false;
  }

  char build_opts[256];
  snprintf(build_opts, sizeof(build_opts),
           "-DWARP_SIZE=%d -DROWS_PER_WG=%d -DWG_SIZE=%d -DTILE_DIM=%d "
           "-cl-fast-relaxed-math -cl-mad-enable",
           g->warp_size, g->rows_per_wg, g->wg_size, g->tile_dim);
  if (g->clBuildProgram(prog, 1, &best_dev, build_opts, NULL, NULL) !=
      VIRA_CL_SUCCESS) {
    snprintf(build_opts, sizeof(build_opts),
             "-DWARP_SIZE=%d -DROWS_PER_WG=%d -DWG_SIZE=%d -DTILE_DIM=%d "
             "-cl-fast-relaxed-math",
             g->warp_size, g->rows_per_wg, g->wg_size, g->tile_dim);
    if (g->clBuildProgram(prog, 1, &best_dev, build_opts, NULL, NULL) !=
        VIRA_CL_SUCCESS) {
      _vira_gpu_dump_build_log(prog, best_dev);
      g->clReleaseProgram(prog);
      if (xq && xq != q) g->clReleaseCommandQueue(xq);
      g->clReleaseCommandQueue(q);
      g->clReleaseContext(ctx);
      return false;
    }
  }

  vira_cl_kernel ks[VIRA_GPU_NUM_KERNELS] = {0};
  bool required_ok = true;
  for (int i = 0; i < VIRA_GPU_NUM_KERNELS; ++i) {
    vira_cl_int ke = 0;
    ks[i] = g->clCreateKernel(prog, _vira_opencl_kernel_names[i], &ke);
    if (ke != VIRA_CL_SUCCESS) {
      if (ks[i] != NULL) {
        g->clReleaseKernel(ks[i]);
      }
      ks[i] = NULL;
    }
    if (i < VIRA_GPU_NUM_REQUIRED_KERNELS && ks[i] == NULL) {
      required_ok = false;
    }
  }
  if (!required_ok) {
    for (int i = 0; i < VIRA_GPU_NUM_KERNELS; ++i) {
      if (ks[i] != NULL) {
        g->clReleaseKernel(ks[i]);
      }
    }
    g->clReleaseProgram(prog);
    if (xq && xq != q) g->clReleaseCommandQueue(xq);
    g->clReleaseCommandQueue(q);
    g->clReleaseContext(ctx);
    return false;
  }

  g->lib_handle = lib;
  g->device = best_dev;
  g->context = ctx;
  g->queue = q;
  g->xfer_queue = xq;
  g->program = prog;

  g->kernel_gemv_q4_0 = ks[0];
  g->kernel_gemv_q4_1 = ks[1];
  g->kernel_gemv_q5_0 = ks[2];
  g->kernel_gemv_q5_1 = ks[3];
  g->kernel_gemv_q8_0 = ks[4];
  g->kernel_gemv_q4_k = ks[5];
  g->kernel_gemv_q5_k = ks[6];
  g->kernel_gemv_q6_k = ks[7];
  g->kernel_gemv_vec8 = ks[8];
  g->kernel_gemv_vec4 = ks[9];
  g->kernel_gemv_scalar = ks[10];
  g->kernel_linear_tiled_h = ks[11];
  g->kernel_linear_tiled_f32 = ks[12];
  g->kernel_matmul_tiled = ks[13];
  g->kernel_rmsnorm = ks[14];
  g->kernel_rope = ks[15];
  g->kernel_softmax = ks[16];
  g->kernel_silu = ks[17];

  g->available = true;
  snprintf(g->backend_name, sizeof(g->backend_name), "OpenCL");

  char dname[256] = {0};
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_NAME, sizeof(dname) - 1,
                         dname, NULL) == VIRA_CL_SUCCESS) {
    snprintf(g->device_name, sizeof(g->device_name), "%s", dname);
  }
  vira_cl_ulong gmem = 0;
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_GLOBAL_MEM_SIZE,
                         sizeof(gmem), &gmem, NULL) == VIRA_CL_SUCCESS) {
    g->vram_bytes = (int64_t)gmem;
    const char *env_b = getenv("VIRA_GPU_VRAM_BUDGET_MB");
    if (env_b != NULL && atoi(env_b) > 0) {
      g->vram_budget_bytes = (int64_t)atoi(env_b) * 1024LL * 1024LL;
    } else {
      g->vram_budget_bytes = (int64_t)((gmem * 88ULL) / 100ULL);
    }
  }
  vira_cl_uint cu = 0;
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_MAX_COMPUTE_UNITS,
                         sizeof(cu), &cu, NULL) == VIRA_CL_SUCCESS) {
    g->compute_units = (int32_t)cu;
  }
  vira_cl_uint um = 0;
  if (g->clGetDeviceInfo(best_dev, VIRA_CL_DEVICE_HOST_UNIFIED_MEMORY,
                         sizeof(um), &um, NULL) == VIRA_CL_SUCCESS) {
    g->unified_mem = (um != 0);
  }

  if (_vira_gpu_debug()) {
    fprintf(stderr,
            "vira gpu: %s | %lld MiB VRAM (budget %lld MiB) | %d CU | "
            "warp=%d, wg=%d, tile=%d | GGUF Q4_0..Q6_K / FP16\n",
            g->device_name, (long long)(g->vram_bytes >> 20),
            (long long)(g->vram_budget_bytes >> 20), g->compute_units,
            g->warp_size, g->wg_size, g->tile_dim);
  }
  return true;
}

/**
 * Thread-safe lazy initializer for the Vira GPU acceleration engine.
 * Usage: Probes system for discrete/integrated GPUs (NVIDIA, AMD, Apple Silicon, Intel) and initializes compute pipelines. Safe to call multiple times.
 */
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
  const char *env_lib = getenv("VIRA_OPENCL_LIB");
  if (env_lib != NULL && env_lib[0] != '\0') {
#ifdef _WIN32
    lib = (void *)LoadLibraryA(env_lib);
#else
    lib = dlopen(env_lib, RTLD_LAZY);
#endif
  }
  if (!lib) {
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
  }
  if (lib != NULL) {
    (void)_vira_gpu_init_opencl_locked(lib);
  }

  __atomic_store_n(&_vira_gpu.initialized, 2, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&_vira_gpu.lock);
}

/**
 * Releases all GPU resources, programs, kernels, command queues, and OpenCL contexts.
 * Usage: Optional clean teardown of the GPU subsystem upon application termination.
 */
static inline void vira_gpu_shutdown(void) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) != 2) {
    return;
  }
  pthread_mutex_lock(&_vira_gpu.lock);
  if (_vira_gpu.available) {
    if (_vira_gpu.queue) {
      _vira_gpu.clFinish(_vira_gpu.queue);
    }
    if (_vira_gpu.xfer_queue && _vira_gpu.xfer_queue != _vira_gpu.queue) {
      _vira_gpu.clFinish(_vira_gpu.xfer_queue);
    }
    _vira_gpu_evict_all_locked();

    vira_cl_kernel ks[VIRA_GPU_NUM_KERNELS] = {
        _vira_gpu.kernel_gemv_q4_0,        _vira_gpu.kernel_gemv_q4_1,
        _vira_gpu.kernel_gemv_q5_0,        _vira_gpu.kernel_gemv_q5_1,
        _vira_gpu.kernel_gemv_q8_0,        _vira_gpu.kernel_gemv_q4_k,
        _vira_gpu.kernel_gemv_q5_k,        _vira_gpu.kernel_gemv_q6_k,
        _vira_gpu.kernel_gemv_vec8,        _vira_gpu.kernel_gemv_vec4,
        _vira_gpu.kernel_gemv_scalar,      _vira_gpu.kernel_linear_tiled_h,
        _vira_gpu.kernel_linear_tiled_f32, _vira_gpu.kernel_matmul_tiled,
        _vira_gpu.kernel_rmsnorm,          _vira_gpu.kernel_rope,
        _vira_gpu.kernel_softmax,          _vira_gpu.kernel_silu};
    for (int i = 0; i < VIRA_GPU_NUM_KERNELS; ++i) {
      if (ks[i]) {
        _vira_gpu.clReleaseKernel(ks[i]);
      }
    }
    if (_vira_gpu.program) _vira_gpu.clReleaseProgram(_vira_gpu.program);
    if (_vira_gpu.xfer_queue && _vira_gpu.xfer_queue != _vira_gpu.queue) {
      _vira_gpu.clReleaseCommandQueue(_vira_gpu.xfer_queue);
    }
    if (_vira_gpu.queue) _vira_gpu.clReleaseCommandQueue(_vira_gpu.queue);
    if (_vira_gpu.context) _vira_gpu.clReleaseContext(_vira_gpu.context);
    _vira_gpu.available = false;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
}

/**
 * Checks if a GPU compute device is detected and operational.
 * @return True if GPU hardware acceleration is active, false if running in CPU SIMD fallback mode.
 * Usage: Used by user programs and benchmarks to inspect hardware capabilities.
 */
static inline bool vira_gpu_is_available(void) {
  vira_gpu_init();
  return _vira_gpu.available;
}

/**
 * Queries whether the active GPU device shares unified system memory (e.g. Apple Silicon M1-M4, AMD/Intel APUs).
 * @return True if unified host memory is detected, false otherwise.
 * Usage: Hardware architecture query and zero-copy optimization selection.
 */
static inline bool vira_gpu_is_unified_memory(void) {
  vira_gpu_init();
  return _vira_gpu.unified_mem;
}

/**
 * Returns the human-readable brand name of the active compute device (e.g., "NVIDIA GeForce RTX 4090" or "Apple M3 Pro").
 * @return Vira string containing device name.
 * Usage: Telemetry and system diagnostics (`gpu::device_name()`).
 */
static inline vira_string vira_gpu_device_name(void) {
  vira_gpu_init();
  return vira_str_lit(_vira_gpu.device_name,
                      (int32_t)strlen(_vira_gpu.device_name));
}

/**
 * Returns the active compute backend identifier (e.g., "OpenCL" or "CPU-SIMD").
 * @return Vira string with backend name.
 * Usage: Telemetry and runtime queries (`gpu::backend()`).
 */
static inline vira_string vira_gpu_backend_name(void) {
  vira_gpu_init();
  return vira_str_lit(_vira_gpu.backend_name,
                      (int32_t)strlen(_vira_gpu.backend_name));
}

/**
 * Returns the total capacity of GPU video memory in bytes.
 * @return Total VRAM in bytes, or 0 if GPU is unavailable.
 * Usage: Memory budgeting and system diagnostics.
 */
static inline int64_t vira_gpu_vram_bytes(void) {
  vira_gpu_init();
  return _vira_gpu.vram_bytes;
}

/**
 * Returns the current amount of GPU VRAM allocated by cached weights and scratch buffers in bytes.
 * @return Allocated VRAM in bytes.
 * Usage: Used to monitor memory usage and prevent out-of-memory faults.
 */
static inline int64_t vira_gpu_allocated_bytes(void) {
  vira_gpu_init();
  return __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
}

/**
 * Returns the cumulative count of operations that fell back to CPU execution due to VRAM exhaustion.
 * @return Number of fallback events.
 * Usage: Telemetry and profiling to verify model fit in GPU memory.
 */
static inline int64_t vira_gpu_oom_fallbacks(void) {
  vira_gpu_init();
  return __atomic_load_n(&_vira_gpu.oom_fallbacks, __ATOMIC_RELAXED);
}

/**
 * Enables or disables GPU compute acceleration dynamically at runtime.
 * @param enabled True to enable GPU dispatch, false to force CPU fallback.
 * Usage: Allows benchmarking CPU vs GPU or running under low-power modes (`gpu::set_enabled(false)`).
 */
static inline void vira_gpu_set_enabled(bool enabled) {
  vira_gpu_init();
  _vira_gpu.enabled = enabled;
}

/**
 * Queries whether GPU compute acceleration is currently enabled.
 * @return True if enabled, false otherwise.
 * Usage: Conditional execution logic based on GPU settings.
 */
static inline bool vira_gpu_is_enabled(void) {
  vira_gpu_init();
  return _vira_gpu.enabled;
}

/**
 * Testing hook to force simulated out-of-memory conditions and verify graceful CPU fallback.
 * @param enable True to simulate OOM, false for normal execution.
 * Usage: Automated unit testing of fallback resilience.
 */
static inline void vira_gpu_simulate_oom(bool enable) {
  vira_gpu_init();
  _vira_gpu.simulate_oom = enable;
}

/**
 * Blocks host execution until all queued GPU compute tasks have finished.
 * Usage: Benchmarking, profiling, and synchronization points.
 */
static inline void vira_gpu_sync(void) {
  vira_gpu_init();
  if (_vira_gpu.available && _vira_gpu.queue != NULL) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu.clFinish(_vira_gpu.queue);
    pthread_mutex_unlock(&_vira_gpu.lock);
  }
}

// ============================================================================
// GPU LINEAR PROJECTION: `y = x @ w^T` (Q4_0..Q6_K / FP16 / FP32)
// ============================================================================
/**
 * Executes hardware-accelerated linear layer projection `y = x @ w^T` using specialized Warp GEMV kernels (Q4_0..Q6_K, FP16) or 2D tiled GEMM.
 * @param opt_arena Optional memory arena for output allocation (or NULL for heap).
 * @param x Input activation tensor of shape (M, K).
 * @param w Weight matrix tensor of shape (N, K).
 * @return Output tensor of shape (M, N).
 * Usage: Core building block for Transformer attention projections and MLP feed-forward layers.
 */
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
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int64_t total_ops = (int64_t)M * (int64_t)N * (int64_t)K;
  if (!_vira_gpu.available || !_vira_gpu.enabled || total_ops < 65536LL ||
      _vira_gpu.max_wg_size < 64) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  size_t x_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  size_t y_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);
  size_t w_bytes = (size_t)N * (size_t)K * sizeof(vira_f32);

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  _vira_gpu_wfmt wfmt = VIRA_WFMT_F16;
  vira_cl_mem w_buf = NULL;

  bool w_is_activation =
      w.is_arena || _vira_gpu_is_activation_ptr_locked(w.data);
  if (w_is_activation) {
    // Dynamic activation tensor projection (e.g. Q @ K^T in attention)
    bool w_hit = false;
    w_buf = _vira_gpu_get_or_upload_activation_locked(w.data, w_bytes, &w_hit,
                                                      &err);
    wfmt = VIRA_WFMT_F32;
  } else {
    // Model parameter weight lookup from persistent weight cache
    w_buf = _vira_gpu_get_or_upload_weight_locked(w.data, (size_t)N * (size_t)K,
                                                  true, &wfmt, &err);
  }

  if (err != VIRA_CL_SUCCESS || w_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  // Look up activation tensor `x` in residency ring
  bool x_hit = false;
  vira_cl_mem x_buf = _vira_gpu_get_or_upload_activation_locked(
      x.data, x_bytes, &x_hit, &err);
  if (err != VIRA_CL_SUCCESS || x_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  // Allocate destination activation slot in VRAM
  _vira_gpu_act_entry *y_slot = _vira_gpu_alloc_activation_slot_locked(
      y_bytes, x_buf, w_buf, &err);
  if (err != VIRA_CL_SUCCESS || y_slot == NULL || y_slot->dev_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  vira_cl_mem y_buf = y_slot->dev_buf;

  // Dispatch appropriate OpenCL kernel
  if (wfmt == VIRA_WFMT_Q4_0 || wfmt == VIRA_WFMT_Q4_1 ||
      wfmt == VIRA_WFMT_Q5_0 || wfmt == VIRA_WFMT_Q5_1 ||
      wfmt == VIRA_WFMT_Q8_0) {
    vira_cl_kernel k;
    switch (wfmt) {
    case VIRA_WFMT_Q4_0: k = _vira_gpu.kernel_gemv_q4_0; break;
    case VIRA_WFMT_Q4_1: k = _vira_gpu.kernel_gemv_q4_1; break;
    case VIRA_WFMT_Q5_0: k = _vira_gpu.kernel_gemv_q5_0; break;
    case VIRA_WFMT_Q5_1: k = _vira_gpu.kernel_gemv_q5_1; break;
    default:             k = _vira_gpu.kernel_gemv_q8_0; break;
    }
    int32_t num_blocks_k = K >> 5;
    err = _vira_gpu_set_gemm_args(k, &x_buf, &w_buf, &y_buf, M, num_blocks_k, N);
    size_t l_work[2] = {(size_t)_vira_gpu.warp_size,
                        (size_t)_vira_gpu.rows_per_wg};
    size_t num_groups_n =
        ((size_t)N + _vira_gpu.rows_per_wg - 1) / _vira_gpu.rows_per_wg;
    size_t g_work[2] = {num_groups_n * _vira_gpu.warp_size,
                        (size_t)M * _vira_gpu.rows_per_wg};
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, k, 2, NULL, g_work,
                                             l_work, 0, NULL, NULL);
    }
  } else if (wfmt == VIRA_WFMT_Q4_K || wfmt == VIRA_WFMT_Q5_K ||
             wfmt == VIRA_WFMT_Q6_K) {
    vira_cl_kernel k;
    switch (wfmt) {
    case VIRA_WFMT_Q4_K: k = _vira_gpu.kernel_gemv_q4_k; break;
    case VIRA_WFMT_Q5_K: k = _vira_gpu.kernel_gemv_q5_k; break;
    default:             k = _vira_gpu.kernel_gemv_q6_k; break;
    }
    int32_t num_sb_k = K >> 8;
    err = _vira_gpu_set_gemm_args(k, &x_buf, &w_buf, &y_buf, M, num_sb_k, N);
    size_t l_work[2] = {(size_t)_vira_gpu.warp_size,
                        (size_t)_vira_gpu.rows_per_wg};
    size_t num_groups_n =
        ((size_t)N + _vira_gpu.rows_per_wg - 1) / _vira_gpu.rows_per_wg;
    size_t g_work[2] = {num_groups_n * _vira_gpu.warp_size,
                        (size_t)M * _vira_gpu.rows_per_wg};
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, k, 2, NULL, g_work,
                                             l_work, 0, NULL, NULL);
    }
  } else if (wfmt == VIRA_WFMT_F32) {
    // Dynamic FP32 tiled linear projection (activation x activation^T)
    err = _vira_gpu_set_gemm_args(_vira_gpu.kernel_linear_tiled_f32, &x_buf,
                                  &w_buf, &y_buf, M, K, N);
    size_t td = (size_t)_vira_gpu.tile_dim;
    size_t l_work[2] = {td, td};
    size_t g_work[2] = {((size_t)N + td - 1) & ~(td - 1),
                        ((size_t)M + td - 1) & ~(td - 1)};
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(
          _vira_gpu.queue, _vira_gpu.kernel_linear_tiled_f32, 2, NULL, g_work,
          l_work, 0, NULL, NULL);
    }
  } else if (M <= 4) {
    vira_cl_kernel k;
    int32_t kdim;
    if ((K & 7) == 0) {
      k = _vira_gpu.kernel_gemv_vec8;
      kdim = K >> 3;
    } else if ((K & 3) == 0) {
      k = _vira_gpu.kernel_gemv_vec4;
      kdim = K >> 2;
    } else {
      k = _vira_gpu.kernel_gemv_scalar;
      kdim = K;
    }
    err = _vira_gpu_set_gemm_args(k, &x_buf, &w_buf, &y_buf, M, kdim, N);
    size_t l_work[2] = {(size_t)_vira_gpu.warp_size,
                        (size_t)_vira_gpu.rows_per_wg};
    size_t num_groups_n =
        ((size_t)N + _vira_gpu.rows_per_wg - 1) / _vira_gpu.rows_per_wg;
    size_t g_work[2] = {num_groups_n * _vira_gpu.warp_size,
                        (size_t)M * _vira_gpu.rows_per_wg};
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, k, 2, NULL, g_work,
                                             l_work, 0, NULL, NULL);
    }
  } else {
    err = _vira_gpu_set_gemm_args(_vira_gpu.kernel_linear_tiled_h, &x_buf,
                                  &w_buf, &y_buf, M, K, N);
    size_t td = (size_t)_vira_gpu.tile_dim;
    size_t l_work[2] = {td, td};
    size_t g_work[2] = {((size_t)N + td - 1) & ~(td - 1),
                        ((size_t)M + td - 1) & ~(td - 1)};
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(
          _vira_gpu.queue, _vira_gpu.kernel_linear_tiled_h, 2, NULL, g_work,
          l_work, 0, NULL, NULL);
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

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  return out;
}

// ============================================================================
// GPU MATRIX MULTIPLICATION: `c = a @ b` (FP32, Tiled)
// ============================================================================
/**
 * Executes hardware-accelerated matrix multiplication `c = a @ b` using high-performance 2D tiled OpenCL kernels.
 * @param opt_arena Optional memory arena for output allocation (or NULL for heap).
 * @param a Left input tensor (M, K).
 * @param b Right input tensor (K, N).
 * @return Result tensor (M, N).
 * Usage: General matrix multiplication operator for neural networks and scientific computing.
 */
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
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int32_t M = a.rows;
  int32_t K = a.cols;
  int32_t N = b.cols;
  int64_t total_ops = (int64_t)M * (int64_t)K * (int64_t)N;
  if (!_vira_gpu.available || !_vira_gpu.enabled || a.n_dims > 2 ||
      b.n_dims > 2 || K != b.rows || total_ops < 65536LL ||
      _vira_gpu.max_wg_size < 64) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  size_t a_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  size_t b_bytes = (size_t)K * (size_t)N * sizeof(vira_f32);
  size_t c_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;

  bool a_hit = false, b_hit = false;
  vira_cl_mem a_buf = _vira_gpu_get_or_upload_activation_locked(
      a.data, a_bytes, &a_hit, &err);
  vira_cl_mem b_buf = (err == VIRA_CL_SUCCESS)
                          ? _vira_gpu_get_or_upload_activation_locked(
                                b.data, b_bytes, &b_hit, &err)
                          : NULL;

  _vira_gpu_act_entry *c_slot = (err == VIRA_CL_SUCCESS)
                                    ? _vira_gpu_alloc_activation_slot_locked(
                                          c_bytes, a_buf, b_buf, &err)
                                    : NULL;

  if (err != VIRA_CL_SUCCESS || a_buf == NULL || b_buf == NULL ||
      c_slot == NULL || c_slot->dev_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  vira_cl_mem c_buf = c_slot->dev_buf;

  err = _vira_gpu_set_gemm_args(_vira_gpu.kernel_matmul_tiled, &a_buf, &b_buf,
                                &c_buf, M, K, N);
  size_t td = (size_t)_vira_gpu.tile_dim;
  size_t l_work[2] = {td, td};
  size_t g_work[2] = {((size_t)N + td - 1) & ~(td - 1),
                      ((size_t)M + td - 1) & ~(td - 1)};
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_matmul_tiled, 2,
                                           NULL, g_work, l_work, 0, NULL, NULL);
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, c_buf, 1, 0, c_bytes,
                                        out.data, 0, NULL, NULL);
  }

  if (err == VIRA_CL_SUCCESS) {
    c_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  return out;
}

// ============================================================================
// GPU TRANSFORMER OPERATORS (RMSNORM, ROPE, SOFTMAX, SILU)
// ============================================================================
/**
 * Computes Root Mean Square Layer Normalization with OpenCL acceleration or SIMD fallback.
 * @param x Input activation tensor.
 * @param weight Scaling weights tensor.
 * @param eps Epsilon value for numerical stability.
 * @return Normalized tensor.
 * Usage: Pre- and post-attention layer normalization in modern LLMs (Llama, Qwen, Mistral).
 */
static inline vira_tensor_f32 vira_gpu_rmsnorm(vira_tensor_f32 x,
                                               vira_tensor_f32 weight,
                                               float eps) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&x);
  _vira_tensor_norm_shape(&weight);

  if (_vira_gpu.simulate_oom) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_rmsnorm(x, weight, eps);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int32_t total = x.rows * x.cols;
  if (!_vira_gpu.available || !_vira_gpu.enabled || total < 1024 ||
      _vira_gpu.kernel_rmsnorm == NULL) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_rmsnorm(x, weight, eps);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  size_t x_bytes = (size_t)total * sizeof(vira_f32);
  size_t w_bytes = (size_t)weight.rows * (size_t)weight.cols * sizeof(vira_f32);

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  bool x_hit = false;
  vira_cl_mem x_buf = _vira_gpu_get_or_upload_activation_locked(
      x.data, x_bytes, &x_hit, &err);

  vira_cl_mem w_buf = NULL;
  _vira_gpu_wfmt wfmt = VIRA_WFMT_F32;
  if (weight.is_arena) {
    bool w_hit = false;
    w_buf = _vira_gpu_get_or_upload_activation_locked(weight.data, w_bytes,
                                                      &w_hit, &err);
  } else {
    w_buf = _vira_gpu_get_or_upload_weight_locked(
        weight.data, (size_t)weight.rows * (size_t)weight.cols, false, &wfmt,
        &err);
  }

  _vira_gpu_act_entry *y_slot = (err == VIRA_CL_SUCCESS)
                                    ? _vira_gpu_alloc_activation_slot_locked(
                                          x_bytes, x_buf, w_buf, &err)
                                    : NULL;

  if (err != VIRA_CL_SUCCESS || x_buf == NULL || w_buf == NULL ||
      y_slot == NULL || y_slot->dev_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_rmsnorm(x, weight, eps);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  vira_cl_mem y_buf = y_slot->dev_buf;

  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 0,
                                  sizeof(vira_cl_mem), &x_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 1,
                                  sizeof(vira_cl_mem), &w_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 2,
                                  sizeof(vira_cl_mem), &y_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 3, sizeof(int32_t),
                                  &x.cols);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 4, sizeof(float),
                                  &eps);

  size_t l_work = (size_t)_vira_gpu.wg_size;
  size_t g_work = (size_t)x.rows * l_work;
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_rmsnorm, 1, NULL,
                                           &g_work, &l_work, 0, NULL, NULL);
  }

  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_buf, 1, 0, x_bytes,
                                        out.data, 0, NULL, NULL);
  }

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_rmsnorm(x, weight, eps);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  return out;
}

/**
 * Computes Rotary Position Embeddings (RoPE) with OpenCL acceleration or SIMD fallback.
 * @param q Query or key tensor.
 * @param pos Sequence token position offset.
 * @param freq_base Frequency base (e.g. 10000.0f or 1000000.0f).
 * @return Rotated tensor with position encodings applied.
 * Usage: Rotary positional encoding in modern LLM self-attention layers.
 */
static inline vira_tensor_f32 vira_gpu_rope(vira_tensor_f32 q, int64_t pos,
                                            float freq_base) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&q);

  if (_vira_gpu.simulate_oom) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_rope(q, pos, freq_base);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int32_t total = q.rows * q.cols;
  if (!_vira_gpu.available || !_vira_gpu.enabled || total < 1024 ||
      (q.cols & 1) != 0 || _vira_gpu.kernel_rope == NULL) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_rope(q, pos, freq_base);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  size_t q_bytes = (size_t)total * sizeof(vira_f32);
  int32_t half_dim = q.cols / 2;
  float pos_f = (float)pos;

  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  bool q_hit = false;
  vira_cl_mem q_buf = _vira_gpu_get_or_upload_activation_locked(
      q.data, q_bytes, &q_hit, &err);

  _vira_gpu_act_entry *y_slot = (err == VIRA_CL_SUCCESS)
                                    ? _vira_gpu_alloc_activation_slot_locked(
                                          q_bytes, q_buf, NULL, &err)
                                    : NULL;

  if (err != VIRA_CL_SUCCESS || q_buf == NULL || y_slot == NULL ||
      y_slot->dev_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_rope(q, pos, freq_base);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  vira_cl_mem y_buf = y_slot->dev_buf;

  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 0,
                                  sizeof(vira_cl_mem), &q_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 1,
                                  sizeof(vira_cl_mem), &y_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 2, sizeof(int32_t),
                                  &q.rows);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 3, sizeof(int32_t),
                                  &q.cols);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 4, sizeof(int32_t),
                                  &half_dim);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 5, sizeof(float),
                                  &pos_f);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 6, sizeof(float),
                                  &freq_base);

  size_t g_work[2] = {((size_t)half_dim + 15) & ~15ULL, (size_t)q.rows};
  size_t l_work[2] = {16, 1};
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_rope, 2, NULL,
                                           g_work, l_work, 0, NULL, NULL);
  }

  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(q.n_dims, q.dims);
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_buf, 1, 0, q_bytes,
                                        out.data, 0, NULL, NULL);
  }

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_rope(q, pos, freq_base);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  return out;
}

/**
 * Computes numerically stable Softmax probability distribution across tensor rows with OpenCL acceleration or SIMD fallback.
 * @param logits Input logits tensor.
 * @return Softmax probabilities tensor summing to 1.0 along trailing dimension.
 * Usage: Self-attention score normalization and next-token probability distribution.
 */
static inline vira_tensor_f32 vira_gpu_softmax(vira_tensor_f32 logits) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&logits);

  if (_vira_gpu.simulate_oom) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_softmax(logits);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int32_t total = logits.rows * logits.cols;
  if (!_vira_gpu.available || !_vira_gpu.enabled || total < 1024 ||
      _vira_gpu.kernel_softmax == NULL) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_softmax(logits);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  size_t bytes = (size_t)total * sizeof(vira_f32);
  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  bool in_hit = false;
  vira_cl_mem in_buf = _vira_gpu_get_or_upload_activation_locked(
      logits.data, bytes, &in_hit, &err);

  _vira_gpu_act_entry *y_slot = (err == VIRA_CL_SUCCESS)
                                    ? _vira_gpu_alloc_activation_slot_locked(
                                          bytes, in_buf, NULL, &err)
                                    : NULL;

  if (err != VIRA_CL_SUCCESS || in_buf == NULL || y_slot == NULL ||
      y_slot->dev_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_softmax(logits);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  vira_cl_mem y_buf = y_slot->dev_buf;

  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_softmax, 0,
                                  sizeof(vira_cl_mem), &in_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_softmax, 1,
                                  sizeof(vira_cl_mem), &y_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_softmax, 2, sizeof(int32_t),
                                  &logits.rows);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_softmax, 3, sizeof(int32_t),
                                  &logits.cols);

  size_t l_work = (size_t)_vira_gpu.wg_size;
  size_t g_work = (size_t)logits.rows * l_work;
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_softmax, 1, NULL,
                                           &g_work, &l_work, 0, NULL, NULL);
  }

  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(logits.n_dims, logits.dims);
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_buf, 1, 0, bytes,
                                        out.data, 0, NULL, NULL);
  }

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_softmax(logits);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  return out;
}

/**
 * Computes the SiLU (Swish) non-linear activation function `f(x) = x * sigmoid(x)` with OpenCL acceleration or SIMD fallback.
 * @param x Input tensor.
 * @return Activated tensor.
 * Usage: Feed-forward network gate activation in modern Transformer blocks.
 */
static inline vira_tensor_f32 vira_gpu_silu(vira_tensor_f32 x) {
  vira_gpu_init();
  _vira_tensor_norm_shape(&x);

  if (_vira_gpu.simulate_oom) {
    pthread_mutex_lock(&_vira_gpu.lock);
    _vira_gpu_evict_all_locked();
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_silu(x);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int32_t total = x.rows * x.cols;
  if (!_vira_gpu.available || !_vira_gpu.enabled || total < 1024 ||
      _vira_gpu.kernel_silu == NULL) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_silu(x);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  size_t bytes = (size_t)total * sizeof(vira_f32);
  pthread_mutex_lock(&_vira_gpu.lock);
  vira_cl_int err = 0;
  bool in_hit = false;
  vira_cl_mem in_buf = _vira_gpu_get_or_upload_activation_locked(
      x.data, bytes, &in_hit, &err);

  _vira_gpu_act_entry *y_slot = (err == VIRA_CL_SUCCESS)
                                    ? _vira_gpu_alloc_activation_slot_locked(
                                          bytes, in_buf, NULL, &err)
                                    : NULL;

  if (err != VIRA_CL_SUCCESS || in_buf == NULL || y_slot == NULL ||
      y_slot->dev_buf == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_silu(x);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  vira_cl_mem y_buf = y_slot->dev_buf;

  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_silu, 0,
                                  sizeof(vira_cl_mem), &in_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_silu, 1,
                                  sizeof(vira_cl_mem), &y_buf);
  err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_silu, 2, sizeof(int32_t),
                                  &total);

  size_t l_work = (size_t)_vira_gpu.wg_size;
  size_t g_work = ((size_t)total + l_work - 1) & ~(l_work - 1);
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue,
                                           _vira_gpu.kernel_silu, 1, NULL,
                                           &g_work, &l_work, 0, NULL, NULL);
  }

  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  if (err == VIRA_CL_SUCCESS) {
    err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_buf, 1, 0, bytes,
                                        out.data, 0, NULL, NULL);
  }

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_silu(x);
    _vira_tls_gpu_fallback = false;
    return res;
  }
  return out;
}

#endif // VIRA_GPU_H
