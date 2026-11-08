#ifndef VIRA_GPU_H
#define VIRA_GPU_H

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "vira_rt.h"
#include "vira_cuda_ptx.h"
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


// ============================================================================
// ZERO-SDK NVIDIA CUDA DRIVER API DYNAMIC TYPEDEFS & SYMBOLS
// ============================================================================
typedef int vira_cu_result;
typedef int vira_cu_device;
typedef void *vira_cu_context;
typedef void *vira_cu_module;
typedef void *vira_cu_function;
typedef unsigned long long vira_cu_deviceptr;
typedef void *vira_cu_stream;

#define VIRA_CUDA_SUCCESS 0
#define VIRA_CUDA_ERROR_OUT_OF_MEMORY 2
#define VIRA_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define VIRA_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define VIRA_CU_DEVICE_ATTRIBUTE_WARP_SIZE 10
#define VIRA_CU_DEVICE_ATTRIBUTE_INTEGRATED 18
#define VIRA_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define VIRA_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

typedef vira_cu_result(VIRA_CL_API *pfn_cuInit)(unsigned int);
typedef vira_cu_result(VIRA_CL_API *pfn_cuDriverGetVersion)(int *);
typedef vira_cu_result(VIRA_CL_API *pfn_cuDeviceGetCount)(int *);
typedef vira_cu_result(VIRA_CL_API *pfn_cuDeviceGet)(vira_cu_device *, int);
typedef vira_cu_result(VIRA_CL_API *pfn_cuDeviceGetName)(char *, int, vira_cu_device);
typedef vira_cu_result(VIRA_CL_API *pfn_cuDeviceTotalMem_v2)(size_t *, vira_cu_device);
typedef vira_cu_result(VIRA_CL_API *pfn_cuDeviceGetAttribute)(int *, int, vira_cu_device);
typedef vira_cu_result(VIRA_CL_API *pfn_cuCtxCreate_v2)(vira_cu_context *, unsigned int, vira_cu_device);
typedef vira_cu_result(VIRA_CL_API *pfn_cuCtxDestroy_v2)(vira_cu_context);
typedef vira_cu_result(VIRA_CL_API *pfn_cuCtxSetCurrent)(vira_cu_context);
typedef vira_cu_result(VIRA_CL_API *pfn_cuCtxSynchronize)(void);
typedef vira_cu_result(VIRA_CL_API *pfn_cuModuleLoadData)(vira_cu_module *, const void *);
typedef vira_cu_result(VIRA_CL_API *pfn_cuModuleUnload)(vira_cu_module);
typedef vira_cu_result(VIRA_CL_API *pfn_cuModuleGetFunction)(vira_cu_function *, vira_cu_module, const char *);
typedef vira_cu_result(VIRA_CL_API *pfn_cuMemAlloc_v2)(vira_cu_deviceptr *, size_t);
typedef vira_cu_result(VIRA_CL_API *pfn_cuMemFree_v2)(vira_cu_deviceptr);
typedef vira_cu_result(VIRA_CL_API *pfn_cuMemcpyHtoD_v2)(vira_cu_deviceptr, const void *, size_t);
typedef vira_cu_result(VIRA_CL_API *pfn_cuMemcpyDtoH_v2)(void *, vira_cu_deviceptr, size_t);
typedef vira_cu_result(VIRA_CL_API *pfn_cuMemcpyHtoDAsync_v2)(vira_cu_deviceptr, const void *, size_t, vira_cu_stream);
typedef vira_cu_result(VIRA_CL_API *pfn_cuMemcpyDtoHAsync_v2)(void *, vira_cu_deviceptr, size_t, vira_cu_stream);
typedef vira_cu_result(VIRA_CL_API *pfn_cuStreamCreate)(vira_cu_stream *, unsigned int);
typedef vira_cu_result(VIRA_CL_API *pfn_cuStreamDestroy_v2)(vira_cu_stream);
typedef vira_cu_result(VIRA_CL_API *pfn_cuStreamSynchronize)(vira_cu_stream);
typedef vira_cu_result(VIRA_CL_API *pfn_cuLaunchKernel)(vira_cu_function,
                                                        unsigned int, unsigned int, unsigned int,
                                                        unsigned int, unsigned int, unsigned int,
                                                        unsigned int, vira_cu_stream,
                                                        void **, void **);

typedef enum {
  VIRA_GPU_BACKEND_CPU = 0,
  VIRA_GPU_BACKEND_OPENCL = 1,
  VIRA_GPU_BACKEND_CUDA = 2,
} _vira_gpu_backend_type;

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
  vira_cu_deviceptr cu_buf;
  uint64_t last_used;
  _vira_gpu_wfmt fmt;
  bool uncacheable;
} _vira_gpu_cache_entry;

typedef struct {
  const vira_f32 *host_ptr;
  size_t bytes;
  uint64_t epoch;
  vira_cl_mem dev_buf;
  vira_cu_deviceptr cu_buf;
  size_t buf_cap;
  uint64_t last_used;
} _vira_gpu_act_entry;

typedef struct {
  volatile int32_t initialized;
  bool available;
  bool enabled;
  bool unified_mem;
  bool has_tensor_cores;
  int compute_capability_major;
  int compute_capability_minor;
  bool vram_saturated;
  volatile bool simulate_oom;
  _vira_gpu_backend_type active_backend;
  char requested_backend[64];
  char requested_device[256];
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

  // OpenCL Driver API & handles
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

  // OpenCL Compute kernels
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

  // NVIDIA CUDA Driver API & handles
  void *cuda_lib_handle;
  pfn_cuInit cuInit;
  pfn_cuDriverGetVersion cuDriverGetVersion;
  pfn_cuDeviceGetCount cuDeviceGetCount;
  pfn_cuDeviceGet cuDeviceGet;
  pfn_cuDeviceGetName cuDeviceGetName;
  pfn_cuDeviceTotalMem_v2 cuDeviceTotalMem;
  pfn_cuDeviceGetAttribute cuDeviceGetAttribute;
  pfn_cuCtxCreate_v2 cuCtxCreate;
  pfn_cuCtxDestroy_v2 cuCtxDestroy;
  pfn_cuCtxSetCurrent cuCtxSetCurrent;
  pfn_cuCtxSynchronize cuCtxSynchronize;
  pfn_cuModuleLoadData cuModuleLoadData;
  pfn_cuModuleUnload cuModuleUnload;
  pfn_cuModuleGetFunction cuModuleGetFunction;
  pfn_cuMemAlloc_v2 cuMemAlloc;
  pfn_cuMemFree_v2 cuMemFree;
  pfn_cuMemcpyHtoD_v2 cuMemcpyHtoD;
  pfn_cuMemcpyDtoH_v2 cuMemcpyDtoH;
  pfn_cuMemcpyHtoDAsync_v2 cuMemcpyHtoDAsync;
  pfn_cuMemcpyDtoHAsync_v2 cuMemcpyDtoHAsync;
  pfn_cuStreamCreate cuStreamCreate;
  pfn_cuStreamDestroy_v2 cuStreamDestroy;
  pfn_cuStreamSynchronize cuStreamSynchronize;
  pfn_cuLaunchKernel cuLaunchKernel;

  vira_cu_device cu_device;
  vira_cu_context cu_context;
  vira_cu_stream cu_stream;
  vira_cu_module cu_module;
  vira_cu_module cu_wmma_module;

  // CUDA Compute kernels
  vira_cu_function cu_k_gemv_q4_0;
  vira_cu_function cu_k_gemv_q4_1;
  vira_cu_function cu_k_gemv_q5_0;
  vira_cu_function cu_k_gemv_q5_1;
  vira_cu_function cu_k_gemv_q8_0;
  vira_cu_function cu_k_gemv_q4_k;
  vira_cu_function cu_k_gemv_q5_k;
  vira_cu_function cu_k_gemv_q6_k;
  vira_cu_function cu_k_gemv_vec8;
  vira_cu_function cu_k_gemv_vec4;
  vira_cu_function cu_k_gemv_scalar;
  vira_cu_function cu_k_linear_tiled_h;
  vira_cu_function cu_k_linear_tiled_f32;
  vira_cu_function cu_k_wmma_linear_h;
  vira_cu_function cu_k_matmul_tiled;
  vira_cu_function cu_k_wmma_matmul;
  vira_cu_function cu_k_rmsnorm;
  vira_cu_function cu_k_rope;
  vira_cu_function cu_k_softmax;
  vira_cu_function cu_k_silu;

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
    .has_tensor_cores = false,
    .compute_capability_major = 0,
    .compute_capability_minor = 0,
    .vram_saturated = false,
    .simulate_oom = false,
    .active_backend = VIRA_GPU_BACKEND_CPU,
    .requested_backend = {0},
    .requested_device = {0},
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
    .cuda_lib_handle = NULL,
};

static inline void vira_gpu_init(void);
static inline void vira_gpu_shutdown(void);
static inline void _vira_gpu_evict_all_locked(void);
static inline void _vira_gpu_evict_ptr_locked(const vira_f32 *host_ptr);
static inline void _vira_gpu_evict_lru_bytes_locked(size_t needed_bytes);
static inline void _vira_gpu_invalidate_activations(void);
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

static inline bool _vira_gpu_debug(void) {
  const char *env = getenv("VIRA_GPU_DEBUG");
  return env != NULL && env[0] != '\0' && env[0] != '0';
}

static inline uint16_t _vira_f32_bits_to_f16(uint32_t x) {
  uint32_t sign = (x >> 31) & 1u;
  uint32_t exp = (x >> 23) & 0xFFu;
  uint32_t mant = x & 0x7FFFFFu;

  if (exp == 255) {
    if (mant != 0) {
      return (uint16_t)((sign << 15) | 0x7E00u);
    } else {
      return (uint16_t)((sign << 15) | 0x7C00u);
    }
  }

  int32_t new_exp = (int32_t)exp - 127 + 15;
  if (new_exp >= 31) {
    return (uint16_t)((sign << 15) | 0x7C00u);
  }
  if (new_exp <= 0) {
    if (new_exp < -10) {
      return (uint16_t)(sign << 15);
    }
    mant = mant | 0x800000u;
    uint32_t shift = (uint32_t)(1 - new_exp + 13);
    uint32_t half_mant = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) {
      half_mant += 1;
    }
    return (uint16_t)((sign << 15) | half_mant);
  }

  uint32_t half_mant = mant >> 13;
  if ((mant >> 12) & 1u) {
    half_mant += 1;
    if (half_mant >= 0x0400u) {
      half_mant = 0;
      new_exp += 1;
      if (new_exp >= 31) {
        return (uint16_t)((sign << 15) | 0x7C00u);
      }
    }
  }

  return (uint16_t)((sign << 15) | ((uint32_t)new_exp << 10) | half_mant);
}

static inline bool _vira_f32_to_f16_block(const vira_f32 *src, uint16_t *dst,
                                         size_t n) {
  for (size_t i = 0; i < n; ++i) {
    union {
      float f;
      uint32_t u;
    } conv;
    conv.f = (float)src[i];
    dst[i] = _vira_f32_bits_to_f16(conv.u);
  }
  return true;
}

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

static inline void _vira_gpu_free_weight_buffer_locked(_vira_gpu_cache_entry *e) {
  if (e->dev_buf != NULL && _vira_gpu.clReleaseMemObject) {
    _vira_gpu.clReleaseMemObject(e->dev_buf);
    e->dev_buf = NULL;
  }
  if (e->cu_buf != 0 && _vira_gpu.cuMemFree) {
    _vira_gpu.cuMemFree(e->cu_buf);
    e->cu_buf = 0;
  }
}

static inline void _vira_gpu_free_act_buffer_locked(_vira_gpu_act_entry *a) {
  if (a->dev_buf != NULL && _vira_gpu.clReleaseMemObject) {
    _vira_gpu.clReleaseMemObject(a->dev_buf);
    a->dev_buf = NULL;
  }
  if (a->cu_buf != 0 && _vira_gpu.cuMemFree) {
    _vira_gpu.cuMemFree(a->cu_buf);
    a->cu_buf = 0;
  }
}

static inline void _vira_gpu_cache_entry_reset(_vira_gpu_cache_entry *e) {
  e->host_ptr = NULL;
  e->elems = 0;
  e->bytes = 0;
  e->dev_buf = NULL;
  e->cu_buf = 0;
  e->last_used = 0;
  e->fmt = VIRA_WFMT_F32;
  e->uncacheable = false;
}

static inline void _vira_gpu_act_entry_reset(_vira_gpu_act_entry *e) {
  e->host_ptr = NULL;
  e->bytes = 0;
  e->epoch = 0;
  e->last_used = 0;
}

static inline void _vira_gpu_invalidate_activations(void) {
  __atomic_add_fetch(&_vira_gpu.act_epoch, 1, __ATOMIC_RELEASE);
}

static inline bool _vira_gpu_is_activation_ptr_locked(const vira_f32 *ptr) {
  if (ptr == NULL) return false;
  uint64_t ep = __atomic_load_n(&_vira_gpu.act_epoch, __ATOMIC_ACQUIRE);
  for (int i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    bool has_buf = (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA)
                       ? (_vira_gpu.act_ring[i].cu_buf != 0)
                       : (_vira_gpu.act_ring[i].dev_buf != NULL);
    if (has_buf && _vira_gpu.act_ring[i].epoch == ep &&
        _vira_gpu.act_ring[i].host_ptr == ptr) {
      return true;
    }
  }
  return false;
}

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
      bool has_buf = (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA)
                         ? (e->cu_buf != 0)
                         : (e->dev_buf != NULL);
      if (has_buf && !e->uncacheable) {
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
    _vira_gpu_free_weight_buffer_locked(slot);
    __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)slot->bytes,
                       __ATOMIC_RELAXED);
    _vira_gpu_cache_entry_reset(slot);
    cur_alloc = __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
  }
}

static inline _vira_gpu_act_entry *_vira_gpu_get_or_upload_activation_locked(
    const vira_f32 *ptr, size_t bytes, bool *out_hit, vira_cl_int *out_err) {
  uint64_t ep = __atomic_load_n(&_vira_gpu.act_epoch, __ATOMIC_ACQUIRE);
  uint64_t now = ++_vira_gpu.access_clock;

  // 1. Check for residency hit
  for (int i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    _vira_gpu_act_entry *e = &_vira_gpu.act_ring[i];
    bool has_buf = (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA)
                       ? (e->cu_buf != 0)
                       : (e->dev_buf != NULL);
    if (has_buf && e->epoch == ep && e->host_ptr == ptr && e->bytes == bytes) {
      e->last_used = now;
      if (out_hit) *out_hit = true;
      *out_err = VIRA_CL_SUCCESS;
      return e;
    }
  }

  // 2. Cache miss: find an empty, expired, or LRU slot
  if (out_hit) *out_hit = false;
  int best_slot = -1;
  uint64_t oldest_clock = 0xFFFFFFFFFFFFFFFFULL;
  for (int i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    _vira_gpu_act_entry *e = &_vira_gpu.act_ring[i];
    bool has_buf = (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA)
                       ? (e->cu_buf != 0)
                       : (e->dev_buf != NULL);
    if (!has_buf || e->epoch != ep) {
      best_slot = i;
      break;
    }
    if (e->last_used < oldest_clock) {
      oldest_clock = e->last_used;
      best_slot = i;
    }
  }

  _vira_gpu_act_entry *slot = &_vira_gpu.act_ring[best_slot];
  size_t new_cap = (bytes < (1024ULL * 1024ULL))
                       ? (1024ULL * 1024ULL)
                       : ((bytes + 65535ULL) & ~65535ULL);

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    if (slot->cu_buf == 0 || slot->buf_cap < bytes) {
      if (slot->cu_buf != 0) {
        _vira_gpu.cuMemFree(slot->cu_buf);
        __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)slot->buf_cap,
                           __ATOMIC_RELAXED);
        slot->cu_buf = 0;
        slot->buf_cap = 0;
      }
      vira_cu_deviceptr nb = 0;
      if (_vira_gpu.cuMemAlloc(&nb, new_cap) != VIRA_CUDA_SUCCESS || nb == 0) {
        *out_err = VIRA_CL_ERR_NO_ROOM;
        return NULL;
      }
      slot->cu_buf = nb;
      slot->buf_cap = new_cap;
      __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)new_cap,
                         __ATOMIC_RELAXED);
    }
    if (_vira_gpu.cuMemcpyHtoD(slot->cu_buf, ptr, bytes) != VIRA_CUDA_SUCCESS) {
      *out_err = VIRA_CL_ERR_NO_ROOM;
      return NULL;
    }
  } else {
    if (slot->dev_buf == NULL || slot->buf_cap < bytes) {
      if (slot->dev_buf != NULL) {
        _vira_gpu.clReleaseMemObject(slot->dev_buf);
        __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)slot->buf_cap,
                           __ATOMIC_RELAXED);
        slot->dev_buf = NULL;
        slot->buf_cap = 0;
      }
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
  }

  slot->host_ptr = ptr;
  slot->bytes = bytes;
  slot->epoch = ep;
  slot->last_used = now;
  *out_err = VIRA_CL_SUCCESS;
  return slot;
}

static inline _vira_gpu_act_entry *_vira_gpu_alloc_activation_slot_locked(
    size_t needed_bytes, vira_cl_mem avoid1_cl, vira_cl_mem avoid2_cl,
    vira_cu_deviceptr avoid1_cu, vira_cu_deviceptr avoid2_cu,
    vira_cl_int *out_err) {
  uint64_t ep = __atomic_load_n(&_vira_gpu.act_epoch, __ATOMIC_ACQUIRE);
  uint64_t now = ++_vira_gpu.access_clock;

  int best_slot = -1;
  uint64_t oldest_clock = 0xFFFFFFFFFFFFFFFFULL;
  for (int i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    _vira_gpu_act_entry *e = &_vira_gpu.act_ring[i];
    if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
      if (e->cu_buf != 0 && (e->cu_buf == avoid1_cu || e->cu_buf == avoid2_cu)) {
        continue;
      }
      if (e->cu_buf == 0 || e->epoch != ep) {
        best_slot = i;
        break;
      }
    } else {
      if (e->dev_buf != NULL && (e->dev_buf == avoid1_cl || e->dev_buf == avoid2_cl)) {
        continue;
      }
      if (e->dev_buf == NULL || e->epoch != ep) {
        best_slot = i;
        break;
      }
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
  size_t new_cap = (needed_bytes < (1024ULL * 1024ULL))
                       ? (1024ULL * 1024ULL)
                       : ((needed_bytes + 65535ULL) & ~65535ULL);

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    if (slot->cu_buf == 0 || slot->buf_cap < needed_bytes) {
      if (slot->cu_buf != 0) {
        _vira_gpu.cuMemFree(slot->cu_buf);
        __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)slot->buf_cap,
                           __ATOMIC_RELAXED);
        slot->cu_buf = 0;
        slot->buf_cap = 0;
      }
      vira_cu_deviceptr nb = 0;
      if (_vira_gpu.cuMemAlloc(&nb, new_cap) != VIRA_CUDA_SUCCESS || nb == 0) {
        *out_err = VIRA_CL_ERR_NO_ROOM;
        return NULL;
      }
      slot->cu_buf = nb;
      slot->buf_cap = new_cap;
      __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)new_cap,
                         __ATOMIC_RELAXED);
    }
  } else {
    if (slot->dev_buf == NULL || slot->buf_cap < needed_bytes) {
      if (slot->dev_buf != NULL) {
        _vira_gpu.clReleaseMemObject(slot->dev_buf);
        __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)slot->buf_cap,
                           __ATOMIC_RELAXED);
        slot->dev_buf = NULL;
        slot->buf_cap = 0;
      }
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
  }

  slot->host_ptr = NULL;
  slot->bytes = needed_bytes;
  slot->epoch = ep;
  slot->last_used = now;
  *out_err = VIRA_CL_SUCCESS;
  return slot;
}

static inline void _vira_gpu_evict_ptr_locked(const vira_f32 *host_ptr) {
  if (!_vira_gpu.available || host_ptr == NULL) {
    return;
  }
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
      _vira_gpu_free_weight_buffer_locked(e);
      __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)e->bytes,
                         __ATOMIC_RELAXED);
      _vira_gpu_cache_entry_reset(e);
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
    _vira_gpu_free_weight_buffer_locked(&_vira_gpu.weight_cache[i]);
    _vira_gpu_cache_entry_reset(&_vira_gpu.weight_cache[i]);
  }
  for (int32_t i = 0; i < VIRA_GPU_ACT_RING_CAP; ++i) {
    _vira_gpu_free_act_buffer_locked(&_vira_gpu.act_ring[i]);
    _vira_gpu.act_ring[i].buf_cap = 0;
    _vira_gpu_act_entry_reset(&_vira_gpu.act_ring[i]);
  }
  _vira_gpu.vram_saturated = false;
  __atomic_store_n(&_vira_gpu.allocated_bytes, 0, __ATOMIC_RELAXED);
  _vira_gpu_invalidate_activations();
}

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
    _vira_gpu_free_weight_buffer_locked(ev);
    __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)ev->bytes,
                       __ATOMIC_RELAXED);
    _vira_gpu_cache_entry_reset(ev);
    empty_slot = oldest_probe;
  }

  if (empty_slot < 0) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    return;
  }

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    vira_cu_deviceptr cu_b = 0;
    if (_vira_gpu.cuMemAlloc(&cu_b, raw_bytes) == VIRA_CUDA_SUCCESS && cu_b != 0) {
      if (_vira_gpu.cuMemcpyHtoD(cu_b, raw_blocks, raw_bytes) == VIRA_CUDA_SUCCESS) {
        _vira_gpu_cache_entry *slot = &_vira_gpu.weight_cache[empty_slot];
        slot->host_ptr = host_ptr;
        slot->elems = elems;
        slot->bytes = raw_bytes;
        slot->cu_buf = cu_b;
        slot->dev_buf = NULL;
        slot->last_used = ++_vira_gpu.access_clock;
        slot->fmt = fmt;
        slot->uncacheable = false;
        __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)raw_bytes,
                           __ATOMIC_RELAXED);
      } else {
        _vira_gpu.cuMemFree(cu_b);
      }
    }
  } else {
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
      slot->cu_buf = 0;
      slot->last_used = ++_vira_gpu.access_clock;
      slot->fmt = fmt;
      slot->uncacheable = false;
      __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)raw_bytes,
                         __ATOMIC_RELAXED);
    }
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline _vira_gpu_cache_entry *_vira_gpu_get_or_upload_weight_locked(
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
      bool has_buf = (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA)
                         ? (e->cu_buf != 0)
                         : (e->dev_buf != NULL);
      if (has_buf) {
        if (!prefer_half && e->fmt != VIRA_WFMT_F32) {
          // Format incompatible, skip
        } else {
          e->last_used = now;
          if (out_fmt) *out_fmt = e->fmt;
          *out_err = VIRA_CL_SUCCESS;
          return e;
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
    _vira_gpu_free_weight_buffer_locked(ev);
    __atomic_sub_fetch(&_vira_gpu.allocated_bytes, (int64_t)ev->bytes,
                       __ATOMIC_RELAXED);
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
  slot->cu_buf = 0;
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

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    vira_cu_deviceptr cu_b = 0;
    if (_vira_gpu.cuMemAlloc(&cu_b, bytes) != VIRA_CUDA_SUCCESS || cu_b == 0) {
      _vira_gpu.vram_saturated = true;
      *out_err = VIRA_CL_ERR_NO_ROOM;
      return NULL;
    }
    if (prefer_half) {
      uint16_t *tmp = (uint16_t *)malloc(bytes);
      if (tmp == NULL || !_vira_f32_to_f16_block(host_ptr, tmp, elems) ||
          _vira_gpu.cuMemcpyHtoD(cu_b, tmp, bytes) != VIRA_CUDA_SUCCESS) {
        if (tmp) free(tmp);
        _vira_gpu.cuMemFree(cu_b);
        *out_err = VIRA_CL_ERR_FP16_RANGE;
        return NULL;
      }
      free(tmp);
    } else {
      if (_vira_gpu.cuMemcpyHtoD(cu_b, host_ptr, bytes) != VIRA_CUDA_SUCCESS) {
        _vira_gpu.cuMemFree(cu_b);
        *out_err = VIRA_CL_ERR_NO_ROOM;
        return NULL;
      }
    }
    slot->cu_buf = cu_b;
  } else {
    vira_cl_int err = 0;
    vira_cl_mem buf = NULL;
    if (prefer_half) {
      const size_t chunk = elems < VIRA_GPU_UPLOAD_CHUNK_ELEMS ? elems : VIRA_GPU_UPLOAD_CHUNK_ELEMS;
      uint16_t *tmp[2];
      tmp[0] = (uint16_t *)malloc(chunk * sizeof(uint16_t));
      tmp[1] = (uint16_t *)malloc(chunk * sizeof(uint16_t));
      buf = _vira_gpu.clCreateBuffer(_vira_gpu.context, VIRA_CL_MEM_READ_ONLY, bytes, NULL, &err);
      if (err == VIRA_CL_SUCCESS && buf && tmp[0] && tmp[1]) {
        vira_cl_command_queue q = _vira_gpu.xfer_queue ? _vira_gpu.xfer_queue : _vira_gpu.queue;
        int cur_buf = 0;
        for (size_t off = 0; off < elems; off += chunk) {
          const size_t n = (elems - off < chunk) ? (elems - off) : chunk;
          if (!_vira_f32_to_f16_block(host_ptr + off, tmp[cur_buf], n)) {
            err = VIRA_CL_ERR_FP16_RANGE;
            break;
          }
          err = _vira_gpu.clEnqueueWriteBuffer(q, buf, 1, off * sizeof(uint16_t),
                                               n * sizeof(uint16_t), tmp[cur_buf], 0, NULL, NULL);
          if (err != VIRA_CL_SUCCESS) break;
          cur_buf ^= 1;
        }
      }
      if (tmp[0]) free(tmp[0]);
      if (tmp[1]) free(tmp[1]);
    } else {
      vira_cl_bitfield flags = VIRA_CL_MEM_READ_ONLY;
      if (_vira_gpu.unified_mem && (((uintptr_t)host_ptr & 63) == 0)) {
        flags |= VIRA_CL_MEM_USE_HOST_PTR;
      } else {
        flags |= VIRA_CL_MEM_COPY_HOST_PTR;
      }
      buf = _vira_gpu.clCreateBuffer(_vira_gpu.context, flags, bytes, (void *)host_ptr, &err);
    }
    if (err != VIRA_CL_SUCCESS || buf == NULL) {
      _vira_gpu.vram_saturated = true;
      *out_err = (err != VIRA_CL_SUCCESS) ? err : VIRA_CL_ERR_NO_ROOM;
      return NULL;
    }
    slot->dev_buf = buf;
  }

  slot->uncacheable = false;
  if (out_fmt) *out_fmt = slot->fmt;
  __atomic_add_fetch(&_vira_gpu.allocated_bytes, (int64_t)bytes,
                     __ATOMIC_RELAXED);
  *out_err = VIRA_CL_SUCCESS;
  return slot;
}

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

static inline bool _vira_gpu_load_sym(void *lib, const char *name,
                                      void **out_sym) {
#ifdef _WIN32
  *out_sym = (void *)GetProcAddress((HMODULE)lib, name);
#else
  *out_sym = dlsym(lib, name);
#endif
  return *out_sym != NULL;
}

static inline bool _vira_gpu_init_cuda_locked(void *lib) {
  _vira_gpu_state *g = &_vira_gpu;
  bool ok = true;
#define VIRA_CU_LOAD(sym) ok = ok && _vira_gpu_load_sym(lib, #sym, (void **)&g->sym)
  VIRA_CU_LOAD(cuInit);
  VIRA_CU_LOAD(cuDriverGetVersion);
  VIRA_CU_LOAD(cuDeviceGetCount);
  VIRA_CU_LOAD(cuDeviceGet);
  VIRA_CU_LOAD(cuDeviceGetName);
  VIRA_CU_LOAD(cuDeviceGetAttribute);
#undef VIRA_CU_LOAD
  if (!ok) return false;

  (void)_vira_gpu_load_sym(lib, "cuDeviceTotalMem_v2", (void **)&g->cuDeviceTotalMem);
  if (!g->cuDeviceTotalMem) {
    (void)_vira_gpu_load_sym(lib, "cuDeviceTotalMem", (void **)&g->cuDeviceTotalMem);
  }
  (void)_vira_gpu_load_sym(lib, "cuCtxCreate_v2", (void **)&g->cuCtxCreate);
  if (!g->cuCtxCreate) {
    (void)_vira_gpu_load_sym(lib, "cuCtxCreate", (void **)&g->cuCtxCreate);
  }
  (void)_vira_gpu_load_sym(lib, "cuCtxDestroy_v2", (void **)&g->cuCtxDestroy);
  if (!g->cuCtxDestroy) {
    (void)_vira_gpu_load_sym(lib, "cuCtxDestroy", (void **)&g->cuCtxDestroy);
  }
  (void)_vira_gpu_load_sym(lib, "cuCtxSetCurrent", (void **)&g->cuCtxSetCurrent);
  (void)_vira_gpu_load_sym(lib, "cuCtxSynchronize", (void **)&g->cuCtxSynchronize);
  (void)_vira_gpu_load_sym(lib, "cuModuleLoadData", (void **)&g->cuModuleLoadData);
  (void)_vira_gpu_load_sym(lib, "cuModuleUnload", (void **)&g->cuModuleUnload);
  (void)_vira_gpu_load_sym(lib, "cuModuleGetFunction", (void **)&g->cuModuleGetFunction);
  (void)_vira_gpu_load_sym(lib, "cuMemAlloc_v2", (void **)&g->cuMemAlloc);
  if (!g->cuMemAlloc) {
    (void)_vira_gpu_load_sym(lib, "cuMemAlloc", (void **)&g->cuMemAlloc);
  }
  (void)_vira_gpu_load_sym(lib, "cuMemFree_v2", (void **)&g->cuMemFree);
  if (!g->cuMemFree) {
    (void)_vira_gpu_load_sym(lib, "cuMemFree", (void **)&g->cuMemFree);
  }
  (void)_vira_gpu_load_sym(lib, "cuMemcpyHtoD_v2", (void **)&g->cuMemcpyHtoD);
  if (!g->cuMemcpyHtoD) {
    (void)_vira_gpu_load_sym(lib, "cuMemcpyHtoD", (void **)&g->cuMemcpyHtoD);
  }
  (void)_vira_gpu_load_sym(lib, "cuMemcpyDtoH_v2", (void **)&g->cuMemcpyDtoH);
  if (!g->cuMemcpyDtoH) {
    (void)_vira_gpu_load_sym(lib, "cuMemcpyDtoH", (void **)&g->cuMemcpyDtoH);
  }
  (void)_vira_gpu_load_sym(lib, "cuStreamCreate", (void **)&g->cuStreamCreate);
  (void)_vira_gpu_load_sym(lib, "cuStreamDestroy_v2", (void **)&g->cuStreamDestroy);
  if (!g->cuStreamDestroy) {
    (void)_vira_gpu_load_sym(lib, "cuStreamDestroy", (void **)&g->cuStreamDestroy);
  }
  (void)_vira_gpu_load_sym(lib, "cuStreamSynchronize", (void **)&g->cuStreamSynchronize);
  (void)_vira_gpu_load_sym(lib, "cuLaunchKernel", (void **)&g->cuLaunchKernel);

  if (!g->cuCtxCreate || !g->cuModuleLoadData || !g->cuLaunchKernel ||
      !g->cuMemAlloc || !g->cuMemFree || !g->cuMemcpyHtoD || !g->cuMemcpyDtoH) {
    return false;
  }

  if (g->cuInit(0) != VIRA_CUDA_SUCCESS) {
    return false;
  }

  int dev_count = 0;
  if (g->cuDeviceGetCount(&dev_count) != VIRA_CUDA_SUCCESS || dev_count <= 0) {
    return false;
  }

  const char *env_dev = g->requested_device[0] ? g->requested_device : getenv("VIRA_GPU_DEVICE");
  vira_cu_device best_dev = -1;
  int64_t best_score = -1;
  char best_name[256] = {0};
  size_t best_vram = 0;
  int best_sm = 0;
  int best_integrated = 0;

  for (int i = 0; i < dev_count; ++i) {
    vira_cu_device d;
    if (g->cuDeviceGet(&d, i) != VIRA_CUDA_SUCCESS) continue;

    char cur_dname[256] = {0};
    g->cuDeviceGetName(cur_dname, sizeof(cur_dname) - 1, d);

    if (env_dev != NULL && env_dev[0] != '\0') {
      bool is_num = (env_dev[0] >= '0' && env_dev[0] <= '9');
      if (is_num) {
        if (atoi(env_dev) != i) continue;
      } else {
        if (!_vira_strcasestr_match(cur_dname, env_dev)) continue;
      }
    }

    size_t vram = 0;
    if (g->cuDeviceTotalMem) {
      g->cuDeviceTotalMem(&vram, d);
    }
    int sm_count = 0;
    g->cuDeviceGetAttribute(&sm_count, VIRA_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d);
    int integrated = 0;
    g->cuDeviceGetAttribute(&integrated, VIRA_CU_DEVICE_ATTRIBUTE_INTEGRATED, d);
    int cc_major = 0;
    g->cuDeviceGetAttribute(&cc_major, VIRA_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);
    int cc_minor = 0;
    g->cuDeviceGetAttribute(&cc_minor, VIRA_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);

    int64_t score = (int64_t)(vram >> 20) * (integrated == 0 ? 4LL : 1LL) + (int64_t)sm_count * 64LL;
    if (score > best_score) {
      best_score = score;
      best_dev = d;
      strncpy(best_name, cur_dname, sizeof(best_name) - 1);
      best_vram = vram;
      best_sm = sm_count;
      best_integrated = integrated;
      g->compute_capability_major = cc_major;
      g->compute_capability_minor = cc_minor;
    }
  }

  if (best_dev == -1) {
    return false;
  }

  vira_cu_context ctx = NULL;
  if (g->cuCtxCreate(&ctx, 0, best_dev) != VIRA_CUDA_SUCCESS || ctx == NULL) {
    return false;
  }

  vira_cu_stream stream = NULL;
  if (g->cuStreamCreate) {
    g->cuStreamCreate(&stream, 0);
  }

  vira_cu_module mod = NULL;
  if (g->cuModuleLoadData(&mod, _vira_cuda_kernel_ptx) != VIRA_CUDA_SUCCESS || mod == NULL) {
    if (stream && g->cuStreamDestroy) g->cuStreamDestroy(stream);
    g->cuCtxDestroy(ctx);
    return false;
  }

  vira_cu_function ks[VIRA_GPU_NUM_KERNELS] = {0};
  static const char *cu_kernel_names[VIRA_GPU_NUM_KERNELS] = {
      "vira_cu_gemv_q4_0",        "vira_cu_gemv_q4_1",
      "vira_cu_gemv_q5_0",        "vira_cu_gemv_q5_1",
      "vira_cu_gemv_q8_0",        "vira_cu_gemv_q4_k",
      "vira_cu_gemv_q5_k",        "vira_cu_gemv_q6_k",
      "vira_cu_gemv_vec8_h",      "vira_cu_gemv_vec4_h",
      "vira_cu_gemv_scalar_h",    "vira_cu_linear_tiled_h",
      "vira_cu_linear_tiled_f32", "vira_cu_matmul_tiled_f32",
      "vira_cu_rmsnorm_f32",      "vira_cu_rope_f32",
      "vira_cu_softmax_f32",      "vira_cu_silu_f32",
  };

  bool required_ok = true;
  for (int i = 0; i < VIRA_GPU_NUM_KERNELS; ++i) {
    if (g->cuModuleGetFunction(&ks[i], mod, cu_kernel_names[i]) != VIRA_CUDA_SUCCESS || ks[i] == NULL) {
      if (i < VIRA_GPU_NUM_REQUIRED_KERNELS) {
        required_ok = false;
      }
    }
  }

  if (!required_ok) {
    g->cuModuleUnload(mod);
    if (stream && g->cuStreamDestroy) g->cuStreamDestroy(stream);
    g->cuCtxDestroy(ctx);
    return false;
  }

  g->cuda_lib_handle = lib;
  g->cu_device = best_dev;
  g->cu_context = ctx;
  g->cu_stream = stream;
  g->cu_module = mod;

  g->cu_k_gemv_q4_0 = ks[0];
  g->cu_k_gemv_q4_1 = ks[1];
  g->cu_k_gemv_q5_0 = ks[2];
  g->cu_k_gemv_q5_1 = ks[3];
  g->cu_k_gemv_q8_0 = ks[4];
  g->cu_k_gemv_q4_k = ks[5];
  g->cu_k_gemv_q5_k = ks[6];
  g->cu_k_gemv_q6_k = ks[7];
  g->cu_k_gemv_vec8 = ks[8];
  g->cu_k_gemv_vec4 = ks[9];
  g->cu_k_gemv_scalar = ks[10];
  g->cu_k_linear_tiled_h = ks[11];
  g->cu_k_linear_tiled_f32 = ks[12];
  g->cu_k_matmul_tiled = ks[13];
  g->cu_k_rmsnorm = ks[14];
  g->cu_k_rope = ks[15];
  g->cu_k_softmax = ks[16];
  g->cu_k_silu = ks[17];

  g->has_tensor_cores = (g->compute_capability_major >= 7);
  g->cu_wmma_module = NULL;
  g->cu_k_wmma_linear_h = NULL;
  g->cu_k_wmma_matmul = NULL;

  if (g->has_tensor_cores && g->cuModuleLoadData) {
    vira_cu_module wmma_mod = NULL;
    if (g->cuModuleLoadData(&wmma_mod, _vira_cuda_wmma_kernel_ptx) == VIRA_CUDA_SUCCESS && wmma_mod != NULL) {
      g->cu_wmma_module = wmma_mod;
      (void)g->cuModuleGetFunction(&g->cu_k_wmma_linear_h, wmma_mod, "vira_cu_wmma_linear_h");
      (void)g->cuModuleGetFunction(&g->cu_k_wmma_matmul, wmma_mod, "vira_cu_wmma_matmul_f32");
    }
  }

  g->active_backend = VIRA_GPU_BACKEND_CUDA;
  g->available = true;
  snprintf(g->backend_name, sizeof(g->backend_name), "CUDA");
  snprintf(g->device_name, sizeof(g->device_name), "%s", best_name);

  g->warp_size = 32;
  g->rows_per_wg = 8;
  g->wg_size = 256;
  g->tile_dim = 16;
  g->compute_units = best_sm;
  g->unified_mem = (best_integrated != 0);
  g->vram_bytes = (int64_t)best_vram;

  const char *env_b = getenv("VIRA_GPU_VRAM_BUDGET_MB");
  if (env_b != NULL && atoi(env_b) > 0) {
    g->vram_budget_bytes = (int64_t)atoi(env_b) * 1024LL * 1024LL;
  } else {
    g->vram_budget_bytes = (int64_t)((best_vram * 88ULL) / 100ULL);
  }

  if (_vira_gpu_debug()) {
    fprintf(stderr,
            "vira gpu: [CUDA Driver API] %s (sm_%d%d%s) | %lld MiB VRAM (budget %lld MiB) | %d SMs | "
            "warp=%d, wg=%d, tile=%d | GGUF Q4_0..Q6_K / FP16%s\n",
            g->device_name, g->compute_capability_major, g->compute_capability_minor,
            g->has_tensor_cores ? " + Tensor Cores" : "",
            (long long)(g->vram_bytes >> 20),
            (long long)(g->vram_budget_bytes >> 20), g->compute_units,
            g->warp_size, g->wg_size, g->tile_dim,
            g->has_tensor_cores ? " / WMMA" : "");
  }

  return true;
}

static inline bool _vira_gpu_init_opencl_locked(void *lib) {
  _vira_gpu_state *g = &_vira_gpu;
  bool ok = true;
#define VIRA_GPU_LOAD(sym) ok = ok && _vira_gpu_load_sym(lib, #sym, (void **)&g->sym)
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
  (void)_vira_gpu_load_sym(lib, "clGetProgramBuildInfo", (void **)&g->clGetProgramBuildInfo);
  (void)_vira_gpu_load_sym(lib, "clEnqueueMapBuffer", (void **)&g->clEnqueueMapBuffer);
  (void)_vira_gpu_load_sym(lib, "clEnqueueUnmapMemObject", (void **)&g->clEnqueueUnmapMemObject);
  (void)_vira_gpu_load_sym(lib, "clFlush", (void **)&g->clFlush);
  (void)_vira_gpu_load_sym(lib, "clWaitForEvents", (void **)&g->clWaitForEvents);
  (void)_vira_gpu_load_sym(lib, "clReleaseEvent", (void **)&g->clReleaseEvent);

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
  const char *env_dev = g->requested_device[0] ? g->requested_device : getenv("VIRA_GPU_DEVICE");
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

  g->active_backend = VIRA_GPU_BACKEND_OPENCL;
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
            "vira gpu: [OpenCL] %s | %lld MiB VRAM (budget %lld MiB) | %d CU | "
            "warp=%d, wg=%d, tile=%d | GGUF Q4_0..Q6_K / FP16\n",
            g->device_name, (long long)(g->vram_bytes >> 20),
            (long long)(g->vram_budget_bytes >> 20), g->compute_units,
            g->warp_size, g->wg_size, g->tile_dim);
  }
  return true;
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
  _vira_gpu.active_backend = VIRA_GPU_BACKEND_CPU;
  _vira_gpu.available = false;
  _vira_gpu.enabled = true;

  const char *req_b = _vira_gpu.requested_backend[0] ? _vira_gpu.requested_backend : getenv("VIRA_GPU_BACKEND");
  bool allow_cuda = true;
  bool allow_opencl = true;

  if (req_b != NULL && req_b[0] != '\0') {
    if (_vira_strcasestr_match(req_b, "cpu")) {
      allow_cuda = false;
      allow_opencl = false;
    } else if (_vira_strcasestr_match(req_b, "cuda")) {
      allow_cuda = true;
      allow_opencl = false;
    } else if (_vira_strcasestr_match(req_b, "opencl") || _vira_strcasestr_match(req_b, "cl")) {
      allow_cuda = false;
      allow_opencl = true;
    }
  }

  // 1. Probe CUDA Driver if allowed
  if (allow_cuda) {
    void *cuda_lib = NULL;
    const char *env_cuda_lib = getenv("VIRA_CUDA_LIB");
    if (env_cuda_lib != NULL && env_cuda_lib[0] != '\0') {
#ifdef _WIN32
      cuda_lib = (void *)LoadLibraryA(env_cuda_lib);
#else
      cuda_lib = dlopen(env_cuda_lib, RTLD_LAZY);
#endif
    }
    if (!cuda_lib) {
#ifdef _WIN32
      cuda_lib = (void *)LoadLibraryA("nvcuda.dll");
#elif !defined(__APPLE__)
      cuda_lib = dlopen("libcuda.so.1", RTLD_LAZY);
      if (!cuda_lib) cuda_lib = dlopen("libcuda.so", RTLD_LAZY);
#endif
    }
    if (cuda_lib != NULL) {
      if (_vira_gpu_init_cuda_locked(cuda_lib)) {
        __atomic_store_n(&_vira_gpu.initialized, 2, __ATOMIC_RELEASE);
        pthread_mutex_unlock(&_vira_gpu.lock);
        return;
      }
#ifdef _WIN32
      FreeLibrary((HMODULE)cuda_lib);
#else
      dlclose(cuda_lib);
#endif
    }
  }

  // 2. Probe OpenCL if allowed
  if (allow_opencl) {
    void *cl_lib = NULL;
    const char *env_cl_lib = getenv("VIRA_OPENCL_LIB");
    if (env_cl_lib != NULL && env_cl_lib[0] != '\0') {
#ifdef _WIN32
      cl_lib = (void *)LoadLibraryA(env_cl_lib);
#else
      cl_lib = dlopen(env_cl_lib, RTLD_LAZY);
#endif
    }
    if (!cl_lib) {
#ifdef _WIN32
      cl_lib = (void *)LoadLibraryA("OpenCL.dll");
#elif defined(__APPLE__)
      cl_lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_LAZY);
#else
      cl_lib = dlopen("libOpenCL.so.1", RTLD_LAZY);
      if (!cl_lib) cl_lib = dlopen("libOpenCL.so", RTLD_LAZY);
#endif
    }
    if (cl_lib != NULL) {
      if (_vira_gpu_init_opencl_locked(cl_lib)) {
        __atomic_store_n(&_vira_gpu.initialized, 2, __ATOMIC_RELEASE);
        pthread_mutex_unlock(&_vira_gpu.lock);
        return;
      }
#ifdef _WIN32
      FreeLibrary((HMODULE)cl_lib);
#else
      dlclose(cl_lib);
#endif
    }
  }

  // 3. Fallback to CPU-SIMD
  _vira_gpu.active_backend = VIRA_GPU_BACKEND_CPU;
  _vira_gpu.available = false;
  _vira_gpu.enabled = true;
  snprintf(_vira_gpu.device_name, sizeof(_vira_gpu.device_name),
           "Multi-Threaded CPU SIMD (%s)", vira_simd_backend_name());
  snprintf(_vira_gpu.backend_name, sizeof(_vira_gpu.backend_name), "CPU-SIMD");

  __atomic_store_n(&_vira_gpu.initialized, 2, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline void vira_gpu_shutdown(void) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) != 2) {
    return;
  }
  pthread_mutex_lock(&_vira_gpu.lock);
  if (_vira_gpu.available) {
    _vira_gpu_evict_all_locked();

    if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
      if (_vira_gpu.cuCtxSynchronize) _vira_gpu.cuCtxSynchronize();
      if (_vira_gpu.cu_wmma_module && _vira_gpu.cuModuleUnload) {
        _vira_gpu.cuModuleUnload(_vira_gpu.cu_wmma_module);
        _vira_gpu.cu_wmma_module = NULL;
      }
      _vira_gpu.cu_k_wmma_linear_h = NULL;
      _vira_gpu.cu_k_wmma_matmul = NULL;
      _vira_gpu.has_tensor_cores = false;
      if (_vira_gpu.cu_module && _vira_gpu.cuModuleUnload) {
        _vira_gpu.cuModuleUnload(_vira_gpu.cu_module);
        _vira_gpu.cu_module = NULL;
      }
      if (_vira_gpu.cu_stream && _vira_gpu.cuStreamDestroy) {
        _vira_gpu.cuStreamDestroy(_vira_gpu.cu_stream);
        _vira_gpu.cu_stream = NULL;
      }
      if (_vira_gpu.cu_context && _vira_gpu.cuCtxDestroy) {
        _vira_gpu.cuCtxDestroy(_vira_gpu.cu_context);
        _vira_gpu.cu_context = NULL;
      }
      if (_vira_gpu.cuda_lib_handle) {
#ifdef _WIN32
        FreeLibrary((HMODULE)_vira_gpu.cuda_lib_handle);
#else
        dlclose(_vira_gpu.cuda_lib_handle);
#endif
        _vira_gpu.cuda_lib_handle = NULL;
      }
    } else if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_OPENCL) {
      if (_vira_gpu.queue) {
        _vira_gpu.clFinish(_vira_gpu.queue);
      }
      if (_vira_gpu.xfer_queue && _vira_gpu.xfer_queue != _vira_gpu.queue) {
        _vira_gpu.clFinish(_vira_gpu.xfer_queue);
      }
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
      if (_vira_gpu.lib_handle) {
#ifdef _WIN32
        FreeLibrary((HMODULE)_vira_gpu.lib_handle);
#else
        dlclose(_vira_gpu.lib_handle);
#endif
        _vira_gpu.lib_handle = NULL;
      }
    }
    _vira_gpu.available = false;
  }
  _vira_gpu.active_backend = VIRA_GPU_BACKEND_CPU;
  __atomic_store_n(&_vira_gpu.initialized, 0, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline bool vira_gpu_is_available(void) {
  vira_gpu_init();
  return _vira_gpu.available;
}

static inline bool vira_gpu_is_unified_memory(void) {
  vira_gpu_init();
  return _vira_gpu.unified_mem;
}

static inline bool vira_gpu_has_tensor_cores(void) {
  vira_gpu_init();
  return _vira_gpu.has_tensor_cores;
}

static inline vira_string vira_gpu_device_name(void) {
  vira_gpu_init();
  return vira_str_lit(_vira_gpu.device_name, (int32_t)strlen(_vira_gpu.device_name));
}

static inline vira_string vira_gpu_backend_name(void) {
  vira_gpu_init();
  return vira_str_lit(_vira_gpu.backend_name, (int32_t)strlen(_vira_gpu.backend_name));
}

static inline int64_t vira_gpu_vram_bytes(void) {
  vira_gpu_init();
  return _vira_gpu.vram_bytes;
}

static inline int64_t vira_gpu_allocated_bytes(void) {
  return __atomic_load_n(&_vira_gpu.allocated_bytes, __ATOMIC_RELAXED);
}

static inline int64_t vira_gpu_oom_fallbacks(void) {
  return __atomic_load_n(&_vira_gpu.oom_fallbacks, __ATOMIC_RELAXED);
}

static inline void vira_gpu_set_enabled(bool enabled) {
  pthread_mutex_lock(&_vira_gpu.lock);
  _vira_gpu.enabled = enabled;
  pthread_mutex_unlock(&_vira_gpu.lock);
}

static inline bool vira_gpu_is_enabled(void) {
  return _vira_gpu.enabled;
}

static inline void vira_gpu_simulate_oom(bool enable) {
  _vira_gpu.simulate_oom = enable;
}

static inline void vira_gpu_set_backend(vira_string backend) {
  vira_gpu_shutdown();
  pthread_mutex_lock(&_vira_gpu.lock);
  if (backend.data != NULL && backend.len > 0) {
    size_t copy_len = backend.len < sizeof(_vira_gpu.requested_backend) - 1 ? backend.len : sizeof(_vira_gpu.requested_backend) - 1;
    memcpy(_vira_gpu.requested_backend, backend.data, copy_len);
    _vira_gpu.requested_backend[copy_len] = '\0';
  } else {
    _vira_gpu.requested_backend[0] = '\0';
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
  vira_gpu_init();
}

static inline void vira_gpu_set_device(vira_string device) {
  vira_gpu_shutdown();
  pthread_mutex_lock(&_vira_gpu.lock);
  if (device.data != NULL && device.len > 0) {
    size_t copy_len = device.len < sizeof(_vira_gpu.requested_device) - 1 ? device.len : sizeof(_vira_gpu.requested_device) - 1;
    memcpy(_vira_gpu.requested_device, device.data, copy_len);
    _vira_gpu.requested_device[copy_len] = '\0';
  } else {
    _vira_gpu.requested_device[0] = '\0';
  }
  pthread_mutex_unlock(&_vira_gpu.lock);
  vira_gpu_init();
}

static inline void vira_gpu_sync(void) {
  if (__atomic_load_n(&_vira_gpu.initialized, __ATOMIC_ACQUIRE) == 2 &&
      _vira_gpu.available) {
    pthread_mutex_lock(&_vira_gpu.lock);
    if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
      if (_vira_gpu.cuCtxSynchronize) _vira_gpu.cuCtxSynchronize();
    } else {
      if (_vira_gpu.queue) _vira_gpu.clFinish(_vira_gpu.queue);
    }
    pthread_mutex_unlock(&_vira_gpu.lock);
  }
}

static inline vira_tensor_f32 vira_gpu_linear(vira_arena *opt_arena,
                                              vira_tensor_f32 x,
                                              vira_tensor_f32 w) {
  vira_gpu_init();
  if (!_vira_gpu.available || !_vira_gpu.enabled || _vira_gpu.simulate_oom ||
      x.data == NULL || w.data == NULL || x.n_dims < 1 || w.n_dims < 2) {
    if (_vira_gpu.simulate_oom) {
      __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    }
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int32_t K = w.dims[1];
  int32_t N = w.dims[0];
  int32_t M = 1;
  for (int32_t i = 0; i < x.n_dims - 1; ++i) {
    M *= x.dims[i];
  }
  if (x.dims[x.n_dims - 1] != K || K <= 0 || N <= 0 || M <= 0) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  pthread_mutex_lock(&_vira_gpu.lock);
  bool prefer_half = ((M <= 4) && (K >= 32)) || (_vira_gpu.has_tensor_cores && (M >= 16) && ((K & 15) == 0));
  _vira_gpu_wfmt wfmt = VIRA_WFMT_F32;
  vira_cl_int err = 0;

  bool is_act_w = _vira_gpu_is_activation_ptr_locked(w.data);
  _vira_gpu_cache_entry *w_slot = NULL;
  _vira_gpu_act_entry *w_act_slot = NULL;

  if (is_act_w) {
    wfmt = VIRA_WFMT_F32;
    bool w_hit = false;
    w_act_slot = _vira_gpu_get_or_upload_activation_locked(
        w.data, (size_t)K * (size_t)N * sizeof(vira_f32), &w_hit, &err);
  } else {
    w_slot = _vira_gpu_get_or_upload_weight_locked(
        w.data, (size_t)K * (size_t)N, prefer_half, &wfmt, &err);
  }

  if (err != VIRA_CL_SUCCESS || (w_slot == NULL && w_act_slot == NULL)) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  const size_t x_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  const size_t y_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);

  bool x_hit = false;
  _vira_gpu_act_entry *x_slot = _vira_gpu_get_or_upload_activation_locked(
      x.data, x_bytes, &x_hit, &err);
  if (err != VIRA_CL_SUCCESS || x_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  vira_cl_mem avoid1_cl = x_slot->dev_buf;
  vira_cl_mem avoid2_cl = is_act_w ? w_act_slot->dev_buf : (w_slot ? w_slot->dev_buf : NULL);
  vira_cu_deviceptr avoid1_cu = x_slot->cu_buf;
  vira_cu_deviceptr avoid2_cu = is_act_w ? w_act_slot->cu_buf : (w_slot ? w_slot->cu_buf : 0);

  _vira_gpu_act_entry *y_slot = _vira_gpu_alloc_activation_slot_locked(
      y_bytes, avoid1_cl, avoid2_cl, avoid1_cu, avoid2_cu, &err);
  if (err != VIRA_CL_SUCCESS || y_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    vira_cu_deviceptr x_buf = x_slot->cu_buf;
    vira_cu_deviceptr w_buf = is_act_w ? w_act_slot->cu_buf : w_slot->cu_buf;
    vira_cu_deviceptr y_buf = y_slot->cu_buf;

    if (wfmt == VIRA_WFMT_Q4_0 || wfmt == VIRA_WFMT_Q4_1 ||
        wfmt == VIRA_WFMT_Q5_0 || wfmt == VIRA_WFMT_Q5_1 ||
        wfmt == VIRA_WFMT_Q8_0) {
      vira_cu_function k;
      switch (wfmt) {
      case VIRA_WFMT_Q4_0: k = _vira_gpu.cu_k_gemv_q4_0; break;
      case VIRA_WFMT_Q4_1: k = _vira_gpu.cu_k_gemv_q4_1; break;
      case VIRA_WFMT_Q5_0: k = _vira_gpu.cu_k_gemv_q5_0; break;
      case VIRA_WFMT_Q5_1: k = _vira_gpu.cu_k_gemv_q5_1; break;
      default:             k = _vira_gpu.cu_k_gemv_q8_0; break;
      }
      int32_t num_blocks_k = K >> 5;
      void *args[] = {&x_buf, &w_buf, &y_buf, &M, &num_blocks_k, &N};
      unsigned int gx = (unsigned int)(((size_t)N + _vira_gpu.rows_per_wg - 1) / _vira_gpu.rows_per_wg);
      unsigned int gy = (unsigned int)M;
      err = (_vira_gpu.cuLaunchKernel(k, gx, gy, 1, _vira_gpu.warp_size, _vira_gpu.rows_per_wg, 1,
                                      0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
                ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
    } else if (wfmt == VIRA_WFMT_Q4_K || wfmt == VIRA_WFMT_Q5_K || wfmt == VIRA_WFMT_Q6_K) {
      vira_cu_function k;
      switch (wfmt) {
      case VIRA_WFMT_Q4_K: k = _vira_gpu.cu_k_gemv_q4_k; break;
      case VIRA_WFMT_Q5_K: k = _vira_gpu.cu_k_gemv_q5_k; break;
      default:             k = _vira_gpu.cu_k_gemv_q6_k; break;
      }
      int32_t num_sb_k = K >> 8;
      void *args[] = {&x_buf, &w_buf, &y_buf, &M, &num_sb_k, &N};
      unsigned int gx = (unsigned int)(((size_t)N + _vira_gpu.rows_per_wg - 1) / _vira_gpu.rows_per_wg);
      unsigned int gy = (unsigned int)M;
      err = (_vira_gpu.cuLaunchKernel(k, gx, gy, 1, _vira_gpu.warp_size, _vira_gpu.rows_per_wg, 1,
                                      0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
                ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
    } else if (wfmt == VIRA_WFMT_F32) {
      void *args[] = {&x_buf, &w_buf, &y_buf, &M, &K, &N};
      unsigned int gx = (unsigned int)(((size_t)N + _vira_gpu.tile_dim - 1) / _vira_gpu.tile_dim);
      unsigned int gy = (unsigned int)(((size_t)M + _vira_gpu.tile_dim - 1) / _vira_gpu.tile_dim);
      err = (_vira_gpu.cuLaunchKernel(_vira_gpu.cu_k_linear_tiled_f32, gx, gy, 1,
                                      _vira_gpu.tile_dim, _vira_gpu.tile_dim, 1,
                                      0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
                ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
    } else if (M <= 4) {
      vira_cu_function k;
      int32_t kdim;
      if ((K & 7) == 0) {
        k = _vira_gpu.cu_k_gemv_vec8;
        kdim = K >> 3;
      } else if ((K & 3) == 0) {
        k = _vira_gpu.cu_k_gemv_vec4;
        kdim = K >> 2;
      } else {
        k = _vira_gpu.cu_k_gemv_scalar;
        kdim = K;
      }
      void *args[] = {&x_buf, &w_buf, &y_buf, &M, &kdim, &N};
      unsigned int gx = (unsigned int)(((size_t)N + _vira_gpu.rows_per_wg - 1) / _vira_gpu.rows_per_wg);
      unsigned int gy = (unsigned int)M;
      err = (_vira_gpu.cuLaunchKernel(k, gx, gy, 1, _vira_gpu.warp_size, _vira_gpu.rows_per_wg, 1,
                                      0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
                ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
    } else if (_vira_gpu.has_tensor_cores && _vira_gpu.cu_k_wmma_linear_h != NULL && M >= 16 && ((K & 15) == 0)) {
      void *args[] = {&x_buf, &w_buf, &y_buf, &M, &K, &N};
      unsigned int gx = (unsigned int)(((size_t)N + 15) / 16);
      unsigned int gy = (unsigned int)(((size_t)M + 15) / 16);
      err = (_vira_gpu.cuLaunchKernel(_vira_gpu.cu_k_wmma_linear_h, gx, gy, 1,
                                      32, 1, 1,
                                      0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
                ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
    } else {
      void *args[] = {&x_buf, &w_buf, &y_buf, &M, &K, &N};
      unsigned int gx = (unsigned int)(((size_t)N + _vira_gpu.tile_dim - 1) / _vira_gpu.tile_dim);
      unsigned int gy = (unsigned int)(((size_t)M + _vira_gpu.tile_dim - 1) / _vira_gpu.tile_dim);
      err = (_vira_gpu.cuLaunchKernel(_vira_gpu.cu_k_linear_tiled_h, gx, gy, 1,
                                      _vira_gpu.tile_dim, _vira_gpu.tile_dim, 1,
                                      0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
                ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
    }
  } else {
    vira_cl_mem x_buf = x_slot->dev_buf;
    vira_cl_mem w_buf = is_act_w ? w_act_slot->dev_buf : w_slot->dev_buf;
    vira_cl_mem y_buf = y_slot->dev_buf;

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
      size_t l_work[2] = {(size_t)_vira_gpu.warp_size, (size_t)_vira_gpu.rows_per_wg};
      size_t num_groups_n = ((size_t)N + _vira_gpu.rows_per_wg - 1) / _vira_gpu.rows_per_wg;
      size_t g_work[2] = {num_groups_n * _vira_gpu.warp_size, (size_t)M * _vira_gpu.rows_per_wg};
      if (err == VIRA_CL_SUCCESS) {
        err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, k, 2, NULL, g_work, l_work, 0, NULL, NULL);
      }
    } else if (wfmt == VIRA_WFMT_Q4_K || wfmt == VIRA_WFMT_Q5_K || wfmt == VIRA_WFMT_Q6_K) {
      vira_cl_kernel k;
      switch (wfmt) {
      case VIRA_WFMT_Q4_K: k = _vira_gpu.kernel_gemv_q4_k; break;
      case VIRA_WFMT_Q5_K: k = _vira_gpu.kernel_gemv_q5_k; break;
      default:             k = _vira_gpu.kernel_gemv_q6_k; break;
      }
      int32_t num_sb_k = K >> 8;
      err = _vira_gpu_set_gemm_args(k, &x_buf, &w_buf, &y_buf, M, num_sb_k, N);
      size_t l_work[2] = {(size_t)_vira_gpu.warp_size, (size_t)_vira_gpu.rows_per_wg};
      size_t num_groups_n = ((size_t)N + _vira_gpu.rows_per_wg - 1) / _vira_gpu.rows_per_wg;
      size_t g_work[2] = {num_groups_n * _vira_gpu.warp_size, (size_t)M * _vira_gpu.rows_per_wg};
      if (err == VIRA_CL_SUCCESS) {
        err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, k, 2, NULL, g_work, l_work, 0, NULL, NULL);
      }
    } else if (wfmt == VIRA_WFMT_F32) {
      err = _vira_gpu_set_gemm_args(_vira_gpu.kernel_linear_tiled_f32, &x_buf, &w_buf, &y_buf, M, K, N);
      size_t td = (size_t)_vira_gpu.tile_dim;
      size_t l_work[2] = {td, td};
      size_t g_work[2] = {((size_t)N + td - 1) & ~(td - 1), ((size_t)M + td - 1) & ~(td - 1)};
      if (err == VIRA_CL_SUCCESS) {
        err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, _vira_gpu.kernel_linear_tiled_f32, 2, NULL, g_work, l_work, 0, NULL, NULL);
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
      size_t l_work[2] = {(size_t)_vira_gpu.warp_size, (size_t)_vira_gpu.rows_per_wg};
      size_t num_groups_n = ((size_t)N + _vira_gpu.rows_per_wg - 1) / _vira_gpu.rows_per_wg;
      size_t g_work[2] = {num_groups_n * _vira_gpu.warp_size, (size_t)M * _vira_gpu.rows_per_wg};
      if (err == VIRA_CL_SUCCESS) {
        err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, k, 2, NULL, g_work, l_work, 0, NULL, NULL);
      }
    } else {
      err = _vira_gpu_set_gemm_args(_vira_gpu.kernel_linear_tiled_h, &x_buf, &w_buf, &y_buf, M, K, N);
      size_t td = (size_t)_vira_gpu.tile_dim;
      size_t l_work[2] = {td, td};
      size_t g_work[2] = {((size_t)N + td - 1) & ~(td - 1), ((size_t)M + td - 1) & ~(td - 1)};
      if (err == VIRA_CL_SUCCESS) {
        err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, _vira_gpu.kernel_linear_tiled_h, 2, NULL, g_work, l_work, 0, NULL, NULL);
      }
    }
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  if (err == VIRA_CL_SUCCESS) {
    if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
      if (_vira_gpu.cuMemcpyDtoH(out.data, y_slot->cu_buf, y_bytes) != VIRA_CUDA_SUCCESS) {
        err = VIRA_CL_ERR_NO_ROOM;
      }
    } else {
      err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_slot->dev_buf, 1, 0, y_bytes,
                                          out.data, 0, NULL, NULL);
    }
  }

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_linear(opt_arena, x, w);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  return out;
}

static inline vira_tensor_f32 vira_gpu_matmul(vira_arena *opt_arena,
                                              vira_tensor_f32 a,
                                              vira_tensor_f32 b) {
  vira_gpu_init();
  if (!_vira_gpu.available || !_vira_gpu.enabled || _vira_gpu.simulate_oom ||
      a.data == NULL || b.data == NULL || a.n_dims < 2 || b.n_dims < 2) {
    if (_vira_gpu.simulate_oom) {
      __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    }
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  int32_t M = a.dims[0];
  int32_t K = a.dims[1];
  int32_t Kb = b.dims[0];
  int32_t N = b.dims[1];
  if (K != Kb || M <= 0 || K <= 0 || N <= 0) {
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  pthread_mutex_lock(&_vira_gpu.lock);
  const size_t a_bytes = (size_t)M * (size_t)K * sizeof(vira_f32);
  const size_t b_bytes = (size_t)K * (size_t)N * sizeof(vira_f32);
  const size_t c_bytes = (size_t)M * (size_t)N * sizeof(vira_f32);

  vira_cl_int err = 0;
  bool a_hit = false;
  _vira_gpu_act_entry *a_slot = _vira_gpu_get_or_upload_activation_locked(
      a.data, a_bytes, &a_hit, &err);
  if (err != VIRA_CL_SUCCESS || a_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  bool b_hit = false;
  _vira_gpu_act_entry *b_slot = _vira_gpu_get_or_upload_activation_locked(
      b.data, b_bytes, &b_hit, &err);
  if (err != VIRA_CL_SUCCESS || b_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  _vira_gpu_act_entry *c_slot = _vira_gpu_alloc_activation_slot_locked(
      c_bytes, a_slot->dev_buf, b_slot->dev_buf, a_slot->cu_buf, b_slot->cu_buf, &err);
  if (err != VIRA_CL_SUCCESS || c_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    if (_vira_gpu.has_tensor_cores && _vira_gpu.cu_k_wmma_matmul != NULL &&
        M >= 16 && ((K & 15) == 0)) {
      void *args[] = {&a_slot->cu_buf, &b_slot->cu_buf, &c_slot->cu_buf, &M, &K, &N};
      unsigned int gx = (unsigned int)(((size_t)N + 15) / 16);
      unsigned int gy = (unsigned int)(((size_t)M + 15) / 16);
      err = (_vira_gpu.cuLaunchKernel(_vira_gpu.cu_k_wmma_matmul, gx, gy, 1,
                                      32, 1, 1, 0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
                ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
    } else {
      void *args[] = {&a_slot->cu_buf, &b_slot->cu_buf, &c_slot->cu_buf, &M, &K, &N};
      unsigned int gx = (unsigned int)(((size_t)N + _vira_gpu.tile_dim - 1) / _vira_gpu.tile_dim);
      unsigned int gy = (unsigned int)(((size_t)M + _vira_gpu.tile_dim - 1) / _vira_gpu.tile_dim);
      err = (_vira_gpu.cuLaunchKernel(_vira_gpu.cu_k_matmul_tiled, gx, gy, 1,
                                      _vira_gpu.tile_dim, _vira_gpu.tile_dim, 1,
                                      0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
                ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
    }
  } else {
    err = _vira_gpu_set_gemm_args(_vira_gpu.kernel_matmul_tiled, &a_slot->dev_buf,
                                  &b_slot->dev_buf, &c_slot->dev_buf, M, K, N);
    size_t td = (size_t)_vira_gpu.tile_dim;
    size_t l_work[2] = {td, td};
    size_t g_work[2] = {((size_t)N + td - 1) & ~(td - 1), ((size_t)M + td - 1) & ~(td - 1)};
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, _vira_gpu.kernel_matmul_tiled,
                                             2, NULL, g_work, l_work, 0, NULL, NULL);
    }
  }

  int32_t out_dims[2] = {M, N};
  vira_tensor_f32 out =
      opt_arena ? vira_tensor_f32_arena_alloc_nd(opt_arena, 2, out_dims)
                : vira_tensor_f32_alloc_nd(2, out_dims);

  if (err == VIRA_CL_SUCCESS) {
    if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
      if (_vira_gpu.cuMemcpyDtoH(out.data, c_slot->cu_buf, c_bytes) != VIRA_CUDA_SUCCESS) {
        err = VIRA_CL_ERR_NO_ROOM;
      }
    } else {
      err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, c_slot->dev_buf, 1, 0, c_bytes,
                                          out.data, 0, NULL, NULL);
    }
  }

  if (err == VIRA_CL_SUCCESS) {
    c_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    _vira_tls_gpu_fallback = true;
    vira_tensor_f32 res = vira_tensor_matmul_f32(opt_arena, a, b);
    _vira_tls_gpu_fallback = false;
    return res;
  }

  return out;
}

static inline vira_tensor_f32 vira_gpu_rmsnorm(vira_tensor_f32 x,
                                               vira_tensor_f32 weight,
                                               float eps) {
  vira_gpu_init();
  if (!_vira_gpu.available || !_vira_gpu.enabled || _vira_gpu.simulate_oom ||
      x.data == NULL || weight.data == NULL || x.n_dims < 1) {
    if (_vira_gpu.simulate_oom) {
      __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    }
    return vira_tensor_rmsnorm(x, weight, eps);
  }

  int32_t cols = x.dims[x.n_dims - 1];
  int32_t rows = 1;
  for (int32_t i = 0; i < x.n_dims - 1; ++i) {
    rows *= x.dims[i];
  }
  if (cols <= 0 || rows <= 0 || weight.dims[weight.n_dims - 1] != cols) {
    return vira_tensor_rmsnorm(x, weight, eps);
  }

  pthread_mutex_lock(&_vira_gpu.lock);
  const size_t x_bytes = (size_t)rows * (size_t)cols * sizeof(vira_f32);
  const size_t w_bytes = (size_t)cols * sizeof(vira_f32);

  vira_cl_int err = 0;
  bool x_hit = false;
  _vira_gpu_act_entry *x_slot = _vira_gpu_get_or_upload_activation_locked(
      x.data, x_bytes, &x_hit, &err);
  if (err != VIRA_CL_SUCCESS || x_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_rmsnorm(x, weight, eps);
  }

  bool w_hit = false;
  _vira_gpu_act_entry *w_slot = _vira_gpu_get_or_upload_activation_locked(
      weight.data, w_bytes, &w_hit, &err);
  if (err != VIRA_CL_SUCCESS || w_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_rmsnorm(x, weight, eps);
  }

  _vira_gpu_act_entry *y_slot = _vira_gpu_alloc_activation_slot_locked(
      x_bytes, x_slot->dev_buf, w_slot->dev_buf, x_slot->cu_buf, w_slot->cu_buf, &err);
  if (err != VIRA_CL_SUCCESS || y_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_rmsnorm(x, weight, eps);
  }

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    void *args[] = {&x_slot->cu_buf, &w_slot->cu_buf, &y_slot->cu_buf, &cols, &eps};
    err = (_vira_gpu.cuLaunchKernel(_vira_gpu.cu_k_rmsnorm, (unsigned int)rows, 1, 1,
                                    _vira_gpu.wg_size, 1, 1, 0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
              ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
  } else {
    err = _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 0, sizeof(vira_cl_mem), &x_slot->dev_buf);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 1, sizeof(vira_cl_mem), &w_slot->dev_buf);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 2, sizeof(vira_cl_mem), &y_slot->dev_buf);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 3, sizeof(int32_t), &cols);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rmsnorm, 4, sizeof(float), &eps);
    size_t l_work = (size_t)_vira_gpu.wg_size;
    size_t g_work = (size_t)rows * l_work;
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, _vira_gpu.kernel_rmsnorm, 1,
                                             NULL, &g_work, &l_work, 0, NULL, NULL);
    }
  }

  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  if (err == VIRA_CL_SUCCESS) {
    if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
      if (_vira_gpu.cuMemcpyDtoH(out.data, y_slot->cu_buf, x_bytes) != VIRA_CUDA_SUCCESS) {
        err = VIRA_CL_ERR_NO_ROOM;
      }
    } else {
      err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_slot->dev_buf, 1, 0, x_bytes,
                                          out.data, 0, NULL, NULL);
    }
  }

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_rmsnorm(x, weight, eps);
  }

  return out;
}

static inline vira_tensor_f32 vira_gpu_rope(vira_tensor_f32 q, int64_t pos,
                                            float freq_base) {
  vira_gpu_init();
  if (!_vira_gpu.available || !_vira_gpu.enabled || _vira_gpu.simulate_oom ||
      q.data == NULL || q.n_dims < 2) {
    if (_vira_gpu.simulate_oom) {
      __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    }
    return vira_tensor_rope(q, pos, freq_base);
  }

  int32_t cols = q.dims[q.n_dims - 1];
  int32_t rows = 1;
  for (int32_t i = 0; i < q.n_dims - 1; ++i) {
    rows *= q.dims[i];
  }
  if ((cols & 1) != 0 || cols <= 0 || rows <= 0) {
    return vira_tensor_rope(q, pos, freq_base);
  }

  int32_t half_dim = cols / 2;
  float pos_f = (float)pos;

  pthread_mutex_lock(&_vira_gpu.lock);
  const size_t q_bytes = (size_t)rows * (size_t)cols * sizeof(vira_f32);

  vira_cl_int err = 0;
  bool q_hit = false;
  _vira_gpu_act_entry *q_slot = _vira_gpu_get_or_upload_activation_locked(
      q.data, q_bytes, &q_hit, &err);
  if (err != VIRA_CL_SUCCESS || q_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_rope(q, pos, freq_base);
  }

  _vira_gpu_act_entry *y_slot = _vira_gpu_alloc_activation_slot_locked(
      q_bytes, q_slot->dev_buf, NULL, q_slot->cu_buf, 0, &err);
  if (err != VIRA_CL_SUCCESS || y_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_rope(q, pos, freq_base);
  }

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    void *args[] = {&q_slot->cu_buf, &y_slot->cu_buf, &rows, &cols, &half_dim, &pos_f, &freq_base};
    unsigned int gx = (unsigned int)((half_dim + 15) / 16);
    unsigned int gy = (unsigned int)((rows + 15) / 16);
    err = (_vira_gpu.cuLaunchKernel(_vira_gpu.cu_k_rope, gx, gy, 1, 16, 16, 1,
                                    0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
              ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
  } else {
    err = _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 0, sizeof(vira_cl_mem), &q_slot->dev_buf);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 1, sizeof(vira_cl_mem), &y_slot->dev_buf);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 2, sizeof(int32_t), &rows);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 3, sizeof(int32_t), &cols);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 4, sizeof(int32_t), &half_dim);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 5, sizeof(float), &pos_f);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_rope, 6, sizeof(float), &freq_base);
    size_t l_work[2] = {16, 16};
    size_t g_work[2] = {((size_t)half_dim + 15) & ~15ULL, ((size_t)rows + 15) & ~15ULL};
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, _vira_gpu.kernel_rope, 2,
                                             NULL, g_work, l_work, 0, NULL, NULL);
    }
  }

  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(q.n_dims, q.dims);
  if (err == VIRA_CL_SUCCESS) {
    if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
      if (_vira_gpu.cuMemcpyDtoH(out.data, y_slot->cu_buf, q_bytes) != VIRA_CUDA_SUCCESS) {
        err = VIRA_CL_ERR_NO_ROOM;
      }
    } else {
      err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_slot->dev_buf, 1, 0, q_bytes,
                                          out.data, 0, NULL, NULL);
    }
  }

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_rope(q, pos, freq_base);
  }

  return out;
}

static inline vira_tensor_f32 vira_gpu_softmax(vira_tensor_f32 logits) {
  vira_gpu_init();
  if (!_vira_gpu.available || !_vira_gpu.enabled || _vira_gpu.simulate_oom ||
      logits.data == NULL || logits.n_dims < 1) {
    if (_vira_gpu.simulate_oom) {
      __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    }
    return vira_tensor_softmax(logits);
  }

  int32_t cols = logits.dims[logits.n_dims - 1];
  int32_t rows = 1;
  for (int32_t i = 0; i < logits.n_dims - 1; ++i) {
    rows *= logits.dims[i];
  }
  if (cols <= 0 || rows <= 0) {
    return vira_tensor_softmax(logits);
  }

  pthread_mutex_lock(&_vira_gpu.lock);
  const size_t bytes = (size_t)rows * (size_t)cols * sizeof(vira_f32);

  vira_cl_int err = 0;
  bool l_hit = false;
  _vira_gpu_act_entry *l_slot = _vira_gpu_get_or_upload_activation_locked(
      logits.data, bytes, &l_hit, &err);
  if (err != VIRA_CL_SUCCESS || l_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_softmax(logits);
  }

  _vira_gpu_act_entry *y_slot = _vira_gpu_alloc_activation_slot_locked(
      bytes, l_slot->dev_buf, NULL, l_slot->cu_buf, 0, &err);
  if (err != VIRA_CL_SUCCESS || y_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_softmax(logits);
  }

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    void *args[] = {&l_slot->cu_buf, &y_slot->cu_buf, &rows, &cols};
    err = (_vira_gpu.cuLaunchKernel(_vira_gpu.cu_k_softmax, (unsigned int)rows, 1, 1,
                                    _vira_gpu.wg_size, 1, 1, 0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
              ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
  } else {
    err = _vira_gpu.clSetKernelArg(_vira_gpu.kernel_softmax, 0, sizeof(vira_cl_mem), &l_slot->dev_buf);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_softmax, 1, sizeof(vira_cl_mem), &y_slot->dev_buf);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_softmax, 2, sizeof(int32_t), &rows);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_softmax, 3, sizeof(int32_t), &cols);
    size_t l_work = (size_t)_vira_gpu.wg_size;
    size_t g_work = (size_t)rows * l_work;
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, _vira_gpu.kernel_softmax, 1,
                                             NULL, &g_work, &l_work, 0, NULL, NULL);
    }
  }

  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(logits.n_dims, logits.dims);
  if (err == VIRA_CL_SUCCESS) {
    if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
      if (_vira_gpu.cuMemcpyDtoH(out.data, y_slot->cu_buf, bytes) != VIRA_CUDA_SUCCESS) {
        err = VIRA_CL_ERR_NO_ROOM;
      }
    } else {
      err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_slot->dev_buf, 1, 0, bytes,
                                          out.data, 0, NULL, NULL);
    }
  }

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_softmax(logits);
  }

  return out;
}

static inline vira_tensor_f32 vira_gpu_silu(vira_tensor_f32 x) {
  vira_gpu_init();
  if (!_vira_gpu.available || !_vira_gpu.enabled || _vira_gpu.simulate_oom ||
      x.data == NULL || x.n_dims < 1) {
    if (_vira_gpu.simulate_oom) {
      __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    }
    return vira_tensor_silu(x);
  }

  int32_t total = 1;
  for (int32_t i = 0; i < x.n_dims; ++i) {
    total *= x.dims[i];
  }
  if (total <= 0) {
    return vira_tensor_silu(x);
  }

  pthread_mutex_lock(&_vira_gpu.lock);
  const size_t bytes = (size_t)total * sizeof(vira_f32);

  vira_cl_int err = 0;
  bool x_hit = false;
  _vira_gpu_act_entry *x_slot = _vira_gpu_get_or_upload_activation_locked(
      x.data, bytes, &x_hit, &err);
  if (err != VIRA_CL_SUCCESS || x_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_silu(x);
  }

  _vira_gpu_act_entry *y_slot = _vira_gpu_alloc_activation_slot_locked(
      bytes, x_slot->dev_buf, NULL, x_slot->cu_buf, 0, &err);
  if (err != VIRA_CL_SUCCESS || y_slot == NULL) {
    pthread_mutex_unlock(&_vira_gpu.lock);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_silu(x);
  }

  if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
    void *args[] = {&x_slot->cu_buf, &y_slot->cu_buf, &total};
    unsigned int gx = (unsigned int)(((size_t)total + _vira_gpu.wg_size - 1) / _vira_gpu.wg_size);
    err = (_vira_gpu.cuLaunchKernel(_vira_gpu.cu_k_silu, gx, 1, 1,
                                    _vira_gpu.wg_size, 1, 1, 0, _vira_gpu.cu_stream, args, NULL) == VIRA_CUDA_SUCCESS)
              ? VIRA_CL_SUCCESS : VIRA_CL_ERR_NO_ROOM;
  } else {
    err = _vira_gpu.clSetKernelArg(_vira_gpu.kernel_silu, 0, sizeof(vira_cl_mem), &x_slot->dev_buf);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_silu, 1, sizeof(vira_cl_mem), &y_slot->dev_buf);
    err |= _vira_gpu.clSetKernelArg(_vira_gpu.kernel_silu, 2, sizeof(int32_t), &total);
    size_t l_work = (size_t)_vira_gpu.wg_size;
    size_t g_work = (((size_t)total + l_work - 1) / l_work) * l_work;
    if (err == VIRA_CL_SUCCESS) {
      err = _vira_gpu.clEnqueueNDRangeKernel(_vira_gpu.queue, _vira_gpu.kernel_silu, 1,
                                             NULL, &g_work, &l_work, 0, NULL, NULL);
    }
  }

  vira_tensor_f32 out = vira_tensor_f32_alloc_nd(x.n_dims, x.dims);
  if (err == VIRA_CL_SUCCESS) {
    if (_vira_gpu.active_backend == VIRA_GPU_BACKEND_CUDA) {
      if (_vira_gpu.cuMemcpyDtoH(out.data, y_slot->cu_buf, bytes) != VIRA_CUDA_SUCCESS) {
        err = VIRA_CL_ERR_NO_ROOM;
      }
    } else {
      err = _vira_gpu.clEnqueueReadBuffer(_vira_gpu.queue, y_slot->dev_buf, 1, 0, bytes,
                                          out.data, 0, NULL, NULL);
    }
  }

  if (err == VIRA_CL_SUCCESS) {
    y_slot->host_ptr = out.data;
  }
  pthread_mutex_unlock(&_vira_gpu.lock);

  if (err != VIRA_CL_SUCCESS) {
    vira_tensor_f32_free(&out);
    __atomic_add_fetch(&_vira_gpu.oom_fallbacks, 1, __ATOMIC_RELAXED);
    return vira_tensor_silu(x);
  }

  return out;
}

#endif // VIRA_GPU_H
