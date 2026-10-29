# Tutorial 13: Hardware Acceleration, GPUs, NPUs & Heterogeneous Computing

> **Difficulty:** Advanced  
> **Matching Code Example:** [`docs/examples/13_hardware_acceleration_gpu.vir`](../examples/13_hardware_acceleration_gpu.vir)

---

## 1. The Heterogeneous Compute Landscape

Modern high-performance computing and machine learning no longer run on single-architecture CPUs. Production systems run on **Heterogeneous Hardware**:

```
+───────────────────────────────────────────────────────────────────────────────+
│                               HOST SYSTEM                                     │
│                                                                               │
│  Host CPU (AVX2 / AVX-512)     Discrete GPU (NVIDIA CUDA)     Apple Silicon   │
│   ├── 8x f32 / Cycle (SIMD)     ├── 10,000+ CUDA Cores         ├── Unified DRAM│
│   ├── Low-Latency Cache         ├── Tensor Cores (FP16/BF16)   ├── Metal GPU   │
│   └── Small-to-Mid Batches      └── High-Throughput Matrix Math└── Neural Eng. │
+───────────────────────────────────────────────────────────────────────────────+
```

When building high-throughput AI pipelines, developers face three major hardware classes:
1. **Host CPU SIMD (AVX2, AVX-512, ARM NEON, AMX):** Ideal for low-latency, small-batch inferences (batch size 1–32) where PCIe transfer latency would dwarf computation time.
2. **Discrete GPUs (NVIDIA CUDA, AMD ROCm / HIP):** Designed for massive matrix parallelism with thousands of SIMT cores and dedicated High Bandwidth Memory (HBM).
3. **Unified Memory Architectures (Apple Silicon Metal, NVIDIA Grace Hopper):** Where CPU, GPU, and NPU share the **same physical DRAM address space**, eliminating PCIe copies entirely.
4. **Dedicated NPUs / TPUs (Qualcomm Hexagon, Intel NPU, Google TPU):** Fixed-function ASICs optimized for low-power matrix convolutions.

---

## 2. The Traditional GPU Friction in Python & Go

Languages like Python and Go struggle with hardware acceleration due to fundamental architectural mismatches:

### The "Pointer-Scattering" Problem
In Python and Go, multi-dimensional structures are implemented as slices of slices or lists of lists (`[][]float32` or `list[list[float]]`). 
- Memory is fragmented into thousands of scattered heap pointers.
- To send this data to a GPU over PCIe, the runtime must perform an expensive **serialization / flattening copy** into a contiguous buffer before calling `cudaMemcpy`.

### The FFI & GIL Latency Tax
In Python, invoking a CUDA kernel requires crossing the CPython C-API, checking GIL locks, and converting Python PyObject pointers. For fast inference loops, **the FFI overhead can exceed the GPU kernel execution time**.

### The Fatal CUDA OOM Problem
In standard C++ or PyTorch servers, if a batch exceeds GPU VRAM, CUDA raises a fatal `CUDA_ERROR_OUT_OF_MEMORY` or `CUDA error: an illegal memory access was encountered`. This **aborts the entire process**, killing active user connections and crashing the deployment.

---

## 3. How Vira Solves the Hardware Interop Problem

> [!IMPORTANT]
> **CPU SIMD Built-in vs. C99 FFI Accelerator Offload:** Vira's single-header runtime (`runtime/vira_rt.h`) is an honest, **100% zero-dependency multi-threaded CPU SIMD engine** (`AVX2+FMA`, `SSE2`, `ARM NEON`)—it never requires CUDA, ROCm, or OpenCL drivers to compile or run. Discrete GPU/NPU kernels (`cuBLAS`, `hipBLAS`, `Metal`, `OpenCL`) integrate cleanly as external C99 libraries via `@[intrinsic: '...']`, 64-byte aligned `tensor.data` pointers, and `vira_str_cstr` FFI helpers, supervised by Vira actors as demonstrated below.

Vira addresses hardware acceleration at the type system, memory layout, and concurrency levels:

### 1. Strict 64-Byte Cache-Line Alignment & 1D–4D Strided Layout
Every `tensor[f32]` in Vira is allocated on a 64-byte aligned boundary with explicit 1D–4D shape and stride metadata:

```c
// Vira runtime 1D-4D strided tensor layout (runtime/vira_rt.h)
typedef struct {
    vira_f32* data;       // 64-byte cache-line aligned pointer
    void*     raw_alloc;  // underlying pointer for free()
    int32_t   dims[4];    // 1D-4D shape extents (e.g. Batch, Heads, Seq, Dim)
    int32_t   strides[4]; // row-major element strides
    int32_t   n_dims;     // active rank (1..4)
    int32_t   rows;       // 2D leading-dimension compatibility view
    int32_t   cols;       // 2D trailing-dimension compatibility view
    bool      is_arena;   // bump-arena lifetime flag
    bool      is_view;    // O(1) zero-copy arena reshape view flag
} vira_tensor_f32;
```

- **Direct Memory Access (DMA) Ready:** 64-byte alignment matches PCIe packet boundaries and GPU cache line widths.
- **Zero-Copy C-ABI Handoff:** The raw data pointer (`tensor.data`) can be passed directly to `cudaMemcpyAsync()`, `hipMemcpy()`, or Metal's `newBufferWithBytesNoCopy()` without copying a single byte.

### 2. Zero-Copy on Unified Memory (Apple Silicon & Grace Hopper)
On modern unified architectures, the CPU and GPU share the same physical memory bus:
- Vira bump arenas (`arena { ... }`) allocate memory directly on the unified bus.
- CPU vector units (ARM NEON) and GPU compute kernels can read and write the **exact same memory buffer** with zero transfer latency.

### 3. Erlang-Style Fault Containment for GPU OOM
In Vira, GPU device offloading is managed by **Supervised Actors**:

```v
pub actor DeviceQueueWorker {
pub:
	device_id int
	target_hardware string
mut:
	jobs_dispatched int = 0
}

pub fn (mut worker DeviceQueueWorker) execute_kernel(batch_multiplier f32, force_oom bool) f32 {
	worker.jobs_dispatched = worker.jobs_dispatched + 1

	if force_oom {
		panic('CUDA_ERROR_OUT_OF_MEMORY: device memory allocation failed on Device #${worker.device_id}')
	}

	mut output_metric := 0.0f32
	arena {
		inputs := tensor_fill(32, 64, batch_multiplier)
		weights := tensor_fill(64, 32, 0.05)
		result := inputs @ weights
		output_metric = tensor_sum(result)
	}
	return output_metric
}
```

```
[GPU Worker Thread]
      │
      ├──> execute_kernel(force_oom=true)
      │      └── panic('CUDA_ERROR_OUT_OF_MEMORY')
      │
      ▼
[Supervisor Catch Frame]
      ├── 1. Intercepts GPU panic without crashing the OS process
      ├── 2. Wipes actor's isolated scratch arena in O(1)
      ├── 3. Resets device queue state (OneForOne policy)
      └── 4. Returns typed error Result to awaiting caller: Result{ .is_err = true }
```

When a GPU worker runs out of memory, **only that worker's task is failed**; the supervisor resets the device state, and peer workers continue executing without interruption.

---

## 4. Multi-Stream Asynchronous Offloading

Vira integrates device execution with its native non-blocking send (`<-`) and work-stealing scheduler:

```v
mut gpu_node_0 := DeviceQueueWorker{ device_id: 0, target_hardware: 'GPU_CUDA', jobs_dispatched: 0 }
mut gpu_node_1 := DeviceQueueWorker{ device_id: 1, target_hardware: 'GPU_CUDA', jobs_dispatched: 0 }
mut npu_node   := DeviceQueueWorker{ device_id: 2, target_hardware: 'NPU_ACCEL', jobs_dispatched: 0 }

// Dispatch compute jobs to discrete device streams asynchronously
fut_gpu0 := gpu_node_0 <- execute_kernel(1.0, false)
fut_gpu1 := gpu_node_1 <- execute_kernel(2.0, true)  // Injected fault
fut_npu  := npu_node   <- execute_kernel(0.5, false)

// Synchronize streams via .await()
res_gpu0 := fut_gpu0.await() or { 0.0 }

// Trapped hardware fault handled gracefully with zero pipeline stall
res_gpu1 := fut_gpu1.await() or {
	println('Supervisor intercepted hardware fault -> "${err}"')
	0.0 // Fallback prevents server crash
}

res_npu := fut_npu.await() or { 0.0 }
```

While the host thread waits for the GPU kernel to finish via `.await()`, Vira's Chase-Lev scheduler does not put the thread to sleep; instead, it steals and runs other CPU tasks while the GPU compute queue drains.

---

## 5. Heterogeneous Execution Strategy: When to Use CPU vs. GPU

| Metric / Scenario | **Host CPU SIMD (AVX2 / NEON)** | **Discrete GPU (CUDA / ROCm)** |
| :--- | :--- | :--- |
| **Optimal Batch Size** | 1 to 64 | 64 to 10,000+ |
| **Transfer Latency** | **Zero** (already in L1/L2/L3 cache) | 10 to 50 microseconds PCIe latency |
| **Throughput** | 0.5 – 3 TFLOPs | 30 – 120+ TFLOPs |
| **Best Used For** | Real-time audio, robotics, embeddings, latency-critical inference | Large LLM training, batch image processing, massive matrix convolutions |
| **Failure Modes** | Extremely rare (handled via supervisor) | Frequent (OOM, driver timeouts, bus resets) |

---

## 6. Running the Example Program

Run the verified heterogeneous acceleration example:

```powershell
.\vira.exe run docs/examples/13_hardware_acceleration_gpu.vir
```

### Execution Output
```
=================================================================
   VIRA HETEROGENEOUS HARDWARE ACCELERATION & GPU COMPUTING       
=================================================================
--- 1. ACCELERATOR PROFILES & ZERO-COPY BUFFER ALIGNMENT ---
Detected Hardware Compute Targets:
  [1] Host CPU (AVX2 + FMA) -> 256-bit vector registers
  [2] NVIDIA RTX / Tensor Core -> Target: GPU_CUDA (65.00 TFLOPs)
  [3] Apple Silicon GPU / Neural Engine -> Target: GPU_METAL (Unified DMA)
Zero-Copy DMA Ready Memory Architecture:
  * All Vira tensors are aligned to strict 64-byte cache lines.
  * Memory pointers pass directly to CUDA, ROCm, Metal, or GGML without marshalling.
--- 2. ASYNCHRONOUS DEVICE WORKERS & STREAMS ---
--- 3. SYNCHRONIZING DEVICE STREAMS & FAULT RECOVERY ---
[vira_supervisor | worker_thread=0] Actor 'DeviceQueueWorker' (#1) trapped fault ('CUDA_ERROR_OUT_OF_MEMORY: device memory allocation failed on Device #1') -> O(1) arena wipe & OneForOne restart [1/3]
GPU Device #0 Output: 3276.80
Supervisor intercepted hardware fault -> "CUDA_ERROR_OUT_OF_MEMORY: device memory allocation failed on Device #1"
GPU Device #1 Result (Handled Safely): 0.00
NPU Compute Engine Output: 1638.40
--- 4. POST-RESTART GPU STREAM RE-ATTACHMENT ---
GPU Device #1 recovered after supervisor restart: 6553.61
=================================================================
   END OF HARDWARE ACCELERATION DEMO                             
=================================================================
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vira_rt] pool_workers=4 | actor_tasks_run=4 | work_steals=3
[vira_rt] active_leaks=0 | total_freed=18
```

---

## 7. The Vira Hardware Acceleration Roadmap

Vira's hardware roadmap is structured in three progressive tiers:

```
Tier 1: Host SIMD (Current)      Tier 2: Zero-Copy C-ABI (Current)    Tier 3: Native Device Directives (Roadmap)
┌───────────────────────────┐    ┌───────────────────────────┐         ┌───────────────────────────┐
│ • 256-bit AVX2 + FMA      │    │ • Direct C pointers       │         │ • tensor[f32, device: cuda]│
│ • 128-bit ARM NEON        │ ──►│ • cuBLAS / cuDNN bindings │ ──────► │ • Native kernel dispatch  │
│ • 64-byte cache alignment │    │ • Metal MPS & GGML bridges│         │ • Automatic multi-GPU     │
│ • Zero external deps      │    │ • Zero marshalling        │         │   sharding & pipeline     │
└───────────────────────────┘    └───────────────────────────┘         └───────────────────────────┘
```
