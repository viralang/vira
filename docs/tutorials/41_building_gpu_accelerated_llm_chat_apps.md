# Tutorial 41: Building GPU-Accelerated LLM Chat Applications — Zero-SDK CUDA Driver Backend, Tensor Core (WMMA) Specialization, CLI & Native 60 FPS Desktop Studio

In this tutorial, you will learn how to build production-grade, hardware-accelerated Large Language Model (LLM) applications in 100% pure Vira with **zero external C FFI bindings (`extern "C"` or `unsafe`)** and **zero compile-time SDK dependencies**.

We will explore two complete, end-to-end applications powered by Alibaba Cloud's 24-layer, 490M-parameter **Qwen 2.5 (0.5B-Instruct)** model:
1. **Interactive CLI Chat Engine** ([`43_qwen2_5_gguf_chat_inference.vir`](../examples/43_qwen2_5_gguf_chat_inference.vir)): High-throughput terminal inference with ChatML formatting, stateful 24-layer KV cache, temperature sampling, and multi-turn REPL.
2. **Native 60 FPS Desktop Chat Studio** ([`45_gui_qwen2_5_gguf_chat_studio.vir`](../examples/45_gui_qwen2_5_gguf_chat_studio.vir)): Real-time immediate-mode GUI with non-blocking frame-by-frame layer streaming, live autoregressive canvas drawing, and hardware telemetry.

---

## 1. The Vira Zero-SDK GPU Architecture

Traditional GPU programming requires installing gigabytes of SDKs (NVIDIA CUDA Toolkit, AMD ROCm, Vulkan SDK), configuring platform compilers (`nvcc`), and linking against platform-specific libraries (`-lcuda`, `-lcudart`).

Vira takes a radically different approach: **Compile-Once, Accelerate Anywhere**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Vira Source (.vir) -> Native Binary                  │
│   (Compiled on any machine: Mac, Linux, Windows, with or without GPU)  │
│      - 18 JIT PTX Kernels embedded in .rodata as constant strings      │
│      - Secondary sm_70+ Tensor Core (WMMA) PTX module embedded         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Runtime Dynamic Probing
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ 1. NVIDIA CUDA  │        │   2. OpenCL     │        │ 3. CPU SIMD     │
│   Driver API    │        │   Driver API    │        │    Fallback     │
│  libcuda.so.1   │        │   libOpenCL.so  │        │ AVX2+FMA / NEON │
│   nvcuda.dll    │        │    OpenCL.dll   │        │ 16-Worker Pool  │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ Tensor Cores    │        │ High-Compute    │        │ Fault-Tolerant  │
│ WMMA (sm_70+)   │        │ GPU Workgroups  │        │ OOM Supervision │
│ 4x - 8x Speedup │        │ Unified DMA     │        │ active_leaks=0  │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

### Key Architectural Pillars:
1. **Zero Compile-Time SDK Dependencies:**
   Vira resolves GPU drivers dynamically at runtime via `dlopen` / `LoadLibraryA`. You can build a binary on a budget laptop with no GPU and copy it to an NVIDIA H100 GPU server where it will immediately detect the driver, JIT-compile the embedded PTX, and run at maximum speed.
2. **Embedded JIT PTX Assembly (`sm_52` baseline & `sm_70` WMMA):**
   - **Baseline (`sm_52`):** Maxwell and Pascal GPUs execute 18 JIT-compiled kernels covering quantized GEMV (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`), FP16 GEMV, tiled GEMM, RMSNorm, RoPE, Softmax, and SiLU.
   - **Tensor Core Specialization (`sm_70`+):** Volta, Turing, Ampere, Ada Lovelace, Hopper, and Blackwell GPUs dynamically load `_vira_cuda_wmma_kernel_ptx`, utilizing hardware `wmma.mma.sync.aligned` matrix multiply instructions for batched prompt prefill ($M \ge 16$).
3. **Direct Quantized Uploads & Persistent Cache:**
   When loading GGUF weights, raw quantized byte blocks are transferred directly into VRAM without host FP32 expansion, saving **75%–85% VRAM**. The 4096-entry persistent weight cache avoids PCIe transfer overhead on repeated tokens.
4. **Activation Residency Ring:**
   Intermediate activations stay resident on GPU across consecutive transformer operators (RMSNorm $\rightarrow$ QKV Projection $\rightarrow$ RoPE $\rightarrow$ Softmax $\rightarrow$ SwiGLU), eliminating ping-pong PCIe transfers.

---

## 2. Using the `import gpu` API

Vira's standard library exposes the GPU runtime through `import gpu`:

```vira
import gpu

// 1. Hardware Discovery & Telemetry
dev := gpu.info()
println('Device: ${dev.device_name} (backend=${dev.backend_name})')
println('Hardware GPU: ${dev.available} | Enabled: ${dev.enabled}')
println('VRAM: ${gpu.allocated_kb() / 1024} MB used / ${gpu.vram_mb()} MB total')
println('Tensor Cores: ${gpu.has_tensor_cores()}')

// 2. Flexible Backend & Device Selection
gpu.set_backend(gpu.backend_cuda)  // "cuda", "opencl", "cpu", "auto"
gpu.set_device('0')               // Index "0" or substring filter like "RTX"

// 3. High-Performance Tensor Operations
x := tensor_fill(128, 512, 0.5)
w := tensor_fill(1024, 512, 0.25)

// Linear projection with persistent weight caching: y = x @ w.T
y := x.gpu_linear(w)

// Transformer building blocks
norm := gpu.rmsnorm(y, weight_tensor, 0.00001)
rotated := gpu.rope(norm, pos, 1000000.0)
activated := gpu.silu(rotated)
probs := gpu.softmax(activated)

gpu.sync()
```

---

## 3. Building the CLI Chat Engine (Example 43)

Let's examine how [`docs/examples/43_qwen2_5_gguf_chat_inference.vir`](../examples/43_qwen2_5_gguf_chat_inference.vir) is structured:

### Step 1: Flexible CLI Options & Help System
The application allows fine-grained control over computation, sampling, and model files:

```vira
cli_args := os.args()
mut a_idx := 1
for a_idx < cli_args.len {
    arg := cli_args[a_idx]
    if arg == '-h' || arg == '--help' {
        println('Usage: vira run docs/examples/43_qwen2_5_gguf_chat_inference.vir [options]')
        println('  -i, --interactive, --repl    Launch multi-turn interactive ChatML REPL')
        println('  --gpu                        Enable hardware GPU compute acceleration')
        println('  --cpu                        Force CPU SIMD compute engine')
        println('  --gpu-backend <backend>      Force backend: "cuda", "opencl", "cpu", or "auto"')
        println('  --gpu-device <filter>        Select GPU device by index ("0") or name filter ("RTX")')
        return
    } else if arg == '--gpu-backend' && a_idx + 1 < cli_args.len {
        gpu.set_backend(cli_args[a_idx + 1])
        a_idx += 1
    } else if arg == '--gpu-device' && a_idx + 1 < cli_args.len {
        gpu.set_device(cli_args[a_idx + 1])
        a_idx += 1
    }
    a_idx += 1
}
```

### Step 2: Auto-Download from Hugging Face
If the GGUF model is not present on disk, Vira automatically downloads the official quantized model via `curl` into `models/`:

```vira
if !os.exists(model_path) {
    if !os.is_dir('models') {
        _ := os.mkdir_all('models', 493) or { false }
    }
    mut dl_cmd := exec.Command.new('curl', ['-L', '-s', '-o', model_path, hf_url])
    dl_code := dl_cmd.run() or { os.exit(1) }
}
```

### Step 3: Fast GGUF Header & ChatML Vocab Parsing
Using `os.read_bytes_head(model_path, 6500000)`, the engine reads only the first 6.5 MB of the model file to parse all metadata and the 151,936-token vocabulary into an $O(1)$ 65,536-bucket hash table in under **30 ms**.

### Step 4: Full 24-Layer Transformer Block with QKV Attention Biases
Qwen 2.5 uses Attention Biases (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`), Grouped-Query Attention (14 Query heads / 2 KV heads, 7:1 ratio), High-Frequency NeoX RoPE (`base=1,000,000`), and SwiGLU FFN:

```vira
// QKV projections with Qwen2 Attention Biases
q_lin := xn.gpu_linear(l_wq[l]) + l_bq[l]
k_lin := xn.gpu_linear(l_wk[l]) + l_bk[l]
v_lin := xn.gpu_linear(l_wv[l]) + l_bv[l]

// SwiGLU Feed-Forward Network
gate := tensor_silu(x2n.gpu_linear(l_w_gate[l]))
up := x2n.gpu_linear(l_w_up[l])
ffn_hidden := gate * up
down := ffn_hidden.gpu_linear(l_w_down[l])
```

### Step 5: Autoregressive Decode with Bump Arena Memory Isolation
By wrapping each token generation step in an `arena { ... }` block, all intermediate activations are allocated on an $O(1)$ bump allocator that resets in 1 clock cycle at the end of each token. This guarantees **zero heap fragmentation** and **`active_leaks=0`** throughout thousands of generation turns.

---

## 4. Building the Native 60 FPS Desktop Chat Studio (Example 45)

[`docs/examples/45_gui_qwen2_5_gguf_chat_studio.vir`](../examples/45_gui_qwen2_5_gguf_chat_studio.vir) demonstrates how to marry heavy tensor inference with a smooth 60 FPS immediate-mode desktop GUI.

```
┌────────────────────────────────────────────────────────────────────────┐
│ Top Bar: Model Status Badge | Active Backend (CUDA/OpenCL) | WMMA Badge │
├───────────────────────────────────┬────────────────────────────────────┤
│ Left Column: Controls (460 px)    │ Right Column: Live Chat (512 px)   │
│  - GGUF File Path Input           │  - ChatML Transcript Turns         │
│  - System Prompt Input            │  - Live Autoregressive Output      │
│  - GPU Offload Toggle Checkbox    │  - KV Cache Progress Indicator     │
│  - Temperature & Max-Tokens       │  - Architecture & Throughput tok/s │
│  - Quick Prompt Preset Buttons    │  - VRAM Budget & Allocation Bar    │
└───────────────────────────────────┴────────────────────────────────────┘
```

### 1. Non-Blocking Frame-by-Frame Layer Streaming
In traditional applications, loading a 490 MB model on startup causes the OS window to hang and trigger "Not Responding" warnings.

In Vira, we use a frame-driven state machine:
* `engine_state == 1`: Load & parse metadata header (frame 1).
* `engine_state == 2`: Load **1 transformer layer per frame** (`load_layer_idx++`).
* Each frame renders a live progress bar: `progress_pct = float(load_layer_idx) / 24.0`.
* The GUI stays completely interactive and responsive at 60 FPS while the model streams into memory!

```vira
if engine_state == 2 {
    if load_layer_idx < meta.n_layers {
        l := load_layer_idx
        status_msg = 'Streaming layer ${l + 1}/${meta.n_layers}...'
        l_an.push(load_gguf_or_die(model_path, 'blk.${l}.attn_norm.weight'))
        l_wq.push(load_gguf_or_die(model_path, 'blk.${l}.attn_q.weight'))
        // ... load remaining layer tensors ...
        load_layer_idx += 1
        progress_pct = f32(load_layer_idx) / f32(meta.n_layers)
    } else {
        engine_state = 3 // Model Ready!
    }
}
```

### 2. Live Autoregressive Token-by-Token Streaming
When the user sends a message:
* `engine_state = 4`: Prefills the ChatML prompt into the KV cache.
* `engine_state = 5`: Generates **1 token per frame**. Decodes the token BPE string and immediately appends it to `chat_reply_2`. The user sees words typing out smoothly onto the canvas in real time.

### 3. Reactive Hardware Badges
The GUI queries telemetry every frame to display live hardware state:

```vira
if use_gpu && gpu.is_available() {
    cached_mb := gpu.allocated_kb() / 1024
    be_name := gpu.backend_name()
    gui_draw_badge(720, 16, '${be_name} (${cached_mb}MB)', 0x2D4A3E, 0xA6E3A1)
    if gpu.has_tensor_cores() {
        gui_draw_badge(880, 16, 'WMMA CORES', 0x1E3A4A, 0x89DCEB)
    }
} else {
    gui_draw_badge(840, 16, 'CPU SIMD', 0x313244, 0x89B4FA)
}
```

### 4. Dual-Column Independent Scroll Containers & Auto-Wrapping Text
To maintain a responsive 60 FPS UI on screens of any size and prevent multi-line LLM responses from spilling off-canvas:
* **Scroll Containers (`gui_scroll_begin` / `gui_scroll_end`):** Both the Left Column (controls and presets) and the Right Column (chat transcript and telemetry) are wrapped in independent scroll viewports with mouse wheel support, interactive drag thumbs, and hardware scissor clipping.
* **Auto-Wrapping Labels (`gui_lbl`):** Long prompts and streaming replies automatically word-wrap to the container interior width (`node->width`), expanding the card height dynamically without overlapping or escaping the window.

```vira
scroll_h := gui_window_height() - 88

// Right Column: Live Streaming ChatML Transcript & Diagnostics (Scrollable)
gui_vstack_begin(504, 72, right_w, 12)
gui_scroll_begin(right_w, scroll_h)
    gui_card_begin('Live Streaming ChatML Transcript', right_w - 12)
        gui_lbl('User: ${chat_user_1}')
        gui_lbl('Qwen2.5: ${chat_reply_1}')
    gui_card_end()

    gui_spacing(8)

    gui_card_begin('Transformer & KV-Cache Diagnostic Telemetry', right_w - 12)
        gui_lbl('Compute Engine: ${gpu.device_name()} (${gpu.backend_name()})')
    gui_card_end()
gui_scroll_end()
gui_stack_end()
```

---

## 5. Running and Deploying Your Chat Applications

### Running the CLI Engine
```bash
# 1. Automatic GPU discovery (uses NVIDIA CUDA if present, else OpenCL, else CPU)
vira run docs/examples/43_qwen2_5_gguf_chat_inference.vir -p "Explain quantum computing briefly."

# 2. Force CUDA Driver API on device 0
vira run docs/examples/43_qwen2_5_gguf_chat_inference.vir --gpu-backend cuda --gpu-device 0

# 3. Launch interactive ChatML REPL
vira run docs/examples/43_qwen2_5_gguf_chat_inference.vir -i
```

### Running the Desktop Chat Studio
```bash
# Launch the 60 FPS GUI Studio
vira run docs/examples/45_gui_qwen2_5_gguf_chat_studio.vir

# Launch forced on NVIDIA CUDA
vira run docs/examples/45_gui_qwen2_5_gguf_chat_studio.vir --gpu-backend cuda
```

### Building Standalone Production Binaries
Compile standalone, optimized native binaries with zero external shared library dependencies:

```bash
# Compile optimized CLI binary
vira build -prod -o qwen_chat docs/examples/43_qwen2_5_gguf_chat_inference.vir

# Compile optimized GUI binary
vira build -prod -o qwen_studio docs/examples/45_gui_qwen2_5_gguf_chat_studio.vir

# Run standalone
./qwen_chat -i
./qwen_studio
```

---

## 6. Summary: Building Custom AI Applications in Vira

| Need | Vira Solution | Benefit |
| :--- | :--- | :--- |
| **Model Ingestion** | `gguf_load_tensor(path, name)` | Direct binary parser for Q4_0, Q8_0, FP16, and FP32 weights |
| **GPU Acceleration** | `x.gpu_linear(w)` / `gpu.matmul(a, b)` | Zero-SDK CUDA Driver API & OpenCL with persistent caching |
| **Tensor Cores** | `gpu.has_tensor_cores()` & WMMA kernels | 4x to 8x acceleration on Volta, Turing, Ampere, Ada, Hopper |
| **Memory Isolation** | `arena { ... }` | $O(1)$ bump allocator reset per token step, verified `active_leaks=0` |
| **Desktop UI** | Immediate-mode `gui_begin()` / `gui_end()` | 60 FPS reactive desktop interface with zero framework bloat |
| **Concurrency** | `actor Worker` with fault supervision | Automatic CPU SIMD fallback if GPU runs out of VRAM |

