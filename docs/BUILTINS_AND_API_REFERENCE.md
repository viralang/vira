# Vira Standard Library & Builtin API Reference

Welcome to the definitive API reference for the **Vira Programming Language**. This document describes every built-in function, data type, operator, and runtime primitive available out-of-the-box in Vira with exact type signatures, behavioral details, memory ownership rules, and executable code snippets.

---

## Table of Contents

1. [Core Runtime & Diagnostics](#1-core-runtime--diagnostics)
2. [AI Tensors (1D–4D) & Multi-Threaded CPU SIMD Math](#2-ai-tensors-1d4d--multi-threaded-cpu-simd-math)
3. [Autograd Backpropagation & Optimizers](#3-autograd-backpropagation--optimizers)
4. [SafeTensors & GGUF Local LLM Inference](#4-safetensors--gguf-local-llm-inference)
5. [Actor Concurrency & Supervisors](#5-actor-concurrency--supervisors)
6. [Native File I/O & Filesystem](#6-native-file-io--filesystem)
7. [Networking & HTTP 1.1 Web Engine](#7-networking--http-11-web-engine)
8. [RFC 8259 JSON (`import json`) & Fast JSON Builtins](#8-rfc-8259-json-import-json--fast-json-builtins)
9. [Native GUI & Declarative Layout Engine](#9-native-gui--declarative-layout-engine)
10. [Memory, Bump Arenas, Autofree & C FFI Helpers](#10-memory-bump-arenas-autofree--c-ffi-helpers)
11. [Native Collections (`[]T` & `[N]T`) & String Methods](#11-native-collections-t--nt--string-methods)
12. [Standard Library Packages (`io`, `net`, `http`, `json`, `viraserve`)](#12-standard-library-packages-io-net-http-json-viraserve)

---

## 1. Core Runtime & Diagnostics

These functions are available globally in all modules without imports.

### `println(msg string) void`
Prints a UTF-8 string followed by a newline character to standard output (`stdout`).
- **Memory Rule:** The `msg` argument is borrowed. If constructed via string interpolation `${...}`, it is automatically hoisted and freed deterministically upon statement exit.
```v
println('Server started on port ${port}')
```

### `print(msg string) void`
Prints a UTF-8 string without a trailing newline to standard output (`stdout`).
```v
print('Loading assets... ')
```

### `eprintln(msg string) void`
Prints a UTF-8 string followed by a newline to standard error (`stderr`).
```v
eprintln('ERROR: Connection reset by peer')
```

### `eprint(msg string) void`
Prints a UTF-8 string without a trailing newline to standard error (`stderr`).
```v
eprint('Warning: ')
```

### `time_sleep_ms(ms int) void`
Suspends execution of the calling thread for `ms` milliseconds.
```v
time_sleep_ms(100)
```

### `const` / `pub const` Declaration
Defines compile-time immutable typed constants. Available at module and local scope.
```v
pub const BUFFER_SIZE = 4096
const DEFAULT_HOST = '127.0.0.1'
```

### `defer` Statement
Defines a cleanup expression or statement that executes unconditionally when the enclosing function or block scope exits, evaluated in strict **LIFO** (Last-In, First-Out) order.
```v
mut f := io.File.open('data.bin', 'rb') or { return }
defer f.close()
```

### `panic(msg string) void`
Triggers an immediate fault.
- **Inside an Actor:** Trapped safely by the worker thread's `setjmp`/`longjmp` fault boundary. Increments the actor's restart count, wipes its isolated bump arena in $O(1)$, resets mutable fields, and informs the supervisor.
- **Outside an Actor:** Prints the error message to `stderr` and aborts the process with exit code 1.
```v
if divisor == 0 {
    panic('division by zero')
}
```

### `time_now_us() i64`
Returns the current monotonic timestamp in **microseconds** ($\mu\text{s}$). Uses Win32 `QueryPerformanceCounter` on Windows and `clock_gettime(CLOCK_MONOTONIC)` on Linux/macOS. Ideal for sub-microsecond benchmarking and latency profiling.
```v
start := time_now_us()
do_work()
elapsed := time_now_us() - start
println('Completed in ${elapsed} µs')
```

### `time_now_ms() i64`
Returns the current monotonic timestamp in **milliseconds** ($\text{ms}$).
```v
start_ms := time_now_ms()
```

---

## 2. AI Tensors (1D–4D) & Multi-Threaded CPU SIMD Math

1D–4D strided numerical tensors are first-class primitives in Vira (`tensor[f32]`, backed by `dims[4]`, `strides[4]`, `n_dims`, and 64-byte aligned buffers). The built-in tensor runtime is a **zero-dependency multi-threaded CPU SIMD engine** (`AVX2+FMA`, `SSE2`, `ARM NEON`) requiring no external GPU drivers.

### `tensor_fill(rows int, cols int, val f32) tensor[f32]`
Allocates a 2D tensor of dimensions `rows × cols` and fills every element with scalar value `val`.
- **Hardware Acceleration:** Uses `_mm256_set1_ps` and `_mm256_storeu_ps` on x86_64; uses `vdupq_n_f32` on ARM NEON.
- **Memory Ownership:** If called inside an `arena { ... }` block, allocation occurs on the bump arena with $O(1)$ slab teardown; otherwise, tracked and freed automatically by Autofree upon scope exit.
```v
weights := tensor_fill(64, 64, 0.25)
```

### `tensor_fill_3d(d0 int, d1 int, d2 int, val f32) tensor[f32]`
Allocates a 3D contiguous tensor of shape `(d0, d1, d2)` (e.g., `(Batch, Seq, Hidden)`) filled with `val`.
```v
hidden := tensor_fill_3d(2, 16, 64, 0.5)
```

### `tensor_fill_4d(d0 int, d1 int, d2 int, d3 int, val f32) tensor[f32]`
Allocates a 4D contiguous tensor of shape `(d0, d1, d2, d3)` (e.g., `(Batch, Heads, Seq, HeadDim)`) filled with `val`.
```v
q := tensor_fill_4d(2, 8, 16, 64, 0.125)
```

### N-D Tensor Properties & Receiver Methods
| Property / Method | Signature | Description |
| :--- | :--- | :--- |
| `.ndim` | `int` | Number of active dimensions (`1..4`) |
| `.rows` / `.cols` | `int` | Flattened outer rows and trailing inner dimension (`dims[ndim-1]`) |
| `.numel()` | `fn () int` | Total element count across all dimensions |
| `.dim(axis int)` | `fn (axis int) int` | Extent along `axis` (`0..ndim-1`) |
| `.stride(axis int)` | `fn (axis int) int` | Element stride along `axis` |
| `.reshape_2d(d0, d1 int)` | `fn (int, int) tensor[f32]` | Reshapes to 2D (`O(1)` zero-copy view inside `arena {}`, safe copy on heap) |
| `.reshape_3d(d0, d1, d2 int)` | `fn (int, int, int) tensor[f32]` | Reshapes to 3D preserving element count |
| `.reshape_4d(d0, d1, d2, d3 int)` | `fn (int, int, int, int) tensor[f32]` | Reshapes to 4D preserving element count |
| `.transpose()` | `fn () tensor[f32]` | Swaps the last two axes `(..., M, N) -> (..., N, M)` |
| `.clone()` | `fn () tensor[f32]` | Deep-clones tensor data and N-D shape metadata |

### Matrix Multiplication Operator: `@`
Computes hardware-accelerated 2D, batched 3D/4D, or broadcast N-D × 2D matrix multiplication $C = A @ B$.
- **Supported Shapes:**
  - **2D × 2D:** $(M, K) @ (K, N) \to (M, N)$
  - **Batched 3D / 4D:** $(B, H, M, K) @ (B, H, K, N) \to (B, H, M, N)$
  - **Broadcast N-D × 2D:** $(B, S, K) @ (K, N) \to (B, S, N)$
- **Multi-Core CPU SIMD Kernel:** 64×64 L2 cache-tiled FMA3 / SSE2 / NEON kernel; automatically parallelizes row tiles across the Chase-Lev work-stealing thread pool when $M \times K \times N \ge 128^3$.
```v
q := tensor_fill_4d(2, 4, 16, 32, 0.25)
k := tensor_fill_4d(2, 4, 16, 32, 0.5)
attn_scores := q @ k.transpose() // Shape: (2, 4, 16, 16)
```

### Element-Wise Operators (`+`, `-`, `tensor_add`, `tensor_elemwise_mul`) & Bias Broadcasting
`a + b`, `a - b`, `tensor_add(a, b)`, and `tensor_elemwise_mul(a, b)` preserve N-D shapes on exact match and also support trailing-dimension bias broadcasting (e.g., `(B, S, D) + (1, D)` or `(M, D) + (D)`).

### `tensor_sum(t tensor[f32]) f32`
Computes the scalar sum of all elements in tensor `t` using horizontal vector additions (`_mm256_add_ps` + `_mm_hadd_ps`).
```v
total := tensor_sum(y)
println('Energy sum: ${total}')
```

### `tensor_relu(t tensor[f32]) tensor[f32]`
Applies the Rectified Linear Unit activation function element-wise: $\text{ReLU}(x) = \max(0, x)$.
- **Vectorization:** Uses `_mm256_max_ps` on AVX2 or `vmaxq_f32` on NEON.
```v
activated := tensor_relu(y)
```

### `tensor_add(a tensor[f32], b tensor[f32]) tensor[f32]`
Performs element-wise tensor addition: $C_{i,j} = A_{i,j} + B_{i,j}$. Shapes must match exactly.
```v
biased := tensor_add(layer_out, bias)
```

### `tensor_scale(t tensor[f32], scale f32) tensor[f32]`
Multiplies every element in tensor `t` by a scalar float `scale`: $C_{i,j} = t_{i,j} \times \text{scale}$.
```v
normalized := tensor_scale(activated, 0.01)
```

### `tensor_load_bin(rows int, cols int, path string) !tensor[f32]`
Loads raw binary single-precision float weights from disk directly into a 64-byte aligned tensor buffer. Returns a Result (`!tensor[f32]`).
```v
weights := tensor_load_bin(128, 64, 'weights.bin') or {
    panic('Failed to load weights: ${err}')
}
```

### `tensor_save_bin(t tensor[f32], path string) !bool`
Writes the raw IEEE 754 binary floating-point contents of tensor `t` to `path`.
```v
_ := tensor_save_bin(weights, 'checkpoint.bin') or {
    eprintln('Save failed: ${err}')
    false
}
```

---

## 3. Autograd Backpropagation & Optimizers

Vira provides native reverse-mode automatic differentiation kernels for training neural networks without Python or external frameworks.

### `tensor_transpose(t tensor[f32]) tensor[f32]`
Returns a new tensor with transposed dimensions: $(M \times N) \to (N \times M)$.
```v
w_t := tensor_transpose(weights)
```

### `tensor_matmul_grad_a(grad_out tensor[f32], b tensor[f32]) tensor[f32]`
Computes the input gradient with respect to matrix $A$ in forward pass $C = A @ B$:
$$\frac{\partial L}{\partial A} = \text{grad\_out} @ B^T$$
```v
grad_a := tensor_matmul_grad_a(grad_loss, weights)
```

### `tensor_matmul_grad_b(a tensor[f32], grad_out tensor[f32]) tensor[f32]`
Computes the parameter gradient with respect to weights $B$ in forward pass $C = A @ B$:
$$\frac{\partial L}{\partial B} = A^T @ \text{grad\_out}$$
```v
grad_weights := tensor_matmul_grad_b(inputs, grad_loss)
```

### `tensor_relu_backward(input tensor[f32], grad_out tensor[f32]) tensor[f32]`
Computes the backward pass for ReLU activations:
$$\text{grad\_act}_{i} = \begin{cases} \text{grad\_out}_{i} & \text{if } \text{input}_{i} > 0 \\ 0 & \text{otherwise} \end{cases}$$
```v
grad_act := tensor_relu_backward(activations, grad_loss)
```

### `tensor_sgd_step(mut param tensor[f32], grad tensor[f32], lr f32) void`
Performs an in-place Stochastic Gradient Descent parameter update using SIMD:
$$\text{param} \leftarrow \text{param} - (\text{lr} \times \text{grad})$$
```v
tensor_sgd_step(mut weights, grad_weights, 0.01)
```

---

## 4. SafeTensors & GGUF Local LLM Inference

Vira features native binary readers for Hugging Face `.safetensors` files and quantized `.gguf` Large Language Models (Q4_0, Q8_0, FP16, FP32).

### `safetensors_load(path string, tensor_name string) !tensor[f32]`
Opens a Hugging Face `.safetensors` file, parses its 8-byte length prefix and JSON metadata header, locates `tensor_name`, reads its shape, and loads the tensor data into a 64-byte aligned SIMD buffer with zero intermediate copies.
```v
w_attn := safetensors_load('model.safetensors', 'model.layers.0.self_attn.q_proj.weight') or {
    panic('Tensor not found: ${err}')
}
```

### `gguf_load_tensor(path string, tensor_name string) !tensor[f32]`
Loads a tensor from a GGUF v2/v3 model file (e.g. LLaMA, Mistral, Qwen). Automatically handles on-the-fly SIMD dequantization from **Q4_0** (4-bit nibbles + FP16 scale) and **Q8_0** (8-bit quantized blocks + FP16 scale) into native FP32 tensors.
```v
token_emb := gguf_load_tensor('llama-3.2-1b.gguf', 'token_embd.weight') or {
    panic('GGUF load failed: ${err}')
}
```

### `tensor_rmsnorm(x tensor[f32], weight tensor[f32], eps f32) tensor[f32]`
Root Mean Square Layer Normalization used across modern LLMs (LLaMA, Gemma, Mistral):
$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}} \times w_i$$
```v
normed := tensor_rmsnorm(hidden_state, rms_weights, 1e-5)
```

### `tensor_rope(q tensor[f32], pos int, freq_base f32) tensor[f32]`
Applies Rotary Position Embeddings (RoPE) to query or key tensor `q` at token position `pos` with frequency base `freq_base` (typically $10000.0$ or $500000.0$).
```v
q_rotated := tensor_rope(q, current_pos, 10000.0)
```

### `tensor_silu(x tensor[f32]) tensor[f32]`
Computes the Sigmoid Linear Unit (SiLU / Swish) activation: $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.
```v
gate := tensor_silu(gate_proj)
```

### `tensor_elemwise_mul(a tensor[f32], b tensor[f32]) tensor[f32]`
Computes element-wise multiplication: $C_{i,j} = A_{i,j} \times B_{i,j}$. Used for SwiGLU activations: $(A @ W_{\text{gate}}) \cdot (A @ W_{\text{up}})$.
```v
swiglu := tensor_elemwise_mul(tensor_silu(gate), up)
```

### `tensor_softmax(logits tensor[f32]) tensor[f32]`
Computes numerically stable Softmax across each row:
$$\text{Softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$
```v
probs := tensor_softmax(attn_scores)
```

### `tensor_embedding(table tensor[f32], token_id int) tensor[f32]`
Extracts the 1D embedding vector for token ID `token_id` from 2D vocabulary table `table`. Returns a $1 \times D$ tensor.
```v
token_vec := tensor_embedding(vocab_table, 128000)
```

### `sample_token(logits tensor[f32], temperature f32) int`
Samples a discrete token index from probability distribution `logits`.
- If `temperature <= 0.05`: Executes greedy argmax sampling.
- If `temperature > 0.05`: Scales logits by $1/\text{temp}$, computes softmax, and samples from cumulative distribution function (CDF).
```v
next_token := sample_token(logits, 0.7)
```

---

## 5. Actor Concurrency & Supervisors

Actors provide lock-free, multicore message passing scheduled across a Chase-Lev work-stealing pool with isolated memory heaps and fault containment.

### Actor Syntax
```v
pub actor Gateway {
pub:
    id int
mut:
    requests_processed int = 0
}

// Actor methods serve as message handlers
pub fn (mut g Gateway) handle_request(req_id int) string {
    g.requests_processed = g.requests_processed + 1
    return 'OK: request ${req_id} handled'
}
```

### Non-Blocking Message Dispatch: `<-`
Sends a message to an actor's mailbox asynchronously without blocking the calling thread. Returns a `Future[T]`.
```v
mut gw := Gateway{ id: 1, requests_processed: 0 }
fut := gw <- handle_request(42) // Instant return!
```

### Future Resolution: `.await()`
Waits for the actor task to complete and unwraps the result. While waiting, the calling thread helps execute tasks stolen from the work-stealing queue.
```v
response := fut.await() or {
    println('Gateway error or panic: ${err}')
    'fallback'
}
```

### Detached Fire-and-Forget Sends
If the return value is not needed, dispatch the send as a standalone statement or discard the handle with `_ :=`:
```v
gw <- handle_request(99)       // Standalone statement send
_ := gw <- handle_request(100) // Discarded binding
```
Both forms compile to `_send_detached_` (`done = -1`). The runtime worker thread automatically frees the task envelope upon completion without creating a `Future[T]` handle or leaking memory (`active_leaks = 0`).

### Zero-Alloc Synchronous Shorthand
When an actor message is directly unwrapped with `or { ... }`, the compiler eliminates heap envelope allocation, executing synchronously with a stack-allocated envelope:
```v
res := gw <- handle_request(101) or { 'fallback' }
```

### Actor Lifecycle & Fault Containment
- **`VIRA_ACTOR_RUNNING`**: Default operational state.
- **Intermittent Fault Recovery**: Panics trigger $O(1)$ isolated scratch arena wipes and `OneForOne` state resets. A subsequent successful message execution resets consecutive `restarts = 0`.
- **`VIRA_ACTOR_STOPPED`**: If consecutive faults exceed `max_restarts` (default: 3), the actor stops permanently. Subsequent message sends are rejected with `"Actor has stopped due to unhandled faults"`.

---

## 6. Native File I/O & Filesystem

All file functions return a `!T` Result type.

### `file_exists(path string) bool`
Returns `true` if the file exists and is accessible, `false` otherwise.
```v
if file_exists('config.json') {
    println('Found configuration file')
}
```

### `read_file(path string) !string`
Reads the entire contents of a file into an autofreed heap string.
```v
content := read_file('app.log') or {
    panic('Failed to read log: ${err}')
}
```

### `write_file(path string, content string) !bool`
Writes `content` to `path`, creating the file if it does not exist or truncating it if it does.
```v
_ := write_file('output.txt', 'Job completed') or {
    eprintln('Write error: ${err}')
    false
}
```

### `append_file(path string, content string) !bool`
Appends `content` to the end of the file at `path`.
```v
_ := append_file('audit.log', '2026-10-18: Login Alice\n') or { false }
```

### `remove_file(path string) !bool`
Deletes the file at `path`.
```v
_ := remove_file('temp.lock') or { false }
```

---

## 7. Networking & HTTP 1.1 Web Engine

Built-in TCP socket primitives and HTTP 1.1 response generation.

| Function | Signature | Description |
| :--- | :--- | :--- |
| `net_listen` | `(port int) int` | Binds to `0.0.0.0:port` and starts listening. Returns a server file descriptor (`fd`). |
| `net_accept` | `(server_fd int) int` | Accepts an incoming client connection. Returns client socket `fd`. |
| `net_read` | `(client_fd int, max_bytes int) string` | Reads up to `max_bytes` into a string. |
| `net_write` | `(client_fd int, msg string) int` | Sends string bytes over socket. Returns byte count sent. |
| `net_close` | `(fd int) void` | Closes socket descriptor. |
| `http_response`| `(status int, ctype string, body string) string` | Formats a valid HTTP/1.1 response string with `Content-Length`. |

```v
server := net_listen(8080)
client := net_accept(server)
req := net_read(client, 4096)
resp := http_response(200, 'application/json', '{"status":"ok"}')
_ := net_write(client, resp)
net_close(client)
```

---

## 8. RFC 8259 JSON (`import json`) & Fast JSON Builtins

### Standard Library `import json` (Pure `.vir`, Recommended)
Full RFC 8259 JSON validation, `{}`/`[]` depth-aware key & array queries, string escaping/unescaping (including `\uXXXX`), and $O(N)$ `JsonBuilder` serialization written in 100% pure Vira:

| Function / Method | Signature | Description |
| :--- | :--- | :--- |
| `json.is_valid` | `(raw string) bool` | Validates RFC 8259 JSON syntax |
| `json.has_key` | `(raw string, key string) bool` | Depth-aware top-level object key check |
| `json.get_raw` | `(raw string, key string) !string` | Extracts raw nested JSON object/array/token at `key` |
| `json.get_str` | `(raw string, key string) !string` | Extracts and unescapes a string field at `key` |
| `json.get_int` / `json.get_i64` | `(raw string, key string) !int` / `!i64` | Extracts a 32-bit or 64-bit integer field at `key` |
| `json.get_bool` | `(raw string, key string) !bool` | Extracts `true` or `false` at `key` |
| `json.array_len` | `(raw string) !int` | Element count of a JSON array `[...]` |
| `json.array_get_raw` / `array_get_str` | `(raw string, idx int) !string` | Extracts raw or unescaped string element at `idx` |
| `json.escape` / `json.unescape` | `(s string) string` | Escapes or unescapes RFC 8259 JSON string content |
| `json.JsonBuilder.new()` | `() json.JsonBuilder` | Creates an $O(N)$ byte-buffered JSON object serializer (`.set_str`, `.set_int`, `.set_i64`, `.set_bool`, `.set_null`, `.set_raw`, `.build`) |

```v
import json

mut jb := json.JsonBuilder.new()
jb.set_str('model', 'llama-3.2')
jb.set_int('max_tokens', 512)
payload := jb.build()

model := json.get_str(payload, 'model') or { 'default' }
```

### Lightweight Builtin Shims (`json_get_str`, `json_get_int`, `json_get_f32`)
Depth-aware and escape-aware convenience builtins available without imports:
- `json_get_str(json string, key string) string`
- `json_get_int(json string, key string) i64`
- `json_get_f32(json string, key string) f32`

---

## 9. Native GUI & Declarative Layout Engine

Vira includes a double-buffered 60 FPS Immediate-Mode GUI (IMGUI) engine and an automatic layout cursor stack.

### Window Lifecycle & Polling
- `gui_init(width int, height int, title string) void`
- `gui_should_close() bool`
- `gui_set_target_fps(fps int) void`
- `gui_begin() void` / `gui_end() void`
- `gui_clear(hex_color int) void`
- `gui_close() void`
- `gui_cancel_close() void`
- `gui_frame_time() f32`

### Declarative Auto-Layout Cursor Stack
- `gui_vstack_begin(x int, y int, width int, spacing int) void`
- `gui_hstack_begin(height int, spacing int) void`
- `gui_stack_end() void`
- `gui_spacing(px int) void`
- `gui_card_begin(title string, width int) void` / `gui_card_end() void`
- `gui_separator() void`

### Auto-Layout Zero-Math Widgets
- `gui_btn(label string) bool`
- `gui_check(label string, checked bool) bool`
- `gui_slide(left string, right string, val f32, min f32, max f32) f32`
- `gui_lbl(text string) void`
- `gui_input(placeholder string, text string) string`
- `gui_progress(percent f32) void`
- `gui_radio(label string, is_selected bool) bool`
- `gui_badge(text string, hex_bg int, hex_fg int) void`
- `gui_tab_item(label string, is_active bool) bool`
- `gui_modal_begin(title string, width int, height int) void` / `gui_modal_end() void`

```v
gui_init(800, 600, 'Vira Dashboard')
for !gui_should_close() {
    gui_begin()
    gui_clear(0x1E1E2E)

    gui_vstack_begin(50, 50, 300, 10)
        gui_card_begin('Settings', 300)
            gui_lbl('Worker Configuration')
            if gui_btn('Start Processing') {
                println('Started!')
            }
        gui_card_end()
    gui_stack_end()

    gui_end()
}
gui_close()
```

---

## 10. Memory, Bump Arenas, Autofree & C FFI Helpers

Vira eliminates both manual `free()` and garbage collection pauses through **Static Escape Analysis (Autofree)** and **64-byte aligned Bump Arenas**.

### Deterministic Autofree (No GC)
Every variable has an escape state determined at compile time:
- `ScopedLocal`: Destroyed immediately at scope exit via LIFO destructors (`vira_str_free`, `vira_tensor_f32_free`, `vira_future_free`). Inside actors, active stack allocations are also registered on `_vira_actor_ctx.defer_stack` so `panic()` unwinds them cleanly before `longjmp`.
- `EscapesViaReturn`: Transferred to the caller without destructor injection.
- `MovedToContainer`: Transferred to a struct or collection; freed when the parent container exits scope.

### C FFI NUL-Terminated String Marshaling (`vira_rt.h`)
When writing C99 intrinsics (`@[intrinsic: '...']`), `vira_rt.h` provides zero-truncation NUL-terminated string marshaling helpers:
- `char *cstr = vira_str_cstr(s);` ... `vira_cstr_free(cstr);` — exact-length heap C string copy.
- `vira_str_to_cstr_buf(s, stack_buf, sizeof(stack_buf))` / `vira_cstr_buf_free(&cb)` — zero-malloc stack fast-path with automatic exact-size heap fallback for arbitrarily long paths and payloads.

### Bump Arenas (`arena { ... }`)
For high-frequency temporary allocations (e.g. forward passes, string formatting in tight loops):
```v
arena {
    // All tensor_fill and matrix multiplication calls inside this block
    // allocate sequentially on a 64-byte aligned contiguous slab (chaining slabs automatically if exceeded).
    t1 := tensor_fill(128, 128, 1.0)
    t2 := tensor_fill(128, 128, 2.0)
    t3 := t1 @ t2
    println('Sum: ${tensor_sum(t3)}')
} // Instant O(1) slab teardown here! 0 memory fragmentation!
```

---

## 11. Native Collections (`[]T` & `[N]T`) & String Methods

### Dynamic Slices (`[]T`)
Contiguous, heap-allocated dynamic arrays (`[]u8`, `[]int`, `[]i64`, `[]f32`, `[]f64`, `[]bool`, `[]string`, or `[]Struct`) automatically freed by Autofree at scope exit.

| Property / Method | Signature | Description |
| :--- | :--- | :--- |
| `.len` | `int` | Current number of elements in the slice |
| `.cap` | `int` | Allocated capacity of the underlying buffer |
| `.is_empty()` | `fn () bool` | Returns `true` when `.len == 0` |
| `.push(val T)` | `fn (mut s []T, val T) void` | Appends `val` to the end of `s`, growing capacity by $2\times$ if full |
| `.pop()` | `fn (mut s []T) T` | Removes and returns the last element (panics if empty) |
| `.get(idx int)` | `fn (s []T, idx int) !T` | Bounds-checked lookup returning `!T` Result |
| `.contains(val T)` | `fn (s []T, val T) bool` | Returns `true` if `val` is present in `s` |
| `.slice(start int, end int)` | `fn (s []T, start int, end int) []T` | Returns a newly allocated sub-slice `s[start..end]` |
| `.clone()` | `fn (s []T) []T` | Deep-clones the slice and all owned element heap fields |
| `.clear()` | `fn (mut s []T) void` | Resets `.len` to `0` while retaining allocated capacity |

### Fixed-Size Stack Arrays (`[N]T`)
Stack-allocated value arrays (`[N]T`) with zero heap overhead.

| Property / Method | Signature | Description |
| :--- | :--- | :--- |
| `.len` | `int` | Compile-time constant length `N` |
| `.get(idx int)` | `fn (a [N]T, idx int) !T` | Safe bounds-checked element access returning `!T` |
| `.contains(val T)` | `fn (a [N]T, val T) bool` | Returns `true` if `val` is in the array |

### Built-in `string` Methods
Vira `string` values are immutable UTF-8 views (`data`, `len`, `is_lit`) with built-in methods:

| Property / Method | Signature | Description |
| :--- | :--- | :--- |
| `.len` | `int` | Byte length of the UTF-8 string |
| `.is_empty()` | `fn () bool` | Returns `true` if `.len == 0` |
| `.contains(sub string)` | `fn (sub string) bool` | Returns `true` if `sub` occurs within the string |
| `.starts_with(prefix string)` | `fn (prefix string) bool` | Returns `true` if the string starts with `prefix` |
| `.ends_with(suffix string)` | `fn (suffix string) bool` | Returns `true` if the string ends with `suffix` |
| `.index_of(needle string)` | `fn (needle string) int` | Byte index of first occurrence of `needle`, or `-1` |
| `.last_index_of(needle string)` | `fn (needle string) int` | Byte index of last occurrence of `needle`, or `-1` |
| `.substr(start int, end int)` | `fn (start int, end int) string` | Newly allocated substring covering half-open byte range `[start, end)` |
| `.trim()` | `fn () string` | Newly allocated string with leading and trailing whitespace removed |
| `.trim_left()` | `fn () string` | Strips leading whitespace |
| `.trim_right()` | `fn () string` | Strips trailing whitespace |
| `.repeat(count int)` | `fn (count int) string` | Replicates the string `count` times |
| `.lines()` | `fn () []string` | Splits string into lines by `\n` or `\r\n` |
| `.byte_at(idx int)` | `fn (idx int) !u8` | Returns byte at `idx` or error if out of bounds |
| `.split(delim string)` | `fn (delim string) []string` | Splits the string by `delim` into a slice of strings (`[]string`) |
| `.replace(old string, new string)` | `fn (old string, new string) string` | Replaces all occurrences of `old` with `new` |
| `.to_upper()` / `.to_lower()` | `fn () string` | Returns an ASCII uppercase or lowercase copy |
| `.to_int()` / `.to_i64()` | `fn () !int` / `fn () !i64` | Parses a base-10 integer, returning a Result |
| `.to_f64()` / `.to_f32()` | `fn () !f64` / `fn () !f32` | Parses a floating-point number, returning a Result |
| `.bytes()` | `fn () []u8` | Copies the string bytes into a mutable `[]u8` slice |

---

## 12. Standard Library Packages

Import first-party standard library modules using `import <pkg>`:

### `import math`
High-performance C99 math intrinsics:
- **Trigonometry**: `sin(x)`, `cos(x)`, `tan(x)`, `asin(x)`, `acos(x)`, `atan(x)`, `atan2(y, x)`
- **Hyperbolics**: `sinh(x)`, `cosh(x)`, `tanh(x)`
- **Exponential & Logarithmic**: `exp(x)`, `exp2(x)`, `log(x)`, `log2(x)`, `log10(x)`
- **Power & Roots**: `pow(x, y)`, `sqrt(x)`, `cbrt(x)`, `hypot(x, y)`
- **Rounding & Modulo**: `ceil(x)`, `floor(x)`, `trunc(x)`, `round(x)`, `abs(x)`
- **Constants**: `math.PI`, `math.E`, `math.LN2`, `math.LN10`

### `import ctype`
Fast ASCII character classification and conversion:
- **Classifiers**: `is_alpha(c)`, `is_digit(c)`, `is_alnum(c)`, `is_space(c)`, `is_upper(c)`, `is_lower(c)`, `is_xdigit(c)`, `is_print(c)`, `is_punct(c)`, `is_cntrl(c)`
- **Converters**: `to_lower(c)`, `to_upper(c)`

### `import strings`
Pure-Vira string processing utilities:
- `contains(s, sub)`, `starts_with(s, prefix)`, `ends_with(s, suffix)`, `has_prefix(s, p)`, `has_suffix(s, p)`
- `index_of(s, needle)`, `last_index_of(s, needle)`, `split(s, delim)`, `join(parts, sep)`, `replace(s, old, new)`
- `trim(s)`, `trim_left(s)`, `trim_right(s)`, `repeat(s, count)`, `lines(s)`, `fields(s)`

### `import time`
Wall-clock timestamps, formatting and sleeping:
- `time.now() i64`: Current Unix timestamp in seconds.
- `time.sleep_ms(ms int) void`: Sleep current thread for `ms` milliseconds.
- `time.format(epoch i64, fmt string) string`: Format epoch via `strftime` in local time.
- `time.utc_format(epoch i64, fmt string) string`: Format epoch via `strftime` in UTC.
- `time.parse(str string, fmt string) !i64`: Parse formatted string into Unix epoch.

### `import io`
File streaming, buffered reader, and standard I/O:
- `io.File.open(path, mode) !File`, `f.read(mut buf)`, `f.write(buf)`, `f.write_string(s)`, `f.close()`
- `io.BufReader.new(file)`, `r.read_line()`, `r.read_byte()`, `r.read_exact(n)`, `r.read_all()`, `r.close()`
- `io.read_line() !string`, `io.read_all_stdin() !string`, `io.prompt(msg) !string`
- `io.rename(old, new) !bool`, `io.remove(path) !bool`, `io.temp_file() !string`

### `import os`
Process environment, filesystem manipulation, and platform queries:
- `os.args() []string`, `os.exit(code int)`, `os.set_quiet_rt(quiet bool)`
- `os.getenv(k)`, `os.setenv(k, v)!`, `os.unsetenv(k)!`, `os.environ() []string`
- `os.platform() string`, `os.is_windows() bool`, `os.path_separator() string`
- `os.getpid()`, `os.getppid()`, `os.getuid()`, `os.getgid()`, `os.hostname()!`
- `os.exists(path)`, `os.is_file(path)`, `os.is_dir(path)`, `os.stat(path)!`
- `os.read_file(path)!`, `os.write_file(path, content)!`, `os.read_bytes(path)!`, `os.write_bytes(path, b)!`
- `os.mkdir(path, perm)!`, `os.mkdir_all(path, perm)!`, `os.remove(path)!`, `os.remove_all(path)!`, `os.read_dir(path)!`
- `os.join_path(p1, p2)`, `os.base_name(path)`, `os.dir_name(path)`, `os.ext(path)`, `os.clean_path(path)`

### `import syscall`
Raw OS system calls, file descriptors, and Linux container primitives:
- `syscall.open(path, flags, mode)!`, `syscall.read(fd, mut buf, n)!`, `syscall.write(fd, buf, n)!`, `syscall.close(fd)!`
- `syscall.pipe(mut fds)!`, `syscall.dup(fd)!`, `syscall.dup2(old_fd, new_fd)!`, `syscall.isatty(fd)`
- `syscall.stat(path)!`, `syscall.mkdir(path, perm)!`, `syscall.readdir(path)`, `syscall.getuid()`, `syscall.getgid()`
- Linux container support: `syscall.mount(src, target, fs, flags, data)!`, `syscall.umount2(target, flags)!`, `syscall.chroot(path)!`, `syscall.mknod(path, mode, dev)!`, `syscall.makedev(major, minor)`, `syscall.unshare(flags)!`, `syscall.sethostname(name)!`
- Namespace flags: `CLONE_NEWPID`, `CLONE_NEWUTS`, `CLONE_NEWNS`, `CLONE_NEWNET`, `CLONE_NEWIPC`

### `import exec`
Process execution and supervision:
- `exec.Command.new(prog string, args []string) Command`
- `cmd.set_dir(path)`, `cmd.set_clone_flags(flags)`, `cmd.set_stdin(text)`, `cmd.set_inherit_stdio(inherit)`
- `cmd.start() !int`, `cmd.wait() !int`, `cmd.run() !int`, `cmd.output() !string`, `cmd.combined_output() !string`
- `exec.combined_output(prog string, args []string) !string`

### `import flag`
POSIX/GNU command-line argument parser:
- `flag.FlagSet.new(name string)`
- `fs.bool_flag(name, short, def, desc)`, `fs.string_flag(name, short, def, desc)`, `fs.int_flag(name, short, def, desc)`, `fs.i64_flag(name, short, def, desc)`, `fs.f64_flag(name, short, def, desc)`
- `fs.parse(args []string) !bool`, `fs.get_bool(name)`, `fs.get_string(name)`, `fs.get_int(name)`, `fs.get_i64(name)`, `fs.get_f64(name)`
- `fs.args []string` (remaining positional arguments after `--` or flags)
- `fs.help_string() string` (auto-formatted options table)

### `import log`
Leveled stderr logger:
- `log.Logger.new(level int)`
- `l.set_level(level)`, `l.set_prefix(prefix)`, `l.set_timestamp(enabled)`
- `l.debug(msg)`, `l.info(msg)`, `l.warn(msg)`, `l.error(msg)`, `l.fatal(msg)`
- Package-level: `log.debug(msg)`, `log.info(msg)`, `log.warn(msg)`, `log.error(msg)`, `log.fatal(msg)`
- Levels: `log.LEVEL_DEBUG`, `log.LEVEL_INFO`, `log.LEVEL_WARN`, `log.LEVEL_ERROR`, `log.LEVEL_FATAL`

### `import net`, `import http`, `import json`, `import viraserve`
- **`net`**: `TcpListener`, `TcpStream`, `UdpSocket`, `SocketAddr`, `Poller`
- **`http`**: `Request`, `Response`, `HeaderMap`, `HttpServer`, `SseStream`
- **`json`**: Pure-`.vir` RFC 8259 parser, validator, extractor, and `JsonBuilder`
- **`viraserve`**: Actor-backed microservice and SSE streaming framework

