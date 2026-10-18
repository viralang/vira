# Vira Standard Library & Builtin API Reference

Welcome to the definitive API reference for the **Vira Programming Language**. This document describes every built-in function, data type, operator, and runtime primitive available out-of-the-box in Vira with exact type signatures, behavioral details, memory ownership rules, and executable code snippets.

---

## Table of Contents

1. [Core Runtime & Diagnostics](#1-core-runtime--diagnostics)
2. [AI Tensors & SIMD Math](#2-ai-tensors--simd-math)
3. [Autograd Backpropagation & Optimizers](#3-autograd-backpropagation--optimizers)
4. [SafeTensors & GGUF Local LLM Inference](#4-safetensors--gguf-local-llm-inference)
5. [Actor Concurrency & Supervisors](#5-actor-concurrency--supervisors)
6. [Native File I/O & Filesystem](#6-native-file-io--filesystem)
7. [Networking & HTTP 1.1 Web Engine](#7-networking--http-11-web-engine)
8. [Fast JSON Field Parser](#8-fast-json-field-parser)
9. [Native GUI & Declarative Layout Engine](#9-native-gui--declarative-layout-engine)
10. [Memory, Bump Arenas & Autofree](#10-memory-bump-arenas--autofree)

---

## 1. Core Runtime & Diagnostics

These functions are available globally in all modules without imports.

### `println(msg string) void`
Prints a UTF-8 string followed by a newline character to standard output (`stdout`).
- **Memory Rule:** The `msg` argument is borrowed. If constructed via string interpolation `${...}`, it is automatically hoisted and freed deterministically upon statement exit.
```v
println('Server started on port ${port}')
```

### `eprintln(msg string) void`
Prints a UTF-8 string followed by a newline to standard error (`stderr`).
```v
eprintln('ERROR: Connection reset by peer')
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

## 2. AI Tensors & SIMD Math

2D numerical tensors are first-class primitives in Vira (`tensor[f32]`). All tensor memory is 64-byte aligned to maximize AVX2 (256-bit) and ARM NEON (128-bit) vector load/store throughput.

### `tensor_fill(rows int, cols int, val f32) tensor[f32]`
Allocates a 2D tensor of dimensions `rows × cols` and fills every element with scalar value `val`.
- **Hardware Acceleration:** Uses `_mm256_set1_ps` and `_mm256_storeu_ps` on x86_64; uses `vdupq_n_f32` on ARM.
- **Memory Ownership:** If called inside an `arena { ... }` block, allocation occurs on the bump arena with $O(1)$ slab teardown; otherwise, tracked and freed automatically by Autofree upon scope exit.
```v
// Create a 64x64 matrix filled with 0.25
weights := tensor_fill(64, 64, 0.25)
```

### Matrix Multiplication Operator: `@`
Computes the hardware-accelerated matrix multiplication $C = A @ B$.
- **Shape Requirement:** $A$'s column count must equal $B$'s row count: $(M \times K) @ (K \times N) \to (M \times N)$. Shape mismatch triggers a panic.
- **Hardware Kernel:** Tiled register-blocked Fused Multiply-Add (FMA3) delivering up to 8 single-precision FLOPs per cycle per core.
```v
w := tensor_fill(64, 32, 0.5)
x := tensor_fill(32, 16, 1.0)
y := w @ x // Result shape: 64x16
```

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

### Fire-and-Forget
If the return value is not needed, discard the future with `_`:
```v
_ := gw <- handle_request(99)
```

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

## 8. Fast JSON Field Parser

Zero-allocation string search extracting values from JSON payloads without external libraries.

### `json_get_str(json string, key string) string`
Extracts a string property value matching `"key": "value"`.
```v
payload := '{"model": "llama-3", "prompt": "Hello"}'
model_name := json_get_str(payload, 'model') // 'llama-3'
```

### `json_get_int(json string, key string) i64`
Extracts an integer property value matching `"key": 1234`.
```v
tokens := json_get_int(payload, 'max_tokens')
```

### `json_get_f32(json string, key string) f32`
Extracts a floating-point property value matching `"key": 0.85`.
```v
temp := json_get_f32(payload, 'temperature')
```

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

## 10. Memory, Bump Arenas & Autofree

Vira eliminates both manual `free()` and garbage collection pauses through **Static Escape Analysis (Autofree)** and **64-byte aligned Bump Arenas**.

### Deterministic Autofree (No GC)
Every variable has an escape state determined at compile time:
- `ScopedLocal`: Destroyed immediately at scope exit via LIFO destructors (`vira_str_free`, `vira_tensor_f32_free`).
- `EscapesViaReturn`: Transferred to the caller without destructor injection.
- `MovedToContainer`: Transferred to a struct or collection; freed when the parent container exits scope.

### Bump Arenas (`arena { ... }`)
For high-frequency temporary allocations (e.g. forward passes, string formatting in tight loops):
```v
arena {
    // All tensor_fill and matrix multiplication calls inside this block
    // allocate sequentially on a 64-byte aligned contiguous slab.
    t1 := tensor_fill(128, 128, 1.0)
    t2 := tensor_fill(128, 128, 2.0)
    t3 := t1 @ t2
    println('Sum: ${tensor_sum(t3)}')
} // Instant O(1) slab teardown here! 0 memory fragmentation!
```
