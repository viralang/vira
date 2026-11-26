# Appendix B: Built-in Functions & Standard Library Quick Reference

Every built-in function, type method, runtime telemetry control, and standard library export listed in this appendix is verified directly against `internal/checker/checker.go` and `packages/*/src/*.vir`.

---

## B.1. Global Built-in Functions

Built-in functions are available in every Vira file without an `import`.

### Console Output, Panics & Errors

| Signature | Description |
| :--- | :--- |
| `print(msg string)` | Writes `msg` to `stdout` without a trailing newline. |
| `println(msg string)` | Writes `msg` to `stdout` followed by `\n`. |
| `eprint(msg string)` | Writes `msg` to `stderr` without a trailing newline. |
| `eprintln(msg string)` | Writes `msg` to `stderr` followed by `\n`. |
| `panic(msg string)` | Unwinds and faults the current actor (supervised if `restarts < max_restarts`), or aborts the process outside an actor. |
| `error(msg string) !T` | Constructs an error value for returning from a `!T` (or `!`) function. |

### Type Casts & Structural Interface Boxing

| Syntax | Description |
| :--- | :--- |
| `u8(x)`, `byte(x)`, `int(x)`, `i64(x)`, `f32(x)`, `f64(x)`, `bool(x)` | Explicit primitive casts (also `int(e)` / `i64(e)` on `enum` values). Rejects `string` (use `.to_int()`, `.to_i64()`, `.to_f64()`, `.to_f32()`). |
| `InterfaceName(struct_val)` | Boxes a struct value into a 16-byte `{ _object, _vtable }` fat pointer if it structurally satisfies `InterfaceName`. |

### Monotonic Time & Basic File I/O

| Signature | Description |
| :--- | :--- |
| `time_now_us() i64` | High-precision monotonic timestamp in microseconds (`CLOCK_MONOTONIC` / `QueryPerformanceCounter`). |
| `time_now_ms() i64` | Monotonic timestamp in milliseconds. |
| `time_sleep_ms(ms int)` | Sleeps the calling thread for `ms` milliseconds. |
| `file_exists(path string) bool` | Returns `true` if `path` exists. |
| `read_file(path string) !string` | Reads an entire file into a `string` (includes chunked fallback for `/proc` virtual files). |
| `write_file(path string, content string) !bool` | Creates or overwrites `path` with `content`. |
| `append_file(path string, content string) !bool` | Appends `content` to `path`. |
| `remove_file(path string) !bool` | Deletes `path`. |

### Low-Level Net & JSON Primitives

*(Prefer `import net`, `import http`, and `import json` for application code.)*

| Signature | Description |
| :--- | :--- |
| `net_listen(port int) int` | Opens a TCP listener socket on `port` and returns its fd. |
| `net_accept(server_fd int) int` | Accepts a client connection and returns its fd. |
| `net_read(client_fd int, max_bytes int) string` | Reads up to `max_bytes` from `client_fd`. |
| `net_write(client_fd int, msg string) int` | Writes `msg` to `client_fd` and returns bytes written. |
| `net_close(fd int)` | Closes socket `fd`. |
| `http_response(status int, content_type string, body string) string` | Formats a raw HTTP/1.1 response string. |
| `json_get_str(json string, key string) string` | Extracts a top-level string value for `key`. |
| `json_get_int(json string, key string) i64` | Extracts a top-level integer value for `key`. |
| `json_get_f32(json string, key string) f32` | Extracts a top-level `f32` value for `key`. |

### Tensor, Autograd, Weights & LLM Builtins

| Signature | Description |
| :--- | :--- |
| `tensor_fill(rows int, cols int, val f32) tensor[f32]` | Allocates a 2D `tensor[f32]` filled with `val`. |
| `tensor_fill_3d(d0 int, d1 int, d2 int, val f32) tensor[f32]` | Allocates a 3D `tensor[f32]` filled with `val`. |
| `tensor_fill_4d(d0 int, d1 int, d2 int, d3 int, val f32) tensor[f32]` | Allocates a 4D `tensor[f32]` filled with `val`. |
| `tensor_sum(t tensor[f32]) f32` | Sums all elements of `t`. |
| `tensor_relu(t tensor[f32]) tensor[f32]` | Element-wise $\max(0, x)$. |
| `tensor_add(a tensor[f32], b tensor[f32]) tensor[f32]` | Element-wise $a + b$ (supports row broadcasting). |
| `tensor_scale(t tensor[f32], scale f32) tensor[f32]` | Element-wise $t \cdot \text{scale}$. |
| `tensor_transpose(t tensor[f32]) tensor[f32]` | Returns transposed tensor. |
| `tensor_matmul_grad_a(grad_out tensor[f32], b tensor[f32]) tensor[f32]` | Backward pass gradient $\partial L / \partial A = \text{grad\_out} @ B^T$. |
| `tensor_matmul_grad_b(a tensor[f32], grad_out tensor[f32]) tensor[f32]` | Backward pass gradient $\partial L / \partial B = A^T @ \text{grad\_out}$. |
| `tensor_relu_backward(input tensor[f32], grad_out tensor[f32]) tensor[f32]` | ReLU gradient mask applied to `grad_out`. |
| `tensor_sgd_step(mut param tensor[f32], grad tensor[f32], lr f32)` | In-place SGD update `param -= lr * grad`. |
| `tensor_load_bin(rows int, cols int, path string) !tensor[f32]` | Loads raw float32 binary weights. |
| `tensor_save_bin(t tensor[f32], path string) !bool` | Saves `t` to a raw float32 binary file. |
| `safetensors_load(path string, tensor_name string) !tensor[f32]` | Loads a named tensor from a Hugging Face `.safetensors` file. |
| `gguf_load_tensor(path string, tensor_name string) !tensor[f32]` | Loads a named tensor from a GGUF v2/v3 file (with lazy FP32 and raw quant registry). |
| `tensor_rmsnorm(x tensor[f32], weight tensor[f32], eps f32) tensor[f32]` | RMS normalization. |
| `tensor_rope(q tensor[f32], pos int, freq_base f32) tensor[f32]` | Interleaved LLaMA-style RoPE. |
| `tensor_rope_norm(q tensor[f32], pos int, freq_base f32) tensor[f32]` | Split-half NeoX-style RoPE (Qwen 2.5). |
| `tensor_silu(x tensor[f32]) tensor[f32]` | SiLU activation $x \cdot \sigma(x)$. |
| `tensor_elemwise_mul(a tensor[f32], b tensor[f32]) tensor[f32]` | Element-wise Hadamard product $a \odot b$. |
| `tensor_softmax(logits tensor[f32]) tensor[f32]` | Row-wise numerically stable softmax. |
| `tensor_embedding(table tensor[f32], token_id int) tensor[f32]` | Gathers a single embedding row for `token_id`. |
| `tensor_embedding_batch(table tensor[f32], token_ids []int) tensor[f32]` | Gathers a batch of embedding rows. |
| `sample_token(logits tensor[f32], temperature f32) int` | Temperature-scaled / argmax token sampling. |

### GPU Compute Builtins (`gpu_*`)

All `gpu_*` builtins are also wrapped cleanly by `import gpu`:
`gpu_is_available() bool`, `gpu_device_name() string`, `gpu_backend_name() string`, `gpu_vram_bytes() i64`, `gpu_allocated_bytes() i64`, `gpu_stats_reset()`, `gpu_stats_print(label string, per i64)`, `gpu_oom_fallbacks() i64`, `gpu_set_enabled(enabled bool)`, `gpu_is_enabled() bool`, `gpu_has_tensor_cores() bool`, `gpu_set_backend(backend string)`, `gpu_set_device(device string)`, `gpu_sync()`, `gpu_kv_f16(t tensor[f32]) bool`, `gpu_wait()`, `gpu_simulate_oom(enable bool)`, `gpu_matmul(a, b) tensor[f32]`, `gpu_linear(x, weight) tensor[f32]`, `gpu_rmsnorm(x, weight, eps) tensor[f32]`, `gpu_rope(q, pos, freq_base) tensor[f32]`, `gpu_rope_norm(q, pos, freq_base) tensor[f32]`, `gpu_rope_seq(q, start_pos, n_heads, freq_base) tensor[f32]`, `gpu_rope_seq_norm(q, start_pos, n_heads, freq_base) tensor[f32]`, `gpu_softmax(logits) tensor[f32]`, `gpu_silu(x) tensor[f32]`, `gpu_linear_swiglu(x, w_gate, w_up) tensor[f32]`, `gpu_silu_mul(gate, up) tensor[f32]`, `gpu_gqa_attention(q_heads, k_cache, v_cache, kv_pos, scale) tensor[f32]`, `gpu_gqa_prefill(q_seq, k_cache, v_cache, start_pos, seq_len, n_heads, scale) tensor[f32]`.

### Native GUI Engine Builtins (`gui_*`)

- **Window & Frame Loop:** `gui_init(width, height, title)`, `gui_should_close() bool`, `gui_set_target_fps(fps)`, `gui_begin()`, `gui_end()`, `gui_close()`, `gui_clear(hex_color)`, `gui_cancel_close()`, `gui_window_width() int`, `gui_window_height() int`, `gui_frame_time() f32`
- **Immediate Primitives:** `gui_draw_text(text, x, y, size, hex_color)`, `gui_draw_text_wrapped(text, x, y, max_width, size, hex_color)`, `gui_draw_rect(x, y, width, height, hex_color)`, `gui_draw_circle(cx, cy, radius f32, hex_color)`, `gui_draw_line(x1, y1, x2, y2, hex_color)`, `gui_draw_badge(x, y, text, hex_bg, hex_fg)`
- **Positioned Controls:** `gui_button(x, y, w, h, label) bool`, `gui_slider(x, y, w, h, text_left, text_right, val, min_val, max_val) f32`, `gui_checkbox(x, y, w, h, label, checked) bool`, `gui_label(x, y, w, h, text)`, `gui_text_input(x, y, w, h, placeholder, text) string`, `gui_progress_bar(x, y, w, h, percent f32)`, `gui_radio_button(x, y, label, is_selected) bool`, `gui_dropdown(x, y, w, h, items []string, selected_idx int) int`, `gui_table(x, y, w, h, headers []string, rows [][]string, search_query string, selected_row int) int`
- **Auto-Layout Stacks & Controls:** `gui_vstack_begin(x, y, width, spacing)`, `gui_hstack_begin(height, spacing)`, `gui_stack_end()`, `gui_spacing(px)`, `gui_card_begin(title, width)`, `gui_card_end()`, `gui_scroll_begin(width, height)`, `gui_scroll_end()`, `gui_modal_begin(title, width, height)`, `gui_modal_end()`, `gui_btn(label) bool`, `gui_btn_danger(label) bool`, `gui_check(label, checked) bool`, `gui_slide(text_left, text_right, val, min_val, max_val) f32`, `gui_lbl(text)`, `gui_lbl_wrap(text, max_width)`, `gui_input(placeholder, text) string`, `gui_progress(percent f32)`, `gui_radio(label, is_selected) bool`, `gui_separator()`, `gui_badge(text, hex_bg, hex_fg)`, `gui_tab_item(label, is_active) bool`, `gui_select(items []string, selected_idx int) int`, `gui_tbl(headers, rows, search_query, selected_row) int`, `gui_table_sort_col() int`, `gui_table_sort_asc() bool`, `gui_table_set_sort(col int, asc bool)`
- **Input & Images:** `gui_mouse_x() int`, `gui_mouse_y() int`, `gui_mouse_wheel() f32`, `gui_is_mouse_down(button) bool`, `gui_is_key_pressed(key) bool`, `gui_load_image(path) int`, `gui_image_width(id) int`, `gui_image_height(id) int`, `gui_draw_image(id, x, y, w, h)`, `gui_image(id, w, h)`, `gui_free_image(id)`

---

## B.2. Runtime Telemetry & Environment Variables

At process exit (`vira_report_mem_stats` in `runtime/rt/platform/telemetry.h`), Vira prints diagnostic lines unless compiled with `-quiet-rt` (`VIRA_QUIET_RT`), disabled via `os.set_quiet_rt(true)`, or suppressed at runtime with `VIRA_RT_STATS=0`:

```text
[vira_rt] gpu=<device> (<backend>) | vram_peak=<N> MiB / <M> MiB | oom_fallbacks=<K>
[vira_rt] simd=<active SIMD backend>
[vira_rt] pool_workers=<W> | actor_tasks_run=<T> | work_steals=<S>
[vira_rt] active_leaks=0 | total_freed=<F>
```

| Environment Variable / CLI Flag | Effect |
| :--- | :--- |
| `VIRA_RT_STATS=0` | Suppresses the exit telemetry banner. |
| `VIRA_GPU_STATS=1` | Prints per-transfer/launch GPU statistics on exit. |
| `VIRA_WORKERS=N` | Overrides actor worker-pool thread count (clamped to `[1, 32]`; default is logical cores clamped to `[2, 16]`). |
| `VIRA_THREADS=N` | Overrides compute worker-team thread count for tensor/LLM ops (clamped to `[1, 16]`; default is physical core count). |
| `VIRA_LAZY_HOST_FP32=0` | Disables lazy host FP32 materialization for GGUF quantized tensors, dequantizing eagerly on load. |
| `vira test -allow-leaks` | Reports `active_leaks` without failing tests (`vira test` fails on `active_leaks != 0` by default). |

---

## B.3. Built-in Type Fields & Methods

### `string`

- **Field:** `.len` (`int` byte length)
- **Inspection & Slicing:** `.len() int`, `.contains(substr string) bool`, `.index_of(substr string) int`, `.last_index_of(substr string) int`, `.starts_with(prefix string) bool`, `.ends_with(suffix string) bool`, `.substr(start int, end int) string`, `.slice(start int, end int) string`, `.byte_at(index int) !u8`
- **Transformations & Splitting:** `.trim() string`, `.trim_left() string`, `.trim_right() string`, `.to_lower() string`, `.to_upper() string`, `.replace(old string, new_str string) string`, `.repeat(count int) string`, `.split(sep string) []string`, `.lines() []string`, `.clone() string`, `.bytes() []u8`, `.to_bytes() []u8`
- **Parsing:** `.to_int() !int`, `.to_i64() !i64`, `.to_f64() !f64`, `.to_f32() !f32`

### Fixed Arrays (`[N]T`)

- **Fields:** `.len` (`int`), `.cap` (`int`)
- **Methods:** `.len() int`, `.get(index int) !T`, `.contains(val T) bool`, `.is_empty() bool`

### Dynamic Slices (`[]T`)

- **Fields:** `.len` (`int`), `.cap` (`int`)
- **Read / Query Methods:** `.len() int`, `.cap() int`, `.get(index int) !T`, `.contains(val T) bool`, `.is_empty() bool`, `.clone() []T`, `.slice(start int, end int) []T`
- **Mutating Methods** *(require `mut` binding)*: `.push(item T)`, `.append(item T)`, `.pop() T`, `.set(index int, val T)`, `.clear()`
- **Byte-Slice (`[]u8`) Only:** `.to_string() string`, `.to_str() string`

### Maps (`map[K]V`)

*(Valid key types `K`: `string`, integer types, `bool`, `enum`. Missing keys read as zero value on `m[k]`.)*

- **Field:** `.len` (`int`)
- **Methods:** `.has(key K) bool`, `.delete(key K) bool` *(mut)*, `.keys() []K`, `.values() []V`, `.clear()` *(mut)*

### Tensors (`tensor[f32]`)

- **Fields:** `.rows` (`int`), `.cols` (`int`), `.ndim` (`int`), `.n_dims` (`int`)
- **Shape & Access:** `.rows() int`, `.cols() int`, `.ndim() int`, `.numel() i64`, `.dim(axis int) int`, `.stride(axis int) int`, `.reshape_2d(d0, d1)`, `.reshape_3d(d0, d1, d2)`, `.reshape_4d(d0, d1, d2, d3)`, `.transpose()`, `.clone()`, `.get(row int, col int) f32`, `.set(row int, col int, val f32)`, `.copy_row(dst_row int, src tensor[f32])`, `.copy_rows(dst_row_start int, src tensor[f32])`, `.slice_rows(start int, end int) tensor[f32]`
- **Compute:** `.linear(weight tensor[f32]) tensor[f32]`, `.gpu_linear(weight tensor[f32]) tensor[f32]`, `.gpu_matmul(b tensor[f32]) tensor[f32]`

### Futures (`Future[T]`) & Actor References (`Ref[T]`)

- **`Future[T]`:** `.await() !T` (consumes the future, with work-helping and cycle detection), `.ready() bool` (non-consuming readiness poll)
- **Actor Handle `A`:** `.ref() Ref[A]` (creates a copyable, non-owning reference)
- **`Ref[T]`:** `.alive() bool` (returns `true` while the referenced actor exists and is not stopped)

---

## B.4. Standard Library Packages (`packages/*/src/*.vir`)

Vira ships **17 standard library packages** (use `import <pkg>` and qualify members as `<pkg>.<Symbol>`):

| Package | Public Types, Constants & Functions |
| :--- | :--- |
| **`ctype`** | `is_alpha(c u8) bool`, `is_digit(c u8) bool`, `is_alnum(c u8) bool`, `is_space(c u8) bool`, `is_upper(c u8) bool`, `is_lower(c u8) bool`, `is_hex_digit(c u8) bool`, `is_print(c u8) bool`, `is_punct(c u8) bool`, `is_cntrl(c u8) bool`, `to_lower(c u8) u8`, `to_upper(c u8) u8` |
| **`eventbus`** | `TopicStats`, `actor Topic[T]` (`.consumer`, `.unregister`, `.publish`, `.send`, `.consumers`, `.stats`), `actor Service[T, R]` (`.handler`, `.unregister`, `.handlers_count`, `.pick`), `request[T, R](svc Ref[Service[T, R]], v T) !R` |
| **`exec`** | `Command` (`Command.new`, `command`, `.set_dir`, `.set_clone_flags`, `.set_stdin`, `.set_inherit_stdio`, `.all_args`, `.run`/`.Run`, `.output`/`.Output`, `.combined_output`/`.CombinedOutput`, `.start`/`.Start`, `.wait`/`.Wait`), `combined_output(name, extra) !string` |
| **`flag`** | `Flag`, `FlagSet` (`FlagSet.new`, `.bool_flag`, `.int_flag`, `.i64_flag`, `.f64_flag`, `.string_flag`, `.parse`, `.get_bool`, `.get_int`, `.get_i64`, `.get_f64`, `.get_string`, `.positional`, `.help_string`, `.print_help`), `ansi_bold`, `ansi_red`, `ansi_green`, `ansi_yellow`, `ansi_blue`, `ansi_cyan` |
| **`gpu`** | `DeviceInfo`, `backend_auto`, `backend_cuda`, `backend_opencl`, `backend_cpu`, `is_available`, `is_enabled`, `set_enabled`, `has_tensor_cores`, `device_name`, `backend_name`, `set_backend`, `set_device`, `vram_bytes`, `vram_mb`, `allocated_bytes`, `allocated_kb`, `oom_fallbacks`, `simulate_oom`, `stats_reset`, `stats_print`, `sync`, `wait`, `kv_f16`, `info`, `linear`, `matmul`, `rmsnorm`, `rope`, `rope_norm`, `rope_seq`, `rope_seq_norm`, `softmax`, `silu`, `linear_swiglu`, `silu_mul`, `gqa_attention`, `gqa_prefill` |
| **`http`** | `MAX_HEAD_BYTES`, `READ_CHUNK`, `FILL_BUDGET`, `DEFAULT_MAX_BODY_BYTES`, `MAX_HEADER_LINE`, `MAX_HEADER_COUNT`, `HEAD_DEADLINE_MS`; `Header`, `HeaderMap` (`.new`, `.set`, `.get`, `.get_or`, `.contains`, `.format`), `Request` (`.new`, `.parse`, `.parse_limit`, `.parse_with`), `Response` (`.new`, `.ok`, `.text`, `.json`, `.with_status`, `.bad_request`, `.not_found`, `.internal_error`, `.set_header`, `.format`, `.send`), `Conn` (`.new`, `.fill`, `.next_request`, `.queue`, `.queue_bytes`, `.queue_response`, `.flush`, `.done`, `.fd`, `.pending_out`, `.close`), `Incoming`, `HttpServer` (`.bind`, `.accept_request`, `.local_addr`, `.close`), `RequestContext` (`.respond`, `.respond_text`, `.respond_json`, `.sse`), `SseWriter` (`.start`, `.start_cors`, `.send_event`, `.send_data`, `.send_token`, `.send_done`, `.close`), `parse_head`, `sse_head`, `sse_frame` |
| **`io`** | Interfaces: `Reader`, `Writer`, `Closer`, `ReadCloser`, `WriteCloser`; `File` (`.open`, `.read`, `.write`, `.write_string`, `.seek`, `.flush`, `.size`, `.close`), `BufReader` (`.new`, `.from_string`, `.from_bytes`, `.read_byte`, `.read_exact`, `.read_line`, `.read_all`, `.close`); Functions: `prompt`, `read_line`, `read_all_stdin`, `remove`, `rename`, `temp_file` |
| **`json`** | `JsonBuilder` (`.new`, `.set_str`, `.set_int`, `.set_i64`, `.set_bool`, `.set_null`, `.set_raw`, `.build`); Functions: `is_valid`, `has_key`, `get_str`, `get_int`, `get_i64`, `get_bool`, `get_raw`, `array_len`, `array_get_str`, `array_get_raw`, `escape`, `unescape` |
| **`log`** | `LEVEL_DEBUG`, `LEVEL_INFO`, `LEVEL_WARN`, `LEVEL_ERROR`, `LEVEL_FATAL`; `Logger` (`.new`, `.set_level`, `.set_prefix`, `.set_timestamp`, `.debug`, `.info`, `.warn`, `.error`, `.fatal`); Package helpers: `debug`, `info`, `warn`, `error`, `fatal` |
| **`math`** | `PI`, `TAU`, `E`, `SQRT2`; `abs`, `abs_int`, `abs_i64`, `min`, `max`, `min_int`, `max_int`, `min_i64`, `max_i64`, `clamp`, `clamp_int`, `clamp_i64`, `sqrt`, `cbrt`, `pow`, `exp`, `exp2`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `round`, `trunc`, `fmod`, `hypot`, `is_nan`, `is_inf`, `srand`, `rand_int`, `rand_f64`, `sort_ints`, `sort_f64s` |
| **`net`** | `POLL_READ`, `POLL_WRITE`, `POLL_ERROR`; `SocketAddr` (`.new`, `.parse`, `.to_string`), `PollEvent`, `Poller` (`.new`, `.poll_one`, `.wait_readable`, `.wait_writable`), `poll_many`, `TcpStream` (`.connect`, `.connect_timeout`, `.read`, `.read_append`, `.read_string`, `.read_exact`, `.read_line`, `.read_line_limit`, `.read_line_limit_idle`, `.read_all`, `.read_byte`, `.write`, `.write_str`, `.write_string`, `.write_at`, `.set_nonblocking`, `.set_nodelay`, `.set_timeout`, `.shutdown`, `.local_addr`, `.peer_addr`, `.close`), `TcpListener` (`.bind`, `.accept`, `.set_nonblocking`, `.local_addr`, `.close`), `UdpPacket`, `UdpSocket` (`.bind`, `.recv_from`, `.send_to`, `.send_to_string`, `.set_nonblocking`, `.set_timeout`, `.local_addr`, `.close`), `new_bytes`, `append_bytes`, `append_string`, `find_bytes` |
| **`os`** | `args`, `environ`, `getenv`, `setenv`, `unsetenv`, `executable`, `hostname`, `platform`, `is_windows`, `getpid`, `getppid`, `getuid`, `getgid`, `exists`, `stat`/`Stat`, `is_file`, `is_dir`, `read_file`, `write_file`, `read_bytes`, `read_bytes_head`, `write_bytes`, `read_dir`, `mkdir`, `mkdir_all`, `remove`/`Remove`, `remove_all`, `join_path`, `base_name`, `dir_name`, `ext`, `clean_path`, `set_quiet_rt`, `exit`/`Exit` |
| **`strings`** | `StringBuilder` (`.new`, `.write`, `.write_line`, `.clear`, `.len`, `.to_string`); Functions: `compare`, `count`, `fields`, `has_prefix`, `has_suffix`, `join`, `split`, `trim`, `trim_prefix`, `trim_suffix`, `to_lower`, `to_upper`, `pad_left`, `pad_right`, `format_i64_base`, `parse_i64_base`, `sort` |
| **`syscall`** | Flags: `O_*`, `CLONE_NEW*`, `MS_*`, `MNT_*`, `S_IF*`, `SIG*`; Functions: `open`, `close`, `read`, `write`, `lseek`, `dup`, `dup2`, `pipe`, `isatty`, `getpid`, `getppid`, `getuid`, `getgid`, `get_errno`, `strerror`, `kill`, `signal_notify`, `signal_poll`, `signal_wait`, `unshare`, `mount`, `unmount`/`Unmount`, `chroot`/`Chroot`, `sethostname`/`Sethostname`, `mknod`/`Mknod`, `makedev` |
| **`time`** | `NANOSECOND`, `MICROSECOND`, `MILLISECOND`, `SECOND`, `MINUTE`, `HOUR`; `Duration` (`.from_nanos`, `.from_millis`, `.from_secs`, `.millis`, `.secs`); Functions: `now_ns`, `now_ms`, `now_unix`, `monotonic_ns`, `sleep_ms`, `sleep_ms_i64`, `sleep_us`, `format` |
| **`viraserve`** | `Request` (`.new`, `.is_get`, `.is_post`, `.is_put`, `.is_delete`, `.json_str`, `.json_int`, `.json_f32`), `Response` (`.new`, `.status`, `.text`, `.json`, `.html`, `.bad_request`, `.not_found`, `.internal_error`, `.serialize`), `ServerStats` (`.new`, `.record`, `.avg_latency_us`), `RouteID` (`.not_found`, `.home`, `.health`, `.metrics`, `.predict`, `.users`, `.echo`), `Router` (`.new`, `.match_route`, `.describe_route`), `actor RequestWorker` (`.execute_route`), `ViraServe` (`.new`, `.dispatch_request`, `.listen_and_serve`, `.serve`, `.serve_one`, `.print_banner`, `.print_metrics`), `log_access` |
| **`widget`** | Theme & constants: `COLOR_*`, `Theme` (`.mocha`, `.latte`, `.midnight`), `EdgeInsets` (`.all`, `.symmetric`, `.only`, `.zero`), `TextStyle` (`.heading`, `.subheading`, `.body`, `.caption`); Navigation & layout: `App` (`.new`, `.init`, `.should_close`, `.begin_frame`, `.end_frame`, `.close`, `.cancel_close`, `.cleanup`), `Scaffold`, `Modal` / `modal_*` / `dialog_*`, `Tab`/`tab`, `Column`, `Row`, `Card`, `Padding`, `Center`, `ScrollView`, `SingleChildScrollView`, `Divider`/`divider`, `Spacer`/`spacer` (+ snake_case `*_begin`/`*_end` helpers, `center_offset`, `mouse_wheel_delta`); Responsive: `Screen`, `MediaQuery` (`.of`, `.width`, `.height`, `.pct_w`, `.pct_h`, `.content_width`, `.is_mobile`, `.is_tablet`, `.is_desktop`) + top-level equivalents; Controls & Data (`PascalCase` and `snake_case`): `Button`, `ElevatedButton`, `SecondaryButton`, `DangerButton`, `TextField`, `Slider`, `Checkbox`, `Radio`, `Dropdown`, `Select`, `Heading`, `Subheading`, `Text`, `TextWrap`, `Caption`, `Badge`, `StatusBadge`, `ProgressBar`, `ProgressPercent`, `DataTable`, `LoadImage`, `Image`, `FreeImage` |
