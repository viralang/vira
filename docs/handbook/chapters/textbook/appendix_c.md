# Appendix C: C99 Lowering & Runtime Memory Layout Guide

Vira lowers every module into a single portable C99 translation unit backed by the modular `runtime/rt/` headers (`runtime/vira_rt.h`). This appendix documents the exact C99 symbol mangling, memory layouts, concurrency structures, and tensor/quantization tables implemented in `internal/cgen/` and `runtime/rt/`.

---

## C.1. C99 Symbol Mangling & Scalar Mapping

### Symbol Mangling (`internal/cgen/cgen.go`, `closures.go`)

| Vira Construct | Generated C99 Symbol | Notes |
| :--- | :--- | :--- |
| Entry point `fn main()` | `int main(int argc, char** argv)` | Captures CLI args (`_vira_rt_init_args`), calls `vira_report_mem_stats()` before returning `0`. |
| Module function `pkg.foo` | `pkg__foo` | Main module functions use `main__foo`. |
| Struct / Actor / Interface | `pkg__Name` | `typedef struct pkg__Name pkg__Name;`. Actors are passed by handle (`pkg__Name*`). |
| Method / Static method | `pkg__Type_method` | Receiver methods take `const pkg__Type*` (or `pkg__Type*` if `mut`) as first argument. |
| Enum name helper | `pkg__Enum__name(int32_t v)` | Used by `${e}` string interpolation to print the variant name. |
| `@[export]` / `@[export: "c_fn"]` | `VIRA_EXPORT R fn(...)` / `c_fn` | Unmangled C-ABI linkage. |
| `extern "C" { fn foo(...) }` | `extern R foo(...)` | Emitted as direct C prototype; preserves `CName`. |
| Generic instance `f[int]`, `Box[int]` | `pkg__f__int`, `pkg__Box__int` | Monomorphized in the checker before codegen; zero runtime dispatch. |
| Result wrapper `!T` | `Result_<T>` (or `Result_void`, `Result_closure`) | Struct `{ bool is_err; vira_string err; T val; }`. |
| Slice `[]T` / Fixed array `[N]T` | `vira_slice_<T>` / `vira_array_<N>_<T>` | Predefined slices (`u8`, `int`, `i64`, `f32`, `f64`, `bool`, `string`, `[][]string`) live in `slice.h`. |

### Scalar & Primitive Type Mapping (`cType`)

| Vira Type | C99 Type | Vira C-ABI Type | C99 Type |
| :--- | :--- | :--- | :--- |
| `int` | `int32_t` | `c_char` / `c_uchar` | `char` / `unsigned char` |
| `i64` | `int64_t` | `c_short` / `c_ushort` | `short` / `unsigned short` |
| `u8` / `byte` | `uint8_t` | `c_int` / `c_uint` | `int` / `unsigned int` |
| `f32` | `vira_f32` (`float`) | `c_long` / `c_ulong` | `long` / `unsigned long` |
| `f64` | `double` | `c_longlong` / `c_ulonglong` | `long long` / `unsigned long long` |
| `bf16` | `vira_bf16` (`uint16_t`) | `c_size` / `c_ssize` | `size_t` / `ssize_t` |
| `bool` | `bool` | `c_float` / `c_double` | `float` / `double` |
| `enum` | `int32_t` | `voidptr` | `void*` |

---

## C.2. Strings, Slices & Maps

### 16-Byte `vira_string` (`runtime/rt/string/string.h`)

```c
typedef struct {
  const uint8_t *data;
  int32_t len;
  bool is_lit;
  bool is_borrowed;
} vira_string; // 16 bytes on 64-bit
```

Three ownership states share the two `bool` flags inside existing struct padding:
1. **`is_lit == false` (OWNED):** `malloc`-allocated NUL-terminated buffer; freed by `vira_str_free`.
2. **`is_lit == true, is_borrowed == false` (STATIC):** Compile-time literal (`vira_str_lit`); `vira_str_free` and `vira_str_clone` are no-ops.
3. **`is_lit == true, is_borrowed == true` (BORROWED):** Non-owning view into an arena slab (`vira_arena_str_format`), parent string (`vira_str_char_at`, `trim_left`), or map key (`vira_map_entry_skey`). Before escaping to a field, slice, or return value, `vira_str_own()` clones borrowed views to the heap.

- **Single-Allocation Concatenation:** `a + b + c` flattens all leaves into one `vira_str_concat_n(parts, n)` call (or `vira_arena_str_concat_n` inside an `arena`).
- **FFI C-String Marshaling:** `vira_str_ffi_cstr(s)` passes owned strings directly (already NUL-terminated) and copies literals/views into a 16-slot thread-local ring buffer (`VIRA_FFI_CSTR_SLOTS = 16`).

### Dynamic Slices `vira_slice_<T>` (`runtime/rt/slice/slice.h`, `cgen.go`)

```c
typedef struct {
  T *data;
  int32_t len;
  int32_t cap;
  bool is_arena;
} vira_slice_<T>;
```

- `_vira_slice_grow` starts at capacity `4`, doubles (`* 2`) up to `INT32_MAX` (faulting via `_vira_rt_fault` on overflow), and records allocations in `_vira_active_heap_allocs`.
- Structs and containers with heap fields receive generated `<pkg>__<Struct>_autofree` and `<pkg>__<Struct>_clone` helpers. Slices of heap-owning types free every element in `<slice>_free` when `!is_arena`. Slices of actors (`[]MyActor`) store `pkg__MyActor*` handles (`T** data`), so growing never relocates actor mailboxes.

### Insertion-Ordered Hash Map `vira_map` (`runtime/rt/map/map.h`, `maps.go`)

```c
typedef struct {
  size_t val_size;
  bool key_is_str;
  void (*clone_val)(void *dst, const void *src); // NULL if POD
  void (*free_val)(void *val);                   // NULL if POD
} vira_map_ops;

typedef struct vira_map {
  const vira_map_ops *ops;
  char *entries;     // dense insertion-ordered array: n_entries * stride bytes
  int32_t *index;    // open-addressing hash table storing (entry_idx + 1), 0 = empty
  int32_t len;       // live entries
  int32_t n_entries; // live + dead slots
  int32_t cap_entries;
  int32_t index_cap;
} vira_map;
```

- **Hashing:** 64-bit FNV-1a for `string` keys; 64-bit SplitMix-style avalanche (`0xff51afd7ed558ccdULL`, `0xc4ceb9fe1a85ec53ULL`) for integer/bool/enum keys widened to `int64_t`.
- **Order & Compaction:** Entries are appended sequentially in `entries` (16-byte aligned header `vira_map_entry_hdr` + 16-byte aligned value payload) so `for k, v in m` (`vira_map_next`) preserves insertion order. `vira_map_delete` marks `e->live = false`; `_vira_map_rebuild` compacts dead entries when growing or when load exceeds 50% of `index_cap`.

---

## C.3. Interfaces & Reference-Counted Closures

### 16-Byte Interface Fat Pointer (`internal/cgen/cgen.go`)

```c
typedef struct pkg__Iface_vtable {
  Ret (*method1)(const void* _self, ...);
  void (*_autofree)(void* _self);
} pkg__Iface_vtable;

typedef struct pkg__Iface {
  void* _object;
  const pkg__Iface_vtable* _vtable;
} pkg__Iface; // 16 bytes
```

- Converting a struct `S` to interface `I` invokes `I_box_S(S val)`, which `malloc`s a heap copy of `S` and pairs it with the static `const` vtable `_vtable_S_for_I`. When the interface leaves scope, `I_autofree` calls `_vtable->_autofree(_object)` (`S_autofree` + `free(_object)`).

### Atomic Ref-Counted Closures (`runtime/rt/closure/closure.h`, `closures.go`)

```c
typedef struct vira_closure_env {
  int32_t refs;
  void (*drop)(struct vira_closure_env *env);
} vira_closure_env;

typedef struct {
  void *fn;              // trampoline: R (*)(vira_closure_env*, A...)
  vira_closure_env *env; // NULL if zero captures or named function thunk
} vira_closure;          // 16 bytes
```

- The checker lifts every closure literal to a top-level function `_lambda_N(captures..., params...)` and codegen emits a uniform trampoline + `vira_closure_env`-prefixed capture struct. Named functions used as values receive a zero-env `_fnref_<sym>` thunk.
- Because closures can be sent across actor threads, `vira_closure_retain` and `vira_closure_release` maintain `env->refs` atomically (`__atomic_add_fetch(..., __ATOMIC_RELAXED)` and `__atomic_sub_fetch(..., __ATOMIC_ACQ_REL)`), calling `env->drop(env)` when the last reference drops.

---

## C.4. Actor Handles, Chase-Lev Scheduler, Supervision & Cycle Detection

### Actor Memory & Lazy Arena (`scheduler.h`, `cgen.go`)

- Every `actor` is allocated on the heap (`malloc(sizeof(pkg__Actor))`) and passed by handle (`pkg__Actor*`). Its first field is `vira_actor_ctx _ctx`:
  - `isolated_heap` (`vira_arena`) starts completely zeroed (`buf = NULL`, `cap = 0`, no slab allocated until first allocation), keeping idle actor creation at ~64 bytes of context overhead (~408 bytes resident after first message).
  - Single-owner cleanup calls `pkg__Actor__vira_release(&h)` → `vira_actor_ctx_destroy(&self->_ctx)` (which closes its `Ref` slot, runs `vira_actor_quiesce` to wait for any active drain task with work-helping, frees `defer_stack`, and destroys `isolated_heap`) → frees struct fields → `free(*h)`.

### Work-Stealing Pool & Compute Team (`runtime/rt/scheduler/scheduler.h`)

- **Worker Pool (`_vira_sched`):** Up to `VIRA_MAX_WORKERS = 32` worker pthreads (default logical cores clamped to `[2, 16]`, overridable via `VIRA_WORKERS`). Each worker owns a cache-line-aligned `vira_ws_deque` of `VIRA_WS_CAP = 4096` slots (`top` CAS-stolen FIFO by thieves, `bottom` LIFO pushed/popped by owner, `push_lock` for external multi-producer sends) backed by a mutex-guarded `vira_overflow_queue`. Idle workers spin (`VIRA_SPIN_ITERS = 64`), yield (`VIRA_YIELD_ITERS = 128`), and park on a `CLOCK_MONOTONIC` condvar (`50 ms` fallback poll).
- **Mailboxes & Quotas:** Messages are linked via intrusive MPSC `next_actor_msg` pointers (`mbox_head` / `mbox_tail`). Actor state transitions atomically between `VIRA_ACTOR_IDLE (0)`, `VIRA_ACTOR_SCHEDULED (1)`, and `VIRA_ACTOR_NOTIFIED (2)`. `_vira_actor_drain_task` processes up to `quota = 64` messages per turn before re-queueing.
- **Work-Helping `Future[T].await()`:** `vira_scheduler_await` calls `_vira_sched_help_one` (capped at `VIRA_MAX_HELP_DEPTH = 32` to prevent C stack overflow) so waiting threads execute pending tasks instead of blocking.
- **Persistent Compute Team (`_vira_team`):** Tensor/LLM parallel CPU kernels use up to `VIRA_TEAM_MAX_THREADS = 16` spin-then-sleep pthreads (default physical cores, overridable via `VIRA_THREADS`; disabled inside active actors to avoid oversubscription).

### `setjmp`/`longjmp` Fault Containment & `Ref[T]`

1. Before invoking a handler, the generated wrapper binds `_vira_tls_actor = &self->_ctx`, sets `defer_top = 0`, and arms `setjmp(self->_ctx.fault_boundary)`.
2. Any `panic(...)` or `_vira_rt_fault(...)` (OOB index, `pop()` on empty slice, integer `/ 0` or `% 0`, tensor/LLM fault) inside an active actor calls `_vira_actor_defer_unwind(ctx)` (releasing stack-frame heap allocations in LIFO order), saves `last_error`, and `longjmp`s back to the boundary.
3. The supervisor resets `isolated_heap` in $O(1)$ (`vira_arena_reset`), increments `restarts`, and if `restarts < max_restarts` (default `3`) runs `pkg__Actor_supervisor_reset(self)`—freeing and resetting **only `mut:` / `pub mut:` fields** while preserving immutable configuration fields. Once `max_restarts` is exceeded, `state` becomes `VIRA_ACTOR_STOPPED`.
4. **`Ref[T]` (`uint64_t vira_ref`):** Allocates a slot in chunked `_vira_ref_chunks` (4,096 slots/chunk, up to 4,096 chunks = 16M actors) packing `(generation << 32) | (slot + 1)`. Sends pin the slot (`VIRA_REF_PINS`), enqueue, and unpin; actor destruction sets `VIRA_REF_CLOSING`, waits for pins to drain, and bumps `generation` so stale refs fail cleanly with `"actor is gone"`.

### Wait-For Graph Cycle Detection

- Each actor receives a `wf_slot` in non-freed `_vira_wf_chunks` storing a 64-bit `edge = (wait_seq << 32) | (target_slot + 1)`.
- While inside `vira_actor_await_msg`, the waiter publishes its edge, pushes `me->wf_slot` onto `_vira_tls_wf_stack` (tracking actors suspended beneath work-helping frames), and every 256 spin iterations calls `_vira_wf_walk`.
- If two consecutive walks (`prev` and `cur`) observe the **exact same edge sequence** reaching an actor on `_vira_tls_wf_stack`, a true cycle exists; `vira_actor_await_msg` atomically CASes `task->done` from `0` to `VIRA_TASK_DETACHED`, abandons the wait, and returns a cycle error instead of deadlocking.

---

## C.5. 64-Byte Aligned Bump Arenas & LIFO Lexical `defer` Lowering

### Bump Arenas (`runtime/rt/arena/arena.h`)

- `arena { ... }` initializes a `vira_arena` with a **1 MB** (`1024 * 1024`) slab (`_arena_N`).
- Every `vira_arena_alloc` rounds both the base offset and allocation size up to **64-byte cache-line alignment** (`(x + 63) & ~63`) for AVX-512 / AVX2 / NEON loads. When a slab fills, it chains the old slab onto `overflow_slabs` and doubles capacity.
- Exiting the `arena` block calls `vira_arena_destroy`, releasing all slabs in $O(\text{slabs})$ with zero per-object `free()` calls.

### Lexical `defer` Lowering (`internal/autofree/`, `cgen.go`)

- **Normal Exit:** `internal/autofree` inserts `DeferStmt` bodies and `DeferFreeStmt` destructors in **strict LIFO reverse order** at every lexical block exit (`}`, `return`, `break`, `continue`).
- **Arena-Safe Defers (`ArenaScoped`):** A `defer` written inside a specific `arena` slab evaluates against that slab (`s.ArenaVar`) even if triggered by an early `return` from a nested `arena` block whose inner slab has already been torn down.
- **Fault Unwind Stack:** In non-`main` functions, owning locals push `(fn, ptr)` pairs onto the active actor's dynamic `defer_stack` (`_vira_actor_defer_push`) and pop them on normal `DeferFreeStmt` execution (`_vira_actor_defer_pop`), bounded at function entry/return via `_vira_actor_defer_mark()` / `_vira_actor_defer_restore(_vira_defer_base)`.

---

## C.6. Tensors, Quantization Registry, TLS Cache & GGUF LRU Cache

### `vira_tensor_f32` & 64-bit `numel()` (`runtime/rt/tensor/core.h`)

```c
typedef struct {
  vira_f32 *data;  // 64-byte aligned pointer
  void *raw_alloc; // underlying malloc/VirtualAlloc pointer
  int32_t rows;    // flattened outer dims (1 for 1D)
  int32_t cols;    // trailing dim dims[n_dims - 1]
  int32_t dims[4];
  int32_t strides[4];
  int32_t n_dims;
  bool is_arena;
  bool is_view;
  bool is_vmem;    // reserved virtual memory for lazy host FP32 on Windows
} vira_tensor_f32;
```

- `tensor.numel()` lowers to `vira_tensor_numel64(t)` returning `int64_t` (`i64`) so tensors exceeding $2^{31}-1$ elements do not overflow.

### `_vira_rt_quant_table` & 256-Entry `_vira_tls_qcache` (`quant_registry.h`)

- When `gguf_load_tensor` loads a 2D+ quantized tensor with $\ge 65{,}536$ elements (and `VIRA_LAZY_HOST_FP32 != "0"`), it stores the raw quantized blocks in `_vira_rt_quant_table` (an open-addressed hash table keyed by `host_ptr`, guarded by a 4,096-byte atomic Bloom filter `_vira_rt_quant_bloom` so non-quantized tensors skip locking in $O(1)$) and defers host FP32 dequantization (`fp32_state = 0`).
- Hot CPU SIMD / GPU dispatch paths look up `host_ptr` via `_vira_rt_quant_try_get`, which first probes a **lock-free 256-entry thread-local cache** (`static __thread _vira_tls_qcache_entry _vira_tls_qcache[256]`) indexed by `hash(host_ptr) & 255` and validated against the global monotonic generation counter `_vira_rt_quant_gen`.

### 4-Slot LRU GGUF Directory Cache (`runtime/rt/weights/gguf.h`)

- `vira_gguf_load_tensor` maintains a mutex-protected **4-slot LRU cache** (`VIRA_GGUF_DIR_CACHE_SLOTS = 4`) of open GGUF files (`FILE *fp` with 64 KB stdio buffer, FNV-1a open-addressed `hash_slots` tensor directory, and `file_size`/`file_mtime`/`file_ino` identity checks).
- Parsing enforces header count sanity bounds (`kv_count <= file_size / 12`, `tensor_count <= file_size / 24`) and full file-extent bounds checks before seeking or reading tensor blocks.
