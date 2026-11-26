# Appendix D: Syntax Evolution Log

*Notes and language improvements made across Vira's development and while writing this book, grounded in `ROADMAP.md` and the compiler/runtime commit history (`v0.1` through `v0.9` + hardening).*

---

## D.1. Ownership, Autofree & Strings Refinement

- **Single-Owner Move vs. Deep Copy (`internal/autofree/autofree.go`):** Storing a heap-owning value (`string`, `[]T`, `map[K]V`, `tensor[f32]`, owning `struct`) copies it by default, **unless** static escape analysis proves the source variable is never read again—including across `if` / `switch` / `{ }` branches (`canMoveOuter` with `ZeroOnMove` zeroing) and when pushing/setting into slices (`MovedStoreArgs`)—in which case ownership moves with zero allocation.
- **Borrowed Function Arguments:** Functions and methods borrow parameters by default without cloning at call sites, while `mut` parameters/receivers pass pointers and dereference cleanly when passed or stored by value (`identValueExpr`).
- **Single-Allocation String `+` and `+=` Chains:** String concatenation (`a + b + c` and `s += t`) was added with checker type enforcement and C99 lowering (`emitStringConcatWith`) that flattens arbitrarily long chains into a single `vira_str_concat_n` (or `vira_arena_str_concat_n`) allocation.
- **Temporary Receiver, `slice.pop()`, and `.clone()` Hoisting (`V14`/`V15`):**
  - Unassigned temporary strings used as method receivers (e.g., `make_str().trim()`) are hoisted into synthetic locals (`_vira_tmp_str_N`) and freed after the statement.
  - Discarded `slice.pop()` statements on heap-owning element types (`xs.pop()`) now capture and free the popped element instead of leaking it.
  - Removed an obsolete autofree skip on `.clone()` temporaries, ensuring `xs.clone().len` and `m.keys().len` free their temporary containers cleanly.
- **Borrowed String Views (`is_borrowed`) & NUL-Safe Interpolation:** Added `is_borrowed` to `vira_string` so zero-copy views (`char_at`, `trim_left`, arena strings, map iteration keys) are automatically promoted to heap copies via `vira_str_own()` when escaping, and updated string interpolation to concatenate length-delimited pieces whenever a `string` hole is present so embedded NUL bytes never truncate output.

---

## D.2. Control Flow, Lexer, Parser & Cleanup Refinement

- **Multi-Branch `switch` Statements:** Added `switch` with automatic `break` (no implicit fallthrough), comma-separated cases (`case 200, 201:`), `string` matching via `vira_str_eq`, tagless boolean condition arms (`switch { case x >= 90: ... }`), and per-case LIFO autofree cleanup.
- **Lexical-Block `defer` & Arena-Safe Exit Points:** `defer` statements execute in strict LIFO order on every lexical exit (`}`, `return`, `break`, `continue`), record their enclosing `ArenaVar` (`ArenaScoped`) so early returns from nested `arena` blocks evaluate against the right slab, and register on the active actor's runtime `defer_stack` for fault unwinding.
- **Newline Expression Boundaries (`T-118`):** Stopped Pratt expression continuation across newlines before `@`, `+`, `-`, `[`, and `(` (`stopsExprAtNewline`), preventing multi-line statements like `x\n@[export]` or `a\n-b` from misparsing as binary/postfix operators, while adding unary `+` on numeric types.
- **Zero-Allocation Scientific Float Exponents (`T-087`):** Extended `internal/lexer/lexer.go` to parse `1e5`, `1e-12`, `2.5E-3`, and `3e2f32` without heap allocations, gated on a digit following `e`/`E` (+ optional sign) so `obj.e` member access remains unambiguous.
- **Strict Numeric Parsing & Overflow Guards (`T-088`, `T-112`, `T-116`):** `string.to_int()` and `.to_i64()` reject embedded NUL bytes and enforce `ERANGE` / `INT32` bounds; `_vira_slice_grow`, `str.repeat`, and `str.replace` guard against `INT32_MAX` overflow; C99 builds compile with `-fwrapv` and checked `vira_div_*` / `vira_mod_*` helpers.

---

## D.3. Actors, Concurrency, Generics & Type System (`v0.9`)

- **Non-Owning `Ref[T]`, `Future[T].ready()`, and `packages/eventbus`:** Because actors have single-owner handle semantics and never move, `actor.ref()` produces a copyable, generation-checked `Ref[T]` (`r.alive()`) enabling direct actor-to-actor graphs and typed pub/sub + request/reply (`Topic[T]`, `Service[T, R]`, `eventbus.request`). `Future[T].ready()` lets event loops poll futures without blocking.
- **Wait-For Cycle & Self-Await Deadlock Avoidance:** An actor awaiting a message sent to itself (`_vira_future_detach_if_self`), or actors forming a cyclic wait chain (`_vira_wf_walk` double-walk confirmation in `vira_actor_await_msg`), detaches unstarted messages and returns an error instead of hanging indefinitely.
- **Enums, Maps, Closures & Generics (`v0.9`):**
  - `enum` with optional explicit 32-bit discriminants, `switch` support, `int()` casts, and variant names in `${...}` interpolation.
  - Insertion-ordered `map[K]V` (`has`, `delete`, `keys`, `values`, `clear`, `for k, v in m`, zero-value reads on missing keys) backed by a single untyped `vira_map` + generated `vira_map_ops`.
  - First-class `fn (A) R` types, named function values, and value-captured read-only closures backed by atomically ref-counted `vira_closure_env`.
  - Monomorphized generic functions (`fn f[T](...)` with type inference) and generic structs/methods (`struct Box[T]`, `fn Box[T].new()`).

---

## D.4. High-Scale AI & Runtime Hardening

- **64-Bit `tensor.numel()` (`T-011`):** Widened `.numel()` from `int` (`int32_t`) to `i64` (`int64_t`, `vira_tensor_numel64`) so >2B-element LLM tensors report accurate counts.
- **Supervised Runtime Faults (`_vira_rt_fault`):** Converted fatal `exit(1)` calls on out-of-bounds indexing, `pop()` on empty slices, integer `/ 0` / `% 0`, and tensor/LLM runtime validation errors into `_vira_rt_fault`—trapped cleanly inside actors via `setjmp`/`longjmp` and `OneForOne` mutable-field-only restarts.
- **GGUF Hardening & 4-Slot LRU Cache (`T-012`, `T-109`):** `gguf_load_tensor` validates file extents and header counts, keeps raw blocks in `_vira_rt_quant_table` next to lazy FP32 tensors (`VIRA_LAZY_HOST_FP32`) with a 256-entry `_vira_tls_qcache`, and caches up to 4 open GGUF directories in a thread-safe LRU (`VIRA_GGUF_DIR_CACHE_SLOTS = 4`).
- **Hardened `http.SseWriter` (`T-091`, `T-113`):** Replaced default wildcard CORS with opt-in `SseWriter.start_cors(mut stream, allow_origin)` (with CRLF injection checks), transfers socket fd ownership (`stream.fd = i64(-1)`) to prevent double-close, formats multi-line `data:` frames, and JSON-escapes `send_token`.
- **Zero-Leak Enforcement & CLI Ergonomics (`T-086`, `T-115`, `T-116`):** `vira test` fails any test file with `active_leaks != 0` unless `-allow-leaks` is passed; runtime headers extract atomically into version-stamped `~/.vira/runtime/<version>`; CLI flags work in any position relative to file/directory targets.

---

## D.5. Documented Upcoming `v0.9.x` / `v1.0` Candidates

As tracked in `ROADMAP.md`, the following syntax and type-system extensions are planned for the next milestones:

- **Exhaustive Pattern Matching (`match` expressions):** Multi-branch expression matching over primitives, strings, `!T` Results, enums, and discriminated unions (`TokMatch` is already reserved in `internal/token/token.go`).
- **Zero-Copy Slice Range Expressions:** Native range syntax `buf[0..128]` for strings, slices, and tensors (the lexer already avoids consuming `.` into a float when followed by `.` for `..`).
- **Generic Extensions:** Explicit generic constraints/bounds, generic interfaces and enums, method-level type parameters, and explicit non-ident type arguments (e.g. `f[[]int]`).
- **Ecosystem & Tooling (`v1.0`):** `vira doc` static HTML documentation generator, `vira bench` harness, `crypto/*` and `compress/*` stdlib packages, and v1.0 syntax/ABI freeze.
