# Chapter 17: Unsafe Vira & Native C / Python FFI

Vira compiles directly to C99, which gives it zero-overhead access to the entire C ecosystem—operating system APIs, existing C/C++ libraries, and high-performance native shared libraries callable from Python, C, or Rust. At the same time, Vira refuses to let raw pointer operations erode the memory safety of ordinary code. Every foreign call and pointer dereference is fenced behind an explicit `unsafe` boundary, and Vira provides exact C-ABI primitive types, predictable struct layouts, and `@[export]` linkage to make cross-language boundaries both fast and auditable.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch17_unsafe_and_c_ffi.vir`](../code/ch17_unsafe_and_c_ffi.vir)

---

## 17.1. The `unsafe { ... }` Boundary and Raw Pointers (`&x`, `&mut x`, `*ptr`, `voidptr`)

In safe Vira code, Autofree manages heap allocations automatically, slices perform bounds checks, and values cannot be aliased unsafely across threads. Whenever you need to call an `extern "C"` function or dereference a raw pointer, the type checker requires you to opt in explicitly with an `unsafe { ... }` block (or mark an entire helper function with `@[unsafe]`):

```v
mut counter := 42
ro_ptr := &counter      // Immutable pointer (&int)
mut_ptr := &mut counter // Mutable pointer (&mut int)

unsafe {
	val := *ro_ptr
	println('Dereferenced value: ${val}')
}
```

### Raw Pointer Primitives

| Syntax | Type | Meaning & Rules |
| :--- | :--- | :--- |
| **`&x`** | `&T` | Takes a read-only address-of pointer to `x`. |
| **`&mut x`** | `&mut T` | Takes a mutable pointer to a `mut` variable `x` for passing to C out-parameters. |
| **`*ptr`** | `T` | Dereferences `ptr` inside `unsafe { ... }` to read the underlying value. |
| **`voidptr`** | `void*` | Untyped C pointer; accepts any `&T`, `&mut T`, or integer `0` (C `NULL`). |

Because `voidptr` interoperates seamlessly with typed references and `0` (`NULL`), you can write clean null-guard helpers without manual casts:

```v
pub fn is_non_null_ptr(p voidptr) bool {
	return p != 0
}
```

> **Encapsulate `unsafe` in Safe Wrappers:** Keep `unsafe { ... }` blocks small and local. Wrap low-level C calls inside normal Vira functions (`pub fn safe_c_strlen(text string) int`) that validate arguments and return clean Vira values or `!T` Results, so callers outside the wrapper never need `unsafe`.

---

## 17.2. C-ABI Primitive Types, `@[c_layout]` Structs, and `@[opaque]` Handles

C compilers vary integer widths across platforms (`long` is 32 bits on Windows x64 LLP64, 64 bits on Linux x86_64 LP64). To guarantee an exact Application Binary Interface (ABI) match with your system C headers, Vira provides dedicated C-ABI scalar primitives:

| Vira Type | C99 Equivalent | Typical Use |
| :--- | :--- | :--- |
| **`c_char`** / **`c_uchar`** | `char` / `unsigned char` | C character strings (`&c_char` maps to `const char*`). |
| **`c_short`** / **`c_ushort`** | `short` / `unsigned short` | 16-bit C struct fields and socket port fields. |
| **`c_int`** / **`c_uint`** | `int` / `unsigned int` | C status codes, POSIX file descriptors, boolean flags. |
| **`c_long`** / **`c_ulong`** | `long` / `unsigned long` | Platform-width C `long` APIs. |
| **`c_longlong`** / **`c_ulonglong`** | `long long` / `unsigned long long` | 64-bit C integers. |
| **`c_size`** / **`c_ssize`** | `size_t` / `ssize_t` | Byte counts, buffer lengths (`strlen`, `fread`). |
| **`c_float`** / **`c_double`** | `float` / `double` | C floating-point math (`sqrt`, `hypot`). |
| **`voidptr`** | `void*` | Untyped foreign pointers and opaque contexts. |

### Exact Memory Layout with `@[c_layout]`

Standard Vira structs may reorder or augment fields to support Autofree metadata. Annotate a struct with `@[c_layout]` to force standard C field ordering, alignment, and padding so it matches a C `struct` or Python `ctypes.Structure` byte-for-byte:

```v
@[c_layout]
pub struct NativeVec2 {
pub mut:
	x c_double
	y c_double
}
```

### Foreign Opaque Handles with `@[opaque]`

Many C libraries (such as SQLite's `sqlite3*`, OpenSSL's `SSL_CTX*`, or compression streams) allocate structs internally and never expose their fields in public headers. Declare those handles with `@[opaque]` so Vira represents them as strongly typed pointers without requiring field definitions or generating struct destructors:

```v
@[opaque]
pub struct OpaqueContext {}
```

---

## 17.3. Calling C Libraries with `#include`, `#flag`, and `extern "C"`

Integrating an external C library requires three declarations at the top of a `.vir` file:

1. **`#include <header.h>`** — Emits the C `#include` directive at the top of the generated C99 translation unit.
2. **`#flag <args>`** — Forwards compiler and linker flags directly to `gcc` / `clang` / `mingw-w64` (for example, `#flag -lm` or OS-scoped `#flag linux -lsqlite3`, `#flag windows -lws2_32`).
3. **`extern "C" { ... }`** — Declares the exact C function signatures so the Vira type checker validates every call site.

```v
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#flag -lm

extern "C" {
	fn strlen(s &c_char) c_size
	fn sqrt(x c_double) c_double
	fn hypot(x c_double, y c_double) c_double
	fn abs(n c_int) c_int
	fn atoi(s &c_char) c_int
}
```

### Automatic String Marshaling & Safe Wrappers

When an `extern "C"` parameter is typed as `&c_char`, you can pass a Vira `string` directly inside `unsafe { ... }`: Vira marshals the underlying UTF-8 bytes as a null-terminated `const char*` buffer automatically.

Wrapping those unsafe calls in safe Vira functions isolates the FFI boundary cleanly:

```v
pub fn safe_c_strlen(text string) int {
	unsafe {
		return int(strlen(text))
	}
}

pub fn safe_vec2_norm(v NativeVec2) f64 {
	unsafe {
		return sqrt(v.x * v.x + v.y * v.y)
	}
}
```

---

## 17.4. Exporting Shared Libraries (`@[export]`, `vira build -shared`) for Python `ctypes`, C, and Rust

Vira is not limited to *calling* C—it can also *act as* a native C library for Python, C, C++, Go, or Rust. Annotate any `pub fn` whose parameters and return type use C-ABI types with `@[export: "symbol_name"]`:

```v
@[export: "vira_add_integers"]
pub fn vira_add_integers(a c_int, b c_int) c_int {
	return a + b
}

@[export: "vira_vec2_norm"]
pub fn vira_vec2_norm(x c_double, y c_double) c_double {
	unsafe {
		return sqrt(x * x + y * y)
	}
}

@[export: "vira_string_length"]
pub fn vira_string_length(s string) c_size {
	unsafe {
		return strlen(s)
	}
}
```

Build a platform-native shared library (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows) using `vira build -shared`:

```bash
./vira build -shared -o /tmp/ch17_test.so docs/handbook/chapters/code/ch17_unsafe_and_c_ffi.vir
```

### Calling from Python (`ctypes`)

Once compiled, Python can load the shared library with zero Cython or `setuptools` boilerplate (see also [`docs/examples/40_python_ffi_caller.py`](../../../examples/40_python_ffi_caller.py)):

```python
import ctypes

lib = ctypes.CDLL("/tmp/ch17_test.so")
lib.vira_add_integers.argtypes = [ctypes.c_int, ctypes.c_int]
lib.vira_add_integers.restype = ctypes.c_int

lib.vira_vec2_norm.argtypes = [ctypes.c_double, ctypes.c_double]
lib.vira_vec2_norm.restype = ctypes.c_double

assert lib.vira_add_integers(19, 23) == 42
assert abs(lib.vira_vec2_norm(6.0, 8.0) - 10.0) < 1e-6
```

Similarly, a C or Rust caller only needs an `extern "C"` declaration matching `int vira_add_integers(int a, int b);` and `double vira_vec2_norm(double x, double y);` and can link directly against the generated `.so`/`.dylib`/`.dll`.

---

## 17.5. Running the Complete Chapter 17 Example

Run [`docs/handbook/chapters/code/ch17_unsafe_and_c_ffi.vir`](../code/ch17_unsafe_and_c_ffi.vir) and verify the shared library build:

```bash
./vira run docs/handbook/chapters/code/ch17_unsafe_and_c_ffi.vir
./vira build -shared -o /tmp/ch17_test.so docs/handbook/chapters/code/ch17_unsafe_and_c_ffi.vir
rm -f /tmp/ch17_test.so
```

Output:
```text
[1] Pointers: *ro_ptr=42, voidptr_non_nil=true
[2] C-ABI Struct: vec=(3.00, 4.00), norm=5.00
[3] C libc: strlen=10, sqrt(144)=12.00, hypot(5,12)=13.00, abs(-42)=42, atoi=2026
    Safe wrapper: safe_c_strlen=19
[4] Exported C-ABI: add=42, norm=10.00, strlen=14
[vira_rt] simd=x86_64 SSSE3 + SSE2 (128-bit Integer/Float SIMD, Multi-Threaded)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=5
✓ Built standalone binary /tmp/ch17_test.so in 2.074s (backend: gcc [dev (-O2 -march=native)])
```

---

## Summary & Exercises

1. **Enforced `unsafe` Boundary:** Copy the `strlen('test')` call outside the `unsafe { ... }` block in `ch17_unsafe_and_c_ffi.vir` and run `./vira check` to inspect the semantic error diagnostic requiring `unsafe` or `@[unsafe]`.
2. **Wrapping a POSIX / libc Function:** Add `#include <time.h>` and `fn time(t voidptr) c_long` inside `extern "C" { ... }`, write a safe `pub fn unix_epoch_sec() i64` wrapper passing `0` (`NULL`) as `voidptr`, and print the current timestamp.
3. **Python `ctypes` Integration:** Compile `ch17_unsafe_and_c_ffi.vir` with `vira build -shared -o /tmp/ch17_test.so` and run a short Python one-liner using `ctypes.CDLL('/tmp/ch17_test.so')` to verify `vira_add_integers(100, 23) == 123` and `vira_vec2_norm(5.0, 12.0) == 13.0` before removing the temporary `.so`.
