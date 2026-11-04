# Tutorial 38: Native C & Python FFI — Zero-Overhead Interop, Shared Libraries & ctypes

Modern systems and AI engineering require seamless interoperability with existing native ecosystems. Whether you need to invoke operating system APIs, interface with vendor SDKs (such as CUDA, Vulkan, or SQLite), or expose high-performance compiled routines to dynamic languages like Python, foreign function interop must be **zero-overhead, type-safe, and deterministic**.

Starting in **v0.7**, Vira provides first-class native C and Python FFI:
1. **Zero-overhead C-ABI calls**: Direct symbol invocation without wrapper trampolines or Marshalling overhead.
2. **Deterministic safety boundaries**: Mandatory `unsafe { ... }` blocks or `@[unsafe]` function attributes prevent memory safety violations from leaking into safe code.
3. **C-ABI primitive scalar types and raw pointers**: Explicit scalar types (`c_char`, `c_int`, `c_double`, `voidptr`) and reference/dereference operators (`&x`, `&mut x`, `*ptr`).
4. **C-compatible struct layouts and opaque handles**: `@[c_layout]` for predictable field alignment and `@[opaque]` for type-safe foreign pointer handles.
5. **Shared library compilation (`vira build -shared`)**: Compile Vira code directly into native `.dll`, `.so`, or `.dylib` shared libraries with exported C symbols via `@[export]`.
6. **Zero-boilerplate Python interop**: Call compiled Vira functions directly from Python using the standard library `ctypes` module.

---

## 1. Architectural Overview & C-ABI Contract

Vira compiles Ahead-Of-Time (AOT) to optimized C99, targeting GCC, Clang, or MinGW-w64. Because the intermediate representation is native C, interop with C is not an emulation or emulation layer—it is a **native ABI match**.

```
┌────────────────────────────────────────────────────────┐
│                   Safe Vira Code                       │
│  - Immutability by default                             │
│  - Deterministic compiler autofree (no GC)             │
│  - Strict Result monad error handling (!T)             │
└───────────────────────────┬────────────────────────────┘
                            │ Safe Wrapper API
┌───────────────────────────▼────────────────────────────┐
│              Unsafe Safety Boundary Block              │
│                      unsafe { }                        │
├───────────────────────────┬────────────────────────────┤
│   External C Invocations   │    Raw Pointer Operations  │
│   - extern "C" { puts }   │    - Address-of (&x, &mut) │
│   - libc / SDK symbols    │    - Dereference (*ptr)    │
│   - #include <stdio.h>    │    - voidptr casting       │
└───────────────────────────┴────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│      Compiled Shared Library (.dll / .so / .dylib)     │
│                 vira build -shared                     │
│               @[export: "c_symbol"]                    │
└───────────────────────────┬────────────────────────────┘
                            │ Direct C-ABI
┌───────────────────────────▼────────────────────────────┐
│           Python ctypes / C / C++ / Rust Callers       │
└────────────────────────────────────────────────────────┘
```

When Vira calls an external C function, the compiler generates a direct C function call. There are no runtime reflection lookups, no JNI bridges, and no garbage collection root scanning.

---

## 2. Native Directives: `#include` and `#flag`

Vira source files can directly specify the C header files and linker flags required by external libraries using top-level hash directives:

```v
#include <stdio.h>
#include <string.h>
#include <math.h>
#flag -lm
#flag -lsqlite3
```

- **`#include <header.h>` or `#include "header.h"`**: Emitted directly into the generated C99 translation unit before external declarations.
- **`#flag <args>`**: Forwards compiler and linker flags directly to the underlying C backend (GCC/Clang/MinGW-w64).

---

## 3. Declaring External Functions: `extern "C"`

External functions are declared inside an `extern "C" { ... }` block as bodiless function prototypes:

```v
extern "C" {
    fn puts(str &c_char) c_int
    fn strlen(str &c_char) c_size
    fn sqrt(x c_double) c_double
    fn abs(n c_int) c_int
}
```

External functions have no Vira function body; their definitions are resolved at link time or run time by the dynamic linker.

---

## 4. The Safety Boundary: `unsafe { ... }` & `@[unsafe]`

C functions can dereference invalid pointers, write out of bounds, or leak resources. To guarantee that Vira's memory-safety guarantees remain rock-solid, calling any `extern "C"` function or performing raw pointer dereferencing requires an **`unsafe` context**.

Attempting to call an `extern` function outside `unsafe` triggers a compile-time error:

```v
// ❌ COMPILE ERROR: call to extern function 'puts' requires unsafe block or @[unsafe] function
fn bad_call() {
    puts("Hello world")
}

//  CORRECT: Explicit unsafe block marks the safety boundary
fn good_call() {
    unsafe {
        puts("Hello world")
    }
}
```

### Best Practice: Encapsulating Unsafe Code in Safe Wrappers

Idiomatic Vira code isolates `unsafe` blocks inside small, clean helper functions so caller code remains 100% safe:

```v
pub fn safe_c_strlen(text string) int {
    unsafe {
        len := strlen(text)
        return int(len)
    }
}
```

---

## 5. C-ABI Scalar Types and Raw Pointer Operations

Vira defines an explicit set of scalar primitives mapped directly to their standard C counterparts:

| Vira Type | C99 Equivalent | Description |
| :--- | :--- | :--- |
| `c_char` / `c_uchar` | `char` / `unsigned char` | 8-bit character |
| `c_short` / `c_ushort` | `short` / `unsigned short` | 16-bit integer |
| `c_int` / `c_uint` | `int` / `unsigned int` | Standard 32-bit integer |
| `c_long` / `c_ulong` | `long` / `unsigned long` | Platform-dependent integer (32-bit on Win64, 64-bit on POSIX) |
| `c_longlong` / `c_ulonglong` | `long long` / `unsigned long long` | 64-bit integer |
| `c_size` / `c_ssize` | `size_t` / `ssize_t` | Pointer-sized word length |
| `c_float` | `float` | 32-bit IEEE 754 floating point |
| `c_double` | `double` | 64-bit IEEE 754 floating point |
| `voidptr` | `void*` | Generic untyped pointer |

### Raw Pointers and Address-of Operators

Inside an `unsafe` block, you can take the address of variables and dereference raw pointers:

```v
unsafe {
    mut counter := 42
    ptr := &counter       // Pointer to counter (&T)
    val := *ptr           // Dereference raw pointer (*ptr)
    println("Value: ${val}")

    mut x := 100
    mptr := &mut x        // Mutable raw pointer (&mut T)
}
```

### Automatic Argument Marshaling

Vira automatically marshals high-level language constructs when passed to external C functions:
- Passing a Vira `string` to a `&c_char` parameter automatically extracts `string.data` as a NUL-terminated `const char*`.
- Passing a Vira `tensor` or `slice` automatically extracts `.data` pointer.

---

## 6. Struct Layouts: `@[c_layout]` and `@[opaque]`

When exchanging data structures with C libraries, field layout must match C compiler padding and alignment rules:

```v
@[c_layout]
pub struct NativeVec2 {
pub:
    x c_double
    y c_double
}
```

For handles allocated and managed internally by foreign C libraries (such as database connections or window handles), use `@[opaque]`:

```v
@[opaque]
struct Sqlite3Db {}

extern "C" {
    fn sqlite3_open(filename &c_char, db &mut &Sqlite3Db) c_int
    fn sqlite3_close(db &Sqlite3Db) c_int
}
```

---

## 7. Exporting Shared Libraries: `@[export]` and `vira build -shared`

To expose Vira functions to C, C++, Rust, or Python:
1. Tag public functions with `@[export]` or `@[export: "custom_c_symbol"]`.
2. Compile with `vira build -shared`.

```v
// math_exports.vir
module main

#include <math.h>

extern "C" {
    fn sqrt(x c_double) c_double
}

@[export: "vira_add_integers"]
pub fn vira_add_integers(a c_int, b c_int) c_int {
    return a + b
}

@[export: "vira_vector_norm"]
pub fn vira_vector_norm(x c_double, y c_double) c_double {
    unsafe {
        return sqrt(x * x + y * y)
    }
}
```

### Compiling to Shared Library

```powershell
# Windows -> DLL
vira build -shared -o build/vira_ffi_demo.dll math_exports.vir

# Linux -> Shared Object (.so)
vira build -shared -o build/libvira_ffi_demo.so math_exports.vir

# macOS -> Dynamic Library (.dylib)
vira build -shared -o build/libvira_ffi_demo.dylib math_exports.vir
```

The compiler automatically emits `__declspec(dllexport)` on Windows and `__attribute__((visibility("default")))` on GCC/Clang.

---

## 8. Python Interop via `ctypes`

Python's standard library `ctypes` module can load compiled Vira shared libraries with zero third-party dependencies:

```python
import ctypes

# 1. Load the shared library
lib = ctypes.CDLL("./build/vira_ffi_demo.dll")

# 2. Configure function signatures (argtypes and restype)
lib.vira_add_integers.argtypes = [ctypes.c_int, ctypes.c_int]
lib.vira_add_integers.restype = ctypes.c_int

lib.vira_vector_norm.argtypes = [ctypes.c_double, ctypes.c_double]
lib.vira_vector_norm.restype = ctypes.c_double

# 3. Call functions with native C types
sum_val = lib.vira_add_integers(15, 27)
print(f"15 + 27 = {sum_val}")  # 42

norm_val = lib.vira_vector_norm(3.0, 4.0)
print(f"Norm(3.0, 4.0) = {norm_val}")  # 5.0
```

### Passing C Structures from Python

You can define matching Python `ctypes.Structure` classes to pass compound data to Vira:

```python
class NativeVec2(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
    ]

pt = NativeVec2(12.0, 5.0)
norm = lib.vira_vector_norm(pt.x, pt.y)
print(f"Vector norm: {norm}")  # 13.0
```

---

## 9. Complete Runnable Example Walkthrough

The repository includes a complete runnable example demonstrating all aspects of native FFI:
- Vira program: [`docs/examples/40_native_c_and_python_ffi.vir`](../examples/40_native_c_and_python_ffi.vir)
- Companion Python caller: [`docs/examples/40_python_ffi_caller.py`](../examples/40_python_ffi_caller.py)

### 1. Execute the Vira Program Directly

```powershell
.\vira.exe run docs/examples/40_native_c_and_python_ffi.vir
```

Expected terminal output:
```text
=== VIRA v0.7 NATIVE C & PYTHON FFI RUNTIME DEMO ===
[C libc puts] Direct invocation of puts() from C runtime
[C libc strlen] Measured length: 26
[C libc sqrt] sqrt(144.0): 12.00
[C libc abs] abs(-42): 42
[Safe Wrapper] String length: 19
[Safe Wrapper] Square root: 9.00
[Exported vira_add_integers] 25 + 17 = 42
[Exported vira_vector_norm] Norm of (3.0, 4.0) = 5.00
[Exported vira_string_length] Length = 21
[Raw Pointers] &counter dereferenced (*ptr): 99
=== FFI DEMO COMPLETED SUCCESSFULLY ===
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle, Multi-Threaded)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=9
```

### 2. Run the Companion Python Caller

```powershell
python docs/examples/40_python_ffi_caller.py
```

Expected output:
```text
=== Python FFI Caller for Vira Native Shared Library ===
[Python FFI] Loading shared library: C:\Users\hp\vex\build\vira_ffi_demo.dll
[Python FFI] vira_add_integers(15, 27) = 42
[Python FFI] vira_vector_norm(3.0, 4.0) = 5.0
[Python FFI] vira_vector_norm(6.0, 8.0) = 10.0
[Python FFI] NativeVec2(12.0, 5.0) norm = 13.0
=== All Python ctypes FFI assertions PASSED successfully! ===
```

---

## 10. Summary & Production Guidelines

- **Always isolate `unsafe` code**: Never expose `unsafe` calls directly to client application logic. Encapsulate them inside safe, idiomatic Vira functions.
- **Match types exactly**: Use `c_int`, `c_double`, and `c_size` when interfacing with C-ABI signatures rather than assuming fixed bit widths.
- **Enforce memory lifetimes**: Data passed across FFI boundaries is not garbage collected. If C allocates memory, ensure corresponding C deallocation routines (such as `free()`) are wrapped cleanly.
- **Zero-leak guarantee**: Vira's compiler tracks temporary allocations deterministically (`active_leaks=0`).

