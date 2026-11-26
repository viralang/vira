# Appendix A: Keywords, Operators & Tokens Reference

This appendix documents Vira's complete lexical grammar, keywords, attributes, preprocessor directives, literal forms, operator precedence, type compatibility rules, and newline statement-termination rules as implemented in the Stage-0 compiler (`internal/token/token.go`, `internal/lexer/lexer.go`, `internal/parser/parser.go`, and `internal/checker/checker.go`).

---

## A.1. Reserved Keywords

The Vira lexer (`internal/token/token.go`) recognizes **37 reserved keywords** via `LookupIdent`. None of these may be used as ordinary variable, function, struct, or field identifiers.

### Active Language Keywords (32)

| Category | Keyword | Purpose |
| :--- | :--- | :--- |
| **Modules & Visibility** | `module` | Declares the file's module namespace (defaults to `main` if omitted). |
| | `import` | Imports a local, stdlib, or external package (`import net`, `import net as n`, `import (...)`). |
| | `as` | Binds an import alias (`import strings as s`). |
| | `pub` | Exports top-level declarations (`pub fn`, `pub struct`, `pub enum`, `pub interface`, `pub const`) or marks struct access sections (`pub:`, `pub mut:`). |
| **Bindings & Mutability** | `mut` | Declares mutable bindings (`mut x := 1`), mutable parameters/receivers, mutable call-site arguments, mutable pointers (`&mut T`), and `mut:` sections. |
| | `const` | Declares compile-time / module constants (`const MAX = 64`, `pub const (...)`). |
| **Types & Declarations** | `struct` | Declares a value struct (or `@[opaque]` / `@[c_layout]` C struct). |
| | `interface` | Declares a structural interface satisfied implicitly by matching fields/methods. |
| | `enum` | Declares a 32-bit integer enumeration (`enum Color { red green = 4 blue }`). |
| | `fn` | Declares a function, method, static type method, function type `fn (A) R`, or anonymous closure. |
| | `actor` | Declares a heap-allocated isolated actor with a lock-free mailbox. |
| | `extern` | Declares C-ABI external prototypes inside `extern "C" { ... }`. |
| **Control Flow & Cleanup** | `if` | Conditional branch (`if cond { ... }`, `cond` must be `bool`). |
| | `else` | Fallback branch (`else { ... }` or `else if cond { ... }`). |
| | `switch` | Multi-branch conditional (`switch tag { case ...: ... default: ... }` or tagless `switch { case cond: ... }`). |
| | `case` | Branch arm inside a `switch` statement (supports comma-separated values). |
| | `default` | Fallback arm inside a `switch` statement (at most one per `switch`). |
| | `for` | Unified loop (`for { ... }`, `for cond { ... }`, `for i := 0; i < n; i += 1 { ... }`, `for x in xs { ... }`, `for k, v in m { ... }`). |
| | `in` | Iteration separator in `for ... in` loops. |
| | `break` | Exits the innermost `for` loop (automatically unwinding active `or` error temporaries in that loop). |
| | `continue` | Advances to the next iteration of the innermost `for` loop. |
| | `return` | Returns from a function (auto-wraps `T` or `error(...)` when returning `!T` or `?T`). |
| | `defer` | Registers a statement or `{ ... }` block to execute in LIFO order on lexical scope exit or actor fault unwind. |
| | `or` | Unwraps a `!T` Result or `?T` Option at a call site (`call() or { ... }`), binding implicit `err string`. |
| **Memory & Safety** | `arena` | Opens a 64-byte aligned bump-pointer allocation scope (`arena { ... }`). |
| | `unsafe` | Opens an unsafe block (`unsafe { ... }`) required for C FFI calls, pointer dereferencing, and `@[unsafe]` functions. |
| **Literals & Built-in Types** | `true` | Boolean literal (`bool`). |
| | `false` | Boolean literal (`bool`). |
| | `f32` | 32-bit IEEE 754 float type (also usable as cast `f32(x)`). |
| | `f16` | 16-bit half-precision float token/suffix (`0.5f16`). |
| | `bf16` | 16-bit Brain Float type (`bf16`) and literal suffix (`2.0bf16`). |
| | `tensor` | Multi-dimensional SIMD/GPU tensor type constructor (`tensor[f32]`). |

### Lexer-Reserved Keywords (5)

These 5 identifiers are reserved as keyword tokens in `internal/token/token.go` for upcoming or internal syntax (and therefore cannot be used as user identifiers):

| Keyword | Token | Status |
| :--- | :--- | :--- |
| `match` | `TokMatch` | Reserved for upcoming `v0.9.x` exhaustive pattern matching expressions. |
| `type` | `TokType` | Reserved in the lexer for future type alias declarations. |
| `is` | `TokIs` | Reserved in the lexer (precedence level 3). |
| `spawn` | `TokSpawn` | Reserved in the lexer. |
| `supervise` | `TokSupervise` | Reserved in the lexer. |

---

## A.2. Built-in Type Names & C99 Reserved Identifier Guard

### Built-in & Contextual Type Identifiers

Unlike `f32`, `f16`, `bf16`, and `tensor`, the following type names and constructors are resolved contextually by the type checker (`internal/checker/checker.go`) rather than being lexer keywords:

- **Vira Primitives:** `int` (`int32_t`), `i64` (`int64_t`), `u8` / `byte` (`uint8_t`), `f64` (`double`), `bool`, `string`, `void`
- **Built-in Generic Containers & Concurrency Types:** `map[K]V`, `Future[T]`, `Ref[T]`
- **C-ABI FFI Primitives:** `c_char`, `c_uchar`, `c_short`, `c_ushort`, `c_int`, `c_uint`, `c_long`, `c_ulong`, `c_longlong`, `c_ulonglong`, `c_size`, `c_ssize`, `c_float`, `c_double`, `voidptr`
- **Special Built-in Constructor / Placeholder:** `error(...)`, `_` (blank discard identifier; cannot be read as a value)

### C99 Reserved Identifier Check

Because Vira emits local variables, parameters, loop variables, receivers, and struct fields directly into C99, the checker rejects C99 keywords and `_Bool` / `_Alignas`-style names when used as Vira identifiers (`cReserved` in `internal/checker/checker.go`):

`auto`, `char`, `double`, `enum`, `extern`, `float`, `goto`, `inline`, `int`, `long`, `register`, `restrict`, `short`, `signed`, `sizeof`, `static`, `typedef`, `union`, `unsigned`, `void`, `volatile`, `bool`, `_Bool`, `_Complex`, `_Imaginary`, `_Alignas`, `_Alignof`, `_Atomic`, `_Generic`, `_Noreturn`, `_Static_assert`, `_Thread_local`.

---

## A.3. Attributes & Preprocessor Directives

### Declaration Attributes (`@[...]`)

Attributes attach metadata to the immediately following `fn` or `struct` declaration (`internal/parser/parser.go`). Multiple attributes may be stacked or comma-separated inside `@[ ... ]`.

| Attribute | Target | Effect |
| :--- | :--- | :--- |
| `@[export]` | `fn` | Exports the function with C linkage (`VIRA_EXPORT` — `__declspec(dllexport)` on Windows, `__attribute__((visibility("default")))` on POSIX) using its Vira name. |
| `@[export: "c_name"]` | `fn` | Exports the function under the explicit C symbol `c_name` (accepts `:` or `=`). |
| `@[c_layout]` | `struct` | Marks the struct as following standard C-ABI alignment and field layout for FFI. |
| `@[opaque]` | `struct` | Declares an opaque foreign C struct; emits no C struct body or destructor (used via `&T` / `&mut T`). |
| `@[unsafe]` | `fn` | Marks a Vira function as unsafe; callers must be inside an `unsafe { ... }` block or another `@[unsafe]` function. |
| `@[intrinsic: "c_fn"]` | `fn` | Internal compiler/stdlib attribute binding a bodiless Vira prototype to a runtime C helper `c_fn` (accepts `:` or `=`). |

### Preprocessor Directives (`#...`)

Any top-level line starting with `#` is scanned as a single `TokDirective` token up to `\r` or `\n`:

| Directive | Example | Effect |
| :--- | :--- | :--- |
| `#include` | `#include <sqlite3.h>` or `#include "mylib.h"` | Emitted verbatim near the top of the generated C99 file, after `vira_rt.h`. |
| `#flag` | `#flag -lsqlite3` or `#flag -O2 -lm` | Appended to the C compiler/linker command line when building the translation unit. |

---

## A.4. Comments & Literal Forms

### Comments

- **Single-line:** `// comment until end of line`
- **Multi-line:** `/* comment spanning one or more lines */` (non-nesting)

### String Literals & Interpolation

- **Delimiters:** Single quotes `'...'` or double quotes `"..."` (no semantic difference).
- **Escape Sequences (`unescapeString`):** `\n` (newline), `\r` (carriage return), `\t` (tab), `\\` (backslash), `\'` (single quote), `\"` (double quote), `\0` (NUL byte).
- **Interpolation (`${expr}`):** Any valid Vira expression may appear inside `${...}` (supports nested `{...}` blocks up to depth 8). Each interpolated expression is evaluated **once**, formatted by type (`int`/`u8` → `%d`, `i64`/`c_long`/`c_size` → `%lld`, `f32`/`f64`/`bf16` → `%.2f`, `bool` → `true`/`false`, `enum` → variant name string, pointers → `%p`, `string` → length-preserving concatenation so embedded NUL bytes do not truncate).

### Integer Literals (`TokInt`)

| Base | Prefix | Digit Separator | Suffix | Example | Inferred Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Decimal | *(none)* | `_` | optional `i64` | `42`, `1_000_000`, `100i64` | `int` (or `i64` with `i64` suffix) |
| Hexadecimal | `0x` / `0X` | `_` | optional `i64` | `0xFF`, `0x1E_1E_2E`, `0x10i64` | `int` (or `i64`) |
| Octal | `0o` / `0O` | `_` | optional `i64` | `0o755`, `0o010_000` | `int` (or `i64`) |
| Binary | `0b` / `0B` | `_` | optional `i64` | `0b1010_0101`, `0b11i64` | `int` (or `i64`) |

### Floating-Point Literals (`TokFloat`)

- **Decimal Floats:** `3.14`, `1_000.5` (a `.` followed by another `.` is **not** consumed as a decimal point, reserving `..` for range expressions).
- **Scientific Exponents (`T-087`):** `e` or `E`, optional `+` or `-`, followed by at least one decimal digit (e.g., `1e5`, `1.0e31`, `2.5E-3`, `1e-12`, `3e2f32`). Requiring a digit after `e`/`E` ensures member accesses like `obj.e` are never mislexed as exponents.
- **Precision Suffixes:** `f64` (default when unsuffixed), `f32`, `f16`, `bf16` (e.g., `1.0f32`, `0.5f16`, `2.0bf16`).

### Composite & Function Literals

- **Fixed Array Literal:** `[4]int{1, 2, 3, 4}` (stack-allocated `[N]T`)
- **Slice Literal:** `[]string{'a', 'b'}`, `[]u8{}` (explicit empty slice), or inferred `[1, 2, 3]` (empty `[]` without a type defaults to `[]int`)
- **Map Literal:** `map[string]int{ 'a': 1, 'b': 2 }`
- **Struct / Actor Literal:** `User{ name: 'Ada', age: 30 }` (requires PascalCase type name; suppressed inside `if`, `switch`, and `for` header expressions so `{` always begins the block body)
- **Null Actor Reference:** `Ref[Worker]{}` (takes no fields; sends fail with `"actor is gone"`)
- **Anonymous Function / Closure:** `fn (x int) int { return x + k }` (captures outer variables by value, read-only; parameters cannot be `mut`)

---

## A.5. Operator Precedence, Associativity & Type Compatibility

The Pratt parser (`internal/token/token.go` and `internal/parser/parser.go`) assigns **8 expression precedence levels** (higher binds tighter), plus statement-level binding/assignment operators. All binary operators are left-associative.

### Precedence & Compatibility Table

| Prec | Operator(s) | Description | Allowed Operand Types | Result Type |
| :---: | :--- | :--- | :--- | :--- |
| **Stmt** | `:=` | Variable declaration (`x := e`, `mut x := e`) | Any non-void expression | *(statement)* |
| **Stmt** | `=` | Assignment (`x = e`) | Matching mutable target & value types | *(statement)* |
| **Stmt** | `+=`, `-=`, `*=`, `/=`, `%=` | Compound assignment | Same as corresponding binary `+`, `-`, `*`, `/`, `%` | *(statement)* |
| **8** | `f(args)` | Function / method / closure call | Function, method, static method, or `fn(...)` value | Return type |
| **8** | `x[i]` | Indexing | `[]T` / `[N]T` / `string` (`int`/`i64` index), `map[K]V` (`K` index) | `T`, `string` (1-byte view), `V` |
| **8** | `x.field` | Field, method, enum variant, or module selector | Struct, actor (immutable fields only from outside), `string.len`, `[]T.len`/`.cap`, `map.len`, `tensor.rows`/`.cols`/`.ndim`, `Enum.variant` | Member type |
| **8** | `a <- m(...)` | Actor message send (`actor` or `Ref[T]`) | Actor handle or `Ref[T]` | `Future[T]` (or `T` with `or`/`!`) |
| **8** | `!`, `?` | Result / Option propagation suffix | Call returning `!T` or `?T` inside matching fn | `T` |
| **8** | `or { ... }` | Result / Option fallback unwrap block | Call returning `!T` or `?T` | `T` |
| **7 (Prefix)** | `!x` | Logical NOT | `bool` | `bool` |
| **7 (Prefix)** | `-x`, `+x` | Unary negation / identity | `int`, `i64`, `f32`, `f64` | Same numeric type |
| **7 (Prefix)** | `&x`, `&mut x` | Address-of (raw pointer) | Any addressable place (`&mut` requires `mut` root) | `&T` / `&mut T` |
| **7 (Prefix)** | `*ptr` | Pointer dereference (requires `unsafe`) | `&T` / `&mut T` | `T` |
| **7** | `*`, `/` | Multiplication, division | Matching numeric types (integer `/` checks `/ 0` at runtime) | Same numeric type |
| **7** | `%` | Remainder / modulo | Matching integer types (checks `% 0` at runtime) | Same integer type |
| **7** | `@` | Matrix multiplication | `tensor[T] @ tensor[T]` (same `DType`) | `tensor[T]` |
| **6** | `+` | Addition or string concatenation | Numeric types, `string` (1-alloc chain), pointers/`voidptr` | Same type |
| **6** | `-` | Subtraction | Numeric types, pointers/`voidptr` | Same type |
| **5** | `<<`, `>>` | Bitwise left / right shift | Matching integer types | Same integer type |
| **4** | `&`, `\|`, `^` | Bitwise AND, OR, XOR | Matching integer types or `bool` | Same type |
| **3** | `==`, `!=` | Equality / inequality | Primitives (incl. `string` content eq), pointers/`&T`, `enum` | `bool` |
| **3** | `<`, `<=`, `>`, `>=` | Relational ordering | Numeric types, `string` (lexicographical), pointers/`&T` | `bool` |
| **2** | `&&` | Short-circuit logical AND | `bool` | `bool` |
| **1** | `\|\|` | Short-circuit logical OR | `bool` | `bool` |

### Type Compatibility & Arithmetic Semantics

- **No Implicit `string` + Number:** Both operands of `+` must have matching types (`string + string` is valid; `string + int` is a compile error—use `'${x}'`).
- **Primitive Numeric Promotion in `Equals` vs. `ExactEquals`:** Integer-like primitives (`u8`, `int`, `i64`, and C integer types) are mutually compatible in `Equals`; float-like primitives (`f32`, `f64`, `bf16`, `c_float`, `c_double`) are mutually compatible in `Equals`. Container element types (`[]T`, `[N]T`, `tensor[T]`, `map[K]V`, `fn(...)`, `Future[T]`, `&T`) and `mut` parameters use **strict `ExactEquals`** (no implicit widening). If either side of a binary arithmetic/bitwise op is `i64`, the result type promotes to `i64`.
- **Two's-Complement Wrapping & Checked Division:** Vira builds C99 with `-fwrapv`, so signed integer overflow wraps deterministically. Integer `/` and `%` emit calls to `vira_div_*` / `vira_mod_*`, turning division by zero into a supervised `_vira_rt_fault` (caught inside actors) and defining `INT_MIN / -1` without `SIGFPE`.
- **No `==` on Composites:** Structs, slices, maps, tensors, interfaces, closures, and actors do not support `==` or `!=` directly.
- **C FFI Pointer Coercions:** Inside calls to `extern "C"` functions, `string` auto-marshals to `&c_char`/`&u8`/`voidptr` (via `vira_str_ffi_cstr`), `[]T` and `tensor[T]` auto-marshal to `&T`/`voidptr` (`.data`), and `&T` / `voidptr` interoperate bidirectionally.

---

## A.6. Newline Statement-Termination Rule (`T-118`)

Vira does not require semicolons (`;`) at the end of statements (the only required `;` is the separator in a 3-part C-style `for init; cond; post { ... }` header).

To prevent an expression on one line from accidentally swallowing the start of a statement on the next line, the Pratt parser (`stopsExprAtNewline` in `internal/parser/parser.go`) **stops binary/postfix expression parsing across a newline** whenever the first token on the new line is one of these **5 tokens**:

| Token | Symbol | Why It Stops Across a Newline |
| :--- | :--- | :--- |
| `TokAt` | `@` | Prevents `x` followed by `@[export]` on the next line from parsing as `x @ [...]`. |
| `TokPlus` | `+` | Prevents a trailing expression from consuming a unary `+` on the next line. |
| `TokMinus` | `-` | Prevents `a\n-b` from parsing as `a - b`. |
| `TokLBracket` | `[` | Prevents `a\n[1, 2]` from parsing as indexing `a[...]`. |
| `TokLParen` | `(` | Prevents `a\n(b)` from parsing as a call `a(b)`. |

Additional newline rules in `internal/parser/parser.go`:
- **`return`:** A bare `return` followed by a newline parses no return value (`return\n x` returns void, not `x`).
- **Typed slice/array literals (`[]T{...}`, `[N]T{...}`):** The element type `T` must begin on the same line as `]`.
- **Function types (`fn (A) R`):** The optional return type `R` must begin on the same line as `)`.
- **Struct embedding:** Inside `struct Name { ... }`, an identifier followed immediately by a newline or `}` (or `[T]` not starting with `[]`/`[N]`) is parsed as an embedded struct rather than a field.
