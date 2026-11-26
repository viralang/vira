# Chapter 6: Strings, Arrays, Slices & Maps

Real programs spend most of their time working with text and structured collections. Vira gives you four built-in collection and text primitives—UTF-8 `string` values, stack-allocated fixed arrays `[N]T`, heap-backed dynamic slices `[]T`, and insertion-ordered hash maps `map[K]V`—all governed by the same single-owner, zero-leak Autofree model introduced in Chapter 5.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch06_strings_and_collections.vir`](../code/ch06_strings_and_collections.vir)

---

## 6.1. UTF-8 Strings (`string`), Interpolation (`${...}`), Concatenation (`+`, `+=`), and Zero-Copy Views

In Vira, a `string` is an immutable 16-byte header (`vira_string`) holding a UTF-8 byte pointer, byte length (`.len`), and ownership flags (`is_lit`, `is_borrowed`). String literals are stored in static read-only memory (`is_lit = true`), so creating or passing a literal allocates nothing on the heap.

### Concatenation (`+`, `+=`) and Interpolation (`${...}`)

- **Single-allocation `+` chains:** Writing `a + b + c + d` lowers to a single `vira_str_concat_n` call that computes the total byte length once and allocates one destination buffer—never $N-1$ temporary strings.
- **In-place `+=` updates:** `log += 'step-${i};'` builds the combined string, frees the previous owned buffer in `log`, and stores the new one. Self-appends (`s += s`) are safe.
- **String interpolation (`${expr}`):** Evaluates each embedded expression exactly once and formats integers, floats, booleans, enums, and strings into a single newly allocated `string`.
- **Content comparison:** `==`, `!=`, `<`, `<=`, `>`, and `>=` compare strings by UTF-8 byte content, never pointer address.

```v
service := 'vira'
region := 'eu-west'
// Chained `+` performs a single allocation for the entire expression.
endpoint := 'https://' + service + '.' + region + '.internal'
mut log := ''
for mut i := 1; i <= 3; i += 1 {
	log += 'step-${i};'
}
```

### Zero-Copy Views and Safe Storage

String slicing and trimming operations—such as `s.trim()`, `s.trim_left()`, `s.trim_right()`, and `s.slice(start, end)`—can return lightweight **borrowed views** (`is_borrowed = true`) pointing into the source buffer. Whenever you store a view into a variable, struct field, slice, or return value, the compiler ensures it is safely owned (`vira_str_own`) so reassigning or freeing the original string can never leave a dangling pointer:

```v
mut raw := '   payload:alpha:42   '
trimmed := raw.trim()
raw = 'overwritten'
parts := trimmed.split(':')
first_byte := trimmed.byte_at(0) or { u8(0) }
```

### Common Built-in `string` Methods

| Method / Property | Signature / Type | Description |
| :--- | :--- | :--- |
| `.len` | `int` | Byte length of the UTF-8 string. |
| `.trim()`, `.trim_left()`, `.trim_right()` | `string` | Strips leading/trailing ASCII whitespace. |
| `.slice(start, end)` | `string` | Substring over byte range `[start, end)`. |
| `.split(sep)` | `[]string` | Splits into a slice of strings by separator `sep`. |
| `.contains(sub)`, `.starts_with(p)`, `.ends_with(s)` | `bool` | Substring and prefix/suffix checks. |
| `.replace(old, new_str)` | `string` | Replaces occurrences of `old` with `new_str`. |
| `.to_upper()`, `.to_lower()` | `string` | ASCII case conversion. |
| `.byte_at(index)` | `!u8` | Bounds-checked byte lookup returning a Result `!u8`. |
| `.to_int()`, `.to_i64()`, `.to_f64()` | `!int` / `!i64` / `!f64` | Parses numeric text into a typed Result. |

---

## 6.2. Fixed-Size Stack Arrays (`[N]T`) and Value Semantics

When the element count is known at compile time, a fixed array `[N]T` lives directly on the stack with **zero heap allocation**:

```v
mut readings := [4]int{12, 18, 25, 30}
mut copy := readings // Stack value copy: independent from `readings`
copy[0] = 99

hit := readings.contains(25)
safe_ok := readings.get(2) or { -1 }
safe_oob := readings.get(9) or { -1 }
mut sum := 0
for val in readings {
	sum += val
}
```

Key characteristics of `[N]T`:
1. **Value Semantics:** Assigning `mut copy := readings` or passing `[N]T` by value copies the $N$ elements inline; mutating `copy[0]` leaves `readings[0]` untouched.
2. **Bounds Checking & Safe `.get(i)`:** Direct indexing (`readings[i]`) is bounds-checked at runtime and panics if out of range. For untrusted indices, `readings.get(i)` returns `!T` so you can handle out-of-bounds indices with `or { ... }`.
3. **Zero-Initialization Shorthand:** `zeros := [8]int{0}` initializes all 8 slots to `0`.
4. **Built-in Properties & Methods:** `.len`, `.is_empty()`, `.contains(val)`, and `.get(index)`.

---

## 6.3. Dynamic Growable Slices (`[]T`), `push`, `pop`, `clone`, and Safe Indexing (`.get(i) or { ... }`)

When the length changes at runtime, use a dynamic slice `[]T`. Under the hood, a slice stores `{ data*, len, cap, is_arena }` and grows geometrically ($2\times$ capacity) as elements are pushed:

```v
mut stages := ['ingest', 'parse']
stages.push('check')
stages.push('codegen')
stages.set(1, 'lex-parse')

mut cloned := stages.clone()
popped := stages.pop()
safe_item := stages.get(1) or { 'none' }
safe_miss := stages.get(99) or { 'missing' }
```

### Slice Literals & Mutability Rules

- **Inferred literal:** `mut nums := [1, 2, 3]` has type `[]int`.
- **Empty typed slice:** `mut buf := []string{}` starts empty with zero initial heap buffer until the first `.push(...)`.
- **Immutable by default:** Calling `.push()`, `.pop()`, `.set()`, `.clear()`, or assigning `s[i] = val` requires the slice binding to be declared with `mut`.

### Slice Method Reference

| Method / Property | Receiver | Return | Description |
| :--- | :--- | :--- | :--- |
| `.len`, `.cap` | `[]T` | `int` | Active element count and allocated capacity. |
| `.push(val)` | `mut []T` | `void` | Appends an element (moves `val` on last use, otherwise copies). |
| `.pop()` | `mut []T` | `T` | Removes and returns the last element (panics if empty). |
| `.get(i)` | `[]T` | `!T` | Bounds-safe lookup returning `!T` for `or { ... }` / `!`. |
| `.set(i, val)` | `mut []T` | `void` | Replaces element at `i` (frees previous heap element if owned). |
| `.clone()` | `[]T` | `[]T` | Allocates an independent deep copy of the slice and elements. |
| `.slice(start, end)` | `[]T` | `[]T` | Creates a new slice copying elements in `[start, end)`. |
| `.contains(val)` | `[]T` | `bool` | Checks whether `val` appears in the slice. |
| `.is_empty()` | `[]T` | `bool` | Returns `true` when `.len == 0`. |
| `.clear()` | `mut []T` | `void` | Releases owned elements and resets `.len` to `0`. |

---

## 6.4. Insertion-Ordered Hash Maps (`map[K]V`), Lookups, Mutation, and Key/Value Iteration

Vira's built-in `map[K]V` is an **insertion-ordered** hash map. Keys can be `string`, any integer type (`u8`, `int`, `i64`), `bool`, or an `enum`; values can be any copyable type (primitives, strings, structs, slices, closures, or nested maps—everything except single-owner actors).

```v
mut quotas := map[string]int{
	'gpu': 4
	'cpu': 16
}
quotas['npu'] = 2
quotas['gpu'] = 8
quotas['tpu'] += 1 // Missing key starts at zero value (0), then becomes 1

missing := quotas['fpga']
has_gpu := quotas.has('gpu')
removed := quotas.delete('cpu')
```

### Important Map Semantics

1. **Missing Keys Return the Zero Value:** Reading `quotas['fpga']` when `'fpga'` is absent returns `0` (`''` for `string`, `false` for `bool`, empty struct/slice for composites) rather than panicking. Use `.has(key)` whenever you need to distinguish an absent key from a stored zero value.
2. **Insertion-Ordered Iteration:** `for k, v in quotas` (or key-only `for k in quotas`) visits entries in the order they were inserted. `quotas.keys()` and `quotas.values()` likewise return `[]K` and `[]V` slices in insertion order. Do not insert into or delete from a map while iterating over it.
3. **Independent Value Copies:** `copy := quotas` creates a deep, independent copy of the map.
4. **Copy-Out / Store-Back for Nested Containers:** Because reading `m[k]` returns a copy (or a zero value if absent), Vira rejects mutating a container element in place (`tags_by_env['prod'].push('audited')`). Instead, copy the value into a `mut` variable, mutate it, and assign it back:

```v
mut tags_by_env := map[string][]string{}
tags_by_env['prod'] = ['fast']
mut prod_tags := tags_by_env['prod']
prod_tags.push('audited')
tags_by_env['prod'] = prod_tags
```

### Map Method & Operator Summary

| Operation | Meaning |
| :--- | :--- |
| `m[k]` | Reads value for `k`, or the zero value of `V` if missing. |
| `m[k] = v`, `m[k] += v` | Inserts or replaces entry (frees replaced heap value automatically). |
| `m.len` | Number of active entries. |
| `m.has(k)` | Returns `true` if `k` exists in the map. |
| `m.delete(k)` | Removes `k` and frees its key/value; returns `true` if `k` was present. |
| `m.keys()`, `m.values()` | Returns newly allocated `[]K` / `[]V` slices in insertion order. |
| `m.clear()` | Removes and frees all entries. |

---

## 6.5. Running the Complete Chapter 6 Example

Run [`docs/handbook/chapters/code/ch06_strings_and_collections.vir`](../code/ch06_strings_and_collections.vir):

```bash
./vira run docs/handbook/chapters/code/ch06_strings_and_collections.vir
```

Output:
```text
[1] Strings:
    endpoint=https://vira.eu-west.internal (len=29)
    log=step-1;step-2;step-3;
    view_safe="payload:alpha:42" (raw_now="overwritten"), parts=3, head=PAYLOAD, b0=112
[2] Fixed Arrays ([N]T):
    readings[0]=12, copy[0]=99, len=4, empty=false
    contains(25)=true, get(2)=25, get(9)=-1, sum=85
[3] Dynamic Slices ([]T):
    len=3, cap=4, popped=codegen, get(1)=lex-parse, get(99)=missing
    cloned_len=4, contains_codegen=true
    after_clear: empty=true, len=0
[4] Hash Maps (map[K]V):
    len=3, missing=0, has_gpu=true, deleted_cpu=true
    order=[gpu=8, npu=2, tpu=1], keys=3, val_sum=11
    prod_tags_len=2
    cleared_len=0
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=54
```

---

## Summary & Exercises

1. **Safe Byte Iteration:** Write a function `count_colons(s string) int` using `s.byte_at(i) or { u8(0) }` inside a `for mut i := 0; i < s.len; i += 1` loop and test it in `ch06_strings_and_collections.vir`.
2. **Word Frequency Counter:** Write `word_counts(text string) map[string]int` using `text.split(' ')` and `counts[w] += 1`, then print every word and count in insertion order.
3. **Immutable Slice Guard:** Remove `mut` from `stages` in `demo_slices()` and run `./vira check` to inspect the compile-time diagnostic preventing accidental mutation of immutable slices.
