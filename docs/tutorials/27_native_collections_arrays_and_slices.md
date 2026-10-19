# Tutorial 27: Native Collections — Fixed Arrays (`[N]T`) & Dynamic Slices (`[]T`)

> **Difficulty:** Intermediate  
> **Matching Code Example:** [`docs/examples/28_native_collections.vir`](../examples/28_native_collections.vir)

---

## 1. Overview & Mechanical Sympathy

Modern systems programming requires data structures that strike an optimal balance between **cache locality**, **zero unnecessary allocations**, and **memory safety**. In Vira, collections are built with high mechanical sympathy for hardware:

1. **Fixed-Size Stack Arrays (`[N]T`)**:
   - Allocated entirely on the stack (zero heap allocation, zero pointer chasing).
   - Wrapped in C99 value structs (`typedef struct { T data[N]; } vira_array_N_T;`), making them assignable, copyable, and returnable from functions by value.
   - Ideal for mathematical vectors, fixed buffers, coordinates, and fixed-sized state.

2. **Growable Dynamic Slices (`[]T`)**:
   - Represents a contiguous, dynamically allocated backing array.
   - Internal representation:
     ```c
     typedef struct {
         T*      data;     // contiguous element pointer
         int32_t len;      // current active elements count
         int32_t cap;      // allocated capacity
         bool    is_arena; // true if memory is managed by bump arena
     } vira_slice_T;
     ```
   - Automatically doubles backing capacity ($2\times$ geometric expansion) on demand.
   - Integrated with Vira's **Autofree** engine: automatically frees heap buffers (and cleans up individual heap elements like strings) when exiting lexical scope, ensuring `active_leaks = 0`.

---

## 2. Fixed-Size Stack Arrays (`[N]T`)

### Declaration & Initialization

Declare fixed arrays with explicit lengths:

```v
// Explicit length and type
mut scores := [4]int{95, 88, 72, 91}

// Zero-initialized array
zeros := [8]int{0}
```

### Access & Mutation

Elements are accessed via bracket indexing:

```v
first := scores[0]     // Reads 95
scores[2] = 85         // Mutates the 3rd element
```

All index operations (`scores[i]`) undergo hardware-efficient bounds checking at runtime. Accessing an index outside `[0, scores.len - 1]` triggers a clean panic detailing the file, line, index, and length:
```
out.c:12: vira panic: index out of bounds: index 10 with len 4
```

### Array Properties & Methods

- `arr.len`: Returns the fixed capacity $N$ (available as both property `arr.len` and method `arr.len()`).
- `arr.is_empty()`: Returns `true` if $N == 0$, otherwise `false`.
- `arr.contains(val)`: Returns `bool` indicating whether the element exists.
- `arr.get(index)`: Returns a Result `!T`, allowing safe retrieval without risking runtime panics.

```v
// Safe indexing with Result unwrap
val := scores.get(0) or { -1 }

// Out-of-bounds safe fallback
fallback := scores.get(100) or { 0 } // Evaluates or-block and binds 0
```

---

## 3. Dynamic Growable Slices (`[]T`)

### Declaration & Literals

Dynamic slices are declared with inferred literals, typed literals, or empty constructors:

```v
// Inferred slice ([]int)
mut items := [10, 20, 30]

// Explicitly typed slice
mut names := []string{"Alice", "Bob"}

// Empty slice with zero initial allocations
mut buffer := []int{}
```

### Pushing and Popping Elements

Slices grow dynamically with amortized $O(1)$ geometric expansion:

```v
mut nums := [1, 2, 3]

// Append elements
nums.push(40)
nums.push(50)
println("Length: ${nums.len}, Capacity: ${nums.cap}") // len: 5, cap: 6

// Pop elements from the end
last := nums.pop()
println("Popped: ${last}") // 50
```

### Modifying and Duplicating Slices

```v
// Mutation by index
nums[0] = 999

// In-place update method
nums.set(1, 888)

// Deep cloning (allocates a fresh backing buffer)
cloned := nums.clone()

// Clear all elements
nums.clear()
println("Is empty: ${nums.is_empty()}") // true
```

---

## 4. Safe Retrieval with the Result Monad (`!T`)

In production environments, panicking on unexpected user input or transient indexes is unacceptable. Both arrays and slices provide `.get(index)`, which returns a Result `!T`:

```v
fn find_user(users []string, idx int) string {
    return users.get(idx) or {
        println("Index ${idx} out of range; defaulting to guest.")
        return "Guest"
    }
}
```

Because `.get()` returns `!T`, it supports:
1. Fallback expressions: `val := list.get(i) or { default_val }`
2. Block handling:
   ```v
   val := list.get(i) or {
       eprintln("Lookup failed: ${err}")
       return
   }
   ```
3. Error propagation with `!`:
   ```v
   val := list.get(i)!
   ```

---

## 5. Clean Iteration with `for ... in`

Iterate cleanly through any fixed array, slice, or string without managing index counters:

```v
numbers := [10, 20, 30, 40]

mut sum := 0
for n in numbers {
    sum = sum + n
}
println("Sum = ${sum}")
```

For string slices, loop variables borrow elements without extraneous string copies:

```v
hosts := ["127.0.0.1", "0.0.0.0", "192.168.1.1"]

for host in hosts {
    println("Binding to: ${host}")
}
```

---

## 6. Deterministic Autofree & Zero Leaks

Vira's static escape analyzer manages collection memory deterministically:
1. **Scope-based Lifetimes**: When a slice variable leaves its declaring block, an automatic cleanup statement is injected (`vira_slice_int_free(&items)`).
2. **Deep Deallocation for String Slices**: When `[]string` is freed, individual string buffers are freed before releasing the slice buffer.
3. **No Double-Frees**: When elements are moved or reassigned, previous buffers are released cleanly.

Every program verifies memory cleanup upon exit:
```
[vira_rt] active_leaks=0 | total_freed=30
```

---

## 7. Method Reference Summary

| Method / Property | Receiver | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `.len` | `[N]T`, `[]T` | `int` | Current number of elements |
| `.cap` | `[]T` | `int` | Total capacity of allocated backing buffer |
| `.push(val)` | `mut []T` | `void` | Appends element, growing capacity if needed |
| `.pop()` | `mut []T` | `T` | Removes and returns the last element |
| `.get(index)` | `[N]T`, `[]T` | `!T` | Safe access returning Result Monad |
| `.set(index, val)` | `mut []T` | `void` | Overwrites element at index with bounds check |
| `.contains(val)` | `[N]T`, `[]T` | `bool` | Reports whether element is present |
| `.is_empty()` | `[N]T`, `[]T` | `bool` | Reports whether length is 0 |
| `.clone()` | `[]T` | `[]T` | Allocates a fresh deep copy of the slice |
| `.clear()` | `mut []T` | `void` | Resets length to 0 (cleans elements) |

---

## 8. Runnable Hands-On Example

Execute the complete tutorial code example directly:

```powershell
.\vira.exe run docs/examples/28_native_collections.vir
```
