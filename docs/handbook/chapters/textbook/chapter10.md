# Chapter 10: Generics & Closures

Generic programming and first-class functions allow you to write reusable containers and algorithmic combinators once while preserving full static type safety and native C99 execution speed. In Vira, generics are compiled via **zero-cost Ahead-of-Time (AOT) monomorphization**, and closures capture outer variables **by immutable value** with atomic reference counting so they can be safely stored in collections, returned from factories, and even sent across threads to actors.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch10_generics_and_closures.vir`](../code/ch10_generics_and_closures.vir)

---

## 10.1. Generic Functions (`fn name[T, U](...)`) and Type Argument Inference

A generic function declares one or more type parameters inside square brackets `[T, U]` immediately after the function name:

```v
fn first[T](xs []T) T {
	return xs[0]
}

fn map_slice[T, U](xs []T, f fn (T) U) []U {
	mut out := []U{}
	for x in xs {
		out.push(f(x))
	}
	return out
}
```

At call sites, Vira infers type arguments directly from the argument expressions whenever possible, or lets you spell them explicitly with `[Type]`:

```v
a := first([7, 8, 9])          // infers T = int
b := first(['alpha', 'beta'])  // infers T = string
c := first[int]([42])          // explicit type argument
```

Type parameters can appear in parameter types, container types (`[]T`, `map[K]int`), function signatures (`fn (T) U`), and return types. In `count_by`, for example, `K` is inferred from the return type of the `key` callback while `T` is inferred from the input slice `xs`:

```v
fn count_by[K, T](xs []T, key fn (T) K) map[K]int {
	mut counts := map[K]int{}
	for x in xs {
		counts[key(x)] += 1
	}
	return counts
}
```

> **Duck-Typed Instantiation:** Vira does not use separate generic trait-bound syntax on `[T]`. Instead, operators (`a > b`) and indexing inside a generic body are type-checked against each concrete instantiation (for example `max_of(3, 9)` and `max_of('pear', 'apple')`). If a concrete type does not support an operation used by the template, the type checker reports an error for that specific instantiation.

---

## 10.2. Generic Structs (`struct Box[T]`) and Generic Methods

Structs can also be parameterized over one or more types (`Box[T]`, `Pair[A, B]`). Receiver methods and associated static constructors reference the struct's type parameters directly:

```v
struct Box[T] {
pub mut:
	value T
}

fn Box[T].new(v T) Box[T] {
	return Box[T]{
		value: v
	}
}

fn (b Box[T]) get() T {
	return b.value
}

fn (mut b Box[T]) set(v T) {
	b.value = v
}
```

Unlike generic functions, generic struct literals, static constructors, and field annotations always specify their concrete type arguments:

```v
mut num_box := Box[int].new(41)
num_box.set(num_box.get() + 1)

str_box := Box[string]{ value: 'monomorphized' }
nested := Box[[]string].new(['x', 'y'])
```

Notice `Box[[]string]`: type arguments can be composite types such as slices, maps, or other generic structs, and Autofree automatically generates destructors for any concrete struct specialization whose fields own heap memory.

---

## 10.3. How Zero-Cost Monomorphization Works

Vira pays **zero runtime boxing or virtual dispatch overhead** for generics. During semantic analysis:

1. **Template Registration:** Generic structs (`Box[T]`) and generic functions (`first[T]`, `map_slice[T, U]`) are stored as AST templates and are never emitted directly to C.
2. **On-Demand Specialization:** Each unique combination of concrete type arguments triggers a fresh AST instantiation with a mangled C-safe identifier (such as `main__Box__int`, `main__Box__string`, `main__Box__sl_string`, or `main__Pair__string__int`), deduplicated so identical specializations are generated only once.
3. **Full Pipeline Integration:** The synthesized concrete declarations replace the generic templates before the Autofree escape-analysis pass and C99 code generator run.

Because Autofree and C99 codegen only ever see concrete declarations, `Box[int]` becomes a plain 4-byte `int32_t` struct in C with no heap wrapper, while `Box[[]string]` receives a tailored `main__Box__sl_string_autofree` destructor that releases its inner string slice deterministically.

---

## 10.4. Anonymous Functions, First-Class Function Types (`fn(A) R`), and Value-Captured Closures

Functions in Vira are first-class values with type syntax `fn (ArgTypes...) ReturnType` (or `fn () ReturnType` for zero-argument functions). Both named functions and inline anonymous `fn (...) ... { ... }` expressions share this uniform representation:

```v
fn apply(f fn (int) int, x int) int {
	return f(x)
}

fn double(x int) int {
	return x * 2
}

fn make_adder(n int) fn (int) int {
	return fn (x int) int {
		return x + n
	}
}
```

### Value Capture & Atomic Reference Counting

When an anonymous function references variables from an enclosing scope (such as `n` in `make_adder` or `tag` in `describe_service`), Vira enforces three strict safety rules:

1. **Read-Only Value Capture:** Outer variables are captured **by value** (cloned if they own heap memory) and are **read-only** inside the closure body. Assigning to a captured outer variable is a compile-time error, preventing hidden shared mutable state.
2. **Actor Ownership Isolation:** A closure cannot capture an `actor` handle because actors enforce the single-owner rule; pass actors explicitly as arguments instead.
3. **Shared Immutable Environment (`vira_closure`):** At runtime, a closure is a 16-byte C struct `{ void *fn; vira_closure_env *env; }`. Zero-capture closures and named-function references (`apply(double, 21)`) set `env = NULL` and allocate nothing. Capturing closures allocate a `vira_closure_env` holding the captured values plus an atomic reference count (`int32_t refs`) and a generated `_drop` destructor. Copying a closure (`copy_summarizer := summarizer`) retains the environment (`vira_closure_retain`), and Autofree releases each copy (`vira_closure_release`), freeing the captured values and environment deterministically when the last copy leaves scope—even across actor threads.

Closures can be stored directly inside slices (`[]fn (int) int`), maps (`map[int]fn (int) string`), and struct fields (`Button{ on_click: ... }`):

```v
struct Button {
pub:
	label string
	on_click fn (int) string
}

mut pipeline := []fn (int) int{}
pipeline.push(double)
pipeline.push(add10)
pipeline.push(fn (x int) int { return x - 1 })
```

---

## 10.5. Running the Complete Chapter 10 Example

Run [`docs/handbook/chapters/code/ch10_generics_and_closures.vir`](../code/ch10_generics_and_closures.vir):

```bash
./vira run docs/handbook/chapters/code/ch10_generics_and_closures.vir
```

Output:
```text
[1] first: 7 alpha explicit=42
    max_of: int=9, str=pear, f64=2.50
[2] Box[int]=42, Box[string]=monomorphized, Pair=port:8080, nested_len=2
[3] map=#1..#5, evens=2, sum=15, counts(a=2,b=2,c=1)
[4] closures: triple(5)=15, apply(double,21)=42, add10(5)=15, compose(4)=10
[5] pipeline=11, button=Save(clicked 2x), captured=infer:2
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=42
```

---

## Summary & Exercises

1. **Inspect Monomorphized C99:** Run `./vira cgen docs/handbook/chapters/code/ch10_generics_and_closures.vir | grep "Box__"` to verify that `Box[int]`, `Box[string]`, and `Box[[]string]` each compile to distinct C structs with zero runtime type tags.
2. **Generic `zip_pairs`:** Write a generic function `zip_pairs[A, B](xs []A, ys []B) []Pair[A, B]` that pairs elements up to the shorter slice length, and test it in `main()`.
3. **Immutable Capture Check:** Try mutating `factor` inside the `triple` closure in `ch10_generics_and_closures.vir` (`factor += 1`) and run `./vira check` to inspect Vira's value-capture diagnostic.
