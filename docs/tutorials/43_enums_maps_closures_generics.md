# Tutorial 43: Enums, Maps, Closures & Generics

> **Runnable example:** [`48_enums_maps_closures_generics.vir`](../examples/48_enums_maps_closures_generics.vir)
> **Language reference:** [Language Guide §14](../LANGUAGE_GUIDE.md#14-enums-maps-closures--generics-v09)

Vira v0.9 adds four features that work together: **enums** name a fixed set of values, **maps** store values by key, **closures** pass behaviour around as values, and **generics** let one function or struct serve many types. All four follow the rules you already know: values are copied unless moved, autofree releases what they own, and nothing leaks (`active_leaks=0`).

| Feature | Syntax | C lowering |
| :-- | :-- | :-- |
| Enum | `enum Color { red green }`, `Color.red` | `int32_t` plus a name lookup for interpolation |
| Map | `map[string]int{'a': 1}`, `m[k]`, `for k, v in m` | one insertion-ordered hash map in the runtime, with a per-value-type ops table |
| Closure | `fn (x int) int { return x + n }`, type `fn (int) int` | function pointer + reference-counted environment of captured values |
| Generics | `fn first[T](xs []T) T`, `struct Box[T] { ... }` | monomorphisation: one concrete copy per type (`first__int`) |

---

## 1. Enums

```vira
enum Category {
	fruit
	dairy
	bakery
}

pub enum Status {
	ok = 200
	not_found = 404
	later          // 405
}
```

- Variants count up from 0, or from the last explicit value.
- Write a variant as `Category.fruit`, or `shapes.Kind.square` from another module (the enum must be `pub`).
- Compare with `==` and `!=`, `switch` on them, store them in structs, slices and maps.
- `int(c)` gives the number; `'${c}'` prints the variant name.

```vira
fn label(c Category) string {
	switch c {
		case Category.fruit:
			return 'fresh fruit'
		case Category.dairy:
			return 'dairy'
		default:
			return 'baked goods'
	}
}
```

The checker rejects unknown variants (`enum 'Color' has no variant 'purple'`), duplicate variants, values that do not fit in 32 bits, ordering operators on enums, and comparing an enum with an integer.

---

## 2. Maps

```vira
mut stock := map[string]int{'apple': 12, 'milk': 4}
stock['bread'] = 2         // insert
stock['pear'] += 9         // a missing key starts at the zero value
n := stock['kiwi']         // 0: reading a missing key gives the zero value
if stock.has('milk') {
	stock.delete('milk')
}
for name, count in stock {  // insertion order
	println('${name}: ${count}')
}
```

| Operation | Meaning |
| :-- | :-- |
| `m[k]` | the value, or the zero value of V when k is missing |
| `m[k] = v`, `m[k] += v` | insert or replace (the old value is released) |
| `m.len` | number of entries |
| `m.has(k)`, `m.delete(k)` | membership, removal (`delete` returns whether k was there) |
| `m.keys()`, `m.values()` | new slices, in insertion order |
| `m.clear()` | remove everything |
| `for k in m`, `for k, v in m` | iterate in insertion order |

**Keys** are `string`, any integer type, `bool` or an enum. **Values** can be anything except actors: structs, slices, other maps, closures.

**Maps are values.** `copy := stock` is an independent copy; a map in a struct field is copied with the struct and freed with it; a `mut m map[string]int` parameter changes the caller's map.

**Change elements by storing them back.** Reading `m[k]` of a missing key returns a shared zero value, so changing an element in place is rejected:

```vira
mut accounts := map[string]Account{}
accounts['a'].tags.push('x')     // error: cannot change a map element in place
mut acc := accounts['a']         // copy out
acc.tags.push('x')               // change
accounts['a'] = acc              // store back
```

Do not insert into or delete from a map while a `for ... in` loop is iterating over it.

---

## 3. Closures & Function Types

A function type is written `fn (A, B) R` (no `R` for none). An anonymous function captures the outer variables it uses:

```vira
fn cheaper_than(limit int) fn (Item) bool {
	return fn (it Item) bool {
		return it.price < limit     // `limit` is captured
	}
}

is_cheap := cheaper_than(100)
println('${is_cheap(apple)}')
```

- **Captured by value, read-only.** The closure gets its own copy of each captured variable when it is created. Assigning to a captured variable is an error.
- **Named functions are values too:** `apply(double, 21)`, `apply(mathx.square, 3)`.
- Closures can be stored in variables, struct fields (`on_click fn (int) string`), slices (`[]fn (int) int`) and maps, returned, and sent to actors in messages.
- Copies of a closure share its captured values (they cannot change), so copying is cheap; the last copy to go frees them.
- Not allowed: `mut` parameters on an anonymous function, using a function with `mut` parameters as a value, and capturing an actor (it has a single owner; pass it as an argument).

---

## 4. Generics

### Generic functions

```vira
fn filter[T](xs []T, keep fn (T) bool) []T {
	mut out := []T{}
	for x in xs {
		if keep(x) {
			out.push(x)
		}
	}
	return out
}

fn map_to[T, U](xs []T, f fn (T) U) []U { ... }

cheap := filter(items, cheaper_than(100))                         // T = Item
names := map_to(cheap, fn (it Item) string { return it.name })   // T = Item, U = string
first[int]([5])                                                    // explicit
```

Type arguments are inferred from the arguments: through slices, maps, function types and generic structs. When a parameter does not appear in the arguments (`fn make[T]() []T`), write it: `make[int]()`. Explicit arguments must be simple type names.

### Generic structs

```vira
struct Stack[T] {
mut:
	items []T
}

fn Stack[T].new() Stack[T] {
	return Stack[T]{ items: []T{} }
}

fn (mut s Stack[T]) push(v T) {
	s.items.push(v)
}

mut orders := Stack[string].new()
orders.push('order-1')
```

Methods name the struct's parameters in the receiver (`fn (mut s Stack[T])`) or the static type (`fn Stack[T].new()`). From another module: `coll.Stack[int]{ ... }`, `coll.Stack[string].new()`.

### How it compiles

Each distinct use creates a concrete copy: `filter__Item`, `map_to__Item__string`, `Stack__string`. The copies are type-checked and compiled like code you wrote, so there is no runtime dispatch and no boxing. There are no constraints: if the concrete type does not support an operation, the error points at the template and names the instance:

```text
error[SEMANTIC]: operator '>' is not defined for type 'P'
 9 |     if a > b {
   = help: in max_of[P]
```

---

## 5. Run the Example

```bash
vira run docs/examples/48_enums_maps_closures_generics.vir
```

```text
fruit (fresh fruit): 21 in stock
dairy (dairy): 4 in stock
bakery (baked goods): 2 in stock
under 1.00: 2 items, first apple
stock value: 2310 cents
last order order-2, 1 left
[vira_rt] active_leaks=0 | total_freed=26
```

The compiler's own tests for these features are `testdata/run/20_enums.vir` to `25_generics.vir` (run under AddressSanitizer and UBSan) and `testdata/checker/07_enums.vir` to `10_generics.vir` (the error messages).
