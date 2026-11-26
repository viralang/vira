# Chapter 8: Enums & Structural Interfaces

While structs group data fields together, real systems also need **closed sets of named states** and **polymorphic contracts** across unrelated types. Vira covers both cleanly without class inheritance: **enums** define type-safe named 32-bit integer constants with built-in string formatting, and **structural interfaces** provide zero-boilerplate duck-typed polymorphism backed by 16-byte fat pointers, static vtables, and automatic destructor chaining.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch08_enums_and_interfaces.vir`](../code/ch08_enums_and_interfaces.vir)

---

## 8.1. Defining Enums, Explicit Integer Discriminants, and String Formatting

An `enum` defines a finite set of named variants backed by 32-bit signed integers (`int32_t`). By default, variants start at `0` and increment by `1`, or you can assign explicit integer discriminants and let subsequent variants continue counting upward:

```v
enum Priority {
	low
	medium
	high
	critical
}

pub enum HttpCode {
	ok          = 200
	created // Auto-increments to 201
	bad_request = 400
	not_found   = 404
}
```

### Enum Operations and Type Safety

- **Qualified Variant Syntax:** Always qualify variants with the enum type name (`Priority.high`, `HttpCode.created`).
- **Equality & `switch` Dispatch:** Compare enums with `==` and `!=`, or branch on them with `switch` (using comma-separated cases where variants share behavior).
- **Explicit Integer Conversion (`int(e)`):** Enums are strongly typed and never silently coerce to `int` or compare against raw integers (`Priority.high == 2` is a compile error). Use `int(code)` when you need the numeric discriminant.
- **String Interpolation (`${e}`):** Interpolating an enum prints its variant name (`'${Priority.high}'` formats as `'high'`, `'${HttpCode.created}'` formats as `'created'`).
- **Collections & Map Keys:** Enums can be stored in structs, slices (`[]Priority`), and used directly as `map[Priority]V` keys.

```v
fn priority_tag(p Priority) string {
	switch p {
		case Priority.low:
		return 'P3'
		case Priority.medium:
		return 'P2'
		case Priority.high, Priority.critical:
		return 'P1-URGENT'
	default:
		return 'P?'
	}
}
```

---

## 8.2. Structural Interfaces and Implicit Satisfaction (`interface`, `mut:` methods)

Vira uses **structural subtyping** (compile-time checked duck typing). There is no `implements` keyword: any struct that defines methods matching an interface's signatures automatically satisfies that interface.

Just like structs, interfaces distinguish read-only methods from mutating methods using an optional `mut:` section:

```v
pub interface Transport {
	label() string
	sent_count() int
mut:
	send(payload string) string
}
```

Both `HttpTransport` and `QueueTransport` satisfy `Transport` implicitly—simply by defining `label()`, `sent_count()`, and `mut` receiver `send(payload string) string`:

```v
pub struct HttpTransport {
pub:
	endpoint string
mut:
	count int = 0
}

pub fn (h HttpTransport) label() string {
	return 'HTTP(${h.endpoint})'
}

pub fn (h HttpTransport) sent_count() int {
	return h.count
}

pub fn (mut h HttpTransport) send(payload string) string {
	h.count += 1
	return 'POST ${h.endpoint} [${payload}] #${h.count}'
}
```

And `QueueTransport` carries an additional heap-allocated `last_payload string` field:

```v
pub struct QueueTransport {
pub:
	topic string
mut:
	count        int = 0
	last_payload string
}

pub fn (q QueueTransport) label() string {
	return 'QUEUE(${q.topic})'
}

pub fn (q QueueTransport) sent_count() int {
	return q.count
}

pub fn (mut q QueueTransport) send(payload string) string {
	q.count += 1
	q.last_payload = payload
	return 'ENQUEUE ${q.topic} [${q.last_payload}] #${q.count}'
}
```

### Compile-Time Satisfaction Checks

When you box a struct into an interface (`Transport(http)`), the type checker verifies every requirement:
- Missing a required method produces an immediate error (`struct '...' does not implement interface '...': missing method '...'`).
- Defining a `mut` receiver method on the struct when the interface declared it immutable is rejected (`struct '...' method '...' requires a 'mut' receiver, but interface '...' declares it immutable`).
- Calling a `mut:` interface method (such as `t1.send(...)`) requires the interface binding itself to be declared `mut` (`mut t1 := Transport(http)`). Immutable bindings (`t`) can still call read-only interface methods like `t.label()` and `t.sent_count()`.

---

## 8.3. 16-Byte Fat-Pointer Boxing, Dynamic vtable Dispatch, and Automatic Box Cleanup

To erase concrete struct sizes behind a uniform interface handle, boxing via `Transport(http)` allocates the struct on the heap and returns a **16-byte fat pointer** composed of two 64-bit pointers:

```text
+--------------------------------------+--------------------------------------+
|  ._object (void* to boxed struct)    |  ._vtable (const Transport_vtable*)  |
+--------------------------------------+--------------------------------------+
```

### How Boxing, Dispatch, and Cleanup Lower to C99

1. **Boxing (`Transport_box_HttpTransport`):** Allocates `sizeof(HttpTransport)` bytes on the heap, moves or copies the struct value into it (`*heap_obj = val`), records the heap allocation with the runtime leak tracker, and pairs it with the static `_vtable_HttpTransport_for_Transport` table.
2. **Dynamic vtable Dispatch:** Calling `t1.send('deploy-v1')` lowers directly to `t1._vtable->send(t1._object, ...)`—a single pointer indirection with no reflection or hash lookups. Notice that `mut:` methods in the vtable take `void* _self`, while read-only methods take `const void* _self`.
3. **Automatic Box Cleanup (`_autofree` Hook):** Every generated vtable appends an internal `void (*_autofree)(void* _self)` function pointer. When `t1` or `t2` leaves scope in `main()`, Autofree emits `vtable_autofree(&t2)` and `vtable_autofree(&t1)`:
   - For `QueueTransport`, `_vtable_free_QueueTransport_as_Transport` first calls `QueueTransport_autofree` (which frees `topic` and the updated `last_payload` string) and then `free(_obj)` on the boxed struct container itself.
   - Because `http` and `queue` were moved into their boxes (`MOVED_TO_CONTAINER`), Autofree skips freeing the original stack locals—preventing double-frees while ensuring `active_leaks=0`.

---

## 8.4. Running the Complete Chapter 8 Example

Run [`docs/handbook/chapters/code/ch08_enums_and_interfaces.vir`](../code/ch08_enums_and_interfaces.vir):

```bash
./vira run docs/handbook/chapters/code/ch08_enums_and_interfaces.vir
```

Output:
```text
[1] Enums: prio=high(2, P1-URGENT), code=created(201), not_found=404, map_high=4
[2] Dynamic Dispatch via Transport Interface:
    -> POST https://events.vira.dev [deploy-v1] #1
    -> POST https://events.vira.dev [health-ok] #2
    -> ENQUEUE alerts [cpu-spike] #1
    -> ENQUEUE alerts [recovered] #2
[3] Polymorphic Summary & Automatic Box Cleanup:
    status: HTTP(https://events.vira.dev) -> sent=2
    status: QUEUE(alerts) -> sent=2
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=25
```

---

## Summary & Exercises

1. **Compiler Satisfaction Diagnostic:** Add a `flush() bool` signature to `Transport` without adding `flush()` to `QueueTransport`, then run `./vira check docs/handbook/chapters/code/ch08_enums_and_interfaces.vir` to inspect the structural mismatch error.
2. **Enum Exhaustiveness & Mapping:** Add a `trace` variant at the start of `Priority` (shifting `low`, `medium`, `high`, `critical` up by 1), update `priority_tag`, and verify `int(p)` reflects the new discriminant.
3. **Inspect Generated VTable C99:** Run `./vira cgen docs/handbook/chapters/code/ch08_enums_and_interfaces.vir` and locate `Transport_vtable`, `_vtable_QueueTransport_for_Transport`, and `_vtable_free_QueueTransport_as_Transport` in the generated C99 output.
