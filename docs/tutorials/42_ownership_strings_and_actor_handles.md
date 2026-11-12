# Tutorial 42: Ownership, Strings & Actor Handles

This tutorial covers the language rules that changed after the v0.7.7 runtime and compiler review. Read it if you wrote Vira code before that point: some programs that used to compile are now compile errors, and a few behave differently.

Runnable example: [`47_ownership_strings_and_actor_handles.vir`](../examples/47_ownership_strings_and_actor_handles.vir).

| Topic | Rule in one line |
| :-- | :-- |
| [Strings](#1-strings-interpolation--and-) | `a + b` and `s += t` work on strings; a `+` chain makes one allocation. |
| [Operators](#2-operators-and-their-types) | Each operator works only on the types listed below; anything else is a compile error. |
| [Ownership](#3-ownership-copies-moves-and-borrowed-arguments) | Storing a value copies it, unless the source is never read again; then it moves. |
| [Actors](#4-actors-are-handles) | An actor lives on the heap, never moves, has one owner and is reached from outside only by messages. |
| [Deadlocks](#5-actors-that-wait-on-each-other) | Actors that await each other get an error instead of hanging. |
| [Faults & workers](#6-faults-restarts-and-the-worker-pool) | Divide by zero and bounds errors are supervised; the pool size follows the machine. |
| [Migrating](#7-migrating-code-written-for-v077-and-earlier) | What to change in older code. |

---

## 1. Strings: interpolation, `+` and `+=`

```vira
first := 'Ada'
last := 'Lovelace'
full := first + ' ' + last        // one allocation for the whole chain
greeting := 'Hello, ${full}!'     // interpolation, also one allocation

mut log := ''
for mut i := 1; i <= 3; i += 1 {
    log += 'step-${i};'           // builds a new string, frees the old one
}
```

- `+` joins two strings. A chain `a + b + c + d` is compiled into a single call that allocates once; there are no intermediate strings.
- `s += t` replaces `s` with `s + t` and frees the old value. `s += s` is fine.
- `==`, `!=`, `<`, `<=`, `>`, `>=` compare strings by content.
- Inside `arena { }`, the result of `+` lives on the arena like an interpolated string.
- Each expression inside `${...}` is evaluated exactly once.

Some string methods return a **view** into another string (`s.trim_left()`, `s[i]`). A view is copied automatically when you store it, so it can never outlive the string it points into.

---

## 2. Operators and their types

Both operands must have the same type, and the operator must be defined for that type:

| Operators | Defined for |
| :-- | :-- |
| `+` | numbers, strings (concatenation), tensors, pointers |
| `-` | numbers, tensors, pointers |
| `*`, `/` | numbers |
| `%` | integers |
| `&`, `\|`, `^` | integers, `bool` |
| `<<`, `>>` | integers |
| `<`, `<=`, `>`, `>=` | numbers, strings, pointers |
| `==`, `!=` | primitives (strings by content) and pointers; **not** structs, slices, tensors or actors |
| `&&`, `\|\|` | `bool` |
| `@` | tensors |

The same table applies to `+=`, `-=`, `*=`, `/=`, `%=`. Anything else is a compile error that names the operator and the type:

```text
error[SEMANTIC]: operator '-' is not defined for type 'string'
error[SEMANTIC]: operator '==' is not defined for type 'Point'
```

To compare structs, compare their fields or write a method such as `fn (p Point) equals(q Point) bool`.

**Arithmetic is defined, not undefined C behaviour:**

- Integer `/` and `%` by zero panic with `vira panic: integer division by zero`. Inside an actor the supervisor catches the panic (see section 6); in `main` the program stops with that message.
- Signed integer overflow wraps around (`2147483647 + 1 == -2147483648`), on every optimisation level.
- Function arguments are evaluated left to right.

---

## 3. Ownership: copies, moves and borrowed arguments

Every heap value (string, slice, tensor, struct with heap fields) has exactly **one owner**: a variable, a struct field, a slice element or an actor. The owner frees it when it goes out of scope. You never call `free`, and two owners never share memory.

### Storing a value copies it, unless the source is never read again

```vira
t := make_ticket(7)
copy := t                 // `t` is used again below: `copy` is an independent deep copy
println(headline(t))
println(copy.title)

mut titles := []string{}
for mut i := 0; i < 3; i += 1 {
    title := 'job-' + '${i}'
    titles.push(title)    // last use of `title`: the slice takes it, no copy
}
```

"Storing" means binding to another variable (`b := a`), assigning (`x = y`), putting into a struct field (`Box{ name: s }`) or into a slice (`xs.push(s)`, `xs.set(i, s)`).

The compiler **moves** instead of copying when it can prove the source is never read again on any path:

| Situation | Copy or move |
| :-- | :-- |
| Source not used after the store, same block | move |
| Source declared in an enclosing block, store inside an `if` or `switch` branch, not used after | move (the variable is reset, so its cleanup frees nothing) |
| Source used again later | copy |
| Store inside a loop, source declared outside the loop | copy (the next iteration would read it) |
| Source is a parameter, receiver, `for ... in` variable, field or element | copy |

You do not write anything to get a move; the program means the same either way. Modifying `a` after `b := a` never affects `b`.

### Arguments are borrowed

```vira
fn headline(t Ticket) string {
    return t.title + ' (' + '${t.labels.len}' + ' labels)'
}
```

A function receives its arguments without copying them; they still belong to the caller. If the function **stores** an argument (in a struct, a slice or a return value), that store is a copy. Returning a local variable moves it to the caller:

```vira
fn make_ticket(n int) Ticket {
    mut labels := []string{}
    labels.push('area-' + '${n % 3}')
    return Ticket{ title: 'Ticket #' + '${n}', labels: labels }   // `labels` moves into the struct
}
```

### Arenas

Values created inside `arena { }` live on the arena and are freed together when the block ends. A value assigned to a variable declared **outside** the arena is allocated on the heap instead, so it survives the block.

### Seeing what happened

Every program prints its allocation counters on exit:

```text
[vira_rt] active_leaks=0 | total_freed=336
```

`active_leaks=0` means everything was freed; a lower `total_freed` for the same work means fewer copies were made.

---

## 4. Actors are handles

```vira
pub actor Counter {
pub:
    name string          // immutable: readable from outside
mut:
    total int = 0        // mutable: only the actor's own methods touch it
}

pub fn (mut c Counter) add(n int) int {
    c.total += n
    return c.total
}
```

An actor is created on the heap when you construct it, and the variable holds a **handle** to it. The actor never moves: you can return it from a function, keep it in a struct field or grow a slice of actors while messages to it are still queued. When its owner goes out of scope, the actor finishes its queued messages and is destroyed.

### Reaching an actor from outside: messages only

| From outside the actor | Allowed? |
| :-- | :-- |
| `c <- add(1)` (send), `f := c <- add(1)` then `f.await()`, `c <- add(1) or { 0 }` | yes |
| Read an immutable field: `c.name` | yes |
| Read or write a mutable field: `c.total` | `cannot access mutable field 'Counter.total' from outside the actor` |
| Call a method directly: `c.add(1)` | `cannot call method 'add()' directly on actor 'Counter'` |

Sending requires a `mut` binding (`mut c := Counter{...}`), because a message may change the actor. Inside its own methods an actor uses its fields and methods directly.

### One owner per actor

```vira
fn make_counter(name string) Counter {
    mut c := Counter{ name: name, total: 0 }
    c <- add(10)          // still queued when the function returns: fine
    return c              // the local moves to the caller
}

mut pool := []Counter{}
for mut i := 0; i < 100; i += 1 {
    pool.push(Counter{ name: 'w-${i}', total: 0 })   // built in place
    pool[i] <- add(i)
}
```

An actor cannot be copied, and an existing actor cannot be stored in a second place, because the actor would then be destroyed twice:

| Code | Error |
| :-- | :-- |
| `b := a`, `x := pool[0]` | `actor 'Counter' cannot be copied or moved into another variable` |
| `Holder{ c: a }`, `pool.push(a)`, `pool.set(0, pool[1])` | `actor 'Counter' cannot be stored in ...: it already has an owner (construct a new actor in place instead)` |
| `return param` (a parameter, receiver or loop variable) | `cannot return actor 'Counter' through 'param': it is borrowed, the caller still owns it` |
| `return h.c`, `return pool[0]` | `cannot return actor 'Counter' from a field or element: its container owns it` |
| `pool.clone()`, `pool.slice(..)`, `pool.get(i)` | `cannot call '.clone()' on a slice of actor 'Counter': actors cannot be copied` |

What you can do instead:

- **Build it where it lives:** `Holder{ c: Counter{...} }`, `pool.push(Counter{...})`, or `pool.push(make_counter('x'))`.
- **Use it in place:** `pool[i] <- add(1)`, `h.c <- add(1)`.
- **Lend it to a function:** `fn report(c Counter)` can read immutable fields; `fn poke(mut c Counter)` can also send messages. Call them as `report(a)` and `poke(mut a)`.
- **Take it out of a slice:** `mut last := pool.pop()` moves the actor out; `pool.set(i, Counter{...})` and `pool.clear()` destroy the actors they replace.

**One limit to know:** an actor passed as an argument **inside a message** (`other <- ask(mut back)`) is lent to that message. Its owner must keep it alive until the message has run, for example by awaiting the result before the owner's scope ends.

---

## 5. Actors that wait on each other

An actor that awaits a message it sent to itself, or a group of actors that await each other in a cycle, would wait forever. Vira detects both and returns an error to one of the waiters instead:

```vira
pub fn (mut c Counter) ask(mut other Counter) int {
    return other <- add(1) or { -1 }
}

pub fn (mut c Counter) relay(mut other Counter, mut back Counter) int {
    f := other <- ask(mut back)
    return f.await() or { -100 }
}

// a.relay awaits b.ask, and b.ask awaits a: a cycle.
r := a <- relay(mut b, mut a) or { -2 }   // b's await fails, so b answers -1, so r == -1
```

| Situation | Error the waiter receives |
| :-- | :-- |
| An actor awaits a message it sent to itself | `deadlock avoided: an actor cannot wait on a message it sent to itself` |
| Actors await each other in a cycle (any length) | `deadlock avoided: actors are waiting on each other` |

The waiter that gets the error is the one whose message has not started yet. That message is not lost: it runs later, as if it had been sent without waiting. The rest of the cycle then completes normally. Waits that do not form a cycle are never interrupted, however deeply they nest.

Handle the error with `or { }` like any other failed message.

---

## 6. Faults, restarts and the worker pool

- A `panic`, an out-of-bounds index, `pop()` on an empty slice and integer division by zero inside an actor are all **caught by the supervisor**. The awaiting caller gets the panic message as its error, and the actor restarts.
- A restart resets the actor's **mutable** fields to their declared default (or zero). **Immutable** fields (ids, configuration, loaded model weights) and any actors stored in its fields are kept.
- An actor restarts up to 3 times in a row. A fourth consecutive fault stops it, and later messages fail with `Actor has stopped due to unhandled faults`. A message that completes normally resets the count.
- Messages from one sender to one actor run in the order they were sent.
- The worker pool has one thread per logical core, clamped to 2–16. Set `VIRA_WORKERS=<n>` (1–32) to choose another size, for example to compare benchmark runs.

An actor takes about 370 bytes plus its fields, and allocates a scratch arena only when a handler first uses one. To measure actor performance on your machine, run:

```bash
bash scripts/bench_actors.sh                 # 15 runs, 4 workers
RUNS=25 WORKERS=8 bash scripts/bench_actors.sh
```

It prints the median, minimum and maximum of five measurements and appends them to `bench/results.tsv`.

---

## 7. Migrating code written for v0.7.7 and earlier

| Old code | Change to |
| :-- | :-- |
| `println(worker.count)` (mutable field, from outside) | add a method `fn (mut w Worker) count() int` and use `n := worker <- count() or { 0 }` |
| `worker.method()` called directly on an actor | `worker <- method()` (or `... or { }` to wait for the result) |
| `ys := xs` and then relying on `ys` seeing later changes to `xs` | keep using `xs`, or reassign `ys = xs` after the change: copies are independent |
| `b := actor_a`, `pool.push(existing_actor)` | construct the actor where it is stored |
| `struct_a == struct_b` | compare fields |
| `'a' - 'b'`, `true + false`, `1.5 % 2.0` | these were never meaningful; the compiler now rejects them |
| Programs that expected exactly 8 worker threads | set `VIRA_WORKERS=8` |
| Code that relied on a crash for integer division by zero | it now panics with a message (supervised inside actors) |

---

## 8. Run the example

```bash
vira run docs/examples/47_ownership_strings_and_actor_handles.vir
```

```text
--- 1. Strings ---
Ada Lovelace
step-1;step-2;step-3;
same: true
--- 2. Ownership ---
Ticket #7 (1 labels)
Ticket #7
3 titles, first job-0
--- 3. Actors ---
a total 15
pool of 100, last total 99
cycle broken: b answered -1
a total after the cycle: 16
[vira_rt] active_leaks=0 | total_freed=336
```

`a total after the cycle: 16` shows that the message abandoned to break the cycle still ran: `a` received 10, 5 and then the `add(1)` from `b`.
