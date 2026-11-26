# Chapter 13: Fearless Actor Concurrency

Shared-memory multithreading with raw locks invites data races, lock-ordering bugs, and subtle memory corruption. Vira takes a different path: **isolated, single-owner actors** backed by a multi-threaded Chase-Lev work-stealing scheduler. Every actor owns its mutable state exclusively, processes one message at a time from a lock-free mailbox, and enforces isolation at **compile time** so data races across actors cannot type-check.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch13_actor_concurrency.vir`](../code/ch13_actor_concurrency.vir)

---

## 13.1. Defining Actors (`actor`) and Single-Owner Handle Semantics

An actor is declared with the `actor` keyword using the same 4-quadrant visibility blocks (`pub:`, `mut:`, `pub mut:`, and default private immutable) as structs:

```v
pub actor TileWorker {
pub:
	id int
	zone string
mut:
	completed int = 0
	energy_sum f32 = 0.0
	audit_log []string
}
```

### Single-Owner Handle Semantics

Under the hood, an actor is **not** a plain by-value struct copied around on the stack. In generated C99, an actor value is a **heap-allocated handle (`T*`)** paired with an embedded `vira_actor_ctx` (mailbox pointers, scheduling state, wait-for slot, reference slot, and scratch arena). Because worker threads and queued messages hold pointers to that context, Vira enforces strict **single-owner handle rules** in the type checker:

1. **No Copying or Aliasing:** Assigning an existing actor variable to another variable (`b := a`) or calling `.clone()` on a `[]TileWorker` slice is a compile-time error. Each actor has **exactly one owner** (a variable, struct field, or slice element).
2. **In-Place Construction in Containers:** You can build an actor inside a helper function and `return` it—even with messages already queued in its mailbox—or push newly constructed actors directly into a slice (`pool.push(TileWorker{ ... })`). Because the handle points to a fixed heap object, returning or growing a slice never relocates the actor's address, and Autofree calls `vira_actor_quiesce` (draining queued work while helping the scheduler) before destroying the handle.
3. **No Maps of Actors & No Closure Capture:** `map[K]Actor` and capturing an actor directly inside an anonymous function (`fn () { ... }`) are rejected at compile time. Pass the actor as a `mut` parameter, or create a non-owning `Ref[T]` handle (covered in Chapter 14).
4. **No `==` Operator:** Like structs and slices, actor handles do not define `==`.

```v
// Actors live on the heap behind a handle, so returning one with queued work is safe.
fn spawn_primed_worker(id int, zone string) TileWorker {
	mut w := TileWorker{
		id: id,
		zone: zone,
		completed: 0,
		energy_sum: 0.0,
		audit_log: []string{},
	}
	w <- record_note('boot:${zone}')
	return w
}
```

---

## 13.2. Compile-Time Actor State Isolation

Vira's type checker tracks whether it is currently checking a method body belonging to an actor (`curActor`). Outside that actor's own methods, **state isolation is enforced statically**:

| Operation Outside Actor | Allowed? | Why |
| :--- | :--- | :--- |
| Reading `pub:` immutable field (`lead.id`) | **Yes** | Immutable `pub:` fields are set once at construction and never mutate. |
| Reading or writing `mut:` / `pub mut:` field (`lead.completed`) | **No** | Compile error: mutable state can only be touched through a message (`lead <- method()`). |
| Direct method call (`lead.compute_tile(...)`) | **No** | Compile error: calling a method directly would bypass the actor's mailbox and run without serialization. |
| Message send (`lead <- compute_tile(...)`) | **Yes** | Enqueues the call on the actor's mailbox and executes it sequentially on the worker pool. |

Inside `TileWorker`'s own methods (`fn (mut w TileWorker) compute_tile(...)`), `w` has direct, uncontended access to all of its fields—no mutexes are needed in user code because the runtime guarantees at most one handler for a given actor executes at any instant.

```v
pub fn (mut w TileWorker) compute_tile(dim int, scale f32) f32 {
	w.completed += 1
	weights := tensor_fill(dim, dim, 0.25)
	inputs := tensor_fill(dim, dim, scale)
	product := weights @ inputs
	total := tensor_sum(product)
	w.energy_sum += total
	return total
}
```

> **Concurrency Safety of Arguments:** When you send a message `actor <- method(arg)`, non-`mut` heap arguments (`string`, `[]T`, `tensor[f32]`, structs) are cloned into the message envelope so the sender and the receiving worker thread never share mutable buffers.

---

## 13.3. Message Passing Modes (`<-`): Async `Future[T]` (`.await()`, `.ready()`), Fire-and-Forget, and Synchronous Calls

Every interaction with an actor from the outside uses the `<-` send operator. Depending on how you use the expression at the call site, Vira lowers `<-` into one of **three execution modes**:

### 1. Asynchronous `Future[T]` (`.await()` and `.ready()`)

Assigning or storing `actor <- method(...)` without an inline `or { ... }` block immediately enqueues the message and returns a `Future[T]` handle without blocking the caller:

```v
fut_lead := lead <- compute_tile(16, 2.0)
mut polls := 0
for !fut_lead.ready() {
	polls += 1
	time_sleep_ms(1)
}
lead_energy := fut_lead.await() or { f32(0.0) }
```

- **`.ready()`**: Returns `true` as soon as the message has finished (or faulted), without blocking or consuming the future. This lets an event loop in `main` check replies on each tick. Calling `.ready()` again after `.await()` also returns `true`.
- **`.await() or { ... }`**: Extracts the `!T` result, using **work-helping** while waiting—if the task is not finished yet, the waiting thread steals and runs other pending tasks from the scheduler instead of sitting idle.
- **Automatic Future Cleanup:** If a `Future[T]` goes out of scope without `.await()` (for instance due to an early `return`), Autofree emits `vira_future_free`, which waits for the worker task to finish before freeing the unconsumed envelope and result.

You can store futures in a `[]Future[T]` slice to implement parallel fan-out / fan-in:

```v
mut futures := []Future[f32]{}
for mut step := 0; step < 8; step += 1 {
	idx := step % pool.len
	scale := f32(step + 1) * 0.5
	futures.push(pool[idx] <- compute_tile(16, scale))
}
```

### 2. Fire-and-Forget (Detached Statement Send)

When you write a bare send statement (`lead <- record_note('checkpoint-1')`) or discard it (`_ := lead <- record_note(...)`), the compiler marks the task with `VIRA_TASK_DETACHED`. The caller returns immediately without creating a `Future[T]`, and the actor's drain loop automatically drops any result and frees the message envelope once the handler finishes.

### 3. Synchronous Calls (`<- ... or { ... }`)

Attaching `or { ... }` directly to a send turns it into a **synchronous call**:

```v
lead_summary := lead <- summary() or { 'unavailable' }
```

Instead of heap-allocating an envelope and a `Future[T]`, the compiler places the message envelope on the caller's **stack**, enqueues it on the target actor's mailbox, and awaits it inline with work-helping—achieving zero heap allocations for the call envelope itself.

---

## 13.4. Inside the Lock-Free Chase-Lev Work-Stealing Scheduler

Vira's actor runtime (`runtime/rt/scheduler/scheduler.h`) initializes lazily the first time a program spawns or messages an actor (`vira_scheduler_ensure_init`). By default, it sizes the pool from the machine's logical core count clamped to `[2, 16]` (overrideable via `VIRA_WORKERS`, up to `VIRA_MAX_WORKERS = 32`).

```text
  Sender (main or worker)
     │
     ▼  atomic exchange on mbox_tail
  ┌──────────────────────────┐
  │ Actor Intrusive MPSC     │  state: IDLE -> SCHEDULED -> NOTIFIED
  │ Mailbox (mbox_head/tail) │
  └────────────┬─────────────┘
               │ pushes drain_task
               ▼
  ┌──────────────────────────┐        steal (FIFO CAS on top)
  │ Worker i: vira_ws_deque  │ ◀──────────────────────────── Idle Worker j
  │ [4096-slot ring buffer]  │
  └────────────┬─────────────┘
               │ pop (LIFO bottom)
               ▼
     _vira_actor_drain_task (up to quota = 64 messages per turn)
```

Key architectural properties of the scheduler:

1. **Cache-Line-Isolated Chase-Lev Deques (`vira_ws_deque`):** Each worker thread owns a 4096-slot ring buffer (`VIRA_WS_CAP = 4096`). Its `top`, `bottom`, `push_lock`, and `ring` fields are cache-line aligned to prevent false sharing. The owning worker pushes and pops at `bottom` in **LIFO** order for hot L1/L2 cache locality, while idle peers steal from `top` in **FIFO** order via an atomic compare-and-swap (`__atomic_compare_exchange_n`). If a local deque saturates, `vira_ws_push_mp` falls back to a global `vira_overflow_queue`.
2. **Intrusive MPSC Mailboxes & Fairness Quota:** Each actor has a lock-free multi-producer single-consumer linked list (`mbox_head` / `mbox_tail`) and an atomic 3-state scheduler flag (`VIRA_ACTOR_IDLE`, `VIRA_ACTOR_SCHEDULED`, `VIRA_ACTOR_NOTIFIED`). Sending to an idle actor schedules a single self-managed `drain_task` that processes up to `quota = 64` messages before yielding back to the deque—preventing one chatty actor from starving others.
3. **Bounded Work-Helping & Low-Power Parking:** When a caller awaits a future (`vira_scheduler_await`), `_vira_sched_help_one` steals and executes pending tasks on the waiting thread (bounded at `VIRA_MAX_HELP_DEPTH = 32` to protect the C stack). When all deques and the overflow queue are empty, idle workers spin (`VIRA_SPIN_ITERS = 64`), yield (`VIRA_YIELD_ITERS = 128`), and finally park on a `CLOCK_MONOTONIC` condition variable (`_vira_park` with a 50 ms fallback poll) so idle actor pools consume near-zero CPU.

---

## 13.5. Running the Complete Chapter 13 Example

Run [`docs/handbook/chapters/code/ch13_actor_concurrency.vir`](../code/ch13_actor_concurrency.vir):

```bash
./vira run docs/handbook/chapters/code/ch13_actor_concurrency.vir
```

Output:
```text
=== [1] Actor Construction & Single-Owner Handles ===
Lead actor id=0, zone=north
Pool ready: 4 workers

=== [2] Asynchronous Fan-Out (<-) & .ready() Polling ===
Lead tile ready after polling (ready_after_await=true): energy=2048.00
Fan-in aggregated 8 tiles: total_energy=18432.00

=== [3] Fire-and-Forget & Zero-Alloc Synchronous Calls ===
  worker#0(north): tasks=1, notes=2, energy=2048.00
  worker#1(shard-0): tasks=2, notes=1, energy=3072.00
[vira_rt] simd=x86_64 SSSE3 + SSE2 (128-bit Integer/Float SIMD, Multi-Threaded)
[vira_rt] pool_workers=4 | actor_tasks_run=14 | work_steals=5
[vira_rt] active_leaks=0 | total_freed=46
```

*(Exact `work_steals` counts may vary slightly across runs depending on OS thread scheduling, while `active_leaks=0` holds deterministically.)*

---

## Summary & Exercises

1. **Testing State Isolation:** Try calling `lead.compute_tile(8, 1.0)` directly without `<-`, or reading `lead.completed` directly inside `main()` in `ch13_actor_concurrency.vir`, and run `./vira check` to inspect the compile-time actor isolation diagnostics.
2. **Single-Owner Handle Enforcement:** Try assigning `alias := lead` or adding `pool.clone()` and run `./vira check` to observe how Vira prevents actor aliasing and double-destruction.
3. **Tuning Worker Pool Size:** Run `VIRA_WORKERS=2 ./vira run docs/handbook/chapters/code/ch13_actor_concurrency.vir` and compare `pool_workers` and `work_steals` in the `[vira_rt]` telemetry line against the default core-detected worker count.
