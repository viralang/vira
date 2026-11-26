# Chapter 14: Fault Supervision, Actor References (`Ref[T]`) & Event Bus

In real concurrent systems, handlers fail—slices go out of bounds, external gateways time out, and cycles of actors can accidentally wait on one another. At the same time, real topologies rarely look like a single star around `main`: workers need to report back to managers, pipelines need to forward work from stage to stage, and publishers need to broadcast events to dynamic subscribers. This chapter covers Vira's **Erlang-style actor supervision**, **wait-for deadlock avoidance**, **non-owning `Ref[T]` addresses**, and the **`eventbus`** package.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch14_supervision_refs_and_eventbus.vir`](../code/ch14_supervision_refs_and_eventbus.vir)

---

## 14.1. Erlang-Style Fault Containment (`setjmp`/`longjmp`, LIFO `defer` Unwinding, $O(1)$ Arena Wipe)

Outside an actor, an unrecoverable `panic(...)` or runtime trap (such as integer division by zero or an out-of-bounds slice index) prints an error and terminates the process. **Inside an actor message handler**, Vira wraps execution in a per-thread fault boundary so a failing actor never brings down its worker thread or sibling actors.

When a worker thread runs an actor method (`emitActorSupervisorWrapper` in the C backend):
1. It binds thread-local `_vira_tls_actor = &self->_ctx`, sets `self->_ctx.in_actor = true`, and arms `setjmp(self->_ctx.fault_boundary)`.
2. Every heap local (`string`, `[]T`, `map[K]V`) and scoped `arena` created inside the actor handler registers its cleanup hook on `ctx->defer_stack` (`_vira_actor_defer_push_ex`).
3. If `panic(...)` or `_vira_rt_fault(...)` triggers:
   - `vira_panic` copies the error string onto the heap (if it resided inside the actor's scratch arena or stack) into `ctx->last_error`.
   - `_vira_actor_defer_unwind(ctx)` walks `ctx->defer_stack` in **LIFO order**, freeing every live stack-frame heap local **before** unwinding the C stack.
   - `longjmp(ctx->fault_boundary, 1)` jumps cleanly back to the supervisor wrapper on the same worker thread.
4. In the `else` recovery branch, `vira_arena_reset(&self->_ctx.isolated_heap)` wipes the actor's scratch arena in **$O(1)$ time**, and the error message is delivered to the caller's `Future[T]` / `or { ... }` block as `err`.

```v
pub fn (mut l LedgerWorker) apply(delta int, mode string) int {
	// Heap locals (`trace`, `scratch`) are tracked on `ctx->defer_stack` so `vira_panic`
	// unwinds them LIFO before `longjmp`, while the actor's scratch arena is wiped in O(1).
	mut trace := []string{}
	trace.push('op:${mode}')
	scratch := 'ledger:${l.account}:${mode}'
	_ := scratch
	l.ops += 1
	if mode == 'panic' {
		panic('gateway fault on ${l.account}')
	}
	if mode == 'oob' {
		xs := [10, 20, 30]
		idx := l.ops + 5
		return xs[idx]
	}
	l.balance += delta
	return l.balance
}
```

---

## 14.2. `OneForOne` Supervisor Restarts and Dead-Letter Stopping (`VIRA_ACTOR_STOPPED`)

After trapping a fault, the supervisor checks the actor's consecutive restart counter (`restarts` against `max_restarts`, initialized to `3`):

- **Healthy Recovery (`restarts < max_restarts`):**
  1. `restarts` increments by 1 and `state` returns to `VIRA_ACTOR_RUNNING`.
  2. `<Actor>_supervisor_reset(self)` resets all mutable (`mut:` / `pub mut:`) fields back to their declared default values while **preserving `pub:` identity fields** (such as `account: 'ops-core'`) and the actor's `ref_slot`.
  3. The runtime logs `[vira_supervisor | worker_thread=...] Actor '...' trapped fault (...) -> O(1) arena wipe & OneForOne restart [k/3]`.
  4. Any subsequent handler that completes without panicking resets `self->_ctx.restarts = 0`—so intermittent faults never permanently kill a long-lived service.
- **Permanent Stop (`VIRA_ACTOR_STOPPED`):**
  If consecutive faults exhaust `max_restarts` (`3` restarts, so the 4th consecutive failure), the supervisor transitions the actor to `VIRA_ACTOR_STOPPED`. Every later message sent to that actor hits the guard at the top of the wrapper: its cloned parameters are freed and the caller immediately receives the dead-letter error `"Actor has stopped due to unhandled faults"`.

```text
  VIRA_ACTOR_RUNNING ──(panic / rt fault, restarts < 3)──▶ OneForOne Reset (restarts++)
         ▲                                                         │
         └────────────────(any successful handler: restarts = 0)───┘
         │
         └──(4th consecutive fault: restarts >= max_restarts)──▶ VIRA_ACTOR_STOPPED
                                                                    └─▶ Dead-letter error
```

---

## 14.3. Wait-For Graph Cycle Detection and Automatic Deadlock Avoidance

Because synchronous actor sends (`other <- method() or { ... }`) and `.await()` block the calling handler until the target replies, two actors waiting on each other (`A` awaits `B`, `B` awaits `A`)—or an actor awaiting a message it sent to itself—would normally deadlock forever. Vira resolves both cases automatically:

1. **Self-Await Detachment (`_vira_future_detach_if_self`):** If an actor handler sends a message to *itself* and attempts to await it, that message is queued behind the currently running handler and can never start while the handler waits. The runtime atomically CAS-transitions the message from pending (`0`) to `VIRA_TASK_DETACHED` (so it still runs after the current handler finishes and the drain loop frees its envelope) and fails the wait immediately with `"deadlock avoided: an actor cannot wait on a message it sent to itself"`.
2. **Global Wait-For Graph (`vira_actor_await_msg`):** Every actor allocates a slot in a lock-free chunked wait-for table (`_vira_wf_chunks`). While an actor handler awaits another actor, it publishes a 64-bit edge `(seq << 32) | (target_wf + 1)` and records nested work-helped actors on `_vira_tls_wf_stack`. Every 256 spin iterations, it walks the edges from its target (`_vira_wf_walk`).
   - Only if **two separate walks** see the exact same edge chain reaching back to the waiter (or an actor suspended beneath it on the same thread's work-helping stack)—proving the cycle held continuously rather than being a transient chain—does the waiter CAS `task->done` from `0` to `VIRA_TASK_DETACHED` and abort the wait with `"deadlock avoided: actors are waiting on each other"`.
   - Because only a pending (`0`) message can be abandoned (a `VIRA_TASK_RUNNING` message already started), exactly one waiter in the cycle breaks it, its abandoned message still runs cleanly once the cycle unwinds, and non-cyclic chains (`A -> B -> C`) are never interrupted.

---

## 14.4. Non-Owning Actor References (`Ref[T]`, `.ref()`, `.alive()`) for Direct Actor-to-Actor Topologies

Because an actor has a single owner, storing an actor handle inside multiple peers or closures is forbidden. Instead, Vira provides **`Ref[T]`**—a small, copyable, **non-owning address** to an actor of type `T`:

- **Creating & Checking `Ref[T]`:**
  - Call `actor.ref()` (either from the owner or from inside the actor's own method) to get a `Ref[T]`.
  - Construct an empty/null reference with `Ref[T]{}`.
  - Call `r.alive()` to check whether the target actor still exists and is not `VIRA_ACTOR_STOPPED`.
- **Where `Ref[T]` Can Go:** Unlike an owning handle, `Ref[T]` can be freely copied, stored in struct/actor fields or slices, sent inside actor messages, and captured by closures (`fn () { r <- ... }`).
- **Generation-Safe Slot Table:** Internally, `Ref[T]` is a 64-bit `(generation << 32) | (slot + 1)` token pointing into chunked `_vira_ref_chunks`. Sending through a `Ref[T]` atomically **pins** the slot (`vira_ref_pin`), enqueues the message, and unpins. When the owning actor is destroyed (`vira_actor_ctx_destroy`), it sets `VIRA_REF_CLOSING`, drains active pins while helping the scheduler, quiesces the mailbox, and bumps the slot's generation counter (`_vira_ref_release`). As a result:
  - A stale `Ref[T]` to a destroyed actor can never touch freed memory or collide with a newly allocated actor reusing that slot.
  - Synchronous calls and `.await()` on a stale `Ref[T]` fail cleanly with `"actor is gone"`; fire-and-forget sends drop silently.
  - Supervisor restarts **preserve `ref_slot`**, so existing `Ref[T]` handles remain valid across `OneForOne` restarts!

---

## 14.5. Decoupled Pub/Sub and Request/Reply with `import eventbus`

Built directly on top of `Ref[T]`, `import eventbus` provides two generic actor abstractions for decoupled systems:

### 1. `eventbus.Topic[T]` (Broadcast `publish` and Round-Robin `send`)

A `Topic[T]` is itself an actor holding registered consumer closures `fn (T) bool`:

```v
mut bus_actor := eventbus.Topic[TelemetryEvent]{ name: 'metrics.cpu' }
bus := bus_actor.ref()

sub_a := bus <- consumer(fn (ev TelemetryEvent) bool {
	ra <- record(ev)
	return ra.alive() // returning false auto-unregisters a gone consumer!
}) or { 0 }

bus <- publish(TelemetryEvent{ source: 'node-1', value: 15 }) // all consumers
bus <- send(TelemetryEvent{ source: 'node-3', value: 10 })    // one consumer (round-robin)
_ := bus <- unregister(sub_a) or { false }
st := bus <- stats() or { eventbus.TopicStats{} }
```

Because each consumer closure forwards to a `Ref[Actor]` and returns `ra.alive()`, the topic automatically prunes consumers whose actors have been dropped or stopped (`st.removed`).

### 2. `eventbus.Service[T, R]` (Request/Reply Load Balancing)

A `Service[T, R]` pairs request handlers `fn (T) R` with liveness predicates `fn () bool`:

```v
mut svc_actor := eventbus.Service[int, int]{ name: 'math.scale' }
svc := svc_actor.ref()

_ := svc <- handler(fn (x int) int {
	return rm1 <- compute(x) or { 0 }
}, fn () bool {
	return rm1.alive()
}) or { 0 }

r1 := eventbus.request(svc, 7) or { 0 }
```

Notice the critical design detail in `eventbus.request(svc, v)`: `svc <- pick()` selects the next live handler round-robin and returns the `fn (T) R` closure to the caller, so the handler executes in the **caller's own context**—keeping the `Service` actor completely unblocked while downstream workers process requests.

---

## 14.6. Running the Complete Chapter 14 Example

Run [`docs/handbook/chapters/code/ch14_supervision_refs_and_eventbus.vir`](../code/ch14_supervision_refs_and_eventbus.vir):

```bash
./vira run docs/handbook/chapters/code/ch14_supervision_refs_and_eventbus.vir
```

Output:
```text
=== [1] Fault Containment, LIFO defer & OneForOne Restarts ===
Healthy apply: balance=150
[vira_supervisor | worker_thread=-1] Actor 'LedgerWorker' (#1) trapped fault ('gateway fault on ops-core') -> O(1) arena wipe & OneForOne restart [1/3]
Trapped fault #1: gateway fault on ops-core
After restart #1 (Ref still alive=true): account=ops-core, balance=100, ops=0
Healthy call resets restart counter: balance=125
[vira_supervisor | worker_thread=-1] Actor 'LedgerWorker' (#1) trapped fault ('/tmp/vira_run_25329792/out.c:4633: vira panic: index out of bounds: index 7 with len 3') -> O(1) arena wipe & OneForOne restart [1/3]
Fault burst 1/4: /tmp/vira_run_25329792/out.c:4633: vira panic: index out of bounds: index 7 with len 3
[vira_supervisor | worker_thread=-1] Actor 'LedgerWorker' (#1) trapped fault ('/tmp/vira_run_25329792/out.c:4633: vira panic: index out of bounds: index 6 with len 3') -> O(1) arena wipe & OneForOne restart [2/3]
Fault burst 2/4: /tmp/vira_run_25329792/out.c:4633: vira panic: index out of bounds: index 6 with len 3
[vira_supervisor | worker_thread=-1] Actor 'LedgerWorker' (#1) trapped fault ('/tmp/vira_run_25329792/out.c:4633: vira panic: index out of bounds: index 6 with len 3') -> O(1) arena wipe & OneForOne restart [3/3]
Fault burst 3/4: /tmp/vira_run_25329792/out.c:4633: vira panic: index out of bounds: index 6 with len 3
Fault burst 4/4: /tmp/vira_run_25329792/out.c:4633: vira panic: index out of bounds: index 6 with len 3
After 4 consecutive faults: alive=false
Dead-letter rejection: Actor has stopped due to unhandled faults

=== [2] Wait-For Graph Cycle & Self-Deadlock Avoidance ===
Self-wait result=-77, two-way cycle result=0, abandoned pings ran=1

=== [3] Non-Owning Ref[T] & Stale Reference Safety ===
Stale ref alive=false
Stale ref sync error: actor is gone

=== [4] Decoupled Pub/Sub & Request/Reply (import eventbus) ===
Topic publish: sink_a=40, sink_b=40
Topic after unregister+send: sink_b=50, stats(consumers=1, published=3, delivered=5, removed=1)
Service round-robin replies: 14, 70
[vira_rt] pool_workers=4 | actor_tasks_run=44 | work_steals=36
[vira_rt] active_leaks=0 | total_freed=92
```

---

## Summary & Exercises

1. **Three-Way Wait-For Cycle:** Add a third `PeerNode` (`c := PeerNode{ id: 3 }`) and a `relay3` method so `a` awaits `b`, `b` awaits `c`, and `c` awaits `a`. Verify the cycle breaks without hanging and all abandoned `ping()` calls still complete.
2. **Auto-Pruning Dead Consumers:** Register a consumer from a helper function that creates a temporary `MetricSink` actor and returns its `Ref[MetricSink]`. Publish two events and inspect `stats().removed` to see `eventbus.Topic` automatically drop the dead consumer.
3. **Intermittent vs. Consecutive Faults:** Modify the fault burst loop in `ch14_supervision_refs_and_eventbus.vir` to interleave a healthy `apply(10, 'ok')` call after every 2nd fault, and verify `lr.alive()` stays `true` indefinitely because successful handlers reset `restarts` to `0`.
