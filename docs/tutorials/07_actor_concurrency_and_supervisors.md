# Tutorial 7: Actor Concurrency & Fault Supervision

> **Difficulty:** Advanced  
> **Matching Code Example:** [`docs/examples/07_actors_and_supervisors.vir`](../examples/07_actors_and_supervisors.vir)

---

## 1. Overview

Shared-memory multithreading with mutexes and locks is notoriously prone to deadlocks, race conditions, and priority inversions.

Vira adopts the **Actor Model of Concurrency**:
- Actors maintain completely isolated state (no shared mutable memory).
- Communication occurs exclusively through asynchronous message passing (`<-`).
- Calls return a non-blocking `Future[T]`.
- Work is scheduled across a lock-free Chase-Lev work-stealing thread pool.
- Failures (`panic`) are intercepted by Erlang-style supervisor fault boundaries with automatic **O(1)** arena wipes and `OneForOne` restarts.

---

## 2. Defining an Actor

Declare an actor using the `actor` keyword:

```vira
pub actor WorkerNode {
pub:
	id int
mut:
	jobs_completed int
}

pub fn (mut w WorkerNode) process_batch(workload_size int, trigger_fault bool) f32 {
	w.jobs_completed = w.jobs_completed + 1

	weights := tensor_fill(32, 32, 0.5)
	inputs := tensor_fill(32, 32, 1.0)
	result := weights @ inputs

	if trigger_fault {
		panic('WorkerNode numerical anomaly detected')
	}

	return tensor_sum(result)
}
```

Every actor automatically receives a dedicated mailbox spinlock and an isolated scratch arena slab.

---

---

## 3. Asynchronous Message Passing (`<-`) & Message Modes

Vira supports three distinct messaging patterns:

### 3.1 Asynchronous Futures (`fut := actor <- method()`)
Both jobs are queued onto the work-stealing pool immediately without blocking the caller:

```v
mut node_a := WorkerNode{ id: 101, jobs_completed: 0 }
mut node_b := WorkerNode{ id: 102, jobs_completed: 0 }

fut_a := node_a <- process_batch(32, false)
fut_b := node_b <- process_batch(32, true)
```

### 3.2 Detached Fire-and-Forget Sends (`actor <- method()`)
When the caller does not need the return value, dispatch the message as a standalone statement or discard the binding (`_ :=`):

```v
node_a <- process_batch(16, false)       // Standalone statement
_ := node_b <- process_batch(16, false)  // Discarded binding
```

Detached sends compile to `_send_detached_` (`done = -1`). The background worker thread automatically deallocates the task envelope when finished without creating or awaiting a `Future[T]` handle (`active_leaks = 0`).

### 3.3 Zero-Alloc Synchronous Calls (`res := actor <- method() or { ... }`)
If the caller immediately unwraps an actor call with an `or { ... }` fallback block, the compiler elides the heap-allocated envelope. The message envelope is stack-allocated on the caller's frame and synchronized directly:

```v
sync_res := node_a <- process_batch(32, false) or { 0.0 }
```

---

## 4. Work-Stealing Pool & `.await()`

To retrieve the result of an asynchronous actor call, invoke `.await()` on the returned `Future`:

```v
score_a := fut_a.await() or {
	eprintln('Node A unexpected error: ${err}')
	0.0
}
```

### The Chase-Lev Work-Stealing Scheduler
- **Cacheline Alignment:** Deque `top` and `bottom` pointers are isolated on separate 64-byte cachelines (`_Alignas(64)`), eliminating false sharing across threads.
- **Round-Robin Cursor:** Tasks submitted from non-worker threads use a dedicated round-robin cursor to balance load.
- **Monotonic Clocks:** Scheduler park conditions use `CLOCK_MONOTONIC` to avoid clock drift issues.
- **Work-Helping:** Calling `.await()` cooperates with the scheduler: if the awaited future is still running, the awaiting thread steals and executes pending tasks from victim queues instead of sleeping.

---

## 5. Fault Supervision, Restarts & Actor Lifecycle

When a worker thread panics (e.g. from a division by zero, invalid tensor dimension, or explicit `panic('...')` call):

```
+-------------------------------------------------------------+
| Worker Thread executes Actor Task                           |
|   ├── setjmp(fault_boundary) established                    |
|   ├── Actor calls panic('...')                              |
|   └── longjmp unwinds execution back to fault handler       |
+-------------------------------------------------------------+
                                │
                                ▼
+-------------------------------------------------------------+
| Supervisor Traps Error                                      |
|   ├── O(1) Arena Wipe: isolated_heap.offset = 0             |
|   ├── Check restarts < max_restarts (default: 3)            |
|   ├── If within limit:                                      |
|   │     ├── restarts++                                      |
|   │     ├── OneForOne State Reset: reset mutable fields     |
|   │     └── Return error via Result container               |
|   └── If limit exceeded:                                    |
|         ├── Transition to VIRA_ACTOR_STOPPED                |
|         └── Reject future messages as dead-letters          |
+-------------------------------------------------------------+
```

### Trapping the Fault
The supervisor logs the event to stdout and delivers the error message directly to the caller's `or { ... }` block:

```v
score_b := fut_b.await() or {
	println('Future caught supervised Node B fault: "${err}"')
	0.0 // Graceful fallback!
}
```

### Automatic Recovery & Restart Reset
Because the actor was reset and its memory was cleared in O(1), the actor is immediately healthy and ready to accept subsequent jobs. When an actor successfully completes a job, its consecutive `restarts` counter is automatically reset back to 0:

```v
fut_b_retry := node_b <- process_batch(32, false)
score_b_retry := fut_b_retry.await() or { 0.0 }
```

### Stopped State & Dead-Letter Rejection
If an actor suffers consecutive faults exceeding `max_restarts` (default: 3), it enters `VIRA_ACTOR_STOPPED`. Any further message dispatched to it fails immediately with `"Actor has stopped due to unhandled faults"`, protecting upstream callers from cascading failures. See [`docs/examples/32_actor_detached_and_lifecycle.vir`](../examples/32_actor_detached_and_lifecycle.vir) for a runnable demonstration.

---

## 6. Running the Code

```powershell
.\vira.exe run docs/examples/07_actors_and_supervisors.vir
```

Expected output:
```text
=== 1. INITIALIZING SUPERVISED ACTORS ===
=== 2. ASYNCHRONOUS FAN-OUT (<-) ACROSS THREAD POOL ===
=== 3. WORK-STEALING AWAIT WITH FAULT ISOLATION ===
Node A completed successfully: sum=16384.00
[vira_supervisor | worker_thread=0] Actor 'WorkerNode' (#102) trapped fault ('WorkerNode numerical anomaly detected') -> O(1) arena wipe & OneForOne restart [1/3]
Future caught supervised Node B fault: "WorkerNode numerical anomaly detected"
Node B result handled safely with fallback: 0.00
=== 4. POST-RESTART RECOVERY VERIFICATION ===
Node B recovered and completed subsequent job: sum=16384.00
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vira_rt] pool_workers=4 | actor_tasks_run=3 | work_steals=3
[vira_rt] active_leaks=0 | total_freed=9
```

---

## Next Steps

Proceed to [**Tutorial 8: Building Production Systems**](./08_building_production_systems.md) to see all these features integrated into an end-to-end production AI pipeline.
