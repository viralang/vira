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

## 3. Asynchronous Message Passing (`<-`)

Send messages to an actor asynchronously using the `<-` send operator:

```v
mut node_a := WorkerNode{ id: 101, jobs_completed: 0 }
mut node_b := WorkerNode{ id: 102, jobs_completed: 0 }

// Both jobs are queued onto the work-stealing pool immediately
fut_a := node_a <- process_batch(32, false)
fut_b := node_b <- process_batch(32, true)
```

The calling thread does not block. The message is packaged into an envelope and submitted to the scheduler, immediately returning a `Future[f32]`.

---

## 4. Work-Stealing Pool & `.await()`

To retrieve the result of an actor call, invoke `.await()` on the returned `Future`:

```v
score_a := fut_a.await() or {
	eprintln('Node A unexpected error: ${err}')
	0.0
}
```

### The Chase-Lev Work-Stealing Scheduler
- Each worker thread in the pool has its own lock-free deque.
- When an idle thread runs out of tasks, it steals tasks from the tails of other workers' deques using atomic CAS (compare-and-swap) operations.
- Calling `.await()` cooperates with the scheduler: if the awaited future is still running, the awaiting thread can steal and execute pending tasks instead of sleeping.

---

## 5. Fault Supervision: `setjmp`/`longjmp` & `OneForOne` Restarts

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
|   ├── State Reset: WorkerNode_supervisor_reset(&self)       |
|   ├── Increment restart counter (OneForOne policy)          |
|   └── Pack error into Future: Result{ is_err: true, err }   |
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

### Automatic Recovery
Because the actor was reset and its memory was cleared in O(1), the actor is immediately healthy and ready to accept subsequent jobs:

```v
fut_b_retry := node_b <- process_batch(32, false)
score_b_retry := fut_b_retry.await() or { 0.0 }
```

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
