# Tutorial 10: Threads, Actors, Synchronization, and Fault Tolerance

> **Difficulty:** Advanced  
> **Matching Code Example:** [`docs/examples/10_threads_actors_fault_tolerance.vir`](../examples/10_threads_actors_fault_tolerance.vir)

---

## 1. Overview & Architecture

Modern multicore systems require concurrency models that eliminate data races, deadlocks, and cascading failures. Traditional shared-memory multithreading with manual mutexes, semaphores, and condition variables often leads to:
- **Heisenbugs:** Non-deterministic race conditions that depend on thread scheduling.
- **Deadlocks & Inversions:** Inconsistent lock ordering causing full application freeze.
- **State Corruption on Panic:** If a thread crashes while holding a lock, memory is left in an undefined state.

Vira solves these problems by combining **Actor Isolation**, **Chase-Lev Work-Stealing Threads**, **Non-blocking Message Passing**, **Future Synchronization**, and **Erlang-Style Supervisor Boundaries**:

```
+───────────────────────────────────────────────────────────────────────────────+
│                               VEX RUNTIME POOL                                │
│                                                                               │
│  Worker Thread 0       Worker Thread 1       Worker Thread 2   Worker Thread 3│
│   [Chase-Lev Deque]     [Chase-Lev Deque]     [Chase-Lev Deque] [Chase-Lev Deque]
│          ▲                     ▲                     ▲                 ▲      │
│          │   Steal Tasks       │   Steal Tasks       │                 │      │
│          └─────────────────────┴─────────────────────┴─────────────────┘      │
+───────────────────────────────────────────────────────────────────────────────+
                                       ▲
                                       │ Non-blocking Dispatch (<-)
+──────────────────────────────────────┴────────────────────────────────────────+
│                              ACTOR INSTANCES                                  │
│                                                                               │
│  Actor Node 1               Actor Node 2               Actor Node 3           │
│  ├── Mailbox Spinlock       ├── Mailbox Spinlock       ├── Mailbox Spinlock   │
│  ├── Isolated Scratch Heap  ├── Isolated Scratch Heap  ├── Isolated Heap (O1) │
│  └── State: {batches: 2}    └── State: {batches: 1}    └── Fault Boundary     │
│                                                              (setjmp/longjmp) │
+───────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Defining an Actor

An actor in Vira is declared with the `actor` keyword. Like structs, actors support 4-quadrant field encapsulation (`pub:`, `mut:`, `pub mut:`, or default private):

```vex
pub actor ComputeNode {
pub:
	node_id int
mut:
	batches_completed int = 0
	is_degraded bool = false
}
```

### What the Compiler Generates Under the Hood
When you declare `actor ComputeNode`, the C99 backend emits:
1. The user fields (`node_id`, `batches_completed`, `is_degraded`).
2. An embedded `VexActorContext _ctx` containing:
   - `atomic_flag mailbox_lock`: A lock-free spinlock ensuring sequential message processing.
   - `VexArena isolated_heap`: A private 64-byte aligned memory slab allocated exclusively for this actor.
   - `int restarts`: Supervisor fault restart counter.
   - `jmp_buf fault_env`: The `setjmp`/`longjmp` unwinding checkpoint for crash containment.

---

## 3. Actor Methods & Workloads

Methods on actors take a receiver (`(mut node ComputeNode)`). Inside an actor method, you can allocate local objects, perform SIMD matrix mathematics, and update internal state safely:

```vex
pub fn (mut node ComputeNode) compute_batch(multiplier f32, force_fault bool) f32 {
	node.batches_completed = node.batches_completed + 1

	// Hardware-accelerated matrix multiplication
	weights := tensor_fill(32, 32, 0.25)
	inputs := tensor_fill(32, 32, multiplier)
	output := weights @ inputs

	// Injected fault to test supervisor isolation
	if force_fault {
		node.is_degraded = true
		panic('Node #${node.node_id} thermal throttling / NaN anomaly')
	}

	return tensor_sum(output)
}
```

Because actor state is completely isolated, **no other thread can inspect or mutate `node` while `compute_batch` is running**.

---

## 4. Asynchronous Message Passing (`<-`)

In Vex, you interact with an actor by sending messages using the asynchronous arrow operator (`<-`):

```vex
mut n1 := ComputeNode{ node_id: 1, batches_completed: 0, is_degraded: false }
mut n2 := ComputeNode{ node_id: 2, batches_completed: 0, is_degraded: false }
mut n3 := ComputeNode{ node_id: 3, batches_completed: 0, is_degraded: false }
mut n4 := ComputeNode{ node_id: 4, batches_completed: 0, is_degraded: false }

// Asynchronous Fan-Out: non-blocking dispatch across the thread pool
fut1 := n1 <- compute_batch(1.0, false)
fut2 := n2 <- compute_batch(2.0, false)
fut3 := n3 <- compute_batch(3.0, true)  // Injected fault!
fut4 := n4 <- compute_batch(4.0, false)
```

### The Fan-Out Mechanism
1. The calling thread packages the target actor pointer and method arguments into a task closure.
2. The task is pushed onto the current worker's Chase-Lev lock-free deque.
3. The call returns **immediately**, yielding a typed `Future[f32]`.
4. Idle worker threads across the pool detect pending tasks and steal them using atomic CAS (compare-and-swap) operations.

---

## 5. Synchronization with Futures (`.await()`)

To synchronize and collect results (Fan-In), invoke `.await()` on the `Future` handle. In Vex, `.await()` returns a Result type (`!T`), requiring unwrap handling with `or { ... }`:

```vex
res1 := fut1.await() or {
	eprintln('Unexpected error on Node 1: ${err}')
	0.0
}

res2 := fut2.await() or {
	eprintln('Unexpected error on Node 2: ${err}')
	0.0
}
```

### Cooperative Awaiting
If the awaited future has not yet finished when `.await()` is called:
- The awaiting thread does **not** perform an expensive kernel sleep.
- Instead, it checks the work-stealing deque and executes other pending tasks while waiting, maximizing CPU core saturation and throughput.

---

## 6. Fault Tolerance & Erlang-Style Supervision

What happens when an actor panics? In languages like C++ or Go, an uncaught panic or segmentation fault terminates the entire process.

In Vex, the runtime encapsulates actor execution within an **Erlang-style Supervisor Boundary**:

```
[Worker Thread]
      │
      ├──> setjmp(node->_ctx.fault_env)
      │      ├── Execute actor method: compute_batch(...)
      │      │     └── panic('Node #3 thermal throttling / NaN anomaly')
      │      │
      │      └── longjmp unwinds stack back to supervisor boundary
      │
      ▼
[Supervisor Catch Block]
      ├── 1. Intercept panic message
      ├── 2. O(1) Arena Wipe: node->_ctx.isolated_heap.offset = 0 (prevents memory leaks)
      ├── 3. Increment node->_ctx.restarts (OneForOne policy)
      ├── 4. Reset mailbox spinlock to unlocked
      └── 5. Mark future as completed with Error: Result{ .is_err = true, .err = panic_msg }
```

### Handling Actor Crashes Gracefully
Because `.await()` returns a Result, the panic message is surfaced directly into the `or { ... }` block as the implicit `err` variable:

```vex
res3 := fut3.await() or {
	println('Supervisor trapped fault on Node 3 -> "${err}"')
	0.0 // Default fallback keeps the pipeline running
}
```

Output:
```
Supervisor trapped fault on Node 3 -> "Node #3 thermal throttling / NaN anomaly"
```

---

## 7. Self-Healing & Post-Restart Recovery

Because the supervisor wiped the actor's scratch memory in **O(1)** and reset its mailbox spinlock, the actor is immediately healthy and ready to accept new messages:

```vex
// Node 3 is fully operational; send a retry batch
fut3_retry := n3 <- compute_batch(3.0, false)
res3_recovered := fut3_retry.await() or {
	eprintln('Retry failed: ${err}')
	0.0
}
println('Node 3 successfully processed retry batch after restart: ${res3_recovered}')
```

Output:
```
Node 3 successfully processed retry batch after restart: 24576.00
```

---

## 8. Sequential Mailbox Guarantees

When multiple messages are sent to the **same** actor concurrently:

```vex
f_seq1 := n1 <- compute_batch(1.5, false)
f_seq2 := n1 <- compute_batch(2.5, false)

s1 := f_seq1.await() or { 0.0 }
s2 := f_seq2.await() or { 0.0 }

println('Sequential Mailbox for Node 1: run1=${s1}, run2=${s2}')
println('Total batches completed on Node 1: ${n1.batches_completed}')
```

- Each actor processes messages sequentially through its atomic spinlock (`__atomic_test_and_set`).
- State updates like `node.batches_completed = node.batches_completed + 1` are guaranteed to be serialized without races.
- Different actors (`n1`, `n2`, `n3`, `n4`) execute in parallel across different OS threads.

---

## 9. Running the Complete Example

Run the verified standalone program:

```powershell
.\vira.exe run docs/examples/10_threads_actors_fault_tolerance.vir
```

### Terminal Output
```
=================================================================
   VEX MULTICORE WORK-STEALING ACTORS & SUPERVISOR DEMO          
=================================================================
--- 1. INITIALIZING INDEPENDENT ACTOR NODES ---
Nodes initialized with isolated mailboxes and private arena slabs
--- 2. ASYNCHRONOUS FAN-OUT (<-) ACROSS CHASE-LEV WORK-STEALING POOL ---
Dispatched 4 asynchronous jobs returning Future[f32] handles
--- 3. SYNCHRONIZING & AWAITING RESULTS (.await) ---
Supervisor trapped fault on Node 3 -> "Node #3 thermal throttling / NaN anomaly"
Parallel Fan-In Sync Complete:
  Node 1 Result: 8192.00
  Node 2 Result: 16384.00
  Node 3 Result (Faulted & Handled): 0.00
  Node 4 Result: 32768.00
--- 4. POST-SUPERVISOR SELF-HEALING & RETRY ---
Node 3 successfully processed retry batch after restart: 24576.00
--- 5. SEQUENTIAL MAILBOX GUARANTEE VERIFICATION ---
Sequential Mailbox for Node 1: run1=12288.00, run2=20480.00
Total batches completed on Node 1: 3
=================================================================
   END OF CONCURRENCY DEMO (Review [vira_rt] Telemetry Below)     
=================================================================
[vira_rt] pool_workers=4 | actor_tasks_run=7 | work_steals=7 | active_leaks=0 | total_freed=19
```

### Key Telemetry Takeaways
- `pool_workers=4`: Automatically spawned 4 OS native worker threads matched to system CPU cores.
- `actor_tasks_run=7`: Successfully scheduled and completed 7 actor method invocations.
- `work_steals=7`: Idle worker threads dynamically stole tasks from busy workers' deques, proving multicore load balancing.
- `active_leaks=0`: Zero memory leaks despite dynamic string allocations, tensor matrix multiplications, and simulated panics.

---

## 10. Summary & Cheat Sheet

| Concurrency Need | Traditional C / C++ / Go | Vira Actor Model |
| :--- | :--- | :--- |
| **Data Sharing** | Shared mutable pointers with `mutex` | Completely isolated actor state |
| **Message Dispatch** | Channel buffering or thread pools | Native `<-` operator returning `Future[T]` |
| **Scheduler** | OS thread context switches | Lock-free Chase-Lev work-stealing pool |
| **Synchronization** | Blocking thread join or condition variables | Non-blocking `.await()` with cooperative stealing |
| **Crash Handling** | Process termination or corrupted lock state | Erlang-style supervisor `OneForOne` restart |
| **Memory Cleanup** | GC sweeps or manual pointer frees | Deterministic autofree + O(1) arena wipe |
