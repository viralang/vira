# Tutorial 26: Complete Actor System, Fault Tolerance & High-Concurrency Benchmarks

Vira combines the fault isolation and concurrency model of **Erlang/OTP** with the raw execution speed and deterministic memory management of **C/C++**. This tutorial explores the architecture, implementation patterns, empirical benchmarks, and architectural trade-offs of Vira's native Actor system.

---

## 1. Actor Model Core Architecture

In Vira, an `actor` is an isolated computational entity with private state, a lock-free mailbox that runs one message at a time, a scratch bump arena allocated on first use, and Erlang-style supervisor crash protection. Each actor is a heap object reached through a handle: it never moves, has exactly one owner, and is reached from outside only by messages ([Tutorial 42](./42_ownership_strings_and_actor_handles.md#4-actors-are-handles)).

```
                           +-------------------------------------------------+
                           |           Vira Application / Caller             |
                           +-------------------------------------------------+
                                     |                           ^
              actor <- method(args)  |                           |  fut.await() or { ... }
             (non-blocking dispatch) |                           |  (work-helping await)
                                     v                           |
    +---------------------------------------------------------------------------------+
    |               Chase-Lev Lock-Free Work-Stealing Scheduler Pool                  |
    |                                                                                 |
    |  +--------------------+  +--------------------+       +--------------------+    |
    |  | Worker 0 Deque     |  | Worker 1 Deque     |  ...  | Worker N Deque     |    |
    |  | [Task] [Task] [..] |  | [Task] [Task]      |       | [Task]             |    |
    |  +--------------------+  +--------------------+       +--------------------+    |
    |           |                        ^                                            |
    |           +--- (FIFO Steal via CAS)+                                            |
    +---------------------------------------------------------------------------------+
                                     |
                                     v  Executes task under thread-local fault boundary
    +---------------------------------------------------------------------------------+
    |                               Active Actor Instance                             |
    |                                                                                 |
    |  - Private State: Mut fields protected by atomic mailbox lock                   |
    |  - Fault Boundary: setjmp / longjmp traps panics without crashing thread        |
    |  - Scratch Arena: Wiped in O(1) on panic or task exit                           |
    |  - OneForOne Supervisor: Automatic restart & state reset on failure             |
    +---------------------------------------------------------------------------------+
```

### Key Language Primitives

1. **`actor Name { ... }`**: Declares an actor struct. The compiler automatically synthesizes `_ctx` (mailbox lock, jump buffer, restart counter, and lazy arena).
2. **`actor <- method(args)`**: Asynchronously pushes a task onto the work-stealing pool and returns immediately with a `Future[Result[T]]`.
3. **`fut.await() or { err => ... }`**: Awaits future completion. While waiting, the caller thread steals pending tasks from peer workers (work-helping), eliminating idle thread stalls.
4. **`panic('reason')` inside Actor**: Trapped immediately by the actor's thread-local fault boundary. Triggers O(1) arena slab wiping and supervisor OneForOne restart.

---

## 2. In-Depth Architectural Trade-Offs

| Dimension | Vira Lightweight Actors | Traditional OS Threads (pthreads / Win32) | Traditional GC Tracing Actors (Erlang BEAM / Akka) |
| :--- | :--- | :--- | :--- |
| **Memory Footprint** | **~370 bytes / actor** + its fields (scratch arena allocated on first use) | **1 MB to 8 MB** per thread stack | **300 bytes to 2 KB** per actor process |
| **Spawn Rate** | **10,000,000+ actors/sec** (create + destroy, measured on a 2-core VM) | ~10,000 threads/sec (OS bottleneck) | ~1,000,000 actors/sec |
| **Scheduling** | User-space **Chase-Lev work-stealing deques** | Kernel context switching (ring 0 / ring 3) | Preemptive bytecode reduction counter |
| **Fault Isolation** | **`setjmp` / `longjmp` per actor** (O(1) arena wipe) | Process abort (`SIGSEGV` / unhandled exception) | Process link / monitor signal propagation |
| **Memory Management** | **Deterministic Autofree** (0 pause, 0 GC leaks) | Manual `malloc` / `free` (vulnerable to leaks) | Stop-the-world / concurrent generational GC |
| **Data Sharing** | Isolated mutable state; message arguments are copied into the message | Shared mutable memory (requires mutexes) | Strict deep copying of message terms |

### Trade-Off 1: Lightweight Actors vs. OS Kernel Threads

- **Advantage:** Operating system threads incur substantial overhead: a default 1 MB stack reservation, page table allocations, and kernel-mode context switches taking 1,000–3,000 nanoseconds. A Vira actor takes about 370 bytes plus its fields, so 100,000 actors fit in roughly 40 MB. A message hop between actors measured 460–880 ns on a 2-core cloud VM.
- **Constraint:** Vira actors run cooperative work-stealing tasks across a fixed thread pool: one thread per logical core, clamped to 2–16 (override with `VIRA_WORKERS`, at most 32). A task that performs an infinite blocking system call (e.g. non-asynchronous file I/O) can occupy a worker thread until completion.

### Trade-Off 2: Isolated Per-Actor Bump Arena vs. Global Shared Heap

- **Advantage:** When an actor allocates intermediate buffers (such as SIMD tensor multiplications or string formatting), memory is allocated from its private bump arena. If the actor panics, resetting `offset = 0` reclaims 100% of intermediate allocations in **O(1) time** with zero memory fragmentation.
- **Constraint:** State that must outlive an actor or be shared with other actors is passed by value or managed via deterministic Autofree containers.

### Trade-Off 3: Lock-Free Chase-Lev Deques vs. Centralized MPMC Mailboxes

- **Advantage:** Traditional actor runtimes use a single centralized mutex-protected task queue or per-actor unbounded mutex queues. Under high concurrency, worker threads spend over 50% of CPU cycles spinning on lock contention. Vira's scheduler assigns an independent lock-free Chase-Lev deque (`VIRA_WS_CAP = 4,096` tasks) to each worker. The owning worker pushes and pops from the bottom without synchronization; peer threads steal from the top using atomic CAS (`__atomic_compare_exchange_n`).
- **Constraint:** When a worker's deque is full, further tasks go to a shared overflow queue. Nothing is dropped or blocked, but the overflow queue is slower than the deques.

### Trade-Off 4: `setjmp`/`longjmp` Panic Isolation vs. Process Abort

- **Advantage:** In standard C99/C++, an unhandled exception or assertion aborts the entire operating system process. Vira sets an explicit `fault_boundary` jump buffer before each actor task. When `panic(...)` is invoked:
  1. The worker thread is saved from termination.
  2. The actor's scratch arena is wiped in O(1).
  3. The supervisor increments `restarts` and triggers `Actor_supervisor_reset(&self)`, which resets the mutable fields and keeps the immutable ones.
  4. The caller's `fut.await()` receives a structured `Result_T{ .is_err = true, .err = msg }`.
- **Constraint:** Code executed inside an actor must not violate raw memory safety (e.g. raw unsafe pointer corruption from external C libraries) that would cause a kernel segmentation fault before `panic()` is reached.

### Trade-Off 5: Zero-Leak Autofree vs. Tracing Garbage Collection

- **Advantage:** Tracing garbage collectors (Java, Go, BEAM) suffer from unpredictable stop-the-world pauses and high memory amplification (2x–3x heap headroom). Vira's Autofree compiler engine statically tracks lifetimes and automatically hoists temporary strings and futures into deterministic destructors (`_vira_tmp_str_N`), guaranteeing `active_leaks = 0` at all times.
- **Constraint:** Cyclic reference graphs cannot be established without explicit weak references or container ownership patterns. Every value has exactly one owner; storing a value copies it unless its source is never read again ([Tutorial 42](./42_ownership_strings_and_actor_handles.md#3-ownership-copies-moves-and-borrowed-arguments)).

### Trade-Off 6: Work-Helping `await` vs. Blocking Waits

- **Advantage:** A thread that awaits a future runs other pending tasks instead of sleeping, so awaiting inside an actor does not waste a worker.
- **Constraint:** Actors that await each other in a cycle can never finish. Vira detects such cycles (and an actor awaiting itself) and returns `deadlock avoided: actors are waiting on each other` to one waiter; its message still runs later. Prefer fire-and-forget sends or replying through a third actor where a cycle is possible.

---

## 3. High-Concurrency Stress Benchmark Suite

The repository includes a dedicated stress benchmark in [`examples/actor_stress_benchmark.vir`](../../examples/actor_stress_benchmark.vir).

### Benchmark Results (`-prod` Mode)

Run the benchmark locally:
```bash
vira build -prod -o actor_bench examples/actor_stress_benchmark.vir
VIRA_WORKERS=4 ./actor_bench
```

#### Reference run (Linux x86_64 cloud VM, 2 cores, `VIRA_WORKERS=2`)

Timings on a shared 2-core VM vary by 20–70% between runs; treat them as orders of magnitude, and compare builds on your own machine.

```
--- BENCHMARK 1: High-Density Actor Spawn (10,000 Actors) ---
✓ Spawned 10000 lightweight actors in 501 µs
✓ Spawn Throughput: 10000000 actors/sec
✓ Memory Overhead: ~370 bytes/actor + fields (scratch arena allocated on first use)

--- BENCHMARK 2: Work-Stealing Message Passing (Fan-Out / Fan-In) ---
✓ Executed 40000 async messages in 22744 µs
✓ Message Throughput: 1739130 messages/sec across work-stealing pool

--- BENCHMARK 3: Chaos Fault-Tolerance & Supervisor Restarts ---
✓ Total Injected Faults: 715
✓ Supervisor Trapped Faults: 715 (100% recovery rate)
✓ Normal Tasks Completed: 4285
✓ Supervisor Fault-Isolation Latency: 4333 µs total
✓ 0 worker thread crashes, 0 process crashes, 0 memory corruption

--- BENCHMARK 4: Actor Ring Pipeline (10,000 Hops) ---
✓ Traversed 10000 ring hops in 8542 µs
✓ Ring Token Rate: 1111111 hops/sec
✓ Average Hop Latency: 854 ns/hop

--- BENCHMARK 5: Multicore Parallel Compute & Concurrency Proof ---
✓ Sequential Execution Time (one actor at a time): 43944 µs
✓ Parallel Execution Time (4 actors at once): 21464 µs
✓ Multicore Parallel Speedup: 2.0x

[vira_rt] pool_workers=2 | actor_tasks_run=55016 | work_steals=36890
[vira_rt] active_leaks=0 | total_freed=65047
```

With 2 workers the best possible speedup in benchmark 5 is 2x; on a 4-core machine with `VIRA_WORKERS=4` expect close to 4x.

### Repeatable before/after measurements

To compare two builds of the runtime, use the benchmark script. It builds [`scripts/bench/actor_baseline.vir`](../../scripts/bench/actor_baseline.vir), runs it repeatedly with a fixed worker count and reports the median, minimum, maximum and spread of five metrics (spawn, fan-out, fault recovery, ring, and a growing slice of actors), plus the allocation count:

```bash
bash scripts/bench_actors.sh                         # 15 runs, VIRA_WORKERS=4, ./vira
RUNS=25 WORKERS=8 VIRA=./vira-old bash scripts/bench_actors.sh
```

Results are appended to `bench/results.tsv`. Run both builds back to back on the same idle machine; a change smaller than the reported spread is noise.

---

## 4. Multi-Example Showcase

The documentation includes 4 complete, production-grade actor examples:

### Example 1: Resilient Banking & Account Supervisor
*File:* [`docs/examples/25_actors_banking_fault_tolerance.vir`](../examples/25_actors_banking_fault_tolerance.vir)

Demonstrates:
- Isolated bank accounts maintaining atomic balance state.
- Injected network timeout faults during withdrawal.
- Erlang-style supervisor recovery: account state safely rolled back to safe default without corrupting account ledger.
- Retrying transactions after supervisor healing.

```bash
vira run docs/examples/25_actors_banking_fault_tolerance.vir
```

### Example 2: Matrix Compute Worker Pool & Fan-Out / Fan-In
*File:* [`docs/examples/26_actors_worker_pool_fanout.vir`](../examples/26_actors_worker_pool_fanout.vir)

Demonstrates:
- Managing a pool of `MatrixWorker` actors.
- Asynchronously fanning out 8 SIMD AVX2 tensor multiplication tasks (`dim x dim`) across the Chase-Lev scheduler with 64×64 L2-cache blocking.
- Work-helping `.await()` synchronizing and aggregating the matrix sum.
- 0 memory leaks across all temporary tensors and futures.

```bash
vira run docs/examples/26_actors_worker_pool_fanout.vir
```

### Example 3: Pipeline & Sequential Token Ring
*File:* [`docs/examples/27_actors_ring_pipeline.vir`](../examples/27_actors_ring_pipeline.vir)

Demonstrates:
- Chained actor stages (`Ingress -> Validation -> Enrichment -> Egress`).
- Token hand-off between stages with sequential transformation (about 1M hops/sec on a 2-core VM).
- Interleaved concurrent token streams processing without cross-talk or race conditions.

```bash
vira run docs/examples/27_actors_ring_pipeline.vir
```

### Example 4: Detached Fire-and-Forget Sends & Lifecycle Management
*File:* [`docs/examples/32_actor_detached_and_lifecycle.vir`](../examples/32_actor_detached_and_lifecycle.vir)

Demonstrates:
- Detached fire-and-forget message passing (`actor <- method()` and `_ := actor <- method()`) with zero future overhead and automatic worker envelope reclamation.
- Synchronous calls (`actor <- method() or { ... }`), which use a stack envelope when called from non-actor code.
- Actor state transition to `VIRA_ACTOR_STOPPED` when consecutive restarts exceed `max_restarts`.
- Automatic reset of consecutive `restarts = 0` on healthy message execution.
- Dead-letter rejection preventing message delivery to stopped actors.

```bash
vira run docs/examples/32_actor_detached_and_lifecycle.vir
```

---

## 5. Summary & Best Practices

1. **Keep Actor Messages Focused:** Pass data as message arguments (they are copied into the message). Let the actor own and mutate its internal state, and read it from outside through a method, not a field.
2. **Embrace "Let It Crash":** Instead of defensive NULL checks or deeply nested error flags, use `panic(...)` when an internal invariant is violated. The supervisor will trap the fault, wipe scratch memory in O(1), and restart the actor cleanly.
3. **Use Non-Blocking Fan-Out:** Dispatch concurrent actor messages in bursts with `<-`, and await them in a batch. The runtime's work-helping mechanism will steal and execute tasks while awaiting.
4. **Use Detached Sends for Telemetry & Ingestion:** Discarding or ignoring return values with `w <- log(...)` or `_ := w <- log(...)` automatically bypasses future allocation and frees task envelopes asynchronously.
5. **Use Synchronous Calls for Immediate Results:** `res := w <- compute(...) or { fallback }` waits for the result. From `main` it needs no heap allocation; from inside another actor it goes through a future so a wait cycle can be broken.
6. **Rely on Autofree:** Temporary interpolated strings passed to calls (`log.info('...')`, `println('...')`) are automatically reclaimed upon statement exit with zero leaks (`active_leaks=0`).
