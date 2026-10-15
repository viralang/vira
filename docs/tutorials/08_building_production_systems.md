# Tutorial 8: Building Production AI Pipelines

> **Difficulty:** Advanced  
> **Matching Code Example:** [`docs/examples/08_production_pipeline.vir`](../examples/08_production_pipeline.vir)

---

## 1. Overview

In this capstone tutorial, you will assemble all of Vex's core features into a unified, production-ready AI batch inference pipeline:
- Struct embedding for telemetry and audit tracking
- Structural interfaces for polymorphic job reporting
- Multi-threaded actors for parallel model inference
- Result error handling and graceful fallbacks
- Zero memory leak auditing

---

## 2. Architecture of the Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    InferenceJob (OOP)                       │
│   ├── MetricHeader (Embedded: timestamp, ops_count)         │
│   ├── job_id, batch_size, status                            │
│   └── implements Auditable interface                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Parallel Multi-Actor Fan-Out                  │
│                                                             │
│   ┌───────────────────────┐       ┌───────────────────────┐ │
│   │ InferenceEngine 1 (<-)│       │ InferenceEngine 2 (<-)│ │
│   │   [weights @ inputs]  │       │   [weights @ inputs]  │ │
│   └───────────┬───────────┘       └───────────┬───────────┘ │
│               │                               │             │
│               ▼                               ▼             │
│         Future 1.await()                Future 2.await()    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Zero-Leak Verification                    │
│   Autofree cleans tensors, strings & actors in LIFO order   │
│   Telemetry confirms: active_leaks=0                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Core Implementation Highlights

### 1. Embedded Telemetry
```vex
pub struct MetricHeader {
pub:
	timestamp i64
pub mut:
	ops_count int
}

pub struct InferenceJob {
	MetricHeader // Promoted fields & methods!
pub:
	job_id string
	batch_size int
pub mut:
	status string
}
```

### 2. Interface Boxing
```vex
pub interface Auditable {
	job_summary() string
}

pub fn (j InferenceJob) job_summary() string {
	return 'Job [${j.job_id}]: status=${j.status}, batch=${j.batch_size}'
}

// Boxing without explicit implements:
audit := Auditable(job1)
println(audit.job_summary())
```

### 3. Actor SIMD Forward Pass
```vex
pub actor InferenceEngine {
pub:
	model_id string
mut:
	inferences_served int = 0
}

pub fn (mut e InferenceEngine) predict(scale f32) !f32 {
	if scale <= 0.0 {
		return error('invalid inference scale: must be positive')
	}
	e.inferences_served = e.inferences_served + 1

	weights := tensor_fill(32, 32, 0.25)
	inputs := tensor_fill(32, 32, scale)
	outputs := weights @ inputs

	return tensor_sum(outputs)
}
```

---

## 4. Production Compilation & Optimization

When preparing Vira applications for production deployment, compile with the `-prod` flag:

```powershell
.\vira.exe build -prod -o pipeline.exe docs/examples/08_production_pipeline.vir
```

### What `-prod` Activates Under the Hood:
- `-O3`: Aggressive loop unrolling and function inlining
- `-march=native`: Unlocks full AVX2 / FMA / AVX-512 hardware vector instructions
- `-flto`: Link-Time Optimization across runtime and generated code
- Strips debug symbols for minimal binary size

### Testing for Memory Safety (`-sanitize`):
To run automated sanitizers during CI/CD:
```powershell
.\vira.exe build -sanitize -o pipeline_sanitized.exe docs/examples/08_production_pipeline.vir
```
This enables LLVM/GCC AddressSanitizer (`ASan`) and UndefinedBehaviorSanitizer (`UBSan`) to verify zero heap corruption.

---

## 5. Running the Pipeline

```powershell
.\vira.exe run docs/examples/08_production_pipeline.vir
```

Expected output:
```text
=== 1. PREPARING PIPELINE JOBS & TELEMETRY ===
Job [job_prod_001]: status=PENDING, batch=32, ops=1
=== 2. PARALLEL MULTI-ACTOR INFERENCE FAN-OUT ===
=== 3. AWAITING RESULTS & FINALIZING STATUS ===
Inference #1 Output Sum: 12288.00
Inference #2 Output Sum: 16384.00
=== 4. FINAL PIPELINE AUDIT REPORT ===
Job [job_prod_001]: status=COMPLETED, batch=32, ops=1
Job [job_prod_002]: status=COMPLETED, batch=64, ops=1
[vira_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vira_rt] pool_workers=4 | actor_tasks_run=2 | work_steals=2
[vira_rt] active_leaks=0 | total_freed=10
```

---

## Summary

You now have a complete mastery of the Vira programming language! You can build high-performance systems, machine learning microservices, and concurrent backends with fearless concurrency and zero garbage-collector overhead.
