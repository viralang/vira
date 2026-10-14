# Tutorial 9: File I/O, Buffer Operations & Streams

> **Difficulty:** Advanced  
> **Matching Code Example:** [`docs/examples/09_file_io.vex`](../examples/09_file_io.vex)

---

## 1. Overview

Input/Output (I/O) is a critical pillar of systems programming, data engineering, and machine learning pipelines. 

In Vex, I/O operations are designed around three principles:
1. **Safety & Zero-Cost Result Handling:** All mutating or fallible operations return `!bool` or `!string`. I/O failures cannot be ignored.
2. **Automatic Buffer Reclaim:** File reading allocates native heap buffers that are tracked and reclaimed deterministically by the **Autofree engine** without manual `free()` calls and without garbage collection.
3. **Concurrency Compatibility:** File buffers can be seamlessly dispatched across multi-threaded **Actors** over lock-free work-stealing deques.

---

## 2. File I/O Primitives

Vex provides native, hardware-accelerated file primitives:

| Function Signature | Description | Return Type |
| :--- | :--- | :--- |
| `file_exists(path string) bool` | Checks if a file exists on the filesystem | `bool` |
| `read_file(path string) !string` | Reads entire file into a heap-allocated UTF-8 string | `!string` (Result) |
| `write_file(path string, content string) !bool` | Writes content to file, overwriting if existing | `!bool` (Result) |
| `append_file(path string, content string) !bool` | Appends content to the end of a file | `!bool` (Result) |
| `remove_file(path string) !bool` | Deletes a file from the filesystem | `!bool` (Result) |

---

## 3. Writing and Appending Files

### Writing Files (`write_file`)

`write_file` opens or truncates a file in binary mode (`"wb"`), writing the buffer atomically:

```vex
test_file := 'config.json'
content := '{"server": "vex.internal", "workers": 4}\n'

_ := write_file(test_file, content) or {
	eprintln('Failed to write file: ${err}')
	return
}
```

### Appending to Files (`append_file`)

`append_file` opens a file in append mode (`"ab"`), preserving existing content:

```vex
log_line := '[AUDIT] Job dispatched at timestamp 1760000000\n'

_ := append_file(test_file, log_line) or {
	eprintln('Failed to append log: ${err}')
	return
}
```

---

## 4. Reading Files & Zero-Leak Buffer Autofree

When you read a file with `read_file(path)`, the Vex runtime:
1. Opens the file descriptor in read-binary mode (`"rb"`).
2. Seeks to the end to determine the exact file length in bytes.
3. Allocates a memory buffer of size `len + 1` and reads the file in a single block.
4. Records the allocation in the telemetry counters (`_vex_record_alloc()`).
5. Returns a `Result_string` containing the unescaped UTF-8 string.

```vex
data := read_file('config.json') or {
	eprintln('Failed to read config: ${err}')
	return
}
println('Read Configuration File:')
println(data)
```

### How Autofree Reclaims the Buffer
Because `data` is bound as a local variable of type `string`, the **Static Escape Analyzer** tracks it. When `data` goes out of scope at the end of the enclosing block, the compiler automatically injects:

```c
vex_str_free(&data);
```

This guarantees that file buffers never leak memory, even when reading gigabytes of data in nested loops.

---

## 5. Deleting Files (`remove_file`)

To delete a file, invoke `remove_file(path)`:

```vex
_ := remove_file('config.json') or {
	eprintln('Could not delete file: ${err}')
	return
}

exists := file_exists('config.json')
println('File exists after deletion: ${exists}') // prints false
```

---

## 6. Advanced Pattern: Concurrent I/O + Actor Processing

You can combine File I/O with Vex's **Actor Concurrency** to build parallel log or dataset processing pipelines:

```vex
module log_processor

pub actor LogWorker {
pub:
	worker_id int
mut:
	lines_processed int = 0
}

pub fn (mut w LogWorker) process_chunk(path string) int {
	data := read_file(path) or {
		return 0
	}
	w.lines_processed = w.lines_processed + data.len
	return data.len
}

fn main() {
	mut worker := LogWorker{ worker_id: 1, jobs_completed: 0 }
	fut := worker <- process_chunk('dataset.csv')
	processed := fut.await() or { 0 }
	println('Processed ${processed} bytes across worker thread')
}
```

---

## 7. Running the Code

Execute the matching example from the Vex CLI:

```powershell
.\vex.exe run docs/examples/09_file_io.vex
```

Expected output:
```text
=== 1. CHECK FILE EXISTENCE ===
File sample_output.txt exists initially: false
=== 2. ATOMIC FILE WRITE ===
Successfully wrote initial content to sample_output.txt
=== 3. APPENDING TO FILE ===
Successfully appended log record to sample_output.txt
=== 4. READING FILE CONTENTS ===
--- Read File Content ---
Vex Language: Fast, Safe, Concurrent, and GC-Free
[LOG] Tensor pipeline initialized at timestamp 1760000000
=== 5. CLEANUP / REMOVE FILE ===
File sample_output.txt deleted; exists now: false
[vex_rt] simd=x86_64 AVX2 + FMA (256-bit / 8x f32 per cycle)
[vex_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vex_rt] active_leaks=0 | total_freed=5
```

---

## Next Steps

Review the [**Vex Master Curriculum Index**](./README.md) or explore the [**Vex Runnable Examples Catalog**](../examples/README.md) to continue expanding your Vex applications.
