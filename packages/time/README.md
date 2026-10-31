# `packages/time`: Wall-Clock, High-Resolution Timers & Durations

The `time` package provides nanosecond wall-clock timestamps, monotonic hardware clocks, microsecond sleep, and `strftime` formatting in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Current Time**:
  - `time.now_unix() i64`: Unix epoch in seconds.
  - `time.now_ms() i64`: Unix epoch in milliseconds.
  - `time.now_ns() i64`: Unix epoch in nanoseconds.
  - `time.monotonic_ns() i64`: Monotonic clock for benchmarking with no leap-second jumps.
- **High-Precision Sleeping**: `time.sleep_ms(ms int)`, `time.sleep_us(us int)`.
- **String Formatting**: `time.format(fmt string, unix_sec i64, utc bool) !string`.
- **`Duration`**: Strongly-typed duration abstractions (`from_millis`, `from_secs`, `millis()`, `secs()`).
- **Constants**: `NANOSECOND`, `MICROSECOND`, `MILLISECOND`, `SECOND`, `MINUTE`, `HOUR`.

---

## 🛠️ Usage Example

```v
module main

import time

fn main() {
    // 1. High-precision latency benchmarking
    start := time.monotonic_ns()
    time.sleep_ms(15)
    elapsed_ns := time.monotonic_ns() - start
    println('Elapsed: ${elapsed_ns / 1000000} ms')

    // 2. Formatting current epoch to ISO-like local datetime
    now := time.now_unix()
    formatted := time.format('%Y-%m-%d %H:%M:%S', now, false) or { 'unknown' }
    println('Formatted time: ${formatted}')

    // 3. Durations
    d := time.Duration.from_secs(5i64)
    println('Duration in millis: ${d.millis()} ms')
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/time
```
