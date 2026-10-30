# `packages/log`: Leveled Structured Stderr Logger

The `log` package formats leveled diagnostic messages to standard error (`stderr`) with timestamps, custom prefixes, and process termination in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Standard Severity Levels**:
  - `log.LEVEL_DEBUG` (0)
  - `log.LEVEL_INFO` (1)
  - `log.LEVEL_WARN` (2)
  - `log.LEVEL_ERROR` (3)
  - `log.LEVEL_FATAL` (4) — Prints error and terminates process with exit code 1.
- **Convenience Functions**: `log.debug(msg)`, `log.info(msg)`, `log.warn(msg)`, `log.error(msg)`, `log.fatal(msg)`.
- **Custom Logger Instances**: `log.Logger.new(level)` with `.set_prefix(prefix)`, `.set_timestamp(bool)`.

---

## 🛠️ Usage Example

```v
module main

import log

fn main() {
    // 1. Package-level quick logging
    log.info('System boot sequence initiated')
    log.warn('Disk storage above 85% utilization')

    // 2. Custom logger instance with component prefix
    mut logger := log.Logger.new(log.LEVEL_DEBUG)
    logger.set_prefix('worker-pool')
    logger.debug('Spawned 8 Chase-Lev worker threads')
    logger.info('Work-stealing pool active')

    // 3. Fatal logging on unrecoverable failures
    // log.fatal('Unrecoverable database corruption') // Prints to stderr and exits with code 1
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/log
```
