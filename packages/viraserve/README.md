# ViraServe: Asynchronous HTTP 1.1 Microframework for Vira

`viraserve` is a production-grade, asynchronous HTTP/1.1 web microframework and routing engine written specifically for the **Vira Programming Language**.

It combines **non-blocking actor connection dispatch**, **zero-copy JSON field parsing**, **declarative routing**, and **deterministic memory reclamation (zero GC pauses, zero leaks)**.

---

## 🚀 Key Features

* **Actor-Powered Concurrency:** Every incoming request is handled across Vira's multicore Chase-Lev work-stealing pool via isolated actor workers (`RequestWorker`).
* **Zero GC Latency Pauses:** Static compile-time Autofree ensures request buffers and response payloads are cleaned up deterministically upon statement completion (`active_leaks=0`).
* **Built-in Native AI Serving:** First-class support for executing 64-byte aligned SIMD tensor inference directly inside route handlers.
* **Declarative Route Dispatch:** High-speed route matching for standard REST methods (`GET`, `POST`, `PUT`, `DELETE`).
* **JSON Context Abstraction:** Fluent `Request` and `Response` APIs with direct JSON field extraction (`json_str`, `json_int`, `json_f32`).
* **Built-in Telemetry:** Sub-microsecond latency recording (`time_now_us()`) and structured access logging.

---

## 📦 Installation

Add `viraserve` to your Vira project using the native package manager:

```bash
# Add as a dependency
vira add github.com/viralang/viraserve

# Or vendor for 100% air-gapped offline builds
vira vendor
```

---

## 🛠️ Quickstart Example

```v
module main

import viraserve

fn main() {
    // 1. Initialize server
    mut app := viraserve.ViraServe.new('AICloudService', 8080)
    app.print_banner()

    // 2. Dispatch an incoming request
    req := viraserve.Request.new('GET', '/health', '', '', 10)
    res := app.dispatch_request(req)

    println('Response: ${res.body}')
    println('Status:   ${res.status_code}')

    // 3. Print runtime telemetry metrics
    app.print_metrics()
}
```

---

## 🧩 Architectural Layout

```
packages/viraserve/
├── vira.mod              # Package manifest
├── README.md             # Documentation & Guide
├── src/
│   ├── context.vir       # Request & Response structs & JSON helpers
│   ├── router.vir        # Declarative route matching & route registry
│   ├── middleware.vir    # Server telemetry & structured access logging
│   ├── worker.vir        # Concurrent RequestWorker actor
│   └── server.vir        # Master ViraServe application orchestrator
├── examples/
│   └── api_server.vir    # Full multi-route REST API demo
└── tests/
    └── server_test.vir   # Automated test suite (asserts 0 leaks)
```

---

## 📊 Endpoints Supported by Default Router

| Method | Path | Description | Example Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Service discovery & root info | None |
| `GET` | `/health` | Cluster health & worker status | None |
| `GET` | `/metrics` | Prometheus-style runtime metrics | None |
| `POST` | `/api/v1/predict` | Native SIMD tensor inference pass | `{"model":"deepseek","scale":2.5}` |
| `GET` | `/api/users` | List active users | None |
| `POST` | `/api/users` | Create new user entity | `{"username":"alex","role":"admin"}` |
| `POST` | `/api/echo` | Payload reflection & debug | Any valid JSON |

---

## 🧪 Testing & Verification

Run the comprehensive unit test suite:

```bash
vira run tests/server_test.vir
```

All tests execute with **`active_leaks=0`** and full actor supervisor containment.

---

## 📄 License

Licensed under the [MIT License](../../LICENSE.txt).
