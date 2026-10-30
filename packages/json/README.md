# `packages/json`: Pure-Vira RFC 8259 JSON Parser & Builder

The `json` package provides a 100% pure-Vira RFC 8259 compliant JSON validator, zero-copy field extractor, array parser, and $O(N)$ programmatic serializer (`JsonBuilder`) in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Validation**: `json.is_valid(payload string) bool` — strict RFC 8259 conformity test.
- **Field Extraction**:
  - `json.get_str(payload, key) !string`
  - `json.get_int(payload, key) !int`
  - `json.get_i64(payload, key) !i64`
  - `json.get_bool(payload, key) !bool`
  - `json.get_raw(payload, key) !string` (extracts raw JSON subtree or numeric literal)
  - `json.array_get_str(payload, key) ![]string`
- **Programmatic Serialization**: `json.JsonBuilder`
  - `jb.set_str(k, v)`
  - `jb.set_int(k, v)`
  - `jb.set_i64(k, v)`
  - `jb.set_bool(k, v)`
  - `jb.set_raw(k, raw_json)`
  - `jb.build() string`
- **Zero Memory Leaks**: Full verification under Autofree escape analysis (`active_leaks=0`).

---

## 🛠️ Usage Example

```v
module main

import json

fn main() {
    payload := '{"service":"inference","port":8080,"active":true,"workers":["w1","w2"]}'

    // 1. Validate compliance
    println('Is valid JSON: ${json.is_valid(payload)}')

    // 2. Extract fields
    svc := json.get_str(payload, 'service') or { 'unknown' }
    port := json.get_int(payload, 'port') or { 0 }
    active := json.get_bool(payload, 'active') or { false }
    println('Service: ${svc}, Port: ${port}, Active: ${active}')

    // 3. Extract array elements
    workers := json.array_get_str(payload, 'workers') or { []string{} }
    println('Workers count: ${workers.len}')

    // 4. Construct JSON with JsonBuilder
    mut jb := json.JsonBuilder.new()
    jb.set_str('status', 'healthy')
    jb.set_int('uptime_sec', 3600)
    jb.set_bool('ready', true)
    println('Constructed JSON:\n${jb.build()}')
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/json
```
