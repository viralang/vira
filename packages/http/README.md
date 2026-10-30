# `packages/http`: HTTP/1.1 Request/Response Engine & SSE

The `http` package provides HTTP/1.1 request parsing, response formatting, header maps, connection loops, and Server-Sent Events (SSE) in the **Vira Programming Language**.

---

## 🚀 Key Features

- **Request Parsing**: `http.Request.parse(raw string) !Request` (`method`, `path`, `headers`, `body`).
- **Response Building**: `http.Response.new(status int)` with fluent `.header(k, v)`, `.body(text)`, `.to_string()`.
- **Headers Management**: Case-insensitive `http.HeaderMap` with lookup and formatting.
- **Server-Sent Events**: `http.SseStream` for real-time streaming connections.
- **Micro-Server Loop**: `http.HttpServer.bind(addr string)`.

---

## 🛠️ Usage Example

```v
module main

import http

fn main() {
    // 1. Parse raw incoming HTTP request
    raw := 'GET /api/v1/health HTTP/1.1\r\nHost: localhost\r\n\r\n'
    req := http.Request.parse(raw) or { panic(err) }
    println('Method: ${req.method}, Path: ${req.path}')

    // 2. Build HTTP response
    mut res := http.Response.new(200)
    res.header('Content-Type', 'application/json')
    res.body('{"status":"ok"}')

    println('Response string:\n${res.to_string()}')
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/http
```
