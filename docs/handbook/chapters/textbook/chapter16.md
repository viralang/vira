# Chapter 16: Networking, HTTP Servers & Pure-Vira JSON

Vira ships a pure-Vira networking and web stack with zero external dependencies: `net` for TCP, UDP, and OS-level `poll` multiplexing; `json` for RFC 8259 validation, extraction, and linear-time JSON construction; `http` for hardened HTTP/1.1 parsing, non-blocking connection framing, and Server-Sent Events (SSE); and `viraserve` for actor-powered HTTP microservices.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch16_networking_http_and_json.vir`](../code/ch16_networking_http_and_json.vir)

---

## 16.1. TCP Listeners, Streams, UDP Sockets, and Non-Blocking Polling (`import net`)

The `net` package provides four core primitives:

1. **`net.SocketAddr`:** Parses IPv4 and bracketed IPv6 `host:port` strings (`SocketAddr.parse('127.0.0.1:0')`) with range checks (`0..65535`). Binding port `0` asks the OS kernel for an ephemeral loopback port, which you inspect afterward via `.local_addr()`.
2. **`net.TcpListener` & `net.TcpStream`:**
   - Server side: `TcpListener.bind(addr) !TcpListener`, `.accept() !TcpStream`, `.set_nonblocking(bool)`, `.local_addr() !SocketAddr`, `.close()`.
   - Client/connection side: `TcpStream.connect(addr)` or `TcpStream.connect_timeout(addr, timeout_ms)`, `.read`, `.read_byte`, `.read_line`, `.read_line_limit`, `.read_exact(n)`, `.read_all`, `.read_string(max_len)`, `.write`, `.write_string` / `.write_str`, `.set_nodelay`, `.set_nonblocking`, `.set_timeout(read_ms, write_ms)`, `.peer_addr`, `.shutdown`, and `.close`.
   - Non-blocking buffer helpers: `.read_append(mut dst, max_len)` and `.write_at(buf, off, len)` operate directly on `[]u8` slices without extra copies.
3. **`net.UdpSocket`:** Connectionless datagrams via `UdpSocket.bind(addr)`, `.send_to(buf, dest)`, `.send_to_string(msg, dest)`, and `.recv_from(mut buf) !UdpPacket` (returning `len` and `sender SocketAddr`). Note that `recv_from` reads up to `buf.cap`, so allocate receive buffers with `net.new_bytes(cap)`.
4. **`net.Poller` & `net.poll_many`:**
   - Single-socket readiness checks with `Poller.new(timeout_ms)` (`poll_one`, `wait_readable`, `wait_writable`).
   - Multi-socket event loops with `net.poll_many(fds []i64, want []int, timeout_ms int) ![]int` using bitmasks `POLL_READ` (`1`), `POLL_WRITE` (`2`), and `POLL_ERROR` (`4`).

```v
mut listener := net.TcpListener.bind('127.0.0.1:0') or { panic('bind failed') }
l_addr := listener.local_addr() or { panic('local_addr failed') }

mut client := net.TcpStream.connect(l_addr.to_string()) or { panic('connect failed') }
client.set_nodelay(true) or { false }
client.write_str('PING\n') or { panic('write failed') }

poller := net.Poller.new(500)
ready := poller.wait_readable(listener.fd) or { false }
mut accepted := listener.accept() or { panic('accept failed') }
```

---

## 16.2. Fast RFC 8259 JSON Parsing and $O(N)$ `JsonBuilder` (`import json`)

The `json` package operates directly on UTF-8 byte slices without building intermediate AST trees:

- **Validation & Escaping:** `json.is_valid(raw)` verifies strict RFC 8259 grammar (disallowing trailing commas, unescaped control bytes, and invalid numbers). `json.escape` and `json.unescape` handle `\"`, `\\`, `\n`, `\r`, `\t`, `\b`, `\f`, and `\uXXXX` UTF-8 decoding.
- **Depth-Aware Field Extraction:** `json.get_raw(raw, key)`, `json.has_key(raw, key)`, `json.get_str(raw, key)`, `json.get_int(raw, key)`, `json.get_i64(raw, key)`, and `json.get_bool(raw, key)` skip nested objects and arrays depth-aware so inner keys never shadow outer keys.
- **Array Accessors:** `json.array_len(raw) !int`, `json.array_get_raw(raw, idx) !string`, and `json.array_get_str(raw, idx) !string`.
- **$O(N)$ `JsonBuilder`:** Instead of concatenating strings quadratically (`s += ...`), `JsonBuilder` appends bytes into a single `[]u8` buffer and wraps it with `{` and `}` on `.build()`:

```v
mut jb := json.JsonBuilder.new()
jb.set_str('service', 'vira-api')
jb.set_int('port', 8080)
jb.set_bool('healthy', true)
jb.set_raw('tags', '["actors","simd","json"]')
payload := jb.build()
```

---

## 16.3. HTTP/1.1 Clients, Servers, and Server-Sent Events (`import http`)

The `http` package layers structured HTTP/1.1 types on top of `net`:

### Headers, Requests, Responses & Blocking `HttpServer`

- **`http.HeaderMap`:** Case-insensitive header lookup and formatting (`set`, `get`, `get_or`, `contains`, `format`).
- **`http.Response`:** Constructors `Response.new()`, `Response.ok(body, ctype)`, `Response.json(json)`, `Response.text(text)`, `Response.with_status(code, body)`, `Response.bad_request`, `Response.not_found`, and `Response.internal_error`. Binary bodies containing `\0` bytes preserve their exact length on the wire.
- **`http.HttpServer` & `http.RequestContext`:** `HttpServer.bind(addr)` binds a listener with bounded defaults (`max_body_bytes = DEFAULT_MAX_BODY_BYTES` [64 MiB], `head_deadline_ms = 30000`, `MAX_HEADER_LINE = 8192`, `MAX_HEADER_COUNT = 100`). `accept_request()` enforces 64-bit `Content-Length` parsing and answers oversized bodies (`413`), invalid lengths (`400`), oversized/excess headers (`431`), and slow request heads (`408`) automatically before returning `RequestContext` (`req` + `stream`) with `.respond`, `.respond_json`, `.respond_text`, and `.sse()`.
- **HTTP Client Connections:** Clients connect with `net.TcpStream.connect(...)`, write an HTTP/1.1 request (`write_str(...)`), and read the status line, headers, and body (`read_line` / `read_exact` / `read_all`).

### Non-Blocking `http.Conn` Framing

For concurrent single-threaded event loops driven by `net.poll_many`, `http.Conn.new(stream, max_body)` switches the socket to non-blocking mode and provides `fill()`, `next_request() Incoming`, `queue(s)` / `queue_response(res, close)`, `flush() bool`, and `done() bool`—so one slow peer never stalls other connections.

### Server-Sent Events (`SseWriter`: `start` vs. `start_cors` & FD Ownership Transfer)

`http.SseWriter` streams `text/event-stream` responses (`send_event`, `send_data`, `send_token`, `send_done`, `close`) with two important safety guarantees:

1. **Opt-In CORS:** `SseWriter.start(mut stream)` (and `ctx.sse()`) emits **no** `Access-Control-Allow-Origin` header by default. When cross-origin streaming is explicitly required, call `SseWriter.start_cors(mut stream, allow_origin)` (which validates that `allow_origin` contains no CR/LF injection).
2. **Socket FD Ownership Transfer:** When `SseWriter.start` or `start_cors` succeeds, it moves the underlying file descriptor into `SseWriter.stream` and invalidates the caller's stream (`stream.fd = i64(-1); stream.is_open = false`). Calling `ctx.stream.close()` afterward becomes a safe no-op and can never double-close the live SSE socket.

---

## 16.4. Building Actor-Powered Web Services (`import viraserve`)

The `viraserve` microframework connects `http.HttpServer` to an isolated `RequestWorker` actor:

- **`viraserve.ViraServe.new(name, port)`:** Holds a `Router`, `ServerStats` (request count, error count, average latency in microseconds), and a `RequestWorker` actor.
- **Actor Isolation + SIMD Inference:** `dispatch_request(req)` matches routes (`RouteID.home()`, `.health()`, `.metrics()`, `.predict()`, `.users()`, `.echo()`) and sends the route into the worker mailbox (`s.worker <- execute_route(...)`). The `/health` route runs inside a fast `arena { ... }` block, while `POST /api/v1/predict` executes a native 16×16 SIMD tensor matrix multiplication (`w_mat @ x_mat`) inside the supervised worker actor.
- **Serving Loops:** Use `app.serve_one(mut http_srv)` to handle a single request cleanly (ideal for tests and deterministic demos), `app.serve(mut http_srv)` to loop on a bound `HttpServer`, or `app.listen_and_serve()` to bind `127.0.0.1:<port>` and serve continuously.

---

## 16.5. Running the Complete Chapter 16 Example

Run [`docs/handbook/chapters/code/ch16_networking_http_and_json.vir`](../code/ch16_networking_http_and_json.vir):

```bash
./vira run docs/handbook/chapters/code/ch16_networking_http_and_json.vir
```

Output:
```text
=== [1] TCP, UDP & Non-Blocking Polling (import net) ===
TCP loopback: listener_readable=true, req="PING", resp="PONG"
UDP loopback: received 9 bytes: "hello-udp"

=== [2] RFC 8259 JSON Parsing & JsonBuilder (import json) ===
Built JSON: {"service":"vira-api","port":8080,"healthy":true,"tags":["actors","simd","json"]}
Parsed: valid=true, service=vira-api, port=8080, healthy=true, tags=3(first="actors")

=== [3] HTTP/1.1 Server, Non-Blocking Conn & SSE (import http) ===
HTTP roundtrip: HTTP/1.1 200 OK, body={"received":{"ok":123}}
SSE: ownership_transferred=true, cors=true, ev1="data: {"token":"Hi"}", ev2="data: [DONE]"
Non-blocking Conn: ready=true, path=/ping, resp="HTTP/1.1 200 OK"

=== [4] Actor-Powered Web Service (import viraserve) ===
[viraserve] POST /api/v1/predict -> 200 OK (258 µs)
ViraServe /api/v1/predict response received (165 bytes):
HTTP/1.1 200 OK
Content-Length: 75
Connection: close
Content-Type: application/json

{"model":"vira-mini","status":"inferred","energy_score":4096.00,"worker":1}
--- VIRASERVE TELEMETRY METRICS ---
• Total Requests: 1
• Total Errors:   0
• Avg Latency:    258 µs
• Memory Safety:  0 GC Pauses | Deterministic Autofree
-----------------------------------
[vira_rt] simd=x86_64 SSSE3 + SSE2 (128-bit Integer/Float SIMD, Multi-Threaded)
[vira_rt] pool_workers=4 | actor_tasks_run=1 | work_steals=1
[vira_rt] active_leaks=0 | total_freed=408
```

---

## Summary & Exercises

1. **Multi-Client `net.poll_many` Loop:** Extend `demo_http_and_sse` to accept two simultaneous client sockets, wrap both in `http.Conn`, pass `[c1.fd(), c2.fd()]` to `net.poll_many`, and serve both requests in one event-loop pass.
2. **HTTP Request Limits:** Set `srv.max_body_bytes = 16` on a bound `http.HttpServer`, send a 32-byte `POST` payload from a client `TcpStream`, and assert that the server returns `HTTP/1.1 413 Payload Too Large` while keeping `active_leaks=0`.
3. **JSON Escape Roundtrip:** Use `json.JsonBuilder` to encode a string containing quotes, backslashes, and newlines (`'a"b\\c\nd'`), validate it with `json.is_valid`, and verify `json.get_str` restores the exact original text.
