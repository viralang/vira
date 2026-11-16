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

## ⚡ Non-blocking connections (event loops)

`Request.parse` and `HttpServer.accept_request` block until a whole request has arrived, so one slow client holds up a single-threaded server. For servers that must stay responsive, use `http.Conn` with `net.poll_many`:

- `http.Conn.new(stream, max_body)`: makes the socket non-blocking.
- `fill()`: reads what has arrived (never waits; up to 4 MiB per call). `closed` is set when the peer is gone.
- `next_request()`: returns `Incoming{ ready, req }` once the head and the whole body are in. Refuses with the status already queued: `413` (body over `max_body`, default 64 MiB), `431` (head over 64 KiB), `400` (bad head or `Content-Length`), `501` (`Transfer-Encoding` bodies).
- `queue(s)`, `queue_bytes(b)`, `queue_response(res, close)`: buffer output (NUL bytes kept). `flush()` writes what the socket takes now and returns `true` when nothing is left; partial writes resume on the next call.
- `pending_out()`, `done()`, `close()`; `sse_head()` and `sse_frame(data)` for streaming.

```v
module main

import net
import http

// Minimal non-blocking server: one poll over all sockets, never waits on a client.
fn main() {
	mut listener := net.TcpListener.bind('127.0.0.1:8099') or { panic(err) }
	listener.set_nonblocking(true) or { false }
	mut conns := []http.Conn{}
	mut served := 0
	for served < 1 {
		mut fds := [listener.fd]
		mut want := [net.POLL_READ]
		for c in conns {
			fds.push(c.fd())
			mut w := net.POLL_READ
			if c.pending_out() > 0 {
				w = w | net.POLL_WRITE
			}
			want.push(w)
		}
		net.poll_many(fds, want, 500) or { []int{} }
		for {
			s := listener.accept() or { break }
			conns.push(http.Conn.new(s, 0))
		}
		mut i := 0
		for i < conns.len {
			conns[i].fill()
			inc := conns[i].next_request()
			if inc.ready {
				conns[i].queue_response(http.Response.text('hello ${inc.req.path}\n'), true)
				served += 1
			}
			conns[i].flush()
			i += 1
		}
	}
	mut i := 0
	for i < conns.len {
		for !conns[i].flush() {}
		conns[i].close()
		i += 1
	}
	listener.close() or { false }
}
```

`net` adds the pieces this needs: `poll_many(fds, want, timeout_ms)` (one `poll()` for many sockets), `TcpStream.read_append` / `write_at` (non-blocking, no per-byte copies), `append_string`, `append_bytes`, `find_bytes`. `TcpStream.write` now finishes partial sends instead of dropping the rest.

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/http
```
