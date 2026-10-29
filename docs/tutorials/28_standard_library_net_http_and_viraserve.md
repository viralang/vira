# Tutorial 28: Standard Library Networking, HTTP, JSON, & ViraServe

In this tutorial, you will learn how to use Vira's native standard library packages—`io`, `net`, `http`, and `json`—and build high-performance microservices with `viraserve`.

---

## 1. Standard Library Architecture

Vira standard library packages live directly under `packages/` and are resolved automatically by the compiler (`ResolveImportCandidates` tier 0) without requiring external package downloads:

```
packages/
├── io/          # Stream abstractions, File handles, BufReader/BufWriter
├── net/         # TCP Listener/Stream, UDP Socket, Poller, DNS resolver
├── http/        # HTTP/1.1 request parser, response builder, header maps, SSE writer
├── json/        # Pure-Vira RFC 8259 JSON validator, depth-aware parser & O(N) JsonBuilder
└── viraserve/   # Actor-driven HTTP server microframework with worker pools
```

---

## 2. Low-Level I/O (`packages/io`)

The `io` package provides unified structural traits:
- `Reader`: `read(mut buf []u8) !int`
- `Writer`: `write(buf []u8) !int`
- `Closer`: `close() !bool`

### Working with Files:
```vira
import io

fn main() {
    mut f := io.File.create('test.txt') or { panic(err) }
    f.write_string('Hello from Vira!') or { panic(err) }
    f.close() or { false }

    mut reader := io.File.open('test.txt') or { panic(err) }
    content := reader.read_to_string() or { panic(err) }
    println('Read: ${content}')
    reader.close() or { false }
}
```

---

## 3. TCP & UDP Networking (`packages/net`)

The `net` package exposes cross-platform OS network sockets with zero leaks:

### TCP Echo Server & Client:
```vira
import net

fn run_server() {
    mut listener := net.TcpListener.bind('127.0.0.1:8080') or { panic(err) }
    println('Listening on 8080...')
    mut conn := listener.accept() or { panic(err) }

    msg := conn.read_string(1024) or { panic(err) }
    println('Received: ${msg}')
    conn.write_str('Echo: ${msg}') or { panic(err) }
    conn.close() or { false }
    listener.close() or { false }
}
```

---

## 4. HTTP Codec (`packages/http`)

The `http` package provides zero-allocation parsing and formatting of HTTP/1.1 requests and responses:

### Handling Requests:
```vira
import http
import json

fn handle_request(req http.Request) http.Response {
    switch req.method {
    case 'GET':
        switch req.path {
        case '/health':
            mut jb := json.JsonBuilder.new()
            jb.set_str('status', 'ok')
            jb.set_bool('healthy', true)
            return http.Response.json(jb.build())
        default:
            return http.Response.with_status(404, '{"error":"not_found"}')
        }
    default:
        return http.Response.with_status(405, '{"error":"method_not_allowed"}')
    }
}
```

### Running an HTTP Server:
```vira
import http

fn main() {
    mut srv := http.HttpServer.bind('127.0.0.1:8080') or { panic(err) }
    println('Server running at http://127.0.0.1:8080')

    for true {
        mut ctx := srv.accept_request() or { break }
        res := handle_request(ctx.req)
        ctx.respond(res) or { false }
    }
}
```

---

## 5. Pure-Vira RFC 8259 JSON Codec (`packages/json`)

`packages/json` is implemented in **100% pure Vira** (`.vir`) without C parser bloat, providing:
- Full RFC 8259 syntax validation (`json.is_valid`)
- `{}` / `[]` depth-aware key queries (`json.has_key`, `json.get_raw`, `json.get_str`, `json.get_int`, `json.get_i64`, `json.get_bool`)
- JSON array inspection (`json.array_len`, `json.array_get_raw`, `json.array_get_str`)
- String escaping/unescaping (`json.escape`, `json.unescape`, including `\uXXXX`)
- $O(N)$ byte-buffered `json.JsonBuilder` serialization with `active_leaks=0`

```vira
import json

fn parse_and_reply(raw string) !string {
    model := json.get_str(raw, 'model')!
    heads := json.get_int(raw, 'heads')!

    mut jb := json.JsonBuilder.new()
    jb.set_str('model', model)
    jb.set_int('heads', heads)
    jb.set_bool('accepted', true)
    return jb.build()
}
```

---

## 6. ViraServe Microframework (`packages/viraserve`)

`viraserve` builds on top of `http.HttpServer` and actor concurrency to provide:
- Work-stealing request worker pools (`RequestWorker`)
- Dynamic route matching (`Router`)
- Live telemetry metrics (throughput, average latency in microseconds, error rates)
- Deterministic memory safety with zero garbage collection pauses (`active_leaks=0`)

```vira
import viraserve
import http

fn main() {
    mut app := viraserve.ViraServe.new('APIGateway', 8080)
    app.print_banner()

    mut srv := http.HttpServer.bind('0.0.0.0:8080') or { panic(err) }
    for true {
        app.serve_one(mut srv) or { break }
    }
}
```
