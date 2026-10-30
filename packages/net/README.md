# `packages/net`: Cross-Platform Sockets & Networking

The `net` package provides cross-platform TCP listeners, TCP streams, UDP sockets, DNS resolution, and non-blocking I/O multiplexing in the **Vira Programming Language**.

---

## 🚀 Key Features

- **TCP Server & Client**:
  - `net.TcpListener.bind(addr string) !TcpListener`
  - `listener.accept() !TcpStream`
  - `net.TcpStream.connect(addr string) !TcpStream`
  - `conn.read(mut buf)`, `conn.write(buf)`, `conn.read_string(max_len)`, `conn.close()`
- **UDP Datagrams**: `net.UdpSocket.bind(addr)`, `sock.send_to(buf, addr)`, `sock.recv_from(mut buf)`.
- **I/O Multiplexing**: Non-blocking `net.Poller` for high-concurrency event loops.
- **Addressing**: `net.SocketAddr` and hostname resolution.

---

## 🛠️ Usage Example

```v
module main

import net

fn main() {
    // 1. Bind TCP loopback listener
    mut listener := net.TcpListener.bind('127.0.0.1:9090') or { panic(err) }
    println('Server listening on 127.0.0.1:9090')

    // 2. Connect client stream
    mut client := net.TcpStream.connect('127.0.0.1:9090') or { panic(err) }
    client.write('PING\n'.bytes()) or { panic(err) }

    // 3. Accept incoming connection
    mut conn := listener.accept() or { panic(err) }
    req := conn.read_string(1024) or { '' }
    println('Received: ${req}')

    conn.write('PONG\n'.bytes()) or { panic(err) }
    conn.close() or { false }
    client.close() or { false }
    listener.close() or { false }
}
```

---

## 🧪 Testing

Run the package test suite:

```bash
vira test packages/net
```
