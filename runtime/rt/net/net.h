#ifndef VIRA_RT_NET_H
#define VIRA_RT_NET_H

// ============================================================================
// 12. HIGH-PERFORMANCE NETWORKING & HTTP 1.1 WEB ENGINE
// ============================================================================

/**
 * Initializes platform socket subsystem (WSAStartup on Windows; no-op on POSIX).
 * Usage: Internal helper called once before any socket operations.
 */
static inline void _vira_net_init(void) {
#ifdef _WIN32
  static int initialized = 0;
  if (!initialized) {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    initialized = 1;
  }
#else
  static int initialized = 0;
  if (!initialized) {
    signal(SIGPIPE, SIG_IGN);
    initialized = 1;
  }
#endif
}

// --- LEGACY MINIMAL SOCKET WRAPPERS ---
/**
 * Creates and listens on a TCP socket bound to 0.0.0.0 at the specified port (legacy minimal wrapper).
 * @param port TCP port number.
 * @return Server socket descriptor, or negative on error.
 * Usage: Simple HTTP server socket binding.
 */
static inline int64_t vira_net_listen(int64_t port) {
  _vira_net_init();
  vira_socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd == VIRA_INVALID_SOCKET)
    return -1;

  int opt = 1;
#ifdef _WIN32
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));
#else
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)port);

  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == VIRA_SOCKET_ERROR) {
    closesocket(fd);
    return -1;
  }

  if (listen(fd, 128) == VIRA_SOCKET_ERROR) {
    closesocket(fd);
    return -1;
  }

  return (int64_t)fd;
}

/**
 * Accepts an incoming client connection on a listening server socket descriptor.
 * @param server_fd Server socket descriptor.
 * @return Accepted client socket descriptor, or negative on error.
 * Usage: Accepting client connections in server event loops.
 */
static inline int64_t vira_net_accept(int64_t server_fd) {
  _vira_net_init();
  struct sockaddr_in client_addr;
  socklen_t addr_len = sizeof(client_addr);
  vira_socket_t client = accept((vira_socket_t)server_fd,
                                (struct sockaddr *)&client_addr, &addr_len);
  if (client == VIRA_INVALID_SOCKET)
    return -1;
  return (int64_t)client;
}

/**
 * Reads incoming text from a client socket descriptor up to `max_bytes` into a `vira_string`.
 * @param client_fd Client socket descriptor.
 * @param max_bytes Maximum bytes to read.
 * @return String containing received data.
 * Usage: Simple string socket reception.
 */
static inline vira_string vira_net_read(int64_t client_fd, int64_t max_bytes) {
  if (max_bytes <= 0)
    max_bytes = 4096;
  char *buf = (char *)malloc(max_bytes + 1);
  if (!buf)
    return vira_str_lit("", 0);
  _vira_record_alloc();

  int n = recv((vira_socket_t)client_fd, buf, (int)max_bytes, 0);
  if (n <= 0) {
    free(buf);
    _vira_record_free();
    return vira_str_lit("", 0);
  }
  buf[n] = '\0';
  return (vira_string){.data = (const uint8_t *)buf, .len = n, .is_lit = false};
}

/**
 * Transmits a `vira_string` message across a client socket descriptor.
 * @param client_fd Client socket descriptor.
 * @param msg String payload to send.
 * @return Number of bytes written, or negative on error.
 * Usage: Simple string socket transmission.
 */
static inline int64_t vira_net_write(int64_t client_fd, vira_string msg) {
  if (msg.len <= 0 || msg.data == NULL)
    return 0;
  int sent = send((vira_socket_t)client_fd, (const char *)msg.data, msg.len, 0);
  return (int64_t)sent;
}

/**
 * Closes a network socket descriptor.
 * @param fd Socket descriptor to close.
 * Usage: Closes network connections.
 */
static inline void vira_net_close(int64_t fd) {
  if (fd >= 0) {
    closesocket((vira_socket_t)fd);
  }
}

// --- STANDARD VIRA NETWORKING (TCP, UDP, MULTIPLEXING) ---

/**
 * Creates, binds, and configures a TCP server socket on host and port with specified backlog.
 * @param host Bind address string (e.g. "0.0.0.0" or "127.0.0.1").
 * @param port TCP port number.
 * @param backlog Connection listen backlog queue size.
 * @return Listening socket descriptor, or negative on error.
 * Usage: High-performance TCP server initialization (`net::tcp_listen("0.0.0.0", 8080, 128)`).
 */
static inline int64_t vira_net_tcp_listen(vira_string host, int32_t port,
                                          int32_t backlog) {
  _vira_net_init();
  if (port < 0 || port > 65535)
    return -1;
  if (backlog <= 0)
    backlog = 128;

  char stack_host[256];
  vira_cstr_buf hbuf = vira_str_to_cstr_buf(host, stack_host, sizeof(stack_host));

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL, *p = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  const char *node =
      (host.len > 0 && strcmp(hbuf.ptr, "0.0.0.0") != 0) ? hbuf.ptr : NULL;
  int gai = getaddrinfo(node, port_buf, &hints, &res);
  vira_cstr_buf_free(&hbuf);
  if (gai != 0) {
    return -1;
  }

  vira_socket_t listen_fd = VIRA_INVALID_SOCKET;
  for (p = res; p != NULL; p = p->ai_next) {
    listen_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (listen_fd == VIRA_INVALID_SOCKET)
      continue;

    int opt = 1;
#ifdef _WIN32
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
               sizeof(opt));
#else
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif

    if (bind(listen_fd, p->ai_addr, (socklen_t)p->ai_addrlen) == 0) {
      break;
    }

    closesocket(listen_fd);
    listen_fd = VIRA_INVALID_SOCKET;
  }

  freeaddrinfo(res);

  if (listen_fd == VIRA_INVALID_SOCKET)
    return -1;
  if (listen(listen_fd, (int)backlog) == VIRA_SOCKET_ERROR) {
    closesocket(listen_fd);
    return -1;
  }

  return (int64_t)listen_fd;
}

/**
 * Accepts an incoming connection on a TCP listening socket.
 * @param server_fd Listening socket descriptor.
 * @return Connected client socket descriptor, or negative on error.
 * Usage: Accepting client connections in multi-threaded web servers.
 */
static inline int64_t vira_net_tcp_accept(int64_t server_fd) {
  _vira_net_init();
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  vira_socket_t client =
      accept((vira_socket_t)server_fd, (struct sockaddr *)&addr, &addr_len);
  if (client == VIRA_INVALID_SOCKET)
    return -1;
  return (int64_t)client;
}

/**
 * Retrieves the remote peer IP address and port formatted as "IP:port" string.
 * @param fd Connected socket descriptor.
 * @return Formatted peer address string.
 * Usage: Connection logging and remote host identification (`net::peer_addr(fd)`).
 */
static inline vira_string vira_net_peer_addr(int64_t fd) {
  if (fd < 0)
    return vira_str_lit("", 0);
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  if (getpeername((vira_socket_t)fd, (struct sockaddr *)&addr, &addr_len) !=
      0) {
    return vira_str_lit("", 0);
  }
  if (addr.ss_family == AF_INET) {
    struct sockaddr_in *s = (struct sockaddr_in *)&addr;
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip));
    int port = ntohs(s->sin_port);
    return vira_str_format("%s:%d", ip, port);
  } else if (addr.ss_family == AF_INET6) {
    struct sockaddr_in6 *s = (struct sockaddr_in6 *)&addr;
    char ip[INET6_ADDRSTRLEN];
    inet_ntop(AF_INET6, &s->sin6_addr, ip, sizeof(ip));
    int port = ntohs(s->sin6_port);
    return vira_str_format("[%s]:%d", ip, port);
  }
  return vira_str_lit("", 0);
}

/**
 * Retrieves the local bound IP address and port formatted as "IP:port" string.
 * @param fd Socket descriptor.
 * @return Formatted local address string.
 * Usage: Inspecting dynamically bound port numbers (`net::local_addr(fd)`).
 */
static inline vira_string vira_net_local_addr(int64_t fd) {
  if (fd < 0)
    return vira_str_lit("", 0);
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  if (getsockname((vira_socket_t)fd, (struct sockaddr *)&addr, &addr_len) !=
      0) {
    return vira_str_lit("", 0);
  }
  if (addr.ss_family == AF_INET) {
    struct sockaddr_in *s = (struct sockaddr_in *)&addr;
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip));
    int port = ntohs(s->sin_port);
    return vira_str_format("%s:%d", ip, port);
  } else if (addr.ss_family == AF_INET6) {
    struct sockaddr_in6 *s = (struct sockaddr_in6 *)&addr;
    char ip[INET6_ADDRSTRLEN];
    inet_ntop(AF_INET6, &s->sin6_addr, ip, sizeof(ip));
    int port = ntohs(s->sin6_port);
    return vira_str_format("[%s]:%d", ip, port);
  }
  return vira_str_lit("", 0);
}

/**
 * Establishes an outbound TCP connection to target host and port with timeout.
 * @param host Remote hostname or IP address.
 * @param port Remote TCP port number.
 * @param timeout_ms Connection timeout in milliseconds.
 * @return Connected socket descriptor, or negative on error/timeout.
 * Usage: Outbound TCP client connections (`net::tcp_connect("api.example.com", 443, 5000)`).
 */
static inline int64_t vira_net_tcp_connect(vira_string host, int32_t port,
                                           int64_t timeout_ms) {
  _vira_net_init();
  if (port <= 0 || port > 65535)
    return -1;

  char stack_host[256];
  vira_cstr_buf hbuf = vira_str_to_cstr_buf(host, stack_host, sizeof(stack_host));

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL, *p = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int gai = getaddrinfo(hbuf.ptr, port_buf, &hints, &res);
  vira_cstr_buf_free(&hbuf);
  if (gai != 0) {
    return -1;
  }

  vira_socket_t conn_fd = VIRA_INVALID_SOCKET;
  for (p = res; p != NULL; p = p->ai_next) {
    conn_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (conn_fd == VIRA_INVALID_SOCKET)
      continue;

    if (timeout_ms <= 0) {
      if (connect(conn_fd, p->ai_addr, (socklen_t)p->ai_addrlen) == 0) {
        break;
      }
    } else {
#ifdef _WIN32
      u_long mode = 1;
      ioctlsocket(conn_fd, FIONBIO, &mode);
#else
      int orig_flags = fcntl((int)conn_fd, F_GETFL, 0);
      fcntl((int)conn_fd, F_SETFL, orig_flags | O_NONBLOCK);
#endif
      int ret = connect(conn_fd, p->ai_addr, (socklen_t)p->ai_addrlen);
      bool ok = false;
      if (ret == 0) {
        ok = true;
      } else {
#ifdef _WIN32
        int err = WSAGetLastError();
        if (err == WSAEWOULDBLOCK) {
#else
        if (errno == EINPROGRESS) {
#endif
          struct pollfd pfd;
          pfd.fd = conn_fd;
          pfd.events = POLLOUT;
          pfd.revents = 0;
          int pr = vira_poll(&pfd, 1, (int)timeout_ms);
          if (pr > 0 && (pfd.revents & POLLOUT)) {
            int sock_err = 0;
            socklen_t errlen = sizeof(sock_err);
            getsockopt(conn_fd, SOL_SOCKET, SO_ERROR, (char *)&sock_err,
                       &errlen);
            if (sock_err == 0) {
              ok = true;
            }
          }
        }
      }

#ifdef _WIN32
      mode = 0;
      ioctlsocket(conn_fd, FIONBIO, &mode);
#else
      fcntl((int)conn_fd, F_SETFL, orig_flags);
#endif
      if (ok) {
        break;
      }
    }

    closesocket(conn_fd);
    conn_fd = VIRA_INVALID_SOCKET;
  }

  freeaddrinfo(res);
  if (conn_fd == VIRA_INVALID_SOCKET)
    return -1;
  return (int64_t)conn_fd;
}

/**
 * Reads binary bytes from a TCP socket into a buffer slice up to `max_len`.
 * @param fd Connected socket descriptor.
 * @param buf Pointer to destination byte slice.
 * @param max_len Maximum bytes to receive.
 * @return Number of bytes received, 0 on connection closed, or negative on error.
 * Usage: Binary network data streaming (`net::tcp_read(fd, &buf, 4096)`).
 */
static inline int32_t vira_net_tcp_read(int64_t fd, vira_slice_u8 *buf,
                                        int32_t max_len) {
  if (fd < 0 || buf == NULL || buf->data == NULL || max_len <= 0) {
    return 0;
  }
  int32_t can_read = buf->cap < max_len ? buf->cap : max_len;
  if (can_read <= 0)
    return 0;

  int n = recv((vira_socket_t)fd, (char *)buf->data, (int)can_read, 0);
  if (n > 0) {
    buf->len = (int32_t)n;
    return (int32_t)n;
  }
  if (n == 0) {
    buf->len = 0;
    return 0;
  }
#ifdef _WIN32
  int werr = WSAGetLastError();
  if (werr == WSAEWOULDBLOCK)
    return -2;
#else
  if (errno == EAGAIN || errno == EWOULDBLOCK)
    return -2;
#endif
  return -1;
}

/**
 * Transmits binary bytes from a buffer slice across a TCP socket.
 * @param fd Connected socket descriptor.
 * @param buf Source byte slice.
 * @param len Number of bytes to transmit.
 * @return Number of bytes transmitted, or negative on error.
 * Usage: Binary network transmission (`net::tcp_write(fd, buf, len)`).
 */
static inline int32_t vira_net_tcp_write(int64_t fd, vira_slice_u8 buf,
                                         int32_t len) {
  if (fd < 0 || buf.data == NULL || len <= 0) {
    return 0;
  }
  int32_t to_write = buf.len < len ? buf.len : len;
  if (to_write <= 0)
    return 0;

  int flags = 0;
#ifdef MSG_NOSIGNAL
  flags |= MSG_NOSIGNAL;
#endif
  int sent =
      send((vira_socket_t)fd, (const char *)buf.data, (int)to_write, flags);
  return (int32_t)sent;
}

/**
 * Closes a TCP socket descriptor.
 * @param fd Socket descriptor.
 * @return 0 on success, negative on error.
 * Usage: Terminates TCP connection (`net::tcp_close(fd)`).
 */
static inline int32_t vira_net_tcp_close(int64_t fd) {
  if (fd < 0)
    return 0;
  return closesocket((vira_socket_t)fd) == 0 ? 0 : -1;
}

/**
 * Shuts down read, write, or both halves of a full-duplex TCP connection.
 * @param fd Socket descriptor.
 * @param how 0 for read, 1 for write, 2 for both.
 * @return 0 on success, negative on error.
 * Usage: Orderly socket termination signaling (`net::tcp_shutdown(fd, 1)`).
 */
static inline int32_t vira_net_tcp_shutdown(int64_t fd, int32_t how) {
  if (fd < 0)
    return 0;
#ifdef _WIN32
  int c_how = (how == 0) ? SD_RECEIVE : ((how == 1) ? SD_SEND : SD_BOTH);
#else
  int c_how = (how == 0) ? SHUT_RD : ((how == 1) ? SHUT_WR : SHUT_RDWR);
#endif
  return shutdown((vira_socket_t)fd, c_how) == 0 ? 0 : -1;
}

/**
 * Enables or disables non-blocking I/O mode on a socket descriptor.
 * @param fd Socket descriptor.
 * @param nonblocking True for non-blocking, false for blocking.
 * @return 0 on success, negative on error.
 * Usage: Configures sockets for asynchronous polling and event loops.
 */
static inline int32_t vira_net_set_nonblocking(int64_t fd, bool nonblocking) {
  if (fd < 0)
    return -1;
#ifdef _WIN32
  u_long mode = nonblocking ? 1 : 0;
  return ioctlsocket((vira_socket_t)fd, FIONBIO, &mode) == 0 ? 0 : -1;
#else
  int flags = fcntl((int)fd, F_GETFL, 0);
  if (flags < 0)
    return -1;
  flags = nonblocking ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
  return fcntl((int)fd, F_SETFL, flags) == 0 ? 0 : -1;
#endif
}

/**
 * Enables or disables TCP_NODELAY (disabling Nagle's algorithm for low latency).
 * @param fd Socket descriptor.
 * @param nodelay True to disable Nagle's algorithm.
 * @return 0 on success, negative on error.
 * Usage: Optimizes low-latency protocols and real-time streaming.
 */
static inline int32_t vira_net_set_nodelay(int64_t fd, bool nodelay) {
  if (fd < 0)
    return -1;
  int opt = nodelay ? 1 : 0;
  return setsockopt((vira_socket_t)fd, IPPROTO_TCP, TCP_NODELAY,
                    (const char *)&opt, sizeof(opt)) == 0
             ? 0
             : -1;
}

/**
 * Enables or disables SO_REUSEADDR on a socket descriptor.
 * @param fd Socket descriptor.
 * @param reuse True to permit local address reuse.
 * @return 0 on success, negative on error.
 * Usage: Prevents "Address already in use" errors during rapid server restarts.
 */
static inline int32_t vira_net_set_reuseaddr(int64_t fd, bool reuse) {
  if (fd < 0)
    return -1;
  int opt = reuse ? 1 : 0;
  return setsockopt((vira_socket_t)fd, SOL_SOCKET, SO_REUSEADDR,
                    (const char *)&opt, sizeof(opt)) == 0
             ? 0
             : -1;
}

/**
 * Sets socket read and write timeout intervals in milliseconds.
 * @param fd Socket descriptor.
 * @param read_timeout_ms Read timeout in ms (or 0 for infinite).
 * @param write_timeout_ms Write timeout in ms (or 0 for infinite).
 * @return 0 on success, negative on error.
 * Usage: Enforcing network I/O deadlines.
 */
static inline int32_t vira_net_set_timeout(int64_t fd, int64_t read_timeout_ms,
                                           int64_t write_timeout_ms) {
  if (fd < 0)
    return -1;
#ifdef _WIN32
  DWORD r_to = (DWORD)read_timeout_ms;
  DWORD w_to = (DWORD)write_timeout_ms;
  if (read_timeout_ms >= 0)
    setsockopt((vira_socket_t)fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&r_to,
               sizeof(r_to));
  if (write_timeout_ms >= 0)
    setsockopt((vira_socket_t)fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&w_to,
               sizeof(w_to));
#else
  if (read_timeout_ms >= 0) {
    struct timeval tv;
    tv.tv_sec = read_timeout_ms / 1000;
    tv.tv_usec = (read_timeout_ms % 1000) * 1000;
    setsockopt((int)fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  }
  if (write_timeout_ms >= 0) {
    struct timeval tv;
    tv.tv_sec = write_timeout_ms / 1000;
    tv.tv_usec = (write_timeout_ms % 1000) * 1000;
    setsockopt((int)fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  }
#endif
  return 0;
}

/**
 * Polls a socket descriptor for read/write readiness with a timeout in milliseconds.
 * @param fd Socket descriptor.
 * @param events Bitmask of poll events (1 for read, 2 for write).
 * @param timeout_ms Timeout in milliseconds.
 * @return Positive bitmask of ready events, 0 on timeout, negative on error.
 * Usage: Multiplexing socket readiness.
 */
static inline int32_t vira_net_poll(int64_t fd, int32_t events,
                                    int32_t timeout_ms) {
  if (fd < 0)
    return -1;
  struct pollfd pfd;
  pfd.fd = (vira_socket_t)fd;
  pfd.events = 0;
  if (events & 1)
    pfd.events |= POLLIN;
  if (events & 2)
    pfd.events |= POLLOUT;
  pfd.revents = 0;

  int n = vira_poll(&pfd, 1, (int)timeout_ms);
  if (n <= 0)
    return n;

  int32_t res = 0;
  if (pfd.revents & POLLIN)
    res |= 1;
  if (pfd.revents & POLLOUT)
    res |= 2;
  if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
    res |= 4;
  return res;
}

#if defined(__GNUC__) || defined(__clang__)
static __thread struct sockaddr_storage _vira_last_udp_sender;
static __thread socklen_t _vira_last_udp_sender_len = 0;
#else
static struct sockaddr_storage _vira_last_udp_sender;
static socklen_t _vira_last_udp_sender_len = 0;
#endif

/**
 * Creates and binds a UDP datagram socket to specified host and port.
 * @param host Local bind address string.
 * @param port Local UDP port number.
 * @return Bound UDP socket descriptor, or negative on error.
 * Usage: UDP server or listener setup (`net::udp_bind("0.0.0.0", 9000)`).
 */
static inline int64_t vira_net_udp_bind(vira_string host, int32_t port) {
  _vira_net_init();
  if (port < 0 || port > 65535)
    return -1;

  char stack_host[256];
  vira_cstr_buf hbuf = vira_str_to_cstr_buf(host, stack_host, sizeof(stack_host));

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL, *p = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_flags = AI_PASSIVE;

  const char *node =
      (host.len > 0 && strcmp(hbuf.ptr, "0.0.0.0") != 0) ? hbuf.ptr : NULL;
  int gai = getaddrinfo(node, port_buf, &hints, &res);
  vira_cstr_buf_free(&hbuf);
  if (gai != 0) {
    return -1;
  }

  vira_socket_t udp_fd = VIRA_INVALID_SOCKET;
  for (p = res; p != NULL; p = p->ai_next) {
    udp_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (udp_fd == VIRA_INVALID_SOCKET)
      continue;

    int opt = 1;
#ifdef _WIN32
    setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
               sizeof(opt));
#else
    setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    setsockopt(udp_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif

    if (bind(udp_fd, p->ai_addr, (socklen_t)p->ai_addrlen) == 0) {
      break;
    }

    closesocket(udp_fd);
    udp_fd = VIRA_INVALID_SOCKET;
  }

  freeaddrinfo(res);
  if (udp_fd == VIRA_INVALID_SOCKET)
    return -1;
  return (int64_t)udp_fd;
}

/**
 * Sends a UDP datagram packet to the specified remote host and port.
 * @param fd UDP socket descriptor.
 * @param buf Byte slice containing payload.
 * @param len Number of bytes to send.
 * @param host Remote destination host.
 * @param port Remote destination port.
 * @return Number of bytes transmitted, or negative on error.
 * Usage: Transmitting UDP packets (`net::udp_send_to(...)`).
 */
static inline int32_t vira_net_udp_send_to(int64_t fd, vira_slice_u8 buf,
                                           int32_t len, vira_string host,
                                           int32_t port) {
  if (fd < 0 || buf.data == NULL || len <= 0 || port < 0 || port > 65535) {
    return 0;
  }
  int32_t to_write = buf.len < len ? buf.len : len;
  if (to_write <= 0)
    return 0;

  char stack_host[256];
  vira_cstr_buf hbuf = vira_str_to_cstr_buf(host, stack_host, sizeof(stack_host));

  char port_buf[16];
  snprintf(port_buf, sizeof(port_buf), "%lld", (long long)port);

  struct addrinfo hints, *res = NULL;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;

  int gai = getaddrinfo(hbuf.ptr, port_buf, &hints, &res);
  vira_cstr_buf_free(&hbuf);
  if (gai != 0 || !res) {
    return -1;
  }

  int sent = sendto((vira_socket_t)fd, (const char *)buf.data, (int)to_write, 0,
                    res->ai_addr, (socklen_t)res->ai_addrlen);
  freeaddrinfo(res);
  return (int32_t)sent;
}

/**
 * Receives a UDP datagram packet into a buffer slice, recording the sender address.
 * @param fd UDP socket descriptor.
 * @param buf Pointer to destination byte slice.
 * @param max_len Maximum bytes to receive.
 * @return Number of bytes received, or negative on error.
 * Usage: Receiving UDP packets (`net::udp_recv_from(...)`).
 */
static inline int32_t vira_net_udp_recv_from(int64_t fd, vira_slice_u8 *buf,
                                             int32_t max_len) {
  if (fd < 0 || buf == NULL || buf->data == NULL || max_len <= 0) {
    return 0;
  }
  int32_t can_read = buf->cap < max_len ? buf->cap : max_len;
  if (can_read <= 0)
    return 0;

  _vira_last_udp_sender_len = sizeof(_vira_last_udp_sender);
  memset(&_vira_last_udp_sender, 0, sizeof(_vira_last_udp_sender));

  int n = recvfrom((vira_socket_t)fd, (char *)buf->data, (int)can_read, 0,
                   (struct sockaddr *)&_vira_last_udp_sender,
                   &_vira_last_udp_sender_len);
  if (n >= 0) {
    buf->len = (int32_t)n;
    return (int32_t)n;
  }
  return -1;
}

/**
 * Retrieves the address of the sender that transmitted the last received UDP packet as "IP:port".
 * @return Formatted sender address string.
 * Usage: Answering incoming UDP datagrams.
 */
static inline vira_string vira_net_udp_last_sender(void) {
  if (_vira_last_udp_sender_len == 0) {
    return vira_str_lit("", 0);
  }
  if (_vira_last_udp_sender.ss_family == AF_INET) {
    struct sockaddr_in *s = (struct sockaddr_in *)&_vira_last_udp_sender;
    char ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip));
    int port = ntohs(s->sin_port);
    return vira_str_format("%s:%d", ip, port);
  } else if (_vira_last_udp_sender.ss_family == AF_INET6) {
    struct sockaddr_in6 *s = (struct sockaddr_in6 *)&_vira_last_udp_sender;
    char ip[INET6_ADDRSTRLEN];
    inet_ntop(AF_INET6, &s->sin6_addr, ip, sizeof(ip));
    int port = ntohs(s->sin6_port);
    return vira_str_format("[%s]:%d", ip, port);
  }
  return vira_str_lit("", 0);
}

/**
 * Formats a complete HTTP/1.1 response string with Content-Length header and body.
 * @param status_code HTTP status code (200, 404, 500, etc.).
 * @param content_type MIME type string (e.g. "application/json").
 * @param body Response payload string.
 * @return Fully formatted HTTP response `vira_string`.
 * Usage: Generating web responses in HTTP handlers (`http::response(200, "text/plain", "OK")`).
 */
static inline vira_string vira_http_response(int64_t status_code,
                                             vira_string content_type,
                                             vira_string body) {
  const char *status_text = "OK";
  if (status_code == 201)
    status_text = "Created";
  else if (status_code == 400)
    status_text = "Bad Request";
  else if (status_code == 404)
    status_text = "Not Found";
  else if (status_code == 500)
    status_text = "Internal Server Error";

  char stack_ctype[128];
  vira_cstr_buf cbuf =
      vira_str_to_cstr_buf(content_type, stack_ctype, sizeof(stack_ctype));

  int32_t body_len = body.len > 0 ? body.len : 0;
  size_t ctype_len = content_type.len > 0 ? (size_t)content_type.len : 0;
  size_t hdr_cap = 256 + ctype_len;
  size_t alloc_sz = hdr_cap + (size_t)body_len + 1;
  char *res = (char *)malloc(alloc_sz);
  if (!res) {
    vira_cstr_buf_free(&cbuf);
    return vira_str_lit("", 0);
  }
  _vira_record_alloc();

  int n = snprintf(res, hdr_cap,
                   "HTTP/1.1 %lld %s\r\n"
                   "Content-Type: %s\r\n"
                   "Content-Length: %d\r\n"
                   "Connection: close\r\n"
                   "\r\n",
                   (long long)status_code, status_text, cbuf.ptr, body_len);
  vira_cstr_buf_free(&cbuf);
  if (n < 0)
    n = 0;
  if ((size_t)n >= hdr_cap)
    n = (int)(hdr_cap - 1);

  if (body_len > 0 && body.data) {
    memcpy(res + n, body.data, (size_t)body_len);
    n += body_len;
  }
  res[n] = '\0';
  return (vira_string){.data = (const uint8_t *)res, .len = n, .is_lit = false};
}

#endif // VIRA_RT_NET_H
