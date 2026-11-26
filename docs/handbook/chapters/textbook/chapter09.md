# Chapter 9: Error Handling (`!T` Results & Fault Boundaries)

Reliable systems software must separate expected operational failures—such as malformed user input, closed sockets, or insufficient account balances—from programmer bugs and invariant violations. Vira draws a clean boundary between the two: recoverable failures use the compile-time checked **Result type** (`!T`), while unrecoverable faults trigger `panic(...)`, which aborts the process in sequential code or isolates and restarts the failing actor under supervision.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch09_error_handling.vir`](../code/ch09_error_handling.vir)

---

## 9.1. Recoverable Errors with Result Types (`!T`) and `return error('...')`

In Vira, there are no exceptions and no hidden control-flow jumps. Any function or method that can fail announces that fact in its return type by prefixing the success type with `!`:

```v
struct LedgerAccount {
pub:
	id string
pub mut:
	balance_cents i64
}

fn (mut a LedgerAccount) withdraw(amount i64) !i64 {
	if amount <= 0 {
		return error('withdrawal amount must be positive')
	}
	if amount > a.balance_cents {
		return error('insufficient funds in ${a.id}: balance=${a.balance_cents}, requested=${amount}')
	}
	a.balance_cents -= amount
	return a.balance_cents
}
```

Inside a function returning `!T`:
- **Returning failure:** Call `return error('message')` with any `string` (including interpolated error context).
- **Returning success:** Return a normal value of type `T` (`return a.balance_cents`). Under the hood, Vira compiles `!T` to a compact value-returned `Result_T` struct (`{ bool is_err; vira_string err; T val; }`) with zero exception tables and deterministic Autofree cleanup on every return path.

### Status-Only Results (`!bool` and `!`)

When a fallible check has no meaningful payload on success, Vira's type syntax supports both `!bool` and the void Result `!` (`Result_void`). Across the Vira standard library (such as `File.flush() !bool` and `TcpStream.close() !bool`), status-only operations conventionally return `!bool` with an explicit `return true` on success:

```v
fn validate_service_port(port int) !bool {
	if port < 1024 {
		return error('port ${port} is privileged (< 1024)')
	}
	return true
}
```

---

## 9.2. Mandatory Unwrapping at Call Sites (`or { ... }`, Fallback Values, and the Implicit `err`)

You can never accidentally ignore a `!T` result. Calling a function that returns `!T` without either an `or { ... }` handler or `!` propagation is a **compile-time semantic error**:

```text
error[SEMANTIC]: unhandled Result '!int' from 'parse_port()': handle it with 'or { ... }' or propagate with '!'
```

When you handle a Result with `or { ... }`, the compiler injects an immutable `err` variable of type `string` into the `or` block scope containing the error message. You have four idiomatic ways to resolve the block:

1. **Early Exit (`return`)**: Log the error and return from the enclosing function:
```v
after_ok := acc.withdraw(12500) or {
	eprintln('Unexpected error: ${err}')
	return
}
```

2. **Fallback Value**: End the `or { ... }` block with an expression of type `T` so execution continues with a safe default:
```v
after_overdraft := acc.withdraw(90000) or {
	println('[2] Overdraft caught: ${err}')
	acc.balance_cents
}
```

3. **Loop Control (`continue` / `break`)**: Skip bad records or stop scanning inside a `for` loop:
```v
for raw in raw_ports {
	ep := build_endpoint('api.local', raw) or {
		println('    Skipped "${raw}": ${err}')
		continue
	}
	valid_endpoints.push(ep)
}
```

4. **Translating Errors (`return error(...)`)**: Catch a low-level error and wrap it in a domain-specific message:
```v
val := strings.parse_i64_base(trimmed, 10) or {
	return error('invalid port number "${trimmed}"')
}
```

> **Note on Empty `or {}`:** An `or {}` block may only be empty when the inner Result type is `void` (`!`). For any non-void `!T` (including `!bool`), every path through `or { ... }` must either diverge (`return`, `break`, `continue`, `panic`) or evaluate to a fallback value of type `T`.

---

## 9.3. Propagating Errors Up the Call Stack (`!`)

When a helper function returns `!U` and calls several lower-level fallible functions, repeating `or { return error(err) }` at every step adds unnecessary boilerplate. Appending `!` directly after a call expression unwraps the success value or immediately returns the error to the caller:

```v
fn build_endpoint(host string, raw_port string) !string {
	clean_host := host.trim()
	if clean_host.len == 0 {
		return error('host cannot be empty')
	}
	port := parse_port(raw_port)!
	_ := validate_service_port(port)!
	return '${clean_host}:${port}'
}
```

During compilation, the semantic checker lowers `parse_port(raw_port)!` directly into `or { return error(err) }`—running all active `defer` blocks and Autofree destructors (such as freeing `clean_host`) before returning the `Result_string` error value.

You may only use postfix `!` inside a function whose own return type is a Result (`!T` or `!`). Attempting to propagate `!` out of a non-Result function (such as `fn main()`) is rejected at compile time:

```text
error[SEMANTIC]: cannot propagate '!' on 'build_endpoint()': enclosing function returns 'void', not a Result '!T'
```

---

## 9.4. Unrecoverable Faults (`panic`) and When to Use Result vs. Panic

Use `panic(message)` only when an internal program invariant is broken and continuing execution would corrupt state:

```v
fn require_non_empty(label string, value string) string {
	if value.len == 0 {
		panic('invariant failed: ${label} must not be empty')
	}
	return value
}
```

Runtime safety traps—integer division or modulo by zero, out-of-bounds slice indexing, popping from an empty slice, and calling an uninitialized function value—also invoke the runtime fault handler. What happens next depends on the execution context:
- **In `main()` or plain functions:** `panic('msg')` prints `vira unrecoverable panic: msg` to `stderr` and exits the process with status code `1`.
- **Inside an `actor` message handler:** As detailed in Chapter 14, the runtime fault handler unwinds local stack frames via LIFO `defer`, wipes the actor's isolated arena in $O(1)$ time, and transfers control back to the actor's `setjmp` fault boundary so its supervisor can restart it without crashing the rest of the application.

| Failure Category | Mechanism | Examples |
| :--- | :--- | :--- |
| **External / Expected Failure** | `!T` + `return error('...')` | Invalid user input, missing files, closed TCP sockets, rejected business rules. |
| **Broken Internal Invariant** | `panic('...')` | Impossible state transitions, corrupt internal structures, test assertion failures. |

---

## 9.5. Running the Complete Chapter 9 Example

Run [`docs/handbook/chapters/code/ch09_error_handling.vir`](../code/ch09_error_handling.vir):

```bash
./vira run docs/handbook/chapters/code/ch09_error_handling.vir
```

Output:
```text
[1] Withdraw ok: remaining=37500 cents
[2] Overdraft caught: insufficient funds in acct_ops: balance=37500, requested=90000
    Fallback preserved balance=37500 cents
[3] Port check rejected: port 80 is privileged (< 1024)
    Port 8080 passed validation
[4] Built endpoint: 127.0.0.1:8443
    Skipped "80": port 80 is privileged (< 1024)
    Skipped "abc": invalid port number "abc"
    Accepted endpoints: 1 (api.local:9000)
[5] Invariant tag verified: vira-gateway
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=31
```

---

## Summary & Exercises

1. **Contextual Wrapping:** Modify `build_endpoint` in `ch09_error_handling.vir` so that instead of forwarding `parse_port(raw_port)!` verbatim, it uses `or { return error('endpoint "${clean_host}" failed: ${err}') }` and observe the enriched message in the loop output.
2. **Unhandled Result Diagnostic:** Remove `or { ... }` from one `acc.withdraw(...)` call in `main()` and run `./vira check docs/handbook/chapters/code/ch09_error_handling.vir` to inspect the compile-time error.
3. **Deposit Validation:** Add a `deposit(amount i64) !i64` method on `LedgerAccount` that rejects non-positive amounts with `return error(...)`, and test both valid and invalid deposits in `main()` while verifying `active_leaks=0`.
