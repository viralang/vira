# Tutorial 3: Structural Interfaces & Dynamic Dispatch

> **Difficulty:** Intermediate  
> **Matching Code Example:** [`docs/examples/03_interfaces.vex`](../examples/03_interfaces.vex)

---

## 1. Overview

Vex implements **structural subtyping** (often called "duck typing" with compile-time type safety). You do not explicitly declare that a type implements an interface; if a struct defines methods matching an interface's signatures, it satisfies that interface automatically.

In this tutorial, you will learn:
- How to declare interfaces
- Implicit interface satisfaction
- Fat-pointer boxing (`._object` + `._vtable`)
- Dynamic polymorphic method dispatch
- Lifetime tracking and automatic destructor chaining

---

## 2. Defining Interfaces

Interfaces declare method signatures that implementors must provide. If a method modifies state, mark it with `mut:`:

```vex
pub interface Notifier {
	channel_name() string
	send(recipient string, body string) string
}
```

---

## 3. Implicit Satisfaction

Structs implement interfaces simply by defining the corresponding receiver methods:

```vex
pub struct EmailService {
pub:
	smtp_host string
}

pub fn (e EmailService) channel_name() string {
	return 'Email [SMTP via ${e.smtp_host}]'
}

pub fn (e EmailService) send(recipient string, body string) string {
	return 'Delivered email to ${recipient}: ${body}'
}

pub struct WebhookService {
pub:
	endpoint string
}

pub fn (w WebhookService) channel_name() string {
	return 'Webhook [POST ${w.endpoint}]'
}

pub fn (w WebhookService) send(recipient string, body string) string {
	return 'Triggered webhook for ${recipient} with payload "${body}"'
}
```

Neither `EmailService` nor `WebhookService` contains an `implements Notifier` clause.

---

## 4. Fat-Pointer Boxing & Dynamic Dispatch

To utilize runtime polymorphism, box a satisfying struct instance into an interface value:

```vex
email := EmailService{ smtp_host: 'smtp.vex.internal:587' }
webhook := WebhookService{ endpoint: 'https://api.vex.dev/events' }

// Boxing into 16-byte fat-pointers
n1 := Notifier(email)
n2 := Notifier(webhook)
```

### Memory Layout Under the Hood

An interface value in Vex is a 16-byte fat pointer consisting of two 64-bit addresses:

```
+-----------------------------------+------------------------------------+
|  ._object (Pointer to heap data)  |  ._vtable (Pointer to method table) |
+-----------------------------------+------------------------------------+
```

When you invoke a method on an interface variable:
```vex
println(n1.channel_name())
```
The Vex compiler generates a direct vtable dereference in C99:
```c
n1._vtable->channel_name(n1._object);
```

---

## 5. Automatic Destructor Chaining

Every generated vtable includes an internal `._autofree` hook:

```c
typedef struct Notifier_vtable {
    vex_string (*channel_name)(const void* _self);
    vex_string (*send)(const void* _self, vex_string, vex_string);
    void (*_autofree)(void* _self); // Injected by compiler
} Notifier_vtable;
```

When an interface variable goes out of scope, the autofree engine automatically executes:
```c
n1._vtable->_autofree(n1._object);
```
This ensures that both the boxed heap container and any heap-allocated fields (like strings or tensors) owned by the underlying struct are safely reclaimed with zero leaks.

---

## 6. Running the Code

```powershell
.\vex.exe run docs/examples/03_interfaces.vex
```

Expected output:
```text
=== 1. IMPLICIT INTERFACE SATISFACTION & FAT-POINTER BOXING ===
=== 2. POLYMORPHIC VTABLE DISPATCH ===
Email [SMTP via smtp.vex.internal:587]
Delivered email to ops@company.com: Database migration completed
Webhook [POST https://api.vex.dev/events]
Triggered webhook for slack_bot with payload "All deployment checks passed"
[vex_rt] active_leaks=0 | total_freed=6
```

---

## Next Steps

Proceed to [**Tutorial 4: Error Handling with Results (`!T`)**](./04_error_handling_and_results.md) to master Vex's zero-cost, unignorable error handling paradigm.
