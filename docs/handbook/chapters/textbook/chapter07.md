# Chapter 7: Structs, 4-Quadrant Encapsulation & Composition

Vira models data and behavior with **structs, explicit receiver methods, and struct embedding** rather than class inheritance hierarchies. To prevent accidental state mutation and leaky APIs across module boundaries, Vira separates field **visibility** (`pub` vs. module-private) from **mutability** (`mut` vs. immutable) using a crisp **4-quadrant encapsulation matrix**.

> **Companion Runnable Code:** [`docs/handbook/chapters/code/ch07_structs_and_encapsulation.vir`](../code/ch07_structs_and_encapsulation.vir)

---

## 7.1. Defining and Instantiating Structs

A struct bundles related fields under a named type. Like functions and constants, a struct is private to its module unless declared with `pub struct`:

```v
pub struct AuditStamp {
pub:
	created_at i64
pub mut:
	updated_at i64
mut:
	revision int = 1
}
```

Fields can declare default values (such as `revision int = 1`), or receive values explicitly in a struct literal:

```v
readonly := ServiceAccount{
	AuditStamp: AuditStamp{
		created_at: 1750000000
		updated_at: 1750000000
		revision: 1
	}
	api_secret: 'tok-readonly'
	service_id: 'svc-reader'
	owner: 'platform'
	active: true
	rate_limit: 250
}
```

Remember Vira's immutability-first rule:
- Binding `readonly := ServiceAccount{ ... }` creates an **immutable** struct value—none of its fields can be mutated through `readonly`, even fields marked `pub mut:`.
- To mutate a struct's mutable fields or call `mut` receiver methods on it, bind it with `mut svc := ...`.
- Structs do **not** define `==` or `!=` automatically; compare fields explicitly or write a helper method.

---

## 7.2. The 4 Access Quadrants (`default` Private Immutable, `mut:`, `pub:`, `pub mut:`)

Most languages give you a single axis (`public` vs. `private`), leaving public fields wide open to external mutation. Vira's struct body groups fields into four access quadrants:

```v
pub struct ServiceAccount {
	AuditStamp // Embedded struct: fields & methods promoted to ServiceAccount
	api_secret string // Quadrant 1 (default): Private & Immutable
mut:
	failed_logins int = 0 // Quadrant 2 (mut:): Private & Mutable
pub:
	service_id string // Quadrant 3 (pub:): Public & Read-Only
	owner      string
pub mut:
	active     bool // Quadrant 4 (pub mut:): Public & Mutable
	rate_limit int
}
```

### The 4-Quadrant Access Matrix

| Quadrant | Section Header | Read Access | Write Access (on `mut` binding) | Typical Use |
| :--- | :--- | :--- | :--- | :--- |
| **1. Private & Immutable** | *(default, before any label)* | Defining module only | Construction only (never mutated) | Secrets, cryptographic tokens, fixed internal handles |
| **2. Private & Mutable** | `mut:` | Defining module only | Defining module only | Internal counters, caches, state machine fields |
| **3. Public & Read-Only** | `pub:` | All modules | Defining module only | IDs, owner names, creation timestamps |
| **4. Public & Mutable** | `pub mut:` | All modules | All modules | Configuration flags, rate limits, mutable status |

Inside the defining module (`module main` in the companion file), methods and helper functions can inspect and update private fields like `failed_logins`. Outside the defining module, callers can read `svc.service_id` (`pub:`) and update `svc.rate_limit = 5000` (`pub mut:`), while attempts to read `svc.api_secret` or write `svc.service_id` fail at compile time.

---

## 7.3. Value Receivers vs. Mutable Receivers (`fn (s T)` vs. `fn (mut s T)`)

Methods are functions declared with a receiver parameter before the method name:

### Value Receivers (`fn (s ServiceAccount)`)

A value receiver borrows the struct as read-only and can be called on **both** immutable (`readonly`) and mutable (`mut svc`) bindings:

```v
pub fn (s ServiceAccount) describe() string {
	return 'ServiceAccount(${s.service_id}, owner=${s.owner}, active=${s.active}, limit=${s.rate_limit}, fails=${s.failed_logins})'
}

pub fn (s ServiceAccount) verify_secret(candidate string) bool {
	return s.api_secret == candidate
}
```

### Mutable Receivers (`fn (mut s ServiceAccount)`)

When a method modifies fields of the receiver, mark the receiver `mut`. In the generated C99 code, `mut` receivers pass by pointer (`ServiceAccount* s`), updating the caller's struct in place:

```v
pub fn (mut s ServiceAccount) record_auth(ok bool, ts i64) string {
	if !ok {
		s.failed_logins += 1
		if s.failed_logins >= 3 {
			s.active = false
		}
		s.touch(ts) // Calls promoted `AuditStamp.touch`
		return 'denied'
	}
	s.failed_logins = 0
	s.touch(ts)
	return 'granted'
}
```

If you try to call `readonly.record_auth(...)` on an immutable binding, the type checker stops compilation with:
`cannot call mutable receiver method 'record_auth()' on immutable binding 'readonly' (declare 'mut readonly := ...')`.

---

## 7.4. Associated Static Factory Methods (`fn Type.new(...)`)

To encapsulate invariants, populate defaults, or hide private fields from external callers constructing a struct, define an associated static factory method using `StructName.method_name`:

```v
pub fn ServiceAccount.new(service_id string, owner string, secret string, now i64) ServiceAccount {
	return ServiceAccount{
		AuditStamp: AuditStamp{
			created_at: now
			updated_at: now
			revision: 1
		}
		api_secret: secret
		failed_logins: 0
		service_id: service_id
		owner: owner
		active: true
		rate_limit: 1000
	}
}
```

Callers instantiate it cleanly without manually assembling every nested field:

```v
mut svc := ServiceAccount.new('svc-billing', 'payments', 'sec-991', 1750000100)
```

Because `ServiceAccount.new` returns a newly constructed local value, Autofree classifies the return as `ESCAPES_VIA_RETURN` and moves ownership of all heap strings inside `ServiceAccount` directly to `svc` with zero extra copies.

---

## 7.5. Struct Embedding, Composition, and Promoted Fields/Methods

Instead of inheritance (`extends`), Vira supports **struct embedding**: placing a struct type name on its own line inside another struct's body (with no field name):

```v
pub struct ServiceAccount {
	AuditStamp // Embedded struct: fields & methods promoted to ServiceAccount
	api_secret string
...
```

Embedding `AuditStamp` inside `ServiceAccount` promotes both fields and methods directly onto `ServiceAccount`:
1. **Promoted Fields:** You can read `svc.created_at` and `svc.updated_at` directly on `svc`, or qualify them explicitly as `svc.AuditStamp.updated_at`.
2. **Promoted Methods:** Calling `s.touch(ts)` inside `record_auth` or `svc.stamp_summary()` in `main` dispatches directly to `AuditStamp.touch` and `AuditStamp.stamp_summary` on the embedded sub-struct.
3. **Ambiguity Checking:** If two embedded structs define fields or methods with the same name, the compiler reports an ambiguity error at compile time rather than silently picking one.
4. **Automatic Recursive Cleanup:** If an embedded struct owns heap fields (such as strings or slices), Vira's Autofree pass automatically generates destructor calls for the embedded struct inside the outer struct's `_autofree` function.

---

## 7.6. Running the Complete Chapter 7 Example

Run [`docs/handbook/chapters/code/ch07_structs_and_encapsulation.vir`](../code/ch07_structs_and_encapsulation.vir):

```bash
./vira run docs/handbook/chapters/code/ch07_structs_and_encapsulation.vir
```

Output:
```text
[1] Literal & Value Receiver: ServiceAccount(svc-reader, owner=platform, active=true, limit=250, fails=0)
    secret_ok=true
[2] Factory & Promoted Read: id=svc-billing, created=1750000100, rev=1, updated=1750000100
[3] Auth checks: r1=denied, r2=granted, rev=3, updated=1750000300
[4] Final State: ServiceAccount(svc-billing, owner=payments, active=false, limit=5000, fails=0)
[vira_rt] pool_workers=0 | actor_tasks_run=0 | work_steals=0
[vira_rt] active_leaks=0 | total_freed=12
```

---

## Summary & Exercises

1. **Immutable Receiver Enforcement:** Call `readonly.record_auth(true, 1750000500)` in `main()` and run `./vira check` to see how Vira protects immutable bindings from `mut` receiver methods.
2. **Account Lockout Test:** Call `svc.record_auth(false, ...)` three times in a row on a `ServiceAccount` and verify that `svc.active` transitions to `false` automatically.
3. **Second Embedded Component:** Define a `QuotaTracker` struct (`mut: requests_used int = 0` and `pub fn (mut q QuotaTracker) consume(n int)`), embed it in `ServiceAccount`, and verify the promoted `.consume(...)` method works cleanly with `active_leaks=0`.
