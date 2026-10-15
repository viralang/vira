# Tutorial 4: Zero-Cost Error Handling with Results (`!T`)

> **Difficulty:** Intermediate  
> **Matching Code Example:** [`docs/examples/04_result_handling.vir`](../examples/04_result_handling.vir)

---

## 1. Overview

Exceptions in traditional languages break control flow, introduce invisible exit paths, and carry heavy runtime stack-unwinding penalties.

Vira replaces exceptions with **mandatory, zero-cost `!T` Result types**:
- Functions declare potential failure in their type signature (`!i64`, `!string`).
- Call sites **must** handle the Result using an `or { ... }` block or propagate it with `!`.
- The compiler enforces error handling at compile-time—unhandled Results abort the build.

---

## 2. Returning a Result (`!T`)

Prefix a return type with an exclamation mark (`!`) to indicate that the function returns a Result:

```vex
pub fn (mut a BankAccount) withdraw(amount i64) !i64 {
	if amount <= 0 {
		return error('withdrawal amount must be greater than zero')
	}
	if amount > a.balance {
		return error('insufficient funds in account ${a.account_id}')
	}
	a.balance = a.balance - amount
	return a.balance // Automatically wrapped in success Result
}
```

The built-in `error('...')` function constructs an error Result with the provided message.

---

## 3. Unwrapping with `or { ... }`

When calling a function that returns `!T`, you must provide an `or { ... }` block to handle potential failure:

```vex
new_bal := acc.withdraw(15000) or {
	eprintln('Unexpected failure: ${err}')
	return
}
```

### The Implicit `err` Variable
Inside the `or { ... }` block, the error message returned by `error('...')` is automatically bound to the implicit variable `err`.

---

## 4. Providing Fallback Default Values

If you do not want to exit early, you can return a fallback value directly from the `or { ... }` block:

```vex
// If withdraw() fails, failed_bal will be set to 0
failed_bal := acc.withdraw(999999) or {
	println('Caught business logic error: ${err}')
	0 // Evaluates to fallback default value
}
```

You can also return an existing variable or computed expression:

```vex
zero_bal := acc.withdraw(0) or {
	println('Validation rejected: ${err}')
	acc.balance // Retain current balance as fallback
}
```

---

## 5. Compile-Time Enforcement

If you attempt to call a function returning `!T` without an `or { ... }` block:

```vex
bad := acc.withdraw(500) // ERROR!
```

The Vira compiler refuses to generate code and highlights the unhandled Result:

```
error[SEMANTIC]: unhandled Result '!i64' from 'BankAccount.withdraw()': handle it with 'or { ... }' or propagate with '!'
  --> main.vir:25:15
   |
25 |     bad := acc.withdraw(500)
   |            ^^^^^^^^^^^^^^^^^
```

This guarantees that errors can never be silently ignored in production.

---

## 6. Running the Code

```powershell
.\vira.exe run docs/examples/04_result_handling.vir
```

Expected output:
```text
=== 1. HANDLING SUCCESSFUL RESULT (!T) ===
Withdrawal approved! New balance: 35000 cents
=== 2. HANDLING ERROR RESULT WITH or { ... } ===
Caught business logic error: insufficient funds in account acc_001
Fallback balance returned: 0
=== 3. ZERO/NEGATIVE VALIDATION WITH or { ... } ===
Validation rejected: withdrawal amount must be greater than zero
Preserved balance: 35000 cents
[vira_rt] active_leaks=0 | total_freed=6
```

---

## Next Steps

Proceed to [**Tutorial 5: Deterministic Autofree & Bump Arenas**](./05_autofree_and_bump_arenas.md) to understand how Vira achieves automatic memory management without a garbage collector.
