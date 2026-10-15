# Tutorial 2: OOP, 4-Quadrant Structs & Composition

> **Difficulty:** Intermediate  
> **Matching Code Example:** [`docs/examples/02_oop_composition.vir`](../examples/02_oop_composition.vir)

---

## 1. Overview

Object-Oriented Programming in Vira is designed to prevent data races and encapsulation leaks. Rather than complex class inheritance hierarchies, Vira uses:
- **4-Quadrant Field Encapsulation**
- **Struct Embedding (Composition)**
- **Explicit Receiver Methods**
- **Type-Associated Static Constructors**

---

## 2. 4-Quadrant Field Encapsulation

In most languages, fields are either public or private. Vira introduces **4-quadrant field encapsulation**, decoupling visibility from mutability within a single struct definition:

```vex
pub struct UserProfile {
	Timestamp            // Embedded struct (promoted)
	hashed_pw string     // Quadrant 1: Private & Immutable (default)

mut:
	login_attempts int   // Quadrant 2: Private & Mutable

pub:
	username string      // Quadrant 3: Public & Read-Only
	email string

pub mut:
	is_active bool       // Quadrant 4: Public & Mutable
	score int
}
```

### Access Matrix

| Quadrant | Modifiers | Read Access | Write Access | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **1. Private & Immutable** | *(default)* | Defining module only | Defining module only (at initialization) | Secrets, internal invariants, salts |
| **2. Private & Mutable** | `mut:` | Defining module only | Defining module only (on `mut` instances) | Internal caches, retry counters |
| **3. Public & Read-Only** | `pub:` | Everywhere | Defining module only | Primary IDs, creation dates, names |
| **4. Public & Mutable** | `pub mut:` | Everywhere | Everywhere (on `mut` instances) | Configuration flags, counters, status |

---

## 3. Struct Embedding & Composition

Vira uses struct embedding instead of class inheritance. When you embed a struct, all of its fields and methods are automatically promoted to the outer struct:

```vex
pub struct Timestamp {
pub:
	created_at i64
pub mut:
	updated_at i64
}

pub fn (mut t Timestamp) touch() {
	t.updated_at = 1760000000
}

pub struct UserProfile {
	Timestamp // Embedded!
pub:
	username string
}
```

Calling promoted methods or fields is completely transparent:

```vex
mut user := UserProfile{ ... }
user.touch()                  // Dispatches to embedded Timestamp.touch()
println('${user.updated_at}') // Reads promoted field Timestamp.updated_at
```

---

## 4. Methods: Value vs Pointer Receivers

Methods specify an explicit receiver parameter before the function name:

### Value Receiver (Read-Only)
Called on both immutable and mutable instances:
```vex
pub fn (u UserProfile) display() string {
	return 'User @${u.username} (${u.email}) [score=${u.score}]'
}
```

### Mutable Pointer Receiver
Modifies the underlying struct; can **only** be invoked on instances declared with `mut`:
```vex
pub fn (mut u UserProfile) award_points(points int) {
	u.score = u.score + points
	u.touch()
}
```

If you try to call a mutable receiver method on an immutable binding, the compiler stops you:
```vex
user := UserProfile{ ... } // immutable!
user.award_points(50)     // ERROR: cannot call mutable receiver method 'award_points()' on immutable binding 'user'
```

---

## 5. Static Constructors

To associate a static factory function with a struct, use `StructName.method_name`:

```vex
pub fn UserProfile.new(username string, email string, secret string) UserProfile {
	return UserProfile{
		Timestamp: Timestamp{ created_at: 1750000000, updated_at: 1750000000 },
		hashed_pw: secret,
		login_attempts: 0,
		username: username,
		email: email,
		is_active: true,
		score: 100
	}
}

// Invoke via:
user := UserProfile.new('alice', 'alice@vex.dev', 'hash123')
```

---

## 6. Running the Code

```powershell
.\vira.exe run docs/examples/02_oop_composition.vir
```

Expected output:
```text
=== 1. STATIC FACTORY CONSTRUCTOR ===
User @alice (alice@vex.dev) [score=100, active=true]
=== 2. PROMOTED FIELDS & METHODS (STRUCT EMBEDDING) ===
Initial timestamp: 1750000000
Updated timestamp after touch(): 1760000000
User @alice (alice@vex.dev) [score=150, active=true]
=== 3. 4-QUADRANT MUTABILITY ===
User active status toggled: false
```

---

## Next Steps

Proceed to [**Tutorial 3: Structural Interfaces & Dynamic Dispatch**](./03_interfaces_and_polymorphism.md) to explore runtime polymorphism without inheritance boilerplate.
