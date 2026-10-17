# Tutorial 24: Scopes & Multi-Package Architecture for Large Projects

> **Difficulty:** Advanced  
> **Matching Code Example:** [`docs/examples/23_multi_package_architecture.vir`](../examples/23_multi_package_architecture.vir)  
> **Enterprise Multi-Directory Project:** [`examples/enterprise_multi_package/`](../../examples/enterprise_multi_package/)

---

## 1. Overview

As applications grow beyond single-file prototypes into production enterprise systems, maintaining clean separation of concerns, preventing name collisions, and enforcing strict data encapsulation become essential.

Vira approaches large-scale software engineering through two complementary systems:
1. **The 5-Tier Deterministic Scoping Model:** Eliminates data races, circular imports, and global variable leakage.
2. **Directory-as-Package Multi-Module Architecture:** Allows dividing projects into clean subpackages (`config/`, `models/`, `services/`, `ui/`) with automatic recursive compilation.

---

## 2. The 5-Tier Scoping Hierarchy

```
┌────────────────────────────────────────────────────────────────────────┐
│                        VIRA SCOPING HIERARCHY                          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Module Scope      ──► module name (exported with pub fn / pub struct)│
│    │                                                                   │
│    ├── 2. Struct 4-Quadrant Scope ──► default, mut:, pub:, pub mut:     │
│    │                                                                   │
│    ├── 3. Actor Isolation Scope   ──► Separate mailbox & thread heap   │
│    │                                                                   │
│    ├── 4. Function Scope          ──► Autofree LIFO destructors        │
│    │                                                                   │
│    └── 5. Block / Arena Scope     ──► O(1) Slab bulk wipe on exit      │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Module Scope
Every `.vir` file belongs to a declared module:
```vira
module enterprise_app
```
- Symbols declared **without** `pub` are package-internal.
- Symbols declared **with** `pub` (e.g., `pub struct`, `pub fn`, `pub actor`) are exported and available across all files and packages in the project.

### 2. Struct 4-Quadrant Scope
Vira structs enforce fine-grained field encapsulation across four distinct quadrants:
```vira
pub struct AppConfig {
    // Quadrant 1 (default): Private & Immutable
    // Accessible only within methods of AppConfig; read-only.
    secret_salt string

mut:
    // Quadrant 2: Private & Mutable
    // Modifiable only by methods of AppConfig.
    reload_count int

pub:
    // Quadrant 3: Public & Read-Only
    // Accessible from any package across the project, but immutable from outside.
    app_name string
    environment string

pub mut:
    // Quadrant 4: Public & Mutable
    // Freely readable and modifiable across any package.
    debug_mode bool
}
```

### 3. Actor Isolation Scope
Actors communicate strictly through **asynchronous message passing (`<-`)**:
- Actors do not share mutable memory with any other thread.
- Memory allocated inside an actor method is isolated to the actor's thread pool context and freed deterministically upon task completion.

### 4. Function Scope & Autofree
Variables allocated within a function are tracked by compile-time escape analysis:
- Local variables that do not escape are assigned LIFO destructors (`autofree`) at scope exit.
- Results returned to the caller are promoted to the caller's stack frame with zero unnecessary heap allocations.

### 5. Bump Arena Scope (`arena { ... }`)
When running batch workloads, machine learning inference, or large graph traversals:
```vira
arena {
    t1 := tensor_fill(64, 64, 1.5)
    t2 := tensor_fill(64, 64, 2.0)
    out := t1 @ t2
    println('Computed Sum: ${tensor_sum(out)}')
}
// Entire memory slab wiped in O(1) time here!
```

---

## 3. Scaffolding Modern Projects: `vira init`

Vira includes a built-in project generator that creates an enterprise-ready repository structure:

```bash
# Initialize a new project
vira init my_service
```

This creates the canonical modern project layout:

```
my_service/
├── vira.mod               # Project manifest
├── .gitignore             # Pre-configured ignores for build/, bin/, .vir/
├── README.md              # Project documentation and build commands
├── src/
│   ├── main.vir           # Application orchestration & entry point
│   ├── config/
│   │   └── settings.vir   # 4-quadrant encapsulated configuration
│   ├── models/
│   │   └── data.vir       # Domain data models & business rules
│   └── services/
│       └── compute.vir    # Business logic, actors, & pipelines
└── tests/
    └── app_test.vir       # Unit & integration test suite
```

---

## 4. Multi-Package Architecture: Step-by-Step

Let's look at how the packages interact in a large system:

### 1. Configuration Package (`src/config/settings.vir`)
```vira
module my_service

pub struct AppConfig {
    secret_salt string
mut:
    reload_count int
pub:
    app_name string
    version string
pub mut:
    debug_mode bool
}

pub fn AppConfig.default() AppConfig {
    return AppConfig{
        secret_salt: 'vault_sec_99',
        reload_count: 0,
        app_name: 'ClusterService',
        version: '1.0.0',
        debug_mode: true
    }
}
```

### 2. Domain Models (`src/models/data.vir`)
```vira
module my_service

pub struct WorkItem {
pub:
    id int
    title string
mut:
    completed bool
}

pub fn WorkItem.new(id int, title string) WorkItem {
    return WorkItem{
        id: id,
        title: title,
        completed: false
    }
}

pub fn (mut w WorkItem) mark_done() {
    w.completed = true
}
```

### 3. Business Services & Actors (`src/services/compute.vir`)
```vira
module my_service

pub actor ComputeService {
pub:
    worker_id int
mut:
    batches_run int
}

pub fn ComputeService.new(worker_id int) ComputeService {
    return ComputeService{
        worker_id: worker_id,
        batches_run: 0
    }
}

pub fn (mut s ComputeService) process(item WorkItem) f32 {
    s.batches_run += 1
    weights := tensor_fill(32, 32, 0.5)
    inputs := tensor_fill(32, 32, 1.0)
    out := weights @ inputs
    return tensor_sum(out)
}
```

### 4. Main Orchestration (`src/main.vir`)
```vira
module my_service

fn main() {
    cfg := AppConfig.default()
    println('=== ${cfg.app_name} (v${cfg.version}) ===')

    mut item := WorkItem.new(101, 'Data Ingestion Batch')
    mut service := ComputeService.new(1)

    // Asynchronous actor message passing
    fut := service <- process(item)
    score := fut.await() or { 0.0 }

    item.mark_done()
    println('WorkItem #${item.id} completed: ${item.completed} (Score=${score})')
}
```

---

## 5. Compiling and Building

The Vira compiler automatically walks subdirectories recursively to collect and link all packages:

```bash
# 1. Run the entire application across all packages
vira run .

# 2. Run automated unit tests
vira test tests/

# 3. Build standalone native production binary
vira build -prod -o bin/app .
./bin/app
```

---

## 6. Best Practices for Large Vira Projects

1. **Use Static Factory Methods:** Instead of constructing structs with raw literals everywhere, expose `StructName.new(...)` or `StructName.default()`.
2. **Encapsulate Mutable State:** Keep sensitive internal state in the `mut:` quadrant and expose mutation only through receiver methods.
3. **Isolate Concurrent Logic with Actors:** Use `actor` for background workers, queues, and network consumers to avoid shared-memory deadlocks.
4. **Use Bump Arenas for Intermediate Bursts:** Inside computationally intensive pipelines, wrap matrix and tensor calculations in `arena { ... }` for O(1) bulk teardown.
