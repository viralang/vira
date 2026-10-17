# Vira Enterprise Multi-Package Architecture Example

This example demonstrates how to build and organize a real-world, large-scale application using the **Vira Programming Language**.

## Architecture Overview

```
enterprise_multi_package/
├── vira.mod                    # Project manifest defining root module
├── README.md                   # Architecture & usage documentation
├── src/
│   ├── main.vir                # Application entrypoint & orchestration
│   ├── config/
│   │   └── settings.vir        # 4-quadrant encapsulated configuration
│   ├── models/
│   │   └── cluster.vir         # Domain data models & business rules
│   └── services/
│       └── compute_service.vir # Isolated concurrent actors & SIMD workloads
└── tests/
    └── cluster_test.vir        # Unit & integration test suite
```

## Running the Project

```bash
# 1. Run the entire application across all packages
vira run .

# 2. Run unit tests
vira test tests/

# 3. Build an optimized standalone production binary
vira build -prod -o bin/enterprise_app .
./bin/enterprise_app
```
