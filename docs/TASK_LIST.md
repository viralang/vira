# Task Checklist: Vira Core, C-Backed Stdlib, `syscall`/`exec` & CLI Stack

## Phase 1: Core Compiler & CLI Upgrades (`internal/*` & `cmd/vira`)
- [x] Add `const` / `pub const` (`TokConst`, `ast.ConstDecl`) across `token`, `lexer`, `ast`, `parser`, `formatter`, `checker`, `cgen`, and `lsp`
- [x] Add `defer` statement (`TokDefer`, `ast.DeferStmt`) across `token`, `lexer`, `ast`, `parser`, `formatter`, `checker`, `autofree`, and `cgen`
- [x] Add octal (`0o755`) and binary (`0b1010`) integer literal lexing & C99 lowering
- [x] Add primitive numeric casts (`u8(x)`, `int(x)`, `i64(x)`, `f32(x)`, `f64(x)`) and `i64` unary `-`
- [x] Emit custom & nested slices (`[]MyStruct`, `[][]T`) with deep `_clone`/`_free` destructors + `.slice(start, end)` sub-slicing
- [x] Register `print`, `eprint`, `time_sleep_ms`, and extended `string` methods (`.to_f64()`, `.to_f32()`, `.replace()`, `.last_index_of()`, `.trim_left()`, `.trim_right()`, `.repeat()`, `.lines()`, `.byte_at()`)
- [x] Upgrade `main` emission to `int main(int argc, char **argv)` (`_vira_rt_init_args` / `_vira_rt_free_args`) and add `-quiet-rt` CLI flag in `cmd/vira/main.go`

## Phase 2: C99 Runtime Intrinsics (`runtime/vira_rt.h`)
- [x] `argc`/`argv` capture (`vira_os_args`), `vira_os_exit`, `vira_os_set_quiet_rt` / `VIRA_RT_STATS=0`
- [x] `<stdio.h>` & `<string.h>` & `<stdlib.h>` shims (`print`/`eprint`, `stdin`/`stdout`/`stderr`, dynamic `read_line`, `strtod`/`strtof`/`strtoll(base)`, `replace`/`last_index_of`/`trim_left`/`trim_right`/`repeat`/`lines`/`byte_at`, `qsort` slice helpers, RNG)
- [x] `<ctype.h>` & `<math.h>` & `<time.h>` shims (`vira_ctype_*`, `vira_math_*`, `vira_time_*` + `strftime`/`localtime`/`gmtime`/`mktime`)
- [x] Cross-platform OS & Linux container `vira_sys_*` shims (`open`/`read`/`write`/`close`/`lseek`/`pipe`/`dup`/`dup2`/`isatty`/`stat`/`mkdir`/`readdir`/`errno`/`strerror`, Linux `mount`/`umount2`/`chroot`/`mknod`/`makedev`/`unshare`/`sethostname`/`getuid`/`getgid`, async-signal-safe signal pipe, and split `proc_start`/`proc_wait` + `SYS_clone` + `combined_output`)

## Phase 3: Pure-Vira Standard Library Packages (`packages/*`)
- [x] `packages/math` + `tests/math_test.vir` (Verified 100% `active_leaks=0`)
- [x] `packages/ctype` + `tests/ctype_test.vir` (Verified 100% `active_leaks=0`)
- [x] `packages/strings` + `tests/strings_test.vir` (Verified 100% `active_leaks=0`)
- [x] `packages/time` + `tests/time_test.vir` (Verified 100% `active_leaks=0`)
- [x] `packages/io` expansion (`stdin`/`stdout`/`stderr`, `read_line`, `prompt`, `SEEK_*`, `rename`, `remove`, `temp_file`) + `tests/io_test.vir` (Verified 100% `active_leaks=0`)
- [x] `packages/syscall` + `tests/syscall_test.vir` (Verified 100% `active_leaks=0`)
- [x] `packages/os` + `tests/os_test.vir` (Verified 100% `active_leaks=0`)
- [x] `packages/exec` + `tests/exec_test.vir` (Verified 100% `active_leaks=0`)
- [x] `packages/flag` + `tests/flag_test.vir` (Verified 100% `active_leaks=0`)
- [x] `packages/log` + `tests/log_test.vir` (Verified 100% `active_leaks=0`)

## Phase 4: Tutorials, Examples, Docs & VS Code / LSP (`0.1.9`)
- [x] Tutorial 31 (`31_basic_cli_application_development.md`) + `docs/examples/34_basic_cli_app.vir` (Verified 100% `active_leaks=0`)
- [x] Tutorial 32 (`32_advanced_cli_flags_syscall_and_exec.md`) + `docs/examples/35_advanced_cli_flags_syscall_exec.vir` + `docs/examples/36_linux_container_runtime_reference.vir` (Verified 100% `active_leaks=0`)
- [x] Update `docs/BUILTINS_AND_API_REFERENCE.md`, `docs/LANGUAGE_GUIDE.md`, root `LANGUAGE_GUIDE.md`, and `README.md`
- [x] Update `internal/lsp/server.go` + `editors/vscode` (`0.1.9` VSIX packaged)
- [x] Full verification: `go test -count=1 ./...` & `vira test` across all 10 stdlib packages (`active_leaks=0`)
