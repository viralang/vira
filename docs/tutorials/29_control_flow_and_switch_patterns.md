# Tutorial 29: Control Flow & Switch Statement Patterns

In this tutorial, you will explore Vira's multi-branch control flow statement: the `switch` statement.

---

## 1. Overview & Key Characteristics

Vira's `switch` statement simplifies complex branching logic compared to chained `if ... else if` trees:
1. **Implicit Break:** Each case breaks automatically on completion. No accidental fallthrough bugs.
2. **First-Class Types:** Compare integers, floating points, enums, booleans, and native `string` types.
3. **Multiple Matches:** Match multiple values per case using comma separation (`case 1, 2, 3:` or `case 'GET', 'HEAD':`).
4. **Tagless Switches:** Evaluate clean boolean condition ladders without a tag expression.
5. **Autofree Memory Safety:** Variables declared within cases are scoped and deterministically cleaned up at block exit.

---

## 2. Integer & Enum Switch

```vira
fn http_category(code int) string {
    mut cat := ''
    switch code {
    case 200, 201, 204:
        cat = 'Success'
    case 301, 302, 304:
        cat = 'Redirection'
    case 400, 401, 403, 404:
        cat = 'Client Error'
    case 500, 502, 503:
        cat = 'Server Error'
    default:
        cat = 'Custom or Unofficial'
    }
    return cat
}
```

---

## 3. String Switch

In systems and networking code, switching directly on strings is essential for protocol dispatch and command parsing:

```vira
fn execute_cli_command(cmd string, arg string) {
    switch cmd {
    case 'build', 'compile':
        println('Building binary with target: ${arg}')
    case 'run', 'start':
        println('Executing application: ${arg}')
    case 'test', 'check':
        println('Running test suites matching: ${arg}')
    case 'clean':
        println('Purging build artifacts')
    default:
        println('Unrecognized command: ${cmd}')
    }
}
```

---

## 4. Tagless Switch (Boolean Conditions)

A tagless switch has no subject expression and evaluates conditions in top-down order (equivalent to `switch true` in Go):

```vira
fn score_grade(score int) string {
    mut grade := ''
    switch {
    case score >= 90:
        grade = 'A'
    case score >= 80:
        grade = 'B'
    case score >= 70:
        grade = 'C'
    case score >= 60:
        grade = 'D'
    default:
        grade = 'F'
    }
    return grade
}
```

---

## 5. Scope Isolation & Memory Safety

Each case branch in Vira creates an isolated lexical scope. Variables declared within a case do not collide with adjacent cases, and their destructors are injected deterministically by the autofree engine:

```vira
switch msg_type {
case 'json':
    payload := parse_json(raw)
    println('Parsed JSON payload: ${payload}')
    // payload is freed here before case exit
case 'xml':
    xml_doc := parse_xml(raw)
    println('Parsed XML document: ${xml_doc}')
    // xml_doc is freed here before case exit
default:
    println('Raw text message')
}
```
