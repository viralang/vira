# GitHub Linguist Registration Guide for Vira

This guide documents the process of registering the **Vira Programming Language** globally in [`github-linguist/linguist`](https://github.com/github-linguist/linguist), enabling native `.vir` syntax highlighting and language metrics across all of GitHub.

---

## Prerequisites (GitHub Acceptance Criteria)

GitHub Linguist merges new language grammars once the language has gained adoption across GitHub (typically ~200 unique public repositories or active ecosystem usage).

Until that threshold is met, repository owners can use the **Tier 1 Repository-Level Override** (`.gitattributes`) provided below.

---

## Tier 1: Immediate Recognition via `.gitattributes` (Active Now)

In your repository root (`.gitattributes`):

```gitattributes
# 1. Tell GitHub Linguist to syntax-highlight .vir files and count them in the repo language bar
*.vir linguist-detectable=true
*.vir linguist-language=V

# Also recognize .vex files
*.vex linguist-detectable=true
*.vex linguist-language=V

# 2. Exclude generated C99 outputs, packages, and docs from skewing the language percentage bar
build/** linguist-generated=true
dist/** linguist-generated=true
scratch/** linguist-generated=true
*.md linguist-documentation=true
docs/** linguist-documentation=true
```

> **Why `linguist-language=V`?**  
> Vira shares syntax conventions with V (`fn`, `pub`, `mut`, `struct`, `interface`, `${...}` string interpolation, `match`, `or {}`), providing clean, instant syntax coloring on GitHub immediately.

---

## Tier 2: Official TextMate Grammar (Maintained in this Repo)

The official TextMate grammar is maintained at:
- **Location:** [`editors/vscode/syntaxes/vira.tmLanguage.json`](syntaxes/vira.tmLanguage.json)
- **Scope Name:** `source.vira`
- **Extensions:** `.vir`, `.vex`

This grammar powers:
1. The **Vira VS Code / Cursor Extension** (`vira-language-0.1.1.vsix`).
2. The official upstream grammar source for **GitHub Linguist**.

---

## Tier 3: Global Registration in `github-linguist/linguist`

When ready to submit the Pull Request to [github-linguist/linguist](https://github.com/github-linguist/linguist), include the following three components:

### 1. Update `lib/linguist/languages.yml`

Add the Vira entry into alphabetical order in `languages.yml`:

```yaml
Vira:
  type: programming
  color: "#7C3AED"
  extensions:
    - ".vir"
  tm_scope: source.vira
  ace_mode: golang
  codemirror_mode: go
  codemirror_mime_type: text/x-go
  language_id: 994821042
```

> *Note: Run `script/set-language-ids` in the linguist repo to ensure a unique 32-bit integer `language_id`.*

### 2. Add TextMate Grammar Submodule

Add the grammar tracking the `viralang/vira` repository in `vendor/README.md` or git submodule:
- **Grammar Repo:** `https://github.com/viralang/vira`
- **Path:** `editors/vscode/syntaxes/vira.tmLanguage.json`
- **License:** MIT

### 3. Add Sample Code

Create `samples/Vira/example.vir` with an idiomatic Vira program showcasing actors, tensors, and 4-quadrant structs:

```vir
module main

pub struct TensorNode {
pub:
    id: int
mut:
    weight: tensor[f32]
}

actor Worker {
    inbox: chan int
}

fn main() {
    node := TensorNode{ id: 1, weight: tensor_fill(2, 2, 1.0f32) }
    println('Initialized Node: ${node.id}')
}
```
