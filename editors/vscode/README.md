# Vira Programming Language - VS Code & Cursor Extension

Official Visual Studio Code and Cursor extension for the **Vira Programming Language** (`.vir`).

---

## Features

- **Syntax Highlighting:** Complete TextMate grammar (`vira.tmLanguage.json`) supporting all Vira language features:
  - First-class `import` statements, package aliases (`import ... as ...`), and module declarations (`module <name>`)
  - Qualified type signatures (`pkg.Type`) and qualified static calls
  - Multi-branch `switch`, `case`, and `default` control flow
  - 4-Quadrant mutability modifiers (`pub:`, `mut:`, `pub mut:`, `default:`)
  - Native collections (`[N]T` stack arrays, `[]T` dynamic slices)
  - 1D–4D strided AI tensor primitives (`tensor[f32]`, `tensor_fill_3d`, `tensor_fill_4d`, `.reshape_2d/3d/4d()`, `.transpose()`, `@` batched matrix multiplication)
  - Concurrency & Actor tokens (`actor`, `spawn`, `supervise`, `<-` message passing, `.await()`)
  - Memory management (`arena`)
  - String interpolation (`'Hello, ${name}!'`)
  - Built-in tensor, GGUF/LLM, standard library (`io`, `net`, `http`, `json`, `viraserve`), and native GUI API functions
- **Multi-File Import Resolution & Cross-File Navigation:**
  - Understands `import "./file.vir"`, `import pkg`, and `import pkg as alias`
  - Cross-file symbol resolution eliminates false diagnostics
  - **Go to Definition (`F12`):** Jump directly from an imported symbol or embedded struct to its declaration file
  - **Rich Hover Documentation:** Inspect struct quadrants, methods, embedded promotions, keywords, 1D–4D tensor primitives, and source file origins
  - **Smart Autocomplete & Dot Completion:** Trigger completion after `.` (e.g., `user.`, `slice.`, `tensor.`, `json.`, `net.`) to see fields, methods, module exports, and promoted members from embedded structs across imported modules
  - **Document Symbols & Outline:** Full breadcrumbs and symbol outline for modules, imports, structs, interfaces, methods, and embeds
- **Language Configuration:** Automatic bracket matching, smart case/default decrease indentation, surrounding pairs, and comment toggling (`//`, `/* */`).
- **Snippets:** Built-in productivity snippets for loops (`forin`), constants (`const`), cleanup (`defer`), collections (`arr`, `slice`, `push`, `getor`), control flow (`switch`), modules (`mod`, `impas`), CLI flags (`flagset`), structured logging (`logger`), process execution (`exec`), 3D/4D tensors (`tensor3d`, `tensor4d`), and RFC 8259 JSON building (`jsonb`).
- **Language Server (LSP) Integration:** Real-time semantic diagnostics, error highlighting, and code formatting via `vira lsp` and `vira fmt`.
- **GitHub Linguist Grammar:** Serves as the upstream grammar for GitHub syntax highlighting (see [LINGUIST.md](LINGUIST.md)).

---

## Installation

### Option 1: Install Prebuilt VSIX Extension

Run in your terminal:

```bash
code --install-extension editors/vscode/vira-language-0.2.0.vsix
```

Or in Cursor:

```bash
cursor --install-extension editors/vscode/vira-language-0.2.0.vsix
```

### Option 2: Install from Source

```bash
cd editors/vscode
npm install
npm run package # builds vira-language-*.vsix
```

---

## Getting Started

1. Open any `.vir` source file.
2. The language mode will automatically activate as **Vira**.
3. Format with `Shift+Alt+F` or run `vira fmt <file>` from the integrated terminal.

---

## GitHub Linguist Support

See **[LINGUIST.md](LINGUIST.md)** for instructions on repository-level `.gitattributes` configuration and global registration in `github-linguist/linguist`.

---

## License

MIT License.
