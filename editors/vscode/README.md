# Vira Programming Language - VS Code & Cursor Extension

Official Visual Studio Code and Cursor extension for the **Vira Programming Language** (`.vir`).

---

## Features

- **Syntax Highlighting:** Complete TextMate grammar (`vira.tmLanguage.json`) supporting all Vira language features:
  - 4-Quadrant mutability modifiers (`pub:`, `mut:`, `pub mut:`, `default:`)
  - AI tensor primitives (`tensor[f32]`, `@` matrix multiplication)
  - Concurrency & Actor tokens (`actor`, `spawn`, `supervise`, `<-` message passing)
  - Memory management (`arena`)
  - String interpolation (`'Hello, ${name}!'`)
  - Built-in tensor, GGUF/LLM, and native GUI API functions
- **Language Configuration:** Automatic bracket matching, indent rules, surrounding pairs, and comment toggling (`//`, `/* */`).
- **Language Server (LSP) Integration:** Real-time semantic diagnostics, error highlighting, and code formatting via `vira lsp` and `vira fmt`.
- **GitHub Linguist Grammar:** Serves as the upstream grammar for GitHub syntax highlighting (see [LINGUIST.md](LINGUIST.md)).

---

## Installation

### Option 1: Install Prebuilt VSIX Extension

Run in your terminal:

```bash
code --install-extension editors/vscode/vira-language-0.1.1.vsix
```

Or in Cursor:

```bash
cursor --install-extension editors/vscode/vira-language-0.1.1.vsix
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
