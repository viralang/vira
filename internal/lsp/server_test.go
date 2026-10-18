package lsp_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/viralang/vira/internal/lsp"
)

func sendRPC(w *bytes.Buffer, method string, params interface{}, id interface{}) {
	body, _ := json.Marshal(map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  method,
		"params":  params,
	})
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(body))
	w.WriteString(header)
	w.Write(body)
}

func TestLSPLifecycleAndDiagnostics(t *testing.T) {
	var in bytes.Buffer
	var out bytes.Buffer

	// Send initialize
	sendRPC(&in, "initialize", map[string]interface{}{}, 1)

	// Send textDocument/didOpen with valid code
	sendRPC(&in, "textDocument/didOpen", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri":  "file:///main.vir",
			"text": "module main\n\npub fn add(a int, b int) int {\n\treturn a + b\n}\n",
		},
	}, nil)

	// Send hover
	sendRPC(&in, "textDocument/hover", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri": "file:///main.vir",
		},
		"position": map[string]interface{}{
			"line":      2,
			"character": 5,
		},
	}, 2)

	// Send shutdown
	sendRPC(&in, "shutdown", map[string]interface{}{}, 3)

	server := lsp.NewServer(&in, &out)
	err := server.Run()
	if err != nil {
		t.Fatalf("unexpected error in server.Run: %v", err)
	}

	res := out.String()
	if len(res) == 0 {
		t.Errorf("expected LSP response output, got empty string")
	}
	if strings.Contains(res, `"diagnostics":null`) {
		t.Errorf("diagnostics must never be null (breaks vscode-languageclient): %s", res)
	}
	if !strings.Contains(res, `"diagnostics":[]`) {
		t.Errorf("expected clean file to output 'diagnostics:[]', got: %s", res)
	}
}

func TestLSPGotoDefinition(t *testing.T) {
	var in bytes.Buffer
	var out bytes.Buffer

	sendRPC(&in, "initialize", map[string]interface{}{}, 1)

	code := "module main\n\npub struct Invoice {\n\tid string\n}\n\nfn main() {\n\tinv := Invoice{ id: '101' }\n\t_ := inv\n}\n"
	sendRPC(&in, "textDocument/didOpen", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri":  "file:///main.vir",
			"text": code,
		},
	}, nil)

	// Jump to definition on `Invoice` at line 7, col 9
	sendRPC(&in, "textDocument/definition", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri": "file:///main.vir",
		},
		"position": map[string]interface{}{
			"line":      7,
			"character": 9,
		},
	}, 2)

	sendRPC(&in, "shutdown", map[string]interface{}{}, 3)

	server := lsp.NewServer(&in, &out)
	if err := server.Run(); err != nil {
		t.Fatalf("server.Run failed: %v", err)
	}

	res := out.String()
	if !strings.Contains(res, "file:///main.vir") {
		t.Errorf("expected location containing URI in definition response: %s", res)
	}
	if !strings.Contains(res, "\"definitionProvider\":true") {
		t.Errorf("expected definitionProvider capability in initialize response: %s", res)
	}
}

func TestLSPGotoDefinition_NotFoundReturnsNullResult(t *testing.T) {
	var in bytes.Buffer
	var out bytes.Buffer

	sendRPC(&in, "initialize", map[string]interface{}{}, 1)

	code := "module main\n\nfn main() {\n}\n"
	sendRPC(&in, "textDocument/didOpen", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri":  "file:///main.vir",
			"text": code,
		},
	}, nil)

	// Jump to definition on whitespace at line 0, col 0
	sendRPC(&in, "textDocument/definition", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri": "file:///main.vir",
		},
		"position": map[string]interface{}{
			"line":      0,
			"character": 0,
		},
	}, 99)

	sendRPC(&in, "shutdown", map[string]interface{}{}, 100)

	server := lsp.NewServer(&in, &out)
	if err := server.Run(); err != nil {
		t.Fatalf("server.Run failed: %v", err)
	}

	res := out.String()
	// Must contain {"jsonrpc":"2.0","id":99,"result":null}
	if !strings.Contains(res, `"id":99,"result":null`) && !strings.Contains(res, `"result":null,"id":99`) {
		t.Errorf("expected response to contain explicit null result for id 99, got: %s", res)
	}
	// Must contain shutdown response {"id":100,"result":null}
	if !strings.Contains(res, `"id":100,"result":null`) && !strings.Contains(res, `"result":null,"id":100`) {
		t.Errorf("expected shutdown response to contain explicit null result, got: %s", res)
	}
}

func TestLSPImports_CompletionAndDefinition(t *testing.T) {
	tempDir := t.TempDir()
	tsPath := filepath.Join(tempDir, "timestamp.vir")
	userPath := filepath.Join(tempDir, "user.vir")

	tsCode := `module main

pub struct AuditMeta {
pub:
	created_at int
}

pub fn (mut a AuditMeta) touch() {
	a.created_at = 100
}
`
	if err := os.WriteFile(tsPath, []byte(tsCode), 0644); err != nil {
		t.Fatalf("failed to write timestamp.vir: %v", err)
	}

	userCode := `module main

import "./timestamp.vir"

pub struct User {
	AuditMeta
pub:
	name string
}

pub fn (u User) display() {
}

fn test_func() {
	user := User{ name: "Alice" }
	user.
}
`
	if err := os.WriteFile(userPath, []byte(userCode), 0644); err != nil {
		t.Fatalf("failed to write user.vir: %v", err)
	}

	userURI := "file:///" + filepath.ToSlash(userPath)
	tsURI := "file:///" + filepath.ToSlash(tsPath)
	_ = tsURI

	var in bytes.Buffer
	var out bytes.Buffer

	sendRPC(&in, "initialize", map[string]interface{}{}, 1)

	// Open user.vir
	sendRPC(&in, "textDocument/didOpen", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri":  userURI,
			"text": userCode,
		},
	}, nil)

	// Completion on `user.` (line 15, col 6)
	sendRPC(&in, "textDocument/completion", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri": userURI,
		},
		"position": map[string]interface{}{
			"line":      15,
			"character": 6,
		},
	}, 2)

	// Hover on `AuditMeta` in `User` (line 5, col 4)
	sendRPC(&in, "textDocument/hover", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri": userURI,
		},
		"position": map[string]interface{}{
			"line":      5,
			"character": 4,
		},
	}, 3)

	// Go to definition on `AuditMeta` (line 5, col 4) -> should jump to timestamp.vir
	sendRPC(&in, "textDocument/definition", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri": userURI,
		},
		"position": map[string]interface{}{
			"line":      5,
			"character": 4,
		},
	}, 4)

	// Document symbols on user.vir
	sendRPC(&in, "textDocument/documentSymbol", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri": userURI,
		},
	}, 5)

	sendRPC(&in, "shutdown", map[string]interface{}{}, 6)

	server := lsp.NewServer(&in, &out)
	if err := server.Run(); err != nil {
		t.Fatalf("server.Run failed: %v", err)
	}

	res := out.String()

	// 1. Completion verification:
	// Should contain "created_at" (promoted field), "touch" (promoted method), "name" (own field), "display" (own method)
	if !strings.Contains(res, `"created_at"`) {
		t.Errorf("expected completion to include promoted field 'created_at': %s", res)
	}
	if !strings.Contains(res, `"touch"`) {
		t.Errorf("expected completion to include promoted method 'touch': %s", res)
	}
	if !strings.Contains(res, `"name"`) {
		t.Errorf("expected completion to include field 'name': %s", res)
	}
	if !strings.Contains(res, `"display"`) {
		t.Errorf("expected completion to include method 'display': %s", res)
	}

	// 2. Hover verification:
	if !strings.Contains(res, "timestamp.vir") && !strings.Contains(res, "AuditMeta") {
		t.Errorf("expected hover to contain AuditMeta / timestamp.vir: %s", res)
	}

	// 3. Definition verification:
	// Should point to timestamp.vir
	cleanBase := filepath.Base(tsPath)
	if !strings.Contains(res, cleanBase) {
		t.Errorf("expected definition result to point to %s, got: %s", cleanBase, res)
	}

	// 4. Document symbols verification:
	if !strings.Contains(res, `"User"`) || !strings.Contains(res, `"display"`) {
		t.Errorf("expected document symbols for User and display: %s", res)
	}
}

func TestLSPOOPCompositionExampleWithURLEncodedURI(t *testing.T) {
	oopPath, err := filepath.Abs("../../docs/examples/02_oop_composition.vir")
	if err != nil {
		t.Fatalf("failed to resolve path to 02_oop_composition.vir: %v", err)
	}
	contentBytes, err := os.ReadFile(oopPath)
	if err != nil {
		t.Fatalf("failed to read 02_oop_composition.vir: %v", err)
	}
	code := string(contentBytes)

	// Construct VS Code Windows URI format with %3A
	cleanSlash := filepath.ToSlash(oopPath)
	encodedURI := "file:///" + strings.Replace(cleanSlash, ":", "%3A", 1)

	var in bytes.Buffer
	var out bytes.Buffer

	sendRPC(&in, "initialize", map[string]interface{}{}, 1)

	// Send didOpen with %3A URI
	sendRPC(&in, "textDocument/didOpen", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri":  encodedURI,
			"text": code,
		},
	}, nil)

	// Hover on Timestamp at line 7, col 2
	sendRPC(&in, "textDocument/hover", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri": encodedURI,
		},
		"position": map[string]interface{}{
			"line":      7,
			"character": 2,
		},
	}, 2)

	sendRPC(&in, "shutdown", map[string]interface{}{}, 3)

	server := lsp.NewServer(&in, &out)
	if err := server.Run(); err != nil {
		t.Fatalf("server.Run failed: %v", err)
	}

	res := out.String()

	// Must NOT contain any checker errors about Timestamp, updated_at, or touch
	if strings.Contains(res, "unknown field 'Timestamp'") {
		t.Errorf("unexpected 'unknown field Timestamp' error in diagnostics: %s", res)
	}
	if strings.Contains(res, "has no field 'updated_at'") {
		t.Errorf("unexpected 'has no field updated_at' error in diagnostics: %s", res)
	}
	if strings.Contains(res, "method 'touch' not found") {
		t.Errorf("unexpected 'method touch not found' error in diagnostics: %s", res)
	}

	// Must have clean diagnostics:[]
	if !strings.Contains(res, `"diagnostics":[]`) {
		t.Errorf("expected clean diagnostics [] on 02_oop_composition.vir, got: %s", res)
	}

	// Hover should reflect imported Timestamp
	if !strings.Contains(res, "Timestamp") || !strings.Contains(res, "02_timestamp.vir") {
		t.Errorf("expected hover to show Timestamp imported from 02_timestamp.vir, got: %s", res)
	}
}



