package lsp_test

import (
	"bytes"
	"encoding/json"
	"fmt"
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

