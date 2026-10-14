package lsp_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vexlang/vex/internal/lsp"
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
			"uri":  "file:///main.vex",
			"text": "module main\n\npub fn add(a int, b int) int {\n\treturn a + b\n}\n",
		},
	}, nil)

	// Send hover
	sendRPC(&in, "textDocument/hover", map[string]interface{}{
		"textDocument": map[string]interface{}{
			"uri": "file:///main.vex",
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
}
