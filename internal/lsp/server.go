package lsp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/vexlang/vex/internal/checker"
	"github.com/vexlang/vex/internal/formatter"
	"github.com/vexlang/vex/internal/parser"
	"github.com/vexlang/vex/internal/source"
)

type Server struct {
	in  *bufio.Reader
	out io.Writer
	docs map[string]string // URI -> document content
}

func NewServer(in io.Reader, out io.Writer) *Server {
	return &Server{
		in:   bufio.NewReader(in),
		out:  out,
		docs: make(map[string]string),
	}
}

type jsonrpcMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  interface{}     `json:"result,omitempty"`
	Error   *jsonrpcError   `json:"error,omitempty"`
}

type jsonrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type Position struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

type Range struct {
	Start Position `json:"start"`
	End   Position `json:"end"`
}

type Diagnostic struct {
	Range    Range  `json:"range"`
	Severity int    `json:"severity"` // 1 = Error, 2 = Warning, 3 = Info, 4 = Hint
	Source   string `json:"source"`
	Message  string `json:"message"`
}

type PublishDiagnosticsParams struct {
	URI         string       `json:"uri"`
	Diagnostics []Diagnostic `json:"diagnostics"`
}

func (s *Server) Run() error {
	for {
		msg, err := s.readMessage()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		s.handleMessage(msg)
	}
}

func (s *Server) readMessage() (*jsonrpcMessage, error) {
	var contentLength int
	for {
		line, err := s.in.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		if strings.HasPrefix(line, "Content-Length:") {
			val := strings.TrimSpace(strings.TrimPrefix(line, "Content-Length:"))
			contentLength, _ = strconv.Atoi(val)
		}
	}

	body := make([]byte, contentLength)
	if _, err := io.ReadFull(s.in, body); err != nil {
		return nil, err
	}

	var msg jsonrpcMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		return nil, err
	}
	return &msg, nil
}

func (s *Server) writeResponse(msg *jsonrpcMessage) {
	msg.JSONRPC = "2.0"
	bytes, _ := json.Marshal(msg)
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(bytes))
	_, _ = io.WriteString(s.out, header)
	_, _ = s.out.Write(bytes)
}

func (s *Server) sendNotification(method string, params interface{}) {
	pBytes, _ := json.Marshal(params)
	msg := &jsonrpcMessage{
		JSONRPC: "2.0",
		Method:  method,
		Params:  pBytes,
	}
	bytes, _ := json.Marshal(msg)
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(bytes))
	_, _ = io.WriteString(s.out, header)
	_, _ = s.out.Write(bytes)
}

func (s *Server) handleMessage(msg *jsonrpcMessage) {
	switch msg.Method {
	case "initialize":
		res := map[string]interface{}{
			"capabilities": map[string]interface{}{
				"textDocumentSync": 1, // Full document sync
				"hoverProvider":    true,
				"documentFormattingProvider": true,
			},
			"serverInfo": map[string]string{
				"name":    "vexls",
				"version": "0.1.0",
			},
		}
		s.writeResponse(&jsonrpcMessage{ID: msg.ID, Result: res})

	case "textDocument/didOpen":
		var params struct {
			TextDocument struct {
				URI  string `json:"uri"`
				Text string `json:"text"`
			} `json:"textDocument"`
		}
		_ = json.Unmarshal(msg.Params, &params)
		s.docs[params.TextDocument.URI] = params.TextDocument.Text
		s.validateDocument(params.TextDocument.URI, params.TextDocument.Text)

	case "textDocument/didChange":
		var params struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
			ContentChanges []struct {
				Text string `json:"text"`
			} `json:"contentChanges"`
		}
		_ = json.Unmarshal(msg.Params, &params)
		if len(params.ContentChanges) > 0 {
			newText := params.ContentChanges[len(params.ContentChanges)-1].Text
			s.docs[params.TextDocument.URI] = newText
			s.validateDocument(params.TextDocument.URI, newText)
		}

	case "textDocument/formatting":
		var params struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
		}
		_ = json.Unmarshal(msg.Params, &params)
		text := s.docs[params.TextDocument.URI]
		formatted := formatter.Format(text)

		lines := strings.Split(text, "\n")
		lastLine := len(lines) - 1
		lastChar := 0
		if lastLine >= 0 {
			lastChar = len(lines[lastLine])
		}

		edit := []map[string]interface{}{
			{
				"range": Range{
					Start: Position{Line: 0, Character: 0},
					End:   Position{Line: lastLine + 1, Character: lastChar},
				},
				"newText": formatted,
			},
		}
		s.writeResponse(&jsonrpcMessage{ID: msg.ID, Result: edit})

	case "textDocument/hover":
		var params struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
			Position Position `json:"position"`
		}
		_ = json.Unmarshal(msg.Params, &params)
		text := s.docs[params.TextDocument.URI]
		hoverText := s.computeHover(text, params.Position)
		res := map[string]interface{}{
			"contents": map[string]string{
				"kind":  "markdown",
				"value": hoverText,
			},
		}
		s.writeResponse(&jsonrpcMessage{ID: msg.ID, Result: res})

	case "shutdown":
		s.writeResponse(&jsonrpcMessage{ID: msg.ID, Result: nil})

	case "exit":
		os.Exit(0)
	}
}

func (s *Server) validateDocument(uri string, text string) {
	var diagnostics []Diagnostic
	f := source.New(uri, text)

	p := parser.New(text)
	astFile := p.ParseFile()
	if parseErrs := p.Errors(); len(parseErrs) > 0 {
		for _, pe := range parseErrs {
			startLine, startCol := f.Location(pe.Span.Start)
			endLine, endCol := f.Location(pe.Span.End)
			if endLine < startLine || (endLine == startLine && endCol <= startCol) {
				endLine = startLine
				endCol = startCol + 1
			}
			diagnostics = append(diagnostics, Diagnostic{
				Range: Range{
					Start: Position{Line: startLine - 1, Character: startCol - 1},
					End:   Position{Line: endLine - 1, Character: endCol - 1},
				},
				Severity: 1,
				Source:   "vex parser",
				Message:  pe.Msg,
			})
		}
	} else {
		chk := checker.New()
		chk.CheckFile(astFile)
		if semErrs := chk.Errors(); len(semErrs) > 0 {
			for _, se := range semErrs {
				startLine, startCol := f.Location(se.Span.Start)
				endLine, endCol := f.Location(se.Span.End)
				if endLine < startLine || (endLine == startLine && endCol <= startCol) {
					endLine = startLine
					endCol = startCol + 1
				}
				diagnostics = append(diagnostics, Diagnostic{
					Range: Range{
						Start: Position{Line: startLine - 1, Character: startCol - 1},
						End:   Position{Line: endLine - 1, Character: endCol - 1},
					},
					Severity: 1,
					Source:   "vex checker",
					Message:  se.Msg,
				})
			}
		}
	}

	s.sendNotification("textDocument/publishDiagnostics", PublishDiagnosticsParams{
		URI:         uri,
		Diagnostics: diagnostics,
	})
}

func (s *Server) computeHover(text string, pos Position) string {
	lines := strings.Split(text, "\n")
	if pos.Line < 0 || pos.Line >= len(lines) {
		return "*(Vex Symbol)*"
	}
	line := lines[pos.Line]
	words := strings.Fields(line)
	for _, w := range words {
		switch w {
		case "actor":
			return "**Vex Actor Keyword**\n\nDefines an isolated, concurrent message-passing entity managed by a lock-free work-stealing pool and supervised for fault tolerance."
		case "arena":
			return "**Vex Bump Arena (`arena { ... }`)**\n\nAllocates intermediate objects and tensors inside an ultra-fast O(1) contiguous memory slab. Wiped instantly on block exit without `free()`."
		case "tensor":
			return "**Vex First-Class SIMD Tensor (`tensor[f32]`)**\n\n64-byte cache-line aligned 2D tensor supporting hardware-accelerated matrix multiplication (`@`) and zero-copy DMA to GPUs."
		case "@":
			return "**Vex Hardware MatMul Operator (`@`)**\n\nCompiles directly to 256-bit AVX2+FMA or 128-bit ARM NEON vector instructions."
		case "<-":
			return "**Vex Asynchronous Actor Send Operator (`<-`)**\n\nNon-blocking message dispatch returning a `Future[T]` without stalling the caller thread."
		}
	}
	return "**Vex Language Symbol**\n\n*Type inference & definition ready.*"
}
