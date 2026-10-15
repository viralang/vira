package lsp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/vexlang/vex/internal/ast"
	"github.com/vexlang/vex/internal/checker"
	"github.com/vexlang/vex/internal/formatter"
	"github.com/vexlang/vex/internal/parser"
	"github.com/vexlang/vex/internal/source"
	"github.com/vexlang/vex/internal/token"
)

type Server struct {
	in   *bufio.Reader
	out  io.Writer
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
}

type jsonrpcResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      interface{}   `json:"id"`
	Result  interface{}   `json:"result"`
	Error   *jsonrpcError `json:"error,omitempty"`
}

func (r jsonrpcResponse) MarshalJSON() ([]byte, error) {
	if r.Error != nil {
		return json.Marshal(struct {
			JSONRPC string        `json:"jsonrpc"`
			ID      interface{}   `json:"id"`
			Error   *jsonrpcError `json:"error"`
		}{
			JSONRPC: "2.0",
			ID:      r.ID,
			Error:   r.Error,
		})
	}
	return json.Marshal(struct {
		JSONRPC string      `json:"jsonrpc"`
		ID      interface{} `json:"id"`
		Result  interface{} `json:"result"`
	}{
		JSONRPC: "2.0",
		ID:      r.ID,
		Result:  r.Result,
	})
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

type Location struct {
	URI   string `json:"uri"`
	Range Range  `json:"range"`
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

func (p PublishDiagnosticsParams) MarshalJSON() ([]byte, error) {
	type Alias PublishDiagnosticsParams
	diagnostics := p.Diagnostics
	if diagnostics == nil {
		diagnostics = []Diagnostic{}
	}
	return json.Marshal(&struct {
		Alias
		Diagnostics []Diagnostic `json:"diagnostics"`
	}{
		Alias:       Alias(p),
		Diagnostics: diagnostics,
	})
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

func (s *Server) writeResponse(id interface{}, result interface{}) {
	resp := jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	}
	bytes, _ := json.Marshal(resp)
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(bytes))
	_, _ = io.WriteString(s.out, header)
	_, _ = s.out.Write(bytes)
}

func (s *Server) sendNotification(method string, params interface{}) {
	pBytes, _ := json.Marshal(params)
	msg := struct {
		JSONRPC string          `json:"jsonrpc"`
		Method  string          `json:"method"`
		Params  json.RawMessage `json:"params"`
	}{
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
				"textDocumentSync":           1, // Full document sync
				"hoverProvider":              true,
				"definitionProvider":         true,
				"documentFormattingProvider": true,
			},
			"serverInfo": map[string]string{
				"name":    "vexls",
				"version": "0.1.0",
			},
		}
		s.writeResponse(msg.ID, res)

	case "textDocument/definition":
		var params struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
			Position Position `json:"position"`
		}
		_ = json.Unmarshal(msg.Params, &params)
		text := s.docs[params.TextDocument.URI]
		loc := s.findDefinition(params.TextDocument.URI, text, params.Position)
		s.writeResponse(msg.ID, loc)

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
		s.writeResponse(msg.ID, edit)

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
		s.writeResponse(msg.ID, res)

	case "shutdown":
		s.writeResponse(msg.ID, nil)

	case "exit":
		os.Exit(0)
	}
}

func (s *Server) validateDocument(uri string, text string) {
	diagnostics := make([]Diagnostic, 0)
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

func (s *Server) findDefinition(uri string, text string, pos Position) interface{} {
	word := getWordAt(text, pos)
	if word == "" {
		return nil
	}

	p := parser.New(text)
	astFile := p.ParseFile()
	f := source.New(uri, text)

	// 1. Search top-level declarations
	for _, decl := range astFile.Decls {
		switch d := decl.(type) {
		case *ast.FnDecl:
			if d.Name.Name == word {
				loc := makeLocation(uri, f, d.Name.NamePos)
				return loc
			}
			for _, param := range d.Params {
				if param.Name.Name == word {
					loc := makeLocation(uri, f, param.Name.NamePos)
					return loc
				}
			}
			if d.Body != nil {
				if loc := findInBlock(uri, f, d.Body, word); loc != nil {
					return *loc
				}
			}

		case *ast.StructDecl:
			if d.Name.Name == word {
				loc := makeLocation(uri, f, d.Name.NamePos)
				return loc
			}
			for _, field := range d.Fields {
				if field.Name.Name == word {
					loc := makeLocation(uri, f, field.Name.NamePos)
					return loc
				}
			}

		case *ast.InterfaceDecl:
			if d.Name.Name == word {
				loc := makeLocation(uri, f, d.Name.NamePos)
				return loc
			}
			for _, method := range d.Methods {
				if method.Name.Name == word {
					loc := makeLocation(uri, f, method.Name.NamePos)
					return loc
				}
			}
			for _, field := range d.Fields {
				if field.Name.Name == word {
					loc := makeLocation(uri, f, field.Name.NamePos)
					return loc
				}
			}
		}
	}

	return nil
}

func findInBlock(uri string, f *source.File, block *ast.BlockStmt, word string) *Location {
	for _, stmt := range block.Stmts {
		switch st := stmt.(type) {
		case *ast.VarDeclStmt:
			if st.Name.Name == word {
				loc := makeLocation(uri, f, st.Name.NamePos)
				return &loc
			}
		case *ast.IfStmt:
			if st.Then != nil {
				if loc := findInBlock(uri, f, st.Then, word); loc != nil {
					return loc
				}
			}
			if st.Else != nil {
				if loc := findInBlock(uri, f, st.Else, word); loc != nil {
					return loc
				}
			}
		case *ast.ForStmt:
			if st.Body != nil {
				if loc := findInBlock(uri, f, st.Body, word); loc != nil {
					return loc
				}
			}
		}
	}
	return nil
}

func makeLocation(uri string, f *source.File, span token.Span) Location {
	startLine, startCol := f.Location(span.Start)
	endLine, endCol := f.Location(span.End)
	if endLine < startLine || (endLine == startLine && endCol <= startCol) {
		endLine = startLine
		endCol = startCol + 1
	}
	return Location{
		URI: uri,
		Range: Range{
			Start: Position{Line: startLine - 1, Character: startCol - 1},
			End:   Position{Line: endLine - 1, Character: endCol - 1},
		},
	}
}

func getWordAt(text string, pos Position) string {
	lines := strings.Split(text, "\n")
	if pos.Line < 0 || pos.Line >= len(lines) {
		return ""
	}
	line := lines[pos.Line]
	if pos.Character < 0 || pos.Character > len(line) {
		return ""
	}

	start := pos.Character
	for start > 0 && isIdentChar(line[start-1]) {
		start--
	}
	end := pos.Character
	for end < len(line) && isIdentChar(line[end]) {
		end++
	}
	if start >= end {
		return ""
	}
	return line[start:end]
}

func isIdentChar(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b == '_'
}
