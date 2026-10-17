package lsp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/checker"
	"github.com/viralang/vira/internal/formatter"
	"github.com/viralang/vira/internal/parser"
	"github.com/viralang/vira/internal/source"
	"github.com/viralang/vira/internal/token"
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
					Source:   "vira checker",
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
	word := getWordAt(text, pos)
	switch word {
	case "actor":
		return "**Vira Actor Keyword**\n\nDefines an isolated, concurrent message-passing entity managed by a lock-free work-stealing pool and supervised for fault tolerance."
	case "arena":
		return "**Vira Bump Arena (`arena { ... }`)**\n\nAllocates intermediate objects and tensors inside an ultra-fast O(1) contiguous memory slab. Wiped instantly on block exit without `free()`."
	case "tensor":
		return "**Vira First-Class SIMD Tensor (`tensor[f32]`)**\n\n64-byte cache-line aligned 2D tensor supporting hardware-accelerated matrix multiplication (`@`) and zero-copy DMA to GPUs."
	case "gui_init":
		return "**`gui_init(width: int, height: int, title: string)`**\n\nInitializes a hardware-accelerated native window (Win32 GDI on Windows, X11 on Linux) with zero third-party dependencies."
	case "gui_should_close":
		return "**`gui_should_close() -> bool`**\n\nReturns `true` if the user requested window closure (titlebar 'X' clicked, Alt+F4, or window manager signal)."
	case "gui_cancel_close":
		return "**`gui_cancel_close()`**\n\nIntercepts and cancels an OS close request, resetting the `should_close` flag to keep the window open (e.g. to display an unsaved changes modal)."
	case "gui_set_target_fps":
		return "**`gui_set_target_fps(fps: int)`**\n\nSets the target frame rate (default 60 FPS) with hardware-timed VSync sleep."
	case "gui_window_width":
		return "**`gui_window_width() -> int`**\n\nReturns the current client viewport width in pixels, dynamically updated on window resize."
	case "gui_window_height":
		return "**`gui_window_height() -> int`**\n\nReturns the current client viewport height in pixels, dynamically updated on window resize."
	case "gui_begin":
		return "**`gui_begin()`**\n\nBegins a new double-buffered frame and dispatches non-blocking OS input events."
	case "gui_end":
		return "**`gui_end()`**\n\nFinalizes the current frame, presents the backbuffer via BitBlt/XCopyArea, and throttles to the target frame rate."
	case "gui_clear":
		return "**`gui_clear(hex_color: int)`**\n\nFills the double-buffered backbuffer canvas with a 24-bit RGB hex color (e.g. `0x181825`)."
	case "gui_vstack_begin":
		return "**`gui_vstack_begin(x: int, y: int, width: int, spacing: int)`**\n\nBegins an auto-flowing vertical column layout on the layout cursor stack."
	case "gui_hstack_begin":
		return "**`gui_hstack_begin(height: int, spacing: int)`**\n\nBegins an auto-flowing horizontal row layout on the layout cursor stack."
	case "gui_stack_end":
		return "**`gui_stack_end()`**\n\nCloses the topmost layout context (VStack or HStack) on the cursor stack."
	case "gui_spacing":
		return "**`gui_spacing(px: int)`**\n\nAdvances the current layout cursor along its primary axis by `px` pixels."
	case "gui_card_begin":
		return "**`gui_card_begin(title: string, width: int)`**\n\nBegins an elevated auto-sized card panel with header styling."
	case "gui_card_end":
		return "**`gui_card_end()`**\n\nCloses the active card layout container and auto-calculates total card height."
	case "gui_separator":
		return "**`gui_separator()`**\n\nRenders a 1px divider line across the current layout container."
	case "gui_badge":
		return "**`gui_badge(text: string, hex_bg: int, hex_fg: int)`**\n\nRenders a pill badge with custom background and foreground colors."
	case "gui_btn":
		return "**`gui_btn(label: string) -> bool`**\n\nRenders a zero-math declarative button sized to the active layout. Returns `true` when clicked."
	case "gui_check":
		return "**`gui_check(label: string, checked: bool) -> bool`**\n\nRenders a declarative checkbox. Returns updated checked state."
	case "gui_slide":
		return "**`gui_slide(text_left: string, text_right: string, val: f32, min_val: f32, max_val: f32) -> f32`**\n\nRenders a declarative slider control. Returns updated value."
	case "gui_lbl":
		return "**`gui_lbl(text: string)`**\n\nRenders a declarative text label in the active layout container."
	case "gui_input":
		return "**`gui_input(placeholder: string, text: string) -> string`**\n\nRenders a declarative single-line text input field."
	case "gui_progress":
		return "**`gui_progress(percent: f32)`**\n\nRenders a declarative progress bar with value between 0.0 and 1.0."
	case "gui_radio":
		return "**`gui_radio(label: string, is_selected: bool) -> bool`**\n\nRenders a declarative radio button option."
	case "gui_tab_item":
		return "**`gui_tab_item(label: string, is_active: bool) -> bool`**\n\nRenders a navigation tab item with hover state and active indicator. Returns `true` when clicked."
	case "gui_modal_begin":
		return "**`gui_modal_begin(title: string, width: int, height: int)`**\n\nRenders a dimmed full-window backdrop (`0x0C0C14`) and centers an elevated modal dialog."
	case "gui_modal_end":
		return "**`gui_modal_end()`**\n\nCloses the active modal dialog layout context."
	}

	lines := strings.Split(text, "\n")
	if pos.Line >= 0 && pos.Line < len(lines) {
		line := lines[pos.Line]
		if strings.Contains(line, "@") {
			return "**Vira Hardware MatMul Operator (`@`)**\n\nCompiles directly to 256-bit AVX2+FMA or 128-bit ARM NEON vector instructions."
		}
		if strings.Contains(line, "<-") {
			return "**Vira Asynchronous Actor Send Operator (`<-`)**\n\nNon-blocking message dispatch returning a `Future[T]` without stalling the caller thread."
		}
	}
	return "**Vira Language Symbol**\n\n*Type inference & definition ready.*"
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
