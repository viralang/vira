package lsp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/checker"
	"github.com/viralang/vira/internal/formatter"
	"github.com/viralang/vira/internal/mod"
	"github.com/viralang/vira/internal/parser"
	"github.com/viralang/vira/internal/pkg"
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

type CompletionItem struct {
	Label         string `json:"label"`
	Kind          int    `json:"kind"` // 2=Method, 3=Function, 5=Field, 6=Variable, 7=Class/Struct, 14=Keyword
	Detail        string `json:"detail,omitempty"`
	Documentation string `json:"documentation,omitempty"`
}

type DocumentSymbol struct {
	Name           string           `json:"name"`
	Detail         string           `json:"detail,omitempty"`
	Kind           int              `json:"kind"`
	Range          Range            `json:"range"`
	SelectionRange Range            `json:"selectionRange"`
	Children       []DocumentSymbol `json:"children,omitempty"`
}

type ImportedDoc struct {
	Path       string
	URI        string
	SourceFile *source.File
	AST        *ast.File
}

func uriToPath(uri string) string {
	if decoded, err := url.PathUnescape(uri); err == nil {
		uri = decoded
	}

	p := uri
	if strings.HasPrefix(p, "file:///") {
		p = strings.TrimPrefix(p, "file:///")
	} else if strings.HasPrefix(p, "file://") {
		p = strings.TrimPrefix(p, "file://")
	}

	// On Windows, strip leading slash before drive letter e.g. "/c:/..." or "/C:/..."
	if len(p) >= 3 && p[0] == '/' && p[2] == ':' {
		p = p[1:]
	}
	if len(p) >= 2 && p[1] == ':' {
		return filepath.Clean(filepath.FromSlash(p))
	}

	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	return filepath.Clean(filepath.FromSlash(p))
}

func pathToURI(p string) string {
	clean := filepath.ToSlash(filepath.Clean(p))
	if len(clean) >= 2 && clean[1] == ':' {
		return "file:///" + clean
	}
	if !strings.HasPrefix(clean, "/") {
		return "file:///" + clean
	}
	return "file://" + clean
}

func normalizeURI(uri string) string {
	return pathToURI(uriToPath(uri))
}

func (s *Server) getDocumentText(uri string) string {
	norm := normalizeURI(uri)
	if txt, ok := s.docs[norm]; ok {
		return txt
	}
	if txt, ok := s.docs[uri]; ok {
		return txt
	}
	for dURI, dt := range s.docs {
		if strings.EqualFold(normalizeURI(dURI), norm) {
			return dt
		}
	}
	diskPath := uriToPath(norm)
	if data, err := os.ReadFile(diskPath); err == nil {
		return string(data)
	}
	return ""
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
				"completionProvider": map[string]interface{}{
					"triggerCharacters": []string{".", ":"},
				},
				"documentSymbolProvider": true,
			},
			"serverInfo": map[string]string{
				"name":    "vexls",
				"version": "0.1.2",
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
		text := s.getDocumentText(params.TextDocument.URI)
		loc := s.findDefinition(params.TextDocument.URI, text, params.Position)
		s.writeResponse(msg.ID, loc)

	case "textDocument/completion":
		var params struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
			Position Position `json:"position"`
		}
		_ = json.Unmarshal(msg.Params, &params)
		text := s.getDocumentText(params.TextDocument.URI)
		items := s.computeCompletions(params.TextDocument.URI, text, params.Position)
		s.writeResponse(msg.ID, map[string]interface{}{
			"isIncomplete": false,
			"items":        items,
		})

	case "textDocument/documentSymbol":
		var params struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
		}
		_ = json.Unmarshal(msg.Params, &params)
		text := s.getDocumentText(params.TextDocument.URI)
		symbols := s.computeDocumentSymbols(params.TextDocument.URI, text)
		s.writeResponse(msg.ID, symbols)

	case "textDocument/didOpen":
		var params struct {
			TextDocument struct {
				URI  string `json:"uri"`
				Text string `json:"text"`
			} `json:"textDocument"`
		}
		_ = json.Unmarshal(msg.Params, &params)
		normURI := normalizeURI(params.TextDocument.URI)
		s.docs[normURI] = params.TextDocument.Text
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
			normURI := normalizeURI(params.TextDocument.URI)
			s.docs[normURI] = newText
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
		text := s.getDocumentText(params.TextDocument.URI)
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
		text := s.getDocumentText(params.TextDocument.URI)
		hoverText := s.computeHover(params.TextDocument.URI, text, params.Position)
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

func (s *Server) loadDoc(cleanPath string) *ImportedDoc {
	candURI := pathToURI(cleanPath)
	var text string

	if openText, ok := s.docs[candURI]; ok {
		text = openText
	} else if openText, ok := s.docs[normalizeURI(candURI)]; ok {
		text = openText
	} else {
		norm := normalizeURI(candURI)
		for dURI, dt := range s.docs {
			if strings.EqualFold(normalizeURI(dURI), norm) {
				text = dt
				break
			}
		}
	}

	if text == "" {
		if data, err := os.ReadFile(cleanPath); err == nil {
			text = string(data)
		} else {
			return nil
		}
	}

	p := parser.New(text)
	impAST := p.ParseFile()
	srcFile := source.New(candURI, text)

	return &ImportedDoc{
		Path:       cleanPath,
		URI:        candURI,
		SourceFile: srcFile,
		AST:        impAST,
	}
}

// resolveImports discovers and parses all files imported by astFile (recursively).
func (s *Server) resolveImports(uri string, astFile *ast.File) []ImportedDoc {
	var results []ImportedDoc
	if astFile == nil {
		return results
	}

	rootPath := uriToPath(uri)
	fileDir := filepath.Dir(rootPath)
	pRoot, _, _ := mod.FindProjectRoot(fileDir)

	visited := make(map[string]bool)
	visited[filepath.Clean(rootPath)] = true

	type workItem struct {
		filePath string
		ast      *ast.File
	}

	workList := []workItem{{filePath: rootPath, ast: astFile}}

	// Discover all sibling source files in the same project root or directory
	searchDir := ""
	if pRoot != "" {
		searchDir = pRoot
	} else if fileDir != "" && fileDir != "/" && fileDir != "\\" && fileDir != string(filepath.Separator) && fileDir != "." && !(len(fileDir) <= 3 && strings.Contains(fileDir, ":")) {
		if fi, err := os.Stat(fileDir); err == nil && fi.IsDir() {
			searchDir = fileDir
		}
	}

	isTest := strings.HasSuffix(rootPath, "_test.vir") || strings.HasSuffix(rootPath, "_test.vex") ||
		strings.Contains(rootPath, string(filepath.Separator)+"tests"+string(filepath.Separator))

	if searchDir != "" {
		_ = filepath.Walk(searchDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			if info.IsDir() {
				base := filepath.Base(path)
				if base == ".git" || base == "build" || base == "bin" || base == "dist" || base == "node_modules" || base == ".vir" || base == ".vira" || base == "vendor" {
					return filepath.SkipDir
				}
				if !isTest && base == "tests" {
					return filepath.SkipDir
				}
				if base == "examples" && !strings.Contains(rootPath, string(filepath.Separator)+"examples"+string(filepath.Separator)) {
					return filepath.SkipDir
				}
				return nil
			}
			cleanPath := filepath.Clean(path)
			if visited[cleanPath] {
				return nil
			}
			isSource := strings.HasSuffix(info.Name(), ".vir") || strings.HasSuffix(info.Name(), ".vex")
			if !isTest {
				isSource = isSource && !strings.HasSuffix(info.Name(), "_test.vir") && !strings.HasSuffix(info.Name(), "_test.vex")
			}
			if isSource {
				doc := s.loadDoc(cleanPath)
				if doc != nil && doc.AST != nil {
					if astFile.Module.Name == "" || doc.AST.Module.Name == astFile.Module.Name || doc.AST.Module.Name == "" {
						visited[cleanPath] = true
						results = append(results, *doc)
						workList = append(workList, workItem{filePath: cleanPath, ast: doc.AST})
					}
				}
			}
			return nil
		})

		for docURI := range s.docs {
			docPath := uriToPath(docURI)
			cleanDocPath := filepath.Clean(docPath)
			if visited[cleanDocPath] {
				continue
			}
			rel, err := filepath.Rel(searchDir, cleanDocPath)
			if err == nil && !strings.HasPrefix(rel, "..") {
				isSource := strings.HasSuffix(cleanDocPath, ".vir") || strings.HasSuffix(cleanDocPath, ".vex")
				if !isTest {
					isSource = isSource && !strings.HasSuffix(cleanDocPath, "_test.vir") && !strings.HasSuffix(cleanDocPath, "_test.vex")
				}
				if isSource {
					doc := s.loadDoc(cleanDocPath)
					if doc != nil && doc.AST != nil {
						if astFile.Module.Name == "" || doc.AST.Module.Name == astFile.Module.Name || doc.AST.Module.Name == "" {
							visited[cleanDocPath] = true
							results = append(results, *doc)
							workList = append(workList, workItem{filePath: cleanDocPath, ast: doc.AST})
						}
					}
				}
			}
		}
	}

	visitedPkg := make(map[string]bool)

	for len(workList) > 0 {
		curr := workList[0]
		workList = workList[1:]

		cDir := filepath.Dir(curr.filePath)

		for _, imp := range curr.ast.Imports {
			impTarget := imp.Name
			if visitedPkg[impTarget] {
				continue
			}
			candidates := pkg.ResolveImportCandidates(cDir, pRoot, impTarget)

			for _, cand := range candidates {
				cleanCand := filepath.Clean(cand)
				if visited[cleanCand] {
					continue
				}

				cStat, cErr := os.Stat(cleanCand)
				if cErr == nil && cStat.IsDir() {
					visited[cleanCand] = true
					visitedPkg[impTarget] = true
					_ = filepath.Walk(cleanCand, func(subPath string, subInfo os.FileInfo, err error) error {
						if err != nil {
							return nil
						}
						if subInfo.IsDir() {
							if subInfo.Name() == "tests" || subInfo.Name() == ".git" {
								return filepath.SkipDir
							}
							return nil
						}
						if (strings.HasSuffix(subInfo.Name(), ".vir") || strings.HasSuffix(subInfo.Name(), ".vex")) &&
							!strings.HasSuffix(subInfo.Name(), "_test.vir") && !strings.HasSuffix(subInfo.Name(), "_test.vex") {
							cleanSub := filepath.Clean(subPath)
							if !visited[cleanSub] {
								visited[cleanSub] = true
								doc := s.loadDoc(cleanSub)
								if doc != nil {
									results = append(results, *doc)
									workList = append(workList, workItem{filePath: cleanSub, ast: doc.AST})
								}
							}
						}
						return nil
					})
					break
				}

				doc := s.loadDoc(cleanCand)
				if doc != nil {
					visited[cleanCand] = true
					visitedPkg[impTarget] = true
					results = append(results, *doc)
					workList = append(workList, workItem{filePath: cleanCand, ast: doc.AST})
					break
				}
			}
		}
	}

	return results
}

func belongsToFile(sp token.Span, file *ast.File) bool {
	if file == nil {
		return false
	}
	if sp.Start >= file.Module.NamePos.Start && sp.End <= file.Module.NamePos.End {
		return true
	}
	for _, imp := range file.Imports {
		if sp.Start >= imp.NamePos.Start && sp.End <= imp.NamePos.End {
			return true
		}
	}
	for _, d := range file.Decls {
		dSpan := d.Span()
		if sp.Start >= dSpan.Start && sp.End <= dSpan.End {
			return true
		}
	}
	return false
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
		// Check for missing/unresolved imports and report diagnostic errors
		docFileDir := filepath.Dir(uriToPath(uri))
		docPRoot, _, _ := mod.FindProjectRoot(docFileDir)
		for _, imp := range astFile.Imports {
			impTarget := imp.Name
			candList := pkg.ResolveImportCandidates(docFileDir, docPRoot, impTarget)
			foundImp := false
			for _, c := range candList {
				candURI := pathToURI(filepath.Clean(c))
				if _, ok := s.docs[candURI]; ok {
					foundImp = true
					break
				}
				if _, ok := s.docs[normalizeURI(candURI)]; ok {
					foundImp = true
					break
				}
				if _, err := os.Stat(c); err == nil {
					foundImp = true
					break
				}
			}
			if !foundImp {
				startLine, startCol := f.Location(imp.NamePos.Start)
				endLine, endCol := f.Location(imp.NamePos.End)
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
					Source:   "vira import",
					Message:  fmt.Sprintf("cannot find imported file or package '%s'", impTarget),
				})
			}
		}

		// Include imported files so imported structs, methods & types are recognized without false errors
		imported := s.resolveImports(uri, astFile)
		var allFiles []*ast.File
		for _, imp := range imported {
			allFiles = append(allFiles, imp.AST)
		}
		allFiles = append(allFiles, astFile)

		chk := checker.New()
		chk.CheckFiles(allFiles)
		if semErrs := chk.Errors(); len(semErrs) > 0 {
			for _, se := range semErrs {
				// Only publish diagnostics that belong to the active document being validated
				if belongsToFile(se.Span, astFile) {
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
	}

	s.sendNotification("textDocument/publishDiagnostics", PublishDiagnosticsParams{
		URI:         uri,
		Diagnostics: diagnostics,
	})
}

func (s *Server) computeHover(uri string, text string, pos Position) string {
	word := getWordAt(text, pos)
	if word == "" {
		return ""
	}

	p := parser.New(text)
	astFile := p.ParseFile()

	// 1. Check local declarations
	if h := formatDeclHover(astFile, word, ""); h != "" {
		return h
	}

	// 2. Check imported files
	imported := s.resolveImports(uri, astFile)
	for _, impDoc := range imported {
		originName := filepath.Base(impDoc.Path)
		if h := formatDeclHover(impDoc.AST, word, originName); h != "" {
			return h
		}
	}

	// 3. Fallback to builtins, keywords, operators
	return s.computeBuiltinHover(text, pos, word)
}

func formatTypeExpr(t ast.TypeExpr) string {
	if t == nil {
		return "void"
	}
	switch v := t.(type) {
	case *ast.ArrayType:
		if v.Len != nil {
			if lit, ok := v.Len.(*ast.LiteralExpr); ok {
				return fmt.Sprintf("[%s]%s", lit.Value, formatTypeExpr(v.Elem))
			}
			return fmt.Sprintf("[N]%s", formatTypeExpr(v.Elem))
		}
		return "[]" + formatTypeExpr(v.Elem)
	case *ast.SelectorType:
		if len(v.TypeParams) > 0 {
			var params []string
			for _, p := range v.TypeParams {
				params = append(params, formatTypeExpr(p))
			}
			return fmt.Sprintf("%s.%s[%s]", v.Pkg.Name, v.Name.Name, strings.Join(params, ", "))
		}
		return fmt.Sprintf("%s.%s", v.Pkg.Name, v.Name.Name)
	case *ast.IdentType:
		if len(v.TypeParams) > 0 {
			var params []string
			for _, p := range v.TypeParams {
				params = append(params, formatTypeExpr(p))
			}
			return fmt.Sprintf("%s[%s]", v.Name.Name, strings.Join(params, ", "))
		}
		return v.Name.Name
	case *ast.ResultType:
		if v.Inner == nil {
			return "!"
		}
		return "!" + formatTypeExpr(v.Inner)
	case *ast.OptionType:
		return "?" + formatTypeExpr(v.Inner)
	case *ast.RefType:
		if v.IsMut {
			return "&mut " + formatTypeExpr(v.Inner)
		}
		return "&" + formatTypeExpr(v.Inner)
	default:
		return "unknown"
	}
}

func getEmbedInfo(emb ast.TypeExpr) (string, token.Span) {
	if it, ok := emb.(*ast.IdentType); ok {
		return it.Name.Name, it.Name.NamePos
	}
	if emb != nil {
		return formatTypeExpr(emb), emb.Span()
	}
	return "", token.Span{}
}

func formatDeclHover(astFile *ast.File, word string, originFile string) string {
	if astFile == nil {
		return ""
	}

	originNote := ""
	if originFile != "" {
		originNote = fmt.Sprintf(" *(imported from `%s`)*", originFile)
	}

	for _, decl := range astFile.Decls {
		switch d := decl.(type) {
		case *ast.StructDecl:
			if d.Name.Name == word {
				kindStr := "struct"
				if d.IsActor {
					kindStr = "actor"
				}
				var sb strings.Builder
				sb.WriteString(fmt.Sprintf("### `%s %s`%s\n\n", kindStr, d.Name.Name, originNote))
				if len(d.Embeds) > 0 {
					sb.WriteString("**Embedded Structs (Promoted):**\n")
					for _, emb := range d.Embeds {
						embName, _ := getEmbedInfo(emb)
						sb.WriteString(fmt.Sprintf("- `%s`\n", embName))
					}
					sb.WriteString("\n")
				}
				if len(d.Fields) > 0 {
					sb.WriteString("**Fields:**\n")
					for _, f := range d.Fields {
						accessStr := f.Access.String()
						sb.WriteString(fmt.Sprintf("- `%s`: `%s` (%s)\n", f.Name.Name, formatTypeExpr(f.Type), accessStr))
					}
				}
				return sb.String()
			}

		case *ast.FnDecl:
			if d.Name.Name == word {
				var sb strings.Builder
				receiverStr := ""
				if d.Receiver != nil {
					mutStr := ""
					if d.Receiver.IsMut {
						mutStr = "mut "
					}
					receiverStr = fmt.Sprintf("(%s%s %s) ", mutStr, d.Receiver.Name.Name, formatTypeExpr(d.Receiver.Type))
				} else if d.StaticType != nil {
					receiverStr = fmt.Sprintf("%s.", d.StaticType.Name)
				}
				retStr := ""
				if d.ReturnType != nil {
					retStr = " " + formatTypeExpr(d.ReturnType)
				}
				var params []string
				for _, p := range d.Params {
					mutStr := ""
					if p.IsMut {
						mutStr = "mut "
					}
					params = append(params, fmt.Sprintf("%s%s %s", mutStr, p.Name.Name, formatTypeExpr(p.Type)))
				}
				sb.WriteString(fmt.Sprintf("### `fn %s%s(%s)%s`%s\n\n", receiverStr, d.Name.Name, strings.Join(params, ", "), retStr, originNote))
				if d.Receiver != nil {
					sb.WriteString(fmt.Sprintf("Method on `%s`.\n", formatTypeExpr(d.Receiver.Type)))
				}
				return sb.String()
			}

		case *ast.InterfaceDecl:
			if d.Name.Name == word {
				var sb strings.Builder
				sb.WriteString(fmt.Sprintf("### `interface %s`%s\n\n", d.Name.Name, originNote))
				if len(d.Methods) > 0 {
					sb.WriteString("**Methods:**\n")
					for _, m := range d.Methods {
						sb.WriteString(fmt.Sprintf("- `fn %s(...)`\n", m.Name.Name))
					}
				}
				return sb.String()
			}
		}
	}
	return ""
}

func (s *Server) computeBuiltinHover(text string, pos Position, word string) string {
	switch word {
	case "actor":
		return "**Vira Actor Keyword**\n\nDefines an isolated, concurrent message-passing entity managed by a lock-free work-stealing pool and supervised for fault tolerance."
	case "arena":
		return "**Vira Bump Arena (`arena { ... }`)**\n\nAllocates intermediate objects and tensors inside an ultra-fast O(1) contiguous memory slab. Wiped instantly on block exit without `free()`."
	case "tensor":
		return "**Vira First-Class Strided SIMD Tensor (`tensor[f32]`)**\n\n64-byte cache-line aligned 1D–4D strided tensor (`dims[4]`, `strides[4]`, `n_dims`) supporting batched/broadcast matrix multiplication (`@`) parallelized across the multi-threaded CPU SIMD engine (`AVX2+FMA`, `SSE2`, `ARM NEON`) and zero-copy C99 FFI pointer interop."
	case "tensor_fill":
		return "**`tensor_fill(rows: int, cols: int, val: f32) -> tensor[f32]`**\n\nAllocates a 64-byte cache-line aligned 2D `tensor[f32]` (`n_dims = 2`) initialized to `val`."
	case "tensor_fill_3d":
		return "**`tensor_fill_3d(d0: int, d1: int, d2: int, val: f32) -> tensor[f32]`**\n\nAllocates a 64-byte cache-line aligned 3D strided `tensor[f32]` (`n_dims = 3`, e.g. `Batch x Seq x Hidden`) initialized to `val`."
	case "tensor_fill_4d":
		return "**`tensor_fill_4d(d0: int, d1: int, d2: int, d3: int, val: f32) -> tensor[f32]`**\n\nAllocates a 64-byte cache-line aligned 4D strided `tensor[f32]` (`n_dims = 4`, e.g. `Batch x Heads x Seq x HeadDim`) initialized to `val`."
	case "ndim":
		return "**`tensor.ndim -> int`**\n\nReturns the active rank (`1` to `4`) of the strided tensor."
	case "numel":
		return "**`tensor.numel() -> int`**\n\nReturns the total element count across all active dimensions of the tensor."
	case "dim":
		return "**`tensor.dim(axis: int) -> int`**\n\nReturns the extent of dimension `axis` (`0 <= axis < tensor.ndim`)."
	case "stride":
		return "**`tensor.stride(axis: int) -> int`**\n\nReturns the row-major element stride along dimension `axis`."
	case "reshape_2d":
		return "**`tensor.reshape_2d(d0: int, d1: int) -> tensor[f32]`**\n\nReshapes tensor into 2D `(d0, d1)`. Returns an O(1) zero-copy view inside `arena { ... }`, or an independent autofreed buffer outside arenas."
	case "reshape_3d":
		return "**`tensor.reshape_3d(d0: int, d1: int, d2: int) -> tensor[f32]`**\n\nReshapes tensor into 3D `(d0, d1, d2)`. Returns an O(1) zero-copy view inside `arena { ... }`, or an independent autofreed buffer outside arenas."
	case "reshape_4d":
		return "**`tensor.reshape_4d(d0: int, d1: int, d2: int, d3: int) -> tensor[f32]`**\n\nReshapes tensor into 4D `(d0, d1, d2, d3)`. Returns an O(1) zero-copy view inside `arena { ... }`, or an independent autofreed buffer outside arenas."
	case "transpose":
		return "**`tensor.transpose() -> tensor[f32]`**\n\nTransposes the trailing two matrix axes `(..., M, N) -> (..., N, M)` for 2D, 3D, or 4D strided tensors."
	case "await":
		return "**`future.await() -> !T`**\n\nSynchronizes an asynchronous actor `Future[T]` with cooperative Chase-Lev work helping, returning `!T` for `or { ... }` fault recovery."
	case "switch":
		return "**Vira Switch Statement (`switch expr { case ... }`)**\n\nMulti-branch control flow evaluating integer, string, or boolean conditions with automatic break semantics."
	case "case":
		return "**Vira Switch Case Branch (`case val1, val2:`)**\n\nMatches one or more comma-separated values or boolean conditions. Breaks automatically upon branch completion."
	case "default":
		return "**Vira Switch Default Branch (`default:`)**\n\nFallback branch executed when no preceding `case` condition matches."
	case "module":
		return "**Vira Module Declaration (`module <name>`)**\n\nDeclares the primary package namespace for exported structs, interfaces, and functions, preventing cross-package naming collisions."
	case "push", "append":
		return "**`slice.push(val: T)`**\n\nAppends an element to the dynamic slice with amortized O(1) geometric buffer expansion."
	case "pop":
		return "**`slice.pop() -> T`**\n\nRemoves and returns the last element of the slice. Panics if the slice is empty."
	case "len":
		return "**`collection.len -> int`**\n\nReturns the number of elements in the array or slice."
	case "cap":
		return "**`slice.cap -> int`**\n\nReturns the current allocated backing buffer capacity of the dynamic slice."
	case "get":
		return "**`collection.get(index: int) -> !T`**\n\nSafely retrieves element by index returning Result Monad `!T`. Handle with `or { default }` or propagate with `!`."
	case "set":
		return "**`slice.set(index: int, val: T)`**\n\nOverwrites element at index with runtime bounds checking."
	case "clear":
		return "**`slice.clear()`**\n\nCleans and resets length to 0, retaining backing capacity."
	case "clone":
		return "**`slice.clone() -> []T`**\n\nAllocates a fresh deep copy of the slice with an independent backing buffer."
	case "contains":
		return "**`collection.contains(val: T) -> bool`**\n\nReturns `true` if the specified value exists in the array or slice."
	case "is_empty":
		return "**`collection.is_empty() -> bool`**\n\nReturns `true` if the collection length is 0, otherwise `false`."
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
			return "**Vira Hardware MatMul Operator (`@`)**\n\nBatched & broadcast 1D–4D strided matrix multiplication parallelized across the multi-threaded CPU SIMD engine (`AVX2+FMA`, `SSE2`, `ARM NEON`)."
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

	// Check if cursor is on an import target itself
	lines := strings.Split(text, "\n")
	if pos.Line >= 0 && pos.Line < len(lines) {
		line := lines[pos.Line]
		if strings.Contains(line, "import") {
			imported := s.resolveImports(uri, astFile)
			for _, impDoc := range imported {
				base := filepath.Base(impDoc.Path)
				if strings.Contains(line, base) || (len(impDoc.AST.Module.Name) > 0 && strings.Contains(line, impDoc.AST.Module.Name)) {
					return Location{
						URI: impDoc.URI,
						Range: Range{
							Start: Position{Line: 0, Character: 0},
							End:   Position{Line: 0, Character: 0},
						},
					}
				}
			}
		}
	}

	// 1. Search local declarations in current file
	if loc := searchASTForWord(uri, f, astFile, word); loc != nil {
		return *loc
	}

	// 2. Search imported files
	imported := s.resolveImports(uri, astFile)
	for _, impDoc := range imported {
		if loc := searchASTForWord(impDoc.URI, impDoc.SourceFile, impDoc.AST, word); loc != nil {
			return *loc
		}
	}

	return nil
}

func searchASTForWord(uri string, f *source.File, astFile *ast.File, word string) *Location {
	if astFile == nil {
		return nil
	}

	for _, decl := range astFile.Decls {
		switch d := decl.(type) {
		case *ast.FnDecl:
			if d.Name.Name == word {
				loc := makeLocation(uri, f, d.Name.NamePos)
				return &loc
			}
			for _, param := range d.Params {
				if param.Name.Name == word {
					loc := makeLocation(uri, f, param.Name.NamePos)
					return &loc
				}
			}
			if d.Body != nil {
				if loc := findInBlock(uri, f, d.Body, word); loc != nil {
					return loc
				}
			}

		case *ast.StructDecl:
			if d.Name.Name == word {
				loc := makeLocation(uri, f, d.Name.NamePos)
				return &loc
			}
			for _, field := range d.Fields {
				if field.Name.Name == word {
					loc := makeLocation(uri, f, field.Name.NamePos)
					return &loc
				}
			}

		case *ast.InterfaceDecl:
			if d.Name.Name == word {
				loc := makeLocation(uri, f, d.Name.NamePos)
				return &loc
			}
			for _, method := range d.Methods {
				if method.Name.Name == word {
					loc := makeLocation(uri, f, method.Name.NamePos)
					return &loc
				}
			}
			for _, field := range d.Fields {
				if field.Name.Name == word {
					loc := makeLocation(uri, f, field.Name.NamePos)
					return &loc
				}
			}
		}
	}

	return nil
}

func (s *Server) computeCompletions(uri string, text string, pos Position) []CompletionItem {
	var items []CompletionItem

	lines := strings.Split(text, "\n")
	if pos.Line < 0 || pos.Line >= len(lines) {
		return items
	}
	line := lines[pos.Line]
	if pos.Character > len(line) {
		pos.Character = len(line)
	}

	p := parser.New(text)
	astFile := p.ParseFile()
	imported := s.resolveImports(uri, astFile)

	allDecls := append([]ast.Decl{}, astFile.Decls...)
	declOrigins := make(map[ast.Decl]string)
	for _, impDoc := range imported {
		for _, decl := range impDoc.AST.Decls {
			allDecls = append(allDecls, decl)
			declOrigins[decl] = filepath.Base(impDoc.Path)
		}
	}

	findStruct := func(name string) *ast.StructDecl {
		for _, decl := range allDecls {
			if sd, ok := decl.(*ast.StructDecl); ok && sd.Name.Name == name {
				return sd
			}
		}
		return nil
	}

	// Case 1: Member completion after '.'
	prefix := line[:pos.Character]
	if strings.HasSuffix(prefix, ".") {
		identEnd := len(prefix) - 1
		identStart := identEnd
		for identStart > 0 && isIdentChar(prefix[identStart-1]) {
			identStart--
		}
		receiverIdent := prefix[identStart:identEnd]

		// 1A. Module / Package Namespace Completion (e.g., 'http.' or 'net.')
		for _, imp := range astFile.Imports {
			cleanTarget := strings.Trim(imp.Name, `"`)
			alias := imp.Alias
			modName := filepath.Base(cleanTarget)
			if alias == "" {
				alias = modName
			}
			if receiverIdent == alias || receiverIdent == modName {
				for _, impDoc := range imported {
					docMod := impDoc.AST.Module.Name
					if docMod == "" {
						docMod = filepath.Base(filepath.Dir(impDoc.Path))
						if docMod == "src" {
							docMod = filepath.Base(filepath.Dir(filepath.Dir(impDoc.Path)))
						}
					}
					if docMod == modName || (imp.Alias != "" && docMod == imp.Alias) || strings.Contains(filepath.ToSlash(impDoc.Path), "/"+cleanTarget+"/") {
						origin := filepath.Base(impDoc.Path)
						for _, decl := range impDoc.AST.Decls {
							switch d := decl.(type) {
							case *ast.StructDecl:
								detail := "struct"
								if d.IsActor {
									detail = "actor"
								}
								detail += fmt.Sprintf(" (from %s)", origin)
								items = append(items, CompletionItem{
									Label:  d.Name.Name,
									Kind:   7, // Class/Struct
									Detail: detail,
								})
							case *ast.InterfaceDecl:
								items = append(items, CompletionItem{
									Label:  d.Name.Name,
									Kind:   8, // Interface
									Detail: fmt.Sprintf("interface (from %s)", origin),
								})
							case *ast.FnDecl:
								if d.Receiver == nil && d.StaticType == nil {
									items = append(items, CompletionItem{
										Label:  d.Name.Name,
										Kind:   3, // Function
										Detail: fmt.Sprintf("function (from %s)", origin),
									})
								}
							}
						}
					}
				}
				if len(items) > 0 {
					return items
				}
			}
		}

		// 1B. Static methods on struct name
		if sd := findStruct(receiverIdent); sd != nil {
			for _, decl := range allDecls {
				if fn, ok := decl.(*ast.FnDecl); ok && fn.StaticType != nil && fn.StaticType.Name == receiverIdent {
					items = append(items, CompletionItem{
						Label:         fn.Name.Name,
						Kind:          2, // Method
						Detail:        fmt.Sprintf("static fn %s.%s(...)", receiverIdent, fn.Name.Name),
						Documentation: "Static constructor / factory method",
					})
				}
			}
			return items
		}

		// 1B0. Check if receiver is a 1D-4D strided tensor[f32]
		isTensorReceiver := false
		for _, decl := range astFile.Decls {
			if fn, ok := decl.(*ast.FnDecl); ok {
				for _, param := range fn.Params {
					if param.Name.Name == receiverIdent && strings.HasPrefix(formatTypeExpr(param.Type), "tensor[") {
						isTensorReceiver = true
						break
					}
				}
			}
			if isTensorReceiver {
				break
			}
		}
		if !isTensorReceiver {
			for _, l := range lines {
				lTrim := strings.TrimSpace(l)
				if strings.Contains(lTrim, receiverIdent) && strings.Contains(lTrim, ":=") {
					for _, marker := range []string{"tensor_fill(", "tensor_fill_3d(", "tensor_fill_4d(", ".reshape_2d(", ".reshape_3d(", ".reshape_4d(", ".transpose(", " @ ", "tensor_relu(", "tensor_add(", "tensor_scale(", "tensor_elemwise_mul(", "safetensors_load(", "gguf_tensor_dequant("} {
						if strings.Contains(lTrim, marker) {
							isTensorReceiver = true
							break
						}
					}
				}
				if isTensorReceiver {
					break
				}
			}
		}
		if isTensorReceiver {
			items = append(items,
				CompletionItem{Label: "ndim", Kind: 10, Detail: "ndim: int", Documentation: "Active rank (1 to 4) of the strided tensor"},
				CompletionItem{Label: "rows", Kind: 10, Detail: "rows: int", Documentation: "Leading row count (2D compatibility view)"},
				CompletionItem{Label: "cols", Kind: 10, Detail: "cols: int", Documentation: "Trailing column count (2D compatibility view)"},
				CompletionItem{Label: "numel", Kind: 2, Detail: "numel() -> int", Documentation: "Total element count across all active dimensions"},
				CompletionItem{Label: "dim", Kind: 2, Detail: "dim(axis: int) -> int", Documentation: "Dimension size along axis (0 <= axis < ndim)"},
				CompletionItem{Label: "stride", Kind: 2, Detail: "stride(axis: int) -> int", Documentation: "Row-major element stride along axis"},
				CompletionItem{Label: "reshape_2d", Kind: 2, Detail: "reshape_2d(d0: int, d1: int) -> tensor[f32]", Documentation: "Reshape to 2D (O(1) zero-copy view inside arena)"},
				CompletionItem{Label: "reshape_3d", Kind: 2, Detail: "reshape_3d(d0: int, d1: int, d2: int) -> tensor[f32]", Documentation: "Reshape to 3D (O(1) zero-copy view inside arena)"},
				CompletionItem{Label: "reshape_4d", Kind: 2, Detail: "reshape_4d(d0: int, d1: int, d2: int, d3: int) -> tensor[f32]", Documentation: "Reshape to 4D (O(1) zero-copy view inside arena)"},
				CompletionItem{Label: "transpose", Kind: 2, Detail: "transpose() -> tensor[f32]", Documentation: "Swap trailing two matrix axes (..., M, N) -> (..., N, M)"},
				CompletionItem{Label: "clone", Kind: 2, Detail: "clone() -> tensor[f32]", Documentation: "Allocate an independent deep copy of the tensor"},
			)
			return items
		}

		// 1B. Check if receiver is an array or slice collection
		for _, l := range lines {
			lTrim := strings.TrimSpace(l)
			if strings.Contains(lTrim, receiverIdent) && strings.Contains(lTrim, ":=") {
				if strings.Contains(lTrim, "[") && strings.Contains(lTrim, "]") {
					isFixedArray := false
					for _, prim := range []string{"int", "i64", "f32", "f64", "bool", "string"} {
						if strings.Contains(lTrim, "]"+prim+"{") {
							isFixedArray = true
							break
						}
					}
					items = append(items,
						CompletionItem{Label: "len", Kind: 10, Detail: "len: int", Documentation: "Length of the collection"},
						CompletionItem{Label: "get", Kind: 2, Detail: "get(index: int) -> !T", Documentation: "Safely get element by index with Result !T fallback"},
						CompletionItem{Label: "contains", Kind: 2, Detail: "contains(val: T) -> bool", Documentation: "Returns true if value exists in collection"},
						CompletionItem{Label: "is_empty", Kind: 2, Detail: "is_empty() -> bool", Documentation: "Returns true if collection is empty"},
					)
					if !isFixedArray {
						items = append(items,
							CompletionItem{Label: "cap", Kind: 10, Detail: "cap: int", Documentation: "Allocated backing buffer capacity"},
							CompletionItem{Label: "push", Kind: 2, Detail: "push(val: T)", Documentation: "Append element with geometric growth"},
							CompletionItem{Label: "pop", Kind: 2, Detail: "pop() -> T", Documentation: "Remove and return the last element"},
							CompletionItem{Label: "set", Kind: 2, Detail: "set(index: int, val: T)", Documentation: "Update element at index"},
							CompletionItem{Label: "clear", Kind: 2, Detail: "clear()", Documentation: "Reset slice length to 0"},
							CompletionItem{Label: "clone", Kind: 2, Detail: "clone() -> []T", Documentation: "Deep copy of the slice"},
						)
					}
					return items
				}
			}
		}

		// 1C. Inferred type for local variables
		var inferredType string
		for _, l := range lines {
			lTrim := strings.TrimSpace(l)
			if strings.Contains(lTrim, receiverIdent) && strings.Contains(lTrim, ":=") {
				for _, decl := range allDecls {
					if sd, ok := decl.(*ast.StructDecl); ok {
						if strings.Contains(lTrim, sd.Name.Name+".new") || strings.Contains(lTrim, sd.Name.Name+"{") {
							inferredType = sd.Name.Name
							break
						}
					}
				}
			}
			if inferredType != "" {
				break
			}
		}

		if inferredType == "" {
			for _, decl := range astFile.Decls {
				if fn, ok := decl.(*ast.FnDecl); ok && fn.Receiver != nil && fn.Receiver.Name.Name == receiverIdent {
					if it, ok := fn.Receiver.Type.(*ast.IdentType); ok {
						inferredType = it.Name.Name
						break
					}
				}
			}
		}

		if inferredType != "" {
			var addStructMembers func(sd *ast.StructDecl, isPromoted bool)
			addStructMembers = func(sd *ast.StructDecl, isPromoted bool) {
				if sd == nil {
					return
				}
				origin := declOrigins[sd]
				promotedNote := ""
				if isPromoted {
					promotedNote = fmt.Sprintf(" (promoted from %s)", sd.Name.Name)
				} else if origin != "" {
					promotedNote = fmt.Sprintf(" (from %s)", origin)
				}

				// Fields
				for _, f := range sd.Fields {
					items = append(items, CompletionItem{
						Label:         f.Name.Name,
						Kind:          5, // Field
						Detail:        fmt.Sprintf("%s: %s%s", f.Name.Name, formatTypeExpr(f.Type), promotedNote),
						Documentation: fmt.Sprintf("Struct field [%s]", f.Access.String()),
					})
				}

				// Methods having this struct as receiver
				for _, decl := range allDecls {
					if fn, ok := decl.(*ast.FnDecl); ok && fn.Receiver != nil {
						if it, ok := fn.Receiver.Type.(*ast.IdentType); ok && it.Name.Name == sd.Name.Name {
							items = append(items, CompletionItem{
								Label:         fn.Name.Name,
								Kind:          2, // Method
								Detail:        fmt.Sprintf("fn (%s) %s(...)%s", it.Name.Name, fn.Name.Name, promotedNote),
								Documentation: "Receiver method",
							})
						}
					}
				}

				// Embedded structs
				for _, emb := range sd.Embeds {
					embName, _ := getEmbedInfo(emb)
					if embSd := findStruct(embName); embSd != nil {
						addStructMembers(embSd, true)
					}
				}
			}

			if targetSd := findStruct(inferredType); targetSd != nil {
				addStructMembers(targetSd, false)
				return items
			}
		}
	}

	// Case 2: General completion
	for _, decl := range allDecls {
		switch d := decl.(type) {
		case *ast.StructDecl:
			origin := declOrigins[d]
			detail := "struct"
			if d.IsActor {
				detail = "actor"
			}
			if origin != "" {
				detail += fmt.Sprintf(" (from %s)", origin)
			}
			items = append(items, CompletionItem{
				Label:  d.Name.Name,
				Kind:   7, // Class/Struct
				Detail: detail,
			})
		case *ast.FnDecl:
			if d.Receiver == nil && d.StaticType == nil {
				origin := declOrigins[d]
				detail := "function"
				if origin != "" {
					detail += fmt.Sprintf(" (from %s)", origin)
				}
				items = append(items, CompletionItem{
					Label:  d.Name.Name,
					Kind:   3, // Function
					Detail: detail,
				})
			}
		}
	}

	for _, kw := range []string{"module", "import", "pub", "mut", "fn", "struct", "actor", "interface", "return", "if", "else", "switch", "case", "default", "for", "arena", "true", "false"} {
		items = append(items, CompletionItem{
			Label:  kw,
			Kind:   14, // Keyword
			Detail: "keyword",
		})
	}

	for _, bi := range []string{"println", "print", "eprintln", "panic", "time_now_us", "time_now_ms", "tensor_fill", "tensor_fill_3d", "tensor_fill_4d", "tensor_scale", "tensor_sum", "tensor_relu", "tensor_add", "tensor_elemwise_mul", "tensor_transpose", "gui_init", "gui_begin", "gui_end", "gui_button", "gui_clear", "file_exists", "read_file", "write_file"} {
		items = append(items, CompletionItem{
			Label:  bi,
			Kind:   3, // Function
			Detail: "builtin",
		})
	}

	return items
}

func (s *Server) computeDocumentSymbols(uri string, text string) []DocumentSymbol {
	symbols := make([]DocumentSymbol, 0)
	p := parser.New(text)
	astFile := p.ParseFile()
	f := source.New(uri, text)

	makeRange := func(span token.Span) Range {
		sl, sc := f.Location(span.Start)
		el, ec := f.Location(span.End)
		if el < sl || (el == sl && ec <= sc) {
			el = sl
			ec = sc + 1
		}
		return Range{
			Start: Position{Line: sl - 1, Character: sc - 1},
			End:   Position{Line: el - 1, Character: ec - 1},
		}
	}

	if astFile.Module.Name != "" {
		rng := makeRange(astFile.Module.NamePos)
		symbols = append(symbols, DocumentSymbol{
			Name:           "module " + astFile.Module.Name,
			Kind:           2, // Module
			Range:          rng,
			SelectionRange: rng,
		})
	}

	for _, imp := range astFile.Imports {
		rng := makeRange(imp.NamePos)
		name := "import " + imp.Path
		if imp.Alias != "" {
			name += " as " + imp.Alias
		}
		symbols = append(symbols, DocumentSymbol{
			Name:           name,
			Kind:           3, // Namespace
			Range:          rng,
			SelectionRange: rng,
		})
	}

	for _, decl := range astFile.Decls {
		switch d := decl.(type) {
		case *ast.StructDecl:
			rng := makeRange(d.DeclSpan)
			selRng := makeRange(d.Name.NamePos)
			kind := 23 // Struct
			var children []DocumentSymbol
			for _, field := range d.Fields {
				fRng := makeRange(field.Name.NamePos)
				children = append(children, DocumentSymbol{
					Name:           field.Name.Name,
					Detail:         formatTypeExpr(field.Type),
					Kind:           8, // Field
					Range:          fRng,
					SelectionRange: fRng,
				})
			}
			for _, emb := range d.Embeds {
				embName, embPos := getEmbedInfo(emb)
				eRng := makeRange(embPos)
				children = append(children, DocumentSymbol{
					Name:           embName,
					Detail:         "embed",
					Kind:           8,
					Range:          eRng,
					SelectionRange: eRng,
				})
			}
			symbols = append(symbols, DocumentSymbol{
				Name:           d.Name.Name,
				Kind:           kind,
				Range:          rng,
				SelectionRange: selRng,
				Children:       children,
			})

		case *ast.FnDecl:
			rng := makeRange(d.DeclSpan)
			selRng := makeRange(d.Name.NamePos)
			kind := 12 // Function
			name := d.Name.Name
			if d.Receiver != nil {
				kind = 6 // Method
				name = fmt.Sprintf("(%s) %s", formatTypeExpr(d.Receiver.Type), d.Name.Name)
			} else if d.StaticType != nil {
				name = fmt.Sprintf("%s.%s", d.StaticType.Name, d.Name.Name)
			}
			symbols = append(symbols, DocumentSymbol{
				Name:           name,
				Kind:           kind,
				Range:          rng,
				SelectionRange: selRng,
			})

		case *ast.InterfaceDecl:
			rng := makeRange(d.DeclSpan)
			selRng := makeRange(d.Name.NamePos)
			symbols = append(symbols, DocumentSymbol{
				Name:           d.Name.Name,
				Kind:           11, // Interface
				Range:          rng,
				SelectionRange: selRng,
			})
		}
	}

	return symbols
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
		case *ast.SwitchStmt:
			for _, cs := range st.Cases {
				dummyBlock := &ast.BlockStmt{Stmts: cs.Body}
				if loc := findInBlock(uri, f, dummyBlock, word); loc != nil {
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

