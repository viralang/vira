package autofree

import (
	"fmt"
	"strings"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/checker"
	"github.com/viralang/vira/internal/token"
)

type EscapeState uint8

const (
	StateScopedLocal       EscapeState = iota // Freed deterministically when its lexical scope exits
	StateEscapesViaReturn                     // Returned to caller -> ownership transferred, do NOT free
	StateMovedToContainer                     // Stored into a struct literal or field -> container owns it
	StateManagedByArena                       // Allocated inside `arena { ... }` -> freed in O(1) by slab
)

func (s EscapeState) String() string {
	switch s {
	case StateEscapesViaReturn:
		return "ESCAPES_VIA_RETURN (ownership -> caller)"
	case StateMovedToContainer:
		return "MOVED_TO_CONTAINER (ownership -> struct/iface)"
	case StateManagedByArena:
		return "ARENA_MANAGED (O(1) slab free)"
	default:
		return "SCOPED_LOCAL (autofree injected)"
	}
}

type TrackedVar struct {
	Name        string
	Type        checker.Type
	CleanupKind ast.CleanupKind
	State       EscapeState
	DeclSpan    token.Span
}

type LexicalScope struct {
	Parent  *LexicalScope
	InArena bool
	Vars    []*TrackedVar
}

func (ls *LexicalScope) Find(name string) *TrackedVar {
	for cur := ls; cur != nil; cur = cur.Parent {
		for _, v := range cur.Vars {
			if v.Name == name {
				return v
			}
		}
	}
	return nil
}

// FnReport stores the escape analysis audit trail for a single function.
type FnReport struct {
	FnName        string
	TrackedVars   []*TrackedVar
	InjectedFrees []*ast.DeferFreeStmt
}

type Analyzer struct {
	chk             *checker.Checker
	StructNeedsFree map[string]bool
	Reports         []*FnReport
	arenaCounter    int
	tmpCounter      int
}

func New(chk *checker.Checker) *Analyzer {
	a := &Analyzer{
		chk:             chk,
		StructNeedsFree: make(map[string]bool),
	}
	a.computeStructDestructors()
	return a
}

// computeStructDestructors determines which structs transitively own heap memory.
func (a *Analyzer) computeStructDestructors() {
	structs := a.chk.Structs()
	changed := true
	for changed {
		changed = false
		for name, st := range structs {
			if a.StructNeedsFree[name] {
				continue
			}
			if st.IsActor {
				a.StructNeedsFree[name] = true
				changed = true
				continue
			}
			for _, f := range st.Fields {
				if ok, _ := a.typeNeedsCleanup(f.Type); ok {
					a.StructNeedsFree[name] = true
					changed = true
					break
				}
			}
			if !a.StructNeedsFree[name] {
				for _, emb := range st.Embeds {
					if a.StructNeedsFree[emb.Name] {
						a.StructNeedsFree[name] = true
						changed = true
						break
					}
				}
			}
		}
	}
}

func (a *Analyzer) typeNeedsCleanup(t checker.Type) (bool, ast.CleanupKind) {
	if t == nil {
		return false, 0
	}
	switch v := t.(type) {
	case *checker.PrimitiveType:
		if v.Kind == checker.PrimString {
			return true, ast.CleanupString
		}
		return false, 0
	case *checker.TensorType:
		return true, ast.CleanupTensor
	case *checker.FutureType:
		return true, ast.CleanupFuture
	case *checker.StructType:
		if v.IsActor {
			return true, ast.CleanupActor
		}
		if a.StructNeedsFree[v.Name] {
			return true, ast.CleanupStruct
		}
		return false, 0
	case *checker.InterfaceType:
		return true, ast.CleanupInterface
	case *checker.SliceType:
		return true, ast.CleanupSlice
	default:
		return false, 0
	}
}

// AnalyzeFile runs escape analysis and injects DeferFreeStmt nodes across all functions in the file.
func (a *Analyzer) AnalyzeFile(file *ast.File) {
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FnDecl); ok {
			if fn.Body == nil {
				continue
			}
			report := &FnReport{FnName: a.formatFnName(fn)}
			rootScope := &LexicalScope{}
			a.rewriteBlock(fn.Body, rootScope, true, report)
			a.Reports = append(a.Reports, report)
		}
	}
}

func (a *Analyzer) formatFnName(fn *ast.FnDecl) string {
	if fn.Receiver != nil {
		recvType := fn.Receiver.Type
		if identType, ok := recvType.(*ast.IdentType); ok {
			if st := a.chk.Structs()[identType.Name.Name]; st != nil {
				return fmt.Sprintf("(%s).%s", st.Name, fn.Name.Name)
			}
		}
		return fmt.Sprintf("(recv).%s", fn.Name.Name)
	}
	if fn.StaticType != nil {
		return fmt.Sprintf("%s.%s", fn.StaticType.Name, fn.Name.Name)
	}
	return fn.Name.Name
}

// rewriteBlock analyzes statements in `block`, marks escaping variables, and injects LIFO DeferFreeStmts.
func (a *Analyzer) rewriteBlock(
	block *ast.BlockStmt,
	scope *LexicalScope,
	isFnRoot bool,
	report *FnReport,
) {
	if block == nil {
		return
	}
	var rewritten []ast.Stmt
	diverged := false

	for _, stmt := range block.Stmts {
		switch s := stmt.(type) {
		case *ast.VarDeclStmt:
			// 1. Check if the RHS expression contains an `or { ... }` block or transfers ownership
			a.inspectExprMovesAndOrBlocks(s.Value, scope, report)

			preStmts, postStmts := a.hoistTemporaryStrings(s.Value, scope, report, true)
			rewritten = append(rewritten, preStmts...)

			// 2. Register variable in current scope if its type owns heap memory
			valType := a.chk.ExprTypes[s.Value]
			if needsFree, kind := a.typeNeedsCleanup(valType); needsFree && s.Name.Name != "_" {
				// Variables initialized by borrowing from containers or struct fields do not own the memory
				isBorrow := false
				if _, isIdx := s.Value.(*ast.IndexExpr); isIdx {
					isBorrow = true
				} else if _, isSel := s.Value.(*ast.SelectorExpr); isSel {
					isBorrow = true
				}
				if !isBorrow {
					state := StateScopedLocal
					if scope.InArena {
						state = StateManagedByArena
					}
					tv := &TrackedVar{
						Name:        s.Name.Name,
						Type:        valType,
						CleanupKind: kind,
						State:       state,
						DeclSpan:    s.StmtSpan,
					}
					scope.Vars = append(scope.Vars, tv)
					report.TrackedVars = append(report.TrackedVars, tv)
				}
			}
			rewritten = append(rewritten, s)
			rewritten = append(rewritten, postStmts...)

		case *ast.AssignStmt:
			a.inspectExprMovesAndOrBlocks(s.Value, scope, report)
			preStmts, postStmts := a.hoistTemporaryStrings(s.Value, scope, report, true)
			rewritten = append(rewritten, preStmts...)
			rewritten = append(rewritten, s)
			rewritten = append(rewritten, postStmts...)

		case *ast.IfStmt:
			thenScope := &LexicalScope{Parent: scope, InArena: scope.InArena}
			a.rewriteBlock(s.Then, thenScope, false, report)
			if s.Else != nil {
				elseScope := &LexicalScope{Parent: scope, InArena: scope.InArena}
				a.rewriteBlock(s.Else, elseScope, false, report)
			}
			rewritten = append(rewritten, s)

		case *ast.SwitchStmt:
			if s.Tag != nil {
				a.inspectExprMovesAndOrBlocks(s.Tag, scope, report)
			}
			for i := range s.Cases {
				cs := &s.Cases[i]
				for _, expr := range cs.Exprs {
					a.inspectExprMovesAndOrBlocks(expr, scope, report)
				}
				caseScope := &LexicalScope{Parent: scope, InArena: scope.InArena}
				dummyBlock := &ast.BlockStmt{Stmts: cs.Body, BlockSpan: cs.CaseSpan}
				a.rewriteBlock(dummyBlock, caseScope, false, report)
				cs.Body = dummyBlock.Stmts
			}
			rewritten = append(rewritten, s)

		case *ast.ForStmt:
			loopScope := &LexicalScope{Parent: scope, InArena: scope.InArena}
			if s.Init != nil {
				if vdecl, ok := s.Init.(*ast.VarDeclStmt); ok {
					valType := a.chk.ExprTypes[vdecl.Value]
					if needsFree, kind := a.typeNeedsCleanup(valType); needsFree && vdecl.Name.Name != "_" {
						state := StateScopedLocal
						if loopScope.InArena {
							state = StateManagedByArena
						}
						tv := &TrackedVar{
							Name:        vdecl.Name.Name,
							Type:        valType,
							CleanupKind: kind,
							State:       state,
							DeclSpan:    vdecl.StmtSpan,
						}
						loopScope.Vars = append(loopScope.Vars, tv)
						report.TrackedVars = append(report.TrackedVars, tv)
					}
				}
			}
			a.rewriteBlock(s.Body, loopScope, false, report)
			rewritten = append(rewritten, s)

		case *ast.ForInStmt:
			loopScope := &LexicalScope{Parent: scope, InArena: scope.InArena}
			a.rewriteBlock(s.Body, loopScope, false, report)
			rewritten = append(rewritten, s)

		case *ast.BranchStmt:
			rewritten = append(rewritten, s)

		case *ast.ArenaStmt:
			a.arenaCounter++
			s.ArenaVar = fmt.Sprintf("_vira_arena_%d", a.arenaCounter)
			arenaScope := &LexicalScope{Parent: scope, InArena: true}
			a.rewriteBlock(s.Body, arenaScope, false, report)

			// Inject O(1) slab teardown at the end of the `arena { ... }` block
			arenaFree := &ast.DeferFreeStmt{
				VarName:  s.ArenaVar,
				TypeName: "vira_arena",
				Kind:     ast.CleanupArena,
				Reason:   "arena block exit (O(1) slab deallocation)",
				StmtSpan: s.StmtSpan,
			}
			s.Body.Stmts = append(s.Body.Stmts, arenaFree)
			report.InjectedFrees = append(report.InjectedFrees, arenaFree)
			rewritten = append(rewritten, s)

		case *ast.ReturnStmt:
			if ident, ok := s.Value.(*ast.IdentExpr); ok {
				if tv := scope.Find(ident.Ident.Name); tv != nil {
					tv.State = StateEscapesViaReturn
				}
			} else if s.Value != nil {
				a.inspectExprMovesAndOrBlocks(s.Value, scope, report)

				// If returning a non-literal expression (e.g. `return slice.to_string()` or `return "hello ${x}"`),
				// hoist it into a temporary variable so it is evaluated BEFORE unwinding scope destructors,
				// preventing any use-after-free of local variables!
				isErrorCall := false
				if call, isCall := s.Value.(*ast.CallExpr); isCall {
					if id, isId := call.Callee.(*ast.IdentExpr); isId && id.Ident.Name == "error" {
						isErrorCall = true
					}
				}
				if !isErrorCall {
					if _, isLit := s.Value.(*ast.LiteralExpr); !isLit {
						a.tmpCounter++
						retType := a.chk.ExprTypes[s.Value]
						tmpName := fmt.Sprintf("_vira_ret_val_%d", a.tmpCounter)
						tmpIdent := ast.Ident{Name: tmpName, NamePos: s.Value.Span()}
						varDecl := &ast.VarDeclStmt{
							Name:     tmpIdent,
							Value:    s.Value,
							IsMut:    false,
							StmtSpan: s.Value.Span(),
						}
						a.chk.ExprTypes[varDecl.Value] = retType
						identExpr := &ast.IdentExpr{Ident: tmpIdent}
						a.chk.ExprTypes[identExpr] = retType

						rewritten = append(rewritten, varDecl)
						s.Value = identExpr
					}
				}
			}

			// Unwind all active scopes up to function root in LIFO order before `return`
			frees := a.collectScopeChainFrees(scope, "before return")
			for _, f := range frees {
				rewritten = append(rewritten, f)
				report.InjectedFrees = append(report.InjectedFrees, f)
			}
			rewritten = append(rewritten, s)
			diverged = true

		case *ast.ExprStmt:
			a.inspectExprMovesAndOrBlocks(s.Expr, scope, report)
			preStmts, postStmts := a.hoistTemporaryStrings(s.Expr, scope, report, false)
			rewritten = append(rewritten, preStmts...)
			rewritten = append(rewritten, s)
			rewritten = append(rewritten, postStmts...)

		default:
			rewritten = append(rewritten, s)
		}
	}

	// If control flow reaches the end of the block without an explicit `return`,
	// inject LIFO cleanup for variables declared in THIS lexical block.
	if !diverged {
		reason := "block scope exit"
		if isFnRoot {
			reason = "function scope exit"
		}
		frees := a.collectSingleScopeFrees(scope, reason)
		for _, f := range frees {
			rewritten = append(rewritten, f)
			report.InjectedFrees = append(report.InjectedFrees, f)
		}
	}

	block.Stmts = rewritten
}

func (a *Analyzer) inspectExprMovesAndOrBlocks(expr ast.Expr, scope *LexicalScope, report *FnReport) {
	if expr == nil {
		return
	}
	switch e := expr.(type) {
	case *ast.StructInitExpr:
		// Any local variable passed into a struct field (`Invoice{ id: local_id }`) transfers ownership
		for _, f := range e.Fields {
			if ident, ok := f.Value.(*ast.IdentExpr); ok {
				if tv := scope.Find(ident.Ident.Name); tv != nil && tv.State == StateScopedLocal {
					tv.State = StateMovedToContainer
				}
			} else {
				a.inspectExprMovesAndOrBlocks(f.Value, scope, report)
			}
		}

	case *ast.CallExpr:
		// If this is an interface boxing cast `Payable(inv)`, the interface fat pointer borrows/boxes `inv`
		if identCallee, ok := e.Callee.(*ast.IdentExpr); ok {
			if _, isIface := a.chk.ExprTypes[e].(*checker.InterfaceType); isIface && len(e.Args) == 1 {
				_ = identCallee
				if argIdent, ok := e.Args[0].Expr.(*ast.IdentExpr); ok {
					if tv := scope.Find(argIdent.Ident.Name); tv != nil && tv.State == StateScopedLocal {
						tv.State = StateMovedToContainer
					}
				}
			}
		}
		for _, arg := range e.Args {
			a.inspectExprMovesAndOrBlocks(arg.Expr, scope, report)
		}
		if e.OrBlock != nil {
			orScope := &LexicalScope{Parent: scope, InArena: scope.InArena}
			a.rewriteBlock(e.OrBlock, orScope, false, report)
		}

	case *ast.UnaryExpr:
		a.inspectExprMovesAndOrBlocks(e.Expr, scope, report)
	}
}

// hoistTemporaryStrings extracts unassigned temporary heap strings (e.g., *ast.StringInterpExpr
// or *ast.CallExpr returning string passed as call arguments) into scoped synthetic variables,
// injecting immediate statement-exit DeferFreeStmt cleanup for zero-leak memory management.
func (a *Analyzer) hoistTemporaryStrings(
	expr ast.Expr,
	scope *LexicalScope,
	report *FnReport,
	isRootAssigned bool,
) ([]ast.Stmt, []ast.Stmt) {
	if scope.InArena || expr == nil {
		return nil, nil
	}

	var preStmts []ast.Stmt
	var postFrees []*ast.DeferFreeStmt

	// If root expr itself is an unassigned StringInterpExpr (e.g. in ExprStmt: `'hello ${x}'`)
	if !isRootAssigned {
		if interp, ok := expr.(*ast.StringInterpExpr); ok {
			a.tmpCounter++
			tmpName := fmt.Sprintf("_vira_tmp_str_%d", a.tmpCounter)
			tmpIdent := ast.Ident{Name: tmpName, NamePos: interp.Span()}
			varDecl := &ast.VarDeclStmt{
				Name:     tmpIdent,
				Value:    interp,
				IsMut:    false,
				StmtSpan: interp.Span(),
			}
			a.chk.ExprTypes[varDecl.Value] = checker.TypeString
			identExpr := &ast.IdentExpr{Ident: tmpIdent}
			a.chk.ExprTypes[identExpr] = checker.TypeString

			freeStmt := &ast.DeferFreeStmt{
				VarName:  tmpName,
				TypeName: "string",
				Kind:     ast.CleanupString,
				Reason:   "zero-leak temporary string statement cleanup",
				StmtSpan: interp.Span(),
			}
			preStmts = append(preStmts, varDecl)
			postFrees = append(postFrees, freeStmt)
			report.InjectedFrees = append(report.InjectedFrees, freeStmt)
			return preStmts, []ast.Stmt{freeStmt}
		}
	}

	a.walkAndHoistCallArgs(expr, &preStmts, &postFrees, report)

	var postStmts []ast.Stmt
	// LIFO cleanup
	for i := len(postFrees) - 1; i >= 0; i-- {
		postStmts = append(postStmts, postFrees[i])
	}
	return preStmts, postStmts
}

func (a *Analyzer) walkAndHoistCallArgs(
	expr ast.Expr,
	preStmts *[]ast.Stmt,
	postFrees *[]*ast.DeferFreeStmt,
	report *FnReport,
) {
	if expr == nil {
		return
	}
	switch e := expr.(type) {
	case *ast.CallExpr:
		isContainerStore := false
		if sel, isSel := e.Callee.(*ast.SelectorExpr); isSel {
			switch sel.Field.Name {
			case "push", "append", "set", "insert", "add", "set_header":
				isContainerStore = true
			}
		}
		if isContainerStore {
			return
		}
		for i := range e.Args {
			arg := &e.Args[i]
			if arg.Expr == nil {
				continue
			}
			// 1. Recursively walk nested subexpressions
			a.walkAndHoistCallArgs(arg.Expr, preStmts, postFrees, report)

			// 2. Hoist StringInterpExpr passed as argument
			if interp, ok := arg.Expr.(*ast.StringInterpExpr); ok {
				a.tmpCounter++
				tmpName := fmt.Sprintf("_vira_tmp_str_%d", a.tmpCounter)
				tmpIdent := ast.Ident{Name: tmpName, NamePos: interp.Span()}
				varDecl := &ast.VarDeclStmt{
					Name:     tmpIdent,
					Value:    interp,
					IsMut:    false,
					StmtSpan: interp.Span(),
				}
				a.chk.ExprTypes[varDecl.Value] = checker.TypeString
				identExpr := &ast.IdentExpr{Ident: tmpIdent}
				a.chk.ExprTypes[identExpr] = checker.TypeString

				freeStmt := &ast.DeferFreeStmt{
					VarName:  tmpName,
					TypeName: "string",
					Kind:     ast.CleanupString,
					Reason:   "zero-leak temporary string argument cleanup",
					StmtSpan: interp.Span(),
				}
				*preStmts = append(*preStmts, varDecl)
				*postFrees = append(*postFrees, freeStmt)
				report.InjectedFrees = append(report.InjectedFrees, freeStmt)
				arg.Expr = identExpr
			} else if callArg, isCall := arg.Expr.(*ast.CallExpr); isCall {
				// Do not hoist and free .clone() calls because clone() produces an owned copy
				if sel, isSel := callArg.Callee.(*ast.SelectorExpr); isSel && sel.Field.Name == "clone" {
					continue
				}
				// 3. Hoist CallExpr returning string passed as argument
				retType := a.chk.ExprTypes[callArg]
				if retType != nil && retType.Equals(checker.TypeString) {
					a.tmpCounter++
					tmpName := fmt.Sprintf("_vira_tmp_str_%d", a.tmpCounter)
					tmpIdent := ast.Ident{Name: tmpName, NamePos: callArg.Span()}
					varDecl := &ast.VarDeclStmt{
						Name:     tmpIdent,
						Value:    callArg,
						IsMut:    false,
						StmtSpan: callArg.Span(),
					}
					a.chk.ExprTypes[varDecl.Value] = checker.TypeString
					identExpr := &ast.IdentExpr{Ident: tmpIdent}
					a.chk.ExprTypes[identExpr] = checker.TypeString

					freeStmt := &ast.DeferFreeStmt{
						VarName:  tmpName,
						TypeName: "string",
						Kind:     ast.CleanupString,
						Reason:   "zero-leak temporary string call argument cleanup",
						StmtSpan: callArg.Span(),
					}
					*preStmts = append(*preStmts, varDecl)
					*postFrees = append(*postFrees, freeStmt)
					report.InjectedFrees = append(report.InjectedFrees, freeStmt)
					arg.Expr = identExpr
				}
			}
		}
		if sel, ok := e.Callee.(*ast.SelectorExpr); ok {
			a.walkAndHoistCallArgs(sel.Left, preStmts, postFrees, report)
		}

	case *ast.BinaryExpr:
		a.walkAndHoistCallArgs(e.Left, preStmts, postFrees, report)
		a.walkAndHoistCallArgs(e.Right, preStmts, postFrees, report)

	case *ast.UnaryExpr:
		a.walkAndHoistCallArgs(e.Expr, preStmts, postFrees, report)

	case *ast.SelectorExpr:
		a.walkAndHoistCallArgs(e.Left, preStmts, postFrees, report)
	}
}

// collectSingleScopeFrees emits LIFO destructors for variables in a single block scope.
func (a *Analyzer) collectSingleScopeFrees(scope *LexicalScope, reason string) []*ast.DeferFreeStmt {
	var frees []*ast.DeferFreeStmt
	for i := len(scope.Vars) - 1; i >= 0; i-- {
		tv := scope.Vars[i]
		if tv.State == StateScopedLocal {
			frees = append(frees, &ast.DeferFreeStmt{
				VarName:  tv.Name,
				TypeName: tv.Type.String(),
				Kind:     tv.CleanupKind,
				Reason:   reason,
				StmtSpan: tv.DeclSpan,
			})
		}
	}
	return frees
}

// collectScopeChainFrees walks from the current scope up to the function root in LIFO order for early returns.
func (a *Analyzer) collectScopeChainFrees(scope *LexicalScope, reason string) []*ast.DeferFreeStmt {
	var frees []*ast.DeferFreeStmt
	for cur := scope; cur != nil; cur = cur.Parent {
		frees = append(frees, a.collectSingleScopeFrees(cur, reason)...)
	}
	return frees
}

// FormatReport renders a human-readable summary of struct destructors, escape states, and injected frees.
func (a *Analyzer) FormatReport() string {
	var out strings.Builder
	out.WriteString("1. Structs Requiring Generated `_autofree` Destructors:\n")
	for name, needs := range a.StructNeedsFree {
		if needs {
			out.WriteString(fmt.Sprintf("   ├── struct %-12s -> requires `%s_autofree(&self)` (owns heap fields)\n", name, name))
		} else {
			out.WriteString(fmt.Sprintf("   ├── struct %-12s -> trivial stack value (0 heap fields, no destructor)\n", name))
		}
	}

	out.WriteString("\n2. Function Escape Analysis & Injected LIFO Destructors:\n")
	for _, rep := range a.Reports {
		out.WriteString(fmt.Sprintf("\n   [fn %s]\n", rep.FnName))
		if len(rep.TrackedVars) == 0 && len(rep.InjectedFrees) == 0 {
			out.WriteString("     └── (pure stack primitives — 0 heap allocations)\n")
			continue
		}
		for _, tv := range rep.TrackedVars {
			out.WriteString(fmt.Sprintf("     ├── var %-12s (%-8s) : %s\n", tv.Name, tv.Type, tv.State))
		}
		for _, df := range rep.InjectedFrees {
			out.WriteString(fmt.Sprintf("     └── INJECTED FREE: %s(&%s) [%s] (%s)\n",
				df.Kind, df.VarName, df.TypeName, df.Reason))
		}
	}
	return out.String()
}
