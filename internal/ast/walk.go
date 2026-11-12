package ast

// CountIdentUses adds, for every identifier referenced inside node, the number of references to
// counts. It descends into nested blocks, `or { ... }` blocks and defers. Synthetic cleanup
// statements injected by autofree are ignored.
//
// Counting is purely by name (shadowed variables are counted too), which makes it conservative
// for its one consumer: deciding that a variable is never used again.
func CountIdentUses(node Node, counts map[string]int) {
	switch n := node.(type) {
	case nil:
		return

	// --- statements ---
	case *BlockStmt:
		if n == nil {
			return
		}
		for _, st := range n.Stmts {
			CountIdentUses(st, counts)
		}
	case *VarDeclStmt:
		CountIdentUses(n.Value, counts)
	case *AssignStmt:
		CountIdentUses(n.Target, counts)
		CountIdentUses(n.Value, counts)
	case *ReturnStmt:
		if n.Value != nil {
			CountIdentUses(n.Value, counts)
		}
	case *ExprStmt:
		CountIdentUses(n.Expr, counts)
	case *IfStmt:
		CountIdentUses(n.Cond, counts)
		CountIdentUses(n.Then, counts)
		if n.Else != nil {
			CountIdentUses(n.Else, counts)
		}
	case *SwitchStmt:
		if n.Tag != nil {
			CountIdentUses(n.Tag, counts)
		}
		for i := range n.Cases {
			for _, e := range n.Cases[i].Exprs {
				CountIdentUses(e, counts)
			}
			for _, st := range n.Cases[i].Body {
				CountIdentUses(st, counts)
			}
		}
	case *ForStmt:
		if n.Init != nil {
			CountIdentUses(n.Init, counts)
		}
		if n.Cond != nil {
			CountIdentUses(n.Cond, counts)
		}
		if n.Post != nil {
			CountIdentUses(n.Post, counts)
		}
		CountIdentUses(n.Body, counts)
	case *ForInStmt:
		CountIdentUses(n.Iter, counts)
		CountIdentUses(n.Body, counts)
	case *ArenaStmt:
		CountIdentUses(n.Body, counts)
	case *DeferStmt:
		if n.Body != nil {
			CountIdentUses(n.Body, counts)
		}
	case *BranchStmt, *DeferFreeStmt:
		return

	// --- expressions ---
	case *IdentExpr:
		counts[n.Ident.Name]++
	case *LiteralExpr:
		return
	case *StringInterpExpr:
		for _, e := range n.Exprs {
			CountIdentUses(e, counts)
		}
	case *SelectorExpr:
		CountIdentUses(n.Left, counts)
	case *StructInitExpr:
		for i := range n.Fields {
			CountIdentUses(n.Fields[i].Value, counts)
		}
	case *CallExpr:
		CountIdentUses(n.Callee, counts)
		for i := range n.Args {
			if n.Args[i].Expr != nil {
				CountIdentUses(n.Args[i].Expr, counts)
			}
		}
		if n.OrBlock != nil {
			CountIdentUses(n.OrBlock, counts)
		}
	case *BinaryExpr:
		CountIdentUses(n.Left, counts)
		CountIdentUses(n.Right, counts)
	case *UnaryExpr:
		CountIdentUses(n.Expr, counts)
	case *IndexExpr:
		CountIdentUses(n.Left, counts)
		CountIdentUses(n.Index, counts)
	case *ArrayLitExpr:
		for _, e := range n.Elems {
			CountIdentUses(e, counts)
		}
	}
}
