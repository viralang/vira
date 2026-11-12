package ast

// pureCastCallees are conversion "calls" that have no side effects of their own.
var pureCastCallees = map[string]bool{
	"int": true, "i64": true, "u8": true, "byte": true, "f32": true, "f64": true, "bf16": true,
	"bool": true, "string": true, "voidptr": true,
	"c_char": true, "c_uchar": true, "c_short": true, "c_ushort": true, "c_int": true, "c_uint": true,
	"c_long": true, "c_ulong": true, "c_longlong": true, "c_ulonglong": true, "c_size": true,
	"c_ssize": true, "c_float": true, "c_double": true,
}

// ExprMayHaveSideEffects reports whether evaluating expr can run user code: any call other than
// a plain numeric conversion or a `.clone()`. It is used to keep call arguments evaluated in
// source order (C leaves argument evaluation order unspecified).
//
// `unsupported` is set when the expression contains `or { ... }` or `!` propagation, whose
// lowering cannot be moved into a temporary safely; callers then leave the call untouched.
func ExprMayHaveSideEffects(expr Expr, unsupported *bool) bool {
	switch e := expr.(type) {
	case *CallExpr:
		if e.OrBlock != nil || e.Propagate {
			if unsupported != nil {
				*unsupported = true
			}
			return true
		}
		argsEffect := false
		for _, a := range e.Args {
			if ExprMayHaveSideEffects(a.Expr, unsupported) {
				argsEffect = true
			}
		}
		if id, ok := e.Callee.(*IdentExpr); ok && pureCastCallees[id.Ident.Name] {
			return argsEffect
		}
		if sel, ok := e.Callee.(*SelectorExpr); ok && sel.Field.Name == "clone" && len(e.Args) == 0 {
			return ExprMayHaveSideEffects(sel.Left, unsupported)
		}
		return true
	case *BinaryExpr:
		l := ExprMayHaveSideEffects(e.Left, unsupported)
		r := ExprMayHaveSideEffects(e.Right, unsupported)
		return l || r
	case *UnaryExpr:
		return ExprMayHaveSideEffects(e.Expr, unsupported)
	case *SelectorExpr:
		return ExprMayHaveSideEffects(e.Left, unsupported)
	case *IndexExpr:
		l := ExprMayHaveSideEffects(e.Left, unsupported)
		r := ExprMayHaveSideEffects(e.Index, unsupported)
		return l || r
	case *StringInterpExpr:
		res := false
		for _, sub := range e.Exprs {
			if ExprMayHaveSideEffects(sub, unsupported) {
				res = true
			}
		}
		return res
	case *StructInitExpr:
		res := false
		for _, f := range e.Fields {
			if ExprMayHaveSideEffects(f.Value, unsupported) {
				res = true
			}
		}
		return res
	case *ArrayLitExpr:
		res := false
		for _, el := range e.Elems {
			if ExprMayHaveSideEffects(el, unsupported) {
				res = true
			}
		}
		return res
	}
	return false
}
