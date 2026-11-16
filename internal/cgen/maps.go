package cgen

// C lowering of `map[K]V`.
//
// Every map is the runtime's untyped `vira_map` (runtime/rt/map/map.h). What
// depends on the value type V lives in a per-type `vira_map_ops` table that is
// generated here: the value size and the clone / free code for V, so the map
// owns its values exactly like a slice owns its elements. Keys are passed to the
// runtime as (int64_t ikey, vira_string skey): integer-like and enum keys use
// ikey, string keys use skey.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/checker"
	"github.com/viralang/vira/internal/token"
)

const mapOpsMarker = "/*__VIRA_MAP_OPS__*/\n"

func mapKeyIsString(mt *checker.MapType) bool {
	return checker.ExactEquals(mt.Key, checker.TypeString)
}

// mapOpsTag names the ops table for a map type. Tables only depend on V and on
// whether keys are strings.
func (g *Generator) mapOpsTag(mt *checker.MapType) string {
	kind := "i"
	if mapKeyIsString(mt) {
		kind = "s"
	}
	cv := strings.NewReplacer("*", "ptr", " ", "_").Replace(g.cType(mt.Value))
	return kind + "_" + cv
}

// registerMapTypes records a map type before emission: its ops table, and the slice
// types that `keys()`, `values()` and slice-valued entries need.
func (g *Generator) registerMapTypes(mt *checker.MapType) {
	tag := g.mapOpsTag(mt)
	if _, seen := g.mapOps[tag]; seen {
		return
	}
	g.mapOps[tag] = mt
	g.registerSliceType(&checker.SliceType{Elem: mt.Key})
	g.registerSliceType(&checker.SliceType{Elem: mt.Value})
	switch v := mt.Value.(type) {
	case *checker.SliceType:
		g.registerSliceType(v)
	case *checker.MapType:
		g.registerMapTypes(v)
	}
}

func (g *Generator) mapOpsRef(mt *checker.MapType) string {
	g.registerMapTypes(mt)
	return "&_vira_mapops_" + g.mapOpsTag(mt)
}

func (g *Generator) mapZeroRef(mt *checker.MapType) string {
	g.registerMapTypes(mt)
	return "&_vira_mapzero_" + g.mapOpsTag(mt)
}

// emitMapOpsDefinitions writes one ops table (and one zero value) per value type.
func (g *Generator) emitMapOpsDefinitions() string {
	if len(g.mapOps) == 0 {
		return ""
	}
	var tags []string
	for tag := range g.mapOps {
		tags = append(tags, tag)
	}
	sort.Strings(tags)
	var out strings.Builder
	out.WriteString("// --- 1c. MAP VALUE OPS ---\n")
	for _, tag := range tags {
		mt := g.mapOps[tag]
		cv := g.cType(mt.Value)
		cloneFn, freeFn := "NULL", "NULL"
		if g.typeOwnsHeap(mt.Value) {
			cloneFn = "_vira_mapclone_" + tag
			freeFn = "_vira_mapfree_" + tag
			out.WriteString(fmt.Sprintf("static void %s(void* dst, const void* src) {\n", cloneFn))
			out.WriteString(fmt.Sprintf("    *(%s*)dst = %s;\n", cv, g.emitCloneExpr(mt.Value, fmt.Sprintf("(*(%s*)src)", cv))))
			out.WriteString("}\n")
			out.WriteString(fmt.Sprintf("static void %s(void* val) {\n", freeFn))
			g.emitFreeLValueStmt(&out, "    ", mt.Value, fmt.Sprintf("(*(%s*)val)", cv))
			out.WriteString("}\n")
		}
		out.WriteString(fmt.Sprintf("static const vira_map_ops _vira_mapops_%s = { sizeof(%s), %t, %s, %s };\n",
			tag, cv, mapKeyIsString(mt), cloneFn, freeFn))
		out.WriteString(fmt.Sprintf("static %s _vira_mapzero_%s;\n\n", cv, tag))
	}
	return out.String()
}

// mapAddr emits a `vira_map*` for a map place expression.
func (g *Generator) mapAddr(e ast.Expr) string {
	if ident, ok := e.(*ast.IdentExpr); ok && g.isPointerReceiverVar(ident.Ident.Name) {
		return ident.Ident.Name
	}
	return "&(" + g.emitLValue(e) + ")"
}

// mapKeyArgs turns an emitted key into the runtime's (ikey, skey) argument pair.
func mapKeyArgs(mt *checker.MapType, key string) string {
	if mapKeyIsString(mt) {
		return fmt.Sprintf("0, %s", key)
	}
	return fmt.Sprintf("(int64_t)(%s), (vira_string){0}", key)
}

// mapKey emits a key expression. A fresh string key (`m['a' + b]`) is bound to a
// temporary that the caller frees with the returned cleanup statement.
func (g *Generator) mapKey(mt *checker.MapType, key ast.Expr) (bind string, use string, cleanup string) {
	k := g.emitExpr(key)
	if ident, ok := key.(*ast.IdentExpr); ok {
		k = g.identValueExpr(ident)
	}
	if mapKeyIsString(mt) && isFreshOwningRValue(key) {
		tmp := g.nextTmp("mk")
		return fmt.Sprintf("vira_string %s = %s;", tmp, k), tmp, fmt.Sprintf("vira_str_free(&%s);", tmp)
	}
	return "", k, ""
}

// emitMapRead emits `m[k]`: the stored value, or V's zero value for a missing key.
func (g *Generator) emitMapRead(e *ast.IndexExpr, mt *checker.MapType) string {
	cv := g.cType(mt.Value)
	bind, key, cleanup := g.mapKey(mt, e.Index)
	get := fmt.Sprintf("(*(%s*)vira_map_get_or(%s, %s, %s))", cv, g.mapAddr(e.Left), mapKeyArgs(mt, key), g.mapZeroRef(mt))
	if bind == "" {
		return get
	}
	tmp := g.nextTmp("mv")
	return fmt.Sprintf("({ %s %s %s = %s; %s %s; })", bind, cv, tmp, get, cleanup, tmp)
}

// emitMapStoreStmts emits statements that store value under key in the map at addr. The new
// value is computed first (it may read the map), then the slot is found or created and its
// previous value released.
func (g *Generator) emitMapStoreStmts(addr string, mt *checker.MapType, key ast.Expr, value ast.Expr, moved bool, indent string) string {
	cv := g.cType(mt.Value)
	newTmp := g.nextTmp("nv")
	slot := g.nextTmp("slot")
	var b strings.Builder
	b.WriteString(fmt.Sprintf("%s{ %s %s = %s;", indent, cv, newTmp, g.emitStoredValue(value, mt.Value, moved, true)))
	bind, k, cleanup := g.mapKey(mt, key)
	if bind != "" {
		b.WriteString(" " + bind)
	}
	b.WriteString(fmt.Sprintf(" bool _had_%s = vira_map_has(%s, %s);", slot, addr, mapKeyArgs(mt, k)))
	b.WriteString(fmt.Sprintf(" %s* %s = (%s*)vira_map_slot(%s, %s, %s);", cv, slot, cv, addr, g.mapOpsRef(mt), mapKeyArgs(mt, k)))
	if g.typeOwnsHeap(mt.Value) {
		var free strings.Builder
		g.emitFreeLValueStmt(&free, "", mt.Value, "(*"+slot+")")
		b.WriteString(fmt.Sprintf(" if (_had_%s) { %s }", slot, strings.TrimSpace(free.String())))
	} else {
		b.WriteString(fmt.Sprintf(" (void)_had_%s;", slot))
	}
	b.WriteString(fmt.Sprintf(" *%s = %s;", slot, newTmp))
	if cleanup != "" {
		b.WriteString(" " + cleanup)
	}
	b.WriteString(" }\n")
	return b.String()
}

// emitMapAssign lowers `m[k] = v` and `m[k] op= v`.
func (g *Generator) emitMapAssign(out *strings.Builder, s *ast.AssignStmt, idx *ast.IndexExpr, mt *checker.MapType, indent string, fnRet checker.Type) {
	value := s.Value
	moved := s.MoveValue
	if call, ok := value.(*ast.CallExpr); ok && call.OrBlock != nil {
		// Unwrap `f() or { ... }` into an owned temporary, then move it into the map.
		valType := g.chk.ExprTypes[value]
		tmp := g.nextTmp("or_val")
		out.WriteString(fmt.Sprintf("%s%s %s;\n", indent, g.cType(valType), tmp))
		out.WriteString(fmt.Sprintf("%smemset(&%s, 0, sizeof(%s));\n", indent, tmp, tmp))
		g.emitUnwrappedAssign(out, tmp, valType, call, indent, fnRet)
		tmpIdent := &ast.IdentExpr{Ident: ast.Ident{Name: tmp, NamePos: call.Span()}}
		g.chk.ExprTypes[tmpIdent] = valType
		value = tmpIdent
		moved = true
	}
	addr := g.mapAddr(idx.Left)
	if s.Op == token.TokAssign {
		out.WriteString(g.emitMapStoreStmts(addr, mt, idx.Index, value, moved, indent))
		return
	}

	// Compound assignment: read-modify-write through the slot (a missing key starts at zero).
	cv := g.cType(mt.Value)
	slot := g.nextTmp("slot")
	bind, k, cleanup := g.mapKey(mt, idx.Index)
	out.WriteString(fmt.Sprintf("%s{\n", indent))
	if bind != "" {
		out.WriteString(fmt.Sprintf("%s    %s\n", indent, bind))
	}
	rhsType := g.chk.ExprTypes[value]
	if s.Op == token.TokPlusAssign && checker.ExactEquals(mt.Value, checker.TypeString) {
		oldTmp := g.nextTmp("old_val")
		out.WriteString(fmt.Sprintf("%s    vira_string %s = %s;\n", indent, oldTmp, g.emitMapRead(idx, mt)))
		prevArena := g.activeArena
		g.activeArena = ""
		newVal := g.emitStringConcatWith(oldTmp, value)
		g.activeArena = prevArena
		out.WriteString(fmt.Sprintf("%s    vira_string _nv_%s = %s;\n", indent, slot, newVal))
		out.WriteString(fmt.Sprintf("%s    %s* %s = (%s*)vira_map_slot(%s, %s, %s);\n", indent, cv, slot, cv, addr, g.mapOpsRef(mt), mapKeyArgs(mt, k)))
		out.WriteString(fmt.Sprintf("%s    vira_str_free(%s);\n", indent, slot))
		out.WriteString(fmt.Sprintf("%s    *%s = _nv_%s;\n", indent, slot, slot))
	} else {
		rhs := g.emitExpr(value)
		out.WriteString(fmt.Sprintf("%s    %s* %s = (%s*)vira_map_slot(%s, %s, %s);\n", indent, cv, slot, cv, addr, g.mapOpsRef(mt), mapKeyArgs(mt, k)))
		target := "(*" + slot + ")"
		if s.Op == token.TokDivAssign || s.Op == token.TokModAssign {
			binOp := token.TokSlash
			if s.Op == token.TokModAssign {
				binOp = token.TokPercent
			}
			if helper := g.checkedIntDivHelper(binOp, mt.Value, rhsType, value); helper != "" {
				out.WriteString(fmt.Sprintf("%s    %s = %s(%s, %s);\n", indent, target, helper, target, rhs))
				goto done
			}
		}
		out.WriteString(fmt.Sprintf("%s    %s %s %s;\n", indent, target, s.Op.String(), rhs))
	}
done:
	if cleanup != "" {
		out.WriteString(fmt.Sprintf("%s    %s\n", indent, cleanup))
	}
	out.WriteString(fmt.Sprintf("%s}\n", indent))
}

// emitMapForIn lowers `for k in m` and `for k, v in m` (insertion order). The loop
// variables are borrowed views of the map's own key and value.
func (g *Generator) emitMapForIn(out *strings.Builder, s *ast.ForInStmt, mt *checker.MapType, indent string, fnRet checker.Type) {
	it := g.nextTmp("mit")
	idx := g.nextTmp("mi")
	owned := false
	out.WriteString(fmt.Sprintf("%s{\n", indent))
	switch s.Iter.(type) {
	case *ast.CallExpr, *ast.MapLitExpr:
		owned = true
		out.WriteString(fmt.Sprintf("%s    vira_map %s_val = %s;\n", indent, it, g.emitExpr(s.Iter)))
		out.WriteString(fmt.Sprintf("%s    vira_map* %s = &%s_val;\n", indent, it, it))
	default:
		out.WriteString(fmt.Sprintf("%s    vira_map* %s = %s;\n", indent, it, g.mapAddr(s.Iter)))
	}
	out.WriteString(fmt.Sprintf("%s    for (int32_t %s = vira_map_next(%s, 0); %s >= 0; %s = vira_map_next(%s, %s + 1)) {\n",
		indent, idx, it, idx, idx, it, idx))
	keyExpr := fmt.Sprintf("(%s)vira_map_entry_ikey(%s, %s)", g.cType(mt.Key), it, idx)
	if mapKeyIsString(mt) {
		keyExpr = fmt.Sprintf("vira_map_entry_skey(%s, %s)", it, idx)
	}
	out.WriteString(fmt.Sprintf("%s        %s %s = %s;\n", indent, g.cType(mt.Key), s.Var.Name, keyExpr))
	if s.ValueVar != nil {
		cv := g.cType(mt.Value)
		out.WriteString(fmt.Sprintf("%s        %s %s = *(%s*)vira_map_entry_val(%s, %s);\n", indent, cv, s.ValueVar.Name, cv, it, idx))
	}
	g.emitBlockStmts(out, s.Body, indent+"        ", fnRet)
	out.WriteString(fmt.Sprintf("%s    }\n", indent))
	if owned {
		out.WriteString(fmt.Sprintf("%s    vira_map_free(%s);\n", indent, it))
	}
	out.WriteString(fmt.Sprintf("%s}\n", indent))
}

// emitMapMethod lowers m.has(k), m.delete(k), m.keys(), m.values() and m.clear().
func (g *Generator) emitMapMethod(call *ast.CallExpr, callee *ast.SelectorExpr, mt *checker.MapType) string {
	addr := g.mapAddr(callee.Left)
	switch callee.Field.Name {
	case "has", "delete":
		fn := "vira_map_has"
		if callee.Field.Name == "delete" {
			fn = "vira_map_delete"
		}
		bind, k, cleanup := g.mapKey(mt, call.Args[0].Expr)
		if bind == "" {
			return fmt.Sprintf("%s(%s, %s)", fn, addr, mapKeyArgs(mt, k))
		}
		return fmt.Sprintf("({ %s bool _r = %s(%s, %s); %s _r; })", bind, fn, addr, mapKeyArgs(mt, k), cleanup)
	case "clear":
		return fmt.Sprintf("vira_map_clear(%s)", addr)
	case "keys", "values":
		elem := mt.Key
		if callee.Field.Name == "values" {
			elem = mt.Value
		}
		sl := &checker.SliceType{Elem: elem}
		g.registerSliceType(sl)
		cSl := g.cSliceName(elem)
		ce := g.cType(elem)
		m := g.nextTmp("mk")
		res := g.nextTmp("ks")
		i := g.nextTmp("ki")
		var item string
		if callee.Field.Name == "values" {
			item = g.emitCloneExpr(elem, fmt.Sprintf("(*(%s*)vira_map_entry_val(%s, %s))", ce, m, i))
		} else if mapKeyIsString(mt) {
			item = fmt.Sprintf("vira_str_clone(vira_map_entry_skey(%s, %s))", m, i)
		} else {
			item = fmt.Sprintf("(%s)vira_map_entry_ikey(%s, %s)", ce, m, i)
		}
		return fmt.Sprintf("({ const vira_map* %s = %s; %s %s = %s_new(%s->len); "+
			"for (int32_t %s = vira_map_next(%s, 0); %s >= 0; %s = vira_map_next(%s, %s + 1)) { %s_push(&%s, %s); } %s; })",
			m, addr, cSl, res, cSl, m, i, m, i, i, m, i, cSl, res, item, res)
	}
	return "/* unknown map method */0"
}
