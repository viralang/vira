package checker

// Generics by monomorphisation.
//
// A generic function (`fn first[T](xs []T) T`) or struct (`struct Box[T] { v T }`,
// with its methods `fn (b Box[T]) get() T` and static methods `fn Box[T].new(v T) Box[T]`)
// is kept as a template and never checked on its own. Each distinct use creates an
// instance: a copy of the template's AST with every type parameter replaced by the
// concrete type, under a mangled name (`first__int`, `Box__string`). Instances are
// checked like ordinary code, appended to the file the template came from, and the
// templates themselves are removed, so autofree and codegen only ever see concrete
// declarations.
//
// Type arguments of a function are inferred from the call's arguments, or given
// explicitly: `first[int](xs)`. Struct type arguments are always written:
// `Box[int]{ v: 1 }`, `Box[int].new(1)`, a field `b Box[string]`.

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/token"
)

type genericTemplate struct {
	decl    ast.Decl // *ast.StructDecl or *ast.FnDecl
	file    *ast.File
	module  string
	params  []string
	methods []*ast.FnDecl // struct templates: receiver and static methods
}

// InstanceInfo records which template and arguments a struct instance came from.
type InstanceInfo struct {
	Template string
	Args     []Type
}

type pendingBody struct {
	decl   *ast.FnDecl
	file   *ast.File
	module string
	ctx    string
}

type genericsState struct {
	genericStructs  map[string]*genericTemplate // "mod.Name"
	genericFns      map[string]*genericTemplate // "mod.name"
	templates       map[ast.Decl]bool
	structInstances map[string]*StructType
	fnInstances     map[string]*FnSig
	InstanceArgs    map[*StructType]*InstanceInfo
	pending         []pendingBody
	preChecked      map[ast.Expr]Type
	instCtx         string
}

func (g *genericsState) init() {
	g.genericStructs = make(map[string]*genericTemplate)
	g.genericFns = make(map[string]*genericTemplate)
	g.templates = make(map[ast.Decl]bool)
	g.structInstances = make(map[string]*StructType)
	g.fnInstances = make(map[string]*FnSig)
	g.InstanceArgs = make(map[*StructType]*InstanceInfo)
	g.preChecked = make(map[ast.Expr]Type)
}

// IsGenericTemplate reports a declaration that is only a template (never emitted).
func (c *Checker) IsGenericTemplate(d ast.Decl) bool {
	return c.generics.templates[d]
}

func identNames(ids []ast.Ident) []string {
	var out []string
	for _, id := range ids {
		out = append(out, id.Name)
	}
	return out
}

// registerGenericStruct records a struct template (pass 1A).
func (c *Checker) registerGenericStruct(d *ast.StructDecl, file *ast.File, mod string) {
	c.getOrCreateModule(mod)
	key := mod + "." + d.Name.Name
	c.generics.genericStructs[key] = &genericTemplate{decl: d, file: file, module: mod, params: identNames(d.TypeParams)}
	c.generics.templates[d] = true
}

// collectGenericTemplates records generic functions and the methods of generic structs.
// It runs after every struct (and struct template) name is known.
func (c *Checker) collectGenericTemplates(files []*ast.File) {
	for _, file := range files {
		fileMod := file.Module.Name
		if fileMod == "" {
			fileMod = "main"
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FnDecl)
			if !ok {
				continue
			}
			mod := fn.Module
			if mod == "" {
				mod = fileMod
			}
			if fn.Receiver != nil {
				if it, ok := fn.Receiver.Type.(*ast.IdentType); ok {
					if tpl := c.generics.genericStructs[mod+"."+it.Name.Name]; tpl != nil {
						tpl.methods = append(tpl.methods, fn)
						c.generics.templates[fn] = true
					}
				}
				continue
			}
			if fn.StaticType != nil {
				if tpl := c.generics.genericStructs[mod+"."+fn.StaticType.Name]; tpl != nil {
					tpl.methods = append(tpl.methods, fn)
					c.generics.templates[fn] = true
				}
				continue
			}
			if len(fn.TypeParams) > 0 {
				c.getOrCreateModule(mod)
				c.generics.genericFns[mod+"."+fn.Name.Name] = &genericTemplate{decl: fn, file: file, module: mod, params: identNames(fn.TypeParams)}
				c.generics.templates[fn] = true
			}
		}
	}
}

func (c *Checker) lookupGenericStruct(mod string, name string) *genericTemplate {
	if tpl := c.generics.genericStructs[mod+"."+name]; tpl != nil {
		return tpl
	}
	return nil
}

func (c *Checker) lookupGenericFn(mod string, name string) *genericTemplate {
	return c.generics.genericFns[mod+"."+name]
}

// mangleType gives a C-identifier-safe spelling of a type for instance names.
func mangleType(t Type) string {
	switch v := t.(type) {
	case *PrimitiveType:
		return string(v.Kind)
	case *SliceType:
		return "sl_" + mangleType(v.Elem)
	case *ArrayType:
		return fmt.Sprintf("arr%d_%s", v.Len, mangleType(v.Elem))
	case *MapType:
		return "map_" + mangleType(v.Key) + "_" + mangleType(v.Value)
	case *FnType:
		var ps []string
		for _, p := range v.Params {
			ps = append(ps, mangleType(p))
		}
		return "fn_" + strings.Join(ps, "_") + "_r_" + mangleType(v.Return)
	case *StructType:
		if v.Module != "" && v.Module != "main" {
			return v.Module + "_" + v.Name
		}
		return v.Name
	case *EnumType:
		if v.Module != "" && v.Module != "main" {
			return v.Module + "_" + v.Name
		}
		return v.Name
	case *InterfaceType:
		return v.Name
	case *OptionType:
		return "opt_" + mangleType(v.Inner)
	case *ResultType:
		return "res_" + mangleType(v.Inner)
	case *RefType:
		return "ref_" + mangleType(v.Inner)
	case *TensorType:
		return "tensor_" + mangleType(v.DType)
	}
	var b strings.Builder
	for _, r := range t.String() {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	return b.String()
}

func instanceName(base string, args []Type) string {
	parts := []string{base}
	for _, a := range args {
		parts = append(parts, mangleType(a))
	}
	return strings.Join(parts, "__")
}

func displayInstance(base string, args []Type) string {
	var as []string
	for _, a := range args {
		as = append(as, a.String())
	}
	return base + "[" + strings.Join(as, ", ") + "]"
}

// cloneWithSubst deep-copies an AST node, replacing type names found in subst.
func cloneWithSubst(node any, subst map[string]Type) any {
	return cloneValue(reflect.ValueOf(node), subst).Interface()
}

var (
	identTypePtr    = reflect.TypeOf(&ast.IdentType{})
	resolvedTypePtr = reflect.TypeOf(&ast.ResolvedType{})
)

func cloneValue(v reflect.Value, subst map[string]Type) reflect.Value {
	switch v.Kind() {
	case reflect.Ptr:
		if v.IsNil() {
			return v
		}
		if v.Type() == resolvedTypePtr {
			return v // carries a checker.Type: shared, never copied
		}
		n := reflect.New(v.Type().Elem())
		n.Elem().Set(cloneValue(v.Elem(), subst))
		return n
	case reflect.Interface:
		if v.IsNil() {
			return v
		}
		inner := v.Elem()
		if inner.Type() == identTypePtr {
			it := inner.Interface().(*ast.IdentType)
			if t, ok := subst[it.Name.Name]; ok && len(it.TypeParams) == 0 {
				out := reflect.New(v.Type()).Elem()
				out.Set(reflect.ValueOf(&ast.ResolvedType{Type: t, Pos: it.Name.NamePos}))
				return out
			}
		}
		out := reflect.New(v.Type()).Elem()
		out.Set(cloneValue(inner, subst))
		return out
	case reflect.Struct:
		n := reflect.New(v.Type()).Elem()
		for i := 0; i < v.NumField(); i++ {
			if n.Field(i).CanSet() {
				n.Field(i).Set(cloneValue(v.Field(i), subst))
			}
		}
		return n
	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		n := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		for i := 0; i < v.Len(); i++ {
			n.Index(i).Set(cloneValue(v.Index(i), subst))
		}
		return n
	default:
		return v
	}
}

func substMap(params []string, args []Type) map[string]Type {
	m := make(map[string]Type, len(params))
	for i, p := range params {
		if i < len(args) {
			m[p] = args[i]
		}
	}
	return m
}

// withTemplateContext runs fn with the checker positioned in a template's module and file.
func (c *Checker) withTemplateContext(tpl *genericTemplate, ctx string, fn func()) {
	savedMod, savedFile := c.currentMod, c.curFile
	savedImports := c.currentImports
	savedRet, savedActor, savedCtx := c.curFnReturn, c.curActor, c.generics.instCtx
	c.currentMod, c.curFile = tpl.module, tpl.file
	c.SetFileImports(tpl.file.Imports)
	c.generics.instCtx = ctx
	fn()
	c.currentMod, c.curFile = savedMod, savedFile
	c.currentImports = savedImports
	c.curFnReturn, c.curActor, c.generics.instCtx = savedRet, savedActor, savedCtx
}

func (c *Checker) addSynthesized(file *ast.File, d ast.Decl) {
	c.synthesized = append(c.synthesized, d)
	c.synthByFile[file] = append(c.synthByFile[file], d)
}

// instantiateStruct returns the StructType for Template[args...], creating it (with its
// methods) on first use.
func (c *Checker) instantiateStruct(tpl *genericTemplate, args []Type, span token.Span) Type {
	sd := tpl.decl.(*ast.StructDecl)
	if len(args) != len(tpl.params) {
		c.errorf(span, "generic struct '%s' expects %d type argument(s), got %d", sd.Name.Name, len(tpl.params), len(args))
		return TypeVoid
	}
	name := instanceName(sd.Name.Name, args)
	key := tpl.module + "." + name
	if st := c.generics.structInstances[key]; st != nil {
		return st
	}
	ctx := displayInstance(sd.Name.Name, args)
	subst := substMap(tpl.params, args)
	clone := cloneWithSubst(sd, subst).(*ast.StructDecl)
	clone.Name.Name = name
	clone.TypeParams = nil
	clone.Module = tpl.module

	st := &StructType{
		Module:          tpl.module,
		Name:            name,
		IsPub:           sd.IsPub,
		IsActor:         sd.IsActor,
		Fields:          make(map[string]*FieldInfo),
		Methods:         make(map[string]*FnSig),
		StaticMethods:   make(map[string]*FnSig),
		PromotedFields:  make(map[string]PromotedField),
		PromotedMethods: make(map[string]PromotedMethod),
		Span:            sd.DeclSpan,
	}
	scope := c.getOrCreateModule(tpl.module)
	scope.Structs[name] = st
	c.structs[tpl.module+"."+name] = st
	if _, exists := c.structs[name]; !exists || tpl.module == "main" {
		c.structs[name] = st
	}
	c.generics.structInstances[key] = st
	c.generics.InstanceArgs[st] = &InstanceInfo{Template: sd.Name.Name, Args: args}
	c.addSynthesized(tpl.file, clone)

	c.withTemplateContext(tpl, ctx, func() {
		c.registerStructMembers(clone)
		c.promoteEmbeddedMembers(st)
		for _, m := range tpl.methods {
			// The method names the struct's parameters itself: `fn (b Box[T])` or `fn Box[T].new`.
			var mParams []string
			if m.Receiver != nil {
				if it, ok := m.Receiver.Type.(*ast.IdentType); ok {
					for _, tp := range it.TypeParams {
						if id, ok := tp.(*ast.IdentType); ok {
							mParams = append(mParams, id.Name.Name)
						}
					}
				}
			} else {
				mParams = identNames(m.TypeParams)
			}
			if len(mParams) != len(args) {
				c.errorf(m.DeclSpan, "method '%s' of generic struct '%s' must name its %d type parameter(s), e.g. 'fn (x %s[%s]) ...'",
					m.Name.Name, sd.Name.Name, len(args), sd.Name.Name, strings.Join(tpl.params, ", "))
				continue
			}
			mc := cloneWithSubst(m, substMap(mParams, args)).(*ast.FnDecl)
			mc.TypeParams = nil
			mc.Module = tpl.module
			if mc.Receiver != nil {
				mc.Receiver.Type = &ast.IdentType{Name: ast.Ident{Name: name, NamePos: m.Receiver.RecvSpan}}
			} else {
				mc.StaticType = &ast.Ident{Name: name, NamePos: m.StaticType.NamePos}
			}
			c.registerFnHeader(mc)
			c.addSynthesized(tpl.file, mc)
			c.generics.pending = append(c.generics.pending, pendingBody{decl: mc, file: tpl.file, module: tpl.module, ctx: ctx})
		}
	})
	return st
}

// instantiateFn returns the signature of a generic function instance, creating it on first use.
func (c *Checker) instantiateFn(tpl *genericTemplate, args []Type) *FnSig {
	fd := tpl.decl.(*ast.FnDecl)
	name := instanceName(fd.Name.Name, args)
	key := tpl.module + "." + name
	if sig := c.generics.fnInstances[key]; sig != nil {
		return sig
	}
	ctx := displayInstance(fd.Name.Name, args)
	clone := cloneWithSubst(fd, substMap(tpl.params, args)).(*ast.FnDecl)
	clone.Name.Name = name
	clone.TypeParams = nil
	clone.Module = tpl.module
	var sig *FnSig
	c.withTemplateContext(tpl, ctx, func() {
		c.registerFnHeader(clone)
		sig = c.getOrCreateModule(tpl.module).Functions[name]
	})
	c.generics.fnInstances[key] = sig
	c.addSynthesized(tpl.file, clone)
	c.generics.pending = append(c.generics.pending, pendingBody{decl: clone, file: tpl.file, module: tpl.module, ctx: ctx})
	return sig
}

// checkPendingInstances checks the bodies of instances; checking one may create more.
func (c *Checker) checkPendingInstances() {
	for len(c.generics.pending) > 0 {
		p := c.generics.pending[0]
		c.generics.pending = c.generics.pending[1:]
		tpl := &genericTemplate{file: p.file, module: p.module}
		c.withTemplateContext(tpl, p.ctx, func() {
			c.checkFnBody(p.decl)
		})
	}
}

// unify matches a template parameter type against an argument type, binding type
// parameters. It reports a conflicting binding as an error string.
func (c *Checker) unify(te ast.TypeExpr, t Type, params map[string]bool, bind map[string]Type) string {
	if te == nil || t == nil {
		return ""
	}
	switch x := te.(type) {
	case *ast.IdentType:
		if params[x.Name.Name] && len(x.TypeParams) == 0 {
			if prev, ok := bind[x.Name.Name]; ok {
				if !ExactEquals(prev, t) {
					return fmt.Sprintf("type parameter %s is both '%s' and '%s'", x.Name.Name, prev, t)
				}
				return ""
			}
			bind[x.Name.Name] = t
			return ""
		}
		if len(x.TypeParams) > 0 {
			if st, ok := t.(*StructType); ok {
				if info := c.generics.InstanceArgs[st]; info != nil && info.Template == x.Name.Name && len(info.Args) == len(x.TypeParams) {
					for i := range x.TypeParams {
						if msg := c.unify(x.TypeParams[i], info.Args[i], params, bind); msg != "" {
							return msg
						}
					}
				}
			}
		}
	case *ast.ArrayType:
		if x.Len == nil {
			if sl, ok := t.(*SliceType); ok {
				return c.unify(x.Elem, sl.Elem, params, bind)
			}
		} else if arr, ok := t.(*ArrayType); ok {
			return c.unify(x.Elem, arr.Elem, params, bind)
		}
	case *ast.MapType:
		if mt, ok := t.(*MapType); ok {
			if msg := c.unify(x.Key, mt.Key, params, bind); msg != "" {
				return msg
			}
			return c.unify(x.Value, mt.Value, params, bind)
		}
	case *ast.FnType:
		if ft, ok := t.(*FnType); ok && len(ft.Params) == len(x.Params) {
			for i := range x.Params {
				if msg := c.unify(x.Params[i], ft.Params[i], params, bind); msg != "" {
					return msg
				}
			}
			if x.Return != nil {
				return c.unify(x.Return, ft.Return, params, bind)
			}
		}
	case *ast.OptionType:
		if ot, ok := t.(*OptionType); ok {
			return c.unify(x.Inner, ot.Inner, params, bind)
		}
	case *ast.ResultType:
		if rt, ok := t.(*ResultType); ok && x.Inner != nil {
			return c.unify(x.Inner, rt.Inner, params, bind)
		}
	case *ast.RefType:
		if rt, ok := t.(*RefType); ok {
			return c.unify(x.Inner, rt.Inner, params, bind)
		}
	}
	return ""
}

// inferCall instantiates a generic function for a call, inferring its type arguments from
// the argument types (explicit arguments, when given, come first).
func (c *Checker) inferCall(tpl *genericTemplate, call *ast.CallExpr, scope *Scope, explicit []Type, displayName string) *FnSig {
	fd := tpl.decl.(*ast.FnDecl)
	if len(explicit) > len(tpl.params) {
		c.errorf(call.CallSpan, "'%s' has %d type parameter(s), got %d type argument(s)", displayName, len(tpl.params), len(explicit))
		return nil
	}
	params := map[string]bool{}
	for _, p := range tpl.params {
		params[p] = true
	}
	bind := map[string]Type{}
	for i, t := range explicit {
		bind[tpl.params[i]] = t
	}
	if len(call.Args) != len(fd.Params) {
		c.errorf(call.CallSpan, "call to '%s' expects %d arguments, got %d", displayName, len(fd.Params), len(call.Args))
		return nil
	}
	{
		for i, arg := range call.Args {
			at := c.checkExpr(arg.Expr, scope)
			c.generics.preChecked[arg.Expr] = at
			if msg := c.unify(fd.Params[i].Type, at, params, bind); msg != "" {
				c.errorf(arg.ArgSpan, "cannot infer the type arguments of '%s': %s", displayName, msg)
				return nil
			}
		}
	}
	var args []Type
	for _, p := range tpl.params {
		t, ok := bind[p]
		if !ok {
			c.errorf(call.CallSpan, "cannot infer type parameter '%s' of '%s'; give it explicitly, e.g. %s[int](...)", p, displayName, displayName)
			return nil
		}
		args = append(args, t)
	}
	return c.instantiateFn(tpl, args)
}

// typeArgsFromIndex reads explicit type arguments written as `f[int]` (an index expression).
func (c *Checker) typeArgsFromIndex(idx ast.Expr) []Type {
	if id, ok := idx.(*ast.IdentExpr); ok {
		return []Type{c.resolveTypeExpr(&ast.IdentType{Name: id.Ident})}
	}
	c.errorf(idx.Span(), "type argument must be a type name; for other types let the compiler infer it from the arguments")
	return nil
}

// rewriteGenericCall resolves generic callees before ordinary call checking: explicit
// `f[int](x)`, inferred `f(x)` / `pkg.f(x)`, and static methods of generic structs
// `Box[int].new(x)`. The callee is rewritten to the instance's name.
func (c *Checker) rewriteGenericCall(call *ast.CallExpr, scope *Scope) bool {
	nErr := len(c.errors)
	c.rewriteGenericCallInner(call, scope)
	return len(c.errors) == nErr
}

func (c *Checker) rewriteGenericCallInner(call *ast.CallExpr, scope *Scope) {
	switch callee := call.Callee.(type) {
	case *ast.IndexExpr:
		id, ok := callee.Left.(*ast.IdentExpr)
		if !ok {
			return
		}
		if _, inScope := scope.Lookup(id.Ident.Name); inScope {
			return
		}
		tpl := c.lookupGenericFn(c.currentMod, id.Ident.Name)
		if tpl == nil {
			return
		}
		explicit := c.typeArgsFromIndex(callee.Index)
		if sig := c.inferCall(tpl, call, scope, explicit, id.Ident.Name); sig != nil {
			call.Callee = &ast.IdentExpr{Ident: ast.Ident{Name: sig.Name, NamePos: id.Ident.NamePos}}
		}
	case *ast.IdentExpr:
		if _, inScope := scope.Lookup(callee.Ident.Name); inScope {
			return
		}
		if tpl := c.lookupGenericFn(c.currentMod, callee.Ident.Name); tpl != nil {
			if sig := c.inferCall(tpl, call, scope, nil, callee.Ident.Name); sig != nil {
				callee.Ident.Name = sig.Name
			}
		}
	case *ast.SelectorExpr:
		if inner, ok := callee.Left.(*ast.SelectorExpr); ok && len(inner.TypeArgs) > 0 {
			// pkg.Box[int].new(...)
			if pkg, ok := inner.Left.(*ast.IdentExpr); ok {
				t := c.resolveTypeExpr(&ast.SelectorType{Pkg: pkg.Ident, Name: inner.Field, TypeParams: inner.TypeArgs})
				if st, ok := t.(*StructType); ok {
					inner.Field.Name = st.Name
				}
			}
			inner.TypeArgs = nil
			return
		}
		if id, ok := callee.Left.(*ast.IdentExpr); ok {
			if _, inScope := scope.Lookup(id.Ident.Name); inScope {
				return
			}
			if len(id.TypeArgs) > 0 {
				// Box[int].new(...)
				t := c.resolveTypeExpr(&ast.IdentType{Name: id.Ident, TypeParams: id.TypeArgs})
				if st, ok := t.(*StructType); ok {
					id.Ident.Name = st.Name
				}
				id.TypeArgs = nil
				return
			}
			target := id.Ident.Name
			if resolved, ok := c.currentImports[target]; ok {
				target = resolved
			}
			if tpl := c.lookupGenericFn(target, callee.Field.Name); tpl != nil {
				if sig := c.inferCall(tpl, call, scope, nil, id.Ident.Name+"."+callee.Field.Name); sig != nil {
					if tpl.module != c.currentMod && !sig.IsPub {
						c.errorf(callee.SelSpan, "function '%s.%s' is private to module '%s'", id.Ident.Name, callee.Field.Name, target)
					}
					callee.Field.Name = sig.Name
				}
			}
		}
	}
}
