package cgen

// C lowering of closures (`fn (A) R` values).
//
// The checker lifts every anonymous function into a top-level function whose
// leading parameters are the variables it captured. Here each lifted function
// gets an environment struct (the runtime header plus one field per capture), a
// drop function that releases the captured values, and a trampoline with the
// uniform closure signature `R tramp(vira_closure_env*, A...)` that unpacks the
// environment and calls the lifted function. Named functions used as values get
// a thunk with the same signature and no environment.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/checker"
)

func liftedCName(module string, name string) string {
	if module == "" {
		module = "main"
	}
	return fmt.Sprintf("%s__%s", module, name)
}

func closureEnvName(info *checker.ClosureInfo) string {
	return liftedCName(info.Module, info.Lit.LiftedName) + "_env"
}

func closureTrampName(info *checker.ClosureInfo) string {
	return liftedCName(info.Module, info.Lit.LiftedName) + "_tramp"
}

func fnSigCName(sig *checker.FnSig) string {
	if sig.IsExport {
		if sig.ExportName != "" {
			return sig.ExportName
		}
		return sig.Name
	}
	return liftedCName(sig.Module, sig.Name)
}

func fnRefThunkName(sig *checker.FnSig) string {
	return "_fnref_" + fnSigCName(sig)
}

// closureFnPtrType is the C function-pointer type a closure of type ft is called through.
func (g *Generator) closureFnPtrType(ft *checker.FnType) string {
	params := []string{"vira_closure_env*"}
	for _, p := range ft.Params {
		params = append(params, g.cType(p))
	}
	return fmt.Sprintf("%s (*)(%s)", g.cType(ft.Return), strings.Join(params, ", "))
}

// emitClosureDefinitions writes environments, drop functions, trampolines and function
// reference thunks. It runs after the function prototypes.
func (g *Generator) emitClosureDefinitions(out *strings.Builder) {
	if len(g.chk.ClosureList) == 0 && len(g.chk.FnRefs) == 0 {
		return
	}
	out.WriteString("// --- 4b. CLOSURE ENVIRONMENTS & TRAMPOLINES ---\n")
	for _, info := range g.chk.ClosureList {
		env := closureEnvName(info)
		caps := info.Sig.Params[:info.NumCaps]
		lits := info.Sig.Params[info.NumCaps:]
		if len(caps) > 0 {
			out.WriteString(fmt.Sprintf("typedef struct {\n    vira_closure_env hdr;\n"))
			for i, c := range caps {
				out.WriteString(fmt.Sprintf("    %s c%d; // %s\n", g.cType(c.Type), i, c.Name))
			}
			out.WriteString(fmt.Sprintf("} %s;\n", env))
			out.WriteString(fmt.Sprintf("static void %s_drop(vira_closure_env* _env) {\n    %s* x = (%s*)_env;\n    (void)x;\n", env, env, env))
			for i, c := range caps {
				if g.typeOwnsHeap(c.Type) {
					g.emitFreeLValueStmt(out, "    ", c.Type, fmt.Sprintf("x->c%d", i))
				}
			}
			out.WriteString("}\n")
		}
		params := []string{"vira_closure_env* _env"}
		var args []string
		for i := range caps {
			args = append(args, fmt.Sprintf("((%s*)_env)->c%d", env, i))
		}
		for i, p := range lits {
			params = append(params, fmt.Sprintf("%s a%d", g.cType(p.Type), i))
			args = append(args, fmt.Sprintf("a%d", i))
		}
		ret := g.cType(info.Sig.ReturnType)
		out.WriteString(fmt.Sprintf("static %s %s(%s) {\n    (void)_env;\n", ret, closureTrampName(info), strings.Join(params, ", ")))
		call := fmt.Sprintf("%s(%s)", liftedCName(info.Module, info.Lit.LiftedName), strings.Join(args, ", "))
		if ret == "void" {
			out.WriteString(fmt.Sprintf("    %s;\n}\n\n", call))
		} else {
			out.WriteString(fmt.Sprintf("    return %s;\n}\n\n", call))
		}
	}

	seen := map[string]*checker.FnSig{}
	for _, sig := range g.chk.FnRefs {
		seen[fnRefThunkName(sig)] = sig
	}
	var names []string
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		sig := seen[n]
		params := []string{"vira_closure_env* _env"}
		var args []string
		for i, p := range sig.Params {
			params = append(params, fmt.Sprintf("%s a%d", g.cType(p.Type), i))
			args = append(args, fmt.Sprintf("a%d", i))
		}
		ret := g.cType(sig.ReturnType)
		out.WriteString(fmt.Sprintf("static %s %s(%s) {\n    (void)_env;\n", ret, n, strings.Join(params, ", ")))
		call := fmt.Sprintf("%s(%s)", fnSigCName(sig), strings.Join(args, ", "))
		if ret == "void" {
			out.WriteString(fmt.Sprintf("    %s;\n}\n\n", call))
		} else {
			out.WriteString(fmt.Sprintf("    return %s;\n}\n\n", call))
		}
	}
}

// emitClosureLit creates a closure value: the captured variables are copied into a fresh
// environment (cloning anything that owns heap memory).
func (g *Generator) emitClosureLit(e *ast.FnLitExpr) string {
	info := g.chk.Closures[e]
	if info == nil {
		return "((vira_closure){0})"
	}
	tramp := closureTrampName(info)
	if info.NumCaps == 0 {
		return fmt.Sprintf("((vira_closure){ .fn = (void*)%s, .env = NULL })", tramp)
	}
	env := closureEnvName(info)
	tmp := g.nextTmp("env")
	var b strings.Builder
	b.WriteString(fmt.Sprintf("({ %s* %s = (%s*)vira_closure_env_new(sizeof(%s), %s_drop); ", env, tmp, env, env, env))
	for i, c := range info.Sig.Params[:info.NumCaps] {
		val := g.identValueExpr(&ast.IdentExpr{Ident: ast.Ident{Name: c.Name}})
		b.WriteString(fmt.Sprintf("%s->c%d = %s; ", tmp, i, g.emitCloneExpr(c.Type, val)))
	}
	b.WriteString(fmt.Sprintf("(vira_closure){ .fn = (void*)%s, .env = &%s->hdr }; })", tramp, tmp))
	return b.String()
}

// emitFnRef turns a named function into a closure value.
func (g *Generator) emitFnRef(sig *checker.FnSig) string {
	return fmt.Sprintf("((vira_closure){ .fn = (void*)%s, .env = NULL })", fnRefThunkName(sig))
}

// emitClosureCall calls through a closure value. Arguments are borrowed, like any call; a
// freshly created callee (`make_adder(1)(2)`) is released after the call.
func (g *Generator) emitClosureCall(call *ast.CallExpr, ft *checker.FnType) string {
	var calleeExpr string
	if ident, ok := call.Callee.(*ast.IdentExpr); ok {
		calleeExpr = g.identValueExpr(ident)
	} else {
		calleeExpr = g.emitExpr(call.Callee)
	}
	c := g.nextTmp("clo")
	args := []string{c + ".env"}
	for _, a := range call.Args {
		if ident, ok := a.Expr.(*ast.IdentExpr); ok {
			args = append(args, g.identValueExpr(ident))
		} else {
			args = append(args, g.emitExpr(a.Expr))
		}
	}
	invoke := fmt.Sprintf("((%s)%s.fn)(%s)", g.closureFnPtrType(ft), c, strings.Join(args, ", "))
	ret := g.cType(ft.Return)
	if !isFreshOwningRValue(call.Callee) {
		return fmt.Sprintf("({ vira_closure %s = %s; if (!%s.fn) _vira_closure_null_call(); %s; })", c, calleeExpr, c, invoke)
	}
	if ret == "void" {
		return fmt.Sprintf("({ vira_closure %s = %s; if (!%s.fn) _vira_closure_null_call(); %s; vira_closure_release(&%s); })",
			c, calleeExpr, c, invoke, c)
	}
	r := g.nextTmp("res")
	return fmt.Sprintf("({ vira_closure %s = %s; if (!%s.fn) _vira_closure_null_call(); %s %s = %s; vira_closure_release(&%s); %s; })",
		c, calleeExpr, c, ret, r, invoke, c, r)
}
