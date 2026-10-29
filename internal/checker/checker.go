package checker

import (
	"fmt"
	"strings"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/token"
)

type SemanticError struct {
	Span token.Span
	Msg  string
}

func (e SemanticError) Error() string {
	return fmt.Sprintf("[%d..%d] semantic error: %s", e.Span.Start, e.Span.End, e.Msg)
}

type VarSymbol struct {
	Name  string
	Type  Type
	IsMut bool
	Span  token.Span
}

type Scope struct {
	parent *Scope
	vars   map[string]*VarSymbol
}

func NewScope(parent *Scope) *Scope {
	return &Scope{
		parent: parent,
		vars:   make(map[string]*VarSymbol),
	}
}

func (s *Scope) Define(sym *VarSymbol) bool {
	if sym.Name == "_" {
		return true
	}
	if _, exists := s.vars[sym.Name]; exists {
		return false
	}
	s.vars[sym.Name] = sym
	return true
}

func (s *Scope) Lookup(name string) (*VarSymbol, bool) {
	for cur := s; cur != nil; cur = cur.parent {
		if sym, ok := cur.vars[name]; ok {
			return sym, true
		}
	}
	return nil, false
}

type ConstSymbol struct {
	Module string
	Name   string
	IsPub  bool
	Type   Type
	Value  ast.Expr
	Span   token.Span
}

type ModuleScope struct {
	Name       string
	Structs    map[string]*StructType
	Interfaces map[string]*InterfaceType
	Functions  map[string]*FnSig
	Consts     map[string]*ConstSymbol
}

func NewModuleScope(name string) *ModuleScope {
	return &ModuleScope{
		Name:       name,
		Structs:    make(map[string]*StructType),
		Interfaces: make(map[string]*InterfaceType),
		Functions:  make(map[string]*FnSig),
		Consts:     make(map[string]*ConstSymbol),
	}
}

type Checker struct {
	currentMod     string
	currentImports map[string]string // alias or pkg name -> target module name
	modules        map[string]*ModuleScope
	structs        map[string]*StructType
	interfaces     map[string]*InterfaceType
	functions      map[string]*FnSig
	consts         map[string]*ConstSymbol
	errors         []SemanticError

	// Semantic side-tables consumed by Phase 3 (autofree) and Phase 4 (cgen)
	ExprTypes      map[ast.Expr]Type
	SelectorPaths  map[*ast.SelectorExpr][]string // records embedded hop path (e.g. ["AuditMeta"])
	ResolvedConsts map[ast.Expr]*ConstSymbol      // records resolved top-level constant references

	// Function context during Pass 3
	curFnReturn    Type
	inDeferDiscard bool
}

func New() *Checker {
	c := &Checker{
		currentImports: make(map[string]string),
		modules:        make(map[string]*ModuleScope),
		structs:        make(map[string]*StructType),
		interfaces:     make(map[string]*InterfaceType),
		functions:      make(map[string]*FnSig),
		consts:         make(map[string]*ConstSymbol),
		ExprTypes:      make(map[ast.Expr]Type),
		SelectorPaths:  make(map[*ast.SelectorExpr][]string),
		ResolvedConsts: make(map[ast.Expr]*ConstSymbol),
	}
	c.registerBuiltins()
	return c
}

func (c *Checker) Errors() []SemanticError {
	return c.errors
}

func (c *Checker) Structs() map[string]*StructType {
	return c.structs
}

func (c *Checker) Interfaces() map[string]*InterfaceType {
	return c.interfaces
}

func (c *Checker) Functions() map[string]*FnSig {
	return c.functions
}

func (c *Checker) Consts() map[string]*ConstSymbol {
	return c.consts
}

func (c *Checker) LookupConstInModule(modName, constName string) *ConstSymbol {
	targetMod := modName
	if resolved, ok := c.currentImports[modName]; ok {
		targetMod = resolved
	}
	if m, ok := c.modules[targetMod]; ok {
		return m.Consts[constName]
	}
	return nil
}

func (c *Checker) ModuleName() string {
	return c.currentMod
}

func (c *Checker) Modules() map[string]*ModuleScope {
	return c.modules
}

func (c *Checker) getOrCreateModule(name string) *ModuleScope {
	if name == "" {
		name = "main"
	}
	m, ok := c.modules[name]
	if !ok {
		m = NewModuleScope(name)
		c.modules[name] = m
	}
	return m
}

func (c *Checker) LookupStructInModule(modName, structName string) *StructType {
	targetMod := modName
	if resolved, ok := c.currentImports[modName]; ok {
		targetMod = resolved
	}
	if m, ok := c.modules[targetMod]; ok {
		return m.Structs[structName]
	}
	return nil
}

func (c *Checker) LookupInterfaceInModule(modName, ifaceName string) *InterfaceType {
	targetMod := modName
	if resolved, ok := c.currentImports[modName]; ok {
		targetMod = resolved
	}
	if m, ok := c.modules[targetMod]; ok {
		return m.Interfaces[ifaceName]
	}
	return nil
}

func (c *Checker) LookupFunctionInModule(modName, fnName string) *FnSig {
	targetMod := modName
	if resolved, ok := c.currentImports[modName]; ok {
		targetMod = resolved
	}
	if m, ok := c.modules[targetMod]; ok {
		return m.Functions[fnName]
	}
	return nil
}

func (c *Checker) LookupStruct(name string) *StructType {
	if strings.Contains(name, ".") {
		parts := strings.SplitN(name, ".", 2)
		targetMod := parts[0]
		if resolved, ok := c.currentImports[parts[0]]; ok {
			targetMod = resolved
		}
		if st := c.LookupStructInModule(targetMod, parts[1]); st != nil {
			return st
		}
	}
	if c.currentMod != "" {
		if st := c.LookupStructInModule(c.currentMod, name); st != nil {
			return st
		}
	}
	return c.structs[name]
}

func (c *Checker) LookupInterface(name string) *InterfaceType {
	if strings.Contains(name, ".") {
		parts := strings.SplitN(name, ".", 2)
		targetMod := parts[0]
		if resolved, ok := c.currentImports[parts[0]]; ok {
			targetMod = resolved
		}
		if iface := c.LookupInterfaceInModule(targetMod, parts[1]); iface != nil {
			return iface
		}
	}
	if c.currentMod != "" {
		if iface := c.LookupInterfaceInModule(c.currentMod, name); iface != nil {
			return iface
		}
	}
	return c.interfaces[name]
}

func (c *Checker) setFileImports(imports []ast.ImportSpec) {
	c.currentImports = make(map[string]string)
	for _, imp := range imports {
		targetMod := imp.Path
		if idx := strings.LastIndex(targetMod, "/"); idx >= 0 {
			targetMod = targetMod[idx+1:]
		}
		targetMod = strings.TrimSuffix(strings.TrimSuffix(targetMod, ".vir"), ".vex")

		localName := imp.Name
		if localName == "" {
			localName = targetMod
		}
		c.currentImports[localName] = targetMod
	}
}

func (c *Checker) errorf(span token.Span, format string, args ...any) {
	c.errors = append(c.errors, SemanticError{
		Span: span,
		Msg:  fmt.Sprintf(format, args...),
	})
}

func (c *Checker) registerBuiltins() {
	c.functions["print"] = &FnSig{
		Name:       "print",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "msg", Type: TypeString, IsMut: false}},
		ReturnType: TypeVoid,
	}
	c.functions["println"] = &FnSig{
		Name:       "println",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "msg", Type: TypeString, IsMut: false}},
		ReturnType: TypeVoid,
	}
	c.functions["eprint"] = &FnSig{
		Name:       "eprint",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "msg", Type: TypeString, IsMut: false}},
		ReturnType: TypeVoid,
	}
	c.functions["eprintln"] = &FnSig{
		Name:       "eprintln",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "msg", Type: TypeString, IsMut: false}},
		ReturnType: TypeVoid,
	}
	c.functions["panic"] = &FnSig{
		Name:       "panic",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "msg", Type: TypeString}},
		ReturnType: TypeVoid,
	}
	c.functions["time_now_us"] = &FnSig{
		Name:       "time_now_us",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeI64,
	}
	c.functions["time_now_ms"] = &FnSig{
		Name:       "time_now_ms",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeI64,
	}
	c.functions["time_sleep_ms"] = &FnSig{
		Name:       "time_sleep_ms",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "ms", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["tensor_fill"] = &FnSig{
		Name:  "tensor_fill",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "rows", Type: TypeInt},
			{Name: "cols", Type: TypeInt},
			{Name: "val", Type: TypeF32},
		},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_fill_3d"] = &FnSig{
		Name:  "tensor_fill_3d",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "d0", Type: TypeInt},
			{Name: "d1", Type: TypeInt},
			{Name: "d2", Type: TypeInt},
			{Name: "val", Type: TypeF32},
		},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_fill_4d"] = &FnSig{
		Name:  "tensor_fill_4d",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "d0", Type: TypeInt},
			{Name: "d1", Type: TypeInt},
			{Name: "d2", Type: TypeInt},
			{Name: "d3", Type: TypeInt},
			{Name: "val", Type: TypeF32},
		},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_sum"] = &FnSig{
		Name:       "tensor_sum",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "t", Type: &TensorType{DType: TypeF32}}},
		ReturnType: TypeF32,
	}
	c.functions["tensor_relu"] = &FnSig{
		Name:  "tensor_relu",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "t", Type: &TensorType{DType: TypeF32}},
		},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_add"] = &FnSig{
		Name:  "tensor_add",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "a", Type: &TensorType{DType: TypeF32}},
			{Name: "b", Type: &TensorType{DType: TypeF32}},
		},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_scale"] = &FnSig{
		Name:  "tensor_scale",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "t", Type: &TensorType{DType: TypeF32}},
			{Name: "scale", Type: TypeF32},
		},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_load_bin"] = &FnSig{
		Name:  "tensor_load_bin",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "rows", Type: TypeInt},
			{Name: "cols", Type: TypeInt},
			{Name: "path", Type: TypeString},
		},
		ReturnType: &ResultType{Inner: &TensorType{DType: TypeF32}},
	}
	c.functions["tensor_save_bin"] = &FnSig{
		Name:  "tensor_save_bin",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "t", Type: &TensorType{DType: TypeF32}},
			{Name: "path", Type: TypeString},
		},
		ReturnType: &ResultType{Inner: TypeBool},
	}
	c.functions["file_exists"] = &FnSig{
		Name:       "file_exists",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "path", Type: TypeString}},
		ReturnType: TypeBool,
	}
	c.functions["read_file"] = &FnSig{
		Name:       "read_file",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "path", Type: TypeString}},
		ReturnType: &ResultType{Inner: TypeString},
	}
	c.functions["write_file"] = &FnSig{
		Name:   "write_file",
		IsPub:  true,
		Params: []ParamInfo{{Name: "path", Type: TypeString}, {Name: "content", Type: TypeString}},
		ReturnType: &ResultType{Inner: TypeBool},
	}
	c.functions["append_file"] = &FnSig{
		Name:   "append_file",
		IsPub:  true,
		Params: []ParamInfo{{Name: "path", Type: TypeString}, {Name: "content", Type: TypeString}},
		ReturnType: &ResultType{Inner: TypeBool},
	}
	c.functions["remove_file"] = &FnSig{
		Name:       "remove_file",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "path", Type: TypeString}},
		ReturnType: &ResultType{Inner: TypeBool},
	}

	// Option 1: Networking & Web Primitives
	c.functions["net_listen"] = &FnSig{
		Name:       "net_listen",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "port", Type: TypeInt}},
		ReturnType: TypeInt,
	}
	c.functions["net_accept"] = &FnSig{
		Name:       "net_accept",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "server_fd", Type: TypeInt}},
		ReturnType: TypeInt,
	}
	c.functions["net_read"] = &FnSig{
		Name:       "net_read",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "client_fd", Type: TypeInt}, {Name: "max_bytes", Type: TypeInt}},
		ReturnType: TypeString,
	}
	c.functions["net_write"] = &FnSig{
		Name:       "net_write",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "client_fd", Type: TypeInt}, {Name: "msg", Type: TypeString}},
		ReturnType: TypeInt,
	}
	c.functions["net_close"] = &FnSig{
		Name:       "net_close",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "fd", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["http_response"] = &FnSig{
		Name:   "http_response",
		IsPub:  true,
		Params: []ParamInfo{{Name: "status", Type: TypeInt}, {Name: "content_type", Type: TypeString}, {Name: "body", Type: TypeString}},
		ReturnType: TypeString,
	}

	// Option 1: JSON Parser Primitives
	c.functions["json_get_str"] = &FnSig{
		Name:   "json_get_str",
		IsPub:  true,
		Params: []ParamInfo{{Name: "json", Type: TypeString}, {Name: "key", Type: TypeString}},
		ReturnType: TypeString,
	}
	c.functions["json_get_int"] = &FnSig{
		Name:   "json_get_int",
		IsPub:  true,
		Params: []ParamInfo{{Name: "json", Type: TypeString}, {Name: "key", Type: TypeString}},
		ReturnType: TypeI64,
	}
	c.functions["json_get_f32"] = &FnSig{
		Name:   "json_get_f32",
		IsPub:  true,
		Params: []ParamInfo{{Name: "json", Type: TypeString}, {Name: "key", Type: TypeString}},
		ReturnType: TypeF32,
	}

	// Option 3: SafeTensors Zero-Copy Loader
	c.functions["safetensors_load"] = &FnSig{
		Name:   "safetensors_load",
		IsPub:  true,
		Params: []ParamInfo{{Name: "path", Type: TypeString}, {Name: "tensor_name", Type: TypeString}},
		ReturnType: &ResultType{Inner: &TensorType{DType: TypeF32}},
	}

	// Option 3: Autograd SIMD Primitives
	c.functions["tensor_transpose"] = &FnSig{
		Name:       "tensor_transpose",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "t", Type: &TensorType{DType: TypeF32}}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_matmul_grad_a"] = &FnSig{
		Name:       "tensor_matmul_grad_a",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "grad_out", Type: &TensorType{DType: TypeF32}}, {Name: "b", Type: &TensorType{DType: TypeF32}}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_matmul_grad_b"] = &FnSig{
		Name:       "tensor_matmul_grad_b",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "a", Type: &TensorType{DType: TypeF32}}, {Name: "grad_out", Type: &TensorType{DType: TypeF32}}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_relu_backward"] = &FnSig{
		Name:       "tensor_relu_backward",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "input", Type: &TensorType{DType: TypeF32}}, {Name: "grad_out", Type: &TensorType{DType: TypeF32}}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_sgd_step"] = &FnSig{
		Name:   "tensor_sgd_step",
		IsPub:  true,
		Params: []ParamInfo{{Name: "param", Type: &TensorType{DType: TypeF32}, IsMut: true}, {Name: "grad", Type: &TensorType{DType: TypeF32}}, {Name: "lr", Type: TypeF32}},
		ReturnType: TypeVoid,
	}

	// GGUF v2/v3 Zero-Copy Loader
	c.functions["gguf_load_tensor"] = &FnSig{
		Name:       "gguf_load_tensor",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "path", Type: TypeString}, {Name: "tensor_name", Type: TypeString}},
		ReturnType: &ResultType{Inner: &TensorType{DType: TypeF32}},
	}

	// LLM Transformer Operators
	c.functions["tensor_rmsnorm"] = &FnSig{
		Name:       "tensor_rmsnorm",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "x", Type: &TensorType{DType: TypeF32}}, {Name: "weight", Type: &TensorType{DType: TypeF32}}, {Name: "eps", Type: TypeF32}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_rope"] = &FnSig{
		Name:       "tensor_rope",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "q", Type: &TensorType{DType: TypeF32}}, {Name: "pos", Type: TypeInt}, {Name: "freq_base", Type: TypeF32}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_silu"] = &FnSig{
		Name:       "tensor_silu",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "x", Type: &TensorType{DType: TypeF32}}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_elemwise_mul"] = &FnSig{
		Name:       "tensor_elemwise_mul",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "a", Type: &TensorType{DType: TypeF32}}, {Name: "b", Type: &TensorType{DType: TypeF32}}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_softmax"] = &FnSig{
		Name:       "tensor_softmax",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "logits", Type: &TensorType{DType: TypeF32}}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["tensor_embedding"] = &FnSig{
		Name:       "tensor_embedding",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "table", Type: &TensorType{DType: TypeF32}}, {Name: "token_id", Type: TypeInt}},
		ReturnType: &TensorType{DType: TypeF32},
	}
	c.functions["sample_token"] = &FnSig{
		Name:       "sample_token",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "logits", Type: &TensorType{DType: TypeF32}}, {Name: "temperature", Type: TypeF32}},
		ReturnType: TypeInt,
	}

	// Native Hardware GUI Engine Builtins
	c.functions["gui_init"] = &FnSig{
		Name:       "gui_init",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "width", Type: TypeInt}, {Name: "height", Type: TypeInt}, {Name: "title", Type: TypeString}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_should_close"] = &FnSig{
		Name:       "gui_should_close",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeBool,
	}
	c.functions["gui_set_target_fps"] = &FnSig{
		Name:       "gui_set_target_fps",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "fps", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_begin"] = &FnSig{
		Name:       "gui_begin",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeVoid,
	}
	c.functions["gui_end"] = &FnSig{
		Name:       "gui_end",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeVoid,
	}
	c.functions["gui_close"] = &FnSig{
		Name:       "gui_close",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeVoid,
	}
	c.functions["gui_clear"] = &FnSig{
		Name:       "gui_clear",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "hex_color", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_draw_text"] = &FnSig{
		Name:  "gui_draw_text",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "text", Type: TypeString},
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "size", Type: TypeInt},
			{Name: "hex_color", Type: TypeInt},
		},
		ReturnType: TypeVoid,
	}
	c.functions["gui_draw_rect"] = &FnSig{
		Name:  "gui_draw_rect",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "width", Type: TypeInt},
			{Name: "height", Type: TypeInt},
			{Name: "hex_color", Type: TypeInt},
		},
		ReturnType: TypeVoid,
	}
	c.functions["gui_draw_circle"] = &FnSig{
		Name:  "gui_draw_circle",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "cx", Type: TypeInt},
			{Name: "cy", Type: TypeInt},
			{Name: "radius", Type: TypeF32},
			{Name: "hex_color", Type: TypeInt},
		},
		ReturnType: TypeVoid,
	}
	c.functions["gui_draw_line"] = &FnSig{
		Name:  "gui_draw_line",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x1", Type: TypeInt},
			{Name: "y1", Type: TypeInt},
			{Name: "x2", Type: TypeInt},
			{Name: "y2", Type: TypeInt},
			{Name: "hex_color", Type: TypeInt},
		},
		ReturnType: TypeVoid,
	}
	c.functions["gui_button"] = &FnSig{
		Name:  "gui_button",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "width", Type: TypeInt},
			{Name: "height", Type: TypeInt},
			{Name: "label", Type: TypeString},
		},
		ReturnType: TypeBool,
	}
	c.functions["gui_slider"] = &FnSig{
		Name:  "gui_slider",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "width", Type: TypeInt},
			{Name: "height", Type: TypeInt},
			{Name: "text_left", Type: TypeString},
			{Name: "text_right", Type: TypeString},
			{Name: "val", Type: TypeF32},
			{Name: "min_val", Type: TypeF32},
			{Name: "max_val", Type: TypeF32},
		},
		ReturnType: TypeF32,
	}
	c.functions["gui_checkbox"] = &FnSig{
		Name:  "gui_checkbox",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "width", Type: TypeInt},
			{Name: "height", Type: TypeInt},
			{Name: "label", Type: TypeString},
			{Name: "checked", Type: TypeBool},
		},
		ReturnType: TypeBool,
	}
	c.functions["gui_label"] = &FnSig{
		Name:  "gui_label",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "width", Type: TypeInt},
			{Name: "height", Type: TypeInt},
			{Name: "text", Type: TypeString},
		},
		ReturnType: TypeVoid,
	}
	c.functions["gui_text_input"] = &FnSig{
		Name:  "gui_text_input",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "width", Type: TypeInt},
			{Name: "height", Type: TypeInt},
			{Name: "placeholder", Type: TypeString},
			{Name: "text", Type: TypeString},
		},
		ReturnType: TypeString,
	}
	c.functions["gui_progress_bar"] = &FnSig{
		Name:  "gui_progress_bar",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "width", Type: TypeInt},
			{Name: "height", Type: TypeInt},
			{Name: "percent", Type: TypeF32},
		},
		ReturnType: TypeVoid,
	}
	c.functions["gui_radio_button"] = &FnSig{
		Name:  "gui_radio_button",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "label", Type: TypeString},
			{Name: "is_selected", Type: TypeBool},
		},
		ReturnType: TypeBool,
	}
	c.functions["gui_draw_badge"] = &FnSig{
		Name:  "gui_draw_badge",
		IsPub: true,
		Params: []ParamInfo{
			{Name: "x", Type: TypeInt},
			{Name: "y", Type: TypeInt},
			{Name: "text", Type: TypeString},
			{Name: "hex_bg", Type: TypeInt},
			{Name: "hex_fg", Type: TypeInt},
		},
		ReturnType: TypeVoid,
	}
	c.functions["gui_mouse_x"] = &FnSig{
		Name:       "gui_mouse_x",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeInt,
	}
	c.functions["gui_mouse_y"] = &FnSig{
		Name:       "gui_mouse_y",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeInt,
	}
	c.functions["gui_is_mouse_down"] = &FnSig{
		Name:       "gui_is_mouse_down",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "button", Type: TypeInt}},
		ReturnType: TypeBool,
	}
	c.functions["gui_is_key_pressed"] = &FnSig{
		Name:       "gui_is_key_pressed",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "key", Type: TypeInt}},
		ReturnType: TypeBool,
	}
	c.functions["gui_frame_time"] = &FnSig{
		Name:       "gui_frame_time",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeF32,
	}

	// Declarative Auto-Layout Engine Builtins
	c.functions["gui_vstack_begin"] = &FnSig{
		Name:       "gui_vstack_begin",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "x", Type: TypeInt}, {Name: "y", Type: TypeInt}, {Name: "width", Type: TypeInt}, {Name: "spacing", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_hstack_begin"] = &FnSig{
		Name:       "gui_hstack_begin",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "height", Type: TypeInt}, {Name: "spacing", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_stack_end"] = &FnSig{
		Name:       "gui_stack_end",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeVoid,
	}
	c.functions["gui_spacing"] = &FnSig{
		Name:       "gui_spacing",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "px", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_card_begin"] = &FnSig{
		Name:       "gui_card_begin",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "title", Type: TypeString}, {Name: "width", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_card_end"] = &FnSig{
		Name:       "gui_card_end",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeVoid,
	}
	c.functions["gui_btn"] = &FnSig{
		Name:       "gui_btn",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "label", Type: TypeString}},
		ReturnType: TypeBool,
	}
	c.functions["gui_check"] = &FnSig{
		Name:       "gui_check",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "label", Type: TypeString}, {Name: "checked", Type: TypeBool}},
		ReturnType: TypeBool,
	}
	c.functions["gui_slide"] = &FnSig{
		Name:       "gui_slide",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "text_left", Type: TypeString}, {Name: "text_right", Type: TypeString}, {Name: "val", Type: TypeF32}, {Name: "min_val", Type: TypeF32}, {Name: "max_val", Type: TypeF32}},
		ReturnType: TypeF32,
	}
	c.functions["gui_lbl"] = &FnSig{
		Name:       "gui_lbl",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "text", Type: TypeString}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_input"] = &FnSig{
		Name:       "gui_input",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "placeholder", Type: TypeString}, {Name: "text", Type: TypeString}},
		ReturnType: TypeString,
	}
	c.functions["gui_progress"] = &FnSig{
		Name:       "gui_progress",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "percent", Type: TypeF32}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_radio"] = &FnSig{
		Name:       "gui_radio",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "label", Type: TypeString}, {Name: "is_selected", Type: TypeBool}},
		ReturnType: TypeBool,
	}
	c.functions["gui_separator"] = &FnSig{
		Name:       "gui_separator",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeVoid,
	}
	c.functions["gui_badge"] = &FnSig{
		Name:       "gui_badge",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "text", Type: TypeString}, {Name: "hex_bg", Type: TypeInt}, {Name: "hex_fg", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_cancel_close"] = &FnSig{
		Name:       "gui_cancel_close",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeVoid,
	}
	c.functions["gui_window_width"] = &FnSig{
		Name:       "gui_window_width",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeInt,
	}
	c.functions["gui_window_height"] = &FnSig{
		Name:       "gui_window_height",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeInt,
	}
	c.functions["gui_tab_item"] = &FnSig{
		Name:       "gui_tab_item",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "label", Type: TypeString}, {Name: "is_active", Type: TypeBool}},
		ReturnType: TypeBool,
	}
	c.functions["gui_modal_begin"] = &FnSig{
		Name:       "gui_modal_begin",
		IsPub:      true,
		Params:     []ParamInfo{{Name: "title", Type: TypeString}, {Name: "width", Type: TypeInt}, {Name: "height", Type: TypeInt}},
		ReturnType: TypeVoid,
	}
	c.functions["gui_modal_end"] = &FnSig{
		Name:       "gui_modal_end",
		IsPub:      true,
		Params:     []ParamInfo{},
		ReturnType: TypeVoid,
	}
}

// CheckFile executes all 3 semantic passes on a parsed AST File.
func (c *Checker) CheckFile(file *ast.File) {
	c.CheckFiles([]*ast.File{file})
}

// CheckFiles executes all 3 semantic passes across multiple AST Files, handling cross-module namespaces.
func (c *Checker) CheckFiles(files []*ast.File) {
	// Pass 1A: Register type names (StructDecl, Actor, & InterfaceDecl shells)
	for _, file := range files {
		modName := file.Module.Name
		if modName == "" {
			modName = "main"
		}
		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.StructDecl:
				dMod := d.Module
				if dMod == "" {
					dMod = modName
				}
				dScope := c.getOrCreateModule(dMod)
				st := &StructType{
					Module:          dMod,
					Name:            d.Name.Name,
					IsPub:           d.IsPub,
					IsActor:         d.IsActor,
					Fields:          make(map[string]*FieldInfo),
					Methods:         make(map[string]*FnSig),
					StaticMethods:   make(map[string]*FnSig),
					PromotedFields:  make(map[string]PromotedField),
					PromotedMethods: make(map[string]PromotedMethod),
					Span:            d.DeclSpan,
				}
				dScope.Structs[d.Name.Name] = st
				c.structs[dMod+"."+d.Name.Name] = st
				if _, exists := c.structs[d.Name.Name]; !exists || dMod == "main" {
					c.structs[d.Name.Name] = st
				}
			case *ast.InterfaceDecl:
				dMod := d.Module
				if dMod == "" {
					dMod = modName
				}
				dScope := c.getOrCreateModule(dMod)
				iface := &InterfaceType{
					Module:  dMod,
					Name:    d.Name.Name,
					IsPub:   d.IsPub,
					Fields:  make(map[string]*FieldInfo),
					Methods: make(map[string]*FnSig),
					Span:    d.DeclSpan,
				}
				dScope.Interfaces[d.Name.Name] = iface
				c.interfaces[dMod+"."+d.Name.Name] = iface
				if _, exists := c.interfaces[d.Name.Name]; !exists || dMod == "main" {
					c.interfaces[d.Name.Name] = iface
				}
			}
		}
	}

	// Pass 1B: Resolve struct fields, embeds, interface members, and function/method headers
	for _, file := range files {
		fileMod := file.Module.Name
		if fileMod == "" {
			fileMod = "main"
		}
		c.currentMod = fileMod
		c.setFileImports(file.Imports)

		for _, decl := range file.Decls {
			savedMod := c.currentMod
			if d, ok := decl.(*ast.StructDecl); ok && d.Module != "" {
				c.currentMod = d.Module
			} else if d, ok := decl.(*ast.InterfaceDecl); ok && d.Module != "" {
				c.currentMod = d.Module
			} else if d, ok := decl.(*ast.FnDecl); ok && d.Module != "" {
				c.currentMod = d.Module
			} else if d, ok := decl.(*ast.ConstDecl); ok && d.Module != "" {
				c.currentMod = d.Module
			}

			switch d := decl.(type) {
			case *ast.StructDecl:
				c.registerStructMembers(d)
			case *ast.InterfaceDecl:
				c.registerInterfaceMembers(d)
			case *ast.FnDecl:
				c.registerFnHeader(d)
			case *ast.ConstDecl:
				c.registerConstDecl(d)
			}
			c.currentMod = savedMod
		}
	}

	// Pass 2: Flatten & promote embedded struct fields and methods across all modules
	visited := make(map[*StructType]bool)
	for _, st := range c.structs {
		if !visited[st] {
			c.promoteEmbeddedMembers(st)
			visited[st] = true
		}
	}
	for _, m := range c.modules {
		for _, st := range m.Structs {
			if !visited[st] {
				c.promoteEmbeddedMembers(st)
				visited[st] = true
			}
		}
	}

	// Pass 3: Type-check function/method bodies, mutability, 4-quadrant visibility, and !T handling
	for _, file := range files {
		fileMod := file.Module.Name
		if fileMod == "" {
			fileMod = "main"
		}
		c.currentMod = fileMod
		c.setFileImports(file.Imports)

		for _, decl := range file.Decls {
			if fn, ok := decl.(*ast.FnDecl); ok {
				savedMod := c.currentMod
				if fn.Module != "" {
					c.currentMod = fn.Module
				}
				c.checkFnBody(fn)
				c.currentMod = savedMod
			}
		}
	}

	// Reset currentMod to root file's module so c.ModuleName() reports the primary module
	rootMod := "main"
	if len(files) > 0 && files[0].Module.Name != "" {
		rootMod = files[0].Module.Name
	}
	c.currentMod = rootMod
}

// --- PASS 1: HEADER & MEMBER REGISTRATION ---

func (c *Checker) registerStructMembers(d *ast.StructDecl) {
	st := c.LookupStruct(d.Name.Name)
	if st == nil {
		return
	}

	for _, embExpr := range d.Embeds {
		embType := c.resolveTypeExpr(embExpr)
		embStruct, ok := embType.(*StructType)
		if !ok {
			c.errorf(embExpr.Span(), "embedded type '%s' in struct '%s' must be a struct", embType, d.Name.Name)
			continue
		}
		if embStruct.Name == st.Name {
			c.errorf(embExpr.Span(), "struct '%s' cannot embed itself recursively", st.Name)
			continue
		}
		st.Embeds = append(st.Embeds, embStruct)
	}

	for _, f := range d.Fields {
		if _, exists := st.Fields[f.Name.Name]; exists {
			c.errorf(f.FieldSpan, "duplicate field '%s' in struct '%s'", f.Name.Name, st.Name)
			continue
		}
		fType := c.resolveTypeExpr(f.Type)
		st.Fields[f.Name.Name] = &FieldInfo{
			Name:       f.Name.Name,
			Type:       fType,
			Access:     f.Access,
			OwnerMod:   c.currentMod,
			HasDefault: f.DefaultExpr != nil,
			Span:       f.FieldSpan,
		}
		st.FieldOrder = append(st.FieldOrder, f.Name.Name)
	}
}

func (c *Checker) registerInterfaceMembers(d *ast.InterfaceDecl) {
	iface := c.LookupInterface(d.Name.Name)
	if iface == nil {
		return
	}
	for _, f := range d.Fields {
		acc := ast.AccessPubImm
		if f.IsMut {
			acc = ast.AccessPubMut
		}
		iface.Fields[f.Name.Name] = &FieldInfo{
			Name:     f.Name.Name,
			Type:     c.resolveTypeExpr(f.Type),
			Access:   acc,
			OwnerMod: c.currentMod,
			Span:     f.FieldSpan,
		}
	}
	for _, m := range d.Methods {
		var params []ParamInfo
		for _, p := range m.Params {
			params = append(params, ParamInfo{
				Name:  p.Name.Name,
				Type:  c.resolveTypeExpr(p.Type),
				IsMut: p.IsMut,
			})
		}
		ret := Type(TypeVoid)
		if m.ReturnType != nil {
			ret = c.resolveTypeExpr(m.ReturnType)
		}
		iface.Methods[m.Name.Name] = &FnSig{
			Module:        c.currentMod,
			Name:          m.Name.Name,
			IsPub:         true,
			HasReceiver:   true,
			IsMutReceiver: m.IsMutReceiver,
			Params:        params,
			ReturnType:    ret,
			Span:          m.MethodSpan,
		}
	}
}

func (c *Checker) registerFnHeader(d *ast.FnDecl) {
	var params []ParamInfo
	for _, p := range d.Params {
		params = append(params, ParamInfo{
			Name:  p.Name.Name,
			Type:  c.resolveTypeExpr(p.Type),
			IsMut: p.IsMut,
		})
	}
	ret := Type(TypeVoid)
	if d.ReturnType != nil {
		ret = c.resolveTypeExpr(d.ReturnType)
	}

	sig := &FnSig{
		Module:     c.currentMod,
		Name:       d.Name.Name,
		IsPub:      d.IsPub,
		Params:     params,
		ReturnType: ret,
		Intrinsic:  d.Intrinsic,
		Span:       d.DeclSpan,
	}

	if d.Receiver != nil {
		recvType := c.resolveTypeExpr(d.Receiver.Type)
		sig.HasReceiver = true
		sig.ReceiverType = recvType
		sig.IsMutReceiver = d.Receiver.IsMut
		if st, ok := recvType.(*StructType); ok {
			st.Methods[d.Name.Name] = sig
		} else {
			c.errorf(d.Receiver.RecvSpan, "cannot define receiver method on non-struct type '%s'", recvType)
		}
		return
	}

	if d.StaticType != nil {
		if st := c.LookupStruct(d.StaticType.Name); st != nil {
			st.StaticMethods[d.Name.Name] = sig
		} else {
			c.errorf(d.StaticType.NamePos, "unknown struct '%s' for static method '%s'", d.StaticType.Name, d.Name.Name)
		}
		return
	}

	dMod := d.Module
	if dMod == "" {
		dMod = c.currentMod
	}
	sig.Module = dMod
	modScope := c.getOrCreateModule(dMod)
	modScope.Functions[d.Name.Name] = sig
	c.functions[dMod+"."+d.Name.Name] = sig
	if _, exists := c.functions[d.Name.Name]; !exists || dMod == "main" {
		c.functions[d.Name.Name] = sig
	}
}

func (c *Checker) registerConstDecl(d *ast.ConstDecl) {
	dMod := d.Module
	if dMod == "" {
		dMod = c.currentMod
	}
	constScope := NewScope(nil)
	valType := c.checkExpr(d.Value, constScope)
	finalType := valType
	if d.Type != nil {
		declType := c.resolveTypeExpr(d.Type)
		if !declType.Equals(valType) {
			c.errorf(d.DeclSpan, "constant '%s' declared as '%s', got '%s'", d.Name.Name, declType, valType)
		}
		finalType = declType
		c.ExprTypes[d.Value] = finalType
	}
	sym := &ConstSymbol{
		Module: dMod,
		Name:   d.Name.Name,
		IsPub:  d.IsPub,
		Type:   finalType,
		Value:  d.Value,
		Span:   d.DeclSpan,
	}
	modScope := c.getOrCreateModule(dMod)
	modScope.Consts[d.Name.Name] = sym
	c.consts[dMod+"."+d.Name.Name] = sym
	if _, exists := c.consts[d.Name.Name]; !exists || dMod == "main" {
		c.consts[d.Name.Name] = sym
	}
}

// --- PASS 2: STRUCT EMBEDDING PROMOTION (BFS FLATTENING) ---

func (c *Checker) promoteEmbeddedMembers(root *StructType) {
	type queueItem struct {
		st   *StructType
		path []string
	}
	visited := map[string]bool{root.Name: true}
	var queue []queueItem

	for _, emb := range root.Embeds {
		queue = append(queue, queueItem{st: emb, path: []string{emb.Name}})
	}

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		if visited[curr.st.Name] {
			continue
		}
		visited[curr.st.Name] = true

		// Promote fields not shadowed by root
		for fname, finfo := range curr.st.Fields {
			if _, shadowed := root.Fields[fname]; shadowed {
				continue
			}
			if existing, exists := root.PromotedFields[fname]; exists {
				if len(existing.EmbedPath) == len(curr.path) {
					existing.Ambiguous = true
					root.PromotedFields[fname] = existing
				}
				continue
			}
			root.PromotedFields[fname] = PromotedField{
				Field:     finfo,
				EmbedPath: curr.path,
			}
		}

		// Promote methods not shadowed by root
		for mname, msig := range curr.st.Methods {
			if _, shadowed := root.Methods[mname]; shadowed {
				continue
			}
			if existing, exists := root.PromotedMethods[mname]; exists {
				if len(existing.EmbedPath) == len(curr.path) {
					existing.Ambiguous = true
					root.PromotedMethods[mname] = existing
				}
				continue
			}
			root.PromotedMethods[mname] = PromotedMethod{
				Method:    msig,
				EmbedPath: curr.path,
			}
		}

		// Enqueue deeper embedded structs
		for _, deeper := range curr.st.Embeds {
			nextPath := append(append([]string{}, curr.path...), deeper.Name)
			queue = append(queue, queueItem{st: deeper, path: nextPath})
		}
	}
}

// --- PASS 3: BODY TYPE CHECKING, MUTABILITY & RESULT (!T) VERIFICATION ---

func (c *Checker) checkFnBody(d *ast.FnDecl) {
	scope := NewScope(nil)

	if d.Receiver != nil {
		recvType := c.resolveTypeExpr(d.Receiver.Type)
		scope.Define(&VarSymbol{
			Name:  d.Receiver.Name.Name,
			Type:  recvType,
			IsMut: d.Receiver.IsMut,
			Span:  d.Receiver.RecvSpan,
		})
	}

	for _, p := range d.Params {
		scope.Define(&VarSymbol{
			Name:  p.Name.Name,
			Type:  c.resolveTypeExpr(p.Type),
			IsMut: p.IsMut,
			Span:  p.ParamSpan,
		})
	}

	c.curFnReturn = TypeVoid
	if d.ReturnType != nil {
		c.curFnReturn = c.resolveTypeExpr(d.ReturnType)
	}

	c.checkBlock(d.Body, scope)
}

func (c *Checker) checkBlock(block *ast.BlockStmt, scope *Scope) {
	if block == nil {
		return
	}
	for _, stmt := range block.Stmts {
		c.checkStmt(stmt, scope)
	}
}

func (c *Checker) checkStmt(stmt ast.Stmt, scope *Scope) {
	switch s := stmt.(type) {
	case *ast.VarDeclStmt:
		valType := c.checkExpr(s.Value, scope)
		if s.Name.Name != "_" {
			if !scope.Define(&VarSymbol{
				Name:  s.Name.Name,
				Type:  valType,
				IsMut: s.IsMut,
				Span:  s.StmtSpan,
			}) {
				c.errorf(s.Name.NamePos, "redeclaration of variable '%s' in the same scope", s.Name.Name)
			}
		}

	case *ast.AssignStmt:
		lhsType := c.checkExpr(s.Target, scope)
		rhsType := c.checkExpr(s.Value, scope)

		// 1. Verify target mutability (both variable binding and 4-quadrant struct field rules)
		c.verifyAssignmentMutability(s.Target, scope, s.StmtSpan)

		// 2. Verify type compatibility
		if !lhsType.Equals(rhsType) {
			c.errorf(s.StmtSpan, "cannot assign value of type '%s' to target of type '%s'", rhsType, lhsType)
		}

	case *ast.IfStmt:
		condType := c.checkExpr(s.Cond, scope)
		if !condType.Equals(TypeBool) {
			c.errorf(s.Cond.Span(), "if condition must be boolean, got '%s'", condType)
		}
		thenScope := NewScope(scope)
		c.checkBlock(s.Then, thenScope)
		if s.Else != nil {
			elseScope := NewScope(scope)
			c.checkBlock(s.Else, elseScope)
		}

	case *ast.SwitchStmt:
		var tagType Type
		if s.Tag != nil {
			tagType = c.checkExpr(s.Tag, scope)
		}
		defaultCount := 0
		for _, cs := range s.Cases {
			if cs.IsDefault {
				defaultCount++
				if defaultCount > 1 {
					c.errorf(cs.CaseSpan, "multiple default cases in switch statement")
				}
			} else {
				for _, expr := range cs.Exprs {
					caseType := c.checkExpr(expr, scope)
					if s.Tag != nil {
						isMatch := caseType.Equals(tagType)
						if !isMatch && (caseType.Equals(TypeInt) || caseType.Equals(TypeI64) || caseType.Equals(TypeU8)) &&
							(tagType.Equals(TypeInt) || tagType.Equals(TypeI64) || tagType.Equals(TypeU8)) {
							isMatch = true
						}
						if !isMatch {
							c.errorf(expr.Span(), "type mismatch in switch case: cannot compare '%s' with '%s'", caseType, tagType)
						}
					} else {
						if !caseType.Equals(TypeBool) {
							c.errorf(expr.Span(), "tagless switch case condition must be boolean, got '%s'", caseType)
						}
					}
				}
			}
			caseScope := NewScope(scope)
			for _, caseStmt := range cs.Body {
				c.checkStmt(caseStmt, caseScope)
			}
		}

	case *ast.ForStmt:
		loopScope := NewScope(scope)
		if s.Init != nil {
			c.checkStmt(s.Init, loopScope)
		}
		if s.Cond != nil {
			condType := c.checkExpr(s.Cond, loopScope)
			if !condType.Equals(TypeBool) {
				c.errorf(s.Cond.Span(), "for condition must be boolean, got '%s'", condType)
			}
		}
		if s.Post != nil {
			c.checkStmt(s.Post, loopScope)
		}
		c.checkBlock(s.Body, loopScope)

	case *ast.ForInStmt:
		iterType := c.checkExpr(s.Iter, scope)
		var elemType Type = TypeVoid
		if sl, ok := iterType.(*SliceType); ok {
			elemType = sl.Elem
		} else if arr, ok := iterType.(*ArrayType); ok {
			elemType = arr.Elem
		} else if iterType.Equals(TypeString) {
			elemType = TypeString
		} else {
			c.errorf(s.Iter.Span(), "cannot iterate over type '%s' in for-in loop", iterType)
		}
		loopScope := NewScope(scope)
		loopScope.Define(&VarSymbol{Name: s.Var.Name, Type: elemType, IsMut: false, Span: s.Var.NamePos})
		c.checkBlock(s.Body, loopScope)

	case *ast.BranchStmt:
		// break and continue are valid inside loops

	case *ast.ArenaStmt:
		arenaScope := NewScope(scope)
		c.checkBlock(s.Body, arenaScope)

	case *ast.DeferStmt:
		deferScope := NewScope(scope)
		if s.Body != nil {
			for _, st := range s.Body.Stmts {
				if es, ok := st.(*ast.ExprStmt); ok {
					if call, isCall := es.Expr.(*ast.CallExpr); isCall && call.OrBlock == nil && !call.Propagate {
						prev := c.inDeferDiscard
						c.inDeferDiscard = true
						c.checkExpr(es.Expr, deferScope)
						c.inDeferDiscard = prev
						continue
					}
				}
				c.checkStmt(st, deferScope)
			}
		}

	case *ast.ReturnStmt:
		if s.Value == nil {
			if !c.curFnReturn.Equals(TypeVoid) {
				c.errorf(s.StmtSpan, "expected return value of type '%s', got bare return", c.curFnReturn)
			}
			return
		}
		valType := c.checkExpr(s.Value, scope)
		// If function returns `!T` and body returns `T`, automatically wrap `T` into `!T`
		if resRet, ok := c.curFnReturn.(*ResultType); ok {
			if valType.Equals(resRet.Inner) || valType.Equals(resRet) {
				return
			}
		}
		if optRet, ok := c.curFnReturn.(*OptionType); ok {
			if valType.Equals(optRet.Inner) || valType.Equals(optRet) {
				return
			}
		}
		if !c.curFnReturn.Equals(valType) {
			c.errorf(s.StmtSpan, "cannot return '%s' from function expecting '%s'", valType, c.curFnReturn)
		}

	case *ast.ExprStmt:
		c.checkExpr(s.Expr, scope)
	}
}

// verifyAssignmentMutability enforces V-style immutability-by-default and 4-quadrant field access.
func (c *Checker) verifyAssignmentMutability(target ast.Expr, scope *Scope, span token.Span) {
	switch t := target.(type) {
	case *ast.IdentExpr:
		if cSym, isConst := c.ResolvedConsts[t]; isConst {
			c.errorf(span, "cannot reassign constant '%s'", cSym.Name)
			return
		}
		sym, ok := scope.Lookup(t.Ident.Name)
		if !ok {
			return
		}
		if !sym.IsMut {
			c.errorf(span, "cannot reassign immutable variable '%s' (declare with 'mut %s := ...')",
				t.Ident.Name, t.Ident.Name)
		}

	case *ast.SelectorExpr:
		if cSym, isConst := c.ResolvedConsts[t]; isConst {
			c.errorf(span, "cannot reassign constant '%s.%s'", cSym.Module, cSym.Name)
			return
		}
		// 1. Root binding must be declared `mut`
		rootSym := c.findRootSymbol(t.Left, scope)
		if rootSym != nil && !rootSym.IsMut {
			c.errorf(span, "cannot mutate field '%s' because root binding '%s' is immutable (declare 'mut %s')",
				t.Field.Name, rootSym.Name, rootSym.Name)
			return
		}

		// 2. Check the struct field's 4-quadrant mutability and module boundary
		leftType := c.ExprTypes[t.Left]
		if st, ok := leftType.(*StructType); ok {
			finfo, _, _ := st.LookupField(t.Field.Name)
			if finfo == nil {
				return
			}
			sameMod := finfo.OwnerMod == c.currentMod
			switch finfo.Access {
			case ast.AccessPrivateImm:
				c.errorf(span, "field '%s.%s' is private and immutable (move to 'mut:' or 'pub mut:' block)",
					st.Name, finfo.Name)
			case ast.AccessPubImm:
				c.errorf(span, "field '%s.%s' is 'pub:' (read-only) and cannot be mutated",
					st.Name, finfo.Name)
			case ast.AccessPrivateMut:
				if !sameMod {
					c.errorf(span, "field '%s.%s' is module-private ('mut:') and cannot be mutated outside module '%s'",
						st.Name, finfo.Name, finfo.OwnerMod)
				}
			case ast.AccessPubMut:
				// Allowed since rootSym.IsMut is true!
			}
		}

	case *ast.IndexExpr:
		rootSym := c.findRootSymbol(t.Left, scope)
		if rootSym != nil && !rootSym.IsMut {
			c.errorf(span, "cannot mutate element because array/slice binding '%s' is immutable (declare 'mut %s := ...')",
				rootSym.Name, rootSym.Name)
		}
	}
}

func (c *Checker) findRootSymbol(expr ast.Expr, scope *Scope) *VarSymbol {
	switch e := expr.(type) {
	case *ast.IdentExpr:
		sym, _ := scope.Lookup(e.Ident.Name)
		return sym
	case *ast.SelectorExpr:
		return c.findRootSymbol(e.Left, scope)
	case *ast.IndexExpr:
		return c.findRootSymbol(e.Left, scope)
	default:
		return nil
	}
}

// --- EXPRESSION TYPE INFERENCE & !T / ?T UNWRAPPING ---

func (c *Checker) checkExpr(expr ast.Expr, scope *Scope) (resolved Type) {
	defer func() {
		if expr != nil && resolved != nil {
			c.ExprTypes[expr] = resolved
		}
	}()

	switch e := expr.(type) {
	case *ast.LiteralExpr:
		switch e.Kind {
		case token.TokInt:
			if strings.HasSuffix(e.Value, "i64") {
				return TypeI64
			}
			return TypeInt
		case token.TokFloat:
			if strings.HasSuffix(e.Value, "f32") {
				return TypeF32
			}
			if strings.HasSuffix(e.Value, "bf16") {
				return TypeBF16
			}
			return TypeF64
		case token.TokString:
			return TypeString
		case token.TokTrue, token.TokFalse:
			return TypeBool
		default:
			return TypeVoid
		}

	case *ast.StringInterpExpr:
		for _, sub := range e.Exprs {
			c.checkExpr(sub, scope)
		}
		return TypeString

	case *ast.IdentExpr:
		if e.Ident.Name == "_" {
			c.errorf(e.Ident.NamePos, "cannot use '_' as value or expression")
			return TypeVoid
		}
		sym, ok := scope.Lookup(e.Ident.Name)
		if ok {
			return sym.Type
		}
		if cSym := c.LookupConstInModule(c.currentMod, e.Ident.Name); cSym != nil {
			c.ResolvedConsts[e] = cSym
			return cSym.Type
		}
		if cSym, hasGlobal := c.consts[e.Ident.Name]; hasGlobal && (cSym.Module == c.currentMod || cSym.IsPub) {
			c.ResolvedConsts[e] = cSym
			return cSym.Type
		}
		c.errorf(e.Ident.NamePos, "undefined variable '%s'", e.Ident.Name)
		return TypeVoid

	case *ast.StructInitExpr:
		typ := c.resolveTypeExpr(e.Type)
		st, ok := typ.(*StructType)
		if !ok {
			c.errorf(e.InitSpan, "'%s' is not a struct type", typ)
			return TypeVoid
		}
		for _, initField := range e.Fields {
			valType := c.checkExpr(initField.Value, scope)
			if finfo, _, ambig := st.LookupField(initField.Name.Name); finfo != nil {
				if ambig {
					c.errorf(initField.FieldSpan, "ambiguous promoted field '%s' in struct '%s'", initField.Name.Name, st.Name)
				}
				if !finfo.Type.Equals(valType) {
					c.errorf(initField.FieldSpan, "field '%s.%s' expects type '%s', got '%s'",
						st.Name, initField.Name.Name, finfo.Type, valType)
				}
				continue
			}
			// Check if initializing an embedded struct directly (`AuditMeta: AuditMeta{...}`)
			matchedEmbed := false
			for _, emb := range st.Embeds {
				if emb.Name == initField.Name.Name {
					matchedEmbed = true
					if !emb.Equals(valType) {
						c.errorf(initField.FieldSpan, "embedded struct '%s' expects '%s', got '%s'",
							emb.Name, emb, valType)
					}
					break
				}
			}
			if !matchedEmbed {
				c.errorf(initField.FieldSpan, "unknown field '%s' in struct literal '%s'",
					initField.Name.Name, st.Name)
			}
		}
		return st

	case *ast.SelectorExpr:
		if identLeft, ok := e.Left.(*ast.IdentExpr); ok {
			if _, inScope := scope.Lookup(identLeft.Ident.Name); !inScope {
				if cSym := c.LookupConstInModule(identLeft.Ident.Name, e.Field.Name); cSym != nil {
					targetMod := identLeft.Ident.Name
					if resolved, ok := c.currentImports[targetMod]; ok {
						targetMod = resolved
					}
					if targetMod != c.currentMod && !cSym.IsPub {
						c.errorf(e.SelSpan, "constant '%s.%s' is private to module '%s'", identLeft.Ident.Name, e.Field.Name, targetMod)
						return TypeVoid
					}
					c.ResolvedConsts[e] = cSym
					return cSym.Type
				}
			}
		}
		leftType := c.checkExpr(e.Left, scope)
		if sl, ok := leftType.(*SliceType); ok {
			if e.Field.Name == "len" || e.Field.Name == "cap" {
				return TypeInt
			}
			c.errorf(e.SelSpan, "slice '%s' has no field '%s' (use .len or methods like .push())", sl, e.Field.Name)
			return TypeVoid
		}
		if arr, ok := leftType.(*ArrayType); ok {
			if e.Field.Name == "len" || e.Field.Name == "cap" {
				return TypeInt
			}
			c.errorf(e.SelSpan, "array '%s' has no field '%s' (use .len)", arr, e.Field.Name)
			return TypeVoid
		}
		if leftType.Equals(TypeString) && e.Field.Name == "len" {
			return TypeInt
		}
		if tt, ok := leftType.(*TensorType); ok {
			switch e.Field.Name {
			case "rows", "cols", "ndim", "n_dims":
				return TypeInt
			default:
				c.errorf(e.SelSpan, "tensor '%s' has no field '%s' (available: .rows, .cols, .ndim)", tt, e.Field.Name)
				return TypeVoid
			}
		}
		if st, ok := leftType.(*StructType); ok {
			finfo, path, ambig := st.LookupField(e.Field.Name)
			if ambig {
				c.errorf(e.SelSpan, "ambiguous field '%s' on struct '%s' (multiple embedded structs define it)",
					e.Field.Name, st.Name)
				return TypeVoid
			}
			if finfo != nil {
				if (finfo.Access == ast.AccessPrivateImm || finfo.Access == ast.AccessPrivateMut) && finfo.OwnerMod != c.currentMod {
					c.errorf(e.SelSpan, "cannot read private field '%s.%s' from outside module '%s'",
						st.Name, finfo.Name, finfo.OwnerMod)
				}
				if len(path) > 0 {
					c.SelectorPaths[e] = path
				}
				return finfo.Type
			}
			c.errorf(e.SelSpan, "struct '%s' has no field '%s'", st.Name, e.Field.Name)
			return TypeVoid
		}
		return TypeVoid

	case *ast.IndexExpr:
		leftType := c.checkExpr(e.Left, scope)
		idxType := c.checkExpr(e.Index, scope)
		if !idxType.Equals(TypeInt) && !idxType.Equals(TypeI64) {
			c.errorf(e.IndexSpan, "index must be integer, got '%s'", idxType)
		}
		if sl, ok := leftType.(*SliceType); ok {
			return sl.Elem
		}
		if arr, ok := leftType.(*ArrayType); ok {
			return arr.Elem
		}
		if leftType.Equals(TypeString) {
			return TypeString
		}
		c.errorf(e.IndexSpan, "cannot index value of type '%s'", leftType)
		return TypeVoid

	case *ast.ArrayLitExpr:
		if e.Type != nil {
			targetType := c.resolveTypeExpr(e.Type)
			if arr, ok := targetType.(*ArrayType); ok {
				if len(e.Elems) > arr.Len {
					c.errorf(e.LitSpan, "too many elements in array literal: expected %d, got %d", arr.Len, len(e.Elems))
				}
				for _, el := range e.Elems {
					elType := c.checkExpr(el, scope)
					if !elType.Equals(arr.Elem) {
						c.errorf(el.Span(), "mismatched element type in array literal: expected '%s', got '%s'", arr.Elem, elType)
					}
				}
				return arr
			}
			if sl, ok := targetType.(*SliceType); ok {
				for _, el := range e.Elems {
					elType := c.checkExpr(el, scope)
					if !elType.Equals(sl.Elem) {
						c.errorf(el.Span(), "mismatched element type in slice literal: expected '%s', got '%s'", sl.Elem, elType)
					}
				}
				return sl
			}
			c.errorf(e.LitSpan, "invalid array literal type '%s'", targetType)
			return targetType
		}
		if len(e.Elems) == 0 {
			return &SliceType{Elem: TypeInt}
		}
		firstElemType := c.checkExpr(e.Elems[0], scope)
		for i := 1; i < len(e.Elems); i++ {
			elType := c.checkExpr(e.Elems[i], scope)
			if !elType.Equals(firstElemType) {
				c.errorf(e.Elems[i].Span(), "mismatched element type in slice literal: expected '%s', got '%s'", firstElemType, elType)
			}
		}
		return &SliceType{Elem: firstElemType}

	case *ast.CallExpr:
		return c.checkCallExpr(e, scope)

	case *ast.BinaryExpr:
		l := c.checkExpr(e.Left, scope)
		r := c.checkExpr(e.Right, scope)
		if e.Op == token.TokAt || e.Op == token.TokPlus || e.Op == token.TokMinus {
			lt, lok := l.(*TensorType)
			rt, rok := r.(*TensorType)
			if lok && rok && lt.DType.Equals(rt.DType) {
				return lt
			}
			if lok && rok {
				c.errorf(e.BinSpan, "mismatched tensor dtypes in operation '%s': '%s' and '%s'", e.Op, lt.DType, rt.DType)
				return lt
			}
		}
		if !l.Equals(r) {
			c.errorf(e.BinSpan, "mismatched types in binary operation '%s': '%s' and '%s'", e.Op, l, r)
		}
		switch e.Op {
		case token.TokEq, token.TokNeq, token.TokLt, token.TokLte, token.TokGt, token.TokGte, token.TokAnd, token.TokOrLogical:
			return TypeBool
		default:
			if ExactEquals(l, TypeI64) || ExactEquals(r, TypeI64) {
				return TypeI64
			}
			return l
		}

	case *ast.UnaryExpr:
		t := c.checkExpr(e.Expr, scope)
		switch e.Op {
		case token.TokBang:
			if !t.Equals(TypeBool) {
				c.errorf(e.UnarySpan, "operator '!' expects boolean, got '%s'", t)
			}
			return TypeBool
		case token.TokMinus:
			if !t.Equals(TypeInt) && !t.Equals(TypeI64) && !t.Equals(TypeF32) && !t.Equals(TypeF64) {
				c.errorf(e.UnarySpan, "operator '-' expects numeric type, got '%s'", t)
			}
			return t
		default:
			return t
		}

	default:
		return TypeVoid
	}
}

func (c *Checker) checkCallExpr(call *ast.CallExpr, scope *Scope) Type {
	var sig *FnSig
	var callName string

	switch callee := call.Callee.(type) {
	case *ast.IdentExpr:
		callName = callee.Ident.Name
		// Built-in `error('msg')` constructor
		if callName == "error" {
			if len(call.Args) != 1 || !c.checkExpr(call.Args[0].Expr, scope).Equals(TypeString) {
				c.errorf(call.CallSpan, "builtin 'error(...)' expects a single 'string' argument")
			}
			if resRet, ok := c.curFnReturn.(*ResultType); ok {
				return resRet
			}
			return &ResultType{Inner: TypeVoid}
		}

		// Built-in primitive numeric/boolean casts: `u8(x)`, `int(x)`, `i64(x)`, `f32(x)`, `f64(x)`
		switch callName {
		case "u8", "byte", "int", "i64", "f32", "f64", "bool":
			var targetType Type
			switch callName {
			case "u8", "byte":
				targetType = TypeU8
			case "int":
				targetType = TypeInt
			case "i64":
				targetType = TypeI64
			case "f32":
				targetType = TypeF32
			case "f64":
				targetType = TypeF64
			case "bool":
				targetType = TypeBool
			}
			if len(call.Args) != 1 {
				c.errorf(call.CallSpan, "type cast '%s(...)' expects 1 argument, got %d", callName, len(call.Args))
				return targetType
			}
			argType := c.checkExpr(call.Args[0].Expr, scope)
			if argType.Equals(TypeString) {
				c.errorf(call.CallSpan, "cannot cast 'string' directly to '%s' (use .to_int(), .to_i64(), .to_f64(), or .to_f32())", callName)
				return targetType
			}
			if _, isPrim := argType.(*PrimitiveType); !isPrim || argType.Equals(TypeVoid) {
				c.errorf(call.CallSpan, "cannot cast '%s' to '%s'", argType, callName)
			}
			return targetType
		}

		// Built-in structural interface cast: `Payable(inv)`
		if iface, isIface := c.interfaces[callName]; isIface {
			if len(call.Args) != 1 {
				c.errorf(call.CallSpan, "interface cast '%s(...)' expects 1 value", callName)
				return iface
			}
			argType := c.checkExpr(call.Args[0].Expr, scope)
			if st, ok := argType.(*StructType); ok {
				c.CheckInterfaceSatisfaction(st, iface, call.CallSpan)
			} else {
				c.errorf(call.CallSpan, "type '%s' is not a struct and cannot implement interface '%s'", argType, iface.Name)
			}
			return iface
		}

		// 1. Look in current module functions
		if modScope, ok := c.modules[c.currentMod]; ok {
			if fSig, ok := modScope.Functions[callName]; ok {
				sig = fSig
			}
		}
		// 2. Fallback to global c.functions (builtins and global functions)
		if sig == nil {
			sig = c.functions[callName]
		}
		// 3. Did user call an unqualified imported function? Give helpful diagnostic
		if sig == nil {
			for pkgAlias, modTarget := range c.currentImports {
				if modScope, ok := c.modules[modTarget]; ok {
					if fnSig, ok := modScope.Functions[callName]; ok && fnSig.IsPub {
						c.errorf(callee.Span(), "cannot find unqualified function '%s'; did you mean '%s.%s'?",
							callName, pkgAlias, callName)
						sig = fnSig
						break
					}
				}
			}
		}
		if sig == nil {
			c.errorf(callee.Span(), "undefined function '%s'", callName)
			return TypeVoid
		}

	case *ast.SelectorExpr:
		// 1a. Qualified Static Method: `net.TcpStream.connect(...)`
		if selLeft, ok := callee.Left.(*ast.SelectorExpr); ok {
			if pkgIdent, ok := selLeft.Left.(*ast.IdentExpr); ok {
				pkgName := pkgIdent.Ident.Name
				targetMod := pkgName
				if resolvedMod, ok := c.currentImports[pkgName]; ok {
					targetMod = resolvedMod
				}
				modScope := c.modules[targetMod]
				if modScope == nil && targetMod == c.currentMod {
					modScope = c.getOrCreateModule(targetMod)
				}
				if modScope != nil {
					typeName := selLeft.Field.Name
					if st, ok := modScope.Structs[typeName]; ok {
						if targetMod != c.currentMod && !st.IsPub {
							c.errorf(selLeft.SelSpan, "struct '%s.%s' is private to module '%s'", pkgName, typeName, targetMod)
							return TypeVoid
						}
						methodName := callee.Field.Name
						sig = st.StaticMethods[methodName]
						if sig == nil {
							c.errorf(callee.SelSpan, "struct '%s.%s' has no static method '%s'", pkgName, typeName, methodName)
							return TypeVoid
						}
						if targetMod != c.currentMod && !sig.IsPub {
							c.errorf(callee.SelSpan, "static method '%s.%s.%s' is private", pkgName, typeName, methodName)
							return TypeVoid
						}
						callName = fmt.Sprintf("%s.%s.%s", pkgName, typeName, methodName)
					}
				}
			}
		}

		// 1b. Static Type Factory: `Invoice.new(...)`
		if sig == nil {
			if identLeft, ok := callee.Left.(*ast.IdentExpr); ok {
				if st := c.LookupStruct(identLeft.Ident.Name); st != nil {
					if sm, ok := st.StaticMethods[callee.Field.Name]; ok {
						sig = sm
						callName = st.Name + "." + callee.Field.Name
					}
				}
			}
		}

		// 1c. Top-level Module Function: `io.copy(...)`
		if sig == nil {
			if identLeft, ok := callee.Left.(*ast.IdentExpr); ok {
				if _, inScope := scope.Lookup(identLeft.Ident.Name); !inScope {
					pkgName := identLeft.Ident.Name
					targetMod := pkgName
					if resolvedMod, ok := c.currentImports[pkgName]; ok {
						targetMod = resolvedMod
					}
					if modScope, ok := c.modules[targetMod]; ok {
						fnName := callee.Field.Name
						if fnSig, ok := modScope.Functions[fnName]; ok {
							if targetMod != c.currentMod && !fnSig.IsPub {
								c.errorf(callee.SelSpan, "function '%s.%s' is private to module '%s'", pkgName, fnName, targetMod)
								return TypeVoid
							}
							sig = fnSig
							callName = fmt.Sprintf("%s.%s", pkgName, fnName)
						}
					}
				}
			}
		}

		// 2. Check if this is `.await()` on a `Future[T]`
		if sig == nil {
			recvType := c.checkExpr(callee.Left, scope)
			if futType, isFut := recvType.(*FutureType); isFut {
				if callee.Field.Name != "await" {
					c.errorf(callee.SelSpan, "type '%s' has no method '%s' (only '.await()' is supported)",
						futType, callee.Field.Name)
					return TypeVoid
				}
				callName = futType.String() + ".await"
				sig = &FnSig{
					Name:       "await",
					IsPub:      true,
					Params:     nil,
					ReturnType: &ResultType{Inner: futType.Inner},
					Span:       callee.SelSpan,
				}
			} else if sl, isSlice := recvType.(*SliceType); isSlice {
				switch callee.Field.Name {
				case "push", "append":
					rootSym := c.findRootSymbol(callee.Left, scope)
					if rootSym != nil && !rootSym.IsMut {
						c.errorf(call.CallSpan, "cannot call mutating method '%s()' on immutable slice '%s' (declare 'mut %s := ...')",
							callee.Field.Name, rootSym.Name, rootSym.Name)
					}
					sig = &FnSig{
						Name:       callee.Field.Name,
						Params:     []ParamInfo{{Name: "item", Type: sl.Elem}},
						ReturnType: TypeVoid,
						Span:       callee.SelSpan,
					}
					callName = "slice." + callee.Field.Name
				case "pop":
					rootSym := c.findRootSymbol(callee.Left, scope)
					if rootSym != nil && !rootSym.IsMut {
						c.errorf(call.CallSpan, "cannot call mutating method 'pop()' on immutable slice '%s' (declare 'mut %s := ...')",
							rootSym.Name, rootSym.Name)
					}
					sig = &FnSig{
						Name:       "pop",
						Params:     nil,
						ReturnType: sl.Elem,
						Span:       callee.SelSpan,
					}
					callName = "slice.pop"
				case "get":
					sig = &FnSig{
						Name:       "get",
						Params:     []ParamInfo{{Name: "index", Type: TypeInt}},
						ReturnType: &ResultType{Inner: sl.Elem},
						Span:       callee.SelSpan,
					}
					callName = "slice.get"
				case "set":
					rootSym := c.findRootSymbol(callee.Left, scope)
					if rootSym != nil && !rootSym.IsMut {
						c.errorf(call.CallSpan, "cannot call mutating method 'set()' on immutable slice '%s' (declare 'mut %s := ...')",
							rootSym.Name, rootSym.Name)
					}
					sig = &FnSig{
						Name:       "set",
						Params:     []ParamInfo{{Name: "index", Type: TypeInt}, {Name: "val", Type: sl.Elem}},
						ReturnType: TypeVoid,
						Span:       callee.SelSpan,
					}
					callName = "slice.set"
				case "clear":
					rootSym := c.findRootSymbol(callee.Left, scope)
					if rootSym != nil && !rootSym.IsMut {
						c.errorf(call.CallSpan, "cannot call mutating method 'clear()' on immutable slice '%s' (declare 'mut %s := ...')",
							rootSym.Name, rootSym.Name)
					}
					sig = &FnSig{
						Name:       "clear",
						Params:     nil,
						ReturnType: TypeVoid,
						Span:       callee.SelSpan,
					}
					callName = "slice.clear"
				case "clone":
					sig = &FnSig{
						Name:       "clone",
						Params:     nil,
						ReturnType: sl,
						Span:       callee.SelSpan,
					}
					callName = "slice.clone"
				case "slice":
					sig = &FnSig{
						Name:       "slice",
						Params:     []ParamInfo{{Name: "start", Type: TypeInt}, {Name: "end", Type: TypeInt}},
						ReturnType: sl,
						Span:       callee.SelSpan,
					}
					callName = "slice.slice"
				case "contains":
					sig = &FnSig{
						Name:       "contains",
						Params:     []ParamInfo{{Name: "val", Type: sl.Elem}},
						ReturnType: TypeBool,
						Span:       callee.SelSpan,
					}
					callName = "slice.contains"
				case "is_empty":
					sig = &FnSig{
						Name:       "is_empty",
						Params:     nil,
						ReturnType: TypeBool,
						Span:       callee.SelSpan,
					}
					callName = "slice.is_empty"
				case "len", "cap":
					sig = &FnSig{
						Name:       callee.Field.Name,
						Params:     nil,
						ReturnType: TypeInt,
						Span:       callee.SelSpan,
					}
					callName = "slice." + callee.Field.Name
				case "to_string", "to_str":
					if !sl.Elem.Equals(TypeU8) {
						c.errorf(callee.SelSpan, "to_string() is only available on []u8, got %s", sl)
						return TypeVoid
					}
					sig = &FnSig{
						Name:       callee.Field.Name,
						Params:     nil,
						ReturnType: TypeString,
						Span:       callee.SelSpan,
					}
					callName = "slice." + callee.Field.Name
				default:
					c.errorf(callee.SelSpan, "slice '%s' has no method '%s'", sl, callee.Field.Name)
					return TypeVoid
				}
			} else if arr, isArr := recvType.(*ArrayType); isArr {
				switch callee.Field.Name {
				case "get":
					sig = &FnSig{
						Name:       "get",
						Params:     []ParamInfo{{Name: "index", Type: TypeInt}},
						ReturnType: &ResultType{Inner: arr.Elem},
						Span:       callee.SelSpan,
					}
					callName = "array.get"
				case "contains":
					sig = &FnSig{
						Name:       "contains",
						Params:     []ParamInfo{{Name: "val", Type: arr.Elem}},
						ReturnType: TypeBool,
						Span:       callee.SelSpan,
					}
					callName = "array.contains"
				case "is_empty":
					sig = &FnSig{
						Name:       "is_empty",
						Params:     nil,
						ReturnType: TypeBool,
						Span:       callee.SelSpan,
					}
					callName = "array.is_empty"
				case "len":
					sig = &FnSig{
						Name:       "len",
						Params:     nil,
						ReturnType: TypeInt,
						Span:       callee.SelSpan,
					}
					callName = "array.len"
				default:
					c.errorf(callee.SelSpan, "array '%s' has no method '%s'", arr, callee.Field.Name)
					return TypeVoid
				}
			} else if recvType.Equals(TypeString) {
				switch callee.Field.Name {
				case "len":
					sig = &FnSig{Name: "len", Params: nil, ReturnType: TypeInt, Span: callee.SelSpan}
					callName = "string.len"
				case "contains":
					sig = &FnSig{Name: "contains", Params: []ParamInfo{{Name: "substr", Type: TypeString}}, ReturnType: TypeBool, Span: callee.SelSpan}
					callName = "string.contains"
				case "index_of":
					sig = &FnSig{Name: "index_of", Params: []ParamInfo{{Name: "substr", Type: TypeString}}, ReturnType: TypeInt, Span: callee.SelSpan}
					callName = "string.index_of"
				case "substr", "slice":
					sig = &FnSig{Name: callee.Field.Name, Params: []ParamInfo{{Name: "start", Type: TypeInt}, {Name: "end", Type: TypeInt}}, ReturnType: TypeString, Span: callee.SelSpan}
					callName = "string." + callee.Field.Name
				case "starts_with":
					sig = &FnSig{Name: "starts_with", Params: []ParamInfo{{Name: "prefix", Type: TypeString}}, ReturnType: TypeBool, Span: callee.SelSpan}
					callName = "string.starts_with"
				case "ends_with":
					sig = &FnSig{Name: "ends_with", Params: []ParamInfo{{Name: "suffix", Type: TypeString}}, ReturnType: TypeBool, Span: callee.SelSpan}
					callName = "string.ends_with"
				case "split":
					sig = &FnSig{Name: "split", Params: []ParamInfo{{Name: "sep", Type: TypeString}}, ReturnType: &SliceType{Elem: TypeString}, Span: callee.SelSpan}
					callName = "string.split"
				case "trim":
					sig = &FnSig{Name: "trim", Params: nil, ReturnType: TypeString, Span: callee.SelSpan}
					callName = "string.trim"
				case "to_lower":
					sig = &FnSig{Name: "to_lower", Params: nil, ReturnType: TypeString, Span: callee.SelSpan}
					callName = "string.to_lower"
				case "to_upper":
					sig = &FnSig{Name: "to_upper", Params: nil, ReturnType: TypeString, Span: callee.SelSpan}
					callName = "string.to_upper"
				case "to_int":
					sig = &FnSig{Name: "to_int", Params: nil, ReturnType: &ResultType{Inner: TypeInt}, Span: callee.SelSpan}
					callName = "string.to_int"
				case "to_i64":
					sig = &FnSig{Name: "to_i64", Params: nil, ReturnType: &ResultType{Inner: TypeI64}, Span: callee.SelSpan}
					callName = "string.to_i64"
				case "to_f64":
					sig = &FnSig{Name: "to_f64", Params: nil, ReturnType: &ResultType{Inner: TypeF64}, Span: callee.SelSpan}
					callName = "string.to_f64"
				case "to_f32":
					sig = &FnSig{Name: "to_f32", Params: nil, ReturnType: &ResultType{Inner: TypeF32}, Span: callee.SelSpan}
					callName = "string.to_f32"
				case "replace":
					sig = &FnSig{Name: "replace", Params: []ParamInfo{{Name: "old", Type: TypeString}, {Name: "new_str", Type: TypeString}}, ReturnType: TypeString, Span: callee.SelSpan}
					callName = "string.replace"
				case "last_index_of":
					sig = &FnSig{Name: "last_index_of", Params: []ParamInfo{{Name: "substr", Type: TypeString}}, ReturnType: TypeInt, Span: callee.SelSpan}
					callName = "string.last_index_of"
				case "trim_left":
					sig = &FnSig{Name: "trim_left", Params: nil, ReturnType: TypeString, Span: callee.SelSpan}
					callName = "string.trim_left"
				case "trim_right":
					sig = &FnSig{Name: "trim_right", Params: nil, ReturnType: TypeString, Span: callee.SelSpan}
					callName = "string.trim_right"
				case "repeat":
					sig = &FnSig{Name: "repeat", Params: []ParamInfo{{Name: "count", Type: TypeInt}}, ReturnType: TypeString, Span: callee.SelSpan}
					callName = "string.repeat"
				case "lines":
					sig = &FnSig{Name: "lines", Params: nil, ReturnType: &SliceType{Elem: TypeString}, Span: callee.SelSpan}
					callName = "string.lines"
				case "byte_at":
					sig = &FnSig{Name: "byte_at", Params: []ParamInfo{{Name: "index", Type: TypeInt}}, ReturnType: &ResultType{Inner: TypeU8}, Span: callee.SelSpan}
					callName = "string.byte_at"
				case "clone":
					sig = &FnSig{Name: "clone", Params: nil, ReturnType: TypeString, Span: callee.SelSpan}
					callName = "string.clone"
				case "bytes", "to_bytes":
					sig = &FnSig{Name: callee.Field.Name, Params: nil, ReturnType: &SliceType{Elem: TypeU8}, Span: callee.SelSpan}
					callName = "string." + callee.Field.Name
				default:
					c.errorf(callee.SelSpan, "string has no method '%s'", callee.Field.Name)
					return TypeVoid
				}
			} else if tt, isTensor := recvType.(*TensorType); isTensor {
				switch callee.Field.Name {
				case "rows", "cols", "ndim", "numel":
					sig = &FnSig{Name: callee.Field.Name, Params: nil, ReturnType: TypeInt, Span: callee.SelSpan}
					callName = "tensor." + callee.Field.Name
				case "dim", "stride":
					sig = &FnSig{
						Name:       callee.Field.Name,
						Params:     []ParamInfo{{Name: "axis", Type: TypeInt}},
						ReturnType: TypeInt,
						Span:       callee.SelSpan,
					}
					callName = "tensor." + callee.Field.Name
				case "reshape_2d":
					sig = &FnSig{
						Name:       "reshape_2d",
						Params:     []ParamInfo{{Name: "d0", Type: TypeInt}, {Name: "d1", Type: TypeInt}},
						ReturnType: tt,
						Span:       callee.SelSpan,
					}
					callName = "tensor.reshape_2d"
				case "reshape_3d":
					sig = &FnSig{
						Name:       "reshape_3d",
						Params:     []ParamInfo{{Name: "d0", Type: TypeInt}, {Name: "d1", Type: TypeInt}, {Name: "d2", Type: TypeInt}},
						ReturnType: tt,
						Span:       callee.SelSpan,
					}
					callName = "tensor.reshape_3d"
				case "reshape_4d":
					sig = &FnSig{
						Name:       "reshape_4d",
						Params:     []ParamInfo{{Name: "d0", Type: TypeInt}, {Name: "d1", Type: TypeInt}, {Name: "d2", Type: TypeInt}, {Name: "d3", Type: TypeInt}},
						ReturnType: tt,
						Span:       callee.SelSpan,
					}
					callName = "tensor.reshape_4d"
				case "transpose", "clone":
					sig = &FnSig{
						Name:       callee.Field.Name,
						Params:     nil,
						ReturnType: tt,
						Span:       callee.SelSpan,
					}
					callName = "tensor." + callee.Field.Name
				default:
					c.errorf(callee.SelSpan, "tensor '%s' has no method '%s'", tt, callee.Field.Name)
					return TypeVoid
				}
			} else if st, ok := recvType.(*StructType); ok {
				// 3. Instance method, promoted method, or Actor `<-` message send
				var path []string
				var ambig bool
				baseSig, path, ambig := st.LookupMethod(callee.Field.Name)
				if ambig {
					c.errorf(callee.SelSpan, "ambiguous promoted method '%s.%s'", st.Name, callee.Field.Name)
					return TypeVoid
				}
				if baseSig == nil {
					c.errorf(callee.SelSpan, "method '%s' not found on '%s'", callee.Field.Name, st.Name)
					return TypeVoid
				}
				if len(path) > 0 {
					c.SelectorPaths[callee] = path
				}
				callName = st.Name + "." + callee.Field.Name

				if baseSig.IsMutReceiver {
					rootSym := c.findRootSymbol(callee.Left, scope)
					if rootSym == nil || !rootSym.IsMut {
						name := "<expr>"
						if rootSym != nil {
							name = rootSym.Name
						}
						c.errorf(call.CallSpan,
							"cannot call mutable receiver method '%s()' on immutable binding '%s' (declare 'mut %s := ...')",
							callee.Field.Name, name, name)
					}
				}

				// Clone signature to adapt return type for Actor `<-` sends
				sigCopy := *baseSig
				if callee.IsActorSend {
					if !st.IsActor {
						c.errorf(callee.SelSpan, "cannot use actor send '<-' on non-actor struct '%s'", st.Name)
					}
					innerRet := baseSig.ReturnType
					if res, isRes := innerRet.(*ResultType); isRes {
						innerRet = res.Inner
					}
					// If no inline `or {}` or `!` is attached, `actor <- method()` returns `Future[T]`
					if call.OrBlock == nil && !call.Propagate {
						sigCopy.ReturnType = &FutureType{Inner: innerRet}
					} else {
						// Inline shorthand: `actor <- method() or { ... }` awaits immediately
						sigCopy.ReturnType = &ResultType{Inner: innerRet}
					}
				}
				sig = &sigCopy
			} else if iface, isIface := recvType.(*InterfaceType); isIface {
				msig, exists := iface.Methods[callee.Field.Name]
				if !exists {
					c.errorf(callee.SelSpan, "interface '%s' has no method '%s'", iface.Name, callee.Field.Name)
					return TypeVoid
				}
				if msig.IsMutReceiver {
					rootSym := c.findRootSymbol(callee.Left, scope)
					if rootSym == nil || !rootSym.IsMut {
						name := "<expr>"
						if rootSym != nil {
							name = rootSym.Name
						}
						c.errorf(call.CallSpan,
							"cannot call mutable receiver method '%s()' on immutable binding '%s' (declare 'mut %s := ...')",
							callee.Field.Name, name, name)
					}
				}
				callName = iface.Name + "." + callee.Field.Name
				sig = msig
			} else {
				c.errorf(callee.SelSpan, "cannot call method '%s' on type '%s'", callee.Field.Name, recvType)
				return TypeVoid
			}
		}
	}

	// Verify argument count, types, and explicit `mut` call-site modifiers
	if len(call.Args) != len(sig.Params) {
		c.errorf(call.CallSpan, "call to '%s' expects %d arguments, got %d",
			callName, len(sig.Params), len(call.Args))
	} else {
		for i, arg := range call.Args {
			argType := c.checkExpr(arg.Expr, scope)
			param := sig.Params[i]

			if param.IsMut && !arg.IsMut {
				c.errorf(arg.ArgSpan, "parameter '%s' of '%s' is mutable; pass it explicitly with 'mut'",
					param.Name, callName)
			}
			if arg.IsMut {
				rootSym := c.findRootSymbol(arg.Expr, scope)
				if rootSym == nil || !rootSym.IsMut {
					c.errorf(arg.ArgSpan, "cannot pass immutable variable as 'mut' argument to '%s'", callName)
				}
			}
			if param.IsMut {
				if !ExactEquals(param.Type, argType) {
					c.errorf(arg.ArgSpan, "argument %d ('%s') of '%s' expects 'mut %s', got '%s'",
						i+1, param.Name, callName, param.Type, argType)
				}
			} else if !param.Type.Equals(argType) {
				c.errorf(arg.ArgSpan, "argument %d ('%s') of '%s' expects '%s', got '%s'",
					i+1, param.Name, callName, param.Type, argType)
			}
		}
	}

	// Verify Result (`!T`) and Option (`?T`) handling (`or {}` block or `!`/`?` propagation)
	switch ret := sig.ReturnType.(type) {
	case *ResultType:
		if call.Propagate {
			if _, ok := c.curFnReturn.(*ResultType); !ok {
				c.errorf(call.CallSpan,
					"cannot propagate '!' on '%s()': enclosing function returns '%s', not a Result '!T'",
					callName, c.curFnReturn)
			} else {
				errIdent := &ast.IdentExpr{Ident: ast.Ident{Name: "err", NamePos: call.CallSpan}}
				c.ExprTypes[errIdent] = TypeString
				errCallee := &ast.IdentExpr{Ident: ast.Ident{Name: "error", NamePos: call.CallSpan}}
				errCall := &ast.CallExpr{
					Callee:   errCallee,
					Args:     []ast.CallArg{{Expr: errIdent, ArgSpan: call.CallSpan}},
					CallSpan: call.CallSpan,
				}
				c.ExprTypes[errCall] = c.curFnReturn
				retStmt := &ast.ReturnStmt{Value: errCall, StmtSpan: call.CallSpan}
				call.OrBlock = &ast.BlockStmt{Stmts: []ast.Stmt{retStmt}, BlockSpan: call.CallSpan}
				call.Propagate = false
			}
			return ret.Inner
		}
		if call.OrBlock != nil {
			orScope := NewScope(scope)
			// Inject implicit `err` string into `or { ... }` block scope
			orScope.Define(&VarSymbol{Name: "err", Type: TypeString, IsMut: false, Span: call.OrBlock.BlockSpan})
			c.checkOrBlock(call.OrBlock, orScope, ret.Inner)
			return ret.Inner
		}
		if c.inDeferDiscard {
			call.OrBlock = &ast.BlockStmt{BlockSpan: call.CallSpan}
			return ret.Inner
		}
		c.errorf(call.CallSpan,
			"unhandled Result '%s' from '%s()': handle it with 'or { ... }' or propagate with '!'",
			ret, callName)
		return ret.Inner

	case *OptionType:
		if call.Propagate {
			if _, ok := c.curFnReturn.(*OptionType); !ok {
				c.errorf(call.CallSpan,
					"cannot propagate '?' on '%s()': enclosing function returns '%s', not an Option '?T'",
					callName, c.curFnReturn)
			}
			return ret.Inner
		}
		if call.OrBlock != nil {
			orScope := NewScope(scope)
			c.checkOrBlock(call.OrBlock, orScope, ret.Inner)
			return ret.Inner
		}
		if c.inDeferDiscard {
			call.OrBlock = &ast.BlockStmt{BlockSpan: call.CallSpan}
			return ret.Inner
		}
		c.errorf(call.CallSpan,
			"unhandled Option '%s' from '%s()': handle it with 'or { ... }' or propagate with '?'",
			ret, callName)
		return ret.Inner

	default:
		if call.OrBlock != nil || call.Propagate {
			c.errorf(call.CallSpan, "function '%s()' returns '%s', which is not a Result (!T) or Option (?T)",
				callName, sig.ReturnType)
		}
		return sig.ReturnType
	}
}

func (c *Checker) checkOrBlock(block *ast.BlockStmt, scope *Scope, expectedType Type) {
	if len(block.Stmts) == 0 {
		if !expectedType.Equals(TypeVoid) {
			c.errorf(block.BlockSpan, "empty 'or {}' block for non-void type '%s': must return or provide a default value", expectedType)
		}
		return
	}

	for i, stmt := range block.Stmts {
		isLast := i == len(block.Stmts)-1
		if isLast {
			switch last := stmt.(type) {
			case *ast.ReturnStmt:
				if last.Value != nil {
					c.checkStmt(last, scope)
				}
				return
			case *ast.BranchStmt:
				c.checkStmt(last, scope)
				return
			case *ast.ExprStmt:
				if IsDivergentExpr(last.Expr) {
					c.checkExpr(last.Expr, scope)
					return
				}
				valType := c.checkExpr(last.Expr, scope)
				if !expectedType.Equals(TypeVoid) && !valType.Equals(expectedType) {
					c.errorf(last.Span(), "'or' block fallback expression has type '%s', expected '%s'",
						valType, expectedType)
				}
				return
			}
		}
		c.checkStmt(stmt, scope)
	}
}

func (c *Checker) CheckInterfaceSatisfaction(st *StructType, iface *InterfaceType, span token.Span) bool {
	ok := true

	// 1. Check required interface fields
	for fname, reqField := range iface.Fields {
		gotField, _, ambig := st.LookupField(fname)
		if ambig || gotField == nil {
			c.errorf(span, "struct '%s' does not implement interface '%s': missing field '%s %s'",
				st.Name, iface.Name, fname, reqField.Type)
			ok = false
			continue
		}
		if !gotField.Type.Equals(reqField.Type) {
			c.errorf(span, "struct '%s' field '%s' has type '%s', but interface '%s' requires '%s'",
				st.Name, fname, gotField.Type, iface.Name, reqField.Type)
			ok = false
		}
		if reqField.Access == ast.AccessPubMut && gotField.Access != ast.AccessPubMut && gotField.Access != ast.AccessPrivateMut {
			c.errorf(span, "interface '%s' requires mutable field '%s', but '%s.%s' is immutable",
				iface.Name, fname, st.Name, fname)
			ok = false
		}
	}

	// 2. Check required interface methods
	for mname, reqMethod := range iface.Methods {
		gotMethod, _, ambig := st.LookupMethod(mname)
		if ambig || gotMethod == nil {
			c.errorf(span, "struct '%s' does not implement interface '%s': missing method '%s'",
				st.Name, iface.Name, mname)
			ok = false
			continue
		}
		if gotMethod.IsMutReceiver && !reqMethod.IsMutReceiver {
			c.errorf(span, "struct '%s' method '%s' requires a 'mut' receiver, but interface '%s' declares it immutable",
				st.Name, mname, iface.Name)
			ok = false
		}
		if !gotMethod.ReturnType.Equals(reqMethod.ReturnType) {
			c.errorf(span, "struct '%s' method '%s' returns '%s', but interface '%s' expects '%s'",
				st.Name, mname, gotMethod.ReturnType, iface.Name, reqMethod.ReturnType)
			ok = false
		}
	}

	return ok
}

// IsDivergentExpr reports whether expr terminates process/actor execution without returning (panic, exit, os.exit, log.fatal).
func IsDivergentExpr(expr ast.Expr) bool {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return false
	}
	if id, isId := call.Callee.(*ast.IdentExpr); isId && (id.Ident.Name == "panic" || id.Ident.Name == "exit") {
		return true
	}
	if sel, isSel := call.Callee.(*ast.SelectorExpr); isSel {
		if pkgId, isId := sel.Left.(*ast.IdentExpr); isId {
			if (pkgId.Ident.Name == "os" && sel.Field.Name == "exit") ||
				(pkgId.Ident.Name == "log" && sel.Field.Name == "fatal") {
				return true
			}
		}
	}
	return false
}

func (c *Checker) resolveTypeExpr(te ast.TypeExpr) Type {
	if te == nil {
		return TypeVoid
	}
	switch t := te.(type) {
	case *ast.SelectorType:
		pkgName := t.Pkg.Name
		targetMod := pkgName
		if resolvedMod, ok := c.currentImports[pkgName]; ok {
			targetMod = resolvedMod
		}
		modScope := c.modules[targetMod]
		if modScope == nil && targetMod == c.currentMod {
			modScope = c.getOrCreateModule(targetMod)
		}
		if modScope == nil {
			c.errorf(t.Pkg.NamePos, "module '%s' is not imported or does not exist", pkgName)
			return TypeVoid
		}
		typeName := t.Name.Name
		if st, ok := modScope.Structs[typeName]; ok {
			if targetMod != c.currentMod && !st.IsPub {
				c.errorf(t.Name.NamePos, "type '%s.%s' is private to module '%s'", pkgName, typeName, targetMod)
			}
			return st
		}
		if iface, ok := modScope.Interfaces[typeName]; ok {
			if targetMod != c.currentMod && !iface.IsPub {
				c.errorf(t.Name.NamePos, "interface '%s.%s' is private to module '%s'", pkgName, typeName, targetMod)
			}
			return iface
		}
		c.errorf(t.Name.NamePos, "module '%s' has no type '%s'", pkgName, typeName)
		return TypeVoid

	case *ast.IdentType:
		switch t.Name.Name {
		case "u8", "byte":
			return TypeU8
		case "int":
			return TypeInt
		case "i64":
			return TypeI64
		case "f32":
			return TypeF32
		case "bf16":
			return TypeBF16
		case "f64":
			return TypeF64
		case "bool":
			return TypeBool
		case "string":
			return TypeString
		case "void":
			return TypeVoid
		case "tensor":
			if len(t.TypeParams) == 1 {
				return &TensorType{DType: c.resolveTypeExpr(t.TypeParams[0])}
			}
			return &TensorType{DType: TypeF32}
		case "Future":
			if len(t.TypeParams) == 1 {
				return &FutureType{Inner: c.resolveTypeExpr(t.TypeParams[0])}
			}
			return &FutureType{Inner: TypeVoid}
		}

		// 1. Look in current module scope
		if modScope, ok := c.modules[c.currentMod]; ok {
			if st, ok := modScope.Structs[t.Name.Name]; ok {
				return st
			}
			if iface, ok := modScope.Interfaces[t.Name.Name]; ok {
				return iface
			}
		}

		// 2. Check if it exists in one of the imported modules (enforcing strict qualification)
		for pkgAlias, modTarget := range c.currentImports {
			if modScope, ok := c.modules[modTarget]; ok {
				if st, ok := modScope.Structs[t.Name.Name]; ok && st.IsPub {
					c.errorf(t.Name.NamePos, "cannot find unqualified type '%s'; did you mean '%s.%s'?",
						t.Name.Name, pkgAlias, t.Name.Name)
					return st
				}
				if iface, ok := modScope.Interfaces[t.Name.Name]; ok && iface.IsPub {
					c.errorf(t.Name.NamePos, "cannot find unqualified interface '%s'; did you mean '%s.%s'?",
						t.Name.Name, pkgAlias, t.Name.Name)
					return iface
				}
			}
		}

		// 3. Fallback to global c.structs/c.interfaces (for main/unspecified module and backwards compatibility)
		if c.currentMod == "main" || c.currentMod == "" {
			if st, ok := c.structs[t.Name.Name]; ok {
				return st
			}
			if iface, ok := c.interfaces[t.Name.Name]; ok {
				return iface
			}
		}

		c.errorf(t.Name.NamePos, "unknown type '%s'", t.Name.Name)
		return TypeVoid

	case *ast.ResultType:
		inner := Type(TypeVoid)
		if t.Inner != nil {
			inner = c.resolveTypeExpr(t.Inner)
		}
		return &ResultType{Inner: inner}

	case *ast.OptionType:
		return &OptionType{Inner: c.resolveTypeExpr(t.Inner)}

	case *ast.RefType:
		return &RefType{
			Inner: c.resolveTypeExpr(t.Inner),
			IsMut: t.IsMut,
		}

	case *ast.ArrayType:
		elem := c.resolveTypeExpr(t.Elem)
		if t.Len != nil {
			if lit, ok := t.Len.(*ast.LiteralExpr); ok && lit.Kind == token.TokInt {
				var n int
				fmt.Sscanf(lit.Value, "%d", &n)
				if n <= 0 {
					c.errorf(t.LBracketPos, "array length must be positive, got %d", n)
					n = 1
				}
				return &ArrayType{Len: n, Elem: elem}
			}
			c.errorf(t.LBracketPos, "array length must be a constant integer")
			return &ArrayType{Len: 1, Elem: elem}
		}
		return &SliceType{Elem: elem}

	default:
		return TypeVoid
	}
}

func (c *Checker) FormatPromotions(structName string) string {
	st, ok := c.structs[structName]
	if !ok {
		return ""
	}
	var out strings.Builder
	for name, pf := range st.PromotedFields {
		out.WriteString(fmt.Sprintf("  ├── PromotedField  %-12s (%s) via %s\n",
			name, pf.Field.Type, strings.Join(pf.EmbedPath, ".")))
	}
	for name, pm := range st.PromotedMethods {
		out.WriteString(fmt.Sprintf("  ├── PromotedMethod %-12s %s via %s\n",
			name, pm.Method, strings.Join(pm.EmbedPath, ".")))
	}
	return out.String()
}
