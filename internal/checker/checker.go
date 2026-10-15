package checker

import (
	"fmt"
	"strings"

	"github.com/vexlang/vex/internal/ast"
	"github.com/vexlang/vex/internal/token"
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

type Checker struct {
	currentMod string
	structs    map[string]*StructType
	interfaces map[string]*InterfaceType
	functions  map[string]*FnSig
	errors     []SemanticError

	// Semantic side-tables consumed by Phase 3 (autofree) and Phase 4 (cgen)
	ExprTypes     map[ast.Expr]Type
	SelectorPaths map[*ast.SelectorExpr][]string // records embedded hop path (e.g. ["AuditMeta"])

	// Function context during Pass 3
	curFnReturn Type
}

func New() *Checker {
	c := &Checker{
		structs:       make(map[string]*StructType),
		interfaces:    make(map[string]*InterfaceType),
		functions:     make(map[string]*FnSig),
		ExprTypes:     make(map[ast.Expr]Type),
		SelectorPaths: make(map[*ast.SelectorExpr][]string),
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

func (c *Checker) ModuleName() string {
	return c.currentMod
}

func (c *Checker) LookupStruct(name string) *StructType {
	return c.structs[name]
}

func (c *Checker) errorf(span token.Span, format string, args ...any) {
	c.errors = append(c.errors, SemanticError{
		Span: span,
		Msg:  fmt.Sprintf(format, args...),
	})
}

func (c *Checker) registerBuiltins() {
	c.functions["println"] = &FnSig{
		Name:       "println",
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
}

// CheckFile executes all 3 semantic passes on a parsed AST File.
func (c *Checker) CheckFile(file *ast.File) {
	c.currentMod = file.Module.Name
	if c.currentMod == "" {
		c.currentMod = "main"
	}

	// Pass 1A: Register type names (StructDecl, Actor, & InterfaceDecl shells)
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.StructDecl:
			c.structs[d.Name.Name] = &StructType{
				Module:          c.currentMod,
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
		case *ast.InterfaceDecl:
			c.interfaces[d.Name.Name] = &InterfaceType{
				Module:  c.currentMod,
				Name:    d.Name.Name,
				IsPub:   d.IsPub,
				Fields:  make(map[string]*FieldInfo),
				Methods: make(map[string]*FnSig),
				Span:    d.DeclSpan,
			}
		}
	}

	// Pass 1B: Resolve struct fields, embeds, interface members, and function/method headers
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.StructDecl:
			c.registerStructMembers(d)
		case *ast.InterfaceDecl:
			c.registerInterfaceMembers(d)
		case *ast.FnDecl:
			c.registerFnHeader(d)
		}
	}

	// Pass 2: Flatten & promote embedded struct fields and methods
	for _, st := range c.structs {
		c.promoteEmbeddedMembers(st)
	}

	// Pass 3: Type-check function/method bodies, mutability, 4-quadrant visibility, and !T handling
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FnDecl); ok {
			c.checkFnBody(fn)
		}
	}
}

// --- PASS 1: HEADER & MEMBER REGISTRATION ---

func (c *Checker) registerStructMembers(d *ast.StructDecl) {
	st := c.structs[d.Name.Name]

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
	iface := c.interfaces[d.Name.Name]
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
		Name:       d.Name.Name,
		IsPub:      d.IsPub,
		Params:     params,
		ReturnType: ret,
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
		if st, ok := c.structs[d.StaticType.Name]; ok {
			st.StaticMethods[d.Name.Name] = sig
		} else {
			c.errorf(d.StaticType.NamePos, "unknown struct '%s' for static method '%s'", d.StaticType.Name, d.Name.Name)
		}
		return
	}

	c.functions[d.Name.Name] = sig
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

	case *ast.BranchStmt:
		// break and continue are valid inside loops

	case *ast.ArenaStmt:
		arenaScope := NewScope(scope)
		c.checkBlock(s.Body, arenaScope)

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
		sym, ok := scope.Lookup(t.Ident.Name)
		if !ok {
			return
		}
		if !sym.IsMut {
			c.errorf(span, "cannot reassign immutable variable '%s' (declare with 'mut %s := ...')",
				t.Ident.Name, t.Ident.Name)
		}

	case *ast.SelectorExpr:
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
	}
}

func (c *Checker) findRootSymbol(expr ast.Expr, scope *Scope) *VarSymbol {
	switch e := expr.(type) {
	case *ast.IdentExpr:
		sym, _ := scope.Lookup(e.Ident.Name)
		return sym
	case *ast.SelectorExpr:
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
			return TypeInt
		case token.TokFloat:
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
		if !ok {
			c.errorf(e.Ident.NamePos, "undefined variable '%s'", e.Ident.Name)
			return TypeVoid
		}
		return sym.Type

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
		leftType := c.checkExpr(e.Left, scope)
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
			if !t.Equals(TypeInt) && !t.Equals(TypeF32) && !t.Equals(TypeF64) {
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

		sig = c.functions[callName]
		if sig == nil {
			c.errorf(callee.Span(), "undefined function '%s'", callName)
			return TypeVoid
		}

	case *ast.SelectorExpr:
		// 1. Static Type Factory: `Invoice.new(...)`
		if identLeft, ok := callee.Left.(*ast.IdentExpr); ok {
			if st, isStruct := c.structs[identLeft.Ident.Name]; isStruct {
				callName = st.Name + "." + callee.Field.Name
				sig = st.StaticMethods[callee.Field.Name]
				if sig == nil {
					c.errorf(callee.SelSpan, "struct '%s' has no static method '%s'", st.Name, callee.Field.Name)
					return TypeVoid
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
			if !param.Type.Equals(argType) {
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
			case *ast.ExprStmt:
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

func (c *Checker) resolveTypeExpr(te ast.TypeExpr) Type {
	if te == nil {
		return TypeVoid
	}
	switch t := te.(type) {
	case *ast.IdentType:
		switch t.Name.Name {
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
		if st, ok := c.structs[t.Name.Name]; ok {
			return st
		}
		if iface, ok := c.interfaces[t.Name.Name]; ok {
			return iface
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
