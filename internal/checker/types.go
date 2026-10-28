package checker

import (
	"fmt"
	"strings"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/token"
)

// Type represents a resolved semantic type in Vex.
type Type interface {
	String() string
	Equals(other Type) bool
}

type PrimitiveKind string

const (
	PrimVoid   PrimitiveKind = "void"
	PrimU8     PrimitiveKind = "u8"
	PrimInt    PrimitiveKind = "int"
	PrimI64    PrimitiveKind = "i64"
	PrimF32    PrimitiveKind = "f32"
	PrimBF16   PrimitiveKind = "bf16"
	PrimF64    PrimitiveKind = "f64"
	PrimBool   PrimitiveKind = "bool"
	PrimString PrimitiveKind = "string"
)

type PrimitiveType struct {
	Kind PrimitiveKind
}

func (p *PrimitiveType) String() string { return string(p.Kind) }

func (p *PrimitiveType) Equals(other Type) bool {
	o, ok := other.(*PrimitiveType)
	if !ok {
		return false
	}
	// Allow seamless integer literal promotion between u8, int and i64 for ergonomics
	if (p.Kind == PrimInt && o.Kind == PrimI64) || (p.Kind == PrimI64 && o.Kind == PrimInt) ||
		(p.Kind == PrimU8 && (o.Kind == PrimInt || o.Kind == PrimI64)) ||
		((p.Kind == PrimInt || p.Kind == PrimI64) && o.Kind == PrimU8) {
		return true
	}
	// Allow float literals (f64) to promote cleanly to f32 / bf16
	if (p.Kind == PrimF32 || p.Kind == PrimF64 || p.Kind == PrimBF16) &&
		(o.Kind == PrimF32 || o.Kind == PrimF64 || o.Kind == PrimBF16) {
		return true
	}
	return p.Kind == o.Kind
}

// ExactEquals checks strict type identity without implicit numeric primitive promotion.
// Used for invariant container element types ([]T, [N]T, &T, tensor[T]) and mutable pointer arguments.
func ExactEquals(a, b Type) bool {
	if a == nil || b == nil {
		return a == b
	}
	if pa, ok := a.(*PrimitiveType); ok {
		pb, okB := b.(*PrimitiveType)
		return okB && pa.Kind == pb.Kind
	}
	return a.Equals(b)
}

var (
	TypeVoid   = &PrimitiveType{Kind: PrimVoid}
	TypeU8     = &PrimitiveType{Kind: PrimU8}
	TypeInt    = &PrimitiveType{Kind: PrimInt}
	TypeI64    = &PrimitiveType{Kind: PrimI64}
	TypeF32    = &PrimitiveType{Kind: PrimF32}
	TypeBF16   = &PrimitiveType{Kind: PrimBF16}
	TypeF64    = &PrimitiveType{Kind: PrimF64}
	TypeBool   = &PrimitiveType{Kind: PrimBool}
	TypeString = &PrimitiveType{Kind: PrimString}
)

// ResultType represents `!T` (or bare `!` when Inner is TypeVoid).
type ResultType struct {
	Inner Type
}

func (r *ResultType) String() string {
	if r.Inner == nil || r.Inner.Equals(TypeVoid) {
		return "!"
	}
	return "!" + r.Inner.String()
}

func (r *ResultType) Equals(other Type) bool {
	o, ok := other.(*ResultType)
	return ok && r.Inner.Equals(o.Inner)
}

// OptionType represents `?T`.
type OptionType struct {
	Inner Type
}

func (o *OptionType) String() string { return "?" + o.Inner.String() }

func (o *OptionType) Equals(other Type) bool {
	ot, ok := other.(*OptionType)
	return ok && o.Inner.Equals(ot.Inner)
}

// RefType represents `&T` or `&mut T`.
type RefType struct {
	Inner Type
	IsMut bool
}

func (r *RefType) String() string {
	if r.IsMut {
		return "&mut " + r.Inner.String()
	}
	return "&" + r.Inner.String()
}

func (r *RefType) Equals(other Type) bool {
	o, ok := other.(*RefType)
	return ok && r.IsMut == o.IsMut && ExactEquals(r.Inner, o.Inner)
}

// TensorType represents `tensor[T]`.
type TensorType struct {
	DType Type
}

func (t *TensorType) String() string { return fmt.Sprintf("tensor[%s]", t.DType) }

func (t *TensorType) Equals(other Type) bool {
	o, ok := other.(*TensorType)
	return ok && ExactEquals(t.DType, o.DType)
}

// FutureType represents `Future[T]` returned by asynchronous actor sends (`worker <- msg()`).
type FutureType struct {
	Inner Type
}

func (f *FutureType) String() string { return fmt.Sprintf("Future[%s]", f.Inner) }

func (f *FutureType) Equals(other Type) bool {
	o, ok := other.(*FutureType)
	return ok && ExactEquals(f.Inner, o.Inner)
}

// ArrayType represents a fixed-size stack array `[N]T`.
type ArrayType struct {
	Len  int
	Elem Type
}

func (a *ArrayType) String() string { return fmt.Sprintf("[%d]%s", a.Len, a.Elem) }

func (a *ArrayType) Equals(other Type) bool {
	o, ok := other.(*ArrayType)
	return ok && a.Len == o.Len && ExactEquals(a.Elem, o.Elem)
}

// SliceType represents a growable dynamic array / slice `[]T`.
type SliceType struct {
	Elem Type
}

func (s *SliceType) String() string { return fmt.Sprintf("[]%s", s.Elem) }

func (s *SliceType) Equals(other Type) bool {
	o, ok := other.(*SliceType)
	return ok && ExactEquals(s.Elem, o.Elem)
}

// FieldInfo stores resolved metadata for a struct or interface field.
type FieldInfo struct {
	Name       string
	Type       Type
	Access     ast.AccessMode
	OwnerMod   string
	HasDefault bool
	Span       token.Span
}

// ParamInfo stores a resolved parameter signature.
type ParamInfo struct {
	Name  string
	Type  Type
	IsMut bool
}

// FnSig stores the signature of a free function, receiver method, or static factory.
type FnSig struct {
	Module        string
	Name          string
	IsPub         bool
	HasReceiver   bool
	ReceiverType  Type
	IsMutReceiver bool
	Params        []ParamInfo
	ReturnType    Type
	Intrinsic     string
	Span          token.Span
}

func (f *FnSig) String() string {
	var params []string
	for _, p := range f.Params {
		prefix := ""
		if p.IsMut {
			prefix = "mut "
		}
		params = append(params, fmt.Sprintf("%s%s %s", prefix, p.Name, p.Type))
	}
	return fmt.Sprintf("fn(%s) %s", strings.Join(params, ", "), f.ReturnType)
}

// PromotedField tracks a field promoted via struct embedding along with its hop chain.
type PromotedField struct {
	Field     *FieldInfo
	EmbedPath []string // e.g., ["AuditMeta"] so inv.created_at lowers to inv.AuditMeta.created_at
	Ambiguous bool
}

// PromotedMethod tracks a receiver method promoted via struct embedding.
type PromotedMethod struct {
	Method    *FnSig
	EmbedPath []string // e.g., ["AuditMeta"] so inv.touch() lowers to AuditMeta_touch(&inv.AuditMeta)
	Ambiguous bool
}

// StructType models a resolved Vira struct or actor with 4-quadrant fields, methods, and embedded promotions.
type StructType struct {
	Module          string
	Name            string
	IsPub           bool
	IsActor         bool // true if declared with `actor Name { ... }`
	Embeds          []*StructType
	Fields          map[string]*FieldInfo
	FieldOrder      []string
	Methods         map[string]*FnSig
	StaticMethods   map[string]*FnSig
	PromotedFields  map[string]PromotedField
	PromotedMethods map[string]PromotedMethod
	Span            token.Span
}

func (s *StructType) String() string {
	if s.Module != "" && s.Module != "main" {
		return s.Module + "." + s.Name
	}
	return s.Name
}

func (s *StructType) Equals(other Type) bool {
	o, ok := other.(*StructType)
	return ok && s.Module == o.Module && s.Name == o.Name
}

// LookupField resolves direct fields first (shadowing), then promoted embedded fields.
func (s *StructType) LookupField(name string) (*FieldInfo, []string, bool) {
	if f, ok := s.Fields[name]; ok {
		return f, nil, false
	}
	if pf, ok := s.PromotedFields[name]; ok {
		return pf.Field, pf.EmbedPath, pf.Ambiguous
	}
	return nil, nil, false
}

// LookupMethod resolves direct receiver methods first, then promoted embedded methods.
func (s *StructType) LookupMethod(name string) (*FnSig, []string, bool) {
	if m, ok := s.Methods[name]; ok {
		return m, nil, false
	}
	if pm, ok := s.PromotedMethods[name]; ok {
		return pm.Method, pm.EmbedPath, pm.Ambiguous
	}
	return nil, nil, false
}

// InterfaceType models a structural interface with required fields and methods.
type InterfaceType struct {
	Module  string
	Name    string
	IsPub   bool
	Fields  map[string]*FieldInfo
	Methods map[string]*FnSig
	Span    token.Span
}

func (i *InterfaceType) String() string {
	if i.Module != "" && i.Module != "main" {
		return i.Module + "." + i.Name
	}
	return i.Name
}

func (i *InterfaceType) Equals(other Type) bool {
	o, ok := other.(*InterfaceType)
	return ok && i.Module == o.Module && i.Name == o.Name
}
