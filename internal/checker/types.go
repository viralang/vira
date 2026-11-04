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

	// C-ABI Primitive Types (v0.7 FFI)
	PrimCChar      PrimitiveKind = "c_char"
	PrimCUChar     PrimitiveKind = "c_uchar"
	PrimCShort     PrimitiveKind = "c_short"
	PrimCUShort    PrimitiveKind = "c_ushort"
	PrimCInt       PrimitiveKind = "c_int"
	PrimCUInt      PrimitiveKind = "c_uint"
	PrimCLong      PrimitiveKind = "c_long"
	PrimCULong     PrimitiveKind = "c_ulong"
	PrimCLongLong  PrimitiveKind = "c_longlong"
	PrimCULongLong PrimitiveKind = "c_ulonglong"
	PrimCSize      PrimitiveKind = "c_size"
	PrimCSSize     PrimitiveKind = "c_ssize"
	PrimCFloat     PrimitiveKind = "c_float"
	PrimCDouble    PrimitiveKind = "c_double"
	PrimVoidPtr    PrimitiveKind = "voidptr"
)

func IsIntLike(k PrimitiveKind) bool {
	switch k {
	case PrimU8, PrimInt, PrimI64, PrimCChar, PrimCUChar, PrimCShort, PrimCUShort,
		PrimCInt, PrimCUInt, PrimCLong, PrimCULong, PrimCLongLong, PrimCULongLong,
		PrimCSize, PrimCSSize:
		return true
	default:
		return false
	}
}

func isIntLike(k PrimitiveKind) bool {
	return IsIntLike(k)
}

func IsFloatLike(k PrimitiveKind) bool {
	switch k {
	case PrimF32, PrimF64, PrimBF16, PrimCFloat, PrimCDouble:
		return true
	default:
		return false
	}
}

func isFloatLike(k PrimitiveKind) bool {
	return IsFloatLike(k)
}

type PrimitiveType struct {
	Kind PrimitiveKind
}

func (p *PrimitiveType) String() string { return string(p.Kind) }

func (p *PrimitiveType) Equals(other Type) bool {
	o, ok := other.(*PrimitiveType)
	if !ok {
		// voidptr matches any &T or &mut T raw pointer
		if p.Kind == PrimVoidPtr {
			_, isRef := other.(*RefType)
			return isRef
		}
		if p.Kind == PrimVoidPtr {
			if _, isRef := other.(*RefType); isRef {
				return true
			}
		}
		if oRef, isRef := other.(*RefType); isRef {
			if _, isInnerVoid := oRef.Inner.(*PrimitiveType); isInnerVoid && p.Kind == PrimVoidPtr {
				return true
			}
		}
		return false
	}
	if p.Kind == o.Kind {
		return true
	}
	// voidptr matches any pointer or voidptr
	if p.Kind == PrimVoidPtr || o.Kind == PrimVoidPtr {
		return true
	}
	// Allow seamless integer literal and C integer promotion
	if isIntLike(p.Kind) && isIntLike(o.Kind) {
		return true
	}
	// Allow float literals and C float promotion
	if isFloatLike(p.Kind) && isFloatLike(o.Kind) {
		return true
	}
	return false
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

	// C-ABI Primitive Types (v0.7 FFI)
	TypeCChar      = &PrimitiveType{Kind: PrimCChar}
	TypeCUChar     = &PrimitiveType{Kind: PrimCUChar}
	TypeCShort     = &PrimitiveType{Kind: PrimCShort}
	TypeCUShort    = &PrimitiveType{Kind: PrimCUShort}
	TypeCInt       = &PrimitiveType{Kind: PrimCInt}
	TypeCUInt      = &PrimitiveType{Kind: PrimCUInt}
	TypeCLong      = &PrimitiveType{Kind: PrimCLong}
	TypeCULong     = &PrimitiveType{Kind: PrimCULong}
	TypeCLongLong  = &PrimitiveType{Kind: PrimCLongLong}
	TypeCULongLong = &PrimitiveType{Kind: PrimCULongLong}
	TypeCSize      = &PrimitiveType{Kind: PrimCSize}
	TypeCSSize     = &PrimitiveType{Kind: PrimCSSize}
	TypeCFloat     = &PrimitiveType{Kind: PrimCFloat}
	TypeCDouble    = &PrimitiveType{Kind: PrimCDouble}
	TypeVoidPtr    = &PrimitiveType{Kind: PrimVoidPtr}
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
	if pt, ok := other.(*PrimitiveType); ok && pt.Kind == PrimVoidPtr {
		return true
	}
	if other.Equals(TypeString) && (r.Inner.Equals(TypeCChar) || r.Inner.Equals(TypeU8)) {
		return true
	}
	o, ok := other.(*RefType)
	if !ok {
		return false
	}
	if o.Inner.Equals(TypeVoidPtr) || r.Inner.Equals(TypeVoidPtr) {
		return true
	}
	return r.IsMut == o.IsMut && ExactEquals(r.Inner, o.Inner)
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
	IsExtern      bool
	CName         string
	IsUnsafe      bool
	IsExport      bool
	ExportName    string
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
	IsOpaque        bool // `@[opaque]`: C struct known only by pointer
	IsCLayout       bool // `@[c_layout]`: struct matching exact C memory layout
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
