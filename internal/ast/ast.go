package ast

import "github.com/viralang/vira/internal/token"

// Node is the base interface for all AST elements.
type Node interface {
	Span() token.Span
}

type Decl interface {
	Node
	declNode()
}

type Stmt interface {
	Node
	stmtNode()
}

type Expr interface {
	Node
	exprNode()
}

type TypeExpr interface {
	Node
	typeExprNode()
}

// Ident represents a source identifier slice.
type Ident struct {
	Name    string
	NamePos token.Span
}

func (i Ident) Span() token.Span { return i.NamePos }

// AccessMode encodes the 4-quadrant struct encapsulation matrix.
type AccessMode uint8

const (
	AccessPrivateImm AccessMode = iota // default: private & immutable
	AccessPrivateMut                   // `mut:` block: private & mutable
	AccessPubImm                       // `pub:` block: public read-only
	AccessPubMut                       // `pub mut:` block: public & mutable
)

func (a AccessMode) String() string {
	switch a {
	case AccessPrivateMut:
		return "mut"
	case AccessPubImm:
		return "pub"
	case AccessPubMut:
		return "pub mut"
	default:
		return "private"
	}
}

// File represents a single parsed .vir source file.
type File struct {
	Module   Ident
	Imports  []Ident
	Decls    []Decl
	FileSpan token.Span
}

func (f *File) Span() token.Span { return f.FileSpan }

// --- TYPE EXPRESSIONS ---

type IdentType struct {
	Name       Ident
	TypeParams []TypeExpr // e.g., List[Invoice], tensor[f32]
}

func (t *IdentType) Span() token.Span { return t.Name.Span() }
func (t *IdentType) typeExprNode()    {}

// ResultType represents `!T` (or `!` for void Result).
type ResultType struct {
	BangPos token.Span
	Inner   TypeExpr // nil if bare `!`
}

func (t *ResultType) Span() token.Span { return t.BangPos }
func (t *ResultType) typeExprNode()    {}

// OptionType represents `?T`.
type OptionType struct {
	QuestionPos token.Span
	Inner       TypeExpr
}

func (t *OptionType) Span() token.Span { return t.QuestionPos }
func (t *OptionType) typeExprNode()    {}

// RefType represents `&T` or `&mut T`.
type RefType struct {
	AmpPos token.Span
	IsMut  bool
	Inner  TypeExpr
}

func (t *RefType) Span() token.Span { return t.AmpPos }
func (t *RefType) typeExprNode()    {}

// ArrayType represents fixed-size `[N]T` (when Len != nil) or dynamic slice `[]T` (when Len == nil).
type ArrayType struct {
	LBracketPos token.Span
	Len         Expr     // nil for slice `[]T`, constant integer expr for `[N]T`
	Elem        TypeExpr // element type
}

func (t *ArrayType) Span() token.Span { return t.LBracketPos }
func (t *ArrayType) typeExprNode()    {}

// --- OOP & TOP-LEVEL DECLARATIONS ---

// StructField represents a single field inside one of the 4 access quadrants.
type StructField struct {
	Name        Ident
	Type        TypeExpr
	Access      AccessMode
	DefaultExpr Expr // optional default value (`= 0`)
	FieldSpan   token.Span
}

// StructDecl models a Vira struct or actor with composition (Embeds) and 4-quadrant fields.
type StructDecl struct {
	IsPub      bool
	IsActor    bool // true if declared with `actor Name { ... }`
	Name       Ident
	TypeParams []Ident       // generic parameters `[T]`
	Embeds     []TypeExpr    // anonymous embedded structs (e.g., `AuditMeta`)
	Fields     []StructField // fields tagged with their AccessMode quadrant
	DeclSpan   token.Span
}

func (d *StructDecl) Span() token.Span { return d.DeclSpan }
func (d *StructDecl) declNode()        {}

// InterfaceMethod represents a method signature inside an `interface` block.
type InterfaceMethod struct {
	Name          Ident
	IsMutReceiver bool // true if declared inside a `mut:` section of the interface
	Params        []Param
	ReturnType    TypeExpr
	MethodSpan    token.Span
}

// InterfaceField represents a V-style required field inside an `interface` block.
type InterfaceField struct {
	Name      Ident
	Type      TypeExpr
	IsMut     bool
	FieldSpan token.Span
}

// InterfaceDecl models a structural interface supporting both fields and methods.
type InterfaceDecl struct {
	IsPub     bool
	Name      Ident
	Embeds    []TypeExpr
	Fields    []InterfaceField
	Methods   []InterfaceMethod
	DeclSpan  token.Span
}

func (d *InterfaceDecl) Span() token.Span { return d.DeclSpan }
func (d *InterfaceDecl) declNode()        {}

// Receiver represents `(inv Invoice)` or `(mut inv Invoice)` on a method.
type Receiver struct {
	Name     Ident
	Type     TypeExpr
	IsMut    bool
	RecvSpan token.Span
}

type Param struct {
	Name      Ident
	Type      TypeExpr
	IsMut     bool
	ParamSpan token.Span
}

// FnDecl models free functions, receiver methods, and static type methods (`Invoice.new`).
type FnDecl struct {
	IsPub      bool
	Receiver   *Receiver // non-nil for instance methods: `fn (mut inv Invoice) mark_paid()`
	StaticType *Ident    // non-nil for static type factories: `fn Invoice.new(...)`
	Name       Ident
	TypeParams []Ident
	Params     []Param
	ReturnType TypeExpr // nil means void
	Body       *BlockStmt
	DeclSpan   token.Span
}

func (d *FnDecl) Span() token.Span { return d.DeclSpan }
func (d *FnDecl) declNode()        {}

// --- STATEMENTS ---

type BlockStmt struct {
	Stmts     []Stmt
	BlockSpan token.Span
}

func (s *BlockStmt) Span() token.Span { return s.BlockSpan }
func (s *BlockStmt) stmtNode()        {}

type VarDeclStmt struct {
	Name     Ident
	IsMut    bool
	Value    Expr
	StmtSpan token.Span
}

func (s *VarDeclStmt) Span() token.Span { return s.StmtSpan }
func (s *VarDeclStmt) stmtNode()        {}

type AssignStmt struct {
	Target   Expr // IdentExpr or SelectorExpr (`inv.is_paid = true`)
	Op       token.Kind
	Value    Expr
	StmtSpan token.Span
}

func (s *AssignStmt) Span() token.Span { return s.StmtSpan }
func (s *AssignStmt) stmtNode()        {}

type ReturnStmt struct {
	Value    Expr
	StmtSpan token.Span
}

func (s *ReturnStmt) Span() token.Span { return s.StmtSpan }
func (s *ReturnStmt) stmtNode()        {}

type ExprStmt struct {
	Expr Expr
}

func (s *ExprStmt) Span() token.Span { return s.Expr.Span() }
func (s *ExprStmt) stmtNode()        {}

// IfStmt represents `if cond { ... } else { ... }`.
type IfStmt struct {
	Cond     Expr
	Then     *BlockStmt
	Else     *BlockStmt // optional
	StmtSpan token.Span
}

func (s *IfStmt) Span() token.Span { return s.StmtSpan }
func (s *IfStmt) stmtNode()        {}

// ForStmt represents `for cond { ... }`, `for { ... }`, or `for init; cond; post { ... }`.
type ForStmt struct {
	Init     Stmt // optional initialization
	Cond     Expr // optional condition
	Post     Stmt // optional post-statement
	Body     *BlockStmt
	StmtSpan token.Span
}

func (s *ForStmt) Span() token.Span { return s.StmtSpan }
func (s *ForStmt) stmtNode()        {}

// ForInStmt represents `for item in items { ... }`.
type ForInStmt struct {
	Var      Ident
	Iter     Expr
	Body     *BlockStmt
	StmtSpan token.Span
}

func (s *ForInStmt) Span() token.Span { return s.StmtSpan }
func (s *ForInStmt) stmtNode()        {}

// BranchStmt represents `break` or `continue`.
type BranchStmt struct {
	Tok      token.Kind
	StmtSpan token.Span
}

func (s *BranchStmt) Span() token.Span { return s.StmtSpan }
func (s *BranchStmt) stmtNode()        {}

// ArenaStmt represents a lexical `arena { ... }` slab allocation block.
type ArenaStmt struct {
	ArenaVar string // synthetic C variable name (e.g., "_vira_arena_1")
	Body     *BlockStmt
	StmtSpan token.Span
}

func (s *ArenaStmt) Span() token.Span { return s.StmtSpan }
func (s *ArenaStmt) stmtNode()        {}

// CleanupKind classifies how `cgen` lowers a synthetic scope-exit destructor.
type CleanupKind uint8

const (
	CleanupString    CleanupKind = iota // emits `vira_str_free(&var);`
	CleanupStruct                       // emits `Module__Struct_autofree(&var);`
	CleanupInterface                    // emits `var._vtable->_autofree(var._object);`
	CleanupArena                        // emits `vira_arena_destroy(&_arena_1);`
	CleanupTensor                       // emits `vira_tensor_f32_free(&var);`
	CleanupActor                        // emits `Module__Actor_autofree(&var);`
	CleanupFuture                       // emits `vira_future_free(&var);`
	CleanupSlice                        // emits slice destructor
)

func (k CleanupKind) String() string {
	switch k {
	case CleanupString:
		return "vira_str_free"
	case CleanupStruct:
		return "struct_autofree"
	case CleanupInterface:
		return "vtable_autofree"
	case CleanupArena:
		return "vira_arena_destroy"
	case CleanupTensor:
		return "vira_tensor_f32_free"
	case CleanupActor:
		return "actor_autofree"
	case CleanupFuture:
		return "vira_future_free"
	case CleanupSlice:
		return "slice_autofree"
	default:
		return "unknown_free"
	}
}

// DeferFreeStmt is a synthetic AST node injected by `internal/autofree` prior to scope exits.
type DeferFreeStmt struct {
	VarName  string
	TypeName string
	Kind     CleanupKind
	Reason   string // diagnostic explanation (e.g. "scope exit", "early return in or-block")
	StmtSpan token.Span
}

func (s *DeferFreeStmt) Span() token.Span { return s.StmtSpan }
func (s *DeferFreeStmt) stmtNode()        {}

// --- EXPRESSIONS ---

type IdentExpr struct {
	Ident Ident
}

func (e *IdentExpr) Span() token.Span { return e.Ident.Span() }
func (e *IdentExpr) exprNode()        {}

type LiteralExpr struct {
	Kind    token.Kind
	Value   string
	LitSpan token.Span
}

func (e *LiteralExpr) Span() token.Span { return e.LitSpan }
func (e *LiteralExpr) exprNode()        {}

// StringInterpExpr represents `'Invoice #${inv.id}: ${inv.amount_cents}'`.
type StringInterpExpr struct {
	Segments   []string // static string parts
	Exprs      []Expr   // interpolated expressions between segments
	InterpSpan token.Span
}

func (e *StringInterpExpr) Span() token.Span { return e.InterpSpan }
func (e *StringInterpExpr) exprNode()        {}

// SelectorExpr represents field access, method selection, or actor message send (`inv.id`, `worker <- predict`).
type SelectorExpr struct {
	Left        Expr
	Field       Ident
	IsActorSend bool // true when invoked via `actor <- method(...)`
	SelSpan     token.Span
}

func (e *SelectorExpr) Span() token.Span { return e.SelSpan }
func (e *SelectorExpr) exprNode()        {}

// StructInitField represents `id: 'inv_01'` or embedded `AuditMeta: AuditMeta{...}`.
type StructInitField struct {
	Name      Ident
	Value     Expr
	FieldSpan token.Span
}

// StructInitExpr represents `Invoice{ id: '123', amount_cents: 5000 }`.
type StructInitExpr struct {
	Type     TypeExpr
	Fields   []StructInitField
	InitSpan token.Span
}

func (e *StructInitExpr) Span() token.Span { return e.InitSpan }
func (e *StructInitExpr) exprNode()        {}

// CallArg represents an argument passed to a function, tracking explicit `mut` at the call site.
type CallArg struct {
	IsMut   bool
	Expr    Expr
	ArgSpan token.Span
}

// CallExpr represents `fetch_user(42)`, `inv.mark_paid()`, `worker <- predict()`, or `increment(mut count)`.
type CallExpr struct {
	Callee    Expr
	Args      []CallArg
	OrBlock   *BlockStmt // optional `or { ... }` attached to `!T` / `?T` calls
	Propagate bool       // true if suffixed with `!` or `?` propagation operator
	CallSpan  token.Span
}

func (e *CallExpr) Span() token.Span { return e.CallSpan }
func (e *CallExpr) exprNode()        {}

type BinaryExpr struct {
	Left    Expr
	Op      token.Kind
	Right   Expr
	BinSpan token.Span
}

func (e *BinaryExpr) Span() token.Span { return e.BinSpan }
func (e *BinaryExpr) exprNode()        {}

type UnaryExpr struct {
	Op        token.Kind
	Expr      Expr
	UnarySpan token.Span
}

func (e *UnaryExpr) Span() token.Span { return e.UnarySpan }
func (e *UnaryExpr) exprNode()        {}

// IndexExpr represents `arr[index]`.
type IndexExpr struct {
	Left      Expr
	Index     Expr
	IndexSpan token.Span
}

func (e *IndexExpr) Span() token.Span { return e.IndexSpan }
func (e *IndexExpr) exprNode()        {}

// ArrayLitExpr represents `[1, 2, 3]` or typed `[]int{1, 2, 3}` or `[4]int{1, 2, 3, 4}`.
type ArrayLitExpr struct {
	Type    TypeExpr // optional explicit type (nil if inferred: `[1, 2, 3]`)
	Elems   []Expr
	LitSpan token.Span
}

func (e *ArrayLitExpr) Span() token.Span { return e.LitSpan }
func (e *ArrayLitExpr) exprNode()        {}

