package token

// Span represents a zero-allocation byte range in a source file.
type Span struct {
	Start uint32
	End   uint32
}

// Token is a 12-byte value type. No heap strings are allocated during lexing.
type Token struct {
	Kind Kind
	Span Span
}

// Lexeme slices the original source string on demand without heap allocation.
func (t Token) Lexeme(src string) string {
	if int(t.Span.Start) > len(src) || int(t.Span.End) > len(src) || t.Span.Start > t.Span.End {
		return ""
	}
	return src[t.Span.Start:t.Span.End]
}

type Kind uint8

const (
	TokIllegal Kind = iota
	TokEOF
	TokComment

	// Identifiers & Literals
	TokIdent     // main, Invoice, mark_paid
	TokInt       // 42, 0xFF, 1_000_000
	TokFloat     // 3.14
	TokString    // 'plain string without interpolation'
	TokStrStart  // leading segment of interpolated string: 'Invoice #
	TokStrPart   // middle segment of interpolated string
	TokStrEnd    // trailing segment of interpolated string
	TokInterpBeg // ${
	TokInterpEnd // } closing an interpolation expression

	// Keywords
	TokModule
	TokImport
	TokPub
	TokMut
	TokStruct
	TokInterface
	TokEnum
	TokType
	TokFn
	TokReturn
	TokIf
	TokElse
	TokSwitch
	TokCase
	TokDefault
	TokMatch
	TokFor
	TokIn
	TokOr
	TokIs
	TokAs
	TokTrue
	TokFalse
	TokArena

	TokBreak
	TokContinue

	// AI & Concurrency Keywords
	TokF32
	TokF16
	TokBF16
	TokTensor
	TokActor
	TokSpawn
	TokSupervise

	// Operators & Assignments
	TokDeclAssign // :=
	TokAssign     // =
	TokPlusAssign // +=
	TokSubAssign  // -=
	TokMulAssign  // *=
	TokDivAssign  // /=
	TokModAssign  // %=
	TokPlus       // +
	TokMinus      // -
	TokStar       // *
	TokSlash      // /
	TokPercent    // %
	TokAmp        // &
	TokPipe       // |
	TokCaret      // ^
	TokShl        // <<
	TokShr        // >>
	TokBang       // ! (Result type prefix, error propagation suffix, or logical NOT)
	TokQuestion   // ? (Option type prefix or propagation suffix)
	TokEq         // ==
	TokNeq        // !=
	TokLt         // <
	TokLte        // <=
	TokGt         // >
	TokGte        // >=
	TokAnd        // &&
	TokOrLogical  // ||
	TokArrow      // ->
	TokAt         // @ (Matrix multiplication operator)
	TokChanSend   // <- (Actor message send / future receive)

	// Delimiters
	TokLParen    // (
	TokRParen    // )
	TokLBrace    // {
	TokRBrace    // }
	TokLBracket  // [
	TokRBracket  // ]
	TokComma     // ,
	TokDot       // .
	TokColon     // :
	TokSemicolon // ;
)

var keywords = map[string]Kind{
	"module":    TokModule,
	"import":    TokImport,
	"pub":       TokPub,
	"mut":       TokMut,
	"struct":    TokStruct,
	"interface": TokInterface,
	"enum":      TokEnum,
	"type":      TokType,
	"fn":        TokFn,
	"return":    TokReturn,
	"break":     TokBreak,
	"continue":  TokContinue,
	"if":        TokIf,
	"else":      TokElse,
	"switch":    TokSwitch,
	"case":      TokCase,
	"default":   TokDefault,
	"match":     TokMatch,
	"for":       TokFor,
	"in":        TokIn,
	"or":        TokOr,
	"is":        TokIs,
	"as":        TokAs,
	"true":      TokTrue,
	"false":     TokFalse,
	"arena":     TokArena,
	"f32":       TokF32,
	"f16":       TokF16,
	"bf16":      TokBF16,
	"tensor":    TokTensor,
	"actor":     TokActor,
	"spawn":     TokSpawn,
	"supervise": TokSupervise,
}

// LookupIdent maps an identifier slice to a keyword Kind or TokIdent.
func LookupIdent(ident string) Kind {
	if k, ok := keywords[ident]; ok {
		return k
	}
	return TokIdent
}

// Precedence returns the binary/postfix operator precedence for the Pratt parser.
func (k Kind) Precedence() int {
	switch k {
	case TokOrLogical:
		return 1
	case TokAnd:
		return 2
	case TokEq, TokNeq, TokLt, TokLte, TokGt, TokGte, TokIn, TokIs:
		return 3
	case TokPipe, TokCaret, TokAmp:
		return 4
	case TokShl, TokShr:
		return 5
	case TokPlus, TokMinus:
		return 6
	case TokStar, TokSlash, TokPercent, TokAt:
		return 7
	case TokLParen, TokLBracket, TokDot, TokChanSend, TokBang, TokQuestion, TokOr:
		return 8
	default:
		return 0
	}
}

func (k Kind) String() string {
	names := [...]string{
		TokIllegal:    "ILLEGAL",
		TokEOF:        "EOF",
		TokComment:    "COMMENT",
		TokIdent:      "IDENT",
		TokInt:        "INT",
		TokFloat:      "FLOAT",
		TokString:     "STRING",
		TokStrStart:   "STR_START",
		TokStrPart:    "STR_PART",
		TokStrEnd:     "STR_END",
		TokInterpBeg:  "INTERP_BEG(${)",
		TokInterpEnd:  "INTERP_END(})",
		TokModule:     "module",
		TokImport:     "import",
		TokPub:        "pub",
		TokMut:        "mut",
		TokStruct:     "struct",
		TokInterface:  "interface",
		TokEnum:       "enum",
		TokType:       "type",
		TokFn:         "fn",
		TokReturn:     "return",
		TokBreak:      "break",
		TokContinue:   "continue",
		TokIf:         "if",
		TokElse:       "else",
		TokSwitch:     "switch",
		TokCase:       "case",
		TokDefault:    "default",
		TokMatch:      "match",
		TokFor:        "for",
		TokIn:         "in",
		TokOr:         "or",
		TokIs:         "is",
		TokAs:         "as",
		TokTrue:       "true",
		TokFalse:      "false",
		TokArena:      "arena",
		TokF32:        "f32",
		TokF16:        "f16",
		TokBF16:       "bf16",
		TokTensor:     "tensor",
		TokActor:      "actor",
		TokSpawn:      "spawn",
		TokSupervise:  "supervise",
		TokDeclAssign: ":=",
		TokAssign:     "=",
		TokPlusAssign: "+=",
		TokSubAssign:  "-=",
		TokMulAssign:  "*=",
		TokDivAssign:  "/=",
		TokModAssign:  "%=",
		TokPlus:       "+",
		TokMinus:      "-",
		TokStar:       "*",
		TokSlash:      "/",
		TokPercent:    "%",
		TokAmp:        "&",
		TokPipe:       "|",
		TokCaret:      "^",
		TokShl:        "<<",
		TokShr:        ">>",
		TokBang:       "!",
		TokQuestion:   "?",
		TokEq:         "==",
		TokNeq:        "!=",
		TokLt:         "<",
		TokLte:        "<=",
		TokGt:         ">",
		TokGte:        ">=",
		TokAnd:        "&&",
		TokOrLogical:  "||",
		TokArrow:      "->",
		TokAt:         "@",
		TokChanSend:   "<-",
		TokLParen:     "(",
		TokRParen:     ")",
		TokLBrace:     "{",
		TokRBrace:     "}",
		TokLBracket:   "[",
		TokRBracket:   "]",
		TokComma:      ",",
		TokDot:        ".",
		TokColon:      ":",
		TokSemicolon:  ";",
	}
	if int(k) < len(names) {
		return names[k]
	}
	return "UNKNOWN"
}
