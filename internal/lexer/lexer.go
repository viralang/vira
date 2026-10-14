package lexer

import "github.com/vexlang/vex/internal/token"

type Lexer struct {
	src         string
	pos         uint32
	inString    bool
	pendingBeg  bool     // true right after hitting `${` inside a string
	interpDepth uint8    // current nesting depth of `${ ... }`
	braceStack  [8]uint8 // tracks inner `{` braces inside each `${ ... }` level
}

func New(src string) *Lexer {
	return &Lexer{src: src}
}

func (l *Lexer) Next() token.Token {
	// 1. If we just stopped a string segment at `${`, emit TokInterpBeg now.
	if l.pendingBeg {
		l.pendingBeg = false
		start := l.pos
		l.pos += 2 // consume "${"
		if l.interpDepth < uint8(len(l.braceStack)) {
			l.braceStack[l.interpDepth] = 0
			l.interpDepth++
		}
		return l.makeTok(token.TokInterpBeg, start, l.pos)
	}

	// 2. If resuming string scan after `}` closed an interpolation hole.
	if l.inString {
		return l.scanStringSegment(false)
	}

	l.skipWhitespaceAndComments()

	if int(l.pos) >= len(l.src) {
		return l.makeTok(token.TokEOF, l.pos, l.pos)
	}

	start := l.pos
	ch := l.src[l.pos]

	// 3. Identifiers and Keywords
	if isLetter(ch) {
		l.pos++
		for int(l.pos) < len(l.src) && isIdentPart(l.src[l.pos]) {
			l.pos++
		}
		lit := l.src[start:l.pos]
		return l.makeTok(token.LookupIdent(lit), start, l.pos)
	}

	// 4. Numeric Literals (integers, floats, `_` digit separators, and precision suffixes)
	if isDigit(ch) {
		kind := token.TokInt
		l.pos++
		for int(l.pos) < len(l.src) {
			c := l.src[l.pos]
			if isDigit(c) || c == '_' {
				l.pos++
			} else if c == '.' && l.peek(1) != '.' && kind == token.TokInt {
				kind = token.TokFloat
				l.pos++
			} else {
				break
			}
		}
		// Check for optional float precision suffix (e.g. 1.0f32, 0.5f16, 2.0bf16)
		if int(l.pos) < len(l.src) {
			rem := l.src[l.pos:]
			if len(rem) >= 4 && rem[:4] == "bf16" {
				l.pos += 4
				kind = token.TokFloat
			} else if len(rem) >= 3 && (rem[:3] == "f32" || rem[:3] == "f16" || rem[:3] == "f64") {
				l.pos += 3
				kind = token.TokFloat
			}
		}
		return l.makeTok(kind, start, l.pos)
	}

	// 5. Single-quoted string literals with `${...}` interpolation
	if ch == '\'' {
		return l.scanStringSegment(true)
	}

	// 6. Operators and Delimiters
	l.pos++
	switch ch {
	case '@':
		return l.makeTok(token.TokAt, start, l.pos)
	case '<':
		if l.match('-') {
			return l.makeTok(token.TokChanSend, start, l.pos)
		}
		if l.match('=') {
			return l.makeTok(token.TokLte, start, l.pos)
		}
		if l.match('<') {
			return l.makeTok(token.TokShl, start, l.pos)
		}
		return l.makeTok(token.TokLt, start, l.pos)
	case '>':
		if l.match('=') {
			return l.makeTok(token.TokGte, start, l.pos)
		}
		if l.match('>') {
			return l.makeTok(token.TokShr, start, l.pos)
		}
		return l.makeTok(token.TokGt, start, l.pos)
	case ':':
		if l.match('=') {
			return l.makeTok(token.TokDeclAssign, start, l.pos)
		}
		return l.makeTok(token.TokColon, start, l.pos)
	case '=':
		if l.match('=') {
			return l.makeTok(token.TokEq, start, l.pos)
		}
		return l.makeTok(token.TokAssign, start, l.pos)
	case '+':
		if l.match('=') {
			return l.makeTok(token.TokPlusAssign, start, l.pos)
		}
		return l.makeTok(token.TokPlus, start, l.pos)
	case '-':
		if l.match('=') {
			return l.makeTok(token.TokSubAssign, start, l.pos)
		}
		if l.match('>') {
			return l.makeTok(token.TokArrow, start, l.pos)
		}
		return l.makeTok(token.TokMinus, start, l.pos)
	case '!':
		if l.match('=') {
			return l.makeTok(token.TokNeq, start, l.pos)
		}
		return l.makeTok(token.TokBang, start, l.pos)
	case '&':
		if l.match('&') {
			return l.makeTok(token.TokAnd, start, l.pos)
		}
		return l.makeTok(token.TokAmp, start, l.pos)
	case '|':
		if l.match('|') {
			return l.makeTok(token.TokOrLogical, start, l.pos)
		}
		return l.makeTok(token.TokPipe, start, l.pos)
	case '*':
		if l.match('=') {
			return l.makeTok(token.TokMulAssign, start, l.pos)
		}
		return l.makeTok(token.TokStar, start, l.pos)
	case '/':
		if l.match('=') {
			return l.makeTok(token.TokDivAssign, start, l.pos)
		}
		return l.makeTok(token.TokSlash, start, l.pos)
	case '%':
		if l.match('=') {
			return l.makeTok(token.TokModAssign, start, l.pos)
		}
		return l.makeTok(token.TokPercent, start, l.pos)
	case '^':
		return l.makeTok(token.TokCaret, start, l.pos)
	case '?':
		return l.makeTok(token.TokQuestion, start, l.pos)
	case '(':
		return l.makeTok(token.TokLParen, start, l.pos)
	case ')':
		return l.makeTok(token.TokRParen, start, l.pos)
	case '[':
		return l.makeTok(token.TokLBracket, start, l.pos)
	case ']':
		return l.makeTok(token.TokRBracket, start, l.pos)
	case ',':
		return l.makeTok(token.TokComma, start, l.pos)
	case '.':
		return l.makeTok(token.TokDot, start, l.pos)
	case ';':
		return l.makeTok(token.TokSemicolon, start, l.pos)
	case '{':
		if l.interpDepth > 0 {
			l.braceStack[l.interpDepth-1]++
		}
		return l.makeTok(token.TokLBrace, start, l.pos)
	case '}':
		if l.interpDepth > 0 {
			top := l.interpDepth - 1
			if l.braceStack[top] == 0 {
				l.interpDepth--
				l.inString = true
				return l.makeTok(token.TokInterpEnd, start, l.pos)
			}
			l.braceStack[top]--
		}
		return l.makeTok(token.TokRBrace, start, l.pos)
	default:
		return l.makeTok(token.TokIllegal, start, l.pos)
	}
}

// scanStringSegment scans either a complete string or a segment up to `${` or `'`.
func (l *Lexer) scanStringSegment(isFirst bool) token.Token {
	start := l.pos
	if isFirst {
		l.pos++ // skip opening '\''
	}

	for int(l.pos) < len(l.src) {
		ch := l.src[l.pos]
		if ch == '\\' {
			l.pos += 2 // skip escaped character
			continue
		}
		if ch == '$' && l.peek(1) == '{' {
			l.inString = false
			l.pendingBeg = true
			kind := token.TokStrPart
			if isFirst {
				kind = token.TokStrStart
			}
			return l.makeTok(kind, start, l.pos)
		}
		if ch == '\'' {
			l.pos++ // consume closing '\''
			l.inString = false
			kind := token.TokStrEnd
			if isFirst {
				kind = token.TokString
			}
			return l.makeTok(kind, start, l.pos)
		}
		l.pos++
	}

	l.inString = false
	return l.makeTok(token.TokIllegal, start, l.pos)
}

func (l *Lexer) skipWhitespaceAndComments() {
	for int(l.pos) < len(l.src) {
		ch := l.src[l.pos]
		if ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' {
			l.pos++
			continue
		}
		// Single-line // comments
		if ch == '/' && l.peek(1) == '/' {
			l.pos += 2
			for int(l.pos) < len(l.src) && l.src[l.pos] != '\n' {
				l.pos++
			}
			continue
		}
		// Multi-line /* ... */ comments
		if ch == '/' && l.peek(1) == '*' {
			l.pos += 2
			for int(l.pos)+1 < len(l.src) {
				if l.src[l.pos] == '*' && l.src[l.pos+1] == '/' {
					l.pos += 2
					break
				}
				l.pos++
			}
			continue
		}
		break
	}
}

func (l *Lexer) match(expected byte) bool {
	if int(l.pos) < len(l.src) && l.src[l.pos] == expected {
		l.pos++
		return true
	}
	return false
}

func (l *Lexer) peek(offset uint32) byte {
	idx := l.pos + offset
	if int(idx) < len(l.src) {
		return l.src[idx]
	}
	return 0
}

func (l *Lexer) makeTok(kind token.Kind, start, end uint32) token.Token {
	return token.Token{
		Kind: kind,
		Span: token.Span{Start: start, End: end},
	}
}

func isLetter(ch byte) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'
}

func isDigit(ch byte) bool {
	return ch >= '0' && ch <= '9'
}

func isIdentPart(ch byte) bool {
	return isLetter(ch) || isDigit(ch)
}
