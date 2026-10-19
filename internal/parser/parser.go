package parser

import (
	"fmt"
	"strings"

	"github.com/viralang/vira/internal/ast"
	"github.com/viralang/vira/internal/lexer"
	"github.com/viralang/vira/internal/token"
)

type ParseError struct {
	Span token.Span
	Msg  string
}

func (e ParseError) Error() string {
	return fmt.Sprintf("[%d..%d] parse error: %s", e.Span.Start, e.Span.End, e.Msg)
}

type Parser struct {
	src             string
	lex             *lexer.Lexer
	cur             token.Token
	peek            token.Token
	errors          []ParseError
	allowStructInit bool
}

func New(src string) *Parser {
	l := lexer.New(src)
	p := &Parser{
		src:             src,
		lex:             l,
		allowStructInit: true,
	}
	// Prime two-token lookahead buffer
	p.cur = p.lex.Next()
	p.peek = p.lex.Next()
	return p
}

func (p *Parser) Errors() []ParseError {
	return p.errors
}

func (p *Parser) advance() token.Token {
	prev := p.cur
	p.cur = p.peek
	p.peek = p.lex.Next()
	return prev
}

func (p *Parser) expect(k token.Kind) token.Token {
	if p.cur.Kind == k {
		return p.advance()
	}
	p.errorf(p.cur.Span, "expected token %s, found %s (%q)", k, p.cur.Kind, p.cur.Lexeme(p.src))
	return p.advance()
}

func (p *Parser) errorf(span token.Span, format string, args ...any) {
	p.errors = append(p.errors, ParseError{
		Span: span,
		Msg:  fmt.Sprintf(format, args...),
	})
}

func (p *Parser) parseIdent() ast.Ident {
	tok := p.expect(token.TokIdent)
	return ast.Ident{
		Name:    tok.Lexeme(p.src),
		NamePos: tok.Span,
	}
}

// hasNewlineBetween checks if a line break exists between two token spans in O(gap) with zero allocations.
func (p *Parser) hasNewlineBetween(a, b token.Span) bool {
	if a.End >= b.Start || int(b.Start) > len(p.src) {
		return false
	}
	return strings.IndexByte(p.src[a.End:b.Start], '\n') >= 0
}

// --- FILE & TOP-LEVEL DECLARATIONS ---

func (p *Parser) ParseFile() *ast.File {
	file := &ast.File{
		FileSpan: token.Span{Start: p.cur.Span.Start},
	}

	if p.cur.Kind == token.TokModule {
		p.advance()
		file.Module = p.parseIdent()
	}

	for p.cur.Kind == token.TokImport {
		p.advance()
		if p.cur.Kind == token.TokLParen {
			p.advance()
			for p.cur.Kind != token.TokRParen && p.cur.Kind != token.TokEOF {
				if p.cur.Kind == token.TokString {
					tok := p.advance()
					raw := tok.Lexeme(p.src)
					file.Imports = append(file.Imports, ast.Ident{
						Name:    strings.Trim(raw, "'\""),
						NamePos: tok.Span,
					})
				} else if p.cur.Kind == token.TokIdent {
					file.Imports = append(file.Imports, p.parseIdent())
				} else {
					p.errorf(p.cur.Span, "expected import target identifier or string, got %s", p.cur.Kind)
					p.advance()
				}
			}
			p.expect(token.TokRParen)
		} else if p.cur.Kind == token.TokString {
			tok := p.advance()
			raw := tok.Lexeme(p.src)
			file.Imports = append(file.Imports, ast.Ident{
				Name:    strings.Trim(raw, "'\""),
				NamePos: tok.Span,
			})
		} else {
			file.Imports = append(file.Imports, p.parseIdent())
		}
	}

	for p.cur.Kind != token.TokEOF {
		if decl := p.parseDecl(); decl != nil {
			file.Decls = append(file.Decls, decl)
		}
	}

	file.FileSpan.End = p.cur.Span.End
	return file
}

func (p *Parser) parseDecl() ast.Decl {
	startSpan := p.cur.Span
	isPub := false
	if p.cur.Kind == token.TokPub {
		isPub = true
		p.advance()
	}

	switch p.cur.Kind {
	case token.TokStruct:
		return p.parseStructDecl(isPub, false, startSpan)
	case token.TokActor:
		return p.parseStructDecl(isPub, true, startSpan)
	case token.TokInterface:
		return p.parseInterfaceDecl(isPub, startSpan)
	case token.TokFn:
		return p.parseFnDecl(isPub, startSpan)
	default:
		p.errorf(p.cur.Span, "unexpected top-level token %s (%q)", p.cur.Kind, p.cur.Lexeme(p.src))
		p.advance()
		return nil
	}
}

// --- OOP: STRUCTS & ACTORS WITH 4-QUADRANT ACCESS & EMBEDDING ---

func (p *Parser) parseStructDecl(isPub bool, isActor bool, startSpan token.Span) *ast.StructDecl {
	if isActor {
		p.expect(token.TokActor)
	} else {
		p.expect(token.TokStruct)
	}
	name := p.parseIdent()
	typeParams := p.parseOptionalTypeParams()
	p.expect(token.TokLBrace)

	decl := &ast.StructDecl{
		IsPub:      isPub,
		IsActor:    isActor,
		Name:       name,
		TypeParams: typeParams,
	}

	mode := ast.AccessPrivateImm

	for p.cur.Kind != token.TokRBrace && p.cur.Kind != token.TokEOF {
		// 1. Check for 4-quadrant access section labels: `mut:`, `pub:`, `pub mut:`
		if p.cur.Kind == token.TokMut && p.peek.Kind == token.TokColon {
			p.advance() // consume `mut`
			p.advance() // consume `:`
			mode = ast.AccessPrivateMut
			continue
		}
		if p.cur.Kind == token.TokPub {
			p.advance() // consume `pub`
			if p.cur.Kind == token.TokMut {
				p.advance() // consume `mut`
				p.expect(token.TokColon)
				mode = ast.AccessPubMut
			} else {
				p.expect(token.TokColon)
				mode = ast.AccessPubImm
			}
			continue
		}

		// 2. Parse either an Embedded Struct Type or a Field Declaration
		firstIdent := p.parseIdent()

		// If a newline or `}` immediately follows the identifier (or `[T]` generic args),
		// it is an anonymous embedded struct (e.g. `AuditMeta`).
		if p.cur.Kind == token.TokLBracket || p.cur.Kind == token.TokRBrace || p.hasNewlineBetween(firstIdent.NamePos, p.cur.Span) {
			var tParams []ast.TypeExpr
			if p.cur.Kind == token.TokLBracket {
				tParams = p.parseTypeArgList()
			}
			decl.Embeds = append(decl.Embeds, &ast.IdentType{
				Name:       firstIdent,
				TypeParams: tParams,
			})
			continue
		}

		// Otherwise, `firstIdent` is a field name followed by its TypeExpr on the same line.
		fieldType := p.parseTypeExpr()
		var defaultExpr ast.Expr
		if p.cur.Kind == token.TokAssign {
			p.advance()
			defaultExpr = p.parseExpr(0)
		}

		endSpan := fieldType.Span().End
		if defaultExpr != nil {
			endSpan = defaultExpr.Span().End
		}

		decl.Fields = append(decl.Fields, ast.StructField{
			Name:        firstIdent,
			Type:        fieldType,
			Access:      mode,
			DefaultExpr: defaultExpr,
			FieldSpan:   token.Span{Start: firstIdent.NamePos.Start, End: endSpan},
		})
	}

	endTok := p.expect(token.TokRBrace)
	decl.DeclSpan = token.Span{Start: startSpan.Start, End: endTok.Span.End}
	return decl
}

// --- OOP: STRUCTURAL INTERFACES ---

func (p *Parser) parseInterfaceDecl(isPub bool, startSpan token.Span) *ast.InterfaceDecl {
	p.expect(token.TokInterface)
	name := p.parseIdent()
	p.expect(token.TokLBrace)

	decl := &ast.InterfaceDecl{
		IsPub: isPub,
		Name:  name,
	}

	isMutSection := false
	for p.cur.Kind != token.TokRBrace && p.cur.Kind != token.TokEOF {
		if p.cur.Kind == token.TokMut && p.peek.Kind == token.TokColon {
			p.advance()
			p.advance()
			isMutSection = true
			continue
		}

		memberName := p.parseIdent()
		// Method signature if followed by `(`
		if p.cur.Kind == token.TokLParen {
			params := p.parseParamList()
			var retType ast.TypeExpr
			if !p.hasNewlineBetween(p.cur.Span, p.cur.Span) && p.isTypeStart(p.cur.Kind) && p.cur.Kind != token.TokRBrace {
				retType = p.parseTypeExpr()
			}
			decl.Methods = append(decl.Methods, ast.InterfaceMethod{
				Name:          memberName,
				IsMutReceiver: isMutSection,
				Params:        params,
				ReturnType:    retType,
				MethodSpan:    memberName.NamePos,
			})
		} else {
			// Required interface field: `id string`
			fieldType := p.parseTypeExpr()
			decl.Fields = append(decl.Fields, ast.InterfaceField{
				Name:      memberName,
				Type:      fieldType,
				IsMut:     isMutSection,
				FieldSpan: token.Span{Start: memberName.NamePos.Start, End: fieldType.Span().End},
			})
		}
	}

	endTok := p.expect(token.TokRBrace)
	decl.DeclSpan = token.Span{Start: startSpan.Start, End: endTok.Span.End}
	return decl
}

// --- FUNCTIONS, RECEIVER METHODS & STATIC TYPE FACTORIES ---

func (p *Parser) parseFnDecl(isPub bool, startSpan token.Span) *ast.FnDecl {
	p.expect(token.TokFn)

	decl := &ast.FnDecl{IsPub: isPub}

	// 1. Optional Receiver: `fn (inv Invoice)` or `fn (mut inv Invoice)`
	if p.cur.Kind == token.TokLParen {
		lparen := p.advance()
		isMut := false
		if p.cur.Kind == token.TokMut {
			isMut = true
			p.advance()
		}
		recvName := p.parseIdent()
		recvType := p.parseTypeExpr()
		rparen := p.expect(token.TokRParen)
		decl.Receiver = &ast.Receiver{
			Name:     recvName,
			Type:     recvType,
			IsMut:    isMut,
			RecvSpan: token.Span{Start: lparen.Span.Start, End: rparen.Span.End},
		}
	}

	// 2. Function Name or Static Type Method (`fn Invoice.new(...)`)
	firstIdent := p.parseIdent()
	if decl.Receiver == nil && p.cur.Kind == token.TokDot {
		p.advance() // consume `.`
		decl.StaticType = &firstIdent
		decl.Name = p.parseIdent()
	} else {
		decl.Name = firstIdent
	}

	decl.TypeParams = p.parseOptionalTypeParams()
	decl.Params = p.parseParamList()

	// 3. Optional Return Type (`!User`, `?int`, `Invoice`, etc.) if not `{`
	if p.cur.Kind != token.TokLBrace && p.isTypeStart(p.cur.Kind) {
		decl.ReturnType = p.parseTypeExpr()
	}

	decl.Body = p.parseBlock()
	decl.DeclSpan = token.Span{Start: startSpan.Start, End: decl.Body.BlockSpan.End}
	return decl
}

func (p *Parser) parseParamList() []ast.Param {
	p.expect(token.TokLParen)
	var params []ast.Param
	for p.cur.Kind != token.TokRParen && p.cur.Kind != token.TokEOF {
		start := p.cur.Span.Start
		isMut := false
		if p.cur.Kind == token.TokMut {
			isMut = true
			p.advance()
		}
		name := p.parseIdent()
		typ := p.parseTypeExpr()
		params = append(params, ast.Param{
			Name:      name,
			Type:      typ,
			IsMut:     isMut,
			ParamSpan: token.Span{Start: start, End: typ.Span().End},
		})
		if p.cur.Kind == token.TokComma {
			p.advance()
		}
	}
	p.expect(token.TokRParen)
	return params
}

// --- TYPE EXPRESSIONS ---

func (p *Parser) isTypeStart(k token.Kind) bool {
	return k == token.TokIdent || k == token.TokBang || k == token.TokQuestion || k == token.TokAmp ||
		k == token.TokF32 || k == token.TokF16 || k == token.TokBF16 || k == token.TokTensor || k == token.TokLBracket
}

func (p *Parser) parseTypeExpr() ast.TypeExpr {
	switch p.cur.Kind {
	case token.TokLBracket:
		lb := p.advance()
		var lenExpr ast.Expr
		if p.cur.Kind != token.TokRBracket {
			lenExpr = p.parseExpr(0)
		}
		p.expect(token.TokRBracket)
		elem := p.parseTypeExpr()
		return &ast.ArrayType{
			LBracketPos: lb.Span,
			Len:         lenExpr,
			Elem:        elem,
		}
	case token.TokBang:
		bang := p.advance()
		// Bare `!` void Result vs `!T`
		if p.cur.Kind == token.TokLBrace || p.hasNewlineBetween(bang.Span, p.cur.Span) {
			return &ast.ResultType{BangPos: bang.Span, Inner: nil}
		}
		return &ast.ResultType{BangPos: bang.Span, Inner: p.parseTypeExpr()}
	case token.TokQuestion:
		q := p.advance()
		return &ast.OptionType{QuestionPos: q.Span, Inner: p.parseTypeExpr()}
	case token.TokAmp:
		amp := p.advance()
		isMut := false
		if p.cur.Kind == token.TokMut {
			isMut = true
			p.advance()
		}
		return &ast.RefType{AmpPos: amp.Span, IsMut: isMut, Inner: p.parseTypeExpr()}
	default:
		identTok := p.advance()
		ident := ast.Ident{Name: identTok.Lexeme(p.src), NamePos: identTok.Span}
		var tArgs []ast.TypeExpr
		if p.cur.Kind == token.TokLBracket {
			tArgs = p.parseTypeArgList()
		}
		return &ast.IdentType{Name: ident, TypeParams: tArgs}
	}
}

func (p *Parser) parseOptionalTypeParams() []ast.Ident {
	if p.cur.Kind != token.TokLBracket {
		return nil
	}
	p.advance()
	var params []ast.Ident
	for p.cur.Kind != token.TokRBracket && p.cur.Kind != token.TokEOF {
		params = append(params, p.parseIdent())
		if p.cur.Kind == token.TokComma {
			p.advance()
		}
	}
	p.expect(token.TokRBracket)
	return params
}

func (p *Parser) parseTypeArgList() []ast.TypeExpr {
	p.expect(token.TokLBracket)
	var args []ast.TypeExpr
	for p.cur.Kind != token.TokRBracket && p.cur.Kind != token.TokEOF {
		args = append(args, p.parseTypeExpr())
		if p.cur.Kind == token.TokComma {
			p.advance()
		}
	}
	p.expect(token.TokRBracket)
	return args
}

// --- STATEMENTS ---

func (p *Parser) parseBlock() *ast.BlockStmt {
	lbrace := p.expect(token.TokLBrace)
	var stmts []ast.Stmt
	for p.cur.Kind != token.TokRBrace && p.cur.Kind != token.TokEOF {
		if s := p.parseStmt(); s != nil {
			stmts = append(stmts, s)
		}
	}
	rbrace := p.expect(token.TokRBrace)
	return &ast.BlockStmt{
		Stmts:     stmts,
		BlockSpan: token.Span{Start: lbrace.Span.Start, End: rbrace.Span.End},
	}
}

func (p *Parser) parseStmt() ast.Stmt {
	switch p.cur.Kind {
	case token.TokReturn:
		retTok := p.advance()
		var val ast.Expr
		if p.cur.Kind != token.TokRBrace && !p.hasNewlineBetween(retTok.Span, p.cur.Span) {
			val = p.parseExpr(0)
		}
		end := retTok.Span.End
		if val != nil {
			end = val.Span().End
		}
		return &ast.ReturnStmt{
			Value:    val,
			StmtSpan: token.Span{Start: retTok.Span.Start, End: end},
		}

	case token.TokMut:
		mutTok := p.advance()
		name := p.parseIdent()
		p.expect(token.TokDeclAssign)
		val := p.parseExpr(0)
		return &ast.VarDeclStmt{
			Name:     name,
			IsMut:    true,
			Value:    val,
			StmtSpan: token.Span{Start: mutTok.Span.Start, End: val.Span().End},
		}

	case token.TokBreak, token.TokContinue:
		tok := p.advance()
		return &ast.BranchStmt{
			Tok:      tok.Kind,
			StmtSpan: tok.Span,
		}

	case token.TokIf:
		ifTok := p.advance()
		prevAllow := p.allowStructInit
		p.allowStructInit = false
		cond := p.parseExpr(0)
		p.allowStructInit = prevAllow
		thenBlock := p.parseBlock()
		var elseBlock *ast.BlockStmt
		endSpan := thenBlock.BlockSpan.End
		if p.cur.Kind == token.TokElse {
			p.advance()
			if p.cur.Kind == token.TokIf {
				nestedIf := p.parseStmt()
				elseBlock = &ast.BlockStmt{
					Stmts:     []ast.Stmt{nestedIf},
					BlockSpan: nestedIf.Span(),
				}
				endSpan = nestedIf.Span().End
			} else {
				elseBlock = p.parseBlock()
				endSpan = elseBlock.BlockSpan.End
			}
		}
		return &ast.IfStmt{
			Cond:     cond,
			Then:     thenBlock,
			Else:     elseBlock,
			StmtSpan: token.Span{Start: ifTok.Span.Start, End: endSpan},
		}

	case token.TokFor:
		forTok := p.advance()
		// 1. Infinite loop `for { ... }`
		if p.cur.Kind == token.TokLBrace {
			body := p.parseBlock()
			return &ast.ForStmt{
				Body:     body,
				StmtSpan: token.Span{Start: forTok.Span.Start, End: body.BlockSpan.End},
			}
		}

		prevAllow := p.allowStructInit
		p.allowStructInit = false

		// 2. For-in loop: `for item in items { ... }`
		if p.cur.Kind == token.TokIdent && p.peek.Kind == token.TokIn {
			name := p.parseIdent()
			p.advance() // consume `in`
			iter := p.parseExpr(0)
			p.allowStructInit = prevAllow
			body := p.parseBlock()
			return &ast.ForInStmt{
				Var:      name,
				Iter:     iter,
				Body:     body,
				StmtSpan: token.Span{Start: forTok.Span.Start, End: body.BlockSpan.End},
			}
		}

		// 3. 3-part loop: `for mut i := 0; i < 10; i += 1 { ... }` or `for i := 0; ...`
		if p.cur.Kind == token.TokMut || (p.cur.Kind == token.TokIdent && p.peek.Kind == token.TokDeclAssign) {
			initStmt := p.parseStmt()
			p.expect(token.TokSemicolon)
			var cond ast.Expr
			if p.cur.Kind != token.TokSemicolon {
				cond = p.parseExpr(0)
			}
			p.expect(token.TokSemicolon)
			var post ast.Stmt
			if p.cur.Kind != token.TokLBrace {
				post = p.parseStmt()
			}
			p.allowStructInit = prevAllow
			body := p.parseBlock()
			return &ast.ForStmt{
				Init:     initStmt,
				Cond:     cond,
				Post:     post,
				Body:     body,
				StmtSpan: token.Span{Start: forTok.Span.Start, End: body.BlockSpan.End},
			}
		}

		// 3. Condition-only loop: `for cond { ... }`
		cond := p.parseExpr(0)
		p.allowStructInit = prevAllow
		body := p.parseBlock()
		return &ast.ForStmt{
			Cond:     cond,
			Body:     body,
			StmtSpan: token.Span{Start: forTok.Span.Start, End: body.BlockSpan.End},
		}

	case token.TokArena:
		arenaTok := p.advance()
		body := p.parseBlock()
		return &ast.ArenaStmt{
			Body:     body,
			StmtSpan: token.Span{Start: arenaTok.Span.Start, End: body.BlockSpan.End},
		}

	default:
		// Immutable declaration `x := expr`
		if p.cur.Kind == token.TokIdent && p.peek.Kind == token.TokDeclAssign {
			name := p.parseIdent()
			p.advance() // consume `:=`
			val := p.parseExpr(0)
			return &ast.VarDeclStmt{
				Name:     name,
				IsMut:    false,
				Value:    val,
				StmtSpan: token.Span{Start: name.NamePos.Start, End: val.Span().End},
			}
		}

		// Parse LHS expression
		lhs := p.parseExpr(0)
		if p.cur.Kind == token.TokAssign || p.cur.Kind == token.TokPlusAssign || p.cur.Kind == token.TokSubAssign ||
			p.cur.Kind == token.TokMulAssign || p.cur.Kind == token.TokDivAssign || p.cur.Kind == token.TokModAssign {
			op := p.advance().Kind
			rhs := p.parseExpr(0)
			return &ast.AssignStmt{
				Target:   lhs,
				Op:       op,
				Value:    rhs,
				StmtSpan: token.Span{Start: lhs.Span().Start, End: rhs.Span().End},
			}
		}

		return &ast.ExprStmt{Expr: lhs}
	}
}

// --- PRATT EXPRESSION PARSER ---

func (p *Parser) parseExpr(minPrec int) ast.Expr {
	left := p.parsePrefix()
	if left == nil {
		return nil
	}

	for {
		prec := p.cur.Kind.Precedence()
		if prec <= minPrec {
			break
		}
		// Postfix & Infix dispatch (`led`)
		switch p.cur.Kind {
		case token.TokDot, token.TokChanSend:
			isSend := p.advance().Kind == token.TokChanSend
			field := p.parseIdent()
			left = &ast.SelectorExpr{
				Left:        left,
				Field:       field,
				IsActorSend: isSend,
				SelSpan:     token.Span{Start: left.Span().Start, End: field.NamePos.End},
			}
		case token.TokLParen:
			left = p.parseCallExpr(left)
		case token.TokLBracket:
			p.advance()
			idx := p.parseExpr(0)
			rb := p.expect(token.TokRBracket)
			left = &ast.IndexExpr{
				Left:      left,
				Index:     idx,
				IndexSpan: token.Span{Start: left.Span().Start, End: rb.Span.End},
			}
		default:
			opTok := p.advance()
			right := p.parseExpr(prec)
			left = &ast.BinaryExpr{
				Left:    left,
				Op:      opTok.Kind,
				Right:   right,
				BinSpan: token.Span{Start: left.Span().Start, End: right.Span().End},
			}
		}
	}

	return left
}

func (p *Parser) parsePrefix() ast.Expr {
	switch p.cur.Kind {
	case token.TokIdent:
		ident := p.parseIdent()
		// Struct initialization literal: `Invoice{ id: '123', amount_cents: 5000 }`
		if p.allowStructInit && p.cur.Kind == token.TokLBrace && isPascalCase(ident.Name) {
			return p.parseStructInit(&ast.IdentType{Name: ident})
		}
		return &ast.IdentExpr{Ident: ident}

	case token.TokInt, token.TokFloat, token.TokTrue, token.TokFalse:
		tok := p.advance()
		return &ast.LiteralExpr{
			Kind:    tok.Kind,
			Value:   tok.Lexeme(p.src),
			LitSpan: tok.Span,
		}

	case token.TokString:
		tok := p.advance()
		raw := tok.Lexeme(p.src)
		return &ast.LiteralExpr{
			Kind:    tok.Kind,
			Value:   strings.Trim(raw, "'"),
			LitSpan: tok.Span,
		}

	case token.TokStrStart:
		return p.parseStringInterp()

	case token.TokLParen:
		p.advance()
		expr := p.parseExpr(0)
		p.expect(token.TokRParen)
		return expr

	case token.TokBang, token.TokMinus:
		opTok := p.advance()
		operand := p.parseExpr(7)
		return &ast.UnaryExpr{
			Op:        opTok.Kind,
			Expr:      operand,
			UnarySpan: token.Span{Start: opTok.Span.Start, End: operand.Span().End},
		}

	case token.TokLBracket:
		lb := p.advance()
		if p.cur.Kind == token.TokRBracket {
			rb := p.advance()
			if p.isTypeStart(p.cur.Kind) {
				elemType := p.parseTypeExpr()
				p.expect(token.TokLBrace)
				var elems []ast.Expr
				for p.cur.Kind != token.TokRBrace && p.cur.Kind != token.TokEOF {
					elems = append(elems, p.parseExpr(0))
					if p.cur.Kind == token.TokComma {
						p.advance()
					}
				}
				rbrace := p.expect(token.TokRBrace)
				return &ast.ArrayLitExpr{
					Type: &ast.ArrayType{
						LBracketPos: lb.Span,
						Len:         nil,
						Elem:        elemType,
					},
					Elems:   elems,
					LitSpan: token.Span{Start: lb.Span.Start, End: rbrace.Span.End},
				}
			}
			return &ast.ArrayLitExpr{
				Elems:   nil,
				LitSpan: token.Span{Start: lb.Span.Start, End: rb.Span.End},
			}
		}

		firstExpr := p.parseExpr(0)
		if p.cur.Kind == token.TokRBracket && p.isTypeStart(p.peek.Kind) {
			p.advance()
			elemType := p.parseTypeExpr()
			p.expect(token.TokLBrace)
			var elems []ast.Expr
			for p.cur.Kind != token.TokRBrace && p.cur.Kind != token.TokEOF {
				elems = append(elems, p.parseExpr(0))
				if p.cur.Kind == token.TokComma {
					p.advance()
				}
			}
			rbrace := p.expect(token.TokRBrace)
			return &ast.ArrayLitExpr{
				Type: &ast.ArrayType{
					LBracketPos: lb.Span,
					Len:         firstExpr,
					Elem:        elemType,
				},
				Elems:   elems,
				LitSpan: token.Span{Start: lb.Span.Start, End: rbrace.Span.End},
			}
		}

		var elems []ast.Expr
		elems = append(elems, firstExpr)
		if p.cur.Kind == token.TokComma {
			p.advance()
		}
		for p.cur.Kind != token.TokRBracket && p.cur.Kind != token.TokEOF {
			elems = append(elems, p.parseExpr(0))
			if p.cur.Kind == token.TokComma {
				p.advance()
			}
		}
		rb := p.expect(token.TokRBracket)
		return &ast.ArrayLitExpr{
			Elems:   elems,
			LitSpan: token.Span{Start: lb.Span.Start, End: rb.Span.End},
		}

	default:
		p.errorf(p.cur.Span, "unexpected token in expression: %s (%q)", p.cur.Kind, p.cur.Lexeme(p.src))
		tok := p.advance()
		return &ast.LiteralExpr{Kind: token.TokIllegal, Value: "", LitSpan: tok.Span}
	}
}

func (p *Parser) parseStructInit(typ ast.TypeExpr) *ast.StructInitExpr {
	p.expect(token.TokLBrace)
	var fields []ast.StructInitField

	for p.cur.Kind != token.TokRBrace && p.cur.Kind != token.TokEOF {
		fieldName := p.parseIdent()
		p.expect(token.TokColon)
		val := p.parseExpr(0)
		fields = append(fields, ast.StructInitField{
			Name:      fieldName,
			Value:     val,
			FieldSpan: token.Span{Start: fieldName.NamePos.Start, End: val.Span().End},
		})
		if p.cur.Kind == token.TokComma {
			p.advance()
		}
	}

	rbrace := p.expect(token.TokRBrace)
	return &ast.StructInitExpr{
		Type:     typ,
		Fields:   fields,
		InitSpan: token.Span{Start: typ.Span().Start, End: rbrace.Span.End},
	}
}

func (p *Parser) parseCallExpr(callee ast.Expr) *ast.CallExpr {
	p.expect(token.TokLParen)
	var args []ast.CallArg

	for p.cur.Kind != token.TokRParen && p.cur.Kind != token.TokEOF {
		argStart := p.cur.Span.Start
		isMut := false
		if p.cur.Kind == token.TokMut {
			isMut = true
			p.advance()
		}
		expr := p.parseExpr(0)
		args = append(args, ast.CallArg{
			IsMut:   isMut,
			Expr:    expr,
			ArgSpan: token.Span{Start: argStart, End: expr.Span().End},
		})
		if p.cur.Kind == token.TokComma {
			p.advance()
		}
	}

	rparen := p.expect(token.TokRParen)
	endSpan := rparen.Span.End

	// Optional Result/Option propagation (`call()!` or `call()?`) or `or { ... }` block
	propagate := false
	var orBlock *ast.BlockStmt

	if p.cur.Kind == token.TokBang || p.cur.Kind == token.TokQuestion {
		propagate = true
		endSpan = p.advance().Span.End
	} else if p.cur.Kind == token.TokOr {
		p.advance() // consume `or`
		orBlock = p.parseBlock()
		endSpan = orBlock.BlockSpan.End
	}

	return &ast.CallExpr{
		Callee:    callee,
		Args:      args,
		OrBlock:   orBlock,
		Propagate: propagate,
		CallSpan:  token.Span{Start: callee.Span().Start, End: endSpan},
	}
}

func (p *Parser) parseStringInterp() *ast.StringInterpExpr {
	startTok := p.expect(token.TokStrStart)
	firstSeg := strings.TrimPrefix(startTok.Lexeme(p.src), "'")

	segments := []string{firstSeg}
	var exprs []ast.Expr
	endPos := startTok.Span.End

	for p.cur.Kind == token.TokInterpBeg {
		p.advance() // consume `${`
		expr := p.parseExpr(0)
		exprs = append(exprs, expr)
		p.expect(token.TokInterpEnd) // consume closing `}`

		if p.cur.Kind == token.TokStrPart {
			partTok := p.advance()
			segments = append(segments, partTok.Lexeme(p.src))
			endPos = partTok.Span.End
		} else if p.cur.Kind == token.TokStrEnd {
			endTok := p.advance()
			lastSeg := strings.TrimSuffix(endTok.Lexeme(p.src), "'")
			segments = append(segments, lastSeg)
			endPos = endTok.Span.End
			break
		} else {
			p.errorf(p.cur.Span, "expected string continuation after interpolation, got %s", p.cur.Kind)
			break
		}
	}

	return &ast.StringInterpExpr{
		Segments:   segments,
		Exprs:      exprs,
		InterpSpan: token.Span{Start: startTok.Span.Start, End: endPos},
	}
}

func isPascalCase(s string) bool {
	return len(s) > 0 && s[0] >= 'A' && s[0] <= 'Z'
}
