package parser_test

import (
	"testing"

	"github.com/viralang/vira/internal/parser"
)

// `import mypkg/sub` means the same as `import "mypkg/sub"`.
func TestUnquotedSubPackageImport(t *testing.T) {
	src := "module main\n\nimport mypkg/sub\nimport \"mypkg/other\"\nimport (\n\ta/b/c as abc\n\tplain\n)\n\nfn main() {\n}\n"
	p := parser.New(src)
	f := p.ParseFile()
	if errs := p.Errors(); len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	want := []struct{ path, name, alias string }{
		{"mypkg/sub", "mypkg/sub", ""},
		{"mypkg/other", "mypkg/other", ""},
		{"a/b/c", "abc", "abc"},
		{"plain", "plain", ""},
	}
	if len(f.Imports) != len(want) {
		t.Fatalf("got %d imports, want %d: %+v", len(f.Imports), len(want), f.Imports)
	}
	for i, w := range want {
		got := f.Imports[i]
		if got.Path != w.path || got.Name != w.name || got.Alias != w.alias {
			t.Errorf("import %d: got %+v, want path=%q name=%q alias=%q", i, got, w.path, w.name, w.alias)
		}
	}

	bad := parser.New("module main\n\nimport mypkg/\n\nfn main() {\n}\n")
	bad.ParseFile()
	if len(bad.Errors()) == 0 {
		t.Errorf("expected a syntax error for a trailing '/' in an import path")
	}
}
