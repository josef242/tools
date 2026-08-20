#!/usr/bin/env python3
"""A small query language for record metadata, for curating corpora interactively.

    ('Category': 'Gen' OR 'Characters') AND 'author': SUBSTR('xenak') AND 'words' < 100

Design notes, all driven by what curation actually needs:

* CASE-INSENSITIVE THROUGHOUT. Corpus metadata is human-entered; case is noise.

* ':' IS CONTAINS, '=' IS EXACT-ELEMENT. Many fields hold comma-separated lists
  ("Merlin (BBC), xkcd"), so ':' finds a substring anywhere, while '=' matches one whole
  element and ignores the others. A scalar field is just a one-element list, so the two
  behave consistently without a special case.

* 'OR' JOINS VALUES INSIDE A FIELD as well as whole predicates. In
  "'Category': 'Gen' OR 'Characters'" the second string is another VALUE of Category, not
  a field name -- which is how people actually think about faceted filters. A bare string
  that is not followed by an operator continues the preceding field.

* AN UNKNOWN FIELD IS AN ERROR, never a silently-false predicate. Typing 'Gen' when the
  key is 'Gender' should say so rather than return zero rows and let you conclude your
  corpus lacks something it has.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class QueryError(ValueError):
    """Raised for a malformed query or an unknown field."""


# =============================================================================
# Lexer
# =============================================================================

_TOKEN_RE = re.compile(r"""
      (?P<ws>\s+)
    | (?P<lparen>\()
    | (?P<rparen>\))
    | (?P<comma>,)
    | (?P<op>!=|<=|>=|[:=<>~])
    | (?P<str>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
    | (?P<num>-?\d+(?:\.\d+)?)
    | (?P<word>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
""", re.VERBOSE)

_KEYWORDS = {'AND', 'OR', 'NOT'}
_FUNCS = {'SUBSTR', 'EXISTS', 'HAS'}


class Tok:
    __slots__ = ('kind', 'val', 'pos')

    def __init__(self, kind: str, val: str, pos: int):
        self.kind, self.val, self.pos = kind, val, pos

    def __repr__(self):
        return f"Tok({self.kind},{self.val!r})"


def lex(q: str) -> List[Tok]:
    out: List[Tok] = []
    i = 0
    while i < len(q):
        m = _TOKEN_RE.match(q, i)
        if not m:
            raise QueryError(f"Unexpected character {q[i]!r} at position {i}")
        i = m.end()
        kind = m.lastgroup
        text = m.group()
        if kind == 'ws':
            continue
        if kind == 'str':
            out.append(Tok('str', _unquote(text), m.start()))
        elif kind == 'word':
            word = text.strip()
            upper = word.upper()
            if upper in _KEYWORDS:
                out.append(Tok(upper, upper, m.start()))
            elif upper in _FUNCS:
                out.append(Tok('func', upper, m.start()))
            else:
                # A bare word is a field name or an unquoted value. Bare words may NOT
                # contain spaces: allowing them made "AND NOT" lex as one word and quietly
                # swallow the operators. Names with spaces must be quoted --
                # 'Additional Tags' -- which is what a user writes anyway.
                out.append(Tok('str', word, m.start()))
        else:
            out.append(Tok(kind, text, m.start()))
    return out


def _unquote(s: str) -> str:
    body = s[1:-1]
    return re.sub(r'\\(.)', r'\1', body)


# =============================================================================
# AST
# =============================================================================

class Node:
    def evaluate(self, table) -> np.ndarray:
        raise NotImplementedError

    def fields(self) -> Set[str]:
        raise NotImplementedError


class And(Node):
    def __init__(self, kids: Sequence[Node]):
        self.kids = list(kids)

    def evaluate(self, table):
        m = self.kids[0].evaluate(table)
        for kid in self.kids[1:]:
            if not m.any():
                return m                      # short-circuit: nothing left to narrow
            m = m & kid.evaluate(table)
        return m

    def fields(self):
        return set().union(*(k.fields() for k in self.kids))

    def __repr__(self):
        return f"And({self.kids!r})"


class Or(Node):
    def __init__(self, kids: Sequence[Node]):
        self.kids = list(kids)

    def evaluate(self, table):
        m = self.kids[0].evaluate(table)
        for kid in self.kids[1:]:
            m = m | kid.evaluate(table)
        return m

    def fields(self):
        return set().union(*(k.fields() for k in self.kids))

    def __repr__(self):
        return f"Or({self.kids!r})"


class Not(Node):
    def __init__(self, kid: Node):
        self.kid = kid

    def evaluate(self, table):
        return ~self.kid.evaluate(table)

    def fields(self):
        return self.kid.fields()

    def __repr__(self):
        return f"Not({self.kid!r})"


class Pred(Node):
    """One field/operator/value test."""

    def __init__(self, field: str, op: str, value: Any):
        self.field, self.op, self.value = field, op, value

    def fields(self):
        return {self.field}

    def __repr__(self):
        return f"Pred({self.field!r},{self.op},{self.value!r})"

    def evaluate(self, table):
        col = table.column(self.field)          # pandas Series of strings (NaN allowed)
        n = len(col)

        if self.op == 'exists':
            return (col.notna() & (col.astype(str).str.strip() != '')).to_numpy()

        if self.op in ('<', '<=', '>', '>=', '==num', '!=num'):
            nums = pd.to_numeric(col, errors='coerce')
            v = float(self.value)
            with np.errstate(invalid='ignore'):
                if self.op == '<':
                    m = nums < v
                elif self.op == '<=':
                    m = nums <= v
                elif self.op == '>':
                    m = nums > v
                elif self.op == '>=':
                    m = nums >= v
                elif self.op == '==num':
                    m = nums == v
                else:
                    m = nums != v
            # A value that will not parse as a number cannot satisfy a numeric test.
            return (m & nums.notna()).to_numpy()

        s = col.fillna('').astype(str)
        val = str(self.value)

        if self.op == ':':
            return s.str.contains(re.escape(val), case=False, regex=True, na=False).to_numpy()

        if self.op == '~':
            try:
                return s.str.contains(val, case=False, regex=True, na=False).to_numpy()
            except re.error as e:
                raise QueryError(f"Invalid regex {val!r}: {e}")

        if self.op in ('=', '!='):
            # Match one whole COMMA-SEPARATED ELEMENT, so 'Merlin (BBC)' hits a record whose
            # Fandoms is "Merlin (BBC), xkcd" without also matching a longer fandom that
            # merely starts with it. Anchored on element boundaries rather than string ends.
            pat = r'(?:^|,)\s*' + re.escape(val) + r'\s*(?:,|$)'
            m = s.str.contains(pat, case=False, regex=True, na=False).to_numpy()
            return ~m if self.op == '!=' else m

        raise QueryError(f"Unsupported operator {self.op!r}")


# =============================================================================
# Parser  (recursive descent)
# =============================================================================

class _Parser:
    def __init__(self, toks: List[Tok], known: Optional[Set[str]]):
        self.toks, self.i = toks, 0
        self.known = known

    def peek(self, off=0) -> Optional[Tok]:
        j = self.i + off
        return self.toks[j] if j < len(self.toks) else None

    def next(self) -> Tok:
        t = self.peek()
        if t is None:
            raise QueryError("Unexpected end of query")
        self.i += 1
        return t

    def expect(self, kind: str) -> Tok:
        t = self.next()
        if t.kind != kind:
            raise QueryError(f"Expected {kind} but found {t.val!r} at position {t.pos}")
        return t

    # -- grammar ----------------------------------------------------------
    def parse(self) -> Node:
        node = self.parse_or()
        if self.peek() is not None:
            t = self.peek()
            raise QueryError(f"Unexpected {t.val!r} at position {t.pos}")
        return node

    def parse_or(self) -> Node:
        kids = [self.parse_and()]
        while self.peek() and self.peek().kind == 'OR':
            self.next()
            kids.append(self.parse_and(prev=kids[-1]))
        return kids[0] if len(kids) == 1 else Or(kids)

    def parse_and(self, prev: Optional[Node] = None) -> Node:
        kids = [self.parse_not(prev=prev)]
        while self.peek() and self.peek().kind == 'AND':
            self.next()
            kids.append(self.parse_not())
        return kids[0] if len(kids) == 1 else And(kids)

    def parse_not(self, prev: Optional[Node] = None) -> Node:
        if self.peek() and self.peek().kind == 'NOT':
            self.next()
            return Not(self.parse_not())
        return self.parse_primary(prev=prev)

    def parse_primary(self, prev: Optional[Node] = None) -> Node:
        t = self.peek()
        if t is None:
            raise QueryError("Unexpected end of query")

        if t.kind == 'lparen':
            self.next()
            node = self.parse_or()
            self.expect('rparen')
            return node

        if t.kind == 'func':
            return self.parse_func()

        if t.kind == 'str':
            nxt = self.peek(1)
            # A bare value with no operator after it continues the PREVIOUS field:
            #   'Category': 'Gen' OR 'Characters'
            # This is what makes faceted filters read naturally.
            if nxt is None or nxt.kind not in ('op',):
                if prev is not None:
                    base = _last_pred(prev)
                    if base is not None:
                        self.next()
                        return Pred(base.field, base.op, t.val)
                raise QueryError(
                    f"{t.val!r} at position {t.pos} has no operator. Write a test such as "
                    f"{t.val!r}: 'value', or use EXISTS({t.val!r}) to test for presence.")
            field = self.next().val
            self.check_field(field, t.pos)
            op = self.next().val
            return self.parse_value(field, op)

        raise QueryError(f"Unexpected {t.val!r} at position {t.pos}")

    def parse_func(self) -> Node:
        fn = self.next().val
        self.expect('lparen')
        arg = self.expect('str').val
        self.expect('rparen')
        if fn == 'EXISTS':
            self.check_field(arg, 0)
            return Pred(arg, 'exists', None)
        raise QueryError(f"{fn}() must follow a field, e.g. 'author': {fn}('xenak')")

    def parse_value(self, field: str, op: str) -> Node:
        t = self.peek()
        if t is None:
            raise QueryError(f"Missing value after {field!r} {op}")

        if t.kind == 'func':
            fn = self.next().val
            self.expect('lparen')
            arg = self.expect('str').val
            self.expect('rparen')
            if fn == 'SUBSTR':
                return Pred(field, ':', arg)
            if fn == 'HAS':
                return Pred(field, '=', arg)
            raise QueryError(f"{fn}() is not valid as a value for {field!r}")

        if t.kind == 'lparen':
            # 'Fandoms': ('Merlin' OR 'xkcd')  -- an explicit value group
            self.next()
            vals = [self.value_token(field, op)]
            while self.peek() and self.peek().kind in ('OR', 'comma'):
                self.next()
                vals.append(self.value_token(field, op))
            self.expect('rparen')
            return vals[0] if len(vals) == 1 else Or(vals)

        return self.value_token(field, op)

    def value_token(self, field: str, op: str) -> Pred:
        t = self.next()
        if t.kind not in ('str', 'num'):
            raise QueryError(f"Expected a value for {field!r} but found {t.val!r}")
        if op in ('<', '<=', '>', '>='):
            return Pred(field, op, float(t.val))
        if t.kind == 'num' and op in ('=', '!='):
            return Pred(field, '==num' if op == '=' else '!=num', float(t.val))
        return Pred(field, op, t.val)

    def check_field(self, field: str, pos: int):
        if self.known is None or field in self.known:
            return
        lower = {k.lower(): k for k in self.known}
        if field.lower() in lower:
            return                                     # case-insensitive field names too
        near = [k for k in self.known if field.lower() in k.lower()][:6]
        hint = f" Did you mean: {', '.join(sorted(near))}?" if near else ""
        raise QueryError(f"No key named {field!r}.{hint}")


def _last_pred(node: Node) -> Optional[Pred]:
    """Rightmost Pred of a node, used to carry a field across a bare OR value."""
    if isinstance(node, Pred):
        return node
    if isinstance(node, (And, Or)):
        return _last_pred(node.kids[-1])
    if isinstance(node, Not):
        return _last_pred(node.kid)
    return None


def parse(query: str, known_fields: Optional[Set[str]] = None) -> Node:
    """Parse a metadata query. `known_fields` enables unknown-key errors."""
    toks = lex(query)
    if not toks:
        raise QueryError("Empty query")
    return _Parser(toks, known_fields).parse()
