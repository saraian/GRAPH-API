#!/usr/bin/env python3
"""Report identifiers a viewer's inline JavaScript uses but never declares.

Why this exists: three defects in found/dashboard/viewer.html (GA-86/87/88) were all
this one shape -- a function called and never defined, a variable read and never
declared, and a second copy of a function that hoisted over the first. None of them
raise until the branch runs, and all three sat in a page that looked healthy. A
browser finds them one ReferenceError at a time; this finds them without a browser.

It is deliberately noisy in one direction only: it over-reports (a `let a = 1, b = 2;`
declares b in a form the regex misses) and does not under-report. Treat every hit as a
candidate to confirm, not as a verdict.

    python3 check_undeclared.py viewer.html [more.html ...]

Exit 1 if any file has a candidate or a duplicated function declaration.

A check that only ever reports NONE has asserted nothing, so this one has a negative
control. Run against found/dashboard/viewer.html as it stood before GA-86/87/88 were
applied, it reports exactly the three defects and nothing else:

    DUPLICATE function declarations (the later one wins): drawBEV
    undeclared candidates (2): drawBEVOffline, lastGraphData

Against both viewers after the fix it reports NONE. Re-run the control after changing
this file; a scanner bug shows up as silence, not as an error.
"""
import re
import sys
from pathlib import Path

GLOBALS = set("""
window document console navigator location history screen Math JSON Object Array String
Number Boolean Date RegExp Error Map Set WeakMap WeakSet Promise Symbol Proxy Reflect
BigInt Intl parseInt parseFloat isNaN isFinite encodeURIComponent decodeURIComponent
encodeURI decodeURI setTimeout clearTimeout setInterval clearInterval requestAnimationFrame
cancelAnimationFrame fetch alert confirm prompt localStorage sessionStorage FormData
Headers Request Response URL URLSearchParams Blob File FileReader Image Audio Video
WebSocket EventSource XMLHttpRequest AbortController TextEncoder TextDecoder
undefined null true false NaN Infinity this arguments super new typeof instanceof void
delete in of let const var function return if else for while do switch case break
continue try catch finally throw class extends import export default async await yield
static get set target performance structuredClone queueMicrotask crypto
cytoscape Cytoscape lucide L d3 Chart THREE Plotly io jQuery
""".split())

# Declaration forms we recognise. Anything declared in a form not listed here shows up
# as a false positive, which is the safe direction.
DECL = [
    r'\bfunction\s+([A-Za-z_$][\w$]*)',
    r'\bclass\s+([A-Za-z_$][\w$]*)',
    r'\bcatch\s*\(\s*([A-Za-z_$][\w$]*)',
    r'([A-Za-z_$][\w$]*)\s*=>',
]

# Forms that declare SEVERAL names at once. Each of these produced a false positive on a
# real viewer, which is why they are listed separately rather than folded into DECL:
#   let bevPanX = 0, bevPanY = 0;      -- only the first name was seen
#   const [x1, y1, x2, y2] = box;      -- array destructuring
#   const { act, extra } = opts;       -- object destructuring
#   function sendAction(act, extra) {} -- the second parameter
MULTI = [
    # every declarator in a let/const/var statement, including destructured ones
    (r'\b(?:let|const|var)\s+([^;=\n]*(?:=[^;\n]*)?(?:,[^;\n]*)*)', r'[A-Za-z_$][\w$]*'),
    # every parameter of a named function
    (r'\bfunction\s*[\w$]*\s*\(([^)]*)\)', r'[A-Za-z_$][\w$]*'),
    # every parameter of an arrow function written with parentheses
    (r'\(([^()]*)\)\s*=>', r'[A-Za-z_$][\w$]*'),
]



def inline_js(html: str) -> str:
    return "\n".join(re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', html, re.S))


def strip_noise(js: str) -> str:
    """Return only the parts of `js` that are code.

    A regex cannot do this. These viewers nest template literals inside their own
    interpolations -- `... ${cond ? `a` : `b`} ...` -- and a non-greedy backtick pattern
    ends the outer template at the first inner backtick, so every following quote is
    mismatched and whole blocks of HTML text leak in as identifiers. That produced
    twelve false hits on the bridge viewer. Scan once, tracking the nesting.

    Template TEXT is dropped; the ${...} expressions inside it are kept, because that is
    where a template actually reads a variable.
    """
    out = []
    i, n = 0, len(js)
    stack = []        # 'tmpl', or an int: brace depth inside a ${...} interpolation
    prev = ''         # last significant code character: decides regex vs division
    while i < n:
        c = js[i]

        if stack and stack[-1] == 'tmpl':
            if c == '\\':
                i += 2
            elif c == '`':
                stack.pop()
                out.append(' ')
                i += 1
            elif js[i:i + 2] == '${':
                stack.append(0)
                out.append(' ')
                i += 2
            else:
                i += 1          # literal template text: dropped
            continue

        if js[i:i + 2] == '//':
            j = js.find('\n', i)
            i = n if j < 0 else j
            continue
        if js[i:i + 2] == '/*':
            j = js.find('*/', i + 2)
            i = n if j < 0 else j + 2
            out.append(' ')
            continue

        if c in '"\'':
            q, i = c, i + 1
            while i < n and js[i] != q:
                i += 2 if js[i] == '\\' else 1
            i += 1
            out.append(' "" ')
            prev = '"'
            continue

        if c == '`':
            stack.append('tmpl')
            out.append(' ')
            i += 1
            continue

        if isinstance(stack[-1] if stack else None, int):
            if c == '{':
                stack[-1] += 1
            elif c == '}':
                if stack[-1] == 0:
                    stack.pop()
                    out.append(' ')
                    i += 1
                    continue
                stack[-1] -= 1

        # a `/` here starts a regex literal, not a division, when the previous
        # significant character cannot end an expression. /[:.]/g was otherwise
        # reported as an undeclared `g`.
        if c == '/' and prev in '=(,:[!&|?+{;~^%<>*-':
            j, in_class = i + 1, False
            while j < n:
                d = js[j]
                if d == '\\':
                    j += 2
                    continue
                if d == '\n':
                    break
                if d == '[':
                    in_class = True
                elif d == ']':
                    in_class = False
                elif d == '/' and not in_class:
                    break
                j += 1
            if j < n and js[j] == '/':
                j += 1
                while j < n and js[j] in 'gimsuy':
                    j += 1
                i = j
                out.append(' 0 ')
                prev = '0'
                continue

        out.append(c)
        if not c.isspace():
            prev = c
        i += 1
    return ''.join(out)


def scan(path: Path):
    js = strip_noise(inline_js(path.read_text()))

    declared = set(GLOBALS)
    for pat in DECL:
        for m in re.finditer(pat, js):
            declared.add(m.group(1).strip())
    for pat, name_re in MULTI:
        for m in re.finditer(pat, js):
            # a declarator list is `a = f(b), c` -- only names LEFT of an `=` are
            # declared, the rest are uses, so each chunk contributes its first name only
            for chunk in m.group(1).split(','):
                head = chunk.split('=')[0]
                found = re.findall(name_re, head)
                if found:
                    declared.add(found[0] if '[' not in head and '{' not in head else None)
                    declared.update(found)
            declared.discard(None)

    # a.b and {b: ...} and case 'b': are not free identifiers
    used = set()
    # The \b matters: without it the engine backtracks off the last character to make a
    # trailing (?!\s*:) succeed, so `amount:` is reported as an undeclared `amoun`.
    for m in re.finditer(r'(?<![.\w$])([A-Za-z_$][\w$]*)\b(?!\s*:)', js):
        used.add(m.group(1))

    candidates = sorted(n for n in used - declared if not n.isupper() or len(n) > 3)

    funcs = re.findall(r'\bfunction\s+([A-Za-z_$][\w$]*)', js)
    dupes = sorted({f for f in funcs if funcs.count(f) > 1})

    return candidates, dupes


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    bad = False
    for arg in argv[1:]:
        p = Path(arg)
        cands, dupes = scan(p)
        print(f"== {p}")
        # A later `function f(){}` hoists over an earlier one and silently wins. This is
        # the JavaScript equivalent of ruff's F811 and is always a defect here.
        if dupes:
            print(f"   DUPLICATE function declarations (the later one wins): {', '.join(dupes)}")
            bad = True
        if cands:
            print(f"   undeclared candidates ({len(cands)}), confirm each in a running page:")
            for c in cands:
                print(f"     {c}")
            bad = True
        if not dupes and not cands:
            print("   NONE")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
