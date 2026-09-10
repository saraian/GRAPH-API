#!/usr/bin/env python3
"""A shell comment inside the run_metadata heredoc lands verbatim in the JSON.

GA-436 (2026-09-10). `cat <<EOF > "$RUN_DIR/run_metadata.json"` is an UNQUOTED heredoc: it expands
variables and command substitutions, and passes everything else through as text. A `#` line inside it
is not a comment -- it is five lines of prose in the middle of a JSON object.

MEASURED, on Gin, bundle 20260910_115818_hm3d_00861: five comment lines above "tour_shape" produced

    Expecting property name enclosed in double quotes: line 73 column 5 (char 6696)
    !! run_metadata.json is not valid JSON -- aborting rather than shipping an unreadable bundle

and the run aborted before the stack started. `bash -n` passes the file; the damage is in the
artefact, not in the shell. That is rule 32's shape (a `#` after a line continuation eats the lines
above it and `bash -n` reports the file clean), and nothing in the tree checked for it.

TWO SPANS INSIDE THE HEREDOC ARE REAL COMMENT CONTEXTS and this test must not flag them:
  - a backtick block, which is command substitution: the launcher already carries the extension's
    policy comment that way, on purpose;
  - an embedded `python3 - <<'PY'` span, where `#` is a Python comment.
Everything else at JSON level is text.

Run: python3 test_metadata_heredoc_comments.py   (or under pytest)
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "live_run.sh")
OPEN_RE = re.compile(r'cat\s+<<EOF\s*>\s*"\$RUN_DIR/run_metadata\.json"')


def offending_lines(text):
    """-> [(line number, line)] for every raw shell comment at JSON level.

    A `#` inside a command substitution is a comment in whatever language runs there, and the
    launcher uses two of them deliberately: the extension's policy note rides inside a BACKTICK
    block, and the effective_config value is `python3 -c "` spanning twenty lines, where `#` is a
    Python comment.

    THE TEST IS "IS THIS LINE STILL INSIDE SOMETHING THE PREVIOUS LINE OPENED", not a parser.
    At JSON level every line closes its own quotes -- `"key": "value",` -- so an unbalanced double
    quote means a string is still open and this line belongs to another language. Counting
    parentheses does NOT work here and was tried: the embedded Python is full of `)` with no `$(`,
    so the depth collapses to zero inside it and the block reads as JSON level.
    """
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if OPEN_RE.search(ln))
    bad, in_dquote, in_backtick = [], False, False
    for n in range(start + 1, len(lines)):
        ln = lines[n]
        at_json_level = not in_dquote and not in_backtick
        if at_json_level and ln.strip() == "EOF":
            break
        if at_json_level and ln.lstrip().startswith("#"):
            bad.append((n + 1, ln.strip()))
        i = 0
        while i < len(ln):
            c = ln[i]
            if c == "\\":
                i += 2
                continue
            if c == "`" and not in_dquote:
                in_backtick = not in_backtick
            elif c == '"':
                in_dquote = not in_dquote
            i += 1
    return bad


def test_no_raw_shell_comment_inside_the_metadata_heredoc():
    bad = offending_lines(open(SCRIPT).read())
    assert not bad, (
        "these lines are inside the run_metadata heredoc and will be written into the JSON "
        "verbatim, so the bundle is unreadable and the run aborts:\n"
        + "\n".join(f"  {SCRIPT}:{n}: {ln[:90]}" for n, ln in bad)
        + "\nPut the text above the heredoc, or make it a sibling key such as tour_shape_note.")


def test_the_check_sees_a_comment_when_one_is_there():
    """The instrument is exercised where the answer is PRESENT, not only where it is absent."""
    planted = '''cat <<EOF > "$RUN_DIR/run_metadata.json"
{
  "a": 1,
  # this one must be caught
  `# this one is a command substitution and is fine`
  "b": 2
}
EOF
'''
    bad = offending_lines(planted)
    assert [n for n, _ in bad] == [4], bad


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
