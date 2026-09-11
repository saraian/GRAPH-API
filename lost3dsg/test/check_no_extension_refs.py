#!/usr/bin/env python3
"""This repository must not name any package that extends it. Enforced, not merely written.

WHY MECHANICAL. On 2026-08-28 the owner ruled that `live_run.sh` contains ZERO references to the
extension, with the extension owning its own mounts and variables. Nothing checked it. By
2026-09-09 that one file held 112, the launcher REFUSED to start without the extension's package,
and every one of those references had reached the upstream repository through an ordinary merge.
A ruling with no check is a preference, and this one survived eleven days and reversed itself.

WHAT IS ALLOWED, and why each: the English word in a message a human reads, and a submodule that
happens to share the name. Everything else -- a variable, a mount, an absolute path, an import --
is coupling pointing the wrong way: from the generic stack onto the thing built on top of it.

    python3 check_no_extension_refs.py            # -> exit 1 and the offending lines

Run it from anywhere in the checkout.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

# A name this repository must not depend on. Add one per extension, never a pattern that would
# also match ordinary English -- the check has to be cheap to trust.
FORBIDDEN = re.compile(
    r"FOUND_[A-Z]"          # a policy variable
    r"|/DATA/FOUND|/found/" # a path or a mount
    r"|\bFOUND\b"           # the package, named
    # a MODULE path: quoted, imported, or written as a dotted spec. NOT `found.add(`,
    # which is an ordinary local set -- the first version of this pattern flagged it,
    # and a check that cries wolf is switched off within a week.
    r"|[\"']found\.[a-z_]+|import\s+found\b|\bfound\.[a-z_]+:"
)

# Exact lines that are NOT coupling. Each is listed with its reason, so an addition here is an
# argument rather than a silencing: a bare skip list is how a check stops meaning anything.
ALLOWED = {
    "[BEST MATCH FOUND]": "the English word, in a message a human reads",
    "🔍 [BEST MATCH FOUND]": "the English word, in a message a human reads",
    'FOUND-Dataset': "a submodule that shares the name; a different repository, required by the experiments",
    "FOUND AND REMOVED": "the English word",
    "has FOUND onto": "the English word",
    # The checker cannot look for a path without naming it. Every other pattern above is written so
    # it does not match its own source (FOUND_ has no uppercase after it, \bFOUND\b has a word char
    # before the F); a literal path has no such escape, so it is named here with its reason like any
    # other line. The key carries the surrounding pipes, so a real reference cannot borrow it.
    r'|/DATA/FOUND|/found/': "this file's own pattern: the definition of what to forbid, not a use of it",
    "python3 -m tools.class_counts": "guarded by a -f test in the same block and skipped with a printed reason; "
                                     "its home is EXT_POST_RUN, which the extension supplies",
}

# A MODULE THIS REPOSITORY INVOKES BUT DOES NOT CONTAIN. The patterns above find the extension BY
# NAME, and that is exactly what they cannot do here: `python3 -m tools.class_counts` names no
# extension, contains no "found", and resolves only from a workspace this repository does not own.
# It ran on every Gin launch and printed ModuleNotFoundError, reported by the experiment lane on
# 2026-09-10. So this check asks a different question -- is the module HERE? -- and needs no list of
# forbidden names to do it.
#
# `python` must appear before the `-m`, because "under -m the bare" is a sentence in a comment in
# replay_server.py and a check that flags English is a check somebody switches off.
EXTERNAL_M = re.compile(r"python[0-9.]*\s+(?:-\S+\s+)*-m\s+([A-Za-z_][A-Za-z0-9_.]*)")


def offenders(root: pathlib.Path):
    # TRACKED **AND** UNTRACKED-BUT-NOT-IGNORED. `ls-files` alone lists only tracked files, so a
    # BRAND-NEW file passes this check right up until it is committed -- and then fails on the next
    # clean checkout, in somebody else's clone. That happened: an example config added on
    # 2026-09-10 named an extension in two comments, passed here while it was still untracked, and
    # broke the check for the next person to clone. `-o --exclude-standard` closes it while still
    # honouring .gitignore, so a deployment's own uncommitted wiring is not flagged.
    files = subprocess.run(["git", "-C", str(root), "ls-files", "-c", "-o", "--exclude-standard"],
                           capture_output=True, text=True).stdout.split()
    out = []
    for rel in files:
        p = root / rel
        try:
            text = p.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            # EVERY MATCH ON THE LINE, NOT THE FIRST. `search` reports that a line matched, which
            # on a MINIFIED OR SINGLE-LINE FILE is one instance out of an unknown population:
            # envelopes.json is one line and held TWO references -- a module path under
            # `generated_by` and an absolute source path under `source_tree` -- and this printed
            # only the first of them. A fix guided by
            # that report would have left the other in place. Found by the ontology lane on
            # 2026-09-10, which grepped the whole file instead of trusting the line report.
            hits = [m.group(0) for m in FORBIDDEN.finditer(line)]
            if not hits:
                continue
            if any(a in line for a in ALLOWED):
                continue
            shown = line.strip()[:110]
            uniq = sorted(set(hits))
            if len(hits) > 1:
                shown = f"{len(hits)} references on this line {uniq}: {shown}"
            out.append((rel, n, shown))
    out.extend(external_modules(root, files))
    return out


def _importable(name: str) -> bool:
    """True when `name` resolves from the stdlib or an installed package.

    Run with cwd=/ ON PURPOSE: from inside the repository its own directories satisfy the import
    and every module would look available, which is the answer this check exists to distrust.
    """
    return subprocess.run([sys.executable, "-c",
                           "import importlib.util,sys;"
                           "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)", name],
                          cwd="/", capture_output=True).returncode == 0


def _in_repo(root: pathlib.Path, name: str) -> bool:
    """True when the repository itself carries the top-level module or package."""
    return (any(root.rglob(f"{name}/__init__.py"))
            or any(p for p in root.rglob(f"{name}.py") if ".git" not in p.parts))


def external_modules(root: pathlib.Path, files):
    seen, out = {}, []
    for rel in files:
        p = root / rel
        try:
            text = p.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            m = EXTERNAL_M.search(line)
            if not m or any(a in line for a in ALLOWED):
                continue
            top = m.group(1).split(".")[0]
            if top not in seen:
                seen[top] = _importable(top) or _in_repo(root, top)
            if not seen[top]:
                out.append((rel, n, f"invokes `{m.group(1)}`, which this repository does not "
                                    f"contain: {line.strip()[:70]}"))
    return out


def main() -> int:
    # RULE 77. THIS CHECK USED TO REPORT SUCCESS WHEN IT COULD SEE NOTHING. Exported with
    # `git archive` into a scratch directory it printed "no extension references in : the seam
    # holds" -- empty repository name, ZERO FILES SCANNED, rc 0 -- because `--show-toplevel` fails
    # outside a repository and `ls-files` then returns nothing. A boundary check that passes on an
    # empty file list is worse than no check: it is a green light nobody earned. Found by the
    # simulator lane on 2026-09-10 while verifying a tip before pushing it.
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                         cwd=pathlib.Path(__file__).parent)
    if top.returncode != 0 or not top.stdout.strip():
        print("REFUSED: not inside a git repository, so no file list can be built.\n"
              "  This check enumerates with `git ls-files`. Run it in the working tree, or in a\n"
              "  real git context -- `git worktree add --detach <tip>` -- never in an archive.")
        return 2
    root = pathlib.Path(top.stdout.strip())
    n_files = len(subprocess.run(["git", "-C", str(root), "ls-files", "-c", "-o",
                                  "--exclude-standard"],
                                 capture_output=True, text=True).stdout.split())
    if n_files == 0:
        print(f"REFUSED: {root} lists no files, so nothing was examined.")
        return 2
    bad = offenders(root)
    if not bad:
        print(f"no extension references in {root.name}: the seam holds "
              f"({n_files} files examined)")
        return 0
    print(f"{len(bad)} extension reference(s) — this repository must not name what extends it:\n")
    for rel, n, line in bad:
        print(f"  {rel}:{n}: {line}")
    print("\nMove it to the extension. The seam is EXT_ENV_FILE / EXT_ENV_PASS / EXT_MOUNTS /"
          "\nEXT_TREES / EXT_MOUNT_POINT / EXT_POLICY_JSON / EXT_STORE_REPAIR in live_run.sh.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
