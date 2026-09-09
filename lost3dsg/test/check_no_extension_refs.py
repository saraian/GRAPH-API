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
}


def offenders(root: pathlib.Path):
    files = subprocess.run(["git", "-C", str(root), "ls-files"],
                           capture_output=True, text=True).stdout.split()
    out = []
    for rel in files:
        p = root / rel
        try:
            text = p.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if not FORBIDDEN.search(line):
                continue
            if any(a in line for a in ALLOWED):
                continue
            out.append((rel, n, line.strip()[:110]))
    return out


def main() -> int:
    root = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True,
                                       cwd=pathlib.Path(__file__).parent).stdout.strip())
    bad = offenders(root)
    if not bad:
        print(f"no extension references in {root.name}: the seam holds")
        return 0
    print(f"{len(bad)} extension reference(s) — this repository must not name what extends it:\n")
    for rel, n, line in bad:
        print(f"  {rel}:{n}: {line}")
    print("\nMove it to the extension. The seam is EXT_ENV_FILE / EXT_ENV_PASS / EXT_MOUNTS /"
          "\nEXT_TREES / EXT_MOUNT_POINT / EXT_POLICY_JSON / EXT_STORE_REPAIR in live_run.sh.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
