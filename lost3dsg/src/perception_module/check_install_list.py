#!/usr/bin/env python3
"""Does every module perception_module imports from itself appear in CMakeLists.txt?

GA-128. `perception_2.py` imported `detection_archive`, which was not in the install list,
so the module existed in `src/` and was absent from `lib/lost3dsg` — the tree `ros2 run`
actually loads. The node would have passed the preflight gate and then died at startup with
`ModuleNotFoundError`, which is a far worse place to find it than a probe that names the
module.

THE UNDERLYING PROBLEM IS NOT THE MISSING LINE, IT IS THAT THE LIST IS HAND-MAINTAINED and
nothing compares it to the imports. Anyone adding a module to this package gets working code
for themselves and a broken install for everyone else, and the gap only shows up in a run.
So: compare the two mechanically, exit non-zero on a gap, and let it run in the gate.

An import that works in `src/` says nothing about what the install tree can load. That is
the same lesson as a7 (executed tree vs mounted tree) in a third place.

Usage:  python3 check_install_list.py [package_root]
Exit:   0 = every locally-imported module is installed; 1 = at least one is not.
"""

import ast
import os
import re
import sys

# Not imported by the package at runtime: harnesses and the stub loader.
EXCLUDE = {"rosstub", "check_install_list"}


def local_modules(pkg_dir):
    return {f[:-3] for f in os.listdir(pkg_dir) if f.endswith(".py")}


def installed_modules(cmake_path):
    txt = open(cmake_path).read()
    # ignore commented-out lines so a disabled entry never reads as installed
    live = "\n".join(ln for ln in txt.splitlines() if not ln.strip().startswith("#"))
    return set(re.findall(r"src/perception_module/(\w+)\.py", live))


def imports_of(path):
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


def audit(root):
    pkg = os.path.join(root, "src", "perception_module")
    cmake = os.path.join(root, "CMakeLists.txt")
    if not os.path.isdir(pkg) or not os.path.isfile(cmake):
        print(f"not a lost3dsg package root: {root}")
        return 2

    local = local_modules(pkg)
    installed = installed_modules(cmake)
    gaps = {}
    for fname in sorted(os.listdir(pkg)):
        if not fname.endswith(".py"):
            continue
        stem = fname[:-3]
        if stem in EXCLUDE or stem.startswith("test_"):
            continue
        for mod in imports_of(os.path.join(pkg, fname)):
            if mod in local and mod not in installed and mod not in EXCLUDE:
                gaps.setdefault(mod, set()).add(fname)

    if gaps:
        print(f"{len(gaps)} module(s) imported but NOT INSTALLED — the node will die at "
              f"startup with ModuleNotFoundError:")
        for mod, users in sorted(gaps.items()):
            print(f"  {mod}.py   imported by {', '.join(sorted(users))}")
        print(f"\nFix: add src/perception_module/<name>.py to the install(PROGRAMS ...) "
              f"list in {cmake}")
        return 1

    print(f"install list OK — {len(local)} modules present, "
          f"{len(installed)} installed, every locally-imported one covered")
    return 0


def demo():
    """Self-check: the audit must FAIL on a package whose list is missing an import."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="cil_")
    try:
        pkg = os.path.join(tmp, "src", "perception_module")
        os.makedirs(pkg)
        open(os.path.join(pkg, "alpha.py"), "w").write("import beta\n")
        open(os.path.join(pkg, "beta.py"), "w").write("X = 1\n")
        cm = os.path.join(tmp, "CMakeLists.txt")

        open(cm, "w").write("install(PROGRAMS\n  src/perception_module/alpha.py\n)\n")
        assert audit(tmp) == 1, "a missing module must be reported"
        print("  detects a missing module")

        open(cm, "w").write("install(PROGRAMS\n  src/perception_module/alpha.py\n"
                            "  src/perception_module/beta.py\n)\n")
        assert audit(tmp) == 0, "a complete list must pass"
        print("  passes a complete list")

        # a COMMENTED-OUT entry must not count as installed
        open(cm, "w").write("install(PROGRAMS\n  src/perception_module/alpha.py\n"
                            "  # src/perception_module/beta.py\n)\n")
        assert audit(tmp) == 1, "a commented-out entry must not read as installed"
        print("  a commented-out entry does NOT count as installed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\ncheck_install_list self-check OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        demo()
    else:
        root = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        sys.exit(audit(root))
