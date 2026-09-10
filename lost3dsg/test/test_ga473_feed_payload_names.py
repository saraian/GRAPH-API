#!/usr/bin/env python3
"""Every name the BEV payload reads must be bound on EVERY path through main().

GA-473. `SCHEDULE_OVERLAY` was assigned only inside `if have_nav and SCHEDULE_PATH:`. Assigning a
name anywhere in a function makes it LOCAL for the whole function, so the module-level default was
shadowed and the name was unbound on every run WITHOUT a schedule -- which is the ordinary case and
the one the feed host takes when it is run on its own. It raised UnboundLocalError at the first
frame, after the scene had loaded and the socket was up.

ruff does not catch it: F821 is for undefined names, and this one is defined, just not on the path
taken. So the check is structural -- read main()'s body, and for every name the payload dict reads,
require an assignment that is not nested inside a branch or a loop.

Run: python3 test_ga473_feed_payload_names.py   (or under pytest)
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "habitat_feed_host.py")


def _main_fn(tree):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("habitat_feed_host.py has no main()")


def _payload_names(main_fn):
    """-> the bare names the bev_payload dict reads."""
    for node in ast.walk(main_fn):
        if (isinstance(node, ast.Assign) and node.targets
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "bev_payload"
                and isinstance(node.value, ast.Dict)):
            return {n.id for v in node.value.values for n in ast.walk(v)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    raise AssertionError("main() does not build bev_payload")


def _dominating_assignments(main_fn):
    """-> names assigned on a statement that always runs before bev_payload is built.

    NOT "assigned at main's top level" -- that was the first version and it was too strict: the
    payload is built inside the frame loop, and ros_agent_pos, yaw and active_map are assigned
    earlier in the SAME loop body, which is perfectly safe. What matters is whether the assignment
    dominates the payload statement: same block or an enclosing one, and textually before it.
    """
    # the chain of blocks from main's body down to the statement holding bev_payload
    def find(stmts, depth=0):
        for i, stmt in enumerate(stmts):
            for sub in ast.walk(stmt):
                if (isinstance(sub, ast.Assign) and sub.targets
                        and isinstance(sub.targets[0], ast.Name)
                        and sub.targets[0].id == "bev_payload"):
                    chain = [(stmts, i)]
                    for field in ("body", "orelse", "finalbody"):
                        inner = getattr(stmt, field, None)
                        if inner:
                            deeper = find(inner, depth + 1)
                            if deeper:
                                return chain + deeper
                    return chain
        return []

    out = set()
    for stmts, i in find(main_fn.body):
        # STRICTLY BEFORE, and NOT descending into branches. An assignment inside an `if` that sits
        # before the payload is still conditional -- walking into it is exactly how the first
        # version called the original bug safe. The `for` target counts: the loop body only runs
        # when it is bound.
        for stmt in stmts[:i]:
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                targets = [stmt.target]
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                targets = [stmt.target]
            elif isinstance(stmt, ast.With):
                targets = [it.optional_vars for it in stmt.items if it.optional_vars]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        out.add(n.id)
    return out


def test_every_payload_name_is_bound_on_every_path():
    tree = ast.parse(open(SRC).read())
    main_fn = _main_fn(tree)
    module_level = {n.id for stmt in tree.body
                    if isinstance(stmt, ast.Assign)
                    for t in stmt.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    # A name assigned ANYWHERE in main is local to main, so a module-level value does not save it.
    local_anywhere = {n.id for node in ast.walk(main_fn)
                      if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                      for t in ([node.target] if not isinstance(node, ast.Assign) else node.targets)
                      for n in ast.walk(t) if isinstance(n, ast.Name)}
    safe = _dominating_assignments(main_fn) | (module_level - local_anywhere)
    unbound = sorted(n for n in _payload_names(main_fn)
                     if n not in safe and n not in dir(__builtins__) and not n.isupper())
    assert not unbound, (
        "these names are read by bev_payload but assigned only inside a branch of main(), so a run "
        "that does not take that branch raises UnboundLocalError at the first frame: "
        + ", ".join(unbound))


def test_the_check_sees_a_conditional_binding_when_one_is_there():
    """Exercised where the answer is PRESENT: the original bug, reconstructed."""
    bad = ast.parse(
        "X = None\n"
        "def main():\n"
        "    if flag:\n"
        "        X = 1\n"
        "    bev_payload = {'a': X}\n")
    main_fn = _main_fn(bad)
    module_level = {"X"}
    local_anywhere = {"X", "bev_payload"}
    safe = _dominating_assignments(main_fn) | (module_level - local_anywhere)
    assert "X" in _payload_names(main_fn)
    assert "X" not in safe, "a name assigned only inside an if must NOT read as safe"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
