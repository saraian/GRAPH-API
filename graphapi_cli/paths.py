"""Resolve host symlinks without escaping the planner's host filesystem view."""
from collections import deque
from pathlib import Path

from .configuration import ConfigurationError


def host_path(path, host_root=None):
    """Return a host absolute path, including when the host is mounted at /host.

    Absolute symlink targets belong to the host namespace, not the planner's
    container. Missing suffixes are allowed for output directories.
    """
    path = Path(path)
    if host_root is None:
        return path.resolve()
    pending = deque(path.parts[1:])
    resolved = Path("/")
    links = 0
    while pending:
        part = pending.popleft()
        if part == "..":
            resolved = resolved.parent
            continue
        if part == ".":
            continue
        candidate = resolved / part
        visible = Path(host_root) / str(candidate).lstrip("/")
        if visible.is_symlink():
            links += 1
            if links > 40:
                raise ConfigurationError(f"too many symlinks while resolving {path}")
            target = visible.readlink()
            if target.is_absolute():
                resolved = Path("/")
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            resolved = candidate
    return resolved


def output_directory(path):
    """Create a writable output target, preserving a user-supplied symlink."""
    target = Path(path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target
