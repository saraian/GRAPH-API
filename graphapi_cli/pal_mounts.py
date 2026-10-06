"""Resolve the selected checkout through an existing private container's mounts."""
from pathlib import Path
import json
import subprocess
import sys


def checkout_source(root, mounts):
    root = Path(root).resolve()
    candidates = []
    for mount in mounts:
        if mount.get("Type") != "bind":
            continue
        source = Path(mount["Source"]).resolve()
        try:
            relative = root.relative_to(source)
        except ValueError:
            continue
        destination = Path(mount["Destination"]) / relative
        # A nested mount can hide the selected checkout even if an ancestor
        # bind would otherwise make it available.
        hidden = False
        for other in mounts:
            other_destination = Path(other["Destination"])
            if other_destination == Path(mount["Destination"]):
                continue
            required_trees = (destination / "lost3dsg", destination / "graphapi_cli")
            overlaps_sources = any(tree.is_relative_to(other_destination) or other_destination.is_relative_to(tree)
                                   for tree in required_trees)
            if overlaps_sources and other_destination.is_relative_to(Path(mount["Destination"])):
                hidden = True
                break
        if not hidden:
            candidates.append((len(source.parts), str(destination / "lost3dsg")))
    if not candidates:
        raise ValueError("private PAL container does not mount this checkout; recreate it with this repository mounted at /graph_api")
    return max(candidates)[1]


def main():
    container, root = sys.argv[1:]
    result = subprocess.run(["docker", "inspect", container, "--format", "{{json .Mounts}}"], capture_output=True, text=True, check=True)
    print(checkout_source(root, json.loads(result.stdout)))


if __name__ == "__main__":
    main()
