"""Extend DynamicGSG's colour book so a long tour can exceed 201 native objects.

DynamicGSG paints one colour per native object index and reads it as
``color_book[objects[i]['idx']]``.  Native indices are assigned monotonically and
never reused, so the 201-row ScanNet200 book is a hard ceiling: the run raises
``IndexError: list index out of range`` the moment a 202nd object is created.
Measured on the DGX, four Pack A scenes all died there at native index 197-199,
between frames 374 and 438, with only 109-139 objects alive at the time.

This writes a NEW file and never edits the upstream one.  The original rows are
copied byte for byte and keep their positions, so every index the upstream book
could serve resolves to exactly the same colour; only indices beyond its end
gain a value.  Added colours come from a deterministic bit-reversal walk of an
RGB lattice, and any colour already in the book is skipped, so entries stay
distinct -- the colour is an object-identity signal in the native feature
channel, not decoration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

HEADER = "R G B"


def read_rows(path: Path) -> tuple[str, list[str]]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty colour book: {path}")
    return lines[0], lines[1:]


def _triplet(row: str) -> tuple[int, int, int]:
    r, g, b = (float(value) for value in row.split())
    return int(round(r)), int(round(g)), int(round(b))


def _bit_reversed(value: int, bits: int) -> int:
    out = 0
    for _ in range(bits):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


def generate(existing: set[tuple[int, int, int]], count: int, step: int = 16):
    """Well-spread lattice colours in a deterministic, reproducible order."""
    side = 256 // step
    total = side ** 3
    bits = max(1, (total - 1).bit_length())
    added, seen = [], set(existing)
    for index in range(1 << bits):
        if len(added) == count:
            break
        scrambled = _bit_reversed(index, bits)
        if scrambled >= total:
            continue
        r = (scrambled % side) * step
        g = ((scrambled // side) % side) * step
        b = ((scrambled // (side * side)) % side) * step
        colour = (r, g, b)
        if colour in seen:
            continue
        seen.add(colour)
        added.append(colour)
    if len(added) != count:
        raise ValueError(f"Lattice with step {step} cannot supply {count} distinct colours")
    return added


def extend(source, output, entries=4096):
    source, output = Path(source), Path(output)
    header, rows = read_rows(source)
    if header.split() != HEADER.split():
        raise ValueError(f"Unexpected colour-book header: {header!r}")
    if entries <= len(rows):
        raise ValueError(f"{source} already has {len(rows)} entries; asked for {entries}")
    existing = {_triplet(row) for row in rows}
    added = generate(existing, entries - len(rows))
    lines = [header, *rows]
    lines += [" ".join(f"{float(value):.18e}" for value in colour) for colour in added]
    output.write_text("\n".join(lines) + "\n")
    # The prefix must survive verbatim, or every object built so far changes colour.
    written_header, written_rows = read_rows(output)
    if written_header != header or written_rows[:len(rows)] != rows:
        raise RuntimeError("Extended colour book altered the upstream prefix")
    if len({_triplet(row) for row in written_rows}) != len(written_rows):
        raise RuntimeError("Extended colour book contains a duplicate colour")
    return {"source": str(source), "output": str(output),
            "upstream_entries": len(rows), "total_entries": len(written_rows),
            "added_entries": len(added),
            "prefix_is_verbatim": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Upstream colour book, never modified")
    parser.add_argument("--output", required=True)
    parser.add_argument("--entries", type=int, default=4096)
    args = parser.parse_args(argv)
    import json
    print(json.dumps(extend(args.source, args.output, args.entries), indent=2))


if __name__ == "__main__":
    main()
