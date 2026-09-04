"""Lemmatise and de-duplicate the open-vocabulary detector's label list. GA-285.

WHY. The VLM proposes a label list per frame and the detector is asked for EVERY term in it.
When the list contains two names for one thing, the detector dutifully finds both, on the same
pixels, in every frame. Measured on run 20260903_123748: four `doorway` boxes nested inside
`door` boxes at IoS = 1.00 -- perfect containment -- with near-identical scores (0.45/0.46,
0.53/0.47, 0.57/0.56, 0.53/0.57).

NMS CANNOT CLEAN THIS UP, and that is the point. `apply_nms` is deliberately CLASS-AWARE, and
its own comment says why: "This prevents removing a 'book' just because it overlaps with a
'table'." That reasoning is correct for book/table and wrong for door/doorway -- the second
pair are two names WE handed the detector in the same request. The duplicates must not be
created; they cannot be suppressed afterwards without also suppressing genuine overlaps.

NO DEPENDENCY, BY REQUIREMENT. This must work in an all-local setup with no network and no
optional data mount, so it uses rules and a small explicit table rather than WordNet or a
lemmatiser package. The rules cover English noun plurals as they appear in furniture
vocabularies; they are not a general-purpose lemmatiser and do not pretend to be.

FIRST OCCURRENCE WINS. The VLM's own ordering is the only ranking available, and inventing a
preference between two names it offered would be a modelling claim made in a utility function.
The dropped term is returned alongside so a caller can log what was collapsed rather than
having terms disappear silently.

    python3 label_norm.py --selfcheck
"""
import re
import sys

# Irregular plurals that the rules below would get wrong. Small on purpose: every entry is a
# word an indoor detector actually emits.
IRREGULAR = {
    "shelves": "shelf", "leaves": "leaf", "knives": "knife", "loaves": "loaf",
    "people": "person", "children": "child", "feet": "foot", "teeth": "tooth",
    "mice": "mouse", "boxes": "box", "dishes": "dish", "benches": "bench",
    "couches": "couch", "glasses": "glass", "vases": "vase", "mattresses": "mattress",
}

# Words ending in 's' that are NOT plurals. Without this, "glass" becomes "glas".
NOT_PLURAL = {
    "glass", "grass", "dress", "mattress", "class", "gas", "bass", "compass",
    "canvas", "iris", "lens", "chess", "moss", "cactus", "status", "bus",
}

# SYNONYM CLASSES. Each set is names an open-vocabulary detector emits for ONE physical thing,
# so asking for more than one of them guarantees duplicate boxes. This is NOT a semantic claim
# that the words mean the same -- a doorway is an aperture and a door is a leaf, and GA-269
# turned on exactly that distinction. It is a claim about the DETECTOR: given both terms it
# returns the same pixels twice, measured at IoS 1.00.
#
# Kept short and explicit. A large automatic synonym source would collapse pairs nobody
# checked, and deciding that two labels denote one object is the alignment problem, which
# belongs in the ontology and not here.
SYNONYMS = [
    {"door", "doorway"},
    {"tv", "television", "tv screen", "flat screen tv"},
    {"couch", "sofa"},
    {"nightstand", "night stand", "bedside table"},
    {"dresser", "chest of drawers"},
    {"rug", "carpet", "area rug"},
    {"trash can", "garbage can", "waste bin", "trash bin"},
    {"picture", "picture frame", "framed picture"},
    {"light switch", "switch plate", "wall switch"},
    {"cabinet", "cupboard"},
    {"stove", "cooktop", "range"},
    {"faucet", "tap"},
]


def singularise(word):
    """English noun plural -> singular, by rule. Not a general lemmatiser."""
    w = word.strip().lower()
    if not w:
        return w
    last = w.rsplit(" ", 1)[-1]          # only the HEAD noun is pluralised: "wine glasses"
    head = IRREGULAR.get(last)
    if head is None:
        if last in NOT_PLURAL or not last.endswith("s") or last.endswith("ss"):
            head = last
        elif last.endswith("ies") and len(last) > 4:
            head = last[:-3] + "y"       # canopies -> canopy
        elif last.endswith(("ches", "shes", "xes", "zes", "ses")):
            head = last[:-2]             # benches -> bench
        else:
            head = last[:-1]             # chairs -> chair
    return (w[: -len(last)] + head) if len(w) > len(last) else head


def normalise(label):
    """Lower-case, strip punctuation and articles, collapse whitespace, singularise."""
    s = str(label or "").lower().strip()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"^\s*(a|an|the)\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return singularise(s) if s else s


def _synonym_key(label):
    """-> a stable key shared by every member of a synonym class, else the label itself."""
    for i, group in enumerate(SYNONYMS):
        if label in group:
            return f"__syn{i}"
    return label


def clean(labels):
    """[raw labels] -> (kept, dropped).

    `kept` preserves the caller's order; `dropped` is [(dropped, kept_instead, why)] so a
    caller can log what was collapsed. Nothing disappears without a record.
    """
    kept, dropped, seen_exact, seen_syn = [], [], set(), {}
    for raw in labels or []:
        n = normalise(raw)
        if not n:
            continue
        if n in seen_exact:
            dropped.append((str(raw), n, "duplicate after lemmatising"))
            continue
        key = _synonym_key(n)
        if key != n and key in seen_syn:
            dropped.append((str(raw), seen_syn[key], "synonym of a label already requested"))
            continue
        seen_exact.add(n)
        seen_syn[key] = n
        kept.append(n)
    return kept, dropped


def _selfcheck():
    k, d = clean(["door", "doorway"])
    assert k == ["door"], k
    assert d and d[0][1] == "door" and "synonym" in d[0][2], d
    print(f"  ok  door + doorway  -> {k}   (the measured IoS-1.00 duplicate)")

    k, _ = clean(["doorway", "door"])
    assert k == ["doorway"], "first occurrence wins, whichever it is"
    print("  ok  first occurrence wins, in either order")

    k, _ = clean(["chairs", "chair", "Chair", "  chair  "])
    assert k == ["chair"], k
    print(f"  ok  plurals/case/whitespace collapse -> {k}")

    k, _ = clean(["shelves", "shelf"])
    assert k == ["shelf"], k
    k, _ = clean(["wine glasses"])
    assert k == ["wine glass"], k
    print("  ok  irregular plurals: shelves->shelf, wine glasses->wine glass")

    # THE ONE THAT BREAKS A NAIVE 's' RULE.
    for w in ("glass", "mattress", "compass"):
        assert singularise(w) == w, w
    print("  ok  glass/mattress/compass are not treated as plurals")

    k, _ = clean(["television", "tv", "TV screen"])
    assert k == ["television"], k
    k, _ = clean(["couch", "sofa", "chair"])
    assert k == ["couch", "chair"], k
    print("  ok  synonym classes collapse; unrelated labels survive")

    # DISTINCT THINGS MUST SURVIVE. Collapsing these would lose real objects.
    k, _ = clean(["table", "desk", "nightstand", "shelf", "cabinet", "chair", "stool"])
    assert len(k) == 7, k
    print(f"  ok  seven distinct furniture terms all survive ({len(k)})")

    k, _ = clean(["the table", "a chair", "  ", None, "", "door!"])
    assert k == ["table", "chair", "door"], k
    print("  ok  articles, punctuation, empties and None handled")

    k, d = clean([])
    assert k == [] and d == []
    print("  ok  empty input -> empty output, no crash")
    print("  label_norm selfcheck OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck() if "--selfcheck" in sys.argv or True else 0)
