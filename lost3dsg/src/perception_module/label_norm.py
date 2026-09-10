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
# W5. "buses" and "lenses" are hand entries for the SAME reason "glasses"/"vases" already
# were: they are true s-stems + "es", and the "-se"+s reading below would take "buses" to
# "buse". The vocabulary is indoor-scene nouns; a rule cannot be spelled for both
# "houses"/"bookcases" and "buses" without a dictionary, so the exceptions are explicit.
IRREGULAR = {
    "shelves": "shelf", "leaves": "leaf", "knives": "knife", "loaves": "loaf",
    "people": "person", "children": "child", "feet": "foot", "teeth": "tooth",
    "mice": "mouse", "boxes": "box", "dishes": "dish", "benches": "bench",
    "couches": "couch", "glasses": "glass", "vases": "vase", "mattresses": "mattress",
    "buses": "bus", "lenses": "lens",
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
        elif last.endswith(("ches", "shes", "xes", "zes")):
            head = last[:-2]             # benches -> bench, boxes -> box
        else:
            # W5. NO "ses" arm: "-ses" is ambiguous by spelling alone and the old 2-char
            # strip mangled every "-se"+s stem (houses->"hous", bookcases->"bookcas",
            # suitcases->"suitcas"), asking the detector for a non-word and losing the
            # object. The "-se" reading is kept -- it is the common furniture-vocabulary
            # case -- and the genuine s-stems (buses, lenses; glasses, vases already) are
            # hand entries in IRREGULAR, the precedent this file already used.
            head = last[:-1]             # chairs -> chair, houses -> house, bookcases -> bookcase
    return (w[: -len(last)] + head) if len(w) > len(last) else head


def normalise(label):
    """Lower-case, strip punctuation and articles, collapse whitespace, singularise."""
    s = str(label or "").lower().strip()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"^\s*(a|an|the)\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return singularise(s) if s else s


# W4. The table above stores the forms a PERSON writes (plural "chest of drawers"); the
# labels arriving here are NORMALISED (head noun singularised: "chest of drawer"). Exact
# membership then missed the only plural-headed member, and `dresser` + `chest of drawers`
# both reached the detector -- the exact duplicate boxes this module exists to prevent.
# Normalised-vs-normalised, computed once here.
_SYNONYMS_NORMALISED = tuple(frozenset(normalise(m) for m in group) for group in SYNONYMS)


def _synonym_key(label):
    """-> a stable key shared by every member of a synonym class, else the label itself."""
    for i, group in enumerate(_SYNONYMS_NORMALISED):
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

    # W4. THE ONE PLURAL-HEADED MEMBER. Measured defect: clean(['dresser','chest of
    # drawers']) -> (['dresser','chest of drawer'], []) -- the group never matched after
    # the head noun was singularised, and both terms reached the detector.
    k, d = clean(["dresser", "chest of drawers"])
    assert k == ["dresser"], k
    assert d and d[0][0] == "chest of drawers" and "synonym" in d[0][2], d
    k, _ = clean(["chest of drawers", "dresser"])
    assert k == ["chest of drawer"], "first occurrence wins, whichever it is"
    print("  ok  plural-headed synonym member collapses (dresser / chest of drawers)")

    # W5. "-ses" no longer strips two chars off "-se"+s stems.
    for plural, singular in (("houses", "house"), ("bookcases", "bookcase"),
                             ("suitcases", "suitcase"), ("buses", "bus"),
                             ("benches", "bench"), ("boxes", "box")):
        assert singularise(plural) == singular, f"{plural} -> {singularise(plural)!r}"
    k, _ = clean(["bookcases", "bookcase", "Bookcase"])
    assert k == ["bookcase"], k
    print("  ok  houses/bookcases/suitcases survive; buses via the hand entry; benches/boxes still strip")

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
