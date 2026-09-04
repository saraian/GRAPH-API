import os
import re

import numpy as np
import webcolors
from config import CFG

EMBEDDING_MODEL = "text-embedding-3-small"
_EMB_CACHE = {}


class SemanticEmbedder:
    """High-accuracy semantic similarity encoder using sentence-transformers or word2vec."""

    def __init__(self):
        self.st_model = None
        self.w2v_model = None

        # 1. Try SentenceTransformer (high accuracy, handles any open-vocabulary phrase)
        try:
            from sentence_transformers import SentenceTransformer
            cache_folder = os.environ.get("HF_HOME") or "/models/hf"
            self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=cache_folder)
            print("[NLP] Loaded SentenceTransformer (all-MiniLM-L6-v2) for semantic matching ✅")
            return
        except Exception:
            pass

        # 2. Fallback to Word2Vec KeyedVectors
        try:
            from gensim.models import KeyedVectors
            path = CFG.get("embedding", {}).get("word2vec_path", "")
            limit = CFG.get("embedding", {}).get("word2vec_limit", 200000)
            if path and os.path.exists(path):
                self.w2v_model = KeyedVectors.load_word2vec_format(path, binary=True, limit=limit)
                print(f"[NLP] Loaded Word2Vec from {path} ✅")
        except Exception as exc:
            print(f"[NLP] Word2Vec fallback notice: {exc}")

    def encode(self, text: str):
        text = text.lower().strip()
        if text in _EMB_CACHE:
            return _EMB_CACHE[text]

        if self.st_model is not None:
            try:
                emb = self.st_model.encode(text)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                _EMB_CACHE[text] = emb
                return emb
            except Exception:
                pass

        if self.w2v_model is not None:
            words = text.replace("_", " ").split()
            vectors = [self.w2v_model[w] for w in words if w in self.w2v_model]
            if vectors:
                emb = np.mean(vectors, axis=0)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                _EMB_CACHE[text] = emb
                return emb

        return None

    def similarity(self, text1: str, text2: str) -> float:
        t1, t2 = text1.lower().strip(), text2.lower().strip()
        if t1 == t2:
            return 1.0
        v1 = self.encode(t1)
        v2 = self.encode(t2)
        if v1 is None or v2 is None:
            return 0.0
        return max(0.0, float(np.dot(v1, v2)))


world2vec = SemanticEmbedder()


def semantic_similarity(word2vec_model, word1: str, word2: str) -> float:
    """Calculate semantic similarity between two words/phrases."""
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()

    if word1 == word2:
        return 1.0

    # GA-02: a missing model is a missing COMPONENT, not evidence of difference.
    # Returning 0.0 here answered "these two things are completely dissimilar" to the
    # question "did the embedding model load?". This function supplies the label term of
    # the LSF (always present, weight alpha), the material term, and the colour fallback
    # -- three routes by which a model that never loaded became a confident assertion
    # that everything differs from everything. Rule 14: crash, and say which component.
    if word2vec_model is None:
        raise ValueError(
            "semantic_similarity: no embedding model. This is a missing component, not a "
            "similarity of 0.0 -- the label, material and colour-fallback terms of the LSF "
            "all route through here, so answering 0.0 turns a failed load into evidence "
            "that every pair of objects is unrelated.")

    if isinstance(word2vec_model, SemanticEmbedder):
        return word2vec_model.similarity(word1, word2)

    def get_phrase_vector(phrase):
        words = phrase.replace('_', ' ').split()
        vectors = []
        for word in words:
            if word in word2vec_model:
                vectors.append(word2vec_model[word])
        if vectors:
            return np.mean(vectors, axis=0)
        return None

    vec1 = get_phrase_vector(word1)
    vec2 = get_phrase_vector(word2)

    # This 0.0 is KEPT and it is a different statement from the one removed above: both
    # phrases were looked up and neither is in the vocabulary, so the model has no
    # evidence relating them. That is a real answer about the words. The deleted handler
    # returned the same number when the COMPUTATION CRASHED, which was not.
    if vec1 is None or vec2 is None:
        return 0.0

    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return max(0.0, float(similarity))



def color_name_to_rgb(color_name: str) -> tuple:
    """
    Convert color name to normalized RGB values [0-1].

    Args:
        color_name: Color name (e.g. 'red', 'blue', 'dark green')

    Returns:
        tuple: (r, g, b) normalized [0-1], or None if color is not recognized
    """
    if not color_name or color_name.strip() == "":
        return None

    color_name = color_name.lower().strip()

    try:
        # Try with webcolors (supports standard CSS names)
        rgb = webcolors.name_to_rgb(color_name)
        return (rgb.red / 255.0, rgb.green / 255.0, rgb.blue / 255.0)
    except Exception:
        pass

    # Fallback: basic color dictionary
    BASIC_COLORS = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
        'beige': (245, 245, 220),
        'tan': (210, 180, 140),
        'silver': (192, 192, 192),
        'gold': (255, 215, 0)
    }

    if color_name in BASIC_COLORS:
        rgb = BASIC_COLORS[color_name]
        return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

    # Se non riconosciuto, ritorna None
    return None


def color_similarity_rgb(color1: str, color2: str, word2vec_model=None) -> float:
    """
    Similarity between two colours, by Euclidean RGB distance.
    Se i colori non sono riconosciuti, fa fallback a word2vec.

    Args:
        color1, color2: Nomi dei colori da confrontare
        word2vec_model: Modello word2vec per fallback (opzionale)

    Returns:
        float: Similarità [0, 1] dove 1 = colori identici, 0 = colori molto diversi
    """
    # Se sono esattamente uguali
    if color1.lower().strip() == color2.lower().strip():
        return 1.0

    # Converti i nomi in RGB
    rgb1 = color_name_to_rgb(color1)
    rgb2 = color_name_to_rgb(color2)

    # Se uno dei due non è riconosciuto, usa fallback word2vec
    if rgb1 is None or rgb2 is None:
        if word2vec_model is not None:
            # Fallback a similarità semantica word2vec
            return semantic_similarity(word2vec_model, color1, color2)
        else:
            return 0.0

    # Euclidean distance in the normalised RGB space
    # Largest possible distance = sqrt(3), white to black
    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
    max_distance = np.sqrt(3.0)  # sqrt(1^2 + 1^2 + 1^2)

    # Distance to similarity: distance 0 gives similarity 1
    similarity = 1.0 - (distance / max_distance)

    return max(0.0, min(1.0, similarity))

def get_embedding(model, text):
    """
    Restituisce embedding vettoriale tramite Word2Vec (calcolando la media delle parole).
    Sostituisce OpenAI per funzionare 100% offline e senza API key.
    """
    try:
        if not text or not _known(text):
            return None  # no description = no evidence (see lost_similarity)

        # GA-90. The live model is a SemanticEmbedder, which exposes encode()/similarity()
        # and implements NEITHER __contains__ NOR __getitem__. The code below uses
        # `w in model` and `model[w]`, so every call raised
        #     TypeError: argument of type 'SemanticEmbedder' is not iterable
        # on the FIRST word, the handler at the bottom printed it, and this returned None.
        #
        # So every description embedding was None WITH THE SENTENCETRANSFORMER LOADED AND
        # WORKING. lost_similarity drops a term with no evidence on either side and
        # renormalises, so the description term -- weight 0.50, the LARGEST of the four --
        # has never once entered an association decision. Label 0.05 / colour 0.30 /
        # material 0.15 renormalise to 0.10 / 0.60 / 0.30: colour has been deciding
        # associations at double its configured weight.
        #
        # Era (rule 34): the in/[] pattern is pre-FOUND and was correct for the model it
        # had; the None-state dates to 683a2b0, 2026-08-26, when the SemanticEmbedder
        # migration fixed semantic_similarity's call site and missed this one. That
        # function has had this isinstance branch ever since. This one did not.
        if isinstance(model, SemanticEmbedder):
            return model.encode(text)

        # Below here `model` is a real mapping-style KeyedVectors, where `in` and `[]`
        # are the correct protocol.
        # Estrae le singole parole dalla descrizione
        words = re.findall(r'\w+', text)
        
        vectors = []
        for w in words:
            # Word2Vec Google News è case-sensitive. Cerchiamo la parola esatta o in minuscolo
            if w in model:
                vectors.append(model[w])
            elif w.lower() in model:
                vectors.append(model[w.lower()])
        
        if not vectors:
            # When no word of the description is in the Word2Vec vocabulary
            return None
            
        # Calcola il vettore medio dell'intera frase
        sentence_vector = np.mean(vectors, axis=0)
        
        # Normalizza il vettore (molto importante per calcolare correttamente le similarità successive)
        norm = np.linalg.norm(sentence_vector)
        if norm > 0:
            sentence_vector = sentence_vector / norm
            
        return sentence_vector

    except Exception as e:
        # GA-90. This printed to stdout and returned None, and lost_similarity reads None
        # as "this object has no description" -- so a TYPE ERROR was indistinguishable
        # from an undescribed object, which is how the defect above survived from
        # 2026-08-26 to now. A None from this function is again only ever the two
        # explicit "no evidence" returns above.
        raise RuntimeError(f"get_embedding failed for {text!r} with model {type(model).__name__}") from e

def _comparable_embeddings(a, b):
    """Can these two embeddings actually be compared? GA-171.

    `desc1 is not None and desc2 is not None` was not enough. An EMPTY ARRAY is not None --
    it passed the guard and crashed inside the dot product with
    `shapes (384,) and (0,) not aligned`, because `_serialize_embedding` encodes absence as
    `[]` and the decoder turned that back into a real array of length zero.

    The root cause is fixed at the boundary (object_services.normalise_embedding), but this
    guard states the requirement in the place that depends on it: an embedding is EVIDENCE
    only if it exists, has length, and has the SAME length as the one it is compared with.
    Two vectors of different widths are not a low similarity -- they are two things that
    cannot be compared, and GA-101's whole design is that such a term is DROPPED and the
    remaining weights renormalised, never scored.
    """
    if a is None or b is None:
        return False
    try:
        la, lb = len(a), len(b)
    except TypeError:
        return False
    return la > 0 and la == lb


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)



def lost_similarity(word2vec_model, label1, label2, color1, color2, material1, material2, desc1, desc2):
    """
    Calculate overall similarity between two objects using:
    lost_similarity = alpha*label_sim + beta*color_sim + gamma*material_sim + delta*desc_sim

    MODIFIED v7: Optimized weights to give more importance to description
    - alpha (label):       0.25 (25%)
    - beta (color):        0.20 (20%) - USES RGB DISTANCE instead of word2vec
    - gamma (material):    0.15 (15%) - reduced because "paper" matched too much
    - delta (description): 0.40 (40%) - increased to distinguish similar objects

    Args:
        word2vec_model: Word2vec model for semantic similarity
        label1, label2: Labels of the two objects
        color1, color2: Colors of the two objects (color name strings)
        material1, material2: Materials of the two objects
        desc1, desc2: Descriptions of the two objects (embeddings)

    Returns:
        float: Overall similarity [0, 1]
    """
    return lost_similarity_detailed(word2vec_model, label1, label2, color1, color2,
                                    material1, material2, desc1, desc2)[0]


def lost_similarity_detailed(word2vec_model, label1, label2, color1, color2,
                             material1, material2, desc1, desc2):
    """As lost_similarity, but also reports WHAT WAS ACTUALLY MEASURED.

    GA-101. The renormalisation below is correct as far as it goes and its comment is
    honest about the bug it removed. What it does not say is that dropping the unmeasurable
    terms leaves the DIVISOR equal to the surviving weight, so when colour, material and
    description are all absent the score is the label term alone -- and two identical label
    strings short-circuit to 1.0 inside semantic_similarity before any model is consulted.

    Measured against the live config:
        same label, nothing else measured      -> 1.0000   (merge gate is 0.9250: MERGES)
        same label, colours disagree           -> 0.1429
        different labels, nothing else         -> 0.0000

    So ADDING EVIDENCE LOWERS THE SCORE. The least-informed pair in the map is the highest
    scoring one, and the gate is most confident exactly where it knows least. That is why
    raising the threshold cannot help: the pairs it is meant to catch sit ABOVE the ones
    with real evidence.

    A bare float cannot express the difference between "four axes agreed" and "nothing was
    comparable", so the caller cannot refuse on it. This returns the count as well, rather
    than a sentinel score, deliberately: a magic value for "no evidence" is the shape that
    produced this defect in the first place.

    Returns:
        (score, evidence) where evidence is
        {"label": True, "color": bool, "material": bool, "description": bool,
         "optional_count": 0..3}
        `label` is always True -- it is the one term with no evidence gate -- and
        `optional_count` counts only the three that can be absent.
    """
    alpha = CFG["similarity"]["label"]
    beta = CFG["similarity"]["color"]
    gamma = CFG["similarity"]["material"]
    delta = CFG["similarity"]["description"]

    evidence = {"label": True, "color": False, "material": False, "description": False}

    # "unknown"/empty attributes are ABSENCE of evidence, not agreement. Before,
    # unknown==unknown scored 1.0 on color, material and description, so any two
    # undescribed objects reached 0.95 > SIM_THRESHOLD and every detection merged into the
    # first object in memory. Terms without evidence on both sides are dropped and the
    # remaining weights renormalised -- see the GA-101 note above for what that leaves.
    terms = [(alpha, semantic_similarity(word2vec_model, label1, label2))]
    if _known(color1) and _known(color2):
        terms.append((beta, color_similarity_rgb(color1, color2, word2vec_model)))
        evidence["color"] = True
    if _known(material1) and _known(material2):
        terms.append((gamma, semantic_similarity(word2vec_model, material1, material2)))
        evidence["material"] = True
    if _comparable_embeddings(desc1, desc2):
        terms.append((delta, cosine_similarity(desc1, desc2)))
        evidence["description"] = True

    evidence["optional_count"] = sum(
        1 for k in ("color", "material", "description") if evidence[k])

    weight = sum(w for w, _ in terms)
    score = sum(w * s for w, s in terms) / weight if weight > 0 else 0.0
    return score, evidence


def _known(value):
    return bool(value) and str(value).strip().lower() not in ("unknown", "none", "n/a")
