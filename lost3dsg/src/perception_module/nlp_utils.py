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

    if word2vec_model is None:
        return 0.0

    if isinstance(word2vec_model, SemanticEmbedder):
        return word2vec_model.similarity(word1, word2)

    try:
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

        if vec1 is None or vec2 is None:
            return 0.0

        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return max(0.0, float(similarity))

    except Exception:
        return 0.0



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
    Calcola la similarità tra due colori usando la distanza RGB euclidea.
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

    # Calcola distanza euclidea nello spazio RGB normalizzato
    # Distanza massima possibile = sqrt(3) (da bianco a nero)
    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
    max_distance = np.sqrt(3.0)  # sqrt(1^2 + 1^2 + 1^2)

    # Converti distanza in similarità: 0 distanza = 1 similarità
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
            # Se nessuna parola della descrizione è nel vocabolario di Word2Vec
            return None
            
        # Calcola il vettore medio dell'intera frase
        sentence_vector = np.mean(vectors, axis=0)
        
        # Normalizza il vettore (molto importante per calcolare correttamente le similarità successive)
        norm = np.linalg.norm(sentence_vector)
        if norm > 0:
            sentence_vector = sentence_vector / norm
            
        return sentence_vector

    except Exception as e:
        print(f"Error during Word2Vec embedding creation: {e}")
        print("\n\n\n") 
        return None

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
    alpha = CFG["similarity"]["label"]
    beta = CFG["similarity"]["color"]
    gamma = CFG["similarity"]["material"]
    delta = CFG["similarity"]["description"]

    # FIX: "unknown"/empty attributes are ABSENCE of evidence, not agreement.
    # Before, unknown==unknown scored 1.0 on color, material and description,
    # so any two undescribed objects reached 0.95 > SIM_THRESHOLD and every
    # detection merged into the first object in memory. Terms without evidence
    # on both sides are dropped and the remaining weights renormalised.
    terms = [(alpha, semantic_similarity(word2vec_model, label1, label2))]
    if _known(color1) and _known(color2):
        terms.append((beta, color_similarity_rgb(color1, color2, word2vec_model)))
    if _known(material1) and _known(material2):
        terms.append((gamma, semantic_similarity(word2vec_model, material1, material2)))
    if desc1 is not None and desc2 is not None:
        terms.append((delta, cosine_similarity(desc1, desc2)))

    weight = sum(w for w, _ in terms)
    return sum(w * s for w, s in terms) / weight if weight > 0 else 0.0


def _known(value):
    return bool(value) and str(value).strip().lower() not in ("unknown", "none", "n/a")
