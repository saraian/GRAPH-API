"""
query_map_semantic.py
----------------
Interroga la mappa temporale con linguaggio naturale tramite VERO Semantic Search.
Nessun dizionario hardcoded: confronta la domanda vettorializzata con 
[label + color + description] degli oggetti nel DB usando Word2Vec.
"""

import sys
import re
import sqlite3
import signal
import os
import numpy as np
from pathlib import Path

# ── WORD2VEC & EMBEDDINGS ─────────────────────────────────────────────────────

_W2V_MODEL = None

def load_w2v(path: str):
    global _W2V_MODEL
    if _W2V_MODEL is not None:
        return _W2V_MODEL
    try:
        from gensim.models import KeyedVectors
        print(f"[SEMANTIC] Carico word2vec da {path} … (~1 min)")
        _W2V_MODEL = KeyedVectors.load_word2vec_format(path, binary=True)
        print("[SEMANTIC] word2vec pronto ✅\n")
    except Exception as e:
        print(f"[SEMANTIC] ⚠️ word2vec non disponibile: {e}")
        sys.exit(1)
    return _W2V_MODEL

def get_sentence_embedding(model, text: str):
    """Calcola il vettore medio di una frase usando Word2Vec."""
    if not text or not model:
        return None
    
    words = re.findall(r'\w+', text.lower())
    vectors = []
    
    for w in words:
        if w in model:
            vectors.append(model[w])
        elif w.capitalize() in model:  # Fallback per Word2Vec case-sensitive
            vectors.append(model[w.capitalize()])
            
    if not vectors:
        return None
        
    vec = np.mean(vectors, axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else None

def cosine_similarity(vec1, vec2):
    """Calcola la similarità coseno tra due vettori normalizzati."""
    if vec1 is None or vec2 is None:
        return 0.0
    return float(np.dot(vec1, vec2))


# ── DATABASE MAP QUERY ────────────────────────────────────────────────────────

class MapQuery:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser().resolve())
        if not Path(self.db_path).exists():
            print(f"[ERRORE] File DB non trovato: {self.db_path}")
            sys.exit(1)

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_all_objects(self, only_active=True):
        """Recupera tutti gli oggetti (inclusa la descrizione se esiste)."""
        with self._conn() as conn:
            # Uso PRAGMA per verificare se c'è la colonna description
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(objects)")
            columns = [info[1] for info in cursor.fetchall()]
            
            desc_col = "description" if "description" in columns else "'' as description"
            
            q = f"SELECT id, label, color, {desc_col}, x, y, z, is_active, last_event, last_seen, is_uncertain FROM objects"
            if only_active:
                q += " WHERE is_active=1"
                
            return [dict(r) for r in conn.execute(q).fetchall()]

    def quanti(self):
        with self._conn() as conn:
            attivi = conn.execute("SELECT COUNT(*) FROM objects WHERE is_active=1").fetchone()[0]
            totale = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
            return attivi, totale
            
    def storia(self, obj_id):
        with self._conn() as conn:
            q = """SELECT timestamp, event_type, phase, step,
                          x_old, y_old, z_old, x_new, y_new, z_new, distance
                   FROM object_history WHERE object_id = ? ORDER BY timestamp DESC LIMIT 10"""
            # Se il tuo schema non ha object_id ma usa label, cambia la query qui
            try:
                return [dict(r) for r in conn.execute(q, [obj_id]).fetchall()]
            except sqlite3.OperationalError:
                # Fallback se non c'è object_id nella history
                return []


# ── MOTORE DI RICERCA SEMANTICA ───────────────────────────────────────────────

def find_best_matches(query: str, objects: list, model, top_k=3, threshold=0.2, debug=True):
    """
    Confronta la domanda con gli oggetti evitando l'effetto "diluizione" delle descrizioni
    e rimuovendo le stopwords che ingannano Word2Vec (es. "dove" = colomba in inglese).
    """
    # 1. Rimuoviamo le parole inutili (Stopwords)
    STOPWORDS = {"dove", "dov", "è", "e", "il", "lo", "la", "i", "gli", "le", "un", "una", 
                 "c", "ci", "sono", "cerco", "trova", "mostrami", "where", "is", "the", "a", "an"}
    
    words = re.findall(r'\w+', query.lower())
    meaningful_words = [w for w in words if w not in STOPWORDS]
    clean_query = " ".join(meaningful_words)
    
    if debug:
        print("\n  📊 DEBUG MATCHING PROCESS")
        print(f"  ├─ Query originale: '{query}'")
        print(f"  ├─ Parole rilevanti: {meaningful_words}")
        print(f"  └─ Query pulita: '{clean_query}'")
    
    # Se dopo la pulizia non rimane nulla, usa la query originale
    if not clean_query:
        clean_query = query 
        
    query_vec = get_sentence_embedding(model, clean_query)
    if query_vec is None:
        if debug:
            print("  ❌ Impossibile creare embedding per la query!")
        return []
    
    if debug:
        print(f"  ├─ Query embedding: shape={query_vec.shape}, norm={np.linalg.norm(query_vec):.4f}")

    scored_objects = []
    
    for idx, obj in enumerate(objects):
        color = obj.get('color', '') or ''
        label = obj.get('label', '') or ''
        label_clean = re.sub(r'#\d+$', '', label)
        desc = obj.get('description', '') or ''
        
        if debug:
            print(f"\n  ┌─ OGGETTO #{idx}: {color} {label}")
        
        # 2. Calcoliamo i vettori SEPARATI per Label e per Descrizione
        core_text = f"{color} {label_clean}".strip()
        core_vec = get_sentence_embedding(model, core_text)
        desc_vec = get_sentence_embedding(model, desc) if desc else None
        
        if debug:
            if core_vec is not None:
                print(f"  ├─ Core text: '{core_text}'")
                print(f"  ├─ Core vec norm: {np.linalg.norm(core_vec):.4f}")
            else:
                print("  ├─ ⚠️ Core vec: None")
            
            if desc_vec is not None:
                print(f"  ├─ Description: '{desc[:50]}...' " if len(desc) > 50 else f"  ├─ Description: '{desc}'")
                print(f"  ├─ Desc vec norm: {np.linalg.norm(desc_vec):.4f}")
            else:
                print("  ├─ Description vec: None (no description)")
        
        # Similarità diretta col nome dell'oggetto (Es. "computer" vs "monitor")
        sim_core = cosine_similarity(query_vec, core_vec) if core_vec is not None else 0.0
        
        # Similarità con la descrizione (se la query contiene dettagli specifici)
        sim_desc = cosine_similarity(query_vec, desc_vec) if desc_vec is not None else 0.0
        
        if debug:
            print(f"  ├─ Sim(query, core): {sim_core:.4f}")
            print(f"  ├─ Sim(query, desc): {sim_desc:.4f}")
            print(f"  ├─ Sim(desc) × 0.8: {sim_desc * 0.8:.4f}")
        
        # 3. Prendiamo il punteggio MIGLIORE. 
        # Moltiplichiamo sim_desc per 0.8 per dare sempre una leggera priorità al nome esatto
        final_sim = max(sim_core, sim_desc)
        
        if debug:
            status = "✅ MATCH" if final_sim >= threshold else "❌ FILTERED"
            print(f"  └─ FINAL SCORE: {final_sim:.4f} [{status}]")
        
        if final_sim >= threshold:
            scored_objects.append((final_sim, obj, core_text))
            
    # Ordina per similarità decrescente
    scored_objects.sort(key=lambda x: x[0], reverse=True)
    
    if debug:
        print(f"\n  🏆 TOP {top_k} MATCHES:")
        for rank, (sim, obj, txt) in enumerate(scored_objects[:top_k], 1):
            print(f"  {rank}. {obj['color']} {obj['label']} → Score: {sim:.4f}")
    
    return scored_objects[:top_k]


# ── HELPERS STAMPA ────────────────────────────────────────────────────────────

def pos(x, y, z):
    try:
        return f"({float(x):.2f}, {float(y):.2f}, {float(z):.2f})"
    except:
        return "(null)"

# ── REPL PRINCIPALE ───────────────────────────────────────────────────────────

def signal_handler(sig, frame):
    print("\n⏹️ Operazione interrotta.")

signal.signal(signal.SIGINT, signal_handler)

def main():
    db_path  = "/root/exchange/output/tiago_temporal_map_1.db"
    w2v_path = None

    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--db"  and i + 1 < len(args): db_path  = args[i + 1]
        if a == "--w2v" and i + 1 < len(args): w2v_path = args[i + 1]

    if w2v_path is None:
        candidates = [
            "/root/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz"
        ]
        for c in candidates:
            if Path(c).exists():
                w2v_path = c
                break

    db = MapQuery(db_path)
    model = load_w2v(w2v_path)

    print("\n🤖 TIAGO MAP — PURE SEMANTIC SEARCH")
    print(f"   DB  : {db_path}")
    print(f"   W2V : {w2v_path}")
    print("─" * 60)
    print("  Fai una domanda per cercare un oggetto in base a label, color e description.")
    print("  Comandi di sistema: lista / tutto / quanti / clear / exit")
    print("─" * 60 + "\n")

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCiao!")
            break

        if not raw:
            continue

        tl = raw.lower().strip("?!. ")

        # ── COMANDI DI SISTEMA ──────────────────────────────────────────
        if tl in ("exit", "quit", "esci"):
            print("Ciao!")
            break

        if tl == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue

        if tl == "quanti":
            attivi, totale = db.quanti()
            print(f"Attivi: {attivi}  |  Totale storico: {totale}\n")
            continue

        if tl == "lista" or tl == "tutto":
            only_active = (tl == "lista")
            objs = db.get_all_objects(only_active=only_active)
            print(f"\n  {'='*60}\n  OGGETTI {'ATTIVI' if only_active else 'TUTTI'} — {len(objs)}\n  {'='*60}")
            for o in objs:
                stato = "✅" if o["is_active"] else "❌"
                unc = " ⚠️" if o["is_uncertain"] else ""
                print(f"  {stato} {str(o['color']):8s} {str(o['label']):15s} {pos(o['x'],o['y'],o['z'])} [{o['last_event']}]{unc}")
            print(f"  {'='*60}\n")
            continue

        # ── RICERCA SEMANTICA ───────────────────────────────────────────
        
        # Estraiamo un eventuale intento specifico (storia/spostato) prima di cercare
        check_history = "stori" in tl or "history" in tl
        # unused — the moved-intent flag is never consumed (history is); restore when a MOVED query path exists
        # check_moved = "spostat" in tl or "mov" in tl
        
        # Puliamo la query da queste parole per non inquinare l'embedding
        clean_query = re.sub(r'(storia|history|spostat[oaie]|moved?)', '', tl).strip()
        if not clean_query:
            clean_query = tl

        # 1. Recupera tutti gli oggetti
        all_objs = db.get_all_objects(only_active=False if check_history else True)
        
        # 2. Trova i match semantici
        matches = find_best_matches(clean_query, all_objs, model, top_k=3, threshold=0.25, debug=True)
        
        if not matches:
            print(f"  ❓ Nessun oggetto corrispondente trovato per '{raw}' (Score troppo basso).")
            continue

        # 3. Mostra i risultati
        best_sim, best_obj, best_text = matches[0]
        
        print(f"\n  🎯 MIGLIOR MATCH: {best_obj['color']} {best_obj['label']} (Similarità: {best_sim:.2f})")
        if best_obj['description']:
            print(f"     Descrizione: {best_obj['description']}")
        print(f"     Posizione  : {pos(best_obj['x'], best_obj['y'], best_obj['z'])}")
        print(f"     Stato      : {'Attivo ✅' if best_obj['is_active'] else 'Rimosso ❌'} | Ultimo Evento: {best_obj['last_event']}")

        # Se ci sono altri match vicini, li suggerisce
        if len(matches) > 1:
            print("  💡 Altri oggetti simili trovati:")
            for sim, obj, txt in matches[1:]:
                print(f"     • {obj['color']} {obj['label']} (Sim: {sim:.2f})")

        print()

if __name__ == "__main__":
    main()