import sqlite3
import pandas as pd
import os
from pyvis.network import Network

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = "/root/exchange/lost3dsg"
TARGET_SUBDIR = "grafici_output"
DB_PATH = os.path.join(BASE_DIR, "/root/exchange/output/tiago_temporal_map_5.db")
OUTPUT_HTML = os.path.join(BASE_DIR, TARGET_SUBDIR, "albero_mondo.html")

# ID UTENTE HOST (Standard Ubuntu: 1000)
HOST_UID = 1000 
HOST_GID = 1000

def genera_grafo_corretto():
    # 1. Creazione Cartella
    target_path = os.path.join(BASE_DIR, TARGET_SUBDIR)
    os.makedirs(target_path, exist_ok=True)
    
    # 2. Lettura Dati
    if not os.path.exists(DB_PATH):
        print(f"❌ DB non trovato: {DB_PATH}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT label, room_id FROM objects", conn)

    if df.empty:
        print("⚠️ Database vuoto.")
        return

    # 3. Creazione Grafo (Sfondo Bianco)
    net = Network(height='850px', width='100%', bgcolor='#ffffff', font_color='#000000', directed=True)

    # Nodo Radice: WORLD
    net.add_node("ROOT", label="World", shape="circle", color="#FF5733", size=40, font={'size': 25})

    # 4. Costruzione Gerarchia
    # Pulizia Room ID per evitare il doppio "Room Room"
    def pulisci_room_name(val):
        s = str(val).lower().replace('room', '').strip()
        return f"Room {s}"

    df['room_clean'] = df['room_id'].apply(pulisci_room_name)
    
    rooms = df['room_clean'].unique()
    
    for r_name in rooms:
        # Nodo Room (singolo nome corretto)
        net.add_node(r_name, label=r_name, shape="box", color="#3498DB", size=30)
        net.add_edge("ROOT", r_name, color="#2C3E50")

        # Filtra oggetti per questa stanza
        obj_list = df[df['room_clean'] == r_name]['label'].tolist()
        
        for obj in obj_list:
            # ID univoco per evitare che oggetti con stesso nome in stanze diverse si fondano
            obj_id = f"{r_name}_{obj}"
            net.add_node(obj_id, label=str(obj), shape="dot", color="#F1C40F", size=15)
            net.add_edge(r_name, obj_id, color="#BDC3C7")

    # 5. Configurazione Layout
    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "UD",
          "sortMethod": "directed",
          "nodeSpacing": 200,
          "levelSeparation": 150
        }
      },
      "physics": { "enabled": false },
      "edges": { "arrows": { "to": { "enabled": true } } }
    }
    """)

    # 6. Salvataggio e Sblocco Permessi
    try:
        if os.path.exists(OUTPUT_HTML):
            os.remove(OUTPUT_HTML)
        
        net.save_graph(OUTPUT_HTML)
        
        # Tenta il cambio proprietario e permessi
        try:
            os.chown(OUTPUT_HTML, HOST_UID, HOST_GID)
        except:
            pass
        os.chmod(OUTPUT_HTML, 0o666)
        
        print(f"\n✅ Grafo generato con successo!")
        print(f"📍 Percorso: {OUTPUT_HTML}")
        print(f"✨ Etichette corrette: 'World' -> 'Room X' -> 'Oggetto'")
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    genera_grafo_corretto()
