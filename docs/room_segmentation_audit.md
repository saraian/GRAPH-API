# Audit generale della segmentazione delle stanze — 4 novembre 2026

Prove indipendenti da HM3D 00861 e dalla sua ground truth. Nessuna soglia del
segmentatore è stata modificata nell'audit. Esecuzione riproducibile:

```bash
python3 tools/analysis/probe_room_segmentation.py --output output/room_algorithm_audit
```

I risultati completi, i parametri effettivi, l'hash del sorgente e la versione
OpenCV sono in `output/room_algorithm_audit/results.json`. Sono usati numpy e
OpenCV reali e RoomManager reale, con le sole dipendenze ROS simulate. VLM porte e
finestre disabilitati; non sono forniti cloud o osservazioni di muri 3D.

## 1. Continuità dei muri: due distinti punti di perdita

Fixture: due stanze da circa 12 m², porta da 1 m, muro separatore da 0,6 m.
Risoluzioni 2,5 / 5 / 10 cm; test del taglio anche dopo rotazione di 90°.

- Il taglio corretto separa le due stanze con muro continuo (6/6 casi).
- Aggiungendo un'altra apertura di un solo pixel nel separatore, il taglio
  principale non divide più la componente: `rejected_no_split` (6/6).
- Una discontinuità da 20 cm ha lo stesso effetto sul validatore (6/6).
- Nella pipeline completa, il caso da un pixel viene recuperato dal preprocessing:
  2 regioni a tutte le risoluzioni. Non è quindi corretto affermare che un singolo
  pixel aperto faccia necessariamente fallire l'intera pipeline.
- Nella pipeline completa, la discontinuità da 20 cm produce 1 regione in tutte
  e tre le risoluzioni. In questo caso non viene nemmeno proposto un taglio:
  i rami GVD sono scartati per larghezza o minimo all'estremità.

Conclusione: le divisioni dipendono sia dalla connettività globale della mappa,
che dalla sopravvivenza dei candidati durante preprocessing e GVD. I tagli
congiunti non possono recuperare candidati che non vengono generati. Questa
fixture non prova un fallimento in presenza di supporto 3D: quel canale non è
stato fornito.

Intervento generale da valutare: diagnosticare separatamente la perdita dei
candidati e i tagli senza divisione; recuperare piccole discontinuità soltanto
quando esiste supporto strutturale coerente. Chiudere indiscriminatamente tutti
i buchi rischia di sigillare passaggi reali.

## 2. Frammenti minuscoli: difetto del criterio di validazione

Una piccola cavità nel muro, collegata alla porta da un collo stretto, viene
isolata dal taglio insieme alle due stanze principali. Il validatore esegue
`areas.min() < min_component_px`, perciò rifiuta l'intera suddivisione.

| Risoluzione | Aree dopo il taglio (m²) | Esito |
| --- | --- | --- |
| 2,5 cm | 12,225; 12,200; 0,071 | rifiutato |
| 5 cm | 12,150; 12,100; 0,058 | rifiutato |
| 10 cm | 12,000; 11,850; 0,010 | rifiutato |

Il risultato si ripete a 90°: 6/6 rifiuti. Le due componenti principali superano
largamente la soglia invariata di 1,5 m². Anche la validazione collettiva conserva
il controllo sul minimo di tutte le componenti.

Limite importante: nella pipeline completa questa specifica cavità sintetica
viene ripulita e si ottengono 2 regioni in tutte le risoluzioni. Il test dimostra
la fragilità del validatore quando il frammento gli arriva, non che ogni cavità
attraversi il preprocessing. I log reali precedentemente analizzati contengono
un caso [80,498; 2,973; 0,018] rifiutato, coerente con lo stesso difetto.

Intervento generale: richiedere almeno due componenti significative e trattare
separatamente i residui di rumore, con un limite alla loro area complessiva e
senza ricongiungere regioni attraverso una porta confermata. Test negativi
necessari: una grande stanza più soli frammenti, molti frammenti con area totale
rilevante, e una vera stanza piccola. Non basta abbassare la soglia minima.

## 3. Callback: contesa reale nel codice, causa della run non dimostrata

`process_grid` mantiene un RLock durante segmentazione, stabilizzazione,
aggiornamento delle stanze, pubblicazione e salvataggio. Anche le callback RGB,
camera-info e cloud usano questo lock. Il gruppo ROS è Reentrant, ma questo
non elimina il lock interno.

ObjectManager usa quattro worker. La callback `/detected_wall_segments` sta nel
gruppo predefinito, insieme alla ricezione bbox, descrizioni e watchdog; entra
in `ingest_detected_walls`, che acquisisce il medesimo lock di RoomManager.
Quindi può bloccare il gruppo delle bbox in attesa della segmentazione. Inoltre,
più callback reentrant in attesa del lock possono occupare i worker disponibili.

Prova deterministica con pool da quattro worker e callback RoomManager reali:
un `process_grid` trattiene il lock, tre callback camera-info attendono quel
lock e un quinto task indipendente non parte. Dopo il rilascio tutti terminano.
Il tempo di trattenimento è controllato artificialmente: la prova mostra il
meccanismo di accodamento, non misura un arresto ROS né riproduce i 550 s reali.

### Costo misurato del solo validatore dei tagli

Mediana di tre esecuzioni, host x86_64, Python 3.12.3, OpenCV 4.6.0 (20 thread).
Mappe libere con proposte corte senza divisione; tempi in millisecondi.

| Raster | 1 candidato | 8 candidati | 32 candidati |
| --- | ---: | ---: | ---: |
| 200 × 200 | 0,6 | 4,6 | 16,4 |
| 400 × 400 | 2,5 | 15,0 | 54,6 |
| 800 × 800 | 10,9 | 59,8 | 211,7 |

Ogni proposta copia e rietichetta il raster intero. Queste misure escludono
cloud, GVD, aggiornamenti del grafo, pubblicazioni e disco; non sono tempi ROS
end-to-end. La pipeline completa sulla piccola fixture a due stanze richiede
circa 5–96 ms secondo la risoluzione, senza cloud/VLM.

Le correzioni precedenti invalidano inoltre la cache a ogni cloud o osservazione
di muro. Questo evita di ignorare nuova evidenza, ma può aumentare il numero di
segmentazioni anche quando la griglia di occupazione non cambia. Non c'è un
worker dedicato che accorpi richieste usando soltanto l'ultimo stato disponibile.

Intervento generale da validare: callback brevi che aggiornano uno snapshot,
un unico worker di segmentazione con accorpamento delle richieste e applicazione
dei risultati sotto lock breve. Misurare attesa del lock, tempo di segmentazione,
frequenza effettiva, richieste accorpate e latenza delle callback bbox. Per il
watchdog distinguere l'assenza di messaggi dall'impossibilità di eseguire le
callback; il solo timestamp di ingresso nella callback non identifica la causa.

## Priorità

1. Correggere e testare la validazione dei residui piccoli: difetto circoscritto.
2. Strumentare le latenze e isolare il lavoro pesante dalle callback ROS.
3. Valutare recupero dei candidati e discontinuità con evidenza strutturale,
   includendo controlli negativi su open space e passaggi reali.

Queste prove non misurano precisione/recall su un insieme di scene e non
consentono di attribuire il timeout osservato alle modifiche del segmentatore.
Servono come casi riproducibili e criteri di verifica per le prossime correzioni.

## Correzioni implementate dopo l'audit

I risultati iniziali in `output/room_algorithm_audit` rimangono il riferimento
prima delle modifiche. I risultati successivi sono in `output/room_algorithm_fixed`.

- `room_cut_noise_max_area_m2=0.10` e `room_cut_noise_max_ratio=0.01` limitano
  l'area TOTALE dei residui, sia in assoluto sia rispetto alle componenti
  risultanti. Devono comunque esistere almeno due regioni sopra la soglia di
  1,5 m². I residui accettati vengono rimossi dalla sola topologia del taglio,
  senza riaprire il separatore. Il criterio è condiviso dai tagli GVD singoli,
  collettivi e dai tagli derivati dai muri. Le statistiche espongono
  `discarded_noise_area_m2`.
- `gvd_wall_gap_max_m=0.25` e `gvd_wall_gap_support_m=0.30` consentono riparazioni
  locali prima della classificazione finale degli ostacoli strutturali. Sono
  necessari tratti occupati misurati allineati da entrambi i lati e supporto
  strutturale preesistente su almeno un lato. Nessuna riparazione si basa su
  una riparazione precedente. Gli spazi sconosciuti e le aperture confermate
  sono protetti. Il massimo gap è comunque limitato a metà della larghezza
  minima configurata per una porta da muri. Le statistiche espongono
  `wall_gap_repair.repaired_cells`.

Queste sono ipotesi metriche configurabili, non parametri stimati sulla scena
00861. Un gap non confermato sotto il limite può essere interpretato come una
lacuna del muro: l'algoritmo non può sapere da sola occupazione se sia un'apertura
fisica minuscola. La ricerca lungo quattro direzioni raster privilegia riparazioni
conservative; muri sottili con altri orientamenti possono restare non riparati.
Impostare il massimo gap a zero disabilita il recupero; impostare a zero il budget
di rumore ripristina il rifiuto di qualunque residuo sottosoglia.

Verifiche: 11 nuovi test con sottocasi a 2,5/5/10 cm e rotazioni 0/90/180/270°,
più un caso diagonale. Inclusi controlli negativi per open space, porte da
0,55–2 m, spazio sconosciuto, aperture confermate, mobili isolati, rumore
cumulativo e una sola componente significativa. Passano anche i 13 test precedenti
sulla recall, i parametri e il test delle due stanze con OpenCV reale.

Risultato: il caso sintetico con lacuna di 20 cm produce ora due stanze a tutte
le risoluzioni; il taglio con due stanze grandi e un piccolo residuo viene
accettato e lascia due componenti. Questo dimostra le regressioni corrette,
non un incremento già misurato di recall su nuove scene. Il costo del recupero
va incluso nelle prossime misure ROS; il problema delle callback è separato.
