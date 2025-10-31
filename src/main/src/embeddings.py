# embedding.py => per creare il database vettoriale e gestire gli input correttamente per LLM

"""
Embeddings & Indexing utilities
- CPU-friendly sentence embeddings (SentenceTransformers)
- FAISS cosine index (default) or pgvector
- Reusable functions for other modules

Dependencies:
  pip install sentence-transformers faiss-cpu orjson
  # con pgvector:
  pip install psycopg2-binary

Config:
  vedi config.py per i path (CHUNKS_JSONL, FAISS_INDEX, FAISS_META) e i nomi modello.
"""

from __future__ import annotations
import os, json, orjson, glob, math
from typing import Dict, Any, List, Optional, Tuple, Iterable
import numpy as np
import faiss
from config import (
    CHUNKS_JSONL, FAISS_INDEX, FAISS_METADATA,
    EMBED_MODEL_NAME, USE_PGVECTOR, PG_DSN, PG_TABLE, DEVICE
)
from pathlib import Path
from sentence_transformers import SentenceTransformer


# B. Classe del modello EMBEDDING con singleton semplice => serve per indicizzare i dati nel DB vettoriale # Si creano degli embedding vettoriali dalle parole ed essi vengono indicizzati dal modello per generare i vari token 
# # 1. Classe 
class EmbeddingModel: 
    _instance = None 
    def _new_(cls): 
        if cls._instance is None: 
            m = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE) # includo modello pre-addestrato embedding dal config 
            cls._instance = m # associo il modello all'istanza attuale dell classe 
        return cls._instance 

_MODEL: Optional[SentenceTransformer] = None # Model initialization 
# 2. Funzione di restituzione 
def get_model() -> SentenceTransformer: 
    global _MODEL 
    if _MODEL is None: 
        _MODEL = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE) 
    return _MODEL

# A. Utility helpers di I/O per JSONL
# 1. Caricamento file JSON in lettura binaria
def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            # YIELD -> keyword di restituzione che si usa dentro una funzione per trasformarla in un generatore. 
            # Un generatore è una funzione che non restituisce tutti i risultati subito, 
            # ma uno alla volta, solo quando serve, mantenendo il suo stato interno tra una chiamata e l’altra.
            # e non resettandosi quindi ad ogni chiamata come con la "return"

# 2. Salvataggio file JSON in scrittura binaria
def save_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None: # Tale funzione non restituisce nulla ma fa solo da procedura
    with open(path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")


# C. Chunking per il testo => dividiamo il testo globale in tanti blocchi di dimensione fissa con un leggero overlapping di transizione tra un blocco e l'altro
# per gestire bene il meccanismo di self-attention del LLM
# - text: str => testo da chunkare
# - chunk_size: int => di quanto chunkarlo
# - overlap: int => di quanto sovrappore i vari chunk tra loro
# Return: List => restituisco l'array dei vari chunk
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = text or "" # se non esiste text, restituisco vuoto
    n = len(text)
    if n == 0 or chunk_size <= 0:
        return []

    # limiti di sanity
    overlap = max(0, min(overlap, chunk_size - 1)) # evita che sia minore di zero e maggiore di chunk size
    step = chunk_size - overlap # sliding window con overlapping tra i vari chunks -> si avanza di chunk in chunk tenendo conto della sovrapposizione tra essi

    out = []    # array chunkato in output
    for i in range(0, n, step): # for (i=0; i<n; i+=step)
        j = min(n, i + chunk_size) # indice di chunking
        out.append(text[i:j])   # appendo i vari chunk all'output finale
        if j >= n:  # il chunk non può mai superare la lunghezza totale del testo, se succede esco dal ciclo
            break
    
    return out

# D. Conversione dei chunk testuali in embeddings vettoriali numpy (float32)
# - chunks: List[str] => Lista di stringhe (blocchi testuali in lista)
# - batch_size: int => quanto di quei chunks processare alla volta in batch uguali per generare embeddings
# - normalize: bool = True => effettuiamo normalizzazione L2 cosine similarity su embeddings
# Return: np.ndarray => si fornisce in output un array numpy di vettori embedding generati dai chunks
def encoding(chunks: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    model = get_model() 
    # procedura numpy per generare embedding
    embeds = model.encode(chunks,
                       batch_size=batch_size,
                       convert_to_numpy=True,
                       show_progress_bar=True,
                       normalize_embeddings=normalize
    )
    # SentenceTransformers restituisce già float32 di solito, ma garantiamo con guardia esplicita
    return embeds.astype(np.float32, copy=False)

# E. Costruzione degli indici vettoriali FAISS a partire dai Chunks di testo in input
# Si indicizzano vettorialmente gli embeddings ottenuti con Faiss per poter poi fare ricerche top-k
# Legge i chunk da chunks.jsonl, calcola embeddings e crea un indice FAISS (cosine con IndexFlatIP).
# Salva anche i metadati (uno per riga in faiss_meta_path).
# Return -> Tuple (num_chunks, dim)
def build_faiss(
        chunks: str = CHUNKS_JSONL,
        faiss_index : str = FAISS_INDEX,
        faiss_meta: str = FAISS_METADATA,
        text_field: str = "text",
        meta_field: str = "metadata",
        id_field: str   = "chunk_id",
        batch_size: int = 64,
) -> Tuple [int, int]:
    
    # 1) Chunks loading and processing
    tmp_text = list(load_jsonl(chunks))
    if not tmp_text:  # gestione eccezione
        raise RuntimeError(f"Nessun chunk da processare trovato in {chunks}")
    
    out: List[str] = []    # Lista di stringhe => chunks effettivi
    metas: List[Dict[str, Any]] = []  # Lista di dizionari => metadati
    for t in tmp_text:
        row = t.get(text_field, "") or ""
        if not isinstance(row, str): # se non è stringa -> lo converto
            txt = str(txt)
        out.append(txt) # genero la lista di stringhe dai chunks caricati

        # gestione dei metadati allo stesso modo
        meta = dict(t.get(meta_field, {}) or {})
        meta["chunk_id"] = t.get(id_field)  # indicizziamo ogni FAISS anche col proprio chunk di origine per capire da dove proviene
        meta["text"] = txt # contesto direttamente nella top-k search
        metas.append(meta)
    
    # 2) Embeddings dimension and indexes
    emb = encoding(out, batch_size=batch_size, normalize=True) # (N, dim)
    n, dim = emb.shape # estraggo il numero e la dimensione di ogni embedding

    # 3) FAISS indexes => cosine = dot su vettori normalizzati
    index = faiss.IndexFlatIP(dim) # creo indici FAISS
    index.add(emb) # indicizzo embeddings normalizzati

    # 4) Salvataggio degli indici FAISS
    faiss.write_index(index, faiss_index)
    with open(faiss_meta, "wb") as f:
        for m in metas:
            f.write(orjson.dumps(m) + b"\n")

    # 5) Return and log
    print(f"[faiss] wrote index={faiss_index} | metas={faiss_meta} | n={n} dim={dim}")
    return n, dim

# Helper per caricare gli indici FAISS e la lista dei metadati correlati riga per riga
def load_faiss(
        faiss_index: str = FAISS_INDEX,
        faiss_meta: str = FAISS_METADATA,
) -> Tuple[Any, List[Dict[str, Any]]]:
        
        if not Path(faiss_index).exists():
            raise FileNotFoundError(f"Indice FAISS mancante: {faiss_index}.")
        if not Path(faiss_meta).exists():
            raise FileNotFoundError(f"Metadati corrispondenti non trovati.")
        
        # lettura indici e metadati correlati
        index = faiss.read_index(faiss_index)
        with open(faiss_meta, "r", encoding="utf-8") as f:
            metas = [json.loads(x) for x in f if x.strip()]

        return index, metas

# Codifica `texts` in batch, aggiunge i vettori a `index` e appende i `metadatas` a `metas`.
# Salva su disco index e metadati aggiornati.
# Ritorna (index, metas).
def add_texts_to_faiss(
    index,
    metas: list[dict],
    texts: list[str],
    metadatas: list[dict],
    batch_size: int = 64,
    normalize: bool = True,
    index_path: str = FAISS_INDEX,
    meta_path: str = FAISS_METADATA,
):
    assert len(texts) == len(metadatas), "texts e metadatas devono avere stessa lunghezza"
    model = get_model()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=normalize)
        vecs = vecs.astype(np.float32, copy=False)
        index.add(vecs)
        metas.extend(metadatas[i:i+batch_size])

    # persist
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    return index, metas

# Funzione per fare ricerca top-k sugli embeddings con indici FAISS
# Praticamente, quando si genera una risposta ad un prompt, gli embeddings vengono indicizzati tramite indici FAISS
# ma tali indici FAISS come vengono chiamati e ricercati? Grazie alla ricerca top-k.
# Senza ricerca top-k non è possibile fare in modo che LLM produca risposte sensate
def top_k_search(
        query: str,
        k: int,
        index,
        metas: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    # trasformo in embedding allo stesso modo anche la richiesta che viene effettuata come query al LLM
    # Perché? LLM deve essere in grado di indicizzare la richiesta allo stesso modo dei chunk di testo
    # altrimenti non riesce a dare risposte sensate
    # grazie all'Attention riesce a pesare la sequenza di token ad ogni passo e a restituire risultati vicini
    # nello spazio vettoriale degli embedding con complessità computazionale: O(n^2)
    # embedding di richiesta vicino a quello di chunk = alta probabilità che tale token sia giusto per la risposta in questione
    q = encoding([query], batch_size=1, normalize=True) # spezzetto tutti i vari caratteri della richiesta in token per embedding
    D, I = index.search(q.astype())
    found = []
    for score, idx in zip(D[0], I[0]): # si ricerca nello spazio degli indici
        if idx < 0:  # FAISS mette -1 se non trova e procede col successivo
            continue
        m = metas[idx] # metadati corrispettivi
        found.append({ # una volta trovati i relativi dati, si appendono nella lista finale e si restituisce
            "chunk_id": m.get("chunk_id"),
            "scores": float(score),
            "metadata": m
        })
    
    return found

# OPZIONALE: Funzione pgvector per creare database vettoriale con gli embeddings generati
# Non è essenziale, ma è molto comodo avere una tabella vettorialmente indicizzabile con gli embeddings
# Inserisce i chunk (testo, metadata, embedding dei chunks) nella tabella vettoriale pgvector del DB
# Assicura schema vettoriale base. Ritorna numero di record inseriti per comodità.
# UNICO requisito -> richiede di avere un DB vettoriale (ma non è fondamentale averlo, si può fare anche senza)
# anche se non verrà utilizzato, predispongo il codice per poterlo utilizzare
def pgvector(
        chunks_jsonl: str = CHUNKS_JSONL,
        text_field: str = "text",
        meta_field: str = "metadata",
        id_field: str = "chunk_id",
        batch_size: int = 64
) -> int:
    
    # try/catch sugli import che servono per creare DB vettoriale
    # è solo una safeguard -> si potrebbe tranquillamente solo importare la libreria nel file e usarla
    try:
        import psycopg2 as psy
        import psycopg2.extras
    except Exception as e:
        raise RuntimeError("Dipendenze non installate. 'pip install psycopg2-binary'") from e
    
    # import dei chunks e controllo che siano > 0
    try:
        rows = list(load_jsonl)
    except Exception as ex:
        raise RuntimeError(f"Nessun chunk trovato da processare in {chunks_jsonl}") from ex
    
    # connessione al DB vettoriale
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor() # usiamo il cursore per la gestione dei dati
    # name - type - key_type (if it exists)
    cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_TABLE}(
                    chunk_id text PRIMARY KEY,
                    text text,
                    metadata jsonb,
                    embedding vector(384)
                );
    """)

    # committiamo la connessione al DB dopo che è stata fatta la creazione della tabella
    conn.commit()

    # raccolgo i vari tipi di dato con dict e list comprehension
    texts = [r.get(text_field, "") or "" for r in rows]
    metas = [(r.get(meta_field) or {}) | {"chunk_id": r.get(id_field)} for r in rows]
    ids = [r.get(id_field) for r in rows]
    if any(x is None for x in ids):
        raise ValueError("Alcuni chunk non hanno id_field")

    # encoding runtime a batch per evitare sovraccarico in memoria
    model = get_model()
    inserted = 0
    page = 0
    total_pages = math.ceil(len(texts) / batch_size) # contenuto effettivo dei batch: testo totale / numero di batch
    '''Perché i:i+batch_size invece di solo i?
    - i da solo sarebbe un singolo indice (un solo elemento).
    - i:i+batch_size è uno slice: prende un blocco di elementi da i incluso a i+batch_size escluso.
    - Questo è il cuore del batching: processi p.es. 64 testi alla volta invece di tutti insieme (risparmio memoria, migliore parallelismo).
    - Esempio pratico con batch_size=3 e len(texts)=8:
            i=0 → texts[0:3] → elementi 0,1,2
            i=3 → texts[3:6] → elementi 3,4,5
            i=6 → texts[6:9] → elementi 6,7 (fine lista, lo slice tronca senza errori)
    - Perché farlo così:
        Memoria: non crei embedding di tutti i testi in una volta.
        Performance: i modelli di embedding lavorano bene su batch moderati.
        Sicurezza: lo slice oltre la fine non esplode; restituisce una lista più corta.'''
    for page, i in enumerate(range(0, len(texts), batch_size), start=1): # cicliamo tutto il testo partendo da 0 e andando avanti di batch in batch
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]
        # log per capire quanti batch abbiamo processato rispetto al totale dei batch presenti
        print(f"Batch: {page}/{total_pages}")

        # embedding sul testo
        vecs = model.encode(batch_texts, normalize_embeddings=True, convert_to_numpy=True)
        vecs = vecs.astype(np.float32) # conversione a float32

        # payload dizionario json finale con i dati embedded
        payload = [
            (cid, t, json.dumps(m), v.tolist())
            for cid, t, m, v in zip(batch_ids, batch_texts, batch_metas, vecs)
        ]

        # log di debug payload
        print(f"Payload finale: {payload}")

        # inserimento del payload dati dentro la tabella effettiva del database vettoriale
        psycopg2.extras.execute_batch(
            cur,
            f"""INSERT INTO {PG_TABLE}(chunk_id, text, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING""",
            payload, page_size=100
        )
        # commit effettivo della query di inserimento
        conn.commit()
        # int di dati inseriti
        inserted += len(payload)
        page += 1
        print(f"[pgvector] page {page}/{total_pages} inserted {len(payload)}")
    
    # chiudo connessioni
    cur.close()
    conn.close()

    # tutti i record inseriti nel DB vettoriale dopo il loop
    print(f"[pgvector] total inserted: {inserted}")

    return inserted