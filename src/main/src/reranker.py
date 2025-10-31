# reranker.py

"""
Reranking utilities for RAG
- Cross-encoder reranker (CPU-friendly)
- Score normalization & blending with vector scores
- Optional Reciprocal Rank Fusion (RRF)
- Small CLI to test with FAISS results from embeddings.py

Deps:
  pip install sentence-transformers orjson

Description:
  Il cross-encoder reranker serve per ottimizzare al meglio la qualità delle risposte che LLM dà in output.
  E' un ausilio importante che gira bene su CPU e che vale la pena implementare
  Il cross-encoder reranker è un modello che riconsidera (re-rank) i risultati del retriever con un’analisi molto più fine, basata sull’interazione completa tra query e documento.
"""
from __future__ import annotations
import os
import math
import numpy as np
import hashlib
from typing import Tuple, List, Dict, Optional, Any, Iterable

# import con safeguard per librerie più specialistiche
try:
    from sentence_transformers import CrossEncoder
except Exception as e:
    raise RuntimeError("Libreria 'sentence-transformers' mancante => pip install sentence-transformers") from e
    
try:
    from config import RERANKER_NAME, RERANK_ALPHA
except Exception:
    RERANKER_NAME = os.getenv("RERANKER_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # blending tra score del reranker (0..1, dopo sigmoid) e score vettoriale FAISS (cosine sim normalizzata)
    RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", "0.7"))

# helpers per il reranker
# 1. Ricava testo del candidato attuale dai metadati
# embeddings.top_k_search salva il testo in metadata['text'].
def extract_text(cand: Dict[str, Any]) -> str:
    if "text" in cand and isinstance(cand["text"], str): # se il testo è presente
        return cand["text"]
    meta = cand.get("metadata") or {} # dizionario coi campi dei metadati
    txt = meta.get("text", "") # contenuto dei metadati

    return txt if isinstance(txt, str) else str (txt or "")  # deve per forza ritornare una stringa

# 2. Recupera lo score vettoriale del reranker dal candidato in questione
# In embeddings.top_k_search c'era 'scores' (plurale). Gestiamo entrambi: qui lo score è del candidato singolo
# restituisce un float che è il valore dello score => optional perchè potrebbe anche non esistere
def get_vec_score(cand: Dict[str, Any]) -> Optional[float]:
    if "score" in cand: # del singolo
        return float(cand["score"])
    if "scores" in cand: # dell'insieme dei token
        return float(cand["scores"])
    return None

# 3. definisce una funzione di attivazione sigmoidale di classificazione binaria per il reranker
def sigmoid(x: np.ndarray):
    return (1.0 / 1.0 + np.exp(-x)) # f(x) = 1 / 1 + e^(−x) dove 'e' è il numero di Eulero che vale circa 2,71828

# 4. definisce una funzione di scaling normalization MinMaxScaler() per il reranker
# la funzione min_max prende due estremi (basso e alto) e normalizza i dati dentro quel range
def min_max_scaler(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0: # se l'array è vuoto lo restituisce
        return arr
    low = float(arr.min())
    high = float(arr.max())
    if high - low < 1e-12: # Se l’intervallo è praticamente nullo (hi - lo < 1e-12) → significa che i valori sono tutti uguali (o quasi): ritorna tutti zero per evitare instabilità numeriche.
        return np.zeros_like(arr)
    # x′ = (x - min) / (max - min)​ ∈ [0,1]
    return (arr - low) / (high - low) # altrimenti, normalizzo min-max

# 5. definisce una funzione di scaling standardization StandardScaler() per le features del reranker
def standard_scaler(x, axis=0, ddof=1, eps=1e-12, return_params=False):
        """
        Standardizza x: z = (x - μ) / σ
        - axis=0: per-feature (righe = campioni, colonne = feature)
        - ddof=0: std popolazione (come scikit-learn); 1 per std campionaria di default
        - eps: evita divisioni per zero -> fondamentale in un'operazione di normalizzazione
        - return_params=True -> ritorna anche (μ, σ) per riuso su test set
        """
        x = np.asarray(x, dtype=float) # caricamento array di float da normalizzare
        if x.size == 0:
            return (x, None, None) if return_params else x
        
        μ = np.nanmean(x, axis=axis, keepdims=True) # parametro μ dello scaler
        σ = np.nanstd(x, axis=axis, keepdims=True) # parametro σ dello scaler
        # clamp di sicurezza
        σ = np.where(σ < eps, 1.0, σ) # non possono essere minori di 0

        # formula della standardizzazione con deviazione standard
        z = (x - μ) / σ
        if return_params: # ritorno i vari parametri appiattiti perché perché NumPy estende automaticamente (1, n_features) su tutte le righe di X (forma (n_samples, n_features)).
            # a noi invece interessa solamente il valore effettivo e non [[μ, σ]] dove si avrebbe dimensione 1 della lista che contiene la lista con i valori e dimensione 2 della lista dei valori
            # voglio ottenere SOLO la dimensione della lista dei valori => in questo caso: 2
            return z, np.squeeze(μ, axis=axis), np.squeeze(σ, axis=axis)
        # in entrambi casi restituisco il risultato della deviazione standard
        return z

# Gestione del cross-encoder reranker
MODEL: Optional[CrossEncoder] = None # inizializzazione

# prendo il puntatore al modello dal config.py
def get_ranker() -> CrossEncoder:
    global MODEL
    if MODEL is None:
        # trust_remote_code=False per sicurezza; il modello MiniLM è leggero e gira bene su CPU
        # SETTARE: device="gpu" se è disponibile
        MODEL = CrossEncoder(RERANKER_NAME, device="cpu", trust_remote_code=False) # indirizzo il reranker da config.py
    return MODEL

# Gestione delle API interface del core principale del Reranker
def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int, # K top candidates
    blend_with_vector: bool = True,
    α: float = RERANK_ALPHA
) -> List[Dict[str, Any]]:
    """
    Rerank basato su cross-encoder model.
    - query: testo della query
    - candidates: lista di dict; si aspetta metadata['text'] o text
      (compatibile con embeddings.top_k_search)
    - top_k: quanti risultati finali restituire
    - blend_with_vector: se True fa blending con lo score vettoriale originario
    - alpha: peso del reranker nel blending [0..1]

    Ritorna una lista di candidate arricchiti con:
      - 'rerank_logit' (raw score del cross-encoder)
      - 'rerank_score' (sigmoid)
      - 'combined_score' (se blend attivo)
    Ordinati per score decrescente.
    """

    # se è vuoto, ritorno lista vuota
    if not candidates:
        return []
    
    model = get_ranker() # richiamiamo il reranker

    # si gestiscono le coppie da utilizzare formate da query e numero di candidati su cui tale query viene richiamata => (query, passage)
    # Estrae il testo di ogni candidato (da c["text"] o c["metadata"]["text"], dipende da extract_text).
    # Costruisce le coppie (query, passage) richieste dal cross-encoder.
    '''query = la domanda dell’utente (stringa). Es.: "Qual è la politica di cancellazione?"
       passage = il testo del candidato (un singolo chunk/documento breve) su cui valutare la rilevanza rispetto alla query.
       pairs = [(query, p) for p in passages] costruisce la lista di coppie (query, passage) richiesta dal cross-encoder.
       Cross-encoder (BERT/DeBERTa MiniLM, bge-reranker, ecc.) si aspettano proprio una lista di tuple (domanda, testo_candidato) per calcolare 1 punteggio di rilevanza per coppia.'''
    passages = [extract_text(c) for c in candidates]
    pairs = [(query, p) for p in passages]

    # passo successivo: inferenza cross-encoder -> logits
    # il cross-encoder MiniLM fa return su un 1 logit correlato per coppia -> prevale rilevanza
    # Inferenza del cross-encoder: per ogni coppia produce un logit (valore reale, non limitato).
    # Applica una sigmoide: mappa i logit in [0, 1] → comodo per blending e soglie.
    # Intuizione: più alto il logit, più il documento è rilevante per la query, più è probabile che venga utilizzato per la risposta finale
    logits = np.asarray(model.predict(pairs), dtype=np.float32).reshape(-1)
    rerank_score = sigmoid(logits)  # grazie alla sigmoide -> risultato tra 0..1
    
    # Se sono disponibili, si utilizzano anche gli score vettoriali del retriever (quelli prima di aver usato il reranker)
    vals = [get_vec_score(c) for c in candidates]
    vec_scores = np.array([v if v is not None else np.nan for v in vals], dtype=np.float32)
    has_vec = ~np.isnan(vec_scores) # has_vec è una maschera booleana: “ha uno score valido?”.
    # se c'è score valido, normalizzo immediatamente lo score
    if has_vec.any():
        vs = vec_scores[has_vec]
        vec_scores_scaled = vec_scores.copy()
        vec_scores_scaled[has_vec] = min_max_scaler(vs)
    else: # altrimenti restituisco 0
        vec_scores_scaled = np.zeros_like(vec_scores)

    # Blending: retrevial score + reranking score
    if blend_with_vector == True and has_vec.any():
        combined_score = α * rerank_score + (1 - α) * vec_scores_scaled
    else: # se non c'è retrevial => reranking score only
        combined_score = rerank_score
    
    # Ogni punteggio deve essere appeso al proprio relativo candidato
    # costruendo una lista di dizionari in cui, per ogni candidato, vi è il suo punteggio combinato e singolo
    enriched: List[Dict[str, Any]] = [] # initialization
    for candidate, logit, rr_s, comb_s, vec_ss in zip(candidates, logits, rerank_score, combined_score, vec_scores_scaled):
        cc = dict(candidate)
        cc["rerank_logit"] = float(logit)
        cc["rerank_score"] = float(rr_s)
        if blend_with_vector == True and has_vec.any():
            cc["vec_score_norm"] = float(vec_ss)
            cc["combined_score"] = float(comb_s)
        else:
            cc["vec_score_norm"] = np.nan
            cc["combined_score"] = np.nan
        enriched.append(cc)

    # ordinamento finale e clamp
    key = "combined_score" if ("combined_score" in enriched[0]) else "rerank_norm"
    enriched.sort(key=lambda x: x[key], reverse=True) # sorting the list in DESC order to analyze the top_k biggest score
    # restituisco i top_k punteggi
    return enriched[: max(1, top_k)]

# funzione helper per estrarre id in modo stabile (sia che si tratti di hash sia che si tratti di quello tradizionale)
def stable_id(item: Dict[str, Any]) -> str:
    cid = item.get("chunk_id")
    if cid:
        return str(cid)
    # fallback: hash stabile del testo estratto
    text = extract_text(item)  # funzione per estrarre testo
    return "hash:" + hashlib.sha1(text.encode("utf-8")).hexdigest()

# Questa funzione implementa - Reciprocal Rank Fusion (RRF): 
# un metodo semplice e robusto per fondere più classifiche (liste ordinate di risultati) 
# in un’unica classifica senza dover allineare o normalizzare gli score originali.
# Ogni elemento deve avere un id stabile per essere tenuto in considerazione
# — usiamo 'chunk_id' se presente, altrimenti hash del testo.
'''
Input: ranked_lists è una lista di liste; ogni sotto-lista 
       è una classifica (es. output di FAISS, BM25, cross-encoder, motori diversi, query espanse, ecc.).
Per ogni lista, si iterano gli elementi in ordine e si assegna a ciascun item un punteggio additivo:

- contributo = 1 / 𝑘 + rank
- dove rank parte da 1 (1° posto, 2° posto, …) e k (tipicamente 60) smorza 
  le differenze tra posizioni molto alte/basse e stabilizza la fusione.
- aggiunge 'rrf_score'
- preserva l'oggetto 'migliore' visto per ciascun id
- Identificazione item:

    - usa chunk_id se presente,
    - altrimenti crea un id fallback tramite helper "stable_id" con hash(extract_text(item)) 
      (piccolo rischio di collisioni; meglio un hash stabile tipo hashlib.sha1 del testo normalizzato).

- Somma i contributi su tutte le liste per lo stesso id → score aggregato.
- Ordina in base allo score e ritorna List[Tuple[item_id, score]].'''
def reciprocal_rank_fusion(
        # si ha in ingresso una lista di liste di dizionari 
        # che verrà collassata in una sola lista di tuple (item_id, score_correlato) 
        # ordinata DESC con i relativi punteggi "fusi" di ogni lista di dizionari
        ranked_lists: List[List[Dict[str, Any]]], # lista ordinata di liste di dizionari 
        k: int = 60, # iperparametro RRF con default value = 60
        top_n: int | None = None
) -> List[Tuple[str, float]]:
    
    scores = Dict[str, float] = {} # buffer temporaneo per memorizzare i punteggi di ogni lista
    best_item: Dict[str, Dict[str, Any]] = {} # buffer temporaneo per memorizzare il punteggio più alto trovato

    for l in ranked_lists: # scorro la lista di liste ordinate
        for rank, item in enumerate(l, start=1): # scorro la lista di dizionari annidata e per ogni dizionario
            cid = stable_id(item) # raccolgo suo ID nel contenuto del dizionario corrente
            # assegno il punteggio per ciascun ID esistente
            # con formula RRF => 1 / (k + rank)​
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            # trovo il risultato migliore
            if cid not in best_item:
                best_item[cid] = item
            else:
                # ricerca del massimo tra i best_item
                prev = best_item[cid]
                if len(extract_text(item)) > len(extract_text(prev)):
                    best_item[cid] = item
    
    # costruzione lista arricchita con gli oggetti completi
    # Perché serve?
    # 1. Si passano direttamente i chunk (con testo + metadata) allo step successivo, senza dover fare un secondo join da id→oggetto.
    # 2. Si aggiornano i campi (es. aggiungere rrf_score, tenere il miglior oggetto visto, evitare collisioni d’hash).
    # 3. Meno fragilità: se cambi formazione dell’ID, l’oggetto è già lì.
    fused = []
    for cid, score in scores.items():
        obj = dict(best_item[cid]) # dizionario con i punteggi migliori
        obj["chunk_id"] = obj.get("chunk_id", cid)
        obj["rrf_score"] = float(score)
        fused.append(obj)
    # ordino lista DESC secondo il punteggio
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    # se voglio restituire solo una parte della lista devo dare un valore a top_n
    # altrimenti viene restituita tutta
    return fused if top_n is None else fused[:top_n] 