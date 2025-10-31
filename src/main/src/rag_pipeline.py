# rag_pipeline.py

from __future__ import annotations
import os, json
from typing import List, Dict, Any, Optional, Tuple
import torch as thc # libreria PyTorch per i modelli transformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import (
    GEN_MODEL_NAME, EMBED_MODEL_NAME, TOP_K, TOP_K_RERANK,
    FAISS_INDEX, FAISS_METADATA, TOKEN, DTYPE, device_map_for_transformers
)
from embeddings import load_faiss, top_k_search
from reranker import rerank

# Gestione di tutta la pipeline di esecuzione RAG
# inizializzazione modelli
GEN = None
TOK = None

# helper per gestione del modello nella pipeline del RAG
def get_generator():
    global GEN, TOK
    if GEN is None or TOK is None:
        TOK = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, use_fast=True)
        # con solo CPU -> dtype=float32; con GPU -> torch.float16/bfloat16
        GEN = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            dtype=DTYPE,
            low_cpu_mem_usage=True,
            device_map=device_map_for_transformers(), # cambiare a "gpu" quando sarà disponibile
            trust_remote_code=False
        )
    # restituisco i modelli (LLM e EMBEDDER) ottenuti
    return GEN, TOK

# prompt example => cambiare a piacimento mantenendo la logica del RAG (generare diverso per newsletter, offerte ecc.)
SYSTEM_PROMPT = (
    "Sei un assistente altamente specializzato per hotel/villaggi. Rispondi in modo conciso e con citazioni sorgente.\n"
    "Se l'informazione non è nei documenti, dillo esplicitamente."
)

# utility per fare analizzare il prompt scritto al modello e correlare la risposta che andrà a generare ad esso
def build_prompt(query: str,
                 passages: List[Dict[str, Any]]
) -> str:
    ctx_blocks = []
    for i, c in enumerate(passages, 1):
        meta = c.get("metadata") or {} # estrapolo i metadati
        txt = meta.get("text", "")
        src = meta.get("source") or meta.get("url") or meta.get("chunk_id")
        # creazione del blocco di analisi del prompt
        ctx_blocks.append(f"[{i}] (source={src})\n{txt}") # si ha la formulazione del blocco di analisi del prompt in {i}
                                                          # ad esso si forniscono indirizzo della fonte dei dati {src} e testo preso dalla fonte {txt}
    context = "\n\n".join(ctx_blocks) if ctx_blocks else "Nessun contesto da visualizzare."
    # restituisco l'auto-analisi del PROMPT da parte del modello PRIMA di fornire la risposta
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n</|system|>\n"
        f"<|user|>\nDomanda: {query}\n\nContesto:\n{context}\n\n"
        "Regole:\n- Cita i numeri dei blocchi tra [ ] accanto alle frasi usate.\n"
        "- Se mancano dati, scrivi cosa manca senza inventare nulla.\n</|user|>\n<|assistant|>\n"
    )

# cuore dell'elaborazione del RAG model => la risposta che il modello deve generare
def answer(query: str, top_k: int = TOP_K, top_k_rerank: int = TOP_K_RERANK, max_new_tokens: int = TOKEN) -> Dict[str, Any]:
    # gestione dei passaggi della creazione della risposta per il RAG

    # 1) Load FAISS vectorial indexes + metadati
    index, metas = load_faiss(FAISS_INDEX, FAISS_METADATA)

    # 2) Vector retrieval answer
    initial = top_k_search(query=query, k=top_k, index=index, metas=metas)

    # 3) Reranking answer
    reranked = rerank(query, initial, top_k=top_k_rerank, blend_with_vector=True)

    # 4) Prompt + answer generation
    gen, tok = get_generator()
    prompt = build_prompt(query, reranked)
    inputs = tok([prompt], return_tensors="pt")
    with thc.inference_mode():
        out = gen.generate(
            **inputs,   # tensore multidimensionale (uso due '*' per indicarlo)
            max_new_tokens=max_new_tokens,
            do_sample=False,
            #temperature=0.0, # abilitare e settare solo con "do_sample=True"
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    # 5) decoding della parte generata
    gen_only = out[:, inputs["input_ids"].shape[1]:]
    text = tok.batch_decode(gen_only, skip_special_tokens=True)[0].strip() # pulizia della risposta

    # 6) restituzione del dizionario della risposta generata con reranked e query di prompt
    return {
        "query": query,
        "answer": text,
        "contexts": reranked,
    }