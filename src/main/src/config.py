# config.py

import os
import torch as thc

# Dati/artefatti
NORMALIZED_JSONL = os.getenv("NORMALIZED_JSONL", "normalized_requests.jsonl") # file JSON normalizzato in uscita
DOCS_DIR         = os.getenv("DOCS_DIR", "docs_raw")          # cartella con PDF/HTML/TXT originali
CHUNKS_JSONL     = os.getenv("CHUNKS_JSONL", "chunks.jsonl")  # output chunking per la gestione del JSON
FAISS_INDEX      = os.getenv("FAISS_INDEX", "faiss.index")
FAISS_METADATA   = os.getenv("FAISS_META",  "faiss_meta.jsonl")
DATA_DIR         = os.getenv("DATA_DIR")
DEVICE           = os.getenv("DEVICE", "cuda:0" if thc.cuda.is_available() else "cpu")
DTYPE            = thc.float16 if DEVICE.startswith("cuda") else thc.float32

def device_map_for_transformers():
    """
    Per transformers.from_pretrained:
    - su GPU: metto tutto sulla stessa GPU
    - su CPU: resto su CPU
    """
    return {"": DEVICE} if not os.getenv("HF_DEVICE_MAP_AUTO") else "auto"

# Scelte modello
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")  # veloce su CPU
GEN_MODEL_NAME   = os.getenv("GEN_MODEL_NAME", "microsoft/Phi-3.5-mini-instruct")  # cambiare in Qwen quando c'è GPU

# Chunking
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150")) # sovrapposizione tra un chunk e l'altro

# DB / Vector store
USE_PGVECTOR = os.getenv("USE_PGVECTOR", "0") == "1"
PG_DSN = os.getenv("PG_DSN", "postgresql://user:pass@localhost:5432/mydb")
PG_TABLE = os.getenv("PG_TABLE", "rag_chunks")

# Reranker
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_ALPHA = 0.7  # peso del reranker nel blending con il punteggio vettoriale per generare la risposta finale

# Numero di token da ammettere
TOKEN = 600

# Retrieval
TOP_K = int(os.getenv("TOP_K", "8"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "4"))