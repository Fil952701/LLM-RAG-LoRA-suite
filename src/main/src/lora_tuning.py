# lora_tuning.py
"""
Fine-tuning LoRA/QLoRA per l'adattamento ai dati aziendali e la normalizzazione JSON (RAG mapping).
- Input: coppie (raw_form.json, gold_normalized.json) in due JSONL paralleli
- Output: adapter LoRA (e opzionalmente modello mergiato)
Il file:
- legge due JSONL in parallelo (--train-inputs + --train-gold, idem per --val-*);
- costruisce il prompt con lo stesso stile che usi già (riutilizza funzioni di input_mapping_normalization.py se disponibili, altrimenti ricade su un prompt interno);
- supporta LoRA classica su CPU o QLoRA (8-bit/4-bit) su GPU se bitsandbytes è disponibile;
- usa peft + transformers (Trainer) con Causal LM;
- salva l’adapter LoRA fine-tunato e, opzionalmente, la versione merged del modello (--save-merged);
- è CPU-friendly per default, ma con flag per bf16/cuda GPU se presenti.

Esempi:
  # LoRA standard su CPU
  python lora_tuning.py \
    --base-model microsoft/Phi-3.5-mini-instruct \
    --train-inputs data/train_raw.jsonl \
    --train-gold   data/train_gold.jsonl \
    --val-inputs   data/val_raw.jsonl \
    --val-gold     data/val_gold.jsonl \
    --output-dir   adapters/phi35-jsonmap-lora \
    --epochs 2 --batch 4 --grad-accum 8 --max-len 2048

  # QLoRA (con CUDA + bitsandbytes) su GPU
  python lora_tuning.py \
    --base-model mistralai/Mistral-7B-Instruct-v0.3 \
    --train-inputs ... --train-gold ... \
    --val-inputs ...   --val-gold ... \
    --output-dir adapters/mistral-jsonmap-qlora \
    --use-qlora --bf16 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
"""
from __future__ import annotations
import os, json, argparse, math, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset

# librerie fondamentali di ML per effettuare ri-addestramento modello
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# import bitsandbytes
try:
    import bitsandbytes as bnb  # noqa: F401
    BNB_EXISTS = True
except Exception:
    BNB_EXISTS = False