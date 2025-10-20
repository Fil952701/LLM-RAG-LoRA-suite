import torch as thc
import os, re, json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Funzione per estrarre il primo oggetto JSON valido dalla stringa di risposta usando regex e bilanciamento parentesi 
def extract_first_json(s: str):
    # 1) tentativo “semplice” con regex non greedy tra { ... }
    m = re.search(r'\{.*?\}', s, flags=re.S)
    if m:
        # 2) bilanciamento parentesi per evitare tagli
        start = m.start()
        depth = 0
        for idx in range(start, len(s)):
            if s[idx] == '{':
                depth += 1
            elif s[idx] == '}':
                depth -= 1
                if depth == 0:
                    candidate = s[start:idx+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
    return None

# Pulisce la cache CUDA all'inizio per evitare OOM e seleziona GPU d'uso
thc.cuda.empty_cache()
print("CUDA available:", thc.cuda.is_available())
print("CUDA version (compiled):", thc.version.cuda)
if thc.cuda.is_available():
    print("GPU:", thc.cuda.get_device_name(0))
    x = thc.randn(2,2, device="cuda")
    print(x)

# Modello Qwen 2.5 7B Instruct (ottimizzato per GPU con 8GB+ VRAM)
#MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Modello Phi-3.5 Mini Instruct (4.3B) di Microsoft (ottimizzato per CPU)
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# Impostazioni per prestazioni e log
# TORCH_CPP_LOG_LEVEL='info' per log dettagliati di PyTorch (debug)
# TRANSFORMERS_VERBOSITY='info' per log dettagliati di Transformers (debug)
# TORCH_CPP_LOG_LEVEL='error' per log solo errori (default)
TRANSFORMERS_VERBOSITY='info'
TORCH_CPP_LOG_LEVEL='info'
TORCH_CPP_LOG_LEVEL='error'

# Imposta variabili d'ambiente per log
os.environ["TRANSFORMERS_VERBOSITY"] = TRANSFORMERS_VERBOSITY
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # log silenziosi di default
os.environ["TORCH_CPP_LOG_LEVEL"] = TORCH_CPP_LOG_LEVEL
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # per debug CUDA (più lento)
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # per debug OOM (più lento)
#os.environ["TOKENIZERS_PARALLELISM"] = "false"  # per evitare warning sui tokenizer (più lento)

# Consigliato su PyTorch 2.x per migliori prestazioni su GPU NVIDIA con Tensor Cores
# Imposta precisione matmul a "high" per usare TF32 e FP16/FP8 quando possibile
thc.set_float32_matmul_precision("medium") # "highest" è prestazioni massime, "high" è preferito, "medium" è più lento ma più preciso

# Carica tokenizer e modello. Con RTX 4090, FP16 automatico va bene.
# con pretrained non c'è problema di utilizzo pesi in fp16
# se si usa fine-tuning con PEFT, conviene salvare in fp16
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# Caricamento modello pre-addestrato con ottimizzazioni per memoria e velocità
# Usa device_map="auto" per distribuire su più GPU se necessario
# low_cpu_mem_usage=True per ridurre l'uso di RAM
# offload_folder e offload_state_dict per fallback offload su disco se RAM/GPU non bastano
# trust_remote_code=False per sicurezza (True solo se il repo richiede codice custom)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    #dtype="auto",               # usa FP16/FP8 se disponibile
    #device_map="auto",          # mette il modello sulla GPU con accelerazione
    dtype="float32",            # forza FP32 (per test su CPU)
    device_map={"": "cpu"},     # forza il modello in CPU (per test su CPU)
    low_cpu_mem_usage=True,     # riduce l'uso di RAM
    trust_remote_code=False,    # True solo se il repo richiede codice custom
    offload_folder="offload",   # cartella per offload su disco se RAM/GPU non bastano
    offload_state_dict=True     # offload pesi su disco se RAM/GPU non bastano
)

# Schema JSON per la risposta strutturata di Qwen
# la risposta deve seguire esattamente questo schema JSON
# "sources" è una lista vuota se non ci sono fonti
schema_hint = (
    'Rispondi SOLO con UN UNICO oggetto JSON: '
    '{"answer":"testo breve in italiano inerente alla domanda","sources":"[]"}. '
    'Se non sai rispondere: {"answer":"","sources":[]}. '
    'Rispetta coerentemente la grammatica italiana nella risposta.'
    'Dai importanza alla brevità, alla precisione e concisione nella risposta.'
)

# Esempio di domanda con schema JSON per la risposta strutturata di Qwen
# la risposta deve essere breve e concisa
messages = [
    {"role": "system", "content": "Sei Qwen e rispondi SOLO in JSON: {\"answer\":\"...\"\n,\"sources\":[] }.\n"},
    {"role": "user", "content": f'{schema_hint}\nDomanda: Che azienda è "Titanka! Nati per il turismo S.p.A." in poche righe?'}
]

# Applica il template chat di Qwen e prepara input
# add_generation_prompt=True aggiunge il prompt di generazione alla fine
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False, # non tokenizza qui, lo fa il tokenizer apposito dopo
    add_generation_prompt=True  # aggiunge il prompt di generazione alla fine
)

# Generazione testo con Qwen 2.5 7B Instruct (ottimizzato per GPU con 8GB+ VRAM)
inputs = tokenizer([text], return_tensors="pt") # tokenizzazione input e manda a GPU

# Genera output con torch.inference_mode() per efficienza e velocità
with thc.inference_mode():
    output_ids = model.generate(
        **inputs,               # puntatore di input_ids e attention_mask da inputs
        max_new_tokens=400,     # massimo 400 token generati
        do_sample=False,        # greedy per stabilità
        pad_token_id=tokenizer.eos_token_id, # padding con eos_token_id
        eos_token_id=tokenizer.eos_token_id, # termina alla fine del testo
        temperature=0.0,        # temperatura 0 per greedy
        top_p=0.7,              # top-p sampling (niente se temperature=0)
        top_k=40,               # top-k sampling (niente se temperature=0)
        repetition_penalty=1.1, # penalità per ripetizioni (1.0 = nessuna)
        num_return_sequences=1, # una sola sequenza di output
    )

# Ritaglia solo la parte generata (esclude input)
# calcola la lunghezza di input per ogni batch item
gen_only = [o[len(idx):] for idx, o in zip(inputs.input_ids, output_ids)] # gen_only è una lista di tensori formata da input_ids che vengono zippati insieme agli output_ids

# Decodifica e stampa la risposta generata
# skip_special_tokens=True rimuove token speciali come <s>, </s>, <pad>
response = tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]
print(f"\nRisposta: {response}\n")

# Estrai il primo oggetto JSON dalla risposta e stampalo
'''first_json = extract_first_json(response) or '{"answer":"","sources":""}'
print(f"\nRisposta: {first_json}\n")'''