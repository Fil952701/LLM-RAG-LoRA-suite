# main.py: entrypoint unico per risposta e sua valutazione
# Valutazione pipeline di normalizzazione/mappatura (stampa mismatch a terminale)

"""
Valuta la qualità dell'output rispetto a un gold standard.

Modalità:
1. deterministica-only (USE_LLM=False)
2. ibrida con LLM (USE_LLM=True)

Input:
- --inputs RAW.jsonl     (ogni riga: raw form JSON)
- --gold   GOLD.jsonl    (ogni riga: JSON normalizzato atteso; chiavi come da SCHEMA)
Output:
- stampa riepilogo metriche a schermo
- stampa i mismatch a terminale (tabellina)
- salva metrics.json

Usage esempi:
  python eval.py --inputs raw.jsonl --gold gold.jsonl --use-llm 0 --show 100
  python eval.py --inputs raw.jsonl --gold gold.jsonl --use-llm 1 --only-fields treatment_code request_target
  python eval.py --inputs raw.jsonl --gold gold.jsonl --per-record
"""
import argparse, json, orjson, re, sys
from typing import Dict, Any, List, Tuple
import input_mapping_normalization as imn
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import GEN_MODEL_NAME, DTYPE, TOKEN, device_map_for_transformers
from collections import defaultdict

# chiavi dell'input iniziale obbligatorie
REQ_KEYS = [
    "id_attivita","id_record","source_name","request_lang","table_name",
    "request_target","treatment","treatment_code","accommodation_type",
    "request_date","check_in_date","check_out_date",
    "adults_number","children_number","pet","children_age",
    "country_code","campaign_data","attribution_data"
]

# pattern regex compilati usati per riconoscere stringhe in un formato specifico.
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DT_RE   = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

# utilities per riprodurre la pipeline e valutarne le metriche in stile TEST
# 0. risposta test al prompt richiesto
def test_answer(ns):
    try:
        from rag_pipeline import answer
        output = answer(
            query=ns.query,
            top_k=ns.k if ns.k else None,
            top_k_rerank=ns.kr if ns.kr else None,
            max_new_tokens=ns.max_new_tokens
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        print(f"Risposta: {output}\n")
    except Exception as e:
        # fallback: solo deterministico e proseguo
        print(f"[FATAL ERROR]: {e}\nRisposta: '{answer}' non conforme alle specifiche.", file=sys.stderr)
        sys.exit(1)

# 1. caricamento JSON
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    try:
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.load(line))
    except FileNotFoundError as fe:
        print(f"Eccezione sollevata: {fe}\nFile non esistente.\n")
    return rows

# 2. funzione booleana per effettuare paragone del risultato atteso e quello ottenuto
def eq(x, y) -> bool:
    if isinstance(x, list) and isinstance(y, list):
        return x == y
    return str(x) == str(y)

# 3. funzione booleana per verificare la correttezza del formato dei dati
def format_ok(key: str, val: Any) -> bool:
    if key == "request_date":
        return isinstance(val, str) and bool(DT_RE.match(val))

# 4. funzione per troncare la lunghezza dei max_token
def truncate(s: str, maxlen: int = 140) -> str:
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    if len(s) <= maxlen: return s # se rispetta la lunghezza la restituisco
    return s[:maxlen-1] + "_" # altrimenti tronco

# 5. funzione booleana generica per stringhe
def c(enabled: bool, code: str) -> str:
    return code if enabled else ""

# 6. pipeline di normalizzazione dei dati di test sulla base della pre_normalization già creata
def normalize_test(raw: Dict[str, Any], use_llm: bool) -> Dict[str, Any]:
    pre = imn.pre_normalization(raw)["pre"]
    base = imn.build_final_candidate(pre, raw) # schema json normalizzato
    # se non usiamo LLM, usiamo una valutazione deterministica
    if not use_llm:
        return imn.validate_or_fix(base)
    try:
        tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME,
            dtype=DTYPE,
            device_map=device_map_for_transformers(), # cambiare a "gpu" quando sarà disponibile
            low_cpu_mem_usage=True,
            trust_remote_code=False
        )
        # Input unico ibrido per LLM: raw + hints_pre
        llm_input = {
            "hints_pre": pre,        # indizi deterministici (il modello può copiarli perché sono sempre quelli)
            "raw_form":  raw         # raw originale
        }

        # prompt messages
        system_msg = "Sei un valido assistente per analizzare JSON di richieste informazioni turistiche e di generare un JSON formattato"
        user_msg = imn.make_user_message(llm_input, imn.istruzioni, imn.schema_atteso)
        text = f"{system_msg}\n\n{user_msg}\n{{"
        inputs = tok([text], return_tensors="pt")   # testo del prompt trasformato in tensore
        # se il tokenizer non ha pad_token, riallineo al eos
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        # inferenza da parte del modello
        with imn.thc.inference_mode():
            out = mdl.generate(
                **inputs,  # tensore di input
                max_new_tokens=TOKEN,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )
        # costruzione del prompt finale
        # associo la relativa parte di input alla parte di output corrispondente
        gen_only = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        raw_out = tok.batch_decode(gen_only, skip_special_tokens=True)[0]
        # estrazione del primo JSON valido dai dati tokenizzati
        draft = imn.extract_first_json(raw_out)
        # whitelist delle chiavi che LLM può andare a toccare
        llm_keys = [
            "request_target","treatment","treatment_code","accommodation_type",
            "request_lang","country_code","pet", "adults_number","children_number","children_age"
        ]

        # Backfill -> se è già valorizzato deterministicamente non vado a sovrascriverlo
        # se invece non è presente allora lo valorizzo
        prefer_base_if_present = {
            "id_attivita","id_record","source_name","table_name", "check_in_date",
            "check_out_date", "request_date", "campaign_data","attribution_data", "source_reference"}
        for k in llm_keys + list(prefer_base_if_present):
            if k in draft and draft[k] is not None:
                if k in prefer_base_if_present and base.get(k) not in (None, "", []):
                    # già valorizzato in modo deterministico: non toccare
                    continue
                base[k] = draft[k]

        # associo le chiavi che mi servono a quelle del dizionario per fare un merge controllato
        if isinstance(draft, dict):
            for k in llm_keys:
                if k in draft and draft[k] is not None:
                    base[k] = draft[k]

    except Exception:
        # fallback: solo deterministico e proseguo
        pass

    # schema di ritorno validato come JSON
    return imn.validate_or_fix(base)

# 7. valutazione a terminale delle prestazioni e dei mismatch tra expected and predicted
def print_mismatches(
        mismatches: List[Tuple[int, str, Any, Any]],
        show: int,
        color: bool,
        per_record: bool
):
    # codifica di colori per evidenziare a terminale i mismatches
    bold = c(color, "\033[1m")
    dim  = c(color, "\033[2m")
    red  = c(color, "\033[31m")
    grn  = c(color, "\033[32m")
    yel  = c(color, "\033[33m")
    rst  = c(color, "\033[0m")

    # caso base: se non c'è mismatch stampo un log e chiudo
    if not mismatches:
        print(f"{grn}Nessun mismatch trovato.{rst}")
        return
    # analizzo ogni record e cerco mismatch
    if per_record:
        # raggruppo per indice
        byrec = defaultdict(list)
        for idx, k, pred, gold in mismatches:
            byrec[idx].append((k, pred, gold))
        shown = 0 # counter for mismatches tracking
        for idx in sorted(byrec.keys()):
            if shown >= show: # esco dal ciclo se i mismatches ottenuti superano quelli che ho richiesto di vedere
                break
            print(f"\n{bold}Record #{idx}{rst}")
            # per ogni record stampo il numero di mismatches
            for (k, pred, gold) in byrec[idx]:
                print(f"  {yel}{k}{rst}")
                print(f"    pred: {red}{truncate(pred)}{rst}")
                print(f"    exp: {truncate(gold)}")
            shown += 1
        return shown # ritorno contatore per comodità (potrei anche omettere il return)
    
    # Flat table-like con il resoconto troncando i risultati per risparmiare memoria
    print(f"{bold}MISMATCH (max {show}){rst}")
    header = f"{bold}{'idx': > 5} {'key': < 22} {'pred': < 50} {'exp': < 50}{rst}"
    print(header)
    print(dim + "-"*len(header) + rst)
    for _, (idx, k, pred, gold) in enumerate(mismatches[:show], start=1):
        print(f"{idx:>5}  {k:<22}  {red}{truncate(pred, 50):<50}{rst}  {truncate(gold, 50):<50}")

# 8. funzione per valutare le metriche sul test effettivo finale
def eval(ns):
    color = not ns.no_color
    fields = set(ns.only_fields) if ns.only_fields else set(REQ_KEYS)
    # import dei dati
    raw_rows  = load_jsonl(ns.inputs)
    gold_rows = load_jsonl(ns.gold)
    if len(raw_rows) != len(gold_rows):
        print(f"[WARN] Lunghezze diverse inputs({len(raw_rows)}) vs gold({len(gold_rows)})", file=sys.stderr)

    minimum = min(len(raw_rows), len(gold_rows)) # calcola il minimo tra i due
    correct = {k:0 for k in REQ_KEYS} # le chiavi corrette sono quelle dentro REQ_KEYS, quindi se ne trovo una la classifico corretta
    fmt_ok   = {k:0 for k in ("request_date","check_in_date","check_out_date")}
    schema_ok = 0

    # lista dei mismatches che vengono trovati
    mismatches: List[Tuple[int,str,Any,Any]] = []   # idx, k, pred, gold

    # routine per validare le prestazioni
    for i in range(minimum):
        pred = normalize_test(raw_rows[i], bool(ns.use_llm)) # predicted 
        exps = gold_rows[i] # expected
        # validazione dello schema
        try:
            import input_mapping_normalization as imn # lazy import per non saturare RAM
            imn.validate_or_fix(pred)
            schema_ok += 1
        except Exception:
            pass

        # field-wise
        for k in REQ_KEYS:
            if k not in fields:
                continue
            pm_pred = pred.get(k)
            gm_exp = exps.get(k)
            compare = eq(pm_pred, gm_exp)
            if compare == True:
                correct[k] += 1
            else:
                mismatches.append((i, k, pm_pred, gm_exp))

        # format-wise
        for k in ("request_date","check_in_date","check_out_date"):
            if format_ok(k, pred.get(k)): 
                fmt_ok[k]+=1

    # dizionario con metriche finali
    metrics = {
        "n_eval": minimum,  # numero di valutazioni effettuate
        "schema_pass_rate": schema_ok / max(1, minimum), # tasso di successo in percentuale
        "field_accuracy": {k: correct[k] / max(1, minimum) for k in REQ_KEYS if k in fields},
    }

    # stampa finale
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Metriche: {metrics}\n")
    print_mismatches(mismatches, show=ns.show, color=color, per_record=ns.per_record)

    # creo il file json per le metriche
    if not ns.no_metrics_file:
        with open(ns.metrics_json, "wb") as f:
            f.write(orjson.dumps(metrics, option=orjson.OPT_INDENT_2))

# 9. Funzione per il parsing a terminale
def parsing():
    ap = argparse.ArgumentParser(
        prog="rag-model",
        description="RAG pipeline runner & evaluator"
    )
    sp = ap.add_subparsers(dest="cmd", required=True)

    # answer    
    ap_ans = sp.add_parser("answer", help="Esegue la pipeline RAG e genera una risposta")
    ap_ans.add_argument("-q", "--query", required=True, help="Domanda utente")
    ap_ans.add_argument("--k", type=int, default=None, help="Top-k retriever (override)")
    ap_ans.add_argument("--kr", type=int, default=None, help="Top-k reranker (override)")
    ap_ans.add_argument("--max_new_tokens", type=int, default=350)
    ap_ans.set_defaults(func=test_answer)

    # evaluation
    ap_eval = sp.add_parser("eval", help="Valuta la normalizzazione rispetto a un gold standard")
    ap_eval.add_argument("--inputs", required=True)
    ap_eval.add_argument("--gold",   required=True)
    ap_eval.add_argument("--use-llm", type=int, default=0)
    ap_eval.add_argument("--show", type=int, default=50, help="quanti mismatch mostrare a terminale")
    ap_eval.add_argument("--only-fields", nargs="*", help="valuta/mostra solo questi campi")
    ap_eval.add_argument("--per-record", action="store_true", help="stampa mismatch raggruppati per record")
    ap_eval.add_argument("--no-color", action="store_true", help="disabilita colori ANSI")
    ap_eval.add_argument("--metrics-json", default="metrics.json")
    ap_eval.add_argument("--no-metrics-file", action="store_true")
    ap_eval.set_defaults(func=eval)

    return ap

# procedura main
def main():
    ap = parsing()
    ns = ap.parse_args()
    ns.func(ns)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)