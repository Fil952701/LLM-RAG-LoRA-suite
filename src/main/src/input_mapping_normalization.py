# Con questo file andiamo ad effettuare tutte le mappature necessarie sui dati di input PRIMA di installare il RAG
# Ad esempio, possiamo convertire i dati in un formato JSONL standardizzato, rimuovere campi inutili, rinominare chiavi, ecc.
# Questo aiuta a mantenere il codice di caricamento dati pulito e modulare.
# Possiamo anche aggiungere funzioni di validazione per assicurarci che i dati siano nel formato corretto prima di procedere con l'addestramento o l'inferenza.
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"            # warning+
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"           # disattiva oneDNN msg
import datetime
import orjson
import re
from typing import List, Dict, Any, Optional, Tuple, Counter
import torch as thc
import logging
from pprint import pprint
try:
    import MySQLdb  # mysqlclient (Linux/macOS)
    from MySQLdb.cursors import DictCursor, SSCursor
    MYSQL_FLAVOR = "mysqldb"
except ModuleNotFoundError:
    import pymysql as MySQLdb # PyMySQL (Windows)
    from pymysql.cursors import DictCursor, SSCursor
    MYSQL_FLAVOR = "pymysql"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from jsonschema import validate, ValidationError
from config import GEN_MODEL_NAME, DTYPE, ATTN_IMPL, device_map_for_transformers, DEVICE, TOKEN, BATCH, USE_LLM
from transformers import StoppingCriteria, StoppingCriteriaList

# GESTIONE DEI DATI DAL DB
# 1. Config
ALLOWED_ENVS = {"LOCAL", "DEPLOY"}
APP_ENV = (os.getenv("APP_ENV", "LOCAL") or "LOCAL").strip().upper()
if APP_ENV not in ALLOWED_ENVS:
    logging.warning(
        "APP_ENV '%s' non ammesso: fallback a 'LOCAL' (consentiti: %s)",
        APP_ENV, sorted(ALLOWED_ENVS)
    )
    APP_ENV = "LOCAL"
LOCAL_LIMIT = int(os.getenv("LOCAL_LIMIT", "10"))  # quanti record in LOCAL
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "1000")) # quanti record per fetchmany() in DEPLOY
USE_SSCURSOR = True  # Server-Side cursor per non saturare RAM in stream su milioni di righe
OUTPUT_JSONL = "normalized_requests.jsonl" # file per richieste
ERRORS_JSONL = "normalized_errors.jsonl"   # file per errori
TABLE_NAME = "archivio_email_da_hosting" # tabella di origine
JSON_COL   = "content_json"              # colonna che contiene il JSON del form
PK_COL     = "id"                        # facoltativo ma utile per logging/errori
DATE_COL   = "data"                      # per ordinare/filtrare in ASC o DESC
CAMPAIGN_COL = "info_campagna_json"      # dizionario da integrare nel dizionario presente sotto il campo corrispondente
ID_ATTIVITA_COL = "id_attivita"          # chiave da tenere in considerazione per fare join
ID_SOURCE_COL   = "id_source"            # usato per "source_reference"
SCRIPT_NAME_COL = "script_name"          # colonna del DB per source_reference

# contatori di OK e ERR
ok_count = 0
err_count = 0

# logging inizializzato
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# pulizia file di output
open("normalized_requests.jsonl","w").close()
open("normalized_errors.jsonl","w").close()

# credenziali DB
DB_CFG = {
    "host":   os.getenv("DB_HOST")   or "mysql.abcweb.local",
    "user":   os.getenv("DB_USER")   or "filippo_matte",
    "passwd": os.getenv("DB_PASS")   or "fmj893qhr193hf9fa4895rhj193r134",
    "db":     os.getenv("DB_NAME")   or "abc",
    "charset": "utf8mb4",
    "use_unicode": True,
    "connect_timeout": 10,
    "read_timeout": 20,
    "write_timeout": 20,
}

# chavi che LLM deve analizzare
LLM_KEYS = [
    "adults_number",
    "children_number",
    "accommodation_type",   # "housing_unit" | "pitch" | null
    "country_code",         # es. "IT", "DE" ecc.
    "pet",                  # true|false
    "request_target",       # "family"|"couple"|"single"|"group"|null
    "treatment_code",       # "full_board"|"half_board"|"bed_and_breakfast"|"all_inclusive"|"room_only"|null
    "children_age"          # array[int] 0..17
]

# mappatura chiavi note -> utm standard
_CAMPAIGN_KEY_MAP = {
    "utm_campaign": ["utm_campaign","campaign","cmp","campagna"],
    "utm_source": ["utm_source","source","src"],
    "utm_source_platform": ["utm_source_platform","source_platform","src_platform"],
    "utm_medium": ["utm_medium","medium","med"],
    "utm_content": ["utm_content","content","cnt"],
    "utm_term": ["utm_term","term","trm","keyword"],
    "utm_creative_format": ["utm_creative_format","creative_format","cr_fmt"],
    "utm_marketing_tactic": ["utm_marketing_tactic","marketing_tactic","mkt_tactic"],
    "utm_id": ["utm_id","campaign_id","cmp_id","id"],
    # anche acquisitionChannel che ci serve per attribution_data
    "acquisitionChannel": ["acquisitionChannel","acquisition_channel","channel"]
}


# setto GPU per il modello e i componenti
def move_inputs_to_inference_device(mdl, inputs: dict):
    # HuggingFace 4.43+ espone _get_inference_device(); fallback sicuro al device del primo parametro
    try:
        dev = mdl._get_inference_device()
    except Exception:
        dev = next(mdl.parameters()).device
    return {k: v.to(dev) for k, v in inputs.items()}

# ricerca di occorrenze iniziali
def first_present(d: dict, aliases: list[str]) -> Optional[str]:
    for k in aliases:
        if k in d and isinstance(d[k], (str, int, float)):
            return str(d[k])
    return None

# helper per identificare le colonne esistenti nel DB ed evitare di indicizzare NULL
def log_table_columns():
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute(f"SHOW COLUMNS FROM `{TABLE_NAME}`")
            cols = [r[0] for r in c.fetchall()]
            print(f"[INFO] Colonne presenti in {TABLE_NAME}: {cols}")
    finally:
        conn.close()

# funzione per normalizzare il dizionario campaign_data annidato dentro quello principale
def normalize_campaign_dict(campaign_raw: dict) -> dict:
    out = {}
    if not isinstance(campaign_raw, dict):
        return {}
    # estrai standard
    for std_key, aliases in _CAMPAIGN_KEY_MAP.items():
        val = first_present(campaign_raw, aliases)
        if val is not None:
            out[std_key] = val

    # PER DATI ETEROGENEI
    # 1.
    # conserva eventuali altre chiavi come extra opzionali che possono essere mappate tra i clienti
    for k, v in campaign_raw.items():
        if k not in {a for aliases in _CAMPAIGN_KEY_MAP.values() for a in aliases}:
            # evita collisioni con le standard già mappate
            if k not in out:
                out[k] = v

    # 2.
    # gli utm_* devono esistere almeno come stringa vuota in tutti i clienti per normalizzare e rendere omogenei
    # se un cliente non possiede i campi, glieli si inseriscono come vuoti per rendere lo schema unificato tra tutti i clienti
    for k in [
        "utm_campaign","utm_source","utm_source_platform","utm_medium",
        "utm_content","utm_term","utm_creative_format","utm_marketing_tactic","utm_id"
    ]:
        out.setdefault(k, "")

    return out

# helper parse JSON robusto 
def parse_json(v):
    if v is None: # caso base guardrail
        return None
    if isinstance(v, (bytes, bytearray)): # se è binario lo decodifichiamo in UTF-8
        v = v.decode("utf-8", errors="ignore")
    if isinstance(v, str): # se è una stringa, si fa trim prima e dopo
        v = v.strip()
        if not v:
            return None
        try:
            return json.loads(v)
        except Exception:
            return None
    if isinstance(v, (dict, list)): # se è un dizionario oppure una lista, lo restituisco immediatamente così com'è
        return v

    return None # in tutti gli altri casi, lo restituisco vuoto perché non c'è un tipo di dato coerente

# 2. Connessione al DB con le varie credenziali
def get_conn():
    cfg = dict(DB_CFG)
    if MYSQL_FLAVOR == "pymysql":
        cfg["password"] = cfg.pop("passwd", None)
        cfg["database"] = cfg.pop("db", None)
    return MySQLdb.connect(**cfg)

# 3. Funzione per popolare il dataset con i dizionari di interesse presi dal DB
# Andiamo a mettere dentro campaign_data il dizionario preso da info_campagna_json
# in questo modo:
''' 'campaign_data': {'acquisitionChannel': 'social',
                    'utm_T_ida': '8964',
                    'utm_campaign': 'SP - FB Ads',
                    'utm_content': 'Generazione di contatti [prosp]',
                    'utm_medium': 'paidsocial',
                    'utm_source': 'facebook',
                    'utm_term': 'SP - Generazione Contatti [it-it] 8964  [v1]'}'''
def iter_records(limit: int | None = None):
    """
    Generator che restituisce (pk, payload) dal DB in modo streaming e memory-safe.
    Robustezza:
      - fail-fast su connessione fallita (log + raise)
      - cleanup sicuro di conn/cursor
      - logging strutturato su JSON malformati
      - rispetto di limit sia in LOCAL che in DEPLOY
    """

    conn = None
    cur = None
    fetched = 0
    try:
        # Connessione (fail-fast con logging)
        try:
            conn = get_conn()
        except Exception:
            logging.exception("Connessione DB fallita (env=%s, table=%s)", APP_ENV, TABLE_NAME)
            raise

        if APP_ENV == "LOCAL":
            cur = conn.cursor(DictCursor)
            sql = (
                f"SELECT {PK_COL},{ID_ATTIVITA_COL},{JSON_COL},{CAMPAIGN_COL},{SCRIPT_NAME_COL} "
                f"FROM {TABLE_NAME} "
                f"WHERE {JSON_COL} IS NOT NULL "
                f"ORDER BY {DATE_COL} ASC "
                f"{'LIMIT %s' if APP_ENV == 'LOCAL' else ''}"
            )
            cur.execute(sql, (limit or LOCAL_LIMIT,))

            for row in cur:
                pk = row.get(PK_COL)
                payload = parse_json(row.get(JSON_COL))
                if not isinstance(payload, dict):
                    logging.warning("[WARN] JSON malformato pk=%s", pk)
                    continue

                campaign_raw = parse_json(row.get(CAMPAIGN_COL))
                payload["campaign_data"] = campaign_raw if campaign_raw else {}

                payload["id_attivita"] = row.get(ID_ATTIVITA_COL)
                payload["id_record"]   = pk
                payload["source_reference"] = payload.get("source_reference")
                payload["source_name"] = row.get(SCRIPT_NAME_COL) or payload.get("script_name") or "form"
                payload["table_name"]  = TABLE_NAME

                yield pk, payload
                fetched += 1
                if limit and fetched >= limit:
                    return
        else: # APP_ENV == "DEPLOY"
            conn.ping(reconnect=True)
            cur = conn.cursor(SSCursor) if USE_SSCURSOR else conn.cursor()
            sql = (
                f"SELECT {PK_COL},{JSON_COL},{CAMPAIGN_COL},{ID_ATTIVITA_COL},{SCRIPT_NAME_COL} "
                f"FROM {TABLE_NAME} "
                f"WHERE {JSON_COL} IS NOT NULL "
                f"ORDER BY {DATE_COL} ASC"
            )
            cur.execute(sql)

            while True:
                rows = cur.fetchmany(BATCH_SIZE)
                if not rows:
                    break

                for pk, raw_json, raw_campaign, id_att, script_name in rows:
                    payload = parse_json(raw_json)
                    if not isinstance(payload, dict):
                        logging.warning("[WARN] JSON malformato pk=%s", pk)
                        continue

                    campaign_raw = parse_json(raw_campaign)
                    payload["campaign_data"] = campaign_raw if campaign_raw else {}

                    payload["id_attivita"] = id_att
                    payload["id_record"]   = pk
                    payload["source_reference"] = payload.get("source_reference")
                    payload["source_name"]  = script_name or payload.get("script_name") or "form"
                    payload["table_name"]   = TABLE_NAME

                    yield pk, payload
                    fetched += 1
                    if limit and fetched >= limit:
                        return
    finally:
        # Cleanup sicuro
        try:
            if cur is not None:
                cur.close()
        except Exception:
            pass
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

# Usiamo un'unica sessione per tutti i record.
# Semplice funzione helper per serializzare subito su JSONL
def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "ab") as f:  # "ab" modalità orjson: append + binario
        f.write(orjson.dumps(obj) + b"\n")

# Schema JSON per la validazione dei dati normalizzati
# IMPORTANTE => I dati dal DB sono ETEROGENEI
# per cui è necessario tollerare tale eterogeneità senza perdere VALIDAZIONE
# aggiungendo campi opportuni da fare computare al LLM
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id_attivita": {"type": ["integer", "null"]},
        "source_name": {"type": "string"},
        "id_record": {"type": ["integer", "null"]},
        "source_reference": {"type": ["string","null"]},
        "table_name": {"type": "string"},
        "request_date": {"type": "string"},       # "YYYY-MM-DD HH:MM:SS"
        "check_in_date": {"type": "string"},      # "YYYY-MM-DD"
        "check_out_date": {"type": "string"},     # "YYYY-MM-DD"
        "adults_number": {"type": "integer"},
        "children_number": {"type": ["number", "integer"]},
        "children_age": {
            "type": "array",
            "items": {"type": "integer"}
        },
        "pet": {"type": ["null", "string", "boolean"]},
        "country_code": {"type": "string"},
        "request_lang": {"type": "string"},
        "accommodation_type": {"type": ["string", "null"]},
        "treatment": {"type": "string"},
        "treatment_code": {"type": ["null", "string"]},
        "request_target": {"type": ["null", "string"]},
        "campaign_data": {"type": "object"},
        "attribution_data": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "acquisition_channel": {
                    "type": "string",
                    "enum": [
                        "affiliates","direct","display","email","organic",
                        "other","other_advertising","paid_search","referral","social"
                    ]
                },
                "medium": {
                    "type": "string",
                    "enum": ["api","app","form","form-myreply","import-portali","manuale"]
                },
                "medium_section": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["altro","email","form","telefono","chat"]
                }
            },
            "required": ["acquisition_channel","category","medium","medium_section"]
        },
        "extras": {"type": "object"} # campo "stabilizzatore" delle info aggiuntive per clienti eterogenei
    },
    "required": [
      "id_attivita","id_record","source_name","request_lang","table_name",
      "request_target","treatment","treatment_code","accommodation_type",
      "request_date","check_in_date","check_out_date",
      "adults_number","children_number","pet","children_age",
      "country_code","campaign_data","attribution_data"
    ]
}

# Funzione per estrarre il primo oggetto JSON valido (dizionari o liste) dalla stringa di risposta usando bilanciamento parentesi
# supporta sia {} che [] per gestire i dizionari e gli array
# restituisce l'oggetto JSON decodificato o None se non trovato
def extract_first_bracketed(s: str, open_ch: str, close_ch: str) -> Optional[Tuple[int,int]]:
    start = s.find(open_ch) # trova la prima occorrenza del carattere di apertura
    if start == -1: # se non trovato, restituisce None
        return None
    # bilanciamento parentesi con gestione stringhe
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)): # scorre la stringa dal carattere di apertura
        ch = s[i] # carattere corrente
        if in_string: # se siamo dentro una stringa
            if escape: 
                escape = False # se il carattere precedente era una barra rovesciata, salta il controllo
            elif ch == '\\': 
                escape = True # se troviamo una barra rovesciata, il prossimo carattere è escape
            elif ch == '"': 
                in_string = False # se troviamo una doppia virgoletta, usciamo dalla stringa
        else: # se non siamo dentro una stringa
            if ch == '"': 
                in_string = True # se troviamo una doppia virgoletta, entriamo dunque in una stringa
            elif ch == open_ch: 
                depth += 1 # se troviamo un carattere di apertura, aumentiamo la profondità perché è un nuovo oggetto
            elif ch == close_ch: # se troviamo un carattere di chiusura
                depth -= 1 # diminuiamo la profondità perché chiudiamo un oggetto
                # se la profondità è zero, abbiamo trovato la fine dell'oggetto
                if depth == 0: # restituiamo gli indici di inizio e fine dell'oggetto trovato 
                    return (start, i+1) # +1 per includere il carattere di chiusura nella sottostringa
    return None

# Funzione per estrarre il primo oggetto JSON valido (dizionari o liste) dalla stringa di risposta usando bilanciamento parentesi
# supporta sia {} che []
# restituisce l'oggetto JSON decodificato o None se non trovato
def extract_first_json(s: str) -> Optional[Any]:
    for opener, closer in (('{','}'), ('[',']')): # prova sia con {} che con [] per gestire gli array e i dizionari
        span = extract_first_bracketed(s, opener, closer) # trova il primo oggetto bilanciato 
        if not span: # se non lo trova, passa al prossimo
            continue
        # se trovato, prova a decodificarlo
        start, end = span # indici di inizio e fine
        try: # prova a decodificare l'oggetto JSON
            return json.loads(s[start:end]) # restituisce l'oggetto JSON decodificato racchiuso tra Start ed End
        except json.JSONDecodeError: # se fallisce, continua a cercare
            continue # se non riesce a decodificare, continua col prossimo
    return None # se non trova nulla, restituisce None

# Generatore per iterare su tutti gli oggetti JSON bilanciati trovati nella stringa
# restituisce ogni oggetto JSON decodificato
def iter_balanced_json(s: str):
    i = 0
    n = len(s)
    while i < n:
        start = s.find("{", i)
        if start == -1:
            break
        span = extract_first_bracketed(s[start:], "{", "}")
        if not span:
            break
        a, b = span  # intervalli iniziali e finali relativi a 'start'
        chunk = s[start + a:start + b] # costruisco il chunk JSON
        try:
            yield json.loads(chunk)
        except json.JSONDecodeError:
            pass
        i = start + b  # continua dopo il blocco trovato

# GESTIONE UTILITIES DI NORMALIZZAZIONE OUTPUT SCHEMA
# Utilities per la validazione JSON dei dati in formato datetime
# Funzioni per validare o correggere il JSON in base allo schema atteso
def _to_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def _ensure_date_iso(s: str) -> str:
    # accetta "YYYY-MM-DD" o "DD/MM/YYYY" e normalizza a "YYYY-MM-DD"
    if not s:
        return ""
    iso = to_iso_date(s)
    return iso or ""

def _ensure_datetime_iso(s: str) -> str:
    # accetta "DD/MM/YYYY HH:MM" o "YYYY-MM-DD HH:MM:SS" e normalizza a "YYYY-MM-DD HH:MM:SS"
    if not s:
        return ""
    iso = to_iso_datetime(s)
    return iso or ""

def sanitize_enum_token(s: str) -> str:
    # lower, strip, spazi->underscore, rimozione char non alfanumerici/_
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

def _enum_or(value: str, allowed: list[str], fallback: str) -> str:
    s = sanitize_enum_token(value)
    if isinstance(s, str) and s in allowed:
        return s
    return fallback

# funzione principale per validare o correggere il dizionario di input
# come funziona?
# - Riempie i campi mancanti con default sensati
# - Applica fix (date, adulti>=1 se bimbi>0, clamp enum)
# - Valida contro SCHEMA (lancia se non valido)
# restituisce il dizionario corretto
# lancia ValidationError se non valido
# usa jsonschema per la validazione
# esempio di uso:
# try:
#   valid_data = validate_or_fix(input_data)
# except ValidationError as e:
#   print("Dati non validi:", e)
# valid_data ora contiene i dati normalizzati e validati
def validate_or_fix(d: dict) -> dict:
    """
    - Riempie i campi mancanti con default sensati
    - Applica fix (date, adulti>=1 se bimbi>0, clamp enum)
    - Valida contro SCHEMA (lancia se non valido)
    """
    if not isinstance(d, dict):
        d = {}

    # baseline con default
    out = {
        "id_attivita": d.get("id_attivita", None),
        "source_name": d.get("source_name", ""),
        "id_record": d.get("id_record", None),
        "table_name": d.get("table_name", "requests"),
        "source_reference": d.get("source_reference", ""),
        "request_date": d.get("request_date", ""),
        "check_in_date": d.get("check_in_date", ""),
        "check_out_date": d.get("check_out_date", ""),
        "adults_number": _to_int(d.get("adults_number", 0), 0),
        "children_number": _to_int(d.get("children_number", 0), 0),
        "children_age": d.get("children_age", []),
        "pet": d.get("pet", False),
        "country_code": d.get("country_code", "IT"),
        "request_lang": d.get("request_lang", "it"),
        "accommodation_type": d.get("accommodation_type", None),
        "treatment": d.get("treatment", ""),
        "treatment_code": d.get("treatment_code", None),
        "request_target": d.get("request_target", None),
        "campaign_data": d.get("campaign_data", {}),
        "attribution_data": d.get("attribution_data", {
            "acquisition_channel": "other",
            "medium": "form",
            "medium_section": "",
            "category": "form"
        })
    }

    # normalizza formati data
    out["request_date"] = _ensure_datetime_iso(out["request_date"])
    out["check_in_date"] = _ensure_date_iso(out["check_in_date"])
    out["check_out_date"] = _ensure_date_iso(out["check_out_date"])

    # se check_out < check_in → inverti
    try:
        if out["check_in_date"] and out["check_out_date"]:
            ci = datetime.datetime.strptime(out["check_in_date"], "%Y-%m-%d").date()
            co = datetime.datetime.strptime(out["check_out_date"], "%Y-%m-%d").date()
            if co < ci:
                out["check_in_date"], out["check_out_date"] = out["check_out_date"], out["check_in_date"]
    except Exception:
        pass  # se il parsing fallisce lasciamo com’è (comunque rispetta string)

    # regola adulti/bambini (almeno 1 adulto se ci sono bambini)
    if out["children_number"] > 0 and out["adults_number"] == 0:
        out["adults_number"] = 1

    # clamp enum attribution_data
    # se mancante, usa valori di default
    ad = out["attribution_data"] or {}
    ad["acquisition_channel"] = _enum_or(
        ad.get("acquisition_channel", "other"),
        ["affiliates","direct","display","email","organic","other","other_advertising","paid_search","referral","social"],
        "other"
    )
    ad["medium"] = _enum_or(
        ad.get("medium", "form"),
        ["api","app","form","form-myreply","import-portali","manuale"],
        "form"
    )
    ad["medium_section"] = ad.get("medium_section", "")
    ad["category"] = _enum_or(
        ad.get("category", "form"),
        ["altro","email","form","telefono","chat"],
        "form"
    )
    out["attribution_data"] = ad

    # tipi corretti per children_age (assicurare che sia lista di interi)
    if not isinstance(out["children_age"], list):
        out["children_age"] = []
    else:
        out["children_age"] = [int(x) for x in out["children_age"] if isinstance(x, (int, float))]

    # accomodation type → consenti solo housing_unit / pitch / null
    if out["accommodation_type"] not in (None, "housing_unit", "pitch"):
        out["accommodation_type"] = None

    # treatment_code clamp (o null)
    allowed_treat = {"all_inclusive","full_board","half_board","bed_and_breakfast","room_only"}
    if out["treatment_code"] not in allowed_treat:
        out["treatment_code"] = None

    # request_target clamp (o null)
    allowed_target = {"single","couple","group","family"}
    if out["request_target"] not in allowed_target:
        out["request_target"] = None

    # campi stringa not-null
    # se mancanti, li imposta a stringa vuota
    for k in [
        "source_name","table_name","country_code","request_lang","treatment",
        "request_date","check_in_date","check_out_date","source_reference"
    ]:
        if out.get(k) is None:
            out[k] = ""

    # validazione jsonschema (alza eccezione se non valido)
    validate(instance=out, schema=SCHEMA)

    return out

# GESTIONE UTILITIES DI NORMALIZZAZIONE INPUT
# 0. Masking di campi sensibili (es. email, telefono)
# a. Funzione per mascherare nome
def mask_name(s: Optional[str]) -> Optional[str]:
    if not s: 
        return s  # se la stringa è vuota o None, restituisce None
    s = s.strip() # rimuove spazi bianchi iniziali e finali
    if len(s) <= 2:
        return s[0] + "*" * (len(s)-1)  # maschera tutto tranne la prima lettera
    return s[0] + "*" * (max(0, len(s)-1)) + s[-1]  # maschera tutto tranne la prima e l'ultima lettera

# b. Funzione per mascherare email
def mask_email(email: Optional[str]) -> Optional[str]:
    if not email: 
        return email  # se la stringa è vuota o None, restituisce None
    parts = email.split("@") # divide l'email in parte locale e dominio
    if len(parts) != 2:
        return email  # se non è un'email valida, restituisce l'input originale
    local, domain = parts
    if len(local) <= 2:
        masked_local = local[0] + "*" * (len(local)-1)  # maschera tutto tranne la prima lettera
    else:
        masked_local = local[0] + "*" * (max(0, len(local)-2)) + local[-1]  # maschera tutto tranne la prima e l'ultima lettera
    return masked_local + "@" + domain  # ricostruisce l'email mascherata

# c. Funzione per mascherare numero di telefono
def mask_phone(phone: Optional[str]) -> Optional[str]:
    if not phone: 
        return phone  # se la stringa è vuota o None, restituisce None
    digits = [c for c in phone if c.isdigit()]
    if len(digits) <= 2:
        return "***"
    # mostra solo le ultime 2 cifre
    masked = "*"*(len(digits)-2) + "".join(digits[-2:])
    # mantieni il formato semplice
    return masked

# d. Funzione generale per mascherare campi sensibili elencati in un dizionario
def mask_sensitive_fields(record: Dict) -> Dict:
    out = dict(record)  # copia il dizionario originale
    if "nome" in out:
        out["nome"] = mask_name(out["nome"])
    if "cognome" in out:
        out["cognome"] = mask_name(out["cognome"])
    if "email" in out:
        out["email"] = mask_email(out["email"])
    if "telefono" in out:
        out["telefono"] = mask_phone(out["telefono"])
    return out

# 1. Utilities per la validazione JSON dei dati in formato datetime
# a. Converte una data che è in formato "DD/MM/YYYY HH:MM" o "DD/MM/YYYY HH:MM:SS" in ISO "YYYY-MM-DD HH:MM:SS"
def to_iso_datetime(dt_str: str) -> Optional[str]: 
    # es: "01/10/2025 00:00" -> "2025-10-01 00:00:00"
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(dt_str, fmt) # parse della data 
            return dt.strftime("%Y-%m-%d %H:%M:%S") # formato ISO con secondi
        except ValueError:
            pass
    return None

# b. Converte una data che è in formato "DD/MM/YYYY" o "YYYY-MM-DD" in ISO "YYYY-MM-DD" senza orario
def to_iso_date(d_str: str) -> Optional[str]:
    # es: "13/06/2026" -> "2026-06-13"
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            d = datetime.datetime.strptime(d_str, fmt) # parse della data
            return d.strftime("%Y-%m-%d") # formato ISO senza orario
        except ValueError:
            pass
    return None

# 2. Funzione per mappare e validare un singolo record di input
def map_accommodation_type(raw: str) -> Optional[str]: # mappa il tipo di alloggio in categorie standard
    if not raw: # se la stringa è vuota o None, restituisce None
        return None
    s = raw.lower() # converte la stringa in minuscolo per confronti case-insensitive
    # controlla parole chiave per determinare il tipo di alloggio
    if any(k in s for k in ["casa mobile", "camera", "appart", "suite", "bungalow", "loggia"]):
        return "housing_unit" # unità abitativa
    # controlla parole chiave per tenda
    if any(k in s for k in ["piazzola", "tenda", "roulotte", "camper", "pitch"]):
        return "pitch" # piazzola
    return None

# 3. Funzione per mappare il codice di trattamento in categorie standard
def map_treatment_code(raw_treatment: str, notes: str = "") -> Optional[str]:
    S = (f"{raw_treatment or ''} {notes or ''}").lower() # unisce trattamento e note, converte in minuscolo
    # controlla parole chiave per determinare il codice di trattamento
    if "solo pernott" in S or "room only" in S:
        return "room_only"
    if "b&b" in S or "bed and breakfast" in S:
        return "bed_and_breakfast"
    if "mezza pensione" in S or "half board" in S:
        return "half_board"
    if "pensione completa" in S or "full board" in S:
        return "full_board"
    if "all inclusive" in S:
        return "all_inclusive"
    return None

# 4. Funzione per mappare il paese di provenienza in codice ISO2
def detect_country_code(lingua: Optional[str], email: Optional[str], notes: Optional[str]) -> Optional[str]:
    # euristica molto semplice (migliorabile con libreria)
    if lingua:
        m = lingua.lower().split("-")[0] # prende la parte prima del trattino se presente 
        mapping = {"it":"IT","en":"US","de":"DE","fr":"FR","nl":"NL","es":"ES","pt":"PT"}
        if m in mapping: 
            return mapping[m] # restituisce il codice paese se trovato nella mappatura
    if email and email.lower().endswith(".nl"): # controlla il dominio email 
        return "NL"
    return None

# 5. Funzione per convertire valori di tipo sì/no in booleani -> FONDAMENTALE PER NORMALIZZARE INPUT VARIABILI
def bool_from(v: Any) -> Optional[bool]:
    if v is None: 
        return None
    s = str(v).strip().lower() # converte in stringa, rimuove spazi e converte in minuscolo
    if s in ["1","si","sì","yes","true","y"]: 
        return True
    if s in ["0","no","false","n"]: 
        return False
    return None

# 6. Funzione di pre-normalizzazione per normalizzare l'input prima di usarlo nel RAG
def pre_normalization(inp: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(inp) # copia l'input originale in un nuovo dizionario di output 
    # date in formato ISO (utile per filtri successivi) 
    out["request_date_iso"] = to_iso_datetime(inp.get("data_richiesta","")) or ""
    out["check_in"]  = to_iso_date(inp.get("giorno_di_arrivo","")) or ""
    out["check_out"] = to_iso_date(inp.get("giorno_di_partenza","")) or ""

    # adulti/bambini 
    adults = int(inp.get("sys_adulti") or 0) # converte in intero, default 0 se mancante
    children = int(inp.get("sys_bambini") or 0) # converte in intero, default 0 se mancante
    children_age: List[int] = [] # lista delle età dei bambini

    # gestione animali domestici e richieste particolari
    pet = bool_from(inp.get("cfield-viaggi-con-un-cane")) or bool_from(inp.get("richieste_particolari"))

    # lingua / country code
    request_lang = (inp.get("lingua") or inp.get("lang") or "").lower() or None
    country_code = detect_country_code(request_lang, inp.get("email"), inp.get("richieste_particolari"))

    # accommodation type and treatment
    accommodation_type = map_accommodation_type(inp.get("sistemazione_1","") or inp.get("tipologia",""))
    treatment = inp.get("trattamento") or ""
    treatment_code = map_treatment_code(treatment, inp.get("richieste_particolari",""))

    # campaign_data già normalizzato in records[]
    campaign_data = inp.get("campaign_data", {}) if isinstance(inp.get("campaign_data"), dict) else {}

    # attribution_data
    ms = (inp.get("medium_section") or "").lower()
    # priorità a campaign_data.acquisitionChannel se presente
    acq_from_campaign = (campaign_data.get("acquisitionChannel") or "").lower()
    if acq_from_campaign in {"affiliates","direct","display","email","organic","other","other_advertising","paid_search","referral","social"}:
        acquisition_channel = acq_from_campaign
    else:
        if "paid-search" in ms: 
            acquisition_channel = "paid_search"
        elif "social" in ms: 
            acquisition_channel = "social"
        elif "email" in ms: 
            acquisition_channel = "email"
        elif "organic" in ms: 
            acquisition_channel = "organic"
        else: 
            acquisition_channel = "other"

    # OUTPUT NORMALIZZATO
    # aggiorna il dizionario di output con i nuovi campi normalizzati e mappati
    out["pre"] = {
        "request_date": out["request_date_iso"],
        "check_in_date": out["check_in"],
        "check_out_date": out["check_out"],
        "adults_number": adults,
        "children_number": children,
        "children_age": children_age,
        "pet": pet if pet is not None else False,
        "country_code": country_code or "IT",
        "request_lang": request_lang or "it",
        "accommodation_type": accommodation_type,
        "treatment": treatment,
        "treatment_code": treatment_code,
        "request_target": None,
        "campaign_data": campaign_data,
        "attribution_data": {
            "acquisition_channel": acquisition_channel,
            "medium": inp.get("medium") or "",
            "medium_section": inp.get("medium_section") or "",
            "category": inp.get("category") or "form"
        },
        "id_attivita": inp.get("id_attivita"),
        "id_record":   inp.get("id_record"),
        "source_name": inp.get("source_name") or inp.get("script_name") or "form", # da iter_records
        "table_name":  inp.get("table_name") or "requests",
        "source_reference": (inp.get("source_reference") 
                            or (inp.get("content_json", {}) or {}).get("source_reference"))

    }
    return out

# funzione per verificare se un oggetto somiglia al formato finale atteso
# controlla la presenza delle chiavi obbligatorie e riduce FP sui prompt LLM
'''def looks_like_final(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    # chiavi strutturali ma "leggere"
    must = {"id_attivita", "table_name"}
    return must.issubset(obj.keys())'''

def looks_like_final(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    must_any = {"adults_number","children_number","accommodation_type","country_code","pet","request_target","treatment","treatment_code","children_age"}
    # considera "finale" se ha almeno 3 chiavi del set (il resto lo aggiusta validate_or_fix/merge)
    return len(must_any.intersection(obj.keys())) >= 3

# funzione per pulire e normalizzare il JSON che inizia con chiave forte, se non lo trova fa fallback al primo JSON bilanciato
def extract_json_keys(s: str, key_markers=(
    "\"adults_number\"", "\"children_number\"", "\"accommodation_type\"", "\"country_code\"",
    "\"pet\"", "\"request_target\"", "\"treatment_code\"", "\"children_age\""
)):
    s = s.replace("```json", "").replace("```", "").strip()
    for key in key_markers:
        key_pos = s.find(key)
        if key_pos != -1:
            brace_pos = s.rfind("{", 0, key_pos)
            if brace_pos != -1:
                for obj in iter_balanced_json(s[brace_pos:]):
                    if looks_like_final(obj):
                        return obj
                for obj in iter_balanced_json(s[brace_pos:]):
                    return obj
    for obj in iter_balanced_json(s):
        if looks_like_final(obj):
            return obj
    for obj in iter_balanced_json(s):
        return obj
    return None

# funzione per "clampare" i campi LLM a valori validi
# esempio: adulti_number >=0, children_age in [0-17], accommodation_type in {housing_unit,pitch}
# restituisce un dizionario con i campi corretti
# output: adulti_number (int), children_number (int), children_age (list of int),
# accommodation_type (str or None), country_code (str or None),
# pet (bool), request_target (str or None), treatment_code (str or None)
def clamp_llm_fields(d: dict) -> dict:
    out = {}

    # adults_number / children_number
    def to_nonneg_int(x):
        try:
            v = int(x)
            return max(0, v)
        except Exception:
            return 0

    out["adults_number"]   = to_nonneg_int(d.get("adults_number"))
    out["children_number"] = to_nonneg_int(d.get("children_number"))

    # children_age
    ages = d.get("children_age", [])
    if not isinstance(ages, list):
        ages = []
    clean_ages = []
    for a in ages:
        try:
            ai = int(a)
            if 0 <= ai <= 17:
                clean_ages.append(ai)
        except Exception:
            pass
    out["children_age"] = clean_ages

    # accommodation_type
    acc = d.get("accommodation_type")
    if acc not in (None, "housing_unit", "pitch"):
        acc = None
    out["accommodation_type"] = acc

    # country_code
    cc = d.get("country_code")
    if isinstance(cc, str):
        cc = cc.strip().upper()
        if not re.fullmatch(r"[A-Z]{2}", cc):
            cc = None
    else:
        cc = None
    out["country_code"] = cc

    # pet
    def to_bool(x):
        s = str(x).strip().lower()
        if s in ("true","1","si","sì","y","yes"):  return True
        if s in ("false","0","no","n"):           return False
        return False
    out["pet"] = to_bool(d.get("pet"))

    # request_target
    tgt = d.get("request_target")
    allowed_tgt = {"single","couple","group","family"}
    if tgt not in allowed_tgt:
        tgt = None
    out["request_target"] = tgt

    # treatment_code
    trt = d.get("treatment_code")
    allowed_trt = {"all_inclusive","full_board","half_board","bed_and_breakfast","room_only"}
    if trt not in allowed_trt:
        trt = None
    out["treatment_code"] = trt

    return out

# costruttore output ibrido finale deterministico + LLM
def build_final_candidate(pre_input: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "id_attivita": pre_input.get("id_attivita"),
        "source_name": pre_input.get("source_name","") or "",
        "id_record": pre_input.get("id_record"),
        "table_name": pre_input.get("table_name","requests") or "requests",
        "source_reference": pre_input.get("source_reference"),
        "request_date": pre_input.get("request_date",""),
        "check_in_date": pre_input.get("check_in_date",""),
        "check_out_date": pre_input.get("check_out_date",""),
        "adults_number": pre_input.get("adults_number", 0),
        "children_number": pre_input.get("children_number", 0),
        "children_age": pre_input.get("children_age", []),
        "pet": pre_input.get("pet", False),
        "country_code": pre_input.get("country_code","IT"),
        "request_lang": pre_input.get("request_lang","it"),
        "accommodation_type": pre_input.get("accommodation_type"),
        "treatment": pre_input.get("treatment",""),
        "treatment_code": pre_input.get("treatment_code"),
        "request_target": pre_input.get("request_target"),
        "campaign_data": pre_input.get("campaign_data", {}),
        "attribution_data": pre_input.get("attribution_data", {
            "acquisition_channel": "other",
            "medium": "form",
            "medium_section": "",
            "category": "form",
        }),
    }
    return base

# Prendo i dati dal DB o da file CSV/JSONL
# Query dal DB o caricamento da file
records = []
for i, (pk, payload) in enumerate(iter_records(limit=LOCAL_LIMIT)):
    # payload["campaign_data"] è già stato popolato in iter_records (se presente)
    # normalizzazione del dizionario di campaign_data
    if "campaign_data" in payload and isinstance(payload["campaign_data"], dict):
        payload["campaign_data"] = normalize_campaign_dict(payload["campaign_data"])
    else:
        payload["campaign_data"] = normalize_campaign_dict({})
    records.append(payload)
if not records:
    raise RuntimeError("Nessun record caricato")
print(f"\nCaricati {len(records)} record.\n")
# masking PII dei campi sensibili
masked = [mask_sensitive_fields(r) for r in records]
# dati normalizzati per il RAG
prepped = [pre_normalization(r) for r in masked]
print("DATI NORMALIZZATI:\n")
pprint(prepped, width=100)

# dati di test per un solo record
raw_input  = records[0]
pre_input  = prepped[0]["pre"]  # indice e chiave del primo record
print("\n#################################################################################")
print(f"Used device: {DEVICE}")
print("\nStarting AI Transformers operations...")

# 7. Preparazione del prompt per il modello LLM di normalizzazione/mappatura
# ISTRUZIONI operative per il testo PHP da convertire in JSON 
istruzioni = """
Genera UN SOLO oggetto JSON minificato che contenga ESCLUSIVAMENTE le seguenti chiavi:
- adults_number (int >= 0)
- children_number (int >= 0)
- accommodation_type: "housing_unit" | "pitch" | null
- country_code: codice ISO2 (2 lettere maiuscole) o null se non deducibile
- pet: true | false
- request_target: "family" | "couple" | "single" | "group" | null
- treatment: "valorizza questo campo con il nome reale del trattamento richiesto. Se non è possibile dedurlo dai dati di input, usa una stringa vuota."
- treatment_code: "all_inclusive" | "full_board" | "half_board" | "bed_and_breakfast" | "room_only" | null
- children_age: array di interi 0..17 (se assenti → [])

Regole:
- Non aggiungere altre chiavi.
- Se un valore è sconosciuto, PRIMA DI DARE LA RISPOSTA cercalo in tutto il JSON di input -> SOLO se non lo trovi fallback su null (o [] per children_age).
- country_code deve essere una stringa di 2 lettere maiuscole se presente, altrimenti null.
- Nessun testo prima o dopo il JSON. Nessun backtick.
- Output MINIFICATO su UNA riga.
"""

schema_atteso = """
{
  "request_date": "YYYY-MM-DD HH:MM:SS",
  "adults_number": int,
  "children_number": number,
  "children_age": [int,...],
  "pet": null|string|boolean,
  "country_code": "CC",
  "accommodation_type": "housing_unit"|"pitch"|null,
  "treatment": "string",
  "treatment_code": null|"all_inclusive"|"full_board"|"half_board"|"bed_and_breakfast"|"room_only",
  "request_target": null|"single"|"couple"|"group"|"family",
}
"""

# Gestione della parte del LLM
USE_LLM = True  # False: per pipeline tutta deterministica (debug / fallback)

def make_user_message(llm_input: dict, istruzioni: str, schema_atteso: str) -> str:
    raw_form   = json.dumps(llm_input["raw_form"], ensure_ascii=False)
    hints_pre  = json.dumps(llm_input["hints_pre"], ensure_ascii=False)
    return (
        "Compito: estrai SOLO gli 8 campi richiesti.\n"
        f"{istruzioni}\n"
        "Esempio di struttura (solo per forma, i valori vanno dedotti):\n"
        f"{schema_atteso}\n"
        "Dati di partenza (puoi usarli per dedurre i campi):\n"
        f"RAW_FORM={raw_form}\n"
        f"HINTS_PRE={hints_pre}\n"
        "Inizia subito con '{' e chiudi il JSON su una sola riga.\n"
    )

# generatore LLM tokenizer
gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
# criterio di stop personalizzato
# configurazione bitsandbytes 4bit per ridurre l'uso di VRAM
bnb = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=thc.bfloat16, # oppure float16 se è supportato
    bnb_4bit_use_double_quant=True
)

# config generator del transformer
gen_cfg = GenerationConfig(
    do_sample=False,
    max_new_tokens=TOKEN, # 260
    pad_token_id=gen_tok.eos_token_id,
    eos_token_id=gen_tok.eos_token_id
)

mdl = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL_NAME,  # su config.py -> cambiare modello a Llama o Qwen/Qwen2.5-7B-Instruct quando si userà GPU
    dtype=DTYPE,
    device_map=device_map_for_transformers(),
    low_cpu_mem_usage=True,
    quantization_config=bnb,
    trust_remote_code=False,
    attn_implementation=ATTN_IMPL
)


bucket_texts = []
bucket_meta = []   # terrà (idx, raw_input, pre_input, final_candidate)
# pad_token per tokenizer
if gen_tok.pad_token_id is None:
    gen_tok.pad_token = gen_tok.eos_token

# LOOP principale
for idx, (raw_input, pre_input) in enumerate(zip(records, (p["pre"] for p in prepped)), start=1):

    # 4a) costruttore deterministico 
    final_candidate = build_final_candidate(pre_input, raw_input)

    if not USE_LLM:
        # validazione+persistenza senza LLM
        try:
            final = validate_or_fix(final_candidate)
            write_jsonl(OUTPUT_JSONL, final)
            ok_count += 1
        except ValidationError as ve:
            write_jsonl(ERRORS_JSONL, {
                "idx": idx, 
                "error": f"ValidationError: {str(ve)}",
                "raw_input": raw_input, 
                "pre_input": pre_input})
            err_count += 1
        continue

    # (2) Input unico ibrido per LLM: raw + hints_pre
    llm_input = {
        "hints_pre": pre_input,        # indizi deterministici (il modello può copiarli perché sono sempre quelli)
        "raw_form":  raw_input         # raw originale
    }

    # prompt messages
    system_msg = "Sei un valido assistente per analizzare JSON di richieste informazioni turistiche e di generare un JSON formattato"
    user_msg = make_user_message(llm_input, istruzioni, schema_atteso)
    text = f"{system_msg}\n\n{user_msg}\n"
    
    # Accumulo per batch
    bucket_texts.append(text)
    bucket_meta.append((idx, raw_input, pre_input, final_candidate))
    # Se raggiungo il batch o sono all'ultima riga → eseguo la generazione in blocco
    if len(bucket_texts) == BATCH or idx == len(records):
        try:
            inputs = gen_tok(
                bucket_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048          # o 1536 con poca VRAM
            )
            # sposto tutti i componenti del LLM su CUDA
            inputs = move_inputs_to_inference_device(mdl, inputs)
            
            # generazione dei token con LLM
            with thc.inference_mode():
                out = mdl.generate(
                    **inputs,
                    generation_config=gen_cfg,
                    max_new_tokens=TOKEN,
                    repetition_penalty=1.03, 
                    use_cache=True
                )

            input_ids      = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # lunghezza effettiva dell'input per ogni elemento del batch
            input_lengths = attention_mask.sum(dim=1)  # shape: [batch], valori int

            # out: Tensor [batch, total_len]; estrai SOLO i nuovi token generati
            gen_only = [out[i, input_lengths[i].item():] for i in range(out.size(0))]
            decoded = gen_tok.batch_decode(gen_only, skip_special_tokens=True)
        
            # pulizia di sicurezza (nel caso il modello abbia messo backticks o preamboli)
            # Post-processing riga per riga nel batch
            for (idx_i, raw_i, pre_i, final_cand_i), raw_out in zip(bucket_meta, decoded):
                try:
                    draft = extract_json_keys(raw_out)
                    if draft is None:
                        write_jsonl(
                            ERRORS_JSONL,
                            {"idx": idx_i, "error": "NoJSONFound", "model_raw": raw_out[:2000]}
                        )
                        err_count += 1
                        continue
                    # (3) Merge: LLM tocca solo i campi ambigui
                    llm_patch = clamp_llm_fields(draft)
                    for k in llm_patch:
                        final_cand_i[k] = llm_patch[k]

                    # (4) Validazione e persistenza
                    final = validate_or_fix(final_cand_i)
                    write_jsonl(OUTPUT_JSONL, final)
                    ok_count += 1
                        
                except ValidationError as ve:
                    write_jsonl(
                        ERRORS_JSONL,
                        {
                            "idx": idx_i,
                            "error": f"ValidationError: {str(ve)}",
                            "model_raw": raw_out[:2000],
                            "raw_input": raw_i,
                            "pre_input": pre_i
                        }
                    )
                    err_count += 1
                except Exception as e:
                    write_jsonl(
                        ERRORS_JSONL,
                        {
                            "idx": idx_i,
                            "error": repr(e),
                            "model_raw": raw_out[:2000],
                            "raw_input": raw_i,
                            "pre_input": pre_i
                        }
                    )
                    err_count += 1

        finally:
            # svuoto i bucket indipendentemente da errori per non duplicare
            bucket_texts.clear()
            bucket_meta.clear()

# Report finale
# verifica a posteriori che non ci siano record duplicati in output
with open("normalized_requests.jsonl","r", encoding="utf-8") as f:
    ids = [json.loads(line)["id_record"] for line in f if line.strip()]
dup = [k for k, v in Counter(ids).items() if v > 1]
if dup:
    print(f"Attenzione: record duplicati trovati in output: {dup}\n")
else:
    print("Nessun record duplicato trovato in output.\n")
# resto dei logs
print(f"\nDone. OK: {ok_count} | ERR: {err_count} | Tot: {len(records)}")
print(f"Output log: {OUTPUT_JSONL}")
if err_count:
    print(f"Error log: {ERRORS_JSONL}")
