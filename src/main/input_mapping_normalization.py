# Con questo file andiamo ad effettuare tutte le mappature necessarie sui dati di input PRIMA di installare il RAG
# Ad esempio, possiamo convertire i dati in un formato JSONL standardizzato, rimuovere campi inutili, rinominare chiavi, ecc.
# Questo aiuta a mantenere il codice di caricamento dati pulito e modulare.
# Possiamo anche aggiungere funzioni di validazione per assicurarci che i dati siano nel formato corretto prima di procedere con l'addestramento o l'inferenza.
import json, os, datetime, orjson
from typing import List, Dict, Any, Optional, Tuple
import torch as thc
import numpy as np
from pprint import pprint
try:
    import MySQLdb  # mysqlclient (Linux/macOS)
    from MySQLdb.cursors import DictCursor, SSCursor
    MYSQL_FLAVOR = "mysqldb"
except ModuleNotFoundError:
    import pymysql as MySQLdb # PyMySQL (Windows)
    from pymysql.cursors import DictCursor, SSCursor
    MYSQL_FLAVOR = "pymysql"
from transformers import AutoTokenizer, AutoModelForCausalLM
from jsonschema import validate, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

# GESTIONE DEI DATI DAL DB
# 1. Config
APP_ENV = os.getenv("APP_ENV", "LOCAL").upper()  # "LOCAL" | "DEPLOY" per alternare tra test locale e produzione reale
LOCAL_LIMIT = int(os.getenv("LOCAL_LIMIT", "10"))  # quanti record in LOCAL
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "1000")) # quanti record per fetchmany() in DEPLOY
USE_SSCURSOR = True  # Server-Side cursor per stream su milioni di righe
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

# credenziali DB
DB_CFG = {
    "host":   os.getenv("DB_HOST")
    "user":   os.getenv("DB_USER")
    "passwd": os.getenv("DB_PASS")
    "db":     os.getenv("DB_NAME")
    "charset": "utf8mb4",
    "use_unicode": True,
    "connect_timeout": 10,
    "read_timeout": 20,
    "write_timeout": 20,
}

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

# ricerca di occorrenze
def first_present(d: dict, aliases: list[str]) -> Optional[str]:
    for k in aliases:
        if k in d and isinstance(d[k], (str,int,float)):
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

    # conserva eventuali altre chiavi come extra opzionali che possono essere mappate tra i clienti
    for k, v in campaign_raw.items():
        if k not in {a for aliases in _CAMPAIGN_KEY_MAP.values() for a in aliases}:
            # evita collisioni con le standard già mappate
            if k not in out:
                out[k] = v

    # gli utm_* devono esistere almeno come stringa vuota in tutti i clienti per normalizzare e rendere omogenei
    for k in [
        "utm_campaign","utm_source","utm_source_platform","utm_medium",
        "utm_content","utm_term","utm_creative_format","utm_marketing_tactic","utm_id"
    ]:
        out.setdefault(k, "")

    return out

# helper parse JSON robusto 
def parse_json(v):
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="ignore")
    if isinstance(v, str):
        v = v.strip()
        if not v:
            return None
        try:
            return json.loads(v)
        except Exception:
            return None
    if isinstance(v, dict):
        return v
    return None

# 2. Connessione al DB
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
    conn = get_conn()
    try:
        if APP_ENV == "LOCAL":
            cur = conn.cursor(DictCursor)
            sql = f"""
                SELECT
                    {PK_COL},
                    {ID_ATTIVITA_COL},
                    {JSON_COL},
                    {CAMPAIGN_COL},
                    {SCRIPT_NAME_COL}
                FROM {TABLE_NAME}
                WHERE {JSON_COL} IS NOT NULL
                ORDER BY {DATE_COL} ASC
                { 'LIMIT %s' if APP_ENV == 'LOCAL' else '' }
            """
            cur.execute(sql, (limit or LOCAL_LIMIT,))
            for row in cur:
                pk = row[PK_COL]
                payload = parse_json(row[JSON_COL])
                if not isinstance(payload, dict):
                    print(f"[WARN] JSON malformato pk={pk}")
                    continue

                campaign_raw = parse_json(row.get(CAMPAIGN_COL))
                # 1) dati di "info_campagna_json" nel campo "campaign_data" del DIZIONARIO PRINCIPALE
                payload["campaign_data"] = campaign_raw if campaign_raw else {}
                
                # 2) id_attivita / id_record / id_source (source_reference)
                payload["id_attivita"] = row.get(ID_ATTIVITA_COL)
                payload["id_record"]   = row.get(PK_COL)  # campo "id" del record corrente
                payload["source_reference"] = payload.get("source_reference") # dalla colonna "content_json"
                payload["source_name"] = row.get(SCRIPT_NAME_COL) or payload.get("script_name") or "form"
                payload["table_name"]  = TABLE_NAME # nome vero e proprio della tabella

                yield pk, payload

        elif APP_ENV == "DEPLOY":
            conn.ping(reconnect=True)
            cur = conn.cursor(SSCursor) if USE_SSCURSOR else conn.cursor()
            sql = f"""
                SELECT
                    {PK_COL},
                    {JSON_COL},
                    {CAMPAIGN_COL},
                    {ID_ATTIVITA_COL},
                    {SCRIPT_NAME_COL}
                FROM {TABLE_NAME}
                WHERE {JSON_COL} IS NOT NULL
                ORDER BY {DATE_COL} ASC
            """
            cur.execute(sql)
            fetched = 0
            while True:
                rows = cur.fetchmany(BATCH_SIZE)
                if not rows: break
                for pk, raw_json, raw_campaign, id_att, script_name in rows:
                    payload = parse_json(raw_json)
                    if not isinstance(payload, dict):
                        print(f"[WARN] JSON malformato pk={pk}")
                        continue
                    # 1) dati di "info_campagna_json" nel campo "campaign_data" del DIZIONARIO PRINCIPALE
                    campaign_raw = parse_json(raw_campaign)
                    payload["campaign_data"] = campaign_raw if campaign_raw else {}
                    # override piatti
                    payload["id_attivita"] = id_att
                    payload["id_record"]   = pk
                    payload["source_reference"] = payload.get("source_reference")
                    payload["source_name"]  = script_name or payload.get("script_name") or "form"
                    payload["table_name"]  = TABLE_NAME                
                
                    yield pk, payload
                    fetched += 1
                    if limit and fetched >= limit:
                        return
        else:
            print("Errore: APP_ENV deve essere LOCAL o DEPLOY.")
            exit(-1)
    finally:
        try: cur.close()
        except Exception: pass
        conn.close()

# Usiamo un'unica sessione per tutti i record.
# Semplice funzione helper per serializzare subito su JSONL
def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "ab") as f:  # binario + orjson
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
        "extras": {"type": "object"} # info aggiuntive per clienti eterogenei
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
            if escape: escape = False # se il carattere precedente era una barra rovesciata, salta il controllo
            elif ch == '\\': escape = True # se troviamo una barra rovesciata, il prossimo carattere è escape
            elif ch == '"': in_string = False # se troviamo una doppia virgoletta, usciamo dalla stringa
        else: # se non siamo dentro una stringa
            if ch == '"': in_string = True # se troviamo una doppia virgoletta, entriamo in una stringa
            elif ch == open_ch: depth += 1 # se troviamo un carattere di apertura, aumentiamo la profondità perché è un nuovo oggetto
            elif ch == close_ch: # se troviamo un carattere di chiusura, diminuiamo la profondità perché chiudiamo un oggetto
                depth -= 1 # se la profondità è zero, abbiamo trovato la fine dell'oggetto
                if depth == 0: # restituiamo gli indici di inizio e fine dell'oggetto trovato 
                    return (start, i+1) # +1 per includere il carattere di chiusura nella sottostringa
    return None

# Funzione per estrarre il primo oggetto JSON valido (dizionari o liste) dalla stringa di risposta usando bilanciamento parentesi
# supporta sia {} che []
# restituisce l'oggetto JSON decodificato o None se non trovato
def extract_first_json(s: str) -> Optional[Any]:
    for opener, closer in (('{','}'), ('[',']')): # prova sia con {} che con [] per gestire gli array e i dizionari
        span = extract_first_bracketed(s, opener, closer) # trova il primo oggetto bilanciato 
        if span: # se trovato, prova a decodificarlo
            start, end = span # indici di inizio e fine
            try: # prova a decodificare l'oggetto JSON
                return json.loads(s[start:end]) # restituisce l'oggetto JSON decodificato
            except json.JSONDecodeError: # se fallisce, continua a cercare
                return None # se non riesce a decodificare, restituisce None
    return None

# GESTIONE UTILITIES DI NORMALIZZAZIONE OUTPUT SCHEMA
# Utilities per la validazione JSON dei dati in formato datetime
# Funzioni per validare o correggere il JSON in base allo schema atteso
def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _ensure_date_iso(s: str) -> str:
    # accetta "YYYY-MM-DD" o "DD/MM/YYYY" e normalizza a "YYYY-MM-DD"
    if not s:
        return ""
    iso = to_iso_date(s)
    return iso or s

def _ensure_datetime_iso(s: str) -> str:
    # accetta "DD/MM/YYYY HH:MM" o "YYYY-MM-DD HH:MM:SS" e normalizza a "YYYY-MM-DD HH:MM:SS"
    if not s:
        return ""
    iso = to_iso_datetime(s)
    return iso or s

def _enum_or(value: str, allowed: list[str], fallback: str) -> str:
    if isinstance(value, str) and value in allowed:
        return value
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
    for k in ["source_name","table_name","country_code","request_lang","treatment","request_date","check_in_date","check_out_date"]:
        if out.get(k) is None:
            out[k] = ""

    # validazione jsonschema (alza eccezione se non valido)
    validate(instance=out, schema=SCHEMA)

    return out

# GESTIONE UTILITIES DI NORMALIZZAZIONE INPUT
# 0. Masking di campi sensibili (es. email, telefono)
# a. Funzione per mascherare nome
def mask_name(s: Optional[str]) -> Optional[str]:
    if not s: return s  # se la stringa è vuota o None, restituisce None
    s = s.strip() # rimuove spazi bianchi iniziali e finali
    if len(s) <= 2:
        return s[0] + "*" * (len(s)-1)  # maschera tutto tranne la prima lettera
    return s[0] + "*" * (max(0, len(s)-1)) + s[-1]  # maschera tutto tranne la prima e l'ultima lettera

# b. Funzione per mascherare email
def mask_email(email: Optional[str]) -> Optional[str]:
    if not email: return email  # se la stringa è vuota o None, restituisce None
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
    if not phone: return phone  # se la stringa è vuota o None, restituisce None
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
        if m in mapping: return mapping[m] # restituisce il codice paese se trovato nella mappatura
    if email and email.lower().endswith(".nl"): # controlla il dominio email 
        return "NL"
    return None

# 5. Funzione per convertire valori di tipo sì/no in booleani -> FONDAMENTALE PER NORMALIZZARE INPUT VARIABILI
def bool_from(v: Any) -> Optional[bool]:
    if v is None: return None
    s = str(v).strip().lower() # converte in stringa, rimuove spazi e converte in minuscolo
    if s in ["1","si","sì","yes","true","y"]: return True
    if s in ["0","no","false","n"]: return False
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
        if "paid-search" in ms: acquisition_channel = "paid_search"
        elif "social" in ms: acquisition_channel = "social"
        elif "email" in ms: acquisition_channel = "email"
        elif "organic" in ms: acquisition_channel = "organic"
        else: acquisition_channel = "other"

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
        "ids": {
            "id_attivita": inp.get("id_attivita"),
            "id_record":   inp.get("id_record"),
            "source_name": inp.get("source_name") or inp.get("script_name") or "form", # da iter_records
            "table_name":  inp.get("table_name") or "requests",
            "source_reference": (inp.get("source_reference") 
                            or (inp.get("content_json", {}) or {}).get("source_reference"))

        }
    }
    return out

# costruttore output ibrido finale deterministico + LLM
def build_final_candidate(pre_input: Dict[str, Any], raw_input: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "id_attivita": pre_input.get("ids",{}).get("id_attivita"),
        "source_name": pre_input.get("ids",{}).get("source_name","") or "",
        "id_record": pre_input.get("ids",{}).get("id_record"),
        "table_name": pre_input.get("ids",{}).get("table_name","requests") or "requests",
        "source_reference": pre_input.get("ids",{}).get("source_reference"),
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
print(f"Caricati {len(records)} record.\n")
# masking PII dei campi sensibili
masked = [mask_sensitive_fields(r) for r in records]
# dati normalizzati per il RAG
prepped = [pre_normalization(r) for r in masked]
print(f"\nDATI NORMALIZZATI: {prepped} \n")
pprint(prepped, width=100)

# dati di test per un solo record
raw_input  = records[0]
pre_input  = prepped[0]["pre"]  # indice e chiave del primo record
print("\n#################################################################################")
print("\nStarting AI Transformers operations...")

# 7. Preparazione del prompt per il modello LLM di normalizzazione/mappatura
# ISTRUZIONI operative per il testo PHP da convertire in JSON 
istruzioni = """
    - Formatta le date:
        - request_date: "YYYY-MM-DD HH:MM:SS" (da "data_richiesta" in formato "DD/MM/YYYY HH:MM").
        - check_in_date / check_out_date: "YYYY-MM-DD"; se check_out_date < check_in_date, invertile.
    - adults_number: se assente → 0; se children_number>0 e adults_number==0 → imposta 1.
    - children_number: numero di bambini (0–17 anni); se assente → 0. children_age: array di età (se assenti → []).
    - pet: deduci dal testo/flag; se non menzionato → false.
    - accommodation_type: "pitch" (piazzola/tenda/camper) oppure "housing_unit" (camera/casa mobile/appartamento). Se ignoto → null.
    - treatment: copia testuale del trattamento richiesto (es. "Solo Pernottamento").
    - treatment_code: deduci da note/richieste e poi dal campo trattamento. Valori ammessi:
        all_inclusive | full_board | half_board | bed_and_breakfast | room_only | null
    - source_reference: valorizza con "source_reference" di "content_json"
    - id_record: valorizza con il valore del campo "id" del record DB
    - source_name: valorizza con "script_name" del record DB
    - campaign_data: valorizza con "info_campagna_json" (dizionario del record DB)
    - table_name: "archivio_email_da_hosting" (fisso al momento)
    - request_target: deduci tra {single, couple, group, family} (o null se non deducibile).
    - country_code: deduci in base a lingua/nome/email/note → codice ISO2 (es. IT, NL, DE).
    - request_lang: la lingua del form o della pagina visitata.
    - campaign_data: oggetto; se assente restituisci {}.
    - attribution_data: valorizza con i campi del form:
        - acquisition_channel ∈ {affiliates,direct,display,email,organic,other,other_advertising,paid_search,referral,social}
        - medium ∈ {api,app,form,form-myreply,import-portali,manuale}
        - medium_section: string
        - category ∈ {altro,email,form,telefono,chat}
    - Campi richiesti: devono sempre esistere (usa null/""/[] quando previsto).
    - Restituisci ESCLUSIVAMENTE UN UNICO oggetto JSON, senza testo extra.
"""
schema_atteso = """
{
  "id_attivita": int|null,
  "source_name": "string",
  "id_record": int|null,
  "table_name": "string",
  "request_date": "YYYY-MM-DD HH:MM:SS",
  "check_in_date": "YYYY-MM-DD",
  "check_out_date": "YYYY-MM-DD",
  "adults_number": int,
  "children_number": number,
  "children_age": [int,...],
  "pet": null|string|boolean,
  "country_code": "CC",
  "request_lang": "string",
  "accommodation_type": "housing_unit"|"pitch"|null,
  "treatment": "string",
  "treatment_code": null|"all_inclusive"|"full_board"|"half_board"|"bed_and_breakfast"|"room_only",
  "request_target": null|"single"|"couple"|"group"|"family",
  "campaign_data": {},
  "attribution_data": {
    "acquisition_channel": "affiliates"|"direct"|"display"|"email"|"organic"|"other"|"other_advertising"|"paid_search"|"referral"|"social",
    "medium": "api"|"app"|"form"|"form-myreply"|"import-portali"|"manuale",
    "medium_section": "string",
    "category": "altro"|"email"|"form"|"telefono"|"chat"
  }
}
"""

USE_LLM = True  # False: per pipeline tutta deterministica (debug / fallback)

def make_user_message(llm_input: dict, istruzioni: str, schema_atteso: str) -> str:
    """
    Costruisce il messaggio utente per il modello:
    - UN solo blocco JSON che include raw_form + hints_pre
    - Fence ben formattati e chiusi
    - Nessun carattere 'appeso'
    """
    json_block = json.dumps(llm_input, ensure_ascii=False, indent=2)
    return (
f"""Analizza il JSON fornito:
```json
{json_block}
```
Restituisci in output un json così formattato:
```json
{schema_atteso}
```
Indicazioni:
{istruzioni}
IMPORTANTE:
Restituisci SOLO il JSON, senza testo aggiuntivo.
Se un campo è sconosciuto usa null/""/[] secondo lo schema.
"""
)

# Pre-trained models initialization
tok = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    torch_dtype=thc.float32,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
    trust_remote_code=False
)

for idx, (raw_input, pre_input) in enumerate(zip(records, (p["pre"] for p in prepped)), start=1):
    if idx % 50 == 1:
        print(f"[INFO] Processing record {idx}…")

    # 4a) costruttore deterministico 
    final_candidate = build_final_candidate(pre_input, raw_input)

    if not USE_LLM:
        # validazione+persistenza senza LLM
        try:
            final = validate_or_fix(final_candidate)
            write_jsonl(OUTPUT_JSONL, final)
            ok_count += 1
        except ValidationError as ve:
            write_jsonl(ERRORS_JSONL, {"idx": idx, "error": f"ValidationError: {str(ve)}",
                                       "raw_input": raw_input, "pre_input": pre_input})
            err_count += 1
        continue

    # (2) Input unico per LLM: raw + hints_pre
    llm_input = {
        "hints_pre": pre_input,        # indizi deterministici (il modello può copiarli)
        "raw_form":  raw_input         # raw originale
    }

    # prompt messages
    system_msg = "Sei un valido assistente per analizzare JSON di richieste informazioni turistiche e di generare un JSON formattato"
    user_msg = make_user_message(llm_input, istruzioni, schema_atteso)
    
    # consegnati a LLM
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "{"}  # nudge: inizia direttamente un oggetto JSON
    ]

    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt")

        # generazione dei token con LLM
        with thc.inference_mode():
            out = mdl.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )

        gen_only = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        raw_out = tok.batch_decode(gen_only, skip_special_tokens=True)[0]
        draft = extract_first_json(raw_out)
        if draft is None:
            write_jsonl(ERRORS_JSONL, {"idx": idx, "error": "NoJSONFound", "model_raw": raw_out[:2000]})
            continue

        # (3) Merge: LLM tocca solo i campi ambigui
        llm_keys = [
            "request_target","treatment","treatment_code","accommodation_type",
            "request_lang","country_code","pet","campaign_data","attribution_data",
            "request_date","check_in_date","check_out_date",
            "adults_number","children_number","children_age",
            "id_attivita","id_record","source_name","table_name", "source_reference"
        ]
        for k in llm_keys:
            if k in draft and draft[k] is not None:
                final_candidate[k] = draft[k]

        # (4) Validazione e persistenza
        final = validate_or_fix(final_candidate)
        write_jsonl(OUTPUT_JSONL, final)
        ok_count += 1

    except ValidationError as ve:
        write_jsonl(ERRORS_JSONL, {
            "idx": idx,
            "error": f"ValidationError: {str(ve)}",
            "model_raw": raw_out[:2000] if 'raw_out' in locals() else "",
            "raw_input": raw_input,
            "pre_input": pre_input
        })
        err_count += 1
    except Exception as e:
        write_jsonl(ERRORS_JSONL, {
            "idx": idx,
            "error": repr(e),
            "model_raw": raw_out[:2000] if 'raw_out' in locals() else "",
            "raw_input": raw_input,
            "pre_input": pre_input
        })
        err_count += 1

# Report finale
print(f"\nDone. OK: {ok_count} | ERR: {err_count} | Tot: {len(records)}")
print(f"Output: {OUTPUT_JSONL}")
if err_count:
    print(f"Error log: {ERRORS_JSONL}")

