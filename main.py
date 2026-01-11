import os
import re
import csv
import json
import uuid
import queue
import sqlite3
import threading
import time
import requests
from io import BytesIO, StringIO
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Optional deps
try:
    from docx import Document  # python-docx
except Exception:
    Document = None


# =========================================================
# Config
# =========================================================

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
CASES_DIR = os.path.join(BASE_DIR, "cases")
ACTIVE_CASE_FILE = os.path.join(CASES_DIR, "_active_case.txt")

os.makedirs(CASES_DIR, exist_ok=True)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# --- LLM configuration (OpenAI-style route, e.g. Ollama / DeepSeek gateway) ---
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v3.1:671b-cloud")
LLM_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.2)
LLM_TOP_P = _env_float("OLLAMA_TOP_P", 0.9)
LLM_TOP_K = _env_int("OLLAMA_TOP_K", 40)
# ✅ on augmente le timeout par défaut (les grosses procédures / modèles cloud peuvent être lents)
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 600)

# =========================================================
# App
# =========================================================

app = FastAPI(title="Timeline · V2 (Cases)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =========================================================
# Models (API requests)
# =========================================================

class ChatTimelineRequest(BaseModel):
    question: str
    q: Optional[str] = None
    event_type: Optional[str] = None
    start_after: Optional[str] = None
    start_before: Optional[str] = None
    limit: int = 200
    offset: int = 0
    # vue: "page" (par défaut) ou "acte"
    view: Optional[str] = None


# =========================================================
# Helpers
# =========================================================

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def make_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): make_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [make_jsonable(v) for v in x]
    return str(x)

def _extract_first_json_object(text: str) -> str:
    if not text:
        raise ValueError("Réponse vide")
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    if start < 0:
        raise ValueError("Aucun objet JSON trouvé.")
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i+1]
    raise ValueError("Objet JSON incomplet.")

def call_llm_raw(system: str, user: str, timeout_s: Optional[int] = None) -> str:
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "top_k": LLM_TOP_K,
    }
    try:
        r = requests.post(LLM_API_BASE_URL, json=payload, timeout=int(timeout_s or LLM_TIMEOUT))
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Erreur LLM ({r.status_code}): {r.text}")
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data, ensure_ascii=False, default=str)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur appel LLM: {e}")

def call_llm_json(system: str, user: str, retry_once: bool = True, timeout_s: Optional[int] = None) -> Dict[str, Any]:
    txt = call_llm_raw(system, user, timeout_s=timeout_s)
    try:
        obj = json.loads(_extract_first_json_object(txt))
        if not isinstance(obj, dict):
            raise ValueError("JSON n'est pas un objet.")
        return obj
    except Exception as e:
        if not retry_once:
            raise HTTPException(status_code=500, detail=f"LLM JSON invalide: {e}. Réponse: {txt[:1200]}")
        strict_user = (
            user
            + "\n\nIMPORTANT: Réponds uniquement par un JSON valide (un objet), sans texte avant/après."
            + " Si tu hésites, renvoie events=[] et explique dans warnings."
        )
        txt2 = call_llm_raw(system, strict_user, timeout_s=timeout_s)
        try:
            obj2 = json.loads(_extract_first_json_object(txt2))
            if not isinstance(obj2, dict):
                raise ValueError("JSON n'est pas un objet.")
            return obj2
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"LLM JSON invalide après retry: {e2}. Réponse: {txt2[:1200]}")

def read_pdf_bytes_pages(data: bytes) -> Tuple[List[str], int]:
    try:
        reader = PdfReader(BytesIO(data), strict=False)
    except PdfReadError as e:
        raise HTTPException(status_code=400, detail=f"PDF illisible: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le PDF: {e}")

    pages_text: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages_text.append((t or "").replace("\x00", " ").strip())
    return pages_text, len(reader.pages)

def read_docx_bytes(data: bytes) -> str:
    if Document is None:
        raise HTTPException(status_code=500, detail="python-docx n'est pas installé (DOCX import failed).")
    try:
        doc = Document(BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le DOCX: {e}")
    return ("\n".join(p.text for p in doc.paragraphs) or "").replace("\x00", " ").strip()

def read_text_bytes(data: bytes) -> str:
    try:
        t = data.decode("utf-8", errors="ignore")
    except Exception:
        t = data.decode("latin-1", errors="ignore")
    return (t or "").replace("\x00", " ").strip()

def _normalize_iso_bound(value: Optional[str], bound: str) -> Optional[str]:
    """
    Rend la recherche fiable quand l'utilisateur saisit une date simple.
    - YYYY-MM-DD -> after: 00:00:00Z ; before: 23:59:59Z
    - YYYY-MM-DDTHH:MM -> ajoute :00Z
    - YYYY-MM-DDTHH:MM:SS -> ajoute Z
    - sinon: renvoie tel quel (ex déjà avec Z)
    """
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None

    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s + ("T23:59:59Z" if bound == "before" else "T00:00:00Z")

    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", s):
        return s + ":00Z"

    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", s):
        return s + "Z"

    return s


# =========================================================
# Case management + DB (1 case = 1 sqlite)
# =========================================================

def _case_id_safe(case_id: str) -> str:
    case_id = (case_id or "").strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id vide")
    if not re.match(r"^[a-zA-Z0-9_\-]{2,64}$", case_id):
        raise HTTPException(status_code=400, detail="case_id invalide (utilise lettres/chiffres/_/-)")
    return case_id

def case_path(case_id: str) -> str:
    return os.path.join(CASES_DIR, _case_id_safe(case_id))

def case_db_path(case_id: str) -> str:
    return os.path.join(case_path(case_id), "case.db")

def case_uploads_dir(case_id: str) -> str:
    return os.path.join(case_path(case_id), "uploads")

def case_cache_dir(case_id: str) -> str:
    return os.path.join(case_path(case_id), "cache")

def get_active_case_id() -> Optional[str]:
    if os.path.isfile(ACTIVE_CASE_FILE):
        cid = open(ACTIVE_CASE_FILE, "r", encoding="utf-8").read().strip()
        return cid or None
    return None

def set_active_case_id(case_id: str) -> None:
    os.makedirs(CASES_DIR, exist_ok=True)
    with open(ACTIVE_CASE_FILE, "w", encoding="utf-8") as f:
        f.write(case_id)

def ensure_case(case_id: str) -> None:
    p = case_path(case_id)
    os.makedirs(p, exist_ok=True)
    os.makedirs(case_uploads_dir(case_id), exist_ok=True)
    os.makedirs(case_cache_dir(case_id), exist_ok=True)
    init_db(case_id)

def db_connect(case_id: str) -> sqlite3.Connection:
    ensure_case(case_id)
    con = sqlite3.connect(case_db_path(case_id), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db(case_id: str) -> None:
    dbp = case_db_path(case_id)
    if os.path.isfile(dbp):
        return
    con = sqlite3.connect(dbp, check_same_thread=False)
    try:
        cur = con.cursor()
        cur.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS documents (
          doc_id TEXT PRIMARY KEY,
          file_name TEXT NOT NULL,
          doc_type TEXT NOT NULL,
          size_bytes INTEGER NOT NULL,
          created_at TEXT NOT NULL,
          page_count INTEGER,
          file_path TEXT,
          text_path TEXT,
          llm_status TEXT NOT NULL DEFAULT 'idle',  -- idle|queued|running|done|error
          llm_pages_done INTEGER NOT NULL DEFAULT 0,
          llm_error TEXT,
          llm_events_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS events (
          event_id TEXT PRIMARY KEY,
          doc_id TEXT NOT NULL,
          domain TEXT NOT NULL,
          type TEXT NOT NULL,
          start_time TEXT,
          end_time TEXT,
          time_precision TEXT,
          confidence REAL,
          summary TEXT,
          payload_json TEXT,
          FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        );

        CREATE TABLE IF NOT EXISTS actors (
          event_id TEXT NOT NULL,
          kind TEXT NOT NULL,
          value TEXT NOT NULL,
          role TEXT,
          FOREIGN KEY(event_id) REFERENCES events(event_id)
        );

        CREATE TABLE IF NOT EXISTS sources (
          event_id TEXT NOT NULL,
          doc_id TEXT NOT NULL,
          file_name TEXT,
          page INTEGER,
          quote TEXT,
          FOREIGN KEY(event_id) REFERENCES events(event_id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
        CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time);
        CREATE INDEX IF NOT EXISTS idx_sources_event ON sources(event_id);
        CREATE INDEX IF NOT EXISTS idx_actors_event ON actors(event_id);

        CREATE TABLE IF NOT EXISTS jobs (
          job_id TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          doc_id TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          started_at TEXT,
          finished_at TEXT,
          message TEXT,
          progress_current INTEGER NOT NULL DEFAULT 0,
          progress_total INTEGER NOT NULL DEFAULT 0,
          events_added INTEGER NOT NULL DEFAULT 0,
          error TEXT
        );
        """)
        con.commit()
    finally:
        con.close()


# =========================================================
# Procedure LLM extraction (PDF → events)
# =========================================================

ALLOWED_PROC_TYPES = [
    "AUDITION",
    "INTERROGATOIRE",
    "IPC",
    "INTERROGATOIRE_SUR_LE_FOND",
    "CONFRONTATION",
    "GARDE_A_VUE",
    "PERQUISITION",
    "TRANSPORT_SUR_LES_LIEUX",
    "SAISIE",
    "RESTITUTION",
    "NOTIFICATION",
    "REQUISITION_EMISE",
    "REQUISITION_RECEPTION",
    "EXPERTISE",
    "CRT",
    "RAPPORT_TECHNIQUE",
    "RAPPORT_PSYCHOLOGIQUE",
    "RAPPORT_PSYCHIATRIQUE",
    "PV_REDACTION",
    "INTERROGATOIRE_PREMIERE_COMPARUTION",
    "COMMISSION_ROGATOIRE_TECHNIQUE",
    "PROCEDURE_EVENT"
]

ALLOWED_ROLES_FR = {
    "TEMOIN", "MIS_EN_CAUSE", "VICTIME",
    "CIBLE_PERQUISITION", "GARDE_A_VUE",
    "LIEU", "SERVICE",
    "AUTRE"
}

ROLE_ALIAS_TO_FR = {
    "WITNESS": "TEMOIN",
    "SUSPECT": "MIS_EN_CAUSE",
    "VICTIM": "VICTIME",
    "SEARCH_TARGET": "CIBLE_PERQUISITION",
    "CUSTODY": "GARDE_A_VUE",
    "PLACE": "LIEU",
    "OTHER": "AUTRE",
    "TÉMOIN": "TEMOIN",
    "TEMOIN": "TEMOIN",
    "MISENCAUSE": "MIS_EN_CAUSE",
    "PERQUISITIONNE": "CIBLE_PERQUISITION",
    "PERQUISITIONNÉ": "CIBLE_PERQUISITION",
    "GAV": "GARDE_A_VUE",
}

RE_OPJ_NAME_CONTEXT = re.compile(
    r"\b(opj|officier de police judiciaire|gardien de la paix|brigadier|capitaine|commandant|lieutenant)\b",
    re.IGNORECASE | re.UNICODE
)

RE_REDAC_PHRASES = re.compile(
    r"(nous\s+soussign[eé]s?.{0,220}?)(?=\.|\n|$)|"
    r"(r[eé]dig[eé]\s+par\s+.+?)(?=\.|\n|$)|"
    r"(rapport\s+r[eé]dig[eé]\s+par\s+.+?)(?=\.|\n|$)",
    re.IGNORECASE | re.UNICODE
)

def normalize_proc_type(t: str) -> str:
    """
    Normalise les types d'actes vers une liste canonique (ALLOWED_PROC_TYPES).
    On accepte des abréviations/variantes fréquentes rencontrées dans les procédures.
    """
    tt = (t or "").strip().upper()
    if not tt:
        return "PROCEDURE_EVENT"
    if tt in ALLOWED_PROC_TYPES:
        return tt

    # Normalisation "agressive" : on garde lettres/chiffres/underscore uniquement
    tt2 = re.sub(r"[^A-Z0-9_]", "", tt)

    alias = {# Abréviations / variantes générales
        "GAV": "GARDE_A_VUE",
        "GARDEAVUE": "GARDE_A_VUE",
        "PV": "PV_REDACTION",
        "PROCESVERBAL": "PV_REDACTION",
        "PROCESVERBAUX": "PV_REDACTION",
        "PROCÈSVERBAL": "PV_REDACTION",
        "PROCÉSVERBAL": "PV_REDACTION",
        "PROCVERBAL": "PV_REDACTION",
        "REQUISITION": "REQUISITION_EMISE",
        "RAPPORT": "RAPPORT_TECHNIQUE",
        # IPC / interrogatoire de première comparution
        "IPC": "IPC",
        "INTERROGATOIREPREMIERECOMPARUTION": "IPC",
        "INTERROGATOIREDEPREMIERECOMPARUTION": "IPC",
        "INTERROGATOIRE1ERECOMPARUTION": "IPC",
        "PREMIERECOMPARUTION": "IPC",
        "1ERECOMPARUTION": "IPC",
        # Interrogatoire sur le fond
        "INTERROGATOIRESURLEFOND": "INTERROGATOIRE_SUR_LE_FOND",
        "INTERROGATOIREAUFOND": "INTERROGATOIRE_SUR_LE_FOND",
        "ISF": "INTERROGATOIRE_SUR_LE_FOND",
        # CRT / commission rogatoire technique (écoutes, suivi, sonorisation...)
        "CRT": "CRT",
        "COMMISSIONROGATOIRETECHNIQUE": "CRT",
        "COMMROGTECH": "CRT",
        "ECOUTE": "CRT",
        "ECOUTES": "CRT",
        "SONORISATION": "CRT",
        "SUIVI": "CRT",
        "GEOLOCALISATION": "CRT",
        # Rapports spécialisés
        "RAPPORTPSYCHOLOGIQUE": "RAPPORT_PSYCHOLOGIQUE",
        "RAPPORTPSY": "RAPPORT_PSYCHOLOGIQUE",
        "RAPPORTPSYCHIATRIQUE": "RAPPORT_PSYCHIATRIQUE",
        "RAPPORTPSYCHIATRIE": "RAPPORT_PSYCHIATRIQUE",}

    mapped = alias.get(tt2) or alias.get(tt)
    if mapped and mapped in ALLOWED_PROC_TYPES:
        return mapped

    # Heuristiques simples (mots-clés)
    if "PSYCHIATR" in tt2:
        return "RAPPORT_PSYCHIATRIQUE"
    if "PSYCHOLOG" in tt2 or tt2.endswith("PSY"):
        return "RAPPORT_PSYCHOLOGIQUE"
    if "COMMISSIONROGATOIRE" in tt2 and ("TECH" in tt2 or "ECOUT" in tt2 or "SONOR" in tt2):
        return "CRT"
    if "COMPARUTION" in tt2 and "INTERROGATOIRE" in tt2:
        return "IPC"
    if "INTERROGATOIRE" in tt2 and ("FOND" in tt2 or "AUFOND" in tt2):
        return "INTERROGATOIRE_SUR_LE_FOND"

    return "PROCEDURE_EVENT"

def to_actor_list(raw_actors: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw_actors, list):
        return out
    allowed_kinds = {"PHONE", "PERSON", "SERVICE", "PLACE", "ID", "OTHER"}
    for a in raw_actors:
        if not isinstance(a, dict):
            continue
        kind = str(a.get("kind") or "OTHER").strip().upper()
        value = str(a.get("value") or "").strip()
        role = a.get("role")
        role = str(role).strip().upper() if isinstance(role, str) and role.strip() else None
        if not value:
            continue
        if kind not in allowed_kinds:
            kind = "OTHER"
        if role:
            role = ROLE_ALIAS_TO_FR.get(role, role)
            role = re.sub(r"[^A-Z_]", "", role)
            if role not in ALLOWED_ROLES_FR:
                role = "AUTRE"
        out.append({"kind": kind, "value": value, "role": role})
    return out

def sanitize_summary(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = RE_REDAC_PHRASES.sub("", s).strip()
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def build_llm_prompt_for_pages(pages_text: List[str], page_start: int, page_end: int, file_name: str, max_events: int) -> str:
    chunk_pages = pages_text[page_start-1:page_end]
    joined = []
    for i, txt in enumerate(chunk_pages, start=page_start):
        t = (txt or "").strip()
        if not t:
            continue
        joined.append(f"[PAGE {i}]\n{t}")
    chunk_text = "\n\n".join(joined).strip()
    if not chunk_text:
        chunk_text = f"[PAGE {page_start}] (page vide ou texte non extrait)"

    schema_hint = {
        "events": [
            {
                "type": "AUDITION",
                "start_time": "2023-03-12T08:45:00Z",
                "end_time": None,
                "time_precision": "MINUTE",
                "confidence": 0.78,
                "actors": [
                    {"kind": "PERSON", "value": "M. X", "role": "MIS_EN_CAUSE"},
                    {"kind": "PERSON", "value": "Mme Y", "role": "TEMOIN"},
                    {"kind": "PLACE", "value": "domicile de M. X", "role": "CIBLE_PERQUISITION"}
                ],
                "summary": "Audition libre de M. X",
                "evidence_quote": "Le 12 mars 2023 à 08 h 45, ...",
                "source_page": 12
            }
        ],
        "warnings": []
    }

    return f"""
Tu lis un extrait de procédure pénale provenant du fichier "{file_name}".
Ta tâche est d'extraire une LISTE d'ÉVÉNEMENTS pour construire une frise chronologique.

Contraintes IMPORTANTES :
- Réponds uniquement par un JSON valide (un objet) suivant le schéma ci-dessous.
- Ne crée AUCUN événement sans preuve textuelle : chaque événement doit contenir "evidence_quote" (copie exacte).
- IMPORTANT: ne considère pas comme "acteurs" les rédacteurs/OPJ/agents qui rédigent l'acte.
- Si l'heure est absente, mets start_time à minuit et time_precision="DAY".
- "type" doit être dans cette liste (sinon "PROCEDURE_EVENT") :
  {", ".join(ALLOWED_PROC_TYPES)}
- ACTEURS: utilise des rôles FR :
  TEMOIN, MIS_EN_CAUSE, VICTIME, CIBLE_PERQUISITION, GARDE_A_VUE, LIEU, SERVICE, AUTRE
- NE PAS inclure OPJ / rédacteurs / enquêteurs comme acteurs.
- "source_page" = entier correspondant à la page du passage (repère [PAGE N]).
- Limite la sortie à {max_events} événements.

Schéma JSON attendu :
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}

Voici le texte (pages {page_start} à {page_end}) :
\"\"\"
{chunk_text}
\"\"\"
""".strip()

def llm_extract_events_for_pdf_slice(
    doc_id: str,
    file_name: str,
    pages_text: List[str],
    page_start: int,
    page_end: int,
    max_events: int = 150,
    timeout_s: Optional[int] = None,
) -> List[Dict[str, Any]]:
    system = "Tu es un assistant d'extraction d'événements pour des procédures pénales."
    user = build_llm_prompt_for_pages(pages_text, page_start, page_end, file_name, max_events=max_events)

    obj = call_llm_json(system=system, user=user, retry_once=True, timeout_s=timeout_s)
    raw_events = obj.get("events", [])
    if not isinstance(raw_events, list) or not raw_events:
        return []

    extracted: List[Dict[str, Any]] = []

    for r in raw_events[:max_events]:
        if not isinstance(r, dict):
            continue

        ev_type = normalize_proc_type(str(r.get("type") or "PROCEDURE_EVENT"))
        start_time = r.get("start_time")
        end_time = r.get("end_time")

        tp = str(r.get("time_precision") or "INCONNU").upper()
        tp_map = {
            "SECOND": "SECONDE",
            "MINUTE": "MINUTE",
            "HOUR": "HEURE",
            "DAY": "JOUR",
            "APPROX": "APPROX",
            "UNKNOWN": "INCONNU",
            "SECONDE": "SECONDE",
            "HEURE": "HEURE",
            "JOUR": "JOUR",
        }
        time_precision = tp_map.get(tp, "INCONNU")

        confidence = r.get("confidence")
        try:
            conf = float(confidence) if confidence is not None else 0.6
        except Exception:
            conf = 0.6

        actors = to_actor_list(r.get("actors"))
        actors = [a for a in actors if not ((a.get("kind") == "PERSON") and RE_OPJ_NAME_CONTEXT.search(a.get("value") or ""))]

        summary = sanitize_summary(str(r.get("summary") or "").strip())
        evidence_quote = str(r.get("evidence_quote") or "").strip()

        page_num = r.get("source_page")
        try:
            page_num = int(page_num) if page_num is not None else None
        except Exception:
            page_num = None

        if evidence_quote:
            eq = re.sub(r"\s+", " ", evidence_quote).strip()
            evidence_quote = (eq[:350] + "…") if len(eq) > 350 else eq

        if not summary:
            summary = evidence_quote[:120] if evidence_quote else "(sans résumé)"
        if len(summary) > 160:
            summary = summary[:157] + "..."

        quote_for_source = evidence_quote[:220] + ("…" if len(evidence_quote) > 220 else "") if evidence_quote else None

        extracted.append({
            "event_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "domain": "PROCEDURE",
            "type": ev_type,
            "start_time": str(start_time).strip() if start_time else None,
            "end_time": str(end_time).strip() if end_time else None,
            "time_precision": time_precision,
            "confidence": conf,
            "summary": summary,
            "payload": {
                "preuve_texte": evidence_quote or None,
                "llm": {"modele": MODEL_NAME, "pages": [page_start, page_end]},
                "raw_llm_event": make_jsonable(r),
            },
            "actors": actors,
            "source": {"doc_id": doc_id, "file_name": file_name, "page": page_num, "quote": quote_for_source},
        })

    return extracted


# =========================================================
# Jobs (background extraction queue)
# =========================================================

JOB_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
JOB_CANCEL = set()
WORKER_STARTED = False

def _job_worker_loop():
    while True:
        job = JOB_QUEUE.get()
        try:
            _run_job(job)
        except Exception as e:
            try:
                case_id = job.get("case_id")
                job_id = job.get("job_id")
                if case_id and job_id:
                    con = db_connect(case_id)
                    try:
                        con.execute("UPDATE jobs SET status=?, finished_at=?, error=?, message=? WHERE job_id=?",
                                    ("error", now_iso(), str(e), "Erreur worker", job_id))
                        con.commit()
                    finally:
                        con.close()
            except Exception:
                pass
        finally:
            JOB_QUEUE.task_done()

def ensure_worker():
    global WORKER_STARTED
    if WORKER_STARTED:
        return
    t = threading.Thread(target=_job_worker_loop, daemon=True)
    t.start()
    WORKER_STARTED = True

def _load_pages_text(case_id: str, doc_id: str) -> List[str]:
    con = db_connect(case_id)
    try:
        row = con.execute("SELECT text_path, file_path, doc_type FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="doc_id inconnu")
        if (row["doc_type"] or "").lower() != "pdf":
            raise HTTPException(status_code=400, detail="Extraction LLM : PDF uniquement")
        text_path = row["text_path"]
        file_path = row["file_path"]
    finally:
        con.close()

    if text_path and os.path.isfile(text_path):
        return json.loads(open(text_path, "r", encoding="utf-8").read())

    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail="Fichier PDF absent sur disque")

    raw = open(file_path, "rb").read()
    pages_text, _ = read_pdf_bytes_pages(raw)

    cache_path = os.path.join(case_cache_dir(case_id), f"pages_text_{doc_id}.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(pages_text, ensure_ascii=False))

    con2 = db_connect(case_id)
    try:
        con2.execute("UPDATE documents SET text_path=? WHERE doc_id=?", (cache_path, doc_id))
        con2.commit()
    finally:
        con2.close()

    return pages_text

def _insert_events(case_id: str, events: List[Dict[str, Any]]) -> int:
    if not events:
        return 0
    con = db_connect(case_id)
    try:
        cur = con.cursor()
        added = 0
        for e in events:
            payload_json = json.dumps(e.get("payload") or {}, ensure_ascii=False, default=str)
            cur.execute("""
              INSERT OR IGNORE INTO events(event_id, doc_id, domain, type, start_time, end_time, time_precision, confidence, summary, payload_json)
              VALUES(?,?,?,?,?,?,?,?,?,?)
            """, (
                e["event_id"], e["doc_id"], e["domain"], e["type"],
                e.get("start_time"), e.get("end_time"), e.get("time_precision"),
                float(e.get("confidence") or 0.6),
                e.get("summary") or "",
                payload_json
            ))
            for a in (e.get("actors") or []):
                if not a or not a.get("value"):
                    continue
                cur.execute("INSERT INTO actors(event_id, kind, value, role) VALUES(?,?,?,?)",
                            (e["event_id"], a.get("kind") or "OTHER", a.get("value"), a.get("role")))
            s = e.get("source") or {}
            cur.execute("INSERT INTO sources(event_id, doc_id, file_name, page, quote) VALUES(?,?,?,?,?)",
                        (e["event_id"], s.get("doc_id") or e["doc_id"], s.get("file_name"), s.get("page"), s.get("quote")))
            added += 1
        con.commit()
        return added
    finally:
        con.close()

def _is_timeout_http_exc(exc: Exception) -> bool:
    # FastAPI HTTPException wraps our message; check common timeout substrings
    msg = str(exc) or ""
    return ("Read timed out" in msg) or ("timed out" in msg) or ("Timeout" in msg) or ("timeout" in msg)

def _run_job(job: Dict[str, Any]) -> None:
    case_id = job["case_id"]
    job_id = job["job_id"]
    doc_id = job["doc_id"]
    chunk_pages = int(job.get("chunk_pages", 20))
    max_events_per_chunk = int(job.get("max_events_per_chunk", 150))
    # ✅ reprise: si fourni, on démarre à cette page (sinon: llm_pages_done+1)
    start_page = int(job.get("start_page", 0) or 0)

    if job_id in JOB_CANCEL:
        return

    con = db_connect(case_id)
    try:
        con.execute("UPDATE jobs SET status=?, started_at=?, message=? WHERE job_id=?",
                    ("running", now_iso(), "Démarrage extraction", job_id))
        con.execute("UPDATE documents SET llm_status=?, llm_error=? WHERE doc_id=?",
                    ("running", None, doc_id))
        con.commit()
    finally:
        con.close()

    pages_text = _load_pages_text(case_id, doc_id)
    total_pages = len(pages_text)

    conx = db_connect(case_id)
    try:
        doc = conx.execute("SELECT file_name, llm_pages_done FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        file_name = (doc["file_name"] if doc else doc_id)
        pages_done = int(doc["llm_pages_done"] or 0) if doc else 0
    finally:
        conx.close()

    events_total_added = 0

    if start_page >= 1:
        page_start = start_page
    else:
        page_start = min(total_pages, (pages_done + 1 if pages_done >= 1 else 1))

    conp = db_connect(case_id)
    try:
        conp.execute("UPDATE jobs SET progress_total=?, progress_current=? WHERE job_id=?",
                     (total_pages, max(0, page_start - 1), job_id))
        conp.commit()
    finally:
        conp.close()

    # ✅ stratégie anti-timeout : retry avec chunk réduit si timeout
    min_chunk = 5

    while page_start <= total_pages:
        if job_id in JOB_CANCEL:
            con_cancel = db_connect(case_id)
            try:
                con_cancel.execute("UPDATE jobs SET status=?, finished_at=?, message=? WHERE job_id=?",
                                   ("canceled", now_iso(), "Annulé", job_id))
                con_cancel.execute("UPDATE documents SET llm_status=?, llm_error=? WHERE doc_id=?",
                                   ("error", "Annulé", doc_id))
                con_cancel.commit()
            finally:
                con_cancel.close()
            return

        page_end = min(total_pages, page_start + chunk_pages - 1)

        con2 = db_connect(case_id)
        try:
            con2.execute("UPDATE jobs SET message=?, progress_current=? WHERE job_id=?",
                         (f"Extraction pages {page_start}-{page_end}", page_end, job_id))
            con2.execute("UPDATE documents SET llm_pages_done=? WHERE doc_id=?",
                         (page_end, doc_id))
            con2.commit()
        finally:
            con2.close()

        extracted: List[Dict[str, Any]] = []
        attempt = 0
        local_chunk = chunk_pages

        while attempt < 3:
            attempt += 1
            try:
                extracted = llm_extract_events_for_pdf_slice(
                    doc_id=doc_id,
                    file_name=file_name,
                    pages_text=pages_text,
                    page_start=page_start,
                    page_end=min(total_pages, page_start + local_chunk - 1),
                    max_events=max_events_per_chunk,
                    timeout_s=LLM_TIMEOUT,
                )
                # succès
                page_end = min(total_pages, page_start + local_chunk - 1)
                break
            except HTTPException as he:
                # timeout -> réduire le chunk puis retry
                if _is_timeout_http_exc(he) and local_chunk > min_chunk:
                    local_chunk = max(min_chunk, local_chunk // 2)
                    time.sleep(1.0)
                    continue
                raise
            except Exception as e:
                if _is_timeout_http_exc(e) and local_chunk > min_chunk:
                    local_chunk = max(min_chunk, local_chunk // 2)
                    time.sleep(1.0)
                    continue
                raise

        added = _insert_events(case_id, extracted)
        events_total_added += added

        con3 = db_connect(case_id)
        try:
            con3.execute("UPDATE jobs SET events_added=? WHERE job_id=?",
                         (events_total_added, job_id))
            con3.execute("UPDATE documents SET llm_events_count=llm_events_count+?, llm_pages_done=? WHERE doc_id=?",
                         (added, page_end, doc_id))
            con3.commit()
        finally:
            con3.close()

        page_start = page_end + 1

    con_done = db_connect(case_id)
    try:
        con_done.execute("UPDATE jobs SET status=?, finished_at=?, message=?, progress_current=? WHERE job_id=?",
                         ("done", now_iso(), f"Terminé (+{events_total_added} événements)", total_pages, job_id))
        con_done.execute("UPDATE documents SET llm_status=?, llm_error=?, llm_pages_done=? WHERE doc_id=?",
                         ("done", None, total_pages, doc_id))
        con_done.commit()
    finally:
        con_done.close()


# =========================================================
# UI + Ping
# =========================================================

@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    static_index = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(static_index):
        return FileResponse(static_index)
    return JSONResponse({"status": "ok", "message": "UI not found (index.html missing)."})

@app.get("/api/ping")
def ping():
    return {"status": "ok", "ts": now_iso()}




@app.get("/help", include_in_schema=False)
def help_page():
    """Page d'aide utilisateur (HTML)."""
    html = r"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Timeline · Aide</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" crossorigin="anonymous">
  <style>
    body{background:#0b1220;color:#e5e7eb;}
    .card{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);}
    code,kbd{background:rgba(255,255,255,0.08);padding:.1rem .35rem;border-radius:.35rem}
    a{color:#93c5fd}
    .step{border-left:3px solid rgba(59,130,246,.65);padding-left:1rem;margin-bottom:1rem}
    .small-muted{color:rgba(229,231,235,.75)}
  </style>
</head>
<body>
<div class="container py-4">
  <div class="d-flex justify-content-between align-items-center flex-wrap gap-2 mb-3">
    <div>
      <h1 class="h3 mb-1">Aide · Timeline V2</h1>
      <div class="small-muted">Guide de prise en main — import → extraction LLM → lecture en frise (pages) ou par acte → analyses & rapports.</div>
    </div>
    <a class="btn btn-outline-light btn-sm" href="/">Retour à l’app</a>
  </div>

  <div class="row g-3">
    <div class="col-12 col-lg-7">
      <div class="card p-3">
        <h2 class="h5">1) Démarrer une enquête (base dédiée)</h2>
        <div class="step">
          <ol class="mb-0">
            <li>Dans l’entête, clique sur <strong>Créer</strong> et donne un identifiant simple (ex: <kbd>flag_2026</kbd>).</li>
            <li>Vérifie qu’elle est affichée comme <strong>Enquête active</strong>.</li>
            <li>Chaque enquête a sa propre base <em>SQLite</em> : documents, événements, acteurs, sources et jobs.</li>
          </ol>
        </div>

        <h2 class="h5 mt-3">2) Importer une procédure / des pièces</h2>
        <div class="step">
          <ul class="mb-0">
            <li>Glisse-dépose tes fichiers (PDF, DOCX, TXT/MD) dans la zone d’upload.</li>
            <li>Seuls les PDF peuvent être traités par l’extraction LLM (bouton <strong>Extraire (LLM)</strong>).</li>
          </ul>
        </div>

        <h2 class="h5 mt-3">3) Lancer l’extraction LLM</h2>
        <div class="step">
          <ul class="mb-0">
            <li>Dans <strong>Documents</strong>, clique <strong>Extraire (LLM)</strong> sur le PDF.</li>
            <li>Tu vois une progression (pages traitées / total). Les événements arrivent au fil de l’eau.</li>
            <li>Si ça bloque: ton serveur LLM peut time-out. Augmente <kbd>LLM_TIMEOUT</kbd> (ex: 600) et/ou baisse <kbd>chunk_pages</kbd> (ex: 10).</li>
          </ul>
        </div>

        <h2 class="h5 mt-3">4) Lire la frise</h2>
        <div class="step">
          <ul class="mb-0">
            <li><strong>Vue Pages</strong> : granularité fine (souvent 1…N événements par page).</li>
            <li><strong>Vue Actes</strong> : regroupe plusieurs pages d’un même acte (PV, IPC, CRT, rapports…).</li>
            <li>Astuce : clique une ligne pour ouvrir le détail (acteurs, sources, payload).</li>
          </ul>
        </div>

        <h2 class="h5 mt-3">5) Filtrer efficacement</h2>
        <div class="step">
          <ul class="mb-0">
            <li><strong>Recherche libre</strong> : cherche dans résumé, type, acteurs, quote, nom du fichier.</li>
            <li><strong>Type</strong> : filtre un type d’acte (AUDITION, GAV, PV_REDACTION…).</li>
            <li><strong>Début après/avant</strong> : accepte <kbd>YYYY-MM-DD</kbd> ou ISO complet. Exemple: <kbd>2026-01-01</kbd>.</li>
          </ul>
        </div>

        <h2 class="h5 mt-3">6) Rapports</h2>
        <div class="step">
          <ul class="mb-0">
            <li><strong>Rapport par personne</strong> : événements + exemples avec sources.</li>
            <li><strong>Rapport par type d’acte</strong> : volumes + top acteurs + exemples.</li>
            <li>Ces rapports sont “déterministes” (pas du LLM), donc reproductibles.</li>
          </ul>
        </div>

        <h2 class="h5 mt-3">7) Analyse LLM (sur la sélection)</h2>
        <div class="step">
          <ul class="mb-0">
            <li>Applique d’abord tes filtres, puis pose une question.</li>
            <li>Le LLM doit citer <kbd>SRC</kbd> / <kbd>QUOTE</kbd> pour appuyer chaque point.</li>
            <li>Utilise les boutons rapides (Résumé, Contradictions…).</li>
          </ul>
        </div>

        <h2 class="h5 mt-3">8) Export CSV</h2>
        <div class="step">
          <ul class="mb-0">
            <li>Export de la sélection courante (filtres). Idéal pour Excel / analyse.</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="col-12 col-lg-5">
      <div class="card p-3 mb-3">
        <h2 class="h5">Conseils “qualité de données”</h2>
        <ul class="mb-0">
          <li>Si un acte est “éclaté” : la vue <strong>Actes</strong> le regroupe automatiquement par référence (PV n°, IPC, CRT…).</li>
          <li>Pour améliorer encore : privilégie un PDF avec un texte bien extractible (pas un scan image).</li>
          <li>Les acteurs “OPJ/rédacteurs” sont filtrés autant que possible.</li>
        </ul>
      </div>

      <div class="card p-3 mb-3">
        <h2 class="h5">Types d’actes reconnus (canonique)</h2>
        <div class="small-muted mb-2">La reconnaissance reste progressive (liste évolutive). Les alias courants (IPC, CRT…) sont normalisés.</div>
        <ul class="mb-0">
          <li>AUDITION, INTERROGATOIRE, CONFRONTATION</li>
          <li>GARDE_A_VUE, PERQUISITION, SAISIE, RESTITUTION</li>
          <li>TRANSPORT_SUR_LES_LIEUX, NOTIFICATION</li>
          <li>REQUISITION_EMISE, REQUISITION_RECEPTION</li>
          <li>EXPERTISE, RAPPORT_TECHNIQUE, PV_REDACTION</li>
          <li><strong>INTERROGATOIRE_PREMIERE_COMPARUTION</strong> (IPC)</li>
          <li><strong>COMMISSION_ROGATOIRE_TECHNIQUE</strong> (CRT)</li>
          <li><strong>RAPPORT_PSYCHOLOGIQUE</strong>, <strong>RAPPORT_PSYCHIATRIQUE</strong></li>
          <li><strong>INTERROGATOIRE_SUR_LE_FOND</strong></li>
          <li>PROCEDURE_EVENT (fallback)</li>
        </ul>
      </div>

      <div class="card p-3">
        <h2 class="h5">Dépannage rapide</h2>
        <ul class="mb-0">
          <li><strong>404 favicon</strong> : non bloquant. (On le neutralise côté front.)</li>
          <li><strong>Timeout LLM</strong> : augmente <kbd>LLM_TIMEOUT</kbd> ou baisse <kbd>chunk_pages</kbd>.</li>
          <li><strong>“bloqué à 1000”</strong> : augmente la limite côté UI ou utilise la pagination (offset/limit).</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="small-muted mt-4">
    © Timeline V2 — aide locale (aucune donnée envoyée ailleurs que ton LLM configuré).
  </div>
</div>
</body>
</html>"""
    return HTMLResponse(content=html, status_code=200)
# =========================================================
# API: Cases
# =========================================================

@app.get("/api/cases")
def list_cases():
    items = []
    for name in os.listdir(CASES_DIR):
        if name.startswith("_"):
            continue
        p = os.path.join(CASES_DIR, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "case.db")):
            items.append(name)
    items.sort()
    return {"status": "ok", "cases": items, "active_case": get_active_case_id()}

@app.post("/api/cases")
def create_case(case_id: str = Query(...)):
    case_id = _case_id_safe(case_id)
    ensure_case(case_id)
    set_active_case_id(case_id)
    return {"status": "ok", "case_id": case_id, "active_case": case_id}

@app.post("/api/cases/use")
def use_case(case_id: str = Query(...)):
    case_id = _case_id_safe(case_id)
    ensure_case(case_id)
    set_active_case_id(case_id)
    return {"status": "ok", "active_case": case_id}

def _require_active_case() -> str:
    cid = get_active_case_id()
    if not cid:
        cid = "case_default"
        ensure_case(cid)
        set_active_case_id(cid)
    else:
        ensure_case(cid)
    return cid


# =========================================================
# API: Docs
# =========================================================

@app.get("/api/docs")
def list_docs():
    case_id = _require_active_case()
    con = db_connect(case_id)
    try:
        rows = con.execute("""
            SELECT doc_id, file_name, doc_type, size_bytes, created_at, page_count,
                   llm_status, llm_pages_done, llm_error, llm_events_count
            FROM documents
            ORDER BY created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    case_id = _require_active_case()

    filename = file.filename or "document"
    name_lower = filename.lower()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    doc_id = str(uuid.uuid4())
    created_at = now_iso()

    safe_name = re.sub(r"[^a-zA-Z0-9_\-. ]", "_", filename)[:140]
    file_path = os.path.join(case_uploads_dir(case_id), f"{doc_id}_{safe_name}")

    if name_lower.endswith(".pdf"):
        doc_type = "pdf"
        pages_text, page_count = read_pdf_bytes_pages(raw)
        with open(file_path, "wb") as f:
            f.write(raw)
        text_path = os.path.join(case_cache_dir(case_id), f"pages_text_{doc_id}.json")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(pages_text, ensure_ascii=False))
    elif name_lower.endswith(".docx"):
        doc_type = "docx"
        _ = read_docx_bytes(raw)
        page_count = None
        with open(file_path, "wb") as f:
            f.write(raw)
        text_path = None
    elif name_lower.endswith(".txt") or name_lower.endswith(".md"):
        doc_type = "txt"
        _ = read_text_bytes(raw)
        page_count = None
        with open(file_path, "wb") as f:
            f.write(raw)
        text_path = None
    else:
        raise HTTPException(status_code=400, detail="Extension non supportée (PDF/DOCX/TXT/MD).")

    con = db_connect(case_id)
    try:
        con.execute("""
            INSERT INTO documents(doc_id, file_name, doc_type, size_bytes, created_at, page_count, file_path, text_path,
                                  llm_status, llm_pages_done, llm_error, llm_events_count)
            VALUES(?,?,?,?,?,?,?,?, 'idle', 0, NULL, 0)
        """, (doc_id, filename, doc_type, len(raw), created_at, page_count, file_path, text_path))
        con.commit()
    finally:
        con.close()

    return {"status": "ok", "doc_id": doc_id, "file_name": filename, "doc_type": doc_type, "size_bytes": len(raw), "created_at": created_at, "page_count": page_count}

@app.post("/api/delete_doc")
def delete_doc(doc_id: str = Query(...)):
    case_id = _require_active_case()
    con = db_connect(case_id)
    try:
        doc = con.execute("SELECT file_path, text_path FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="doc_id inconnu")

        ev_ids = [r["event_id"] for r in con.execute("SELECT event_id FROM events WHERE doc_id=?", (doc_id,)).fetchall()]
        for eid in ev_ids:
            con.execute("DELETE FROM actors WHERE event_id=?", (eid,))
            con.execute("DELETE FROM sources WHERE event_id=?", (eid,))
        con.execute("DELETE FROM events WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM jobs WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        con.commit()

        for p in [doc["file_path"], doc["text_path"]]:
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        return {"status": "ok"}
    finally:
        con.close()


# =========================================================
# API: Jobs
# =========================================================

@app.post("/api/jobs/extract")
def start_extract_job(
    doc_id: str = Query(...),
    chunk_pages: int = Query(20, ge=1, le=80),
    max_events_per_chunk: int = Query(150, ge=10, le=300),
    resume: int = Query(1, ge=0, le=1, description="1=reprendre à llm_pages_done+1 ; 0=démarrer à 1"),
):
    case_id = _require_active_case()
    ensure_worker()

    con = db_connect(case_id)
    try:
        doc = con.execute("SELECT doc_type, page_count, llm_status, llm_pages_done FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="doc_id inconnu")
        if (doc["doc_type"] or "").lower() != "pdf":
            raise HTTPException(status_code=400, detail="Extraction LLM : PDF uniquement")
        if (doc["llm_status"] or "") in ("queued", "running"):
            row = con.execute("SELECT job_id FROM jobs WHERE doc_id=? AND status IN ('queued','running') ORDER BY created_at DESC LIMIT 1", (doc_id,)).fetchone()
            return {"status": "ok", "job_id": (row["job_id"] if row else None), "note": "Extraction déjà en cours"}
        total = int(doc["page_count"] or 0)
        pages_done = int(doc["llm_pages_done"] or 0)
    finally:
        con.close()

    job_id = str(uuid.uuid4())
    created_at = now_iso()
    start_page = (pages_done + 1) if (resume == 1 and pages_done >= 1) else 1

    con2 = db_connect(case_id)
    try:
        con2.execute("""
            INSERT INTO jobs(job_id, kind, doc_id, status, created_at, message, progress_current, progress_total, events_added)
            VALUES(?, 'extract_llm_pdf', ?, 'queued', ?, ?, ?, ?, 0)
        """, (job_id, doc_id, created_at, f"En file (départ p.{start_page})", max(0, start_page-1), total))
        con2.execute("UPDATE documents SET llm_status=?, llm_error=? WHERE doc_id=?", ("queued", None, doc_id))
        con2.commit()
    finally:
        con2.close()

    JOB_QUEUE.put({
        "case_id": case_id,
        "job_id": job_id,
        "doc_id": doc_id,
        "chunk_pages": chunk_pages,
        "max_events_per_chunk": max_events_per_chunk,
        "start_page": start_page,
    })
    return {"status": "ok", "job_id": job_id, "start_page": start_page}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    case_id = _require_active_case()
    con = db_connect(case_id)
    try:
        row = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="job_id inconnu")
        return dict(row)
    finally:
        con.close()

@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    case_id = _require_active_case()
    JOB_CANCEL.add(job_id)
    con = db_connect(case_id)
    try:
        con.execute("UPDATE jobs SET status=?, message=? WHERE job_id=? AND status IN ('queued','running')",
                    ("canceled", "Annulation demandée", job_id))
        con.commit()
    finally:
        con.close()
    return {"status": "ok"}


# =========================================================
# API: Timeline (page) + pagination + "acte" grouping
# =========================================================

def filter_events_db(
    case_id: str,
    q: Optional[str] = None,
    event_type: Optional[str] = None,
    start_after: Optional[str] = None,
    start_before: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    start_after_n = _normalize_iso_bound(start_after, "after")
    start_before_n = _normalize_iso_bound(start_before, "before")
    event_type_n = (event_type or "").strip().upper() or None

    limit = max(1, min(int(limit), 20000))
    offset = max(0, int(offset))

    con = db_connect(case_id)
    try:
        where = []
        params: List[Any] = []

        if event_type_n:
            where.append("upper(e.type) = ?")
            params.append(event_type_n)

        if start_after_n:
            where.append("(e.start_time IS NOT NULL AND e.start_time >= ?)")
            params.append(start_after_n)

        if start_before_n:
            where.append("(e.start_time IS NOT NULL AND e.start_time <= ?)")
            params.append(start_before_n)

        if q:
            ql = q.strip().lower()
            like = f"%{ql}%"
            where.append("""
            (
              lower(e.summary) LIKE ?
              OR lower(e.type) LIKE ?
              OR lower(e.payload_json) LIKE ?
              OR EXISTS (SELECT 1 FROM sources s WHERE s.event_id=e.event_id AND (lower(COALESCE(s.quote,'')) LIKE ? OR lower(COALESCE(s.file_name,'')) LIKE ?))
              OR EXISTS (SELECT 1 FROM actors a WHERE a.event_id=e.event_id AND lower(a.value) LIKE ?)
            )
            """)
            params.extend([like, like, like, like, like, like])

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        rows = con.execute(f"""
            SELECT e.*
            FROM events e
            {where_sql}
            ORDER BY (e.start_time IS NULL) ASC, e.start_time ASC
            LIMIT ? OFFSET ?
        """, (*params, int(limit), int(offset))).fetchall()

        events: List[Dict[str, Any]] = []
        for r in rows:
            ev = dict(r)
            arows = con.execute("SELECT kind,value,role FROM actors WHERE event_id=?", (ev["event_id"],)).fetchall()
            ev["actors"] = [dict(ar) for ar in arows]
            srows = con.execute("SELECT doc_id,file_name,page,quote FROM sources WHERE event_id=? ORDER BY page ASC LIMIT 10", (ev["event_id"],)).fetchall()
            ev["sources"] = [dict(sr) for sr in srows]
            try:
                ev["payload"] = json.loads(ev.get("payload_json") or "{}")
            except Exception:
                ev["payload"] = {}
            del ev["payload_json"]
            events.append(ev)

        return events
    finally:
        con.close()

def _actors_signature(ev: Dict[str, Any]) -> str:
    # signature stable (value+role) triée
    parts = []
    for a in (ev.get("actors") or []):
        if not a or not a.get("value"):
            continue
        v = str(a.get("value")).strip().lower()
        r = str(a.get("role") or "").strip().upper()
        parts.append(f"{r}:{v}" if r else v)
    parts.sort()
    return "|".join(parts)

def _first_page(ev: Dict[str, Any]) -> Optional[int]:
    ss = ev.get("sources") or []
    if not ss:
        return None
    p = ss[0].get("page")
    try:
        return int(p) if p is not None else None
    except Exception:
        return None

# =========================================================
# Act grouping (ACTE view)
# =========================================================

RE_PV_NUM = re.compile(r"\b(?:PV|P\.V\.|PROC[ÈE]S[-\s]?VERBAL)\s*(?:n[°o]\s*)?(\d{1,5})\b", re.IGNORECASE)
RE_IPC = re.compile(r"\bIPC\b|interrogatoire\s+de\s+premi[eè]re\s+comparution", re.IGNORECASE)
RE_CRT = re.compile(r"\bCRT\b|commission\s+rogatoire\s+technique", re.IGNORECASE)
RE_RAPPORT_PSY = re.compile(r"rapport\s+psychologique", re.IGNORECASE)
RE_RAPPORT_PSYCHIA = re.compile(r"rapport\s+psychiatrique", re.IGNORECASE)
RE_INTERRO_FOND = re.compile(r"interrogatoire\s+sur\s+le\s+fond", re.IGNORECASE)

def infer_act_ref(ev: Dict[str, Any]) -> Optional[str]:
    """
    Essaie de donner une “référence d’acte” stable pour regrouper plusieurs pages.
    Retourne une chaîne courte (ex: 'PV 12', 'IPC', 'CRT', 'RAPPORT_PSYCHIATRIQUE') ou None.
    """
    texts: List[str] = []
    if isinstance(ev.get("summary"), str):
        texts.append(ev["summary"])
    # evidence quotes / payload
    p = ev.get("payload") or {}
    if isinstance(p, dict):
        if isinstance(p.get("preuve_texte"), str):
            texts.append(p["preuve_texte"])
        raw = p.get("raw_llm_event")
        if isinstance(raw, dict):
            for k in ("summary", "evidence_quote", "title", "heading"):
                if isinstance(raw.get(k), str):
                    texts.append(raw.get(k))
    big = " \n ".join([t for t in texts if t]).strip()
    if not big:
        return None

    m = RE_PV_NUM.search(big)
    if m:
        return f"PV {m.group(1)}"

    if RE_IPC.search(big):
        return "IPC"

    if RE_CRT.search(big):
        return "CRT"

    if RE_RAPPORT_PSYCHIA.search(big):
        return "RAPPORT_PSYCHIATRIQUE"

    if RE_RAPPORT_PSY.search(big):
        return "RAPPORT_PSYCHOLOGIQUE"

    if RE_INTERRO_FOND.search(big):
        return "INTERROGATOIRE_SUR_LE_FOND"

    return None

def group_events_into_acts(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ⚠️ Heuristique volontairement simple (MVP V2) :
    on regroupe des événements "par page" en "actes" quand ils sont consécutifs :
    - même doc_id
    - même type
    - même jour (YYYY-MM-DD de start_time) ou start_time vide
    - signature d'acteurs similaire
    - pages contiguës (gap <= 1)
    """
    if not events:
        return []

    def day_of(ev):
        st = ev.get("start_time") or ""
        return st[:10] if len(st) >= 10 else ""

    def key_of(ev):
        return (ev.get("doc_id") or "", ev.get("type") or "", day_of(ev), _actors_signature(ev))

    out: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for ev in events:
        ev_page = _first_page(ev)
        if cur is None:
            cur = {
            "event_id": str(uuid.uuid4()),
                "doc_id": ev.get("doc_id"),
                "domain": ev.get("domain") or "PROCEDURE",
                "type": ev.get("type"),
                "start_time": ev.get("start_time"),
                "end_time": ev.get("end_time"),
                "time_precision": ev.get("time_precision"),
                "confidence": ev.get("confidence"),
                "summary": (f"{infer_act_ref(ev)} — " if infer_act_ref(ev) else "") + (ev.get("summary") or ""),
                "actors": ev.get("actors") or [],
                "sources": ev.get("sources") or [],
                "payload": {
                    "mode": "ACTE",
                    "children_event_ids": [ev.get("event_id")],
                    "pages": {"from": ev_page, "to": ev_page},
                },
            }
            continue

        cur_key = (cur.get("doc_id") or "", cur.get("type") or "", (cur.get("start_time") or "")[:10], _actors_signature(cur))
        ev_key = key_of(ev)

        cur_pages = (cur.get("payload") or {}).get("pages") or {}
        cur_to = cur_pages.get("to")
        try:
            cur_to_i = int(cur_to) if cur_to is not None else None
        except Exception:
            cur_to_i = None

        contiguous = (cur_to_i is None or ev_page is None or abs(ev_page - cur_to_i) <= 1)

        if (ev_key == cur_key) and contiguous:
            # merge into current act
            if ev_page is not None:
                if cur_pages.get("from") is None:
                    cur_pages["from"] = ev_page
                cur_pages["to"] = ev_page
            cur["payload"]["pages"] = cur_pages
            cur["payload"]["children_event_ids"].append(ev.get("event_id"))

            # merge sources (keep first 10)
            ss = cur.get("sources") or []
            for s in (ev.get("sources") or []):
                if len(ss) >= 10:
                    break
                ss.append(s)
            cur["sources"] = ss

            # merge actors (union by value+role+kind)
            seen = set()
            merged = []
            for a in (cur.get("actors") or []) + (ev.get("actors") or []):
                if not a or not a.get("value"):
                    continue
                sig = f"{a.get('kind','')}|{a.get('role','')}|{a.get('value','')}".lower()
                if sig in seen:
                    continue
                seen.add(sig)
                merged.append(a)
            cur["actors"] = merged

            # summary: keep first + add short continuation if it brings info
            summ = (ev.get("summary") or "").strip()
            if summ and summ not in (cur.get("summary") or "") and len(cur.get("summary") or "") < 260:
                cur["summary"] = (cur.get("summary") or "").strip()
                if cur["summary"]:
                    cur["summary"] += " · " + summ[:140]
                else:
                    cur["summary"] = summ[:140]
            continue

        # close current
        # make summary more explicit with pages range
        pages = (cur.get("payload") or {}).get("pages") or {}
        pfrom, pto = pages.get("from"), pages.get("to")
        if pfrom is not None and pto is not None and pfrom != pto:
            cur["summary"] = f"{cur.get('summary')}".strip()
            cur["payload"]["pages"] = {"from": pfrom, "to": pto}
        out.append(cur)

        # start new
        cur = {
            "event_id": str(uuid.uuid4()),
            "doc_id": ev.get("doc_id"),
            "domain": ev.get("domain") or "PROCEDURE",
            "type": ev.get("type"),
            "start_time": ev.get("start_time"),
            "end_time": ev.get("end_time"),
            "time_precision": ev.get("time_precision"),
            "confidence": ev.get("confidence"),
            "summary": (f"{infer_act_ref(ev)} — " if infer_act_ref(ev) else "") + (ev.get("summary") or ""),
            "actors": ev.get("actors") or [],
            "sources": ev.get("sources") or [],
            "payload": {
                "mode": "ACTE",
                "children_event_ids": [ev.get("event_id")],
                "pages": {"from": ev_page, "to": ev_page},
            },
        }

    if cur is not None:
        out.append(cur)
    return out

@app.get("/api/timeline")
def timeline(
    q: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    start_after: Optional[str] = Query(None),
    start_before: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=20000),
    offset: int = Query(0, ge=0, le=500000),
    view: str = Query("page", description="page|acte"),
):
    case_id = _require_active_case()
    events = filter_events_db(case_id, q=q, event_type=event_type, start_after=start_after, start_before=start_before, limit=limit, offset=offset)
    if (view or "").lower() == "acte":
        return group_events_into_acts(events)
    return events

@app.get("/api/event_types")
def event_types():
    case_id = _require_active_case()
    con = db_connect(case_id)
    try:
        rows = con.execute("""
            SELECT type, COUNT(*) as n
            FROM events
            GROUP BY type
            ORDER BY n DESC, type ASC
            LIMIT 200
        """).fetchall()
        return {"status": "ok", "types": [{"type": r["type"], "count": r["n"]} for r in rows]}
    finally:
        con.close()


# =========================================================
# API: Chat timeline (Markdown-ready)
# =========================================================

def _compact_events_for_llm(events: List[Dict[str, Any]], max_chars: int = 90000) -> str:
    lines: List[str] = []
    total = 0
    for i, e in enumerate(events, start=1):
        src = (e.get("sources") or [None])[0] if e.get("sources") else None
        file_name = (src.get("file_name") if isinstance(src, dict) else "")
        page = (src.get("page") if isinstance(src, dict) else None)
        quote = (src.get("quote") if isinstance(src, dict) else "")

        actors = ", ".join([a.get("value") for a in (e.get("actors") or []) if a and a.get("value")]) or ""
        dt = e.get("start_time") or ""
        typ = e.get("type") or ""
        summ = (e.get("summary") or "").replace("\n", " ").strip()

        line = f"{i}. {dt} {typ} | {actors} | {summ}"
        if file_name or page or quote:
            line += f" | SRC: {file_name}"
            if page is not None:
                line += f" p.{page}"
            if quote:
                line += f" | QUOTE: {quote}"
        line += "\n"

        if total + len(line) > max_chars:
            lines.append(f"...(contexte tronqué à {i-1} éléments)\n")
            break
        lines.append(line)
        total += len(line)
    return "".join(lines)

@app.post("/api/chat_timeline")
def chat_timeline(req: ChatTimelineRequest):
    case_id = _require_active_case()
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question vide.")

    view = (req.view or "page").lower().strip()
    events = filter_events_db(
        case_id,
        q=req.q,
        event_type=req.event_type,
        start_after=req.start_after,
        start_before=req.start_before,
        limit=max(1, min(int(req.limit or 200), 500)),
        offset=max(0, int(req.offset or 0)),
    )
    if view == "acte":
        events_for_llm = group_events_into_acts(events)
    else:
        events_for_llm = events

    context = _compact_events_for_llm(events_for_llm)

    system = (
        "Tu es un assistant d'analyse d'enquête. "
        "On te donne une sélection d'événements de timeline extraits depuis des documents. "
        "Tu dois répondre UNIQUEMENT sur ce contexte. "
        "Ne devine pas. Si une info manque, dis-le. "
        "Quand tu affirmes un fait, cite au moins 1 QUOTE ou SRC."
    )

    user = (
        "Voici le CONTEXTE (événements visibles/filtrés) :\n"
        "-----\n"
        f"{context}\n"
        "-----\n\n"
        f"QUESTION: {question}\n\n"
        "Réponds en MARKDOWN propre et lisible (comme un rapport) avec :\n"
        "## 1) Synthèse\n"
        "## 2) Chronologie utile (si pertinent)\n"
        "## 3) Actes / types concernés\n"
        "## 4) Acteurs / rôles\n"
        "## 5) Éléments probants (avec citations SRC/QUOTE)\n"
        "## 6) Incohérences / zones d'ombre\n"
        "Contraintes de forme:\n"
        "- titres markdown (##)\n"
        "- paragraphes courts\n"
        "- listes à puces\n"
        "- citations en bloc (> ) quand tu cites QUOTE\n"
    )

    answer = call_llm_raw(system=system, user=user)
    return {"answer": answer, "events_used": len(events_for_llm), "note": f"Réponse basée sur la sélection actuelle (view={view})."}


# =========================================================
# API: Reports deterministic
# =========================================================

def _event_src_string(e: Dict[str, Any]) -> str:
    ss = (e.get("sources") or [])
    if not ss:
        return ""
    s = ss[0]
    fn = s.get("file_name") or s.get("doc_id") or ""
    p = f" p.{s.get('page')}" if s.get("page") is not None else ""
    q = f" — {s.get('quote')}" if s.get("quote") else ""
    return f"{fn}{p}{q}"

def _event_best_quote(e: Dict[str, Any]) -> str:
    ss = (e.get("sources") or [])
    if ss and ss[0].get("quote"):
        return str(ss[0].get("quote"))
    p = e.get("payload") or {}
    if isinstance(p.get("preuve_texte"), str) and p.get("preuve_texte").strip():
        return p.get("preuve_texte").strip()
    return ""

@app.get("/api/report_by_person")
def report_by_person(
    q: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    start_after: Optional[str] = Query(None),
    start_before: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
    max_examples: int = Query(5, ge=1, le=20)
):
    case_id = _require_active_case()
    events = filter_events_db(case_id, q=q, event_type=event_type, start_after=start_after, start_before=start_before, limit=limit, offset=0)

    groups: Dict[str, Dict[str, Any]] = {}

    def add_event_to_person(name: str, e: Dict[str, Any]):
        if not name:
            return
        g = groups.get(name)
        if not g:
            g = {"person": name, "events": 0, "types": {}, "first_time": None, "last_time": None, "examples": []}
            groups[name] = g

        g["events"] += 1
        typ = str(e.get("type") or "—")
        g["types"][typ] = int(g["types"].get(typ, 0)) + 1

        st = e.get("start_time") or ""
        if st:
            if g["first_time"] is None or st < g["first_time"]:
                g["first_time"] = st
            if g["last_time"] is None or st > g["last_time"]:
                g["last_time"] = st

        if len(g["examples"]) < max_examples:
            g["examples"].append({
                "start_time": e.get("start_time"),
                "type": e.get("type"),
                "summary": e.get("summary"),
                "src": _event_src_string(e),
                "quote": _event_best_quote(e),
            })

    for e in events:
        names = []
        for a in (e.get("actors") or []):
            if a and a.get("value"):
                names.append(str(a.get("value")).strip())
        seen = set()
        for n in names:
            if not n or n in seen:
                continue
            seen.add(n)
            add_event_to_person(n, e)

    rows = list(groups.values())
    for r in rows:
        types_sorted = sorted(r["types"].items(), key=lambda x: x[1], reverse=True)
        r["types"] = [{"type": k, "count": v} for k, v in types_sorted]
    rows.sort(key=lambda x: x["events"], reverse=True)

    return {"status": "ok", "total_events": len(events), "people": rows}

@app.get("/api/report_by_type")
def report_by_type(
    q: Optional[str] = Query(None),
    start_after: Optional[str] = Query(None),
    start_before: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
    max_examples: int = Query(5, ge=1, le=20)
):
    case_id = _require_active_case()
    events = filter_events_db(case_id, q=q, event_type=None, start_after=start_after, start_before=start_before, limit=limit, offset=0)

    groups: Dict[str, Dict[str, Any]] = {}

    def add_event_to_type(t: str, e: Dict[str, Any]):
        t = t or "—"
        g = groups.get(t)
        if not g:
            g = {"type": t, "events": 0, "actors": {}, "first_time": None, "last_time": None, "examples": []}
            groups[t] = g

        g["events"] += 1
        st = e.get("start_time") or ""
        if st:
            if g["first_time"] is None or st < g["first_time"]:
                g["first_time"] = st
            if g["last_time"] is None or st > g["last_time"]:
                g["last_time"] = st

        for a in (e.get("actors") or []):
            if a and a.get("value"):
                key = str(a.get("value")).strip()
                g["actors"][key] = int(g["actors"].get(key, 0)) + 1

        if len(g["examples"]) < max_examples:
            g["examples"].append({
                "start_time": e.get("start_time"),
                "summary": e.get("summary"),
                "src": _event_src_string(e),
                "quote": _event_best_quote(e),
            })

    for e in events:
        add_event_to_type(str(e.get("type") or "—"), e)

    rows = list(groups.values())
    for r in rows:
        actors_sorted = sorted(r["actors"].items(), key=lambda x: x[1], reverse=True)[:30]
        r["top_actors"] = [{"actor": k, "count": v} for k, v in actors_sorted]
        del r["actors"]

    rows.sort(key=lambda x: x["events"], reverse=True)
    return {"status": "ok", "total_events": len(events), "types": rows}


# =========================================================
# Export CSV
# =========================================================

@app.get("/api/export.csv")
def export_csv(
    q: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    start_after: Optional[str] = Query(None),
    start_before: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
):
    case_id = _require_active_case()
    events = filter_events_db(case_id, q=q, event_type=event_type, start_after=start_after, start_before=start_before, limit=limit, offset=0)

    def generate():
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "event_id", "domain", "type", "start_time", "end_time", "time_precision",
            "confidence", "summary", "actors", "source_file", "source_page", "source_quote", "payload_json"
        ])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for e in events:
            src = (e.get("sources") or [None])[0] if e.get("sources") else None
            acteurs = ""
            if e.get("actors"):
                acteurs = " | ".join([
                    f"{(a.get('role') + ':') if a.get('role') else ''}{a.get('value')}" for a in e.get("actors") if a and a.get("value")
                ])
            payload_json = json.dumps(e.get("payload") or {}, ensure_ascii=False, default=str)

            writer.writerow([
                e.get("event_id"),
                e.get("domain"),
                e.get("type"),
                e.get("start_time") or "",
                e.get("end_time") or "",
                e.get("time_precision") or "",
                e.get("confidence") or "",
                e.get("summary") or "",
                acteurs,
                (src.get("file_name") if isinstance(src, dict) else ""),
                (src.get("page") if isinstance(src, dict) and src.get("page") is not None else ""),
                (src.get("quote") if isinstance(src, dict) else ""),
                payload_json
            ])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    return StreamingResponse(generate(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="timeline_export.csv"'})
