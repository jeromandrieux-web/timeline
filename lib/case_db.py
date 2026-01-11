import os
import sqlite3
from typing import Optional, Dict, Any, List, Tuple

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  file_name TEXT NOT NULL,
  doc_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  page_count INTEGER,
  file_path TEXT NOT NULL,

  llm_status TEXT NOT NULL DEFAULT 'idle',   -- idle|queued|running|done|error
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

  FOREIGN KEY(event_id) REFERENCES events(event_id),
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_events_doc_id ON events(doc_id);
CREATE INDEX IF NOT EXISTS idx_events_domain ON events(domain);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time);

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,                 -- extract_llm
  doc_id TEXT NOT NULL,
  status TEXT NOT NULL,               -- queued|running|done|error|canceled
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  message TEXT,
  progress_current INTEGER NOT NULL DEFAULT 0,
  progress_total INTEGER NOT NULL DEFAULT 0,
  result_json TEXT,

  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);
"""

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def case_root(cases_dir: str, case_id: str) -> str:
    return os.path.join(cases_dir, case_id)

def case_db_path(cases_dir: str, case_id: str) -> str:
    return os.path.join(case_root(cases_dir, case_id), "case.db")

def case_uploads_dir(cases_dir: str, case_id: str) -> str:
    return os.path.join(case_root(cases_dir, case_id), "uploads")

def open_db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_case(cases_dir: str, case_id: str) -> str:
    root = case_root(cases_dir, case_id)
    ensure_dir(root)
    ensure_dir(case_uploads_dir(cases_dir, case_id))

    dbp = case_db_path(cases_dir, case_id)
    con = open_db(dbp)
    try:
        con.executescript(SCHEMA_SQL)
        con.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('case_id',?)", (case_id,))
        con.commit()
    finally:
        con.close()
    return dbp

def list_cases(cases_dir: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(cases_dir):
        return []
    out = []
    for name in sorted(os.listdir(cases_dir)):
        p = os.path.join(cases_dir, name)
        if not os.path.isdir(p):
            continue
        dbp = os.path.join(p, "case.db")
        if os.path.isfile(dbp):
            out.append({"case_id": name, "path": p, "db_path": dbp})
    return out

def get_active_case_id(state_dir: str) -> Optional[str]:
    p = os.path.join(state_dir, "active_case.txt")
    try:
        with open(p, "r", encoding="utf-8") as f:
            v = f.read().strip()
            return v or None
    except Exception:
        return None

def set_active_case_id(state_dir: str, case_id: str) -> None:
    ensure_dir(state_dir)
    p = os.path.join(state_dir, "active_case.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(case_id)

def require_active_case(cases_dir: str, state_dir: str) -> Tuple[str, str]:
    case_id = get_active_case_id(state_dir)
    if not case_id:
        raise RuntimeError("Aucune enquête active. Crée/sélectionne une enquête.")
    dbp = case_db_path(cases_dir, case_id)
    if not os.path.isfile(dbp):
        raise RuntimeError(f"Enquête active introuvable sur disque: {case_id}")
    return case_id, dbp
