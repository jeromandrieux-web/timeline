import json
import threading
import time
import uuid
from datetime import datetime
from typing import Callable, Dict, Any, Optional

from .case_db import open_db

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

class JobRunner:
    """
    Runner simple (thread) pour exécuter des jobs asynchrones.
    Chaque job met à jour la table jobs + documents (progress/status).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}

    def start_extract_job(
        self,
        db_path: str,
        doc_id: str,
        total_pages: int,
        worker_fn: Callable[[int], Dict[str, Any]],  # worker_fn(page_start)-> result {page_end, events_added, next_page}
        chunk_pages: int,
        max_events_per_chunk: int,
    ) -> str:
        job_id = str(uuid.uuid4())
        created = now_iso()

        con = open_db(db_path)
        try:
            con.execute(
                "INSERT INTO jobs(job_id,kind,doc_id,status,created_at,updated_at,message,progress_current,progress_total) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (job_id, "extract_llm", doc_id, "queued", created, created, "En attente", 0, total_pages),
            )
            con.execute(
                "UPDATE documents SET llm_status=?, llm_pages_done=?, llm_error=? WHERE doc_id=?",
                ("queued", 0, None, doc_id),
            )
            con.commit()
        finally:
            con.close()

        def run():
            con2 = open_db(db_path)
            try:
                con2.execute("UPDATE jobs SET status=?, updated_at=?, message=? WHERE job_id=?",
                             ("running", now_iso(), "Démarrage extraction", job_id))
                con2.execute("UPDATE documents SET llm_status=?, llm_error=? WHERE doc_id=?",
                             ("running", None, doc_id))
                con2.commit()
            finally:
                con2.close()

            pages_done = 0
            page_start = 1
            total_events = 0

            try:
                while True:
                    r = worker_fn(page_start)
                    page_end = int(r.get("page_end") or min(total_pages, page_start + chunk_pages - 1))
                    added = int(r.get("events_added") or 0)
                    nxt = r.get("next_page")
                    total_events += added

                    pages_done = max(pages_done, page_end)

                    con3 = open_db(db_path)
                    try:
                        con3.execute(
                            "UPDATE jobs SET updated_at=?, message=?, progress_current=? WHERE job_id=?",
                            (now_iso(),
                             f"Pages {page_start}-{page_end} : +{added} événements (total {total_events})",
                             pages_done,
                             job_id)
                        )
                        con3.execute(
                            "UPDATE documents SET llm_pages_done=?, llm_events_count=llm_events_count+? WHERE doc_id=?",
                            (pages_done, added, doc_id)
                        )
                        con3.commit()
                    finally:
                        con3.close()

                    if not nxt:
                        break
                    page_start = int(nxt)

                con4 = open_db(db_path)
                try:
                    con4.execute(
                        "UPDATE jobs SET status=?, updated_at=?, message=?, progress_current=?, result_json=? WHERE job_id=?",
                        ("done", now_iso(), f"Terminé: {total_events} événements", total_pages,
                         json.dumps({"events_added": total_events}, ensure_ascii=False),
                         job_id)
                    )
                    con4.execute(
                        "UPDATE documents SET llm_status=?, llm_pages_done=? WHERE doc_id=?",
                        ("done", total_pages, doc_id)
                    )
                    con4.commit()
                finally:
                    con4.close()

            except Exception as e:
                con5 = open_db(db_path)
                try:
                    con5.execute(
                        "UPDATE jobs SET status=?, updated_at=?, message=? WHERE job_id=?",
                        ("error", now_iso(), f"Erreur: {e}", job_id)
                    )
                    con5.execute(
                        "UPDATE documents SET llm_status=?, llm_error=? WHERE doc_id=?",
                        ("error", str(e), doc_id)
                    )
                    con5.commit()
                finally:
                    con5.close()

        t = threading.Thread(target=run, daemon=True)
        with self._lock:
            self._threads[job_id] = t
        t.start()
        return job_id
