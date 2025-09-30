# core/repo.py
import json, os
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR     = PROJECT_ROOT / "repository"
UPLOADS_DIR  = PROJECT_ROOT / "uploads"
JSON_PATH    = REPO_DIR / "meeting_notes.json"

def ensure_dirs():
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def load_existing() -> List[Dict]:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return []

def append_records(new_records: List[Dict]) -> int:
    """Append to repository/meeting_notes.json; returns total record count."""
    ensure_dirs()
    data = load_existing()
    data.extend(new_records)
    JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(data)

def save_snapshot(text: str) -> Path:
    """Save the pasted notes into uploads/ for audit; returns file path."""
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:8]
    path = UPLOADS_DIR / f"notes_{ts}_{digest}.txt"
    path.write_text(text, encoding="utf-8")
    return path

# --- Logging helpers ---
import pandas as pd

LOGS_DIR   = PROJECT_ROOT / "logs"
LOG_PATH   = LOGS_DIR / "repo_log.csv"

def _ensure_logs_dir():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def append_log_rows(rows: list[dict]) -> None:
    """
    rows: list of {"Notes Type": str, "Date": str, "Upload Date": str, "Source ID": str}
    """
    _ensure_logs_dir()
    df_new = pd.DataFrame(rows)
    if LOG_PATH.exists():
        df_old = pd.read_csv(LOG_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(LOG_PATH, index=False)

def read_log_df() -> pd.DataFrame:
    _ensure_logs_dir()
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["Notes Type", "Date", "Upload Date", "Source ID"])
