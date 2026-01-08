# core/repo.py
import json, os, base64, hashlib
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import urllib.request, urllib.error

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR     = PROJECT_ROOT / "repository"
UPLOADS_DIR  = PROJECT_ROOT / "uploads"
JSON_PATH    = REPO_DIR / "meeting_notes.json"

LOGS_DIR   = PROJECT_ROOT / "logs"
LOG_PATH   = LOGS_DIR / "repo_log.csv"


# -----------------------------------------------------------------------------
# Basic local storage helpers (as before)
# -----------------------------------------------------------------------------
def ensure_dirs():
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_existing() -> List[Dict]:
    """Load all records from repository/meeting_notes.json."""
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return []


# -----------------------------------------------------------------------------
# GitHub sync helpers
# -----------------------------------------------------------------------------
def _get_github_config():
    """
    Read GitHub settings from Streamlit secrets (preferred) or environment vars.
    Returns dict or None if not configured.
    """
    token = repo = branch = None
    # Prefer Streamlit secrets when running inside the app
    try:
        import streamlit as st  # type: ignore
        token = st.secrets.get("GITHUB_TOKEN")
        repo = st.secrets.get("GITHUB_REPO")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
    except Exception:
        # Fallback: environment variables (useful for local/dev)
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPO")
        branch = os.getenv("GITHUB_BRANCH", "main")

    if not token or not repo:
        return None
    return {"token": token, "repo": repo, "branch": branch or "main"}


def _github_request(method: str, url: str, token: str, body: dict | None = None) -> dict:
    """
    Minimal GitHub REST request using stdlib (no external dependencies).
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "meeting-notes-studio",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def push_file_to_github(local_path: Path, repo_rel_path: str, message: str) -> None:
    """
    Commit the given local file into the GitHub repo at `repo_rel_path`.
    - Creates the file if it doesn't exist
    - Updates it if it does (using existing SHA)
    Silently ignores errors so the UI is never broken by sync failures.
    """
    cfg = _get_github_config()
    if not cfg:
        # GitHub sync not configured; nothing to do
        return

    token = cfg["token"]
    repo  = cfg["repo"]
    branch = cfg["branch"]

    url = f"https://api.github.com/repos/{repo}/contents/{repo_rel_path}"

    # Compute file content (base64-encoded)
    content_bytes = local_path.read_bytes()
    b64_content = base64.b64encode(content_bytes).decode("utf-8")

    # Try to get existing file SHA (if file already exists on that branch)
    sha = None
    try:
        existing = _github_request("GET", f"{url}?ref={branch}", token)
        sha = existing.get("sha")
    except Exception:
        # File may not exist yet; that's fine
        sha = None

    payload = {
        "message": message,
        "content": b64_content,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        _github_request("PUT", url, token, body=payload)
    except Exception as e:
        try:
            import streamlit as st
            st.error(f"GitHub sync failed: {e}")
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Append records + snapshot + logs (with GitHub sync)
# -----------------------------------------------------------------------------
def append_records(new_records: List[Dict]) -> int:
    """
    Append to repository/meeting_notes.json; returns total record count.
    Also pushes the updated JSON to GitHub if configured.
    """
    ensure_dirs()
    data = load_existing()
    data.extend(new_records)
    JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # NEW: push JSON to GitHub
    try:
        rel_path = "repository/meeting_notes.json"
        msg = f"Update meeting notes ({len(data)} records)"
        push_file_to_github(JSON_PATH, rel_path, msg)
    except Exception:
        # Never fail the main flow due to sync problems
        pass

    return len(data)


def save_snapshot(text: str) -> Path:
    """
    Save the pasted notes into uploads/ for audit; returns file path.
    Also pushes the .txt snapshot into GitHub if configured.
    """
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:8]
    path = UPLOADS_DIR / f"notes_{ts}_{digest}.txt"
    path.write_text(text, encoding="utf-8")

    # NEW: push snapshot .txt to GitHub
    try:
        rel_path = f"uploads/{path.name}"
        msg = f"Add notes snapshot {path.name}"
        push_file_to_github(path, rel_path, msg)
    except Exception:
        pass

    return path


# -----------------------------------------------------------------------------
# Logging helpers (unchanged, plus ensure_dirs usage)
# -----------------------------------------------------------------------------
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
