# core/repo.py
import json, os, hashlib, base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import urllib.request, urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR     = PROJECT_ROOT / "repository"
UPLOADS_DIR  = PROJECT_ROOT / "uploads"
LOGS_DIR     = PROJECT_ROOT / "logs"

JSON_PATH = REPO_DIR / "meeting_notes.json"
LOG_PATH  = LOGS_DIR / "repo_log.csv"


def ensure_dirs():
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_existing() -> List[Dict]:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return []


def _get_github_config():
    try:
        import streamlit as st
        token  = st.secrets.get("GITHUB_TOKEN")
        repo   = st.secrets.get("GITHUB_REPO")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
    except Exception:
        token  = os.getenv("GITHUB_TOKEN")
        repo   = os.getenv("GITHUB_REPO")
        branch = os.getenv("GITHUB_BRANCH", "main")

    if not token or not repo:
        return None

    return {"token": token, "repo": repo, "branch": branch}


def _github_request(method: str, url: str, token: str, body: dict | None = None):
    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "meeting-notes-sync",
        "Accept": "application/vnd.github+json",
    }
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def push_file(local_path: Path, repo_rel_path: str, message: str):
    cfg = _get_github_config()
    if not cfg:
        return  # Sync disabled

    token  = cfg["token"]
    repo   = cfg["repo"]
    branch = cfg["branch"]

    url = f"https://api.github.com/repos/{repo}/contents/{repo_rel_path}"

    # Read & base64-encode
    content = base64.b64encode(local_path.read_bytes()).decode("utf-8")

    # Get existing SHA
    try:
        existing = _github_request("GET", f"{url}?ref={branch}", token)
        sha = existing.get("sha")
    except Exception:
        sha = None

    payload = {
        "message": message,
        "content": content,
        "branch": branch
    }
    if sha:
        payload["sha"] = sha

    try:
        _github_request("PUT", url, token, payload)
    except Exception as e:
        try:
            import streamlit as st
            st.error(f"GitHub sync failed: {e}")
        except:
            print("GitHub sync failed:", e)


def append_records(new_records: List[Dict]) -> int:
    ensure_dirs()
    data = load_existing()
    data.extend(new_records)

    JSON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # SYNC JSON
    push_file(JSON_PATH, "repository/meeting_notes.json",
              f"Update meeting notes ({len(data)} records)")

    return len(data)


def save_snapshot(text: str) -> Path:
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h  = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    path = UPLOADS_DIR / f"notes_{ts}_{h}.txt"
    path.write_text(text, encoding="utf-8")

    # SYNC UPLOAD
    push_file(path, f"uploads/{path.name}", f"Add snapshot {path.name}")

    return path


def append_log_rows(rows: List[Dict]):
    ensure_dirs()
    new = pd.DataFrame(rows)
    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        all = pd.concat([old, new], ignore_index=True)
    else:
        all = new
    all.to_csv(LOG_PATH, index=False)


def read_log_df():
    ensure_dirs()
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["Notes Type", "Date", "Upload Date", "Source ID"])
