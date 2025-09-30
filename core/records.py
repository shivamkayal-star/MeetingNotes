# core/records.py
# Convert the edited DataFrame (Phase 1) into JSON records compatible with your pipeline.
from typing import List, Dict
import pandas as pd

REQUIRED_OUT_COLS = [
    "Date", "Notes Type", "Participants", "Speaker", "Owner",
    "Workstream", "Sub-workstream", "Action / Comment", "File Link"
]

def _canonize_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure final columns exist and are strings; map File Attached -> File Link; add Sub-workstream blank."""
    out = df.copy()

    # Rename File Attached -> File Link (your FAISS/JSON expects 'File Link')
    if "File Link" not in out.columns and "File Attached" in out.columns:
        out = out.rename(columns={"File Attached": "File Link"})

    # Ensure Sub-workstream exists (you said: no sub-WS in the simplified schema)
    if "Sub-workstream" not in out.columns:
        out["Sub-workstream"] = ""

    # Make sure all required columns exist (fill missing with "")
    for col in REQUIRED_OUT_COLS:
        if col not in out.columns:
            out[col] = ""

    # Coerce to string and strip
    for c in REQUIRED_OUT_COLS:
        out[c] = out[c].fillna("").astype(str).map(lambda x: x.strip())

    return out[REQUIRED_OUT_COLS]

def df_to_records(df: pd.DataFrame) -> List[Dict]:
    """
    Produce the exact shape used by convert_excel_to_json.py:
      [{"embedded_text": "...", "metadata": {...}}, ...]
    """
    df2 = _canonize_output_df(df)

    def compose_embedded_text(row: pd.Series) -> str:
        parts = []
        def add(label, val):
            if val: parts.append(f"{label}: {val}")
        add("Notes Type", row["Notes Type"])
        add("Participants", row["Participants"])
        add("Speaker", row["Speaker"])
        add("Owner", row["Owner"])
        add("Workstream", row["Workstream"])
        add("Sub-workstream", row["Sub-workstream"])
        add("Comment", row["Action / Comment"])
        return " | ".join(parts)

    records: List[Dict] = []
    for _, r in df2.iterrows():
        text = compose_embedded_text(r)
        if not text.strip():
            # skip rows that have no searchable content
            continue
        meta = {
            "Date": r["Date"],
            "Notes Type": r["Notes Type"],
            "Participants": r["Participants"],
            "Speaker": r["Speaker"],
            "Owner": r["Owner"],
            "Workstream": r["Workstream"],
            "Sub-workstream": r["Sub-workstream"],
            "Action / Comment": r["Action / Comment"],
            "File Link": r["File Link"],
        }
        records.append({
            "embedded_text": text,
            "metadata": meta
        })
    return records

