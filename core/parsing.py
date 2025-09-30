import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import unicodedata

# ----- Regex & constants -----
# Relaxed header: allow descriptors between "Notes" and the pipe, but DO NOT change capture indexes
HEADER_RE = re.compile(
    r'^\s*(Call Notes|EoD Notes)\b(?:[^|]*)\|\s*([0-9]{1,2})[-/\.]([0-9]{1,2})(?:[-/\.]([0-9]{2,4}))?\s*(?:[-–—:]\s*.*)?$',
    re.I
)
# Optional "w/ X" or "with X" before the pipe in the SAME header line
HEADER_WITH_PEOPLE_RE = re.compile(r'\b(?:w/|with)\s*([^|]+?)\s*(?=\|)', re.I)

HI_RE = re.compile(r'^\s*Hi\s+(.+?)\s*[,-]?\s*$', re.I)
WITH_RE = re.compile(r'notes\s+from\s+(?:the\s+)?call\s+(?:with|w/)\s+(.+?)[\.:]?\s*$', re.I)

ATTACH_RE = re.compile(r'([\w\-\s()&\.]+?\.(?:pptx?|xlsx?|docx?|pdf|oft|csv))', re.I)
URL_RE    = re.compile(r'(https?://[^\s)]+)', re.I)

# --- Normalize odd Unicode in header lines (NBSP, full-width pipe, smart dashes, ZWSP) ---
def _normalize_header_line(s: str) -> str:
    if not s:
        return s
    # keep your existing mappings
    s = (
        s.replace('\u00A0', ' ')   # NBSP -> space
         .replace('\u200B', '')    # zero-width space
         .replace('\uFF5C', '|')   # full-width pipe -> ASCII pipe
         .replace('\u2013', '-')   # en dash -> hyphen
         .replace('\u2014', '-')   # em dash -> hyphen
         .replace('\u2212', '-')   # minus sign -> hyphen
    )
    # NEW: tolerate " I " before a date; turn it into a pipe
    s = re.sub(r'\sI\s+(?=\d{1,2}[-/\.]\d{1,2}(?:[-/\.]\d{2,4})?\s*$)', ' | ', s)
    return s

# NEW: "arrow" workstream markers
WS_ARROW_TITLE_RE  = re.compile(r'^\s*([A-Za-z0-9][\w\s/&().\-]+?)\s*>\s*$', re.I)        # e.g., "Offsets >"
WS_ARROW_INLINE_RE = re.compile(r'^\s*([A-Za-z0-9][\w\s/&().\-]+?)\s*>\s*(.+)$', re.I)    # e.g., "Offsets > Send draft" or "Offsets > file.pptx"

SKIP_LINE_PATTERNS = [
    re.compile(r'please find below', re.I),
    re.compile(r'please find.*notes', re.I),
    re.compile(r'^\s*hope (you(?:\'|’)re|you are)\s*(doing\s*)?well[\s\.,!]*$', re.I),
    re.compile(r'^\s*hope all is well[\s\.,!]*$', re.I),
    re.compile(r'^\s*good (morning|afternoon|evening)[\s\.,!]*$', re.I),
    re.compile(r'^\s*(thanks|thank you|many thanks)[\s,\.!]*([A-Za-z .\'\-]{0,30})?$', re.I),
    re.compile(r'^\s*(best|best regards|regards|warm regards|kind regards|cheers|sincerely)[\s,\.!]*([A-Za-z .\'\-]{0,30})?$', re.I),
]

OWNER_PATTERNS = [
    re.compile(r'^\s*([A-Z][A-Za-z& ]+?)\s+(?:to|will|shall)\b'),
    re.compile(r'\bassigned\s+to\s+([A-Z][A-Za-z& ]+)\b', re.I),
]
OWNER_STOPWORDS = {
    "we", "plan", "preference", "timeline", "status", "agenda",
    "notes", "discussion", "discussions", "actions", "action items",
    "context", "updates"
}

SPEAKER_PATTERNS = [
    re.compile(r'^\s*([A-Z][A-Za-z]+)\s+(?:said|noted|asked|expressed|shared|suggested)\b'),
]

def normalize_ws_title(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().rstrip(':-'))

def normalize_owner_candidate(cand: str) -> Optional[str]:
    if not cand:
        return None
    c = re.sub(r'\s+', ' ', cand.strip()); cl = c.lower()
    if cl == "we":
        return "BCN Carbon Team"
    if cl in {"ben", "estabrook", "ben estabrook"}:
        return "Ben Estabrook"
    if cl.startswith("sam"):
        return "Sam"
    if cl in OWNER_STOPWORDS:
        return None
    if cl.startswith("carbon team"):
        return "BCN Carbon Team"
    if re.search(r'\b(team|dept|department|group|committee|office|BCN|Carbon)\b', c, re.I):
        return c
    if re.fullmatch(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?', c):
        return c
    return None

def normalize_person_list(raw: str) -> List[str]:
    """
    Split a free-form list of names, normalize known variants,
    and drop generic greetings like 'team', 'all', 'everyone'.
    """
    if not raw:
        return []
    s = raw.strip().rstrip(",;:.")
    # Normalize Ben Estabrook formats
    s = re.sub(r'\bEstabrook\s*,\s*Ben\b', 'Ben Estabrook', s, flags=re.I)
    s = re.sub(r'\bBen\s*,\s*Estabrook\b', 'Ben Estabrook', s, flags=re.I)
    # Split on common conjunctions/separators
    s = re.sub(r'\s*(?:,|&|\+|and|\|)\s*', ' | ', s, flags=re.I)
    parts = [p.strip() for p in s.split(' | ') if p.strip()]

    out: List[str] = []
    seen = set()
    for p in parts:
        k = p.lower()
        if k in {"team", "all", "everyone"}:     # drop generic addressees
            continue
        # normalize common short forms
        if re.fullmatch(r'(?i)(ben|estabrook|ben\s+estabrook)', p):
            p = "Ben Estabrook"
            k = p.lower()
        elif re.fullmatch(r'(?i)sam(\s+.*)?', p):
            p = "Sam"; k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out

def normalize_person_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip()
    if re.fullmatch(r'(?i)(ben|estabrook|ben\s+estabrook)', n):
        return "Ben Estabrook"
    if re.fullmatch(r'(?i)sam(\s+.*)?', n):
        return "Sam"
    return n

def ensure_bcn_second(participants: List[str], default_second="BCN Carbon Team") -> List[str]:
    names = [p for p in participants if p]
    if not any(p.lower() == default_second.lower() for p in names):
        names.append(default_second)
    others = [p for p in names if p.lower() != default_second.lower()]
    result = []
    if others:
        result.append(others[0]); result.append(default_second); result.extend(others[1:])
    else:
        result = [default_second]
    dedup = []; seen = set()
    for p in result:
        k = p.lower()
        if k not in seen:
            seen.add(k); dedup.append(p)
    return dedup

def parse_mmdd_to_date(mm: int, dd: int, yy: Optional[int], default_year: Optional[int]) -> str:
    if yy is None:
        yy = default_year if default_year else datetime.now().year
    if yy < 100:
        yy += 2000
    return datetime(yy, mm, dd).date().isoformat()

def find_owner(text: str) -> Optional[str]:
    for rx in OWNER_PATTERNS:
        m = rx.search(text)
        if m:
            owner_norm = normalize_owner_candidate(m.group(1))
            if owner_norm:
                return owner_norm
    return None

def find_speaker_explicit(text: str) -> Optional[str]:
    for rx in SPEAKER_PATTERNS:
        m = rx.search(text)
        if m:
            return m.group(1).strip()
    return None

def find_attachment(text: str) -> Optional[str]:
    m_url = URL_RE.search(text)
    if m_url:
        return m_url.group(1).strip()
    m = ATTACH_RE.search(text)
    if m:
        return m.group(1).strip()
    return None

# ----- Block parsing -----
def split_blocks(text: str, default_year: Optional[int]) -> List[Dict]:
    lines = text.splitlines()
    blocks = []; cur = None
    for line in lines:
        line_norm = _normalize_header_line(line)
        m = HEADER_RE.match(line_norm)
        if m:
            if cur: 
                blocks.append(cur)
            typ = m.group(1).strip().title()
            mm, dd, yy = int(m.group(2)), int(m.group(3)), m.group(4)
            yy = int(yy) if yy else None
            date_iso = parse_mmdd_to_date(mm, dd, yy, default_year)

            # Header-level "w/ <people>" capture on the normalized header line
            header_people = None
            header_with_person = None
            m_with = HEADER_WITH_PEOPLE_RE.search(line_norm)
            if m_with:
                raw_people = m_with.group(1)
                ppl = normalize_person_list(raw_people)
                ppl = [normalize_person_name(x) for x in ppl]
                if ppl:
                    header_people = ensure_bcn_second(ppl)
                    header_with_person = normalize_person_name(ppl[0])

            cur = {
                "notes_type": "Meeting Notes" if typ.lower().startswith("call") else "EoD Notes",
                "date": date_iso,
                "lines": [],
                "header_participants": header_people,
                "header_with_person": header_with_person,
            }
        else:
            if cur is None:
                continue
            cur["lines"].append(line)

    if cur: 
        blocks.append(cur)
    return blocks

def extract_participants_and_with(
    lines: List[str],
    header_participants: Optional[List[str]] = None,
    header_with_person: Optional[str] = None
) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Try (1) header-provided participants, (2) greetings, then (3) 'notes from the call with/w/ X'.
    Returns (participants_list, with_person_for_speaker_fallback).
    """
    if header_participants:
        return header_participants, header_with_person

    with_person = None
    # Greeting within first few lines
    for line in lines[:10]:
        m = HI_RE.match(line.strip())
        if m:
            names = normalize_person_list(m.group(1))
            names = [normalize_person_name(x) for x in names]
            return names, None
    # "notes from the call with ..."
    for line in lines[:10]:
        m = WITH_RE.search(line.strip())
        if m:
            raw = m.group(1).strip()
            names = normalize_person_list(raw)
            names = [normalize_person_name(x) for x in names]
            with_person = normalize_person_name(names[0]) if names else None
            return names, with_person
    return None, None

def parse_block(block: Dict, default_second_participant: str = "BCN Carbon Team") -> List[Dict]:
    lines = block["lines"]

    participants_list, with_person = extract_participants_and_with(
        lines,
        header_participants=block.get("header_participants"),
        header_with_person=block.get("header_with_person"),
    )
    if participants_list:
        participants_list = [normalize_person_name(p) for p in participants_list]
        participants_list = ensure_bcn_second(participants_list, default_second_participant)

    rows: List[Dict] = []
    current_workstream: Optional[str] = None
    current_ws_attachment: Optional[str] = None
    global_attachment: Optional[str] = None

    # Skip boilerplate before the substance starts
    start_idx = 0
    for i, line in enumerate(lines[:10]):
        if HI_RE.match(line) or re.search(r'please find below', line, re.I):
            continue
        start_idx = i
        break

    for raw_line in lines[start_idx:]:
        line = raw_line.strip()
        if not line:
            continue

        # boilerplate & separators
        if any(p.search(line) for p in SKIP_LINE_PATTERNS):
            continue
        if re.fullmatch(r'[-–—*•\s]{1,10}', line):
            continue

        # --- Workstream via ">" marker (explicit, overrides heuristics) ---
        m_arrow_inline = WS_ARROW_INLINE_RE.match(line)
        if m_arrow_inline:
            candidate = normalize_ws_title(m_arrow_inline.group(1))
            remainder = m_arrow_inline.group(2).strip()
            current_workstream = candidate
            current_ws_attachment = None  # reset for new WS

            # If remainder is a file/URL, treat as attachment (no bullet row)
            att_inline = find_attachment(remainder)
            if att_inline:
                current_ws_attachment = att_inline
                continue
            # Otherwise, treat remainder as the bullet text
            line = remainder

        else:
            m_arrow_title = WS_ARROW_TITLE_RE.match(line)
            if m_arrow_title:
                candidate = normalize_ws_title(m_arrow_title.group(1))
                current_workstream = candidate
                current_ws_attachment = None
                continue  # header-only line

        # --- Attachments (standalone) ---
        att = find_attachment(line)
        if att:
            if current_workstream:
                current_ws_attachment = att
            else:
                global_attachment = att
            continue

        # --- Owner / Speaker ---
        owner = find_owner(line)
        if owner:
            o = owner.strip().lower()
            if o == "we":
                owner = "BCN Carbon Team"
            elif o in {"plan", "preference", "timeline", "status", "agenda", "notes", "discussion", "context", "updates"}:
                owner = None

        # Speaker: explicit only for EoD; fallbacks allowed for Meeting Notes
        speaker = find_speaker_explicit(line)
        if speaker is None and block["notes_type"] != "EoD Notes":
            if block.get("header_with_person"):
                speaker = block.get("header_with_person")
            elif with_person:
                speaker = with_person
            elif participants_list:
                pl = [p.lower() for p in participants_list]
                if any(p.startswith("sam") for p in pl):
                    speaker = "Sam"
                elif "ben estabrook" in pl:
                    speaker = "Ben Estabrook"
        speaker = normalize_person_name(speaker)

        participants = " + ".join(participants_list) if participants_list else None
        owner_final = owner if owner else ("BCN Carbon Team" if participants and "BCN Carbon Team" in participants else None)

        # Effective attachment: WS-specific if present
        effective_attachment = current_ws_attachment

        rows.append({
            "Date": block["date"],
            "Notes Type": block["notes_type"],
            "Participants": participants,
            "Workstream": current_workstream,
            "Speaker": speaker,
            "Owner": owner_final,
            "Action / Comment": line,
            "File Attached": effective_attachment
        })

    # Propagate any global attachment to all rows without one (or append)
    if global_attachment:
        for r in rows:
            if r["File Attached"] and global_attachment not in str(r["File Attached"]):
                r["File Attached"] = f'{r["File Attached"]} | {global_attachment}'
            elif not r["File Attached"]:
                r["File Attached"] = global_attachment

    return rows

def parse_notes(text: str, default_year: Optional[int] = None) -> pd.DataFrame:
    blocks = split_blocks(text, default_year=default_year)
    all_rows: List[Dict] = []
    for b in blocks:
        all_rows.extend(parse_block(b))
    cols = [
        "Date","Notes Type","Participants","Workstream",
        "Speaker","Owner","Action / Comment","File Attached"
    ]
    return pd.DataFrame(all_rows, columns=cols)
