import json
import re


_TOKEN_RE = re.compile(r"^\d{8,}$")


def normalize_token_id(value):
    text = str(value or "").strip().strip('"').strip("'")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text if _TOKEN_RE.fullmatch(text) else None


def parse_token_id_list(raw_value):
    if raw_value is None:
        return []

    seq = []
    if isinstance(raw_value, (list, tuple)):
        seq = list(raw_value)
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                seq = parsed
            else:
                seq = [parsed]
        except Exception:
            # Fallback for partially formatted list-like strings.
            text = text.strip("[]")
            seq = [part.strip() for part in text.split(",") if part.strip()]

    out = []
    for item in seq:
        token = normalize_token_id(item)
        if token and token not in out:
            out.append(token)
    return out
