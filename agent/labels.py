from __future__ import annotations


def canonicalize_label(raw_label: str | None) -> str | None:
    if raw_label is None:
        return None
    text = str(raw_label).strip()
    if not text:
        return None

    upper_text = text.upper()
    direct_map = {
        "BCC": "BCC",
        "ACK": "ACK",
        "NEV": "NEV",
        "SEK": "SEK",
        "SCC": "SCC",
        "MEL": "MEL",
    }
    if upper_text in direct_map:
        return direct_map[upper_text]

    lowered = text.lower()
    keyword_map = [
        ("basal cell", "BCC"),
        ("bcc", "BCC"),
        ("actinic keratos", "ACK"),
        ("ack", "ACK"),
        ("seborrheic keratos", "SEK"),
        ("seborrhoeic keratos", "SEK"),
        ("sek", "SEK"),
        ("squamous cell", "SCC"),
        ("scc", "SCC"),
        ("melanoma", "MEL"),
        ("mel", "MEL"),
        ("atypical nevus", "NEV"),
        ("atypical naevus", "NEV"),
        ("nevus", "NEV"),
        ("naevus", "NEV"),
        ("nevi", "NEV"),
        ("mole", "NEV"),
        ("nev", "NEV"),
    ]
    for keyword, canonical in keyword_map:
        if keyword in lowered:
            return canonical
    return None


def labels_match(left: str | None, right: str | None) -> bool | None:
    if left is None or right is None:
        return None
    if str(left) == str(right):
        return True
    left_canonical = canonicalize_label(left)
    right_canonical = canonicalize_label(right)
    if left_canonical and right_canonical:
        return left_canonical == right_canonical
    return False
