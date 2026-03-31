"""
Name normalization and similarity utilities.

Implements Jaro-Winkler similarity and initial matching for inventor disambiguation.
"""

import re
import unicodedata
from typing import Optional


def normalize_name(name: Optional[str]) -> str:
    """
    Lowercase, strip accents, remove non-alpha characters, collapse whitespace.
    Returns empty string for None or empty input.
    """
    if not name:
        return ""
    # Unicode normalization (NFD) to decompose accented characters
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = re.sub(r"[^a-z\s]", "", name)
    return name.strip()


def initial_match(a: str, b: str) -> bool:
    """
    Return True if one name is an initial of the other
    (e.g. 'J' matches 'John' or 'Jane').
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    if len(na) == 1 and nb.startswith(na):
        return True
    if len(nb) == 1 and na.startswith(nb):
        return True
    return False


# ---------------------------------------------------------------------------
# Jaro-Winkler similarity
# ---------------------------------------------------------------------------

def _jaro(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    match_distance = max(match_distance, 0)

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """
    Compute Jaro-Winkler similarity.  Returns a value in [0, 1].
    prefix_weight p is typically 0.1 (Winkler standard).
    """
    jaro_sim = _jaro(s1, s2)
    # Count common prefix (up to 4 characters)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro_sim + prefix * prefix_weight * (1 - jaro_sim)


def name_similarity(a: Optional[str], b: Optional[str]) -> float:
    """
    Compute a name similarity score in [0, 1]:
      - 1.0  exact match (after normalization)
      - 0.90 initial match
      - Jaro-Winkler similarity otherwise
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if initial_match(na, nb):
        return 0.90
    return jaro_winkler(na, nb)
