"""
Company name normalization and similarity utilities.

Handles abbreviations (Inc., LLC, Corp.), token-level overlap,
and fuzzy similarity for matching assignee names to LinkedIn companies.
"""

import re
from typing import Optional
from .name_utils import jaro_winkler

# Common legal suffixes to strip before comparison
_LEGAL_SUFFIXES = re.compile(
    r"\b("
    r"inc\.?|incorporated|"
    r"llc\.?|l\.l\.c\.?|"
    r"ltd\.?|limited|"
    r"corp\.?|corporation|"
    r"co\.?|company|"
    r"lp\.?|l\.p\.?|"
    r"llp\.?|l\.l\.p\.?|"
    r"plc\.?|"
    r"gmbh|"
    r"sa|s\.a\.|"
    r"ag|a\.g\.|"
    r"bv|b\.v\.|"
    r"nv|n\.v\.|"
    r"pty\.?|"
    r"technologies|technology|tech|"
    r"solutions|systems|services|"
    r"international|global|"
    r"group|holdings|enterprises|"
    r"laboratories|labs?"
    r")\b",
    re.IGNORECASE,
)

# Common abbreviation expansions
_ABBREV_MAP = {
    "intl": "international",
    "int'l": "international",
    "natl": "national",
    "nat'l": "national",
    "mfg": "manufacturing",
    "mfr": "manufacturer",
    "mgmt": "management",
    "mgr": "manager",
    "dept": "department",
    "univ": "university",
    "inst": "institute",
    "assoc": "associates",
    "assn": "association",
    "sys": "systems",
    "svc": "services",
    "svcs": "services",
    "tech": "technologies",
    "dev": "development",
    "res": "research",
    "engr": "engineering",
    "engrg": "engineering",
    "pharma": "pharmaceuticals",
    "semi": "semiconductor",
    "elec": "electronics",
    "comms": "communications",
    "comm": "communications",
    "med": "medical",
    "bio": "biosciences",
    "sci": "sciences",
    "mfg": "manufacturing",
    "indus": "industries",
}


def normalize_company(name: Optional[str]) -> str:
    """
    Normalize a company name:
    1. Lowercase
    2. Expand known abbreviations
    3. Strip legal suffixes
    4. Remove punctuation / extra whitespace
    """
    if not name:
        return ""

    text = name.lower()
    # Remove punctuation except hyphens (keep for compound names)
    text = re.sub(r"[^\w\s\-]", " ", text)
    # Expand abbreviations (whole word match)
    tokens = text.split()
    tokens = [_ABBREV_MAP.get(t, t) for t in tokens]
    text = " ".join(tokens)
    # Strip legal suffixes
    text = _LEGAL_SUFFIXES.sub(" ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_overlap(a: str, b: str) -> float:
    """
    Jaccard-like token overlap: |A ∩ B| / |A ∪ B|.
    Ignores single-character tokens (initials / noise).
    """
    tokens_a = {t for t in a.split() if len(t) > 1}
    tokens_b = {t for t in b.split() if len(t) > 1}
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def company_similarity(a: Optional[str], b: Optional[str]) -> float:
    """
    Compute a composite company similarity score in [0, 1]:
      - 1.0  exact normalized match
      - max(token_overlap, jaro_winkler) otherwise
    Both inputs are normalized before comparison.
    """
    na, nb = normalize_company(a), normalize_company(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    token_score = _token_overlap(na, nb)
    jw_score = jaro_winkler(na, nb)
    return max(token_score, jw_score)


def best_company_match(assignee: Optional[str], linkedin_companies: list) -> float:
    """
    Return the highest company similarity score between the patent assignee
    and any company in the LinkedIn profile's company list.
    """
    if not assignee or not linkedin_companies:
        return 0.0
    return max(company_similarity(assignee, c) for c in linkedin_companies)
