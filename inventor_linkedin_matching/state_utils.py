"""
US state normalization and matching utilities.

Handles both full state names and two-letter abbreviations,
plus common city-to-state inference for when state is missing.
"""

from typing import Optional

# Full name -> abbreviation
_STATE_TO_ABBREV: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
    # DC and territories
    "district of columbia": "DC", "washington dc": "DC",
    "puerto rico": "PR", "guam": "GU", "virgin islands": "VI",
}

# Abbreviation -> full name (reverse mapping)
_ABBREV_TO_STATE: dict[str, str] = {v: k for k, v in _STATE_TO_ABBREV.items()}


def normalize_state(state: Optional[str]) -> Optional[str]:
    """
    Return the canonical two-letter state abbreviation (uppercase),
    or None if the input cannot be resolved.
    """
    if not state:
        return None
    s = state.strip().lower()
    # Already an abbreviation?
    if len(s) == 2:
        abbrev = s.upper()
        if abbrev in _ABBREV_TO_STATE:
            return abbrev
    # Full name lookup
    if s in _STATE_TO_ABBREV:
        return _STATE_TO_ABBREV[s]
    # Partial / prefix match (e.g. "Calif" -> CA) — try startswith on full names
    for full_name, abbrev in _STATE_TO_ABBREV.items():
        if full_name.startswith(s) and len(s) >= 4:
            return abbrev
    return None


def state_match_score(inventor_state: Optional[str], linkedin_state: Optional[str]) -> float:
    """
    Return a state match score:
      - 1.0  both states resolve and are equal
      - 0.0  both states resolve but differ, OR at least one is unresolvable
    """
    a = normalize_state(inventor_state)
    b = normalize_state(linkedin_state)
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0
