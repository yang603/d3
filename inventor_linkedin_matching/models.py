"""
Data models for inventor records and LinkedIn profiles.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InventorRecord:
    """
    Represents a patent inventor record, sourced from PatentsView / USPTO data.

    Attributes mirror the 14-column TSV format used by InventorDisambiguator:
    patent_id, first_name, last_name, middle_name, assignee, city, state, class_id.
    """

    inventor_id: str                    # Unique identifier (e.g. "patent_number-sequence")
    first_name: str                     # Inventor first name
    last_name: str                      # Inventor last name
    middle_name: Optional[str] = None   # Middle name or initial
    assignee: Optional[str] = None      # Company/assignee on the patent
    city: Optional[str] = None          # City of record
    state: Optional[str] = None         # US state abbreviation or full name
    patent_id: Optional[str] = None     # Patent number
    class_id: Optional[str] = None      # USPTO class ID

    def full_name(self) -> str:
        parts = [self.first_name]
        if self.middle_name:
            parts.append(self.middle_name)
        parts.append(self.last_name)
        return " ".join(p for p in parts if p)


@dataclass
class LinkedInProfile:
    """
    Represents a LinkedIn profile record.
    """

    profile_id: str                         # Unique LinkedIn profile identifier
    first_name: str                         # First name on LinkedIn
    last_name: str                          # Last name on LinkedIn
    current_company: Optional[str] = None   # Current employer
    past_companies: list = field(default_factory=list)  # Past employers
    location_state: Optional[str] = None    # US state (abbreviated or full)
    location_city: Optional[str] = None     # City
    headline: Optional[str] = None          # LinkedIn headline/title

    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    def all_companies(self) -> list:
        """Return all companies (current + past) for matching."""
        companies = []
        if self.current_company:
            companies.append(self.current_company)
        companies.extend(self.past_companies)
        return companies


@dataclass
class MatchResult:
    """
    Represents a match between an InventorRecord and a LinkedInProfile,
    with a composite score and per-field breakdown.
    """

    inventor: InventorRecord
    profile: LinkedInProfile
    score: float                        # Composite match score in [0.0, 1.0]
    first_name_score: float = 0.0
    last_name_score: float = 0.0
    state_score: float = 0.0
    company_score: float = 0.0
    is_match: bool = False              # True if score >= threshold

    def __repr__(self) -> str:
        return (
            f"MatchResult(inventor_id={self.inventor.inventor_id!r}, "
            f"profile_id={self.profile.profile_id!r}, "
            f"score={self.score:.3f}, is_match={self.is_match})"
        )
