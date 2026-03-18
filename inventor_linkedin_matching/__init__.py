"""
Inventor-LinkedIn Matching Package

Matches patent inventor records with LinkedIn profiles based on
first name, last name, state, and company information.

Inspired by the USPTO InventorDisambiguator project:
https://github.com/PatentsView/InventorDisambiguator
"""

from .models import InventorRecord, LinkedInProfile, MatchResult
from .matcher import InventorLinkedInMatcher

__all__ = ["InventorRecord", "LinkedInProfile", "MatchResult", "InventorLinkedInMatcher"]
