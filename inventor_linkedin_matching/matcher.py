"""
Core inventor-LinkedIn matching algorithm.

Design principles (inspired by USPTO InventorDisambiguator):
  - Three hard gates: last name, first name, and state must all pass.
  - Company similarity is the sole continuous ranking signal among survivors.
  - Configurable thresholds for each gate.
  - Returns ranked list of MatchResult objects.

Matching logic
--------------
  1. last_name similarity  >= last_name_threshold  (hard gate, fast exit)
  2. first_name similarity >= first_name_threshold (hard gate, fast exit)
  3. state must match exactly                      (hard gate, bypassed if either side is missing)
  4. score = company_similarity                    (ranking signal only)
  5. is_match = True when all gates pass AND score >= match_threshold
"""

from dataclasses import dataclass
from typing import Optional
import logging

from .models import InventorRecord, LinkedInProfile, MatchResult
from .name_utils import name_similarity, normalize_name
from .company_utils import best_company_match
from .state_utils import state_match_score

logger = logging.getLogger(__name__)


@dataclass
class MatchConfig:
    """
    Configuration for the InventorLinkedInMatcher.

    Gates (hard pass/fail)
    ----------------------
    last_name_threshold : minimum last-name similarity to proceed.
    first_name_threshold: minimum first-name similarity to proceed.
    state_threshold     : required state match score (binary 1.0/0.0).
                          Gate is bypassed when either side has no state data.

    Ranking
    -------
    Company similarity is the sole score used to rank survivors.
    match_threshold sets a minimum company score floor (default 0.0 = accept all
    gate-passing pairs regardless of company data).
    """

    # --- Gate thresholds ---
    last_name_threshold: float = 0.80   # hard gate: skip if LN too dissimilar
    first_name_threshold: float = 0.50  # hard gate: skip if FN too dissimilar
    state_threshold: float = 1.0        # hard gate: state must match exactly

    # --- Company score floor (ranking signal) ---
    match_threshold: float = 0.0        # minimum company score to classify as a match

    # --- Output ---
    top_k: Optional[int] = None         # Return only top-k results (None = all)


class InventorLinkedInMatcher:
    """
    Match a list of InventorRecord objects against a list of LinkedInProfile
    objects and return ranked MatchResult objects.

    Usage example
    -------------
    >>> matcher = InventorLinkedInMatcher()
    >>> results = matcher.match(inventors, profiles)
    >>> for r in results:
    ...     print(r)

    Or match a single inventor against all profiles:
    >>> results = matcher.match_one(inventor, profiles)
    """

    def __init__(self, config: Optional[MatchConfig] = None) -> None:
        self.config = config or MatchConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        inventors: list[InventorRecord],
        profiles: list[LinkedInProfile],
    ) -> list[MatchResult]:
        """
        Match all inventors against all LinkedIn profiles.

        Returns all MatchResult objects with is_match=True, sorted by score
        descending.  If config.top_k is set, returns at most top_k results
        per inventor.
        """
        results: list[MatchResult] = []
        for inventor in inventors:
            results.extend(self.match_one(inventor, profiles))
        # Global sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def match_one(
        self,
        inventor: InventorRecord,
        profiles: list[LinkedInProfile],
    ) -> list[MatchResult]:
        """
        Match a single inventor against all LinkedIn profiles.

        Returns MatchResult objects with is_match=True, sorted by score
        descending. If config.top_k is set, returns at most top_k results.
        """
        cfg = self.config
        results: list[MatchResult] = []

        for profile in profiles:
            result = self._score_pair(inventor, profile)
            if result.is_match:
                results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        if cfg.top_k is not None:
            results = results[: cfg.top_k]

        return results

    def score_pair(
        self,
        inventor: InventorRecord,
        profile: LinkedInProfile,
    ) -> MatchResult:
        """
        Compute and return a MatchResult for a specific (inventor, profile) pair,
        regardless of whether they exceed the match threshold.
        Useful for debugging / analysis.
        """
        return self._score_pair(inventor, profile)

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_pair(
        self,
        inventor: InventorRecord,
        profile: LinkedInProfile,
    ) -> MatchResult:
        """Compute a composite match score for an (inventor, profile) pair."""
        cfg = self.config

        # --- Last name (hard gate) ---
        ln_score = name_similarity(inventor.last_name, profile.last_name)
        if ln_score < cfg.last_name_threshold:
            # Fast exit: surnames too different, cannot be the same person
            return MatchResult(
                inventor=inventor,
                profile=profile,
                score=0.0,
                last_name_score=ln_score,
                is_match=False,
            )

        # --- First name (hard gate) ---
        fn_score = name_similarity(inventor.first_name, profile.first_name)
        if fn_score < cfg.first_name_threshold:
            return MatchResult(
                inventor=inventor,
                profile=profile,
                score=0.0,
                first_name_score=fn_score,
                last_name_score=ln_score,
                is_match=False,
            )

        # --- State (hard gate) ---
        # Bypass if either side has no state data (avoid penalising missing fields).
        st_score = state_match_score(inventor.state, profile.location_state)
        if inventor.state and profile.location_state and st_score < cfg.state_threshold:
            return MatchResult(
                inventor=inventor,
                profile=profile,
                score=0.0,
                first_name_score=fn_score,
                last_name_score=ln_score,
                state_score=st_score,
                is_match=False,
            )

        # --- Company (sole ranking signal) ---
        co_score = best_company_match(inventor.assignee, profile.all_companies())

        is_match = co_score >= cfg.match_threshold

        logger.debug(
            "inventor=%s | profile=%s | ln=%.3f fn=%.3f st=%.3f co=%.3f is_match=%s",
            inventor.inventor_id, profile.profile_id,
            ln_score, fn_score, st_score, co_score, is_match,
        )

        return MatchResult(
            inventor=inventor,
            profile=profile,
            score=co_score,
            first_name_score=fn_score,
            last_name_score=ln_score,
            state_score=st_score,
            company_score=co_score,
            is_match=is_match,
        )

    # ------------------------------------------------------------------
    # Batch / candidate-filtering helpers
    # ------------------------------------------------------------------

    def prefilter_by_last_name(
        self,
        inventor: InventorRecord,
        profiles: list[LinkedInProfile],
    ) -> list[LinkedInProfile]:
        """
        Fast pre-filter: return only profiles whose normalized last name
        starts with the same letter as the inventor's last name.

        This mirrors the sparse-matrix block strategy in InventorDisambiguator
        and dramatically reduces the O(I×P) comparison space.
        """
        ln = normalize_name(inventor.last_name)
        if not ln:
            return profiles
        initial = ln[0]
        return [p for p in profiles if normalize_name(p.last_name).startswith(initial)]

    def match_with_prefilter(
        self,
        inventors: list[InventorRecord],
        profiles: list[LinkedInProfile],
    ) -> list[MatchResult]:
        """
        Same as match() but applies last-name initial pre-filtering first,
        reducing comparisons by ~26× on average.
        """
        results: list[MatchResult] = []
        for inventor in inventors:
            candidates = self.prefilter_by_last_name(inventor, profiles)
            results.extend(self.match_one(inventor, candidates))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
