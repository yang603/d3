"""
Core inventor-LinkedIn matching algorithm.

Design principles (inspired by USPTO InventorDisambiguator):
  - Multi-attribute matching: first name, last name, state, company.
  - Two matching tiers:
      STRONG match  — last name exact/near-exact AND first name matches AND
                      (state OR company matches)
      WEAK match    — last name exact/near-exact AND first name initial matches AND
                      state AND company both match
  - Configurable per-field weights and thresholds.
  - Returns ranked list of MatchResult objects.

Scoring formula
---------------
  score = w_ln * ln_score
        + w_fn * fn_score
        + w_state * state_score
        + w_company * company_score

Weights sum to 1.0.  Default weights reflect the relative discriminative
power of each field observed in the InventorDisambiguator codebase
(last name > first name > company ≈ state).
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

    Field weights
    -------------
    Must sum to 1.0.  Adjust to reflect your data quality:
      - Increase w_company if patent assignees are reliable.
      - Decrease w_state if LinkedIn locations are often missing.

    Thresholds
    ----------
    match_threshold     : minimum composite score to be considered a match.
    last_name_threshold : minimum last-name score to proceed (hard gate).
    first_name_threshold: minimum first-name score to proceed (hard gate).
    """

    # --- Field weights (must sum to 1.0) ---
    w_last_name: float = 0.40
    w_first_name: float = 0.30
    w_state: float = 0.15
    w_company: float = 0.15

    # --- Score thresholds ---
    match_threshold: float = 0.70       # minimum to classify as a match
    last_name_threshold: float = 0.80   # hard gate: skip if LN too dissimilar
    first_name_threshold: float = 0.50  # hard gate: skip if FN too dissimilar

    # --- Output ---
    top_k: Optional[int] = None         # Return only top-k results (None = all)

    def __post_init__(self) -> None:
        total = self.w_last_name + self.w_first_name + self.w_state + self.w_company
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Field weights must sum to 1.0, got {total:.4f}. "
                "Adjust w_last_name, w_first_name, w_state, w_company."
            )


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

        # --- State ---
        st_score = state_match_score(inventor.state, profile.location_state)

        # --- Company ---
        co_score = best_company_match(inventor.assignee, profile.all_companies())

        # --- Composite score ---
        composite = (
            cfg.w_last_name * ln_score
            + cfg.w_first_name * fn_score
            + cfg.w_state * st_score
            + cfg.w_company * co_score
        )

        # Apply two-tier matching logic (mirroring InventorDisambiguator):
        # STRONG: full name match + at least one corroborating field
        # WEAK:   first-initial + state + company both match
        strong_match = (
            ln_score >= 0.92
            and fn_score >= 0.80
            and (st_score >= 1.0 or co_score >= 0.70)
        )
        weak_match = (
            ln_score >= 0.80
            and fn_score >= 0.50  # allows initial match
            and st_score >= 1.0
            and co_score >= 0.60
        )

        is_match = composite >= cfg.match_threshold or strong_match or weak_match

        logger.debug(
            "inventor=%s | profile=%s | ln=%.3f fn=%.3f st=%.3f co=%.3f "
            "composite=%.3f strong=%s weak=%s is_match=%s",
            inventor.inventor_id, profile.profile_id,
            ln_score, fn_score, st_score, co_score,
            composite, strong_match, weak_match, is_match,
        )

        return MatchResult(
            inventor=inventor,
            profile=profile,
            score=composite,
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
