#!/usr/bin/env python3
"""verification_gate — citation existence verification API (Delta 5).

Public functions:
  - verify_citation(entry, clients, *, ref_slug, anchor=None, cache=None, bandit=None)
        -> CitationVerificationOutcome
  - verify_passport(passport, clients, *, ref_slug_by_key, anchors=None,
        cache=None, bandit=None) -> list[outcome]

Composes the four resolvers (crossref / openalex / semantic_scholar / arxiv),
maps each resolver's execution to a {status, queried_by} outcome, derives the
3-class lookup_verified via the Delta 4 reducer (narrowed-false, C-V6(a)),
reads anchor_present from the v3.7.3 anchor marker, and stamps
verification_timestamp.

When a `bandit` is provided (ResolverBandit instance), the resolver execution
order is optimized by the bandit's contextual multi-armed bandit algorithm:
resolvers are tried sequentially in predicted-best order, and the search stops
on the first match. Unattempted resolvers are reported as "skipped". When no
bandit is provided, all four resolvers run in parallel (legacy behavior).

Spec: docs/design/2026-05-21-v3.10-182-promote-citation-gate-spec.md §2 Delta 5.
Bandit integration: scripts/resolver_bandit.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

try:
    from citation_verification_summary import (
        STATUS_MATCHED,
        STATUS_SKIPPED,
        STATUS_UNMATCHED,
        STATUS_UNREACHABLE,
        reduce_lookup_verified,
    )
    from crossref_client import CrossrefUnavailable
    from openalex_client import OpenAlexUnavailable
    from arxiv_client import ArxivUnavailable
    from contamination_signals import (
        SemanticScholarUnavailable,
        _resolve_arxiv_id_then_title,
        _resolve_doi_then_title,
        queried_by_for,
    )
    from resolver_bandit import ResolverBandit, CitationFeatures
except ImportError:  # pragma: no cover - dual-path import
    from scripts.citation_verification_summary import (
        STATUS_MATCHED,
        STATUS_SKIPPED,
        STATUS_UNMATCHED,
        STATUS_UNREACHABLE,
        reduce_lookup_verified,
    )
    from scripts.crossref_client import CrossrefUnavailable
    from scripts.openalex_client import OpenAlexUnavailable
    from scripts.arxiv_client import ArxivUnavailable
    from scripts.contamination_signals import (
        SemanticScholarUnavailable,
        _resolve_arxiv_id_then_title,
        _resolve_doi_then_title,
        queried_by_for,
    )
    from scripts.resolver_bandit import ResolverBandit, CitationFeatures

_ANCHOR_PRESENT_KINDS = frozenset({"quote", "page", "section", "paragraph"})

_bandit: ResolverBandit | None = None

ALL_RESOLVERS = ("crossref", "openalex", "semantic_scholar", "arxiv")


def get_bandit(cache_path: str | None = None) -> ResolverBandit:
    """Return the shared module-level ResolverBandit singleton.

    The bandit accumulates statistics across calls, learning which resolvers
    perform best for each citation profile. Persisted to disk via cache_path.
    """
    global _bandit
    if _bandit is None:
        _bandit = ResolverBandit(cache_path=cache_path)
    return _bandit


def reset_bandit():
    """Reset the shared bandit singleton (for testing or fresh start)."""
    global _bandit
    _bandit = None


def _is_valid_ref_slug(ref_slug: Any) -> bool:
    """A ref_slug is valid iff it is a non-empty string: the summary schema
    requires ref_slug as a string, and an empty slug joins to no
    <!--ref:slug--> prose marker. Single definition so verify_citation (the
    emission point) and verify_passport (the join layer) agree on "bad slug"
    (#332)."""
    return isinstance(ref_slug, str) and bool(ref_slug)


def _outcome(status: str, queried_by: str | None,
             response_summary: str | None = None) -> dict[str, Any]:
    return {"status": status, "queried_by": queried_by,
            "response_summary": response_summary}


def _ran_outcome(unmatched: bool, queried_by: str | None) -> dict[str, Any]:
    """Map a ran-resolver result (matched/unmatched) to an outcome dict."""
    return _outcome(STATUS_UNMATCHED if unmatched else STATUS_MATCHED, queried_by)


def _run_doi_then_title(entry, client, unavailable_exc) -> dict[str, Any]:
    """Run a doi-then-title resolver, mapping execution to a status outcome.
    Used by crossref / openalex (same flow + DOI key, different exception).
    The manual exemption is short-circuited upstream in verify_citation."""
    try:
        unmatched, _matched_by, queried_by = _resolve_doi_then_title(entry, client)
    except unavailable_exc:
        return _outcome(STATUS_UNREACHABLE, None)
    return _ran_outcome(unmatched, queried_by)


def _run_semantic_scholar(entry, client) -> dict[str, Any]:
    """S2's lookup(entry) is a single entry-keyed call (DOI-first then title
    internally). queried_by follows the has-an-id rule (C-V6(a)). The manual
    exemption is short-circuited upstream in verify_citation."""
    queried_by = queried_by_for(entry, id_field="doi")
    try:
        matched = bool(client.lookup(entry).get("matched", False))
    except SemanticScholarUnavailable:
        return _outcome(STATUS_UNREACHABLE, None)
    return _ran_outcome(not matched, queried_by)


def _run_arxiv(entry, client) -> dict[str, Any]:
    """arXiv resolver is applicable only when the citation has an arXiv ID;
    otherwise it is skipped (not unmatched) per Delta 1 / spec line 119. The
    manual exemption is short-circuited upstream in verify_citation."""
    if not entry.get("arxiv_id"):
        return _outcome(STATUS_SKIPPED, None)  # non-arXiv citation → not applicable
    try:
        unmatched, _matched_by, queried_by = _resolve_arxiv_id_then_title(entry, client)
    except ArxivUnavailable:
        return _outcome(STATUS_UNREACHABLE, None)
    return _ran_outcome(unmatched, queried_by)


def _run_single_resolver(resolver: str, entry, clients) -> dict[str, Any]:
    """Dispatch one resolver by name, returning the outcome dict."""
    client = clients.get(resolver)
    if not client:
        return _outcome(STATUS_UNREACHABLE, None)
    if resolver == "crossref":
        return _run_doi_then_title(entry, client, CrossrefUnavailable)
    elif resolver == "openalex":
        return _run_doi_then_title(entry, client, OpenAlexUnavailable)
    elif resolver == "semantic_scholar":
        return _run_semantic_scholar(entry, client)
    elif resolver == "arxiv":
        return _run_arxiv(entry, client)
    return _outcome(STATUS_UNREACHABLE, None)


def _anchor_present(anchor: Any) -> bool:
    """True iff the v3.7.3 anchor marker has kind ∈ {quote,page,section,paragraph}
    (not none). `anchor` is the already-parsed {kind, value} marker sourced from
    writer prose and joined by ref_slug upstream — NEVER read off the corpus
    entry (the literature_corpus_entry schema has no anchor field; reading it
    there would be a permanent silent False)."""
    if not isinstance(anchor, Mapping):
        return False
    return anchor.get("kind") in _ANCHOR_PRESENT_KINDS


def _run_bandit_optimized(entry, clients, bandit) -> dict[str, str]:
    """Run resolvers in bandit-optimized order, stopping at first match.

    Returns a {resolver_name: outcome_dict} mapping with all four resolvers
    represented. Resolvers not attempted due to an early match are marked
    STATUS_SKIPPED so the summary schema is always complete.
    """
    features = CitationFeatures.from_entry(dict(entry))
    resolver_order = bandit.select_resolvers(features)
    resolver_outcomes: dict[str, Any] = {}

    for resolver in resolver_order:
        outcome = _run_single_resolver(resolver, entry, clients)
        resolver_outcomes[resolver] = outcome
        matched = outcome.get("status") == STATUS_MATCHED
        bandit.update(resolver, reward=1.0 if matched else 0.0, features=features)
        if matched:
            break

    for r in ALL_RESOLVERS:
        resolver_outcomes.setdefault(r, _outcome(STATUS_SKIPPED, None))
    return resolver_outcomes


def verify_citation(
    entry: Mapping[str, Any],
    clients: Mapping[str, Any],
    *,
    ref_slug: str,
    anchor: Mapping[str, Any] | None = None,
    cache=None,
    bandit: ResolverBandit | None = None,
) -> dict[str, Any]:
    """Verify one citation's existence across the four resolvers.

    `entry` carries citation_key, title, authors, year, source_pointer, optional
    doi / arxiv_id, obtained_via. `clients` is a mapping {crossref, openalex,
    semantic_scholar, arxiv} of resolver clients (injected so callers control
    network / cache).

    `ref_slug` is the writer-prose `<!--ref:slug-->` marker this citation renders
    under, supplied by the caller — never read off the corpus entry (same
    provenance rule as `anchor`; the entry schema forbids the field, #332).

    `anchor` is the v3.7.3 anchor marker ({kind, value}) for this citation's
    ref_slug, already parsed from writer prose and joined upstream (None when no
    anchor marker exists for the ref_slug).

    When `bandit` is provided, the resolver execution order is optimized by
    the bandit's contextual multi-armed bandit algorithm: resolvers are tried
    sequentially in predicted-best order, stopping at the first match. Without
    a bandit, all four resolvers run in parallel (legacy behavior).

    Returns a dict validating against citation_verification_summary.schema.json:
    {citation_key, ref_slug, lookup_verified, anchor_present,
     verification_timestamp, resolver_outcomes}.
    """
    if cache is not None:
        raise NotImplementedError(
            "cache-through at the verification_gate layer is not yet wired "
            "(#182 Delta-2 follow-up); pass cache=None"
        )
    if not _is_valid_ref_slug(ref_slug):
        raise ValueError(
            f"ref_slug must be a non-empty string (the writer-prose join key), "
            f"got {ref_slug!r}; corpus entries do not carry ref_slug (#332)"
        )
    if entry.get("obtained_via") == "manual":
        resolver_outcomes = {
            r: _outcome(STATUS_SKIPPED, None)
            for r in ALL_RESOLVERS
        }
    elif bandit is not None:
        resolver_outcomes = _run_bandit_optimized(entry, clients, bandit)
    else:
        resolver_outcomes = {
            "crossref": _run_doi_then_title(
                entry, clients["crossref"], CrossrefUnavailable),
            "openalex": _run_doi_then_title(
                entry, clients["openalex"], OpenAlexUnavailable),
            "semantic_scholar": _run_semantic_scholar(
                entry, clients["semantic_scholar"]),
            "arxiv": _run_arxiv(entry, clients["arxiv"]),
        }
    return {
        "citation_key": entry.get("citation_key"),
        "ref_slug": ref_slug,
        "lookup_verified": reduce_lookup_verified(resolver_outcomes),
        "anchor_present": _anchor_present(anchor),
        "verification_timestamp": datetime.now(timezone.utc).isoformat(),
        "resolver_outcomes": resolver_outcomes,
    }


def verify_passport(
    passport: Mapping[str, Any],
    clients: Mapping[str, Any],
    *,
    ref_slug_by_key: Mapping[str, str],
    anchors: Mapping[str, Mapping[str, Any]] | None = None,
    cache=None,
    bandit: ResolverBandit | None = None,
) -> list[dict[str, Any]]:
    """Batch helper: run verify_citation over every entry in the passport's
    literature_corpus[].

    `ref_slug_by_key` is the {citation_key: ref_slug} join map.
    `anchors` is the {ref_slug: anchor-marker} join map.

    When `bandit` is provided, each citation is verified using bandit-optimized
    sequential resolver selection (early break on match). Shared bandit instance
    learns from all citations across the batch.
    """
    corpus = passport.get("literature_corpus") or []
    anchors = anchors or {}
    outcomes: list[dict[str, Any]] = []
    for entry in corpus:
        citation_key = entry.get("citation_key")
        ref_slug = ref_slug_by_key.get(citation_key)
        if not _is_valid_ref_slug(ref_slug):
            raise ValueError(
                f"no valid ref_slug joined for citation_key {citation_key!r} "
                f"(got {ref_slug!r}): the citation_key→ref_slug prose join must "
                "cover every corpus entry with a non-empty string "
                "(corpus entries do not carry ref_slug; #332)"
            )
        outcomes.append(verify_citation(
            entry, clients, ref_slug=ref_slug,
            anchor=anchors.get(ref_slug), cache=cache, bandit=bandit))
    return outcomes
