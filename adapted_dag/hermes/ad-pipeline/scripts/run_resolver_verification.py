#!/usr/bin/env python3
"""run_resolver_verification CLI — bandit-optimized citation verification.

    python3 scripts/run_resolver_verification.py --input references.json --output report.json

Loads a references list (JSON, each entry has doi/arxiv_id/title/authors/year/venue),
runs verification_gate.verify_citation() for each entry with a shared ResolverBandit,
and writes a structured report including per-citation outcomes, bandit stats, and
a summary table.

The bandit learns across citations: earlier results inform subsequent resolver
ordering, reducing API calls over time.

Integration target for:
  - academic-pipeline Stage 2.5 / 4.5 integrity verification (A0.5 step)
  - Ad-hoc citation verification during systematic review

Input JSON format (list of dicts):
  [
    {
      "citation_key": "smith2024",
      "title": "A Survey of RLHF",
      "authors": ["Smith, J.", "Lee, K."],
      "year": 2024,
      "venue": "NeurIPS 2024",
      "doi": "10.1234/example",
      "arxiv_id": null
    },
    ...
  ]

Output: JSON report with outcomes[], resolver_stats, summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from verification_gate import verify_citation, get_bandit
    from verification_gate import ALL_RESOLVERS as RESOLVER_NAMES
except ImportError:
    from scripts.verification_gate import verify_citation, get_bandit
    from scripts.verification_gate import ALL_RESOLVERS as RESOLVER_NAMES


def _build_clients() -> dict:
    """Construct the four production resolver clients."""
    try:
        from crossref_client import CrossrefClient
        from openalex_client import OpenAlexClient
        from arxiv_client import ArxivClient
        from semantic_scholar_client import SemanticScholarClient
    except ImportError:
        from scripts.crossref_client import CrossrefClient
        from scripts.openalex_client import OpenAlexClient
        from scripts.arxiv_client import ArxivClient
        from scripts.semantic_scholar_client import SemanticScholarClient
    return {
        "crossref": CrossrefClient(),
        "openalex": OpenAlexClient(),
        "semantic_scholar": SemanticScholarClient(),
        "arxiv": ArxivClient(),
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_resolver_verification",
        description="Bandit-optimized citation verification across 4 resolvers.",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input JSON file (list of citation entries).",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to write the JSON verification report.",
    )
    parser.add_argument(
        "--bandit-cache",
        default=None,
        help="Path to bandit state cache (persists learning across runs).",
    )
    parser.add_argument(
        "--no-bandit", action="store_true",
        help="Disable bandit optimization (run all 4 resolvers in parallel).",
    )
    parser.add_argument(
        "--synthetic-ref-slug", choices=["citation_key"], default="citation_key",
        help="Synthesize ref_slug from citation_key (required for standalone use).",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[ERROR] input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        entries = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[ERROR] invalid JSON in {input_path}: {e}", file=sys.stderr)
        return 1

    if not isinstance(entries, list):
        print("[ERROR] input must be a JSON list of citation entries", file=sys.stderr)
        return 1

    clients = _build_clients()
    bandit = None if args.no_bandit else get_bandit(cache_path=args.bandit_cache)

    outcomes = []
    total_start = time.monotonic()
    verified_count = 0
    not_found_count = 0
    unreachable_count = 0

    for entry in entries:
        citation_key = entry.get("citation_key", "unknown")
        ref_slug = citation_key if args.synthetic_ref_slug else citation_key
        try:
            result = verify_citation(
                entry, clients,
                ref_slug=ref_slug,
                bandit=bandit,
            )
        except Exception as e:
            result = {
                "citation_key": citation_key,
                "error": str(e),
                "lookup_verified": "ERROR",
            }

        lv = result.get("lookup_verified", "UNKNOWN")
        if lv == "VERIFIED":
            verified_count += 1
        elif lv in ("NOT_FOUND", "UNVERIFIED"):
            not_found_count += 1
        else:
            unreachable_count += 1

        summary = {
            "citation_key": citation_key,
            "lookup_verified": lv,
            "error": result.get("error"),
        }
        outcomes.append(summary)

    total_duration = time.monotonic() - total_start

    report = {
        "meta": {
            "total_citations": len(entries),
            "total_duration_seconds": round(total_duration, 3),
            "bandit_enabled": bandit is not None,
            "resolvers_available": list(RESOLVER_NAMES) if isinstance(RESOLVER_NAMES, (list, tuple)) else list(RESOLVER_NAMES),
        },
        "summary": {
            "verified": verified_count,
            "not_found": not_found_count,
            "unreachable_or_error": unreachable_count,
        },
        "outcomes": outcomes,
        "resolver_stats": bandit.get_stats_report() if bandit else None,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] {len(entries)} citations verified in {total_duration:.1f}s")
    print(f"     Verified: {verified_count}  Not found: {not_found_count}  Errors: {unreachable_count}")
    print(f"     Report written to {output_path}")

    if bandit:
        stats = bandit.get_stats_report()
        print("\n--- Resolver Bandit Stats ---")
        for name, s in stats.items():
            print(f"  {name:20s}  success={s['success_rate']:.1%}  calls={s['total_calls']}  avg_lat={s['avg_latency']:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(run())
