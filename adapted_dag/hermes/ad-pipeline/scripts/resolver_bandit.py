#!/usr/bin/env python3
"""Contextual bandit for multi-resolver citation verification scheduling.

Optimizes resolver selection order based on citation features (DOI present,
arXiv ID present, title length, field, etc.) using an epsilon-greedy
multi-armed bandit. Tracks per-resolver success rate, latency, and cost
to dynamically choose the best resolver ordering per citation.

Usage:
    bandit = ResolverBandit()
    order = bandit.select_resolver_order(features={"has_doi": True, "has_arxiv_id": False})
    bandit.update("semantic_scholar", reward=1.0, latency=0.8, cost=0.01)
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RESOLVER_NAMES = ["semantic_scholar", "crossref", "openalex", "arxiv"]


@dataclass
class ResolverStats:
    success_count: int = 0
    total_count: int = 0
    total_latency: float = 0.0
    total_cost: float = 0.0
    last_latency: float = 0.0
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_count, 1)

    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(self.total_count, 1)


@dataclass
class CitationFeatures:
    has_doi: bool = False
    has_arxiv_id: bool = False
    title_length: int = 0
    has_venue: bool = False
    is_preprint: bool = False
    year: int = 0
    field: str = "unknown"

    def to_vector(self) -> list[float]:
        return [
            1.0 if self.has_doi else 0.0,
            1.0 if self.has_arxiv_id else 0.0,
            min(self.title_length / 200.0, 1.0),
            1.0 if self.has_venue else 0.0,
            1.0 if self.is_preprint else 0.0,
            min(max((self.year - 1900) / 200.0, 0.0), 1.0),
        ]

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> CitationFeatures:
        return cls(
            has_doi=bool(entry.get("doi")),
            has_arxiv_id=bool(entry.get("arxiv_id")),
            title_length=len(entry.get("title", "") or ""),
            has_venue=bool(entry.get("venue")),
            is_preprint=cls._is_preprint_venue(entry.get("venue", "")),
            year=entry.get("year") or 0,
            field=entry.get("field") or "unknown",
        )

    @staticmethod
    def _is_preprint_venue(venue: str) -> bool:
        return venue.lower() in {
            "arxiv", "biorxiv", "medrxiv", "ssrn", "research square",
            "preprints.org", "chemrxiv", "earthaarxiv", "osf preprints",
            "techrxiv",
        }


class ResolverBandit:
    DEFAULT_RESOLVER_ORDER = ["semantic_scholar", "crossref", "openalex", "arxiv"]

    FEATURE_WEIGHTS: dict[str, list[float]] = {
        "has_doi": [1.0, 0.8, 0.8, 0.3],
        "no_doi": [0.6, 0.3, 0.3, 0.5],
        "has_arxiv": [0.5, 0.3, 0.3, 1.0],
        "preprint": [0.7, 0.4, 0.4, 0.9],
        "default": [0.8, 0.6, 0.6, 0.4],
    }

    def __init__(self, epsilon: float = 0.15, gamma: float = 0.1,
                 cache_path: str | None = None):
        self.epsilon = epsilon
        self.gamma = gamma
        self.cache_path = cache_path
        self.stats: dict[str, ResolverStats] = {
            name: ResolverStats() for name in RESOLVER_NAMES
        }
        self.context_counts: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.context_rewards: defaultdict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._arm_order_weights: dict[str, list[float]] = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_path and Path(self.cache_path).exists():
            try:
                data = json.loads(Path(self.cache_path).read_text())
                for name, s in data.get("stats", {}).items():
                    if name in self.stats:
                        self.stats[name] = ResolverStats(**s)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    def _save_cache(self):
        if not self.cache_path:
            return
        data = {
            "stats": {
                name: {
                    "success_count": s.success_count,
                    "total_count": s.total_count,
                    "total_latency": s.total_latency,
                    "total_cost": s.total_cost,
                    "last_latency": s.last_latency,
                    "consecutive_failures": s.consecutive_failures,
                }
                for name, s in self.stats.items()
            },
        }
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cache_path).write_text(json.dumps(data, indent=2))

    def select_resolver_order(self, features: CitationFeatures | None = None,
                              top_k: int | None = None) -> list[str]:
        if features is None:
            order = list(self.DEFAULT_RESOLVER_ORDER)
            if top_k and top_k < len(order):
                order = order[:top_k]
            return order

        vec = features.to_vector()
        scores = self._compute_scores(vec, features)
        ranked = sorted(
            zip(RESOLVER_NAMES, scores),
            key=lambda x: -x[1],
        )
        ordered = [name for name, _ in ranked]

        if random.random() < self.epsilon:
            self._shuffle_tail(ordered, exploration_strength=2)

        if top_k and top_k < len(ordered):
            ordered = ordered[:top_k]

        return ordered

    def _compute_scores(self, vector: list[float],
                        features: CitationFeatures) -> list[float]:
        base_key = "default"
        if features.has_doi:
            base_key = "has_doi"
        elif features.has_arxiv_id:
            base_key = "has_arxiv"
        if features.is_preprint:
            base_key = "preprint"

        heuristic_weights = self.FEATURE_WEIGHTS.get(base_key, self.FEATURE_WEIGHTS["default"])

        scores = []
        for i, name in enumerate(RESOLVER_NAMES):
            stat = self.stats[name]
            exploration_bonus = math.sqrt(
                2 * math.log(max(sum(s.total_count for s in self.stats.values()), 1) + 1)
                / max(stat.total_count, 1)
            ) * 0.3

            ucb_score = stat.success_rate + exploration_bonus
            heuristic = heuristic_weights[i] if i < len(heuristic_weights) else 0.3
            context_score = self._contextual_score(features, name)

            combined = (
                0.5 * ucb_score +
                0.3 * heuristic +
                0.2 * context_score
            )
            scores.append(combined)

        return scores

    def _contextual_score(self, features: CitationFeatures,
                          resolver: str) -> float:
        context_key = self._context_key(features)
        n = self.context_counts[context_key].get(resolver, 0)
        if n == 0:
            return 0.5
        return self.context_rewards[context_key].get(resolver, 0.5) / n

    def _context_key(self, features: CitationFeatures) -> str:
        if features.has_doi:
            return "doi_present"
        if features.has_arxiv_id:
            return "arxiv_present"
        if features.is_preprint:
            return "preprint"
        return "title_only"

    @staticmethod
    def _shuffle_tail(items: list[str], exploration_strength: int = 2):
        if len(items) <= exploration_strength + 1:
            return
        tail = items[-exploration_strength:]
        random.shuffle(tail)
        items[-exploration_strength:] = tail

    def update(self, resolver: str, reward: float, latency: float = 0.0,
               cost: float = 0.0, features: CitationFeatures | None = None):
        if resolver not in self.stats:
            return
        stat = self.stats[resolver]
        stat.total_count += 1
        stat.total_latency += latency
        stat.total_cost += cost
        stat.last_latency = latency

        if reward > 0.5:
            stat.success_count += 1
            stat.consecutive_failures = 0
        else:
            stat.consecutive_failures += 1

        if features:
            ck = self._context_key(features)
            self.context_counts[ck][resolver] += 1
            self.context_rewards[ck][resolver] += reward

        if stat.total_count % 10 == 0:
            self._maybe_adjust_epsilon()
            self._save_cache()

    def _maybe_adjust_epsilon(self):
        max_rate = max(s.success_rate for s in self.stats.values())
        min_rate = min(s.success_rate for s in self.stats.values())
        gap = max_rate - min_rate
        if gap < 0.1:
            self.epsilon = min(self.epsilon + self.gamma, 0.5)
        elif gap > 0.4:
            self.epsilon = max(self.epsilon - self.gamma, 0.05)

    def update_batch(self, results: list[dict[str, Any]],
                     features_list: list[CitationFeatures] | None = None):
        for i, result in enumerate(results):
            resolver = result.get("resolver", "")
            reward = 1.0 if result.get("matched") else 0.0
            latency = result.get("latency", 0.0)
            cost = result.get("cost", 0.0)
            feat = features_list[i] if features_list else None
            self.update(resolver, reward, latency, cost, feat)

    def best_resolver(self, features: CitationFeatures | None = None) -> str:
        order = self.select_resolver_order(features, top_k=1)
        return order[0] if order else self.DEFAULT_RESOLVER_ORDER[0]

    def get_stats_report(self) -> dict[str, Any]:
        return {
            name: {
                "success_rate": round(stat.success_rate, 3),
                "total_calls": stat.total_count,
                "avg_latency": round(stat.avg_latency, 3),
                "total_cost": round(stat.total_cost, 4),
                "consecutive_failures": stat.consecutive_failures,
            }
            for name, stat in self.stats.items()
        }

    def adaptive_timeout(self, resolver: str, base_timeout: float = 30.0) -> float:
        stat = self.stats.get(resolver)
        if not stat or stat.total_count < 3:
            return base_timeout
        avg_lat = stat.avg_latency
        std = self._estimate_std(resolver)
        return min(base_timeout, max(avg_lat + 3 * std, 5.0))

    def _estimate_std(self, resolver: str) -> float:
        return self.stats[resolver].avg_latency * 0.5

    def adaptive_backoff(self, resolver: str, base_backoff: float = 2.0) -> float:
        stat = self.stats.get(resolver)
        if not stat or stat.consecutive_failures == 0:
            return base_backoff
        return min(base_backoff * (2 ** min(stat.consecutive_failures, 4)), 30.0)

    def should_skip_resolver(self, resolver: str, features: CitationFeatures) -> bool:
        if resolver == "arxiv" and not features.has_arxiv_id:
            return True
        stat = self.stats.get(resolver)
        if stat and stat.consecutive_failures >= 5:
            return True
        if resolver == "crossref" and not features.has_doi and features.title_length < 10:
            return True
        return False

    def select_resolvers(self, features: CitationFeatures) -> list[str]:
        order = self.select_resolver_order(features)
        return [r for r in order if not self.should_skip_resolver(r, features)]

    def verify_citation_bandit(self, entry: dict[str, Any],
                               clients: dict[str, Any],
                               cache=None) -> dict[str, Any]:
        features = CitationFeatures.from_entry(entry)
        resolver_order = self.select_resolvers(features)
        outcomes: dict[str, dict[str, Any]] = {}

        for resolver in resolver_order:
            client = clients.get(resolver)
            if not client:
                continue
            start = time.monotonic()
            matched = False
            try:
                if resolver == "semantic_scholar":
                    matched = bool(client.lookup(entry).get("matched", False))
                elif resolver == "arxiv":
                    if not entry.get("arxiv_id"):
                        outcomes[resolver] = {"status": "skipped"}
                        continue
                    matched = client.arxiv_id_lookup(
                        entry.get("arxiv_id"), entry.get("title", "")
                    ) is not None
                else:
                    hit = client.doi_lookup_with_title_check(
                        entry.get("doi"), entry.get("title", "")
                    )
                    if hit is None and entry.get("title"):
                        hit = client.title_search(entry.get("title"))
                    matched = hit is not None
                latency = time.monotonic() - start
                reward = 1.0 if matched else 0.0
                self.update(resolver, reward, latency, features=features)
                status = "matched" if matched else "unmatched"
            except Exception:
                latency = time.monotonic() - start
                self.update(resolver, 0.0, latency, features=features)
                status = "unreachable"

            outcomes[resolver] = {"status": status}
            if matched:
                break

        return {
            "outcomes": outcomes,
            "resolver_order_used": resolver_order,
        }
