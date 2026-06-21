#!/usr/bin/env python3
"""Tests for resolver_bandit module."""
from __future__ import annotations

import sys
import unittest

try:
    from resolver_bandit import (
        ResolverBandit, CitationFeatures, ResolverStats,
        RESOLVER_NAMES,
    )
except ImportError:
    from scripts.resolver_bandit import (
        ResolverBandit, CitationFeatures, ResolverStats,
        RESOLVER_NAMES,
    )


class TestCitationFeatures(unittest.TestCase):
    def test_from_entry_with_doi(self):
        features = CitationFeatures.from_entry({"doi": "10.1234/test", "title": "Test"})
        self.assertTrue(features.has_doi)
        self.assertFalse(features.has_arxiv_id)

    def test_from_entry_with_arxiv(self):
        features = CitationFeatures.from_entry({"arxiv_id": "2301.12345"})
        self.assertTrue(features.has_arxiv_id)

    def test_from_entry_preprint_venue(self):
        features = CitationFeatures.from_entry({"venue": "arXiv", "title": "Test"})
        self.assertTrue(features.is_preprint)

    def test_from_entry_regular_venue(self):
        features = CitationFeatures.from_entry({"venue": "Nature", "title": "Test"})
        self.assertFalse(features.is_preprint)

    def test_vector_length(self):
        features = CitationFeatures(has_doi=True, title_length=100)
        vec = features.to_vector()
        self.assertEqual(len(vec), 6)

    def test_vector_doi_present(self):
        features = CitationFeatures(has_doi=True)
        self.assertEqual(features.to_vector()[0], 1.0)

    def test_vector_doi_absent(self):
        features = CitationFeatures(has_doi=False)
        self.assertEqual(features.to_vector()[0], 0.0)

    def test_vector_title_length_clamped(self):
        features = CitationFeatures(title_length=500)
        self.assertEqual(features.to_vector()[2], 1.0)

    def test_vector_year_normalized(self):
        features = CitationFeatures(year=2024)
        vec = features.to_vector()
        self.assertAlmostEqual(vec[5], (2024 - 1900) / 200.0)

    def test_from_entry_missing_fields(self):
        features = CitationFeatures.from_entry({})
        self.assertFalse(features.has_doi)
        self.assertEqual(features.field, "unknown")

    def test_is_preprint_venue_case_insensitive(self):
        self.assertTrue(CitationFeatures._is_preprint_venue("arxiv"))
        self.assertTrue(CitationFeatures._is_preprint_venue("arXiv"))
        self.assertTrue(CitationFeatures._is_preprint_venue("bioRxiv"))


class TestResolverStats(unittest.TestCase):
    def test_success_rate_default(self):
        stats = ResolverStats()
        self.assertEqual(stats.success_rate, 0.0)

    def test_success_rate_computed(self):
        stats = ResolverStats(success_count=3, total_count=4)
        self.assertEqual(stats.success_rate, 0.75)

    def test_avg_latency_default(self):
        stats = ResolverStats()
        self.assertEqual(stats.avg_latency, 0.0)

    def test_avg_latency_computed(self):
        stats = ResolverStats(total_latency=5.0, total_count=2)
        self.assertEqual(stats.avg_latency, 2.5)

    def test_consecutive_failures(self):
        stats = ResolverStats(consecutive_failures=3)
        self.assertEqual(stats.consecutive_failures, 3)


class TestResolverBanditInitialization(unittest.TestCase):
    def test_default_resolver_names(self):
        self.assertEqual(RESOLVER_NAMES, ["semantic_scholar", "crossref", "openalex", "arxiv"])

    def test_all_resolvers_have_stats(self):
        bandit = ResolverBandit()
        self.assertEqual(set(bandit.stats.keys()), set(RESOLVER_NAMES))

    def test_default_epsilon(self):
        bandit = ResolverBandit()
        self.assertEqual(bandit.epsilon, 0.15)

    def test_custom_epsilon(self):
        bandit = ResolverBandit(epsilon=0.3)
        self.assertEqual(bandit.epsilon, 0.3)


class TestResolverBanditSelection(unittest.TestCase):
    def setUp(self):
        self.bandit = ResolverBandit()

    def test_select_order_returns_four_resolvers(self):
        order = self.bandit.select_resolver_order()
        self.assertEqual(len(order), 4)

    def test_select_order_all_resolvers_present(self):
        order = self.bandit.select_resolver_order()
        self.assertEqual(set(order), set(RESOLVER_NAMES))

    def test_select_order_with_doi_features(self):
        features = CitationFeatures(has_doi=True)
        order = self.bandit.select_resolver_order(features)
        self.assertEqual(len(order), 4)

    def test_select_order_with_arxiv_features(self):
        features = CitationFeatures(has_arxiv_id=True)
        order = self.bandit.select_resolver_order(features)
        self.assertEqual(len(order), 4)

    def test_select_order_with_top_k(self):
        order = self.bandit.select_resolver_order(top_k=2)
        self.assertEqual(len(order), 2)

    def test_best_resolver_returns_string(self):
        best = self.bandit.best_resolver()
        self.assertIn(best, RESOLVER_NAMES)

    def test_best_resolver_with_features(self):
        features = CitationFeatures(has_doi=True)
        best = self.bandit.best_resolver(features)
        self.assertIn(best, RESOLVER_NAMES)


class TestResolverBanditUpdates(unittest.TestCase):
    def setUp(self):
        self.bandit = ResolverBandit()

    def test_update_increases_count(self):
        initial = self.bandit.stats["semantic_scholar"].total_count
        self.bandit.update("semantic_scholar", reward=1.0, latency=0.5)
        self.assertEqual(self.bandit.stats["semantic_scholar"].total_count, initial + 1)

    def test_update_success_increments_success_count(self):
        self.bandit.update("crossref", reward=1.0)
        self.assertEqual(self.bandit.stats["crossref"].success_count, 1)

    def test_update_failure_does_not_increment_success(self):
        self.bandit.update("openalex", reward=0.0)
        self.assertEqual(self.bandit.stats["openalex"].success_count, 0)

    def test_update_tracks_latency(self):
        self.bandit.update("arxiv", reward=1.0, latency=2.5)
        self.assertEqual(self.bandit.stats["arxiv"].total_latency, 2.5)

    def test_update_tracks_cost(self):
        self.bandit.update("semantic_scholar", reward=0.0, cost=0.05)
        self.assertEqual(self.bandit.stats["semantic_scholar"].total_cost, 0.05)

    def test_update_consecutive_failures_reset_on_success(self):
        self.bandit.update("crossref", reward=0.0)
        self.bandit.update("crossref", reward=0.0)
        self.bandit.update("crossref", reward=1.0)
        self.assertEqual(self.bandit.stats["crossref"].consecutive_failures, 0)

    def test_update_consecutive_failures_accumulate(self):
        self.bandit.update("arxiv", reward=0.0)
        self.bandit.update("arxiv", reward=0.0)
        self.bandit.update("arxiv", reward=0.0)
        self.assertEqual(self.bandit.stats["arxiv"].consecutive_failures, 3)

    def test_update_invalid_resolver_noop(self):
        initial = dict(self.bandit.stats)
        self.bandit.update("nonexistent", reward=1.0)
        self.assertEqual(dict(self.bandit.stats), initial)

    def test_update_with_features(self):
        features = CitationFeatures(has_doi=True)
        self.bandit.update("semantic_scholar", reward=1.0, features=features)
        self.assertIn("doi_present", self.bandit.context_counts)

    def test_update_batch_multiple(self):
        results = [
            {"resolver": "semantic_scholar", "matched": True, "latency": 0.3, "cost": 0.01},
            {"resolver": "crossref", "matched": False, "latency": 1.2, "cost": 0.02},
        ]
        self.bandit.update_batch(results)
        self.assertEqual(self.bandit.stats["semantic_scholar"].total_count, 1)
        self.assertEqual(self.bandit.stats["crossref"].total_count, 1)


class TestResolverBanditAdaptive(unittest.TestCase):
    def setUp(self):
        self.bandit = ResolverBandit()

    def test_adaptive_timeout_no_history(self):
        timeout = self.bandit.adaptive_timeout("semantic_scholar")
        self.assertEqual(timeout, 30.0)

    def test_adaptive_timeout_with_history(self):
        for _ in range(5):
            self.bandit.update("semantic_scholar", reward=1.0, latency=2.0)
        timeout = self.bandit.adaptive_timeout("semantic_scholar")
        self.assertLess(timeout, 30.0)

    def test_adaptive_backoff_no_failures(self):
        backoff = self.bandit.adaptive_backoff("crossref")
        self.assertEqual(backoff, 2.0)

    def test_adaptive_backoff_with_failures(self):
        for _ in range(3):
            self.bandit.update("crossref", reward=0.0)
        backoff = self.bandit.adaptive_backoff("crossref")
        self.assertGreater(backoff, 2.0)

    def test_adaptive_backoff_capped(self):
        for _ in range(10):
            self.bandit.update("arxiv", reward=0.0)
        backoff = self.bandit.adaptive_backoff("arxiv")
        self.assertLessEqual(backoff, 30.0)

    def test_should_skip_arxiv_without_id(self):
        features = CitationFeatures(has_arxiv_id=False)
        self.assertTrue(self.bandit.should_skip_resolver("arxiv", features))

    def test_should_not_skip_arxiv_with_id(self):
        features = CitationFeatures(has_arxiv_id=True)
        self.assertFalse(self.bandit.should_skip_resolver("arxiv", features))

    def test_should_skip_after_consecutive_failures(self):
        features = CitationFeatures(has_doi=True)
        for _ in range(5):
            self.bandit.update("crossref", reward=0.0)
        self.assertTrue(self.bandit.should_skip_resolver("crossref", features))

    def test_select_resolvers_filters_skipped(self):
        features = CitationFeatures(has_arxiv_id=False)
        selected = self.bandit.select_resolvers(features)
        self.assertNotIn("arxiv", selected)


class TestResolverBanditStats(unittest.TestCase):
    def test_get_stats_report_keys(self):
        self.bandit = ResolverBandit()
        report = self.bandit.get_stats_report()
        self.assertEqual(set(report.keys()), set(RESOLVER_NAMES))

    def test_get_stats_report_fields(self):
        self.bandit = ResolverBandit()
        self.bandit.update("semantic_scholar", reward=1.0, latency=1.5, cost=0.01)
        report = self.bandit.get_stats_report()
        ss = report["semantic_scholar"]
        self.assertIn("success_rate", ss)
        self.assertIn("avg_latency", ss)
        self.assertIn("total_calls", ss)
        self.assertEqual(ss["total_calls"], 1)


class TestResolverBanditContextual(unittest.TestCase):
    def test_context_key_doi(self):
        features = CitationFeatures(has_doi=True)
        bandit = ResolverBandit()
        key = bandit._context_key(features)
        self.assertEqual(key, "doi_present")

    def test_context_key_arxiv(self):
        features = CitationFeatures(has_arxiv_id=True, has_doi=False)
        bandit = ResolverBandit()
        key = bandit._context_key(features)
        self.assertEqual(key, "arxiv_present")

    def test_context_key_title_only(self):
        features = CitationFeatures(has_doi=False, has_arxiv_id=False)
        bandit = ResolverBandit()
        key = bandit._context_key(features)
        self.assertEqual(key, "title_only")


class TestResolverBanditVerifyCitation(unittest.TestCase):
    class MockClient:
        def __init__(self, should_match=False):
            self.should_match = should_match

        def lookup(self, entry):
            return {"matched": self.should_match}

        def doi_lookup_with_title_check(self, doi, title):
            return {"doi": doi} if self.should_match else None

        def title_search(self, title):
            return {"title": title} if self.should_match else None

        def arxiv_id_lookup(self, arxiv_id, title):
            return {"arxiv_id": arxiv_id} if self.should_match else None

    def test_verify_citation_bandit_matched(self):
        bandit = ResolverBandit()
        entry = {"doi": "10.1234/test", "title": "Test Paper", "arxiv_id": None}
        clients = {
            "semantic_scholar": self.MockClient(should_match=True),
            "crossref": self.MockClient(should_match=False),
            "openalex": self.MockClient(should_match=False),
            "arxiv": self.MockClient(should_match=False),
        }
        result = bandit.verify_citation_bandit(entry, clients)
        self.assertIn("outcomes", result)
        self.assertIn("resolver_order_used", result)

    def test_verify_citation_no_match(self):
        bandit = ResolverBandit()
        entry = {"title": "Obscure Paper", "doi": None, "arxiv_id": None}
        clients = {
            "semantic_scholar": self.MockClient(should_match=False),
            "crossref": self.MockClient(should_match=False),
            "openalex": self.MockClient(should_match=False),
            "arxiv": self.MockClient(should_match=False),
        }
        result = bandit.verify_citation_bandit(entry, clients)
        self.assertIn("outcomes", result)

    def test_verify_citation_stops_on_match(self):
        bandit = ResolverBandit()
        entry = {"doi": "10.1234/test", "title": "Test"}
        clients = {
            "semantic_scholar": self.MockClient(should_match=False),
            "crossref": self.MockClient(should_match=True),
            "openalex": self.MockClient(should_match=False),
            "arxiv": self.MockClient(should_match=False),
        }
        result = bandit.verify_citation_bandit(entry, clients)
        self.assertIn("resolver_order_used", result)

    def test_verify_citation_handles_error(self):
        class ErrorClient:
            @staticmethod
            def lookup(entry):
                raise ConnectionError("API down")

        bandit = ResolverBandit()
        entry = {"doi": "10.1234/test", "title": "Test", "arxiv_id": None}
        clients = {
            "semantic_scholar": ErrorClient(),
            "crossref": ErrorClient(),
            "openalex": ErrorClient(),
            "arxiv": ErrorClient(),
        }
        result = bandit.verify_citation_bandit(entry, clients)
        self.assertIn("outcomes", result)


class TestResolverBanditEpsilon(unittest.TestCase):
    def test_epsilon_adjustment_converged(self):
        bandit = ResolverBandit(epsilon=0.5, gamma=0.1)
        for _ in range(100):
            for name in RESOLVER_NAMES:
                bandit.update(name, reward=0.9, latency=0.1)
        self.assertLessEqual(bandit.epsilon, 0.5)

    def test_epsilon_adjustment_diverged(self):
        bandit = ResolverBandit(epsilon=0.05, gamma=0.1)
        for _ in range(50):
            bandit.update("semantic_scholar", reward=0.9, latency=0.1)
            for name in RESOLVER_NAMES[1:]:
                bandit.update(name, reward=0.3, latency=0.5)
        self.assertGreaterEqual(bandit.epsilon, 0.05)


class TestResolverBanditDeterministic(unittest.TestCase):
    def test_feature_weights_structure(self):
        for key, weights in ResolverBandit.FEATURE_WEIGHTS.items():
            self.assertEqual(len(weights), 4, f"{key} has {len(weights)} weights, expected 4")

    def test_default_order_is_consistent(self):
        self.assertEqual(
            ResolverBandit.DEFAULT_RESOLVER_ORDER,
            ["semantic_scholar", "crossref", "openalex", "arxiv"],
        )


if __name__ == "__main__":
    unittest.main()
