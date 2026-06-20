#!/usr/bin/env python3
"""Tests for test_selector module."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest

try:
    from test_selector import TestSelector, SelectedTests, TestMapping
except ImportError:
    from scripts.test_selector import TestSelector, SelectedTests, TestMapping


class TestSelectedTests(unittest.TestCase):
    def test_default_values(self):
        s = SelectedTests()
        self.assertEqual(s.selected, [])
        self.assertEqual(s.total_available, 0)
        self.assertEqual(s.selection_ratio, 1.0)

    def test_with_values(self):
        s = SelectedTests(
            selected=["a", "b"],
            skipped_from_full=["c"],
            total_available=3,
            selection_ratio=0.667,
            reason="test",
        )
        self.assertEqual(len(s.selected), 2)
        self.assertEqual(len(s.skipped_from_full), 1)

    def test_selection_ratio_zero(self):
        s = SelectedTests(total_available=0, selection_ratio=0.0)
        self.assertEqual(s.selection_ratio, 0.0)


class TestSelectorWithFixtures(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        scripts_dir = self.tmpdir / "scripts"
        scripts_dir.mkdir(parents=True)

        self.test_files = [
            "test_pipeline_dag.py",
            "test_resolver_bandit.py",
            "test_text_similarity.py",
            "test_contamination_signals.py",
            "test_verification_gate.py",
            "test_block_parser.py",
        ]
        for tf in self.test_files:
            (scripts_dir / tf).write_text(
                "import unittest\n"
            )

        self.selector = TestSelector(
            repo_root=str(self.tmpdir),
            manifest_path=str(scripts_dir / "_ci_pytest_manifest.toml"),
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.tmpdir))


class TestSelectorCore(unittest.TestCase):
    def test_import(self):
        from test_selector import TestSelector
        self.assertIsNotNone(TestSelector)

    def test_explicit_mapping_contains_expected_entries(self):
        selector = TestSelector()
        self.assertIn("scripts/test_text_similarity.py", selector._explicit_mappings)
        self.assertIn("scripts/test_contamination_signals.py", selector._explicit_mappings)
        self.assertIn("scripts/test_pipeline_dag.py", selector._explicit_mappings)

    def test_pipeline_dag_mapping(self):
        selector = TestSelector()
        mapping = selector._explicit_mappings.get("scripts/test_pipeline_dag.py", set())
        self.assertIn("scripts/pipeline_dag.py", mapping)
        self.assertIn("scripts/pipeline_scheduler.py", mapping)

    def test_resolver_bandit_mapping(self):
        selector = TestSelector()
        mapping = selector._explicit_mappings.get("scripts/test_resolver_bandit.py", set())
        self.assertIn("scripts/resolver_bandit.py", mapping)

    def test_select_for_changes_empty(self):
        selector = TestSelector()
        result = selector.select_for_changes([])
        self.assertIsInstance(result, SelectedTests)
        self.assertGreaterEqual(result.total_available, 0)
        self.assertEqual(result.selection_ratio, 1.0)

    def test_categorize_changes_test_files(self):
        selector = TestSelector()
        result = selector.categorize_changes(["scripts/test_pipeline_dag.py"])
        self.assertIn("scripts/test_pipeline_dag.py", result["test_files"])

    def test_categorize_changes_source_scripts(self):
        selector = TestSelector()
        result = selector.categorize_changes(["scripts/pipeline_dag.py"])
        self.assertIn("scripts/pipeline_dag.py", result["source_scripts"])

    def test_categorize_changes_config(self):
        selector = TestSelector()
        result = selector.categorize_changes(["scripts/_ci_pytest_manifest.toml"])
        self.assertIn("scripts/_ci_pytest_manifest.toml", result["config_files"])

    def test_categorize_changes_docs(self):
        selector = TestSelector()
        result = selector.categorize_changes(["README.md"])
        self.assertIn("README.md", result["docs"])

    def test_categorize_changes_separate_categories(self):
        selector = TestSelector()
        result = selector.categorize_changes([
            "scripts/test_pipeline_dag.py",
            "scripts/pipeline_dag.py",
            "README.md",
        ])
        self.assertEqual(len(result["test_files"]), 1)
        self.assertEqual(len(result["source_scripts"]), 1)
        self.assertEqual(len(result["docs"]), 1)

    def test_run_selected_returns_list(self):
        selector = TestSelector()
        result = selector.run_selected(["scripts/pipeline_dag.py"])
        self.assertIsInstance(result, list)

    def test_build_mappings_creates_mappings(self):
        selector = TestSelector()
        mappings = selector.build_mappings()
        self.assertIsInstance(mappings, list)

    def test_coverage_report_structure(self):
        selector = TestSelector()
        report = selector.coverage_report()
        self.assertIn("total_tests", report)
        self.assertIn("tests_without_coverage", report)
        self.assertIn("scripts_without_tests", report)
        self.assertIn("coverage", report)

    def test_relative_path_resolution(self):
        selector = TestSelector()
        mapping = selector._resolve_imports("scripts/test_pipeline_dag.py")
        self.assertIsInstance(mapping, set)


class TestSelectorEdgeCases(unittest.TestCase):
    def test_missing_manifest_path(self):
        selector = TestSelector(manifest_path="/nonexistent/manifest.toml")
        mappings = selector.build_mappings()
        self.assertIsInstance(mappings, list)

    def test_select_for_changes_no_mappings(self):
        selector = TestSelector()
        result = selector.select_for_changes(["nonexistent/file.py"])
        self.assertIsInstance(result, SelectedTests)
        self.assertIsInstance(result.selected, list)


class TestTestMapping(unittest.TestCase):
    def test_default_values(self):
        mapping = TestMapping(test_path="a", coverage=set())
        self.assertEqual(mapping.test_path, "a")
        self.assertEqual(mapping.coverage, set())
        self.assertEqual(mapping.manifest_id, "")

    def test_with_values(self):
        mapping = TestMapping(
            test_path="scripts/test_x.py",
            coverage={"scripts/x.py"},
            manifest_id="test-x",
        )
        self.assertEqual(mapping.manifest_id, "test-x")
        self.assertIn("scripts/x.py", mapping.coverage)


if __name__ == "__main__":
    unittest.main()
