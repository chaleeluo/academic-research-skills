#!/usr/bin/env python3
"""Test selector — minimal test set selection based on file changes.

Analyzes a set of changed files and selects the minimal set of tests
needed to validate those changes. Uses file-to-test mapping extracted
from imports, explicit test annotations, and heuristic rules.

Usage:
    selector = TestSelector(Path("scripts/_ci_pytest_manifest.toml"))
    tests = selector.select_for_changes(["scripts/pipeline_dag.py", "scripts/resolver_bandit.py"])
    print(tests.selected)  # list of test file paths
    print(tests.skipped_from_full)  # tests excluded vs full suite
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SelectedTests:
    selected: list[str] = field(default_factory=list)
    skipped_from_full: list[str] = field(default_factory=list)
    total_available: int = 0
    selection_ratio: float = 1.0
    reason: str = ""


@dataclass
class TestMapping:
    test_path: str
    coverage: set[str]
    manifest_id: str = ""


IMPORT_PATTERNS = [
    re.compile(r"^(?:from|import)\s+scripts\.(\w+)"),
    re.compile(r"^(?:from|import)\s+(\w+)"),
    re.compile(r"from\s+scripts\.(\w+)\s+import"),
]


class TestSelector:
    def __init__(self, repo_root: str | Path | None = None,
                 manifest_path: str | Path | None = None):
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        if manifest_path:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.repo_root / "scripts" / "_ci_pytest_manifest.toml"
        self._mappings: list[TestMapping] = []
        self._explicit_mappings: dict[str, set[str]] = {
            "scripts/test_text_similarity.py": {
                "scripts/_text_similarity.py",
            },
            "scripts/test_contamination_signals.py": {
                "scripts/contamination_signals.py",
                "scripts/_text_similarity.py",
            },
            "scripts/test_citation_verification_summary.py": {
                "scripts/citation_verification_summary.py",
            },
            "scripts/test_verification_gate.py": {
                "scripts/verification_gate/",
            },
            "scripts/test_semantic_scholar_client.py": {
                "scripts/semantic_scholar_client.py",
                "scripts/_text_similarity.py",
            },
            "scripts/test_crossref_client.py": {
                "scripts/crossref_client.py",
                "scripts/_text_similarity.py",
            },
            "scripts/test_openalex_client.py": {
                "scripts/openalex_client.py",
                "scripts/_text_similarity.py",
            },
            "scripts/test_arxiv_client.py": {
                "scripts/arxiv_client.py",
                "scripts/_text_similarity.py",
            },
            "scripts/test_verification_cache.py": {
                "scripts/verification_cache.py",
            },
            "scripts/test_block_parser.py": {
                "scripts/_block_parser.py",
            },
            "scripts/test_ars_anchorize_draft.py": {
                "scripts/ars_anchorize_draft.py",
                "scripts/_block_parser.py",
            },
            "scripts/test_ars_apply_revision_patch.py": {
                "scripts/ars_apply_revision_patch.py",
                "scripts/_block_parser.py",
            },
            "scripts/test_pipeline_dag.py": {
                "scripts/pipeline_dag.py",
                "scripts/pipeline_scheduler.py",
            },
            "scripts/test_resolver_bandit.py": {
                "scripts/resolver_bandit.py",
            },
            "scripts/test_test_selector.py": {
                "scripts/test_selector.py",
                "scripts/test_scheduler.py",
            },
            "scripts/test_test_scheduler.py": {
                "scripts/test_scheduler.py",
            },
            "scripts/test_check_spec_consistency.py": {
                "scripts/check_spec_consistency.py",
            },
            "scripts/test_check_ci_pytest_manifest.py": {
                "scripts/_ci_pytest_manifest.toml",
                "scripts/check_ci_pytest_manifest.py",
                "scripts/run_ci_pytest_manifest.py",
            },
        }

    def _resolve_imports(self, test_path: str) -> set[str]:
        resolved: set[str] = set()

        if test_path in self._explicit_mappings:
            resolved.update(self._explicit_mappings[test_path])

        full_path = self.repo_root / test_path
        if not full_path.exists():
            return resolved

        content = full_path.read_text(encoding="utf-8", errors="replace")
        for pattern in IMPORT_PATTERNS:
            for match in pattern.finditer(content):
                module = match.group(1)
                resolved.add(f"scripts/{module}.py")
                resolved.add(f"scripts/{module}/")

        return resolved

    def build_mappings(self) -> list[TestMapping]:
        if self._mappings:
            return self._mappings

        pattern = re.compile(r"scripts/test_[a-zA-Z0-9_]+\.py$")
        test_files: list[str] = []

        scripts_dir = self.repo_root / "scripts"
        if scripts_dir.exists():
            for f in sorted(scripts_dir.iterdir()):
                if pattern.match(f.name):
                    test_files.append(f"scripts/{f.name}")

        if self.manifest_path.exists():
            for line in self.manifest_path.read_text().splitlines():
                m = re.match(r'^\s*path\s*=\s*"([^"]+)"', line)
                if m:
                    p = m.group(1)
                    if p not in test_files:
                        test_files.append(p)

        for tf in sorted(set(test_files)):
            deps = self._resolve_imports(tf)
            self._mappings.append(TestMapping(
                test_path=tf, coverage=deps,
                manifest_id=Path(tf).stem,
            ))

        return self._mappings

    def select_for_changes(self, changed_files: list[str]) -> SelectedTests:
        self.build_mappings()
        changed_set = set(changed_files)
        all_test_files = [m.test_path for m in self._mappings]

        if not changed_set:
            return SelectedTests(
                selected=all_test_files,
                total_available=len(all_test_files),
                selection_ratio=1.0,
                reason="no changes detected, running all tests",
            )

        triggering_tests: list[str] = []
        direct_hits: list[str] = []

        for mapping in self._mappings:
            if mapping.test_path in changed_set:
                direct_hits.append(mapping.test_path)
                triggering_tests.append(mapping.test_path)
                continue

            for dep in mapping.coverage:
                for changed in changed_set:
                    if changed == dep or changed.startswith(dep.rstrip("/")):
                        triggering_tests.append(mapping.test_path)
                        break
                if mapping.test_path in triggering_tests:
                    break

        generic_scripts = [f for f in changed_set if f.startswith("scripts/")
                           and not f.startswith("scripts/test_")
                           and not f.endswith(".toml")
                           and f not in self._get_all_covered_scripts()]

        if generic_scripts:
            for mapping in self._mappings:
                if mapping.test_path not in triggering_tests:
                    for changed in generic_scripts:
                        if any(changed.startswith(dep.rstrip("/")) or dep.startswith(changed)
                               for dep in mapping.coverage):
                            triggering_tests.append(mapping.test_path)
                            break

        triggering_tests = sorted(set(triggering_tests))
        skipped = [t for t in all_test_files if t not in triggering_tests]

        ratio = len(triggering_tests) / max(len(all_test_files), 1)
        reason_parts = []
        if direct_hits:
            reason_parts.append(f"{len(direct_hits)} test files directly changed")
        if changed_set:
            reason_parts.append(f"{len(changed_set)} source files changed")

        return SelectedTests(
            selected=triggering_tests,
            skipped_from_full=skipped,
            total_available=len(all_test_files),
            selection_ratio=round(ratio, 3),
            reason="; ".join(reason_parts) if reason_parts else "test selection based on file mapping",
        )

    def _get_all_covered_scripts(self) -> set[str]:
        covered: set[str] = set()
        for mapping in self._mappings:
            covered.update(mapping.coverage)
        return covered

    def run_selected(self, changed_files: list[str]) -> list[str]:
        selection = self.select_for_changes(changed_files)
        return selection.selected

    def categorize_changes(self, changed_files: list[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {
            "test_files": [],
            "source_scripts": [],
            "config_files": [],
            "docs": [],
            "other": [],
        }
        for f in changed_files:
            if f.startswith("scripts/test_") and f.endswith(".py"):
                result["test_files"].append(f)
            elif f.startswith("scripts/") and f.endswith(".py"):
                result["source_scripts"].append(f)
            elif f.endswith(".toml") or f.endswith(".yaml") or f.endswith(".yml"):
                result["config_files"].append(f)
            elif f.startswith("docs/"):
                result["docs"].append(f)
            elif f.startswith("README"):
                result["docs"].append(f)
            else:
                result["other"].append(f)
        return result

    def coverage_report(self) -> dict[str, Any]:
        self.build_mappings()
        unlinked_tests = [
            m.test_path for m in self._mappings if not m.coverage
        ]
        all_scripts = set()
        for mapping in self._mappings:
            all_scripts.update(mapping.coverage)

        unlinked_scripts = []
        scripts_dir = self.repo_root / "scripts"
        if scripts_dir.exists():
            for f in scripts_dir.iterdir():
                name = f"scripts/{f.name}"
                if f.name.endswith(".py") and not f.name.startswith("test_") and not f.name.startswith("_"):
                    if name not in all_scripts and name not in {m.test_path for m in self._mappings}:
                        unlinked_scripts.append(name)

        return {
            "total_tests": len(self._mappings),
            "tests_without_coverage": unlinked_tests,
            "scripts_without_tests": sorted(unlinked_scripts),
            "coverage": {
                m.test_path: sorted(m.coverage)
                for m in self._mappings
            },
        }
