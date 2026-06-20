#!/usr/bin/env python3
"""Test scheduler — parallel test execution with dependency awareness.

Distributes selected tests across worker processes respecting
test-level dependencies and resource constraints. Integrates with the
existing CI manifest system.

Usage:
    scheduler = TestScheduler(max_workers=4)
    results = scheduler.run_selected(["scripts/test_pipeline_dag.py", ...])
    print(results.summary)

    # Integration with manifest runner
    manifest_plan = scheduler.plan_from_manifest("scripts/_ci_pytest_manifest.toml")
    scheduler.run_plan(manifest_plan)
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    test_id: str
    test_path: str
    passed: bool
    duration_sec: float
    exit_code: int = 0
    output: str = ""
    error: str = ""


@dataclass
class TestPlan:
    groups: list[list[str]] = field(default_factory=list)
    total_tests: int = 0
    estimated_duration_sequential: float = 0.0
    estimated_duration_parallel: float = 0.0


@dataclass
class SchedulerResult:
    results: list[TestResult] = field(default_factory=list)
    total_duration: float = 0.0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    all_passed: bool = True
    summary: str = ""

    def __post_init__(self):
        self.passed = sum(1 for r in self.results if r.passed)
        self.failed = sum(1 for r in self.results if not r.passed and r.exit_code != -1)
        self.errors = sum(1 for r in self.results if r.exit_code == -1)
        self.all_passed = self.failed == 0 and self.errors == 0


TEST_DURATION_ESTIMATES: dict[str, float] = {}
TEST_DEPENDENCIES: dict[str, list[str]] = {}

TEST_SIZE_CLASSIFICATION: dict[str, str] = {}

DEFAULT_TEST_TIMEOUT = 120


class TestScheduler:
    def __init__(self, max_workers: int | None = None,
                 timeout: int = DEFAULT_TEST_TIMEOUT,
                 python_path: str | None = None):
        self.max_workers = max_workers or self._detect_workers()
        self.timeout = timeout
        self.python_path = python_path or sys.executable
        self.history: dict[str, float] = dict(TEST_DURATION_ESTIMATES)
        self.deps: dict[str, list[str]] = dict(TEST_DEPENDENCIES)

    @staticmethod
    def _detect_workers() -> int:
        cpus = os.cpu_count() or 2
        return max(1, cpus - 1) if cpus > 2 else 2

    def run_selected(self, test_paths: list[str],
                     extra_args: list[str] | None = None,
                     label: str = "selected") -> SchedulerResult:
        if not test_paths:
            return SchedulerResult(summary=f"{label}: no tests to run")

        plan = self._build_parallel_plan(test_paths)
        results: list[TestResult] = []
        start = time.monotonic()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            all_futures = []
            for group_idx, group in enumerate(plan.groups):
                for test_path in group:
                    fut = executor.submit(
                        self._run_single_test, test_path, extra_args
                    )
                    all_futures.append(fut)

            for future in concurrent.futures.as_completed(all_futures):
                result = future.result()
                results.append(result)

        total_duration = time.monotonic() - start
        sr = SchedulerResult(
            results=results,
            total_duration=total_duration,
        )

        lines = [f"{label}: {sr.passed}/{len(results)} passed "
                 f"({sr.failed} failed, {sr.errors} errors) "
                 f"in {total_duration:.1f}s"]

        savings = ""
        if plan.estimated_duration_sequential > 0:
            ratio = total_duration / plan.estimated_duration_sequential
            savings = f" (vs ~{plan.estimated_duration_sequential:.0f}s sequential, "
            savings += f"{'%.1f' % ((1-ratio)*100)}% faster)" if ratio < 1 else " no speedup)"
        lines[0] += savings

        for r in sorted(results, key=lambda x: x.duration_sec, reverse=True)[:3]:
            if not r.passed:
                status = "FAIL" if r.exit_code != -1 else "ERROR"
                lines.append(f"  [{status}] {r.test_id} ({r.duration_sec:.1f}s)")

        if sr.failed > 0:
            for r in results:
                if not r.passed and r.error:
                    lines.append(f"  {r.test_id}: {r.error[:200]}")

        sr.summary = "\n".join(lines)
        return sr

    def _build_parallel_plan(self, test_paths: list[str]) -> TestPlan:
        sorted_paths = sorted(set(test_paths))
        plan = TestPlan(total_tests=len(sorted_paths))

        if not sorted_paths:
            return plan

        groups: list[list[str]] = [[] for _ in range(self.max_workers)]

        for i, path in enumerate(sorted_paths):
            group_idx = i % self.max_workers
            groups[group_idx].append(path)
            est = self.history.get(path, 30.0)
            plan.estimated_duration_sequential += est

        plan.groups = [g for g in groups if g]
        if plan.groups:
            plan.estimated_duration_parallel = max(
                sum(self.history.get(p, 30.0) for p in g)
                for g in plan.groups
            )
        return plan

    def _run_single_test(self, test_path: str,
                         extra_args: list[str] | None = None) -> TestResult:
        test_id = Path(test_path).stem
        start = time.monotonic()
        try:
            cmd = [self.python_path, "-m", "pytest", test_path, "-v"]
            if extra_args:
                cmd.extend(extra_args)

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONPATH": "."},
            )
            duration = time.monotonic() - start
            passed = proc.returncode == 0

            self.history[test_path] = (
                0.9 * self.history.get(test_path, duration) + 0.1 * duration
            )

            return TestResult(
                test_id=test_id,
                test_path=test_path,
                passed=passed,
                duration_sec=round(duration, 2),
                exit_code=proc.returncode,
                output=proc.stdout[-2000:] if proc.stdout else "",
                error=proc.stderr[-2000:] if proc.stderr and not passed else "",
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return TestResult(
                test_id=test_id,
                test_path=test_path,
                passed=False,
                duration_sec=round(duration, 2),
                exit_code=-1,
                error=f"TIMEOUT after {self.timeout}s",
            )
        except Exception as e:
            duration = time.monotonic() - start
            return TestResult(
                test_id=test_id,
                test_path=test_path,
                passed=False,
                duration_sec=round(duration, 2),
                exit_code=-1,
                error=str(e),
            )

    def plan_from_manifest(self, manifest_path: str | Path) -> TestPlan:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            return TestPlan()

        content = manifest_path.read_text()
        test_ids: list[str] = []
        current_id = ""
        for line in content.splitlines():
            m_id = __import__("re").match(r'^\s*id\s*=\s*"([^"]+)"', line)
            m_path = __import__("re").match(r'^\s*path\s*=\s*"([^"]+)"', line)
            if m_id:
                current_id = m_id.group(1)
            if m_path and current_id:
                test_ids.append(m_path.group(1))
                current_id = ""

        return self._build_parallel_plan(test_ids)

    def run_plan(self, plan: TestPlan) -> SchedulerResult:
        all_tests = [t for group in plan.groups for t in group]
        return self.run_selected(all_tests, label="manifest")

    def adaptive_workers(self, test_paths: list[str]) -> int:
        total_est = sum(self.history.get(p, 30.0) for p in test_paths)
        num_tests = len(test_paths)

        if num_tests <= 2:
            return 1
        if total_est < 30:
            return min(2, self.max_workers)
        if total_est < 120:
            return min(self.max_workers // 2 + 1, self.max_workers)
        return self.max_workers

    def distribute_across_shards(self, test_paths: list[str],
                                 num_shards: int,
                                 shard_index: int) -> list[str]:
        sorted_paths = sorted(set(test_paths))
        shard_size = max(1, len(sorted_paths) // num_shards)
        start = shard_index * shard_size
        if shard_index == num_shards - 1:
            return sorted_paths[start:]
        return sorted_paths[start:start + shard_size]
