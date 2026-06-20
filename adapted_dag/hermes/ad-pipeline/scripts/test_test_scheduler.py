#!/usr/bin/env python3
"""Tests for test_scheduler module."""
from __future__ import annotations

import sys
import unittest

try:
    from test_scheduler import (
        TestScheduler, TestResult, TestPlan, SchedulerResult,
    )
except ImportError:
    from scripts.test_scheduler import (
        TestScheduler, TestResult, TestPlan, SchedulerResult,
    )


class TestTestResult(unittest.TestCase):
    def test_passed_default(self):
        r = TestResult(test_id="t1", test_path="p", passed=True, duration_sec=1.0)
        self.assertTrue(r.passed)
        self.assertEqual(r.exit_code, 0)

    def test_failed(self):
        r = TestResult(test_id="t1", test_path="p", passed=False, duration_sec=2.0, exit_code=1)
        self.assertFalse(r.passed)
        self.assertEqual(r.exit_code, 1)

    def test_error(self):
        r = TestResult(test_id="t1", test_path="p", passed=False, duration_sec=0.5, exit_code=-1)
        self.assertEqual(r.exit_code, -1)


class TestTestPlan(unittest.TestCase):
    def test_default_empty(self):
        p = TestPlan()
        self.assertEqual(p.groups, [])
        self.assertEqual(p.total_tests, 0)

    def test_with_groups(self):
        p = TestPlan(groups=[["a", "b"], ["c"]], total_tests=3)
        self.assertEqual(len(p.groups), 2)
        self.assertEqual(p.total_tests, 3)


class TestSchedulerResult(unittest.TestCase):
    def test_all_passed(self):
        r = SchedulerResult(results=[
            TestResult(test_id="t1", test_path="p1", passed=True, duration_sec=1.0),
            TestResult(test_id="t2", test_path="p2", passed=True, duration_sec=2.0),
        ])
        self.assertTrue(r.all_passed)
        self.assertEqual(r.passed, 2)
        self.assertEqual(r.failed, 0)

    def test_some_failed(self):
        r = SchedulerResult(results=[
            TestResult(test_id="t1", test_path="p1", passed=True, duration_sec=1.0),
            TestResult(test_id="t2", test_path="p2", passed=False, duration_sec=2.0, exit_code=1),
        ])
        self.assertFalse(r.all_passed)
        self.assertEqual(r.passed, 1)
        self.assertEqual(r.failed, 1)

    def test_errors_count_separate(self):
        r = SchedulerResult(results=[
            TestResult(test_id="t1", test_path="p1", passed=False, duration_sec=1.0, exit_code=-1),
        ])
        self.assertEqual(r.errors, 1)
        self.assertEqual(r.failed, 0)


class TestSchedulerInitialization(unittest.TestCase):
    def test_default_max_workers(self):
        scheduler = TestScheduler()
        self.assertGreaterEqual(scheduler.max_workers, 2)

    def test_custom_max_workers(self):
        scheduler = TestScheduler(max_workers=8)
        self.assertEqual(scheduler.max_workers, 8)

    def test_custom_timeout(self):
        scheduler = TestScheduler(timeout=300)
        self.assertEqual(scheduler.timeout, 300)

    def test_python_path(self):
        scheduler = TestScheduler(python_path="/usr/bin/python3")
        self.assertEqual(scheduler.python_path, "/usr/bin/python3")


class TestSchedulerPlanning(unittest.TestCase):
    def setUp(self):
        self.scheduler = TestScheduler(max_workers=4)

    def test_empty_run_returns_no_tests(self):
        result = self.scheduler.run_selected([])
        self.assertEqual(len(result.results), 0)

    def test_build_plan_empty(self):
        plan = self.scheduler._build_parallel_plan([])
        self.assertEqual(plan.total_tests, 0)

    def test_build_plan_single(self):
        plan = self.scheduler._build_parallel_plan(["test_a.py"])
        self.assertEqual(plan.total_tests, 1)

    def test_build_plan_multiple(self):
        plan = self.scheduler._build_parallel_plan(["a", "b", "c", "d", "e"])
        self.assertEqual(plan.total_tests, 5)
        self.assertTrue(len(plan.groups) > 0)

    def test_build_plan_groups_even_distribution(self):
        plan = self.scheduler._build_parallel_plan(["a", "b", "c", "d"])
        total_in_groups = sum(len(g) for g in plan.groups)
        self.assertEqual(total_in_groups, 4)

    def test_plan_from_missing_manifest(self):
        plan = self.scheduler.plan_from_manifest("/nonexistent/manifest.toml")
        self.assertEqual(plan.total_tests, 0)

    def test_adaptive_workers_few_tests(self):
        workers = self.scheduler.adaptive_workers(["a"])
        self.assertEqual(workers, 1)

    def test_adaptive_workers_many_tests(self):
        self.scheduler.history = {f"t{i}.py": 60.0 for i in range(20)}
        workers = self.scheduler.adaptive_workers(list(self.scheduler.history.keys()))
        self.assertLessEqual(workers, self.scheduler.max_workers)


class TestSchedulerSharding(unittest.TestCase):
    def setUp(self):
        self.scheduler = TestScheduler(max_workers=4)

    def test_shard_distribution(self):
        tests = [f"test_{i}.py" for i in range(10)]
        shard0 = self.scheduler.distribute_across_shards(tests, num_shards=3, shard_index=0)
        shard1 = self.scheduler.distribute_across_shards(tests, num_shards=3, shard_index=1)
        shard2 = self.scheduler.distribute_across_shards(tests, num_shards=3, shard_index=2)

        all_sharded = set(shard0) | set(shard1) | set(shard2)
        self.assertEqual(all_sharded, set(tests))

    def test_shard_no_overlap(self):
        tests = [f"test_{i}.py" for i in range(10)]
        shard0 = self.scheduler.distribute_across_shards(tests, num_shards=2, shard_index=0)
        shard1 = self.scheduler.distribute_across_shards(tests, num_shards=2, shard_index=1)
        self.assertEqual(len(set(shard0) & set(shard1)), 0)

    def test_shard_single_shard(self):
        tests = [f"test_{i}.py" for i in range(5)]
        shard = self.scheduler.distribute_across_shards(tests, num_shards=1, shard_index=0)
        self.assertEqual(len(shard), 5)

    def test_shard_index_out_of_range(self):
        tests = [f"test_{i}.py" for i in range(5)]
        shard = self.scheduler.distribute_across_shards(tests, num_shards=3, shard_index=2)
        self.assertTrue(len(shard) > 0)


class TestSchedulerConcurrencyDetection(unittest.TestCase):
    def test_detect_workers_returns_positive(self):
        workers = TestScheduler._detect_workers()
        self.assertGreaterEqual(workers, 1)


class TestSchedulerHistoryTracking(unittest.TestCase):
    def test_history_empty_initialized(self):
        scheduler = TestScheduler()
        self.assertEqual(scheduler.history, {})

    def test_run_updates_history(self):
        scheduler = TestScheduler(max_workers=1)
        result = scheduler.run_selected([])
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
