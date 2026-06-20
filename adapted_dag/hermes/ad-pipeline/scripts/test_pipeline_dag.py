#!/usr/bin/env python3
"""Tests for pipeline_dag and pipeline_scheduler modules."""
from __future__ import annotations

import sys
import unittest

try:
    from pipeline_dag import PipelineDAG, StageType, ResourceConstraint, auto_detect_entry_stage
    from pipeline_scheduler import OptimizedPipelineScheduler
except ImportError:
    from scripts.pipeline_dag import PipelineDAG, StageType, ResourceConstraint, auto_detect_entry_stage
    from scripts.pipeline_scheduler import OptimizedPipelineScheduler


class TestPipelineDAGStructure(unittest.TestCase):
    def setUp(self):
        self.dag = PipelineDAG()

    def test_all_stages_present(self):
        expected = {"stage1", "stage2", "stage2_5", "stage3", "stage3_reject",
                    "stage4_minor", "stage4_major", "stage3_prime", "stage4_prime",
                    "stage4_5", "stage5", "stage6"}
        self.assertEqual(set(self.dag.STAGE_REGISTRY.keys()), expected)

    def test_topological_sort_starts_with_stage1(self):
        order = self.dag.topological_sort("stage1")
        self.assertEqual(order[0], "stage1")

    def test_dependencies_stage3(self):
        deps = self.dag.dependencies("stage3")
        self.assertIn("stage2_5", deps)

    def test_dependents_stage2(self):
        deps = self.dag.dependents("stage2")
        self.assertIn("stage2_5", deps)

    def test_parallelizable_layers_nonempty(self):
        layers = self.dag.parallelizable_stages()
        self.assertTrue(len(layers) > 0)
        for layer in layers:
            self.assertTrue(len(layer) > 0)

    def test_critical_path_returns_ordered_list(self):
        path = self.dag.critical_path()
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0], "stage1")
        self.assertEqual(path[-1], "stage6")

    def test_conditional_stages_identified(self):
        self.assertTrue(self.dag.is_conditional("stage3_reject"))
        self.assertTrue(self.dag.is_conditional("stage4_minor"))
        self.assertFalse(self.dag.is_conditional("stage1"))
        self.assertFalse(self.dag.is_conditional("stage2_5"))

    def test_material_dependencies(self):
        mats = self.dag.material_dependencies("stage4_minor")
        names = {m.name for m in mats}
        self.assertIn("Review Reports", names)
        self.assertIn("Editorial Decision", names)

    def test_material_produced(self):
        mats = self.dag.material_produced("stage2_5")
        names = {m.name for m in mats}
        self.assertIn("Integrity Report", names)
        self.assertIn("Verified Paper", names)

    def test_schedule_returns_ordered_plan(self):
        schedule = self.dag.schedule()
        self.assertTrue(len(schedule) > 0)

    def test_cost_estimate(self):
        estimate = self.dag.cost_estimate()
        self.assertIn("total_token_cost_estimate", estimate)
        self.assertIn("total_duration_min", estimate)
        self.assertIn("retry_overhead_min", estimate)
        self.assertTrue(estimate["total_token_cost_estimate"] > 0)

    def test_cost_estimate_with_retries(self):
        estimate = self.dag.cost_estimate(with_retries=True)
        self.assertTrue(estimate["retry_overhead_min"] > 0)

    def test_cost_estimate_without_retries(self):
        estimate = self.dag.cost_estimate(with_retries=False)
        self.assertEqual(estimate["retry_overhead_min"], 0)

    def test_to_dict_returns_full_graph(self):
        d = self.dag.to_dict()
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertIn("materials", d)
        self.assertEqual(len(d["nodes"]), len(self.dag.STAGE_REGISTRY))

    def test_auto_detect_entry_stage_empty(self):
        self.assertEqual(auto_detect_entry_stage([]), "stage1")

    def test_auto_detect_entry_stage_with_draft(self):
        self.assertEqual(auto_detect_entry_stage(["Paper Draft"]), "stage2")

    def test_auto_detect_entry_stage_with_reviews(self):
        self.assertEqual(auto_detect_entry_stage(["Review Reports"]), "stage3")

    def test_auto_detect_with_final_paper(self):
        self.assertEqual(auto_detect_entry_stage(["Final Paper"]), "stage5")


class TestPipelineDAGConstraints(unittest.TestCase):
    def test_custom_constraints(self):
        constraints = ResourceConstraint(max_concurrent_stages=5, max_token_budget=500000)
        dag = PipelineDAG(constraints)
        self.assertEqual(dag.constraints.max_concurrent_stages, 5)

    def test_default_constraints(self):
        dag = PipelineDAG()
        self.assertEqual(dag.constraints.max_concurrent_stages, 3)

    def test_schedule_respects_concurrency(self):
        dag = PipelineDAG(ResourceConstraint(max_concurrent_stages=1))
        schedule = dag.schedule()
        in_progress_count = sum(
            1 for s in schedule
            if str(s["status"]) == "StageStatus.IN_PROGRESS"
        )
        self.assertLessEqual(in_progress_count, 2)


class TestPipelineDAGParallelGroups(unittest.TestCase):
    def test_stage1_has_parallel_groups(self):
        groups = PipelineDAG().intra_stage_parallel_groups("stage1")
        self.assertIn("lit_search", groups)

    def test_stage2_has_parallel_groups(self):
        groups = PipelineDAG().intra_stage_parallel_groups("stage2")
        self.assertIn("viz_gen", groups)

    def test_stage3_no_parallel_groups(self):
        groups = PipelineDAG().intra_stage_parallel_groups("stage3")
        self.assertEqual(groups, [])


class TestPipelineScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = OptimizedPipelineScheduler()

    def test_build_optimized_plan_returns_plan(self):
        plan = self.scheduler.build_optimized_plan()
        self.assertTrue(len(plan.execution_order) > 0)
        self.assertIn("critical_path", plan.__dict__)
        self.assertIn("parallel_stages", plan.__dict__)

    def test_plan_with_materials(self):
        plan = self.scheduler.build_optimized_plan(available_materials=["Paper Draft"])
        self.assertTrue(len(plan.execution_order) > 0)

    def test_parallel_stages_detected(self):
        plan = self.scheduler.build_optimized_plan()
        if plan.parallel_stages:
            for layer in plan.parallel_stages:
                self.assertIsInstance(layer, set)

    def test_critical_path_in_plan(self):
        plan = self.scheduler.build_optimized_plan()
        self.assertTrue(len(plan.critical_path) >= 2)

    def test_savings_calculated(self):
        plan = self.scheduler.build_optimized_plan()
        self.assertIn("duration_reduction_min", plan.savings_vs_sequential)
        self.assertIn("duration_reduction_pct", plan.savings_vs_sequential)

    def test_incremental_plan_no_remaining(self):
        plan = self.scheduler.incremental_plan(
            completed_stages=["stage1", "stage2", "stage2_5", "stage3", "stage4_5", "stage5", "stage6"]
        )
        self.assertEqual(plan.total_duration_min, 0)

    def test_incremental_plan_with_remaining(self):
        plan = self.scheduler.incremental_plan(
            completed_stages=["stage1", "stage2"],
            available_materials=["Integrity Report"]
        )
        self.assertTrue(len(plan.execution_order) > 0)

    def test_token_budget_allocation(self):
        plan = self.scheduler.build_optimized_plan()
        allocation = self.scheduler.allocate_token_budget(plan, total_budget=200000)
        self.assertTrue(len(allocation) > 0)
        total = sum(allocation.values())
        self.assertAlmostEqual(total, 200000, delta=20000)

    def test_token_budget_empty_plan(self):
        empty_plan = self.scheduler.build_optimized_plan(
            available_materials=[], target_stages=[]
        )
        allocation = self.scheduler.allocate_token_budget(empty_plan, total_budget=100000)
        self.assertEqual(allocation, {})

    def test_bottleneck_analysis(self):
        plan = self.scheduler.build_optimized_plan()
        bottlenecks = self.scheduler.analyze_bottleneck(plan)
        self.assertIsInstance(bottlenecks, list)

    def test_bottleneck_recommendation_integrity(self):
        plan = self.scheduler.build_optimized_plan()
        bottlenecks = self.scheduler.analyze_bottleneck(plan)
        for b in bottlenecks:
            self.assertIn("recommendation", b)
            self.assertIn("stage_id", b)
            self.assertTrue(len(b["recommendation"]) > 0)

    def test_suggested_concurrency_range(self):
        plan = self.scheduler.build_optimized_plan()
        self.assertGreaterEqual(plan.suggested_concurrency, 1)
        self.assertLessEqual(plan.suggested_concurrency, 3)


class TestPipelineDAGRegression(unittest.TestCase):
    def test_no_circular_dependencies(self):
        dag = PipelineDAG()
        visited = set()
        stack = set()

        def dfs(node):
            if node in stack:
                return False
            if node in visited:
                return True
            visited.add(node)
            stack.add(node)
            for t in dag._outgoing.get(node, []):
                if t.condition and t.to_stage == node:
                    continue
                if not dfs(t.to_stage):
                    return False
            stack.remove(node)
            return True

        for stage in dag._all_stages:
            if stage not in visited:
                result = dfs(stage)
                if not result:
                    pass

        no_cycle = True
        for stage in dag._all_stages:
            visited.clear()
            stack.clear()
            if not dfs(stage):
                no_cycle = False
        self.assertTrue(no_cycle)

    def test_all_materials_have_producer_and_consumer(self):
        dag = PipelineDAG()
        for mat in dag.MATERIALS:
            self.assertIn(mat.produced_by, dag._all_stages,
                          f"{mat.name} producer {mat.produced_by} not a stage")
            self.assertIn(mat.consumed_by, dag._all_stages,
                          f"{mat.name} consumer {mat.consumed_by} not a stage")

    def test_required_materials_are_on_critical_path(self):
        dag = PipelineDAG()
        critical = set(dag.critical_path())
        for mat in dag.MATERIALS:
            if mat.required:
                self.assertIn(mat.produced_by, critical,
                              f"Required material {mat.name} producer {mat.produced_by} not on critical path")


if __name__ == "__main__":
    unittest.main()
