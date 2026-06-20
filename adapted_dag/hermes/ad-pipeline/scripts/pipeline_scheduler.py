#!/usr/bin/env python3
"""Pipeline scheduler — cost-aware DAG scheduler with parallelism detection.

Builds on pipeline_dag.PipelineDAG to produce optimized execution plans
that respect token budgets, concurrency limits, and API rate constraints.
Supports incremental scheduling (re-schedule from a given stage) and
critical-path-based resource allocation.

Usage:
    from pipeline_scheduler import OptimizedPipelineScheduler

    scheduler = OptimizedPipelineScheduler()
    plan = scheduler.build_optimized_plan(available_materials=[...])
    print(plan.execution_order)
    print(plan.parallel_stages)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from pipeline_dag import (
        PipelineDAG, ResourceConstraint, StageNode, StageStatus,
        StageType, auto_detect_entry_stage,
    )
except ImportError:
    from scripts.pipeline_dag import (
        PipelineDAG, ResourceConstraint, StageNode, StageStatus,
        StageType, auto_detect_entry_stage,
    )


@dataclass
class ExecutionSlot:
    stage_id: str
    start_min: int = 0
    end_min: int = 0
    parallel_group: str | None = None
    token_cost: int = 0


@dataclass
class OptimizedPlan:
    execution_order: list[str] = field(default_factory=list)
    parallel_stages: list[set[str]] = field(default_factory=list)
    total_duration_min: int = 0
    total_token_cost: int = 0
    suggested_concurrency: int = 1
    critical_path: list[str] = field(default_factory=list)
    execution_slots: list[ExecutionSlot] = field(default_factory=list)
    savings_vs_sequential: dict[str, float] = field(default_factory=dict)


class OptimizedPipelineScheduler:
    def __init__(self, constraints: ResourceConstraint | None = None):
        self.dag = PipelineDAG(constraints)
        self.constraints = constraints or ResourceConstraint()

    def build_optimized_plan(self, available_materials: list[str] | None = None,
                             target_stages: list[str] | None = None,
                             prefer_quality: bool = True) -> OptimizedPlan:
        entry = auto_detect_entry_stage(available_materials or [])
        plan = OptimizedPlan()

        plan.critical_path = self.dag.critical_path()
        plan.parallel_stages = self.dag.parallelizable_stages()

        sequential_order = self.dag.topological_sort(entry)
        def _should_count(sid: str) -> bool:
            node = self.dag.STAGE_REGISTRY.get(sid)
            if not node:
                return False
            if target_stages and sid not in target_stages:
                return False
            return node.stage_type != StageType.CONDITIONAL

        sequential_duration = sum(
            self.dag.STAGE_REGISTRY[s].estimated_duration_min
            for s in sequential_order if _should_count(s)
        )

        execution_order: list[str] = []
        completed: set[str] = set()
        current_time = 0

        stages_to_run = target_stages or sequential_order

        while len(completed) < len(stages_to_run):
            ready = []
            for sid in stages_to_run:
                if sid in completed:
                    continue
                node = self.dag.STAGE_REGISTRY.get(sid)
                if node and node.stage_type == StageType.CONDITIONAL:
                    if sid not in ("stage3_reject", "stage4_prime", "stage4_minor", "stage4_major"):
                        if target_stages is not None and sid not in target_stages:
                            continue
                deps = [t.from_stage for t in self.dag._incoming.get(sid, [])
                        if t.condition is None]
                if all(d in completed for d in deps):
                    ready.append(sid)

            if not ready:
                break

            can_parallel = []
            must_serial = []
            for sid in ready:
                node = self.dag.STAGE_REGISTRY.get(sid)
                if node and node.parallel_groups:
                    can_parallel.append(sid)
                else:
                    must_serial.append(sid)

            for sid in must_serial:
                execution_order.append(sid)
                completed.add(sid)
                node = self.dag.STAGE_REGISTRY.get(sid)
                if node:
                    slot = ExecutionSlot(
                        stage_id=sid,
                        start_min=current_time,
                        end_min=current_time + node.estimated_duration_min,
                        token_cost=node.estimated_token_cost,
                    )
                    plan.execution_slots.append(slot)
                    current_time += node.estimated_duration_min

            if can_parallel:
                groups: dict[str, list[str]] = {}
                for sid in can_parallel:
                    node = self.dag.STAGE_REGISTRY.get(sid)
                    if node and node.parallel_groups:
                        for g in node.parallel_groups:
                            groups.setdefault(g, []).append(sid)
                    else:
                        groups.setdefault("_singleton", []).append(sid)

                max_group_duration = 0
                for gname, gstages in groups.items():
                    group_duration = 0
                    for sid in gstages:
                        execution_order.append(sid)
                        completed.add(sid)
                        node = self.dag.STAGE_REGISTRY.get(sid)
                        if node:
                            dur = node.estimated_duration_min
                            slot = ExecutionSlot(
                                stage_id=sid,
                                start_min=current_time,
                                end_min=current_time + dur,
                                parallel_group=gname,
                                token_cost=node.estimated_token_cost,
                            )
                            plan.execution_slots.append(slot)
                            group_duration = max(group_duration, dur)
                    max_group_duration = max(max_group_duration, group_duration)
                current_time += max_group_duration

        plan.execution_order = execution_order
        plan.total_duration_min = current_time

        total_tokens = 0
        for sid in execution_order:
            node = self.dag.STAGE_REGISTRY.get(sid)
            if node:
                total_tokens += node.estimated_token_cost
        plan.total_token_cost = total_tokens

        plan.suggested_concurrency = max(
            1, min(
                len([s for s in execution_order if self.dag.STAGE_REGISTRY[s].parallel_groups]),
                self.constraints.max_concurrent_stages,
            )
        )

        parallel_duration = plan.total_duration_min
        plan.savings_vs_sequential = {
            "duration_reduction_min": max(0, sequential_duration - parallel_duration),
            "duration_reduction_pct": round(
                (1 - parallel_duration / max(sequential_duration, 1)) * 100, 1
            ) if sequential_duration > 0 else 0,
            "sequential_duration_min": sequential_duration,
            "parallel_duration_min": parallel_duration,
        }

        return plan

    def incremental_plan(self, completed_stages: list[str],
                         available_materials: list[str] | None = None) -> OptimizedPlan:
        entry = auto_detect_entry_stage(available_materials or [])
        target = self.dag.topological_sort(entry)
        remaining = [s for s in target if s not in completed_stages]
        if not remaining:
            plan = OptimizedPlan()
            plan.execution_order = completed_stages
            plan.total_duration_min = 0
            plan.total_token_cost = 0
            return plan
        return self.build_optimized_plan(
            available_materials=available_materials,
            target_stages=remaining,
        )

    def allocate_token_budget(self, plan: OptimizedPlan,
                              total_budget: int = 200_000) -> dict[str, int]:
        if not plan.execution_order:
            return {}

        critical_set = set(plan.critical_path)
        base_per_stage = total_budget // len(plan.execution_order) if plan.execution_order else 0

        allocation: dict[str, int] = {}
        critical_bonus = int(total_budget * 0.15)
        per_critical = critical_bonus // max(len(critical_set), 1)

        for sid in plan.execution_order:
            alloc = base_per_stage
            if sid in critical_set:
                alloc += per_critical
            node = self.dag.STAGE_REGISTRY.get(sid)
            if node and node.stage_type == StageType.MANDATORY:
                alloc = int(alloc * 1.2)
            allocation[sid] = min(alloc, int(total_budget * 0.4))

        remaining = total_budget - sum(allocation.values())
        if remaining > 0 and allocation:
            bonus = remaining // len(allocation)
            for sid in allocation:
                allocation[sid] += bonus

        return allocation

    def analyze_bottleneck(self, plan: OptimizedPlan) -> list[dict[str, Any]]:
        bottlenecks = []
        stage_durations = {}
        for slot in plan.execution_slots:
            dur = slot.end_min - slot.start_min
            stage_durations[slot.stage_id] = dur

        if stage_durations:
            avg_dur = sum(stage_durations.values()) / len(stage_durations)
            for sid, dur in sorted(stage_durations.items(), key=lambda x: -x[1]):
                if dur > avg_dur * 1.5:
                    node = self.dag.STAGE_REGISTRY.get(sid)
                    bottlenecks.append({
                        "stage_id": sid,
                        "name": node.name if node else sid,
                        "duration_min": dur,
                        "above_average_pct": round((dur / avg_dur - 1) * 100, 1),
                        "on_critical_path": sid in plan.critical_path,
                        "recommendation": self._bottleneck_recommendation(sid, dur),
                    })
        return bottlenecks

    def _bottleneck_recommendation(self, stage_id: str, duration: int) -> str:
        node = self.dag.STAGE_REGISTRY.get(stage_id)
        if not node:
            return ""
        if node.max_retries > 0:
            return f"Consider reducing max_retries ({node.max_retries}) or parallelizing integrity checks"
        if node.parallel_groups:
            return f"Intra-stage parallelism available in groups: {node.parallel_groups}"
        if node.stage_type == StageType.MANDATORY:
            return "Consider splitting into smaller sub-stages for partial parallelism"
        return "Consider if this stage can run with reduced token budget"
