#!/usr/bin/env python3
"""Pipeline DAG modeling — stages as nodes, dependencies as edges.

Models the ARS 10-stage pipeline as a directed acyclic graph (DAG) with
conditional branches, retry loops, and material dependencies. Enables
topological scheduling, critical-path analysis, and parallelism detection.

Usage:
    dag = PipelineDAG()
    order = dag.topological_sort(entry_stage="stage1")
    parallel = dag.parallelizable_stages()
    critical = dag.critical_path()
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class StageStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    SKIPPED = auto()
    BLOCKED = auto()
    FAILED = auto()


class StageType(Enum):
    MANDATORY = auto()
    CONDITIONAL = auto()
    OPTIONAL = auto()


@dataclass
class Material:
    name: str
    produced_by: str
    consumed_by: str
    required: bool


@dataclass
class StageNode:
    id: str
    name: str
    stage_type: StageType = StageType.MANDATORY
    estimated_token_cost: int = 0
    estimated_duration_min: int = 0
    max_retries: int = 0
    parallel_groups: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)


@dataclass
class StageTransition:
    from_stage: str
    to_stage: str
    condition: str | None = None
    weight: float = 1.0


@dataclass
class ResourceConstraint:
    max_concurrent_stages: int = 3
    max_token_budget: int = 200_000
    api_rate_limit_rps: float = 10.0


class PipelineDAG:
    STAGE_REGISTRY: dict[str, StageNode] = {
        "stage1": StageNode("stage1", "RESEARCH", StageType.MANDATORY, 40000, 30, 0, ["lit_search", "rq_formulation"]),
        "stage2": StageNode("stage2", "WRITE", StageType.MANDATORY, 80000, 60, 0, ["viz_gen", "arg_builder"]),
        "stage2_5": StageNode("stage2_5", "INTEGRITY_PRE", StageType.MANDATORY, 30000, 20, 3),
        "stage3": StageNode("stage3", "REVIEW", StageType.MANDATORY, 50000, 40, 0),
        "stage3_reject": StageNode("stage3_reject", "REVIEW_REJECT", StageType.CONDITIONAL, 0, 0),
        "stage4_minor": StageNode("stage4_minor", "REVISE_MINOR", StageType.CONDITIONAL, 30000, 30, 0),
        "stage4_major": StageNode("stage4_major", "REVISE_MAJOR", StageType.CONDITIONAL, 50000, 45, 0),
        "stage3_prime": StageNode("stage3_prime", "RE_REVIEW", StageType.CONDITIONAL, 30000, 25, 0),
        "stage4_prime": StageNode("stage4_prime", "RE_REVISE", StageType.CONDITIONAL, 40000, 35, 0),
        "stage4_5": StageNode("stage4_5", "FINAL_INTEGRITY", StageType.MANDATORY, 30000, 20, 3),
        "stage5": StageNode("stage5", "FINALIZE", StageType.MANDATORY, 20000, 15, 0),
        "stage6": StageNode("stage6", "PROCESS_SUMMARY", StageType.OPTIONAL, 5000, 5, 0),
    }

    MATERIALS: list[Material] = [
        Material("RQ Brief", "stage1", "stage2", False),
        Material("Bibliography", "stage1", "stage2", False),
        Material("Synthesis Report", "stage1", "stage2", False),
        Material("Paper Draft", "stage2", "stage2_5", True),
        Material("Integrity Report", "stage2_5", "stage3", True),
        Material("Verified Paper", "stage2_5", "stage3", True),
        Material("Review Reports", "stage3", "stage4_minor", True),
        Material("Review Reports", "stage3", "stage4_major", True),
        Material("Editorial Decision", "stage3", "stage4_minor", True),
        Material("Revision Roadmap", "stage3", "stage4_minor", True),
        Material("Revised Draft", "stage4_minor", "stage3_prime", True),
        Material("Response to Reviewers", "stage4_minor", "stage3_prime", False),
        Material("Re-Review Report", "stage3_prime", "stage4_prime", True),
        Material("Re-Revised Draft", "stage4_prime", "stage4_5", True),
        Material("Final Integrity Report", "stage4_5", "stage5", True),
        Material("Final Paper", "stage5", "stage6", True),
    ]

    def __init__(self, constraints: ResourceConstraint | None = None):
        self.constraints = constraints or ResourceConstraint()
        self._build_graph()

    def _build_graph(self):
        self._outgoing: dict[str, list[StageTransition]] = defaultdict(list)
        self._incoming: dict[str, list[StageTransition]] = defaultdict(list)
        self._all_stages: set[str] = set()

        for sid in self.STAGE_REGISTRY:
            self._all_stages.add(sid)

        transitions = [
            StageTransition("stage1", "stage2"),
            StageTransition("stage2", "stage2_5"),
            StageTransition("stage2_5", "stage3", "integrity_pass"),
            StageTransition("stage2_5", "stage2_5", "integrity_fail"),
            StageTransition("stage3", "stage3_reject", "decision_reject"),
            StageTransition("stage3", "stage4_5", "decision_accept"),
            StageTransition("stage3", "stage4_minor", "decision_minor"),
            StageTransition("stage3", "stage4_major", "decision_major"),
            StageTransition("stage4_minor", "stage3_prime"),
            StageTransition("stage4_major", "stage3_prime"),
            StageTransition("stage3_prime", "stage4_5", "re_review_accept"),
            StageTransition("stage3_prime", "stage4_prime", "re_review_major"),
            StageTransition("stage4_prime", "stage4_5"),
            StageTransition("stage4_5", "stage5", "integrity_pass"),
            StageTransition("stage4_5", "stage4_5", "integrity_fail"),
            StageTransition("stage5", "stage6"),
        ]
        for t in transitions:
            self._outgoing[t.from_stage].append(t)
            self._incoming[t.to_stage].append(t)

    @property
    def stages(self) -> list[str]:
        return sorted(self._all_stages, key=lambda s: self.STAGE_REGISTRY[s].estimated_token_cost, reverse=True)

    def dependencies(self, stage_id: str) -> list[str]:
        return [t.from_stage for t in self._incoming.get(stage_id, [])
                if t.from_stage != stage_id]

    def dependents(self, stage_id: str) -> list[str]:
        return [t.to_stage for t in self._outgoing.get(stage_id, [])]

    def is_conditional(self, stage_id: str) -> bool:
        return self.STAGE_REGISTRY[stage_id].stage_type == StageType.CONDITIONAL

    def topological_sort(self, entry_stage: str = "stage1") -> list[str]:
        in_degree: dict[str, int] = {}
        for sid in self._all_stages:
            deps = set()
            for t in self._incoming[sid]:
                if t.from_stage != sid:
                    deps.add(t.from_stage)
            in_degree[sid] = len(deps)

        ready = deque([s for s in self._all_stages if in_degree.get(s, 0) == 0])
        if entry_stage != "stage1" and entry_stage in self._all_stages:
            ready = deque([entry_stage])

        result = []
        while ready:
            node = ready.popleft()
            result.append(node)
            for t in self._outgoing[node]:
                if t.to_stage != node:
                    if t.to_stage in in_degree:
                        in_degree[t.to_stage] -= 1
                        if in_degree[t.to_stage] == 0:
                            ready.append(t.to_stage)
        return result

    def parallelizable_stages(self) -> list[set[str]]:
        layers: list[set[str]] = []
        visited: set[str] = set()
        remaining = set(self._all_stages)

        while remaining:
            current_layer: set[str] = set()
            for sid in remaining:
                deps = set()
                for t in self._incoming[sid]:
                    if t.from_stage != sid:
                        deps.add(t.from_stage)
                if deps.issubset(visited) or not deps:
                    current_layer.add(sid)
            if not current_layer:
                break
            layers.append(current_layer)
            visited.update(current_layer)
            remaining -= current_layer
        return layers

    def critical_path(self) -> list[str]:
        dist: dict[str, int] = {s: 0 for s in self._all_stages}
        prev: dict[str, str | None] = {s: None for s in self._all_stages}

        topo = self.topological_sort()
        for node in topo:
            for t in self._outgoing[node]:
                if t.from_stage == t.to_stage:
                    continue
                cost = self.STAGE_REGISTRY[t.to_stage].estimated_token_cost
                new_dist = dist[node] + cost + self.STAGE_REGISTRY[node].estimated_token_cost
                if new_dist > dist[t.to_stage]:
                    dist[t.to_stage] = new_dist
                    prev[t.to_stage] = node

        end = max(self._all_stages, key=lambda s: dist[s])
        path = []
        visited = set()
        while end is not None:
            if end in visited:
                break
            visited.add(end)
            path.append(end)
            end = prev[end]
        return list(reversed(path))

    def material_dependencies(self, stage_id: str) -> list[Material]:
        return [m for m in self.MATERIALS if m.consumed_by == stage_id]

    def material_produced(self, stage_id: str) -> list[Material]:
        return [m for m in self.MATERIALS if m.produced_by == stage_id]

    def intra_stage_parallel_groups(self, stage_id: str) -> list[str]:
        node = self.STAGE_REGISTRY.get(stage_id)
        return node.parallel_groups if node else []

    def schedule(self, entry_stage: str = "stage1",
                 constraints: ResourceConstraint | None = None) -> list[dict[str, Any]]:
        c = constraints or self.constraints
        topo = self.topological_sort(entry_stage)
        schedule: list[dict[str, Any]] = []
        current_token_cost = 0
        active_stages = 0

        for sid in topo:
            node = self.STAGE_REGISTRY[sid]
            if node.stage_type == StageType.CONDITIONAL and sid not in ("stage3_reject", "stage4_prime"):
                continue

            can_run = True
            for dep in self.dependencies(sid):
                if not any(item["stage_id"] == dep and item["status"] == StageStatus.COMPLETED
                           for item in schedule):
                    can_run = False
                    break

            stage_entry = {
                "stage_id": sid,
                "name": node.name,
                "type": node.stage_type,
                "status": StageStatus.PENDING,
                "estimated_cost": node.estimated_token_cost,
                "estimated_duration": node.estimated_duration_min,
                "can_parallel": False,
                "parallel_groups": node.parallel_groups,
            }

            if can_run:
                if active_stages < c.max_concurrent_stages and current_token_cost + node.estimated_token_cost <= c.max_token_budget:
                    stage_entry["status"] = StageStatus.IN_PROGRESS
                    stage_entry["can_parallel"] = True
                    current_token_cost += node.estimated_token_cost
                    active_stages += 1
                else:
                    stage_entry["status"] = StageStatus.PENDING

            schedule.append(stage_entry)

        total_cost = sum(s["estimated_cost"] for s in schedule if s["status"] == StageStatus.IN_PROGRESS)
        return schedule

    def cost_estimate(self, stages: list[str] | None = None,
                      with_retries: bool = True) -> dict[str, Any]:
        target = stages or list(self._all_stages)
        total_tokens = 0
        total_duration = 0
        total_retries = 0
        retry_overhead = 0

        for sid in target:
            node = self.STAGE_REGISTRY[sid]
            total_tokens += node.estimated_token_cost
            total_duration += node.estimated_duration_min
            if with_retries and node.max_retries > 0:
                retry_overhead += node.max_retries * node.estimated_duration_min * 0.5
                total_retries += node.max_retries

        return {
            "stages": target,
            "total_token_cost_estimate": total_tokens,
            "total_duration_min": total_duration,
            "retry_overhead_min": retry_overhead,
            "max_retries": total_retries,
            "recommended_budget_tokens": total_tokens + total_retries * 10000,
        }

    def to_dict(self) -> dict[str, Any]:
        edges = []
        for src in self._outgoing:
            for t in self._outgoing[src]:
                edges.append({
                    "from": t.from_stage,
                    "to": t.to_stage,
                    "condition": t.condition,
                })
        return {
            "nodes": {sid: {
                "name": n.name,
                "type": n.stage_type.name,
                "estimated_token_cost": n.estimated_token_cost,
                "estimated_duration_min": n.estimated_duration_min,
                "max_retries": n.max_retries,
                "parallel_groups": n.parallel_groups,
            } for sid, n in self.STAGE_REGISTRY.items()},
            "edges": edges,
            "materials": [
                {"name": m.name, "produced_by": m.produced_by,
                 "consumed_by": m.consumed_by, "required": m.required}
                for m in self.MATERIALS
            ],
        }


def auto_detect_entry_stage(available_materials: list[str]) -> str:
    stage_markers = {
        "stage5": ["Final Paper"],
        "stage4_5": ["Final Integrity Report"],
        "stage4_minor": ["Revised Draft"],
        "stage4_major": ["Revised Draft"],
        "stage3": ["Review Reports", "Editorial Decision"],
        "stage2_5": ["Integrity Report", "Verified Paper"],
        "stage2": ["Paper Draft"],
        "stage1": ["RQ Brief", "Bibliography", "Synthesis Report"],
    }
    for stage_id in ("stage5", "stage4_5", "stage4_minor",
                      "stage3", "stage2_5", "stage2", "stage1"):
        markers = stage_markers[stage_id]
        for m in markers:
            if m in available_materials:
                return stage_id
    return "stage1"
