#!/usr/bin/env python3
"""CLI entry point for DAG-based pipeline optimization.

Run as:  python3 scripts/run_pipeline_optimizer.py <action> [options]

Actions:
  analyze       -- Show DAG structure, critical path, parallelism
  plan          -- Generate optimized execution plan
  incremental   -- Plan from mid-entry (specify completed stages)
  bottleneck    -- Analyze bottlenecks given completed stages
  cost-estimate -- Show total token/duration cost estimate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from pipeline_dag import PipelineDAG, ResourceConstraint, auto_detect_entry_stage
    from pipeline_scheduler import OptimizedPipelineScheduler
except ImportError:
    from scripts.pipeline_dag import PipelineDAG, ResourceConstraint, auto_detect_entry_stage
    from scripts.pipeline_scheduler import OptimizedPipelineScheduler


def cmd_analyze(args: list[str]) -> int:
    dag = PipelineDAG()
    entry = args[0] if args else "stage1"
    order = dag.topological_sort(entry)
    parallel = dag.parallelizable_stages()
    critical = dag.critical_path()
    conditional = [s for s in dag.stages if dag.is_conditional(s)]
    mandatory = [s for s in dag.stages if not dag.is_conditional(s)]
    total_tokens = sum(dag.STAGE_REGISTRY[s].estimated_token_cost for s in mandatory)
    total_duration = sum(dag.STAGE_REGISTRY[s].estimated_duration_min for s in mandatory)

    out = {
        "entry_stage": entry,
        "total_stages": len(dag.stages),
        "topological_order": order,
        "critical_path": critical,
        "critical_path_duration_min": sum(
            dag.STAGE_REGISTRY[s].estimated_duration_min for s in critical if s in dag.STAGE_REGISTRY
        ),
        "mandatory_stages": mandatory,
        "conditional_stages": conditional,
        "parallelizable_layers": [
            sorted(stages) for stages in parallel
        ],
        "total_token_estimate": total_tokens,
        "total_duration_estimate_min": total_duration,
        "stage_details": [
            {
                "id": sid,
                "name": node.name,
                "type": node.stage_type.name,
                "tokens": node.estimated_token_cost,
                "duration_min": node.estimated_duration_min,
                "max_retries": node.max_retries,
                "parallel_groups": node.parallel_groups,
            }
            for sid, node in sorted(dag.STAGE_REGISTRY.items())
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


def cmd_plan(args: list[str]) -> int:
    scheduler = OptimizedPipelineScheduler()
    materials = args or []
    plan = scheduler.build_optimized_plan(available_materials=materials)

    # Token budget allocation
    budget = scheduler.allocate_token_budget(plan, total_budget=200_000)
    bottleneck = scheduler.analyze_bottleneck(plan)

    out = {
        "execution_order": plan.execution_order,
        "parallel_stages": [sorted(s) for s in plan.parallel_stages],
        "total_duration_min": plan.total_duration_min,
        "total_token_cost": plan.total_token_cost,
        "suggested_concurrency": plan.suggested_concurrency,
        "critical_path": plan.critical_path,
        "savings_vs_sequential": plan.savings_vs_sequential,
        "execution_slots": [
            {"stage": s.stage_id, "start": s.start_min, "end": s.end_min,
             "group": s.parallel_group, "tokens": s.token_cost}
            for s in plan.execution_slots
        ],
        "token_allocation": budget,
        "bottlenecks": [
            {"stage": b["stage_id"], "duration_min": b["duration_min"],
             "recommendation": b["recommendation"]}
            for b in bottleneck
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


def cmd_incremental(args: list[str]) -> int:
    if not args:
        print('Usage: incremental <completed_stage1> [completed_stage2 ...]', file=sys.stderr)
        return 1
    completed = args
    scheduler = OptimizedPipelineScheduler()
    plan = scheduler.incremental_plan(completed_stages=set(completed))
    out = {
        "completed_stages": completed,
        "next_stages": plan.execution_order,
        "total_duration_min": plan.total_duration_min,
        "total_token_cost": plan.total_token_cost,
        "suggested_concurrency": plan.suggested_concurrency,
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


def cmd_bottleneck(args: list[str]) -> int:
    scheduler = OptimizedPipelineScheduler()
    completed = set(args)
    plan = scheduler.incremental_plan(completed_stages=completed) if completed else scheduler.build_optimized_plan()
    bottlenecks = scheduler.analyze_bottleneck(plan)
    out = {
        "bottlenecks": [
            {"stage": b["stage_id"], "duration_min": b["duration_min"],
             "recommendation": b["recommendation"]}
            for b in bottlenecks
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


def cmd_cost_estimate(args: list[str]) -> int:
    dag = PipelineDAG()
    total_tokens = sum(n.estimated_token_cost for n in dag.STAGE_REGISTRY.values())
    total_duration = sum(n.estimated_duration_min for n in dag.STAGE_REGISTRY.values())
    estimated_cost_usd = (total_tokens / 1_000_000) * 15.0
    out = {
        "total_tokens": total_tokens,
        "total_duration_min": total_duration,
        "estimated_cost_usd": round(estimated_cost_usd, 2),
        "by_stage": [
            {"stage": sid, "name": n.name, "tokens": n.estimated_token_cost,
             "duration_min": n.estimated_duration_min}
            for sid, n in sorted(dag.STAGE_REGISTRY.items())
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    if not argv:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__, file=sys.stderr)
        return 0

    action = argv[0]
    rest = argv[1:]

    dispatch = {
        "analyze": cmd_analyze,
        "plan": cmd_plan,
        "incremental": cmd_incremental,
        "bottleneck": cmd_bottleneck,
        "cost-estimate": cmd_cost_estimate,
    }
    fn = dispatch.get(action)
    if fn is None:
        print(f"Unknown action: {action!r}. Actions: {', '.join(dispatch)}", file=sys.stderr)
        return 1
    return fn(rest)


if __name__ == "__main__":
    sys.exit(main())
