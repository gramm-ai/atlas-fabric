from __future__ import annotations
from typing import Dict, Any
from copy import deepcopy
from .simulate import simulate_run

DEFAULT_SLA_P99_MS = 120.0

def optimize(
    target: Dict[str, Any],
    workload: Dict[str, Any],
    base_knobs: Dict[str, Any],
    seed: int,
    sla_p99_ms: float = DEFAULT_SLA_P99_MS,
    hardware_path: str | None = None,
) -> Dict[str, Any]:
    """Hill-climb simple knobs to improve tokens/sec while p99 <= SLA (if inference)."""
    knobs = deepcopy(base_knobs)
    best = simulate_run(target, workload, knobs, seed, hardware_path=hardware_path)
    best_knobs = deepcopy(knobs)

    candidates = [
        {"cuda_graphs": True},
        {"gpudirect_rdma": True},
        {"microbatch": max(1, base_knobs.get("microbatch",1)*2)},
        {"fabric": "IB"},
        {"virt": "bare"},
        {"microbatch": max(1, base_knobs.get("microbatch",1)*4)},
    ]

    for change in candidates:
        trial = deepcopy(knobs); trial.update(change)
        rec = simulate_run(target, workload, trial, seed+1, hardware_path=hardware_path)
        if _better(rec, best, workload, sla_p99_ms):
            best, best_knobs = rec, deepcopy(trial)

    return {"knobs": best_knobs, "record": best}

def _better(rec:Dict[str,Any], base:Dict[str,Any], wl:Dict[str,Any], sla:float)->bool:
    is_infer = "inference" in wl["name"]
    if is_infer:
        if rec["p99_ms"] > max(sla, 1.0):  # must pass SLA
            return False
    return rec["tokens_per_sec"] > base["tokens_per_sec"] * 1.03  # 3% threshold
