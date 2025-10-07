from __future__ import annotations

from typing import Dict, Any, List, Tuple
from copy import deepcopy
import math

from .simulate import simulate_run


REQUEST_PROFILE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "steady_state": {"arrival_load_factor": 0.7, "arrival_cv": 1.0},
    "peak": {"arrival_load_factor": 0.85, "arrival_cv": 1.1},
    "spike": {"arrival_load_factor": 1.15, "arrival_cv": 1.3},
    "low": {"arrival_load_factor": 0.4, "arrival_cv": 0.8}
}


class Adapter:
    """Generic adapter that delegates to the simulator."""

    def __init__(self, target: Dict[str, Any]):
        self.target = target

    def prepare(self, workload: Dict[str, Any]) -> bool:
        """Hook for any target-specific preparation (noop for simulator)."""
        return True

    def run_once(
        self,
        workload: Dict[str, Any],
        knobs: Dict[str, Any],
        seed: int,
        hardware_path: str | None = None,
    ) -> Dict[str, Any]:
        schedule_cfg = workload.get("schedule") or {}
        phases = schedule_cfg.get("phases") if isinstance(schedule_cfg, dict) else None

        if not phases:
            return simulate_run(self.target, workload, knobs, seed, hardware_path=hardware_path)

        base_workload = {k: deepcopy(v) for k, v in workload.items() if k != "schedule"}
        if not base_workload.get("duration_minutes"):
            base_workload["duration_minutes"] = workload.get("duration_minutes", 5)

        phase_results: List[Dict[str, Any]] = []
        total_weight = 0.0

        for idx, phase in enumerate(phases):
            if not isinstance(phase, dict):
                continue

            phase_workload = deepcopy(base_workload)
            overrides = phase.get("overrides") if isinstance(phase.get("overrides"), dict) else {}
            _deep_merge_dict(phase_workload, overrides)

            duration = float(phase.get("duration_minutes", phase_workload.get("duration_minutes", 0) or 1))
            if duration <= 0:
                duration = float(phase_workload.get("duration_minutes", 1))
            phase_workload["duration_minutes"] = duration

            phase_record = simulate_run(self.target, phase_workload, knobs, seed + idx, hardware_path=hardware_path)
            phase_record = deepcopy(phase_record)
            _apply_phase_scalars(phase, phase_record)

            request_profile = phase.get("request_profile")
            if request_profile is None and isinstance(schedule_cfg, dict):
                request_profile = schedule_cfg.get("default_request_profile")

            phase_queue = _derive_queue_metrics(
                request_profile=request_profile,
                phase=phase,
                record=phase_record,
                target=self.target,
                workload=phase_workload,
                knobs=knobs,
            )
            if phase_queue:
                phase_record["queue_metrics"] = phase_queue
                phase_record.setdefault("queue_adjusted", {})
                phase_record["queue_adjusted"]["p50_ms"] = phase_queue["p50_latency_ms"]
                phase_record["queue_adjusted"]["p95_ms"] = phase_queue["p95_latency_ms"]
                phase_record["queue_adjusted"]["p99_ms"] = phase_queue["p99_latency_ms"]
                phase_record["queue_adjusted"]["ttft_ms"] = phase_queue["ttft_latency_ms"]
                phase_record["queue_adjusted"]["waiting_fraction_p99"] = phase_queue["waiting_fraction_p99"]

            phase_results.append({
                "name": phase.get("name", f"phase_{idx+1}"),
                "weight_minutes": duration,
                "settings": {
                    "tokens_scale": phase.get("tokens_scale"),
                    "autoscale_factor": phase.get("autoscale_factor"),
                    "request_profile": phase.get("request_profile")
                },
                "overrides": overrides,
                "record": phase_record
            })
            total_weight += duration

        if not phase_results:
            return simulate_run(self.target, workload, knobs, seed, hardware_path=hardware_path)

        aggregated = _aggregate_phase_results(phase_results, total_weight)
        aggregated["schedule_breakdown"] = phase_results
        aggregated["schedule_total_minutes"] = total_weight
        aggregated["schedule_effects"] = True
        return aggregated

    def profile(self) -> Dict[str, Any]:
        return {"profiler": "sim"}


def _deep_merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _apply_phase_scalars(phase: Dict[str, Any], record: Dict[str, Any]) -> None:
    tokens_scale = float(phase.get("tokens_scale", 1.0) or 1.0)
    autoscale_factor = float(phase.get("autoscale_factor", 1.0) or 1.0)

    total_scale = tokens_scale * autoscale_factor
    if total_scale != 1.0:
        record["tokens_per_sec"] = max(1.0, record.get("tokens_per_sec", 0.0) * total_scale)
        if record.get("step_time_ms", 0.0):
            record["step_time_ms"] = record["step_time_ms"] / max(total_scale, 1e-6)
        record["node_power_w"] = record.get("node_power_w", 0.0) * autoscale_factor


def _derive_queue_metrics(
    request_profile: Any,
    phase: Dict[str, Any],
    record: Dict[str, Any],
    target: Dict[str, Any],
    workload: Dict[str, Any],
    knobs: Dict[str, Any]
) -> Dict[str, float] | None:
    try:
        is_infer = "inference" in workload.get("name", "")
        generate_tokens = int(workload["sequence"].get("generate", 0))
    except Exception:
        is_infer = False
        generate_tokens = 0

    if not is_infer or generate_tokens <= 0:
        return None

    if not record.get("p50_ms"):
        return None

    profile_key = request_profile.lower() if isinstance(request_profile, str) else None
    profile_cfg = REQUEST_PROFILE_DEFAULTS.get(profile_key, REQUEST_PROFILE_DEFAULTS["steady_state"])

    arrival_cfg = phase.get("arrival") if isinstance(phase.get("arrival"), dict) else {}
    target_rate = float(arrival_cfg.get("requests_per_sec") or 0.0)
    max_dp = int(workload["parallelism"].get("dp", 1))
    dp_override = arrival_cfg.get("dp_override")
    if isinstance(dp_override, int) and dp_override > 0:
        max_dp = dp_override

    tokens_per_req = int(workload["sequence"].get("prompt", 0)) + generate_tokens
    tokens_per_sec = float(record.get("tokens_per_sec", 0.0))
    if tokens_per_sec <= 0 or tokens_per_req <= 0:
        return None

    service_mean = (tokens_per_req / tokens_per_sec) * 1000.0

    util = min(0.99, max(0.01, float(record.get("flop_utilization", 0.5))))

    arrival_rate = target_rate
    if arrival_rate <= 0:
        eff_capacity = tokens_per_sec / max(1, tokens_per_req)
        arrival_rate = eff_capacity * profile_cfg["arrival_load_factor"]

    concurrency_limit = max(1, max_dp * int(knobs.get("microbatch", 1)))

    service_cv = max(0.1, min(5.0, float(arrival_cfg.get("service_cv") or profile_cfg["arrival_cv"])))
    arrival_cv = max(0.1, min(5.0, float(arrival_cfg.get("arrival_cv") or profile_cfg["arrival_cv"])))

    ρ_single = min(0.999, arrival_rate * service_mean / 1000.0)
    if ρ_single <= 0:
        return None

    m = max(1, int(arrival_cfg.get("servers") or concurrency_limit))
    ρ = min(0.999, ρ_single / m)

    if ρ >= 0.99:
        safety = ρ / 0.98
        service_mean *= safety
        ρ = 0.98

    ca2 = arrival_cv ** 2
    cs2 = service_cv ** 2

    if m == 1:
        wq = ((ca2 + cs2) / 2.0) * (ρ / (1.0 - ρ)) * service_mean
    else:
        sqrt_term = math.sqrt(2.0 * (m + 1))
        ca_g = math.sqrt(ca2)
        cs_g = math.sqrt(cs2)
        term = (ca_g + cs_g) / 2.0
        wq = term * (ρ ** (sqrt_term)) / (1.0 - ρ) * service_mean

    p50_service = float(record.get("p50_ms", service_mean))
    p95_service = float(record.get("p95_ms", service_mean * 2.0))
    p99_service = float(record.get("p99_ms", service_mean * 3.0))

    wait_p50 = max(0.0, min(wq, p50_service * 0.5))
    wait_p95 = min(max(wq * 1.5, wait_p50), p95_service * 0.8)
    wait_p99 = min(max(wq * 2.0, wait_p95), p99_service * 0.9)

    p50_latency = p50_service + wait_p50
    p95_latency = p95_service + wait_p95
    p99_latency = p99_service + wait_p99

    ttft_service = float(record.get("ttft_ms", p50_service))
    wait_ttft = wait_p50
    ttft_latency = ttft_service + wait_ttft

    waiting_fraction_p99 = 0.0
    if p99_latency > 0:
        waiting_fraction_p99 = wait_p99 / p99_latency

    return {
        "arrival_rate": arrival_rate,
        "servers": m,
        "service_mean_ms": service_mean,
        "service_cv": float(service_cv),
        "arrival_cv": float(arrival_cv),
        "utilization": ρ,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "ttft_latency_ms": ttft_latency,
        "wait_p50_ms": wait_p50,
        "wait_p95_ms": wait_p95,
        "wait_p99_ms": wait_p99,
        "wait_ttft_ms": wait_ttft,
        "waiting_fraction_p99": waiting_fraction_p99,
    }


def _aggregate_phase_results(phase_results: List[Dict[str, Any]], total_weight: float) -> Dict[str, Any]:
    if total_weight <= 0:
        total_weight = sum(max(1.0, entry.get("weight_minutes", 1.0)) for entry in phase_results)

    def weighted_avg_service(key: str, default: float = 0.0) -> float:
        if total_weight <= 0:
            return default
        accum = 0.0
        for entry in phase_results:
            weight = entry.get("weight_minutes", 0.0)
            value = entry["record"].get(key, default)
            accum += weight * value
        return accum / total_weight if total_weight else default

    def weighted_avg_queue(key: str, default: float = 0.0) -> float:
        if total_weight <= 0:
            return default
        accum = 0.0
        for entry in phase_results:
            weight = entry.get("weight_minutes", 0.0)
            queue_adjusted = entry["record"].get("queue_adjusted") or {}
            value = queue_adjusted.get(key, entry["record"].get(key, default))
            accum += weight * value
        return accum / total_weight if total_weight else default

    def weighted_avg_key(func, default: float = 0.0) -> float:
        if total_weight <= 0:
            return default
        accum = 0.0
        for entry in phase_results:
            weight = entry.get("weight_minutes", 0.0)
            value = func(entry) if func else default
            accum += weight * value
        return accum / total_weight if total_weight else default

    aggregated = {
        "tokens_per_sec": weighted_avg_service("tokens_per_sec"),
        "flop_utilization": weighted_avg_service("flop_utilization"),
        "hbm_bw_util": weighted_avg_service("hbm_bw_util"),
        "comm_compute_ratio": weighted_avg_service("comm_compute_ratio"),
        "step_time_ms": weighted_avg_service("step_time_ms"),
        "node_power_w": weighted_avg_service("node_power_w")
    }

    aggregated["service_latency_ms"] = {
        "p50": weighted_avg_service("p50_ms"),
        "p95": weighted_avg_service("p95_ms"),
        "p99": weighted_avg_service("p99_ms"),
        "ttft": weighted_avg_service("ttft_ms")
    }

    aggregated["queue_latency_ms"] = {
        "p50": weighted_avg_queue("p50_ms"),
        "p95": weighted_avg_queue("p95_ms"),
        "p99": weighted_avg_queue("p99_ms"),
        "ttft": weighted_avg_queue("ttft_ms")
    }

    aggregated["queue_wait_ms"] = {
        "p50": weighted_avg_key(lambda entry: (entry["record"].get("queue_metrics", {}) or {}).get("wait_p50_ms", 0.0)),
        "p95": weighted_avg_key(lambda entry: (entry["record"].get("queue_metrics", {}) or {}).get("wait_p95_ms", 0.0)),
        "p99": weighted_avg_key(lambda entry: (entry["record"].get("queue_metrics", {}) or {}).get("wait_p99_ms", 0.0)),
        "ttft": weighted_avg_key(lambda entry: (entry["record"].get("queue_metrics", {}) or {}).get("wait_ttft_ms", 0.0))
    }

    aggregated["queue_utilization"] = weighted_avg_key(lambda entry: (entry["record"].get("queue_metrics", {}) or {}).get("utilization", 0.0))
    aggregated["queue_arrival_rate_rps"] = weighted_avg_key(lambda entry: (entry["record"].get("queue_metrics", {}) or {}).get("arrival_rate", 0.0))
    aggregated["queue_servers"] = weighted_avg_key(lambda entry: float((entry["record"].get("queue_metrics", {}) or {}).get("servers", 0.0)))
    aggregated["queue_wait_fraction_p99"] = weighted_avg_key(lambda entry: (entry["record"].get("queue_metrics", {}) or {}).get("waiting_fraction_p99", 0.0))

    aggregated["p50_ms"] = aggregated["queue_latency_ms"]["p50"]
    aggregated["p95_ms"] = aggregated["queue_latency_ms"]["p95"]
    aggregated["p99_ms"] = aggregated["queue_latency_ms"]["p99"]
    aggregated["ttft_ms"] = aggregated["queue_latency_ms"]["ttft"]

    aggregated["peak_mem_gb"] = max(entry["record"].get("peak_mem_gb", 0.0) for entry in phase_results)
    aggregated["retries"] = sum(int(entry["record"].get("retries", 0)) for entry in phase_results)
    aggregated["failures"] = sum(int(entry["record"].get("failures", 0)) for entry in phase_results)
    aggregated["artifacts"] = phase_results[0]["record"].get("artifacts", {}) if phase_results else {}
    aggregated["hbm_bw_util"] = min(1.0, aggregated.get("hbm_bw_util", 0.0))
    aggregated["flop_utilization"] = min(1.0, aggregated.get("flop_utilization", 0.0))

    return aggregated

