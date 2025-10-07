from __future__ import annotations
from typing import Dict, Any
import os
import yaml

DEFAULT_MODEL: Dict[str, Any] = {
    "compute": {
        "base_tokens_per_accel": 1000.0,
        "factors": {"training": 1.0, "inference": 1.0},
    },
    "utilization": {
        "base": 0.55,
        "knob_increments": {"cuda_graphs": 0.1, "gpudirect_rdma": 0.05, "inference_bonus": 0.1},
        "max": 0.9,
    },
    "communication": {
        "intensity": {"base": 0.12, "per_tp": 0.02, "per_pp": 0.015, "inference_scale": 0.6},
        "fabric_factors": {"IB": 1.0, "RoCE": 1.12, "Vendor": 1.05},
        "jitter_per_sriov": 0.05,
        "gpudirect_multiplier": 0.5,
        "min_overhead": 0.02,
    },
    "throughput": {
        "overhead_weight": 8.0,
    },
    "latency": {
        "min_p50_ms": 5.0,
        "base_p50_coeff": 1000.0,
        "tail": {"p95_multiplier": 1.5, "p99_multiplier": 2.2, "sriov_tail": 0.2, "comm_tail": 0.8},
    },
    "memory": {"peak_tokens_factor": 0.001},
    "hbm": {"base": 0.5, "seq_scale": 0.5, "seq_norm": 8192.0, "max": 0.95},
    "noise": {"amplitude": 0.03},
    "retries": {"overhead_threshold": 0.3},
}


def _safe_load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
        return data or {}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name)


def load_hardware_model(hardware_path: str, target: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a hardware performance model for a given target.

    Resolution order (first match wins):
    1) File named after target name (normalized) in directory: <hardware_path>/<normalized_target>.yaml
    2) File named after vendor (lowercased) in directory: <hardware_path>/<vendor>.yaml
    3) If hardware_path is a file: use that single file
    4) Fallback to DEFAULT_MODEL
    """
    if not hardware_path:
        return DEFAULT_MODEL

    if os.path.isdir(hardware_path):
        normalized_target = _normalize_name(target.get("name", ""))
        vendor = target.get("vendor", "").lower()

        by_target = os.path.join(hardware_path, f"{normalized_target}.yaml")
        if os.path.exists(by_target):
            return _merge_dict(DEFAULT_MODEL, _safe_load_yaml(by_target))

        by_vendor = os.path.join(hardware_path, f"{vendor}.yaml")
        if os.path.exists(by_vendor):
            return _merge_dict(DEFAULT_MODEL, _safe_load_yaml(by_vendor))

        # Try any yaml in dir that has a matching 'match' section
        for fname in os.listdir(hardware_path):
            if not fname.lower().endswith((".yaml", ".yml")):
                continue
            path = os.path.join(hardware_path, fname)
            data = _safe_load_yaml(path)
            match = data.get("match", {})
            if match.get("name") == target.get("name") or \
               (match.get("vendor") or "").lower() == vendor:
                return _merge_dict(DEFAULT_MODEL, data)

        return DEFAULT_MODEL

    # hardware_path is a file
    try:
        return _merge_dict(DEFAULT_MODEL, _safe_load_yaml(hardware_path))
    except Exception:
        return DEFAULT_MODEL


