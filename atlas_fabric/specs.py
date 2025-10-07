from __future__ import annotations
from typing import Dict, Any, List
import yaml

REQUIRED_TARGET_KEYS = ["name","vendor","instance_type","accelerators_per_node","num_nodes",
                        "memory_gb_per_accel","interconnect","host_cpu_arch","numeric",
                        "collectives","max_batch_tokens","container_base","cost"]
REQUIRED_WORKLOAD_KEYS = ["name","framework","model_family","params","parallelism",
                          "sequence","precision","dataloader","duration_minutes","track"]
OPTIONAL_WORKLOAD_KEYS = {"runtime_tuning"}

def _load_yaml(path:str):
    with open(path) as f:
        return yaml.safe_load(f)

def load_target(path:str, name:str):
    data = _load_yaml(path)
    if isinstance(data, list):
        candidates = [t for t in data if t.get("name")==name]
        if not candidates:
            raise KeyError(f"target {name} not found in {path}")
        t = candidates[0]
    else:
        t = data[name]
    for k in REQUIRED_TARGET_KEYS:
        if k not in t:
            raise KeyError(f"target missing key {k}")
    return t

def load_workload(path:str, name:str):
    data = _load_yaml(path)
    if isinstance(data, list):
        candidates = [w for w in data if w.get("name")==name]
        if not candidates:
            raise KeyError(f"workload {name} not found in {path}")
        w = candidates[0]
    else:
        w = data[name]
    for k in REQUIRED_WORKLOAD_KEYS:
        if k not in w:
            raise KeyError(f"workload missing key {k}")
    for k in w:
        if k not in REQUIRED_WORKLOAD_KEYS and k not in OPTIONAL_WORKLOAD_KEYS:
            continue
    return w
