from __future__ import annotations
import os, glob, statistics
from typing import Dict, Any
from .util import load_json

def aggregate(records_dir:str) -> Dict[str,Any]:
    files = sorted(glob.glob(os.path.join(records_dir,"*.json")))
    items = [load_json(f) for f in files if "last_run" not in f]
    groups = {}
    for it in items:
        key = (it["target"]["name"], it["workload"]["name"], it["track"])
        groups.setdefault(key, []).append(it)
    summary = []
    for key, arr in groups.items():
        tname, wname, track = key
        tps = [x["record"]["tokens_per_sec"] for x in arr]
        perfw = [x["record"]["tokens_per_sec"]/max(1.0,x["record"]["node_power_w"]) for x in arr]
        cost = [x.get("dollar_per_1k_tokens", _dptk(x)) for x in arr]
        row = {
            "target": tname, "workload": wname, "track": track,
            "runs": len(arr),
            "tokens_per_sec_mean": round(statistics.mean(tps),2),
            "perf_per_watt_mean": round(statistics.mean(perfw),6),
            "dollar_per_1k_tokens_mean": round(statistics.mean(cost),4),
        }
        summary.append(row)
    return {"rows": summary, "count": len(items)}

def _dptk(rec:Dict[str,Any]):
    hourly = float(rec["target"]["cost"]["hourly_usd"])
    pue = float(rec["target"]["cost"]["pue"])
    node_power = float(rec["record"]["node_power_w"])
    kwh_price = 0.12
    energy_cost_hr = (node_power * pue / 1000.0) * kwh_price
    tps = rec["record"]["tokens_per_sec"]
    return (hourly + energy_cost_hr)/max(1e-6, tps*3.6)

def render_markdown(summary:Dict[str,Any], out_path:str)->None:
    lines = ["# Atlas Fabric Report", "", f"Total runs: {summary['count']}", "", "| Target | Workload | Track | Runs | tokens/sec (mean) | Perf/W (mean) | $/1k tokens (mean) |", "|---|---|---|---:|---:|---:|---:|"]
    for r in sorted(summary["rows"], key=lambda x:(x["workload"],x["track"],x["dollar_per_1k_tokens_mean"])):
        lines.append(f"| {r['target']} | {r['workload']} | {r['track']} | {r['runs']} | {r['tokens_per_sec_mean']} | {r['perf_per_watt_mean']} | {r['dollar_per_1k_tokens_mean']} |")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\\n".join(lines)+"\\n")
