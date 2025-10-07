from __future__ import annotations
import argparse, json, os
from .specs import load_target, load_workload
from .adapter import Adapter
from .optimizer import optimize
from .util import save_json, now_iso

def pick_adapter(target):
    return Adapter(target)

def cmd_run(args):
    target = load_target(args.targets, args.target)
    workload = load_workload(args.workloads, args.workload)
    workload["track"] = args.track
    adapter = pick_adapter(target)
    adapter.prepare(workload)
    runtime_tuning = workload.get("runtime_tuning") or {}
    base_defaults = {
        "microbatch": 1,
        "cuda_graphs": False,
        "gpudirect_rdma": False,
        "virt": "bare",
        "fabric": "IB",
    }
    knobs = base_defaults.copy()
    knobs.update(runtime_tuning)
    if args.microbatch is not None:
        knobs["microbatch"] = args.microbatch
    if args.cuda_graphs is not None:
        knobs["cuda_graphs"] = args.cuda_graphs
    if args.gpudirect is not None:
        knobs["gpudirect_rdma"] = args.gpudirect
    if args.virt is not None:
        knobs["virt"] = args.virt
    if args.fabric is not None:
        knobs["fabric"] = args.fabric
    rec = adapter.run_once(workload, knobs, seed=args.seed, hardware_path=args.hardware)
    hourly = float(target["cost"]["hourly_usd"]); pue=float(target["cost"]["pue"]); node_power=rec["node_power_w"]; kwh=0.12
    energy_cost_hr = (node_power * pue / 1000.0) * kwh
    rec2 = {
        "ts": now_iso(),
        "target": target,
        "workload": workload,
        "track": args.track,
        "knobs": knobs,
        "record": rec,
        "dollar_per_1k_tokens": (hourly + energy_cost_hr)/max(1e-6, rec["tokens_per_sec"]*3.6),
        "label": "pre-opt"
    }
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.out}/{args.target}_{args.workload}_{args.track}_{args.seed}.json"
    save_json(name, rec2)
    save_json(f"{args.out}/last_run.json", rec2)
    print("Wrote", name)

def cmd_optimize(args):
    rec = json.load(open(args.record))
    target = rec["target"]; workload = rec["workload"]; base_knobs = rec["knobs"]
    result = optimize(target, workload, base_knobs, seed=args.seed, sla_p99_ms=args.sla, hardware_path=args.hardware)
    out = {
        "ts": now_iso(), "target": target, "workload": workload, "track": rec.get("track","parity"),
        "knobs": result["knobs"], "record": result["record"], "label": "post-opt"
    }
    hourly = float(target["cost"]["hourly_usd"]); pue=float(target["cost"]["pue"]); node_power=out["record"]["node_power_w"]; kwh=0.12
    energy_cost_hr = (node_power * pue / 1000.0) * kwh
    out["dollar_per_1k_tokens"] = (hourly + energy_cost_hr)/max(1e-6, out["record"]["tokens_per_sec"]*3.6)
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.out}/optimized_{target['name']}_{workload['name']}_{args.seed}.json"
    save_json(name, out)
    save_json(f"{args.out}/last_run.json", out)
    print("Optimized and wrote", name)

def cmd_report(args):
    from .reporter import aggregate, render_markdown
    summary = aggregate(args.records)
    render_markdown(summary, args.out)
    print("Report at", args.out)

def main():
    ap = argparse.ArgumentParser(prog="atlas_fabric")
    sub = ap.add_subparsers()

    ap_run = sub.add_parser("run", help="Run a workload on a target (simulated)")
    ap_run.add_argument("--targets", required=True)
    ap_run.add_argument("--target", required=True)
    ap_run.add_argument("--workloads", required=True)
    ap_run.add_argument("--workload", required=True)
    ap_run.add_argument("--track", choices=["parity","best"], default="parity")
    def mark_override(parser_attr: str):
        def _setter(ns: argparse.Namespace, value, option_string=None):
            setattr(ns, parser_attr, True)
            return value
        return _setter

    ap_run.add_argument("--microbatch", type=int, default=None,
                        help="Override microbatch size")
    ap_run.add_argument("--cuda_graphs", action="store_true", default=None,
                        help="Force-enable CUDA Graphs")
    ap_run.add_argument("--gpudirect", action="store_true", default=None,
                        help="Force-enable GPUDirect RDMA")
    ap_run.add_argument("--virt", choices=["bare","sriov"], default=None,
                        help="Override virtualization mode")
    ap_run.add_argument("--fabric", choices=["IB","RoCE","Vendor"], default=None,
                        help="Override interconnect fabric setting")
    ap_run.add_argument("--seed", type=int, default=1337)
    ap_run.add_argument("--hardware", default="accelerators")
    ap_run.add_argument("--out", default="out")
    ap_run.set_defaults(func=cmd_run)

    ap_opt = sub.add_parser("optimize", help="Run optimizer based on last run")
    ap_opt.add_argument("--record", required=True)
    ap_opt.add_argument("--sla", type=float, default=120.0)
    ap_opt.add_argument("--seed", type=int, default=2025)
    ap_opt.add_argument("--hardware", default="accelerators")
    ap_opt.add_argument("--out", default="out")
    ap_opt.set_defaults(func=cmd_optimize)

    ap_rep = sub.add_parser("report", help="Aggregate runs into a markdown report")
    ap_rep.add_argument("--records", default="out")
    ap_rep.add_argument("--out", default="out/report.md")
    ap_rep.set_defaults(func=cmd_report)

    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
