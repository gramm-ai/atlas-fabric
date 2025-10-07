from __future__ import annotations
from typing import Dict, Any
from .util import deterministic_rand
from .hardware import load_hardware_model

def simulate_run(target:Dict[str,Any], workload:Dict[str,Any], knobs:Dict[str,Any], seed:int, hardware_model:Dict[str,Any]|None=None, hardware_path:str|None=None) -> Dict[str,Any]:
    rnd = deterministic_rand(seed)

    vendor = target["vendor"].lower()
    accels = target["accelerators_per_node"] * target["num_nodes"]
    bw_fabric = float(target["interconnect"]["bw_GBps"])
    hourly = float(target["cost"]["hourly_usd"])
    pue = float(target["cost"]["pue"])
    node_power = float(target["cost"]["energy_watts_node"])

    # Load hardware model (defaults/vendor/target overrides)
    if hardware_model is None:
        hw_path = hardware_path
        if hw_path is None:
            raise KeyError("hardware_path must be provided")
        hardware_model = load_hardware_model(hw_path, target)

    params = workload["params"]
    hidden = int(params["hidden_size"])
    layers = int(params["n_layers"])
    seq = int(workload["sequence"]["prompt"]) + int(workload["sequence"]["generate"])
    duration_min = int(workload["duration_minutes"])
    is_infer = "inference" in workload["name"]

    microbatch = int(knobs["microbatch"])
    tp = int(workload["parallelism"]["tp"])
    pp = int(workload["parallelism"]["pp"])
    dp = int(workload["parallelism"]["dp"])
    cuda_graphs = 1 if bool(knobs["cuda_graphs"]) else 0
    gpudirect = 1 if bool(knobs["gpudirect_rdma"]) else 0
    sriov = 1 if knobs["virt"] == "sriov" else 0
    fabric = knobs["fabric"]

    tokens_per_step = max(1, dp * microbatch * seq)
    # Compute capability factor from model
    compute_cfg = hardware_model["compute"]
    base_tokens_per_accel = float(compute_cfg["base_tokens_per_accel"])
    factors = compute_cfg["factors"]
    mode_key = "inference" if is_infer else "training"
    vendor_factor = float(factors[mode_key])

    # Utilization model
    util_cfg = hardware_model["utilization"]
    util = float(util_cfg["base"])
    inc = util_cfg["knob_increments"]
    util += float(inc["cuda_graphs"]) * cuda_graphs
    util += float(inc["gpudirect_rdma"]) * gpudirect
    if is_infer:
        util += float(inc["inference_bonus"])
    util = min(util, float(util_cfg["max"]))

    # Communication model
    comm_cfg = hardware_model["communication"]
    intensity = comm_cfg["intensity"]
    comm_intensity = float(intensity["base"]) + float(intensity["per_tp"])*(tp-1) + float(intensity["per_pp"])*(pp-1)
    if is_infer:
        comm_intensity *= float(intensity["inference_scale"])
    fabric_factors = comm_cfg["fabric_factors"]
    fabric_factor = float(fabric_factors[fabric])
    jitter_penalty = 1.0 + float(comm_cfg["jitter_per_sriov"])*sriov

    gpudirect_mult = float(comm_cfg["gpudirect_multiplier"]) if gpudirect else 1.0
    comm_overhead = comm_intensity * fabric_factor * jitter_penalty * gpudirect_mult
    comm_overhead = max(float(comm_cfg["min_overhead"]), comm_overhead)

    # Throughput model
    overhead_weight = float(hardware_model["throughput"]["overhead_weight"])
    comm_penalty_coeff = float(compute_cfg["comm_penalty_coeff"])
    compute_rate = vendor_factor * util * accels * base_tokens_per_accel
    tokens_sec = compute_rate / (1.0 + overhead_weight*comm_overhead)

    # Latency model
    lat_cfg = hardware_model["latency"]
    min_p50 = float(lat_cfg["min_p50_ms"])
    base_coeff = float(lat_cfg["base_p50_coeff"])
    tail = lat_cfg["tail"]
    base_p50 = max(min_p50, base_coeff / max(1.0, compute_rate/accels))
    tail_mult = 1.0 + float(tail["comm_tail"])*comm_overhead + float(tail["sriov_tail"])*sriov
    p50 = base_p50
    p95 = p50 * (float(tail["p95_multiplier"])*tail_mult)
    p99 = p50 * (float(tail["p99_multiplier"])*tail_mult)
    ttft_ratio = float(lat_cfg["ttft_ratio"])

    if is_infer and workload["sequence"]["generate"] > 0:
        gen = int(workload["sequence"]["generate"])
        tokens_sec = tokens_sec * (gen / max(1, seq))

    power_cfg = hardware_model["power"]
    idle_fraction = float(power_cfg["idle_fraction"])
    util_fraction = float(power_cfg["util_fraction"])
    node_power_w = node_power * (idle_fraction + util_fraction*util)

    flop_util = util * (1.0 - comm_penalty_coeff*comm_overhead)
    hbm_cfg = hardware_model["hbm"]
    hbm_util = min(float(hbm_cfg["max"]), float(hbm_cfg["base"]) + float(hbm_cfg["seq_scale"])*(seq/float(hbm_cfg["seq_norm"])))

    noise_cfg = hardware_model["noise"]
    noise_amp = float(noise_cfg["amplitude"])
    noise = (rnd.random()-0.5)*noise_amp
    tokens_sec *= (1.0 + noise)
    tail_noise_scale = float(noise_cfg["tail_scale"])
    p95 *= (1.0 + noise*tail_noise_scale)
    p99 *= (1.0 + noise*tail_noise_scale)

    record = {
        "tokens_per_sec": max(1.0, tokens_sec),
        "flop_utilization": max(0.0, min(1.0, flop_util)),
        "hbm_bw_util": max(0.0, min(1.0, hbm_util)),
        "comm_compute_ratio": max(0.01, min(0.99, comm_overhead/(1.0+comm_overhead))),
        "step_time_ms": (tokens_per_step / max(1.0, tokens_sec)) * 1000.0 if not is_infer else 0.0,
        "p50_ms": p50 if is_infer else 0.0,
        "p95_ms": p95 if is_infer else 0.0,
        "p99_ms": p99 if is_infer else 0.0,
        "ttft_ms": p50*ttft_ratio if is_infer else 0.0,
        "peak_mem_gb": min(float(target["memory_gb_per_accel"]), float(hardware_model["memory"]["peak_tokens_factor"])*hidden*layers),
        "node_power_w": node_power_w,
        "retries": 0 if comm_overhead < float(hardware_model["retries"]["overhead_threshold"]) else 1,
        "failures": 0,
        "artifacts": {"logs":"sim://logs","trace":"sim://trace","env":"sim://manifest"}
    }
    return record
