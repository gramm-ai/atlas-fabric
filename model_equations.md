# Atlas Fabric Simulation Model Equations

## 1. Purpose and Structure

This document consolidates the mathematical formulation used by the Atlas Fabric simulator to estimate training and inference performance, latency, energy, and cost. Each section introduces the relevant symbols before presenting the equations so that the model can be understood progressively. Scheduling effects (multi-phase workloads) and parameter assumptions are documented explicitly.

## 2. Notation and Inputs

The simulator consumes three structured inputs:

- Hardware target `T` (e.g., accelerator count, power draw)
- Workload profile `W` (e.g., model size, sequence lengths, schedules)
- Control knobs `K` (e.g., micro-batch size, CUDA Graphs)

From these, the simulator derives the following symbols (per run):

| Symbol | Description |
| --- | --- |
| `N` | Total accelerators = `accelerators_per_node * num_nodes` |
| `B_base` | Base token rate per accelerator from hardware profile |
| `f_vendor(mode)` | Vendor- or mode-specific compute multiplier (`training` or `inference`) |
| `rho_base` | Base utilization factor from hardware profile |
| `delta_rho_cuda`, `delta_rho_gpudirect`, `delta_rho_infer` | Utilization boosts from knobs / workload mode |
| `chi_tp`, `chi_pp` | Communication intensities per tensor/pipeline-parallel stage |
| `gamma_fabric`, `gamma_jitter`, `gamma_gpudirect` | Communication penalties due to interconnect, virtualization, GPUDirect |
| `omega` | Throughput overhead weight |
| `kappa_comm` | FLOP penalty coefficient used to infer FP utilization |
| `lat_min`, `lat_base_coeff`, `tail_comm`, `tail_sriov` | Latency configuration coefficients |
| `p_node_nominal` | Nominal node power from target spec |
| `phi_idle`, `phi_util` | Idle and utilization-linked fractions of node power |
| `sigma_noise`, `sigma_tail` | Random jitter amplitudes |
| `step_tokens` | Tokens per training step (`dp * microbatch * (prompt + generate)`) |

All parameters above are required in the YAML specifications; the simulator raises if any field is missing. Workload-level parameters (hidden size, layers, sequence lengths) also feed the memory headroom model.

## 3. Compute and Communication Throughput

### 3.1 Utilization and Communication Overhead

Utilization incorporates base hardware efficiency plus additive knob effects:

`rho = min(rho_base + delta_rho_cuda * cuda_graphs + delta_rho_gpudirect * gpudirect + delta_rho_infer * indicator(mode == "inference"), rho_max)`  
```52:58:atlas_fabric/simulate.py
util = float(util_cfg["base"])
inc = util_cfg["knob_increments"]
util += float(inc["cuda_graphs"]) * cuda_graphs
util += float(inc["gpudirect_rdma"]) * gpudirect
if is_infer:
    util += float(inc["inference_bonus"])
util = min(util, float(util_cfg["max"]))
```

Communication intensity scales with tensor (`tp`) and pipeline (`pp`) parallel degrees and is attenuated for inference workloads:

`chi = (chi_base + chi_tp * (tp - 1) + chi_pp * (pp - 1)) * (chi_infer_scale if mode == "inference" else 1)`  
```60:65:atlas_fabric/simulate.py
comm_intensity = float(intensity["base"]) + float(intensity["per_tp"])*(tp-1) + float(intensity["per_pp"])*(pp-1)
if is_infer:
    comm_intensity *= float(intensity["inference_scale"])
```

The end-to-end communication overhead factor combines network, jitter, and GPUDirect multipliers:

`omega_comm = max(omega_min, chi * gamma_fabric * gamma_jitter * gamma_gpudirect)`  
```66:73:atlas_fabric/simulate.py
fabric_factors = comm_cfg["fabric_factors"]
fabric_factor = float(fabric_factors[fabric])
jitter_penalty = 1.0 + float(comm_cfg["jitter_per_sriov"])*sriov
gpudirect_mult = float(comm_cfg["gpudirect_multiplier"]) if gpudirect else 1.0
comm_overhead = comm_intensity * fabric_factor * jitter_penalty * gpudirect_mult
comm_overhead = max(float(comm_cfg["min_overhead"]), comm_overhead)
```

### 3.2 Effective Token Throughput

The compute headroom is computed from accelerator scaling and vendor factor:

`base_rate = N * B_base * f_vendor(mode)`  
`tokens_per_second = (base_rate * rho) / (1 + omega * omega_comm)`  
```75:78:atlas_fabric/simulate.py
compute_rate = vendor_factor * util * accels * base_tokens_per_accel
tokens_sec = compute_rate / (1.0 + overhead_weight*comm_overhead)
```

For autoregressive inference with generation length `g` and total per-request tokens `s = prompt + generate`, the simulator scales throughput by `g/s` to reflect decode-only steady state:

`tokens_per_second = tokens_per_second * (g / max(1, s))` (applied only when `g > 0`)  
```92:95:atlas_fabric/simulate.py
if is_infer and workload["sequence"]["generate"] > 0:
    gen = int(workload["sequence"]["generate"])
    tokens_sec = tokens_sec * (gen / max(1, seq))
```

The per-step duration for training is:

`step_time_ms = (max(1, step_tokens) / tokens_per_second) * 1000`  
```114:115:atlas_fabric/simulate.py
"step_time_ms": (tokens_per_step / max(1.0, tokens_sec)) * 1000.0
```

### 3.3 FLOP and Memory Utilization

FP utilization reports communication-induced losses:

`flop_util = rho * (1 - kappa_comm * omega_comm)`

HBM utilization depends on sequence length `s` relative to a normalization factor:

`hbm_util = min(hbm_max, hbm_base + hbm_seq_scale * s / hbm_seq_norm)`  
```103:104:atlas_fabric/simulate.py
hbm_util = min(float(hbm_cfg["max"]), float(hbm_cfg["base"]) + float(hbm_cfg["seq_scale"])*(seq/float(hbm_cfg["seq_norm"])))
```

Random perturbations apply multiplicatively to throughput and tail latencies:

`tokens_per_second = tokens_per_second * (1 + epsilon)` with `epsilon ~ U[-sigma_noise/2, sigma_noise/2]`  
`p95 = p95 * (1 + epsilon * sigma_tail)` and `p99 = p99 * (1 + epsilon * sigma_tail)`  
```105:111:atlas_fabric/simulate.py
noise_cfg = hardware_model["noise"]
noise_amp = float(noise_cfg["amplitude"])
noise = (rnd.random()-0.5)*noise_amp
tokens_sec *= (1.0 + noise)
tail_noise_scale = float(noise_cfg["tail_scale"])
p95 *= (1.0 + noise*tail_noise_scale)
p99 *= (1.0 + noise*tail_noise_scale)
```

## 4. Latency Model (Inference)

Latency is derived from the compute headroom per accelerator and communication overhead. The simulator reports the **service component** (isolated inference time) and augments it with queueing delay once schedules are aggregated.

### 4.1 Service-Time Latency

`p50_service = max(lat_min, lat_base_coeff / max(1, base_rate / N))`

Tail percentiles depend on communication and virtualization penalties:

`tail_mult = 1 + tail_comm * omega_comm + tail_sriov · 𝟙{virt = sriov}`

`p95_service = p50_service · m95 · tail_mult`, `p99_service = p50_service · m99 · tail_mult`

Time-to-first-token is `ttft_service = p50_service · r_ttft`.

```81:91:atlas_fabric/simulate.py
base_p50 = max(min_p50, base_coeff / max(1.0, compute_rate/accels))
tail_mult = 1.0 + float(tail["comm_tail"])*comm_overhead + float(tail["sriov_tail"])*sriov
p50 = base_p50
p95 = p50 * (float(tail["p95_multiplier"])*tail_mult)
p99 = p50 * (float(tail["p99_multiplier"])*tail_mult)
ttft_ratio = float(lat_cfg["ttft_ratio"])
```

### 4.2 Queueing Latency for Scheduled Phases

For each schedule phase that declares a request profile (or fallback default), the adapter constructs a G/G/m approximation using the service distribution above:

- Mean service time per request: `E[S] = tokens_per_request / tokens_per_sec · 1000` (ms).
- Arrival rate `λ` derives from explicit `arrival.requests_per_sec` or inferred from throughput via `λ = load_factor · τ / tokens_per_request`.
- Service and arrival coefficients of variation default to profile-specific priors (`REQUEST_PROFILE_DEFAULTS`).
- Server count `m` matches data-parallel replicas or overrides (`arrival.servers`).

Using Whitt's approximation, per-phase waiting time is estimated as:

`Wq = G/G/1` Allen–Cunneen when `m = 1` or the Whitt square-root staffing rule for `m > 1`.

End-to-end latency percentiles add bounded waiting contributions:

`P50 = P50_service + min(Wq, 0.5 · P50_service)`

`P95 = P95_service + min(max(1.5 · Wq, P50_wait), 0.8 · P95_service)`

`P99 = P99_service + min(max(2 · Wq, P95_wait), 0.9 · P99_service)`

`TTFT = TTFT_service + P50_wait`

Each phase stores `queue_metrics` (arrival rate, utilization, waiting fractions) and `queue_adjusted` service/tail percentiles.

### 4.3 Schedule Aggregation

Aggregated schedule records include:

`service_latency_ms = Σ w_i · Pxx_service / Σ w_i`

`queue_latency_ms = Σ w_i · Pxx_total / Σ w_i`

`queue_wait_ms = Σ w_i · Wxx / Σ w_i` (for P50/P95/P99/TTFT)

`queue_utilization = Σ w_i · ρ_i / Σ w_i`

The reported percentiles (`p50_ms`, `p95_ms`, `p99_ms`, `ttft_ms`) default to the queue-adjusted values when queue modeling is applicable.

```274:303:atlas_fabric/adapter.py
aggregated["service_latency_ms"] = { ... }
aggregated["queue_latency_ms"] = { ... }
aggregated["queue_wait_ms"] = { ... "ttft": ... }
aggregated["p50_ms"] = aggregated["queue_latency_ms"]["p50"]
```

## 5. Power and Energy

Node power is split into idle and utilization-linked fractions:

`node_power_w = p_node_nominal * (phi_idle + phi_util * rho)`  
```96:99:atlas_fabric/simulate.py
power_cfg = hardware_model["power"]
idle_fraction = float(power_cfg["idle_fraction"])
util_fraction = float(power_cfg["util_fraction"])
node_power_w = node_power * (idle_fraction + util_fraction*util)
```

This power is reused for energy cost modeling.

## 6. Cost Formulation

Energy cost per node-hour uses a fixed electricity price (default `0.12 USD/kWh`) and PUE multiplier:

`energy_cost_per_hour = (node_power_w * pue / 1000) * cost_per_kwh`

The simulator translates throughput into kilo-tokens per hour `τ_k = τ · 3.6` (because `3.6 = 3600 / 1000`). The total hourly spend is infrastructure plus energy, yielding the primary economic metric:

`cost_per_1k_tokens = (hourly_cost + energy_cost_per_hour) / tokens_per_kilotoken_hour`

```28:38:atlas_fabric/cli.py
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
```

This same expression is reused across reports and optimizations. There is no separate training vs. inference cost equation; the mode-specific effects enter through `τ` and `P_node,avg`.

## 7. Scheduling Phases and Delay Modeling

Workloads may declare multi-phase schedules. Each phase defines:

- `duration_minutes` (`w_i`)
- Optional `tokens_scale` and `autoscale_factor`
- Overrides to workload parameters (e.g., shorter sequence length, different parallelism)

Each phase is simulated independently with its own random seed offset. The simulator applies multiplicative scaling to throughput and node power:

`tokens_per_second_i = tokens_per_second_i * (tokens_scale * autoscale_factor)`  
`node_power_i = node_power_i * autoscale_factor`

```91:95:atlas_fabric/adapter.py
record["tokens_per_sec"] = max(1.0, record.get("tokens_per_sec", 0.0) * total_scale)
if record.get("step_time_ms", 0.0):
    record["step_time_ms"] = record["step_time_ms"] / max(total_scale, 1e-6)
record["node_power_w"] = record.get("node_power_w", 0.0) * autoscale_factor
```

Aggregated metrics are duration-weighted averages:

`tokens_per_second_sched = (sum(w_i * tokens_per_second_i) / sum(w_i))`  
`latency_sched = (sum(w_i * latency_i) / sum(w_i))` for each percentile  
`node_power_sched = (sum(w_i * node_power_i) / sum(w_i))`

```101:126:atlas_fabric/adapter.py
"tokens_per_sec": weighted_avg("tokens_per_sec"),
"flop_utilization": weighted_avg("flop_utilization"),
"hbm_bw_util": weighted_avg("hbm_bw_util"),
"comm_compute_ratio": weighted_avg("comm_compute_ratio"),
"step_time_ms": weighted_avg("step_time_ms"),
"p50_ms": weighted_avg("p50_ms"),
"p95_ms": weighted_avg("p95_ms"),
"p99_ms": weighted_avg("p99_ms"),
"ttft_ms": weighted_avg("ttft_ms"),
"node_power_w": weighted_avg("node_power_w"),
```

_(Note: schedule scaling still relies on `dict.get()` defaults in `adapter.py`; ensure schedule overrides cover these fields when relying on strict modeling.)_

The schedule metadata preserves `schedule_total_minutes = ∑ w_i`, representing the simulated horizon. Because each phase uses a constant multiplier, scheduling “delays” manifest as reduced effective throughput during phases with `tokens_scale < 1` or increased autoscaling overhead (`autoscale_factor > 1`). Inference schedules can additionally override data parallelism to reflect burst capacity.

## 8. Parameter Assumptions

- **Hardware Priors.** Each accelerator YAML file extends the default hardware model (`atlas_fabric.hardware.DEFAULT_MODEL`). Missing entries fall back to defaults (e.g., base utilization `0.55`, power idle fraction `0.4`).
- **Workload Geometry.** Hidden size, layer count, and sequence lengths primarily affect memory headroom (`peak_mem_gb = min(memory_limit, κ_mem · hidden · layers)`).
- **Knobs.** Binary knobs (`cuda_graphs`, `gpudirect_rdma`, `virt`) toggle incremental utilization gains or penalties exactly as configured in the hardware profile.
- **Noise.** Randomness is deterministic per seed via `deterministic_rand` to produce reproducible jitter.
- **Energy Price.** The electricity price `0.12 USD/kWh` is currently fixed in `cli.py` and `reporter.py` for consistency across runs.

## 9. Optimization Procedure

Atlas Fabric includes an optional hill-climb optimizer that evaluates post-simulation knob tweaks to improve steady-state throughput while respecting mode-specific constraints. Starting from the baseline knobs used for the initial run, the optimizer reuses `simulate_run` to score each candidate change with a deterministic seed offset (to avoid identical noise samples while keeping runs reproducible).

- **Candidate knobs.** The search space is a fixed list of toggles: enabling CUDA Graphs, enabling GPUDirect RDMA, switching the interconnect fabric to `IB`, switching virtualization to `bare`, and doubling or quadrupling the microbatch size.
- **Evaluation loop.** Candidates are applied one at a time to the current best knob set, simulated, and compared against the incumbent record. The baseline configuration remains the fallback if no improvement is accepted.
- **Acceptance rule.** Training workloads accept a candidate only if it improves `tokens_per_sec` by at least 3%. Inference workloads must also satisfy the same throughput uplift while keeping `p99_ms` below the SLA (`DEFAULT_SLA_P99_MS = 120 ms` unless overridden at the CLI).
- **Output.** The optimizer returns the best knob dictionary and its associated record; if every candidate fails the acceptance rule, the original configuration is returned unchanged.

```8:36:atlas_fabric/optimizer.py
best = simulate_run(target, workload, knobs, seed)
for change in candidates:
    trial = deepcopy(knobs); trial.update(change)
    rec = simulate_run(target, workload, trial, seed+1)
    if _better(rec, best, workload, sla_p99_ms):
        best, best_knobs = rec, deepcopy(trial)
return {"knobs": best_knobs, "record": best}
```

The mode-aware `_better` function enforces the acceptance rule, gating inference configurations on both throughput and SLA, and training configurations solely on throughput gain.

## 10. Summary

Atlas Fabric combines simple, tunable analytical components to estimate training and inference efficiency:

1. Compute headroom (`tokens_per_sec`) derived from utilization, accelerator count, and communication drag.
2. Latency projections (`p50/p95/p99`, `ttft`) composed of service-time baselines plus queue adjustments when schedules are present.
3. Energy and cost estimates grounded in node power modeling, PUE, and throughput-normalized billing.
4. Optional hill-climb optimization that toggles a small set of knobs while respecting inference SLAs.
5. Schedule-aware aggregation that blends per-phase outputs into duration-weighted metrics for reporting and optimization comparisons.

Together these elements let Atlas Fabric simulate end-to-end effects of hardware targets, workload geometry, and controllable knobs for both training and inference modes.
