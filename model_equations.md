# Simulation Model Equations

## 1. Overview

Mathematical formulas used by Atlas Fabric to calculate performance, latency, energy, and cost for training and inference workloads.

**Model Type: Semi-Empirical**  
This is a **semi-empirical model** that combines theoretical performance equations with empirically measured parameters. The model structure is based on fundamental computer architecture principles, but the key coefficients must be obtained through systematic measurement of actual hardware behavior.

**Core Principle:** The simulator models performance as hardware capability modified by utilization efficiency and degraded by communication overhead:

**Performance = Hardware Capability × Utilization ÷ Overhead**

- **Hardware Capability**: What the hardware can theoretically achieve (peak performance)
- **Utilization**: How efficiently we can use the hardware (typically 50-90%)
- **Overhead**: Performance losses from coordination and communication between devices

### Simulation Flow
```
[Hardware Specs] + [Workload Profile] + [Optimization Parameters]
                            ↓
                    [Core Simulation Engine]
                            ↓
        ┌──────────────┬────────────────┬────────────────┐
        ↓              ↓                ↓                ↓
  [Throughput]   [Latency]        [Utilization]      [Cost]
```

## 2. Notation

**Inputs:**
- Hardware target `T` - The specific hardware configuration (GPU types, interconnects)
- Workload profile `W` - What we're running (model parameters, batch sizes, sequence lengths)
- Optimization parameters `K` - Tunable settings to improve performance (CUDA graphs, batch sizing)

**Derived symbols:**

| Symbol | Description | Typical Values |
| --- | --- | --- |
| `N` | Total accelerators = `accelerators_per_node * num_nodes` | 8-1024 |
| `B_base` | Base token rate per accelerator from hardware profile | 100-1000 tokens/sec |
| `f_vendor(mode)` | Vendor- or mode-specific compute multiplier (`training` or `inference`) | 0.5-2.0 |
| `rho_base` | Base utilization factor from hardware profile | 0.4-0.6 |
| `delta_rho_cuda`, `delta_rho_gpudirect`, `delta_rho_infer` | Utilization boosts from optimization parameters / workload mode | +0.1, +0.05, +0.15 |
| `chi_tp`, `chi_pp` | Communication intensities per tensor/pipeline-parallel stage | 0.1-0.3 per stage |
| `gamma_fabric`, `gamma_jitter`, `gamma_gpudirect` | Communication penalties due to interconnect, virtualization, GPUDirect | 1.0-2.0 |
| `omega` | Throughput overhead weight | 0.5-1.5 |
| `kappa_comm` | FLOP penalty coefficient used to infer FP utilization | 0.1-0.3 |
| `lat_min`, `lat_base_coeff`, `tail_comm`, `tail_sriov` | Latency configuration coefficients | 1-10ms, varies |
| `p_node_nominal` | Nominal node power from target spec | 5-20 kW |
| `phi_idle`, `phi_util` | Idle and utilization-linked fractions of node power | 0.4, 0.6 |
| `sigma_noise`, `sigma_tail` | Random jitter amplitudes | 0.05-0.15 |
| `step_tokens` | Tokens per training step (`dp * microbatch * (prompt + generate)`) | 1K-100K |

### Parameters Requiring Empirical Measurement

The following parameters **MUST** be obtained through systematic measurement on actual hardware:

**Hardware Performance Parameters:**
- `B_base` - Base token throughput per accelerator (measured via single-GPU benchmarks)
- `rho_base` - Base utilization efficiency (measured from actual vs theoretical FLOPS)
- `f_vendor(mode)` - Vendor-specific scaling factors for training/inference

**Communication Parameters:**
- `chi_base`, `chi_tp`, `chi_pp` - Communication intensity coefficients (measured via multi-GPU scaling tests)
- `gamma_fabric` - Network fabric overhead factors (Ethernet vs InfiniBand measurements)
- `gamma_jitter` - Virtualization-induced jitter (measured in cloud environments)

**Latency Parameters:**
- `lat_min`, `lat_base_coeff` - Base latency coefficients (measured from single-request benchmarks)
- `tail_comm`, `tail_sriov` - Tail latency multipliers (from percentile analysis)
- P95/P99 multipliers - Tail distribution factors (from production traces)

**Power Parameters:**
- `p_node_nominal` - Nominal power consumption (measured via power meters)
- `phi_idle`, `phi_util` - Idle and active power fractions (from power profiling)

**Optimization Impact:**
- `delta_rho_cuda`, `delta_rho_gpudirect` - Performance improvements from optimizations (A/B testing)

See `model_calibration.md` for detailed measurement procedures.

## 3. Compute and Communication Throughput

The simulator calculates throughput through a pipeline approach: **[Theoretical Max] → [Utilization Loss] → [Communication Loss] → [Actual Throughput]**

### 3.1 Utilization

**Utilization** represents how efficiently the hardware is being used (0 to 1, where 1 = 100% efficient). Base hardware efficiency plus optimization parameter effects:

`rho = min(rho_base + delta_rho_cuda * cuda_graphs + delta_rho_gpudirect * gpudirect + delta_rho_infer * indicator(mode == "inference"), rho_max)`

This equation shows how optimization parameters improve utilization from the base hardware efficiency:
- Base utilization (e.g., 0.55 or 55%)
- +0.10 if CUDA Graphs enabled
- +0.05 if GPUDirect RDMA enabled
- +0.15 bonus for inference mode
- Capped at maximum (e.g., 0.90 or 90%)

```52:58:atlas_fabric/simulate.py
util = float(util_cfg["base"])
inc = util_cfg["knob_increments"]
util += float(inc["cuda_graphs"]) * cuda_graphs
util += float(inc["gpudirect_rdma"]) * gpudirect
if is_infer:
    util += float(inc["inference_bonus"])
util = min(util, float(util_cfg["max"])
```



### Communication Intensity
Communication intensity measures how much data exchange is needed between accelerators. It quantifies the performance overhead from coordinating multiple devices during distributed training and inference.

`chi = (chi_base + chi_tp * (tp - 1) + chi_pp * (pp - 1)) * (chi_infer_scale if mode == "inference" else 1)`

### Components Breakdown

**Base Communication (`chi_base`):**
- Represents fundamental coordination overhead even with single-device workloads
- Includes synchronization barriers, gradient accumulation, and memory management
- Typical range: 0.05-0.15 (5-15% overhead)

**Tensor Parallelism Overhead (`chi_tp * (tp - 1)`):**
- Scales linearly with tensor parallelism degree
- Each additional TP stage requires:
  - **Forward pass**: Broadcasting activations across devices
  - **Backward pass**: All-reducing gradients
  - **Memory**: Storing intermediate activations for gradient computation
- Communication volume scales with model dimensions (hidden_size, attention heads)
- Typical `chi_tp`: 0.08-0.25 per additional TP stage

**Pipeline Parallelism Overhead (`chi_pp * (pp - 1)`):**
- Scales with pipeline depth
- Each pipeline stage boundary requires:
  - **Activation passing**: Forwarding intermediate results
  - **Gradient passing**: Backward propagation through pipeline
  - **Micro-batching**: Synchronization between pipeline stages
- Communication volume depends on activation size and sequence length
- Typical `chi_pp`: 0.10-0.30 per additional PP stage

**Inference Scaling (`chi_infer_scale`):**
- Inference workloads typically have lower communication intensity
- Reasons:
  - No gradient computation (eliminates backward pass communication)
  - Often uses smaller batch sizes
  - KV-cache reduces recomputation overhead
- Typical `chi_infer_scale`: 0.3-0.7 (30-70% of training communication)

### Communication Patterns by Parallelism Strategy

**Data Parallelism (DP):**
- Communication: All-reduce of gradients
- Frequency: Once per training step
- Volume: Model parameter size
- Overhead: Included in `chi_base`

**Tensor Parallelism (TP):**
- Communication: Activation and gradient exchanges
- Frequency: Every forward/backward pass
- Volume: Activation tensors (varies by layer)
- Overhead: `chi_tp * (tp - 1)`

**Pipeline Parallelism (PP):**
- Communication: Activation forwarding between stages
- Frequency: Every micro-batch
- Volume: Intermediate activations
- Overhead: `chi_pp * (pp - 1)`

### Real-World Communication Examples

**GPT-3 Scale (175B parameters, TP=8, PP=8):**
- High communication intensity due to large model size
- TP=8: ~8x activation exchanges per layer
- PP=8: ~8 pipeline stage boundaries
- Total `chi` ≈ 0.15 + 8×0.20 + 8×0.25 = 3.55 (355% overhead)

**Smaller Model (7B parameters, TP=2, PP=2):**
- Lower communication intensity
- TP=2: Minimal tensor parallelism overhead
- PP=2: Simple two-stage pipeline
- Total `chi` ≈ 0.10 + 1×0.15 + 1×0.20 = 0.45 (45% overhead)

### Measurement and Calibration

**Empirical Measurement Required:**
- `chi_base`, `chi_tp`, `chi_pp` must be measured on actual hardware
- Measurement approach:
  1. Run single-device baseline
  2. Scale up TP gradually (2, 4, 8) and measure throughput degradation
  3. Scale up PP gradually (2, 4, 8) and measure throughput degradation
  4. Fit linear regression to extract coefficients
- Network fabric affects measured values (InfiniBand vs Ethernet)

**Typical Measured Values:**
- Modern GPUs (H100, A100): `chi_tp` ≈ 0.15-0.25, `chi_pp` ≈ 0.20-0.30
- Older GPUs (V100): `chi_tp` ≈ 0.20-0.35, `chi_pp` ≈ 0.25-0.40
- Cloud instances: Higher values due to virtualization overhead

```60:65:atlas_fabric/simulate.py
comm_intensity = float(intensity["base"]) + float(intensity["per_tp"])*(tp-1) + float(intensity["per_pp"])*(pp-1)
if is_infer:
    comm_intensity *= float(intensity["inference_scale"])
```

The **end-to-end communication overhead** factor represents the total performance penalty from data movement. It combines network, jitter, and GPUDirect multipliers:

`omega_comm = max(omega_min, chi * gamma_fabric * gamma_jitter * gamma_gpudirect)`

This multiplies:
- `chi`: How much communication is needed
- `gamma_fabric`: Network fabric penalty (Ethernet vs InfiniBand)
- `gamma_jitter`: Virtualization-induced timing variability
- `gamma_gpudirect`: Penalty when not using direct GPU memory access  
```66:73:atlas_fabric/simulate.py
fabric_factors = comm_cfg["fabric_factors"]
fabric_factor = float(fabric_factors[fabric])
jitter_penalty = 1.0 + float(comm_cfg["jitter_per_sriov"])*sriov
gpudirect_mult = float(comm_cfg["gpudirect_multiplier"]) if gpudirect else 1.0
comm_overhead = comm_intensity * fabric_factor * jitter_penalty * gpudirect_mult
comm_overhead = max(float(comm_cfg["min_overhead"]), comm_overhead)
```

### 3.2 Token Throughput

**Token throughput** measures how many tokens (word pieces) the system can process per second. This is the key performance metric.

**Step 1: Calculate Theoretical Maximum**

Accelerator scaling with vendor factor:

`base_rate = N * B_base * f_vendor(mode)` (Theoretical maximum)

This multiplies:
- `N`: Total number of accelerators in the system
- `B_base`: Tokens each accelerator can theoretically process per second
- `f_vendor(mode)`: Vendor-specific adjustment factor (training vs inference)

**Step 2: Apply Utilization and Communication Overhead**

`tokens_per_second = (base_rate * rho) / (1 + omega * omega_comm)`

**This is the core performance equation**: theoretical max × utilization ÷ overhead

Where:
- `rho` = Effective utilization factor (0.55 means only 55% of hardware is utilized)
- `omega` = Weight factor for how much communication impacts throughput
- `omega_comm` = Communication overhead from multi-accelerator coordination
- The denominator `(1 + omega * omega_comm)` reduces throughput based on communication costs  
```75:78:atlas_fabric/simulate.py
compute_rate = vendor_factor * util * accels * base_tokens_per_accel
tokens_sec = compute_rate / (1.0 + overhead_weight*comm_overhead)
```

For **autoregressive inference** (generating text token-by-token), throughput scales by `g/s` (decode-only steady state):

`tokens_per_second = tokens_per_second * (g / max(1, s))` (applied only when `g > 0`)

- `g`: Number of tokens to generate
- `s`: Total sequence length
- This reflects that generation is more efficient than processing the initial prompt  
```92:95:atlas_fabric/simulate.py
if is_infer and workload["sequence"]["generate"] > 0:
    gen = int(workload["sequence"]["generate"])
    tokens_sec = tokens_sec * (gen / max(1, seq))
```

The **per-step duration** for training (time to process one batch) is:

`step_time_ms = (max(1, step_tokens) / tokens_per_second) * 1000`

- `step_tokens`: Total tokens in one training step (batch size × sequence length)
- Converted to milliseconds (× 1000)
- Lower is better for training speed  
```114:115:atlas_fabric/simulate.py
"step_time_ms": (tokens_per_step / max(1.0, tokens_sec)) * 1000.0
```

### 3.3 FLOP and Memory

**FLOP utilization** measures how efficiently the floating-point compute units are used. Communication reduces compute efficiency:

`flop_util = rho * (1 - kappa_comm * omega_comm)`

- Starts with base utilization `rho`
- Reduced by communication overhead (more communication = less compute)

**HBM (High Bandwidth Memory) utilization** measures GPU memory bandwidth usage. It depends on sequence length `s` relative to a normalization factor:

`hbm_util = min(hbm_max, hbm_base + hbm_seq_scale * s / hbm_seq_norm)`

- Longer sequences require more memory bandwidth
- Capped at `hbm_max` to prevent exceeding physical limits  
```103:104:atlas_fabric/simulate.py
hbm_util = min(float(hbm_cfg["max"]), float(hbm_cfg["base"]) + float(hbm_cfg["seq_scale"])*(seq/float(hbm_cfg["seq_norm"])))
```

**Random perturbations** simulate real-world variability. They apply multiplicatively to throughput and tail latencies:

`tokens_per_second = tokens_per_second * (1 + epsilon)` with `epsilon ~ U[-sigma_noise/2, sigma_noise/2]`  
`p95 = p95 * (1 + epsilon * sigma_tail)` and `p99 = p99 * (1 + epsilon * sigma_tail)`

- `epsilon`: Random noise factor (uniform distribution)
- Tail latencies (p95, p99) get amplified noise to model real-world spikes  
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

**Latency** is how long it takes to process a request (measured in milliseconds). Lower is better for user experience.

Latency is derived from the compute headroom per accelerator and communication overhead. The simulator distinguishes between:
- **Service Time**: Time to process a single request in isolation (no other requests)
- **Total Latency**: Service time + queueing delay under load (realistic scenario with multiple requests)

The simulator reports the **service component** (isolated inference time) and augments it with queueing delay once schedules are aggregated.

### 4.1 Service Time

**Median service time** (p50 = 50th percentile) is the typical response time:

`p50_service = max(lat_min, lat_base_coeff / max(1, base_rate / N))`

- Inversely proportional to compute rate per accelerator
- Has a minimum floor (`lat_min`) representing unavoidable overhead

**Tail percentiles** (p95, p99) represent worst-case latencies (95th and 99th percentile). They include communication/virtualization penalties:

`tail_mult = 1 + tail_comm * omega_comm + tail_sriov · 𝟙{virt = sriov}`

`p95_service = p50_service · m95 · tail_mult`, `p99_service = p50_service · m99 · tail_mult`

- `tail_mult`: Multiplicative penalty for tail latencies
- `m95`, `m99`: Base multipliers for tail percentiles
- SR-IOV virtualization adds extra latency spikes

**Time-to-first-token (TTFT)** is how long until the first token appears in streaming generation:

`ttft_service = p50_service · r_ttft`

- Usually faster than full response time
- Critical for perceived responsiveness in chat applications

```81:91:atlas_fabric/simulate.py
base_p50 = max(min_p50, base_coeff / max(1.0, compute_rate/accels))
tail_mult = 1.0 + float(tail["comm_tail"])*comm_overhead + float(tail["sriov_tail"])*sriov
p50 = base_p50
p95 = p50 * (float(tail["p95_multiplier"])*tail_mult)
p99 = p50 * (float(tail["p99_multiplier"])*tail_mult)
ttft_ratio = float(lat_cfg["ttft_ratio"])
```

### 4.2 Queue Modeling

**Queue modeling** simulates what happens when multiple requests arrive and must wait for processing. Uses **G/G/m queue theory** (General arrival/General service/m servers):

- Service time: `E[S] = tokens_per_request / tokens_per_sec · 1000` ms
- Arrival rate: From `arrival.requests_per_sec` or inferred from throughput
- Servers: Data-parallel replicas or explicit override
- Waiting time: Whitt's approximation (Allen-Cunneen for m=1)

**Latency with bounded queue delay** (realistic latencies including wait time):

`P50 = P50_service + min(Wq, 0.5 · P50_service)`

`P95 = P95_service + min(max(1.5 · Wq, P50_wait), 0.8 · P95_service)`

`P99 = P99_service + min(max(2 · Wq, P95_wait), 0.9 · P99_service)`

`TTFT = TTFT_service + P50_wait`

- `Wq`: Average queue wait time
- Queue delays are bounded to prevent unrealistic spikes
- Higher percentiles have proportionally higher queue delays

Each phase stores `queue_metrics` (arrival rate, utilization, waiting fractions) and `queue_adjusted` service/tail percentiles.

### 4.3 Aggregation

**Aggregation** combines metrics from multiple phases (e.g., different load periods) using duration-weighted averages:

- `service_latency_ms = Σ w_i · Pxx_service / Σ w_i`
- `queue_latency_ms = Σ w_i · Pxx_total / Σ w_i`
- `queue_wait_ms = Σ w_i · Wxx / Σ w_i`
- `queue_utilization = Σ w_i · ρ_i / Σ w_i`

```274:303:atlas_fabric/adapter.py
aggregated["service_latency_ms"] = { ... }
aggregated["queue_latency_ms"] = { ... }
aggregated["queue_wait_ms"] = { ... "ttft": ... }
aggregated["p50_ms"] = aggregated["queue_latency_ms"]["p50"]
```

## 5. Power

**Power consumption** (in Watts) consists of idle plus utilization-based components:

`node_power_w = p_node_nominal * (phi_idle + phi_util * rho)`

- `p_node_nominal`: Rated power of the node (e.g., 10kW)
- `phi_idle`: Fraction consumed when idle (typically ~40%)
- `phi_util`: Additional fraction when fully utilized
- Higher utilization = higher power consumption  
```96:99:atlas_fabric/simulate.py
power_cfg = hardware_model["power"]
idle_fraction = float(power_cfg["idle_fraction"])
util_fraction = float(power_cfg["util_fraction"])
node_power_w = node_power * (idle_fraction + util_fraction*util)
```

This power is reused for energy cost modeling.

## 6. Cost

**Energy cost** includes electricity price and datacenter overhead (PUE = Power Usage Effectiveness):

`energy_cost_per_hour = (node_power_w * pue / 1000) * cost_per_kwh`

- PUE accounts for cooling and infrastructure (typically 1.1-1.5)
- Converted from Watts to kilowatts (/1000)
- Standard rate: $0.12/kWh

**Total cost per 1K tokens** combines infrastructure and energy costs:

`cost_per_1k_tokens = (hourly_cost + energy_cost_per_hour) / tokens_per_kilotoken_hour`

- `hourly_cost`: Hardware rental/amortization per hour
- Normalized to cost per 1000 tokens for easy comparison
- Key metric for economic viability

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

Training vs. inference differences come from throughput and power variations.

## 7. Multi-Phase Schedules

**Multi-phase schedules** model varying workload conditions over time (e.g., peak vs off-peak). Each phase can specify:
- Duration (`w_i`) - how long this phase lasts
- Scaling factors (`tokens_scale`, `autoscale_factor`) - load adjustments
- Parameter overrides (sequence length, parallelism) - configuration changes

Per-phase scaling:

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

Phases with `tokens_scale < 1` reduce throughput; `autoscale_factor > 1` increases overhead. Inference can adjust data parallelism for burst capacity.

## 8. Default Parameters

- **Hardware:** Base utilization 0.55, idle power 40%
- **Workload:** Memory scales with model size
- **Optimization Parameters:** Binary toggles and multipliers for performance improvements
- **Noise:** Deterministic jitter via seeded RNG
- **Energy:** Fixed $0.12/kWh electricity price

## 9. Optimization

**Automatic optimization** uses a hill-climbing algorithm to find the best configuration:

**Search space (optimization parameters):**
- Enable CUDA Graphs - reduces kernel launch overhead (~10% improvement)
- Enable GPUDirect RDMA - allows direct GPU-to-GPU memory transfers (~5% improvement)
- Switch to InfiniBand or bare metal - better network fabric (reduces communication overhead)
- Increase microbatch size (2x, 4x) - better GPU utilization with larger batches

**Acceptance criteria:**
- Training: >3% throughput gain
- Inference: >3% throughput + P99 < SLA (120ms default)

**Process:** 
1. Start with baseline configuration
2. Test each parameter change individually
3. Keep changes that improve performance
4. Return the best configuration found

This greedy approach quickly finds good (though not necessarily optimal) configurations

```8:36:atlas_fabric/optimizer.py
best = simulate_run(target, workload, params, seed)
for change in candidates:
    trial = deepcopy(params); trial.update(change)
    rec = simulate_run(target, workload, trial, seed+1)
    if _better(rec, best, workload, sla_p99_ms):
        best, best_params = rec, deepcopy(trial)
return {"params": best_params, "record": best}
```
(Note: 'knobs' in code refers to optimization parameters)


## 10. Summary

### Model Philosophy
This semi-empirical model bridges the gap between pure theoretical models (which often miss real-world effects) and pure empirical models (which don't generalize well). By using a theoretically-sound structure with empirically-measured coefficients, we achieve:
- **Accuracy**: Parameters reflect actual hardware behavior
- **Generalization**: Model structure applies across different hardware
- **Interpretability**: Each parameter has physical meaning

Key components:
1. **Throughput:** Accelerator count × utilization ÷ communication overhead
2. **Latency:** Service time + queue delay (with percentiles)
3. **Cost:** Infrastructure + energy based on throughput
4. **Optimization:** Automated parameter tuning with SLA constraints
5. **Schedules:** Multi-phase workloads with weighted aggregation

The model follows these key principles:
- **Performance degrades through three main bottlenecks**: 
  - Base utilization limits (hardware is never 100% efficient)
  - Communication overhead (GPUs must exchange data)
  - System noise (random variations in real systems)
- **Optimization parameters improve performance** by reducing these bottlenecks:
  - CUDA Graphs reduce kernel launch overhead (+10% utilization)
  - GPUDirect enables faster GPU communication (+5% throughput)
  - Better interconnects (InfiniBand) reduce communication delays
- **Trade-offs exist** between throughput, latency, and cost:
  - Higher throughput often means higher latency (batching)
  - Better performance requires more expensive hardware
  - Optimization choices depend on workload requirements

This model enables rapid comparison of hardware, workloads, and configurations for training and inference.
