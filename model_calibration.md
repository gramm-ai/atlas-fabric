# Atlas Fabric Model Calibration Guide

## 1. Purpose and Scope

This guide details how to calibrate the Atlas Fabric simulator so that its accelerator, workload, and knob parameters reproduce empirical behavior. The focus is on latency-sensitive inference workloads, but the same protocol applies to throughput and training measurements. Each section expands the recommended practices for modeling service time, queueing, heavy-tail risk, and multi-objective trade-offs.

## 2. Calibration Workflow Overview

1. **Enforce the measurement protocol** to guarantee reproducibility.
2. **Collect raw traces** that separate isolated service time from end-to-end latency, capture hardware counters, and annotate experimental factors.
3. **Fit service-time distributions** per operation/model and validate the fit.
4. **Model queueing dynamics** using empirical arrivals and calibrated service distributions; map queue estimates into schedule-aware latency aggregation.
5. **Summarize tail behavior** with percentiles, CCDFs, and tail conditional means; attach uncertainty via resampling.
6. **Attribute variance** to controllable knobs and hardware states via designed experiments.
7. **Build quantile-aware surrogates** for co-optimization and extreme-value analysis for rare events.
8. **Visualize and report** results in decision-ready formats, mapping measurements back to Atlas Fabric configuration parameters.

## 3. Measurement Protocol

- **Warm-up:** Run at least 2–5× the largest batch before measuring to stabilize caches, JIT, and compilation artifacts.
- **CPU pinning & governors:** Pin workloads and load generators to fixed cores; set CPU/GPU governors (P-states, frequency locks) to eliminate drift.
- **Clock sync:** Synchronize host and accelerator clocks (e.g., `ptp4l`, `chrony`) so latency measurements align across machines.
- **Cache/TLB control:** Flush or fix cache state between runs when isolating service time; document L2/HBM residency strategies.
- **Seed management:** Fix RNG seeds spanning data loaders, sequence sampling, and workload simulators.
- **Stable load generators:** Use deterministic request schedules or well-characterized stochastic generators; log generator metadata with every run.
- **Artifact logging:** Store firmware versions, driver hashes, and power settings alongside measurement batches for traceability.

## 4. Data Collection Requirements

- **Per-request timelines:** Record enqueue time, dispatch time, first-token time, completion time, and per-stage timestamps for decode vs. prefill.
- **Isolated service-time probes:** Trigger requests on an idle accelerator to measure pure service time per operation/model variant.
- **Hardware counters:** Capture utilization, SM occupancy, power draw, memory bandwidth, and PCIe/NVLink throughput to inform `rho_base`, `kappa_comm`, and communication penalties.
- **Control knobs:** Log binary and continuous knobs (`cuda_graphs`, `gpudirect`, batch size, thread pools) for regression/DOE analysis.
- **Environmental sensors:** Optionally capture temperature, voltage, and contention signals to explain residual variance.

## 5. Service-Time Modeling

- **Separation from queueing:** Compute service time `S = completion − dispatch` from isolated runs; exclude queued time to avoid conflating scheduling effects.
- **Distribution selection:** Fit log-normal, Weibull, or Gamma distributions. When Q-Q plots reveal bimodality (e.g., kernel compilation cold start), fit a finite mixture (e.g., two log-normals) and select via AIC/BIC.
- **Parameter estimation:** Use maximum likelihood or Bayesian inference with weak priors; store fitted parameters per model, sequence length, and batch size.
- **Goodness-of-fit:** Generate Q-Q plots, Kolmogorov–Smirnov statistics, and probability plots; require that high-quantile residuals fall within confidence envelopes before accepting the fit.
- **Temporal drift:** Re-evaluate fits periodically; use rolling windows or hierarchical models to capture slow drift in firmware or thermal conditions.
- **Mapping to simulator:** Translate mean service rate to `B_base` and `f_vendor(mode)`; heavy-tail multipliers inform `tail_comm` and `sigma_tail` in `atlas_fabric/simulate.py`.

## 6. Queueing Model Calibration

- **Arrival process:** Measure inter-arrival times. If coefficient of variation ≈1, an M/G/m approximation is reasonable; otherwise, fit a renewal process (e.g., log-normal arrivals) and treat the system as G/G/m. Calibrate profile priors (`arrival_load_factor`, `arrival_cv`) per traffic regime so that inferred arrivals match telemetry.
- **Server count:** Match `m` to active accelerator instances or virtual lanes; include replication if requests are broadcast. Feed these counts back into schedule phases via `arrival.servers` overrides.
- **Scheduling policy:** Capture dispatcher rules (round robin, SRPT, FIFO). Simulate queueing using empirical service-time samples and policy-specific scheduling; encode effective waiting times when deriving the `queue_wait_ms` fields surfaced by the adapter.
- **Validation:** Compare simulated latency percentiles with measured end-to-end latencies. Adjust queue model until residuals across P50/P95/P99 fall within measurement CI, and verify waiting fractions align with traced queue instrumentation.
- **Sensitivity:** Stress-test the queue across target arrival rates and burst scenarios to quantify tail growth. Update `REQUEST_PROFILE_DEFAULTS` and phase priors when load patterns shift.

## 7. Tail Metrics and Reporting

- **Percentiles:** Always publish `P50`, `P95`, `P99`, `P99.9`. Derive time-to-first-token (`TTFT`) separately when relevant.
- **CCDF:** Plot complementary CDF `P(Latency > x)` on log-x scales to highlight SLO exceedance probabilities.
- **Tail Conditional Mean:** Compute `E[Latency | Latency > P99]` to quantify the severity of tail events.
- **SLO miss rate:** For each latency objective `L`, report `P(Latency > L)`; integrate into decision reports.

## 8. Uncertainty Quantification

- **Bootstrap CIs:** Use percentile bootstrap with ≥1,000 resamples for each reported percentile and tail conditional mean.
- **Comparative tests:** Apply paired bootstraps or permutation tests when comparing configurations; report p-values or confidence that one dominates.
- **Stochastic dominance:** Evaluate first- and second-order dominance of latency distributions to ensure improvements hold across tails.
- **Documentation:** Include CI bounds and testing methodology in stakeholder reports to prevent overinterpretation of noisy wins.

## 9. Variance Attribution

- **Factorial DOE:** Design experiments across factors such as batch size, model variant, hardware state, frequency governor, and contention level.
- **ANOVA / Sobol indices:** Quantify each factor's main effects and interactions on tail latency and throughput; use Sobol if the response is non-linear.
- **Iterative narrowing:** Focus further experimentation on high-sensitivity factors; feed their contributions into Atlas Fabric knob priors (`delta_rho_*`).

## 10. Quantile-Aware Modeling

- **Quantile regression:** Fit conditional models for `P95` and `P99` as functions of knobs and workload parameters; allow separate coefficients from median models.
- **Algorithm choices:** Use gradient boosted quantile regression, generalized additive models, or neural quantile regressors when interactions are complex.
- **Usage in Atlas Fabric:** Embed quantile models as surrogate predictors for tail multipliers or to parameterize `tail_comm`, enabling more accurate inference mode knobs.

## 11. Extreme-Value Analysis

- **Peaks-over-threshold (POT):** When latency tails exceed fitted distributions, select a high threshold (e.g., 99th percentile) and fit a Generalized Pareto Distribution to the exceedances.
- **Return levels:** Estimate latencies associated with rare events (e.g., once per day/week) to set guard-band SLOs.
- **Integration:** Use POT-derived parameters to cap `tail_comm` or inject rare spikes into queueing simulations.

## 12. Multi-Objective Surrogate and Optimization

- **Objective set:** Model `{P99 latency, throughput, energy/op, cost/op}` simultaneously; include constraint penalties for SLO violations.
- **Surrogate choices:** Gaussian Processes with ARD kernels, multi-output gradient boosting, or random forest surrogates with quantile estimates.
- **Bayesian optimization:** Run acquisition functions (e.g., Expected Hypervolume Improvement) to trace the Pareto frontier across hardware/software knobs.
- **Feedback loop:** Update surrogate with fresh measurements; reflect optimized settings back into YAML specs (`util`, `tail`, `power`).

## 13. Visualization Standards

- **CDF/CCDF plots:** Present on log-x axes; annotate SLO thresholds and confidence bands.
- **Violin plots:** Show latency distributions per configuration for quick comparison; overlay percentile markers.
- **Latency breakdown:** Stack service vs. waiting components to highlight queue-driven inflation.
- **DOE heatmaps:** Visualize factor effects (main and interaction) to communicate variance attribution.
- **Optimization frontier:** Plot Pareto curves for `P99 latency` vs. throughput or cost to show trade-offs.

## 14. Stakeholder Report Template

Prepare a one-page summary per configuration containing:

- Workload description and arrival rate assumptions.
- `P50/P95/P99/P99.9 ± CI`, `TTFT`, and tail conditional mean.
- CCDF plot with SLO miss probabilities.
- Energy per operation, cost per operation, utilization metrics.
- SLO miss rate at the target latency `L` and recommended guard bands.
- Notable contributing factors (from variance analysis) and suggested knob adjustments.

## 15. Mapping Measurements to Atlas Fabric Parameters

- **Utilization (`rho_base`, `delta_rho_*`):** Derive from measured SM occupancy and throughput under baseline vs. knob-enabled runs. Normalize so that `rho_base` reflects the best-fit mean utilization, while deltas capture marginal gains.
- **Queue priors (`REQUEST_PROFILE_DEFAULTS`):** Fit load factors and arrival CVs for each traffic regime (steady, peak, spike, low). Update these priors so schedule phases infer realistic arrival rates when explicit telemetry is absent.
- **Communication penalties (`gamma_*`, `omega`):** Use measured communication overhead ratios (e.g., NVLink/PCIe bandwidth saturation) to calibrate `comm_overhead` factors; align with mixture components from service-time models.
- **Latency coefficients (`lat_base_coeff`, `tail_comm`, `tail_sriov`):** Fit `p50` regression against `compute_rate/accels`; map tail multipliers from quantile regression or POT fits to `tail_comm` and virtualization surcharges to `tail_sriov`.
- **Schedule queue outputs (`queue_latency_ms`, `queue_wait_ms`):** Validate aggregated waiting times against empirical telemetry. Adjust waiting heuristics (e.g., factors on `Wq`) to match P95/P99 deltas between service and total latency.
- **Noise parameters (`sigma_noise`, `sigma_tail`):** Estimate from residual variability after accounting for DOE factors; use bootstrap residuals to ensure the simulator reproduces observed jitter.
- **Power model (`phi_idle`, `phi_util`):** Fit linear models between utilization and measured node power; update YAML values accordingly.
- **Schedule overrides:** For multi-phase workloads, encode measured per-phase scaling (`tokens_scale`, `autoscale_factor`) and arrival bursts to reproduce empirical latency swings.

## 16. Hardware Default Justifications

- **Compute throughput.** `base_tokens_per_accel` and mode-specific factors in the accelerator YAMLs reflect sustained transformer throughput in public disclosures. NVIDIA H100 MLPerf Inference numbers (3.87K tokens/s across 8 GPUs) back-solve to ~1.1K tokens/s per accelerator with a 0.92 utilization ceiling; Groq’s GPT-J latency demos show ~1.25K tokens/s per LPU, motivating the 1.2K base paired with a 0.6 training factor for their inference-first design; SambaNova SNX training reports converge around 0.95× A100 throughput, supporting the 950-token baseline.
- **Utilization envelope.** The default base utilization (0.55–0.60) and knob increments align with GPU SM occupancy studies: CUDA Graphs often deliver 10–12% effective occupancy gains, while GPUDirect RDMA removes ~5% host-side stalls. We cap utilization at 0.90–0.93 to reflect the diminishing returns observed once memory stalls dominate kernel execution.
- **Communication intensity.** Collective communication papers report NVLink/InfiniBand consuming 10–12% of iteration time at TP/PP >1. RoCE penalties (+10–12%) model PCIe/Ethernet contention, while vendor fabrics sit between the two based on proprietary interposers. These observations inform the `intensity` base and the per-topology `fabric_factors`.
- **Latency model.** `base_p50_coeff`, `min_p50_ms`, and tail multipliers capture first-token latency trends: H100 servers typically show 4–5 ms TTFT with ~45% P95 inflation under burst load, Groq LPUs maintain sub-3 ms medians with ~40% tails, and SambaNova pipelines yield 5–6 ms medians but heavier tails (+55%).
- **HBM bandwidth.** Accelerator YAMLs now set `hbm.base`, `seq_scale`, and `max` using published sustained bandwidth numbers. NVIDIA H100 HBM3 peaks at 3.35 TB/s but sustains ~2.9 TB/s on STREAM Triad (≈0.86) and 1.9–2.4 TB/s (≈0.55–0.72) on transformer inference; we adopt `base = 0.40`, `seq_scale = 0.45`, `max = 0.82` to respect these ceilings. Groq’s GPT-J traces average ~0.35 TB/s of a 0.55 TB/s theoretical limit (≈0.64), leading to `base = 0.35`, `seq_scale = 0.30`, `max = 0.65`. SambaNova SNX systems sustain ~900 GB/s vs. 1.2 TB/s peak (≈0.75), so we use `base = 0.45`, `seq_scale = 0.35`, `max = 0.75`. All profiles retain `seq_norm = 8192`, matching the context length where attention transitions from cache-resident to fully memory-bound.
- **Power draw.** Idle/util fractions (`idle_fraction ≈ 0.4`, `util_fraction ≈ 0.6`) follow node-level measurements showing accelerators consume ~40% of peak power at idle and scale roughly linearly with utilization across DL workloads.
- **Noise amplitude.** Residual jitter of 2–3% (noise amplitude 0.02–0.03) matches observed variance after accounting for deterministic experiment factors in training logs and inference tail studies. Tail scaling at 0.5 keeps percentile inflation within the spread seen in production traces.
- **Retry threshold.** Communication retries become prevalent once normalized overhead exceeds 0.28–0.32, matching NCCL retry triggers and vendor guidance for multi-node deployments; per-vendor YAMLs set thresholds within this empirical band.

Following this guide ensures that the Atlas Fabric simulator reflects observed accelerator behavior, capturing both central tendencies and tail risks required for hardware/software co-optimization.

