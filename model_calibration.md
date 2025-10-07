# Model Calibration Guide

## Overview

This guide provides detailed procedures for empirically measuring the parameters required by the Atlas Fabric semi-empirical performance model. Accurate calibration is essential for reliable performance predictions.

**Key Principle**: Measure in controlled conditions, isolate variables, and validate against production workloads.

## 1. Hardware Performance Parameters

### 1.1 Base Token Throughput (`B_base`)

**What it measures**: Maximum tokens per second a single accelerator can process under ideal conditions.

**Measurement Procedure**:
```python
# 1. Run single-GPU benchmark with optimal batch size
# 2. Use a standard model (e.g., GPT-2 1.5B)
# 3. Disable all multi-GPU communication
# 4. Measure for different sequence lengths

for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    for seq_length in [512, 1024, 2048, 4096]:
        tokens_per_second = benchmark_single_gpu(
            model="gpt2-1.5B",
            batch_size=batch_size,
            sequence_length=seq_length,
            warmup_steps=10,
            measure_steps=100
        )
        record_measurement(tokens_per_second)

# B_base = maximum observed tokens_per_second
```

**Validation**: 
- Should be within 20% of theoretical peak (TFLOPS × model_efficiency)
- Consistent across multiple runs (< 5% variance)

### 1.2 Base Utilization (`rho_base`)

**What it measures**: Fraction of theoretical compute actually achieved in practice.

**Measurement Procedure**:
```python
# 1. Measure actual FLOPS during model execution
actual_flops = profile_gpu_flops(
    model=standard_model,
    duration=60_seconds,
    tool="nvidia-smi" or "rocm-smi"
)

# 2. Calculate theoretical FLOPS
theoretical_flops = num_accelerators * peak_tflops_per_accelerator * 1e12

# 3. Compute utilization
rho_base = actual_flops / theoretical_flops
```

**Expected Values**:
- Training: 0.4-0.6 (40-60% utilization)
- Inference: 0.3-0.5 (30-50% utilization)
- Memory-bound workloads: 0.2-0.4

### 1.3 Vendor Scaling Factor (`f_vendor`)

**What it measures**: Hardware-specific performance multipliers for different operational modes.

**Measurement Procedure**:
```python
# Baseline measurement (e.g., NVIDIA A100)
baseline_throughput = measure_throughput(
    hardware="A100",
    mode="training",
    model=standard_benchmark
)

# Target hardware measurement
target_throughput = measure_throughput(
    hardware=target_hardware,
    mode=mode,  # "training" or "inference"
    model=standard_benchmark
)

f_vendor[mode] = target_throughput / baseline_throughput
```

## 2. Communication Parameters

### 2.1 Communication Intensity (`chi_base`, `chi_tp`, `chi_pp`)

**What it measures**: Data transfer requirements for different parallelization strategies.

**Measurement Procedure**:
```python
# Measure baseline (no parallelism)
single_gpu_time = benchmark_time(tp=1, pp=1)

# Measure tensor parallelism scaling
for tp in [2, 4, 8]:
    multi_gpu_time = benchmark_time(tp=tp, pp=1)
    comm_overhead = (multi_gpu_time - single_gpu_time) / single_gpu_time
    
    # Use linear regression to find chi_tp
    # overhead = chi_base + chi_tp * (tp - 1)

# Measure pipeline parallelism scaling  
for pp in [2, 4, 8]:
    multi_gpu_time = benchmark_time(tp=1, pp=pp)
    comm_overhead = (multi_gpu_time - single_gpu_time) / single_gpu_time
    
    # Use linear regression to find chi_pp
    # overhead = chi_base + chi_pp * (pp - 1)
```

**Data Collection Points**:
- Measure with NCCL profiling enabled
- Record bytes transferred per step
- Calculate as fraction of compute time

### 2.2 Network Fabric Factors (`gamma_fabric`)

**What it measures**: Performance penalty for different interconnect technologies.

**Measurement Procedure**:
```python
fabric_measurements = {}

# Baseline: InfiniBand or NVLink
fabric_measurements["infiniband"] = measure_allreduce_bandwidth(
    fabric="infiniband",
    message_sizes=[1MB, 10MB, 100MB, 1GB],
    pattern="ring"
)

# Compare other fabrics
for fabric in ["ethernet_10g", "ethernet_25g", "ethernet_100g", "nvlink"]:
    bandwidth = measure_allreduce_bandwidth(fabric=fabric)
    gamma_fabric[fabric] = fabric_measurements["infiniband"] / bandwidth
```

**Expected Values**:
- NVLink/InfiniBand: 1.0 (baseline)
- 100G Ethernet: 1.2-1.5
- 25G Ethernet: 2.0-3.0
- Cloud vNIC: 2.5-4.0

### 2.3 Virtualization Jitter (`gamma_jitter`)

**What it measures**: Performance variability in virtualized/cloud environments.

**Measurement Procedure**:
```python
# Run identical workload multiple times
measurements = []
for i in range(100):
    runtime = benchmark_standard_workload()
    measurements.append(runtime)

# Calculate jitter metrics
mean_runtime = np.mean(measurements)
std_runtime = np.std(measurements)
p95_runtime = np.percentile(measurements, 95)

gamma_jitter = p95_runtime / mean_runtime
```

## 3. Latency Parameters

### 3.1 Base Latency Coefficients (`lat_min`, `lat_base_coeff`)

**What it measures**: Fundamental latency characteristics of the system.

**Measurement Procedure**:
```python
# Measure minimum achievable latency
lat_min = measure_latency(
    batch_size=1,
    sequence_length=1,
    model="minimal",  # Smallest possible model
    iterations=1000
).min()

# Measure scaling with compute
latencies = []
compute_rates = []
for batch_size in [1, 2, 4, 8, 16]:
    latency = measure_latency(batch_size=batch_size)
    rate = measure_throughput(batch_size=batch_size)
    latencies.append(latency)
    compute_rates.append(rate)

# Fit: latency = max(lat_min, lat_base_coeff / compute_rate)
lat_base_coeff = fit_inverse_relationship(latencies, compute_rates)
```

### 3.2 Tail Latency Multipliers

**What it measures**: How much worse tail latencies are compared to median.

**Measurement Procedure**:
```python
# Collect large sample of latencies under load
latencies = []
for i in range(10000):
    latency = measure_request_latency(
        load_level=0.7,  # 70% of peak throughput
        model=production_model
    )
    latencies.append(latency)

# Calculate percentiles
p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)

tail_multipliers = {
    "p95": p95 / p50,
    "p99": p99 / p50
}
```

**Production Validation**:
- Collect actual production traces
- Compare model predictions with observed percentiles
- Adjust multipliers if error > 10%

## 4. Power Parameters

### 4.1 Node Power Consumption (`p_node_nominal`)

**What it measures**: Total system power draw under load.

**Measurement Procedure**:
```python
# Use hardware power monitoring
power_measurements = []

# Idle power
idle_power = measure_power(
    duration=300,  # 5 minutes
    workload="idle"
)

# Full load power
max_power = measure_power(
    duration=300,
    workload="max_flops_benchmark"
)

p_node_nominal = max_power  # Rated power
```

**Tools**:
- IPMI/BMC power monitoring
- External power meters (most accurate)
- nvidia-smi / rocm-smi (GPU only)

### 4.2 Power Fractions (`phi_idle`, `phi_util`)

**What it measures**: How power scales with utilization.

**Measurement Procedure**:
```python
power_curve = []
utilizations = []

for util_target in [0, 0.25, 0.5, 0.75, 1.0]:
    # Run workload at target utilization
    workload = create_workload(target_util=util_target)
    power = measure_power(workload=workload)
    
    power_curve.append(power)
    utilizations.append(util_target)

# Fit linear model: power = p_nominal * (phi_idle + phi_util * util)
phi_idle = power_curve[0] / p_node_nominal  # Power at 0% util
phi_util = (power_curve[-1] - power_curve[0]) / p_node_nominal
```

**Expected Values**:
- `phi_idle`: 0.3-0.5 (30-50% of peak at idle)
- `phi_util`: 0.5-0.7 (additional 50-70% at full load)

## 5. Optimization Impact Parameters

### 5.1 CUDA Graphs Impact (`delta_rho_cuda`)

**What it measures**: Utilization improvement from CUDA Graphs optimization.

**Measurement Procedure**:
```python
# Baseline without CUDA Graphs
baseline_util = measure_utilization(
    cuda_graphs=False,
    model=standard_model,
    iterations=1000
)

# With CUDA Graphs enabled
optimized_util = measure_utilization(
    cuda_graphs=True,
    model=standard_model,
    iterations=1000
)

delta_rho_cuda = optimized_util - baseline_util
```

**Expected Improvement**: 0.05-0.15 (5-15% utilization increase)

### 5.2 GPUDirect RDMA Impact (`delta_rho_gpudirect`)

**What it measures**: Performance improvement from GPUDirect RDMA.

**Measurement Procedure**:
```python
# A/B testing approach
throughput_without = benchmark_multi_gpu(
    gpudirect_rdma=False,
    nodes=4,
    gpus_per_node=8
)

throughput_with = benchmark_multi_gpu(
    gpudirect_rdma=True,
    nodes=4,
    gpus_per_node=8
)

# Convert to utilization impact
delta_rho_gpudirect = (throughput_with - throughput_without) / theoretical_max
```

## 6. Validation Procedures

### 6.1 Cross-Validation

Run standard benchmarks across different configurations:
```python
validation_suite = [
    {"model": "gpt-3-13b", "batch": 32, "seq": 2048},
    {"model": "llama-70b", "batch": 8, "seq": 4096},
    {"model": "t5-11b", "batch": 64, "seq": 512}
]

for config in validation_suite:
    measured = run_actual_benchmark(config)
    predicted = model_predict(config, calibrated_params)
    error = abs(measured - predicted) / measured
    
    assert error < 0.15, f"Error {error:.1%} exceeds 15% threshold"
```

### 6.2 Production Validation

Compare model predictions with production metrics:
1. Collect 24-hour production traces
2. Extract p50, p95, p99 latencies
3. Compare with model predictions
4. Adjust parameters if systematic bias observed

### 6.3 Scaling Validation

Verify model accuracy across scale:
```python
for num_gpus in [1, 2, 4, 8, 16, 32, 64]:
    measured = benchmark_at_scale(num_gpus)
    predicted = model_predict(num_gpus)
    
    # Error should remain bounded as scale increases
    assert relative_error(measured, predicted) < 0.20
```

## 7. Calibration Frequency

**Initial Calibration**: 
- Complete measurement suite
- 2-3 days of dedicated testing
- Multiple validation runs

**Regular Updates**:
- Monthly: Spot-check key parameters
- Quarterly: Full recalibration
- After hardware changes: Complete remeasurement
- After software updates: Revalidate optimization impacts

## 8. Automation Tools

### Sample Calibration Script
```bash
#!/bin/bash
# Automated calibration pipeline

# Phase 1: Hardware characterization
python calibrate_hardware.py --output hardware_params.yaml

# Phase 2: Communication profiling  
python calibrate_communication.py --output comm_params.yaml

# Phase 3: Latency distribution
python calibrate_latency.py --output latency_params.yaml

# Phase 4: Power profiling
python calibrate_power.py --output power_params.yaml

# Phase 5: Validation
python validate_model.py --params *.yaml --threshold 0.15

# Generate report
python generate_calibration_report.py
```

## 9. Common Pitfalls

1. **Insufficient Warmup**: Always include warmup iterations before measurement
2. **Thermal Throttling**: Monitor GPU temperatures and power limits
3. **Background Processes**: Ensure exclusive access to hardware during calibration
4. **Network Congestion**: Test communication parameters in isolation
5. **Caching Effects**: Clear caches between measurements
6. **Statistical Significance**: Collect enough samples (minimum 100 for latency percentiles)

## 10. Parameter Confidence Intervals

Document uncertainty in measured parameters:

| Parameter | Typical Value | Confidence Interval | Measurement Error |
|-----------|--------------|-------------------|------------------|
| `B_base` | 500 tok/s | ±5% | ±2% |
| `rho_base` | 0.55 | ±0.05 | ±3% |
| `chi_tp` | 0.15 | ±0.03 | ±5% |
| `lat_min` | 5ms | ±1ms | ±10% |
| `p95_mult` | 2.5x | ±0.3x | ±8% |

## Summary

Accurate calibration requires:
1. **Systematic measurement** under controlled conditions
2. **Statistical rigor** with sufficient sample sizes
3. **Production validation** against real workloads
4. **Regular updates** as hardware/software evolves
5. **Documentation** of measurement procedures and confidence intervals

The semi-empirical model's accuracy depends entirely on the quality of these measurements. Invest time in proper calibration to ensure reliable performance predictions.