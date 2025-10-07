# Atlas Fabric — LLM Infrastructure Benchmarking

## Overview

Atlas Fabric benchmarks and compares LLM infrastructure options. Test different accelerators, optimize configs, and estimate costs for both training and inference workloads.

- See [`model_equations.md`](./model_equations.md) for the math behind performance and cost calculations
- See [`model_calibration.md`](./model_calibration.md) for hardware tuning and calibration

### Features

- **Multi-Accelerator**: NVIDIA, Groq, SambaNova profiles (add your own via YAML)
- **Key Metrics**: Throughput, latency, cost, power
- **Auto-Optimization**: Tunes performance vs cost trade-offs
- **Visualizations**: Performance charts, cost breakdowns, utilization maps
- **Demo Mode**: Pre-configured benchmark scenarios
- **Reports**: HTML and Markdown output with analysis

## Quick Start

### Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# (Optional) Extras for plotting or notebooks can be added as needed
```

### Basic Usage

#### 1. Run a Simple Benchmark

```bash
# Benchmark NVIDIA H100 on inference
python -m atlas_fabric.cli run \
  --targets workload/targets.yaml --target NVIDIA_H100_8x_NVLink \
  --workloads workload/workloads.yaml --workload gpt5_1t_inference \
  --track parity \
  --hardware accelerators \
  --seed 1337 \
  --out out

# View the result
cat out/last_run.json | python -m json.tool
```

#### 2. Optimize Configuration

```bash
# Find optimal settings for your workload
python -m atlas_fabric.cli optimize \
  --record out/last_run.json \
  --sla 100.0 \
  --out out
```

#### 3. Generate Report

```bash
# Create comparison report
python -m atlas_fabric.cli report \
  --records out \
  --out out/report.md
```

### Interactive Demo

Run pre-configured benchmarks:

```bash
python demo.py
```

This runs two default scenarios and saves reports to `out/demo_*/`:
- `gpt5_1t_training`: Training throughput and cost across accelerators
- `gpt5_1t_inference`: Inference capacity and latency under various traffic loads

To add custom scenarios:
1. Define workloads in `workload/workloads.yaml`
2. Add targets in `workload/targets.yaml`
3. Update `demo.py` or use the CLI directly

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────┐
│                 User Interface                      │
│        (CLI / Interactive Demo / Web UI)            │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                    Core Engine                      │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ LLM Workload │  │ Accelerator│  │   Runtime    │ │
│  │     Specs    │  │    Specs   │  │    Tuning    │ │
│  └──────────────┘  └────────────┘  └──────────────┘ │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                  Simulation Engine                  │
│  - Performance Model                                │
│  - Cost Calculator                                  │
│  - Power Estimator                                  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│               Analysis & Reporting                  │
│  - Optimizer           - Visualizer                 │
│  - Reporter            - Exporter                   │
└─────────────────────────────────────────────────────┘
```

## Usage Examples

### Scenario: Training Large Models

```bash
# GPT-5 scale training example
python -m atlas_fabric.cli run \
  --targets workload/targets.yaml \
  --target NVIDIA_H100_8x_NVLink \
  --workloads workload/workloads.yaml \
  --workload gpt5_1t_train \
  --track parity \
  --hardware accelerators \
  --seed 1337 \
  --out results
```

Key settings for `gpt5_1t_train`:
- Sequence length: 32k tokens
- Parallelism: `tp:8`, `pp:8`, `dp:16` (1024-way total)
- Precision: FP8 weights, BF16 activations

## Configuration

### Custom Workload Definition

```yaml
# my_workload.yaml
- name: custom_model
  framework: pytorch
  model_family: transformer
  params:
    hidden_size: 8192
    n_layers: 64
    vocab_size: 50000
  parallelism:
    tp: 4  # Tensor parallel
    pp: 2  # Pipeline parallel
    dp: 2  # Data parallel
  sequence:
    prompt: 4096
    generate: 1024
  precision:
    weights: fp16
    activations: fp16
  dataloader: synthetic
  duration_minutes: 10
  track: parity
```

### Custom Target Definition

```yaml
# my_target.yaml
- name: CUSTOM_ACCELERATOR
  vendor: CustomVendor
  instance_type: CA.X1
  accelerators_per_node: 8
  num_nodes: 1
  memory_gb_per_accel: 128
  interconnect:
    type: CustomLink
    bw_GBps: 1200
  host_cpu_arch: x86
  numeric: [bf16, fp16, int8]
  collectives: [allreduce, allgather]
  max_batch_tokens: 65536
  container_base: custom:latest
  cost:
    hourly_usd: 120.0
    energy_watts_node: 4000
    pue: 1.2
```

## Sample Output

The simulator generates JSON data and Markdown reports. Example visualizations:

- **Training comparison** — throughput and cost across accelerators
![Training performance comparison](./out/demo_20251006_214147_training/performance_comparison.png)

- **Inference latency** — latency distribution with queueing effects
![Inference latency distribution](./out/demo_20251006_214200_inference/latency_distribution.png)

### Documentation

- [`model_equations.md`](./model_equations.md) — Mathematical model and formulas (semi-empirical)
- [`model_calibration.md`](./model_calibration.md) — How to measure and calibrate model parameters
- [`workload_authoring.md`](./workload_authoring.md) — How to create custom workloads