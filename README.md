# Atlas Fabric — Optimized Resource Benchmarking for Infrastructure & Training AI

## Overview

Atlas Fabric is a comprehensive benchmarking and decision-support framework for evaluating LLM infrastructure choices. It provides a unified interface to compare different accelerators, optimize configurations, and make data-driven infrastructure decisions.

- If you are interested in the underlying mathematical model for training and inference performance and cost estimation, see the companion document [`model_equations.md`](./model_equations.md). Every field referenced there must be provided in the YAML and knob inputs (the simulator no longer applies hidden defaults). For calibration methodology and tuning guidance for each hardware accelerator, refer to [`model_calibration.md`](./model_calibration.md).

### Key Features

- **Multi-Accelerator Support**: Example profiles for NVIDIA, Groq, and SambaNova (extendable via YAML)
- **Comprehensive Metrics**: Throughput, latency, cost, power efficiency
- **Configuration Optimization**: Automatic tuning for performance vs cost
- **Rich Visualizations**: Performance charts, cost analysis, utilization heatmaps
- **Interactive Demo**: Guided scenarios for common use cases
- **Detailed Reporting**: HTML and Markdown reports with insights

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

The demo provides guided scenarios:

```bash
python demo.py
```

Running the demo saves a Markdown report (`report.md`) inside the newly created `out/demo_*` directory for each scenario.

The default automated run executes two scenarios:
- `gpt5_1t_training`: multi-phase training throughput and cost evaluation across the accelerators defined in your YAML profiles.
- `gpt5_1t_inference`: traffic-aware inference capacity and cost comparison using the same target definitions.

You can extend these scenarios by adding additional models or workloads to `workload/workloads.yaml` and pointing them at new target definitions in `workload/targets.yaml`. Once the YAML entries are in place, update `demo.py` (or run the CLI directly) to reference the new workload names, and the demo will render reports for your custom scenarios alongside the defaults.

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
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐   │
│  │ LLM Workload │  │Accelerator │  │   Runtime    │   │
│  │     Specs    │  │    Specs   │  │    Tuning    │   │
│  └──────────────┘  └────────────┘  └──────────────┘   │
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

Considerations for `gpt5_1t_train`:
- Sequence length is 32k tokens; adjust if your curriculum uses shorter contexts early on.
- Parallelism defaults to `tp:8`, `pp:8`, `dp:16` (1024-way); modify to reflect the topology you plan to benchmark.
- Precision defaults to FP8 weights with BF16 activations; override via `workload.schedule` phases if you need different regimes.

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

## Visualization

Visualization helpers referenced in earlier versions were removed during the simulator refactor. The current reference CLI and demo runs emit structured JSON and Markdown reports for analysis.