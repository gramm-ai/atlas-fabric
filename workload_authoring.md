# Creating Workload Configurations

How to add workloads to `workload/workloads.yaml`.

## File Structure

Each entry defines model architecture, precision, data pipeline, and optional schedules:

```yaml
- name: my_workload
  framework: pytorch
  model_family: gpt
  params:
    hidden_size: 12288
    n_layers: 96
    vocab_size: 200000
    n_heads: 96
  parallelism:
    tp: 4
    pp: 4
    dp: 8
  sequence:
    prompt: 4096
    generate: 4096
  precision:
    weights: fp8
    activations: bf16
  dataloader: synthetic
  duration_minutes: 5
  track: parity
  runtime_tuning:
    microbatch: 1
    cuda_graphs: true
    gpudirect_rdma: true
    virt: bare
    fabric: RoCE
  notes: "Optional description here"
  schedule:
    phases:
      - name: steady_state
        duration_minutes: 60
        request_profile: peak
```

## Required Fields

- `name`: Unique identifier
- `framework`: Runtime (`pytorch`, etc.)
- `model_family`: Model type (`gpt`, `llama`, `moe`)
- `params`: Model dimensions (`hidden_size`, `n_layers`, `vocab_size`)
- `parallelism`: TP/PP/DP degrees
- `sequence`: `prompt` and `generate` token counts (set `generate: 0` for training-only)
- `precision`: Weights and activations (`fp8`, `bf16`, `fp16`)
- `dataloader`: Data source (`streaming`, `synthetic`, `replay`)
- `duration_minutes`: Phase duration
- `track`: `parity` (production) or `best` (optimized)

Missing fields will cause errors.

## Optional Fields

- `notes`: Description or context
- `runtime_tuning`: Default execution knobs (see below)
- `schedule`: Multi-phase configurations
- Custom keys are preserved and passed through

### Runtime Tuning

Default execution knobs:

- `microbatch`: Batch size (int)
- `cuda_graphs`: Enable CUDA Graphs (bool)
- `gpudirect_rdma`: Enable GPUDirect (bool)
- `virt`: `bare` or `sriov`
- `fabric`: `IB`, `RoCE`, or `Vendor`

CLI flags override these defaults.

### Schedule Phases

For multi-phase workloads:

- `name`: Phase identifier
- `duration_minutes`: Phase weight
- `tokens_scale` / `autoscale_factor`: Throughput/power multipliers
- `request_profile`: Queue profile (`steady_state`, `peak`, `spike`, `low`)
- `overrides`: Modify base workload settings

No schedule = single phase with base config.

## Adding a New Workload

1. Copy an existing entry as template
2. Update `name` and required fields
3. Add `runtime_tuning` defaults (recommended)
4. Add `schedule` for multi-phase behavior
5. Test:

   ```bash
   python -m atlas_fabric.cli run \
     --targets workload/targets.yaml --target NVIDIA_H100_8x_NVLink \
     --workloads workload/workloads.yaml --workload my_workload \
     --out out
   ```

6. Check `out/last_run.json` for results

## Validation

- Missing fields: Check error messages for required keys
- Unrealistic metrics: Adjust `runtime_tuning` or schedules
- Document rationale in the `notes` field

## Custom Extensions

Add custom keys as needed - they're preserved and accessible to downstream code. For validated fields, update `atlas_fabric/specs.py`.

## Next Steps

- Add hardware profiles to `workload/targets.yaml` or accelerator YAMLs
- Include in `demo.py` for automated testing
- Generate reports with the new workload

