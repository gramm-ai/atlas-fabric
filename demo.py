#!/usr/bin/env python3
"""
Atlas Fabric Demo Runner
Automated benchmarking showcase for training and inference cost scenarios
"""

import argparse
import json
import os
import sys
import time
import subprocess
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(step_num: int, text: str):
    """Print formatted step"""
    print(f"{Colors.CYAN}[Step {step_num}]{Colors.ENDC} {text}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} {text}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}[i]{Colors.ENDC} {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}[!]{Colors.ENDC} {text}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}[x]{Colors.ENDC} {text}")


SCENARIOS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "gpt5_1t_training",
        "name": "Training Performance with Phase Scheduling",
        "description": "Curriculum phases with schedule-driven knobs",
        "runs": (
            ("NVIDIA_H100_8x_NVLink", "gpt5_1t_train"),
            ("SAMBANOVA_SNX_NODE", "gpt5_1t_train"),
            ("GROQ_LPU_NODE", "gpt5_1t_train")
        ),
        "optimize": True,
        "report": True
    },
    {
        "key": "gpt5_1t_inference",
        "name": "Inference Cost with Traffic Profiles",
        "description": "Traffic-aware demand, autoscaling, and latency SLO tiers",
        "runs": (
            ("NVIDIA_H100_8x_NVLink", "gpt5_1t_inference"),
            ("SAMBANOVA_SNX_NODE", "gpt5_1t_inference"),
            ("GROQ_LPU_NODE", "gpt5_1t_inference")
        ),
        "optimize": True,
        "report": True,
        "focus": "cost"
    }
)

DEFAULT_SCENARIO_KEYS: Tuple[str, ...] = (
    "gpt5_1t_training",
    "gpt5_1t_inference",
)

OUTPUT_KEY_ALIASES: Dict[str, str] = {
    "gpt5_1t_training": "training",
    "gpt5_1t_inference": "inference",
}


def progress_bar(current: int, total: int, text: str = "", width: int = 24):
    """Display a progress bar"""
    percent = current / total
    filled = int(width * percent)
    bar = "=" * filled + "." * (width - filled)
    print(f"\r{text}\t\t[{bar}] {percent*100:.1f}%", end="", flush=True)
    if current >= total:
        print()

def _run_module(module_name:str, module_args:list[str]) -> int:
    """Run a Python module with args using the current interpreter, silencing output.
    Returns the subprocess return code.
    """
    try:
        proc = subprocess.run([sys.executable, "-m", module_name, *module_args],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              check=False)
        return proc.returncode
    except Exception:
        return 1

def run_benchmark(target: str, workload: str, output_dir: str = "out", hardware_dir: str = "accelerators") -> Dict[str, Any]:
    """Run a single benchmark"""
    args = [
        "run",
        "--targets", "workload/targets.yaml",
        "--target", target,
        "--workloads", "workload/workloads.yaml",
        "--workload", workload,
        "--track", "parity",
        "--hardware", hardware_dir,
        "--out", output_dir,
    ]
    
    # Simulate execution with progress
    for i in range(11):
        progress_bar(i, 10, f"  Running {target[:20]:20}")
        time.sleep(0.1)
    
    # Execute the actual command
    result = _run_module("atlas_fabric.cli", args)
    
    if result == 0:
        # Load and return the result
        result_file = f"{output_dir}/last_run.json"
        if os.path.exists(result_file):
            with open(result_file) as f:
                return json.load(f)
    else:
        print_error(f"Failed to run {target} on {workload}")
    
    return {}

def optimize_config(record_file: str, output_dir: str = "out", hardware_dir: str = "accelerators") -> Dict[str, Any]:
    """Run optimization on a configuration"""
    args = [
        "optimize",
        "--record", record_file,
        "--hardware", hardware_dir,
        "--out", output_dir,
    ]
    
    try:
        with open(record_file) as f:
            record = json.load(f)
    except Exception:
        record = {}

    accel_name = record.get("target", {}).get("name", "configuration")
    for i in range(6):
        progress_bar(i, 5, f"  Optimizing {accel_name}")
        time.sleep(0.1)
    
    result = _run_module("atlas_fabric.cli", args)
    
    if result == 0:
        result_file = f"{output_dir}/last_run.json"
        if os.path.exists(result_file):
            with open(result_file) as f:
                return json.load(f)
    else:
        print_error("Optimization failed")
    
    return {}

def generate_report(output_dir: str = "out") -> bool:
    """Generate the comparison report"""
    args = [
        "report",
        "--records", output_dir,
        "--out", f"{output_dir}/report.md",
    ]
    
    result = _run_module("atlas_fabric.cli", args)
    
    if result == 0:
        print_success(f"Report generated: {output_dir}/report.md")
        return True
    else:
        print_error("Report generation failed")
        return False

def _discover_targets_from_hardware(hardware_dir: str, targets_yaml: str) -> list[str]:
    """Discover target names whose vendor or normalized name has a hardware card."""
    try:
        import yaml
        # Load all targets
        with open(targets_yaml) as f:
            targets = yaml.safe_load(f)
        if not isinstance(targets, list):
            # dict form not expected here, but handle gracefully
            targets = list(targets.values())
    except Exception:
        return []

    # Build available hardware keys
    available = set()
    if os.path.isdir(hardware_dir):
        for fname in os.listdir(hardware_dir):
            if fname.lower().endswith((".yaml", ".yml")):
                key = os.path.splitext(fname)[0].lower()
                available.add(key)

    def _norm(s: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in s)

    discovered = []
    for t in targets:
        name = t.get("name", "")
        vendor = t.get("vendor", "").lower()
        nname = _norm(name)
        if vendor in available or nname in available:
            discovered.append(name)
    return discovered

def display_results_summary(results: List[Dict[str, Any]]):
    """Display a formatted summary of results"""
    if not results:
        return

    # Decide header based on workload type in this result set
    try:
        valid_results_probe = [r for r in results if r and "workload" in r]
        is_inference_set = (
            len(valid_results_probe) > 0 and all("inference" in r["workload"]["name"] for r in valid_results_probe)
        )
    except Exception:
        is_inference_set = False

    header_text = "INFERENCE PERFORMANCE" if is_inference_set else "TRAINING PERFORMANCE"
    print_header(header_text)

    # Create summary table
    print(f"{'Accelerator':30} {'Workload':20} {'Tokens/sec':>12} {'$/1k tokens':>12} {'P99 Latency':>14}")
    print("-" * 92)
    
    valid_results = [r for r in results if r and "target" in r and "workload" in r]
    label_order = {"pre-opt": 0, "post-opt": 1}
    valid_results.sort(key=lambda r: (
        r["target"]["name"],
        r["workload"]["name"],
        label_order.get(r.get("label"), 2),
        r.get("label", "")
    ))

    for result in valid_results:
        label = result.get("label")
        target_display = result["target"]["name"]
        if label:
            target_display = f"{target_display} {label}"
        target_name = target_display[:29]
        workload_name = result["workload"]["name"][:19]
        tokens_sec = result["record"]["tokens_per_sec"]
        cost = result.get("dollar_per_1k_tokens", 0)
        # Show P99 only for inference workloads; otherwise display N/A
        row_is_infer = "inference" in result["workload"].get("name", "")
        if row_is_infer:
            p99_val = result["record"].get("p99_ms", 0.0)
            p99_str = f"{p99_val:12.2f}ms"
        else:
            p99_str = f"{'N/A':>12}  "

        print(f"{target_name:30} {workload_name:20} {tokens_sec:12.2f} {cost:12.4f} {p99_str}")
    

def visualize_results(results_dir: str):
    """Generate visualizations from results"""
    try:
        from atlas_fabric.visualizer import BenchmarkVisualizer

        print_header("GENERATING VISUALIZATIONS")

        viz = BenchmarkVisualizer(results_dir)
        results = viz.load_results(results_dir)

        if not results:
            # fall back to immediate subdirectories (e.g., out/demo_*)
            from pathlib import Path

            root = Path(results_dir)
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    child_results = viz.load_results(str(child))
                    if child_results:
                        print_info(f"Found results in {child}")
                        results = child_results
                        viz = BenchmarkVisualizer(str(child))
                        results_dir = str(child)
                        break

        if not results:
            print_warning("No results found to visualize")
            return

        print_info(f"Loaded {len(results)} benchmark results")

        # Generate charts
        charts = viz.generate_all_charts(results)

        for chart_type, chart_path in charts.items():
            if chart_path:
                print_success(f"Generated {chart_type}: {chart_path}")

        # Generate HTML report
        html_path = viz.create_html_report(results, charts)
        print_success(f"HTML report: {html_path}")

        print(f"\n{Colors.GREEN}Visualizations complete!{Colors.ENDC}")
        print(f"Open {html_path} in your browser to view the interactive report.")

    except ImportError:
        print_warning("Matplotlib not installed. Install with: pip install matplotlib numpy")
        print_info("Falling back to text-based reports")


def _scenario_list() -> List[Dict[str, Any]]:
    return [dict(scenario) for scenario in SCENARIOS]


def _sanitize_key(raw_key: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in raw_key).strip('_')


def run_scenario(scenario: Dict[str, Any], base_output_dir: str, *, skip_visuals: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute a single scenario end-to-end and return the output directory plus results."""
    raw_key = scenario.get("key", scenario.get("name", "scenario")).lower()
    alias_key = OUTPUT_KEY_ALIASES.get(raw_key, raw_key)
    sanitized_key = _sanitize_key(alias_key)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scenario_dir = Path(base_output_dir) / f"demo_{timestamp}_{sanitized_key}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print_header(f"Scenario: {scenario['name']}")

    results: List[Dict[str, Any]] = []
    step = 1
    hardware_dir = "accelerators"

    print_step(step, "Running benchmarks...")
    runs_cfg = scenario.get("runs")
    if runs_cfg == "DISCOVER_ALL":
        targets = _discover_targets_from_hardware(hardware_dir, "workload/targets.yaml")
        workloads = scenario.get("workloads") or [scenario.get("workload", "inference_decode")]
        for target in targets:
            for workload in workloads:
                result = run_benchmark(target, workload, str(scenario_dir), hardware_dir)
                if result:
                    result["label"] = "pre-opt"
                    results.append(result)
    elif isinstance(runs_cfg, (list, tuple)):
        for target, workload in runs_cfg:
            result = run_benchmark(target, workload, str(scenario_dir), hardware_dir)
            if result:
                result["label"] = "pre-opt"
                results.append(result)
    else:
        print_warning("Scenario runs configuration missing; skipping benchmark execution.")
    step += 1

    if scenario.get("optimize") and results:
        print_step(step, "Optimizing configurations...")
        for json_file in scenario_dir.iterdir():
            if json_file.suffix == ".json" and "last_run" not in json_file.name:
                opt_result = optimize_config(str(json_file), str(scenario_dir), hardware_dir)
                if opt_result:
                    opt_result["label"] = "post-opt"
                    summary = {
                        "source": "optimize",
                        "baseline_file": json_file.name
                    }
                    opt_result.setdefault("metadata", {}).update(summary)
                    results.append(opt_result)
        step += 1

    if scenario.get("report"):
        print_step(step, "Generating report...")
        generate_report(str(scenario_dir))
        step += 1

    display_results_summary(results)

    html_path = scenario_dir / "report.html"
    report_path = scenario_dir / "report.md"

    if not skip_visuals:
        print_step(step, "Creating visualization package...")
        visualize_results(str(scenario_dir))
        step += 1

    print(f"\n{Colors.GREEN}Scenario complete!{Colors.ENDC}")
    print(f"Detailed report saved to: {report_path}")
    if html_path.exists():
        print(f"Visualization report saved to: {html_path}")

    return str(scenario_dir), results

def main():
    parser = argparse.ArgumentParser(description="Atlas Fabric Demo Runner")
    parser.add_argument("--output", default="out",
                      help="Output directory for results")
    parser.add_argument("--scenario", choices=[s["key"] for s in SCENARIOS],
                      help="Run only the specified scenario key")
    parser.add_argument("--skip-visuals", action="store_true",
                      help="Skip visualization generation")

    args = parser.parse_args()

    # Check if atlas_fabric module exists
    try:
        import atlas_fabric
    except ImportError:
        print_error("Error: atlas_fabric module not found. Please ensure you're in the project directory.")
        sys.exit(1)

    print_header("Atlas Fabric Automated Demo")

    selected_keys = [args.scenario] if args.scenario else list(DEFAULT_SCENARIO_KEYS)
    available = {scenario["key"]: scenario for scenario in SCENARIOS}

    run_any = False
    for key in selected_keys:
        scenario = available.get(key)
        if not scenario:
            print_warning(f"Scenario '{key}' not found; skipping.")
            continue

        scenario_dir, results = run_scenario(scenario, args.output, skip_visuals=args.skip_visuals)
        run_any = True

        if args.skip_visuals:
            print_info("Visualizations were requested to be skipped; remove --skip-visuals to generate them.")

    if not run_any:
        print_warning("No scenarios were executed. Specify --scenario or ensure defaults are available.")

if __name__ == "__main__":
    main()

