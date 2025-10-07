"""
Atlas Fabric Visualization Module
Generate charts and visualizations from benchmark results
"""

from __future__ import annotations
import json
import os
import html as html_module
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class BenchmarkVisualizer:
    """Generate visualizations from benchmark results"""
    
    def __init__(self, output_dir: str = "out"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_results(self, results_dir: str) -> List[Dict[str, Any]]:
        """Load all JSON results from a directory"""
        results = []
        results_path = Path(results_dir)
        
        for json_file in results_path.glob("*.json"):
            if "last_run" not in json_file.name:
                try:
                    with open(json_file) as f:
                        results.append(json.load(f))
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        return results
    
    def generate_performance_comparison(self, results: List[Dict[str, Any]], 
                                       save_path: Optional[str] = None) -> str:
        """Generate a performance comparison bar chart"""
        if not HAS_MATPLOTLIB:
            return self._generate_ascii_chart(results, "performance")
        
        # Collect pre/post optimization data per accelerator
        perf_by_target: Dict[str, Dict[str, Any]] = {}
        for result in results:
            if not result:
                continue

            target_info = result.get("target", {})
            workload_info = result.get("workload", {})
            record = result.get("record", {})

            target = target_info.get("name")
            tokens_per_sec = record.get("tokens_per_sec")
            if not target or tokens_per_sec is None:
                continue

            label = (result.get("label") or "").lower()
            workload_name = workload_info.get("name")

            entry = perf_by_target.setdefault(target, {
                "pre": None,
                "post": None,
                "workloads": set()
            })

            if workload_name:
                entry["workloads"].add(workload_name)

            if label == "post-opt":
                entry["post"] = max(entry["post"], tokens_per_sec) if entry["post"] is not None else tokens_per_sec
            else:
                # Treat explicit pre-opt label as priority, otherwise baseline counts as pre-opt
                if label == "pre-opt" or entry["pre"] is None:
                    entry["pre"] = tokens_per_sec
                else:
                    entry["pre"] = max(entry["pre"], tokens_per_sec)

        if not perf_by_target:
            return ""

        # Sort targets for consistent display
        sorted_targets = sorted(perf_by_target.keys())

        pre_vals = []
        post_vals = []
        tick_labels = []
        for target in sorted_targets:
            entry = perf_by_target[target]
            pre_vals.append(entry["pre"] or 0)
            post_vals.append(entry["post"] or 0)

            workloads = sorted(entry["workloads"])
            if workloads:
                tick_labels.append(f"{target}\n{', '.join(workloads)}")
            else:
                tick_labels.append(target)

        x = np.arange(len(sorted_targets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(sorted_targets) * 1.8), 6))
        bars_pre = ax.bar(x - width / 2, pre_vals, width, label='Pre-Optimization', color='#4C78A8')
        bars_post = ax.bar(x + width / 2, post_vals, width, label='Post-Optimization', color='#F58518')

        # Annotate bars with values or N/A when missing
        for idx, bar in enumerate(bars_pre):
            height = bar.get_height()
            entry = perf_by_target[sorted_targets[idx]]
            if entry["pre"] is None:
                ax.text(bar.get_x() + bar.get_width() / 2, height + max(height * 0.02, 1),
                        'N/A', ha='center', va='bottom', fontsize=8)
            elif height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8)

        for idx, bar in enumerate(bars_post):
            height = bar.get_height()
            entry = perf_by_target[sorted_targets[idx]]
            if entry["post"] is None:
                ax.text(bar.get_x() + bar.get_width() / 2, height + max(height * 0.02, 1),
                        'N/A', ha='center', va='bottom', fontsize=8)
            elif height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Accelerator', fontweight='bold')
        ax.set_ylabel('Tokens/Second', fontweight='bold')
        ax.set_title('Performance Comparison Pre vs Post Optimization', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, ha='center')
        ax.tick_params(axis='x', labelrotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "performance_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_cost_efficiency_chart(self, results: List[Dict[str, Any]], 
                                      save_path: Optional[str] = None) -> str:
        """Generate a cost efficiency scatter plot"""
        if not HAS_MATPLOTLIB:
            return self._generate_ascii_chart(results, "cost")
        
        # Prepare data
        data_points = []
        for result in results:
            if not result:
                continue
            
            tokens_sec = result["record"]["tokens_per_sec"]
            cost = result.get("dollar_per_1k_tokens", 0)
            target = result["target"]["name"]
            workload = result["workload"]["name"]
            opt_label = result.get("label", "unknown")
            
            data_points.append({
                "tokens_sec": tokens_sec,
                "cost": cost,
                "target": target,
                "workload": workload,
                "opt_label": opt_label,
                "label": f"{target.split('_')[0]}_{workload.split('_')[0]}"
            })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by target
        targets = list(set(dp["target"] for dp in data_points))
        colors = plt.cm.Set2(np.linspace(0, 1, len(targets)))
        target_colors = {t: colors[i] for i, t in enumerate(targets)}
        
        for dp in data_points:
            ax.scatter(dp["tokens_sec"], dp["cost"], 
                      color=target_colors[dp["target"]],
                      s=100, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Create annotation with optimization status
            annotation_text = f"{dp['label']}\n({dp['opt_label']})"
            ax.annotate(annotation_text, (dp["tokens_sec"], dp["cost"]),
                       fontsize=8, ha='center', va='bottom',
                       xytext=(0, 6), textcoords='offset points')
        
        ax.set_xlabel('Throughput (Tokens/Second)', fontweight='bold')
        ax.set_ylabel('Cost ($/1k Tokens)', fontweight='bold')

        workload_names = {dp["workload"].lower() for dp in data_points}
        if workload_names and all("train" in name for name in workload_names):
            chart_title = 'Training Cost Efficiency Comparison'
        elif workload_names and all("inference" in name for name in workload_names):
            chart_title = 'Inference Cost Efficiency Comparison'
        else:
            chart_title = 'Cost Efficiency Comparison'

        ax.set_title(chart_title, fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=target_colors[t], 
                            markersize=10, label=t) for t in targets]
        ax.legend(handles=handles, loc='upper right')
        
        # Efficiency frontier plot removed per latest requirements
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "cost_efficiency.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_latency_distribution(self, results: List[Dict[str, Any]], 
                                     save_path: Optional[str] = None) -> str:
        """Generate latency distribution chart for inference workloads"""
        if not HAS_MATPLOTLIB:
            return self._generate_ascii_chart(results, "latency")
        
        # Filter inference results
        inference_results = [r for r in results 
                           if r and "inference" in r["workload"]["name"]]
        
        if not inference_results:
            return ""
        
        # Prepare data
        targets = sorted(set(r["target"]["name"] for r in inference_results))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(targets))
        width = 0.25
        
        p50_values = []
        p95_values = []
        p99_values = []
        
        for target in targets:
            target_results = [r for r in inference_results 
                             if r["target"]["name"] == target]
            if target_results:
                result = target_results[0]
                p50_values.append(result["record"].get("p50_ms", 0))
                p95_values.append(result["record"].get("p95_ms", 0))
                p99_values.append(result["record"].get("p99_ms", 0))
            else:
                p50_values.append(0)
                p95_values.append(0)
                p99_values.append(0)
        
        bars1 = ax.bar(x - width, p50_values, width, label='P50', color='green', alpha=0.7)
        bars2 = ax.bar(x, p95_values, width, label='P95', color='yellow', alpha=0.7)
        bars3 = ax.bar(x + width, p99_values, width, label='P99', color='red', alpha=0.7)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Accelerator', fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.set_title('Inference Latency Distribution (first-token response)',
                     fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "latency_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_utilization_heatmap(self, results: List[Dict[str, Any]], 
                                    save_path: Optional[str] = None) -> str:
        """Generate resource utilization heatmap"""
        if not HAS_MATPLOTLIB:
            return self._generate_ascii_chart(results, "utilization")
        
        # Prepare data
        grouped = {}
        for result in results:
            if not result:
                continue
            key = (result["target"]["name"], result["workload"]["name"])
            grouped[key] = result
        
        targets = sorted(list(set(k[0] for k in grouped.keys())))
        workloads = sorted(list(set(k[1] for k in grouped.keys())))
        
        # Create utilization matrices
        flop_util = np.zeros((len(workloads), len(targets)))
        hbm_util = np.zeros((len(workloads), len(targets)))
        
        for i, workload in enumerate(workloads):
            for j, target in enumerate(targets):
                key = (target, workload)
                if key in grouped:
                    flop_util[i, j] = grouped[key]["record"]["flop_utilization"] * 100
                    hbm_util[i, j] = grouped[key]["record"]["hbm_bw_util"] * 100
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # FLOP Utilization heatmap
        im1 = ax1.imshow(flop_util, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax1.set_xticks(np.arange(len(targets)))
        ax1.set_yticks(np.arange(len(workloads)))
        ax1.set_xticklabels(targets, rotation=45, ha='right')
        ax1.set_yticklabels(workloads)
        ax1.set_title('FLOP Utilization (%)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(workloads)):
            for j in range(len(targets)):
                if flop_util[i, j] > 0:
                    text = ax1.text(j, i, f'{flop_util[i, j]:.1f}',
                                  ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im1, ax=ax1)
        
        # Memory Bandwidth Utilization heatmap
        im2 = ax2.imshow(hbm_util, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax2.set_xticks(np.arange(len(targets)))
        ax2.set_yticks(np.arange(len(workloads)))
        ax2.set_xticklabels(targets, rotation=45, ha='right')
        ax2.set_yticklabels(workloads)
        ax2.set_title('Memory Bandwidth Utilization (%)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(workloads)):
            for j in range(len(targets)):
                if hbm_util[i, j] > 0:
                    text = ax2.text(j, i, f'{hbm_util[i, j]:.1f}',
                                  ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im2, ax=ax2)
        
        plt.suptitle('Resource Utilization Analysis', fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "utilization_heatmap.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _generate_ascii_chart(self, results: List[Dict[str, Any]], chart_type: str) -> str:
        """Generate ASCII art charts when matplotlib is not available"""
        print(f"Matplotlib not available; skipping {chart_type} chart generation.")
        return ""
    
    def generate_all_charts(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate all available charts"""
        chart_paths = {}
        
        perf_chart = self.generate_performance_comparison(results)
        if perf_chart:
            chart_paths['performance'] = perf_chart
        
        cost_chart = self.generate_cost_efficiency_chart(results)
        if cost_chart:
            chart_paths['cost_efficiency'] = cost_chart
        
        # Only generate latency chart if there are inference results
        if any(r and "inference" in r["workload"]["name"] for r in results):
            latency_chart = self.generate_latency_distribution(results)
            if latency_chart:
                chart_paths['latency'] = latency_chart
        
        util_chart = self.generate_utilization_heatmap(results)
        if util_chart:
            chart_paths['utilization'] = util_chart

        schedule_chart = self.generate_schedule_overhead_chart(results)
        if schedule_chart:
            chart_paths['schedule_overhead'] = schedule_chart
        
        return chart_paths

    def generate_schedule_overhead_chart(self, results: List[Dict[str, Any]],
                                         save_path: Optional[str] = None) -> str:
        """Compare steady-state vs optimized vs schedule-adjusted throughput."""
        metrics = self._collect_schedule_metrics(results)
        if not metrics:
            return ""

        if not HAS_MATPLOTLIB:
            return self._generate_ascii_chart(results, "schedule_overhead")

        sorted_items = sorted(metrics.items(), key=lambda x: (x[0][0], x[0][1]))
        categories = [f"{t}\n{w}" for (t, w), _ in sorted_items]
        pre_vals = []
        post_vals = []
        epoch_vals = []
        pre_mbs = []
        post_mbs = []

        for _, data in sorted_items:
            pre_val = data.get("pre_opt_linear") or 0.0
            post_val = data.get("post_opt")
            if post_val is None:
                post_val = pre_val
            epoch_val = data.get("schedule_epoch") or 0.0

            pre_vals.append(pre_val)
            post_vals.append(post_val)
            epoch_vals.append(epoch_val)
            pre_mbs.append(data.get("pre_microbatch"))
            post_mbs.append(data.get("post_microbatch"))

        x = np.arange(len(categories))
        width = 0.22

        fig, ax = plt.subplots(figsize=(max(10, len(categories) * 2.5), 6))
        bars_pre = ax.bar(x - width, pre_vals, width, label='Pre-Opt Linear', color='#4C78A8')
        bars_post = ax.bar(x, post_vals, width, label='Post-Opt Linear', color='#F58518')
        bars_epoch = ax.bar(x + width, epoch_vals, width, label='Schedule Epoch', color='#54A24B')

        ax.set_ylabel('Tokens per Second', fontweight='bold')
        ax.set_title('Steady-State vs Schedule Impact', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, ha='center')
        ax.tick_params(axis='x', labelrotation=0)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()

        # Annotate bars with numeric values (tokens/sec) on top of each bar
        def _format_value(value: float) -> str:
            try:
                # Use thousands separator; fall back to one decimal for small values
                return f"{value:,.0f}" if value >= 10 else f"{value:.1f}"
            except Exception:
                return f"{value}"

        for bars in (bars_pre, bars_post, bars_epoch):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        _format_value(height),
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "schedule_overhead.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def _collect_schedule_metrics(self, results: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for result in results:
            if not result:
                continue

            target = result.get("target", {}).get("name")
            workload = result.get("workload", {}).get("name")
            record = result.get("record", {})
            label = (result.get("label") or "").lower()

            if not target or not workload:
                continue

            key = (target, workload)
            entry = metrics.setdefault(key, {})

            if label == "pre-opt":
                schedule_breakdown = record.get("schedule_breakdown")
                if isinstance(schedule_breakdown, list) and schedule_breakdown:
                    main_phase = max(schedule_breakdown, key=lambda phase: phase.get("weight_minutes", 0.0))
                    phase_record = main_phase.get("record", {})
                    entry["pre_opt_linear"] = phase_record.get("tokens_per_sec", record.get("tokens_per_sec"))
                else:
                    entry["pre_opt_linear"] = record.get("tokens_per_sec")

                entry["schedule_epoch"] = record.get("tokens_per_sec")
                entry["pre_microbatch"] = result.get("knobs", {}).get("microbatch")

            elif label == "post-opt":
                entry["post_opt"] = record.get("tokens_per_sec")
                entry["post_microbatch"] = result.get("knobs", {}).get("microbatch")

        # Remove entries that are missing required data
        cleaned = {
            key: value
            for key, value in metrics.items()
            if value.get("pre_opt_linear") is not None and value.get("schedule_epoch") is not None
        }

        return cleaned
    
    def create_html_report(self, results: List[Dict[str, Any]], 
                          chart_paths: Dict[str, str],
                          save_path: Optional[str] = None) -> str:
        """Create an HTML report with embedded charts and results"""
        html = []
        html.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Atlas Fabric Benchmark Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { 
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        .summary-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #3498db;
            color: white;
        }
        tr:hover {
            background: #f1f1f1;
        }
        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.75em;
            font-weight: 600;
            margin-left: 8px;
            background: #ecf0f1;
            color: #2c3e50;
        }
        .tag.pre-opt {
            background: #e8f6ff;
            color: #1b76d2;
        }
        .tag.post-opt {
            background: #f5e8ff;
            color: #7b2cbf;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <h1>Atlas Fabric Benchmark Report</h1>
    <p><strong>Generated:</strong> """ + str(Path().cwd()) + """</p>
""")
        
        # Summary metrics
        if results:
            best_throughput = max(results, key=lambda x: x["record"]["tokens_per_sec"] if x else 0)
            best_cost = min(results, key=lambda x: x.get("dollar_per_1k_tokens", float('inf')) if x else float('inf'))
            total_runs = len(results)
            
            html.append("""
    <div class="summary-card">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Total Benchmarks</div>
                <div class="metric-value">""" + str(total_runs) + """</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Best Throughput</div>
                <div class="metric-value">""" + f"{best_throughput['record']['tokens_per_sec']:.0f}" + """</div>
                <div class="metric-label">tokens/sec</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Best Cost</div>
                <div class="metric-value">$""" + f"{best_cost['dollar_per_1k_tokens']:.3f}" + """</div>
                <div class="metric-label">per 1k tokens</div>
            </div>
        </div>
    </div>
""")
        
        # Results table
        html.append("""
    <div class="summary-card">
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Accelerator</th>
                    <th>Workload</th>
                    <th>Throughput (tokens/sec)</th>
                    <th>Cost ($/1k tokens)</th>
                    <th>P99 Latency (ms)</th>
                    <th>FLOP Util (%)</th>
                </tr>
            </thead>
            <tbody>
""")
        
        def _label_order(item: Dict[str, Any]) -> tuple:
            label_value = (item.get('label') or '').lower()
            label_rank = {"pre-opt": 0, "post-opt": 1}.get(label_value, 2)
            throughput = item['record'].get('tokens_per_sec', 0)
            return (label_rank, -throughput)

        def _format_label(label: Optional[str]) -> str:
            if not label:
                return ''
            normalized = html_module.escape(label.lower().replace(' ', '-'))
            safe_label = html_module.escape(label.upper())
            return f'<span class="tag {normalized}">{safe_label}</span>'

        grouped_results: Dict[tuple, List[Dict[str, Any]]] = {}
        for result in results:
            if not result:
                continue
            key = (
                result['target']['name'],
                result['workload']['name'],
                result.get('track', '')
            )
            grouped_results.setdefault(key, []).append(result)

        for key in sorted(grouped_results.keys()):
            group = sorted(grouped_results[key], key=_label_order)
            for result in group:
                label_html = _format_label(result.get('label'))
                accelerator_cell = f"{html_module.escape(result['target']['name'])}{label_html}"
                html.append(
                    "                <tr>\n"
                    f"                    <td>{accelerator_cell}</td>\n"
                    f"                    <td>{html_module.escape(result['workload']['name'])}</td>\n"
                    f"                    <td>{result['record']['tokens_per_sec']:.2f}</td>\n"
                    f"                    <td>${result.get('dollar_per_1k_tokens', 0):.4f}</td>\n"
                    f"                    <td>{(result['record'].get('p99_ms') if 'inference' in result['workload'].get('name','') else 'N/A') if ('inference' in result['workload'].get('name','')) else 'N/A'}</td>\n"
                    f"                    <td>{result['record']['flop_utilization']*100:.1f}%</td>\n"
                    "                </tr>\n"
                )
        
        html.append("""
            </tbody>
        </table>
    </div>
""")
        
        # Schedule summary table (if applicable)
        schedule_metrics = self._collect_schedule_metrics(results)
        if schedule_metrics:
            html.append("""
    <div class="summary-card">
        <h2>Schedule Impact Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Accelerator</th>
                    <th>Workload</th>
                    <th>Pre-Opt MB</th>
                    <th>Pre-Opt Linear TPS</th>
                    <th>Post-Opt MB</th>
                    <th>Post-Opt TPS</th>
                    <th>Schedule Epoch TPS</th>
                    <th>Overhead vs Pre</th>
                </tr>
            </thead>
            <tbody>
""")

            for key in sorted(schedule_metrics.keys(), key=lambda k: (k[0], k[1])):
                target, workload = key
                data = schedule_metrics[key]

                pre_mb = data.get("pre_microbatch")
                post_mb = data.get("post_microbatch")
                pre_tps = data.get("pre_opt_linear") or 0.0
                post_tps = data.get("post_opt") or 0.0
                epoch_tps = data.get("schedule_epoch") or 0.0

                overhead_pct = "—"
                if pre_tps > 0 and epoch_tps > 0:
                    overhead_pct = f"{max(0.0, (1 - (epoch_tps / pre_tps)) * 100):.1f}%"

                html.append(
                    "                <tr>\n"
                    f"                    <td>{html_module.escape(target)}</td>\n"
                    f"                    <td>{html_module.escape(workload)}</td>\n"
                    f"                    <td>{html_module.escape(str(pre_mb) if pre_mb is not None else '—')}</td>\n"
                    f"                    <td>{pre_tps:.2f}</td>\n"
                    f"                    <td>{html_module.escape(str(post_mb) if post_mb is not None else '—')}</td>\n"
                    f"                    <td>{post_tps:.2f}</td>\n"
                    f"                    <td>{epoch_tps:.2f}</td>\n"
                    f"                    <td>{overhead_pct}</td>\n"
                    "                </tr>\n"
                )

            html.append("""
            </tbody>
        </table>
    </div>
""")

        # Charts
        html.append('<h2>Performance Visualizations</h2>')

        for chart_name in ["performance", "cost_efficiency", "latency", "utilization", "schedule_overhead"]:
            chart_path = chart_paths.get(chart_name)
            chart_title = chart_name.replace('_', ' ').title()

            if chart_path and os.path.exists(chart_path):
                suffix = Path(chart_path).suffix.lower()

                if suffix == ".png":
                    rel_path = Path(chart_path).name
                    html.append(f"""
    <div class="chart-container">
        <h3>{chart_title}</h3>
        <img src="{rel_path}" alt="{chart_title}">
    </div>
""")
            else:
                html.append(f"""
    <div class="chart-container">
        <h3>{chart_title}</h3>
        <p>Chart skipped (matplotlib not available).</p>
    </div>
""")
        
        # Footer
        html.append("""
    <div class="footer">
        <p>Generated by Atlas Fabric Benchmark Suite</p>
    </div>
</body>
</html>
""")
        
        # Save HTML
        if save_path is None:
            save_path = self.output_dir / "report.html"
        
        with open(save_path, 'w') as f:
            f.write(''.join(html))
        
        return str(save_path)
