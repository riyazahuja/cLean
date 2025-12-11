#!/usr/bin/env python3
"""
cLean GPU Benchmark Visualization

Generates matplotlib plots comparing:
- CPU vs cLean GPU vs CUDA baseline
- Timing breakdown (H2D, kernel, D2H, overhead)
- Scaling analysis across input sizes

Usage:
    python3 Benchmarks/visualize.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color scheme
COLORS = {
    'cpu': '#2ecc71',       # Green
    'clean_gpu': '#3498db', # Blue
    'cuda': '#e74c3c',      # Red
    'h2d': '#f39c12',       # Orange
    'd2h': '#9b59b6',       # Purple
    'kernel': '#1abc9c',    # Teal
    'json_ser': '#e67e22',  # Dark orange
    'spawn': '#95a5a6',     # Gray
    'json_parse': '#34495e' # Dark gray
}


def load_benchmark_data(csv_path='Benchmarks/results/benchmark_data.csv',
                        json_path='Benchmarks/results/benchmark_data.json'):
    """Load benchmark results from CSV or JSON."""
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        # Handle N/A values
        df['cuda_ms'] = pd.to_numeric(df['cuda_ms'], errors='coerce')
        df['gpu_util'] = pd.to_numeric(df['gpu_util'], errors='coerce')
        return df
    elif Path(json_path).exists():
        with open(json_path) as f:
            data = json.load(f)
        # Flatten JSON to DataFrame
        rows = []
        for suite in data:
            for result in suite['results']:
                row = {
                    'kernel': result['kernelName'],
                    'input_size': result['inputSize'],
                    'cpu_ms': result['cpuTimeMs'],
                    'gpu_total_ms': result['gpuTotalTimeMs'],
                    'gpu_kernel_ms': result['gpuKernelOnlyMs'],
                    'cuda_ms': result.get('cudaReferenceMs'),
                    'correct': result['correct']
                }
                if result.get('breakdown'):
                    bd = result['breakdown']
                    row.update({
                        'h2d_ms': bd.get('h2dTransferMs', 0),
                        'd2h_ms': bd.get('d2hTransferMs', 0),
                        'json_serialize_ms': bd.get('jsonSerializeMs', 0),
                        'process_spawn_ms': bd.get('processSpawnMs', 0),
                        'json_parse_ms': bd.get('jsonParseMs', 0)
                    })
                rows.append(row)
        return pd.DataFrame(rows)
    else:
        print(f"Error: No benchmark data found at {csv_path} or {json_path}")
        sys.exit(1)


def plot_speedup_comparison(df, output_path='Benchmarks/results/plots/speedup_comparison.png'):
    """Bar chart comparing speedup across kernels."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate max speedup per kernel
    speedup_data = df.groupby('kernel').apply(
        lambda g: g['speedup_vs_cpu'].max() if 'speedup_vs_cpu' in g.columns
        else (g['cpu_ms'] / g['gpu_total_ms']).max()
    ).sort_values(ascending=False)

    kernels = speedup_data.index.tolist()
    speedups = speedup_data.values

    x = np.arange(len(kernels))
    bars = ax.bar(x, speedups, color=COLORS['clean_gpu'], edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(f'{speedup:.1f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Breakeven')
    ax.set_ylabel('Max Speedup vs CPU')
    ax.set_title('cLean GPU Speedup by Kernel (Max across input sizes)')
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_timing_breakdown(df, kernel_name, output_path):
    """Stacked bar chart showing timing breakdown for a specific kernel."""
    fig, ax = plt.subplots(figsize=(10, 6))

    kernel_df = df[df['kernel'] == kernel_name].sort_values('input_size')

    if kernel_df.empty:
        print(f"  Warning: No data for kernel {kernel_name}")
        return

    sizes = kernel_df['input_size'].astype(str).values

    # Get timing components (with defaults)
    h2d = kernel_df.get('h2d_ms', pd.Series([0]*len(kernel_df))).fillna(0).values
    kernel_time = kernel_df.get('gpu_kernel_ms', pd.Series([0]*len(kernel_df))).fillna(0).values
    d2h = kernel_df.get('d2h_ms', pd.Series([0]*len(kernel_df))).fillna(0).values
    json_ser = kernel_df.get('json_serialize_ms', pd.Series([0]*len(kernel_df))).fillna(0).values
    spawn = kernel_df.get('process_spawn_ms', pd.Series([0]*len(kernel_df))).fillna(0).values
    json_parse = kernel_df.get('json_parse_ms', pd.Series([0]*len(kernel_df))).fillna(0).values

    x = np.arange(len(sizes))
    width = 0.6

    # Stack the bars
    bottom = np.zeros(len(sizes))

    ax.bar(x, h2d, width, label='H2D Transfer', color=COLORS['h2d'], bottom=bottom)
    bottom += h2d

    ax.bar(x, kernel_time, width, label='Kernel', color=COLORS['kernel'], bottom=bottom)
    bottom += kernel_time

    ax.bar(x, d2h, width, label='D2H Transfer', color=COLORS['d2h'], bottom=bottom)
    bottom += d2h

    ax.bar(x, json_ser, width, label='JSON Serialize', color=COLORS['json_ser'], bottom=bottom)
    bottom += json_ser

    ax.bar(x, spawn, width, label='Process Spawn', color=COLORS['spawn'], bottom=bottom)
    bottom += spawn

    ax.bar(x, json_parse, width, label='JSON Parse', color=COLORS['json_parse'], bottom=bottom)

    ax.set_xlabel('Input Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'{kernel_name} Timing Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scaling_analysis(df, output_path='Benchmarks/results/plots/scaling_analysis.png'):
    """Log-log plot showing scaling behavior of kernel time vs input size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for kernel in df['kernel'].unique():
        kernel_df = df[df['kernel'] == kernel].sort_values('input_size')
        if 'gpu_kernel_ms' in kernel_df.columns:
            ax.loglog(kernel_df['input_size'], kernel_df['gpu_kernel_ms'],
                      marker='o', label=kernel, linewidth=2, markersize=5)

    ax.set_xlabel('Input Size')
    ax.set_ylabel('Kernel Time (ms)')
    ax.set_title('Kernel Execution Time Scaling')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cpu_vs_gpu(df, output_path='Benchmarks/results/plots/cpu_vs_gpu.png'):
    """Line plot comparing CPU and GPU times across input sizes."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    kernels = df['kernel'].unique()

    for i, kernel in enumerate(kernels):
        if i >= len(axes):
            break
        ax = axes[i]
        kernel_df = df[df['kernel'] == kernel].sort_values('input_size')

        ax.plot(kernel_df['input_size'], kernel_df['cpu_ms'],
                marker='o', color=COLORS['cpu'], label='CPU', linewidth=2)
        ax.plot(kernel_df['input_size'], kernel_df['gpu_total_ms'],
                marker='s', color=COLORS['clean_gpu'], label='GPU Total', linewidth=2)
        ax.plot(kernel_df['input_size'], kernel_df['gpu_kernel_ms'],
                marker='^', color=COLORS['kernel'], label='GPU Kernel', linewidth=2, linestyle='--')

        ax.set_xlabel('Input Size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(kernel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(len(kernels), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('CPU vs GPU Execution Time by Kernel', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_overhead_analysis(df, output_path='Benchmarks/results/plots/overhead_analysis.png'):
    """Pie chart showing average time breakdown across all benchmarks."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate average across all benchmarks
    components = {
        'H2D Transfer': df.get('h2d_ms', pd.Series([0])).mean(),
        'Kernel Execution': df.get('gpu_kernel_ms', pd.Series([0])).mean(),
        'D2H Transfer': df.get('d2h_ms', pd.Series([0])).mean(),
        'JSON Serialize': df.get('json_serialize_ms', pd.Series([0])).mean(),
        'Process Spawn': df.get('process_spawn_ms', pd.Series([0])).mean(),
        'JSON Parse': df.get('json_parse_ms', pd.Series([0])).mean(),
    }

    # Filter out zero values
    components = {k: v for k, v in components.items() if v > 0}

    if not components:
        print("  Warning: No timing data available for overhead analysis")
        return

    labels = list(components.keys())
    sizes = list(components.values())
    colors = [COLORS['h2d'], COLORS['kernel'], COLORS['d2h'],
              COLORS['json_ser'], COLORS['spawn'], COLORS['json_parse']][:len(labels)]

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90)

    ax.set_title('Average Time Distribution Across All Benchmarks')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_correctness_summary(df, output_path='Benchmarks/results/plots/correctness_summary.png'):
    """Bar chart showing correctness results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    correctness = df.groupby('kernel')['correct'].agg(['sum', 'count'])
    correctness['pct'] = correctness['sum'] / correctness['count'] * 100

    x = np.arange(len(correctness))
    bars = ax.bar(x, correctness['pct'], color=COLORS['clean_gpu'], edgecolor='black')

    # Color bars based on correctness
    for bar, pct in zip(bars, correctness['pct']):
        if pct == 100:
            bar.set_color('#27ae60')  # Green
        elif pct > 0:
            bar.set_color('#f39c12')  # Orange
        else:
            bar.set_color('#e74c3c')  # Red

    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    ax.set_ylabel('Correctness (%)')
    ax.set_title('Benchmark Correctness by Kernel')
    ax.set_xticks(x)
    ax.set_xticklabels(correctness.index, rotation=45, ha='right')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_plots(data_path='Benchmarks/results/benchmark_data.csv'):
    """Generate all visualization plots."""
    print("Loading benchmark data...")
    df = load_benchmark_data(data_path)
    print(f"  Loaded {len(df)} benchmark results for {df['kernel'].nunique()} kernels")

    # Create output directory
    output_dir = Path('Benchmarks/results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")

    # Generate overview plots
    plot_speedup_comparison(df)
    plot_scaling_analysis(df)
    plot_cpu_vs_gpu(df)
    plot_overhead_analysis(df)
    plot_correctness_summary(df)

    # Generate timing breakdown for each kernel
    print("\nGenerating per-kernel timing breakdowns...")
    for kernel in df['kernel'].unique():
        safe_name = kernel.lower().replace(' ', '_')
        plot_timing_breakdown(df, kernel,
            f'Benchmarks/results/plots/timing_breakdown_{safe_name}.png')

    print("\n" + "="*50)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*50)


if __name__ == '__main__':
    generate_all_plots()
