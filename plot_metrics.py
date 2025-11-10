"""
Utility script to plot training metrics from log files.

Usage:
    python plot_metrics.py runs/quick_test/20251106-223912
    python plot_metrics.py runs/quick_test/20251106-223912 --metrics loss residual_loss
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt


def load_metrics_history(log_dir: Path) -> Dict[str, List]:
    """Load metrics history from JSONL file.
    
    Args:
        log_dir: Directory containing metrics_history.jsonl
        
    Returns:
        Dictionary mapping metric names to lists of (step, value) tuples
    """
    metrics_file = log_dir / 'metrics_history.jsonl'
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    metrics_data = {}
    
    with open(metrics_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            name = entry['name']
            step = entry['step']
            value = entry['value']
            
            if name not in metrics_data:
                metrics_data[name] = {'steps': [], 'values': []}
            
            metrics_data[name]['steps'].append(step)
            metrics_data[name]['values'].append(value)
    
    return metrics_data


def plot_metrics(
    log_dir: Path,
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
):
    """Plot training metrics.
    
    Args:
        log_dir: Directory containing metrics files
        metrics_to_plot: List of metric names to plot (None = plot all)
        save_path: Path to save plot (None = display only)
    """
    metrics_data = load_metrics_history(log_dir)
    
    # Filter metrics if specified
    if metrics_to_plot is not None:
        # Support both full names and partial matches
        filtered_data = {}
        for name in metrics_data.keys():
            for pattern in metrics_to_plot:
                if pattern in name:
                    filtered_data[name] = metrics_data[name]
                    break
        metrics_data = filtered_data
    
    if not metrics_data:
        print("No metrics to plot!")
        return
    
    # Separate train and eval metrics
    train_metrics = {k: v for k, v in metrics_data.items() if k.startswith('train/')}
    eval_metrics = {k: v for k, v in metrics_data.items() if k.startswith('eval/')}
    
    # Create subplots
    n_train = len(train_metrics)
    n_eval = len(eval_metrics)
    n_plots = n_train + (1 if n_eval > 0 else 0)
    
    if n_plots == 0:
        print("No metrics to plot!")
        return
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    # Plot training metrics
    for idx, (name, data) in enumerate(train_metrics.items()):
        ax = axes[idx]
        ax.plot(data['steps'], data['values'], linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    # Plot evaluation metrics together
    if n_eval > 0:
        ax = axes[-1]
        for name, data in eval_metrics.items():
            ax.plot(data['steps'], data['values'], marker='o', label=name, linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title('Evaluation Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def print_summary(log_dir: Path):
    """Print metrics summary.
    
    Args:
        log_dir: Directory containing metrics_summary.json
    """
    summary_file = log_dir / 'metrics_summary.json'
    
    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}")
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    
    for name, stats in summary.items():
        print(f"\n{name}:")
        print(f"  Final: {stats['final']:.6f}")
        print(f"  Min:   {stats['min']:.6f}")
        print(f"  Max:   {stats['max']:.6f}")
        print(f"  Mean:  {stats['mean']:.6f}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from log files')
    
    parser.add_argument('log_dir', type=str,
                        help='Directory containing metrics files')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                        help='Specific metrics to plot (e.g., loss residual_loss)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save plot (default: display only)')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return
    
    # Print summary if requested
    if args.summary:
        print_summary(log_dir)
    
    # Plot metrics
    save_path = Path(args.save) if args.save else None
    plot_metrics(log_dir, args.metrics, save_path)


if __name__ == '__main__':
    main()

