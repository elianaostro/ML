# src/utils.py
import sys
from typing import Dict, Any

def update_progress_bar(current, total, bar_length=50, metrics=None):
    """
    Display a progress bar with optional metrics.
    
    Args:
        current: Current progress value
        total: Total value
        bar_length: Length of the progress bar
        metrics: Dictionary of metrics to display
    """
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    metrics_str = ""
    if metrics:
        metrics_str = " - " + " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    sys.stdout.write(f"\rEpoch: [{arrow + spaces}] {int(percent * 100)}%{metrics_str}")
    sys.stdout.flush()