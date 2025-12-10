#!/usr/bin/env python3
"""
Compare Stochastic vs Deterministic Simulation Modes
====================================================

Quick script to compare diversity dynamics between the two modes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_stats(output_dir):
    """Load diversity statistics from a simulation output directory."""
    stats_file = os.path.join(output_dir, "diversity_stats.json")
    if not os.path.exists(stats_file):
        return None
    with open(stats_file, 'r') as f:
        return json.load(f)

def plot_comparison(stochastic_dir, deterministic_dir, output_file="mode_comparison.png"):
    """Create comparison plot of the two modes."""
    
    stoch_stats = load_stats(stochastic_dir)
    det_stats = load_stats(deterministic_dir)
    
    if stoch_stats is None or det_stats is None:
        print("ERROR: Could not load stats from one or both directories")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean pairwise distance over time
    ax = axes[0, 0]
    ax.plot(stoch_stats['days'], stoch_stats['mean_pairwise_distance'], 
            'o-', label='Stochastic', alpha=0.7, linewidth=2)
    ax.plot(det_stats['days'], det_stats['mean_pairwise_distance'], 
            's-', label='Deterministic', alpha=0.7, linewidth=2)
    ax.set_xlabel('Day')
    ax.set_ylabel('Mean Pairwise Distance')
    ax.set_title('Mean Pairwise Distance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Segregating sites
    ax = axes[0, 1]
    ax.plot(stoch_stats['days'], stoch_stats['num_segregating_sites'], 
            'o-', label='Stochastic', alpha=0.7, linewidth=2)
    ax.plot(det_stats['days'], det_stats['num_segregating_sites'], 
            's-', label='Deterministic', alpha=0.7, linewidth=2)
    ax.set_xlabel('Day')
    ax.set_ylabel('Number of Segregating Sites')
    ax.set_title('Segregating Sites Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Diversity variability (coefficient of variation)
    ax = axes[1, 0]
    
    # Calculate rolling statistics
    stoch_mean = np.mean(stoch_stats['mean_pairwise_distance'])
    stoch_std = np.std(stoch_stats['mean_pairwise_distance'])
    stoch_cv = stoch_std / stoch_mean if stoch_mean > 0 else 0
    
    det_mean = np.mean(det_stats['mean_pairwise_distance'])
    det_std = np.std(det_stats['mean_pairwise_distance'])
    det_cv = det_std / det_mean if det_mean > 0 else 0
    
    bars = ax.bar(['Stochastic', 'Deterministic'], 
                   [stoch_cv, det_cv], 
                   color=['steelblue', 'coral'])
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Diversity Stability (lower = more stable)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Stochastic', 'Deterministic'],
        ['', '', ''],
        ['Mean diversity', f'{stoch_mean:.1f}', f'{det_mean:.1f}'],
        ['Std diversity', f'{stoch_std:.1f}', f'{det_std:.1f}'],
        ['Min diversity', f'{min(stoch_stats["mean_pairwise_distance"]):.1f}', 
         f'{min(det_stats["mean_pairwise_distance"]):.1f}'],
        ['Max diversity', f'{max(stoch_stats["mean_pairwise_distance"]):.1f}', 
         f'{max(det_stats["mean_pairwise_distance"]):.1f}'],
        ['', '', ''],
        ['Days simulated', f'{len(stoch_stats["days"])}', f'{len(det_stats["days"])}'],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.325, 0.325])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Stochastic vs Deterministic Evolution Models', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {output_file}")
    plt.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nStochastic Mode:")
    print(f"  Mean diversity: {stoch_mean:.1f} ± {stoch_std:.1f}")
    print(f"  Range: {min(stoch_stats['mean_pairwise_distance']):.1f} - {max(stoch_stats['mean_pairwise_distance']):.1f}")
    print(f"  Coefficient of variation: {stoch_cv:.3f}")
    
    print(f"\nDeterministic Mode:")
    print(f"  Mean diversity: {det_mean:.1f} ± {det_std:.1f}")
    print(f"  Range: {min(det_stats['mean_pairwise_distance']):.1f} - {max(det_stats['mean_pairwise_distance']):.1f}")
    print(f"  Coefficient of variation: {det_cv:.3f}")
    
    print(f"\nImprovement in stability: {(1 - det_cv/stoch_cv)*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_modes.py <stochastic_dir> <deterministic_dir>")
        print("\nExample:")
        print("  python compare_modes.py test_stochastic test_deterministic_v2")
        sys.exit(1)
    
    stoch_dir = sys.argv[1]
    det_dir = sys.argv[2]
    
    plot_comparison(stoch_dir, det_dir)
