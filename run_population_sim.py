#!/usr/bin/env python3
"""
Population Genetics Simulation Script
=====================================

Simulates multiple evolving lineages using a Wright-Fisher model.
Outputs daily FASTA files with sequences from all individuals in the population.

This script uses PhylogenyTracker from simulation_lib.py to track genetic evolution
over time without the hospital outbreak framework.

Usage:
    python run_population_sim.py --num_lineages 3 --pop_size 100 --mutation_rate 0.0001 \
                                   --burn_in 30 --sim_days 90 --genome_length 29903 \
                                   --output_dir ./population_sim_output
"""

import argparse
import os
import sys
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd

from simulation_lib import PhylogenyTracker


def run_population_simulation(
    num_lineages: int,
    pop_size: int,
    mutation_rate: float,
    burn_in_days: int,
    simulation_days: int,
    genome_length: int,
    output_dir: str,
    reference_path: str = None,
    transition_prob: float = 0.7,
    seed: int = 42
):
    """
    Run a population genetics simulation with multiple lineages.
    
    Args:
        num_lineages: Number of distinct lineages/variants to simulate
        pop_size: Population size per lineage (constant via Wright-Fisher)
        mutation_rate: Mutation rate per site per day
        burn_in_days: Days of evolution before Day 0 to establish diversity
        simulation_days: Number of days to simulate forward from Day 0
        genome_length: Length of genome in bases
        output_dir: Directory for output files
        reference_path: Optional path to reference genome FASTA
        transition_prob: Probability of transition vs transversion mutations
        seed: Random seed for reproducibility
    """
    
    print("=" * 80)
    print("POPULATION GENETICS SIMULATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Number of lineages: {num_lineages}")
    print(f"  Population size per lineage: {pop_size}")
    print(f"  Mutation rate: {mutation_rate} per site per day")
    print(f"  Burn-in period: {burn_in_days} days")
    print(f"  Simulation period: {simulation_days} days")
    print(f"  Genome length: {genome_length} bp")
    print(f"  Output directory: {output_dir}")
    print(f"  Random seed: {seed}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "num_lineages": num_lineages,
        "pop_size": pop_size,
        "mutation_rate": mutation_rate,
        "burn_in_days": burn_in_days,
        "simulation_days": simulation_days,
        "genome_length": genome_length,
        "reference_path": reference_path,
        "transition_prob": transition_prob,
        "seed": seed,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Set random seed
    np.random.seed(seed)
    
    # Initialize PhylogenyTracker
    # community_diversity_level is set to 0 since we control diversity via burn_in
    print("\nInitializing phylogeny tracker and evolving lineages...")
    phylogeny = PhylogenyTracker(
        genome_length=genome_length,
        mutation_rate=mutation_rate,
        community_diversity_level=0.0,  # We use burn_in instead
        num_community_lineages=num_lineages,
        community_pop_size=pop_size,
        burn_in_days=burn_in_days,
        reference_path=reference_path,
        transition_prob=transition_prob
    )
    
    print(f"✓ Burn-in complete. Starting forward simulation...")
    
    # Create daily sequences directory
    daily_dir = os.path.join(output_dir, "daily_sequences")
    os.makedirs(daily_dir, exist_ok=True)
    
    # Track all nodes for each day
    daily_census = []
    
    # Simulate forward for simulation_days
    for day in range(simulation_days):
        if day % 10 == 0:
            print(f"  Day {day}/{simulation_days}")
        
        # Step the community forward (Wright-Fisher)
        phylogeny.step_community(day)
        
        # Record all current individuals from all lineages
        for lineage_idx, lineage in enumerate(phylogeny.community_lineages):
            for individual_idx, node_id in enumerate(lineage):
                daily_census.append({
                    'day': day,
                    'lineage': lineage_idx,
                    'individual_id': f"L{lineage_idx}_I{individual_idx}",
                    'node_id': node_id
                })
    
    print(f"✓ Forward simulation complete.")
    
    # Finalize the tree sequence
    print("\nFinalizing tree sequence...")
    max_time = simulation_days
    ts = phylogeny.finalize_tree(max_time)
    
    print(f"✓ Tree sequence finalized:")
    print(f"    Total nodes: {ts.num_nodes}")
    print(f"    Total samples: {ts.num_samples}")
    print(f"    Total mutations: {ts.num_mutations}")
    print(f"    Total sites: {ts.num_sites}")
    
    # Save tree sequence
    ts_path = os.path.join(output_dir, "tree_sequence.trees")
    ts.dump(ts_path)
    print(f"✓ Tree sequence saved to {ts_path}")
    
    # Generate daily FASTA files
    print("\nGenerating daily FASTA files...")
    save_daily_fastas_population(ts, daily_census, output_dir, simulation_days)
    print(f"✓ Daily FASTA files saved to {daily_dir}")
    
    # Calculate and save diversity statistics
    print("\nCalculating diversity statistics...")
    stats = calculate_diversity_stats(ts, daily_census, simulation_days, num_lineages)
    
    stats_path = os.path.join(output_dir, "diversity_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Diversity statistics saved to {stats_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_diversity_over_time(stats, output_dir, simulation_days)
    print(f"✓ Plots saved to {output_dir}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  - config.json: Simulation parameters")
    print(f"  - tree_sequence.trees: Full phylogenetic tree (tskit format)")
    print(f"  - daily_sequences/: FASTA files for each day (day_0.fasta to day_{simulation_days-1}.fasta)")
    print(f"  - diversity_stats.json: Diversity metrics over time")
    print(f"  - pairwise_distance_over_time.png: Mean/median pairwise distances by day")
    print(f"  - pairwise_distance_by_week.png: Mean/median pairwise distances by week")
    print(f"  - diversity_breakdown.png: Within vs between lineage diversity")
    print(f"  - segregating_sites.png: Number of segregating sites over time")
    print("=" * 80)


def save_daily_fastas_population(ts, daily_census, output_dir, max_days):
    """
    Save daily FASTA files containing sequences of all individuals on each day.
    
    Args:
        ts: TreeSequence from tskit
        daily_census: List of dicts with {day, lineage, individual_id, node_id}
        output_dir: Output directory
        max_days: Number of days simulated
    """
    import collections
    
    daily_dir = os.path.join(output_dir, "daily_sequences")
    os.makedirs(daily_dir, exist_ok=True)
    
    # Group census by day
    census_by_day = collections.defaultdict(list)
    for record in daily_census:
        census_by_day[record['day']].append(record)
    
    # Build node_id -> sequence mapping
    node_to_seq = {}
    samples_list = ts.samples()
    
    for i, seq in enumerate(ts.haplotypes()):
        node_id = samples_list[i]
        node_to_seq[node_id] = seq
    
    # Write daily FASTA files
    for day in range(max_days):
        filename = os.path.join(daily_dir, f"day_{day}.fasta")
        with open(filename, "w") as f:
            if day in census_by_day:
                for record in census_by_day[day]:
                    node_id = record['node_id']
                    if node_id in node_to_seq:
                        seq = node_to_seq[node_id]
                        # Header format: >Lineage_Individual_Day
                        header = f">{record['individual_id']}_Day{day}"
                        f.write(f"{header}\n{seq}\n")


def calculate_diversity_stats(ts, daily_census, simulation_days, num_lineages):
    """
    Calculate diversity statistics over time.
    
    Returns dict with:
        - total_diversity: Average pairwise differences across all samples per day
        - within_lineage_diversity: Average pairwise differences within each lineage
        - between_lineage_diversity: Average pairwise differences between lineages
    """
    import collections
    
    # Group census by day
    census_by_day = collections.defaultdict(list)
    for record in daily_census:
        census_by_day[record['day']].append(record)
    
    # Build node_id -> sequence mapping
    node_to_seq = {}
    samples_list = ts.samples()
    
    for i, seq in enumerate(ts.haplotypes()):
        node_id = samples_list[i]
        node_to_seq[node_id] = seq
    
    stats = {
        "days": [],
        "total_diversity": [],
        "within_lineage_diversity": [],
        "between_lineage_diversity": [],
        "num_segregating_sites": [],
        "mean_pairwise_distance": [],
        "median_pairwise_distance": [],
        "all_pairwise_distances": []  # Store all distances for final analysis
    }
    
    # Sample every 10 days to avoid computational overload
    sample_days = list(range(0, simulation_days, max(1, simulation_days // 20)))
    
    for day in sample_days:
        if day not in census_by_day:
            continue
        
        records = census_by_day[day]
        
        # Group by lineage
        lineage_nodes = collections.defaultdict(list)
        all_nodes = []
        
        for record in records:
            node_id = record['node_id']
            if node_id in node_to_seq:
                lineage_nodes[record['lineage']].append(node_id)
                all_nodes.append(node_id)
        
        if len(all_nodes) < 2:
            continue
        
        # Calculate total diversity (returns mean, median, all_distances)
        total_div_mean, total_div_median, all_distances = calculate_pairwise_distances(all_nodes, node_to_seq)
        
        # Calculate within-lineage diversity
        within_div_list = []
        for lineage_idx in range(num_lineages):
            if lineage_idx in lineage_nodes and len(lineage_nodes[lineage_idx]) > 1:
                div_mean, _, _ = calculate_pairwise_distances(lineage_nodes[lineage_idx], node_to_seq)
                within_div_list.append(div_mean)
        
        within_div = np.mean(within_div_list) if within_div_list else 0.0
        
        # Calculate between-lineage diversity (sample pairs from different lineages)
        between_div_list = []
        lineage_indices = list(lineage_nodes.keys())
        
        if len(lineage_indices) > 1:
            for i in range(len(lineage_indices)):
                for j in range(i + 1, len(lineage_indices)):
                    lin_i = lineage_indices[i]
                    lin_j = lineage_indices[j]
                    
                    if lineage_nodes[lin_i] and lineage_nodes[lin_j]:
                        # Sample up to 10 pairs to avoid combinatorial explosion
                        n_pairs = min(10, len(lineage_nodes[lin_i]) * len(lineage_nodes[lin_j]))
                        
                        for _ in range(n_pairs):
                            node_a = np.random.choice(lineage_nodes[lin_i])
                            node_b = np.random.choice(lineage_nodes[lin_j])
                            dist = hamming_distance(node_to_seq[node_a], node_to_seq[node_b])
                            between_div_list.append(dist)
        
        between_div = np.mean(between_div_list) if between_div_list else 0.0
        
        # Count segregating sites
        seqs = [node_to_seq[n] for n in all_nodes]
        seg_sites = count_segregating_sites(seqs)
        
        stats["days"].append(day)
        stats["total_diversity"].append(total_div_mean)
        stats["within_lineage_diversity"].append(within_div)
        stats["between_lineage_diversity"].append(between_div)
        stats["num_segregating_sites"].append(seg_sites)
        stats["mean_pairwise_distance"].append(total_div_mean)
        stats["median_pairwise_distance"].append(total_div_median)
        stats["all_pairwise_distances"].append(all_distances)
    
    return stats


def calculate_pairwise_distances(node_list, node_to_seq):
    """
    Calculate pairwise Hamming distances for a list of nodes.
    
    Returns:
        mean_distance: Mean pairwise distance
        median_distance: Median pairwise distance
        all_distances: List of all pairwise distances
    """
    if len(node_list) < 2:
        return 0.0, 0.0, []
    
    distances = []
    n = len(node_list)
    
    # Sample up to 1000 pairs to avoid O(n^2) explosion
    max_pairs = min(1000, n * (n - 1) // 2)
    
    if n * (n - 1) // 2 <= max_pairs:
        # Calculate all pairs
        for i in range(n):
            for j in range(i + 1, n):
                seq_a = node_to_seq[node_list[i]]
                seq_b = node_to_seq[node_list[j]]
                dist = hamming_distance(seq_a, seq_b)
                distances.append(dist)
    else:
        # Randomly sample pairs
        for _ in range(max_pairs):
            i, j = np.random.choice(n, size=2, replace=False)
            seq_a = node_to_seq[node_list[i]]
            seq_b = node_to_seq[node_list[j]]
            dist = hamming_distance(seq_a, seq_b)
            distances.append(dist)
    
    if distances:
        return np.mean(distances), np.median(distances), distances
    else:
        return 0.0, 0.0, []


def hamming_distance(seq_a, seq_b):
    """Calculate Hamming distance between two sequences."""
    return sum(a != b for a, b in zip(seq_a, seq_b))


def plot_diversity_over_time(stats, output_dir, simulation_days):
    """
    Generate plots showing diversity metrics over time.
    
    Creates:
        1. Pairwise distance over time (mean and median by day)
        2. Pairwise distance by week (mean and median)
        3. Diversity breakdown (within vs between lineage)
    """
    if not stats["days"]:
        print("  Warning: No statistics to plot")
        return
    
    df = pd.DataFrame({
        'day': stats['days'],
        'mean_distance': stats['mean_pairwise_distance'],
        'median_distance': stats['median_pairwise_distance'],
        'within_lineage': stats['within_lineage_diversity'],
        'between_lineage': stats['between_lineage_diversity'],
        'segregating_sites': stats['num_segregating_sites']
    })
    
    # --- PLOT 1: Pairwise Distance Over Time (Daily) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['day'], df['mean_distance'], 'o-', label='Mean Pairwise Distance', 
            color='steelblue', linewidth=2, markersize=5, alpha=0.7)
    ax.plot(df['day'], df['median_distance'], 's-', label='Median Pairwise Distance', 
            color='coral', linewidth=2, markersize=5, alpha=0.7)
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Pairwise Distance (SNPs)', fontsize=12)
    ax.set_title('Pairwise Genetic Distance Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairwise_distance_over_time.png'), dpi=300)
    plt.close()
    
    # --- PLOT 2: Pairwise Distance By Week ---
    # Group by week
    df['week'] = df['day'] // 7
    weekly_stats = df.groupby('week').agg({
        'mean_distance': 'mean',
        'median_distance': 'mean',  # Average of medians per week
        'day': 'max'  # Last day of week
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.35
    x_pos = np.arange(len(weekly_stats))
    
    bars1 = ax.bar(x_pos - bar_width/2, weekly_stats['mean_distance'], bar_width,
                   label='Mean Pairwise Distance', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + bar_width/2, weekly_stats['median_distance'], bar_width,
                   label='Median Pairwise Distance', color='coral', alpha=0.8)
    
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Pairwise Distance (SNPs)', fontsize=12)
    ax.set_title('Mean and Median Pairwise Distance by Week', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Week {int(w)}' for w in weekly_stats['week']], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairwise_distance_by_week.png'), dpi=300)
    plt.close()
    
    # --- PLOT 3: Diversity Breakdown (Within vs Between Lineage) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['day'], df['mean_distance'], 'o-', label='Total Diversity', 
            color='black', linewidth=2.5, markersize=6, alpha=0.8)
    ax.plot(df['day'], df['within_lineage'], 's-', label='Within-Lineage Diversity', 
            color='green', linewidth=2, markersize=5, alpha=0.7)
    ax.plot(df['day'], df['between_lineage'], '^-', label='Between-Lineage Diversity', 
            color='red', linewidth=2, markersize=5, alpha=0.7)
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Pairwise Distance (SNPs)', fontsize=12)
    ax.set_title('Diversity Breakdown: Within vs Between Lineages', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diversity_breakdown.png'), dpi=300)
    plt.close()
    
    # --- PLOT 4: Segregating Sites Over Time ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['day'], df['segregating_sites'], 'o-', label='Segregating Sites', 
            color='purple', linewidth=2, markersize=5, alpha=0.7)
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Number of Segregating Sites', fontsize=12)
    ax.set_title('Segregating Sites Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segregating_sites.png'), dpi=300)
    plt.close()


def count_segregating_sites(sequences):
    """Count the number of segregating (polymorphic) sites in a set of sequences."""
    if not sequences:
        return 0
    
    seq_length = len(sequences[0])
    seg_sites = 0
    
    for pos in range(seq_length):
        bases_at_pos = set(seq[pos] for seq in sequences)
        if len(bases_at_pos) > 1:
            seg_sites += 1
    
    return seg_sites


def main():
    parser = argparse.ArgumentParser(
        description="Population genetics simulation with multiple lineages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core simulation parameters
    parser.add_argument("--num_lineages", type=int, default=3,
                        help="Number of distinct lineages to simulate")
    parser.add_argument("--pop_size", type=int, default=100,
                        help="Population size per lineage (constant)")
    parser.add_argument("--mutation_rate", type=float, default=1e-4,
                        help="Mutation rate per site per day")
    
    # Time parameters
    parser.add_argument("--burn_in", type=int, default=30,
                        help="Days of burn-in evolution before Day 0")
    parser.add_argument("--sim_days", type=int, default=90,
                        help="Number of days to simulate forward")
    
    # Genome parameters
    parser.add_argument("--genome_length", type=int, default=29903,
                        help="Genome length in base pairs (default: SARS-CoV-2)")
    parser.add_argument("--reference", type=str, default=None,
                        help="Path to reference genome FASTA (optional)")
    parser.add_argument("--transition_prob", type=float, default=0.7,
                        help="Probability of transition vs transversion mutation")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./population_sim_output",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_lineages < 1:
        print("ERROR: num_lineages must be >= 1")
        sys.exit(1)
    
    if args.pop_size < 1:
        print("ERROR: pop_size must be >= 1")
        sys.exit(1)
    
    if args.mutation_rate < 0:
        print("ERROR: mutation_rate must be non-negative")
        sys.exit(1)
    
    # Run simulation
    run_population_simulation(
        num_lineages=args.num_lineages,
        pop_size=args.pop_size,
        mutation_rate=args.mutation_rate,
        burn_in_days=args.burn_in,
        simulation_days=args.sim_days,
        genome_length=args.genome_length,
        output_dir=args.output_dir,
        reference_path=args.reference,
        transition_prob=args.transition_prob,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
