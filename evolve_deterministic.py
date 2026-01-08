#!/usr/bin/env python3
"""
Deterministic Evolution Script
==============================

Evolves an input alignment by applying deterministic mutations in a rolling fashion.
Preserves population structure while adding temporal divergence.
"""

import argparse
import os
import sys
import numpy as np
from Bio import SeqIO, Phylo, AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from copy import deepcopy

# ==========================================
# 1. DETERMINISTIC MUTATION GENERATOR
# ==========================================

class DeterministicMutationGenerator:
    def __init__(self, genome_length):
        self.genome_length = genome_length
        self.counter = 0
        self.active_sites = set()
        
    def get_next_mutation(self, current_base_at_pos=None):
        """
        Returns (position, new_base) for the next mutation event.
        Cycles through genome positions and bases deterministically.
        """
        # Simple deterministic strategy:
        # Cycle through positions 0 to L-1
        # Cycle through bases A -> C -> G -> T (skipping current if known)
        
        pos = self.counter % self.genome_length
        self.counter += 1
        
        # Deterministic base selection based on position index
        # This ensures every time we hit this 'mutation index', we get the same result
        # regardless of which sequence it is applied to.
        
        bases = ['A', 'C', 'G', 'T']
        # We pick a base based on the mutation round (counter // length)
        # to ensure we don't just toggle back and forth active sites too quickly
        
        base_idx = (self.counter // self.genome_length) % 4
        new_base = bases[base_idx]
        
        # If we knew the current base, we could avoid identity, but for a 
        # global mutation stream, we just define the "Target State".
        # If a sequence already has this state, it technically stays same, 
        # but statistically this is rare for long genomes.
        
        return pos, new_base

# ==========================================
# 2. PLOTTING FUNCTIONS (Adapted from run_population_sim.py)
# ==========================================

def get_x_coordinates(tree):
    """Calculate x-coordinates (distance from root) for all nodes."""
    x_coords = {tree.root: 0}
    stack = [tree.root]
    while stack:
        parent = stack.pop()
        for child in parent.clades:
            x_coords[child] = x_coords[parent] + (child.branch_length or 0.0)
            stack.append(child)
    return x_coords

def get_y_coordinates(tree, dist=1.0):
    """
    Calculate y-coordinates for all nodes in the tree.
    Tips are assigned y-coordinates 0, 1, ...
    """
    y_coords = {}
    max_y = 0
    for clade in tree.get_terminals():
        y_coords[clade] = max_y
        max_y += dist
    for clade in tree.get_nonterminals(order='postorder'):
        y_coords[clade] = np.mean([y_coords[c] for c in clade.clades])
    return y_coords

def draw_tree_custom(ax, tree, title, color_attr='custom_color'):
    """Custom tree drawing function."""
    x_coords = get_x_coordinates(tree)
    y_coords = get_y_coordinates(tree)
    
    # Draw branches
    for clade in tree.find_clades(order='level'):
        if clade.clades:
            x_root = x_coords[clade]
            y_root = y_coords[clade]
            for child in clade.clades:
                x_child = x_coords[child]
                y_child = y_coords[child]
                
                # Color logic
                color = 'black'
                if hasattr(child, color_attr) and getattr(child, color_attr):
                     color = getattr(child, color_attr)
                
                # Horizontal line
                ax.plot([x_root, x_child], [y_child, y_child], color=color, lw=1.5)
                # Vertical line
                ax.plot([x_root, x_root], [y_root, y_child], color=color, lw=1.5)
    
    # Draw tips
    for clade in tree.get_terminals():
        x = x_coords[clade]
        y = y_coords[clade]
        color = 'black'
        if hasattr(clade, color_attr) and getattr(clade, color_attr):
            color = getattr(clade, color_attr)
        ax.plot(x, y, 'o', color=color, markersize=4)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Divergence")
    ax.set_ylabel("")
    ax.set_yticks([])
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

# ==========================================
# 3. MAIN SIMULATION
# ==========================================

def run_deterministic_evolution(
    input_fasta: str,
    output_dir: str,
    sim_days: int,
    mutations_per_week: float
):
    print("=" * 60)
    print("DETERMINISTIC EVOLUTION SIMULATION")
    print("=" * 60)
    
    # 1. Load Sequences
    print(f"Loading sequences from {input_fasta}...")
    original_records = list(SeqIO.parse(input_fasta, "fasta"))
    num_sequences = len(original_records)
    genome_length = len(original_records[0].seq)
    
    print(f"  Sequences: {num_sequences}")
    print(f"  Genome Length: {genome_length}")
    print(f"  Simulation Days: {sim_days}")
    print(f"  Rate: {mutations_per_week} mutations/week (per genome)")
    
    # Convert to mutable lists for efficient modification
    # Structure: [ ['A', 'C', ...], ['G', 'T', ...] ]
    current_sequences = [list(str(rec.seq)) for rec in original_records]
    ids = [rec.id for rec in original_records]
    
    # 2. Setup Loop
    os.makedirs(output_dir, exist_ok=True)
    daily_seq_dir = os.path.join(output_dir, "daily_sequences")
    os.makedirs(daily_seq_dir, exist_ok=True)
    
    # Rolling Logic
    # Calculates how many sequences "advance" to the next mutation state per day
    updates_per_day = (mutations_per_week / 7.0) * num_sequences
    print(f"  Updates per day: {updates_per_day:.4f} sequences")
    
    # Generator
    mut_gen = DeterministicMutationGenerator(genome_length)
    
    # State tracking
    # We maintain a 'global mutation list'. 
    # Each sequence index `i` is at `current_mutation_stage[i]`.
    # But since we roll 1-N, we can just simplify:
    # A single pointer `next_seq_to_update` cycles 0..N-1.
    # When it wraps around, we generate a NEW mutation for the global pool.
    
    next_seq_to_update = 0
    accumulator = 0.0
    
    # We need to remember "Active Mutation K" until it has been applied to everyone.
    # Actually, simpler: 
    # We have a "Current Active Mutation" (pos, base).
    # We apply it to seq 0, then seq 1, ..., seq N.
    # Once applied to seq N, we behave as if the "wave" is done, and generate next mutation?
    # YES. This ensures Seq 0 is always ahead or equal to Seq N.
    
    # However, if rate is HIGH (e.g. 2 waves per day), we might need multiple active mutations.
    # The 'next_seq_to_update' logic handles this naturally.
    # If we update 200 sequences (and N=100), we wrap around twice.
    # Wrap 1: Everyone gets Mut A.
    # Wrap 2: Everyone gets Mut B.
    
    # To support this, we need a list of applied mutations PER SEQUENCE? 
    # No, we just need to know: When seq `i` is updated, WHICH mutation does it get?
    # It gets the next one it hasn't seen.
    # Let's map `seq_index -> mutation_count`.
    # Global `generated_mutations` list.
    # If `seq[i].count == len(generated_mutations)`, we generate a new one.
    
    seq_mutation_counts = [0] * num_sequences
    generated_mutations = [] # List of (pos, new_base)
    
    # Save Day 0
    save_daily_fasta(ids, current_sequences, daily_seq_dir, 0)
    
    for day in range(1, sim_days + 1):
        accumulator += updates_per_day
        num_updates = int(accumulator)
        accumulator -= num_updates
        
        if num_updates > 0:
            for _ in range(num_updates):
                target_idx = next_seq_to_update
                
                # Check which mutation this sequence needs
                needed_mut_idx = seq_mutation_counts[target_idx]
                
                # Generate if not exists
                if needed_mut_idx >= len(generated_mutations):
                    generated_mutations.append(mut_gen.get_next_mutation())
                
                # Apply mutation
                pos, new_base = generated_mutations[needed_mut_idx]
                current_sequences[target_idx][pos] = new_base
                
                # Update counters
                seq_mutation_counts[target_idx] += 1
                next_seq_to_update = (next_seq_to_update + 1) % num_sequences
        
        # Save daily
        save_daily_fasta(ids, current_sequences, daily_seq_dir, day)
        if day % 10 == 0:
            print(f"Day {day}: Completed. (Total mutations generated: {len(generated_mutations)})")

    # 3. Final Analysis & Plotting
    print("\nGenerating final phylogeny...")
    final_tree_path = os.path.join(output_dir, "final_tree_plot.png")
    plot_final_tree(ids, current_sequences, final_tree_path)
    print(f"Done. Output in {output_dir}")

def save_daily_fasta(ids, sequences, output_dir, day):
    filename = os.path.join(output_dir, f"day_{day}.fasta")
    with open(filename, "w") as f:
        for id_str, seq_list in zip(ids, sequences):
            f.write(f">{id_str}\n{''.join(seq_list)}\n")

def plot_final_tree(ids, sequences, output_path):
    # Create SeqRecords
    records = [SeqRecord(Seq("".join(s)), id=i) for i, s in zip(ids, sequences)]
    
    # Build Tree (UPGMA or NJ)
    # Using simple DistanceCalculator
    print("  Calculating distance matrix...")
    calculator = DistanceCalculator('identity')
    constructor = DistanceTreeConstructor(calculator, 'nj')
    tree = constructor.build_tree(AlignIO.MultipleSeqAlignment(records))
    
    # Root at midpoint to better visualize divergence
    try:
        tree.root_at_midpoint()
    except Exception as e:
        print(f"  Warning: Could not root at midpoint (likely zero divergence or star tree). Keeping default root. Error: {e}")
    
    # Plot
    print("  Drawing tree...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color tips
    # We can color by index to show the 'rolling' gradient
    # Or just black
    
    # Create a simple attribute for coloring
    norm = mcolors.Normalize(vmin=0, vmax=len(ids))
    cmap = cm.viridis
    
    id_map = {id_str: i for i, id_str in enumerate(ids)}
    
    for clade in tree.get_terminals():
        idx = id_map.get(clade.name, 0)
        clade.custom_color = mcolors.to_hex(cmap(norm(idx)))
        
    draw_tree_custom(ax, tree, "Final Population Structure (Colored by Cycle Order)", color_attr='custom_color')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Evolution")
    parser.add_argument("--input_fasta", required=True, help="Input alignment")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--days", type=int, default=100, help="Simulation days")
    parser.add_argument("--rate", type=float, default=1.0, help="Mutations per week (per genome)")
    
    args = parser.parse_args()
    
    run_deterministic_evolution(
        args.input_fasta, 
        args.output_dir, 
        args.days, 
        args.rate
    )
