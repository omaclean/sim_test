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
import random
import numpy as np
from collections import defaultdict
from Bio import SeqIO, Phylo, AlignIO
from Bio.Seq import Seq
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, DistanceMatrix, ParsimonyScorer, NNITreeSearcher, ParsimonyTreeConstructor
from Bio.Align import MultipleSeqAlignment
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from copy import deepcopy

# ==========================================
# 1. DETERMINISTIC MUTATION GENERATOR
# ==========================================

class DeterministicMutationGenerator:
    def __init__(self, sequences):
        self.sequences = sequences
        self.num_sequences = len(sequences)
        self.genome_length = len(sequences[0])
        self.valid_sites = self._find_valid_sites()
        
        # Shuffle sites to ensure random but fixed order (clock-like)
        print(f"  Shuffling {len(self.valid_sites)} sites for fixed mutation order...")
        random.shuffle(self.valid_sites)
        self.site_pointer = 0
        
        # Track current state of sites to avoid identity mutations on second pass
        self.site_states = {pos: base for pos, base in self.valid_sites}
        
    def _find_valid_sites(self):
        """
        Identify sites that are:
        1. Conserved (all sequences have the same base)
        2. Not 'N' (and not gap '-')
        Returns list of (pos, base)
        """
        valid = []
        possible_bases = set(['A', 'C', 'G', 'T'])
        
        print("  Scanning for valid conserved sites (non-N)...")
        # Optimization: Don't transpose whole matrix if large.
        # Just iterate positions.
        
        for pos in range(self.genome_length):
            # Check first seq base
            ref_base = self.sequences[0][pos]
            
            # Fast fail if not standard base
            if ref_base not in possible_bases:
                continue
            
            # Check conservation
            is_conserved = True
            for i in range(1, self.num_sequences):
                if self.sequences[i][pos] != ref_base:
                    is_conserved = False
                    break
            
            if is_conserved:
                valid.append((pos, ref_base))
                
        print(f"  Found {len(valid)} valid conserved sites out of {self.genome_length} ({len(valid)/self.genome_length*100:.1f}%)")
        if len(valid) == 0:
             raise ValueError("No conserved non-N sites found in alignment!")
        return valid

    def get_next_mutation(self, current_base_at_pos=None):
        """
        Returns (position, new_base) for the next mutation event.
        Iterates through shuffled valid sites in order.
        """
        # Get next site in fixed order
        pos, _ = self.valid_sites[self.site_pointer]
        
        # Increment pointer (loop if saturated)
        self.site_pointer = (self.site_pointer + 1) % len(self.valid_sites)
        
        # Get current state of this site to ensure change
        current_base = self.site_states[pos]
        
        # Pick new base != current
        options = ['A', 'C', 'G', 'T']
        if current_base in options:
            options.remove(current_base)
            
        new_base = random.choice(options)
        
        # Update state tracker
        self.site_states[pos] = new_base
        
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
    mutation_rate_site_day: float,
    downsample: int = -1,
    tree_samples: int = 200
):
    print("=" * 60)
    print("DETERMINISTIC EVOLUTION SIMULATION")
    print("=" * 60)
    
    # 1. Load Sequences
    print(f"Loading sequences from {input_fasta}...")
    original_records = list(SeqIO.parse(input_fasta, "fasta"))
    
    if not original_records:
         print("Error: No sequences found in input file.")
         sys.exit(1)
         
    num_sequences = len(original_records)
    
    # Calculate genome length (use max length to handle potential incomplete/gappy inputs?)
    # or just use first one. Ideally they are aligned.
    genome_length = len(original_records[0].seq)

    # Check for unusually short genome
    if genome_length < 20000:
        print(f"  Warning: Detected genome length is {genome_length}. This seems short for SARS-CoV-2 (~29903).")
        print("  Checking other sequences...")
        lengths = [len(r.seq) for r in original_records]
        max_len = max(lengths)
        if max_len > genome_length:
             print(f"  Found longer sequence with {max_len} bp. Using that as genome length reference.")
             genome_length = max_len
             # We might need to pad/align if they are different lengths?
             # For now, just assume aligned but maybe first one was partial?
    
    if downsample != -1 and len(original_records) > downsample:
        print(f"  Downsampling from {len(original_records)} to {downsample} sequences.")
        original_records = random.sample(original_records, downsample)
        
    # Re-calculate num_sequences after downsample
    num_sequences = len(original_records)
    
    # Convert rate: site/day -> week/genome
    mutations_per_week = mutation_rate_site_day * genome_length * 7.0
    
    print(f"  Sequences: {num_sequences}")
    print(f"  Genome Length: {genome_length}")
    print(f"  Simulation Days: {sim_days}")
    print(f"  Input Rate: {mutation_rate_site_day} mutations/site/day")
    print(f"  Calc Rate: {mutations_per_week:.4f} mutations/week (per genome)")
    
    # Convert to mutable lists for efficient modification
    # Structure: [ ['A', 'C', ...], ['G', 'T', ...] ]
    current_sequences = []
    ids = []
    
    for rec in original_records:
        seq_str = str(rec.seq)
        if len(seq_str) < genome_length:
             # Pad with N or -
             seq_str = seq_str.ljust(genome_length, 'N')
        elif len(seq_str) > genome_length:
             # This shouldn't happen if we set genome_length to max, 
             # but if first one was weird...
             seq_str = seq_str[:genome_length]
             
        current_sequences.append(list(seq_str))
        ids.append(rec.id)
    
    # 2. Setup Loop
    os.makedirs(output_dir, exist_ok=True)
    daily_seq_dir = os.path.join(output_dir, "daily_sequences")
    os.makedirs(daily_seq_dir, exist_ok=True)
    
    # Rolling Logic
    # Calculates how many sequences "advance" to the next mutation state per day
    updates_per_day = (mutations_per_week / 7.0) * num_sequences
    print(f"  Updates per day: {updates_per_day:.4f} sequences")
    
    # Generator
    mut_gen = DeterministicMutationGenerator(current_sequences)
    
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
    print(f"  Collecting {tree_samples} sequences for tree (sampled uniformly over time)...")
    
    records_for_tree = []
    id_to_day_map = {}
    
    # Sample uniformly
    # Interval = sim_days / N
    # We use linspace to get N points in [0, sim_days]
    sample_days = np.linspace(0, sim_days, tree_samples, dtype=int)
    
    # We might have duplicates if tree_samples > sim_days
    # Use counter to know how many to pick per day
    day_counts = defaultdict(int)
    for d in sample_days:
        if d <= sim_days:
            day_counts[d] += 1
            
    print(f"  Sampling from {len(day_counts)} distinct timepoints...")

    for d, count in sorted(day_counts.items()):
        fname = os.path.join(daily_seq_dir, f"day_{d}.fasta")
        if not os.path.exists(fname):
             continue
        
        # Load file
        day_records = list(SeqIO.parse(fname, "fasta"))
        if not day_records: continue
        
        # Sample 'count' sequences randomly
        if count > len(day_records):
            selected = random.choices(day_records, k=count)
        else:
            selected = random.sample(day_records, count)
            
        for rec in selected:
            unique_id = f"{rec.id}_day{d}_{random.randint(1000,9999)}"
            new_rec = SeqRecord(Seq(str(rec.seq)), id=unique_id, name=unique_id, description=unique_id)
            records_for_tree.append(new_rec)
            id_to_day_map[unique_id] = d
    # But `tree_samples` is not defined yet.
    # So I need to update the signature first.


    plot_final_tree(records_for_tree, id_to_day_map, sim_days, output_dir, mutation_rate_site_day)
    print(f"Done. Output in {output_dir}")

def save_daily_fasta(ids, sequences, output_dir, day):
    filename = os.path.join(output_dir, f"day_{day}.fasta")
    with open(filename, "w") as f:
        for id_str, seq_list in zip(ids, sequences):
            f.write(f">{id_str}\n{''.join(seq_list)}\n")

def plot_final_tree(records, id_to_day_map, max_day, output_dir, mutation_rate):
    # Build Tree (UPGMA or NJ)
    print(f"  Calculating distance matrix for {len(records)} sequences (Optimized)...")
    
    if not records:
        print("No records to plot.")
        return

    # 0. Export Subsampled FASTA
    print("  Exporting subsampled FASTA...")
    fasta_path = os.path.join(output_dir, "subsampled_tree_sequences.fasta")
    SeqIO.write(records, fasta_path, "fasta")

    # 1. Convert sequences to numpy byte array (N, L)
    L = len(records[0].seq)
    N = len(records)
    
    # Pre-allocate numpy array of uint8
    seq_matrix = np.zeros((N, L), dtype=np.uint8)
    names = []
    
    print("  Encoding sequences...")
    for i, rec in enumerate(records):
        names.append(rec.id)
        seq_str = str(rec.seq)
        if len(seq_str) > L: seq_str = seq_str[:L]
        elif len(seq_str) < L: seq_str = seq_str.ljust(L, '-')
        seq_matrix[i] = np.frombuffer(seq_str.encode('latin-1'), dtype=np.uint8)

    # 2. Compute Distance Matrix (Hamming)
    print("  Computing pairwise hamming distances...")
    condensed_dist = distance.pdist(seq_matrix, 'hamming')
    square_dist = distance.squareform(condensed_dist)
    
    # 2b. Export Distance Matrix
    print("  Exporting distance matrix...")
    dm_path = os.path.join(output_dir, "distance_matrix.csv")
    with open(dm_path, "w") as f:
        f.write("," + ",".join(names) + "\n")
        for i in range(N):
            # Format to 6 decimal places
            row_vals = [f"{x:.6f}" for x in square_dist[i]]
            f.write(f"{names[i]}," + ",".join(row_vals) + "\n")
    
    # 3. Convert to BioPython DistanceMatrix
    lower_triangular = [square_dist[i, :i+1].tolist() for i in range(N)]
    dm = DistanceMatrix(names, lower_triangular)
    
    # 4. Build Trees (NJ, UPGMA, Parsimony)
    algorithms = ['nj', 'upgma' ] #, 'parsimony']
    constructor = DistanceTreeConstructor()
    nj_tree = None # To store for Parsimony starting tree
    
    for algo in algorithms:
        print(f"  Building {algo.upper()} tree...")
        
        # Setup Figure
        fig, ax = plt.subplots(figsize=(12, 10))
        norm = mcolors.Normalize(vmin=0, vmax=max_day)
        cmap = cm.turbo 
        title = f"Population Structure ({algo.upper()})\nRate: {mutation_rate} muts/site/day\nTime: {max_day} Days"
        
        tree = None
        
        if algo == 'parsimony':
            print("    Calculating Parsimony tree (this may take a while)...")
            try:
                # 1. Prepare Alignment (ensure rectangular)
                aligned_records = []
                for rec in records:
                    seq_str = str(rec.seq)
                    if len(seq_str) > L: seq_str = seq_str[:L]
                    elif len(seq_str) < L: seq_str = seq_str.ljust(L, '-')
                    # Create clean record for alignment
                    # Note: Parsimony works with Seq/SeqRecord objects
                    aligned_records.append(SeqRecord(Seq(seq_str), id=rec.id, name=rec.id, description=rec.id))
                
                alignment = MultipleSeqAlignment(aligned_records)
                
                # 2. Setup Parsimony Search
                # We use the NJ tree as a starting point to be efficient
                # Recalculate NJ tree to ensure a clean starting point and avoid deepcopy issues
                start_tree = constructor.nj(dm)
                
                scorer = ParsimonyScorer()
                searcher = NNITreeSearcher(scorer)
                pars_constructor = ParsimonyTreeConstructor(searcher, start_tree)
                
                # Build tree
                tree = pars_constructor.build_tree(alignment)
                
            except Exception as e:
                print(f"    Error building Parsimony tree: {e}")
                plt.close()
                continue
                
        elif algo == 'nj':
            tree = constructor.nj(dm)
        elif algo == 'upgma':
            tree = constructor.upgma(dm)

        # Root at earliest sequence
        if tree:
            try:
                # Find the earliest day
                min_day = min(id_to_day_map.values()) if id_to_day_map else 0
                
                # Find a sequence from that earliest day
                outgroup = None
                for name, day in id_to_day_map.items():
                    if day == min_day:
                        outgroup = name
                        break
                
                if outgroup:
                    print(f"    Rooting {algo.upper()} tree with earliest sequence (Day {min_day}): {outgroup}")
                    outgroup_clade = None
                    for c in tree.get_terminals():
                        if c.name == outgroup:
                            outgroup_clade = c
                            break
                    if outgroup_clade:
                         tree.root_with_outgroup(outgroup_clade)
                    else:
                         tree.root_with_outgroup({"name": outgroup})
                else:
                     print("    Warning: No sequence found for rooting. Using midpoint rooting.")
                     tree.root_at_midpoint()
            except Exception as e:
                print(f"    Warning: Could not root {algo.upper()} tree. {e}")
                try: tree.root_at_midpoint()
                except: pass
            
            # Color mapping
            for clade in tree.get_terminals():
                cid = clade.name
                day = id_to_day_map.get(cid, 0)
                clade.custom_color = mcolors.to_hex(cmap(norm(day)))
            
            # Ladderize
            tree.ladderize()
            
            draw_tree_custom(ax, tree, title, color_attr='custom_color')
        
        # Add Colorbar (Common)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.04)
        cbar.set_label('Day of Sample')
        
        output_path = os.path.join(output_dir, f"tree_{algo}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Evolution")
    parser.add_argument("--input_fasta", required=True, help="Input alignment")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--days", type=int, default=100, help="Simulation days")
    parser.add_argument("--rate", type=float, default=2.7e-6, help="Mutations per site per day")
    parser.add_argument("--downsample", type=int, default=-1, help="Downsample initial population to N sequences")
    parser.add_argument("--tree_samples", type=int, default=200, help="Number of sequences to sample for final tree")
    
    args = parser.parse_args()
    
    run_deterministic_evolution(
        args.input_fasta, 
        args.output_dir, 
        args.days, 
        args.rate,
        args.downsample,
        args.tree_samples
    )
# mkdir /home1/oml4h/sim_test_cam/output
# python /home1/oml4h/sim_test/evolve_deterministic.py --input_fasta /home1/oml4h/COG_UK_results/CAMBS/weekly_sequences_2021-12-06_to_2021-12-12.fasta --output_dir /home1/oml4h/sim_test/test_output/CAMBS --days 1000 --rate 2.7e-6 --downsample 30
#   python /home1/oml4h/sim_test/evolve_deterministic.py --input_fasta /home1/oml4h/COG_UK_results/CAMBS/weekly_sequences_2022-01-10_to_2022-01-16.fasta --output_dir /home1/oml4h/sim_test/test_output/CAMBS/Jan --days 1000 --rate 2.7e-6 --downsample 30



#python /home1/oml4h/sim_test/evolve_deterministic.py --input_fasta /home1/oml4h/sim_test/test_output/poly_a_sequences.fasta --output_dir /home1/oml4h/sim_test/test_output/polyA --days 1000 --rate 2.7e-6 --downsample 30