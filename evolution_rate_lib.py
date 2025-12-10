
import collections
import os
import numpy as np
import matplotlib.pyplot as plt
import tskit

def calculate_evolutionary_rate(ts, daily_census, output_dir, genome_length):
    """
    Calculate evolutionary rate (substitution rate) from root-to-tip distances.
    
    Args:
        ts: TreeSequence
        daily_census: List of dicts with {day, lineage, individual_id, node_id}
        output_dir: Output directory
        genome_length: Length of genome
        
    Returns:
        dict with rate statistics
    """
    print("Calculating evolutionary rate...")
    
    # Map node_id to day
    node_to_day = {}
    for record in daily_census:
        node_to_day[record['node_id']] = record['day']
        
    # Calculate root-to-tip distances (number of mutations)
    # We can use the number of mutations on the path from root to sample
    # tskit doesn't have a direct "root-to-tip mutations" function, but we can traverse
    
    # Or simpler: for each sample, count mutations above it
    # Since we have the full tree, we can iterate mutations and count for each sample
    
    # Efficient way:
    # 1. Iterate all mutations.
    # 2. For each mutation, find all samples below it.
    # 3. Increment counter for those samples.
    
    sample_mutations = collections.defaultdict(int)
    
    # Iterate over all trees (should be just one if no recombination, but safe to iterate)
    for tree in ts.trees():
        for mutation in tree.mutations():
            # A mutation at node u is inherited by all samples in the subtree of u
            # (unless back-mutated, but tskit handles this if we just count mutations on path)
            # Actually, tskit mutations are associated with a node.
            # Any sample descending from that node (and not blocked by another mutation at same site?)
            # In this simulation, we don't have back mutations in the tskit sense (infinite sites-ish or just overwrites)
            # But wait, simulation_lib uses tskit tables directly.
            # If we use tree.mutations(), it gives us mutations on the tree.
            
            # Let's use a simpler approach:
            # For each sample, traverse up to root and count mutations on edges.
            pass
            
    # Better approach using tskit's built-in capabilities
    # We can compute the number of mutations on the path from root to each sample
    
    samples = list(ts.samples())
    sample_mut_counts = {}
    sample_days = []
    sample_counts = []
    
    # We need to be careful about which tree we use if there are multiple (recombination)
    # But here we likely have one tree or consistent topology.
    # Let's assume the first tree covers the whole genome or use the average?
    # Actually, mutations are global in the table.
    
    # Let's iterate over all mutations and count how many ancestors of each sample have mutations
    # This is O(num_mutations * tree_depth), which is fine.
    
    # Even better:
    # node_mutations = collections.defaultdict(int)
    # for mut in ts.mutations():
    #     node_mutations[mut.node] += 1
    
    # Then for each sample, sum node_mutations on path to root.
    
    node_mut_counts = collections.defaultdict(int)
    for mut in ts.mutations():
        node_mut_counts[mut.node] += 1
        
    tree = ts.first()
    
    # Subsample if too many
    if len(samples) > 1000:
        sampled_indices = np.random.choice(len(samples), 1000, replace=False)
        samples_subset = [samples[i] for i in sampled_indices]
    else:
        samples_subset = samples
        
    for u in samples_subset:
        if u not in node_to_day:
            continue
            
        day = node_to_day[u]
        mut_count = 0
        
        # Traverse up
        curr = u
        while curr != tskit.NULL:
            mut_count += node_mut_counts[curr]
            curr = tree.parent(curr)
            
        sample_days.append(day)
        sample_counts.append(mut_count)
        
    if not sample_days:
        return {}
        
    # Linear regression
    slope, intercept = np.polyfit(sample_days, sample_counts, 1)
    rate_per_site_per_day = slope / genome_length
    rate_per_site_per_year = rate_per_site_per_day * 365
    
    print(f"  Estimated substitution rate: {slope:.4f} mutations/day")
    print(f"  Estimated substitution rate: {rate_per_site_per_year:.2e} subs/site/year")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sample_days, sample_counts, alpha=0.3, s=10)
    
    x_vals = np.array([min(sample_days), max(sample_days)])
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, 'r-', label=f'Rate: {slope:.4f} muts/day\n({rate_per_site_per_year:.2e} s/s/y)')
    
    plt.xlabel('Day')
    plt.ylabel('Root-to-Tip Distance (Mutations)')
    plt.title('Root-to-Tip Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'root_to_tip_regression.png'), dpi=300)
    plt.close()
    
    return {
        "slope_muts_per_day": slope,
        "intercept": intercept,
        "rate_subs_per_site_per_year": rate_per_site_per_year,
        "genome_length": genome_length
    }
