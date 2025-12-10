import tskit
import argparse
import os
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Bio import Phylo
from io import StringIO

def debug_plot(ts_path, daily_census_path, output_dir):
    ts = tskit.load(ts_path)
    
    # Load census
    # We need to reconstruct daily_census from the file or just mock it?
    # The user didn't provide the census file, but it's in memory in the main script.
    # I can try to reconstruct it from the tree metadata if it was saved, but it wasn't.
    # However, I can infer the day from the node time.
    # In the simulation:
    # node.time is the time of the node.
    # For samples, time = day (roughly, maybe with epsilon).
    
    # Let's just use node time for coloring to test.
    
    # Subsample
    all_samples = list(ts.samples())
    if len(all_samples) > 200:
        sampled_nodes = list(np.random.choice(all_samples, 200, replace=False))
    else:
        sampled_nodes = all_samples
        
    print(f"Plotting {len(sampled_nodes)} tips.")
    
    # Simplify
    ts_simplified, node_map = ts.simplify(samples=sampled_nodes, map_nodes=True)
    
    # Count mutations per node in simplified tree
    node_mut_counts = collections.defaultdict(int)
    for mut in ts_simplified.mutations():
        node_mut_counts[mut.node] += 1
        
    # Get Newick
    tree_obj = ts_simplified.first()
    # Label ALL nodes with their ID so we can map them back
    node_labels = {n: str(n) for n in range(ts_simplified.num_nodes)}
    newick_str = tree_obj.newick(node_labels=node_labels)
    
    tree_viz = Phylo.read(StringIO(newick_str), "newick")
    
    # Assign branch lengths based on mutations
    max_dist = 0
    for clade in tree_viz.find_clades():
        if clade.name:
            try:
                node_id = int(clade.name)
                muts = node_mut_counts[node_id]
                clade.branch_length = muts
                # print(f"Node {node_id}: {muts} mutations")
            except ValueError:
                pass
    
    # Color by time (using node time from ts_simplified)
    cmap_time = plt.get_cmap("viridis")
    # Time in ts is forward time?
    # In simulation:
    # child_time = day + epsilon
    # So max time is simulation_end.
    
    max_time = np.max(ts_simplified.nodes_time)
    norm = plt.Normalize(0, max_time)
    
    colored_count = 0
    for clade in tree_viz.get_terminals():
        if clade.name:
            node_id = int(clade.name)
            time = ts_simplified.nodes_time[node_id]
            rgba = cmap_time(norm(time))
            clade.color = mcolors.to_hex(rgba)
            colored_count += 1
            
    print(f"Colored {colored_count} tips.")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    Phylo.draw(tree_viz, axes=ax, do_show=False, show_confidence=False, label_func=lambda x: None)
    
    plt.xlabel("Divergence (mutations)")
    plt.savefig(os.path.join(output_dir, "debug_tree.png"))
    print(f"Saved to {os.path.join(output_dir, 'debug_tree.png')}")

if __name__ == "__main__":
    debug_plot("test_output/testoutOM4/tree_sequence.trees", None, "test_output/testoutOM4")
