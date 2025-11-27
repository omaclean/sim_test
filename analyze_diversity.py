import tskit
import pandas as pd
import numpy as np

def analyze_diversity():
    print("Loading tree sequence...")
    ts = tskit.load("hospital_outbreak.trees")
    
    print("Loading hospital node IDs...")
    with open("hospital_node_ids.txt", "r") as f:
        hospital_nodes = [int(line.strip()) for line in f]
    
    hospital_set = set(hospital_nodes)
    all_samples = list(ts.samples())
    community_nodes = [n for n in all_samples if n not in hospital_set]
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Hospital samples: {len(hospital_nodes)}")
    print(f"Community samples: {len(community_nodes)}")
    
    if len(hospital_nodes) < 2 or len(community_nodes) < 2:
        print("Not enough samples to calculate diversity.")
        return

    # Calculate diversity (pi)
    # diversity returns the average number of pairwise differences per site
    # We multiply by sequence length to get average number of mutations difference
    div_hospital = ts.diversity(sample_sets=[hospital_nodes])[0]
    div_community = ts.diversity(sample_sets=[community_nodes])[0]
    
    # Convert to absolute number of differences
    seq_len = ts.sequence_length
    abs_div_hospital = div_hospital * seq_len
    abs_div_community = div_community * seq_len
    
    print("\n--- Diversity Results ---")
    print(f"Hospital Diversity (mean pairwise diffs): {abs_div_hospital:.2f}")
    print(f"Community Diversity (mean pairwise diffs): {abs_div_community:.2f}")
    
    if abs_div_community > abs_div_hospital:
        print("\nResult: Community diversity is GREATER than hospital diversity.")
        ratio = abs_div_community / abs_div_hospital if abs_div_hospital > 0 else float('inf')
        print(f"Ratio (Community/Hospital): {ratio:.2f}x")
    else:
        print("\nResult: Community diversity is LESS than or equal to hospital diversity.")

if __name__ == "__main__":
    analyze_diversity()
