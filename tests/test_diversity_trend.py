
import os
import sys
import numpy as np
import tskit
import collections
from simulation_lib import PhylogenyTracker

def calculate_pairwise_distances(node_list, tracker):
    # Helper to get mutations for a node
    def get_mutations(node_id):
        muts = set()
        curr = node_id
        while curr != -1:
            if curr in tracker.node_mutations:
                muts.update(tracker.node_mutations[curr])
            curr = tracker.parent_map.get(curr, -1)
        return muts

    distances = []
    n = len(node_list)
    # Sample pairs if too many
    pairs = []
    if n > 20:
        for _ in range(100):
            i, j = np.random.choice(n, 2, replace=False)
            pairs.append((node_list[i], node_list[j]))
    else:
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((node_list[i], node_list[j]))
                
    for u, v in pairs:
        muts_u = get_mutations(u)
        muts_v = get_mutations(v)
        dist = len(muts_u.symmetric_difference(muts_v))
        distances.append(dist)
            
    return np.mean(distances) if distances else 0.0

def test_diversity_trend():
    print("Testing Diversity Trend in Deterministic Mode (Large Pop)...")
    
    genome_length = 29903
    mutation_rate = 2.7e-7
    pop_size = 1000
    burn_in_days = 50
    sim_days = 100
    target_dist = 10
    branching_interval = 10
    
    # Initialize tracker
    tracker = PhylogenyTracker(
        genome_length=genome_length,
        mutation_rate=mutation_rate,
        community_diversity_level=0.0,
        num_community_lineages=1,
        community_pop_size=pop_size,
        burn_in_days=burn_in_days,
        deterministic=True,
        target_pairwise_distance=target_dist,
        branching_interval=branching_interval
    )
    
    print(f"Day 0 Diversity: {calculate_pairwise_distances(tracker.community_lineages[0], tracker):.2f}")
    
    # Run forward
    for day in range(sim_days):
        tracker.step_community(day)
        if day % 10 == 0:
            div = calculate_pairwise_distances(tracker.community_lineages[0], tracker)
            print(f"Day {day+1} Diversity: {div:.2f}")

if __name__ == "__main__":
    test_diversity_trend()
