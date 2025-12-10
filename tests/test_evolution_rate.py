
import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_lib import PhylogenyTracker

class TestEvolutionRate(unittest.TestCase):
    def test_deterministic_rate(self):
        print("\nTesting Deterministic Evolution Rate...")
        
        # Setup
        genome_length = 10000
        mutation_rate = 1e-4 # 1 mutation per day
        sim_days = 100
        pop_size = 100
        target_dist = 10
        
        tracker = PhylogenyTracker(
            genome_length=genome_length,
            mutation_rate=mutation_rate,
            community_diversity_level=0.0,
            num_community_lineages=1,
            community_pop_size=pop_size,
            burn_in_days=10,
            deterministic=True,
            target_pairwise_distance=target_dist,
            branching_interval=1,
            turnover_rate=0.1
        )
        
        # Helper to get mutation positions for a node
        def get_mutation_positions(node_id):
            pos_set = set()
            curr = node_id
            while curr != -1:
                if curr in tracker.node_mutations:
                    # Get site IDs
                    site_ids = tracker.node_mutations[curr]
                    # Look up positions in sites table
                    for site_id in site_ids:
                        pos = tracker.sites.position[site_id]
                        pos_set.add(pos)
                curr = tracker.parent_map.get(curr, -1)
            return pos_set

        # Get mutations at Day 0
        nodes_0 = tracker.community_lineages[0]
        muts_0 = [len(get_mutation_positions(n)) for n in nodes_0]
        mean_muts_0 = np.mean(muts_0)
        
        print(f"Day 0: Mean mutation count = {mean_muts_0:.2f}")
        
        # Run to Day 100
        for day in range(sim_days):
            tracker.step_community(day)
            
        nodes_100 = tracker.community_lineages[0]
        muts_100 = [len(get_mutation_positions(n)) for n in nodes_100]
        mean_muts_100 = np.mean(muts_100)
        
        print(f"Day {sim_days}: Mean mutation count = {mean_muts_100:.2f}")
        
        shift = mean_muts_100 - mean_muts_0
        expected_shift = mutation_rate * genome_length * sim_days
        
        print(f"Observed Shift (Muts): {shift:.2f}")
        print(f"Expected Shift (Muts): {expected_shift:.2f}")
        
        # Check if shift is within reasonable bounds (e.g. +/- 20%)
        self.assertAlmostEqual(shift, expected_shift, delta=max(10, expected_shift * 0.2))

if __name__ == "__main__":
    unittest.main()
