"""
Tests for run_population_sim.py

Tests the population genetics simulation functionality including:
- Parameter validation
- Phylogeny tracker initialization
- Diversity calculations
- Output file generation
"""

import unittest
import tempfile
import shutil
import os
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_lib import PhylogenyTracker
from run_population_sim import (
    hamming_distance,
    calculate_pairwise_distances,
    count_segregating_sites,
    save_daily_fastas_population
)


class TestHammingDistance(unittest.TestCase):
    """Test Hamming distance calculation."""
    
    def test_identical_sequences(self):
        """Identical sequences should have distance 0."""
        seq1 = "ACGTACGT"
        seq2 = "ACGTACGT"
        self.assertEqual(hamming_distance(seq1, seq2), 0)
    
    def test_different_sequences(self):
        """Different sequences should return correct distance."""
        seq1 = "ACGTACGT"
        seq2 = "ACGTCCGT"
        self.assertEqual(hamming_distance(seq1, seq2), 1)
        
        seq1 = "AAAAAAAA"
        seq2 = "TTTTTTTT"
        self.assertEqual(hamming_distance(seq1, seq2), 8)
    
    def test_empty_sequences(self):
        """Empty sequences should have distance 0."""
        self.assertEqual(hamming_distance("", ""), 0)


class TestSeggregatingSites(unittest.TestCase):
    """Test segregating sites counting."""
    
    def test_no_variation(self):
        """Identical sequences have 0 segregating sites."""
        seqs = ["ACGT", "ACGT", "ACGT"]
        self.assertEqual(count_segregating_sites(seqs), 0)
    
    def test_single_site(self):
        """One variable position."""
        seqs = ["ACGT", "ACGT", "TCGT"]
        self.assertEqual(count_segregating_sites(seqs), 1)
    
    def test_multiple_sites(self):
        """Multiple variable positions."""
        seqs = ["AAAA", "ATAT", "TTTT"]
        # Positions: 0(A/T), 1(A/T), 2(A/T), 3(A/T) = 4 segregating sites
        self.assertEqual(count_segregating_sites(seqs), 4)
    
    def test_empty_list(self):
        """Empty sequence list."""
        self.assertEqual(count_segregating_sites([]), 0)


class TestPairwiseDistances(unittest.TestCase):
    """Test pairwise distance calculations."""
    
    def test_two_nodes(self):
        """Calculate distance between two nodes."""
        node_to_seq = {
            0: "ACGTACGT",
            1: "ACGTCCGT"
        }
        mean, median, distances = calculate_pairwise_distances([0, 1], node_to_seq)
        
        self.assertEqual(mean, 1.0)
        self.assertEqual(median, 1.0)
        self.assertEqual(len(distances), 1)
        self.assertEqual(distances[0], 1)
    
    def test_multiple_nodes(self):
        """Calculate distances for multiple nodes."""
        node_to_seq = {
            0: "AAAA",
            1: "AAAT",
            2: "AATT",
            3: "ATTT"
        }
        mean, median, distances = calculate_pairwise_distances([0, 1, 2, 3], node_to_seq)
        
        # Expected distances: 0-1:1, 0-2:2, 0-3:3, 1-2:1, 1-3:2, 2-3:1
        # Mean = (1+2+3+1+2+1)/6 = 10/6 = 1.667
        self.assertAlmostEqual(mean, 10/6, places=2)
        self.assertEqual(len(distances), 6)
    
    def test_single_node(self):
        """Single node returns zeros."""
        node_to_seq = {0: "ACGT"}
        mean, median, distances = calculate_pairwise_distances([0], node_to_seq)
        
        self.assertEqual(mean, 0.0)
        self.assertEqual(median, 0.0)
        self.assertEqual(len(distances), 0)


class TestPhylogenyTracker(unittest.TestCase):
    """Test PhylogenyTracker initialization and basic operations."""
    
    def test_initialization(self):
        """Test basic initialization."""
        phylogeny = PhylogenyTracker(
            genome_length=1000,
            mutation_rate=0.0001,
            community_diversity_level=0.0,
            num_community_lineages=2,
            community_pop_size=10,
            burn_in_days=5
        )
        
        self.assertEqual(phylogeny.genome_length, 1000)
        self.assertEqual(phylogeny.mutation_rate, 0.0001)
        self.assertEqual(len(phylogeny.community_lineages), 2)
        self.assertEqual(len(phylogeny.community_lineages[0]), 10)
        self.assertEqual(len(phylogeny.community_lineages[1]), 10)
    
    def test_step_community(self):
        """Test stepping community forward."""
        phylogeny = PhylogenyTracker(
            genome_length=1000,
            mutation_rate=0.0001,
            community_diversity_level=0.0,
            num_community_lineages=1,
            community_pop_size=5,
            burn_in_days=2
        )
        
        initial_gen = list(phylogeny.community_lineages[0])
        phylogeny.step_community(1)
        new_gen = phylogeny.community_lineages[0]
        
        # Population size should remain constant
        self.assertEqual(len(new_gen), 5)
        
        # New generation should be different nodes
        self.assertNotEqual(set(initial_gen), set(new_gen))
    
    def test_finalize_tree(self):
        """Test tree finalization."""
        phylogeny = PhylogenyTracker(
            genome_length=1000,
            mutation_rate=0.0001,
            community_diversity_level=0.0,
            num_community_lineages=1,
            community_pop_size=5,
            burn_in_days=2
        )
        
        # Step forward a few days
        for day in range(3):
            phylogeny.step_community(day)
        
        ts = phylogeny.finalize_tree(max_time=3)
        
        # Basic checks
        self.assertIsNotNone(ts)
        self.assertGreater(ts.num_nodes, 0)
        self.assertGreater(ts.num_samples, 0)


class TestOutputGeneration(unittest.TestCase):
    """Test output file generation."""
    
    def setUp(self):
        """Create temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_fasta_generation(self):
        """Test FASTA file generation."""
        # Create a simple phylogeny
        phylogeny = PhylogenyTracker(
            genome_length=100,
            mutation_rate=0.001,
            community_diversity_level=0.0,
            num_community_lineages=1,
            community_pop_size=3,
            burn_in_days=1
        )
        
        # Run for 2 days
        daily_census = []
        for day in range(2):
            phylogeny.step_community(day)
            for lineage_idx, lineage in enumerate(phylogeny.community_lineages):
                for individual_idx, node_id in enumerate(lineage):
                    daily_census.append({
                        'day': day,
                        'lineage': lineage_idx,
                        'individual_id': f"L{lineage_idx}_I{individual_idx}",
                        'node_id': node_id
                    })
        
        ts = phylogeny.finalize_tree(max_time=2)
        
        # Generate FASTAs
        save_daily_fastas_population(ts, daily_census, self.test_dir, 2)
        
        # Check files exist
        daily_dir = os.path.join(self.test_dir, "daily_sequences")
        self.assertTrue(os.path.exists(daily_dir))
        self.assertTrue(os.path.exists(os.path.join(daily_dir, "day_0.fasta")))
        self.assertTrue(os.path.exists(os.path.join(daily_dir, "day_1.fasta")))
        
        # Check FASTA content
        with open(os.path.join(daily_dir, "day_0.fasta"), "r") as f:
            content = f.read()
            self.assertIn(">", content)  # Has headers
            self.assertIn("L0_I", content)  # Has lineage IDs
            # Count sequences (3 sequences = 3 headers)
            self.assertEqual(content.count(">"), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for full simulation pipeline."""
    
    def setUp(self):
        """Create temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_small_simulation(self):
        """Test a complete small simulation."""
        from run_population_sim import run_population_simulation
        
        # Run a tiny simulation
        run_population_simulation(
            num_lineages=2,
            pop_size=5,
            mutation_rate=0.001,
            burn_in_days=2,
            simulation_days=3,
            genome_length=500,
            output_dir=self.test_dir,
            seed=42
        )
        
        # Check all expected outputs exist
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "tree_sequence.trees")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "diversity_stats.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "pairwise_distance_over_time.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "pairwise_distance_by_week.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "diversity_breakdown.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "segregating_sites.png")))
        
        # Check daily sequences directory
        daily_dir = os.path.join(self.test_dir, "daily_sequences")
        self.assertTrue(os.path.exists(daily_dir))
        
        # Should have day_0.fasta, day_1.fasta, day_2.fasta
        for day in range(3):
            fasta_file = os.path.join(daily_dir, f"day_{day}.fasta")
            self.assertTrue(os.path.exists(fasta_file))
            
            # Check FASTA has content
            with open(fasta_file, "r") as f:
                content = f.read()
                self.assertGreater(len(content), 0)
                self.assertIn(">", content)
        
        # Check config file has correct parameters
        with open(os.path.join(self.test_dir, "config.json"), "r") as f:
            config = json.load(f)
            self.assertEqual(config["num_lineages"], 2)
            self.assertEqual(config["pop_size"], 5)
            self.assertEqual(config["simulation_days"], 3)
        
        # Check diversity stats
        with open(os.path.join(self.test_dir, "diversity_stats.json"), "r") as f:
            stats = json.load(f)
            self.assertIn("days", stats)
            self.assertIn("mean_pairwise_distance", stats)
            self.assertIn("median_pairwise_distance", stats)


if __name__ == '__main__':
    unittest.main()
