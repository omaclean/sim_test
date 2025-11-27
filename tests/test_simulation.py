import pytest
import os
import shutil
import tempfile
import sys
from unittest.mock import patch

# Add parent directory to path to import sim_test_full
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sim_test_full

def test_simulation_runs_and_produces_output():
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock arguments
        class Args:
            output_dir = temp_dir
            days = 10
            n_patients = 50
            n_hcw = 20
            n_wards = 2
            
        args = Args()
        
        # Run simulation
        sim_test_full.run_simulation(args)
        
        # Check if files are created
        expected_files = [
            "sir_curves.png",
            "hospital_outbreak.trees",
            "mutations_per_patient.csv",
            "sampled_sequences.fasta",
            "hospital_node_ids.txt",
            "recovered_split.png"
            # "hospital_tree_ward.png", # Might not be created if no samples
            # "hospital_tree_time.png",
            # "community_tree_time.png"
        ]
        
        # Note: Plots might not be generated if no infections/samples occur in 10 days with small pop
        # But we can check for the main data files which are always created (even if empty or just headers)
        
        for f in expected_files:
            assert os.path.exists(os.path.join(temp_dir, f)), f"File {f} was not created"
            
        # Check if mutation csv has header
        with open(os.path.join(temp_dir, "mutations_per_patient.csv"), 'r') as f:
            header = f.readline()
            assert "agent_id,role,ward,position,alt" in header

def test_simulation_with_custom_output_dir():
    # Test that it creates the directory if it doesn't exist
    test_dir = "test_output_custom"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    try:
        class Args:
            output_dir = test_dir
            days = 5
            n_patients = 20
            n_hcw = 5
            n_wards = 1
            
        args = Args()
        sim_test_full.run_simulation(args)
        
        assert os.path.exists(test_dir)
        assert os.path.exists(os.path.join(test_dir, "hospital_outbreak.trees"))
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_invalid_parameters():
    # Test invalid ward count
    class Args:
        output_dir = "dummy"
        days = 10
        n_patients = 10
        n_hcw = 10
        n_wards = 0 # Invalid
        
    with pytest.raises(ValueError, match="n_wards must be positive"):
        sim_test_full.run_simulation(Args())
        
    # Test negative patients
    class Args2:
        output_dir = "dummy"
        days = 10
        n_patients = -5 # Invalid
        n_hcw = 10
        n_wards = 5
        
    with pytest.raises(ValueError, match="n_patients must be non-negative"):
        sim_test_full.run_simulation(Args2())

def test_zero_days():
    # Simulation should run but produce empty/minimal output
    with tempfile.TemporaryDirectory() as temp_dir:
        class Args:
            output_dir = temp_dir
            days = 0
            n_patients = 10
            n_hcw = 10
            n_wards = 2
            
        sim_test_full.run_simulation(Args())
        
        # Check that files exist even if empty
        assert os.path.exists(os.path.join(temp_dir, "sir_curves.png"))
        assert os.path.exists(os.path.join(temp_dir, "hospital_outbreak.trees"))
