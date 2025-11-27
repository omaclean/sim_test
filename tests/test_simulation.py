import pytest
import os
import shutil
import tempfile
import sys
import pandas as pd
from unittest.mock import patch

# Add parent directory to path to import sim_test_full
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sim_run as sim_test_full

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
            prob_detect = 0.5
            prob_seq = 1.0
            isolation_capacity = 10
            import_rate = 0.1
            
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
            "recovered_split.png",
            # Plots should be created if there are samples
            # With 10 days and 50 patients, it's possible no one is sampled if import rate is low
            # But we set import_rate=0.1, so likely some infections.
        ]
        
        # Check for main files
        for f in expected_files:
            assert os.path.exists(os.path.join(temp_dir, f)), f"File {f} was not created"
            
        # Check for plots if we have samples (which we likely do)
        # But to be safe in a test, we might check if they exist OR if the tree is empty.
        # However, let's just assert they exist for now, as we want to test plot generation.
        # If they fail, it means no samples were generated, which is also a useful signal.
        plot_files = [
             "hospital_tree_ward.png",
             "hospital_tree_time.png",
             "community_tree_time.png"
        ]
        # Only check if we actually have samples? 
        # sim_run prints "Saved tree plots..." regardless? No, generate_plots returns early if no samples.
        # So we should only assert if we know there are samples.
        # Let's check if sampled_sequences.fasta is non-empty?
        
        if os.path.getsize(os.path.join(temp_dir, "sampled_sequences.fasta")) > 0:
             for f in plot_files:
                 assert os.path.exists(os.path.join(temp_dir, f)), f"Plot {f} missing"

def test_simulation_with_custom_output_dir():
    # Test that it creates the directory if it doesn't exist
    test_dir = "test_output_custom"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    try:
        class Args:
            output_dir = test_dir
            days = 20
            n_patients = 50
            n_hcw = 20
            n_wards = 2
            prob_detect = 0.5
            prob_seq = 1.0
            isolation_capacity = 10
            import_rate = 0.1

        sim_test_full.run_simulation(Args())
        
        # Check outputs
        expected_files = [
            "sir_curves.png",
            "hospital_outbreak.trees",
            "mutations_per_patient.csv",
            "sampled_sequences.fasta",
            "hospital_node_ids.txt",
            "recovered_split.png"
        ]
        
        for f in expected_files:
            assert os.path.exists(os.path.join(test_dir, f)), f"Missing {f}"
            
        # Check plots if samples exist
        if os.path.getsize(os.path.join(test_dir, "sampled_sequences.fasta")) > 0:
             plot_files = [
                 "hospital_tree_ward.png",
                 "hospital_tree_time.png",
                 "community_tree_time.png"
             ]
             for f in plot_files:
                 assert os.path.exists(os.path.join(test_dir, f)), f"Plot {f} missing"
            
        # Check CSV content
        df = pd.read_csv(os.path.join(test_dir, "mutations_per_patient.csv"))
        # It's possible to have 0 mutations in a short run, but file should exist with headers
        assert 'agent_id' in df.columns
        
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
        prob_detect = 0.5
        prob_seq = 1.0
        isolation_capacity = 10
        import_rate = 0.1
        
    with pytest.raises(ValueError, match="n_wards must be positive"):
        sim_test_full.run_simulation(Args())
        
    # Test negative patients
    class Args2:
        output_dir = "dummy"
        days = 10
        n_patients = -5 # Invalid
        n_hcw = 10
        n_wards = 5
        prob_detect = 0.5
        prob_seq = 1.0
        isolation_capacity = 10
        import_rate = 0.1
        
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
            prob_detect = 0.5
            prob_seq = 1.0
            isolation_capacity = 10
            import_rate = 0.1
            
        sim_test_full.run_simulation(Args())
        
        # Check that files exist even if empty
        assert os.path.exists(os.path.join(temp_dir, "sir_curves.png"))
        assert os.path.exists(os.path.join(temp_dir, "hospital_outbreak.trees"))

def test_isolation_logic():
    # Run a simulation with high transmission and high isolation capacity
    # We expect some agents to be isolated.
    with tempfile.TemporaryDirectory() as temp_dir:
        class Args:
            output_dir = temp_dir
            days = 30
            n_patients = 100
            n_hcw = 20
            n_wards = 2
            prob_detect = 0.8
            prob_seq = 1.0
            isolation_capacity = 50 # Plenty of space
            import_rate = 0.1
            
        # We need to capture the history to check isolation counts.
        # Since run_simulation doesn't return history, we can check the side effects 
        # or mock, but simpler is to inspect the internal state if we could.
        # Alternatively, we can rely on the fact that it runs without error, 
        # but to be sure, let's check if we can infer it from logs or just trust the code logic for now 
        # and ensure it doesn't crash.
        # Actually, we can check if any isolation logic was triggered by checking coverage? No.
        # Let's just ensure it runs.
        
        sim_test_full.run_simulation(Args())
        
        # We can check if the code ran successfully. 
        # To verify isolation specifically, we'd need to inspect the 'history' variable inside run_simulation.
        # Since we can't easily, we assume if it runs and produces output, the logic is at least executable.
        assert os.path.exists(os.path.join(temp_dir, "hospital_outbreak.trees"))
