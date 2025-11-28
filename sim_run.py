import argparse
import os
import numpy as np
import simulation_lib as sim
import matplotlib.pyplot as plt
import pandas as pd

def run_simulation(args):
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parameters
    params = {
        'SIMULATION_DAYS': args.days,
        'N_PATIENTS': args.n_patients,
        'N_HCW': args.n_hcw,
        'N_WARDS': args.n_wards,
        'PROB_DETECT': args.prob_detect,
        'PROB_SEQ': args.prob_seq,
        'ISOLATION_CAPACITY': args.isolation_capacity,
        'IMPORTATION_DAILY_PROB': args.import_rate,

        'GENOME_LENGTH': 29903,
        'MUTATION_RATE': 2.7e-6,
        'COMMUNITY_DIVERSITY_LEVEL': 0.001,
        'NUM_COMMUNITY_LINEAGES': 2,
        'COMMUNITY_POP_SIZE': 100,
        'BETA_ROOM': 0.15,
        'BETA_WARD': 0.01,
        'BETA_HCW_PAT': 0.015,
        'MEAN_INCUBATION': 3,
        'MEAN_RECOVERY': 10,
        'SYMPTOM_DETECT_AND_SAMPLE_MEAN': 4,
        'HCW_CROSS_WARD_PROB': 0.1,
        'GENETIC_LINK_THRESHOLD': 1 # SNPs
    }
    
    hospital = sim.Hospital(params['N_WARDS'], params['N_PATIENTS'], params['N_HCW'], isolation_capacity=params['ISOLATION_CAPACITY'])
    tracker = sim.PhylogenyTracker(
        params['GENOME_LENGTH'], 
        params['MUTATION_RATE'], 
        params['COMMUNITY_DIVERSITY_LEVEL'], 
        params['NUM_COMMUNITY_LINEAGES'], 
        params['COMMUNITY_POP_SIZE'],
        reference_path=args.reference_file,
        transition_prob=args.transition_prob
    )
    
    history = []
    known_sequences = [] # List of (agent, node_id)
    daily_census = [] # List of dicts for daily FASTA export
    
    print(f"Starting Simulation: {params['N_PATIENTS']+params['N_HCW']} Agents, {params['SIMULATION_DAYS']} Days.")
    
    for day in range(params['SIMULATION_DAYS']):
        # 0. Discharge / De-isolate
        for a in hospital.agents:
            if a.status == 'R' and a.is_isolated:
                hospital.discharge_patient(a)
                
        # 1. Evolve Community
        tracker.step_community(day)

        # 2. Importation Events
        if np.random.random() < params['IMPORTATION_DAILY_PROB']:
            # Pick a random agent from the ENTIRE population
            target = np.random.choice(hospital.agents)
            
            if target.status == 'S':
                target.status = 'I'
                target.infection_time = day
                target.symptom_time = day + np.random.poisson(params['MEAN_INCUBATION'])
                
                # Determine if/when detection happens
                if np.random.random() < params['PROB_DETECT']:
                    target.scheduled_detection_time = target.symptom_time + np.random.poisson(params['SYMPTOM_DETECT_AND_SAMPLE_MEAN'])
                else:
                    target.scheduled_detection_time = None
                
                variant = np.random.randint(0, params['NUM_COMMUNITY_LINEAGES'])
                node_id = tracker.add_importation(day, variant)
                target.infected_by_node = node_id

        # 3. Contact & Transmission
        contacts = hospital.get_contacts(day_seed=day, params=params)
        new_infections = []
        
        for p1, p2, prob in contacts:
            source, target = None, None
            if p1.status == 'I' and p2.status == 'S':
                source, target = p1, p2
            elif p2.status == 'I' and p1.status == 'S':
                source, target = p2, p1
            
            if source:
                if np.random.random() < prob:
                    if target not in [x[0] for x in new_infections]:
                        new_infections.append((target, source))

        # Apply Infections
        for target, source in new_infections:
            target.status = 'I'
            target.infection_time = day
            target.symptom_time = day + np.random.poisson(params['MEAN_INCUBATION'])
            
            # Determine if/when detection happens
            if np.random.random() < params['PROB_DETECT']:
                target.scheduled_detection_time = target.symptom_time + np.random.poisson(params['SYMPTOM_DETECT_AND_SAMPLE_MEAN'])
            else:
                target.scheduled_detection_time = None
            
            new_node, updated_source_node = tracker.add_transmission(source.infected_by_node, day)
            target.infected_by_node = new_node
            source.infected_by_node = updated_source_node

        # 4. Clinical Progression, Sampling & Infection Control
        for a in hospital.agents:
            if a.status == 'I':
                # --- A. Clinical Detection Logic ---
                if not a.is_detected:
                    if a.scheduled_detection_time is not None and day >= a.scheduled_detection_time:
                        a.is_detected = True
                        
                        # --- B. Clinical Sequencing & Isolation (Immediate on Detection) ---
                        # Sequencing Check
                        if np.random.random() < params['PROB_SEQ']:
                            # Note: We create a specific 'clinical sample' node here for the record
                            # But for daily validation, we use the census node below.
                            # Actually, let's keep the clinical sample separate in the tree structure
                            # so we know exactly what the hospital "saw".
                            sample_node, updated_node = tracker.add_sample(a.infected_by_node, day)
                            a.is_sampled = True
                            a.sample_node = sample_node
                            a.infected_by_node = updated_node
                            a.sample_time = day
                            
                            # --- REAL-TIME INFECTION CONTROL ---
                            # Compare with known sequences to infer links
                            linked = False
                            for known_agent, known_node in known_sequences:
                                dist = tracker.get_pairwise_distance(sample_node, known_node)
                                if dist <= params['GENETIC_LINK_THRESHOLD']:
                                    linked = True
                                    # Isolate the known contact if still active
                                    if known_agent.status == 'I':
                                        hospital.try_isolate_patient(known_agent, day)
                            
                            if linked:
                                hospital.try_isolate_patient(a, day)
                                
                            known_sequences.append((a, sample_node))
                
                # --- C. Daily Census Sequencing (Validation) ---
                # We record the virus in EVERY infected agent, every day.
                # This creates a dense tree but ensures we have ground truth.
                census_node, updated_node_census = tracker.add_sample(a.infected_by_node, day)
                a.infected_by_node = updated_node_census # Update internal state to this new backbone
                
                daily_census.append({
                    'day': day,
                    'agent_id': a.id,
                    'role': a.role,
                    'ward': a.ward_id,
                    'node_id': census_node,
                    'is_detected': a.is_detected
                })
                
                if day > a.infection_time + params['MEAN_RECOVERY']:
                    a.status = 'R'
                    if a.is_isolated:
                        hospital.discharge_patient(a)

        # 5. Data Logging
        # We track the state of the epidemic daily.
        s_count = sum(1 for a in hospital.agents if a.status == 'S')
        i_count = sum(1 for a in hospital.agents if a.status == 'I')
        r_count = sum(1 for a in hospital.agents if a.status == 'R')
        
        # Split by Role for detailed analysis
        s_pat = sum(1 for a in hospital.agents if a.status == 'S' and a.role == 'PATIENT')
        i_pat = sum(1 for a in hospital.agents if a.status == 'I' and a.role == 'PATIENT')
        r_pat = sum(1 for a in hospital.agents if a.status == 'R' and a.role == 'PATIENT')
        
        s_hcw = sum(1 for a in hospital.agents if a.status == 'S' and a.role == 'HCW')
        i_hcw = sum(1 for a in hospital.agents if a.status == 'I' and a.role == 'HCW')
        r_hcw = sum(1 for a in hospital.agents if a.status == 'R' and a.role == 'HCW')
        
        # Track Isolation
        n_iso = hospital.n_isolated
        
        history.append({
            'day': day, 
            'S': s_count, 'I': i_count, 'R': r_count,
            'S_pat': s_pat, 'I_pat': i_pat, 'R_pat': r_pat,
            'S_hcw': s_hcw, 'I_hcw': i_hcw, 'R_hcw': r_hcw,
            'Isolated': n_iso
        })
        
    # ==========================================
    # 5. EXPORT & FINALIZATION
    # ==========================================
    
    # A. Plot Epidemic Curves
    sim.save_sir_curves(history, args.output_dir)
    print(f"Saved sir_curves.png to {args.output_dir}")
    
    sim.save_recovered_split(history, args.output_dir)
    print(f"Saved recovered_split.png to {args.output_dir}")

    # B. Finalize Phylogeny
    # This converts the recorded nodes and edges into a tskit TreeSequence.
    # It also simulates mutations along the branches.
    ts = tracker.finalize_tree(max_time=params['SIMULATION_DAYS'])
    ts.dump(os.path.join(args.output_dir, "hospital_outbreak.trees"))
    print(f"Saved hospital_outbreak.trees to {args.output_dir}")

    # C. Extract Genetic Data
    # Convert the tree structure into a CSV of mutations per patient.
    sim.save_mutations_csv(ts, hospital, args.output_dir)
    print(f"Saved mutations_per_patient.csv to {args.output_dir}")

    # D. Generate FASTA Sequences
    # Reconstruct full genome sequences for each sample from the tree.
    sim.save_fasta(ts, hospital, args.output_dir)
    print(f"Saved sampled_sequences.fasta to {args.output_dir}")

    # Generate daily FASTA files for validation
    if args.daily_fastas:
        sim.save_daily_fastas(ts, daily_census, args.output_dir, params['SIMULATION_DAYS'])
        print(f"Saved daily FASTA files to {args.output_dir}/daily_sequences")

    sim.save_node_ids(hospital, args.output_dir)
    print(f"Saved hospital_node_ids.txt to {args.output_dir}")

    print("Generating split phylogenetic trees...")
    sim.generate_plots(ts, hospital, args.output_dir, params['SIMULATION_DAYS'])
    print(f"Saved tree plots to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hospital Outbreak Simulation")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save output files")
    parser.add_argument("--days", type=int, default=100, help="Number of simulation days")
    parser.add_argument("--n-patients", type=int, default=700, help="Number of patients")
    parser.add_argument("--n-hcw", type=int, default=300, help="Number of HCWs")
    parser.add_argument("--n-wards", type=int, default=10, help="Number of wards")
    parser.add_argument("--prob-detect", type=float, default=0.4, help="Probability of detecting a case")
    parser.add_argument("--prob-seq", type=float, default=1.0, help="Probability of sequencing a detected case")
    parser.add_argument("--isolation-capacity", type=int, default=20, help="Number of isolation rooms available")
    parser.add_argument("--import_rate", type=float, default=0.05, help="Importation rate")
    parser.add_argument("--reference-file", type=str, default=None, help="Path to reference FASTA file")
    parser.add_argument("--transition-prob", type=float, default=0.7, help="Probability of transition (vs transversion)")
    parser.add_argument("--daily-fastas", action="store_true", help="Generate daily FASTA files")
    # above should be a function of wards presumably rather than a fixed number, same thing within a ward should have another hierarchy
    
    args = parser.parse_args()
    run_simulation(args)