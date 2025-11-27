import tskit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import collections
from dataclasses import dataclass, field
from typing import List, Optional
from Bio import Phylo
from io import StringIO
import copy
import matplotlib.colors as mcolors

# ==========================================
# 1. SIMULATION PARAMETERS (TUNE HERE)
# ==========================================

# -- Time & Pop --
SIMULATION_DAYS = 100
N_PATIENTS = 700
N_HCW = 300
N_WARDS = 10

# -- Viral (SARS-CoV-2 Model) --
GENOME_LENGTH = 29903
# approx 2 substitutions per month ~ 2.2e-6 per site per day? 
# let's calculate: 24 muts / year / 30000 sites approx.
MUTATION_RATE = 2.7e-6  # substitutions/site/day

# -- Community / Background Diversity (The "Option B" Flexibility) --
# If > 0, we create multiple background lineages separated by this genetic distance
# This simulates different variants (e.g., Delta vs Omicron) co-circulating.
# 0.001 ~ 30 mutations difference. 0.0 = they start identical.
COMMUNITY_DIVERSITY_LEVEL = 0.001  # Genetic distance between background lineages
NUM_COMMUNITY_LINEAGES = 2         # e.g., 2 distinct variants circulating
COMMUNITY_POP_SIZE = 100           # Effective population size for each background lineage
COMMUNITY_EVOLUTION_RATE = MUTATION_RATE # Rate of evolution in community

# -- Transmission --
BETA_ROOM = 0.15     # Prob transmission in same room
BETA_WARD = 0.01     # Prob transmission in same ward (common area)
BETA_HCW_PAT = 0.015  # Prob transmission HCW <-> Patient
IMPORTATION_DAILY_PROB = 0.05 # Prob a new case enters from community daily

# -- Clinical / Sampling --
MEAN_INCUBATION = 3
MEAN_RECOVERY = 10
SAMPLE_LAG_MEAN = 4  # Days from symptom onset to sampling
PROB_DETECT_SEQUENCE = 0.4 # Probability an infection is eventually sequenced

# -- HCW Movement --
HCW_CROSS_WARD_PROB = 0.1 # Probability HCW visits a random ward instead of home ward

# ==========================================
# 2. DATA STRUCTURES & AGENTS
# ==========================================

@dataclass
class Agent:
    id: int
    role: str # 'PATIENT' or 'HCW'
    ward_id: int
    room_id: Optional[int] = None # Only for patients
    
    # State
    status: str = 'S' # S, E, I, R
    infection_time: float = -1.0
    symptom_time: float = -1.0
    
    # Genetics
    infected_by_node: int = -1 # The tskit node ID of the virus inside this person
    
    # Sampling
    is_sampled: bool = False
    sample_time: float = -1.0
    sample_node: int = -1 # The tskit node ID of the sample taken from this person

class Hospital:
    def __init__(self, n_wards, n_patients, n_hcw):
        self.wards = collections.defaultdict(list)
        self.agents = []
        
        # Create Wards and Rooms
        # Structure: Each ward has 20 rooms.
        # Patients assigned to rooms (some multi-occupancy).
        
        # 1. Create Patients
        for i in range(n_patients):
            w_id = i % n_wards
            r_id = (i // n_wards) % 20 # 20 rooms per ward
            a = Agent(id=i, role='PATIENT', ward_id=w_id, room_id=r_id)
            self.agents.append(a)
            self.wards[w_id].append(a)
            
        # 2. Create HCWs
        for i in range(n_patients, n_patients + n_hcw):
            w_id = i % n_wards # Home ward
            a = Agent(id=i, role='HCW', ward_id=w_id)
            self.agents.append(a)
            
    def get_contacts(self, day_seed):
        """
        Returns a list of potential contact pairs (source_agent, target_agent, transmission_prob).
        Logic:
        - Patients in same room: High risk
        - Patients in same ward: Low risk
        - HCW: Visits home ward + random ward
        """
        rng = np.random.default_rng(day_seed)
        contacts = []
        
        # Pre-group by location for speed
        ward_map = collections.defaultdict(list)
        room_map = collections.defaultdict(list)
        
        for a in self.agents:
            # HCW movement logic
            current_ward = a.ward_id
            if a.role == 'HCW':
                if rng.random() < HCW_CROSS_WARD_PROB:
                    current_ward = rng.integers(0, N_WARDS)
            
            ward_map[current_ward].append(a)
            if a.role == 'PATIENT':
                room_map[(current_ward, a.room_id)].append(a)
        
        # 1. Room Contacts (High Intensity)
        for room_agents in room_map.values():
            if len(room_agents) > 1:
                for i in range(len(room_agents)):
                    for j in range(i+1, len(room_agents)):
                        contacts.append((room_agents[i], room_agents[j], BETA_ROOM))
                        
        # 2. Ward Contacts (Medium Intensity - mixing in hallways/HCW interaction)
        for w_id, agents_in_ward in ward_map.items():
            # Randomly sample pairs to avoid N^2 explosion, assume density dependent
            n_in_ward = len(agents_in_ward)
            # Each person contacts k random others in the ward
            k_contacts = 5 
            for agent in agents_in_ward:
                targets = rng.choice(agents_in_ward, min(k_contacts, n_in_ward), replace=False)
                for target in targets:
                    if agent.id == target.id: continue
                    
                    prob = BETA_WARD
                    # If one is HCW and one is Patient, use specific rate
                    if agent.role != target.role:
                        prob = BETA_HCW_PAT
                        
                    contacts.append((agent, target, prob))
                    
        return contacts

# ==========================================
# 3. PHYLOGENY TRACKER (TSKIT WRAPPER)
# ==========================================

class PhylogenyTracker:
    def __init__(self):
        self.tables = tskit.TableCollection(sequence_length=GENOME_LENGTH)
        self.nodes = self.tables.nodes
        self.edges = self.tables.edges
        self.sites = self.tables.sites
        self.mutations = self.tables.mutations
        
        # Metadata storage
        self.metadata = [] # ID, Type, Date
        
        # Initialize Community Reservoir (The "Spectral Bank")
        # List of lists: self.community_lineages[variant_idx] = [node_ids_current_generation]
        self.community_lineages = []
        
        # Create a deep ancestor to link the lineages and generate diversity
        # Distance = 2 * T * mu. We want distance = COMMUNITY_DIVERSITY_LEVEL * GENOME_LENGTH (approx mutations)
        # But MUTATION_RATE is per site per day.
        # Target mutations = COMMUNITY_DIVERSITY_LEVEL * GENOME_LENGTH
        # Branch length needed = Target mutations / (MUTATION_RATE * GENOME_LENGTH)
        # T_ago = Branch length
        
        target_muts = COMMUNITY_DIVERSITY_LEVEL * GENOME_LENGTH
        muts_per_day = MUTATION_RATE * GENOME_LENGTH
        t_ago = target_muts / muts_per_day if muts_per_day > 0 else 0
        
        # Create Grand MRCA
        # Time is forward. Current is 0. Ancestor is at -t_ago.
        mrca = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=-t_ago)
        
        for i in range(NUM_COMMUNITY_LINEAGES):
            # Create an intermediate root for this lineage at time 0 (or slightly before to be parent)
            # This ensures the lineage has a single common ancestor at time 0 that links to the Grand MRCA
            lineage_root = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=-1e-8)
            self.edges.add_row(parent=mrca, child=lineage_root, left=0, right=GENOME_LENGTH)
            
            # Create initial population for this lineage at time 0
            current_gen = []
            for _ in range(COMMUNITY_POP_SIZE):
                u = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                # Link to Lineage Root (NOT Grand MRCA directly)
                self.edges.add_row(parent=lineage_root, child=u, left=0, right=GENOME_LENGTH)
                current_gen.append(u)
            self.community_lineages.append(current_gen)

    def step_community(self, day):
        """
        Advance the community lineages by one generation (Wright-Fisher).
        """
        rng = np.random.default_rng(day + 999) # distinct seed
        
        for i in range(NUM_COMMUNITY_LINEAGES):
            prev_gen = self.community_lineages[i]
            next_gen = []
            
            # Create new generation
            for _ in range(COMMUNITY_POP_SIZE):
                # Pick parent uniformly at random
                parent = rng.choice(prev_gen)
                
                # Ensure time ordering
                parent_time = self.nodes.time[parent]
                child_time = day
                if child_time <= parent_time:
                    child_time = parent_time + 1e-8
                
                child = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
                self.edges.add_row(parent=parent, child=child, left=0, right=GENOME_LENGTH)
                next_gen.append(child)
            
            self.community_lineages[i] = next_gen

    def add_transmission(self, source_node, time_now):
        """
        A infects B.
        We create a new node for the virus in B.
        We also update the virus in A to reflect evolution up to time_now.
        """
        parent_time = self.nodes.time[source_node]
        
        # 1. Update Source Backbone (Evolution within A)
        current_source_node = source_node
        if time_now > parent_time:
             # Create intermediate node for source at time_now
             current_source_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_now)
             self.edges.add_row(parent=source_node, child=current_source_node, left=0, right=GENOME_LENGTH)
             parent_time = time_now
        
        # 2. Create Child (Virus in B)
        # Child must be slightly younger than parent in forward time
        child_time = time_now
        if child_time <= parent_time:
             child_time = parent_time + 1e-8
             
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
        self.edges.add_row(parent=current_source_node, child=child_node, left=0, right=GENOME_LENGTH)
        
        return child_node, current_source_node

    def add_importation(self, time_now, variant_idx=0):
        """
        Infection comes from community.
        Parent is picked from the current generation of the background lineage.
        """
        # Pick a background lineage
        lineage_idx = variant_idx % len(self.community_lineages)
        current_gen = self.community_lineages[lineage_idx]
        
        # Pick a random individual from the current generation
        # Note: In a real WF model, we might pick from the *previous* generation if we consider
        # the importation happens *during* the day. Let's pick from the current state.
        root = np.random.choice(current_gen)
        
        # Root is at time_now (or slightly before). 
        parent_time = self.nodes.time[root]
        if time_now <= parent_time:
            time_now = parent_time + 1e-8
        
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_now)
        self.edges.add_row(parent=root, child=child_node, left=0, right=GENOME_LENGTH)
        return child_node

    def add_sample(self, infected_node, sample_time):
        """
        Patient is sequenced.
        We create a tip node representing the sample.
        We also update the patient's virus lineage to the sample time.
        """
        parent_time = self.nodes.time[infected_node]
        
        # Update Backbone
        current_node = infected_node
        if sample_time > parent_time:
            current_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=sample_time)
            self.edges.add_row(parent=infected_node, child=current_node, left=0, right=GENOME_LENGTH)
            parent_time = sample_time
            
        # Create Sample Tip
        tip_time = sample_time
        if tip_time <= parent_time:
            tip_time = parent_time + 1e-8
            
        sample_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=tip_time)
        self.edges.add_row(parent=current_node, child=sample_node, left=0, right=GENOME_LENGTH)
        
        return sample_node, current_node
    
    def finalize_tree(self, max_time):
        """
        tskit usually expects time to be 'time ago'. 
        Our simulation ran in forward time (0 -> max).
        We simply invert the times: node_time = max_time - node.birth_time
        """
        times = self.nodes.time
        self.nodes.time = max_time - times
        
        # Sort tables (required by tskit)
        self.tables.sort()
        
        # Add Mutations
        # We simulate mutations on the tree based on branch lengths
        # Manual implementation since tskit.sim_mutations is not available in 0.5.7
        rng = np.random.default_rng(42)
        
        # Map position -> site_id
        site_map = {}
        
        # Generate mutations
        generated_mutations = [] # (pos, node)
        node_times = self.tables.nodes.time
        
        for edge in self.tables.edges:
            parent = edge.parent
            child = edge.child
            left = edge.left
            right = edge.right
            
            t_parent = node_times[parent]
            t_child = node_times[child]
            branch_len = t_parent - t_child
            
            if branch_len < 0: 
                branch_len = 0
                
            span = right - left
            expected_muts = MUTATION_RATE * branch_len * span
            n_muts = rng.poisson(expected_muts)
            
            if n_muts > 0:
                # discrete positions
                positions = rng.integers(int(left), int(right), size=n_muts)
                for pos in positions:
                    generated_mutations.append((pos, child))
                    
        # Sort by position to add sites in order
        generated_mutations.sort(key=lambda x: x[0])
        
        for pos, node in generated_mutations:
            if pos not in site_map:
                site_id = self.tables.sites.add_row(position=pos, ancestral_state="A")
                site_map[pos] = site_id
            
            site_id = site_map[pos]
            self.tables.mutations.add_row(site=site_id, node=node, derived_state="T")
            
        # Sort again to ensure validity
        self.tables.sort()
        
        ts = self.tables.tree_sequence()
        # ts_mutated = tskit.sim_mutations(ts, rate=MUTATION_RATE, random_seed=42)
        
        return ts

# ==========================================
# 4. MAIN SIMULATION LOOP
# ==========================================

def run_simulation():
    hospital = Hospital(N_WARDS, N_PATIENTS, N_HCW)
    tracker = PhylogenyTracker()
    
    # Trackers for SIR curves
    history = [] # Day, S, I, R, Ward...
    
    print(f"Starting Simulation: {N_PATIENTS+N_HCW} Agents, {SIMULATION_DAYS} Days.")
    
    for day in range(SIMULATION_DAYS):
        # 0. Evolve Community
        tracker.step_community(day)

        # 1. Importation Events (Background Spillover)
        # Randomly infect a susceptible person from community
        if np.random.random() < IMPORTATION_DAILY_PROB:
            # Pick a susceptible
            susceptibles = [a for a in hospital.agents if a.status == 'S']
            if susceptibles:
                target = np.random.choice(susceptibles)
                target.status = 'I'
                target.infection_time = day
                target.symptom_time = day + np.random.poisson(MEAN_INCUBATION)
                
                # Assign genetics: Pick a random community variant
                variant = np.random.randint(0, NUM_COMMUNITY_LINEAGES)
                node_id = tracker.add_importation(day, variant)
                target.infected_by_node = node_id
                # print(f"Day {day}: Importation event (Variant {variant}) -> Agent {target.id}")

        # 2. Contact & Transmission
        contacts = hospital.get_contacts(day_seed=day)
        new_infections = []
        
        for p1, p2, prob in contacts:
            # Check S-I pairs
            source, target = None, None
            if p1.status == 'I' and p2.status == 'S':
                source, target = p1, p2
            elif p2.status == 'I' and p1.status == 'S':
                source, target = p2, p1
            
            if source:
                if np.random.random() < prob:
                    # Successful transmission
                    # Avoid double infection in same step
                    if target not in [x[0] for x in new_infections]:
                        new_infections.append((target, source))

        # Apply Infections
        for target, source in new_infections:
            target.status = 'I'
            target.infection_time = day
            target.symptom_time = day + np.random.poisson(MEAN_INCUBATION)
            
            # Genetics: Source Node -> Target Node
            new_node, updated_source_node = tracker.add_transmission(source.infected_by_node, day)
            target.infected_by_node = new_node
            source.infected_by_node = updated_source_node

        # 3. Clinical Progression & Sampling
        for a in hospital.agents:
            if a.status == 'I':
                # Check for sampling
                if not a.is_sampled:
                    # Logic: if symptomatic and probabilistic check passed
                    if day >= a.symptom_time:
                         # Check if we sample this person (once only)
                         if np.random.random() < PROB_DETECT_SEQUENCE:
                             # Lag calculation
                             sample_date = a.symptom_time + np.random.poisson(SAMPLE_LAG_MEAN)
                             if day >= sample_date:
                                 # Take sample
                                 sample_node, updated_node = tracker.add_sample(a.infected_by_node, day)
                                 a.is_sampled = True
                                 a.sample_node = sample_node
                                 a.infected_by_node = updated_node
                                 a.sample_time = day
                
                # Recovery
                if day > a.infection_time + MEAN_RECOVERY:
                    a.status = 'R'

        # 4. Data Logging
        s_count = sum(1 for a in hospital.agents if a.status == 'S')
        i_count = sum(1 for a in hospital.agents if a.status == 'I')
        r_count = sum(1 for a in hospital.agents if a.status == 'R')
        history.append({'day': day, 'S': s_count, 'I': i_count, 'R': r_count})
        
    # ==========================================
    # 5. EXPORT & FINALIZATION
    # ==========================================
    
    # A. SIR Curves
    df_hist = pd.DataFrame(history)
    plt.figure(figsize=(10,6))
    plt.plot(df_hist['day'], df_hist['I'], label='Infected', color='red')
    plt.plot(df_hist['day'], df_hist['R'], label='Recovered', color='green')
    plt.title("Hospital Outbreak SIR Curve")
    plt.xlabel("Day")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("sir_curves.png")
    print("Saved sir_curves.png")

    # B. Phylogeny (Trees)
    ts = tracker.finalize_tree(max_time=SIMULATION_DAYS)
    ts.dump("hospital_outbreak.trees")
    print(f"Saved hospital_outbreak.trees with {ts.num_mutations} mutations.")

    # C. Efficient Mutation Structure (By Patient)
    # We map Sample Node ID -> Agent ID
    sample_node_to_agent = {a.sample_node: a for a in hospital.agents if a.is_sampled}
    
    mut_data = []
    for variant in ts.variants():
        # variant.site.position
        # variant.alleles
        # variant.genotypes (array of allele indices per sample)
        pos = int(variant.site.position)
        alt_allele = variant.alleles[1] if len(variant.alleles) > 1 else "N"
        
        for sample_index, allele_idx in enumerate(variant.genotypes):
            if allele_idx > 0: # Has mutation
                # tskit sample_index maps to node_id
                node_id = ts.samples()[sample_index]
                if node_id in sample_node_to_agent:
                    agent = sample_node_to_agent[node_id]
                    mut_data.append({
                        'agent_id': agent.id,
                        'role': agent.role,
                        'ward': agent.ward_id,
                        'position': pos,
                        'alt': alt_allele
                    })
    
    df_mut = pd.DataFrame(mut_data)
    df_mut.to_csv("mutations_per_patient.csv", index=False)
    print("Saved mutations_per_patient.csv")

    # D. FASTA Output
    # Generate haplotypes
    with open("sampled_sequences.fasta", "w") as f:
        for sample_index, h in enumerate(ts.haplotypes()):
            node_id = ts.samples()[sample_index]
            if node_id in sample_node_to_agent:
                agent = sample_node_to_agent[node_id]
                header = f">Agent_{agent.id}_{agent.role}_Ward{agent.ward_id}_Day{agent.sample_time}"
                f.write(f"{header}\n{h}\n")
    print("Saved sampled_sequences.fasta")

    # E. Save Hospital Node IDs for Analysis
    with open("hospital_node_ids.txt", "w") as f:
        for node_id in sample_node_to_agent.keys():
            f.write(f"{node_id}\n")
    print("Saved hospital_node_ids.txt")

    
    # ==========================================
    # E. VISUALISATION (Split Plots)
    # ==========================================
    print("Generating split phylogenetic trees...")
    
    # Helper to plot tree
    def plot_tree(ts_subset, title, filename, color_func, legend_func=None):
        # Export to Newick
        tree_obj = ts_subset.first()
        node_labels = {n: str(n) for n in ts_subset.samples()}
        newick_str = tree_obj.newick(node_labels=node_labels)
        
        # Parse
        tree = Phylo.read(StringIO(newick_str), "newick")
        
        # Color tips
        for clade in tree.get_terminals():
            if clade.name:
                node_id = int(clade.name)
                # We need to map back to original agent if possible, or use metadata
                # Since we simplified, node IDs changed! 
                # We need to track the mapping. ts.simplify returns a map.
                # Actually, ts.simplify renumbers nodes. 
                # But we can keep the metadata? The current script doesn't use metadata tables heavily.
                # Alternative: Don't simplify. Just prune in Biopython? 
                # Or use the map returned by simplify.
                pass
        
        # WAIT: ts.simplify renumbers nodes. We lose the mapping to 'sample_node_to_agent' keys.
        # We should use the 'map_nodes' argument or just rely on the fact that we can filter the Newick?
        # No, simplify is best for topology.
        # Let's use the fact that we can pass 'keep_input_roots=True' etc? No.
        # We can add metadata to the nodes BEFORE simplifying.
        # Or we can just use Biopython to prune.
        # Given the tree size (~14k nodes), Biopython prune might be slow but maybe acceptable for 1000 tips?
        # Actually, let's try Biopython prune for Hospital (800 tips) and Community (100 tips).
        # It should be fine.
        return tree

    # 1. Parse the FULL tree once
    tree_obj = ts.first()
    node_labels = {n: str(n) for n in ts.samples()}
    newick_str = tree_obj.newick(node_labels=node_labels)
    full_tree = Phylo.read(StringIO(newick_str), "newick")
    
    # Identify Hospital and Community Tips
    hospital_tips = set(sample_node_to_agent.keys())
    all_tips = set(n for n in ts.samples())
    community_tips = list(all_tips - hospital_tips)
    
    # --- PLOT 1: Hospital Cases (Ward) ---
    print("Plotting Hospital Tree (Ward)...")
    
    # Use ts.simplify to get hospital-only tree
    hospital_nodes_list = list(hospital_tips)
    ts_hospital = ts.simplify(samples=hospital_nodes_list)
    
    # Export
    tree_obj_hosp = ts_hospital.first()
    # Map back to original IDs?
    # ts.simplify renumbers nodes 0..N-1.
    # But the order is preserved.
    # So node `i` in ts_hospital corresponds to `hospital_nodes_list[i]`.
    # We need to construct a mapping from NEW node ID -> Original Agent
    
    new_id_to_agent = {}
    for new_id, original_node_id in enumerate(hospital_nodes_list):
        if original_node_id in sample_node_to_agent:
            new_id_to_agent[new_id] = sample_node_to_agent[original_node_id]
            
    # Label tips with NEW IDs
    hosp_labels = {n: str(n) for n in ts_hospital.samples()}
    newick_hosp = tree_obj_hosp.newick(node_labels=hosp_labels)
    tree_ward = Phylo.read(StringIO(newick_hosp), "newick")
            
    # Color by Ward
    ward_colours = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", 
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000080"
    ]
    hcw_colour = "#d62728"
    
    for clade in tree_ward.get_terminals():
        if clade.name:
            new_node_id = int(clade.name)
            if new_node_id in new_id_to_agent:
                agent = new_id_to_agent[new_node_id]
                if agent.role == 'HCW':
                    clade.color = hcw_colour
                else:
                    clade.color = ward_colours[agent.ward_id % 10]
                    
    fig, ax = plt.subplots(figsize=(10, max(5, len(hospital_tips)*0.15)))
    Phylo.draw(tree_ward, axes=ax, do_show=False, show_confidence=False, label_func=lambda x: None)
    plt.title("Hospital Outbreak: Colored by Ward")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("hospital_tree_ward.png", dpi=300)
    plt.close()
    
    # --- PLOT 2: Hospital Cases (Time) ---
    print("Plotting Hospital Tree (Time)...")
    tree_time = copy.deepcopy(tree_ward) # Reuse the structure
    
    # Color by Time (Gradient)
    # Map time 0-60 to colormap
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0, SIMULATION_DAYS)
    
    for clade in tree_time.get_terminals():
        if clade.name:
            new_node_id = int(clade.name)
            if new_node_id in new_id_to_agent:
                agent = new_id_to_agent[new_node_id]
                # Map time to hex color
                rgba = cmap(norm(agent.sample_time))
                # Biopython needs hex or name? It accepts RGB tuples in some versions, but hex is safer.
                # Actually Biopython Color objects are specific. 
                # But we can assign a string hex to clade.color usually if using matplotlib backend?
                # Let's convert rgba to hex
                hex_color = mcolors.to_hex(rgba)
                clade.color = hex_color

    fig, ax = plt.subplots(figsize=(10, max(5, len(hospital_tips)*0.15)))
    Phylo.draw(tree_time, axes=ax, do_show=False, show_confidence=False, label_func=lambda x: None)
    plt.title("Hospital Outbreak: Colored by Time")
    plt.axis("off")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Day", fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("hospital_tree_time.png", dpi=300)
    plt.close()

    # --- PLOT 3: Community Subsample (Time) ---
    print("Plotting Community Tree (Time)...")
    # Subsample 100 community nodes
    if len(community_tips) > 100:
        community_subset = np.random.choice(community_tips, 100, replace=False)
    else:
        community_subset = community_tips
        
    
    # Pruning 14000 tips one by one is SLOW.
    # Better approach for Community: Use ts.simplify!
    # We don't need agent metadata for community nodes (just time).
    # And ts.simplify preserves time.
    
    comm_subset_list = list(community_subset)
    ts_comm = ts.simplify(samples=comm_subset_list)
    
    # Export simplified tree
    tree_obj_comm = ts_comm.first()
    # Node labels will be 0..99. We need to map them to times.
    # ts_comm.nodes.time gives us the time!
    
    # Wait, Phylo newick parser uses node labels. tskit newick uses 1-based or 0-based?
    # tskit newick uses node IDs as labels if we don't provide them? No, it uses 1-based labels by default or something.
    # Let's provide labels.
    comm_labels = {n: str(n) for n in ts_comm.samples()}
    newick_comm = tree_obj_comm.newick(node_labels=comm_labels)
    tree_comm_viz = Phylo.read(StringIO(newick_comm), "newick")
    
    # Color by Time
    node_times = ts_comm.tables.nodes.time
    # Note: ts.simplify might change node indices, but samples 0..N-1 correspond to the input samples in order?
    # No, "The samples in the output tree sequence will be the supplied samples, in the order they were given."
    # So sample 0 in ts_comm is comm_subset_list[0].
    
    for clade in tree_comm_viz.get_terminals():
        if clade.name:
            node_id = int(clade.name)
            # Get time from ts_comm
            # Note: node_id in tree corresponds to node_id in ts_comm
            time = node_times[node_id]
            # Invert time if needed? No, we used forward time in simulation, 
            # but finalize_tree inverted it: node_time = max_time - birth_time.
            # So time 0 = end of sim, time 60 = start.
            # We want to color by "Day" (0..60).
            # Day = SIMULATION_DAYS - time
            day = SIMULATION_DAYS - time
            
            rgba = cmap(norm(day))
            hex_color = mcolors.to_hex(rgba)
            clade.color = hex_color

    fig, ax = plt.subplots(figsize=(10, 8))
    Phylo.draw(tree_comm_viz, axes=ax, do_show=False, show_confidence=False, label_func=lambda x: None)
    plt.title("Community Subsample: Colored by Time")
    plt.axis("off")
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Day", fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("community_tree_time.png", dpi=300)
    plt.close()
    
    print("Saved hospital_tree_ward.png, hospital_tree_time.png, community_tree_time.png")

if __name__ == "__main__":
    run_simulation()