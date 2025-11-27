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
import os

# ==========================================
# 1. DATA STRUCTURES & AGENTS
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
        if n_wards <= 0:
            raise ValueError(f"n_wards must be positive, got {n_wards}")
        if n_patients < 0:
            raise ValueError(f"n_patients must be non-negative, got {n_patients}")
        if n_hcw < 0:
            raise ValueError(f"n_hcw must be non-negative, got {n_hcw}")
            
        self.wards = collections.defaultdict(list)
        self.agents = []
        self.n_wards = n_wards
        
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
            
    def get_contacts(self, day_seed, params):
        """
        Returns a list of potential contact pairs (source_agent, target_agent, transmission_prob).
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
                if rng.random() < params['HCW_CROSS_WARD_PROB']:
                    current_ward = rng.integers(0, self.n_wards)
            
            ward_map[current_ward].append(a)
            if a.role == 'PATIENT':
                room_map[(current_ward, a.room_id)].append(a)
        
        # 1. Room Contacts (High Intensity)
        for room_agents in room_map.values():
            if len(room_agents) > 1:
                for i in range(len(room_agents)):
                    for j in range(i+1, len(room_agents)):
                        contacts.append((room_agents[i], room_agents[j], params['BETA_ROOM']))
                        
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
                    
                    prob = params['BETA_WARD']
                    # If one is HCW and one is Patient, use specific rate
                    if agent.role != target.role:
                        prob = params['BETA_HCW_PAT']
                        
                    contacts.append((agent, target, prob))
                    
        return contacts

# ==========================================
# 2. PHYLOGENY TRACKER (TSKIT WRAPPER)
# ==========================================

class PhylogenyTracker:
    """
    Manages the viral phylogeny using tskit.
    
    This class simulates the evolution of the virus across the transmission network.
    It uses a 'forward-time' simulation approach where nodes are added as infections occur.
    
    Key Concepts:
    1. **Forward Time**: We simulate from Day 0 to Day N. tskit nodes are created with 'birth time' = current day.
       However, standard tskit/coalescent theory uses 'time ago' (0 = present). 
       We handle this conversion in `finalize_tree`.
       
    2. **Within-Host Evolution**: To avoid 'star phylogenies' (where all secondary infections branch from the 
       primary infection time), we model a viral backbone within each host.
       - When a host transmits, we update their viral lineage to the current time (creating an intermediate node).
       - The new infection branches from this updated node.
       - This ensures that serial infections share the evolutionary history that occurred within the host.
       
    3. **Community Reservoir**: We maintain a pool of background lineages to simulate importations of distinct variants.
    """
    def __init__(self, genome_length, mutation_rate, community_diversity_level, num_community_lineages, community_pop_size, burn_in_days=21):
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        
        # Initialize tskit tables
        # TableCollection is the mutable structure used to build the tree sequence.
        self.tables = tskit.TableCollection(sequence_length=genome_length)
        self.nodes = self.tables.nodes
        self.edges = self.tables.edges
        self.sites = self.tables.sites
        self.mutations = self.tables.mutations
        
        # Metadata storage (optional, for tracking extra info if needed)
        self.metadata = [] 
        
        # --- Community Reservoir Initialization ---
        # We create a "spectral bank" of lineages to represent the diversity circulating in the community.
        # This allows importations to introduce genetically distinct variants (e.g., Delta vs Omicron).
        
        self.community_lineages = []
        self.num_community_lineages = num_community_lineages
        self.community_pop_size = community_pop_size
        
        # Calculate the time depth needed to generate the desired genetic distance
        # Distance = 2 * T * mu * L
        # We want average distance = COMMUNITY_DIVERSITY_LEVEL * GENOME_LENGTH
        target_muts = community_diversity_level * genome_length
        muts_per_day = mutation_rate * genome_length
        
        # Time ago for the Grand MRCA (Most Recent Common Ancestor) of all community lineages
        # We add burn_in_days to ensure the MRCA is older than the start of the burn-in
        t_ago = (target_muts / muts_per_day if muts_per_day > 0 else 0) + burn_in_days
        
        # Create Grand MRCA
        # This node is the root of the entire simulation.
        # Time is negative relative to start of sim because it existed before the simulation began.
        # flag=NODE_IS_SAMPLE: This tells tskit to treat this node as a "sample" of interest.
        # In many analyses, we only retain the history of "sample" nodes. 
        # Here, we mark everything as a sample to ensure we can trace back to it easily, 
        # although strictly only the tips need to be samples for simplification.
        mrca = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=-t_ago)
        
        for i in range(num_community_lineages):
            # Create an intermediate root for this specific lineage.
            # We start this lineage 'burn_in_days' before Day 0 to allow for some initial diversity ("fuzz")
            # to accumulate before the simulation starts.
            lineage_start_time = -burn_in_days
            
            lineage_root = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=lineage_start_time - 1e-8)
            self.edges.add_row(parent=mrca, child=lineage_root, left=0, right=genome_length)
            
            # Create initial population for this lineage at the start of burn-in
            current_gen = []
            for _ in range(community_pop_size):
                u = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=lineage_start_time)
                # Link to Lineage Root
                self.edges.add_row(parent=lineage_root, child=u, left=0, right=genome_length)
                current_gen.append(u)
            self.community_lineages.append(current_gen)
            
        # --- Burn-in Phase ---
        # Evolve the community lineages from -burn_in_days up to Day 0.
        # This ensures that at Day 0, the individuals in a lineage are not identical clones
        # but have some "fuzz" (genetic variation) consistent with their population size.
        print(f"Evolving community lineages for {burn_in_days} days of burn-in...")
        for d in range(-burn_in_days + 1, 1): # Evolve up to Day 0
             self.step_community(d)

    def step_community(self, day):
        """
        Advance the community lineages by one generation using a Wright-Fisher model.
        
        This keeps the background diversity alive and evolving throughout the simulation.
        Each generation replaces the previous one.
        """
        rng = np.random.default_rng(day + 999) # distinct seed
        
        for i in range(self.num_community_lineages):
            prev_gen = self.community_lineages[i]
            next_gen = []
            
            # Create new generation of size N
            for _ in range(self.community_pop_size):
                # 1. Pick parent uniformly at random from previous generation
                parent = rng.choice(prev_gen)
                
                # 2. Ensure strict time ordering for tskit
                # Child time must be > Parent time.
                parent_time = self.nodes.time[parent]
                child_time = day
                if child_time <= parent_time:
                    child_time = parent_time + 1e-8
                
                # 3. Add new node and edge
                child = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
                self.edges.add_row(parent=parent, child=child, left=0, right=self.genome_length)
                next_gen.append(child)
            
            self.community_lineages[i] = next_gen

    def add_transmission(self, source_node, time_now):
        """
        Record a transmission event: A infects B.
        
        To model within-host evolution correctly:
        1. We update the source (A) by creating a new node at `time_now` that descends from their previous node.
           This represents the virus evolving within A up to the moment of transmission.
        2. We create a new node for the recipient (B) that descends from A's updated node.
        
        Returns:
            child_node: The node ID for the virus in the recipient (B).
            current_source_node: The updated node ID for the virus in the source (A).
        """
        parent_time = self.nodes.time[source_node]
        
        # 1. Update Source Backbone (Evolution within A)
        # If time has passed since the source was last updated (infected or transmitted),
        # we add a node to represent their current viral state.
        current_source_node = source_node
        if time_now > parent_time:
             current_source_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_now)
             self.edges.add_row(parent=source_node, child=current_source_node, left=0, right=self.genome_length)
             parent_time = time_now
        
        # 2. Create Child (Virus in B)
        # The child node represents the start of infection in B.
        # It must be slightly younger than the parent to satisfy tskit's constraints.
        child_time = time_now
        if child_time <= parent_time:
             child_time = parent_time + 1e-8
             
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
        self.edges.add_row(parent=current_source_node, child=child_node, left=0, right=self.genome_length)
        
        return child_node, current_source_node

    def add_importation(self, time_now, variant_idx=0):
        """
        Record an importation event: Community -> Hospital.
        
        We pick a random individual from the specified community lineage to be the source.
        """
        # Pick the background lineage (e.g., Variant A vs Variant B)
        lineage_idx = variant_idx % len(self.community_lineages)
        current_gen = self.community_lineages[lineage_idx]
        
        # Pick a random source from the current community generation
        root = np.random.choice(current_gen)
        
        # Ensure time ordering
        parent_time = self.nodes.time[root]
        if time_now <= parent_time:
            time_now = parent_time + 1e-8
        
        # Create the new infection node in the hospital
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_now)
        self.edges.add_row(parent=root, child=child_node, left=0, right=self.genome_length)
        return child_node

    def add_sample(self, infected_node, sample_time):
        """
        Record a sampling event: Patient -> Sequence.
        
        Similar to transmission, we update the patient's viral lineage to the time of sampling
        before branching off the sample node. This ensures the sample reflects evolution
        up to the sampling date.
        
        Returns:
            sample_node: The node ID of the sample (tip of the tree).
            current_node: The updated node ID for the patient.
        """
        parent_time = self.nodes.time[infected_node]
        
        # 1. Update Backbone (Evolution within Patient)
        current_node = infected_node
        if sample_time > parent_time:
            current_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=sample_time)
            self.edges.add_row(parent=infected_node, child=current_node, left=0, right=self.genome_length)
            parent_time = sample_time
            
        # 2. Create Sample Tip
        tip_time = sample_time
        if tip_time <= parent_time:
            tip_time = parent_time + 1e-8
            
        sample_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=tip_time)
        self.edges.add_row(parent=current_node, child=sample_node, left=0, right=self.genome_length)
        
        return sample_node, current_node
    
    def finalize_tree(self, max_time):
        """
        Convert the recorded nodes and edges into a final tskit TreeSequence with mutations.
        
        Steps:
        1. **Time Inversion**: Convert 'forward time' (0=start) to 'time ago' (0=end).
           tskit expects roots to be at high time values and tips at low time values (0).
           
        2. **Mutation Generation**: Simulate mutations along the branches.
           Since tskit's `sim_mutations` might not be available or flexible enough for our specific
           needs in older versions, we manually place mutations based on branch length.
           - Expected mutations = rate * branch_length * genome_length
           - Number of mutations ~ Poisson(Expected)
           - Positions are chosen uniformly at random.
           
        Returns:
            ts: The final TreeSequence object.
        """
        # 1. Invert Time
        # Current state: time 0 = start, time max = end.
        # Target state: time max = start (past), time 0 = end (present).
        times = self.nodes.time
        self.nodes.time = max_time - times
        
        # Sort tables (required by tskit after modifying time/topology)
        self.tables.sort()
        
        # 2. Generate Mutations
        rng = np.random.default_rng(42)
        
        # Map position -> site_id to reuse sites if multiple mutations hit the same spot
        site_map = {}
        
        generated_mutations = [] # List of (position, node_id)
        node_times = self.tables.nodes.time
        
        # Iterate over every branch (edge) in the tree
        for edge in self.tables.edges:
            parent = edge.parent
            child = edge.child
            
            # Calculate branch length in time units (days)
            t_parent = node_times[parent]
            t_child = node_times[child]
            branch_len = t_parent - t_child
            
            if branch_len < 0: 
                branch_len = 0
                
            # Calculate expected number of mutations on this branch
            # rate is per site per day
            span = edge.right - edge.left
            expected_muts = self.mutation_rate * branch_len * span
            
            # Sample number of mutations from Poisson distribution
            n_muts = rng.poisson(expected_muts)
            
            if n_muts > 0:
                # Choose random positions for these mutations
                positions = rng.integers(int(edge.left), int(edge.right), size=n_muts)
                for pos in positions:
                    generated_mutations.append((pos, child))
                    
        # Sort mutations by position (required by tskit)
        generated_mutations.sort(key=lambda x: x[0])
        
        # Add mutations to the table
        for pos, node in generated_mutations:
            if pos not in site_map:
                # Create a site if it doesn't exist
                site_id = self.tables.sites.add_row(position=pos, ancestral_state="A")
                site_map[pos] = site_id
            
            site_id = site_map[pos]
            # Add mutation at this site on the child node
            # We assume a simple A->T model for simplicity here
            self.tables.mutations.add_row(site=site_id, node=node, derived_state="T")
            
        # Sort tables again to ensure mutation validity
        self.tables.sort()
        
        # Build the immutable TreeSequence
        ts = self.tables.tree_sequence()
        return ts

# ==========================================
# 3. HELPER FUNCTIONS (PLOTTING & EXPORT)
# ==========================================

def save_sir_curves(history, output_dir):
    if not history:
        # Create empty dataframe with expected columns
        df_hist = pd.DataFrame(columns=['day', 'S', 'I', 'R', 'S_pat', 'I_pat', 'R_pat', 'S_hcw', 'I_hcw', 'R_hcw'])
    else:
        df_hist = pd.DataFrame(history)
        
    plt.figure(figsize=(10,6))
    if not df_hist.empty:
        plt.plot(df_hist['day'], df_hist['S'], label='Susceptible', color='blue')
        plt.plot(df_hist['day'], df_hist['I'], label='Infected', color='red')
        plt.plot(df_hist['day'], df_hist['R'], label='Recovered', color='green')
    plt.title("Hospital Outbreak SIR Curve")
    plt.xlabel("Day")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "sir_curves.png"))
    plt.close()

def save_recovered_split(history, output_dir):
    """
    Plots the percentage of Recovered individuals, split by HCW and Patients.
    """
    if not history:
        return
        
    df_hist = pd.DataFrame(history)
    if df_hist.empty:
        return

    # Calculate percentages
    # We need total counts for normalization. 
    # Assuming constant population size for simplicity, or we can sum S+I+R at each step.
    
    # Total Patients = S_pat + I_pat + R_pat
    # Total HCW = S_hcw + I_hcw + R_hcw
    
    df_hist['Total_Pat'] = df_hist['S_pat'] + df_hist['I_pat'] + df_hist['R_pat']
    df_hist['Total_HCW'] = df_hist['S_hcw'] + df_hist['I_hcw'] + df_hist['R_hcw']
    
    # Avoid division by zero
    df_hist['Pct_R_Pat'] = np.where(df_hist['Total_Pat'] > 0, 100 * df_hist['R_pat'] / df_hist['Total_Pat'], 0)
    df_hist['Pct_R_HCW'] = np.where(df_hist['Total_HCW'] > 0, 100 * df_hist['R_hcw'] / df_hist['Total_HCW'], 0)
    
    plt.figure(figsize=(10,6))
    plt.plot(df_hist['day'], df_hist['Pct_R_Pat'], label='Patients (Recovered %)', color='orange')
    plt.plot(df_hist['day'], df_hist['Pct_R_HCW'], label='HCWs (Recovered %)', color='purple')
    plt.title("Percentage Recovered by Role")
    plt.xlabel("Day")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, "recovered_split.png"))
    plt.close()

def save_mutations_csv(ts, hospital, output_dir):
    """
    Extracts mutations from the TreeSequence and saves them to a CSV.
    
    This is where the 'tree' representation is converted into 'genotype' data.
    The TreeSequence contains the full evolutionary history and mutations on branches.
    We iterate through these mutations to find which samples (agents) carry them.
    """
    sample_node_to_agent = {a.sample_node: a for a in hospital.agents if a.is_sampled}
    mut_data = []
    for variant in ts.variants():
        pos = int(variant.site.position)
        alt_allele = variant.alleles[1] if len(variant.alleles) > 1 else "N"
        
        for sample_index, allele_idx in enumerate(variant.genotypes):
            if allele_idx > 0: # Has mutation
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
    if df_mut.empty:
        df_mut = pd.DataFrame(columns=['agent_id', 'role', 'ward', 'position', 'alt'])
    df_mut.to_csv(os.path.join(output_dir, "mutations_per_patient.csv"), index=False)

def save_fasta(ts, hospital, output_dir):
    """
    Generates FASTA sequences for all sampled agents.
    
    This function converts the efficient TreeSequence storage into actual string sequences.
    ts.haplotypes() reconstructs the full genome for each sample by traversing the tree
    and applying mutations from the root down to the tip.
    """
    sample_node_to_agent = {a.sample_node: a for a in hospital.agents if a.is_sampled}
    with open(os.path.join(output_dir, "sampled_sequences.fasta"), "w") as f:
        for sample_index, h in enumerate(ts.haplotypes()):
            node_id = ts.samples()[sample_index]
            if node_id in sample_node_to_agent:
                agent = sample_node_to_agent[node_id]
                header = f">Agent_{agent.id}_{agent.role}_Ward{agent.ward_id}_Day{agent.sample_time}"
                f.write(f"{header}\n{h}\n")

def save_node_ids(hospital, output_dir):
    sample_node_to_agent = {a.sample_node: a for a in hospital.agents if a.is_sampled}
    with open(os.path.join(output_dir, "hospital_node_ids.txt"), "w") as f:
        for node_id in sample_node_to_agent.keys():
            f.write(f"{node_id}\n")

def generate_plots(ts, hospital, output_dir, simulation_days):
    sample_node_to_agent = {a.sample_node: a for a in hospital.agents if a.is_sampled}
    
    # 1. Parse the FULL tree once
    tree_obj = ts.first()
    node_labels = {n: str(n) for n in ts.samples()}
    newick_str = tree_obj.newick(node_labels=node_labels)
    # full_tree = Phylo.read(StringIO(newick_str), "newick") # Unused?
    
    hospital_tips = set(sample_node_to_agent.keys())
    all_tips = set(n for n in ts.samples())
    community_tips = list(all_tips - hospital_tips)
    
    # --- PLOT 1: Hospital Cases (Ward) ---
    hospital_nodes_list = list(hospital_tips)
    if not hospital_nodes_list:
        return # No hospital cases to plot
        
    ts_hospital = ts.simplify(samples=hospital_nodes_list)
    
    new_id_to_agent = {}
    for new_id, original_node_id in enumerate(hospital_nodes_list):
        if original_node_id in sample_node_to_agent:
            new_id_to_agent[new_id] = sample_node_to_agent[original_node_id]
            
    hosp_labels = {n: str(n) for n in ts_hospital.samples()}
    tree_obj_hosp = ts_hospital.first()
    newick_hosp = tree_obj_hosp.newick(node_labels=hosp_labels)
    tree_ward = Phylo.read(StringIO(newick_hosp), "newick")
            
    # Dynamic Ward Colors
    # Use a colormap that can handle any number of wards
    n_wards = hospital.n_wards
    cmap_wards = plt.get_cmap("tab20" if n_wards <= 20 else "nipy_spectral")
    
    # Pre-calculate hex codes for each ward
    ward_hex_colours = []
    for i in range(n_wards):
        # Normalize index for colormap
        if n_wards <= 20:
            rgba = cmap_wards(i)
        else:
            rgba = cmap_wards(i / n_wards)
        ward_hex_colours.append(mcolors.to_hex(rgba))
        
    hcw_colour = "#d62728"
    
    for clade in tree_ward.get_terminals():
        if clade.name:
            new_node_id = int(clade.name)
            if new_node_id in new_id_to_agent:
                agent = new_id_to_agent[new_node_id]
                if agent.role == 'HCW':
                    clade.color = hcw_colour
                else:
                    clade.color = ward_hex_colours[agent.ward_id % n_wards]
                    
    fig, ax = plt.subplots(figsize=(10, max(5, len(hospital_tips)*0.15)))
    Phylo.draw(tree_ward, axes=ax, do_show=False, show_confidence=False, label_func=lambda x: None)
    plt.title("Hospital Outbreak: Colored by Ward")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hospital_tree_ward.png"), dpi=300)
    plt.close()
    
    # --- PLOT 2: Hospital Cases (Time) ---
    tree_time = copy.deepcopy(tree_ward)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0, simulation_days)
    
    for clade in tree_time.get_terminals():
        if clade.name:
            new_node_id = int(clade.name)
            if new_node_id in new_id_to_agent:
                agent = new_id_to_agent[new_node_id]
                rgba = cmap(norm(agent.sample_time))
                hex_color = mcolors.to_hex(rgba)
                clade.color = hex_color

    fig, ax = plt.subplots(figsize=(10, max(5, len(hospital_tips)*0.15)))
    Phylo.draw(tree_time, axes=ax, do_show=False, show_confidence=False, label_func=lambda x: None)
    plt.title("Hospital Outbreak: Colored by Time")
    plt.axis("off")
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Day", fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hospital_tree_time.png"), dpi=300)
    plt.close()

    # --- PLOT 3: Community Subsample (Time) ---
    if len(community_tips) > 100:
        community_subset = np.random.choice(community_tips, 100, replace=False)
    else:
        community_subset = community_tips
        
    if len(community_subset) > 0:
        comm_subset_list = list(community_subset)
        ts_comm = ts.simplify(samples=comm_subset_list)
        
        tree_obj_comm = ts_comm.first()
        comm_labels = {n: str(n) for n in ts_comm.samples()}
        newick_comm = tree_obj_comm.newick(node_labels=comm_labels)
        tree_comm_viz = Phylo.read(StringIO(newick_comm), "newick")
        
        node_times = ts_comm.tables.nodes.time
        
        for clade in tree_comm_viz.get_terminals():
            if clade.name:
                node_id = int(clade.name)
                time = node_times[node_id]
                day = simulation_days - time
                
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
        plt.savefig(os.path.join(output_dir, "community_tree_time.png"), dpi=300)
        plt.close()
