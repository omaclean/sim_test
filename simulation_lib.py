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
    def __init__(self, genome_length, mutation_rate, community_diversity_level, num_community_lineages, community_pop_size):
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.tables = tskit.TableCollection(sequence_length=genome_length)
        self.nodes = self.tables.nodes
        self.edges = self.tables.edges
        self.sites = self.tables.sites
        self.mutations = self.tables.mutations
        
        # Metadata storage
        self.metadata = [] # ID, Type, Date
        
        # Initialize Community Reservoir (The "Spectral Bank")
        # List of lists: self.community_lineages[variant_idx] = [node_ids_current_generation]
        self.community_lineages = []
        self.num_community_lineages = num_community_lineages
        self.community_pop_size = community_pop_size
        
        target_muts = community_diversity_level * genome_length
        muts_per_day = mutation_rate * genome_length
        t_ago = target_muts / muts_per_day if muts_per_day > 0 else 0
        
        # Create Grand MRCA
        # Time is forward. Current is 0. Ancestor is at -t_ago.
        mrca = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=-t_ago)
        
        for i in range(num_community_lineages):
            # Create an intermediate root for this lineage at time 0 (or slightly before to be parent)
            lineage_root = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=-1e-8)
            self.edges.add_row(parent=mrca, child=lineage_root, left=0, right=genome_length)
            
            # Create initial population for this lineage at time 0
            current_gen = []
            for _ in range(community_pop_size):
                u = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                # Link to Lineage Root (NOT Grand MRCA directly)
                self.edges.add_row(parent=lineage_root, child=u, left=0, right=genome_length)
                current_gen.append(u)
            self.community_lineages.append(current_gen)

    def step_community(self, day):
        """
        Advance the community lineages by one generation (Wright-Fisher).
        """
        rng = np.random.default_rng(day + 999) # distinct seed
        
        for i in range(self.num_community_lineages):
            prev_gen = self.community_lineages[i]
            next_gen = []
            
            # Create new generation
            for _ in range(self.community_pop_size):
                # Pick parent uniformly at random
                parent = rng.choice(prev_gen)
                
                # Ensure time ordering
                parent_time = self.nodes.time[parent]
                child_time = day
                if child_time <= parent_time:
                    child_time = parent_time + 1e-8
                
                child = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
                self.edges.add_row(parent=parent, child=child, left=0, right=self.genome_length)
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
             self.edges.add_row(parent=source_node, child=current_source_node, left=0, right=self.genome_length)
             parent_time = time_now
        
        # 2. Create Child (Virus in B)
        # Child must be slightly younger than parent in forward time
        child_time = time_now
        if child_time <= parent_time:
             child_time = parent_time + 1e-8
             
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
        self.edges.add_row(parent=current_source_node, child=child_node, left=0, right=self.genome_length)
        
        return child_node, current_source_node

    def add_importation(self, time_now, variant_idx=0):
        """
        Infection comes from community.
        Parent is picked from the current generation of the background lineage.
        """
        # Pick a background lineage
        lineage_idx = variant_idx % len(self.community_lineages)
        current_gen = self.community_lineages[lineage_idx]
        
        root = np.random.choice(current_gen)
        
        # Root is at time_now (or slightly before). 
        parent_time = self.nodes.time[root]
        if time_now <= parent_time:
            time_now = parent_time + 1e-8
        
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_now)
        self.edges.add_row(parent=root, child=child_node, left=0, right=self.genome_length)
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
            self.edges.add_row(parent=infected_node, child=current_node, left=0, right=self.genome_length)
            parent_time = sample_time
            
        # Create Sample Tip
        tip_time = sample_time
        if tip_time <= parent_time:
            tip_time = parent_time + 1e-8
            
        sample_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=tip_time)
        self.edges.add_row(parent=current_node, child=sample_node, left=0, right=self.genome_length)
        
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
            expected_muts = self.mutation_rate * branch_len * span
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
        return ts

# ==========================================
# 3. HELPER FUNCTIONS (PLOTTING & EXPORT)
# ==========================================

def save_sir_curves(history, output_dir):
    df_hist = pd.DataFrame(history)
    plt.figure(figsize=(10,6))
    plt.plot(df_hist['day'], df_hist['I'], label='Infected', color='red')
    plt.plot(df_hist['day'], df_hist['R'], label='Recovered', color='green')
    plt.title("Hospital Outbreak SIR Curve")
    plt.xlabel("Day")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "sir_curves.png"))
    plt.close()

def save_mutations_csv(ts, hospital, output_dir):
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
