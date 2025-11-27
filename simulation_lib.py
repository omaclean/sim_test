import tskit
import numpy as np
import pandas as pd
import collections
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. CORE SIMULATION CLASSES
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
    
    # Isolation
    is_isolated: bool = False
    isolation_time: float = -1.0

class Hospital:
    def __init__(self, n_wards, n_patients, n_hcw, isolation_capacity=0):
        if n_wards <= 0:
            raise ValueError(f"n_wards must be positive, got {n_wards}")
        if n_patients < 0:
            raise ValueError(f"n_patients must be non-negative, got {n_patients}")
        if n_hcw < 0:
            raise ValueError(f"n_hcw must be non-negative, got {n_hcw}")
            
        self.wards = collections.defaultdict(list)
        self.agents = []
        self.n_wards = n_wards
        
        # Isolation
        self.isolation_capacity = isolation_capacity
        self.n_isolated = 0
        
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
            
    def try_isolate_patient(self, agent, day):
        """
        Attempts to move a patient to isolation.
        Returns True if successful, False if no capacity.
        """
        if agent.is_isolated:
            return True # Already isolated
            
        if self.n_isolated < self.isolation_capacity:
            agent.is_isolated = True
            agent.isolation_time = day
            self.n_isolated += 1
            return True
        return False

    def discharge_patient(self, agent):
        """
        Handles patient discharge/recovery, freeing up isolation spots.
        """
        if agent.is_isolated:
            agent.is_isolated = False
            self.n_isolated -= 1

    def get_contacts(self, day_seed, params):
        """
        Returns a list of potential contact pairs (source_agent, target_agent, transmission_prob).
        """
        rng = np.random.default_rng(day_seed)
        contacts = []
        
        # Pre-group by location for speed
        ward_map = collections.defaultdict(list)
        room_map = collections.defaultdict(list)
        
        # Filter out isolated agents from general mixing
        active_agents = [a for a in self.agents if not a.is_isolated]
        
        for a in active_agents:
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
    def __init__(self, genome_length, mutation_rate, community_diversity_level, num_community_lineages, community_pop_size, burn_in_days=21):
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        
        # Initialize tskit tables
        self.tables = tskit.TableCollection(sequence_length=genome_length)
        self.nodes = self.tables.nodes
        self.edges = self.tables.edges
        self.sites = self.tables.sites
        self.mutations = self.tables.mutations
        
        # Map position -> site_id to reuse sites
        self.site_map = {}
        self.rng = np.random.default_rng(42)
        
        # Caches for efficient traversal
        self.parent_map = {} # child_node -> parent_node
        self.node_mutations = collections.defaultdict(list) # node_id -> [site_ids]
        
        # Metadata storage
        self.metadata = [] 
        
        # --- Community Reservoir Initialization ---
        self.community_lineages = []
        self.num_community_lineages = num_community_lineages
        self.community_pop_size = community_pop_size
        
        target_muts = community_diversity_level * genome_length
        muts_per_day = mutation_rate * genome_length
        
        # Time ago for the Grand MRCA
        t_ago = (target_muts / muts_per_day if muts_per_day > 0 else 0) + burn_in_days
        
        # Create Grand MRCA
        mrca = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=-t_ago)
        
        for i in range(num_community_lineages):
            lineage_start_time = -burn_in_days
            lineage_root = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=lineage_start_time - 1e-8)
            self.edges.add_row(parent=mrca, child=lineage_root, left=0, right=genome_length)
            self.parent_map[lineage_root] = mrca
            self._add_mutations(mrca, lineage_root, 0, genome_length)
            
            # Create initial population
            current_gen = []
            for _ in range(community_pop_size):
                u = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=lineage_start_time)
                self.edges.add_row(parent=lineage_root, child=u, left=0, right=genome_length)
                self.parent_map[u] = lineage_root
                self._add_mutations(lineage_root, u, 0, genome_length)
                current_gen.append(u)
            self.community_lineages.append(current_gen)
            
        # --- Burn-in Phase ---
        print(f"Evolving community lineages for {burn_in_days} days of burn-in...")
        for d in range(-burn_in_days + 1, 1):
             self.step_community(d)

    def _add_mutations(self, parent, child, left, right):
        """
        Simulate mutations on the branch from parent to child.
        """
        t_parent = self.nodes.time[parent]
        t_child = self.nodes.time[child]
        branch_len = t_child - t_parent
        if branch_len < 0: branch_len = 0
            
        span = right - left
        expected_muts = self.mutation_rate * branch_len * span
        n_muts = self.rng.poisson(expected_muts)
        
        if n_muts > 0:
            positions = self.rng.integers(int(left), int(right), size=n_muts)
            positions.sort()
            
            for pos in positions:
                if pos not in self.site_map:
                    site_id = self.sites.add_row(position=pos, ancestral_state="A")
                    self.site_map[pos] = site_id
                
                site_id = self.site_map[pos]
                self.mutations.add_row(site=site_id, node=child, derived_state="T")
                self.node_mutations[child].append(site_id)

    def step_community(self, day):
        selection_rng = np.random.default_rng(day + 999) 
        for i in range(self.num_community_lineages):
            prev_gen = self.community_lineages[i]
            next_gen = []
            for _ in range(self.community_pop_size):
                parent = selection_rng.choice(prev_gen)
                parent_time = self.nodes.time[parent]
                child_time = day
                if child_time <= parent_time: child_time = parent_time + 1e-8
                
                child = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
                self.edges.add_row(parent=parent, child=child, left=0, right=self.genome_length)
                self.parent_map[child] = parent
                self._add_mutations(parent, child, 0, self.genome_length)
                next_gen.append(child)
            self.community_lineages[i] = next_gen

    def add_transmission(self, source_node, time_now):
        parent_time = self.nodes.time[source_node]
        
        # 1. Update Source Backbone
        current_source_node = source_node
        if time_now > parent_time:
             current_source_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_now)
             self.edges.add_row(parent=source_node, child=current_source_node, left=0, right=self.genome_length)
             self.parent_map[current_source_node] = source_node
             self._add_mutations(source_node, current_source_node, 0, self.genome_length)
             parent_time = time_now
        
        # 2. Create Child
        child_time = time_now
        if child_time <= parent_time: child_time = parent_time + 1e-8
             
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
        self.edges.add_row(parent=current_source_node, child=child_node, left=0, right=self.genome_length)
        self.parent_map[child_node] = current_source_node
        self._add_mutations(current_source_node, child_node, 0, self.genome_length)
        
        return child_node, current_source_node

    def add_importation(self, time_now, variant_idx=0):
        lineage_idx = variant_idx % len(self.community_lineages)
        current_gen = self.community_lineages[lineage_idx]
        root = np.random.choice(current_gen)
        
        parent_time = self.nodes.time[root]
        if time_now <= parent_time: time_now = parent_time + 1e-8
        
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_now)
        self.edges.add_row(parent=root, child=child_node, left=0, right=self.genome_length)
        self.parent_map[child_node] = root
        self._add_mutations(root, child_node, 0, self.genome_length)
        return child_node

    def add_sample(self, infected_node, sample_time):
        parent_time = self.nodes.time[infected_node]
        
        # 1. Update Backbone
        current_node = infected_node
        if sample_time > parent_time:
            current_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=sample_time)
            self.edges.add_row(parent=infected_node, child=current_node, left=0, right=self.genome_length)
            self.parent_map[current_node] = infected_node
            self._add_mutations(infected_node, current_node, 0, self.genome_length)
            parent_time = sample_time
            
        # 2. Create Sample Tip
        tip_time = sample_time
        if tip_time <= parent_time: tip_time = parent_time + 1e-8
            
        sample_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=tip_time)
        self.edges.add_row(parent=current_node, child=sample_node, left=0, right=self.genome_length)
        self.parent_map[sample_node] = current_node
        self._add_mutations(current_node, sample_node, 0, self.genome_length)
        
        return sample_node, current_node
    
    def finalize_tree(self, max_time):
        times = self.nodes.time
        self.nodes.time = max_time - times
        self.tables.sort()
        ts = self.tables.tree_sequence()
        return ts

    def get_pairwise_distance(self, node_a, node_b):
        muts_a = self._get_mutations_on_lineage(node_a)
        muts_b = self._get_mutations_on_lineage(node_b)
        diff = muts_a.symmetric_difference(muts_b)
        return len(diff)

    def _get_mutations_on_lineage(self, node_id):
        mutations = set()
        current = node_id
        while True:
            # Add mutations that occurred on the branch leading TO current
            if current in self.node_mutations:
                mutations.update(self.node_mutations[current])
            
            # Move to parent
            if current in self.parent_map:
                current = self.parent_map[current]
            else:
                break # Reached root (or node with no recorded parent in this map)
        return mutations

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
    Saves a CSV listing mutations for each sampled agent.
    """
    # 1. Map Sample Nodes to Agents
    sample_node_to_agent = {a.sample_node: a for a in hospital.agents if a.is_sampled}
    
    data = []
    
    # 2. Iterate over variants (mutations)
    # ts.variants() yields (site, alleles, genotypes)
    for variant in ts.variants():
        pos = int(variant.site.position)
        alt_allele = variant.alleles[1] if len(variant.alleles) > 1 else "T"
        
        # Check which samples have the mutation (genotype 1)
        # variant.genotypes is an array of 0s and 1s corresponding to samples
        for sample_index, genotype in enumerate(variant.genotypes):
            if genotype == 1:
                # Get the node ID for this sample index
                node_id = ts.samples()[sample_index]
                
                if node_id in sample_node_to_agent:
                    agent = sample_node_to_agent[node_id]
                    data.append({
                        'agent_id': agent.id,
                        'role': agent.role,
                        'ward': agent.ward_id,
                        'position': pos,
                        'alt': alt_allele
                    })
                    
    df = pd.DataFrame(data)
    if df.empty:
        # Create empty DF with columns to avoid errors
        df = pd.DataFrame(columns=['agent_id', 'role', 'ward', 'position', 'alt'])
        
    df.to_csv(os.path.join(output_dir, "mutations_per_patient.csv"), index=False)

def save_fasta(ts, hospital, output_dir):
    """
    Generates FASTA sequences for all sampled agents.
    
    This function converts the efficient TreeSequence storage into actual string sequences.
    ts.haplotypes() reconstructs the full genome for each sample by traversing the tree
    and applying mutations from the root down to the tip.
    """
    sample_node_to_agent = {a.sample_node: a for a in hospital.agents if a.is_sampled}
    
    with open(os.path.join(output_dir, "sampled_sequences.fasta"), "w") as f:
        # ts.haplotypes() returns an iterator of sequence strings
        # corresponding to the samples in ts.samples() order
        for sample_index, h in enumerate(ts.haplotypes()):
            node_id = ts.samples()[sample_index]
            
            if node_id in sample_node_to_agent:
                agent = sample_node_to_agent[node_id]
                # Header format: >Agent_ID_Role_Ward_Day
                header = f">Agent_{agent.id}_{agent.role}_Ward{agent.ward_id}_Day{agent.sample_time}"
                f.write(f"{header}\n{h}\n")

def save_node_ids(hospital, output_dir):
    with open(os.path.join(output_dir, "hospital_node_ids.txt"), "w") as f:
        for a in hospital.agents:
            if a.is_sampled:
                f.write(f"{a.id},{a.sample_node}\n")

def generate_plots(ts, hospital, output_dir, max_time):
    # This requires SVG/Graphviz support which might be tricky in headless env.
    # We will try to generate a simple SVG using tskit's draw_svg if possible,
    # or just skip if dependencies missing.
    try:
        # Map node IDs to colors based on Ward
        # We need to build a style string for tskit
        
        # Use a colormap that supports many categories (wards)
        cmap = plt.get_cmap('tab20') # 20 distinct colors
        # Or 'nipy_spectral' for even more
        if hospital.n_wards > 20:
            cmap = plt.get_cmap('nipy_spectral')
            
        style = ""
        
        # Helper to get hex color
        def get_hex(val, max_val):
            # val is 0..max_val-1
            # Normalize to 0..1
            norm = val / max(1, max_val - 1)
            rgba = cmap(norm)
            return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"

        for a in hospital.agents:
            if a.is_sampled:
                color = get_hex(a.ward_id, hospital.n_wards)
                style += f".node.n{a.sample_node} > .sym {{fill: {color}}}\n"
                
        # Draw the tree (subset if too large?)
        # If tree is huge, drawing is slow. Let's draw only if < 100 samples?
        if ts.num_samples < 200:
            svg_string = ts.draw_svg(size=(1000, 600), style=style)
            with open(os.path.join(output_dir, "hospital_tree_ward.svg"), "w") as f:
                f.write(svg_string)
    except Exception as e:
        print(f"Could not generate tree plot: {e}")
