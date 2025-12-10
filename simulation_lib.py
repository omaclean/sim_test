import tskit
import numpy as np
import pandas as pd
import collections
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import matplotlib.pyplot as plt
from Bio import Phylo
from io import StringIO
from matplotlib import colors as mcolors
import copy


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
    scheduled_detection_time: Optional[float] = None
    
    # Genetics
    infected_by_node: int = -1 # The tskit node ID of the virus inside this person
    
    # Sampling / Detection
    is_detected: bool = False # Clinical detection (symptoms + test)
    
    # Sampling
    is_sampled: bool = False
    sample_time: float = -1.0
    sample_node: int = -1 # The tskit node ID of the sample taken from this person
    
    # Isolation
    is_isolated: bool = False
    isolation_time: float = -1.0

# ... (Hospital class remains unchanged) ...

def save_daily_fastas(ts, daily_census, output_dir, max_days):
    """
    Saves daily FASTA files containing sequences of ALL infected agents on each day.
    
    Args:
        ts: TreeSequence
        daily_census: List of dicts or tuples with (day, agent_id, role, ward, node_id, is_detected)
        output_dir: Output directory
        max_days: Simulation duration
    """
    daily_dir = os.path.join(output_dir, "daily_sequences")
    os.makedirs(daily_dir, exist_ok=True)
    
    # Group census by day
    census_by_day = collections.defaultdict(list)
    for record in daily_census:
        census_by_day[record['day']].append(record)
        
    # We need to extract sequences for these specific nodes.
    # ts.haplotypes() iterates over ALL samples in the tree.
    # We need to map node_id -> sequence.
    
    # Efficient way:
    # 1. Map node_id -> index in ts.samples()
    # 2. Iterate haplotypes and store relevant ones?
    # Or just iterate haplotypes once and populate a map node_id -> sequence
    
    node_to_seq = {}
    samples_list = ts.samples()
    
    # Create a map of sample_node_id -> index to quickly find which haplotype belongs to which node
    # Actually ts.haplotypes() yields in the order of ts.samples()
    
    # Let's build a map: node_id -> sequence
    for i, seq in enumerate(ts.haplotypes()):
        node_id = samples_list[i]
        node_to_seq[node_id] = seq
        
    # Now write files
    for day in range(max_days):
        filename = os.path.join(daily_dir, f"day_{day}.fasta")
        with open(filename, "w") as f:
            if day in census_by_day:
                for record in census_by_day[day]:
                    node_id = record['node_id']
                    if node_id in node_to_seq:
                        seq = node_to_seq[node_id]
                        # Header: >Agent_{id}_{role}_Ward{ward}_Day{day}_Detected{bool}
                        header = f">Agent_{record['agent_id']}_{record['role']}_Ward{record['ward']}_Day{day}_Detected{record['is_detected']}"
                        f.write(f"{header}\n{seq}\n")

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
    """
    Manages the viral phylogeny using tskit.
    
    This class builds a phylogenetic tree in 'forward time' as the simulation proceeds.
    It uses tskit's TableCollection to efficiently store the genealogy (who infected whom)
    and the genetic mutations that accumulate over time.
    
    KEY CONCEPTS:
    -------------
    1. NODES: Represent a specific viral lineage at a specific point in time.
       - A transmission event (A -> B) creates a new node for the virus in B.
       - It also updates the virus in A (creating a new node for A) to account for evolution within A.
    
    2. EDGES: Represent parent-child relationships (transmission or inheritance).
       - An edge links a parent node to a child node over a specific genomic interval (0 to L).
       - In this simulation, we track the whole genome (0 to genome_length).
    
    3. SITES & MUTATIONS:
       - Sites are specific positions on the genome (0..L-1).
       - Mutations are changes at a site (e.g., A -> T) occurring on a specific node (branch).
       - We simulate mutations on-the-fly as edges are added.
    
    4. TIME:
       - The simulation runs in 'forward time' (Day 0, 1, 2...).
       - tskit natively expects 'time ago' (0 = present, 100 = past).
       - We store forward time in the tables during the sim, and then INVERT it in `finalize_tree`.
    """
    
    def __init__(self, genome_length, mutation_rate, community_diversity_level, num_community_lineages, community_pop_size, burn_in_days=21, reference_path=None, transition_prob=0.7):
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.transition_prob = transition_prob
        
        # --- Load Reference Sequence ---
        if reference_path and os.path.exists(reference_path):
            # Simple FASTA parser
            with open(reference_path, 'r') as f:
                lines = f.readlines()
            seq = "".join([line.strip() for line in lines if not line.startswith(">")]).upper()
            if len(seq) != genome_length:
                print(f"WARNING: Reference length {len(seq)} != genome_length {genome_length}. Truncating or padding.")
                if len(seq) > genome_length:
                    seq = seq[:genome_length]
                else:
                    seq = seq + "A" * (genome_length - len(seq))
            self.reference_sequence = seq
            print(f"Loaded reference genome from {reference_path}")
        else:
            self.reference_sequence = "A" * genome_length
            print("Using default 'A' reference genome.")
        
        # --- Initialize tskit Tables ---
        # TableCollection is the mutable structure used to build the tree sequence step-by-step.
        self.tables = tskit.TableCollection(sequence_length=genome_length)
        
        # Nodes table: Stores [flags, time, population, individual, metadata]
        self.nodes = self.tables.nodes
        
        # Edges table: Stores [left, right, parent, child]
        self.edges = self.tables.edges
        
        # Sites table: Stores [position, ancestral_state]
        self.sites = self.tables.sites
        
        # Mutations table: Stores [site, node, derived_state, parent]
        self.mutations = self.tables.mutations
        
        # --- Caches & Helpers ---
        # Map position -> site_id to reuse sites (if multiple mutations hit the same spot)
        self.site_map = {}
        self.rng = np.random.default_rng(42)
        
        # Caches for efficient traversal (needed for real-time distance calculation)
        # parent_map: child_node_id -> parent_node_id
        # Allows tracing lineages backwards from tip to root.
        self.parent_map = {} 
        
        # node_mutations: node_id -> list of site_ids
        # Stores which mutations occurred specifically on the branch leading to this node.
        self.node_mutations = collections.defaultdict(list) 
        
        self.metadata = [] 
        
        # --- Community Reservoir Initialization ---
        # We simulate a "community reservoir" to provide genetically diverse importations.
        # Instead of all importations being identical (clones), they come from evolving lineages.
        self.community_lineages = []
        self.num_community_lineages = num_community_lineages
        self.community_pop_size = community_pop_size
        
        # Calculate depth of the tree needed to match requested diversity
        target_muts = community_diversity_level * genome_length
        muts_per_day = mutation_rate * genome_length
        
        # Time ago for the Grand MRCA (Most Recent Common Ancestor)
        # This root connects all community lineages.
        t_ago = (target_muts / muts_per_day if muts_per_day > 0 else 0) + burn_in_days
        
        # Create Grand MRCA Node
        # Time is negative relative to Day 0.
        mrca = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=-t_ago)
        
        # Initialize separate lineages (e.g., Delta, Omicron) descending from MRCA
        for i in range(num_community_lineages):
            lineage_start_time = -burn_in_days
            
            # Create a root for this specific lineage
            # Lineage root should be MORE RECENT than MRCA (higher value in forward time)
            lineage_root = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=lineage_start_time + 1e-5)
            self.edges.add_row(parent=mrca, child=lineage_root, left=0, right=genome_length)
            self.parent_map[lineage_root] = mrca
            self._add_mutations(mrca, lineage_root, 0, genome_length)
            
            # Create initial population for this lineage
            # Each individual should be MORE RECENT than lineage root
            current_gen = []
            for j in range(community_pop_size):
                individual_time = lineage_start_time + 2e-5 + j * 1e-5
                u = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=individual_time)
                self.edges.add_row(parent=lineage_root, child=u, left=0, right=genome_length)
                self.parent_map[u] = lineage_root
                self._add_mutations(lineage_root, u, 0, genome_length)
                current_gen.append(u)
            self.community_lineages.append(current_gen)
            
        # --- Burn-in Phase ---
        # Evolve the community lineages for 'burn_in_days' before the simulation starts.
        # This generates a cloud of related but distinct variants at Day 0.
        print(f"Evolving community lineages for {burn_in_days} days of burn-in...")
        for d in range(-burn_in_days + 1, 1):
             self.step_community(d)

    def _add_mutations(self, parent, child, left, right):
        """
        Simulates mutations on the branch connecting 'parent' to 'child'.
        
        Logic:
        1. Calculate branch length (time difference).
        2. Calculate expected number of mutations (rate * length * genome_size).
        3. Draw actual number from Poisson distribution.
        4. Pick random positions and add to mutation table.
        """
        t_parent = self.nodes.time[parent]
        t_child = self.nodes.time[child]
        
        # In forward time, child is usually 'later' (higher value) than parent.
        # But we need the absolute time difference.
        branch_len = t_child - t_parent
        if branch_len < 0: branch_len = 0 # Should not happen if time is monotonic
            
        span = right - left
        expected_muts = self.mutation_rate * branch_len * span
        n_muts = self.rng.poisson(expected_muts)
        
        if n_muts > 0:
            positions = self.rng.integers(int(left), int(right), size=n_muts)
            positions.sort()
            
            for pos in positions:
                # Ensure site exists in the table
                if pos not in self.site_map:
                    ancestral_base = self.reference_sequence[pos]
                    site_id = self.sites.add_row(position=pos, ancestral_state=ancestral_base)
                    self.site_map[pos] = site_id
                
                site_id = self.site_map[pos]
                ancestral_base = self.sites.ancestral_state[site_id]
                
                # Mutation Model with Transition/Transversion Bias
                # Transitions: A <-> G, C <-> T
                # Transversions: All other changes
                
                is_transition = self.rng.random() < self.transition_prob
                
                if ancestral_base == 'A':
                    new_base = 'G' if is_transition else self.rng.choice(['C', 'T'])
                elif ancestral_base == 'G':
                    new_base = 'A' if is_transition else self.rng.choice(['C', 'T'])
                elif ancestral_base == 'C':
                    new_base = 'T' if is_transition else self.rng.choice(['A', 'G'])
                elif ancestral_base == 'T':
                    new_base = 'C' if is_transition else self.rng.choice(['A', 'G'])
                else:
                    # Fallback for N or other characters
                    new_base = self.rng.choice(['A', 'C', 'G', 'T'])
                    while new_base == ancestral_base:
                         new_base = self.rng.choice(['A', 'C', 'G', 'T'])

                self.mutations.add_row(site=site_id, node=child, derived_state=new_base)
                
                # Cache for fast lookup
                self.node_mutations[child].append(site_id)

    def step_community(self, day):
        """
        Advances the community reservoir by one day using a Wright-Fisher model.
        
        - Each lineage (e.g., Variant A) has a fixed population size N.
        - Generation t+1 is formed by sampling N parents from Generation t with replacement.
        - This maintains genetic diversity and drift in the background.
        """
        selection_rng = np.random.default_rng(day + 999) 
        
        for i in range(self.num_community_lineages):
            prev_gen = self.community_lineages[i]
            next_gen = []
            
            for j in range(self.community_pop_size):
                # Pick parent uniformly at random
                parent = selection_rng.choice(prev_gen)
                
                parent_time = self.nodes.time[parent]
                # Use fractional day increments for within-generation variation
                # This ensures strict ordering: day.0 < day.000001 < day.000002 etc
                child_time = day + (j + 1) * 1e-5
                
                # Double check ordering
                if child_time <= parent_time:
                    child_time = parent_time + 1e-5
                
                # Create new node for this individual in the new generation
                child = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
                self.edges.add_row(parent=parent, child=child, left=0, right=self.genome_length)
                
                # Update caches and mutations
                self.parent_map[child] = parent
                self._add_mutations(parent, child, 0, self.genome_length)
                
                next_gen.append(child)
            
            self.community_lineages[i] = next_gen

    def add_transmission(self, source_node, time_now):
        """
        Records a transmission event (Source -> New Infection).
        
        Crucial for Within-Host Evolution:
        ---------------------------------
        The 'source_node' represents the virus in the source at the time they were infected.
        Since then, the virus has evolved within them.
        
        1. We create 'current_source_node' representing the virus in the source *right now*.
           (Source_Old -> Current_Source_Node)
        
        2. We create 'child_node' representing the virus transmitted to the recipient.
           (Current_Source_Node -> Child_Node)
           
        Returns:
            child_node: ID for the recipient.
            current_source_node: Updated ID for the source (they now carry this evolved version).
        """
        parent_time = self.nodes.time[source_node]
        
        # 1. Update Source Backbone (Evolution within Source Host)
        current_source_node = source_node
        if time_now > parent_time:
             current_source_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_now)
             self.edges.add_row(parent=source_node, child=current_source_node, left=0, right=self.genome_length)
             self.parent_map[current_source_node] = source_node
             self._add_mutations(source_node, current_source_node, 0, self.genome_length)
             parent_time = time_now
        
        # 2. Create Child (The Transmitted Virus)
        child_time = time_now
        if child_time <= parent_time: child_time = parent_time + 1e-6
             
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=child_time)
        self.edges.add_row(parent=current_source_node, child=child_node, left=0, right=self.genome_length)
        self.parent_map[child_node] = current_source_node
        self._add_mutations(current_source_node, child_node, 0, self.genome_length)
        
        return child_node, current_source_node

    def add_importation(self, time_now, variant_idx=0):
        """
        Records an importation from the community to the hospital.
        Picks a random individual from the specified community lineage as the source.
        """
        lineage_idx = variant_idx % len(self.community_lineages)
        current_gen = self.community_lineages[lineage_idx]
        
        # Randomly select a source from the community pool
        root = np.random.choice(current_gen)
        
        parent_time = self.nodes.time[root]
        if time_now <= parent_time: time_now = parent_time + 1e-6
        
        child_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_now)
        self.edges.add_row(parent=root, child=child_node, left=0, right=self.genome_length)
        self.parent_map[child_node] = root
        self._add_mutations(root, child_node, 0, self.genome_length)
        return child_node

    def add_sample(self, infected_node, sample_time):
        """
        Records a sampling event (Patient -> Sample).
        
        Similar to transmission, this accounts for evolution within the patient 
        up to the moment of sampling.
        
        1. Update patient's virus to 'sample_time'.
        2. Branch off a 'sample_node' (tip) from that updated state.
        """
        parent_time = self.nodes.time[infected_node]
        
        # 1. Update Backbone (Evolution within Patient)
        current_node = infected_node
        if sample_time > parent_time:
            current_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=sample_time)
            self.edges.add_row(parent=infected_node, child=current_node, left=0, right=self.genome_length)
            self.parent_map[current_node] = infected_node
            self._add_mutations(infected_node, current_node, 0, self.genome_length)
            parent_time = sample_time
            
        # 2. Create Sample Tip
        tip_time = sample_time
        if tip_time <= parent_time: tip_time = parent_time + 1e-6
            
        sample_node = self.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=tip_time)
        self.edges.add_row(parent=current_node, child=sample_node, left=0, right=self.genome_length)
        self.parent_map[sample_node] = current_node
        self._add_mutations(current_node, sample_node, 0, self.genome_length)
        
        return sample_node, current_node
    
    def finalize_tree(self, max_time):
        """
        Finalizes the tree for export/analysis.
        
        CRITICAL STEP: TIME INVERSION
        - We simulated in forward time (0 -> max_time).
        - tskit expects 'time ago' (max_time -> 0).
        - We subtract all node times from max_time + buffer.
        
        Returns:
            ts: An immutable tskit.TreeSequence object.
        """
        # Invert times: forward time -> time ago
        # Use max_time + 1 to ensure positive times after inversion
        times = np.array(self.nodes.time)
        self.nodes.time = (max_time + 1.0) - times
        
        # Sort tables (required by tskit)
        self.tables.sort()
        
        # Build the final object
        ts = self.tables.tree_sequence()
        return ts

    def get_pairwise_distance(self, node_a, node_b):
        """
        Calculates genetic distance (Hamming distance) between two nodes.
        
        Distance = Size of Symmetric Difference of Mutation Sets
        (Mutations in A but not B) + (Mutations in B but not A)
        """
        muts_a = self._get_mutations_on_lineage(node_a)
        muts_b = self._get_mutations_on_lineage(node_b)
        diff = muts_a.symmetric_difference(muts_b)
        return len(diff)

    def _get_mutations_on_lineage(self, node_id):
        """
        Traces the lineage of 'node_id' back to the root, collecting all mutations.
        Uses the cached 'parent_map' and 'node_mutations' for speed.
        """
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

def save_daily_fastas(ts, hospital, output_dir, max_days):
    """
    Saves daily FASTA files containing sequences of samples collected on each day.
    Creates a 'daily_sequences' subfolder.
    """
def save_daily_fastas(ts, daily_census, output_dir, max_days):
    """
    Saves daily FASTA files containing sequences of ALL infected agents on each day.
    
    Args:
        ts: TreeSequence
        daily_census: List of dicts or tuples with (day, agent_id, role, ward, node_id, is_detected)
        output_dir: Output directory
        max_days: Simulation duration
    """
    daily_dir = os.path.join(output_dir, "daily_sequences")
    os.makedirs(daily_dir, exist_ok=True)
    
    # Group census by day
    census_by_day = collections.defaultdict(list)
    for record in daily_census:
        census_by_day[record['day']].append(record)
        
    # We need to extract sequences for these specific nodes.
    # ts.haplotypes() iterates over ALL samples in the tree.
    # We need to map node_id -> sequence.
    
    node_to_seq = {}
    samples_list = ts.samples()
    
    # Create a map of sample_node_id -> sequence
    for i, seq in enumerate(ts.haplotypes()):
        node_id = samples_list[i]
        node_to_seq[node_id] = seq
        
    # Now write files
    for day in range(max_days):
        filename = os.path.join(daily_dir, f"day_{day}.fasta")
        with open(filename, "w") as f:
            if day in census_by_day:
                for record in census_by_day[day]:
                    node_id = record['node_id']
                    if node_id in node_to_seq:
                        seq = node_to_seq[node_id]
                        # Header: >Agent_{id}_{role}_Ward{ward}_Day{day}_Detected{bool}
                        header = f">Agent_{record['agent_id']}_{record['role']}_Ward{record['ward']}_Day{day}_Detected{record['is_detected']}"
                        f.write(f"{header}\n{seq}\n")

def save_node_ids(hospital, output_dir):
    with open(os.path.join(output_dir, "hospital_node_ids.txt"), "w") as f:
        for a in hospital.agents:
            if a.is_sampled:
                f.write(f"{a.id},{a.sample_node}\n")

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
        
    # We must use map_nodes=True to track which new node corresponds to which old node
    ts_hospital, node_map = ts.simplify(samples=hospital_nodes_list, map_nodes=True)
    
    new_id_to_agent = {}
    # node_map is an array where index = old_node_id, value = new_node_id (or -1)
    for old_id, new_id in enumerate(node_map):
        if new_id != -1 and old_id in sample_node_to_agent:
            new_id_to_agent[new_id] = sample_node_to_agent[old_id]
            
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