import pandas as pd
import itertools
import numpy as np

# Load mutations
df = pd.read_csv("mutations_per_patient.csv")

# Group by agent
agent_muts = df.groupby('agent_id')['position'].apply(set).to_dict()

# Get all agents
agents = list(agent_muts.keys())
print(f"Number of agents with mutations: {len(agents)}")

# If there are too many agents, sample a subset
if len(agents) > 100:
    agents = np.random.choice(agents, 100, replace=False)

# Calculate pairwise differences
diffs = []
for i in range(len(agents)):
    for j in range(i+1, len(agents)):
        a1 = agents[i]
        a2 = agents[j]
        
        muts1 = agent_muts.get(a1, set())
        muts2 = agent_muts.get(a2, set())
        
        # Hamming distance (assuming all mutations are unique sites and binary)
        # Distance = size of symmetric difference
        dist = len(muts1.symmetric_difference(muts2))
        diffs.append(dist)

print(f"Mean pairwise distance: {np.mean(diffs)}")
print(f"Median pairwise distance: {np.median(diffs)}")
print(f"Min pairwise distance: {np.min(diffs)}")
print(f"Max pairwise distance: {np.max(diffs)}")

# Histogram
counts, bins = np.histogram(diffs, bins=range(0, max(diffs)+2))
print("Distribution:")
for b, c in zip(bins, counts):
    if c > 0:
        print(f"{b}: {c}")
