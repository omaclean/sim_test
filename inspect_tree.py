import tskit
ts = tskit.load("hospital_outbreak.trees")
print(f"Number of nodes: {ts.num_nodes}")
print(f"Number of edges: {ts.num_edges}")
print(f"Number of mutations: {ts.num_mutations}")
print(f"Number of sites: {ts.num_sites}")
