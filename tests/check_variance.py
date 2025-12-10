
import numpy as np

def check_dist(target_muts, variance):
    muts_per_branch = max(1, int(target_muts * 0.5))
    print(f"Target: {target_muts}, Base Muts: {muts_per_branch}, Variance: {variance}")
    
    counts = []
    for i in range(100):
        pseudo_rand = i
        if pseudo_rand < 10:
            base_offset = -2
        elif pseudo_rand < 30:
            base_offset = -1
        elif pseudo_rand < 70:
            base_offset = 0
        elif pseudo_rand < 90:
            base_offset = 1
        else:
            base_offset = 2
            
        offset = int(base_offset * variance)
        val = max(0, muts_per_branch + offset)
        counts.append(val)
        
    mean_muts = np.mean(counts)
    print(f"Mean Muts per Branch: {mean_muts}")
    print(f"Expected Pairwise: {2 * mean_muts}")
    print(f"Counts: {sorted(counts)}")
    print(f"Zeros: {counts.count(0)}")

check_dist(10, 1.0)
check_dist(10, 3.0)
check_dist(12, 3.0)
