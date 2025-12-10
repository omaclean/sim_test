# Deterministic Ladder-Tree Evolution Model

## Overview

The new deterministic mode implements a **ladder-tree structure** that maintains **constant population diversity** while allowing **divergence through time**. This addresses the problems with the previous Wright-Fisher based deterministic model which accumulated diversity over time.

## How It Works

### Tree Structure

```
Time →
    Backbone (persists through time)
    │
    ├─── Branch 1 (created day 0, dies day 50)
    │
    ├─── Branch 2 (created day 10, dies day 60)
    │
    ├─── Branch 3 (created day 20, dies day 70)
    │
    └─── Branch 4 (created day 30, still alive)
```

### Key Features

1. **Backbone Lineage**: One main lineage persists through entire simulation
   - Accumulates mutations at constant rate (molecular clock)
   - Provides temporal continuity

2. **Side Branches**: Regular branching events create diversity
   - New branches created every `branching_interval` days
   - Old branches die out (replaced by new ones)
   - Population size stays constant

3. **Constant Diversity**: 
   - New mutations added on backbone and new branches
   - Old mutations lost when branches die out
   - Net result: diversity oscillates around constant mean

4. **Divergence Through Time**:
   - Backbone accumulates mutations linearly
   - Pairwise distances increase with time (molecular clock)
   - But **mean pairwise distance** stays constant due to branch turnover

## Parameters

### `--target_pairwise_distance` (optional)
- Target mean pairwise distance in number of mutations
- Controls overall diversity level
- Default: calculated from `mutation_rate * burn_in_days * genome_length`

**Example values:**
- 25-50 mutations: Low diversity (recently diverged populations)
- 50-100 mutations: Moderate diversity
- 100-200 mutations: High diversity (distantly related)

### `--branching_interval` (default: 10)
- Days between branching events
- Shorter interval = more frequent turnover = more stable diversity
- Longer interval = less frequent turnover = more diversity oscillation

**Recommended values:**
- 5 days: Very stable, minimal oscillation
- 10 days: Good balance (default)
- 20+ days: More oscillation, larger diversity range

## Usage Examples

### Basic deterministic simulation:
```bash
python run_population_sim.py \
    --num_lineages 1 \
    --pop_size 50 \
    --mutation_rate 1e-5 \
    --burn_in 30 \
    --sim_days 300 \
    --genome_length 10000 \
    --output_dir my_deterministic_sim \
    --deterministic \
    --target_pairwise_distance 50 \
    --branching_interval 10 \
    --seed 123
```

### High diversity, stable:
```bash
python run_population_sim.py \
    --deterministic \
    --target_pairwise_distance 150 \
    --branching_interval 5 \
    --pop_size 100 \
    --sim_days 500 \
    --output_dir high_diversity_stable
```

### Compare stochastic vs deterministic:
```bash
# Run stochastic
python run_population_sim.py --output_dir test_stochastic --sim_days 200

# Run deterministic
python run_population_sim.py --output_dir test_deterministic \
    --deterministic --target_pairwise_distance 50 --sim_days 200

# Compare
python compare_modes.py test_stochastic test_deterministic
```

## When to Use Deterministic Mode

### ✅ Use deterministic mode when:
- You need **reproducible** results with controlled diversity
- You want **constant mean pairwise distance** through time
- You're testing hypotheses requiring controlled genetic diversity
- You want to avoid stochastic drift effects
- You need predictable, interpretable phylogenies

### ⚠️ Use stochastic mode when:
- You want to model **realistic population dynamics**
- Genetic drift is important for your research question
- You need natural variation in diversity over time
- You're fitting to real-world data (which has drift)

## Differences from Stochastic Mode

| Aspect | Stochastic (Wright-Fisher) | Deterministic (Ladder-Tree) |
|--------|---------------------------|----------------------------|
| **Diversity over time** | Increases/varies stochastically | Oscillates around constant mean |
| **Tree structure** | Random, bushy | Regular, ladder-like |
| **Reproducibility** | Variable (even with same seed*) | Fully reproducible |
| **Branch lengths** | Variable | More uniform |
| **Coalescence** | Random timing | Regular intervals |
| **Lineage survival** | Stochastic (drift) | Deterministic (backbone persists) |

*Stochastic mode is reproducible with seeds, but diversity varies naturally

## Implementation Details

### Mutation Strategy
- **Backbone**: Adds ~1 mutation per day (molecular clock)
- **New branches**: Get small number of distinguishing mutations
- **Old branches**: Continue accumulating mutations slowly
- **Branch death**: Old branches replaced, their mutations lost

### Population Management
- Fixed population size maintained each generation
- Backbone always preserved (index 0)
- Side branches created/removed to maintain pop_size
- Newer branches preferentially kept over old ones

### Time Ordering
- Strict parent → child time ordering enforced
- Uses microsecond offsets (1e-5, 1e-6) for same-day events
- Ensures tskit compatibility

## Output Interpretation

### Diversity Statistics (`diversity_stats.json`)

When viewing results, you should see:

**Stochastic mode:**
```
Mean diversity: 65.3 ± 18.2 (range: 30-95)
Coefficient of variation: 0.279
```

**Deterministic mode:**
```
Mean diversity: 76.5 ± 7.1 (range: 55-87)  
Coefficient of variation: 0.093
```

The deterministic mode shows ~67% reduction in variation!

### Tree Visualization

- **By lineage**: Shows branch structure (ladder pattern in deterministic)
- **By time**: Shows temporal dynamics (regular branching in deterministic)

## Tips for Best Results

1. **Set appropriate target diversity**: 
   - For SARS-CoV-2 at ~30kb: 10-50 mutations typical
   - For your genome length: scale proportionally

2. **Adjust branching interval**:
   - Want smoother diversity? Decrease interval (e.g., 5)
   - Want larger population turnover? Increase interval (e.g., 20)

3. **Population size**:
   - Larger pop_size = more branches = more realistic diversity distribution
   - Minimum ~20-30 for good diversity structure

4. **Burn-in period**:
   - Longer burn-in = more diverse initial population
   - But remember: diversity stays constant after burn-in!

## Validation

Check your results with:

```bash
# View diversity over time
python -c "
import json
data = json.load(open('your_output_dir/diversity_stats.json'))
print('Mean diversity:', sum(data['mean_pairwise_distance'])/len(data['days']))
print('Std diversity:', __import__('numpy').std(data['mean_pairwise_distance']))
"

# Generate comparison plots
python compare_modes.py stochastic_dir deterministic_dir
```

Good results show:
- CV (coefficient of variation) < 0.15 for deterministic
- Mean diversity close to target_pairwise_distance
- Oscillating pattern (not monotonic increase)

## Troubleshooting

**Q: Diversity still increasing over time?**
- Check branching_interval isn't too large
- Verify pop_size is adequate (>20)
- Ensure --deterministic flag is set

**Q: Too much oscillation?**
- Decrease branching_interval (e.g., from 10 to 5)
- Increase pop_size for smoother averaging

**Q: Getting time ordering errors?**
- Update to latest version (bug fixed)
- Check that burn_in_days > 0

**Q: Want different diversity level?**
- Set --target_pairwise_distance explicitly
- Or adjust mutation_rate and burn_in_days
