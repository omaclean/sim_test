# Population Genetics Simulation - Quick Start

## Small Test Run (Fast - ~30 seconds)

```bash
# Minimal test run: 2 lineages, 10 individuals each, 14 days
python run_population_sim.py \
    --num_lineages 2 \
    --pop_size 10 \
    --mutation_rate 0.001 \
    --burn_in 7 \
    --sim_days 14 \
    --genome_length 1000 \
    --output_dir ./test_output \
    --seed 42
```

**Expected output:** 
- 14 FASTA files (day_0.fasta to day_13.fasta)
- Each with 20 sequences (2 lineages × 10 individuals)
- 4 plots showing diversity trends

---

## Medium Test Run (Moderate - ~2-3 minutes)

```bash
# Realistic test: 3 lineages, SARS-CoV-2 genome, 30 days
python run_population_sim.py \
    --num_lineages 3 \
    --pop_size 50 \
    --mutation_rate 0.0001 \
    --burn_in 21 \
    --sim_days 30 \
    --genome_length 29903 \
    --reference SC2_data/Wuhan-Hu-1.fa \
    --output_dir ./medium_test \
    --seed 123
```

**Expected output:**
- 30 FASTA files with 150 sequences each (3 × 50)
- Diversity statistics tracking genetic divergence
- Phylogenetic tree sequence file

---

## Parameter Effects

### **High Mutation Rate** (rapid divergence)
```bash
python run_population_sim.py \
    --num_lineages 2 \
    --pop_size 20 \
    --mutation_rate 0.005 \
    --burn_in 10 \
    --sim_days 20 \
    --genome_length 5000 \
    --output_dir ./high_mutation
```
→ Expect: High pairwise distances, rapid increase in diversity

### **Low Mutation Rate** (slow divergence)
```bash
python run_population_sim.py \
    --num_lineages 2 \
    --pop_size 20 \
    --mutation_rate 0.00001 \
    --burn_in 10 \
    --sim_days 20 \
    --genome_length 5000 \
    --output_dir ./low_mutation
```
→ Expect: Low pairwise distances, slow diversity accumulation

### **Large Population** (maintains diversity)
```bash
python run_population_sim.py \
    --num_lineages 1 \
    --pop_size 200 \
    --mutation_rate 0.0002 \
    --burn_in 30 \
    --sim_days 30 \
    --genome_length 10000 \
    --output_dir ./large_pop
```
→ Expect: More balanced allele frequencies, slower genetic drift

### **Small Population** (strong drift)
```bash
python run_population_sim.py \
    --num_lineages 1 \
    --pop_size 10 \
    --mutation_rate 0.0002 \
    --burn_in 30 \
    --sim_days 30 \
    --genome_length 10000 \
    --output_dir ./small_pop
```
→ Expect: Rapid fixation/loss of alleles, lower diversity

### **Many Lineages** (high between-lineage diversity)
```bash
python run_population_sim.py \
    --num_lineages 10 \
    --pop_size 20 \
    --mutation_rate 0.0003 \
    --burn_in 40 \
    --sim_days 30 \
    --genome_length 10000 \
    --output_dir ./many_lineages
```
→ Expect: High between-lineage diversity in plots

---

## Running Tests

```bash
# Run all tests
cd tests
python test_population_sim.py -v

# Run specific test class
python test_population_sim.py TestHammingDistance -v

# Run integration test only (includes small simulation)
python test_population_sim.py TestIntegration.test_small_simulation -v
```

---

## Output Files Explained

After running, your output directory will contain:

```
output_dir/
├── config.json                          # Simulation parameters
├── tree_sequence.trees                  # Full phylogeny (tskit format)
├── diversity_stats.json                 # Diversity metrics over time
├── pairwise_distance_over_time.png      # Mean/median distances by day
├── pairwise_distance_by_week.png        # Mean/median distances by week (bar chart)
├── diversity_breakdown.png              # Within vs between lineage diversity
├── segregating_sites.png                # Polymorphic sites over time
└── daily_sequences/
    ├── day_0.fasta                      # All sequences from day 0
    ├── day_1.fasta
    └── ...
```

### Understanding the Plots

- **pairwise_distance_by_week.png**: Shows how genetic diversity accumulates weekly
  - Blue bars = mean pairwise SNP distance
  - Orange bars = median pairwise SNP distance
  - Higher values = more genetic diversity

- **diversity_breakdown.png**: Shows diversity partitioning
  - Green line = variation within each lineage
  - Red line = variation between lineages
  - Black line = total variation

---

## Troubleshooting

**Simulation too slow?**
- Reduce `--pop_size` (e.g., 20 instead of 100)
- Reduce `--sim_days` (e.g., 20 instead of 90)
- Reduce `--genome_length` (e.g., 5000 instead of 29903)

**Not enough diversity?**
- Increase `--mutation_rate` (e.g., 0.001 instead of 0.0001)
- Increase `--burn_in` (e.g., 50 instead of 21)
- Increase `--sim_days`

**Too much diversity?**
- Decrease `--mutation_rate`
- Decrease `--burn_in`
- Increase `--pop_size` (larger populations evolve slower)
