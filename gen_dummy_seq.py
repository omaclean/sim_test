def generate_poly_a_fasta(filename, num_seqs=100, length=29000):
    with open(filename, 'w') as f:
        for i in range(1, num_seqs + 1):
            # Header line
            f.write(f">sequence_{i}\n")
            # Sequence line (all A)
            f.write("A" * length + "\n")

if __name__ == "__main__":
    output_name = "poly_a_sequences.fasta"
    generate_poly_a_fasta(output_name)
    print(f"Successfully created {output_name} with 100 sequences.")