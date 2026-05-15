"""
Epitope Conservation Frequency - Multi-Clade Analysis
Supports multiple H5N1 clades with genotype classification within each clade
"""

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


EPITOPES = {
    "E1": [(67,71), (79,82), (98,103), (153,153), (154,154), (161,161)],
    "E2": [(59,59), (61,63), (84,84), (87,91), (106,109),
           (124,126), (128,152), (155,159), (162,162),
           (166,184), (186,189), (196,241),
           (247,262), (267,269), (271,275)],
    "E3": [(413,413), (415,425), (427,427), (428,428),
           (431,431), (432,432), (435,435)],
    "E4": [(31,31), (33,40)],
    "E5": [(56,56), (57,57), (285,290)],
}

GENOTYPE_MAP = {
    "A3": "A3",
    "B3.13": "B3.13",
    "B3.2": "B3.2",
    "B3.6": "B3.6",
    "D1": "D1.1",
}

SIMILARITY_THRESHOLD = 0.95


def parse_fasta(fasta_file):
    """Parse FASTA file and extract sequences with metadata."""
    sequences = {}
    current_header = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences[current_header] = ''.join(current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_header:
            sequences[current_header] = ''.join(current_seq)
    
    return sequences


def extract_clade_and_genotype(header):
    """Extract clade and genotype from FASTA header."""
    
    if header.startswith('H5M3') or header.startswith('VXT'):
        return None, None
    
    parts = header.split('|')
    
    clade = None
    genotype = None
    
    if parts[0] == '' and len(parts) > 1:
        clade = parts[1].strip()
        for part in parts[2:]:
            part = part.strip()
            for key in GENOTYPE_MAP.keys():
                if part.startswith(key):
                    genotype = GENOTYPE_MAP[key]
                    break
            if genotype:
                break
    else:
        genotype_part = parts[0].strip()
        for key in GENOTYPE_MAP.keys():
            if genotype_part.startswith(key):
                genotype = GENOTYPE_MAP[key]
                break
    
    return clade, genotype


def extract_epitope_residues(sequence, epitope_ranges):
    """Extract residues for an epitope from sequence."""
    epitope_seq = []
    for start, end in epitope_ranges:
        start_idx = start - 1
        end_idx = end
        
        if end_idx <= len(sequence):
            epitope_seq.append(sequence[start_idx:end_idx])
    
    return ''.join(epitope_seq)


def calculate_similarity(seq1, seq2):
    """Calculate sequence identity (%)"""
    if len(seq1) != len(seq2):
        return 0.0
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def sort_clades(clades):
    """Sort clades numerically (2.3.4.2 < 2.3.4.4b)"""
    def clade_key(clade):
        parts = clade.split('.')
        result = []
        for part in parts:
            try:
                result.append((0, int(part)))
            except ValueError:
                result.append((1, part))
        return result
    
    return sorted(clades, key=clade_key)


def calculate_conservation(sequences, reference_seq, threshold=SIMILARITY_THRESHOLD):
    """Calculate epitope conservation frequency across clades and genotypes."""
    
    clade_genotype_seqs = defaultdict(list)
    all_clades = set()
    all_genotypes_per_clade = defaultdict(set)
    
    for header, seq in sequences.items():
        if header.startswith('H5M3') or header.startswith('VXT'):
            continue
        
        clade, genotype = extract_clade_and_genotype(header)
        
        if clade is None and genotype:
            clade = "No_Clade"
        
        if clade and genotype:
            key = f"{clade}_{genotype}"
            clade_genotype_seqs[key].append(seq)
            all_clades.add(clade)
            all_genotypes_per_clade[clade].add(genotype)
        elif clade and not genotype:
            key = f"{clade}_Other"
            clade_genotype_seqs[key].append(seq)
            all_clades.add(clade)
            all_genotypes_per_clade[clade].add("Other")
    
    # Sort clades numerically
    sorted_clades = sort_clades(list(all_clades))
    
    # Create ordered list of clade_genotype combinations
    clade_genotype_order = []
    for clade in sorted_clades:
        genotypes = sorted(all_genotypes_per_clade[clade])
        for genotype in genotypes:
            clade_genotype_order.append(f"{clade}_{genotype}")
    
    # Calculate conservation for each epitope
    conservation_data = []
    
    for epitope_name in ["E1", "E2", "E3", "E4", "E5"]:
        epitope_ranges = EPITOPES[epitope_name]
        ref_epitope = extract_epitope_residues(reference_seq, epitope_ranges)
        
        frequencies = {}
        
        for clade_genotype in clade_genotype_order:
            if clade_genotype not in clade_genotype_seqs:
                frequencies[clade_genotype] = 0.0
                continue
            
            sequences_list = clade_genotype_seqs[clade_genotype]
            conservation_count = 0
            
            for seq in sequences_list:
                query_epitope = extract_epitope_residues(seq, epitope_ranges)
                if query_epitope and len(query_epitope) == len(ref_epitope):
                    similarity = calculate_similarity(ref_epitope, query_epitope)
                    if similarity >= threshold:
                        conservation_count += 1
            
            frequency = (conservation_count / len(sequences_list)) * 100 if sequences_list else 0
            frequencies[clade_genotype] = frequency
        
        conservation_data.append({
            'Epitope': epitope_name,
            **frequencies
        })
    
    df = pd.DataFrame(conservation_data).set_index('Epitope')
    df = df[[col for col in clade_genotype_order if col in df.columns]]
    
    return df, clade_genotype_seqs


def create_heatmap(conservation_df, output_file='epitope_conservation_heatmap.png', threshold=SIMILARITY_THRESHOLD):
    """Create professional publication-quality heatmap."""
    
    # Custom colormap: White (0) -> Yellow (75) -> Orange (95) -> Red (100)
    colors = ['white', 'yellow', 'orange', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Calculate figure size to make cells more square
    n_cols = len(conservation_df.columns)
    n_rows = len(conservation_df)
    
    # More square aspect ratio
    fig_width = max(8, n_cols * 0.5)  # 0.5 inches per column (narrower)
    fig_height = max(5, n_rows * 1.2)  # 1.2 inches per row
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    sns.heatmap(
        conservation_df,
        annot=False,  # No values shown
        cmap=cmap,
        cbar_kws={'label': 'Conservation Frequency (%)'},
        linewidths=1,  # Thicker lines
        linecolor='black',  # Black lines instead of gray
        vmin=0,
        vmax=100,
        ax=ax,
    )
    
    ax.set_xlabel('H5N1 Clade | Genotype', fontsize=12, fontweight='bold')
    ax.set_ylabel('B-cell Epitope', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Conservation Frequency of Predicted B-cell Epitopes\nAcross H5N1 Clades and Genotypes (≥{int(threshold*100)}% identity)',
        fontsize=13,
        fontweight='bold',
        pad=20
    )
    
    # Format x-axis labels
    labels = [label.get_text() for label in ax.get_xticklabels()]
    formatted_labels = [label.replace('_', '\n') for label in labels]
    ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved: {output_file}")
    plt.show()


def print_summary(sequences, reference_seq, conservation_df, clade_genotype_seqs, threshold):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("EPITOPE CONSERVATION ANALYSIS - MULTI-CLADE SUMMARY")
    print("="*80)
    
    print(f"\nTotal sequences in dataset: {len(sequences)}")
    print(f"Reference sequence: H5M3/VXT (length: {len(reference_seq)} aa)")
    print(f"Similarity threshold: {int(threshold*100)}%")
    
    print(f"\nSequences per clade and genotype:")
    print("-" * 80)
    
    current_clade = None
    for clade_genotype in conservation_df.columns:
        clade, genotype = clade_genotype.rsplit('_', 1)
        
        if clade != current_clade:
            if current_clade is not None:
                print()
            print(f"  Clade: {clade}")
            current_clade = clade
        
        count = len(clade_genotype_seqs.get(clade_genotype, []))
        print(f"    {genotype:12s}: {count:3d} sequences")
    
    print(f"\nConservation frequencies (%):")
    print("-" * 80)
    print(conservation_df.to_string())
    
    print("\n" + "="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python freq_epi_multiClade.py <fasta_file> [similarity_threshold]")
        print("Example: python freq_epi_multiClade.py H5100.fasta 0.95")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else SIMILARITY_THRESHOLD
    
    if not Path(fasta_file).exists():
        print(f"Error: File '{fasta_file}' not found")
        sys.exit(1)
    
    print(f"Loading sequences from: {fasta_file}")
    sequences = parse_fasta(fasta_file)
    print(f"✓ Loaded {len(sequences)} sequences")
    
    reference_header = None
    for header in sequences.keys():
        if header.startswith('H5M3') or header.startswith('VXT'):
            reference_header = header
            break
    
    if not reference_header:
        print("Error: Reference sequence (H5M3 or VXT) not found")
        sys.exit(1)
    
    reference_seq = sequences[reference_header]
    print(f"✓ Reference: {reference_header} ({len(reference_seq)} aa)")
    
    print(f"\nCalculating conservation (similarity threshold: {int(threshold*100)}%)...")
    conservation_df, clade_genotype_seqs = calculate_conservation(sequences, reference_seq, threshold)
    print("✓ Calculation complete")
    
    print_summary(sequences, reference_seq, conservation_df, clade_genotype_seqs, threshold)
    
    print("\nGenerating heatmap...")
    create_heatmap(conservation_df, threshold=threshold)
    
    csv_file = f'epitope_conservation_frequencies_{int(threshold*100)}.csv'
    conservation_df.to_csv(csv_file)
    print(f"✓ Data saved: {csv_file}")


if __name__ == "__main__":
    main()
