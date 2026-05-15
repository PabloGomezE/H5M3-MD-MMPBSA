# H5M3 Mosaic Hemagglutinin - Structural and Dynamic Characterization

Comprehensive molecular dynamics simulations and MM-PBSA binding energy analysis of H5M3, a rationally designed mosaic hemagglutinin vaccine antigen against highly pathogenic avian influenza H5N1 clade 2.3.4.4b.

## Overview

This repository contains computational validation of H5M3 mosaic vaccine design through:
- **80 ns molecular dynamics simulations** of H5M3 and wild-type H5N1 variants
- **Structural and dynamic characterization** (RMSD, RMSF, SASA, hydrogen bonds, PCA)
- **MM-PBSA binding free energy calculations** validating cross-reactive antibody recognition
- **Multi-clade epitope conservation analysis** across H5N1 genotypes and clades
- **Reproducible analysis scripts** for structural validation

## Contents

### Molecular Dynamics Data
- `MDP_files/` - GROMACS parameter files for MD simulations
  - `trimmer/` - H5 trimeric ectodomain parameters
  - `complex/` - Fab-H5 antibody-antigen complex parameters
  
- `Rundata/` - Complete analysis results from 80 ns MD simulations:
  - `trimmer_RMSD/` - Global structural stability (Cα root mean square deviation)
  - `trimmer_RMSF/` - Residue-level flexibility (root mean square fluctuation)
  - `trimmer_AASA/` - Antibody accessible surface area analysis
  - `trimmer_HBONDS/` - Hydrogen bond formation and stability
  - `complex_MMPBSA/` - Binding free energy calculations (MM-PBSA)
  - `eigenvalues/` - Principal component analysis of conformational space

### Analysis Scripts & Tools
- `Scripts/` - Python analysis scripts for reproducibility
- `epitope_conservation/` - Multi-clade epitope conservation analysis
  - `freq_epi_multiClade.py` - Calculates epitope conservation frequencies across H5N1 clades and genotypes
  - Supports adjustable similarity thresholds (90%, 95%, 100% sequence identity)
  - Generates publication-quality heatmaps for visualization
  - Input: FASTA file with H5 sequences

## Systems Analyzed

### Primary MD Systems (80 ns simulations)
- **H5M3** - Rationally designed mosaic hemagglutinin
- **D1.1** - Wild-type H5N1 clade 2.3.4.4b reference
- **Astrakhan 3212/2020** - Contemporary H5N1 clade 2.3.4.4b isolate

### Extended Epitope Analysis Dataset
- Multiple H5N1 clades (2.3.4.2, 2.3.4.4b, and others)
- Multiple genotypes within clade 2.3.4.4b (A3, B3.13, B3.2, B3.6, D1.1)
- ~1600 complete H5 sequences from diverse global H5N1 isolates

## Key Findings

- H5M3 maintains **comparable structural stability** to wild-type variants
- **Preserved epitope flexibility** at critical B-cell epitope regions
- **Conserved antibody accessibility** across predicted discontinuous epitopes
- **Equivalent binding free energies** (H5M3-Fab ≈ D1.1-Fab; p > 0.05)
- **Broad epitope conservation** across contemporary H5N1 clades and genotypes

## Usage

### Epitope Conservation Analysis
```bash
cd epitope_conservation
python freq_epi_multiClade.py H5100.fasta 0.95
```

**Output**: 
- `epitope_conservation_heatmap.png` - Conservation frequency heatmap
- `epitope_conservation_frequencies_95.csv` - Tabular data

### Epitope Similarity Thresholds
- `0.95` - 95% sequence identity (recommended)
- `0.90` - 90% sequence identity (more permissive)
- `1.00` - 100% exact matches

## Methods

- **Structure Modeling**: AlphaFold3 de novo modeling of H5M3 trimer
- **MD Simulations**: GROMACS 2021+, 80 ns NPT ensemble, ff14SB/AMBER force field
- **Analysis Tools**: GROMACS, MDAnalysis, gmx_MMPBSA
- **Epitope Prediction**: ElliPro (discontinuous B-cell epitopes)
- **Structure Refinement**: PyMOL, YASARA

## Data & Reproducibility

- All MD parameter files (MDP) provided for simulation reproduction
- Analysis output files (.xvg, .txt) included for verification
- Python scripts available for custom analysis
- Raw MD trajectories (.xtc files) available upon request

## Citation

If you use this repository, data, or analysis scripts, please cite the correspinding publication

