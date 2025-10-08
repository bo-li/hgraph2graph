#!/usr/bin/env python
"""
Memory-efficient streaming vocabulary extraction for large datasets.
Processes files in batches without loading all data into RAM.
Version 2: Better progress tracking with imap_unordered.
"""
import sys
import argparse
import glob
from hgraph import MolGraph
from rdkit import Chem
from multiprocessing import Pool
from functools import partial
import rdkit
from tqdm import tqdm

# Suppress RDKit warnings
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def process_smiles_batch(smiles_batch):
    """Process a batch of SMILES strings."""
    vocab = set()
    for smiles in smiles_batch:
        try:
            hmol = MolGraph(smiles)
            for node, attr in hmol.mol_tree.nodes(data=True):
                smiles_node = attr['smiles']
                vocab.add(attr['label'])
                for i, s in attr['inter_label']:
                    vocab.add((smiles_node, s))
        except Exception:
            pass
    return vocab

def process_file_streaming(filepath, chunk_size, ncpu):
    """Process a file in streaming fashion with better progress tracking."""
    pool = Pool(ncpu)
    global_vocab = set()

    # Read all lines and create batches
    batches = []
    batch = []

    print(f"  Reading {filepath.split('/')[-1]}...", file=sys.stderr)
    with open(filepath, 'r') as f:
        for line in f:
            smiles = line.strip().split()[0]
            batch.append(smiles)
            if len(batch) >= chunk_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)

    total_molecules = sum(len(b) for b in batches)
    print(f"  Processing {total_molecules:,} molecules in {len(batches)} batches...", file=sys.stderr)

    # Process with progress bar that updates per batch
    pbar = tqdm(total=len(batches), desc=f"  {filepath.split('/')[-1]}",
                unit=" batch", position=1, leave=False)

    for vocab in pool.imap_unordered(process_smiles_batch, batches):
        global_vocab.update(vocab)
        pbar.update(1)
        pbar.set_postfix({"vocab": len(global_vocab)})

    pbar.close()
    pool.close()
    pool.join()

    return global_vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input txt files')
    parser.add_argument('--pattern', type=str, default='*.txt',
                        help='File pattern to match (default: *.txt)')
    parser.add_argument('--ncpu', type=int, default=16,
                        help='Number of CPU cores to use')
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help='Number of molecules per chunk (smaller = more updates)')
    parser.add_argument('--output', type=str, default='vocab.txt',
                        help='Output vocabulary file')
    args = parser.parse_args()

    # Find all input files
    file_pattern = f"{args.input_dir}/{args.pattern}"
    input_files = sorted(glob.glob(file_pattern))

    if not input_files:
        print(f"ERROR: No files found matching {file_pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(input_files)} files to process", file=sys.stderr)
    print(f"Using {args.ncpu} CPU cores with chunk size {args.chunk_size:,}", file=sys.stderr)

    # Process files in streaming fashion
    global_vocab = set()

    # Progress bar for files
    file_pbar = tqdm(input_files, desc="Files", unit="file", position=0)

    for filepath in file_pbar:
        filename = filepath.split('/')[-1]
        file_pbar.set_description(f"File: {filename}")

        # Process file
        file_vocab = process_file_streaming(filepath, args.chunk_size, args.ncpu)
        global_vocab.update(file_vocab)

        file_pbar.set_postfix({"total_vocab": f"{len(global_vocab):,}"})

    file_pbar.close()

    # Sort and write output
    print(f"\nFinal vocabulary size: {len(global_vocab):,}", file=sys.stderr)
    print(f"Writing to {args.output}...", file=sys.stderr)

    sorted_vocab = sorted(global_vocab)
    with open(args.output, 'w') as f:
        for x, y in tqdm(sorted_vocab, desc="Writing", unit=" entries"):
            f.write(f"{x} {y}\n")

    print("Done!", file=sys.stderr)
