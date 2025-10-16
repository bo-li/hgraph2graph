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

def process_smiles_batch(smiles_batch, skip_invalid=False):
    """Process a batch of SMILES strings."""
    vocab = set()
    bad_smiles = []

    for smiles in smiles_batch:
        try:
            hmol = MolGraph(smiles)
            for node, attr in hmol.mol_tree.nodes(data=True):
                smiles_node = attr['smiles']
                vocab.add(attr['label'])
                for i, s in attr['inter_label']:
                    vocab.add((smiles_node, s))
        except Exception as e:
            if skip_invalid:
                bad_smiles.append(smiles[:100] if len(smiles) > 100 else smiles)
            else:
                # Re-raise with more information
                raise RuntimeError(
                    f"Failed to process SMILES: {smiles[:100]}\n"
                    f"Error: {type(e).__name__}: {str(e)[:200]}\n"
                    f"Hint: Use --skip_invalid flag to skip invalid molecules and continue processing."
                )

    return vocab, bad_smiles

def process_file_streaming(filepath, chunk_size, ncpu, skip_invalid=False):
    """Process a file in streaming fashion with better progress tracking."""
    pool = Pool(ncpu)
    global_vocab = set()
    all_bad_smiles = []

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

    func = partial(process_smiles_batch, skip_invalid=skip_invalid)
    for vocab, bad_smiles in pool.imap_unordered(func, batches):
        global_vocab.update(vocab)
        all_bad_smiles.extend(bad_smiles)
        pbar.update(1)
        pbar.set_postfix({"vocab": len(global_vocab), "skipped": len(all_bad_smiles)})

    pbar.close()
    pool.close()
    pool.join()

    return global_vocab, all_bad_smiles

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
    parser.add_argument('--skip_invalid', action='store_true',
                        help='Skip invalid molecules instead of crashing (logs skipped molecules)')
    args = parser.parse_args()

    # Find all input files
    file_pattern = f"{args.input_dir}/{args.pattern}"
    input_files = sorted(glob.glob(file_pattern))

    if not input_files:
        print(f"ERROR: No files found matching {file_pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(input_files)} files to process", file=sys.stderr)
    print(f"Using {args.ncpu} CPU cores with chunk size {args.chunk_size:,}", file=sys.stderr)
    print(f"Skip invalid:      {args.skip_invalid}", file=sys.stderr)

    # Process files in streaming fashion
    global_vocab = set()
    all_skipped = []

    # Progress bar for files
    file_pbar = tqdm(input_files, desc="Files", unit="file", position=0)

    for filepath in file_pbar:
        filename = filepath.split('/')[-1]
        file_pbar.set_description(f"File: {filename}")

        # Process file
        file_vocab, file_skipped = process_file_streaming(filepath, args.chunk_size, args.ncpu, args.skip_invalid)
        global_vocab.update(file_vocab)
        all_skipped.extend(file_skipped)

        file_pbar.set_postfix({
            "total_vocab": f"{len(global_vocab):,}",
            "skipped": len(all_skipped)
        })

    file_pbar.close()

    # Sort and write output
    print(f"\nFinal vocabulary size: {len(global_vocab):,}", file=sys.stderr)

    # Report skipped molecules if any
    if all_skipped:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"SKIPPED MOLECULES", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(f"Total skipped: {len(all_skipped):,}", file=sys.stderr)
        print(f"\nFirst 10 skipped SMILES:", file=sys.stderr)
        for i, smi in enumerate(all_skipped[:10], 1):
            print(f"  {i}. {smi}", file=sys.stderr)
        if len(all_skipped) > 10:
            print(f"  ... and {len(all_skipped) - 10} more", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)

    print(f"Writing to {args.output}...", file=sys.stderr)

    sorted_vocab = sorted(global_vocab)
    with open(args.output, 'w') as f:
        for x, y in tqdm(sorted_vocab, desc="Writing", unit=" entries"):
            f.write(f"{x} {y}\n")

    print("Done!", file=sys.stderr)
