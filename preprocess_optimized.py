#!/usr/bin/env python
"""
Optimized preprocessing script for large-scale datasets (1B+ molecules).
Optimizations for 128 CPU, 256GB RAM systems.
"""
from multiprocessing import Pool, get_context, set_start_method
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy
from tqdm import tqdm

from hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)

def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) #no need of order for x

def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,) #no need of order for x

if __name__ == "__main__":
    # Safer multiprocessing start (reduces copy-on-write explosions with RDKit/MKL)
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    # Stop NumPy/MKL/BLAS from spawning extra threads inside workers
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Input training file with SMILES')
    parser.add_argument('--vocab', required=True, help='Vocabulary file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Molecules per batch (default: 64, recommended: 64-128 for large RAM)')
    parser.add_argument('--mode', type=str, default='pair',
                        choices=['single', 'pair', 'cond_pair'],
                        help='Processing mode: single molecules, pairs, or conditional pairs')
    parser.add_argument('--ncpu', type=int, default=96,
                        help='Number of worker processes (default: 96, max recommended: 120 for 128-core)')
    parser.add_argument('--chunksize', type=int, default=16,
                        help='Batches per worker task (default: 16, higher = less overhead)')
    parser.add_argument('--shard_size', type=int, default=5000,
                        help='Batches per output file (default: 5000, higher = fewer files)')
    parser.add_argument('--maxtasksperchild', type=int, default=100,
                        help='Tasks per worker before restart (default: 100, prevents memory leaks)')
    parser.add_argument('--output_prefix', type=str, default='tensors',
                        help='Output file prefix (default: tensors)')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Skip shuffling data (faster but less random)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint (skips already processed shards)')
    parser.add_argument('--checkpoint_file', type=str, default='.preprocess_checkpoint.txt',
                        help='Checkpoint file to track progress (default: .preprocess_checkpoint.txt)')
    args = parser.parse_args()

    print("="*80)
    print("OPTIMIZED PREPROCESSING")
    print("="*80)
    print(f"Input file:        {args.train}")
    print(f"Vocabulary:        {args.vocab}")
    print(f"Mode:              {args.mode}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Workers:           {args.ncpu}")
    print(f"Chunk size:        {args.chunksize}")
    print(f"Shard size:        {args.shard_size}")
    print(f"Max tasks/child:   {args.maxtasksperchild}")
    print(f"Shuffle:           {not args.no_shuffle}")
    print(f"Resume mode:       {args.resume}")
    if args.resume:
        print(f"Checkpoint file:   {args.checkpoint_file}")
    print("="*80)

    # Load vocabulary
    print("Loading vocabulary...")
    with open(args.vocab) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    args.vocab = PairVocab(vocab, cuda=False)
    print(f"Vocabulary size: {len(vocab)}")

    # Create worker pool with optimized settings
    pool = get_context("spawn").Pool(
        processes=args.ncpu,
        maxtasksperchild=args.maxtasksperchild
    )

    if not args.no_shuffle:
        random.seed(1)

    # Read and process data based on mode
    print("\nReading input data...")

    if args.mode == 'pair':
        # Dataset contains molecule pairs
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:2] for line in f]

        print(f"Loaded {len(data):,} molecule pairs")

        if not args.no_shuffle:
            print("Shuffling data...")
            random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_pair, vocab = args.vocab)

    elif args.mode == 'cond_pair':
        # Dataset contains molecule pairs with conditions
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:3] for line in f]

        print(f"Loaded {len(data):,} conditional molecule pairs")

        if not args.no_shuffle:
            print("Shuffling data...")
            random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_cond, vocab = args.vocab)

    elif args.mode == 'single':
        # Dataset contains single molecules
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[0] for line in f]

        print(f"Loaded {len(data):,} molecules")

        if not args.no_shuffle:
            print("Shuffling data...")
            random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize, vocab = args.vocab)

    print(f"Created {len(batches):,} batches")
    print(f"Expected output: ~{len(batches) // args.shard_size + 1} shard files")

    # Check for resume
    start_batch = 0
    shard_id = 0
    if args.resume and os.path.exists(args.checkpoint_file):
        with open(args.checkpoint_file, 'r') as f:
            checkpoint_data = f.read().strip().split(',')
            if len(checkpoint_data) >= 2:
                start_batch = int(checkpoint_data[0])
                shard_id = int(checkpoint_data[1])
                print(f"\n{'='*80}")
                print(f"RESUMING FROM CHECKPOINT")
                print(f"{'='*80}")
                print(f"Starting from batch:  {start_batch:,} / {len(batches):,}")
                print(f"Starting from shard:  {shard_id}")
                print(f"Already processed:    {start_batch * args.batch_size:,} molecules")
                print(f"Remaining:            {(len(batches) - start_batch) * args.batch_size:,} molecules")
                print(f"{'='*80}\n")

                # Skip already processed batches
                batches = batches[start_batch:]
    elif args.resume:
        print(f"\nResume requested but no checkpoint file found at {args.checkpoint_file}")
        print("Starting from beginning...\n")

    # Process batches with progress bar
    print("\nProcessing batches...")
    shard, shard_start_batch = [], start_batch
    molecules_processed = start_batch * args.batch_size

    # Use imap_unordered with optimized chunksize
    pbar = tqdm(total=len(batches), desc="Processing", unit=" batch")

    try:
        for i, out in enumerate(pool.imap_unordered(func, batches, chunksize=args.chunksize)):
            shard.append(out)
            molecules_processed += args.batch_size
            current_batch = start_batch + i
            pbar.update(1)
            pbar.set_postfix({
                "molecules": f"{molecules_processed:,}",
                "shard": shard_id,
                "files": shard_id + 1
            })

            # Write shard to disk when full
            if len(shard) >= args.shard_size:
                output_file = f'{args.output_prefix}-{shard_id}.pkl'
                with open(output_file, 'wb') as f:
                    pickle.dump(shard, f, pickle.HIGHEST_PROTOCOL)

                # Update checkpoint after successful shard write
                with open(args.checkpoint_file, 'w') as f:
                    f.write(f"{current_batch + 1},{shard_id + 1}")

                shard, shard_id = [], shard_id + 1

        # Write final shard if any remaining
        if shard:
            output_file = f'{args.output_prefix}-{shard_id}.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(shard, f, pickle.HIGHEST_PROTOCOL)

            # Update final checkpoint
            with open(args.checkpoint_file, 'w') as f:
                f.write(f"{start_batch + len(batches)},{shard_id + 1}")

    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("INTERRUPTED BY USER")
        print("="*80)
        print(f"Processed:         {molecules_processed:,} molecules")
        print(f"Last batch:        {current_batch if 'current_batch' in locals() else start_batch}")
        print(f"Last shard:        {shard_id}")
        print(f"Checkpoint saved:  {args.checkpoint_file}")
        print("="*80)
        print("\nTo resume, run the same command with --resume flag:")
        print(f"  python preprocess_optimized.py --resume [original arguments]")
        print("="*80)
        pbar.close()
        pool.terminate()
        pool.join()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print(f"Checkpoint saved at batch {current_batch if 'current_batch' in locals() else start_batch}")
        pbar.close()
        pool.terminate()
        pool.join()
        raise

    pbar.close()
    pool.close()
    pool.join()

    # Clean up checkpoint file on successful completion
    if os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Total molecules:   {molecules_processed:,}")
    print(f"Total batches:     {start_batch + len(batches):,}")
    print(f"Output files:      {shard_id + 1}")
    print(f"Files pattern:     {args.output_prefix}-*.pkl")
    print(f"Checkpoint:        Removed (processing complete)")
    print("="*80)
    print("\nNext step:")
    print(f"  mkdir -p train_processed")
    print(f"  mv {args.output_prefix}-*.pkl train_processed/")
