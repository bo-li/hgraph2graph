#!/usr/bin/env python
"""
Multi-GPU training script for HierVAE using DataParallel.
Optimized for 2-4 GPUs on a single machine.
Includes Weights & Biases integration for experiment tracking.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

from hgraph import *

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Install with: pip install wandb")
    print("Continuing without W&B logging...")

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=7)

# Multi-GPU settings
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                    help='Comma-separated GPU IDs to use (default: 0,1,2,3)')
parser.add_argument('--batch_size_per_gpu', type=int, default=50,
                    help='Batch size per GPU (default: 50). Total batch = this × num_gpus')

# Model architecture
parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

# Training hyperparameters
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

# Weights & Biases settings
parser.add_argument('--wandb', action='store_true',
                    help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='hgraph2graph',
                    help='W&B project name (default: hgraph2graph)')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='W&B entity/team name (optional)')
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='W&B run name (default: auto-generated)')
parser.add_argument('--wandb_tags', type=str, default=None,
                    help='Comma-separated tags for W&B run')
parser.add_argument('--wandb_notes', type=str, default=None,
                    help='Notes for this W&B run')

args = parser.parse_args()

# Parse GPU IDs
gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
num_gpus = len(gpu_ids)

# Set environment variable for visible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

print("="*80)
print("MULTI-GPU TRAINING CONFIGURATION")
print("="*80)
print(f"GPUs to use:           {gpu_ids} ({num_gpus} GPUs)")
print(f"Batch size per GPU:    {args.batch_size_per_gpu}")
print(f"Total batch size:      {args.batch_size_per_gpu * num_gpus}")
print(f"Save directory:        {args.save_dir}")
print(f"Training data:         {args.train}")
print(f"Vocabulary:            {args.vocab}")
print("="*80)
print()

# Check GPU availability
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    sys.exit(1)

available_gpus = torch.cuda.device_count()
if available_gpus < num_gpus:
    print(f"WARNING: Requested {num_gpus} GPUs but only {available_gpus} available!")
    print(f"Using {available_gpus} GPUs instead.")
    num_gpus = available_gpus

print(f"Available GPUs: {available_gpus}")
for i in range(available_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialize Weights & Biases
use_wandb = args.wandb and WANDB_AVAILABLE
if use_wandb:
    print("="*80)
    print("INITIALIZING WEIGHTS & BIASES")
    print("="*80)

    # Parse tags if provided
    tags = args.wandb_tags.split(',') if args.wandb_tags else []
    tags.append(f"{num_gpus}xGPU")

    # Create config dictionary
    config = {
        # Model architecture
        'rnn_type': args.rnn_type,
        'hidden_size': args.hidden_size,
        'embed_size': args.embed_size,
        'latent_size': args.latent_size,
        'depthT': args.depthT,
        'depthG': args.depthG,
        'diterT': args.diterT,
        'diterG': args.diterG,
        'dropout': args.dropout,

        # Training hyperparameters
        'learning_rate': args.lr,
        'clip_norm': args.clip_norm,
        'step_beta': args.step_beta,
        'max_beta': args.max_beta,
        'warmup': args.warmup,
        'kl_anneal_iter': args.kl_anneal_iter,
        'anneal_rate': args.anneal_rate,
        'anneal_iter': args.anneal_iter,
        'epochs': args.epoch,

        # Multi-GPU settings
        'num_gpus': num_gpus,
        'gpu_ids': args.gpu_ids,
        'batch_size_per_gpu': args.batch_size_per_gpu,
        'effective_batch_size': args.batch_size_per_gpu * num_gpus,

        # Other
        'seed': args.seed,
        'train_data': args.train,
        'vocab_file': args.vocab,
    }

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags,
        notes=args.wandb_notes,
        config=config,
    )

    print(f"W&B Project:  {args.wandb_project}")
    print(f"W&B Run:      {wandb.run.name}")
    print(f"W&B URL:      {wandb.run.url}")
    print("="*80)
    print()
elif args.wandb:
    print("WARNING: W&B logging requested but wandb package not available")
    print()

# Load vocabulary
vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
args.vocab = PairVocab(vocab)

# Create model
model = HierVAE(args)

# Move to GPU and wrap with DataParallel
device = torch.device('cuda:0')
model = model.to(device)

if num_gpus > 1:
    print(f"Wrapping model with DataParallel across {num_gpus} GPUs...")
    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    print(f"DataParallel enabled: batch will be split across GPUs")
else:
    print("Using single GPU (no DataParallel)")

print()

# Count parameters (access module if wrapped in DataParallel)
model_to_count = model.module if isinstance(model, nn.DataParallel) else model
num_params = sum([x.nelement() for x in model_to_count.parameters()]) / 1000
print(f"Model parameters: {num_params:.1f}K")
print()

# Log model parameters to W&B
if use_wandb:
    wandb.config.update({'num_parameters': num_params * 1000}, allow_val_change=True)

# Initialize parameters
for param in model_to_count.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

# Load checkpoint if specified
if args.load_model:
    print(f"Loading checkpoint: {args.load_model}")
    checkpoint = torch.load(args.load_model, map_location=device)

    if isinstance(checkpoint, tuple):
        model_state, optimizer_state, total_step, beta = checkpoint
    else:
        model_state = checkpoint
        optimizer_state = None
        total_step = 0
        beta = 0

    # Handle loading state dict for DataParallel
    if isinstance(model, nn.DataParallel):
        # If checkpoint was saved with DataParallel
        if list(model_state.keys())[0].startswith('module.'):
            model.load_state_dict(model_state)
        else:
            # If checkpoint was saved without DataParallel, add 'module.' prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                new_state_dict['module.' + k] = v
            model.load_state_dict(new_state_dict)
    else:
        # Single GPU: remove 'module.' prefix if present
        if list(model_state.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(model_state)

    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    print(f"Resumed from step {total_step}, beta={beta:.3f}")

    # Update W&B with resume info
    if use_wandb:
        wandb.config.update({'resumed_from_step': total_step, 'resumed_from_checkpoint': args.load_model}, allow_val_change=True)
else:
    total_step = 0
    beta = 0

# Create save directory
os.makedirs(args.save_dir, exist_ok=True)

# Helper functions
param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

# Training loop
print("="*80)
print("STARTING TRAINING")
print("="*80)
print()

# Calculate effective batch size
effective_batch_size = args.batch_size_per_gpu * num_gpus

meters = np.zeros(6)
for epoch in range(args.epoch):
    print(f"Epoch {epoch + 1}/{args.epoch}")

    # Load dataset with effective batch size
    dataset = DataFolder(args.train, effective_batch_size)

    for batch in tqdm(dataset, desc=f"Epoch {epoch+1}"):
        total_step += 1
        model.zero_grad()

        # Forward pass (DataParallel handles batch splitting automatically)
        loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)

        # If using DataParallel, outputs are gathered and averaged automatically
        # But we need to handle the case where they might be on different devices
        if isinstance(loss, torch.Tensor):
            loss = loss.mean()  # Average across GPUs if needed
        if isinstance(kl_div, torch.Tensor):
            kl_div = kl_div.mean()

        # Backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        # Update meters
        meters = meters + np.array([
            kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div,
            loss.item(),
            wacc.cpu().item() * 100 if isinstance(wacc, torch.Tensor) else wacc * 100,
            iacc.cpu().item() * 100 if isinstance(iacc, torch.Tensor) else iacc * 100,
            tacc.cpu().item() * 100 if isinstance(tacc, torch.Tensor) else tacc * 100,
            sacc.cpu().item() * 100 if isinstance(sacc, torch.Tensor) else sacc * 100
        ])

        # Print progress and log to W&B
        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            pnorm = param_norm(model)
            gnorm = grad_norm(model)

            print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" %
                  (total_step, beta, meters[0], meters[1], meters[2], meters[3],
                   meters[4], meters[5], pnorm, gnorm))
            sys.stdout.flush()

            # Log to W&B
            if use_wandb:
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else scheduler.get_lr()[0]
                wandb.log({
                    'train/kl_divergence': meters[0],
                    'train/loss': meters[1],
                    'train/word_accuracy': meters[2],
                    'train/word_accuracy_2': meters[3],
                    'train/topology_accuracy': meters[4],
                    'train/assembly_accuracy': meters[5],
                    'train/param_norm': pnorm,
                    'train/grad_norm': gnorm,
                    'train/beta': beta,
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch + 1,
                }, step=total_step)

            meters *= 0

        # Save checkpoint
        if total_step % args.save_iter == 0:
            # Save model state (unwrap DataParallel if needed)
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            ckpt = (model_to_save.state_dict(), optimizer.state_dict(), total_step, beta)

            save_path = os.path.join(args.save_dir, f"model.ckpt.{total_step}")
            torch.save(ckpt, save_path)
            print(f"Checkpoint saved: {save_path}")

            # Log checkpoint to W&B
            if use_wandb:
                artifact = wandb.Artifact(
                    name=f"model-step-{total_step}",
                    type="model",
                    description=f"Model checkpoint at step {total_step}, beta={beta:.3f}",
                    metadata={
                        'step': total_step,
                        'beta': beta,
                        'epoch': epoch + 1,
                    }
                )
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)

        # Learning rate annealing
        if total_step % args.anneal_iter == 0:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else scheduler.get_lr()[0]
            print(f"Learning rate updated: {current_lr:.6f}")

        # KL annealing
        if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)

print()
print("="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Total steps: {total_step}")
print(f"Final beta: {beta:.3f}")
print(f"Checkpoints saved in: {args.save_dir}")

# Finish W&B run
if use_wandb:
    wandb.finish()
    print(f"W&B run finished: {wandb.run.url}")
