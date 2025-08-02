#!/usr/bin/env python3
"""
Train Cross-Layer Transcoder (CLT) from layer 6 to layer 9
Early stops when FVU < 0.15 or 3 hours elapsed
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import time
import hydra
from omegaconf import DictConfig
import os

class CrossLayerTranscoder(nn.Module):
    """Cross-layer transcoder mapping layer 6 → layer 9 representations"""
    
    def __init__(self, d_model: int = 768, hidden_dim: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        # Improved transcoder with residual connection and dropout
        self.transcoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Initialize weights with smaller variance for stability
        for module in self.transcoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Apply transcoder with residual connection
        transcoded = self.transcoder(x)
        return x + self.residual_weight * transcoded

class CLTDataset(Dataset):
    """Dataset for cross-layer transcoding"""
    
    def __init__(
        self, 
        examples: List[Dict], 
        model: HookedTransformer,
        src_layer: int = 6,
        tgt_layer: int = 9
    ):
        self.examples = examples
        self.model = model
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.src_acts = []
        self.tgt_acts = []
        self.labels = []
        
        # Pre-compute activations
        print(f"Computing activations for layers {src_layer} → {tgt_layer}...")
        self._compute_activations()
    
    def _compute_activations(self):
        self.model.eval()
        
        with torch.no_grad():
            for ex in tqdm(self.examples):
                # Tokenize prompt + CoT
                text = f"{ex['prompt']}\n{ex['cot']}"
                tokens = self.model.to_tokens(text, truncate=True)
                
                # Get activations at both layers
                _, cache = self.model.run_with_cache(tokens)
                
                # Get residual stream at each layer
                src_act = cache[f'blocks.{self.src_layer}.hook_resid_post']  # [batch, seq, dim]
                tgt_act = cache[f'blocks.{self.tgt_layer}.hook_resid_post']
                
                # Average pool over sequence (or take last token)
                src_pooled = src_act[:, -1, :]  # Last token
                tgt_pooled = tgt_act[:, -1, :]
                
                self.src_acts.append(src_pooled.cpu())
                self.tgt_acts.append(tgt_pooled.cpu())
                self.labels.append(1 if not ex.get('faithful', True) else 0)
        
        self.src_acts = torch.cat(self.src_acts, dim=0)
        self.tgt_acts = torch.cat(self.tgt_acts, dim=0)
        self.labels = torch.tensor(self.labels)
    
    def __len__(self):
        return len(self.src_acts)
    
    def __getitem__(self, idx):
        return self.src_acts[idx], self.tgt_acts[idx], self.labels[idx]

def compute_fvu(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Fraction of Variance Unexplained"""
    mse = torch.mean((predictions - targets) ** 2)
    variance = torch.var(targets)
    return (mse / (variance + 1e-8)).item()

def train_clt(
    config: DictConfig,
    dataset: Dataset,
    device: str = 'cuda'
) -> Tuple[CrossLayerTranscoder, Dict]:
    """Train the cross-layer transcoder"""
    
    # Initialize model
    clt = CrossLayerTranscoder(
        d_model=768,  # GPT-2-small
        hidden_dim=config.hidden
    ).to(device)
    
            # Create optimizer with weight decay
        optimizer = optim.AdamW(
            clt.parameters(), 
            lr=config.lr, 
            weight_decay=getattr(config, 'weight_decay', 1e-5)
        )
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.max_steps, 
            eta_min=config.lr * 0.01
        )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch, 
        shuffle=True,
        num_workers=2
    )
    
    # Training metrics
    metrics = {
        'steps': [],
        'kl_loss': [],
        'fvu_loss': [], 
        'total_loss': [],
        'fvu': []
    }
    
    start_time = time.time()
    step = 0
    best_fvu = float('inf')
    
    print(f"Training CLT {config.src_layer}→{config.tgt_layer}...")
    pbar = tqdm(total=config.max_steps)
    
    while step < config.max_steps:
        for src_batch, tgt_batch, labels in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Forward pass
            pred_tgt = clt(src_batch)
            
            # KL divergence loss (treating as distributions)
            kl_loss = nn.functional.kl_div(
                torch.log_softmax(pred_tgt, dim=-1),
                torch.softmax(tgt_batch, dim=-1),
                reduction='batchmean'
            )
            
            # FVU loss (MSE normalized by variance)
            fvu_loss = nn.functional.mse_loss(pred_tgt, tgt_batch)
            
            # Combined loss
            total_loss = (config.loss_mix.kl * kl_loss + 
                         config.loss_mix.fvu * fvu_loss)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Compute FVU metric
            fvu = compute_fvu(pred_tgt, tgt_batch)
            
            # Log metrics
            if step % 50 == 0:
                metrics['steps'].append(step)
                metrics['kl_loss'].append(kl_loss.item())
                metrics['fvu_loss'].append(fvu_loss.item())
                metrics['total_loss'].append(total_loss.item())
                metrics['fvu'].append(fvu)
                
                pbar.set_description(
                    f"FVU: {fvu:.3f} | KL: {kl_loss:.3f} | Total: {total_loss:.3f}"
                )
            
            # Checkpoint
            if step % 500 == 0:
                checkpoint = {
                    'step': step,
                    'model_state': clt.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'metrics': metrics,
                    'config': dict(config)
                }
                torch.save(checkpoint, f'checkpoints/clt_step_{step}.pt')
            
            # Check early stopping
            if fvu < best_fvu:
                best_fvu = fvu
                torch.save(clt.state_dict(), 'clt_weights.pt')
            
            if fvu < config.early_stop_fvu:
                print(f"\nEarly stopping: FVU {fvu:.3f} < {config.early_stop_fvu}")
                break
            
            # Check time limit
            if (time.time() - start_time) > 3 * 3600:  # 3 hours
                print("\nTime limit reached (3 hours)")
                break
            
            step += 1
            pbar.update(1)
            
            if step >= config.max_steps:
                break
    
    pbar.close()
    
    # Save final metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training completed in {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best FVU: {best_fvu:.3f}")
    
    return clt, metrics

@hydra.main(version_base=None, config_path="configs", config_name="clt_config")
def main(cfg: DictConfig):
    """Main training function"""
    print("=== Cross-Layer Transcoder Training ===")
    print(f"Config: {cfg}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load model
    print("Loading GPT-2-small...")
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    
    # Load examples
    print("Loading dataset...")
    examples = []
    with open('dev_tagged.jsonl' if os.path.exists('dev_tagged.jsonl') else 'dev.jsonl', 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Create dataset
    dataset = CLTDataset(
        examples, 
        model,
        src_layer=cfg.src_layer,
        tgt_layer=cfg.tgt_layer
    )
    
    # Train CLT
    clt, metrics = train_clt(cfg, dataset, device)
    
    # Log budget
    with open('budget.log', 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: CLT training completed, credits_remaining: TBD\n")
    
    print("\nCLT training complete!")

if __name__ == "__main__":
    main()