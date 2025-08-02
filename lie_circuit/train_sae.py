#!/usr/bin/env python3
"""
Train Sparse Autoencoders (SAE) on GPT-2-small layers 6 & 9
Identifies high-Δ activation dims between faithful/unfaithful examples
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import time

class SAE(nn.Module):
    """Sparse Autoencoder with L0 regularization"""
    
    def __init__(self, input_dim: int, hidden_dim: int = None, sparsity: float = 0.015):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim * 4
        self.sparsity_target = sparsity
        
        # Encoder and decoder
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        
    def forward(self, x):
        # Encode
        hidden = self.encoder(x)
        activated = torch.relu(hidden)
        
        # Decode
        reconstructed = self.decoder(activated)
        
        return reconstructed, activated
    
    def get_sparsity(self, activated):
        """Calculate sparsity (fraction of non-zero activations)"""
        return (activated > 0).float().mean()

class LieDataset(Dataset):
    """Dataset for loading activations from faithful/unfaithful examples"""
    
    def __init__(self, examples: List[Dict], model: HookedTransformer, layer: int):
        self.examples = examples
        self.model = model
        self.layer = layer
        self.activations = []
        self.labels = []  # 1 for unfaithful, 0 for faithful
        
        # Pre-compute activations
        print(f"Computing activations for layer {layer}...")
        self._compute_activations()
    
    def _compute_activations(self):
        self.model.eval()
        
        with torch.no_grad():
            for ex in tqdm(self.examples):
                # Tokenize prompt + CoT
                text = f"{ex['prompt']}\n{ex['cot']}"
                tokens = self.model.to_tokens(text, truncate=True)
                
                # Get activations at specified layer
                _, cache = self.model.run_with_cache(tokens)
                
                # Average pool over sequence length
                act = cache[f'blocks.{self.layer}.hook_resid_post']  # [batch, seq, dim]
                act_pooled = act.mean(dim=1)  # [batch, dim]
                
                self.activations.append(act_pooled.cpu())
                self.labels.append(1 if not ex.get('faithful', True) else 0)
        
        self.activations = torch.cat(self.activations, dim=0)
        self.labels = torch.tensor(self.labels)
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]

def train_sae(
    dataset: Dataset,
    input_dim: int,
    hidden_dim: int = None,
    sparsity: float = 0.015,
    lambda_l0: float = 5e-5,
    lr: float = 5e-4,
    max_steps: int = 10000,
    max_time: float = 3600,
    device: str = 'cuda'
) -> Tuple[SAE, Dict]:
    """Train a sparse autoencoder"""
    
    sae = SAE(input_dim, hidden_dim, sparsity).to(device)
    optimizer = optim.AdamW(sae.parameters(), lr=lr)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training metrics
    metrics = {
        'steps': [],
        'recon_loss': [],
        'sparsity': [],
        'fvu': []  # Fraction of variance unexplained
    }
    
    start_time = time.time()
    step = 0
    
    print(f"Training SAE with {sae.hidden_dim} hidden dims...")
    pbar = tqdm(total=max_steps)
    
    while step < max_steps and (time.time() - start_time) < max_time:
        for batch_acts, batch_labels in dataloader:
            batch_acts = batch_acts.to(device)
            
            # Forward pass
            recon, hidden = sae(batch_acts)
            
            # Reconstruction loss
            recon_loss = nn.functional.mse_loss(recon, batch_acts)
            
            # Sparsity loss (L0 approximation)
            sparsity = sae.get_sparsity(hidden)
            sparsity_loss = lambda_l0 * torch.abs(sparsity - sae.sparsity_target)
            
            # Total loss
            loss = recon_loss + sparsity_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate FVU
            variance = torch.var(batch_acts)
            fvu = recon_loss / (variance + 1e-8)
            
            # Log metrics
            if step % 100 == 0:
                metrics['steps'].append(step)
                metrics['recon_loss'].append(recon_loss.item())
                metrics['sparsity'].append(sparsity.item())
                metrics['fvu'].append(fvu.item())
                
                pbar.set_description(
                    f"FVU: {fvu:.3f} | Sparsity: {sparsity:.3%} | Loss: {loss:.4f}"
                )
            
            step += 1
            pbar.update(1)
            
            if step >= max_steps or (time.time() - start_time) >= max_time:
                break
    
    pbar.close()
    print(f"Training completed in {time.time() - start_time:.1f}s")
    
    return sae, metrics

def find_high_delta_features(
    sae: SAE, 
    dataset: Dataset,
    n_features: int = 100,
    device: str = 'cuda'
) -> List[int]:
    """Find features with highest activation difference between faithful/unfaithful"""
    
    sae.eval()
    faithful_acts = []
    unfaithful_acts = []
    
    with torch.no_grad():
        for acts, label in DataLoader(dataset, batch_size=32):
            acts = acts.to(device)
            _, hidden = sae(acts)
            
            # Separate by label
            faithful_mask = (label == 0).to(device)
            unfaithful_mask = (label == 1).to(device)
            
            if faithful_mask.any():
                faithful_acts.append(hidden[faithful_mask].mean(dim=0, keepdim=True))
            if unfaithful_mask.any():
                unfaithful_acts.append(hidden[unfaithful_mask].mean(dim=0, keepdim=True))
    
    # Average activations
    faithful_mean = torch.cat(faithful_acts).mean(dim=0)
    unfaithful_mean = torch.cat(unfaithful_acts).mean(dim=0)
    
    # Compute delta (absolute difference)
    delta = torch.abs(unfaithful_mean - faithful_mean)
    
    # Get top features
    top_indices = torch.argsort(delta, descending=True)[:n_features]
    
    return top_indices.cpu().tolist()

def main():
    """Train SAEs on layers 6 and 9"""
    print("=== Lie-Circuit SAE Training ===")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    results = {}
    
    for layer in [6, 9]:
        print(f"\n=== Training SAE for Layer {layer} ===")
        
        # Create dataset
        dataset = LieDataset(examples, model, layer)
        input_dim = model.cfg.d_model  # 768 for GPT-2-small
        
        # Train SAE
        sae, metrics = train_sae(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=input_dim * 4,  # 3072
            sparsity=0.015,
            lambda_l0=5e-5,
            lr=5e-4,
            max_steps=10000,
            max_time=3600,
            device=device
        )
        
        # Find high-delta features
        print("Finding high-delta features...")
        top_features = find_high_delta_features(sae, dataset, n_features=100, device=device)
        
        # Check FVU
        final_fvu = metrics['fvu'][-1] if metrics['fvu'] else 1.0
        print(f"Final FVU: {final_fvu:.3f}")
        
        if final_fvu > 0.2:
            print(f"WARNING: FVU {final_fvu:.3f} exceeds 0.2 threshold!")
        
        # Save results
        results[f'layer_{layer}'] = {
            'final_fvu': final_fvu,
            'final_sparsity': metrics['sparsity'][-1] if metrics['sparsity'] else 0,
            'top_features': top_features,
            'metrics': metrics
        }
        
        # Save SAE weights
        torch.save(sae.state_dict(), f'sae_layer_{layer}.pt')
    
    # Save all results
    with open('sae_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nSAE training complete!")
    print(f"Layer 6 FVU: {results['layer_6']['final_fvu']:.3f}")
    print(f"Layer 9 FVU: {results['layer_9']['final_fvu']:.3f}")

if __name__ == "__main__":
    main()