#!/usr/bin/env python3
"""
Simplified CLT training for Lie-Circuit experiment
Cross-layer transcoder from layer 6 to layer 9
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import time
import os

class SimpleGPT2Wrapper:
    """Simple wrapper around GPT2 for activation extraction"""
    def __init__(self, device='cuda'):
        self.device = device
        self.model = GPT2Model.from_pretrained('gpt2').to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
    def get_activations(self, text: str, layer: int):
        """Get activations from specified layer"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # layer + 1 because index 0 is embeddings
            hidden_states = outputs.hidden_states[layer + 1]
            return hidden_states

class CrossLayerTranscoder(nn.Module):
    """Cross-layer transcoder mapping layer 6 → layer 9 representations"""
    
    def __init__(self, d_model: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        # Two-layer MLP transcoder
        self.transcoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        # Initialize weights
        for module in self.transcoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.transcoder(x)

class CLTDataset(Dataset):
    """Dataset for cross-layer transcoding"""
    
    def __init__(
        self, 
        examples: List[Dict], 
        model: SimpleGPT2Wrapper,
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
        for ex in tqdm(self.examples):
            # Tokenize prompt + CoT
            text = f"{ex['prompt']}\n{ex['cot']}"
            
            # Get activations at both layers
            src_act = self.model.get_activations(text, self.src_layer)
            tgt_act = self.model.get_activations(text, self.tgt_layer)
            
            # Take last token
            src_pooled = src_act[:, -1, :]
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
    dataset: Dataset,
    hidden_dim: int = 512,
    lr: float = 3e-4,
    batch_size: int = 32,
    max_steps: int = 4000,
    early_stop_fvu: float = 0.15,
    device: str = 'cuda'
) -> Tuple[CrossLayerTranscoder, Dict]:
    """Train the cross-layer transcoder"""
    
    # Initialize model
    clt = CrossLayerTranscoder(d_model=768, hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(clt.parameters(), lr=lr)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training metrics
    metrics = {
        'steps': [],
        'mse_loss': [],
        'fvu': []
    }
    
    start_time = time.time()
    step = 0
    best_fvu = float('inf')
    
    print(f"Training CLT 6→9 with hidden_dim={hidden_dim}...")
    pbar = tqdm(total=max_steps)
    
    while step < max_steps:
        for src_batch, tgt_batch, labels in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Forward pass
            pred_tgt = clt(src_batch)
            
            # MSE loss
            loss = nn.functional.mse_loss(pred_tgt, tgt_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute FVU metric
            fvu = compute_fvu(pred_tgt, tgt_batch)
            
            # Log metrics
            if step % 50 == 0:
                metrics['steps'].append(step)
                metrics['mse_loss'].append(loss.item())
                metrics['fvu'].append(fvu)
                
                pbar.set_description(f"FVU: {fvu:.3f} | Loss: {loss:.4f}")
            
            # Save best model
            if fvu < best_fvu:
                best_fvu = fvu
                torch.save(clt.state_dict(), 'clt_weights.pt')
            
            # Early stopping
            if fvu < early_stop_fvu:
                print(f"\nEarly stopping: FVU {fvu:.3f} < {early_stop_fvu}")
                pbar.close()
                return clt, metrics
            
            step += 1
            pbar.update(1)
            
            if step >= max_steps:
                break
    
    pbar.close()
    
    print(f"Training completed in {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best FVU: {best_fvu:.3f}")
    
    return clt, metrics

def main():
    """Main training function"""
    print("=== Cross-Layer Transcoder Training (Simplified) ===")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load model
    print("Loading GPT-2...")
    model = SimpleGPT2Wrapper(device=device)
    
    # Load examples
    print("Loading dataset...")
    examples = []
    with open('dev_tagged.jsonl' if os.path.exists('dev_tagged.jsonl') else 'dev.jsonl', 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Create dataset
    dataset = CLTDataset(examples, model, src_layer=6, tgt_layer=9)
    
    # Train CLT with reduced steps for quick testing
    clt, metrics = train_clt(
        dataset=dataset,
        hidden_dim=512,
        lr=3e-4,
        batch_size=16,  # Smaller batch for limited data
        max_steps=1000,  # Reduced for testing
        early_stop_fvu=0.15,
        device=device
    )
    
    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Log budget
    with open('budget.log', 'a') as f:
        import datetime
        f.write(f"{datetime.datetime.now()}: CLT training completed, FVU={metrics['fvu'][-1]:.3f}\n")
    
    print("\nCLT training complete!")
    print(f"Final FVU: {metrics['fvu'][-1]:.3f}")

if __name__ == "__main__":
    main()