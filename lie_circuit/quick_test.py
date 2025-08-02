#!/usr/bin/env python3
"""
Quick smoke test for Lie-Circuit experiment
Tests core functionality in <120s on A100
"""

import torch
from transformer_lens import HookedTransformer
import time
import json

def run_smoke(n_prompts: int = 5):
    """Run quick smoke test"""
    start_time = time.time()
    
    print("=== Lie-Circuit Smoke Test ===")
    print(f"Testing with {n_prompts} prompts...")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print("\n1. Loading GPT-2-small...")
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    print("✓ Model loaded")
    
    # Test prompts
    test_examples = [
        {
            "prompt": "What is 2 + 2?",
            "cot": "Let me add: 2 + 2 = 4",
            "answer": "4",
            "faithful": True
        },
        {
            "prompt": "What is 3 × 4?",
            "cot": "Let me multiply: 3 × 4 = 11",  # Wrong
            "answer": "12",
            "faithful": False
        }
    ] * (n_prompts // 2 + 1)
    
    test_examples = test_examples[:n_prompts]
    
    # Test forward passes
    print("\n2. Testing forward passes...")
    for i, ex in enumerate(test_examples):
        text = f"{ex['prompt']}\n{ex['cot']}\nAnswer: {ex['answer']}"
        tokens = model.to_tokens(text, truncate=True)
        
        with torch.no_grad():
            logits = model(tokens)
        
        print(f"  Example {i+1}: shape {logits.shape}, faithful={ex['faithful']}")
    
    # Test hook functionality
    print("\n3. Testing activation extraction...")
    activations = []
    
    def save_hook(value, hook):
        activations.append(value.detach().cpu())
        return value
    
    model.add_hook('blocks.6.hook_resid_post', save_hook)
    
    with torch.no_grad():
        _ = model(tokens)
    
    model.reset_hooks()
    
    print(f"  Captured activations: shape {activations[0].shape}")
    
    # Test simple ablation
    print("\n4. Testing ablation hook...")
    
    def ablate_hook(value, hook):
        value[:, :, :10] = 0  # Zero first 10 dims
        return value
    
    model.add_hook('blocks.9.hook_resid_post', ablate_hook)
    
    with torch.no_grad():
        ablated_logits = model(tokens)
    
    model.reset_hooks()
    
    diff = (logits - ablated_logits).abs().mean()
    print(f"  Ablation effect: mean diff = {diff:.6f}")
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n✓ Smoke test completed in {elapsed:.1f}s")
    
    if elapsed < 120:
        print("✓ Performance check PASSED (<120s)")
    else:
        print("✗ Performance check FAILED (>120s)")
    
    # Memory usage
    if device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"\nGPU memory used: {allocated:.2f} GB")
    
    return 0

def main():
    """Run smoke test"""
    return run_smoke(n_prompts=5)

if __name__ == "__main__":
    exit(main())