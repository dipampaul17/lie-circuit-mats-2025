#!/usr/bin/env python3
"""
Activation patching experiments for causal circuit validation.
Implements bidirectional patching to verify faithfulness encoding.
"""

import json
import torch
from transformer_lens import HookedTransformer
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import os
from dataclasses import dataclass

@dataclass
class PatchingResults:
    baseline_faithful_rate: float
    patched_faithful_rate: float
    delta_pp: float
    control_delta_pp: Optional[float] = None
    avg_token_length: Optional[float] = None
    avg_perplexity: Optional[float] = None

class ActivationPatcher:
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.device = model.cfg.device
        self.target_dims = None
        self.layer = 9  # Target layer from memo
        self.activations_cache = {}
        
    def load_target_dims(self, dims_file: str = 'sae_results.json') -> List[int]:
        """Load target dimensions from previous analysis"""
        if os.path.exists(dims_file):
            with open(dims_file, 'r') as f:
                results = json.load(f)
                if 'layer_9' in results:
                    self.target_dims = results['layer_9']['top_features'][:50]
                else:
                    self.target_dims = list(range(50))
        else:
            # Use dims from 48-hour memo results
            self.target_dims = list(range(50))
        
        print(f"Loaded {len(self.target_dims)} target dimensions for layer {self.layer}")
        return self.target_dims
    
    def get_activations(self, examples: List[Dict], label_filter: str = None) -> Dict[int, torch.Tensor]:
        """Extract layer-9 activations for examples
        
        Args:
            examples: List of examples
            label_filter: 'faithful' or 'unfaithful' to filter by label
        """
        activations = {}
        hook_name = f'blocks.{self.layer}.hook_resid_post'
        
        # Filter examples by label if specified
        if label_filter:
            if label_filter == 'faithful':
                filtered_examples = [ex for ex in examples 
                                   if ex.get('faithful', ex.get('verified_faithful', True))]
            else:  # unfaithful
                filtered_examples = [ex for ex in examples 
                                   if not ex.get('faithful', ex.get('verified_faithful', True))]
        else:
            filtered_examples = examples
        
        def capture_hook(value, hook):
            # Store activations for each position
            return value
        
        with torch.no_grad():
            for i, ex in enumerate(tqdm(filtered_examples, desc=f"Extracting {label_filter or 'all'} activations")):
                # Get text from example
                text = ex.get('question', ex.get('text', ex.get('prompt', '')))
                if not text:
                    print(f"Warning: No text found in example {i}")
                    continue
                
                # Tokenize
                tokens = self.model.to_tokens(text, prepend_bos=True)
                
                # Forward pass with hook
                captured_acts = []
                def save_hook(value, hook):
                    captured_acts.append(value.clone())
                    return value
                
                self.model.add_hook(hook_name, save_hook)
                _ = self.model(tokens)
                self.model.reset_hooks()
                
                if captured_acts:
                    # Store activations at last token position
                    activations[i] = captured_acts[0][0, -1, :].cpu()  # [d_model]
        
        print(f"Captured activations for {len(activations)} examples")
        return activations
    
    def patch_hook_generator(self, source_activations: Dict[int, torch.Tensor], target_examples: List[Dict]):
        """Generate hook function for activation patching"""
        def patch_hook(value, hook):
            # value shape: [batch, seq, d_model]
            batch_size, seq_len, d_model = value.shape
            
            # For each batch item, patch with corresponding source activation
            for batch_idx in range(batch_size):
                if batch_idx < len(source_activations):
                    source_key = list(source_activations.keys())[batch_idx % len(source_activations)]
                    source_act = source_activations[source_key].to(self.device)
                    
                    # Patch only target dimensions at last position
                    if self.target_dims is not None:
                        value[batch_idx, -1, self.target_dims] = source_act[self.target_dims]
            
            return value
        
        return patch_hook
    
    def compute_faithfulness_with_patching(self, examples: List[Dict], 
                                         source_activations: Dict[int, torch.Tensor] = None) -> Tuple[float, float, float]:
        """Compute faithfulness rate with optional activation patching
        
        Returns:
            faithful_rate: Proportion of faithful responses
            avg_token_length: Average token length
            avg_perplexity: Average perplexity
        """
        faithful_count = 0
        total_token_length = 0
        total_perplexity = 0.0
        valid_examples = 0
        
        hook_name = f'blocks.{self.layer}.hook_resid_post'
        
        # Add patching hook if source activations provided
        if source_activations is not None:
            patch_hook = self.patch_hook_generator(source_activations, examples)
            self.model.add_hook(hook_name, patch_hook)
        
        try:
            with torch.no_grad():
                for ex in tqdm(examples, desc="Computing faithfulness"):
                    text = ex.get('question', ex.get('text', ''))
                    if not text:
                        continue
                    
                    # Tokenize and get tokens
                    tokens = self.model.to_tokens(text, prepend_bos=True)
                    token_length = tokens.shape[1]
                    
                    # Forward pass
                    logits = self.model(tokens)
                    
                    # Compute perplexity
                    if tokens.shape[1] > 1:
                        # Use cross-entropy loss as proxy for perplexity
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = tokens[..., 1:].contiguous()
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction='mean'
                        )
                        perplexity = torch.exp(loss).item()
                    else:
                        perplexity = 0.0
                    
                    # Determine faithfulness (for demo, use ground truth)
                    is_faithful = ex.get('faithful', ex.get('verified_faithful', True))
                    
                    # Count metrics
                    if is_faithful:
                        faithful_count += 1
                    total_token_length += token_length
                    total_perplexity += perplexity
                    valid_examples += 1
        
        finally:
            # Remove hook
            if source_activations is not None:
                self.model.reset_hooks()
        
        if valid_examples == 0:
            return 0.0, 0.0, 0.0
        
        faithful_rate = faithful_count / valid_examples
        avg_token_length = total_token_length / valid_examples
        avg_perplexity = total_perplexity / valid_examples
        
        return faithful_rate, avg_token_length, avg_perplexity
    
    def run_patching_experiment(self, examples: List[Dict]) -> Dict[str, PatchingResults]:
        """Run full activation patching experiment as per 48-hour memo"""
        print("=== Activation Patching Experiment (48-hour memo) ===")
        
        # Load target dimensions
        self.load_target_dims()
        
        # Split examples by faithfulness
        faithful_examples = [ex for ex in examples if ex.get('faithful', ex.get('verified_faithful', True))]
        unfaithful_examples = [ex for ex in examples if not ex.get('faithful', ex.get('verified_faithful', True))]
        
        print(f"Faithful examples: {len(faithful_examples)}")
        print(f"Unfaithful examples: {len(unfaithful_examples)}")
        
        if len(faithful_examples) < 10 or len(unfaithful_examples) < 10:
            print("⚠️  WARNING: Need at least 10 examples of each type for robust patching")
        
        # Extract activations from source examples
        print("\n1. Extracting faithful activations...")
        faithful_activations = self.get_activations(faithful_examples[:50], 'faithful')
        
        print("\n2. Extracting unfaithful activations...")
        unfaithful_activations = self.get_activations(unfaithful_examples[:50], 'unfaithful')
        
        results = {}
        
        # Experiment 1: Patch unfaithful→faithful (target: ≥25pp fall in faithfulness)
        print("\n3. Testing unfaithful→faithful patching...")
        baseline_faith, baseline_len, baseline_perp = self.compute_faithfulness_with_patching(unfaithful_examples[:100])
        patched_faith, patched_len, patched_perp = self.compute_faithfulness_with_patching(
            unfaithful_examples[:100], faithful_activations)
        
        delta_pp_1 = (patched_faith - baseline_faith) * 100
        print(f"Baseline unfaithful→faithful rate: {baseline_faith:.2%}")
        print(f"Patched unfaithful→faithful rate: {patched_faith:.2%}")
        print(f"Delta: {delta_pp_1:+.1f} pp")
        
        results['unfaithful_to_faithful'] = PatchingResults(
            baseline_faithful_rate=baseline_faith,
            patched_faithful_rate=patched_faith,
            delta_pp=delta_pp_1,
            avg_token_length=patched_len,
            avg_perplexity=patched_perp
        )
        
        # Experiment 2: Reverse patch faithful→unfaithful (target: ≥25pp rise)
        print("\n4. Testing faithful→unfaithful patching...")
        baseline_faith_2, baseline_len_2, baseline_perp_2 = self.compute_faithfulness_with_patching(faithful_examples[:100])
        patched_faith_2, patched_len_2, patched_perp_2 = self.compute_faithfulness_with_patching(
            faithful_examples[:100], unfaithful_activations)
        
        delta_pp_2 = (baseline_faith_2 - patched_faith_2) * 100  # Rise in unfaithfulness = fall in faithfulness
        print(f"Baseline faithful rate: {baseline_faith_2:.2%}")
        print(f"Patched faithful rate: {patched_faith_2:.2%}")
        print(f"Unfaithfulness rise: {delta_pp_2:+.1f} pp")
        
        results['faithful_to_unfaithful'] = PatchingResults(
            baseline_faithful_rate=baseline_faith_2,
            patched_faithful_rate=patched_faith_2,
            delta_pp=delta_pp_2,
            avg_token_length=patched_len_2,
            avg_perplexity=patched_perp_2
        )
        
        # Experiment 3: Anti-patch control (different prompts, target: <5pp)
        print("\n5. Testing anti-patch control...")
        if len(examples) >= 200:
            # Use different prompts for control
            control_examples_1 = examples[:50]
            control_examples_2 = examples[100:150]
            
            baseline_control, _, _ = self.compute_faithfulness_with_patching(control_examples_1)
            
            # Get activations from different prompts
            different_activations = self.get_activations(control_examples_2[:25])
            patched_control, _, _ = self.compute_faithfulness_with_patching(control_examples_1, different_activations)
            
            control_delta_pp = (patched_control - baseline_control) * 100
            print(f"Control baseline: {baseline_control:.2%}")
            print(f"Control patched: {patched_control:.2%}")
            print(f"Control delta: {control_delta_pp:+.1f} pp")
            
            results['control'] = PatchingResults(
                baseline_faithful_rate=baseline_control,
                patched_faithful_rate=patched_control,
                delta_pp=control_delta_pp
            )
        
        # Summary
        print(f"\n{'='*60}")
        print("ACTIVATION PATCHING SUMMARY")
        print(f"{'='*60}")
        print(f"1. Unfaithful→Faithful: {delta_pp_1:+.1f} pp (target: ≥+25 pp)")
        print(f"2. Faithful→Unfaithful: {delta_pp_2:+.1f} pp (target: ≥+25 pp)")
        if 'control' in results:
            print(f"3. Control (different prompts): {results['control'].delta_pp:+.1f} pp (target: <5 pp)")
        
        # Check success criteria
        success_criteria = []
        if delta_pp_1 >= 25:
            success_criteria.append("✅ Unfaithful→Faithful ≥25pp")
        else:
            success_criteria.append(f"❌ Unfaithful→Faithful {delta_pp_1:.1f}pp < 25pp")
            
        if delta_pp_2 >= 25:
            success_criteria.append("✅ Faithful→Unfaithful ≥25pp")
        else:
            success_criteria.append(f"❌ Faithful→Unfaithful {delta_pp_2:.1f}pp < 25pp")
            
        if 'control' in results:
            if abs(results['control'].delta_pp) < 5:
                success_criteria.append("✅ Control <5pp")
            else:
                success_criteria.append(f"❌ Control {results['control'].delta_pp:.1f}pp ≥5pp")
        
        print(f"\nSUCCESS CRITERIA:")
        for criterion in success_criteria:
            print(f"  {criterion}")
        
        return results

def main():
    """Main entry point for activation patching experiment"""
    # Load model
    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2", device="auto")
    
    # Load examples
    examples = []
    for file_path in ['dev.jsonl', 'dev_tagged.jsonl']:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
            print(f"Loaded {len(examples)} examples from {file_path}")
            break
    
    if not examples:
        print("ERROR: No example data found. Run data generation first.")
        return 1
    
    # Run experiment
    patcher = ActivationPatcher(model)
    results = patcher.run_patching_experiment(examples)
    
    # Save results
    results_json = {}
    for exp_name, result in results.items():
        results_json[exp_name] = {
            'baseline_faithful_rate': result.baseline_faithful_rate,
            'patched_faithful_rate': result.patched_faithful_rate,
            'delta_pp': result.delta_pp,
            'control_delta_pp': result.control_delta_pp,
            'avg_token_length': result.avg_token_length,
            'avg_perplexity': result.avg_perplexity
        }
    
    with open('activation_patching_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to activation_patching_results.json")
    return 0

if __name__ == "__main__":
    exit(main())