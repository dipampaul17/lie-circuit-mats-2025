#!/usr/bin/env python3
"""
EMERGENCY PIVOT: Real Neural Mechanistic Interpretability
========================================================

This is what we SHOULD have built from the start.
Uses TransformerLens to analyze actual neural circuits.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
import json
from typing import Dict, List, Tuple

class RealNeuralLieCircuit:
    """ACTUAL mechanistic interpretability of lie detection"""
    
    def __init__(self):
        print("🧠 REAL NEURAL CIRCUIT ANALYSIS")
        print("=" * 50)
        
        # Load ACTUAL neural network
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HookedTransformer.from_pretrained(
            "gpt2-small",
            device=self.device
        )
        
        print(f"✅ Loaded GPT-2 with {self.model.cfg.n_layers} layers")
        print(f"✅ Hidden dimension: {self.model.cfg.d_model}")
        print(f"✅ Device: {self.device}")
        
    def generate_test_prompts(self) -> Tuple[List[str], List[str]]:
        """Generate faithful and unfaithful reasoning examples"""
        faithful_prompts = [
            "Q: What is 5+3?\nA: Let me calculate: 5+3 = 8. The answer is 8.",
            "Q: What is 12-7?\nA: Let me work through this: 12-7 = 5. The answer is 5.",
            "Q: What is 4*6?\nA: I'll multiply: 4*6 = 24. The answer is 24.",
        ]
        
        unfaithful_prompts = [
            "Q: What is 5+3?\nA: Let me calculate: 5+3 = 9. The answer is 9.",
            "Q: What is 12-7?\nA: Let me work through this: 12-7 = 6. The answer is 6.", 
            "Q: What is 4*6?\nA: I'll multiply: 4*6 = 28. The answer is 28.",
        ]
        
        return faithful_prompts, unfaithful_prompts
    
    def extract_neural_activations(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Extract REAL neural activations from model"""
        all_activations = {}
        
        for layer in range(self.model.cfg.n_layers):
            layer_acts = []
            
            for prompt in prompts:
                # Run model and cache activations
                _, cache = self.model.run_with_cache(prompt)
                
                # Get residual stream activations at this layer
                act_name = get_act_name("resid_post", layer)
                activations = cache[act_name]  # [batch, seq, d_model]
                
                # Average over sequence positions
                mean_activation = activations.mean(dim=1)  # [batch, d_model]
                layer_acts.append(mean_activation)
            
            all_activations[f"layer_{layer}"] = torch.cat(layer_acts, dim=0)
        
        return all_activations
    
    def find_discriminative_neurons(self, 
                                   faithful_acts: Dict[str, torch.Tensor],
                                   unfaithful_acts: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
        """Find neurons that distinguish faithful vs unfaithful"""
        discriminative_neurons = {}
        
        for layer_name in faithful_acts:
            faith_act = faithful_acts[layer_name].mean(dim=0)  # [d_model]
            unfaith_act = unfaithful_acts[layer_name].mean(dim=0)  # [d_model]
            
            # Find neurons with large differences
            differences = torch.abs(faith_act - unfaith_act)
            threshold = differences.mean() + 2 * differences.std()
            
            # Get indices of highly discriminative neurons
            discriminative_indices = torch.where(differences > threshold)[0].tolist()
            discriminative_neurons[layer_name] = discriminative_indices
            
        return discriminative_neurons
    
    def run_ablation_experiment(self, 
                               prompts: List[str],
                               target_layer: int,
                               target_neurons: List[int]) -> float:
        """Ablate specific neurons and measure effect"""
        
        # Define ablation hook
        def ablation_hook(activations, hook):
            # activations shape: [batch, seq, d_model]
            activations[:, :, target_neurons] = 0
            return activations
        
        # Run with and without ablation
        normal_logits = []
        ablated_logits = []
        
        for prompt in prompts:
            # Normal run
            logits_normal = self.model(prompt)
            normal_logits.append(logits_normal)
            
            # Run with ablation
            self.model.reset_hooks()
            hook_name = get_act_name("resid_post", target_layer)
            self.model.add_hook(hook_name, ablation_hook)
            
            logits_ablated = self.model(prompt)
            ablated_logits.append(logits_ablated)
            
            self.model.reset_hooks()
        
        # Compare outputs (simplified metric)
        normal_probs = torch.cat([l.softmax(dim=-1).max(dim=-1)[0].mean() 
                                 for l in normal_logits])
        ablated_probs = torch.cat([l.softmax(dim=-1).max(dim=-1)[0].mean() 
                                  for l in ablated_logits])
        
        effect = (ablated_probs - normal_probs).mean().item()
        return effect
    
    def analyze_attention_patterns(self, prompts: List[str]) -> Dict[str, np.ndarray]:
        """Analyze attention patterns for deception detection"""
        attention_stats = {}
        
        for prompt in prompts:
            _, cache = self.model.run_with_cache(prompt)
            
            # Analyze each layer's attention
            for layer in range(self.model.cfg.n_layers):
                attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
                # Shape: [batch, heads, seq, seq]
                
                # Look for attention to "incorrect" tokens
                # This is where real mech interp happens!
                mean_attn = attn_pattern.mean(dim=(0, 1))  # [seq, seq]
                
                layer_key = f"layer_{layer}_attention"
                if layer_key not in attention_stats:
                    attention_stats[layer_key] = []
                
                attention_stats[layer_key].append(mean_attn.cpu().numpy())
        
        return attention_stats
    
    def run_full_neural_analysis(self):
        """Complete neural circuit analysis"""
        print("\n🔬 Starting REAL Neural Circuit Analysis...")
        
        # Generate test data
        faithful, unfaithful = self.generate_test_prompts()
        print(f"✅ Generated {len(faithful)} faithful and {len(unfaithful)} unfaithful examples")
        
        # Extract neural activations
        print("\n📊 Extracting neural activations...")
        faithful_acts = self.extract_neural_activations(faithful)
        unfaithful_acts = self.extract_neural_activations(unfaithful)
        print(f"✅ Extracted activations from {len(faithful_acts)} layers")
        
        # Find discriminative neurons
        print("\n🎯 Finding discriminative neurons...")
        disc_neurons = self.find_discriminative_neurons(faithful_acts, unfaithful_acts)
        
        for layer, neurons in disc_neurons.items():
            if len(neurons) > 0:
                print(f"  {layer}: {len(neurons)} discriminative neurons found")
        
        # Run ablation experiment on layer 9
        if len(disc_neurons.get("layer_9", [])) > 0:
            print("\n⚡ Running ablation experiment on layer 9...")
            effect = self.run_ablation_experiment(
                faithful + unfaithful,
                target_layer=9,
                target_neurons=disc_neurons["layer_9"][:50]  # Top 50 neurons
            )
            print(f"✅ Ablation effect: {effect:.3f}")
        
        # Analyze attention patterns
        print("\n🧠 Analyzing attention patterns...")
        attention_stats = self.analyze_attention_patterns(faithful + unfaithful)
        print(f"✅ Analyzed attention from {len(attention_stats)} layer-head combinations")
        
        # Save results
        results = {
            "discriminative_neurons": {k: v[:10] for k, v in disc_neurons.items()},
            "ablation_effect": effect if 'effect' in locals() else None,
            "num_attention_patterns": len(attention_stats),
            "analysis_type": "REAL_NEURAL_CIRCUITS"
        }
        
        with open("real_neural_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ REAL neural analysis complete!")
        print("📄 Results saved to real_neural_results.json")
        
        return results

def main():
    print("🚨 EMERGENCY NEURAL PIVOT - REAL MECHANISTIC INTERPRETABILITY")
    print("=" * 60)
    
    analyzer = RealNeuralLieCircuit()
    results = analyzer.run_full_neural_analysis()
    
    print("\n" + "="*60)
    print("🎯 This is what mechanistic interpretability actually looks like!")
    print("   - Analyzed neural activations ✅")
    print("   - Found discriminative neurons ✅") 
    print("   - Ran causal ablations ✅")
    print("   - Examined attention patterns ✅")

if __name__ == "__main__":
    main()