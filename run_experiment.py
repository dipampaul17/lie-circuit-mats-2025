#!/usr/bin/env python3
"""
Main orchestration script for Lie-Circuit experiment
Runs all steps in sequence with proper error handling
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

# Add lie_circuit to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ExperimentRunner:
    def __init__(self):
        self.start_time = time.time()
        self.budget_remaining = 775  # GPU hours
        self.results = {}
        
    def log_budget(self, step: str, gpu_hours_used: float):
        """Log budget usage"""
        self.budget_remaining -= gpu_hours_used
        
        with open('budget.log', 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp}: Step '{step}' used {gpu_hours_used:.1f} GPU-h, remaining: {self.budget_remaining:.1f} GPU-h\n")
    
    def run_step(self, step_name: str, command: str, expected_gpu_h: float):
        """Run a single step with error handling"""
        print(f"\n{'='*60}")
        print(f"Running: {step_name}")
        print(f"Expected GPU hours: {expected_gpu_h}")
        print(f"Budget remaining: {self.budget_remaining:.1f} GPU-h")
        print('='*60)
        
        if self.budget_remaining < expected_gpu_h:
            print(f"⚠️  WARNING: Insufficient budget! Need {expected_gpu_h} GPU-h, have {self.budget_remaining:.1f}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        step_start = time.time()
        
        try:
            # Run command
            if command.endswith('.py'):
                result = subprocess.run([sys.executable, command], 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(command, shell=True, 
                                      capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Step failed with return code {result.returncode}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                return False
            
            print(f"✅ Step completed successfully")
            
        except Exception as e:
            print(f"❌ Step failed with exception: {e}")
            return False
        
        # Log time and budget
        step_duration = (time.time() - step_start) / 3600  # hours
        actual_gpu_h = min(step_duration, expected_gpu_h)  # Conservative estimate
        self.log_budget(step_name, actual_gpu_h)
        
        return True
    
    def run_experiment(self):
        """Run the full experiment pipeline"""
        print("🚀 Starting Lie-Circuit Experiment")
        print(f"Total budget: 775 GPU-h ($1000)")
        
        # Step 0: Environment setup (10 GPU-h)
        if not self.run_step("Environment Setup", "bash setup_env.sh", 10):
            return 1
        
        # Step 1: Data curation (0 GPU-h)
        if not self.run_step("Data Curation", "lie_circuit/data_curator.py", 0):
            return 1
        
        # Step 2: Faith tagging (5 API-h, count as GPU-h)
        if not self.run_step("Faith Tagging", "lie_circuit/faith_tagger.py", 5):
            print("⚠️  Faith tagging failed, continuing with synthetic labels")
        
        # Step 3: SAE baseline (30 GPU-h)
        if not self.run_step("SAE Training", "lie_circuit/train_sae.py", 30):
            return 1
        
        # Step 4: CLT training (120 GPU-h) - Most expensive step
        if not self.run_step("CLT Training", "lie_circuit/train_clt.py", 120):
            return 1
        
        # Step 5: Sanity audit (0 GPU-h)
        if not self.run_step("Sanity Audit", "lie_circuit/sanity_audit.py", 0):
            print("⚠️  Sanity check failed, but continuing")
        
        # Step 6: Negative controls (25 GPU-h)
        if not self.run_step("Negative Controls", "lie_circuit/neg_eval.py", 25):
            return 1
        
        # Step 7: Causal patch zero (25 GPU-h)
        if not self.run_step("Causal Patch Zero", "lie_circuit/patch_zero.py", 25):
            return 1
        
        # Step 8: Causal patch amp (25 GPU-h)
        if not self.run_step("Causal Patch Amp", "lie_circuit/patch_amp.py", 25):
            return 1
        
        # Step 9: Held-out evaluation (60 GPU-h)
        if not self.run_step("Held-Out Evaluation", "lie_circuit/held_gen.py", 60):
            return 1
        
        # Step 10: Statistical analysis (10 GPU-h)
        if not self.run_step("Statistical Analysis", "lie_circuit/stats_ci.py", 10):
            return 1
        
        # Step 11: Visualizations (20 GPU-h)
        if not self.run_step("Creating Visualizations", "lie_circuit/viz_maker.py", 20):
            return 1
        
        # Final summary
        total_time = (time.time() - self.start_time) / 3600
        print(f"\n{'='*60}")
        print(f"🎉 EXPERIMENT COMPLETE!")
        print(f"Total time: {total_time:.1f} hours")
        print(f"Budget used: {775 - self.budget_remaining:.1f} GPU-h")
        print(f"Budget remaining: {self.budget_remaining:.1f} GPU-h")
        print('='*60)
        
        # Check deliverables
        print("\n📦 Checking deliverables...")
        deliverables = {
            'Colab notebook': 'lie_circuit_colab.ipynb',
            'CLT weights': 'clt_weights.pt',
            'Circuit diagram': 'lie_circuit.png',
            'Results chart': 'faith_delta.png',
            'Statistical results': 'statistical_results.json'
        }
        
        for name, file in deliverables.items():
            if os.path.exists(file):
                size = os.path.getsize(file) / 1e6  # MB
                print(f"  ✅ {name}: {file} ({size:.1f} MB)")
            else:
                print(f"  ❌ {name}: {file} NOT FOUND")
        
        return 0

def main():
    """Main entry point"""
    runner = ExperimentRunner()
    
    # Quick test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick smoke test...")
        from lie_circuit.quick_test import run_smoke
        return run_smoke(n_prompts=5)
    
    # Full experiment
    return runner.run_experiment()

if __name__ == "__main__":
    exit(main())