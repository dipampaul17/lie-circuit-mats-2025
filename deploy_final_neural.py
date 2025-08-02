#!/usr/bin/env python3
"""
Deploy and run final neural analysis on Lambda
"""

import os
import sys
import subprocess
import time

LAMBDA_IP = "129.146.111.174"
LAMBDA_USER = "ubuntu"
LAMBDA_KEY = "~/.ssh/lambda_protean"

print("🧠 DEPLOYING FINAL NEURAL ANALYSIS TO LAMBDA")
print("="*60)

# Upload script
print("\n📤 Uploading final_neural_analysis.py...")
cmd = f"scp -i {LAMBDA_KEY} final_neural_analysis.py {LAMBDA_USER}@{LAMBDA_IP}:~/"
result = subprocess.run(cmd, shell=True, capture_output=True)
if result.returncode != 0:
    print("❌ Failed to upload")
    sys.exit(1)
print("✅ Uploaded successfully")

# Run analysis
print("\n🚀 Running analysis on Lambda GPU...")
print("Expected runtime: 3-5 minutes")
print("="*60 + "\n")

run_cmd = f"""ssh -i {LAMBDA_KEY} {LAMBDA_USER}@{LAMBDA_IP} '
    source neural_env/bin/activate && 
    python3 final_neural_analysis.py
'"""

# Stream output
process = subprocess.Popen(run_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in iter(process.stdout.readline, ''):
    print(line.rstrip())
process.wait()

if process.returncode == 0:
    print("\n✅ Analysis completed!")
    
    # Download results
    print("\n📥 Downloading results...")
    download_cmd = f"scp -i {LAMBDA_KEY} {LAMBDA_USER}@{LAMBDA_IP}:~/final_neural_results_*.json ./"
    subprocess.run(download_cmd, shell=True)
    
    # Show results
    import glob
    import json
    
    files = glob.glob("final_neural_results_*.json")
    if files:
        latest = sorted(files)[-1]
        print(f"\n📊 Results in: {latest}")
        
        with open(latest, 'r') as f:
            results = json.load(f)
        
        if "best_layer" in results and results["best_layer"] is not None:
            print(f"\n🎯 Key Finding:")
            print(f"  Best layer: {results['best_layer']}")
            print(f"  Effect size: {results['best_effect']:.3f}")
            
            if "validation_test" in results:
                val = results["validation_test"]
                print(f"  Validation p-value: {val['p_value']:.4f}")
                print(f"  Statistically significant: {'YES' if val['significant'] else 'NO'}")
    
    print("\n✅ DONE! Real neural mechanistic interpretability complete.")
else:
    print("\n❌ Analysis failed")
    sys.exit(1)