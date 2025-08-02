#!/usr/bin/env python3
"""
Deploy comprehensive neural mechanistic interpretability analysis to Lambda
"""

import os
import sys
import subprocess
import time

LAMBDA_IP = "129.146.111.174"
LAMBDA_USER = "ubuntu"
LAMBDA_KEY = "~/.ssh/lambda_protean"

def check_connection():
    """Verify SSH connection"""
    print("🔑 Checking Lambda connection...")
    cmd = f"ssh -i {LAMBDA_KEY} -o ConnectTimeout=5 {LAMBDA_USER}@{LAMBDA_IP} 'echo Connected'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Cannot connect to Lambda")
        return False
    print("✅ Connection established")
    return True

def upload_scripts():
    """Upload analysis scripts"""
    print("\n📤 Uploading analysis scripts...")
    
    scripts = [
        "robust_neural_analysis.py",
    ]
    
    for script in scripts:
        if os.path.exists(script):
            cmd = f"scp -i {LAMBDA_KEY} {script} {LAMBDA_USER}@{LAMBDA_IP}:~/"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            if result.returncode == 0:
                print(f"✅ Uploaded {script}")
            else:
                print(f"❌ Failed to upload {script}")
                return False
    return True

def setup_environment():
    """Ensure environment is ready"""
    print("\n🔧 Setting up environment...")
    
    commands = [
        # Check GPU
        "nvidia-smi | grep 'NVIDIA-SMI' && echo '✅ GPU ready'",
        
        # Ensure scipy is installed for statistical tests
        "source neural_env/bin/activate && pip install scipy scikit-learn -q",
    ]
    
    for cmd in commands:
        full_cmd = f"ssh -i {LAMBDA_KEY} {LAMBDA_USER}@{LAMBDA_IP} '{cmd}'"
        subprocess.run(full_cmd, shell=True)
    
    print("✅ Environment ready")
    return True

def run_comprehensive_analysis():
    """Run the comprehensive analysis"""
    print("\n" + "="*60)
    print("🚀 RUNNING COMPREHENSIVE NEURAL ANALYSIS ON LAMBDA")
    print("="*60)
    print("This will analyze multiple layers with proper statistics...")
    print("Expected runtime: 5-10 minutes")
    print("="*60 + "\n")
    
    run_cmd = f"""ssh -i {LAMBDA_KEY} {LAMBDA_USER}@{LAMBDA_IP} '
        source neural_env/bin/activate && 
        export CUDA_VISIBLE_DEVICES=0 &&
        python3 robust_neural_analysis.py
    '"""
    
    # Run and stream output
    process = subprocess.Popen(run_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode == 0:
        print("\n✅ Analysis completed successfully!")
        return True
    else:
        print("\n❌ Analysis failed")
        return False

def download_results():
    """Download all results"""
    print("\n📥 Downloading results...")
    
    # Download JSON results
    download_cmd = f"scp -i {LAMBDA_KEY} {LAMBDA_USER}@{LAMBDA_IP}:~/*neural_results_*.json ./"
    result = subprocess.run(download_cmd, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print("✅ Results downloaded")
        
        # List files
        import glob
        result_files = glob.glob("*neural_results_*.json")
        for f in result_files:
            print(f"  📄 {f}")
            
            # Show file size
            size = os.path.getsize(f)
            print(f"     Size: {size:,} bytes")
        
        return True
    else:
        print("❌ Failed to download results")
        return False

def analyze_results():
    """Quick analysis of downloaded results"""
    print("\n📊 Analyzing results...")
    
    import glob
    import json
    
    result_files = glob.glob("*neural_results_*.json")
    if not result_files:
        print("❌ No result files found")
        return
    
    # Read latest file
    latest_file = sorted(result_files)[-1]
    print(f"\nAnalyzing: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("\n🔬 Key Findings:")
    print(f"  Model: {results.get('model', 'N/A')}")
    print(f"  Device: {results.get('device', 'N/A')}")
    
    if 'best_layer' in results:
        print(f"  Best layer: {results['best_layer']}")
        print(f"  Best effect: {results['best_effect']:.3f}")
    
    if 'layer_analysis' in results:
        print("\n  Layer-by-layer results:")
        for layer, data in results['layer_analysis'].items():
            print(f"    {layer}:")
            print(f"      Significant dims: {data.get('n_significant_dims', 0)}")
            if 'intervention' in data:
                print(f"      Accuracy drop: {data['intervention']['accuracy_drop']:.3f}")
    
    if 'detailed_best_layer' in results:
        detail = results['detailed_best_layer']
        if detail.get('generalization'):
            print("\n  ✅ Effect generalizes to held-out data!")
        else:
            print("\n  ⚠️  Effect may not generalize well")

def main():
    print("🧠 COMPREHENSIVE NEURAL MECHANISTIC INTERPRETABILITY")
    print("Deploying to Lambda Labs A100 for proper analysis")
    print("="*60)
    
    # Check connection
    if not check_connection():
        print("\n❌ Cannot connect to Lambda")
        sys.exit(1)
    
    # Upload scripts
    if not upload_scripts():
        print("\n❌ Failed to upload scripts")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Failed to setup environment")
        sys.exit(1)
    
    # Run analysis
    if not run_comprehensive_analysis():
        print("\n❌ Analysis failed")
        sys.exit(1)
    
    # Download results
    if not download_results():
        print("\n❌ Failed to download results")
        sys.exit(1)
    
    # Analyze results
    analyze_results()
    
    print("\n" + "="*60)
    print("✅ COMPREHENSIVE NEURAL ANALYSIS COMPLETE!")
    print("📊 Results downloaded and analyzed")
    print("🎯 This demonstrates REAL mechanistic interpretability:")
    print("   • Multi-layer analysis")
    print("   • Statistical significance testing")
    print("   • Causal interventions")
    print("   • Generalization testing")
    print("="*60)

if __name__ == "__main__":
    main()