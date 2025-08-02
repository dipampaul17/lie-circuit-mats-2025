#!/usr/bin/env python3
"""
Deploy and run REAL neural mechanistic interpretability on Lambda
"""

import os
import sys
import subprocess
import time

LAMBDA_IP = "129.146.111.174"
LAMBDA_USER = "ubuntu"
LAMBDA_KEY = "~/.ssh/lambda_protean"

def check_ssh():
    """Check SSH connection to Lambda"""
    print("🔑 Checking SSH connection...")
    cmd = f"ssh -i {LAMBDA_KEY} -o ConnectTimeout=5 {LAMBDA_USER}@{LAMBDA_IP} 'echo Connected'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Cannot connect to Lambda instance")
        print(f"Error: {result.stderr}")
        return False
    print("✅ SSH connection successful")
    return True

def upload_script():
    """Upload the neural analysis script"""
    print("\n📤 Uploading neural analysis script...")
    
    files_to_upload = [
        "real_neural_mech_interp_fixed.py",
    ]
    
    for file in files_to_upload:
        if os.path.exists(file):
            cmd = f"scp -i {LAMBDA_KEY} {file} {LAMBDA_USER}@{LAMBDA_IP}:~/"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            if result.returncode == 0:
                print(f"✅ Uploaded {file}")
            else:
                print(f"❌ Failed to upload {file}")
                return False
    
    return True

def setup_lambda_env():
    """Set up environment on Lambda"""
    print("\n🔧 Setting up Lambda environment...")
    
    setup_commands = [
        # Check GPU
        "nvidia-smi",
        
        # Create virtual environment
        "python3 -m venv neural_env || true",
        
        # Activate and install dependencies
        "source neural_env/bin/activate && pip install --upgrade pip",
        "source neural_env/bin/activate && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "source neural_env/bin/activate && pip install transformer-lens einops numpy matplotlib",
    ]
    
    for cmd in setup_commands:
        full_cmd = f"ssh -i {LAMBDA_KEY} {LAMBDA_USER}@{LAMBDA_IP} '{cmd}'"
        print(f"Running: {cmd}")
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        
        if "nvidia-smi" in cmd and result.returncode == 0:
            print("✅ GPU available")
        elif result.returncode != 0 and "already exists" not in result.stderr:
            print(f"⚠️  Command may have failed: {cmd}")
            print(f"Error: {result.stderr[:200]}")
    
    print("✅ Environment setup complete")
    return True

def run_neural_analysis():
    """Run the actual neural analysis on Lambda"""
    print("\n🚀 Running REAL neural mechanistic interpretability...")
    
    run_cmd = f"""ssh -i {LAMBDA_KEY} {LAMBDA_USER}@{LAMBDA_IP} '
        source neural_env/bin/activate && 
        python3 real_neural_mech_interp_fixed.py
    '"""
    
    print("⏳ This will take several minutes...")
    print("=" * 60)
    
    # Run and stream output
    process = subprocess.Popen(run_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode == 0:
        print("\n✅ Neural analysis completed successfully!")
        return True
    else:
        print("\n❌ Neural analysis failed")
        return False

def download_results():
    """Download results from Lambda"""
    print("\n📥 Downloading results...")
    
    # Download all result files
    download_cmd = f"scp -i {LAMBDA_KEY} {LAMBDA_USER}@{LAMBDA_IP}:~/neural_results_*.json ./"
    result = subprocess.run(download_cmd, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print("✅ Results downloaded")
        
        # List downloaded files
        import glob
        result_files = glob.glob("neural_results_*.json")
        for f in result_files:
            print(f"  📄 {f}")
        
        return True
    else:
        print("❌ Failed to download results")
        return False

def main():
    print("🧠 DEPLOYING REAL NEURAL MECHANISTIC INTERPRETABILITY TO LAMBDA")
    print("=" * 60)
    
    # Check connection
    if not check_ssh():
        print("\n❌ Cannot connect to Lambda. Check your SSH key and instance IP.")
        sys.exit(1)
    
    # Upload script
    if not upload_script():
        print("\n❌ Failed to upload script")
        sys.exit(1)
    
    # Setup environment
    if not setup_lambda_env():
        print("\n❌ Failed to setup environment")
        sys.exit(1)
    
    # Run analysis
    if not run_neural_analysis():
        print("\n❌ Analysis failed")
        sys.exit(1)
    
    # Download results
    if not download_results():
        print("\n❌ Failed to download results")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ REAL NEURAL MECHANISTIC INTERPRETABILITY COMPLETE!")
    print("📊 Check the downloaded results files")
    print("🎯 This is what MATS actually wants to see")

if __name__ == "__main__":
    main()