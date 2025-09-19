# Install Required Packages
# Run this cell first to install all necessary dependencies

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")

# List of required packages
packages = [
    "torch>=1.9.0",
    "transformers>=4.20.0", 
    "huggingface_hub>=0.10.0",
    "tokenizers>=0.12.0",
    "datasets",
    "accelerate",
]

print("Installing required packages for Hugging Face model demonstration...")
print("This may take a few minutes...\n")

for package in packages:
    install_package(package)

print("\n" + "="*60)
print("📦 Package installation complete!")
print("💡 For GPU support, you may also want to install:")
print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
print("="*60)