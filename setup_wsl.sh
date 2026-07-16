#!/usr/bin/env bash
# Distill Phase 0 environment setup. Run inside WSL2 Ubuntu from the project root.
#   bash setup_wsl.sh
set -e

echo "== 1. Checking GPU visibility in WSL =="
nvidia-smi || { echo "nvidia-smi failed. Install the NVIDIA driver on Windows and ensure WSL2."; exit 1; }

echo "== 2. Creating Python venv =="
python3 -m venv .venv || { echo "venv failed. Try: sudo apt install -y python3-venv"; exit 1; }
source .venv/bin/activate
python -m pip install --upgrade pip

echo "== 3. Installing PyTorch + torchvision (CUDA 12.1) =="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "== 4. Installing project dependencies =="
pip install -r requirements.txt

echo "== 5. Verifying CUDA in PyTorch =="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "== Setup complete. Activate later with: source .venv/bin/activate =="
