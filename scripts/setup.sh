#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
python -m pip install --upgrade pip

# 安装 PyTorch (Apple Silicon)
echo "Installing PyTorch..."
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# 安装其他依赖
echo "Installing other dependencies..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p data/raw
mkdir -p models
mkdir -p experiment_logs
mkdir -p visualization_results

echo "Setup complete!"