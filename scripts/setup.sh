#!/bin/bash

echo "Checking Python installation..."

# 检查Python是否已安装
if ! command -v python3.9 &> /dev/null; then
    echo "Python 3.9 not found. Attempting to install..."
    
    # 检查并安装 Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
            echo "Error: Failed to install Homebrew"
            echo "Please install Homebrew manually from https://brew.sh"
            exit 1
        }
    fi
    
    echo "Installing Python 3.9..."
    brew install python@3.9 || {
        echo "Error: Failed to install Python"
        echo "Please install Python 3.9 manually from https://www.python.org/downloads/"
        exit 1
    }
fi

echo "Creating virtual environment..."
python3.9 -m venv venv || {
    echo "Error: Failed to create virtual environment"
    echo "Please make sure Python is installed correctly"
    exit 1
}

echo "Activating virtual environment..."
source venv/bin/activate || {
    echo "Error: Failed to activate virtual environment"
    exit 1
}

echo "Installing dependencies..."
python -m pip install --upgrade pip || {
    echo "Error: Failed to upgrade pip"
    exit 1
}

# 安装 PyTorch (Apple Silicon)
echo "Installing PyTorch..."
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu || {
    echo "Error: Failed to install PyTorch"
    echo "Please check your internet connection"
    exit 1
}

# 安装其他依赖
echo "Installing other dependencies..."
for package in numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.1 tqdm==4.65.0; do
    echo "Installing $package..."
    pip install $package || {
        echo "Error: Failed to install $package"
        exit 1
    }
done

echo "Creating necessary directories..."
mkdir -p data/raw
mkdir -p models
mkdir -p experiment_logs
mkdir -p visualization_results

echo "Setup completed successfully!"
echo "All dependencies have been installed."