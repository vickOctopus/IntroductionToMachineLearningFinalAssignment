@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip

:: 从官网下载PyTorch（避免代理问题）
echo Installing PyTorch...
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

:: 安装其他依赖
echo Installing other dependencies...
pip install -r requirements.txt

echo Creating necessary directories...
mkdir data\raw 2>nul
mkdir models 2>nul
mkdir experiment_logs 2>nul
mkdir visualization_results 2>nul

echo Setup complete! 