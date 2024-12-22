@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Creating necessary directories...
mkdir data\raw 2>nul
mkdir models 2>nul
mkdir experiment_logs 2>nul
mkdir visualization_results 2>nul

echo Setup complete! 