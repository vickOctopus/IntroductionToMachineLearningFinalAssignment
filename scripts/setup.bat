@echo off
chcp 65001 > nul

echo Starting setup process...

:: Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Attempting to install Python 3.9...
    winget install -e --id Python.Python.3.9 || (
        echo Error: Failed to install Python
        echo Please install Python 3.9 manually from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo Refreshing environment variables...
    call RefreshEnv.cmd
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv || (
    echo Error: Failed to create virtual environment
    echo Please make sure Python is installed correctly
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat || (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip || (
    echo Error: Failed to upgrade pip
    pause
    exit /b 1
)

:: Install PyTorch
echo Installing PyTorch (this may take a while)...
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo PyTorch installation from official source failed
    echo Trying alternative installation method...
    python -m pip install torch==2.1.0 torchvision==0.16.0
    if errorlevel 1 (
        echo Error: Failed to install PyTorch
        echo Please check your internet connection
        pause
        exit /b 1
    )
)

:: Install other dependencies
echo Installing other dependencies...
for %%p in (
    numpy==1.24.3
    pandas==2.0.3
    matplotlib==3.7.1
    tqdm==4.65.0
) do (
    echo Installing %%p...
    python -m pip install %%p
    if errorlevel 1 (
        echo Error: Failed to install %%p
        pause
        exit /b 1
    )
)

:: Create directories
echo Creating necessary directories...
mkdir data\raw 2>nul
mkdir models 2>nul
mkdir experiment_logs 2>nul
mkdir visualization_results 2>nul

echo Setup completed successfully!
echo All dependencies have been installed.
pause