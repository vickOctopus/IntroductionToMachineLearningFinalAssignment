@echo off
chcp 65001 > nul

:: 设置pip使用清华镜像源
set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PIP_DEFAULT_TIMEOUT=100"
set "PIP_RETRIES=3"

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
    echo Warning: Failed to upgrade pip, continuing with other dependencies...
)

:: Install PyTorch
echo Installing PyTorch (this may take a while)...
python -m pip install torch==2.1.0 torchvision==0.16.0 || (
    echo PyTorch installation failed
    pause
    exit /b 1
)

:: Install other dependencies
echo Installing other dependencies...

:: First install numpy 1.24.3 specifically
echo Installing numpy 1.24.3...
python -m pip install "numpy<2.0" || (
    echo Error: Failed to install numpy
    pause
    exit /b 1
)

:: Then install other packages
for %%p in (
    pandas
    matplotlib
    tqdm
    tabulate
) do (
    echo Installing %%p...
    python -m pip install %%p || (
        echo Error: Failed to install %%p
        pause
        exit /b 1
    )
)

:: Create directories
echo Creating necessary directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "models" mkdir "models"
if not exist "experiment_logs" mkdir "experiment_logs"
if not exist "visualization_results" mkdir "visualization_results"

echo Setup completed successfully!
echo All dependencies have been installed.
pause