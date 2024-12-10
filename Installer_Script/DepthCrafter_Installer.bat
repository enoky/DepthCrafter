@echo off
REM DepthCrafter Installer Script (v0.1 with improvements and debugging)

REM Check if git is installed
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo git is not installed or not available in PATH.
    echo Please install git and ensure it is in your PATH.
    pause
    exit /b 1
)

REM If the DepthCrafter directory exists, prompt user
if exist "DepthCrafter" (
    echo The DepthCrafter directory already exists.
    set /p user_choice="Do you want to remove it and continue? (Y/N): "
    if /i "%user_choice%"=="Y" (
        rmdir /s /q DepthCrafter
        if %errorlevel% neq 0 (
            echo Failed to remove existing DepthCrafter directory.
            pause
            exit /b %errorlevel%
        )
    ) else (
        echo Aborting installation.
        pause
        exit /b 0
    )
)

REM Set environment variable before cloning
set GIT_CLONE_PROTECTION_ACTIVE=false

REM Clone the DepthCrafter repository
git clone https://github.com/Tencent/DepthCrafter.git
if %errorlevel% neq 0 (
    echo Failed to clone the DepthCrafter repository.
    pause
    exit /b %errorlevel%
)

cd DepthCrafter
if %errorlevel% neq 0 (
    echo Failed to change directory into the DepthCrafter repository.
    pause
    exit /b %errorlevel%
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not added to PATH.
    pause
    exit /b 1
)

REM Create a virtual environment
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    pause
    exit /b %errorlevel%
)

REM Activate the virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment activation script not found.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b %errorlevel%
)

REM Upgrade pip and install dependencies from requirements.txt
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip.
    pause
    exit /b %errorlevel%
)

python -m pip install --upgrade -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies from requirements.txt.
    pause
    exit /b %errorlevel%
)

REM Install specific PyTorch and related packages
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo Failed to install PyTorch packages.
    pause
    exit /b %errorlevel%
)

REM Install additional packages
python -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo Failed to install xformers.
    pause
    exit /b %errorlevel%
)

echo All dependencies installed successfully.
pause
