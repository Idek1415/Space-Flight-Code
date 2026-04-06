@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title NLP for PDFs

echo.
echo  ================================================
echo   NLP for PDFs Application
echo  ================================================
echo  Loading...
echo.

:: Try python, then py launcher
python --version >nul 2>&1
if !errorlevel! == 0 (
    python App\launcher.py --detach %*
    goto :end
)

py --version >nul 2>&1
if !errorlevel! == 0 (
    py App\launcher.py --detach %*
    goto :end
)

echo  ERROR: Python 3.10+ not found.
echo.
echo  Please install Python from https://www.python.org/downloads/
echo  Make sure to check "Add Python to PATH" during installation.
echo.
pause
:end
