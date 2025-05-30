@echo off
:: Batch file to run the auto_predict_top_gainers.py script
:: This file can be used with Windows Task Scheduler for daily automation

echo ===================================================
echo Starting Top Gainers Prediction Script
echo %date% %time%
echo ===================================================

:: Set the working directory to the script location
cd /d "%~dp0"

:: Run the Python script
echo Running auto_predict_top_gainers.py...
python auto_predict_top_gainers.py

:: Check if the script ran successfully
if %errorlevel% equ 0 (
    echo.
    echo ===================================================
    echo Script completed successfully!
    echo %date% %time%
    echo ===================================================
) else (
    echo.
    echo ===================================================
    echo ERROR: Script failed with error code %errorlevel%
    echo %date% %time%
    echo ===================================================
)

:: Pause only if run manually (not when run by Task Scheduler)
if "%1"=="" pause
