@echo off
:: Batch file to run the backtest_predictions_historical.py script
:: This file can be used with Windows Task Scheduler for periodic automation

echo ===================================================
echo Starting Historical Prediction Backtesting Process
echo %date% %time%
echo ===================================================

:: Set the working directory to the script location
cd /d "%~dp0"

:: Run the Python script
echo Running backtest_predictions_historical.py...
python backtest_predictions_historical.py

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
