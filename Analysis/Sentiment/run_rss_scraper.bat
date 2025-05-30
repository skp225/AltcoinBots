@echo off
setlocal enabledelayedexpansion

:: Configuration
set "SCRIPT_DIR=%~dp0"
set "PYTHON_SCRIPT=%SCRIPT_DIR%rss_sentiment_scraper.py"
set "OUTPUT_DIR=%SCRIPT_DIR%sentiment_data"
set "LOG_DIR=%SCRIPT_DIR%logs"

:: Create necessary directories
if not exist "%OUTPUT_DIR%\articles" mkdir "%OUTPUT_DIR%\articles"
if not exist "%OUTPUT_DIR%\trends" mkdir "%OUTPUT_DIR%\trends"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Set timestamp for log file
set "timestamp=%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=!timestamp: =0!"
set "log_file=%LOG_DIR%\scraper_run_!timestamp!.log"

:: Start logging
echo ======================================= > "!log_file!"
echo Starting RSS scraper at %date% %time% >> "!log_file!"
echo ======================================= >> "!log_file!"

:: Run the Python script with full path specification
cd /d "%SCRIPT_DIR%"
python "%PYTHON_SCRIPT%" >> "!log_file!" 2>&1

:: Check if the script ran successfully
if %errorlevel% equ 0 (
    echo RSS scraper completed successfully at %date% %time% >> "!log_file!"
    
    :: Move output files to organized directories
    for %%f in ("%SCRIPT_DIR%\rss_articles_*.json") do (
        move "%%f" "%OUTPUT_DIR%\articles\" >nul 2>&1
    )
    
    for %%f in ("%SCRIPT_DIR%\trending_projects_*.json") do (
        move "%%f" "%OUTPUT_DIR%\trends\" >nul 2>&1
    )
    
    :: Create a copy of the latest files for easy access
    for /f "delims=" %%f in ('dir /b /od "%OUTPUT_DIR%\articles\*.json" 2^>nul') do (
        set "latest_article=%%f"
    )
    
    for /f "delims=" %%f in ('dir /b /od "%OUTPUT_DIR%\trends\*.json" 2^>nul') do (
        set "latest_trend=%%f"
    )
    
    if defined latest_article (
        copy "%OUTPUT_DIR%\articles\!latest_article!" "%OUTPUT_DIR%\latest_articles.json" >nul
    )
    
    if defined latest_trend (
        copy "%OUTPUT_DIR%\trends\!latest_trend!" "%OUTPUT_DIR%\latest_trends.json" >nul
    )
) else (
    echo ERROR: RSS scraper failed at %date% %time% >> "!log_file!"
)

echo ======================================= >> "!log_file!"
echo. >> "!log_file!"

:: Exit cleanly
exit /b 0
