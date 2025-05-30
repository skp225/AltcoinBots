@echo off
:: Set console to UTF-8 mode to handle emoji characters
chcp 65001 > nul
echo Running daily stock/ETF data update...
cd /d "%~dp0"
python update_stocketf_daily_data.py > update_stocketf_daily_log_%date:~-4,4%%date:~-7,2%%date:~-10,2%.txt 2>&1
echo Update completed. See log file for details.
