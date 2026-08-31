@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo 株価と配当のデータを最新にします（30分ほどかかります）。
python jp_data.py --refresh
python fundamentals.py
pause
