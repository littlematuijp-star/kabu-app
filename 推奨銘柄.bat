@echo off
chcp 65001 > nul
cd /d "%~dp0"
python recommend.py
echo.
echo 詳しい理由は data\out\recommend.txt に保存されています。
pause
