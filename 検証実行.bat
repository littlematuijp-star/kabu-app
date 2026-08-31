@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo 過去10年の成績を計算し直します（2〜3分）。
python final.py
python test_cost.py
pause
