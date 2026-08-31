@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ブラウザで画面を開きます。閉じるときはこの黒い画面を閉じてください。
python -m streamlit run app.py
pause
