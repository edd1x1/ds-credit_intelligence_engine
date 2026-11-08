@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Could not install dependencies. Check Python 3.10+ and internet access.
  pause
  exit /b 1
)
python app.py
pause
