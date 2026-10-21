@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please create the environment described in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" demo_server.py
if errorlevel 1 pause

