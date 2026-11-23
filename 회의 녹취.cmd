@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m lwm.web --open
if errorlevel 1 pause
