@echo off
setlocal
title Neural Loop -- USB Auto-Launch Watcher
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

"%PYTHON%" -m host.cli watch %*
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Watcher exited.
  pause
)
