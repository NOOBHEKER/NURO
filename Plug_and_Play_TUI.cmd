@echo off
setlocal
title Neural Loop -- Plug ^& Play TUI Workstation
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

"%PYTHON%" -m host.cli tui
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo TUI exited with error code %ERRORLEVEL%.
  pause
)
