@echo off
setlocal
set "REPO=%~dp0"
cd /d "%REPO%"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"

if not defined PY (
  echo Python 3.10+ was not found.
  echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  echo.
  pause
  exit /b 2
)

set "PYTHONPATH=%REPO%"
%PY% -m host.cli setup %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" pause
exit /b %CODE%
