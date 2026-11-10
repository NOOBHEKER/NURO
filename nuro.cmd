@echo off
setlocal
set "REPO=%~dp0"
if exist "%REPO%.venv\Scripts\python.exe" (
  set "PYTHON=%REPO%.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)
pushd "%REPO%"
set "PYTHONPATH=%REPO%"
"%PYTHON%" -m host.cli %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
