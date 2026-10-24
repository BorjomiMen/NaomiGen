@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
pushd "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup.ps1"
if errorlevel 1 (
  popd
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0naomi_zenodo_stage1\pipeline.py" %*
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo Generation or verification failed. See the diagnostic output above.
if not "%NAOMI_NO_PAUSE%"=="1" pause
popd
exit /b %RESULT%
