@echo off
if "%~1"=="" (
  echo Usage: Publish-Git.cmd https://github.com/YOUR_LOGIN/NaomiGen.git
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Publish-Git.ps1" -RemoteUrl "%~1"
exit /b %ERRORLEVEL%
