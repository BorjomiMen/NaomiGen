@echo off
setlocal
cd /d "%~dp0code"
set "PATH=C:\Octave\mingw64\bin;C:\Octave\usr\bin;%PATH%"
set "OPENBLAS_NUM_THREADS=4"
if /I "%~1"=="--check" (
  "C:\Octave\mingw64\bin\octave-cli.exe" --quiet --eval "installNAOMi; test_naomi_core; test_naomi_scanner;"
  exit /b
)
"C:\Octave\mingw64\bin\octave-cli.exe" --quiet --eval "installNAOMi; run_naomi_octave();"
if errorlevel 1 (
  echo NAOMi failed. Copy the error above for diagnosis.
  pause
  exit /b 1
)
echo Results saved in the results folder next to this launcher.
pause
