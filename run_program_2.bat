@echo off
cd /d "%~dp0"

if exist "dist\People_Counter.exe" (
  start "" "dist\People_Counter.exe"
  exit /b 0
)

if exist "dist\People_Counter\People_Counter.exe" (
  start "" "dist\People_Counter\People_Counter.exe"
  exit /b 0
)

echo File not found: dist\People_Counter.exe or dist\People_Counter\People_Counter.exe
echo Make sure folder "dist" is near this .bat file.
pause
