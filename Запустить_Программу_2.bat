@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist "dist\People_Counter.exe" (
  start "" "dist\People_Counter.exe"
  exit /b 0
)

echo Не найден файл dist\People_Counter.exe
echo Проверьте, что папка dist находится рядом с этим .bat
pause
