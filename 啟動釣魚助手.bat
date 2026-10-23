@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "%~dp0gui.py"
    exit /b 0
)

where pythonw >nul 2>&1
if not errorlevel 1 (
    start "" pythonw "%~dp0gui.py"
    exit /b 0
)

where pyw >nul 2>&1
if not errorlevel 1 (
    start "" pyw -3.12 "%~dp0gui.py"
    exit /b 0
)

echo 找不到 Python 3.12。請先安裝 Python，或依 README 建立 .venv。
pause
exit /b 1
