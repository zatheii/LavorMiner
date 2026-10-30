@echo off
rem Rebuilds CrystalMiner.exe from crystal_miner_v18_menu.py. Run this after editing the script.
rem Only the PC doing the build needs Python. Build tools stay in %TEMP%, not your main Python.
cd /d "%~dp0"
set BUILD=%TEMP%\crystal_miner_build
if not exist "%BUILD%\venv" py -m venv "%BUILD%\venv"
"%BUILD%\venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check pyinstaller pyautogui pynput numpy opencv-python-headless pillow
if errorlevel 1 goto failed
"%BUILD%\venv\Scripts\python.exe" -m PyInstaller --noconfirm --onefile --console --name CrystalMiner --distpath . --workpath "%BUILD%\work" --specpath "%BUILD%" --hidden-import pynput.keyboard._win32 --hidden-import pynput.mouse._win32 crystal_miner_v18_menu.py
if errorlevel 1 goto failed
echo.
echo Done: %CD%\CrystalMiner.exe
pause
exit /b 0
:failed
echo.
echo Build failed, see the messages above.
pause
exit /b 1
