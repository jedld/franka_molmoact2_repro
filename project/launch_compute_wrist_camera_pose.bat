@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "BUILD_ROOT=%SCRIPT_DIR%..\_build\windows-x86_64\release"
if not exist "%BUILD_ROOT%\python.bat" (
  echo ERROR: Isaac Sim build not found at %BUILD_ROOT%
  exit /b 1
)
cd /d "%BUILD_ROOT%"
call python.bat "%SCRIPT_DIR%compute_wrist_camera_pose.py" --spawn-robot %*
endlocal
