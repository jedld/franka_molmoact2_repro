@echo off
REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
REM
REM Launch Isaac Sim with the HTTP motion server (motion_server.cpp compatible API).
REM
REM Usage:
REM   launch_isaac_motion_server.bat [--port 34568] [--headless] [--robot-path PATH]
REM
REM Examples:
REM   launch_isaac_motion_server.bat
REM   launch_isaac_motion_server.bat --headless --test
REM   launch_isaac_motion_server.bat --robot-path /World/RobotMount/Franka --spawn-robot

setlocal

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
set BUILD_ROOT=%REPO_ROOT%\_build\windows-x86_64\release

if not exist "%BUILD_ROOT%\python.bat" (
    echo ERROR: Isaac Sim build not found at %BUILD_ROOT%
    echo Run build.bat from the repo root first.
    exit /b 1
)

echo.
echo === Isaac Sim motion server launcher ===
echo   BUILD_ROOT = %BUILD_ROOT%
echo   PROJECT    = %SCRIPT_DIR%
echo.

cd /d "%BUILD_ROOT%"
call python.bat "%SCRIPT_DIR%isaac_motion_server.py" %*
endlocal
