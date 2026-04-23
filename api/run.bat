@echo off
cd /d %~dp0\..
if "%HOST%"=="" set HOST=0.0.0.0
if "%PORT%"=="" set PORT=8000
uvicorn api.app.main:app --host %HOST% --port %PORT% --workers 1
