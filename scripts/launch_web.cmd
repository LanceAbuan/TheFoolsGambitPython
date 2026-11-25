@echo off
REM Web server launcher (LAN/tunnel entry point). Portable.
set "ROOT=%~dp0.."
pushd "%ROOT%"
set "PYTHONPATH=%CD%"
".venv\Scripts\python.exe" -m uvicorn server.main:app --host 0.0.0.0 --port 8000 ^
  >> "data\web.out.log" 2>> "data\web.err.log"
popd
