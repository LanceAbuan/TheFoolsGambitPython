@echo off
REM Trainer launcher. Portable: derives the repo root from this script's path.
set "ROOT=%~dp0.."
pushd "%ROOT%"
set "PYTHONPATH=%CD%"
".venv\Scripts\python.exe" "scripts\train.py" ^
  --profile night-v2 --regime scratch ^
  --run-id second-light-v2 --resume second-light-v2 ^
  >> "data\train.out.log" 2>> "data\train.err.log"
popd
