@echo off
REM Live-board feeder launcher (replays finished games to /watch). Portable.
set "ROOT=%~dp0.."
pushd "%ROOT%"
set "PYTHONPATH=%CD%"
".venv\Scripts\python.exe" "scripts\replay_latest_game.py" ^
  >> "data\feeder.out.log" 2>> "data\feeder.err.log"
popd
