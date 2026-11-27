@echo off
REM Worker fleet launcher (CUDA EP + leaf batching). Portable.
set "ROOT=%~dp0.."
pushd "%ROOT%"
set "PYTHONPATH=%CD%"
set "ORT_DYLIB_PATH=%CD%\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"
".venv\Scripts\python.exe" "scripts\run_rust_workers.py" ^
  --workers 2 ^
  --models-dir "runs\scratch\second-light-v2\onnx" ^
  --spool-dir "runs\scratch\second-light-v2\spool" ^
  --sims 600 --max-plies 200 ^
  --forced-playouts --pcr-p-full 0.25 --pcr-fast-sims 100 --soft-resign ^
  --search-batch 8 ^
  >> "data\rust_workers.out.log" 2>> "data\rust_workers.err.log"
popd
