<#
.SYNOPSIS
    Fool's Gambit training watchdog: keep trainer + fleet + web + feeder alive.

.DESCRIPTION
    Runs every 5 minutes from Task Scheduler (and is safe to run by hand). It
    only acts when something is actually wrong, appends one line per event to
    data\watchdog.log, and never touches runs/ contents.

    Checks, in order:
      1. trainer process            -> restart via scripts\launch_trainer.cmd
      2. trainer liveness           -> restart if no new ONNX/iteration in 25 min
      3. worker fleet (>= 6 of 8)   -> restart via scripts\launch_fleet.cmd
      4. web /api/health on :8000   -> restart via scripts\launch_web.cmd
      5. feeder process             -> restart via scripts\launch_feeder.cmd
      6. disk headroom (>= 15 GB)   -> prune the processed-game archive

    Exit code 0 always (a failing watchdog must not spam Task Scheduler).
#>
$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root 'data'
$Log = Join-Path $LogDir 'watchdog.log'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log([string]$Message) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -Path $Log -Value $line -Encoding utf8
    Write-Output $line
}

function Get-Procs([string]$Pattern) {
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like $Pattern -and $_.Name -eq 'python.exe' })
}

function Start-Hidden([string]$CmdPath) {
    Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', "`"$CmdPath`"" `
        -WorkingDirectory $Root -WindowStyle Hidden
}

$trainerCmd = Join-Path $Root 'scripts\launch_trainer.cmd'
$fleetCmd   = Join-Path $Root 'scripts\launch_fleet.cmd'
$webCmd     = Join-Path $Root 'scripts\launch_web.cmd'
$feederCmd  = Join-Path $Root 'scripts\launch_feeder.cmd'

# --- 1/2. trainer ------------------------------------------------------------
$trainer = Get-Procs '*train.py*'
if ($trainer.Count -eq 0) {
    Write-Log 'ACTION trainer not running -> starting'
    Start-Hidden $trainerCmd
}
else {
    # Liveness = "is it burning CPU", NOT "did it export a model recently".
    # A gate (16 games x 128 sims) legitimately runs 20-40 minutes with no new
    # ONNX - an mtime rule restarted a healthy trainer mid-gate and threw away
    # every iteration since the last snapshot. A hung trainer shows ~0 CPU.
    $pid0 = $trainer[0].ProcessId
    $cpu0 = (Get-Process -Id $pid0 -ErrorAction SilentlyContinue).CPU
    Start-Sleep -Seconds 10
    $cpu1 = (Get-Process -Id $pid0 -ErrorAction SilentlyContinue).CPU
    $logPath = Join-Path $Root 'data\train.err.log'
    $logAge = if (Test-Path $logPath) { ((Get-Date) - (Get-Item $logPath).LastWriteTime).TotalMinutes } else { 999 }
    $busy = ($null -ne $cpu0 -and $null -ne $cpu1 -and ($cpu1 - $cpu0) -gt 2.0)
    if (-not $busy -and $logAge -gt 30) {
        Write-Log ("ACTION trainer hung: {0:N1} CPU-seconds in 10s, log silent {1:N0} min -> restart" -f `
            ($(if ($null -ne $cpu0 -and $null -ne $cpu1) { $cpu1 - $cpu0 } else { 0 })), $logAge)
        $trainer | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 5
        Start-Hidden $trainerCmd
    }
}

# --- 3. worker fleet ---------------------------------------------------------
$workers = Get-Procs '*run_rust_workers*'
$workerCount = (Get-Procs '*rust_worker.py*').Count
if ($workers.Count -eq 0 -or $workerCount -lt 6) {
    Write-Log ("ACTION fleet degraded (supervisors={0}, workers={1}) -> restart" -f $workers.Count, $workerCount)
    Get-Procs '*rust_worker*' | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 5
    Start-Hidden $fleetCmd
}

# --- 4. web server -----------------------------------------------------------
$healthy = $false
try {
    $r = Invoke-WebRequest 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 8
    $healthy = ($r.StatusCode -eq 200)
} catch { $healthy = $false }
if (-not $healthy) {
    Write-Log 'ACTION web server unhealthy -> restart'
    Get-Procs '*uvicorn*' | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
    Start-Hidden $webCmd
}

# --- 5. feeder ---------------------------------------------------------------
if ((Get-Procs '*replay_latest_game*').Count -eq 0) {
    Write-Log 'ACTION feeder not running -> starting'
    Start-Hidden $feederCmd
}

# --- 6. disk headroom --------------------------------------------------------
foreach ($drive in 'C', 'D') {
    $free = (Get-PSDrive -Name $drive -ErrorAction SilentlyContinue).Free / 1GB
    if ($null -ne $free -and $free -lt 15) {
        Write-Log ("ACTION {0}: has only {1:N1} GB free -> pruning processed archive" -f $drive, $free)
        $env:PYTHONPATH = $Root
        & (Join-Path $Root '.venv\Scripts\python.exe') -c `
            "from training.rust_workers import prune_processed_games as p; print(p(r'runs\scratch\second-light-v2\spool', keep=2000, keep_hours=6.0))" 2>&1 |
            ForEach-Object { Write-Log ("  prune: {0}" -f $_) }
    }
}

# Keep the log bounded (~2 MB) so a 3-day run cannot fill the disk with it.
if ((Test-Path $Log) -and ((Get-Item $Log).Length -gt 2MB)) {
    Get-Content $Log -Tail 4000 | Set-Content $Log -Encoding utf8
}
exit 0
