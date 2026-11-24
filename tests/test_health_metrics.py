"""WP5: the smoke run emits a health-metrics journal event per iteration."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_run_emits_health_metrics(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--profile", "smoke",
            "--runs-dir", str(tmp_path / "runs"),
            "--data-dir", str(tmp_path / "data"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    log = (tmp_path / "data" / "train_log.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in log.splitlines() if line.strip()]
    metrics = [e for e in events if e.get("kind") == "metrics"]
    assert metrics, "expected at least one health-metrics event"
    first = metrics[0]
    assert {"policy_ce", "policy_entropy", "p_ul", "value_acc"} <= set(first)
    assert first["policy_ce"] > 0.0
    # strict JSON: no bare NaN tokens may leak into the journal
    assert "NaN" not in log
