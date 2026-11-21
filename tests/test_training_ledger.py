"""Ledger + metrics feed tests (§5.1): JSONL shape, Elo recompute, atomic metrics."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

from training.ledger import (
    ArenaLedger,
    LedgerEntry,
    MetricsWriter,
    recompute_elos,
)


class TestArenaLedger:
    def test_append_writes_jsonl_lines(self, tmp_path):
        ledger = ArenaLedger(tmp_path / "arena.jsonl")
        ledger.append(
            LedgerEntry(
                run_id="r1", regime="scratch", checkpoint="ck1", opponent="none",
                games=0, wins=0, draws=0, elo=800.0, kind="baseline", promoted=True,
            )
        )
        ledger.append(
            LedgerEntry(
                run_id="r1", regime="scratch", checkpoint="ck2", opponent="ck1",
                games=4, wins=3, draws=1, elo=824.0, losses=0,
            )
        )
        lines = (tmp_path / "arena.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        # §5.1 required keys
        for key in ("run_id", "regime", "checkpoint", "opponent", "games", "wins", "draws", "elo"):
            assert key in first
        assert first["opponent"] == "none"
        # §5.1: arena rows carry the SQLite games.kind tag; seed rows are baseline
        assert first["kind"] == "baseline"
        assert json.loads(lines[1])["kind"] == "arena"

    def test_read_roundtrip(self, tmp_path):
        ledger = ArenaLedger(tmp_path / "arena.jsonl")
        entry = LedgerEntry(
            run_id="r9", regime="hybrid", checkpoint="a", opponent="b",
            games=10, wins=6, draws=2, elo=813.4, losses=2,
        )
        ledger.append(entry)
        rows = ledger.read()
        assert len(rows) == 1
        assert rows[0]["checkpoint"] == "a"
        assert rows[0]["games"] == 10
        assert abs(rows[0]["elo"] - 813.4) < 1e-9

    def test_read_empty_when_missing(self, tmp_path):
        assert ArenaLedger(tmp_path / "missing.jsonl").read() == []

    def test_corrupt_lines_skipped(self, tmp_path):
        path = tmp_path / "arena.jsonl"
        path.write_text(
            '{"checkpoint": "good", "games": 2, "wins": 1, "draws": 1, "opponent": "x"}\n'
            "NOT JSON AT ALL\n",
            encoding="utf-8",
        )
        rows = ArenaLedger(path).read()
        assert len(rows) == 1
        assert rows[0]["checkpoint"] == "good"


class TestRecomputeElos:
    def test_baseline_champion_gets_base_elo(self):
        rows = [{"checkpoint": "ck1", "opponent": "none", "games": 0}]
        elos = recompute_elos(rows, base_elo=800.0)
        assert elos["ck1"] == 800.0

    def test_beating_the_champion_raises_candidate(self):
        rows = [
            {"checkpoint": "ck1", "opponent": "none", "games": 0},
            {"checkpoint": "ck2", "opponent": "ck1", "games": 10, "wins": 9, "draws": 1},
        ]
        elos = recompute_elos(rows, base_elo=800.0, k_factor=32.0)
        assert elos["ck2"] > 800.0
        assert elos["ck1"] < 800.0

    def test_elo_matches_manual_k_factor_update(self):
        rows = [{"checkpoint": "ck2", "opponent": "ck1", "games": 10, "wins": 9, "draws": 1}]
        elos = recompute_elos(rows, base_elo=800.0, k_factor=32.0)
        expected = 1.0 / (1.0 + 10.0 ** 0.0)
        assert abs(elos["ck2"] - (800.0 + 32.0 * (0.95 - expected))) < 1e-9

    def test_empty_ledger_gives_empty_map(self):
        assert recompute_elos([]) == {}


class TestMetricsWriter:
    def test_write_required_contract_keys(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.json")
        writer.update(elo=812.5, win_rate=0.58, games_played=42, loss=0.031)
        data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert data["elo"] == 812.5
        assert data["win_rate"] == 0.58
        assert data["games_played"] == 42
        assert data["loss"] == 0.031

    def test_update_merges_preserving_other_keys(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"custom_key": "keep-me", "elo": 700.0}), encoding="utf-8")
        writer = MetricsWriter(path)
        writer.update(elo=850.0, games_played=10)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["custom_key"] == "keep-me"
        assert data["elo"] == 850.0

    def test_none_values_are_not_written(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.json")
        writer.update(elo=None, games_played=5)
        data = writer.read()
        assert "elo" not in data
        assert data["games_played"] == 5

    def test_file_is_valid_json_after_each_write(self, tmp_path):
        """Simulates the backend reading /api/metrics mid-write (atomic replace)."""
        writer = MetricsWriter(tmp_path / "metrics.json")
        for i in range(5):
            writer.update(elo=800.0 + i, games_played=i)
            data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
            assert data["games_played"] == i
        assert not (tmp_path / "metrics.tmp").exists()

    def test_per_checkpoint_elo_map_roundtrip(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.json")
        elo_map = {"ck1": 800.0, "ck2": 821.4}
        writer.update(checkpoints=elo_map, champion="ck2", regime="scratch")
        data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert data["checkpoints"] == elo_map
        assert data["champion"] == "ck2"
