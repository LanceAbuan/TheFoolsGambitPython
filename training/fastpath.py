"""Config gate for the gambit_core (Rust) fast path - slice M3.

One switch controls every Rust-backed branch (encoder/legal-map/FastBoard) so
the Python-vs-Rust ablation is clean and revertible:

    GAMBIT_CORE_FASTPATH=1   # Rust path (gambit_core required)
    GAMBIT_CORE_FASTPATH=    # (default) Python path - python-chess + pure NumPy

The env var is read once at import; tests and benches toggle it with
enable_fastpath() / disable_fastpath(). Default OFF: the training hot path
behaves exactly as before M3 until the orchestrator flips it.

Failure policy: when the fast path is requested but gambit_core is not
importable, the first fast-path call raises ImportError - a silent fallback
would poison the ablation measurement (RESEARCH_STANDARD principle 2: a
number must describe what actually ran), so the gate fails loudly instead.
"""
from __future__ import annotations

import importlib
import logging
import os

log = logging.getLogger(__name__)

ENV_VAR = "GAMBIT_CORE_FASTPATH"
_TRUTHY = {"1", "true", "yes", "on"}

_enabled: bool = os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY
_gambit_core = None  # cached module handle; imported lazily by fast-path users


def fastpath_enabled() -> bool:
    """True when the Rust (gambit_core) branches should run."""
    return _enabled


def enable_fastpath() -> None:
    """Turn the Rust path on (and verify gambit_core imports) NOW.

    Eagerly imports gambit_core so a missing/broken extension fails at enable
    time rather than deep inside a self-play run.
    """
    global _enabled
    import_gambit_core()
    _enabled = True
    log.info("GAMBIT_CORE_FASTPATH enabled: Rust movegen/encode/FastBoard active")


def disable_fastpath() -> None:
    """Turn the Rust path off (pure-Python semantics)."""
    global _enabled
    _enabled = False


def import_gambit_core():
    """Import and cache gambit_core; raises ImportError with context."""
    global _gambit_core
    if _gambit_core is None:
        try:
            _gambit_core = importlib.import_module("gambit_core")
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "GAMBIT_CORE_FASTPATH is on but the gambit_core extension is not "
                "importable. Build it from the repo root with: "
                "uv run maturin develop --release -m gambit_core/Cargo.toml "
                f"(original error: {exc})"
            ) from exc
    return _gambit_core
