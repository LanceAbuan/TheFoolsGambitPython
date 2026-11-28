"""Frozen evaluation set: a stable, comparable progress metric.

The per-iteration `[METRICS]` line samples a random batch from the LIVE replay
buffer, so its distribution shifts every iteration - which is why `policy_ce`
swings 0.8 -> 3.7 without the model changing quality. This module freezes a set
of positions ONCE (from the replay buffer at freeze time) and scores every
checkpoint against exactly the same tensors, so the numbers are comparable
across iterations and across lineages.

It also replaces brittle argmax accuracy (research A88) with proper scoring
rules:
  * `policy_ce`     - cross-entropy against the frozen policy targets (nats)
  * `policy_top1`   - agreement with the target's argmax (kept, but only as a
                      coarse secondary signal)
  * `value_ce`      - cross-entropy against the frozen WDL target
  * `value_brier`   - Brier score over the WDL probabilities (lower is better)
  * `mean_p_true`   - mean probability the model assigns to the true outcome
  * `score_mae`     - |E[score] - actual score| averaged, score = P(win)+0.5P(draw)

Usage (build once per run, then it is scored automatically each iteration):

    python -m training.eval_set build --run-root runs/scratch/second-light-v2 \
        --out runs/scratch/second-light-v2/eval_set.npz --count 2048
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Iterable

import numpy as np

EVAL_SET_FILENAME = "eval_set.npz"
PLANES_SHAPE = (104, 8, 8)
POLICY_SIZE = 4672


def _example_to_arrays(example: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    planes = np.asarray(example["board_tensor"], dtype=np.float16)
    policy = np.asarray(example["policy"], dtype=np.float16)
    if planes.shape != PLANES_SHAPE:
        raise ValueError(f"plane shape {planes.shape} != {PLANES_SHAPE}")
    if policy.shape != (POLICY_SIZE,):
        raise ValueError(f"policy shape {policy.shape} != ({POLICY_SIZE},)")
    return planes, policy, float(example["value"])


def build_eval_set(
    examples: Iterable[dict[str, Any]],
    out_path: pathlib.Path | str,
    count: int = 2048,
    seed: int = 0,
) -> pathlib.Path:
    """Freeze up to ``count`` examples (evenly strided, seeded) into an npz."""
    pool = list(examples)
    if not pool:
        raise ValueError("cannot build an eval set from zero examples")
    rng = np.random.default_rng(seed)
    if len(pool) > count:
        # Even stride keeps the whole training window represented instead of
        # whatever happens to sit at one end of the deque.
        stride = len(pool) / count
        picked = [pool[int(i * stride)] for i in range(count)]
        rng.shuffle(picked)
        pool = picked

    planes = np.stack([_example_to_arrays(ex)[0] for ex in pool])
    policies = np.stack([_example_to_arrays(ex)[1] for ex in pool])
    values = np.array([float(ex["value"]) for ex in pool], dtype=np.float32)

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, planes=planes, policies=policies, values=values)
    return out


def load_eval_set(path: pathlib.Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {
            "planes": npz["planes"].astype(np.float32),
            "policies": npz["policies"].astype(np.float32),
            "values": npz["values"].astype(np.float32),
        }


def score(
    model: Any,
    data: dict[str, np.ndarray],
    device: Any,
    *,
    batch: int = 256,
    value_head: str = "wdl",
) -> dict[str, float]:
    """Score ``model`` against a frozen eval set; lower CE/Brier is better."""
    import torch

    planes = torch.from_numpy(data["planes"])
    policies = torch.from_numpy(data["policies"])
    values = torch.from_numpy(data["values"])

    was_training = model.training
    model.eval()
    total = planes.shape[0]
    policy_ce = 0.0
    policy_top1 = 0.0
    value_ce = 0.0
    value_brier = 0.0
    mean_p_true = 0.0
    score_mae = 0.0
    with torch.no_grad():
        for start in range(0, total, batch):
            sl = slice(start, min(start + batch, total))
            p_batch = planes[sl].to(device)
            t_batch = policies[sl].to(device)
            v_batch = values[sl]
            out_p, out_v = model(p_batch)
            log_p = torch.log_softmax(out_p, dim=-1)

            policy_ce += float(-(t_batch * log_p).sum(dim=-1).sum().item())
            policy_top1 += float(
                (out_p.argmax(dim=-1) == t_batch.argmax(dim=-1)).float().sum().item()
            )

            if value_head == "wdl":
                # VALUE CLASS ORDER IS 0=win, 1=draw, 2=loss (trainer.wdl_targets).
                # The first version of this scorer assumed -1..+1 -> 0..2 and so
                # read p(loss) as p(true); every value metric it printed was
                # measured against the wrong class.
                cls = torch.round(1.0 - v_batch).long().clamp(0, 2).to(device)
                log_v = torch.log_softmax(out_v, dim=-1)
                probs = log_v.exp()
                onehot = torch.zeros_like(probs)
                onehot[torch.arange(cls.shape[0]), cls] = 1.0
                value_ce += float(-(onehot * log_v).sum(dim=-1).sum().item())
                value_brier += float(((probs - onehot) ** 2).sum(dim=-1).sum().item())
                mean_p_true += float((probs * onehot).sum(dim=-1).sum().item())
                # Expected score = P(win) + 0.5 * P(draw); actual = 1 / 0.5 / 0.
                expected = probs[:, 0] * 1.0 + probs[:, 1] * 0.5
                actual = (2.0 - cls.float()) * 0.5
                score_mae += float((expected - actual).abs().sum().item())
            else:
                v_pred = out_v.reshape(-1)
                value_ce += float(torch.nn.functional.mse_loss(v_pred, v_batch, reduction="sum"))
                value_brier += float(((v_pred - v_batch) ** 2).sum())
                mean_p_true += float(v_pred.sum())
                score_mae += float(((v_pred * 0.5 + 0.5) - (v_batch * 0.5 + 0.5)).abs().sum())

    if was_training:
        model.train()
    n = max(total, 1)
    return {
        "eval_n": float(total),
        "eval_policy_ce": policy_ce / n,
        "eval_policy_top1": policy_top1 / n,
        "eval_value_ce": value_ce / n,
        "eval_value_brier": value_brier / n,
        "eval_mean_p_true": mean_p_true / n,
        "eval_score_mae": score_mae / n,
    }


def build_from_run(
    run_root: pathlib.Path | str,
    count: int = 2048,
    seed: int = 0,
) -> pathlib.Path:
    """Build the eval set from a run's newest state file (its replay buffer)."""
    import torch

    root = pathlib.Path(run_root)
    states = sorted((root / "state").glob("state-*.pt"))
    if not states:
        raise FileNotFoundError(f"no state files under {root / 'state'}")
    state = torch.load(states[-1], map_location="cpu", weights_only=False)
    blob = state.get("buffer_npz")
    if not blob:
        raise ValueError(
            f"{states[-1].name} carries no replay buffer (saved under memory pressure); "
            "pass --from-spool instead"
        )
    from .resume import _examples_from_npz

    examples = _examples_from_npz(bytes(blob))
    return build_eval_set(examples, root / EVAL_SET_FILENAME, count=count, seed=seed)


def build_from_spool(
    spool_dir: pathlib.Path | str,
    out_path: pathlib.Path | str | None = None,
    count: int = 2048,
    seed: int = 0,
    worker_prefix: str | None = None,
) -> pathlib.Path:
    """Build a NEUTRAL eval set from finished spool games.

    ``worker_prefix="sf"`` builds it from Stockfish-labelled games, which makes
    the metric comparable ACROSS lineages and champions - the self-play-built
    set is biased toward whatever model generated it.
    """
    from .rust_workers import INBOX_DIRNAME, PROCESSED_DIRNAME, RecordError, read_game_record

    inbox = pathlib.Path(spool_dir) / INBOX_DIRNAME
    candidates: list[pathlib.Path] = []
    for directory in (inbox / PROCESSED_DIRNAME, inbox):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("game-*.jsonl")))
    if worker_prefix:
        candidates = [
            p for p in candidates if p.stem.split("-")[1:2] == [worker_prefix]
        ]
    if not candidates:
        raise FileNotFoundError(
            f"no spool games under {inbox}"
            + (f" for worker {worker_prefix!r}" if worker_prefix else "")
        )

    examples: list[dict[str, Any]] = []
    # Newest first so a bounded read still covers recent play.
    for path in reversed(candidates[-max(count * 4, 32):]):
        try:
            record = read_game_record(path)
        except RecordError:
            continue
        examples.extend(record["examples"])
        if len(examples) >= count * 4:
            break
    out = pathlib.Path(out_path) if out_path else inbox.parent / EVAL_SET_FILENAME
    return build_eval_set(examples, out, count=count, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "build-sf", "score"])
    parser.add_argument("--run-root", default="runs/scratch/second-light-v2")
    parser.add_argument("--out", default=None)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    root = pathlib.Path(args.run_root)
    out = pathlib.Path(args.out) if args.out else root / EVAL_SET_FILENAME
    if args.command == "build":
        path = build_from_run(root, count=args.count, seed=args.seed)
        data = load_eval_set(out if args.out is None else path)
        print(f"[eval-set] wrote {path} with {data['planes'].shape[0]} positions")
        return 0
    if args.command == "build-sf":
        path = build_from_spool(
            root / "spool", out_path=out, count=args.count, seed=args.seed, worker_prefix="sf"
        )
        data = load_eval_set(path)
        print(f"[eval-set] wrote {path} from Stockfish rows: {data['planes'].shape[0]} positions")
        return 0

    import torch

    from .pipeline import _load_champion

    ckpt = pathlib.Path(args.checkpoint) if args.checkpoint else sorted(
        (root / "checkpoints").glob("ckpt-*.pt")
    )[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = _load_champion(ckpt, device)
    if loaded is None:
        raise SystemExit(f"could not load {ckpt}")
    model, tag, _elo = loaded
    data = load_eval_set(out)
    scores = score(model, data, device, value_head="wdl")
    print(f"[eval-set] {tag}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
