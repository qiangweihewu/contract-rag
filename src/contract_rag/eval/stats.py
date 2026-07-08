"""Uncertainty for the headline eval numbers. Every metric in this codebase (field-F1,
source-accuracy, ...) has so far been a single-run point estimate over the golden set —
fine for "is this above the floor" but not for "is constrained 0.661 vs rule 0.676 a real
difference, or run-to-run noise". Two stdlib-only, seeded tools close that gap:

- `bootstrap_metric_ci`: percentile bootstrap over DOCS (rows), reusing `aggregate()` for
  the metric recompute so the F1/accuracy math is never duplicated.
- `paired_permutation_test`: a paired sign-flip test for A/B comparisons on the SAME doc
  set (e.g. docling vs router parse, rule vs constrained extract), where each doc is its
  own control.

Both take `rows` in exactly the shape `metrics.row_for()` produces — the same rows
`run_baseline` already builds and aggregates, just retained instead of discarded."""
from __future__ import annotations

import math
import random
from typing import Callable

from contract_rag.eval.metrics import aggregate


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile: rank = ceil(p * n), clamped to [1, n]."""
    n = len(sorted_vals)
    rank = max(1, min(n, math.ceil(p * n)))
    return sorted_vals[rank - 1]


def bootstrap_metric_ci(
    rows: list[dict], metric_fn: Callable[[list[dict]], float],
    n_boot: int = 1000, seed: int = 0, confidence: float = 0.95,
) -> dict:
    """Percentile bootstrap CI by resampling docs (rows) with replacement, same size as
    `rows`, recomputing `metric_fn` on each resample. Empty rows -> point/lo/hi all 0.0
    (nothing to resample)."""
    if not rows:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": n_boot, "confidence": confidence}

    point = metric_fn(rows)
    rng = random.Random(seed)
    n = len(rows)
    boot_vals = sorted(
        metric_fn([rows[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)
    )
    alpha = 1 - confidence
    lo = _percentile(boot_vals, alpha / 2)
    hi = _percentile(boot_vals, 1 - alpha / 2)
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_boot, "confidence": confidence}


def field_f1_of(rows: list[dict], vertical=None) -> float:
    """Convenience metric_fn: aggregate(rows, vertical)['field_f1']."""
    return aggregate(rows, vertical)["field_f1"]


def source_accuracy_of(rows: list[dict], vertical=None) -> float:
    """Convenience metric_fn: aggregate(rows, vertical)['source_accuracy']."""
    return aggregate(rows, vertical)["source_accuracy"]


def paired_permutation_test(
    rows_a: list[dict], rows_b: list[dict], metric_fn,
    n_perm: int = 2000, seed: int = 0,
) -> dict:
    """Paired sign-flip permutation test: rows_a[i] and rows_b[i] must be the same doc
    under configs A and B. Statistic = metric_fn(rows_a) - metric_fn(rows_b); each
    permutation independently swaps A/B per-doc with p=0.5 and recomputes the diff.
    Two-sided p-value = (1 + #{|perm_diff| >= |observed|}) / (1 + n_perm)."""
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"paired permutation test requires equal-length row sets (same docs): "
            f"{len(rows_a)} != {len(rows_b)}"
        )
    observed = metric_fn(rows_a) - metric_fn(rows_b)
    rng = random.Random(seed)
    n = len(rows_a)
    extreme = 0
    for _ in range(n_perm):
        perm_a, perm_b = [], []
        for i in range(n):
            if rng.random() < 0.5:
                perm_a.append(rows_a[i])
                perm_b.append(rows_b[i])
            else:
                perm_a.append(rows_b[i])
                perm_b.append(rows_a[i])
        diff = metric_fn(perm_a) - metric_fn(perm_b)
        if abs(diff) >= abs(observed):
            extreme += 1
    p_value = (1 + extreme) / (1 + n_perm)
    return {"observed_diff": observed, "p_value": p_value, "n_perm": n_perm}
