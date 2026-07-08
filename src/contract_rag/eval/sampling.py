"""Stratified sampling for golden-set ops (F1): build a representative subset across
strata (doc-type × dirtiness × clause-complexity) deterministically. Pure + seeded so a
re-drawn sample is reproducible."""
from __future__ import annotations

import random
from collections.abc import Callable, Hashable

from contract_rag.eval.golden import GoldenDoc


def stratify(items: list, key_fn: Callable[[object], Hashable]) -> dict:
    groups: dict[Hashable, list] = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    return groups


def _largest_remainder(sizes: dict[Hashable, int], n: int) -> dict[Hashable, int]:
    total = sum(sizes.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in sizes}
    n = min(n, total)
    raw = {k: n * size / total for k, size in sizes.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainder = n - sum(floors.values())
    # hand out the remaining slots to the largest fractional parts (stable by key for determinism)
    order = sorted(sizes, key=lambda k: (-(raw[k] - floors[k]), str(k)))
    for k in order[:remainder]:
        floors[k] += 1
    return {k: min(floors[k], sizes[k]) for k in sizes}


def stratified_sample(
    items: list, key_fn: Callable[[object], Hashable], n: int, seed: int = 0
) -> list:
    groups = stratify(items, key_fn)
    sizes = {k: len(v) for k, v in groups.items()}
    quota = _largest_remainder(sizes, n)
    rng = random.Random(seed)
    out: list = []
    for k in groups:  # insertion order -> deterministic
        members = list(groups[k])
        rng.shuffle(members)
        out.extend(members[: quota[k]])
    return out


def golden_stratum_key(doc: GoldenDoc) -> tuple:
    filled = sum(1 for v in doc.facts.values() if v)
    bucket = "low" if filled <= 1 else ("med" if filled <= 3 else "high")
    return ("clause_complexity", bucket)
