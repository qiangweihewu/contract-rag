"""Golden-set versioning (F1): a content hash so a re-built golden set is identifiable
and a diff shows exactly what changed between versions. Order-independent — re-building
from the same CUAD source yields the same version id."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from contract_rag.eval.golden import GoldenDoc


def _canonical(golden: list[GoldenDoc]) -> str:
    rows = sorted(
        ({"doc_id": g.doc_id, "facts": dict(sorted(g.facts.items()))} for g in golden),
        key=lambda r: r["doc_id"],
    )
    return json.dumps(rows, sort_keys=True, ensure_ascii=False)


def version_id(golden: list[GoldenDoc]) -> str:
    return hashlib.sha256(_canonical(golden).encode("utf-8")).hexdigest()[:16]


class GoldenSetManifest(BaseModel):
    version: str
    n_docs: int
    doc_ids: list[str]
    strata: dict[str, int] = Field(default_factory=dict)
    created_at: str | None = None


def build_manifest(
    golden: list[GoldenDoc], created_at: str | None = None,
    key_fn: Callable[[GoldenDoc], object] | None = None,
) -> GoldenSetManifest:
    strata: dict[str, int] = {}
    if key_fn is not None:
        for g in golden:
            key = str(key_fn(g))
            strata[key] = strata.get(key, 0) + 1
    return GoldenSetManifest(
        version=version_id(golden),
        n_docs=len(golden),
        doc_ids=sorted(g.doc_id for g in golden),
        strata=strata,
        created_at=created_at,
    )


def write_manifest(manifest: GoldenSetManifest, path: str | Path) -> None:
    Path(path).write_text(manifest.model_dump_json(indent=2))


def read_manifest(path: str | Path) -> GoldenSetManifest:
    return GoldenSetManifest.model_validate_json(Path(path).read_text())


def diff_manifests(old: GoldenSetManifest, new: GoldenSetManifest) -> dict:
    old_ids, new_ids = set(old.doc_ids), set(new.doc_ids)
    return {
        "added": sorted(new_ids - old_ids),
        "removed": sorted(old_ids - new_ids),
        "version_changed": old.version != new.version,
    }
