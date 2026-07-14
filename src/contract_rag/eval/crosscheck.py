"""Dual-engine omission crosscheck measured on cached FinCriticalED IRs
(spec 2026-07-14). Zero re-parsing: reads the paddle IRs (primary) and the
dots.ocr IRs (verifier) cached by the 2026-07-13 vision-OCR run.

Pre-registered bar (fixed before numbers): flag-recall >= 0.5 AND page
false-alarm rate <= 0.2. Overall recall is the bar; the digit-fact split is
reported alongside because pure-alpha entity facts are un-catchable by design.

Env: CROSSCHECK_PRIMARY_CACHE (default ~/.cache/contract-rag/fincriticaled-run/ir),
CROSSCHECK_VERIFIER_CACHE (default ~/.cache/contract-rag/fincriticaled-run-dots/ir),
FINCRITICAL_DIR (gold; auto-download fallback as in eval/fincritical),
CROSSCHECK_SET_SIZE (default 100), CROSSCHECK_MIN_MISSING (default 1),
CROSSCHECK_OUT (optional JSON dump; parent dirs created)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from contract_rag.clean.crosscheck import crosscheck, critical_tokens
from contract_rag.eval.fincritical import (
    ensure_dataset,
    evaluate_page,
    load_samples,
    parse_gold_html,
)
from contract_rag.ir import DocumentIR

RECALL_BAR = 0.5
FALSE_ALARM_BAR = 0.2


class PageRow(BaseModel):
    stem: str
    n_gold: int
    n_omitted: int
    flagged: bool
    missing_count: int
    caught: int


class Summary(BaseModel):
    n_pages: int
    n_skipped_missing_ir: int
    n_omitted_facts: int
    caught_facts: int
    flag_recall: float | None
    digit_fact_recall: float | None
    n_clean_pages: int
    false_alarms: int
    false_alarm_rate: float | None
    by_kind: dict
    passed_bar: bool | None


def _load_ir(dir_: Path, stem: str) -> DocumentIR | None:
    f = Path(dir_) / f"{stem}.ir.json"
    return DocumentIR.model_validate_json(f.read_text()) if f.exists() else None


def evaluate_crosscheck(
    samples, primary_dir: Path, verifier_dir: Path, *, min_missing: int = 1
) -> tuple[list[PageRow], Summary]:
    drift = [(i, s.page_id) for i, s in enumerate(samples) if s.page_id != i]
    if drift:
        raise RuntimeError(
            f"sample page_ids no longer contiguous from 0 (first drift: {drift[0]}); "
            "cache stems fincritical_{i} would mispair gold vs IR - re-run "
            "eval.fincritical to rebuild caches or fix stem derivation"
        )
    rows: list[PageRow] = []
    skipped = 0
    omitted_total = caught_total = 0
    digit_omitted = digit_caught = 0
    clean_pages = false_alarms = 0
    by_kind: dict[str, dict[str, int]] = {}
    for i, sample in enumerate(samples):
        stem = f"fincritical_{i}"
        p_ir = _load_ir(primary_dir, stem)
        v_ir = _load_ir(verifier_dir, stem)
        if p_ir is None or v_ir is None:
            skipped += 1
            continue
        facts = parse_gold_html(sample.gold_html)
        outcomes = evaluate_page(p_ir, facts)
        omitted = [o for o in outcomes if not o.in_document]
        cc = crosscheck(p_ir, v_ir, min_missing=min_missing)
        missing = set(cc.missing_tokens)
        caught = 0
        for o in omitted:
            ks = by_kind.setdefault(o.kind, {"omitted": 0, "caught": 0})
            ks["omitted"] += 1
            toks = critical_tokens(o.value)
            hit = cc.flagged and bool(toks & missing)
            if toks:
                digit_omitted += 1
                digit_caught += int(hit)
            if hit:
                caught += 1
                ks["caught"] += 1
        omitted_total += len(omitted)
        caught_total += caught
        if not omitted:
            clean_pages += 1
            false_alarms += int(cc.flagged)
        rows.append(PageRow(stem=stem, n_gold=len(outcomes), n_omitted=len(omitted),
                            flagged=cc.flagged, missing_count=cc.missing_count,
                            caught=caught))
    recall = caught_total / omitted_total if omitted_total else None
    drecall = digit_caught / digit_omitted if digit_omitted else None
    far = false_alarms / clean_pages if clean_pages else None
    passed = (recall >= RECALL_BAR and far <= FALSE_ALARM_BAR) \
        if (recall is not None and far is not None) else None
    return rows, Summary(
        n_pages=len(rows), n_skipped_missing_ir=skipped,
        n_omitted_facts=omitted_total, caught_facts=caught_total,
        flag_recall=round(recall, 3) if recall is not None else None,
        digit_fact_recall=round(drecall, 3) if drecall is not None else None,
        n_clean_pages=clean_pages, false_alarms=false_alarms,
        false_alarm_rate=round(far, 3) if far is not None else None,
        by_kind=by_kind, passed_bar=passed,
    )


def format_report(rows: list[PageRow], s: Summary) -> str:
    kinds = ", ".join(
        f"{k} {v['caught']}/{v['omitted']}" for k, v in sorted(s.by_kind.items())
    )
    verdict = "PASS" if s.passed_bar else ("FAIL" if s.passed_bar is not None else "N/A")
    return "\n".join([
        "=== dual-engine crosscheck (primary vs verifier) ===",
        f"pages {s.n_pages} (skipped {s.n_skipped_missing_ir} missing IR)",
        f"omitted facts {s.n_omitted_facts}, caught {s.caught_facts}"
        f" -> flag-recall {s.flag_recall}",
        f"digit-fact recall {s.digit_fact_recall}",
        f"clean pages {s.n_clean_pages}, false alarms {s.false_alarms}"
        f" -> false-alarm rate {s.false_alarm_rate}",
        f"by kind: {kinds}",
        f"pre-registered bar (recall>={RECALL_BAR}, false-alarm<={FALSE_ALARM_BAR}):"
        f" {verdict}",
    ])


def main() -> None:
    home = Path.home() / ".cache" / "contract-rag"
    primary = Path(os.environ.get("CROSSCHECK_PRIMARY_CACHE",
                                  str(home / "fincriticaled-run" / "ir")))
    verifier = Path(os.environ.get("CROSSCHECK_VERIFIER_CACHE",
                                   str(home / "fincriticaled-run-dots" / "ir")))
    fin_dir = os.environ.get("FINCRITICAL_DIR")
    data_dir = ensure_dataset(Path(fin_dir) if fin_dir else home / "fincriticaled")
    cap = int(os.environ.get("CROSSCHECK_SET_SIZE", "100"))
    mm = int(os.environ.get("CROSSCHECK_MIN_MISSING", "1"))
    samples = load_samples(data_dir, cap=cap)
    rows, summary = evaluate_crosscheck(samples, primary, verifier, min_missing=mm)
    print(format_report(rows, summary))
    out = os.environ.get("CROSSCHECK_OUT")
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"summary": summary.model_dump(),
                                 "rows": [r.model_dump() for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
