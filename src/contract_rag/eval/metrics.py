from __future__ import annotations

from typing import TYPE_CHECKING

from contract_rag.eval.golden import GoldenDoc, normalize
from contract_rag.ir import DocumentIR

if TYPE_CHECKING:
    from contract_rag.verticals.base import Vertical

_SET_MATCH_THRESHOLD = 0.5

# Error taxonomy (market feedback: "average accuracy" hides the failure mode — systems
# fail as Omitters, silently dropping clauses, or Inventors, fabricating values).
# Every (doc, field) with labeled gold is exactly one of correct/omission/invention;
# a prediction on a field with no gold is `unscored` — unverifiable, never an invention,
# mirroring aggregate()'s zero-gold exclusion. `wrong_span` is an orthogonal flag:
# the value is right but its source-attribution fails.
TAXONOMY_LABELS = ("correct", "omission", "invention", "unscored")


def classify_field(*, gold_nonempty: bool, pred_nonempty: bool, correct: bool) -> str | None:
    """One taxonomy label per (doc, field); None when there is nothing to judge
    (no gold and no prediction)."""
    if gold_nonempty:
        if correct:
            return "correct"
        return "omission" if not pred_nonempty else "invention"
    return "unscored" if pred_nonempty else None


# Field risk tiers — error rates on high-risk clauses (caps, penalties, termination)
# run far above simple ones, so one blended F1 flatters the system exactly where it
# matters most. `field_risk` is an OPTIONAL vertical seam (a mapping or a zero-arg
# method); resolved defensively so the Vertical protocol and any third-party vertical
# without it stay unbroken.
RISK_TIERS = ("high", "medium", "low")
_DEFAULT_RISK = "medium"


def field_risk_map(vertical) -> dict[str, str]:
    """Per-field risk tier for a vertical; missing/unknown levels default to medium."""
    raw = getattr(vertical, "field_risk", None)
    mapping = dict(raw() if callable(raw) else (raw or {}))
    return {
        n: (mapping.get(n) if mapping.get(n) in RISK_TIERS else _DEFAULT_RISK)
        for n in vertical.field_names
    }


def _default():
    from contract_rag.verticals.registry import default_vertical
    return default_vertical()


def _entity_set(vertical, value: str) -> set[str]:
    return {normalize(e) for e in vertical.entities(value) if normalize(e)}


def field_scores(pred, gold: GoldenDoc, vertical: Vertical | None = None) -> dict[str, bool]:
    v = vertical or _default()
    out: dict[str, bool] = {}
    for name in v.field_names:
        gold_raw = gold.facts.get(name, "")
        if name in v.set_fields:
            p = _entity_set(v, getattr(pred, name).value)
            g = _entity_set(v, gold_raw)
            out[name] = bool(p and g) and len(p & g) / len(p | g) >= _SET_MATCH_THRESHOLD
        else:
            pred_val = normalize(v.canonicalize_value(name, getattr(pred, name).value))
            gold_val = normalize(v.canonicalize_value(name, gold_raw))
            out[name] = bool(pred_val) and pred_val == gold_val
    return out


def source_attribution_ok(pred, ir: DocumentIR, vertical: Vertical | None = None) -> dict[str, bool]:
    v = vertical or _default()
    block_text = {b.block_id: normalize(b.text) for b in ir.blocks}
    out: dict[str, bool] = {}
    for name in v.field_names:
        clause = getattr(pred, name)
        if not clause.value:
            out[name] = False
            continue
        if name in v.judgment_fields:
            out[name] = True
            continue
        cited = block_text.get(clause.source_block_id or "", "")
        if name in v.set_fields:
            ents = [normalize(e) for e in v.entities(clause.value)]
            out[name] = bool(ents) and all(e in cited for e in ents)
        else:
            out[name] = normalize(clause.value) in cited
    return out


def row_for(pred, gold: GoldenDoc, ir: DocumentIR, vertical: Vertical | None = None) -> dict:
    v = vertical or _default()
    return {
        "scores": field_scores(pred, gold, v),
        "source": source_attribution_ok(pred, ir, v),
        "pred_nonempty": {n: bool(getattr(pred, n).value) for n in v.field_names},
        "gold_nonempty": {n: bool(gold.facts.get(n, "")) for n in v.field_names},
    }


def aggregate(rows: list[dict], vertical: Vertical | None = None) -> dict:
    v = vertical or _default()
    fields = v.field_names
    support = {n: sum(r["gold_nonempty"][n] for r in rows) for n in fields}
    # A field with no gold anywhere is unmeasurable on this dataset — exclude it from the
    # aggregate so its (unjudgeable) predictions don't count as false positives.
    scored = [n for n in fields if support[n] > 0]

    tp = sum(r["scores"][n] for r in rows for n in scored)
    pred_pos = sum(r["pred_nonempty"][n] for r in rows for n in scored)
    gold_pos = sum(support[n] for n in scored)

    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / gold_pos if gold_pos else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    per_field = {
        n: (sum(r["scores"][n] for r in rows) / len(rows) if rows else 0.0) for n in fields
    }
    per_field_on_labeled = {
        n: (sum(r["scores"][n] for r in rows) / support[n] if support[n] else None) for n in fields
    }
    src_total = sum(r["pred_nonempty"][n] for r in rows for n in scored)
    src_ok = sum(r["source"][n] for r in rows for n in scored)
    source_accuracy = src_ok / src_total if src_total else 0.0

    counters = ("wrong_span", *TAXONOMY_LABELS)
    tax_per_field = {n: dict.fromkeys(counters, 0) for n in fields}
    for r in rows:
        for n in fields:
            label = classify_field(gold_nonempty=r["gold_nonempty"][n],
                                   pred_nonempty=r["pred_nonempty"][n],
                                   correct=r["scores"][n])
            if label:
                tax_per_field[n][label] += 1
            if r["scores"][n] and not r["source"][n]:
                tax_per_field[n]["wrong_span"] += 1
    tax_totals = {k: sum(f[k] for f in tax_per_field.values()) for k in counters}

    # Per-tier rollup: same micro-F1 (restricted to labeled fields in the tier) and
    # taxonomy sums, so a report can lead with the high-risk row, not the blend.
    risk = field_risk_map(v)
    per_tier: dict[str, dict] = {}
    for tier in RISK_TIERS:
        tier_fields = [n for n in fields if risk[n] == tier]
        if not tier_fields:
            continue
        t_scored = [n for n in tier_fields if support[n] > 0]
        t_tp = sum(r["scores"][n] for r in rows for n in t_scored)
        t_pred = sum(r["pred_nonempty"][n] for r in rows for n in t_scored)
        t_gold = sum(support[n] for n in t_scored)
        t_p = t_tp / t_pred if t_pred else 0.0
        t_r = t_tp / t_gold if t_gold else 0.0
        t_f1 = (2 * t_p * t_r / (t_p + t_r)) if (t_p + t_r) else 0.0
        per_tier[tier] = {
            "fields": tier_fields,
            "support": t_gold,
            # None when the tier has no gold anywhere: unmeasurable, not zero.
            "f1_on_labeled": t_f1 if t_scored else None,
            "taxonomy": {k: sum(tax_per_field[n][k] for n in tier_fields) for k in counters},
        }

    return {
        "field_f1": f1, "precision": precision, "recall": recall,
        "per_field": per_field, "per_field_on_labeled": per_field_on_labeled,
        "support": support, "scored_fields": scored,
        "source_accuracy": source_accuracy, "n_docs": len(rows),
        "error_taxonomy": {"per_field": tax_per_field, "totals": tax_totals},
        "risk_tiers": {"field_risk": risk, "per_tier": per_tier},
    }
