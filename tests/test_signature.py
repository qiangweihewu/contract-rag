"""Unit tests for the signature-presence detector (eval/signature.py) and the
hoisted GEDI groundtruth label (eval/gedi.has_signature_zone). All dep-free:
hand-built IRs and fake GEDI XML, no OCR / network / cache."""
from __future__ import annotations

import pytest

from contract_rag.eval.gedi import PageZones, Zone, has_signature_zone, parse_gedi
from contract_rag.eval.signature import (
    SignaturePrediction,
    detect_signature,
    evaluate_predictions,
    format_eval,
)
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _block(text: str, conf: float = 0.99, bbox: tuple | None = None, bid: str | None = None) -> DocBlock:
    return DocBlock(
        block_id=bid or f"#/b/{abs(hash((text, bbox))) % 10000}",
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(page=1, x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]) if bbox else None,
        confidence=conf,
        source_engine="paddleocr",
    )


def _ir(blocks: list[DocBlock]) -> DocumentIR:
    return DocumentIR(
        doc_id="d", source_uri="file:///d", file_hash="h", mime_type="application/pdf",
        blocks=blocks,
    )


# ---------------------------------------------------------------- GEDI label

def test_has_signature_zone():
    assert has_signature_zone(PageZones(width=1, height=1, zones=[Zone(kind="DLSignature", x0=0, y0=0, x1=1, y1=1)]))
    assert not has_signature_zone(PageZones(width=1, height=1, zones=[Zone(kind="DLLogo", x0=0, y0=0, x1=1, y1=1)]))
    assert not has_signature_zone(PageZones(width=1, height=1, zones=[]))  # zero-zone = genuine negative


def test_parse_gedi_then_label_roundtrip():
    xml = (
        '<DL_DOCUMENT><DL_PAGE width="2000" height="3000">'
        '<DL_ZONE gedi_type="DLSignature" col="100" row="2500" width="300" height="120"/>'
        '<DL_ZONE gedi_type="DLLogo" col="50" row="40" width="400" height="200"/>'
        "</DL_PAGE></DL_DOCUMENT>"
    )
    pz = parse_gedi(xml)
    assert has_signature_zone(pz)
    assert len(pz.zones) == 2


# ---------------------------------------------------------------- closing signal

def test_closing_salutation_signed():
    ir = _ir([
        _block("Please find enclosed the report."),
        _block("Sincerely,", bbox=(200, 2000, 500, 2050), bid="#/close"),
        _block("J. B. Boder", bbox=(200, 2200, 600, 2260)),
    ])
    pred = detect_signature(ir)
    assert pred.signed
    assert "closing" in pred.signals
    assert "#/close" in pred.evidence_block_ids
    assert pred.confidence >= 0.9


def test_sigword_by_and_slash_s():
    for cue in ["By: John Smith", "/s/ Jane Doe", "duly authorized representative"]:
        pred = detect_signature(_ir([_block("body"), _block(cue)]))
        assert pred.signed, cue
        assert "sigword" in pred.signals


# ------------------------------------------------------------ signature block

def test_signature_block_squiggle_over_name():
    # low-confidence squiggle just above a typed name line in the lower page
    ir = _ir([
        _block("body text near the top", bbox=(200, 400, 900, 460)),
        _block("wm.Hobbr", conf=0.62, bbox=(220, 1900, 480, 1990), bid="#/squiggle"),
        _block("Wm. D. Hobbs", conf=1.0, bbox=(220, 2050, 520, 2110), bid="#/name"),
    ])
    pred = detect_signature(ir)
    assert pred.signed
    assert pred.signals == ["sigblock"]
    assert set(pred.evidence_block_ids) == {"#/name", "#/squiggle"}


def test_signature_block_requires_low_confidence_token():
    # a high-confidence typed name with only high-confidence text above → no squiggle
    ir = _ir([
        _block("mailing list line", conf=1.0, bbox=(220, 1900, 480, 1990)),
        _block("Wm. D. Hobbs", conf=1.0, bbox=(220, 2050, 520, 2110)),
    ])
    assert not detect_signature(ir).signed


def test_signature_block_only_lower_half():
    # name+squiggle in the TOP half (e.g. a letterhead recipient) must not fire
    ir = _ir([
        _block("squiggle", conf=0.5, bbox=(220, 100, 480, 190)),
        _block("Wm. D. Hobbs", conf=1.0, bbox=(220, 250, 520, 310)),
        _block("lots of body text below", bbox=(220, 2800, 900, 2900)),
    ])
    assert not detect_signature(ir).signed


# ---------------------------------------------------------------- negatives

def test_unsigned_plain_page():
    ir = _ir([
        _block("MEMORANDUM", bbox=(200, 100, 900, 160)),
        _block("This is a continued page with body text only.", bbox=(200, 500, 900, 560)),
        _block("...continued...", bbox=(200, 2800, 500, 2860)),
    ])
    pred = detect_signature(ir)
    assert not pred.signed
    assert pred.signals == []
    assert pred.evidence_block_ids == []
    assert pred.confidence < 0.5  # absence of evidence is weak, not certain


def test_confidence_combines_multiple_signals():
    ir = _ir([
        _block("Sincerely,", bbox=(200, 2000, 500, 2050)),
        _block("By: authorized officer", bbox=(200, 2400, 700, 2460)),
    ])
    pred = detect_signature(ir)
    # probabilistic OR: 1 - (1-.9)(1-.8) = 0.98 > either alone
    assert pred.confidence > 0.9
    assert set(pred.signals) == {"closing", "sigword"}


def test_prediction_is_serializable():
    pred = detect_signature(_ir([_block("Sincerely,", bbox=(1, 2000, 2, 2050))]))
    assert isinstance(pred, SignaturePrediction)
    assert set(pred.model_dump()) == {"signed", "confidence", "evidence_block_ids", "signals"}


# ---------------------------------------------------------------- evaluation

def test_evaluate_predictions_confusion_and_baseline():
    # 3 signed, 2 unsigned; predict all 3 signed right, 1 unsigned wrong
    pairs = [(True, True), (True, True), (True, True), (True, False), (False, False)]
    ev = evaluate_predictions(pairs)
    assert ev.matrix.tp == 3 and ev.matrix.fp == 1 and ev.matrix.fn == 0 and ev.matrix.tn == 1
    assert ev.recall == 1.0
    assert ev.precision == pytest.approx(0.75)
    assert ev.n_signed == 3
    # always-signed baseline: precision 3/5, recall 1
    assert ev.baseline_precision == pytest.approx(0.6)


def test_evaluate_predictions_empty_raises():
    with pytest.raises(ValueError):
        evaluate_predictions([])


def test_format_eval_mentions_baseline():
    ev = evaluate_predictions([(True, True), (False, False)])
    out = format_eval(ev)
    assert "baseline" in out
    assert "precision" in out
