"""Unit tests for the FinCriticalED harness — dep-free: hand-built IRs, synthetic
gold HTML, a fake CSV. No network, no OCR, no PIL."""
from __future__ import annotations

from pathlib import Path

from contract_rag.eval.fincritical import (
    FactOutcome,
    GoldFact,
    PageResult,
    canon_fact_text,
    contains_value,
    decode_image,
    evaluate_page,
    format_report,
    load_samples,
    locate_block,
    parse_gold_html,
    reliability_table,
    strip_tags,
    summarize,
    threshold_for_accuracy,
)
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR
from contract_rag.text import tokenize


def _b(text, i, conf=1.0):
    return DocBlock(block_id=f"#/ocr/{i}", type=BlockType.PARAGRAPH, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=10 * i, x1=100, y1=10 * i + 9),
                    confidence=conf, source_engine="paddleocr")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


# ------------------------------------------------------------- canonicalization

def test_canon_drops_thousands_separators_and_currency():
    assert canon_fact_text("$1,200,000") == "1200000"
    assert canon_fact_text("1200000") == "1200000"


def test_canon_keeps_decimal_point_and_percent_and_sign():
    assert canon_fact_text("3.5%") == "3.5%"
    assert canon_fact_text("-5.2") == "-5.2"
    assert canon_fact_text("35") != canon_fact_text("3.5")   # decimal shift stays an error
    assert canon_fact_text("5.2") != canon_fact_text("-5.2")  # dropped sign stays an error


def test_canon_digit_digit_hyphen_is_a_range_not_a_sign():
    assert canon_fact_text("2023-2024") == "2023 2024"


def test_canon_lowercases_unescapes_and_collapses_whitespace():
    assert canon_fact_text("December&nbsp;31,  2024") == "december 31 2024"
    assert canon_fact_text("Apple Inc.") == "apple inc"


def test_contains_value_is_token_boundary():
    assert contains_value("3.5", "growth of 3.5 percent")
    assert not contains_value("3.5", "growth of 13.5 percent")
    assert contains_value("december 31 2024", "ended december 31 2024 the")


# ------------------------------------------------------------------- gold facts

GOLD = """<html><body>
<p>For the fiscal year ended <temporal>December 31, 2024</temporal>,
<reportingentity>Apple Inc.</reportingentity> reported revenue of
<monetaryunit>$1,200,000</monetaryunit>, up <number>3.5%</number> from
<financialconcepts>net income</financialconcepts> last year.</p>
</body></html>"""


def test_parse_gold_html_extracts_all_five_kinds():
    facts = parse_gold_html(GOLD)
    assert [(f.kind, f.value) for f in facts] == [
        ("temporal", "December 31, 2024"),
        ("reportingentity", "Apple Inc."),
        ("monetaryunit", "$1,200,000"),
        ("number", "3.5%"),
        ("financialconcepts", "net income"),
    ]


def test_parse_gold_html_context_is_tag_free_surrounding_text():
    facts = parse_gold_html(GOLD)
    rev = next(f for f in facts if f.kind == "monetaryunit")
    assert "reported revenue of" in rev.context
    assert "<" not in rev.context


def test_parse_gold_html_strips_nested_tags_and_skips_empty_values():
    text = "<p><monetaryunit>$<number>5</number> million</monetaryunit> and <temporal></temporal></p>"
    facts = parse_gold_html(text)
    # the outer tag wins and its value is tag-stripped; the empty temporal is skipped
    assert [(f.kind, f.value) for f in facts] == [("monetaryunit", "$ 5 million")]


def test_strip_tags_unescapes_entities():
    assert strip_tags("<b>AT&amp;T</b> Inc.") == "AT&T Inc."


# ---------------------------------------------------------------------- locating

def _token_sets(texts):
    return [set(tokenize(canon_fact_text(t))) for t in texts]


def test_locate_block_prefers_rare_context_tokens():
    fact = GoldFact(kind="number", value="3.5%", context="revenue increased by from acquisitions")
    sets = _token_sets([
        "the company the company the company",       # boilerplate
        "revenue increased by 3.5% from acquisitions",  # the real home
        "revenue is recognized when",
    ])
    assert locate_block(fact, sets) == 1


def test_locate_block_excludes_value_tokens_so_garbled_values_still_locate():
    # the block lost the value ("3.S%") — context words alone must still find it
    fact = GoldFact(kind="number", value="3.5%", context="operating margin improved to")
    sets = _token_sets(["cash and equivalents", "operating margin improved to 3.S%"])
    assert locate_block(fact, sets) == 1


def test_locate_block_returns_none_without_enough_context_overlap():
    fact = GoldFact(kind="number", value="42", context="entirely absent words here")
    assert locate_block(fact, _token_sets(["nothing matches this block"])) is None
    assert locate_block(fact, []) is None


# ------------------------------------------------------------------ evaluate_page

def test_evaluate_page_survived_fact_pairs_the_value_block_confidence():
    ir = _ir([
        _b("some heading elsewhere", 0, conf=0.5),
        _b("total revenue for the year was $1,200,000 in cash", 1, conf=0.97),
    ])
    facts = [GoldFact(kind="monetaryunit", value="$1,200,000",
                      context="total revenue for the year was in cash")]
    (o,) = evaluate_page(ir, facts)
    assert o.in_document and o.located and o.correct
    assert o.confidence == 0.97  # the block that holds the value, not the context match


def test_evaluate_page_omitted_fact_pairs_context_block_and_is_incorrect():
    ir = _ir([_b("total revenue for the year was $1,2OO,0OO in cash", 0, conf=0.62)])
    facts = [GoldFact(kind="monetaryunit", value="$1,200,000",
                      context="total revenue for the year was in cash")]
    (o,) = evaluate_page(ir, facts)
    assert not o.in_document
    assert o.located and o.correct is False
    assert o.confidence == 0.62  # the context block: where the fact belonged


def test_evaluate_page_value_split_across_lines_attributes_to_its_block():
    # value's digits and unit land on adjacent OCR lines (reading-order window)
    ir = _ir([
        _b("net deferred tax assets valuation allowance", 0, conf=0.9),
        _b("1,234", 1, conf=0.88),
        _b("thousand as of year end", 2, conf=0.9),
    ])
    facts = [GoldFact(kind="number", value="1,234 thousand",
                      context="net deferred tax assets valuation allowance")]
    (o,) = evaluate_page(ir, facts)
    assert o.in_document and o.correct
    assert o.confidence == 0.88  # first block of the ±window span


def test_evaluate_page_correct_uses_whole_doc_not_context_neighbourhood():
    # value survives far (in reading order) from its textual context — still correct
    ir = _ir(
        [_b("registrant financial highlights for the period", 0, conf=0.9)]
        + [_b(f"filler line {i}", i, conf=0.9) for i in range(1, 20)]
        + [_b("42", 20, conf=0.95)]
    )
    facts = [GoldFact(kind="number", value="42",
                      context="registrant financial highlights for the period")]
    (o,) = evaluate_page(ir, facts)
    assert o.in_document and o.correct
    assert o.confidence == 0.95  # the distant value block, not the context block


def test_evaluate_page_omitted_and_uncontextualizable_fact_is_unlocated():
    ir = _ir([_b("completely unrelated boilerplate text", 0)])
    facts = [GoldFact(kind="temporal", value="December 31, 2024",
                      context="fiscal year ended for the registrant")]
    (o,) = evaluate_page(ir, facts)
    assert not o.located and o.confidence is None and o.correct is None
    assert not o.in_document


def test_evaluate_page_drops_unmeasurable_empty_canon_facts():
    ir = _ir([_b("consideration of $ in aggregate", 0)])
    facts = [GoldFact(kind="monetaryunit", value="$", context="consideration of in aggregate")]
    assert evaluate_page(ir, facts) == []


# ------------------------------------------------------------------- aggregation

def test_reliability_table_bins_and_closed_top_edge():
    pairs = [(0.3, False), (0.85, True), (0.85, False), (1.0, True)]
    bins = reliability_table(pairs, edges=(0.0, 0.5, 0.9, 1.0))
    assert [(b.n, b.accuracy) for b in bins] == [(1, 0.0), (2, 0.5), (1, 1.0)]


def test_threshold_for_accuracy_picks_smallest_sufficient_edge():
    pairs = [(0.4, False)] * 5 + [(0.85, True)] * 8 + [(0.85, False)] * 2 + [(0.97, True)] * 10
    # >= 0.8 target: conf >= 0.5 gives 18/20 = 0.9 -> smallest edge 0.5
    assert threshold_for_accuracy(pairs, 0.8, candidates=(0.0, 0.5, 0.9), min_n=5) == 0.5
    # >= 0.99 target: even the top slice is 10/10 -> 0.9
    assert threshold_for_accuracy(pairs, 0.99, candidates=(0.0, 0.5, 0.9), min_n=5) == 0.9
    # unreachable with min_n above the top-slice size
    assert threshold_for_accuracy(pairs, 0.99, candidates=(0.0, 0.5, 0.9), min_n=15) is None


def _page(page_id, outcomes, quality=0.95):
    return PageResult(page_id=page_id, n_facts=len(outcomes),
                      n_omitted=sum(not o.in_document for o in outcomes),
                      quality_score=quality, mean_confidence=0.9, outcomes=outcomes)


def _o(kind="number", in_doc=True, located=True, conf=0.9, correct=True, garbled=False):
    return FactOutcome(kind=kind, value="v", in_document=in_doc, located=located,
                       confidence=conf if located else None,
                       garbled=garbled if located else None,
                       correct=correct if located else None)


def test_summarize_counts_omissions_by_kind_and_splits_garbled():
    results = [_page(0, [
        _o(kind="number", in_doc=True, correct=True, conf=0.98),
        _o(kind="number", in_doc=False, correct=False, conf=0.6, garbled=True),
        _o(kind="temporal", in_doc=False, located=False),
    ], quality=0.99)]
    s = summarize(results)
    assert s.n_pages == 1 and s.n_facts == 3
    assert s.omission_rate == round(2 / 3, 3)
    assert s.by_kind["number"].n_gold == 2 and s.by_kind["number"].n_omitted == 1
    assert s.by_kind["temporal"].omission_rate == 1.0
    assert s.n_located == 2 and s.n_unlocated == 1
    assert s.accuracy_garbled == 0.0 and s.accuracy_clean == 1.0
    assert s.mean_quality == 0.99  # high quality despite 2/3 omission — the blind spot
    assert "0.95" in s.thresholds


def test_format_report_smoke():
    s = summarize([_page(0, [_o()])])
    text = format_report(s)
    assert "OMISSION RATE" in text and "CALIBRATION" in text


# -------------------------------------------------------------------- dataset IO

def _write_fake_dataset(dir_: Path, ids=(2, 0, 1), missing_gold=()):
    import csv as _csv

    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "gold_annotation_html").mkdir(exist_ok=True)
    with (dir_ / "raw_input.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["id", "image", "matched_html"])
        w.writeheader()
        for i in ids:
            w.writerow({"id": i, "image": f"aW1n{i}", "matched_html": "<p>x</p>"})
    for i in ids:
        if i not in missing_gold:
            (dir_ / "gold_annotation_html" / f"gold_{i}.txt").write_text(
                f"<p><number>{i}</number></p>"
            )


def test_decode_image_strips_data_uri_prefix():
    import base64

    payload = base64.b64encode(b"\x89PNG\r\n").decode()
    assert decode_image(f"data:image/png;base64,{payload}") == b"\x89PNG\r\n"
    assert decode_image(payload) == b"\x89PNG\r\n"  # bare payload still works


def test_load_samples_is_deterministic_ascending_id_and_capped(tmp_path: Path):
    _write_fake_dataset(tmp_path, ids=(2, 0, 1))
    samples = load_samples(tmp_path, cap=2)
    assert [s.page_id for s in samples] == [0, 1]
    assert samples[0].gold_html == "<p><number>0</number></p>"


def test_load_samples_skips_rows_without_gold_file(tmp_path: Path):
    _write_fake_dataset(tmp_path, ids=(0, 1, 2), missing_gold=(1,))
    assert [s.page_id for s in load_samples(tmp_path, cap=10)] == [0, 2]
