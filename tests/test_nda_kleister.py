"""Unit tests for the Kleister-NDA golden-set builder + eval mapping.

All synthetic: hand-built in.tsv.xz / expected.tsv fixtures and dummy PDF bytes —
no network, no real dataset, no docling.
"""
from __future__ import annotations

import lzma
from pathlib import Path

import pytest

from contract_rag.eval.golden import load_golden_set
from contract_rag.eval.metrics import field_scores
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.nda.kleister import (
    KleisterNDAExtractor,
    KleisterNDAVertical,
    build_golden_from_kleister,
    canonical_date,
    canonical_duration,
    decode_value,
    parse_expected_line,
    read_split,
)


# ---------------------------------------------------------------- gold parsing

def test_decode_value_replaces_underscores_with_spaces():
    assert decode_value("New_York") == "New York"
    assert decode_value("Liquidmetal_Technology_Inc.") == "Liquidmetal Technology Inc."


def test_parse_expected_line_maps_fields_and_joins_parties():
    line = ("effective_date=2014-05-20 jurisdiction=New_York "
            "party=Liquidmetal_Technology_Inc. party=Visser_Precision_Cast_LLC term=3_years")
    facts = parse_expected_line(line)
    assert facts["effective_date"] == "2014-05-20"
    assert facts["governing_law"] == "New York"
    assert facts["party"] == "Liquidmetal Technology Inc.; Visser Precision Cast LLC"
    assert facts["term"] == "3 years"


def test_parse_expected_line_missing_keys_are_empty():
    facts = parse_expected_line("jurisdiction=Delaware party=Oglethorpe_Power_Corporation")
    assert facts["governing_law"] == "Delaware"
    assert facts["effective_date"] == ""
    assert facts["term"] == ""


def _write_split(split_dir: Path, rows: list[tuple[str, str]]) -> None:
    """rows: (filename, expected-line). in.tsv has extra text columns like the real set."""
    split_dir.mkdir(parents=True, exist_ok=True)
    in_lines = "".join(f"{fname}\tkeys\tsome extracted text\n" for fname, _ in rows)
    (split_dir / "in.tsv.xz").write_bytes(lzma.compress(in_lines.encode()))
    (split_dir / "expected.tsv").write_text("".join(exp + "\n" for _, exp in rows))


def test_read_split_aligns_in_tsv_with_expected(tmp_path):
    _write_split(tmp_path / "train", [
        ("aaa.pdf", "jurisdiction=Delaware party=Acme_Inc."),
        ("bbb.pdf", "effective_date=2020-01-02 party=Globex_LLC term=2_years"),
    ])
    rows = read_split(tmp_path / "train")
    assert [fname for fname, _ in rows] == ["aaa.pdf", "bbb.pdf"]
    assert rows[0][1]["governing_law"] == "Delaware"
    assert rows[1][1]["effective_date"] == "2020-01-02"
    assert rows[1][1]["term"] == "2 years"


def test_read_split_raises_on_misaligned_files(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    (split / "in.tsv.xz").write_bytes(lzma.compress(b"aaa.pdf\tx\nbbb.pdf\tx\n"))
    (split / "expected.tsv").write_text("jurisdiction=Delaware\n")
    with pytest.raises(ValueError, match="line count"):
        read_split(split)


# ---------------------------------------------------------- shared canonicalizers

def test_canonical_date_iso_passthrough_and_prose_forms():
    assert canonical_date("2014-05-20") == "2014-05-20"
    assert canonical_date("May 20, 2014") == "2014-05-20"
    assert canonical_date("dated as of July 1, 2006,") == "2006-07-01"
    assert canonical_date("5/20/2014") == "2014-05-20"
    assert canonical_date("5/20/14") == "2014-05-20"
    assert canonical_date("no date here") == ""


def test_canonical_date_ordinal_and_legalese_forms():
    # day-of legalese, with and without comma / nbsp
    assert canonical_date("6th day of January, 2012") == "2012-01-06"
    assert canonical_date("the 4th day of May\xa02005") == "2005-05-04"
    assert canonical_date("30th day of April, 2009") == "2009-04-30"
    # month + ordinal day, nbsp-padded blanks
    assert canonical_date("April\xa06th\xa0\xa0\xa0, 2005") == "2005-04-06"
    assert canonical_date("January 6th, 2012") == "2012-01-06"
    # day-month-year
    assert canonical_date("6 January 2012") == "2012-01-06"
    # tight comma (letterhead form)
    assert canonical_date("December 11,2014") == "2014-12-11"


def test_canonical_duration_handles_prose_and_kleister_forms():
    assert canonical_duration("3 years") == "3 years"
    assert canonical_duration("three (3) years") == "3 years"
    assert canonical_duration("a term of two (2) years from the date") == "2 years"
    assert canonical_duration("18 months") == "18 months"
    assert canonical_duration("thirty (30) days") == "30 days"
    assert canonical_duration("1 year") == "1 years"  # canonical form is always plural
    assert canonical_duration("no duration") == ""


# ------------------------------------------------------------- vertical + metrics

def test_vertical_entities_splits_segments_and_keeps_person_names():
    v = KleisterNDAVertical()
    ents = v.entities("Nitromed Inc.; Kenneth M. Bate")
    # corporate name canonicalized by party_entities; person name kept verbatim
    assert ents == ["Nitromed Inc", "Kenneth M. Bate"]


def test_field_scores_match_across_answer_spaces():
    v = KleisterNDAVertical()
    from contract_rag.eval.golden import GoldenDoc
    gold = GoldenDoc(doc_id="d", source_pdf="d.pdf", facts={
        "party": "Liquidmetal Technology Inc.; Visser Precision Cast LLC",
        "effective_date": "2014-05-20",
        "term": "3 years",
        "governing_law": "New York",
    })
    pred = v.facts_model(
        party=ExtractedClause(value="Liquidmetal Technology Inc; Visser Precision Cast LLC",
                              source_block_id="#/b/0", confidence=0.7),
        effective_date=ExtractedClause(value="May 20, 2014", source_block_id="#/b/0", confidence=0.6),
        term=ExtractedClause(value="three (3) years", source_block_id="#/b/1", confidence=0.6),
        governing_law=ExtractedClause(value="New York", source_block_id="#/b/2", confidence=0.7),
    )
    scores = field_scores(pred, gold, v)
    assert scores == {"party": True, "effective_date": True, "term": True, "governing_law": True}


def test_field_scores_party_jaccard_threshold():
    v = KleisterNDAVertical()
    from contract_rag.eval.golden import GoldenDoc
    gold = GoldenDoc(doc_id="d", source_pdf="d.pdf",
                     facts={"party": "Acme Inc.; Globex LLC", "effective_date": "",
                            "term": "", "governing_law": ""})
    # one of two entities -> Jaccard 1/2 = 0.5 >= threshold
    pred_half = v.facts_model(party=ExtractedClause(value="Acme Inc", source_block_id="b"))
    assert field_scores(pred_half, gold, v)["party"] is True
    # a disjoint entity -> Jaccard 1/3 < threshold
    pred_bad = v.facts_model(
        party=ExtractedClause(value="Acme Inc; Wrongco Ltd; Otherco Corp", source_block_id="b"))
    assert field_scores(pred_bad, gold, v)["party"] is False


def test_canonical_duration_word_numbers():
    assert canonical_duration("three years from the date of this letter") == "3 years"
    assert canonical_duration("one year") == "1 years"
    assert canonical_duration("thirty-one years") == ""  # no compound half-match


# ------------------------------------------------------------- extractor adapter

def _ir(text: str) -> DocumentIR:
    return DocumentIR(
        doc_id="d", source_uri="file:///d", file_hash="x", mime_type="application/pdf",
        blocks=[DocBlock(block_id="#/b/0", type=BlockType.PARAGRAPH, text=text,
                         confidence=1.0, source_engine="test")],
        metadata={},
    )


def test_extractor_party_is_union_of_disclosing_and_receiving():
    ir = _ir(
        "This Agreement is made by and between Acme Robotics Inc., a Delaware corporation "
        '(the "Disclosing Party"), and Globex Industrial LLC (the "Receiving Party"), '
        "effective as of May 20, 2014."
    )
    pred = KleisterNDAExtractor().extract(ir)
    ents = KleisterNDAVertical().entities(pred.party.value)
    assert set(ents) == {"Acme Robotics Inc", "Globex Industrial LLC"}
    assert pred.party.source_block_id == "#/b/0"
    assert pred.effective_date.value  # passthrough of the NDA finder


def test_extractor_party_empty_when_no_roles_found():
    pred = KleisterNDAExtractor().extract(_ir("Nothing relevant here."))
    assert pred.party.value == ""
    assert pred.party.source_block_id is None


# ------------------------------------------------------------------ builder

def _fixture_kleister(root: Path) -> Path:
    kdir = root / "kleister-nda"
    docs = kdir / "documents"
    docs.mkdir(parents=True)
    _write_split(kdir / "train", [
        ("ccc.pdf", "jurisdiction=Delaware party=Acme_Inc."),
        ("aaa.pdf", "effective_date=2020-01-02 party=Globex_LLC term=2_years"),
        ("missing.pdf", "jurisdiction=Texas party=Ghost_Corp."),  # no PDF on disk
    ])
    _write_split(kdir / "dev-0", [
        ("bbb.pdf", "jurisdiction=New_York party=Initech_Inc. party=John_Q._Public"),
    ])
    for name in ("aaa.pdf", "bbb.pdf", "ccc.pdf"):
        (docs / name).write_bytes(b"%PDF-1.4 fake")
    return kdir


def test_build_golden_writes_loadable_docs_and_copies_pdfs(tmp_path):
    kdir = _fixture_kleister(tmp_path)
    out, data = tmp_path / "golden", tmp_path / "data"
    n = build_golden_from_kleister(kdir, out, data, n=40)
    assert n == 3  # missing.pdf skipped
    golden = load_golden_set(out)
    assert [g.doc_id for g in golden] == ["aaa", "bbb", "ccc"]  # sorted, deterministic
    by_id = {g.doc_id: g for g in golden}
    assert by_id["aaa"].facts["term"] == "2 years"
    assert by_id["bbb"].facts["party"] == "Initech Inc.; John Q. Public"
    assert by_id["ccc"].facts["governing_law"] == "Delaware"
    for g in golden:
        assert (data / g.source_pdf).exists()


def test_build_golden_cap_is_deterministic(tmp_path):
    kdir = _fixture_kleister(tmp_path)
    out1, out2 = tmp_path / "g1", tmp_path / "g2"
    build_golden_from_kleister(kdir, out1, tmp_path / "d1", n=2)
    build_golden_from_kleister(kdir, out2, tmp_path / "d2", n=2)
    ids1 = [g.doc_id for g in load_golden_set(out1)]
    ids2 = [g.doc_id for g in load_golden_set(out2)]
    assert ids1 == ids2 == ["aaa", "bbb"]  # sorted by filename, capped


def test_build_golden_raises_without_splits(tmp_path):
    with pytest.raises(ValueError, match="in.tsv.xz"):
        build_golden_from_kleister(tmp_path, tmp_path / "g", tmp_path / "d")

def test_extractor_party_union_from_preamble_without_role_labels():
    # real SEC preamble: no "Disclosing/Receiving Party" labels at all
    ir = _ir(
        "This Agreement is entered into as of January 1, 2008, by and between "
        "Verso Paper Holdings LLC, a Delaware limited liability company, and "
        "Acme Consulting Corp., a New York corporation."
    )
    pred = KleisterNDAExtractor().extract(ir)
    ents = KleisterNDAVertical().entities(pred.party.value)
    assert set(ents) == {"Verso Paper Holdings LLC", "Acme Consulting Corp"}
    assert pred.party.source_block_id == "#/b/0"


def test_extractor_party_union_cites_one_block_and_keeps_attribution():
    # roles found in DIFFERENT blocks: the union must only carry entities that are
    # literal spans of the single block it cites, so source-attribution holds.
    from contract_rag.eval.metrics import source_attribution_ok

    ir = DocumentIR(
        doc_id="d", source_uri="file:///d", file_hash="x", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="#/b/0", type=BlockType.PARAGRAPH,
                     text='Acme Robotics Inc. shall be the "Disclosing Party" between the parties.',
                     confidence=1.0, source_engine="test"),
            DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH,
                     text='Globex Industrial LLC shall be the "Receiving Party" between the parties.',
                     confidence=1.0, source_engine="test"),
        ],
        metadata={},
    )
    pred = KleisterNDAExtractor().extract(ir)
    assert pred.party.value  # something was extracted
    ok = source_attribution_ok(pred, ir, KleisterNDAVertical())
    assert ok["party"] is True


