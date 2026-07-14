from __future__ import annotations

from pathlib import Path

from contract_rag.eval.crosscheck import evaluate_crosscheck, format_report
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _ir(texts, engine="paddleocr"):
    return DocumentIR(
        doc_id="d", source_uri="file:///d.pdf", file_hash="h",
        mime_type="application/pdf",
        blocks=[DocBlock(block_id=f"#/b{i}", type=BlockType.PARAGRAPH, text=t,
                         confidence=1.0, source_engine=engine)
                for i, t in enumerate(texts)],
    )


def _write(dir_: Path, stem: str, ir: DocumentIR) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{stem}.ir.json").write_text(ir.model_dump_json())


class _Sample:  # duck-types the fincritical Sample: only the gold field is read
    def __init__(self, gold_html: str, page_id: int | None = None):
        self.gold_html = gold_html
        self.page_id = page_id


def test_evaluate_crosscheck_recall_false_alarm_and_skip(tmp_path):
    p_dir, v_dir = tmp_path / "p", tmp_path / "v"
    # page 0: paddle dropped 12000 (gold number), dots read it -> caught
    _write(p_dir, "fincritical_0", _ir(["Revenue was reported in fiscal 2024"]))
    _write(v_dir, "fincritical_0", _ir(["Revenue was 12,000 in fiscal 2024"], "dots"))
    # page 1: clean page, engines agree -> no flag, counts toward false-alarm denom
    _write(p_dir, "fincritical_1", _ir(["Total 500 due 2023"]))
    _write(v_dir, "fincritical_1", _ir(["Total $500 due 2023"], "dots"))
    # page 2: verifier IR missing -> skipped and counted
    _write(p_dir, "fincritical_2", _ir(["whatever 1 2 3"]))
    samples = [
        _Sample('x <number>12,000</number> in fiscal <number>2024</number> x', page_id=0),
        _Sample('Total <number>500</number> due <number>2023</number>', page_id=1),
        _Sample('unused <number>7</number>', page_id=2),
    ]
    rows, s = evaluate_crosscheck(samples, p_dir, v_dir)
    assert s.n_pages == 2 and s.n_skipped_missing_ir == 1
    assert s.n_omitted_facts == 1 and s.caught_facts == 1
    assert s.flag_recall == 1.0 and s.digit_fact_recall == 1.0
    assert s.n_clean_pages == 1 and s.false_alarms == 0 and s.false_alarm_rate == 0.0
    assert s.passed_bar is True
    out = format_report(rows, s)
    assert "flag-recall" in out and "false-alarm" in out and "PASS" in out
