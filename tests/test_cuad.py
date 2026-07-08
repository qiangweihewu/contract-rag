import pytest

from contract_rag.eval.cuad import CUAD_FIELD_MAP, resolve_columns, row_to_golden


def test_resolve_columns_is_case_insensitive_and_substring():
    columns = ["Filename", "PARTIES", "Effective Date", "Governing Law-Answer",
               "Notice Period To Terminate Renewal", "Renewal Term"]
    resolved = resolve_columns(columns, CUAD_FIELD_MAP)
    assert resolved["counterparty"] == "PARTIES"
    assert resolved["effective_date"] == "Effective Date"
    # substring fallback finds the "Governing Law-Answer" column
    assert resolved["governing_law"] == "Governing Law-Answer"
    assert resolved["termination_notice_days"] == "Notice Period To Terminate Renewal"
    assert resolved["auto_renewal"] == "Renewal Term"


def test_resolve_columns_raises_on_missing():
    with pytest.raises(KeyError):
        resolve_columns(["Filename"], CUAD_FIELD_MAP)


def test_row_to_golden_maps_values():
    row = {
        "Parties": "Acme Inc.; Globex LLC",
        "Effective Date": "January 1, 2026",
        "Governing Law": "New York",
    }
    columns = {"counterparty": "Parties", "effective_date": "Effective Date", "governing_law": "Governing Law"}
    g = row_to_golden(row, columns, doc_id="acme-msa", source_pdf="acme-msa.pdf")
    assert g.doc_id == "acme-msa"
    assert g.facts["counterparty"] == "Acme Inc.; Globex LLC"
    assert g.facts["governing_law"] == "New York"
    assert g.facts["effective_date"] == "January 1, 2026"
    assert g.source_pdf == "acme-msa.pdf"


def test_row_to_golden_coerces_nan_and_none_to_empty():
    row = {"Parties": float("nan"), "Effective Date": None, "Governing Law": "New York"}
    columns = {"counterparty": "Parties", "effective_date": "Effective Date", "governing_law": "Governing Law"}
    g = row_to_golden(row, columns, doc_id="d", source_pdf="d.pdf")
    assert g.facts["counterparty"] == ""
    assert g.facts["effective_date"] == ""
    assert g.facts["governing_law"] == "New York"


def test_build_golden_from_cuad_raises_without_master_csv(tmp_path):
    from contract_rag.eval.cuad import build_golden_from_cuad

    cuad_dir = tmp_path / "cuad"
    cuad_dir.mkdir()
    with pytest.raises(ValueError, match="master_clauses.csv"):
        build_golden_from_cuad(cuad_dir, tmp_path / "golden", tmp_path / "data")


def test_build_golden_from_cuad_raises_without_filename_column(tmp_path):
    import pandas as pd

    from contract_rag.eval.cuad import build_golden_from_cuad

    cuad_dir = tmp_path / "cuad"
    cuad_dir.mkdir()
    # Has the clause columns resolve_columns needs, but NO Filename/Document Name column.
    pd.DataFrame(
        [{"Parties": "Acme", "Effective Date": "2026-01-01", "Governing Law": "New York",
          "Notice Period To Terminate Renewal": "30 days", "Renewal Term": ""}]
    ).to_csv(cuad_dir / "master_clauses.csv", index=False)
    with pytest.raises(ValueError, match="Filename"):
        build_golden_from_cuad(cuad_dir, tmp_path / "golden", tmp_path / "data")


def _synthetic_cuad(cuad_dir, stem="Acme_Agreement"):
    """A minimal CUAD release: master_clauses.csv + a (dummy) PDF build_golden only copies."""
    import pandas as pd

    cuad_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "Filename": f"{stem}.pdf",
            "Parties": "Acme Inc.; Globex LLC",
            "Effective Date": "January 1, 2026",
            "Governing Law": "New York",
            "Notice Period To Terminate Renewal": "ninety (90) days",
            "Renewal Term": "automatically renew for successive one-year terms",
        }]
    ).to_csv(cuad_dir / "master_clauses.csv", index=False)
    (cuad_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4 fake")


def test_build_from_settings_writes_golden_and_copies_pdf(tmp_path):
    from contract_rag.config import Settings
    from contract_rag.eval.cuad import build_from_settings
    from contract_rag.eval.golden import load_golden_set

    _synthetic_cuad(tmp_path / "cuad")
    settings = Settings(
        cuad_dir=tmp_path / "cuad",
        golden_set_dir=tmp_path / "golden_set",
        data_dir=tmp_path / "data",
    )

    count = build_from_settings(settings)

    assert count == 1
    docs = load_golden_set(settings.golden_set_dir)
    assert len(docs) == 1
    # counterparty is normalized to its corporate entity set
    assert "Acme Inc" in docs[0].facts["counterparty"]
    assert "Globex LLC" in docs[0].facts["counterparty"]
    # the PDF the golden doc points at was copied into data_dir for the eval harnesses
    assert (settings.data_dir / docs[0].source_pdf).exists()


def test_build_from_settings_respects_n_limit(tmp_path):
    import pandas as pd

    from contract_rag.config import Settings
    from contract_rag.eval.cuad import build_from_settings

    cuad_dir = tmp_path / "cuad"
    cuad_dir.mkdir()
    rows = []
    for i in range(3):
        stem = f"Doc_{i}"
        rows.append({
            "Filename": f"{stem}.pdf", "Parties": f"Party {i}",
            "Effective Date": "", "Governing Law": "",
            "Notice Period To Terminate Renewal": "", "Renewal Term": "",
        })
        (cuad_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4 fake")
    pd.DataFrame(rows).to_csv(cuad_dir / "master_clauses.csv", index=False)
    settings = Settings(
        cuad_dir=cuad_dir, golden_set_dir=tmp_path / "g", data_dir=tmp_path / "d"
    )

    assert build_from_settings(settings, n=2) == 2


def test_format_build_report_mentions_count_and_dir(tmp_path):
    from contract_rag.config import Settings
    from contract_rag.eval.cuad import format_build_report

    settings = Settings(golden_set_dir=tmp_path / "golden_set", data_dir=tmp_path / "data")
    report = format_build_report(7, settings)

    assert "7" in report
    assert str(settings.golden_set_dir) in report
