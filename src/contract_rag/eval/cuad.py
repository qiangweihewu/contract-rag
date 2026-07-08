from __future__ import annotations

import os
import shutil
from pathlib import Path

from contract_rag.config import Settings, get_settings
from contract_rag.eval.golden import GoldenDoc
from contract_rag.verticals.contract.gold import normalize_facts

# total_value has no CUAD column (CUAD doesn't label contract monetary value), so it is
# extracted but unscored on this dataset.
CUAD_FIELD_MAP: dict[str, str] = {
    "counterparty": "Parties",
    "effective_date": "Effective Date",
    "governing_law": "Governing Law",
    "termination_notice_days": "Notice Period To Terminate Renewal",
    "auto_renewal": "Renewal Term",
}


def resolve_columns(columns: list[str], wanted: dict[str, str]) -> dict[str, str]:
    lower = {c.lower(): c for c in columns}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for field, target in wanted.items():
        if target.lower() in lower:
            resolved[field] = lower[target.lower()]
            continue
        hit = next((c for c in columns if target.lower() in c.lower()), None)
        if hit is None:
            missing.append(f"{field} (looking for column ~ '{target}')")
        else:
            resolved[field] = hit
    if missing:
        raise KeyError("unresolved CUAD columns: " + "; ".join(missing))
    return resolved


def _coerce(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and val != val:  # NaN is the only value not equal to itself
        return ""
    return str(val).strip()


def row_to_golden(row: dict, columns: dict[str, str], doc_id: str, source_pdf: str) -> GoldenDoc:
    facts = {field: _coerce(row.get(col)) for field, col in columns.items()}
    return GoldenDoc(doc_id=doc_id, source_pdf=source_pdf, facts=facts)


def find_pdf(cuad_dir: Path, filename_stem: str) -> Path | None:
    matches = list(Path(cuad_dir).rglob(f"{filename_stem}.pdf"))
    return matches[0] if matches else None


def build_golden_from_cuad(cuad_dir: Path, out_dir: Path, data_dir: Path, n: int = 40) -> int:
    import pandas as pd

    cuad_dir = Path(cuad_dir)
    out_dir = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_matches = list(cuad_dir.rglob("master_clauses.csv"))
    if not csv_matches:
        raise ValueError(
            f"no master_clauses.csv found under {cuad_dir}; "
            "point CUAD_DIR at the extracted CUAD release"
        )
    csv_path = csv_matches[0]
    df = pd.read_csv(csv_path)
    columns = resolve_columns(list(df.columns), CUAD_FIELD_MAP)
    fname_col = next(
        (c for c in df.columns if c.lower() in {"filename", "document name"}), None
    )
    if fname_col is None:
        raise ValueError(
            f"no Filename/Document Name column in {csv_path}; columns were: {list(df.columns)}"
        )

    written = 0
    for _, row in df.iterrows():
        if written >= n:
            break
        stem = Path(str(row[fname_col])).stem
        pdf = find_pdf(cuad_dir, stem)
        if pdf is None:
            continue  # skip contracts whose PDF we cannot locate
        doc_id = stem.replace(" ", "_")
        dest_pdf = data_dir / f"{doc_id}.pdf"
        if not dest_pdf.exists():
            shutil.copyfile(pdf, dest_pdf)
        golden = row_to_golden(row.to_dict(), columns, doc_id=doc_id, source_pdf=dest_pdf.name)
        golden = golden.model_copy(update={"facts": normalize_facts(golden.facts)})
        (out_dir / f"{doc_id}.json").write_text(golden.model_dump_json(indent=2))
        written += 1
    return written


def build_from_settings(settings: Settings, n: int = 40) -> int:
    return build_golden_from_cuad(
        settings.cuad_dir, settings.golden_set_dir, settings.data_dir, n=n
    )


def format_build_report(count: int, settings: Settings) -> str:
    return "\n".join([
        "=== Golden-set build (from CUAD) ===",
        f"contracts written: {count}",
        f"golden_set_dir:    {settings.golden_set_dir}",
        f"data_dir (pdfs):   {settings.data_dir}",
    ])


def main() -> None:
    settings = get_settings()
    n = int(os.environ.get("GOLDEN_SET_SIZE", "40"))
    count = build_from_settings(settings, n=n)
    print(format_build_report(count, settings))


if __name__ == "__main__":
    main()
