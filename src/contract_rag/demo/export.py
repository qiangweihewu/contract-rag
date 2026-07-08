"""CLM-aligned facts export — machine-readable CSV/JSON of extracted facts.

The machine-readable counterpart to the HTML data-quality report: one row per
extracted field, carrying the value, its source block, confidence, risk tier and
the `verify()` outcome — so a CLM migration (Salesforce Contracts, Ironclad) can
import cleaned, sourced, confidence-scored facts instead of re-keying them.

**Field-name mapping is best-effort naming alignment, not an official schema.**
The `salesforce` map uses Contract-object-style names (`StartDate`,
`ContractTerm`, `OwnerExpirationNotice` are standard Contract fields;
`GoverningLaw__c` / `ContractValue__c` / `AutoRenewal__c` follow the
custom-field naming convention); `ironclad` uses camelCase property names in the
style of Ironclad's data model (`counterpartyName`, `agreementDate`, ...);
`generic` is the identity mapping. Any field absent from a map keeps our field
name — the export never drops a field just because the target has no analogue.

Pure logic: rows are plain dicts, serializers are stdlib `csv`/`json`, and there
is no I/O here — callers (report/batch CLIs) write the files. Works for any
registered vertical: fields are iterated via `vertical.field_names` (the same
seam metrics/verify use), risk via the defensive `field_risk_map` (verticals
without a `field_risk` seam default to "medium"), and `verified` reuses
`extract/verify.py` semantics exactly (a row is verified iff verify() passed
it — attributed to its cited block and above the confidence floor).
"""
from __future__ import annotations

import csv
import io
import json

from contract_rag.eval.metrics import field_risk_map
from contract_rag.extract.verify import verify
from contract_rag.ir import DocumentIR

COLUMNS = (
    "doc_id", "field", "clm_field", "value",
    "source_block_id", "confidence", "risk_tier", "verified",
)

# Best-effort naming alignment with common CLM import templates (see module
# docstring). Unmapped fields keep our name.
CLM_FIELD_MAPS: dict[str, dict[str, str]] = {
    "generic": {},
    "salesforce": {
        "counterparty": "AccountName",              # Contract.AccountId lookup, by name
        "effective_date": "StartDate",              # standard Contract field
        "term": "ContractTerm",                     # standard Contract field (months)
        "termination_notice_days": "OwnerExpirationNotice",  # standard notice-days field
        "governing_law": "GoverningLaw__c",         # custom-field convention
        "total_value": "ContractValue__c",
        "auto_renewal": "AutoRenewal__c",
        "confidentiality_period": "ConfidentialityPeriod__c",
        "return_of_materials": "ReturnOfMaterials__c",
    },
    "ironclad": {
        "counterparty": "counterpartyName",
        "effective_date": "agreementDate",
        "governing_law": "governingLaw",
        "total_value": "contractValue",
        "termination_notice_days": "terminationNoticeDays",
        "auto_renewal": "autoRenews",
        "term": "termLength",
        "confidentiality_period": "confidentialityPeriod",
        "return_of_materials": "returnOfMaterials",
        "disclosing_party": "disclosingParty",
        "receiving_party": "receivingParty",
    },
}


def clm_field_map(clm: str) -> dict[str, str]:
    try:
        return CLM_FIELD_MAPS[clm]
    except KeyError:
        raise ValueError(
            f"unknown CLM target {clm!r}; available: {sorted(CLM_FIELD_MAPS)}"
        ) from None


def facts_rows(facts, ir: DocumentIR, doc_id: str, *,
               vertical=None, clm: str = "generic") -> list[dict]:
    """One row per extracted field of any vertical's facts model.

    `verified` reuses verify() semantics against the same IR the facts cite;
    `risk_tier` resolves the optional `field_risk` vertical seam defensively
    (verticals without it are all "medium")."""
    from contract_rag.verticals.registry import default_vertical

    v = vertical or default_vertical()
    mapping = clm_field_map(clm)
    risk = field_risk_map(v)
    checks = verify(facts, ir, vertical=v).checks
    return [
        {
            "doc_id": doc_id,
            "field": name,
            "clm_field": mapping.get(name, name),
            "value": getattr(facts, name).value,
            "source_block_id": getattr(facts, name).source_block_id or "",
            "confidence": getattr(facts, name).confidence,
            "risk_tier": risk[name],
            "verified": checks[name].passed,
        }
        for name in v.field_names
    ]


def rows_from_report(data, clm: str = "generic") -> list[dict]:
    """Rows from an already-built ReportData (duck-typed; no import of report).

    The report's FieldRow already carries the verify() outcome and risk tier for
    the *cleaned* facts, so exporting from it guarantees the file matches what
    the HTML report shows — used by the report/batch CLIs."""
    mapping = clm_field_map(clm)
    return [
        {
            "doc_id": data.doc_id,
            "field": f.field,
            "clm_field": mapping.get(f.field, f.field),
            "value": f.cleaned_value,
            "source_block_id": f.source_block_id or "",
            "confidence": f.confidence,
            "risk_tier": f.risk,
            "verified": f.verified,
        }
        for f in data.fields
    ]


_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value):
    """Prefix a leading `'` on string cells that start with a formula-triggering
    character, so Excel/Sheets never interprets an extracted value (e.g. a clause
    beginning with `-` or `=`) as a formula. Non-strings pass through untouched."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: _csv_safe(v) for k, v in row.items()} for row in rows)
    return buf.getvalue()


def to_json(rows: list[dict], stp: dict | None = None) -> str:
    """`stp` (an STP rollup dict, see `demo.report.stp_summary`) is additive and
    optional: omitted (the default), the payload is the plain rows array,
    byte-identical to before. Passed, the payload becomes an object carrying
    both `facts` (the rows) and `stp`, so existing array-shaped consumers of
    the un-flagged call are never affected."""
    payload: object = rows if stp is None else {"facts": rows, "stp": stp}
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def serialize(rows: list[dict], fmt: str, stp: dict | None = None) -> str:
    """`stp` is forwarded to `to_json` only — CSV's row shape is the downstream
    CLM import contract and stays exactly as-is regardless of `stp`."""
    if fmt == "csv":
        return to_csv(rows)
    if fmt == "json":
        return to_json(rows, stp)
    raise ValueError(f"unknown export format {fmt!r}; use 'csv' or 'json'")
