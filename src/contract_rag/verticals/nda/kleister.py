"""Kleister-NDA real golden set for the NDA vertical (vertical #2).

Builds a `load_golden_set`-compatible golden set from the Kleister-NDA benchmark
(https://github.com/applicaai/kleister-nda — SEC EDGAR filings; the repo states no
explicit license, so the dataset is never committed: golden/data dirs are gitignored)
and evaluates the NDA rule extractor against it, credential-free.

Gold mapping (canonicalized to the extractor's answer space, the CUAD convention —
both sides go through the SAME canonicalizer at metric time):

- ``effective_date`` — Kleister is ISO ``YYYY-MM-DD``; the extractor emits prose
  spans ("May 20, 2014"). `canonical_date` maps both to ISO.
- ``jurisdiction`` -> ``governing_law`` via the shared `jurisdiction_in` vocabulary.
- ``term`` — Kleister is ``{n}_{units}``; `canonical_duration` maps both sides to
  ``"{n} {unit}s"`` (it extends the NDA vertical's `duration_in` with days, which
  Kleister term values can use).
- ``party`` — Kleister labels a FLAT party set, not disclosing/receiving roles, so
  the prediction is the UNION of extracted disclosing ∪ receiving entities scored as
  an entity-set field (Jaccard >= 0.5, like contract `counterparty`). This union is
  lossy on role assignment by design — Kleister has no role gold to score against.
  Gold party members without a corporate suffix (person names) are kept verbatim;
  the rule extractor can only emit suffixed corporate entities, so those members are
  honest recall misses.
- ``confidentiality_period`` / ``return_of_materials`` — no Kleister gold (zero-gold
  everywhere); `aggregate()` would exclude them, so `KleisterNDAFacts` simply omits
  them and scores only the four measurable fields.

Usage:
    KLEISTER_DIR=~/.cache/kleister-nda uv run python -m contract_rag.verticals.nda.kleister
    uv run python -m contract_rag.verticals.nda.kleister --eval
"""
from __future__ import annotations

import lzma
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from contract_rag.chunk.models import Chunk
from contract_rag.eval.golden import GoldenDoc
from contract_rag.ir import DocumentIR
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.legal_common import jurisdiction_in, party_entities
from contract_rag.verticals.nda.prompt import EXTRACTION_PROMPT
from contract_rag.verticals.nda.rules import NDARuleExtractor
from contract_rag.verticals.nda.vertical import NDAVertical

SPLITS = ("train", "dev-0")  # test-A gold is hidden — never used

# Kleister expected.tsv key -> our golden fact name
_KEY_MAP = {
    "effective_date": "effective_date",
    "jurisdiction": "governing_law",
    "term": "term",
    "party": "party",
}


# ---------------------------------------------------------------- gold parsing

def decode_value(v: str) -> str:
    """Kleister replaces spaces (and colons) with underscores; decoding restores
    spaces. Colons are unrecoverable — a known, negligible lossiness for NDAs."""
    return v.replace("_", " ").strip()


def parse_expected_line(line: str) -> dict[str, str]:
    """One expected.tsv line ('k=v k=v ...') -> golden facts. Multiple `party=`
    values become one '; '-joined set string; absent keys become ''."""
    parties: list[str] = []
    facts = {name: "" for name in _KEY_MAP.values()}
    for pair in line.strip().split():
        key, _, raw = pair.partition("=")
        if key == "party":
            parties.append(decode_value(raw))
        elif key in _KEY_MAP:
            facts[_KEY_MAP[key]] = decode_value(raw)
    facts["party"] = "; ".join(parties)
    return facts


def read_split(split_dir: Path) -> list[tuple[str, dict[str, str]]]:
    """(document filename, golden facts) rows for one split. in.tsv.xz column 0 is
    the filename; expected.tsv is line-aligned with it."""
    split_dir = Path(split_dir)
    in_path = split_dir / "in.tsv.xz"
    if not in_path.exists():
        raise ValueError(f"no in.tsv.xz in {split_dir}; point KLEISTER_DIR at a kleister-nda checkout")
    in_lines = lzma.decompress(in_path.read_bytes()).decode("utf-8").splitlines()
    exp_lines = (split_dir / "expected.tsv").read_text().splitlines()
    if len(in_lines) != len(exp_lines):
        raise ValueError(
            f"line count mismatch in {split_dir}: in.tsv.xz has {len(in_lines)}, "
            f"expected.tsv has {len(exp_lines)}"
        )
    return [
        (in_line.split("\t", 1)[0].strip(), parse_expected_line(exp_line))
        for in_line, exp_line in zip(in_lines, exp_lines)
    ]


# ---------------------------------------------------- shared canonicalizers
# Applied to BOTH gold and prediction (via KleisterNDAVertical.canonicalize_value),
# so the metric measures the answer, not the surface form.

_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")
_MONTH_NUM = {m.lower(): i for i, m in enumerate(_MONTHS, start=1)}
_MONTHS_ALT = "|".join(_MONTHS)
_ORD = r"(?:st|nd|rd|th)?"
_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
# "the 6th day of January, 2012" / "4th day of May 2005" legalese
_DAY_OF_RE = re.compile(
    rf"\b(\d{{1,2}}){_ORD}\s+day\s+of\s+({_MONTHS_ALT})[\s,]*(\d{{4}})", re.IGNORECASE
)
# "January 6, 2012" / "April 6th , 2005" (ordinal suffix + nbsp runs tolerated)
_PROSE_DATE_RE = re.compile(
    rf"\b({_MONTHS_ALT})\s+(\d{{1,2}}){_ORD}\s*,?\s*(\d{{4}})", re.IGNORECASE
)
# "6 January 2012" day-month-year
_DMY_RE = re.compile(
    rf"\b(\d{{1,2}}){_ORD}\s+({_MONTHS_ALT})\s*,?\s*(\d{{4}})", re.IGNORECASE
)
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_DURATION_RE = re.compile(
    rf"\(?(\d+)\)?\s*(year|month|day)s?"
    rf"|\b(?<!-)({'|'.join(_WORD_NUM)})\s+(year|month|day)s?",
    re.IGNORECASE,
)


def canonical_date(text: str) -> str:
    """Any recognized date form -> ISO 'YYYY-MM-DD' (Kleister's gold form); '' if none."""
    m = _ISO_RE.search(text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = _DAY_OF_RE.search(text)  # before DMY: "6th day of January" starts like a DMY
    if m:
        return f"{int(m.group(3)):04d}-{_MONTH_NUM[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    m = _PROSE_DATE_RE.search(text)
    if m:
        return f"{int(m.group(3)):04d}-{_MONTH_NUM[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    m = _DMY_RE.search(text)
    if m:
        return f"{int(m.group(3)):04d}-{_MONTH_NUM[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    m = _SLASH_DATE_RE.search(text)
    if m:
        year = int(m.group(3))
        if year < 100:  # EDGAR corpus is 1990s+; two-digit years pivot at 40
            year += 2000 if year < 40 else 1900
        return f"{year:04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return ""


def canonical_duration(text: str) -> str:
    """Any duration form -> '{n} {unit}s' ('three (3) years' / 'three years' /
    '3_years'-decoded -> '3 years'). Superset of nda.rules.duration_in: also days."""
    m = _DURATION_RE.search(text)
    if not m:
        return ""
    if m.group(1):
        return f"{m.group(1)} {m.group(2).lower()}s"
    return f"{_WORD_NUM[m.group(3).lower()]} {m.group(4).lower()}s"


# ------------------------------------------------------------- facts + vertical

class KleisterNDAFacts(BaseModel):
    """The Kleister-measurable view of NDAFacts: `party` is the flat union set
    Kleister actually labels; the two zero-gold NDA fields are omitted."""

    party: ExtractedClause = Field(default_factory=ExtractedClause)
    effective_date: ExtractedClause = Field(default_factory=ExtractedClause)
    term: ExtractedClause = Field(default_factory=ExtractedClause)
    governing_law: ExtractedClause = Field(default_factory=ExtractedClause)

    FIELD_NAMES: ClassVar[tuple[str, ...]] = ("party", "effective_date", "term", "governing_law")
    SET_FIELDS: ClassVar[tuple[str, ...]] = ("party",)
    JUDGMENT_FIELDS: ClassVar[tuple[str, ...]] = ()


_SCALAR_CANON = {
    "effective_date": canonical_date,
    "term": canonical_duration,
    "governing_law": lambda v: jurisdiction_in(v) or "",
}


class KleisterNDAVertical:
    """Vertical view for scoring the NDA extractor on Kleister gold. Delegates the
    chunk-level behavior to the real NDA vertical; only the facts schema, the party
    entity rule, and the scalar canonicalizers differ."""

    name = "nda_kleister"
    facts_model = KleisterNDAFacts
    field_names = KleisterNDAFacts.FIELD_NAMES
    set_fields = KleisterNDAFacts.SET_FIELDS
    judgment_fields = KleisterNDAFacts.JUDGMENT_FIELDS
    extraction_prompt = EXTRACTION_PROMPT

    def __init__(self) -> None:
        self._nda = NDAVertical()
        self.rule_extractor = KleisterNDAExtractor()

    def classify_clause(self, chunk: Chunk) -> str:
        return self._nda.classify_clause(chunk)

    def permission_tags(self, chunk: Chunk) -> list[str]:
        return self._nda.permission_tags(chunk)

    def normalize_gold(self, raw: Mapping[str, str]) -> dict[str, str]:
        return dict(raw)  # build_golden_from_kleister already decodes to answer space

    def canonicalize_value(self, name: str, value: str) -> str:
        return _SCALAR_CANON.get(name, lambda v: v)(value)

    def entities(self, value: str) -> list[str]:
        """Set members of a '; '-joined party string. Corporate names canonicalize
        via party_entities; segments without a corporate suffix (person names in
        Kleister gold) are kept verbatim so they stay scoreable set members."""
        out: list[str] = []
        seen: set[str] = set()
        for seg in value.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            for e in party_entities(seg) or [seg]:
                if e.lower() not in seen:
                    seen.add(e.lower())
                    out.append(e)
        return out

    def empty_facts(self) -> KleisterNDAFacts:
        return KleisterNDAFacts()


class KleisterNDAExtractor:
    """Adapter: run the real NDA rule extractor, then project NDAFacts into the
    Kleister view — `party` = union of disclosing ∪ receiving entities. The union
    cites ONE block, so if the two roles were found in different blocks it keeps
    only the entities of the better-populated cited block (ties -> disclosing) —
    trading a little recall for source-attribution that holds by construction (in
    practice both roles come from the same preamble block)."""

    def __init__(self) -> None:
        self._inner = NDARuleExtractor()

    def extract(self, ir: DocumentIR) -> KleisterNDAFacts:
        nda = self._inner.extract(ir)
        cited = [c for c in (nda.disclosing_party, nda.receiving_party) if c.value]
        by_block: dict[str, list[str]] = {}
        for clause in cited:
            ents = by_block.setdefault(clause.source_block_id or "", [])
            for e in party_entities(clause.value):
                if e.lower() not in {x.lower() for x in ents}:
                    ents.append(e)
        party = ExtractedClause()
        if by_block:
            block_id = max(by_block, key=lambda b: len(by_block[b]))
            if by_block[block_id]:
                party = ExtractedClause(
                    value="; ".join(by_block[block_id]),
                    source_block_id=block_id,
                    confidence=min(c.confidence for c in cited),
                )
        return KleisterNDAFacts(
            party=party,
            effective_date=nda.effective_date,
            term=nda.term,
            governing_law=nda.governing_law,
        )


# ------------------------------------------------------------------ builder

def build_golden_from_kleister(
    kleister_dir: Path, out_dir: Path, data_dir: Path, n: int = 40,
    splits: tuple[str, ...] = SPLITS,
) -> int:
    """Write one GoldenDoc JSON per document (capped at `n`) and copy its PDF.
    Selection is deterministic: all train+dev-0 rows, sorted by document filename,
    skipping rows whose PDF is missing from documents/."""
    kleister_dir, out_dir, data_dir = Path(kleister_dir), Path(out_dir), Path(data_dir)
    rows: list[tuple[str, dict[str, str]]] = []
    for split in splits:
        rows.extend(read_split(kleister_dir / split))
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for fname, facts in sorted(rows, key=lambda r: r[0]):
        if written >= n:
            break
        src_pdf = kleister_dir / "documents" / fname
        if not src_pdf.exists():
            continue
        doc_id = Path(fname).stem
        dest_pdf = data_dir / fname
        if not dest_pdf.exists():
            shutil.copyfile(src_pdf, dest_pdf)
        golden = GoldenDoc(doc_id=doc_id, source_pdf=fname, facts=facts)
        (out_dir / f"{doc_id}.json").write_text(golden.model_dump_json(indent=2))
        written += 1
    return written


# --------------------------------------------------------------------- eval

def _golden_dir() -> Path:
    return Path(os.environ.get("KLEISTER_GOLDEN_DIR", "golden_set_nda_kleister"))


def _data_dir() -> Path:
    return Path(os.environ.get("KLEISTER_DATA_DIR", "data_nda_kleister"))


def evaluate_kleister(golden_dir: Path | None = None, data_dir: Path | None = None) -> dict:
    """NDA rule extractor over the built Kleister set: docling parse (IR-cached),
    generic run_baseline + metrics. Credential-free."""
    from contract_rag.baseline import run_baseline
    from contract_rag.config import Settings
    from contract_rag.eval.ir_cache import ir_cache

    def _parse(path: Path) -> DocumentIR:  # lazy: docling is a heavy dep
        from contract_rag.parse.docling_parser import parse_with_docling
        return parse_with_docling(path)

    vertical = KleisterNDAVertical()
    settings = Settings(vertical="nda",
                        golden_set_dir=golden_dir or _golden_dir(),
                        data_dir=data_dir or _data_dir())
    parse_fn = ir_cache(Path(os.environ.get("IR_CACHE_DIR", ".ir_cache/kleister_nda")), _parse)
    return run_baseline(settings, vertical.rule_extractor, parse_fn, vertical=vertical)


def format_kleister_report(agg: dict) -> str:
    lines = [
        "=== NDA vertical on Kleister-NDA (train+dev-0, rule backend) ===",
        f"docs:            {agg['n_docs']}",
        f"field_f1:        {agg['field_f1']:.3f}",
        f"precision:       {agg['precision']:.3f}",
        f"recall:          {agg['recall']:.3f}",
        f"source_accuracy: {agg['source_accuracy']:.3f}",
        f"{'field':<16} {'gold':>4} {'per-doc':>8} {'on-labeled':>10}",
    ]
    for name in agg["per_field"]:
        on_lab = agg["per_field_on_labeled"][name]
        lines.append(
            f"{name:<16} {agg['support'][name]:>4} {agg['per_field'][name]:>8.3f} "
            f"{('%.3f' % on_lab) if on_lab is not None else '—':>10}"
        )
    return "\n".join(lines)


def main() -> None:
    import sys

    from contract_rag.config import get_settings

    if "--eval" in sys.argv:
        print(format_kleister_report(evaluate_kleister()))
        return
    settings = get_settings()
    n = int(os.environ.get("KLEISTER_SET_SIZE", os.environ.get("GOLDEN_SET_SIZE", "40")))
    count = build_golden_from_kleister(settings.kleister_dir, _golden_dir(), _data_dir(), n=n)
    print("\n".join([
        "=== Golden-set build (from Kleister-NDA train+dev-0) ===",
        f"documents written: {count}",
        f"golden dir:        {_golden_dir()}",
        f"data dir (pdfs):   {_data_dir()}",
    ]))


if __name__ == "__main__":
    main()
