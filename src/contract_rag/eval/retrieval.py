"""Context-Recall retrieval eval: for each labeled field, does the retriever surface a
chunk that actually contains the gold answer? Compares BM25 vs dense vs hybrid — the
spec's exit criterion ("hybrid beats vector-only on Context Recall")."""
from __future__ import annotations

from typing import Callable

from contract_rag.chunk.chunker import chunk_ir
from contract_rag.chunk.models import Chunk
from contract_rag.enrich.definitions import extract_definitions, inject_definitions, term_used
from contract_rag.enrich.enricher import enrich_chunks
from contract_rag.eval.golden import GoldenDoc
from contract_rag.extract.rules import jurisdiction_in, party_entities
from contract_rag.extract.schema import ContractFacts
from contract_rag.index.bm25 import BM25Index
from contract_rag.index.dense import DenseIndex
from contract_rag.index.embed import Embedder
from contract_rag.index.hybrid import HybridIndex
from contract_rag.ir import DocumentIR
from contract_rag.text import normalize

_SET_FIELDS = set(ContractFacts.SET_FIELDS)

FIELD_QUERIES: dict[str, str] = {
    "counterparty": "who are the parties to this agreement",
    "effective_date": "what is the effective date of this agreement",
    "governing_law": "which state law governs this agreement",
    "total_value": "what is the total contract value or fees payable",
    "termination_notice_days": "how many days notice are required to terminate",
    "auto_renewal": "does this agreement renew automatically",
}


def supports(chunk: Chunk, field: str, gold: str) -> bool:
    """Does this chunk actually contain the gold answer for `field`?"""
    text = normalize(chunk.text)
    if field in _SET_FIELDS:
        ents = [normalize(e) for e in party_entities(gold)]
        return bool(ents) and any(e in text for e in ents)
    if field == "governing_law":
        return normalize(jurisdiction_in(gold) or gold) in text
    return normalize(gold) in text


def evaluate_retrieval(
    golden: list[GoldenDoc],
    ir_for: Callable[[GoldenDoc], DocumentIR],
    embedder: Embedder,
    k: int = 5,
    reranker=None,
    dense_store_factory: Callable[[], object] | None = None,
    chunk_transform: Callable[[list[Chunk], DocumentIR], list[Chunk]] | None = None,
    defs_split: bool = False,
) -> dict:
    """`dense_store_factory` (optional) is called once per doc to supply the dense
    retriever's VectorStore — pass it to benchmark a `PgVectorStore` (fresh/isolated
    per doc) in place of the default in-memory store. Default: in-memory.

    `chunk_transform` (optional) is applied to the chunk list right after
    `enrich_chunks`, receiving the SAME `DocumentIR` object `ir_for(g)` produced (so a
    transform that needs the IR — e.g. definition injection — sees the doc it was
    chunked from). HONESTY GUARD: `supports()` reads `chunk.text` only, never
    `index_text()`, so a transform can change WHICH chunks a retriever ranks into
    top-k, but it can never fabricate a recall hit — the gold answer must still
    literally appear in the (possibly transformed) chunk's display text.

    `defs_split` (optional, default off — zero extra cost when off) additionally
    tallies, per labeled field, whether ANY chunk that `supports()` the gold answer
    also USES a defined term (`enrich.definitions.extract_definitions` +
    `term_used`) — a DAPEI-style Type A/B split of the recall numbers into
    `recall_defs_dependent` (Type B: the answer sits alongside a defined term) vs
    `recall_defs_independent` (Type A: it doesn't), each with its own `n`. The
    dependent/independent classification is computed once per (doc, field) over the
    FULL indexed chunk pool (not just a method's top-k), so it reflects an inherent
    property of the field/document, not a particular retriever's behavior."""
    methods = ["bm25", "dense", "hybrid"] + (["hybrid_rerank"] if reranker else [])
    hits = {m: 0 for m in methods}
    n = 0
    defs_dependent_hits = {m: 0 for m in methods}
    defs_independent_hits = {m: 0 for m in methods}
    n_defs_dependent = 0
    n_defs_independent = 0
    for g in golden:
        ir = ir_for(g)
        chunks = enrich_chunks(chunk_ir(ir))
        if chunk_transform is not None:
            chunks = chunk_transform(chunks, ir)
        if not chunks:
            continue
        bm25 = BM25Index(); bm25.add(chunks)
        store = dense_store_factory() if dense_store_factory else None
        dense = DenseIndex(embedder, store=store); dense.add(chunks)
        hybrid = HybridIndex(bm25, dense)
        defs = extract_definitions(ir) if defs_split else []
        for field, gold in g.facts.items():
            if not gold or field not in FIELD_QUERIES:
                continue
            n += 1
            q = FIELD_QUERIES[field]
            results = {
                "bm25": [c for c, _ in bm25.search(q, k)],
                "dense": [c for c, _ in dense.search(q, k)],
                "hybrid": hybrid.search(q, k),
            }
            if reranker:
                results["hybrid_rerank"] = hybrid.search(q, k, reranker=reranker)

            dependent = False
            if defs_split:
                dependent = any(
                    supports(c, field, gold) and any(term_used(c.text, d.term) for d in defs)
                    for c in chunks
                )
                if dependent:
                    n_defs_dependent += 1
                else:
                    n_defs_independent += 1

            for method, chunks_out in results.items():
                hit = any(supports(c, field, gold) for c in chunks_out)
                if hit:
                    hits[method] += 1
                if defs_split:
                    if dependent:
                        if hit:
                            defs_dependent_hits[method] += 1
                    elif hit:
                        defs_independent_hits[method] += 1
    recall = {m: (hits[m] / n if n else 0.0) for m in hits}
    result = {"recall": recall, "n": n, "k": k}
    if defs_split:
        result["recall_defs_dependent"] = {
            m: (defs_dependent_hits[m] / n_defs_dependent if n_defs_dependent else 0.0)
            for m in hits
        }
        result["n_defs_dependent"] = n_defs_dependent
        result["recall_defs_independent"] = {
            m: (defs_independent_hits[m] / n_defs_independent if n_defs_independent else 0.0)
            for m in hits
        }
        result["n_defs_independent"] = n_defs_independent
    return result


def format_retrieval(res: dict, embedder_name: str) -> str:
    r = res["recall"]
    lines = [
        f"=== Context Recall@{res['k']} over {res['n']} labeled fields (embedder={embedder_name}) ===",
        f"  bm25 (lexical):  {r['bm25']:.3f}",
        f"  dense (vector):  {r['dense']:.3f}",
        f"  hybrid (RRF):    {r['hybrid']:.3f}",
        f"  hybrid lift vs vector-only: {r['hybrid'] - r['dense']:+.3f}",
    ]
    if "recall_defs_dependent" in res:  # DAPEI-style Type A/B split, only when defs_split=True
        rd, ri = res["recall_defs_dependent"], res["recall_defs_independent"]
        lines += [
            "  --- definition-dependent split (Type B vs Type A) ---",
            f"  defs-dependent   (n={res['n_defs_dependent']}):   "
            f"bm25 {rd['bm25']:.3f}  dense {rd['dense']:.3f}  hybrid {rd['hybrid']:.3f}",
            f"  defs-independent (n={res['n_defs_independent']}):   "
            f"bm25 {ri['bm25']:.3f}  dense {ri['dense']:.3f}  hybrid {ri['hybrid']:.3f}",
        ]
    return "\n".join(lines)


def main() -> None:
    import os
    from pathlib import Path

    from contract_rag.config import get_settings
    from contract_rag.eval.golden import load_golden_set
    from contract_rag.eval.ir_cache import ir_cache
    from contract_rag.index.embed import get_embedder
    from contract_rag.parse.docling_parser import parse_with_docling

    settings = get_settings()
    golden = load_golden_set(settings.golden_set_dir)
    kind = os.environ.get("EMBED_BACKEND", "hashing")
    embedder = get_embedder(settings, kind)
    parse_fn = ir_cache(Path(os.environ.get("IR_CACHE_DIR", ".ir_cache")), parse_with_docling)

    inject_defs = bool(os.environ.get("INJECT_DEFS"))

    def _inject_defs_transform(chunks, ir):
        return inject_definitions(chunks, extract_definitions(ir))

    chunk_transform = _inject_defs_transform if inject_defs else None

    res = evaluate_retrieval(golden, lambda g: parse_fn(settings.data_dir / g.source_pdf),
                             embedder, k=int(os.environ.get("RETRIEVAL_K", "5")),
                             chunk_transform=chunk_transform, defs_split=inject_defs)
    if inject_defs:
        print("(definition injection: ON)")
    print(format_retrieval(res, kind))


if __name__ == "__main__":
    main()
