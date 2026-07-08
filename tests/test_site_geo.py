from __future__ import annotations

import json

from contract_rag.site.geo import json_ld, llms_txt
from contract_rag.site.models import PageMeta, parse_front_matter

_FM = '''+++
title = "Why your contract RAG returns garbage"
description = "A reproducible before/after benchmark."
lang = "en"
slug = "benchmark"
canonical = "https://example.github.io/contract-rag/benchmark.html"
target_queries = ["why is my RAG returning garbage", "how to clean PDFs for RAG"]
[[faq]]
q = "Why does dirty text break RAG?"
a = "Mojibake and repeated headers poison retrieval and extraction."
[[howto]]
step = "Run python -m contract_rag.benchmark"
+++
# Body starts here

Real content.'''


def test_parse_front_matter():
    meta, body = parse_front_matter(_FM)
    assert isinstance(meta, PageMeta)
    assert meta.title.startswith("Why your contract RAG")
    assert meta.lang == "en"
    assert meta.faq[0].q.startswith("Why does dirty")
    assert body.lstrip().startswith("# Body starts here")


def test_json_ld_has_required_types():
    meta, _ = parse_front_matter(_FM)
    graph = json_ld(meta)
    # must be valid json and contain the citable schema types
    types = {node["@type"] for node in graph["@graph"]}
    assert {"TechArticle", "FAQPage", "HowTo"} <= types
    assert json.loads(json.dumps(graph))  # round-trips


def test_llms_txt_lists_pages():
    meta, _ = parse_front_matter(_FM)
    txt = llms_txt([meta], base_url="https://example.github.io/contract-rag")
    assert "# contract-rag" in txt
    assert "benchmark.html" in txt
    assert meta.description in txt
