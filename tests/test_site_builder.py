from __future__ import annotations

from pathlib import Path

import pytest

from contract_rag.benchmark.core import run_nda_benchmark
from contract_rag.site.builder import (
    benchmark_tokens,
    build_site,
    load_static_tokens,
    robots_txt,
    sitemap_xml,
)
from contract_rag.site.models import PageMeta

_PAGE = '''+++
title = "T"
description = "D"
lang = "en"
slug = "benchmark"
canonical = "https://x.github.io/contract-rag/benchmark.html"
target_queries = ["why is my RAG returning garbage"]
[[faq]]
q = "Q?"
a = "A."
[[howto]]
step = "python -m contract_rag.benchmark"
+++
# Heading

Quality lift was {{ quality_lift }} and field-F1 went {{ f1_dirty }} to {{ f1_clean }}.
'''


def test_benchmark_tokens():
    tok = benchmark_tokens(run_nda_benchmark(seed=0))
    assert set(tok) >= {"quality_dirty", "quality_clean", "quality_lift",
                        "f1_dirty", "f1_clean", "f1_lift", "n_docs"}
    assert tok["quality_lift"].startswith(("+", "-", "0"))


def test_build_site_emits_geo_html(tmp_path):
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "benchmark.en.md").write_text(_PAGE)
    out = tmp_path / "site_out"
    written = build_site(content, out, base_url="https://x.github.io/contract-rag",
                         benchmark=run_nda_benchmark(seed=0))
    page = out / "benchmark.html"
    assert page in written
    html = page.read_text()
    assert '<script type="application/ld+json">' in html
    assert '<link rel="canonical"' in html
    assert "TechArticle" in html and "FAQPage" in html
    assert "{{" not in html  # every token substituted, none left dangling
    assert (out / "llms.txt").exists()
    assert (out / "index.html").exists()


_STATIC_PAGE = '''+++
title = "K"
description = "F1 reached {{ kleister_f1_improved }}."
lang = "en"
slug = "kleister"
[[faq]]
q = "Score?"
a = "It was {{ kleister_f1_improved }} at source-acc {{ kleister_source_acc }}."
[[howto]]
step = "Reproduce the {{ kleister_f1_improved }} run"
+++
# Heading

Initial {{ kleister_f1_initial }} improved to {{ kleister_f1_improved }}.
'''


def test_load_static_tokens(tmp_path):
    toml = tmp_path / "vals.toml"
    toml.write_text('kleister_f1_improved = "0.697"\nkleister_n_docs = "40"\n')
    assert load_static_tokens(toml) == {"kleister_f1_improved": "0.697",
                                        "kleister_n_docs": "40"}


def test_build_site_injects_static_tokens_in_body_and_front_matter(tmp_path):
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "kleister.en.md").write_text(_STATIC_PAGE)
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://x.github.io/contract-rag",
               static_tokens={"kleister_f1_initial": "0.523",
                              "kleister_f1_improved": "0.697",
                              "kleister_source_acc": "1.0"})
    html = (out / "kleister.html").read_text()
    assert "0.523" in html and "0.697" in html
    assert "{{" not in html
    # FAQ answer + description flow into the JSON-LD / meta, substituted too
    assert "source-acc 1.0" in html
    assert "F1 reached 0.697." in html
    # HowTo steps flow into the JSON-LD too, substituted as well
    assert "Reproduce the 0.697 run" in html


def test_real_content_dir_builds_with_all_tokens_resolved(tmp_path):
    """The committed articles (benchmark live-injected + kleister static-injected)
    must build with every `{{ token }}` resolved."""
    pytest.importorskip("markdown")
    repo = Path(__file__).resolve().parent.parent
    content = repo / "content"
    tokens: dict[str, str] = {}
    for path in sorted(content.glob("*.toml")):
        tokens.update(load_static_tokens(path))
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://x.github.io/contract-rag",
               benchmark=run_nda_benchmark(seed=0), static_tokens=tokens)
    for page in sorted(out.glob("*.html")):         # every article, both langs
        assert "{{" not in page.read_text(), page.name
    for page in ("kleister-nda.html", "kleister-nda.zh.html"):
        html = (out / page).read_text()
        assert "0.697" in html and "0.523" in html   # headline Kleister numbers
        assert "0/40" in html and "12/40" in html    # structured-decoding result
        assert "2026-07-06" in html                  # point-in-time statement
    for page in ("benchmark.html", "benchmark.zh.html"):
        assert "{{" not in (out / page).read_text(), page


def test_robots_txt_allows_all_and_invites_ai_crawlers():
    text = robots_txt(base_url="https://contractrag.com")
    assert "User-agent: *" in text
    assert "Allow: /" in text
    for ua in ("GPTBot", "ClaudeBot", "Claude-Web", "PerplexityBot", "Google-Extended"):
        assert f"User-agent: {ua}" in text
    assert "Sitemap: https://contractrag.com/sitemap.xml" in text


def test_robots_txt_strips_trailing_slash_from_base_url():
    text = robots_txt(base_url="https://contractrag.com/")
    assert "Sitemap: https://contractrag.com/sitemap.xml" in text
    assert "//sitemap.xml" not in text


def test_sitemap_xml_lists_pages_with_fixed_lastmod():
    pages = [
        PageMeta(title="A", description="d", lang="en", slug="a",
                 canonical="https://contractrag.com/a.html"),
        PageMeta(title="B", description="d", lang="zh", slug="b",
                 canonical="https://contractrag.com/b.html"),
    ]
    xml = sitemap_xml(pages, now="2026-07-09")
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<urlset" in xml
    assert "<loc>https://contractrag.com/a.html</loc>" in xml
    assert "<loc>https://contractrag.com/b.html</loc>" in xml
    assert xml.count("<lastmod>2026-07-09</lastmod>") == 2


def test_build_site_emits_robots_and_sitemap_deterministically(tmp_path):
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "benchmark.en.md").write_text(_PAGE)
    out = tmp_path / "site_out"
    written = build_site(content, out, base_url="https://contractrag.com",
                         benchmark=run_nda_benchmark(seed=0), now="2026-07-09")
    robots_path = out / "robots.txt"
    sitemap_path = out / "sitemap.xml"
    assert robots_path in written
    assert sitemap_path in written
    robots = robots_path.read_text()
    assert "User-agent: GPTBot" in robots
    assert "Sitemap: https://contractrag.com/sitemap.xml" in robots
    sitemap = sitemap_path.read_text()
    assert "<loc>https://contractrag.com/benchmark.html</loc>" in sitemap
    assert "<lastmod>2026-07-09</lastmod>" in sitemap
    # deterministic: re-building with the same fixed `now` reproduces byte-identical output
    out2 = tmp_path / "site_out2"
    build_site(content, out2, base_url="https://contractrag.com",
              benchmark=run_nda_benchmark(seed=0), now="2026-07-09")
    assert (out2 / "sitemap.xml").read_text() == sitemap


def test_build_site_now_defaults_when_omitted(tmp_path):
    """Without an explicit `now`, build_site still emits a sitemap (CLI convenience) —
    just not byte-deterministic across real calendar days."""
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "benchmark.en.md").write_text(_PAGE)
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://x.github.io/contract-rag",
              benchmark=run_nda_benchmark(seed=0))
    sitemap = (out / "sitemap.xml").read_text()
    assert "<lastmod>" in sitemap and "</lastmod>" in sitemap
