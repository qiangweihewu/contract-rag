from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from contract_rag.benchmark.core import run_nda_benchmark
from contract_rag.site.builder import build_site, load_static_tokens, render_landing
from contract_rag.site.geo import faq_json_ld, landing_json_ld
from contract_rag.site.models import FAQItem, PageMeta

_TOKENS = {
    "f1_dirty": "0.14", "f1_clean": "0.98", "f1_lift": "+0.85",
    "quality_dirty": "0.69", "quality_clean": "0.94", "quality_lift": "+0.25",
    "cuad_rule_f1": "0.676", "cuad_rule_f1_ci": "[0.594, 0.746]",
    "cuad_constrained_f1": "0.661-0.672",
    "cuad_constrained_schema_fail": "0/40", "cuad_tools_schema_fail": "12/40",
    "kleister_f1_initial": "0.523", "kleister_f1_improved": "0.697",
    "kleister_source_acc": "1.0",
    "tobacco800_signature_f1": "0.864", "tobacco800_unsigned_flagged": "33/34",
    "fincritical_omission_rate": "7.7%", "fincritical_quality_at_omission": "0.998",
    "edith_misroute_docs": "98.8%",
}

_EN_TOML = '''
title = "T-en"
description = "D-en with {{ f1_dirty }} to {{ f1_clean }}"
lang = "en"
headline = "Your contract RAG returns garbage. Here is the fix, measured."
subhead = "Sourced, verified extraction and retrieval for dirty contract PDFs."
cta_text = "Send us your dirtiest contract"
cta_suffix = "PoC report in 48h"
cta_email = "hello@contractrag.com"
proof_field_f1 = "field-F1 {{ f1_dirty }} to {{ f1_clean }} ({{ f1_lift }})"
proof_quality = "quality {{ quality_dirty }} to {{ quality_clean }} ({{ quality_lift }})"
proof_caption = "real contracts, simulated dirt"
pillars_heading = "What it does"
evidence_heading = "Measured on public datasets"
negatives_heading = "We publish negative results"
faq_heading = "FAQ"
github_url = "https://github.com/qiangweihewu/contract-rag"
research_label = "Research"
footer_project_label = "Project"
footer_language_label = "Language"
lang_switch_label = "中文"
tagline = "open-source, reproducible evals."

[[pillars]]
title = "Parse router"
body = "Real scans + per-page mixed routing."
[[pillars]]
title = "Quality score"
body = "Explainable, HITL-ready; confidence can't see omissions -- we measured it."
[[pillars]]
title = "Sourced extraction"
body = "verify() + CLM export."

[[evidence]]
label = "CUAD 40-doc"
value = "rule {{ cuad_rule_f1 }} (95% CI {{ cuad_rule_f1_ci }})"
link = "benchmark.html"
[[evidence]]
label = "Kleister-NDA"
value = "{{ kleister_f1_initial }} to {{ kleister_f1_improved }}"
link = "kleister-nda.html"

[[negatives]]
text = "DAPEI definition injection: no lift under either embedder."
[[negatives]]
text = "FrankenOCR: parity F1, much slower, not adopted."

[[faq]]
q = "Question one?"
a = "Answer one, field-F1 {{ f1_clean }}."
[[faq]]
q = "Question two?"
a = "Answer two."
'''

_ZH_TOML = '''
title = "T-zh"
description = "D-zh 从 {{ f1_dirty }} 到 {{ f1_clean }}"
lang = "zh"
headline = "你的合同 RAG 检索全是乱码。这是可衡量的修复方案。"
subhead = "面向脏合同 PDF 的可溯源、可验证抽取与检索。"
cta_text = "把你最脏的合同发给我们"
cta_suffix = "48 小时内交付 PoC 报告"
cta_email = "hello@contractrag.com"
proof_field_f1 = "field-F1 从 {{ f1_dirty }} 到 {{ f1_clean }}（{{ f1_lift }}）"
proof_quality = "质量分从 {{ quality_dirty }} 到 {{ quality_clean }}（{{ quality_lift }}）"
proof_caption = "真实合同、模拟脏数据"
pillars_heading = "它做什么"
evidence_heading = "在公开数据集上的实测"
negatives_heading = "我们发布负面结果"
faq_heading = "常见问题"
github_url = "https://github.com/qiangweihewu/contract-rag"
research_label = "研究"
footer_project_label = "项目"
footer_language_label = "语言"
lang_switch_label = "EN"
tagline = "开源、可复现的评测。"

[[pillars]]
title = "解析路由"
body = "覆盖真实扫描件与按页混合路由。"
[[pillars]]
title = "质量评分"
body = "可解释、可用于人工复核；置信度看不到遗漏——我们已经测量过。"
[[pillars]]
title = "可溯源抽取"
body = "verify() 与 CLM 导出。"

[[evidence]]
label = "CUAD 40 篇合同"
value = "规则抽取 {{ cuad_rule_f1 }}"
link = "benchmark.html"
[[evidence]]
label = "Kleister-NDA"
value = "{{ kleister_f1_initial }} 到 {{ kleister_f1_improved }}"
link = "kleister-nda.html"

[[negatives]]
text = "DAPEI 定义注入：两种向量模型下均无提升。"
[[negatives]]
text = "FrankenOCR：F1 持平但慢很多，未采用。"

[[faq]]
q = "问题一？"
a = "回答一，field-F1 {{ f1_clean }}。"
[[faq]]
q = "问题二？"
a = "回答二。"
'''

_PAGES = [
    PageMeta(title="Why RAG returns garbage", description="d", lang="en", slug="benchmark",
             canonical="https://contractrag.com/benchmark.html"),
    PageMeta(title="Kleister-NDA measured", description="d", lang="en", slug="kleister-nda",
             canonical="https://contractrag.com/kleister-nda.html"),
]


def _write_landing_content(content_dir: Path) -> None:
    content_dir.mkdir(parents=True, exist_ok=True)
    (content_dir / "landing.en.toml").write_text(_EN_TOML)
    (content_dir / "landing.zh.toml").write_text(_ZH_TOML)


def test_render_landing_substitutes_tokens_for_both_langs(tmp_path):
    _write_landing_content(tmp_path)
    en_html = render_landing(_TOKENS, "en", _PAGES, content_dir=tmp_path,
                             base_url="https://contractrag.com")
    zh_html = render_landing(_TOKENS, "zh", _PAGES, content_dir=tmp_path,
                             base_url="https://contractrag.com")

    assert '<html lang="en">' in en_html
    assert '<html lang="zh">' in zh_html
    assert "0.14" in en_html and "0.98" in en_html and "+0.85" in en_html
    assert "0.14" in zh_html and "0.98" in zh_html
    assert "Your contract RAG returns garbage" in en_html
    assert "你的合同 RAG 检索全是乱码" in zh_html


def test_render_landing_no_token_leak(tmp_path):
    _write_landing_content(tmp_path)
    for lang in ("en", "zh"):
        html = render_landing(_TOKENS, lang, _PAGES, content_dir=tmp_path,
                              base_url="https://contractrag.com")
        # token-leak guard: no dangling `{{ name }}` placeholder text. (A bare
        # `}}` legitimately appears in the embedded JSON-LD, e.g. `"...}}]}`,
        # so only `{{` -- the opening marker -- is a leak signal.)
        assert "{{" not in html


def test_render_landing_hreflang_pair(tmp_path):
    _write_landing_content(tmp_path)
    en_html = render_landing(_TOKENS, "en", _PAGES, content_dir=tmp_path,
                             base_url="https://contractrag.com")
    zh_html = render_landing(_TOKENS, "zh", _PAGES, content_dir=tmp_path,
                             base_url="https://contractrag.com")
    for html in (en_html, zh_html):
        assert '<link rel="alternate" hreflang="en" href="https://contractrag.com/">' in html
        assert '<link rel="alternate" hreflang="zh" href="https://contractrag.com/zh/">' in html
    assert '<link rel="canonical" href="https://contractrag.com/">' in en_html
    assert '<link rel="canonical" href="https://contractrag.com/zh/">' in zh_html


def test_render_landing_research_links_and_footer(tmp_path):
    _write_landing_content(tmp_path)
    html = render_landing(_TOKENS, "en", _PAGES, content_dir=tmp_path,
                          base_url="https://contractrag.com")
    # Research links are root-absolute, not relative: the zh landing lives at
    # /zh/, so a relative "slug.html" would resolve to /zh/slug.html and 404.
    assert 'href="https://contractrag.com/benchmark.html"' in html
    assert 'href="https://contractrag.com/kleister-nda.html"' in html
    assert "https://github.com/qiangweihewu/contract-rag" in html
    assert 'href="https://contractrag.com/llms.txt"' in html
    assert 'mailto:hello@contractrag.com' in html


def test_render_landing_zh_research_links_are_root_absolute(tmp_path):
    """Regression: the zh landing page sits at /zh/, so its article links MUST be
    root-absolute — a relative href would 404 at /zh/benchmark.zh.html."""
    _write_landing_content(tmp_path)
    zh_pages = [
        PageMeta(title="为什么 RAG 返回垃圾", description="d", lang="zh", slug="benchmark.zh",
                 canonical="https://contractrag.com/benchmark.zh.html"),
    ]
    html = render_landing(_TOKENS, "zh", zh_pages, content_dir=tmp_path,
                          base_url="https://contractrag.com")
    # research nav link (root-absolute)
    assert 'href="https://contractrag.com/benchmark.zh.html"' in html
    assert 'href="benchmark.zh.html"' not in html  # the relative form that broke
    # evidence-table links must also be root-absolute — no relative internal .html
    assert 'href="benchmark.html"' not in html
    assert 'href="kleister-nda.html"' not in html
    assert 'href="https://contractrag.com/benchmark.html"' in html


def test_render_landing_evidence_and_negatives_and_faq_render(tmp_path):
    _write_landing_content(tmp_path)
    html = render_landing(_TOKENS, "en", _PAGES, content_dir=tmp_path,
                          base_url="https://contractrag.com")
    assert "CUAD 40-doc" in html
    assert "0.594, 0.746" in html
    assert "DAPEI definition injection" in html
    assert "FrankenOCR" in html
    assert "Question one?" in html
    assert "Answer one, field-F1 0.98." in html


def test_render_landing_json_ld_present_and_valid(tmp_path):
    _write_landing_content(tmp_path)
    html = render_landing(_TOKENS, "en", _PAGES, content_dir=tmp_path,
                          base_url="https://contractrag.com")
    start = html.index('<script type="application/ld+json">') + len('<script type="application/ld+json">')
    end = html.index("</script>", start)
    data = json.loads(html[start:end])
    types = {node["@type"] for node in data["@graph"]}
    assert {"Organization", "WebSite", "FAQPage"} <= types


def test_faq_json_ld_shape():
    entries = [FAQItem(q="Q1?", a="A1."), FAQItem(q="Q2?", a="A2.")]
    node = faq_json_ld(entries)
    assert node["@type"] == "FAQPage"
    assert len(node["mainEntity"]) == 2
    assert node["mainEntity"][0]["@type"] == "Question"
    assert node["mainEntity"][0]["acceptedAnswer"]["@type"] == "Answer"
    assert json.loads(json.dumps(node))


def test_landing_json_ld_shape():
    entries = [FAQItem(q="Q1?", a="A1.")]
    graph = landing_json_ld(name="contract-rag", url="https://contractrag.com/",
                             description="d", github_url="https://github.com/x/y", faq=entries)
    types = {node["@type"] for node in graph["@graph"]}
    assert {"Organization", "WebSite", "FAQPage"} <= types
    org = next(n for n in graph["@graph"] if n["@type"] == "Organization")
    assert org["sameAs"] == ["https://github.com/x/y"]
    assert json.loads(json.dumps(graph))


# --- build_site integration -------------------------------------------------

def test_build_site_writes_landing_pages_when_content_present(tmp_path):
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "benchmark.en.md").write_text('''+++
title = "T"
description = "D"
lang = "en"
slug = "benchmark"
+++
# H

Body {{ f1_dirty }}.
''')
    _write_landing_content(content)
    out = tmp_path / "site_out"
    written = build_site(content, out, base_url="https://contractrag.com",
                         benchmark=run_nda_benchmark(seed=0), static_tokens=dict(_TOKENS),
                         now="2026-07-09")
    index = out / "index.html"
    zh_index = out / "zh" / "index.html"
    assert index in written and zh_index in written
    assert index.exists() and zh_index.exists()
    en_html = index.read_text()
    zh_html = zh_index.read_text()
    assert "{{" not in en_html and "{{" not in zh_html
    assert '<html lang="en">' in en_html
    assert '<html lang="zh">' in zh_html
    # Research nav links to the existing article page (root-absolute, no auto-index)
    assert 'href="https://contractrag.com/benchmark.html"' in en_html
    sitemap = (out / "sitemap.xml").read_text()
    assert "<loc>https://contractrag.com/</loc>" in sitemap
    assert "<loc>https://contractrag.com/zh/</loc>" in sitemap


def test_build_site_falls_back_to_auto_index_without_landing_content(tmp_path):
    """Backward-compat: content dirs without landing.{en,zh}.toml keep emitting
    the old auto-generated article-list index.html (existing tests rely on this)."""
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "benchmark.en.md").write_text('''+++
title = "T"
description = "D"
lang = "en"
slug = "benchmark"
+++
# H

Body {{ f1_dirty }}.
''')
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://contractrag.com",
              benchmark=run_nda_benchmark(seed=0), now="2026-07-09")
    index_html = (out / "index.html").read_text()
    assert "<ul>" in index_html
    assert not (out / "zh").exists()


def test_existing_article_pages_byte_identical_with_landing_page_added(tmp_path):
    """Global constraint 3: adding the landing page must not change any existing
    article's rendered bytes. llms.txt is the one deliberate exception (item 4):
    it legitimately gains two landing-page links + its own content once the
    landing pages exist (`include_landing`), so it is checked separately below
    rather than asserted byte-identical."""
    pytest.importorskip("markdown")
    repo = Path(__file__).resolve().parent.parent
    content = repo / "content"

    out_with = tmp_path / "with_landing"
    build_site(content, out_with, base_url="https://contractrag.com",
              benchmark=run_nda_benchmark(seed=0), now="2026-07-09")

    stripped = tmp_path / "content_stripped"
    stripped.mkdir()
    for f in content.iterdir():
        if f.name in {"landing.en.toml", "landing.zh.toml"}:
            continue
        shutil.copy(f, stripped / f.name)
    out_without = tmp_path / "without_landing"
    build_site(stripped, out_without, base_url="https://contractrag.com",
              benchmark=run_nda_benchmark(seed=0), now="2026-07-09")

    for name in ("benchmark.html", "benchmark.zh.html", "kleister-nda.html", "kleister-nda.zh.html"):
        assert (out_with / name).read_text() == (out_without / name).read_text(), name
    # llms.txt: article-list portion is unchanged; the landing build additionally
    # prepends the two landing-page links (item 4) — both link the GitHub repo.
    llms_with = (out_with / "llms.txt").read_text()
    llms_without = (out_without / "llms.txt").read_text()
    assert "https://contractrag.com/): Product landing page" in llms_with
    assert "https://contractrag.com/): Product landing page" not in llms_without
    assert "https://github.com/qiangweihewu/contract-rag" in llms_with
    assert "https://github.com/qiangweihewu/contract-rag" in llms_without


def test_build_site_real_landing_content_resolves_all_tokens(tmp_path):
    """The committed landing.{en,zh}.toml must build with every token resolved,
    using the real content/ dir (exercises the actual copy + landing_results.toml)."""
    pytest.importorskip("markdown")
    repo = Path(__file__).resolve().parent.parent
    content = repo / "content"
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://contractrag.com",
              benchmark=run_nda_benchmark(seed=0), now="2026-07-09")
    en_html = (out / "index.html").read_text()
    zh_html = (out / "zh" / "index.html").read_text()
    assert "{{" not in en_html, "en leaked token"
    assert "{{" not in zh_html, "zh leaked token"
    assert "0.676" in en_html  # CUAD rule field-F1 evidence-wall token resolved
    assert "0.864" in en_html  # Tobacco800 signature F1


def test_zh_landing_page_has_no_hardcoded_english_chrome(tmp_path):
    """C1 regression: every piece of page chrome (section headings, footer column
    headers, CTA suffix, closing tagline, nav labels) must come from the per-
    language TOML — the zh page must not leak English template strings."""
    pytest.importorskip("markdown")
    repo = Path(__file__).resolve().parent.parent
    out = tmp_path / "site_out"
    build_site(repo / "content", out, base_url="https://contractrag.com",
              benchmark=run_nda_benchmark(seed=0), now="2026-07-09")
    zh_html = (out / "zh" / "index.html").read_text()
    for english_chrome in (
        "What it does",
        "Measured on public datasets",
        "We publish negative results",
        ">FAQ<",
        ">Project<",
        ">Language<",
        "PoC report in 48h",
        "open-source, reproducible, credential-free",
    ):
        assert english_chrome not in zh_html, english_chrome


def test_build_site_without_benchmark_leaks_no_tokens_on_landing(tmp_path):
    """Important regression: the landing pages must not depend on the live
    benchmark object — with benchmark=None and landing content present, both
    landing pages still render with zero `{{` leaks (hero chips use static
    point-in-time tokens, not live benchmark tokens)."""
    pytest.importorskip("markdown")
    repo = Path(__file__).resolve().parent.parent
    content = tmp_path / "content"
    content.mkdir()
    for name in ("landing.en.toml", "landing.zh.toml", "landing_results.toml",
                 "kleister_results.toml"):
        shutil.copy(repo / "content" / name, content / name)
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://contractrag.com",
              benchmark=None, now="2026-07-09")
    for page in (out / "index.html", out / "zh" / "index.html"):
        html = page.read_text()
        assert "{{" not in html, page.name


def test_static_tokens_auto_merge_excludes_landing_content_files(tmp_path):
    """landing.{en,zh}.toml are structured copy, not flat measured-value tables —
    they must not get flattened into the generic static-token merge."""
    repo = Path(__file__).resolve().parent.parent
    content = repo / "content"
    tokens: dict[str, str] = {}
    for path in sorted(content.glob("*.toml")):
        if path.name in {"landing.en.toml", "landing.zh.toml"}:
            continue
        tokens.update(load_static_tokens(path))
    # sanity: landing.en.toml is NOT flat TOML (has nested tables), so
    # load_static_tokens would choke or produce garbage if fed to it directly
    assert "pillars" not in tokens
    assert "evidence" not in tokens
