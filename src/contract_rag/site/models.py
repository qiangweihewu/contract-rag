"""Page front matter (TOML) parsed with stdlib tomllib — no extra dependency."""
from __future__ import annotations

import tomllib

from pydantic import BaseModel, Field

_FENCE = "+++"


class FAQItem(BaseModel):
    q: str
    a: str


class HowToStep(BaseModel):
    step: str


class PageMeta(BaseModel):
    title: str
    description: str
    lang: str
    slug: str
    canonical: str = ""
    target_queries: list[str] = Field(default_factory=list)
    faq: list[FAQItem] = Field(default_factory=list)
    howto: list[HowToStep] = Field(default_factory=list)


def parse_front_matter(text: str) -> tuple[PageMeta, str]:
    """Split a `+++`-fenced TOML front matter block from the markdown body."""
    if not text.lstrip().startswith(_FENCE):
        raise ValueError("document must start with a +++ TOML front-matter fence")
    rest = text.split(_FENCE, 1)[1]
    fm, _, body = rest.partition(_FENCE)
    return PageMeta(**tomllib.loads(fm)), body.lstrip("\n")


class LandingPillar(BaseModel):
    title: str
    body: str


class LandingEvidenceRow(BaseModel):
    label: str
    value: str
    link: str = ""


class LandingNegative(BaseModel):
    text: str


class LandingContent(BaseModel):
    """Structured TOML copy for the bilingual product landing page — no markdown
    body, so copy edits never touch Python (see `builder.load_landing_content`,
    which parses + `{{ token }}`-substitutes the raw TOML before validation)."""
    title: str
    description: str
    lang: str
    headline: str
    subhead: str
    cta_text: str
    cta_suffix: str
    cta_email: str
    proof_field_f1: str
    proof_quality: str
    proof_caption: str
    pillars_heading: str
    evidence_heading: str
    negatives_heading: str
    faq_heading: str
    pillars: list[LandingPillar] = Field(default_factory=list)
    evidence: list[LandingEvidenceRow] = Field(default_factory=list)
    negatives: list[LandingNegative] = Field(default_factory=list)
    faq: list[FAQItem] = Field(default_factory=list)
    github_url: str
    research_label: str
    footer_project_label: str
    footer_language_label: str
    lang_switch_label: str
    tagline: str
