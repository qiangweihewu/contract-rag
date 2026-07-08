from __future__ import annotations

from contract_rag.verticals.nda.prompt import EXTRACTION_PROMPT


def test_prompt_mentions_every_field_and_verbatim_rule():
    for field in ("disclosing_party", "receiving_party", "effective_date", "term",
                  "confidentiality_period", "return_of_materials", "governing_law"):
        assert field in EXTRACTION_PROMPT
    assert "VERBATIM" in EXTRACTION_PROMPT
    assert "source_block_id" in EXTRACTION_PROMPT
