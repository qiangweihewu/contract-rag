from __future__ import annotations

EXTRACTION_PROMPT = (
    "Extract NDA (non-disclosure agreement) fields ONLY from the block-tagged text below.\n"
    "Rules:\n"
    "- Copy each value VERBATIM as a contiguous substring of ONE block — never "
    "paraphrase, reorder, or reformat. Set source_block_id to that block.\n"
    "- disclosing_party: the party disclosing confidential information (the "
    '"Disclosing Party" / "Discloser"), copied verbatim.\n'
    "- receiving_party: the party receiving confidential information (the "
    '"Receiving Party" / "Recipient"), copied verbatim.\n'
    "- effective_date: ONLY a date the NDA explicitly states as its effective date; "
    "empty if none is stated.\n"
    "- term: how long the agreement remains in force (e.g. 'two (2) years'); empty if unstated.\n"
    "- confidentiality_period: how long the confidentiality obligations survive "
    "(e.g. 'five (5) years'); empty if unstated.\n"
    "- return_of_materials: 'yes' if the agreement requires returning or destroying "
    "confidential materials, 'no' if it explicitly does not, empty if unclear.\n"
    "- governing_law: the governing jurisdiction, copied verbatim from the clause.\n"
    "- If the text gives no basis for a field, leave value empty and confidence 0. "
    "Never invent values.\n\n"
)
