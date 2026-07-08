from __future__ import annotations

EXTRACTION_PROMPT = (
    "Extract contract fields ONLY from the block-tagged text below.\n"
    "Rules:\n"
    "- Copy each value VERBATIM as a contiguous substring of ONE block — never "
    "paraphrase, reorder, reformat, or merge wording. Set source_block_id to that block.\n"
    "- counterparty: the contracting parties, copied verbatim from the block.\n"
    "- effective_date: ONLY a date the contract explicitly states as its effective date. "
    "If there is no explicit effective-date clause, leave it empty — do NOT substitute an "
    "execution, signature, agreement, or filing date.\n"
    "- governing_law: the governing jurisdiction, copied verbatim from the clause.\n"
    "- total_value: the total monetary value/fees of the contract, copied verbatim "
    "(e.g. '$1,000,000'); empty if none is stated.\n"
    "- termination_notice_days: the notice period required to terminate, copied verbatim "
    "(e.g. 'ninety (90) days'); empty if none is stated.\n"
    "- auto_renewal: 'yes' if the contract renews automatically, 'no' if it has a fixed "
    "term or renews only by agreement, empty if unclear.\n"
    "- If the text gives no basis for a field, leave value empty and confidence 0. "
    "Never invent values.\n\n"
)
