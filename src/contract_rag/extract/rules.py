"""Back-compat shim. The contract rule extractor now lives in
contract_rag.verticals.contract.rules."""
from __future__ import annotations

from contract_rag.verticals.contract.rules import *  # noqa: F401,F403
from contract_rag.verticals.contract.rules import (  # explicit: underscores + named API
    _DATE,
    RuleExtractor,
    auto_renewal_signal,
    days_in,
    find_auto_renewal,
    find_counterparty,
    find_effective_date,
    find_governing_law,
    find_termination_notice_days,
    find_total_value,
    jurisdiction_in,
    money_digits,
    party_entities,
)
